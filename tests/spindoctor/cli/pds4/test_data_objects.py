"""Tests for the data objects a backplane FITS holds, as its data label describes them.

Every expected value here comes from the file: the headers' offsets and lengths from
the file's own bytes, read without astropy, and the arrays and their units from
astropy reading the file separately from the builder under test.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from astropy.io import fits

from spindoctor.cli.pds4.data_objects import (
    UndescribableFitsError,
    describe_backplane_fits,
    unusable_masked_value,
)
from spindoctor.config import DEFAULT_CONFIG, Config

MASKED_VALUE = -999.0
"""The masked value the tests hand the builder, which a 32-bit float holds exactly."""

FITS_BLOCK = 2880
"""The length of a FITS record; every header and every data unit fills whole ones."""

CARD = 80
"""The length of one header card."""

NUMPY_TYPES = {'IEEE754MSBSingle': '>f4', 'SignedMSB4': '>i4'}
"""The numpy type each PDS4 data type names: most significant byte first, as it says."""

TABLE_DEPRECATIONS = (
    'ignore:The chararray class is deprecated:DeprecationWarning',
    'ignore:Setting the dtype on a NumPy array has been deprecated:DeprecationWarning',
)
"""Numpy's deprecations of what astropy's binary tables still use.

Writing and reading a binary table draws both from inside astropy, which is not the
code under test.
"""


def _write_backplane_like(path: Path) -> None:
    """Write a FITS shaped as the backplane writer shapes one, 3 lines by 5 samples.

    An empty primary HDU, a body identity map of 32-bit integers with no unit, and two
    float planes, one declaring a unit and one not.  The frame is not square, so an
    exchange of lines and samples cannot pass unnoticed.

    Parameters:
        path: Where the FITS goes.
    """
    body_id_map = np.zeros((3, 5), dtype=np.int32)
    body_id_map[1:, 2:4] = 602
    latitude = fits.ImageHDU(
        data=np.arange(15, dtype=np.float32).reshape(3, 5), name='BODY_LATITUDE'
    )
    latitude.header['BUNIT'] = 'rad'
    unitless = fits.ImageHDU(
        data=np.full((3, 5), MASKED_VALUE, dtype=np.float32), name='PLANE_WITHOUT_UNIT'
    )
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(data=body_id_map, name='BODY_ID_MAP')]
    fits.HDUList([*hdus, latitude, unitless]).writeto(path)


def _headers_in(raw: bytes) -> list[tuple[int, int]]:
    """Return where each header of a FITS file begins and how many bytes it takes.

    Read from the bytes alone: a header begins a record with ``SIMPLE`` or
    ``XTENSION`` and ends with the record that holds its ``END`` card.

    Parameters:
        raw: The whole file.

    Returns:
        Each header's offset and length, in the order the file holds them.
    """
    headers = []
    for start in range(0, len(raw), FITS_BLOCK):
        if raw[start : start + 8] not in (b'SIMPLE  ', b'XTENSION'):
            continue
        end = start
        while not any(
            raw[card : card + 8] == b'END     ' for card in range(end, end + FITS_BLOCK, CARD)
        ):
            end += FITS_BLOCK
        headers.append((start, end + FITS_BLOCK - start))
    return headers


def test_every_header_and_array_is_where_the_file_holds_it(tmp_path: Path) -> None:
    """Each header's offset and length, and each array read at its offset, are the file's.

    The headers are found in the file's bytes, and each array is read out of those bytes
    at its stated offset, as its stated lines, samples and type, and held to what
    astropy reads for the same HDU.
    """
    path = tmp_path / 'backplanes.fits'
    _write_backplane_like(path)
    raw = path.read_bytes()
    described = describe_backplane_fits(path, masked_value=MASKED_VALUE)
    headers = [(hdu.header_offset, hdu.header_length) for hdu in described.hdus]
    assert headers == _headers_in(raw)
    read = [
        np.frombuffer(
            raw,
            dtype=NUMPY_TYPES[array.data_type],
            count=array.lines * array.samples,
            offset=array.offset,
        )
        .reshape(array.lines, array.samples)
        .tolist()
        for array in described.arrays
    ]
    with fits.open(path) as hdul:
        expected = [hdu.data.tolist() for hdu in hdul[1:]]
    assert read == expected


def test_each_array_is_named_and_qualified_as_its_hdu_is(tmp_path: Path) -> None:
    """An array's identifier, unit, missing constant and description follow its HDU.

    Its identifier is the HDU's name in lower case and its unit the HDU's ``BUNIT``,
    none when there is none; a float array declares the masked value it is handed and
    the integer body identity map none, saying instead what its values are.  Each
    float array says what it holds: its name, the method it is handed for it, its
    unit and the missing constant, and neither method nor unit where it has none.
    """
    path = tmp_path / 'backplanes.fits'
    _write_backplane_like(path)
    arrays = describe_backplane_fits(
        path, masked_value=MASKED_VALUE, methods={'body_latitude': 'latitude'}
    ).arrays
    with fits.open(path) as hdul:
        names = [hdu.name.lower() for hdu in hdul[1:]]
        units = [hdu.header.get('BUNIT') for hdu in hdul[1:]]
    assert [array.local_identifier for array in arrays] == names
    assert [array.unit for array in arrays] == units
    constants = [array.missing_constant for array in arrays]
    assert [None if text is None else float(text) for text in constants] == [
        None,
        MASKED_VALUE,
        MASKED_VALUE,
    ]
    body_id_map, latitude, unitless = (str(array.description) for array in arrays)
    assert '0 where no body claimed it' in body_id_map
    assert 'otherwise the NAIF ID of the body that did' in body_id_map
    assert 'The body_latitude backplane, from the oops backplane method latitude,' in latitude
    assert ', in rad.' in latitude
    assert f'holds the missing constant, {MASKED_VALUE!r}.' in latitude
    assert 'oops backplane method' not in unitless
    assert ', in ' not in unitless
    assert f'holds the missing constant, {MASKED_VALUE!r}.' in unitless


def test_a_fits_with_only_a_primary_hdu_has_one_header_and_no_array(tmp_path: Path) -> None:
    """A FITS holding no array is described by its one header, where the file holds it."""
    path = tmp_path / 'backplanes.fits'
    fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    described = describe_backplane_fits(path, masked_value=MASKED_VALUE)
    headers = [(hdu.header_offset, hdu.header_length) for hdu in described.hdus]
    assert headers == _headers_in(path.read_bytes())
    assert described.arrays == ()


def _with_image(
    data: np.ndarray, *, name: str | None = 'PLANE', **cards: float
) -> Callable[[Path], None]:
    """Return a writer of a FITS holding an empty primary HDU and one image HDU.

    Parameters:
        data: The image's array.
        name: The image HDU's name; None leaves it unnamed.
        **cards: Header cards to set on the image HDU before it is written.

    Returns:
        A function writing that FITS at the path it is given.
    """

    def write(path: Path) -> None:
        image = fits.ImageHDU(data=data) if name is None else fits.ImageHDU(data=data, name=name)
        for keyword, value in cards.items():
            image.header[keyword] = value
        fits.HDUList([fits.PrimaryHDU(), image]).writeto(path)

    return write


def _write_primary_with_data(path: Path) -> None:
    """Write a FITS whose primary HDU holds an image.

    Parameters:
        path: Where the FITS goes.
    """
    fits.HDUList([fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32))]).writeto(path)


def _write_table(path: Path) -> None:
    """Write a FITS whose one extension is a binary table.

    Parameters:
        path: Where the FITS goes.
    """
    column = fits.Column(name='VALUE', format='E', array=np.zeros(2, dtype=np.float32))
    table = fits.BinTableHDU.from_columns([column], name='TABLE')
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)


def _write_two_of_a_name(path: Path) -> None:
    """Write a FITS holding two image HDUs whose names differ only in case.

    Parameters:
        path: Where the FITS goes.
    """
    upper = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name='PLANE')
    mixed = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32))
    mixed.header['EXTNAME'] = 'Plane'
    fits.HDUList([fits.PrimaryHDU(), upper, mixed]).writeto(path)


@pytest.mark.parametrize(
    ('write', 'refusal'),
    [
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.int16)),
            r'HDU 1 \(PLANE\) of .* BITPIX = 16',
            id='BITPIX 16',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float64)),
            r'HDU 1 \(PLANE\) of .* BITPIX = -64',
            id='BITPIX -64',
        ),
        pytest.param(
            _with_image(np.zeros((2, 3, 4), dtype=np.float32)),
            r'HDU 1 \(PLANE\) of .* is not two-dimensional \(NAXIS = 3\)',
            id='three axes',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.int32), BSCALE=2.0),
            r'HDU 1 \(PLANE\) of .* carries BSCALE,',
            id='BSCALE',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.int32), BZERO=5.0),
            r'HDU 1 \(PLANE\) of .* carries BZERO,',
            id='BZERO',
        ),
        pytest.param(
            _write_primary_with_data,
            r'HDU 0 \(PRIMARY\) of .* holds data \(NAXIS = 2\)',
            id='a primary HDU holding data',
        ),
        pytest.param(
            _write_table,
            r'HDU 1 \(TABLE\) of .* is a BinTableHDU, not an image',
            id='a table',
            marks=[pytest.mark.filterwarnings(ignored) for ignored in TABLE_DEPRECATIONS],
        ),
        pytest.param(
            lambda path: fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.CompImageHDU(data=np.zeros((16, 16), dtype=np.float32), name='PLANE'),
                ]
            ).writeto(path),
            r'HDU 1 \(PLANE\) of .* is a BinTableHDU, not an image',
            id='a tile-compressed image',
            marks=[pytest.mark.filterwarnings(ignored) for ignored in TABLE_DEPRECATIONS],
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float32), name=None),
            r'HDU 1 \(unnamed\) of .* is not a local identifier',
            id='an unnamed image',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float32), name='1PLANE'),
            r'HDU 1 \(1PLANE\) of .* is not a local identifier',
            id='a name beginning with a digit',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float32), name='PLANE 1'),
            r'HDU 1 \(PLANE 1\) of .* is not a local identifier',
            id='a name holding a space',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float32), name='PLANE:1'),
            r'HDU 1 \(PLANE:1\) of .* is not a local identifier',
            id='a name holding a colon',
        ),
        pytest.param(
            _write_two_of_a_name,
            r'more than one image HDU named plane in lower case',
            id='two names alike in lower case',
        ),
        pytest.param(
            _with_image(np.zeros((2, 2), dtype=np.float32), name='NAVIGATION-DETAILS'),
            r'HDU 1 \(NAVIGATION-DETAILS\) of .* which the data label gives to another',
            id='the name the label gives its supplemental file',
        ),
    ],
)
def test_a_fits_the_backplane_writer_does_not_write_is_refused(
    tmp_path: Path, write: Callable[[Path], None], refusal: str
) -> None:
    """What the backplane writer does not write is refused, naming the file and the HDU.

    Parameters:
        tmp_path: Where the FITS goes.
        write: Writes the FITS at the path it is given.
        refusal: What the refusal has to say.
    """
    path = tmp_path / 'backplanes.fits'
    write(path)
    with pytest.raises(UndescribableFitsError, match=refusal) as excinfo:
        describe_backplane_fits(path, masked_value=MASKED_VALUE)
    assert str(path) in str(excinfo.value)


def _configured(masked_value: object) -> Config:
    """Return a configuration declaring one masked value and nothing else.

    Parameters:
        masked_value: The ``backplanes.masked_value`` it declares.

    Returns:
        The configuration.
    """
    return cast(Config, SimpleNamespace(backplanes=SimpleNamespace(masked_value=masked_value)))


@pytest.mark.parametrize(
    ('masked_value', 'problem'),
    [
        pytest.param(-999.1, 'is not one a 32-bit float holds exactly', id='inexact'),
        pytest.param(float('nan'), 'is not a finite number', id='not finite'),
        pytest.param('-999', 'is not a number', id='a string'),
    ],
)
def test_a_masked_value_no_float_plane_can_hold_is_unusable(
    masked_value: object, problem: str
) -> None:
    """A masked value that is not a number, not finite or not a float32's is named.

    Parameters:
        masked_value: The configured masked value.
        problem: What the check has to say of it.
    """
    reason = str(unusable_masked_value(_configured(masked_value)))
    assert repr(masked_value) in reason
    assert problem in reason


@pytest.mark.parametrize(
    'masked_value',
    [pytest.param(None, id='the shipped value'), pytest.param(-999, id='an integer')],
)
def test_a_masked_value_a_float_plane_holds_is_usable(masked_value: object) -> None:
    """The shipped masked value, and an integer a 32-bit float holds, are usable.

    Parameters:
        masked_value: The configured masked value; None for the shipped
            configuration's.
    """
    config = DEFAULT_CONFIG if masked_value is None else _configured(masked_value)
    assert unusable_masked_value(config) is None
