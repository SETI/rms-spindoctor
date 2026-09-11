"""Tests for the data objects a backplane FITS holds, as its data label describes them.

Every expected value here comes from the file: the headers' offsets and lengths from
the file's own bytes, read without astropy, and the arrays and their units from
astropy reading the file separately from the builder under test.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from spindoctor.cli.pds4.data_objects import (
    UndescribableFitsError,
    describe_backplane_fits,
)

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
    the integer body identity map none, saying instead what its values are.
    """
    path = tmp_path / 'backplanes.fits'
    _write_backplane_like(path)
    arrays = describe_backplane_fits(path, masked_value=MASKED_VALUE).arrays
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
    descriptions = [array.description for array in arrays]
    assert descriptions[1:] == [None, None]
    body_id_map = str(descriptions[0])
    assert '0 where no body claimed it' in body_id_map
    assert 'otherwise the NAIF ID of the body that did' in body_id_map


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
    """Write a FITS holding two image HDUs of the same name.

    Parameters:
        path: Where the FITS goes.
    """
    planes = [fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name='PLANE')] * 2
    fits.HDUList([fits.PrimaryHDU(), *planes]).writeto(path)


def _write_truncated(path: Path) -> None:
    """Write a backplane-shaped FITS cut short partway through its last array.

    Parameters:
        path: Where the FITS goes.
    """
    _write_backplane_like(path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - FITS_BLOCK // 2])


def _write_not_fits(path: Path) -> None:
    """Write a file that is not FITS at all.

    Parameters:
        path: Where the file goes.
    """
    path.write_bytes(b'not a FITS file')


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
            _with_image(np.zeros((2, 2), dtype=np.float32), name=None),
            r'HDU 1 \(unnamed\) of .* is not a local identifier',
            id='an unnamed image',
        ),
        pytest.param(
            _write_two_of_a_name,
            r'more than one image HDU named plane in lower case',
            id='two images of one name',
        ),
        pytest.param(
            _write_truncated,
            r'astropy reads cleanly: File may have been truncated',
            id='a truncated file',
        ),
        pytest.param(
            _write_not_fits,
            r'astropy reads cleanly: No SIMPLE card found',
            id='not FITS',
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


@pytest.mark.parametrize('masked_value', [0.1, float('inf')], ids=['inexact', 'not finite'])
def test_a_masked_value_no_float_plane_can_hold_is_refused(
    tmp_path: Path, masked_value: float
) -> None:
    """A masked value a 32-bit float does not hold exactly, or at all, is refused.

    Parameters:
        tmp_path: Where the FITS goes.
        masked_value: The masked value handed to the builder.
    """
    path = tmp_path / 'backplanes.fits'
    _write_backplane_like(path)
    with pytest.raises(ValueError, match='is not a finite number a 32-bit float holds exactly'):
        describe_backplane_fits(path, masked_value=masked_value)


def test_a_fits_that_is_not_there_is_not_found(tmp_path: Path) -> None:
    """A path with no file behind it raises FileNotFoundError, naming the path."""
    path = tmp_path / 'backplanes.fits'
    with pytest.raises(FileNotFoundError, match=re.escape(str(path))):
        describe_backplane_fits(path, masked_value=MASKED_VALUE)
