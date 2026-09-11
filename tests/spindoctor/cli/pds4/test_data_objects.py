"""Tests for the data objects a backplane FITS holds, as its data label describes them.

Every expected value here comes from the file: the headers' offsets and lengths from
the file's own bytes, read without astropy, and the arrays and their units from
astropy reading the file separately from the builder under test.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from astropy.io import fits

from spindoctor.cli.pds4.data_objects import describe_backplane_fits, unusable_masked_value
from spindoctor.config import DEFAULT_CONFIG, Config

MASKED_VALUE = -999.0
"""The masked value the tests hand the builder, which a 32-bit float holds exactly."""

FITS_BLOCK = 2880
"""The length of a FITS record; every header and every data unit fills whole ones."""

CARD = 80
"""The length of one header card."""

NUMPY_TYPES = {'IEEE754MSBSingle': '>f4', 'SignedMSB4': '>i4'}
"""The numpy type each PDS4 data type names: most significant byte first, as it says."""


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
