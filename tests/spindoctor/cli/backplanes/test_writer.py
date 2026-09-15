"""Spec-first tests for the backplane FITS + sidecar writer.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "FITS writer" and
docs/user_guide/user_guide_backplanes.rst "Outputs"): the FITS file has an empty
primary HDU, an int32 BODY_ID_MAP as the first image HDU (emitted only when some
pixel has a non-zero ID), and one float32 ImageHDU per non-all-zero backplane with
BUNIT taken from the per-backplane config units.  A sidecar
``<stub>_backplane_metadata.json`` carries per-body inventory information and
per-backplane statistics, taken from the merged planes the FITS holds.
"""

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from astropy.io import fits
from filecache import FCPath

from spindoctor.cli.backplanes.backplanes_bodies import BODY_LONGITUDE
from spindoctor.cli.backplanes.backplanes_rings import RING_LONGITUDE, RING_LONGITUDINAL_RESOLUTION
from spindoctor.cli.backplanes.merge import body_naif_id, merge_sources_into_master
from spindoctor.cli.backplanes.writer import write_fits
from spindoctor.config import DEFAULT_CONFIG

from .conftest import (
    MASKED_VALUE,
    FakeBackplanesConfig,
    HermeticObs,
    inventory_entry,
    make_snapshot,
)

SHAPE_VU = (6, 11)


def _default_config() -> FakeBackplanesConfig:
    """Build a config declaring one body plane, one ring plane, and ring distance."""
    return FakeBackplanesConfig(
        bodies=[{'name': 'body_latitude', 'method': 'latitude', 'units': 'rad'}],
        rings=[
            {'name': 'ring_radius', 'method': 'ring_radius', 'units': 'km'},
            {'name': 'distance', 'method': 'distance', 'units': 'km'},
        ],
    )


def _master_with(value: float, *, name: str = 'body_latitude') -> dict[str, np.ndarray]:
    """Build a merge-style master dict: NaN background with one valid pixel.

    Parameters:
        value: Value stored at pixel (0, 0).
        name: Backplane type name.
    """
    arr = np.full(SHAPE_VU, np.nan, dtype=np.float32)
    arr[0, 0] = value
    return {name: arr}


def _id_map(naif_id: int = 699) -> np.ndarray:
    """Build a BODY_ID_MAP with one non-zero pixel.

    Parameters:
        naif_id: NAIF ID stored at pixel (0, 0).
    """
    ids = np.zeros(SHAPE_VU, dtype=np.int32)
    ids[0, 0] = naif_id
    return ids


def _write(
    tmp_path: Path,
    *,
    master: dict[str, np.ndarray],
    id_map: np.ndarray,
    snapshot: HermeticObs | None = None,
    config: FakeBackplanesConfig | None = None,
    bodies_result: dict[str, Any] | None = None,
    rings_result: dict[str, Any] | None = None,
    stub: str = 'IMG1',
) -> tuple[Path, Path]:
    """Invoke write_fits into tmp_path and return the FITS and sidecar paths.

    Parameters:
        tmp_path: Directory receiving the outputs.
        master: Master per-type backplane arrays.
        id_map: Per-pixel NAIF ID map.
        snapshot: Observation; defaults to a simulated one with an empty inventory.
        config: Fake config; defaults to :func:`_default_config`.
        bodies_result: Per-body result dict for the sidecar.
        rings_result: Ring result dict for the sidecar.
        stub: Results path stub used in the output file names.
    """
    snap = (
        snapshot
        if snapshot is not None
        else make_snapshot(shape_vu=SHAPE_VU, simulated=True, sim_inventory={})
    )
    cfg = config if config is not None else _default_config()
    fits_path = tmp_path / f'{stub}_backplanes.fits'
    write_fits(
        fits_file_path=FCPath(fits_path),
        snapshot=snap,
        master_by_type=master,
        body_id_map=id_map,
        config=cfg.as_config(),
        bodies_result=bodies_result if bodies_result is not None else {},
        rings_result=rings_result,
    )
    return fits_path, tmp_path / f'{stub}_backplane_metadata.json'


def test_write_fits_primary_hdu_is_empty_placeholder(tmp_path: Path) -> None:
    """The first HDU is a conventional empty primary HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert isinstance(hdul[0], fits.PrimaryHDU)
        assert hdul[0].data is None


def test_write_fits_body_id_map_is_first_image_hdu(tmp_path: Path) -> None:
    """BODY_ID_MAP is the first image HDU, before any backplane HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert hdul[1].name == 'BODY_ID_MAP'


def test_write_fits_body_id_map_dtype_and_values(tmp_path: Path) -> None:
    """BODY_ID_MAP is int32 and round-trips the per-pixel NAIF IDs.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map(699))
    with fits.open(fits_path) as hdul:
        assert hdul['BODY_ID_MAP'].data.dtype.name == 'int32'
        assert int(hdul['BODY_ID_MAP'].data[0, 0]) == 699


def test_write_fits_body_id_map_omitted_when_all_zero(tmp_path: Path) -> None:
    """A BODY_ID_MAP with no non-zero pixel is not written.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    zeros = np.zeros(SHAPE_VU, dtype=np.int32)
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=zeros)
    with fits.open(fits_path) as hdul:
        assert 'BODY_ID_MAP' not in [hdu.name for hdu in hdul]


def test_write_fits_backplane_hdu_name_is_uppercased(tmp_path: Path) -> None:
    """The configured backplane name becomes an upper-case FITS EXTNAME.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert 'BODY_LATITUDE' in [hdu.name for hdu in hdul]


def test_write_fits_backplane_dtype_float32(tmp_path: Path) -> None:
    """Backplane HDUs are written as float32.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert hdul['BODY_LATITUDE'].data.dtype.name == 'float32'


def test_write_fits_bunit_from_config_units(tmp_path: Path) -> None:
    """The BUNIT header comes from the per-backplane config units field.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    master = _master_with(1.0)
    master.update(_master_with(2.0, name='ring_radius'))
    fits_path, _ = _write(tmp_path, master=master, id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert hdul['BODY_LATITUDE'].header['BUNIT'] == 'rad'
        assert hdul['RING_RADIUS'].header['BUNIT'] == 'km'


def test_write_fits_no_bunit_for_unconfigured_name(tmp_path: Path) -> None:
    """A master plane not named in the config gets no BUNIT header.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0, name='mystery'), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert 'BUNIT' not in hdul['MYSTERY'].header


def test_write_fits_fully_masked_backplane_omitted(tmp_path: Path) -> None:
    """A backplane that is entirely the masked value contributes no HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    master = {'body_latitude': np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)}
    fits_path, _ = _write(tmp_path, master=master, id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert 'BODY_LATITUDE' not in [hdu.name for hdu in hdul]


def test_write_fits_all_invalid_backplane_omitted(tmp_path: Path) -> None:
    """A backplane with no valid pixel anywhere contributes no HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    master = {'body_latitude': np.full(SHAPE_VU, np.nan, dtype=np.float32)}
    fits_path, _ = _write(tmp_path, master=master, id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert 'BODY_LATITUDE' not in [hdu.name for hdu in hdul]


def test_write_fits_hdus_follow_master_insertion_order(tmp_path: Path) -> None:
    """Backplane HDUs appear in master-dict insertion order after BODY_ID_MAP.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    master: dict[str, np.ndarray] = {}
    master.update(_master_with(1.0, name='alpha'))
    master.update(_master_with(2.0, name='beta'))
    fits_path, _ = _write(tmp_path, master=master, id_map=_id_map())
    with fits.open(fits_path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert names[2:] == ['ALPHA', 'BETA']


def test_write_fits_preserves_non_square_shape(tmp_path: Path) -> None:
    """Backplane HDU data keeps the sensor's (rows, columns) shape.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    fits_path, _ = _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        assert hdul['BODY_LATITUDE'].data.shape == SHAPE_VU


def test_write_fits_empty_pipeline_writes_primary_only(tmp_path: Path) -> None:
    """No bodies and no rings yields a FITS with just the primary HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    zeros = np.zeros(SHAPE_VU, dtype=np.int32)
    fits_path, _ = _write(tmp_path, master={}, id_map=zeros)
    with fits.open(fits_path) as hdul:
        assert len(hdul) == 1


def test_write_fits_empty_pipeline_sidecar(tmp_path: Path) -> None:
    """No bodies and no rings yields an empty-but-well-formed sidecar.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    zeros = np.zeros(SHAPE_VU, dtype=np.int32)
    _, sidecar = _write(tmp_path, master={}, id_map=zeros)
    metadata = json.loads(sidecar.read_text())
    assert metadata == {'bodies': {}, 'rings': {}}


def test_write_fits_overwrites_existing_file(tmp_path: Path) -> None:
    """Writing twice to the same path succeeds and the second write wins.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    _write(tmp_path, master=_master_with(1.0), id_map=_id_map())
    fits_path, _ = _write(tmp_path, master=_master_with(2.0, name='ring_radius'), id_map=_id_map())
    with fits.open(fits_path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert 'RING_RADIUS' in names
        assert 'BODY_LATITUDE' not in names


def test_write_fits_sidecar_path_naming(tmp_path: Path) -> None:
    """The sidecar is named <stub>_backplane_metadata.json next to the FITS file.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    _, sidecar = _write(tmp_path, master=_master_with(1.0), id_map=_id_map(), stub='N123')
    assert sidecar.name == 'N123_backplane_metadata.json'
    assert sidecar.exists()


def _moon_a_metadata(tmp_path: Path) -> dict[str, Any]:
    """Write the products of a simulated frame showing MOON_A, and read back its sidecar.

    MOON_A has an inventory entry and claims a block of the body identity map, where the
    body latitude plane holds -10 degrees and, at one pixel, 25 degrees, in radians.

    Parameters:
        tmp_path: Directory receiving the outputs.

    Returns:
        The backplane metadata document the writer wrote.
    """
    inventory = {
        'MOON_A': inventory_entry(
            u_min=2,
            u_max=5,
            v_min=1,
            v_max=3,
            body_range=500000.0,
            center_uv=(3.5, 2.0),
            u_pixel_size=4.0,
            v_pixel_size=3.0,
        )
    }
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True, sim_inventory=inventory)
    latitude = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    latitude[1:3, 2:5] = math.radians(-10.0)
    latitude[1, 2] = math.radians(25.0)
    id_map = np.zeros(SHAPE_VU, dtype=np.int32)
    id_map[1:3, 2:5] = body_naif_id(snap, 'MOON_A')
    _, sidecar = _write(
        tmp_path,
        master={'body_latitude': latitude},
        id_map=id_map,
        snapshot=snap,
        bodies_result={'MOON_A': {'arrays': {}, 'masks': {}, 'distance': 500000.0}},
    )
    return cast(dict[str, Any], json.loads(sidecar.read_text()))


def test_write_fits_sidecar_body_statistics(tmp_path: Path) -> None:
    """A body's statistics, in degrees for a plane in radians, are under its backplanes.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    assert metadata['bodies']['MOON_A']['backplanes'] == {
        'body_latitude': {'min': pytest.approx(-10.0), 'max': pytest.approx(25.0), 'units': 'deg'}
    }


def test_write_fits_sidecar_center_uv_is_swapped_to_vu(tmp_path: Path) -> None:
    """The sidecar center_uv key actually stores (v, u) swapped from the inventory.

    The inventory carries center_uv as (u, v); the writer swaps to (v, u) while
    keeping the key name.  This characterizes the on-disk convention consumed by
    the PDS4 stage; the docs do not specify the ordering.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    assert metadata['bodies']['MOON_A']['center_uv'] == [2.0, 3.5]


def test_write_fits_sidecar_center_range_and_size(tmp_path: Path) -> None:
    """The sidecar carries center_range and size_uv from the inventory.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    assert metadata['bodies']['MOON_A']['center_range'] == 500000.0
    assert metadata['bodies']['MOON_A']['size_uv'] == [4.0, 3.0]


RINGS_RESULT: dict[str, Any] = {
    'target_key': 'PLANET_RING_SYSTEM',
    'incidence_angle': {'value': 40.0, 'units': 'deg'},
    'pixel_incidence': np.full(SHAPE_VU, MASKED_VALUE),
}
"""A ring result less its planes: its target, and the incidence angle at its center.

No pixel has an incidence angle, so the rings block records the center's alone.
"""


def test_write_fits_sidecar_ring_statistics(tmp_path: Path) -> None:
    """The ring target, its incidence angle and the ring statistics are written under rings.

    The statistics are the ring planes' the FITS holds, over the pixels where each has a
    value.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    radius = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    radius[2, 3] = 70000.0
    radius[4, 5] = 140000.0
    _, sidecar = _write(
        tmp_path, master={'ring_radius': radius}, id_map=_id_map(), rings_result=RINGS_RESULT
    )
    metadata = json.loads(sidecar.read_text())
    assert metadata['rings'] == {
        'target': 'PLANET_RING_SYSTEM',
        'incidence_angle': {'value': 40.0, 'units': 'deg'},
        'backplanes': {'ring_radius': {'min': 70000.0, 'max': 140000.0, 'units': 'km'}},
    }


COVERED_LATITUDE = 1.5
"""PLANET's latitude, in radians, where MOON_A covers it: no pixel of PLANET shows it."""

COVERED_RADIUS = 50000.0
"""The rings' radius, in km, where MOON_A covers them: no pixel of the rings shows it."""

COVERED_LONGITUDE = 200.0
"""The rings' longitude, in degrees, where MOON_A covers them; 10 everywhere else."""

COVERED_INCIDENCE = 60.0
"""The incidence angle, in degrees, where MOON_A covers the rings: no ring pixel shows it."""


def _covered_incidence() -> np.ndarray:
    """Return the incidence angle at each pixel of the covered frame, in radians.

    Rows 2 to 4 hold 41 degrees and row 5 holds 49, except that the last pixel of each
    holds none, though the ring planes have values there.  Rows 0 and 1, where MOON_A
    covers the rings, hold :data:`COVERED_INCIDENCE`.  Over the ring pixels the FITS holds
    the least is 41, the greatest 49 and the mean 43, where the median is 41, and none of
    them is the 40 degrees at the ring center.

    Returns:
        The full-frame array, the masked value where a pixel has no angle.
    """
    incidence = np.full(SHAPE_VU, math.radians(COVERED_INCIDENCE))
    incidence[2:5, :] = math.radians(41.0)
    incidence[5, :] = math.radians(49.0)
    incidence[2:6, -1] = MASKED_VALUE
    return incidence


LONGITUDE_PLANES = [
    {'name': RING_LONGITUDE, 'method': 'ring_longitude', 'units': 'rad'},
    {
        'name': RING_LONGITUDINAL_RESOLUTION,
        'method': 'ring_angular_resolution',
        'units': 'rad/pixel',
    },
]
"""The ring longitude and the longitudinal size of a pixel, as the configuration declares them."""


def _rows(first: int, stop: int) -> np.ndarray:
    """Return a mask true on the frame's rows from ``first`` up to ``stop``.

    Parameters:
        first: The first row.
        stop: The row after the last.

    Returns:
        The full-frame mask.
    """
    mask = np.zeros(SHAPE_VU, dtype=bool)
    mask[first:stop, :] = True
    return mask


def _plane(mask: np.ndarray, value: float, covered_value: float) -> np.ndarray:
    """Return a source's plane: one value on its mask, another where MOON_A covers it.

    Parameters:
        mask: Where the source has a value.
        value: Its value there.
        covered_value: Its value on rows 0 and 1, which MOON_A covers.

    Returns:
        The full-frame plane, the masked value off the mask.
    """
    plane = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    plane[mask] = value
    plane[mask & _rows(0, 2)] = covered_value
    return plane


def _covered_metadata(tmp_path: Path) -> dict[str, Any]:
    """Merge and write a frame in which a nearer body covers the rings and two bodies.

    MOON_A, the nearest source, fills rows 0 and 1.  Behind it are PLANET, which fills
    rows 0 to 3, MOON_B, which is behind MOON_A and nowhere else, and the rings, which
    fill every row, nearer than PLANET.  Where MOON_A covers them, PLANET's latitude and
    the rings' radius hold values seen nowhere else, so a statistic taken over pixels a
    nearer body covers shows them.

    Parameters:
        tmp_path: Directory receiving the outputs.

    Returns:
        The backplane metadata document the writer wrote.
    """
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True, sim_inventory={})
    near, planet, rings = _rows(0, 2), _rows(0, 4), _rows(0, SHAPE_VU[0])

    def body(mask: np.ndarray, plane: np.ndarray, distance: float) -> dict[str, Any]:
        """Return a body's result, one latitude plane on its mask at one distance.

        Parameters:
            mask: Where the body is seen.
            plane: Its latitude plane.
            distance: How far it is.

        Returns:
            The body's entry in the body stage's result.
        """
        return {
            'arrays': {'body_latitude': plane},
            'masks': {'body_latitude': mask},
            'distance': distance,
        }

    bodies_result = {
        'MOON_A': body(near, _plane(near, 0.2, 0.2), 1.0e5),
        'PLANET': body(planet, _plane(planet, 0.5, COVERED_LATITUDE), 1.0e6),
        'MOON_B': body(near, _plane(near, 0.7, 0.7), 2.0e6),
    }
    ring_planes = {
        'ring_radius': _plane(rings, 100000.0, COVERED_RADIUS),
        RING_LONGITUDE: _plane(rings, math.radians(10.0), math.radians(COVERED_LONGITUDE)),
        RING_LONGITUDINAL_RESOLUTION: _plane(rings, 1.0e-4, 1.0e-4),
    }
    rings_result = {
        **RINGS_RESULT,
        'pixel_incidence': _covered_incidence(),
        'arrays': ring_planes,
        'masks': dict.fromkeys(ring_planes, rings),
        'distance': np.full(SHAPE_VU, 5.0e5, dtype=np.float32),
    }
    master, id_map = merge_sources_into_master(
        snap, bodies_result=bodies_result, rings_result=rings_result
    )
    config = _default_config()
    config.backplanes.rings.extend(LONGITUDE_PLANES)
    _, sidecar = _write(
        tmp_path,
        master=master,
        id_map=id_map,
        snapshot=snap,
        config=config,
        bodies_result=bodies_result,
        rings_result=rings_result,
    )
    return cast(dict[str, Any], json.loads(sidecar.read_text()))


def test_ring_statistics_leave_out_the_rings_a_nearer_body_covers(tmp_path: Path) -> None:
    """A ring pixel a nearer body covers has no ring value in the FITS, nor in a statistic.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    radius = _covered_metadata(tmp_path)['rings']['backplanes']['ring_radius']
    assert radius == {'min': 100000.0, 'max': 100000.0, 'units': 'km'}


def test_a_body_s_statistics_leave_out_what_a_nearer_body_covers(tmp_path: Path) -> None:
    """A body's statistics are over the pixels the body identity map gives it.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    latitude = _covered_metadata(tmp_path)['bodies']['PLANET']['backplanes']['body_latitude']
    expected = math.degrees(0.5)
    assert latitude == {
        'min': pytest.approx(expected),
        'max': pytest.approx(expected),
        'units': 'deg',
    }


def test_the_wrapped_ring_longitude_leaves_out_the_rings_a_nearer_body_covers(
    tmp_path: Path,
) -> None:
    """The arc of longitude the rings cover is the merged plane's, as its range is.

    Counting the covered pixels, at 200 degrees, beside the rest, at 10, would give an arc
    from 200 across zero to 10.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    longitude = _covered_metadata(tmp_path)['rings']['backplanes'][RING_LONGITUDE]
    wrapped = (longitude['wrapped_min'], longitude['wrapped_max'])
    assert wrapped == (pytest.approx(10.0), pytest.approx(10.0))


def test_the_incidence_range_is_over_the_ring_pixels_the_fits_holds(tmp_path: Path) -> None:
    """The least, greatest and mean incidence over the ring pixels, beside the center's.

    The rings MOON_A covers do not count, nor does a ring pixel with no incidence angle.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    incidence = _covered_metadata(tmp_path)['rings']['incidence_angle']
    assert incidence['value'] == 40.0
    assert incidence['min'] == pytest.approx(41.0)
    assert incidence['max'] == pytest.approx(49.0)
    assert incidence['mean'] == pytest.approx(43.0)
    assert incidence['units'] == 'deg'


def test_the_ring_longitude_s_wrapped_range_measures_gaps_against_the_coarsest_pixel(
    tmp_path: Path,
) -> None:
    """Longitudes 6 degrees apart all round cover the circle where a pixel spans 6.5.

    The frame's other pixels span 3 degrees, against which the same longitudes would
    leave gaps: the coarsest pixel is the one the gaps are measured against.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    longitude = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    longitude.flat[:60] = np.radians(np.arange(60) * 6.0)
    resolution = np.where(longitude != MASKED_VALUE, math.radians(3.0), MASKED_VALUE)
    resolution.flat[0] = math.radians(6.5)
    _, sidecar = _write(
        tmp_path,
        master={
            RING_LONGITUDE: longitude,
            RING_LONGITUDINAL_RESOLUTION: resolution.astype(np.float32),
        },
        id_map=_id_map(),
        config=FakeBackplanesConfig(bodies=[], rings=LONGITUDE_PLANES),
        rings_result=RINGS_RESULT,
    )
    statistic = json.loads(sidecar.read_text())['rings']['backplanes'][RING_LONGITUDE]
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (0.0, 360.0)


def test_a_ring_longitude_without_its_resolution_records_no_wrapped_range(tmp_path: Path) -> None:
    """With no longitudinal resolution to measure gaps against, no wrapped range is recorded.

    The configuration declares the ring longitude and not its resolution, so the image's
    metadata records the longitude's plain range alone, and is written as ever.
    """
    longitude = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    longitude.flat[:10] = np.radians(np.arange(10) * 6.0)
    _, sidecar = _write(
        tmp_path,
        master={RING_LONGITUDE: longitude},
        id_map=_id_map(),
        config=FakeBackplanesConfig(bodies=[], rings=LONGITUDE_PLANES[:1]),
        rings_result=RINGS_RESULT,
    )
    statistic = json.loads(sidecar.read_text())['rings']['backplanes'][RING_LONGITUDE]
    assert statistic == {'min': 0.0, 'max': pytest.approx(54.0), 'units': 'deg'}


BODY_LONGITUDE_PLANES = [{'name': BODY_LONGITUDE, 'method': 'longitude', 'units': 'rad'}]
"""The body longitude, as the configuration declares it."""


def _body_longitude(
    tmp_path: Path, degrees: np.ndarray, *, below: np.ndarray | None = None
) -> dict[str, Any]:
    """Write a frame in which MOON_A shows these longitudes, and read back their statistic.

    MOON_A claims a block at the frame's corner as large as ``degrees``, where the body
    longitude plane holds them, in radians, and has no value where ``degrees`` is NaN.
    MOON_B, when ``below`` is given, claims the block of that size just below MOON_A's,
    where the plane holds its longitudes.  Every other pixel holds the masked value.

    Parameters:
        tmp_path: Directory receiving the outputs.
        degrees: MOON_A's longitude at each pixel of its block, in degrees; NaN where the
            longitude plane has no value.
        below: MOON_B's longitude at each pixel of its block, in degrees, or None when the
            frame shows MOON_A alone.

    Returns:
        MOON_A's body longitude statistic, as the writer recorded it.
    """
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True, sim_inventory={})
    blocks = {'MOON_A': degrees} if below is None else {'MOON_A': degrees, 'MOON_B': below}
    longitude = np.full(SHAPE_VU, MASKED_VALUE, dtype=np.float32)
    id_map = np.zeros(SHAPE_VU, dtype=np.int32)
    first = 0
    for name, block in blocks.items():
        rows, columns = block.shape
        longitude[first : first + rows, :columns] = np.where(
            np.isnan(block), MASKED_VALUE, np.radians(block)
        )
        id_map[first : first + rows, :columns] = body_naif_id(snap, name)
        first += rows
    _, sidecar = _write(
        tmp_path,
        master={BODY_LONGITUDE: longitude},
        id_map=id_map,
        snapshot=snap,
        config=FakeBackplanesConfig(bodies=BODY_LONGITUDE_PLANES, rings=[]),
        bodies_result={name: {'arrays': {}, 'masks': {}, 'distance': 500000.0} for name in blocks},
    )
    metadata = json.loads(sidecar.read_text())
    return cast(dict[str, Any], metadata['bodies']['MOON_A']['backplanes'][BODY_LONGITUDE])


def test_a_body_seen_across_its_prime_meridian_records_the_arc_it_covers(tmp_path: Path) -> None:
    """Longitudes from 350 across zero to 10 degrees record an arc from 350 to 10.

    Neighboring pixels step by 5 degrees, across the prime meridian as elsewhere, so the
    gap from 10 round to 350 is longitude the body does not show.
    """
    degrees = np.tile(np.array([350.0, 355.0, 0.0, 5.0, 10.0]), (3, 1))
    statistic = _body_longitude(tmp_path, degrees)
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (
        pytest.approx(350.0),
        pytest.approx(10.0),
    )


def test_a_body_seen_all_round_records_the_whole_circle(tmp_path: Path) -> None:
    """Longitudes all round, as round a pole in view, record the whole circle, 0 to 360.

    The longitudes are 6 degrees apart, and neighboring pixels step by as much as 60, as
    they do round a pole, where every longitude meets: the gaps are the sampling's, and
    no longitude is out of view.
    """
    degrees = (np.arange(60, dtype=np.float64) * 6.0).reshape(6, 10)
    statistic = _body_longitude(tmp_path, degrees)
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (0.0, 360.0)


FALLING = np.array([330.0, 264.0, 198.0, 132.0, 66.0, 0.0])
"""Six longitudes 66 degrees apart, each below the one before it; from the last round to
the first is 30."""


@pytest.mark.parametrize(
    'degrees',
    [np.tile(FALLING[:, np.newaxis], (1, 3)), np.tile(FALLING, (3, 1))],
    ids=['down the frame', 'across the frame'],
)
def test_steps_count_down_and_across_the_frame_whichever_way_they_go(
    tmp_path: Path, degrees: np.ndarray
) -> None:
    """Longitudes changing along one axis alone cover the circle, down it or across it.

    Each pixel's neighbor along one axis holds a longitude 66 degrees below its own, and
    its neighbor along the other the same longitude.  The step that leaves no longitude
    out of view is found along that one axis alone, and the longitude falls along it, so
    a step counts in either direction of the frame and whichever way the longitude goes.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        degrees: MOON_A's longitudes: six rows, or six columns, of one longitude each.
    """
    statistic = _body_longitude(tmp_path, degrees)
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (0.0, 360.0)


def test_a_pixel_of_the_body_with_no_longitude_is_left_out(tmp_path: Path) -> None:
    """A column the body claims where the longitude plane has no value adds nothing.

    The body shows longitudes from 100 to 140 degrees but for one column, where the plane
    holds the masked value, so its arc is 100 to 140, whatever longitude the masked value
    would read as.
    """
    degrees = np.tile(np.arange(100.0, 145.0, 5.0), (5, 1))
    degrees[:, 4] = np.nan
    statistic = _body_longitude(tmp_path, degrees)
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (
        pytest.approx(100.0),
        pytest.approx(140.0),
    )


def test_a_neighboring_pixel_of_another_body_does_not_count(tmp_path: Path) -> None:
    """A body's steps are between its own pixels, not to a body touching it.

    MOON_A shows longitudes from 0 to 200 degrees, 20 apart, so the gap from 200 round to
    0 is longitude it does not show.  MOON_B, just below it, shows 20 degrees, 180 from
    MOON_A's 200; counting that step would read MOON_A's view as the whole circle.
    """
    degrees = np.tile(np.arange(0.0, 220.0, 20.0), (3, 1))
    statistic = _body_longitude(tmp_path, degrees, below=np.full((3, 11), 20.0))
    assert (statistic['wrapped_min'], statistic['wrapped_max']) == (
        pytest.approx(0.0),
        pytest.approx(200.0),
    )


def test_the_shipped_ring_longitude_and_its_resolution_are_in_radians() -> None:
    """The wrapped range is found in degrees, from these two planes' statistics."""
    units = {entry['name']: entry['units'] for entry in DEFAULT_CONFIG.backplanes.rings}
    assert (units[RING_LONGITUDE], units[RING_LONGITUDINAL_RESOLUTION]) == ('rad', 'rad/pixel')


def test_a_body_a_nearer_body_hides_entirely_has_no_statistic(tmp_path: Path) -> None:
    """A body the merge gives no pixel is recorded with no statistic.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    assert _covered_metadata(tmp_path)['bodies']['MOON_B']['backplanes'] == {}


def test_write_fits_non_simulated_uses_config_satellites(tmp_path: Path) -> None:
    """For real images the sidecar inventory covers the planet plus its satellites.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    inventory = {'MOON_A': inventory_entry(u_min=2, u_max=5, v_min=1, v_max=3, body_range=500000.0)}
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=False,
        closest_planet='PLANET',
        canned_inventory=inventory,
    )
    config = FakeBackplanesConfig(
        bodies=[{'name': 'body_latitude', 'method': 'latitude', 'units': 'rad'}],
        rings=[],
        satellites={'PLANET': ['MOON_A', 'MOON_B']},
    )
    _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        config=config,
        bodies_result={},
    )
    assert config.satellites_calls == ['PLANET']
    assert snap.inventory_calls == [['PLANET', 'MOON_A', 'MOON_B']]


@pytest.mark.xfail(
    strict=True,
    reason='#253: suspected doc drift: dev guide promises per-backplane min/max/mean/'
    'valid-pixel-count statistics, but the writer stores only min and max',
)
def test_write_fits_sidecar_includes_mean_and_valid_count(tmp_path: Path) -> None:
    """Sidecar statistics include the documented mean field.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    assert 'mean' in metadata['bodies']['MOON_A']['backplanes']['body_latitude']


@pytest.mark.xfail(
    strict=True,
    reason='#253: suspected doc drift: dev guide promises per-body NAIF ID and predicted '
    'bounding box in the sidecar, but the writer stores neither',
)
def test_write_fits_sidecar_includes_naif_id(tmp_path: Path) -> None:
    """The per-body sidecar entry carries the documented NAIF ID.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    body_keys = set(metadata['bodies']['MOON_A'])
    assert 'naif_id' in body_keys


@pytest.mark.xfail(
    strict=True,
    reason='#253: suspected doc drift: dev guide says the sidecar contains per-image '
    'dataset / instrument / observation metadata, but only bodies and rings are '
    'written',
)
def test_write_fits_sidecar_includes_observation_metadata(tmp_path: Path) -> None:
    """The sidecar carries per-image metadata beyond the bodies and rings blocks.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    metadata = _moon_a_metadata(tmp_path)
    assert len(set(metadata) - {'bodies', 'rings'}) > 0
