"""Spec-first tests for the backplane FITS + sidecar writer.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "FITS writer" and
docs/user_guide/user_guide_backplanes.rst "Outputs"): the FITS file has an empty
primary HDU, an int32 BODY_ID_MAP as the first image HDU (emitted only when some
pixel has a non-zero ID), and one float32 ImageHDU per non-all-zero backplane with
BUNIT taken from the per-backplane config units.  A sidecar
``<stub>_backplane_metadata.json`` carries per-body inventory information and
per-backplane statistics.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits
from filecache import FCPath

from spindoctor.cli.backplanes.writer import write_fits

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


def _moon_a_setup() -> tuple[HermeticObs, dict[str, Any]]:
    """Build a simulated snapshot with a MOON_A inventory and its bodies_result."""
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
    bodies_result = {
        'MOON_A': {
            'arrays': {},
            'masks': {},
            'distance': 500000.0,
            'statistics': {'body_latitude': {'min': -10.0, 'max': 25.0, 'units': 'deg'}},
        }
    }
    return snap, bodies_result


def test_write_fits_sidecar_body_statistics(tmp_path: Path) -> None:
    """Per-body min/max statistics are written under bodies.<name>.backplanes.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
    assert metadata['bodies']['MOON_A']['backplanes'] == {
        'body_latitude': {'min': -10.0, 'max': 25.0, 'units': 'deg'}
    }


def test_write_fits_sidecar_center_uv_is_swapped_to_vu(tmp_path: Path) -> None:
    """The sidecar center_uv key actually stores (v, u) swapped from the inventory.

    The inventory carries center_uv as (u, v); the writer swaps to (v, u) while
    keeping the key name.  This characterizes the on-disk convention consumed by
    the PDS4 stage; the docs do not specify the ordering.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
    assert metadata['bodies']['MOON_A']['center_uv'] == [2.0, 3.5]


def test_write_fits_sidecar_center_range_and_size(tmp_path: Path) -> None:
    """The sidecar carries center_range and size_uv from the inventory.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
    assert metadata['bodies']['MOON_A']['center_range'] == 500000.0
    assert metadata['bodies']['MOON_A']['size_uv'] == [4.0, 3.0]


def test_write_fits_sidecar_ring_statistics(tmp_path: Path) -> None:
    """Ring statistics are written under rings.backplanes.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    rings_result = {
        'planet': 'PLANET',
        'target_key': 'PLANET_RING_SYSTEM',
        'arrays': {},
        'masks': {},
        'distance': None,
        'statistics': {'ring_radius': {'min': 70000.0, 'max': 140000.0, 'units': 'km'}},
    }
    _, sidecar = _write(
        tmp_path, master=_master_with(1.0), id_map=_id_map(), rings_result=rings_result
    )
    metadata = json.loads(sidecar.read_text())
    assert metadata['rings'] == {
        'backplanes': {'ring_radius': {'min': 70000.0, 'max': 140000.0, 'units': 'km'}}
    }


def test_write_fits_sidecar_omits_body_with_no_content(tmp_path: Path) -> None:
    """A body with no statistics key and no inventory entry is left out.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        bodies_result={'GHOST': {'arrays': {}, 'masks': {}, 'distance': 1.0}},
    )
    metadata = json.loads(sidecar.read_text())
    assert 'GHOST' not in metadata['bodies']


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
        bodies_result={'MOON_A': {'statistics': {}}},
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
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
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
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
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
    snap, bodies_result = _moon_a_setup()
    _, sidecar = _write(
        tmp_path,
        master=_master_with(1.0),
        id_map=_id_map(),
        snapshot=snap,
        bodies_result=bodies_result,
    )
    metadata = json.loads(sidecar.read_text())
    assert len(set(metadata) - {'bodies', 'rings'}) > 0
