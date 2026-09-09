"""Spec-first tests for ring backplane generation.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "Rings" and
docs/user_guide/user_guide_backplanes.rst): the ring step evaluates the configured
methods against the snapshot's full-frame Backplane for the closest planet's ring
system (SATURN uses the SATURN_MAIN_RINGS target), produces per-pixel arrays plus
a per-pixel distance array for the merge, computes min/max statistics (degrees for
'rad' units), and treats the special 'distance' entry as merge-ordering data only,
never as a written FITS HDU.
"""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.ma as ma
import pytest
from astropy.io import fits
from filecache import FCPath

from spindoctor.cli.backplanes.backplanes_rings import create_ring_backplanes
from spindoctor.cli.backplanes.merge import merge_sources_into_master
from spindoctor.cli.backplanes.writer import write_fits
from spindoctor.config import IMAGE_LOGGER

from .conftest import (
    MASKED_VALUE,
    FakeBackplanesConfig,
    FakeRingBackplane,
    HermeticObs,
    make_snapshot,
)

SHAPE_VU = (6, 8)

RADIUS_CFG = {'name': 'ring_radius', 'method': 'ring_radius', 'units': 'km'}
LON_CFG = {'name': 'ring_longitude', 'method': 'ring_longitude', 'units': 'rad'}


def _rings_config(entries: list[dict[str, Any]] | None = None) -> FakeBackplanesConfig:
    """Build a fake config with the given (or default) ring backplane entries.

    Parameters:
        entries: ``backplanes.rings`` entries; defaults to ring_radius only.
    """
    return FakeBackplanesConfig(rings=entries if entries is not None else [RADIUS_CFG])


def _ring_arrays(
    *, value: float = 100000.0, distance: float = 2.0e6
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the ring validity mask and canned per-method masked arrays.

    The ring is valid on rows 2..3 (all columns); everything else is masked.

    Parameters:
        value: Ring backplane value inside the valid region.
        distance: Ring intersection distance inside the valid region.

    Returns:
        The boolean validity mask and the method-name to masked-array map.
    """
    valid = np.zeros(SHAPE_VU, dtype=bool)
    valid[2:4, :] = True
    radius = ma.MaskedArray(np.full(SHAPE_VU, value), mask=~valid)
    longitude = ma.MaskedArray(np.full(SHAPE_VU, 1.5), mask=~valid)
    dist = ma.MaskedArray(np.full(SHAPE_VU, distance), mask=~valid)
    return valid, {'ring_radius': radius, 'ring_longitude': longitude, 'distance': dist}


def _snapshot_with_fake_bp(
    method_values: dict[str, Any], *, planet: str = 'SATURN'
) -> tuple[HermeticObs, FakeRingBackplane]:
    """Build a real-mode snapshot whose full-frame Backplane is a recording fake.

    Parameters:
        method_values: Method-name to masked-array map served by the fake.
        planet: The snapshot's closest planet.

    Returns:
        The observation and the injected fake Backplane.
    """
    snap = make_snapshot(
        shape_vu=SHAPE_VU, simulated=False, closest_planet=planet, canned_inventory={}
    )
    fake = FakeRingBackplane(method_values)
    snap._bp = cast(Any, fake)
    return snap, fake


def test_simulated_snapshot_returns_none() -> None:
    """Simulated observations produce no ring backplanes."""
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is None


def test_no_closest_planet_returns_none() -> None:
    """An observation with no closest planet produces no ring backplanes."""
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=False, closest_planet=None)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is None


def test_missing_rings_config_section_raises() -> None:
    """A config without backplanes.rings is rejected."""
    snap, _ = _snapshot_with_fake_bp(_ring_arrays()[1])
    config = FakeBackplanesConfig(bodies=[])
    with pytest.raises(ValueError, match='no rings section'):
        create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


def test_ring_entry_missing_method_raises() -> None:
    """A ring backplane entry without a method is rejected."""
    snap, _ = _snapshot_with_fake_bp(_ring_arrays()[1])
    config = _rings_config([{'name': 'ring_radius', 'units': 'km'}])
    with pytest.raises(ValueError, match='"method" is required'):
        create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


def test_ring_entry_missing_units_raises() -> None:
    """A ring backplane entry without units is rejected."""
    snap, _ = _snapshot_with_fake_bp(_ring_arrays()[1])
    config = _rings_config([{'name': 'ring_radius', 'method': 'ring_radius'}])
    with pytest.raises(ValueError, match='"units" is required'):
        create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


def test_saturn_uses_main_rings_target() -> None:
    """For Saturn the backplane methods are evaluated on SATURN_MAIN_RINGS."""
    snap, fake = _snapshot_with_fake_bp(_ring_arrays()[1])
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert result['target_key'] == 'SATURN_MAIN_RINGS'
    assert all(call[1] == 'SATURN_MAIN_RINGS' for call in fake.calls)


def test_other_planet_uses_ring_system_target() -> None:
    """For non-Saturn planets the target is <PLANET>_RING_SYSTEM."""
    snap, fake = _snapshot_with_fake_bp(_ring_arrays()[1], planet='JUPITER')
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert result['target_key'] == 'JUPITER_RING_SYSTEM'
    assert all(call[1] == 'JUPITER_RING_SYSTEM' for call in fake.calls)


def test_result_records_planet() -> None:
    """The result carries the closest planet name."""
    snap, _ = _snapshot_with_fake_bp(_ring_arrays()[1])
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert result['planet'] == 'SATURN'


def test_ring_arrays_masked_outside_mask() -> None:
    """Ring arrays carry values where valid and the masked value where absent."""
    valid, method_values = _ring_arrays(value=100000.0)
    snap, _ = _snapshot_with_fake_bp(method_values)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    arr = result['arrays']['ring_radius']
    assert np.all(arr[valid] == np.float32(100000.0))
    assert np.all(arr[~valid] == MASKED_VALUE)


def test_ring_masks_true_where_valid() -> None:
    """The returned validity mask is True exactly where oops left pixels unmasked."""
    valid, method_values = _ring_arrays()
    snap, _ = _snapshot_with_fake_bp(method_values)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert result['arrays']['ring_radius'].shape == SHAPE_VU
    np.testing.assert_array_equal(result['masks']['ring_radius'], valid)


def test_fully_masked_ring_plane_omitted() -> None:
    """A ring plane with no valid pixel is dropped from arrays, masks, and stats."""
    all_masked = ma.MaskedArray(np.full(SHAPE_VU, 1.0), mask=np.ones(SHAPE_VU, dtype=bool))
    _, method_values = _ring_arrays()
    method_values['ring_radius'] = all_masked
    snap, _ = _snapshot_with_fake_bp(method_values)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert 'ring_radius' not in result['arrays']
    assert 'ring_radius' not in result['masks']
    assert 'ring_radius' not in result['statistics']


def test_ring_stats_convert_radians_to_degrees() -> None:
    """Statistics for a 'rad' ring plane are reported in degrees."""
    _, method_values = _ring_arrays()
    snap, _ = _snapshot_with_fake_bp(method_values)
    config = _rings_config([LON_CFG])
    result = create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    stats = result['statistics']['ring_longitude']
    assert stats['min'] == pytest.approx(math.degrees(1.5))
    assert stats['max'] == pytest.approx(math.degrees(1.5))


def test_ring_stats_keep_km_units() -> None:
    """Statistics for a km ring plane are not unit converted."""
    _, method_values = _ring_arrays(value=100000.0)
    snap, _ = _snapshot_with_fake_bp(method_values)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert result['statistics']['ring_radius']['min'] == pytest.approx(100000.0)


def test_configured_distance_entry_feeds_merge_distance() -> None:
    """A configured 'distance' entry populates the per-pixel merge distance."""
    valid, method_values = _ring_arrays(distance=2.0e6)
    snap, _ = _snapshot_with_fake_bp(method_values)
    config = _rings_config([{'name': 'distance', 'method': 'distance', 'units': 'km'}])
    result = create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert np.all(result['distance'][valid] == np.float32(2.0e6))
    assert np.all(np.isinf(result['distance'][~valid]))


def test_fallback_distance_computed_when_not_configured() -> None:
    """Without a configured distance entry the merge distance is computed anyway."""
    valid, method_values = _ring_arrays(distance=3.0e6)
    snap, fake = _snapshot_with_fake_bp(method_values)
    result = create_ring_backplanes(snap, _rings_config().as_config(), logger=IMAGE_LOGGER)
    assert result is not None
    assert np.all(result['distance'][valid] == np.float32(3.0e6))
    assert ('distance', 'SATURN_MAIN_RINGS', {'direction': 'dep'}) in fake.calls


def test_distance_entry_is_not_written_as_hdu(tmp_path: Path) -> None:
    """The special 'distance' ring entry never becomes a FITS HDU.

    Parameters:
        tmp_path: pytest-provided temporary directory.
    """
    _, method_values = _ring_arrays(distance=2.0e6)
    snap, _ = _snapshot_with_fake_bp(method_values)
    config = _rings_config([RADIUS_CFG, {'name': 'distance', 'method': 'distance', 'units': 'km'}])
    rings_result = create_ring_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)
    master, body_id_map = merge_sources_into_master(
        snap, bodies_result={}, rings_result=rings_result
    )
    fits_path = FCPath(tmp_path) / 'IMG1_backplanes.fits'
    write_fits(
        fits_file_path=fits_path,
        snapshot=snap,
        master_by_type=master,
        body_id_map=body_id_map,
        config=config.as_config(),
        bodies_result={},
        rings_result=rings_result,
    )
    with fits.open(fits_path.get_local_path()) as hdul:
        assert 'DISTANCE' not in [hdu.name for hdu in hdul]
