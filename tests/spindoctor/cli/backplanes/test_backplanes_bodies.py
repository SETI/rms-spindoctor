"""Spec-first tests for per-body backplane generation.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "Bodies"): the body
step walks the per-image inventory, clips each body's unclipped bounding box into
the sensor, evaluates the configured methods over a meshgrid restricted to that
box, masks against the body silhouette, embeds the result into a sensor-shaped
frame, records the per-body distance for the merge, and computes per-backplane
min/max statistics (converted from radians to degrees when units are 'rad').
Simulated bodies synthesize a constant value confined to the simulated body mask.
"""

import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
import numpy.ma as ma
import pytest

from spindoctor.cli.backplanes import backplanes_bodies as bodies_mod
from spindoctor.cli.backplanes.backplanes_bodies import (
    _create_simulated_body_backplane,
    create_body_backplanes,
)
from spindoctor.config import IMAGE_LOGGER

from .conftest import (
    MASKED_VALUE,
    FakeBackplanesConfig,
    HermeticObs,
    inventory_entry,
    make_fake_body_backplane_cls,
    make_snapshot,
)

SHAPE_VU = (8, 10)

LAT_CFG = {'name': 'body_latitude', 'method': 'latitude', 'units': 'rad'}
RES_CFG = {'name': 'body_finest_resolution', 'method': 'finest_resolution', 'units': 'km/pixel'}


def _bodies_config(entries: list[dict[str, Any]] | None = None) -> FakeBackplanesConfig:
    """Build a fake config with the given (or default) body backplane entries.

    Parameters:
        entries: ``backplanes.bodies`` entries; defaults to latitude only.
    """
    return FakeBackplanesConfig(bodies=entries if entries is not None else [LAT_CFG])


def _sim_snapshot_with_mimas() -> tuple[HermeticObs, np.ndarray]:
    """Build a simulated snapshot containing one masked body, MIMAS.

    Returns:
        The observation and the full-frame boolean MIMAS mask (v 2..3, u 3..5).
    """
    mask = np.zeros(SHAPE_VU, dtype=bool)
    mask[2:4, 3:6] = True
    inventory = {'MIMAS': inventory_entry(u_min=3, u_max=5, v_min=2, v_max=3, body_range=500000.0)}
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_inventory=inventory,
        sim_body_mask_map={'MIMAS': mask},
    )
    return snap, mask


# ---------------------------------------------------------------------------
# Simulated path
# ---------------------------------------------------------------------------


def test_simulated_body_values_confined_to_body_mask() -> None:
    """Simulated backplane values appear only inside the simulated body mask."""
    snap, mask = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    arr = result['MIMAS']['arrays']['body_latitude']
    assert arr.shape == SHAPE_VU
    assert np.all(arr[mask] > 0.0)
    assert np.all(arr[~mask] == MASKED_VALUE)


def test_simulated_body_mask_matches_sim_mask() -> None:
    """The returned validity mask equals the simulated body mask."""
    snap, mask = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    np.testing.assert_array_equal(result['MIMAS']['masks']['body_latitude'], mask)


def test_simulated_body_value_is_constant_in_range() -> None:
    """The synthesized value is a single constant between 1 and 100."""
    snap, mask = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    values = np.unique(result['MIMAS']['arrays']['body_latitude'][mask])
    assert len(values) == 1
    assert 1.0 <= float(values[0]) <= 100.0


def test_simulated_body_value_deterministic_within_process() -> None:
    """The synthesized value is reproducible for the same body/backplane names."""
    snap, _ = _sim_snapshot_with_mimas()
    config = _bodies_config().as_config()
    first = create_body_backplanes(snap, config, logger=IMAGE_LOGGER)
    second = create_body_backplanes(snap, config, logger=IMAGE_LOGGER)
    val_first = first['MIMAS']['arrays']['body_latitude'][2, 3]
    val_second = second['MIMAS']['arrays']['body_latitude'][2, 3]
    assert val_first == val_second


_SIM_VALUE_SCRIPT = """\
from types import SimpleNamespace

import numpy as np

from spindoctor.cli.backplanes.backplanes_bodies import _create_simulated_body_backplane

snapshot = SimpleNamespace(
    data=np.zeros((8, 10), dtype=np.float32),
    sim_body_mask_map={},
    sim_body_order_near_to_far=[],
    sim_body_index_map=None,
    config=SimpleNamespace(backplanes=SimpleNamespace(masked_value=-999.0)),
)
full, _ = _create_simulated_body_backplane(snapshot, 'MIMAS', 'body_latitude', 2, 3, 3, 5)
print(repr(float(full[2, 3])))
"""


def test_simulated_body_value_deterministic_across_processes() -> None:
    """The synthesized value must not depend on the interpreter hash seed.

    A value derived from the salted built-in ``hash()`` would vary between
    interpreter runs; generating it in fresh subprocesses under different
    PYTHONHASHSEED values detects that, which a within-process rerun cannot.
    The subprocesses drive ``_create_simulated_body_backplane`` on a minimal
    stub snapshot and must agree with each other and with the value produced
    in this process for the same MIMAS / body_latitude inputs.
    """
    snap, _ = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    in_process = float(result['MIMAS']['arrays']['body_latitude'][2, 3])
    values = []
    for seed in ('1', '2', '3'):
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        proc = subprocess.run(
            [sys.executable, '-c', _SIM_VALUE_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        values.append(float(proc.stdout.strip()))
    assert len(set(values)) == 1
    assert values[0] == in_process


def test_simulated_body_distance_from_inventory_range() -> None:
    """The per-body merge distance is the inventory range."""
    snap, _ = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert result['MIMAS']['distance'] == 500000.0


def test_simulated_body_stats_convert_radians_to_degrees() -> None:
    """Statistics for a 'rad' plane are reported in degrees."""
    snap, _ = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    val = float(result['MIMAS']['arrays']['body_latitude'][2, 3])
    stats = result['MIMAS']['statistics']['body_latitude']
    assert stats['min'] == pytest.approx(math.degrees(val))
    assert stats['max'] == pytest.approx(math.degrees(val))


def test_simulated_body_stats_name_the_unit_they_are_in() -> None:
    """The statistic carries the unit it ended up in, not the array's."""
    snap, _ = _sim_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert result['MIMAS']['statistics']['body_latitude']['units'] == 'deg'


def test_simulated_body_stats_keep_non_angle_units() -> None:
    """Statistics for a non-'rad' plane are not unit converted."""
    snap, _ = _sim_snapshot_with_mimas()
    config = _bodies_config([RES_CFG])
    result = create_body_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)
    val = float(result['MIMAS']['arrays']['body_finest_resolution'][2, 3])
    stats = result['MIMAS']['statistics']['body_finest_resolution']
    assert stats['min'] == pytest.approx(val)


def test_bodies_ordered_by_increasing_range() -> None:
    """Result bodies are ordered nearest first even if the inventory is not."""
    mask = np.ones(SHAPE_VU, dtype=bool)
    inventory = {
        'FARBODY': inventory_entry(u_min=0, u_max=2, v_min=0, v_max=2, body_range=9.0e5),
        'NEARBODY': inventory_entry(u_min=4, u_max=6, v_min=4, v_max=6, body_range=1.0e5),
    }
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_inventory=inventory,
        sim_body_mask_map={'FARBODY': mask, 'NEARBODY': mask},
    )
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert list(result) == ['NEARBODY', 'FARBODY']


def test_body_outside_fov_contributes_nothing() -> None:
    """A body whose bounding box misses the sensor is skipped entirely."""
    mask = np.ones(SHAPE_VU, dtype=bool)
    inventory = {'MIMAS': inventory_entry(u_min=-20, u_max=-10, v_min=2, v_max=3, body_range=1.0e5)}
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_inventory=inventory,
        sim_body_mask_map={'MIMAS': mask},
    )
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert result == {}


def test_body_bounding_box_clipped_into_sensor() -> None:
    """A bounding box extending beyond the frame is clipped before evaluation."""
    mask = np.ones(SHAPE_VU, dtype=bool)
    inventory = {'MIMAS': inventory_entry(u_min=-2, u_max=4, v_min=3, v_max=9, body_range=1.0e5)}
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_inventory=inventory,
        sim_body_mask_map={'MIMAS': mask},
    )
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    arr = result['MIMAS']['arrays']['body_latitude']
    assert np.all(arr[3:8, 0:5] > 0.0)
    assert np.all(arr[0:3, :] == MASKED_VALUE)
    assert np.all(arr[:, 5:] == MASKED_VALUE)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


def test_missing_bodies_config_section_raises() -> None:
    """A config without backplanes.bodies is rejected."""
    snap, _ = _sim_snapshot_with_mimas()
    config = FakeBackplanesConfig(rings=[])
    with pytest.raises(ValueError, match='no bodies section'):
        create_body_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


def test_body_entry_missing_method_raises() -> None:
    """A body backplane entry without a method is rejected."""
    snap, _ = _sim_snapshot_with_mimas()
    config = _bodies_config([{'name': 'body_latitude', 'units': 'rad'}])
    with pytest.raises(ValueError, match='"method" is required'):
        create_body_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


def test_body_entry_missing_units_raises() -> None:
    """A body backplane entry without units is rejected."""
    snap, _ = _sim_snapshot_with_mimas()
    config = _bodies_config([{'name': 'body_latitude', 'method': 'latitude'}])
    with pytest.raises(ValueError, match='"units" is required'):
        create_body_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)


# ---------------------------------------------------------------------------
# Real (non-simulated) path with a stubbed oops Backplane
# ---------------------------------------------------------------------------


def test_no_closest_planet_returns_no_bodies() -> None:
    """A real image with no closest planet produces no body backplanes."""
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=False, closest_planet=None)
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert result == {}


def test_inventory_queried_for_planet_and_satellites() -> None:
    """The inventory is built for the closest planet plus its config satellites."""
    snap = make_snapshot(
        shape_vu=SHAPE_VU, simulated=False, closest_planet='PLANET', canned_inventory={}
    )
    config = FakeBackplanesConfig(bodies=[LAT_CFG], satellites={'PLANET': ['MIMAS', 'ENCELADUS']})
    create_body_backplanes(snap, config.as_config(), logger=IMAGE_LOGGER)
    assert snap.inventory_calls == [['PLANET', 'MIMAS', 'ENCELADUS']]


def _real_snapshot_with_mimas() -> HermeticObs:
    """Build a non-simulated snapshot with a canned MIMAS inventory (u 3..6, v 2..4)."""
    inventory = {'MIMAS': inventory_entry(u_min=3, u_max=6, v_min=2, v_max=4, body_range=200000.0)}
    return make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=False,
        closest_planet='PLANET',
        canned_inventory=inventory,
    )


def _patch_backplane_constant(
    monkeypatch: pytest.MonkeyPatch, *, value: float, masked_corner: bool = False
) -> None:
    """Replace the oops Backplane with a fake returning a constant masked array.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
        value: Constant value returned at every meshgrid pixel.
        masked_corner: Whether to mask the first (local (0, 0)) meshgrid pixel.
    """

    def values_fn(method: str, body_name: str, shape: tuple[int, int]) -> Any:
        """Return a constant masked array, optionally masking the first pixel.

        Parameters:
            method: The oops Backplane method name (ignored).
            body_name: The body being evaluated (ignored).
            shape: The meshgrid shape as (nv, nu).
        """
        data = np.full(shape, value)
        mask = np.zeros(shape, dtype=bool)
        if masked_corner:
            mask[0, 0] = True
        return ma.MaskedArray(data, mask=mask)

    monkeypatch.setattr(bodies_mod, 'Backplane', make_fake_body_backplane_cls(values_fn))


def test_real_body_values_embedded_at_bounding_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluated values land in the sensor frame exactly at the clipped box.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
    """
    _patch_backplane_constant(monkeypatch, value=0.75)
    snap = _real_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    arr = result['MIMAS']['arrays']['body_latitude']
    assert arr[2, 3] == np.float32(0.75)
    assert arr[4, 6] == np.float32(0.75)
    assert np.all(arr[5:, :] == MASKED_VALUE)
    assert np.all(arr[:2, :] == MASKED_VALUE)
    assert np.all(arr[:, 7:] == MASKED_VALUE)


def test_real_body_masked_pixels_carry_the_masked_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pixels masked by oops carry the configured masked value and are marked invalid.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
    """
    _patch_backplane_constant(monkeypatch, value=0.75, masked_corner=True)
    snap = _real_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    arr = result['MIMAS']['arrays']['body_latitude']
    mask = result['MIMAS']['masks']['body_latitude']
    assert arr[2, 3] == MASKED_VALUE
    assert not bool(mask[2, 3])
    assert arr[2, 4] == np.float32(0.75)
    assert bool(mask[2, 4])


def test_real_body_distance_is_inventory_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """The merge distance for a real body is the inventory range.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
    """
    _patch_backplane_constant(monkeypatch, value=0.75)
    snap = _real_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert result['MIMAS']['distance'] == 200000.0


def test_real_body_stats_convert_radians_to_degrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path statistics for a 'rad' plane are reported in degrees.

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
    """
    _patch_backplane_constant(monkeypatch, value=0.5)
    snap = _real_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    stats = result['MIMAS']['statistics']['body_latitude']
    assert stats['min'] == pytest.approx(math.degrees(0.5))
    assert stats['max'] == pytest.approx(math.degrees(0.5))


def test_real_body_no_stats_when_fully_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plane with no valid pixel gets no statistics entry (arrays still present).

    Parameters:
        monkeypatch: pytest monkeypatch fixture.
    """

    def values_fn(method: str, body_name: str, shape: tuple[int, int]) -> Any:
        """Return a fully masked constant array (no valid pixel anywhere).

        Parameters:
            method: The oops Backplane method name (ignored).
            body_name: The body being evaluated (ignored).
            shape: The meshgrid shape as (nv, nu).
        """
        return ma.MaskedArray(np.full(shape, 1.0), mask=np.ones(shape, dtype=bool))

    monkeypatch.setattr(bodies_mod, 'Backplane', make_fake_body_backplane_cls(values_fn))
    snap = _real_snapshot_with_mimas()
    result = create_body_backplanes(snap, _bodies_config().as_config(), logger=IMAGE_LOGGER)
    assert 'body_latitude' not in result['MIMAS']['statistics']
    assert 'body_latitude' in result['MIMAS']['arrays']


# ---------------------------------------------------------------------------
# _create_simulated_body_backplane internals
# ---------------------------------------------------------------------------


def test_simulated_backplane_mask_map_lookup_is_case_insensitive() -> None:
    """A lower-case body name matches its upper-case mask-map entry."""
    mask = np.zeros(SHAPE_VU, dtype=bool)
    mask[2:4, 3:6] = True
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True, sim_body_mask_map={'MIMAS': mask})
    full, full_mask = _create_simulated_body_backplane(snap, 'mimas', 'body_latitude', 2, 3, 3, 5)
    assert np.all(full[mask] > 0.0)
    assert np.all(full_mask == mask)


def test_simulated_backplane_index_map_fallback() -> None:
    """Without a mask map the body's slot in the index map defines the mask."""
    index_map = np.zeros(SHAPE_VU, dtype=np.int32)
    index_map[2, 3] = 2
    index_map[3, 4] = 1
    snap = make_snapshot(
        shape_vu=SHAPE_VU,
        simulated=True,
        sim_body_order_near_to_far=['ALPHA', 'BETA'],
        sim_body_index_map=index_map,
    )
    full, full_mask = _create_simulated_body_backplane(snap, 'BETA', 'body_latitude', 2, 4, 3, 5)
    assert full[2, 3] > 0.0
    assert full[3, 4] == MASKED_VALUE
    assert bool(full_mask[2, 3])
    assert not bool(full_mask[3, 4])


def test_simulated_backplane_rect_fallback_fills_whole_box() -> None:
    """A body absent from every sim structure falls back to filling the rectangle."""
    snap = make_snapshot(shape_vu=SHAPE_VU, simulated=True)
    full, full_mask = _create_simulated_body_backplane(snap, 'UNKNOWN', 'body_latitude', 2, 3, 3, 5)
    assert np.all(full[2:4, 3:6] > 0.0)
    assert np.all(full_mask[2:4, 3:6])
    assert np.all(full[4:, :] == MASKED_VALUE)
