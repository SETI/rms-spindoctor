"""Spec-first tests for the distance-aware backplane merge.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst, "Distance-aware
merge"): every pixel is owned by the source (body or ring) with the smallest
finite distance along the line of sight; the winner's per-backplane values are
copied into the master arrays; pixels with no finite distance from any source
stay zero; a per-pixel BODY_ID_MAP carries the NAIF ID of the winning source,
with bodies using real NAIF IDs and rings a deterministic ring-system ID.
"""

import os
import subprocess
import sys
from typing import Any

import cspyce
import numpy as np
import pytest

from spindoctor.cli.backplanes.merge import fake_naif_id, merge_sources_into_master

from .conftest import MASKED_VALUE, HermeticObs, make_snapshot

SHAPE_VU = (7, 9)


def _rect_mask(shape_vu: tuple[int, int], v0: int, v1: int, u0: int, u1: int) -> np.ndarray:
    """Build a boolean mask that is True on the inclusive rectangle (v0..v1, u0..u1).

    Parameters:
        shape_vu: Full-frame shape as (rows, columns).
        v0: First row of the rectangle.
        v1: Last row of the rectangle (inclusive).
        u0: First column of the rectangle.
        u1: Last column of the rectangle (inclusive).
    """
    mask = np.zeros(shape_vu, dtype=bool)
    mask[v0 : v1 + 1, u0 : u1 + 1] = True
    return mask


def _body_entry(
    shape_vu: tuple[int, int],
    mask: np.ndarray,
    *,
    value: float,
    distance: float,
    bp_type: str = 'body_latitude',
) -> dict[str, Any]:
    """Build a bodies_result entry with one constant-valued backplane.

    Parameters:
        shape_vu: Full-frame shape as (rows, columns).
        mask: Boolean validity mask (True where the body silhouette is).
        value: Constant backplane value inside the mask.
        distance: Scalar body distance in km.
        bp_type: Backplane type name.
    """
    arr = np.zeros(shape_vu, dtype=np.float32)
    arr[mask] = value
    return {
        'arrays': {bp_type: arr},
        'masks': {bp_type: mask},
        'distance': distance,
        'statistics': {},
    }


def _rings_result(
    arrays: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    distance: np.ndarray,
) -> dict[str, Any]:
    """Build a rings_result dict in the shape create_ring_backplanes returns.

    Parameters:
        arrays: Ring backplane arrays keyed by type name.
        masks: Ring validity masks keyed by type name.
        distance: Per-pixel ring distance array (inf where no intersection).
    """
    return {
        'planet': 'SATURN',
        'target_key': 'SATURN_MAIN_RINGS',
        'arrays': arrays,
        'masks': masks,
        'distance': distance,
        'statistics': {},
    }


def _ring_only(
    shape_vu: tuple[int, int],
    mask: np.ndarray,
    *,
    value: float,
    distance_value: float,
    bp_type: str = 'ring_radius',
) -> dict[str, Any]:
    """Build a rings_result with one constant-valued ring plane.

    Parameters:
        shape_vu: Full-frame shape as (rows, columns).
        mask: Boolean validity mask for the ring plane.
        value: Constant ring backplane value inside the mask.
        distance_value: Ring distance inside the mask (inf outside).
        bp_type: Ring backplane type name.
    """
    arr = np.full(shape_vu, np.nan, dtype=np.float32)
    arr[mask] = value
    distance = np.full(shape_vu, np.inf, dtype=np.float32)
    distance[mask] = distance_value
    return _rings_result({bp_type: arr}, {bp_type: mask}, distance)


@pytest.fixture
def snapshot() -> HermeticObs:
    """Non-simulated hermetic snapshot with a non-square frame."""
    return make_snapshot(shape_vu=SHAPE_VU, simulated=False)


def test_merge_no_sources_returns_empty_master(snapshot: HermeticObs) -> None:
    """With no bodies and no rings the master dict is empty.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    master, _ = merge_sources_into_master(snapshot, bodies_result={}, rings_result=None)
    assert master == {}


def test_merge_no_sources_id_map_shape_and_zero(snapshot: HermeticObs) -> None:
    """With no sources the BODY_ID_MAP is all zeros at the sensor shape.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    _, body_id_map = merge_sources_into_master(snapshot, bodies_result={}, rings_result=None)
    assert body_id_map.shape == SHAPE_VU
    assert body_id_map.dtype == np.int32
    assert np.all(body_id_map == 0)


def test_merge_single_body_copies_values_inside_silhouette(snapshot: HermeticObs) -> None:
    """A lone body's values are copied into the master at every valid pixel.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 3, 2, 5)
    bodies = {'SATURN': _body_entry(SHAPE_VU, mask, value=0.5, distance=1.0e6)}
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)
    assert np.all(master['body_latitude'][mask] == np.float32(0.5))


def test_merge_single_body_id_map_carries_naif_id(snapshot: HermeticObs) -> None:
    """BODY_ID_MAP carries the body's real NAIF ID at every valid pixel.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 3, 2, 5)
    bodies = {'SATURN': _body_entry(SHAPE_VU, mask, value=0.5, distance=1.0e6)}
    _, body_id_map = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)
    assert np.all(body_id_map[mask] == int(cspyce.bodn2c('SATURN')))
    assert np.all(body_id_map[~mask] == 0)


def test_merge_unclaimed_pixels_carry_the_masked_value(snapshot: HermeticObs) -> None:
    """Pixels outside every source silhouette carry the masked value.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 3, 2, 5)
    bodies = {'SATURN': _body_entry(SHAPE_VU, mask, value=0.5, distance=1.0e6)}
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)
    assert np.all(master['body_latitude'][~mask] == MASKED_VALUE)


def test_merge_nearer_body_wins_overlap_values(snapshot: HermeticObs) -> None:
    """Where two bodies overlap, the one with the smaller range owns the pixel.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    far_mask = _rect_mask(SHAPE_VU, 1, 4, 1, 4)
    near_mask = _rect_mask(SHAPE_VU, 2, 5, 3, 6)
    bodies = {
        'SATURN': _body_entry(SHAPE_VU, far_mask, value=100.0, distance=10.0),
        'MIMAS': _body_entry(SHAPE_VU, near_mask, value=200.0, distance=5.0),
    }
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)
    overlap = far_mask & near_mask
    assert np.all(master['body_latitude'][overlap] == np.float32(200.0))
    assert np.all(master['body_latitude'][far_mask & ~near_mask] == np.float32(100.0))
    assert np.all(master['body_latitude'][near_mask & ~far_mask] == np.float32(200.0))


def test_merge_nearer_body_wins_overlap_id_map(snapshot: HermeticObs) -> None:
    """The BODY_ID_MAP mirrors per-pixel body ownership in an overlap.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    far_mask = _rect_mask(SHAPE_VU, 1, 4, 1, 4)
    near_mask = _rect_mask(SHAPE_VU, 2, 5, 3, 6)
    bodies = {
        'SATURN': _body_entry(SHAPE_VU, far_mask, value=100.0, distance=10.0),
        'MIMAS': _body_entry(SHAPE_VU, near_mask, value=200.0, distance=5.0),
    }
    _, body_id_map = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)
    assert np.all(body_id_map[near_mask] == int(cspyce.bodn2c('MIMAS')))
    assert np.all(body_id_map[far_mask & ~near_mask] == int(cspyce.bodn2c('SATURN')))


def test_merge_single_pixel_body(snapshot: HermeticObs) -> None:
    """A single-pixel body claims exactly one pixel in master and ID map.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = np.zeros(SHAPE_VU, dtype=bool)
    mask[4, 7] = True
    bodies = {'MIMAS': _body_entry(SHAPE_VU, mask, value=3.5, distance=1.0e5)}
    master, body_id_map = merge_sources_into_master(
        snapshot, bodies_result=bodies, rings_result=None
    )
    assert master['body_latitude'][4, 7] == np.float32(3.5)
    assert int(np.count_nonzero(body_id_map)) == 1


def test_merge_inconsistent_masks_within_body_raise(snapshot: HermeticObs) -> None:
    """Differing per-type masks for one body are rejected.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask_a = _rect_mask(SHAPE_VU, 1, 2, 1, 2)
    mask_b = _rect_mask(SHAPE_VU, 3, 4, 3, 4)
    arr = np.zeros(SHAPE_VU, dtype=np.float32)
    bodies = {
        'SATURN': {
            'arrays': {'a': arr, 'b': arr},
            'masks': {'a': mask_a, 'b': mask_b},
            'distance': 1.0,
            'statistics': {},
        }
    }
    with pytest.raises(ValueError, match='not all the same'):
        merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)


def test_merge_missing_backplane_type_for_body_raises(snapshot: HermeticObs) -> None:
    """A backplane type present on one body but missing on another is rejected.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 2, 1, 2)
    bodies = {
        'SATURN': _body_entry(SHAPE_VU, mask, value=1.0, distance=1.0, bp_type='a'),
        'MIMAS': _body_entry(SHAPE_VU, mask, value=2.0, distance=2.0, bp_type='b'),
    }
    with pytest.raises(ValueError, match='not found for body'):
        merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)


def test_merge_unknown_body_name_raises_for_real_data(snapshot: HermeticObs) -> None:
    """A body name unknown to SPICE is a hard error on real (non-simulated) images.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 2, 1, 2)
    bodies = {'NOT_A_BODY_XYZ': _body_entry(SHAPE_VU, mask, value=1.0, distance=1.0)}
    with pytest.raises(KeyError, match='NOT_A_BODY_XYZ'):
        merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=None)


def test_merge_unknown_simulated_body_gets_fake_id() -> None:
    """On simulated images an unknown body gets a synthesized int32-range NAIF ID."""
    sim = make_snapshot(shape_vu=SHAPE_VU, simulated=True)
    mask = _rect_mask(SHAPE_VU, 1, 2, 1, 2)
    bodies = {'BLOB_XQZ': _body_entry(SHAPE_VU, mask, value=1.0, distance=1.0)}
    _, body_id_map = merge_sources_into_master(sim, bodies_result=bodies, rings_result=None)
    fake_id = int(body_id_map[1, 1])
    assert fake_id >= 10000
    assert fake_id < 30000


def test_merge_simulated_fake_id_deterministic_across_processes() -> None:
    """The synthesized fake NAIF ID must not depend on the interpreter hash seed."""
    expr = "from spindoctor.cli.backplanes.merge import fake_naif_id; print(fake_naif_id('MIMAS'))"
    ids = []
    for seed in ('1', '2', '3'):
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        proc = subprocess.run(
            [sys.executable, '-c', expr],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        ids.append(int(proc.stdout.strip()))
    assert len(set(ids)) == 1
    assert ids[0] == fake_naif_id('MIMAS')


def test_merge_rings_only_fill_valid_pixels(snapshot: HermeticObs) -> None:
    """With no bodies, ring values are copied wherever the ring plane is valid.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    rings = _ring_only(SHAPE_VU, ring_mask, value=123456.0, distance_value=2.0e6)
    master, _ = merge_sources_into_master(snapshot, bodies_result={}, rings_result=rings)
    assert np.all(master['ring_radius'][ring_mask] == np.float32(123456.0))
    assert np.all(master['ring_radius'][~ring_mask] == MASKED_VALUE)


@pytest.mark.xfail(
    strict=True,
    reason='#251: suspected bug: merge never writes a ring-system NAIF ID into BODY_ID_MAP; '
    'the dev guide says rings use a deterministic ring-system ID for pixels they win',
)
def test_merge_ring_pixels_carry_ring_system_id(snapshot: HermeticObs) -> None:
    """Pixels won by the ring system carry a non-zero ring ID in BODY_ID_MAP.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    rings = _ring_only(SHAPE_VU, ring_mask, value=123456.0, distance_value=2.0e6)
    _, body_id_map = merge_sources_into_master(snapshot, bodies_result={}, rings_result=rings)
    assert np.all(body_id_map[ring_mask] != 0)


def test_merge_body_in_front_occludes_ring(snapshot: HermeticObs) -> None:
    """A body nearer than the ring plane blanks the ring values at its pixels.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    body_mask = _rect_mask(SHAPE_VU, 2, 3, 2, 4)
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    bodies = {'MIMAS': _body_entry(SHAPE_VU, body_mask, value=1.5, distance=5.0)}
    rings = _ring_only(SHAPE_VU, ring_mask, value=99999.0, distance_value=10.0)
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=rings)
    assert np.all(master['ring_radius'][body_mask] == MASKED_VALUE)
    assert np.all(master['ring_radius'][ring_mask & ~body_mask] == np.float32(99999.0))
    assert np.all(master['body_latitude'][body_mask] == np.float32(1.5))


def test_merge_ring_in_front_of_body_keeps_ring_values(snapshot: HermeticObs) -> None:
    """A ring nearer than a body is not occluded: its values survive the merge.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    body_mask = _rect_mask(SHAPE_VU, 2, 3, 2, 4)
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    bodies = {'MIMAS': _body_entry(SHAPE_VU, body_mask, value=1.5, distance=10.0)}
    rings = _ring_only(SHAPE_VU, ring_mask, value=99999.0, distance_value=5.0)
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=rings)
    assert np.all(master['ring_radius'][ring_mask] == np.float32(99999.0))


@pytest.mark.xfail(
    strict=True,
    reason='#252: suspected bug or doc drift: dev guide says the nearest source owns the '
    'pixel, but body planes keep the occluded body values when the ring is nearer',
)
def test_merge_ring_in_front_of_body_owns_body_plane(snapshot: HermeticObs) -> None:
    """Body planes are unclaimed where a nearer ring wins the pixel.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    body_mask = _rect_mask(SHAPE_VU, 2, 3, 2, 4)
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    bodies = {'MIMAS': _body_entry(SHAPE_VU, body_mask, value=1.5, distance=10.0)}
    rings = _ring_only(SHAPE_VU, ring_mask, value=99999.0, distance_value=5.0)
    master, _ = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=rings)
    assert np.all(master['body_latitude'][body_mask] == 0.0)


@pytest.mark.xfail(
    strict=True,
    reason='#252: suspected bug or doc drift: BODY_ID_MAP records the occluded body even at '
    'pixels the dev guide says are won by the nearer ring',
)
def test_merge_ring_in_front_of_body_owns_id_map(snapshot: HermeticObs) -> None:
    """BODY_ID_MAP does not report the body at pixels a nearer ring won.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    body_mask = _rect_mask(SHAPE_VU, 2, 3, 2, 4)
    ring_mask = _rect_mask(SHAPE_VU, 2, 4, 0, 8)
    bodies = {'MIMAS': _body_entry(SHAPE_VU, body_mask, value=1.5, distance=10.0)}
    rings = _ring_only(SHAPE_VU, ring_mask, value=99999.0, distance_value=5.0)
    _, body_id_map = merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=rings)
    assert np.all(body_id_map[body_mask] != int(cspyce.bodn2c('MIMAS')))


def test_merge_body_ring_type_name_collision_raises(snapshot: HermeticObs) -> None:
    """A backplane type used by both a body and the rings is rejected.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    mask = _rect_mask(SHAPE_VU, 1, 2, 1, 2)
    bodies = {'SATURN': _body_entry(SHAPE_VU, mask, value=1.0, distance=1.0, bp_type='shared')}
    rings = _ring_only(SHAPE_VU, mask, value=2.0, distance_value=0.5, bp_type='shared')
    with pytest.raises(ValueError, match='already exists in master_by_type'):
        merge_sources_into_master(snapshot, bodies_result=bodies, rings_result=rings)


def test_merge_rings_result_with_no_arrays(snapshot: HermeticObs) -> None:
    """A rings result carrying no valid planes contributes nothing.

    Parameters:
        snapshot: Hermetic observation fixture.
    """
    rings = _rings_result({}, {}, np.full(SHAPE_VU, np.inf, dtype=np.float32))
    master, body_id_map = merge_sources_into_master(snapshot, bodies_result={}, rings_result=rings)
    assert master == {}
    assert np.all(body_id_map == 0)
