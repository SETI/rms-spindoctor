"""Truth checks for the simulator's validation scenes.

Every validation scene plants an offset, and every one whose correct outcome is a
success is compared with that truth.  This module holds the fused answer of each
validation scene outside ``algorithmic_invariants`` to its measured tolerance in
:mod:`tests.integration.sim_truth` (the ``algorithmic_invariants`` scenes are checked
by ``test_sim_algorithmic_invariants``, in the default suite).  A scene whose answer is
wrong for a known reason carries a band around its known error in its ``expected``
block instead, which ``test_sim_expected`` asserts.

The structural checks need no navigation: every validation scene has a truth check,
every tolerance names a validation scene, and no validation scene plants an offset
whose fractional part is 0 or 0.5 on either axis, where a disc or star error can vanish
by symmetry.

Navigation renders each scene and runs the full ensemble, so the module is
``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import functools
from pathlib import Path
from typing import Any

import pytest

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene import load_sim_scene, scene_class_for_path
from tests.integration.sim_expected import navigate_scene
from tests.integration.sim_truth import (
    FUSED,
    TRUTH_TOLERANCES,
    assert_within_tolerance,
    validation_scene_paths,
)

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_VALIDATION_PATHS = validation_scene_paths(_SCENES_ROOT)
_SCENE_BY_NAME = {p.stem: p for p in _VALIDATION_PATHS}

# disc_subpixel_offset plants a whole-pixel offset on purpose: the bias it guards is
# largest at integer offsets.
_WHOLE_PIXEL_SUBJECTS = frozenset({'disc_subpixel_offset'})
# The rank-1 flat-ring scene is checked on its one observable component, against
# RANK1_NORMAL_ERROR, in test_sim_algorithmic_invariants.
_COMPONENT_CHECKED = frozenset({'planted_rank1_ring_flat'})


def _expects_success(scene: dict[str, Any]) -> bool:
    """True unless the scene's expected block declares a failed or conflicted outcome."""
    expected = scene.get('expected')
    return not isinstance(expected, dict) or expected.get('status') == 'success'


def _has_known_error(scene: dict[str, Any]) -> bool:
    """True when the scene's expected block holds a known fused error."""
    expected = scene.get('expected')
    return isinstance(expected, dict) and expected.get('known_offset_error_px') is not None


_SUCCESS_PATHS = [p for p in _VALIDATION_PATHS if _expects_success(load_sim_scene(p))]
_SUCCESS_IDS = [p.stem for p in _SUCCESS_PATHS]
_FUSED_CHECK_PATHS = [
    p
    for p in _SUCCESS_PATHS
    if scene_class_for_path(p) != 'algorithmic_invariants' and (p.stem, FUSED) in TRUTH_TOLERANCES
]
_FUSED_CHECK_IDS = [p.stem for p in _FUSED_CHECK_PATHS]


@functools.cache
def _navigated(scene_name: str) -> NavResult:
    """Render and navigate a validation scene once per test process."""
    return navigate_scene(load_sim_scene(_SCENE_BY_NAME[scene_name]))


def test_every_validation_scene_has_a_truth_check() -> None:
    """Each validation scene expected to succeed has a tolerance or a known error."""
    unchecked = [
        p.stem
        for p in _SUCCESS_PATHS
        if not _has_known_error(load_sim_scene(p))
        and not any(name == p.stem for name, _technique in TRUTH_TOLERANCES)
        and p.stem not in _COMPONENT_CHECKED
    ]
    assert unchecked == []


def test_every_tolerance_names_a_validation_scene() -> None:
    """No truth tolerance outlives its scene or names a sweep scene."""
    assert {name for name, _technique in TRUTH_TOLERANCES} <= set(_SCENE_BY_NAME)


def test_validation_offsets_are_not_whole_or_half_pixels() -> None:
    """No validation scene plants an offset whose fraction is 0 or 0.5 on an axis."""
    symmetric = []
    for path in _VALIDATION_PATHS:
        if path.stem in _WHOLE_PIXEL_SUBJECTS:
            continue
        scene = load_sim_scene(path)
        for key in ('offset_v', 'offset_u'):
            fraction = abs(float(scene.get(key, 0.0))) % 0.5
            if min(fraction, 0.5 - fraction) < 1.0e-9:
                symmetric.append(f'{path.stem}.{key}')
    assert symmetric == []


@pytest.mark.parametrize('path', _FUSED_CHECK_PATHS, ids=_FUSED_CHECK_IDS)
def test_fused_offset_is_within_its_truth_tolerance(path: Path) -> None:
    """Each validation scene's fused offset lies within its measured tolerance."""
    scene = load_sim_scene(path)
    result = _navigated(path.stem)
    assert result.status == 'success', f'{path.stem}: expected success, got {result.status}'
    assert_within_tolerance(result.offset_px, scene, FUSED)
