"""Truth checks for the simulator's validation scenes.

Every validation scene plants an offset.  Each one whose correct outcome is a success is
compared with that truth.  This module holds the fused answer of each validation scene
outside ``algorithmic_invariants`` to its measured tolerance in
:mod:`tests.integration.sim_truth`.  ``test_sim_algorithmic_invariants`` checks the
``algorithmic_invariants`` scenes in the default suite.  If a scene's fused answer is
wrong for a known reason, its ``expected`` block holds a band around the known error
instead, and ``test_sim_expected`` asserts it.  This module also holds each known wrong
answer of a single technique to its band.  Last, it checks the fused error of every
validation scene expected to succeed against the sigma the navigator reports.

The structural checks need no navigation: every validation scene has a truth check,
every check refers to a validation scene, and no validation scene plants an offset
whose fractional part is 0 or 0.5 on either axis, where a disc or star error can vanish
by symmetry.  They cover every validation scene.  The navigated checks leave out the
classes in :data:`~tests.integration.sim_truth.OWN_MODULE_CLASSES`, whose own modules
apply the same checks (:mod:`tests.integration.sim_scene_checks`).

Navigation renders each scene and runs the full ensemble, so the module is
``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import functools
from pathlib import Path

import pytest

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene.io import load_sim_scene, scene_class_for_path
from tests.integration.sim_expected import navigate_scene
from tests.integration.sim_scene_checks import (
    SCENES_ROOT,
    assert_fused_within_tolerance,
    assert_technique_in_band,
    assert_within_sigma_multiple,
    fused_tolerance_paths,
    has_known_error,
    has_own_module,
    success_paths,
    technique_band_keys,
)
from tests.integration.sim_truth import (
    FUSED,
    KNOWN_ERRORS,
    SIGMA_BANDS,
    TRUTH_TOLERANCES,
    validation_scene_paths,
)

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_VALIDATION_PATHS = validation_scene_paths(SCENES_ROOT)
_SCENE_BY_NAME = {p.stem: p for p in _VALIDATION_PATHS}
_NAVIGATED_PATHS = [p for p in _VALIDATION_PATHS if not has_own_module(p)]

# disc_subpixel_offset plants a whole-pixel offset on purpose: the bias it guards is
# largest at integer offsets.
_WHOLE_PIXEL_SUBJECTS = frozenset({'disc_subpixel_offset'})
# The rank-1 flat-ring scene is checked on its one observable component, against
# RANK1_NORMAL_ERROR, in test_sim_algorithmic_invariants.
_COMPONENT_CHECKED = frozenset({'planted_rank1_ring_flat'})

_SUCCESS_PATHS = success_paths(_VALIDATION_PATHS)
_SIGMA_CHECK_PATHS = success_paths(_NAVIGATED_PATHS)
_SIGMA_CHECK_IDS = [p.stem for p in _SIGMA_CHECK_PATHS]
_FUSED_CHECK_PATHS = fused_tolerance_paths(_NAVIGATED_PATHS)
_FUSED_CHECK_IDS = [p.stem for p in _FUSED_CHECK_PATHS]
_TECHNIQUE_BAND_KEYS = technique_band_keys(_NAVIGATED_PATHS)
_TECHNIQUE_BAND_IDS = [f'{name}-{technique}' for name, technique in _TECHNIQUE_BAND_KEYS]


@functools.cache
def _navigated(scene_name: str) -> NavResult:
    """Render and navigate a validation scene once per test process."""
    return navigate_scene(load_sim_scene(_SCENE_BY_NAME[scene_name]))


def _has_truth_check(path: Path) -> bool:
    """True when the scene has the truth check its class requires.

    A scene outside ``algorithmic_invariants`` needs a check on its fused answer: a
    tolerance, a known-error band, or a known error in its expected block.  An
    ``algorithmic_invariants`` scene may instead be checked on one technique or one
    component.
    """
    if has_known_error(load_sim_scene(path)):
        return True
    checked = set(TRUTH_TOLERANCES) | set(KNOWN_ERRORS)
    if scene_class_for_path(path) != 'algorithmic_invariants':
        return (path.stem, FUSED) in checked
    return path.stem in _COMPONENT_CHECKED or any(name == path.stem for name, _ in checked)


def test_every_validation_scene_has_a_truth_check() -> None:
    """Each validation scene expected to succeed has the truth check its class requires."""
    assert [p.stem for p in _SUCCESS_PATHS if not _has_truth_check(p)] == []


def test_every_truth_check_refers_to_a_validation_scene() -> None:
    """No tolerance or band outlives its scene or refers to a sweep scene."""
    checked = set(TRUTH_TOLERANCES) | set(KNOWN_ERRORS)
    assert {name for name, _technique in checked} | set(SIGMA_BANDS) <= set(_SCENE_BY_NAME)


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
    assert_fused_within_tolerance(load_sim_scene(path), _navigated(path.stem))


@pytest.mark.parametrize(('name', 'technique'), _TECHNIQUE_BAND_KEYS, ids=_TECHNIQUE_BAND_IDS)
def test_known_technique_error_stays_in_its_band(name: str, technique: str) -> None:
    """Each known wrong answer of one technique stays unflagged and inside its band."""
    scene = load_sim_scene(_SCENE_BY_NAME[name])
    assert_technique_in_band(scene, _navigated(name), technique)


@pytest.mark.parametrize('path', _SIGMA_CHECK_PATHS, ids=_SIGMA_CHECK_IDS)
def test_fused_error_is_within_the_sigma_multiple(path: Path) -> None:
    """Each validation scene's fused error stays within a multiple of its reported sigma.

    A scene that sits above the multiple for a known reason is held to its band in
    sigmas instead, so a fix leaves the band.
    """
    assert_within_sigma_multiple(load_sim_scene(path), _navigated(path.stem))
