"""Truth checks for the simulator's validation scenes.

Every validation scene plants an offset, and every one whose correct outcome is a
success is compared with that truth.  This module holds the fused answer of each
validation scene outside ``algorithmic_invariants`` to its measured tolerance in
:mod:`tests.integration.sim_truth` (the ``algorithmic_invariants`` scenes are checked
by ``test_sim_algorithmic_invariants``, in the default suite).  A scene whose fused
answer is wrong for a known reason carries a band around its known error in its
``expected`` block instead, which ``test_sim_expected`` asserts.  The module also holds
each known wrong answer of a single technique to its band, and checks every validation
scene's fused error against the sigma it reports.

The structural checks need no navigation: every validation scene has a truth check,
every check refers to a validation scene, and no validation scene plants an offset
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
    KNOWN_ERRORS,
    SIGMA_BANDS,
    SIGMA_MULTIPLE,
    TRUTH_TOLERANCES,
    assert_matches_truth,
    error_in_sigmas,
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
_TECHNIQUE_BAND_KEYS = sorted(key for key in KNOWN_ERRORS if key[1] != FUSED)
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
    if _has_known_error(load_sim_scene(path)):
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
    scene = load_sim_scene(path)
    result = _navigated(path.stem)
    assert result.status == 'success', f'{path.stem}: expected success, got {result.status}'
    assert_matches_truth(result.offset_px, scene, FUSED)


@pytest.mark.parametrize(('name', 'technique'), _TECHNIQUE_BAND_KEYS, ids=_TECHNIQUE_BAND_IDS)
def test_known_technique_error_stays_in_its_band(name: str, technique: str) -> None:
    """Each known wrong answer of one technique stays unflagged and inside its band."""
    scene = load_sim_scene(_SCENE_BY_NAME[name])
    matches = [t for t in _navigated(name).per_technique if t.technique_name == technique]
    assert matches, f'{name}: {technique} produced no technique result'
    assert not matches[0].spurious, f'{name}: {technique} is now flagged spurious'
    assert_matches_truth(matches[0].offset_px, scene, technique)


@pytest.mark.parametrize('path', _SUCCESS_PATHS, ids=_SUCCESS_IDS)
def test_fused_error_is_within_the_sigma_multiple(path: Path) -> None:
    """Each validation scene's fused error stays within a multiple of its reported sigma.

    A scene that sits above the multiple for a known reason is held to its band in
    sigmas instead, so a fix leaves the band.
    """
    scene = load_sim_scene(path)
    result = _navigated(path.stem)
    assert result.offset_px is not None, f'{path.stem}: navigation gave no offset'
    sigmas = error_in_sigmas(result, scene)
    if path.stem in SIGMA_BANDS:
        band = SIGMA_BANDS[path.stem]
        assert band.contains(sigmas), (
            f'{path.stem}: fused error is {sigmas:.2f} sigma, outside its known band '
            f'[{band.low}, {band.high}] sigma'
        )
        return
    assert sigmas <= SIGMA_MULTIPLE, (
        f'{path.stem}: fused error is {sigmas:.2f} sigma, above the {SIGMA_MULTIPLE} sigma bound'
    )
