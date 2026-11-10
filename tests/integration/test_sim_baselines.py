"""Regression baselines for the sim scene catalog.

Each catalog scene has a baseline JSON recording its navigation outcome.  These
tests re-render and re-navigate every scene and assert the outcome still matches
its baseline, turning "this scene navigates to X" into a tripwire.  The status and
the confidence rank must match exactly, and the offsets and the confidence within
the tolerances :mod:`tests.integration.sim_baseline` sets, so a last-bit change
upstream that moves an answer by a step of a technique's grid does not fail the
scene.

Regenerate the baselines with ``python -m tests.integration.update_sim_baselines``
after a deliberate change, and review the diff before committing.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from spindoctor.sim.scene import iter_scene_paths, load_sim_scene
from tests.integration.sim_baseline import (
    CONFIDENCE_TOLERANCE,
    OFFSET_TOLERANCE_PX,
    SimBaseline,
    baseline_for_scene,
    baseline_mismatches,
    discover_sim_baseline_paths,
    load_sim_baseline,
)

# Exact-match navigation reproduction is fragile under parallel load (the same
# reason the real-image baseline test is integration-marked), so this runs in
# the deliberate integration layer rather than the fast default suite.  It needs
# no holdings -- everything is rendered and navigated in-process.
pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_BASELINES_DIR = Path(__file__).parent / 'sim_baselines'

_SCENE_BY_NAME = {p.stem: p for p in iter_scene_paths(_SCENES_ROOT)}
_BASELINE_PATHS = discover_sim_baseline_paths(_BASELINES_DIR)
_BASELINE_IDS = [p.stem for p in _BASELINE_PATHS]


def test_every_scene_has_a_baseline() -> None:
    """No catalog scene is left without a regression baseline."""
    baselined = {p.stem for p in _BASELINE_PATHS}
    assert set(_SCENE_BY_NAME) <= baselined


def test_no_orphan_baselines() -> None:
    """Every baseline corresponds to an existing catalog scene."""
    baselined = {p.stem for p in _BASELINE_PATHS}
    assert baselined <= set(_SCENE_BY_NAME)


@pytest.mark.parametrize('baseline_path', _BASELINE_PATHS, ids=_BASELINE_IDS)
def test_scene_matches_baseline(baseline_path: Path) -> None:
    """Re-navigating each scene reproduces its recorded baseline within tolerance."""
    baseline = load_sim_baseline(baseline_path)
    scene = load_sim_scene(_SCENE_BY_NAME[baseline.scene_name])
    assert baseline_mismatches(baseline, baseline_for_scene(scene)) == []


def test_baseline_json_round_trips(tmp_path: Path) -> None:
    """A baseline serialized and reloaded compares equal."""
    baseline = SimBaseline(
        scene_name='example',
        status='success',
        confidence_rank='medium',
        offset_dv_px=1.2345678901234567,
        offset_du_px=-6.789,
        confidence=0.4999999999999999,
    )
    path = tmp_path / 'example.json'
    path.write_text(baseline.to_json())
    assert load_sim_baseline(path) == baseline


def test_failed_baseline_round_trips(tmp_path: Path) -> None:
    """A failed-scene baseline (null offsets) serializes and reloads."""
    baseline = SimBaseline(
        scene_name='failed_example',
        status='failed',
        confidence_rank='failed',
        offset_dv_px=None,
        offset_du_px=None,
        confidence=0.0,
    )
    path = tmp_path / 'failed_example.json'
    path.write_text(baseline.to_json())
    assert load_sim_baseline(path) == baseline


def _success_baseline() -> SimBaseline:
    """A successful baseline the comparison tests perturb one field at a time."""
    return SimBaseline(
        scene_name='example',
        status='success',
        confidence_rank='high',
        offset_dv_px=2.5,
        offset_du_px=-1.25,
        confidence=0.8,
    )


def test_an_identical_outcome_matches() -> None:
    """An outcome equal to its baseline has no mismatches."""
    assert baseline_mismatches(_success_baseline(), _success_baseline()) == []


def test_offsets_within_tolerance_match() -> None:
    """Offsets inside the per-axis tolerance on both axes still match."""
    nudge = 0.9 * OFFSET_TOLERANCE_PX
    actual = replace(_success_baseline(), offset_dv_px=2.5 + nudge, offset_du_px=-1.25 - nudge)
    assert baseline_mismatches(_success_baseline(), actual) == []


def test_an_offset_beyond_tolerance_fails() -> None:
    """One axis outside the tolerance is reported by name."""
    actual = replace(_success_baseline(), offset_du_px=-1.25 + 1.1 * OFFSET_TOLERANCE_PX)
    assert [line.split(':')[0] for line in baseline_mismatches(_success_baseline(), actual)] == [
        'offset_du_px'
    ]


def test_a_missing_offset_fails() -> None:
    """An offset present in the baseline but absent in the outcome is reported."""
    actual = replace(_success_baseline(), offset_dv_px=None)
    assert [line.split(':')[0] for line in baseline_mismatches(_success_baseline(), actual)] == [
        'offset_dv_px'
    ]


def test_a_changed_confidence_rank_fails() -> None:
    """The confidence rank is compared exactly."""
    actual = replace(_success_baseline(), confidence_rank='medium')
    assert [line.split(':')[0] for line in baseline_mismatches(_success_baseline(), actual)] == [
        'confidence_rank'
    ]


def test_a_changed_status_fails() -> None:
    """The status is compared exactly."""
    actual = replace(_success_baseline(), status='conflicted')
    assert [line.split(':')[0] for line in baseline_mismatches(_success_baseline(), actual)] == [
        'status'
    ]


def test_a_confidence_beyond_tolerance_fails() -> None:
    """A confidence outside its tolerance is reported."""
    actual = replace(_success_baseline(), confidence=0.8 - 1.1 * CONFIDENCE_TOLERANCE)
    assert [line.split(':')[0] for line in baseline_mismatches(_success_baseline(), actual)] == [
        'confidence'
    ]


def test_a_baseline_without_a_confidence_rank_is_refused(tmp_path: Path) -> None:
    """A baseline file that lacks the confidence rank fails to load, naming the key."""
    path = tmp_path / 'old.json'
    path.write_text(
        '{"confidence": 0.5, "offset_du_px": 0.0, "offset_dv_px": 0.0, '
        '"scene_name": "old", "status": "success"}\n'
    )
    with pytest.raises(KeyError, match='confidence_rank'):
        load_sim_baseline(path)
