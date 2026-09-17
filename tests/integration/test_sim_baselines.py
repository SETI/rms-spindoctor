"""Deterministic regression baselines for the sim scene catalog (Phase T2).

Each catalog scene has a baseline JSON recording the exact rounded navigation
outcome.  These tests re-render and re-navigate every scene and assert the
outcome still matches its baseline, turning "this scene navigates to X" into a
tripwire.  Navigation is deterministic, so the match is exact-equal on rounded
values.  Everything is in-process, so this runs in the default suite.

Regenerate the baselines with ``python -m tests.integration.update_sim_baselines``
after a deliberate change, and review the diff before committing.
"""

from pathlib import Path

import pytest

from spindoctor.sim.scene import iter_scene_paths, load_sim_scene
from tests.integration.sim_baseline import (
    SimBaseline,
    baseline_for_scene,
    discover_sim_baseline_paths,
    load_sim_baseline,
)

# Exact-match navigation reproduction is fragile under parallel load (the same
# reason the real-image baseline test is integration-marked), so this runs in
# the deliberate integration layer rather than the fast default suite.  It needs
# no holdings -- everything is rendered and navigated in-process.
pytestmark = pytest.mark.integration

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
    """Re-navigating each scene reproduces its recorded baseline exactly."""
    baseline = load_sim_baseline(baseline_path)
    scene = load_sim_scene(_SCENE_BY_NAME[baseline.scene_name])
    assert baseline_for_scene(scene) == baseline


def test_baseline_json_round_trips(tmp_path: Path) -> None:
    """A baseline serialized and reloaded compares equal."""
    baseline = SimBaseline(
        scene_name='example',
        status='success',
        offset_dv_px=1.2345,
        offset_du_px=-6.789,
        confidence=0.5,
    )
    path = tmp_path / 'example.json'
    path.write_text(baseline.to_json())
    assert load_sim_baseline(path) == baseline


def test_failed_baseline_round_trips(tmp_path: Path) -> None:
    """A failed-scene baseline (null offsets) serializes and reloads."""
    baseline = SimBaseline(
        scene_name='failed_example',
        status='failed',
        offset_dv_px=None,
        offset_du_px=None,
        confidence=0.0,
    )
    path = tmp_path / 'failed_example.json'
    path.write_text(baseline.to_json())
    assert load_sim_baseline(path) == baseline
