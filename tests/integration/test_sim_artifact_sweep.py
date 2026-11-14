"""Navigation under structured telemetry loss (the artifact stress axis).

Each scene under ``sim_scenes/artifact_sweep/`` plants a modest structured
telemetry loss (a few whole missing lines plus a few truncated lines per frame)
on top of a clean planted-offset frame, in a uniform-placement and an
adversarial-placement variant.  These tests navigate each scene and assert the
navigator still reaches success.  This is the survivable end of the artifact axis, the
level at which realism does not break navigation.  ``test_sim_truth_checks`` holds the
recovered offset to each scene's measured truth tolerance.  The scene's seed also places
the lost lines, so those tolerances were measured with the lost lines where the
committed seed puts them and only the noise draw varying.  Everything is in-process
(no holdings or SPICE).
"""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SWEEP_DIR = Path(__file__).parent / 'sim_scenes' / 'artifact_sweep'
_SCENE_PATHS = [p for p in iter_scene_paths(_SWEEP_DIR.parent) if p.parent.name == 'artifact_sweep']
_SCENE_IDS = [p.stem for p in _SCENE_PATHS]


def _navigate(scene: dict[str, Any]) -> Any:
    obs = ObsSim.from_file('/tmp/artifact_sweep.yaml', sim_params=scene)
    orchestrator = NavOrchestrator(build_models_for_obs(obs), only_models='*', only_techniques='*')
    return orchestrator.navigate(obs)


def test_there_are_artifact_sweep_scenes() -> None:
    """The artifact_sweep class is populated."""
    assert _SCENE_PATHS


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_artifact_scene_navigates(path: Path) -> None:
    """Each structured-loss scene still navigates to a success status."""
    result = _navigate(load_sim_scene(path))
    assert result.status == 'success'
