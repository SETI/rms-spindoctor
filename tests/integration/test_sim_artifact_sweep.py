"""Navigation under structured telemetry loss (the artifact stress axis).

Each scene under ``sim_scenes/artifact_sweep/`` plants a modest structured
telemetry loss (a few whole missing lines plus a few truncated lines per frame)
on top of a clean planted-offset frame, in a uniform-placement and an
adversarial-placement variant.  These tests navigate each scene and assert the
navigator still reaches success and recovers the planted offset within tolerance
-- the survivable end of the artifact axis, the level at which realism does not
break navigation.  Everything is in-process (no holdings or SPICE), so these run
alongside the other planted-offset invariants.

Unlike the algorithmic invariants these are not correct-by-construction (loss is
stochastic per seed), so they are pinned scenes with committed baselines; the
tolerance here is the survivability claim, not an exact recovery.
"""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene

pytestmark = pytest.mark.integration

_SWEEP_DIR = Path(__file__).parent / 'sim_scenes' / 'artifact_sweep'
_SCENE_PATHS = [p for p in iter_scene_paths(_SWEEP_DIR.parent) if p.parent.name == 'artifact_sweep']
_SCENE_IDS = [p.stem for p in _SCENE_PATHS]

# The loss is survivable at this expected count, but a lost line that crosses a
# feature moves the recovered offset a few tenths of a pixel, so the tolerance is
# looser than the clean-frame invariant bound.
_OFFSET_TOLERANCE_PX = 0.5


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


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_artifact_scene_recovers_planted_v(path: Path) -> None:
    """Each structured-loss scene recovers its planted v offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_artifact_scene_recovers_planted_u(path: Path) -> None:
    """Each structured-loss scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX
