"""Expected-outcome regression for sim scenes carrying an ``expected`` block.

Discovers every catalog scene with a scene-level ``expected`` block, renders and
navigates it in-process, and asserts the fused outcome matches the declaration
through the :mod:`tests.integration.sim_expected` machinery.  Two scene kinds
live here: the false-positive-characterization scenes (a wrong-catalog or
overwhelming-clutter frame whose CORRECT outcome is failure / low confidence,
asserted so a confident wrong offset is a test failure) and the honest pins
(scenes whose confidently wrong offset is a documented, currently unmitigated
hazard: ``known_offset_error_px`` bands the measured error so a worsening
regression fails and a genuine fix fails loudly for a deliberate re-pin).

Navigation is heavier than the structural checks (each scene renders and runs
the full ensemble), so the module is ``@pytest.mark.integration`` -- the same
deliberate tier as the baselines and sweeps.  It needs no holdings or SPICE.
"""

from pathlib import Path

import pytest

from spindoctor.sim.scene import load_sim_scene
from tests.integration.sim_expected import (
    assert_result_matches_expected,
    expected_from_scene,
    iter_expected_scene_paths,
    navigate_scene,
)

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_EXPECTED_PATHS = iter_expected_scene_paths(_SCENES_ROOT)
_EXPECTED_IDS = [f'{p.parent.name}/{p.stem}' for p in _EXPECTED_PATHS]


def test_expected_scenes_exist() -> None:
    """At least one catalog scene declares an expected outcome."""
    assert _EXPECTED_PATHS


@pytest.mark.parametrize('path', _EXPECTED_PATHS, ids=_EXPECTED_IDS)
def test_scene_matches_expected_outcome(path: Path) -> None:
    """Each scene with an expected block navigates to its declared outcome."""
    scene = load_sim_scene(path)
    expected = expected_from_scene(scene)
    assert expected is not None
    result = navigate_scene(scene)
    planted = (float(scene.get('offset_v', 0.0)), float(scene.get('offset_u', 0.0)))
    assert_result_matches_expected(
        scene_name=scene['scene_name'], expected=expected, result=result, planted_offset_vu=planted
    )
