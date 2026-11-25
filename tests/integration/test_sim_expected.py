"""Expected-outcome regression for sim scenes carrying an ``expected`` block.

Discovers every catalog scene with a scene-level ``expected`` block, renders and
navigates it in-process, and asserts the fused outcome matches the declaration
through the :mod:`tests.integration.sim_expected` machinery.  Two scene kinds
live here: the false-positive-characterization scenes (a wrong-catalog or
overwhelming-clutter frame whose CORRECT outcome is failure / low confidence,
asserted so a confident wrong offset is a test failure) and the known-error
checks (scenes whose confidently wrong offset is a documented, currently
unmitigated hazard: ``known_offset_error_px`` bands the measured error so a
worsening regression fails, and a genuine fix fails loudly so that someone
re-measures the known error).

The classes in :data:`~tests.integration.sim_truth.OWN_MODULE_CLASSES` are left
out: their own modules navigate each scene once and assert its expected block on
that result.

Navigation is heavier than the structural checks (each scene renders and runs
the full ensemble), so the module is ``@pytest.mark.integration`` -- the same
deliberate tier as the baselines and sweeps.  It needs no holdings or SPICE.
"""

from pathlib import Path

import pytest

from spindoctor.sim.scene.io import load_sim_scene
from tests.integration.sim_expected import iter_expected_scene_paths, navigate_scene
from tests.integration.sim_scene_checks import SCENES_ROOT, assert_expected_outcome, has_own_module

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_EXPECTED_PATHS = [p for p in iter_expected_scene_paths(SCENES_ROOT) if not has_own_module(p)]
_EXPECTED_IDS = [f'{p.parent.name}/{p.stem}' for p in _EXPECTED_PATHS]


def test_expected_scenes_exist() -> None:
    """At least one catalog scene declares an expected outcome."""
    assert _EXPECTED_PATHS


@pytest.mark.parametrize('path', _EXPECTED_PATHS, ids=_EXPECTED_IDS)
def test_scene_matches_expected_outcome(path: Path) -> None:
    """Each scene with an expected block navigates to its declared outcome."""
    scene = load_sim_scene(path)
    assert_expected_outcome(scene, navigate_scene(scene))
