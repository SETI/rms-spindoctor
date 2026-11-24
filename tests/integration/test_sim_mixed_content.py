"""Guard scenes that mix kinds of content on one realistic frame.

Each scene in ``sim_scenes/mixed_content/`` puts two kinds of navigable content in one
frame, as real frames do: stars with a body, or a moon or a planet with rings.  Each
kind feeds its own techniques, and the result-combining step has to reconcile their
answers; reconciling them is that step's main job.  The scenes use the camera's real
frame size or a real binned readout, its detector chain, a pointing error of the size
real frames carry, and the camera's measured twist with roll fitting off, as the cameras
ship.  The simulator models no filter, so each exposure is sized to keep the content
within the simulator's unfiltered brightness scale, and it is not always the exposure a
real frame of the same scene would use.

A frame of this size costs far more to navigate than the small catalog scenes, so this
module navigates each scene once per run and applies every check of
:mod:`tests.integration.sim_scene_checks` to that one result: the truth tolerance or the
expected block, each single-technique band or tolerance, the sigma check, and the
regression baseline.  ``test_sim_truth_checks``, ``test_sim_expected``, and ``test_sim_baselines``
leave the class out of their navigation.

Navigation renders each scene and runs the full ensemble, so the module is
``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import pytest

from tests.integration.sim_scene_checks import scene_class_tests

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

TestMixedContent = scene_class_tests(__file__)
