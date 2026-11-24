"""Guard scenes for each camera the simulator catalogs, at the camera's real frame size.

Each scene in ``sim_scenes/full_frame_cameras/`` is a full-size frame from one cataloged
camera, with the camera's whole detector chain on, a realistic exposure, a pointing
error of the size real frames carry, and the camera's measured twist planted with roll
fitting off, as the cameras ship.  The scenes guard the renderer and the navigator
against changes that would move a realistic frame's answer.

:func:`~tests.integration.sim_scene_checks.scene_class_tests` builds the tests: it
navigates each scene once per run and holds that one result to every check that
applies to the scene.  Navigation renders each scene and runs the full ensemble, so the
module is ``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import pytest

from tests.integration.sim_scene_checks import scene_class_tests

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

TestFullFrameCameras = scene_class_tests(__file__)
