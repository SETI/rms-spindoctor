"""Guard scenes that carry several detector and downlink defects on one realistic frame.

Each scene in ``sim_scenes/mixed_defects/`` is a full-size frame from one cataloged
camera with the camera's whole detector chain on and the defects a real frame of that
kind carries together, such as lossy compression, lost lines, and motion smear.  Each
combination follows real frames, which the scene comments cite.  The scenes
guard the renderer and the navigator against changes that would move a damaged frame's
answer.

:func:`~tests.integration.sim_scene_checks.scene_class_tests` builds the tests: it
navigates each scene once per run and holds that one result to every check that
applies to the scene.  Navigation renders each scene and runs the full ensemble, so the
module is ``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import pytest

from tests.integration.sim_scene_checks import scene_class_tests

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

TestMixedDefects = scene_class_tests(__file__)
