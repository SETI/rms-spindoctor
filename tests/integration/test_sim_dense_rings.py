"""Guard scenes of Saturn's rings at real scale, on full-size narrow-angle frames.

Each scene in ``sim_scenes/dense_rings/`` draws a stretch of the main rings with the
cataloged edges at their measured radii and structure the navigator does not know about
beside them: a dense radial profile measured on a navigated frame, the F ring and the
Encke gap ringlet, a ringlet whose drawn orbit is eccentric while the navigator's orbit
is circular, or a ring sheet that saturates the detector.  One scene views the rings at
a low opening angle.  Every scene has the camera's whole detector chain on, a realistic
exposure, a pointing error of the size real frames carry, and the camera's measured
twist planted with roll fitting off, as the camera ships.  The scenes guard the renderer
and the navigator against changes that would move these answers, and the ones the
navigator gets wrong hold that answer in a band tied to the issue that explains it.

:func:`~tests.integration.sim_scene_checks.scene_class_tests` builds the tests: it
navigates each scene once per run and holds that one result to every check that
applies to the scene.  Navigation renders each scene and runs the full ensemble, so the
module is ``@pytest.mark.integration``.  It needs no holdings or SPICE.
"""

import pytest

from tests.integration.sim_scene_checks import scene_class_tests

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

TestDenseRings = scene_class_tests(__file__)
