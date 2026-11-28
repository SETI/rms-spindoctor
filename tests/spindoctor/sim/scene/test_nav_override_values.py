"""The values inside a body's nav_override pass the checks the body's own values do.

The navigator reads an override value in place of the body's, so the override is typed
by the body's idealized type: a value the validator refuses on the body is refused
inside the override too, with a message that gives the override's key path.
"""

import re
from typing import get_type_hints

import pytest
from tests.spindoctor.sim.scenes import body_scene

from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.idealized import IdealizedBody
from spindoctor.sim.scene.truth_body import Body
from spindoctor.sim.scene.validate import validate_sim_params


def test_the_override_is_typed_by_the_idealized_body() -> None:
    """A nav_override holds exactly the keys, and the ranges, of the idealized body."""
    assert get_type_hints(Body)['nav_override'] == IdealizedBody | None


def test_a_latitude_past_the_pole_is_refused() -> None:
    """A sub-observer latitude beyond 90 degrees is refused inside the override."""
    scene = body_scene(body={'nav_override': {'sub_observer_lat_deg': 120.0}})
    message = (
        'bodies[0].nav_override.sub_observer_lat_deg: expected a number in [-90, 90], got 120.0'
    )
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_mesh_with_no_latitude_bands_is_refused() -> None:
    """A mesh latitude count below 2 is refused inside the override, as on the body."""
    scene = body_scene(body={'nav_override': {'shape_model': 'polyhedral_mesh', 'mesh_n_lat': 0}})
    message = 'bodies[0].nav_override.mesh_n_lat: expected an integer of at least 2, got 0'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_an_override_needs_no_size() -> None:
    """An override that leaves the semi-axes out keeps the body's, so it is accepted."""
    scene = body_scene(body={'nav_override': {'mesh_lumpiness': 0.0}})
    assert validate_sim_params(scene, source='probe') is scene
