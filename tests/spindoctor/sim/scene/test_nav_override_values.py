"""The values inside a body's nav_override pass the checks the body's own values do.

The navigator reads an override value in place of the body's, so a value the validator
refuses on the body is refused inside the override too, with a message that gives the
override's key path.
"""

import math
import re

import pytest
from tests.spindoctor.sim.scenes import body_scene

from spindoctor.sim.scene.boundary import _BODY_IDEALIZED_KEYS
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.rules_body import _IDEALIZED_CHECKED_KEYS
from spindoctor.sim.scene.validate import validate_sim_params


@pytest.mark.parametrize('value', [math.nan, math.inf], ids=['nan', 'inf'])
def test_a_non_finite_center_is_refused(value: float) -> None:
    """A center that isn't finite is refused inside the override, as on the body."""
    scene = body_scene(body={'nav_override': {'center_v': value}})
    message = f'bodies[0].nav_override.center_v must be finite; got {value!r}'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_latitude_past_the_pole_is_refused() -> None:
    """A sub-observer latitude beyond 90 degrees is refused inside the override."""
    scene = body_scene(body={'nav_override': {'sub_observer_lat_deg': 120.0}})
    message = 'bodies[0].nav_override.sub_observer_lat_deg must lie in [-90, 90] degrees; got 120.0'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_two_entry_pose_is_refused() -> None:
    """A pose of two angles is refused inside the override, as on the body."""
    scene = body_scene(body={'nav_override': {'pose_euler_deg': [1.0, 2.0]}})
    message = 'bodies[0].nav_override.pose_euler_deg must be a list of 3 angles'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_lumpiness_that_is_not_a_number_is_refused() -> None:
    """A mesh lumpiness that isn't a number is refused inside the override."""
    scene = body_scene(body={'nav_override': {'mesh_lumpiness': 'lumpy'}})
    message = 'bodies[0].nav_override.mesh_lumpiness must be a number when present'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_an_unknown_shape_model_is_refused() -> None:
    """A shape model other than the two the renderer draws is refused inside the override."""
    scene = body_scene(body={'nav_override': {'shape_model': 'cube'}})
    message = "bodies[0].nav_override.shape_model must be 'ellipsoid' or 'polyhedral_mesh'"
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_every_idealized_key_has_a_value_check() -> None:
    """The keys whose values are checked include every key a nav_override may set."""
    assert _BODY_IDEALIZED_KEYS <= _IDEALIZED_CHECKED_KEYS


def test_a_mesh_integer_that_is_not_an_integer_is_refused() -> None:
    """A mesh latitude count that isn't an integer is refused inside the override."""
    scene = body_scene(body={'nav_override': {'mesh_n_lat': 2.5}})
    message = 'bodies[0].nav_override.mesh_n_lat must be an integer'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_body_mesh_integer_that_is_not_an_integer_is_refused() -> None:
    """The body's own mesh integers are refused with the same form of path."""
    scene = body_scene(body={'mesh_n_lon': 2.5})
    message = 'bodies[0].mesh_n_lon must be an integer'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_negative_detail_octave_count_is_refused() -> None:
    """A negative mesh detail octave count is refused inside the override."""
    scene = body_scene(body={'nav_override': {'mesh_detail_octaves': -1}})
    message = 'bodies[0].nav_override.mesh_detail_octaves must be a non-negative integer'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_name_that_is_not_a_string_is_refused() -> None:
    """A name that isn't a string is refused inside the override."""
    scene = body_scene(body={'nav_override': {'name': 7}})
    message = 'bodies[0].nav_override.name must be a string when present'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')
