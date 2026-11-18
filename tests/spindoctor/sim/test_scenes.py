"""The shared base scenes validate, and no scene shares a mapping with its inputs."""

import copy
from collections.abc import Callable
from typing import Any

import pytest
from tests.spindoctor.sim.scenes import (
    body_scene,
    disc_body,
    frame_scene,
    quiet_noise,
    ring_system_block,
    star_entry,
)

from spindoctor.sim.scene.validate import validate_sim_params

_RINGLET = {
    'name': 'RINGLET',
    'kind': 'ringlet',
    'tau': 1.0,
    'width_px': 4.0,
    'orbit': {'a_px': 20.0},
}

# Each block helper inside the frame it is used in.
_BLOCK_SCENES: dict[str, Callable[[], dict[str, Any]]] = {
    'named_quiet': lambda: frame_scene(scene_name='probe', noise=quiet_noise()),
    'disc_body': lambda: frame_scene(bodies=[disc_body(20.0, 40.0, 6.0, range_km=1.0e6)]),
    'star_entry': lambda: frame_scene(stars=[star_entry(32.5, 32.5, 6.0, psf_sigma=1.5)]),
    'ring_system_block': lambda: frame_scene(
        ring_system=ring_system_block(32.0, 32.0, [_RINGLET], opening_deg_obs=30.0)
    ),
}


@pytest.mark.parametrize('fixture_name', ['empty_frame', 'lit_body_frame'])
def test_each_fixture_scene_validates(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """The fixture scenes pass scene validation, which returns the scene itself."""
    scene = request.getfixturevalue(fixture_name)
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize('name', sorted(_BLOCK_SCENES))
def test_each_block_validates_in_a_frame(name: str) -> None:
    """A frame carrying each shared block passes scene validation."""
    scene = _BLOCK_SCENES[name]()
    assert validate_sim_params(scene) is scene


# Each builder, handed a nested mapping through each argument that copies a value in.
_ISOLATION_CASES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    'frame_scene': lambda given: frame_scene(extra=given),
    'body_scene': lambda given: body_scene(body={'extra': given}),
    'disc_body': lambda given: disc_body(0.0, 0.0, 1.0, extra=given),
    'star_entry': lambda given: star_entry(0.0, 0.0, 1.0, extra=given),
    'ring_system_block_features': lambda given: ring_system_block(0.0, 0.0, [{'orbit': given}]),
    'ring_system_block_keys': lambda given: ring_system_block(0.0, 0.0, [], extra=given),
}


@pytest.mark.parametrize('name', sorted(_ISOLATION_CASES))
def test_a_built_mapping_keeps_no_reference_to_its_inputs(name: str) -> None:
    """Changing a nested input after building leaves the built mapping as it was built."""
    given: dict[str, Any] = {'inner': {'value': 1.0}}
    built = _ISOLATION_CASES[name](given)
    expected = copy.deepcopy(built)
    given['inner']['value'] = 2.0
    assert built == expected
