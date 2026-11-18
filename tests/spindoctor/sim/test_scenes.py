"""The shared base scenes validate, and no scene shares a mapping with its inputs."""

from collections.abc import Callable
from typing import Any

import pytest
from tests.spindoctor.sim.scenes import (
    disc_body,
    frame_scene,
    quiet_noise,
    ring_system_block,
    star_entry,
)

from spindoctor.sim.scene import validate_sim_params

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
    """The fixture scenes pass scene validation unchanged."""
    scene = request.getfixturevalue(fixture_name)
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize('name', sorted(_BLOCK_SCENES))
def test_each_block_validates_in_a_frame(name: str) -> None:
    """A frame carrying each shared block passes scene validation."""
    scene = _BLOCK_SCENES[name]()
    assert validate_sim_params(scene) is scene


def test_a_scene_keeps_no_reference_to_a_block_it_was_given() -> None:
    """Changing a block after building the scene leaves the scene as it was built."""
    noise = quiet_noise()
    scene = frame_scene(noise=noise)
    noise['poisson'] = True
    assert scene['noise']['poisson'] is False
