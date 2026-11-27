"""Rendering validates its scene.

A scene built in memory must meet the rules a scene file meets when it loads, so
:func:`~spindoctor.sim.render.driver.render_scene` refuses one that fails
validation.  Validation is cached on the scene's JSON, and a scene file that JSON
cannot encode is still validated.
"""

import re
from pathlib import Path
from typing import Any

import pytest
from tests.spindoctor.sim.scenes import frame_scene, star_entry

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.idealized import MIN_FRAME_SIZE_PX
from spindoctor.sim.scene.io import load_sim_scene

_SIZE = 32


def _scene(**extra: Any) -> dict[str, Any]:
    """A small valid scene with one body, updated with ``extra``."""
    body = {
        'name': 'PEBBLE',
        'center_v': _SIZE / 2.0,
        'center_u': _SIZE / 2.0,
        'semi_a_px': 6.0,
        'semi_b_px': 5.0,
    }
    scene = frame_scene(size=_SIZE, bodies=[body])
    scene.update(extra)
    return scene


def test_render_refuses_an_invalid_scene() -> None:
    """A scene with a misspelled key is refused with the validation message."""
    with pytest.raises(
        SimSceneValidationError,
        match=re.escape("sim_params: unknown keys: ['exposure_seconds']"),
    ):
        render_scene(_scene(exposure_seconds=2.0))


def test_a_scene_file_json_cannot_encode_is_validated(tmp_path: Path) -> None:
    """An unquoted YAML date, which JSON cannot encode, still meets validation."""
    scene_path = tmp_path / 'dated.yaml'
    scene_path.write_text(
        'schema_version: 2\nscene_name: dated\nsize_v: 32\nsize_u: 32\nlaunch_date: 2004-07-01\n'
    )
    with pytest.raises(SimSceneValidationError, match=re.escape("unknown keys: ['launch_date']")):
        load_sim_scene(scene_path)


@pytest.mark.parametrize('key', ['size_v', 'size_u'])
def test_render_refuses_a_frame_below_the_minimum_size(key: str) -> None:
    """A frame under 10 pixels is refused with a message naming the key and the bound."""
    scene = _scene(**{key: 1})
    with pytest.raises(
        SimSceneValidationError,
        match=re.escape(f'{key}: expected an integer of at least {MIN_FRAME_SIZE_PX}, got 1'),
    ):
        render_scene(scene)


def test_render_draws_a_frame_of_the_minimum_size() -> None:
    """A frame of exactly 10 pixels renders with the camera's distortion and a star in view."""
    middle = MIN_FRAME_SIZE_PX / 2.0
    scene = frame_scene(
        instrument='vgiss_1_nac',
        size=MIN_FRAME_SIZE_PX,
        artifacts={'instrument_defaults': True},
        stars=[star_entry(middle, middle, 5.0)],
    )
    image, _ = render_scene(scene)
    assert image.shape == (MIN_FRAME_SIZE_PX, MIN_FRAME_SIZE_PX)
