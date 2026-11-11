"""Rendering validates its scene; the scene editor's live preview does not.

A scene built in memory must meet the rules a scene file meets when it loads, so
:func:`~spindoctor.sim.render.render_combined_model` refuses one that fails
validation.  The editor's preview renders half-edited scenes on purpose, through
:func:`~spindoctor.sim.render.render_unvalidated_preview`.  Validation is cached on
the scene's JSON, and a scene file that JSON cannot encode is still validated.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model, render_unvalidated_preview
from spindoctor.sim.scene import SimSceneValidationError, load_sim_scene

_SIZE = 32


def _scene(**extra: Any) -> dict[str, Any]:
    """A small valid scene with one body, updated with ``extra``."""
    scene: dict[str, Any] = {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 1,
        'bodies': [
            {
                'name': 'PEBBLE',
                'center_v': _SIZE / 2.0,
                'center_u': _SIZE / 2.0,
                'axis1': 12.0,
                'axis2': 10.0,
            }
        ],
    }
    scene.update(extra)
    return scene


def test_render_refuses_an_invalid_scene() -> None:
    """A scene with a misspelled key is refused with the validation message."""
    with pytest.raises(
        SimSceneValidationError,
        match=re.escape("sim_params: unknown scene keys: ['exposure_seconds']"),
    ):
        render_combined_model(_scene(exposure_seconds=2.0))


def test_preview_renders_an_invalid_scene() -> None:
    """The preview renders a scene that fails validation as it renders the valid one."""
    preview, _ = render_unvalidated_preview(_scene(exposure_seconds=2.0))
    expected, _ = render_combined_model(_scene(), ignore_offset=True)
    assert np.array_equal(preview, expected)


def test_a_scene_file_json_cannot_encode_is_validated(tmp_path: Path) -> None:
    """An unquoted YAML date, which JSON cannot encode, still meets validation."""
    scene_path = tmp_path / 'dated.yaml'
    scene_path.write_text(
        'schema_version: 2\nscene_name: dated\nsize_v: 32\nsize_u: 32\nlaunch_date: 2004-07-01\n'
    )
    with pytest.raises(
        SimSceneValidationError, match=re.escape("unknown scene keys: ['launch_date']")
    ):
        load_sim_scene(scene_path)
