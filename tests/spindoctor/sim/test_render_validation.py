"""Rendering validates its scene; the scene editor's live preview does not.

A scene built in memory must meet the rules a scene file meets when it loads, so
:func:`~spindoctor.sim.render.render_combined_model` refuses one that fails
validation.  The editor's preview renders half-edited scenes on purpose, through
:func:`~spindoctor.sim.render.render_unvalidated_preview`.
"""

import re
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model, render_unvalidated_preview
from spindoctor.sim.scene import SimSceneValidationError

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
    """The preview renders a scene validation refuses, as the valid scene renders."""
    preview, _ = render_unvalidated_preview(_scene(exposure_seconds=2.0))
    expected, _ = render_combined_model(_scene(), ignore_offset=True)
    assert np.array_equal(preview, expected)
