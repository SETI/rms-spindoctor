"""Rendering validates its scene; the scene editor's live preview does not.

A scene built in memory must meet the rules a scene file meets when it loads, so
:func:`~spindoctor.sim.render.render_combined_model` refuses one that fails
validation.  The editor's preview renders half-edited scenes on purpose, through
:func:`~spindoctor.sim.render.render_unvalidated_preview`.  A pass is cached on the
scene alone, and a refusal names the label of the call that met it.
"""

import re
from typing import Any

import numpy as np
import pytest

from spindoctor.sim import render as render_module
from spindoctor.sim.render import (
    clear_render_caches,
    render_combined_model,
    render_unvalidated_preview,
    validate_scene,
)
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params

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


def test_one_scene_under_two_labels_is_validated_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene that passed under one label is not validated again under another."""
    calls: list[dict[str, Any]] = []

    def _counting(sim_params: dict[str, Any], **kwargs: Any) -> None:
        calls.append(sim_params)
        validate_sim_params(sim_params, **kwargs)

    clear_render_caches()
    monkeypatch.setattr(render_module, 'validate_sim_params', _counting)
    validate_scene(_scene(), source='first.yaml')
    validate_scene(_scene(), source='second.yaml')
    assert len(calls) == 1


def test_a_refusal_starts_with_the_callers_label() -> None:
    """A refused scene's error names the label of the call that met it."""
    with pytest.raises(
        SimSceneValidationError,
        match=re.escape("scene_a.yaml: unknown scene keys: ['exposure_seconds']"),
    ):
        validate_scene(_scene(exposure_seconds=2.0), source='scene_a.yaml')
