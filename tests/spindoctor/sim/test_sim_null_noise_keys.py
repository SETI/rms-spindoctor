"""A null scene ``noise`` key reads as absent, to the renderer as to the validator.

The validator accepts null for every ``noise`` and ``noise.vidicon`` key, and a
null ``detector`` key already means the instrument's default.  The detector
resolver and the telemetry stage read a null noise key the same way: it resolves
exactly as the scene without the key, under the clean floor and under
``artifacts.instrument_defaults`` alike, so ``noise: {poisson: null}`` keeps the
opt-in's shot noise rather than turning it off.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_checks import _NOISE_KEYS, _VIDICON_NOISE_KEYS

_NOISE_VALUE_KEYS = sorted(_NOISE_KEYS - {'vidicon'})
_INSTRUMENTS = ('coiss_nac', 'vgiss', 'coiss_calib_nac', 'generic')


def _scene(instrument: str, *, instrument_defaults: bool, **extra: Any) -> dict[str, Any]:
    """A small disc scene on ``instrument``, with the physical chain on or off."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'null_noise',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 7,
        'instrument': instrument,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 16.0,
                'center_u': 16.0,
                'axis1': 8.0,
                'axis2': 8.0,
                'axis3': 8.0,
                'illumination_angle': 0.0,
                'phase_angle': 0.0,
            }
        ],
    }
    if instrument_defaults:
        scene['artifacts'] = {'instrument_defaults': True}
    scene.update(extra)
    return scene


@pytest.mark.parametrize('instrument_defaults', [False, True], ids=['floor', 'defaults'])
@pytest.mark.parametrize('instrument', _INSTRUMENTS)
@pytest.mark.parametrize('key', _NOISE_VALUE_KEYS)
def test_null_noise_key_resolves_as_absent(
    key: str, instrument: str, instrument_defaults: bool
) -> None:
    """A validated null noise key gives the detector view of a scene without the key."""
    scene = _scene(instrument, instrument_defaults=instrument_defaults, noise={key: None})
    validate_sim_params(scene, source='null')
    absent = _scene(instrument, instrument_defaults=instrument_defaults)
    assert resolve_detector_params(scene) == resolve_detector_params(absent)


@pytest.mark.parametrize('key', sorted(_VIDICON_NOISE_KEYS))
def test_null_vidicon_noise_key_resolves_as_absent(key: str) -> None:
    """A validated null noise.vidicon key keeps the catalog's vidicon noise."""
    scene = _scene('vgiss', instrument_defaults=True, noise={'vidicon': {key: None}})
    validate_sim_params(scene, source='null')
    absent = _scene('vgiss', instrument_defaults=True)
    assert resolve_detector_params(scene) == resolve_detector_params(absent)


def test_null_poisson_keeps_shot_noise_under_instrument_defaults() -> None:
    """noise.poisson: null under instrument_defaults leaves shot noise on."""
    scene = _scene('coiss_nac', instrument_defaults=True, noise={'poisson': None})
    assert resolve_detector_params(scene).poisson is True


def test_null_missing_data_rate_renders_as_absent() -> None:
    """A null missing_data_rate renders the frame a scene without the key renders."""
    scene = _scene('coiss_nac', instrument_defaults=False, noise={'missing_data_rate': None})
    validate_sim_params(scene, source='null')
    image, _ = render_combined_model(scene)
    expected, _ = render_combined_model(_scene('coiss_nac', instrument_defaults=False))
    np.testing.assert_array_equal(image, expected)
