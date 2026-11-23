"""The artifact modes a render resolves, and the truth records they leave.

A mode takes its parameters from the scene, then from what the camera's instrument
defaults set, then from the camera's shape defaults.
"""

from typing import Any

import pytest
from tests.spindoctor.sim.scenes import disc_scene

from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene


def test_an_instrument_default_mode_renders_with_the_cameras_shape() -> None:
    """A mode the instrument defaults turn on takes the camera's shape defaults."""
    dp = resolve_detector_params(disc_scene('gossi', artifacts={'instrument_defaults': True}))
    assert dp.detector_modes['banding'] == {
        'probability': 1.0,
        'amplitude': 65.0,
        'period_px': 42.0,
        'orientation': 'vertical',
        'freq_step_factor': 1.0,
        'dark_step_dn': 0.0,
    }


def test_a_scene_changes_one_key_of_an_instrument_default_mode() -> None:
    """A key the scene sets replaces the camera's value and leaves the mode on."""
    dp = resolve_detector_params(
        disc_scene('gossi', artifacts={'instrument_defaults': True, 'banding': {'amplitude': 9.0}})
    )
    cfg = dp.detector_modes['banding']
    assert (cfg['amplitude'], cfg['period_px']) == (9.0, 42.0)


def test_a_scene_turns_an_instrument_default_mode_off() -> None:
    """An activation key of 0 turns off a mode the instrument defaults turn on."""
    dp = resolve_detector_params(
        disc_scene(
            'gossi', artifacts={'instrument_defaults': True, 'banding': {'probability': 0.0}}
        )
    )
    assert 'banding' not in dp.detector_modes


def test_detector_mode_records_truth() -> None:
    """A rendered detector mode records its realized geometry in the frame truth."""
    _img, truth = render_scene(
        disc_scene(
            'coiss_nac', artifacts={'fixed_pattern': {'probability': 1.0, 'dust_donut_count': 3}}
        )
    )
    artifacts = truth.get('artifacts', {})
    assert artifacts.get('fixed_pattern', {}).get('active') is True


def test_floor_scene_records_no_detector_artifacts() -> None:
    """A scene with no artifact modes records no detector-artifact truth."""
    _img, truth = render_scene(disc_scene('coiss_nac'))
    assert 'artifacts' not in truth or not truth['artifacts']


# Each camera's modes under instrument_defaults, with the keys that set its character:
# the mode's activation key where the camera states it, and the shape a review checks.
_DEFAULT_MODES: dict[str, dict[str, dict[str, Any]]] = {
    'coiss_wac': {
        'hot_pixels': {'pixel_fraction': 2.0e-3},
        'banding': {'orientation': 'horizontal'},
        'readout_dark': {'amplitude': 33.0},
        'bias_structure': {},
        'bloom': {},
    },
    'coiss_calib_nac': {
        'hot_pixels': {'pixel_fraction': 1.6e-5},
        'readout_dark': {'amplitude': 55.0},
        'bloom': {},
    },
    'gossi': {
        'hot_pixels': {'pixel_fraction': 1.4e-6, 'amplitude_e': 4.0e4, 'column_factor': 0.5},
        'banding': {'orientation': 'vertical'},
        'bias_structure': {},
        'bloom': {'bloom_length': 6},
    },
    'nhlorri': {
        'frame_transfer_smear': {},
        'banding': {'orientation': 'horizontal'},
        'bias_structure': {'pedestal_sigma_dn': 1.0, 'row_gradient_dn': 0.5},
    },
    'vgiss': {
        'banding': {'amplitude': 0.25, 'orientation': 'vertical'},
    },
}


@pytest.mark.parametrize('instrument', sorted(_DEFAULT_MODES))
def test_each_camera_turns_on_its_own_modes(instrument: str) -> None:
    """The instrument defaults turn on exactly the camera's modes, at the camera's values."""
    expected = _DEFAULT_MODES[instrument]
    dp = resolve_detector_params(disc_scene(instrument, artifacts={'instrument_defaults': True}))
    shown = {
        name: {key: cfg[key] for key in expected.get(name, {})}
        for name, cfg in dp.detector_modes.items()
    }
    assert shown == expected
