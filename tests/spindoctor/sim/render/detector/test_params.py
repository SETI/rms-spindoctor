"""The detector values and artifact modes a render resolves, and the truth records.

A detector value comes from the scene's ``detector`` block, then from the instrument
entry, and never from the navigator's configuration.  A mode takes its parameters from
the scene, then from what the camera's instrument defaults set, then from the camera's
shape defaults.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene, frame_scene

from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instruments.resolvers import resolve_calibrated_sky_if
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params


def test_a_scene_ceiling_clips_the_image() -> None:
    """The scene's detector.saturation_dn is the largest DN in the frame."""
    img, _ = render_scene(disc_scene('coiss_nac', detector={'saturation_dn': 1000.0}))
    assert float(img.max()) == 1000.0


def test_a_scene_ceiling_stays_out_of_the_navigators_settings() -> None:
    """The navigator keeps its own ceiling on a frame that clips lower."""
    scene = disc_scene('coiss_nac', detector={'saturation_dn': 1000.0})
    obs = ObsSim.from_file('/tmp/sim_planted_ceiling.json', sim_params=scene)
    assert instrument_settings_from_obs(obs).saturation_dn == 4095.0


def test_the_navigators_ceiling_leaves_the_image_alone() -> None:
    """Overriding the ceiling the navigator believes changes no pixel."""
    scene = disc_scene('coiss_nac')
    believed = {**scene, 'instrument_config': {'noise': {'saturation_dn': 1000.0}}}
    expected, _ = render_scene(scene)
    img, _ = render_scene(believed)
    np.testing.assert_array_equal(img, expected)


def test_the_calibration_keeps_delivered_pixels_off_the_navigators_marker() -> None:
    """A dark sky that lands on the navigator's marker moves off it.

    The noise-free dark sky of a calibrated Cassini frame reads exactly the calibrated
    sky residual, so a scene that makes the residual the navigator's marker puts every
    sky pixel on it before the calibration moves them.
    """
    sky = resolve_calibrated_sky_if('coiss_calib_nac')
    scene = frame_scene(
        instrument='coiss_calib_nac', instrument_config={'noise': {'marker_value': sky}}
    )
    img, _ = render_scene(scene)
    assert int((img == sky).sum()) == 0


def test_a_scene_full_well_replaces_the_ccd_well_through_the_gain() -> None:
    """On the CCD chain a full well of 3000 DN is 3000 gains of electrons."""
    dp = resolve_detector_params(disc_scene('coiss_nac', detector={'full_well_dn': 3000.0}))
    assert dp.full_well_e == 3000.0 * dp.gain_e_per_dn


def test_a_scene_full_well_replaces_the_vidicon_full_scale() -> None:
    """On the vidicon chain the full well is the DN full scale."""
    dp = resolve_detector_params(disc_scene('vgiss_1_nac', detector={'full_well_dn': 200.0}))
    assert dp.full_well_dn == 200.0


def test_the_loss_modes_write_no_value_above_a_scene_ceiling() -> None:
    """A garbled line's wrong values stay within the scene's own ceiling."""
    scene = disc_scene(
        'gossi',
        detector={'saturation_dn': 100.0},
        artifacts={'line_garble': {'expected_events': 20.0}},
    )
    img, _ = render_scene(scene)
    assert float(img.max()) <= 100.0


@pytest.mark.parametrize('key', ['bias_dn', 'signal_full_scale_frac', 'missing_data_rate'])
def test_a_detector_value_in_the_noise_block_is_refused(key: str) -> None:
    """The noise block holds only noise, so a detector value there is an unknown key."""
    scene = disc_scene('coiss_nac', noise={key: 0.1})
    with pytest.raises(SimSceneValidationError, match=rf"noise: unknown keys: \['{key}'\]"):
        validate_sim_params(scene)


@pytest.mark.parametrize('key', ['saturation_dn', 'full_well_dn'])
def test_a_zero_ceiling_or_well_is_refused(key: str) -> None:
    """The ADC ceiling and the full well must be positive."""
    scene = disc_scene('coiss_nac', detector={key: 0.0})
    with pytest.raises(SimSceneValidationError, match=rf'detector\.{key} must be a positive'):
        validate_sim_params(scene)


def test_a_full_well_override_of_the_navigator_block_is_refused() -> None:
    """The navigator reads no full well, so a scene can't set one in its block."""
    scene = disc_scene('coiss_nac', instrument_config={'noise': {'full_well_dn': 3000.0}})
    message = r"instrument_config\.noise: unknown keys: \['full_well_dn'\]"
    with pytest.raises(SimSceneValidationError, match=message):
        validate_sim_params(scene)


def test_an_instrument_default_mode_renders_with_the_cameras_shape() -> None:
    """A mode the instrument defaults turn on takes the camera's shape defaults."""
    dp = resolve_detector_params(disc_scene('gossi', artifacts={'instrument_defaults': True}))
    assert dp.detector_modes['banding'] == {
        'probability': 1.0,
        'amplitude': 65.0,
        'period_px': 42.0,
        'orientation': 'vertical',
        'horizontal_pattern': 'periodic',
        'freq_step_factor': 1.0,
        'dark_step_dn': 0.0,
        'ringing_amplitude': 0.0,
        'ringing_period_px': 0.0,
        'ringing_decay_px': 0.0,
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
        'hot_pixels': {'pixel_fraction': 7.2e-3},
        'banding': {'orientation': 'horizontal'},
        'readout_dark': {'amplitude': 33.0},
        'bias_structure': {},
        'bloom': {},
    },
    'coiss_calib_nac': {
        'hot_pixels': {'pixel_fraction': 3.3e-5, 'readout_e': 1000.0},
        'readout_dark': {'amplitude': 55.0},
        'bloom': {},
    },
    'gossi': {
        'hot_pixels': {
            'pixel_fraction': 9.0e-5,
            'rate_e_per_sec': 0.0,
            'readout_e': 150.0,
            'streak_e': 0.75,
        },
        'banding': {'orientation': 'vertical'},
        'bias_structure': {},
        'bloom': {'bloom_length': 6},
    },
    'nhlorri': {
        'frame_transfer_smear': {},
        'banding': {
            'amplitude': 21.0,
            'orientation': 'horizontal',
            'horizontal_pattern': 'random',
            'ringing_amplitude': 15.8,
            'ringing_period_px': 103.0,
            'ringing_decay_px': 71.0,
        },
        'bias_structure': {'pedestal_sigma_dn': 1.0, 'row_gradient_dn': 0.5},
    },
    'vgiss_1_nac': {
        'banding': {'amplitude': 0.25, 'orientation': 'vertical'},
    },
    'vgiss_1_wac': {
        'banding': {'amplitude': 0.25, 'orientation': 'vertical'},
    },
    'vgiss_2_nac': {
        'banding': {'amplitude': 0.25, 'orientation': 'vertical'},
    },
    'vgiss_2_wac': {
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
