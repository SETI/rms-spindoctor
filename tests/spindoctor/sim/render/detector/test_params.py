"""The artifact modes a render resolves, and the truth records they leave.

A mode takes its parameters from the scene, then from what the camera's instrument
defaults set, then from the camera's shape defaults.
"""

from tests.spindoctor.sim.scenes import disc_scene

from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene


def test_an_instrument_default_mode_renders_with_the_cameras_shape() -> None:
    """A mode the instrument defaults turn on takes the camera's shape defaults."""
    dp = resolve_detector_params(disc_scene('gossi', artifacts={'instrument_defaults': True}))
    assert dp.detector_modes['banding'] == {
        'probability': 1.0,
        'amplitude_e': 65.0,
        'period_px': 42.0,
        'orientation': 'vertical',
        'freq_step_factor': 1.0,
        'dark_step_dn': 0.0,
    }


def test_a_scene_changes_one_key_of_an_instrument_default_mode() -> None:
    """A key the scene sets replaces the camera's value and leaves the mode on."""
    dp = resolve_detector_params(
        disc_scene(
            'gossi', artifacts={'instrument_defaults': True, 'banding': {'amplitude_e': 9.0}}
        )
    )
    cfg = dp.detector_modes['banding']
    assert (cfg['amplitude_e'], cfg['period_px']) == (9.0, 42.0)


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
