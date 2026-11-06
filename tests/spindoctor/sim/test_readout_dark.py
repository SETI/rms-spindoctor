"""The readout dark: dark charge each line collects while it waits to be read.

``noise.readout_dark_e`` is the charge the last line collects, growing linearly
from 0 at the first line, whatever the exposure.  It is added before the shot
noise, scales with a binned pixel's native-pixel count when the catalog supplies
it, and a calibrated product subtracts it again, as a real pipeline's dark model
does.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.render import render_combined_model

_QUIET = {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0}


def _scene(instrument: str, noise: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A dark 128 x 64 scene on ``instrument``."""
    scene: dict[str, Any] = {
        'size_v': 128,
        'size_u': 64,
        'random_seed': 3,
        'instrument': instrument,
        'exposure_sec': 0.1,
        'noise': noise,
    }
    scene.update(extra)
    return scene


@pytest.mark.parametrize('exposure_sec', [0.01, 10.0], ids=['short', 'long'])
def test_a_raw_frame_ramps_from_zero_to_the_readout_dark(exposure_sec: float) -> None:
    """Row means rise linearly to readout_dark_e / gain at the last line, at any exposure."""
    scene = _scene('coiss_nac', {**_QUIET, 'readout_dark_e': 300.0}, exposure_sec=exposure_sec)
    img, _ = render_combined_model(scene)
    gain = resolve_detector_params(scene).gain_e_per_dn
    expected = 300.0 / gain * np.linspace(0.0, 1.0, 128)
    assert float(np.abs(img.mean(axis=1) - expected).max()) <= 0.5


def test_the_cataloged_readout_dark_scales_with_the_binned_pixel() -> None:
    """A SUM2 pixel holds four native pixels' readout dark."""
    full = resolve_detector_params(_scene('coiss_nac', {}, artifacts={'instrument_defaults': True}))
    summed = resolve_detector_params(
        _scene('coiss_nac', {}, artifacts={'instrument_defaults': True}, detector={'binning': 2})
    )
    assert summed.readout_dark_e == pytest.approx(4.0 * full.readout_dark_e)


def test_a_scene_readout_dark_states_the_frame_pixel_value() -> None:
    """A scene key is the frame pixel's own value, unscaled by the binning."""
    dp = resolve_detector_params(
        _scene('coiss_nac', {'readout_dark_e': 50.0}, detector={'binning': 2})
    )
    assert dp.readout_dark_e == 50.0


def test_the_readout_dark_is_off_without_the_physical_chain() -> None:
    """Below instrument_defaults the knob defaults to 0."""
    assert resolve_detector_params(_scene('coiss_nac', {})).readout_dark_e == 0.0


def test_a_calibrated_vidicon_frame_subtracts_no_readout_dark() -> None:
    """The vidicon chain adds no readout dark, so its calibration removes none."""
    plain, _ = render_combined_model(_scene('vgiss', {}))
    keyed, _ = render_combined_model(_scene('vgiss', {'readout_dark_e': 300.0}))
    assert np.array_equal(plain, keyed)


def test_a_calibrated_product_removes_the_readout_dark() -> None:
    """The calibration inverse subtracts the ramp, leaving a flat calibrated sky."""
    scene = _scene('coiss_calib_nac', {**_QUIET, 'readout_dark_e': 300.0})
    img, _ = render_combined_model(scene)
    row_means = img.mean(axis=1)
    # One DN at this exposure bounds the rounding of the ramp before it is removed.
    dp = resolve_detector_params(scene)
    step_if = 1.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    assert float(row_means.max() - row_means.min()) <= step_if
