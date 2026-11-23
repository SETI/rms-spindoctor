"""The readout dark: dark charge each line collects while it waits to be read.

The ``readout_dark`` mode's ``amplitude_e`` is the charge a native pixel of the last
line collects, growing from 0 at the first line, whatever the exposure.  It is added
before the shot noise, a binned pixel sums its native pixels' share, and a calibrated
CCD product subtracts it again, as a real pipeline's dark model does.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise

from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene


def _scene(instrument: str, readout_dark: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A dark, quiet 128 x 64 scene on ``instrument`` with the mode always on."""
    scene = frame_scene(
        instrument=instrument,
        size=(128, 64),
        seed=3,
        exposure_sec=0.1,
        noise=quiet_noise(),
        artifacts={'readout_dark': {'probability': 1.0, **readout_dark}},
    )
    scene.update(extra)
    return scene


@pytest.mark.parametrize('exposure_sec', [0.01, 10.0], ids=['short', 'long'])
def test_a_raw_frame_ramps_from_zero_to_the_readout_dark(exposure_sec: float) -> None:
    """Row means rise linearly to amplitude_e / gain at the last line, at any exposure."""
    scene = _scene('coiss_nac', {'amplitude_e': 300.0}, exposure_sec=exposure_sec)
    img, _ = render_scene(scene)
    gain = resolve_detector_params(scene).gain_e_per_dn
    expected = 300.0 / gain * np.linspace(0.0, 1.0, 128)
    assert float(np.abs(img.mean(axis=1) - expected).max()) <= 0.5


def test_a_binned_pixel_sums_its_native_pixels_readout_dark() -> None:
    """A SUM2 pixel holds four native pixels' readout dark."""
    _img, truth = render_scene(_scene('coiss_nac', {'amplitude_e': 50.0}, detector={'binning': 2}))
    assert truth['artifacts']['readout_dark']['peak_e'] == pytest.approx(200.0)


def test_the_instrument_defaults_supply_the_measured_readout_dark() -> None:
    """Under instrument_defaults the Cassini NAC shows its measured 55 electrons."""
    dp = resolve_detector_params(
        frame_scene(instrument='coiss_nac', artifacts={'instrument_defaults': True})
    )
    assert dp.detector_modes['readout_dark']['amplitude_e'] == 55.0


def test_a_calibrated_vidicon_frame_keeps_its_readout_dark() -> None:
    """The vidicon's readout dark is what its calibrated product keeps."""
    scene = _scene('vgiss', {'amplitude_e': 40.0}, detector={'quantization': 'exact'})
    img, _ = render_scene(scene)
    dp = resolve_detector_params(scene)
    rise_if = 40.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    assert float(img[-1].mean() - img[0].mean()) == pytest.approx(rise_if, rel=0.02)


def test_a_calibrated_product_removes_the_readout_dark() -> None:
    """The calibration inverse subtracts the ramp, leaving a flat calibrated sky."""
    scene = _scene('coiss_calib_nac', {'amplitude_e': 300.0})
    img, _ = render_scene(scene)
    row_means = img.mean(axis=1)
    # One DN at this exposure bounds the rounding of the ramp before it is removed.
    dp = resolve_detector_params(scene)
    step_if = 1.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    assert float(row_means.max() - row_means.min()) <= step_if


def test_a_calibrated_product_keeps_the_enhanced_columns() -> None:
    """The pipeline's dark model is per line, so the enhanced columns stay in the product."""
    scene = _scene('coiss_calib_nac', {'amplitude_e': 600.0, 'rbi_column_factor': 1.0})
    img, truth = render_scene(scene)
    enhanced = truth['artifacts']['readout_dark']['enhanced_columns']
    plain = [u for u in range(64) if u not in enhanced]
    dp = resolve_detector_params(scene)
    step_if = 1.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    excess = float(img[-1, enhanced].mean() - img[-1, plain].mean())
    assert excess == pytest.approx(600.0 / dp.gain_e_per_dn * step_if, rel=0.05)
