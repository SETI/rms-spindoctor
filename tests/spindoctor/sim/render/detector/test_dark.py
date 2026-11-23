"""The dark signal: the dark current and the readout dark.

The dark current builds up over the exposure, the same on every line.  The
``readout_dark`` mode's ``amplitude`` is the charge a native pixel of the last line
collects, growing from 0 at the first line, whatever the exposure.  It is added before
the shot noise, a binned pixel sums its native pixels' share, and a calibrated CCD
product subtracts it again, as a real pipeline's dark model does.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise

from spindoctor.sim.render.detector.dark import (
    add_dark_current,
    add_readout_dark,
    readout_dark_profile,
)
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene

_SIZE = 64


def test_dark_current_adds_uniform_pedestal() -> None:
    """Dark current adds rate * exposure electrons uniformly."""
    electrons = np.zeros((8, 8), dtype=np.float64)
    add_dark_current(electrons, rate_e_per_sec=5.0, exposure_sec=2.0)
    assert np.all(electrons == 10.0)


def test_dark_current_disabled_at_zero_rate() -> None:
    """A zero dark-current rate is a no-op."""
    electrons = np.zeros((8, 8), dtype=np.float64)
    add_dark_current(electrons, rate_e_per_sec=0.0, exposure_sec=2.0)
    assert np.all(electrons == 0.0)


def test_readout_dark_grows_with_line_number() -> None:
    """The profile is zero at the first line and the amplitude at the last."""
    profile = readout_dark_profile(_SIZE, amplitude=100.0, nonlinear=1.0, binning=1)
    assert profile[0] == 0.0
    assert profile[-1] == pytest.approx(100.0)
    assert profile[_SIZE // 2] == pytest.approx(100.0 * (_SIZE // 2) / (_SIZE - 1))


def test_a_nonlinear_readout_dark_grows_late() -> None:
    """An exponent above 1 holds the profile below the linear one between the ends."""
    linear = readout_dark_profile(_SIZE, amplitude=100.0, nonlinear=1.0, binning=1)
    bent = readout_dark_profile(_SIZE, amplitude=100.0, nonlinear=2.0, binning=1)
    assert np.all(bent[1:-1] < linear[1:-1])


def test_readout_dark_enhances_the_rbi_columns() -> None:
    """The RBI columns get the extra fraction of the profile, and the rest don't."""
    plane = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_readout_dark(
        plane,
        readout_dark_profile(_SIZE, amplitude=100.0, nonlinear=1.0, binning=1),
        rbi_column_factor=0.5,
        rbi_columns=np.array([10, 20], dtype=np.int64),
    )
    assert float(plane[-1, 10]) == pytest.approx(150.0)
    assert float(plane[-1, 5]) == pytest.approx(100.0)


def test_readout_dark_disabled_at_zero_amplitude() -> None:
    """A zero amplitude is a no-op."""
    plane = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_readout_dark(
        plane,
        readout_dark_profile(_SIZE, amplitude=0.0, nonlinear=1.0, binning=1),
        rbi_column_factor=0.5,
        rbi_columns=np.array([10], dtype=np.int64),
    )
    assert record['active'] is False
    assert np.all(plane == 0.0)


def _readout_scene(instrument: str, readout_dark: dict[str, Any], **extra: Any) -> dict[str, Any]:
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
    """Row means rise linearly to amplitude / gain at the last line, at any exposure."""
    scene = _readout_scene('coiss_nac', {'amplitude': 300.0}, exposure_sec=exposure_sec)
    img, _ = render_scene(scene)
    gain = resolve_detector_params(scene).gain_e_per_dn
    expected = 300.0 / gain * np.linspace(0.0, 1.0, 128)
    assert float(np.abs(img.mean(axis=1) - expected).max()) <= 0.5


def test_a_binned_pixel_sums_its_native_pixels_readout_dark() -> None:
    """A SUM2 pixel holds four native pixels' readout dark."""
    _img, truth = render_scene(
        _readout_scene('coiss_nac', {'amplitude': 50.0}, detector={'binning': 2})
    )
    assert truth['artifacts']['readout_dark']['peak'] == pytest.approx(200.0)


def test_the_instrument_defaults_supply_the_measured_readout_dark() -> None:
    """Under instrument_defaults the Cassini NAC shows its measured 55 electrons."""
    dp = resolve_detector_params(
        frame_scene(instrument='coiss_nac', artifacts={'instrument_defaults': True})
    )
    assert dp.detector_modes['readout_dark']['amplitude'] == 55.0


def test_a_calibrated_vidicon_frame_keeps_its_readout_dark() -> None:
    """The vidicon's readout dark is what its calibrated product keeps."""
    scene = _readout_scene('vgiss', {'amplitude': 40.0}, detector={'quantization': 'exact'})
    img, _ = render_scene(scene)
    dp = resolve_detector_params(scene)
    rise_if = 40.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    assert float(img[-1].mean() - img[0].mean()) == pytest.approx(rise_if, rel=0.02)


def test_a_calibrated_product_removes_the_readout_dark() -> None:
    """The calibration inverse subtracts the ramp, leaving a flat calibrated sky."""
    scene = _readout_scene('coiss_calib_nac', {'amplitude': 300.0})
    img, _ = render_scene(scene)
    row_means = img.mean(axis=1)
    # One DN at this exposure bounds the rounding of the ramp before it is removed.
    dp = resolve_detector_params(scene)
    step_if = 1.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    assert float(row_means.max() - row_means.min()) <= step_if


def test_a_calibrated_product_keeps_the_enhanced_columns() -> None:
    """The pipeline's dark model is per line, so the enhanced columns stay in the product."""
    scene = _readout_scene('coiss_calib_nac', {'amplitude': 600.0, 'rbi_column_factor': 1.0})
    img, truth = render_scene(scene)
    enhanced = truth['artifacts']['readout_dark']['enhanced_columns']
    plain = [u for u in range(64) if u not in enhanced]
    dp = resolve_detector_params(scene)
    step_if = 1.0 / (dp.calibration_scale_dn_per_s_per_if * 0.1)
    excess = float(img[-1, enhanced].mean() - img[-1, plain].mean())
    assert excess == pytest.approx(600.0 / dp.gain_e_per_dn * step_if, rel=0.05)


def test_the_readout_dark_carries_shot_noise() -> None:
    """The readout dark joins the charge before the shot noise, so its last line is noisy.

    The last line collects 90,000 e-, whose shot noise is 300 e-, or 10 DN at the NAC's
    gain of 30 e- per DN.
    """
    scene = _readout_scene(
        'coiss_nac',
        {'amplitude': 9.0e4},
        noise={**quiet_noise(), 'poisson': True},
        detector={'quantization': 'exact'},
    )
    img, _ = render_scene(scene)
    assert float(img[-1].std()) == pytest.approx(10.0, rel=0.3)
