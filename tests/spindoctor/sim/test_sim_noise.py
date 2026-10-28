"""Detector electron unit chain and the structured noise stages.

These tests drive the resolved detector chain on flat synthetic signal fields so
the statistical properties of each term can be asserted in DN, plus each
structured noise stage in isolation for its shape and its disabled floor.
"""

from typing import Any

import numpy as np

from spindoctor.sim.forward.detector import (
    apply_detector,
    quantize_dn,
    resolve_detector_params,
)
from spindoctor.sim.forward.detector.noise_stages import (
    add_banding,
    add_bias_structure,
    add_cosmic_rays,
    add_dark_current,
    add_hot_pixels,
)
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed

# coiss_nac catalog: full well 110k e-, gain state 2 = 30 e-/DN, bias 20 DN.
_NAC_GAIN = 30.0
_NAC_FULL_WELL_E = 110.0e3
_NAC_BIAS = 20.0


def _run_detector(
    signal_value: float,
    *,
    size: int = 160,
    noise: dict[str, Any] | None = None,
    seed: int = 1,
    **extra: object,
) -> np.ndarray:
    """Run the detector stage over a flat signal field and return the DN image."""
    frame = SimFrame(
        signal=np.full((size, size), signal_value, dtype=np.float64),
        point_e=np.zeros((size, size), dtype=np.float64),
    )
    params: dict[str, Any] = {'instrument': 'coiss_nac', 'random_seed': seed, 'exposure_sec': 1.0}
    if noise is not None:
        params['noise'] = noise
    params.update(extra)
    rng = np.random.default_rng(derive_effect_seed(seed, 'detector'))
    apply_detector(frame, params=params, rng=rng)
    return frame.signal


def test_signal_maps_to_dn_through_the_electron_chain() -> None:
    """A flat signal maps to frac * full_well_e / gain + bias DN, noise-free."""
    dn = _run_detector(0.5)
    expected = round(0.5 * 0.5 * _NAC_FULL_WELL_E / _NAC_GAIN + _NAC_BIAS)
    assert np.all(dn == expected)


def test_floor_scene_adds_no_shot_noise() -> None:
    """With no noise block the detector converts to DN without stochastic noise."""
    dn = _run_detector(0.3)
    assert float(dn.std()) == 0.0


def test_poisson_variance_tracks_electron_mean() -> None:
    """Electron-domain shot noise gives DN variance near mean_e / gain^2."""
    dn = _run_detector(0.5, noise={'poisson': True})
    mean_e = 0.5 * 0.5 * _NAC_FULL_WELL_E
    expected_var_dn = mean_e / _NAC_GAIN**2
    assert abs(float(dn.var()) - expected_var_dn) < 0.2 * expected_var_dn


def test_read_noise_dn_override_converts_through_gain() -> None:
    """A DN read-noise override yields that DN spread after the gain divide."""
    dn = _run_detector(0.3, noise={'poisson': False, 'read_noise_dn': 10.0})
    assert abs(float(dn.std()) - 10.0) < 1.0


def test_cosmic_rays_reach_the_adc_ceiling() -> None:
    """A nonzero cosmic-ray rate plants events that clip at the ADC ceiling."""
    dn = _run_detector(
        0.1, size=100, noise={'poisson': False, 'cosmic_ray_events_per_pixel_per_sec': 0.01}
    )
    assert int((dn >= 4095.0).sum()) >= 1


def test_detector_is_deterministic_for_equal_seeds() -> None:
    """Equal seeds and inputs produce byte-identical noisy output."""
    dn_a = _run_detector(0.5, size=64, noise={'poisson': True, 'read_noise_dn': 5.0}, seed=11)
    dn_b = _run_detector(0.5, size=64, noise={'poisson': True, 'read_noise_dn': 5.0}, seed=11)
    assert np.array_equal(dn_a, dn_b)


def test_detector_differs_for_different_seed() -> None:
    """Changing the scene seed changes the realized noise field."""
    dn_a = _run_detector(0.5, size=64, noise={'poisson': True, 'read_noise_dn': 5.0}, seed=11)
    dn_b = _run_detector(0.5, size=64, noise={'poisson': True, 'read_noise_dn': 5.0}, seed=99)
    assert not np.array_equal(dn_a, dn_b)


def test_resolved_gain_selects_the_state_2_value() -> None:
    """The default coiss_nac gain state resolves to the state-2 catalog value."""
    dp = resolve_detector_params({'instrument': 'coiss_nac', 'random_seed': 1})
    assert dp.gain_e_per_dn == _NAC_GAIN


def test_read_noise_absent_is_zero_on_the_floor() -> None:
    """With no noise block the resolved read noise is zero (the honest floor)."""
    dp = resolve_detector_params({'instrument': 'coiss_nac', 'random_seed': 1})
    assert dp.read_noise_e == 0.0


# --- quantization sub-modes -------------------------------------------------


def test_exact_quantization_rounds_to_integer() -> None:
    """The exact sub-mode rounds DN to the nearest integer."""
    dn = np.array([[10.4, 10.6]], dtype=np.float64)
    out = quantize_dn(dn, mode='exact', saturation_dn=4095.0)
    assert np.array_equal(out, np.array([[10.0, 11.0]]))


def test_8bit_quantization_rounds_and_clips_at_255() -> None:
    """The 8-bit sub-mode rounds to integer DN and clips at the 255 code ceiling."""
    dn = np.array([[10.4, 254.6, 300.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='8bit', saturation_dn=4095.0)
    assert np.array_equal(out, np.array([[10.0, 255.0, 255.0]]))


def test_8bit_ceiling_is_independent_of_saturation_dn() -> None:
    """A deeper detector's saturation_dn does not raise the 8-bit code ceiling."""
    dn = np.array([[1000.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='8bit', saturation_dn=65535.0)
    assert float(out[0, 0]) == 255.0


def test_uneven_12bit_snaps_to_power_of_two_boundaries() -> None:
    """The uneven-12bit sub-mode concentrates codes at power-of-two boundaries."""
    dn = np.array([[255.0, 256.0, 257.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='uneven_12bit', saturation_dn=4095.0)
    assert np.all(out == 256.0)


def test_sqrt_lut_leaves_a_signal_dependent_residual() -> None:
    """The sqrt-LUT companding round-trip leaves a residual that grows with signal."""
    dn = np.array([[100.0, 3000.0]], dtype=np.float64)
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0)
    residual = np.abs(out - dn)
    assert float(residual[0, 1]) > float(residual[0, 0])


# --- structured noise stages, each disabled at zero -------------------------


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


def test_hot_pixels_elevate_a_sparse_population() -> None:
    """Hot pixels raise a fraction of pixels well above zero."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons,
        fraction=0.01,
        amplitude_e=1.0e4,
        column_factor=0.0,
        rng=np.random.default_rng(3),
    )
    assert int((electrons > 0.0).sum()) > 0


def test_hot_pixel_warm_column_is_a_total_charge_fraction() -> None:
    """The warm-column integral is bounded by column_factor times the charge.

    The column contamination is a fraction of the hot pixel's TOTAL charge
    spread along the column, not a per-pixel amplitude, so the streak stays
    conservative and frame-size-invariant.
    """
    electrons = np.zeros((512, 512), dtype=np.float64)
    column_factor = 0.3
    add_hot_pixels(
        electrons,
        fraction=1.0 / (512 * 512),  # exactly one hot pixel
        amplitude_e=4.0e4,
        column_factor=column_factor,
        rng=np.random.default_rng(7),
    )
    hot_v, hot_u = np.unravel_index(int(np.argmax(electrons)), electrons.shape)
    hot_charge = float(electrons[hot_v, hot_u])
    column_integral = float(electrons[:hot_v, hot_u].sum())
    assert hot_charge > 0.0
    assert column_integral > 0.0
    assert column_integral <= column_factor * hot_charge + 1e-6


def test_hot_pixel_warm_column_is_frame_size_invariant() -> None:
    """The same column_factor bleeds the same total charge at any frame height."""
    integrals = []
    for size in (128, 512):
        electrons = np.zeros((size, size), dtype=np.float64)
        add_hot_pixels(
            electrons,
            fraction=1.0 / (size * size),
            amplitude_e=4.0e4,
            column_factor=0.3,
            rng=np.random.default_rng(7),
        )
        hot_v, hot_u = np.unravel_index(int(np.argmax(electrons)), electrons.shape)
        hot_charge = float(electrons[hot_v, hot_u])
        integrals.append(float(electrons[:hot_v, hot_u].sum()) / hot_charge)
    assert abs(integrals[0] - integrals[1]) < 1e-9


def test_hot_pixels_disabled_at_zero_fraction() -> None:
    """A zero hot-pixel fraction is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons, fraction=0.0, amplitude_e=1.0e4, column_factor=0.5, rng=np.random.default_rng(3)
    )
    assert np.all(electrons == 0.0)


def test_banding_is_row_correlated() -> None:
    """Coherent banding is constant along a row and varies between rows."""
    electrons = np.zeros((64, 64), dtype=np.float64)
    add_banding(electrons, amplitude_e=100.0, period_px=16.0, rng=np.random.default_rng(5))
    row_spread = float(np.ptp(electrons, axis=1).max())
    col_spread = float(np.ptp(electrons, axis=0).max())
    assert row_spread < 1e-9
    assert col_spread > 1.0


def test_banding_disabled_at_zero_amplitude() -> None:
    """A zero banding amplitude is a no-op."""
    electrons = np.zeros((64, 64), dtype=np.float64)
    add_banding(electrons, amplitude_e=0.0, period_px=16.0, rng=np.random.default_rng(5))
    assert np.all(electrons == 0.0)


def test_bias_structure_adds_a_pedestal_and_gradients() -> None:
    """Bias structure changes the DN field when any amplitude is nonzero."""
    dn = np.zeros((32, 48), dtype=np.float64)
    add_bias_structure(
        dn,
        pedestal_sigma_dn=2.0,
        row_gradient_dn=4.0,
        col_gradient_dn=2.0,
        rng=np.random.default_rng(7),
    )
    assert float(dn.std()) > 0.0


def test_bias_structure_disabled_at_zero_amplitudes() -> None:
    """Bias structure with every amplitude zero is a no-op."""
    dn = np.zeros((32, 48), dtype=np.float64)
    add_bias_structure(
        dn,
        pedestal_sigma_dn=0.0,
        row_gradient_dn=0.0,
        col_gradient_dn=0.0,
        rng=np.random.default_rng(7),
    )
    assert np.all(dn == 0.0)


def test_cosmic_rays_deposit_events() -> None:
    """A nonzero cosmic-ray rate deposits charge somewhere in the frame."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=0.01,
        exposure_sec=1.0,
        amplitude_e=1.0e5,
        rng=np.random.default_rng(9),
    )
    assert int((electrons > 0.0).sum()) >= 1


def test_cosmic_rays_disabled_at_zero_rate() -> None:
    """A zero cosmic-ray rate is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=0.0,
        exposure_sec=1.0,
        amplitude_e=1.0e5,
        rng=np.random.default_rng(9),
    )
    assert np.all(electrons == 0.0)
