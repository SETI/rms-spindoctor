"""Detector electron unit chain and the structured noise stages.

These tests drive the resolved detector chain on flat synthetic signal fields so
the statistical properties of each term can be asserted in DN, plus each
structured noise stage in isolation for its shape and its disabled floor.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults
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
    detector_hot_pixel_map,
)
from spindoctor.sim.forward.detector.sqrt_lut import SqrtLut, resolve_sqrt_lut
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


_CASSINI_LUT = resolve_sqrt_lut(
    resolve_detector_defaults('coiss_nac')['sqrt_lut'], saturation_dn=4095.0
)
_CASSINI_DN = _CASSINI_LUT.levels()


@pytest.mark.parametrize(
    ('code', 'step'),
    [(10, 3.0), (25, 5.0), (55, 8.0), (100, 13.0), (200, 24.0)],
    ids=['code-10', 'code-25', 'code-55', 'code-100', 'code-200'],
)
def test_the_cassini_table_steps_grow_with_the_code(code: int, step: float) -> None:
    """The table's decode steps match the steps measured on real TABLE frames."""
    assert float(_CASSINI_DN[code + 1] - _CASSINI_DN[code]) == step


def test_the_cassini_table_decodes_256_whole_dn_up_to_4086() -> None:
    """Every code decodes to its own whole DN, the top code to 4086."""
    assert len(_CASSINI_DN) == 256
    assert np.array_equal(_CASSINI_DN, np.round(_CASSINI_DN))
    assert float(_CASSINI_DN[-1]) == 4086.0


def test_the_sqrt_lut_snaps_each_value_to_the_nearest_decoded_dn() -> None:
    """Values between two decoded DN go to the nearer one; above the table, to its top."""
    lower, upper = float(_CASSINI_DN[40]), float(_CASSINI_DN[41])
    dn = np.array([[lower + 0.4 * (upper - lower), lower + 0.6 * (upper - lower), 5000.0]])
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=_CASSINI_LUT)
    assert out.tolist() == [[lower, upper, float(_CASSINI_DN[-1])]]


def test_the_sqrt_lut_keeps_a_nan_pixel_nan() -> None:
    """An invalid pixel stays invalid instead of decoding to the table's top."""
    out = quantize_dn(
        np.array([[np.nan, 100.0]]), mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=_CASSINI_LUT
    )
    assert bool(np.isnan(out[0, 0]))


def test_a_camera_without_a_table_gets_the_plain_square_root_curve() -> None:
    """Without a catalog entry the curve is a pure square root topping out at the ADC ceiling."""
    curve = resolve_sqrt_lut(None, saturation_dn=4095.0)
    assert curve == SqrtLut(
        codes=256, linear_dn=0.0, quadratic_dn=4095.0 / 255.0**2, whole_dn=False
    )
    assert float(curve.levels()[-1]) == pytest.approx(4095.0)


def test_a_table_entry_sets_its_own_code_count() -> None:
    """A 10-bit table decodes 1024 levels, its top code at the curve's value for 1023."""
    entry = {'codes': 1024, 'linear_dn': 0.0, 'quadratic_dn': 0.01, 'whole_dn': False}
    levels = resolve_sqrt_lut(entry, saturation_dn=16383.0).levels()
    assert len(levels) == 1024
    assert float(levels[-1]) == pytest.approx(0.01 * 1023.0**2)


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'codes': 1}, r'sqrt_lut codes must be at least 2, got 1$'),
        ({'codes': 256.5}, r'sqrt_lut codes must be a whole number, got 256\.5$'),
        ({'linear_dn': float('nan')}, r'must be finite, got linear_dn nan and quadratic_dn 1\.0$'),
        (
            {'quadratic_dn': float('inf')},
            r'must be finite, got linear_dn 0\.0 and quadratic_dn inf$',
        ),
    ],
    ids=['one-code', 'fractional-codes', 'nan-linear', 'infinite-quadratic'],
)
def test_an_unusable_table_entry_is_refused(override: dict[str, float], message: str) -> None:
    """A table entry the encoder can't use raises instead of quantizing badly."""
    entry = {'codes': 256, 'linear_dn': 0.0, 'quadratic_dn': 1.0, 'whole_dn': False, **override}
    with pytest.raises(ValueError, match=message):
        resolve_sqrt_lut(entry, saturation_dn=4095.0)


def test_the_plain_curve_on_a_zero_ceiling_sends_every_value_to_zero() -> None:
    """The plain curve never decodes above the ADC ceiling, even a zero one."""
    dn = np.array([[0.4, 37.0, 2500.0]])
    out = quantize_dn(dn, mode='sqrt_lut', saturation_dn=0.0)
    assert out.tolist() == [[0.0, 0.0, 0.0]]


def test_the_sqrt_lut_mode_defaults_to_the_plain_curve() -> None:
    """A call without a curve quantizes through the plain square-root curve."""
    dn = np.array([[0.4, 37.0, 2500.0]])
    plain = resolve_sqrt_lut(None, saturation_dn=4095.0)
    default = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0)
    explicit = quantize_dn(dn, mode='sqrt_lut', saturation_dn=4095.0, sqrt_lut=plain)
    assert np.array_equal(default, explicit)


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
        detector_id='coiss_nac',
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
        fraction=1.0 / (512 * 512),  # exactly one hot pixel, drawn from a one-pixel pool
        amplitude_e=4.0e4,
        column_factor=column_factor,
        detector_id='coiss_nac',
        rng=np.random.default_rng(7),
        candidate_pool=(np.array([300]), np.array([200])),
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
            detector_id='coiss_nac',
            rng=np.random.default_rng(7),
            candidate_pool=(np.array([100]), np.array([50])),
        )
        hot_v, hot_u = np.unravel_index(int(np.argmax(electrons)), electrons.shape)
        hot_charge = float(electrons[hot_v, hot_u])
        integrals.append(float(electrons[:hot_v, hot_u].sum()) / hot_charge)
    assert abs(integrals[0] - integrals[1]) < 1e-9


def test_adversarial_hot_pixels_are_distinct_and_capped_at_the_pool() -> None:
    """A request beyond the pool plants each pool pixel once, at its recorded charge."""
    electrons = np.zeros((64, 64), dtype=np.float64)
    pool_v, pool_u = np.divmod(np.arange(10, dtype=np.intp) * 37, 64)
    record = add_hot_pixels(
        electrons,
        fraction=25.0 / (64 * 64),
        amplitude_e=1.0e4,
        column_factor=0.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(11),
        candidate_pool=(pool_v, pool_u),
    )
    assert sorted(map(tuple, record['pixels'])) == sorted(
        zip(pool_v.tolist(), pool_u.tolist(), strict=True)
    )
    planted = [float(electrons[v, u]) for v, u in record['pixels']]
    assert planted == pytest.approx(record['amplitudes_e'])


def test_hot_pixels_disabled_at_zero_fraction() -> None:
    """A zero hot-pixel fraction is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons,
        fraction=0.0,
        amplitude_e=1.0e4,
        column_factor=0.5,
        detector_id='coiss_nac',
        rng=np.random.default_rng(3),
    )
    assert np.all(electrons == 0.0)


def _hot_set(shape: tuple[int, int], fraction: float, detector_id: str) -> set[tuple[int, int]]:
    """The detector map's hot pixels over ``shape`` as a coordinate set."""
    hot_v, hot_u, _amps = detector_hot_pixel_map(shape, fraction=fraction, detector_id=detector_id)
    return set(zip(hot_v.tolist(), hot_u.tolist(), strict=True))


def test_the_detector_map_ignores_the_scene_seed() -> None:
    """Two populations on one detector share every pixel and amplitude, whatever the rng."""
    planted = []
    for seed in (1, 99):
        electrons = np.zeros((256, 256), dtype=np.float64)
        add_hot_pixels(
            electrons,
            fraction=0.002,
            amplitude_e=1.0e4,
            column_factor=0.0,
            detector_id='gossi',
            rng=np.random.default_rng(seed),
        )
        planted.append(electrons)
    assert np.array_equal(planted[0], planted[1])


def test_a_smaller_frame_sees_the_corner_of_the_detector_map() -> None:
    """The 128 px map is the 512 px map's top-left corner."""
    big = _hot_set((512, 512), 0.002, 'coiss_nac')
    corner = {(v, u) for v, u in big if v < 128 and u < 128}
    assert _hot_set((128, 128), 0.002, 'coiss_nac') == corner


def test_a_higher_fraction_only_adds_hot_pixels() -> None:
    """The map at a higher fraction contains the map at a lower one."""
    low = _hot_set((256, 256), 0.001, 'coiss_nac')
    high = _hot_set((256, 256), 0.004, 'coiss_nac')
    assert low < high


def test_two_detectors_have_different_maps() -> None:
    """The NAC and WAC maps differ."""
    assert _hot_set((256, 256), 0.002, 'coiss_nac') != _hot_set((256, 256), 0.002, 'coiss_wac')


def test_the_detector_map_count_follows_the_fraction() -> None:
    """A 1024 px map holds the expected count within five binomial sigmas."""
    fraction = 1.0e-3
    n = 1024 * 1024
    count = len(_hot_set((1024, 1024), fraction, 'coiss_nac'))
    sigma = (n * fraction * (1.0 - fraction)) ** 0.5
    assert abs(count - n * fraction) < 5.0 * sigma


def test_the_detector_map_amplitudes_average_one() -> None:
    """The relative amplitudes are exponential with mean 1 (within 10 percent on ~1000)."""
    _v, _u, amps = detector_hot_pixel_map((1024, 1024), fraction=1.0e-3, detector_id='gossi')
    assert float(amps.mean()) == pytest.approx(1.0, rel=0.1)


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
