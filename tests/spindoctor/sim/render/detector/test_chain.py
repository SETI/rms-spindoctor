"""The detector's electron unit chain, driven on flat signal fields.

A flat field lets the statistical properties of each term be asserted in DN.
"""

from typing import Any

import numpy as np

from spindoctor.sim.render.detector import apply_detector
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.stages import SimFrame

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
    apply_detector(frame, params=params)
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
    """With no noise block the resolved read noise is zero."""
    dp = resolve_detector_params({'instrument': 'coiss_nac', 'random_seed': 1})
    assert dp.read_noise_e == 0.0
