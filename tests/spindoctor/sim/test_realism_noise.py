"""Unit tests for the FOM 1 sky-noise machinery (noise estimators, patches, PSD)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.optimize

from spindoctor.sim.realism.noise import (
    SkyPatch,
    dark_sky_patches,
    find_uniform_patches,
    lattice_noise_sigma,
    noisy_patches,
    paired_difference_sigma,
    patch_noise_sigma,
    quantization_step,
    radial_power_spectrum,
)


def test_paired_difference_recovers_known_sigma() -> None:
    """On a pure-noise frame the estimator recovers the planted sigma."""
    rng = np.random.default_rng(11)
    patch = rng.normal(100.0, 3.0, (64, 64))
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(3.0, rel=0.1)


def test_paired_difference_ignores_smooth_gradient() -> None:
    """A strong smooth gradient does not inflate the noise estimate.

    This is the reason for local differencing: naive std() would report
    the gradient as noise.
    """
    rng = np.random.default_rng(12)
    vv, _uu = np.mgrid[0:64, 0:64].astype(np.float64)
    patch = 50.0 * vv / 64.0 + rng.normal(0.0, 2.0, (64, 64))
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(2.0, rel=0.1)
    assert float(patch.std()) > 5.0  # naive std really is inflated


def test_paired_difference_robust_to_spikes() -> None:
    """A handful of huge spikes barely moves the MAD-based estimate."""
    rng = np.random.default_rng(13)
    patch = rng.normal(0.0, 1.5, (64, 64))
    patch[10, 10] = 1e5
    patch[40, 20] = -1e5
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(1.5, rel=0.1)


def test_paired_difference_degenerate_patch_is_nan() -> None:
    """A 1-D or tiny patch yields NaN, not a crash."""
    assert np.isnan(paired_difference_sigma(np.zeros((1, 5))))


def test_quantized_noise_below_one_step_is_recovered() -> None:
    """Gaussian noise of sigma 0.3 DN rounded to whole DN gives back 0.3 DN.

    Paired differences report 0 here: most neighbors share a value.
    """
    rng = np.random.default_rng(21)
    patch = np.round(rng.normal(100.2, 0.3, (64, 64)))
    assert patch_noise_sigma(patch) == pytest.approx(0.3, rel=0.05)


def test_quantized_sky_with_gradient_and_spikes_is_recovered() -> None:
    """A real-like whole-DN sky recovers its sigma.

    The sky carries a gradient of 3 DN across the patch, sigma 2.5 DN, two hot pixels,
    and a cosmic ray.
    """
    rng = np.random.default_rng(22)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    sky = 540.0 + 3.0 * (vv + uu) / 64.0 + rng.normal(0.0, 2.5, (32, 32))
    sky[4, 7] += 300.0
    sky[20, 11] += 80.0
    sky[12:14, 25] += 2000.0
    assert patch_noise_sigma(np.round(sky)) == pytest.approx(2.5, rel=0.08)


def test_quantized_noise_on_a_fractional_lattice_is_recovered() -> None:
    """A lattice whose step is not 1, such as 8-bit I/F, is fitted in its own units."""
    rng = np.random.default_rng(23)
    step = 1.0 / 255.0
    patch = np.round(rng.normal(3.3, 0.6, (64, 64))) * step
    assert patch_noise_sigma(patch) == pytest.approx(0.6 * step, rel=0.05)


def test_quantization_step_is_none_for_continuous_values() -> None:
    """Values that take any real value sit on no lattice."""
    rng = np.random.default_rng(24)
    assert quantization_step(rng.normal(0.0, 1.0, (32, 32))) is None


def test_quantization_step_is_none_for_float32_values() -> None:
    """Float32 values sit on the float spacing, which is no quantization."""
    rng = np.random.default_rng(27)
    patch = rng.normal(3e-3, 2e-4, (32, 32)).astype(np.float32)
    assert quantization_step(patch) is None


def test_two_level_patch_sigma_is_not_measurable() -> None:
    """A patch whose pixels take only two values gives NaN, since any sigma fits it."""
    rng = np.random.default_rng(25)
    patch = np.round(rng.normal(100.5, 0.15, (32, 32)))
    assert np.isnan(lattice_noise_sigma(patch, 1.0))


def test_two_levels_and_one_stray_pixel_are_not_measurable() -> None:
    """One stray pixel on a third level doesn't make a two-level patch measurable."""
    rng = np.random.default_rng(28)
    patch = np.round(rng.normal(100.5, 0.15, (32, 32)))
    patch[5, 5] = 102.0
    assert np.isnan(patch_noise_sigma(patch))


def test_a_patch_of_one_value_is_not_measurable() -> None:
    """A tile of one value has no measurable sigma, as a two-level patch has none."""
    assert np.isnan(patch_noise_sigma(np.full((32, 32), 7.0)))


def test_quantized_noise_on_a_curved_sky_is_recovered() -> None:
    """Curvature across the patch is fitted, never read as noise.

    The sky is a bowl 4 DN deep with sigma 1 DN; a fitted plane leaves the bowl in the
    residuals and reads about 1.31 DN.
    """
    rng = np.random.default_rng(29)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    bowl = 2.0 * (((vv - 15.5) / 16.0) ** 2 + ((uu - 15.5) / 16.0) ** 2)
    patch = np.round(100.3 + bowl + rng.normal(0.0, 1.0, (32, 32)))
    assert patch_noise_sigma(patch) == pytest.approx(1.0, rel=0.05)


def test_quantized_noise_well_below_one_step_on_a_gradient_is_recovered() -> None:
    """Sigma 0.2 DN on a gradient of 1 DN across the patch gives back 0.2 DN.

    The gradient is fitted together with the noise; a gradient fixed beforehand by least
    squares reads about 11% high here.  The median runs over 20 patches at random levels,
    since one patch this far below a step carries only a few pixels off its level.
    """
    rng = np.random.default_rng(30)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    gradient = (vv + uu) / 64.0
    sigmas = [
        patch_noise_sigma(
            np.round(100.0 + rng.uniform() + gradient + rng.normal(0.0, 0.2, (32, 32)))
        )
        for _ in range(20)
    ]
    assert float(np.nanmedian(sigmas)) == pytest.approx(0.2, rel=0.03)


def test_quantized_noise_clipped_at_the_floor_is_recovered() -> None:
    """Sky at the zero floor, where half the values are clipped to 0, gives back its sigma."""
    rng = np.random.default_rng(31)
    patch = np.clip(np.round(rng.normal(0.0, 1.0, (32, 32))), 0.0, None)
    assert patch_noise_sigma(patch) == pytest.approx(1.0, rel=0.08)


def _fit_over_drawn_sigma(rng: np.random.Generator, sigma: float, clip: str) -> float:
    """Draw one 32 x 32 sky of noise ``sigma``, round it, and divide its fit by the drawn sigma.

    The sky is unclipped at about 1000, or clipped at the zero floor or at a ceiling of
    4095, each with about half its values on the clipped level.  Dividing by the standard
    deviation of the noise drawn, before rounding and clipping, removes the sampling
    scatter of the sigma itself.
    """
    noise = rng.normal(0.0, sigma, (32, 32))
    sky = rng.uniform() + noise
    if clip == 'floor':
        patch = np.maximum(np.round(sky), 0.0)
    elif clip == 'ceiling':
        patch = np.minimum(np.round(4094.0 + sky), 4095.0)
    else:
        patch = np.round(1000.0 + sky)
    return patch_noise_sigma(patch) / float(noise.std())


@pytest.mark.parametrize('sigma', [10.0, 20.0])
def test_quantized_noise_of_many_steps_is_recovered(sigma: float) -> None:
    """Noise of many steps leaves empty levels in its tails, and every pixel still counts.

    A fit that drops the pixels beyond the first empty level truncates the distribution
    and reads about 3% low at 10 steps and 8% low at 20.
    """
    rng = np.random.default_rng(33)
    ratios = [_fit_over_drawn_sigma(rng, sigma, 'none') for _ in range(6)]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    ('clip', 'sigma'), [('floor', 3.0), ('ceiling', 5.0)], ids=['floor-3', 'ceiling-5']
)
def test_clipped_quantized_noise_of_a_few_steps_is_recovered(clip: str, sigma: float) -> None:
    """Sky clipped at the floor or at saturation, half its values on one level, keeps its sigma.

    Half the residuals share a value there, so a robust first guess of the sigma comes out
    far too small; a fit held to the window that guess builds reads about 15% low at 3
    steps and 40% low at 5.  Half of each patch is clipped, so the median runs over 20.
    """
    rng = np.random.default_rng(34)
    ratios = [_fit_over_drawn_sigma(rng, sigma, clip) for _ in range(20)]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.025)


def test_a_quantized_fit_that_fails_to_converge_is_not_measurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit the optimizer reports as failed gives NaN rather than where it stopped."""
    real_minimize = scipy.optimize.minimize

    def failing_minimize(*args: Any, **kwargs: Any) -> scipy.optimize.OptimizeResult:
        result = real_minimize(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(scipy.optimize, 'minimize', failing_minimize)
    rng = np.random.default_rng(35)
    assert np.isnan(lattice_noise_sigma(np.round(rng.normal(100.0, 1.0, (32, 32))), 1.0))


@pytest.mark.parametrize(
    ('scale', 'offset', 'level'),
    [(1.234e-3, 0.0, 3000.0), (1e-4, 0.5, 30.0), (1e-4, 8.0, 30.0)],
    ids=['scaled', 'scaled-and-offset', 'offset-beyond-the-step-tolerance'],
)
def test_quantization_step_finds_whole_dn_scaled_into_float32(
    scale: float, offset: float, level: float
) -> None:
    """Whole DN scaled into float32 sit on a lattice although each value carries rounding.

    Near 8.0, a float32 value's rounding exceeds a thousandth of a 1e-4 step, so the
    tolerance there comes from the float precision at the values' magnitude.
    """
    rng = np.random.default_rng(32)
    patch = (np.round(rng.normal(level, 3.0, (32, 32))) * scale + offset).astype(np.float32)
    assert quantization_step(patch) == pytest.approx(scale, rel=1e-4)


def test_find_uniform_patches_keeps_quantized_sky_below_one_step() -> None:
    """Whole-DN sky whose noise is below one DN still yields every patch."""
    rng = np.random.default_rng(26)
    image = np.round(rng.normal(20.0, 0.4, (64, 64)))
    assert len(find_uniform_patches(image, patch_size=32, max_mean_quantile=None)) == 4


def test_find_uniform_patches_rejects_structured_tiles() -> None:
    """Tiles crossed by a sharp edge are rejected; pure-noise tiles kept."""
    rng = np.random.default_rng(14)
    image = rng.normal(10.0, 1.0, (128, 128))
    image[:, 64:] += 500.0  # hard step through the right half tiles' columns
    patches = find_uniform_patches(image, patch_size=32, max_mean_quantile=None)
    # The step lands between tile columns, so tiles remain uniform except
    # none straddle it; shift the step to cut through tiles instead.
    image2 = rng.normal(10.0, 1.0, (128, 128))
    image2[:, 48:] += 500.0  # cuts through the second tile column
    patches2 = find_uniform_patches(image2, patch_size=32, max_mean_quantile=None)
    coords2 = {(p.v0, p.u0) for p in patches2}
    assert all((v0, 32) not in coords2 for v0 in range(0, 128, 32))
    assert len(patches) == 16


def test_find_uniform_patches_sky_cut_keeps_low_means() -> None:
    """With the sky cut on, only the lowest-mean patches survive."""
    rng = np.random.default_rng(15)
    image = rng.normal(10.0, 1.0, (128, 128))
    image[64:, :] += 100.0  # bottom half is bright
    patches = find_uniform_patches(image, patch_size=32, max_mean_quantile=0.25)
    assert patches
    assert all(p.mean < 50.0 for p in patches)


def test_radial_power_spectrum_flat_for_white_noise() -> None:
    """White noise has an approximately flat radially averaged spectrum."""
    rng = np.random.default_rng(16)
    freq, power = radial_power_spectrum(rng.normal(0.0, 1.0, (64, 64)), n_bins=8)
    assert freq.shape == (8,)
    finite = power[np.isfinite(power)]
    assert finite.size >= 6
    assert float(finite.max() / finite.min()) < 5.0


def test_radial_power_spectrum_peaks_at_banding_frequency() -> None:
    """Coherent banding concentrates power in its frequency bin."""
    vv = np.arange(64)[:, np.newaxis].astype(np.float64)
    period_px = 8.0
    banding = 5.0 * np.sin(2.0 * np.pi * vv / period_px) * np.ones((64, 64))
    rng = np.random.default_rng(17)
    freq, power = radial_power_spectrum(banding + rng.normal(0.0, 0.1, (64, 64)), n_bins=16)
    peak_bin = int(np.nanargmax(power))
    expected_freq = 1.0 / period_px
    assert freq[peak_bin] == pytest.approx(expected_freq, abs=0.5 / 16)


def test_dark_sky_is_selected_by_level_alone() -> None:
    """A patch above the ceiling is dropped however noisy it is, and one below is kept.

    The ring patch lies within three of its own sigmas of the ceiling, so a selection
    scaled by any sigma could keep it.
    """
    sky = SkyPatch(v0=0, u0=0, size=32, mean=2e-4, sigma=1e-8)
    ring = SkyPatch(v0=0, u0=32, size=32, mean=0.07, sigma=0.05)
    assert dark_sky_patches([sky, ring], 2.54e-3) == [sky]


def test_dark_sky_keeps_a_patch_far_below_the_ceiling() -> None:
    """A patch far below the ceiling is dark sky."""
    dark = SkyPatch(v0=0, u0=0, size=32, mean=-1e-3, sigma=1e-4)
    assert dark_sky_patches([dark], 2.54e-3) == [dark]


def test_a_constant_patch_has_no_noise_to_compare() -> None:
    """A patch whose sigma is 0 is dropped."""
    constant = SkyPatch(v0=0, u0=0, size=32, mean=0.0, sigma=0.0)
    sky = SkyPatch(v0=0, u0=32, size=32, mean=4e-5, sigma=1e-4)
    assert noisy_patches([constant, sky]) == [sky]
