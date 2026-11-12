"""Unit tests for the FOM 1 sky-noise machinery (paired differences, PSD)."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.sim.realism.noise import (
    SkyPatch,
    find_uniform_patches,
    paired_difference_sigma,
    radial_power_spectrum,
    sky_patches_in_band,
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


def test_sky_band_drops_a_patch_on_content() -> None:
    """A patch many of its sigmas above the matched sky is content; one within them is sky."""
    sky = SkyPatch(v0=0, u0=0, size=32, mean=8e-4, sigma=5e-4)
    ring = SkyPatch(v0=0, u0=32, size=32, mean=0.07, sigma=1e-3)
    assert sky_patches_in_band([sky, ring], 1e-5) == [sky]
