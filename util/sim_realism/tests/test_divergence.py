"""Unit tests for the realism-match W1 divergence and support labels."""

from __future__ import annotations

import numpy as np
import pytest
from util.sim_realism.divergence import (
    CohortSupport,
    cohort_support,
    w1_between_densities,
    w1_divergence,
)


def test_w1_identical_samples_is_zero() -> None:
    """W1 between a sample and itself is exactly zero."""
    rng = np.random.default_rng(3)
    sample = rng.normal(5.0, 2.0, 5000)
    result = w1_divergence(sample, sample)
    assert result.w1 == 0.0


def test_w1_pure_shift_recovers_the_shift() -> None:
    """W1 between X and X + c is exactly c (transport moves every point c)."""
    rng = np.random.default_rng(4)
    sample = rng.normal(0.0, 1.0, 20000)
    result = w1_divergence(sample, sample + 0.75)
    assert result.w1 == pytest.approx(0.75, rel=0.02)


def test_w1_normalization_uses_real_iqr() -> None:
    """The normalized value is w1 / IQR of the raw real sample."""
    rng = np.random.default_rng(5)
    real = rng.normal(0.0, 1.0, 20000)
    sim = real + 1.0
    result = w1_divergence(real, sim)
    expected_iqr = float(np.quantile(real, 0.75) - np.quantile(real, 0.25))
    assert result.real_iqr == pytest.approx(expected_iqr)
    assert result.w1_normalized == pytest.approx(result.w1 / expected_iqr)


def test_w1_clip_suppresses_displaced_tail() -> None:
    """A 0.5% displaced far tail barely moves the clipped W1.

    Unclipped W1 grows linearly with the displaced-tail distance; the
    1st/99th percentile winsorization is what keeps a noise FOM from
    silently measuring an artifact FOM's tail.
    """
    rng = np.random.default_rng(6)
    real = rng.normal(0.0, 1.0, 20000)
    sim = rng.normal(0.0, 1.0, 20000)
    contaminated = sim.copy()
    contaminated[:100] += 1.0e6  # 0.5% of samples displaced enormously far.
    clean = w1_divergence(real, sim)
    spiked = w1_divergence(real, contaminated)
    # Without the clip the tail would add ~0.005 * 1e6 = 5000 to W1.
    assert abs(spiked.w1 - clean.w1) < 0.5


def test_w1_small_sample_is_unusable() -> None:
    """Below the minimum sample count the result is flagged unusable."""
    result = w1_divergence(np.arange(3.0), np.arange(100.0))
    assert not result.usable
    assert np.isnan(result.w1)


def test_w1_zero_iqr_yields_nan_normalization() -> None:
    """A degenerate (constant) real sample cannot normalize the distance."""
    real = np.full(100, 7.0)
    sim = np.linspace(0.0, 1.0, 100)
    result = w1_divergence(real, sim)
    assert np.isfinite(result.w1)
    assert np.isnan(result.w1_normalized)


def test_w1_records_sample_counts() -> None:
    """Finite-sample counts land in the result."""
    real = np.concatenate([np.arange(50.0), [np.nan, np.inf]])
    sim = np.arange(30.0)
    result = w1_divergence(real, sim)
    assert result.n_real == 50
    assert result.n_sim == 30


def test_density_w1_identical_densities_is_zero() -> None:
    """Equal density curves have zero transport distance."""
    x = np.linspace(0.0, 0.5, 32)
    density = np.exp(-x * 10.0)
    result = w1_between_densities(x, density, density)
    assert result.w1 == pytest.approx(0.0, abs=1e-12)


def test_density_w1_point_mass_shift() -> None:
    """Two unit point masses one bin apart transport by the bin spacing."""
    x = np.linspace(0.0, 1.0, 11)
    real = np.zeros(11)
    sim = np.zeros(11)
    real[3] = 1.0
    sim[5] = 1.0
    result = w1_between_densities(x, real, sim)
    assert result.w1 == pytest.approx(0.2)


def test_density_w1_empty_curve_is_unusable() -> None:
    """A zero-mass curve cannot be compared."""
    x = np.linspace(0.0, 1.0, 11)
    result = w1_between_densities(x, np.zeros(11), np.ones(11))
    assert not result.usable


def test_cohort_support_thresholds() -> None:
    """Support labels follow the frame-count thresholds."""
    assert cohort_support(0) is CohortSupport.UNSUPPORTED
    assert cohort_support(1) is CohortSupport.UNSUPPORTED
    assert cohort_support(2) is CohortSupport.LIMITED
    assert cohort_support(7) is CohortSupport.LIMITED
    assert cohort_support(8) is CohortSupport.SUPPORTED
