"""Unit tests for FOM 5 dynamic-range stats and exposure stratification."""

from __future__ import annotations

import numpy as np
import pytest
from util.sim_realism.dynamic_range import (
    SIGNAL_PERCENTILES,
    frame_dynamic_range,
    stratify_by_exposure,
)


def test_frame_dynamic_range_saturation_fraction() -> None:
    """Pixels at or above the saturation level are counted exactly."""
    image = np.zeros((10, 10))
    image[:2, :5] = 4095.0
    stats = frame_dynamic_range(image, saturation_level=4095.0, noise_sigma=1.0)
    assert stats.frac_saturated == pytest.approx(0.1)


def test_frame_dynamic_range_near_floor_fraction() -> None:
    """The near-floor fraction counts pixels within one sigma of the floor."""
    rng = np.random.default_rng(21)
    image = rng.normal(20.0, 2.0, (100, 100))
    image[:50] += 500.0  # upper half far from floor
    stats = frame_dynamic_range(image, saturation_level=4095.0, noise_sigma=2.0)
    assert 0.0 < stats.frac_near_floor < 0.5


def test_frame_dynamic_range_percentile_count() -> None:
    """One percentile value per configured level."""
    stats = frame_dynamic_range(np.arange(100.0), saturation_level=99.0, noise_sigma=0.0)
    assert len(stats.percentiles) == len(SIGNAL_PERCENTILES)
    assert stats.percentiles[3] == pytest.approx(49.5)


def test_frame_dynamic_range_empty_is_nan() -> None:
    """An all-NaN frame yields NaN statistics."""
    stats = frame_dynamic_range(np.full((4, 4), np.nan), saturation_level=1.0, noise_sigma=1.0)
    assert np.isnan(stats.frac_saturated)


def test_stratify_by_exposure_default_edges() -> None:
    """Frames land in the documented strata; None goes to 'unknown'."""
    strata = stratify_by_exposure([0.01, 0.1, 1.0, 10.0, None, 0.4])
    assert strata['lt_0.05s'] == [0]
    assert strata['0.05s_to_0.5s'] == [1, 5]
    assert strata['0.5s_to_5s'] == [2]
    assert strata['ge_5s'] == [3]
    assert strata['unknown'] == [4]


def test_stratify_by_exposure_boundary_goes_up() -> None:
    """An exposure exactly on an edge lands in the higher stratum."""
    strata = stratify_by_exposure([0.5])
    assert strata == {'0.5s_to_5s': [0]}


def test_stratify_by_exposure_omits_empty_strata() -> None:
    """Only populated strata appear in the mapping."""
    strata = stratify_by_exposure([10.0])
    assert list(strata) == ['ge_5s']
