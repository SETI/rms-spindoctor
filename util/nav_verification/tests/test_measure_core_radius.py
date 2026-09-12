"""Unit tests for the ring-mosaic core measurement."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from util.nav_verification.measure_core_radius import (
    _column_bins,
    median_filter,
    steps_between_adjacent,
)


def header(**keys: object) -> fits.Header:
    """A primary header carrying the given keys.

    Parameters:
        keys: The header keys to set.
    """
    # Long keys become HIERARCH cards, which astropy warns about and the
    # mosaic writer suppresses for the same reason.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        h = fits.Header()
        for key, value in keys.items():
            h[key] = value
    return h


def test_sparse_columns_take_the_longitudes_with_data() -> None:
    """A sparse mosaic keeps one column per longitude it holds data for."""
    antimask = np.array([True, False, True, True, False])
    bins = _column_bins(header(LONGITUDE_RANGE_NONE=True), antimask, 3)
    assert list(bins) == [0, 2, 3]


def test_full_columns_take_every_bin() -> None:
    """A full mosaic keeps every bin of the circle, data or not."""
    antimask = np.array([True, False, True, True, False])
    bins = _column_bins(header(LONGITUDE_RANGE_NONE=True), antimask, 5)
    assert list(bins) == [0, 1, 2, 3, 4]


def test_bounded_columns_start_at_the_range() -> None:
    """A bounded mosaic keeps every bin between two longitudes."""
    antimask = np.zeros(100, dtype=bool)
    bins = _column_bins(header(LONGITUDE_RANGE_0=0.3, LONGITUDE_RESOLUTION=0.1), antimask, 4)
    assert list(bins) == [3, 4, 5, 6]


def test_columns_matching_no_shape_are_refused() -> None:
    """A column count belonging to none of the three shapes has no longitude."""
    antimask = np.array([True, False, True, True, False])
    with pytest.raises(ValueError, match='match neither'):
        _column_bins(header(LONGITUDE_RANGE_NONE=True), antimask, 4)


def test_a_step_between_neighbours_is_found() -> None:
    """Two columns one bin apart that differ are a step."""
    bins = np.array([0, 1, 2])
    values = np.array([0.0, 0.0, 100.0])
    assert list(steps_between_adjacent(bins, values, step_km=25.0)) == [1]


def test_a_jump_across_a_gap_is_not_a_step() -> None:
    """Two columns far apart in longitude are not evidence of a bad frame."""
    bins = np.array([0, 1, 600])
    values = np.array([0.0, 0.0, 100.0])
    assert steps_between_adjacent(bins, values, step_km=25.0).size == 0


def test_a_change_under_the_threshold_is_not_a_step() -> None:
    """Only a change larger than the threshold counts."""
    bins = np.array([0, 1])
    values = np.array([0.0, 10.0])
    assert steps_between_adjacent(bins, values, step_km=25.0).size == 0


def test_median_filter_removes_a_single_spike() -> None:
    """One bad column does not survive to be read as a step."""
    values = np.array([0.0, 0.0, 500.0, 0.0, 0.0])
    assert median_filter(values) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_median_filter_keeps_a_real_step() -> None:
    """A change that persists is not filtered away."""
    values = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    assert median_filter(values)[-1] == pytest.approx(100.0)
