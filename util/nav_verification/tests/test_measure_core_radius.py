"""Unit tests for the ring-mosaic core measurement."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from util.nav_verification.measure_core_radius import (
    _centroid,
    _column_bins,
    contiguous_runs,
    core_offsets,
    despike_within_runs,
    median_filter,
    steps_between_adjacent,
)

# One bin of 0.02 degrees, as a ring mosaic records it.
BIN_RADIANS = 0.000349065850398865

# The synthetic mosaic below: 200 rows of 5 km from -500 km, so the row at
# radius offset zero is row 100, and 10 columns of the 20-bin circle.
ROWS = 200
COLUMNS = 10
RADIUS_INNER_KM = -500.0
RADIUS_RESOLUTION_KM = 5.0
CORE_ROW = 100
BINS_IN_CIRCLE = 20


def _header(**keys: object) -> fits.Header:
    """A primary header carrying the given keys.

    Parameters:
        keys: The header keys to set.

    Returns:
        The header.
    """
    # Long keys become HIERARCH cards, which astropy warns about and the
    # mosaic writer suppresses for the same reason.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        h = fits.Header()
        for key, value in keys.items():
            h[key] = value
    return h


def _mosaic_file(tmp_path: Path) -> Path:
    """Write a mosaic whose core sits at radius offset zero in every column.

    One column also carries a far brighter feature well outside the core's
    band, where a cosmic ray or a field star leaves one.  The rough pass finds
    that feature in that column, the median over the rough pass is still the
    core, and the refined pass searching a band around that median must come
    back to the core in every column including this one.

    Parameters:
        tmp_path: The directory to write the file into.

    Returns:
        The file.
    """
    image = np.zeros((ROWS, COLUMNS), dtype=float)
    image[CORE_ROW - 1, :] = 1.0
    image[CORE_ROW, :] = 2.0
    image[CORE_ROW + 1, :] = 1.0
    image[189:192, 3] = [50.0, 100.0, 50.0]
    antimask = np.zeros(BINS_IN_CIRCLE, dtype=np.uint8)
    antimask[:COLUMNS] = 1
    path = tmp_path / 'mosaic.fits'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        fits.HDUList(
            [
                fits.PrimaryHDU(
                    header=_header(
                        LONGITUDE_RANGE_NONE=True,
                        LONGITUDE_RESOLUTION=BIN_RADIANS,
                        RADIUS_INNER=RADIUS_INNER_KM,
                        RADIUS_RESOLUTION=RADIUS_RESOLUTION_KM,
                        ORBIT_MODEL_NAME='F-RING-CORE-ALBERS-2007',
                    )
                ),
                fits.ImageHDU(data=image, name='IMG'),
                fits.ImageHDU(data=np.zeros((ROWS, COLUMNS), dtype=np.uint8), name='IMG_MASK'),
                fits.ImageHDU(data=antimask, name='LONGITUDE_ANTIMASK'),
            ]
        ).writeto(path)
    return path


def test_sparse_columns_take_the_longitudes_with_data() -> None:
    """A sparse mosaic keeps one column per longitude it holds data for."""
    antimask = np.array([True, False, True, True, False])
    bins = _column_bins(_header(LONGITUDE_RANGE_NONE=True), antimask, 3)
    assert list(bins) == [0, 2, 3]


def test_full_columns_take_every_bin() -> None:
    """A full mosaic keeps every bin of the circle, data or not."""
    antimask = np.array([True, False, True, True, False])
    bins = _column_bins(_header(LONGITUDE_RANGE_NONE=True), antimask, 5)
    assert list(bins) == [0, 1, 2, 3, 4]


def test_bounded_columns_start_at_the_range() -> None:
    """A bounded mosaic keeps every bin between two longitudes."""
    antimask = np.zeros(100, dtype=bool)
    bins = _column_bins(_header(LONGITUDE_RANGE_0=0.3, LONGITUDE_RESOLUTION=0.1), antimask, 4)
    assert list(bins) == [3, 4, 5, 6]


def test_a_header_naming_no_range_is_read_by_its_shape() -> None:
    """A header carrying neither range card is one of the other two shapes."""
    antimask = np.array([True, False, True, True, False])
    assert list(_column_bins(_header(), antimask, 3)) == [0, 2, 3]


def test_columns_matching_no_shape_are_refused() -> None:
    """A column count belonging to none of the three shapes has no longitude."""
    antimask = np.array([True, False, True, True, False])
    with pytest.raises(ValueError, match='match neither'):
        _column_bins(_header(LONGITUDE_RANGE_NONE=True), antimask, 4)


def test_a_step_between_neighbors_is_found() -> None:
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


def test_runs_end_where_the_longitudes_stop_being_adjacent() -> None:
    """A stretch ends where the next column is more than one bin away."""
    assert contiguous_runs(np.array([0, 1, 600, 601, 602])) == [(0, 2), (2, 5)]


def test_a_value_across_a_gap_does_not_reach_the_filter() -> None:
    """A column is smoothed against the columns that neighbor it in longitude.

    Over storage order the three zeros past the gap outvote this stretch's own
    two columns and erase a value that is really there, which is a step beside
    a gap going unreported.
    """
    bins = np.array([0, 1, 600, 601, 602])
    values = np.array([0.0, 100.0, 0.0, 0.0, 0.0])
    assert despike_within_runs(bins, values)[1] == pytest.approx(100.0)


def test_a_spike_inside_a_run_is_still_removed() -> None:
    """Filtering each stretch on its own does not stop the filter doing its work."""
    bins = np.arange(5)
    values = np.array([0.0, 0.0, 500.0, 0.0, 0.0])
    assert despike_within_runs(bins, values)[2] == pytest.approx(0.0)


def test_a_centroid_weights_the_rows_around_the_peak() -> None:
    """The answer is the brightness-weighted radius of the rows in the window."""
    radii = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    column = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    found = _centroid(column, radii, low_km=100.0, high_km=104.0, half_window=1)
    assert found == pytest.approx(102.0)


def test_a_peak_outside_the_band_is_not_the_one_measured() -> None:
    """The band is what keeps a cosmic ray from being measured instead of the core."""
    radii = np.arange(100.0, 110.0)
    column = np.array([0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0])
    found = _centroid(column, radii, low_km=100.0, high_km=104.0, half_window=1)
    assert found == pytest.approx(102.0)


def test_fewer_than_three_usable_samples_measure_nothing() -> None:
    """Two samples are not enough of a feature to centroid."""
    radii = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    column = np.array([0.0, 0.0, 7.0, 5.0, 0.0])
    found = _centroid(column, radii, low_km=100.0, high_km=104.0, half_window=1)
    assert math.isnan(found)


def test_a_column_that_is_all_gap_measures_nothing() -> None:
    """A column with no brightness anywhere has no peak to find."""
    radii = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    found = _centroid(np.full(5, np.nan), radii, low_km=100.0, high_km=104.0, half_window=1)
    assert math.isnan(found)


def test_the_refined_pass_finds_the_core_in_every_column(tmp_path: Path) -> None:
    """The band around the rough median keeps one column's bright feature out.

    Without the refined pass that column would report the feature's radius
    instead of the core's, which is 450 km away from it.
    """
    profile = core_offsets(_mosaic_file(tmp_path))
    assert profile.core == pytest.approx(np.zeros(COLUMNS))
