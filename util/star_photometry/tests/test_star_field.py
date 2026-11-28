"""Unit tests for the star-field image measurements."""

from __future__ import annotations

import numpy as np
import pytest
from util.star_photometry.star_field import clipped_mean


def test_clipped_mean_of_an_integer_sky_is_between_its_levels() -> None:
    """Sixty 10s, forty 11s, and three bright outliers give a sky of 10.4."""
    values = np.array([10.0] * 60 + [11.0] * 40 + [500.0] * 3)
    assert clipped_mean(values) == pytest.approx(10.4)


def test_clipped_mean_keeps_the_level_next_to_the_sky() -> None:
    """Ninety-five 10s and five 11s give a sky of 10.05, not 10."""
    values = np.array([10.0] * 95 + [11.0] * 5)
    assert clipped_mean(values) == pytest.approx(10.05)
