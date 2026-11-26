"""Unit tests for the star-peak pointing vote."""

from __future__ import annotations

import numpy as np
from util.star_photometry.star_field import PeakList
from util.star_photometry.star_pointing import PointingFix, find_pointing

# The offset every test plants between predicted and observed positions.
OFFSET_V, OFFSET_U = 10.0, 20.0


def _fix(pred: list[tuple[float, float]], peaks: list[tuple[float, float]]) -> PointingFix:
    """Run the vote with settings like the measurement programs' on a 1000-pixel field.

    Parameters:
        pred: Predicted ``(v, u)`` of each star, brightest first.
        peaks: Observed ``(v, u)`` of each peak, brightest first.

    Returns:
        The fix.
    """
    peak_list = PeakList(
        v=np.array([p[0] for p in peaks]),
        u=np.array([p[1] for p in peaks]),
        snr=np.full(len(peaks), 50.0),
    )
    return find_pointing(
        np.array([p[0] for p in pred]),
        np.array([p[1] for p in pred]),
        peak_list,
        center_vu=(499.5, 499.5),
        search_px=50.0,
        vote_bin_px=1.5,
        agree_px=1.0,
        min_agree=3,
        max_chance_probability=1e-3,
    )


def test_three_separate_stars_agree() -> None:
    """Three stars, each on its own peak at the planted offset, give an accepted fix."""
    pred = [(100.0, 100.0), (300.0, 100.0), (100.0, 300.0)]
    fix = _fix(pred, [(v + OFFSET_V, u + OFFSET_U) for v, u in pred])
    assert fix.accepted
    assert fix.n_agree == 3


def test_one_peak_confirms_one_star() -> None:
    """Three predictions within a fraction of a pixel of one peak count as one agreeing star."""
    pred = [(100.0, 100.0), (100.3, 100.0), (100.0, 100.3)]
    fix = _fix(pred, [(100.1 + OFFSET_V, 100.1 + OFFSET_U)])
    assert fix.n_agree == 1
    assert not fix.accepted
    assert fix.reason == 'only 1 stars agree'
