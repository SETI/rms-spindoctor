"""Unit tests for the star-peak pointing vote."""

from __future__ import annotations

import numpy as np
import pytest
from util.star_photometry.star_field import PeakList
from util.star_photometry.star_pointing import PointingFix, _refine, find_pointing

# The offset every test plants between predicted and observed positions.
OFFSET_V, OFFSET_U = 10.0, 20.0


def _peak_list(peaks: list[tuple[float, float]]) -> PeakList:
    """Return peaks of equal SNR at the given positions.

    Parameters:
        peaks: Observed ``(v, u)`` of each peak, brightest first.

    Returns:
        The peak list.
    """
    return PeakList(
        v=np.array([p[0] for p in peaks]),
        u=np.array([p[1] for p in peaks]),
        snr=np.full(len(peaks), 50.0),
    )


def _fix(pred: list[tuple[float, float]], peaks: list[tuple[float, float]]) -> PointingFix:
    """Run the vote with settings like the measurement programs' on a 1000-pixel field.

    Parameters:
        pred: Predicted ``(v, u)`` of each star, brightest first.
        peaks: Observed ``(v, u)`` of each peak, brightest first.

    Returns:
        The fix.
    """
    return find_pointing(
        np.array([p[0] for p in pred]),
        np.array([p[1] for p in pred]),
        _peak_list(peaks),
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
    assert (fix.offset_v, fix.offset_u) == pytest.approx((OFFSET_V, OFFSET_U), abs=1e-9)


def test_one_peak_confirms_one_star() -> None:
    """Three predictions within a fraction of a pixel of one peak count as one agreeing star."""
    pred = [(100.0, 100.0), (100.3, 100.0), (100.0, 100.3)]
    fix = _fix(pred, [(100.1 + OFFSET_V, 100.1 + OFFSET_U)])
    assert fix.n_agree == 1
    assert not fix.accepted
    assert fix.reason == 'only 1 stars agree'


def test_refinement_continues_while_a_star_changes_peak() -> None:
    """The refinement stops when the pairs stop changing, not only the agreeing stars.

    Four stars and their peaks, offset along lines only: from a zero seed the fit moves to
    0.6 px, where the first star takes its 0.95 px peak (0.8375 px), and then to 0.8375 px,
    where the second star takes its 1.35 px peak.  The agreeing stars are the same four
    throughout, so a stop on the stars alone would end at 0.8375 px.
    """
    pred = np.array([[100.0, 100.0], [300.0, 100.0], [100.0, 300.0], [300.0, 300.0]])
    peaks = _peak_list(
        [
            (100.0, 100.0),
            (100.95, 100.0),
            (300.0, 100.0),
            (301.35, 100.0),
            (101.2, 300.0),
            (301.2, 300.0),
        ]
    )
    dv, du, _, pairs, _ = _refine(pred[:, 0], pred[:, 1], peaks, 0.0, 0.0, (499.5, 499.5), 1.0)
    assert (dv, du) == pytest.approx((1.175, 0.0), abs=1e-9)
    assert pairs == [(0, 1), (1, 3), (2, 4), (3, 5)]


def test_chance_describes_the_pairs_the_fix_keeps() -> None:
    """A final fix that keeps none of the kept tier's pairs is refused as insignificant.

    Three bright stars agree at the planted offset, the only cluster any tier sees.  Twenty
    faint stars, ranked past every tier by sixty stars far from any peak, agree 1.5 px away
    and pull the final refinement onto their own offset, which drops the bright three.
    """
    bright = [(100.0, 100.0), (100.0, 900.0), (900.0, 500.0)]
    filler = [(2000.0 + 10.0 * k, 2000.0) for k in range(57)]
    faint = [(150.0 + 700.0 * k / 19, 850.0 - 700.0 * k / 19) for k in range(20)]
    peaks = [(v + OFFSET_V, u + OFFSET_U) for v, u in bright]
    peaks += [(v + OFFSET_V + 1.5, u + OFFSET_U) for v, u in faint]
    fix = _fix(bright + filler + faint, peaks)
    assert fix.tier_agree == 3
    assert fix.tier_retained == 0
    assert not fix.accepted
