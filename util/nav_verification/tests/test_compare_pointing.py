"""Unit tests for the per-frame comparison's arithmetic."""

from __future__ import annotations

from typing import Any

import pytest
from util.nav_verification.compare_pointing import (
    FrameComparison,
    consensus_spread,
    remove_common_offset,
)


def navigation(*techniques: dict[str, Any], excluded: list[str] | None = None) -> dict[str, Any]:
    """A navigation result holding the given per-technique answers.

    Parameters:
        techniques: The per-technique entries.
        excluded: Technique names the ensemble threw out as outliers.
    """
    return {
        'per_technique': list(techniques),
        'excluded_from_consensus': excluded or [],
    }


def technique(name: str, offset: list[float], *, spurious: bool = False) -> dict[str, Any]:
    """One per-technique answer.

    Parameters:
        name: The technique's name.
        offset: The offset it proposed.
        spurious: Whether the pass called it spurious.
    """
    return {'technique_name': name, 'offset_px': offset, 'spurious': spurious}


def test_spread_of_one_contributor_is_zero() -> None:
    """A frame one technique answered has nothing to disagree with."""
    assert consensus_spread(navigation(technique('A', [1.0, 2.0])))[0] == 0.0


def test_spread_counts_the_contributors() -> None:
    """The count is of techniques that formed the answer."""
    result = navigation(technique('A', [0.0, 0.0]), technique('B', [3.0, 4.0]))
    assert consensus_spread(result)[1] == 2


def test_spread_measures_the_widest_pair() -> None:
    """The spread is the largest distance between any two contributors."""
    result = navigation(
        technique('A', [0.0, 0.0]), technique('B', [3.0, 4.0]), technique('C', [0.0, 1.0])
    )
    assert consensus_spread(result)[0] == pytest.approx(5.0)


def test_an_outlier_the_ensemble_threw_out_does_not_widen_the_spread() -> None:
    """A technique excluded from consensus is not part of the answer.

    Without this, one rejected technique hundreds of pixels away puts a spread
    of hundreds of pixels on a frame that one technique answered cleanly.
    """
    result = navigation(
        technique('A', [0.0, 0.0]),
        technique('B', [-281.5, -519.6]),
        excluded=['B'],
    )
    assert consensus_spread(result)[0] == 0.0


def test_a_spurious_technique_does_not_widen_the_spread() -> None:
    """A technique the pass called spurious is not part of the answer either."""
    result = navigation(technique('A', [0.0, 0.0]), technique('B', [100.0, 100.0], spurious=True))
    assert consensus_spread(result)[0] == 0.0


def comparison(x: float, y: float, error: float) -> FrameComparison:
    """A comparison carrying one measured difference.

    Parameters:
        x: The difference along the camera's first axis.
        y: The difference along its second.
        error: The unsigned difference.
    """
    return FrameComparison(image='N1', error_px=error, error_x_px=x, error_y_px=y)


def test_a_constant_offset_is_measured_and_removed() -> None:
    """A difference every frame shares is a datum, and comes out of the errors."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    common = remove_common_offset(rows, tolerance_px=2.0)
    assert common == pytest.approx((0.5, 0.5))


def test_what_is_left_after_a_constant_is_the_per_frame_difference() -> None:
    """A frame that carries only the constant disagrees by nothing."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[0].residual_px == pytest.approx(0.0, abs=1e-9)


def test_a_badly_navigated_frame_does_not_move_the_constant() -> None:
    """The constant is a median over frames that already agree."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(comparison(60.0, 60.0, 84.85))
    common = remove_common_offset(rows, tolerance_px=2.0)
    assert common == pytest.approx((0.5, 0.5))


def test_a_badly_navigated_frame_keeps_its_error() -> None:
    """Removing the constant does not excuse a frame that is genuinely wrong."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(comparison(60.0, 60.0, 84.85))
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[-1].residual_px == pytest.approx(84.15, rel=1e-3)


def test_too_few_agreeing_frames_measure_no_constant() -> None:
    """A median over three numbers is three numbers, not a datum."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(3)]
    assert remove_common_offset(rows, tolerance_px=2.0) is None


def test_with_no_constant_the_residual_is_the_error() -> None:
    """A run too small to measure a datum reports what it measured."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(3)]
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[0].residual_px == pytest.approx(0.707)


def test_a_frame_with_nothing_to_compare_gets_no_residual() -> None:
    """A frame the bundle holds no answer for is left alone."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(FrameComparison(image='N2'))
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[-1].residual_px is None
