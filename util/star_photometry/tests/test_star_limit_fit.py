"""Unit tests for the photometric fit of measured star fields."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from util.star_photometry.star_limit_fit import FitResult, _fit_zero_point

# The planted zero point (2.5 log10 of DN per second for V = 0), filter offset, and gain term.
ZERO_POINT, RED_OFFSET, GAIN_40K = 15.0, 2.0, -1.5


def _zp_row(filter_name: str, gain: str, vmag: float, zp_error: float = 0.0) -> dict[str, Any]:
    """Return a zero-point star row whose DN follows the planted photometry.

    Parameters:
        filter_name: ``'CLEAR'`` or ``'RED'``.
        gain: ``'10K'`` or ``'40K'``.
        vmag: The star's V magnitude.
        zp_error: Magnitudes added to the star's zero point.

    Returns:
        The row.
    """
    zp = ZERO_POINT + zp_error + (GAIN_40K if gain == '40K' else 0.0)
    zp -= RED_OFFSET if filter_name == 'RED' else 0.0
    return {
        'filter': filter_name,
        'gain': gain,
        'vmag': vmag,
        'bmag': None,
        'texp': 1.0,
        'total_dn': 10 ** (0.4 * (zp - vmag)),
    }


def _rows(states: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Return 20 stars in each filter and gain state, with 0.05 mag of scatter.

    Parameters:
        states: ``(filter, gain)`` of each group.

    Returns:
        The rows.
    """
    rng = np.random.default_rng(130)
    return [
        _zp_row(f, g, float(rng.uniform(6.0, 10.0)), float(rng.normal(0.0, 0.05)))
        for f, g in states
        for _ in range(20)
    ]


def _fit(rows: list[dict[str, Any]]) -> FitResult:
    """Fit rows against CLEAR and 10K.

    Parameters:
        rows: The rows.

    Returns:
        The fit.
    """
    result = FitResult(camera='TEST')
    _fit_zero_point(result, rows, reference_filter='CLEAR', reference_gain='10K')
    return result


def test_zero_point_separates_filter_and_gain() -> None:
    """Stars in three filter-gain combinations recover the planted filter offset and gain."""
    result = _fit(_rows([('CLEAR', '10K'), ('RED', '40K'), ('RED', '10K')]))
    assert result.zero_point_identified
    assert result.filter_offsets['RED'] == pytest.approx(RED_OFFSET, abs=0.05)
    assert 2.5 * math.log10(result.gain_ratio['40K']) == pytest.approx(GAIN_40K, abs=0.05)


def test_zero_point_unidentified_when_filter_and_gain_coincide() -> None:
    """Every RED star at 40K and every CLEAR star at 10K cannot separate filter from gain."""
    result = _fit(_rows([('CLEAR', '10K'), ('RED', '40K')]))
    assert not result.zero_point_identified
    assert result.filter_offsets == {}
    assert result.gain_ratio == {}
    assert math.isnan(result.rate_vmag0)


def test_zero_point_unidentified_when_weighting_drops_the_separating_stars() -> None:
    """Two discordant RED stars at 10K, rejected as outliers, leave filter and gain mixed."""
    rows = _rows([('CLEAR', '10K'), ('RED', '40K')])
    rows += [_zp_row('RED', '10K', 8.0, 5.0), _zp_row('RED', '10K', 8.0, -5.0)]
    result = _fit(rows)
    assert not result.zero_point_identified
    assert result.filter_offsets == {}
