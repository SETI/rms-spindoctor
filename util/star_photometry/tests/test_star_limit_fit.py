"""Unit tests for the photometric fit of measured star fields."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import scipy.special
from util.star_photometry.star_limit_fit import (
    FitResult,
    _fit_detection_limit,
    _measured_terms,
    fit_camera,
)
from util.star_photometry.star_zero_point import fit_zero_point

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
    fit_zero_point(
        result,
        rows,
        reference_filter='CLEAR',
        reference_gain='10K',
        min_term_stars=15,
        calibration_offsets={'RED': 2.5},
        calibration_gain_ratios={},
    )
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


def test_zero_point_fixes_a_filter_with_too_few_stars() -> None:
    """Three RED stars cannot pin down RED's offset, which takes the calibration value."""
    rows = _rows([('CLEAR', '10K'), ('CLEAR', '40K')]) + _rows([('RED', '10K')])[:3]
    result = _fit(rows)
    assert result.filter_offsets['RED'] == 2.5
    assert result.filters_not_measured == ['RED']


def test_detection_limit_leaves_out_unmeasured_terms() -> None:
    """A star through a filter or gain state the zero point did not measure is left out."""
    result = FitResult(camera='TEST', filters_not_measured=['RED'], gains_not_measured=['40K'])
    rows = [
        {'filter': 'CLEAR', 'gain': '10K'},
        {'filter': 'RED', 'gain': '10K'},
        {'filter': 'CLEAR', 'gain': '40K'},
    ]
    assert _measured_terms(result, rows) == [{'filter': 'CLEAR', 'gain': '10K'}]
    assert result.n_limit_stars_unmeasured == 2


def _limit_row(pred_dn: float, detected: bool, k: int) -> dict[str, Any]:
    """Return a detection-limit star row of readout mode ``'FULL'``, five stars per frame.

    Parameters:
        pred_dn: The star's predicted DN.
        detected: Whether it was detected.
        k: The star's number, which sets its frame.

    Returns:
        The row.
    """
    return {'pred_dn': pred_dn, 'detected': detected, 'mode': 'FULL', 'image_id': f'F{k // 5}'}


def _limit_fit(rows: list[dict[str, Any]]) -> FitResult:
    """Fit the detection limit of rows.

    Parameters:
        rows: The rows.

    Returns:
        The fit.
    """
    result = FitResult(camera='TEST')
    _fit_detection_limit(result, rows)
    return result


def test_detection_limit_found_across_the_rise() -> None:
    """Stars from 1 to 10000 DN whose detection rises around 100 DN give the 80% point."""
    rng = np.random.default_rng(130)
    log_dn = rng.uniform(0.0, 4.0, 400)
    detected = rng.uniform(size=400) < 0.5 * (1.0 + scipy.special.erf((log_dn - 2.0) / 0.2))
    pairs = enumerate(zip(log_dn, detected, strict=True))
    result = _limit_fit([_limit_row(10**x, bool(d), k) for k, (x, d) in pairs])
    assert result.min_detectable_dn == pytest.approx(10**2.12, rel=0.2)
    assert result.min_detectable_dn_by_mode['FULL'][0] == pytest.approx(10**2.12, rel=0.2)
    assert result.min_detectable_dn_insufficient == {}


def test_detection_limit_undefined_with_detections_at_one_dn() -> None:
    """Forty-five stars detected at one predicted DN cannot place the rise."""
    result = _limit_fit([_limit_row(100.0, True, k) for k in range(45)])
    assert math.isnan(result.min_detectable_dn)
    assert result.min_detectable_dn_by_mode == {}
    assert result.min_detectable_dn_insufficient == {
        'all': '45 detected and 0 undetected stars',
        'FULL': '45 detected and 0 undetected stars',
    }


def test_detection_limit_undefined_over_a_narrow_dn_range() -> None:
    """Detections and non-detections within 0.1 decade of each other cannot show the rise."""
    rows = [_limit_row(100.0 + k % 25, k % 2 == 0, k) for k in range(60)]
    result = _limit_fit(rows)
    assert math.isnan(result.min_detectable_dn)
    assert result.min_detectable_dn_insufficient['all'].startswith('predicted DN spread')


def test_detection_limit_undefined_without_a_rise() -> None:
    """Half the stars detected at every DN leaves the fitted rise at the edge of the data."""
    rows = [_limit_row(10 ** (k / 50.0), k % 2 == 0, k) for k in range(200)]
    result = _limit_fit(rows)
    assert math.isnan(result.min_detectable_dn)
    assert result.min_detectable_dn_insufficient['all'] == (
        'the fitted rise lies at the edge of the data'
    )


def test_weak_fit_keeps_the_frame_table() -> None:
    """A camera with too few fitting stars still lists its accepted frames, fit columns NaN."""
    records = [
        {
            'frame': {
                'image_id': f'F{k}',
                'filter_name': 'CLEAR',
                'gain_state': '10K',
                'mode': 'FULL',
                'texp': 1.0,
                'extra': {},
            },
            'fix': {'accepted': True, 'n_agree': 3},
            'current_limit': 10.0,
            'deep_limit': 14.0,
            'stars': [],
        }
        for k in range(2)
    ]
    result = fit_camera('TEST', records, reference_filter='CLEAR', reference_gain='10K')
    assert [row['image_id'] for row in result.frames] == ['F0', 'F1']
    assert math.isnan(result.frames[0]['proposed_limit'])
