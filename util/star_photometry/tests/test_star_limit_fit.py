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
from util.star_photometry.star_zero_point import _tukey_lstsq, fit_star_rates, fit_zero_point

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
        'image_id': 'F0',
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


def _fit(rows: list[dict[str, Any]], *, calibrated: bool = True) -> FitResult:
    """Fit rows against CLEAR and 10K, with a minimum of 15 stars per term.

    Parameters:
        rows: The rows.
        calibrated: Whether RED has a calibration offset (2.5 mag); no other state has one.

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
        calibration_offsets={'RED': 2.5} if calibrated else {},
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


def test_zero_point_keeps_an_exact_majority() -> None:
    """Sixteen stars with one zero point and four 1 mag off give the sixteen's rate."""
    rows = [_zp_row('CLEAR', '10K', 8.0)] * 16 + [_zp_row('CLEAR', '10K', 8.0, 1.0)] * 4
    result = _fit(rows)
    assert result.zero_point_identified
    assert result.rate_vmag0 == pytest.approx(10 ** (0.4 * ZERO_POINT), rel=1e-9)


def test_zero_point_of_identical_stars() -> None:
    """Twenty stars with one zero point give that zero point's rate."""
    result = _fit([_zp_row('CLEAR', '10K', 8.0)] * 20)
    assert result.zero_point_identified
    assert result.rate_vmag0 == pytest.approx(10 ** (0.4 * ZERO_POINT), rel=1e-9)


def test_zero_point_keeps_a_term_whose_stars_scatter_by_nanomagnitudes() -> None:
    """Sixteen stars scattered by 2e-9 mag about their own term keep it, beside 20 exact ones."""
    a = np.array([[1.0, 0.0]] * 20 + [[1.0, 1.0]] * 16)
    b = np.array([10.0] * 20 + [12.0 - 2e-9, 12.0 + 2e-9] * 8)
    x, _, _, weights = _tukey_lstsq(a, b)
    assert x == pytest.approx([10.0, 2.0], abs=1e-6)
    assert np.all(weights > 0.99)


def test_zero_point_drops_a_filter_left_without_its_gain_states() -> None:
    """RED's 15 stars, all at gain states with too few stars, leave the CLEAR fit standing."""
    rows = _rows([('CLEAR', '10K')] * 5)
    rows += _rows([('RED', '40K')])[:8] + _rows([('RED', '100K')])[:7]
    result = _fit(rows, calibrated=False)
    assert result.zero_point_identified
    assert result.n_zp_stars == 100
    assert result.filters_not_measured == ['RED']
    assert result.gains_not_measured == ['100K', '40K']


def test_zero_point_reports_uncalibrated_states_not_measured() -> None:
    """A filter and a gain state with too few stars and no calibration value are reported."""
    rows = _rows([('CLEAR', '10K')]) + _rows([('RED', '10K')])[:3] + _rows([('CLEAR', '40K')])[:3]
    result = _fit(rows, calibrated=False)
    assert result.filters_not_measured == ['RED']
    assert result.gains_not_measured == ['40K']
    assert 'RED' not in result.filter_offsets
    assert '40K' not in result.gain_ratio


def test_zero_point_counts_only_frames_with_a_fitting_star() -> None:
    """A frame whose only stars are through a filter with too few stars is not counted."""
    red = _rows([('RED', '10K')])[:3]
    for row in red:
        row['image_id'] = 'F1'
    result = _fit(_rows([('CLEAR', '10K')]) + red)
    assert result.n_fit_frames == 1


# The planted rate fit: 2.5 log10 of the rate of a V = 0 star of B - V 0.65, its color term.
RATE_ZP, RATE_COLOR = 21.0, 0.3


def _rate_rows(
    n: int, *, gain_offset: float = 0.0, radius_slope: float = 0.0, frame_offset: float = 0.0
) -> list[dict[str, Any]]:
    """Return star rows whose rates follow the planted rate fit, with 0.05 mag of scatter.

    Every other star is at gain ``'12'`` and reads ``gain_offset`` magnitudes brighter;
    every star reads ``radius_slope`` magnitudes brighter for each unit of its ``radius``;
    the stars of even frames read ``frame_offset`` brighter and those of odd frames that
    much fainter.

    Parameters:
        n: The number of stars.
        gain_offset: The planted offset of the gain-12 stars (magnitudes).
        radius_slope: The planted change per unit of radius (magnitudes).
        frame_offset: The planted offset of each frame from the rate (magnitudes).

    Returns:
        The rows, three stars to a frame.
    """
    rng = np.random.default_rng(1294)
    rows = []
    for k in range(n):
        vmag, bv, radius = rng.uniform(10.0, 14.0), rng.uniform(0.2, 1.4), rng.uniform(0.0, 1.0)
        gain = '12' if k % 2 else '29'
        z = RATE_ZP + RATE_COLOR * (bv - 0.65) + rng.normal(0.0, 0.05) + radius_slope * radius
        z += gain_offset if gain == '12' else 0.0
        z += frame_offset if (k // 3) % 2 == 0 else -frame_offset
        rows.append(
            {
                'image_id': f'F{k // 3}',
                'vmag': vmag,
                'bv': bv,
                'gain': gain,
                'radius': radius,
                'rate': 10 ** (0.4 * (z - vmag)),
            }
        )
    return rows


def test_rate_fit_recovers_the_rate_at_the_reference_color() -> None:
    """Stars of many colors give the planted rate of a V = 0 star of B - V 0.65."""
    fit = fit_star_rates(_rate_rows(200), reference_color=0.65)
    assert fit is not None
    assert 2.5 * math.log10(fit.rate_vmag0) == pytest.approx(RATE_ZP, abs=0.02)


def test_rate_fit_recovers_the_color_term() -> None:
    """Redder stars reading brighter give the planted color slope."""
    fit = fit_star_rates(_rate_rows(200), reference_color=0.65)
    assert fit is not None
    assert fit.color_slope == pytest.approx(RATE_COLOR, abs=0.03)


def test_rate_fit_reports_a_group_offset() -> None:
    """Gain-12 stars reading 0.2 mag brighter sit 0.2 mag from the gain-29 stars."""
    fit = fit_star_rates(
        _rate_rows(200, gain_offset=0.2), reference_color=0.65, group_keys=['gain']
    )
    assert fit is not None
    offset = fit.groups['gain']['12'][0] - fit.groups['gain']['29'][0]
    assert offset == pytest.approx(0.2, abs=0.03)


def test_rate_fit_reports_a_covariate_slope() -> None:
    """Stars reading 0.3 mag brighter per unit of radius give a residual slope of 0.3."""
    fit = fit_star_rates(
        _rate_rows(200, radius_slope=0.3), reference_color=0.65, covariates=['radius']
    )
    assert fit is not None
    assert fit.slopes['radius'][0] == pytest.approx(0.3, abs=0.04)


def test_rate_fit_scatter_between_frames() -> None:
    """Twenty frames alternately 0.3 mag bright and faint give a frame scatter of 0.3 mag."""
    fit = fit_star_rates(_rate_rows(60, frame_offset=0.3), reference_color=0.65)
    assert fit is not None
    assert fit.frame_scatter_mag == pytest.approx(0.3, abs=0.03)


def test_rate_fit_uncertainty_from_the_frames() -> None:
    """The rate's frame uncertainty is the frame scatter over the root of the frame count."""
    fit = fit_star_rates(_rate_rows(60, frame_offset=0.3), reference_color=0.65)
    assert fit is not None
    assert fit.frame_scatter_mag is not None
    assert fit.rate_frame_sigma_mag == pytest.approx(fit.frame_scatter_mag / math.sqrt(20))


def test_rate_fit_needs_five_stars() -> None:
    """Four stars give no fit."""
    assert fit_star_rates(_rate_rows(4), reference_color=0.65) is None


def test_rate_fit_refuses_stars_of_one_color() -> None:
    """Stars that all share one color cannot separate the rate from the color term."""
    rows = _rate_rows(20)
    for r in rows:
        r['bv'] = 1.65
    assert fit_star_rates(rows, reference_color=0.65) is None


def test_rate_fit_refuses_when_rejection_leaves_one_color() -> None:
    """Two stars of another color, far off in opposite directions, are rejected."""
    rows = _rate_rows(20)
    for k, r in enumerate(rows):
        r['bv'] = 0.5 if k < 18 else 1.2
        if k >= 18:
            r['rate'] *= 10 ** (4.0 if k == 18 else -4.0)
    assert fit_star_rates(rows, reference_color=0.65) is None


def _terms_result() -> FitResult:
    """Return a fit that measured CLEAR and 10K and took RED and 40K from calibration.

    Returns:
        The fit.
    """
    return FitResult(
        camera='TEST',
        filter_offsets={'CLEAR': 0.0, 'RED': 2.5},
        gain_ratio={'10K': 1.0, '40K': 0.2},
        filters_not_measured=['RED'],
        gains_not_measured=['40K'],
    )


def test_detection_limit_leaves_out_unmeasured_terms() -> None:
    """Stars through an unmeasured filter or gain state are left out."""
    result = _terms_result()
    rows = [
        {'filter': 'CLEAR', 'gain': '10K'},
        {'filter': 'RED', 'gain': '10K'},
        {'filter': 'CLEAR', 'gain': '40K'},
    ]
    assert _measured_terms(result, rows) == [{'filter': 'CLEAR', 'gain': '10K'}]
    assert result.n_limit_stars_unmeasured == 2


def test_detection_limit_leaves_out_terms_without_a_value() -> None:
    """A star through a filter without a fitted or calibration value is left out."""
    result = _terms_result()
    rows = [{'filter': 'CLEAR', 'gain': '10K'}, {'filter': 'VIOLET', 'gain': '10K'}]
    assert _measured_terms(result, rows) == [{'filter': 'CLEAR', 'gain': '10K'}]
    assert result.n_limit_stars_unmeasured == 1


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


def test_rate_fit_refuses_when_rejection_leaves_under_five_stars() -> None:
    """Three exact stars and two opposite outliers at the reference color leave three stars."""
    rows = _rate_rows(5)
    for k, r in enumerate(rows):
        z = RATE_ZP + RATE_COLOR * (r['bv'] - 0.65)
        r['rate'] = 10 ** (0.4 * (z - r['vmag']))
        if k >= 3:
            r['bv'] = 0.65
            r['rate'] = 10 ** (0.4 * (RATE_ZP + (10.0 if k == 3 else -10.0) - r['vmag']))
    assert fit_star_rates(rows, reference_color=0.65) is None
