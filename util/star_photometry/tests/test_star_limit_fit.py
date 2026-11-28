"""Unit tests for the photometric fit of measured star fields."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.special
from util.star_photometry.star_limit_fit import (
    FitResult,
    _dimmest_visible,
    _fit_detection_limit,
    _measured_terms,
    _predict_dn,
    _set_camera_mags,
    fit_camera,
    load_records,
)
from util.star_photometry.star_zero_point import _tukey_lstsq, fit_zero_point

# The planted zero point (2.5 log10 of DN per second for V = 0), filter offset, gain term,
# and color term.
ZERO_POINT, RED_OFFSET, GAIN_40K, COLOR_TERM = 15.0, 2.0, -1.5, -0.4


def _zp_row(
    filter_name: str,
    gain: str,
    vmag: float,
    zp_error: float = 0.0,
    b_minus_v: float | None = None,
) -> dict[str, Any]:
    """Return a zero-point star row whose DN follows the planted photometry.

    Parameters:
        filter_name: ``'CLEAR'`` or ``'RED'``.
        gain: ``'10K'`` or ``'40K'``.
        vmag: The star's V magnitude.
        zp_error: Magnitudes added to the star's zero point.
        b_minus_v: The star's measured color, which dims it by ``COLOR_TERM`` times
            itself, or None for a star without a measured color.

    Returns:
        The row.
    """
    zp = ZERO_POINT + zp_error + (GAIN_40K if gain == '40K' else 0.0)
    zp -= RED_OFFSET if filter_name == 'RED' else 0.0
    zp -= 0.0 if b_minus_v is None else COLOR_TERM * b_minus_v
    return {
        'image_id': 'F0',
        'filter': filter_name,
        'gain': gain,
        'vmag': vmag,
        'bmag': None if b_minus_v is None else vmag + b_minus_v,
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


def _fit(
    rows: list[dict[str, Any]], *, calibrated: bool = True, color_term: bool = False
) -> FitResult:
    """Fit rows against CLEAR and 10K, with a minimum of 15 stars per term.

    Parameters:
        rows: The rows.
        calibrated: Whether RED has a calibration offset (2.5 mag); no other state has one.
        color_term: Whether to fit the color term.

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
        color_term=color_term,
    )
    return result


def _colored_rows() -> list[dict[str, Any]]:
    """Return 30 CLEAR stars at 10K with colors from 0 to 1.5, and 10 without a color.

    Returns:
        The rows.
    """
    rng = np.random.default_rng(130)
    colored = [
        _zp_row(
            'CLEAR',
            '10K',
            float(rng.uniform(6.0, 10.0)),
            float(rng.normal(0.0, 0.02)),
            b_minus_v=float(rng.uniform(0.0, 1.5)),
        )
        for _ in range(30)
    ]
    return colored + [_zp_row('CLEAR', '10K', 8.0, 1.0) for _ in range(10)]


def test_zero_point_fits_a_planted_color_term() -> None:
    """Stars whose DN follows V + c (B - V) give back c."""
    assert _fit(_colored_rows(), color_term=True).color_term == pytest.approx(COLOR_TERM, abs=0.05)


def test_color_term_fit_leaves_out_stars_without_a_color() -> None:
    """Only the stars with a measured color enter a fit of the color term."""
    assert _fit(_colored_rows(), color_term=True).n_zp_stars == 30


def test_color_term_fit_needs_the_minimum_of_colored_stars() -> None:
    """Two CLEAR stars left after 13 RED stars are pruned cannot carry a color term."""
    rows = _colored_rows()[:2] + [
        _zp_row('RED', '10K', 8.0 + 0.1 * k, b_minus_v=0.1 * k) for k in range(13)
    ]
    assert not _fit(rows, color_term=True).zero_point_identified


def test_predicted_dn_reads_the_camera_magnitude() -> None:
    """A star of V 10 and B - V 1 is predicted at camera magnitude 9.6 for c = -0.4."""
    result = FitResult(
        camera='TEST',
        rate_vmag0=1.0e6,
        color_term=COLOR_TERM,
        filter_offsets={'CLEAR': 0.0},
        gain_ratio={'10K': 1.0},
    )
    rows = [{'vmag': 10.0, 'b_minus_v': 1.0, 'filter': 'CLEAR', 'gain': '10K', 'texp': 1.0}]
    _set_camera_mags(result, rows)
    assert _predict_dn(result, rows[0]) == pytest.approx(1.0e6 * 10 ** (-0.4 * 9.6))


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


def test_dimmest_visible_reads_the_limit_of_the_frames_mode_and_gain_state() -> None:
    """A frame of a pair with its own limit takes it in place of its mode's."""
    result = FitResult(
        camera='TEST',
        rate_vmag0=1.0e6,
        filter_offsets={'CLEAR': 0.0},
        gain_ratio={'10K': 1.0},
        min_detectable_dn_by_mode={'FULL': [1000.0]},
        min_detectable_dn_by_mode_gain={'FULL': {'10K': [100.0]}},
    )
    assert _dimmest_visible(result, 'CLEAR', '10K', 1.0, 'FULL') == pytest.approx(2.5 * 4.0)


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


def test_load_records_refuses_a_frame_measured_without_star_colors(tmp_path: Path) -> None:
    """A frame whose stars lack ``b_minus_v`` is refused with the remedy in the message."""
    path = tmp_path / 'records.jsonl'
    record = {'frame': {'image_id': 'F0'}, 'stars': [{'vmag': 12.0}]}
    path.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='frame F0 has stars without b_minus_v') as exc_info:
        load_records(path)
    assert 'run the measurement again' in str(exc_info.value)
