"""Unit tests for the Cassini ISS star zero-point measurement."""

from __future__ import annotations

from typing import Any

import pytest
from util.star_photometry.cassini_iss_star_zero_points import (
    StarRows,
    _eligible,
    _saturation_dn,
    calibration_terms,
    star_rows,
)

# The calibrated label's terms every synthetic frame carries.
GAIN, OFFSET_S, TEXP = 30.27, 0.00275, 1.0

LABEL = """
  GAIN_CORRECTION = '30.27 e-/DN'
  EXPOSURE_CORRECTION_FLAG = 1
  SHUTTER_OFFSET_FILE_NAME = 'nacfm_so_p5.img'
  EXPOSURE_OFFSET = '2.75 ms'
"""


def _star(**changes: Any) -> dict[str, Any]:
    """Return a clean V 12 star of 1000 raw DN within r and 1100 within 2r, at SNR 110.

    Parameters:
        **changes: Fields to replace.

    Returns:
        The star's record.
    """
    star = {
        'catalog': 'ucac4',
        'vmag': 12.0,
        'johnson_v': 12.0,
        'bmag': 12.65,
        'v': 100.0,
        'u': 200.0,
        'conflict': '',
        'blended': False,
        'smeared': False,
        'smear_px': 0.1,
        'no_data': False,
        'raw_r': 1000.0,
        'raw_2r': 1100.0,
        'raw_3r': 1120.0,
        'raw_sky': 20.0,
        'raw_sky_sigma': 1.0,
        'raw_npix': 50,
        'raw_max': 300.0,
        'flux_big': 1100.0,
        'flux_big_sigma': 10.0,
        'cal_r': 1.0e-3,
        'cal_2r': 1.1e-3,
        'cal_3r': 1.12e-3,
    }
    star.update(changes)
    return star


def _record(
    stars: list[dict[str, Any]],
    *,
    conversion: str = '12BIT',
    texp: float = TEXP,
    spread_px: float = 0.2,
    binning: int = 1,
) -> dict[str, Any]:
    """Return a NAC clear-filter record whose pointing fix was accepted.

    Parameters:
        stars: The frame's stars.
        conversion: The frame's ``DATA_CONVERSION_TYPE``.
        texp: The exposure (seconds).
        spread_px: The RMS of the pointing fix's agreeing stars (frame pixels).
        binning: The summation factor.

    Returns:
        The record.
    """
    return {
        'frame': {
            'image_id': 'N1',
            'camera': 'NAC',
            'filter_name': 'CL1+CL2',
            'gain_state': '29',
            'mode': {1: 'FULL', 2: 'SUM2'}[binning],
            'texp': texp,
            'extra': {
                'source': 'sample',
                'image_time': '2008-001T00:00:00',
                'conversion': conversion,
            },
        },
        'fix': {'accepted': True, 'n_agree': 10, 'spread_px': spread_px},
        'shape': [1024 // binning, 1024 // binning],
        'binning': binning,
        'gain_e_per_dn': GAIN,
        'exposure_offset_s': OFFSET_S,
        'sun_distance_au': 9.5,
        'stars': stars,
    }


def _rows(record: dict[str, Any]) -> StarRows:
    """Return the NAC clear-filter fitting stars of one record at a rate of 2.3e8 e-/s.

    Parameters:
        record: The frame's record.

    Returns:
        The selection.
    """
    return star_rows([record], 'NAC', rate=2.3e8, filters=frozenset(['CL1+CL2']))


def test_calibration_terms_read_the_gain_and_the_shutter_offset() -> None:
    """The label's GAIN_CORRECTION and EXPOSURE_OFFSET give e-/DN and seconds."""
    assert calibration_terms(LABEL) == pytest.approx((GAIN, OFFSET_S))


def test_calibration_terms_refuse_a_label_without_them() -> None:
    """A label stating neither term is refused."""
    with pytest.raises(ValueError, match='states no GAIN_CORRECTION or EXPOSURE_OFFSET'):
        calibration_terms("UNITS = 'I/F'")


def test_saturation_at_gain_state_2_is_the_full_well() -> None:
    """At 30.27 e-/DN the 110,000 e- full well fills before the ADC."""
    assert _saturation_dn('NAC', GAIN, 1, 20.0) == pytest.approx(110000.0 / GAIN)


def test_saturation_at_gain_state_3_is_the_adc() -> None:
    """At 12 e-/DN the ADC's 4095 DN, less the sky, comes first."""
    assert _saturation_dn('NAC', 12.0, 1, 20.0) == pytest.approx(4075.0)


def test_rate_is_the_corrected_light_over_the_effective_exposure() -> None:
    """A star's rate is its DN within r times the aperture correction and the gain, per second."""
    selected = _rows(_record([_star(), _star(u=400.0)]))
    assert selected.aperture_correction['FULL'] == pytest.approx(1.1)
    assert selected.rows[0]['rate'] == pytest.approx(1000.0 * 1.1 * GAIN / (TEXP - OFFSET_S))


def test_aperture_correction_leaves_out_a_star_under_snr_100() -> None:
    """A star whose recorded flux within 2r is at SNR 75 does not set the correction."""
    faint = _star(u=400.0, raw_2r=1500.0, flux_big=1500.0, flux_big_sigma=20.0)
    selected = _rows(_record([_star(), faint]))
    assert selected.aperture_correction['FULL'] == pytest.approx(1.1)


def test_a_table_frame_is_not_fitted() -> None:
    """The stars of a frame sent through the downlink table are counted and left out."""
    selected = _rows(_record([_star(), _star(u=400.0)], conversion='TABLE'))
    assert selected.excluded['frame_not_12_bit'] == 2


def test_a_smeared_star_is_not_fitted() -> None:
    """A star predicted to move more than a pixel is counted as smeared and left out."""
    selected = _rows(_record([_star(), _star(u=400.0, smeared=True, smear_px=3.0)]))
    assert selected.excluded['smeared'] == 1
    assert len(selected.rows) == 1


def test_an_elongated_frame_is_not_fitted() -> None:
    """A frame whose stars' PSF fits are twice as long as wide is counted as elongated."""
    psf = {'sigma_major': 2.0, 'sigma_minor': 1.0}
    stars = [_star(u=200.0 + 100.0 * k, psf=psf) for k in range(3)]
    selected = _rows(_record(stars))
    assert selected.excluded['frame_elongated'] == 3


def test_a_star_predicted_too_faint_is_not_fitted() -> None:
    """A V 14 star in a 1 s frame, predicted under SNR 10 at the provisional rate, is left out."""
    selected = _rows(_record([_star(), _star(u=400.0, johnson_v=14.0, bmag=14.65)]))
    assert selected.excluded['predicted_snr_low'] == 1
    assert len(selected.rows) == 1


def test_saturation_of_a_summed_pixel_holds_four_wells() -> None:
    """A 2x2-summed pixel holds four full wells, so at 30.27 e-/DN the ADC comes first."""
    assert _saturation_dn('NAC', GAIN, 2, 20.0) == pytest.approx(4075.0)


def test_a_summed_frame_calibrated_rate_counts_four_native_pixels() -> None:
    """A 2x2-summed star of the same I/F sum reads four times the calibrated rate."""
    full = _rows(_record([_star(), _star(u=400.0)]))
    summed = _rows(_record([_star(), _star(u=400.0)], binning=2))
    assert summed.rows[0]['rate_cal'] / full.rows[0]['rate_cal'] == pytest.approx(4.0)


def test_a_star_predicted_near_saturation_is_not_fitted() -> None:
    """A V 10 star in a 20 s frame, predicted past 60% of saturation, is left out."""
    selected = _rows(_record([_star(), _star(u=400.0, johnson_v=10.0, bmag=10.65)], texp=20.0))
    assert selected.excluded['predicted_near_saturation'] == 1


def test_a_star_measured_saturated_is_not_fitted() -> None:
    """A star whose peak above the sky reaches 95% of the full well is left out."""
    peak = 20.0 + 0.95 * 110000.0 / GAIN
    selected = _rows(_record([_star(), _star(u=400.0), _star(u=600.0, raw_max=peak)]))
    assert selected.excluded['measured_saturated'] == 1


def test_a_loose_pointing_fix_is_not_fitted() -> None:
    """The stars of a frame whose fix's agreeing stars spread 0.8 px are left out."""
    selected = _rows(_record([_star(), _star(u=400.0)], spread_px=0.8))
    assert selected.excluded['frame_fix_loose'] == 2


def test_a_mode_without_an_aperture_correction_is_not_fitted() -> None:
    """The stars of a summed frame with no star at SNR 100 within 2r are left out."""
    faint = {'flux_big_sigma': 20.0}
    record = _record([_star(**faint), _star(u=400.0, **faint)], binning=2)
    selected = _rows(record)
    assert selected.excluded['mode_without_aperture_correction'] == 2


def _index_row(image_time: str) -> dict[str, Any]:
    """Return an index row of a sampleable 1 s clear-filter sky frame.

    Parameters:
        image_time: The row's ``IMAGE_TIME``.

    Returns:
        The row.
    """
    return {
        'TARGET_NAME': 'SKY',
        'DATA_CONVERSION_TYPE': '12BIT',
        'SHUTTER_STATE_ID': 'ENABLED',
        'CALIBRATION_LAMP_STATE_FLAG': 'OFF',
        'FILTER_NAME': 'CL1+CL2',
        'IMAGE_TIME': image_time,
        'EXPOSURE_DURATION': 1000.0,
    }


def test_a_frame_on_the_last_spice_day_is_sampled() -> None:
    """A frame taken on 2017 September 15 (day 258) lies inside the SPICE window."""
    assert _eligible(_index_row('2017-258T10:00:00.000'), frozenset(['CL1+CL2']))


def test_a_frame_a_day_after_the_spice_window_is_not_sampled() -> None:
    """A frame taken on 2017 September 16 (day 259) lies outside the SPICE window."""
    assert not _eligible(_index_row('2017-259T10:00:00.000'), frozenset(['CL1+CL2']))
