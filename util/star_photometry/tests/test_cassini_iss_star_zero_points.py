"""Unit tests for the Cassini ISS star zero-point measurement."""

from __future__ import annotations

from typing import Any

import pytest
from util.star_photometry.cassini_iss_star_zero_points import (
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
    """Return a clean V 12 star of 1000 raw DN within r and 1100 within 2r.

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
        'cal_r': 1.0e-3,
        'cal_2r': 1.1e-3,
        'cal_3r': 1.12e-3,
    }
    star.update(changes)
    return star


def _record(stars: list[dict[str, Any]], *, conversion: str = '12BIT') -> dict[str, Any]:
    """Return a full-frame NAC clear-filter record whose pointing fix was accepted.

    Parameters:
        stars: The frame's stars.
        conversion: The frame's ``DATA_CONVERSION_TYPE``.

    Returns:
        The record.
    """
    return {
        'frame': {
            'image_id': 'N1',
            'camera': 'NAC',
            'filter_name': 'CL1+CL2',
            'gain_state': '29',
            'mode': 'FULL',
            'texp': TEXP,
            'extra': {
                'source': 'sample',
                'image_time': '2008-001T00:00:00',
                'conversion': conversion,
            },
        },
        'fix': {'accepted': True, 'n_agree': 10, 'spread_px': 0.2},
        'shape': [1024, 1024],
        'binning': 1,
        'gain_e_per_dn': GAIN,
        'exposure_offset_s': OFFSET_S,
        'sun_distance_au': 9.5,
        'stars': stars,
    }


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
    rows, apcorr, _, _ = star_rows(
        [_record([_star(), _star(u=400.0)])], 'NAC', rate=2.3e8, filters=frozenset(['CL1+CL2'])
    )
    assert apcorr['FULL'] == pytest.approx(1.1)
    assert rows[0]['rate'] == pytest.approx(1000.0 * 1.1 * GAIN / (TEXP - OFFSET_S))


def test_a_table_frame_is_not_fitted() -> None:
    """The stars of a frame sent through the downlink table are counted and left out."""
    _, _, _, excluded = star_rows(
        [_record([_star(), _star(u=400.0)], conversion='TABLE')],
        'NAC',
        rate=2.3e8,
        filters=frozenset(['CL1+CL2']),
    )
    assert excluded['frame_not_12_bit'] == 2


def test_a_smeared_star_is_not_fitted() -> None:
    """A star predicted to move more than a pixel is counted as smeared and left out."""
    rows, _, _, excluded = star_rows(
        [_record([_star(), _star(u=400.0, smeared=True, smear_px=3.0)])],
        'NAC',
        rate=2.3e8,
        filters=frozenset(['CL1+CL2']),
    )
    assert excluded['smeared'] == 1
    assert len(rows) == 1


def test_an_elongated_frame_is_not_fitted() -> None:
    """A frame whose stars' PSF fits are twice as long as wide is counted as elongated."""
    psf = {'sigma_major': 2.0, 'sigma_minor': 1.0}
    stars = [_star(u=200.0 + 100.0 * k, psf=psf) for k in range(3)]
    _, _, _, excluded = star_rows(
        [_record(stars)], 'NAC', rate=2.3e8, filters=frozenset(['CL1+CL2'])
    )
    assert excluded['frame_elongated'] == 3


def test_a_star_predicted_too_faint_is_not_fitted() -> None:
    """A V 14 star in a 1 s frame, predicted under SNR 10 at the provisional rate, is left out."""
    rows, _, _, excluded = star_rows(
        [_record([_star(), _star(u=400.0, johnson_v=14.0, bmag=14.65)])],
        'NAC',
        rate=2.3e8,
        filters=frozenset(['CL1+CL2']),
    )
    assert excluded['predicted_snr_low'] == 1
    assert len(rows) == 1
