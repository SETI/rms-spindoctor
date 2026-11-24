"""Unit tests for Titan's haze as a function of wavelength and its resolution."""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.instruments.passbands import Passband
from spindoctor.sim.render.bodies.titan_haze import (
    TITAN_MEAN_RADIUS_KM,
    effective_titan_haze,
    methane_band_fraction,
    resolve_model_atmosphere,
    tangent_depth_at_reference,
    uncovered_fraction,
)
from spindoctor.sim.render.bodies.titan_haze_table import TITAN_HAZE_ROWS


def _passband(wavelengths: list[float], weights: list[float]) -> Passband:
    """A passband of the given bins, its weights normalized."""
    weight = np.array(weights, dtype=np.float64)
    return Passband(
        filter_name='TEST',
        wavelength_nm=np.array(wavelengths, dtype=np.float64),
        weight=weight / weight.sum(),
    )


def _titan_body(**extra: Any) -> dict[str, Any]:
    """A spherical body of radius 85 px that takes Titan's haze by name."""
    return {
        'name': 'HAZEMOON',
        'semi_a_px': 85.0,
        'semi_b_px': 85.0,
        'semi_c_px': 85.0,
        'atmosphere': {'haze_model': 'titan'},
        **extra,
    }


def test_the_tangent_depth_at_650_nm_follows_the_disr_power_law() -> None:
    """At 650 nm the haze's tangent optical depth at 140 km is 14.3.

    The optical depth above 80 km, 1.012e7 * 650**-2.339 = 2.66, falls with a 65 km
    scale height to 140 km and is integrated along a tangent path through a 45 km
    scale height.
    """
    depth = float(tangent_depth_at_reference(np.array([650.0]))[0])
    expected = (
        1.012e7
        * 650.0**-2.339
        / 65.0
        * math.exp(-60.0 / 65.0)
        * math.sqrt(2.0 * math.pi * (2575.0 + 140.0) * 45.0)
    )
    assert depth == pytest.approx(expected, rel=1e-12)


def test_a_one_bin_passband_takes_the_table_row() -> None:
    """A passband at one tabulated wavelength takes that row's phase function and albedo."""
    wavelength, g, g2, fraction, albedo = TITAN_HAZE_ROWS[2]
    haze = effective_titan_haze(_passband([wavelength], [1.0]))
    assert (haze.g, haze.g2, haze.g2_fraction, haze.albedo) == pytest.approx(
        (g, g2, fraction, albedo), rel=1e-12
    )


def test_the_optical_depth_averages_in_its_logarithm() -> None:
    """Over two equally weighted bins the optical depth is their geometric mean."""
    depths = tangent_depth_at_reference(np.array([500.0, 800.0]))
    haze = effective_titan_haze(_passband([500.0, 800.0], [1.0, 1.0]))
    assert haze.tangent_depth == pytest.approx(math.sqrt(depths[0] * depths[1]), rel=1e-12)


def test_the_methane_band_fraction_counts_the_band_cores() -> None:
    """A passband with a quarter of its light at 889 nm puts 0.25 in the band cores."""
    passband = _passband([700.0, 889.0], [3.0, 1.0])
    assert methane_band_fraction(passband) == pytest.approx(0.25, rel=1e-12)


def test_the_uncovered_fraction_counts_light_outside_the_table() -> None:
    """A passband with a fifth of its light at 400 nm leaves 0.2 outside the table."""
    passband = _passband([400.0, 600.0], [1.0, 4.0])
    assert uncovered_fraction(passband) == pytest.approx(0.2, rel=1e-12)


def test_a_body_without_a_model_is_returned_unchanged() -> None:
    """A body whose atmosphere states its own keys comes back as the same mapping."""
    body = {'name': 'MOON', 'atmosphere': {'scale_height_px': 4.0, 'tau_ref': 1.0}}
    assert resolve_model_atmosphere(body, instrument='coiss_nac', filter_name=None) is body


def test_a_titan_body_stands_for_titan_at_its_own_size() -> None:
    """Without km_per_pixel the 85 px radius stands for 2575 km, so 45 km is 1.49 px."""
    resolved = resolve_model_atmosphere(_titan_body(), instrument='coiss_nac', filter_name=None)
    expected = 45.0 * 85.0 / TITAN_MEAN_RADIUS_KM
    assert resolved['atmosphere']['scale_height_px'] == pytest.approx(expected, rel=1e-12)


def test_a_titan_body_uses_its_stated_scale() -> None:
    """A body that sets km_per_pixel takes its reference altitude from it."""
    body = _titan_body(km_per_pixel=20.0)
    resolved = resolve_model_atmosphere(body, instrument='coiss_nac', filter_name=None)
    assert resolved['atmosphere']['ref_altitude_px'] == pytest.approx(140.0 / 20.0, rel=1e-12)


def test_the_haze_limb_sits_higher_in_the_blue() -> None:
    """A blue filter sees a thicker haze than a red one, so its tangent depth is larger."""
    blue = resolve_model_atmosphere(_titan_body(), instrument='coiss_wac', filter_name='CL1+BL1')
    red = resolve_model_atmosphere(_titan_body(), instrument='coiss_wac', filter_name='CL1+RED')
    assert blue['atmosphere']['tau_ref'] > 2.0 * red['atmosphere']['tau_ref']
