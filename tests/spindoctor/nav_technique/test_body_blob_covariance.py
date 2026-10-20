"""Tests for ``BodyBlobNav``'s per-blob covariance and generalized-least-squares fit."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from spindoctor.nav_technique.body_blob_covariance import (
    BlobCentroidBiasModel,
    _along_sun_fractions,
    _blob_covariances,
    _generalized_least_squares,
    _lambert_photocenter_fraction,
    load_blob_bias_model,
)

_MODEL = BlobCentroidBiasModel(
    size_frac=0.05,
    along_sun_frac_regular=0.9,
    along_sun_frac_irregular=0.4,
    across_sun_frac=0.2,
    regular_irregularity_max=0.02,
    irregular_irregularity_min=0.06,
    bias_radius_cap_px=10.0,
)


def _covariances(**overrides: Any) -> np.ndarray:
    """One blob's covariance under ``_MODEL``, with keyword overrides of the inputs."""
    inputs: dict[str, Any] = {
        'photon_vars_px2': np.array([0.01]),
        'diameters_px': np.array([8.0]),
        'phase_angles_deg': np.array([90.0]),
        'irregularity_factors': np.array([0.04]),
        'sun_dirs_vu': np.array([[0.6, 0.8]]),
        'model': _MODEL,
    }
    inputs.update(overrides)
    return _blob_covariances(**inputs)


# ---------------------------------------------------------------------------
# Lambertian photocenter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('phase_deg', 'expected'),
    [
        (0.0, 0.0),
        (30.0, 0.19861),
        (90.0, 3.0 * math.pi / 16.0),
        (180.0, 9.0 * math.pi / 32.0),
    ],
    ids=['full', 'gibbous', 'half', 'crescent-limit'],
)
def test_lambert_photocenter_fraction_matches_closed_form(
    phase_deg: float, expected: float
) -> None:
    """The Lambertian photocenter offset per unit radius matches its known values."""
    got = _lambert_photocenter_fraction(np.array([phase_deg]))
    assert float(got[0]) == pytest.approx(expected, abs=2e-5)


def test_lambert_photocenter_fraction_matches_numeric_integral() -> None:
    """The closed form equals a direct moment of a rendered Lambertian sphere at 135 deg."""
    alpha = math.radians(135.0)
    grid = np.linspace(-1.0, 1.0, 1601)
    xx, yy = np.meshgrid(grid, grid)
    on_disc = xx * xx + yy * yy < 1.0
    zz = np.sqrt(np.clip(1.0 - xx * xx - yy * yy, 0.0, None))
    mu0 = xx * math.sin(alpha) + zz * math.cos(alpha)
    brightness = np.where(on_disc & (mu0 > 0.0), mu0, 0.0)
    numeric = float((brightness * xx).sum() / brightness.sum())
    got = float(_lambert_photocenter_fraction(np.array([135.0]))[0])
    assert got == pytest.approx(numeric, abs=2e-3)


# ---------------------------------------------------------------------------
# Along-Sun fraction and per-blob covariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('irregularity', 'expected'),
    [(0.0, 0.9), (0.02, 0.9), (0.04, 0.65), (0.06, 0.4), (0.5, 0.4)],
    ids=['round', 'at-regular', 'midway', 'at-irregular', 'very-irregular'],
)
def test_along_sun_fraction_interpolates_in_irregularity(
    irregularity: float, expected: float
) -> None:
    """The along-Sun fraction holds its end values and is linear between the factors."""
    got = _along_sun_fractions(np.array([irregularity]), _MODEL)
    assert float(got[0]) == pytest.approx(expected, rel=1e-12)


_ISO = 0.01 + (0.05**2 + 0.04**2) * 16.0
"""Isotropic variance of the default blob: photon, size fraction, irregularity 0.04."""
_G_R = 3.0 * math.pi / 16.0 * 4.0
"""Photocenter offset of the default blob (radius 4 px, 90 deg phase)."""


def test_blob_covariance_along_the_sun_direction() -> None:
    """The variance along the direction toward the Sun carries the along-Sun phase term."""
    cov = _covariances()[0]
    sun = np.array([0.6, 0.8])
    assert float(sun @ cov @ sun) == pytest.approx(_ISO + (0.65 * _G_R) ** 2, rel=1e-12)


def test_blob_covariance_across_the_sun_direction() -> None:
    """The variance across the direction toward the Sun carries the across-Sun phase term."""
    cov = _covariances()[0]
    across = np.array([-0.8, 0.6])
    assert float(across @ cov @ across) == pytest.approx(_ISO + (0.2 * _G_R) ** 2, rel=1e-12)


def test_blob_covariance_has_no_correlation_between_sun_axes() -> None:
    """The along- and across-Sun directions are the covariance's principal axes."""
    cov = _covariances()[0]
    assert float(np.array([0.6, 0.8]) @ cov @ np.array([-0.8, 0.6])) == pytest.approx(
        0.0, abs=1e-12
    )


def test_blob_covariance_without_a_sun_direction_is_isotropic() -> None:
    """With no direction toward the Sun the two phase terms are split evenly."""
    cov = _covariances(sun_dirs_vu=np.array([[0.0, 0.0]]))[0]
    expected = 0.5 * np.trace(_covariances()[0])
    np.testing.assert_allclose(cov, expected * np.eye(2), rtol=1e-12)


def test_blob_covariance_takes_a_negative_irregularity_as_zero() -> None:
    """A negative irregularity factor contributes nothing."""
    negative = _covariances(irregularity_factors=np.array([-0.5]))
    zero = _covariances(irregularity_factors=np.array([0.0]))
    np.testing.assert_allclose(negative, zero, rtol=1e-12)


def test_blob_covariance_caps_the_phase_and_irregularity_radius() -> None:
    """Past the radius cap the phase and irregularity terms stop growing; the size term does not.

    A regular body 100 px across (radius 50) at 90 deg phase with irregularity 0.01:
    along the Sun ``0.01 + 0.05**2 * 50**2 + (0.01 * 10)**2 + (0.9 * g * 10)**2``.
    """
    cov = _covariances(diameters_px=np.array([100.0]), irregularity_factors=np.array([0.01]))[0]
    sun = np.array([0.6, 0.8])
    g_rc = 3.0 * math.pi / 16.0 * 10.0
    expected = 0.01 + 0.05**2 * 2500.0 + 0.1**2 + (0.9 * g_rc) ** 2
    assert float(sun @ cov @ sun) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Generalized least squares
# ---------------------------------------------------------------------------


def test_gls_single_blob_returns_its_offset() -> None:
    """One blob's offset is the joint offset."""
    cov = np.array([[[4.0, 1.0], [1.0, 2.0]]])
    fit = _generalized_least_squares(np.array([[3.0, -2.0]]), cov)
    assert fit.offset_vu == pytest.approx((3.0, -2.0), abs=1e-12)


def test_gls_single_blob_returns_its_covariance() -> None:
    """One blob's covariance is the joint covariance, unscaled."""
    cov = np.array([[[4.0, 1.0], [1.0, 2.0]]])
    fit = _generalized_least_squares(np.array([[3.0, -2.0]]), cov)
    np.testing.assert_allclose(fit.covariance, cov[0], rtol=1e-12)


def test_gls_weights_each_axis_by_its_own_precision() -> None:
    """Two blobs precise on different axes each set the joint offset on their own axis."""
    covs = np.array([np.diag([0.01, 100.0]), np.diag([100.0, 0.01])])
    fit = _generalized_least_squares(np.array([[1.0, 5.0], [4.0, 2.0]]), covs)
    assert fit.offset_vu == pytest.approx((1.0003, 2.0003), abs=1e-4)


def test_gls_inflates_the_covariance_when_blobs_disagree() -> None:
    """Blobs that disagree beyond their covariances scale the joint covariance by chi2/dof.

    Two unit-variance blobs 4 px apart in v: chi2 = 2 * 2**2 = 8 over 2 degrees of
    freedom, so the inverse-information covariance 0.5 grows to 2.0.
    """
    covs = np.array([np.eye(2), np.eye(2)])
    fit = _generalized_least_squares(np.array([[0.0, 0.0], [4.0, 0.0]]), covs)
    np.testing.assert_allclose(fit.covariance, 2.0 * np.eye(2), rtol=1e-12)


def test_gls_does_not_shrink_the_covariance_when_blobs_agree() -> None:
    """Blobs that agree better than their covariances keep the inverse information."""
    covs = np.array([np.eye(2), np.eye(2)])
    fit = _generalized_least_squares(np.array([[0.0, 0.0], [0.1, 0.0]]), covs)
    np.testing.assert_allclose(fit.covariance, 0.5 * np.eye(2), rtol=1e-12)


# ---------------------------------------------------------------------------
# Model loading and validation
# ---------------------------------------------------------------------------


def test_load_blob_bias_model_reads_every_term() -> None:
    """The tuning keys map onto the model's terms."""
    tuning = {
        'model_error_size_frac': 0.05,
        'along_sun_frac_regular': 0.9,
        'along_sun_frac_irregular': 0.4,
        'across_sun_frac': 0.2,
        'regular_irregularity_max': 0.02,
        'irregular_irregularity_min': 0.06,
        'bias_radius_cap_px': 10.0,
    }
    assert load_blob_bias_model(tuning) == _MODEL


def test_bias_model_rejects_a_negative_term() -> None:
    """A negative term fails at construction, naming the term."""
    with pytest.raises(ValueError, match=r'BodyBlobNav\.across_sun_frac must be finite and >= 0'):
        replace(_MODEL, across_sun_frac=-0.1)


def test_bias_model_rejects_irregularity_bounds_out_of_order() -> None:
    """The irregular bound must exceed the regular one."""
    with pytest.raises(ValueError, match='irregular_irregularity_min must exceed'):
        replace(_MODEL, irregular_irregularity_min=0.02)


def test_bias_model_rejects_a_zero_radius_cap() -> None:
    """The radius cap must be positive."""
    with pytest.raises(ValueError, match='bias_radius_cap_px must be > 0'):
        replace(_MODEL, bias_radius_cap_px=0.0)
