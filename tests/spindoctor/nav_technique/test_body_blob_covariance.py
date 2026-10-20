"""Tests for ``BodyBlobNav``'s joint-fit covariance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spindoctor.nav_technique.body_blob_covariance import (
    _centroid_bias_fractions,
    _centroid_model_error_var,
    _joint_covariance,
    _lambert_photocenter_fraction,
)


def test_joint_covariance_two_point_reduced_chi_square() -> None:
    """Two-blob covariance matches the analytic reduced-chi-square weighted mean.

    Derivation (per-axis: var = chi2_nu / sum(w), with
    chi2_nu = sum(w * r^2) / max(N - p, 1), p = 2):

    w = [1, 3], sum(w) = 4, N = 2, dof = max(2 - 2, 1) = 1.

    V axis: offsets = [2, 4], mean = (1*2 + 3*4)/4 = 3.5,
            r = [-1.5, 0.5], sum(w r^2) = 1*2.25 + 3*0.25 = 3.0,
            chi2_nu = 3.0/1 = 3.0, var = 3.0/4 = 0.75 (> floor 1/4 = 0.25).
    U axis: offsets = [-1, 1], mean = (1*-1 + 3*1)/4 = 0.5,
            r = [-1.5, 0.5], sum(w r^2) = 3.0, var = 0.75 (identical).
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.5
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 0.5
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    assert cov[0, 0] == pytest.approx(0.75, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.75, abs=1e-9)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert cov[1, 0] == pytest.approx(0.0, abs=1e-9)


def test_joint_covariance_n_point_reduced_chi_square() -> None:
    """Four-blob covariance matches the analytic reduced-chi-square weighted mean.

    w = [1, 1, 1, 1], sum(w) = 4, N = 4, dof = max(4 - 2, 1) = 2.

    V axis: offsets = [0, 2, 4, 6], mean = 12/4 = 3.0,
            r = [-3, -1, 1, 3], sum(w r^2) = 9 + 1 + 1 + 9 = 20,
            chi2_nu = 20/2 = 10, var = 10/4 = 2.5 (> floor 0.25).
    U axis: offsets = [1, 1, 1, 1], mean = 1.0, r = 0 everywhere,
            sum(w r^2) = 0, chi2_nu = 0, candidate var = 0 -> floored at
            1/sum(w) = 1/4 = 0.25.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    offsets_v = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    offsets_u = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.0
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 1.0
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    assert cov[0, 0] == pytest.approx(2.5, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)


def test_joint_covariance_single_blob_is_large_not_overconfident() -> None:
    """NAV-005: a single blob yields the (large) inverse-precision floor.

    One point cannot constrain two translation parameters, so the
    covariance must be large rather than a tiny over-confident value.
    With a single small-weight blob (w = 0.01) the per-axis variance is
    the pure inverse precision 1/sum(w) = 1/0.01 = 100.0 -- there is no
    residual scatter (the lone residual is zero by construction).  The
    old ``sum(w r^2)/(sum w)^2`` form would have reported 0 here (no
    floor at the correct power), which the over-confidence fix corrects.
    """
    weights = np.array([0.01], dtype=np.float64)
    offsets_v = np.array([3.0], dtype=np.float64)
    offsets_u = np.array([-4.0], dtype=np.float64)
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=3.0,
        du=-4.0,
        extents_px=np.zeros_like(offsets_v),
    )
    # 1 / sum(w) = 1 / 0.01 = 100.0 on each axis.
    assert cov[0, 0] == pytest.approx(100.0, abs=1e-9)
    assert cov[1, 1] == pytest.approx(100.0, abs=1e-9)
    # Far larger than a well-determined multi-blob fit (~O(1)).
    assert cov[0, 0] > 1.0


def test_joint_covariance_model_error_floor_inflates_diagonal_by_square() -> None:
    """ORCH-001: model_error_floor_px>0 adds exactly its square to the diagonal.

    Reuses the two-point fixture (var = 0.75 per axis with no floor).
    With model_error_floor_px = 2.0 each diagonal grows by exactly
    2.0**2 = 4.0 -> 0.75 + 4.0 = 4.75; the off-diagonal stays zero.
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = 3.5
    du = 0.5
    base = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    floored = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
        model_error_floor_px=2.0,
    )
    assert floored[0, 0] == pytest.approx(4.75, abs=1e-9)
    assert floored[1, 1] == pytest.approx(4.75, abs=1e-9)
    assert floored[0, 0] - base[0, 0] == pytest.approx(4.0, abs=1e-9)
    assert floored[1, 1] - base[1, 1] == pytest.approx(4.0, abs=1e-9)
    assert floored[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_centroid_model_error_var_single_blob_is_radius_fraction_squared() -> None:
    """One blob's size-error variance is exactly (frac * radius)**2."""
    extents = np.array([100.0], dtype=np.float64)  # diameter -> radius 50
    var = _centroid_model_error_var(extents, 0.05)
    assert var == pytest.approx((0.05 * 50.0) ** 2, rel=1e-9)


def test_centroid_model_error_var_combines_by_inverse_variance() -> None:
    """Two equal blobs halve the single-blob size-error variance."""
    extents = np.array([100.0, 100.0], dtype=np.float64)
    single = _centroid_model_error_var(extents[:1], 0.05)
    both = _centroid_model_error_var(extents, 0.05)
    assert both == pytest.approx(single / 2.0, rel=1e-9)


def test_centroid_model_error_var_disabled_returns_zero() -> None:
    """A zero fraction disables the size term."""
    extents = np.array([100.0, 40.0], dtype=np.float64)
    assert _centroid_model_error_var(extents, 0.0) == 0.0


def test_joint_covariance_size_term_tracks_body_radius() -> None:
    """A single-blob covariance scales with body size through the size term.

    The reported per-axis sigma of a lone blob should land within a factor
    of 1.5 of the size-scaled centroid-model error (frac * radius) plus the
    absolute floor -- the calibration property the photon-only weight lacked.
    """
    weights = np.array([100.0], dtype=np.float64)  # bright: photon floor ~0.01 px
    offsets_v = np.array([0.0], dtype=np.float64)
    offsets_u = np.array([0.0], dtype=np.float64)
    extents = np.array([200.0], dtype=np.float64)  # radius 100 px
    frac = 0.05
    floor = 0.1
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=0.0,
        du=0.0,
        extents_px=extents,
        model_error_floor_px=floor,
        centroid_model_error_frac=frac,
    )
    reported_sigma = float(np.sqrt(cov[0, 0]))
    expected_sigma = float(np.sqrt((frac * 100.0) ** 2 + floor**2 + 1.0 / 100.0))
    assert 1.0 / 1.5 <= reported_sigma / expected_sigma <= 1.5


# ---------------------------------------------------------------------------
# Coarse blob-shaped-disc acquisition
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


def test_centroid_bias_fractions_add_phase_and_irregularity_in_quadrature() -> None:
    """A blob's bias fraction is the phase term and its irregularity factor in quadrature."""
    phase = np.array([90.0])
    irregular = np.array([0.4])
    got = _centroid_bias_fractions(phase, irregular, phase_frac=0.3)
    expected = math.hypot(0.3 * 3.0 * math.pi / 16.0, 0.4)
    assert float(got[0]) == pytest.approx(expected, rel=1e-9)


def test_centroid_model_error_var_adds_bias_fraction_to_size_fraction() -> None:
    """A per-blob bias fraction adds to the shared size fraction in quadrature."""
    extents = np.array([100.0], dtype=np.float64)  # radius 50
    var = _centroid_model_error_var(extents, 0.05, bias_fracs=np.array([0.2]))
    assert var == pytest.approx((0.05**2 + 0.2**2) * 50.0**2, rel=1e-9)


def test_joint_covariance_crescent_blob_reports_its_centroid_bias() -> None:
    """A small irregular body seen as a crescent carries a bias sigma near 2 px.

    Atlas (diameter 7.5 px, phase-irregularity factor 0.447) at 134.8 deg phase,
    the frame on which a 0.21 px sigma let the blob pull the answer 2 px.
    """
    extents = np.array([7.477])
    bias = _centroid_bias_fractions(np.array([134.83]), np.array([0.447]), phase_frac=0.3)
    cov = _joint_covariance(
        offsets_v=np.array([0.0]),
        offsets_u=np.array([0.0]),
        weights=np.array([1.0e4]),
        dv=0.0,
        du=0.0,
        extents_px=extents,
        model_error_floor_px=0.1,
        centroid_model_error_frac=0.05,
        bias_fracs=bias,
    )
    assert math.sqrt(cov[0, 0]) == pytest.approx(1.91, abs=0.02)
