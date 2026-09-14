"""Unit tests for the pure-numpy residual-field decomposition."""

from __future__ import annotations

import math

import numpy as np
import pytest
from util.fov_distortion.decompose import (
    decompose_frame,
    fit_radial_distortion,
    weighted_rigid_fit,
)


def _star_grid(half: float = 100.0, step: float = 25.0) -> np.ndarray:
    """A square grid of star positions centered on the origin."""
    coords = np.arange(-half, half + step, step)
    vv, uu = np.meshgrid(coords, coords)
    return np.column_stack([vv.ravel(), uu.ravel()]).astype(np.float64)


def _rotate(points: np.ndarray, center: np.ndarray, theta: float) -> np.ndarray:
    """Rotate points about a center by theta radians in the (v, u) frame."""
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    rotated: np.ndarray = (points - center) @ rot.T + center
    return rotated


def test_rigid_fit_recovers_pure_rotation() -> None:
    center = (0.0, 0.0)
    pred = _star_grid()
    theta = math.radians(0.3)
    det = _rotate(pred, np.asarray(center), theta)
    fit = weighted_rigid_fit(pred, det, center)
    assert fit.rotation_deg == pytest.approx(0.3, abs=1e-6)


def test_rigid_fit_recovers_rotation_plus_translation() -> None:
    center = (500.0, 500.0)
    pred = _star_grid() + np.asarray(center)
    theta = math.radians(-0.15)
    det = _rotate(pred, np.asarray(center), theta) + np.asarray([3.2, -1.7])
    fit = weighted_rigid_fit(pred, det, center)
    assert fit.rotation_deg == pytest.approx(-0.15, abs=1e-6)


def test_rigid_fit_translation_is_recovered() -> None:
    center = (0.0, 0.0)
    pred = _star_grid()
    det = pred + np.asarray([4.0, -2.0])
    fit = weighted_rigid_fit(pred, det, center)
    assert fit.translation_vu[0] == pytest.approx(4.0, abs=1e-9)
    assert fit.translation_vu[1] == pytest.approx(-2.0, abs=1e-9)


def test_rigid_fit_residual_zero_for_exact_transform() -> None:
    center = (0.0, 0.0)
    pred = _star_grid()
    det = _rotate(pred, np.asarray(center), math.radians(0.2))
    fit = weighted_rigid_fit(pred, det, center)
    assert fit.rms_px == pytest.approx(0.0, abs=1e-9)


def test_rigid_fit_sigma_shrinks_with_more_stars() -> None:
    center = (0.0, 0.0)
    rng = np.random.default_rng(1)
    small = _star_grid(half=100.0, step=50.0)
    large = _star_grid(half=100.0, step=10.0)
    theta = math.radians(0.1)
    det_small = _rotate(small, np.asarray(center), theta) + rng.normal(0, 0.1, small.shape)
    det_large = _rotate(large, np.asarray(center), theta) + rng.normal(0, 0.1, large.shape)
    fit_small = weighted_rigid_fit(small, det_small, center)
    fit_large = weighted_rigid_fit(large, det_large, center)
    assert fit_large.sigma_rotation_deg < fit_small.sigma_rotation_deg


def test_rigid_fit_requires_two_stars() -> None:
    with pytest.raises(ValueError, match='at least two stars'):
        weighted_rigid_fit(np.zeros((1, 2)), np.zeros((1, 2)), (0.0, 0.0))


def test_radial_fit_recovers_planted_distortion() -> None:
    center = (0.0, 0.0)
    rho_ref = 200.0
    pred = _star_grid(half=180.0, step=20.0)
    # Remove the exact-center star so every star has a defined radial direction.
    rho = np.hypot(pred[:, 0], pred[:, 1])
    pred = pred[rho > 0.0]
    rho = rho[rho > 0.0]
    rhat = pred / rho[:, None]
    k1, k2 = 0.03, -0.01
    rho_n = rho / rho_ref
    radial_disp = rho_ref * (k1 * rho_n**3 + k2 * rho_n**5)
    residuals = radial_disp[:, None] * rhat
    model = fit_radial_distortion(pred, residuals, center, rho_ref, powers=(3, 5))
    assert model.k_sim[0] == pytest.approx(k1, abs=1e-6)
    assert model.k_sim[1] == pytest.approx(k2, abs=1e-6)


def test_radial_fit_separates_nonradial_component() -> None:
    center = (0.0, 0.0)
    rho_ref = 200.0
    pred = _star_grid(half=180.0, step=20.0)
    rho = np.hypot(pred[:, 0], pred[:, 1])
    pred = pred[rho > 0.0]
    rho = rho[rho > 0.0]
    rhat = pred / rho[:, None]
    that = np.column_stack([-rhat[:, 1], rhat[:, 0]])
    # Pure tangential displacement: no radial signal at all.
    residuals = 0.5 * that
    model = fit_radial_distortion(pred, residuals, center, rho_ref, powers=(3, 5))
    assert model.rms_radial_px == pytest.approx(0.0, abs=1e-9)
    assert model.rms_nonradial_px == pytest.approx(0.5, abs=1e-9)


def test_decompose_frame_separates_twist_and_distortion() -> None:
    center = (512.0, 512.0)
    rho_ref = 723.0
    pred = _star_grid(half=400.0, step=40.0) + np.asarray(center)
    rho = np.hypot(pred[:, 0] - center[0], pred[:, 1] - center[1])
    keep = rho > 0.0
    pred = pred[keep]
    rho = rho[keep]
    theta = math.radians(0.25)
    rotated = _rotate(pred, np.asarray(center), theta) + np.asarray([6.0, -3.0])
    rhat = (pred - np.asarray(center)) / rho[:, None]
    k1 = 0.02
    rho_n = rho / rho_ref
    radial_disp = rho_ref * (k1 * rho_n**3)
    det = rotated + radial_disp[:, None] * rhat
    decomp = decompose_frame(pred, det, center, rho_ref, powers=(3, 5))
    assert decomp.twist.rotation_deg == pytest.approx(0.25, abs=1e-3)
    assert decomp.radial.k_sim[0] == pytest.approx(k1, abs=1e-3)
    assert decomp.rms_after_radial_px < 1e-2


def test_decompose_frame_rms_ordering() -> None:
    center = (256.0, 256.0)
    rho_ref = 362.0
    rng = np.random.default_rng(7)
    pred = _star_grid(half=200.0, step=25.0) + np.asarray(center)
    rho = np.hypot(pred[:, 0] - center[0], pred[:, 1] - center[1])
    keep = rho > 0.0
    pred = pred[keep]
    rho = rho[keep]
    theta = math.radians(0.4)
    rotated = _rotate(pred, np.asarray(center), theta) + np.asarray([10.0, 5.0])
    rhat = (pred - np.asarray(center)) / rho[:, None]
    rho_n = rho / rho_ref
    det = rotated + (rho_ref * 0.05 * rho_n**3)[:, None] * rhat + rng.normal(0, 0.05, pred.shape)
    decomp = decompose_frame(pred, det, center, rho_ref, powers=(3, 5))
    assert decomp.rms_after_twist_px < decomp.rms_raw_px
    assert decomp.rms_after_radial_px < decomp.rms_after_twist_px
