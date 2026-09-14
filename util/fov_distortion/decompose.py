"""Decompose a per-star residual field into twist plus radial distortion.

Given, for one image, the predicted catalog positions of a set of stars and
their measured (centroided) positions, this module separates the total
predicted-to-measured displacement into three physically distinct parts:

1. A global translation -- the residual spacecraft pointing offset that plain
   navigation already removes.  It is not a camera defect and is discarded.
2. A rigid rotation about the optical center -- the FOV twist.  A twist that is
   the same on every frame of an instrument is a static camera-frame alignment
   error correctable in a pointing kernel; a twist that scatters frame to frame
   is genuine per-frame attitude error.  Which case holds is decided by
   :mod:`aggregate`, not here; this module only measures one frame's twist and
   its uncertainty.
3. The displacement remaining after translation and rotation are removed -- the
   lateral residual distortion.  Its radial part is fitted with the same
   low-order polynomial the simulator plants, so the measured coefficients feed
   the simulator distortion stage directly.  Its non-radial part is reported as
   a scalar RMS.

Coordinates are image ``(v, u)`` = ``(row, column)`` throughout.  The axis order
is all this module fixes: the star positions, ``center_vu`` and ``pivot_vu`` need
only be in one continuous pixel coordinate system, whichever the caller works in.
:mod:`measure` passes pixel corner positions, because that is what the star
catalog records and what ``psf.find_position`` returns.  Everything here is pure
numpy: the module carries no navigation dependency so it can be exercised on
synthetic point clouds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'FrameDecomposition',
    'RadialModel',
    'RigidFit',
    'decompose_frame',
    'fit_radial_distortion',
    'weighted_rigid_fit',
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RigidFit:
    """Rigid similarity (rotation + translation, no scale) mapping predicted to detected.

    The transform maps a predicted point ``p`` to ``R(theta) @ p +
    translation``; ``rotation_rad`` is the twist and ``translation_vu`` the
    global pointing offset that plain navigation removes.  ``pivot_vu`` is the
    optical center used only as the lever-arm origin for the twist uncertainty,
    not as a rotation pivot in the returned transform.

    Parameters:
        rotation_rad: Fitted twist angle in radians (positive sense follows the
            image ``(v, u)`` frame; the same convention is used for every
            instrument so per-instrument consistency is meaningful).
        sigma_rotation_rad: One-sigma uncertainty on the twist from the inlier
            residual scatter and the star lever arm.
        translation_vu: Global ``(dv, du)`` translation.
        pivot_vu: Optical center used as the lever-arm origin for the twist
            uncertainty.
        residuals_vu: ``(N, 2)`` residual ``detected - transform(predicted)``
            for every star, after both rotation and translation are removed.
        rms_px: Root-mean-square magnitude of ``residuals_vu``.
        n_stars: Number of stars in the fit.
    """

    rotation_rad: float
    sigma_rotation_rad: float
    translation_vu: tuple[float, float]
    pivot_vu: tuple[float, float]
    residuals_vu: FloatArray
    rms_px: float
    n_stars: int

    @property
    def rotation_deg(self) -> float:
        """Twist angle in degrees."""
        return math.degrees(self.rotation_rad)

    @property
    def sigma_rotation_deg(self) -> float:
        """Twist uncertainty in degrees."""
        return math.degrees(self.sigma_rotation_rad)


@dataclass(frozen=True)
class RadialModel:
    """Low-order radial distortion model fitted to the post-twist residuals.

    The radial residual (the component of each post-twist residual along the
    line from the optical center) is fitted as
    ``sum_k coeffs_px[k] * rho_n ** powers[k]`` where ``rho_n = |p - center| /
    rho_ref_px`` is the normalized field radius.  With the default powers
    ``(3, 5)`` this matches the simulator distortion warp
    ``source = center + (p - center) * (1 + k1 * rho_n**2 + k2 * rho_n**4)``,
    for which the radial displacement in pixels is
    ``rho_ref_px * (k1 * rho_n**3 + k2 * rho_n**5)``.  The simulator
    coefficients are therefore ``k_sim[k] = coeffs_px[k] / rho_ref_px``.

    Parameters:
        powers: Polynomial powers of the normalized radius used as the fit
            basis.
        coeffs_px: Fitted coefficients in pixels, one per power.
        rho_ref_px: Normalizing radius (half the image diagonal) in pixels.
        k_sim: ``coeffs_px / rho_ref_px`` -- the simulator distortion-stage
            coefficients, one per power.
        rms_radial_px: RMS of the radial residual component (the signal being
            modeled) before the fit is subtracted.
        rms_nonradial_px: RMS of the tangential (non-radial) residual component
            -- the seed for the simulator non-radial wander amplitude.
        rms_unmodeled_px: RMS of the radial component after the fitted model is
            subtracted -- the part the low-order model does not capture.
        n_stars: Number of stars in the fit.
    """

    powers: tuple[int, ...]
    coeffs_px: tuple[float, ...]
    rho_ref_px: float
    k_sim: tuple[float, ...]
    rms_radial_px: float
    rms_nonradial_px: float
    rms_unmodeled_px: float
    n_stars: int

    def radial_displacement_px(self, rho_px: FloatArray) -> FloatArray:
        """Evaluate the fitted radial displacement at the given field radii.

        Parameters:
            rho_px: Field radii in pixels.

        Returns:
            The modeled radial displacement in pixels at each radius.
        """
        rho_n = np.asarray(rho_px, dtype=np.float64) / self.rho_ref_px
        out = np.zeros_like(rho_n)
        for coeff, power in zip(self.coeffs_px, self.powers, strict=True):
            out = out + coeff * rho_n**power
        return out


@dataclass(frozen=True)
class FrameDecomposition:
    """Full decomposition of one frame's predicted-to-detected residual field.

    Parameters:
        n_stars: Number of stars used.
        twist: The rigid twist + translation fit.
        radial: The radial distortion model fitted to the post-twist residuals.
        rms_raw_px: RMS of the raw predicted-to-detected displacement (before
            any term is removed) with the median translation taken out, so a
            large uncorrected pointing offset does not dominate the number.
        rms_after_twist_px: RMS residual after translation and twist are
            removed.
        rms_after_radial_px: RMS residual after translation, twist, and the
            radial model are removed -- the irreducible floor set by centroid
            noise and star-catalog astrometry.
    """

    n_stars: int
    twist: RigidFit
    radial: RadialModel
    rms_raw_px: float
    rms_after_twist_px: float
    rms_after_radial_px: float


def _normalized_weights(weights: FloatArray | None, n: int) -> FloatArray:
    """Return non-negative weights of length ``n`` (uniform if none supplied)."""
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f'weights must be 1-D of length {n}; got shape {w.shape}')
    if (w < 0.0).any():
        raise ValueError('weights must be non-negative')
    return w


def weighted_rigid_fit(
    predicted_vu: FloatArray,
    detected_vu: FloatArray,
    pivot_vu: tuple[float, float],
    *,
    weights: FloatArray | None = None,
) -> RigidFit:
    """Fit the weighted rigid rotation + translation mapping predicted to detected.

    Solves the weighted orthogonal-Procrustes (Kabsch) problem for the proper
    rotation and translation that minimise the weighted squared residual
    ``sum_i w_i |detected_i - (R @ predicted_i + t)|**2``.  The determinant of
    the SVD reconstruction is forced positive so the result is a proper
    rotation rather than a reflection.  The rotation angle is invariant to the
    choice of ``pivot_vu``; the pivot only sets how the translation is
    expressed and the frame the residuals are reported in.

    Parameters:
        predicted_vu: ``(N, 2)`` predicted star positions in ``(v, u)``.
        detected_vu: ``(N, 2)`` measured positions, same order.
        pivot_vu: Optical center ``(v, u)`` to express the rotation about.
        weights: Optional ``(N,)`` non-negative weights; uniform if omitted.

    Returns:
        A :class:`RigidFit`.

    Raises:
        ValueError: if fewer than two stars are supplied or the shapes disagree.
    """
    pred = np.asarray(predicted_vu, dtype=np.float64)
    det = np.asarray(detected_vu, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f'predicted_vu must have shape (N, 2); got {pred.shape}')
    if det.shape != pred.shape:
        raise ValueError(f'detected_vu must match predicted_vu; got {det.shape} vs {pred.shape}')
    n = pred.shape[0]
    if n < 2:
        raise ValueError(f'a rigid fit needs at least two stars; got {n}')
    w = _normalized_weights(weights, n)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError('weights sum to zero')

    pred_bar = (w[:, None] * pred).sum(axis=0) / total
    det_bar = (w[:, None] * det).sum(axis=0) / total
    pred_c = pred - pred_bar
    det_c = det - det_bar
    cross = (w[:, None] * det_c).T @ pred_c
    u_mat, _s, vt = np.linalg.svd(cross)
    sign = np.eye(2, dtype=np.float64)
    sign[1, 1] = math.copysign(1.0, float(np.linalg.det(u_mat @ vt)))
    rotation = u_mat @ sign @ vt
    theta = float(math.atan2(rotation[1, 0], rotation[0, 0]))

    pivot = np.asarray(pivot_vu, dtype=np.float64)
    # Global translation that maps predicted -> detected under the fitted
    # rotation: modeled = rotation @ predicted + translation.
    translation = det_bar - rotation @ pred_bar
    modeled = pred @ rotation.T + translation
    residuals = det - modeled
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    # Twist uncertainty: an angle error is a position error divided by the
    # lever arm, reduced by sqrt(N).  Use the isotropic per-star residual
    # variance over the weighted mean-square lever arm about the pivot.
    lever = pred - pivot
    lever_sq = float((w * np.sum(lever**2, axis=1)).sum() / total)
    resid_var = float((w * np.sum(residuals**2, axis=1)).sum() / total) / 2.0
    if lever_sq > 0.0 and total > 0.0:
        sigma_theta = math.sqrt(resid_var / (total * lever_sq))
    else:
        sigma_theta = float('inf')

    return RigidFit(
        rotation_rad=theta,
        sigma_rotation_rad=sigma_theta,
        translation_vu=(float(translation[0]), float(translation[1])),
        pivot_vu=(float(pivot[0]), float(pivot[1])),
        residuals_vu=residuals,
        rms_px=rms,
        n_stars=n,
    )


def fit_radial_distortion(
    predicted_vu: FloatArray,
    residuals_vu: FloatArray,
    center_vu: tuple[float, float],
    rho_ref_px: float,
    *,
    powers: tuple[int, ...] = (3, 5),
    weights: FloatArray | None = None,
) -> RadialModel:
    """Fit a low-order radial distortion model to post-twist residuals.

    The radial component of each residual (its projection onto the outward
    line from ``center_vu``) is fitted against a polynomial in the normalized
    field radius; the tangential component's RMS is reported separately as the
    non-radial wander.

    Parameters:
        predicted_vu: ``(N, 2)`` predicted star positions in ``(v, u)``.
        residuals_vu: ``(N, 2)`` residuals after the rigid twist is removed.
        center_vu: Optical center ``(v, u)``.
        rho_ref_px: Normalizing radius in pixels (half the image diagonal).
        powers: Polynomial powers of the normalized radius for the fit basis.
        weights: Optional ``(N,)`` non-negative weights; uniform if omitted.

    Returns:
        A :class:`RadialModel`.  When there are fewer stars than free
        coefficients the model is returned with zero coefficients and the
        measured radial / non-radial RMS still populated.

    Raises:
        ValueError: if the shapes disagree or ``rho_ref_px`` is not positive.
    """
    pred = np.asarray(predicted_vu, dtype=np.float64)
    res = np.asarray(residuals_vu, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f'predicted_vu must have shape (N, 2); got {pred.shape}')
    if res.shape != pred.shape:
        raise ValueError(f'residuals_vu must match predicted_vu; got {res.shape} vs {pred.shape}')
    if not rho_ref_px > 0.0:
        raise ValueError(f'rho_ref_px must be positive; got {rho_ref_px}')
    n = pred.shape[0]
    w = _normalized_weights(weights, n)

    center = np.asarray(center_vu, dtype=np.float64)
    offset = pred - center
    rho = np.hypot(offset[:, 0], offset[:, 1])
    safe = rho > 0.0
    rhat = np.zeros_like(offset)
    rhat[safe] = offset[safe] / rho[safe, None]
    # Tangential unit vector (rotate radial by +90 deg in the (v, u) frame).
    that = np.zeros_like(offset)
    that[safe, 0] = -rhat[safe, 1]
    that[safe, 1] = rhat[safe, 0]

    radial_comp = np.sum(res * rhat, axis=1)
    tangential_comp = np.sum(res * that, axis=1)
    rms_radial = _weighted_rms(radial_comp, w)
    rms_nonradial = _weighted_rms(tangential_comp, w)

    rho_n = rho / rho_ref_px
    design = np.stack([rho_n**p for p in powers], axis=1)
    if n >= len(powers) and np.any(w > 0.0):
        sqrt_w = np.sqrt(w)
        coeffs, *_ = np.linalg.lstsq(sqrt_w[:, None] * design, sqrt_w * radial_comp, rcond=None)
        coeffs_px = tuple(float(c) for c in coeffs)
        modeled_radial = design @ coeffs
    else:
        coeffs_px = tuple(0.0 for _ in powers)
        modeled_radial = np.zeros_like(radial_comp)
    rms_unmodeled = _weighted_rms(radial_comp - modeled_radial, w)
    k_sim = tuple(c / rho_ref_px for c in coeffs_px)

    return RadialModel(
        powers=tuple(int(p) for p in powers),
        coeffs_px=coeffs_px,
        rho_ref_px=float(rho_ref_px),
        k_sim=k_sim,
        rms_radial_px=rms_radial,
        rms_nonradial_px=rms_nonradial,
        rms_unmodeled_px=rms_unmodeled,
        n_stars=n,
    )


def decompose_frame(
    predicted_vu: FloatArray,
    detected_vu: FloatArray,
    center_vu: tuple[float, float],
    rho_ref_px: float,
    *,
    powers: tuple[int, ...] = (3, 5),
    weights: FloatArray | None = None,
    n_iterations: int = 3,
) -> FrameDecomposition:
    """Decompose one frame's residual field into twist plus radial distortion.

    A radial distortion field and a rigid rotation are not quite orthogonal: a
    strong radial term biases a single-pass rotation fit by a small amount and
    vice versa.  The two are decoupled by alternating the fits -- fit the twist
    on the distortion-removed detections, fit the radial model on the
    twist-removed residuals, repeat -- which converges in a couple of
    iterations.  The reported twist rotation and its uncertainty come from the
    distortion-removed fit; the reported ``residuals_vu`` on the twist are the
    full post-twist residual (radial plus non-radial plus noise) so that
    :func:`fit_radial_distortion` and the plots see the distortion signal.

    Parameters:
        predicted_vu: ``(N, 2)`` predicted star positions in ``(v, u)``.
        detected_vu: ``(N, 2)`` measured positions, same order.
        center_vu: Optical center ``(v, u)`` used as the twist pivot and the
            radial-distortion origin.
        rho_ref_px: Normalizing radius in pixels (half the image diagonal).
        powers: Polynomial powers for the radial fit basis.
        weights: Optional ``(N,)`` non-negative weights; uniform if omitted.
        n_iterations: Number of twist / radial alternations.

    Returns:
        A :class:`FrameDecomposition`.

    Raises:
        ValueError: if fewer than two stars are supplied.
    """
    pred = np.asarray(predicted_vu, dtype=np.float64)
    det = np.asarray(detected_vu, dtype=np.float64)
    n = pred.shape[0]
    w = _normalized_weights(weights, n)

    center = np.asarray(center_vu, dtype=np.float64)
    offset = pred - center
    rho = np.hypot(offset[:, 0], offset[:, 1])
    safe = rho > 0.0
    rhat = np.zeros_like(offset)
    rhat[safe] = offset[safe] / rho[safe, None]

    # Alternate the twist and radial fits to decouple them.
    radial_vec = np.zeros_like(det)
    twist = weighted_rigid_fit(pred, det, center_vu, weights=w)
    radial = fit_radial_distortion(
        pred, twist.residuals_vu, center_vu, rho_ref_px, powers=powers, weights=w
    )
    for _ in range(max(0, n_iterations - 1)):
        radial_vec = radial.radial_displacement_px(rho)[:, None] * rhat
        twist = weighted_rigid_fit(pred, det - radial_vec, center_vu, weights=w)
        modeled_rigid = (det - radial_vec) - twist.residuals_vu
        post_twist_resid = det - modeled_rigid
        radial = fit_radial_distortion(
            pred, post_twist_resid, center_vu, rho_ref_px, powers=powers, weights=w
        )

    # Final consistent products from the converged twist + radial model.
    modeled_rigid = (det - radial_vec) - twist.residuals_vu
    post_twist_resid = det - modeled_rigid
    radial_vec = radial.radial_displacement_px(rho)[:, None] * rhat
    after_radial = post_twist_resid - radial_vec

    # Report the full post-twist residual on the twist (the distortion signal),
    # keeping the decoupled rotation and its uncertainty.
    twist = replace(
        twist,
        residuals_vu=post_twist_resid,
        rms_px=_weighted_rms_vec(post_twist_resid, w),
    )

    # Raw displacement with the median translation removed so an uncorrected
    # pointing offset does not swamp the pre-decomposition number.
    raw = det - pred
    raw_centered = raw - np.median(raw, axis=0)
    rms_raw = _weighted_rms_vec(raw_centered, w)
    rms_after_radial = _weighted_rms_vec(after_radial, w)

    return FrameDecomposition(
        n_stars=n,
        twist=twist,
        radial=radial,
        rms_raw_px=rms_raw,
        rms_after_twist_px=twist.rms_px,
        rms_after_radial_px=rms_after_radial,
    )


def _weighted_rms(values: FloatArray, weights: FloatArray) -> float:
    """Weighted RMS of a 1-D array."""
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sqrt(float((weights * values**2).sum()) / total))


def _weighted_rms_vec(vectors: FloatArray, weights: FloatArray) -> float:
    """Weighted RMS magnitude of an ``(N, 2)`` array of vectors."""
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sqrt(float((weights * np.sum(vectors**2, axis=1)).sum()) / total))
