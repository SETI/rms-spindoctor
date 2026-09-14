"""Aggregate per-frame twists into a per-instrument verdict and recommendation.

A single frame's twist is one noisy measurement.  The question the analysis
exists to answer is a property of the *instrument*: across many frames, is the
twist a single common value, or does it scatter?

- A common value is a static camera-frame alignment error.  It can be baked
  into a corrected pointing kernel once and removed for everyone, and per-frame
  rotation fitting during navigation buys little.
- A scatter that is large in its own right is genuine per-frame attitude error.
  No static kernel can remove it, so navigation must fit the rotation per frame
  where accuracy at the field edge matters.

The discriminator is the frame-to-frame twist scatter expressed as its
displacement at the field corner in pixels -- the quantity navigation actually
cares about -- not the formal chi-square.  A very precise per-frame twist fit
makes the reduced chi-square explode on a scatter that is operationally
negligible, so the chi-square is reported as a diagnostic but the verdict keys
on the corner-pixel scatter.  This module is pure numpy with no navigation
dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'RotationRecommendation',
    'TwistConsistency',
    'recommend_rotation_fitting',
    'twist_consistency',
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TwistConsistency:
    """Consistency statistics for one instrument's per-frame twists.

    Parameters:
        n_frames: Number of frames contributing a twist.
        weighted_mean_deg: Inverse-variance-weighted mean twist in degrees.
        sigma_mean_deg: One-sigma uncertainty on the weighted mean.
        scatter_deg: Sample standard deviation of the per-frame twists (the raw
            frame-to-frame spread, ignoring per-frame uncertainties).
        median_sigma_deg: Median per-frame twist uncertainty, i.e. the typical
            statistical precision of one frame's twist fit.
        reduced_chi_square: Chi-square of the twists about the weighted mean per
            degree of freedom.  A diagnostic only: with a very precise per-frame
            fit it grows large on a scatter that is operationally negligible, so
            it does not drive the verdict.
        rho_ref_px: Field-corner radius (half the image diagonal) used to
            convert angles to corner displacements.
        mean_corner_px: Signed displacement the mean twist produces at the field
            corner, ``sin(mean) * rho_ref_px``.
        scatter_corner_px: Displacement the frame-to-frame twist scatter
            produces at the field corner, ``sin(scatter) * rho_ref_px`` -- the
            operational size of the variation navigation would see.
        consistent: True when ``scatter_corner_px`` is at or below
            ``scatter_corner_threshold_px``.
        scatter_corner_threshold_px: The corner-displacement threshold below
            which the twist counts as one common value.
    """

    n_frames: int
    weighted_mean_deg: float
    sigma_mean_deg: float
    scatter_deg: float
    median_sigma_deg: float
    reduced_chi_square: float
    rho_ref_px: float
    mean_corner_px: float
    scatter_corner_px: float
    consistent: bool
    scatter_corner_threshold_px: float


@dataclass(frozen=True)
class RotationRecommendation:
    """Per-instrument recommendation derived from the twist consistency.

    Parameters:
        fit_camera_rotation: Recommended value of the navigation
            ``fit_camera_rotation`` flag.  True when the twist scatters frame to
            frame (navigation must fit it) or when a consistent twist is large
            enough to matter but is not yet removed by a kernel.
        kernel_twist_correction_deg: A consistent, non-zero twist that should be
            folded into a corrected pointing kernel, or ``None`` when the twist
            is not consistent or is negligible.
        rationale: One-line human-readable explanation.
    """

    fit_camera_rotation: bool
    kernel_twist_correction_deg: float | None
    rationale: str


def twist_consistency(
    twists_deg: FloatArray,
    sigmas_deg: FloatArray,
    rho_ref_px: float,
    *,
    scatter_corner_threshold_px: float = 0.15,
) -> TwistConsistency:
    """Summarize the frame-to-frame consistency of an instrument's twists.

    Parameters:
        twists_deg: Per-frame twist angles in degrees.
        sigmas_deg: Per-frame one-sigma uncertainties in degrees, same order.
        rho_ref_px: Field-corner radius (half the image diagonal) in pixels,
            used to express twist angles as corner displacements.
        scatter_corner_threshold_px: Corner displacement at or below which the
            frame-to-frame scatter counts as a single common twist.  The
            default of 0.15 px is a small fraction of a pixel at the field
            corner -- below it, per-frame rotation fitting cannot buy
            navigation meaningful accuracy.

    Returns:
        A :class:`TwistConsistency`.

    Raises:
        ValueError: if the inputs are empty, disagree in length, carry a
            non-positive sigma, or ``rho_ref_px`` is not positive.
    """
    twists = np.asarray(twists_deg, dtype=np.float64)
    sigmas = np.asarray(sigmas_deg, dtype=np.float64)
    if twists.ndim != 1 or sigmas.shape != twists.shape:
        raise ValueError('twists_deg and sigmas_deg must be 1-D arrays of equal length')
    n = twists.shape[0]
    if n == 0:
        raise ValueError('at least one frame is required')
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError('sigmas_deg must be finite and strictly positive')
    if not rho_ref_px > 0.0:
        raise ValueError(f'rho_ref_px must be positive; got {rho_ref_px}')

    inv_var = 1.0 / sigmas**2
    weighted_mean = float((inv_var * twists).sum() / inv_var.sum())
    sigma_mean = float(math.sqrt(1.0 / inv_var.sum()))
    scatter = float(np.std(twists, ddof=1)) if n > 1 else 0.0
    median_sigma = float(np.median(sigmas))

    if n > 1:
        chi_square = float((inv_var * (twists - weighted_mean) ** 2).sum())
        reduced_chi_square = chi_square / (n - 1)
    else:
        reduced_chi_square = 0.0

    mean_corner = math.sin(math.radians(weighted_mean)) * rho_ref_px
    scatter_corner = math.sin(math.radians(scatter)) * rho_ref_px
    consistent = scatter_corner <= scatter_corner_threshold_px

    return TwistConsistency(
        n_frames=n,
        weighted_mean_deg=weighted_mean,
        sigma_mean_deg=sigma_mean,
        scatter_deg=scatter,
        median_sigma_deg=median_sigma,
        reduced_chi_square=reduced_chi_square,
        rho_ref_px=float(rho_ref_px),
        mean_corner_px=mean_corner,
        scatter_corner_px=scatter_corner,
        consistent=consistent,
        scatter_corner_threshold_px=scatter_corner_threshold_px,
    )


def recommend_rotation_fitting(
    consistency: TwistConsistency,
    *,
    significance_corner_px: float = 0.15,
) -> RotationRecommendation:
    """Turn a twist-consistency summary into a rotation-fitting recommendation.

    Parameters:
        consistency: The per-instrument :class:`TwistConsistency`.
        significance_corner_px: Corner displacement below which a consistent
            mean twist is treated as negligible -- no kernel correction is worth
            making and per-frame rotation fitting is not needed.

    Returns:
        A :class:`RotationRecommendation`.
    """
    mean = consistency.weighted_mean_deg
    mean_corner = consistency.mean_corner_px
    significant = (
        abs(mean_corner) >= significance_corner_px and abs(mean) >= 3.0 * consistency.sigma_mean_deg
    )

    if not consistency.consistent:
        return RotationRecommendation(
            fit_camera_rotation=True,
            kernel_twist_correction_deg=None,
            rationale=(
                f'Twist scatters frame to frame by {consistency.scatter_deg:.3f} deg '
                f'({consistency.scatter_corner_px:.2f} px at the field corner) across '
                f'{consistency.n_frames} frames, above the '
                f'{consistency.scatter_corner_threshold_px:.2f} px threshold; no static '
                f'kernel can remove it, so fit rotation per frame.'
            ),
        )
    if significant:
        return RotationRecommendation(
            fit_camera_rotation=False,
            kernel_twist_correction_deg=mean,
            rationale=(
                f'Twist is consistent at {mean:+.3f} +/- {consistency.sigma_mean_deg:.3f} deg '
                f'({mean_corner:+.2f} px at the field corner) across {consistency.n_frames} '
                f'frames; fold this static offset into a corrected pointing kernel and leave '
                f'per-frame rotation fitting off.'
            ),
        )
    return RotationRecommendation(
        fit_camera_rotation=False,
        kernel_twist_correction_deg=None,
        rationale=(
            f'Twist is consistent and negligible ({mean:+.3f} deg, {mean_corner:+.2f} px at '
            f'the field corner, over {consistency.n_frames} frames); no kernel correction '
            f'needed and per-frame rotation fitting is unnecessary.'
        ),
    )
