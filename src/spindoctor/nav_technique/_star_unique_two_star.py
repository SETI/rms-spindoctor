"""Pair geometry and covariances for ``StarUniqueMatchNav``'s two-star path.

The two-star path of :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.\
StarUniqueMatchNav` detects two catalog stars and then solves for the offset that
carries the predictions onto the detections.  The pure arithmetic of that solve lives
here: choosing the detection-to-prediction assignment with the smaller residual (with
or without a rotation), and building the translation covariance a match reports and
the three-parameter covariance of a pair fitted with a rotation.  The detection, the
gates, the confidence, and the logging stay with the technique.

The submodule is private (leading underscore on the file name); its functions carry no
underscore prefix because the technique module imports them, but they are not part of
the package's public API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique._star_helpers import SimilarityFit, similarity_transform_fit
from spindoctor.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'TwoStarAssignment',
    'build_star_covariance',
    'similarity_fit_assignment',
    'solve_two_star_assignment',
    'two_star_covariance',
    'two_star_covariance_3dof',
]


@dataclass(frozen=True)
class TwoStarAssignment:
    """The better of a pair's two detection-to-prediction assignments.

    Parameters:
        offset_vu: ``(dv, du)`` offset carrying the predictions onto the detections.
        residual_px: Residual of the assignment (the centroid-relative misfit of the
            first detection), the quantity the pair's cross-check bounds.
        assignment: ``'a->1,b->2'`` when the first detection matches the first star,
            ``'a->2,b->1'`` when the detections are swapped.
        rotation_rad: Fitted rotation in radians when a rotation was fitted, else
            ``None``.
    """

    offset_vu: tuple[float, float]
    residual_px: float
    assignment: str
    rotation_rad: float | None


def build_star_covariance(feature: NavFeature, *, residual_px: float) -> NDArrayFloatType:
    """Return a 2x2 covariance for a star match.

    Uses the per-feature anisotropic CRLB covariance carried on the
    ``NavFeature.position_cov_px`` (computed from predicted SNR + smear
    by ``NavModelStars``) as the floor and inflates it by
    ``residual_px**2`` so a noisy match honestly reports its scatter
    rather than the noise-free CRLB lower bound.

    The same inflation applies in both 1-star and 2-star modes: the
    1-star path's ``residual_px`` is the prediction-to-detection
    distance (no second observation to subtract), and the 2-star path's
    ``residual_px`` is already a per-axis RMS, so squaring either
    yields the right per-axis variance addend.

    Parameters:
        feature: The star feature whose CRLB cov is the floor.
        residual_px: Final per-axis residual magnitude.

    Returns:
        2x2 float64 covariance matrix; the identity stands in for a missing floor.
    """
    floor = feature.position_cov_px
    base = (
        np.eye(2, dtype=np.float64) if floor is None else np.asarray(floor, dtype=np.float64).copy()
    )
    inflation = residual_px * residual_px
    return base + inflation * np.eye(2, dtype=np.float64)


def solve_two_star_assignment(
    det_a: tuple[float, float],
    det_b: tuple[float, float],
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
    *,
    fit_rotation: bool,
) -> TwoStarAssignment:
    """Solve both assignments of two detections to two predictions and keep the better.

    Without a rotation the offset of an assignment is the mean of its two
    detection-minus-prediction differences.  With a rotation each assignment takes a
    Procrustes (rotation plus translation) fit; with two points the rigid fit is exact,
    so the residual compared is the same centroid-relative quantity.  Ties go to the
    direct assignment.

    Parameters:
        det_a: First detection ``(v, u)``.
        det_b: Second detection ``(v, u)``.
        pred_a: Prediction of the first star ``(v, u)``.
        pred_b: Prediction of the second star ``(v, u)``.
        fit_rotation: Whether to fit a rotation as well as a translation.

    Returns:
        The assignment with the smaller residual.
    """
    if fit_rotation:
        det_arr = np.asarray([det_a, det_b], dtype=np.float64)
        sim_1, residual_1 = similarity_fit_assignment(det_arr, pred_a, pred_b, swap=False)
        sim_2, residual_2 = similarity_fit_assignment(det_arr, pred_a, pred_b, swap=True)
        if residual_1 <= residual_2:
            sim_chosen, residual_px, assignment = sim_1, residual_1, 'a->1,b->2'
        else:
            sim_chosen, residual_px, assignment = sim_2, residual_2, 'a->2,b->1'
        offset_v, offset_u = sim_chosen.translation_vu
        return TwoStarAssignment(
            offset_vu=(offset_v, offset_u),
            residual_px=residual_px,
            assignment=assignment,
            rotation_rad=float(sim_chosen.rotation_rad),
        )
    # Assignment 1: det_a -> pred_a, det_b -> pred_b.
    offset_1_v = 0.5 * ((det_a[0] - pred_a[0]) + (det_b[0] - pred_b[0]))
    offset_1_u = 0.5 * ((det_a[1] - pred_a[1]) + (det_b[1] - pred_b[1]))
    residual_1 = math.hypot(
        (det_a[0] - pred_a[0]) - offset_1_v,
        (det_a[1] - pred_a[1]) - offset_1_u,
    )
    # Assignment 2: det_a -> pred_b, det_b -> pred_a.
    offset_2_v = 0.5 * ((det_a[0] - pred_b[0]) + (det_b[0] - pred_a[0]))
    offset_2_u = 0.5 * ((det_a[1] - pred_b[1]) + (det_b[1] - pred_a[1]))
    residual_2 = math.hypot(
        (det_a[0] - pred_b[0]) - offset_2_v,
        (det_a[1] - pred_b[1]) - offset_2_u,
    )
    if residual_1 <= residual_2:
        return TwoStarAssignment(
            offset_vu=(offset_1_v, offset_1_u),
            residual_px=residual_1,
            assignment='a->1,b->2',
            rotation_rad=None,
        )
    return TwoStarAssignment(
        offset_vu=(offset_2_v, offset_2_u),
        residual_px=residual_2,
        assignment='a->2,b->1',
        rotation_rad=None,
    )


def similarity_fit_assignment(
    det_arr: NDArrayFloatType,
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
    *,
    swap: bool,
) -> tuple[SimilarityFit, float]:
    """Run a single-assignment Procrustes fit and report its residual.

    Parameters:
        det_arr: Ordered detection pair ``[det_a, det_b]`` shaped
            ``(2, 2)``.
        pred_a: Catalog prediction for the first chosen star.
        pred_b: Catalog prediction for the second chosen star.
        swap: When False the catalog cohort is ``[pred_a, pred_b]``;
            when True the cohort is ``[pred_b, pred_a]`` (the
            swapped detection-to-prediction assignment).

    Returns:
        ``(SimilarityFit, residual_px)`` where the residual is the
        centroid-relative scalar used to compare assignments.
        For a two-point rigid fit the Procrustes residuals are zero
        by construction; the comparable scalar -- the same one the
        translation-only path uses -- is computed directly
        from the centroid-relative offset so the assignment-
        selection criterion is the same with and without a rotation.
    """
    if swap:
        cat_arr = np.asarray([pred_b, pred_a], dtype=np.float64)
    else:
        cat_arr = np.asarray([pred_a, pred_b], dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    fit = similarity_transform_fit(det_arr, cat_arr, weights)
    # Residuals from a 2-point rigid fit are zero by construction; the
    # comparable scalar for the assignment selection is the
    # centroid-relative residual prior to rotation, computed directly from
    # the assignment so the criterion is the same with and without a
    # rotation.
    det_a = det_arr[0]
    cat_first = cat_arr[0]
    offset = np.mean(det_arr - cat_arr, axis=0)
    residual = math.hypot(
        (det_a[0] - cat_first[0]) - float(offset[0]),
        (det_a[1] - cat_first[1]) - float(offset[1]),
    )
    return fit, residual


def two_star_covariance(
    star_a: NavFeature,
    star_b: NavFeature,
    *,
    residual_px: float,
    fit_rotation: bool,
) -> NDArrayFloatType:
    """Return a two-star match's covariance.

    The translation block averages the two stars' covariances from
    :func:`build_star_covariance` as a coarse two-star floor (symmetrized).  When a
    rotation was fitted the result is the 3x3 of :func:`two_star_covariance_3dof`.

    Parameters:
        star_a: The first matched star.
        star_b: The second matched star.
        residual_px: The pair's residual, inflating each star's floor.
        fit_rotation: Whether a rotation was fitted.

    Returns:
        A 2x2 covariance, or a 3x3 one when ``fit_rotation`` is True.
    """
    cov_a = build_star_covariance(star_a, residual_px=residual_px)
    cov_b = build_star_covariance(star_b, residual_px=residual_px)
    cov_2x2 = 0.5 * (cov_a + cov_b)
    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
    if not fit_rotation:
        return cov_2x2
    return two_star_covariance_3dof(
        cov_2x2=cov_2x2,
        pred_a=star_a.geometry.predicted_vu,  # type: ignore[union-attr]
        pred_b=star_b.geometry.predicted_vu,  # type: ignore[union-attr]
    )


def two_star_covariance_3dof(
    *,
    cov_2x2: NDArrayFloatType,
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
) -> NDArrayFloatType:
    """Build the 3x3 covariance for a 2-star Procrustes fit.

    Translation block uses the per-feature CRLB-derived 2x2; the
    rotation diagonal scales the sum of per-axis position variances
    by ``1 / L**2`` where ``L`` is the catalog separation between
    the two stars.  This is the analytic small-angle uncertainty:
    a per-axis position error ``sigma_pos`` perturbs the rotation
    by approximately ``sigma_pos / L`` per star, and the two stars
    contribute independently.

    Parameters:
        cov_2x2: Per-feature CRLB-derived 2x2 translation covariance
            (already symmetrized).
        pred_a: Catalog prediction for the first chosen star.
        pred_b: Catalog prediction for the second chosen star.

    Returns:
        3x3 covariance with the input ``cov_2x2`` in the
        translation block and either the analytic
        ``2 * (sigma_v**2 + sigma_u**2) / L**2`` rotation variance
        on the diagonal, or :data:`ROTATION_UNOBSERVABLE_VARIANCE`
        when the two predictions coincide (a degenerate input the
        upstream guard already rejects).
    """
    separation_sq = (pred_a[0] - pred_b[0]) ** 2 + (pred_a[1] - pred_b[1]) ** 2
    if separation_sq <= 0.0:
        sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
    else:
        # Sum of per-axis position variances aggregates the (v, u)
        # pixel uncertainty assumed isotropic by the centroid CRLB;
        # 2 / L^2 reflects the two-point lever-arm.
        sigma_theta_sq = 2.0 * float(cov_2x2[0, 0] + cov_2x2[1, 1]) / separation_sq
    cov = np.zeros((3, 3), dtype=np.float64)
    cov[:2, :2] = cov_2x2
    cov[2, 2] = sigma_theta_sq
    return cov
