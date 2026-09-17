"""Continuous gradient-ridge sub-pixel refinement.

The final stage after the DT Levenberg-Marquardt: fits directly to the
un-quantized gradient magnitude, removing the sub-pixel-phase bias floor the
integer-quantized DT zero-set leaves.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_LM_STEP_TOLERANCE,
    DEFAULT_PINVH_RCOND,
    DEFAULT_RIDGE_HALF_WIDTH_PX,
    DEFAULT_RIDGE_MAX_ITERATIONS,
    DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX,
    DEFAULT_RIDGE_SAMPLE_STEP_PX,
    DEFAULT_TUKEY_C,
)
from spindoctor.nav_technique.dt_fitting.transforms import (
    _rotate_directions,
    _rotate_vertices,
    _shift_vertices,
    _step_norm_px,
    _weighted_normal_equations,
)
from spindoctor.nav_technique.dt_fitting.weights import tukey_biweight_weights
from spindoctor.support.distance_transform import sample_dt_bilinear
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'RidgeRefineResult',
    'gradient_ridge_refine',
]


def _ridge_normal_distances(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    gradient_magnitude: NDArrayFloatType,
    dv: float,
    du: float,
    dtheta: float,
    t_samples: NDArrayFloatType,
    sample_step_px: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Locate the sub-pixel gradient-magnitude ridge along each vertex normal.

    For each vertex at its current (rotated + shifted) position the
    continuous gradient magnitude is bilinearly sampled at
    ``position + t * normal`` for every ``t`` in ``t_samples``; the signed
    normal distance ``t*`` to the ridge is the parabola-interpolated
    location of the sampled maximum.  ``t*`` is the residual the
    gradient-ridge Gauss-Newton stage drives to zero (the vertex then sits
    on the continuous edge ridge).

    Parameters:
        vertices_vu: ``(N, 2)`` base (unshifted) model vertices.
        normals_vu: ``(N, 2)`` base model normals (unit length).
        pivot_vu: rotation pivot.
        gradient_magnitude: ``(H, W)`` continuous gradient magnitude image.
        dv, du: current translation.
        dtheta: current rotation (radians).
        t_samples: ``(K,)`` monotone normal offsets to sample.
        sample_step_px: spacing of ``t_samples`` (pixels).

    Returns:
        ``(t_star, normals_current, valid)``:

        - ``t_star`` ``(N,)`` signed normal distance to the ridge peak.
        - ``normals_current`` ``(N, 2)`` normals rotated by ``dtheta`` (the
          directions the translation Jacobian rows ``-n`` use).
        - ``valid`` ``(N,)`` boolean: True where the sampled maximum is a
          strict interior peak (a usable sub-pixel residual).  A maximum at
          either sampling boundary means the true ridge lies outside the
          search window, so that vertex carries no usable ridge residual.
    """
    norms_current = _rotate_directions(normals_vu, dtheta)
    pos = _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta), dv, du)
    n = pos.shape[0]
    k = t_samples.shape[0]
    # Sample points along each normal: (N, K, 2).
    sample_pts = pos[:, None, :] + t_samples[None, :, None] * norms_current[:, None, :]
    # ``sample_dt_bilinear`` is a clamped bilinear interpolator over any 2-D
    # field, so it samples the continuous gradient magnitude here just as it
    # samples the DT elsewhere.
    mag = sample_dt_bilinear(gradient_magnitude, sample_pts.reshape(-1, 2)).reshape(n, k)
    kmax = np.argmax(mag, axis=1)
    rows = np.arange(n)
    interior = (kmax > 0) & (kmax < k - 1)
    # Clamp the peak index so the three-point gather stays in bounds; the
    # boundary peaks are masked out via ``valid`` regardless.
    kc = np.clip(kmax, 1, k - 2)
    y_minus = mag[rows, kc - 1]
    y_zero = mag[rows, kc]
    y_plus = mag[rows, kc + 1]
    denom = y_minus - 2.0 * y_zero + y_plus
    # A genuine peak is concave-down (denom < 0); a non-concave triple has no
    # interior vertex, so fall back to the sampled location (delta = 0).
    concave = denom < -1.0e-12
    delta = np.where(concave, 0.5 * (y_minus - y_plus) / np.where(concave, denom, -1.0), 0.0)
    # Keep the parabola vertex inside the central cell so a noisy triple cannot
    # throw ``t*`` past the neighboring samples.
    delta = np.clip(delta, -0.5, 0.5)
    t_star = t_samples[kc] + delta * sample_step_px
    return cast(NDArrayFloatType, t_star), norms_current, cast(NDArrayFloatType, interior)


@dataclass(frozen=True)
class RidgeRefineResult:
    """Structured output of :func:`gradient_ridge_refine`.

    Parameters:
        offset_vu: ``(dv, du)`` refined translation in pixels.
        rotation_rad: refined rotation in radians (unchanged from the input
            when ``fit_rotation`` was False).
        iterations: Gauss-Newton iterations performed.
        converged: True if the step-norm tolerance was met before the cap.
        applied: True if the refined pose was accepted.  False when the
            stage found too few valid ridge residuals to fit, or the
            cumulative displacement from the input offset exceeded the
            displacement cap -- in both cases ``offset_vu`` / ``rotation_rad``
            equal the inputs unchanged and the caller keeps the DT-LM pose.
    """

    offset_vu: tuple[float, float]
    rotation_rad: float
    iterations: int
    converged: bool
    applied: bool


def gradient_ridge_refine(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    sigma_normal_per_vertex_px: NDArrayFloatType,
    gradient_magnitude: NDArrayFloatType,
    initial_offset_vu: tuple[float, float],
    weight_mask: NDArrayFloatType | None = None,
    initial_rotation_rad: float = 0.0,
    fit_rotation: bool = False,
    pivot_vu: tuple[float, float] | None = None,
    pivot_distance_px: float = 0.0,
    max_iterations: int = DEFAULT_RIDGE_MAX_ITERATIONS,
    step_tolerance_px: float = DEFAULT_LM_STEP_TOLERANCE,
    tukey_c: float = DEFAULT_TUKEY_C,
    half_width_px: float = DEFAULT_RIDGE_HALF_WIDTH_PX,
    sample_step_px: float = DEFAULT_RIDGE_SAMPLE_STEP_PX,
    max_total_displacement_px: float = DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX,
) -> RidgeRefineResult:
    """Polish a polyline alignment against the continuous gradient-ridge field.

    This is the final, sub-pixel stage after the coarse-NCC + DT
    Levenberg-Marquardt acquisition.  The DT-LM minimizes distance to an
    integer-quantized edge mask, whose zero-set snaps the recovered edge to
    integer pixels and leaves an SNR-independent sub-pixel-phase bias floor.
    This stage removes that floor by fitting directly to the *continuous*
    (un-quantized) gradient magnitude: for each vertex it finds the
    sub-pixel signed normal distance ``t*`` to the gradient ridge and runs
    Gauss-Newton with Tukey-biweight reweighting to drive every ``t*`` to
    zero.

    The residual for vertex ``i`` is ``r_i = t*_i`` (signed distance from
    the vertex to the ridge along its outward normal ``n_i``).  Moving the
    vertex by a translation ``delta`` changes the residual by ``-(n_i .
    delta)``, so the translation Jacobian rows are ``[-n_v, -n_u]``; the
    rotation column (when fitted) is central-differenced.  The normal
    equations and the Tukey reweighting reuse the same machinery as the
    DT-LM stage.

    Parameters:
        vertices_vu: ``(N, 2)`` base (unshifted) model vertices.
        normals_vu: ``(N, 2)`` model normals; need not be unit length but
            should be (the DT techniques pass unit normals).
        sigma_normal_per_vertex_px: ``(N,)`` strictly-positive prior sigma;
            ``1 / sigma**2`` is the prior precision weight, identical to the
            DT-LM stage.
        gradient_magnitude: ``(H, W)`` continuous gradient magnitude image
            (``hypot`` of the gradient-vector components).
        initial_offset_vu: ``(dv, du)`` starting translation (the DT-LM
            optimum).
        weight_mask: optional ``(N,)`` multiplicative mask (e.g. the DT-LM
            polarity acceptance) applied to every vertex weight.  ``None``
            keeps all vertices.
        initial_rotation_rad: starting rotation (the DT-LM optimum).
        fit_rotation: when True the parameter vector is ``(dv, du, dtheta)``.
        pivot_vu: rotation pivot; defaults to the centroid of
            ``vertices_vu``.
        pivot_distance_px: pivot-to-image-center distance for the rotation
            step-norm conversion.  Required when ``fit_rotation`` is True.
        max_iterations: Gauss-Newton iteration cap.
        step_tolerance_px: step-norm threshold for convergence.
        tukey_c: Holland-Welsch Tukey constant.
        half_width_px: half-width of the normal search window.
        sample_step_px: spacing of the normal sample points.
        max_total_displacement_px: cap on cumulative displacement from
            ``initial_offset_vu``; exceeding it discards the refinement.

    Returns:
        :class:`RidgeRefineResult`.

    Raises:
        ValueError: if shape requirements are violated, the prior sigmas
            are not strictly positive, or ``fit_rotation`` is True without a
            positive ``pivot_distance_px``.
    """
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    sigmas = np.asarray(sigma_normal_per_vertex_px, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] == 0:
        raise ValueError(f'vertices_vu must have shape (N, 2) with N > 0; got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    if sigmas.ndim != 1 or sigmas.shape[0] != verts.shape[0]:
        raise ValueError(
            'sigma_normal_per_vertex_px must be a 1-D vector matching '
            f'vertices_vu; got {sigmas.shape}'
        )
    if (sigmas <= 0.0).any() or not np.isfinite(sigmas).all():
        raise ValueError('sigma_normal_per_vertex_px entries must be finite and > 0')
    if gradient_magnitude.ndim != 2:
        raise ValueError(f'gradient_magnitude must be 2-D; got ndim={gradient_magnitude.ndim}')
    if fit_rotation and not (pivot_distance_px > 0.0):
        raise ValueError(
            'fit_rotation=True requires pivot_distance_px > 0 for the convergence test'
        )
    base_mask = (
        np.ones(verts.shape[0], np.float64)
        if weight_mask is None
        else np.asarray(weight_mask, np.float64)
    )
    pivot = (
        (float(verts[:, 0].mean()), float(verts[:, 1].mean()))
        if pivot_vu is None
        else (float(pivot_vu[0]), float(pivot_vu[1]))
    )
    inv_sigma_sq = 1.0 / (sigmas * sigmas)
    half = float(half_width_px)
    step = float(sample_step_px)
    # Symmetric, monotone sample offsets including 0; the central sample lets
    # a converged (t* -> 0) vertex report a zero residual exactly.
    n_side = round(half / step)
    t_samples = np.arange(-n_side, n_side + 1, dtype=np.float64) * step
    dv0, du0 = float(initial_offset_vu[0]), float(initial_offset_vu[1])
    dv, du, dtheta = dv0, du0, float(initial_rotation_rad)
    eps_t = 1.0e-3
    iterations = 0
    converged = False
    for _ in range(max_iterations):
        t_star, norms_current, valid = _ridge_normal_distances(
            vertices_vu=verts,
            normals_vu=norms,
            pivot_vu=pivot,
            gradient_magnitude=gradient_magnitude,
            dv=dv,
            du=du,
            dtheta=dtheta,
            t_samples=t_samples,
            sample_step_px=step,
        )
        scaled = t_star / sigmas
        tukey_w = tukey_biweight_weights(scaled, c=tukey_c)
        weights = inv_sigma_sq * tukey_w * valid * base_mask
        if not np.any(weights > 0):
            # No usable ridge evidence; keep whatever pose we have.
            break
        if fit_rotation:
            t_plus, _, _ = _ridge_normal_distances(
                vertices_vu=verts,
                normals_vu=norms,
                pivot_vu=pivot,
                gradient_magnitude=gradient_magnitude,
                dv=dv,
                du=du,
                dtheta=dtheta + eps_t,
                t_samples=t_samples,
                sample_step_px=step,
            )
            t_minus, _, _ = _ridge_normal_distances(
                vertices_vu=verts,
                normals_vu=norms,
                pivot_vu=pivot,
                gradient_magnitude=gradient_magnitude,
                dv=dv,
                du=du,
                dtheta=dtheta - eps_t,
                t_samples=t_samples,
                sample_step_px=step,
            )
            drdth = (t_plus - t_minus) / (2.0 * eps_t)
            jacobian = np.stack([-norms_current[:, 0], -norms_current[:, 1], drdth], axis=-1)
        else:
            jacobian = np.stack([-norms_current[:, 0], -norms_current[:, 1]], axis=-1)
        hessian, rhs = _weighted_normal_equations(jacobian, t_star, weights)
        try:
            gn_step = -np.linalg.solve(hessian, rhs)
        except np.linalg.LinAlgError:
            gn_step = -pinvh(hessian, rtol=DEFAULT_PINVH_RCOND) @ rhs
        dv += float(gn_step[0])
        du += float(gn_step[1])
        if fit_rotation:
            dtheta += float(gn_step[2])
        iterations += 1
        step_norm = _step_norm_px(
            gn_step,
            fit_rotation=fit_rotation,
            pivot_distance_px=pivot_distance_px,
        )
        if step_norm < step_tolerance_px:
            converged = True
            break
    displacement = math.hypot(dv - dv0, du - du0)
    if iterations == 0 or displacement > max_total_displacement_px:
        # Either no Gauss-Newton step ran (no valid ridge evidence) or the
        # stage walked too far to trust; keep the DT-LM pose.
        return RidgeRefineResult(
            offset_vu=(dv0, du0),
            rotation_rad=float(initial_rotation_rad),
            iterations=iterations,
            converged=converged,
            applied=False,
        )
    return RidgeRefineResult(
        offset_vu=(dv, du),
        rotation_rad=dtheta,
        iterations=iterations,
        converged=converged,
        applied=True,
    )
