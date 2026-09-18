"""Levenberg-Marquardt sub-pixel refinement against the image distance transform.

The core of the shared DT pipeline: IRLS with Tukey reweighting, an optional
trust region and Tikhonov anchor, and an optional final gradient-ridge polish.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.nav_technique.dt_fitting import ridge as _ridge
from spindoctor.nav_technique.dt_fitting.coarse import polarity_filter
from spindoctor.nav_technique.dt_fitting.constants import (
    _INFINITY_DT_PENALTY_PX,
    DEFAULT_LM_DAMPING,
    DEFAULT_LM_MAX_ITERATIONS,
    DEFAULT_LM_STEP_TOLERANCE,
    DEFAULT_PINVH_RCOND,
    DEFAULT_TUKEY_C,
)
from spindoctor.nav_technique.dt_fitting.transforms import (
    _compute_residuals_and_jacobian,
    _rotate_vertices,
    _shift_vertices,
    _step_norm_px,
    _weighted_cost,
    _weighted_normal_equations,
)
from spindoctor.nav_technique.dt_fitting.weights import (
    information_matrix_to_covariance,
    tukey_biweight_weights,
)
from spindoctor.support.distance_transform import sample_dt_bilinear
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'LMRefineResult',
    'lm_subpixel_refine',
]


@dataclass(frozen=True)
class LMRefineResult:
    """Structured output of :func:`lm_subpixel_refine`.

    Parameters:
        offset_vu: ``(dv, du)`` converged translation in pixels.
        rotation_rad: Converged rotation in radians; ``0.0`` when
            ``fit_rotation`` was False.
        covariance: ``(2, 2)`` or ``(3, 3)`` parameter covariance derived
            from the M-estimator information matrix ``J^T diag(w) J`` at
            the converged point.  This is the DATA-ONLY covariance: it
            deliberately EXCLUDES the Tikhonov anchor diagonal that
            ``tikhonov_alpha`` adds to the iteration Hessian, because that
            anchor is a fitting aid that biases the step rather than
            measured information and so must not shrink the reported
            uncertainty.
        residuals_px: ``(N,)`` per-vertex DT residuals at the final
            estimate (raw, unweighted).
        weights: ``(N,)`` per-vertex weights at the final estimate
            (prior precision times Tukey biweight).
        rms_px: Weighted root-mean-square of the residuals, computed as
            ``sqrt(sum(w * r**2) / sum(w))`` over surviving vertices;
            ``float('inf')`` when every vertex was rejected (the
            degenerate case), so the downstream spurious gates' ``rms_px
            > floor`` test fires instead of reading a zero RMS as a good
            fit.
        raw_rms_px: Unweighted root-mean-square of the residuals over the
            polarity-ACCEPTED vertices, computed as ``sqrt(mean(r**2))``
            with no Tukey weighting.  Because the Tukey reweighting can
            down-weight a wholly mis-aligned but polarity-accepted arc to
            ~0, the weighted ``rms_px`` collapses to near zero on exactly
            such a mis-convergence and slips past a ``rms_px > floor``
            gate; the raw RMS retains those Tukey outliers and surfaces the
            bad fit.  Polarity-rejected vertices are excluded (they carry
            the ``_INFINITY_DT_PENALTY_PX`` sentinel and would otherwise
            dominate the mean); ``float('inf')`` when every vertex is
            polarity-rejected, so the degenerate case still trips the gate.
        iterations: Number of LM iterations actually performed.
        converged: True if the reported pose is LOCALLY converged: either
            the DT-LM met its step-norm tolerance before the iteration cap,
            or the final gradient-ridge stage applied and met its own step
            tolerance (a quantized-DT stall that burns the LM iteration
            budget at the integer seed is routine on dense edge scenes --
            the exact condition the ridge stage exists to polish).  This
            certifies local optimality only: the ridge is seeded from the
            DT-LM pose and bounded by its displacement cap, so it can only
            polish the SAME lock, and a wrong acquisition whose ridge
            converges is reported as converged.  ``False`` means neither
            stage reached a local optimum, which the shared DT fit-quality
            gates treat as an unverified fit.
        inlier_count: Number of vertices that retained a strictly
            positive Tukey weight at the final estimate.
        degenerate: True when no vertex survived reweighting
            (``inlier_count == 0`` or the surviving weights sum to zero).
            In this case ``rms_px`` is ``+inf`` and ``covariance`` is
            all-``inf``; consumers treat it as a spurious fit.
        polarity_rejected_count: Number of vertices the polarity filter
            rejected at the seed offset (``0`` when ``use_polarity`` was
            False).  ``polarity_rejected_count / N`` is the
            polarity-rejection fraction the fit-quality gates consume.
    """

    offset_vu: tuple[float, float]
    rotation_rad: float
    covariance: NDArrayFloatType
    residuals_px: NDArrayFloatType
    weights: NDArrayFloatType
    rms_px: float
    raw_rms_px: float
    iterations: int
    converged: bool
    inlier_count: int
    degenerate: bool
    polarity_rejected_count: int = 0

    def __post_init__(self) -> None:
        """Freeze the numpy arrays and validate shapes."""
        for name in ('covariance', 'residuals_px', 'weights'):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f'LMRefineResult.{name} must be an ndarray')
            arr.setflags(write=False)


@dataclass
class _LMState:
    """Mutable LM iteration state.  Internal to :func:`lm_subpixel_refine`."""

    dv: float
    du: float
    dtheta: float
    polarity_mask: NDArrayBoolType
    iteration: int = 0
    converged: bool = False
    raw_residuals: NDArrayFloatType = field(default_factory=lambda: np.zeros(0, np.float64))
    weights: NDArrayFloatType = field(default_factory=lambda: np.zeros(0, np.float64))
    jacobian: NDArrayFloatType = field(default_factory=lambda: np.zeros((0, 2), np.float64))


def lm_subpixel_refine(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    sigma_normal_per_vertex_px: NDArrayFloatType,
    image_edge_dt: NDArrayFloatType,
    image_gradient_vu: NDArrayFloatType | None = None,
    initial_offset_vu: tuple[float, float] = (0.0, 0.0),
    initial_rotation_rad: float = 0.0,
    fit_rotation: bool = False,
    pivot_vu: tuple[float, float] | None = None,
    pivot_distance_px: float = 0.0,
    use_polarity: bool = True,
    max_iterations: int = DEFAULT_LM_MAX_ITERATIONS,
    damping: float = DEFAULT_LM_DAMPING,
    step_tolerance_px: float = DEFAULT_LM_STEP_TOLERANCE,
    tukey_c: float = DEFAULT_TUKEY_C,
    pinvh_rcond: float = DEFAULT_PINVH_RCOND,
    trust_region_px: float | None = None,
    tikhonov_alpha: float = 0.0,
    final_gradient_ridge: bool = False,
) -> LMRefineResult:
    """Refine a polyline-vs-image alignment by Levenberg-Marquardt.

    The cost function is

    .. math::
        C(p) = \\sum_i w_i \\, \\bigl[\\mathrm{DT}\\bigl(R(\\theta)\\,(x_i - x_p)
                                              + x_p + (\\Delta v, \\Delta u)\\bigr)\\bigr]^2

    where :math:`x_i` are the input vertices, :math:`x_p` is the rotation
    pivot, :math:`R(\\theta)` is the in-plane rotation, ``DT`` is the
    bilinearly-sampled image distance transform, and the per-vertex
    weight :math:`w_i` is the product of the prior precision
    :math:`1 / \\sigma_i^2` and the Tukey biweight evaluated at the
    scaled residual :math:`r_i / \\sigma_i`.  The Tukey weights are
    recomputed after each accepted LM step (iteratively-reweighted
    least squares).

    Parameters:
        vertices_vu: ``(N, 2)`` model vertex positions.
        normals_vu: ``(N, 2)`` model outward normals (only used if
            ``use_polarity`` is True).
        sigma_normal_per_vertex_px: ``(N,)`` strictly-positive prior
            sigma in pixels.  ``1 / sigma**2`` is the prior precision
            weight.
        image_edge_dt: ``(H, W)`` precomputed image distance transform.
        image_gradient_vu: ``(H, W, 2)`` gradient vector image; required
            when ``use_polarity`` is True, ignored otherwise.
        initial_offset_vu: ``(dv0, du0)`` starting translation.
        initial_rotation_rad: Starting rotation in radians.
        fit_rotation: When True the parameter vector is
            ``(dv, du, dtheta)``; otherwise ``(dv, du)``.
        pivot_vu: ``(v_p, u_p)`` rotation pivot.  Defaults to the
            centroid of ``vertices_vu``.
        pivot_distance_px: Approximate pivot-to-image-center distance in
            pixels; used to convert rotation steps into pixel-equivalent
            increments for the convergence test.  Required when
            ``fit_rotation`` is True; ignored otherwise.
        use_polarity: When True, polarity-rejected vertices are assigned
            an effectively-infinite residual so the Tukey biweight zeroes
            their contribution.
        max_iterations: Iteration cap.
        damping: Initial Levenberg-Marquardt damping ``lambda``.
        step_tolerance_px: Step-norm threshold below which the iteration
            terminates with ``converged=True``.
        tukey_c: Holland-Welsch Tukey biweight constant.
        pinvh_rcond: Pseudoinverse cutoff for the final covariance.
        trust_region_px: Optional radius (pixels) around
            ``initial_offset_vu`` outside which a trial step is rejected
            without committing.  ``None`` (default) leaves the LM
            unconstrained, which is the default.  When set, every
            trial offset is checked against
            ``hypot(trial_dv - dv0, trial_du - du0) <= trust_region_px``;
            a violation marks the step as rejected (lambda doubled,
            iteration counter advanced) without updating ``dv`` / ``du``.
            This contains the joint LM + IRLS instability whereby
            Tukey reweighting can drag the polyline off the integer
            coarse-NCC seed onto an unrelated DT minimum (a crater
            rim, terminator edge, or surface boundary).
        tikhonov_alpha: Strength of a soft Tikhonov anchor pulling the
            translation back toward ``initial_offset_vu``.  ``0`` (default)
            disables the term, which is the default.  When positive,
            the cost adds a per-iteration penalty
            ``alpha * sum(weights) * ||(dv, du) - (dv0, du0)||^2`` which
            scales with the data so the LM trades off raw DT
            improvement against displacement.  The penalty is applied
            only to the translation degrees of freedom; rotation is
            never penalized.  The trust region is the hard outer
            bound; Tikhonov pulls the LM toward the seed *inside*
            that bound when the DT cost surface has a deeper but
            wrong minimum on the way (crater rims, terminator edges).
            The anchor biases the step only; it is excluded from the
            reported data-only covariance (see ``LMRefineResult``).
        final_gradient_ridge: When True, after the DT Levenberg-Marquardt
            converges, a final :func:`gradient_ridge_refine` stage polishes
            the offset against the *continuous* gradient-magnitude ridge,
            removing the sub-pixel-phase bias floor the integer-quantized DT
            zero-set leaves.  ``image_gradient_vu`` must be supplied (it is
            already required when ``use_polarity`` is True).  The reported
            ``residuals_px`` / ``weights`` / ``rms_px`` / ``covariance`` are
            recomputed against the DT at the ridge-refined pose, so the
            spurious gates and the reported uncertainty stay on the same DT
            footing as without the stage.  ``False`` (default) leaves the
            DT-LM optimum unchanged.
    Returns:
        :class:`LMRefineResult`.

    Raises:
        ValueError: if any shape requirement is violated, the prior
            sigmas are not strictly positive, or ``fit_rotation`` is
            True without a positive ``pivot_distance_px``.
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
    if image_edge_dt.ndim != 2:
        raise ValueError(f'image_edge_dt must be 2-D; got ndim={image_edge_dt.ndim}')
    if fit_rotation and not (pivot_distance_px > 0.0):
        raise ValueError(
            'fit_rotation=True requires pivot_distance_px > 0 for the convergence test'
        )
    if final_gradient_ridge and image_gradient_vu is None:
        raise ValueError('final_gradient_ridge=True requires image_gradient_vu')
    if use_polarity:
        if image_gradient_vu is None:
            raise ValueError('use_polarity=True requires image_gradient_vu')
        polarity_mask = polarity_filter(
            verts,
            norms,
            image_gradient_vu,
            offset_vu=initial_offset_vu,
        )
    else:
        polarity_mask = np.ones(verts.shape[0], dtype=bool)
    pivot = (
        (float(verts[:, 0].mean()), float(verts[:, 1].mean()))
        if pivot_vu is None
        else (float(pivot_vu[0]), float(pivot_vu[1]))
    )
    inv_sigma_sq = 1.0 / (sigmas * sigmas)
    state = _LMState(
        dv=float(initial_offset_vu[0]),
        du=float(initial_offset_vu[1]),
        dtheta=float(initial_rotation_rad),
        polarity_mask=polarity_mask,
    )
    lambda_ = float(damping)
    best_cost = math.inf
    n_params = 3 if fit_rotation else 2
    while state.iteration < max_iterations:
        residuals, jacobian = _compute_residuals_and_jacobian(
            vertices_vu=verts,
            pivot_vu=pivot,
            image_dt=image_edge_dt,
            dv=state.dv,
            du=state.du,
            dtheta=state.dtheta,
            fit_rotation=fit_rotation,
        )
        # Polarity-reject vertices contribute the constant penalty that
        # the Tukey biweight will zero on the next reweighting step.
        residuals = np.where(state.polarity_mask, residuals, _INFINITY_DT_PENALTY_PX)
        # Compute Tukey biweight weights from the (sigma-scaled) residuals.
        scaled = residuals / sigmas
        tukey_w = tukey_biweight_weights(scaled, c=tukey_c)
        # Zero polarity-rejected vertices directly via the mask rather than
        # relying on the large-penalty residual being driven past the Tukey
        # cutoff: that chain breaks for an enormous per-vertex sigma
        # (scaled = penalty / sigma <= c), and ``navigate`` may not raise.
        # The explicit mask makes rejection independent of sigma.
        weights = inv_sigma_sq * tukey_w * state.polarity_mask
        cost_before = _weighted_cost(weights, residuals)
        state.raw_residuals = residuals
        state.weights = weights
        state.jacobian = jacobian
        if not np.any(weights > 0):
            break
        hessian, rhs = _weighted_normal_equations(jacobian, residuals, weights)
        # Tikhonov anchor toward the initial seed on translation only.
        # Scaled by ``sum(weights)`` so the penalty tracks the data
        # size: ``alpha`` is a dimensionless ratio (penalty per
        # weighted-residual-equivalent at displacement = 1 px).
        # ``rotation`` is never penalized — only the translation
        # block of the (2 or 3)-DoF Hessian / RHS receives the term.
        if tikhonov_alpha > 0.0:
            tikhonov_lambda = float(tikhonov_alpha) * float(weights.sum())
            displacement_v = state.dv - float(initial_offset_vu[0])
            displacement_u = state.du - float(initial_offset_vu[1])
            hessian = hessian.copy()
            rhs = rhs.copy()
            hessian[0, 0] += tikhonov_lambda
            hessian[1, 1] += tikhonov_lambda
            rhs[0] += tikhonov_lambda * displacement_v
            rhs[1] += tikhonov_lambda * displacement_u
            cost_before = cost_before + tikhonov_lambda * (
                displacement_v * displacement_v + displacement_u * displacement_u
            )
        # LM dampening: H_lm = H + lambda * diag(H).
        diag = np.diag(np.diag(hessian))
        hessian_lm = hessian + lambda_ * diag
        try:
            step = -np.linalg.solve(hessian_lm, rhs)
        except np.linalg.LinAlgError:
            step = -pinvh(hessian_lm, rtol=pinvh_rcond) @ rhs
        trial_dv = state.dv + float(step[0])
        trial_du = state.du + float(step[1])
        trial_dtheta = state.dtheta + float(step[2]) if fit_rotation else state.dtheta
        # Trust-region rejection: refuse trial offsets that have walked
        # too far from the integer-precision coarse seed.  The IRLS-LM
        # combo can otherwise drift the polyline onto unrelated DT
        # minima (crater rims, terminator edges) when Tukey reweighting
        # at the trial offset finds a different inlier set.
        if trust_region_px is not None:
            disp = math.hypot(
                trial_dv - float(initial_offset_vu[0]),
                trial_du - float(initial_offset_vu[1]),
            )
            if disp > trust_region_px:
                lambda_ = min(lambda_ * 2.0, 1.0e6)
                state.iteration += 1
                if lambda_ >= 1.0e6:
                    break
                continue
        # Evaluate cost at the trial point.
        trial_residuals, _ = _compute_residuals_and_jacobian(
            vertices_vu=verts,
            pivot_vu=pivot,
            image_dt=image_edge_dt,
            dv=trial_dv,
            du=trial_du,
            dtheta=trial_dtheta,
            fit_rotation=fit_rotation,
        )
        trial_residuals = np.where(state.polarity_mask, trial_residuals, _INFINITY_DT_PENALTY_PX)
        # Compare ``trial_cost`` against ``cost_before`` using the SAME
        # weights computed at the start of this iteration.  Recomputing
        # Tukey biweights at the trial offset (the alternative)
        # parameterizes the cost function by the offset itself, so an
        # "improvement" can mean "the trial offset's reweighting found
        # a different inlier set whose sum-of-squares is lower" rather
        # than "the trial offset has smaller residuals at the current
        # inlier set".  Freezing the weights for the inner LM step is
        # the standard IRLS / LM separation that keeps the cost
        # function fixed during a single Gauss-Newton step; IRLS
        # reweighting still happens between iterations of the outer
        # loop.  Without this, the LM can drift onto unrelated DT
        # minima (crater rims, terminator edges) on textured bodies
        # where multiple basins are reachable from the seed.
        trial_cost = _weighted_cost(weights, trial_residuals)
        if tikhonov_alpha > 0.0:
            trial_disp_v = trial_dv - float(initial_offset_vu[0])
            trial_disp_u = trial_du - float(initial_offset_vu[1])
            trial_cost = trial_cost + tikhonov_lambda * (
                trial_disp_v * trial_disp_v + trial_disp_u * trial_disp_u
            )
        if trial_cost < cost_before:
            state.dv = trial_dv
            state.du = trial_du
            state.dtheta = trial_dtheta
            lambda_ = max(lambda_ * 0.5, 1.0e-12)
            best_cost = trial_cost
            step_norm = _step_norm_px(
                step,
                fit_rotation=fit_rotation,
                pivot_distance_px=pivot_distance_px,
            )
            state.iteration += 1
            # Recompute residuals / weights / Jacobian at the accepted pose
            # immediately so diagnostics (rms_px, inlier_count, covariance)
            # reflect the committed parameters even if the loop exits via
            # max_iterations on the next check.  Without this, an accepted
            # step in the final iteration would leave state.raw_residuals /
            # state.weights / state.jacobian reflecting the pre-step pose.
            residuals_final, jacobian_final = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
            residuals_final = np.where(
                state.polarity_mask, residuals_final, _INFINITY_DT_PENALTY_PX
            )
            final_scaled = residuals_final / sigmas
            final_tukey = tukey_biweight_weights(final_scaled, c=tukey_c)
            state.raw_residuals = residuals_final
            state.weights = inv_sigma_sq * final_tukey * state.polarity_mask
            state.jacobian = jacobian_final
            if step_norm < step_tolerance_px:
                state.converged = True
                break
        else:
            lambda_ = min(lambda_ * 2.0, 1.0e6)
            state.iteration += 1
            if lambda_ >= 1.0e6:
                break
    if best_cost == math.inf:
        # Iteration exited without finding any improvement: fall back to
        # the latest residual / weight set already cached on ``state``.
        residuals_final = state.raw_residuals
        if residuals_final.size == 0:
            residuals_final = sample_dt_bilinear(
                image_edge_dt,
                _shift_vertices(_rotate_vertices(verts, pivot, state.dtheta), state.dv, state.du),
            )
            residuals_final = np.where(
                state.polarity_mask, residuals_final, _INFINITY_DT_PENALTY_PX
            )
            final_scaled = residuals_final / sigmas
            final_tukey = tukey_biweight_weights(final_scaled, c=tukey_c)
            state.raw_residuals = residuals_final
            state.weights = inv_sigma_sq * final_tukey * state.polarity_mask
            _, state.jacobian = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
    # Final continuous gradient-ridge stage.  The DT-LM above minimized
    # distance to an integer-quantized edge mask, whose zero-set snaps the
    # recovered edge to integer pixels and leaves a sub-pixel-phase bias
    # floor; this stage polishes the offset against the un-quantized
    # gradient magnitude.  Only run when there is surviving DT evidence to
    # anchor it (a degenerate DT-LM has no inlier set to start from); the
    # ridge stage itself keeps the DT-LM pose if it finds no usable ridge
    # residual or walks too far.  The DT residuals / weights / Jacobian are
    # then recomputed at the refined pose so the reported rms / covariance /
    # spurious gates stay on the same DT footing as the no-ridge path.
    if final_gradient_ridge and image_gradient_vu is not None and np.any(state.weights > 0):
        gradient_magnitude = np.hypot(image_gradient_vu[..., 0], image_gradient_vu[..., 1]).astype(
            np.float64
        )
        ridge = _ridge.gradient_ridge_refine(
            vertices_vu=verts,
            normals_vu=norms,
            sigma_normal_per_vertex_px=sigmas,
            gradient_magnitude=gradient_magnitude,
            initial_offset_vu=(state.dv, state.du),
            weight_mask=state.polarity_mask.astype(np.float64),
            initial_rotation_rad=state.dtheta,
            fit_rotation=fit_rotation,
            pivot_vu=pivot,
            pivot_distance_px=pivot_distance_px,
            tukey_c=tukey_c,
        )
        if ridge.applied:
            state.dv, state.du = ridge.offset_vu
            state.dtheta = ridge.rotation_rad
            # A converged ridge stage verifies the reported pose even when
            # the DT-LM burned its iteration budget stalled on the
            # integer-quantized DT (see ``LMRefineResult.converged``).
            state.converged = state.converged or ridge.converged
            residuals_ridge, jacobian_ridge = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
            residuals_ridge = np.where(
                state.polarity_mask, residuals_ridge, _INFINITY_DT_PENALTY_PX
            )
            ridge_scaled = residuals_ridge / sigmas
            ridge_tukey = tukey_biweight_weights(ridge_scaled, c=tukey_c)
            state.raw_residuals = residuals_ridge
            state.weights = inv_sigma_sq * ridge_tukey * state.polarity_mask
            state.jacobian = jacobian_ridge
    final_weights = state.weights
    final_residuals = state.raw_residuals
    inlier_count = int(np.sum(final_weights > 0))
    degenerate = inlier_count == 0 or final_weights.sum() == 0.0
    if not degenerate:
        rms_px = float(math.sqrt(np.sum(final_weights * final_residuals**2) / final_weights.sum()))
    else:
        # Every vertex was rejected: there is no surviving evidence to
        # constrain the fit.  Report +inf (not 0.0) so the DT techniques'
        # ``result.rms_px > floor`` spurious test fires; a zero RMS would
        # otherwise be read downstream as a perfect fit.
        rms_px = float('inf')
    # Unweighted RMS over the POLARITY-ACCEPTED vertices, using their real
    # DT residuals.  Unlike the Tukey-weighted ``rms_px``, this does not
    # down-weight Tukey outliers, so a mis-converged fit whose bad arc was
    # rejected to ~0 weight still reports a large raw RMS -- the gate's
    # purpose.  Polarity-rejected vertices are excluded here: they carry the
    # large ``_INFINITY_DT_PENALTY_PX`` sentinel in ``final_residuals``, so
    # pooling them in would make ``raw_rms_px`` explode to ~1e6 / sqrt(N)
    # whenever even one vertex is polarity-rejected, degenerating the
    # limb / terminator ``raw_rms_px > floor`` gate into "any polarity
    # rejection is spurious".  That wrongly kills multi-body fits where a
    # small secondary body contributes wrong-polarity vertices while a
    # dominant body's limb fits cleanly.  Polarity rejection is instead
    # governed by the inlier-fraction gate; ``raw_rms_px`` measures only the
    # geometry of the accepted vertices.  When every vertex is polarity-
    # rejected there is no evidence, so report ``+inf`` (consistent with the
    # ``rms_px = +inf`` degenerate branch above).
    accepted = state.polarity_mask
    if final_residuals.size and bool(accepted.any()):
        accepted_residuals = final_residuals[accepted]
        raw_rms_px = float(math.sqrt(float(np.mean(accepted_residuals**2))))
    else:
        raw_rms_px = float('inf')
    covariance: NDArrayFloatType
    # The reported covariance is DATA-ONLY: it is the pseudoinverse of the
    # M-estimator information matrix ``J^T diag(w) J`` evaluated at the
    # converged pose and deliberately EXCLUDES the Tikhonov anchor
    # contribution (``tikhonov_alpha`` adds ``alpha * sum(w)`` to the
    # translation diagonal of the *iteration* Hessian to bias the step,
    # but that prior is a fitting aid, not measured information, so it
    # must not shrink the reported uncertainty).
    #
    # Guard against the "no information" cases that would otherwise let
    # information_matrix_to_covariance produce a misleading zero-covariance
    # answer (pinvh of the zero information matrix is zero — which would
    # falsely advertise perfect certainty about the fit).  These conditions
    # mean there is no inlier evidence to constrain the parameters; the inf
    # sentinel correctly signals "fully unconstrained" and stays consistent
    # with the ``rms_px = +inf`` degenerate result above.
    if state.jacobian.size == 0 or state.jacobian.shape[1] != n_params or degenerate:
        covariance = cast(NDArrayFloatType, np.full((n_params, n_params), np.inf, dtype=np.float64))
    else:
        covariance = information_matrix_to_covariance(
            state.jacobian, final_weights, rcond=pinvh_rcond
        )
    return LMRefineResult(
        offset_vu=(state.dv, state.du),
        rotation_rad=state.dtheta,
        covariance=covariance,
        residuals_px=final_residuals,
        weights=final_weights,
        rms_px=rms_px,
        raw_rms_px=raw_rms_px,
        iterations=state.iteration,
        converged=state.converged,
        inlier_count=inlier_count,
        degenerate=degenerate,
        polarity_rejected_count=int(np.count_nonzero(~state.polarity_mask)),
    )
