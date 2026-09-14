"""Pose transforms and weighted normal equations shared by the fitting stages.

Pure array helpers: rotate / shift a polyline, sample the DT and
finite-difference its parameter Jacobian, and assemble the weighted normal
equations the LM and ridge steps solve.
"""

import math
from typing import cast

import numpy as np

from spindoctor.support.distance_transform import sample_dt_bilinear
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    '_compute_residuals_and_jacobian',
    '_rotate_directions',
    '_rotate_vertices',
    '_shift_vertices',
    '_step_norm_px',
    '_weighted_cost',
    '_weighted_normal_equations',
]


def _rotate_vertices(
    vertices_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    theta: float,
) -> NDArrayFloatType:
    """Rotate ``vertices_vu`` about ``pivot_vu`` by ``theta`` radians."""
    if theta == 0.0:
        return vertices_vu
    pv, pu = pivot_vu
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centered_v = vertices_vu[:, 0] - pv
    centered_u = vertices_vu[:, 1] - pu
    new_v = pv + cos_t * centered_v - sin_t * centered_u
    new_u = pu + sin_t * centered_v + cos_t * centered_u
    return cast(NDArrayFloatType, np.stack([new_v, new_u], axis=-1))


def _rotate_directions(
    directions_vu: NDArrayFloatType,
    theta: float,
) -> NDArrayFloatType:
    """Rotate direction vectors (normals) by ``theta`` radians about the origin.

    Unlike :func:`_rotate_vertices` there is no pivot: a normal is a free
    vector, so only the in-plane rotation applies.  Used by the
    gradient-ridge stage to keep each vertex's outward normal aligned with
    the body after a rotation step.
    """
    if theta == 0.0:
        return directions_vu
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    nv = directions_vu[:, 0]
    nu = directions_vu[:, 1]
    rot_v = cos_t * nv - sin_t * nu
    rot_u = sin_t * nv + cos_t * nu
    return cast(NDArrayFloatType, np.stack([rot_v, rot_u], axis=-1))


def _shift_vertices(
    vertices_vu: NDArrayFloatType,
    dv: float,
    du: float,
) -> NDArrayFloatType:
    """Return a copy of ``vertices_vu`` shifted by ``(dv, du)``."""
    out = vertices_vu.copy()
    out[:, 0] += dv
    out[:, 1] += du
    return out


def _compute_residuals_and_jacobian(
    *,
    vertices_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    image_dt: NDArrayFloatType,
    dv: float,
    du: float,
    dtheta: float,
    fit_rotation: bool,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Sample the DT and finite-difference its parameter Jacobian.

    Bilinear DT samples are differentiable almost everywhere, so a
    central-difference Jacobian with a small step gives a good local
    linearisation.  The step is fixed at 0.25 pixels — large enough to
    cross several DT bilinear cells (the DT grows by 1 every full
    pixel), small enough to remain within the bilinear interpolant's
    local quadratic regime.
    """
    base_pos = _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta), dv, du)
    residuals = sample_dt_bilinear(image_dt, base_pos)
    # Central differences with a sub-pixel step.  A 0.25 px step is large
    # enough to cross several DT bilinear cells (the DT grows by 1 every
    # full pixel) but small enough that the linearisation stays inside
    # the local quadratic regime around the converged offset.
    eps = 0.25
    pdv = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, eps, 0.0))
    mdv = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, -eps, 0.0))
    pdu = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, 0.0, eps))
    mdu = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, 0.0, -eps))
    drdv = (pdv - mdv) / (2.0 * eps)
    drdu = (pdu - mdu) / (2.0 * eps)
    if not fit_rotation:
        jacobian = np.stack([drdv, drdu], axis=-1)
        return residuals, cast(NDArrayFloatType, jacobian)
    # Numerical derivative w.r.t. dtheta
    eps_t = 1.0e-3
    plus = sample_dt_bilinear(
        image_dt,
        _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta + eps_t), dv, du),
    )
    minus = sample_dt_bilinear(
        image_dt,
        _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta - eps_t), dv, du),
    )
    drdth = (plus - minus) / (2.0 * eps_t)
    jacobian = np.stack([drdv, drdu, drdth], axis=-1)
    return residuals, cast(NDArrayFloatType, jacobian)


def _weighted_normal_equations(
    jacobian: NDArrayFloatType,
    residuals: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return ``(J^T W J, J^T W r)`` weighted by ``weights``."""
    sqrt_w = np.sqrt(weights)
    weighted_j = sqrt_w[:, None] * jacobian
    weighted_r = sqrt_w * residuals
    hessian = weighted_j.T @ weighted_j
    rhs = weighted_j.T @ weighted_r
    return cast(NDArrayFloatType, hessian), cast(NDArrayFloatType, rhs)


def _weighted_cost(weights: NDArrayFloatType, residuals: NDArrayFloatType) -> float:
    """Sum of ``w_i * r_i**2`` (the quantity LM minimizes)."""
    return float(np.sum(weights * residuals * residuals))


def _step_norm_px(
    step: NDArrayFloatType,
    *,
    fit_rotation: bool,
    pivot_distance_px: float,
) -> float:
    """Return the LM convergence step norm in pixel-equivalent units.

    Translation steps contribute their Euclidean magnitude directly; the
    rotation step (when present) is multiplied by ``pivot_distance_px``
    to convert radians into a pixel displacement at the pivot's typical
    distance.  Combined as a Euclidean norm so the same tolerance
    threshold applies to translation-only and translation-plus-rotation
    fits.
    """
    if not fit_rotation:
        return float(math.hypot(step[0], step[1]))
    rotation_step_px = float(step[2]) * pivot_distance_px
    return float(math.sqrt(step[0] ** 2 + step[1] ** 2 + rotation_step_px**2))
