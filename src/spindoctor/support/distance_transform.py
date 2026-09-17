"""Chamfer-matching helpers built on top of the distance transform.

DT-based techniques (BodyLimbNav, BodyTerminatorNav, RingEdgeNav) all rely on
the same primitive: given an image-side distance transform of an edge map and
a model polyline, evaluate the cost of a candidate offset by sampling the DT
at the shifted polyline vertices.

This module owns the polyline-shifting and DT-sampling routines so the three
techniques don't reimplement them.  Bilinear DT interpolation is used for
sub-pixel precision; the orchestrator's ``NavContext`` carries the per-image
DT array so callers reach in once and the gradient is shared.
"""

from typing import cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'apply_translation',
    'sample_dt_bilinear',
]


def apply_translation(vertices_vu: NDArrayFloatType, dv: float, du: float) -> NDArrayFloatType:
    """Return ``vertices_vu`` shifted by ``(dv, du)``.

    Parameters:
        vertices_vu: ``(N, 2)`` array of (v, u) vertex positions.
        dv: Shift along the v axis (rows).
        du: Shift along the u axis (columns).

    Returns:
        ``(N, 2)`` shifted vertex positions; original is not modified.

    Raises:
        ValueError: if ``vertices_vu`` is not ``(N, 2)``.
    """
    arr = np.asarray(vertices_vu, np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'vertices_vu must have shape (N, 2); got {arr.shape}')
    out = arr.copy()
    out[:, 0] += dv
    out[:, 1] += du
    return cast(NDArrayFloatType, out)


def sample_dt_bilinear(dt: NDArrayFloatType, vertices_vu: NDArrayFloatType) -> NDArrayFloatType:
    """Bilinear-sample a distance transform at sub-pixel vertex positions.

    Vertices outside ``dt`` are clamped to the boundary value at the closest
    in-bounds pixel.  The result is the DT cost contribution per vertex; the
    sum (or the weighted sum, with M-estimator weights) is what an LM
    refinement minimizes.

    The sampler reads a vertex pixel centric: it clamps to ``h - 1`` and
    ``w - 1`` and takes ``floor(v)``, so a whole number samples that pixel's
    own value.  This is the crossing point for every distance-transform
    technique.  A caller handing it pixel corner coordinates displaces every
    vertex by half a pixel in the same direction, and that error does not
    cancel: the fit minimizes the residual rather than differencing two of
    them, so the bias lands in the answer.

    Parameters:
        dt: 2-D distance-transform array (output of ``apply_filter`` with
            ``DISTANCE_TRANSFORM`` kind, or an externally-built DT).
        vertices_vu: ``(N, 2)`` array of (v, u) sub-pixel vertex positions,
            pixel centric.

    Returns:
        ``(N,)`` array of bilinear-interpolated DT values at each vertex.

    Raises:
        ValueError: if shape requirements are violated.
    """
    if dt.ndim != 2:
        raise ValueError(f'dt must be 2-D; got ndim={dt.ndim}')
    arr = np.asarray(vertices_vu, np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f'vertices_vu must have shape (N, 2); got {arr.shape}')
    h, w = dt.shape
    v = np.clip(arr[:, 0], 0.0, h - 1.0)
    u = np.clip(arr[:, 1], 0.0, w - 1.0)
    v0 = np.floor(v).astype(np.int64)
    u0 = np.floor(u).astype(np.int64)
    v1 = np.minimum(v0 + 1, h - 1)
    u1 = np.minimum(u0 + 1, w - 1)
    fv = v - v0
    fu = u - u0
    a = dt[v0, u0]
    b = dt[v0, u1]
    c = dt[v1, u0]
    d = dt[v1, u1]
    interp = a * (1.0 - fv) * (1.0 - fu) + b * (1.0 - fv) * fu + c * fv * (1.0 - fu) + d * fv * fu
    return cast(NDArrayFloatType, interp.astype(np.float64))
