"""Sub-pixel refinement of body-boundary polyline vertices by geometric probing.

The discrete limb / terminator extraction places each polyline vertex at the
integer center of a boundary ridge pixel, so the emitted polyline is
quantized to the pixel grid: a sub-pixel change in the predicted pointing
re-rasterizes the ridge instead of translating it, and any fit against the
polyline inherits a sub-pixel-phase-dependent bias of up to half a pixel.
This module removes that quantization at the source.  For every ridge vertex
a short ladder of probe points is laid out along the vertex's outward
normal, each probe is classified inside / outside the target region by
evaluating the body geometry at that exact sub-pixel line of sight, and the
vertex is moved to the midpoint of the two probes that bracket the boundary.
The refined vertex positions then move continuously with the predicted
geometry, restoring shift equivariance of the downstream distance-transform
fits.

The probing is purely geometric (an oops ``Backplane`` over a scattered
``Meshgrid`` of the probe positions); it does not depend on the rendering
oversample factor, so it stays sub-pixel-accurate for bodies too large for
oversampled rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from oops import Meshgrid
from oops.backplane import Backplane
from polymath import Pair

from spindoctor.obs import ObsSnapshot
from spindoctor.support.constants import HALFPI
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'PROBE_OFFSETS_PX',
    'boundary_crossing_offsets',
    'probe_positions_uv',
    'refine_polyline_vertices',
    'refined_vertex_positions',
]

PROBE_OFFSETS_PX: NDArrayFloatType = np.linspace(-1.0, 1.5, 21)
"""Probe ladder along each vertex normal, in pixels (step 0.125).

A ridge pixel is the outermost pixel whose sampled center (or any rendering
subsample) lies inside the region, so the true boundary lies within one
pixel outward of the vertex along the boundary normal -- up to ``sqrt(2)``
pixels for a diagonally quantized normal, hence the ``+1.5`` outer reach.
The ``-1.0`` inner reach covers ridge pixels whose own center falls just
outside the region (possible when the rendering oversample is above one).
The 0.125 px step bounds the refined vertex quantization at 1/16 px after
midpoint bracketing, well below the technique fits' sub-0.1 px resolution.
"""
PROBE_OFFSETS_PX.setflags(write=False)


def boundary_crossing_offsets(inside: NDArrayBoolType, ts: NDArrayFloatType) -> NDArrayFloatType:
    """Locate the region boundary along each vertex's probe ladder.

    For each row the crossing is the midpoint of the two adjacent probes
    that bracket the region boundary nearest the vertex: when the vertex's
    own probe (``t = 0``) is inside the region the first outside probe
    outward of it is used; when it is outside, the first inside probe inward
    of it.  A row with no such bracket (entirely inside, or no region found
    inward) yields ``NaN``, meaning the crossing is undetermined and the
    vertex should be left where it is.

    Parameters:
        inside: ``(N, T)`` boolean matrix; ``inside[i, j]`` is True when
            probe ``j`` of vertex ``i`` falls inside the target region.
        ts: ``(T,)`` strictly-increasing probe offsets in pixels, containing
            an exact ``0.0`` entry (the vertex itself).

    Returns:
        ``(N,)`` crossing offsets in pixels along each vertex's normal,
        ``NaN`` where undetermined.

    Raises:
        ValueError: ``ts`` does not match the ``inside`` columns, is not
            finite and strictly increasing, or has no exact ``0.0`` entry.
    """
    n_vertices, n_probes = inside.shape
    if ts.shape != (n_probes,):
        raise ValueError(
            f'ts must have shape ({n_probes},) matching inside columns; got {ts.shape}'
        )
    if not np.all(np.isfinite(ts)) or not np.all(np.diff(ts) > 0.0):
        raise ValueError('ts must be finite and strictly increasing')
    zero_matches = np.nonzero(ts == 0.0)[0]
    if zero_matches.size != 1:
        raise ValueError('ts must contain exactly one 0.0 entry (the vertex itself)')
    zero_idx = int(zero_matches[0])
    offsets = np.full(n_vertices, np.nan, dtype=np.float64)

    inside0 = inside[:, zero_idx]
    # Vertex inside the region: first outside probe outward of it.
    outward = ~inside[:, zero_idx:]
    has_exit = outward.any(axis=1)
    exit_idx = np.argmax(outward, axis=1)
    rows = inside0 & has_exit
    k = zero_idx + exit_idx[rows]
    offsets[rows] = 0.5 * (ts[k - 1] + ts[k])
    # Vertex outside the region: first inside probe inward of it.
    inward = inside[:, :zero_idx][:, ::-1]
    has_region = inward.any(axis=1)
    region_idx = np.argmax(inward, axis=1)
    rows = ~inside0 & has_region
    j = zero_idx - 1 - region_idx[rows]
    offsets[rows] = 0.5 * (ts[j] + ts[j + 1])
    return offsets


def probe_positions_uv(
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    ts: NDArrayFloatType,
    *,
    margin_vu: tuple[int, int],
) -> NDArrayFloatType:
    """Build the ``(N, T, 2)`` probe positions in FOV ``(u, v)`` coordinates.

    Each vertex's probes step along its outward normal by the offsets in
    ``ts``.  Vertices are given in extended-FOV pixel coordinates (integer
    values name pixel centers); FOV coordinates place pixel ``i``'s center
    at ``i + 0.5``, so the conversion subtracts the extended-FOV margin and
    adds the half-pixel center offset.

    Parameters:
        vertices_vu: ``(N, 2)`` vertex positions in extended-FOV ``(v, u)``.
        normals_vu: ``(N, 2)`` unit outward normals in ``(v, u)``.
        ts: ``(T,)`` probe offsets in pixels.
        margin_vu: ``(margin_v, margin_u)`` extended-FOV margins in pixels.

    Returns:
        ``(N, T, 2)`` array of probe positions with ``(u, v)`` in the last
        axis, ready to wrap in a ``polymath.Pair`` for an oops ``Meshgrid``.
    """
    margin_v, margin_u = margin_vu
    v_fov = vertices_vu[:, 0] - margin_v + 0.5
    u_fov = vertices_vu[:, 1] - margin_u + 0.5
    u_probe = u_fov[:, None] + ts[None, :] * normals_vu[:, 1][:, None]
    v_probe = v_fov[:, None] + ts[None, :] * normals_vu[:, 0][:, None]
    return np.stack([u_probe, v_probe], axis=-1)


def refined_vertex_positions(
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    offsets: NDArrayFloatType,
) -> NDArrayFloatType:
    """Apply per-vertex boundary crossings along the normals.

    Parameters:
        vertices_vu: ``(N, 2)`` vertex positions in extended-FOV ``(v, u)``.
        normals_vu: ``(N, 2)`` unit outward normals in ``(v, u)``.
        offsets: ``(N,)`` crossing offsets in pixels; ``NaN`` leaves the
            corresponding vertex unchanged.

    Returns:
        ``(N, 2)`` refined copy of ``vertices_vu``.  A vertex whose offset is
        non-finite, or whose normal has any non-finite component, keeps its
        input position whole: the refinement must never corrupt a vertex it
        cannot place.
    """
    valid = np.isfinite(offsets) & np.all(np.isfinite(normals_vu), axis=1)
    shift = np.where(valid, offsets, 0.0)
    safe_normals: NDArrayFloatType = np.where(valid[:, None], normals_vu, 0.0)
    return vertices_vu + shift[:, None] * safe_normals


def refine_polyline_vertices(
    obs: ObsSnapshot,
    body_name: str,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    *,
    region: Literal['silhouette', 'lit'],
    meshgrid_cls: Callable[..., Any] = Meshgrid,
    backplane_cls: Callable[..., Any] = Backplane,
) -> NDArrayFloatType:
    """Refine ridge-pixel polyline vertices onto the body's sub-pixel boundary.

    Evaluates the body geometry at every probe position via an oops
    ``Backplane`` over a scattered ``Meshgrid`` (about ``21 * N`` lines of
    sight -- negligible next to the model's silhouette render) and moves
    each vertex to the probed boundary crossing along its outward normal.
    Vertices whose crossing is undetermined keep their original position.

    Parameters:
        obs: The observation whose FOV defines the probe lines of sight.
        body_name: SPICE body name whose boundary is probed.
        vertices_vu: ``(N, 2)`` ridge vertex positions in extended-FOV
            ``(v, u)`` coordinates.
        normals_vu: ``(N, 2)`` unit outward normals (region inside to
            outside) in ``(v, u)``.
        region: ``'silhouette'`` probes the body-intercept silhouette (the
            limb boundary); ``'lit'`` probes the lit portion of the disc
            (the boundary is the terminator on the sunward side).
        meshgrid_cls: Constructor called as ``meshgrid_cls(fov, uv_pair)``.
            The caller may pass its own module's (patchable) name so a test
            double intercepts the probe exactly as it intercepts the render.
        backplane_cls: Constructor called as
            ``backplane_cls(obs, meshgrid=...)``; same substitution hook.

    Returns:
        ``(N, 2)`` refined vertex positions.
    """
    if vertices_vu.shape[0] == 0:
        return vertices_vu
    uv = probe_positions_uv(
        vertices_vu,
        normals_vu,
        PROBE_OFFSETS_PX,
        margin_vu=(obs.extfov_margin_v, obs.extfov_margin_u),
    )
    meshgrid = meshgrid_cls(obs.fov, Pair(uv))
    backplane = backplane_cls(obs, meshgrid=meshgrid)
    incidence = backplane.incidence_angle(body_name)
    intercepted = ~incidence.expand_mask().mask
    if region == 'silhouette':
        inside = intercepted
    else:
        inside = intercepted & (incidence.vals < HALFPI)
    offsets = boundary_crossing_offsets(inside, PROBE_OFFSETS_PX)
    return refined_vertex_positions(vertices_vu, normals_vu, offsets)
