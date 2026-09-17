"""The navigator's predicted-ring renderer for simulated scenes.

Predicts a ``ring_system`` feature's appearance from its idealized catalog
view (the ``obs.nav_params['ring_system']`` block): the shared projection
geometry, the feature's kind/shape keys, and its catalog orbit -- never the
planted ``orbit_error`` or the photometric truth, which the boundary filter
strips before this code can see them.  Projection and orbit math are shared
with the image-side renderer through :mod:`spindoctor.sim.ring_geometry`, so
a predicted edge lands where the rendered edge would land if the scene
planted no error: the planted pointing offset and the planted orbit error
are the only discrepancies, by construction.

The prediction is geometric, not photometric: a banded feature yields a
solid anti-aliased coverage template (the navigator's opaque-annulus
convention; the tau photometry is truth) plus one border polyline per
catalog edge with outward radial normals for the distance-transform fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from spindoctor.sim.ring_geometry import (
    compute_antialiasing_shade,
    compute_orbit_radii,
    ring_orbit_from_mapping,
    ring_plane_from_sky,
    ring_radial_scale,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'PredictedRingEdge',
    'PredictedRingFeature',
    'predict_ring_feature',
]


@dataclass(frozen=True)
class PredictedRingEdge:
    """One predicted catalog edge of a ring feature.

    Parameters:
        edge_type: 'inner' or 'outer' (the banded kinds), or 'edge' (the
            single-boundary kinds).
        mask: Boolean border mask on the prediction grid (1-pixel polyline).
        vertices_vu: ``(N, 2)`` border-pixel positions.
        normals_vu: ``(N, 2)`` unit normals pointing radially outward in the
            ring plane (the direction of increasing ring radius on the sky).
    """

    edge_type: str
    mask: NDArrayBoolType
    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType


@dataclass(frozen=True)
class PredictedRingFeature:
    """The navigator's rendered prediction of one ring feature.

    Parameters:
        template: Solid anti-aliased coverage in [0, 1] for the correlation
            path, or None for kinds with no bounded band (edge / ramp /
            wave, and gaps, which reveal rather than emit).
        mask: Pixels the feature's band covers (or its border for the
            single-boundary kinds); the annotation avoid mask.
        edges: The predicted catalog edges.
    """

    template: NDArrayFloatType | None
    mask: NDArrayBoolType
    edges: list[PredictedRingEdge]


def _border_from_signed_distance(diff: NDArrayFloatType) -> NDArrayBoolType:
    """The 1-pixel border where a signed distance field changes sign.

    A pixel joins the border when its sign differs from a 4-neighbor's and
    it is at least as close to the zero crossing (mirroring the historical
    border-atop rasterization, so predicted edges keep their established
    1-pixel sampling).

    Parameters:
        diff: Per-pixel signed distance to the edge (any consistent units).

    Returns:
        The border mask.
    """
    sign = np.sign(diff)
    abs_diff = np.abs(diff)
    border: NDArrayBoolType = abs_diff == 0.0
    border[:-1, :] |= (sign[:-1, :] == -sign[1:, :]) & (abs_diff[:-1, :] <= abs_diff[1:, :])
    border[1:, :] |= (sign[1:, :] == -sign[:-1, :]) & (abs_diff[1:, :] <= abs_diff[:-1, :])
    border[:, :-1] |= (sign[:, :-1] == -sign[:, 1:]) & (abs_diff[:, :-1] <= abs_diff[:, 1:])
    border[:, 1:] |= (sign[:, 1:] == -sign[:, :-1]) & (abs_diff[:, 1:] <= abs_diff[:, :-1])
    return border


def _edge_from_diff(diff: NDArrayFloatType, edge_type: str) -> PredictedRingEdge:
    """Build a predicted edge (border mask, vertices, outward normals).

    The normal at each border pixel is the normalized gradient of the
    signed radial distance field: the sky-plane direction of increasing
    ring-plane radius, which is the outward radial direction however the
    projection foreshortens the edge.

    Parameters:
        diff: Per-pixel ``r - r_edge(lam)`` signed distance field.
        edge_type: The edge label ('inner' / 'outer' / 'edge').

    Returns:
        The predicted edge.
    """
    mask = _border_from_signed_distance(diff)
    if not mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return PredictedRingEdge(
            edge_type=edge_type, mask=mask, vertices_vu=empty, normals_vu=empty
        )
    grads = np.gradient(diff)
    grad_v = cast(NDArrayFloatType, grads[0])
    grad_u = cast(NDArrayFloatType, grads[1])
    vs, us = np.where(mask)
    vertices_vu = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    normals = np.stack([grad_v[vs, us], grad_u[vs, us]], axis=1)
    norms = np.hypot(normals[:, 0], normals[:, 1])
    norms[norms == 0.0] = 1.0
    normals_vu = normals / norms[:, None]
    return PredictedRingEdge(
        edge_type=edge_type, mask=mask, vertices_vu=vertices_vu, normals_vu=normals_vu
    )


def predict_ring_feature(
    shape: tuple[int, int],
    feature: dict[str, Any],
    *,
    center_v: float,
    center_u: float,
    opening_deg_obs: float,
    node_deg: float,
    time: float = 0.0,
    epoch: float = 0.0,
) -> PredictedRingFeature:
    """Predict one ring_system feature on the given (extfov) grid.

    Parameters:
        shape: The prediction-grid shape (detector resolution).
        feature: The feature's idealized mapping from ``nav_params``.
        center_v: Projected ring center v on the prediction grid.
        center_u: Projected ring center u on the prediction grid.
        opening_deg_obs: Observer ring opening angle B in degrees.
        node_deg: Sky position angle of the ascending node in degrees.
        time: Scene time in TDB seconds.
        epoch: Ring epoch in TDB seconds.

    Returns:
        The rendered :class:`PredictedRingFeature`.  Empty (no edges, no
        coverage) for an exactly edge-on geometry, which renders nothing.
    """
    size_v, size_u = shape
    empty_mask: NDArrayBoolType = np.zeros(shape, dtype=np.bool_)
    if math.sin(math.radians(opening_deg_obs)) == 0.0:
        return PredictedRingFeature(template=None, mask=empty_mask, edges=[])

    # The grid starts as array rows and columns and the projection works in
    # the geometry layer's pixel corner coordinates, where the center the
    # caller hands in is stated, so the half pixel goes on here.
    v_coords = np.arange(size_v, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_coords = np.arange(size_u, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')
    r, lam, x, y = ring_plane_from_sky(
        v_grid - center_v, u_grid - center_u, opening_deg_obs=opening_deg_obs, node_deg=node_deg
    )
    radial_scale = ring_radial_scale(r, x, y, opening_deg_obs=opening_deg_obs)

    kind = str(feature.get('kind'))
    orbit = ring_orbit_from_mapping(feature.get('orbit') or {})
    r_edge = compute_orbit_radii(lam, orbit, epoch=epoch, time=time)

    if kind in ('ringlet', 'gap'):
        width = float(feature.get('width', 0.0))
        r_outer = compute_orbit_radii(lam, orbit.widened(width), epoch=epoch, time=time)
        inner_shade = compute_antialiasing_shade((r - r_edge) / radial_scale, 1.0)
        outer_shade = compute_antialiasing_shade((r_outer - r) / radial_scale, 1.0)
        coverage = cast(NDArrayFloatType, np.minimum(inner_shade, outer_shade))
        band_mask: NDArrayBoolType = coverage > 0.0
        edges = [
            _edge_from_diff(cast(NDArrayFloatType, r - r_edge), 'inner'),
            _edge_from_diff(cast(NDArrayFloatType, r - r_outer), 'outer'),
        ]
        # A gap reveals the background rather than emitting, so it carries
        # no correlation template; its edges are still fittable boundaries.
        template = coverage if kind == 'ringlet' else None
        return PredictedRingFeature(template=template, mask=band_mask, edges=edges)

    if kind == 'ramp':
        # Only the ramp's sharp end is a fittable boundary; the linear end
        # fades into the background with no gradient to fit.
        width = float(feature.get('width', 0.0))
        side = str(feature.get('side', 'out'))
        if side == 'out':
            r_sharp = compute_orbit_radii(lam, orbit.widened(width), epoch=epoch, time=time)
            edge_type = 'outer'
        else:
            r_sharp = r_edge
            edge_type = 'inner'
        edge = _edge_from_diff(cast(NDArrayFloatType, r - r_sharp), edge_type)
        return PredictedRingFeature(template=None, mask=edge.mask, edges=[edge])

    # 'edge' (a one-sided step) and 'wave' (a train launched at the orbit
    # radius) each predict the single catalog boundary.
    edge = _edge_from_diff(cast(NDArrayFloatType, r - r_edge), 'edge')
    return PredictedRingFeature(template=None, mask=edge.mask, edges=[edge])
