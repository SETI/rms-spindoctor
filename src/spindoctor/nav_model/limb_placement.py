"""Place a planet's predicted limb where the image shows it.

The discrete silhouette extraction in
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` marks the outermost
pixel whose center lies on the body, so every limb vertex sits at a pixel center
somewhere between zero and one pixel inside the silhouette.  For a planet with an
atmosphere the image's edge is not the silhouette of its reference ellipsoid at
all but the haze some tens to hundreds of kilometers above it.  On a partial arc
the limb fit has no radial freedom to absorb that difference, so it turns into a
translation along the arc's normal.

:class:`~spindoctor.config.visible_limb.VisibleLimbModel` holds an instrument's
measured height of that edge for one body, and :func:`place_on_visible_limb` moves
each vertex along its outward normal to it, using the geometry layer's
limb-altitude backplane: the altitude of the line of sight's closest approach above
the body's ellipsoid, which is negative on the body and grows by one pixel's worth
of kilometers per pixel off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import polymath
from oops import Meshgrid
from oops.backplane import Backplane

from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from oops import Observation

__all__ = [
    'VisibleLimbPlacement',
    'limb_altitudes_km',
    'place_on_visible_limb',
]


@dataclass(frozen=True)
class VisibleLimbPlacement:
    """Per-vertex result of :func:`place_on_visible_limb`.

    Parameters:
        offset_px: ``(N,)`` distance (pixels) each vertex moves along its outward
            normal to reach the visible limb.  NaN where the geometry gives no
            usable answer.
        px_per_km: ``(N,)`` pixels moved along the normal per kilometer of limb
            altitude, the sensitivity of the vertex position to the altitude.
            NaN where ``offset_px`` is NaN.
        valid: ``(N,)`` True where both are finite.
    """

    offset_px: NDArrayFloatType
    px_per_km: NDArrayFloatType
    valid: NDArrayBoolType


def limb_altitudes_km(
    obs: Observation,
    body_name: str,
    uv: NDArrayFloatType,
) -> NDArrayFloatType:
    """Return the limb altitude of the line of sight through each sensor point.

    The limb altitude is the height above the body's ellipsoid of the point where
    the line of sight passes closest to it: positive off the body, negative on it.

    Parameters:
        obs: The observation; its field of view maps sensor coordinates to lines
            of sight.
        body_name: SPICE name of the body.
        uv: ``(M, 2)`` sensor ``(u, v)`` points in pixel-corner coordinates.

    Returns:
        ``(M,)`` altitudes in kilometers; NaN where the geometry layer masks one.
    """
    meshgrid = Meshgrid(obs.fov, polymath.Pair(np.asarray(uv, dtype=np.float64)))
    altitude = Backplane(obs, meshgrid).limb_altitude(body_name)
    values = np.asarray(altitude.vals, dtype=np.float64)
    masked = np.broadcast_to(np.asarray(altitude.mask, dtype=bool), values.shape)
    return cast(NDArrayFloatType, np.where(masked, np.nan, values))


def place_on_visible_limb(
    obs: Observation,
    body_name: str,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    *,
    altitude_km: float,
) -> VisibleLimbPlacement:
    """Measure how far each limb vertex lies from the visible limb along its normal.

    The limb altitude (the closest approach of the pixel's line of sight above the
    body's ellipsoid, negative on the body) is evaluated at each vertex and one pixel
    out along its normal.  Between the two the altitude is linear in the distance
    along the normal to well under a percent of a pixel, so the vertex reaches
    altitude ``altitude_km`` after ``(altitude_km - a0) / (a1 - a0)`` pixels.  A
    vertex whose altitude does not increase outward (a normal that does not point
    off the body) or whose altitude is masked gets NaN.

    Parameters:
        obs: The observation; its field of view maps sensor coordinates to lines
            of sight.
        body_name: SPICE name of the body.
        vertices_vu: ``(N, 2)`` vertices in extended-frame, pixel-centric
            coordinates.
        normals_vu: ``(N, 2)`` unit outward normals.
        altitude_km: Altitude of the visible limb above the ellipsoid, in
            kilometers; 0 places the vertices on the silhouette.

    Returns:
        The per-vertex :class:`VisibleLimbPlacement`.
    """
    vertices = np.asarray(vertices_vu, dtype=np.float64)
    normals = np.asarray(normals_vu, dtype=np.float64)
    n = vertices.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return VisibleLimbPlacement(
            offset_px=empty, px_per_km=empty.copy(), valid=np.empty(0, dtype=bool)
        )
    margin_v = float(obs.extfov_margin_v)
    margin_u = float(obs.extfov_margin_u)
    # Extended-frame pixel-centric (v, u) -> sensor pixel-corner (u, v), the
    # coordinates the field of view reads.
    uv0 = np.stack(
        [
            vertices[:, 1] - margin_u + PIXEL_CENTER_TO_CORNER_PX,
            vertices[:, 0] - margin_v + PIXEL_CENTER_TO_CORNER_PX,
        ],
        axis=-1,
    )
    uv1 = uv0 + normals[:, ::-1]
    values = limb_altitudes_km(obs, body_name, np.concatenate([uv0, uv1], axis=0))
    a0 = values[:n]
    a1 = values[n:]
    with np.errstate(invalid='ignore', divide='ignore'):
        slope_km_per_px = a1 - a0
        rising = np.isfinite(slope_km_per_px) & (slope_km_per_px > 0.0)
        offset = np.where(rising, (altitude_km - a0) / slope_km_per_px, np.nan)
        px_per_km = np.where(rising, 1.0 / slope_km_per_px, np.nan)
    valid = np.isfinite(offset) & np.isfinite(px_per_km)
    return VisibleLimbPlacement(
        offset_px=cast(NDArrayFloatType, offset),
        px_per_km=cast(NDArrayFloatType, px_per_km),
        valid=valid,
    )
