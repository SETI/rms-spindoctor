"""Projection math for alternate 2-D display modes and the 3-D sphere view.

All projection functions operate in *normalized display units*.  The conversion
between normalized units and viewport pixels uses the linear transform::

    viewport_x = normalized_x * scale + cx
    viewport_y = normalized_y * scale + cy

where ``(cx, cy)`` is the projection center in viewport pixels and ``scale``
is pixels per normalized unit.  :class:`ProjectionParams` carries these values.

Normalized-unit conventions per kind
-------------------------------------
- **RECT** -- 1 unit = 1 degree.  x = longitude [0, 360), y = -latitude.
  This projection is rendered by the existing body-sphere canvas path;
  these functions are used only for graticule overlays.
- **POLAR_N** -- 1 unit = stereographic equatorial radius.  r = 0 is the
  north pole, r = 1 is the equator.  0 deg longitude appears at the top
  (screen-y direction).
- **POLAR_S** -- same as POLAR_N with latitude sign-flipped.  r = 0 is the
  south pole, r = 1 is the equator.
- **MOLLWEIDE** -- 1 unit = sqrt(2).  The full-globe ellipse spans
  x in [-2*sqrt(2), 2*sqrt(2)] and y in [-sqrt(2), sqrt(2)].
- **SPHERE_3D** -- 1 unit = sphere radius.  The visible disk has r <= 1.
  The camera views from the +X direction; yaw/pitch define which geographic
  point faces the camera.

Coordinate system
-----------------
Screen y increases *downward* (PyQt6 convention).  Normalized y also increases
downward to match, so ``vy = yn * scale + cy`` with no sign flip needed in
the viewport transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

_MOLLWEIDE_MAX_ITERATIONS = 10
_MOLLWEIDE_DENOM_EPSILON = 1e-14
_MOLLWEIDE_COS_EPSILON = 1e-10
_MOLLWEIDE_ELLIPSE_TOLERANCE = 1e-9
_FIT_SCALE_MARGIN = 0.9
_PROJECTION_SCALE_MIN = 1e-15


class ProjectionKind(Enum):
    """Available projection modes for the body mosaic viewer."""

    RECT = 'rect'
    POLAR_N = 'polar_n'
    POLAR_S = 'polar_s'
    MOLLWEIDE = 'mollweide'
    SPHERE_3D = 'sphere_3d'


_KNOWN_PROJECTION_KINDS: frozenset[ProjectionKind] = frozenset(ProjectionKind)


@dataclass
class ProjectionParams:
    """Viewport geometry for a single projection frame.

    Parameters:
        kind: Which projection to use.
        cx: Viewport X coordinate of the projection center (pixels from left).
        cy: Viewport Y coordinate of the projection center (pixels from top).
        scale: Pixels per normalized unit (see module docstring).
        yaw_deg: SPHERE_3D only -- longitude at the center of the view (deg).
        pitch_deg: SPHERE_3D only -- latitude at the center of the view (deg).
    """

    kind: ProjectionKind
    cx: float
    cy: float
    scale: float
    yaw_deg: float = field(default=0.0)
    pitch_deg: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Sphere-3D rotation helpers
# ---------------------------------------------------------------------------


def _sphere_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Return the 3x3 rotation R = Ry(pitch) @ Rz(-yaw) (float64).

    Applied to a world-frame unit-sphere point P gives a camera-frame point Q:

    - ``Q[0]`` -- along the viewing axis (positive = front hemisphere)
    - ``Q[1]`` -- to the right in the image
    - ``Q[2]`` -- upward in the image

    Screen projection: ``screen_x = Q[1] * scale``,
    ``screen_y = -Q[2] * scale``.

    With yaw=0 and pitch=0 the camera faces the point at (lon=0, lat=0).
    Increasing yaw rotates the view to show eastern longitudes at center.
    Increasing pitch tilts the view to show higher latitudes at center.

    Parameters:
        yaw_deg: Longitude of the view center (deg).
        pitch_deg: Latitude of the view center (deg).

    Returns:
        3x3 float64 rotation matrix.
    """
    yr = math.radians(yaw_deg)
    pr = math.radians(pitch_deg)
    cy, sy = math.cos(-yr), math.sin(-yr)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cp, sp = math.cos(pr), math.sin(pr)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    return ry @ rz


def _cartesian_from_lonlat(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert geographic degrees to unit-sphere Cartesian coordinates.

    Parameters:
        lon_deg: Longitudes in degrees, any broadcastable shape against ``lat_deg``.
        lat_deg: Latitudes in degrees, same shape as ``lon_deg`` after broadcasting.

    Returns:
        ``np.ndarray`` of dtype float64, shape ``(..., 3)``, unit vectors
        ``(x, y, z)`` on the sphere.
    """
    lon_r = np.deg2rad(lon_deg)
    lat_r = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat_r)
    return np.stack(
        [cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Mollweide auxiliary
# ---------------------------------------------------------------------------


def _mollweide_theta_forward(lat_rad: np.ndarray) -> np.ndarray:
    """Solve 2*theta + sin(2*theta) = pi*sin(lat) by Newton-Raphson.

    Converges in fewer than ``_MOLLWEIDE_MAX_ITERATIONS`` iterations for any latitude.

    Parameters:
        lat_rad: Latitude array in radians, shape (...).

    Returns:
        Auxiliary angle theta in radians, same shape.
    """
    theta = lat_rad.copy()
    target = math.pi * np.sin(lat_rad)
    for _ in range(_MOLLWEIDE_MAX_ITERATIONS):
        sin2t = np.sin(2.0 * theta)
        denom = 2.0 + 2.0 * np.cos(2.0 * theta)
        denom = np.where(np.abs(denom) < _MOLLWEIDE_DENOM_EPSILON, _MOLLWEIDE_DENOM_EPSILON, denom)
        theta -= (2.0 * theta + sin2t - target) / denom
    return theta


# ---------------------------------------------------------------------------
# Scale fitting
# ---------------------------------------------------------------------------


def fit_scale(kind: ProjectionKind, vw: int, vh: int) -> float:
    """Return a scale (pixels per normalized unit) that fits the projection.

    The result leaves a margin ``_FIT_SCALE_MARGIN`` (10 % inset) around the
    projection so it does not touch the viewport edge at the initial zoom level.

    Parameters:
        kind: Projection kind.
        vw: Viewport width in pixels.
        vh: Viewport height in pixels.

    Returns:
        Scale in pixels per normalized unit.

    Raises:
        ValueError: If ``vw`` or ``vh`` is not strictly positive.
    """
    if vw <= 0 or vh <= 0:
        raise ValueError(f'fit_scale requires vw and vh > 0; got vw={vw}, vh={vh}')
    margin = _FIT_SCALE_MARGIN
    if kind in (ProjectionKind.POLAR_N, ProjectionKind.POLAR_S, ProjectionKind.SPHERE_3D):
        return min(vw, vh) * 0.5 * margin
    if kind == ProjectionKind.MOLLWEIDE:
        sqrt2 = math.sqrt(2.0)
        return min(vw / (4.0 * sqrt2), vh / (2.0 * sqrt2)) * margin
    # RECT: uses x_zoom/y_zoom rather than proj_scale; return a placeholder
    return float(min(vw, vh)) * 0.5 * margin


# ---------------------------------------------------------------------------
# Forward projection: (lon, lat) -> (viewport_x, viewport_y, visible)
# ---------------------------------------------------------------------------


def lonlat_to_display(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    params: ProjectionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map geographic coordinates to viewport pixel positions.

    Parameters:
        lon_deg: Longitude array (deg), any range.
        lat_deg: Latitude array (deg), values in [-90, 90].
        params: Current projection parameters.

    Returns:
        Tuple ``(vx, vy, visible)`` where ``vx`` and ``vy`` are float64
        arrays of viewport pixel coordinates and ``visible`` is a bool array
        (False for points not representable in this projection at ``params``).

    Raises:
        ValueError: If ``params.kind`` is unknown or ``params.scale`` is zero / non-finite.
    """
    if params.kind not in _KNOWN_PROJECTION_KINDS:
        raise ValueError(f'Unknown ProjectionKind: {params.kind!r}')
    sc = float(params.scale)
    if not math.isfinite(sc) or abs(sc) < _PROJECTION_SCALE_MIN:
        raise ValueError(
            f'lonlat_to_display requires non-zero finite params.scale; got {params.scale!r}'
        )
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    kind = params.kind

    if kind == ProjectionKind.RECT:
        xn = np.mod(lon, 360.0)
        yn = -lat
        vis = np.ones(xn.shape, dtype=bool)

    elif kind == ProjectionKind.POLAR_N:
        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)
        half_angle = np.clip(0.5 * (0.5 * math.pi - lat_r), 0.0, 0.5 * math.pi)
        r = np.tan(half_angle)
        xn = r * np.sin(lon_r)
        yn = -r * np.cos(lon_r)
        vis = np.ones(xn.shape, dtype=bool)

    elif kind == ProjectionKind.POLAR_S:
        lat_r = np.deg2rad(-lat)  # mirror: treat south pole as north
        lon_r = np.deg2rad(lon)
        half_angle = np.clip(0.5 * (0.5 * math.pi - lat_r), 0.0, 0.5 * math.pi)
        r = np.tan(half_angle)
        xn = r * np.sin(lon_r)
        yn = -r * np.cos(lon_r)
        vis = np.ones(xn.shape, dtype=bool)

    elif kind == ProjectionKind.MOLLWEIDE:
        # Center on lon=0; shift input to [-180, 180]
        lon_r = np.deg2rad(np.mod(lon + 180.0, 360.0) - 180.0)
        lat_r = np.deg2rad(lat)
        theta = _mollweide_theta_forward(lat_r)
        xn = (2.0 * math.sqrt(2.0) / math.pi) * lon_r * np.cos(theta)
        yn = math.sqrt(2.0) * np.sin(theta)
        vis = np.ones(xn.shape, dtype=bool)

    elif kind == ProjectionKind.SPHERE_3D:
        p_cart = _cartesian_from_lonlat(lon, lat)  # (..., 3)
        R = _sphere_rotation_matrix(params.yaw_deg, params.pitch_deg)
        orig_shape = p_cart.shape[:-1]
        p_flat = p_cart.reshape(-1, 3).T  # (3, N)
        q_flat = R @ p_flat  # (3, N)
        q = q_flat.T.reshape((*orig_shape, 3))
        xn = q[..., 1]
        yn = -q[..., 2]
        vis = q[..., 0] > 0.0

    else:
        raise ValueError(f'Unknown ProjectionKind: {kind!r}')

    vx = xn * sc + params.cx
    vy = yn * sc + params.cy
    return vx, vy, vis


# ---------------------------------------------------------------------------
# Inverse projection: (viewport_x, viewport_y) -> (lon, lat, valid)
# ---------------------------------------------------------------------------


def display_to_lonlat(
    vx: np.ndarray,
    vy: np.ndarray,
    params: ProjectionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map viewport pixel positions to geographic coordinates.

    Parameters:
        vx: Viewport X pixel array.
        vy: Viewport Y pixel array.
        params: Current projection parameters.

    Returns:
        Tuple ``(lon_deg, lat_deg, valid)`` where ``valid`` is a bool mask
        (False for pixels outside the valid domain of the projection).

    Raises:
        ValueError: If ``params.kind`` is unknown or ``params.scale`` is zero / non-finite.
    """
    if params.kind not in _KNOWN_PROJECTION_KINDS:
        raise ValueError(f'Unknown ProjectionKind: {params.kind!r}')
    sc = float(params.scale)
    if not math.isfinite(sc) or abs(sc) < _PROJECTION_SCALE_MIN:
        raise ValueError(
            f'display_to_lonlat requires non-zero finite params.scale; got {params.scale!r}'
        )
    xn = (np.asarray(vx, dtype=np.float64) - params.cx) / sc
    yn = (np.asarray(vy, dtype=np.float64) - params.cy) / sc
    kind = params.kind

    if kind == ProjectionKind.RECT:
        lon_deg = np.mod(xn, 360.0)
        lat_deg = -yn
        valid = (lat_deg >= -90.0) & (lat_deg <= 90.0)
        return lon_deg, lat_deg, valid

    if kind == ProjectionKind.POLAR_N:
        r = np.sqrt(xn**2 + yn**2)
        lat_deg = 90.0 - np.degrees(2.0 * np.arctan(r))
        lon_deg = np.mod(np.degrees(np.arctan2(xn, -yn)), 360.0)
        valid = np.ones(r.shape, dtype=bool)
        return lon_deg, lat_deg, valid

    if kind == ProjectionKind.POLAR_S:
        r = np.sqrt(xn**2 + yn**2)
        lat_mirrored = 90.0 - np.degrees(2.0 * np.arctan(r))
        lat_deg = -lat_mirrored
        lon_deg = np.mod(np.degrees(np.arctan2(xn, -yn)), 360.0)
        valid = np.ones(r.shape, dtype=bool)
        return lon_deg, lat_deg, valid

    if kind == ProjectionKind.MOLLWEIDE:
        sqrt2 = math.sqrt(2.0)
        # theta from y: yn = sqrt(2)*sin(theta)
        theta_raw = np.clip(yn / sqrt2, -1.0, 1.0)
        theta = np.arcsin(theta_raw)
        sin_lat = (2.0 * theta + np.sin(2.0 * theta)) / math.pi
        lat_deg = np.degrees(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))
        cos_theta = np.cos(theta)
        # lon from x: xn = (2*sqrt2/pi)*lon_r*cos(theta)
        denom = (2.0 * sqrt2 / math.pi) * cos_theta
        # Avoid division by zero at poles (cos_theta ~ 0)
        safe_denom = np.where(np.abs(cos_theta) > _MOLLWEIDE_COS_EPSILON, denom, 1.0)
        lon_r = np.where(np.abs(cos_theta) > _MOLLWEIDE_COS_EPSILON, xn / safe_denom, 0.0)
        lon_r = np.clip(lon_r, -math.pi, math.pi)
        lon_deg = np.mod(np.degrees(lon_r), 360.0)
        # Inside the Mollweide ellipse: (x/(2*sqrt2))^2 + (y/sqrt2)^2 <= 1
        valid = (xn / (2.0 * sqrt2)) ** 2 + (yn / sqrt2) ** 2 <= 1.0 + _MOLLWEIDE_ELLIPSE_TOLERANCE
        return lon_deg, lat_deg, valid

    if kind == ProjectionKind.SPHERE_3D:
        return sphere_pixel_to_lonlat(vx, vy, params)

    raise ValueError(f'Unknown ProjectionKind: {kind!r}')


def sphere_pixel_to_lonlat(
    vx: np.ndarray,
    vy: np.ndarray,
    params: ProjectionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast orthographic sphere: viewport pixels to (lon_deg, lat_deg, hit).

    Points outside the sphere disk or on the back hemisphere return
    ``hit=False``.

    Parameters:
        vx: Viewport X pixel array.
        vy: Viewport Y pixel array.
        params: Projection parameters; ``kind`` must be ``SPHERE_3D``.

    Returns:
        Tuple ``(lon_deg, lat_deg, hit)``.

    Raises:
        ValueError: If ``params.kind`` is not ``SPHERE_3D`` or ``params.scale`` is zero
            / non-finite.
    """
    if params.kind != ProjectionKind.SPHERE_3D:
        raise ValueError(
            f'sphere_pixel_to_lonlat requires params.kind SPHERE_3D, got {params.kind!r}'
        )
    sc = float(params.scale)
    if not math.isfinite(sc) or abs(sc) < _PROJECTION_SCALE_MIN:
        raise ValueError(
            f'sphere_pixel_to_lonlat requires non-zero finite params.scale; got {params.scale!r}'
        )
    xn = (np.asarray(vx, dtype=np.float64) - params.cx) / sc
    yn = (np.asarray(vy, dtype=np.float64) - params.cy) / sc
    r2 = xn**2 + yn**2
    hit = r2 <= 1.0

    # Camera-frame components: (q0=front, q1=screen-right, q2=screen-up)
    q0 = np.where(hit, np.sqrt(np.maximum(0.0, 1.0 - r2)), 0.0)
    q1 = xn
    q2 = -yn  # screen y down -> camera y up negated

    R = _sphere_rotation_matrix(params.yaw_deg, params.pitch_deg)
    orig_shape = r2.shape
    q_flat = np.stack([q0.ravel(), q1.ravel(), q2.ravel()], axis=0)  # (3, N)
    p_flat = R.T @ q_flat  # (3, N) world coords
    lon_deg = np.mod(np.degrees(np.arctan2(p_flat[1], p_flat[0])).reshape(orig_shape), 360.0)
    lat_deg = np.degrees(np.arcsin(np.clip(p_flat[2], -1.0, 1.0))).reshape(orig_shape)
    return lon_deg, lat_deg, hit
