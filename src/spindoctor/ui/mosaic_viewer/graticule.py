"""Lat/lon graticule polylines for non-rectangular body-mosaic projections.

:func:`graticule_polylines` samples parallels and meridians in geographic
coordinates, projects them through the current :class:`ProjectionParams`, and
returns screen-space polyline segments ready for ``QPainter.drawPolyline``.

Polylines are split wherever consecutive samples jump discontinuously (e.g.
at the limb of a sphere or a map seam), so each returned segment is a
continuous visible arc.
"""

from __future__ import annotations

import math

import numpy as np

from spindoctor.ui.mosaic_viewer.projections import (
    ProjectionKind,
    ProjectionParams,
    lonlat_to_display,
)

# Maximum allowed squared-pixel step between consecutive samples before the
# polyline is broken.  Prevents stitching across jumps at the sphere limb or
# the antimeridian.
_MAX_STEP_SQ: float = 100.0**2  # 100 px gap

# Meridian label radius in polar projections, as a fraction of ``ProjectionParams.scale``.
_POLAR_MERIDIAN_LABEL_RADIUS_FRAC: float = 0.55


def graticule_polylines(
    params: ProjectionParams,
    *,
    lat_step_deg: float,
    lon_step_deg: float,
    samples_per_line: int = 181,
    show_parallels: bool = True,
    show_meridians: bool = True,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Compute graticule screen-space polylines for the given projection.

    Each "line" in the returned lists is a list of (vx, vy) tuples that can
    be drawn as a connected polyline.  Lines are pre-split at discontinuities
    (sphere limb, map seam) so callers do not need to check for jumps.

    Parameters:
        params: Current projection parameters.  For RECT mode the caller
            should use the existing straight-line overlay path instead.
        lat_step_deg: Spacing of parallels (deg).  Pass 0 to disable.
        lon_step_deg: Spacing of meridians (deg).  Pass 0 to disable.
        samples_per_line: Number of sample points along each graticule line.
        show_parallels: Include parallel (constant-latitude) lines.
        show_meridians: Include meridian (constant-longitude) lines.

    Returns:
        Tuple ``(parallel_lines, meridian_lines)`` where each element is a
        list of polyline segments.  A segment is a list of ``(vx, vy)``
        float tuples.

    Raises:
        TypeError: If ``params`` is ``None``.
        ValueError: If ``lat_step_deg`` or ``lon_step_deg`` is negative, or if
            ``samples_per_line`` is less than 2.
    """
    if params is None:
        raise TypeError('graticule_polylines: params must not be None')
    if lat_step_deg < 0 or lon_step_deg < 0:
        raise ValueError('graticule_polylines: lat_step_deg and lon_step_deg must be non-negative')
    if samples_per_line < 2:
        raise ValueError('graticule_polylines: samples_per_line must be at least 2')
    parallel_segs: list[list[tuple[float, float]]] = []
    meridian_segs: list[list[tuple[float, float]]] = []

    if show_parallels and lat_step_deg > 0:
        i_lat_min = math.ceil(-90.0 / lat_step_deg)
        i_lat_max = math.floor((90.0 + 1e-9) / lat_step_deg)
        for i_lat in range(i_lat_min, i_lat_max + 1):
            lat = i_lat * lat_step_deg
            lons = np.linspace(0.0, 360.0, samples_per_line)
            lats = np.full_like(lons, lat)
            vx, vy, vis = lonlat_to_display(lons, lats, params)
            parallel_segs.extend(_split_polyline(vx, vy, vis))

    if show_meridians and lon_step_deg > 0:
        i_lon_max = math.floor((360.0 - 1e-6) / lon_step_deg)
        for i_lon in range(0, i_lon_max + 1):
            lon = i_lon * lon_step_deg
            lats = np.linspace(-90.0, 90.0, samples_per_line)
            lons = np.full_like(lats, lon)
            vx, vy, vis = lonlat_to_display(lons, lats, params)
            meridian_segs.extend(_split_polyline(vx, vy, vis))

    return parallel_segs, meridian_segs


def graticule_label_anchors(
    params: ProjectionParams,
    *,
    lat_step_deg: float,
    lon_step_deg: float,
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """Return label anchor positions for visible graticule lines.

    Finds the best visible point along each parallel and meridian at which to
    draw a tick label.  For parallels the anchor is the point closest to the
    screen center along the line; for meridians similarly.

    Parameters:
        params: Current projection parameters.
        lat_step_deg: Spacing of parallels (deg).
        lon_step_deg: Spacing of meridians (deg).

    Returns:
        Tuple ``(parallel_anchors, meridian_anchors)``.  Each anchor is
        ``(vx, vy, label_text)``.

    Raises:
        TypeError: If ``params`` is ``None``.
        ValueError: If ``lat_step_deg`` or ``lon_step_deg`` is negative.
    """
    if params is None:
        raise TypeError('graticule_label_anchors: params must not be None')
    if lat_step_deg < 0 or lon_step_deg < 0:
        raise ValueError(
            'graticule_label_anchors: lat_step_deg and lon_step_deg must be non-negative'
        )
    parallel_anchors: list[tuple[float, float, str]] = []
    meridian_anchors: list[tuple[float, float, str]] = []
    cx, cy = params.cx, params.cy
    is_polar = params.kind in (ProjectionKind.POLAR_N, ProjectionKind.POLAR_S)

    if lat_step_deg > 0:
        i_lat_min = math.ceil(-90.0 / lat_step_deg)
        i_lat_max = math.floor((90.0 + 1e-9) / lat_step_deg)
        for i_lat in range(i_lat_min, i_lat_max + 1):
            lat = i_lat * lat_step_deg
            lons = np.linspace(0.0, 360.0, 181)
            lats = np.full_like(lons, lat)
            vx, vy, vis = lonlat_to_display(lons, lats, params)
            if is_polar:
                # Parallels are concentric circles; label at the rightmost point
                # so all parallel labels appear consistently on the east side.
                anchor = _rightmost_visible(vx, vy, vis)
            else:
                anchor = _nearest_visible_to_centre(vx, vy, vis, cx, cy)
            if anchor is not None:
                parallel_anchors.append((anchor[0], anchor[1], f'{lat:.0f}°'))

    if lon_step_deg > 0:
        i_lon_max = math.floor((360.0 - 1e-6) / lon_step_deg)
        for i_lon in range(0, i_lon_max + 1):
            lon = i_lon * lon_step_deg
            lats = np.linspace(-90.0, 90.0, 181)
            lons = np.full_like(lats, lon)
            vx, vy, vis = lonlat_to_display(lons, lats, params)
            if is_polar:
                # Meridians converge at the pole (the projection center), so
                # nearest-to-centre puts every label on top of every other.
                # Instead place each label at ~55 % of the projection radius.
                anchor = _at_target_radius(
                    vx, vy, vis, cx, cy, _POLAR_MERIDIAN_LABEL_RADIUS_FRAC * params.scale
                )
            else:
                anchor = _nearest_visible_to_centre(vx, vy, vis, cx, cy)
            if anchor is not None:
                meridian_anchors.append((anchor[0], anchor[1], f'{lon:.0f}°'))

    return parallel_anchors, meridian_anchors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_polyline(
    vx: np.ndarray,
    vy: np.ndarray,
    vis: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """Split (vx, vy, vis) arrays into continuous visible segments.

    A new segment starts whenever:
    - ``vis[i]`` is False (invisible / off-disk sample), or
    - The distance between consecutive visible points exceeds
      ``sqrt(_MAX_STEP_SQ)`` pixels (discontinuous seam).

    Parameters:
        vx: Viewport X coordinates.
        vy: Viewport Y coordinates.
        vis: Visibility boolean mask (same length).

    Returns:
        List of polyline segments; each segment has >= 2 points.
    """
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for i in range(len(vx)):
        if not vis[i]:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue

        pt = (float(vx[i]), float(vy[i]))

        if current:
            dx = pt[0] - current[-1][0]
            dy = pt[1] - current[-1][1]
            if dx * dx + dy * dy > _MAX_STEP_SQ:
                if len(current) >= 2:
                    segments.append(current)
                current = []

        current.append(pt)

    if len(current) >= 2:
        segments.append(current)
    return segments


def _nearest_visible_to_centre(
    vx: np.ndarray,
    vy: np.ndarray,
    vis: np.ndarray,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    """Return the visible point closest to (cx, cy), or None if none visible."""
    if not np.any(vis):
        return None
    dx = vx[vis] - cx
    dy = vy[vis] - cy
    idx = int(np.argmin(dx**2 + dy**2))
    vx_vis = vx[vis]
    vy_vis = vy[vis]
    return float(vx_vis[idx]), float(vy_vis[idx])


def _at_target_radius(
    vx: np.ndarray,
    vy: np.ndarray,
    vis: np.ndarray,
    cx: float,
    cy: float,
    target_r: float,
) -> tuple[float, float] | None:
    """Return the visible point whose distance from (cx, cy) is closest to target_r."""
    if not np.any(vis):
        return None
    vx_vis = vx[vis]
    vy_vis = vy[vis]
    r = np.sqrt((vx_vis - cx) ** 2 + (vy_vis - cy) ** 2)
    idx = int(np.argmin(np.abs(r - target_r)))
    return float(vx_vis[idx]), float(vy_vis[idx])


def _rightmost_visible(
    vx: np.ndarray,
    vy: np.ndarray,
    vis: np.ndarray,
) -> tuple[float, float] | None:
    """Return the visible point with the largest viewport-x coordinate."""
    if not np.any(vis):
        return None
    vx_vis = vx[vis]
    vy_vis = vy[vis]
    idx = int(np.argmax(vx_vis))
    return float(vx_vis[idx]), float(vy_vis[idx])
