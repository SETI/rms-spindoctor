"""Where each line of sight first meets a relief body's terrain.

The ellipsoid renderer shades a body with limb relief by the terrain each pixel
shows.  On rough terrain that can lie far from where the line of sight meets the
smooth body: near the limb, and on a ridge that hides the terrain behind it.
:func:`visible_surface_directions` searches each line of sight for the first point
it meets.

The search works in the body's parameter space, like the field it reads
(:mod:`spindoctor.sim.render.bodies.relief`).
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.sim.render.bodies.relief import ReliefField
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['visible_surface_directions']

# The line-of-sight search (visible_surface_directions) steps down a meridian by
# at least this fraction of a field grid cell, and then narrows the crossing step
# this many times by linear interpolation before a last one.  Against a brute-force
# search, one narrowing puts 99% of the points within 1e-4 rad (none narrowing,
# 8e-4 rad), against a field cell of 0.05 rad at 15 degrees.
_SIGHT_STEPS_PER_CELL = 4.0
_SIGHT_REFINEMENTS = 1
# A step of that minimum length can pass over a sliver of terrain whose top pokes
# into the line of sight between two samples outside it.  Wherever the slope bound
# leaves room for one, the step is halved until its pieces are this fraction of a
# field grid cell, so a sliver the search misses is narrower than that and rises at
# most slope_max / 512 of a cell into the line of sight.
_SIGHT_SLIVER_CELLS = 1.0 / 256.0
# Steps whose slivers are searched together, which bounds the search's memory.
_SIGHT_SLIVER_CHUNK = 1 << 12


def _sight_gap(
    field: ReliefField, lat: NDArrayFloatType, phi: NDArrayFloatType, rho: NDArrayFloatType
) -> NDArrayFloatType:
    """How far inside the relief surface a line of sight lies at a latitude.

    Parameters:
        field: The relief realization.
        lat: Latitudes along each line of sight's meridian.
        phi: The meridians' longitudes, the same shape as ``lat``.
        rho: The lines of sight's distances from the observer axis, the same shape.

    Returns:
        ``(1 + h(lat, phi)) cos(lat) - rho``, positive inside the surface.
    """
    return np.asarray((1.0 + field.sample(lat, phi)) * np.cos(lat) - rho, dtype=np.float64)


@dataclass(frozen=True)
class _Crossing:
    """The highest crossing found in each of a set of latitude intervals.

    Parameters:
        found: Whether the interval holds a crossing the search found.
        upper: The upper end of the crossing's bracket, outside the surface.
        lower: The lower end of the bracket, inside the surface.
        g_upper: The line of sight's gap (:func:`_sight_gap`) at ``upper``.
        g_lower: The gap at ``lower``.
    """

    found: NDArrayBoolType
    upper: NDArrayFloatType
    lower: NDArrayFloatType
    g_upper: NDArrayFloatType
    g_lower: NDArrayFloatType


def _latitude_slope_bound(
    row_slope: NDArrayFloatType,
    ends: tuple[NDArrayFloatType, NDArrayFloatType],
    phi: NDArrayFloatType,
) -> NDArrayFloatType:
    """A bound on the field's latitude slope along short stretches of meridians.

    Along a meridian the bilinear field is linear between grid rows, with a slope
    between those of the two grid columns beside it.  A stretch no longer than a
    grid cell touches at most two row gaps, so the largest of the four slopes there
    bounds the field's slope along it.

    Parameters:
        row_slope: ``|h(row + 1) - h(row)|`` over the row spacing, at every grid node.
        ends: The stretches' upper and lower latitudes, at most a grid cell apart.
        phi: The meridians' longitudes.

    Returns:
        The bound for each stretch, per radian of latitude.
    """
    n = row_slope.shape[0]
    scale = n / (2.0 * np.pi)
    col = np.floor(np.mod(phi, 2.0 * np.pi) * scale).astype(np.intp) % n
    col_next = (col + 1) % n
    rows = [np.floor((lat + np.pi) * scale).astype(np.intp) % n for lat in ends]
    return np.asarray(
        np.max([row_slope[row, c] for row in rows for c in (col, col_next)], axis=0),
        dtype=np.float64,
    )


def _first_crossing_between(
    field: ReliefField,
    rho: NDArrayFloatType,
    phi: NDArrayFloatType,
    ends: tuple[NDArrayFloatType, NDArrayFloatType],
    gaps: tuple[NDArrayFloatType, NDArrayFloatType],
    *,
    slope: NDArrayFloatType,
    min_width: float,
) -> _Crossing:
    """Find the highest crossing in latitude intervals whose ends lie outside the surface.

    Over an interval the gap ``g`` changes by at most ``s + (1 + h_max) |sin(lat)|``
    per radian, where ``s`` bounds the field's latitude slope along the interval and
    ``|sin(lat)|`` is the larger of its two ends'.  So it can reach 0 inside the
    interval only if its two end values plus that bound times the width add up to at
    least 0.  An interval that passes this test is halved at a
    sample.  A midpoint inside the surface brackets a crossing with the upper end;
    otherwise both halves are tested again, down to pieces narrower than
    ``min_width``, and the highest bracket found is kept.  The intervals are searched
    in chunks of :data:`_SIGHT_SLIVER_CHUNK`, which bounds the memory the search
    takes.

    Parameters:
        field: The relief realization.
        rho: Each line of sight's distance from the observer axis.
        phi: Each line of sight's longitude.
        ends: The intervals' upper and lower latitudes.
        gaps: The gap at the upper and lower ends, both negative.
        slope: For each interval, a bound on the field's latitude slope along it
            (:func:`_latitude_slope_bound`), which also holds for its pieces.
        min_width: The narrowest piece searched, in radians.

    Returns:
        For each interval, the bracket of the highest crossing found.
    """
    count = rho.size
    found = np.zeros(count, dtype=np.bool_)
    upper = np.zeros(count)
    lower = np.zeros(count)
    g_upper = np.zeros(count)
    g_lower = np.zeros(count)
    top = 1.0 + field.h_max
    for start in range(0, count, _SIGHT_SLIVER_CHUNK):
        which = np.arange(start, min(start + _SIGHT_SLIVER_CHUNK, count))
        a, b = ends[0][which], ends[1][which]
        g_a, g_b = gaps[0][which], gaps[1][which]
        while which.size:
            width = a - b
            bound = slope[which] + top * np.maximum(np.abs(np.sin(a)), np.abs(np.sin(b)))
            live = (width >= min_width) & (g_a + g_b + bound * width >= 0.0)
            # A piece below a crossing already found can hold only a lower one.
            live &= ~(found[which] & (a <= lower[which]))
            which, a, b, g_a, g_b = which[live], a[live], b[live], g_a[live], g_b[live]
            mid = 0.5 * (a + b)
            g_mid = _sight_gap(field, mid, phi[which], rho[which])
            hit = g_mid >= 0.0
            # Keep the highest bracket per interval: write the hits from low to high,
            # so the last write to each interval is its highest.
            order = np.nonzero(hit)[0]
            order = order[np.argsort(a[order], kind='stable')]
            order = order[~found[which[order]] | (a[order] > upper[which[order]])]
            target = which[order]
            found[target] = True
            upper[target] = a[order]
            lower[target] = mid[order]
            g_upper[target] = g_a[order]
            g_lower[target] = g_mid[order]
            rest = ~hit
            which = np.concatenate([which[rest], which[rest]])
            a, b = np.concatenate([a[rest], mid[rest]]), np.concatenate([mid[rest], b[rest]])
            g_a = np.concatenate([g_a[rest], g_mid[rest]])
            g_b = np.concatenate([g_mid[rest], g_b[rest]])
    return _Crossing(found, upper, lower, g_upper, g_lower)


def visible_surface_directions(
    x: NDArrayFloatType, y: NDArrayFloatType, field: ReliefField
) -> NDArrayFloatType:
    """Where each line of sight first meets the relief surface, as unit directions.

    Everything is in the body's parameter space (the smooth body is the unit sphere,
    the field's poles on the ``z`` axis), with the observer far out on the ``+z``
    axis.  The line of sight through ``(x, y)`` runs parallel to ``z`` and stays in
    the meridian plane at longitude ``phi = atan2(y, x)``.  At latitude ``lat`` it
    lies at radius ``rho / cos(lat)``, with ``rho = hypot(x, y)``, so it is inside the
    relief surface ``r = 1 + h`` where ``g(lat) = (1 + h(lat, phi)) cos(lat) - rho``
    is positive.  The first point it meets is the highest latitude where ``g``
    reaches 0.

    The search walks down the meridian from the highest latitude the terrain can
    reach, ``acos(rho / (1 + h_max))``.  Each step is as long as the field's slope
    bound allows without passing a crossing, and at least a quarter of a field grid
    cell, so the walk resolves the terrain at its own resolution whatever the body's
    size in pixels.  A step of that minimum length can pass over a sliver of terrain
    between two samples outside it, so wherever the slope bound leaves room for one
    the step is searched by halving (:func:`_first_crossing_between`).  The step
    that crosses is then narrowed by linear interpolation of ``g``.

    A line of sight that misses the surface, outside its outline, reads the latitude
    of the walk's sample that came closest to the surface, the one with the largest
    ``g``.  For a pixel just outside the outline that is the outline point beside
    it, to the walk's quarter-cell spacing.  A line of sight with ``rho`` at or
    beyond ``1 + h_max`` reads latitude 0, the limb direction.

    Parameters:
        x: Parameter-space coordinates of the lines of sight along the body's first
            axis (any shape).
        y: The coordinates along its second axis, the same shape as ``x``.
        field: The relief realization.

    Returns:
        ``(..., 3)`` unit directions of the points seen, ``x.shape + (3,)``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    x_flat = x_arr.ravel()
    y_flat = np.asarray(y, dtype=np.float64).ravel()
    rho = stable_hypot(x_flat, y_flat)
    phi = np.mod(np.arctan2(y_flat, x_flat), 2.0 * np.pi)
    top = 1.0 + field.h_max
    bottom = 1.0 + field.h_min
    # No point of the surface is inside the line of sight above lat_top, and every
    # line of sight with rho below 1 + h_min is inside it by lat_low.  A line of
    # sight farther out can meet only terrain no lower than -lat_top.
    lat_top = np.arccos(np.minimum(rho / top, 1.0))
    lat_low = np.where(rho < bottom, np.arccos(np.minimum(rho / bottom, 1.0)), -lat_top)
    min_step = 2.0 * np.pi / field.grid.shape[0] / _SIGHT_STEPS_PER_CELL

    def inside(lat: NDArrayFloatType, index: NDArrayIntType) -> NDArrayFloatType:
        return _sight_gap(field, lat, phi[index], rho[index])

    # hi is the last latitude outside the surface and lo the first inside it.
    hi = lat_top.copy()
    lo = lat_top.copy()
    g_hi = np.full(rho.size, -np.inf)
    g_lo = np.zeros(rho.size)
    found = np.zeros(rho.size, dtype=np.bool_)
    best_lat = np.zeros(rho.size)
    best_g = np.full(rho.size, -np.inf)
    lat = lat_top.copy()
    # Whether the step that led to each line of sight's current sample was longer
    # than the slope bound allows, and so could have passed over a sliver.
    forced = np.zeros(rho.size, dtype=np.bool_)
    sliver_width = 2.0 * np.pi / field.grid.shape[0] * _SIGHT_SLIVER_CELLS
    row_slope = np.abs(np.roll(field.grid, -1, axis=0) - field.grid) * (
        field.grid.shape[0] / (2.0 * np.pi)
    )
    active = np.arange(rho.size)
    while active.size:
        lat_a = lat[active]
        g = inside(lat_a, active)
        closer = g > best_g[active]
        best_g[active[closer]] = g[closer]
        best_lat[active[closer]] = lat_a[closer]
        crossed = g >= 0.0
        found[active[crossed]] = True
        lo[active[crossed]] = lat_a[crossed]
        g_lo[active[crossed]] = g[crossed]
        # A forced step that ends outside the surface may have passed over a sliver.
        suspect = np.nonzero(~crossed & forced[active])[0]
        if suspect.size:
            pix = active[suspect]
            ends = (hi[pix], lat_a[suspect])
            sliver = _first_crossing_between(
                field,
                rho[pix],
                phi[pix],
                ends,
                (g_hi[pix], g[suspect]),
                slope=_latitude_slope_bound(row_slope, ends, phi[pix]),
                min_width=sliver_width,
            )
            inner = pix[sliver.found]
            crossed[suspect[sliver.found]] = True
            found[inner] = True
            hi[inner] = sliver.upper[sliver.found]
            g_hi[inner] = sliver.g_upper[sliver.found]
            lo[inner] = sliver.lower[sliver.found]
            g_lo[inner] = sliver.g_lower[sliver.found]
        walking = ~crossed & (lat_a > lat_low[active])
        moving = active[walking]
        hi[moving] = lat_a[walking]
        g_hi[moving] = g[walking]
        # Going down the meridian g rises by at most slope_max + (1 + h_max) sin(lat)
        # per radian, so a step of -g over that bound cannot pass a crossing.
        bound = field.slope_max + top * np.maximum(np.sin(lat_a[walking]), 0.0)
        safe = -g[walking] / np.maximum(bound, 1e-300)
        lat[moving] = np.maximum(lat_a[walking] - np.maximum(safe, min_step), lat_low[moving])
        forced[moving] = lat_a[walking] - lat[moving] > safe
        active = moving

    hit = np.where(found, lo, best_lat)
    # A crossing found at lat_top itself needs no narrowing; elsewhere g goes from
    # negative at hi to non-negative at lo.
    narrow = np.nonzero(found & np.isfinite(g_hi))[0]
    for _ in range(_SIGHT_REFINEMENTS):
        if narrow.size == 0:
            break
        a, b = hi[narrow], lo[narrow]
        ga, gb = g_hi[narrow], g_lo[narrow]
        mid = a + (b - a) * (-ga / np.maximum(gb - ga, 1e-300))
        g_mid = inside(mid, narrow)
        crossed = g_mid >= 0.0
        lo[narrow[crossed]] = mid[crossed]
        g_lo[narrow[crossed]] = g_mid[crossed]
        hi[narrow[~crossed]] = mid[~crossed]
        g_hi[narrow[~crossed]] = g_mid[~crossed]
    if narrow.size:
        a, b = hi[narrow], lo[narrow]
        ga, gb = g_hi[narrow], g_lo[narrow]
        hit[narrow] = a + (b - a) * (-ga / np.maximum(gb - ga, 1e-300))

    cos_lat = np.cos(hit)
    directions = np.stack([cos_lat * np.cos(phi), cos_lat * np.sin(phi), np.sin(hit)], axis=-1)
    return directions.reshape(*shape, 3)
