"""Where each line of sight first meets a relief body's terrain.

The ellipsoid renderer shades a body with limb relief by the terrain each pixel
shows.  On rough terrain that can lie far from where the line of sight meets the
smooth body: near the limb, and on a ridge that hides the terrain behind it.
:func:`visible_surface_directions` searches each line of sight for the first point
it meets, :func:`points_on_sight_lines` puts each point found on the line of sight
that sees it, and :func:`settle_on_surface` moves it along that line of sight onto the
surface.

The search works in the body's parameter space, like the field it reads
(:mod:`spindoctor.sim.render.bodies.relief`).
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.sim.render.bodies.relief import ReliefField, directions_about_view
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['points_on_sight_lines', 'settle_on_surface', 'visible_surface_directions']

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
# settle_on_surface moves a point along its line of sight onto the surface by steps
# that start at this fraction of a field grid cell and double, and then halves the
# step that crosses the surface this many times.  The steps stop at a whole cell, so
# the point ends within 1/4096 of a cell of the surface, the length over which
# relief.turned_from_sun reads the terrain there, and typically far closer.
_SETTLE_FIRST_STEP_CELLS = 1.0 / 4096.0
_SETTLE_HALVINGS = 12


class _SightLines:
    """Parallel lines of sight through a relief field.

    Each line of sight stays in one meridian plane about the direction toward the
    observer.  Latitude is measured from the plane perpendicular to that direction,
    toward the observer, so a line of sight at the distance ``rho`` from the axis
    through the body center lies at radius ``rho / cos(lat)``.

    Parameters:
        x: Parameter-space coordinates of the lines of sight along the turned first
            axis (see :func:`visible_surface_directions`), one dimensional.
        y: The coordinates along the second axis, the same shape.
        field: The relief realization.
        view: The unit direction toward the observer, its ``x`` and ``z`` components
            in the field's frame (its ``y`` component is 0).
    """

    def __init__(
        self,
        x: NDArrayFloatType,
        y: NDArrayFloatType,
        field: ReliefField,
        *,
        view: tuple[float, float],
    ) -> None:
        self.field = field
        self.rho = stable_hypot(x, y)
        self.phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        self._view = view
        # Seen along the field's own pole, the meridians are the field's own.  There
        # the slope along a stretch is bounded by the row differences beside it.
        self._along_pole = view[0] == 0.0
        self._row_slope = np.abs(np.roll(field.grid, -1, axis=0) - field.grid) * (
            field.grid.shape[0] / (2.0 * np.pi)
        )

    def directions(self, lat: NDArrayFloatType, index: NDArrayIntType) -> NDArrayFloatType:
        """The field-frame unit directions at latitudes along the lines of sight.

        Parameters:
            lat: One latitude for each line of sight.
            index: The lines of sight.

        Returns:
            ``(N, 3)`` unit directions.
        """
        return directions_about_view(lat, self.phi[index], self._view)

    def gap(self, lat: NDArrayFloatType, index: NDArrayIntType) -> NDArrayFloatType:
        """How far inside the relief surface lines of sight lie at latitudes.

        Parameters:
            lat: One latitude for each line of sight.
            index: The lines of sight.

        Returns:
            ``(1 + h) cos(lat) - rho``, positive inside the surface.
        """
        height = (
            self.field.sample(lat, self.phi[index])
            if self._along_pole
            else self.field.sample_points(self.directions(lat, index))
        )
        return np.asarray((1.0 + height) * np.cos(lat) - self.rho[index], dtype=np.float64)

    def slope_bound(
        self, ends: tuple[NDArrayFloatType, NDArrayFloatType], index: NDArrayIntType
    ) -> NDArrayFloatType:
        """A bound on the field's slope along short stretches of the lines of sight.

        Along one of the field's own meridians the bilinear field is linear between
        grid rows, with a slope between those of the two grid columns beside it.  A
        stretch no longer than a grid cell touches at most two row gaps, so the
        largest of the four slopes there bounds the field's slope along it.  Any
        other meridian crosses the columns too, and takes the bound on the field's
        surface slope anywhere.

        Parameters:
            ends: The stretches' upper and lower latitudes, at most a grid cell apart.
            index: The stretches' lines of sight.

        Returns:
            The bound for each stretch, per radian of latitude.
        """
        if not self._along_pole:
            return np.full(index.size, self.field.slope_max)
        n = self.field.grid.shape[0]
        scale = n / (2.0 * np.pi)
        col = np.floor(self.phi[index] * scale).astype(np.intp) % n
        col_next = (col + 1) % n
        rows = [np.floor((lat + np.pi) * scale).astype(np.intp) % n for lat in ends]
        return np.asarray(
            np.max([self._row_slope[row, c] for row in rows for c in (col, col_next)], axis=0),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class _Crossing:
    """The highest crossing found in each of a set of latitude intervals.

    Parameters:
        found: Whether the interval holds a crossing the search found.
        upper: The upper end of the crossing's bracket, outside the surface.
        lower: The lower end of the bracket, inside the surface.
        g_upper: The line of sight's gap (:meth:`_SightLines.gap`) at ``upper``.
        g_lower: The gap at ``lower``.
    """

    found: NDArrayBoolType
    upper: NDArrayFloatType
    lower: NDArrayFloatType
    g_upper: NDArrayFloatType
    g_lower: NDArrayFloatType


def _first_crossing_between(
    lines: _SightLines,
    index: NDArrayIntType,
    ends: tuple[NDArrayFloatType, NDArrayFloatType],
    gaps: tuple[NDArrayFloatType, NDArrayFloatType],
    *,
    slope: NDArrayFloatType,
    min_width: float,
) -> _Crossing:
    """Find the highest crossing in latitude intervals whose ends lie outside the surface.

    Over an interval the gap ``g`` changes by at most ``s + (1 + h_max) |sin(lat)|``
    per radian, where ``s`` bounds the field's slope along the interval and
    ``|sin(lat)|`` is the larger of its two ends'.  So it can reach 0 inside the
    interval only if its two end values plus that bound times the width add up to at
    least 0.  An interval that passes this test is halved at a
    sample.  A midpoint inside the surface brackets a crossing with the upper end;
    otherwise both halves are tested again, down to pieces narrower than
    ``min_width``, and the highest bracket found is kept.  The intervals are searched
    in chunks of :data:`_SIGHT_SLIVER_CHUNK`, which bounds the memory the search
    takes.

    Parameters:
        lines: The lines of sight.
        index: The line of sight each interval belongs to.
        ends: The intervals' upper and lower latitudes.
        gaps: The gap at the upper and lower ends, both negative.
        slope: For each interval, a bound on the field's slope along it
            (:meth:`_SightLines.slope_bound`), which also holds for its pieces.
        min_width: The narrowest piece searched, in radians.

    Returns:
        For each interval, the bracket of the highest crossing found.
    """
    count = index.size
    found = np.zeros(count, dtype=np.bool_)
    upper = np.zeros(count)
    lower = np.zeros(count)
    g_upper = np.zeros(count)
    g_lower = np.zeros(count)
    top = 1.0 + lines.field.h_max
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
            g_mid = lines.gap(mid, index[which])
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
    x: NDArrayFloatType,
    y: NDArrayFloatType,
    field: ReliefField,
    *,
    view: tuple[float, float] = (0.0, 1.0),
) -> NDArrayFloatType:
    """Where each line of sight first meets the relief surface, as unit directions.

    Everything is in the body's parameter space (the smooth body is the unit sphere,
    the field's poles on the ``z`` axis), with the observer far out along ``view``.
    The plane perpendicular to ``view`` has two axes: the ``x`` axis turned with the
    view, which is the first, and ``y``.  The line of sight through ``(x, y)`` on that
    plane runs parallel to ``view`` and stays in one meridian plane about it, at
    longitude ``phi = atan2(y, x)``.  At latitude ``lat``, measured from the plane
    toward the observer, it lies at radius ``rho / cos(lat)``, with
    ``rho = hypot(x, y)``, so it is inside the relief surface ``r = 1 + h`` where
    ``g(lat) = (1 + h) cos(lat) - rho`` is positive, with ``h`` the height in the
    direction ``lat`` and ``phi`` point to (:func:`directions_about_view`).  The first
    point it meets is the highest latitude where ``g`` reaches 0.

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
        x: Parameter-space coordinates of the lines of sight along the turned first
            axis (any shape).
        y: The coordinates along the second axis, the same shape as ``x``.
        field: The relief realization.
        view: The unit direction toward the observer, its ``x`` and ``z`` components
            in the field's frame (its ``y`` component is 0).

    Returns:
        ``(..., 3)`` unit directions of the points seen in the field's frame,
        ``x.shape + (3,)``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    cell = 2.0 * np.pi / field.grid.shape[0]
    min_step = cell / _SIGHT_STEPS_PER_CELL
    lines = _SightLines(
        x_arr.ravel(),
        np.asarray(y, dtype=np.float64).ravel(),
        field,
        view=view,
    )
    rho = lines.rho
    top = 1.0 + field.h_max
    bottom = 1.0 + field.h_min
    # No point of the surface is inside the line of sight above lat_top, and every
    # line of sight with rho below 1 + h_min is inside it by lat_low.  A line of
    # sight farther out can meet only terrain no lower than -lat_top.
    lat_top = np.arccos(np.minimum(rho / top, 1.0))
    lat_low = np.where(rho < bottom, np.arccos(np.minimum(rho / bottom, 1.0)), -lat_top)

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
    sliver_width = cell * _SIGHT_SLIVER_CELLS
    active = np.arange(rho.size)
    while active.size:
        lat_a = lat[active]
        g = lines.gap(lat_a, active)
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
                lines,
                pix,
                ends,
                (g_hi[pix], g[suspect]),
                slope=lines.slope_bound(ends, pix),
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
        g_mid = lines.gap(mid, narrow)
        crossed = g_mid >= 0.0
        lo[narrow[crossed]] = mid[crossed]
        g_lo[narrow[crossed]] = g_mid[crossed]
        hi[narrow[~crossed]] = mid[~crossed]
        g_hi[narrow[~crossed]] = g_mid[~crossed]
    if narrow.size:
        a, b = hi[narrow], lo[narrow]
        ga, gb = g_hi[narrow], g_lo[narrow]
        hit[narrow] = a + (b - a) * (-ga / np.maximum(gb - ga, 1e-300))

    return lines.directions(hit, np.arange(rho.size)).reshape(*shape, 3)


def points_on_sight_lines(
    x: NDArrayFloatType,
    y: NDArrayFloatType,
    surface: NDArrayFloatType,
    *,
    view: tuple[float, float] = (0.0, 1.0),
) -> NDArrayFloatType:
    """Move surface points across the view onto the lines of sight that see them.

    The surface point a line of sight meets, as :func:`visible_surface_directions`
    finds it, lies on that line of sight only to the search's precision.  Each point
    is moved perpendicular to the view onto its own line of sight, keeping its
    distance toward the observer, so a ray from the point seen starts on the line of
    sight that sees it.

    Parameters:
        x: Parameter-space coordinates of the lines of sight along the turned first
            axis, as for :func:`visible_surface_directions`, one dimensional.
        y: The coordinates along the second axis, the same shape.
        surface: ``(N, 3)`` parameter-space positions of the surface points the lines
            of sight see, in the field's frame.
        view: The unit direction toward the observer, its ``x`` and ``z`` components
            in the field's frame (its ``y`` component is 0).

    Returns:
        ``(N, 3)`` positions on the lines of sight, in the field's frame.
    """
    view_x, view_z = view
    toward = surface[:, 0] * view_x + surface[:, 2] * view_z
    x_arr = np.asarray(x, dtype=np.float64)
    return np.stack(
        [
            x_arr * view_z + toward * view_x,
            np.asarray(y, dtype=np.float64),
            toward * view_z - x_arr * view_x,
        ],
        axis=-1,
    )


def settle_on_surface(
    points: NDArrayFloatType,
    field: ReliefField,
    *,
    view: tuple[float, float] = (0.0, 1.0),
) -> NDArrayFloatType:
    """Move points on their lines of sight to where those lines cross the relief surface.

    A point the search of :func:`visible_surface_directions` finds can lie up to a
    field grid cell inside or outside the terrain, and there the terrain may be
    another facet than the one the line of sight crosses.  Each point is moved along
    its line of sight, toward the observer when it lies inside the terrain and away
    when it lies outside, by steps that double from :data:`_SETTLE_FIRST_STEP_CELLS`
    of a cell until one ends on the other side of the surface.  That last step is
    then halved :data:`_SETTLE_HALVINGS` times, and the point is placed at the end
    outside the terrain.  A point whose steps reach a whole cell without crossing
    stays where it is.

    Parameters:
        points: ``(N, 3)`` parameter-space positions on lines of sight, in the field's
            frame.
        field: The relief realization.
        view: The unit direction toward the observer, its ``x`` and ``z`` components
            in the field's frame (its ``y`` component is 0).

    Returns:
        ``(N, 3)`` positions on the same lines of sight, at the surface.
    """
    start = np.asarray(points, dtype=np.float64)
    toward = np.array([view[0], 0.0, view[1]])
    inside = field.depth_below(start) > 0.0
    # Inside the terrain the surface lies toward the observer; outside, away from it.
    sense = np.where(inside, 1.0, -1.0)
    cell = 2.0 * np.pi / field.grid.shape[0]
    near_end = np.zeros(start.shape[0])
    far_end = np.full(start.shape[0], np.nan)
    length = cell * _SETTLE_FIRST_STEP_CELLS
    pending = np.arange(start.shape[0])
    while pending.size and length <= cell:
        moved = start[pending] + (sense[pending] * length)[:, np.newaxis] * toward
        crossed = (field.depth_below(moved) > 0.0) != inside[pending]
        far_end[pending[crossed]] = length
        near_end[pending[~crossed]] = length
        pending = pending[~crossed]
        length *= 2.0
    found = np.nonzero(np.isfinite(far_end))[0]
    near, far = near_end[found], far_end[found]
    for _ in range(_SETTLE_HALVINGS):
        middle = 0.5 * (near + far)
        moved = start[found] + (sense[found] * middle)[:, np.newaxis] * toward
        crossed = (field.depth_below(moved) > 0.0) != inside[found]
        near = np.where(crossed, near, middle)
        far = np.where(crossed, middle, far)
    outside_end = np.where(inside[found], far, near)
    settled = start.copy()
    settled[found] = start[found] + (sense[found] * outside_end)[:, np.newaxis] * toward
    return settled
