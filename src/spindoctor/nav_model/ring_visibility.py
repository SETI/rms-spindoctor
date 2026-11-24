"""Whether a frame can see any of a planet's ring catalog, decided without a dense backplane.

The ring model asks this before it pays for the ring-radius backplane over the whole
extended frame. The question is geometric: does the part of the ring plane that lies
within the catalog's outermost radius appear anywhere in the extended frame?

That part of the ring plane is a disc centered on the planet, and the directions from the
observer to the points of a disc form one connected patch of sky. Take the frame's
boundary to be the closed loop through the outermost pixel centers of the extended frame.
If the patch covers any pixel center inside that loop, then either the patch crosses the
loop, or it lies wholly inside the loop and so contains the planet center's direction.
The test therefore follows the loop's trace on the ring plane and projects the planet
center into the frame, and declares the disc out of view only when the trace stays
farther from the planet than the disc's radius and the center falls outside the frame.

The loop's trace is built from one backplane over the boundary pixel centers, which reads
the ring-plane radius and longitude where each pixel's ray meets the plane. A camera with
no distortion carries a straight run of pixels onto a straight line in the ring plane, and
over one pixel a real camera's distortion is far too small to matter. So the trace
between two neighboring boundary pixels whose rays both meet the plane is the straight
segment joining their intercepts. That segment is what catches a band thinner than a
pixel crossing the boundary between two pixel centers, which neither pixel sees on its
own. Where a side of the frame crosses the ring plane's horizon, the intercepts along
that side run out to infinity, and the trace past the last pixel that meets the plane is
the ray continuing the line through it and its neighbor. Two neighbors whose rays both
miss the plane have nothing between them on it.

Nothing in this argument depends on the geometry's details. A disc seen nearly edge-on,
one far enough away to span a few pixels, one the observer sits above, one partly behind
the observer, and a frame most of whose rays miss the ring plane are all handled the same
way, and the boundary pixels are evaluated by the same backplane machinery the dense path
uses, so the camera's distortion is included. The test does not skip when the trace
cannot be followed, because only one pixel on a side that crosses the horizon meets the
plane, or when the planet center cannot be projected. Any doubt therefore sends
the frame to the dense path, which is always correct and only slower.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from oops import Meshgrid
from oops.backplane import Backplane

from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'RING_CENTER_MARGIN_PX',
    'RingDiscView',
    'boundary_closest_approach_km',
    'frame_edge_sides',
    'frame_edge_uv',
    'ring_disc_view',
]


RING_CENTER_MARGIN_PX: float = 1.0
"""How far outside the extended frame's outermost pixel centers the planet center may
project and still count as in the frame.

The boundary intercepts and the projected center come from two separate oops
computations, whose light-time and aberration handling can differ by a small fraction of
a pixel. The margin absorbs that difference; a larger one would only send a few more
frames down the dense path.
"""


@dataclass(frozen=True)
class RingDiscView:
    """What the extended frame shows of the ring-plane disc out to a given radius.

    Parameters:
        edge_radii_km: Ring-plane radius, in km, at each boundary pixel center whose ray
            meets the ring plane; boundary pixels whose rays miss the plane are left out.
        edge_closest_km: How close, in km, the boundary's trace on the ring plane comes
            to the planet: infinite when no boundary ray meets the plane, and zero when
            the trace could not be followed.
        center_in_frame: True when the planet center projects within
            :data:`RING_CENTER_MARGIN_PX` of the extended frame's outermost pixel
            centers, or could not be projected at all.
        outer_radius_km: Radius of the disc being tested, in km.
    """

    edge_radii_km: NDArrayFloatType
    edge_closest_km: float
    center_in_frame: bool
    outer_radius_km: float

    @property
    def edge_meets_plane(self) -> bool:
        """True when at least one boundary pixel's ray meets the ring plane."""
        return self.edge_radii_km.size > 0

    @property
    def disc_outside_frame(self) -> bool:
        """True when no part of the disc can appear in the extended frame.

        That is the case when the planet center lies outside the frame and the
        boundary's trace on the ring plane stays farther than ``outer_radius_km`` from
        the planet.
        """
        if self.center_in_frame:
            return False
        return self.edge_closest_km > self.outer_radius_km


def frame_edge_uv(obs: ObsSnapshotInst) -> NDArrayFloatType:
    """Every pixel center on the boundary of the extended frame, in order around it.

    The boundary is the outermost row and column on each side of the extended frame. The
    samples run along the top row, down the right column, back along the bottom row, and
    up the left column, so each one's neighbors in the array are its neighbors on the
    frame and the last sample sits next to the first. The coordinates are the geometry
    layer's pixel corner coordinates, in which a pixel's center sits half a pixel past
    its array index.

    Parameters:
        obs: Observation snapshot supplying the extended-frame bounds.

    Returns:
        An array of shape ``(N, 2)`` holding ``(u, v)`` for each boundary pixel center,
        each corner pixel appearing once.
    """
    # The extended-frame bounds are array bounds; the meshgrid reads the geometry
    # layer's pixel corner coordinates, so the half pixel goes on here.
    u_centers = (
        np.arange(obs.extfov_u_min, obs.extfov_u_max + 1, dtype=np.float64)
        + PIXEL_CENTER_TO_CORNER_PX
    )
    v_centers = (
        np.arange(obs.extfov_v_min, obs.extfov_v_max + 1, dtype=np.float64)
        + PIXEL_CENTER_TO_CORNER_PX
    )
    u_first, u_last = u_centers[0], u_centers[-1]
    v_first, v_last = v_centers[0], v_centers[-1]
    sides = (
        (u_centers, np.full_like(u_centers, v_first)),
        (np.full_like(v_centers[1:], u_last), v_centers[1:]),
        (u_centers[-2::-1], np.full_like(u_centers[1:], v_last)),
        (np.full_like(v_centers[1:-1], u_first), v_centers[-2:0:-1]),
    )
    return np.concatenate([np.stack(side, axis=-1) for side in sides], axis=0)


def frame_edge_sides(n_u: int, n_v: int) -> list[NDArrayIntType]:
    """Indices into :func:`frame_edge_uv`'s samples of each side of the frame, in order.

    Each side runs from one corner to the next and includes both corners, so the four
    sides together close the loop.

    Parameters:
        n_u: Number of pixel columns in the extended frame.
        n_v: Number of pixel rows in the extended frame.

    Returns:
        Four integer arrays, the top, right, bottom, and left sides, each listing the
        indices of that side's samples in order along the loop.
    """
    n_samples = 2 * (n_u + n_v) - 4
    starts = (0, n_u - 1, n_u + n_v - 2, 2 * n_u + n_v - 3)
    lengths = (n_u, n_v, n_u, n_v)
    return [
        (start + np.arange(length, dtype=np.int64)) % n_samples
        for start, length in zip(starts, lengths, strict=True)
    ]


def _closest_on_segments(start: NDArrayFloatType, end: NDArrayFloatType) -> float:
    """Return the least distance from the origin to any of a set of segments.

    Parameters:
        start: Array of shape ``(M, 2)`` of segment start points.
        end: Array of shape ``(M, 2)`` of segment end points.

    Returns:
        The least distance, or infinity when there are no segments.
    """
    if start.shape[0] == 0:
        return math.inf
    step = end - start
    length2 = np.sum(step * step, axis=-1)
    along = -np.sum(start * step, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(length2 > 0.0, along / length2, 0.0)
    nearest = start + np.clip(frac, 0.0, 1.0)[:, None] * step
    return float(np.min(np.hypot(nearest[:, 0], nearest[:, 1])))


def _closest_on_ray(origin: NDArrayFloatType, through: NDArrayFloatType) -> float:
    """Return the distance from the origin to the ray from ``through`` away from ``origin``.

    The ray starts at ``through`` and runs along the line from ``origin`` through it.

    Parameters:
        origin: The point behind the ray's start, shape ``(2,)``.
        through: The ray's start, shape ``(2,)``.

    Returns:
        The least distance from the coordinate origin to the ray.
    """
    step = through - origin
    length2 = float(step @ step)
    frac = max(0.0, -float(through @ step) / length2) if length2 > 0.0 else 0.0
    nearest = through + frac * step
    return math.hypot(float(nearest[0]), float(nearest[1]))


def boundary_closest_approach_km(xy_km: NDArrayFloatType, sides: Sequence[NDArrayIntType]) -> float:
    """How close the boundary's trace on the ring plane comes to the ring center.

    A straight side of the frame crosses the ring plane's horizon at most once, so the
    samples on it that meet the plane form one unbroken run, and a masked sample between
    two that meet it lies on the segment joining them (oops masks a few samples whose
    intercept did not converge). Along each side the trace therefore joins each sample
    that meets the plane to the next one that does. Where the run stops short of the
    side's end, the side crosses the horizon there, and the trace continues as the ray
    from the run's end sample along the line through it and its neighbor in the run. A
    side on which only one sample meets the plane gives that line no direction, so the
    trace cannot be followed and the answer is zero.

    Parameters:
        xy_km: Array of shape ``(N, 2)`` of ring-plane intercepts, in km, in a frame
            centered on the ring center, with NaN where a sample misses the plane.
        sides: Index arrays into ``xy_km``, one per side, as from
            :func:`frame_edge_sides`.

    Returns:
        The least distance, in km, from the ring center to the trace; infinity when no
        sample meets the plane, and zero when the trace could not be followed.
    """
    closest = math.inf
    for side in sides:
        pts = xy_km[side]
        hit_idx = np.flatnonzero(np.all(np.isfinite(pts), axis=-1))
        if hit_idx.size == 0:
            continue
        starts_late = hit_idx[0] > 0
        ends_early = hit_idx[-1] < len(side) - 1
        if hit_idx.size == 1:
            if starts_late or ends_early:
                return 0.0
            continue
        run = pts[hit_idx]
        closest = min(closest, _closest_on_segments(run[:-1], run[1:]))
        if starts_late:
            closest = min(closest, _closest_on_ray(run[1], run[0]))
        if ends_early:
            closest = min(closest, _closest_on_ray(run[-2], run[-1]))
    return closest


def ring_disc_view(
    obs: ObsSnapshotInst, ring_target: str, planet: str, outer_radius_km: float
) -> RingDiscView:
    """Trace the extended frame's boundary on the ring plane and locate the planet.

    Builds one backplane over the boundary pixel centers from :func:`frame_edge_uv` and
    reads ``ring_radius`` and ``ring_longitude`` there, which oops computes from one
    ring-plane intercept, then follows the trace with
    :func:`boundary_closest_approach_km`. It projects the planet center into the frame
    with ``obs.uv_from_path``; a center that comes back masked or not finite counts as in
    the frame.

    Parameters:
        obs: Observation snapshot.
        ring_target: Backplane key of the ring plane, ``'<planet>:ring'``.
        planet: SPICE name of the planet at the ring plane's center.
        outer_radius_km: Radius of the disc being tested, in km.

    Returns:
        The boundary radii, the trace's closest approach, and the center test, from
        which :attr:`RingDiscView.disc_outside_frame` decides whether the dense path can
        be skipped.
    """
    edge_bp = Backplane(obs, meshgrid=Meshgrid(obs.fov, frame_edge_uv(obs)))
    radii = edge_bp.ring_radius(ring_target).mvals
    longitudes = edge_bp.ring_longitude(ring_target).mvals
    radii_vals = np.asarray(np.ma.getdata(radii), dtype=np.float64).reshape(-1)
    lon_vals = np.asarray(np.ma.getdata(longitudes), dtype=np.float64).reshape(-1)
    # Both backplanes come from one intercept event and so share its mask.
    mask = np.ma.getmaskarray(radii).reshape(-1)
    with np.errstate(invalid='ignore'):
        xy_km = np.stack((radii_vals * np.cos(lon_vals), radii_vals * np.sin(lon_vals)), axis=-1)
    meets = ~mask & np.all(np.isfinite(xy_km), axis=-1)
    xy_km[~meets] = np.nan
    sides = frame_edge_sides(
        obs.extfov_u_max - obs.extfov_u_min + 1, obs.extfov_v_max - obs.extfov_v_min + 1
    )

    center = obs.uv_from_path(planet)
    center_uv = np.asarray(center.vals, dtype=np.float64).reshape(-1)
    if bool(np.any(center.mask)) or not bool(np.all(np.isfinite(center_uv))):
        center_in_frame = True
    else:
        lo_u = obs.extfov_u_min + PIXEL_CENTER_TO_CORNER_PX - RING_CENTER_MARGIN_PX
        hi_u = obs.extfov_u_max + PIXEL_CENTER_TO_CORNER_PX + RING_CENTER_MARGIN_PX
        lo_v = obs.extfov_v_min + PIXEL_CENTER_TO_CORNER_PX - RING_CENTER_MARGIN_PX
        hi_v = obs.extfov_v_max + PIXEL_CENTER_TO_CORNER_PX + RING_CENTER_MARGIN_PX
        center_in_frame = bool(lo_u <= center_uv[0] <= hi_u) and bool(lo_v <= center_uv[1] <= hi_v)
    return RingDiscView(
        edge_radii_km=radii_vals[meets],
        edge_closest_km=boundary_closest_approach_km(xy_km, sides),
        center_in_frame=center_in_frame,
        outer_radius_km=outer_radius_km,
    )
