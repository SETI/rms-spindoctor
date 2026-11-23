"""Whether a frame can see any of a planet's ring catalog, decided without a dense backplane.

The ring model asks this before it pays for the ring-radius backplane over the whole
extended frame. The question is geometric: does the part of the ring plane that lies
within the catalog's outermost radius appear anywhere in the extended frame?

That part of the ring plane is a disc centered on the planet, and the directions from the
observer to the points of a disc form one connected patch of sky. If that patch and the
frame overlap, then either the patch crosses the frame's boundary, so that some boundary
pixel looks onto the ring plane inside the outermost radius, or the patch lies wholly
inside the frame and so contains the planet center's direction. The test therefore reads
the ring radius at every pixel center along the boundary of the extended frame and
projects the planet center into the frame, and declares the disc out of view only when
neither shows it.

Nothing in this argument depends on the geometry's details. A disc seen nearly edge-on,
one far enough away to span a few pixels, one the observer sits above, one partly behind
the observer, and a frame most of whose rays miss the ring plane are all handled the same
way, and the boundary pixels are evaluated by the same backplane machinery the dense path
uses, so the camera's distortion is included. What it can miss is a band thinner than a
pixel where it crosses the boundary, which the dense path's pixel-center samples are
equally liable to miss. A planet center that cannot be projected counts as in the frame,
so the test errs toward running the dense path, which is always correct and only slower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from oops import Meshgrid
from oops.backplane import Backplane

from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'RING_CENTER_MARGIN_PX',
    'RingDiscView',
    'frame_edge_uv',
    'ring_disc_view',
]


RING_CENTER_MARGIN_PX: float = 1.0
"""How far outside the extended frame's outermost pixel centers the planet center may
project and still count as in the frame.

The boundary radii and the projected center come from two separate oops computations,
whose light-time and aberration handling can differ by a small fraction of a pixel. The
margin absorbs that difference; a larger one would only send a few more frames down the
dense path.
"""


@dataclass(frozen=True)
class RingDiscView:
    """What the extended frame shows of the ring-plane disc out to a given radius.

    Parameters:
        edge_radii_km: Ring-plane radius, in km, at each boundary pixel center whose ray
            meets the ring plane; boundary pixels whose rays miss the plane are left out.
        center_in_frame: True when the planet center projects within
            :data:`RING_CENTER_MARGIN_PX` of the extended frame's outermost pixel
            centers, or could not be projected at all.
        outer_radius_km: Radius of the disc being tested, in km.
    """

    edge_radii_km: NDArrayFloatType
    center_in_frame: bool
    outer_radius_km: float

    @property
    def edge_meets_plane(self) -> bool:
        """True when at least one boundary pixel's ray meets the ring plane."""
        return self.edge_radii_km.size > 0

    @property
    def disc_outside_frame(self) -> bool:
        """True when no part of the disc can appear in the extended frame.

        That is the case when the planet center lies outside the frame and every boundary
        pixel either misses the ring plane or meets it beyond ``outer_radius_km``.
        """
        if self.center_in_frame:
            return False
        return not bool(np.any(self.edge_radii_km <= self.outer_radius_km))


def frame_edge_uv(obs: ObsSnapshotInst) -> NDArrayFloatType:
    """Every pixel center on the boundary of the extended frame, in geometry coordinates.

    The boundary is the outermost row and column on each side of the extended frame. The
    coordinates are the geometry layer's pixel corner coordinates, in which a pixel's
    center sits half a pixel past its array index.

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
    v_inner = v_centers[1:-1]
    sides = (
        (u_centers, np.full_like(u_centers, v_centers[0])),
        (u_centers, np.full_like(u_centers, v_centers[-1])),
        (np.full_like(v_inner, u_centers[0]), v_inner),
        (np.full_like(v_inner, u_centers[-1]), v_inner),
    )
    return np.concatenate([np.stack(side, axis=-1) for side in sides], axis=0)


def ring_disc_view(
    obs: ObsSnapshotInst, ring_target: str, planet: str, outer_radius_km: float
) -> RingDiscView:
    """Evaluate the ring radius along the extended frame's boundary and locate the planet.

    Builds one backplane over the boundary pixel centers from :func:`frame_edge_uv` and
    reads ``ring_radius`` there, then projects the planet center into the frame with
    ``obs.uv_from_path``. A center that comes back masked or not finite counts as in the
    frame.

    Parameters:
        obs: Observation snapshot.
        ring_target: Backplane key of the ring plane, ``'<planet>:ring'``.
        planet: SPICE name of the planet at the ring plane's center.
        outer_radius_km: Radius of the disc being tested, in km.

    Returns:
        The boundary radii and the center test, from which
        :attr:`RingDiscView.disc_outside_frame` decides whether the dense path can be
        skipped.
    """
    edge_bp = Backplane(obs, meshgrid=Meshgrid(obs.fov, frame_edge_uv(obs)))
    radii = edge_bp.ring_radius(ring_target).mvals
    radii_vals = np.asarray(np.ma.getdata(radii), dtype=np.float64)
    meets = ~np.ma.getmaskarray(radii) & np.isfinite(radii_vals)
    edge_radii_km = radii_vals[meets].reshape(-1)

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
        edge_radii_km=edge_radii_km,
        center_in_frame=center_in_frame,
        outer_radius_km=outer_radius_km,
    )
