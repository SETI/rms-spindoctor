"""Image-plane geometry of a body's haze layer.

The bounding box the haze is evaluated in, the latitude of the haze column a pixel
sees, and the emission and incidence cosines a pixel takes from the body or a scaled
copy of it, for :mod:`spindoctor.sim.render.bodies.atmosphere`.
"""

import math
from typing import cast

import numpy as np

from spindoctor.sim.render.bodies.haze import latitude_sine
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidGeometry,
    surface_normals,
    surface_unit_coords,
)
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'band_bbox',
    'column_latitude_sine',
    'disc_cosines',
    'limb_incidence',
    'scaled_body_cosines',
]


def band_bbox(
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    reach_a: float,
    reach_b: float,
    cos_rz: float,
    sin_rz: float,
) -> tuple[slice, slice] | None:
    """Grid slices bounding the outer band ellipse of the haze.

    The band's outer boundary in centered pixel coordinates satisfies
    ``|v*cos_rz - u*sin_rz| <= reach_a`` and ``|v*sin_rz + u*cos_rz| <=
    reach_b`` (the rotated-frame extents of the ellipse, with the tilt's
    foreshortening already divided out of ``reach_a``), so its axis-aligned
    bounding box follows by rotating those extents back.  One pixel of slack
    absorbs the half pixel between the coordinates the extents are stated in
    and the cells the slices name; every pixel outside the returned slices lies
    strictly outside the band.

    Parameters:
        shape: The render-grid shape ``(V, U)``.
        center_v: Body center v in grid pixels.
        center_u: Body center u in grid pixels.
        reach_a: Rotated-frame half-extent along semi-axis a, in pixels.
        reach_b: Rotated-frame half-extent along semi-axis b, in pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        ``(v_slice, u_slice)`` into the grid, or None when the box misses
        the grid entirely (the haze contributes nothing).
    """
    size_v, size_u = shape
    reach_v = reach_a * abs(cos_rz) + reach_b * abs(sin_rz)
    reach_u = reach_a * abs(sin_rz) + reach_b * abs(cos_rz)
    v0 = max(math.floor(center_v - reach_v) - 1, 0)
    v1 = min(math.ceil(center_v + reach_v) + 1, size_v)
    u0 = max(math.floor(center_u - reach_u) - 1, 0)
    u1 = min(math.ceil(center_u + reach_u) + 1, size_u)
    if v0 >= v1 or u0 >= u1:
        return None
    return slice(v0, v1), slice(u0, u1)


def column_latitude_sine(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    geometry: EllipsoidGeometry,
) -> NDArrayFloatType:
    """The sine of the latitude of the haze column each pixel sees.

    On the disc it is the latitude of the surface point under the pixel.  Above the
    limb it is the latitude of the limb point at the pixel's azimuth on the outline,
    where the column the line of sight grazes stands.  For a body seen equator-on
    this is :func:`~spindoctor.sim.render.bodies.haze.latitude_sine`.

    Parameters:
        v_rot: Each pixel's coordinate along the rotated first axis.
        u_rot: Each pixel's coordinate along the rotated second axis.
        e2: Each pixel's squared normalized outline distance.
        geometry: The body's shape and orientation.

    Returns:
        The sine of latitude per pixel, in [-1, 1].
    """
    if geometry.sin_lat == 0.0:
        return latitude_sine(v_rot, e2, semi_a=geometry.semi_a)
    x, _y, _z = surface_unit_coords(v_rot, u_rot, e2, geometry=geometry)
    sin_latitude: NDArrayFloatType = np.clip(x, -1.0, 1.0)
    return sin_latitude


def disc_cosines(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    geometry: EllipsoidGeometry,
    cos_rz: float,
    sin_rz: float,
    illum_v: float,
    illum_u: float,
    illum_z: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Cosines of the emission and incidence angles at on-disc haze pixels.

    Both come from the shared projection's surface normal in image coordinates
    (:func:`~spindoctor.sim.shared.ellipsoid.surface_normals`): the emission cosine is
    its component toward the observer, and the incidence cosine is its dot product
    with the illumination direction, so the haze stays aligned with the disc shading
    and the two cosines agree at phase 0 on any ellipsoid.  On a sphere the emission
    cosine is ``sqrt(1 - e2)``, which equals the normal's component toward the
    observer, and it is computed that way.

    Parameters:
        v_rot: Rotated-frame first-axis coordinate of each pixel.
        u_rot: Rotated-frame second-axis coordinate of each pixel.
        e2: Squared normalized outline distance at each pixel.
        geometry: The body's shape and orientation.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        illum_v: V component of the unit illumination direction.
        illum_u: U component of the unit illumination direction.
        illum_z: Z component of the unit illumination direction.

    Returns:
        ``(cos(emission), cos(incidence))`` per pixel; the incidence cosine is
        negative on the night side.
    """
    normal_v, normal_u, normal_z = surface_normals(
        v_rot, u_rot, e2, geometry=geometry, cos_rz=cos_rz, sin_rz=sin_rz
    )
    incidence = cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + normal_z * illum_z)
    if geometry.semi_a == geometry.semi_b == geometry.semi_c:
        emission = cast(NDArrayFloatType, np.sqrt(np.maximum(1.0 - e2, 0.0)))
    else:
        emission = normal_z
    return emission, incidence


def limb_incidence(
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    geometry: EllipsoidGeometry,
    cos_rz: float,
    sin_rz: float,
    illum_v: float,
    illum_u: float,
) -> NDArrayFloatType:
    """The incidence cosine of the limb point at each pixel's azimuth on the outline.

    The limb point is where the line from the body center through the pixel crosses the
    outline.  Its surface normal lies in the image plane, perpendicular to the outline
    there (:func:`~spindoctor.sim.shared.ellipsoid.surface_normals`), so the incidence
    cosine is that normal's dot product with the in-plane part of the illumination
    direction.  A scaled copy of the body has the same incidence cosine where that line
    crosses the scaled outline, so a pixel's incidence varies smoothly as the scaled
    outline passes it.  On a sphere the normal points along the line itself, and it is
    computed that way.

    Parameters:
        v_ctr: Each pixel's v coordinate relative to the body center, before rotation.
        u_ctr: Each pixel's u coordinate relative to the body center, before rotation.
        v_rot: Each pixel's coordinate along the rotated first axis.
        u_rot: Each pixel's coordinate along the rotated second axis.
        e2: Each pixel's squared normalized outline distance.
        geometry: The body's shape and orientation.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        illum_v: V component of the unit illumination direction.
        illum_u: U component of the unit illumination direction.

    Returns:
        The incidence cosine per pixel, negative where the limb faces away from the
        light.
    """
    if geometry.semi_a == geometry.semi_b == geometry.semi_c:
        rho = np.maximum(stable_hypot(v_ctr, u_ctr), 1e-9)
        return cast(NDArrayFloatType, (v_ctr * illum_v + u_ctr * illum_u) / rho)
    # A pixel inside the outline moves out onto it along its azimuth, as one outside
    # moves in, so every pixel takes the normal of a limb point.
    normal_v, normal_u, _normal_z = surface_normals(
        v_rot, u_rot, np.maximum(e2, 1.0), geometry=geometry, cos_rz=cos_rz, sin_rz=sin_rz
    )
    return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u)


def scaled_body_cosines(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    scale: float | NDArrayFloatType,
    limb_incidence: NDArrayFloatType,
    geometry: EllipsoidGeometry,
    cos_rz: float,
    sin_rz: float,
    illum_v: float,
    illum_u: float,
    illum_z: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """The emission and incidence cosines on the body scaled by ``scale``.

    A pixel the scaled outline covers takes them from the scaled surface, whose normal
    at an image point is the body's own normal at that point divided by the scale
    (:func:`disc_cosines`).  A pixel outside it is a grazing line of sight (emission
    cosine 0) whose incidence is that of the limb at the pixel's azimuth, the column
    standing above the limb there (:func:`limb_incidence`).

    Parameters:
        v_rot: Rotated-frame first-axis coordinate of each pixel.
        u_rot: Rotated-frame second-axis coordinate of each pixel.
        e2: Squared normalized outline distance of each pixel on the body itself.
        scale: The size of the scaled body relative to the body, one value or one
            per pixel.
        limb_incidence: The incidence cosine of the limb at each pixel's azimuth.
        geometry: The body's shape and orientation.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        illum_v: V component of the unit illumination direction.
        illum_u: U component of the unit illumination direction.
        illum_z: Z component of the unit illumination direction.

    Returns:
        ``(cos(emission), cos(incidence))`` per pixel, as new arrays.
    """
    scale_px = np.asarray(scale, dtype=np.float64)
    e2_scaled = e2 / (scale_px * scale_px)
    on_disc = e2_scaled < 1.0
    if not on_disc.any():
        return np.zeros_like(e2_scaled), limb_incidence.copy()
    everywhere = bool(on_disc.all())
    if scale_px.ndim and not everywhere:
        scale_on = scale_px[on_disc]
    else:
        scale_on = scale_px
    disc_mu, disc_mu0 = disc_cosines(
        (v_rot if everywhere else v_rot[on_disc]) / scale_on,
        (u_rot if everywhere else u_rot[on_disc]) / scale_on,
        e2_scaled if everywhere else e2_scaled[on_disc],
        geometry=geometry,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
        illum_v=illum_v,
        illum_u=illum_u,
        illum_z=illum_z,
    )
    if everywhere:
        return disc_mu, disc_mu0
    mu = np.zeros_like(e2_scaled)
    mu[on_disc] = disc_mu
    mu0 = limb_incidence.copy()
    mu0[on_disc] = disc_mu0
    return mu, mu0
