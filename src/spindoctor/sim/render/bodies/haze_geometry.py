"""Image-plane geometry of a body's haze layer.

The bounding box the haze is evaluated in, the latitude of the haze column a pixel
sees, and the surface incidence under an on-disc pixel, for
:mod:`spindoctor.sim.render.bodies.atmosphere`.
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
from spindoctor.support.types import NDArrayFloatType

__all__ = ['band_bbox', 'column_latitude_sine', 'disc_incidence']


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


def disc_incidence(
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
) -> NDArrayFloatType:
    """Cosine of the surface incidence angle at on-disc haze pixels.

    The shared projection's surface normal in image coordinates
    (:func:`~spindoctor.sim.shared.ellipsoid.surface_normals`) dotted with the
    illumination direction, so the haze stays aligned with the disc shading.

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
        ``cos(incidence)`` per pixel (negative on the night side).
    """
    normal_v, normal_u, normal_z = surface_normals(
        v_rot, u_rot, e2, geometry=geometry, cos_rz=cos_rz, sin_rz=sin_rz
    )
    return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + normal_z * illum_z)
