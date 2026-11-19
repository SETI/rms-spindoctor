"""Shared ellipsoid geometry and shading for the simulator's two sides.

The image-side ellipsoid renderer (``spindoctor.sim.render.bodies.ellipsoid``) and the
navigator-side predicted-body renderer (``spindoctor.nav_model.simulated.predict_body``)
place an ellipsoid on a pixel grid through one projection, :func:`project_ellipsoid`,
and take its surface normals from :func:`ellipsoid_image_normals`.  With shared
conventions the planted scene error is the only error in a recovery measurement;
independent implementations would each carry their own conventions and any delta
between them would contaminate the measurement as an unknown systematic.

The navigator's predicted body is the smooth Lambert ellipsoid that
:func:`render_lambert_ellipsoid` draws.  The image side composes its own image from
the same projection and normals, adding relief, craters, photometric laws, and
textures, so the two sides never share a finished limb.

These helpers take explicit geometry arguments only.  They never read a
scene parameter mapping, so they cannot carry truth-side information across
the information boundary (see ``spindoctor.sim.scene``).
"""

from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DARK_SIDE_ILLUM_STRENGTH',
    'EllipsoidProjection',
    'ellipsoid_image_normals',
    'lambert_from_normals',
    'project_ellipsoid',
    'render_lambert_ellipsoid',
]

# Lambertian floor for the visible-but-unlit hemisphere.  A pixel whose
# illumination is clipped to this value sits past the terminator (its true
# cos(incidence) is at or below the floor), so a strictly-greater comparison
# against it separates the lit region from the unlit disc -- the boundary the
# terminator polyline traces.  The mesh shading path mirrors this floor.
DARK_SIDE_ILLUM_STRENGTH: float = 0.01


@dataclass(frozen=True)
class EllipsoidProjection:
    """An ellipsoid projected onto one pixel grid.

    Produced by :func:`project_ellipsoid`.  Every array covers the whole grid, or the
    box that :meth:`crop` selects from it, and every length is in the grid's pixels.
    The grid is the one the caller stated the geometry on, supersampled by
    ``aa_scale``.

    Parameters:
        aa_scale: Pixels of this grid per pixel of the grid the caller stated the
            geometry on, along each axis.
        center: Body center (v, u) in the whole grid's pixel corner coordinates.  A
            crop keeps it unchanged, so ``v_coords + center[0]`` and
            ``u_coords + center[1]`` are always a pixel's position on the whole grid.
        semi_a: Semi-axis a in this grid's pixels.
        semi_b: Semi-axis b in this grid's pixels.
        semi_c: Depth semi-axis c in this grid's pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        v_coords: The v coordinate of each pixel center relative to the body center,
            before rotation.  It is a read-only view of one value per row.
        u_coords: The u coordinate of each pixel center relative to the body center,
            before rotation.  It is a read-only view of one value per column.
        v_rot: The v coordinate of each pixel in the body's rotated frame, multiplied
            by the cosine of the tilt.
        u_rot: The u coordinate of each pixel in the body's rotated frame.
        ellipse_dist_sq: The squared normalized ellipse distance of each pixel,
            ``(v_rot / semi_a)**2 + (u_rot / semi_b)**2``, which is 1 on the limb.
    """

    aa_scale: int
    center: tuple[float, float]
    semi_a: float
    semi_b: float
    semi_c: float
    cos_rz: float
    sin_rz: float
    v_coords: NDArrayFloatType
    u_coords: NDArrayFloatType
    v_rot: NDArrayFloatType
    u_rot: NDArrayFloatType
    ellipse_dist_sq: NDArrayFloatType

    def crop(self, rows: slice, cols: slice) -> 'EllipsoidProjection':
        """The part of the projection inside a box on its grid.

        The coordinates keep their origin at the body center, so each pixel keeps
        the values it has in the whole projection.  The center stays in the whole
        grid's coordinates, so adding it to the cropped coordinates still gives each
        pixel's position on the whole grid.  The image side's crater carving needs
        both: it places craters by those positions, and its fallback crater seed is
        keyed on the center, which a crop must not change.

        Parameters:
            rows: The box's rows.
            cols: The box's columns.

        Returns:
            The cropped projection, whose arrays are views into this one's.
        """
        return replace(
            self,
            v_coords=self.v_coords[rows, cols],
            u_coords=self.u_coords[rows, cols],
            v_rot=self.v_rot[rows, cols],
            u_rot=self.u_rot[rows, cols],
            ellipse_dist_sq=self.ellipse_dist_sq[rows, cols],
        )


def project_ellipsoid(
    size: tuple[int, int],
    center: tuple[float, float],
    semi_a_px: float,
    *,
    semi_b_px: float,
    semi_c_px: float,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    anti_aliasing: float = 0.0,
) -> EllipsoidProjection:
    """Project an ellipsoid onto a pixel grid.

    The navigator's predicted body is projected onto its supersampled frame here, and
    the image side's ellipsoid renderer projects each body onto its working grid and
    its detector grid.  The body is rotated in the image plane by ``rotation_z``.
    Then its rotated v coordinate is multiplied by the cosine of ``rotation_tilt``,
    which stretches the outline along that axis by ``1 / cos(rotation_tilt)``.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point
            pixels.  (0.0, 0.0) is the top-left corner of pixel (0,0),
            (0.5, 0.5) is the center of pixel (0,0).
        semi_a_px: Semi-axis a of the ellipsoid in pixels, half its width along axis 1.
        semi_b_px: Semi-axis b of the ellipsoid in pixels, half its width along axis 2.
        semi_c_px: Semi-axis c of the ellipsoid in pixels, half its depth along axis 3.
        rotation_z: Rotation angle around the viewing axis in radians.
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
        anti_aliasing: Anti-aliasing amount in [0, 1].  Above 0 the grid is
            supersampled by ``int(1 + 3 * anti_aliasing)`` along each axis, up to 4;
            at 0 it is the ``size`` grid itself.

    Returns:
        The :class:`EllipsoidProjection` on the supersampled grid.
    """
    size_v, size_u = size
    aa_scale = int(1 + 3 * anti_aliasing) if anti_aliasing > 0 else 1

    center_v = center[0] * aa_scale
    center_u = center[1] * aa_scale
    semi_a = semi_a_px * aa_scale
    semi_b = semi_b_px * aa_scale

    # Pixel-center coordinates relative to the body center.  The grid starts as
    # array rows and columns and the projection works in the geometry layer's pixel
    # corner coordinates, where the caller states the center, so the half pixel goes
    # on here; that alignment holds at every supersampling scale because the center
    # is scaled with the grid.  Each coordinate varies along one axis only, so it is
    # computed once per row or column and broadcast over the grid.
    shape = (size_v * aa_scale, size_u * aa_scale)
    v_axis = (np.arange(shape[0], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX) - center_v
    u_axis = (np.arange(shape[1], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX) - center_u

    # Apply rotation_z (in-plane rotation around z-axis, clockwise)
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    v_rot1 = (v_axis * cos_rz)[:, np.newaxis] - (u_axis * sin_rz)[np.newaxis, :]
    u_rot = (v_axis * sin_rz)[:, np.newaxis] + (u_axis * cos_rz)[np.newaxis, :]

    # The tilt multiplies the rotated v coordinate by cos(tilt), so the outline
    # reaches 1 / cos(tilt) times farther along that axis.  A body tilted toward or
    # away from the viewer should look shorter there, not longer; the stretch is a
    # known defect (#893).
    v_rot = v_rot1 * np.cos(rotation_tilt)

    # The projected outline is the ellipse (v_rot/a)^2 + (u_rot/b)^2 = 1.
    ellipse_dist_sq = (v_rot / semi_a) ** 2 + (u_rot / semi_b) ** 2

    return EllipsoidProjection(
        aa_scale=aa_scale,
        center=(center_v, center_u),
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_c_px * aa_scale,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
        v_coords=np.broadcast_to(v_axis[:, np.newaxis], shape),
        u_coords=np.broadcast_to(u_axis[np.newaxis, :], shape),
        v_rot=v_rot,
        u_rot=u_rot,
        ellipse_dist_sq=ellipse_dist_sq,
    )


def ellipsoid_image_normals(
    projection: EllipsoidProjection,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Unit surface normals of the ellipsoid in image coordinates, at every pixel.

    For a 3D ellipsoid, the surface normal at body-frame point (v, u, z) is
    (v/a^2, u/b^2, z/c^2) normalized, where z is the depth of the visible hemisphere
    under the pixel.  The in-plane components are then rotated back from the
    ellipsoid's rotated frame to image coordinates through the inverse of the
    in-plane rotation; the z component is perpendicular to the image plane and
    unaffected.

    Outside the limb the depth is 0, so a pixel there carries the normal of the limb
    point on the line from the body center through it.  That normal lies in the
    image plane, with ``normal_z`` 0, so Lambert shading leaves the pixel dark.  A
    renderer that masks the body by its own outline can still read the normals
    there, and its shading continues smoothly across the limb.

    Parameters:
        projection: The ellipsoid's projection onto the grid.

    Returns:
        Tuple of (normal_v, normal_u, normal_z) unit-normal component arrays in
        image coordinates, on the projection's grid.
    """
    z_coords = projection.semi_c * np.sqrt(np.maximum(0.0, 1.0 - projection.ellipse_dist_sq))
    normal_v_local = projection.v_rot / (projection.semi_a**2)
    normal_u_local = projection.u_rot / (projection.semi_b**2)
    normal_z_local = z_coords / (projection.semi_c**2)

    # Normalize the normal vectors
    normal_mag = np.sqrt(normal_v_local**2 + normal_u_local**2 + normal_z_local**2)
    normal_mag = np.maximum(normal_mag, 1e-10)  # Avoid division by zero
    normal_v_local /= normal_mag
    normal_u_local /= normal_mag
    normal_z_local /= normal_mag

    # Rotate normal back to image coordinates (only v and u components)
    # The z component stays in the depth direction
    # Use inverse rotation (negate sin) to match the coordinate transformation
    cos_rz = projection.cos_rz
    sin_rz = projection.sin_rz
    normal_v = normal_v_local * cos_rz + normal_u_local * sin_rz
    normal_u = -normal_v_local * sin_rz + normal_u_local * cos_rz
    return normal_v, normal_u, normal_z_local


def lambert_from_normals(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Lambertian illumination strength for image-frame unit surface normals.

    The normals must already be unit length (or zero outside the body).

    Parameters:
        normal_v: V component of the unit surface normal in image coordinates.
        normal_u: U component of the unit surface normal in image coordinates.
        normal_z: Z (toward-observer) component of the unit surface normal.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Illumination strength array in [0, 1]; 0 wherever ``normal_z`` is 0 or less.
    """
    illum_v_3d, illum_u_3d, illum_z_norm = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )

    # Only illuminate points on the visible hemisphere (facing toward observer)
    # The z-component of the normal should be positive (pointing toward observer)
    visible_hemisphere = normal_z > 0

    # Compute cosine of incidence angle (Lambertian shading)
    # cos(incidence) = dot(normal, illumination_direction)
    cos_incidence = normal_v * illum_v_3d + normal_u * illum_u_3d + normal_z * illum_z_norm

    # Lambertian shading: I = I_0 * max(0, cos(incidence))
    # Only apply to visible hemisphere and clip to [0, 1] range
    light_side_illum_gamma = 1  # TODO make config parameter
    illum_strength = np.where(
        visible_hemisphere, np.clip(cos_incidence, DARK_SIDE_ILLUM_STRENGTH, 1.0), 0.0
    )
    illum_strength **= light_side_illum_gamma

    return cast(NDArrayFloatType, illum_strength)


def render_lambert_ellipsoid(
    projection: EllipsoidProjection,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Render a smooth ellipsoid with Lambert shading.

    This is the navigator's predicted body.  Each pixel of the projection's grid is
    shaded by :func:`lambert_from_normals` with the normals from
    :func:`ellipsoid_image_normals`.  The result is then averaged over blocks of
    ``aa_scale`` by ``aa_scale`` pixels.  Before averaging, a pixel whose center lies
    outside the limb is 0, and one on the visible hemisphere past the terminator is
    :data:`DARK_SIDE_ILLUM_STRENGTH`.

    Parameters:
        projection: The ellipsoid's projection onto its grid, whose dimensions must
            be multiples of ``aa_scale``, as :func:`project_ellipsoid` builds them.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        The intensity in [0, 1], on the grid the caller stated the geometry on: the
        projection's grid divided by ``aa_scale`` along each axis.
    """
    # Outside the limb every normal lies in the image plane, so the Lambert term is
    # already 0 there and the image needs no outline mask.
    intensity = lambert_from_normals(
        *ellipsoid_image_normals(projection),
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
    )
    scale = projection.aa_scale
    if scale > 1:
        rows, cols = intensity.shape
        intensity = intensity.reshape(rows // scale, scale, cols // scale, scale).mean(axis=(1, 3))
    return np.clip(intensity, 0.0, 1.0)
