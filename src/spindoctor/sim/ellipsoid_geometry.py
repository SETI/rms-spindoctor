"""Shared ellipsoid shading geometry for the simulator's two sides.

The image-side forward renderer (``spindoctor.sim.forward.body``) and the
navigator-side predicted-body renderer (``spindoctor.nav_model.sim_body``)
must shade an ellipsoid with byte-identical conventions: the same surface
normals, the same light direction, and the same Lambert clamp.  With shared
conventions the planted scene error is the only error in a recovery
measurement; independent implementations would each carry their own
conventions and any delta between them would contaminate the measurement as
an unknown systematic.

These helpers take explicit geometry arguments only.  They never read a
scene parameter mapping, so they cannot carry truth-side information across
the information boundary (see ``spindoctor.sim.scene``).
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DARK_SIDE_ILLUM_STRENGTH',
    'EllipsoidProjection',
    'ellipsoid_image_normals',
    'illumination_vector',
    'lambert_from_normals',
    'project_ellipsoid',
]

# Lambertian floor for the visible-but-unlit hemisphere.  A pixel whose
# illumination is clipped to this value sits past the terminator (its true
# cos(incidence) is at or below the floor), so a strictly-greater comparison
# against it separates the lit region from the unlit disc -- the boundary the
# terminator polyline traces.  The mesh shading path mirrors this floor.
DARK_SIDE_ILLUM_STRENGTH: float = 0.01


@dataclass(frozen=True)
class EllipsoidProjection:
    """The projected-ellipsoid working grids both simulator sides shade from.

    Produced by :func:`project_ellipsoid`; every array lives on the working
    grid (the render grid supersampled by ``aa_scale``).

    Parameters:
        aa_scale: The anti-aliasing supersampling factor of the working grid.
        work_center_v: Body center v at working resolution.
        work_center_u: Body center u at working resolution.
        work_semi_major: Semi-major axis (a) at working resolution.
        work_semi_minor: Semi-minor axis (b) at working resolution.
        work_semi_c: Depth semi-axis (c) at working resolution.
        cos_rz: Cosine of the rotation_z angle.
        sin_rz: Sine of the rotation_z angle.
        v_coords: Centered v pixel coordinates at working resolution.
        u_coords: Centered u pixel coordinates at working resolution.
        v_rot: Rotated-frame v coordinate of each pixel (tilt applied).
        u_rot: Rotated-frame u coordinate of each pixel.
        z_coords: Depth of the visible ellipsoid surface at each pixel.
        ellipse_dist_sq: Squared normalized ellipse distance of each pixel.
        inside_mask: Pixels at or inside the projected ellipse boundary.
        ellipse_mask: Ellipse coverage mask (a soft anti-aliased rim when
            ``anti_aliasing`` > 0, else the hard boundary as floats).
    """

    aa_scale: int
    work_center_v: float
    work_center_u: float
    work_semi_major: float
    work_semi_minor: float
    work_semi_c: float
    cos_rz: float
    sin_rz: float
    v_coords: NDArrayFloatType
    u_coords: NDArrayFloatType
    v_rot: NDArrayFloatType
    u_rot: NDArrayFloatType
    z_coords: NDArrayFloatType
    ellipse_dist_sq: NDArrayFloatType
    inside_mask: NDArrayBoolType
    ellipse_mask: NDArrayFloatType


def project_ellipsoid(
    size: tuple[int, int],
    center: tuple[float, float],
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    anti_aliasing: float = 0.0,
) -> EllipsoidProjection:
    """Project an ellipsoid onto the image plane and build the shading grids.

    This is the single implementation of the projection both simulator sides
    shade from: the working-grid coordinate frames (in-plane rotation, tilt),
    the visible-hemisphere depth, and the (optionally anti-aliased) coverage
    mask.  The image-side renderer carves craters into these grids and the
    navigator-side predicted-body renderer shades them smooth, so the planted
    scene error is the only geometric difference between the two.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point
            pixels.  (0.0, 0.0) is the top-left corner of pixel (0,0),
            (0.5, 0.5) is the center of pixel (0,0).
        axis1: The full width of axis 1 (a) of the ellipsoid in pixels.
        axis2: The full width of axis 2 (b) of the ellipsoid in pixels.
        axis3: The full width of axis 3 (c) of the ellipsoid in pixels (depth).
        rotation_z: Rotation angle around the viewing axis in radians.
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
        anti_aliasing: Anti-aliasing amount in [0, 1]; > 0 supersamples the
            working grid (up to 4x) and softens the limb rim of
            ``ellipse_mask``.

    Returns:
        The :class:`EllipsoidProjection` working grids.
    """
    size_v, size_u = size

    # Convert full-width axes to half-widths for ellipsoid math
    semi_major_axis = axis1 / 2.0
    semi_minor_axis = axis2 / 2.0
    semi_c_axis = axis3 / 2.0

    # Determine anti-aliasing scale factor (only for limb smoothing)
    if anti_aliasing > 0:
        # Scale factor: 1 (no AA) to 4 (max AA)
        aa_scale = int(1 + 3 * anti_aliasing)
    else:
        aa_scale = 1

    # Work at higher resolution for anti-aliasing at limb
    work_v = size_v * aa_scale
    work_u = size_u * aa_scale
    work_center_v = center[0] * aa_scale
    work_center_u = center[1] * aa_scale
    work_semi_major = semi_major_axis * aa_scale
    work_semi_minor = semi_minor_axis * aa_scale
    work_semi_c = semi_c_axis * aa_scale

    # Create coordinate grids at pixel centers.  The grid starts as array rows
    # and columns and the projection works in the geometry layer's pixel corner
    # coordinates, where the caller states the center, so the half pixel goes
    # on here; that alignment holds at every supersampling scale because the
    # center is scaled with the grid.
    v_coords, u_coords = np.mgrid[0:work_v, 0:work_u].astype(float)
    v_coords += PIXEL_CENTER_TO_CORNER_PX
    u_coords += PIXEL_CENTER_TO_CORNER_PX
    v_coords -= work_center_v
    u_coords -= work_center_u

    # Apply rotation_z (in-plane rotation around z-axis, clockwise)
    cos_rz = np.cos(rotation_z)
    sin_rz = np.sin(rotation_z)
    v_rot1 = v_coords * cos_rz - u_coords * sin_rz
    u_rot1 = v_coords * sin_rz + u_coords * cos_rz

    # Apply rotation_tilt (rotation around u-axis, tilting toward/away from viewer)
    # This affects the apparent shape and the z-coordinate
    cos_rt = np.cos(rotation_tilt)

    # After tilt, the v coordinate is affected
    # v_rot = v_rot1 * cos_rt (compressed by tilt)
    # z coordinate appears: z = v_rot1 * sin_rt (tilted depth)
    v_rot = v_rot1 * cos_rt
    u_rot = u_rot1
    # z will be computed from ellipsoid equation

    # Compute distance from ellipse center in local coordinates (2D projection)
    # For the visible ellipse: (v_rot/a)^2 + (u_rot/b)^2 <= 1
    ellipse_dist_sq = (v_rot / work_semi_major) ** 2 + (u_rot / work_semi_minor) ** 2
    ellipse_dist = np.sqrt(ellipse_dist_sq)

    # Compute z coordinate for 3D ellipsoid
    # Ellipsoid equation: (v_rot/a)^2 + (u_rot/b)^2 + (z/c)^2 = 1
    # For visible hemisphere: z = c * sqrt(1 - (v_rot/a)^2 - (u_rot/b)^2)
    # Only compute for points inside the ellipse
    z_coords = np.zeros_like(v_rot)
    inside_mask = ellipse_dist_sq <= 1.0
    z_sq = np.maximum(0.0, 1.0 - ellipse_dist_sq[inside_mask])
    z_coords[inside_mask] = work_semi_c * np.sqrt(z_sq)

    # Create base ellipse mask (1.0 inside, 0.0 outside)
    # Anti-aliasing only applied at the limb (edge)
    if anti_aliasing > 0:
        # Smooth transition zone: about 1 pixel at work resolution, only at edge
        edge_width = 3.0
        ellipse_mask = np.clip(1.0 - np.maximum(0, ellipse_dist - 1.0) / edge_width, 0.0, 1.0)
    else:
        ellipse_mask = (ellipse_dist <= 1.0).astype(float)

    return EllipsoidProjection(
        aa_scale=aa_scale,
        work_center_v=work_center_v,
        work_center_u=work_center_u,
        work_semi_major=work_semi_major,
        work_semi_minor=work_semi_minor,
        work_semi_c=work_semi_c,
        cos_rz=float(cos_rz),
        sin_rz=float(sin_rz),
        v_coords=v_coords,
        u_coords=u_coords,
        v_rot=v_rot,
        u_rot=u_rot,
        z_coords=z_coords,
        ellipse_dist_sq=ellipse_dist_sq,
        inside_mask=inside_mask,
        ellipse_mask=ellipse_mask,
    )


def ellipsoid_image_normals(
    ellipse_mask: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    *,
    z_coords: NDArrayFloatType,
    work_semi_major: float,
    work_semi_minor: float,
    work_semi_c: float,
    cos_rz: float,
    sin_rz: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Unit surface normals of the base ellipsoid in image coordinates.

    For a 3D ellipsoid, the surface normal at body-frame point (v, u, z) is
    (v/a^2, u/b^2, z/c^2) normalized.  The in-plane components are then rotated
    back from the ellipsoid's rotated frame to image coordinates through the
    inverse of the rotation_z coordinate transformation; the z component is
    perpendicular to the image plane and unaffected.  Both the smooth and the
    cratered shading paths derive their base normals here, so the two paths
    share a single illumination convention.

    Parameters:
        ellipse_mask: Ellipse coverage mask; normals are computed where > 0.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        z_coords: Depth of the visible ellipsoid surface at each pixel.
        work_semi_major: Semi-major axis (a) at working resolution.
        work_semi_minor: Semi-minor axis (b) at working resolution.
        work_semi_c: Depth semi-axis (c) at working resolution.
        cos_rz: Cosine of the rotation_z angle.
        sin_rz: Sine of the rotation_z angle.

    Returns:
        Tuple of (normal_v, normal_u, normal_z) unit-normal component arrays in
        image coordinates.
    """
    normal_v_local = np.zeros_like(v_rot)
    normal_u_local = np.zeros_like(u_rot)
    normal_z_local = np.zeros_like(z_coords)

    # Only compute normals for points inside the ellipsoid
    inside_mask = ellipse_mask > 0
    normal_v_local[inside_mask] = v_rot[inside_mask] / (work_semi_major**2)
    normal_u_local[inside_mask] = u_rot[inside_mask] / (work_semi_minor**2)
    normal_z_local[inside_mask] = z_coords[inside_mask] / (work_semi_c**2)

    # Normalize the normal vectors
    normal_mag = np.sqrt(normal_v_local**2 + normal_u_local**2 + normal_z_local**2)
    normal_mag = np.maximum(normal_mag, 1e-10)  # Avoid division by zero
    normal_v_local /= normal_mag
    normal_u_local /= normal_mag
    normal_z_local /= normal_mag

    # Rotate normal back to image coordinates (only v and u components)
    # The z component stays in the depth direction
    # Use inverse rotation (negate sin) to match the coordinate transformation
    normal_v = normal_v_local * cos_rz + normal_u_local * sin_rz
    normal_u = -normal_v_local * sin_rz + normal_u_local * cos_rz
    return normal_v, normal_u, normal_z_local


def illumination_vector(
    *, illumination_angle: float, phase_angle: float
) -> tuple[float, float, float]:
    """The unit body-to-sun direction in image coordinates.

    Both simulator sides derive the light direction here, so their
    illumination conventions cannot diverge.

    The in-plane direction comes from ``illumination_angle`` (0 = from the
    top of the image, pi/2 = from the right; the v component is negated
    because v increases downward).  The out-of-plane component encodes the
    phase angle -- the observer-body-sun angle: ``z = cos(phase_angle)`` so
    phase 0 (full) lights the visible face head-on and phase pi (new) lights
    it from behind, while the in-plane magnitude is ``sin(phase_angle)``.

    Parameters:
        illumination_angle: In-plane light direction in radians; 0 is from
            the top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Tuple of (v, u, z) components of the unit illumination direction;
        z points toward the observer.
    """
    illum_v_2d = -np.cos(illumination_angle)  # Negative because v increases downward
    illum_u_2d = np.sin(illumination_angle)

    illum_z = np.cos(phase_angle)
    illum_scale_2d = np.sin(phase_angle)
    illum_v_3d = illum_v_2d * illum_scale_2d
    illum_u_3d = illum_u_2d * illum_scale_2d

    # Normalize the 3D illumination direction (already unit up to rounding;
    # the guard covers a degenerate zero vector only).
    illum_mag = np.sqrt(illum_v_3d**2 + illum_u_3d**2 + illum_z**2)
    if illum_mag > 1e-10:
        illum_v_3d /= illum_mag
        illum_u_3d /= illum_mag
        illum_z_norm = illum_z / illum_mag
    else:
        illum_z_norm = 1.0  # Directly toward observer
    return float(illum_v_3d), float(illum_u_3d), float(illum_z_norm)


def lambert_from_normals(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Lambertian illumination strength for image-frame unit surface normals.

    The normals must already be unit length (or zero outside the body).  Both
    the smooth and the cratered shading paths use this single implementation
    so their illumination conventions cannot diverge.

    Parameters:
        normal_v: V component of the unit surface normal in image coordinates.
        normal_u: U component of the unit surface normal in image coordinates.
        normal_z: Z (toward-observer) component of the unit surface normal.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Illumination strength array in [0, 1]; 0 on the far hemisphere.
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
