"""The navigator's predicted-body renderer for simulated scenes.

Renders the smooth Lambert ellipsoid ``NavModelBodySimulated`` predicts
from a scene's idealized body geometry.  This is deliberately the
navigator's *best model*, not the image: surface texture (craters, relief)
is truth-side information rendered only by the image-side twin
(``spindoctor.sim.render.bodies.render``), and the difference between the two is
exactly the model error a scene plants.  Shading conventions are shared
with the image side through :mod:`spindoctor.sim.shared.ellipsoid`, so
that planted difference is the only difference.
"""

from typing import cast

import numpy as np

from spindoctor.sim.shared.ellipsoid import (
    ellipsoid_image_normals,
    lambert_from_normals,
    project_ellipsoid,
)
from spindoctor.support.types import NDArrayFloatType

__all__ = ['create_predicted_body_image']


def create_predicted_body_image(
    size: tuple[int, int],
    center: tuple[float, float],
    semi_a_px: float,
    *,
    semi_b_px: float,
    semi_c_px: float,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    anti_aliasing: float = 0.0,
) -> NDArrayFloatType:
    """Render the predicted body: a smooth ellipsoid with Lambertian shading.

    The body is modeled as a 3D ellipsoid projected onto 2D and illuminated
    using Lambertian shading (cos(incidence)) based on the illumination
    direction and phase angle.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point pixels.
            (0.0, 0.0) is the top-left corner of pixel (0,0), (0.5, 0.5) is the
            center of pixel (0,0).
        semi_a_px: Semi-axis a of the ellipsoid in pixels, half its width along axis 1.
        semi_b_px: Semi-axis b of the ellipsoid in pixels, half its width along axis 2.
        semi_c_px: Semi-axis c of the ellipsoid in pixels, half its depth along axis 3.
        rotation_z: Rotation angle around the viewing axis (z-axis) in radians (0 to 2pi).
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
            Controls how much the ellipsoid is tilted toward/away from the viewer.
        illumination_angle: Direction of illumination in the image plane in radians (0 to 2pi).
            0 radians is at the top of the image, pi/2 is to the right.
        phase_angle: Phase angle in radians (0 to pi).
            0 = head-on illumination (fully illuminated),
            pi/2 = side illumination (half illuminated),
            pi = back illumination (no visible illumination).
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.

    Returns:
        A 2D numpy array of shape (size_v, size_u) with float values from 0.0 to 1.0,
        where 0.0 is black and 1.0 is full white.
    """
    size_v, size_u = size

    # The projection grids (working-resolution coordinate frames, hemisphere
    # depth, coverage mask) come from the shared geometry module, so this
    # predicted-body renderer and the image-side renderer project through one
    # implementation.
    proj = project_ellipsoid(
        size,
        center,
        semi_a_px,
        semi_b_px=semi_b_px,
        semi_c_px=semi_c_px,
        rotation_z=rotation_z,
        rotation_tilt=rotation_tilt,
        anti_aliasing=anti_aliasing,
    )
    aa_scale = proj.aa_scale
    ellipse_mask = proj.ellipse_mask

    normal_v, normal_u, normal_z = ellipsoid_image_normals(
        ellipse_mask,
        proj.v_rot,
        proj.u_rot,
        z_coords=proj.z_coords,
        work_semi_major=proj.work_semi_major,
        work_semi_minor=proj.work_semi_minor,
        work_semi_c=proj.work_semi_c,
        cos_rz=proj.cos_rz,
        sin_rz=proj.sin_rz,
    )
    intensity = (
        lambert_from_normals(
            normal_v,
            normal_u,
            normal_z,
            illumination_angle=illumination_angle,
            phase_angle=phase_angle,
        )
        * ellipse_mask
    )

    # Downsample if anti-aliasing was used
    if aa_scale > 1:
        # Simple box filter downsampling
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))

    # Ensure values are in [0, 1] range
    return cast(NDArrayFloatType, np.clip(intensity, 0.0, 1.0))
