"""The ellipsoid renderer's relief shading: relief normals and cast shadows on a grid.

The ellipsoid renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`) shades a body
with limb relief from the relief surface's own normals, read where each pixel's line
of sight first meets the surface, and darkens the pixels the terrain shadows.  This
module computes both, on one grid (:func:`relief_shading`, :func:`relief_normals`) or
on the detector grid with the tilt interpolated onto the working grid
(:func:`add_detector_relief`), and decides which of the two a body gets
(:func:`relief_resolved_on_detector_grid`).

The relief surface's synthesis, its sampling, and the cast-shadow march live in
:mod:`spindoctor.sim.render.bodies.relief`, and the search along a line of sight in
:mod:`spindoctor.sim.render.bodies.relief_sight`.
"""

import math
from dataclasses import dataclass

import numpy as np

from spindoctor.sim.render.bodies.ellipsoid_grid import (
    cubic_upsample,
    day_night_edge_cells,
    expand_to_working_grid,
    limb_band,
    neighboring_cells,
    smooth_surface_directions,
    stretched_sight_cells,
)
from spindoctor.sim.render.bodies.photometry import incidence_cosines
from spindoctor.sim.render.bodies.relief import ReliefField, cast_shadows
from spindoctor.sim.render.bodies.relief_sight import visible_surface_directions
from spindoctor.sim.shared.ellipsoid import EllipsoidProjection, ellipsoid_image_normals
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.stable_math import stable_matmul3
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ReliefShading',
    'add_detector_relief',
    'relief_normals',
    'relief_resolved_on_detector_grid',
    'relief_shading',
]

# The cast-shadow band: points with |tan(sun elevation)| below the field's
# slope bound (over 1 + h_min) are marched.  The bound is the grid's own
# finite-difference maximum, so this factor covers the bilinear field
# between grid nodes.
_SLOPE_BAND_MARGIN = 1.5
# The march samples the terrain at its own resolution: this many samples
# per relief-field grid cell, whatever the body's size in pixels.  A step
# tied to the shading pixel would skip ridge tips on a small body with fine
# relief and leave their shadows lit.
_MARCH_SAMPLES_PER_CELL = 8.0
# Half-width, in detector pixels beyond the relief outline's reach, of the band at
# the limb where a supersampled body's relief normals and shadows are computed on
# the working grid instead of interpolated from the detector grid.
_LIMB_BAND_PX = 3.0
# The shortest relief correlation length, in detector pixels, whose tilt is
# interpolated from the detector grid.  Finer relief is shaded wholly on the
# working grid, which the cubic spline cannot match there and which costs less,
# since the exactly shaded bands cover most of such a disc anyway (#1068).
_MIN_INTERPOLATED_CORR_PX = 5.0


@dataclass(frozen=True)
class ReliefShading:
    """The relief surface's shading at a set of pixels of one grid.

    Parameters:
        near: The grid mask of the pixels shaded.
        normal_v: The relief normal's v component at those pixels, in row-major order.
        normal_u: The normal's u component at the same pixels.
        normal_z: The normal's toward-observer component, at least 0.
        seen: ``(N, 3)`` parameter-space directions of the surface points the pixels
            see.
        shadowed: The grid mask of the pixels in a cast shadow.
    """

    near: NDArrayBoolType
    normal_v: NDArrayFloatType
    normal_u: NDArrayFloatType
    normal_z: NDArrayFloatType
    seen: NDArrayFloatType
    shadowed: NDArrayBoolType


def relief_resolved_on_detector_grid(
    field: ReliefField, detector_grid: EllipsoidProjection
) -> bool:
    """Whether the detector grid resolves the relief well enough to interpolate its tilt.

    Parameters:
        field: The relief realization.
        detector_grid: The body's projection onto the detector grid.

    Returns:
        True when the relief's correlation length, in radians times the body's
        smaller projected semi-axis, spans at least :data:`_MIN_INTERPOLATED_CORR_PX`
        detector pixels.
    """
    corr_px = math.radians(field.corr_deg) * min(detector_grid.semi_a, detector_grid.semi_b)
    return corr_px >= _MIN_INTERPOLATED_CORR_PX


def relief_normals(
    grid: EllipsoidProjection,
    field: ReliefField,
    normals: tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType],
    *,
    illumination_angle: float,
    phase_angle: float,
) -> ReliefShading:
    """Replace the smooth normals with the relief surface's, and cast its shadows.

    Parameters:
        grid: The geometry to evaluate on.
        field: The relief realization.
        normals: The smooth normals on ``grid``; the arrays are updated in place
            where the relief reaches.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians.

    Returns:
        The relief's shading at the pixels it reaches, which include every pixel
        inside its outline.
    """
    normal_v, normal_u, normal_z = normals
    # Relief reaches only pixels inside its outline, plus the pixel an
    # interpolation reads beyond it; farther out the smooth limb normals carry
    # the continuity and the silhouette masks them.
    reach = 1.0 + max(field.h_max, 0.0) + 2.0 / max(min(grid.semi_a_outline, grid.semi_b), 1.0)
    near = grid.ellipse_dist_sq < reach * reach
    shading = relief_shading(
        grid, field, near, illumination_angle=illumination_angle, phase_angle=phase_angle
    )
    normal_v[near] = shading.normal_v
    normal_u[near] = shading.normal_u
    normal_z[near] = shading.normal_z
    return shading


def add_detector_relief(
    work_grid: EllipsoidProjection,
    detector_grid: EllipsoidProjection,
    field: ReliefField,
    normals: tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType],
    *,
    up_factor: int,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayBoolType:
    """Tilt the working grid's smooth normals by the relief, and cast its shadows.

    The relief's normals are computed on the detector grid, and the tilt they give
    the smooth normals is interpolated onto the working grid.  The relief's normals
    and shadows are computed on the working grid instead within
    :data:`_LIMB_BAND_PX` detector pixels of the limb, along the relief's own
    day-night edge, in every cell its cast shadows reach, and in the cells
    :func:`~spindoctor.sim.render.bodies.ellipsoid_grid.stretched_sight_cells` marks,
    with their neighbors.

    Parameters:
        work_grid: The working-grid geometry, cropped to the body.
        detector_grid: The detector-grid geometry, cropped to the same rectangle.
        field: The relief realization.
        normals: The smooth normals on the working grid, updated in place.
        up_factor: Pixels of the working grid per detector pixel.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians.

    Returns:
        The relief's cast-shadow mask on the working grid.
    """
    normal_v, normal_u, normal_z = normals
    base_v, base_u, base_z = ellipsoid_image_normals(detector_grid)
    relief_v, relief_u, relief_z = base_v.copy(), base_u.copy(), base_z.copy()
    detector = relief_normals(
        detector_grid,
        field,
        (relief_v, relief_u, relief_z),
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
    )
    smooth = smooth_surface_directions(detector_grid)
    seen = smooth.copy()
    seen[detector.near] = detector.seen
    # The tilt is interpolated with a cubic spline: on relief whose correlation
    # length spans 13 detector pixels a bilinear tilt put the light 0.012 px
    # from shading on the working grid, and the spline 0.0007 px (#931).  It is
    # added without renormalizing: interpolation shortens the tilt but not the
    # normal's second-order shrink toward the smooth normal, so a renormalized
    # sum would brighten rough terrain.  As it is, on the day side a Lambert
    # surface's brightness is the smooth brightness plus the interpolated
    # difference between the relief and smooth brightness.
    normal_v += cubic_upsample(relief_v - base_v, up_factor)
    normal_u += cubic_upsample(relief_u - base_u, up_factor)
    normal_z += cubic_upsample(relief_z - base_z, up_factor)
    relief_cos_i = incidence_cosines(
        relief_v,
        relief_u,
        relief_z,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
    )
    # A cast shadow has a sharp edge, which interpolation would blur, so every
    # cell a cast shadow darkens is shaded on the working grid.  Only a pixel
    # facing the sun is darkened by a cast shadow; one facing away is night by
    # its own normal.
    band = limb_band(work_grid, field, width_px=_LIMB_BAND_PX * up_factor)
    band |= expand_to_working_grid(day_night_edge_cells(relief_cos_i > 0.0), up_factor)
    band |= expand_to_working_grid(
        neighboring_cells(detector.shadowed & (relief_cos_i > 0.0)), up_factor
    )
    # Where a ridge hides the terrain behind it, the point a pixel sees jumps from
    # the ridge to the terrain beyond, and the tilt jumps with it; where a slope
    # turns nearly edge-on, it changes faster than one detector cell can follow.
    # Interpolation would blur both, so those cells are shaded on the working grid.
    band |= expand_to_working_grid(
        neighboring_cells(stretched_sight_cells(seen, smooth)), up_factor
    )
    shading = relief_shading(
        work_grid,
        field,
        band,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
    )
    normal_v[band] = shading.normal_v
    normal_u[band] = shading.normal_u
    normal_z[band] = shading.normal_z
    return shading.shadowed


def relief_shading(
    grid: EllipsoidProjection,
    field: ReliefField,
    near: NDArrayBoolType,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> ReliefShading:
    """The relief surface's image-frame normals and cast shadows at the ``near`` pixels.

    Each shading pixel reads the terrain where its line of sight first meets the
    relief surface, at the direction ``p`` that
    :func:`~spindoctor.sim.render.bodies.relief_sight.visible_surface_directions` finds
    in the body's parameter space (the smooth body is the unit sphere the shared
    projection's surface coordinates lie on).  The relief surface ``(1 + h) p`` has the
    parameter-space normal ``p - grad h / (1 + h)``, carried to the body frame by the
    ellipsoid's inverse semi-axes, tipped by the sub-observer latitude, and rotated
    into the image like the smooth normals.  The sun direction goes to parameter space
    by the same affine map, where
    :func:`~spindoctor.sim.render.bodies.relief.cast_shadows` marches the pixels whose
    sun elevation lies within the field's slope band; a pixel below the band always
    lies in shadow.  A normal that leans away from the observer is laid into the sky
    plane, so every returned ``normal_z`` is at least 0.

    Parameters:
        grid: The shading-resolution geometry.
        field: The relief realization.
        near: The shading pixels the relief can reach.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians.

    Returns:
        The normals and the points seen at the ``near`` pixels, and the cast-shadow
        mask on ``grid``.
    """
    semi = np.array([grid.semi_a, grid.semi_b, grid.semi_c], dtype=np.float64)
    cos_rz = grid.cos_rz
    sin_rz = grid.sin_rz
    geometry = grid.geometry
    e2 = grid.ellipse_dist_sq[near]
    p = visible_surface_directions(
        grid.v_rot[near] / grid.semi_a_outline, grid.u_rot[near] / grid.semi_b, field
    )
    h = field.sample_points(p)
    normal_param = p - field.surface_gradient(p) / (1.0 + h)[..., np.newaxis]
    normal_body = normal_param / semi
    normal_view = np.stack(
        geometry.to_view(normal_body[..., 0], normal_body[..., 1], normal_body[..., 2]),
        axis=-1,
    )
    # The surface a line of sight meets first faces the observer, but a normal taken
    # by the field's half-cell differences can lean slightly away where the line of
    # sight grazes the terrain, and one read where the line of sight misses the
    # surface can lean away by any amount.  Such a normal is laid into the sky plane,
    # where the smooth body's normals lie beyond its limb.
    normal_view[..., 2] = np.maximum(normal_view[..., 2], 0.0)
    normal_view /= np.maximum(np.linalg.norm(normal_view, axis=-1, keepdims=True), 1e-300)
    normal_v = normal_view[..., 0] * cos_rz + normal_view[..., 1] * sin_rz
    normal_u = -normal_view[..., 0] * sin_rz + normal_view[..., 1] * cos_rz
    normal_z = normal_view[..., 2]

    # The sun in the body frame (the inverse of the normals' in-plane
    # rotation), then in parameter space, where directions scale by 1 / semi.
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )
    sun_body = np.array(
        geometry.to_body(
            illum_v * cos_rz - illum_u * sin_rz,
            illum_v * sin_rz + illum_u * cos_rz,
            illum_z,
        )
    )
    sun = sun_body / semi
    sun /= float(np.linalg.norm(sun))

    on_body = e2 < (1.0 + max(field.h_max, 0.0)) ** 2
    cos_beta = stable_matmul3(p.reshape(-1, 3), sun).reshape(p.shape[:-1])
    tan_elevation = cos_beta / np.sqrt(np.maximum(1.0 - cos_beta**2, 1e-24))
    band = _SLOPE_BAND_MARGIN * field.slope_max / (1.0 + field.h_min)
    shadowed_near = on_body & (tan_elevation <= -band)
    marched = on_body & (np.abs(tan_elevation) < band)
    if marched.any():
        cell = 2.0 * np.pi / field.grid.shape[0]
        shadowed_near[marched] = cast_shadows(
            p[marched], sun, field, step=cell / _MARCH_SAMPLES_PER_CELL
        )
    shadowed = np.zeros(grid.ellipse_dist_sq.shape, dtype=np.bool_)
    shadowed[near] = shadowed_near
    return ReliefShading(
        near=near,
        normal_v=normal_v,
        normal_u=normal_u,
        normal_z=normal_z,
        seen=p,
        shadowed=shadowed,
    )
