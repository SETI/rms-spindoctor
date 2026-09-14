"""Image-side ellipsoid body renderer and its render-path dispatch.

A smooth Lambert ellipsoid at oversample 1 renders through the classic
path here (:func:`create_simulated_body`, optionally with procedural
crater texture).  A body that needs more -- a limb-relief field, a
non-Lambert photometric law, an opposition surge, or an oversampled
radiance grid -- dispatches to the topographic renderer
(:mod:`spindoctor.sim.forward.body_topo`), which shares this module's
crater carving and the ellipsoid shading conventions of
:mod:`spindoctor.sim.ellipsoid_geometry`, so a scene's planted geometry
error is the only difference between rendered and predicted silhouettes.

The crater and relief knobs live only on this side of the information
boundary: they are truth keys the navigator never sees (its best model is
the smooth Lambert ellipsoid).
"""

import math
from functools import lru_cache
from typing import Any, cast

import numpy as np
from scipy import ndimage

from spindoctor.sim.ellipsoid_geometry import (
    ellipsoid_image_normals,
    lambert_from_normals,
    project_ellipsoid,
)
from spindoctor.sim.forward.atmosphere import (
    HaloScreen,
    apply_atmosphere,
    atmosphere_spec_from_params,
)
from spindoctor.sim.forward.body_texture import (
    albedo_spec_from_params,
    disc_texture_spec_from_params,
    transit_specs_from_params,
)
from spindoctor.sim.forward.body_topo import TopoBodySpec, create_topographic_body
from spindoctor.sim.seeds import derive_effect_seed, stable_param_seed
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['create_simulated_body', 'render_single_body']


def create_simulated_body(
    size: tuple[int, int],
    center: tuple[float, float],
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    crater_fill: float = 0.0,
    crater_min_radius: float = 0.05,
    crater_max_radius: float = 0.25,
    crater_power_law_exponent: float = 3.0,
    crater_relief_scale: float = 0.6,
    anti_aliasing: float = 0.0,
    seed: int | None = None,
) -> NDArrayFloatType:
    """Create a simulated planetary body as an ellipsoid with shading and surface features.

    The body is modeled as a 3D ellipsoid projected onto 2D. The ellipsoid can have internal
    craters of varying sizes and depths. The body is illuminated using Lambertian shading
    (cos(incidence)) based on the illumination direction and phase angle.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point pixels.
            (0.0, 0.0) is the top-left corner of pixel (0,0), (0.5, 0.5) is the
            center of pixel (0,0).
        axis1: The full width of axis 1 (a) of the ellipsoid in pixels.
        axis2: The full width of axis 2 (b) of the ellipsoid in pixels.
        axis3: The full width of axis 3 (c) of the ellipsoid in pixels (depth).
        rotation_z: Rotation angle around the viewing axis (z-axis) in radians (0 to 2pi).
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
            Controls how much the ellipsoid is tilted toward/away from the viewer.
        illumination_angle: Direction of illumination in the image plane in radians (0 to 2pi).
            0 radians is at the top of the image, pi/2 is to the right.
        phase_angle: Phase angle in radians (0 to pi).
            0 = head-on illumination (fully illuminated),
            pi/2 = side illumination (half illuminated),
            pi = back illumination (no visible illumination).
        crater_fill: Approximate fraction of the ellipse to fill with craters.
        crater_min_radius: Minimum radius of a crater as a fraction of axis1.
        crater_max_radius: Maximum radius of a crater as a fraction of axis1.
        crater_power_law_exponent: Power law exponent for the crater radius distribution.
        crater_relief_scale: Scale factor for the crater depth.
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.
        seed: Random seed for crater generation. If None, uses a hash-based seed.

    Returns:
        A 2D numpy array of shape (size_v, size_u) with float values from 0.0 to 1.0,
        where 0.0 is black and 1.0 is full white.
    """
    size_v, size_u = size

    # The projection grids (working-resolution coordinate frames, hemisphere
    # depth, coverage mask) come from the shared geometry module, so this
    # image-side renderer and the navigator-side predicted-body renderer
    # project through one implementation.
    proj = project_ellipsoid(
        size,
        center,
        axis1,
        axis2=axis2,
        axis3=axis3,
        rotation_z=rotation_z,
        rotation_tilt=rotation_tilt,
        anti_aliasing=anti_aliasing,
    )
    aa_scale = proj.aa_scale
    semi_major_axis = axis1 / 2.0
    work_center_v = proj.work_center_v
    work_center_u = proj.work_center_u
    work_semi_major = proj.work_semi_major
    work_semi_minor = proj.work_semi_minor
    work_semi_c = proj.work_semi_c
    cos_rz = proj.cos_rz
    sin_rz = proj.sin_rz
    v_coords = proj.v_coords
    u_coords = proj.u_coords
    v_rot = proj.v_rot
    u_rot = proj.u_rot
    z_coords = proj.z_coords
    ellipse_dist_sq = proj.ellipse_dist_sq
    ellipse_mask = proj.ellipse_mask

    if crater_fill > 0:
        total_area = np.sum(proj.inside_mask)
        avg_crater_area = (crater_max_radius * semi_major_axis * aa_scale) ** 2
        n_craters = np.clip(int(crater_fill * total_area / avg_crater_area), 0, 1000)

        # Use the provided seed, or fall back to a process-stable seed derived
        # from the body geometry.  The built-in hash is salted per process for
        # some types, so it cannot be used where determinism is required.
        if seed is not None:
            seed_value = int(seed) & 0x7FFFFFFF
        else:
            seed_value = stable_param_seed(axis1, axis2, axis3, center) & 0x7FFFFFFF
        rng = np.random.default_rng(seed_value)
        # Choose crater centers strictly inside the ellipse (exclude AA rim and exterior)
        ellipse_mask_nz = ellipse_dist_sq < 1.0
        nz = np.argwhere(ellipse_mask_nz)

        intensity = _add_craters_and_shading(
            ellipse_mask_nz,
            v_coords,
            u_coords,
            nz=nz,  # indices of non-zero ellipse pixels (list/array of (v,u))
            rng=rng,
            n_craters=n_craters,
            R_min=crater_min_radius * semi_major_axis,
            R_max=crater_max_radius * semi_major_axis,
            crater_power_law_exponent=crater_power_law_exponent,
            crater_relief_scale=crater_relief_scale,
            work_center_v=work_center_v,
            work_center_u=work_center_u,
            aa_scale=aa_scale,
            lighting_angle=illumination_angle,  # 0 = from top, pi/2 = from right
            phase_angle=phase_angle,  # 0 = from front, pi/2 = from side, pi = from back
            ellipse_mask=ellipse_mask,
            z_coords=z_coords,
            v_rot=v_rot,
            u_rot=u_rot,
            work_semi_major=work_semi_major,
            work_semi_minor=work_semi_minor,
            work_semi_c=work_semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
        )

    else:
        normal_v, normal_u, normal_z = ellipsoid_image_normals(
            ellipse_mask,
            v_rot,
            u_rot,
            z_coords=z_coords,
            work_semi_major=work_semi_major,
            work_semi_minor=work_semi_minor,
            work_semi_c=work_semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
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
    intensity = np.clip(intensity, 0.0, 1.0)

    return intensity


def _add_craters_and_shading(
    ellipse_mask_nz: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    nz: NDArrayIntType,
    rng: np.random.Generator,
    n_craters: int,
    R_min: float,
    R_max: float,
    crater_power_law_exponent: float,
    crater_relief_scale: float,
    work_center_v: float,
    work_center_u: float,
    aa_scale: int,
    lighting_angle: float,
    phase_angle: float,
    ellipse_mask: NDArrayFloatType,
    z_coords: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    work_semi_major: float,
    work_semi_minor: float,
    work_semi_c: float,
    cos_rz: float,
    sin_rz: float,
) -> NDArrayFloatType:
    """Carve craters into the ellipsoid surface and apply Lambertian shading.

    The base-surface normals and the Lambertian illumination model are shared
    with the smooth (no-crater) path via ``ellipsoid_image_normals`` and
    ``lambert_from_normals``, so a cratered body and a smooth body with the
    same pose and illumination shade identically outside the craters.

    Parameters:
        ellipse_mask_nz: Boolean mask of pixels strictly inside the ellipse.
        v_coords: Centered v pixel coordinates at working resolution.
        u_coords: Centered u pixel coordinates at working resolution.
        nz: Array of (v, u) indices of pixels inside the ellipse.
        rng: Seeded random stream for crater placement and sizes.
        n_craters: Number of craters to place.
        R_min: Minimum crater radius in pixels (before aa scaling).
        R_max: Maximum crater radius in pixels (before aa scaling).
        crater_power_law_exponent: Power law exponent for crater radii.
        crater_relief_scale: Scale factor for crater depth.
        work_center_v: Body center v at working resolution.
        work_center_u: Body center u at working resolution.
        aa_scale: Anti-aliasing supersampling factor.
        lighting_angle: In-plane light direction in radians (0 = from top).
        phase_angle: Phase angle in radians (0 = fully lit, pi = backlit).
        ellipse_mask: Ellipse coverage mask including the soft AA rim.
        z_coords: Depth of the visible ellipsoid surface at each pixel.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        work_semi_major: Semi-major axis (a) at working resolution.
        work_semi_minor: Semi-minor axis (b) at working resolution.
        work_semi_c: Depth semi-axis (c) at working resolution.
        cos_rz: Cosine of the rotation_z angle.
        sin_rz: Sine of the rotation_z angle.

    Returns:
        New intensity array with craters and lighting applied.
    """
    height = carve_crater_heights(
        ellipse_mask_nz,
        v_coords,
        u_coords,
        nz=nz,
        rng=rng,
        n_craters=n_craters,
        R_min=R_min,
        R_max=R_max,
        crater_power_law_exponent=crater_power_law_exponent,
        crater_relief_scale=crater_relief_scale,
        work_center_v=work_center_v,
        work_center_u=work_center_u,
        aa_scale=aa_scale,
    )

    # ----------------------------------------------------------------------
    # Lambertian shading: base-ellipsoid normals perturbed by crater relief
    # ----------------------------------------------------------------------
    # The base-surface normal comes from the same analytic formula (and the
    # same rotation_z back-rotation to image coordinates) as the smooth,
    # no-crater path, so both shading paths share one illumination convention.
    normal_v, normal_u, normal_z = ellipsoid_image_normals(
        ellipse_mask,
        v_rot,
        u_rot,
        z_coords=z_coords,
        work_semi_major=work_semi_major,
        work_semi_minor=work_semi_minor,
        work_semi_c=work_semi_c,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
    )
    perturbed_v, perturbed_u, perturbed_z = perturb_normals_by_height(
        normal_v, normal_u, normal_z, height
    )

    lambert = lambert_from_normals(
        perturbed_v,
        perturbed_u,
        perturbed_z,
        illumination_angle=lighting_angle,
        phase_angle=phase_angle,
    )

    # Apply ellipse mask (with AA edge)
    intensity_out = lambert * ellipse_mask
    # Ensure area strictly outside the ellipse is zeroed, to avoid any bleed with AA
    intensity_out[~ellipse_mask_nz] = 0.0
    return intensity_out


def carve_crater_heights(
    ellipse_mask_nz: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    nz: NDArrayIntType,
    rng: np.random.Generator,
    n_craters: int,
    R_min: float,
    R_max: float,
    crater_power_law_exponent: float,
    crater_relief_scale: float,
    work_center_v: float,
    work_center_u: float,
    aa_scale: int,
) -> NDArrayFloatType:
    """Carve the crater height field (bowl + walls + raised rim per crater).

    Shared by both body render paths: the classic Lambert path shades the
    working grid with these heights directly, and the topographic path
    carves them on its shading grid.  The rng consumption order (center,
    radius, depth noise per crater) is the paths' shared determinism
    contract for a given seed and grid.

    Parameters:
        ellipse_mask_nz: Boolean mask of pixels strictly inside the ellipse.
        v_coords: Centered v pixel coordinates at working resolution.
        u_coords: Centered u pixel coordinates at working resolution.
        nz: Array of (v, u) indices of pixels inside the ellipse.
        rng: Seeded random stream for crater placement and sizes.
        n_craters: Number of craters to place.
        R_min: Minimum crater radius in pixels (before aa scaling).
        R_max: Maximum crater radius in pixels (before aa scaling).
        crater_power_law_exponent: Power law exponent for crater radii.
        crater_relief_scale: Scale factor for crater depth.
        work_center_v: Body center v at working resolution.
        work_center_u: Body center u at working resolution.
        aa_scale: Anti-aliasing supersampling factor of the working grid.

    Returns:
        The crater height field over the working grid.
    """

    # ------------------------------------------------------------------
    # 0. Heightmap we will add craters to
    # ------------------------------------------------------------------
    height = np.zeros_like(ellipse_mask_nz, dtype=float)

    # ------------------------------------------------------------------
    # 1. Radius distribution: power law in [R_min, R_max]
    # ------------------------------------------------------------------
    def power_law_radius(
        rng: np.random.Generator,
        R_min: float,
        R_max: float,
        *,
        alpha: float,
        size: int | None = None,
    ) -> NDArrayFloatType:
        """Sample R from p(R) ~ R^(-alpha) on [R_min, R_max], alpha > 1."""
        if alpha <= 1:
            raise ValueError('alpha must be > 1 for a proper power law.')
        a = 1.0 - alpha
        R_min_a = R_min**a
        R_max_a = R_max**a
        u = rng.uniform(0.0, 1.0, size=size)
        R = (R_min_a + u * (R_max_a - R_min_a)) ** (1.0 / a)
        return cast(NDArrayFloatType, R)

    # ------------------------------------------------------------------
    # 2. Crater placement + geometry
    # ------------------------------------------------------------------
    for _ in range(n_craters):
        # Random crater center in non-zero ellipse area
        v_crater, u_crater = nz[int(rng.integers(len(nz)))]

        # Radius from power-law distribution
        crater_radius = power_law_radius(
            rng, R_min * aa_scale, R_max * aa_scale, alpha=crater_power_law_exponent
        )

        # Distances from the crater centre, taken in the frame's own pixel
        # corner coordinates so the two frames cannot disagree.  v_coords and
        # u_coords are pixel corner coordinates with the body centre already
        # taken off, so adding it back restores them.  The crater centre picked
        # above is an array row and column, so the half pixel goes on here.
        v_abs = v_coords + work_center_v
        u_abs = u_coords + work_center_u
        center_v_abs = float(v_crater) + PIXEL_CENTER_TO_CORNER_PX
        center_u_abs = float(u_crater) + PIXEL_CENTER_TO_CORNER_PX
        v_dist = v_abs - center_v_abs
        u_dist = u_abs - center_u_abs
        crater_dist = np.sqrt(v_dist**2 + u_dist**2)

        # Mask where crater affects height (slightly beyond radius for rim)
        # TODO make config parameter
        crater_mask = (crater_dist < crater_radius * 1.1) & ellipse_mask_nz
        if not np.any(crater_mask):
            continue

        r = crater_dist[crater_mask]
        D = 2.0 * crater_radius  # diameter

        # ------------------------------------------------------------------
        # 2a. Depth as function of size (small craters deeper, large shallower)
        # ------------------------------------------------------------------
        # Choose size range where we blend d/D ratio
        D_small = 10.0 * aa_scale
        D_large = R_max * 2.0  # roughly largest diameters

        # Blend factor in [0,1]
        t = np.clip((D - D_small) / (D_large - D_small + 1e-9), 0.0, 1.0)
        # d/D: small -> 0.20, large -> 0.07
        d_over_D_mean = (1.0 - t) * 0.20 + t * 0.07

        # Multiplicative noise around mean
        noise = np.exp(rng.normal(loc=0.0, scale=0.25))  # adjust scatter
        d_over_D = d_over_D_mean * noise

        crater_depth = crater_relief_scale * d_over_D * D

        # ------------------------------------------------------------------
        # 2b. Geometric profile: bowl + walls + raised rim
        # ------------------------------------------------------------------
        # TODO make config parameters
        R_floor = 0.6 * crater_radius  # flat-ish floor
        R_rim = crater_radius  # inner rim radius
        R_outer = 1.3 * crater_radius  # where rim merges back to surface

        local_profile = np.zeros_like(r)

        # Central bowl / floor (parabolic-ish)
        inside_floor = r <= R_floor
        local_profile[inside_floor] = -crater_depth * (1.0 - (r[inside_floor] / R_floor) ** 2)

        # Wall up to rim
        wall = (r > R_floor) & (r <= R_rim)
        t_wall = (r[wall] - R_floor) / (R_rim - R_floor + 1e-9)
        local_profile[wall] = -crater_depth * (1.0 - t_wall)

        # Raised rim outside
        rim = (r > R_rim) & (r <= R_outer)
        t_rim = (r[rim] - R_rim) / (R_outer - R_rim + 1e-9)
        rim_height = crater_depth * 0.25
        local_profile[rim] = rim_height * (1.0 - t_rim)

        # Add crater relief to global heightmap
        height[crater_mask] += local_profile

    return height


def perturb_normals_by_height(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    height: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Perturb base-surface unit normals by a height field carved along z.

    The height field is carved along the viewing axis, so the exact normal
    of the combined surface (base + height) is proportional to
    ``(-d(z + height)/dv, -d(z + height)/du, 1)``.  Scaling that vector by
    the base normal's z component rewrites it as the base normal plus a
    height-gradient term, which stays finite at the limb where the base
    surface gradient diverges.

    Parameters:
        normal_v: V component of the base unit surface normal.
        normal_u: U component of the base unit surface normal.
        normal_z: Z (toward-observer) component of the base unit normal.
        height: The height field on the same grid.

    Returns:
        Tuple of (v, u, z) perturbed unit-normal component arrays.
    """
    dheight_dv, dheight_du = np.gradient(height)
    perturbed_v = normal_v - normal_z * dheight_dv
    perturbed_u = normal_u - normal_z * dheight_du
    perturbed_mag = np.sqrt(perturbed_v**2 + perturbed_u**2 + normal_z**2)
    perturbed_mag = np.maximum(perturbed_mag, 1e-10)  # Avoid division by zero
    perturbed_v /= perturbed_mag
    perturbed_u /= perturbed_mag
    perturbed_z = normal_z / perturbed_mag
    return perturbed_v, perturbed_u, perturbed_z


@lru_cache(maxsize=30)
def _render_body_shape_cached(
    size_v: int,
    size_u: int,
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    rotation_z: float,
    rotation_tilt: float,
    illumination_angle: float,
    phase_angle: float,
    crater_fill: float,
    crater_min_radius: float,
    crater_max_radius: float,
    crater_power_law_exponent: float,
    crater_relief_scale: float,
    anti_aliasing: float,
    body_seed: int | None,
) -> NDArrayFloatType:
    """First layer cache: compute body shape at reference center (image center).

    Caches body shapes based on all parameters except center_v/center_u.
    Max size 30 allows caching up to 30 different body configurations.
    """
    # Use image center as reference - we'll translate when positioning
    ref_center_v = size_v / 2.0
    ref_center_u = size_u / 2.0

    sim_body = create_simulated_body(
        size=(size_v, size_u),
        center=(ref_center_v, ref_center_u),
        axis1=axis1,
        axis2=axis2,
        axis3=axis3,
        rotation_z=rotation_z,
        rotation_tilt=rotation_tilt,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        crater_fill=crater_fill,
        crater_min_radius=crater_min_radius,
        crater_max_radius=crater_max_radius,
        crater_power_law_exponent=crater_power_law_exponent,
        crater_relief_scale=crater_relief_scale,
        anti_aliasing=anti_aliasing,
        seed=body_seed,
    )
    return sim_body


@lru_cache(maxsize=30)
def _render_topo_shape_cached(size_v: int, size_u: int, spec: TopoBodySpec) -> NDArrayFloatType:
    """Cache topographic body shapes at the reference center (image center).

    The frozen spec is the cache key, so bodies differing in any geometry,
    texture, relief, or photometry knob occupy distinct entries; the caller
    translates the shape into place exactly as the classic path does.
    """
    return create_topographic_body((size_v, size_u), (size_v / 2.0, size_u / 2.0), spec)


def render_single_body(
    img: NDArrayFloatType,
    body_params: dict[str, Any],
    offset_v: float,
    *,
    offset_u: float,
    seed: int | None = None,
    body_index: int = 0,
    ref_center_v: float,
    ref_center_u: float,
    oversample: int = 1,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render a single body into the image.

    Parameters:
        img: Image array to modify in-place.
        body_params: Body parameters dictionary.
        offset_v: V offset to apply.
        offset_u: U offset to apply.
        seed: Scene-level crater seed; a per-body sub-seed is derived from it
            so bodies with identical geometry get independent crater patterns
            (and, further derived, independent relief terrains).
        body_index: Stable index of this body in the scene's body list, mixed
            into the per-body crater sub-seed alongside the body name.
        ref_center_v: Reference center V for body shape caching.
        ref_center_u: Reference center U for body shape caching.
        oversample: The render grid's oversampling factor.  The body's
            pixel-space parameters must already be scaled to this grid; the
            factor selects the topographic renderer's split-resolution path,
            which shades at the detector grid.

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        name, inventory item, and model params.
    """
    size_v, size_u = img.shape
    # The positional fallback matches the default the radiance stage assigns,
    # so a direct caller and the pipeline agree on an unnamed body's identity.
    body_name = str(body_params.get('name', f'SIM-BODY-{body_index + 1}')).upper()

    center_v = float(body_params.get('center_v', size_v / 2.0)) + offset_v
    center_u = float(body_params.get('center_u', size_u / 2.0)) + offset_u

    axis1 = float(body_params.get('axis1', 0.0))
    axis2 = float(body_params.get('axis2', 0.0))
    axis3 = float(body_params.get('axis3', min(axis1, axis2)))

    rotation_z = np.radians(body_params.get('rotation_z', 0.0))
    rotation_tilt = np.radians(body_params.get('rotation_tilt', 0.0))
    illumination_angle = np.radians(body_params.get('illumination_angle', 0.0))
    phase_angle = np.radians(body_params.get('phase_angle', 0.0))

    crater_fill = float(body_params.get('crater_fill', 0.0))
    crater_min_radius = float(body_params.get('crater_min_radius', 0.05))
    crater_max_radius = float(body_params.get('crater_max_radius', 0.25))
    crater_power_law_exponent = float(body_params.get('crater_power_law_exponent', 3.0))
    crater_relief_scale = float(body_params.get('crater_relief_scale', 0.6))
    anti_aliasing = float(body_params.get('anti_aliasing', 1.0))

    limb_relief_rms = float(body_params.get('limb_relief_rms', 0.0))
    limb_relief_corr_deg = float(body_params.get('limb_relief_corr_deg', 15.0))
    photometric_law = str(body_params.get('photometric_law', 'lambert'))
    minnaert_k = float(body_params.get('minnaert_k', 0.5))
    surge = body_params.get('opposition_surge') or {}
    surge_amplitude = float(surge.get('amplitude', 0.0))
    surge_width_deg = float(surge.get('width_deg', 6.0))

    # Mix a stable per-body identity (scene index and name) into the scene's
    # crater seed so two bodies with identical geometry draw independent crater
    # patterns and occupy distinct shape-cache entries.  The derivation is
    # SHA-256 based (never Python's salted hash), so it is process-stable.  A
    # body without craters never consumes the seed; None keeps identical
    # smooth bodies sharing one cached shape.
    body_seed: int | None
    if crater_fill <= 0.0:
        body_seed = None
    elif seed is not None:
        body_seed = derive_effect_seed(seed, f'body:{body_index}:{body_name}')
    else:
        body_seed = body_params.get('seed')

    # Projected half-extents of the body in the image plane.  The rendered
    # silhouette is the axis1/axis2 ellipse rotated in-plane by rotation_z
    # (axis3 is the depth axis and does not project), so the bounding
    # half-extents along the image axes follow from the rotated ellipse.
    semi_1 = axis1 / 2.0
    semi_2 = axis2 / 2.0
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    half_extent_v = float(np.hypot(semi_1 * cos_rz, semi_2 * sin_rz))
    half_extent_u = float(np.hypot(semi_1 * sin_rz, semi_2 * cos_rz))

    # Get cached body shape (at reference center).  An irregular body renders
    # from a polyhedral mesh at a scene-supplied pose; otherwise an ellipsoid.
    if str(body_params.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh':
        # Local import: body_mesh imports this module's finisher, so the mesh
        # path is resolved lazily to keep the module import graph acyclic.
        from spindoctor.sim.forward.body_mesh import render_single_mesh_body

        return render_single_mesh_body(
            img,
            body_params,
            body_name,
            center_v=center_v,
            center_u=center_u,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            illumination_angle=float(illumination_angle),
            phase_angle=float(phase_angle),
            # Silhouette anti-aliasing is consumed only at oversample 1,
            # matching the topographic path: an oversampled scene already
            # resolves the limb on its own grid, and supersampling the mesh
            # rasterizer on top of it would multiply the render cost for no
            # rendered difference worth having.
            anti_aliasing=anti_aliasing if oversample == 1 else 0.0,
            ref_center_v=ref_center_v,
            ref_center_u=ref_center_u,
            seed=seed,
            body_index=body_index,
        )

    has_texture = any(
        body_params.get(key) is not None for key in ('albedo_texture', 'disc_texture', 'transits')
    )

    # The classic smooth-Lambert path stays byte-for-byte for the scenes it
    # has always rendered; anything more (relief, a non-Lambert law, a surge,
    # surface texture / transits, or an oversampled grid, where the
    # split-resolution shading is also the render-time fast path) goes to the
    # topographic renderer.
    use_topo = (
        oversample > 1
        or limb_relief_rms > 0.0
        or photometric_law != 'lambert'
        or surge_amplitude > 0.0
        or has_texture
    )
    if use_topo:
        # The relief terrain and the albedo noise field each draw from their
        # own named stream of the per-body identity seed, so they are
        # independent of the crater draws (and of each other) and stable when
        # other randomized features are toggled.
        relief_seed = 0
        albedo_seed = 0
        if limb_relief_rms > 0.0 or body_params.get('albedo_texture') is not None:
            if seed is not None:
                identity_seed = derive_effect_seed(seed, f'body:{body_index}:{body_name}')
            elif body_params.get('seed') is not None:
                identity_seed = int(body_params['seed']) & 0x7FFFFFFF
            else:
                identity_seed = stable_param_seed(axis1, axis2, axis3, body_index, body_name)
            if limb_relief_rms > 0.0:
                relief_seed = derive_effect_seed(identity_seed, 'relief')
            if body_params.get('albedo_texture') is not None:
                albedo_seed = derive_effect_seed(identity_seed, 'albedo')
        spec = TopoBodySpec(
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=float(rotation_z),
            rotation_tilt=float(rotation_tilt),
            illumination_angle=float(illumination_angle),
            phase_angle=float(phase_angle),
            crater_fill=crater_fill,
            crater_min_radius=crater_min_radius,
            crater_max_radius=crater_max_radius,
            crater_power_law_exponent=crater_power_law_exponent,
            crater_relief_scale=crater_relief_scale,
            anti_aliasing=anti_aliasing,
            crater_seed=body_seed,
            oversample=int(oversample),
            limb_relief_rms=limb_relief_rms,
            limb_relief_corr_deg=limb_relief_corr_deg,
            relief_seed=relief_seed,
            photometric_law=photometric_law,
            minnaert_k=minnaert_k,
            surge_amplitude=surge_amplitude,
            surge_width_deg=surge_width_deg,
            albedo_texture=albedo_spec_from_params(body_params, seed=albedo_seed),
            disc_texture=disc_texture_spec_from_params(body_params),
            transits=transit_specs_from_params(body_params),
        )
        body_shape = _render_topo_shape_cached(size_v, size_u, spec)
    else:
        body_shape = _render_body_shape_cached(
            size_v,
            size_u,
            axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=rotation_z,
            rotation_tilt=rotation_tilt,
            illumination_angle=illumination_angle,
            phase_angle=phase_angle,
            crater_fill=crater_fill,
            crater_min_radius=crater_min_radius,
            crater_max_radius=crater_max_radius,
            crater_power_law_exponent=crater_power_law_exponent,
            crater_relief_scale=crater_relief_scale,
            anti_aliasing=anti_aliasing,
            body_seed=body_seed,
        )

    # The exponential haze layer evaluates over the reference-centred disc
    # after shading, so the same call serves both render paths; a body with no
    # 'atmosphere' block never enters the haze code and renders hard-limbed.
    # The on-disc haze joins the opaque disc paint; the above-limb glow rides
    # along as a translucent halo screen the radiance stage composites.
    # apply_atmosphere returns fresh arrays, so the shared render cache the
    # shape may come from is never mutated.
    halo: HaloScreen | None = None
    atmosphere_spec = atmosphere_spec_from_params(body_params, oversample=int(oversample))
    if atmosphere_spec is not None:
        layers = apply_atmosphere(
            body_shape,
            atmosphere_spec,
            center_v=ref_center_v,
            center_u=ref_center_u,
            semi_a=axis1 / 2.0,
            semi_b=axis2 / 2.0,
            semi_c=axis3 / 2.0,
            rotation_z=float(rotation_z),
            rotation_tilt=float(rotation_tilt),
            illumination_angle=float(illumination_angle),
            phase_angle=float(phase_angle),
        )
        body_shape = layers.disc
        halo = layers.halo

    return finish_single_body(
        img,
        body_shape,
        body_params,
        body_name=body_name,
        center_v=center_v,
        center_u=center_u,
        half_extent_v=half_extent_v,
        half_extent_u=half_extent_u,
        ref_center_v=ref_center_v,
        ref_center_u=ref_center_u,
        halo=halo,
    )


def _shifted_box(box: slice, delta: float, size: int) -> slice:
    """The frame slice conservatively covering a box translated by ``delta``.

    The order-1 interpolation reads one source pixel to each side, so the
    translated content lies within the source span shifted by ``delta`` and
    widened by one pixel each way, clipped to the frame.

    Parameters:
        box: The source slice.
        delta: The translation along this axis, in pixels.
        size: The frame extent along this axis.

    Returns:
        The translated, widened, clipped slice.
    """
    start = max(math.floor(box.start + delta) - 1, 0)
    stop = min(math.ceil(box.stop + delta) + 1, size)
    if start >= stop:
        return slice(0, 0)
    return slice(start, stop)


def finish_single_body(
    img: NDArrayFloatType,
    body_shape: NDArrayFloatType,
    body_params: dict[str, Any],
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    ref_center_v: float,
    ref_center_u: float,
    halo: HaloScreen | None = None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Translate a reference-centred body shape into place and composite it.

    Only the opaque body paints here (last writer wins); an atmospheric
    body's translucent halo is translated alongside it and returned on the
    info dict (key ``'halo'``) for the radiance stage to composite as a
    transmission screen, so the returned mask and the painted pixels are the
    solid silhouette only.

    Parameters:
        img: Image array to modify in-place.
        body_shape: Body intensity rendered at the reference center.
        body_params: Body parameters dictionary.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        ref_center_v: Reference center V the shape was rendered at.
        ref_center_u: Reference center U the shape was rendered at.
        halo: The body's translucent halo screen at the reference centre, or
            None for a body without an atmosphere.

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        name, inventory item, and model params (plus the positioned halo
        screen under ``'halo'`` when the body carries one).
    """
    dv = center_v - ref_center_v
    du = center_u - ref_center_u
    # An exactly zero translation is the bitwise identity for the order-1
    # spline, so a body already at the reference centre skips the full-frame
    # interpolation (the shape is only read, never written).
    if dv == 0.0 and du == 0.0:
        positioned_body = body_shape
    else:
        positioned_body = ndimage.shift(body_shape, (dv, du), order=1, mode='constant', cval=0.0)
    mask = positioned_body > 0
    img[mask] = positioned_body[mask]

    positioned_halo: HaloScreen | None = None
    if halo is not None:
        if dv == 0.0 and du == 0.0:
            # The screen already excludes every pixel the (unshifted) disc
            # painted, so it is positioned as-is.
            positioned_halo = halo
        else:
            size_v, size_u = img.shape
            emission = ndimage.shift(halo.emission, (dv, du), order=1, mode='constant', cval=0.0)
            transmission = ndimage.shift(
                halo.transmission, (dv, du), order=1, mode='constant', cval=1.0
            )
            box_v = _shifted_box(halo.box_v, dv, size_v)
            box_u = _shifted_box(halo.box_u, du, size_u)
            # The translation's linear interpolation bleeds the halo one pixel
            # into the painted rim; a screen never overlaps its own opaque
            # paint.  Outside the box the screen is identity already.
            mask_box = mask[box_v, box_u]
            emission_view = emission[box_v, box_u]
            emission_view[mask_box] = 0.0
            transmission_view = transmission[box_v, box_u]
            transmission_view[mask_box] = 1.0
            halo_mask = (emission_view > 0.0) | (transmission_view < 1.0)
            positioned_halo = HaloScreen(
                emission=emission,
                transmission=transmission,
                box_v=box_v,
                box_u=box_u,
                mask=halo_mask,
            )

    inventory_item = {
        'v_min_unclipped': center_v - half_extent_v,
        'v_max_unclipped': center_v + half_extent_v,
        'u_min_unclipped': center_u - half_extent_u,
        'u_max_unclipped': center_u + half_extent_u,
        'v_pixel_size': 2 * half_extent_v,
        'u_pixel_size': 2 * half_extent_u,
        # The inventory mirrors the oops inventory contract for real frames,
        # whose distance key is named 'range'; the scene schema key is the
        # per-body 'range_km'.
        'range': body_params.get('range_km', 1.0),
    }

    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': inventory_item,
        'params': body_params,
    }
    if positioned_halo is not None:
        body_info['halo'] = positioned_halo
    return mask, body_info
