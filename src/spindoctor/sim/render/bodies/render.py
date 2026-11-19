"""Image-side body placement: one body into the frame, plus crater carving.

:func:`render_single_body` draws one scene body.  An ellipsoid goes to the
ellipsoid renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`) and a
polyhedral-mesh body to the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`); either way the body is drawn once at
a reference center, cached, and translated into place.  The crater carving the
ellipsoid renderer uses lives here too.

The crater and relief knobs live only on this side of the information
boundary: they are truth keys the navigator never sees (its best model is
the smooth Lambert ellipsoid).
"""

from functools import lru_cache
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.atmosphere import (
    HaloScreen,
    apply_atmosphere,
    atmosphere_spec_from_params,
)
from spindoctor.sim.render.bodies.ellipsoid import (
    EllipsoidBodyRender,
    EllipsoidBodySpec,
    render_ellipsoid_body,
)
from spindoctor.sim.render.bodies.spec import BodySpec
from spindoctor.sim.render.bodies.texture import albedo_spec_from_params
from spindoctor.sim.render.bodies.translate import nonzero_box, shifted_box, translate_box
from spindoctor.sim.seeds import derive_effect_seed, stable_param_seed
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['render_single_body']


def carve_crater_heights(
    ellipse_mask_nz: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    crater_seed: int,
    n_craters: int,
    R_min: float,
    R_max: float,
    crater_power_law_exponent: float,
    crater_relief_scale: float,
    work_center_v: float,
    work_center_u: float,
) -> NDArrayFloatType:
    """Carve the crater height field (bowl + walls + raised rim per crater).

    The ellipsoid renderer carves its craters on the detector grid; the draw rule
    below is its determinism contract for a given seed and grid.

    Each crater draws from its own stream, seeded by
    ``derive_effect_seed(crater_seed, 'crater:<index>')``: one uniform value for the
    radius, then one normal value for the depth scatter, then its center (see
    :func:`_draw_crater_center`).  A crater's values therefore depend only on the
    seed, its index, the grid's dimensions, and the mask at the candidate pixels
    its own stream picks.  Changing one interior pixel moves only a crater whose
    center search reaches that pixel, and a change in the crater count adds or
    drops craters at the end of the list without touching the others.

    Parameters:
        ellipse_mask_nz: Boolean mask of pixels strictly inside the ellipse.
        v_coords: Centered v pixel coordinates of the grid.
        u_coords: Centered u pixel coordinates of the grid.
        crater_seed: The body's crater seed, from which each crater derives its stream.
        n_craters: Number of craters to place.
        R_min: Minimum crater radius in pixels.
        R_max: Maximum crater radius in pixels.
        crater_power_law_exponent: Power law exponent for crater radii, above 1.
        crater_relief_scale: Scale factor for crater depth.
        work_center_v: Body center v on the grid.
        work_center_u: Body center u on the grid.

    Returns:
        The crater height field over the grid, in its pixels.

    Raises:
        ValueError: If ``crater_power_law_exponent`` is not above 1 and a crater is
            placed.
    """

    # ------------------------------------------------------------------
    # 0. Heightmap we will add craters to
    # ------------------------------------------------------------------
    height = np.zeros_like(ellipse_mask_nz, dtype=float)
    if n_craters <= 0 or not ellipse_mask_nz.any():
        return height
    if crater_power_law_exponent <= 1:
        raise ValueError('alpha must be > 1 for a proper power law.')

    # ------------------------------------------------------------------
    # 1. Radius distribution: power law p(R) ~ R^(-alpha) on [R_min, R_max]
    # ------------------------------------------------------------------
    power = 1.0 - crater_power_law_exponent
    R_min_a = R_min**power
    R_max_a = R_max**power

    # Distances from each crater center are taken in the frame's own pixel
    # corner coordinates so the two frames cannot disagree.  v_coords and
    # u_coords are pixel corner coordinates with the body center already taken
    # off, so adding it back restores them.
    v_abs = v_coords + work_center_v
    u_abs = u_coords + work_center_u

    # ------------------------------------------------------------------
    # 2. Crater placement + geometry
    # ------------------------------------------------------------------
    for index in range(n_craters):
        rng = np.random.default_rng(derive_effect_seed(crater_seed, f'crater:{index}'))
        crater_radius = float((R_min_a + rng.uniform() * (R_max_a - R_min_a)) ** (1.0 / power))
        # Multiplicative depth scatter around the size-dependent mean of 2a
        noise = np.exp(rng.normal(loc=0.0, scale=0.25))
        v_crater, u_crater = _draw_crater_center(ellipse_mask_nz, rng)

        # The crater center is an array row and column, so the half pixel goes on here.
        v_dist = v_abs - (v_crater + PIXEL_CENTER_TO_CORNER_PX)
        u_dist = u_abs - (u_crater + PIXEL_CENTER_TO_CORNER_PX)
        crater_dist = np.sqrt(v_dist**2 + u_dist**2)

        # Mask where crater affects height (slightly beyond radius for rim)
        # TODO make config parameter
        crater_mask = (crater_dist < crater_radius * 1.1) & ellipse_mask_nz

        r = crater_dist[crater_mask]
        D = 2.0 * crater_radius  # diameter

        # ------------------------------------------------------------------
        # 2a. Depth as function of size (small craters deeper, large shallower)
        # ------------------------------------------------------------------
        # Choose size range where we blend d/D ratio
        D_small = 10.0
        D_large = R_max * 2.0  # roughly largest diameters

        # Blend factor in [0,1]
        t = np.clip((D - D_small) / (D_large - D_small + 1e-9), 0.0, 1.0)
        # d/D: small -> 0.20, large -> 0.07
        d_over_D_mean = (1.0 - t) * 0.20 + t * 0.07
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


# Candidate pixels drawn per round of the crater center search
_CENTER_CANDIDATES_PER_ROUND = 256


def _draw_crater_center(mask: NDArrayBoolType, rng: np.random.Generator) -> tuple[int, int]:
    """Draw a crater center uniformly among a mask's pixels.

    Candidates are drawn in rounds of a fixed number of pixels, each uniform over
    the whole grid, and the first candidate inside the mask is the center.  The
    candidates depend only on the stream and the grid's dimensions, never on the
    mask, so changing one mask pixel changes the center only when one of the
    candidates up to and including the chosen one is that pixel.

    Parameters:
        mask: The pixels a center may land on; at least one must be set.
        rng: The crater's own random stream.

    Returns:
        The center's array row and column.
    """
    flat = mask.ravel()
    while True:
        candidates = rng.integers(flat.size, size=_CENTER_CANDIDATES_PER_ROUND)
        hits = candidates[flat[candidates]]
        if hits.size:
            row, col = divmod(int(hits[0]), mask.shape[1])
            return row, col


def perturb_normals_by_slope(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    slope_v: NDArrayFloatType,
    slope_u: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Perturb base-surface unit normals by the slopes of a height field carved along z.

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
        slope_v: The height field's slope along v (height per pixel of the same
            length unit).
        slope_u: The height field's slope along u.

    Returns:
        Tuple of (v, u, z) perturbed unit-normal component arrays.
    """
    perturbed_v = normal_v - normal_z * slope_v
    perturbed_u = normal_u - normal_z * slope_u
    perturbed_mag = np.sqrt(perturbed_v**2 + perturbed_u**2 + normal_z**2)
    perturbed_mag = np.maximum(perturbed_mag, 1e-10)  # Avoid division by zero
    perturbed_v /= perturbed_mag
    perturbed_u /= perturbed_mag
    perturbed_z = normal_z / perturbed_mag
    return perturbed_v, perturbed_u, perturbed_z


@lru_cache(maxsize=30)
def _render_ellipsoid_shape_cached(
    size_v: int, size_u: int, spec: EllipsoidBodySpec
) -> EllipsoidBodyRender:
    """Cache ellipsoid body shapes at the reference center (image center).

    The frozen spec is the cache key, so bodies differing in any geometry,
    texture, relief, or photometry knob occupy distinct entries; the caller
    translates the shape and its silhouette into place.
    """
    return render_ellipsoid_body((size_v, size_u), (size_v / 2.0, size_u / 2.0), spec)


def render_single_body(
    img: NDArrayFloatType,
    spec: BodySpec,
    offset_v: float,
    *,
    offset_u: float,
    body_streams_seed: int | None = None,
    body_index: int = 0,
    ref_center_v: float,
    ref_center_u: float,
    oversample: int = 1,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render a single body into the image.

    Parameters:
        img: Image array to modify in-place.
        spec: The body, parsed by
            :func:`~spindoctor.sim.render.bodies.spec.body_spec_from_params`.
            Its centers and axes must already be scaled to the render grid.  Its
            atmosphere's pixel lengths are scaled here, by ``oversample``.  Its
            surface texture and transit lengths stay in detector pixels, the grid
            the ellipsoid renderer lays surface detail out on.
        offset_v: V offset to apply.
        offset_u: U offset to apply.
        body_streams_seed: The scene's seed for every body's random streams.  Each
            body mixes its index and name into it to get its own identity seed,
            and its craters, relief terrain, albedo noise, and mesh pose scatter
            each draw from a named stream of that identity seed, so bodies with
            identical geometry still get independent surfaces.  None falls back
            to the body's own ``seed`` and then to a seed derived from its
            geometry.
        body_index: Stable index of this body in the scene's body list, mixed
            into the body's identity seed alongside the body name.
        ref_center_v: Reference center V for body shape caching.
        ref_center_u: Reference center U for body shape caching.
        oversample: The render grid's oversampling factor.  An ellipsoid is shaded
            on its working grid, the oversampled grid or, at factor 1, its
            anti-aliasing supersample grid (see
            :class:`~spindoctor.sim.render.bodies.ellipsoid.EllipsoidBodySpec`).

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        the upper-cased name, the inventory item, and, for a mesh body whose
        pose scatter drew a perturbation, ``pose_scatter_drawn_deg``.
    """
    size_v, size_u = img.shape
    body_name = spec.name.upper()

    center_v = (size_v / 2.0 if spec.center_v is None else spec.center_v) + offset_v
    center_u = (size_u / 2.0 if spec.center_u is None else spec.center_u) + offset_u

    semi_a_px = spec.semi_a_px
    semi_b_px = spec.semi_b_px
    semi_c_px = spec.semi_c_px

    rotation_z = np.radians(spec.rotation_z)
    rotation_tilt = np.radians(spec.rotation_tilt)
    illumination_angle = np.radians(spec.illumination_angle)
    phase_angle = np.radians(spec.phase_angle)

    crater_fill = spec.crater_fill
    crater_min_radius = spec.crater_min_radius
    crater_max_radius = spec.crater_max_radius
    crater_power_law_exponent = spec.crater_power_law_exponent
    crater_relief_scale = spec.crater_relief_scale
    anti_aliasing = spec.anti_aliasing

    limb_relief_rms = spec.limb_relief_rms
    limb_relief_corr_deg = spec.limb_relief_corr_deg
    photometric_law = spec.photometric_law

    # Mix a stable per-body identity (scene index and name) into the scene's
    # body streams seed so two bodies with identical geometry draw independent crater
    # patterns and occupy distinct shape-cache entries.  The derivation is
    # SHA-256 based (never Python's salted hash), so it is process-stable.  A
    # body without craters never consumes the seed; None keeps identical
    # smooth bodies sharing one cached shape.
    crater_seed: int | None
    if crater_fill <= 0.0:
        crater_seed = None
    elif body_streams_seed is not None:
        crater_seed = derive_effect_seed(body_streams_seed, f'body:{body_index}:{body_name}')
    else:
        crater_seed = spec.seed

    # Projected half-extents of the body in the image plane.  The rendered
    # silhouette is the ellipse of semi-axes a and b rotated in-plane by
    # rotation_z (semi-axis c is the depth axis and does not project), so the
    # bounding half-extents along the image axes follow from the rotated ellipse.
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    half_extent_v = float(stable_hypot(semi_a_px * cos_rz, semi_b_px * sin_rz))
    half_extent_u = float(stable_hypot(semi_a_px * sin_rz, semi_b_px * cos_rz))

    # Get cached body shape (at reference center).  An irregular body renders
    # from a polyhedral mesh at a scene-supplied pose; otherwise an ellipsoid.
    if spec.shape_model == 'polyhedral_mesh':
        # Local import: mesh.py imports this module's finisher, so the mesh
        # path is resolved lazily to keep the module import graph acyclic.
        from spindoctor.sim.render.bodies.mesh import render_single_mesh_body

        return render_single_mesh_body(
            img,
            spec,
            body_name,
            center_v=center_v,
            center_u=center_u,
            semi_a_px=semi_a_px,
            semi_b_px=semi_b_px,
            semi_c_px=semi_c_px,
            illumination_angle=float(illumination_angle),
            phase_angle=float(phase_angle),
            # Silhouette anti-aliasing is consumed only at oversample 1,
            # matching the ellipsoid renderer: an oversampled scene already
            # resolves the limb on its own grid, and supersampling the mesh
            # rasterizer on top of it would multiply the render cost for no
            # rendered difference worth having.
            anti_aliasing=anti_aliasing if oversample == 1 else 0.0,
            ref_center_v=ref_center_v,
            ref_center_u=ref_center_u,
            body_streams_seed=body_streams_seed,
            body_index=body_index,
        )

    # The relief terrain and the albedo noise field each draw from their own
    # named stream of the per-body identity seed, so they are independent of the
    # crater draws (and of each other) and stable when other randomized features
    # are toggled.
    relief_seed = 0
    albedo_seed = 0
    if limb_relief_rms > 0.0 or spec.albedo_texture is not None:
        if body_streams_seed is not None:
            identity_seed = derive_effect_seed(body_streams_seed, f'body:{body_index}:{body_name}')
        elif spec.seed is not None:
            identity_seed = spec.seed & 0x7FFFFFFF
        else:
            # The fallback seed is keyed on the full widths, twice the semi-axes; the
            # terrain of the committed seed-less scenes depends on this key (#913).
            identity_seed = stable_param_seed(
                2.0 * semi_a_px, 2.0 * semi_b_px, 2.0 * semi_c_px, body_index, body_name
            )
        if limb_relief_rms > 0.0:
            relief_seed = derive_effect_seed(identity_seed, 'relief')
        if spec.albedo_texture is not None:
            albedo_seed = derive_effect_seed(identity_seed, 'albedo')
    ellipsoid_spec = EllipsoidBodySpec(
        semi_a_px=semi_a_px,
        semi_b_px=semi_b_px,
        semi_c_px=semi_c_px,
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
        crater_seed=crater_seed,
        oversample=int(oversample),
        limb_relief_rms=limb_relief_rms,
        limb_relief_corr_deg=limb_relief_corr_deg,
        relief_seed=relief_seed,
        photometric_law=photometric_law,
        minnaert_k=spec.minnaert_k,
        surge_amplitude=spec.surge_amplitude,
        surge_width_deg=spec.surge_width_deg,
        albedo_texture=albedo_spec_from_params(spec.albedo_texture, seed=albedo_seed),
        disc_texture=spec.disc_texture,
        transits=spec.transits or (),
    )
    # The silhouette comes back separately from the radiance: a transit umbra or a
    # zero-albedo spot renders black but still belongs to the body.
    ellipsoid_render = _render_ellipsoid_shape_cached(size_v, size_u, ellipsoid_spec)
    body_shape = ellipsoid_render.intensity
    silhouette = ellipsoid_render.silhouette

    # The exponential haze layer evaluates over the reference-centered disc
    # after shading; a body with no 'atmosphere' block never enters the haze
    # code and renders hard-limbed.
    # The on-disc haze joins the opaque disc paint; the above-limb glow rides
    # along as a translucent halo screen the radiance stage composites.
    # apply_atmosphere returns fresh arrays, so the shared render cache the
    # shape may come from is never mutated.
    halo: HaloScreen | None = None
    atmosphere_spec = atmosphere_spec_from_params(spec.atmosphere, oversample=int(oversample))
    if atmosphere_spec is not None:
        layers = apply_atmosphere(
            body_shape,
            atmosphere_spec,
            center_v=ref_center_v,
            center_u=ref_center_u,
            semi_a=semi_a_px,
            semi_b=semi_b_px,
            semi_c=semi_c_px,
            rotation_z=float(rotation_z),
            rotation_tilt=float(rotation_tilt),
            illumination_angle=float(illumination_angle),
            phase_angle=float(phase_angle),
            silhouette=silhouette,
        )
        body_shape = layers.disc
        halo = layers.halo

    return finish_single_body(
        img,
        body_shape,
        body_name=body_name,
        center_v=center_v,
        center_u=center_u,
        half_extent_v=half_extent_v,
        half_extent_u=half_extent_u,
        ref_center_v=ref_center_v,
        ref_center_u=ref_center_u,
        range_km=spec.range_km,
        halo=halo,
        silhouette=silhouette,
    )


def finish_single_body(
    img: NDArrayFloatType,
    body_shape: NDArrayFloatType,
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    ref_center_v: float,
    ref_center_u: float,
    range_km: float | None,
    halo: HaloScreen | None = None,
    silhouette: NDArrayFloatType | None = None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Translate a reference-centered body shape into place and composite it.

    Only the opaque body paints here (last writer wins); an atmospheric
    body's translucent halo is translated alongside it and returned on the
    info dict (key ``'halo'``) for the radiance stage to composite as a
    transmission screen, so the returned mask and the painted pixels are the
    solid silhouette only.  The mask is every pixel the silhouette touches,
    so the body paints, occludes stars and farther bodies, and claims its
    pixels in the truth maps there even where it renders black.

    Parameters:
        img: Image array to modify in-place.
        body_shape: Body intensity rendered at the reference center.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        ref_center_v: Reference center V the shape was rendered at.
        ref_center_u: Reference center U the shape was rendered at.
        range_km: The body's distance from the observer, recorded in the
            inventory.  None records 1.0.
        halo: The body's translucent halo screen at the reference center, or
            None for a body without an atmosphere.
        silhouette: The body's solid-outline pixel coverage at the reference
            center, or None when every pixel inside the body renders above
            zero, so ``body_shape > 0`` is the outline.

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        the name and the inventory item (plus the positioned halo screen under
        ``'halo'`` when the body carries one).
    """
    dv = center_v - ref_center_v
    du = center_u - ref_center_u
    # An exactly zero translation is the bitwise identity for the order-1
    # spline, so a body already at the reference center skips the full-frame
    # interpolation (the shape is only read, never written).
    if dv == 0.0 and du == 0.0:
        positioned_body = body_shape
        positioned_outline = body_shape if silhouette is None else silhouette
    else:
        # Only the box the body covers is resampled; see the translate module.
        lit_planes = (body_shape,) if silhouette is None else (body_shape, silhouette)
        box_v, box_u = nonzero_box(*lit_planes)
        positioned_body = translate_box(body_shape, dv, du, box_v, box_u)
        positioned_outline = (
            positioned_body
            if silhouette is None
            else translate_box(silhouette, dv, du, box_v, box_u)
        )
    mask = positioned_outline > 0
    img[mask] = positioned_body[mask]

    positioned_halo: HaloScreen | None = None
    if halo is not None:
        if dv == 0.0 and du == 0.0:
            # The screen already excludes every pixel the (unshifted) disc
            # painted, so it is positioned as-is.
            positioned_halo = halo
        else:
            size_v, size_u = img.shape
            emission = translate_box(halo.emission, dv, du, halo.box_v, halo.box_u)
            transmission = translate_box(
                halo.transmission, dv, du, halo.box_v, halo.box_u, cval=1.0
            )
            box_v = shifted_box(halo.box_v, dv, size_v)
            box_u = shifted_box(halo.box_u, du, size_u)
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
        'range': 1.0 if range_km is None else range_km,
    }

    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': inventory_item,
    }
    if positioned_halo is not None:
        body_info['halo'] = positioned_halo
    return mask, body_info
