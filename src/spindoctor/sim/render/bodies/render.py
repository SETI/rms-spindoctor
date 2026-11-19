"""Image-side body placement: one body into the frame.

:func:`render_single_body` draws one scene body.  An ellipsoid goes to the
ellipsoid renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`) and a
polyhedral-mesh body to the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`); either way the body is drawn once at
a reference center, cached, and translated into place.

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
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['render_single_body']


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
