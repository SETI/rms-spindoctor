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
from spindoctor.sim.render.bodies.mesh import render_single_mesh_body
from spindoctor.sim.render.bodies.placement import finish_single_body
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed
from spindoctor.sim.render.bodies.texture import albedo_spec_from_params
from spindoctor.sim.seeds import derive_effect_seed
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

    # Each body's craters draw from its identity seed, so two bodies with identical
    # geometry draw independent crater patterns and occupy distinct shape-cache
    # entries.  A body without craters never consumes the seed; None keeps
    # identical smooth bodies sharing one cached shape.
    crater_seed: int | None = None
    if crater_fill > 0.0:
        # The fallback digest hashes the full widths, twice the semi-axes, and the
        # body center on the detector grid, where the craters are carved, of the
        # shape drawn at the frame center.  A body drawn without seeds keeps its
        # craters (#913).
        os_in = max(1, int(oversample))
        crater_seed = body_identity_seed(
            body_streams_seed,
            spec,
            body_index,
            fallback_key=(
                2.0 * semi_a_px,
                2.0 * semi_b_px,
                2.0 * semi_c_px,
                (size_v / 2.0 / os_in, size_u / 2.0 / os_in),
            ),
        )

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
        # The fallback digest hashes the full widths, twice the semi-axes, so a body
        # drawn without seeds keeps its terrain (#913).
        identity_seed = body_identity_seed(
            body_streams_seed,
            spec,
            body_index,
            fallback_key=(2.0 * semi_a_px, 2.0 * semi_b_px, 2.0 * semi_c_px, body_index, body_name),
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
