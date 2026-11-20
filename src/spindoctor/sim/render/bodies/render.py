"""Image-side body placement: one body into the frame.

:func:`render_single_body` draws one scene body.  An ellipsoid goes to the
ellipsoid renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`) and a
polyhedral-mesh body to the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`); either way the body is drawn at a
reference center on its canvas, cached, and translated into place (see
:mod:`spindoctor.sim.render.bodies.placement`).

The crater and relief knobs live only on this side of the information
boundary: they are truth keys the navigator never sees (its best model is
the smooth Lambert ellipsoid).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.atmosphere import (
    AtmosphereSpec,
    HaloScreen,
    apply_atmosphere,
    atmosphere_spec_from_params,
    halo_reach,
)
from spindoctor.sim.render.bodies.ellipsoid import (
    EllipsoidBodyRender,
    EllipsoidBodySpec,
    render_ellipsoid_body,
)
from spindoctor.sim.render.bodies.ellipsoid_grid import CraterPlane
from spindoctor.sim.render.bodies.mesh import render_single_mesh_body
from spindoctor.sim.render.bodies.placement import (
    PARTIAL_DRAWINGS_KEPT,
    BodyCanvas,
    body_canvas,
    finish_single_body,
    finish_unseen_body,
)
from spindoctor.sim.render.bodies.relief import synthesize_relief_field
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed
from spindoctor.sim.render.bodies.texture import albedo_spec_from_params
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['clear_ellipsoid_shape_caches', 'render_single_body']


def _reference_center(size_v: int, size_u: int) -> tuple[float, float]:
    """The center an ellipsoid is drawn at for the shape cache: the frame center.

    Parameters:
        size_v: The render grid's rows.
        size_u: The render grid's columns.

    Returns:
        The center (v, u) in pixel corner coordinates.
    """
    return size_v / 2.0, size_u / 2.0


@lru_cache(maxsize=30)
def _render_ellipsoid_shape_cached(
    size_v: int, size_u: int, spec: EllipsoidBodySpec
) -> EllipsoidBodyRender:
    """Cache ellipsoid body shapes drawn on the frame at the reference center.

    The frozen spec is the cache key, so bodies differing in any geometry,
    texture, relief, or photometry knob occupy distinct entries; the caller
    translates the shape and its silhouette into place.
    """
    return render_ellipsoid_body((size_v, size_u), _reference_center(size_v, size_u), spec)


@lru_cache(maxsize=PARTIAL_DRAWINGS_KEPT)
def _render_partial_ellipsoid_cached(
    shape: tuple[int, int],
    center: tuple[float, float],
    crater_plane: CraterPlane | None,
    spec: EllipsoidBodySpec,
) -> EllipsoidBodyRender:
    """Cache the last ellipsoid bodies drawn on a canvas reaching past the frame.

    The cache keeps
    :data:`~spindoctor.sim.render.bodies.placement.PARTIAL_DRAWINGS_KEPT` drawings,
    enough for a scene with a few such bodies, each drawn twice when the scene
    smears each layer apart, and few enough that an offset sweep, which never reuses
    one, holds little memory.
    """
    return render_ellipsoid_body(shape, center, spec, crater_plane=crater_plane)


def clear_ellipsoid_shape_caches() -> None:
    """Drop the cached ellipsoid body shapes (test/tooling helper)."""
    _render_ellipsoid_shape_cached.cache_clear()
    _render_partial_ellipsoid_cached.cache_clear()


def render_single_body(
    img: NDArrayFloatType,
    spec: BodySpec,
    offset_v: float,
    *,
    offset_u: float,
    body_streams_seed: int | None = None,
    body_index: int = 0,
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
            body mixes its index and name into it to get its own identity seed (see
            :func:`~spindoctor.sim.render.bodies.spec.body_identity_seed`).  Its
            craters draw from that identity seed, and its relief terrain, albedo
            noise, and mesh pose scatter each from a named stream of it, so bodies
            with identical geometry still get independent surfaces.  None falls
            back to the body's own ``seed`` and then to a seed derived from its
            geometry.
        body_index: Stable index of this body in the scene's body list, mixed
            into the body's identity seed alongside the body name.
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
    center_v = (size_v / 2.0 if spec.center_v is None else spec.center_v) + offset_v
    center_u = (size_u / 2.0 if spec.center_u is None else spec.center_u) + offset_u

    # An irregular body renders from a polyhedral mesh at a scene-supplied pose;
    # every other body is an ellipsoid.
    if spec.shape_model == 'polyhedral_mesh':
        return render_single_mesh_body(
            img,
            spec,
            center_v=center_v,
            center_u=center_u,
            semi_a_px=spec.semi_a_px,
            semi_b_px=spec.semi_b_px,
            semi_c_px=spec.semi_c_px,
            illumination_angle=float(np.radians(spec.illumination_angle)),
            phase_angle=float(np.radians(spec.phase_angle)),
            # Silhouette anti-aliasing is consumed only at oversample 1,
            # matching the ellipsoid renderer: an oversampled scene already
            # resolves the limb on its own grid, and supersampling the mesh
            # rasterizer on top of it would multiply the render cost for no
            # rendered difference worth having.
            anti_aliasing=spec.anti_aliasing if oversample == 1 else 0.0,
            oversample=oversample,
            body_streams_seed=body_streams_seed,
            body_index=body_index,
        )

    return _render_single_ellipsoid_body(
        img,
        spec,
        center_v=center_v,
        center_u=center_u,
        body_streams_seed=body_streams_seed,
        body_index=body_index,
        oversample=oversample,
    )


def _render_single_ellipsoid_body(
    img: NDArrayFloatType,
    spec: BodySpec,
    *,
    center_v: float,
    center_u: float,
    body_streams_seed: int | None,
    body_index: int,
    oversample: int,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render one ellipsoid body into the image in place.

    The body is drawn at the reference center on its canvas through the shape cache,
    given its haze layer, and translated into place.

    Parameters:
        img: Image array to modify in-place.
        spec: The body, scaled to the render grid.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        body_streams_seed: The scene's seed for every body's random streams, or None.
        body_index: The body's position in the scene's body list.
        oversample: The render grid's oversampling factor.

    Returns:
        Tuple of (body_mask, body_info_dict), as :func:`render_single_body` returns.
    """
    size_v, size_u = img.shape
    seeds = _surface_seeds(
        spec,
        body_streams_seed=body_streams_seed,
        body_index=body_index,
        size=(size_v, size_u),
        oversample=oversample,
    )
    atmosphere_spec = atmosphere_spec_from_params(spec.atmosphere, oversample=int(oversample))
    reach_v, reach_u = _ellipsoid_reach(spec, atmosphere_spec, relief_seed=seeds.relief_seed)
    canvas = body_canvas(
        (size_v, size_u),
        center_v=center_v,
        center_u=center_u,
        reach_v=reach_v,
        reach_u=reach_u,
        oversample=oversample,
    )
    half_extent_v, half_extent_u = _projected_half_extents(spec)
    if canvas.is_empty:
        return finish_unseen_body(
            (size_v, size_u),
            body_name=spec.name.upper(),
            center_v=center_v,
            center_u=center_u,
            half_extent_v=half_extent_v,
            half_extent_u=half_extent_u,
            range_km=spec.range_km,
        )
    # The silhouette comes back separately from the radiance: a transit umbra or a
    # zero-albedo spot renders black but still belongs to the body.
    ellipsoid_spec = _ellipsoid_spec(spec, seeds, oversample=oversample)
    ellipsoid_render = (
        _render_ellipsoid_shape_cached(size_v, size_u, ellipsoid_spec)
        if canvas.on_frame
        else _render_partial_ellipsoid_cached(
            canvas.shape,
            canvas.center,
            _crater_plane(canvas, spec, oversample=oversample),
            ellipsoid_spec,
        )
    )
    body_shape, halo = _apply_body_atmosphere(
        ellipsoid_render, spec, atmosphere_spec, center=canvas.center
    )
    return finish_single_body(
        img,
        body_shape,
        body_name=spec.name.upper(),
        center_v=center_v,
        center_u=center_u,
        half_extent_v=half_extent_v,
        half_extent_u=half_extent_u,
        canvas=canvas,
        range_km=spec.range_km,
        halo=halo,
        silhouette=ellipsoid_render.silhouette,
    )


def _crater_plane(canvas: BodyCanvas, spec: BodySpec, *, oversample: int) -> CraterPlane | None:
    """The detector grid a partly drawn body's craters are laid out on.

    Parameters:
        canvas: The body's canvas.
        spec: The body (its crater fill is read).
        oversample: The render grid's oversampling factor.

    Returns:
        The crater plane in detector pixels, or None for a body drawn on the frame
        or without craters, whose craters are laid out on the canvas itself.
    """
    if canvas.on_frame or spec.crater_fill <= 0.0:
        return None
    step = max(1, int(oversample))
    return CraterPlane(
        shape=(
            (canvas.plane_rows.stop - canvas.plane_rows.start) // step,
            (canvas.plane_cols.stop - canvas.plane_cols.start) // step,
        ),
        origin=(
            (canvas.rows.start - canvas.plane_rows.start) // step,
            (canvas.cols.start - canvas.plane_cols.start) // step,
        ),
    )


def _ellipsoid_reach(
    spec: BodySpec, atmosphere_spec: AtmosphereSpec | None, *, relief_seed: int
) -> tuple[float, float]:
    """How far an ellipsoid body, its relief, and its halo reach from its center.

    The outline is the ellipse of semi-axes a and b, stretched along a by the tilt
    (see :func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid`) and rotated
    in-plane by ``rotation_z``.  Relief can lift the limb by the field's largest
    height, and a haze layer reaches out to its outer edge.

    Parameters:
        spec: The body, scaled to the render grid.
        atmosphere_spec: The body's haze spec on the render grid, or None.
        relief_seed: The seed of the body's relief terrain.

    Returns:
        The reach along v and along u, in render-grid pixels.
    """
    rotation_z = math.radians(spec.rotation_z)
    rotation_tilt = math.radians(spec.rotation_tilt)
    cos_rz = math.cos(rotation_z)
    sin_rz = math.sin(rotation_z)
    stretched_a = spec.semi_a_px / max(abs(math.cos(rotation_tilt)), 1e-6)
    lift = 1.0
    if spec.limb_relief_rms > 0.0:
        field = synthesize_relief_field(
            spec.limb_relief_rms, spec.limb_relief_corr_deg, relief_seed
        )
        lift += max(abs(field.h_min), abs(field.h_max))
    reach_v = lift * math.hypot(stretched_a * cos_rz, spec.semi_b_px * sin_rz)
    reach_u = lift * math.hypot(stretched_a * sin_rz, spec.semi_b_px * cos_rz)
    if atmosphere_spec is not None:
        halo_v, halo_u = halo_reach(
            atmosphere_spec,
            semi_a=spec.semi_a_px,
            semi_b=spec.semi_b_px,
            rotation_z=rotation_z,
            rotation_tilt=rotation_tilt,
            phase_angle=math.radians(spec.phase_angle),
        )
        reach_v = max(reach_v, halo_v)
        reach_u = max(reach_u, halo_u)
    return reach_v, reach_u


@dataclass(frozen=True)
class _SurfaceSeeds:
    """The seeds of an ellipsoid's randomized surface detail.

    Parameters:
        crater_seed: The crater seed, or None for a body without craters.
        relief_seed: The seed of the relief terrain (0 for a body without relief).
        albedo_seed: The seed of the albedo noise field (0 for a body without an
            albedo texture).
    """

    crater_seed: int | None
    relief_seed: int
    albedo_seed: int


def _surface_seeds(
    spec: BodySpec,
    *,
    body_streams_seed: int | None,
    body_index: int,
    size: tuple[int, int],
    oversample: int,
) -> _SurfaceSeeds:
    """The seeds of an ellipsoid's craters, relief terrain, and albedo noise.

    All three derive from the body's identity seed
    (:func:`~spindoctor.sim.render.bodies.spec.body_identity_seed`).  The craters
    take the identity seed itself; the relief and the albedo noise each take their
    own named stream of it, so they are independent of the craters and of each
    other, and stable when other randomized features are toggled.

    Parameters:
        spec: The body (its name, geometry, surface keys, and seed are read).
        body_streams_seed: The scene's seed for every body's random streams, or None.
        body_index: The body's position in the scene's body list.
        size: The render grid's dimensions (rows, columns).
        oversample: The render grid's oversampling factor.

    Returns:
        The body's surface seeds.
    """
    size_v, size_u = size
    semi_a_px = spec.semi_a_px
    semi_b_px = spec.semi_b_px
    semi_c_px = spec.semi_c_px

    # Two bodies with identical geometry draw independent crater patterns and
    # occupy distinct shape-cache entries.  A body without craters never consumes
    # the seed; None keeps identical smooth bodies sharing one cached shape.
    crater_seed: int | None = None
    if spec.crater_fill > 0.0:
        # The fallback digest hashes the full widths, twice the semi-axes, and the
        # body center on the detector grid, where the craters are carved, of the
        # shape drawn at the reference center (#913).  render_ellipsoid_body builds
        # its detector grid at that center divided by the same factor.
        os_in = max(1, int(oversample))
        ref_v, ref_u = _reference_center(size_v, size_u)
        crater_seed = body_identity_seed(
            body_streams_seed,
            spec,
            body_index,
            fallback_key=(
                2.0 * semi_a_px,
                2.0 * semi_b_px,
                2.0 * semi_c_px,
                (ref_v / os_in, ref_u / os_in),
            ),
        )

    relief_seed = 0
    albedo_seed = 0
    if spec.limb_relief_rms > 0.0 or spec.albedo_texture is not None:
        # The fallback digest hashes the full widths, twice the semi-axes (#913).
        identity_seed = body_identity_seed(
            body_streams_seed,
            spec,
            body_index,
            fallback_key=(
                2.0 * semi_a_px,
                2.0 * semi_b_px,
                2.0 * semi_c_px,
                body_index,
                spec.name.upper(),
            ),
        )
        if spec.limb_relief_rms > 0.0:
            relief_seed = derive_effect_seed(identity_seed, 'relief')
        if spec.albedo_texture is not None:
            albedo_seed = derive_effect_seed(identity_seed, 'albedo')
    return _SurfaceSeeds(crater_seed=crater_seed, relief_seed=relief_seed, albedo_seed=albedo_seed)


def _ellipsoid_spec(spec: BodySpec, seeds: _SurfaceSeeds, *, oversample: int) -> EllipsoidBodySpec:
    """The ellipsoid renderer's spec for one body: its angles in radians, its seeds set.

    Parameters:
        spec: The body.
        seeds: The body's surface seeds.
        oversample: The render grid's oversampling factor.

    Returns:
        The ellipsoid spec, which is also the body's shape-cache key.
    """
    return EllipsoidBodySpec(
        semi_a_px=spec.semi_a_px,
        semi_b_px=spec.semi_b_px,
        semi_c_px=spec.semi_c_px,
        rotation_z=float(np.radians(spec.rotation_z)),
        rotation_tilt=float(np.radians(spec.rotation_tilt)),
        illumination_angle=float(np.radians(spec.illumination_angle)),
        phase_angle=float(np.radians(spec.phase_angle)),
        crater_fill=spec.crater_fill,
        crater_min_radius=spec.crater_min_radius,
        crater_max_radius=spec.crater_max_radius,
        crater_power_law_exponent=spec.crater_power_law_exponent,
        crater_relief_scale=spec.crater_relief_scale,
        anti_aliasing=spec.anti_aliasing,
        crater_seed=seeds.crater_seed,
        crater_turn=float(np.radians(spec.roll_deg)),
        oversample=int(oversample),
        limb_relief_rms=spec.limb_relief_rms,
        limb_relief_corr_deg=spec.limb_relief_corr_deg,
        relief_seed=seeds.relief_seed,
        photometric_law=spec.photometric_law,
        minnaert_k=spec.minnaert_k,
        surge_amplitude=spec.surge_amplitude,
        surge_width_deg=spec.surge_width_deg,
        albedo_texture=albedo_spec_from_params(spec.albedo_texture, seed=seeds.albedo_seed),
        disc_texture=spec.disc_texture,
        transits=spec.transits or (),
    )


def _apply_body_atmosphere(
    ellipsoid_render: EllipsoidBodyRender,
    spec: BodySpec,
    atmosphere_spec: AtmosphereSpec | None,
    *,
    center: tuple[float, float],
) -> tuple[NDArrayFloatType, HaloScreen | None]:
    """Add a body's haze layer to its reference-centered drawing.

    The exponential haze layer evaluates over the reference-centered disc after
    shading.  The on-disc haze joins the opaque disc paint, and the above-limb glow
    rides along as a translucent halo screen the radiance stage composites.  A body
    that has no ``atmosphere`` block never enters the haze code and renders
    hard-limbed.

    Parameters:
        ellipsoid_render: The body drawn at the reference center on its canvas.
        spec: The body (its geometry is read).
        atmosphere_spec: The body's haze spec on the render grid, or None for a body
            without an atmosphere.
        center: The reference center (v, u) on the canvas, where the body was drawn.

    Returns:
        The body's opaque intensity and its halo screen, or the drawing's own
        intensity and None for a body without an atmosphere.
    """
    if atmosphere_spec is None:
        return ellipsoid_render.intensity, None
    # apply_atmosphere returns fresh arrays, so the shared render cache the
    # drawing may come from is never mutated.
    layers = apply_atmosphere(
        ellipsoid_render.intensity,
        atmosphere_spec,
        center_v=center[0],
        center_u=center[1],
        semi_a=spec.semi_a_px,
        semi_b=spec.semi_b_px,
        semi_c=spec.semi_c_px,
        rotation_z=float(np.radians(spec.rotation_z)),
        rotation_tilt=float(np.radians(spec.rotation_tilt)),
        illumination_angle=float(np.radians(spec.illumination_angle)),
        phase_angle=float(np.radians(spec.phase_angle)),
        silhouette=ellipsoid_render.silhouette,
    )
    return layers.disc, layers.halo


def _projected_half_extents(spec: BodySpec) -> tuple[float, float]:
    """The half-extents of an ellipsoid's silhouette along the image axes.

    The rendered silhouette is the ellipse of semi-axes a and b rotated in-plane by
    ``rotation_z``; semi-axis c is the depth axis and does not project.

    Parameters:
        spec: The body.

    Returns:
        The half-extents along v and u, in render-grid pixels.
    """
    rotation_z = np.radians(spec.rotation_z)
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    half_extent_v = float(stable_hypot(spec.semi_a_px * cos_rz, spec.semi_b_px * sin_rz))
    half_extent_u = float(stable_hypot(spec.semi_a_px * sin_rz, spec.semi_b_px * cos_rz))
    return half_extent_v, half_extent_u
