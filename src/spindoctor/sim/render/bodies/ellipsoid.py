"""Ellipsoid body renderer: smooth or relief limb, ragged terminator, laws, textures.

This is the image side's one renderer for an ellipsoid body.
:func:`spindoctor.sim.render.bodies.render.render_single_body` sends every ellipsoid
here, whatever its settings: a smooth Lambert disc, a limb-relief field
(:mod:`spindoctor.sim.render.bodies.relief`), a non-Lambert photometric law or
opposition surge (:mod:`spindoctor.sim.render.bodies.photometry`), craters
(:mod:`spindoctor.sim.render.bodies.craters`), and surface textures and transits
(:mod:`spindoctor.sim.render.bodies.texture`).

**Working grid and detector grid.**  The body is drawn on its *working grid*: the
oversampled grid, or at oversample 1 the anti-aliasing supersample grid.  The
silhouette, the smooth ellipsoid's surface normals, and the photometric law are
all evaluated there, so a body is shaded where its outline is drawn.  This matters
most near the limb, where the normal turns as the square root of the distance from
the edge, and at the terminator, where the law clips; shading coarsely and
interpolating would move the light a few thousandths of a pixel away from the sun.
The body is placed on the working grid by
:func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid`, the projection the
navigator's predicted body uses, so a smooth body renders exactly as the predicted
body when the working grid has the same factor.

The relief and the craters are laid out on the *detector grid*, one point per
detector pixel, and interpolated onto the working grid: the tilt the relief gives
the normals by a cubic spline, and the crater slopes bilinearly.  Computing the
relief once per detector pixel instead of once per working pixel keeps a large
body within the render-time budget (see ``tests/integration/test_sim_perf.py``).
Four regions of a relief body are computed on the working grid instead: a band
along the limb, where the relief turns with the smooth normal; the cells along the
relief's own day-night edge, where the law's clip would darken interpolated tilts;
the cells its cast shadows reach, whose sharp edges interpolation would blur; and
the cells where the point a pixel sees jumps behind a ridge or stretches across a
slope seen nearly edge-on (all four in
:mod:`spindoctor.sim.render.bodies.ellipsoid_relief`).  Relief whose correlation
length spans fewer than five detector pixels is too fine for the spline, and such a
body is shaded wholly on its working grid.  Only the rectangle a relief or cratered
body covers is shaded.  Every other body, including one with a surface texture or
transits, whose spots and discs have sharp edges, is shaded wholly on the working
grid.  A surface setting that changes nothing (a
crater fill that places no crater, a texture with no contrast and no spots, or an
empty transit) is dropped first.

**Relief application.**  The renderer's normalized ellipse radial function
``e(p)`` (1 exactly at the unperturbed limb) becomes
``e_adj(p) = e(p) / (1 + delta(theta))``, placing the limb at
``r_ellipse * (1 + delta)``; ``delta`` is the relief surface's own outline,
the envelope ``max over lat of (1 + h) cos(lat) - 1`` along each azimuth.
Each pixel reads the terrain where its line of sight first meets the relief
surface.  The relief tilts every normal by the field's surface gradient, so slopes
facing the sun brighten and slopes facing away darken, and an exact ray march through
the relief shell (both in :mod:`spindoctor.sim.render.bodies.relief`) casts
terrain shadows, which render as night.  Only points whose sun elevation lies
within the field's steepest slope can be shadowed, so the march visits those
alone.

The mesh-body path does not dispatch here: a polyhedral-mesh body applies
the relief field as a per-vertex radial perturbation in
:mod:`spindoctor.sim.render.bodies.mesh`, and the scene validator rejects
the photometric-law, surface-texture, crater, and atmosphere keys on mesh
bodies rather than letting them silently not render.
"""

from dataclasses import KW_ONLY, dataclass, replace
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.craters import (
    CraterSite,
    carve_crater_heights,
    perturb_normals_by_slope,
)
from spindoctor.sim.render.bodies.ellipsoid_grid import (
    CraterPlane,
    bilinear_upsample,
    body_box,
    crater_plane_site,
    silhouette_mask,
)
from spindoctor.sim.render.bodies.ellipsoid_relief import (
    add_detector_relief,
    relief_normals,
    relief_resolved_on_detector_grid,
)
from spindoctor.sim.render.bodies.photometry import incidence_cosines, shade_surface
from spindoctor.sim.render.bodies.relief import ReliefField, synthesize_relief_field
from spindoctor.sim.render.bodies.texture import (
    AlbedoTextureSpec,
    DiscTextureSpec,
    TransitSpec,
    apply_transits,
    surface_texture_factor,
)
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    ellipsoid_image_normals,
    project_ellipsoid,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'EllipsoidBodyRender',
    'EllipsoidBodySpec',
    'create_ellipsoid_body',
    'render_ellipsoid_body',
]


@dataclass(frozen=True)
class EllipsoidBodySpec:
    """Everything the ellipsoid renderer needs beyond size and center.

    All pixel quantities are in units of the incoming render grid (the
    oversampled grid when ``oversample`` > 1).  The three semi-axes may be given
    by position; every other field is keyword-only.

    Parameters:
        semi_a_px: Semi-axis a of the ellipsoid in pixels.
        semi_b_px: Semi-axis b of the ellipsoid in pixels.
        semi_c_px: Semi-axis c of the ellipsoid (the depth axis) in pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        sub_observer_lat: The sub-observer latitude, radians: positive tips the
            north pole, the end of semi-axis a on +v before ``rotation_z``, toward
            the observer.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians (0 fully lit, pi backlit).
        crater_fill: Approximate crater coverage fraction (0 = none).
        crater_min_radius: Minimum crater radius as a fraction of ``semi_a_px``.
        crater_max_radius: Maximum crater radius as a fraction of ``semi_a_px``.
        crater_power_law_exponent: Crater radius power-law exponent.
        crater_relief_scale: Crater depth scale factor.
        anti_aliasing: Anti-aliasing amount in [0, 1]; only consumed at
            ``oversample`` 1, where it supersamples the body by up to 4 (an
            oversampled scene already resolves the limb on its own grid).
        crater_seed: Seed for crater placement, of which the low 31 bits are used.
            None suits only a body that places no crater.
        crater_turn: The planted roll the body was turned by, in radians.  The
            craters are laid out for the body turned back by this angle and then
            turned with it, so a rolled body carries the unrolled body's craters.
        oversample: The incoming grid's oversampling factor.  The body is drawn
            and shaded on its working grid: the incoming grid, or its
            anti-aliasing supersample at oversample 1.  Relief and craters are
            laid out on the detector grid ``size / oversample`` (see
            :func:`render_ellipsoid_body`).
        limb_relief_rms: Commanded limb-slice relief RMS (h/R; 0 = off).
        limb_relief_corr_deg: Relief correlation length, degrees of arc.
        relief_seed: Seed of the relief-field realization.
        photometric_law: One of the photometry module's law names.
        minnaert_k: Minnaert exponent (law 'minnaert' only).
        surge_amplitude: Opposition-surge amplitude (0 = none).
        surge_width_deg: Opposition-surge angular width in degrees.
        albedo_texture: Multiplicative albedo texture (noise field + spots),
            or None.  Applied to the shading, never the silhouette.
        disc_texture: Banded zones/belts plus storm ovals, or None.
        transits: Transiting moon discs and cast shadows (texture on the
            rendered disc).
    """

    semi_a_px: float
    semi_b_px: float
    semi_c_px: float
    _: KW_ONLY
    rotation_z: float = 0.0
    sub_observer_lat: float = 0.0
    illumination_angle: float = 0.0
    phase_angle: float = 0.0
    crater_fill: float = 0.0
    crater_min_radius: float = 0.05
    crater_max_radius: float = 0.25
    crater_power_law_exponent: float = 3.0
    crater_relief_scale: float = 0.6
    anti_aliasing: float = 0.0
    crater_seed: int | None = None
    crater_turn: float = 0.0
    oversample: int = 1
    limb_relief_rms: float = 0.0
    limb_relief_corr_deg: float = 15.0
    relief_seed: int = 0
    photometric_law: str = 'lambert'
    minnaert_k: float = 0.5
    surge_amplitude: float = 0.0
    surge_width_deg: float = 6.0
    albedo_texture: AlbedoTextureSpec | None = None
    disc_texture: DiscTextureSpec | None = None
    transits: tuple[TransitSpec, ...] = ()


@dataclass(frozen=True)
class EllipsoidBodyRender:
    """One ellipsoid body render: its radiance and its solid silhouette.

    Parameters:
        intensity: The intensity image, 0 outside the body and on its night side.
            A limb-brightened law or a bright albedo spot can carry it above 1.
            A surface stage can also leave a lit pixel inside the body at 0, such
            as a transit's full umbra.
        silhouette: The fraction of each pixel the body's solid outline
            covers, in [0, 1], independent of how bright the pixel renders.
    """

    intensity: NDArrayFloatType
    silhouette: NDArrayFloatType


@dataclass(frozen=True)
class _CraterLayout:
    """Where a body's craters are drawn, and how many there are.

    Parameters:
        site: The larger grid the crater centers are drawn on, or None to draw them
            on the detector grid itself.
        count: The number of craters.
    """

    site: CraterSite | None
    count: int


def create_ellipsoid_body(
    size: tuple[int, int],
    center: tuple[float, float],
    spec: EllipsoidBodySpec,
) -> NDArrayFloatType:
    """Render one ellipsoid body onto a black frame.

    Parameters:
        size: Frame dimensions (rows, columns) on the incoming render grid.
        center: Body center (v, u) in incoming-grid pixels.
        spec: The body's geometry, texture, relief, and photometry.

    Returns:
        The ``size`` intensity image (0 outside the body); the intensity of
        :func:`render_ellipsoid_body`.

    Raises:
        ValueError: If the frame size is not divisible by the oversample, or if the
            spec places a crater but has no crater seed.
    """
    return render_ellipsoid_body(size, center, spec).intensity


def render_ellipsoid_body(
    size: tuple[int, int],
    center: tuple[float, float],
    spec: EllipsoidBodySpec,
    *,
    crater_plane: CraterPlane | None = None,
) -> EllipsoidBodyRender:
    """Render one ellipsoid body and its solid silhouette.

    The silhouette is the relief-perturbed outline's pixel coverage, taken
    before any shading, so a pixel that renders black inside the body (the night
    side, a transit umbra, or a zero-albedo spot) still belongs to it.

    Parameters:
        size: Frame dimensions (rows, columns) on the incoming render grid.
        center: Body center (v, u) in incoming-grid pixels.
        spec: The body's geometry, texture, relief, and photometry.
        crater_plane: The detector grid the craters are laid out on when it is not
            this frame's own: a larger grid holding the whole body when the frame
            holds only part of it, or, for a body turned by a planted roll, the
            grid the unrolled body lays its craters out on, which can be smaller
            than this frame.  None lays them out on the frame's own detector grid.

    Returns:
        The :class:`EllipsoidBodyRender` on the ``size`` grid.

    Raises:
        ValueError: If the frame size is not divisible by the oversample, or if the
            spec places a crater but has no crater seed.
    """
    size_v, size_u = size
    os_in = max(1, int(spec.oversample))
    if size_v % os_in or size_u % os_in:
        raise ValueError(f'frame size {size} is not divisible by oversample {os_in}')

    field: ReliefField | None = None
    if spec.limb_relief_rms > 0.0:
        field = synthesize_relief_field(
            spec.limb_relief_rms, spec.limb_relief_corr_deg, spec.relief_seed
        )
        if field.h_max <= field.h_min:
            field = None

    # At oversample 1 the anti-aliasing knob supersamples the body; an oversampled
    # scene already resolves the limb, so its work grid IS the render grid.
    work_grid = project_ellipsoid(
        size,
        center,
        spec.semi_a_px,
        semi_b_px=spec.semi_b_px,
        semi_c_px=spec.semi_c_px,
        rotation_z=spec.rotation_z,
        sub_observer_lat=spec.sub_observer_lat,
        anti_aliasing=spec.anti_aliasing if os_in == 1 else 0.0,
    )
    aa_scale = work_grid.aa_scale
    up_factor = os_in * aa_scale
    # The detector grid is where craters are drawn and where surface detail is laid
    # out.
    detector_grid = work_grid
    shading_spec = spec
    if up_factor > 1:
        detector_grid = project_ellipsoid(
            (size_v // os_in, size_u // os_in),
            (center[0] / os_in, center[1] / os_in),
            spec.semi_a_px / os_in,
            semi_b_px=spec.semi_b_px / os_in,
            semi_c_px=spec.semi_c_px / os_in,
            rotation_z=spec.rotation_z,
            sub_observer_lat=spec.sub_observer_lat,
        )
    craters = _crater_layout(detector_grid, spec, crater_plane)
    if up_factor > 1:
        # A surface setting that changes nothing (a crater fill that places no
        # crater on the detector grid, a texture with no contrast and no spots, a
        # transit with neither a moon nor a shadow) is dropped from the spec the
        # body is shaded with, so it renders exactly as the smooth body.
        shading_spec = _without_inert_surface(spec, crater_count=craters.count)

    mask = silhouette_mask(work_grid, field)
    if up_factor > 1 and _lays_detail_on_detector_grid(shading_spec, field):
        intensity_work = _shade_disc_with_detector_detail(
            work_grid,
            detector_grid,
            shading_spec,
            field,
            up_factor=up_factor,
            craters=craters,
            inside=mask > 0.0,
        )
    else:
        intensity_work = _shade_disc(
            work_grid,
            shading_spec,
            field,
            detector_grid=detector_grid,
            scale=up_factor,
            craters=craters,
            inside=mask > 0.0,
        )
    intensity_work *= mask

    if aa_scale > 1:
        intensity_work = intensity_work.reshape(size_v, aa_scale, size_u, aa_scale).mean(
            axis=(1, 3)
        )
        mask = mask.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
    return EllipsoidBodyRender(intensity=intensity_work, silhouette=mask)


def _lays_detail_on_detector_grid(spec: EllipsoidBodySpec, field: ReliefField | None) -> bool:
    """Whether a supersampled body's relief or craters are laid out on the detector grid.

    The relief tilt and the crater slopes are computed on the detector grid and
    interpolated onto the working grid, which keeps a large body's render within
    its time budget.  Any other body is shaded wholly on its working grid.

    Parameters:
        spec: The body spec, with its inert surface settings already dropped.
        field: The relief realization, or None for the smooth surface.

    Returns:
        True when the body carries relief or craters.
    """
    return field is not None or spec.crater_fill > 0.0


def _crater_slopes(
    spec: EllipsoidBodySpec,
    *,
    detector_grid: EllipsoidProjection,
    scale: int,
    craters: _CraterLayout,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """The slopes of the body's crater field on a grid ``scale`` times the detector's.

    The craters are carved on the detector grid, where their slopes are taken by
    central differences.  On a finer grid those slopes are interpolated bilinearly,
    so every grid shades the same cratered surface, with normals that turn
    continuously.

    Parameters:
        spec: The body spec (crater keys are read).
        detector_grid: The detector-grid geometry.
        scale: Pixels of the grid evaluated on per detector pixel.
        craters: Where the craters are drawn and how many there are.

    Returns:
        ``(slope_v, slope_u)`` on that grid, height per unit of length.
    """
    heights = _crater_heights(detector_grid, spec, craters)
    # A frame one detector pixel tall or wide has no slope across that axis.
    slope_v, slope_u = (
        np.gradient(heights, axis=axis) if heights.shape[axis] > 1 else np.zeros_like(heights)
        for axis in (0, 1)
    )
    return (
        bilinear_upsample(np.asarray(slope_v, dtype=np.float64), scale),
        bilinear_upsample(np.asarray(slope_u, dtype=np.float64), scale),
    )


def _apply_law(
    cos_i: NDArrayFloatType, normal_z: NDArrayFloatType, spec: EllipsoidBodySpec
) -> NDArrayFloatType:
    """Shade incidence cosines under the spec's photometric law and surge.

    The clamped-depth parametrization has no far hemisphere, and relief normals
    face the observer, some of them only just (``normal_z >= 0`` everywhere; night
    is still black through ``mu0 <= 0``), so the emission cosine is kept strictly
    positive: the visibility gate must not zero a relief normal that lies at the
    sky plane.  A smooth body's normals lie in the sky plane only on and past its
    limb, which its silhouette leaves out, so the clamp lights none of its pixels.

    Parameters:
        cos_i: Incidence cosine per pixel (negative on the night side).
        normal_z: The normal's toward-observer component per pixel.
        spec: The body spec (law, Minnaert exponent, and surge are read).

    Returns:
        The brightness per pixel.
    """
    return shade_surface(
        cos_i,
        np.maximum(normal_z, 1e-12),
        law=spec.photometric_law,
        phase_angle=spec.phase_angle,
        minnaert_k=spec.minnaert_k,
        surge_amplitude=spec.surge_amplitude,
        surge_width_deg=spec.surge_width_deg,
    )


def _shade_disc(
    grid: EllipsoidProjection,
    spec: EllipsoidBodySpec,
    field: ReliefField | None,
    *,
    detector_grid: EllipsoidProjection,
    scale: int,
    craters: _CraterLayout,
    inside: NDArrayBoolType,
) -> NDArrayFloatType:
    """Shade the disc on one grid, with all its surface detail evaluated there.

    Normals are computed everywhere with the depth clamped to the visible
    hemisphere, so pixels just outside the limb carry the limb's shading values.
    With relief the normals are the relief surface's and the terrain casts
    shadows (:func:`~spindoctor.sim.render.bodies.ellipsoid_relief.relief_shading`),
    which render as night.

    Parameters:
        grid: The geometry to shade on.
        spec: The body spec (crater texture and photometric law are read).
        field: The relief realization, or None for the smooth surface; it sets
            the outline the transits reach.
        detector_grid: The detector-grid geometry the craters are carved on (``grid``
            itself when ``scale`` is 1).
        scale: Pixels of ``grid`` per detector pixel, the unit of texture and
            transit lengths.
        craters: Where the craters are drawn and how many there are.
        inside: The grid's pixels inside the body's silhouette.  The relief is
            computed at those pixels only, since every other pixel is masked off.

    Returns:
        The intensity on the grid.
    """
    normals = ellipsoid_image_normals(grid)
    shadowed: NDArrayBoolType | None = None
    if field is not None:
        shadowed = relief_normals(
            grid,
            field,
            normals,
            illumination_angle=spec.illumination_angle,
            phase_angle=spec.phase_angle,
            inside=inside,
        ).shadowed
    if spec.crater_fill > 0.0:
        normals = perturb_normals_by_slope(
            *normals,
            *_crater_slopes(spec, detector_grid=detector_grid, scale=scale, craters=craters),
        )
    normal_v, normal_u, normal_z = normals
    cos_i = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
    )
    if shadowed is not None:
        # A cast shadow is night: the law renders it black like the far side of
        # the terminator.
        cos_i = np.where(shadowed, -1.0, cos_i)
    intensity = _apply_law(cos_i, normal_z, spec)
    factor = _texture_factor(grid, spec, scale=scale)
    return _apply_surface_textures(intensity, grid, spec, field, factor=factor, scale=scale)


def _shade_disc_with_detector_detail(
    work_grid: EllipsoidProjection,
    detector_grid: EllipsoidProjection,
    spec: EllipsoidBodySpec,
    field: ReliefField | None,
    *,
    up_factor: int,
    craters: _CraterLayout,
    inside: NDArrayBoolType,
) -> NDArrayFloatType:
    """Shade the disc on the working grid, with relief and craters from the detector grid.

    The smooth ellipsoid's normals and the photometric law are evaluated on the
    working grid, where they are exact: near the limb the normal turns as the square
    root of the distance from the edge, which no interpolation follows.  The relief
    and the craters are computed on the detector grid and interpolated onto the
    working grid: the tilt the relief gives each normal by a cubic spline, and the
    crater slopes bilinearly.
    :func:`~spindoctor.sim.render.bodies.ellipsoid_relief.add_detector_relief` lists
    the regions where the relief's normals and shadows are computed on the working
    grid too.  Relief too fine for the detector grid is computed wholly on the working
    grid.  Textures and transits are drawn on the
    working grid.  Only the rectangle the body covers is shaded.

    Parameters:
        work_grid: The working-grid geometry.
        detector_grid: The detector-grid geometry.
        spec: The body spec.
        field: The relief realization, or None for the smooth surface.
        up_factor: Pixels of the working grid per detector pixel.
        craters: Where the craters are drawn and how many there are.
        inside: The working grid's pixels inside the body's silhouette.  The relief
            is computed on the working grid at those pixels only, since every other
            pixel is masked off.

    Returns:
        The intensity on the working grid.
    """
    full_shape = work_grid.ellipse_dist_sq.shape
    box = body_box(detector_grid, field)
    if box is None:
        return np.zeros(full_shape)
    rows, cols = box
    work_rows = slice(rows.start * up_factor, rows.stop * up_factor)
    work_cols = slice(cols.start * up_factor, cols.stop * up_factor)
    crater_slopes: tuple[NDArrayFloatType, NDArrayFloatType] | None = None
    if spec.crater_fill > 0.0:
        # The craters are drawn on the whole detector grid, whose dimensions their
        # draw depends on, and cropped afterward.
        slope_v, slope_u = _crater_slopes(
            spec, detector_grid=detector_grid, scale=1, craters=craters
        )
        crater_slopes = (slope_v[rows, cols], slope_u[rows, cols])
    detector_grid = detector_grid.crop(rows, cols)
    work_grid = work_grid.crop(work_rows, work_cols)
    inside = inside[work_rows, work_cols]

    normal_v, normal_u, normal_z = ellipsoid_image_normals(work_grid)
    band_shadowed: NDArrayBoolType | None = None
    if field is not None and not relief_resolved_on_detector_grid(field, detector_grid):
        band_shadowed = relief_normals(
            work_grid,
            field,
            (normal_v, normal_u, normal_z),
            illumination_angle=spec.illumination_angle,
            phase_angle=spec.phase_angle,
            inside=inside,
        ).shadowed
    elif field is not None:
        band_shadowed = add_detector_relief(
            work_grid,
            detector_grid,
            field,
            (normal_v, normal_u, normal_z),
            up_factor=up_factor,
            illumination_angle=spec.illumination_angle,
            phase_angle=spec.phase_angle,
            inside=inside,
        )
    if crater_slopes is not None:
        normal_v, normal_u, normal_z = perturb_normals_by_slope(
            normal_v,
            normal_u,
            normal_z,
            bilinear_upsample(crater_slopes[0], up_factor),
            bilinear_upsample(crater_slopes[1], up_factor),
        )
    cos_i = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
    )
    if band_shadowed is not None:
        cos_i = np.where(band_shadowed, -1.0, cos_i)
    intensity = _apply_law(cos_i, normal_z, spec)
    # The texture factor is cheap to evaluate, and its spots and storms have sharp
    # edges, so it is evaluated on the working grid, where a sampled edge cannot
    # move the light.
    factor = _texture_factor(work_grid, spec, scale=up_factor)
    full = np.zeros(full_shape)
    full[work_rows, work_cols] = _apply_surface_textures(
        intensity, work_grid, spec, field, factor=factor, scale=up_factor
    )
    return full


def _texture_factor(
    grid: EllipsoidProjection, spec: EllipsoidBodySpec, *, scale: int
) -> NDArrayFloatType | None:
    """The spec's multiplicative surface-texture factor on one grid.

    Parameters:
        grid: The geometry to evaluate on.
        spec: The body spec (texture fields are read).
        scale: Pixels of ``grid`` per detector pixel, the unit of the albedo
            correlation length.

    Returns:
        The factor per pixel, or None when the spec has no albedo or disc texture.
    """
    if spec.albedo_texture is None and spec.disc_texture is None:
        return None
    x, y, z = grid.surface_unit_coords()
    return surface_texture_factor(
        spec.albedo_texture,
        spec.disc_texture,
        x=x,
        y=y,
        z=z,
        mean_radius_px=(grid.semi_a + grid.semi_b) / 2.0 / scale,
    )


def _apply_surface_textures(
    intensity: NDArrayFloatType,
    grid: EllipsoidProjection,
    spec: EllipsoidBodySpec,
    field: ReliefField | None,
    *,
    factor: NDArrayFloatType | None,
    scale: int,
) -> NDArrayFloatType:
    """Apply a texture factor and the spec's transits to the shading.

    Texture layering: the albedo/disc texture factor multiplies the law
    shading first, then transit shadows multiply the textured result, then
    transiting moon discs overwrite on top (from the pre-texture smooth-law
    shading -- a foreground moon does not inherit the parent's texture).
    The silhouette is rasterized separately, so it is untouched by
    construction.  Transits reach every sample of the body's silhouette: the
    relief outline when the body has relief, else the smooth limb.

    Parameters:
        intensity: The law-shaded intensity on ``grid``.
        grid: The geometry the intensity lies on.
        spec: The body spec (transit fields are read).
        field: The relief realization, or None for the smooth limb.
        factor: The texture factor on ``grid``, or None for no texture.
        scale: Pixels of ``grid`` per detector pixel, the unit of transit lengths.

    Returns:
        The textured intensity (the input array, unchanged, when there is no
        texture and no transit).
    """
    if factor is None and not spec.transits:
        return intensity
    textured = intensity if factor is None else intensity * factor
    if spec.transits:
        if textured is intensity:
            textured = intensity.copy()
        apply_transits(
            textured,
            spec.transits,
            base_intensity=intensity,
            v_ctr=grid.v_coords / scale,
            u_ctr=grid.u_coords / scale,
            disc_mask=silhouette_mask(grid, field) > 0.0,
        )
    return textured


def _without_inert_surface(spec: EllipsoidBodySpec, *, crater_count: int) -> EllipsoidBodySpec:
    """The spec with the recognized no-op surface settings removed.

    A crater fill is inert when it places no crater on the detector grid, the grid the
    craters are drawn on; an albedo texture when its noise field is off (zero RMS or
    zero correlation length) and it has no spots; a disc texture when its bands have
    zero amplitude and it has no storms; and a transit when it has neither a moon nor
    a shadow.

    Parameters:
        spec: The body spec.
        crater_count: The number of craters the crater fill places.

    Returns:
        The spec with the inert settings cleared (``spec`` itself when none is inert).
    """
    changes: dict[str, Any] = {}
    if spec.crater_fill > 0.0 and crater_count == 0:
        changes['crater_fill'] = 0.0
    albedo = spec.albedo_texture
    if albedo is not None and not (albedo.rms > 0.0 and albedo.corr_px > 0.0) and not albedo.spots:
        changes['albedo_texture'] = None
    disc = spec.disc_texture
    if disc is not None and disc.band_amplitude <= 0.0 and not disc.storms:
        changes['disc_texture'] = None
    transits = tuple(t for t in spec.transits if t.moon is not None or t.shadow is not None)
    if len(transits) != len(spec.transits):
        changes['transits'] = transits
    return replace(spec, **changes) if changes else spec


def _crater_layout(
    grid: EllipsoidProjection, spec: EllipsoidBodySpec, plane: CraterPlane | None
) -> _CraterLayout:
    """Where the spec's craters are drawn, and how many its crater fill places.

    The count is the crater fill times the number of pixels inside the body, over
    the area of a crater of the largest radius, between 0 and 1000.  The pixels are
    those of the detector grid, or of the crater plane when the grid holds only part
    of the body.  A body turned by a planted roll counts and draws among the pixels
    of the unrolled body's plane inside the unrolled body.

    Parameters:
        grid: The detector-grid geometry the craters are carved on.
        spec: The body spec (crater fill, maximum radius, orientation, tilt, and
            turn are read).
        plane: The larger grid the craters are laid out on, or None for ``grid``.

    Returns:
        The crater layout; a count of 0 for a body without craters.
    """
    if spec.crater_fill <= 0.0:
        return _CraterLayout(site=None, count=0)
    site: CraterSite | None = None
    draw_grid = grid
    if spec.crater_turn != 0.0:
        rows, cols = grid.ellipse_dist_sq.shape
        draw_grid = project_ellipsoid(
            (rows, cols),
            grid.center,
            grid.semi_a,
            semi_b_px=grid.semi_b,
            semi_c_px=grid.semi_c,
            rotation_z=spec.rotation_z + spec.crater_turn,
            sub_observer_lat=spec.sub_observer_lat,
        )
        plane = plane or CraterPlane(shape=(rows, cols), origin=(0, 0))
    if plane is None:
        total_area = int(np.sum(grid.ellipse_dist_sq < 1.0))
    else:
        site, total_area = crater_plane_site(plane, draw_grid)
    avg_crater_area = (spec.crater_max_radius * grid.semi_a) ** 2
    count = int(np.clip(int(spec.crater_fill * total_area / avg_crater_area), 0, 1000))
    return _CraterLayout(site=site, count=count)


def _crater_heights(
    grid: EllipsoidProjection, spec: EllipsoidBodySpec, craters: _CraterLayout
) -> NDArrayFloatType:
    """Carve the crater height field on the detector grid.

    Uses the crater carving loop (count law, radius power law, and per-crater draw
    rule), seeded by the per-body crater seed.

    Parameters:
        grid: The detector-grid geometry.
        spec: The body spec (crater knobs and seed are read).
        craters: Where the craters are drawn and how many there are.

    Returns:
        The crater height field over the grid.

    Raises:
        ValueError: If the spec places a crater on the grid but has no crater seed.
    """
    inside_strict = grid.ellipse_dist_sq < 1.0
    n_craters = craters.count
    if n_craters == 0 or not inside_strict.any():
        return np.zeros_like(grid.ellipse_dist_sq)

    if spec.crater_seed is None:
        raise ValueError('a body that places craters needs a crater seed')
    return carve_crater_heights(
        inside_strict,
        grid.v_coords,
        grid.u_coords,
        crater_seed=int(spec.crater_seed) & 0x7FFFFFFF,
        n_craters=n_craters,
        R_min=spec.crater_min_radius * grid.semi_a,
        R_max=spec.crater_max_radius * grid.semi_a,
        crater_power_law_exponent=spec.crater_power_law_exponent,
        crater_relief_scale=spec.crater_relief_scale,
        work_center_v=grid.center[0],
        work_center_u=grid.center[1],
        site=craters.site,
        turn=spec.crater_turn,
    )
