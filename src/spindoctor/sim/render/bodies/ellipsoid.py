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
Three regions of a relief body are computed on the working grid instead: a band
along the limb, where the relief turns with the smooth normal; the cells along the
relief's own day-night edge, where the law's clip would darken interpolated tilts;
and the cells its cast shadows reach, whose sharp edges interpolation would blur.
Only the rectangle such a body covers is shaded.  Every other body, including one
with a surface texture or transits, whose spots and discs have sharp edges, is
shaded wholly on the working grid.  A surface setting that changes nothing (a
crater fill that places no crater, a texture with no contrast and no spots, or an
empty transit) is dropped first.

**Relief application.**  The renderer's normalized ellipse radial function
``e(p)`` (1 exactly at the unperturbed limb) becomes
``e_adj(p) = e(p) / (1 + delta(theta))``, placing the limb at
``r_ellipse * (1 + delta)``; ``delta`` is the relief surface's own outline,
the envelope ``max over lat of (1 + h) cos(lat) - 1`` along each azimuth.
The relief tilts every normal by the field's surface gradient, so slopes facing
the sun brighten and slopes facing away darken, and an exact ray march through
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

from spindoctor.sim.render.bodies.craters import carve_crater_heights, perturb_normals_by_slope
from spindoctor.sim.render.bodies.ellipsoid_grid import (
    bilinear_upsample,
    body_box,
    cubic_upsample,
    day_night_edge_cells,
    expand_to_working_grid,
    limb_band,
    neighboring_cells,
    silhouette_mask,
)
from spindoctor.sim.render.bodies.photometry import incidence_cosines, shade_surface
from spindoctor.sim.render.bodies.relief import ReliefField, cast_shadows, synthesize_relief_field
from spindoctor.sim.render.bodies.texture import (
    AlbedoTextureSpec,
    DiscTextureSpec,
    TransitSpec,
    apply_transits,
    surface_texture_factor,
)
from spindoctor.sim.seeds import stable_param_seed
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    ellipsoid_image_normals,
    project_ellipsoid,
)
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.stable_math import stable_matmul3
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'EllipsoidBodyRender',
    'EllipsoidBodySpec',
    'create_ellipsoid_body',
    'render_ellipsoid_body',
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
        rotation_tilt: Tilt toward/away from the viewer, radians.
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
        crater_seed: Seed for crater placement (None with no craters).
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
    rotation_tilt: float = 0.0
    illumination_angle: float = 0.0
    phase_angle: float = 0.0
    crater_fill: float = 0.0
    crater_min_radius: float = 0.05
    crater_max_radius: float = 0.25
    crater_power_law_exponent: float = 3.0
    crater_relief_scale: float = 0.6
    anti_aliasing: float = 0.0
    crater_seed: int | None = None
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
        intensity: The intensity image in [0, 1], 0 outside the body.  A
            surface stage can also leave a pixel inside the body at 0, such
            as a transit's full umbra.
        silhouette: The fraction of each pixel the body's solid outline
            covers, in [0, 1], independent of how bright the pixel renders.
    """

    intensity: NDArrayFloatType
    silhouette: NDArrayFloatType


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
        The ``size`` intensity image in [0, 1] (0 outside the body); the
        intensity of :func:`render_ellipsoid_body`.

    Raises:
        ValueError: If the frame size is not divisible by the oversample.
    """
    return render_ellipsoid_body(size, center, spec).intensity


def render_ellipsoid_body(
    size: tuple[int, int],
    center: tuple[float, float],
    spec: EllipsoidBodySpec,
) -> EllipsoidBodyRender:
    """Render one ellipsoid body and its solid silhouette.

    The silhouette is the relief-perturbed outline's pixel coverage, taken
    before any shading, so a pixel that renders black inside the body (a
    transit umbra, a zero-albedo spot) still belongs to it.

    Parameters:
        size: Frame dimensions (rows, columns) on the incoming render grid.
        center: Body center (v, u) in incoming-grid pixels.
        spec: The body's geometry, texture, relief, and photometry.

    Returns:
        The :class:`EllipsoidBodyRender` on the ``size`` grid.

    Raises:
        ValueError: If the frame size is not divisible by the oversample.
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
        rotation_tilt=spec.rotation_tilt,
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
            rotation_tilt=spec.rotation_tilt,
        )
        # A surface setting that changes nothing (a crater fill that places no
        # crater on the detector grid, a texture with no contrast and no spots, a
        # transit with neither a moon nor a shadow) is dropped from the spec the
        # body is shaded with, so it renders exactly as the smooth body.
        shading_spec = _without_inert_surface(spec, detector_grid)

    if up_factor > 1 and _lays_detail_on_detector_grid(shading_spec, field):
        intensity_work = _shade_disc_with_detector_detail(
            work_grid,
            detector_grid,
            shading_spec,
            field,
            up_factor=up_factor,
        )
    else:
        intensity_work = _shade_disc(
            work_grid,
            shading_spec,
            field,
            detector_grid=detector_grid,
            scale=up_factor,
        )

    mask = silhouette_mask(work_grid, field)
    intensity_work *= mask

    if aa_scale > 1:
        intensity_work = intensity_work.reshape(size_v, aa_scale, size_u, aa_scale).mean(
            axis=(1, 3)
        )
        mask = mask.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
    return EllipsoidBodyRender(intensity=np.clip(intensity_work, 0.0, 1.0), silhouette=mask)


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


def _relief_normals(
    grid: EllipsoidProjection,
    spec: EllipsoidBodySpec,
    field: ReliefField,
    normals: tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType],
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Replace the smooth normals with the relief surface's, and cast its shadows.

    Parameters:
        grid: The geometry to evaluate on.
        spec: The body spec (illumination geometry is read).
        field: The relief realization.
        normals: The smooth normals on ``grid``; the arrays are updated in place
            where the relief reaches.

    Returns:
        ``(normal_v, normal_u, normal_z, shadowed)``: the relief normals and the
        relief's cast-shadow mask on ``grid``.
    """
    normal_v, normal_u, normal_z = normals
    # Relief reaches only pixels inside its outline, plus the pixel an
    # interpolation reads beyond it; farther out the smooth limb normals carry
    # the continuity and the silhouette masks them.
    reach = 1.0 + max(field.h_max, 0.0) + 2.0 / max(min(grid.semi_a, grid.semi_b), 1.0)
    near = grid.ellipse_dist_sq < reach * reach
    relief_v, relief_u, relief_z, shadowed = _relief_shading(grid, spec, field, near)
    normal_v[near] = relief_v
    normal_u[near] = relief_u
    normal_z[near] = relief_z
    return normal_v, normal_u, normal_z, shadowed


def _crater_slopes(
    spec: EllipsoidBodySpec, *, detector_grid: EllipsoidProjection, scale: int
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

    Returns:
        ``(slope_v, slope_u)`` on that grid, height per unit of length.
    """
    heights = _crater_heights(detector_grid, spec)
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
    are mirrored onto the near one (``normal_z >= 0`` everywhere; night still floors
    through ``mu0 <= 0``), so the emission cosine is kept strictly positive: the
    visibility gate must not zero a relief normal that lies in the sky plane.  A
    smooth body's normals lie in the sky plane only on and past its limb, which
    its silhouette leaves out, so the clamp lights none of its pixels.

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
) -> NDArrayFloatType:
    """Shade the disc on one grid, with all its surface detail evaluated there.

    Normals are computed everywhere with the depth clamped to the visible
    hemisphere, so pixels just outside the limb carry the limb's shading values.
    With relief the normals are the relief surface's and the terrain casts
    shadows (:func:`_relief_shading`), which render as night.

    Parameters:
        grid: The geometry to shade on.
        spec: The body spec (crater texture and photometric law are read).
        field: The relief realization, or None for the smooth surface; it sets
            the outline the transits reach.
        detector_grid: The detector-grid geometry the craters are carved on (``grid``
            itself when ``scale`` is 1).
        scale: Pixels of ``grid`` per detector pixel, the unit of texture and
            transit lengths.

    Returns:
        The intensity on the grid.
    """
    normals = ellipsoid_image_normals(grid)
    shadowed: NDArrayBoolType | None = None
    if field is not None:
        *relief, shadowed = _relief_normals(grid, spec, field, normals)
        normals = (relief[0], relief[1], relief[2])
    if spec.crater_fill > 0.0:
        normals = perturb_normals_by_slope(
            *normals, *_crater_slopes(spec, detector_grid=detector_grid, scale=scale)
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
        # A cast shadow is night: the law floors it like the far side of the
        # terminator.
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
) -> NDArrayFloatType:
    """Shade the disc on the working grid, with relief and craters from the detector grid.

    The smooth ellipsoid's normals and the photometric law are evaluated on the
    working grid, where they are exact: near the limb the normal turns as the square
    root of the distance from the edge, which no interpolation follows.  The relief
    and the craters are computed on the detector grid and interpolated onto the
    working grid: the tilt the relief gives each normal by a cubic spline, and the
    crater slopes bilinearly.  Within :data:`_LIMB_BAND_PX` detector pixels of the
    limb, along the relief's own day-night edge, and in every cell its cast shadows
    reach, the relief's normals and shadows are computed on the working grid too.
    Textures and transits are drawn on the working grid.  Only the rectangle the
    body covers is shaded.

    Parameters:
        work_grid: The working-grid geometry.
        detector_grid: The detector-grid geometry.
        spec: The body spec.
        field: The relief realization, or None for the smooth surface.
        up_factor: Pixels of the working grid per detector pixel.

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
        slope_v, slope_u = _crater_slopes(spec, detector_grid=detector_grid, scale=1)
        crater_slopes = (slope_v[rows, cols], slope_u[rows, cols])
    detector_grid = detector_grid.crop(rows, cols)
    work_grid = work_grid.crop(work_rows, work_cols)

    normal_v, normal_u, normal_z = ellipsoid_image_normals(work_grid)
    band_shadowed: NDArrayBoolType | None = None
    if field is not None:
        base_v, base_u, base_z = ellipsoid_image_normals(detector_grid)
        relief_v, relief_u, relief_z, detector_shadowed = _relief_normals(
            detector_grid,
            spec,
            field,
            (base_v.copy(), base_u.copy(), base_z.copy()),
        )
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
            illumination_angle=spec.illumination_angle,
            phase_angle=spec.phase_angle,
        )
        # A cast shadow has a sharp edge, which interpolation would blur, so every
        # cell a cast shadow darkens is shaded on the working grid.  Only a pixel
        # facing the sun is darkened by a cast shadow; one facing away is night by
        # its own normal.
        band = limb_band(work_grid, field, width_px=_LIMB_BAND_PX * up_factor)
        band |= expand_to_working_grid(day_night_edge_cells(relief_cos_i > 0.0), up_factor)
        band |= expand_to_working_grid(
            neighboring_cells(detector_shadowed & (relief_cos_i > 0.0)), up_factor
        )
        band_v, band_u, band_z, band_shadowed = _relief_shading(work_grid, spec, field, band)
        normal_v[band] = band_v
        normal_u[band] = band_u
        normal_z[band] = band_z
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


def _relief_shading(
    grid: EllipsoidProjection,
    spec: EllipsoidBodySpec,
    field: ReliefField,
    near: NDArrayBoolType,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """The relief surface's image-frame normals and cast shadows at the ``near`` pixels.

    Each shading pixel reads the terrain at its smooth-surface direction
    ``p`` in the body's parameter space (the smooth body is the unit sphere,
    its poles on the observer axis; a pixel past the smooth limb reads the
    limb direction).  The relief surface ``(1 + h) p`` has the parameter-space
    normal ``p - grad h / (1 + h)``, carried to the body frame by the
    ellipsoid's inverse semi-axes and rotated into the image like the smooth
    normals.  The sun direction goes to parameter space by the same affine
    map, where :func:`~spindoctor.sim.render.bodies.relief.cast_shadows` marches the
    pixels whose sun elevation lies within the field's slope band; a pixel
    below the band always lies in shadow.  A normal that faces away from the
    observer is mirrored through the sky plane, so every returned
    ``normal_z`` is at least 0.

    Parameters:
        grid: The shading-resolution geometry.
        spec: The body spec (illumination geometry is read).
        field: The relief realization.
        near: The shading pixels the relief can reach.

    Returns:
        ``(normal_v, normal_u, normal_z)`` at the ``near`` pixels, in their
        row-major order, and the shading-grid ``shadowed`` mask.
    """
    semi = np.array([grid.semi_a, grid.semi_b, grid.semi_c], dtype=np.float64)
    cos_rz = grid.cos_rz
    sin_rz = grid.sin_rz
    e2 = grid.ellipse_dist_sq[near]
    x = grid.v_rot[near] / grid.semi_a
    y = grid.u_rot[near] / grid.semi_b
    radial = np.sqrt(np.maximum(e2, 1.0))
    p = np.stack([x / radial, y / radial, np.sqrt(np.maximum(0.0, 1.0 - e2))], axis=-1)
    h = field.sample_points(p)
    normal_param = p - field.surface_gradient(p) / (1.0 + h)[..., np.newaxis]
    normal_body = normal_param / semi
    normal_body /= np.linalg.norm(normal_body, axis=-1, keepdims=True)
    normal_v = normal_body[..., 0] * cos_rz + normal_body[..., 1] * sin_rz
    normal_u = -normal_body[..., 0] * sin_rz + normal_body[..., 1] * cos_rz
    # Near the limb a pixel's smooth-surface direction can land on a slope tilted
    # away from the observer, which the line of sight can't see: it meets the near
    # side of the relief first.  Mirroring the normal through the sky plane stands
    # in for that slope, as steep but facing the observer, so the law never shades
    # a back-facing facet as lit.
    normal_z = np.abs(normal_body[..., 2])

    # The sun in the body frame (the inverse of the normals' in-plane
    # rotation), then in parameter space, where directions scale by 1 / semi.
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=spec.illumination_angle, phase_angle=spec.phase_angle
    )
    sun_body = np.array(
        [
            illum_v * cos_rz - illum_u * sin_rz,
            illum_v * sin_rz + illum_u * cos_rz,
            illum_z,
        ]
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
    return normal_v, normal_u, normal_z, shadowed


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
    x = grid.v_rot / grid.semi_a
    y = grid.u_rot / grid.semi_b
    z = np.sqrt(np.maximum(0.0, 1.0 - grid.ellipse_dist_sq))
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
    textured = intensity if factor is None else np.clip(intensity * factor, 0.0, 1.0)
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


def _without_inert_surface(
    spec: EllipsoidBodySpec, detector_grid: EllipsoidProjection
) -> EllipsoidBodySpec:
    """The spec with the recognized no-op surface settings removed.

    A crater fill is inert when it places no crater on the detector grid, the grid the
    craters are drawn on; an albedo texture when its noise field is off (zero RMS or
    zero correlation length) and it has no spots; a disc texture when its bands have
    zero amplitude and it has no storms; and a transit when it has neither a moon nor
    a shadow.

    Parameters:
        spec: The body spec.
        detector_grid: The detector-grid geometry the crater count is taken on.

    Returns:
        The spec with the inert settings cleared (``spec`` itself when none is inert).
    """
    changes: dict[str, Any] = {}
    if spec.crater_fill > 0.0 and _crater_count(detector_grid, spec) == 0:
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


def _crater_count(grid: EllipsoidProjection, spec: EllipsoidBodySpec) -> int:
    """The number of craters the spec's crater fill places on a grid.

    Parameters:
        grid: The geometry the craters would be carved on.
        spec: The body spec (crater fill and maximum radius are read).

    Returns:
        The crater count, between 0 and 1000.
    """
    total_area = int(np.sum(grid.ellipse_dist_sq < 1.0))
    avg_crater_area = (spec.crater_max_radius * grid.semi_a) ** 2
    return int(np.clip(int(spec.crater_fill * total_area / avg_crater_area), 0, 1000))


def _crater_heights(grid: EllipsoidProjection, spec: EllipsoidBodySpec) -> NDArrayFloatType:
    """Carve the crater height field on the detector grid.

    Uses the crater carving loop (count law, radius power law, and per-crater draw
    rule), seeded by the per-body crater seed.

    Parameters:
        grid: The detector-grid geometry.
        spec: The body spec (crater knobs and seed are read).

    Returns:
        The crater height field over the grid.
    """
    inside_strict = grid.ellipse_dist_sq < 1.0
    n_craters = _crater_count(grid, spec)
    if n_craters == 0 or not inside_strict.any():
        return np.zeros_like(grid.ellipse_dist_sq)

    if spec.crater_seed is not None:
        seed_value = int(spec.crater_seed) & 0x7FFFFFFF
    else:
        # The fallback seed is keyed on the full widths, twice the semi-axes; the
        # craters of the committed seed-less scenes depend on this key (#913).
        widths = (2.0 * spec.semi_a_px, 2.0 * spec.semi_b_px, 2.0 * spec.semi_c_px)
        seed_value = stable_param_seed(*widths, grid.center)
        seed_value &= 0x7FFFFFFF
    return carve_crater_heights(
        inside_strict,
        grid.v_coords,
        grid.u_coords,
        crater_seed=seed_value,
        n_craters=n_craters,
        R_min=spec.crater_min_radius * grid.semi_a,
        R_max=spec.crater_max_radius * grid.semi_a,
        crater_power_law_exponent=spec.crater_power_law_exponent,
        crater_relief_scale=spec.crater_relief_scale,
        work_center_v=grid.center[0],
        work_center_u=grid.center[1],
    )
