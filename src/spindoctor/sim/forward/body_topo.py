"""Topographic ellipsoid body renderer: relief limb, ragged terminator, laws.

This is the image-side body render path used whenever a body needs more
than the classic smooth-Lambert ellipsoid: a limb-relief field
(:mod:`spindoctor.sim.forward.relief`), a non-Lambert photometric law or
opposition surge (:mod:`spindoctor.sim.forward.photometry`), or an
oversampled radiance grid (where it is also the fast path -- see below).
:func:`spindoctor.sim.forward.body.render_single_body` dispatches here; a
smooth Lambert body at oversample 1 keeps the classic path byte-for-byte.

**Split-resolution rendering.**  Disc shading is low-frequency by
construction -- relief moves the silhouette and the terminator, never the
disc shading -- so the shading field (surface normals, crater texture,
photometric law) is computed once at detector resolution and bilinearly
upsampled, while the sharp content is rasterized at the full working grid:
the silhouette mask (with the relief-perturbed limb) and the terminator
shadow march.  This replaces the per-subsample shading that made an
oversampled body render ~16x more expensive than its detector-grid
equivalent, and is what lets a body-bearing scene meet the render-time
budget (see ``tests/integration/test_sim_perf.py``).

**Relief application.**  The renderer's normalized ellipse radial function
``e(p)`` (1 exactly at the unperturbed limb) becomes
``e_adj(p) = e(p) / (1 + delta(theta))``, placing the limb at
``r_ellipse * (1 + delta)``; ``delta`` is the relief surface's own outline,
the envelope ``max over lat of (1 + h) cos(lat) - 1`` along each azimuth.
On the shading grid the relief tilts every normal by the field's surface
gradient, so slopes facing the sun brighten and slopes facing away darken,
and an exact ray march through the relief shell (both in
:mod:`spindoctor.sim.forward.relief`) casts terrain shadows, which render as
night.  Only points whose sun elevation lies within the field's steepest
slope can be shadowed, so the march visits those alone.  Shading and
shadows are evaluated per shading pixel and bilinearly upsampled with the
rest of the shading, so a shadow's edge is as soft as the shading grid.

The mesh-body path does not dispatch here: a polyhedral-mesh body applies
the relief field as a per-vertex radial perturbation in
:mod:`spindoctor.sim.forward.body_mesh`, and the scene validator rejects
the photometric-law, surface-texture, crater, and atmosphere keys on mesh
bodies rather than letting them silently not render.
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.sim.ellipsoid_geometry import ellipsoid_image_normals, illumination_vector
from spindoctor.sim.forward.body_texture import (
    AlbedoTextureSpec,
    DiscTextureSpec,
    TransitSpec,
    apply_transits,
    surface_texture_factor,
)
from spindoctor.sim.forward.photometry import incidence_cosines, shade_surface
from spindoctor.sim.forward.relief import ReliefField, cast_shadows, synthesize_relief_field
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_matmul3
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['TopoBodyRender', 'TopoBodySpec', 'create_topographic_body', 'render_topographic_body']

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


@dataclass(frozen=True)
class TopoBodySpec:
    """Everything the topographic renderer needs beyond size and center.

    All pixel quantities are in units of the incoming render grid (the
    oversampled grid when ``oversample`` > 1).

    Parameters:
        axis1: Full width of ellipsoid axis 1 (a) in pixels.
        axis2: Full width of ellipsoid axis 2 (b) in pixels.
        axis3: Full width of ellipsoid axis 3 (c, depth) in pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        rotation_tilt: Tilt toward/away from the viewer, radians.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians (0 fully lit, pi backlit).
        crater_fill: Approximate crater coverage fraction (0 = none).
        crater_min_radius: Minimum crater radius as a fraction of axis1.
        crater_max_radius: Maximum crater radius as a fraction of axis1.
        crater_power_law_exponent: Crater radius power-law exponent.
        crater_relief_scale: Crater depth scale factor.
        anti_aliasing: Silhouette anti-aliasing amount in [0, 1]; only
            consumed at ``oversample`` 1, where it supersamples the
            silhouette exactly as the classic path does (an oversampled
            scene already resolves the limb on its own grid).
        crater_seed: Seed for crater placement (None with no craters).
        oversample: The incoming grid's oversampling factor; shading is
            computed at the detector grid ``size / oversample``.
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

    axis1: float
    axis2: float
    axis3: float
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


@dataclass
class _Grid:
    """One resolution level of the body geometry.

    Parameters:
        v_ctr: Centered (unrotated) v coordinate of each pixel.
        u_ctr: Centered (unrotated) u coordinate of each pixel.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        e2: Squared normalized ellipse radial function ``e(p)**2``.
        center: Body center (v, u) in this grid's pixels.
        semi_a: Semi-axis a at this grid's resolution, pixels.
        semi_b: Semi-axis b at this grid's resolution, pixels.
        semi_c: Depth semi-axis c at this grid's resolution, pixels.
    """

    v_ctr: NDArrayFloatType
    u_ctr: NDArrayFloatType
    v_rot: NDArrayFloatType
    u_rot: NDArrayFloatType
    e2: NDArrayFloatType
    center: tuple[float, float]
    semi_a: float
    semi_b: float
    semi_c: float


def _body_grid(
    dims: tuple[int, int],
    center: tuple[float, float],
    semi_axes: tuple[float, float, float],
    *,
    cos_rz: float,
    sin_rz: float,
    cos_rt: float,
) -> _Grid:
    """Build the rotated-frame coordinates and radial function for one grid.

    The conventions (the grid built in pixel corner coordinates, rotation then
    tilt compression) match the classic renderer exactly.

    Parameters:
        dims: Grid dimensions (rows, columns).
        center: Body center (v, u) in this grid's pixels.
        semi_axes: Semi-axes (a, b, c) in this grid's pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        cos_rt: Cosine of the tilt.

    Returns:
        The populated :class:`_Grid`.
    """
    size_v, size_u = dims
    semi_a, semi_b, semi_c = semi_axes
    # The grid starts as array rows and columns and the body center is stated
    # in pixel corner coordinates, so the half pixel goes on before the center
    # comes off.
    v_coords, u_coords = np.mgrid[0:size_v, 0:size_u].astype(float)
    v_coords += PIXEL_CENTER_TO_CORNER_PX - center[0]
    u_coords += PIXEL_CENTER_TO_CORNER_PX - center[1]
    v_rot1 = v_coords * cos_rz - u_coords * sin_rz
    u_rot1 = v_coords * sin_rz + u_coords * cos_rz
    v_rot = v_rot1 * cos_rt
    e2 = (v_rot / semi_a) ** 2 + (u_rot1 / semi_b) ** 2
    return _Grid(
        v_ctr=v_coords,
        u_ctr=u_coords,
        v_rot=v_rot,
        u_rot=u_rot1,
        e2=e2,
        center=center,
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_c,
    )


def _bilinear_upsample(field: NDArrayFloatType, factor: int) -> NDArrayFloatType:
    """Bilinearly upsample a field by an integer factor (pixel-center aligned).

    Output sample ``i`` lands at ``(i + PIXEL_CENTER_TO_CORNER_PX) / factor -
    PIXEL_CENTER_TO_CORNER_PX`` in the input field's pixel centric coordinates:
    the output row is lifted to pixel corner coordinates, rescaled to the input
    grid, and lowered again for the array lookup.  Borders clamp to the nearest
    input sample.

    Parameters:
        field: The 2-D input field.
        factor: Integer upsampling factor (1 returns the input unchanged).

    Returns:
        The ``(rows * factor, cols * factor)`` upsampled field.
    """
    if factor == 1:
        return field
    n_v, n_u = field.shape

    def _axis(n: int) -> tuple[NDArrayIntType, NDArrayIntType, NDArrayFloatType]:
        centers = (
            np.arange(n * factor, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
        ) / factor - PIXEL_CENTER_TO_CORNER_PX
        low = np.clip(np.floor(centers).astype(np.intp), 0, n - 1)
        high = np.minimum(low + 1, n - 1)
        weight = np.clip(centers - low, 0.0, 1.0)
        return low, high, weight

    row_lo, row_hi, row_w = _axis(n_v)
    col_lo, col_hi, col_w = _axis(n_u)
    rows = field[row_lo, :] * (1.0 - row_w)[:, np.newaxis] + field[row_hi, :] * row_w[:, np.newaxis]
    return rows[:, col_lo] * (1.0 - col_w) + rows[:, col_hi] * col_w


@dataclass(frozen=True)
class TopoBodyRender:
    """One topographic body render: its radiance and its solid silhouette.

    Parameters:
        intensity: The intensity image in [0, 1], 0 outside the body.  A
            surface stage can also leave a pixel inside the body at 0, such
            as a transit's full umbra.
        silhouette: The fraction of each pixel the body's solid outline
            covers, in [0, 1], independent of how bright the pixel renders.
    """

    intensity: NDArrayFloatType
    silhouette: NDArrayFloatType


def create_topographic_body(
    size: tuple[int, int],
    center: tuple[float, float],
    spec: TopoBodySpec,
) -> NDArrayFloatType:
    """Render one topographic ellipsoid body onto a black frame.

    Parameters:
        size: Frame dimensions (rows, columns) on the incoming render grid.
        center: Body center (v, u) in incoming-grid pixels.
        spec: The body's geometry, texture, relief, and photometry.

    Returns:
        The ``size`` intensity image in [0, 1] (0 outside the body); the
        intensity of :func:`render_topographic_body`.

    Raises:
        ValueError: If the frame size is not divisible by the oversample.
    """
    return render_topographic_body(size, center, spec).intensity


def render_topographic_body(
    size: tuple[int, int],
    center: tuple[float, float],
    spec: TopoBodySpec,
) -> TopoBodyRender:
    """Render one topographic ellipsoid body and its solid silhouette.

    The silhouette is the relief-perturbed outline's pixel coverage, taken
    before any shading, so a pixel that renders black inside the body (a
    transit umbra, a zero-albedo spot) still belongs to it.

    Parameters:
        size: Frame dimensions (rows, columns) on the incoming render grid.
        center: Body center (v, u) in incoming-grid pixels.
        spec: The body's geometry, texture, relief, and photometry.

    Returns:
        The :class:`TopoBodyRender` on the ``size`` grid.

    Raises:
        ValueError: If the frame size is not divisible by the oversample.
    """
    size_v, size_u = size
    os_in = max(1, int(spec.oversample))
    if size_v % os_in or size_u % os_in:
        raise ValueError(f'frame size {size} is not divisible by oversample {os_in}')

    # At oversample 1 the anti-aliasing knob supersamples the silhouette,
    # exactly as the classic path scales its working grid; an oversampled
    # scene already resolves the limb, so its work grid IS the render grid.
    aa_scale = int(1 + 3 * spec.anti_aliasing) if (os_in == 1 and spec.anti_aliasing > 0) else 1
    work_dims = (size_v * aa_scale, size_u * aa_scale)
    shade_dims = (size_v // os_in, size_u // os_in)
    up_factor = os_in * aa_scale

    cos_rz = float(np.cos(spec.rotation_z))
    sin_rz = float(np.sin(spec.rotation_z))
    cos_rt = float(np.cos(spec.rotation_tilt))
    semi = (spec.axis1 / 2.0, spec.axis2 / 2.0, spec.axis3 / 2.0)

    shade_grid = _body_grid(
        shade_dims,
        (center[0] / os_in, center[1] / os_in),
        (semi[0] / os_in, semi[1] / os_in, semi[2] / os_in),
        cos_rz=cos_rz,
        sin_rz=sin_rz,
        cos_rt=cos_rt,
    )
    field: ReliefField | None = None
    if spec.limb_relief_rms > 0.0:
        field = synthesize_relief_field(
            spec.limb_relief_rms, spec.limb_relief_corr_deg, spec.relief_seed
        )
        if field.h_max <= field.h_min:
            field = None

    intensity_shade = _shade_disc(shade_grid, spec, field, cos_rz=cos_rz, sin_rz=sin_rz)

    work_grid = _body_grid(
        work_dims,
        (center[0] * aa_scale, center[1] * aa_scale),
        (semi[0] * aa_scale, semi[1] * aa_scale, semi[2] * aa_scale),
        cos_rz=cos_rz,
        sin_rz=sin_rz,
        cos_rt=cos_rt,
    )

    mask = _silhouette_mask(work_grid, field)
    intensity_work = _bilinear_upsample(intensity_shade, up_factor)
    intensity_work *= mask

    if aa_scale > 1:
        intensity_work = intensity_work.reshape(size_v, aa_scale, size_u, aa_scale).mean(
            axis=(1, 3)
        )
        mask = mask.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
    return TopoBodyRender(intensity=np.clip(intensity_work, 0.0, 1.0), silhouette=mask)


def _shade_disc(
    grid: _Grid,
    spec: TopoBodySpec,
    field: ReliefField | None,
    *,
    cos_rz: float,
    sin_rz: float,
) -> NDArrayFloatType:
    """Shade the disc at the shading grid under the spec's photometric law.

    Normals are computed everywhere with the depth clamped to the visible
    hemisphere, so pixels just outside the limb carry the limb's shading
    values: the bilinear upsample then continues the disc shading smoothly
    across the (sharper) silhouette boundary instead of blending to black.
    With relief the normals are the relief surface's and the terrain casts
    shadows (:func:`_relief_shading`).

    Parameters:
        grid: The shading-resolution geometry.
        spec: The body spec (crater texture and photometric law are read).
        field: The relief realization, or None for the smooth surface; it sets
            the outline the transits reach.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        The intensity on the shading grid.
    """
    z_coords = grid.semi_c * np.sqrt(np.maximum(0.0, 1.0 - grid.e2))
    everywhere = np.ones_like(grid.e2)
    normal_v, normal_u, normal_z = ellipsoid_image_normals(
        everywhere,
        grid.v_rot,
        grid.u_rot,
        z_coords=z_coords,
        work_semi_major=grid.semi_a,
        work_semi_minor=grid.semi_b,
        work_semi_c=grid.semi_c,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
    )
    shadowed: NDArrayBoolType | None = None
    if field is not None:
        # Relief reaches only pixels inside its outline, plus the shading
        # pixel the upsample reads beyond it; farther out the smooth limb
        # normals carry the continuity and the silhouette masks them.
        reach = 1.0 + max(field.h_max, 0.0) + 2.0 / max(min(grid.semi_a, grid.semi_b), 1.0)
        near = grid.e2 < reach * reach
        relief_v, relief_u, relief_z, shadowed = _relief_shading(
            grid, spec, field, near, cos_rz=cos_rz, sin_rz=sin_rz
        )
        normal_v[near] = relief_v
        normal_u[near] = relief_u
        normal_z[near] = relief_z
    cos_i = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
    )

    if spec.crater_fill > 0.0:
        height = _crater_heights(grid, spec)
        # Local import: body.py imports this module's spec for dispatch, so
        # the crater helpers resolve lazily to keep the import graph acyclic.
        from spindoctor.sim.forward.body import perturb_normals_by_height

        normal_v, normal_u, normal_z = perturb_normals_by_height(
            normal_v, normal_u, normal_z, height
        )
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

    # The clamped-depth parametrization has no far hemisphere, and relief
    # normals are mirrored onto the near one (normal_z >= 0 everywhere; night
    # still floors through mu0 <= 0), so the emission
    # cosine is kept strictly positive: the visibility gate must not zero
    # the exact-limb ring, whose values the upsample continues across the
    # sharper silhouette boundary.
    intensity = shade_surface(
        cos_i,
        np.maximum(normal_z, 1e-12),
        law=spec.photometric_law,
        phase_angle=spec.phase_angle,
        minnaert_k=spec.minnaert_k,
        surge_amplitude=spec.surge_amplitude,
        surge_width_deg=spec.surge_width_deg,
    )
    return _apply_surface_textures(intensity, grid, spec, field)


def _relief_shading(
    grid: _Grid,
    spec: TopoBodySpec,
    field: ReliefField,
    near: NDArrayBoolType,
    *,
    cos_rz: float,
    sin_rz: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """The relief surface's image-frame normals and cast shadows at the ``near`` pixels.

    Each shading pixel reads the terrain at its smooth-surface direction
    ``p`` in the body's parameter space (the smooth body is the unit sphere,
    its poles on the observer axis; a pixel past the smooth limb reads the
    limb direction).  The relief surface ``(1 + h) p`` has the parameter-space
    normal ``p - grad h / (1 + h)``, carried to the body frame by the
    ellipsoid's inverse semi-axes and rotated into the image like the smooth
    normals.  The sun direction goes to parameter space by the same affine
    map, where :func:`~spindoctor.sim.forward.relief.cast_shadows` marches the
    pixels whose sun elevation lies within the field's slope band; a pixel
    below the band always lies in shadow.  A normal that faces away from the
    observer is mirrored through the sky plane, so every returned
    ``normal_z`` is at least 0.

    Parameters:
        grid: The shading-resolution geometry.
        spec: The body spec (illumination geometry is read).
        field: The relief realization.
        near: The shading pixels the relief can reach.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        ``(normal_v, normal_u, normal_z)`` at the ``near`` pixels, in their
        row-major order, and the shading-grid ``shadowed`` mask.
    """
    semi = np.array([grid.semi_a, grid.semi_b, grid.semi_c], dtype=np.float64)
    e2 = grid.e2[near]
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
    shadowed = np.zeros(grid.e2.shape, dtype=np.bool_)
    shadowed[near] = shadowed_near
    return normal_v, normal_u, normal_z, shadowed


def _apply_surface_textures(
    intensity: NDArrayFloatType,
    grid: _Grid,
    spec: TopoBodySpec,
    field: ReliefField | None,
) -> NDArrayFloatType:
    """Apply the spec's multiplicative textures and transits to the shading.

    Texture layering: the albedo/disc texture factor multiplies the law
    shading first, then transit shadows multiply the textured result, then
    transiting moon discs overwrite on top (from the pre-texture smooth-law
    shading -- a foreground moon does not inherit the parent's texture).
    Everything happens on the shading grid, so the silhouette (rasterized
    separately on the working grid) is untouched by construction.  Transits
    reach the body's outline: the relief outline when the body has relief,
    else the smooth limb.

    Parameters:
        intensity: The law-shaded intensity on the shading grid.
        grid: The shading-resolution geometry.
        spec: The body spec (texture and transit fields are read).
        field: The relief realization, or None for the smooth limb.

    Returns:
        The textured intensity (the input array, unchanged, when the spec
        has no texture and no transits).
    """
    if spec.albedo_texture is None and spec.disc_texture is None and not spec.transits:
        return intensity
    x = grid.v_rot / grid.semi_a
    y = grid.u_rot / grid.semi_b
    z = np.sqrt(np.maximum(0.0, 1.0 - grid.e2))
    factor = surface_texture_factor(
        spec.albedo_texture,
        spec.disc_texture,
        x=x,
        y=y,
        z=z,
        mean_radius_px=(grid.semi_a + grid.semi_b) / 2.0,
    )
    textured = intensity if factor is None else np.clip(intensity * factor, 0.0, 1.0)
    if spec.transits:
        if textured is intensity:
            textured = intensity.copy()
        apply_transits(
            textured,
            spec.transits,
            base_intensity=intensity,
            v_ctr=grid.v_ctr,
            u_ctr=grid.u_ctr,
            disc_mask=grid.e2 < 1.0 if field is None else _silhouette_mask(grid, field) > 0.0,
        )
    return textured


def _crater_heights(grid: _Grid, spec: TopoBodySpec) -> NDArrayFloatType:
    """Carve the crater height field on the shading grid.

    Uses the classic path's carving loop (count law, radius power law, and
    per-crater rng consumption) at the shading grid's resolution, seeded by
    the same per-body crater seed the classic path derives.

    Parameters:
        grid: The shading-resolution geometry.
        spec: The body spec (crater knobs and seed are read).

    Returns:
        The crater height field over the shading grid.
    """
    # Local import: body.py imports this module's spec for dispatch, so the
    # crater helpers resolve lazily to keep the import graph acyclic.
    from spindoctor.sim.forward.body import carve_crater_heights
    from spindoctor.sim.seeds import stable_param_seed

    inside_strict = grid.e2 < 1.0
    total_area = int(np.sum(inside_strict))
    avg_crater_area = (spec.crater_max_radius * grid.semi_a) ** 2
    n_craters = int(np.clip(int(spec.crater_fill * total_area / avg_crater_area), 0, 1000))
    if n_craters == 0 or not inside_strict.any():
        return np.zeros_like(grid.e2)

    if spec.crater_seed is not None:
        seed_value = int(spec.crater_seed) & 0x7FFFFFFF
    else:
        seed_value = stable_param_seed(spec.axis1, spec.axis2, spec.axis3, grid.center)
        seed_value &= 0x7FFFFFFF
    return carve_crater_heights(
        inside_strict,
        grid.v_ctr,
        grid.u_ctr,
        nz=np.argwhere(inside_strict),
        rng=np.random.default_rng(seed_value),
        n_craters=n_craters,
        R_min=spec.crater_min_radius * grid.semi_a,
        R_max=spec.crater_max_radius * grid.semi_a,
        crater_power_law_exponent=spec.crater_power_law_exponent,
        crater_relief_scale=spec.crater_relief_scale,
        work_center_v=grid.center[0],
        work_center_u=grid.center[1],
        aa_scale=1,
    )


def _silhouette_mask(grid: _Grid, field: ReliefField | None) -> NDArrayFloatType:
    """The body's silhouette on the working grid, relief-perturbed when on.

    With relief, ``e_adj = e / (1 + delta(theta))`` places the limb at
    ``r_ellipse * (1 + delta)``, with ``delta`` the relief surface's outline
    envelope (:meth:`~spindoctor.sim.forward.relief.ReliefField.outline_delta`);
    the perturbation only matters in a band ``|e - 1| <= |delta|_max`` around
    the smooth limb, so azimuths and the outline are evaluated on that band
    only.

    The relief azimuth ``phi`` is the elliptical parametric angle in the
    body's rotated frame, NOT the image azimuth about the body center: the
    relief field is attached to the body, so the outline, the shading, and
    the cast shadows read one consistent surface under any in-plane rotation.
    For a circular disc the two conventions coincide.

    Parameters:
        grid: The working-grid geometry.
        field: The relief realization, or None for the smooth limb.

    Returns:
        Float mask (1 inside the body, 0 outside).
    """
    mask = grid.e2 <= 1.0
    if field is not None:
        delta_cap = max(abs(field.h_min), abs(field.h_max)) * 1.001 + 1e-12
        low = max(0.0, 1.0 - delta_cap)
        high = 1.0 + delta_cap
        band = (grid.e2 > low * low) & (grid.e2 < high * high)
        if band.any():
            e_band = np.sqrt(grid.e2[band])
            # Body-attached azimuth: the rotated-frame elliptical parametric
            # angle (identical to the image azimuth for a circular disc).
            phi = np.arctan2(grid.u_rot[band] / grid.semi_b, grid.v_rot[band] / grid.semi_a)
            delta = field.outline_delta(phi)
            mask[band] = e_band <= 1.0 + delta
    return mask.astype(np.float64)
