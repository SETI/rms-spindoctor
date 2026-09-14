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
``e_adj(p) = e(p) / (1 + delta(theta))``, placing the perturbed limb at
``r_ellipse * (1 + delta)``; ``delta`` is the relief field sampled along
the sub-observer horizon circle.  Shading normals keep the unperturbed
``e``.  The terminator march shadows near-terminator disc points against
upstream terrain in absolute heights (the march itself lives in
:mod:`spindoctor.sim.forward.relief`); shadowed pixels render at the
dark-side floor, and the marched band is bounded by the cap, so raggedness
grows toward the terminator and the cost stays bounded.

The mesh-body path does not dispatch here: a polyhedral-mesh body applies
the relief field as a per-vertex radial perturbation in
:mod:`spindoctor.sim.forward.body_mesh`, and the scene validator rejects
the photometric-law, surface-texture, crater, and atmosphere keys on mesh
bodies rather than letting them silently not render.
"""

import math
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
from spindoctor.sim.forward.photometry import (
    DARK_SIDE_FLOOR,
    incidence_cosines,
    shade_surface,
)
from spindoctor.sim.forward.relief import ReliefField, march_shadows, synthesize_relief_field
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['TopoBodySpec', 'create_topographic_body']

# Coarse terminator-band margins: the candidate band on the
# detector-resolution incidence map is widened by this factor plus a
# few-pixel pad before the exact per-pixel refinement, so discretization
# cannot exclude a marchable pixel.
_BAND_MARGIN_FACTOR = 1.5
_BAND_MARGIN_PAD_PX = 4.0
# The march samples the terrain at its own resolution: this many samples
# per relief-field grid cell (the field is band-limited well above the
# cell scale, so finer sampling changes no shadow decision), never finer
# than the exact one-render-pixel step.
_MARCH_SAMPLES_PER_CELL = 4.0


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
    # The grid starts as array rows and columns and the body centre is stated
    # in pixel corner coordinates, so the half pixel goes on before the centre
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
        The ``size`` intensity image in [0, 1] (0 outside the body).

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
    intensity_shade, cos_i_smooth = _shade_disc(shade_grid, spec, cos_rz=cos_rz, sin_rz=sin_rz)

    work_grid = _body_grid(
        work_dims,
        (center[0] * aa_scale, center[1] * aa_scale),
        (semi[0] * aa_scale, semi[1] * aa_scale, semi[2] * aa_scale),
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

    mask = _silhouette_mask(work_grid, field)
    intensity_work = _bilinear_upsample(intensity_shade, up_factor)
    if field is not None:
        _apply_terminator_shadows(
            intensity_work,
            work_grid,
            field,
            spec=spec,
            cos_i_smooth=cos_i_smooth,
            shade_e2=shade_grid.e2,
            up_factor=up_factor,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
            cos_rt=cos_rt,
        )
    intensity_work *= mask

    if aa_scale > 1:
        intensity_work = intensity_work.reshape(size_v, aa_scale, size_u, aa_scale).mean(
            axis=(1, 3)
        )
    return np.clip(intensity_work, 0.0, 1.0)


def _shade_disc(
    grid: _Grid,
    spec: TopoBodySpec,
    *,
    cos_rz: float,
    sin_rz: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Shade the disc at the shading grid under the spec's photometric law.

    Normals are computed everywhere with the depth clamped to the visible
    hemisphere, so pixels just outside the limb carry the limb's shading
    values: the bilinear upsample then continues the disc shading smoothly
    across the (sharper) silhouette boundary instead of blending to black.

    Parameters:
        grid: The shading-resolution geometry.
        spec: The body spec (crater texture and photometric law are read).
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        Tuple of (intensity, smooth-surface cos(incidence)), both on the
        shading grid.  The incidence map ignores crater perturbation: the
        terminator march band is a property of the smooth surface.
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
    cos_i_smooth = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
    )

    cos_i = cos_i_smooth
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

    # The clamped-depth parametrization has no far hemisphere (normal_z >= 0
    # everywhere; night still floors through mu0 <= 0), so the emission
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
    return _apply_surface_textures(intensity, grid, spec), cos_i_smooth


def _apply_surface_textures(
    intensity: NDArrayFloatType,
    grid: _Grid,
    spec: TopoBodySpec,
) -> NDArrayFloatType:
    """Apply the spec's multiplicative textures and transits to the shading.

    Texture layering: the albedo/disc texture factor multiplies the law
    shading first, then transit shadows multiply the textured result, then
    transiting moon discs overwrite on top (from the pre-texture smooth-law
    shading -- a foreground moon does not inherit the parent's texture).
    Everything happens on the shading grid, so the silhouette (rasterized
    separately on the working grid) is untouched by construction.

    Parameters:
        intensity: The law-shaded intensity on the shading grid.
        grid: The shading-resolution geometry.
        spec: The body spec (texture and transit fields are read).

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
            disc_mask=grid.e2 < 1.0,
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
    ``r_ellipse * (1 + delta)``; the perturbation only matters in a band
    ``|e - 1| <= |delta|_max`` around the smooth limb, so azimuths and the
    limb slice are evaluated on that band only.

    The relief azimuth ``phi`` is the elliptical parametric angle in the
    body's rotated frame, NOT the image azimuth about the body center: the
    relief field is attached to the body, so the silhouette slice and the
    terminator march sample one consistent surface under any in-plane
    rotation.  For a circular disc the two conventions coincide.

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
            delta = field.limb_delta(phi)
            mask[band] = e_band <= 1.0 + delta
    return mask.astype(np.float64)


def _apply_terminator_shadows(
    intensity_work: NDArrayFloatType,
    grid: _Grid,
    field: ReliefField,
    *,
    spec: TopoBodySpec,
    cos_i_smooth: NDArrayFloatType,
    shade_e2: NDArrayFloatType,
    up_factor: int,
    cos_rz: float,
    sin_rz: float,
    cos_rt: float,
) -> None:
    """Shadow near-terminator pixels against upstream relief, in place.

    Candidate pixels are the disc points within the march cap of the
    terminator: on the shading-grid incidence map, points whose sun
    elevation ``eps`` (surface distance to the terminator is ``R * eps``)
    lies within their own cap ``min((h_max - h_min) * cot(eps),
    sqrt(2 * h_max))`` radians of arc, widened by the coarse-band margins
    and refined per-pixel with exact incidence.  The relief and arc maps
    are built only over the band dilated by the cap's reach, and the march
    itself is :func:`spindoctor.sim.forward.relief.march_shadows`, stepping
    at the relief field's own grid resolution (never finer than one
    render pixel of surface arc).

    Parameters:
        intensity_work: Working-grid intensity, modified in place.
        grid: The working-grid geometry.
        field: The relief realization.
        spec: The body spec (illumination geometry is read).
        cos_i_smooth: Smooth-surface incidence map on the shading grid.
        shade_e2: Squared radial function on the shading grid.
        up_factor: Working-grid pixels per shading-grid pixel.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        cos_rt: Cosine of the tilt.
    """
    from scipy import ndimage

    radius_max = max(grid.semi_a, grid.semi_b, grid.semi_c)
    height_range = (field.h_max - field.h_min) * radius_max
    if height_range <= 0.0:
        return
    # In-plane sun direction (unit): the direction the march walks toward.
    sun_v = -math.cos(spec.illumination_angle)
    sun_u = math.sin(spec.illumination_angle)
    if math.sin(spec.phase_angle) <= 0.0:
        # No in-plane sun component: no terminator crosses the disc.
        return

    # Coarse candidate band on the shading grid, refined exactly below: the
    # disc points within the march cap of the terminator.  The sun elevation
    # eps puts a point R * eps of surface arc from the terminator, so the
    # cap condition is eps <= min((h_max - h_min) * cot(eps), sqrt(2 h_max))
    # radians (the local radius cancels), widened by the band margins.
    h_range = field.h_max - field.h_min
    radius_min = min(grid.semi_a, grid.semi_b, grid.semi_c)
    pad_rad = _BAND_MARGIN_PAD_PX * up_factor / max(radius_min, 1.0)
    elevation = np.arcsin(np.clip(cos_i_smooth, 0.0, 1.0))
    with np.errstate(divide='ignore'):
        cap_rad = np.minimum(
            h_range / np.maximum(np.tan(elevation), 1e-12),
            math.sqrt(max(2.0 * field.h_max, 0.0)),
        )
    coarse = (
        (shade_e2 < 1.0)
        & (cos_i_smooth > 0.0)
        & (elevation <= cap_rad * _BAND_MARGIN_FACTOR + pad_rad)
    )
    if not coarse.any():
        return
    coarse = ndimage.binary_dilation(coarse, iterations=1)

    # The maps must cover the longest capped march: d_max surface px, each
    # advancing at most 1 / cos(tilt) image px.
    d_max_global = radius_max * math.sqrt(max(2.0 * field.h_max, 0.0))
    reach_work = d_max_global / max(cos_rt, 0.1) + 2.0
    reach_shade = math.ceil(reach_work / up_factor) + 1
    region_shade = ndimage.maximum_filter(coarse, size=2 * reach_shade + 1)

    disc = grid.e2 < 1.0
    candidate_mask = _upscale_mask(coarse, up_factor) & disc
    region_mask = _upscale_mask(region_shade, up_factor) & disc
    region_v, region_u = np.nonzero(region_mask)
    if region_v.size == 0:
        return

    x = grid.v_rot[region_v, region_u] / grid.semi_a
    y = grid.u_rot[region_v, region_u] / grid.semi_b
    z_n = np.sqrt(np.clip(1.0 - x * x - y * y, 1e-12, 1.0))
    # The sample longitude arctan2(y, x) is the same body-attached
    # rotated-frame parametric azimuth the silhouette slice uses, so the
    # march heights and the limb perturbation read one consistent field.
    h_frac = field.sample(np.arcsin(np.clip(z_n, -1.0, 1.0)), np.mod(np.arctan2(y, x), 2 * np.pi))
    radius_local = np.sqrt(
        (grid.semi_a * x) ** 2 + (grid.semi_b * y) ** 2 + (grid.semi_c * z_n) ** 2
    )

    # Surface arc spanned by one image pixel along the sun direction.
    sun_rv = (sun_v * cos_rz - sun_u * sin_rz) * cos_rt
    sun_ru = sun_v * sin_rz + sun_u * cos_rz
    tangential = (x * sun_rv / grid.semi_a + y * sun_ru / grid.semi_b) / z_n
    arc = np.sqrt(sun_rv**2 + sun_ru**2 + (grid.semi_c * tangential) ** 2)

    height_map = np.zeros(grid.e2.shape, dtype=np.float64)
    arc_map = np.zeros(grid.e2.shape, dtype=np.float64)
    height_map[region_v, region_u] = h_frac * radius_local
    arc_map[region_v, region_u] = arc

    # Exact incidence at the candidate pixels (smooth-surface normals).
    cand_in_region = candidate_mask[region_v, region_u]
    cand_v = region_v[cand_in_region]
    cand_u = region_u[cand_in_region]
    if cand_v.size == 0:
        return
    xc = x[cand_in_region]
    yc = y[cand_in_region]
    zc = z_n[cand_in_region]
    nv_local = xc / grid.semi_a
    nu_local = yc / grid.semi_b
    nz_local = zc / grid.semi_c
    norm = np.sqrt(nv_local**2 + nu_local**2 + nz_local**2)
    nv_local /= norm
    nu_local /= norm
    nz_local /= norm
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=spec.illumination_angle, phase_angle=spec.phase_angle
    )
    cos_i = (
        (nv_local * cos_rz + nu_local * sin_rz) * illum_v
        + (-nv_local * sin_rz + nu_local * cos_rz) * illum_u
        + nz_local * illum_z
    )

    lit = cos_i > 1e-9
    tan_i = np.zeros_like(cos_i)
    tan_i[lit] = np.sqrt(np.maximum(0.0, 1.0 - cos_i[lit] ** 2)) / cos_i[lit]
    r_local_c = radius_local[cand_in_region]
    h_frac_c = h_frac[cand_in_region]
    # The march samples the terrain at the field's own resolution (the field
    # is band-limited well above its grid-cell scale), never finer than
    # the exact one-render-pixel step used for small bodies.
    cell_arc_px = 2.0 * np.pi / field.grid.shape[0] * radius_max
    step_arc = max(1.0, cell_arc_px / _MARCH_SAMPLES_PER_CELL)
    # A pixel is marchable only when the cap admits at least one sample.
    d_max = np.minimum(
        (field.h_max - field.h_min) * r_local_c * tan_i,
        r_local_c * math.sqrt(max(2.0 * field.h_max, 0.0)),
    )
    marchable = lit & (d_max >= step_arc)
    if not marchable.any():
        return

    # The candidates are array rows and columns and the march reads pixel
    # corner coordinates, so the half pixel goes on here.
    shadow, _stats = march_shadows(
        cand_v[marchable] + PIXEL_CENTER_TO_CORNER_PX,
        cand_u[marchable] + PIXEL_CENTER_TO_CORNER_PX,
        h_point=h_frac_c[marchable],
        tan_incidence=tan_i[marchable],
        radius_px=r_local_c[marchable],
        height_map=height_map,
        arc_per_px_map=arc_map,
        domain_mask=region_mask,
        h_max=field.h_max,
        h_min=field.h_min,
        sun_v=sun_v,
        sun_u=sun_u,
        step_arc_px=step_arc,
    )
    shadow_v = cand_v[marchable][shadow]
    shadow_u = cand_u[marchable][shadow]
    intensity_work[shadow_v, shadow_u] = DARK_SIDE_FLOOR


def _upscale_mask(mask: NDArrayBoolType, factor: int) -> NDArrayBoolType:
    """Expand a shading-grid boolean mask to the working grid by block repeat.

    Parameters:
        mask: The shading-grid mask.
        factor: Working-grid pixels per shading-grid pixel.

    Returns:
        The working-grid mask.
    """
    if factor == 1:
        return mask
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
