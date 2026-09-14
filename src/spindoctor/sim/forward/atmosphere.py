"""Image-side atmosphere rendering for haze-limb (Titan-class) bodies.

A body carrying an ``atmosphere`` block gains an exponential haze layer above
its surface, evaluated by :func:`apply_atmosphere` after the disc is shaded.
The haze splits into two compositing components (:class:`AtmosphereLayers`):
the on-disc haze joins the body's opaque paint (the disc is opaque anyway,
so the painted silhouette stays exactly the no-atmosphere silhouette), while
the above-limb glow is a translucent :class:`HaloScreen` -- an emission map
plus a per-pixel transmission ``exp(-tau)`` -- that the radiance stage
composites over the background exactly as the ring system's transmission
screen: ``img = glow + exp(-tau) * img_behind``, with point sources
attenuated by the same factor.  A star behind the halo therefore dims by the
tangent transmission instead of vanishing, and the body's mask / depth /
occlusion truth sees only the solid silhouette, never the halo.  The haze is
a truth key the navigator never sees: its predicted-body model keeps a hard
limb at the reference radius, so the soft rendered limb is a designed
mismatch (the substrate for the Titan altitude-versus-phase problem).

**Tangent optical depth.**  A line of sight grazing the body at tangent
altitude ``h`` (pixels above the reference radius) accumulates a slant
optical depth

    ``tau(h) = tau_ref * exp(-(h - ref_altitude_px) / scale_height_px)``

so ``tau_ref`` is the tangent optical depth at ``ref_altitude_px``.  An
optional detached haze shell adds a Gaussian bump in ``tau`` centred at
``detached_px`` above the surface.  The shell exists only in this above-limb
tangent depth: the on-disc excess column is shell-blind (a geometrically
thin shell projected against the disc adds negligible slant contrast), so
the shell renders solely as a second band in the tangent glow.

**Single scattering.**  The emergent haze brightness is a source term times
an opacity term.  The source is a single-scattering albedo scaled by a
Henyey-Greenstein phase factor (forward-scattering when ``g`` > 0, which
brightens the limb at high phase and, in the limit, produces the ring of
light past phase 150 deg) and a wrapped illumination weight that stays
positive past the terminator over the horizon-dip angle ``sqrt(2 * H / R)``
of arc -- the solar depression at which a column one scale height up loses
direct sunlight -- floored at 0.05 rad so a razor-thin atmosphere still
wraps resolvably (so the terminator brightens past 90 deg incidence instead
of cutting off).  The
opacity is ``1 - exp(-tau)``: above the limb ``tau`` is the tangent optical
depth, so the limb becomes a soft exponential ramp whose apparent radius
grows with the haze brightness (hence with phase); on the disc the slant
optical depth grows toward the limb as ``1 / cos(emission)``, so the haze
concentrates at the limb and stays faint at disc centre.  The disc-side
column scales from the physical vertical depth

    ``tau_vert = tau_ref * exp(ref_altitude_px / H) / sqrt(2 * pi * R / H)``

(the surface tangent depth divided by the grazing enhancement, ``R`` the
mean radius and ``H`` the scale height in pixels), so the two sides of the
limb describe one atmosphere at any reference altitude.

**Symmetry-breaking structure.**  The column described above is exactly
mirror-symmetric about the sunward axis, which would make it useless for
grading a navigator that measures pointing from that symmetry.  The optional
structure keys of :mod:`spindoctor.sim.forward.haze_structure` break it --
tilting the haze's own illumination axis, scaling the falloff length by
hemisphere or by azimuth, scaling one hemisphere's brightness, ramping the
disc interior, and painting clouds.  They are strictly opt-in: a block naming
none of them leaves :attr:`AtmosphereSpec.structure` at None and every
quantity below stays the scalar it always was.

A body without an ``atmosphere`` block never calls into this module and
renders hard-limbed, byte-for-byte as before.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.sim.ellipsoid_geometry import illumination_vector
from spindoctor.sim.forward.haze_structure import (
    HazeStructure,
    apply_disc_structure,
    apply_hemispheric_scaling,
    haze_structure_from_params,
    scale_height_field,
    tilted_illumination_2d,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'AtmosphereLayers',
    'AtmosphereSpec',
    'HaloScreen',
    'apply_atmosphere',
    'atmosphere_spec_from_params',
    'hg_phase_factor',
]

# Single-scattering brightness scale of a fully lit, optically thick grazing
# haze column (before the phase factor and illumination weight).  The haze
# emergent intensity is clipped into the [0, 1] signal plane, so a strong
# forward-scattering peak saturates by design.
_HAZE_ALBEDO = 0.6

# Amplitude of a detached shell's Gaussian tau bump, as a multiple of
# tau_ref; the shell's radial width is one scale height.
_SHELL_AMPLITUDE = 1.0

# The tangent optical depth below which the above-limb glow is treated as
# black: it bounds the rendered halo to a limb band a few scale heights deep
# (and, with the on-disc annulus, sets the bounding box the haze cost scales
# with) rather than the whole frame.
_TAU_EPS = 1e-3

# Floor on cos(emission) for the on-disc slant path, so the limb (where the
# emergent-cosine vanishes) saturates smoothly instead of dividing by zero.
_MU_FLOOR = 1e-3

# Largest exponent the column arithmetic evaluates.  Every optical depth below
# reaches the image only through ``1 - exp(-tau)``, which saturates at 1 long
# before this, so clipping the exponent changes no rendered value while
# keeping a very short falloff length (a structured haze at the far side of
# the disc) from overflowing to infinity.
_MAX_EXP_ARG = 700.0


@dataclass(frozen=True)
class HaloScreen:
    """The translucent above-limb haze layer of one atmospheric body.

    A transmission screen in the ring system's sense: the radiance stage
    composites ``img = emission + transmission * img_behind`` over the
    screen's pixels, and point sources (which sit at infinity, behind every
    halo) multiply by ``transmission``.

    Parameters:
        emission: Per-pixel tangent glow of the halo, in [0, 1]; 0 where the
            halo carries no light.
        transmission: Per-pixel ``exp(-tau)`` along the grazing line of
            sight: the fraction of background light that passes through;
            1 outside the halo.
        box_v: Grid rows outside which the screen is exactly identity
            (emission 0, transmission 1), so consumers may restrict their
            compositing to the box without changing any value.
        box_u: Grid columns of the same bounding box.
        mask: Box-sized mask of the pixels where the screen differs from
            identity, so consumers need not re-derive it from the maps.
    """

    emission: NDArrayFloatType
    transmission: NDArrayFloatType
    box_v: slice
    box_u: slice
    mask: NDArrayBoolType


@dataclass(frozen=True)
class AtmosphereLayers:
    """One atmospheric body's haze, split by compositing role.

    Parameters:
        disc: The body radiance with the on-disc haze composited, in [0, 1].
            Zero exactly where the haze-free render was zero outside the
            silhouette, so it paints (opaquely) the same pixels a
            no-atmosphere body would.
        halo: The translucent above-limb screen, covering every band pixel
            the opaque paint does not own.
    """

    disc: NDArrayFloatType
    halo: HaloScreen


@dataclass(frozen=True)
class AtmosphereSpec:
    """The exponential haze layer of one atmospheric body.

    All pixel quantities are in units of the render grid the haze is
    composited on (the oversampled grid when the scene oversamples), matching
    the body's already-scaled semi-axes.

    The detached shell's tau bump peaks at a fixed multiple of ``tau_ref``,
    not of the smooth column depth at ``detached_px``, so the same smooth
    atmosphere re-expressed at a different ``ref_altitude_px`` (with
    ``tau_ref`` rescaled accordingly) carries a correspondingly rescaled
    shell.  The reference-altitude invariance of the smooth column is
    therefore a parameterization choice that deliberately excludes the
    shell: a spec with ``detached_px`` set is tied to its stated reference
    altitude.

    Parameters:
        scale_height_px: Haze e-folding scale height in pixels (> 0).
        tau_ref: Tangent optical depth at ``ref_altitude_px`` (> 0).
        ref_altitude_px: Altitude above the reference radius at which the
            tangent optical depth equals ``tau_ref``, in pixels.
        g: Henyey-Greenstein asymmetry parameter in (-1, 1); positive is
            forward-scattering (bright limb at high phase).
        detached_px: Altitude of an optional detached haze shell above the
            surface, in pixels; None for no shell.
        structure: The optional symmetry-breaking structure (see
            :mod:`spindoctor.sim.forward.haze_structure`); None for the
            plain mirror-symmetric column, which is what every quantity
            here reduces to.
    """

    scale_height_px: float
    tau_ref: float
    ref_altitude_px: float = 0.0
    g: float = 0.0
    detached_px: float | None = None
    structure: HazeStructure | None = None


def atmosphere_spec_from_params(
    body_params: dict[str, object], *, oversample: int
) -> AtmosphereSpec | None:
    """Build an :class:`AtmosphereSpec` from a body's ``atmosphere`` block.

    Returns None when the body carries no atmosphere, so a body without the
    block never enters the haze path.  Pixel lengths are scaled to the render
    grid by ``oversample`` exactly as the body's axes are.

    Parameters:
        body_params: One scene body entry.
        oversample: The render grid's oversampling factor.

    Returns:
        The scaled spec, or None when the body has no ``atmosphere`` block.
    """
    atmosphere = body_params.get('atmosphere')
    if not isinstance(atmosphere, dict):
        return None
    os_factor = max(1, int(oversample))
    detached = atmosphere.get('detached_px')
    return AtmosphereSpec(
        scale_height_px=float(atmosphere['scale_height_px']) * os_factor,
        tau_ref=float(atmosphere['tau_ref']),
        ref_altitude_px=float(atmosphere.get('ref_altitude_px', 0.0)) * os_factor,
        g=float(atmosphere.get('g', 0.0)),
        detached_px=(float(detached) * os_factor if detached is not None else None),
        structure=haze_structure_from_params(atmosphere, oversample=os_factor),
    )


def hg_phase_factor(g: float, phase_angle: float) -> float:
    """The Henyey-Greenstein phase factor at a phase angle (1 at ``g`` = 0).

    Single scattering turns light through a scattering angle
    ``Theta = pi - phase``, so ``cos(Theta) = -cos(phase)``; a positive ``g``
    peaks at ``Theta`` = 0 (phase = pi), which is why forward-scattering haze
    is brightest at high phase.  The factor is normalized to 1 at ``g`` = 0
    for every phase, so it is a pure angular modulation of the haze source.

    Parameters:
        g: Henyey-Greenstein asymmetry parameter in (-1, 1).
        phase_angle: Phase angle in radians.

    Returns:
        The multiplicative phase factor (> 0).
    """
    cos_theta = -math.cos(phase_angle)
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
    return float((1.0 - g * g) / max(denom, 1e-12))


def tangent_optical_depth(
    h_px: NDArrayFloatType,
    spec: AtmosphereSpec,
    *,
    scale_height: float | NDArrayFloatType | None = None,
) -> NDArrayFloatType:
    """Tangent (slant) optical depth of the haze at tangent altitude ``h_px``.

    The exponential column plus, when the spec carries one, a Gaussian
    detached-shell bump one scale height wide centred at ``detached_px``.

    Parameters:
        h_px: Tangent altitude above the reference radius, in pixels.
        spec: The haze spec.
        scale_height: Falloff length to use, overriding the spec's nominal
            one; a per-pixel array when the spec's structure varies it with
            hemisphere or azimuth.  None uses the nominal value, which is
            what a structureless haze always does.

    Returns:
        The tangent optical depth at each altitude.
    """
    height = max(spec.scale_height_px, 1e-6) if scale_height is None else scale_height
    tau = spec.tau_ref * np.exp(np.minimum(-(h_px - spec.ref_altitude_px) / height, _MAX_EXP_ARG))
    if spec.detached_px is not None:
        bump = (h_px - spec.detached_px) / height
        tau = tau + spec.tau_ref * _SHELL_AMPLITUDE * np.exp(
            np.maximum(-0.5 * bump * bump, -_MAX_EXP_ARG)
        )
    return cast(NDArrayFloatType, tau)


def _scale_height_bounds(spec: AtmosphereSpec) -> tuple[float, float]:
    """The smallest and largest falloff length the spec produces anywhere.

    A structureless haze has one length everywhere, so both bounds are it.
    A structured haze scales it by hemisphere and by azimuth, and the two
    ends serve different purposes: the above-limb band must reach as far as
    the LONGEST length carries the glow, while the on-disc band's inner edge
    must reach as deep as the SHORTEST length makes the vertical column (a
    shorter falloff means a denser atmosphere at the surface, hence a
    grazing excess detectable further in).  Both are real: a hemispheric
    ratio below one and a negative sharpness gradient each shorten the
    length over part of the disc.

    Parameters:
        spec: The haze spec.

    Returns:
        ``(min_scale_height_px, max_scale_height_px)``, both positive.
    """
    nominal = max(spec.scale_height_px, 1e-6)
    if spec.structure is None:
        return nominal, nominal
    return (
        nominal * spec.structure.min_falloff_factor,
        nominal * spec.structure.max_falloff_factor,
    )


def _outer_altitude(spec: AtmosphereSpec) -> float:
    """The tangent altitude beyond which the above-limb glow is negligible.

    Where the smooth column has fallen to :data:`_TAU_EPS`, plus a detached
    shell's reach (its centre and three scale heights).  Evaluated at the
    longest falloff length the spec produces, so a structured haze's
    extended hemisphere is bounded rather than clipped at an artificial
    edge.

    Parameters:
        spec: The haze spec.

    Returns:
        The bounding tangent altitude in pixels.
    """
    _, scale_height = _scale_height_bounds(spec)
    reach = spec.ref_altitude_px + scale_height * math.log(max(spec.tau_ref / _TAU_EPS, 1.0))
    if spec.detached_px is not None:
        reach = max(reach, spec.detached_px + 3.0 * scale_height)
    return max(reach, scale_height)


def _vertical_optical_depth(
    spec: AtmosphereSpec, r_mean: float, scale_height: float | NDArrayFloatType
) -> float | NDArrayFloatType:
    """The physical vertical column depth the on-disc slant paths scale from.

    The SURFACE tangent depth divided by the grazing enhancement
    ``sqrt(2 pi R / H)``.  ``tau_ref`` is the tangent depth at
    ``ref_altitude_px``, so the surface tangent depth carries the
    ``exp(ref_altitude_px / H)`` factor; dropping it would leave the disc
    side that many e-foldings weaker than the limb implies.

    Parameters:
        spec: The haze spec.
        r_mean: Mean apparent body radius in grid pixels.
        scale_height: The falloff length -- a scalar for a structureless
            haze, a per-pixel array when the structure varies it.

    Returns:
        The vertical optical depth, matching the shape of ``scale_height``.
    """
    geom = np.sqrt(2.0 * math.pi * r_mean / scale_height)
    tau_surface = spec.tau_ref * np.exp(
        np.minimum(spec.ref_altitude_px / scale_height, _MAX_EXP_ARG)
    )
    return cast(NDArrayFloatType, tau_surface / np.maximum(geom, 1e-6))


def _band_bbox(
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    reach_a: float,
    reach_b: float,
    cos_rz: float,
    sin_rz: float,
) -> tuple[slice, slice] | None:
    """Grid slices bounding the outer band ellipse of the haze.

    The band's outer boundary in centred pixel coordinates satisfies
    ``|v*cos_rz - u*sin_rz| <= reach_a`` and ``|v*sin_rz + u*cos_rz| <=
    reach_b`` (the rotated-frame extents of the ellipse, with the tilt's
    foreshortening already divided out of ``reach_a``), so its axis-aligned
    bounding box follows by rotating those extents back.  One pixel of slack
    absorbs the half pixel between the coordinates the extents are stated in
    and the cells the slices name; every pixel outside the returned slices lies
    strictly outside the band.

    Parameters:
        shape: The render-grid shape ``(V, U)``.
        center_v: Body centre v in grid pixels.
        center_u: Body centre u in grid pixels.
        reach_a: Rotated-frame half-extent along semi-axis a, in pixels.
        reach_b: Rotated-frame half-extent along semi-axis b, in pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        ``(v_slice, u_slice)`` into the grid, or None when the box misses
        the grid entirely (the haze contributes nothing).
    """
    size_v, size_u = shape
    reach_v = reach_a * abs(cos_rz) + reach_b * abs(sin_rz)
    reach_u = reach_a * abs(sin_rz) + reach_b * abs(cos_rz)
    v0 = max(math.floor(center_v - reach_v) - 1, 0)
    v1 = min(math.ceil(center_v + reach_v) + 1, size_v)
    u0 = max(math.floor(center_u - reach_u) - 1, 0)
    u1 = min(math.ceil(center_u + reach_u) + 1, size_u)
    if v0 >= v1 or u0 >= u1:
        return None
    return slice(v0, v1), slice(u0, u1)


def apply_atmosphere(
    body_shape: NDArrayFloatType,
    spec: AtmosphereSpec,
    *,
    center_v: float,
    center_u: float,
    semi_a: float,
    semi_b: float,
    semi_c: float,
    rotation_z: float,
    rotation_tilt: float,
    illumination_angle: float,
    phase_angle: float,
) -> AtmosphereLayers:
    """Evaluate the haze layer over a reference-centred body radiance.

    The haze is evaluated over a limb band a few scale heights deep and split
    by compositing role: the on-disc haze (every band pixel the disc render
    painted, including its anti-aliased rim) is added to the opaque disc
    radiance, and the above-limb glow outside the painted silhouette becomes
    the translucent :class:`HaloScreen`.  All per-pixel work (coordinate
    grids included) is restricted to the bounding box of the body plus its
    halo out to the detached shell's reach, so the haze cost scales with
    that box, not the frame.  The returned arrays are new arrays; the input
    is never mutated (it may be a shared render cache entry).

    Parameters:
        body_shape: The shaded body radiance at the reference centre, in
            [0, 1], 0 outside the body.
        spec: The haze spec (pixel lengths already on this grid).
        center_v: Body centre v the shape was rendered at, in grid pixels.
        center_u: Body centre u the shape was rendered at, in grid pixels.
        semi_a: Semi-axis a in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Depth semi-axis c in grid pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        rotation_tilt: Tilt toward/away from the viewer, radians.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians (0 fully lit, pi backlit).

    Returns:
        The :class:`AtmosphereLayers`: the opaque disc radiance (clipped to
        [0, 1]) and the translucent halo screen.
    """
    out = np.array(body_shape, dtype=np.float64, copy=True)
    emission = np.zeros_like(out)
    transmission = np.ones_like(out)
    empty = AtmosphereLayers(
        disc=out,
        halo=HaloScreen(
            emission=emission,
            transmission=transmission,
            box_v=slice(0, 0),
            box_u=slice(0, 0),
            mask=np.zeros((0, 0), dtype=np.bool_),
        ),
    )
    semi_a = max(semi_a, 1e-6)
    semi_b = max(semi_b, 1e-6)
    r_mean = 0.5 * (semi_a + semi_b)
    scale_height = max(spec.scale_height_px, 1e-6)

    cos_rz = math.cos(rotation_z)
    sin_rz = math.sin(rotation_z)
    cos_rt = math.cos(rotation_tilt)

    # Restrict all heavy work to a limb band: a halo out to where the tangent
    # glow vanishes, and, on the disc, an annulus in from the limb to where the
    # grazing-excess column drops below _TAU_EPS.  Deep disc interior carries
    # no haze (the on-disc opacity is the excess of the slant path over the
    # nadir path, zero at disc centre), though for a thick haze the annulus
    # can span most of the disc; the guaranteed cost bound is the bounding
    # box computed below.
    #
    # tau_vert (see _vertical_optical_depth) is the physical vertical column
    # depth the on-disc slant paths scale from.  Both band edges are sized
    # from the extreme falloff lengths the spec produces -- the longest for
    # the outer reach, the deepest column for the inner edge -- so a
    # structured haze's hemispheres are bounded, never clipped.
    height_min, height_max = _scale_height_bounds(spec)
    tau_vert_max = max(
        float(_vertical_optical_depth(spec, r_mean, height_min)),
        float(_vertical_optical_depth(spec, r_mean, height_max)),
    )
    e_outer = 1.0 + _outer_altitude(spec) / r_mean
    # Inside, the excess column tau_vert * (1 / mu - 1) drops below _TAU_EPS
    # once cos(emission) exceeds mu_cut; that sets the inner edge of the band.
    mu_cut = 1.0 / (1.0 + _TAU_EPS / max(tau_vert_max, 1e-12))
    e_inner2 = max(0.0, 1.0 - mu_cut * mu_cut)

    # Coordinate grids and every per-pixel term below are built only inside
    # the axis-aligned bounding box of the outer band ellipse (e2 <=
    # e_outer**2, the body plus its halo out to the detached shell's reach),
    # so the grid-build cost scales with that box, not the frame.  The box is
    # conservative -- every pixel outside it has e2 > e_outer**2 -- so
    # restricting the band mask to it is exact.
    size_v, size_u = out.shape
    box = _band_bbox(
        (size_v, size_u),
        center_v=center_v,
        center_u=center_u,
        reach_a=e_outer * semi_a / max(abs(cos_rt), 1e-6),
        reach_b=e_outer * semi_b,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
    )
    if box is None:
        return empty
    box_v, box_u = box
    # The box is named by array rows and columns and the body center is stated
    # in the geometry layer's pixel corner coordinates, so the half pixel goes
    # on before the center comes off.
    v_ctr = (
        np.arange(box_v.start, box_v.stop, dtype=np.float64)
        + (PIXEL_CENTER_TO_CORNER_PX - center_v)
    )[:, None]
    u_ctr = (
        np.arange(box_u.start, box_u.stop, dtype=np.float64)
        + (PIXEL_CENTER_TO_CORNER_PX - center_u)
    )[None, :]

    v_rot = (v_ctr * cos_rz - u_ctr * sin_rz) * cos_rt
    u_rot = v_ctr * sin_rz + u_ctr * cos_rz
    e2 = (v_rot / semi_a) ** 2 + (u_rot / semi_b) ** 2

    # The band splits at the silhouette: the on-disc annulus (added to the
    # opaque paint) and the above-limb halo band, each evaluated on its own
    # gathered pixels.
    band_in = (e2 >= e_inner2) & (e2 < 1.0)
    band_out = (e2 >= 1.0) & (e2 <= e_outer * e_outer)
    if not band_in.any() and not band_out.any():
        return empty

    # The haze lights from its OWN axis, which a structure carrying
    # axis_tilt_deg rotates away from the body's geometric sun direction --
    # so the rendered mirror plane is not the one a navigator derives from
    # the predicted sub-solar point.  The body's own shading is untouched.
    structure = spec.structure
    haze_illumination_angle = illumination_angle + (
        0.0 if structure is None else math.radians(structure.axis_tilt_deg)
    )
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=haze_illumination_angle, phase_angle=phase_angle
    )
    # The in-plane unit direction toward the light, phase foreshortening
    # divided out: the structure fields are image-plane geometry (which way
    # is sunward on the disc), not illumination strength, so they must not
    # collapse with sin(phase) the way the 3-D vector above does.
    axis_v, axis_u = tilted_illumination_2d(structure, illumination_angle)
    hg = hg_phase_factor(spec.g, phase_angle)
    # Terminator-wrap width: the horizon-dip angle sqrt(2 H / R), the solar
    # depression at which a point one scale height above the surface loses
    # direct sunlight.  A modeling choice (a physical wrap would integrate
    # the column's shadowing), floored at 0.05 rad so a razor-thin
    # atmosphere still brightens resolvably past the terminator.
    delta_wrap = max(math.sqrt(2.0 * scale_height / r_mean), 0.05)

    def _source(mu0: NDArrayFloatType) -> NDArrayFloatType:
        """The single-scattering source from the solar elevation weight."""
        elevation = np.arcsin(np.clip(mu0, -1.0, 1.0))
        illum_weight = 0.5 * (1.0 + np.tanh(elevation / delta_wrap))
        return cast(NDArrayFloatType, _HAZE_ALBEDO * hg * illum_weight)

    out_box = out[box_v, box_u]

    # On-disc annulus: the solar elevation comes from the surface incidence,
    # and the opacity is the grazing EXCESS over the nadir column:
    # tau_vert * (1 / mu - 1), zero at disc centre and diverging toward the
    # limb, so the haze concentrates in a limb band.  The haze joins the
    # opaque paint (the disc is opaque anyway).
    if band_in.any():
        e2_in = e2[band_in]
        mu0_in = _disc_incidence(
            v_rot[band_in],
            u_rot[band_in],
            e2_in,
            semi_a=semi_a,
            semi_b=semi_b,
            semi_c=semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
            illum_v=illum_v,
            illum_u=illum_u,
            illum_z=illum_z,
        )
        mu_emit = np.sqrt(np.maximum(1.0 - e2_in, 0.0))
        height_in = scale_height_field(
            structure,
            scale_height,
            v_rot=v_rot[band_in],
            v_ctr=np.broadcast_to(v_ctr, e2.shape)[band_in],
            u_ctr=np.broadcast_to(u_ctr, e2.shape)[band_in],
            illum_v=axis_v,
            illum_u=axis_u,
        )
        tau_vert_in = _vertical_optical_depth(spec, r_mean, height_in)
        excess = tau_vert_in * (1.0 / np.maximum(mu_emit, _MU_FLOOR) - 1.0)
        haze_in = _source(mu0_in) * (1.0 - np.exp(-excess))
        out_box[band_in] = np.clip(out_box[band_in] + haze_in, 0.0, 1.0)

    # Above-limb band: the solar elevation comes from the limb-azimuth
    # incidence (there is no surface, only the atmospheric column standing
    # above the limb at that azimuth) and the opacity from the tangent
    # column.  Band pixels the disc render painted (its anti-aliased rim)
    # stay opaque paint and take their haze additively, so the painted
    # silhouette is exactly the no-atmosphere one; the rest become the
    # translucent halo screen the radiance stage composites over whatever
    # lies behind.
    halo_mask = np.zeros_like(band_out)
    if band_out.any():
        v_out = np.broadcast_to(v_ctr, e2.shape)[band_out]
        u_out = np.broadcast_to(u_ctr, e2.shape)[band_out]
        rho = np.hypot(v_out, u_out)
        rho = np.maximum(rho, 1e-9)
        mu0_out = (v_out * illum_v + u_out * illum_u) / rho
        h_out = (np.sqrt(e2[band_out]) - 1.0) * r_mean
        height_out = scale_height_field(
            structure,
            scale_height,
            v_rot=v_rot[band_out],
            v_ctr=v_out,
            u_ctr=u_out,
            illum_v=axis_v,
            illum_u=axis_u,
        )
        opacity_out = 1.0 - np.exp(-tangent_optical_depth(h_out, spec, scale_height=height_out))
        haze_out = _source(mu0_out) * opacity_out

        shape_out = out_box[band_out]
        solid = shape_out > 0.0
        halo = ~solid
        shape_out[solid] = np.clip(shape_out[solid] + haze_out[solid], 0.0, 1.0)
        out_box[band_out] = shape_out
        emission_vals = np.zeros_like(haze_out)
        emission_vals[halo] = np.clip(haze_out[halo], 0.0, 1.0)
        emission[box_v, box_u][band_out] = emission_vals
        # Halo pixels lie outside the silhouette, where the opacity is
        # 1 - exp(-tau) of the tangent column, so the screen transmission is
        # its complement.
        transmission_vals = np.ones_like(haze_out)
        transmission_vals[halo] = 1.0 - opacity_out[halo]
        transmission[box_v, box_u][band_out] = transmission_vals
        halo_mask[band_out] = halo

    # The disc-side structure rides on top of the composited haze: an axial
    # brightness ramp and any clouds, both confined to the painted
    # silhouette, then a hemispheric brightness scaling of everything --
    # disc radiance and halo glow alike -- so the two halves the mirror maps
    # onto each other differ by a pure scaling.  Transmission is opacity,
    # not brightness, and is deliberately left alone.  The whole block is
    # skipped for a structureless haze: not for tidiness but because every
    # line of it is per-pixel work the cold-render budget does not carry.
    if structure is not None:
        apply_disc_structure(
            out_box,
            structure,
            disc=out_box > 0.0,
            v_ctr=v_ctr,
            u_ctr=u_ctr,
            r_mean=r_mean,
            illum_v=axis_v,
            illum_u=axis_u,
        )
        apply_hemispheric_scaling(out_box, structure, v_rot)
        np.clip(out_box, 0.0, 1.0, out=out_box)
        emission_box = emission[box_v, box_u]
        apply_hemispheric_scaling(emission_box, structure, v_rot)
        np.clip(emission_box, 0.0, 1.0, out=emission_box)
    return AtmosphereLayers(
        disc=out,
        halo=HaloScreen(
            emission=emission,
            transmission=transmission,
            box_v=box_v,
            box_u=box_u,
            mask=halo_mask,
        ),
    )


def _disc_incidence(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    semi_a: float,
    semi_b: float,
    semi_c: float,
    cos_rz: float,
    sin_rz: float,
    illum_v: float,
    illum_u: float,
    illum_z: float,
) -> NDArrayFloatType:
    """Cosine of the surface incidence angle at on-disc haze pixels.

    The ellipsoid surface normal in image coordinates (rotated back from the
    body frame) dotted with the illumination direction; shared shading
    conventions are not required here because the haze never crosses the
    information boundary, but the same normal construction keeps the haze
    aligned with the disc shading.

    Parameters:
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        e2: Squared normalized ellipse radial function at each pixel.
        semi_a: Semi-axis a in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Depth semi-axis c in grid pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        illum_v: V component of the unit illumination direction.
        illum_u: U component of the unit illumination direction.
        illum_z: Z component of the unit illumination direction.

    Returns:
        ``cos(incidence)`` per pixel (negative on the night side).
    """
    z = semi_c * np.sqrt(np.maximum(1.0 - e2, 0.0))
    nv_local = v_rot / (semi_a * semi_a)
    nu_local = u_rot / (semi_b * semi_b)
    nz_local = z / (semi_c * semi_c)
    mag = np.sqrt(nv_local**2 + nu_local**2 + nz_local**2)
    mag = np.maximum(mag, 1e-12)
    nv_local /= mag
    nu_local /= mag
    nz_local /= mag
    normal_v = nv_local * cos_rz + nu_local * sin_rz
    normal_u = -nv_local * sin_rz + nu_local * cos_rz
    return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + nz_local * illum_z)
