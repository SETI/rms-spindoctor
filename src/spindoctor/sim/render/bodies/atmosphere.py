"""Image-side atmosphere rendering for haze-limb (Titan-class) bodies.

A body carrying an ``atmosphere`` block gains an exponential haze layer above
its surface, evaluated by :func:`apply_atmosphere` after the disc is shaded.
The haze splits into two compositing components (:class:`AtmosphereLayers`):
the on-disc haze joins the body's opaque paint, which it dims (the disc is
opaque anyway, so the painted silhouette stays exactly the no-atmosphere
silhouette), while
the above-limb glow is a translucent :class:`HaloScreen` -- an emission map
plus a per-pixel transmission ``exp(-tau)`` -- that the radiance stage
composites over the background exactly as the ring system's transmission
screen: ``img = glow + exp(-tau) * img_behind``, with point sources
attenuated by the same factor.  The body itself then goes over its own screen by
its coverage, so a limb pixel the outline covers by a fraction ``c`` shows the disc
in that fraction and the glow, and what lies behind it, in the rest.  A star
behind the halo therefore dims by the tangent transmission instead of vanishing,
and the body's mask / depth / occlusion truth sees only the solid silhouette,
never the halo.  The haze is
a truth key the navigator never sees: its predicted-body model keeps a hard
limb at the reference radius, so the soft rendered limb is a designed
mismatch (the substrate for the Titan altitude-versus-phase problem).

**Tangent optical depth.**  A line of sight grazing the body at tangent
altitude ``h`` (pixels above the reference radius) accumulates a slant
optical depth

    ``tau(h) = tau_ref * exp(-(h - ref_altitude_px) / scale_height_px)``

so ``tau_ref`` is the tangent optical depth at ``ref_altitude_px``.  An
optional detached haze shell adds a Gaussian bump in ``tau`` centered at
``detached_px`` above the surface.  The shell exists only in this above-limb
tangent depth: the on-disc column is shell-blind (a geometrically
thin shell projected against the disc adds negligible slant contrast), so
the shell renders solely as a second band in the tangent glow.

**Scattering.**  The haze reflects sunlight as a layer of finite optical depth,
single and multiple scattering together, by the reflectance of
:mod:`spindoctor.sim.render.bodies.haze_scattering` (Hapke's isotropic
multiple-scattering approximation, scaled for a finite column).  It takes the
block's single-scattering ``albedo`` (:data:`DEFAULT_HAZE_ALBEDO` when the scene
leaves it out) and a Henyey-Greenstein phase factor ``P`` (forward-scattering when
``g`` > 0, which brightens the limb at high phase and, in the limit, produces the
ring of light past phase 150 deg), with an optional second term (``g2`` with the
share ``g2_fraction``) for a two-term phase function such as the fit to Titan's
haze described in that module.  The image is in I/F, where a white Lambert disc
lit face-on is 1, so a thin column of optical depth ``tau`` along the line of
sight reflects ``albedo * P * tau / 4``, the single-scattering result, and a thick
one reflects what a semi-infinite haze would.  Past the terminator the haze fades
over the horizon-dip angle ``sqrt(2 * H / R)`` of solar depression -- the
depression at which a column one scale height up loses direct sunlight -- floored
at 0.05 rad so a razor-thin atmosphere still wraps resolvably (so the terminator
brightens past 90 deg incidence instead of cutting off).

The column a line of sight crosses depends on where it goes.  Above the limb it
is the tangent optical depth, so the limb becomes a soft exponential ramp whose
apparent radius grows with the haze brightness (hence with phase).  On the disc
it is the slant column from the top of the haze down to the surface, the
vertical depth

    ``tau_vert = tau_ref * exp(ref_altitude_px / H) / sqrt(2 * pi * R / H)``

(the surface tangent depth divided by the grazing enhancement, ``R`` the mean
radius and ``H`` the scale height in pixels) over the effective emission cosine,
which is half the surface tangent depth at the limb.  The surface beneath is
seen through the haze: its brightness is multiplied by the direct transmission
``exp(-tau_vert * (1 / mu + 1 / mu0))`` of the sunlight coming in and the light
going out, with ``mu`` and ``mu0`` the effective emission and incidence cosines.
So a thin haze adds light over the disc that grows toward the limb, and a haze
several optical depths thick hides the surface altogether: the disc and its edge
are then haze, and the solid limb does not show, because a thick column
reflects the same light just inside the limb and just outside it.  Both sides
describe one atmosphere at any reference altitude.

The haze reflects from the level where the tangent optical depth is 1 (the
surface, when the column is thinner): the emission and incidence cosines of its
reflectance are those of the body scaled out to that level, and a line of sight
that misses the scaled body grazes it.  A thick haze therefore looks like a
sphere that much larger, its brightness varying across the solid limb as
smoothly as anywhere else, and a thin one reflects from the surface.  Each
column is evaluated at one point rather than integrated along its curved path,
so at high phase the near side of a thick tangent path, which real sunlight
reaches only through the atmosphere behind the limb, is lit as fully as the
tangent point.

**Symmetry-breaking structure.**  The column described above is exactly
mirror-symmetric about the sunward axis, which would make it useless for
grading a navigator that measures pointing from that symmetry.  The optional
structure keys of :mod:`spindoctor.sim.render.bodies.haze` break it --
tilting the haze's own illumination axis, scaling the falloff length by
hemisphere or by azimuth, scaling one hemisphere's brightness, ramping the
disc interior, and painting clouds.  They are strictly opt-in: a block naming
none of them leaves :attr:`AtmosphereSpec.structure` at None and every
quantity below stays the scalar it always was.

A body without an ``atmosphere`` block never calls into this module and
renders hard-limbed, byte-for-byte as before.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import ndimage

from spindoctor.sim.render.bodies.haze import (
    HazeStructure,
    apply_disc_structure,
    apply_hemispheric_scaling,
    haze_structure_from_params,
    hemisphere_weight,
    scale_height_field,
    tilted_illumination_2d,
)
from spindoctor.sim.render.bodies.haze_geometry import (
    band_bbox,
    column_latitude_sine,
    scaled_body_cosines,
)
from spindoctor.sim.render.bodies.haze_scattering import (
    chandrasekhar_h,
    effective_cosine,
    grazing_constant,
    haze_reflectance,
    mean_cosine,
    twilight_weight,
)
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidGeometry,
    outline_semi_a,
)
from spindoctor.sim.shared.lighting import henyey_greenstein_phase, illumination_vector
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DEFAULT_HAZE_ALBEDO',
    'AtmosphereLayers',
    'AtmosphereSpec',
    'HaloScreen',
    'apply_atmosphere',
    'atmosphere_spec_from_params',
]

DEFAULT_HAZE_ALBEDO: float = 0.6
"""The haze ``albedo`` a scene's ``atmosphere`` block takes when it leaves the key out.

The albedo is the haze particles' single-scattering albedo and lies in [0, 1].  A
fully lit, optically thick grazing haze column reflects ``albedo * P / 4`` in I/F,
where ``P`` is the phase factor.  The haze's emergent intensity is not capped, so a
strong forward-scattering peak can carry the signal above 1, and the detector
saturates there as a real camera does.
"""

# Amplitude of a detached shell's Gaussian tau bump, as a multiple of
# tau_ref; the shell's radial width is one scale height.
_SHELL_AMPLITUDE = 1.0

# The 1 / 4 of the single-scattering source albedo * P / 4.  Sunlight of flux
# pi * F scatters into albedo * P / (4 * pi) of it per steradian per unit of
# optical depth, with P averaging 1 over the sphere, and I/F divides the
# emergent intensity by F.
_SINGLE_SCATTERING_FACTOR = 0.25

# The faintest glow, and the faintest dimming of a full-scale background, the
# halo carries before it ends.  It bounds the rendered halo to a limb band
# several scale heights deep (the bounding box the haze cost scales with)
# rather than the whole frame, and it sits far below one DN of the finest
# emulated ADC word (12 bits, 2.4e-4 of full scale), so the halo's outer edge
# doesn't show as a step however bright a forward-scattering haze makes it.
_HALO_EPS = 1e-5

# Floor on the Henyey-Greenstein denominator.  As ``|g|`` approaches 1 the
# denominator vanishes at phase 0 or pi, so the floor keeps the phase factor
# finite and the halo bounded instead of dividing by zero.
_HG_DENOMINATOR_FLOOR = 1e-12  # See #1057.

# Largest exponent the column arithmetic evaluates.  Every optical depth below
# reaches the image only through ``exp(-tau)``, which vanishes long before
# this, so clipping the exponent changes no rendered value while
# keeping a very short falloff length (a structured haze at the far side of
# the disc) from overflowing to infinity.
_MAX_EXP_ARG = 700.0

# The eight neighbors of a pixel and the pixel itself.  The halo screen reaches from
# every pixel the outline leaves partly open to these, the pixels the linear
# interpolation that moves the body into place reads together with it.
_NEIGHBORS = np.ones((3, 3), dtype=np.bool_)


@dataclass(frozen=True)
class HaloScreen:
    """The translucent above-limb haze layer of one atmospheric body.

    A transmission screen in the ring system's sense: the radiance stage
    composites ``img = emission + transmission * img_behind`` over the
    screen's pixels, and point sources (which sit at infinity, behind every
    halo) multiply by ``transmission``.

    Parameters:
        emission: Per-pixel tangent glow of the halo, at least 0; 0 where the
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
        disc: The body radiance with the on-disc haze composited, weighted by the
            outline's coverage, at least 0.  Zero wherever the outline covers
            nothing, so it covers the same pixels a no-atmosphere body would.
        halo: The translucent above-limb screen, covering every band pixel the
            outline leaves partly or wholly open and the covered pixels beside
            them.
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
        tau_ref: Tangent optical depth at ``ref_altitude_px`` (>= 0; 0 is a clear
            atmosphere that adds no haze).
        ref_altitude_px: Altitude above the reference radius at which the
            tangent optical depth equals ``tau_ref``, in pixels.
        g: Henyey-Greenstein asymmetry parameter in (-1, 1) of the phase
            function's first term; positive is forward-scattering (bright limb
            at high phase).
        g2: Asymmetry parameter in (-1, 1) of the phase function's second term.
        g2_fraction: The share of the scattered light in the second term, in
            [0, 1]; 0 is a one-term Henyey-Greenstein phase function.
        detached_px: Altitude of an optional detached haze shell above the
            surface, in pixels; None for no shell.
        albedo: The haze's single-scattering albedo (at least 0); 0 is a purely
            absorbing haze, which still dims what lies behind the halo but adds no
            light.
        structure: The optional symmetry-breaking structure (see
            :mod:`spindoctor.sim.render.bodies.haze`); None for the
            plain mirror-symmetric column, which is what every quantity
            here reduces to.
    """

    scale_height_px: float
    tau_ref: float
    ref_altitude_px: float = 0.0
    g: float = 0.0
    g2: float = 0.0
    g2_fraction: float = 0.0
    detached_px: float | None = None
    albedo: float = DEFAULT_HAZE_ALBEDO
    structure: HazeStructure | None = None


def atmosphere_spec_from_params(
    atmosphere: Mapping[str, Any] | None, *, oversample: int
) -> AtmosphereSpec | None:
    """Build an :class:`AtmosphereSpec` from a body's ``atmosphere`` block.

    Returns None when the body carries no atmosphere, so a body without the
    block never enters the haze path.  Pixel lengths are scaled to the render
    grid by ``oversample`` exactly as the body's axes are.  An optional key that
    the block leaves out takes its default.

    Parameters:
        atmosphere: One body's ``atmosphere`` mapping, or None.
        oversample: The render grid's oversampling factor.

    Returns:
        The scaled spec, or None when ``atmosphere`` is not a mapping.
    """
    if not isinstance(atmosphere, Mapping):
        return None
    os_factor = max(1, int(oversample))
    detached = atmosphere.get('detached_px')
    ref_altitude = atmosphere.get('ref_altitude_px')
    g = atmosphere.get('g')
    g2 = atmosphere.get('g2')
    g2_fraction = atmosphere.get('g2_fraction')
    albedo = atmosphere.get('albedo')
    return AtmosphereSpec(
        scale_height_px=float(atmosphere['scale_height_px']) * os_factor,
        tau_ref=float(atmosphere['tau_ref']),
        ref_altitude_px=(0.0 if ref_altitude is None else float(ref_altitude)) * os_factor,
        g=0.0 if g is None else float(g),
        g2=0.0 if g2 is None else float(g2),
        g2_fraction=0.0 if g2_fraction is None else float(g2_fraction),
        detached_px=(float(detached) * os_factor if detached is not None else None),
        albedo=DEFAULT_HAZE_ALBEDO if albedo is None else float(albedo),
        structure=haze_structure_from_params(atmosphere, oversample=os_factor),
    )


def tangent_optical_depth(
    h_px: NDArrayFloatType,
    spec: AtmosphereSpec,
    *,
    scale_height: float | NDArrayFloatType | None = None,
) -> NDArrayFloatType:
    """Tangent (slant) optical depth of the haze at tangent altitude ``h_px``.

    The exponential column plus, when the spec carries one, a Gaussian
    detached-shell bump one scale height wide centered at ``detached_px``.

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


def _outer_altitude(spec: AtmosphereSpec, *, peak_glow: float) -> float:
    """The tangent altitude beyond which the above-limb halo is negligible.

    The halo's glow is about ``source * tau`` where the column is thin, and it
    dims a background by about ``tau``, so both stay below :data:`_HALO_EPS`
    once the smooth column falls to ``_HALO_EPS / max(peak_glow, 1)``.  A
    brighter haze therefore reaches farther out, and its glow fades into the
    sky instead of ending at a bright edge.  A detached shell extends that to
    where its Gaussian bump falls to the same depth.  Evaluated at the longest
    falloff length the spec produces, so a structured haze's extended
    hemisphere is bounded rather than clipped at an artificial edge.

    Parameters:
        spec: The haze spec.
        peak_glow: The brightest glow the halo can carry: the bound of
            :func:`_source_scale` times any hemispheric brightening.

    Returns:
        The bounding tangent altitude in pixels.
    """
    _, scale_height = _scale_height_bounds(spec)
    tau_cut = _HALO_EPS / max(peak_glow, 1.0)
    reach = spec.ref_altitude_px + scale_height * math.log(max(spec.tau_ref / tau_cut, 1.0))
    if spec.detached_px is not None:
        shell_peak = spec.tau_ref * _SHELL_AMPLITUDE
        half_width = math.sqrt(2.0 * math.log(max(shell_peak / tau_cut, 1.0)))
        reach = max(reach, spec.detached_px + half_width * scale_height)
    return max(reach, scale_height)


def _phase_factor(spec: AtmosphereSpec, phase_angle: float) -> float:
    """The haze's phase function at the scene's phase angle, one or two terms.

    Parameters:
        spec: The haze spec.
        phase_angle: Phase angle in radians.

    Returns:
        ``(1 - f) * HG(g) + f * HG(g2)``, ``f`` being ``g2_fraction``.
    """
    first = henyey_greenstein_phase(spec.g, phase_angle, denominator_floor=_HG_DENOMINATOR_FLOOR)
    if spec.g2_fraction == 0.0:
        return first
    second = henyey_greenstein_phase(spec.g2, phase_angle, denominator_floor=_HG_DENOMINATOR_FLOOR)
    return (1.0 - spec.g2_fraction) * first + spec.g2_fraction * second


def _source_scale(spec: AtmosphereSpec, hg: float) -> float:
    """An upper bound on the I/F of a fully lit haze column.

    The single-scattering source ``albedo * P / 4`` plus the most the multiple
    scattering adds, ``albedo * (H(1)**2 - 1) / 4`` at the similarity albedo (see
    :func:`~spindoctor.sim.render.bodies.haze_scattering.haze_reflectance`); the
    geometric factor ``mu0 / (mu0 + mu)`` stays below 1.

    Parameters:
        spec: The haze spec.
        hg: The phase function at the scene's phase angle.

    Returns:
        The bound in I/F.
    """
    g_mean = mean_cosine(spec.g, spec.g2, spec.g2_fraction)
    similar_albedo = spec.albedo * (1.0 - g_mean) / max(1.0 - spec.albedo * g_mean, 1e-12)
    h_overhead = float(chandrasekhar_h(np.array(1.0), similar_albedo))
    return spec.albedo * (hg + h_overhead * h_overhead - 1.0) * _SINGLE_SCATTERING_FACTOR


def _outer_scale(spec: AtmosphereSpec, *, hg: float, r_mean: float) -> float:
    """The normalized ellipse radius of the halo's outer edge (1 on the limb).

    Parameters:
        spec: The haze spec.
        hg: The phase function at the scene's phase angle.
        r_mean: Mean apparent body radius in grid pixels.

    Returns:
        One plus the outer altitude (:func:`_outer_altitude`) over the mean radius.
    """
    structure = spec.structure
    brightening = 1.0 + max(0.0, 0.0 if structure is None else structure.ns_asymmetry_amplitude)
    peak_glow = _source_scale(spec, hg) * brightening
    return 1.0 + _outer_altitude(spec, peak_glow=peak_glow) / r_mean


def halo_reach(
    spec: AtmosphereSpec,
    *,
    semi_a: float,
    semi_b: float,
    semi_c: float,
    rotation_z: float,
    sub_observer_lat: float,
    phase_angle: float,
) -> tuple[float, float]:
    """How far a body's haze layer reaches from the body center along each image axis.

    The reach bounds the box :func:`apply_atmosphere` evaluates the haze in, less
    its one pixel of slack, so a canvas this large around the body center holds the
    whole halo.

    Parameters:
        spec: The haze spec (pixel lengths already on the grid).
        semi_a: Semi-axis a, along the pole, in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Semi-axis c in grid pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        sub_observer_lat: The sub-observer latitude, radians.
        phase_angle: Phase angle in radians.

    Returns:
        The reach along v and along u, in grid pixels.
    """
    semi_a = max(semi_a, 1e-6)
    semi_b = max(semi_b, 1e-6)
    hg = _phase_factor(spec, phase_angle)
    e_outer = _outer_scale(spec, hg=hg, r_mean=0.5 * (semi_a + semi_b))
    reach_a = e_outer * outline_semi_a(semi_a, semi_c, sub_observer_lat)
    reach_b = e_outer * semi_b
    cos_rz = abs(math.cos(rotation_z))
    sin_rz = abs(math.sin(rotation_z))
    return reach_a * cos_rz + reach_b * sin_rz, reach_a * sin_rz + reach_b * cos_rz


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
    sub_observer_lat: float,
    illumination_angle: float,
    phase_angle: float,
    silhouette: NDArrayFloatType,
) -> AtmosphereLayers:
    """Evaluate the haze layer over a reference-centered body radiance.

    The haze is evaluated over the disc and a band a few scale heights deep above
    the limb, and split by compositing role.  At every pixel the outline covers,
    the disc radiance is dimmed by the haze column's direct transmission and the
    haze's own light is added, weighted by the coverage.  The above-limb glow
    becomes the translucent :class:`HaloScreen` wherever the outline leaves any of
    a pixel open, and also on the wholly covered pixels next to those.  The radiance
    stage composites the body over its own screen by coverage, so a limb pixel shows
    the glow in the fraction the outline leaves open.  All per-pixel work (coordinate
    grids included) is restricted to the bounding box of the body plus its
    halo out to the detached shell's reach, so the haze cost scales with
    that box, not the frame.  The returned arrays are new arrays; the input
    is never mutated (it may be a shared render cache entry).

    Parameters:
        body_shape: The shaded body radiance at the reference center, weighted
            by the outline's coverage, so 0 outside the body.
        spec: The haze spec (pixel lengths already on this grid).
        center_v: Body center v the shape was rendered at, in grid pixels.
        center_u: Body center u the shape was rendered at, in grid pixels.
        semi_a: Semi-axis a, along the pole, in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Semi-axis c in grid pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        sub_observer_lat: The sub-observer latitude, radians; the haze uses the
            outline and surface the shared projection gives
            (:func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid`).
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians (0 fully lit, pi backlit).
        silhouette: The fraction of each pixel the body's solid outline covers at
            the reference center.  A pixel the outline covers takes on-disc haze
            even where it renders black.

    Returns:
        The :class:`AtmosphereLayers`: the disc radiance, weighted by the coverage,
        and the translucent halo screen.
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
    geometry = EllipsoidGeometry(
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_c,
        cos_lat=math.cos(sub_observer_lat),
        sin_lat=math.sin(sub_observer_lat),
    )
    semi_a_outline = outline_semi_a(semi_a, semi_c, sub_observer_lat)

    structure = spec.structure
    hg = _phase_factor(spec, phase_angle)
    e_outer = _outer_scale(spec, hg=hg, r_mean=r_mean)

    # The whole disc carries haze, the column down to the surface never being
    # zero.  Coordinate grids and every per-pixel term below are built only inside
    # the axis-aligned bounding box of the outer band ellipse (e2 <=
    # e_outer**2, the body plus its halo out to the detached shell's reach),
    # so the grid-build cost scales with that box, not the frame.  The box is
    # conservative -- every pixel outside it has e2 > e_outer**2 -- so
    # restricting the band mask to it is exact.
    size_v, size_u = out.shape
    box = band_bbox(
        (size_v, size_u),
        center_v=center_v,
        center_u=center_u,
        reach_a=e_outer * semi_a_outline,
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

    v_rot = v_ctr * cos_rz - u_ctr * sin_rz
    u_rot = v_ctr * sin_rz + u_ctr * cos_rz
    e2 = (v_rot / semi_a_outline) ** 2 + (u_rot / semi_b) ** 2

    out_box = out[box_v, box_u]
    outline_box = silhouette[box_v, box_u]

    # The band splits into the disc and the above-limb band, each evaluated on
    # its own gathered pixels.  The disc radiance is weighted by the outline's
    # coverage, so a covered pixel takes its coverage times the haze.  The
    # above-limb band is the tangent column, evaluated for two uses.  A covered
    # pixel whose center lies outside the reference ellipse (relief standing
    # above it) is surface seen through the near half of that column.  The halo
    # screen takes the whole column wherever the outline leaves any of a pixel
    # open, a notch where the limb relief dips below the ellipse included, and
    # on the wholly covered pixels beside those.  The body covers its own
    # screen when the two are composited, so a pixel shows the glow in the
    # fraction the outline leaves open, and the wholly covered pixels give the
    # glow a value inside the limb for the translation into place to blend
    # with.
    covered = outline_box > 0.0
    band_in = (e2 < 1.0) & covered
    within_outer = e2 <= e_outer * e_outer
    solid_out = (e2 >= 1.0) & within_outer & covered
    halo_mask = within_outer & ndimage.binary_dilation(outline_box < 1.0, structure=_NEIGHBORS)
    band_out = solid_out | halo_mask
    if not band_in.any() and not band_out.any():
        return empty

    # The haze lights from its OWN axis, which a structure carrying
    # axis_tilt_deg rotates away from the body's geometric sun direction --
    # so the rendered mirror plane is not the one a navigator derives from
    # the predicted sub-solar point.  The body's own shading is untouched.
    # The hemispheric structure terms blend across the equator by the latitude
    # of the haze column each pixel sees.  Only those two terms read the
    # weight, so every other haze skips building it.
    weight = (
        hemisphere_weight(column_latitude_sine(v_rot, u_rot, e2, geometry=geometry))
        if structure is not None and structure.splits_hemispheres
        else None
    )
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
    # Terminator-wrap width: the horizon-dip angle sqrt(2 H / R), the solar
    # depression at which a point one scale height above the surface loses
    # direct sunlight.  A modeling choice (a physical wrap would integrate
    # the column's shadowing), floored at 0.05 rad so a razor-thin
    # atmosphere still brightens resolvably past the terminator.
    delta_wrap = max(math.sqrt(2.0 * scale_height / r_mean), 0.05)

    def _column(
        tau_los: NDArrayFloatType,
        cosines: tuple[NDArrayFloatType, NDArrayFloatType],
        surface: tuple[NDArrayFloatType, NDArrayFloatType],
        height: float | NDArrayFloatType,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """The haze's I/F over a column and the column's direct transmission.

        ``cosines`` are the emission and incidence cosines at the level the haze
        reflects from and ``surface`` those at the solid surface beneath; each
        enters through the effective cosine the column's curvature gives.
        """
        c = grazing_constant(height, r_mean)
        mu_eff = effective_cosine(cosines[0], c)
        mu0_eff = effective_cosine(cosines[1], c)
        tau = np.minimum(tau_los, _MAX_EXP_ARG)
        haze = haze_reflectance(
            tau,
            mu_eff,
            mu0_eff,
            albedo=spec.albedo,
            g=mean_cosine(spec.g, spec.g2, spec.g2_fraction),
            phase_function=hg,
        ) * twilight_weight(cosines[1], delta_wrap)
        mu_surface = effective_cosine(surface[0], c)
        mu0_surface = effective_cosine(surface[1], c)
        direct = np.exp(-np.minimum(tau * (1.0 + mu_surface / mu0_surface), _MAX_EXP_ARG))
        return cast(NDArrayFloatType, haze), cast(NDArrayFloatType, direct)

    gathered: dict[int, tuple[NDArrayFloatType, ...]] = {}

    def _gather(pixels: NDArrayBoolType) -> tuple[NDArrayFloatType, ...]:
        """The rotated coordinates, ``e2``, and limb incidence of some pixels, once."""
        key = id(pixels)
        if key not in gathered:
            v_px = np.broadcast_to(v_ctr, e2.shape)[pixels]
            u_px = np.broadcast_to(u_ctr, e2.shape)[pixels]
            rho = np.maximum(stable_hypot(v_px, u_px), 1e-9)
            limb_mu0 = (v_px * illum_v + u_px * illum_u) / rho
            gathered[key] = (v_rot[pixels], u_rot[pixels], e2[pixels], limb_mu0)
        return gathered[key]

    def _incidence(
        pixels: NDArrayBoolType, scale: float | NDArrayFloatType
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """The emission and incidence cosines of some pixels on the body scaled by ``scale``."""
        v_rot_px, u_rot_px, e2_px, limb_mu0 = _gather(pixels)
        return scaled_body_cosines(
            v_rot_px,
            u_rot_px,
            e2_px,
            scale=scale,
            limb_incidence=limb_mu0,
            geometry=geometry,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
            illum_v=illum_v,
            illum_u=illum_u,
            illum_z=illum_z,
        )

    def _reflecting_scale(height: float | NDArrayFloatType) -> float | NDArrayFloatType:
        """The size of the level the haze reflects from, relative to the body.

        The level where the tangent optical depth is 1, or the surface when the
        column is thinner than that.
        """
        altitude = np.maximum(
            spec.ref_altitude_px + np.asarray(height) * math.log(max(spec.tau_ref, 1e-300)), 0.0
        )
        return cast(float | NDArrayFloatType, 1.0 + altitude / r_mean)

    # The disc: the column runs from the top of the haze down to the surface,
    # the vertical depth over the effective emission cosine.  The haze reflects
    # from the level where the tangent column is 1 (the surface, for a thinner
    # haze), so a thick haze looks like a sphere that size and the solid limb
    # does not show.  The surface shows through the column's direct
    # transmission, and the haze's own light joins it, weighted by the
    # coverage.
    if band_in.any():
        height_in = scale_height_field(
            structure,
            scale_height,
            weight=None if weight is None else weight[band_in],
            v_ctr=np.broadcast_to(v_ctr, e2.shape)[band_in],
            u_ctr=np.broadcast_to(u_ctr, e2.shape)[band_in],
            illum_v=axis_v,
            illum_u=axis_u,
        )
        surface_in = _incidence(band_in, 1.0)
        tau_vert_in = _vertical_optical_depth(spec, r_mean, height_in)
        tau_in = tau_vert_in / effective_cosine(surface_in[0], grazing_constant(height_in, r_mean))
        haze_in, direct_in = _column(
            tau_in, _incidence(band_in, _reflecting_scale(height_in)), surface_in, height_in
        )
        out_box[band_in] *= direct_in
        _add_covered_haze(out_box, outline_box, band_in, haze_in)

    # Above-limb band: the line of sight grazes the column at the tangent
    # altitude.  The halo pixels take the whole tangent column and become the
    # translucent screen the radiance stage composites behind the body, over
    # whatever lies behind it.  Covered pixels outside the ellipse are surface
    # seen through the near half of the column, which is what the disc's column
    # reaches at the limb, and join the opaque paint by their coverage, so the
    # painted silhouette is exactly the no-atmosphere one.
    if band_out.any():
        h_out = (np.sqrt(e2[band_out]) - 1.0) * r_mean
        # The halo on a pixel the outline covers is seen only through the part
        # the outline leaves open, which lies above the limb, so it takes the
        # glow and the dimming at the limb, never those of a column inside it.
        h_out = np.where(covered[band_out], np.maximum(h_out, 0.0), h_out)
        height_out = scale_height_field(
            structure,
            scale_height,
            weight=None if weight is None else weight[band_out],
            v_ctr=np.broadcast_to(v_ctr, e2.shape)[band_out],
            u_ctr=np.broadcast_to(u_ctr, e2.shape)[band_out],
            illum_v=axis_v,
            illum_u=axis_u,
        )
        tau_out = tangent_optical_depth(h_out, spec, scale_height=height_out)
        height_each = np.broadcast_to(np.asarray(height_out, dtype=np.float64), tau_out.shape)

        solid = solid_out[band_out]
        if solid.any():
            haze_solid, direct_solid = _column(
                0.5 * tau_out[solid],
                _incidence(solid_out, _reflecting_scale(height_each[solid])),
                _incidence(solid_out, 1.0),
                height_each[solid],
            )
            out_box[solid_out] *= direct_solid
            _add_covered_haze(out_box, outline_box, solid_out, haze_solid)

        halo = halo_mask[band_out]
        haze_halo, _direct = _column(
            tau_out[halo],
            _incidence(halo_mask, _reflecting_scale(height_each[halo])),
            _incidence(halo_mask, 1.0),
            height_each[halo],
        )
        emission[box_v, box_u][halo_mask] = haze_halo
        # The screen's transmission is exp(-tau) of the tangent column.
        transmission[box_v, box_u][halo_mask] = np.exp(-np.minimum(tau_out[halo], _MAX_EXP_ARG))

    # The disc-side structure rides on top of the composited haze: an axial
    # brightness ramp and any clouds, both confined to the painted
    # silhouette, then a hemispheric brightness scaling of everything --
    # disc radiance and halo glow alike -- so poleward of the equatorial
    # blend band the two halves the mirror maps onto each other differ by a
    # pure scaling.  The floor at 0 that follows catches a pixel a negative
    # ramp or cloud pushed below 0.
    # Transmission is opacity, not brightness, and is deliberately left
    # alone.  The whole block is skipped for a structureless haze: not for
    # tidiness but because every line of it is per-pixel work the
    # cold-render budget does not carry.
    if structure is not None:
        apply_disc_structure(
            out_box,
            structure,
            coverage=outline_box,
            v_ctr=v_ctr,
            u_ctr=u_ctr,
            r_mean=r_mean,
            illum_v=axis_v,
            illum_u=axis_u,
        )
        apply_hemispheric_scaling(out_box, structure, weight)
        np.maximum(out_box, 0.0, out=out_box)
        emission_box = emission[box_v, box_u]
        apply_hemispheric_scaling(emission_box, structure, weight)
        np.maximum(emission_box, 0.0, out=emission_box)
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


def _add_covered_haze(
    out_box: NDArrayFloatType,
    coverage: NDArrayFloatType,
    pixels: NDArrayBoolType,
    haze: NDArrayFloatType,
) -> None:
    """Add haze to the coverage-weighted disc radiance at some pixels, in place.

    Each pixel gains its coverage times the haze.  The sum is not capped, so bright
    haze over a bright disc can pass 1.

    Parameters:
        out_box: The disc radiance over the haze box, weighted by the coverage.
        coverage: The fraction of each box pixel the body's outline covers.
        pixels: The box pixels that take haze.
        haze: The haze brightness at each of those pixels, in their array order.
    """
    out_box[pixels] += coverage[pixels] * haze
