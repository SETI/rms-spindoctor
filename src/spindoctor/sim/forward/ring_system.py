"""Image-side optical-depth ring-system renderer.

Draws the ``ring_system`` scene block: radial tau features (ringlets, gaps,
one-sided edges, ramps, and damped density-wave trains) on mode-1 eccentric
precessing orbits with optional m >= 2 modes and satellite edge waves,
projected through the shared opening-angle geometry
(:mod:`spindoctor.sim.ring_geometry`) and lit by the single-scattering
closed forms.  The output is a set of per-pixel maps -- emitted intensity,
transmission ``exp(-tau/mu)``, and line-of-sight depth -- that the radiance
stage composites against the body stack as a transmission screen:
``img = I_ring + exp(-tau/mu) * img_behind``, evaluated far to near per
pixel, so low-tau features reveal the background instead of erasing it and
stars behind the ring attenuate physically.

Radial tau profiles by feature kind (``r_e(lam)`` the feature's perturbed
orbit radius, ``w`` its radial width, distances in ring-plane radial units):

- ``ringlet``: ``tau`` between ``r_e`` and ``r_e + w`` (anti-aliased edges).
- ``gap``: the same band as a tau suppression (subtracts, clipped at zero).
- ``edge``: a one-sided step; ``side: 'in'`` carries ``tau`` for
  ``r <= r_e`` (a sheet bounded by its outer edge, the B-ring-edge case),
  ``side: 'out'`` for ``r >= r_e``.
- ``ramp``: a linear transition across ``[r_e, r_e + w]``; ``side: 'out'``
  rises from 0 at ``r_e`` to ``tau`` at ``r_e + w`` (compose with an
  ``edge`` there for a sheet with a gradual inner boundary), ``side: 'in'``
  mirrors it.  Zero outside the band; the sharp end is anti-aliased.
- ``wave``: a damped radial sinusoid launched at ``r_e``:
  ``dtau(x) = tau * exp(-x / damping) * sin(2*pi*x / wavelength)`` for
  ``x = r - r_e >= 0`` and exactly zero upstream (the clamp mirrors the
  azimuthal edge wave's: the envelope grows without bound for ``x < 0``).
  A density-wave train is visual clutter riding on a sheet; its negative
  lobes subtract from the composed tau.

Photometry (the normative equation set), with ``mu = |sin B_obs|``,
``mu0 = |sin B_sun|``, lit iff ``sign(B_obs) == sign(B_sun)``, and one-term
Henyey-Greenstein ``P(g, alpha)``:

- lit:   ``I = A/4 * P * mu0/(mu0 + mu) * (1 - exp(-tau*(1/mu0 + 1/mu)))``
- unlit: ``I = A/4 * P * mu0/(mu0 - mu) * (exp(-tau/mu0) - exp(-tau/mu))``,
  with the limit ``A/4 * P * (tau/mu) * exp(-tau/mu)`` when
  ``|mu0 - mu| < 1e-6``.

An opening angle of exactly 0 (either side) renders nothing.  The unlit
branch produces the real inversion: moderate-tau features bright from the
dark side, high-tau features nearly black.

The albedo ``A`` and asymmetry ``g`` are per-feature truth keys; where
features overlap radially the composed emission uses the tau-weighted mean
of their ``A/4 * P`` factors over the positive (ringlet) contributions,
which reduces exactly to the closed form for any single feature.

Truth-side clutter: the ``azimuthal`` block (brightness modulation, a
planet-shadow wedge, seeded spokes) scales the emitted intensity without
touching tau or transmission, and the ``moonlets`` list embeds opaque discs
at the ring's depth, each optionally carrying a propeller tau disturbance.

Each feature is evaluated only inside its radial annulus: the orbit's exact
radial bounds, widened by the kind's shape extent and an anti-aliasing
margin, map through the projection to a pixel bounding box, and outside the
bounded band every contribution is exactly 0.0 (clipped shades, the ramp's
clipped fraction, the wave envelope's float64 underflow).  The bounding
changes rendering cost with feature count, never the rendered values.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from spindoctor.sim.ring_geometry import (
    RingOrbit,
    compute_antialiasing_shade,
    compute_orbit_radii,
    ring_los_depth,
    ring_orbit_from_mapping,
    ring_plane_from_sky,
    ring_radial_scale,
    ring_sky_from_plane,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'RING_ALBEDO_DEFAULT',
    'RING_PHASE_G_DEFAULT',
    'RingSystemMaps',
    'henyey_greenstein_phase',
    'render_ring_system',
    'ring_reflection_factor',
]

# Single-scattering albedo (times normalization) default; dusty features set
# their own higher values per feature.
RING_ALBEDO_DEFAULT: float = 0.5
# Henyey-Greenstein asymmetry default: main-ring backscatter.  Dusty,
# forward-scattering features set g ~ +0.6 per feature.
RING_PHASE_G_DEFAULT: float = -0.3
# Below this |mu0 - mu| the unlit closed form is numerically indeterminate
# (0/0) and the analytic limit replaces it.
_UNLIT_MU_MATCH_TOL: float = 1e-6
# exp(-x) underflows to exactly 0.0 in float64 before x reaches 750, so a
# wave train's tau is bitwise zero beyond this many damping lengths; the
# radial bound uses it to stay exact rather than approximate.
_WAVE_UNDERFLOW_DAMPINGS: float = 750.0
# exp(-0.5 * x**2) underflows to exactly 0.0 in float64 once x passes about
# 38.7 (0.5 * x**2 > ~745.2), so a propeller's Gaussian lobes are bitwise
# zero beyond this many sigmas; the radial bound uses it to stay exact.
_GAUSSIAN_UNDERFLOW_SIGMAS: float = 40.0


@dataclass(frozen=True)
class RingSystemMaps:
    """Per-pixel maps of a rendered ring system on the (oversampled) grid.

    Parameters:
        intensity: Emitted ring brightness ``I`` per pixel (normalized signal
            units).
        transmission: ``exp(-tau/mu)`` per pixel: the fraction of background
            light (bodies behind the ring, stars, sky) that passes through.
        mask: Pixels where the system carries any optical depth.
        depth_km: Observer distance ``range_km - dlos_km`` per pixel, or None
            when the scene gives the system no ``range_km`` (the system then
            has no depth relation to bodies; overlap is a scene error).
    """

    intensity: NDArrayFloatType
    transmission: NDArrayFloatType
    mask: NDArrayBoolType
    depth_km: NDArrayFloatType | None


def henyey_greenstein_phase(g: float, alpha_deg: float) -> float:
    """One-term Henyey-Greenstein phase function at phase angle ``alpha``.

    ``P = (1 - g**2) / (1 + g**2 + 2*g*cos(alpha))**1.5`` with alpha the
    phase angle (0 at opposition): negative ``g`` backscatters (bright at
    low phase), positive ``g`` forward-scatters (dusty features brighten
    strongly toward alpha = 180).

    Parameters:
        g: Asymmetry parameter, strictly inside (-1, 1).
        alpha_deg: Phase angle in degrees, [0, 180].

    Returns:
        The phase function value (positive).
    """
    cos_alpha = math.cos(math.radians(alpha_deg))
    return float((1.0 - g * g) / (1.0 + g * g + 2.0 * g * cos_alpha) ** 1.5)


def ring_reflection_factor(
    tau: NDArrayFloatType,
    mu: float,
    mu0: float,
    *,
    lit: bool,
) -> NDArrayFloatType:
    """The geometric factor of the single-scattering closed forms.

    Multiplied by ``A/4 * P`` this is the emitted intensity: the lit form
    saturates toward ``mu0/(mu0 + mu)`` at high tau, while the unlit form
    peaks at moderate tau and falls to zero for an opaque ring (nothing
    diffuses through) -- the real dark-side inversion.

    Parameters:
        tau: Normal optical depth per pixel (non-negative).
        mu: ``|sin B_obs|``, nonzero.
        mu0: ``|sin B_sun|``, nonzero.
        lit: Whether the observer sees the lit face
            (``sign(B_obs) == sign(B_sun)``).

    Returns:
        The per-pixel geometric factor.
    """
    if lit:
        return mu0 / (mu0 + mu) * (1.0 - np.exp(-tau * (1.0 / mu0 + 1.0 / mu)))
    if abs(mu0 - mu) < _UNLIT_MU_MATCH_TOL:
        # The mu0 -> mu limit of the unlit form (the closed form is 0/0).
        return (tau / mu) * np.exp(-tau / mu)
    return mu0 / (mu0 - mu) * (np.exp(-tau / mu0) - np.exp(-tau / mu))


def render_ring_system(
    shape: tuple[int, int],
    ring_system: Mapping[str, Any],
    *,
    center_v: float,
    center_u: float,
    node_deg: float,
    time: float = 0.0,
    epoch: float = 0.0,
    oversample: int = 1,
    spokes_seed: int = 0,
) -> RingSystemMaps:
    """Render a ring system's per-pixel intensity/transmission/depth maps.

    The caller resolves the sky placement (planted offset, spacecraft-
    ephemeris parallax, camera roll -- a roll rotates the projected pattern,
    which is exactly ``node_deg`` plus the roll angle) and passes the final
    center and node; orbit radii and widths are detector-pixel scene values
    scaled to the render grid here.

    Parameters:
        shape: The (oversampled) render-grid shape ``(V*os, U*os)``.
        ring_system: The validated scene ``ring_system`` mapping.
        center_v: Ring-system center v on the render grid (offset applied).
        center_u: Ring-system center u on the render grid (offset applied).
        node_deg: Sky position angle of the ascending node, in degrees
            (camera roll already added).
        time: Scene time in TDB seconds (mode-1 pericenter precession).
        epoch: Ring epoch in TDB seconds.
        oversample: The render-grid oversampling factor; radii, widths, and
            the anti-aliasing window scale by it, and the depth map converts
            back through it so ``depth_km`` is grid-independent.
        spokes_seed: Seed for the azimuthal spoke field's realization
            (derived by the caller from the scene seed via the
            'scene_radiance/ring_system/spokes' stream).

    Returns:
        The rendered :class:`RingSystemMaps`.

    Raises:
        ValueError: If a feature's kind is not a renderable profile (the
            validator rejects these; this guards direct callers).
    """
    size_v, size_u = shape
    geometry = ring_system.get('geometry') or {}
    b_obs = float(geometry.get('opening_deg_obs', 0.0))
    b_sun = float(geometry.get('opening_deg_sun', 0.0))
    zero_maps = RingSystemMaps(
        intensity=np.zeros(shape, dtype=np.float64),
        transmission=np.ones(shape, dtype=np.float64),
        mask=np.zeros(shape, dtype=np.bool_),
        depth_km=None,
    )
    # An exactly edge-on ring plane (either angle) renders nothing.
    if b_obs == 0.0 or b_sun == 0.0:
        return zero_maps
    features = ring_system.get('features') or []
    moonlets = ring_system.get('moonlets') or []
    if not features and not moonlets:
        return zero_maps

    os = int(oversample)
    mu = abs(math.sin(math.radians(b_obs)))
    mu0 = abs(math.sin(math.radians(b_sun)))
    lit = (b_obs > 0.0) == (b_sun > 0.0)
    alpha_deg = float(ring_system.get('phase_deg', 0.0))

    # Pixel centres relative to the projected ring centre, mapped back to the
    # ring plane through the shared inverse projection.  The grid starts as
    # array rows and columns and the projection works in the geometry layer's
    # pixel corner coordinates, where the scene states its centre, so the half
    # pixel goes on here.
    v_coords = np.arange(size_v, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_coords = np.arange(size_u, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')
    dv = v_grid - center_v
    du = u_grid - center_u
    r, lam, x, y = ring_plane_from_sky(dv, du, opening_deg_obs=b_obs, node_deg=node_deg)
    # Ring-plane radial distance per image pixel: dividing by this converts a
    # radial distance to image pixels, so anti-aliased edges span one detector
    # pixel regardless of the foreshortening direction.
    radial_scale = ring_radial_scale(r, x, y, opening_deg_obs=b_obs)

    # Compose the radial tau profile: emitting kinds add their tau, gaps
    # subtract (tau suppression), a wave's negative lobes subtract, and the
    # sum clips at zero.  The emission's A/4 * P factor is the tau-weighted
    # mean over the positive contributions, so a single feature reproduces
    # its closed form exactly.  Each feature is evaluated only inside its
    # radial annulus (see the module docstring): outside it every
    # contribution is exactly 0.0, so the bounding is a pure optimization.
    #
    # Anti-aliased boundaries reach at most (os/2) * radial_scale ring-plane
    # units past their edge before the shade clips to exactly 0.0, and
    # radial_scale <= 1/|sin B| everywhere, so this margin (twice the strict
    # bound, for safety) keeps the bounding exact.
    aa_margin = float(os) / mu
    tau_map = np.zeros(shape, dtype=np.float64)
    ap_weighted = np.zeros(shape, dtype=np.float64)
    ap_weight = np.zeros(shape, dtype=np.float64)
    for feature in features:
        kind = str(feature.get('kind'))
        # The catalog orbit with the planted (truth-side) ephemeris error
        # applied on this side only -- the navigator predicts from the
        # catalog values -- then scaled to the render grid.  lam is
        # ring-plane longitude, so every orbital angle lives in the
        # ring-plane frame (the node angle entered only the sky projection
        # above).
        orbit = _orbit_with_error(
            ring_orbit_from_mapping(feature.get('orbit') or {}), feature.get('orbit_error')
        ).scaled(float(os))
        lo, hi = _feature_radial_bounds(feature, orbit, os=os, margin=aa_margin)
        slices = _annulus_bbox_slices(
            shape,
            center_v=center_v,
            center_u=center_u,
            r_outer=hi,
            opening_deg_obs=b_obs,
            node_deg=node_deg,
        )
        if slices is None:
            continue
        sub_v, sub_u = slices
        r_sub = r[sub_v, sub_u]
        sel = (r_sub >= lo) & (r_sub <= hi)
        if not bool(np.any(sel)):
            continue
        contribution = _feature_tau_profile(
            feature,
            orbit,
            r=r_sub[sel],
            lam=lam[sub_v, sub_u][sel],
            radial_scale=radial_scale[sub_v, sub_u][sel],
            os=os,
            epoch=epoch,
            time=time,
        )
        tau_view = tau_map[sub_v, sub_u]
        if kind == 'gap':
            tau_view[sel] -= contribution
        else:
            tau_view[sel] += contribution
            albedo = float(feature.get('albedo', RING_ALBEDO_DEFAULT))
            phase_g = float(feature.get('phase_g', RING_PHASE_G_DEFAULT))
            ap = albedo / 4.0 * henyey_greenstein_phase(phase_g, alpha_deg)
            positive = np.clip(contribution, 0.0, None)
            ap_weighted_view = ap_weighted[sub_v, sub_u]
            ap_weighted_view[sel] += ap * positive
            ap_weight_view = ap_weight[sub_v, sub_u]
            ap_weight_view[sel] += positive
    np.clip(tau_map, 0.0, None, out=tau_map)

    # Propeller lobes are local density disturbances, so they modify the
    # composed tau before the photometry evaluates it.  Like the radial
    # features, each propeller is evaluated only inside its radial annulus:
    # beyond the Gaussian-underflow reach both lobes are bitwise 0.0, the
    # factor is exactly 1.0, and multiplying by it is the identity, so the
    # bounding changes cost, never values.
    for moonlet in moonlets:
        propeller = moonlet.get('propeller')
        if not propeller:
            continue
        a_m = float(moonlet.get('a', 0.0)) * os
        width = float(propeller.get('width_px', 0.0)) * os
        reach = width * (1.0 + _GAUSSIAN_UNDERFLOW_SIGMAS)
        slices = _annulus_bbox_slices(
            shape,
            center_v=center_v,
            center_u=center_u,
            r_outer=a_m + reach,
            opening_deg_obs=b_obs,
            node_deg=node_deg,
        )
        if slices is None:
            continue
        sub_v, sub_u = slices
        r_sub = r[sub_v, sub_u]
        sel = (r_sub >= a_m - reach) & (r_sub <= a_m + reach)
        if not bool(np.any(sel)):
            continue
        factor = _propeller_tau_factor(
            moonlet, propeller, r=r_sub[sel], lam=lam[sub_v, sub_u][sel], os=os
        )
        tau_view = tau_map[sub_v, sub_u]
        tau_view[sel] = tau_view[sel] * factor

    ap_map = np.divide(
        ap_weighted, ap_weight, out=np.zeros(shape, dtype=np.float64), where=ap_weight > 0.0
    )
    intensity = ap_map * ring_reflection_factor(tau_map, mu, mu0, lit=lit)
    transmission: NDArrayFloatType = np.exp(-tau_map / mu)
    mask: NDArrayBoolType = tau_map > 0.0

    # Azimuthal structure (modulation, planet-shadow wedge, seeded spokes)
    # is albedo/illumination structure: it scales the emitted intensity and
    # leaves the optical depth -- and so the transmission -- untouched.
    azimuthal = ring_system.get('azimuthal')
    if azimuthal:
        intensity *= _azimuthal_intensity_factor(
            azimuthal, r=r, lam=lam, os=os, spokes_seed=spokes_seed
        )

    # Moonlet discs: opaque point-like bodies at the ring's depth.  Each
    # disc replaces the ring emission with its own, extinguishes the
    # background through the transmission screen, and joins the composited
    # mask so a moonlet sitting in an empty gap still composites.  The
    # coverage is exactly 0.0 beyond half an anti-aliasing window outside
    # the disc, so each disc is evaluated only inside its sky bounding box
    # (outside it the compositing forms are the identity); the bounding
    # changes cost, never values.
    for moonlet in moonlets:
        disc = _moonlet_disc_coverage(
            moonlet,
            shape,
            center_v=center_v,
            center_u=center_u,
            b_obs=b_obs,
            node_deg=node_deg,
            os=os,
        )
        if disc is None:
            continue
        (sub_v, sub_u), coverage = disc
        amplitude = float(moonlet.get('amplitude', 0.0))
        intensity[sub_v, sub_u] = intensity[sub_v, sub_u] * (1.0 - coverage) + amplitude * coverage
        transmission[sub_v, sub_u] = transmission[sub_v, sub_u] * (1.0 - coverage)
        mask[sub_v, sub_u] |= coverage > 0.0

    depth_km: NDArrayFloatType | None = None
    range_km = ring_system.get('range_km')
    if range_km is not None:
        # dlos is positive toward the observer on the render grid; divide by
        # the oversampling to return to detector pixels before the km scale.
        km_per_pixel = float(ring_system.get('km_per_pixel', 1.0))
        dlos_km = ring_los_depth(y, opening_deg_obs=b_obs) / float(os) * km_per_pixel
        depth_km = float(range_km) - dlos_km

    return RingSystemMaps(
        intensity=intensity,
        transmission=transmission,
        mask=mask,
        depth_km=depth_km,
    )


def _wrap_dlam(lam: NDArrayFloatType, center_rad: float) -> NDArrayFloatType:
    """Longitude differences wrapped into [-pi, pi)."""
    return cast(NDArrayFloatType, np.mod(lam - center_rad + math.pi, 2.0 * math.pi) - math.pi)


def _azimuthal_intensity_factor(
    azimuthal: Mapping[str, Any],
    *,
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    os: int,
    spokes_seed: int,
) -> NDArrayFloatType:
    """The truth-side azimuthal brightness factor (intensity only, never tau).

    - ``modulation``: ``1 + amplitude * cos(m * (lam - phase_deg))`` -- the
      self-gravity-wake quadrant asymmetry.
    - ``shadow``: a multiplicative ``1 - darkness`` wedge over
      ``[start_deg, start_deg + extent_deg)`` -- the planet-shadow boundary,
      a strong non-navigable edge crossing every feature.
    - ``spokes``: ``count`` seeded wedges, azimuthally sharp (quartic wedge
      profile) and radially broad (sin^2 envelope over ``r_inner..r_outer``),
      each with a drawn contrast in ``[0.5, 1] * contrast`` (negative for
      the dark low-phase appearance) and a drawn width in ``[0.5, 1.5] *
      width_deg``.

    The combined factor clips at zero: azimuthal structure can darken to
    black but never drive the emission negative.

    Parameters:
        azimuthal: The validated ``ring_system.azimuthal`` mapping.
        r: Per-pixel ring-plane radii on the render grid.
        lam: Per-pixel ring-plane longitudes (radians).
        os: The oversampling factor (scales the spoke radial band).
        spokes_seed: The spoke field's realization seed.

    Returns:
        The per-pixel intensity factor (non-negative).
    """
    factor = np.ones_like(r)
    modulation = azimuthal.get('modulation')
    if modulation:
        amplitude = float(modulation.get('amplitude', 0.0))
        m = int(modulation.get('m', 1))
        phase = math.radians(float(modulation.get('phase_deg', 0.0)))
        factor = factor * (1.0 + amplitude * np.cos(m * (lam - phase)))
    shadow = azimuthal.get('shadow')
    if shadow:
        start = math.radians(float(shadow.get('start_deg', 0.0)))
        extent = math.radians(float(shadow.get('extent_deg', 0.0)))
        darkness = float(shadow.get('darkness', 0.0))
        in_wedge = np.mod(lam - start, 2.0 * math.pi) < extent
        factor = factor * np.where(in_wedge, 1.0 - darkness, 1.0)
    spokes = azimuthal.get('spokes')
    if spokes:
        rng = np.random.default_rng(spokes_seed)
        count = int(spokes.get('count', 0))
        r_inner = float(spokes.get('r_inner', 0.0)) * os
        r_outer = float(spokes.get('r_outer', 0.0)) * os
        contrast = float(spokes.get('contrast', 0.0))
        width_deg = float(spokes.get('width_deg', 0.0))
        band = np.clip((r - r_inner) / max(r_outer - r_inner, 1e-12), 0.0, 1.0)
        radial_env = np.sin(math.pi * band) ** 2
        radial_env[(r < r_inner) | (r > r_outer)] = 0.0
        spoke_sum = np.zeros_like(r)
        for _ in range(count):
            lam_k = rng.uniform(0.0, 2.0 * math.pi)
            half_width = math.radians(width_deg * rng.uniform(0.5, 1.5)) / 2.0
            c_k = contrast * rng.uniform(0.5, 1.0)
            dlam = _wrap_dlam(lam, lam_k)
            wedge = np.clip(1.0 - (dlam / half_width) ** 4, 0.0, 1.0)
            spoke_sum += c_k * wedge
        factor = factor * (1.0 + spoke_sum * radial_env)
    clipped: NDArrayFloatType = np.clip(factor, 0.0, None)
    return clipped


def _propeller_tau_factor(
    moonlet: Mapping[str, Any],
    propeller: Mapping[str, Any],
    *,
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    os: int,
) -> NDArrayFloatType:
    """A propeller's multiplicative tau disturbance around its moonlet.

    Two Gaussian lobes straddle the moonlet radially by ``width_px`` and
    azimuthally by ``length_deg / 2`` in opposite senses (the leading lobe
    outward, the trailing lobe inward -- the stylized S shape of a real
    propeller), each with radial sigma ``width_px`` and azimuthal sigma
    ``length_deg / 4``.  ``contrast`` scales the disturbance: negative
    values carve the local partial gaps a physical propeller opens, positive
    values pile density up.  The factor clips at zero.

    Parameters:
        moonlet: The moonlet mapping (placement).
        propeller: The moonlet's ``propeller`` mapping.
        r: Per-pixel ring-plane radii on the render grid.
        lam: Per-pixel ring-plane longitudes (radians).
        os: The oversampling factor.

    Returns:
        The per-pixel multiplicative tau factor (non-negative).
    """
    a_m = float(moonlet.get('a', 0.0)) * os
    lam_m = math.radians(float(moonlet.get('lam_deg', 0.0)))
    length = math.radians(float(propeller.get('length_deg', 0.0)))
    width = float(propeller.get('width_px', 0.0)) * os
    contrast = float(propeller.get('contrast', 0.0))
    sigma_lam = length / 4.0
    disturbance = np.zeros_like(r)
    for radial_sign, azimuthal_sign in ((1.0, 1.0), (-1.0, -1.0)):
        dr = (r - (a_m + radial_sign * width)) / width
        dlam = _wrap_dlam(lam, lam_m + azimuthal_sign * length / 2.0) / sigma_lam
        disturbance += np.exp(-0.5 * (dr * dr + dlam * dlam))
    return cast(NDArrayFloatType, np.clip(1.0 + contrast * disturbance, 0.0, None))


def _moonlet_disc_coverage(
    moonlet: Mapping[str, Any],
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    b_obs: float,
    node_deg: float,
    os: int,
) -> tuple[tuple[slice, slice], NDArrayFloatType] | None:
    """A moonlet's anti-aliased disc coverage, bounded to its sky box.

    The moonlet sits in the ring plane at polar placement ``(a, lam_deg)``,
    projected to the sky through the shared projection; the disc is drawn in
    sky coordinates (a body, not a ring-plane band, so it does not
    foreshorten with the ring).  The coverage shade is exactly 0.0 wherever
    the pixel centre sits half an anti-aliasing window or more outside the
    disc radius, so the evaluation is restricted to the bounding box of that
    reach (plus one pixel of slack for the pixel-centre convention) and the
    caller composites only inside it -- an exact restriction, not an
    approximation.

    Parameters:
        moonlet: The moonlet mapping.
        shape: The render-grid shape ``(V, U)``.
        center_v: Ring-system center v on the render grid.
        center_u: Ring-system center u on the render grid.
        b_obs: Observer ring opening angle in degrees.
        node_deg: Sky position angle of the ascending node in degrees.
        os: The oversampling factor.

    Returns:
        ``((v_slice, u_slice), coverage)`` with the box-sized per-pixel
        coverage in [0, 1], or None when the box misses the grid entirely.
    """
    size_v, size_u = shape
    a_m = float(moonlet.get('a', 0.0)) * os
    lam_m = math.radians(float(moonlet.get('lam_deg', 0.0)))
    radius = float(moonlet.get('radius_px', 1.0)) * os
    dv, du = ring_sky_from_plane(
        np.asarray([a_m]), np.asarray([lam_m]), opening_deg_obs=b_obs, node_deg=node_deg
    )
    pos_v = center_v + float(dv[0])
    pos_u = center_u + float(du[0])
    reach = radius + float(os) / 2.0
    v0 = max(math.floor(pos_v - reach) - 1, 0)
    v1 = min(math.ceil(pos_v + reach) + 1, size_v)
    u0 = max(math.floor(pos_u - reach) - 1, 0)
    u1 = min(math.ceil(pos_u + reach) + 1, size_u)
    if v0 >= v1 or u0 >= u1:
        return None
    # The box is named by array rows and columns and the moonlet's position is
    # stated in the geometry layer's pixel corner coordinates, so the half
    # pixel goes on here.
    v_centers = np.arange(v0, v1, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_centers = np.arange(u0, u1, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    box_v, box_u = np.meshgrid(v_centers, u_centers, indexing='ij')
    dist = np.hypot(box_v - pos_v, box_u - pos_u)
    coverage = compute_antialiasing_shade(radius - dist, float(os))
    return (slice(v0, v1), slice(u0, u1)), coverage


def _orbit_with_error(orbit: RingOrbit, error: Mapping[str, Any] | None) -> RingOrbit:
    """The catalog orbit with the planted per-feature ephemeris error applied.

    ``orbit_error`` is a truth key: the render side displaces the drawn
    feature by these deltas while the navigator predicts from the unmodified
    catalog orbit, so features are misplaced relative to the model (and to
    each other) exactly as real ring features are relative to their
    published orbit solutions.  A negative ``delta_ae_px`` larger than the
    catalog ``ae`` clamps at a circular edge (a negative radial amplitude is
    not physical).

    Parameters:
        orbit: The parsed catalog orbit, in detector pixels.
        error: The feature's ``orbit_error`` mapping, or None.

    Returns:
        The orbit actually rendered.
    """
    if not error:
        return orbit
    return replace(
        orbit,
        a=orbit.a + float(error.get('delta_a_px', 0.0)),
        ae=max(0.0, orbit.ae + float(error.get('delta_ae_px', 0.0))),
        long_peri=orbit.long_peri + float(error.get('delta_long_peri_deg', 0.0)),
    )


def _orbit_radial_bounds(orbit: RingOrbit) -> tuple[float, float]:
    """Global bounds of :func:`compute_orbit_radii` over all longitudes.

    The mode-1 conic ranges over ``[a - ae, a + ae]`` exactly (pericenter to
    apocenter), and every m >= 2 mode and the edge wave perturb the radius
    by at most their amplitude in each direction.  The bounds hold at every
    epoch: precession moves the pericenter longitude, never the radial
    range.

    Parameters:
        orbit: The parsed orbit model (any consistent radial units).

    Returns:
        ``(lo, hi)`` bounds on the orbit's edge radius, same units.
    """
    reach = orbit.ae + sum(abs(mode.amp) for mode in orbit.modes)
    if orbit.edge_wave is not None:
        reach += abs(orbit.edge_wave.amp)
    return orbit.a - reach, orbit.a + reach


def _feature_radial_bounds(
    feature: Mapping[str, Any],
    orbit: RingOrbit,
    *,
    os: int,
    margin: float,
) -> tuple[float, float]:
    """The ring-plane radial band outside which a feature's tau is exactly 0.0.

    The orbit's radial bounds widen by the kind's shape extent -- the band
    width of a ringlet / gap / ramp, the unbounded side of a one-sided
    edge, and a wave train's decay length out to float64 underflow (beyond
    which its envelope is bitwise zero) -- plus ``margin``, which covers the
    anti-aliasing window's reach in ring-plane units.  Outside the returned
    band every profile form evaluates to exactly 0.0 (clipped shades, the
    ramp's clipped fraction, the wave's clipped upstream argument and
    underflowed envelope), so restricting evaluation to the band is exact.

    Parameters:
        feature: The validated feature mapping.
        orbit: The feature's orbit, scaled to the render grid.
        os: The oversampling factor (scales the shape keys).
        margin: The anti-aliasing safety margin in render-grid radial units.

    Returns:
        ``(lo, hi)`` in render-grid ring-plane units; ``hi`` is ``inf`` for
        an outward-unbounded edge and ``lo`` clips at zero.
    """
    kind = str(feature.get('kind'))
    lo, hi = _orbit_radial_bounds(orbit)
    if kind in ('ringlet', 'gap', 'ramp'):
        hi += float(feature.get('width', 0.0)) * os
    elif kind == 'edge':
        if str(feature.get('side', 'in')) == 'in':
            lo = -math.inf
        else:
            hi = math.inf
    elif kind == 'wave':
        hi += float(feature.get('damping', 0.0)) * os * _WAVE_UNDERFLOW_DAMPINGS
    return max(lo - margin, 0.0), hi + margin


def _annulus_bbox_slices(
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    r_outer: float,
    opening_deg_obs: float,
    node_deg: float,
) -> tuple[slice, slice] | None:
    """Grid slices bounding the sky projection of ring-plane radii <= r_outer.

    A ring-plane circle of radius ``r`` projects to a sky ellipse whose
    extents from the center are the longitude amplitudes of the projection
    equations: ``r * hypot(sin(node), sin(B) * cos(node))`` in v and
    ``r * hypot(cos(node), sin(B) * sin(node))`` in u.  Every pixel whose
    ring-plane radius is at most ``r_outer`` therefore lies inside this
    box; one pixel of slack absorbs the pixel-center convention.

    Parameters:
        shape: The render-grid shape ``(V, U)``.
        center_v: Projected ring-system center v on the render grid.
        center_u: Projected ring-system center u on the render grid.
        r_outer: Outer ring-plane radius bound (render-grid units); ``inf``
            selects the whole grid.
        opening_deg_obs: Observer ring opening angle B in degrees.
        node_deg: Sky position angle of the ascending node in degrees.

    Returns:
        ``(v_slice, u_slice)`` into the grid, or None when the bounding box
        misses the grid entirely (the feature contributes nothing).
    """
    size_v, size_u = shape
    if not math.isfinite(r_outer):
        return slice(0, size_v), slice(0, size_u)
    sin_b = math.sin(math.radians(opening_deg_obs))
    node = math.radians(node_deg)
    extent_v = r_outer * math.hypot(math.sin(node), sin_b * math.cos(node))
    extent_u = r_outer * math.hypot(math.cos(node), sin_b * math.sin(node))
    v0 = max(math.floor(center_v - extent_v) - 1, 0)
    v1 = min(math.ceil(center_v + extent_v) + 1, size_v)
    u0 = max(math.floor(center_u - extent_u) - 1, 0)
    u1 = min(math.ceil(center_u + extent_u) + 1, size_u)
    if v0 >= v1 or u0 >= u1:
        return None
    return slice(v0, v1), slice(u0, u1)


def _feature_tau_profile(
    feature: Mapping[str, Any],
    orbit: RingOrbit,
    *,
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    radial_scale: NDArrayFloatType,
    os: int,
    epoch: float,
    time: float,
) -> NDArrayFloatType:
    """One feature's per-pixel tau contribution (the module's profile forms).

    All radial arithmetic runs in render-grid ring-plane units (``orbit``
    arrives pre-scaled); dividing signed radial distances by ``radial_scale``
    converts them to image pixels so anti-aliased boundaries span one
    detector pixel regardless of foreshortening.  Only the ``wave`` kind can
    return negative values (its subtracting lobes); a ``gap``'s contribution
    is positive here and subtracted by the caller.

    Parameters:
        feature: The validated feature mapping.
        orbit: The feature's orbit, scaled to the render grid.
        r: Per-pixel ring-plane radii on the render grid.
        lam: Per-pixel ring-plane longitudes (radians).
        radial_scale: Per-pixel radial foreshortening scale.
        os: The oversampling factor.
        epoch: Ring epoch (TDB seconds).
        time: Scene time (TDB seconds).

    Returns:
        The per-pixel tau contribution.

    Raises:
        ValueError: If the feature's kind is not a renderable profile (the
            validator rejects these; this guards direct callers).
    """
    kind = str(feature.get('kind'))
    tau = float(feature.get('tau', 0.0))
    r_edge = compute_orbit_radii(lam, orbit, epoch=epoch, time=time)
    if kind in ('ringlet', 'gap'):
        width = float(feature.get('width', 0.0)) * os
        r_outer = compute_orbit_radii(lam, orbit.widened(width), epoch=epoch, time=time)
        inner_shade = compute_antialiasing_shade((r - r_edge) / radial_scale, float(os))
        outer_shade = compute_antialiasing_shade((r_outer - r) / radial_scale, float(os))
        return cast(NDArrayFloatType, tau * np.minimum(inner_shade, outer_shade))
    if kind == 'edge':
        side = str(feature.get('side', 'in'))
        signed = (r_edge - r) if side == 'in' else (r - r_edge)
        return tau * compute_antialiasing_shade(signed / radial_scale, float(os))
    if kind == 'ramp':
        width = float(feature.get('width', 0.0)) * os
        side = str(feature.get('side', 'out'))
        r_outer = compute_orbit_radii(lam, orbit.widened(width), epoch=epoch, time=time)
        x = np.clip((r - r_edge) / width, 0.0, 1.0)
        frac = x if side == 'out' else 1.0 - x
        # The linear transition is continuous at its zero end; the tau end is
        # a step, anti-aliased like any other sharp boundary.
        if side == 'out':
            sharp = compute_antialiasing_shade((r_outer - r) / radial_scale, float(os))
        else:
            sharp = compute_antialiasing_shade((r - r_edge) / radial_scale, float(os))
        return cast(NDArrayFloatType, tau * frac * sharp)
    if kind == 'wave':
        wavelength = float(feature.get('wavelength', 0.0)) * os
        damping = float(feature.get('damping', 0.0)) * os
        # Clamp to the downstream (outward) side: clipping x at zero makes
        # the upstream profile exactly zero (sin(0) = 0) and keeps the
        # envelope from growing without bound inward of the launch radius.
        x = np.clip(r - r_edge, 0.0, None)
        return cast(
            NDArrayFloatType,
            tau * np.exp(-x / damping) * np.sin(2.0 * np.pi * x / wavelength),
        )
    raise ValueError(f'ring feature kind {kind!r} is not renderable')
