"""Image-side optical-depth ring-system renderer.

Draws the ``ring_system`` scene block: radial tau features (ringlets, gaps,
one-sided edges, ramps, and damped density-wave trains) on mode-1 eccentric
precessing orbits with optional m >= 2 modes and satellite edge waves,
projected through the shared opening-angle geometry
(:mod:`spindoctor.sim.shared.ring`) and lit by the single-scattering
closed forms.  The output is a set of per-pixel maps -- emitted intensity,
transmission ``exp(-tau/mu)``, and line-of-sight depth -- that the radiance
stage composites against the body stack as a transmission screen:
``img = I_ring + exp(-tau/mu) * img_behind``, evaluated far to near per
pixel, so low-tau features reveal the background instead of erasing it and
stars behind the ring attenuate physically.

Radial tau profiles by feature kind (``r_e(lam)`` the feature's perturbed
orbit radius, ``w`` its radial width ``width_px``, distances in ring-plane
radial units: the scene's ring-plane pixels times the oversampling factor):

- ``ringlet``: ``tau`` between ``r_e`` and ``r_e + w``.
- ``gap``: the same band as a tau suppression (subtracts, clipped at zero).
- ``edge``: a one-sided step; ``side: 'in'`` (the default) carries ``tau``
  for ``r <= r_e`` (a sheet bounded by its outer edge, the B-ring-edge
  case), ``side: 'out'`` for ``r >= r_e``.
- ``ramp``: a linear transition across ``[r_e, r_e + w]``; ``side: 'out'``
  (the default) rises from 0 at ``r_e`` to ``tau`` at ``r_e + w`` (compose
  with an ``edge`` there for a sheet with a gradual inner boundary),
  ``side: 'in'`` mirrors it.  Zero outside the band; the end that carries
  ``tau`` is a step and the other end a bend.
- ``wave``: a damped radial sinusoid launched at ``r_e``:
  ``dtau(x) = tau * exp(-x / density_wave_damping_px) * sin(2*pi*x / wavelength_px)`` for
  ``x = r - r_e >= 0`` and exactly zero upstream (the clamp mirrors the
  azimuthal edge wave's: the envelope grows without bound for ``x < 0``).
  A density-wave train is visual clutter riding on a sheet; its negative
  lobes subtract from the composed tau.

The profiles have sharp boundaries (:mod:`spindoctor.sim.render.rings.profiles`).
A pixel that a boundary cuts through is integrated across it: its radial
footprint, one render-grid pixel long, is split at every boundary inside it, the
tau, emission, and transmission are evaluated in each piece, and the pieces are
averaged by length.  So a pixel 30% covered by a ringlet gets 30% of the
ringlet's full brightness and 70% of what lies beside it, and a boundary lands
at its authored radius whatever its optical depth.  A pixel that a ramp's
band overlaps is integrated along its footprint by quadrature, because the
emission is not linear in the ramp's changing tau.  Any other pixel is
evaluated at its center.

Photometry (the normative equation set), with ``mu = |sin B_obs|``,
``mu0 = |sin B_sun|``, lit iff ``sign(B_obs) == sign(B_sun)``, and the one-term
Henyey-Greenstein ``P(g, alpha)`` of
:func:`~spindoctor.sim.shared.lighting.henyey_greenstein_phase`:

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
radial bounds, widened by the kind's shape extent and a margin for the pixel
footprint, map through the projection to a pixel bounding box, and outside the
bounded band every contribution is exactly 0.0 (the sharp profiles, the
ramp's clipped fraction, the wave envelope's float64 underflow).  The bounding
changes rendering cost with feature count, never the rendered values.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spindoctor.sim.render.rings.clutter import (
    azimuthal_intensity_factor,
    moonlet_disc_coverage,
    propeller_tau_factor,
)
from spindoctor.sim.render.rings.profiles import (
    FOOTPRINT_HALF_WIDTH_PX,
    FootprintSamples,
    feature_boundaries,
    feature_breakpoint_radii,
    feature_tau,
    feature_tau_slope,
    footprint_samples,
)
from spindoctor.sim.scene.fields import RING_SIDE_DEFAULTS, scene_value
from spindoctor.sim.shared.lighting import henyey_greenstein_phase
from spindoctor.sim.shared.ring import (
    RingOrbit,
    ring_los_depth,
    ring_orbit_from_mapping,
    ring_plane_from_sky,
    ring_radial_scale,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = [
    'RING_ALBEDO_DEFAULT',
    'RING_PHASE_G_DEFAULT',
    'RingSystemMaps',
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
# The largest change in the photometric exponent (tau times 1/mu + 1/mu0)
# across one quadrature piece of a ramp; see the profiles module.
_MAX_PIECE_EXPONENT: float = 2.0


@dataclass(frozen=True)
class RingSystemMaps:
    """Per-pixel maps of a rendered ring system on the (oversampled) grid.

    Parameters:
        intensity: Emitted ring brightness ``I`` per pixel (normalized signal
            units).
        transmission: The fraction of background light (bodies behind the
            ring, stars, sky) that passes through each pixel: ``exp(-tau/mu)``,
            averaged over the pieces of a pixel that a feature boundary cuts.
        mask: Pixels where the system carries any optical depth.
        depth_km: Observer distance ``range_km - dlos_km`` per pixel, or None
            when the scene gives the system no ``range_km`` (the system then
            has no depth relation to bodies; overlap is a scene error).
    """

    intensity: NDArrayFloatType
    transmission: NDArrayFloatType
    mask: NDArrayBoolType
    depth_km: NDArrayFloatType | None


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
        oversample: The render-grid oversampling factor; radii and widths
            scale by it, and the depth map converts back through it so
            ``depth_km`` is grid-independent.  A pixel footprint is one
            render-grid pixel, so the box downsample to the detector
            integrates each boundary over exactly one detector pixel.
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
    b_obs = float(scene_value(geometry, 'opening_deg_obs', 0.0))
    b_sun = float(scene_value(geometry, 'opening_deg_sun', 0.0))
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
    phase_angle = math.radians(float(scene_value(ring_system, 'phase_deg', 0.0)))

    # Pixel centers relative to the projected ring center, mapped back to the
    # ring plane through the shared inverse projection.  The grid starts as
    # array rows and columns and the projection works in the geometry layer's
    # pixel corner coordinates, where the scene states its center, so the half
    # pixel goes on here.
    v_coords = np.arange(size_v, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_coords = np.arange(size_u, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')
    dv = v_grid - center_v
    du = u_grid - center_u
    r, lam, x, y = ring_plane_from_sky(dv, du, opening_deg_obs=b_obs, node_deg=node_deg)
    # Ring-plane radial distance per image pixel: dividing by this converts a
    # radial distance to image pixels, so a pixel footprint spans one
    # render-grid pixel regardless of the foreshortening direction.
    radial_scale = ring_radial_scale(r, x, y, opening_deg_obs=b_obs)

    # Compose the radial tau profile at each pixel center: emitting kinds add
    # their tau, gaps subtract (tau suppression), a wave's negative lobes
    # subtract, and the sum clips at zero.  The emission's A/4 * P factor is
    # the tau-weighted mean over the positive contributions, so a single
    # feature reproduces its closed form exactly.  Each feature is evaluated
    # only inside its radial annulus (see the module docstring): outside it
    # every contribution is exactly 0.0, so the bounding is a pure
    # optimization.  The same pass collects every boundary that falls inside a
    # pixel footprint and every pixel that a ramp's band overlaps; those pixels
    # are integrated across their footprints afterwards.
    #
    # A footprint reaches FOOTPRINT_HALF_WIDTH_PX * radial_scale ring-plane
    # units from its pixel center, and radial_scale <= 1/|sin B| everywhere,
    # so this margin (twice the strict bound, for safety) keeps the bounding
    # exact.
    footprint_margin = 2.0 * FOOTPRINT_HALF_WIDTH_PX / mu
    tau_map = np.zeros(shape, dtype=np.float64)
    ap_weighted = np.zeros(shape, dtype=np.float64)
    ap_weight = np.zeros(shape, dtype=np.float64)
    placed: list[tuple[Mapping[str, Any], RingOrbit, float, float]] = []
    crossing_pixels: list[NDArrayIntType] = []
    crossing_offsets: list[NDArrayFloatType] = []
    sloped_pixels: list[NDArrayIntType] = []
    sloped_pieces: list[NDArrayFloatType] = []
    # The emission and transmission are exponentials in tau at rates up to
    # 1/mu + 1/mu0; a quadrature piece spans at most _MAX_PIECE_EXPONENT of
    # that exponent, so this many pieces per unit of tau change.
    pieces_per_tau = (1.0 / mu + 1.0 / mu0) / _MAX_PIECE_EXPONENT
    for feature in features:
        # The catalog orbit with the planted (truth-side) ephemeris error
        # applied on this side only -- the navigator predicts from the
        # catalog values -- then scaled to the render grid.  lam is
        # ring-plane longitude, so every orbital angle lives in the
        # ring-plane frame (the node angle entered only the sky projection
        # above).
        orbit = _orbit_with_error(
            ring_orbit_from_mapping(feature.get('orbit') or {}), feature.get('orbit_error')
        ).scaled(float(os))
        lo, hi = _feature_radial_bounds(feature, orbit, os=os, margin=footprint_margin)
        placed.append((feature, orbit, lo, hi))
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
        r_sel = r_sub[sel]
        bounds = feature_boundaries(
            feature, orbit, lam[sub_v, sub_u][sel], os=os, epoch=epoch, time=time
        )
        _accumulate_feature(
            feature,
            feature_tau(feature, r_sel, bounds, os=os),
            tau=tau_map[sub_v, sub_u],
            ap_weighted=ap_weighted[sub_v, sub_u],
            ap_weight=ap_weight[sub_v, sub_u],
            sel=sel,
            phase_angle=phase_angle,
        )
        scale_sel = radial_scale[sub_v, sub_u][sel]
        box = _BoxIndex(sel=sel, sub_v=sub_v, sub_u=sub_u, size_u=size_u)
        for r_break in feature_breakpoint_radii(feature, bounds):
            offset = (r_break - r_sel) / scale_sel
            inside = np.abs(offset) < FOOTPRINT_HALF_WIDTH_PX
            if bool(np.any(inside)):
                crossing_pixels.append(box.flat(inside))
                crossing_offsets.append(offset[inside])
        slope = feature_tau_slope(feature, os=os)
        if slope is not None and bounds.outer is not None:
            # A ramp's emission changes across every pixel its band overlaps, so
            # each such pixel is integrated by quadrature.
            footprint_reach = FOOTPRINT_HALF_WIDTH_PX * scale_sel
            overlap = (r_sel + footprint_reach > bounds.inner) & (
                r_sel - footprint_reach < bounds.outer
            )
            if bool(np.any(overlap)):
                sloped_pixels.append(box.flat(overlap))
                sloped_pieces.append(pieces_per_tau * slope * scale_sel[overlap])
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
        a_m = float(scene_value(moonlet, 'a_px', 0.0)) * os
        width = float(scene_value(propeller, 'width_px', 0.0)) * os
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
        factor = propeller_tau_factor(
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

    # Pixels that a boundary cuts are integrated across it: the emission and
    # transmission of each piece of the footprint, each at its own full tau,
    # averaged by length.  Mixing tau instead would brighten the pixel,
    # because the emission levels off as tau grows (#891).  A pixel that a
    # ramp's band overlaps is integrated by quadrature along its footprint.
    if crossing_pixels or sloped_pixels:
        samples = footprint_samples(
            _concat_int(crossing_pixels),
            _concat_float(crossing_offsets),
            sloped_pixels=_concat_int(sloped_pixels),
            sloped_pieces=_concat_float(sloped_pieces),
        )
        _integrate_footprints(
            samples,
            placed,
            moonlets,
            grid=_RingGrid(r=r, lam=lam, radial_scale=radial_scale),
            maps=RingSystemMaps(
                intensity=intensity, transmission=transmission, mask=mask, depth_km=None
            ),
            photometry=_Photometry(mu=mu, mu0=mu0, lit=lit, phase_angle=phase_angle),
            os=os,
            epoch=epoch,
            time=time,
        )

    # Azimuthal structure (modulation, planet-shadow wedge, seeded spokes)
    # is albedo/illumination structure: it scales the emitted intensity and
    # leaves the optical depth -- and so the transmission -- untouched.
    azimuthal = ring_system.get('azimuthal')
    if azimuthal:
        intensity *= azimuthal_intensity_factor(
            azimuthal, r=r, lam=lam, os=os, spokes_seed=spokes_seed
        )

    # Moonlet discs: opaque point-like bodies at the ring's depth.  Each
    # disc replaces the ring emission with its own, extinguishes the
    # background through the transmission screen, and joins the composited
    # mask so a moonlet sitting in an empty gap still composites.  The
    # coverage is exactly 0.0 beyond half a render-grid pixel outside
    # the disc, so each disc is evaluated only inside its sky bounding box
    # (outside it the compositing forms are the identity); the bounding
    # changes cost, never values.
    for moonlet in moonlets:
        disc = moonlet_disc_coverage(
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
        amplitude = float(scene_value(moonlet, 'amplitude', 0.0))
        intensity[sub_v, sub_u] = intensity[sub_v, sub_u] * (1.0 - coverage) + amplitude * coverage
        transmission[sub_v, sub_u] = transmission[sub_v, sub_u] * (1.0 - coverage)
        mask[sub_v, sub_u] |= coverage > 0.0

    depth_km: NDArrayFloatType | None = None
    range_km = ring_system.get('range_km')
    if range_km is not None:
        # dlos is positive toward the observer on the render grid; divide by
        # the oversampling to return to detector pixels before the km scale.
        km_per_pixel = float(scene_value(ring_system, 'km_per_pixel', 1.0))
        dlos_km = ring_los_depth(y, opening_deg_obs=b_obs) / float(os) * km_per_pixel
        depth_km = float(range_km) - dlos_km

    return RingSystemMaps(
        intensity=intensity,
        transmission=transmission,
        mask=mask,
        depth_km=depth_km,
    )


@dataclass
class _BoxIndex:
    """Converts selections inside one feature's bounding box to flat grid indices.

    Parameters:
        sel: The feature's selected pixels within its box.
        sub_v: The box's row slice.
        sub_u: The box's column slice.
        size_u: The render grid's width.
    """

    sel: NDArrayBoolType
    sub_v: slice
    sub_u: slice
    size_u: int
    _selected: NDArrayIntType | None = None

    def flat(self, mask: NDArrayBoolType) -> NDArrayIntType:
        """Return the flat grid index of each selected pixel where ``mask`` is True.

        Parameters:
            mask: One entry per selected pixel, in selection order.

        Returns:
            The flat indices into the render grid.
        """
        if self._selected is None:
            self._selected = np.flatnonzero(self.sel)
        local = self._selected[mask]
        box_width = self.sub_u.stop - self.sub_u.start
        rows = self.sub_v.start + local // box_width
        cols = self.sub_u.start + local % box_width
        flat: NDArrayIntType = rows * self.size_u + cols
        return flat


def _concat_int(parts: list[NDArrayIntType]) -> NDArrayIntType:
    """Concatenate index arrays, or return an empty one."""
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _concat_float(parts: list[NDArrayFloatType]) -> NDArrayFloatType:
    """Concatenate float arrays, or return an empty one."""
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _orbit_with_error(orbit: RingOrbit, error: Mapping[str, Any] | None) -> RingOrbit:
    """The catalog orbit with the planted per-feature ephemeris error applied.

    ``orbit_error`` is a truth key: the render side displaces the drawn
    feature by these deltas while the navigator predicts from the unmodified
    catalog orbit, so features are misplaced relative to the model (and to
    each other) exactly as real ring features are relative to their
    published orbit solutions.  A negative ``delta_ae_px`` larger than the
    catalog ``ae_px`` clamps at a circular edge (a negative radial amplitude is
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
        a=orbit.a + float(scene_value(error, 'delta_a_px', 0.0)),
        ae=max(0.0, orbit.ae + float(scene_value(error, 'delta_ae_px', 0.0))),
        long_peri=orbit.long_peri + float(scene_value(error, 'delta_long_peri_deg', 0.0)),
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
    reach of a pixel footprint in ring-plane units.  Outside the returned
    band every profile form evaluates to exactly 0.0 (the sharp profiles, the
    ramp's clipped fraction, the wave's clipped upstream argument and
    underflowed envelope), so restricting evaluation to the band is exact.

    Parameters:
        feature: The validated feature mapping.
        orbit: The feature's orbit, scaled to the render grid.
        os: The oversampling factor (scales the shape keys).
        margin: The pixel-footprint safety margin in render-grid radial units.

    Returns:
        ``(lo, hi)`` in render-grid ring-plane units; ``hi`` is ``inf`` for
        an outward-unbounded edge and ``lo`` clips at zero.
    """
    kind = str(feature.get('kind'))
    lo, hi = _orbit_radial_bounds(orbit)
    if kind in ('ringlet', 'gap', 'ramp'):
        hi += float(scene_value(feature, 'width_px', 0.0)) * os
    elif kind == 'edge':
        if str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['edge'])) == 'in':
            lo = -math.inf
        else:
            hi = math.inf
    elif kind == 'wave':
        hi += (
            float(scene_value(feature, 'density_wave_damping_px', 0.0))
            * os
            * _WAVE_UNDERFLOW_DAMPINGS
        )
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
    box; one pixel of slack absorbs the half pixel between the coordinates the
    radius is stated in and the cells the box names.

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


@dataclass(frozen=True)
class _RingGrid:
    """The ring-plane coordinates of every render-grid pixel center.

    Parameters:
        r: Ring-plane radius per pixel, in render-grid units.
        lam: Ring-plane longitude per pixel, in radians.
        radial_scale: Ring-plane radial distance per render-grid pixel.
    """

    r: NDArrayFloatType
    lam: NDArrayFloatType
    radial_scale: NDArrayFloatType


@dataclass(frozen=True)
class _Photometry:
    """The viewing geometry the single-scattering closed forms read.

    Parameters:
        mu: ``|sin B_obs|``.
        mu0: ``|sin B_sun|``.
        lit: Whether the observer sees the lit face.
        phase_angle: The phase angle in radians.
    """

    mu: float
    mu0: float
    lit: bool
    phase_angle: float


def _accumulate_feature(
    feature: Mapping[str, Any],
    contribution: NDArrayFloatType,
    *,
    tau: NDArrayFloatType,
    ap_weighted: NDArrayFloatType,
    ap_weight: NDArrayFloatType,
    sel: NDArrayBoolType | NDArrayIntType,
    phase_angle: float,
) -> None:
    """Add one feature's tau to the composed tau and its photometry to the weights.

    A gap's tau is subtracted.  Any other feature adds its tau, and its ``A/4 * P``
    factor joins the tau-weighted mean over the positive contributions.

    Parameters:
        feature: The validated feature mapping.
        contribution: The feature's tau at the selected points.
        tau: The composed tau, updated in place at ``sel``.
        ap_weighted: The tau-weighted sum of ``A/4 * P``, updated in place at ``sel``.
        ap_weight: The sum of the positive tau, updated in place at ``sel``.
        sel: The points ``contribution`` belongs to, as a mask or as indices.
        phase_angle: The phase angle in radians.
    """
    if str(feature.get('kind')) == 'gap':
        tau[sel] -= contribution
        return
    tau[sel] += contribution
    albedo = float(scene_value(feature, 'albedo', RING_ALBEDO_DEFAULT))
    phase_g = float(scene_value(feature, 'phase_g', RING_PHASE_G_DEFAULT))
    ap = albedo / 4.0 * henyey_greenstein_phase(phase_g, phase_angle)
    positive = np.clip(contribution, 0.0, None)
    ap_weighted[sel] += ap * positive
    ap_weight[sel] += positive


def _integrate_footprints(
    samples: FootprintSamples,
    placed: list[tuple[Mapping[str, Any], RingOrbit, float, float]],
    moonlets: list[Mapping[str, Any]],
    *,
    grid: _RingGrid,
    maps: RingSystemMaps,
    photometry: _Photometry,
    os: int,
    epoch: float,
    time: float,
) -> None:
    """Replace the sampled pixels' maps with their integrals along their footprints.

    Every sample is a point at radius ``r + offset * radial_scale`` and at the
    longitude of its pixel center.  Each feature and each propeller is evaluated there
    exactly as at a pixel center, the tau is composed and clipped, and the emission and
    transmission of the samples are summed into their pixel by weight.  The pixel joins
    the mask when any sample carries tau.

    Parameters:
        samples: The footprint samples.
        placed: Each feature with its render-grid orbit and its radial bounds.
        moonlets: The scene's moonlets, for their propellers.
        grid: The ring-plane coordinates of the pixel centers.
        maps: The maps to update in place; ``depth_km`` is not read.
        photometry: The viewing geometry.
        os: The oversampling factor.
        epoch: Ring epoch in TDB seconds.
        time: Scene time in TDB seconds.
    """
    pixel = samples.pixels[samples.slot]
    r_center = grid.r.ravel()[pixel]
    r_point = r_center + samples.offset * grid.radial_scale.ravel()[pixel]
    lam_point = grid.lam.ravel()[pixel]
    tau = np.zeros(r_point.shape, dtype=np.float64)
    ap_weighted = np.zeros(r_point.shape, dtype=np.float64)
    ap_weight = np.zeros(r_point.shape, dtype=np.float64)
    # Samples sorted by pixel-center radius, so each feature finds the samples
    # inside its radial bounds by bisection instead of testing them all.
    by_radius = np.argsort(r_center, kind='stable')
    sorted_radius = r_center[by_radius]
    for feature, orbit, lo, hi in placed:
        # The bounds are tested at the pixel center, as on the grid, so a feature
        # reaches every segment of each pixel it was evaluated at.
        start = int(np.searchsorted(sorted_radius, lo, side='left'))
        stop = int(np.searchsorted(sorted_radius, hi, side='right'))
        if stop <= start:
            continue
        sel = by_radius[start:stop]
        bounds = feature_boundaries(feature, orbit, lam_point[sel], os=os, epoch=epoch, time=time)
        _accumulate_feature(
            feature,
            feature_tau(feature, r_point[sel], bounds, os=os),
            tau=tau,
            ap_weighted=ap_weighted,
            ap_weight=ap_weight,
            sel=sel,
            phase_angle=photometry.phase_angle,
        )
    np.clip(tau, 0.0, None, out=tau)
    for moonlet in moonlets:
        propeller = moonlet.get('propeller')
        if not propeller:
            continue
        tau *= propeller_tau_factor(moonlet, propeller, r=r_point, lam=lam_point, os=os)
    ap = np.divide(
        ap_weighted, ap_weight, out=np.zeros(tau.shape, dtype=np.float64), where=ap_weight > 0.0
    )
    emission = ap * ring_reflection_factor(tau, photometry.mu, photometry.mu0, lit=photometry.lit)
    count = samples.pixels.size
    weight = samples.weight
    maps.intensity.flat[samples.pixels] = np.bincount(
        samples.slot, weights=weight * emission, minlength=count
    )
    maps.transmission.flat[samples.pixels] = np.bincount(
        samples.slot, weights=weight * np.exp(-tau / photometry.mu), minlength=count
    )
    maps.mask.flat[samples.pixels] = (
        np.bincount(samples.slot, weights=weight * (tau > 0.0), minlength=count) > 0.0
    )
