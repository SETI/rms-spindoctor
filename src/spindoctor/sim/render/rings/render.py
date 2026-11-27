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
band overlaps is integrated along its footprint in closed form, because the
emission is not linear in the ramp's changing tau
(:mod:`spindoctor.sim.render.rings.footprint`).  Any other pixel is evaluated
at its center.

The photometry is the single-scattering closed form of
:mod:`spindoctor.sim.render.rings.photometry`, lit by the one-term
Henyey-Greenstein ``P(g, alpha)`` of
:func:`~spindoctor.sim.shared.lighting.henyey_greenstein_phase`.

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
    propeller_radial_reach,
    propeller_tau_factor,
)
from spindoctor.sim.render.rings.footprint import (
    PlacedFeature,
    RingGrid,
    RingPhotometry,
    SlopeMap,
    accumulate_feature,
    integrate_footprints,
    integrate_linear_pixels,
    split_sloped_pixels,
)
from spindoctor.sim.render.rings.photometry import ring_reflection_factor
from spindoctor.sim.render.rings.profiles import (
    FOOTPRINT_HALF_WIDTH_PX,
    WAVE_UNDERFLOW_DAMPINGS,
    feature_boundaries,
    feature_breakpoint_radii,
    feature_sloped_band,
    feature_tau,
)
from spindoctor.sim.scene.fields import RING_SIDE_DEFAULTS, scene_value
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
    'RingSystemMaps',
    'render_ring_system',
]


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
    size_u = shape[1]
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

    range_km = ring_system.get('range_km')
    r, lam, radial_scale, y = _ring_plane_grid(
        shape,
        center_v=center_v,
        center_u=center_u,
        opening_deg_obs=b_obs,
        node_deg=node_deg,
        with_y=range_km is not None,
    )

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
    placed: list[PlacedFeature] = []
    crossing_pixels: list[NDArrayIntType] = []
    crossing_offsets: list[NDArrayFloatType] = []
    slopes: SlopeMap | None = None
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
        placed.append(PlacedFeature(feature=feature, orbit=orbit, lo=lo, hi=hi))
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
        box_v, sub_u = slices
        # The box is evaluated a strip of rows at a time, which keeps each strip's
        # temporaries small (see _GRID_STRIP_ROWS).  Every pixel is evaluated on its
        # own, so the maps and the set of boundary crossings are the same as from one
        # pass over the box; the crossings are sorted by pixel and offset before use.
        for strip_start in range(box_v.start, box_v.stop, _GRID_STRIP_ROWS):
            sub_v = slice(strip_start, min(strip_start + _GRID_STRIP_ROWS, box_v.stop))
            r_sub = r[sub_v, sub_u]
            sel = (r_sub >= lo) & (r_sub <= hi)
            if not bool(np.any(sel)):
                continue
            r_sel = r_sub[sel]
            bounds = feature_boundaries(
                feature, orbit, lam[sub_v, sub_u][sel], os=os, epoch=epoch, time=time
            )
            accumulate_feature(
                feature,
                feature_tau(feature, r_sel, bounds, os=os),
                tau=tau_map[sub_v, sub_u],
                ap_weighted=ap_weighted[sub_v, sub_u],
                ap_weight=ap_weight[sub_v, sub_u],
                sel=sel,
                phase_angle=phase_angle,
            )
            breakpoints = feature_breakpoint_radii(feature, bounds, os=os)
            band = feature_sloped_band(feature, bounds, os=os)
            if not breakpoints and band is None:
                # A density wave has neither, and it reaches over most of the frame.
                continue
            scale_sel = radial_scale[sub_v, sub_u][sel]
            box = _BoxIndex(sel=sel, sub_v=sub_v, sub_u=sub_u, size_u=size_u)
            for r_break in breakpoints:
                offset = (r_break - r_sel) / scale_sel
                inside = np.abs(offset) < FOOTPRINT_HALF_WIDTH_PX
                if bool(np.any(inside)):
                    crossing_pixels.append(box.flat(inside))
                    crossing_offsets.append(offset[inside])
            if band is not None:
                # A ramp's optical depth changes across every pixel its band overlaps,
                # so each such pixel is integrated along its footprint.
                if slopes is None:
                    slopes = SlopeMap.empty(shape)
                slopes.add(band, r=r_sel, radial_scale=scale_sel, sel=sel, sub_v=sub_v, sub_u=sub_u)
    grid = RingGrid(r=r, lam=lam, radial_scale=radial_scale)
    all_crossings = _concat_int(crossing_pixels)
    split = split_sloped_pixels(
        slopes, all_crossings, placed, moonlets, grid=grid, unclipped_tau=tau_map, os=os
    )
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
        a_m, reach = propeller_radial_reach(moonlet, propeller, os=os)
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

    # The grid-sized maps below are built in place: a fresh grid-sized temporary costs
    # more than the arithmetic in it.  Each step is the IEEE operation of the plain
    # expression, so the maps are unchanged to the last bit.
    emitting = ap_weight > 0.0
    ap_map = ap_weighted
    np.divide(ap_map, ap_weight, out=ap_map, where=emitting)
    np.copyto(ap_map, 0.0, where=~emitting)
    intensity = ring_reflection_factor(tau_map, mu, mu0, lit=lit)
    intensity *= ap_map
    transmission: NDArrayFloatType = np.negative(tau_map)
    transmission /= mu
    np.exp(transmission, out=transmission)
    mask: NDArrayBoolType = tau_map > 0.0

    # Pixels that a boundary cuts are integrated across it: the emission and
    # transmission of each piece of the footprint, each at its own full tau,
    # averaged by length.  Mixing tau instead would brighten the pixel,
    # because the emission levels off as tau grows (#891).  A pixel that a
    # ramp's band overlaps is integrated in closed form along its footprint:
    # directly here when nothing cuts or curves it, and by cells otherwise.
    photometry = RingPhotometry(mu=mu, mu0=mu0, lit=lit, phase_angle=phase_angle)
    if split.linear.pixels.size:
        pixels = split.linear.pixels
        emission, linear_transmission, linear_mask = integrate_linear_pixels(
            split.linear, ap_map.flat[pixels], photometry=photometry
        )
        intensity.flat[pixels] = emission
        transmission.flat[pixels] = linear_transmission
        mask.flat[pixels] = linear_mask
    if all_crossings.size or split.cell_pixels.size:
        pixels, pixel_emission, pixel_transmission, pixel_mask = integrate_footprints(
            all_crossings,
            _concat_float(crossing_offsets),
            split.cell_pixels,
            split.curved_pixels,
            placed=placed,
            moonlets=moonlets,
            grid=grid,
            photometry=photometry,
            os=os,
            epoch=epoch,
            time=time,
        )
        intensity.flat[pixels] = pixel_emission
        transmission.flat[pixels] = pixel_transmission
        mask.flat[pixels] = pixel_mask

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
    if range_km is not None:
        assert y is not None
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


# Grid rows mapped to the ring plane at a time.  A strip's temporaries stay a few
# megabytes on a 4,000-pixel-wide oversampled grid, where whole-grid temporaries
# would each cost a fresh allocation larger than its arithmetic.
_GRID_STRIP_ROWS = 256


def _ring_plane_grid(
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    opening_deg_obs: float,
    node_deg: float,
    with_y: bool,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, NDArrayFloatType | None]:
    """Map every render-grid pixel center to the ring plane.

    Pixel centers relative to the projected ring center are mapped back to the ring
    plane through the shared inverse projection.  The grid starts as array rows and
    columns and the projection works in the geometry layer's pixel corner
    coordinates, where the scene states its center, so the half pixel goes on here.
    The grid is mapped a strip of rows at a time; each pixel is computed on its own,
    so the strips give the same bits as one pass over the whole grid.

    Parameters:
        shape: The render-grid shape.
        center_v: Ring-system center v on the render grid.
        center_u: Ring-system center u on the render grid.
        opening_deg_obs: Observer ring opening angle in degrees.
        node_deg: Sky position angle of the ascending node, in degrees.
        with_y: Whether to return the node-aligned in-plane y coordinate, which the
            line-of-sight depth reads.

    Returns:
        The ring-plane radius, the longitude, the radial scale (the ring-plane
        radial distance per image pixel: dividing by it converts a radial distance to
        image pixels, so a pixel footprint spans one render-grid pixel whatever the
        foreshortening direction), and the y coordinate or None, each per pixel.
    """
    size_v, size_u = shape
    v_coords = np.arange(size_v, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_coords = np.arange(size_u, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    # A column and a row, which the projection broadcasts to the strip.
    dv = (v_coords - center_v)[:, np.newaxis]
    du = (u_coords - center_u)[np.newaxis, :]
    r = np.empty(shape, dtype=np.float64)
    lam = np.empty(shape, dtype=np.float64)
    radial_scale = np.empty(shape, dtype=np.float64)
    y = np.empty(shape, dtype=np.float64) if with_y else None
    for start in range(0, size_v, _GRID_STRIP_ROWS):
        rows = slice(start, min(start + _GRID_STRIP_ROWS, size_v))
        r_rows, lam_rows, x_rows, y_rows = ring_plane_from_sky(
            dv[rows], du, opening_deg_obs=opening_deg_obs, node_deg=node_deg
        )
        r[rows] = r_rows
        lam[rows] = lam_rows
        radial_scale[rows] = ring_radial_scale(
            r_rows, x_rows, y_rows, opening_deg_obs=opening_deg_obs
        )
        if y is not None:
            y[rows] = y_rows
    return r, lam, radial_scale, y


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
            * WAVE_UNDERFLOW_DAMPINGS
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
