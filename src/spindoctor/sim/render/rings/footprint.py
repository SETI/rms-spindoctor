"""Integration of ring pixels along their radial footprints.

The renderer first evaluates every pixel at its center.  A pixel that a feature boundary
cuts, or that a ramp's band overlaps, is then integrated along its footprint (see
:mod:`spindoctor.sim.render.rings.profiles`), which is cut into cells at every
boundary inside it.

- A cell of constant optical depth is evaluated once, at its midpoint, with the point
  forms of :mod:`spindoctor.sim.render.rings.photometry`, and weighted by its length.
- In a pixel that a ramp's band overlaps, the optical depth changes linearly along each
  cell, because every place the profile steps or bends is a cell boundary.  The cell is
  evaluated at its quarter and three-quarter points, which fixes the line, and its
  emission and transmission are integrated exactly with the closed-form means.  Where
  a gap or a wave's negative lobe pulls the summed optical depth below zero inside the
  cell, the line is split where it crosses zero, and the part below is empty.  The
  cost does not depend on how steep the ramp is.
- Where a propeller or a density wave also reaches such a pixel, the optical depth
  curves, and each segment is cut into :data:`CURVED_CELLS` equal cells, each
  integrated the same way.  The integral of the curve is then approximate.

The ``A/4 * P`` factor of a linear cell is the tau-weighted mean at the cell's midpoint,
which is exact unless features of different albedo or phase asymmetry overlap there.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from spindoctor.sim.render.rings.clutter import propeller_radial_reach, propeller_tau_factor
from spindoctor.sim.render.rings.photometry import (
    RING_ALBEDO_DEFAULT,
    RING_PHASE_G_DEFAULT,
    ring_reflection_factor,
    ring_reflection_mean,
    ring_transmission_mean,
)
from spindoctor.sim.render.rings.profiles import (
    FOOTPRINT_HALF_WIDTH_PX,
    WAVE_UNDERFLOW_DAMPINGS,
    FootprintCells,
    SlopedBand,
    feature_boundaries,
    feature_tau,
    footprint_cells,
)
from spindoctor.sim.scene.fields import scene_value
from spindoctor.sim.shared.lighting import henyey_greenstein_phase
from spindoctor.sim.shared.ring import RingOrbit
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = [
    'CURVED_CELLS',
    'PlacedFeature',
    'RingGrid',
    'RingPhotometry',
    'SlopeMap',
    'SlopedSplit',
    'accumulate_feature',
    'integrate_footprints',
    'integrate_linear_pixels',
    'split_sloped_pixels',
]

CURVED_CELLS: int = 32
"""How many cells a footprint segment is cut into where its optical depth curves."""

# The most pixels integrated directly at once.
_LINEAR_CHUNK: int = 1 << 20
# About the most cells integrated at once on the cell path; a batch holds whole pixels,
# so it can exceed this by one pixel's cells.
_CELL_BATCH: int = 1 << 20

# How many damping lengths downstream of its launch radius a density wave still
# counts as curving the optical depth of a ramp it overlaps.
_WAVE_CURVED_DAMPINGS: float = 40.0


@dataclass(frozen=True)
class PlacedFeature:
    """A feature with its render-grid orbit and the radial band its tau stays inside.

    Parameters:
        feature: The validated feature mapping.
        orbit: The feature's orbit, scaled to the render grid, with any planted error.
        lo: The smallest render-grid radius where the feature can carry tau.
        hi: The largest render-grid radius where the feature can carry tau.
    """

    feature: Mapping[str, Any]
    orbit: RingOrbit
    lo: float
    hi: float


@dataclass(frozen=True)
class RingGrid:
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
class RingPhotometry:
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


def accumulate_feature(
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
    ap_weight[sel] += positive
    # Scaled in place once the weight has it; ap * positive, to the last bit.
    positive *= ap
    ap_weighted[sel] += positive


@dataclass(frozen=True)
class SlopeMap:
    """Which pixels a ramp's band overlaps, and how fast tau changes across each.

    Parameters:
        mask: The pixels whose footprint a ramp's band overlaps.
        slope: The change in the summed ramps' optical depth per render-grid pixel
            along the footprint, at each pixel whose center lies inside a band.
    """

    mask: NDArrayBoolType
    slope: NDArrayFloatType

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> 'SlopeMap':
        """Return a map with no sloped pixels.

        Parameters:
            shape: The render-grid shape.

        Returns:
            The map.
        """
        return cls(mask=np.zeros(shape, dtype=np.bool_), slope=np.zeros(shape, dtype=np.float64))

    def add(
        self,
        band: SlopedBand,
        *,
        r: NDArrayFloatType,
        radial_scale: NDArrayFloatType,
        sel: NDArrayBoolType,
        sub_v: slice,
        sub_u: slice,
    ) -> None:
        """Add one ramp, evaluated at the selected pixels of a bounding box.

        Parameters:
            band: The ramp's band at the selected pixels.
            r: The selected pixels' center radii.
            radial_scale: The selected pixels' ring-plane radial distance per pixel.
            sel: The selected pixels within the box.
            sub_v: The box's row slice.
            sub_u: The box's column slice.
        """
        reach = FOOTPRINT_HALF_WIDTH_PX * radial_scale
        overlap = (r + reach > band.inner) & (r - reach < band.outer)
        inside = (r >= band.inner) & (r <= band.outer)
        hit = np.zeros(sel.shape, dtype=np.bool_)
        hit[sel] = overlap
        mask_view = self.mask[sub_v, sub_u]
        mask_view |= hit
        slope_view = self.slope[sub_v, sub_u]
        slope_view[sel] += np.where(inside, band.rate * radial_scale, 0.0)


@dataclass(frozen=True)
class LinearPixels:
    """Sloped pixels that no boundary cuts and no propeller or wave reaches.

    Parameters:
        pixels: Their flat grid indices.
        tau: The summed optical depth at each center, before clipping at zero.
        slope: Its change per render-grid pixel along the footprint.
    """

    pixels: NDArrayIntType
    tau: NDArrayFloatType
    slope: NDArrayFloatType


@dataclass(frozen=True)
class SlopedSplit:
    """The sloped pixels, sorted by how they are integrated.

    Parameters:
        linear: The pixels integrated directly from their center and slope.
        cell_pixels: The sloped pixels integrated by cells, because a boundary cuts
            them or their optical depth curves.
        curved_pixels: The sloped pixels that a propeller or a density wave reaches.
    """

    linear: LinearPixels
    cell_pixels: NDArrayIntType
    curved_pixels: NDArrayIntType


def split_sloped_pixels(
    slopes: SlopeMap | None,
    crossing_pixels: NDArrayIntType,
    placed: list[PlacedFeature],
    moonlets: list[Mapping[str, Any]],
    *,
    grid: RingGrid,
    unclipped_tau: NDArrayFloatType,
    os: int,
) -> SlopedSplit:
    """Sort the sloped pixels by how they are integrated.

    A sloped pixel that no boundary cuts and no propeller or wave reaches has an
    optical depth that changes linearly along its whole footprint, by its slope, so it
    is integrated from its center value alone.  Every other sloped pixel is
    integrated by cells.

    Parameters:
        slopes: The slope map, or None when the system has no ramp.
        crossing_pixels: The flat grid index of each boundary inside a footprint.
        placed: Every feature with its orbit and radial band.
        moonlets: The scene's moonlets, for their propellers.
        grid: The ring-plane coordinates of the pixel centers.
        unclipped_tau: The summed optical depth at every pixel center, before clipping.
        os: The oversampling factor.

    Returns:
        The split.
    """
    empty_int = np.zeros(0, dtype=np.int64)
    if slopes is None:
        none = LinearPixels(pixels=empty_int, tau=np.zeros(0), slope=np.zeros(0))
        return SlopedSplit(linear=none, cell_pixels=empty_int, curved_pixels=empty_int)
    sloped = np.flatnonzero(slopes.mask)
    cut = np.zeros(slopes.mask.size, dtype=np.bool_)
    cut[crossing_pixels] = True
    is_cut = cut[sloped]
    is_curved = _curved(sloped, placed, moonlets, grid=grid, os=os)
    simple = sloped[~is_cut & ~is_curved]
    linear = LinearPixels(
        pixels=simple, tau=unclipped_tau.ravel()[simple], slope=slopes.slope.ravel()[simple]
    )
    return SlopedSplit(
        linear=linear, cell_pixels=sloped[is_cut | is_curved], curved_pixels=sloped[is_curved]
    )


def integrate_linear_pixels(
    linear: LinearPixels, ap: NDArrayFloatType, *, photometry: RingPhotometry
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Integrate pixels whose optical depth changes linearly along the footprint.

    Parameters:
        linear: The pixels, their center optical depth, and their slope.
        ap: Each pixel's ``A/4 * P`` factor at its center.
        photometry: The viewing geometry.

    Returns:
        Each pixel's emission, transmission, and whether any of it carries tau.
    """
    count = linear.pixels.size
    emission = np.empty(count, dtype=np.float64)
    pixel_transmission = np.empty(count, dtype=np.float64)
    pixel_mask = np.empty(count, dtype=np.bool_)
    # In chunks, which bounds the temporaries of a ramp that spans much of the frame.
    for start in range(0, count, _LINEAR_CHUNK):
        part = slice(start, start + _LINEAR_CHUNK)
        half = FOOTPRINT_HALF_WIDTH_PX * linear.slope[part]
        factor, transmission, covered = _linear_cell_means(
            linear.tau[part] - half, linear.tau[part] + half, photometry=photometry
        )
        emission[part] = covered * ap[part] * factor
        pixel_transmission[part] = covered * transmission + (1.0 - covered)
        pixel_mask[part] = covered > 0.0
    return emission, pixel_transmission, pixel_mask


def _curved(
    pixels: NDArrayIntType,
    placed: list[PlacedFeature],
    moonlets: list[Mapping[str, Any]],
    *,
    grid: RingGrid,
    os: int,
) -> NDArrayBoolType:
    """Which of the pixels a density wave or a propeller reaches anywhere in its footprint.

    A wave's band already carries the footprint margin of its feature's radial bounds.
    A propeller's reach is widened by the footprint's own reach, because a lobe
    narrower than a pixel can cross a footprint whose center lies beyond the reach.
    """
    radius = grid.r.ravel()[pixels]
    footprint_reach = FOOTPRINT_HALF_WIDTH_PX * grid.radial_scale.ravel()[pixels]
    curved = np.zeros(pixels.size, dtype=np.bool_)
    for item in placed:
        if str(item.feature.get('kind')) == 'wave':
            # The band reaches to float64 underflow, far beyond where the wave's
            # tau matters; past _WAVE_CURVED_DAMPINGS it is under 4e-18 of tau.
            damping = float(scene_value(item.feature, 'density_wave_damping_px', 0.0)) * os
            reach = item.hi - (WAVE_UNDERFLOW_DAMPINGS - _WAVE_CURVED_DAMPINGS) * damping
            curved |= (radius >= item.lo) & (radius <= reach)
    for moonlet in moonlets:
        propeller = moonlet.get('propeller')
        if propeller:
            center, reach = propeller_radial_reach(moonlet, propeller, os=os)
            curved |= np.abs(radius - center) <= reach + footprint_reach
    return curved


def _evaluate(
    r_center: NDArrayFloatType,
    r_point: NDArrayFloatType,
    lam_point: NDArrayFloatType,
    placed: list[PlacedFeature],
    *,
    photometry: RingPhotometry,
    os: int,
    epoch: float,
    time: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """The unclipped composed tau and the photometric weights at a set of points.

    Each feature is tested against its radial bounds at the pixel center, as on the
    grid, so it reaches every point of each pixel it was evaluated at.  The points are
    sorted by pixel-center radius, so each feature finds its points by bisection.
    """
    tau = np.zeros(r_point.shape, dtype=np.float64)
    ap_weighted = np.zeros(r_point.shape, dtype=np.float64)
    ap_weight = np.zeros(r_point.shape, dtype=np.float64)
    by_radius = np.argsort(r_center, kind='stable')
    sorted_radius = r_center[by_radius]
    for item in placed:
        start = int(np.searchsorted(sorted_radius, item.lo, side='left'))
        stop = int(np.searchsorted(sorted_radius, item.hi, side='right'))
        if stop <= start:
            continue
        sel = by_radius[start:stop]
        bounds = feature_boundaries(
            item.feature, item.orbit, lam_point[sel], os=os, epoch=epoch, time=time
        )
        accumulate_feature(
            item.feature,
            feature_tau(item.feature, r_point[sel], bounds, os=os),
            tau=tau,
            ap_weighted=ap_weighted,
            ap_weight=ap_weight,
            sel=sel,
            phase_angle=photometry.phase_angle,
        )
    return tau, ap_weighted, ap_weight


def _ap(ap_weighted: NDArrayFloatType, ap_weight: NDArrayFloatType) -> NDArrayFloatType:
    """The tau-weighted mean ``A/4 * P``, or 0 where nothing emits."""
    result: NDArrayFloatType = np.divide(
        ap_weighted,
        ap_weight,
        out=np.zeros(ap_weight.shape, dtype=np.float64),
        where=ap_weight > 0.0,
    )
    return result


def _linear_cell_means(
    tau_start: NDArrayFloatType,
    tau_end: NDArrayFloatType,
    *,
    photometry: RingPhotometry,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """The mean geometric factor, transmission, and covered fraction of linear cells.

    The optical depth runs linearly from ``tau_start`` to ``tau_end`` along each cell,
    and below zero it is empty: there the cell emits nothing and passes everything.
    The means are over the covered part, the fraction of the cell where the optical
    depth is above zero.
    """
    rising = tau_end - tau_start
    both = (tau_start >= 0.0) & (tau_end >= 0.0)
    neither = (tau_start <= 0.0) & (tau_end <= 0.0) & ~both
    # The fraction of the cell where the line is above zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        crossing = np.where(rising != 0.0, -tau_start / rising, 0.0)
    covered = np.where(
        both, 1.0, np.where(neither, 0.0, np.where(rising > 0.0, 1.0 - crossing, crossing))
    )
    tau_lo = np.where(both, np.minimum(tau_start, tau_end), 0.0)
    tau_hi = np.maximum(np.maximum(tau_start, tau_end), 0.0)
    tau_hi = np.where(neither, 0.0, tau_hi)
    factor = ring_reflection_mean(tau_lo, tau_hi, photometry.mu, photometry.mu0, lit=photometry.lit)
    transmission = ring_transmission_mean(tau_lo, tau_hi, photometry.mu)
    return factor, transmission, np.where(tau_hi > 0.0, covered, 0.0)


def integrate_footprints(
    crossing_pixels: NDArrayIntType,
    crossing_offsets: NDArrayFloatType,
    sloped_pixels: NDArrayIntType,
    curved_pixels: NDArrayIntType,
    *,
    placed: list[PlacedFeature],
    moonlets: list[Mapping[str, Any]],
    grid: RingGrid,
    photometry: RingPhotometry,
    os: int,
    epoch: float,
    time: float,
) -> tuple[NDArrayIntType, NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Integrate the cut and sloped pixels along their footprints.

    Parameters:
        crossing_pixels: The flat grid index of each boundary inside a footprint.
        crossing_offsets: Each boundary's offset from its pixel center, in render-grid
            pixels, strictly between -0.5 and 0.5.
        sloped_pixels: The flat grid indices of the sloped pixels to integrate by cells.
        curved_pixels: The sloped pixels that a propeller or a density wave reaches.
        placed: Every feature with its orbit and radial band.
        moonlets: The scene's moonlets, for their propellers.
        grid: The ring-plane coordinates of the pixel centers.
        photometry: The viewing geometry.
        os: The oversampling factor.
        epoch: Ring epoch in TDB seconds.
        time: Scene time in TDB seconds.

    Returns:
        The flat indices of the integrated pixels, sorted, and, for each, its emitted
        geometric factor times ``A/4 * P``, its transmission, and whether any of it
        carries tau.
    """
    order = np.lexsort((crossing_offsets, crossing_pixels))
    cut_pixels = crossing_pixels[order]
    cut_offsets = crossing_offsets[order]
    sloped = np.unique(sloped_pixels)
    curved = np.unique(curved_pixels)
    pixels = np.union1d(cut_pixels, sloped)
    # An upper bound on each pixel's cell count, used only to size the batches.  Each
    # batch holds whole pixels, so every pixel's cells are summed in the same order
    # however the pixels are batched.
    crossings = np.searchsorted(cut_pixels, pixels, side='right') - np.searchsorted(
        cut_pixels, pixels, side='left'
    )
    found = np.minimum(np.searchsorted(curved, pixels), max(curved.size - 1, 0))
    is_curved = curved[found] == pixels if curved.size else np.zeros(pixels.size, dtype=np.bool_)
    cells_bound = (crossings + 1) * np.where(is_curved, CURVED_CELLS, 1)
    batch = (np.cumsum(cells_bound) - 1) // _CELL_BATCH
    starts = np.concatenate([[0], np.flatnonzero(np.diff(batch)) + 1])
    stops = np.concatenate([starts[1:], [pixels.size]])
    results = [
        _integrate_batch(
            cut_pixels,
            cut_offsets,
            sloped,
            curved,
            first=int(pixels[start]),
            last=int(pixels[stop - 1]),
            placed=placed,
            moonlets=moonlets,
            grid=grid,
            photometry=photometry,
            os=os,
            epoch=epoch,
            time=time,
        )
        for start, stop in zip(starts, stops, strict=True)
        if stop > start
    ]
    if not results:
        return (
            pixels,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.bool_),
        )
    return (
        cast(NDArrayIntType, np.concatenate([r[0] for r in results])),
        cast(NDArrayFloatType, np.concatenate([r[1] for r in results])),
        cast(NDArrayFloatType, np.concatenate([r[2] for r in results])),
        cast(NDArrayBoolType, np.concatenate([r[3] for r in results])),
    )


def _between(values: NDArrayIntType, first: int, last: int) -> slice:
    """The slice of sorted ``values`` that lies from ``first`` through ``last``."""
    return slice(
        int(np.searchsorted(values, first, side='left')),
        int(np.searchsorted(values, last, side='right')),
    )


def _integrate_batch(
    cut_pixels: NDArrayIntType,
    cut_offsets: NDArrayFloatType,
    sloped: NDArrayIntType,
    curved: NDArrayIntType,
    *,
    first: int,
    last: int,
    placed: list[PlacedFeature],
    moonlets: list[Mapping[str, Any]],
    grid: RingGrid,
    photometry: RingPhotometry,
    os: int,
    epoch: float,
    time: float,
) -> tuple[NDArrayIntType, NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Integrate the pixels with flat indices from ``first`` through ``last``."""
    cut = _between(cut_pixels, first, last)
    cells = footprint_cells(
        cut_pixels[cut],
        cut_offsets[cut],
        sloped_pixels=sloped[_between(sloped, first, last)],
        curved_pixels=curved[_between(curved, first, last)],
        curved_cells=CURVED_CELLS,
    )
    emission, transmission, covered = _cell_integrals(
        cells, placed, moonlets, grid=grid, photometry=photometry, os=os, epoch=epoch, time=time
    )
    count = cells.pixels.size
    pixel_emission = np.bincount(cells.slot, weights=emission, minlength=count).astype(np.float64)
    pixel_transmission = np.bincount(cells.slot, weights=transmission, minlength=count).astype(
        np.float64
    )
    pixel_mask: NDArrayBoolType = np.bincount(cells.slot, weights=covered, minlength=count) > 0.0
    return cells.pixels, pixel_emission, pixel_transmission, pixel_mask


def _cell_integrals(
    cells: FootprintCells,
    placed: list[PlacedFeature],
    moonlets: list[Mapping[str, Any]],
    *,
    grid: RingGrid,
    photometry: RingPhotometry,
    os: int,
    epoch: float,
    time: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Each cell's emission, transmission, and covered length, weighted by its length.

    A constant cell has one point, at its midpoint; a linear cell has two, at its
    quarter and three-quarter points, and the line through them gives the optical
    depth at the cell's ends.
    """
    linear = cells.linear
    points = np.where(linear, 2, 1)
    cell = np.repeat(np.arange(cells.slot.size), points)
    within = np.arange(cell.size) - (np.cumsum(points) - points)[cell]
    fraction = np.where(linear[cell], 0.25 + 0.5 * within, 0.5)
    offset = cells.lo[cell] + fraction * cells.length[cell]
    pixel = cells.pixels[cells.slot[cell]]
    r_center = grid.r.ravel()[pixel]
    r_point = r_center + offset * grid.radial_scale.ravel()[pixel]
    lam_point = grid.lam.ravel()[pixel]
    signed, ap_weighted, ap_weight = _evaluate(
        r_center, r_point, lam_point, placed, photometry=photometry, os=os, epoch=epoch, time=time
    )
    # The clipped tau the constant cells read, and the unclipped tau the linear cells
    # read; a propeller factor is non-negative, so it keeps the unclipped tau's sign.
    tau = np.clip(signed, 0.0, None)
    for moonlet in moonlets:
        lobes = moonlet.get('propeller')
        if lobes:
            factor = propeller_tau_factor(moonlet, lobes, r=r_point, lam=lam_point, os=os)
            tau *= factor
            signed *= factor

    length = cells.length
    emission = np.zeros(length.shape, dtype=np.float64)
    transmission = np.zeros(length.shape, dtype=np.float64)
    covered = np.zeros(length.shape, dtype=np.float64)

    # Constant cells: the point forms at the midpoint, exactly as at a pixel center.
    first = np.cumsum(points) - points
    constant = ~linear
    at = first[constant]
    tau_mid = tau[at]
    factor = ring_reflection_factor(tau_mid, photometry.mu, photometry.mu0, lit=photometry.lit)
    emission[constant] = length[constant] * (_ap(ap_weighted[at], ap_weight[at]) * factor)
    transmission[constant] = length[constant] * np.exp(-tau_mid / photometry.mu)
    covered[constant] = length[constant] * (tau_mid > 0.0)

    # Linear cells: the line through the two points, integrated exactly.
    at_q1 = first[linear]
    at_q3 = at_q1 + 1
    quarter = signed[at_q1]
    three_quarter = signed[at_q3]
    tau_start = 1.5 * quarter - 0.5 * three_quarter
    tau_end = 1.5 * three_quarter - 0.5 * quarter
    mean_factor, mean_transmission, fraction_covered = _linear_cell_means(
        tau_start, tau_end, photometry=photometry
    )
    ap_mid = _ap(
        0.5 * (ap_weighted[at_q1] + ap_weighted[at_q3]),
        0.5 * (ap_weight[at_q1] + ap_weight[at_q3]),
    )
    cell_length = length[linear]
    emission[linear] = cell_length * fraction_covered * ap_mid * mean_factor
    transmission[linear] = cell_length * (
        fraction_covered * mean_transmission + (1.0 - fraction_covered)
    )
    covered[linear] = cell_length * fraction_covered
    return emission, transmission, covered
