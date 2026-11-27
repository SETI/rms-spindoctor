"""Radial optical-depth profiles of the ring features and the pixel footprint.

A ring feature's optical depth is a function of ring-plane radius.  The image-side
renderer evaluates it with sharp boundaries and integrates each pixel across them.
Where a boundary cuts through a pixel, the pixel mixes the two sides, weighted by how
much of the pixel lies on each.  The brightness and transmission are computed for each
side at its full optical depth and then mixed.  Scaling the optical depth by the
covered fraction instead would make the mixed pixel too bright, because brightness
levels off as optical depth grows.

The pixel footprint is a radial segment one render-grid pixel long, centered on the
pixel center: offsets ``s`` from -0.5 to 0.5 in render-grid pixels, at ring-plane radius
``r + s * radial_scale``, where ``radial_scale`` is the ring-plane radial distance per
image pixel there.  This is the one-pixel linear coverage ramp: a single boundary at
offset ``s_b`` leaves ``0.5 + s_b`` of the pixel on its inner side.  Where a boundary
falls inside the footprint, the footprint is cut into segments at every boundary
inside it, and :mod:`spindoctor.sim.render.rings.footprint` integrates each segment.
A pixel that a ramp's band overlaps is integrated too, as one segment if no boundary
falls inside it.  Any other pixel is evaluated at its center.  On an oversampled grid
the box downsample then averages the render-grid pixels into each detector pixel, so a
boundary is integrated over exactly one detector pixel.

Profiles by kind, in render-grid ring-plane units (``r_e`` is the feature's orbit
radius at the pixel's longitude, ``r_o`` the orbit radius widened by ``width_px``):

- ``ringlet`` and ``gap``: ``tau`` for ``r_e <= r <= r_o``; both ends are sharp.  The
  caller subtracts a gap's profile from the features under it.
- ``edge``: ``tau`` for ``r <= r_e`` (``side: 'in'``) or ``r >= r_e``
  (``side: 'out'``); the step is sharp.
- ``ramp``: linear from ``r_e`` to ``r_e + w``, rising from 0 to ``tau``
  (``side: 'out'``) or falling from ``tau`` to 0 (``side: 'in'``), and zero inward of
  ``r_e``.  An outward ramp holds ``tau`` from ``r_e + w`` to ``r_o`` and drops to 0
  there; an inward ramp stays at 0 outward of ``r_e + w``.  On a circular orbit
  ``r_o = r_e + w``; on an eccentric one the widened orbit is not exactly ``w`` out, so
  an outward ramp levels off before ``r_o`` or is cut short by it.  The step at
  ``r_e`` (inward), the step at ``r_o`` (outward), and the bends at both ends of the
  linear part are all boundaries.
- ``wave``: ``tau * exp(-x / damping) * sin(2 * pi * x / wavelength)`` for
  ``x = r - r_e >= 0``, and zero inward of the launch radius.  It is continuous, so it
  has no sharp boundary.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from spindoctor.sim.scene.fields import RING_SIDE_DEFAULTS, scene_value
from spindoctor.sim.shared.inplace import result_array, ufunc_result
from spindoctor.sim.shared.ring import RingOrbit, compute_orbit_radii
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = [
    'FOOTPRINT_HALF_WIDTH_PX',
    'WAVE_UNDERFLOW_DAMPINGS',
    'FeatureBoundaries',
    'FootprintCells',
    'SlopedBand',
    'feature_boundaries',
    'feature_breakpoint_radii',
    'feature_sloped_band',
    'feature_tau',
    'footprint_cells',
]

FOOTPRINT_HALF_WIDTH_PX: float = 0.5
"""Half the length of a pixel's radial footprint, in render-grid pixels."""

WAVE_UNDERFLOW_DAMPINGS: float = 750.0
"""Damping lengths past which a wave train's tau is exactly 0.0.

exp(-x) underflows to exactly 0.0 in float64 before x reaches 750, so the radial
bound that uses this stays exact rather than approximate.
"""


@dataclass(frozen=True)
class FeatureBoundaries:
    """A feature's orbit radii at a set of longitudes.

    Parameters:
        inner: The orbit radius ``r_e`` at each longitude.
        outer: The orbit radius widened by ``width_px`` at each longitude, or None for
            the kinds that have no width (``edge`` and ``wave``).
    """

    inner: NDArrayFloatType
    outer: NDArrayFloatType | None


@dataclass(frozen=True)
class FootprintCells:
    """The pieces of the integrated pixels' footprints.

    Parameters:
        pixels: The flat grid indices of the integrated pixels, sorted and unique: each
            has a boundary inside its footprint or lies across a ramp's band.
        slot: For each cell, the index into ``pixels`` of its pixel.
        lo: For each cell, where it starts, in render-grid pixels from the pixel center.
        length: For each cell, its length as a fraction of the footprint; the lengths of
            one pixel sum to 1.
        linear: For each cell, whether the optical depth changes along it.  Such a
            cell lies in a pixel that a ramp's band overlaps; every other cell has a
            constant optical depth.
    """

    pixels: NDArrayIntType
    slot: NDArrayIntType
    lo: NDArrayFloatType
    length: NDArrayFloatType
    linear: NDArrayBoolType


def feature_boundaries(
    feature: Mapping[str, Any],
    orbit: RingOrbit,
    lam: NDArrayFloatType,
    *,
    os: int,
    epoch: float,
    time: float,
) -> FeatureBoundaries:
    """Return a feature's orbit radii at ring-plane longitudes ``lam``.

    Parameters:
        feature: The validated feature mapping.
        orbit: The feature's orbit, scaled to the render grid.
        lam: Ring-plane longitudes from the ascending node, in radians.
        os: The oversampling factor, which scales ``width_px``.
        epoch: Ring epoch in TDB seconds.
        time: Scene time in TDB seconds.

    Returns:
        The radii, in render-grid ring-plane units.
    """
    kind = str(feature.get('kind'))
    inner = compute_orbit_radii(lam, orbit, epoch=epoch, time=time)
    outer: NDArrayFloatType | None = None
    if kind in ('ringlet', 'gap', 'ramp'):
        width = float(scene_value(feature, 'width_px', 0.0)) * os
        outer = compute_orbit_radii(lam, orbit.widened(width), epoch=epoch, time=time)
    return FeatureBoundaries(inner=inner, outer=outer)


def _outer(bounds: FeatureBoundaries) -> NDArrayFloatType:
    """Return the widened radii of a banded kind."""
    if bounds.outer is None:
        raise ValueError('a banded ring feature needs its outer radii')
    return bounds.outer


def feature_tau(
    feature: Mapping[str, Any],
    r: NDArrayFloatType,
    bounds: FeatureBoundaries,
    *,
    os: int,
) -> NDArrayFloatType:
    """Return one feature's optical depth at radii ``r``, with sharp boundaries.

    Only the ``wave`` kind returns negative values.  A ``gap``'s optical depth is
    positive here and subtracted by the caller.

    Parameters:
        feature: The validated feature mapping.
        r: Ring-plane radii in render-grid units, one for each entry of ``bounds``.
        bounds: The feature's orbit radii at the longitudes of ``r``.
        os: The oversampling factor, which scales the kind's shape keys.

    Returns:
        The optical depth at each radius.

    Raises:
        ValueError: If the feature's kind is not a renderable profile (the validator
            rejects these; this guards direct callers).
    """
    kind = str(feature.get('kind'))
    tau = float(scene_value(feature, 'tau', 0.0))
    r_edge = bounds.inner
    if kind in ('ringlet', 'gap'):
        inside = (r >= r_edge) & (r <= _outer(bounds))
        return cast(NDArrayFloatType, np.where(inside, tau, 0.0))
    if kind == 'edge':
        side = str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['edge']))
        inside = (r <= r_edge) if side == 'in' else (r >= r_edge)
        return cast(NDArrayFloatType, np.where(inside, tau, 0.0))
    if kind == 'ramp':
        width = float(scene_value(feature, 'width_px', 0.0)) * os
        side = str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['ramp']))
        x = np.clip((r - r_edge) / width, 0.0, 1.0)
        if side == 'out':
            return cast(NDArrayFloatType, np.where(r <= _outer(bounds), tau * x, 0.0))
        return cast(NDArrayFloatType, np.where(r >= r_edge, tau * (1.0 - x), 0.0))
    if kind == 'wave':
        wavelength = float(scene_value(feature, 'wavelength_px', 0.0)) * os
        damping = float(scene_value(feature, 'density_wave_damping_px', 0.0)) * os
        # Clipping x at zero makes the profile exactly zero inward of the launch
        # radius (sin(0) = 0) and keeps the envelope from growing without bound there.
        x = np.subtract(r, r_edge, out=result_array(r, r_edge))
        np.clip(x, 0.0, None, out=x)
        # tau * exp(-x / damping) * sin(2 * pi * x / wavelength), built in place: a
        # wave train reaches over most of an oversampled frame, where each fresh
        # temporary costs more than its arithmetic.  Every step is the IEEE operation
        # of that expression, so the profile is unchanged to the last bit.
        envelope = np.negative(x, out=np.empty_like(x))
        envelope /= damping
        np.exp(envelope, out=envelope)
        envelope *= tau
        x *= 2.0 * math.pi
        x /= wavelength
        np.sin(x, out=x)
        envelope *= x
        return ufunc_result(envelope)
    raise ValueError(f'ring feature kind {kind!r} is not renderable')


def _ramp_linear_end(
    feature: Mapping[str, Any], bounds: FeatureBoundaries, *, os: int
) -> NDArrayFloatType:
    """Return where a ramp's linear part ends: ``width_px`` outward of its orbit."""
    width = float(scene_value(feature, 'width_px', 0.0)) * os
    end: NDArrayFloatType = bounds.inner + width
    return end


def feature_breakpoint_radii(
    feature: Mapping[str, Any], bounds: FeatureBoundaries, *, os: int
) -> list[NDArrayFloatType]:
    """Return the radii where a feature's optical depth steps or bends.

    A ramp changes linearly from its orbit radius to ``width_px`` outward of it.  On an
    eccentric orbit the widened orbit that bounds an outward ramp does not sit exactly
    ``width_px`` out, so an outward ramp either levels off at full tau before its outer
    boundary or is cut short by it, and an inward ramp ignores it.

    Parameters:
        feature: The validated feature mapping.
        bounds: The feature's orbit radii.
        os: The oversampling factor, which scales ``width_px``.

    Returns:
        One array of radii per boundary: both ends of a ringlet or gap; the orbit
        radius, the end of the linear part, and (outward) the outer boundary of a ramp;
        the step of an edge; and none for a wave.
    """
    kind = str(feature.get('kind'))
    if kind in ('ringlet', 'gap'):
        return [bounds.inner, _outer(bounds)]
    if kind == 'ramp':
        side = str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['ramp']))
        linear_end = _ramp_linear_end(feature, bounds, os=os)
        if side == 'out':
            return [bounds.inner, linear_end, _outer(bounds)]
        return [bounds.inner, linear_end]
    if kind == 'edge':
        return [bounds.inner]
    return []


@dataclass(frozen=True)
class SlopedBand:
    """Where a ramp's optical depth changes with radius, and how fast.

    Parameters:
        inner: The band's inner radius at each longitude.
        outer: The band's outer radius at each longitude: where the ramp's linear part
            ends, or an outward ramp's outer boundary if that comes first.
        rate: The change in optical depth per render-grid ring-plane unit of radius:
            positive for a ramp that rises outward, negative for one that falls.
    """

    inner: NDArrayFloatType
    outer: NDArrayFloatType
    rate: float


def feature_sloped_band(
    feature: Mapping[str, Any], bounds: FeatureBoundaries, *, os: int
) -> SlopedBand | None:
    """Return the band in which a feature's optical depth changes with radius.

    Parameters:
        feature: The validated feature mapping.
        bounds: The feature's orbit radii.
        os: The oversampling factor, which scales ``width_px``.

    Returns:
        A ramp's band, or None for the other kinds.  A ringlet, gap, or edge is
        constant between its boundaries, and a wave is point sampled.
    """
    if str(feature.get('kind')) != 'ramp':
        return None
    tau = float(scene_value(feature, 'tau', 0.0))
    width = float(scene_value(feature, 'width_px', 0.0)) * os
    side = str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['ramp']))
    linear_end = _ramp_linear_end(feature, bounds, os=os)
    if side == 'out':
        return SlopedBand(
            inner=bounds.inner,
            outer=cast(NDArrayFloatType, np.minimum(linear_end, _outer(bounds))),
            rate=tau / width,
        )
    return SlopedBand(inner=bounds.inner, outer=linear_end, rate=-tau / width)


def footprint_cells(
    pixels: NDArrayIntType,
    offsets: NDArrayFloatType,
    *,
    sloped_pixels: NDArrayIntType,
    curved_pixels: NDArrayIntType,
    curved_cells: int,
) -> FootprintCells:
    """Cut pixel footprints at the boundaries inside them.

    Each pixel's footprint runs from -0.5 to 0.5 render-grid pixels.  The boundaries a
    pixel lists split it into segments, and a segment of zero length is dropped.  Each
    segment is one cell, except in a curved pixel, where it is cut into
    ``curved_cells`` equal cells.  A cell in a sloped pixel is linear.

    Parameters:
        pixels: The flat grid index of each boundary crossing; a pixel appears once for
            every boundary inside its footprint.
        offsets: The offset of each crossing from its pixel center, in render-grid
            pixels, strictly between -0.5 and 0.5.
        sloped_pixels: The flat grid index of each pixel a ramp's band overlaps, in any
            order and with repeats.
        curved_pixels: The sloped pixels whose optical depth also curves, because a
            propeller or a density wave reaches them.
        curved_cells: How many cells each segment of a curved pixel is cut into.

    Returns:
        The cells.
    """
    half = FOOTPRINT_HALF_WIDTH_PX
    order = np.lexsort((offsets, pixels))
    cut_pixels = pixels[order]
    cut_offsets = offsets[order]
    # Each crossing ends the segment that starts at the previous crossing of its pixel,
    # or at -0.5 for a pixel's first crossing; each pixel's last crossing also starts
    # a segment that runs to 0.5.  A sloped pixel that no boundary crosses is one
    # segment from -0.5 to 0.5.
    run_start = np.ones(cut_pixels.size, dtype=np.bool_)
    run_start[1:] = cut_pixels[1:] != cut_pixels[:-1]
    run_end = np.ones(cut_pixels.size, dtype=np.bool_)
    run_end[:-1] = run_start[1:]
    previous = np.empty_like(cut_offsets)
    previous[1:] = cut_offsets[:-1]
    previous[run_start] = -half
    cut_unique = cut_pixels[run_start]
    sloped_unique = np.unique(sloped_pixels)
    uncut = sloped_unique
    if uncut.size:
        found = np.searchsorted(cut_unique, uncut)
        hit = found < cut_unique.size
        hit[hit] = cut_unique[found[hit]] == uncut[hit]
        uncut = uncut[~hit]
    seg_pixels = np.concatenate([cut_pixels, cut_pixels[run_end], uncut])
    lo = np.concatenate([previous, cut_offsets[run_end], np.full(uncut.size, -half)])
    hi = np.concatenate([cut_offsets, np.full(int(run_end.sum()), half), np.full(uncut.size, half)])
    length = hi - lo
    keep = length > 0.0
    lo = lo[keep]
    length = length[keep]
    unique = np.union1d(cut_unique, uncut) if uncut.size else cut_unique
    seg_slot = np.searchsorted(unique, seg_pixels[keep])

    is_sloped = np.zeros(unique.size, dtype=np.bool_)
    is_sloped[np.searchsorted(unique, sloped_unique)] = True
    is_curved = np.zeros(unique.size, dtype=np.bool_)
    is_curved[np.searchsorted(unique, curved_pixels)] = True
    count = np.where(is_curved[seg_slot], curved_cells, 1)
    segment = np.repeat(np.arange(lo.size), count)
    within = np.arange(segment.size) - (np.cumsum(count) - count)[segment]
    cell_length = length[segment] / count[segment]
    return FootprintCells(
        pixels=unique,
        slot=cast(NDArrayIntType, seg_slot[segment]),
        lo=cast(NDArrayFloatType, lo[segment] + within * cell_length),
        length=cast(NDArrayFloatType, cell_length),
        linear=cast(NDArrayBoolType, is_sloped[seg_slot][segment]),
    )
