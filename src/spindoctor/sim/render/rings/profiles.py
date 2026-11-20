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
inside it, and each segment is evaluated once, at its midpoint, with a weight equal to
its length.  A pixel whose footprint holds no boundary is evaluated at its center.  On
an oversampled grid the box downsample then averages the render-grid pixels into each
detector pixel, so a boundary is integrated over exactly one detector pixel.

Profiles by kind, in render-grid ring-plane units (``r_e`` is the feature's orbit
radius at the pixel's longitude, ``r_o`` the orbit radius widened by ``width_px``):

- ``ringlet`` and ``gap``: ``tau`` for ``r_e <= r <= r_o``; both ends are sharp.  The
  caller subtracts a gap's profile from the features under it.
- ``edge``: ``tau`` for ``r <= r_e`` (``side: 'in'``) or ``r >= r_e``
  (``side: 'out'``); the step is sharp.
- ``ramp``: a linear transition across ``[r_e, r_o]``, from 0 at ``r_e`` to ``tau`` at
  ``r_o`` (``side: 'out'``) or mirrored (``side: 'in'``), and zero outside the band.
  The end that carries ``tau`` is sharp; the end at 0 is continuous.
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
from spindoctor.sim.shared.ring import RingOrbit, compute_orbit_radii
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'FOOTPRINT_HALF_WIDTH_PX',
    'FeatureBoundaries',
    'FootprintSamples',
    'feature_boundaries',
    'feature_sharp_radii',
    'feature_tau',
    'footprint_samples',
]

FOOTPRINT_HALF_WIDTH_PX: float = 0.5
"""Half the length of a pixel's radial footprint, in render-grid pixels."""


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
class FootprintSamples:
    """The segments of the pixel footprints that sharp boundaries cut.

    Parameters:
        pixels: The flat grid indices of the pixels that have a boundary inside their
            footprint, sorted and unique.
        slot: For each segment, the index into ``pixels`` of its pixel.
        offset: For each segment, its midpoint, in render-grid pixels from the pixel
            center.
        weight: For each segment, its length as a fraction of the footprint; the
            weights of one pixel sum to 1.
    """

    pixels: NDArrayIntType
    slot: NDArrayIntType
    offset: NDArrayFloatType
    weight: NDArrayFloatType


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
        x = np.clip(r - r_edge, 0.0, None)
        return cast(
            NDArrayFloatType,
            tau * np.exp(-x / damping) * np.sin(2.0 * math.pi * x / wavelength),
        )
    raise ValueError(f'ring feature kind {kind!r} is not renderable')


def feature_sharp_radii(
    feature: Mapping[str, Any], bounds: FeatureBoundaries
) -> list[NDArrayFloatType]:
    """Return the radii where a feature's optical depth steps.

    Parameters:
        feature: The validated feature mapping.
        bounds: The feature's orbit radii.

    Returns:
        One array of radii per sharp boundary: both ends of a ringlet or gap, the step
        of an edge, the end of a ramp that carries ``tau``, and none for a wave.
    """
    kind = str(feature.get('kind'))
    if kind in ('ringlet', 'gap'):
        return [bounds.inner, _outer(bounds)]
    if kind == 'edge':
        return [bounds.inner]
    if kind == 'ramp':
        side = str(scene_value(feature, 'side', RING_SIDE_DEFAULTS['ramp']))
        return [_outer(bounds)] if side == 'out' else [bounds.inner]
    return []


def footprint_samples(pixels: NDArrayIntType, offsets: NDArrayFloatType) -> FootprintSamples:
    """Cut pixel footprints at the boundaries inside them.

    Each pixel's footprint runs from -0.5 to 0.5 render-grid pixels.  The boundaries a
    pixel lists split it into segments, and each segment becomes one sample at its
    midpoint, weighted by its length.  A segment of zero length is dropped.

    Parameters:
        pixels: The flat grid index of each boundary crossing; a pixel appears once for
            every boundary inside its footprint.
        offsets: The offset of each crossing from its pixel center, in render-grid
            pixels, strictly between -0.5 and 0.5.

    Returns:
        The segments.
    """
    unique = np.unique(pixels)
    half = FOOTPRINT_HALF_WIDTH_PX
    all_pixels = np.concatenate([pixels, unique, unique])
    all_offsets = np.concatenate([offsets, np.full(unique.size, -half), np.full(unique.size, half)])
    order = np.lexsort((all_offsets, all_pixels))
    sorted_pixels = all_pixels[order]
    sorted_offsets = all_offsets[order]
    # Consecutive breakpoints of one pixel bound one segment; each pixel's run starts
    # at -0.5 and ends at 0.5, so a pair that spans two pixels is not a segment.
    same = sorted_pixels[:-1] == sorted_pixels[1:]
    lo = sorted_offsets[:-1][same]
    hi = sorted_offsets[1:][same]
    seg_pixels = sorted_pixels[:-1][same]
    weight = hi - lo
    keep = weight > 0.0
    return FootprintSamples(
        pixels=unique,
        slot=cast(NDArrayIntType, np.searchsorted(unique, seg_pixels[keep])),
        offset=cast(NDArrayFloatType, 0.5 * (lo[keep] + hi[keep])),
        weight=cast(NDArrayFloatType, weight[keep]),
    )
