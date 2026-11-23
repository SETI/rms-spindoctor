"""Maps fixed to the emulated detector: its hot pixels and its column pattern.

A real camera shows the same hot pixels and the same column pattern in every frame, so
each map is keyed on the detector and never on a scene seed.  Every scene that emulates
the detector sees the same values at the same coordinates, at any frame size.
"""

from __future__ import annotations

import hashlib

import numpy as np

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['detector_column_map', 'detector_hot_pixel_map']

# The hot-pixel map hashes each pixel's (v, u) as v * stride + u, so the stride
# bounds the detector width it keys uniquely; every emulated camera is far
# narrower.
_DETECTOR_STRIDE = 1 << 20
# SplitMix64 constants (Steele, Lea, and Flood 2014): a fast, well-mixed
# bijection on 64-bit words, applied element-wise to the pixel keys.
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
_UNIT_BITS = 53


def _splitmix64(words: NDArrayIntType) -> NDArrayIntType:
    """One SplitMix64 step on each 64-bit word (wrapping unsigned arithmetic)."""
    z = words + _SPLITMIX_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2
    return z ^ (z >> np.uint64(31))


def _unit_interval(words: NDArrayIntType) -> NDArrayFloatType:
    """Map 64-bit words to uniform floats in [0, 1) from their top 53 bits."""
    return (words >> np.uint64(64 - _UNIT_BITS)).astype(np.float64) * 2.0**-_UNIT_BITS


def detector_hot_pixel_map(
    shape: tuple[int, int], *, fraction: float, detector_id: str, binning: int = 1
) -> tuple[NDArrayIntType, NDArrayIntType, NDArrayFloatType]:
    """The emulated detector's fixed hot-pixel population over a frame.

    Each detector pixel ``(v, u)`` is hot when a uniform draw keyed on the
    detector and the pixel, never on a scene seed, falls below ``fraction``, so
    every scene emulating the detector sees the same hot pixels at the same
    coordinates, at any frame size (a smaller frame sees the corner of the
    map from ``(0, 0)``).  The count is binomial with mean ``fraction`` times
    the pixel count, and raising ``fraction`` only adds pixels.  Each hot
    pixel's relative amplitude is a fixed exponential draw keyed the same way.
    A binned frame reads the native map: each frame pixel holds the summed
    amplitudes of the hot native pixels in its block, so the same hot pixels
    show at every binning.

    Parameters:
        shape: The frame's ``(size_v, size_u)`` in frame pixels.
        fraction: Fraction of native detector pixels that are hot.
        detector_id: The emulated detector's name; the calibrated and raw
            products of one camera share it.
        binning: The readout mode's binning factor.

    Returns:
        ``(hot_v, hot_u, relative_amplitudes)`` in frame pixels, row-major and
        each pixel once; a native amplitude is exponentially distributed with
        mean 1.
    """
    if binning > 1:
        native_v, native_u, native_amps = detector_hot_pixel_map(
            (shape[0] * binning, shape[1] * binning), fraction=fraction, detector_id=detector_id
        )
        summed = np.zeros(shape, dtype=np.float64)
        np.add.at(summed, (native_v // binning, native_u // binning), native_amps)
        hot_v, hot_u = np.nonzero(summed)
        return hot_v.astype(np.intp), hot_u.astype(np.intp), summed[hot_v, hot_u]
    size_v, size_u = shape
    key = int.from_bytes(hashlib.sha256(f'hot_pixels:{detector_id}'.encode()).digest()[:8], 'big')
    vs, us = np.indices((size_v, size_u), dtype=np.uint64)
    words = _splitmix64((vs * np.uint64(_DETECTOR_STRIDE) + us) ^ np.uint64(key))
    hot_v, hot_u = np.nonzero(_unit_interval(words) < fraction)
    amp_words = _splitmix64(words[hot_v, hot_u])
    relative = -np.log1p(-_unit_interval(amp_words))
    return hot_v.astype(np.intp), hot_u.astype(np.intp), relative


def detector_column_map(size_u: int, *, effect: str, detector_id: str) -> NDArrayFloatType:
    """A fixed standard-normal value per native detector column.

    Each column ``u`` draws a Gaussian keyed on the effect name, the detector,
    and the column, never on a scene seed, so every scene emulating the
    detector sees the same values in the same columns, at any frame width.

    Parameters:
        size_u: The number of native columns.
        effect: The effect the map belongs to, so two effects draw
            independent maps.
        detector_id: The emulated detector's name.

    Returns:
        A length-``size_u`` array of standard-normal values.
    """
    key = int.from_bytes(hashlib.sha256(f'{effect}:{detector_id}'.encode()).digest()[:8], 'big')
    us = np.arange(size_u, dtype=np.uint64)
    first = _splitmix64(us ^ np.uint64(key))
    second = _splitmix64(first)
    # Box-Muller; 1 - u keeps the logarithm's argument in (0, 1].
    radius = np.sqrt(-2.0 * np.log1p(-_unit_interval(first)))
    result: NDArrayFloatType = radius * np.cos(2.0 * np.pi * _unit_interval(second))
    return result
