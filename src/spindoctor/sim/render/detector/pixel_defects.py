"""Defects of single pixels: hot pixels and the anti-blooming bright and dark pairs.

Both change the charge of scattered pixels.  A hot pixel leaks charge of its own, and
an anti-blooming pair moves charge between two neighbors.  Each takes an optional pool
of pixels, which an adversarial scene uses to place the defects on the navigation
features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.sim.render.detector.detector_maps import detector_hot_pixel_map
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['add_bright_dark_pairs', 'add_hot_pixels']


def add_hot_pixels(
    electrons: NDArrayFloatType,
    *,
    fraction: float,
    amplitude_e: float,
    column_factor: float,
    detector_id: str,
    rng: np.random.Generator,
    candidate_pool: tuple[NDArrayIntType, NDArrayIntType] | None = None,
    binning: int = 1,
) -> dict[str, Any]:
    """Add a hot-pixel population (electrons) in place.

    A hot pixel holds a large fixed charge; on CCDs read through it, a fraction
    of that charge contaminates the column above it (a warm streak).  The
    uniform population is the detector's fixed map
    (:func:`detector_hot_pixel_map`), the same pixels and amplitudes in every
    scene that emulates the detector.  An adversarial population drawn onto
    the navigation features comes from ``candidate_pool`` instead:
    ``round(fraction * size)`` distinct pool pixels, or the whole pool when it
    holds fewer, drawn without replacement from the scene's seeded ``rng``.  A
    zero fraction or amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        fraction: Fraction of pixels that are hot.
        amplitude_e: Hot-pixel amplitude scale in electrons (exponentially
            distributed about this scale, so a few are very hot).
        column_factor: Fraction of a hot pixel's TOTAL charge bled up its
            column: the warm streak's integral is ``column_factor`` times the
            hot pixel's charge, independent of the frame height.
        detector_id: The emulated detector the uniform population belongs to.
        rng: The stage's seeded generator, read only by an adversarial draw.
        candidate_pool: Optional ``(v, u)`` coordinate pool an adversarial
            population is drawn from.  When None or empty, the population is
            the detector's map.
        binning: The readout mode's binning factor, for the detector's map.

    Returns:
        The realized population for the truth record: the planted ``pixels``
        as ``[v, u]`` pairs and their ``amplitudes_e``, empty on a no-op.
    """
    if fraction <= 0.0 or amplitude_e <= 0.0:
        return {'pixels': [], 'amplitudes_e': []}
    size_v, size_u = electrons.shape
    if candidate_pool is not None and candidate_pool[0].size > 0:
        n_hot = round(fraction * size_v * size_u)
        if n_hot <= 0:
            return {'pixels': [], 'amplitudes_e': []}
        pool_v, pool_u = candidate_pool
        pick = rng.choice(pool_v.size, size=min(n_hot, pool_v.size), replace=False)
        hot_v = pool_v[pick]
        hot_u = pool_u[pick]
        amps = amplitude_e * rng.exponential(1.0, size=pick.size)
    else:
        hot_v, hot_u, relative = detector_hot_pixel_map(
            (size_v, size_u), fraction=fraction, detector_id=detector_id, binning=binning
        )
        amps = amplitude_e * relative
    np.add.at(electrons, (hot_v, hot_u), amps)
    record = {
        'pixels': [[int(v), int(u)] for v, u in zip(hot_v.tolist(), hot_u.tolist(), strict=True)],
        'amplitudes_e': [float(a) for a in amps.tolist()],
    }
    if column_factor <= 0.0:
        return record
    # Warm column: each hot pixel bleeds charge onto the pixels above it
    # (decreasing toward the read register), a CCD readout scar.  The linear
    # ramp is normalized so its INTEGRAL is column_factor * amp: the column
    # carries a fixed fraction of the hot pixel's charge however tall the
    # frame is, keeping the contamination conservative and size-invariant.
    for v, u, amp in zip(hot_v.tolist(), hot_u.tolist(), amps.tolist(), strict=True):
        if v <= 0:
            continue
        weights = np.linspace(1.0, 0.0, v, endpoint=False)
        weights /= weights.sum()
        electrons[:v, u] += column_factor * amp * weights
    return record


def add_bright_dark_pairs(
    electrons: NDArrayFloatType,
    *,
    count: int,
    amplitude_e: float,
    rng: np.random.Generator,
    candidate_pool: tuple[NDArrayIntType, NDArrayIntType] | None = None,
) -> dict[str, Any]:
    """Deposit scattered vertical bright/dark pixel pairs (electrons) in place.

    The Cassini anti-blooming mode produces isolated vertical two-pixel pairs in
    unsummed long exposures: one pixel raised by ``amplitude_e``, the pixel below
    it lowered by the same charge.  Placement is uniform, or drawn from
    ``candidate_pool`` for adversarial placement onto the navigation features.  A
    zero count or amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        count: Number of bright/dark pairs to deposit.
        amplitude_e: Pair amplitude in electrons.
        rng: The mode's seeded generator.
        candidate_pool: Optional ``(v, u)`` pool for adversarial placement.

    Returns:
        The realized pair geometry for the truth record.
    """
    if count <= 0 or amplitude_e <= 0.0:
        return {'pairs': []}
    size_v, size_u = electrons.shape
    if candidate_pool is not None and candidate_pool[0].size > 0:
        pool_v, pool_u = candidate_pool
        pick = rng.integers(0, pool_v.size, size=count)
        bv = np.clip(pool_v[pick], 0, size_v - 2)
        bu = np.clip(pool_u[pick], 0, size_u - 1)
    else:
        bv = rng.integers(0, size_v - 1, size=count)
        bu = rng.integers(0, size_u, size=count)
    pairs: list[list[int]] = []
    for v, u in zip(bv.tolist(), bu.tolist(), strict=True):
        electrons[v, u] += amplitude_e
        electrons[v + 1, u] -= amplitude_e
        pairs.append([int(v), int(u)])
    return {'pairs': pairs}
