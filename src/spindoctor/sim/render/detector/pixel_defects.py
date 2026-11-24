"""Defects of single pixels: hot pixels and the anti-blooming bright and dark pairs.

Both change the charge of scattered pixels.  A hot pixel leaks charge of its own, and
an anti-blooming pair moves charge between two neighbors.  Each takes an optional pool
of pixels, which an adversarial scene uses to place the defects on the navigation
features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.sim.render.detector.detector_maps import (
    EXPONENTIAL,
    AmplitudeLaw,
    detector_hot_pixel_map,
)
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['HotPixelPopulation', 'add_bright_dark_pairs', 'add_hot_pixels']


@dataclass(frozen=True)
class HotPixelPopulation:
    """A hot-pixel population: how many pixels, how hot, and what each leaves behind.

    Each hot pixel has a relative amplitude ``a`` drawn from ``law``.  Its own charge
    is ``a * (rate_e_per_sec * (exposure + readout_sec) + readout_e)``, and each packet
    read through it picks up ``a * (rate_e_per_sec * streak_sec + streak_e)`` per
    native row.  The ``_sec`` terms tie a charge to the pixel's dark rate; the ``_e``
    terms are fixed charges, for a detector whose hot-pixel charge and streak are
    constants of the frame mode and do not follow the exposure.

    Parameters:
        fraction: Fraction of pixels that are hot.
        law: The distribution of the relative amplitudes.
        rate_e_per_sec: Dark rate at relative amplitude 1, in electrons per second.
        readout_sec: Seconds of the pixel's dark rate it collects outside the
            exposure each frame.
        readout_e: Charge at relative amplitude 1 the pixel collects each frame
            whatever the exposure, in electrons.
        streak_sec: Seconds each packet read through the pixel dwells there, so
            the seconds of its dark rate the packet picks up per native row.
        streak_e: Charge at relative amplitude 1 each packet read through the pixel
            picks up whatever its rate, in electrons per native row.
    """

    fraction: float
    law: AmplitudeLaw = field(default=EXPONENTIAL)
    rate_e_per_sec: float = 0.0
    readout_sec: float = 0.0
    readout_e: float = 0.0
    streak_sec: float = 0.0
    streak_e: float = 0.0

    def charge_per_unit(self, exposure_sec: float) -> float:
        """The charge of a pixel of relative amplitude 1 at an exposure, in electrons.

        Parameters:
            exposure_sec: The frame's exposure time in seconds; a negative one counts
                as zero.

        Returns:
            The charge.
        """
        return self.rate_e_per_sec * (max(exposure_sec, 0.0) + self.readout_sec) + self.readout_e

    @property
    def streak_per_unit(self) -> float:
        """The per-row streak of a pixel of relative amplitude 1, in electrons."""
        return self.rate_e_per_sec * self.streak_sec + self.streak_e


def add_hot_pixels(
    electrons: NDArrayFloatType,
    population: HotPixelPopulation,
    *,
    exposure_sec: float,
    detector_id: str,
    rng: np.random.Generator,
    candidate_pool: tuple[NDArrayIntType, NDArrayIntType] | None = None,
    binning: int = 1,
) -> dict[str, Any]:
    """Add a hot-pixel population and its warm streaks (electrons) in place.

    A hot pixel is a pixel with a high dark current.  Its own charge and its streak
    follow :class:`HotPixelPopulation`.  During readout every line moves one row
    toward line 0, which is read first, so the charge packets of all the rows below
    the hot pixel (larger line numbers) pass through it on their way out.  Each
    packet picks up the same charge there whatever the exposure: the dark charge the
    pixel generates while the packet sits there (the ``_sec`` term), or a fixed
    charge on a detector whose hot-pixel charge and streak are constants of the
    frame mode (the ``_e`` term).  The warm streak is therefore flat and lies only
    on the rows read after the hot pixel.  A binned row holds
    ``binning`` native rows that each passed the hot pixel, so it collects
    ``binning`` times the native row's streak; no dark frame has tested that
    factor.

    The uniform population is the detector's fixed map
    (:func:`detector_hot_pixel_map`), the same pixels and relative amplitudes in
    every scene that emulates the detector.  An adversarial population drawn onto
    the navigation features comes from ``candidate_pool`` instead:
    ``round(fraction * size)`` distinct pool pixels, or the whole pool when it holds
    fewer, drawn without replacement from the scene's seeded ``rng``, with relative
    amplitudes from the same law.  A zero fraction, or a population that adds no
    charge, is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        population: The population's density, amplitude law, and charges.
        exposure_sec: The frame's exposure time in seconds.
        detector_id: The emulated detector the uniform population belongs to.
        rng: The stage's seeded generator, read only by an adversarial draw.
        candidate_pool: Optional ``(v, u)`` coordinate pool an adversarial
            population is drawn from.  When None or empty, the population is
            the detector's map.
        binning: The readout mode's binning factor, for the detector's map and the
            streak.

    Returns:
        The realized population for the truth record: the planted ``pixels`` as
        ``[v, u]`` pairs and the charge planted in each, ``charges_e``, both empty
        on a no-op.
    """
    charge_per_unit = population.charge_per_unit(exposure_sec)
    streak_per_unit = population.streak_per_unit
    if population.fraction <= 0.0 or (charge_per_unit <= 0.0 and streak_per_unit <= 0.0):
        return {'pixels': [], 'charges_e': []}
    size_v, size_u = electrons.shape
    if candidate_pool is not None and candidate_pool[0].size > 0:
        n_hot = round(population.fraction * size_v * size_u)
        if n_hot <= 0:
            return {'pixels': [], 'charges_e': []}
        pool_v, pool_u = candidate_pool
        pick = rng.choice(pool_v.size, size=min(n_hot, pool_v.size), replace=False)
        hot_v = pool_v[pick]
        hot_u = pool_u[pick]
        relative = population.law.from_uniform(rng.random(pick.size))
    else:
        hot_v, hot_u, relative = detector_hot_pixel_map(
            (size_v, size_u),
            fraction=population.fraction,
            detector_id=detector_id,
            binning=binning,
            law=population.law,
        )
    charges = charge_per_unit * relative
    np.add.at(electrons, (hot_v, hot_u), charges)
    record = {
        'pixels': [[int(v), int(u)] for v, u in zip(hot_v.tolist(), hot_u.tolist(), strict=True)],
        'charges_e': [float(c) for c in charges.tolist()],
    }
    if streak_per_unit <= 0.0:
        return record
    # The warm streak: each hot pixel adds its per-row pickup at the row after it, and
    # a running sum down the columns carries it to the last line.
    below = hot_v + 1 < size_v
    steps = np.zeros((size_v, size_u), dtype=np.float64)
    np.add.at(steps, (hot_v[below] + 1, hot_u[below]), streak_per_unit * binning * relative[below])
    electrons += np.cumsum(steps, axis=0)
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
