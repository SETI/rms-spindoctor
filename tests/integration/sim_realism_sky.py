"""The FOM 1 sky pairing of the realism match.

A frame's sky patches are the darkest quarter of its uniform patches whose noise sigma
is positive.  :func:`pair_sky` keeps, for each frame pair, the patches that lie on dark
sky in both the real and the matched frame, so that the pooled sky statistics compare
sky noise with sky noise.  The runner (:mod:`tests.integration.sim_realism`) calls it
once per pair before it pools the frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.sim.instruments.resolvers import (
    resolve_calibrated_sky_if,
    resolve_dark_sky_ceiling_if,
    resolve_data_units,
)
from spindoctor.sim.realism.noise import SkyPatch, dark_sky_patches, noisy_patches
from tests.integration.sim_realism_support import (
    FrameSamples,
    add_sky_samples,
    median_patch_sigma,
    reported_sky_level,
)

if TYPE_CHECKING:
    from tests.integration.sim_realism import FrameRecord

__all__ = [
    'SkyPair',
    'SkyPairStatus',
    'pair_sky',
    'twins_have_sky',
]


class SkyPairStatus(StrEnum):
    """What became of one frame pair's FOM 1 sky patches.

    A frame's sky patches are the darkest quarter of its uniform patches whose noise
    sigma is positive.  The statuses are:

    - ``COMPARED``: both frames kept dark-sky patches, and those patches entered the
      pooled sky statistics.
    - ``DROPPED``: neither frame added any patches.  Either a frame didn't keep any
      dark-sky patch, or the matched frame has no sky patches while other matched
      frames of the instrument have some.
    - ``NOT_JUDGED``: the instrument doesn't have a dark-sky ceiling, so every patch
      of both frames entered the pools.  Its products are raw DN, or its entry
      doesn't set a ceiling.
    - ``NO_REAL_SKY``: the real frame has no sky patches, so neither frame added
      any.
    - ``NO_SIM_SKY``: no matched frame of the instrument has sky patches.  The real
      frame's dark-sky patches entered the real pool alone.  On an instrument
      that doesn't set a ceiling, all of its sky patches did.
    """

    COMPARED = 'compared'
    DROPPED = 'dropped'
    NOT_JUDGED = 'not_judged'
    NO_REAL_SKY = 'no_real_sky'
    NO_SIM_SKY = 'no_sim_sky'


@dataclass(frozen=True)
class SkyPair:
    """How one real frame's FOM 1 sky patches compared with its matched frame's.

    Parameters:
        image_id: The real frame's image_id.
        quantization: The real frame's downlink quantization (``'sqrt_lut'`` for a
            frame downlinked through a companding table), else None.
        status: What became of the pair's sky patches.
        real_level: The median of the means of the real frame's kept patches, in
            native units.  If the frame didn't keep any, it is the median over all
            its sky patches.  It is NaN if the frame has no sky patches.
        sim_level: The same for the matched frame.
        n_real_patches: The number of the real frame's sky patches.
        n_sim_patches: The number of the matched frame's sky patches.
        n_real_kept: The number of real sky patches kept: those at or below the
            dark-sky ceiling, or all of them if the instrument doesn't set one.  It
            is 0 when the pair is dropped because its matched frame has no sky
            patches.
        n_sim_kept: The number of the matched frame's sky patches kept, counted the
            same way.  It is 0 when either frame has no sky patches.
        real_sigma: The median noise sigma of the real patches that entered the
            pool, NaN when none did.
        sim_sigma: The same for the matched frame.
    """

    image_id: str
    quantization: str | None
    status: SkyPairStatus
    real_level: float
    sim_level: float
    n_real_patches: int
    n_sim_patches: int
    n_real_kept: int
    n_sim_kept: int
    real_sigma: float
    sim_sigma: float

    @property
    def compared(self) -> bool:
        """Whether both frames kept dark-sky patches."""
        return self.status is SkyPairStatus.COMPARED

    @property
    def real_added(self) -> bool:
        """Whether the real frame added sky samples to the pool."""
        return self.status in (
            SkyPairStatus.COMPARED,
            SkyPairStatus.NOT_JUDGED,
            SkyPairStatus.NO_SIM_SKY,
        )

    @property
    def sim_added(self) -> bool:
        """Whether the matched frame added sky samples to the pool."""
        return self.status in (SkyPairStatus.COMPARED, SkyPairStatus.NOT_JUDGED)

    @property
    def sigma_ratio(self) -> float:
        """The sim-to-real ratio of median sky sigma, NaN unless both added samples."""
        if not (np.isfinite(self.sim_sigma) and np.isfinite(self.real_sigma)):
            return float('nan')
        if self.real_sigma <= 0.0:
            return float('nan')
        return self.sim_sigma / self.real_sigma


def _sky_patches(frame: FrameSamples) -> list[SkyPatch]:
    """A frame's sky patches whose noise sigma is positive."""
    return noisy_patches(frame.sky.patches) if frame.sky is not None else []


def twins_have_sky(sims: list[FrameSamples]) -> bool:
    """Whether any of an instrument's matched frames has sky patches.

    Parameters:
        sims: The instrument's matched frames' samples.

    Returns:
        True when at least one matched frame has a sky patch whose noise sigma is
        positive.
    """
    return any(_sky_patches(sim) for sim in sims)


def _dark_sky_ceiling_level(instrument: str) -> float | None:
    """The highest mean a dark-sky patch may have on ``instrument``, or None.

    The level is the calibrated sky residual plus the instrument's dark-sky ceiling.
    It is None on a raw-DN instrument, and on a calibrated one whose entry doesn't set
    a ceiling.
    """
    if resolve_data_units(instrument) != 'calibrated_if':
        return None
    ceiling = resolve_dark_sky_ceiling_if(instrument)
    if ceiling is None:
        return None
    return resolve_calibrated_sky_if(instrument) + ceiling


def pair_sky(
    record: FrameRecord, real: FrameSamples, sim: FrameSamples, *, twins_with_sky: bool
) -> SkyPair:
    """Add one frame pair's FOM 1 sky samples from the patches that lie on dark sky.

    A frame's sky patches are the darkest quarter of its uniform patches whose noise
    sigma is positive.  The sigma of a constant tile, such as zero-filled missing
    data, can't be measured, so such a tile never becomes a patch.  On a frame
    filled with rings or a body, the sky patches lie on that content.  Comparing
    them with the other frame's dark sky would set signal-dependent noise against
    sky noise.  So on an instrument that sets a dark-sky ceiling, a patch of either
    frame counts only when its mean is at most the calibrated sky residual plus that
    ceiling.  The pair adds sky samples only when both frames keep at least one
    patch.  On an instrument without a ceiling, every patch is added.

    Parameters:
        record: The real frame's record.
        real: The real frame's samples.  The sky samples that enter the pool are
            added to it.
        sim: The matched frame's samples, which are handled the same way.
        twins_with_sky: Whether any matched frame of the instrument has sky
            patches.  If none has, the real frame's dark-sky patches are added
            alone, or all its sky patches if the instrument doesn't set a ceiling.
            If some have, a pair whose matched frame has no sky patches is dropped.

    Returns:
        The pair's sky comparison.
    """
    real_patches = _sky_patches(real)
    sim_patches = _sky_patches(sim)
    ceiling_level = _dark_sky_ceiling_level(record.instrument)
    real_kept: list[SkyPatch] = []
    sim_kept: list[SkyPatch] = []
    added_real: list[SkyPatch] = []
    added_sim: list[SkyPatch] = []
    if not real_patches:
        status = SkyPairStatus.NO_REAL_SKY
    elif not sim_patches and not twins_with_sky:
        # No matched frame of the instrument has sky, so the real frame's sky is
        # recorded alone.  Where a ceiling is set, it still has to lie on dark sky.
        real_kept = (
            real_patches if ceiling_level is None else dark_sky_patches(real_patches, ceiling_level)
        )
        if real_kept:
            status = SkyPairStatus.NO_SIM_SKY
            added_real = real_kept
        else:
            status = SkyPairStatus.DROPPED
    elif not sim_patches:
        status = SkyPairStatus.DROPPED
    elif ceiling_level is None:
        # A raw-DN frame's zero point is its bias pedestal.  The real product may
        # have had the pedestal removed, while the matched frame keeps the
        # simulator's, so the two levels can't be compared.  A calibrated camera
        # whose entry doesn't set a ceiling is handled the same way.
        status = SkyPairStatus.NOT_JUDGED
        real_kept = added_real = real_patches
        sim_kept = added_sim = sim_patches
    else:
        # Both frames are tested by level alone.  The reference is the calibrated
        # sky residual, the level at which the simulator draws empty sky.  A noise
        # sigma in the test would make the selection depend on the quantity being
        # compared.  A tolerance scaled by the matched frame's sigma shrinks when
        # the matched frame is too quiet, and one scaled by the real frame's sigma
        # grows on content.  The instrument config holds the ceiling and the
        # measurements behind it.
        real_kept = dark_sky_patches(real_patches, ceiling_level)
        sim_kept = dark_sky_patches(sim_patches, ceiling_level)
        if real_kept and sim_kept:
            status = SkyPairStatus.COMPARED
            added_real = real_kept
            added_sim = sim_kept
        else:
            status = SkyPairStatus.DROPPED
    add_sky_samples(real, added_real)
    add_sky_samples(sim, added_sim)
    return SkyPair(
        image_id=record.image_id,
        quantization=record.quantization,
        status=status,
        real_level=reported_sky_level(real_kept, real_patches),
        sim_level=reported_sky_level(sim_kept, sim_patches),
        n_real_patches=len(real_patches),
        n_sim_patches=len(sim_patches),
        n_real_kept=len(real_kept),
        n_sim_kept=len(sim_kept),
        real_sigma=median_patch_sigma(added_real),
        sim_sigma=median_patch_sigma(added_sim),
    )
