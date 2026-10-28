"""Measured artifact incidence rates for realism FOM 6.

These detectors measure, per frame, the rates of the artifact classes the
catalog defaults describe: line loss (missing / interpolated rows),
stationary hot pixels, and transient single-pixel spikes (cosmic rays and
radiation hits).  The same detectors run on real cohort frames and matched
sim frames, so a detector's bias cancels in the comparison; the absolute
rates additionally document the cohort against the catalog's per-instrument
defaults (loss modes stay off under instrument_defaults, their activation
keys at zero -- FOM 6 is the evidence for keeping or replacing those zeros).

The hot-pixel / transient-spike split needs more than one frame: a spike
that recurs at one detector position across frames is a hot pixel, one that
does not is a transient.  ``measure_artifact_incidence`` therefore reports
per-frame spike candidates, and ``split_stationary_spikes`` performs the
cross-frame split.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'ArtifactIncidence',
    'measure_artifact_incidence',
    'split_stationary_spikes',
]


@dataclass(frozen=True)
class ArtifactIncidence:
    """Per-frame artifact measurements.

    Parameters:
        missing_line_fraction: Fraction of rows that are constant or exact
            neighbor interpolations (telemetry line loss).
        spike_fraction: Fraction of pixels flagged as isolated positive
            spikes (hot pixels + transients, split across frames by
            :func:`split_stationary_spikes`).
        spike_positions_vu: ``(K, 2)`` integer positions of the flagged
            spikes, for the cross-frame stationary split.
    """

    missing_line_fraction: float
    spike_fraction: float
    spike_positions_vu: NDArrayFloatType


def _row_is_lost(row: NDArrayFloatType, above: NDArrayFloatType, below: NDArrayFloatType) -> bool:
    """True when ``row`` is constant or the exact mean of its neighbors."""
    if float(np.ptp(row)) == 0.0:
        return True
    interpolated = 0.5 * (above + below)
    return bool(np.allclose(row, interpolated, rtol=0.0, atol=1e-12))


def measure_artifact_incidence(
    image: NDArrayFloatType, *, spike_q99_factor: float = 10.0
) -> ArtifactIncidence:
    """Measure line loss and single-pixel spikes on one frame.

    Line loss: a row counts as lost when it is constant while its
    neighbors are not, or when it exactly equals the mean of the adjacent
    rows (the interpolation a telemetry gap filler leaves).  Constant rows
    in an all-constant frame (a blank frame) are not counted.

    Spikes: a pixel counts as a spike when it exceeds its 3x3
    median-filtered surrounding by ``spike_q99_factor`` times the 99th
    percentile of the frame's absolute residuals.  Anchoring the threshold
    to a high quantile of the residual distribution itself -- rather than a
    MAD-based sigma -- keeps the detector meaningful on real calibrated
    frames, whose residual texture is strongly non-Gaussian (quantization
    floors compress the MAD while banding and compression texture fatten
    the tails); an artifact spike (hot pixel near full well, cosmic ray)
    sits orders of magnitude above either.  Scene point sources brighter
    than the threshold are also caught, which is why the comparison runs
    the same detector on both cohorts.

    Parameters:
        image: 2-D frame in its native units.
        spike_q99_factor: Spike threshold as a multiple of the residual
            distribution's 99th absolute percentile.

    Returns:
        The per-frame measurements.
    """
    arr = np.asarray(image, dtype=np.float64)
    n_rows = arr.shape[0]
    if n_rows < 3 or arr.shape[1] < 3:
        return ArtifactIncidence(
            missing_line_fraction=float('nan'),
            spike_fraction=float('nan'),
            spike_positions_vu=np.empty((0, 2)),
        )
    # --- line loss ---
    lost_rows = 0
    frame_is_flat = float(np.ptp(arr)) == 0.0
    if not frame_is_flat:
        for v in range(1, n_rows - 1):
            if _row_is_lost(arr[v], arr[v - 1], arr[v + 1]):
                lost_rows += 1
    missing_line_fraction = lost_rows / max(1, n_rows - 2)
    # --- isolated positive spikes ---
    local_median = scipy.ndimage.median_filter(arr, size=3, mode='nearest')
    residual = arr - local_median
    finite = residual[np.isfinite(residual)]
    q99 = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 0.0
    if q99 <= 0.0:
        spikes = np.zeros(arr.shape, dtype=bool)
    else:
        spikes = residual > spike_q99_factor * q99
    positions = np.argwhere(spikes).astype(np.float64)
    spike_fraction = float(np.count_nonzero(spikes)) / arr.size
    return ArtifactIncidence(
        missing_line_fraction=missing_line_fraction,
        spike_fraction=spike_fraction,
        spike_positions_vu=positions,
    )


def split_stationary_spikes(
    per_frame: Sequence[ArtifactIncidence], *, min_recurrence: int = 2
) -> tuple[float, float]:
    """Split measured spikes into stationary (hot pixels) and transients.

    Parameters:
        per_frame: Incidence measurements from frames sharing a detector
            (same instrument, same frame shape).
        min_recurrence: Number of frames a position must recur in to count
            as stationary.

    Returns:
        ``(stationary_fraction, transient_fraction)``: mean per-frame
        fractions of pixels flagged as stationary and transient spikes.
        ``(nan, nan)`` when fewer than ``min_recurrence`` frames are given
        (a single frame cannot distinguish the two).
    """
    if len(per_frame) < min_recurrence:
        return float('nan'), float('nan')
    counts: dict[tuple[int, int], int] = {}
    for frame in per_frame:
        for v, u in frame.spike_positions_vu.astype(int):
            counts[(int(v), int(u))] = counts.get((int(v), int(u)), 0) + 1
    stationary = {pos for pos, n in counts.items() if n >= min_recurrence}
    stationary_fracs: list[float] = []
    transient_fracs: list[float] = []
    for frame in per_frame:
        total = frame.spike_positions_vu.shape[0]
        if total == 0:
            stationary_fracs.append(0.0)
            transient_fracs.append(0.0)
            continue
        n_stationary = sum(
            1 for v, u in frame.spike_positions_vu.astype(int) if (int(v), int(u)) in stationary
        )
        # Convert counts back to per-pixel fractions via the frame's own rate.
        scale = frame.spike_fraction / total if total else 0.0
        stationary_fracs.append(n_stationary * scale)
        transient_fracs.append((total - n_stationary) * scale)
    return float(np.mean(stationary_fracs)), float(np.mean(transient_fracs))
