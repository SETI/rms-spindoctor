"""Exposure-stratified dynamic-range statistics for realism FOM 5.

An unstratified cohort comparison measures what the spacecraft pointed at,
not the forward model: a cohort of long ring exposures saturates more than
a cohort of short satellite snaps regardless of detector fidelity.  Every
FOM 5 comparison therefore runs inside an exposure stratum, and a stratum
is compared only when both the real and the sim side populate it.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DynamicRangeStats',
    'frame_dynamic_range',
    'stratify_by_exposure',
]

# Percentiles reported for the signal distribution of each frame.
SIGNAL_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


@dataclass(frozen=True)
class DynamicRangeStats:
    """Per-frame dynamic-range statistics.

    Parameters:
        frac_saturated: Fraction of pixels at or above the saturation level.
        frac_near_floor: Fraction of pixels within one noise sigma of the
            frame's floor (1st percentile) -- the bias-hugging fraction.
        percentiles: Signal values at :data:`SIGNAL_PERCENTILES`.
    """

    frac_saturated: float
    frac_near_floor: float
    percentiles: tuple[float, ...]


def frame_dynamic_range(
    image: NDArrayFloatType, *, saturation_level: float, noise_sigma: float
) -> DynamicRangeStats:
    """Dynamic-range statistics of one frame in its native units.

    Parameters:
        image: 2-D frame.
        saturation_level: Full-scale value in the frame's units (for
            calibrated frames, the DN full scale propagated through the
            frame's calibration transform).
        noise_sigma: Per-pixel noise sigma in the frame's units (from the
            FOM 1 paired-difference estimator); defines the near-floor band.

    Returns:
        The per-frame statistics.
    """
    arr = np.asarray(image, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan_percentiles = tuple(float('nan') for _ in SIGNAL_PERCENTILES)
        return DynamicRangeStats(
            frac_saturated=float('nan'),
            frac_near_floor=float('nan'),
            percentiles=nan_percentiles,
        )
    floor = float(np.percentile(arr, 1.0))
    band = noise_sigma if np.isfinite(noise_sigma) and noise_sigma > 0.0 else 0.0
    frac_saturated = float(np.count_nonzero(arr >= saturation_level)) / arr.size
    frac_near_floor = float(np.count_nonzero(arr <= floor + band)) / arr.size
    percentiles = tuple(float(np.percentile(arr, p)) for p in SIGNAL_PERCENTILES)
    return DynamicRangeStats(
        frac_saturated=frac_saturated,
        frac_near_floor=frac_near_floor,
        percentiles=percentiles,
    )


def stratify_by_exposure(
    exposures_sec: Sequence[float | None],
    *,
    edges_sec: Sequence[float] = (0.05, 0.5, 5.0),
) -> dict[str, list[int]]:
    """Group frame indices into exposure strata.

    The default edges split at 50 ms, 500 ms, and 5 s, giving four strata
    that separate the short satellite snaps from the long ring and
    star-field exposures across the cohort instruments.  Frames with no
    recorded exposure land in the ``'unknown'`` stratum, which callers
    should compare only frame-by-frame (it mixes regimes).

    Parameters:
        exposures_sec: Per-frame exposure, or None where unrecorded.
        edges_sec: Ascending stratum boundaries in seconds.

    Returns:
        Mapping from stratum label to the indices of its frames.  Labels
        are ``'lt_<edge>'``, ``'<lo>_to_<hi>'``, ``'ge_<edge>'``, and
        ``'unknown'``; only populated strata appear.
    """
    boundaries = [float(e) for e in edges_sec]
    labels: list[str] = []
    labels.append(f'lt_{boundaries[0]:g}s')
    labels.extend(
        f'{boundaries[i]:g}s_to_{boundaries[i + 1]:g}s' for i in range(len(boundaries) - 1)
    )
    labels.append(f'ge_{boundaries[-1]:g}s')
    strata: dict[str, list[int]] = defaultdict(list)
    for idx, exposure in enumerate(exposures_sec):
        if exposure is None or not np.isfinite(exposure):
            strata['unknown'].append(idx)
            continue
        stratum = int(np.searchsorted(boundaries, float(exposure), side='right'))
        strata[labels[stratum]].append(idx)
    return dict(strata)
