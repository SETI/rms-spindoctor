"""Wasserstein-1 divergence and cohort-support labeling for the realism match.

The scalar divergence reported for every figure of merit is the Wasserstein-1
distance computed on quantile-clipped data (1st-99th percentile), normalized
by the real distribution's interquartile range.  W1 is a transport metric in
the variable's own units -- the actual reason to use it -- but it is not
outlier-robust (it grows linearly with displaced-tail distance), so the clip
keeps a noise statistic from silently measuring an artifact statistic's
tails.  No absolute pass/fail threshold is attached: the number is reported
per figure of merit and read by a human, and the regression check
(``tests/integration/realism_regression.py``) holds each supported statistic
to its committed value within a band measured from run-to-run variation.

Each sample is winsorized at its *own* 1st/99th percentiles.  Clipping the
sim sample at the real sample's bounds would cap the measured displacement of
a grossly wrong sim distribution, hiding exactly the mismatch the statistic
exists to surface; clipping each sample at its own tails removes only that
sample's outliers while preserving bulk displacement.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'CohortSupport',
    'W1Result',
    'cohort_support',
    'w1_between_densities',
    'w1_divergence',
]

# Quantile-clip bounds fixed by the acceptance spec (1st-99th percentile).
_CLIP_LO_Q = 0.01
_CLIP_HI_Q = 0.99

# Minimum samples for a meaningful empirical W1: below this the quantile
# clip and the IQR are dominated by individual draws.
_MIN_SAMPLES = 8


@dataclass(frozen=True)
class W1Result:
    """The scalar divergence between one real and one sim sample.

    Parameters:
        w1: Wasserstein-1 distance between the winsorized samples, in the
            variable's own units.
        w1_normalized: ``w1`` divided by the real distribution's IQR; NaN
            when either sample is too small or the real IQR is zero
            (a degenerate real distribution is not a usable yardstick).
        real_iqr: Interquartile range of the raw (unclipped) real sample.
        n_real: Number of finite real samples.
        n_sim: Number of finite sim samples.
    """

    w1: float
    w1_normalized: float
    real_iqr: float
    n_real: int
    n_sim: int

    @property
    def usable(self) -> bool:
        """True when the normalized divergence is a real number."""
        return bool(np.isfinite(self.w1_normalized))


class CohortSupport(Enum):
    """How well a cohort supports a distributional statistic.

    The labels: ``SUPPORTED`` means enough frames for a per-frame
    distribution statement; ``LIMITED`` means a comparison is reported but
    flagged as resting on too few frames for distributional confidence; and
    ``UNSUPPORTED`` means the comparison rests on too few frames to say
    anything about the distributions and is not held to a regression band, so
    the instrument's sim accuracy is bounded by unverified forward-model
    fidelity for this figure of merit.  The frames
    are counted on both sides, real and simulated, and the smaller count sets
    the label.
    """

    SUPPORTED = 'supported'
    LIMITED = 'limited'
    UNSUPPORTED = 'unsupported'


def cohort_support(n_frames: int, *, supported_min: int = 8, limited_min: int = 2) -> CohortSupport:
    """Label the support a cohort of ``n_frames`` gives a per-frame statistic.

    An IQR of two frames is not a statistic: below ``limited_min`` frames the
    figure of merit is unsupported and must be labeled as such rather than
    reported as a distribution.  Between ``limited_min`` and ``supported_min``
    the comparison is reported with an explicit low-count caveat.

    Parameters:
        n_frames: Number of cohort frames contributing to the statistic, on
            whichever side, real or simulated, has fewer.
        supported_min: Frame count at or above which the statistic is
            fully supported.
        limited_min: Frame count at or above which a caveated comparison
            is reported at all.

    Returns:
        The support label.
    """
    if n_frames >= supported_min:
        return CohortSupport.SUPPORTED
    if n_frames >= limited_min:
        return CohortSupport.LIMITED
    return CohortSupport.UNSUPPORTED


def _finite(sample: NDArrayFloatType) -> NDArrayFloatType:
    """Return the finite entries of ``sample`` as a flat float64 array."""
    arr = np.asarray(sample, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def w1_divergence(real: NDArrayFloatType, sim: NDArrayFloatType) -> W1Result:
    """The winsorized, IQR-normalized Wasserstein-1 divergence of two samples.

    Both samples are winsorized at their own 1st/99th percentiles, the
    Wasserstein-1 distance is computed between the winsorized samples, and
    the result is normalized by the raw real sample's IQR.

    Parameters:
        real: Sample drawn from the real cohort (any shape; flattened).
        sim: Sample drawn from the simulated frames (any shape; flattened).

    Returns:
        The :class:`W1Result`; ``w1_normalized`` is NaN when either sample
        has fewer than 8 finite entries or the real IQR is zero.
    """
    real_f = _finite(real)
    sim_f = _finite(sim)
    n_real = int(real_f.size)
    n_sim = int(sim_f.size)
    if n_real < _MIN_SAMPLES or n_sim < _MIN_SAMPLES:
        return W1Result(
            w1=float('nan'),
            w1_normalized=float('nan'),
            real_iqr=float('nan'),
            n_real=n_real,
            n_sim=n_sim,
        )
    q25, q75 = np.quantile(real_f, [0.25, 0.75])
    real_iqr = float(q75 - q25)
    real_clipped = np.clip(real_f, *np.quantile(real_f, [_CLIP_LO_Q, _CLIP_HI_Q]))
    sim_clipped = np.clip(sim_f, *np.quantile(sim_f, [_CLIP_LO_Q, _CLIP_HI_Q]))
    w1 = float(scipy.stats.wasserstein_distance(real_clipped, sim_clipped))
    normalized = w1 / real_iqr if real_iqr > 0.0 else float('nan')
    return W1Result(
        w1=w1,
        w1_normalized=normalized,
        real_iqr=real_iqr,
        n_real=n_real,
        n_sim=n_sim,
    )


def _weighted_quantile(
    values: NDArrayFloatType, weights: NDArrayFloatType, q: NDArrayFloatType
) -> NDArrayFloatType:
    """Quantiles of a discrete weighted distribution.

    Parameters:
        values: Support points, ascending.
        weights: Non-negative weights summing to a positive total.
        q: Quantile levels in [0, 1].

    Returns:
        The quantile values, by inverse-CDF lookup (step CDF, right limit).
    """
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, np.asarray(q, dtype=np.float64), side='left')
    idx = np.clip(idx, 0, values.size - 1)
    return np.asarray(values[idx], dtype=np.float64)


def w1_between_densities(
    x: NDArrayFloatType, real_density: NDArrayFloatType, sim_density: NDArrayFloatType
) -> W1Result:
    """W1 between two normalized densities sharing one support axis.

    Used for profile-like figures of merit (power spectra, radial brightness
    profiles) where the comparison is between curve *shapes*: each curve is
    treated as a probability density over its axis, and W1 measures how far
    mass must move along the axis to turn one shape into the other.  The
    result is in the axis's units, normalized by the real density's IQR
    along the axis.  Quantile clipping is not applied: the support axis is a
    fixed finite grid, so there are no sample outliers to clip.

    Parameters:
        x: Support axis (ascending, e.g. spatial frequency or radius).
        real_density: Non-negative curve values for the real cohort.
        sim_density: Non-negative curve values for the sim frames.

    Returns:
        The :class:`W1Result`; ``n_real``/``n_sim`` record the number of
        positive-mass support points of each curve.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    real_arr = np.asarray(real_density, dtype=np.float64)
    sim_arr = np.asarray(sim_density, dtype=np.float64)
    real_mass = float(real_arr.sum())
    sim_mass = float(sim_arr.sum())
    n_real = int(np.count_nonzero(real_arr > 0.0))
    n_sim = int(np.count_nonzero(sim_arr > 0.0))
    if x_arr.size < 2 or real_mass <= 0.0 or sim_mass <= 0.0:
        return W1Result(
            w1=float('nan'),
            w1_normalized=float('nan'),
            real_iqr=float('nan'),
            n_real=n_real,
            n_sim=n_sim,
        )
    w1 = float(scipy.stats.wasserstein_distance(x_arr, x_arr, real_arr, sim_arr))
    q25, q75 = _weighted_quantile(x_arr, real_arr, np.array([0.25, 0.75]))
    real_iqr = float(q75 - q25)
    normalized = w1 / real_iqr if real_iqr > 0.0 else float('nan')
    return W1Result(
        w1=w1,
        w1_normalized=normalized,
        real_iqr=real_iqr,
        n_real=n_real,
        n_sim=n_sim,
    )
