"""Per-pixel Poisson sampling for the detector's shot noise.

The detector draws one uniform variate per pixel and turns each into a Poisson count
through the inverse of that pixel's cumulative distribution.  The number of random
values drawn depends only on the frame's dimensions, and the uniform that lands on a
pixel depends only on its position, so a change to one pixel's mean can change only
that pixel's count.  A frame-wide Poisson sampler does not have this property: it
draws a different number of values for a zero mean, and switches algorithm at a mean
of 10, so a last-bit change in one pixel can shift the noise of every pixel after it.

The inverse is exact up to a mean of :data:`EXACT_MEAN_LIMIT` electrons: the count is
the smallest ``k`` whose cumulative probability reaches the uniform, found by
correcting a close first guess against SciPy's Poisson distribution function.  Above
the limit the count is the Cornish-Fisher quantile, rounded to the nearest integer.
At a mean of one million a sample differs from the exact inverse with probability
about 3e-5, and then by one count against a standard deviation of a thousand; the
disagreement falls as one over the square root of the mean.
"""

import numpy as np
from scipy import special

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['EXACT_MEAN_LIMIT', 'poisson_inverse_cdf', 'sample_poisson']

# The largest mean, in counts, whose inverse distribution function is computed
# exactly.  SciPy's Poisson distribution function loses accuracy in the far tails
# near a mean of 1e7, so the exact search can take thousands of steps there; the
# Cornish-Fisher quantile is as good as the exact answer long before that.
EXACT_MEAN_LIMIT = 1.0e6

# The smallest positive double, so the normal quantile of a zero uniform is finite.
_SMALLEST_UNIFORM = float(np.finfo(np.float64).tiny)

# A uniform above this takes its comparison from the upper tail, where the tail
# probability and one minus the uniform are both exact to full precision.
_UPPER_TAIL_UNIFORM = 0.5


def sample_poisson(rng: np.random.Generator, mean: NDArrayFloatType) -> NDArrayFloatType:
    """Draw one Poisson count per pixel of ``mean``, from one uniform per pixel.

    The generator supplies ``rng.random(mean.shape)``, so the values drawn from it
    depend only on the array's dimensions, never on its contents.  Each count is
    :func:`poisson_inverse_cdf` of the pixel's uniform and mean.

    Parameters:
        rng: The generator the uniforms are drawn from.
        mean: The Poisson mean of each pixel.  A mean that is not positive gives 0.

    Returns:
        The counts as float64, the same dimensions as ``mean``.
    """
    uniform = rng.random(mean.shape)
    return poisson_inverse_cdf(uniform, mean)


def poisson_inverse_cdf(uniform: NDArrayFloatType, mean: NDArrayFloatType) -> NDArrayFloatType:
    """The Poisson count at each uniform, for each pixel's own mean.

    For a mean up to :data:`EXACT_MEAN_LIMIT` the count is the smallest ``k`` with
    ``P(X <= k) >= u``, the exact inverse distribution function at ``u``.  For a
    larger mean it is the rounded Cornish-Fisher quantile, which agrees with the
    exact answer for all but a few pixels in a hundred thousand and then differs by
    one count.  A mean that is not positive (zero, negative, or NaN) gives 0, as
    does a uniform of exactly 0 for any mean up to the limit.  Each output pixel
    depends only on the uniform and the mean at that pixel.

    Parameters:
        uniform: Uniform variates in [0, 1).
        mean: The Poisson mean of each pixel; the same dimensions as ``uniform``.

    Returns:
        The counts as float64, the same dimensions as ``mean``.
    """
    uniform = np.asarray(uniform, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    counts = np.zeros(mean.shape, dtype=np.float64)
    positive = mean > 0.0
    lam = mean[positive]
    u = uniform[positive]
    guess = _cornish_fisher_quantile(u, lam)
    exact = lam <= EXACT_MEAN_LIMIT
    guess[exact] = _exact_quantile(guess[exact], u[exact], lam[exact])
    counts[positive] = guess
    return counts


def _cornish_fisher_quantile(u: NDArrayFloatType, lam: NDArrayFloatType) -> NDArrayFloatType:
    """The Poisson quantile from the Cornish-Fisher expansion, rounded and kept >= 0.

    Every cumulant of a Poisson distribution equals its mean, so the expansion of
    the quantile in the normal quantile ``z`` is ``lam + sqrt(lam) z + (z**2 -
    1) / 6 + (z - z**3) / (72 sqrt(lam))``.  Rounding to the nearest integer applies
    the continuity correction.
    """
    z = special.ndtri(np.maximum(u, _SMALLEST_UNIFORM))
    root = np.sqrt(lam)
    quantile = lam + root * z + (z * z - 1.0) / 6.0 + (z - z**3) / (72.0 * root)
    result: NDArrayFloatType = np.maximum(np.rint(quantile), 0.0)
    return result


def _cdf_reaches(
    k: NDArrayFloatType, u: NDArrayFloatType, lam: NDArrayFloatType
) -> NDArrayBoolType:
    """Whether ``P(X <= k) >= u`` for each element, comparing in the precise tail.

    Below one half the lower tail is compared with ``u``; above it the upper tail
    ``P(X > k)`` is compared with ``1 - u``, which is exact there, so a uniform near
    1 is not lost to the rounding of a probability near 1.
    """
    result = np.empty(k.shape, dtype=np.bool_)
    upper = u > _UPPER_TAIL_UNIFORM
    lower = ~upper
    result[lower] = special.pdtr(k[lower], lam[lower]) >= u[lower]
    result[upper] = special.pdtrc(k[upper], lam[upper]) <= 1.0 - u[upper]
    return result


def _exact_quantile(
    guess: NDArrayFloatType, u: NDArrayFloatType, lam: NDArrayFloatType
) -> NDArrayFloatType:
    """Correct a first guess to the smallest ``k`` with ``P(X <= k) >= u``.

    A guess below the answer steps up until the distribution function reaches the
    uniform; one above it steps down while the count below still reaches it.  Each
    step evaluates only the elements still moving, and a Cornish-Fisher guess is
    within a step or two of the answer for all but the farthest tails.
    """
    k = guess.copy()
    moving = np.flatnonzero(~_cdf_reaches(k, u, lam))
    while moving.size > 0:
        k[moving] += 1.0
        moving = moving[~_cdf_reaches(k[moving], u[moving], lam[moving])]
    moving = np.flatnonzero(k > 0.0)
    moving = moving[_cdf_reaches(k[moving] - 1.0, u[moving], lam[moving])]
    while moving.size > 0:
        k[moving] -= 1.0
        moving = moving[k[moving] > 0.0]
        moving = moving[_cdf_reaches(k[moving] - 1.0, u[moving], lam[moving])]
    return k
