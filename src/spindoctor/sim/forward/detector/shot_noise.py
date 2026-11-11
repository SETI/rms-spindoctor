"""Per-pixel Poisson sampling for the detector's shot noise.

The detector draws one uniform variate per pixel and turns each into a Poisson count
through the inverse of that pixel's cumulative distribution.  The number of random
values drawn depends only on the frame's dimensions, and the uniform that lands on a
pixel depends only on its position, so a change to one pixel's mean can change only
that pixel's count.  A frame-wide Poisson sampler does not have this property: it
draws a different number of values for a zero mean, and switches algorithm at a mean
of 10, so a last-bit change in one pixel can shift the noise of every pixel after it.

How a pixel's count is found depends on its mean:

* Below :data:`SUM_MEAN_LIMIT` electrons the count is exact, found by adding up the
  probabilities of 0, 1, 2, ... counts until the running total reaches the uniform.
  This is the cheapest exact method for a dark pixel.  A pixel whose total lands
  within rounding distance of its uniform is settled again with SciPy's distribution
  function, so rounding in the sum never changes a count.
* From there up to :data:`EXACT_MEAN_LIMIT` electrons the count is also exact: the
  search starts from a close first guess, steps outward in doubling strides until it
  has the answer bracketed, and then halves the bracket, so the number of
  distribution function evaluations grows only with the logarithm of the guess's
  error.  The guess is held within ten standard deviations of the mean, plus 40
  counts, so every pixel is settled in at most about 40 evaluations, and in two or
  fewer for nearly every pixel.
* Above :data:`EXACT_MEAN_LIMIT` the count is the Cornish-Fisher quantile, rounded to
  the nearest integer.  It differs from the exact inverse on about 0.3% of pixels at a
  mean of 100, 0.1% at 1,000, and 0.03% at 10,000, and then always by one count,
  against a standard deviation of 10 to 100 counts.  Its sample mean and variance are
  indistinguishable from the exact inverse's.  It costs a tenth of the exact search.

The uniforms for the whole frame are drawn at once, and the frame is then converted
in passes of a fixed number of pixels, so the working memory beyond the input and
output arrays does not grow with the frame, and the counts do not depend on the pass
size.  Measured on one core of an i9-13900K, a 1024 by 1024 frame whose sky lies
between 0 and 20 electrons and whose body covers 30% of it at 1,000 to 40,000
electrons takes about 0.08 s, where NumPy's frame-wide Poisson sampler takes 0.03 s.
A 2048 by 2048 frame of means spread evenly from 0 to 1,000 takes about 0.36 s,
against 0.09 s.
"""

import numpy as np
from scipy import special

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['EXACT_MEAN_LIMIT', 'poisson_inverse_cdf', 'sample_poisson']

# The largest mean, in counts, whose inverse distribution function is computed
# exactly.  Above it the Cornish-Fisher quantile differs from the exact count on
# under 0.3% of pixels, by one count against a standard deviation of at least ten,
# and costs a tenth as much.
EXACT_MEAN_LIMIT = 100.0

# Means below this, in counts, are inverted exactly by a running sum of the
# probabilities.  The sum takes about mean + 1 steps per pixel, each a few cheap
# array operations, so it wins over the distribution-function search only for a
# small mean.
SUM_MEAN_LIMIT = 30.0

# The running sum stops after this many steps.  A count above it has probability
# below 1e-38 at any mean below SUM_MEAN_LIMIT, so only a uniform within rounding
# distance of 1 can reach it, and such a pixel is settled by the search instead.
_SUM_MAX_STEPS = 125

# A running total within this distance of its uniform is too close to trust: the
# sum of up to _SUM_MAX_STEPS terms each no larger than 1 carries a rounding error
# well below it.  Such a pixel is settled by the search, which compares in the
# precise tail.
_SUM_TRUST_MARGIN = 1.0e-12

# The smallest positive double, so the normal quantile of a tiny uniform is finite.
_SMALLEST_UNIFORM = float(np.finfo(np.float64).tiny)

# A uniform above this takes its comparison from the upper tail, where the tail
# probability and one minus the uniform are both exact to full precision.
_UPPER_TAIL_UNIFORM = 0.5

# The smallest mean whose first guess includes the expansion's cubic term.  The term
# grows as one over the square root of the mean, so for a small mean it is
# meaningless: at a mean of 1e-300 it alone would put the guess near 1e147.
_CUBIC_TERM_MEAN = 10.0

# The first guess for the exact search is held within this many standard deviations
# of the mean, plus _GUESS_MARGIN_COUNTS.  Every uniform the generator can return
# other than 0 has its answer inside that range, so the clamp only shortens the
# search from a guess that is far off.
_GUESS_SIGMAS = 10.0
_GUESS_MARGIN_COUNTS = 40.0

# Pixels converted in one pass.  The temporaries of a pass take under 150 bytes per
# pixel, so this bounds them to about 40 MB for any frame.
_CHUNK_PIXELS = 1 << 18


def sample_poisson(rng: np.random.Generator, mean: NDArrayFloatType) -> NDArrayFloatType:
    """Draw one Poisson count per pixel of ``mean``, from one uniform per pixel.

    The generator supplies ``rng.random(mean.shape)``, so the values drawn from it
    depend only on the array's dimensions, never on its contents.  Each count is
    :func:`poisson_inverse_cdf` of the pixel's uniform and mean.

    Parameters:
        rng: The generator the uniforms are drawn from.
        mean: The Poisson mean of each pixel.  A zero or negative mean gives 0.

    Returns:
        The counts as float64, the same dimensions as ``mean``.

    Raises:
        ValueError: If any mean is NaN or infinite.
    """
    uniform = rng.random(mean.shape)
    return poisson_inverse_cdf(uniform, mean)


def poisson_inverse_cdf(uniform: NDArrayFloatType, mean: NDArrayFloatType) -> NDArrayFloatType:
    """The Poisson count at each uniform, for each pixel's own mean.

    For a mean up to :data:`EXACT_MEAN_LIMIT` the count is the smallest ``k`` with
    ``P(X <= k) >= u``, the exact inverse distribution function at ``u``.  For a
    larger mean it is the rounded Cornish-Fisher quantile, which differs from the
    exact answer on under 0.3% of pixels, and then by one count.  A zero or negative
    mean gives 0, and so does a uniform of exactly 0, which is the exact inverse there
    for every mean.  Each output pixel depends only on the uniform and the mean at that
    pixel.

    Parameters:
        uniform: Uniform variates in [0, 1).
        mean: The Poisson mean of each pixel; the same dimensions as ``uniform``.

    Returns:
        The counts as float64, the same dimensions as ``mean``.

    Raises:
        ValueError: If the uniforms and the means differ in dimensions, if any
            uniform is NaN or outside [0, 1), or if any mean is NaN or infinite.  A
            uniform of 1 or more has no count, and the search for one would never
            end.  A mean that is not finite has no Poisson distribution, and it
            means an earlier stage went wrong.
    """
    uniform = np.asarray(uniform, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if uniform.shape != mean.shape:
        raise ValueError(
            f'Uniforms and means must have the same dimensions, but the uniforms are '
            f'{uniform.shape} and the means {mean.shape}'
        )
    out_of_range = int(np.count_nonzero(~((uniform >= 0.0) & (uniform < 1.0))))
    if out_of_range:
        raise ValueError(
            f'Uniforms must lie in [0, 1), but {out_of_range} of {uniform.size} are '
            f'NaN or outside it'
        )
    not_finite = int(np.count_nonzero(~np.isfinite(mean)))
    if not_finite:
        raise ValueError(
            f'Poisson means must be finite, but {not_finite} of {mean.size} are NaN or infinite'
        )
    counts = np.zeros(mean.shape, dtype=np.float64)
    flat_uniform = uniform.reshape(-1)
    flat_mean = mean.reshape(-1)
    flat_counts = counts.reshape(-1)
    for start in range(0, flat_mean.size, _CHUNK_PIXELS):
        stop = start + _CHUNK_PIXELS
        flat_counts[start:stop] = _chunk_counts(flat_uniform[start:stop], flat_mean[start:stop])
    return counts


def _chunk_counts(uniform: NDArrayFloatType, mean: NDArrayFloatType) -> NDArrayFloatType:
    """The counts for one pass of pixels, as :func:`poisson_inverse_cdf` defines them."""
    counts = np.zeros(mean.shape, dtype=np.float64)
    live = (mean > 0.0) & (uniform > 0.0)
    small = np.flatnonzero(live & (mean < SUM_MEAN_LIMIT))
    middle = np.flatnonzero(live & (mean >= SUM_MEAN_LIMIT) & (mean <= EXACT_MEAN_LIMIT))
    large = np.flatnonzero(live & (mean > EXACT_MEAN_LIMIT))
    if small.size > 0:
        counts[small] = _running_sum_quantile(uniform[small], mean[small])
    if middle.size > 0:
        counts[middle] = _searched_quantile(uniform[middle], mean[middle])
    if large.size > 0:
        counts[large] = _cornish_fisher_quantile(uniform[large], mean[large])
    return counts


def _searched_quantile(u: NDArrayFloatType, lam: NDArrayFloatType) -> NDArrayFloatType:
    """The exact count by a search from the Cornish-Fisher guess; ``u`` and ``lam`` > 0."""
    spread = _GUESS_SIGMAS * np.sqrt(lam) + _GUESS_MARGIN_COUNTS
    lowest = np.maximum(np.floor(lam - spread), 0.0)
    first = np.clip(_cornish_fisher_quantile(u, lam), lowest, np.ceil(lam + spread))
    return _exact_quantile(first, u, lam)


def _running_sum_quantile(u: NDArrayFloatType, lam: NDArrayFloatType) -> NDArrayFloatType:
    """The exact count by adding up the probabilities of 0, 1, 2, ... counts.

    ``u`` and ``lam`` must be positive and ``lam`` below :data:`SUM_MEAN_LIMIT`.  Each
    step multiplies the last probability by ``lam / k`` and adds it to the running
    total, until the total reaches the uniform or :data:`_SUM_MAX_STEPS` steps have
    run.  A pixel whose total ends below its uniform, or within
    :data:`_SUM_TRUST_MARGIN` of it on either side, is settled again by
    :func:`_searched_quantile`, so the rounding in the sum never decides a count.
    """
    k = np.zeros(lam.shape, dtype=np.float64)
    term = np.exp(-lam)
    total = term.copy()
    moving = np.flatnonzero(total < u)
    step = 0
    while moving.size > 0 and step < _SUM_MAX_STEPS:
        step += 1
        term[moving] *= lam[moving] / step
        total[moving] += term[moving]
        k[moving] = step
        moving = moving[total[moving] < u[moving]]
    below = total - term
    # A pixel the loop left short of its uniform at the step limit has a total below
    # the uniform, so the first test catches it too.
    doubtful = (total - u < _SUM_TRUST_MARGIN) | (u - below < _SUM_TRUST_MARGIN)
    redo = np.flatnonzero(doubtful)
    if redo.size > 0:
        k[redo] = _searched_quantile(u[redo], lam[redo])
    return k


def _cornish_fisher_quantile(u: NDArrayFloatType, lam: NDArrayFloatType) -> NDArrayFloatType:
    """The Poisson quantile from the Cornish-Fisher expansion, rounded and kept >= 0.

    Every cumulant of a Poisson distribution equals its mean, so the expansion of
    the quantile in the normal quantile ``z`` is ``lam + sqrt(lam) z + (z**2 -
    1) / 6 + (z - z**3) / (72 sqrt(lam))``.  The last term is left out below a mean
    of :data:`_CUBIC_TERM_MEAN`, where it does more harm than good.  Rounding to the
    nearest integer applies the continuity correction.
    """
    z = special.ndtri(np.maximum(u, _SMALLEST_UNIFORM))
    root = np.sqrt(lam)
    quantile = lam + root * z + (z * z - 1.0) / 6.0
    cubic = lam >= _CUBIC_TERM_MEAN
    z_cubic = z[cubic]
    quantile[cubic] += (z_cubic - z_cubic**3) / (72.0 * root[cubic])
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

    Every uniform must be positive.  The search keeps, for each element, a count
    ``hi`` whose distribution function reaches the uniform and a count ``lo`` whose
    distribution function falls short of it, with -1 standing for "below every
    count".  A guess that reaches steps down in strides of 1, 2, 4, and so on until a
    probe falls short or would go below 0; one that falls short steps up the same way
    until a probe reaches.  Halving the bracket then leaves ``hi`` one above ``lo``,
    and ``hi`` is the answer.  A guess off by ``d`` counts costs about ``2 log2(d)``
    evaluations, and an exact guess costs two.
    """
    hi = guess.copy()
    lo = guess.copy()
    reaches = _cdf_reaches(guess, u, lam)

    moving = np.flatnonzero(reaches)
    stride = 1.0
    while moving.size > 0:
        probe = hi[moving] - stride
        below_zero = probe < 0.0
        lo[moving[below_zero]] = -1.0
        moving = moving[~below_zero]
        probe = probe[~below_zero]
        still = _cdf_reaches(probe, u[moving], lam[moving])
        hi[moving[still]] = probe[still]
        lo[moving[~still]] = probe[~still]
        moving = moving[still]
        stride *= 2.0

    moving = np.flatnonzero(~reaches)
    stride = 1.0
    while moving.size > 0:
        probe = lo[moving] + stride
        found = _cdf_reaches(probe, u[moving], lam[moving])
        hi[moving[found]] = probe[found]
        lo[moving[~found]] = probe[~found]
        moving = moving[~found]
        stride *= 2.0

    moving = np.flatnonzero(hi - lo > 1.0)
    while moving.size > 0:
        middle = np.floor(0.5 * (lo[moving] + hi[moving]))
        found = _cdf_reaches(middle, u[moving], lam[moving])
        hi[moving[found]] = middle[found]
        lo[moving[~found]] = middle[~found]
        moving = moving[hi[moving] - lo[moving] > 1.0]
    return hi
