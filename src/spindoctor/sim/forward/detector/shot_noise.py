"""Per-pixel Poisson sampling for the detector's shot noise.

The detector draws one uniform variate per pixel and turns each into a Poisson count
through the inverse of that pixel's cumulative distribution.  The number of random
values drawn depends only on the frame's dimensions, and the uniform that lands on a
pixel depends only on its position, so a change to one pixel's mean can change only
that pixel's count.  A frame-wide Poisson sampler does not have this property: it
draws a different number of values for a zero mean, and switches algorithm at a mean
of 10, so a last-bit change in one pixel can shift the noise of every pixel after it.

The inverse is exact up to a mean of :data:`EXACT_MEAN_LIMIT` electrons: the count is
the smallest ``k`` whose cumulative probability reaches the uniform.  The search
starts from a close first guess, steps outward in doubling strides until it has the
answer bracketed, and then halves the bracket, so the number of distribution function
evaluations grows only with the logarithm of the guess's error.  The guess is held
within ten standard deviations of the mean, plus 40 counts, so every pixel is settled
in at most about 40 evaluations whatever its mean and uniform, and in two or fewer
for nearly every pixel.  Above the limit the count is the Cornish-Fisher quantile,
rounded to the nearest integer.  At a mean of one million a sample differs from the
exact inverse with probability about 3e-5, and then by one count against a standard
deviation of a thousand; the disagreement falls as one over the square root of the
mean.

The uniforms for the whole frame are drawn at once, and the frame is then converted
in passes of a fixed number of pixels, so the working memory beyond the input and
output arrays does not grow with the frame, and the counts do not depend on the pass
size.  Measured on a 2048 by 2048 frame of means spread evenly from 0 to 1000, the
sampler takes about 1.4 s, where NumPy's frame-wide Poisson sampler takes 0.12 s,
and the peak memory traced during the call is 105 MB, of which 67 MB is the uniforms
and the returned counts.  A million pixels at a single mean take 0.09 s for a mean
near zero and up to 0.3 s for a mean in the hundreds or above.
"""

import numpy as np
from scipy import special

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['EXACT_MEAN_LIMIT', 'poisson_inverse_cdf', 'sample_poisson']

# The largest mean, in counts, whose inverse distribution function is computed
# exactly.  SciPy's Poisson distribution function loses accuracy in the far tails
# near a mean of 1e7; the Cornish-Fisher quantile is as good as the exact answer long
# before that.
EXACT_MEAN_LIMIT = 1.0e6

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
    """
    uniform = rng.random(mean.shape)
    return poisson_inverse_cdf(uniform, mean)


def poisson_inverse_cdf(uniform: NDArrayFloatType, mean: NDArrayFloatType) -> NDArrayFloatType:
    """The Poisson count at each uniform, for each pixel's own mean.

    For a mean up to :data:`EXACT_MEAN_LIMIT` the count is the smallest ``k`` with
    ``P(X <= k) >= u``, the exact inverse distribution function at ``u``.  For a
    larger mean it is the rounded Cornish-Fisher quantile, which agrees with the
    exact answer for all but a few pixels in a hundred thousand and then differs by
    one count.  A zero or negative mean gives 0, and so does a uniform of exactly 0,
    which is the exact inverse there for every mean.  Each output pixel depends only
    on the uniform and the mean at that pixel.

    Parameters:
        uniform: Uniform variates in [0, 1).
        mean: The Poisson mean of each pixel; the same dimensions as ``uniform``.

    Returns:
        The counts as float64, the same dimensions as ``mean``.
    """
    uniform = np.asarray(uniform, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
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
    active = np.flatnonzero((mean > 0.0) & (uniform > 0.0))
    lam = mean[active]
    u = uniform[active]
    guess = _cornish_fisher_quantile(u, lam)
    exact = lam <= EXACT_MEAN_LIMIT
    lam_exact = lam[exact]
    spread = _GUESS_SIGMAS * np.sqrt(lam_exact) + _GUESS_MARGIN_COUNTS
    lowest = np.maximum(np.floor(lam_exact - spread), 0.0)
    first = np.clip(guess[exact], lowest, np.ceil(lam_exact + spread))
    guess[exact] = _exact_quantile(first, u[exact], lam_exact)
    counts[active] = guess
    return counts


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
