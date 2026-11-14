"""Sky-region noise statistics for realism FOM 1.

Science frames have no flat-field pairs, so noise is estimated inside near-uniform
patches.  :func:`patch_noise_sigma` measures one patch.  How it does so depends on
whether the patch is quantized, that is, whether its values sit on a lattice of equally
spaced levels such as whole DN (:func:`quantization_step`).

A patch of continuous values is measured from paired horizontal pixel differences
(:func:`paired_difference_sigma`).  The differences cancel scene structure that varies
slowly across the patch.  Their median absolute deviation (MAD) gives the sigma, and
the median keeps stars, cosmic rays, and hot pixels from inflating it.  Binning pixels
by signal instead would count scene texture as noise.

A quantized patch is measured from its paired differences too, read from their
histogram, when they give at least two steps and the patch is not clipped.  Otherwise
it is measured by fitting the quantization (:func:`lattice_noise_sigma`): a quadratic
surface plus Gaussian noise, rounded to the lattice, fitted by maximum likelihood.  A
patch the fit can't trust gives NaN.

The sky spatial power spectrum (radially averaged over annuli of spatial
frequency) catches banding and coherent noise that a scalar sigma cannot:
white read noise is flat, banding is a comb.
"""

from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'SkyPatch',
    'dark_sky_patches',
    'find_uniform_patches',
    'lattice_noise_sigma',
    'noisy_patches',
    'paired_difference_sigma',
    'patch_noise_sigma',
    'quantization_step',
    'radial_power_spectrum',
]

# MAD to sigma conversion for a normal distribution.
_MAD_TO_SIGMA = 1.4826

# A level lies on the fitted lattice when it is within this fraction of a step of a
# lattice level, or within _FLOAT_ROUNDING_EPS of its own magnitude, whichever is larger.
# A level outside the fitted ones is also allowed the fit's error there.
_LATTICE_TOLERANCE = 1e-3

# The rounding error a float32 value can carry, relative to its magnitude, is taken as a
# few units of float32's machine epsilon.  Whole DN scaled into float32, such as DN times
# 1.234e-3 near DN 3000, sit off their lattice levels by about this much.  One rounding
# to float32 moves a value by at most half of _FLOAT32_EPS of its magnitude.
_FLOAT32_EPS = float(np.finfo(np.float32).eps)
_FLOAT_ROUNDING_EPS = 4.0 * _FLOAT32_EPS

# At a level outside the fitted ones, the fit's error is taken as this many standard
# errors of the fit carried out to that level.
_FIT_SIGMAS = 4.0

# A step at or below this fraction of a value's magnitude is the spacing of
# floating-point numbers there, not a quantization.  For example, float32 values near
# 3e-3 I/F are 2.3e-10 apart.
_FLOAT_SPACING_FRACTION = 1e-6

# A lattice counts only when every value lies within this many steps of the lowest
# central level.  The noise estimators count each pixel's whole-step index up from the
# lowest value.  Below 2**32, float64 holds each index and each half-step edge exactly,
# and the quantized fit's surface near the sky carries a rounding error under a
# millionth of a step.  A pixel farther out, such as a sentinel value, would break that
# arithmetic.  The bound keeps the arithmetic exact, but it does not make the fit
# converge: a pixel a billion steps from the sky can still make the fit read NaN.
_MAX_LEVEL_STEPS = 2.0**31

# When the central values sit on fewer than two levels, the step is fitted to them and to
# the nearest level on each side.  A side's level is left out when it lies more than this
# many times as far from the central values as the other side's.
_NEIGHBOR_GAPS = 2.0

# The quantized fit conditions each pixel on a window of levels around the fitted
# surface.  The window reaches at least this many fitted sigmas beyond the surface on
# each side, and at least one level past the surface's nearest level.  A pixel outside
# its window counts only as lying beyond the window's edge.  The window is rebuilt from
# each fit until it repeats one already fitted, for at most _MAX_WINDOW_FITS fits.
_WINDOW_SIGMAS = 3.0
_MAX_WINDOW_FITS = 10

# Gaussian noise puts this fraction of its pixels more than three sigmas from the mean,
# counting both sides.  A pixel is censored when it lies beyond its window or is cut off
# from the bulk of the levels by empty levels.  The quantized fit gives NaN when the
# censored pixels exceed that fraction by more than _EXCESS_FRACTION of the patch plus
# _EXCESS_PIXELS pixels.  An end level of the patch counts as clipped when it belongs to
# the bulk and holds more than _EXCESS_FRACTION of the patch plus _EXCESS_PIXELS pixels.
_GAUSSIAN_BEYOND_FRACTION = 0.0027
_EXCESS_FRACTION = 0.01
_EXCESS_PIXELS = 3

# After the fit, a quantized patch gives NaN when the lowest or highest level of its bulk
# holds too many pixels: more than the fitted model predicts, plus _LEVEL_SIGMAS times
# the square root of the prediction, plus _GAUSSIAN_BEYOND_FRACTION of the patch, plus
# _EXCESS_PIXELS.  On 12,000 patches of rounded Gaussian noise from 0.3 to 10 steps,
# flat, tilted, or curved, and clipped or not, no level came within 3.7 pixels of that
# limit.  The levels between are not checked.  In the darkest sky of real Galileo
# frames, the fullest level holds about 1.4 times the predicted pixels, and its
# neighbors hold fewer.
_LEVEL_SIGMAS = 4.0

# A run of occupied levels may span a stretch of empty levels up to this many
# paired-difference sigmas wide, and a wider stretch ends it.  When that sigma is below
# 1/_GAP_SIGMAS of a step, any empty level ends a run.  Noise many steps wide leaves
# empty levels in its tails, but among 1024 pixels a stretch wider than 1.5 sigmas
# between occupied levels is rare.
_GAP_SIGMAS = 1.5

# An unclipped lattice patch is measured by its paired differences when they give at
# least this many steps, and by the quantized fit otherwise.  From two steps up, the
# median estimate on rounded Gaussian noise lies within 1% of the truth.
_DIFFERENCE_MIN_STEPS = 2.0

# An unclipped lattice patch gives NaN when its paired differences read below
# _DIFFERENCE_MIN_STEPS but its quantized fit reads at least this many steps.  The two
# estimators disagree this way on a resampled product, or on a patch holding a block of
# one value.
_DISAGREEMENT_STEPS = 2.5

# The variance, in steps squared, that rounding and spreading add.  Rounding adds 1/12
# to each pixel, so 1/6 to a difference.  Spreading each whole-step difference evenly
# across its step adds 1/12 more.  That makes 1/4 per difference, or 1/8 per pixel.
_GROUPING_VARIANCE = 1.0 / 8.0

# The quantized fit needs at least this many populated levels to separate the noise
# sigma from the surface.  Two levels fix only one proportion, and every sigma can match
# it.  A level is populated when at least _MIN_LEVEL_PIXELS pixels sit on it, so one
# stray pixel beside a two-level patch does not make a third level.
_MIN_LEVELS = 3
_MIN_LEVEL_PIXELS = 2

# The quantized fit searches sigmas from _MIN_SIGMA_STEPS of a step up to
# _MAX_SIGMA_SPANS times the span of the patch's levels.  The lower end keeps the search
# away from sigmas whose interval probabilities underflow.  The upper end lies well
# beyond any sigma a patch can show.
_MIN_SIGMA_STEPS = 1e-3
_MAX_SIGMA_SPANS = 10.0

# The floor put under a probability before its log is taken.
_SMALLEST_PROBABILITY = 1e-300


@dataclass(frozen=True)
class SkyPatch:
    """One near-uniform patch selected from a frame.

    Parameters:
        v0: Top row of the patch (inclusive).
        u0: Left column of the patch (inclusive).
        size: Patch edge length in pixels.
        mean: Mean signal inside the patch.
        sigma: Per-pixel noise sigma inside the patch, from
            :func:`patch_noise_sigma`.
    """

    v0: int
    u0: int
    size: int
    mean: float
    sigma: float


def paired_difference_sigma(patch: NDArrayFloatType) -> float:
    """Robust per-pixel noise sigma from paired horizontal differences.

    Parameters:
        patch: 2-D array of pixel values (a near-uniform region).

    Returns:
        ``MAD(d) * 1.4826 / sqrt(2)`` where ``d`` are horizontal
        neighbor differences; NaN for a degenerate patch.
    """
    arr = np.asarray(patch, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return float('nan')
    diffs = (arr[:, 1:] - arr[:, :-1]).ravel()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 8:
        return float('nan')
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return float(mad * _MAD_TO_SIGMA / np.sqrt(2.0))


def _grouped_difference_sigma(q: NDArrayFloatType) -> float:
    """Per-pixel noise sigma, in steps, from the paired differences of a whole-step patch.

    The horizontal neighbor differences are whole steps.  Each one is treated as spread
    evenly across its step, so the median and the MAD vary smoothly with the noise.  The
    MAD is scaled to a sigma as for a normal distribution.  Rounding and spreading add
    1/8 step squared to each pixel's variance, and that is subtracted.

    Parameters:
        q: 2-D array of whole-step level indices.  A difference that touches a
            non-finite index is left out.

    Returns:
        The sigma in steps.  It is 0 when the differences spread no more than the
        rounding does, or when no difference is finite.
    """
    diffs = (q[:, 1:] - q[:, :-1]).ravel()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    # The cumulative fraction of the differences is piecewise linear.  It rises across
    # each occupied step and is flat across empty ones, so its only knots are the edges
    # of occupied steps.  The mass within a radius of the center changes slope only where
    # the center plus or minus the radius meets one of those edges, so the radii to those
    # edges are its only breakpoints.  Working from those edges alone costs time in
    # proportion to the number of distinct differences, however far out an outlier lies.
    values, counts = np.unique(diffs, return_counts=True)
    edges = np.unique(np.concatenate([values - 0.5, values + 0.5]))
    below = np.concatenate([[0], np.cumsum(counts)])
    cdf = below[np.searchsorted(values, edges)] / diffs.size
    # When exactly half the differences lie below a run of empty steps, every point of
    # the run is a median, and its midpoint is taken, as for a sample of even size.  The
    # counts decide this exactly.
    halves = np.flatnonzero(2 * below[1:-1] == diffs.size)
    if halves.size:
        center = 0.5 * float(values[halves[0]] + values[halves[0] + 1])
    else:
        center = float(np.interp(0.5, cdf, edges))
    mad = _half_mass_plateau(values, below, center)
    if mad is None:
        # The mass is 0 at radius 0 and linear out to the nearest edge, so radius 0 is a
        # breakpoint too, and the deviation may lie inside the central step.
        radii = np.unique(np.concatenate([[0.0], np.abs(edges - center)]))
        mass = np.interp(center + radii, edges, cdf) - np.interp(center - radii, edges, cdf)
        mad = float(np.interp(0.5, mass, radii))
    variance = (mad * _MAD_TO_SIGMA) ** 2 / 2.0 - _GROUPING_VARIANCE
    return float(np.sqrt(max(variance, 0.0)))


def _half_mass_plateau(
    values: NDArrayFloatType, below: NDArrayIntType, center: float
) -> float | None:
    """The midpoint of the radii about the center that hold exactly half the differences.

    The mass within a radius of the center can stay at one half over an interval of
    radii.  That happens when both ends of the radius's span lie in runs of empty steps,
    and exactly half the differences lie between those runs.  Every radius in the
    interval is a median absolute deviation, and the midpoint is taken, as for a sample
    of even size.  The pair of runs is found from the whole-step counts, so float
    rounding of the mass can't pick some other point of the interval.

    Parameters:
        values: The distinct whole-step differences, ascending.
        below: The number of differences below each of ``values``, then their total.
        center: The differences' median.

    Returns:
        The interval's midpoint, or None when the mass reaches one half at a single
        radius.
    """
    total = int(below[-1])
    if total % 2:
        return None
    # The run of empty steps before values[k] starts at values[k - 1] + 0.5 and ends at
    # values[k] - 0.5, with below[k] differences beneath it.  The first run is unbounded
    # below and the last unbounded above.  A run between neighboring values is a point.
    starts = np.concatenate([[-np.inf], values + 0.5])
    ends = np.concatenate([values - 0.5, [np.inf]])
    outer = np.searchsorted(below, below + total // 2)
    inner = np.flatnonzero(outer < below.size)
    inner = inner[below[outer[inner]] == below[inner] + total // 2]
    outer = outer[inner]
    shortest = np.maximum(center - ends[inner], starts[outer] - center)
    longest = np.minimum(center - starts[inner], ends[outer] - center)
    plateau = np.flatnonzero(longest > shortest)
    if plateau.size == 0:
        return None
    return 0.5 * float(shortest[plateau[0]] + longest[plateau[0]])


def _bulk_levels(q: NDArrayFloatType, max_gap: int) -> tuple[int, int]:
    """The lowest and highest level of the run of occupied levels holding the most pixels.

    A run is a stretch of occupied levels separated by at most ``max_gap`` consecutive
    empty levels.  A wider stretch of empty levels ends it.  When two runs hold equally
    many pixels, the lower one is taken.

    Parameters:
        q: Whole-step level indices, 0 or more, every one finite.
        max_gap: The most consecutive empty levels a run may span.

    Returns:
        The run's lowest and highest level.
    """
    levels, counts = np.unique(q.astype(np.int64), return_counts=True)
    breaks = np.flatnonzero(np.diff(levels) - 1 > max_gap)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [levels.size - 1]])
    pixels = [int(counts[a : b + 1].sum()) for a, b in zip(starts, ends, strict=True)]
    best = int(np.argmax(pixels))
    return int(levels[starts[best]]), int(levels[ends[best]])


def _max_gap(sigma_steps: float) -> int:
    """The most consecutive empty levels a run of noise of ``sigma_steps`` may span."""
    return int(np.floor(_GAP_SIGMAS * sigma_steps))


def _excess_limit(n_pixels: int, expected_fraction: float) -> float:
    """The most pixels a patch of ``n_pixels`` may hold beyond ``expected_fraction``."""
    return (expected_fraction + _EXCESS_FRACTION) * n_pixels + _EXCESS_PIXELS


def _is_clipped(q: NDArrayFloatType, sigma_steps: float) -> bool:
    """Whether the patch's lowest or highest level gathers the pixels beyond it.

    Such a level belongs to the bulk of the levels (see :func:`lattice_noise_sigma`) and
    holds more than _EXCESS_FRACTION of the patch plus _EXCESS_PIXELS pixels.  Unclipped
    Gaussian noise of two steps or more leaves only a few pixels on its end levels.

    Parameters:
        q: Whole-step level indices, 0 or more, every one finite.
        sigma_steps: The paired-difference sigma in steps.
    """
    bulk_low, bulk_high = _bulk_levels(q, _max_gap(sigma_steps))
    limit = _excess_limit(q.size, 0.0)
    low_count = int(np.sum(q == 0.0)) if bulk_low == 0 else 0
    high_count = int(np.sum(q == bulk_high)) if bulk_high == int(q.max()) else 0
    return max(low_count, high_count) > limit


def _end_level_overflow(
    q: NDArrayFloatType,
    mu: NDArrayFloatType,
    sigma: float,
    bulk: tuple[int, int],
    *,
    open_low: bool,
    open_high: bool,
) -> float:
    """How far the fuller end level of the bulk overflows the fitted model's prediction.

    The model predicts a level's population as the sum, over every pixel, of the
    probability that the pixel's surface plus Gaussian noise rounds to that level.  An
    open end level is predicted to hold the whole mass beyond its inner edge.  A level's
    allowance is its prediction plus _LEVEL_SIGMAS times the square root of the
    prediction, which is the count's Poisson scatter.  Its overflow is the number of
    pixels it holds beyond that allowance.

    Parameters:
        q: Whole-step level index of each pixel.
        mu: The fitted surface at each pixel, in steps.
        sigma: The fitted sigma in steps.
        bulk: The lowest and highest level of the bulk.
        open_low: Whether the bulk's lowest level gathers everything below it.
        open_high: Whether the bulk's highest level gathers everything above it.

    Returns:
        The larger overflow of the two end levels, in pixels; negative when both hold
        fewer pixels than their allowance.
    """
    low, high = bulk
    # Each pixel's probability of lying below the low level's upper edge, and above the
    # high level's lower edge.
    below_low = scipy.special.ndtr((low + 0.5 - mu) / sigma)
    above_high = scipy.special.ndtr((mu - high + 0.5) / sigma)
    if not open_low:
        below_low = below_low - scipy.special.ndtr((low - 0.5 - mu) / sigma)
    if not open_high:
        above_high = above_high - scipy.special.ndtr((mu - high - 0.5) / sigma)
    overflows = []
    for level, probability in ((low, below_low), (high, above_high)):
        predicted = float(np.sum(probability))
        count = int(np.sum(q == level))
        overflows.append(count - predicted - _LEVEL_SIGMAS * np.sqrt(predicted))
    return float(max(overflows))


def _widened_levels(
    levels: NDArrayFloatType, first: int, last: int, central: tuple[float, float]
) -> tuple[int, int]:
    """The range of the central levels widened by the nearest level on each side.

    A side's nearest level is left out when it lies more than _NEIGHBOR_GAPS times as far
    from the central levels as the other side's.  So a far level, such as a bright
    pixel's, joins the fit unless the other side has a level less than half as far away.

    Parameters:
        levels: The distinct levels, ascending.
        first: The index of the lowest central level.
        last: One past the index of the highest central level.
        central: The lowest and highest central value, which bound the central levels.

    Returns:
        The widened range as the index of its lowest level and one past its highest.
    """
    inner_low = float(levels[first]) if last > first else central[0]
    inner_high = float(levels[last - 1]) if last > first else central[1]
    gap_below = inner_low - float(levels[first - 1]) if first > 0 else np.inf
    gap_above = float(levels[last]) - inner_high if last < levels.size else np.inf
    nearer = min(gap_below, gap_above)
    if first > 0 and gap_below <= _NEIGHBOR_GAPS * nearer:
        first -= 1
    if last < levels.size and gap_above <= _NEIGHBOR_GAPS * nearer:
        last += 1
    return first, last


def quantization_step(values: NDArrayFloatType) -> float | None:
    """The step of the lattice a sample's values sit on, or None if they aren't quantized.

    The step starts as the smallest gap between distinct values.  The step and the
    lattice's origin are then fitted by least squares to the levels of the central 98%
    of the values.  The fit lets whole DN scaled into float32 count as a lattice,
    although each scaled level carries its own rounding error.  When the central values
    sit on fewer than two levels, the fit also uses the nearest level on each side.  It
    leaves out a side's level that lies more than twice as far from the central levels
    as the other side's.  So a far level, such as a bright pixel's, stays out of the fit
    whenever the sky has a level at most half as far away on the other side.

    Each level is then checked against the fitted lattice.  It passes when it lies
    within a thousandth of a step of a lattice level, or within a few float32 rounding
    errors of its own magnitude, whichever is larger.  Float64 values get the same
    float32 allowance, since they may come from a float32 product.  A level outside the
    fitted ones is also allowed four standard errors of the fit carried out to it.  Far
    enough out, that allowance reaches half a step, and the level passes whatever its
    value.  When a value's own float32 rounding reaches half a step, its lattice level
    can't be told, so it is neither fitted nor checked.  A bright pixel far above a
    finely quantized sky is an example.

    The step returned is refitted across the levels the fit can place, which are those
    whose allowance stays under half a step.  When fewer than two levels can be placed,
    the step is unknown, and None is returned.  That happens when the fitted step falls
    far enough below the smallest gap that even the central levels' float32 allowance
    reaches half a step.  An example is a sky on two levels 1.75 apart near 1e6, beside a
    bright pair 1.125 apart.

    A lattice counts only when the sample could fill it.  The central 98% must span no
    more steps than there are values, and the step must be well above the spacing of
    floating-point numbers at the central values.  Without those conditions every
    float32 frame would pass, since its values are all whole multiples of the float
    spacing.  Every value must also lie within 2**31 steps of the lowest central level,
    which keeps the noise estimators' whole-step arithmetic exact.  A sentinel value
    such as -1e17 beside a sky of whole DN breaks that bound, and None is returned.  A
    sentinel inside the bound, around a billion steps out, can still make the quantized
    fit read NaN.

    Parameters:
        values: Pixel values (any shape; non-finite entries are ignored).

    Returns:
        The fitted step when the lattice counts and every level checked passes; None
        for continuous values and for fewer than two distinct values.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    levels = np.unique(finite)
    if levels.size < 2:
        return None
    step = float(np.min(np.diff(levels)))
    low, high = (float(v) for v in np.quantile(finite, [0.01, 0.99]))
    if (high - low) / step > finite.size:
        return None
    if step <= _FLOAT_SPACING_FRACTION * max(abs(low), abs(high)):
        return None
    # A level whose float32 rounding reaches half a step could be on any of several
    # lattice levels, so it is neither fitted nor checked.  Every central level is
    # resolved, because the step passed the float-spacing condition there.
    resolved = levels[_FLOAT_ROUNDING_EPS * np.abs(levels) < 0.5 * step]
    first = int(np.searchsorted(resolved, low, side='left'))
    last = int(np.searchsorted(resolved, high, side='right'))
    if last - first < 2:
        first, last = _widened_levels(resolved, first, last, (low, high))
    if last - first < 2:
        return None
    central = resolved[first:last]
    fitted = central - central[0]
    for _ in range(2):
        index = np.round(fitted / step)
        step, origin = (float(c) for c in np.polyfit(index, fitted, 1))
        if step <= 0.0:
            return None
    if float(np.max(np.abs(levels - central[0]))) > _MAX_LEVEL_STEPS * step:
        return None
    offsets = resolved - central[0]
    lattice_index = np.round((offsets - origin) / step)
    misfit = np.abs(offsets - origin - step * lattice_index)
    # The fit's standard error at a level comes from the fitted levels' scatter about the
    # fit, or from a single float32 rounding at their magnitude, whichever is larger.
    spread = index - np.mean(index)
    residual = fitted - origin - step * index
    scatter = float(np.sqrt(np.sum(residual**2) / (central.size - 2))) if central.size > 2 else 0.0
    sigma = max(scatter, _FLOAT32_EPS / 2.0 * float(np.max(np.abs(central))))
    standard_error = sigma * np.sqrt(
        1.0 / central.size + (lattice_index - np.mean(index)) ** 2 / float(np.sum(spread**2))
    )
    beyond = (lattice_index < index.min()) | (lattice_index > index.max())
    tolerance = _FLOAT_ROUNDING_EPS * np.abs(resolved) + np.where(
        beyond, _FIT_SIGMAS * standard_error, 0.0
    )
    if np.any(misfit > np.maximum(_LATTICE_TOLERANCE * step, tolerance)):
        return None
    placed = tolerance < 0.5 * step
    # The central levels' tolerance is their float32 allowance alone, which is under half
    # the starting step.  A fitted step finer than the starting one can leave even those
    # levels unplaced, and then fewer than two levels pin the step down.
    if int(np.count_nonzero(placed)) < 2:
        return None
    placed_index = lattice_index[placed] - lattice_index[placed][0]
    return float(np.polyfit(placed_index, resolved[placed] - resolved[placed][0], 1)[0])


def _quadratic_design(vv: NDArrayFloatType, uu: NDArrayFloatType, size: int) -> NDArrayFloatType:
    """The columns of a quadratic surface over a patch: 1, y, x, y^2, xy, x^2.

    The coordinates are scaled to run from about -1 to 1 across the patch, which keeps
    the fit well conditioned.
    """
    half = max(size, 2) / 2.0
    y = (vv - (size - 1) / 2.0) / half
    x = (uu - (size - 1) / 2.0) / half
    return np.column_stack([np.ones_like(x), y, x, y * y, x * y, x * x])


def _surface(design: NDArrayFloatType, coef: NDArrayFloatType) -> NDArrayFloatType:
    """Return ``design @ coef`` as a left-to-right sum of scaled columns.

    The simulator keeps BLAS products out of its arithmetic, because their last bit
    depends on the CPU's kernel.  Plain multiplies and adds round the same everywhere.
    """
    total = design[:, 0] * coef[0]
    for col in range(1, design.shape[1]):
        total = total + design[:, col] * coef[col]
    return np.asarray(total, dtype=np.float64)


def _column_sums(design: NDArrayFloatType, weights: NDArrayFloatType) -> NDArrayFloatType:
    """Return ``design.T @ weights`` as one NumPy sum per column, without a BLAS product."""
    return np.array(
        [float(np.sum(design[:, col] * weights)) for col in range(design.shape[1])],
        dtype=np.float64,
    )


def _log_interval_probability(lower: NDArrayFloatType, upper: NDArrayFloatType) -> NDArrayFloatType:
    """``log(Phi(upper) - Phi(lower))`` for ``lower < upper``, accurate in both tails.

    An interval too narrow to resolve in double precision gets the log of the smallest
    positive probability rather than minus infinity.
    """
    flip = lower > 0.0
    low = np.where(flip, -upper, lower)
    high = np.where(flip, -lower, upper)
    log_high = scipy.special.log_ndtr(high)
    log_low = scipy.special.log_ndtr(low)
    fraction = np.maximum(-np.expm1(log_low - log_high), _SMALLEST_PROBABILITY)
    return np.asarray(log_high + np.log(fraction), dtype=np.float64)


def _normal_density(z: NDArrayFloatType) -> NDArrayFloatType:
    """The standard normal density, 0 at infinite ``z``."""
    return np.asarray(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi), dtype=np.float64)


def _z_times_density(z: NDArrayFloatType) -> NDArrayFloatType:
    """``z`` times the standard normal density, 0 at infinite ``z``."""
    finite = np.isfinite(z)
    zf = np.where(finite, z, 0.0)
    return np.where(finite, zf * _normal_density(zf), 0.0)


def _populated_levels(levels: NDArrayFloatType) -> int:
    """How many distinct levels hold at least _MIN_LEVEL_PIXELS pixels each."""
    counts = np.unique(levels, return_counts=True)[1]
    return int(np.sum(counts >= _MIN_LEVEL_PIXELS))


def _first_guess(q: NDArrayFloatType, design: NDArrayFloatType) -> tuple[NDArrayFloatType, float]:
    """A least-squares surface and a robust sigma, refit twice without far pixels.

    The guess only seeds the maximum-likelihood fit and its first window, so it need
    not be accurate.  On a clipped patch, where half the residuals share a value, its
    sigma comes out far too small, and the fit corrects it.
    """
    near = np.ones(q.size, dtype=bool)
    robust_sigma = 0.5
    coef = np.zeros(design.shape[1])
    for _ in range(3):
        coef = np.asarray(np.linalg.lstsq(design[near], q[near], rcond=None)[0], dtype=np.float64)
        resid = q - _surface(design, coef)
        deviation = np.abs(resid[near] - np.median(resid[near]))
        robust_sigma = max(_MAD_TO_SIGMA * float(np.median(deviation)), 0.5)
        near = np.abs(resid) <= 4.0 * robust_sigma + 1.0
    return coef, robust_sigma


def _level_window(mu: NDArrayFloatType, sigma: float) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Each pixel's lowest and highest window level, for surface ``mu`` and ``sigma``.

    The window's levels cover the surface out to _WINDOW_SIGMAS sigmas on each side,
    and at least one level each side of the surface's nearest level.
    """
    nearest = np.round(mu)
    reach = _WINDOW_SIGMAS * sigma
    bottom = np.minimum(nearest - 1.0, np.floor(mu - reach + 0.5))
    top = np.maximum(nearest + 1.0, np.ceil(mu + reach - 0.5))
    return bottom, top


def _fit_censored(
    design: NDArrayFloatType,
    lower_edge: NDArrayFloatType,
    upper_edge: NDArrayFloatType,
    start: NDArrayFloatType,
    log_sigma_bounds: tuple[float, float],
) -> NDArrayFloatType | None:
    """Maximize the likelihood of each pixel lying between its lower and upper edges.

    Parameters:
        design: The quadratic surface's columns at each pixel.
        lower_edge: Each pixel's lower edge in steps, minus infinity when open-ended.
        upper_edge: Each pixel's upper edge in steps, infinity when open-ended.
        start: The surface coefficients and the log of sigma to start from.
        log_sigma_bounds: The range of log sigma searched.

    Returns:
        The fitted surface coefficients followed by log sigma, or None when the
        optimizer reports failure.
    """
    n_coef = design.shape[1]

    def neg_log_likelihood(params: NDArrayFloatType) -> tuple[float, NDArrayFloatType]:
        sigma = float(np.exp(params[n_coef]))
        mu = _surface(design, params[:n_coef])
        z_upper = (upper_edge - mu) / sigma
        z_lower = (lower_edge - mu) / sigma
        log_p = _log_interval_probability(z_lower, z_upper)
        p = np.maximum(np.exp(log_p), _SMALLEST_PROBABILITY)
        d_mu = (_normal_density(z_upper) - _normal_density(z_lower)) / (sigma * p)
        d_log_sigma = (_z_times_density(z_upper) - _z_times_density(z_lower)) / p
        grad = np.empty(n_coef + 1)
        grad[:n_coef] = _column_sums(design, d_mu)
        grad[n_coef] = float(np.sum(d_log_sigma))
        return -float(np.sum(log_p)), grad

    fit = scipy.optimize.minimize(
        neg_log_likelihood,
        start,
        jac=True,
        method='L-BFGS-B',
        bounds=[(None, None)] * n_coef + [log_sigma_bounds],
    )
    if not fit.success:
        return None
    return np.asarray(fit.x, dtype=np.float64)


def lattice_noise_sigma(patch: NDArrayFloatType, step: float) -> float:
    """Noise sigma of a quantized patch, by fitting the quantization.

    Each pixel is modeled as a smooth surface plus Gaussian noise of sigma ``s``,
    rounded to the nearest lattice level.  A pixel whose surface value is ``mu`` lands on
    level ``q`` with probability
    ``Phi((q + step/2 - mu) / s) - Phi((q - step/2 - mu) / s)``.  The surface is a
    quadratic in the pixel's row and column.  Its six coefficients are fitted together
    with ``s`` by maximum likelihood, so neither a gradient nor curvature across the
    patch is read as noise.  The fit recovers a sigma well below one step, where paired
    differences read 0.  But structure the quadratic can't follow, such as a textured
    scene, widens the spread of levels and reads as noise.  That is why
    :func:`patch_noise_sigma` uses this fit only for clipped patches and for noise below
    two steps.

    The bulk of the patch is the run of occupied levels that holds the most pixels.  An
    empty level ends a run, but noise many steps wide leaves empty levels in its tails.
    So a run may span empty stretches up to 1.5 paired-difference sigmas wide.  A pixel
    cut off from the bulk counts only as lying beyond the bulk's end, however far beyond
    it lies.  A hot pixel is one example, and a zero-filled dropout below a sky several
    sigmas above zero is another.

    Each pixel is fitted against a window of levels around the surface, reaching three
    fitted sigmas beyond it on each side.  The window's end levels are open-ended.  A
    pixel on or below the lowest level is known only to lie below that level's upper
    edge, and a pixel on or above the highest level only to lie above its lower edge.
    So every pixel enters the fit, and a pixel far from the noise counts only as one
    pixel beyond the window's edge.  The first window comes from a least-squares surface
    and a robust sigma.  Each fit's surface and sigma then rebuild the window, until it
    repeats a window already fitted, for at most ten fits.

    The patch's lowest and highest levels are open-ended as well when they belong to
    the bulk.  A patch clipped at the detector's floor or at saturation gathers every
    value beyond the clip onto one level.  With that level open-ended, the patch is
    fitted as an unclipped one is.

    The fit returns NaN when it can't trust the patch:

    - Too many pixels lie beyond the noise.  Gaussian noise puts 0.27% of its pixels
      beyond three sigmas.  The fit returns NaN when the pixels beyond their windows or
      cut off from the bulk exceed that fraction by more than 1% of the patch plus three
      pixels.  A pixel on a clipped floor counts only when it lies below its window, and
      a pixel on a clipped ceiling only when it lies above its window.
    - An end level of the bulk is overfull.  A contaminant right next to the bulk, such
      as dropout rows at zero under a sky a few sigmas above the floor, piles onto one
      level.  On the patch's lowest or highest level, the fit takes that pile for a clip
      and widens to follow it.  So after the fit, the bulk's lowest and highest levels
      are each compared with the population the fitted model predicts.  An open end
      level is predicted to hold the model's whole mass beyond its inner edge.  The fit
      returns NaN when an end level holds more pixels than predicted by more than four
      times the square root of the prediction, plus 0.27% of the patch, plus three
      pixels.  The levels between are not checked, because a real camera's levels need
      not be equally wide.  The darkest sky of real Galileo frames puts about 1.4 times
      the predicted pixels on its fullest level.
    - Too few levels are populated.  The sigma can be measured only when at least three
      levels each hold at least two pixels, counting each window's end levels as
      holding the pixels beyond them.  When the noise is so far below one step that the
      pixels take only two values, they fix one proportion along the surface, and every
      sigma, with a matching surface, reproduces it.  A patch of one value fails here
      too.
    - The optimizer fails to converge.

    Parameters:
        patch: 2-D array of pixel values on a lattice of spacing ``step``.
        step: The lattice spacing, as :func:`quantization_step` returns it.

    Returns:
        The fitted sigma in the patch's units, or NaN when the sigma is not
        measurable.
    """
    arr = np.asarray(patch, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.ndim != 2 or int(finite.sum()) < 8:
        return float('nan')
    values = arr[finite]
    q = np.round((values - values.min()) / step)
    if _populated_levels(q) < _MIN_LEVELS:
        return float('nan')
    vv, uu = np.nonzero(finite)
    design = _quadratic_design(vv.astype(np.float64), uu.astype(np.float64), max(arr.shape))
    q_high = float(q.max())
    log_sigma_bounds = (np.log(_MIN_SIGMA_STEPS), np.log(_MAX_SIGMA_SPANS * (q_high + 1.0)))

    q_grid = np.round((arr - values.min()) / step)
    bulk_low, bulk_high = _bulk_levels(q, _max_gap(_grouped_difference_sigma(q_grid)))
    below_bulk = q < bulk_low
    above_bulk = q > bulk_high
    clipped_low = (q <= 0.0) & (bulk_low == 0)
    clipped_high = (q >= q_high) & (bulk_high == int(q_high))
    open_low = below_bulk | clipped_low
    open_high = above_bulk | clipped_high
    q_fit = np.clip(q, bulk_low - 1.0, bulk_high + 1.0)

    coef, sigma = _first_guess(q, design)
    params = np.append(coef, np.clip(np.log(sigma), *log_sigma_bounds))
    fitted_windows: list[tuple[bytes, bytes]] = []
    censored = q_fit
    beyond = below_bulk | above_bulk
    for _ in range(_MAX_WINDOW_FITS):
        bottom, top = _level_window(_surface(design, params[:-1]), float(np.exp(params[-1])))
        censored = np.clip(q_fit, bottom, top)
        beyond = (
            below_bulk
            | above_bulk
            | ((q_fit < bottom) & ~clipped_high)
            | ((q_fit > top) & ~clipped_low)
        )
        lower_edge = np.where((censored <= bottom) | open_low, -np.inf, censored - 0.5)
        upper_edge = np.where((censored >= top) | open_high, np.inf, censored + 0.5)
        window = (lower_edge.tobytes(), upper_edge.tobytes())
        if window in fitted_windows:
            break
        fitted_windows.append(window)
        fitted = _fit_censored(design, lower_edge, upper_edge, params, log_sigma_bounds)
        if fitted is None:
            return float('nan')
        params = fitted
    if _populated_levels(censored) < _MIN_LEVELS:
        return float('nan')
    if int(beyond.sum()) > _excess_limit(q.size, _GAUSSIAN_BEYOND_FRACTION):
        return float('nan')
    sigma = float(np.exp(params[-1]))
    overflow = _end_level_overflow(
        q,
        _surface(design, params[:-1]),
        sigma,
        (bulk_low, bulk_high),
        open_low=bulk_low == 0,
        open_high=bulk_high == int(q_high),
    )
    if overflow > _GAUSSIAN_BEYOND_FRACTION * q.size + _EXCESS_PIXELS:
        return float('nan')
    return sigma * step


def patch_noise_sigma(patch: NDArrayFloatType) -> float:
    """Per-pixel noise sigma of one near-uniform patch, quantized or not.

    A patch of continuous values is measured with :func:`paired_difference_sigma`.  A
    quantized patch, one whose values sit on a lattice (:func:`quantization_step`), is
    measured by its paired differences, read from their whole-step histogram, when they
    give at least two steps and the patch is not clipped.  Otherwise it is fitted with
    :func:`lattice_noise_sigma`.  The differences come first because they cancel
    structure that varies slowly across the patch and ignore a minority of contaminated
    pixels.  The fit reads every level's population as noise.

    A patch is clipped when its lowest or highest level belongs to the bulk of its
    levels and holds more than 1% of its pixels plus three.  The clipped pixels shrink
    the differences, so a clipped patch goes to the fit, which models the clipping.

    An unclipped patch returns NaN when its differences read below two steps but its fit
    reads 2.5 steps or more, because the two estimators disagree.  Paired differences
    assume that neighboring pixels carry independent noise.  A product resampled after
    readout mixes neighbors, so its differences read the noise low, while the fit reads
    each pixel's own spread.  Near two steps, such a patch would otherwise get one
    estimate or the other by chance.  A block of pixels on one value inside the sky's
    range shrinks the differences and widens the fit in the same way.

    Each estimate gives the sigma of the noise before rounding.  A product that was
    quantized and then rescaled pixel by pixel, such as a flat-fielded frame, no longer
    sits on a lattice.  Its estimate therefore also includes the rounding's own spread
    of step/sqrt(12).  That adds about 4% to a sigma of one step, and less to a larger
    one.

    Parameters:
        patch: 2-D array of pixel values (a near-uniform region).

    Returns:
        The sigma in the patch's units.  NaN for a degenerate patch, for a patch of one
        value, and for a quantized patch whose sigma can't be measured.
    """
    arr = np.asarray(patch, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.max()) == float(finite.min()):
        return float('nan')
    step = quantization_step(arr)
    if step is None:
        return paired_difference_sigma(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2 and finite.size == arr.size:
        q = np.round((arr - float(finite.min())) / step)
        sigma = _grouped_difference_sigma(q)
        clipped = _is_clipped(q, sigma)
        if sigma >= _DIFFERENCE_MIN_STEPS and not clipped:
            return sigma * step
        fitted = lattice_noise_sigma(arr, step)
        if not clipped and fitted >= _DISAGREEMENT_STEPS * step:
            return float('nan')
        return fitted
    return lattice_noise_sigma(arr, step)


def find_uniform_patches(
    image: NDArrayFloatType,
    *,
    patch_size: int = 32,
    max_mean_quantile: float | None = 0.25,
    max_structure_ratio: float = 2.0,
) -> list[SkyPatch]:
    """Select near-uniform patches from a frame.

    The frame is tiled into non-overlapping ``patch_size`` squares.  A patch
    qualifies when (a) its mean lies at or below the frame's
    ``max_mean_quantile`` patch-mean quantile (sky selection; disable by
    passing None to accept every signal level for the noise-vs-signal
    statistic), and (b) its internal structure is noise-dominated: the
    patch's total standard deviation does not exceed ``max_structure_ratio``
    times its noise sigma (:func:`patch_noise_sigma`).  Condition (b) rejects
    patches crossed by limbs, rings, or bright stars, whose spatial structure
    would masquerade as noise.  A patch whose sigma is not measurable, such
    as a tile of one value, is skipped.

    Parameters:
        image: 2-D frame in its native units.
        patch_size: Edge length of the square tiles.
        max_mean_quantile: Patch-mean quantile at or below which a patch
            counts as sky, or None to skip the sky cut.
        max_structure_ratio: Maximum allowed ratio of total patch standard
            deviation to noise sigma.

    Returns:
        The qualifying patches with their means and sigmas.
    """
    arr = np.asarray(image, dtype=np.float64)
    n_v = arr.shape[0] // patch_size
    n_u = arr.shape[1] // patch_size
    if n_v == 0 or n_u == 0:
        return []
    candidates: list[SkyPatch] = []
    means: list[float] = []
    for iv in range(n_v):
        for iu in range(n_u):
            v0 = iv * patch_size
            u0 = iu * patch_size
            tile = arr[v0 : v0 + patch_size, u0 : u0 + patch_size]
            if not np.all(np.isfinite(tile)):
                continue
            sigma = patch_noise_sigma(tile)
            if not np.isfinite(sigma):
                continue
            mean = float(tile.mean())
            total_sd = float(tile.std())
            if total_sd > max_structure_ratio * sigma:
                continue
            candidates.append(SkyPatch(v0=v0, u0=u0, size=patch_size, mean=mean, sigma=sigma))
            means.append(mean)
    if max_mean_quantile is None or not candidates:
        return candidates
    cutoff = float(np.quantile(np.asarray(means), max_mean_quantile))
    return [p for p in candidates if p.mean <= cutoff]


def noisy_patches(patches: list[SkyPatch]) -> list[SkyPatch]:
    """Select the patches whose noise sigma is positive.

    A patch of continuous values whose neighbors mostly share a value, such as
    zero-filled missing data with a few stray pixels, has a paired-difference sigma of
    0, so there is no noise to compare.  A tile of one value never becomes a patch,
    because its sigma can't be measured.

    Parameters:
        patches: One frame's sky patches.

    Returns:
        The patches with a finite, positive sigma, in their input order.
    """
    return [p for p in patches if np.isfinite(p.sigma) and p.sigma > 0.0]


def dark_sky_patches(patches: list[SkyPatch], ceiling_level: float) -> list[SkyPatch]:
    """Select the sky patches that lie on dark sky.

    A patch is kept when its mean lies at or below ``ceiling_level``, however far
    below.  The selection depends on each patch's level alone and never on any
    noise sigma, so it never selects on the quantity being compared.

    Parameters:
        patches: One frame's sky patches.
        ceiling_level: The highest mean a dark-sky patch may have, in the
            patches' units.

    Returns:
        The patches at or below the ceiling, in their input order.
    """
    return [p for p in patches if p.mean <= ceiling_level]


def radial_power_spectrum(
    patch: NDArrayFloatType, *, n_bins: int = 16
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Radially averaged spatial power spectrum of one patch.

    The patch is mean-subtracted and windowed (Hann, separable) to suppress
    edge leakage, then the 2-D periodogram is averaged over annuli of
    spatial frequency.  The DC bin is excluded.

    Parameters:
        patch: 2-D square array of pixel values.
        n_bins: Number of radial frequency bins between 0 and the Nyquist
            frequency (0.5 cycles / pixel).

    Returns:
        ``(freq, power)`` arrays of length ``n_bins``: annulus-center
        spatial frequency in cycles / pixel and mean periodogram power.
        Empty annuli carry NaN power.
    """
    arr = np.asarray(patch, dtype=np.float64)
    size_v, size_u = arr.shape
    window = np.outer(np.hanning(size_v), np.hanning(size_u))
    spectrum = np.abs(np.fft.fft2((arr - arr.mean()) * window)) ** 2
    freq_v = np.fft.fftfreq(size_v)[:, np.newaxis]
    freq_u = np.fft.fftfreq(size_u)[np.newaxis, :]
    freq_r = np.sqrt(freq_v**2 + freq_u**2)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    power = np.full(n_bins, np.nan)
    flat_r = freq_r.ravel()
    flat_p = spectrum.ravel()
    keep = flat_r > 0.0
    flat_r = flat_r[keep]
    flat_p = flat_p[keep]
    for i in range(n_bins):
        in_bin = (flat_r >= edges[i]) & (flat_r < edges[i + 1])
        if np.any(in_bin):
            power[i] = float(flat_p[in_bin].mean())
    return centers, power
