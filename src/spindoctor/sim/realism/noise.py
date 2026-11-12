"""Sky-region noise statistics for realism FOM 1.

Science frames have no flat-field pairs, so noise is estimated inside near-uniform
patches, and the estimator depends on whether the patch's values are quantized.

A patch whose values take any real value is measured by local differencing: paired
horizontal pixel differences cancel scene structure that varies slowly across the
patch, and dividing the difference scale by sqrt(2) recovers the per-pixel sigma.  A
robust (MAD-based) scale keeps residual stars, cosmic rays, and hot pixels in a patch
from inflating the estimate; naive signal-binning would conflate scene texture with
noise, which is exactly what this estimator avoids.

A patch whose values sit on a lattice of equally spaced levels, such as whole DN, is
measured by fitting the quantization itself (:func:`lattice_noise_sigma`): a
quadratic surface plus Gaussian noise, rounded to the lattice, fitted by maximum
likelihood.  The paired-difference scale is not valid there: when the noise is below
about one step, more than half the differences are 0, the MAD is 0, and a camera whose
noise is smaller than one step contributes no sky samples at all.  Above one step, the
MAD of whole-step differences is itself a whole or half step, so the estimate can only
take values about 0.5 step apart.

The sky spatial power spectrum (radially averaged over annuli of spatial
frequency) catches banding and coherent noise that a scalar sigma cannot:
white read noise is flat, banding is a comb.
"""

from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special

from spindoctor.support.types import NDArrayFloatType

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

# A patch's distinct values lie on a lattice when each one lies within this fraction of
# a step of the lattice level fitted to it, or within _FLOAT_ROUNDING_EPS of its own
# magnitude, whichever is larger.
_LATTICE_TOLERANCE = 1e-3

# The rounding error a float32 value can carry, relative to its magnitude: a few units
# of float32's machine epsilon.  A lattice of whole DN scaled into float32, such as DN
# times 1.234e-3 near DN 3000, sits off its levels by this much.
_FLOAT_ROUNDING_EPS = 4.0 * float(np.finfo(np.float32).eps)

# A step at or below this fraction of the largest value's magnitude is the spacing of
# floating-point numbers, not a quantization: the float32 values near 3e-3 I/F are
# 2.3e-10 apart.
_FLOAT_SPACING_FRACTION = 1e-6

# The quantized fit conditions each pixel on a window of levels around the fitted surface
# that reaches at least this many fitted sigmas beyond the surface on each side, and at
# least one level beyond the surface's nearest level.  A pixel outside its window counts
# as lying beyond the window's edge.  The window is rebuilt from each fit until no
# pixel's interval changes, or for at most _MAX_WINDOW_FITS fits.
_WINDOW_SIGMAS = 3.0
_MAX_WINDOW_FITS = 10

# The fewest populated levels from which the quantized fit can tell the noise sigma
# from the level and its slope: two levels fix one proportion, which any sigma can
# match.  A level is populated when at least _MIN_LEVEL_PIXELS pixels sit on it, so one
# stray pixel next to a two-level patch cannot make a third level.
_MIN_LEVELS = 3
_MIN_LEVEL_PIXELS = 2

# The quantized fit searches sigmas from _MIN_SIGMA_STEPS of a step, which keeps the
# search away from sigmas whose interval probabilities underflow, up to
# _MAX_SIGMA_SPANS times the span of the patch's levels, well beyond any sigma a patch
# can show.
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


def quantization_step(values: NDArrayFloatType) -> float | None:
    """The spacing of the lattice a sample's values sit on, or None if they sit on none.

    The step starts as the smallest gap between distinct values and is refined by a
    least-squares fit of the values on their whole-step indices, so a lattice of whole
    DN scaled into float32 is found although each level carries its own rounding
    error.  A value lies on the lattice when it is within a thousandth of a step of its
    fitted level, or within a few float32 rounding errors of its own magnitude,
    whichever is larger.

    A lattice counts only when the sample could fill it: its central 98% spans no
    more steps than there are values, and the step is far coarser than the spacing of
    floating-point numbers at the values' magnitude.  Without those conditions every
    float32 frame would pass, since its values are all whole multiples of the float
    spacing.

    Parameters:
        values: Pixel values (any shape; non-finite entries are ignored).

    Returns:
        The fitted step when every distinct value lies on the lattice; None for
        continuous values and for fewer than two distinct values.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    levels = np.unique(finite)
    if levels.size < 2:
        return None
    step = float(np.min(np.diff(levels)))
    magnitude = float(np.max(np.abs(levels)))
    if step <= _FLOAT_SPACING_FRACTION * magnitude:
        return None
    low, high = np.quantile(finite, [0.01, 0.99])
    if (high - low) / step > finite.size:
        return None
    offsets = levels - levels[0]
    for _ in range(2):
        index = np.round(offsets / step)
        step, origin = (float(c) for c in np.polyfit(index, offsets, 1))
        if step <= 0.0:
            return None
    misfit = float(np.max(np.abs(offsets - origin - step * np.round(offsets / step))))
    if misfit > max(_LATTICE_TOLERANCE * step, _FLOAT_ROUNDING_EPS * magnitude):
        return None
    return step


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
    depends on the CPU's kernel; plain multiplies and adds round the same everywhere.
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
    not be accurate: on a clipped patch, where half the residuals share a value, its
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
    rounded to the nearest lattice level, so a pixel lands on level ``q`` with
    probability ``Phi((q + step/2 - mu) / s) - Phi((q - step/2 - mu) / s)``, where
    ``mu`` is the surface at that pixel.  The surface is a quadratic in the pixel's
    row and column, and its six coefficients are fitted together with ``s`` by maximum
    likelihood, so neither a gradient nor curvature across the patch is mistaken for
    noise.  The fit recovers a sigma well below one step, which local differencing
    reports as 0, and a sigma of many steps, where differencing can only return values
    about half a step apart.

    Each pixel is conditioned on a window of levels around the surface, reaching three
    fitted sigmas beyond it on each side.  The window's end levels are open-ended: a
    pixel on or below the lowest level is known only to lie below that level's upper
    edge, and a pixel on or above the highest level only to lie above its lower edge.
    So every pixel enters the fit, and a hot pixel or cosmic ray far from the noise
    counts only as one pixel beyond the window's edge, however far beyond it lies.  The
    window starts from a least-squares surface and a robust sigma, and is rebuilt from
    each fit's surface and sigma until it stops changing.

    The patch's lowest and highest levels are open-ended as well, so a patch clipped
    at the detector's floor or at saturation, where one level gathers every value
    beyond it, is fitted as an unclipped one is.

    The sigma is only measurable when at least three levels, counting each window's
    end levels as holding the pixels beyond them, each hold at least two pixels.  A
    patch whose noise is so far below one step that its pixels take only two values
    gives one proportion along the surface, and any sigma, with a matching surface,
    gives that proportion; such a patch returns NaN, as does a patch of one value and
    a patch whose fit fails to converge.

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

    coef, sigma = _first_guess(q, design)
    params = np.append(coef, np.clip(np.log(sigma), *log_sigma_bounds))
    used: tuple[NDArrayFloatType, NDArrayFloatType] | None = None
    censored = q
    for _ in range(_MAX_WINDOW_FITS):
        bottom, top = _level_window(_surface(design, params[:-1]), float(np.exp(params[-1])))
        censored = np.clip(q, bottom, top)
        lower_edge = np.where((censored <= bottom) | (q <= 0.0), -np.inf, censored - 0.5)
        upper_edge = np.where((censored >= top) | (q >= q_high), np.inf, censored + 0.5)
        if (
            used is not None
            and np.array_equal(lower_edge, used[0])
            and np.array_equal(upper_edge, used[1])
        ):
            break
        used = (lower_edge, upper_edge)
        fitted = _fit_censored(design, lower_edge, upper_edge, params, log_sigma_bounds)
        if fitted is None:
            return float('nan')
        params = fitted
    if _populated_levels(censored) < _MIN_LEVELS:
        return float('nan')
    return float(np.exp(params[-1]) * step)


def patch_noise_sigma(patch: NDArrayFloatType) -> float:
    """Per-pixel noise sigma of one near-uniform patch, quantized or not.

    A patch whose values sit on a lattice is fitted with :func:`lattice_noise_sigma`;
    any other patch is measured with :func:`paired_difference_sigma`.  Both estimate
    the sigma of the noise before any rounding.  A product that was quantized and
    then rescaled pixel by pixel, such as a flat-fielded frame, no longer sits on a
    lattice, so its estimate also carries the rounding's own spread of step/sqrt(12),
    which adds under 4% to a sigma of one step or more.

    Parameters:
        patch: 2-D array of pixel values (a near-uniform region).

    Returns:
        The sigma in the patch's units; NaN for a degenerate patch, for a patch of
        one value, and for a quantized patch whose sigma is not measurable.
    """
    arr = np.asarray(patch, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.max()) == float(finite.min()):
        return float('nan')
    step = quantization_step(arr)
    if step is not None:
        return lattice_noise_sigma(arr, step)
    return paired_difference_sigma(arr)


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
    zero-filled missing data with a few stray pixels, has a paired-difference sigma
    of 0 and holds no noise to compare.  A tile of one value has no measurable sigma
    and never becomes a patch.

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
