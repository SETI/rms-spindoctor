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
measured by fitting the quantization itself (:func:`lattice_noise_sigma`).  The
paired-difference scale is not valid there: when the noise is below about one step,
more than half the differences are 0, the MAD is 0, and a camera whose noise is
smaller than one step contributes no sky samples at all.  Above one step, the MAD of
whole-step differences is itself a whole or half step, so the estimate can only take
values about 0.5 step apart.

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

# A patch's distinct values lie on a lattice when each one's distance from the
# lowest, in units of the smallest gap between them, is a whole number to within
# this fraction of a step.
_LATTICE_TOLERANCE = 1e-3

# A step at or below this fraction of the largest value's magnitude is the spacing of
# floating-point numbers, not a quantization: the float32 values near 3e-3 I/F are
# 2.3e-10 apart.
_FLOAT_SPACING_FRACTION = 1e-6

# The quantized fit uses the levels within this many robust sigmas of each pixel's
# fitted level (and at least _MIN_HALF_WINDOW_STEPS steps), cut back to the run of
# occupied levels around it.
_WINDOW_SIGMAS = 5.0
_MIN_HALF_WINDOW_STEPS = 3

# The fewest occupied levels from which the quantized fit can tell the noise sigma
# from the level: two levels fix one proportion, which any sigma can match.
_MIN_LEVELS = 3


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

    A lattice counts only when the sample could fill it: its central 98% spans no
    more steps than there are values, and the step is far coarser than the spacing of
    floating-point numbers at the values' magnitude.  Without those conditions every
    float32 frame would pass, since its values are all whole multiples of the float
    spacing.

    Parameters:
        values: Pixel values (any shape; non-finite entries are ignored).

    Returns:
        The smallest gap between distinct values when every distinct value lies a
        whole number of those gaps above the lowest; None for continuous values and
        for fewer than two distinct values.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    levels = np.unique(finite)
    if levels.size < 2:
        return None
    step = float(np.min(np.diff(levels)))
    if step <= _FLOAT_SPACING_FRACTION * float(np.max(np.abs(levels))):
        return None
    low, high = np.quantile(finite, [0.01, 0.99])
    if (high - low) / step > finite.size:
        return None
    in_steps = (levels - levels[0]) / step
    if float(np.max(np.abs(in_steps - np.round(in_steps)))) > _LATTICE_TOLERANCE:
        return None
    return step


def lattice_noise_sigma(patch: NDArrayFloatType, step: float) -> float:
    """Noise sigma of a quantized patch, by fitting the quantization.

    Each pixel is modeled as a level on a plane across the patch plus Gaussian noise
    of sigma ``s``, rounded to the nearest lattice level, so a pixel lands on level
    ``q`` with probability ``Phi((q + step/2 - mu) / s) - Phi((q - step/2 - mu) / s)``,
    where ``mu`` is the plane's level at that pixel.  The plane's tilt comes from a
    least-squares fit, which rounding does not bias, and the plane's level and ``s``
    are fitted by maximum likelihood.  The fit recovers a sigma well below one step,
    which local differencing reports as 0, and it is unbiased above one step, where
    differencing can only return values about half a step apart.

    The fit uses each pixel only when its level lies in a window around the pixel's
    first-guess level: within five robust sigmas (at least three steps), and within
    the run of occupied levels around it, so a hot pixel or cosmic ray separated from
    the noise by an empty level is left out.  The likelihood is conditioned on the
    window, so leaving the tails out does not bias the sigma.

    The sigma is only measurable when the window holds at least three occupied
    levels.  A patch whose noise is so far below one step that its pixels take only
    two values gives one proportion, and any sigma, with a matching level, gives that
    proportion; such a patch returns NaN.

    Parameters:
        patch: 2-D array of pixel values on a lattice of spacing ``step``.
        step: The lattice spacing, as :func:`quantization_step` returns it.

    Returns:
        The fitted sigma in the patch's units; 0 for a patch of one value, and NaN
        when the sigma is not measurable.
    """
    arr = np.asarray(patch, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.ndim != 2 or int(finite.sum()) < 8:
        return float('nan')
    values = arr[finite]
    if float(values.max()) == float(values.min()):
        return 0.0
    vv, uu = np.nonzero(finite)
    v_c = vv - vv.mean()
    u_c = uu - uu.mean()
    q = np.round((values - values.min()) / step)
    median = float(np.median(q))
    robust_sigma = max(_MAD_TO_SIGMA * float(np.median(np.abs(q - median))), 0.5)
    near = np.abs(q - median) <= 3.0 * robust_sigma + 1.0
    design = np.column_stack([np.ones(int(near.sum())), v_c[near], u_c[near]])
    coef = np.linalg.lstsq(design, q[near], rcond=None)[0]
    tilt = coef[1] * v_c + coef[2] * u_c
    center = np.round(coef[0] + tilt)
    offset = (q - center).astype(np.int64)
    half = max(_MIN_HALF_WINDOW_STEPS, int(np.ceil(_WINDOW_SIGMAS * robust_sigma)))
    occupied = set(offset[np.abs(offset) <= half].tolist())
    if not occupied:
        return float('nan')
    start = 0 if 0 in occupied else min(occupied, key=abs)
    low = high = start
    while low - 1 in occupied:
        low -= 1
    while high + 1 in occupied:
        high += 1
    keep = (offset >= low) & (offset <= high)
    q_k = q[keep]
    if np.unique(q_k).size < _MIN_LEVELS:
        return float('nan')
    tilt_k = tilt[keep]
    top = center[keep] + high + 0.5
    bottom = center[keep] + low - 0.5

    def neg_log_likelihood(params: NDArrayFloatType) -> float:
        level, log_sigma = float(params[0]), float(params[1])
        sigma = np.exp(log_sigma)
        mu = level + tilt_k
        p_pixel = scipy.special.ndtr((q_k + 0.5 - mu) / sigma) - scipy.special.ndtr(
            (q_k - 0.5 - mu) / sigma
        )
        p_window = scipy.special.ndtr((top - mu) / sigma) - scipy.special.ndtr(
            (bottom - mu) / sigma
        )
        return -float(
            np.sum(np.log(np.maximum(p_pixel, 1e-300)) - np.log(np.maximum(p_window, 1e-300)))
        )

    first_sigma = max(float(np.std(q_k - center[keep])), 0.3)
    fit = scipy.optimize.minimize(
        neg_log_likelihood,
        np.array([coef[0], np.log(first_sigma)]),
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-6},
    )
    return float(np.exp(fit.x[1]) * step)


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
        The sigma in the patch's units; NaN for a degenerate patch and for a
        quantized one whose sigma is not measurable.
    """
    step = quantization_step(patch)
    if step is not None:
        return lattice_noise_sigma(patch, step)
    return paired_difference_sigma(patch)


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
    would masquerade as noise.  A patch whose sigma is not measurable is
    skipped.

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

    A tile of one constant value, such as zero-filled missing data, passes the
    uniformity test with a sigma of 0 but holds no noise to compare.

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
