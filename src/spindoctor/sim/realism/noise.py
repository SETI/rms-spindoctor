"""Sky-region noise statistics for realism FOM 1.

Science frames have no flat-field pairs, so noise is estimated by local
differencing inside near-uniform patches: paired horizontal pixel
differences cancel scene structure that varies slowly across the patch,
and dividing the difference scale by sqrt(2) recovers the per-pixel sigma.
A robust (MAD-based) scale keeps residual stars, cosmic rays, and hot
pixels in a patch from inflating the estimate; naive signal-binning would
conflate scene texture with noise, which is exactly what this estimator
avoids.

The sky spatial power spectrum (radially averaged over annuli of spatial
frequency) catches banding and coherent noise that a scalar sigma cannot:
white read noise is flat, banding is a comb.
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'SkyPatch',
    'dark_sky_patches',
    'find_uniform_patches',
    'noisy_patches',
    'paired_difference_sigma',
    'radial_power_spectrum',
]

# MAD to sigma conversion for a normal distribution.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SkyPatch:
    """One near-uniform patch selected from a frame.

    Parameters:
        v0: Top row of the patch (inclusive).
        u0: Left column of the patch (inclusive).
        size: Patch edge length in pixels.
        mean: Mean signal inside the patch.
        sigma: Paired-difference noise sigma inside the patch.
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
    statistic), and (b) its internal structure is difference-dominated: the
    patch's total standard deviation does not exceed ``max_structure_ratio``
    times its paired-difference sigma.  Condition (b) rejects patches
    crossed by limbs, rings, or bright stars, whose spatial structure would
    masquerade as noise.

    Parameters:
        image: 2-D frame in its native units.
        patch_size: Edge length of the square tiles.
        max_mean_quantile: Patch-mean quantile at or below which a patch
            counts as sky, or None to skip the sky cut.
        max_structure_ratio: Maximum allowed ratio of total patch standard
            deviation to paired-difference sigma.

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
            sigma = paired_difference_sigma(tile)
            if not np.isfinite(sigma):
                continue
            mean = float(tile.mean())
            total_sd = float(tile.std())
            if sigma > 0.0 and total_sd > max_structure_ratio * sigma:
                continue
            if sigma == 0.0 and total_sd > 0.0:
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
