"""Image-side measurements on a star field: noise, peaks, photometry, and PSF.

These functions take plain arrays and know nothing about any instrument, holdings layout, or
the navigator.  The per-instrument measurement programs in this package load a frame, query the
star catalog, and call them.  The pointing fix that matches catalog stars to peaks is in
:mod:`util.star_photometry.star_pointing`.

Coordinates are array-index coordinates ``(v, u)``: the center of array element ``[i, j]`` is
``(i, j)``.  The navigator's pixel coordinates put that center at ``(i + 0.5, j + 0.5)``; the
caller converts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.special
from numpy.typing import NDArray

from spindoctor.sim.realism.profiles import ee_radius, encircled_energy, radial_profile

__all__ = [
    'ApertureResult',
    'GaussianFit',
    'PeakList',
    'aperture_photometry',
    'detect_peaks',
    'detection_test',
    'encircled_energy_radii',
    'fit_gaussian_psf',
    'highpass',
    'integrated_gaussian',
    'noise_sigma',
    'robust_sigma',
]

FloatArray = NDArray[np.float64]

#: Ratio of a normal distribution's standard deviation to its median absolute deviation.
MAD_TO_SIGMA = 1.4826


def robust_sigma(values: FloatArray) -> float:
    """Return the standard deviation of a sample estimated from its median absolute deviation.

    Parameters:
        values: The sample; non-finite values are ignored.

    Returns:
        ``1.4826 * median(|x - median(x)|)``, or NaN for an empty sample.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    return float(MAD_TO_SIGMA * np.median(np.abs(finite - np.median(finite))))


#: A sample with fewer distinct values than this fraction of its size is treated as quantized.
QUANTIZED_FRACTION = 0.25

#: Clipping threshold (in standard deviations) of the clipped standard deviation.
CLIP_SIGMA = 3.0

#: Iterations of the clipped standard deviation.
CLIP_ITERATIONS = 5


def noise_sigma(values: FloatArray) -> float:
    """Return the noise of a sample of background pixels, robust to a few bright outliers.

    For a sample of continuous values this is :func:`robust_sigma`.  An 8-bit image whose
    noise is about one DN holds only a handful of distinct values, where the median absolute
    deviation jumps between 0 and 1 DN; for such a quantized sample (fewer distinct values than
    ``QUANTIZED_FRACTION`` of its size) the noise is instead the standard deviation after
    iteratively clipping values more than ``CLIP_SIGMA`` standard deviations from the mean,
    but never less than the quantization noise, the quantization step over the square root of
    12 (a step of one unit when the sample holds a single value).

    Parameters:
        values: The sample; non-finite values are ignored.

    Returns:
        The noise, or NaN for an empty sample.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    if np.unique(finite).size >= QUANTIZED_FRACTION * finite.size:
        return robust_sigma(finite)
    kept = finite
    for _ in range(CLIP_ITERATIONS):
        mean, std = float(kept.mean()), float(kept.std())
        inside = np.abs(finite - mean) <= CLIP_SIGMA * std
        if std == 0.0 or inside.sum() == kept.size:
            break
        kept = finite[inside]
    levels = np.unique(finite)
    step = float(np.min(np.diff(levels))) if levels.size > 1 else 1.0
    return max(float(kept.std()), step / math.sqrt(12.0))


def highpass(image: FloatArray, size: int, *, subsample: int = 1) -> FloatArray:
    """Return the image minus its running median, which removes the sky and its gradients.

    Parameters:
        image: The image.
        size: The side of the square median window in pixels; it should span several star
            widths so a star does not lift its own median.
        subsample: When above 1, the median is taken on every ``subsample``-th pixel of each
            axis with a window of ``size // subsample`` (rounded up to odd) and expanded back
            by repetition, which is much faster on a large image and changes the smooth sky
            estimate very little.

    Returns:
        The high-passed image as float64.
    """
    data = np.asarray(image, dtype=np.float64)
    if subsample <= 1:
        background = np.asarray(
            scipy.ndimage.median_filter(data, size=size, mode='nearest'), dtype=np.float64
        )
        return data - background
    small_size = max(3, (size // subsample) | 1)
    small = scipy.ndimage.median_filter(
        data[::subsample, ::subsample], size=small_size, mode='nearest'
    )
    expanded = np.repeat(np.repeat(small, subsample, axis=0), subsample, axis=1)
    return data - np.asarray(expanded[: data.shape[0], : data.shape[1]], dtype=np.float64)


@dataclass(frozen=True)
class PeakList:
    """Point sources found in a high-passed image, brightest first.

    Attributes:
        v: Centroid line of each peak (array-index coordinates).
        u: Centroid sample of each peak.
        snr: Peak height of the smoothed image over the smoothed image's noise.
    """

    v: FloatArray
    u: FloatArray
    snr: FloatArray


def detect_peaks(
    hp: FloatArray,
    *,
    smooth_sigma: float,
    threshold_snr: float,
    edge: int,
    max_peaks: int,
    mask: NDArray[np.bool_] | None = None,
    max_per_line: int = 4,
    max_elongation: float = 2.0,
    max_center_fraction: float = 0.7,
) -> PeakList:
    """Find point sources in a high-passed image.

    The image is smoothed with a Gaussian of ``smooth_sigma`` (a matched filter for a star of
    about that width, which also suppresses one-pixel cosmic-ray hits), and every local maximum
    of the smoothed image more than ``threshold_snr`` times its robust noise above zero is a
    peak.  Each peak is then centroided on the unsmoothed image, weighting by the positive
    part of the image within ``max(2, 2 * smooth_sigma)`` pixels.

    Parameters:
        hp: The high-passed image.
        smooth_sigma: The smoothing Gaussian's sigma in pixels.
        threshold_snr: The detection threshold in units of the smoothed image's noise.
        edge: Peaks closer than this many pixels to any border are ignored.
        max_peaks: At most this many peaks are returned, the highest first.
        mask: Optional boolean image; peaks on True pixels are ignored.
        max_per_line: Peaks on an image line or column holding more than this many peaks
            are dropped: they trace a defective line, not stars.
        max_elongation: Peaks whose positive signal within the centroid box has a
            major-to-minor axis ratio (from second moments) above this are dropped as
            particle tracks or bleed trails.
        max_center_fraction: Peaks whose brightest pixel holds more than this fraction of
            the positive signal of the 3 x 3 box about it are dropped as one-pixel hits.

    Returns:
        The peaks, brightest first.
    """
    smooth = scipy.ndimage.gaussian_filter(hp, smooth_sigma, mode='nearest')
    interior = smooth[edge:-edge, edge:-edge].ravel()
    noise = robust_sigma(interior)
    if not noise > 0.0:
        # A heavily quantized image can have a zero median absolute deviation.
        noise = float(np.std(interior))
    radius = max(2, round(2 * smooth_sigma))
    local_max = scipy.ndimage.maximum_filter(smooth, size=2 * radius + 1, mode='nearest')
    is_peak = (smooth == local_max) & (smooth > threshold_snr * noise)
    is_peak[:edge, :] = False
    is_peak[-edge:, :] = False
    is_peak[:, :edge] = False
    is_peak[:, -edge:] = False
    if mask is not None:
        is_peak &= ~mask
    pv, pu = np.nonzero(is_peak)
    line_count = np.bincount(pv, minlength=hp.shape[0])
    column_count = np.bincount(pu, minlength=hp.shape[1])
    on_streak = (line_count[pv] > max_per_line) | (column_count[pu] > max_per_line)
    pv, pu = pv[~on_streak], pu[~on_streak]
    order = np.argsort(-smooth[pv, pu])[:max_peaks]
    pv, pu = pv[order], pu[order]
    keep = np.array(
        [
            _is_point_like(
                hp,
                int(iv),
                int(iu),
                radius,
                max_elongation=max_elongation,
                max_center_fraction=max_center_fraction,
            )
            for iv, iu in zip(pv, pu, strict=True)
        ],
        dtype=bool,
    )
    pv, pu = pv[keep], pu[keep]
    cv = np.empty(pv.size)
    cu = np.empty(pu.size)
    for k, (iv, iu) in enumerate(zip(pv, pu, strict=True)):
        cv[k], cu[k] = _centroid(hp, float(iv), float(iu), radius)
    return PeakList(v=cv, u=cu, snr=smooth[pv, pu] / noise)


def _is_point_like(
    hp: FloatArray,
    iv: int,
    iu: int,
    radius: int,
    *,
    max_elongation: float,
    max_center_fraction: float,
) -> bool:
    """Return whether the peak at ``(iv, iu)`` looks like a star rather than a hit or a track.

    Parameters:
        hp: The high-passed image.
        iv: Peak line (a smoothed-image maximum).
        iu: Peak sample.
        radius: Half-size of the moment box.
        max_elongation: Largest major-to-minor axis ratio of the box's positive signal.
        max_center_fraction: Largest fraction of the 3 x 3 box's positive signal its
            brightest pixel may hold.

    Returns:
        True for a point-like peak.
    """
    v0, v1 = max(iv - radius, 0), min(iv + radius + 1, hp.shape[0])
    u0, u1 = max(iu - radius, 0), min(iu + radius + 1, hp.shape[1])
    box = np.clip(hp[v0:v1, u0:u1], 0.0, None)
    total = float(box.sum())
    if total <= 0.0:
        return False
    gv, gu = np.mgrid[v0:v1, u0:u1]
    mv, mu = (box * gv).sum() / total, (box * gu).sum() / total
    cvv = (box * (gv - mv) ** 2).sum() / total
    cuu = (box * (gu - mu) ** 2).sum() / total
    cvu = (box * (gv - mv) * (gu - mu)).sum() / total
    half_trace, det = 0.5 * (cvv + cuu), cvv * cuu - cvu**2
    root = math.sqrt(max(half_trace**2 - det, 0.0))
    major, minor = half_trace + root, half_trace - root
    if minor <= 0.0 or math.sqrt(major / minor) > max_elongation:
        return False
    w0, w1 = max(iv - 1, 0), min(iv + 2, hp.shape[0])
    x0, x1 = max(iu - 1, 0), min(iu + 2, hp.shape[1])
    small = np.clip(hp[w0:w1, x0:x1], 0.0, None)
    return float(small.max()) <= max_center_fraction * float(small.sum())


def _centroid(image: FloatArray, v: float, u: float, radius: int) -> tuple[float, float]:
    """Return the positive-weighted centroid of ``image`` in a box about ``(v, u)``.

    Parameters:
        image: The (high-passed) image.
        v: Starting line.
        u: Starting sample.
        radius: Half-size of the box in pixels.

    Returns:
        The centroid ``(v, u)``, or the starting point when nothing in the box is positive.
    """
    iv, iu = round(v), round(u)
    v0, v1 = max(iv - radius, 0), min(iv + radius + 1, image.shape[0])
    u0, u1 = max(iu - radius, 0), min(iu + radius + 1, image.shape[1])
    box = np.clip(image[v0:v1, u0:u1], 0.0, None)
    total = float(box.sum())
    if total <= 0.0:
        return v, u
    gv, gu = np.mgrid[v0:v1, u0:u1]
    return float((box * gv).sum() / total), float((box * gu).sum() / total)


@dataclass(frozen=True)
class ApertureResult:
    """Aperture photometry of one source.

    Attributes:
        flux: Background-subtracted sum inside the aperture.
        sky: Median of the sky annulus, per pixel.
        sky_sigma: Noise of the sky annulus per pixel (see :func:`noise_sigma`).
        flux_sigma: One-sigma uncertainty of ``flux`` from the sky noise alone.
        max_value: Largest raw pixel value inside the aperture.
        min_value: Smallest raw pixel value inside the aperture or the sky annulus.
        n_pixels: Pixels inside the aperture.
    """

    flux: float
    sky: float
    sky_sigma: float
    flux_sigma: float
    max_value: float
    min_value: float
    n_pixels: int


def aperture_photometry(
    image: FloatArray,
    v: float,
    u: float,
    *,
    radius: float,
    sky_inner: float,
    sky_outer: float,
) -> ApertureResult | None:
    """Sum an image inside a circular aperture, less the median of a surrounding annulus.

    A pixel is inside a circle when its center is.

    Parameters:
        image: The image.
        v: Aperture center line (array-index coordinates).
        u: Aperture center sample.
        radius: Aperture radius in pixels.
        sky_inner: Inner radius of the sky annulus.
        sky_outer: Outer radius of the sky annulus.

    Returns:
        The photometry, or None when the annulus leaves the image or the center is not
        finite.
    """
    if not (math.isfinite(v) and math.isfinite(u)):
        return None
    half = math.ceil(sky_outer) + 1
    iv, iu = round(v), round(u)
    if iv - half < 0 or iu - half < 0:
        return None
    if iv + half + 1 > image.shape[0] or iu + half + 1 > image.shape[1]:
        return None
    cut = np.asarray(image[iv - half : iv + half + 1, iu - half : iu + half + 1], dtype=float)
    gv, gu = np.mgrid[iv - half : iv + half + 1, iu - half : iu + half + 1]
    r = np.hypot(gv - v, gu - u)
    inside = r <= radius
    ring = (r >= sky_inner) & (r <= sky_outer)
    sky = float(np.median(cut[ring]))
    sky_sigma = noise_sigma(cut[ring])
    n_in = int(inside.sum())
    flux = float((cut[inside] - sky).sum())
    flux_sigma = sky_sigma * math.sqrt(n_in * (1.0 + math.pi / 2.0 * n_in / int(ring.sum())))
    return ApertureResult(
        flux=flux,
        sky=sky,
        sky_sigma=sky_sigma,
        flux_sigma=flux_sigma,
        max_value=float(cut[inside].max()),
        min_value=float(cut[inside | ring].min()),
        n_pixels=n_in,
    )


def detection_test(
    hp: FloatArray,
    v: float,
    u: float,
    *,
    search_px: float,
    noise_inner: float,
    noise_outer: float,
    box_half: int,
) -> tuple[float, int, float, float] | None:
    """Measure the detection statistics of a star expected at ``(v, u)`` in a high-passed image.

    The peak is the brightest pixel within ``search_px`` of the expected position.  The noise
    is :func:`noise_sigma` of the annulus ``noise_inner``-``noise_outer`` about that peak.
    The count is the number of pixels in the ``(2 * box_half + 1)``-square box about the peak
    above three times the noise.

    Parameters:
        hp: The high-passed image.
        v: Expected line (array-index coordinates).
        u: Expected sample.
        search_px: Peak search radius (pixels).
        noise_inner: Inner radius of the noise annulus.
        noise_outer: Outer radius of the noise annulus.
        box_half: Half-size of the counting box.

    Returns:
        ``(peak_snr, n_above_3_sigma, peak_v, peak_u)``, or None near an edge.
    """
    half = math.ceil(noise_outer) + math.ceil(search_px) + 1
    iv, iu = round(v), round(u)
    if iv - half < 0 or iu - half < 0:
        return None
    if iv + half + 1 > hp.shape[0] or iu + half + 1 > hp.shape[1]:
        return None
    s = math.ceil(search_px)
    gv, gu = np.mgrid[iv - s : iv + s + 1, iu - s : iu + s + 1]
    win = hp[iv - s : iv + s + 1, iu - s : iu + s + 1]
    within = np.hypot(gv - v, gu - u) <= search_px
    masked = np.where(within, win, -np.inf)
    k = int(np.argmax(masked))
    pv, pu = int(gv.ravel()[k]), int(gu.ravel()[k])
    o = math.ceil(noise_outer)
    nv, nu = np.mgrid[pv - o : pv + o + 1, pu - o : pu + o + 1]
    r = np.hypot(nv - pv, nu - pu)
    ring = hp[pv - o : pv + o + 1, pu - o : pu + o + 1][(r >= noise_inner) & (r <= noise_outer)]
    noise = noise_sigma(ring)
    if not noise > 0.0:
        return None
    box = hp[pv - box_half : pv + box_half + 1, pu - box_half : pu + box_half + 1]
    return float(hp[pv, pu] / noise), int((box > 3.0 * noise).sum()), float(pv), float(pu)


@dataclass(frozen=True)
class GaussianFit:
    """A pixel-integrated Gaussian fitted to a star cutout.

    Attributes:
        sigma: The circular Gaussian's sigma (pixels).
        sigma_major: Sigma along the elliptical fit's major axis (pixels).
        sigma_minor: Sigma along the minor axis (pixels).
        angle_deg: Position angle of the major axis (degrees from +u toward +v).
        center_v: Fitted center line (array-index coordinates).
        center_u: Fitted center sample.
        amplitude: Total flux of the circular fit.
        rms_residual: RMS of the circular fit's residual over its peak value.
    """

    sigma: float
    sigma_major: float
    sigma_minor: float
    angle_deg: float
    center_v: float
    center_u: float
    amplitude: float
    rms_residual: float


def integrated_gaussian(
    gv: FloatArray,
    gu: FloatArray,
    center_v: float,
    center_u: float,
    sigma_v: float,
    sigma_u: float,
) -> FloatArray:
    """Return a unit-flux axis-aligned Gaussian integrated over each pixel.

    Parameters:
        gv: Pixel-center lines.
        gu: Pixel-center samples.
        center_v: Gaussian center line.
        center_u: Gaussian center sample.
        sigma_v: Sigma along v.
        sigma_u: Sigma along u.

    Returns:
        The fraction of the flux falling in each pixel.
    """
    sv = math.sqrt(2.0) * sigma_v
    su = math.sqrt(2.0) * sigma_u
    fv = 0.5 * (
        scipy.special.erf((gv + 0.5 - center_v) / sv)
        - scipy.special.erf((gv - 0.5 - center_v) / sv)
    )
    fu = 0.5 * (
        scipy.special.erf((gu + 0.5 - center_u) / su)
        - scipy.special.erf((gu - 0.5 - center_u) / su)
    )
    return np.asarray(fv * fu, dtype=np.float64)


def _rotated_gaussian(
    gv: FloatArray,
    gu: FloatArray,
    center_v: float,
    center_u: float,
    sigma_a: float,
    sigma_b: float,
    angle: float,
) -> FloatArray:
    """Return a unit-flux rotated Gaussian sampled on a 5 x 5 sub-grid in each pixel.

    Parameters:
        gv: Pixel-center lines.
        gu: Pixel-center samples.
        center_v: Center line.
        center_u: Center sample.
        sigma_a: Sigma along the axis at ``angle`` from +u.
        sigma_b: Sigma along the perpendicular axis.
        angle: Axis angle in radians.

    Returns:
        The fraction of the flux falling in each pixel.
    """
    offsets = (np.arange(5) - 2) / 5.0
    total = np.zeros_like(gv, dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    for ov in offsets:
        for ou in offsets:
            y = gv + ov - center_v
            x = gu + ou - center_u
            a = x * c + y * s
            b = -x * s + y * c
            total += np.exp(-0.5 * ((a / sigma_a) ** 2 + (b / sigma_b) ** 2))
    return total / (25.0 * 2.0 * math.pi * sigma_a * sigma_b)


def fit_gaussian_psf(image: FloatArray, v: float, u: float, *, half: int) -> GaussianFit | None:
    """Fit pixel-integrated Gaussians to the star near ``(v, u)``.

    The cutout is ``(2 * half + 1)`` pixels square about the rounded position.  Two models are
    fitted with a constant background: a circular pixel-integrated Gaussian (the navigator's
    PSF model) and an elliptical Gaussian, whose axes measure asymmetry.

    Parameters:
        image: The image.
        v: Approximate center line (array-index coordinates).
        u: Approximate center sample.
        half: Half-size of the cutout.

    Returns:
        The fit, or None near an edge or when either fit fails to converge.
    """
    iv, iu = round(v), round(u)
    if iv - half < 0 or iu - half < 0:
        return None
    if iv + half + 1 > image.shape[0] or iu + half + 1 > image.shape[1]:
        return None
    cut = np.asarray(image[iv - half : iv + half + 1, iu - half : iu + half + 1], dtype=float)
    gv, gu = np.mgrid[iv - half : iv + half + 1, iu - half : iu + half + 1].astype(float)
    edge = np.concatenate([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]])
    bkg0 = float(np.median(edge))
    flux0 = float(max((cut - bkg0).sum(), 1e-9))

    def circular(p: FloatArray) -> FloatArray:
        model = p[0] * integrated_gaussian(gv, gu, p[1], p[2], p[3], p[3]) + p[4]
        return np.asarray((model - cut).ravel(), dtype=np.float64)

    p0 = np.array([flux0, v, u, 1.0, bkg0])
    lower = [0.0, v - 3.0, u - 3.0, 0.15, -np.inf]
    upper = [np.inf, v + 3.0, u + 3.0, float(half), np.inf]
    res = scipy.optimize.least_squares(circular, p0, bounds=(lower, upper))
    if not res.success:
        return None
    pc = res.x
    peak = float(np.max(cut - pc[4]))
    rms = float(np.sqrt(np.mean(res.fun**2)) / peak) if peak > 0 else float('nan')

    def elliptical(p: FloatArray) -> FloatArray:
        model = p[0] * _rotated_gaussian(gv, gu, p[1], p[2], p[3], p[4], p[5]) + p[6]
        return np.asarray((model - cut).ravel(), dtype=np.float64)

    qlo = np.array([0.0, v - 3.0, u - 3.0, 0.15, 0.15, -np.pi, -np.inf])
    qhi = np.array([np.inf, v + 3.0, u + 3.0, float(half), float(half), np.pi, np.inf])
    q0 = np.array([pc[0], pc[1], pc[2], pc[3] * 1.1, pc[3] * 0.9, 0.3, pc[4]])
    # Keep the starting point strictly inside the bounds the circular fit may have reached.
    span = qhi - qlo
    q0 = np.clip(
        q0,
        qlo + 1e-6 * np.where(np.isfinite(span), span, 1.0),
        qhi - 1e-6 * np.where(np.isfinite(span), span, 1.0),
    )
    res2 = scipy.optimize.least_squares(elliptical, q0, bounds=(qlo, qhi))
    if not res2.success:
        return None
    sa, sb, ang = float(res2.x[3]), float(res2.x[4]), float(res2.x[5])
    if sb > sa:
        sa, sb, ang = sb, sa, ang + math.pi / 2.0
    ang_deg = (math.degrees(ang) + 90.0) % 180.0 - 90.0
    return GaussianFit(
        sigma=float(pc[3]),
        sigma_major=sa,
        sigma_minor=sb,
        angle_deg=ang_deg,
        center_v=float(pc[1]),
        center_u=float(pc[2]),
        amplitude=float(pc[0]),
        rms_residual=rms,
    )


def encircled_energy_radii(
    image: FloatArray,
    v: float,
    u: float,
    *,
    r_max: float,
    n_bins: int,
) -> tuple[float, float]:
    """Return the radii holding 50% and 80% of a star's light inside ``r_max``.

    The estimator is the one the simulator-realism comparison used on Cassini (an azimuthally
    averaged profile with the median of the annulus ``r_max``-``r_max + 3`` subtracted,
    accumulated as ``intensity * 2 pi r dr`` and normalized at ``r_max``).

    Parameters:
        image: The image.
        v: Star center line (array-index coordinates).
        u: Star center sample.
        r_max: Outer radius of the profile (pixels).
        n_bins: Radial bins between 0 and ``r_max``.

    Returns:
        ``(ee50, ee80)`` in pixels; NaN where the curve does not reach the fraction.
    """
    radius, intensity = radial_profile(image, (v, u), r_max=r_max, n_bins=n_bins)
    r, ee = encircled_energy(radius, intensity)
    return ee_radius(r, ee, 0.5), ee_radius(r, ee, 0.8)
