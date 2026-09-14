"""DAOPHOT-style star detection helpers.

The orchestrator never calls this module — predicted-from-catalog star
features are emitted by ``NavModelStars.to_features`` directly.  The
detector exists so a downstream technique (``StarRefineNav`` etc.) can
sweep the image for centroidable stars when the catalog match is
ambiguous; keeping it in the same package as the catalog reduction
guarantees the matched filter, the smear-aware kernel, and the
shape-based cuts stay in sync.

The pipeline is the canonical DAOPHOT sequence in three stages:

1. **Matched filter.**  Cross-correlate the image with the smeared PSF
   kernel.  The peak amplitude at each pixel is the maximum-likelihood
   estimate of an unsmoothed star signal at that location.
2. **Local maxima.**  Find every pixel that is the local maximum in a
   window matched to the PSF support and is above the per-image
   detection threshold (``k * image_noise_sigma`` after matched
   filtering).
3. **Centroid + cuts.**  Fit a Gaussian to a small box around each
   maximum.  Stars hitting the saturation DN switch to an annular
   moment because the Gaussian fit blows up.  Hot pixels and
   asymmetric stars are rejected via classic DAOPHOT sharpness /
   roundness criteria.

CCD bloom columns are detected up-front so saturated bright stars are
not double-counted as multiple detections along the bloom trail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import correlate, maximum_filter

from spindoctor.nav_model.stars.predicted_snr import psf_sigma_px

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

    from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DAOPHOT_DEFAULT_DETECTION_SIGMA',
    'DAOPHOT_DEFAULT_ROUNDNESS_BOUND',
    'DAOPHOT_DEFAULT_SHARPNESS_MAX',
    'DAOPHOT_DEFAULT_SHARPNESS_MIN',
    'DetectedSource',
    'apply_shape_cuts',
    'centroid_gaussian_fit',
    'centroid_saturated',
    'detect_ccd_bloom_columns',
    'detect_sources',
    'matched_filter_image',
]


DAOPHOT_DEFAULT_DETECTION_SIGMA: float = 4.0
"""Threshold (in image_noise_sigma) for matched-filter peaks.

Matches the DAOPHOT convention of "minimum sigma above sky for a real
source" before shape-based cuts.  Below 4 sigma the cosmic-ray-driven
false-positive rate dominates real detections.
"""


DAOPHOT_DEFAULT_SHARPNESS_MIN: float = 0.2
"""Minimum DAOPHOT sharpness for a real star.

Sharpness < 0.2 is dominated by single-pixel hot spikes; the wing
contribution is too small to be a star.
"""


DAOPHOT_DEFAULT_SHARPNESS_MAX: float = 1.0
"""Maximum DAOPHOT sharpness for a real star.

Sharpness > 1.0 indicates an extended source (galaxy / blended pair)
whose central pixel does not dominate.
"""


DAOPHOT_DEFAULT_ROUNDNESS_BOUND: float = 1.0
r"""Maximum \|roundness\| for a real star.

Computed as the per-axis Gaussian-marginal asymmetry; \> 1 in absolute
value points at a CCD bloom or one-axis trail rather than a smear-
oriented PSF.
"""


@dataclass(frozen=True)
class DetectedSource:
    """One source returned by ``detect_sources``.

    Parameters:
        v: Sub-pixel V (row) centroid.
        u: Sub-pixel U (column) centroid.
        peak_dn: Matched-filter peak amplitude at the centroid.
        sharpness: DAOPHOT sharpness statistic.
        roundness: DAOPHOT roundness statistic (signed).
        saturated: True when the central pixel hit the saturation DN.
    """

    v: float
    u: float
    peak_dn: float
    sharpness: float
    roundness: float
    saturated: bool


def matched_filter_image(image: NDArrayFloatType, *, kernel: NDArrayFloatType) -> NDArrayFloatType:
    """Return the matched-filter response of ``image`` against ``kernel``.

    Implements the standard DAOPHOT matched filter: subtract the
    kernel mean, normalize to unit-energy, and convolve.  The peak
    amplitude at each pixel is then the linear-least-squares estimate
    of the signal scale at that location.

    Parameters:
        image: 2-D float input array.
        kernel: PSF stamp produced by ``psf.eval_rect`` (smear-aware
            when smear is non-trivial).

    Returns:
        Matched-filter response array of the same shape as ``image``.

    Raises:
        ValueError: If ``kernel`` has zero norm (e.g. an all-zero
            stamp).
    """
    if image.ndim != 2:
        raise TypeError(f'matched_filter_image requires a 2-D image; got ndim={image.ndim}')
    if kernel.ndim != 2:
        raise TypeError(f'matched_filter_image kernel must be 2-D; got ndim={kernel.ndim}')
    k = kernel.astype(np.float64)
    k_mean = float(np.mean(k))
    k = k - k_mean
    norm_sq = float(np.sum(k * k))
    if norm_sq <= 0.0:
        raise ValueError('matched_filter_image kernel has zero energy after mean subtraction')
    k = k / norm_sq
    response: NDArrayFloatType = correlate(image.astype(np.float64), k, mode='reflect')
    return response


def detect_ccd_bloom_columns(
    image: NDArrayFloatType, *, full_well_dn: float, min_run: int = 5
) -> NDArrayBoolType:
    """Mark CCD bloom columns where saturation runs vertically.

    A bloom column is detected when at least ``min_run`` consecutive
    saturated pixels appear in any single column.  The whole column is
    marked, not just the saturated stretch, so single-pixel detections
    on the bloom trail are correctly suppressed.

    Parameters:
        image: 2-D float input array.
        full_well_dn: Per-instrument full-well DN.
        min_run: Minimum consecutive saturated pixels to declare a
            bloom column.

    Returns:
        Boolean mask of the same shape as ``image`` with True over the
        identified bloom columns.

    Raises:
        ValueError: If ``min_run`` is < 2.
    """
    if min_run < 2:
        raise ValueError(f'min_run must be >= 2; got {min_run}')
    saturated = image >= full_well_dn
    rows, cols = image.shape
    mask = np.zeros_like(image, dtype=bool)
    for col in range(cols):
        run = 0
        max_run = 0
        for row in range(rows):
            if saturated[row, col]:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        if max_run >= min_run:
            mask[:, col] = True
    return mask


def _local_max_mask(response: NDArrayFloatType, *, size: int, threshold: float) -> NDArrayBoolType:
    """Return a boolean mask True at pixels that are local maxima > threshold."""
    local_max = maximum_filter(response, size=size, mode='reflect')
    out: NDArrayBoolType = (response == local_max) & (response > threshold)
    return out


def centroid_gaussian_fit(
    box: NDArrayFloatType,
) -> tuple[float, float]:
    """Fit a 2-D Gaussian centroid to a small detection box.

    Uses the standard DAOPHOT moment-form: the centroid is
    ``sum(coord * (box - bg)) / sum(box - bg)`` over pixels above the
    box-median background.  The box is small enough (3-5 px on a side)
    that this matches the iterative least-squares centroid to
    sub-pixel agreement.

    Parameters:
        box: Square detection box, ``(2N+1, 2N+1)``.

    Returns:
        ``(dv, du)`` offset in pixels from the center of ``box``.
    """
    if box.ndim != 2 or box.shape[0] != box.shape[1] or box.shape[0] % 2 == 0:
        raise ValueError(f'centroid box must be square odd; got shape {box.shape}')
    n = (box.shape[0] - 1) // 2
    bg = float(np.median(box))
    sub = box - bg
    sub = np.clip(sub, 0.0, None)
    total = float(np.sum(sub))
    if total <= 0.0:
        return 0.0, 0.0
    coords = np.arange(-n, n + 1, dtype=np.float64)
    cv = float(np.sum(coords[:, None] * sub) / total)
    cu = float(np.sum(coords[None, :] * sub) / total)
    return cv, cu


def centroid_saturated(
    box: NDArrayFloatType,
    *,
    full_well_dn: float,
    half_width_inner: int,
    half_width_outer: int,
) -> tuple[float, float]:
    """Return a saturated-star centroid via an annular brightness moment.

    Saturated cores have wrong DN values; the only reliable centroid
    information is in the annulus around the saturated core where the
    PSF wings are still linear.  This routine computes the
    brightness-weighted moment of pixels whose DN is below
    ``full_well_dn`` and whose distance from the box center lies in
    ``[half_width_inner, half_width_outer]``.

    Parameters:
        box: Square detection box, ``(2N+1, 2N+1)``.
        full_well_dn: Saturation DN; pixels at this value are excluded
            from the moment.
        half_width_inner: Inner radius of the annulus (in pixels).
        half_width_outer: Outer radius of the annulus (in pixels).

    Returns:
        ``(dv, du)`` offset in pixels from the center of ``box``.
    """
    if half_width_inner < 0 or half_width_outer <= half_width_inner:
        raise ValueError(
            'expected 0 <= half_width_inner < half_width_outer; got '
            f'inner={half_width_inner}, outer={half_width_outer}'
        )
    if box.ndim != 2 or box.shape[0] != box.shape[1] or box.shape[0] % 2 == 0:
        raise ValueError(f'centroid box must be square odd; got shape {box.shape}')
    n = (box.shape[0] - 1) // 2
    coords = np.arange(-n, n + 1, dtype=np.float64)
    vv, uu = np.meshgrid(coords, coords, indexing='ij')
    radius = np.sqrt(vv * vv + uu * uu)
    valid = (radius >= half_width_inner) & (radius <= half_width_outer)
    valid &= box < full_well_dn
    weights = np.where(valid, np.clip(box - float(np.median(box)), 0.0, None), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0, 0.0
    return (
        float(np.sum(vv * weights) / total),
        float(np.sum(uu * weights) / total),
    )


def _sharpness_roundness(
    box: NDArrayFloatType,
) -> tuple[float, float]:
    """Return DAOPHOT sharpness and roundness for a small detection box.

    ``sharpness = (peak - mean(neighbors)) / peak`` measures how much
    the central pixel dominates its neighbors.  ``roundness`` is the
    signed fractional difference between the box's two marginal
    Gaussian widths; it points to bloom or one-axis trails when its
    absolute value exceeds ``DAOPHOT_DEFAULT_ROUNDNESS_BOUND``.

    Parameters:
        box: Square detection box.

    Returns:
        ``(sharpness, roundness)`` pair.
    """
    if box.ndim != 2 or box.shape[0] != box.shape[1] or box.shape[0] % 2 == 0:
        raise ValueError(f'shape-cut box must be square odd; got shape {box.shape}')
    n = (box.shape[0] - 1) // 2
    center = float(box[n, n])
    if center <= 0.0:
        return 0.0, 0.0
    neighbour_sum = float(np.sum(box) - center)
    neighbour_count = box.size - 1
    sharp = (center - neighbour_sum / neighbour_count) / center
    col_marginal = np.sum(box, axis=0)
    row_marginal = np.sum(box, axis=1)
    col_var = _marginal_variance(col_marginal)
    row_var = _marginal_variance(row_marginal)
    if col_var + row_var == 0.0:
        return float(sharp), 0.0
    round_ = 2.0 * (col_var - row_var) / (col_var + row_var)
    return float(sharp), float(round_)


def _marginal_variance(profile: NDArrayFloatType) -> float:
    """Return the variance of a 1-D marginal profile around its center."""
    profile = np.asarray(profile, dtype=np.float64)
    n = (profile.size - 1) // 2
    coords = np.arange(-n, n + 1, dtype=np.float64)
    bg = float(np.median(profile))
    weights = np.clip(profile - bg, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(coords * weights) / total)
    return float(np.sum((coords - mean) ** 2 * weights) / total)


def apply_shape_cuts(
    sharpness: float,
    roundness: float,
    *,
    sharp_min: float = DAOPHOT_DEFAULT_SHARPNESS_MIN,
    sharp_max: float = DAOPHOT_DEFAULT_SHARPNESS_MAX,
    round_bound: float = DAOPHOT_DEFAULT_ROUNDNESS_BOUND,
) -> bool:
    """Return True if a detection passes the DAOPHOT shape cuts.

    Parameters:
        sharpness: DAOPHOT sharpness from ``_sharpness_roundness``.
        roundness: DAOPHOT roundness from ``_sharpness_roundness``.
        sharp_min, sharp_max: Acceptance window for sharpness.
        round_bound: Maximum absolute roundness.

    Returns:
        True when the detection is keeper-quality.
    """
    if not (sharp_min <= sharpness <= sharp_max):
        return False
    return abs(roundness) <= round_bound


def detect_sources(
    image: NDArrayFloatType,
    *,
    psf: PSF,
    image_noise_sigma: float,
    full_well_dn: float,
    smear_kernel: NDArrayFloatType,
    bloom_mask: NDArrayBoolType | None = None,
    detection_sigma: float = DAOPHOT_DEFAULT_DETECTION_SIGMA,
) -> list[DetectedSource]:
    """Detect candidate stars in ``image`` and return one entry per surviving source.

    Three-stage pipeline:

    1. Matched-filter the image against ``smear_kernel``.
    2. Find local maxima above ``detection_sigma * image_noise_sigma``.
    3. For each maximum, fit a Gaussian centroid (or annular moment when
       saturated), compute DAOPHOT sharpness/roundness, and pass the
       result through ``apply_shape_cuts``.

    Parameters:
        image: 2-D float input array.
        psf: PSF used for the matched-filter shape window.  Per-pixel
            sigma is obtained via
            :func:`spindoctor.nav_model.stars.predicted_snr.psf_sigma_px`,
            which reads ``sigma_x`` / ``sigma_y`` from
            ``psfmodel.GaussianPSF`` (averaged for anisotropic PSFs)
            and falls back to a single ``sigma`` attribute or
            ``fwhm() / 2.3548`` for third-party PSF subclasses.  The
            returned sigma is in pixels.  The smear is already baked
            into ``smear_kernel``; this PSF parameter only sets the
            centroid-fit box half-width.
        image_noise_sigma: Robust per-pixel noise sigma in DN.
        full_well_dn: Saturation DN.
        smear_kernel: Pre-rendered smeared PSF kernel matched to the
            current obs's smear vector.
        bloom_mask: Optional CCD bloom column mask; when provided,
            detections falling on bloom columns are suppressed.
        detection_sigma: Threshold multiplier on
            ``image_noise_sigma``.

    Returns:
        List of ``DetectedSource`` records, one per surviving detection.
    """
    response = matched_filter_image(image, kernel=smear_kernel)
    half_window = max(smear_kernel.shape[0] // 2, smear_kernel.shape[1] // 2, 1)
    window = 2 * half_window + 1
    threshold = detection_sigma * image_noise_sigma
    candidate_mask = _local_max_mask(response, size=window, threshold=threshold)
    if bloom_mask is not None:
        candidate_mask &= ~bloom_mask

    sigma_psf = psf_sigma_px(psf)
    box_half = max(1, math.ceil(2.0 * sigma_psf))

    out: list[DetectedSource] = []
    rows = candidate_mask.shape[0]
    cols = candidate_mask.shape[1]
    candidate_indices = np.argwhere(candidate_mask)
    for v_idx, u_idx in candidate_indices:
        v_min = int(v_idx) - box_half
        v_max = int(v_idx) + box_half + 1
        u_min = int(u_idx) - box_half
        u_max = int(u_idx) + box_half + 1
        if v_min < 0 or u_min < 0 or v_max > rows or u_max > cols:
            continue
        box = image[v_min:v_max, u_min:u_max]
        saturated = bool(np.any(box >= full_well_dn))
        if saturated:
            dv, du = centroid_saturated(
                box,
                full_well_dn=full_well_dn,
                half_width_inner=box_half - 1,
                half_width_outer=box_half,
            )
        else:
            dv, du = centroid_gaussian_fit(box)
        sharp, round_ = _sharpness_roundness(box)
        if not apply_shape_cuts(sharp, round_):
            continue
        out.append(
            DetectedSource(
                v=float(v_idx) + dv,
                u=float(u_idx) + du,
                peak_dn=float(response[v_idx, u_idx]),
                sharpness=float(sharp),
                roundness=float(round_),
                saturated=saturated,
            )
        )
    return out
