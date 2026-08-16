import math
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from numpy.fft import fft2, fftfreq, ifft2, ifftshift
from pdslogger import PdsLogger
from scipy.ndimage import gaussian_filter, sobel

from spindoctor.config import IMAGE_LOGGER, logged_section
from spindoctor.support.image import crop_center, normalize_array, pad_top_left
from spindoctor.support.misc import mad_std
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_NDArrayComplexType = npt.NDArray[np.complexfloating[Any, Any]]

# Floor for masked NCC denominators and weight sums (avoid divide-by-zero).
_NCC_EPS = 1e-12

# ==============================================================
# Small utilities
# ==============================================================


def gradient_magnitude(arr: NDArrayFloatType) -> NDArrayFloatType:
    """Sobel gradient magnitude.

    Emphasizes edges over flat interior regions, which is what you want when
    the raw-intensity NCC has a broad plateau — e.g., a lit body disc that
    overflows the FOV, where interior brightness is near-uniform and the only
    unique-alignment signal lies at the limb.

    Parameters:
        arr: 2-D input array.

    Returns:
        Same-shape gradient-magnitude array, as ``float64``.
    """
    a = np.asarray(arr, np.float64)
    gx = sobel(a, axis=1, mode='constant', cval=0.0)
    gy = sobel(a, axis=0, mode='constant', cval=0.0)
    return cast(NDArrayFloatType, np.hypot(gx, gy))


def int_to_signed(idx: int, size: int) -> int:
    """Map ``[0 .. size-1]`` argmax index to a signed displacement coordinate.

    Parameters:
        idx: Peak index from correlation search in ``[0, size)``.
        size: FFT / correlation length (modulus for wrapping).

    Returns:
        ``int`` signed offset equivalent to ``idx`` when ``idx < size // 2``, else
        ``idx - size`` (negative branch for peaks in the second half).
    """
    return idx if idx < size // 2 else idx - size


# ==============================================================
# Fourier helpers
# ==============================================================


def fourier_shift(img: NDArrayFloatType, dy: float, dx: float) -> NDArrayFloatType:
    """Subpixel shift via Fourier shift theorem (positive = down/right)."""
    fy = fftfreq(img.shape[0])[:, None]
    fx = fftfreq(img.shape[1])[None, :]
    phase = np.exp(-2j * np.pi * (dy * fy + dx * fx))
    return np.real(ifft2(fft2(img) * phase))


def upsampled_dft(
    X: _NDArrayComplexType,
    up_factor: int,
    region_sz: tuple[int, int],
    offsets: tuple[int, int],
) -> _NDArrayComplexType:
    """Localized upsampled DFT.

    From Guizar-Sicairos, 2008. "Efficient subpixel image registration via cross-correlation."
    Optics Leters, 33(2):156-158
    """
    X_v_size, X_u_size = X.shape
    region_v, region_u = region_sz
    oy, ox = offsets
    ky = ifftshift(np.arange(X_v_size))
    kx = ifftshift(np.arange(X_u_size))
    a = np.arange(region_v) - oy
    b = np.arange(region_u) - ox
    j2pi = 1j * 2.0 * np.pi
    # Use +j sign to evaluate localized inverse DFT samples (consistent with ifft).
    Er = np.exp((j2pi / (X_v_size * up_factor)) * (a[:, None] @ ky[None, :]))
    Ec = np.exp((j2pi / (X_u_size * up_factor)) * (kx[:, None] @ b[None, :]))
    return cast(_NDArrayComplexType, Er @ X @ Ec)


# ==============================================================
# Masked NCC (linear correlation via padding)
# ==============================================================


# Default overlap and variance floors for the bi-directional masked NCC
# (only used when ``data_mask`` is provided).  ``w_frac_min`` rejects
# shifts whose effective-overlap weight w(s) is less than this fraction of
# max(w); ``var_frac_min`` similarly rejects shifts where the image or
# model variance under the mask is a tiny fraction of the best case, which
# is where floating-point noise in the denominator would otherwise produce
# spurious |NCC| >> 1.
_NCC_BIDIR_W_FRAC_MIN: float = 0.3
_NCC_BIDIR_VAR_FRAC_MIN: float = 0.1


def masked_ncc(
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    mask: NDArrayBoolType,
    data_mask: NDArrayBoolType | None = None,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Masked normalized cross-correlation surface between image and model.

    Computes shift-wise NCC (Pearson r) using FFT-based sums. Also
    returns the unnormalized NCC numerator (mean-subtracted
    covariance) as a diagnostic.

    The NCC surface contains values in [-1, 1] at every shift where
    the mask overlaps non-constant image content, and is the
    primary peak-selection surface: it is invariant to the image
    variance under the shifted mask and therefore does not suffer
    from the "scale bias" that causes the numerator to prefer shifts
    where the fixed template mask straddles bright/dark boundaries
    in the image (e.g., the limb of a Lambert-shaded body disc).
    The numerator surface scales with sqrt(image-variance-under-mask)
    times the NCC, which is useful as a sanity check but must not
    be used for peak finding when the template is dense.

    Parameters:
        image: Padded image array.
        model: Padded model array (same shape as image).
        mask: Padded boolean mask (same shape as image).  True where
            model pixels are valid.
        data_mask: Optional padded boolean mask indicating where
            ``image`` contains real sensor data (True) versus
            zero-padded margin (False).  When provided, the NCC is
            computed in its bi-directional form so that pixel pairs
            where the shifted model mask straddles zero-padded image
            pixels do not bias the normalization.  This removes the
            edge-ridge artifact that otherwise appears at
            ``|dV| = extfov_margin_v`` when the body model extends
            into the extfov margin.  When ``None``, the standard
            (single-mask) NCC formula is used.

    Returns:
        Tuple of (ncc, numerator) arrays, each with the same shape
        as the inputs.  ``ncc`` is the Pearson-r surface used for
        peak localization; ``numerator`` is the mean-subtracted
        cross-covariance (unnormalized) returned as a diagnostic.
        When ``data_mask`` is provided, shifts with degenerate
        overlap or variance are set to ``-inf`` in the NCC surface.
    """
    ref_shape = image.shape
    if image.shape != model.shape or image.shape != mask.shape:
        raise ValueError(
            'masked_ncc: image, model, and mask must have the same shape; '
            f'got image {image.shape}, model {model.shape}, mask {mask.shape}.'
        )
    if data_mask is not None and data_mask.shape != ref_shape:
        raise ValueError(
            'masked_ncc: data_mask must match image shape '
            f'{ref_shape}; got data_mask {data_mask.shape}.'
        )
    if data_mask is not None:
        return _masked_ncc_bidir(image, model, mask, data_mask)

    image_fft = fft2(image)
    mask_fft = fft2(mask)
    model_mask_fft = fft2(model * mask)

    sum_w: float = float(np.sum(mask))
    safe_w = sum_w + _NCC_EPS

    # Shift-wise sums via FFT (take real to discard floating-point imaginary noise)
    sum_iw = np.real(ifft2(image_fft * np.conj(mask_fft)))
    sum_i2w = np.real(ifft2(fft2(image**2) * np.conj(mask_fft)))
    sum_imw = np.real(ifft2(image_fft * np.conj(model_mask_fft)))

    # Model stats (constant over shifts)
    sum_mw: float = float(np.sum(model * mask))
    mean_m: float = sum_mw / safe_w
    ssd_mw: float = float(np.sum(((model - mean_m) * mask) ** 2)) + _NCC_EPS

    # Shift-wise image mean
    mean_i = sum_iw / safe_w

    # Numerator: sum of (I - mean_I)(M - mean_M) * W over the mask
    num = sum_imw - mean_i * sum_mw

    # Denominator: sqrt( SSD_I(s) * SSD_M )
    # Epsilon inside the sqrt so the floor is sqrt(_NCC_EPS) rather than
    # _NCC_EPS; this prevents floating-point noise in the numerator from
    # producing |NCC| >> 1 at shifts where the image variance is zero.
    ssd_iw = sum_i2w - sum_iw**2 / safe_w
    ssd_iw[ssd_iw < 0] = 0.0
    denom = np.sqrt(ssd_iw * ssd_mw + _NCC_EPS)

    ncc = num / denom
    return ncc, num


def _masked_ncc_bidir(
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    mask: NDArrayBoolType,
    data_mask: NDArrayBoolType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Bi-directional (data-mask aware) masked NCC; see :func:`masked_ncc`.

    The weight at shift s is w(s) = sum_i data_mask(i) * mask(i - s) rather
    than a constant sum(mask); image and model statistics under the mask are
    computed likewise with data_mask(i) as a per-pixel inclusion factor.
    This eliminates the zero-padding bias that drives a spurious peak at
    ``|dV| = extfov_margin_v`` when the model extends into the padded
    margin.  Shifts with small overlap or degenerate variance are set to
    ``-inf`` so that peak finding does not chase floating-point noise.
    """
    image = np.asarray(image, np.float64)
    model = np.asarray(model, np.float64)
    mask_f = mask.astype(np.float64)
    dmask_f = data_mask.astype(np.float64)

    image_fft = fft2(image)
    image2_fft = fft2(image * image)
    dmask_fft = fft2(dmask_f)
    mask_fft = fft2(mask_f)
    model_mask_fft = fft2(model * mask_f)
    model2_mask_fft = fft2((model * model) * mask_f)

    # Effective overlap weight at each shift.
    w_d = np.real(ifft2(dmask_fft * np.conj(mask_fft)))
    w_d_max = float(w_d.max())
    w_floor = max(_NCC_BIDIR_W_FRAC_MIN * w_d_max, _NCC_EPS)
    overlap_ok = w_d >= w_floor

    # Shift-wise sums.  ``image`` is already zero outside data_mask, so the
    # I*mask and I^2*mask sums are equivalent to dmask*I*mask and
    # dmask*I^2*mask respectively.
    sum_iw = np.real(ifft2(image_fft * np.conj(mask_fft)))
    sum_i2w = np.real(ifft2(image2_fft * np.conj(mask_fft)))
    sum_imw = np.real(ifft2(image_fft * np.conj(model_mask_fft)))
    # Model stats are shift-dependent because dmask selects which model
    # pixels participate at each shift.
    sum_mw = np.real(ifft2(dmask_fft * np.conj(model_mask_fft)))
    sum_m2w = np.real(ifft2(dmask_fft * np.conj(model2_mask_fft)))

    safe_w = np.where(overlap_ok, w_d, w_floor)
    mean_i = sum_iw / safe_w

    num = sum_imw - mean_i * sum_mw
    ssd_iw = np.maximum(sum_i2w - sum_iw**2 / safe_w, 0.0)
    ssd_mw = np.maximum(sum_m2w - sum_mw**2 / safe_w, 0.0)

    # Reject shifts where the image or model has near-constant content
    # under the effective mask; their NCC is dominated by floating-point
    # noise in the denominator.
    ssd_i_max = float(ssd_iw.max())
    ssd_m_max = float(ssd_mw.max())
    var_ok = (ssd_iw > _NCC_BIDIR_VAR_FRAC_MIN * ssd_i_max) & (
        ssd_mw > _NCC_BIDIR_VAR_FRAC_MIN * ssd_m_max
    )
    valid = overlap_ok & var_ok

    denom = np.sqrt(ssd_iw * ssd_mw)
    ncc = np.where(valid, num / np.maximum(denom, _NCC_EPS), 0.0)
    # Mathematically |NCC| <= 1; floating-point error can push it slightly
    # above, so clamp before invalidating.
    ncc = np.clip(ncc, -1.0, 1.0)
    ncc[~valid] = -np.inf
    return ncc, num


# ==============================================================
# Peak metrics & selection
# ==============================================================


def psr_metric(corr: NDArrayFloatType, rc: tuple[int, int], guard: int = 5) -> float:
    corr_v, corr_u = corr.shape
    row, col = rc
    peak = corr[row, col]
    y, x = np.ogrid[:corr_v, :corr_u]
    mask = (y - row) ** 2 + (x - col) ** 2 > guard**2
    bg = corr[mask]
    # Exclude invalidated-shift sentinels (-inf) before computing background stats;
    # otherwise ``bg.mean()`` collapses to -inf and PSR becomes NaN.
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        return float('nan')
    return float((peak - bg.mean()) / (bg.std() + 1e-12))


def pmr_metric(corr: NDArrayFloatType, peak_val: float) -> float:
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        return float('nan')
    return float(peak_val / (finite.mean() + 1e-12))


def per_metric(corr: NDArrayFloatType, peak_val: float) -> float:
    finite = corr[np.isfinite(corr)]
    if finite.size == 0:
        return float('nan')
    return float(peak_val / (np.sqrt(np.sum(finite**2)) + 1e-12))


def nms_topk(
    corr: NDArrayFloatType,
    k: int = 5,
    radius: int = 5,
    max_offset_vu: tuple[int, int] | None = None,
) -> list[tuple[int, int, float]]:
    """Non-maximum suppression to get top-k peaks.

    Parameters:
        corr: 2-D correlation surface (V x U).
        k: Maximum number of peaks to return.
        radius: Suppression radius around each selected peak in pixels.
        max_offset_vu: If given, only positions whose signed offset
            satisfies ``|dV| < max_offset_vu[0]`` and
            ``|dU| < max_offset_vu[1]`` are eligible (strict inequality:
            peaks at the exact window boundary are excluded because they
            signal a clipped search and cannot be trusted; the pyramid
            driver additionally marks any final result within 1 pixel of
            the boundary as spurious). Signed offsets are derived from the
            FFT-convention wrap-around used by :func:`int_to_signed`.

    Returns:
        List of ``(row, col, value)`` tuples for up to ``k`` peaks.
    """
    corr_v, corr_u = corr.shape
    work = corr.copy()
    if max_offset_vu is not None:
        max_v, max_u = max_offset_vu
        rows = np.arange(corr_v)
        cols = np.arange(corr_u)
        # Signed offsets via FFT wrap-around convention (same as int_to_signed).
        signed_rows = np.where(rows < corr_v // 2, rows, rows - corr_v)
        signed_cols = np.where(cols < corr_u // 2, cols, cols - corr_u)
        work[np.abs(signed_rows) >= max_v, :] = -np.inf
        work[:, np.abs(signed_cols) >= max_u] = -np.inf
    peaks: list[tuple[int, int, float]] = []
    if not np.any(np.isfinite(work)):
        return peaks
    for _ in range(k):
        idx = int(np.argmax(work))
        v = float(work.flat[idx])
        if not np.isfinite(v):
            break
        row_i, col_i = np.unravel_index(idx, work.shape)
        row, col = int(row_i), int(col_i)
        peaks.append((row, col, v))
        y, x = np.ogrid[:corr_v, :corr_u]
        work[(y - row) ** 2 + (x - col) ** 2 <= radius**2] = -np.inf
    return peaks


# ==============================================================
# Matched-filter (peak-curvature) covariance
# ==============================================================

# Lower bound on the number of statistically independent samples the
# residual is allowed to represent.  The residual over the template
# support is spatially correlated (camera PSF, anti-alias blur, the
# sub-pixel refine low-pass, and coherent silhouette / photometric model
# error), so a template covering thousands of pixels carries far fewer
# independent constraints than its pixel count.  The correlation-area
# inflation divides the effective sample count by the residual's
# correlation area; this floor keeps a near-constant residual (whose
# measured correlation area approaches the whole patch) from collapsing
# the effective count to a value that would drive the covariance to
# infinity.
_MIN_EFFECTIVE_SAMPLES: float = 4.0


def _residual_correlation_area(resid: NDArrayFloatType) -> float:
    """Estimate the residual's spatial correlation area in pixels.

    The matched-filter Cramer-Rao bound assumes white (per-pixel
    independent) noise.  A rendered-template residual is not white: the
    camera PSF, the anti-alias and sub-pixel-refine low-pass filters, and
    -- most importantly -- coherent model error (a silhouette or
    photometric mismatch spread across the whole body) correlate
    neighboring residual pixels.  The number of statistically independent
    samples is therefore ``N_pixels / A_c`` where ``A_c`` is the
    correlation area returned here, and the white-noise covariance must be
    inflated by ``A_c`` to reflect the true information content.

    ``A_c`` is estimated as the product of per-axis correlation lengths,
    each the integral of the residual's normalized autocorrelation over
    its central positive lobe (``1 + 2 * sum_{k>=1} rho(k)`` truncated at
    the first non-positive lag).  A white residual yields ``A_c ~ 1``; a
    residual correlated over ``L`` pixels per axis yields ``A_c ~ L**2``.

    Parameters:
        resid: 2-D residual field (image minus aligned model), any scale.

    Returns:
        Correlation area in pixels, floored at ``1.0``.
    """
    r = np.asarray(resid, np.float64)
    r = r - float(r.mean())
    if r.size == 0 or float(np.mean(r * r)) <= 1e-30:
        return 1.0

    def _axis_length(axis: int) -> float:
        n = r.shape[axis]
        if n < 3:
            return 1.0
        spec = np.fft.rfft(r, axis=axis)
        power = (spec * np.conj(spec)).real
        autocorr = np.fft.irfft(power, n=n, axis=axis)
        # Collapse the perpendicular axis so the 1-D autocorrelation is the
        # per-lag average across the other axis's lines.
        autocorr_1d = autocorr.mean(axis=1 - axis)
        zero_lag = float(autocorr_1d[0])
        if zero_lag <= 0.0:
            return 1.0
        rho = autocorr_1d / zero_lag
        length = 1.0
        for k in range(1, n // 2):
            rho_k = float(rho[k])
            if rho_k <= 0.0:
                break
            length += 2.0 * rho_k
        return max(length, 1.0)

    area = _axis_length(0) * _axis_length(1)
    # Never claim fewer than a handful of independent samples: cap the area
    # so the effective count N_pixels / A_c stays at or above the floor.
    max_area = max(r.size / _MIN_EFFECTIVE_SAMPLES, 1.0)
    return float(min(max(area, 1.0), max_area))


def matched_filter_covariance(
    model_aligned: NDArrayFloatType,
    sigma_n: float,
    *,
    correlation_area: float = 1.0,
) -> NDArrayFloatType:
    """Peak-curvature (matched-filter) covariance of a 2-D translation fit.

    For a template ``M`` aligned to an image ``I = M(x - s) + n`` the shift
    ``s`` maximum-likelihood estimate has Fisher information ``F_ab =
    (1 / sigma_n**2) * sum_x (dM/dx_a)(dM/dx_b)`` and covariance
    ``F**-1 = sigma_n**2 * inv(sum grad M grad M^T)``.

    Two properties are essential for the reported sigma to track the true
    error and are the reason this replaces the previous derivation:

    - **Unit consistency.**  ``sigma_n`` and the gradients of ``M`` must be
      measured on the same intensity scale, or the covariance scales with
      the (arbitrary) template amplitude.  The caller passes a residual
      noise ``sigma_n`` computed on the zero-mean, unit-std normalized
      image and model; this function must therefore receive the *same*
      normalized model so ``grad M`` shares that scale.  Passing the raw
      template here (whose amplitude can be thousands of DN) shrinks the
      covariance by the template variance and is the miscalibration this
      derivation corrects.
    - **Effective sample count.**  ``correlation_area`` inflates the
      white-noise bound by the residual's spatial correlation area so
      correlated model error is not counted as thousands of independent
      constraints (see :func:`_residual_correlation_area`).

    Parameters:
        model_aligned: The aligned template on the *same normalized scale*
            as the residual from which ``sigma_n`` was measured.
        sigma_n: Residual noise standard deviation on that normalized
            scale.
        correlation_area: Residual correlation area in pixels (``>= 1``);
            multiplies the covariance.  Default ``1.0`` reproduces the
            white-noise Cramer-Rao bound.

    Returns:
        The ``2x2`` translation covariance in pixels squared.  A degenerate
        (rank-deficient) gradient structure returns a large isotropic
        covariance so the result is de-weighted rather than trusted.
    """
    sy = np.gradient(model_aligned, axis=0)
    sx = np.gradient(model_aligned, axis=1)
    Sxx = float(np.sum(sx * sx))
    Syy = float(np.sum(sy * sy))
    Sxy = float(np.sum(sx * sy))
    var_n = float(sigma_n) ** 2 + 1e-18
    area = max(float(correlation_area), 1.0)
    fisher = (1.0 / var_n) * np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
    if np.linalg.cond(fisher) > 1e10:
        # Degenerate case: return large uncertainty
        return np.diag([1e6, 1e6])
    cov_white = np.linalg.pinv(fisher + 1e-12 * np.eye(2))
    return cast(NDArrayFloatType, area * cov_white)


def _ncc_quadratic_axis_offset(minus: float, center: float, plus: float) -> float:
    """Sub-pixel offset of a 1-D quadratic through three equally-spaced samples.

    Fits a parabola through ``(-1, minus)``, ``(0, center)``, ``(+1, plus)``
    and returns its vertex abscissa, clamped to ``[-0.5, 0.5]`` (a vertex
    outside that range would contradict ``center`` being the integer argmax).
    Returns ``0.0`` when any sample is non-finite (the bi-directional masked
    NCC marks invalid shifts with ``-inf``) or when the samples do not form
    a local maximum along this axis.

    Parameters:
        minus: Sample one step below the peak.
        center: Sample at the integer peak.
        plus: Sample one step above the peak.

    Returns:
        Vertex offset from the center sample, in samples.
    """
    if not (math.isfinite(minus) and math.isfinite(center) and math.isfinite(plus)):
        return 0.0
    # A neighbor above the center contradicts the center being the axis-local
    # maximum (possible only on malformed input; the caller passes the global
    # argmax); such a triple carries no trustworthy vertex.
    if center < minus or center < plus:
        return 0.0
    denom = minus - 2.0 * center + plus
    # NCC values are O(1), so an absolute curvature floor suffices: a
    # near-flat neighborhood (or one degenerate to rounding noise) carries
    # no sub-pixel information and keeps the integer peak.
    if denom >= -1.0e-12:
        return 0.0
    return float(np.clip(0.5 * (minus - plus) / denom, -0.5, 0.5))


# ==============================================================
# Single-scale, K-peak evaluation (with optional prior)
# ==============================================================


def evaluate_candidate(
    *,
    image_pad: NDArrayFloatType,
    model_pad: NDArrayFloatType,
    mask_pad: NDArrayBoolType,
    corr: NDArrayFloatType,
    rc: tuple[int, int],
    upsample_factor: int,
    model_shape: tuple[int, int],
    image_shape: tuple[int, int],
    logger: PdsLogger | None,
    prior_shift: tuple[float, float] | None = None,
    prior_weight: float = 0.0,
    metric: str = 'psr',
    refine_image_pad: NDArrayFloatType | None = None,
    refine_model_pad: NDArrayFloatType | None = None,
    refine_lowpass_sigma_px: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate a candidate for the navigation.

    Parameters:
        image_pad: The padded image.
        model_pad: The padded model.
        mask_pad: The padded mask.
        corr: The correlation matrix.
        rc: The row and column of the candidate.
        upsample_factor: The upsample factor.
        model_shape: The shape of the model.
        image_shape: The shape of the image.
        logger: Logger for the saturated-refinement diagnostic; ``None``
            disables that debug line and changes nothing else.
        prior_shift: The prior shift.
        prior_weight: The prior weight.
        metric: The metric to use for the navigation.
        refine_image_pad: Optional padded image used *only* for the sub-pixel
            cross-power-spectrum refinement, in place of ``image_pad``.  When the
            coarse peak is found on gradient-magnitude surfaces (``use_gradient``),
            pass the raw-intensity padded image here: the Sobel *magnitude*
            rectifies the signal, so its cross-power peak is non-smooth at the apex
            and the upsampled-DFT sub-pixel estimate is biased (largest near
            whole-pixel offsets), whereas raw intensity reaches the
            ``upsample_factor`` resolution.  Defaults to ``image_pad``.
        refine_model_pad: Optional padded model for the same refinement; defaults
            to ``model_pad``.

    Returns:
        A dictionary containing the navigation result.

    Raises:
        ValueError: If the model is smaller than the image in either dimension
            (the model must be at least image-sized; pad it with
            ``pad_top_left`` first).  This is validated here so the failure is a
            clear contract error rather than an opaque ``crop_center`` error two
            calls deep.
    """

    model_h, model_w = model_shape
    image_h, image_w = image_shape
    if image_h > model_h or image_w > model_w:
        raise ValueError(
            f'model shape {model_shape} must be at least as large as the image '
            f'shape {image_shape} in both dimensions; a model smaller than the '
            f'image is not supported (pad the model to at least image size first)'
        )

    corr_v, corr_u = corr.shape
    p, q = rc
    dy_i, dx_i = int_to_signed(p, corr_v), int_to_signed(q, corr_u)

    # Subpixel refinement: local upsampled DFT of correlation spectrum numerator.
    # Refine on the raw-intensity surfaces when provided (see ``refine_image_pad``)
    # so a gradient-magnitude coarse peak does not carry its rectified-cusp
    # sub-pixel bias into the reported offset.
    refine_image = image_pad if refine_image_pad is None else refine_image_pad
    refine_model = model_pad if refine_model_pad is None else refine_model_pad
    refine_model_masked = refine_model * mask_pad
    # Band-limit both surfaces before the cross-power.  The sub-pixel peak of two
    # sharp, anti-aliased (but non-PSF-blurred) surfaces with differing edge
    # profiles -- a rendered body / ring vs. a Lambert template -- is aliased by a
    # sub-pixel-phase-dependent amount (a ~0.03 px odd S-curve, zero at integer and
    # half-pixel offsets).  A matched Gaussian low-pass removes the high-frequency
    # mismatch that drives it, cutting the disc / ring residual to ~0.01 px; the
    # upsampled DFT still localizes the smoothed peak.  The low-pass is applied to
    # the final surfaces (the image and the already-masked model) so the model's
    # support edge is not re-sharpened by the mask afterwards.  Default 0.0 is a
    # no-op.
    if refine_lowpass_sigma_px > 0.0:
        refine_image = gaussian_filter(refine_image, refine_lowpass_sigma_px)
        refine_model_masked = gaussian_filter(refine_model_masked, refine_lowpass_sigma_px)
    spec = fft2(refine_image) * np.conj(fft2(refine_model_masked))
    # Center of the local upsampled DFT window should be the middle of the
    # evaluation region (e.g., 3x3 -> center index 1), not half the upsample factor.
    # Using upsample_factor here caused increasing bias for large factors.
    # Use a region size that scales with upsample_factor so the true peak
    # (which can be ~0.5*upsample_factor samples from center) is inside the window.
    region_v = upsample_factor + 1
    region_u = upsample_factor + 1
    oy = region_v // 2
    ox = region_u // 2
    Up = upsampled_dft(
        spec,
        upsample_factor,
        (region_v, region_u),
        (oy - dy_i * upsample_factor, ox - dx_i * upsample_factor),
    )
    upy, upx = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (int(upy) - oy) / upsample_factor
    dx = dx_i + (int(upx) - ox) / upsample_factor
    # The refinement window spans the half-pixel neighborhood of the integer
    # NCC peak (exactly +-0.5 px for the even upsample factors in use; an odd
    # factor would make it asymmetric by one upsampled step).  An argmax on
    # the window boundary means the cross-power surface's
    # own peak lies beyond half a pixel from the NCC peak — the two surfaces
    # disagree, which happens when the template constrains an axis poorly
    # (e.g. a disc that overflows the FOV) and the raw surface's scale bias
    # takes over.  Following the raw surface farther is not safe (its peak
    # can sit pixels away from the true alignment), and reporting the
    # boundary value pins the offset at exactly +-0.5 px, which breaks shift
    # equivariance (#447).  Instead fall back, per saturated axis, to the
    # quadratic vertex of the NCC surface itself — the surface that selected
    # the peak — which is bounded to the same half-pixel and consistent with
    # the peak choice.
    saturated_v = upy in (0, region_v - 1)
    saturated_u = upx in (0, region_u - 1)
    if saturated_v or saturated_u:
        corr_v_size, corr_u_size = corr.shape
        if saturated_v:
            dy = dy_i + _ncc_quadratic_axis_offset(
                float(corr[(p - 1) % corr_v_size, q]),
                float(corr[p, q]),
                float(corr[(p + 1) % corr_v_size, q]),
            )
        if saturated_u:
            dx = dx_i + _ncc_quadratic_axis_offset(
                float(corr[p, (q - 1) % corr_u_size]),
                float(corr[p, q]),
                float(corr[p, (q + 1) % corr_u_size]),
            )
        if logger is not None:
            logger.debug(
                'Sub-pixel refinement saturated at the half-pixel window edge '
                '(v: %s, u: %s) around integer peak (%d, %d); using the NCC '
                'quadratic vertex for the saturated axis: offset (%.4f, %.4f)',
                saturated_v,
                saturated_u,
                dy_i,
                dx_i,
                dy,
                dx,
            )

    # Align combined model and compute residual stats (model_h/w, image_h/w
    # were unpacked and validated at the top of the function).
    model_shift = fourier_shift(model_pad[:model_h, :model_w], dy, dx)
    model_crop = crop_center(model_shift, (image_h, image_w))
    image_crop = image_pad[:image_h, :image_w]
    # Normalize both surfaces to zero-mean unit-std before the residual so
    # the residual noise and the template gradients that feed the
    # matched-filter covariance share one intensity scale (a raw-DN
    # template would otherwise shrink the covariance by its own variance).
    model_norm = normalize_array(model_crop)
    resid = normalize_array(image_crop) - model_norm
    sigma_n = mad_std(resid)
    if sigma_n <= 1e-12:
        # Near-perfect match (or a constant residual): fall back to the plain
        # std so the covariance stays finite rather than dividing by zero.
        sigma_n = max(resid.std(), 1e-6)
    # Inflate the white-noise bound by the residual correlation area: the
    # residual is spatially correlated (PSF, anti-alias / refine low-pass,
    # and coherent silhouette / photometric model error), so its pixels are
    # not independent samples.
    correlation_area = _residual_correlation_area(resid)
    cov = matched_filter_covariance(model_norm, sigma_n, correlation_area=correlation_area)

    # Quality metric
    peak_val = corr[p, q]
    if metric == 'psr':
        quality = psr_metric(corr, (p, q))
    elif metric == 'pmr':
        quality = pmr_metric(corr, peak_val)
    elif metric == 'per':
        quality = per_metric(corr, peak_val)
    else:
        raise ValueError(f"metric must be 'psr', 'pmr', or 'per', not '{metric}'")

    # Prior penalty (encourage pyramid consistency or external priors)
    if prior_shift is not None and prior_weight > 0.0:
        # TODO The prior penalty subtracts prior_weight * dist from quality, but the units/scale of
        # quality (PSR/PMR/PER) may not be comparable to distance. This could make the penalty
        # disproportionately strong or weak depending on the metric choice.
        # Consider normalizing the distance penalty or using a separate scoring function that
        # combines quality and distance in a principled way (e.g., weighted sum with normalized
        # components).
        dist = np.hypot(dy - prior_shift[0], dx - prior_shift[1])
        quality -= prior_weight * dist

    return {
        'offset': (float(dy), float(dx)),
        'cov': cov,
        'sigma_xy': (float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))),
        'quality': float(quality.real),
        'peak_val': float(peak_val.real),
        'rc': (int(p), int(q)),
    }


def navigate_single_scale_kpeaks(
    *,
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    mask: NDArrayBoolType,
    logger: PdsLogger | None,
    max_peaks: int = 5,
    upsample_factor: int = 16,
    metric: str = 'psr',
    prior_shift: tuple[float, float] | None = None,
    prior_weight: float = 0.0,
    nms_radius: int = 5,
    max_offset_vu: tuple[int, int] | None = None,
    data_mask: NDArrayBoolType | None = None,
    use_gradient: bool = False,
    refine_lowpass_sigma_px: float = 0.0,
) -> dict[str, Any]:
    """
    One-scale masked NCC + top-K candidate evaluation.

    Parameters:
        image: The image to navigate.
        model: The model to navigate.
        mask: The mask to use for the navigation.
        max_peaks: The number of peaks to use for the navigation.
        upsample_factor: The upsample factor to use for the navigation.
        metric: The metric to use for the navigation.
        prior_shift: The prior shift to use for the navigation.
        prior_weight: The prior weight to use for the navigation.
        nms_radius: The radius to use for the non-maximum suppression.
        logger: The logger to use for the navigation.
        max_offset_vu: If given, only correlation peaks whose signed
            (V, U) offset satisfies ``|dV| <= max_offset_vu[0]`` and
            ``|dU| <= max_offset_vu[1]`` are considered candidates.
            Typically set to the extended-FOV margins so that offsets
            outside the physically-plausible range are never evaluated.
        data_mask: Optional boolean mask (same shape as ``image``) that
            is True where the image contains real sensor data and False
            inside the zero-padded extended-FOV margin.  When provided,
            the NCC is computed in its bi-directional form so that the
            model extending into the padded margin does not bias the
            peak toward ``|dV| = margin_v``.
        use_gradient: When True, replace ``image`` and ``model`` with their
            Sobel gradient magnitudes before the NCC. Use this when the raw
            intensity surface has a broad plateau — e.g., a body that fills
            or overflows the FOV, where the only unique-alignment signal is
            at the limb.

    Returns:
        A dictionary containing the navigation result.
    """

    # Use original image for correlation surfaces; masked NCC computes its own
    # normalization, and using normalized-and-then-padded images biases the
    # unnormalized numerator surface used in refinement.
    image_raw = np.asarray(image, np.float64)
    model_raw = np.asarray(model, np.float64)
    if use_gradient:
        image_orig = np.asarray(gradient_magnitude(image), np.float64)
        model_arr = np.asarray(gradient_magnitude(model), np.float64)
    else:
        image_orig = image_raw
        model_arr = model_raw
    model_h, model_w = model_arr.shape
    image_h, image_w = image_orig.shape
    padded_h, padded_w = image_h + model_h, image_w + model_w

    image_pad = pad_top_left(image_orig, padded_h, padded_w)
    model_pad = pad_top_left(model_arr, padded_h, padded_w)
    mask_pad = pad_top_left(mask, padded_h, padded_w)
    # The gradient surfaces localise the integer peak (and drive quality), but the
    # sub-pixel refinement runs on the raw-intensity surfaces to avoid the
    # gradient-magnitude rectification bias.  Without gradient mode these are the
    # same arrays, so the refinement is unchanged.
    if use_gradient:
        refine_image_pad = pad_top_left(image_raw, padded_h, padded_w)
        refine_model_pad = pad_top_left(model_raw, padded_h, padded_w)
    else:
        refine_image_pad = image_pad
        refine_model_pad = model_pad
    data_mask_pad: NDArrayBoolType | None = None
    if data_mask is not None:
        data_mask_pad = pad_top_left(data_mask.astype(bool), padded_h, padded_w)

    corr, _corr_num = masked_ncc(image_pad, model_pad, mask_pad, data_mask=data_mask_pad)
    # Use the NCC (Pearson r) itself for peak localization, not the unnormalized
    # numerator.  The numerator scales with sqrt(image-variance-under-mask), and
    # for dense templates -- body discs, Lambert-shaded limbs, rings, etc. -- that
    # variance changes systematically with shift because the fixed body-shaped
    # mask straddles more or less of the bright/dark boundary at different shifts.
    # This produces a "scale bias" that can move the numerator's peak several
    # pixels away from the true alignment even though the NCC peak is correct
    # (reproduced on Enceladus N1500045859 where the numerator peak was at
    # (dV, dU) = (+11, +22) but the NCC peak -- and the correct answer -- is at
    # (dV, dU) = (-2, -6)).
    #
    # Restricting to ``max_offset_vu`` prevents the classic NCC failure mode --
    # plateaus at 1.0 over shifts where the mask covers pure noise -- because
    # such shifts are outside any physically plausible offset range.
    peaks = nms_topk(corr, k=max_peaks, radius=nms_radius, max_offset_vu=max_offset_vu)

    if logger is not None:
        logger.debug('Correlation peaks:')

    candidates = []
    for p, q, _ in peaks:
        evaluation = evaluate_candidate(
            image_pad=image_pad,
            model_pad=model_pad,
            mask_pad=mask_pad,
            corr=corr,
            rc=(p, q),
            upsample_factor=upsample_factor,
            model_shape=(model_h, model_w),
            image_shape=(image_h, image_w),
            prior_shift=prior_shift,
            prior_weight=prior_weight,
            metric=metric,
            logger=logger,
            refine_image_pad=refine_image_pad,
            refine_model_pad=refine_model_pad,
            refine_lowpass_sigma_px=refine_lowpass_sigma_px,
        )
        candidates.append(evaluation)
        if logger is not None:
            logger.debug(
                f'  Candidate {p}, {q} results: '
                f'offset {evaluation["offset"][0]:.3f}, {evaluation["offset"][1]:.3f}; '
                f'sigma_xy {evaluation["sigma_xy"][0]:.3f}, '
                f'{evaluation["sigma_xy"][1]:.3f}; '
                f'quality {evaluation["quality"]:.3f}; '
                f'peak_val {evaluation["peak_val"]:.3f}'
            )

    if not candidates:
        # TODO When no candidates are found, the function returns cov: np.diag([1e6, 1e6])
        # and quality: -np.inf. Downstream code might not check for -np.inf quality and could
        # treat this as a valid result. Consider returning None or raising an exception instead.
        # The result-shape contract matches the populated path so callers
        # (e.g. ``navigate_with_pyramid_kpeaks`` debug log) can read every
        # key without a KeyError when the search collapses.
        return {
            'offset': (0.0, 0.0),
            'cov': np.diag([1e6, 1e6]),
            'sigma_xy': (1e3, 1e3),
            'quality': -np.inf,
            'peak_val': 0.0,
            'rc': (0, 0),
            'all_candidates': [],
        }
    winner = max(candidates, key=lambda r: r['quality'])
    # Carry every evaluated candidate so callers that want to inspect
    # runner-up peaks (e.g. peak-to-runner-up ratio diagnostics) can do
    # so without re-running the correlation.  Sorted by quality desc so
    # ``all_candidates[0]`` is the winner and ``[1:]`` are runner-ups.
    # Return a shallow copy of the winner with the per-candidate list
    # attached separately so the returned dict is not self-referential
    # (the original winner remains an entry inside the new list).
    sorted_candidates = sorted(candidates, key=lambda r: r['quality'], reverse=True)
    result = dict(winner)
    result['all_candidates'] = sorted_candidates
    return result


# ==============================================================
# Pyramid wrapper with K-peak final selection
# ==============================================================


_MAX_PEAK_RATIO: float = 1.0e3
"""Upper bound on :func:`peak_to_runner_up_ratio`.

Any ratio at or above this represents an effectively unambiguous winner;
the downstream confidence sigmoid saturates far below it (``divisor=2``,
``cap_at=1`` in ``config_510_techniques.yaml``).  The cap keeps the
returned value -- and the diagnostic that stores it -- bounded instead of
blowing up to ``~1e9`` when the runner-up quality is near zero.
"""


def peak_to_runner_up_ratio(top_k_peaks: list[tuple[float, float, float]]) -> float:
    """Return the ratio of the winning peak's quality to the runner-up's.

    ``top_k_peaks`` is ``[(quality, dv, du), ...]`` sorted by quality
    descending (the convention :func:`navigate_with_pyramid_kpeaks` uses).
    Returns ``1.0`` when only one peak survives non-maximum suppression --
    what an unambiguous correlation looks like, so a value at or above 1.0
    is the "good" tail -- and ``0.0`` when no peaks are present.  When the
    runner-up quality is non-positive (rare; happens with the prior
    penalty) the result is the unambiguous-winner cap ``_MAX_PEAK_RATIO``
    rather than ``winner / 1e-9`` (which scaled with the winner's
    magnitude and could reach ``~1e9``).  The ordinary ratio is likewise
    clamped to ``_MAX_PEAK_RATIO``.

    Parameters:
        top_k_peaks: Peaks sorted by quality descending.

    Returns:
        Capped peak-to-runner-up quality ratio.
    """
    if not top_k_peaks:
        return 0.0
    if len(top_k_peaks) == 1:
        return 1.0
    winner_q = float(top_k_peaks[0][0])
    runner_q = float(top_k_peaks[1][0])
    if runner_q <= 1e-9:
        return _MAX_PEAK_RATIO if winner_q > 0.0 else 0.0
    return min(winner_q / runner_q, _MAX_PEAK_RATIO)


@logged_section('correlate', 'CORRELATE')
def navigate_with_pyramid_kpeaks(
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    mask: NDArrayBoolType,
    pyramid_levels: int = 3,
    max_peaks: int = 5,
    upsample_factor: int = 128,
    metric: str = 'psr',
    quality_thresh: float = 6.0,
    consistency_tol: float = 2.0,
    nms_radius: int = 5,
    prior_weight_final: float = 0.25,
    max_offset_vu: tuple[int, int] | None = None,
    data_mask: NDArrayBoolType | None = None,
    use_gradient: bool | Literal['auto'] = False,
    refine_lowpass_sigma_px: float = 0.0,
    localization_uncertainty_scale: float = 0.0,
    logger: PdsLogger | None = None,
) -> dict[str, Any]:
    """TODO Clean this up
    Build class-aware effective model + mask, run coarse->fine, then evaluate K peaks at final
    scale.
    Returns dict with shift, covariance, sigma_xy, quality, consistency, spurious flag.

    Parameters:
        image: The source image to navigate, unpadded.
        model: The model to navigate against, padded as necessary to include more data around the
            edges. It does not need to be the same size as the image.
        mask: The mask indicating which pixels in the model are valid. Same size as the model.
        pyramid_levels: The number of pyramid levels to use. Each pyramid level divides the
            image and model by an additional factor of 2 (pyramid_levels=3 means to start with
            1/4, then 1/2, then 1/1 downsampling).
        max_peaks: The number of peaks to look for in the correlation at each pyramid level.
        upsample_factor: The upsample factor to use for increased FFT resolution around a peak.
        metric: The metric to use for the navigation. Can be one of 'psr', 'pmr', or 'per'.
        quality_thresh: The quality threshold to use for the navigation.
        consistency_tol: The consistency tolerance to use for the navigation.
        nms_radius: The radius to use for the non-maximum suppression.
        prior_weight_final: The prior weight to use for the final navigation.
        max_offset_vu: Maximum permitted signed offset as ``(max_dV, max_dU)`` in
            full-resolution pixels.  When given, correlation peaks outside this
            range are excluded at every pyramid level (the limit is scaled by the
            downsample factor at each level).  Pass ``obs.extfov_margin_vu`` to
            restrict the search to offsets that are physically reachable given the
            extended FOV padding.
        data_mask: Optional boolean mask (same shape as ``image``) that is True
            where the image contains real sensor data and False inside the
            zero-padded extended-FOV margin.  When provided, the NCC is computed
            in its bi-directional form at every pyramid level so that a body
            model extending into the extfov margin does not bias the peak toward
            ``|dV| = margin_v``.  Pass ``obs.extfov_data_sensor_mask()``.
        use_gradient: Controls gradient-magnitude preprocessing of the image and
            model before the NCC.

            - ``False`` (default): raw-intensity NCC at every pyramid level.
            - ``True``: gradient-magnitude NCC at every pyramid level. Use this
              when the raw surface has a broad plateau — e.g., a body that
              fills or overflows the FOV, where only the limb carries
              unique-alignment signal.
            - ``'auto'``: run both modes, pick the more confident result. A
              non-spurious result is preferred over a spurious one; within the
              same spurious bucket the higher-quality result wins.
        localization_uncertainty_scale: Scales the inter-pyramid-level peak
            migration (``consistency``, in px) into an added translation
            uncertainty.  The peak-curvature covariance measures only the
            statistical (photon / residual) precision at the winning peak; a
            peak that walks between pyramid levels is empirically less well
            localized than a peak that does not, so
            ``(scale * consistency)**2`` is added in quadrature to the
            translation covariance diagonal.  ``0.0`` (default) disables it
            and leaves the bare peak-curvature covariance.
        logger: The logger to use for the navigation.

    Returns:
        A dictionary containing the navigation result:
        - offset: The offset.
        - cov: The covariance matrix.
        - sigma_xy: The sigma_xy.
        - quality: The quality of the navigation.
        - metric: The metric used for the navigation.
        - consistency: The consistency of the navigation.
        - spurious: True if the navigation is spurious, False otherwise.

    Notes:
        The metrics are:

        - PSR (Peak-to-Sidelobe Ratio):
          Measures peak distinctness as (peak - mean_sidelobe) / std_sidelobe,
          where the sidelobe region excludes the peak neighborhood.

        - PMR (Peak-to-Mean Ratio):
          Ratio of the global maximum correlation value to the mean of all correlation values;
          indicates how dominant the main peak is over the average background.

        - PER (Peak-to-Energy Ratio):
          Ratio of the squared peak value to the total correlation energy (sum of squares);
          reflects how much of the total response energy is concentrated in the main peak.
    """

    if logger is None:
        logger = IMAGE_LOGGER

    if use_gradient == 'auto':
        logger.debug('Auto-gradient: running raw-intensity pass')
        result_raw = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=pyramid_levels,
            max_peaks=max_peaks,
            upsample_factor=upsample_factor,
            metric=metric,
            quality_thresh=quality_thresh,
            consistency_tol=consistency_tol,
            nms_radius=nms_radius,
            prior_weight_final=prior_weight_final,
            max_offset_vu=max_offset_vu,
            data_mask=data_mask,
            use_gradient=False,
            refine_lowpass_sigma_px=refine_lowpass_sigma_px,
            localization_uncertainty_scale=localization_uncertainty_scale,
            logger=logger,
        )
        logger.debug('Auto-gradient: running gradient-magnitude pass')
        result_grad = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=pyramid_levels,
            max_peaks=max_peaks,
            upsample_factor=upsample_factor,
            metric=metric,
            quality_thresh=quality_thresh,
            consistency_tol=consistency_tol,
            nms_radius=nms_radius,
            prior_weight_final=prior_weight_final,
            max_offset_vu=max_offset_vu,
            data_mask=data_mask,
            use_gradient=True,
            refine_lowpass_sigma_px=refine_lowpass_sigma_px,
            localization_uncertainty_scale=localization_uncertainty_scale,
            logger=logger,
        )
        # Pick the better result by ordered tiers. At-edge comes before
        # quality: a mode with no real peak will often drift to the search
        # boundary where the bidirectional NCC overlap weight maximizes, and
        # the resulting false peak can still score high quality against a
        # plateau of similar values — so an at-edge result must lose to any
        # interior result even if the interior result has lower quality.
        #   1. Non-spurious beats spurious.
        #   2. Not-at-edge beats at-edge.
        #   3. Higher quality wins within the same tier.
        raw_tier = (not result_raw['spurious'], not result_raw['at_edge'])
        grad_tier = (not result_grad['spurious'], not result_grad['at_edge'])
        reason_parts = []
        if raw_tier != grad_tier:
            winner = result_raw if raw_tier > grad_tier else result_grad
            chosen = 'raw' if raw_tier > grad_tier else 'gradient'
            if raw_tier[0] != grad_tier[0]:
                reason_parts.append('non-spurious beats spurious')
            if raw_tier[1] != grad_tier[1]:
                reason_parts.append('interior beats at-edge')
        elif result_raw['quality'] >= result_grad['quality']:
            winner = result_raw
            chosen = 'raw'
            reason_parts.append('higher quality')
        else:
            winner = result_grad
            chosen = 'gradient'
            reason_parts.append('higher quality')
        logger.debug(
            'Auto-gradient: picked %s (%s) — '
            'raw: quality=%.3f spurious=%s at_edge=%s; '
            'grad: quality=%.3f spurious=%s at_edge=%s',
            chosen,
            ', '.join(reason_parts),
            result_raw['quality'],
            result_raw['spurious'],
            result_raw['at_edge'],
            result_grad['quality'],
            result_grad['spurious'],
            result_grad['at_edge'],
        )
        winner['used_gradient'] = chosen == 'gradient'
        return winner

    if pyramid_levels < 1:
        raise ValueError(f'pyramid_levels must be >= 1; got {pyramid_levels}')

    logger.debug('Navigating with pyramid kpeaks:')
    logger.debug(f'  Pyramid levels: {pyramid_levels}')
    logger.debug(f'  Max peaks: {max_peaks}')
    logger.debug(f'  Upsample factor: {upsample_factor}')
    logger.debug(f'  Metric: {metric}')
    logger.debug(f'  Quality threshold: {quality_thresh}')
    logger.debug(f'  Consistency tolerance: {consistency_tol}')
    logger.debug(f'  NMS radius: {nms_radius}')
    logger.debug(f'  Prior weight final: {prior_weight_final}')
    logger.debug(f'  Use gradient: {use_gradient}')
    if max_offset_vu is not None:
        logger.debug(f'  Max offset V: {max_offset_vu[0]}, U: {max_offset_vu[1]}')

    # Coarse-to-fine prior sequence.  Each level passes its result as the prior
    # for the next finer level so that a bad coarse estimate is penalised at
    # the finer scale rather than allowed to set an unconstrained starting point.
    level_shifts = []
    coarser_prior_fullres: tuple[float, float] | None = None
    for lvl in range(pyramid_levels, 0, -1):
        s = 2 ** (lvl - 1)
        # Anti-alias the image prior to decimation to avoid peak shifts at coarse levels
        image_blurred = gaussian_filter(image, sigma=s / 2.0)
        image_downsampled = image_blurred[::s, ::s]

        # Downsample model & mask with anti-aliasing
        # First blur them both with an appropriate Gaussian kernel so that simple downsampling
        # can be used.
        sigma = s / 2.0
        model_blurred = gaussian_filter(model, sigma=sigma)
        mask_blurred = gaussian_filter(mask.astype(float), sigma=sigma)
        model_downsampled = model_blurred[::s, ::s]
        mask_downsampled = mask_blurred[::s, ::s] > 0.5

        data_mask_downsampled: NDArrayBoolType | None = None
        if data_mask is not None:
            data_mask_blurred = gaussian_filter(data_mask.astype(float), sigma=sigma)
            data_mask_downsampled = data_mask_blurred[::s, ::s] > 0.5

        # Scale the full-res prior down to the current pyramid level's coordinates.
        prior_at_scale: tuple[float, float] | None = (
            (coarser_prior_fullres[0] / s, coarser_prior_fullres[1] / s)
            if coarser_prior_fullres is not None
            else None
        )

        # Scale the max offset limit to the current pyramid level.  Use ceiling
        # division so a fractional full-res margin still maps to a non-zero window
        # at coarse scales; clamp each component to at least 1 pixel.
        max_offset_at_scale: tuple[int, int] | None = (
            (
                max(1, math.ceil(max_offset_vu[0] / s)),
                max(1, math.ceil(max_offset_vu[1] / s)),
            )
            if max_offset_vu is not None
            else None
        )

        # At the coarsest level (no prior), a single peak suffices.
        # At finer levels, evaluate multiple candidates so the prior
        # from the coarser level can steer selection toward the correct peak.
        level_max_peaks = 1 if coarser_prior_fullres is None else max_peaks

        res_lvl = navigate_single_scale_kpeaks(
            image=image_downsampled,
            model=model_downsampled,
            mask=mask_downsampled,
            max_peaks=level_max_peaks,
            upsample_factor=upsample_factor,
            metric=metric,
            prior_shift=prior_at_scale,
            # When ``coarser_prior_fullres`` is still None (coarsest pyramid level),
            # ``prior_at_scale`` is None: set ``prior_weight`` to 0.0 so the coarsest
            # level's quality score has no prior penalty. Finer levels pass
            # ``prior_shift`` and use ``prior_weight_final``.
            prior_weight=prior_weight_final if prior_at_scale is not None else 0.0,
            nms_radius=nms_radius,
            max_offset_vu=max_offset_at_scale,
            data_mask=data_mask_downsampled,
            use_gradient=use_gradient,
            logger=logger,
        )
        logger.debug(
            f'Correlation pyramid level {lvl} results: '
            f'offset {res_lvl["offset"][0] * s:.3f}, {res_lvl["offset"][1] * s:.3f}; '
            f'sigma_xy {res_lvl["sigma_xy"][0] * s:.3f}, {res_lvl["sigma_xy"][1] * s:.3f}; '
            f'quality {res_lvl["quality"]:.3f}; '
            f'peak_val {res_lvl["peak_val"]:.3f}'
        )

        # Scale shift back to full res and record for the next finer level.
        shift_fullres = (res_lvl['offset'][0] * s, res_lvl['offset'][1] * s)
        level_shifts.append(shift_fullres)
        coarser_prior_fullres = shift_fullres

    # Consistency: max deviation to last level's shift
    shifts_arr = np.array(level_shifts, dtype=np.float64)
    final_prior = shifts_arr[-1]
    consistency = float(np.max(np.linalg.norm(shifts_arr - final_prior, axis=1)))
    logger.debug(f'Correlation final prior: {final_prior}')
    logger.debug(f'Correlation consistency: {consistency}')
    logger.debug('Performing final correlation pass')

    # Final level: K-peak evaluation with prior penalty
    result = navigate_single_scale_kpeaks(
        image=image,
        model=model,
        mask=mask,
        max_peaks=max_peaks,
        upsample_factor=upsample_factor,
        metric=metric,
        prior_shift=final_prior,
        prior_weight=prior_weight_final,
        nms_radius=nms_radius,
        max_offset_vu=max_offset_vu,
        data_mask=data_mask,
        use_gradient=use_gradient,
        # Band-limit only the full-resolution sub-pixel refinement; the coarse
        # pyramid levels keep their sharp surfaces for integer-peak selection.
        refine_lowpass_sigma_px=refine_lowpass_sigma_px,
        logger=logger,
    )

    at_edge = False
    if max_offset_vu is not None:
        # 2-pixel margin: the integer NMS argmax can land at |signed| = max - 1
        # (one row inside the strict-inequality exclusion), and subpixel
        # refinement can nudge the reported offset another 0.5 pixel further
        # in. Flag anything within 2 px of the boundary as an unreliable
        # boundary peak.
        at_edge = (
            abs(result['offset'][0]) >= max_offset_vu[0] - 2.0
            or abs(result['offset'][1]) >= max_offset_vu[1] - 2.0
        )
    spurious = (result['quality'] < quality_thresh) or (consistency > consistency_tol) or at_edge
    if at_edge:
        logger.debug(
            'Correlation peak within 2 pixels of max-offset window edge; marking result spurious'
        )

    # Surface the per-peak telemetry from the final-pass single-scale
    # call so callers can derive a peak-to-runner-up ratio without
    # re-running the correlation.  Each entry is
    # ``(quality, offset_dv, offset_du)``; the winner is index 0 and
    # any runner-ups follow in descending quality order.
    all_candidates = result.get('all_candidates', [])
    top_k_peaks: list[tuple[float, float, float]] = [
        (
            float(c['quality']),
            float(c['offset'][0]),
            float(c['offset'][1]),
        )
        for c in all_candidates
    ]

    # Add the localization-spread uncertainty.  The peak-curvature
    # covariance measures the statistical precision at the winning peak
    # only; a peak that migrates across pyramid levels is empirically less
    # trustworthy than a stable one, so fold ``(scale * consistency)**2``
    # into the translation covariance diagonal (no-op when the scale is
    # 0.0 or the peak is perfectly consistent).
    cov_out = np.asarray(result['cov'], np.float64).copy()
    sigma_xy_out = result['sigma_xy']
    if localization_uncertainty_scale > 0.0 and cov_out.shape == (2, 2):
        localization_var = (localization_uncertainty_scale * consistency) ** 2
        cov_out[0, 0] += localization_var
        cov_out[1, 1] += localization_var
        sigma_xy_out = (
            float(np.sqrt(cov_out[0, 0])),
            float(np.sqrt(cov_out[1, 1])),
        )

    ret = {
        'offset': result['offset'],
        'cov': cov_out,
        'sigma_xy': sigma_xy_out,
        'quality': result['quality'],
        'metric': metric,
        'consistency': consistency,
        'spurious': bool(spurious),
        'at_edge': bool(at_edge),
        'used_gradient': bool(use_gradient),
        'top_k_peaks': top_k_peaks,
    }

    logger.debug(
        f'Correlation result: '
        f'offset dU {result["offset"][1]:.3f}, dV {result["offset"][0]:.3f}; '
        f'sigma U {sigma_xy_out[1]:.3f}, V {sigma_xy_out[0]:.3f}; '
        f'consistency {consistency:.3f}; '
        f'spurious {spurious}'
    )

    return ret
