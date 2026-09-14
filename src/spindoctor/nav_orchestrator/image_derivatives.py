"""Shared image-side derivatives consumed by every DT-based technique.

The orchestrator computes one gradient-magnitude image, one gradient-vector
image, and one edge distance-transform image per navigation; every limb /
terminator / ring-edge technique then samples those products at its own
model polylines.  Computing them once keeps the per-image cost bounded
regardless of how many DT techniques run.

Three quantities are produced and attached to the per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`:

``image_gradient_ext``
    Gradient magnitude of the source image after a Gaussian blur at
    :attr:`ImageDerivativesConfig.image_gradient_sigma_px`.  Used by the
    technique-side polarity / gating logic when only a magnitude readout is
    needed.

``image_gradient_vu_ext``
    Per-pixel ``(g_v, g_u)`` gradient vector after the same Gaussian blur.
    Sampled by the polarity filter to compare the image's edge direction
    with the model's outward normal at each polyline vertex.

``image_edge_dt_ext``
    Euclidean distance transform of the binarised gradient image, with the
    threshold chosen as ``edge_threshold_k_sigma * image_noise_sigma``.  The DT
    is truncated at :data:`DEFAULT_DT_HALF_WIDTH_PX` so the per-pixel cost is
    bounded for the DT-based techniques' Levenberg-Marquardt step.  It is not
    signed: the values are non-negative everywhere and the zero locus is the
    edge pixels themselves, at their own centers, not an oriented boundary
    running along pixel edges half a pixel away.

The thresholding intentionally treats *every* edge pixel as a candidate; the
per-technique polarity filter rejects matches that disagree on the gradient
direction.  A non-empty gradient image always produces a fully-defined DT
because :func:`build_image_edge_dt` falls back to a pure-saturation array
when no pixel exceeds the threshold.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from spindoctor.config import logged_section
from spindoctor.support.filters import NavFilterKind, NavFilterSpec, apply_filter
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DEFAULT_DT_HALF_WIDTH_PX',
    'DEFAULT_EDGE_THRESHOLD_K_SIGMA',
    'DEFAULT_IMAGE_GRADIENT_SIGMA_PX',
    'ImageDerivativesConfig',
    'build_image_edge_dt',
    'compute_all_image_derivatives',
    'compute_image_gradient_vu',
]


DEFAULT_IMAGE_GRADIENT_SIGMA_PX: float = 1.2
"""Default Gaussian sigma (pixels) used to smooth before the gradient operator.

A 1.2 pixel sigma matches the typical instrument PSF; below that the
gradient is dominated by single-pixel noise, above it sharp limbs blur out
and the DT loses contrast against the background.
"""


DEFAULT_EDGE_THRESHOLD_K_SIGMA: float = 4.0
"""Default gradient-magnitude threshold expressed as multiples of MAD noise.

Pixels whose smoothed-gradient magnitude exceeds
``edge_threshold_k_sigma * image_noise_sigma`` are kept as edge candidates.
A 4-sigma threshold keeps single-pixel noise excursions out of the DT
input while letting limb / terminator / ring edges through with margin
to spare.
"""


DEFAULT_DT_HALF_WIDTH_PX: float = 64.0
"""Maximum distance returned by the truncated distance transform.

Pixels farther than this from any thresholded gradient pixel saturate at
``DEFAULT_DT_HALF_WIDTH_PX``.  The cap bounds the LM cost contribution from
vertices that fall in completely-empty regions of the image and bounds the
DT array's working range to a documented maximum.
"""


@dataclass(frozen=True)
class ImageDerivativesConfig:
    """Configuration for the shared gradient + DT computation.

    Parameters:
        image_gradient_sigma_px: Gaussian sigma (pixels) used before the
            gradient operator.  Both axes share the same value;
            anisotropic blur is intentionally not exposed here because the
            image-side computation must be feature-agnostic.
        edge_threshold_k_sigma: Multiples of ``image_noise_sigma`` used to
            threshold the gradient magnitude into a binary edge mask.
        dt_half_width_px: Cap on the DT distance.  Vertices farther than
            this from any edge pixel see a constant cost.

    Raises:
        ValueError: if any field is not strictly positive.
    """

    image_gradient_sigma_px: float = DEFAULT_IMAGE_GRADIENT_SIGMA_PX
    edge_threshold_k_sigma: float = DEFAULT_EDGE_THRESHOLD_K_SIGMA
    dt_half_width_px: float = DEFAULT_DT_HALF_WIDTH_PX

    def __post_init__(self) -> None:
        """Validate the three positive-real parameters."""
        if not (math.isfinite(self.image_gradient_sigma_px) and self.image_gradient_sigma_px > 0.0):
            raise ValueError(
                'image_gradient_sigma_px must be a finite positive number; '
                f'got {self.image_gradient_sigma_px!r}'
            )
        if not (math.isfinite(self.edge_threshold_k_sigma) and self.edge_threshold_k_sigma > 0.0):
            raise ValueError(
                'edge_threshold_k_sigma must be a finite positive number; '
                f'got {self.edge_threshold_k_sigma!r}'
            )
        if not (math.isfinite(self.dt_half_width_px) and self.dt_half_width_px > 0.0):
            raise ValueError(
                f'dt_half_width_px must be a finite positive number; got {self.dt_half_width_px!r}'
            )


def _smooth_and_compute_gradients(
    image_ext: NDArrayFloatType,
    sigma_px: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Smooth ``image_ext`` and return ``(g_v, g_u)`` Sobel gradients.

    Shared core of :func:`build_image_edge_dt`,
    :func:`compute_image_gradient_vu`, and
    :func:`compute_all_image_derivatives`.  Each public entry point
    delegates the gaussian + sobel pass to this helper so the heavy
    smoothing only runs once per image when the orchestrator uses the
    combined entry point.

    Parameters:
        image_ext: Extended-FOV image array.  Must be 2-D and finite.
        sigma_px: Gaussian sigma (pixels).  Must be strictly positive.

    Returns:
        Tuple ``(gv, gu)`` of float64 arrays shaped like ``image_ext``.

    Raises:
        TypeError: if ``image_ext`` is not 2-D.
        ValueError: if ``image_ext`` contains NaN or +/-inf, or if
            ``sigma_px`` is not strictly positive.
    """
    if image_ext.ndim != 2:
        raise TypeError(f'image_ext must be 2-D; got ndim={image_ext.ndim}')
    if not np.isfinite(image_ext).all():
        raise ValueError(
            'image_ext must contain only finite values; NaN or +/-inf would '
            'propagate through gaussian_filter / sobel and poison every '
            'derivative downstream'
        )
    if not (math.isfinite(sigma_px) and sigma_px > 0.0):
        raise ValueError(f'sigma_px must be a finite positive number; got {sigma_px!r}')
    smoothed = gaussian_filter(image_ext.astype(np.float64), sigma=(sigma_px, sigma_px))
    gv = sobel(smoothed, axis=0, mode='constant', cval=0.0)
    gu = sobel(smoothed, axis=1, mode='constant', cval=0.0)
    return cast(NDArrayFloatType, gv.astype(np.float64, copy=False)), cast(
        NDArrayFloatType, gu.astype(np.float64, copy=False)
    )


def _build_edge_dt_from_gradients(
    gv: NDArrayFloatType,
    gu: NDArrayFloatType,
    *,
    edge_threshold_k_sigma: float,
    dt_half_width_px: float,
    image_noise_sigma: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return ``(gradient_magnitude, edge_dt)`` from precomputed ``(gv, gu)``.

    The thresholding intentionally treats *every* edge pixel as a
    candidate; the per-technique polarity filter rejects matches that
    disagree on the gradient direction.  Empty thresholded masks fall
    back to a saturated DT so downstream consumers always see a
    fully-defined array.

    Parameters:
        gv: Per-pixel ``g_v`` Sobel-of-Gaussian derivative.
        gu: Per-pixel ``g_u`` Sobel-of-Gaussian derivative.
        edge_threshold_k_sigma: Multiples of ``image_noise_sigma`` used as
            the gradient-magnitude threshold.
        dt_half_width_px: Cap on the truncated distance transform.
        image_noise_sigma: MAD noise sigma for the threshold scaling.

    Returns:
        ``(gradient_magnitude, edge_dt)`` float64 arrays.
    """
    gradient = np.hypot(gv, gu).astype(np.float64)
    # Clamp the noise sigma to a tiny positive floor so a literal ``0.0`` sigma
    # (permitted by the ``>= 0`` validation -- e.g. synthetic or saturated-flat
    # inputs) does not collapse the threshold to ``0`` and promote every
    # non-zero-gradient pixel to an edge candidate, which would degenerate the
    # DT toward all-zero.  Mirrors the ``cr_noise_sigma`` clamp in
    # ``_make_context``.
    threshold = edge_threshold_k_sigma * max(image_noise_sigma, 1e-6)
    # Thin the gradient ridge toward a one-pixel-wide edge map via Canny-style
    # non-maximum suppression along the local gradient direction.  A naive
    # 3x3 NMS would discard most edge pixels along a smooth ridge; the
    # directional check compares each candidate only against the two
    # neighbors along its own gradient direction, leaving the full edge
    # length intact and producing a usable input for both the
    # cross-correlation coarse search and the distance transform.  (On a flat
    # plateau of exactly-equal gradient magnitude the ``>=`` tie-break keeps
    # both sides, so a plateau edge may be two pixels wide -- harmless for DT
    # input.)
    edge_mask = _directional_nms(gradient, gv, gu, threshold)
    edge_dt = apply_filter(
        edge_mask.astype(np.float64, copy=False),
        NavFilterSpec(
            kind=NavFilterKind.DISTANCE_TRANSFORM,
            dt_half_width_px=dt_half_width_px,
        ),
    )
    return cast(NDArrayFloatType, gradient), edge_dt


def build_image_edge_dt(
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    *,
    config: ImageDerivativesConfig | None = None,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Compute the shared gradient and edge-distance-transform images.

    Two independent products are returned, both aligned to ``image_ext``:

    1. The gradient magnitude after a Gaussian smooth at
       ``config.image_gradient_sigma_px``.
    2. The distance transform of the thresholded gradient image, where the
       threshold is ``config.edge_threshold_k_sigma * image_noise_sigma``
       and the result is truncated at ``config.dt_half_width_px``.

    For callers (the orchestrator) that need *both* this product and the
    gradient-vector product, prefer :func:`compute_all_image_derivatives`
    so the heavy smoothing + Sobel pass runs once instead of twice.

    Parameters:
        image_ext: Extended-FOV image array (post any source-image filter).
            Must be 2-D.
        image_noise_sigma: MAD-derived noise sigma (image-native units: DN or I/F) used to scale
            the gradient threshold.  Must be finite and non-negative.
        config: Optional override; when ``None`` the documented defaults
            apply.

    Returns:
        Tuple ``(gradient_ext, edge_dt_ext)`` of float64 arrays, each
        shaped like ``image_ext``.

    Raises:
        TypeError: if ``image_ext`` is not 2-D.
        ValueError: if ``image_ext`` contains NaN or +/-inf, or if
            ``image_noise_sigma`` is negative or non-finite.
    """
    if not (math.isfinite(image_noise_sigma) and image_noise_sigma >= 0.0):
        raise ValueError(f'image_noise_sigma must be finite and >= 0; got {image_noise_sigma!r}')
    cfg = config if config is not None else ImageDerivativesConfig()
    gv, gu = _smooth_and_compute_gradients(image_ext, cfg.image_gradient_sigma_px)
    return _build_edge_dt_from_gradients(
        gv,
        gu,
        edge_threshold_k_sigma=cfg.edge_threshold_k_sigma,
        dt_half_width_px=cfg.dt_half_width_px,
        image_noise_sigma=image_noise_sigma,
    )


def _directional_nms(
    gradient_magnitude: NDArrayFloatType,
    gv: NDArrayFloatType,
    gu: NDArrayFloatType,
    threshold: float,
) -> NDArrayFloatType:
    """Return a Canny-style thin edge mask in float form (0.0 / 1.0).

    Each pixel above ``threshold`` is kept only if its magnitude is at
    least as large as both of its neighbors along the local gradient
    direction.  The gradient direction is quantized to four cardinal
    sectors (0, 45, 90, 135 degrees from horizontal) so the lookup
    reduces to a small fixed set of shifts.

    Parameters:
        gradient_magnitude: 2-D non-negative gradient magnitude image.
        gv: 2-D Sobel-of-Gaussian derivative along v.
        gu: 2-D Sobel-of-Gaussian derivative along u.
        threshold: Magnitude threshold; pixels at or below this value
            are discarded regardless of NMS outcome.

    Returns:
        2-D float64 mask with 1.0 at retained edge pixels and 0.0
        elsewhere.
    """
    abs_gv = np.abs(gv)
    abs_gu = np.abs(gu)
    # Standard Canny quantisation splits the half-circle of gradient
    # directions into four 45-degree sectors with boundaries at 22.5,
    # 67.5, 112.5, and 157.5 degrees from the u-axis.  A direction
    # belongs to the horizontal sector when |angle| <= 22.5 deg, i.e.
    # |gu| >= |gv| * cot(22.5 deg).  cot(22.5 deg) = 1 + sqrt(2).
    cot_22_5 = 2.414213562373095
    sector_horizontal = abs_gu >= abs_gv * cot_22_5
    sector_vertical = abs_gv >= abs_gu * cot_22_5
    sector_diag_pos = (~sector_horizontal) & (~sector_vertical) & (gv * gu >= 0)
    sector_diag_neg = (~sector_horizontal) & (~sector_vertical) & (gv * gu < 0)
    pad = np.pad(gradient_magnitude, 1, mode='constant', constant_values=-np.inf)
    center = pad[1:-1, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    up_left = pad[:-2, :-2]
    down_right = pad[2:, 2:]
    up_right = pad[:-2, 2:]
    down_left = pad[2:, :-2]
    keep_horizontal = sector_horizontal & (center >= left) & (center >= right)
    keep_vertical = sector_vertical & (center >= up) & (center >= down)
    keep_diag_pos = sector_diag_pos & (center >= up_left) & (center >= down_right)
    keep_diag_neg = sector_diag_neg & (center >= up_right) & (center >= down_left)
    keep = keep_horizontal | keep_vertical | keep_diag_pos | keep_diag_neg
    edge_mask = np.where(keep & (gradient_magnitude > threshold), 1.0, 0.0).astype(np.float64)
    return cast(NDArrayFloatType, edge_mask)


def compute_image_gradient_vu(
    image_ext: NDArrayFloatType,
    *,
    sigma_px: float = DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
) -> NDArrayFloatType:
    """Compute the per-pixel ``(g_v, g_u)`` gradient vector image.

    The image is first smoothed with an isotropic Gaussian of standard
    deviation ``sigma_px``; gradients are then computed by Sobel along each
    axis.  The output preserves the sign of the gradient — required by the
    polarity filter, which compares the image gradient direction to the
    model's outward normal.

    For callers (the orchestrator) that need this product alongside the
    gradient magnitude / DT, prefer :func:`compute_all_image_derivatives`
    so the heavy smoothing + Sobel pass runs once instead of twice.

    Parameters:
        image_ext: Extended-FOV image array.  Must be 2-D.
        sigma_px: Gaussian sigma (pixels) used before the Sobel operator.
            Must be strictly positive.

    Returns:
        ``(H, W, 2)`` float64 array; the last axis stacks the v-component
        and the u-component of the gradient (``out[..., 0]`` is ``g_v``,
        ``out[..., 1]`` is ``g_u``).

    Raises:
        TypeError: if ``image_ext`` is not 2-D.
        ValueError: if ``image_ext`` contains NaN or +/-inf, or if
            ``sigma_px`` is not strictly positive.
    """
    gv, gu = _smooth_and_compute_gradients(image_ext, sigma_px)
    out = np.stack([gv, gu], axis=-1).astype(np.float64)
    return cast(NDArrayFloatType, out)


@logged_section('image_derivatives', 'IMAGE DERIVATIVES')
def compute_all_image_derivatives(
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    *,
    config: ImageDerivativesConfig | None = None,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Compute every image-side derivative the orchestrator needs in one pass.

    Returns the same products that :func:`build_image_edge_dt` and
    :func:`compute_image_gradient_vu` produce separately, but shares the
    single gaussian + sobel pass between them.  The orchestrator's
    per-image setup uses this entry point so the heavy smoothing only
    runs once even though three derivative products end up on the
    :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`.

    Parameters:
        image_ext: Extended-FOV image array (post any source-image
            filter).  Must be 2-D.
        image_noise_sigma: MAD-derived noise sigma (image-native units: DN or I/F) used to
            scale the gradient threshold.  Must be finite and
            non-negative.
        config: Optional override; when ``None`` the documented defaults
            apply.

    Returns:
        Tuple ``(gradient_ext, edge_dt_ext, gradient_vu_ext)``:

        - ``gradient_ext`` — gradient magnitude shaped ``(H, W)``.
        - ``edge_dt_ext`` — truncated distance transform of the
          thresholded edge mask, shaped ``(H, W)``.
        - ``gradient_vu_ext`` — signed ``(g_v, g_u)`` gradient vector
          image, shaped ``(H, W, 2)``.

    Raises:
        TypeError: if ``image_ext`` is not 2-D.
        ValueError: if ``image_ext`` contains NaN or +/-inf, or if
            ``image_noise_sigma`` is negative or non-finite.
    """
    if not (math.isfinite(image_noise_sigma) and image_noise_sigma >= 0.0):
        raise ValueError(f'image_noise_sigma must be finite and >= 0; got {image_noise_sigma!r}')
    cfg = config if config is not None else ImageDerivativesConfig()
    gv, gu = _smooth_and_compute_gradients(image_ext, cfg.image_gradient_sigma_px)
    gradient, edge_dt = _build_edge_dt_from_gradients(
        gv,
        gu,
        edge_threshold_k_sigma=cfg.edge_threshold_k_sigma,
        dt_half_width_px=cfg.dt_half_width_px,
        image_noise_sigma=image_noise_sigma,
    )
    gradient_vu = np.stack([gv, gu], axis=-1).astype(np.float64)
    return gradient, edge_dt, cast(NDArrayFloatType, gradient_vu)
