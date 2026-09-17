"""Filter abstraction used by feature extractors and matching techniques.

Each NavFeature carries a ``preferred_filter`` ``NavFilterSpec``; the
consuming technique applies the spec to both the image patch and the model
template before computing its matching metric.  A small number of kinds
covers every feature type used by the v1 pipeline.

The ``apply_filter`` entry point dispatches on ``NavFilterKind`` and runs the
configured operation, with two universal short-circuits:

- ``NavFilterKind.NONE`` returns the input unchanged.
- A spec whose largest principal sigma is below
  ``null_filter_threshold_sigma`` (per-config) is treated as ``NONE``.

Higher-level technique code never indexes by kind itself; it just calls
``apply_filter(arr, spec)``.

Thread safety: all functions in this module are pure / stateless; safe for
concurrent use on independent inputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    grey_dilation,
    rotate,
    sobel,
)

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'NavFilterKind',
    'NavFilterSpec',
    'apply_filter',
]


class NavFilterKind(Enum):
    """The kind of operation a NavFilterSpec describes.

    - ``NONE``: identity filter; ``apply_filter`` returns the input unchanged.
    - ``ISOTROPIC_GAUSSIAN``: symmetric Gaussian blur with a single ``sigma_xy``.
    - ``ANISOTROPIC_GAUSSIAN``: possibly axis-aligned Gaussian blur with a
      full 2x2 covariance.  ``align_axis`` may rotate the blur into a
      non-axis-aligned principal frame.
    - ``BANDPASS_DOG``: difference-of-Gaussians bandpass; subtract a heavy-blur
      from a light-blur of the input to suppress low-frequency content while
      preserving sharper detail.
    - ``DISTANCE_TRANSFORM``: Euclidean distance transform of a thresholded
      edge map.  It is NOT signed: wherever there is an edge to measure from,
      the values are non-negative and the zero locus is the edge pixels
      themselves, at their own centers, rather than an oriented boundary
      running along pixel edges half a pixel away.  Truncated at
      ``dt_half_width_px`` when that is positive and left uncapped when it is
      not.  An input with no edge at all is the one case that returns something
      other than a distance: every pixel comes back at ``dt_half_width_px``,
      which is negative if that is how it was set.  Only meaningful as a
      precomputed image-side quantity, not a generic operator.  ``apply_filter``
      thresholds whatever it is handed at ``!= 0`` rather than rejecting a
      non-binary array.
    - ``GRADIENT_OF_GAUSSIAN``: gradient magnitude of a Gaussian-smoothed
      input; isotropic blur followed by Sobel.
    - ``MORPH_DILATE``: morphological dilation by a structuring element of
      half-width derived from ``sigma_xy``.  Used when building search margins
      for edge-based matching.
    """

    NONE = 'NONE'
    ISOTROPIC_GAUSSIAN = 'ISOTROPIC_GAUSSIAN'
    ANISOTROPIC_GAUSSIAN = 'ANISOTROPIC_GAUSSIAN'
    BANDPASS_DOG = 'BANDPASS_DOG'
    DISTANCE_TRANSFORM = 'DISTANCE_TRANSFORM'
    GRADIENT_OF_GAUSSIAN = 'GRADIENT_OF_GAUSSIAN'
    MORPH_DILATE = 'MORPH_DILATE'


@dataclass(frozen=True, eq=False)
class NavFilterSpec:
    """Description of a filter to be applied uniformly to image and template.

    All fields except ``kind`` are optional; only the ones the chosen kind
    consumes need to be set.  Tests assert that mismatched kind/parameter
    combinations raise on application, not at construction (so techniques
    can carry under-populated specs through identity short-circuits without
    extra ceremony).

    Parameters:
        kind: Which kind of filter operation this spec describes.
        sigma_xy: Per-axis Gaussian sigma in pixels, used by
            ISOTROPIC_GAUSSIAN, GRADIENT_OF_GAUSSIAN, and MORPH_DILATE.
        covariance_px2: 2x2 covariance matrix used by ANISOTROPIC_GAUSSIAN.
            ``None`` for other kinds.
        bandpass_cutoffs_px: ``(lo_sigma, hi_sigma)`` in pixels for
            BANDPASS_DOG.  Only used by that kind.
        dt_half_width_px: Half-width truncation of distance values, in
            pixels.  Used only by DISTANCE_TRANSFORM.
        align_axis: Optional ``(v, u)`` direction to align the principal
            axis of an anisotropic filter.  ``None`` means axis-aligned
            (identity rotation).
    """

    kind: NavFilterKind
    sigma_xy: tuple[float, float] = (0.0, 0.0)
    covariance_px2: NDArrayFloatType | None = None
    bandpass_cutoffs_px: tuple[float, float] = (0.0, 0.0)
    dt_half_width_px: float = 0.0
    align_axis: tuple[float, float] | None = None
    null_filter_threshold_sigma: float = field(default=0.4, repr=False)


def _largest_sigma(spec: NavFilterSpec) -> float:
    """Return the largest principal sigma a spec specifies, in pixels.

    Used to decide whether the spec is below ``null_filter_threshold_sigma``
    and should be treated as identity.
    """
    if spec.kind in (
        NavFilterKind.ISOTROPIC_GAUSSIAN,
        NavFilterKind.GRADIENT_OF_GAUSSIAN,
        NavFilterKind.MORPH_DILATE,
    ):
        return float(max(spec.sigma_xy))
    if spec.kind is NavFilterKind.ANISOTROPIC_GAUSSIAN:
        if spec.covariance_px2 is None:
            return 0.0
        eigvals = np.linalg.eigvalsh(np.asarray(spec.covariance_px2, np.float64))
        # Sigma is sqrt of variance; largest sigma corresponds to largest
        # eigenvalue.  Clamp to >= 0 for floating-point noise on rank-1
        # inputs.
        return float(np.sqrt(max(float(eigvals.max()), 0.0)))
    if spec.kind is NavFilterKind.BANDPASS_DOG:
        return float(max(spec.bandpass_cutoffs_px))
    return 0.0


def _apply_anisotropic_gaussian(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply an axis-aligned anisotropic Gaussian blur from a 2x2 covariance.

    ``align_axis`` rotates the blur so its principal axes line up with the
    given direction; if ``None``, the covariance is interpreted axis-aligned
    in (v, u).

    Implementation note: scipy's ``gaussian_filter`` only supports
    axis-aligned sigmas.  When ``align_axis`` is provided, we rotate the
    array, blur, and rotate back.  This is acceptable for the small
    postage-stamp inputs that techniques actually pass in.
    """
    if spec.covariance_px2 is None:
        raise ValueError('ANISOTROPIC_GAUSSIAN requires covariance_px2')
    cov = np.asarray(spec.covariance_px2, np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f'ANISOTROPIC_GAUSSIAN covariance_px2 must be 2x2, got {cov.shape}')
    if spec.align_axis is None:
        sigma_v = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_u = float(np.sqrt(max(cov[1, 1], 0.0)))
        return cast(
            NDArrayFloatType,
            gaussian_filter(arr.astype(np.float64), sigma=(sigma_v, sigma_u)),
        )
    # Rotate to align axis, blur axis-aligned, rotate back.
    eigvals, eigvecs = np.linalg.eigh(cov)
    sigma_major = float(np.sqrt(max(eigvals[1], 0.0)))
    sigma_minor = float(np.sqrt(max(eigvals[0], 0.0)))
    # Rotation that aligns image axes with eigenvectors.
    angle_rad = float(np.arctan2(eigvecs[0, 1], eigvecs[1, 1]))
    rotated = rotate(arr.astype(np.float64), -np.degrees(angle_rad), reshape=False)
    blurred = gaussian_filter(rotated, sigma=(sigma_major, sigma_minor))
    # ``rotate(..., reshape=False)`` preserves the input shape, so the result is
    # already arr-shaped (no trim/pad needed).
    out = rotate(blurred, np.degrees(angle_rad), reshape=False)
    return cast(NDArrayFloatType, out)


def _apply_bandpass_dog(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply a difference-of-Gaussians bandpass.

    Subtracts a heavily-blurred copy from a lightly-blurred copy.  The result
    suppresses low-frequency content (stray-light gradients) while preserving
    everything sharper than the heavy-blur scale.
    """
    lo, hi = spec.bandpass_cutoffs_px
    if not (lo > 0.0 and hi > 0.0 and lo > hi):
        raise ValueError(
            f'BANDPASS_DOG requires bandpass_cutoffs_px=(lo, hi) with lo > hi > 0; got ({lo}, {hi})'
        )
    a = arr.astype(np.float64)
    light = gaussian_filter(a, sigma=hi)
    heavy = gaussian_filter(a, sigma=lo)
    return cast(NDArrayFloatType, light - heavy)


def _apply_gradient_of_gaussian(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply isotropic Gaussian blur followed by gradient-magnitude.

    Uses ``sigma_xy`` as a single isotropic sigma when its two entries match;
    otherwise blurs anisotropically before computing the gradient magnitude.
    """
    sigma_v, sigma_u = spec.sigma_xy
    a = arr.astype(np.float64)
    blurred = gaussian_filter(a, sigma=(sigma_v, sigma_u))
    gv = sobel(blurred, axis=0, mode='constant', cval=0.0)
    gu = sobel(blurred, axis=1, mode='constant', cval=0.0)
    return cast(NDArrayFloatType, np.hypot(gv, gu))


def _apply_morph_dilate(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply morphological dilation with a per-axis rectangular element.

    The structuring-element half-width on each axis is that axis's ``sigma_xy``
    component rounded up to an integer, so an anisotropic ``sigma_xy`` dilates
    each axis by a different amount instead of collapsing to a single square.
    """
    sigma_v, sigma_u = spec.sigma_xy
    half_v = int(np.ceil(sigma_v))
    half_u = int(np.ceil(sigma_u))
    if half_v <= 0 and half_u <= 0:
        return arr
    size = (2 * max(half_v, 0) + 1, 2 * max(half_u, 0) + 1)
    return cast(NDArrayFloatType, grey_dilation(arr, size=size))


def _apply_distance_transform(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply Euclidean distance transform of a thresholded input.

    Any 2-D float array is accepted: a pixel counts as an edge where it is
    non-zero, so a boolean or 0-1 array is the usual input rather than a
    required one.  The result is the distance from each pixel to the nearest
    edge pixel, zero on the edge pixels themselves.

    Truncation is conditional.  A positive ``dt_half_width_px`` caps the
    distances at that value; a zero or negative one leaves them uncapped, so a
    caller reading the result as a bounded quantity must set it.
    :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig`
    rejects a non-positive width, but this function is reachable without it.

    Parameters:
        arr: The array to threshold and transform.
        spec: Carries ``dt_half_width_px``, the truncation distance.

    Returns:
        The distances, as float64.  An array with no non-zero pixel has no
        edge to measure from, so every pixel is returned at
        ``dt_half_width_px`` -- including when that is zero or negative, which
        returns that value uniformly rather than a distance.
    """
    bin_arr = arr != 0
    if not bin_arr.any():
        # No edge pixels — every pixel is "infinitely far"; use the half-width
        # as the saturation value.
        return cast(
            NDArrayFloatType,
            np.full(arr.shape, spec.dt_half_width_px, dtype=np.float64),
        )
    dt = distance_transform_edt(~bin_arr)
    if spec.dt_half_width_px > 0:
        dt = np.minimum(dt, spec.dt_half_width_px)
    return cast(NDArrayFloatType, dt.astype(np.float64))


def apply_filter(arr: NDArrayFloatType, spec: NavFilterSpec) -> NDArrayFloatType:
    """Apply ``spec`` to ``arr`` and return the filtered array.

    Two universal short-circuits run before kind dispatch:

    1. ``spec.kind == NavFilterKind.NONE`` returns ``arr`` unchanged.
    2. If the largest principal sigma of ``spec`` is below
       ``spec.null_filter_threshold_sigma``, the spec is too small to make a
       meaningful difference; the array is returned unchanged.

    Otherwise the operation indicated by ``spec.kind`` is run.

    Parameters:
        arr: 2-D float input array.
        spec: NavFilterSpec describing the operation.

    Returns:
        Filtered 2-D float array; same shape as the input.

    Raises:
        ValueError: if ``spec`` is missing parameters required by its kind
            (e.g. ``ANISOTROPIC_GAUSSIAN`` without a covariance matrix).
        TypeError: if ``arr`` is not 2-D.
    """
    if arr.ndim != 2:
        raise TypeError(f'apply_filter requires a 2-D array; got ndim={arr.ndim}')
    if spec.kind is NavFilterKind.NONE:
        return arr
    # The null-sigma short-circuit (treat a tiny filter as identity, returning
    # the input unchanged) is valid ONLY for the blur-family kinds, where a
    # negligible blur really is ~identity.  It must NOT apply to
    # GRADIENT_OF_GAUSSIAN (whose output is a gradient magnitude, not the
    # intensity image), MORPH_DILATE (handled by its own half_width<=0 guard),
    # or DISTANCE_TRANSFORM (no "sigma" concept — half_width governs
    # saturation).  Short-circuiting those would silently change the meaning of
    # the result to the raw intensity image.
    _BLUR_FAMILY = (
        NavFilterKind.ISOTROPIC_GAUSSIAN,
        NavFilterKind.ANISOTROPIC_GAUSSIAN,
        NavFilterKind.BANDPASS_DOG,
    )
    if spec.kind in _BLUR_FAMILY and _largest_sigma(spec) < spec.null_filter_threshold_sigma:
        return arr
    if spec.kind is NavFilterKind.ISOTROPIC_GAUSSIAN:
        sigma_v, sigma_u = spec.sigma_xy
        return cast(
            NDArrayFloatType,
            gaussian_filter(arr.astype(np.float64), sigma=(sigma_v, sigma_u)),
        )
    if spec.kind is NavFilterKind.ANISOTROPIC_GAUSSIAN:
        return _apply_anisotropic_gaussian(arr, spec)
    if spec.kind is NavFilterKind.BANDPASS_DOG:
        return _apply_bandpass_dog(arr, spec)
    if spec.kind is NavFilterKind.GRADIENT_OF_GAUSSIAN:
        return _apply_gradient_of_gaussian(arr, spec)
    if spec.kind is NavFilterKind.MORPH_DILATE:
        return _apply_morph_dilate(arr, spec)
    if spec.kind is NavFilterKind.DISTANCE_TRANSFORM:
        return _apply_distance_transform(arr, spec)
    # Defensive: every NavFilterKind value is handled above; this branch
    # only fires if a new kind is added without updating the dispatcher.
    raise ValueError(f'Unknown NavFilterKind: {spec.kind!r}')
