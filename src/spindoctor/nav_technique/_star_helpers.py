"""Internal helpers shared across the three star NavTechniques.

The three star techniques (`StarUniqueMatchNav`, `StarRefineNav`,
`StarFieldFromCatalogNav`) all need the same handful of small
operations on STAR features and image patches: filtering by feature
type / flags, reading the predicted-SNR off the flags dataclass,
converting an SNR ratio to a magnitude difference, and pulling a
sub-pixel centroid from a small image window.  Defining these helpers
in one place avoids the cross-module private-helper imports that
otherwise tie three technique modules together through their
underscore-prefixed surface.

The submodule itself is private (leading underscore on the file
name); the helpers carry no underscore prefix because they are
public-but-internal — every star technique imports them, but they
are not part of the package's public API.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.ndimage import binary_dilation, label, maximum_filter

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model.stars.detection import matched_filter_image
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'SAME_SOURCE_PX',
    'SimilarityFit',
    'WindowSource',
    'brightness_margin_mag',
    'detection_peak_ratio',
    'distinct_stars',
    'gaussian_psf_kernel',
    'local_centroid',
    'predicted_snr',
    'predicted_vu',
    'search_window_bounds',
    'similarity_transform_fit',
    'star_features',
    'too_sharp',
    'usable_stars',
    'window_sources',
]

SAME_SOURCE_PX = 1.0
"""Centroids closer than this (px) are one source."""


def star_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the input subset that carries STAR geometry + StarFlags."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.STAR
        and isinstance(f.geometry, StarGeometry)
        and isinstance(f.flags, StarFlags)
    ]


def usable_stars(features: list[NavFeature]) -> list[NavFeature]:
    """Return STAR features that are not occluded or saturation-flagged.

    Mirrors the autonomous reliability gate's STAR posture: a star inside a
    body silhouette, ring annulus, or saturation/cosmic-ray mask is not a
    candidate for any star technique because the corresponding image pixel
    is dominated by other signal.
    """
    out: list[NavFeature] = []
    for f in star_features(features):
        flags = f.flags
        assert isinstance(flags, StarFlags)
        if flags.in_body_silhouette:
            continue
        if flags.in_saturation_or_cosmic_mask:
            continue
        out.append(f)
    return out


def distinct_stars(ranked: list[NavFeature]) -> list[NavFeature]:
    """Return stars, brightest first, without the ones another star stands for.

    Two catalog entries predicted within ``SAME_SOURCE_PX`` of each other (a star listed in
    two catalogs, or listed twice) are one source in the image, and counting it twice would
    count one detection, or one absence, as two.  The first of them in ``ranked`` is kept.

    Parameters:
        ranked: Star features, brightest predicted SNR first.

    Returns:
        The stars kept, in the same order.
    """
    kept: list[NavFeature] = []
    for star in ranked:
        pv, pu = predicted_vu(star)
        if all(
            math.hypot(pv - predicted_vu(k)[0], pu - predicted_vu(k)[1]) >= SAME_SOURCE_PX
            for k in kept
        ):
            kept.append(star)
    return kept


def predicted_snr(feature: NavFeature) -> float:
    """Return the predicted SNR carried on a STAR feature's flags."""
    flags = feature.flags
    assert isinstance(flags, StarFlags)
    return float(flags.predicted_snr)


def predicted_vu(feature: NavFeature) -> tuple[float, float]:
    """Return the predicted (v, u) carried on a STAR feature's geometry."""
    geometry = feature.geometry
    assert isinstance(geometry, StarGeometry)
    return geometry.predicted_vu


def brightness_margin_mag(brightest_snr: float, runner_up_snr: float) -> float:
    """Return the magnitude difference implied by an SNR ratio.

    For background-limited detection the measured SNR scales linearly with
    flux, so the magnitude difference between two stars whose predicted
    SNRs are ``s1`` (brighter) and ``s2`` (dimmer) is

    ::

        delta_mag = 2.5 * log10(s1 / s2)

    A ratio of 4 corresponds to ~1.5 mag, the default uniqueness floor.

    A non-positive ``brightest_snr`` is unpopulated / garbage input and
    is checked **first** so a zero-SNR cohort cannot accidentally be
    treated as "uniquely bright" — it returns ``0.0``.  Only after the
    brightest is known to carry signal does a non-positive
    ``runner_up_snr`` mean "no other predictable star competes with the
    brightest", which is reported as ``+inf``.
    """
    if brightest_snr <= 0.0:
        return 0.0
    if runner_up_snr <= 0.0:
        return float('inf')
    return 2.5 * math.log10(brightest_snr / runner_up_snr)


@dataclass(frozen=True)
class SimilarityFit:
    """Result of a 2-D similarity-transform fit aligning catalog -> detection.

    Maps a catalog point ``c`` to ``R(theta) @ c + translation`` in the
    image (``v, u``) frame — i.e. ``translation_vu`` is the global-frame
    translation, *not* a pivot-relative offset.  ``pivot_vu`` reports
    the weighted catalog centroid the SVD was centered on so callers can
    recover the equivalent pivot-relative form
    ``t_pivot = translation_vu + (R - I) @ pivot_vu`` when they need
    the rotation expressed about that pivot.

    Parameters:
        translation_vu: ``(dv, du)`` global-frame translation; equal
            to ``det_centroid - R @ cat_centroid``.
        rotation_rad: Rotation angle ``theta`` (radians).
        pivot_vu: Catalog-side weighted centroid used as the SVD
            centring point.  Reported so a downstream caller can emit
            a pivot-aware :class:`NavTechniqueResult` matching the
            project's rotation-pivot convention.
        residuals_vu: ``(N, 2)`` residual vector ``det - (R @ cat + t)``
            for each correspondence (raw, unweighted).
        weights: ``(N,)`` per-correspondence weight applied during the
            fit (Tukey biweight for the M-estimator path; uniform for
            two-point exact fits).
    """

    translation_vu: tuple[float, float]
    rotation_rad: float
    pivot_vu: tuple[float, float]
    residuals_vu: NDArrayFloatType
    weights: NDArrayFloatType


def similarity_transform_fit(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> SimilarityFit:
    """Solve the weighted Kabsch / orthogonal-Procrustes problem.

    Returns the ``(rotation, translation)`` that maps the catalog point
    cloud onto the detection point cloud minimizing the weighted squared
    residual sum.  The fit is rigid (no scale): the SVD's middle
    diagonal is replaced with ``diag(1, det(U @ Vt))`` so the result is
    a proper rotation even when one cohort happens to be a mirror image
    of the other (a numerical artifact for two-point inputs that the
    determinant correction rejects).

    The algorithm follows the textbook weighted-Procrustes derivation:

    1. Compute weighted centroids of both clouds.
    2. Form the weighted cross-covariance ``H = sum_i w_i (det_i - dc) (cat_i - cc).T``.
    3. SVD ``H = U S V.T``.
    4. ``R = U diag(1, det(U V.T)) V.T``.
    5. ``t = dc - R @ cc``.

    The pivot returned on the result is the catalog-side centroid
    ``cc``; that lets a caller emit a pivot-aware
    :class:`NavTechniqueResult` consistent with the project's
    rotation-pivot convention (a centroid-of-points pivot for star
    fits).

    Parameters:
        detection_pts: ``(N, 2)`` detected positions in ``(v, u)``.
        catalog_pts: ``(N, 2)`` predicted catalog positions in
            ``(v, u)``, in the same order as ``detection_pts``.
        weights: ``(N,)`` non-negative weights.  A near-zero total
            weight returns the identity transform with an empty
            residual vector.

    Returns:
        :class:`SimilarityFit`.

    Raises:
        ValueError: if the input shapes disagree or are not 2-D.
    """
    det = np.asarray(detection_pts, np.float64)
    cat = np.asarray(catalog_pts, np.float64)
    w = np.asarray(weights, np.float64)
    if det.ndim != 2 or det.shape[1] != 2:
        raise ValueError(f'detection_pts must have shape (N, 2); got {det.shape}')
    if cat.shape != det.shape:
        raise ValueError(
            f'catalog_pts must match detection_pts shape; got {cat.shape} vs {det.shape}'
        )
    if w.ndim != 1 or w.shape[0] != det.shape[0]:
        raise ValueError(f'weights must be 1-D of length {det.shape[0]}; got shape {w.shape}')
    if (w < 0.0).any():
        raise ValueError('weights must be non-negative')
    total = float(w.sum())
    if total <= 0.0:
        # No usable weight: report an identity transform with empty
        # residuals so callers can fall back to a translation-only path.
        return SimilarityFit(
            translation_vu=(0.0, 0.0),
            rotation_rad=0.0,
            pivot_vu=(float(cat[:, 0].mean()), float(cat[:, 1].mean())) if cat.size else (0.0, 0.0),
            residuals_vu=np.zeros_like(det),
            weights=w,
        )
    det_c_v = float(np.sum(w * det[:, 0]) / total)
    det_c_u = float(np.sum(w * det[:, 1]) / total)
    cat_c_v = float(np.sum(w * cat[:, 0]) / total)
    cat_c_u = float(np.sum(w * cat[:, 1]) / total)
    det_centered = det - np.asarray([det_c_v, det_c_u], np.float64)[None, :]
    cat_centered = cat - np.asarray([cat_c_v, cat_c_u], np.float64)[None, :]
    cross = (w[:, None] * det_centered).T @ cat_centered
    u_mat, _s, vt = np.linalg.svd(cross)
    det_correction = float(np.linalg.det(u_mat @ vt))
    sign_diag = np.eye(2, dtype=np.float64)
    sign_diag[1, 1] = math.copysign(1.0, det_correction)
    rotation = u_mat @ sign_diag @ vt
    theta = float(math.atan2(rotation[1, 0], rotation[0, 0]))
    pivot = (cat_c_v, cat_c_u)
    rotated_centroid = rotation @ np.asarray([cat_c_v, cat_c_u], np.float64)
    translation = (det_c_v - float(rotated_centroid[0]), det_c_u - float(rotated_centroid[1]))
    rotated_cat = cat @ rotation.T
    residuals = det - (rotated_cat + np.asarray(translation, np.float64)[None, :])
    return SimilarityFit(
        translation_vu=translation,
        rotation_rad=theta,
        pivot_vu=pivot,
        residuals_vu=cast(NDArrayFloatType, residuals),
        weights=w,
    )


def gaussian_psf_kernel(sigma_px: float, *, size: int) -> NDArrayFloatType:
    """Return a normalized 2-D isotropic Gaussian matched-filter kernel.

    Every star technique detects sources against this one kernel, so a star that
    one technique measures is the same star another technique measures.

    Parameters:
        sigma_px: PSF sigma in pixels.  Must be strictly positive.
        size: Side length of the (odd) square kernel.  Coerced up to
            the next odd integer when even.

    Returns:
        ``(size, size)`` float64 array, sum-to-unity.

    Raises:
        ValueError: if ``sigma_px`` is not strictly positive.
    """
    if sigma_px <= 0.0:
        raise ValueError(f'sigma_px must be > 0; got {sigma_px!r}')
    if size % 2 == 0:
        size += 1
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    vv, uu = np.meshgrid(coords, coords, indexing='ij')
    kernel = np.exp(-(vv * vv + uu * uu) / (2.0 * sigma_px * sigma_px))
    kernel /= float(kernel.sum())
    return kernel


def search_window_bounds(
    predicted_vu_pos: tuple[float, float],
    search_half_vu: tuple[float, float],
    shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    """Return the ``(v_lo, v_hi, u_lo, u_hi)`` slice bounds of a star search window.

    The window is centered on the pixel that contains the prediction (the working frame
    is pixel-centric, so that is the prediction rounded to the nearest integer, halves
    rounding up) and reaches ``floor(half)`` whole pixels either side of it along each
    axis, so each side is always ``2 * floor(half) + 1`` pixels long before clamping
    and its center never sits more than half a pixel from the prediction.  A window
    built from ``floor(v - W)`` and ``ceil(v + W)`` instead is one pixel wider than that
    for a non-integer prediction and reaches farther on one side than the other, with
    the longer side flipping as the prediction crosses an integer, so a sub-pixel change
    of prediction can admit or drop a whole row or column of candidates.

    Parameters:
        predicted_vu_pos: ``(v, u)`` prediction at the center of the window.
        search_half_vu: ``(v, u)`` half-widths of the window in pixels.
        shape: Shape of the image the window is cut from; the bounds are clamped to it.

    Returns:
        ``(v_lo, v_hi, u_lo, u_hi)`` half-open slice bounds, or ``None`` when the clamped
        window is empty (the prediction lies wholly off the image).
    """
    half_v = math.floor(search_half_vu[0])
    half_u = math.floor(search_half_vu[1])
    center_v = math.floor(predicted_vu_pos[0] + 0.5)
    center_u = math.floor(predicted_vu_pos[1] + 0.5)
    h, w = int(shape[0]), int(shape[1])
    v_lo = max(0, center_v - half_v)
    u_lo = max(0, center_u - half_u)
    v_hi = min(h, center_v + half_v + 1)
    u_hi = min(w, center_u + half_u + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return None
    return v_lo, v_hi, u_lo, u_hi


def _window_peaks(
    image_ext: NDArrayFloatType,
    bounds: tuple[int, int, int, int],
    *,
    kernel: NDArrayFloatType,
) -> NDArrayFloatType:
    """Return the matched-filter response at the window's peaks, ``-inf`` elsewhere.

    A peak is a pixel whose response is the largest within the kernel's footprint
    around it, the local-maximum rule ``StarFieldFromCatalogNav`` detects with.  The
    rule is judged with the image outside the window in view, so a pixel on the edge of
    the window that lies on the flank of a source outside it is no peak: a source whose
    own peak lies outside the window is not in it.  The response is computed on the
    window padded by twice the kernel's half-width (clamped to the image), so every
    response value the rule reads sees the neighborhood a whole-image filter would.

    Parameters:
        image_ext: 2-D extfov image array.
        bounds: ``(v_lo, v_hi, u_lo, u_hi)`` from :func:`search_window_bounds`.
        kernel: Matched-filter kernel from :func:`gaussian_psf_kernel`.

    Returns:
        Array with the window's shape holding the response at each peak and ``-inf`` at
        every other pixel.
    """
    v_lo, v_hi, u_lo, u_hi = bounds
    h, w = image_ext.shape
    size = kernel.shape[0]
    pad = 2 * (size // 2)
    pv_lo = max(0, v_lo - pad)
    pu_lo = max(0, u_lo - pad)
    pv_hi = min(h, v_hi + pad)
    pu_hi = min(w, u_hi + pad)
    padded = np.asarray(image_ext[pv_lo:pv_hi, pu_lo:pu_hi], np.float64)
    response = matched_filter_image(padded, kernel=kernel)
    local_max = maximum_filter(response, size=size, mode='nearest')
    peaks = np.where(response == local_max, response, -np.inf)
    return peaks[v_lo - pv_lo : v_hi - pv_lo, u_lo - pu_lo : u_hi - pu_lo]


_MAX_RECENTER = 5
_MAX_CANDIDATES = 16


def _box(image_ext: NDArrayFloatType, v: int, u: int, half: int) -> NDArrayFloatType:
    """Return the ``(2 * half + 1)`` box around pixel ``(v, u)``, clamped to the image."""
    h, w = image_ext.shape
    return np.asarray(
        image_ext[max(0, v - half) : min(h, v + half + 1), max(0, u - half) : min(w, u + half + 1)],
        np.float64,
    )


def too_sharp(
    image_ext: NDArrayFloatType,
    v: int,
    u: int,
    *,
    half: int,
    max_sharpness: float,
    noise_sigma: float,
    detection_sigma: float,
) -> bool:
    """Return whether the brightest pixel of a candidate's box is sharper than a star.

    The brightest pixel of the box around ``(v, u)`` is compared with the mean of its
    four edge neighbors, both above the box median.  A star spreads its light over its
    neighbors by at least what the narrowest point-spread function allows; a cosmic ray
    or hot pixel puts nearly all of it in one pixel.  The candidate is too sharp when the
    pixel's excess exceeds ``max_sharpness`` times the neighbors' excess by more than
    ``detection_sigma`` times the noise of that difference.  A pixel on the image's edge
    has no complete set of neighbors and is never judged too sharp.

    Parameters:
        image_ext: 2-D extfov image array.
        v: Row of the candidate.
        u: Column of the candidate.
        half: Half-width of the candidate's box in pixels.
        max_sharpness: Largest ratio of a star's peak excess to its neighbors' mean
            excess.
        noise_sigma: Robust per-pixel noise sigma in DN units.
        detection_sigma: Noise multiplier the difference must exceed.

    Returns:
        True when the candidate is too sharp to be a star.
    """
    h, w = image_ext.shape
    v_lo, u_lo = max(0, v - half), max(0, u - half)
    box = np.asarray(image_ext[v_lo : min(h, v + half + 1), u_lo : min(w, u + half + 1)])
    bv, bu = (int(x) for x in np.unravel_index(int(np.argmax(box)), box.shape))
    pv, pu = v_lo + bv, u_lo + bu
    if not (0 < pv < h - 1 and 0 < pu < w - 1):
        return False
    background = float(np.median(box))
    peak = float(image_ext[pv, pu]) - background
    neighbors = (
        float(image_ext[pv - 1, pu])
        + float(image_ext[pv + 1, pu])
        + float(image_ext[pv, pu - 1])
        + float(image_ext[pv, pu + 1])
    ) / 4.0 - background
    noise = noise_sigma * math.sqrt(1.0 + max_sharpness * max_sharpness / 4.0)
    return peak - max_sharpness * neighbors > detection_sigma * noise


def _box_centroid(
    image_ext: NDArrayFloatType, v: int, u: int, half: int
) -> tuple[float, float] | None:
    """Return the background-subtracted brightness-weighted centroid of one box.

    The background is the box median.  Returns ``None`` when nothing in the box rises
    above it.
    """
    h, w = image_ext.shape
    return _region_centroid(
        image_ext, (max(0, v - half), min(h, v + half + 1), max(0, u - half), min(w, u + half + 1))
    )


def _region_centroid(
    image_ext: NDArrayFloatType, region: tuple[int, int, int, int]
) -> tuple[float, float] | None:
    """Return the background-subtracted brightness-weighted centroid of a region.

    ``region`` is ``(v_lo, v_hi, u_lo, u_hi)``, upper bounds exclusive, inside the image.
    The background is the region's median.  Returns ``None`` when nothing in the region
    rises above it.
    """
    v_lo, v_hi, u_lo, u_hi = region
    box = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
    weights = np.clip(box - float(np.median(box)), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    vs = np.arange(v_lo, v_hi, dtype=np.float64)
    us = np.arange(u_lo, u_hi, dtype=np.float64)
    return (
        float(np.sum(vs[:, None] * weights) / total),
        float(np.sum(us[None, :] * weights) / total),
    )


def _flank_outshines(
    window: NDArrayFloatType,
    peaks: NDArrayFloatType,
    peak_dn: float,
    *,
    half: int,
) -> bool:
    """Return whether the window's brightest pixel is the flank of a source outside it.

    The pixel counts as such a flank when it is brighter than the detection's own
    peak, lies within ``half`` pixels of the window's edge, and has no peak of the
    window within ``half`` pixels of it.  Light from a source whose peak lies outside
    the window then outshines everything the window holds, and a detection taken
    inside it is a weaker thing chosen in the shadow of a brighter one.

    Parameters:
        window: The search window's pixels.
        peaks: The window's peak map from :func:`_window_peaks`.
        peak_dn: The brightest DN of the chosen detection's centroid box.
        half: Centroid-box (and kernel) half-width in pixels.

    Returns:
        ``True`` when the window should be refused.
    """
    flat_idx = int(np.argmax(window))
    bv, bu = (int(x) for x in np.unravel_index(flat_idx, window.shape))
    if float(window[bv, bu]) <= peak_dn:
        return False
    rows, cols = window.shape
    on_edge = min(bv, bu, rows - 1 - bv, cols - 1 - bu) < half
    if not on_edge:
        return False
    near = peaks[max(0, bv - half) : bv + half + 1, max(0, bu - half) : bu + half + 1]
    return not bool(np.isfinite(near).any())


def local_centroid(
    image_ext: NDArrayFloatType,
    predicted_vu_pos: tuple[float, float],
    *,
    search_half_vu: tuple[float, float],
    centroid_box_half_px: int,
    psf_sigma_px: float,
    max_peak_sharpness: float,
    image_noise_sigma: float,
    detection_sigma: float,
    refuse_outshone: bool,
) -> tuple[tuple[float, float] | None, float]:
    """Return the strongest local detection's centroid and its peak DN.

    Implements a small DAOPHOT-style detection inside a search window
    centered on the prediction:

    1. Cut the window of :func:`search_window_bounds` around the prediction.
    2. Matched-filter it against the Gaussian PSF kernel of
       :func:`gaussian_psf_kernel` (``psf_sigma_px``, side
       ``2 * centroid_box_half_px + 1``), the same detection statistic
       ``StarFieldFromCatalogNav`` uses, and take the response peaks (pixels
       whose response is the largest within the kernel's footprint) lying
       inside the window.  A window pixel on the flank of a source outside
       the window is no peak, so only a source whose own peak lies in the
       window can be measured.  The response weighs a source by the light it
       spreads over a PSF footprint and gives a uniform bright region no
       response at all.
    3. Take the strongest peak that is not sharper than a star
       (``max_peak_sharpness``, see :func:`too_sharp`).  The response alone
       does not keep a single-pixel spike (a cosmic ray or hot pixel) from
       outranking a star whose point-spread function is narrower than the
       kernel: against a star of sigma 0.54 px a spike wins once it exceeds
       about 1.7 times the star's peak.  The sharpness test skips such a spike,
       so the star it would have outranked is measured.
    4. Reject the candidate when the brightest pixel of its centroid box
       stands less than ``detection_sigma * image_noise_sigma`` above the
       box's median, so a noise bump on the floor of a bright extended
       region (a saturated ring or body) is no detection.
    5. With ``refuse_outshone``, reject the window when its brightest pixel
       is the edge-cut flank of a source outside it that outshines the
       candidate (:func:`_flank_outshines`).
    6. Fit a background-subtracted brightness-weighted moment over a
       ``(2 * centroid_box_half_px + 1)`` box around the candidate, and
       re-center the box on its own centroid until it stops moving (at
       most five times), so a box that first caught a source by its edge,
       as it does a saturated core, moves over the source.  A box that
       moves more than ``centroid_box_half_px`` from the candidate, or whose
       center leaves the window, has walked onto another source, and the
       window holds no detection.

    The star techniques use this purely-local fit rather than a global
    ``detect_sources`` call so they stay feasible on images where the
    global DAOPHOT pipeline would fail (mostly-empty FOV, dim secondary
    stars).

    Parameters:
        image_ext: 2-D extfov image array.
        predicted_vu_pos: ``(v, u)`` prediction at the center of the
            search window.
        search_half_vu: ``(v, u)`` half-widths of the search window in pixels.
        centroid_box_half_px: Half-width of the centroid-fit box (and of
            the matched-filter kernel) in pixels.
        psf_sigma_px: PSF sigma in pixels for the matched-filter kernel.
        max_peak_sharpness: Largest ratio of a star's brightest-pixel excess
            to the mean excess of its four edge neighbors; a sharper peak is a
            cosmic ray or hot pixel and is skipped.
        image_noise_sigma: Robust per-pixel noise sigma in DN units.
        detection_sigma: Threshold multiplier on
            ``image_noise_sigma``; the candidate's brightest pixel must stand
            ``detection_sigma * image_noise_sigma`` above its box's median to
            be accepted.
        refuse_outshone: Refuse a window outshone by a source outside it.
            ``StarRefineNav`` sets it: its small window sits on a prediction
            the pass-1 offset has already corrected, so light spilling in
            from outside means the star has left the window.
            ``StarUniqueMatchNav`` does not: its wide window often cuts a
            ring, whose flank would refuse windows that hold their star.

    Returns:
        ``(centroid, peak_dn)`` where ``centroid`` is the
        ``(v, u)`` sub-pixel center or ``None`` if the candidate was
        rejected.  ``peak_dn`` is the brightest DN in the
        candidate's centroid box regardless of whether the candidate was
        accepted (handy for diagnostics), the brightest DN in the window when
        the window holds no candidate, or 0.0 when the window is empty.
    """
    bounds = search_window_bounds(predicted_vu_pos, search_half_vu, image_ext.shape)
    if bounds is None:
        return None, 0.0
    v_lo, v_hi, u_lo, u_hi = bounds
    half = centroid_box_half_px
    kernel = gaussian_psf_kernel(psf_sigma_px, size=2 * half + 1)
    peaks = _window_peaks(image_ext, bounds, kernel=kernel)
    order = np.argsort(-peaks, axis=None, kind='stable')[:_MAX_CANDIDATES]
    candidate: tuple[int, int] | None = None
    for flat_idx in order:
        pv, pu = (int(x) for x in np.unravel_index(int(flat_idx), peaks.shape))
        if not math.isfinite(float(peaks[pv, pu])):
            break
        if not too_sharp(
            image_ext,
            v_lo + pv,
            u_lo + pu,
            half=half,
            max_sharpness=max_peak_sharpness,
            noise_sigma=image_noise_sigma,
            detection_sigma=detection_sigma,
        ):
            candidate = (v_lo + pv, u_lo + pu)
            break
    if candidate is None:
        # No peak in the window, or none that is not a single-pixel spike.
        window = image_ext[v_lo:v_hi, u_lo:u_hi]
        return None, float(np.max(window))
    peak_v, peak_u = candidate
    box = _box(image_ext, peak_v, peak_u, half)
    peak_dn = float(box.max())
    if peak_dn - float(np.median(box)) < detection_sigma * image_noise_sigma:
        return None, peak_dn
    if refuse_outshone:
        window = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
        if _flank_outshines(window, peaks, peak_dn, half=half):
            return None, peak_dn
    centroid = _recentered_centroid(
        image_ext,
        peak_v,
        peak_u,
        half=half,
        bounds=bounds,
        flat_tolerance_dn=detection_sigma * image_noise_sigma,
    )
    return centroid, peak_dn


_FLAT_TOP_WALL = 10.0
"""Least drop, in flat-top tolerances, from a clipped core to the median of its neighbors."""


def _flat_top_region(
    image_ext: NDArrayFloatType,
    peak_v: int,
    peak_u: int,
    *,
    half: int,
    tolerance_dn: float,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None | Literal[False]:
    """Return the region that holds a clipped core as long as the candidate's box.

    The flat top is the 4-connected set of pixels at or above the brightest pixel of the
    candidate's box less ``tolerance_dn``, grown from that pixel over the window padded by
    ``half``: a single pixel for most stars, and the clipped core and its column bleed for
    a saturated one.  It is such a core when it is at least as long as the box
    (``2 * half + 1`` pixels) along one axis and no wider than the box along the other,
    the candidate lies on it or beside it (the response peaks at a core's edge), and it
    stands on a steep wall: the median of the pixels bordering it lies at least
    ``_FLAT_TOP_WALL`` times ``tolerance_dn`` below its brightest pixel.  A smooth bright
    region (a ring, a limb, a saturated body) is flat within the tolerance over larger
    areas, but slopes away gently or is wider than a star in both axes; a core that
    another source's box only catches at its edge is that source's, not the candidate's.
    The brightest pixel of the box must stand that far above the box's median before the
    flat top is grown at all, which skips the search around every faint peak.

    Parameters:
        image_ext: 2-D extfov image array.
        peak_v: Row of the candidate.
        peak_u: Column of the candidate.
        half: Centroid-box half-width in pixels.
        tolerance_dn: Largest step below the brightest pixel that is still the flat top.
        bounds: ``(v_lo, v_hi, u_lo, u_hi)`` of the search window.

    Returns:
        ``None`` when the candidate's box holds no such clipped core;
        ``False`` when it does and the core reaches the edge of the padded window, so it
        is no point source of the window; otherwise ``(v_lo, v_hi, u_lo, u_hi)``, upper
        bounds exclusive, of the core padded by ``half`` on every side and clamped to the
        image.
    """
    h, w = image_ext.shape
    side = 2 * half + 1
    box_v_lo, box_u_lo = max(0, peak_v - half), max(0, peak_u - half)
    box = _box(image_ext, peak_v, peak_u, half)
    top = float(box.max())
    if top - float(np.median(box)) < _FLAT_TOP_WALL * tolerance_dn:
        return None
    bv, bu = (int(x) for x in np.unravel_index(int(np.argmax(box)), box.shape))
    v_lo, v_hi, u_lo, u_hi = bounds
    r_v_lo, r_v_hi = max(0, v_lo - half), min(h, v_hi + half)
    r_u_lo, r_u_hi = max(0, u_lo - half), min(w, u_hi + half)
    region = np.asarray(image_ext[r_v_lo:r_v_hi, r_u_lo:r_u_hi])
    labels, _count = label(region >= top - tolerance_dn)
    flat = labels == labels[box_v_lo + bv - r_v_lo, box_u_lo + bu - r_u_lo]
    rows, cols = np.nonzero(flat)
    f_v_lo, f_v_hi = int(rows.min()), int(rows.max())
    f_u_lo, f_u_hi = int(cols.min()), int(cols.max())
    extents = sorted((f_v_hi - f_v_lo + 1, f_u_hi - f_u_lo + 1))
    cand_v, cand_u = peak_v - r_v_lo, peak_u - r_u_lo
    off_core = max(f_v_lo - cand_v, cand_v - f_v_hi, f_u_lo - cand_u, cand_u - f_u_hi) > 1
    if extents[1] < side or extents[0] > side or off_core:
        return None
    wall = binary_dilation(flat) & ~flat
    if top - float(np.median(region[wall])) < _FLAT_TOP_WALL * tolerance_dn:
        return None
    if f_v_lo == 0 or f_u_lo == 0 or f_v_hi == flat.shape[0] - 1 or f_u_hi == flat.shape[1] - 1:
        return False
    return (
        max(0, f_v_lo + r_v_lo - half),
        min(h, f_v_hi + r_v_lo + half + 1),
        max(0, f_u_lo + r_u_lo - half),
        min(w, f_u_hi + r_u_lo + half + 1),
    )


def _recentered_centroid(
    image_ext: NDArrayFloatType,
    peak_v: int,
    peak_u: int,
    *,
    half: int,
    bounds: tuple[int, int, int, int],
    flat_tolerance_dn: float,
) -> tuple[float, float] | None:
    """Return a candidate's centroid, its box re-centered until it stops moving.

    A box that caught a source by its edge (the response peaks at the edge of a saturated
    core) moves over the source.  A box that walks farther than its own half-width from
    the candidate, or whose center leaves the window, has left the candidate for another
    source, and the candidate has no centroid.

    A saturated star's clipped core, with its column bleed, is a flat top that can be
    longer than the box.  Re-centering has no single resting place on it: a box wholly on
    the flat top is centered on its own centroid wherever it lies, so the box stops a
    half-width in from the end it started at.  When the candidate's box holds such a
    core (:func:`_flat_top_region`), the centroid is instead measured over the core padded
    by the box's half-width on every side; when the core reaches the edge of the window
    padded by that half-width, or that centroid lies outside the window, the candidate is
    no point source of the window and has none.

    Parameters:
        image_ext: 2-D extfov image array.
        peak_v: Row of the candidate's response peak.
        peak_u: Column of the candidate's response peak.
        half: Centroid-box half-width in pixels.
        bounds: ``(v_lo, v_hi, u_lo, u_hi)`` of the search window.
        flat_tolerance_dn: Largest step below the brightest pixel of the candidate's box
            that is still its flat top.

    Returns:
        The ``(v, u)`` centroid, or ``None`` when the box finds nothing above its median,
        walks away, or holds a clipped core that is no point source of the window.
    """
    v_lo, v_hi, u_lo, u_hi = bounds
    flat = _flat_top_region(
        image_ext, peak_v, peak_u, half=half, tolerance_dn=flat_tolerance_dn, bounds=bounds
    )
    if flat is False:
        return None
    if isinstance(flat, tuple):
        centroid = _region_centroid(image_ext, flat)
        if centroid is None:
            return None
        cv, cu = math.floor(centroid[0] + 0.5), math.floor(centroid[1] + 0.5)
        return centroid if v_lo <= cv < v_hi and u_lo <= cu < u_hi else None
    start_v, start_u = peak_v, peak_u
    centroid = _box_centroid(image_ext, peak_v, peak_u, half)
    for _ in range(_MAX_RECENTER):
        if centroid is None:
            return None
        next_v = math.floor(centroid[0] + 0.5)
        next_u = math.floor(centroid[1] + 0.5)
        if (next_v, next_u) == (peak_v, peak_u):
            break
        if (
            abs(next_v - start_v) > half
            or abs(next_u - start_u) > half
            or not (v_lo <= next_v < v_hi and u_lo <= next_u < u_hi)
        ):
            return None
        peak_v, peak_u = next_v, next_u
        centroid = _box_centroid(image_ext, peak_v, peak_u, half)
    return centroid


@dataclass(frozen=True)
class WindowSource:
    """One star-like source found in a star's search window.

    Parameters:
        centroid_vu: ``(v, u)`` sub-pixel centroid in the extended frame.
        response: Matched-filter response at the source's peak; sources are ranked by it.
    """

    centroid_vu: tuple[float, float]
    response: float


def window_sources(
    image_ext: NDArrayFloatType,
    predicted_vu_pos: tuple[float, float],
    *,
    search_half_vu: tuple[float, float],
    centroid_box_half_px: int,
    psf_sigma_px: float,
    max_peak_sharpness: float,
    image_noise_sigma: float,
    detection_sigma: float,
    accept: Callable[[tuple[float, float]], bool],
    max_sources: int,
) -> list[WindowSource]:
    """Return the strongest star-like sources in a search window, strongest first.

    The window and its matched-filter peaks are those of :func:`local_centroid`, and each
    peak, taken in order of falling response, is measured as it measures its one
    candidate: a peak sharper than a star is skipped, so is one whose box's brightest
    pixel stands less than ``detection_sigma * image_noise_sigma`` above the box's median,
    and so is one whose re-centered box walks away.  A peak that passes and that
    ``accept`` takes (the caller's isolation test) is a source; one whose centroid lies
    within a pixel of a source already found is that source again.  Every peak of the
    window may be put to ``accept``, so it should be cheap:
    :func:`~spindoctor.nav_technique._star_isolation.isolation_lookup` builds the
    isolation test once for a whole window.  Where :func:`local_centroid` measures only
    the window's strongest peak and reports no detection when that peak is not a star,
    this keeps looking, so a ring's flank or a brighter source elsewhere in a wide window
    does not hide the star.

    Parameters:
        image_ext: 2-D extfov image array.
        predicted_vu_pos: ``(v, u)`` prediction at the center of the search window.
        search_half_vu: ``(v, u)`` half-widths of the search window in pixels.
        centroid_box_half_px: Half-width of the centroid-fit box (and of the
            matched-filter kernel) in pixels.
        psf_sigma_px: PSF sigma in pixels for the matched-filter kernel.
        max_peak_sharpness: Largest ratio of a star's brightest-pixel excess to the mean
            excess of its four edge neighbors (:func:`too_sharp`).
        image_noise_sigma: Robust per-pixel noise sigma in DN units.
        detection_sigma: Threshold multiplier on ``image_noise_sigma``.
        accept: Returns whether a centroid is a source the caller counts.
        max_sources: Most sources returned.

    Returns:
        Up to ``max_sources`` sources, strongest response first; empty when the window
        lies off the image or holds none.
    """
    bounds = search_window_bounds(predicted_vu_pos, search_half_vu, image_ext.shape)
    if bounds is None or max_sources <= 0:
        return []
    v_lo, _v_hi, u_lo, _u_hi = bounds
    half = centroid_box_half_px
    kernel = gaussian_psf_kernel(psf_sigma_px, size=2 * half + 1)
    peaks = _window_peaks(image_ext, bounds, kernel=kernel)
    flat = peaks.ravel()
    live = np.flatnonzero(np.isfinite(flat))
    order = live[np.argsort(-flat[live], kind='stable')]
    found: list[WindowSource] = []
    for flat_idx in order:
        pv, pu = (int(x) for x in np.unravel_index(int(flat_idx), peaks.shape))
        peak_v, peak_u = v_lo + pv, u_lo + pu
        box = _box(image_ext, peak_v, peak_u, half)
        if float(box.max()) - float(np.median(box)) < detection_sigma * image_noise_sigma:
            continue
        if too_sharp(
            image_ext,
            peak_v,
            peak_u,
            half=half,
            max_sharpness=max_peak_sharpness,
            noise_sigma=image_noise_sigma,
            detection_sigma=detection_sigma,
        ):
            continue
        centroid = _recentered_centroid(
            image_ext,
            peak_v,
            peak_u,
            half=half,
            bounds=bounds,
            flat_tolerance_dn=detection_sigma * image_noise_sigma,
        )
        if centroid is None:
            continue
        if any(
            math.hypot(centroid[0] - s.centroid_vu[0], centroid[1] - s.centroid_vu[1])
            < SAME_SOURCE_PX
            for s in found
        ):
            continue
        if not accept(centroid):
            continue
        found.append(WindowSource(centroid_vu=centroid, response=float(flat[flat_idx])))
        if len(found) >= max_sources:
            break
    return found


def detection_peak_ratio(
    image_ext: NDArrayFloatType,
    predicted_vu_pos: tuple[float, float],
    detection_vu: tuple[float, float],
    *,
    search_half_vu: tuple[float, float],
    exclude_half_px: int,
) -> float:
    """Return the detection's background-subtracted peak-to-runner-up ratio.

    The one-star match rests on the premise that the search window holds
    a *single unambiguous* detection.  When the target star is marginal,
    the brightest pixel in a multi-thousand-pixel window is routinely a
    noise spike whose amplitude barely exceeds the next spike's -- the
    match then carries no information even though the detection cleared
    the absolute ``detection_sigma`` threshold.  This ratio measures the
    ambiguity directly and stays unit-free (no DN-scale or photometric
    calibration enters): a real star detection towers over the noise
    order statistics; a matched noise spike sits within a few percent of
    its runner-up.

    The peak is the detection's own brightest pixel (inside its exclusion
    box), and the runner-up is the brightest pixel of the rest of the
    window.  :func:`local_centroid` chooses its detection by matched-filter
    response, so the window can hold a single pixel brighter than the
    detection's own peak; the ratio then falls below 1 and reports the
    window as ambiguous, which is the safe answer for a match that has no
    second star to check it against.

    Parameters:
        image_ext: 2-D extfov image array.
        predicted_vu_pos: ``(v, u)`` prediction at the center of the
            search window (same value passed to :func:`local_centroid`).
        detection_vu: The accepted detection centroid whose surrounding
            ``exclude_half_px`` box (the source's own pixels) is removed
            before the runner-up is taken.
        search_half_vu: ``(v, u)`` half-widths of the search window in pixels.
        exclude_half_px: Half-width of the exclusion box around the
            detection; use at least the centroid-box half-width so the
            source's own wings do not masquerade as the runner-up.

    Returns:
        ``(peak - background) / (runner_up - background)`` with the
        window median as background, clamped to ``>= 0``; ``inf`` when
        the runner-up does not exceed the background.  0.0 when the
        window is degenerate or the exclusion box misses it.
    """
    bounds = search_window_bounds(predicted_vu_pos, search_half_vu, image_ext.shape)
    if bounds is None:
        return 0.0
    v_lo, v_hi, u_lo, u_hi = bounds
    window = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
    background = float(np.median(window))
    det_v, det_u = detection_vu
    bv_lo = max(0, round(det_v) - exclude_half_px - v_lo)
    bu_lo = max(0, round(det_u) - exclude_half_px - u_lo)
    bv_hi = min(window.shape[0], round(det_v) + exclude_half_px + 1 - v_lo)
    bu_hi = min(window.shape[1], round(det_u) + exclude_half_px + 1 - u_lo)
    if bv_hi <= bv_lo or bu_hi <= bu_lo:
        return 0.0
    peak = float(window[bv_lo:bv_hi, bu_lo:bu_hi].max()) - background
    if peak <= 0.0:
        return 0.0
    masked = window.copy()
    masked[bv_lo:bv_hi, bu_lo:bu_hi] = background
    runner_up = float(masked.max()) - background
    if runner_up <= 0.0:
        return math.inf
    return peak / runner_up


def smear_length_px(feature: NavFeature) -> float:
    """Return a STAR feature's expected smear length in pixels (0.0 without star flags)."""
    flags = feature.flags
    return float(flags.smear_length_px) if isinstance(flags, StarFlags) else 0.0
