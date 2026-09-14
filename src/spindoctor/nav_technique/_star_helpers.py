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
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'SimilarityFit',
    'brightness_margin_mag',
    'local_centroid',
    'predicted_snr',
    'predicted_vu',
    'similarity_transform_fit',
    'star_features',
    'usable_stars',
]


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


def local_centroid(
    image_ext: NDArrayFloatType,
    predicted_vu_pos: tuple[float, float],
    *,
    search_window_px: float,
    centroid_box_half_px: int,
    image_noise_sigma: float,
    detection_sigma: float,
) -> tuple[tuple[float, float] | None, float]:
    """Return the brightest local detection centroid + matched-filter peak.

    Implements a small DAOPHOT-style detection inside a search window
    centered on the prediction:

    1. Slice the image to a ``(2 * search_window_px + 1)`` half-window
       around the prediction (clamped to image bounds).
    2. Take the brightest pixel inside the window — this is the
       candidate peak.
    3. Reject the candidate when its DN is below
       ``detection_sigma * image_noise_sigma``.
    4. Fit a brightness-weighted moment (Gaussian-equivalent for noise-
       free fixtures) over a ``(2 * centroid_box_half_px + 1)`` box
       around the candidate to pull a sub-pixel centroid.

    The star techniques use this purely-local fit rather than a global
    ``detect_sources`` call so they stay feasible on images where the
    global DAOPHOT pipeline would fail (mostly-empty FOV, dim secondary
    stars).

    Parameters:
        image_ext: 2-D extfov image array.
        predicted_vu_pos: ``(v, u)`` prediction at the center of the
            search window.
        search_window_px: Half-width of the search window in pixels.
        centroid_box_half_px: Half-width of the centroid-fit box in
            pixels.
        image_noise_sigma: Robust per-pixel noise sigma in DN units.
        detection_sigma: Threshold multiplier on
            ``image_noise_sigma``; the brightest peak must clear
            ``detection_sigma * image_noise_sigma`` to be accepted.

    Returns:
        ``(centroid, peak_dn)`` where ``centroid`` is the
        ``(v, u)`` sub-pixel center or ``None`` if no peak cleared
        the threshold.  ``peak_dn`` is the brightest DN in the search
        window regardless of whether the peak was accepted (handy for
        diagnostics).
    """
    v0, u0 = predicted_vu_pos
    h, w = image_ext.shape
    v_lo = max(0, math.floor(v0 - search_window_px))
    u_lo = max(0, math.floor(u0 - search_window_px))
    v_hi = min(h, math.ceil(v0 + search_window_px) + 1)
    u_hi = min(w, math.ceil(u0 + search_window_px) + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return None, 0.0
    window = image_ext[v_lo:v_hi, u_lo:u_hi]
    flat_idx = int(np.argmax(window))
    pv, pu = np.unravel_index(flat_idx, window.shape)
    peak_dn = float(window[pv, pu])
    if peak_dn < detection_sigma * image_noise_sigma:
        return None, peak_dn
    peak_v = v_lo + int(pv)
    peak_u = u_lo + int(pu)
    box_v_lo = max(0, peak_v - centroid_box_half_px)
    box_u_lo = max(0, peak_u - centroid_box_half_px)
    box_v_hi = min(h, peak_v + centroid_box_half_px + 1)
    box_u_hi = min(w, peak_u + centroid_box_half_px + 1)
    box = image_ext[box_v_lo:box_v_hi, box_u_lo:box_u_hi]
    bg = float(np.median(box))
    weights = np.clip(box.astype(np.float64) - bg, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None, peak_dn
    vs = np.arange(box_v_lo, box_v_hi, dtype=np.float64)
    us = np.arange(box_u_lo, box_u_hi, dtype=np.float64)
    centroid_v = float(np.sum(vs[:, None] * weights) / total)
    centroid_u = float(np.sum(us[None, :] * weights) / total)
    return (centroid_v, centroid_u), peak_dn


def detection_peak_ratio(
    image_ext: NDArrayFloatType,
    predicted_vu_pos: tuple[float, float],
    detection_vu: tuple[float, float],
    *,
    search_window_px: float,
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

    Parameters:
        image_ext: 2-D extfov image array.
        predicted_vu_pos: ``(v, u)`` prediction at the center of the
            search window (same value passed to :func:`local_centroid`).
        detection_vu: The accepted detection centroid whose surrounding
            ``exclude_half_px`` box (the source's own pixels) is removed
            before the runner-up is taken.
        search_window_px: Half-width of the search window in pixels.
        exclude_half_px: Half-width of the exclusion box around the
            detection; use at least the centroid-box half-width so the
            source's own wings do not masquerade as the runner-up.

    Returns:
        ``(peak - background) / (runner_up - background)`` with the
        window median as background, clamped to ``>= 0``; ``inf`` when
        the runner-up does not exceed the background.  0.0 when the
        window is degenerate.
    """
    v0, u0 = predicted_vu_pos
    h, w = image_ext.shape
    v_lo = max(0, math.floor(v0 - search_window_px))
    u_lo = max(0, math.floor(u0 - search_window_px))
    v_hi = min(h, math.ceil(v0 + search_window_px) + 1)
    u_hi = min(w, math.ceil(u0 + search_window_px) + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return 0.0
    window = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
    background = float(np.median(window))
    peak = float(window.max()) - background
    if peak <= 0.0:
        return 0.0
    masked = window.copy()
    det_v, det_u = detection_vu
    bv_lo = max(0, round(det_v) - exclude_half_px - v_lo)
    bu_lo = max(0, round(det_u) - exclude_half_px - u_lo)
    bv_hi = min(window.shape[0], round(det_v) + exclude_half_px + 1 - v_lo)
    bu_hi = min(window.shape[1], round(det_u) + exclude_half_px + 1 - u_lo)
    if bv_hi > bv_lo and bu_hi > bu_lo:
        masked[bv_lo:bv_hi, bu_lo:bu_hi] = background
    runner_up = float(masked.max()) - background
    if runner_up <= 0.0:
        return math.inf
    return peak / runner_up
