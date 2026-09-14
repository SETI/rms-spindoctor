"""``BodyBlobNav`` — joint-translation fit from body brightness centroids.

Consumes every ``BODY_BLOB`` feature in the input set, computes a
brightness-weighted-moment centroid for each body inside its predicted
bounding box, and recovers a single 2-D translation that maps the
predicted centroids onto the observed centroids in least-squares.  With
``N >= 2`` blobs the fit is over-determined, which makes the technique
robust to centroid errors on any single body.

The technique reports a confidence intrinsically capped at 0.4: a
brightness-weighted centroid is much weaker than a limb fit, so even an
ideal blob match cannot dominate the ensemble.  Per-blob centroid
uncertainty follows the standard CRLB scaling for a uniform-brightness
disc: ``sigma ~ predicted_diameter_px / (2 * sqrt(N_lit) * SNR)``; the
joint fit inherits that scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pdslogger import PdsLogger
from scipy.signal import fftconvolve

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, body_names_from_features
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import BodyBlobGeometry
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import BodyBlobDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    embed_rotation_unobservable,
    load_ncc_covariance_tuning,
    log_confidence_breakdown,
    reported_position_vu,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.noise_estimate import (
    NOISE_CLIP_SIGMA,
    estimate_background_and_sky_sigma,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyBlobNav']


@dataclass(frozen=True)
class _BlobResiduals:
    """Per-blob residual + statistics arrays for the joint fit.

    ``offsets_v`` / ``offsets_u`` are the per-blob ``observed - predicted``
    centroid vectors, ``weights`` are the per-blob inverse-variance
    weights, and the trailing diagnostic lists feed
    :class:`BodyBlobDiagnostics`.
    """

    consumed: list[NavFeature]
    offsets_v: NDArrayFloatType
    offsets_u: NDArrayFloatType
    weights: NDArrayFloatType
    snrs: list[float]
    extents: list[float]
    phase_angles_deg: list[float]
    phase_irregularity_factors: list[float]


@dataclass(frozen=True)
class _JointFit:
    """Joint translation result derived from per-blob residuals."""

    dv: float
    du: float
    covariance: NDArrayFloatType
    residual_rms: float


def _filter_blob_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset that carries a ``BODY_BLOB`` geometry payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.BODY_BLOB and isinstance(f.geometry, BodyBlobGeometry)
    ]


def _clamp_bbox(
    bbox_extfov_vu: tuple[int, int, int, int],
    extfov_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clamp ``(v_min, u_min, v_max, u_max)`` to lie inside ``extfov_shape``."""
    v_min, u_min, v_max, u_max = bbox_extfov_vu
    h, w = extfov_shape
    return (
        max(0, int(v_min)),
        max(0, int(u_min)),
        min(int(h), int(v_max)),
        min(int(w), int(u_max)),
    )


_BLOB_NOISE_THRESHOLD_SIGMA: float = NOISE_CLIP_SIGMA
"""Pixels above ``background + this * noise_sigma`` count as lit signal.

Aliased to :data:`~spindoctor.support.noise_estimate.NOISE_CLIP_SIGMA` so the
threshold and the background estimator's sky clip stay one value.
"""


_COARSE_CORRELATION_MAX_PHASE_DEG: float = 90.0
"""Phase at which the coarse-acquisition template switches disc -> crescent.

The coarse acquisition correlates a matched-filter template of the predicted
lit silhouette against the observed signal to re-centre each blob's bounding
box on the body.  While the body is at least half-lit a filled disc models
that silhouette, so at or below this phase the disc template runs.  Above it
the sunlit region is a thin crescent whose bright pixels sit a fraction of a
radius off the body center: a disc template locks onto the crescent arc
rather than the center, so the technique instead correlates a crescent
template synthesized at the body's phase and sub-solar direction (see
:func:`_crescent_kernel`).  When that direction is unknown -- a body absent
its illumination geometry -- the high-phase blob keeps its predicted bbox
(no relocation) and relies on the brightness-weighted centroid's existing
crescent handling.  Either way an installed prior bypasses the template
match -- it is a measured offset, not a correlation.
"""


def _disc_kernel(radius_px: float) -> NDArrayFloatType:
    """Return a binary filled-disc matched-filter kernel of the given radius.

    The kernel is the blob-shaped template the coarse acquisition correlates
    against the observed signal: a disc whose response, convolved with the
    lit-signal image, peaks where a body-sized bright region is best centered.
    """
    r = max(math.ceil(radius_px), 1)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disc: NDArrayFloatType = (yy * yy + xx * xx <= radius_px * radius_px).astype(np.float64)
    return disc


def _crescent_kernel(
    radius_px: float, phase_rad: float, sub_solar_dir_vu: tuple[float, float]
) -> NDArrayFloatType:
    """Return a Lambertian-crescent matched-filter kernel.

    Synthesizes the expected lit silhouette of a sphere of the predicted
    radius at the given phase, illuminated from the image-plane direction
    ``sub_solar_dir_vu`` -- the template the coarse acquisition correlates
    when a body is past half phase.  Each pixel inside the projected disc
    carries its Lambertian weight ``max(0, cos(incidence))``, mirroring the
    shading the body NavModel renders, so the bright pixels form the same
    crescent the observed body shows.  Correlating this crescent (rather than
    a full disc) puts the template *center* on the body center instead of
    locking onto the off-center crescent arc.

    At phase 0 the in-plane illumination term vanishes and the kernel is a
    limb-darkened full disc regardless of direction; the caller only reaches
    this path above :data:`_COARSE_CORRELATION_MAX_PHASE_DEG`, where the
    crescent is well-defined.

    Parameters:
        radius_px: Predicted body radius (pixels).
        phase_rad: Phase angle (Sun -> body -> observer), radians.
        sub_solar_dir_vu: Unit ``(v, u)`` image-plane direction toward the
            bright limb (the projected body-to-Sun vector).

    Returns:
        ``(2r + 1, 2r + 1)`` float kernel; zero outside the projected disc
        and on the unlit side.
    """
    r = max(math.ceil(radius_px), 1)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    radius_sq = radius_px * radius_px
    inside = yy * yy + xx * xx <= radius_sq
    z = np.sqrt(np.clip(radius_sq - (yy * yy + xx * xx), 0.0, None))
    # Illumination direction: in-plane component sin(phase) along the
    # sub-solar azimuth, out-of-plane component cos(phase) toward the
    # observer.  This vector is already unit-norm.
    sin_p = math.sin(phase_rad)
    cos_p = math.cos(phase_rad)
    illum_v = sin_p * sub_solar_dir_vu[0]
    illum_u = sin_p * sub_solar_dir_vu[1]
    # cos(incidence) = surface_normal . illumination; the normal at (yy, xx)
    # on the visible hemisphere is (yy, xx, z) / radius.
    cos_incidence = (yy * illum_v + xx * illum_u + z * cos_p) / radius_px
    kernel: NDArrayFloatType = np.where(inside, np.clip(cos_incidence, 0.0, None), 0.0)
    return kernel


def _coarse_correlation_offset(
    image_signal: NDArrayFloatType,
    kernel: NDArrayFloatType,
    predicted_center_vu: tuple[float, float],
    margin_vu: tuple[int, int],
) -> tuple[int, int]:
    """Coarse integer offset from a matched-filter correlation.

    Cross-correlates ``kernel`` (a disc or crescent template) against the
    lit-signal image and returns the integer ``(dv, du)`` shift that
    re-centres the blob's predicted bounding box onto the observed body,
    searched over ``predicted_center +/- margin``.  This extends the blob's
    capture range from the predicted bounding box (a few pixels) to the full
    extended-FOV search window: a brightness-weighted centroid only sees the
    body when it already sits in the predicted box, whereas the correlation
    finds the body anywhere in the window even when SPICE mis-predicts it by
    tens of pixels.  The kernel is flipped before the FFT convolution so the
    operation is a correlation -- the peak lands where the template's geometric
    center best overlaps the body.

    ``predicted_center_vu`` is the predicted *brightness* centroid (the lit
    centroid the feature carries), which on a crescent sits off the geometric
    center.  The correlation peak is the body's geometric center, so the
    kernel's own brightness-centroid offset is added back: the returned shift
    maps the predicted lit centroid onto the observed lit centroid, matching
    the residual the caller forms against ``predicted_center_vu``.  For a
    symmetric disc the offset is zero and the shift is just peak-minus-centre.

    Parameters:
        image_signal: ``(H, W)`` lit signal (image minus background, clipped
            at zero, sky-masked), in extfov coordinates.
        kernel: Odd-sized matched-filter template centered on its middle pixel.
        predicted_center_vu: Predicted body lit centroid in extfov coordinates.
        margin_vu: ``(margin_v, margin_u)`` search half-window about the
            predicted center.

    Returns:
        ``(dv, du)`` integer bbox shift (observed minus predicted lit
        centroid); ``(0, 0)`` when the search window is degenerate or carries
        no signal.
    """
    response = fftconvolve(image_signal, kernel[::-1, ::-1], mode='same')
    h, w = image_signal.shape
    pred_v = round(predicted_center_vu[0])
    pred_u = round(predicted_center_vu[1])
    margin_v, margin_u = margin_vu
    v0 = max(0, pred_v - margin_v)
    v1 = min(h, pred_v + margin_v + 1)
    u0 = max(0, pred_u - margin_u)
    u1 = min(w, pred_u + margin_u + 1)
    if v1 <= v0 or u1 <= u0:
        return (0, 0)
    sub = response[v0:v1, u0:u1]
    if float(sub.max()) <= 0.0:
        return (0, 0)
    rv, ru = np.unravel_index(int(np.argmax(sub)), sub.shape)
    centroid_off_v, centroid_off_u = _kernel_centroid_offset(kernel)
    return (
        v0 + int(rv) - pred_v + round(centroid_off_v),
        u0 + int(ru) - pred_u + round(centroid_off_u),
    )


def _kernel_centroid_offset(kernel: NDArrayFloatType) -> tuple[float, float]:
    """Return a kernel's brightness centroid relative to its middle pixel.

    Zero for a symmetric template (a filled disc); for a crescent it is the
    lit-centroid displacement toward the bright limb, which
    :func:`_coarse_correlation_offset` adds back so the recovered shift is in
    terms of the lit centroid rather than the geometric center.
    """
    total = float(kernel.sum())
    if total <= 0.0:
        return (0.0, 0.0)
    n_v, n_u = kernel.shape
    vs = np.arange(n_v, dtype=np.float64)[:, None]
    us = np.arange(n_u, dtype=np.float64)[None, :]
    centroid_v = float((kernel * vs).sum() / total)
    centroid_u = float((kernel * us).sum() / total)
    return (centroid_v - (n_v - 1) / 2.0, centroid_u - (n_u - 1) / 2.0)


def _clamped_kernel_radius(predicted_diameter_px: float, image_shape: tuple[int, ...]) -> float:
    """Matched-filter kernel radius bounded by the image footprint.

    The template scales with the predicted body diameter, but a template
    larger than the frame adds no localization information (every in-window
    shift sees the same uniform coverage) while the kernel array and its FFT
    convolution allocate memory quadratically in the diameter.  A mostly
    off-frame gas giant predicts a diameter of tens of thousands of pixels,
    which exhausted RAM before this clamp (issue #202).  The frame's
    half-diagonal is the largest radius at which the template's shape can
    still influence the correlation inside the frame.

    Parameters:
        predicted_diameter_px: Predicted body diameter in pixels.
        image_shape: ``(H, W)`` shape of the extfov signal image.

    Returns:
        ``max(predicted_diameter_px / 2, 1)`` clamped to the image
        half-diagonal.
    """
    radius = max(predicted_diameter_px / 2.0, 1.0)
    return min(radius, math.hypot(image_shape[0], image_shape[1]) / 2.0)


def _coarse_disc_offset(
    image_signal: NDArrayFloatType,
    predicted_center_vu: tuple[float, float],
    predicted_diameter_px: float,
    margin_vu: tuple[int, int],
) -> tuple[int, int]:
    """Coarse integer offset from a filled-disc matched filter.

    Convenience wrapper around :func:`_coarse_correlation_offset` for the
    at-least-half-lit case: builds a filled disc of the predicted radius
    (clamped to the image footprint; see :func:`_clamped_kernel_radius`) and
    correlates it.  See :func:`_coarse_correlation_offset` for the offset
    convention and degenerate-window handling.
    """
    radius = _clamped_kernel_radius(predicted_diameter_px, image_signal.shape)
    return _coarse_correlation_offset(
        image_signal, _disc_kernel(radius), predicted_center_vu, margin_vu
    )


def _coarse_crescent_offset(
    image_signal: NDArrayFloatType,
    predicted_center_vu: tuple[float, float],
    predicted_diameter_px: float,
    phase_deg: float,
    sub_solar_dir_vu: tuple[float, float],
    margin_vu: tuple[int, int],
) -> tuple[int, int]:
    """Coarse integer offset from a phase-aware crescent matched filter.

    Builds a Lambertian-crescent template at the body's phase and sub-solar
    direction (:func:`_crescent_kernel`) and correlates it, so a high-phase
    body displaced beyond its predicted bounding box is located without a
    full lit disc to correlate.  The radius is clamped to the image footprint
    (:func:`_clamped_kernel_radius`).  See :func:`_coarse_correlation_offset`
    for the offset convention and degenerate-window handling.
    """
    radius = _clamped_kernel_radius(predicted_diameter_px, image_signal.shape)
    kernel = _crescent_kernel(radius, math.radians(phase_deg), sub_solar_dir_vu)
    return _coarse_correlation_offset(image_signal, kernel, predicted_center_vu, margin_vu)


def _shift_bbox(
    bbox_extfov_vu: tuple[int, int, int, int], dv: int, du: int
) -> tuple[int, int, int, int]:
    """Translate a ``(v_min, u_min, v_max, u_max)`` bbox by ``(dv, du)``."""
    v_min, u_min, v_max, u_max = bbox_extfov_vu
    return (v_min + dv, u_min + du, v_max + dv, u_max + du)


def _brightness_weighted_centroid(
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    geometry: BodyBlobGeometry,
    background: float,
    coarse_offset_vu: tuple[int, int] = (0, 0),
) -> tuple[tuple[float, float] | None, float, int, tuple[int, int, int, int]]:
    """Return the brightness-weighted centroid + signal stats for one blob.

    The centroid is computed over every above-background pixel inside the
    feature's predicted bounding box, **shifted by ``coarse_offset_vu``** so
    the box is re-centred on where the coarse acquisition (a blob-disc
    correlation, or an installed prior) located the body.  Without the shift
    the box only captures the body under small SPICE pointing error (its
    per-body slop); with it the box tracks the body across the full search
    window, so the centroid is computed over the body rather than a clipped
    fragment.  The frame ``background`` (bias + dark pedestal, estimated once
    by :func:`~spindoctor.support.noise_estimate.estimate_background_and_sky_sigma`)
    is subtracted first, so
    the moment weights are signal above background, not raw DN; lit pixels are
    those exceeding ``background + 3 * image_noise_sigma``.  Subtracting the
    background is what keeps a uniform pedestal from dragging the centroid
    toward the bbox center (a phase-dependent bias on high-phase crescents).
    A blob whose (shifted) bbox carries no above-background pixels returns
    ``(None, 0.0, 0, clamped_bbox)`` so the caller can drop it from the
    joint fit and still log its bbox in the per-blob rejection line.

    Returns:
        ``(centroid_vu, mean_signal_above_background, n_lit_pixels,
        clamped_bbox)``.  ``centroid_vu`` is ``None`` when the blob has
        no usable signal.  The centroid is in absolute extfov coordinates,
        so the caller's ``observed - predicted`` residual already includes
        the coarse offset.
    """
    extfov_shape = (image_ext.shape[0], image_ext.shape[1])
    shifted_bbox = _shift_bbox(geometry.bbox_extfov_vu, coarse_offset_vu[0], coarse_offset_vu[1])
    clamped_bbox = _clamp_bbox(shifted_bbox, extfov_shape)
    v_min, u_min, v_max, u_max = clamped_bbox
    if v_max <= v_min or u_max <= u_min:
        return None, 0.0, 0, clamped_bbox
    patch = image_ext[v_min:v_max, u_min:u_max]
    noise_threshold = background + _BLOB_NOISE_THRESHOLD_SIGMA * max(image_noise_sigma, 1e-9)
    signal_mask: NDArrayBoolType = patch > noise_threshold
    n_lit = int(signal_mask.sum())
    if n_lit == 0:
        return None, 0.0, 0, clamped_bbox
    signal = np.where(signal_mask, patch - background, 0.0)
    total_weight = float(signal.sum())
    if total_weight <= 0.0:
        return None, 0.0, 0, clamped_bbox
    vs = np.arange(v_min, v_max, dtype=np.float64)
    us = np.arange(u_min, u_max, dtype=np.float64)
    centroid_v = float(np.sum(signal * vs[:, None]) / total_weight)
    centroid_u = float(np.sum(signal * us[None, :]) / total_weight)
    mean_signal = float(signal[signal_mask].mean())
    return (centroid_v, centroid_u), mean_signal, n_lit, clamped_bbox


def _collect_per_blob_residuals(
    features: list[NavFeature],
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    background: float,
    logger: PdsLogger,
    *,
    image_signal: NDArrayFloatType,
    margin_vu: tuple[int, int],
    prior_offset_vu: tuple[float, float] | None,
) -> _BlobResiduals:
    """Extract the per-blob ``observed - predicted`` residuals + weights.

    Iterates the input features in order and, for each blob, first finds a
    coarse integer offset that re-centres the predicted bbox on the body --
    either the installed pass-1 ``prior_offset_vu`` (rounded), when one is
    available from another technique, or a blob-shaped-disc correlation over
    the search window (:func:`_coarse_disc_offset`).  It then computes a
    background-subtracted brightness-weighted-moment centroid inside the
    shifted bbox.  The coarse stage extends the capture range from the
    bounding box (a few pixels) to the full extfov window; without it the
    centroid silently clips and biases once the body leaves the predicted
    box.  Blobs with no above-background signal in their (shifted) bbox are
    dropped (and logged at DEBUG).  The remaining blobs contribute to the
    joint fit with weight ``N_lit * SNR^2 / radius_px^2`` per the BODY_BLOB
    centroid CRLB.
    """
    consumed: list[NavFeature] = []
    offsets_v: list[float] = []
    offsets_u: list[float] = []
    weights: list[float] = []
    snrs: list[float] = []
    extents: list[float] = []
    phase_angles_deg: list[float] = []
    phase_irregularity_factors: list[float] = []
    prior_coarse = (
        (round(prior_offset_vu[0]), round(prior_offset_vu[1]))
        if prior_offset_vu is not None
        else None
    )
    for feature in features:
        assert isinstance(feature.geometry, BodyBlobGeometry)
        # Coarse acquisition: prefer an installed prior; otherwise correlate a
        # matched-filter template of the predicted lit silhouette to find the
        # body across the full window.  A filled disc models that silhouette
        # while the body is at least half-lit; past half phase a crescent
        # template synthesized at the body's sub-solar direction is needed,
        # since a disc locks onto the off-center crescent arc.  When the
        # crescent's direction is unknown the high-phase blob keeps its
        # predicted bbox (see _COARSE_CORRELATION_MAX_PHASE_DEG).
        phase_deg = float(getattr(feature.flags, 'phase_angle_deg', 0.0))
        sub_solar_dir_vu = getattr(feature.flags, 'sub_solar_dir_vu', (0.0, 0.0))
        if prior_coarse is not None:
            coarse_offset = prior_coarse
        elif phase_deg <= _COARSE_CORRELATION_MAX_PHASE_DEG:
            coarse_offset = _coarse_disc_offset(
                image_signal,
                feature.geometry.predicted_center_vu,
                feature.geometry.predicted_diameter_px,
                margin_vu,
            )
        elif sub_solar_dir_vu != (0.0, 0.0):
            coarse_offset = _coarse_crescent_offset(
                image_signal,
                feature.geometry.predicted_center_vu,
                feature.geometry.predicted_diameter_px,
                phase_deg,
                sub_solar_dir_vu,
                margin_vu,
            )
        else:
            coarse_offset = (0, 0)
        centroid, mean_signal, n_lit, clamped_bbox = _brightness_weighted_centroid(
            image_ext, image_noise_sigma, feature.geometry, background, coarse_offset
        )
        if centroid is None:
            logger.debug(
                'Blob %s has no above-background signal in predicted bbox %s '
                '(background = %.4f DN, threshold = %.4f DN); dropping',
                feature.feature_id,
                clamped_bbox,
                background,
                background + _BLOB_NOISE_THRESHOLD_SIGMA * max(image_noise_sigma, 1e-9),
            )
            continue
        pred_v, pred_u = feature.geometry.predicted_center_vu
        obs_v, obs_u = centroid
        dv = obs_v - pred_v
        du = obs_u - pred_u
        snr = mean_signal / max(image_noise_sigma, 1e-9)
        # Per-blob weight is the inverse of the centroid CRLB
        # variance: weight ~ N_lit * SNR^2 / R^2.  The upstream
        # emission gate guarantees ``predicted_diameter_px >= 8``,
        # so the radius_px denominator is bounded away from zero
        # and no floor is needed.
        radius_px = feature.geometry.predicted_diameter_px / 2.0
        radius_sq = radius_px * radius_px
        weight = float(n_lit) * snr * snr / radius_sq
        consumed.append(feature)
        offsets_v.append(dv)
        offsets_u.append(du)
        weights.append(max(weight, 1e-9))
        snrs.append(snr)
        extents.append(float(feature.geometry.predicted_diameter_px))
        # ``phase_angle_deg`` and ``phase_irregularity_factor`` live on
        # BodyBlobFlags; both default to 0.0 when the feature came from
        # an older NavModel revision so the confidence-formula term
        # degrades gracefully (no penalty) rather than raising.
        flags_phase = float(getattr(feature.flags, 'phase_angle_deg', 0.0))
        flags_factor = float(getattr(feature.flags, 'phase_irregularity_factor', 0.0))
        phase_angles_deg.append(flags_phase)
        phase_irregularity_factors.append(max(0.0, flags_factor))
        # The offsets are differences and stay as they are; the two
        # positions beside them are stated the way a person reads one.
        reported_pred = reported_position_vu((pred_v, pred_u), margin_vu)
        reported_obs = reported_position_vu((obs_v, obs_u), margin_vu)
        logger.debug(
            'Blob %s: predicted (%.2f, %.2f), coarse offset (%d, %d), observed (%.2f, %.2f), '
            'background %.2f DN, SNR %.2f, N_lit %d, weight %.3g',
            feature.feature_id,
            reported_pred[0],
            reported_pred[1],
            coarse_offset[0],
            coarse_offset[1],
            reported_obs[0],
            reported_obs[1],
            background,
            snr,
            n_lit,
            weight,
        )
    return _BlobResiduals(
        consumed=consumed,
        offsets_v=np.asarray(offsets_v, np.float64),
        offsets_u=np.asarray(offsets_u, np.float64),
        weights=np.asarray(weights, np.float64),
        snrs=snrs,
        extents=extents,
        phase_angles_deg=phase_angles_deg,
        phase_irregularity_factors=phase_irregularity_factors,
    )


def _joint_offset_from_residuals(
    residuals: _BlobResiduals,
    *,
    model_error_floor_px: float = 0.0,
    centroid_model_error_frac: float = 0.0,
) -> _JointFit:
    """Solve the precision-weighted joint translation across the per-blob residuals."""
    offsets_v = residuals.offsets_v
    offsets_u = residuals.offsets_u
    weights = residuals.weights
    total_weight = float(weights.sum())
    dv = float(np.sum(weights * offsets_v) / total_weight)
    du = float(np.sum(weights * offsets_u) / total_weight)
    res_norms = np.hypot(offsets_v - dv, offsets_u - du)
    rms = float(np.sqrt(float(np.mean(res_norms * res_norms))))
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.asarray(residuals.extents, np.float64),
        model_error_floor_px=model_error_floor_px,
        centroid_model_error_frac=centroid_model_error_frac,
    )
    return _JointFit(dv=dv, du=du, covariance=cov, residual_rms=rms)


def _centroid_model_error_var(extents_px: NDArrayFloatType, frac: float) -> float:
    """Return the size-scaled centroid-model-error variance for the joint fit.

    A brightness-weighted centroid is a biased estimate of the geometric
    center: the lit hemisphere and any shape irregularity shift it by a
    roughly fixed *fraction* of the body radius, an error the photon-only
    per-blob CRLB is blind to.  Each blob therefore carries a floor
    variance ``(frac * R_i)**2`` with ``R_i`` its predicted radius.  These
    combine by inverse-variance across blobs (the joint centroid is a
    precision-weighted mean), so the joint size-error variance is
    ``1 / sum_i 1 / (frac * R_i)**2`` -- for a single blob this is exactly
    ``(frac * R)**2``, and it shrinks as more blobs constrain the fit.

    Parameters:
        extents_px: Per-blob predicted diameters in pixels.
        frac: Fraction of the body radius taken as the centroid bias sigma;
            ``0.0`` disables the term.

    Returns:
        The joint centroid-model-error variance in pixels squared (``0.0``
        when disabled or no blob has a positive extent).
    """
    if frac <= 0.0:
        return 0.0
    radii = 0.5 * np.asarray(extents_px, np.float64)
    per_blob_var = (frac * radii) ** 2
    positive = per_blob_var[per_blob_var > 0.0]
    if positive.size == 0:
        return 0.0
    return float(1.0 / np.sum(1.0 / positive))


def _joint_covariance(
    *,
    offsets_v: NDArrayFloatType,
    offsets_u: NDArrayFloatType,
    weights: NDArrayFloatType,
    dv: float,
    du: float,
    extents_px: NDArrayFloatType,
    model_error_floor_px: float = 0.0,
    centroid_model_error_frac: float = 0.0,
) -> NDArrayFloatType:
    """Return the per-axis reduced-chi-square covariance of the joint fit.

    The covariance is diagonal.  With ``N`` blobs and ``p = 2`` fitted
    translation parameters, the per-axis reduced chi-square is

    ::

        chi2_nu_axis = sum_i w_i * r_axis_i**2 / max(N - p, 1)

    and the weighted-mean variance is ``chi2_nu_axis / sum(w_i)``.

    NAV-005: a single blob (``N = 1``) cannot constrain a 2-D
    translation -- ``N - p = -1`` so ``max(N - p, 1) = 1`` and there is
    no residual scatter to estimate (the lone residual is zero by
    construction).  The result therefore collapses to the
    inverse-precision floor ``1 / sum(w_i)``, which for a single blob is
    the (large) per-blob centroid CRLB variance -- correctly reflecting
    that one point is near-unobservable for two parameters rather than
    over-confident.  The previous ``sum(w r^2) / (sum w)^2`` form was the
    wrong power of ``sum w`` and carried no degrees-of-freedom factor.

    The positive-definite floor is ``1 / sum(w_i)`` (pure
    inverse-precision), NOT ``1 / (sum w_i)^2``.  Two model-error terms
    are then added in quadrature: the size-scaled centroid-bias variance
    (see :func:`_centroid_model_error_var`), which makes the reported
    sigma track body size the way the photon-only weights cannot, and the
    absolute ``model_error_floor_px**2`` floor.

    The cross-term ``cov(v, u)`` is intentionally zero -- per-axis
    residuals are independent under the BODY_BLOB CRLB derivation,
    and the precision-weighted ensemble combine downstream consumes
    diagonals correctly.  Future readers tempted to add
    ``cov_vu = sum(w * res_v * res_u) / sum(w)``: that term has no
    physical interpretation here because the per-axis errors come
    from independent moment integrals along orthogonal axes.
    """
    total_weight = float(weights.sum())
    floor = 1.0 / max(total_weight, 1e-12)
    model_error = model_error_floor_px * model_error_floor_px + _centroid_model_error_var(
        extents_px, centroid_model_error_frac
    )
    n = int(offsets_v.size)
    if n <= 1:
        # Single blob: under-determined for 2 translation params.  Report
        # the inverse-precision floor (the per-blob centroid CRLB variance),
        # not a tiny over-confident value.
        return float(floor) * np.eye(2, dtype=np.float64) + model_error * np.eye(
            2, dtype=np.float64
        )
    residuals_v = offsets_v - dv
    residuals_u = offsets_u - du
    dof = max(n - 2, 1)
    chi2_nu_v = float(np.sum(weights * residuals_v * residuals_v)) / dof
    chi2_nu_u = float(np.sum(weights * residuals_u * residuals_u)) / dof
    var_v = max(chi2_nu_v / total_weight, floor) + model_error
    var_u = max(chi2_nu_u / total_weight, floor) + model_error
    return np.diag([var_v, var_u]).astype(np.float64)


class BodyBlobNav(NavTechnique):
    """Body-blob brightness-weighted centroid translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({BODY_BLOB})``.
        requires_prior: ``False`` (the technique runs in pass 1).
        tier: ``'fallback'``.

    The ``'fallback'`` tier reflects that the brightness-weighted centroid is a
    weaker observation than a limb fit (already reflected in the technique's
    ``hard_cap: 0.4``).  When a non-spurious primary fit (limb or disc) is
    available for the same body the ensemble drops the blob result rather than
    dilute the geometric techniques' answer with the centroid's lit-hemisphere
    bias.
    """

    name = 'BodyBlobNav'
    accepts_feature_types = frozenset({NavFeatureType.BODY_BLOB})
    requires_prior = False
    tier = 'fallback'
    runs_as_witness = True
    confidence_attributes = frozenset(
        {
            'at_edge',
            'body_snr_inside_predicted_bbox',
            'body_extent_px',
            'blob_count',
            'residual_px',
            'max_phase_angle_deg',
            'max_phase_irregularity_factor',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        # Covariance model-error terms: the brightness-weighted centroid is a
        # biased estimator of the geometric center (lit-hemisphere offset,
        # shape irregularity), and that bias scales with the body radius --
        # a term the photon-only per-blob CRLB cannot see.  ``model_error_size_frac``
        # carries it; ``model_error_floor_px`` is the absolute floor.
        self._cov_tuning = load_ncc_covariance_tuning(self.tuning, self.name)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable BODY_BLOB feature.

        Reads only feature metadata; never any pixels.  The technique
        requires at least one ``BODY_BLOB`` with a non-zero predicted
        diameter — otherwise the centroid moment is degenerate.
        """
        eligible = _eligible_blobs(features)
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_body_blob_features_with_predicted_diameter',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint translation that maps predicted to observed centroids.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Blobs that fall outside the extfov or have no
                above-noise signal in their predicted bbox are dropped.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``image_noise_sigma``, ``obs.extfov_margin_vu``, and
                ``fit_camera_rotation``.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence, and a populated
            :class:`BodyBlobDiagnostics`.  The covariance shape and the
            rotation fields depend on ``context.fit_camera_rotation``:

            - ``False`` (the default Cassini / NHLORRI posture):
              ``covariance_px2`` is ``(2, 2)`` and ``rotation_rad`` /
              ``sigma_rotation_rad`` are ``None``.
            - ``True`` (VGISS / GOSSI): ``covariance_px2`` is the
              rank-deficient ``(3, 3)`` form returned by
              :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`
              (a brightness-weighted centroid is rotation-invariant
              about itself, so the technique carries no rotation
              evidence); ``rotation_rad`` is ``0.0`` and
              ``sigma_rotation_rad`` is the unobservable sentinel.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            eligible = _eligible_blobs(features)
            self.logger.info(
                'Consuming %d BODY_BLOB features (out of %d offered)',
                len(eligible),
                len(features),
            )
            margin_v, margin_u = search_window_for_obs(context)
            self.logger.debug('Search window (v, u) = (%d, %d) px', margin_v, margin_u)
            image_ext = np.asarray(context.image_ext, np.float64)
            global_sigma = float(max(context.image_noise_sigma, 1e-9))
            # The background pedestal and the sky-noise floor are both
            # frame-global, so estimate them once over the valid sky region
            # (sensor data, excluding saturation and cosmic-ray spikes) rather
            # than per-bbox, where a body-filled box would bias the background
            # high.  The sky sigma -- not the body-inflated global sigma -- is
            # the right threshold floor (see estimate_background_and_sky_sigma).
            valid_mask = (
                np.asarray(context.sensor_mask_ext, bool)
                & ~np.asarray(context.saturation_mask_ext, bool)
                & ~np.asarray(context.cosmic_ray_mask_ext, bool)
            )
            background, noise_sigma = estimate_background_and_sky_sigma(
                image_ext,
                valid_mask,
                noise_sigma=global_sigma,
                clip_sigma=_BLOB_NOISE_THRESHOLD_SIGMA,
            )
            self.logger.debug(
                'Frame background pedestal = %.4f DN, sky sigma = %.4f DN (global %.4f)',
                background,
                noise_sigma,
                global_sigma,
            )
            # Lit-signal image for the coarse blob-disc correlation: background
            # subtracted, clipped at zero, and zeroed outside the valid sky
            # (saturation / cosmic rays / non-sensor) so those do not create
            # spurious matched-filter peaks.
            image_signal = np.where(
                valid_mask, np.clip(image_ext - background, 0.0, None), 0.0
            ).astype(np.float64)
            # Use an installed pass-1 prior when present (another technique
            # already located the body); otherwise the per-blob coarse
            # correlation acquires it.
            prior_offset_vu = context.prior_offset_px
            if prior_offset_vu is not None:
                self.logger.debug(
                    'Using pass-1 prior offset (%.2f, %.2f) px to seed blob bboxes',
                    prior_offset_vu[0],
                    prior_offset_vu[1],
                )
            residuals = _collect_per_blob_residuals(
                eligible,
                image_ext,
                noise_sigma,
                background,
                self.logger,
                image_signal=image_signal,
                margin_vu=(margin_v, margin_u),
                prior_offset_vu=prior_offset_vu,
            )
            if not residuals.consumed:
                return self._fail_no_signal(
                    features=eligible,
                    noise_sigma=noise_sigma,
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            fit = _joint_offset_from_residuals(
                residuals,
                model_error_floor_px=self._cov_tuning.model_error_floor_px,
                centroid_model_error_frac=self._cov_tuning.model_error_size_frac,
            )
            at_edge = (
                abs(fit.dv) >= margin_v - self._at_edge_tolerance_px
                or abs(fit.du) >= margin_u - self._at_edge_tolerance_px
            )
            fit_rotation = bool(context.fit_camera_rotation)
            covariance = (
                embed_rotation_unobservable(fit.covariance) if fit_rotation else fit.covariance
            )
            mean_snr = float(np.mean(residuals.snrs))
            mean_extent = float(np.mean(residuals.extents))
            max_phase_angle_deg = (
                float(max(residuals.phase_angles_deg)) if residuals.phase_angles_deg else 0.0
            )
            max_phase_irregularity_factor = (
                float(max(residuals.phase_irregularity_factors))
                if residuals.phase_irregularity_factors
                else 0.0
            )
            diagnostics = BodyBlobDiagnostics(
                body_snr_inside_predicted_bbox=mean_snr,
                body_extent_px=mean_extent,
                blob_count=len(residuals.consumed),
                residual_px=fit.residual_rms,
                max_phase_angle_deg=max_phase_angle_deg,
                max_phase_irregularity_factor=max_phase_irregularity_factor,
            )
            assert self.confidence_spec is not None
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _BlobConfidenceContext(at_edge=at_edge, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, residual RMS %.4f px, mean SNR %.2f, '
                'mean extent %.2f px, blobs %d, confidence %.4f',
                fit.dv,
                fit.du,
                fit.residual_rms,
                mean_snr,
                mean_extent,
                len(residuals.consumed),
                float(confidence),
            )
            if at_edge:
                self.logger.info('Diagnostic flags: at_edge=%s', at_edge)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in residuals.consumed),
                offset_px=(fit.dv, fit.du),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=False,
                at_edge=at_edge,
                diagnostics=diagnostics,
                rotation_rad=0.0 if fit_rotation else None,
                sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
                source_bodies=body_names_from_features(residuals.consumed),
            )

    def _fail_no_signal(
        self, *, features: list[NavFeature], noise_sigma: float, fit_rotation: bool
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result when no blob carries signal.

        Parameters:
            features: Candidate BODY_BLOB features that all failed the
                above-noise signal check (kept on the result so the
                inventory can attribute the rejection per-feature).
            noise_sigma: Image noise sigma (DN) used to compute the
                ``3 * sigma`` rejection threshold; logged in the
                spurious-result message.
            fit_rotation: When True the result carries a ``(3, 3)``
                covariance with the rotation diagonal set to
                :data:`~spindoctor.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE`
                and ``rotation_rad`` / ``sigma_rotation_rad`` populated
                with the rotation-unobservable sentinel; when False the
                result reports a ``(2, 2)`` covariance and both rotation
                fields are ``None``.

        Returns:
            A :class:`NavTechniqueResult` with ``spurious=True``,
            zero confidence, and a populated :class:`BodyBlobDiagnostics`.
        """
        diagnostics = BodyBlobDiagnostics(
            body_snr_inside_predicted_bbox=0.0,
            body_extent_px=0.0,
            blob_count=0,
            residual_px=0.0,
        )
        self.logger.info(
            'No BODY_BLOB feature carried above-noise signal in its predicted bbox '
            '(noise threshold = %.4f DN, %d candidate blob(s)); reporting spurious result',
            3.0 * noise_sigma,
            len(features),
        )
        return self._spurious_result(
            feature_ids=tuple(f.feature_id for f in features),
            diagnostics=diagnostics,
            fit_rotation=fit_rotation,
            source_bodies=body_names_from_features(features),
        )


def _eligible_blobs(features: list[NavFeature]) -> list[NavFeature]:
    """Filter the input set to BODY_BLOB features with non-zero diameter."""
    blob_features = _filter_blob_features(features)
    return [
        f
        for f in blob_features
        if isinstance(f.geometry, BodyBlobGeometry) and f.geometry.predicted_diameter_px > 0.0
    ]


class _BlobConfidenceContext:
    """Adapter binding ``BodyBlobDiagnostics`` plus ``at_edge`` for confidence eval."""

    def __init__(self, *, at_edge: bool, diagnostics: BodyBlobDiagnostics) -> None:
        self.at_edge = at_edge
        self.body_snr_inside_predicted_bbox = diagnostics.body_snr_inside_predicted_bbox
        self.body_extent_px = diagnostics.body_extent_px
        self.blob_count = float(diagnostics.blob_count)
        self.residual_px = diagnostics.residual_px
        self.max_phase_angle_deg = diagnostics.max_phase_angle_deg
        self.max_phase_irregularity_factor = diagnostics.max_phase_irregularity_factor
