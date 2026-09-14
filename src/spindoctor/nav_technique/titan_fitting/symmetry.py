"""Cross-track offset by mirror correlation about the symmetry axis.

A hazy disc with no surface detail is mirror-symmetric about the image-plane
line through its center and the sub-solar point.  Scoring the mirror
symmetry of an annulus around the limb for every candidate shift across that
line, and refining the winning shift to sub-pixel precision, measures the
cross-track component of the pointing offset.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.nav_technique.titan_fitting.grid import (
    dilate_along_t,
    grid_axis,
    resample_rotated_grid,
    rotated_sample_coords,
    sample_bool_nearest,
    validate_image,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'SymmetryFitParams',
    'SymmetryFitResult',
    'symmetry_scan',
]

# A competing correlation peak must be at least this far from the winning
# peak to count as a rival; closer maxima are part of the same lobe.
_SECOND_PEAK_MIN_SEPARATION_PX = 3


@dataclass(frozen=True)
class SymmetryFitParams:
    """Tuning constants for the mirror-correlation cross-track fit.

    Parameters:
        annulus_inner_fraction: Inner edge of the scoring annulus as a
            fraction of the envelope radius.  Restricting the score to a
            band around the limb keeps structure in the disc interior from
            biasing the symmetry estimate.
        annulus_outer_pad_px: Distance in pixels the annulus extends beyond
            the envelope radius.
        angle_refine_deg: Half-range of the optional symmetry-angle search
            in degrees.  Zero disables angle refinement.
        angle_refine_step_deg: Spacing of the angle search in degrees.
        angle_refine_min_gain: Minimum peak-score improvement a refined
            angle must show before it is adopted over the supplied angle.
        min_peak_score: Smallest acceptable Pearson correlation at the
            winning shift.
        min_valid_fraction: Smallest acceptable fraction of annulus mirror
            pairs that were usable at the winning shift.
        max_second_peak_ratio: Largest acceptable normalized height of a
            competing correlation peak.
        cross_sigma_scale: Multiplier applied to the raw cross-track sigma
            estimate.
        sigma_floor_cross_px: Lower clamp on the reported cross-track sigma.
    """

    annulus_inner_fraction: float
    annulus_outer_pad_px: float
    angle_refine_deg: float
    angle_refine_step_deg: float
    angle_refine_min_gain: float
    min_peak_score: float
    min_valid_fraction: float
    max_second_peak_ratio: float
    cross_sigma_scale: float
    sigma_floor_cross_px: float


@dataclass(frozen=True)
class SymmetryFitResult:
    """Outcome of a mirror-correlation scan over candidate cross-track shifts.

    Parameters:
        cross_track_px: Sub-pixel refined cross-track shift, positive along
            ``c_hat``.
        sigma_cross_px: Reported one-sigma cross-track uncertainty, clamped
            to ``[sigma_floor_cross_px, window_px]``.
        theta_rad: Symmetry-axis angle the reported shift belongs to.  It
            differs from the supplied angle only when angle refinement won.
        peak_score: Pearson correlation at the winning integer shift; NaN
            when no candidate shift had enough signal to correlate at all,
            in which case the ``peak_score`` gate fails.
        valid_fraction: Fraction of annulus mirror pairs that were usable at
            the winning integer shift.
        second_peak_ratio: Normalized height of the strongest competing
            peak; ``0.0`` when the scan has no competing local maximum.
        at_edge: True when the winning integer shift sits on the boundary of
            the search window, so the true shift may lie outside it.
        gate_failed: Name of the first failed gate (``'valid_fraction'``,
            ``'peak_score'`` or ``'second_peak'``), or None when the scan
            passed every gate.
    """

    cross_track_px: float
    sigma_cross_px: float
    theta_rad: float
    peak_score: float
    valid_fraction: float
    second_peak_ratio: float
    at_edge: bool
    gate_failed: str | None


def _pearson(x: NDArrayFloatType, y: NDArrayFloatType) -> float:
    """Return the Pearson correlation of two paired samples, or NaN if undefined."""
    if x.size < 2:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom <= 0.0:
        return float('nan')
    return float((xm * ym).sum()) / denom


def _pair_domain(
    t_vals: NDArrayFloatType,
    n_q: int,
    *,
    inner_px: float,
    outer_px: float,
    capsule_half_extent_px: float,
) -> NDArrayBoolType:
    """Return which ``(q, t)`` mirror pairs fall in the scoring annulus.

    The annulus is the set of points whose distance from the candidate axis
    segment ``{(c, t0) : |t0| <= capsule_half_extent_px}`` lies in
    ``[inner_px, outer_px]``.  Both members of a mirror pair sit at the same
    distance from that segment, and the distance does not depend on the
    candidate shift, so one ``(q, t)`` table serves every candidate.
    """
    q_vals = np.arange(1, n_q + 1, dtype=np.float64)[:, np.newaxis]
    t_eff = np.maximum(np.abs(t_vals)[np.newaxis, :] - capsule_half_extent_px, 0.0)
    dist = np.hypot(q_vals, t_eff)
    in_band: NDArrayBoolType = (dist >= inner_px) & (dist <= outer_px)
    return in_band


def _score_candidates(
    grid: NDArrayFloatType,
    grid_valid: NDArrayBoolType,
    pair_domain: NDArrayBoolType,
    pair_unmasked: NDArrayBoolType,
    window_int: int,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Score every integer candidate shift by mirror correlation.

    Parameters:
        grid: Rotated-grid samples with axes ``(s, t)``.
        grid_valid: Per-sample validity of ``grid``.
        pair_domain: ``(n_q, n_t)`` annulus membership per mirror pair.
        pair_unmasked: ``(n_q, n_t)`` contaminant acceptance per mirror
            pair, already shifted with the candidate hypothesis.
        window_int: Largest integer candidate shift to evaluate.

    Returns:
        ``(scores, valid_fractions)``, each of length ``2 * window_int + 1``
        and indexed by candidate shift from ``-window_int`` upward.  A score
        is NaN where the candidate had too few usable pairs, or too little
        brightness variation, to define a correlation at all; NaN is the
        sentinel precisely because every value in ``[-1, 1]`` is a legitimate
        correlation.
    """
    n_s = grid.shape[0]
    center_index = (n_s - 1) // 2
    n_c = 2 * window_int + 1
    scores = np.full(n_c, np.nan, dtype=np.float64)
    fractions = np.zeros(n_c, dtype=np.float64)
    for idx in range(n_c):
        c = idx - window_int
        i0 = center_index + c
        n_q = min(i0, n_s - 1 - i0)
        if n_q < 1:
            continue
        q = np.arange(1, n_q + 1)
        plus = grid[i0 + q]
        minus = grid[i0 - q]
        domain = pair_domain[:n_q]
        n_domain = int(domain.sum())
        if n_domain == 0:
            continue
        usable = domain & pair_unmasked[:n_q] & grid_valid[i0 + q] & grid_valid[i0 - q]
        fractions[idx] = float(usable.sum()) / n_domain
        score = _pearson(plus[usable], minus[usable])
        if math.isfinite(score):
            scores[idx] = score
    return cast(NDArrayFloatType, scores), cast(NDArrayFloatType, fractions)


def _best_score(scores: NDArrayFloatType) -> float:
    """Return the highest score in a scan, or NaN when none has signal."""
    if not np.isfinite(scores).any():
        return float('nan')
    return float(np.nanmax(scores))


def _peak_index(scores: NDArrayFloatType, window_int: int) -> int:
    """Return the index of the winning candidate shift.

    Falls back to the zero-shift index when no candidate produced a defined
    correlation, so the reported shift is the prediction itself and the
    peak-score gate is what rejects the frame.
    """
    if not np.isfinite(scores).any():
        return window_int
    return int(np.nanargmax(scores))


def _second_peak_ratio(scores: NDArrayFloatType, peak_index: int) -> float:
    """Return the normalized height of the strongest competing correlation peak.

    Competing peaks are local maxima at least
    ``_SECOND_PEAK_MIN_SEPARATION_PX`` away from the winning shift, scored
    as ``(score - min) / (peak - min)``.  The two window-boundary shifts
    count, compared against their single neighbor, because a rival lobe
    that happens to peak against the search bound is exactly the ambiguity
    the gate exists to catch.  No-signal candidates are treated as the
    window minimum.  Zero when there is no competing peak.
    """
    if not math.isfinite(float(scores[peak_index])):
        return 0.0
    lowest = float(np.nanmin(scores))
    filled = np.where(np.isfinite(scores), scores, lowest)
    span = float(filled[peak_index]) - lowest
    if span <= 0.0:
        return 0.0
    ratio = 0.0
    for i in range(filled.size):
        if abs(i - peak_index) < _SECOND_PEAK_MIN_SEPARATION_PX:
            continue
        left = float(filled[i - 1]) if i > 0 else -math.inf
        right = float(filled[i + 1]) if i < filled.size - 1 else -math.inf
        if float(filled[i]) >= left and float(filled[i]) >= right:
            ratio = max(ratio, (float(filled[i]) - lowest) / span)
    return ratio


def _refine_peak(scores: NDArrayFloatType, peak_index: int) -> tuple[float, float, float]:
    """Return ``(delta, s_pk, curvature)`` from a parabola through the peak.

    ``delta`` is the sub-sample offset of the vertex from ``peak_index`` and
    ``s_pk`` the vertex height.  The curvature is NaN, and ``delta`` zero,
    when the peak sits on the window boundary or the three samples are flat,
    so no refinement is possible.
    """
    if peak_index == 0 or peak_index == scores.size - 1:
        return 0.0, float(scores[peak_index]), float('nan')
    y_m = float(scores[peak_index - 1])
    y_0 = float(scores[peak_index])
    y_p = float(scores[peak_index + 1])
    curvature = 0.5 * (y_p + y_m - 2.0 * y_0)
    if not curvature < 0.0:
        return 0.0, y_0, float('nan')
    slope = 0.5 * (y_p - y_m)
    delta = -slope / (2.0 * curvature)
    return delta, y_0 + slope * delta + curvature * delta * delta, curvature


def _cross_sigma(
    s_pk: float, curvature: float, *, window_px: float, params: SymmetryFitParams
) -> float:
    """Return the clamped cross-track sigma from the correlation peak shape.

    The estimate ``scale * sqrt((1 - s_pk) / (2 |a|))`` is a noise-deficit
    heuristic: a peak that falls short of perfect correlation over a weakly
    curved score curve is poorly localized.  When no curvature is available
    at all -- a peak pinned to the window boundary, or a flat score curve --
    the reported sigma is the whole search window, the most uncertainty this
    fit can express, never the floor: an unrefinable peak is the least
    trustworthy outcome, not the tightest.
    """
    deficit = max(1.0 - s_pk, 0.0)
    sigma = float('inf')
    if math.isfinite(curvature) and abs(curvature) > 0.0:
        sigma = params.cross_sigma_scale * math.sqrt(deficit / (2.0 * abs(curvature)))
    if not math.isfinite(sigma):
        sigma = window_px
    return min(max(sigma, params.sigma_floor_cross_px), window_px)


def _scan_one_angle(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    *,
    contaminant_mask: NDArrayBoolType | None,
    mask_shift_vu: tuple[float, float],
    theta_rad: float,
    r_env_px: float,
    window_px: float,
    window_int: int,
    pass_pad_px: float,
    capsule_half_extent_px: float,
    params: SymmetryFitParams,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Resample at one axis angle and score every integer candidate shift."""
    half_extent = r_env_px + params.annulus_outer_pad_px + window_px
    grid, grid_valid = resample_rotated_grid(
        image,
        valid_mask,
        center_vu,
        theta_rad=theta_rad,
        s_half_extent_px=half_extent,
        t_half_extent_px=half_extent,
    )
    n_s, n_t = grid.shape
    center_index = (n_s - 1) // 2
    t_vals = grid_axis(half_extent)
    pair_domain = _pair_domain(
        t_vals,
        center_index,
        inner_px=params.annulus_inner_fraction * r_env_px,
        outer_px=r_env_px + params.annulus_outer_pad_px,
        capsule_half_extent_px=capsule_half_extent_px,
    )
    if contaminant_mask is None:
        pair_unmasked = np.ones((center_index, n_t), dtype=bool)
    else:
        anchor = (center_vu[0] - mask_shift_vu[0], center_vu[1] - mask_shift_vu[1])
        vv, uu = rotated_sample_coords(anchor, theta_rad, grid_axis(half_extent), t_vals)
        mask_grid = dilate_along_t(sample_bool_nearest(contaminant_mask, vv, uu), pass_pad_px)
        # A pair member at grid s = c +- q is predicted to sit q pixels from
        # the predicted center whatever the candidate c is, so the mask read
        # is at the fixed indices center +- q: the hypothesis shift cancels.
        q = np.arange(1, center_index + 1)
        pair_unmasked = ~(mask_grid[center_index + q] | mask_grid[center_index - q])
    return _score_candidates(grid, grid_valid, pair_domain, pair_unmasked, window_int)


def symmetry_scan(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    *,
    contaminant_mask: NDArrayBoolType | None,
    theta0_rad: float,
    r_env_px: float,
    window_px: float,
    pass_pad_px: float,
    capsule_half_extent_px: float = 0.0,
    mask_shift_vu: tuple[float, float] = (0.0, 0.0),
    params: SymmetryFitParams,
) -> SymmetryFitResult:
    """Find the cross-track shift that maximizes mirror symmetry.

    The image is resampled onto the axis-aligned grid and, for every integer
    candidate shift ``c`` in ``[-window_px, window_px]``, the mirror pairs
    ``(G(c + q, t), G(c - q, t))`` lying in the scoring annulus are
    correlated.  Pearson correlation is used deliberately: it is invariant
    to an affine brightness relation between the two halves, so a
    hemispheric brightness difference across the axis cannot move the peak,
    while structural asymmetry still costs score.  The winning shift is
    refined by fitting a parabola through its two neighbors.

    When ``angle_refine_deg`` is positive the whole scan is repeated for
    axis angles offset by up to that much, and the best of those angles is
    adopted only if it beats the supplied angle by more than
    ``angle_refine_min_gain``.

    Parameters:
        image: 2-D image to fit.
        valid_mask: Static per-pixel validity of ``image``.
        center_vu: ``(v, u)`` predicted body center in image coordinates,
            used as the grid origin.
        contaminant_mask: Undilated boolean array, of the image shape,
            marking pixels the fit must ignore, or None when nothing is
            masked.  It is read shifted with the candidate hypothesis and
            dilated along the axis by ``pass_pad_px``.
        theta0_rad: Symmetry-axis angle in radians.
        r_env_px: Haze-envelope radius in pixels.
        window_px: Search half-window in pixels.
        pass_pad_px: Along-axis dilation applied to the contaminant mask,
            covering the along-track position error of this pass.
        capsule_half_extent_px: Half length of the axis segment the annulus
            is measured from.  Zero gives a plain annulus about the grid
            origin; a positive value stretches it along the axis so the
            annulus still meets the body when the along-track position is
            not yet known.
        mask_shift_vu: ``(dv, du)`` displacement already applied to
            ``center_vu`` relative to the geometry the contaminant mask was
            built at, so the mask stays anchored to its predicted position.
        params: Tuning constants.

    Returns:
        A :class:`SymmetryFitResult`.  The gates are evaluated in the order
        ``valid_fraction``, ``peak_score``, ``second_peak``, and the first
        failure is named in ``gate_failed``; a winning shift on the window
        boundary sets ``at_edge`` instead of failing a gate.

    Raises:
        ValueError: if the image is not 2-D, ``valid_mask`` has a different
            shape, ``r_env_px`` is not positive, or ``window_px`` is less
            than one pixel.
    """
    validate_image(image, valid_mask)
    if not math.isfinite(r_env_px) or r_env_px <= 0.0:
        raise ValueError(f'r_env_px must be positive and finite; got {r_env_px!r}')
    if not math.isfinite(window_px) or window_px < 1.0:
        raise ValueError(f'window_px must be at least 1 pixel; got {window_px!r}')

    window_int = math.floor(window_px)

    def scan(theta: float) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        return _scan_one_angle(
            image,
            valid_mask,
            center_vu,
            contaminant_mask=contaminant_mask,
            mask_shift_vu=mask_shift_vu,
            theta_rad=theta,
            r_env_px=r_env_px,
            window_px=window_px,
            window_int=window_int,
            pass_pad_px=pass_pad_px,
            capsule_half_extent_px=capsule_half_extent_px,
            params=params,
        )

    theta = theta0_rad
    scores, fractions = scan(theta)
    if params.angle_refine_deg > 0.0 and params.angle_refine_step_deg > 0.0:
        base_peak = _best_score(scores)
        best_gain = params.angle_refine_min_gain
        n_step = math.floor(params.angle_refine_deg / params.angle_refine_step_deg)
        for i in range(-n_step, n_step + 1):
            if i == 0:
                continue
            trial_theta = theta0_rad + math.radians(i * params.angle_refine_step_deg)
            trial_scores, trial_fractions = scan(trial_theta)
            gain = _best_score(trial_scores) - base_peak
            if gain > best_gain:
                best_gain = gain
                theta = trial_theta
                scores, fractions = trial_scores, trial_fractions

    peak_index = _peak_index(scores, window_int)
    peak_c = peak_index - window_int
    peak_score = float(scores[peak_index])
    valid_fraction = float(fractions[peak_index])
    ratio = _second_peak_ratio(scores, peak_index)
    delta, s_pk, curvature = _refine_peak(scores, peak_index)
    at_edge = abs(peak_c) >= window_int
    if at_edge:
        delta, curvature = 0.0, float('nan')
    sigma = _cross_sigma(s_pk, curvature, window_px=window_px, params=params)

    gate: str | None = None
    if valid_fraction < params.min_valid_fraction:
        gate = 'valid_fraction'
    elif not peak_score >= params.min_peak_score:
        # Negated comparison so a NaN peak -- no candidate had enough signal
        # to correlate at all -- fails the gate instead of slipping past it.
        gate = 'peak_score'
    elif ratio > params.max_second_peak_ratio:
        gate = 'second_peak'
    return SymmetryFitResult(
        cross_track_px=peak_c + delta,
        sigma_cross_px=sigma,
        theta_rad=theta,
        peak_score=peak_score,
        valid_fraction=valid_fraction,
        second_peak_ratio=ratio,
        at_edge=at_edge,
        gate_failed=gate,
    )
