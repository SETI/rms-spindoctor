"""The zero-point fit of measured star fields: a reference rate, filter offsets, gain ratios.

A star's zero point is ``2.5 log10(DN / texp) + V``.  A robust least-squares fit (Tukey
biweight) of the zero points on an intercept, one term per filter but the reference filter,
and one term per gain state but the reference state gives the reference rate (DN per second
for a V = 0 star through the reference filter in the reference gain state), the filter
magnitude offsets, and the gain-state DN ratios.  :mod:`util.star_photometry.star_limit_fit`
calls it for each camera.

A filter or gain state with fewer fitting stars than the minimum is not given a term: the
stars cannot pin it down, so the fit reports it as not measured, and it takes the
calibration document's value, fixed, when one is given.  Its stars are left out of the fit.
Had the term been fitted, they would add nothing to the other terms; with the calibration
value held fixed instead, a calibration value that the stars disagree with would move the
reference rate.  Leaving out one state's stars can
take another state below the minimum (a filter whose stars were all taken in unmeasured gain
states), so the selection is repeated until it is stable, and every fitted term rests on at
least the minimum of the stars retained.

:func:`fit_star_rates` fits a camera's absolute rate for the simulator instead: the rate of
a V = 0 star of a reference color from measured rates (electrons per second, say) on an
intercept and a color term, with the residuals tested against each grouping (gain state,
readout mode) and covariate (magnitude, field position, exposure) a caller asks about.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from util.star_photometry.star_limit_fit import FitResult

__all__ = ['RateFit', 'fit_star_rates', 'fit_zero_point']

FloatArray = NDArray[np.float64]

#: Tukey biweight tuning constant (in robust sigmas).
TUKEY_C = 4.685

#: The smallest residual scale (magnitudes) the biweight divides by: well below the
#: precision of any one star's zero point (0.01 mag at the brightest), so it binds only
#: when more than half the zero points agree exactly.
MIN_SCALE_MAG = 0.001


def _tukey_lstsq(
    a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Solve ``a x = b`` by iteratively reweighted least squares with Tukey's biweight.

    The residual scale is the robust scatter, at least ``MIN_SCALE_MAG``.  The first
    weights measure the residuals from their median, so that a least-squares start pulled
    by outliers does not drop the majority with them; the design must hold an intercept.

    Parameters:
        a: Design matrix.
        b: Observations.

    Returns:
        ``(x, sigma_x, residuals, weights)``; ``sigma_x`` from the weighted normal equations
        scaled by the robust residual scatter, ``weights`` the final biweights.
    """
    weights = np.ones(b.size)
    x = np.zeros(a.shape[1])
    for iteration in range(30):
        sw = np.sqrt(weights)
        x, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
        resid = b - a @ x
        median = np.median(resid)
        scale = max(1.4826 * float(np.median(np.abs(resid - median))), MIN_SCALE_MAG)
        u = (resid - (median if iteration == 0 else 0.0)) / (TUKEY_C * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    resid = b - a @ x
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    normal = (a * weights[:, None]).T @ a
    cov = np.linalg.pinv(normal) * scale**2 * (weights.sum() / max(np.sum(weights**2), 1e-9))
    return x, np.sqrt(np.clip(np.diag(cov), 0.0, None)), resid, weights


def _full_rank(a: FloatArray) -> bool:
    """Return whether a design matrix determines every one of its coefficients.

    Parameters:
        a: Design matrix.

    Returns:
        True when its rank equals its number of columns.
    """
    n_rows, n_columns = a.shape
    return bool(n_rows > 0 and int(np.linalg.matrix_rank(a)) == n_columns)


def fit_zero_point(
    result: FitResult,
    rows: list[dict[str, Any]],
    *,
    reference_filter: str,
    reference_gain: str,
    min_term_stars: int,
    calibration_offsets: dict[str, float],
    calibration_gain_ratios: dict[str, float],
) -> None:
    """Fit the reference rate, filter offsets, and gain ratios into ``result``.

    When the stars cannot separate the terms, before or after the robust weighting drops
    outliers, ``result.zero_point_identified`` is set False and no coefficient is stored.

    Parameters:
        result: The camera's result, updated in place.
        rows: The fitting stars.
        reference_filter: The reference filter.
        reference_gain: The reference gain state.
        min_term_stars: The fewest stars a filter or gain term is fitted from.
        calibration_offsets: The calibration document's magnitude offset of each filter
            from the reference filter, used for a filter with too few stars.
        calibration_gain_ratios: The calibration document's DN ratio of each gain state to
            the reference state, used for a gain state with too few stars.
    """
    result.filter_counts = dict(sorted(_count(r['filter'] for r in rows).items()))
    result.gain_counts = dict(sorted(_count(r['gain'] for r in rows).items()))
    observed_filters = set(result.filter_counts) - {reference_filter}
    observed_gains = set(result.gain_counts) - {reference_gain}
    filters, gains = sorted(observed_filters), sorted(observed_gains)
    while True:
        rows = [
            r
            for r in rows
            if (r['filter'] in filters or r['filter'] == reference_filter)
            and (r['gain'] in gains or r['gain'] == reference_gain)
        ]
        filter_counts = _count(r['filter'] for r in rows)
        gain_counts = _count(r['gain'] for r in rows)
        kept_filters = [f for f in filters if filter_counts.get(f, 0) >= min_term_stars]
        kept_gains = [g for g in gains if gain_counts.get(g, 0) >= min_term_stars]
        if kept_filters == filters and kept_gains == gains:
            break
        filters, gains = kept_filters, kept_gains
    result.filters_not_measured = sorted(
        (observed_filters | set(calibration_offsets)) - set(filters) - {reference_filter}
    )
    result.gains_not_measured = sorted(
        (observed_gains | set(calibration_gain_ratios)) - set(gains) - {reference_gain}
    )
    result.n_zp_stars = len(rows)
    result.n_fit_frames = len({r['image_id'] for r in rows})
    n = len(rows)
    a = np.zeros((n, 1 + len(filters) + len(gains)))
    a[:, 0] = 1.0
    b = np.empty(n)
    for i, r in enumerate(rows):
        b[i] = 2.5 * math.log10(r['total_dn'] / r['texp']) + r['vmag']
        if r['filter'] in filters:
            a[i, 1 + filters.index(r['filter'])] = -1.0
        if r['gain'] in gains:
            a[i, 1 + len(filters) + gains.index(r['gain'])] = 1.0
    if not _full_rank(a):
        result.zero_point_identified = False
        return
    x, sx, resid, weights = _tukey_lstsq(a, b)
    if not _full_rank(a[weights > 0.0]):
        result.zero_point_identified = False
        return
    result.rate_vmag0 = 10 ** (0.4 * x[0])
    result.rate_sigma_mag = float(sx[0])
    has_ref_filter = any(r['filter'] == reference_filter for r in rows)
    has_ref_gain = any(r['gain'] == reference_gain for r in rows)
    result.filter_offsets = {reference_filter: 0.0} if has_ref_filter else {}
    result.filter_offset_sigma = {reference_filter: 0.0} if has_ref_filter else {}
    for k, name in enumerate(filters):
        result.filter_offsets[name] = float(x[1 + k])
        result.filter_offset_sigma[name] = float(sx[1 + k])
    result.gain_ratio = {reference_gain: 1.0} if has_ref_gain else {}
    result.gain_ratio_sigma_mag = {reference_gain: 0.0} if has_ref_gain else {}
    for k, name in enumerate(gains):
        result.gain_ratio[name] = float(10 ** (0.4 * x[1 + len(filters) + k]))
        result.gain_ratio_sigma_mag[name] = float(sx[1 + len(filters) + k])
    for name in result.filters_not_measured:
        if name in calibration_offsets:
            result.filter_offsets[name] = calibration_offsets[name]
    for name in result.gains_not_measured:
        if name in calibration_gain_ratios:
            result.gain_ratio[name] = calibration_gain_ratios[name]
    result.zp_scatter_mag = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    bv = np.array([r['bmag'] - r['vmag'] if r['bmag'] is not None else np.nan for r in rows])
    good = np.isfinite(bv) & (np.abs(resid) < 1.0)
    if good.sum() >= 10:
        result.color_slope = float(np.polyfit(bv[good], resid[good], 1)[0])


#: The fewest stars a rate fit, a group's offset, or a covariate's slope rests on.
MIN_RATE_STARS = 5

#: The fewest stars of a frame whose median residual enters the frame-to-frame scatter.
MIN_FRAME_STARS = 3

#: The ratio of the standard error of a median to that of a mean, for normal errors.
_MEDIAN_EFFICIENCY = 1.2533


@dataclass
class RateFit:
    """A camera's absolute star rate and the checks on it.

    A star's zero point here is ``z = 2.5 log10(rate) + V``, so a larger ``z`` is a
    brighter rate for a given magnitude.

    Attributes:
        rate_vmag0: The rate of a V = 0 star of the reference color (per second).
        rate_sigma_mag: The one-sigma formal uncertainty of ``rate_vmag0`` from the
            stars' scatter (magnitudes).  Stars of one frame share its errors, so it
            understates the uncertainty when the frames scatter.
        color_slope: The change of ``z`` per magnitude of B - V; positive when redder
            stars read brighter.
        color_slope_sigma: Its one-sigma uncertainty.
        n_stars: The stars the robust fit kept (nonzero biweight).
        n_frames: The frames those stars come from.
        star_scatter_mag: The robust scatter of the kept stars about the fit.
        frame_scatter_mag: The standard deviation of the frames' median residuals, over
            the frames with at least ``MIN_FRAME_STARS`` kept stars, or None when fewer
            than three frames have that many.
        rate_frame_sigma_mag: The uncertainty of ``rate_vmag0`` from the frames:
            ``frame_scatter_mag`` over the square root of the number of frames it is
            measured on, or None with it.
        groups: For each grouping key and each of its values with at least
            ``MIN_RATE_STARS`` kept stars: ``(median residual, its standard error, star
            count)`` in magnitudes.
        slopes: For each covariate: ``(slope, its one-sigma uncertainty)`` of the kept
            stars' residuals against it, in magnitudes per unit.
    """

    rate_vmag0: float
    rate_sigma_mag: float
    color_slope: float
    color_slope_sigma: float
    n_stars: int
    n_frames: int
    star_scatter_mag: float
    frame_scatter_mag: float | None
    rate_frame_sigma_mag: float | None
    groups: dict[str, dict[str, tuple[float, float, int]]] = field(default_factory=dict)
    slopes: dict[str, tuple[float, float]] = field(default_factory=dict)


def fit_star_rates(
    rows: Sequence[dict[str, Any]],
    *,
    reference_color: float,
    group_keys: Sequence[str] = (),
    covariates: Sequence[str] = (),
) -> RateFit | None:
    """Fit the rate of a V = 0 star of the reference color, and test the residuals.

    A robust least-squares fit (Tukey biweight) of each star's ``z`` on an intercept and
    ``B - V`` less the reference color gives the rate at the reference color.  The
    residuals of the stars the fit keeps are then grouped by each key in ``group_keys``
    (a group's median residual says how far its stars sit from the common rate) and
    regressed on each covariate in ``covariates``.  A row whose covariate is not finite
    is left out of that covariate's slope.

    Parameters:
        rows: One row per star: ``'rate'`` (positive, per second), ``'vmag'`` (Johnson V),
            ``'bv'`` (B - V), ``'image_id'``, and every key named in ``group_keys`` and
            ``covariates``.
        reference_color: The B - V the rate refers to.
        group_keys: Row keys whose values group the residuals.
        covariates: Row keys of numeric values the residuals are regressed on.

    Returns:
        The fit, or None when there are fewer than ``MIN_RATE_STARS`` rows, when the robust
        weighting leaves fewer than that many stars, or when the stars, before or after
        the weighting drops outliers, all share one color so that the rate and the color
        term cannot be told apart.
    """
    if len(rows) < MIN_RATE_STARS:
        return None
    z = np.array([2.5 * math.log10(r['rate']) + r['vmag'] for r in rows])
    color = np.array([r['bv'] - reference_color for r in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(rows)), color])
    if not _full_rank(design):
        return None
    x, sx, resid, weights = _tukey_lstsq(design, z)
    used = weights > 0.0
    if used.sum() < MIN_RATE_STARS or not _full_rank(design[used]):
        return None
    kept = resid[used]
    scatter = float(1.4826 * np.median(np.abs(kept - np.median(kept))))
    by_frame: dict[str, list[float]] = defaultdict(list)
    for r, res, keep in zip(rows, resid, used, strict=True):
        if keep:
            by_frame[str(r['image_id'])].append(float(res))
    medians = [float(np.median(v)) for v in by_frame.values() if len(v) >= MIN_FRAME_STARS]
    frame_scatter = float(np.std(medians, ddof=1)) if len(medians) >= 3 else None
    fit = RateFit(
        rate_vmag0=float(10 ** (0.4 * x[0])),
        rate_sigma_mag=float(sx[0]),
        color_slope=float(x[1]),
        color_slope_sigma=float(sx[1]),
        n_stars=int(used.sum()),
        n_frames=len(by_frame),
        star_scatter_mag=scatter,
        frame_scatter_mag=frame_scatter,
        rate_frame_sigma_mag=(
            None if frame_scatter is None else frame_scatter / math.sqrt(len(medians))
        ),
    )
    for key in group_keys:
        values = np.array([str(r[key]) for r in rows])
        fit.groups[key] = {}
        for value in sorted(set(values[used])):
            group = resid[used & (values == value)]
            if group.size < MIN_RATE_STARS:
                continue
            spread = 1.4826 * float(np.median(np.abs(group - np.median(group))))
            error = _MEDIAN_EFFICIENCY * spread / math.sqrt(group.size)
            fit.groups[key][value] = (float(np.median(group)), error, int(group.size))
    for name in covariates:
        values_f = np.array([float(r[name]) for r in rows])
        sel = used & np.isfinite(values_f)
        if sel.sum() < MIN_RATE_STARS or np.ptp(values_f[sel]) == 0.0:
            continue
        coef, cov = np.polyfit(values_f[sel], resid[sel], 1, cov=True)
        fit.slopes[name] = (float(coef[0]), float(math.sqrt(cov[0, 0])))
    return fit


def _count(values: Iterable[str]) -> dict[str, int]:
    """Return how many times each value occurs.

    Parameters:
        values: The values.

    Returns:
        Value to count.
    """
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return dict(counts)
