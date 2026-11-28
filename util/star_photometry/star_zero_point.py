"""The zero-point fit of measured star fields: a reference rate, filter offsets, gain ratios.

A star's zero point is ``2.5 log10(DN / texp) + V``.  A robust least-squares fit (Tukey
biweight) of the zero points on an intercept, one term per filter but the reference filter,
and one term per gain state but the reference state gives the reference rate (DN per second
for a V = 0 star through the reference filter in the reference gain state), the filter
magnitude offsets, and the gain-state DN ratios.  :mod:`util.star_photometry.star_limit_fit`
calls it for each camera.

A camera whose band differs from V sees a star of color ``B - V`` at the camera magnitude
``V + c (B - V)``.  When asked, the fit adds a term in ``B - V`` whose coefficient is the
color term ``c``; the rate then refers to a V = 0 star of ``B - V = 0``.  Only stars whose
catalog color was measured (a ``bmag``) enter such a fit, since a color guessed from a
default star class would pull ``c`` toward zero.

A filter or gain state with fewer fitting stars than the minimum is not given a term: the
stars cannot pin it down, so the fit reports it as not measured, and it takes the
calibration document's value, fixed, when one is given.  Its stars are left out of the fit.
Had the term been fitted, they would add nothing to the other terms; with the calibration
value held fixed instead, a calibration value that the stars disagree with would move the
reference rate.  Leaving out one state's stars can
take another state below the minimum (a filter whose stars were all taken in unmeasured gain
states), so the selection is repeated until it is stable, and every fitted term rests on at
least the minimum of the stars retained.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from util.star_photometry.star_limit_fit import FitResult

__all__ = ['fit_zero_point']

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
    color_term: bool = False,
) -> None:
    """Fit the reference rate, filter offsets, gain ratios, and color term into ``result``.

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
        color_term: Whether to fit the color term ``c`` (see the module docstring), from
            the stars with a measured color alone.  Like any other term, it needs at
            least ``min_term_stars`` of them, and with fewer the zero point is reported
            as unidentified.  Otherwise ``result.color_term`` stays zero.
    """
    if color_term:
        rows = [r for r in rows if r['bmag'] is not None]
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
    if color_term and len(rows) < min_term_stars:
        # The color term rests on every star the pruning keeps, like the intercept.
        result.zero_point_identified = False
        return
    n = len(rows)
    color_column = 1 + len(filters) + len(gains)
    a = np.zeros((n, color_column + int(color_term)))
    a[:, 0] = 1.0
    b = np.empty(n)
    for i, r in enumerate(rows):
        b[i] = 2.5 * math.log10(r['total_dn'] / r['texp']) + r['vmag']
        if r['filter'] in filters:
            a[i, 1 + filters.index(r['filter'])] = -1.0
        if r['gain'] in gains:
            a[i, 1 + len(filters) + gains.index(r['gain'])] = 1.0
        if color_term:
            # A star of larger B - V is fainter in the camera by c (B - V).
            a[i, color_column] = -(r['bmag'] - r['vmag'])
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
    if color_term:
        result.color_term = float(x[color_column])
        result.color_term_sigma = float(sx[color_column])
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
