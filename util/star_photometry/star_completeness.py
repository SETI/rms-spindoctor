"""The detection-probability model of the star-limit fit, per camera, mode, and gain state.

A star of predicted DN ``DN`` is detected with probability
``floor + (plateau - floor) * Phi((log10 DN - mu) / width)`` (see
:func:`fit_completeness_model`).  The detection limit is the DN at which that probability
covers ``COMPLETENESS`` of its span from floor to plateau.  :mod:`util.star_photometry.
star_limit_fit` fits it per camera and readout mode, and, with its ``--limit-by-gain``
option, per readout mode and gain state (:func:`fit_gain_limits`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import scipy.optimize
import scipy.special
from numpy.typing import NDArray

__all__ = [
    'COMPLETENESS',
    'DN_BIN_DECADES',
    'MIN_LOG_DN_SPAN',
    'MIN_STARS_PER_BIN',
    'N_BOOTSTRAP',
    'WIDTH_BOUNDS',
    'bootstrap_sigma',
    'completeness_curve',
    'crossing',
    'fit_completeness_model',
    'fit_gain_limits',
    'identified_model',
    'insufficient',
    'model_dn',
]

FloatArray = NDArray[np.float64]

#: Detection fraction defining the faintest reliably detected signal.
COMPLETENESS = 0.8

#: Width of the predicted-DN bins of the completeness curve (decades).
DN_BIN_DECADES = 0.1

#: Smallest number of stars in a completeness bin for it to count, the fewest detected and
#: the fewest undetected stars a detection-limit fit needs, and the fewest stars a filter or
#: gain term of the zero point is fitted from.
MIN_STARS_PER_BIN = 15

#: Smallest spread (decades) of the stars' predicted DN, from the 10th to the 90th
#: percentile, that can show the rise from non-detection to detection.
MIN_LOG_DN_SPAN = 0.5

#: Bounds of the detection-probability model's width (decades of DN).
WIDTH_BOUNDS = (0.02, 2.0)

#: Bootstrap resamples for the uncertainty of the detection limit.
N_BOOTSTRAP = 200


def completeness_curve(rows: list[dict[str, Any]]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the detection fraction in bins of predicted DN.

    Parameters:
        rows: Star rows with ``pred_dn``.

    Returns:
        ``(log10 bin center, count, detected fraction)``.
    """
    log_dn = np.array([math.log10(r['pred_dn']) for r in rows])
    detected = np.array([bool(r['detected']) for r in rows])
    edges = np.arange(
        math.floor(log_dn.min() / DN_BIN_DECADES) * DN_BIN_DECADES,
        log_dn.max() + DN_BIN_DECADES,
        DN_BIN_DECADES,
    )
    index = np.digitize(log_dn, edges) - 1
    counts = np.bincount(index, minlength=edges.size)[: edges.size - 1]
    hits = np.bincount(index, weights=detected, minlength=edges.size)[: edges.size - 1]
    centers = edges[:-1] + DN_BIN_DECADES / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = hits / counts
    return centers, counts.astype(float), fraction


def crossing(centers: FloatArray, counts: FloatArray, fraction: FloatArray, level: float) -> float:
    """Return the smallest DN above which the detection fraction stays at or above ``level``.

    Bins with fewer than ``MIN_STARS_PER_BIN`` stars are ignored.  The crossing is
    interpolated linearly in log DN between the last bin below ``level`` and the bin after it.

    Parameters:
        centers: Log10 bin centers.
        counts: Stars per bin.
        fraction: Detection fraction per bin.
        level: The detection fraction.

    Returns:
        The DN, or NaN when the curve never reaches ``level``.
    """
    ok = counts >= MIN_STARS_PER_BIN
    c, f = centers[ok], fraction[ok]
    if c.size == 0 or f[-1] < level:
        return float('nan')
    below = np.nonzero(f < level)[0]
    if below.size == 0:
        return float(10 ** c[0])
    i = int(below[-1])
    if i + 1 >= c.size:
        return float('nan')
    t = (level - f[i]) / (f[i + 1] - f[i])
    return float(10 ** (c[i] + t * (c[i + 1] - c[i])))


def fit_completeness_model(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Fit the detection probability of stars against their predicted DN.

    The model is ``p = floor + (plateau - floor) * Phi((log10 DN - mu) / width)``, with
    ``Phi`` the standard normal integral: ``floor`` is the chance that noise or an unrelated
    source passes the detection test at a faint star's position, and ``plateau`` the detected
    fraction of bright stars, below 1 when some bright stars are missed for reasons unrelated
    to their signal (a catalog magnitude in error, a cosmetic defect).  The fit maximizes the
    likelihood of the individual detections.

    Parameters:
        rows: Star rows with ``pred_dn`` and ``detected``.

    Returns:
        ``(floor, plateau, mu, width)``.
    """
    x = np.array([math.log10(r['pred_dn']) for r in rows])
    y = np.array([bool(r['detected']) for r in rows], dtype=float)

    def neg_log_likelihood(q: FloatArray) -> float:
        floor, plateau, mu, width = q
        p = floor + (plateau - floor) * 0.5 * (
            1.0 + scipy.special.erf((x - mu) / (math.sqrt(2.0) * width))
        )
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    start = np.array([0.01, 0.95, float(np.median(x)), 0.2])
    bounds = [(0.0, 0.3), (0.5, 1.0), (float(x.min()), float(x.max())), WIDTH_BOUNDS]
    best = scipy.optimize.minimize(neg_log_likelihood, start, bounds=bounds, method='L-BFGS-B')
    floor, plateau, mu, width = (float(v) for v in best.x)
    return floor, plateau, mu, width


def model_dn(model: tuple[float, float, float, float], level: float) -> float:
    """Return the DN at which the fitted detection probability covers ``level`` of its span.

    Parameters:
        model: ``(floor, plateau, mu, width)`` from :func:`fit_completeness_model`.
        level: The fraction of the span from the floor to the plateau, in (0, 1).

    Returns:
        The DN.
    """
    _floor, _plateau, mu, width = model
    return float(10 ** (mu + width * math.sqrt(2.0) * scipy.special.erfinv(2.0 * level - 1.0)))


def bootstrap_sigma(rows: list[dict[str, Any]], value: float) -> float:
    """Return the bootstrap (over frames) standard deviation of the detection limit.

    The bootstrap samples the limit's base-10 logarithm, and the spread is carried to DN
    to first order (``value * ln 10`` times the spread in the logarithm), which suits the
    narrow distributions the measured frames give.

    Parameters:
        rows: Star rows with ``pred_dn``.
        value: The limit fitted on all of them.

    Returns:
        The first-order standard deviation in DN.
    """
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[r['image_id']].append(r)
    frames = sorted(by_frame)
    rng = np.random.default_rng(130)
    samples = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(frames), len(frames))
        sample = [r for k in pick for r in by_frame[frames[k]]]
        samples.append(math.log10(model_dn(fit_completeness_model(sample), COMPLETENESS)))
    return float(value * math.log(10) * np.std(samples))


def insufficient(rows: list[dict[str, Any]]) -> str:
    """Return why stars cannot identify the detection limit, or ``''`` when they can.

    Parameters:
        rows: Star rows with a finite, positive ``pred_dn``.

    Returns:
        The reason, or ``''``.
    """
    if len(rows) < MIN_STARS_PER_BIN * 3:
        return f'{len(rows)} stars'
    n_detected = sum(1 for r in rows if r['detected'])
    if min(n_detected, len(rows) - n_detected) < MIN_STARS_PER_BIN:
        return f'{n_detected} detected and {len(rows) - n_detected} undetected stars'
    log_dn = np.array([math.log10(r['pred_dn']) for r in rows])
    p10, p90 = np.percentile(log_dn, [10.0, 90.0])
    if p90 - p10 < MIN_LOG_DN_SPAN:
        return f'predicted DN spread over {p90 - p10:.2f} decades'
    return ''


def identified_model(
    rows: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Fit the detection-probability model when the stars can identify its rise.

    Parameters:
        rows: Star rows with a finite, positive ``pred_dn``.

    Returns:
        ``(model, '')``, or ``(None, reason)`` when the data are insufficient or the fitted
        rise sits at the edge of the stars' DN range or at a width bound.
    """
    reason = insufficient(rows)
    if reason:
        return None, reason
    model = fit_completeness_model(rows)
    log_dn = [math.log10(r['pred_dn']) for r in rows]
    inside = min(log_dn) + DN_BIN_DECADES <= model[2] <= max(log_dn) - DN_BIN_DECADES
    if not inside or not WIDTH_BOUNDS[0] < model[3] < WIDTH_BOUNDS[1]:
        return None, 'the fitted rise lies at the edge of the data'
    return model, ''


def fit_gain_limits(
    rows: list[dict[str, Any]], gain_ratio: dict[str, float]
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, str]], dict[str, str]]:
    """Fit the detection limit per readout mode and gain state, scaling the pairs not fitted.

    In each readout mode, the limit of each gain state whose stars identify the rise (see
    :func:`identified_model`) is fitted from them alone.  Every other gain state of
    ``gain_ratio`` takes the limit of the fitted gain state of the same mode nearest to it
    in DN per electron, scaled by the ratio of the two states' DN per electron: the dimmest
    visible star is taken to hold the same number of electrons at both.  A mode with no
    fitted gain state is left out.

    Parameters:
        rows: Star rows with a finite, positive ``pred_dn``, a ``mode``, and a ``gain``.
        gain_ratio: The DN of each gain state per DN of the reference state.

    Returns:
        ``(limits, scaled_from, insufficient)``: ``limits[mode][gain]`` is ``[DN at
        COMPLETENESS, bootstrap sigma, DN at one half, stars, floor, plateau, mu, width]``
        for a fitted pair and ``[DN, NaN, NaN, 0, NaN, NaN, NaN, NaN]`` for a scaled one;
        ``scaled_from[mode][gain]`` is the gain state a scaled pair was scaled from; and
        ``insufficient['MODE/GAIN']`` is why a pair with stars was not fitted.
    """
    nan = float('nan')
    limits: dict[str, dict[str, list[float]]] = {}
    scaled_from: dict[str, dict[str, str]] = {}
    reasons: dict[str, str] = {}
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pair[(r['mode'], r['gain'])].append(r)
    for mode in sorted({mode for mode, _ in by_pair}):
        fitted: dict[str, list[float]] = {}
        for gain in sorted(g for m, g in by_pair if m == mode):
            members = by_pair[(mode, gain)]
            model, reason = identified_model(members)
            if model is None:
                reasons[f'{mode}/{gain}'] = reason
                continue
            value = model_dn(model, COMPLETENESS)
            fitted[gain] = [
                value,
                bootstrap_sigma(members, value),
                model_dn(model, 0.5),
                float(len(members)),
                *model,
            ]
        if not fitted:
            continue
        limits[mode] = dict(fitted)
        scaled_from[mode] = {}
        for gain, ratio in sorted(gain_ratio.items()):
            if gain in fitted:
                continue
            source = min(fitted, key=lambda g: abs(math.log(gain_ratio[g] / ratio)))
            value = fitted[source][0] * ratio / gain_ratio[source]
            limits[mode][gain] = [value, nan, nan, 0.0, nan, nan, nan, nan]
            scaled_from[mode][gain] = source
    return limits, scaled_from, reasons
