"""Fit the photometric star limit and summarize the star PSF from measured star fields.

Reads the JSON-lines records the per-instrument measurement programs of this package write
(:mod:`util.star_photometry.frame_measure`), and for each camera (a camera is the record's
``camera`` field, prefixed by its ``spacecraft`` field when the record has one):

1. **Aperture correction.**  The median ratio of the large-aperture to the small-aperture DN
   over bright, clean stars, per camera and readout mode, is reported (a measure of how much
   light a star spreads beyond the small aperture).
2. **Zero point.**  A star's DN is its large-aperture DN, which holds a smeared or elongated
   star's light where the small aperture would lose a varying part of it.  For every clean,
   unsaturated star detected at a large-aperture flux SNR of at least ``FIT_MIN_FLUX_SNR``,
   ``2.5 log10(DN / texp) + V`` is a zero point (UCAC4 stars brighter than
   ``UCAC4_MIN_VMAG_FOR_FIT`` are left out); a robust least-squares fit
   (Tukey biweight) of the zero points on an intercept, one term per filter but the reference
   filter, and one term per gain state but the reference state gives the reference rate (DN
   per second for a V = 0 star through the reference filter in the reference gain state), the
   filter magnitude offsets, and the gain-state DN ratios.  A fit whose stars cannot
   separate the terms (a design matrix short of full rank, before or after the robust
   weighting) is reported as unidentified, with no coefficients.
3. **Detection limit.**  Every clean star of the fitting frames gets a predicted DN from the
   fit; a detection-probability model (a floor, a plateau and a normal-integral rise in log
   DN) fitted to the stars' detections gives the DN at which detection is ``COMPLETENESS`` of
   the way from floor to plateau (``min_detectable_dn``), with a bootstrap over frames for its
   uncertainty.  The limit is left undefined, as insufficient data, unless there are three
   times ``MIN_STARS_PER_BIN`` stars, detections and non-detections each number
   ``MIN_STARS_PER_BIN``, the stars' predicted DN spread over ``MIN_LOG_DN_SPAN`` decades,
   and the fitted rise lies inside that spread.  The same holds for each readout mode.
4. **Held-out check.**  On the frames held out of steps 2 and 3 (one in three, chosen by a hash
   of the image ID), the predicted dimmest visible magnitude is compared with the faintest
   stars detected.
5. **Current versus proposed.**  Per frame: the navigator's current limit, the proposed limit
   (dimmest visible plus ``SEARCH_DEPTH_MAG``), and the faintest star detected.
6. **PSF.**  Per camera and mode (and filter): the circular Gaussian sigma, the elliptical axes,
   the encircled-energy radii, and the narrowest star
   (:mod:`util.star_photometry.star_psf_summary`).

Usage::

    python -m util.star_photometry.star_limit_fit RECORDS.jsonl OUT_DIR --reference-filter CLEAR
        --reference-gain 10K [--ee-geometry MODE:R_MAX:N_BINS ...]

writes ``summary.json`` (every fitted quantity), ``frames.csv`` (the per-frame table) and
``stars.csv`` (every measured star) into ``OUT_DIR``, and prints the summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
import scipy.special
from numpy.typing import NDArray
from util.star_photometry.star_psf_summary import gaussian_ee, psf_summary

__all__ = ['FitResult', 'fit_camera', 'load_records', 'main']

FloatArray = NDArray[np.float64]

#: Smallest flux SNR of a star used in the zero-point fit.
FIT_MIN_FLUX_SNR = 10.0

#: UCAC4 stars brighter than this are left out of the zero-point fit: the catalog's
#: magnitudes there read too faint (by 0.5-0.9 mag at V 8-9 on all three cameras measured),
#: past the reach of the catalog reduction's bright-end correction.
UCAC4_MIN_VMAG_FOR_FIT = 9.5

#: Smallest flux SNR of a star used for the aperture correction.
APCORR_MIN_FLUX_SNR = 30.0

#: Detection fraction defining the faintest reliably detected signal.
COMPLETENESS = 0.8

#: The catalog is searched this far past the dimmest visible star, as the previous
#: navigation software did.
SEARCH_DEPTH_MAG = 1.0

#: Width of the predicted-DN bins of the completeness curve (decades).
DN_BIN_DECADES = 0.1

#: Smallest number of stars in a completeness bin for it to count, and the fewest detected
#: and the fewest undetected stars a detection-limit fit needs.
MIN_STARS_PER_BIN = 15

#: Smallest spread (decades) of the stars' predicted DN, from the 10th to the 90th
#: percentile, that can show the rise from non-detection to detection.
MIN_LOG_DN_SPAN = 0.5

#: Bounds of the detection-probability model's width (decades of DN).
WIDTH_BOUNDS = (0.02, 2.0)

#: One frame in this many is held out of the fit.
HOLDOUT_MODULUS = 3

#: Bootstrap resamples for the uncertainty of the detection limit.
N_BOOTSTRAP = 200

#: Tukey biweight tuning constant (in robust sigmas).
TUKEY_C = 4.685

#: Header of an empty ``frames.csv``: the camera and the fields of :func:`_frame_table`.
FRAME_COLUMNS = (
    'camera',
    'image_id',
    'filter',
    'gain',
    'mode',
    'texp',
    'held_out',
    'n_agree',
    'current_limit',
    'deep_limit',
    'proposed_limit',
    'dimmest_visible',
    'faintest_detected',
    'second_faintest_detected',
    'n_detected',
    'n_clean',
)

#: Header of an empty ``stars.csv``: the camera and the frame fields :func:`_star_rows` adds
#: (a star's own fields vary with what was measured).
STAR_COLUMNS = ('camera', 'image_id', 'filter', 'gain', 'mode', 'texp', 'held_out', 'target')


def load_records(path: Path) -> list[dict[str, Any]]:
    """Return the records of a JSON-lines file, leaving out frames that failed to load.

    Parameters:
        path: The file.

    Returns:
        The records.
    """
    with path.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return [r for r in records if 'error' not in r]


def camera_key(record: dict[str, Any]) -> str:
    """Return the camera a record belongs to: its camera, prefixed by its spacecraft if any.

    Parameters:
        record: A frame record.

    Returns:
        The key, for example ``'VG2 NAC'`` or ``'SSI'``.
    """
    camera = str(record['frame']['camera'])
    if 'spacecraft' in record:
        return f'VG{record["spacecraft"]} {camera}'
    return camera


def is_held_out(image_id: str) -> bool:
    """Return whether a frame is held out of the fit (a stable hash of its ID).

    Parameters:
        image_id: The frame's image ID.

    Returns:
        True for one frame in ``HOLDOUT_MODULUS``.
    """
    digest = hashlib.sha256(image_id.encode()).digest()
    return digest[0] % HOLDOUT_MODULUS == 0


def _clean(star: dict[str, Any]) -> bool:
    """Return whether a star is usable at all: measured, unsaturated, isolated, unoccluded.

    Parameters:
        star: A star record.

    Returns:
        True when the star has detection statistics and photometry and no exclusion flag.
    """
    return (
        'detected' in star
        and 'flux' in star
        and not star['saturated']
        and not star.get('no_data', False)
        and not star['blended']
        and star['conflict'] == ''
        and not star['photometry_saturated']
    )


@dataclass
class FitResult:
    """Everything fitted for one camera.

    Attributes:
        camera: The camera key.
        n_frames: Frames with an accepted pointing fix.
        n_fit_frames: Of those, frames used in the fit.
        apcorr: Aperture correction per mode.
        zero_point_identified: False when the fitting stars cannot separate the intercept,
            the filter terms, and the gain terms (for example, every star through one
            filter taken in one gain state and every other star in the other); the rate,
            offsets, and ratios are then left out.
        rate_vmag0: DN per second of a V = 0 star, reference filter and gain state.
        rate_sigma_mag: Uncertainty of the rate, in magnitudes.
        filter_offsets: Magnitude offset of each filter from the reference filter.
        filter_offset_sigma: Uncertainty of each filter offset.
        gain_ratio: DN of each gain state per DN of the reference state.
        gain_ratio_sigma_mag: Uncertainty of each gain ratio, in magnitudes.
        zp_scatter_mag: Robust scatter of the zero-point residuals.
        n_zp_stars: Stars in the zero-point fit.
        color_slope: Slope of the zero-point residual against B - V (mag per mag).
        min_detectable_dn: Predicted DN at which the fitted detection probability covers
            ``COMPLETENESS`` of its span from floor to plateau.
        min_detectable_dn_sigma: Bootstrap (over frames) standard deviation of it.
        min_detectable_dn_50: The same at half the span.
        min_detectable_dn_by_mode: Per readout mode with enough stars, ``[DN at
            COMPLETENESS, bootstrap sigma, DN at one half, stars, floor, plateau, mu,
            width]``; a frame of such a mode takes its mode's DN.
        min_detectable_dn_empirical: The smallest DN above which the binned detection
            fraction stays at or above ``COMPLETENESS`` (a check on the model).
        min_detectable_dn_insufficient: Why the limit is undefined, for the camera (key
            ``'all'``) and for each readout mode whose stars cannot identify the rise.
        completeness_model: ``[floor, plateau, mu, width]`` of the camera's fit.
        completeness: Rows ``[log10 DN, n, detected fraction]`` of the completeness curve.
        false_detection_rate: The fitted detection probability's floor.
        holdout: The held-out check.
        frames: Per-frame rows.
        psf: PSF summary per mode (and filter).
    """

    camera: str
    n_frames: int = 0
    n_fit_frames: int = 0
    apcorr: dict[str, float] = field(default_factory=dict)
    zero_point_identified: bool = True
    rate_vmag0: float = float('nan')
    rate_sigma_mag: float = float('nan')
    filter_offsets: dict[str, float] = field(default_factory=dict)
    filter_offset_sigma: dict[str, float] = field(default_factory=dict)
    filter_counts: dict[str, int] = field(default_factory=dict)
    gain_ratio: dict[str, float] = field(default_factory=dict)
    gain_ratio_sigma_mag: dict[str, float] = field(default_factory=dict)
    gain_counts: dict[str, int] = field(default_factory=dict)
    zp_scatter_mag: float = float('nan')
    n_zp_stars: int = 0
    color_slope: float = float('nan')
    min_detectable_dn: float = float('nan')
    min_detectable_dn_sigma: float = float('nan')
    min_detectable_dn_50: float = float('nan')
    min_detectable_dn_by_mode: dict[str, list[float]] = field(default_factory=dict)
    min_detectable_dn_empirical: float = float('nan')
    min_detectable_dn_insufficient: dict[str, str] = field(default_factory=dict)
    completeness_model: list[float] = field(default_factory=list)
    completeness: list[list[float]] = field(default_factory=list)
    false_detection_rate: float = float('nan')
    holdout: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    psf: dict[str, Any] = field(default_factory=dict)


def _tukey_lstsq(
    a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Solve ``a x = b`` by iteratively reweighted least squares with Tukey's biweight.

    Parameters:
        a: Design matrix.
        b: Observations.

    Returns:
        ``(x, sigma_x, residuals, weights)``; ``sigma_x`` from the weighted normal equations
        scaled by the robust residual scatter, ``weights`` the final biweights.
    """
    weights = np.ones(b.size)
    x = np.zeros(a.shape[1])
    for _ in range(30):
        sw = np.sqrt(weights)
        x, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
        resid = b - a @ x
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        u = resid / (TUKEY_C * scale)
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


def _star_rows(records: Iterable[dict[str, Any]], *, gain_field: str) -> list[dict[str, Any]]:
    """Flatten accepted frames into one row per star with its frame's state.

    Parameters:
        records: Frame records.
        gain_field: The frame field that holds the gain state (``gain_state``, or ``mode``
            for a camera whose DN per electron changes with the readout mode).

    Returns:
        The rows.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record['fix']['accepted']:
            continue
        frame = record['frame']
        for star in record.get('stars', []):
            rows.append(
                {
                    **star,
                    'image_id': frame['image_id'],
                    'filter': frame['filter_name'],
                    'gain': frame[gain_field],
                    'mode': frame['mode'],
                    'texp': float(frame['texp']),
                    'held_out': is_held_out(frame['image_id']),
                    'target': frame['extra'].get('target', ''),
                }
            )
    return rows


def _apcorr(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return the aperture correction per mode from bright clean stars.

    Parameters:
        rows: Star rows of one camera.

    Returns:
        Mode to median large-to-small-aperture DN ratio.
    """
    ratios: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not _clean(row) or row['flux'] <= 0.0:
            continue
        if row['flux_big'] / max(row['flux_big_sigma'], 1e-9) < APCORR_MIN_FLUX_SNR:
            continue
        ratios[row['mode']].append(row['flux_big'] / row['flux'])
    return {mode: float(np.median(v)) for mode, v in ratios.items() if len(v) >= 5}


def fit_camera(
    camera: str,
    records: list[dict[str, Any]],
    *,
    reference_filter: str,
    reference_gain: str,
    gain_field: str = 'gain_state',
    min_texp: float = 0.0,
    limit_exclude_targets: frozenset[str] = frozenset(),
) -> FitResult:
    """Fit one camera's photometric limit and summarize its PSF.

    Parameters:
        camera: The camera key.
        records: The camera's frame records.
        reference_filter: The filter the rate refers to (offsets are relative to it).
        reference_gain: The gain state the rate refers to (ratios are relative to it).
        gain_field: The frame field holding the gain state (see :func:`_star_rows`).
        min_texp: Frames exposed less than this (seconds) are left out of the zero point and
            the detection limit (they stay in the per-frame table): a camera whose shortest
            exposures do not scale with the commanded time, such as one without a shutter
            whose frame transfer adds a fixed exposure, would bias both.
        limit_exclude_targets: Frames of these targets are left out of the detection limit
            and the held-out check (crowded star-cluster fields, whose unresolved light and
            neighbors lower the completeness at a given DN below that of a navigation field).

    Returns:
        The fit.
    """
    result = FitResult(camera=camera)
    accepted = [r for r in records if r['fix']['accepted']]
    result.n_frames = len(accepted)
    rows = _star_rows(accepted, gain_field=gain_field)
    result.apcorr = _apcorr(rows)
    for row in rows:
        row['total_dn'] = row.get('flux_big', float('nan'))
    fit_rows = [
        r
        for r in rows
        if r['texp'] >= min_texp
        and not r['held_out']
        and _clean(r)
        and r['detected']
        and r['flux_big'] > 0.0
        and r['flux_big_sigma'] > 0.0
        and r['flux_big'] / r['flux_big_sigma'] >= FIT_MIN_FLUX_SNR
        and not (r['catalog'] == 'ucac4' and r['vmag'] < UCAC4_MIN_VMAG_FOR_FIT)
    ]
    result.n_fit_frames = len(
        {r['frame']['image_id'] for r in accepted if not is_held_out(r['frame']['image_id'])}
    )
    result.n_zp_stars = len(fit_rows)
    if len(fit_rows) < 5:
        result.psf = psf_summary(rows)
        return result
    _fit_zero_point(
        result, fit_rows, reference_filter=reference_filter, reference_gain=reference_gain
    )
    for row in rows:
        row['pred_dn'] = _predict_dn(result, row)
    limit_rows = [
        r
        for r in rows
        if r['texp'] >= min_texp and r['target'] not in limit_exclude_targets and _clean(r)
    ]
    _fit_detection_limit(result, [r for r in limit_rows if not r['held_out']])
    result.holdout = _holdout_check(result, [r for r in limit_rows if r['held_out']])
    result.frames = _frame_table(result, accepted, rows, gain_field=gain_field)
    result.psf = psf_summary(rows)
    return result


def _fit_zero_point(
    result: FitResult,
    rows: list[dict[str, Any]],
    *,
    reference_filter: str,
    reference_gain: str,
) -> None:
    """Fit the reference rate, filter offsets and gain ratios into ``result``.

    When the stars cannot separate the terms, before or after the robust weighting drops
    outliers, ``result.zero_point_identified`` is set False and no coefficient is stored.

    Parameters:
        result: The camera's result, updated in place.
        rows: The fitting stars.
        reference_filter: The reference filter.
        reference_gain: The reference gain state.
    """
    filters = sorted({r['filter'] for r in rows} - {reference_filter})
    gains = sorted({r['gain'] for r in rows} - {reference_gain})
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
    result.filter_counts = dict(sorted(_count(r['filter'] for r in rows).items()))
    result.gain_counts = dict(sorted(_count(r['gain'] for r in rows).items()))
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


def _predict_dn(result: FitResult, row: dict[str, Any]) -> float:
    """Return the fitted total DN of a star, or NaN for an unfitted filter or gain state.

    Parameters:
        result: The camera's fit.
        row: The star row.

    Returns:
        The predicted DN.
    """
    if row['filter'] not in result.filter_offsets or row['gain'] not in result.gain_ratio:
        return float('nan')
    magnitude = row['vmag'] + result.filter_offsets[row['filter']]
    rate = result.rate_vmag0 * result.gain_ratio[row['gain']]
    return float(rate * 10 ** (-0.4 * magnitude) * row['texp'])


def _completeness_curve(rows: list[dict[str, Any]]) -> tuple[FloatArray, FloatArray, FloatArray]:
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


def _crossing(centers: FloatArray, counts: FloatArray, fraction: FloatArray, level: float) -> float:
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


def _fit_completeness_model(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
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


def _model_dn(model: tuple[float, float, float, float], level: float) -> float:
    """Return the DN at which the fitted detection probability covers ``level`` of its span.

    Parameters:
        model: ``(floor, plateau, mu, width)`` from :func:`_fit_completeness_model`.
        level: The fraction of the span from the floor to the plateau, in (0, 1).

    Returns:
        The DN.
    """
    _floor, _plateau, mu, width = model
    return float(10 ** (mu + width * math.sqrt(2.0) * scipy.special.erfinv(2.0 * level - 1.0)))


def _bootstrap_sigma(rows: list[dict[str, Any]], value: float) -> float:
    """Return the bootstrap (over frames) standard deviation of the detection limit.

    Parameters:
        rows: Star rows with ``pred_dn``.
        value: The limit fitted on all of them.

    Returns:
        The standard deviation in DN.
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
        samples.append(math.log10(_model_dn(_fit_completeness_model(sample), COMPLETENESS)))
    return float(value * math.log(10) * np.std(samples))


def _insufficient(rows: list[dict[str, Any]]) -> str:
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


def _identified_model(
    rows: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Fit the detection-probability model when the stars can identify its rise.

    Parameters:
        rows: Star rows with a finite, positive ``pred_dn``.

    Returns:
        ``(model, '')``, or ``(None, reason)`` when the data are insufficient or the fitted
        rise sits at the edge of the stars' DN range or at a width bound.
    """
    reason = _insufficient(rows)
    if reason:
        return None, reason
    model = _fit_completeness_model(rows)
    log_dn = [math.log10(r['pred_dn']) for r in rows]
    inside = min(log_dn) + DN_BIN_DECADES <= model[2] <= max(log_dn) - DN_BIN_DECADES
    if not inside or not WIDTH_BOUNDS[0] < model[3] < WIDTH_BOUNDS[1]:
        return None, 'the fitted rise lies at the edge of the data'
    return model, ''


def _fit_detection_limit(result: FitResult, rows: list[dict[str, Any]]) -> None:
    """Fit ``min_detectable_dn`` from the completeness of the fitting frames' stars.

    The detection probability model of :func:`_fit_completeness_model` is fitted, and the
    limit is the DN at which it covers ``COMPLETENESS`` of its span from floor to plateau.
    The limit is fitted on all the camera's stars and, where a readout mode has enough stars,
    on that mode's alone.  The binned curve and its empirical crossings are kept as a check.
    Where the stars cannot identify the rise (see :func:`_identified_model`), the limit is
    left undefined and the reason is stored in ``min_detectable_dn_insufficient``.

    Parameters:
        result: The camera's result, updated in place.
        rows: Clean star rows of the fitting frames with ``pred_dn``.
    """
    rows = [r for r in rows if math.isfinite(r['pred_dn']) and r['pred_dn'] > 0.0]
    for mode in sorted({r['mode'] for r in rows}):
        members = [r for r in rows if r['mode'] == mode]
        mode_model, reason = _identified_model(members)
        if mode_model is None:
            result.min_detectable_dn_insufficient[mode] = reason
            continue
        value = _model_dn(mode_model, COMPLETENESS)
        result.min_detectable_dn_by_mode[mode] = [
            value,
            _bootstrap_sigma(members, value),
            _model_dn(mode_model, 0.5),
            float(len(members)),
            *mode_model,
        ]
    model, reason = _identified_model(rows)
    if model is None:
        result.min_detectable_dn_insufficient['all'] = reason
        return
    centers, counts, fraction = _completeness_curve(rows)
    result.completeness = [
        [float(c), float(n), float(f)]
        for c, n, f in zip(centers, counts, fraction, strict=True)
        if n > 0
    ]
    result.min_detectable_dn_empirical = _crossing(centers, counts, fraction, COMPLETENESS)
    result.completeness_model = list(model)
    result.min_detectable_dn = _model_dn(model, COMPLETENESS)
    result.min_detectable_dn_50 = _model_dn(model, 0.5)
    result.false_detection_rate = model[0]
    result.min_detectable_dn_sigma = _bootstrap_sigma(rows, result.min_detectable_dn)


def _dimmest_visible(
    result: FitResult, filter_name: str, gain: str, texp: float, mode: str
) -> float:
    """Return the V magnitude whose predicted DN is the detection limit.

    The limit is the mode's ``min_detectable_dn`` when the mode has one, else the camera's.

    Parameters:
        result: The camera's fit.
        filter_name: The frame's filter.
        gain: The frame's gain state.
        texp: The frame's exposure.
        mode: The frame's readout mode.

    Returns:
        The magnitude, or NaN for an unfitted filter or gain state.
    """
    if filter_name not in result.filter_offsets or gain not in result.gain_ratio:
        return float('nan')
    limit = result.min_detectable_dn_by_mode.get(mode, [result.min_detectable_dn])[0]
    if not math.isfinite(limit) or texp <= 0.0:
        return float('nan')
    dn_vmag0 = result.rate_vmag0 * result.gain_ratio[gain] * texp
    return 2.5 * math.log10(dn_vmag0 / limit) - result.filter_offsets[filter_name]


def _faintest_detected(stars: list[dict[str, Any]], rank: int = 2) -> float:
    """Return the ``rank``-th faintest detected clean star's magnitude (robust to one fluke).

    Parameters:
        stars: Star rows of one frame.
        rank: Which faintest (1 = the faintest).

    Returns:
        The magnitude, or NaN when fewer stars were detected.
    """
    mags = sorted((s['vmag'] for s in stars if s['detected']), reverse=True)
    return float(mags[rank - 1]) if len(mags) >= rank else float('nan')


def _holdout_check(result: FitResult, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the predicted dimmest visible magnitude with the held-out frames' detections.

    For each held-out frame with at least two detected clean stars, the difference is the
    second-faintest detected magnitude less the predicted dimmest visible magnitude.  Over all
    held-out stars together, the detection fraction of stars predicted brighter than the
    dimmest visible magnitude, and of stars predicted between it and one magnitude fainter.

    Parameters:
        result: The camera's fit.
        rows: Clean star rows of the held-out frames.

    Returns:
        The check's numbers.
    """
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[r['image_id']].append(r)
    diffs = []
    brighter, between = [], []
    for stars in by_frame.values():
        first = stars[0]
        limit = _dimmest_visible(
            result, first['filter'], first['gain'], first['texp'], first['mode']
        )
        if not math.isfinite(limit):
            continue
        faint = _faintest_detected(stars)
        if math.isfinite(faint):
            diffs.append(faint - limit)
        for s in stars:
            if s['vmag'] <= limit:
                brighter.append(s['detected'])
            elif s['vmag'] <= limit + SEARCH_DEPTH_MAG:
                between.append(s['detected'])
    out: dict[str, Any] = {'n_frames': len(by_frame), 'n_frames_compared': len(diffs)}
    if diffs:
        out['median_faintest_minus_predicted'] = float(np.median(diffs))
        out['p16_p84'] = [float(np.percentile(diffs, 16)), float(np.percentile(diffs, 84))]
    out['detected_fraction_brighter_than_limit'] = float(np.mean(brighter)) if brighter else None
    out['n_brighter'] = len(brighter)
    out['detected_fraction_within_1_mag_past'] = float(np.mean(between)) if between else None
    out['n_between'] = len(between)
    return out


def _frame_table(
    result: FitResult,
    accepted: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    gain_field: str,
) -> list[dict[str, Any]]:
    """Return the per-frame current-versus-proposed table.

    Parameters:
        result: The camera's fit.
        accepted: The accepted frame records.
        rows: Every star row.
        gain_field: The frame field holding the gain state.

    Returns:
        One row per accepted frame.
    """
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if _clean(r):
            by_frame[r['image_id']].append(r)
    table = []
    for record in accepted:
        frame = record['frame']
        dimmest = _dimmest_visible(
            result, frame['filter_name'], frame[gain_field], float(frame['texp']), frame['mode']
        )
        stars = by_frame.get(frame['image_id'], [])
        table.append(
            {
                'image_id': frame['image_id'],
                'filter': frame['filter_name'],
                'gain': frame[gain_field],
                'mode': frame['mode'],
                'texp': float(frame['texp']),
                'held_out': is_held_out(frame['image_id']),
                'n_agree': record['fix']['n_agree'],
                'current_limit': record['current_limit'],
                'deep_limit': record['deep_limit'],
                'proposed_limit': dimmest + SEARCH_DEPTH_MAG,
                'dimmest_visible': dimmest,
                'faintest_detected': _faintest_detected(stars, rank=1),
                'second_faintest_detected': _faintest_detected(stars, rank=2),
                'n_detected': sum(1 for s in stars if s['detected']),
                'n_clean': len(stars),
            }
        )
    return table


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: Iterable[str]) -> None:
    """Write dictionaries as a CSV with the union of their keys as columns.

    An empty table is written as ``columns`` alone, so that a rerun with no rows replaces the
    previous run's table.

    Parameters:
        path: The CSV file.
        rows: The rows.
        columns: The header of an empty table.
    """
    keys: list[str] = [] if rows else list(columns)
    for row in rows:
        keys += [k for k in row if k not in keys and not isinstance(row[k], dict | list)]
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Fit every camera in a records file and write the summary and tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('records', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--reference-filter', required=True)
    parser.add_argument('--reference-gain', required=True)
    parser.add_argument(
        '--gain-field',
        default='gain_state',
        choices=('gain_state', 'mode'),
        help='The frame field holding the gain state',
    )
    parser.add_argument(
        '--min-texp',
        type=float,
        default=0.0,
        help='Leave frames exposed less than this (s) out of the fits',
    )
    parser.add_argument(
        '--limit-exclude-targets',
        nargs='*',
        default=[],
        help='Leave frames of these targets out of the detection limit',
    )
    parser.add_argument(
        '--ee-geometry',
        nargs='*',
        default=[],
        help='MODE:R_MAX:N_BINS of the encircled-energy profile per mode (default 8:16)',
    )
    args = parser.parse_args()
    geometry = {
        g.split(':')[0]: (float(g.split(':')[1]), int(g.split(':')[2])) for g in args.ee_geometry
    }
    records = load_records(args.records)
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_camera[camera_key(record)].append(record)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    frame_rows: list[dict[str, Any]] = []
    for camera, members in sorted(by_camera.items()):
        result = fit_camera(
            camera,
            members,
            reference_filter=args.reference_filter,
            reference_gain=args.reference_gain,
            gain_field=args.gain_field,
            min_texp=args.min_texp,
            limit_exclude_targets=frozenset(args.limit_exclude_targets),
        )
        frame_rows += [{'camera': camera, **row} for row in result.frames]
        for group in result.psf.values():
            r_max, n_bins = geometry.get(group['mode'], (8.0, 16))
            group['gaussian_ee50_ee80'] = list(gaussian_ee(group['sigma_median'], r_max, n_bins))
        summary[camera] = {k: v for k, v in result.__dict__.items() if k != 'frames'} | {
            'n_frames_measured': len(members)
        }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    _write_csv(args.out_dir / 'frames.csv', frame_rows, columns=FRAME_COLUMNS)
    star_rows = [
        {'camera': camera_key(r), **s}
        for r in records
        if r['fix']['accepted']
        for s in _star_rows([r], gain_field=args.gain_field)
    ]
    _write_csv(args.out_dir / 'stars.csv', star_rows, columns=STAR_COLUMNS)
    print(
        json.dumps(
            {
                c: {k: v for k, v in s.items() if k not in ('completeness', 'psf')}
                for c, s in summary.items()
            },
            indent=1,
            default=float,
        )
    )


if __name__ == '__main__':
    main()
