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
   ``2.5 log10(DN / texp) + V`` is a zero point; a robust least-squares fit
   (Tukey biweight) of the zero points on an intercept, one term per filter but the reference
   filter, and one term per gain state but the reference state gives the reference rate (DN
   per second for a V = 0 star through the reference filter in the reference gain state), the
   filter magnitude offsets, and the gain-state DN ratios.
3. **Detection limit.**  Every clean star of the fitting frames gets a predicted DN from the
   fit; the detection fraction in bins of predicted DN gives the smallest DN detected at least
   ``COMPLETENESS`` of the time (``min_detectable_dn``), with a bootstrap over frames for its
   uncertainty.
4. **Held-out check.**  On the frames held out of steps 2 and 3 (one in three, chosen by a hash
   of the image ID), the predicted dimmest visible magnitude is compared with the faintest
   stars detected.
5. **Current versus proposed.**  Per frame: the navigator's current limit, the proposed limit
   (dimmest visible plus ``SEARCH_DEPTH_MAG``), and the faintest star detected.
6. **PSF.**  Per camera and mode (and filter): the circular Gaussian sigma, the elliptical axes,
   the encircled-energy radii, and the narrowest star.

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
from numpy.typing import NDArray
from util.star_photometry.star_field import encircled_energy_radii, integrated_gaussian

__all__ = ['FitResult', 'fit_camera', 'load_records', 'main']

FloatArray = NDArray[np.float64]

#: Smallest flux SNR of a star used in the zero-point fit.
FIT_MIN_FLUX_SNR = 10.0

#: Smallest flux SNR of a star used for the aperture correction.
APCORR_MIN_FLUX_SNR = 30.0

#: Detection fraction defining the faintest reliably detected signal.
COMPLETENESS = 0.8

#: The catalog is searched this far past the dimmest visible star, as the previous
#: navigation software did.
SEARCH_DEPTH_MAG = 1.0

#: Width of the predicted-DN bins of the completeness curve (decades).
DN_BIN_DECADES = 0.1

#: Smallest number of stars in a completeness bin for it to count.
MIN_STARS_PER_BIN = 15

#: One frame in this many is held out of the fit.
HOLDOUT_MODULUS = 3

#: Bootstrap resamples for the uncertainty of the detection limit.
N_BOOTSTRAP = 200

#: Tukey biweight tuning constant (in robust sigmas).
TUKEY_C = 4.685

#: Smallest smear (pixels) of a star excluded from the PSF summary.
PSF_MAX_SMEAR_PX = 0.5


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
        rate_vmag0: DN per second of a V = 0 star, reference filter and gain state.
        rate_sigma_mag: Uncertainty of the rate, in magnitudes.
        filter_offsets: Magnitude offset of each filter from the reference filter.
        filter_offset_sigma: Uncertainty of each filter offset.
        gain_ratio: DN of each gain state per DN of the reference state.
        gain_ratio_sigma_mag: Uncertainty of each gain ratio, in magnitudes.
        zp_scatter_mag: Robust scatter of the zero-point residuals.
        n_zp_stars: Stars in the zero-point fit.
        color_slope: Slope of the zero-point residual against B - V (mag per mag).
        min_detectable_dn: Smallest predicted DN detected ``COMPLETENESS`` of the time.
        min_detectable_dn_sigma: Bootstrap standard deviation of ``min_detectable_dn``.
        min_detectable_dn_50: The same at a detection fraction of one half.
        completeness: Rows ``[log10 DN, n, detected fraction]`` of the completeness curve.
        false_detection_rate: Detected fraction of stars predicted under a tenth of the limit.
        holdout: The held-out check.
        frames: Per-frame rows.
        psf: PSF summary per mode (and filter).
    """

    camera: str
    n_frames: int = 0
    n_fit_frames: int = 0
    apcorr: dict[str, float] = field(default_factory=dict)
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
    completeness: list[list[float]] = field(default_factory=list)
    false_detection_rate: float = float('nan')
    holdout: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    psf: dict[str, Any] = field(default_factory=dict)


def _tukey_lstsq(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve ``a x = b`` by iteratively reweighted least squares with Tukey's biweight.

    Parameters:
        a: Design matrix.
        b: Observations.

    Returns:
        ``(x, sigma_x, residuals)``; ``sigma_x`` from the weighted normal equations scaled by
        the robust residual scatter.
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
    return x, np.sqrt(np.clip(np.diag(cov), 0.0, None)), resid


def _star_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten accepted frames into one row per star with its frame's state.

    Parameters:
        records: Frame records.

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
                    'gain': frame['gain_state'],
                    'mode': frame['mode'],
                    'texp': float(frame['texp']),
                    'held_out': is_held_out(frame['image_id']),
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
) -> FitResult:
    """Fit one camera's photometric limit and summarize its PSF.

    Parameters:
        camera: The camera key.
        records: The camera's frame records.
        reference_filter: The filter the rate refers to (offsets are relative to it).
        reference_gain: The gain state the rate refers to (ratios are relative to it).

    Returns:
        The fit.
    """
    result = FitResult(camera=camera)
    accepted = [r for r in records if r['fix']['accepted']]
    result.n_frames = len(accepted)
    rows = _star_rows(accepted)
    result.apcorr = _apcorr(rows)
    for row in rows:
        row['total_dn'] = row.get('flux_big', float('nan'))
    fit_rows = [
        r
        for r in rows
        if not r['held_out']
        and _clean(r)
        and r['detected']
        and r['flux_big'] > 0.0
        and r['flux_big'] / max(r['flux_big_sigma'], 1e-9) >= FIT_MIN_FLUX_SNR
    ]
    result.n_fit_frames = len(
        {r['frame']['image_id'] for r in accepted if not is_held_out(r['frame']['image_id'])}
    )
    result.n_zp_stars = len(fit_rows)
    if len(fit_rows) < 5:
        result.psf = _psf_summary(rows)
        return result
    _fit_zero_point(
        result, fit_rows, reference_filter=reference_filter, reference_gain=reference_gain
    )
    for row in rows:
        row['pred_dn'] = _predict_dn(result, row)
    _fit_detection_limit(result, [r for r in rows if not r['held_out'] and _clean(r)])
    result.holdout = _holdout_check(result, [r for r in rows if r['held_out'] and _clean(r)])
    result.frames = _frame_table(result, accepted, rows)
    result.psf = _psf_summary(rows)
    return result


def _fit_zero_point(
    result: FitResult,
    rows: list[dict[str, Any]],
    *,
    reference_filter: str,
    reference_gain: str,
) -> None:
    """Fit the reference rate, filter offsets and gain ratios into ``result``.

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
    x, sx, resid = _tukey_lstsq(a, b)
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
    result.filter_counts = dict(sorted(_count(r['filter'] for r in rows).items()))
    result.gain_counts = dict(sorted(_count(r['gain'] for r in rows).items()))
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


def _fit_detection_limit(result: FitResult, rows: list[dict[str, Any]]) -> None:
    """Fit ``min_detectable_dn`` from the completeness of the fitting frames' stars.

    Parameters:
        result: The camera's result, updated in place.
        rows: Clean star rows of the fitting frames with ``pred_dn``.
    """
    rows = [r for r in rows if math.isfinite(r['pred_dn']) and r['pred_dn'] > 0.0]
    if len(rows) < MIN_STARS_PER_BIN * 3:
        return
    centers, counts, fraction = _completeness_curve(rows)
    result.completeness = [
        [float(c), float(n), float(f)]
        for c, n, f in zip(centers, counts, fraction, strict=True)
        if n > 0
    ]
    result.min_detectable_dn = _crossing(centers, counts, fraction, COMPLETENESS)
    result.min_detectable_dn_50 = _crossing(centers, counts, fraction, 0.5)
    if math.isfinite(result.min_detectable_dn):
        faint = [r for r in rows if r['pred_dn'] < 0.1 * result.min_detectable_dn]
        if faint:
            result.false_detection_rate = float(np.mean([r['detected'] for r in faint]))
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[r['image_id']].append(r)
    frames = sorted(by_frame)
    rng = np.random.default_rng(130)
    samples = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(frames), len(frames))
        sample = [r for k in pick for r in by_frame[frames[k]]]
        value = _crossing(*_completeness_curve(sample), COMPLETENESS)
        if math.isfinite(value):
            samples.append(math.log10(value))
    if len(samples) >= N_BOOTSTRAP // 2:
        result.min_detectable_dn_sigma = float(
            result.min_detectable_dn * math.log(10) * np.std(samples)
        )


def _dimmest_visible(result: FitResult, filter_name: str, gain: str, texp: float) -> float:
    """Return the V magnitude whose predicted DN is ``min_detectable_dn``.

    Parameters:
        result: The camera's fit.
        filter_name: The frame's filter.
        gain: The frame's gain state.
        texp: The frame's exposure.

    Returns:
        The magnitude, or NaN for an unfitted filter or gain state.
    """
    if filter_name not in result.filter_offsets or gain not in result.gain_ratio:
        return float('nan')
    if not math.isfinite(result.min_detectable_dn) or texp <= 0.0:
        return float('nan')
    dn_vmag0 = result.rate_vmag0 * result.gain_ratio[gain] * texp
    return (
        2.5 * math.log10(dn_vmag0 / result.min_detectable_dn) - result.filter_offsets[filter_name]
    )


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
        limit = _dimmest_visible(result, first['filter'], first['gain'], first['texp'])
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
    result: FitResult, accepted: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the per-frame current-versus-proposed table.

    Parameters:
        result: The camera's fit.
        accepted: The accepted frame records.
        rows: Every star row.

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
            result, frame['filter_name'], frame['gain_state'], float(frame['texp'])
        )
        stars = by_frame.get(frame['image_id'], [])
        table.append(
            {
                'image_id': frame['image_id'],
                'filter': frame['filter_name'],
                'gain': frame['gain_state'],
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


def _gaussian_ee(sigma: float, r_max: float, n_bins: int) -> tuple[float, float]:
    """Return EE50 and EE80 of a pixel-integrated Gaussian through the same estimator.

    Parameters:
        sigma: The Gaussian sigma (pixels).
        r_max: The profile's outer radius.
        n_bins: The profile's bins.

    Returns:
        ``(ee50, ee80)``.
    """
    half = math.ceil(r_max) + 8
    gv, gu = np.mgrid[-half : half + 1, -half : half + 1].astype(float)
    ee = [
        encircled_energy_radii(
            integrated_gaussian(gv, gu, cv, cu, sigma, sigma),
            half + cv,
            half + cu,
            r_max=r_max,
            n_bins=n_bins,
        )
        for cv, cu in ((0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.5, 0.5))
    ]
    image_ee = np.array(ee)
    return float(np.mean(image_ee[:, 0])), float(np.mean(image_ee[:, 1]))


def _psf_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the PSF fits per mode, and per mode and filter.

    Parameters:
        rows: Star rows.

    Returns:
        Group name to its summary.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        psf = r.get('psf')
        if psf is None or r['smear_px'] > PSF_MAX_SMEAR_PX or not math.isfinite(psf['ee50']):
            continue
        groups[f'mode {r["mode"]}'].append(r)
        groups[f'mode {r["mode"]} filter {r["filter"]}'].append(r)
    out: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        if len(members) < 3:
            continue
        sig = np.array([m['psf']['sigma'] for m in members])
        major = np.array([m['psf']['sigma_major'] for m in members])
        minor = np.array([m['psf']['sigma_minor'] for m in members])
        ee50 = np.array([m['psf']['ee50'] for m in members])
        ee80 = np.array([m['psf']['ee80'] for m in members])
        resid = np.array([m['psf']['rms_residual'] for m in members])
        out[name] = {
            'n_stars': len(members),
            'n_frames': len({m['image_id'] for m in members}),
            'sigma_median': float(np.median(sig)),
            'sigma_p16_p84': [float(np.percentile(sig, 16)), float(np.percentile(sig, 84))],
            'sigma_min': float(sig.min()),
            'sigma_p2': float(np.percentile(sig, 2)),
            'major_median': float(np.median(major)),
            'minor_median': float(np.median(minor)),
            'axis_ratio_median': float(np.median(major / minor)),
            'ee50_median': float(np.median(ee50)),
            'ee80_median': float(np.median(ee80)),
            'ee80_over_ee50': float(np.median(ee80) / np.median(ee50)),
            'fit_rms_residual_median': float(np.median(resid)),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a CSV with the union of their keys as columns.

    Parameters:
        path: The CSV file.
        rows: The rows.
    """
    if not rows:
        return
    keys: list[str] = []
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
        )
        frame_rows += [{'camera': camera, **row} for row in result.frames]
        for name, group in result.psf.items():
            r_max, n_bins = geometry.get(name.split()[1], (8.0, 16))
            group['gaussian_ee50_ee80'] = list(_gaussian_ee(group['sigma_median'], r_max, n_bins))
        summary[camera] = {k: v for k, v in result.__dict__.items() if k != 'frames'} | {
            'n_frames_measured': len(members)
        }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    _write_csv(args.out_dir / 'frames.csv', frame_rows)
    star_rows = [
        {'camera': camera_key(r), **s}
        for r in records
        if r['fix']['accepted']
        for s in _star_rows([r])
    ]
    _write_csv(args.out_dir / 'stars.csv', star_rows)
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
