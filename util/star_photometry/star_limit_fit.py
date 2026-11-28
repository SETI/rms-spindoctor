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
   filter magnitude offsets, and the gain-state DN ratios.  For a camera given to
   ``--color-term``, the fit also gives the camera's color term ``c`` from the stars with a
   measured color (:mod:`util.star_photometry.star_zero_point`): a star's camera magnitude
   is ``V + c (B - V)``, with the ``B - V`` the navigator gives the star, and every later
   step reads it in place of V.  Without it the camera magnitude is V.  A fit whose stars
   cannot
   separate the terms (a design matrix short of full rank, before or after the robust
   weighting) is reported as unidentified, with no coefficients.  A filter or gain state with
   fewer than ``MIN_STARS_PER_BIN`` fitting stars is not given a term: its stars are left
   out, it takes the calibration document's value (``--calibration-offsets``,
   ``--calibration-gain-ratios``), and it is reported as not measured
   (:mod:`util.star_photometry.star_zero_point`).  Its stars are left out of steps 3 and 4
   as well, since no measured term anchors their predicted DN.
3. **Detection limit.**  Every clean star of the fitting frames gets a predicted DN from the
   fit; a detection-probability model (a floor, a plateau, and a normal-integral rise in
   log DN) fitted to the stars' detections gives the DN at which detection is ``COMPLETENESS`` of
   the way from floor to plateau (``min_detectable_dn``), with a bootstrap over frames for its
   uncertainty.  The limit is left undefined, as insufficient data, unless there are three
   times ``MIN_STARS_PER_BIN`` stars, detections and non-detections each number
   ``MIN_STARS_PER_BIN``, the stars' predicted DN spread over ``MIN_LOG_DN_SPAN`` decades,
   and the fitted rise lies inside that spread.  The same holds for each readout mode, and,
   with ``--limit-by-gain``, for each readout mode and gain state, where a pair without
   enough stars is scaled from another gain state of its mode
   (:func:`util.star_photometry.star_completeness.fit_gain_limits`).  The model itself is
   :mod:`util.star_photometry.star_completeness`.  ``--limit-exclude-smeared`` leaves
   smeared stars out of this step and the next, and ``--limit-max-texp`` frames exposed
   longer than it.
4. **Held-out check.**  On the frames held out of steps 2 and 3 (one in three, chosen by a hash
   of the image ID), the predicted dimmest visible magnitude is compared with the faintest
   stars detected.
5. **Current versus proposed.**  Per frame: the navigator's current limit, the proposed limit
   (dimmest visible plus ``--search-depth-mag``, the camera's search depth), and the faintest
   star detected.
6. **PSF.**  Per camera and mode (and filter): the circular Gaussian sigma, the elliptical axes,
   the encircled-energy radii, and the narrowest star
   (:mod:`util.star_photometry.star_psf_summary`).

Usage::

    python -m util.star_photometry.star_limit_fit RECORDS.jsonl OUT_DIR --reference-filter CLEAR
        --reference-gain 10K [--search-depth-mag 1.0] [--color-term CAMERA ...]
        [--limit-exclude-smeared] [--limit-max-texp SECONDS] [--limit-by-gain]
        [--ee-geometry MODE:R_MAX:N_BINS ...]

writes ``summary.json`` (every fitted quantity), ``frames.csv`` (the per-frame table), and
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
from util.star_photometry.star_completeness import (
    COMPLETENESS,
    MIN_STARS_PER_BIN,
    bootstrap_sigma,
    completeness_curve,
    crossing,
    fit_gain_limits,
    identified_model,
    model_dn,
)
from util.star_photometry.star_psf_summary import gaussian_ee, psf_summary
from util.star_photometry.star_zero_point import fit_zero_point

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

#: One frame in this many is held out of the fit.
HOLDOUT_MODULUS = 3

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
        n_fit_frames: Of those, frames with a star in the zero-point fit.
        apcorr: Aperture correction per mode.
        zero_point_identified: False when the fitting stars cannot separate the intercept,
            the filter terms, and the gain terms (for example, every star through one
            filter taken in one gain state and every other star in the other); the rate,
            offsets, and ratios are then left out.
        rate_vmag0: DN per second of a V = 0 star, reference filter and gain state.
        rate_sigma_mag: Uncertainty of the rate, in magnitudes.
        color_term: The color term ``c`` of the camera magnitude ``V + c (B - V)``; zero
            unless it was fitted.
        color_term_sigma: Uncertainty of the color term; NaN unless it was fitted.
        filter_offsets: Magnitude offset of each filter from the reference filter.
        filter_offset_sigma: Uncertainty of each filter offset.
        gain_ratio: DN of each gain state per DN of the reference state.
        gain_ratio_sigma_mag: Uncertainty of each gain ratio, in magnitudes.
        filters_not_measured: Filters with too few stars for a term, whose offset is the
            calibration document's (in ``filter_offsets``, with no uncertainty).
        gains_not_measured: The same for gain states and ``gain_ratio``.
        n_limit_stars_unmeasured: Clean stars left out of the detection limit and the
            held-out check because their filter or gain state is not measured or has no
            value at all.
        zp_scatter_mag: Robust scatter of the zero-point residuals.
        n_zp_stars: Stars in the zero-point fit.
        color_slope: Slope of the zero-point residual against B - V (mag per mag), after the
            color term.
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
            ``'all'``) and for each readout mode whose stars cannot identify the rise,
            and, with ``limit_by_gain``, for each such pair of mode and gain state (key
            ``'MODE/GAIN'``).
        min_detectable_dn_by_mode_gain: With ``limit_by_gain``, the limit per readout
            mode and gain state (see
            :func:`util.star_photometry.star_completeness.fit_gain_limits`); a frame of
            such a pair takes the pair's DN.
        min_detectable_dn_scaled_from: For each pair in ``min_detectable_dn_by_mode_gain``
            that was scaled rather than fitted, the gain state it was scaled from.
        completeness_model: ``[floor, plateau, mu, width]`` of the camera's fit.
        completeness: Rows ``[log10 DN, n, detected fraction]`` of the completeness curve.
        false_detection_rate: The fitted detection probability's floor.
        holdout: The held-out check.
        search_depth_mag: How far past the dimmest visible star the camera's search reaches.
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
    color_term: float = 0.0
    color_term_sigma: float = float('nan')
    filter_offsets: dict[str, float] = field(default_factory=dict)
    filter_offset_sigma: dict[str, float] = field(default_factory=dict)
    filter_counts: dict[str, int] = field(default_factory=dict)
    gain_ratio: dict[str, float] = field(default_factory=dict)
    gain_ratio_sigma_mag: dict[str, float] = field(default_factory=dict)
    gain_counts: dict[str, int] = field(default_factory=dict)
    filters_not_measured: list[str] = field(default_factory=list)
    gains_not_measured: list[str] = field(default_factory=list)
    n_limit_stars_unmeasured: int = 0
    zp_scatter_mag: float = float('nan')
    n_zp_stars: int = 0
    color_slope: float = float('nan')
    min_detectable_dn: float = float('nan')
    min_detectable_dn_sigma: float = float('nan')
    min_detectable_dn_50: float = float('nan')
    min_detectable_dn_by_mode: dict[str, list[float]] = field(default_factory=dict)
    min_detectable_dn_empirical: float = float('nan')
    min_detectable_dn_insufficient: dict[str, str] = field(default_factory=dict)
    min_detectable_dn_by_mode_gain: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    min_detectable_dn_scaled_from: dict[str, dict[str, str]] = field(default_factory=dict)
    completeness_model: list[float] = field(default_factory=list)
    completeness: list[list[float]] = field(default_factory=list)
    false_detection_rate: float = float('nan')
    holdout: dict[str, Any] = field(default_factory=dict)
    search_depth_mag: float = 1.0
    frames: list[dict[str, Any]] = field(default_factory=list)
    psf: dict[str, Any] = field(default_factory=dict)


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
    limit_exclude_smeared: bool = False,
    limit_max_texp: float = math.inf,
    limit_by_gain: bool = False,
    search_depth_mag: float = 1.0,
    calibration_offsets: dict[str, float] | None = None,
    calibration_gain_ratios: dict[str, float] | None = None,
    color_term: bool = False,
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
        limit_exclude_smeared: Whether stars flagged smeared are left out of the detection
            limit and the held-out check (a frame that tracks a moving target smears every
            star, and a smeared star needs more DN to be detected).
        limit_max_texp: Frames exposed longer than this (seconds) are left out of the
            detection limit and the held-out check (they stay in the zero point): on a
            camera whose long exposures carry several times the noise of a short one, they
            would set the limit of every frame.
        limit_by_gain: Whether the detection limit is also fitted per readout mode and gain
            state, for a camera whose noise in DN changes with its gain state.
        search_depth_mag: The camera's search depth past the dimmest visible star, for the
            proposed limit and the held-out check.
        calibration_offsets: The calibration document's filter offsets, for filters with
            too few stars to fit.
        calibration_gain_ratios: The calibration document's gain ratios, for gain states
            with too few stars to fit.
        color_term: Whether to fit the camera's color term.

    Returns:
        The fit.
    """
    result = FitResult(camera=camera, search_depth_mag=search_depth_mag)
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
    result.n_fit_frames = len({r['image_id'] for r in fit_rows})
    result.n_zp_stars = len(fit_rows)
    if len(fit_rows) < 5:
        # The frames stay in the table, with the fitted columns NaN.
        _set_camera_mags(result, rows)
        result.frames = _frame_table(result, accepted, rows, gain_field=gain_field)
        result.psf = psf_summary(rows)
        return result
    fit_zero_point(
        result,
        fit_rows,
        reference_filter=reference_filter,
        reference_gain=reference_gain,
        min_term_stars=MIN_STARS_PER_BIN,
        calibration_offsets=calibration_offsets or {},
        calibration_gain_ratios=calibration_gain_ratios or {},
        color_term=color_term,
    )
    _set_camera_mags(result, rows)
    for row in rows:
        row['pred_dn'] = _predict_dn(result, row)
    limit_rows = _measured_terms(
        result,
        [
            r
            for r in rows
            if r['texp'] >= min_texp
            and r['target'] not in limit_exclude_targets
            and _clean(r)
            and not (limit_exclude_smeared and r['smeared'])
            and r['texp'] <= limit_max_texp
        ],
    )
    _fit_detection_limit(result, [r for r in limit_rows if not r['held_out']])
    if limit_by_gain:
        (
            result.min_detectable_dn_by_mode_gain,
            result.min_detectable_dn_scaled_from,
            reasons,
        ) = fit_gain_limits(
            [r for r in limit_rows if not r['held_out'] and _positive_dn(r)],
            result.gain_ratio,
        )
        result.min_detectable_dn_insufficient.update(reasons)
    result.holdout = _holdout_check(result, [r for r in limit_rows if r['held_out']])
    result.frames = _frame_table(result, accepted, rows, gain_field=gain_field)
    result.psf = psf_summary(rows)
    return result


def _measured_terms(result: FitResult, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows whose filter and gain state were both measured by the zero point.

    A star through a filter or gain state that the stars could not pin down is predicted
    with the calibration document's value, which nothing anchors, so it is left out of the
    detection limit and the held-out check, as it is left out of the zero point.  So is a
    star through a filter or gain state for which neither the fit nor the calibration
    document gives a value (one the fitting stars never saw, such as a state taken only on
    held-out frames), since its DN cannot be predicted.  The count left out is stored in
    ``result.n_limit_stars_unmeasured``.

    Parameters:
        result: The camera's fit, updated in place.
        rows: Candidate rows.

    Returns:
        The rows kept.
    """
    kept = [
        r
        for r in rows
        if r['filter'] not in result.filters_not_measured
        and r['gain'] not in result.gains_not_measured
        and r['filter'] in result.filter_offsets
        and r['gain'] in result.gain_ratio
    ]
    result.n_limit_stars_unmeasured = len(rows) - len(kept)
    return kept


def _set_camera_mags(result: FitResult, rows: list[dict[str, Any]]) -> None:
    """Give each star row its camera magnitude, ``V + c (B - V)``, as ``camera_mag``.

    The color is the ``b_minus_v`` the navigator gives the star, which a star without a
    measured color takes from its star class, as the navigator does.

    Parameters:
        result: The camera's fit, whose ``color_term`` is ``c``.
        rows: The star rows, updated in place.
    """
    for row in rows:
        row['camera_mag'] = row['vmag'] + result.color_term * row['b_minus_v']


def _predict_dn(result: FitResult, row: dict[str, Any]) -> float:
    """Return the fitted total DN of a star, or NaN for an unfitted filter or gain state.

    Parameters:
        result: The camera's fit.
        row: The star row, with its ``camera_mag``.

    Returns:
        The predicted DN.
    """
    if row['filter'] not in result.filter_offsets or row['gain'] not in result.gain_ratio:
        return float('nan')
    magnitude = row['camera_mag'] + result.filter_offsets[row['filter']]
    rate = result.rate_vmag0 * result.gain_ratio[row['gain']]
    return float(rate * 10 ** (-0.4 * magnitude) * row['texp'])


def _positive_dn(row: dict[str, Any]) -> bool:
    """Return whether a star row has a finite, positive predicted DN.

    Parameters:
        row: The star row, with its ``pred_dn``.

    Returns:
        True when the detection limit can use the row.
    """
    return math.isfinite(row['pred_dn']) and row['pred_dn'] > 0.0


def _fit_detection_limit(result: FitResult, rows: list[dict[str, Any]]) -> None:
    """Fit ``min_detectable_dn`` from the completeness of the fitting frames' stars.

    The detection probability model of :func:`fit_completeness_model` is fitted, and the
    limit is the DN at which it covers ``COMPLETENESS`` of its span from floor to plateau.
    The limit is fitted on all the camera's stars and, where a readout mode has enough stars,
    on that mode's alone.  The binned curve and its empirical crossings are kept as a check.
    Where the stars cannot identify the rise (see :func:`identified_model`), the limit is
    left undefined and the reason is stored in ``min_detectable_dn_insufficient``.

    Parameters:
        result: The camera's result, updated in place.
        rows: Clean star rows of the fitting frames with ``pred_dn``.
    """
    modes = sorted({r['mode'] for r in rows})
    rows = [r for r in rows if math.isfinite(r['pred_dn']) and r['pred_dn'] > 0.0]
    for mode in modes:
        members = [r for r in rows if r['mode'] == mode]
        mode_model, reason = identified_model(members)
        if mode_model is None:
            result.min_detectable_dn_insufficient[mode] = reason
            continue
        value = model_dn(mode_model, COMPLETENESS)
        result.min_detectable_dn_by_mode[mode] = [
            value,
            bootstrap_sigma(members, value),
            model_dn(mode_model, 0.5),
            float(len(members)),
            *mode_model,
        ]
    # The binned curve and its empirical crossing are kept even where the model cannot be
    # identified, since they do not need one.
    if rows:
        centers, counts, fraction = completeness_curve(rows)
        result.completeness = [
            [float(c), float(n), float(f)]
            for c, n, f in zip(centers, counts, fraction, strict=True)
            if n > 0
        ]
        result.min_detectable_dn_empirical = crossing(centers, counts, fraction, COMPLETENESS)
    model, reason = identified_model(rows)
    if model is None:
        result.min_detectable_dn_insufficient['all'] = reason
        return
    result.completeness_model = list(model)
    result.min_detectable_dn = model_dn(model, COMPLETENESS)
    result.min_detectable_dn_50 = model_dn(model, 0.5)
    result.false_detection_rate = model[0]
    result.min_detectable_dn_sigma = bootstrap_sigma(rows, result.min_detectable_dn)


def _dimmest_visible(
    result: FitResult, filter_name: str, gain: str, texp: float, mode: str
) -> float:
    """Return the camera magnitude whose predicted DN is the detection limit.

    The limit is the DN of the frame's pair of readout mode and gain state when the fit has
    one, else the mode's ``min_detectable_dn`` when the mode has one, else the camera's.

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
    by_gain = result.min_detectable_dn_by_mode_gain.get(mode, {})
    if gain in by_gain:
        limit = by_gain[gain][0]
    else:
        limit = result.min_detectable_dn_by_mode.get(mode, [result.min_detectable_dn])[0]
    if not math.isfinite(limit) or texp <= 0.0:
        return float('nan')
    dn_vmag0 = result.rate_vmag0 * result.gain_ratio[gain] * texp
    return 2.5 * math.log10(dn_vmag0 / limit) - result.filter_offsets[filter_name]


def _faintest_detected(stars: list[dict[str, Any]], rank: int = 2) -> float:
    """Return the ``rank``-th faintest detected clean star's camera magnitude.

    The second faintest, the default, is robust to one fluke.

    Parameters:
        stars: Star rows of one frame.
        rank: Which faintest (1 = the faintest).

    Returns:
        The magnitude, or NaN when fewer stars were detected.
    """
    mags = sorted((s['camera_mag'] for s in stars if s['detected']), reverse=True)
    return float(mags[rank - 1]) if len(mags) >= rank else float('nan')


def _holdout_check(result: FitResult, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the predicted dimmest visible magnitude with the held-out frames' detections.

    For each held-out frame with at least two detected clean stars, the difference is the
    second-faintest detected magnitude less the predicted dimmest visible magnitude.  Over all
    held-out stars together, the detection fraction of stars predicted brighter than the
    dimmest visible magnitude, and of stars predicted between it and the camera's search depth
    past it.

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
            if s['camera_mag'] <= limit:
                brighter.append(s['detected'])
            elif s['camera_mag'] <= limit + result.search_depth_mag:
                between.append(s['detected'])
    out: dict[str, Any] = {'n_frames': len(by_frame), 'n_frames_compared': len(diffs)}
    if diffs:
        out['median_faintest_minus_predicted'] = float(np.median(diffs))
        out['p16_p84'] = [float(np.percentile(diffs, 16)), float(np.percentile(diffs, 84))]
    out['detected_fraction_brighter_than_limit'] = float(np.mean(brighter)) if brighter else None
    out['n_brighter'] = len(brighter)
    out['detected_fraction_within_search_depth'] = float(np.mean(between)) if between else None
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
                'proposed_limit': dimmest + result.search_depth_mag,
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


def _calibration(items: list[str], camera: str) -> dict[str, float]:
    """Return a camera's calibration values from ``[CAMERA:]NAME=VALUE`` items.

    Parameters:
        items: The items; one without a camera applies to every camera.
        camera: The camera key.

    Returns:
        Name to value.
    """
    out: dict[str, float] = {}
    for item in items:
        scope, _, assignment = item.rpartition(':')
        if scope in ('', camera):
            name, value = assignment.split('=')
            out[name] = float(value)
    return out


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
        '--limit-exclude-smeared',
        action='store_true',
        help='Leave smeared stars out of the detection limit and the held-out check',
    )
    parser.add_argument(
        '--limit-max-texp',
        type=float,
        default=math.inf,
        help='Leave frames exposed longer than this (s) out of the detection limit',
    )
    parser.add_argument(
        '--limit-by-gain',
        action='store_true',
        help='Also fit the detection limit per readout mode and gain state',
    )
    parser.add_argument(
        '--calibration-offsets',
        nargs='*',
        default=[],
        help='[CAMERA:]FILTER=MAG calibration offsets for filters with too few stars',
    )
    parser.add_argument(
        '--calibration-gain-ratios',
        nargs='*',
        default=[],
        help='[CAMERA:]GAIN=RATIO calibration ratios for gain states with too few stars',
    )
    parser.add_argument(
        '--search-depth-mag', type=float, default=1.0, help='The camera search depth (mag)'
    )
    parser.add_argument(
        '--color-term',
        nargs='*',
        default=[],
        metavar='CAMERA',
        help='Fit a color term for these cameras',
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
            limit_exclude_smeared=args.limit_exclude_smeared,
            limit_max_texp=args.limit_max_texp,
            limit_by_gain=args.limit_by_gain,
            search_depth_mag=args.search_depth_mag,
            calibration_offsets=_calibration(args.calibration_offsets, camera),
            calibration_gain_ratios=_calibration(args.calibration_gain_ratios, camera),
            color_term=camera in args.color_term,
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
