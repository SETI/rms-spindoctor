"""Measure the simulator's Cassini ISS star zero points on real star fields.

The zero point is the electrons a star of Johnson V = 0 deposits each second through the
clear filters (``star_flux.e_per_s_vmag0`` in the simulator's Cassini instrument entries).

**Frames.**  ``measure`` samples frames from the cumulative PDS3 index ``COISS_2999``: target
``SKY``, 12-bit (lossless or uncompressed, never the 12-to-8-bit table), shutter enabled,
calibration lamp off, inside the Saturn-tour SPICE window, and through the filter
combinations asked for (``CL1+CL2`` by default).  The sample is stratified by camera, gain
state, readout mode, and exposure octave, at most ``--per-group`` frames each.  With
``--library`` the image library's Cassini frames are added; their operator-verified offsets
check the independent pointing fix.

**Measurement.**  Each frame is read twice.  The calibrated product, through the navigator's
own reader, is the image the pointing is fixed on
(:func:`util.star_photometry.frame_measure.measure_frame`, which needs no navigator star
technique).  The raw product beside it is the photometric image: a star's light within
radii ``r``, ``2r``, and ``3r`` of its fixed position, less the clipped mean of a sky
annulus outside ``3r``, in raw DN (a median would be held to the raw frame's integer
levels).  The same apertures on the calibrated product, less the annulus's median, give
the star's I/F sums.  The record keeps the gain and the shutter offset CISSCAL wrote into the
calibrated label, and the Sun's distance from Saturn.

**Fit.**  ``fit`` gives each star two rates.  The raw rate, the zero point's, is the raw DN
within ``r`` times the camera and readout mode's aperture correction (the median ratio of
the light within ``2r`` to the light within ``r`` over bright clean stars), times CISSCAL's
gain, over the exposure less CISSCAL's shutter offset: the electrons the detector collected
each second.  The calibrated rate, the check, is the I/F sum within ``r`` times the
calibrated aperture correction, times the binning squared (a summed pixel's I/F is the
mean of the native pixels it sums), times the simulator's clear-filter throughput: the
rate at which a calibrated simulated frame would show the star as the real one does.  It
carries CISSCAL's division by the sunlight at the frame's Sun distance, so it grows with
that distance where the raw rate does not.  Stars are
chosen by catalog magnitude, never by measured flux: UCAC4 stars with an APASS V of
``FIT_VMAG_RANGE``, unblended, clear of bodies and rings, unsmeared, whose predicted signal
at a provisional rate reaches ``MIN_PREDICTED_SNR`` and whose predicted peak stays under
``MAX_PEAK_FRACTION`` of saturation.  :func:`util.star_photometry.star_zero_point.fit_star_rates`
gives the rate of a star of B - V ``REFERENCE_COLOR`` and tests the residuals against gain
state, readout mode, exposure, magnitude, and field position.

Usage::

    python -m util.star_photometry.cassini_iss_star_zero_points measure OUT.jsonl
        [--per-group 6] [--filters CL1+CL2 ...] [--library] [--workers 4] [--limit N]
        [--only ID ...]
    python -m util.star_photometry.cassini_iss_star_zero_points fit RECORDS.jsonl
        [--stars-csv STARS.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pdstable import PdsTable
from ruamel.yaml import YAML
from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)
from util.star_photometry.star_field import aperture_photometry
from util.star_photometry.star_zero_point import RateFit, fit_star_rates
from vicar import VicarImage

from spindoctor.obs import ObsCassiniISS
from spindoctor.sim.instruments.catalog import instrument_entry

__all__ = [
    'PARAMS',
    'CameraFit',
    'StarRows',
    'calibration_terms',
    'cassini_frames',
    'fit_camera',
    'library_frames',
    'main',
    'measure',
    'star_rows',
]

FloatArray = NDArray[np.float64]

#: The Saturn-tour epochs the local SPICE kernels cover (inclusive).
SPICE_START, SPICE_END = date(2004, 1, 1), date(2017, 9, 15)

#: The shortest exposure sampled (seconds): shorter frames hold few stars, and the
#: shutter offset is a large part of them.
MIN_TEXP_SEC = 0.05

#: The native frame size of both cameras; a frame summed by b is this over b.
NATIVE_SIZE_PX = 1024

#: The 12-bit ADC's largest DN.
ADC_MAX_DN = 4095.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: The simulator instrument of each camera (raw products; the calibrated entries share
#: its star zero point, gains, full well, and throughputs).
SIM_INSTRUMENT = {'NAC': 'coiss_nac', 'WAC': 'coiss_wac'}

#: The filter combination the zero point refers to.
CLEAR = 'CL1+CL2'

#: The B - V the zero point refers to: a solar-type star, as for the other cameras.
REFERENCE_COLOR = 0.65

#: The Johnson V range of the fitting stars.  APASS V reads too faint for stars brighter
#: than about V 10, where APASS saturates.
FIT_VMAG_RANGE = (10.0, 14.0)

#: The smallest predicted SNR, at the provisional rate, of a fitting star.
MIN_PREDICTED_SNR = 10.0

#: A fitting star's predicted peak, sky included, stays under this fraction of saturation.
MAX_PEAK_FRACTION = 0.6

#: A measured star whose brightest pixel, less the sky, reaches this fraction of saturation
#: is saturated.
SATURATED_FRACTION = 0.9

#: Stars predicted to move further than this during the exposure (frame pixels) are smeared.
MAX_SMEAR_PX = 1.0

#: A frame whose fitted stars' median ratio of major to minor PSF axis exceeds this is
#: smeared, whatever the predicted motion says: the reconstructed attitude can miss a drift.
MAX_ELONGATION = 1.5

#: The fewest fitted stars the elongation test needs.
MIN_ELONGATION_STARS = 3

#: The smallest number of agreeing stars of a frame's pointing fix used in the fit.
MIN_FIX_AGREE = 5

#: The largest RMS of a pointing fix's agreeing stars used in the fit (frame pixels).
MAX_FIX_SPREAD_PX = 0.7

#: The smallest flux SNR within ``2r`` of a star used for the aperture correction and the
#: peak fraction: the recorded ``flux_big`` over ``flux_big_sigma``, whose noise is the
#: sky's alone.  On full NAC frames the ratio rises as the SNR falls, to 1.08 at SNR 50 to
#: 70 and 1.12 at 30 to 50, against 1.05 to 1.07 above 70, because hot pixels, cosmic rays,
#: and faint stars in the larger aperture weigh more against a fainter star.  The limit
#: keeps clear of that rise.
APCORR_MIN_SNR = 100.0

#: The provisional rates (e-/s for V = 0) that the selection's predicted signal starts from:
#: the clear-filter throughput times the flux of an I/F 1 pixel at Saturn expressed as a
#: magnitude (V 5.5 for a NAC pixel, V 0.5 for a WAC pixel).
PROVISIONAL_RATE = {'NAC': 2.3e8, 'WAC': 2.1e7}

#: Passes of the selection, each starting from the previous pass's rate.
SELECTION_PASSES = 3

#: Kilometers per astronomical unit.
_KM_PER_AU = 1.495978707e8

#: Milliseconds per second (the index and the labels state times in milliseconds).
_MS_PER_S = 1000.0

#: The aperture radius ``r`` of each readout mode (frame pixels).
_APERTURE_BY_BINNING = {1: 4.0, 2: 3.0, 4: 2.0}

#: The navigator's pointing-search half-width of each camera (native pixels).
_SEARCH_NATIVE_PX = {'NAC': 200.0, 'WAC': 40.0}

#: Label patterns of the terms CISSCAL wrote into a calibrated product's label.
_GAIN_RE = re.compile(r"GAIN_CORRECTION\s*=\s*'([0-9.]+)\s*e-/DN'")
_OFFSET_RE = re.compile(r"EXPOSURE_OFFSET\s*=\s*'([-0-9.]+)\s*ms'")

#: Image library classes whose Cassini frames hold stars.
LIBRARY_STAR_CLASSES = (
    'star_dominated',
    'one_bright_star_no_body',
    'two_bright_stars_no_body',
    'faint_stars',
    'stars_plus_body',
    'scattered_light',
)


def _params(camera: str, binning: int) -> MeasureParams:
    """Return the measurement parameters of a camera and readout mode.

    Parameters:
        camera: ``'NAC'`` or ``'WAC'``.
        binning: The summation factor, 1, 2, or 4.

    Returns:
        The parameters; the sky annulus lies outside ``3r``.
    """
    r = _APERTURE_BY_BINNING[binning]
    return MeasureParams(
        catalog_depth_mag=4.0,
        highpass_size=max(31 // binning, 9) | 1,
        smooth_sigma=1.2 if binning == 1 else 0.8,
        peak_threshold_snr=5.0,
        edge_px=int(3 * r + 10),
        max_peaks=300,
        max_vote_stars=60,
        search_px=_SEARCH_NATIVE_PX[camera] / binning,
        vote_bin_px=1.5,
        agree_px=1.0,
        min_agree=3,
        max_chance_probability=1e-3,
        detect_search_px=1.5,
        noise_inner=8.0 / binning**0.5,
        noise_outer=15.0 / binning**0.5,
        box_half=2 if binning == 1 else 1,
        aperture_px=r,
        big_aperture_px=2.0 * r,
        sky_inner=3.0 * r + 2.0,
        sky_outer=3.0 * r + 8.0,
        blend_px=3.0 * r + 8.0,
        blend_dmag=3.0,
        saturation_dn=ADC_MAX_DN,
        max_smear_px=MAX_SMEAR_PX,
        psf_min_snr=20.0,
        psf_half=int(2 * r),
        ee_r_max=2.0 * r,
        ee_n_bins=16,
        missing_value=0.0,
    )


#: Measurement parameters by camera and summation factor.
PARAMS: dict[tuple[str, int], MeasureParams] = {
    (camera, binning): _params(camera, binning)
    for camera in ('NAC', 'WAC')
    for binning in (1, 2, 4)
}


#: The index columns the sample reads.
_INDEX_COLUMNS = (
    'VOLUME_ID',
    'FILE_SPECIFICATION_NAME',
    'FILE_NAME',
    'INSTRUMENT_ID',
    'FILTER_NAME',
    'GAIN_MODE_ID',
    'INSTRUMENT_MODE_ID',
    'EXPOSURE_DURATION',
    'IMAGE_TIME',
    'OBSERVATION_ID',
    'DATA_CONVERSION_TYPE',
    'TARGET_NAME',
    'SHUTTER_STATE_ID',
    'CALIBRATION_LAMP_STATE_FLAG',
)


def _index_rows(holdings: Path) -> list[dict[str, Any]]:
    """Return the rows of the cumulative Cassini ISS index, in index order.

    Parameters:
        holdings: The PDS3 holdings root.

    Returns:
        One dictionary per row, keyed by column name: ``FILTER_NAME`` the filter
        combination (``'CL1+CL2'``), ``EXPOSURE_DURATION`` a float in milliseconds, and
        every other column a string.
    """
    label = holdings / 'metadata' / 'COISS_2xxx' / 'COISS_2999' / 'COISS_2999_index.lbl'
    table = PdsTable(str(label), columns=list(_INDEX_COLUMNS), label_method='fast')
    rows = []
    for raw in table.dicts_by_row():
        row: dict[str, Any] = {name: str(raw[name]).strip() for name in _INDEX_COLUMNS}
        row['FILTER_NAME'] = '+'.join(str(f).strip() for f in raw['FILTER_NAME'])
        row['EXPOSURE_DURATION'] = float(raw['EXPOSURE_DURATION'])
        rows.append(row)
    return rows


def _frame_from_row(holdings: Path, row: dict[str, Any], *, source: str) -> FrameInput:
    """Return the frame of an index row: its calibrated product, the raw one beside it.

    Parameters:
        holdings: The PDS3 holdings root.
        row: The index row.
        source: Where the frame was chosen from (``'sample'`` or ``'library'``).

    Returns:
        The frame.
    """
    volume = row['VOLUME_ID']
    spec = Path(row['FILE_SPECIFICATION_NAME']).with_suffix('')
    raw = holdings / 'volumes' / 'COISS_2xxx' / volume / f'{spec}.IMG'
    calib = holdings / 'calibrated' / 'COISS_2xxx' / volume / f'{spec}_CALIB.IMG'
    return FrameInput(
        image_id=spec.name,
        path=str(calib),
        camera='NAC' if row['INSTRUMENT_ID'] == 'ISSNA' else 'WAC',
        filter_name=row['FILTER_NAME'],
        gain_state=row['GAIN_MODE_ID'].split()[0],
        mode=row['INSTRUMENT_MODE_ID'],
        texp=row['EXPOSURE_DURATION'] / _MS_PER_S,
        extra={
            'raw_path': str(raw),
            'volume': volume,
            'source': source,
            'image_time': row['IMAGE_TIME'],
            'observation_id': row['OBSERVATION_ID'],
            'conversion': row['DATA_CONVERSION_TYPE'],
        },
    )


def _image_date(image_time: str) -> date:
    """Return the date of an index ``IMAGE_TIME`` (``YYYY-DDDThh:mm:ss.sss``, day of year).

    Parameters:
        image_time: The index's image time.

    Returns:
        The date.
    """
    return datetime.strptime(image_time[:8], '%Y-%j').date()


def _eligible(row: dict[str, Any], filters: frozenset[str]) -> bool:
    """Return whether an index row is a sampleable star-field frame.

    Parameters:
        row: The index row.
        filters: The filter combinations accepted.

    Returns:
        True for a 12-bit ``SKY`` frame through an accepted filter combination, shutter
        enabled and lamp off, inside the SPICE window, of at least ``MIN_TEXP_SEC``.
    """
    return (
        row['TARGET_NAME'] == 'SKY'
        and row['DATA_CONVERSION_TYPE'] == '12BIT'
        and row['SHUTTER_STATE_ID'] == 'ENABLED'
        and row['CALIBRATION_LAMP_STATE_FLAG'] in ('OFF', 'N/A')
        and row['FILTER_NAME'] in filters
        and SPICE_START <= _image_date(row['IMAGE_TIME']) <= SPICE_END
        and row['EXPOSURE_DURATION'] / _MS_PER_S >= MIN_TEXP_SEC
    )


def cassini_frames(
    holdings: Path, *, filters: frozenset[str], per_group: int, seed: int = 1294
) -> list[FrameInput]:
    """Return a stratified sample of Cassini ISS star-field frames.

    Frames are grouped by camera, filter combination, gain state, readout mode, and
    exposure octave, and at most ``per_group`` frames are drawn from each group, spread
    over the group's observations.

    Parameters:
        holdings: The PDS3 holdings root.
        filters: The filter combinations accepted.
        per_group: The most frames drawn from one group.
        seed: The seed of the draw.

    Returns:
        The sampled frames.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in _index_rows(holdings):
        if not _eligible(row, filters):
            continue
        octave = math.floor(math.log2(row['EXPOSURE_DURATION'] / _MS_PER_S))
        key: tuple[str, ...] = (
            row['INSTRUMENT_ID'],
            row['FILTER_NAME'],
            row['GAIN_MODE_ID'],
            row['INSTRUMENT_MODE_ID'],
            str(octave),
        )
        groups[key].append(row)
    rng = random.Random(seed)
    frames: list[FrameInput] = []
    for key in sorted(groups):
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in groups[key]:
            by_observation[row['OBSERVATION_ID']].append(row)
        # One frame per observation in turn, so that a long sequence does not fill a group.
        pools = [rng.sample(v, len(v)) for _, v in sorted(by_observation.items())]
        rng.shuffle(pools)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per_group and any(pools):
            for pool in pools:
                if pool and len(chosen) < per_group:
                    chosen.append(pool.pop())
        frames.extend(_frame_from_row(holdings, row, source='sample') for row in chosen)
    return frames


def library_frames(holdings: Path, library: Path) -> list[FrameInput]:
    """Return the image library's Cassini frames of the star-bearing classes.

    Each frame carries the library's verified offset in ``extra['library_offset']``.

    Parameters:
        holdings: The PDS3 holdings root.
        library: The library's ``images`` directory.

    Returns:
        The frames, in index order.
    """
    wanted: dict[str, list[float]] = {}
    for scene_class in LIBRARY_STAR_CLASSES:
        for sidecar in sorted((library / scene_class).glob('*.yaml')):
            doc = YAML(typ='safe').load(sidecar.read_text())
            if doc.get('mission') != 'COISS':
                continue
            truth = doc['ground_truth']
            stem = str(doc['image_id']).removesuffix('_CALIB')
            wanted[stem] = [float(truth['offset_dv_px']), float(truth['offset_du_px'])]
    frames: list[FrameInput] = []
    for row in _index_rows(holdings):
        stem = Path(row['FILE_NAME']).stem
        if stem in wanted:
            frame = _frame_from_row(holdings, row, source='library')
            frame.extra['library_offset'] = wanted.pop(stem)
            frames.append(frame)
    return frames


def calibration_terms(label_text: str) -> tuple[float, float]:
    """Return the gain and the shutter offset CISSCAL wrote into a calibrated label.

    Parameters:
        label_text: The calibrated product's PDS3 label.

    Returns:
        ``(gain in e-/DN, exposure offset in seconds)``.

    Raises:
        ValueError: If the label states either term nowhere.
    """
    gain = _GAIN_RE.search(label_text)
    offset = _OFFSET_RE.search(label_text)
    if gain is None or offset is None:
        raise ValueError('the calibrated label states no GAIN_CORRECTION or EXPOSURE_OFFSET')
    return float(gain.group(1)), float(offset.group(1)) / _MS_PER_S


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one Cassini ISS frame (the pool worker).

    Parameters:
        frame: The frame; ``path`` is the calibrated product and ``extra['raw_path']``
            the raw one.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsCassiniISS.from_file(frame.path, config=config)
    calib = np.asarray(obs.data, dtype=np.float64)
    raw = np.asarray(VicarImage(frame.extra['raw_path'], strict=False).data_2d, np.float64)
    binning = NATIVE_SIZE_PX // calib.shape[0]
    params = PARAMS[(frame.camera, binning)]
    gain, offset = calibration_terms(Path(frame.path).with_suffix('.LBL').read_text())
    record = measure_frame(obs, frame, params, config=config, phot_image=raw)
    radii = (params.aperture_px, params.big_aperture_px, 3.0 * params.aperture_px)
    for star in record.get('stars', []):
        for name, image in (('raw', raw), ('cal', calib)):
            for label, radius in zip(('r', '2r', '3r'), radii, strict=True):
                result = aperture_photometry(
                    image,
                    star['v'],
                    star['u'],
                    radius=radius,
                    sky_inner=params.sky_inner,
                    sky_outer=params.sky_outer,
                    clipped_sky=name == 'raw',
                )
                star[f'{name}_{label}'] = None if result is None else result.flux
                if result is not None and label == 'r':
                    star[f'{name}_sky'] = result.sky
                    star[f'{name}_sky_sigma'] = result.sky_sigma
                    star[f'{name}_npix'] = result.n_pixels
                if result is not None and label == '3r':
                    star[f'{name}_max'] = result.max_value
    record.update(
        binning=binning,
        gain_e_per_dn=gain,
        exposure_offset_s=offset,
        obs_texp=float(obs.texp),
        sun_distance_au=float(obs.sun_body_distance('SATURN')) / _KM_PER_AU,
        radii=list(radii),
    )
    return record


@dataclass
class CameraFit:
    """One camera's measured zero point.

    Attributes:
        camera: ``'NAC'`` or ``'WAC'``.
        raw: The fit of the raw-product electron rates, the zero point.
        calibrated: The fit of the calibrated rates, the check.
        aperture_correction: Raw ``2r``-to-``r`` light ratio by readout mode.
        aperture_correction_3r: Raw ``3r``-to-``2r`` light ratio by readout mode.
        excluded: Stars each selection step removed, in order.
    """

    camera: str
    raw: RateFit | None
    calibrated: RateFit | None
    aperture_correction: dict[str, float]
    aperture_correction_3r: dict[str, float]
    excluded: dict[str, int]


@dataclass
class StarRows:
    """A camera's fitting stars and what the selection measured on the way.

    Attributes:
        rows: One row per fitting star: the raw rate (``rate``), the calibrated rate
            (``rate_cal``), and the star's and frame's fields the checks group on.
        aperture_correction: Raw ``2r``-to-``r`` light ratio by readout mode.
        aperture_correction_cal: Calibrated ``2r``-to-``r`` light ratio by readout mode.
        aperture_correction_3r: Raw ``3r``-to-``2r`` light ratio by readout mode.
        excluded: Stars each selection step removed, in order.
    """

    rows: list[dict[str, Any]]
    aperture_correction: dict[str, float]
    aperture_correction_cal: dict[str, float]
    aperture_correction_3r: dict[str, float]
    excluded: dict[str, int]


def _saturation_dn(camera: str, gain: float, binning: int, sky: float) -> float:
    """Return the raw DN above the sky at which a pixel saturates.

    Parameters:
        camera: ``'NAC'`` or ``'WAC'``.
        gain: The gain (e-/DN).
        binning: The summation factor.
        sky: The local sky (bias included), in raw DN.

    Returns:
        The smaller of the full well over the gain and the ADC's headroom above the sky.
    """
    well = instrument_entry(SIM_INSTRUMENT[camera]).detector.full_well_e
    assert well is not None
    return min(float(well) * binning**2 / gain, ADC_MAX_DN - sky)


def _clear_throughput(camera: str) -> float:
    """Return the simulator's clear-filter throughput (e-/s per native pixel at I/F 1).

    Parameters:
        camera: ``'NAC'`` or ``'WAC'``.

    Returns:
        The throughput.
    """
    entry = instrument_entry(SIM_INSTRUMENT[camera])
    assert entry.filters is not None
    return float(entry.filters.throughput_e_per_s_per_if[CLEAR])


def _ratio(stars: list[dict[str, Any]], top: str, bottom: str) -> float:
    """Return the median ratio of two aperture sums over bright clean stars.

    Parameters:
        stars: Star records with their frame's fields (``_snr2r`` the flux SNR within 2r).
        top: The numerator's key.
        bottom: The denominator's key.

    Returns:
        The median, or NaN when no star qualifies.
    """
    values = [
        s[top] / s[bottom]
        for s in stars
        if s.get(top) is not None
        and s.get(bottom) is not None
        and s[bottom] > 0.0
        and s['_snr2r'] >= APCORR_MIN_SNR
    ]
    return float(np.median(values)) if values else math.nan


def _clean(star: dict[str, Any]) -> bool:
    """Return whether a star is isolated, unsaturated, unsmeared, and fully on data.

    Parameters:
        star: The star's record, with ``_saturated`` set.

    Returns:
        True when the star can be measured.
    """
    return (
        not star['blended']
        and star['conflict'] == ''
        and not star['smeared']
        and not star.get('no_data', True)
        and not star['_saturated']
        and star.get('raw_r') is not None
        and star.get('cal_r') is not None
    )


def _frame_stars(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a frame's star records with the frame's fields and derived values attached.

    Parameters:
        record: The frame's record.

    Returns:
        The stars, each a copy with ``_``-prefixed derived keys.
    """
    frame = record['frame']
    binning = int(record['binning'])
    gain = float(record['gain_e_per_dn'])
    center = (record['shape'][0] - 1) / 2.0
    stars = []
    for s in record.get('stars', []):
        star = dict(s)
        sky = float(star.get('raw_sky') or 0.0)
        limit = _saturation_dn(frame['camera'], gain, binning, sky)
        peak = star.get('raw_max')
        star['_saturation_dn'] = limit
        star['_saturated'] = peak is None or peak - sky >= SATURATED_FRACTION * limit
        flux_big, flux_big_sigma = star.get('flux_big'), star.get('flux_big_sigma')
        star['_snr2r'] = flux_big / flux_big_sigma if flux_big and flux_big_sigma else 0.0
        star['_radius_native'] = binning * math.hypot(star['v'] - center, star['u'] - center)
        stars.append(star)
    return stars


def _usable_frame(record: dict[str, Any], stars: list[dict[str, Any]]) -> str:
    """Return why a frame's stars cannot be fitted, or ``''`` when they can.

    Parameters:
        record: The frame's record.
        stars: Its stars, from :func:`_frame_stars`.

    Returns:
        ``'frame_fix_refused'`` for a pointing fix refused or resting on fewer than
        ``MIN_FIX_AGREE`` stars, ``'frame_fix_loose'`` for one whose agreeing stars spread
        more than ``MAX_FIX_SPREAD_PX``, ``'frame_not_12_bit'`` for a frame whose raw DN
        went through the downlink table, ``'frame_elongated'`` for a frame whose stars are
        elongated (smeared by a motion the predicted smear misses), else ``''``.
    """
    fix = record['fix']
    if not fix['accepted'] or fix['n_agree'] < MIN_FIX_AGREE:
        return 'frame_fix_refused'
    if fix['spread_px'] > MAX_FIX_SPREAD_PX:
        return 'frame_fix_loose'
    if record['frame']['extra']['conversion'] != '12BIT':
        return 'frame_not_12_bit'
    axes = [
        s['psf']['sigma_major'] / s['psf']['sigma_minor']
        for s in stars
        if s.get('psf') is not None and s['psf']['sigma_minor'] > 0.0
    ]
    if len(axes) >= MIN_ELONGATION_STARS and float(np.median(axes)) > MAX_ELONGATION:
        return 'frame_elongated'
    return ''


def star_rows(
    records: list[dict[str, Any]], camera: str, *, rate: float, filters: frozenset[str]
) -> StarRows:
    """Return a camera's fitting stars, the aperture corrections, and the exclusion counts.

    The catalog criteria come first (a UCAC4 star with APASS V and B, V inside
    ``FIT_VMAG_RANGE``), so every later count is of catalog-eligible stars: first the
    frames that cannot be fitted (:func:`_usable_frame`), then each star criterion in turn.

    Parameters:
        records: The frame records.
        camera: ``'NAC'`` or ``'WAC'``.
        rate: The provisional rate (e-/s for V = 0) the predicted signal uses.
        filters: The filter combinations fitted.

    Returns:
        The rows, the aperture corrections, and the exclusion counts.
    """
    per_frame = [
        (r, _frame_stars(r))
        for r in records
        if 'error' not in r
        and r['frame']['camera'] == camera
        and r['frame']['filter_name'] in filters
    ]
    usable = {id(r): _usable_frame(r, stars) for r, stars in per_frame}
    clean_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r, stars in per_frame:
        if not usable[id(r)]:
            clean_by_mode[r['frame']['mode']].extend(s for s in stars if _clean(s))
    apcorr = {m: _ratio(v, 'raw_2r', 'raw_r') for m, v in clean_by_mode.items()}
    apcorr_cal = {m: _ratio(v, 'cal_2r', 'cal_r') for m, v in clean_by_mode.items()}
    apcorr_3r = {m: _ratio(v, 'raw_3r', 'raw_2r') for m, v in clean_by_mode.items()}
    throughput = _clear_throughput(camera)
    excluded: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for r, stars in per_frame:
        frame = r['frame']
        candidates = []
        for s in stars:
            vmag, bmag = s.get('johnson_v'), s.get('bmag')
            if s['catalog'] != 'ucac4' or vmag is None or bmag is None:
                excluded['no_apass_v'] += 1
            elif not FIT_VMAG_RANGE[0] <= vmag <= FIT_VMAG_RANGE[1]:
                excluded['outside_v_range'] += 1
            else:
                candidates.append(s)
        if usable[id(r)]:
            excluded[usable[id(r)]] += len(candidates)
            continue
        mode = frame['mode']
        if not (math.isfinite(apcorr[mode]) and math.isfinite(apcorr_cal[mode])):
            # No star of this readout mode was bright and clean enough to measure its
            # aperture correction.
            excluded['mode_without_aperture_correction'] += len(candidates)
            continue
        binning = int(r['binning'])
        gain = float(r['gain_e_per_dn'])
        t_eff = float(frame['texp']) - float(r['exposure_offset_s'])
        peak_fraction = _peak_fraction(clean_by_mode[mode])
        for s in candidates:
            reason = _star_exclusion(s, rate=rate, gain=gain, t_eff=t_eff, peak=peak_fraction)
            if reason:
                excluded[reason] += 1
                continue
            vmag = float(s['johnson_v'])
            rows.append(
                {
                    'image_id': frame['image_id'],
                    'source': frame['extra']['source'],
                    'filter': frame['filter_name'],
                    'gain': frame['gain_state'],
                    'mode': mode,
                    'texp': float(frame['texp']),
                    'texp_octave': str(math.floor(math.log2(float(frame['texp'])))),
                    'year': frame['extra']['image_time'][:4],
                    'vmag': vmag,
                    'bv': float(s['bmag']) - vmag,
                    'rate': s['raw_r'] * apcorr[mode] * gain / t_eff,
                    'rate_cal': s['cal_r'] * apcorr_cal[mode] * binning**2 * throughput,
                    'log_texp': math.log10(t_eff),
                    'field_radius': s['_radius_native'] / (NATIVE_SIZE_PX / 2.0),
                    'sun_distance_au': float(r['sun_distance_au']),
                    'smear_px': float(s['smear_px']),
                    'gain_e_per_dn': gain,
                }
            )
    return StarRows(
        rows=rows,
        aperture_correction=apcorr,
        aperture_correction_cal=apcorr_cal,
        aperture_correction_3r=apcorr_3r,
        excluded=dict(excluded),
    )


def _predicted_snr(star: dict[str, Any], *, rate: float, gain: float, t_eff: float) -> float:
    """Return a star's SNR within ``r``, predicted from its magnitude at a rate.

    Parameters:
        star: The star, from :func:`_frame_stars`.
        rate: The rate (e-/s for V = 0).
        gain: The frame's gain (e-/DN).
        t_eff: The frame's exposure less the shutter offset (seconds).

    Returns:
        The predicted electrons over their noise: the star's shot noise and the sky noise
        over the aperture's pixels.
    """
    predicted_e = rate * 10 ** (-0.4 * float(star['johnson_v'])) * t_eff
    sky_e = float(star['raw_sky_sigma']) * gain
    return predicted_e / math.sqrt(sky_e**2 * float(star['raw_npix']) + predicted_e)


def _star_exclusion(
    star: dict[str, Any], *, rate: float, gain: float, t_eff: float, peak: float
) -> str:
    """Return why a catalog-eligible star on a usable frame is not fitted, or ``''``.

    Parameters:
        star: The star, from :func:`_frame_stars`.
        rate: The provisional rate (e-/s for V = 0).
        gain: The frame's gain (e-/DN).
        t_eff: The frame's exposure less the shutter offset (seconds).
        peak: The ratio of a star's peak to its light within ``2r`` for the camera and mode.

    Returns:
        The first criterion the star fails, or ``''``.
    """
    if star['conflict'] != '':
        return 'behind_body_or_ring'
    if star['blended']:
        return 'blended'
    if star.get('no_data', True) or star.get('raw_r') is None or star.get('cal_r') is None:
        return 'missing_data'
    if star['smeared']:
        return 'smeared'
    if _predicted_snr(star, rate=rate, gain=gain, t_eff=t_eff) < MIN_PREDICTED_SNR:
        return 'predicted_snr_low'
    predicted_e = rate * 10 ** (-0.4 * float(star['johnson_v'])) * t_eff
    if peak * predicted_e / gain >= MAX_PEAK_FRACTION * star['_saturation_dn']:
        return 'predicted_near_saturation'
    if star['_saturated']:
        return 'measured_saturated'
    if star['raw_r'] <= 0.0 or star['cal_r'] <= 0.0:
        return 'nonpositive_flux'
    return ''


def _peak_fraction(stars: list[dict[str, Any]]) -> float:
    """Return the median ratio of a bright clean star's peak, less the sky, to its light.

    Parameters:
        stars: Clean stars of one camera and mode.

    Returns:
        The median ratio of the brightest pixel within ``3r`` to the light within ``2r``,
        or 0.3 when no star qualifies.
    """
    values = [
        (s['raw_max'] - s['raw_sky']) / s['raw_2r']
        for s in stars
        if s['_snr2r'] >= APCORR_MIN_SNR and (s.get('raw_2r') or 0.0) > 0.0
    ]
    return float(np.median(values)) if values else 0.3


_GROUP_KEYS = ('gain', 'mode', 'texp_octave', 'year', 'source', 'filter')
_COVARIATES = ('vmag', 'field_radius', 'log_texp', 'sun_distance_au', 'smear_px')


def fit_camera(
    records: list[dict[str, Any]], camera: str, *, filters: frozenset[str]
) -> tuple[CameraFit, list[dict[str, Any]]]:
    """Fit one camera's zero point, iterating the selection on the fitted rate.

    Parameters:
        records: The frame records.
        camera: ``'NAC'`` or ``'WAC'``.
        filters: The filter combinations fitted.

    Returns:
        The camera's fit and its final fitting rows.
    """
    rate = PROVISIONAL_RATE[camera]
    selected = star_rows(records, camera, rate=rate, filters=filters)
    raw = _fit_rows(selected.rows, 'rate')
    for _ in range(SELECTION_PASSES - 1):
        if raw is None:
            break
        selected = star_rows(records, camera, rate=raw.rate_vmag0, filters=filters)
        raw = _fit_rows(selected.rows, 'rate')
    fit = CameraFit(
        camera=camera,
        raw=raw,
        calibrated=_fit_rows(selected.rows, 'rate_cal'),
        aperture_correction=selected.aperture_correction,
        aperture_correction_3r=selected.aperture_correction_3r,
        excluded=selected.excluded,
    )
    return fit, selected.rows


def _fit_rows(rows: list[dict[str, Any]], key: str) -> RateFit | None:
    """Fit the rows' rates under one key, with the checks every fit reports.

    Parameters:
        rows: Fitting rows.
        key: The row key of the rate fitted.

    Returns:
        The fit, or None with too few rows or stars of one color.
    """
    return fit_star_rates(
        [{**row, 'rate': row[key]} for row in rows if key in row],
        reference_color=REFERENCE_COLOR,
        group_keys=_GROUP_KEYS,
        covariates=_COVARIATES,
    )


def _library_check(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the library frames' independent fixes beside their verified offsets.

    Parameters:
        records: The frame records.

    Returns:
        One entry per library frame: its fix and the library's offset.
    """
    out = []
    for r in records:
        frame = r['frame']
        if 'error' in r or 'library_offset' not in frame['extra']:
            continue
        fix = r['fix']
        dv, du = frame['extra']['library_offset']
        out.append(
            {
                'image_id': frame['image_id'],
                'accepted': fix['accepted'],
                'n_agree': fix['n_agree'],
                'fix': [round(fix['offset_v'], 2), round(fix['offset_u'], 2)],
                'library': [dv, du],
                'difference_px': round(math.hypot(fix['offset_v'] - dv, fix['offset_u'] - du), 2),
            }
        )
    return out


def _load(path: Path) -> list[dict[str, Any]]:
    """Return the last record of each frame in a JSON-lines file.

    Parameters:
        path: The records.

    Returns:
        The records.
    """
    latest: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                latest[record['frame']['image_id']] = record
    return list(latest.values())


def _main_measure(args: argparse.Namespace) -> None:
    """Run the ``measure`` subcommand.

    Parameters:
        args: The parsed arguments.
    """
    holdings = Path(os.environ['PDS3_HOLDINGS_DIR'])
    frames = cassini_frames(holdings, filters=frozenset(args.filters), per_group=args.per_group)
    if args.library:
        library = Path(__file__).resolve().parents[2] / 'tests/integration/image_library/images'
        chosen = library_frames(holdings, library)
        # A library frame the sample also drew is measured once, as a library frame.
        ids = {f.image_id for f in chosen}
        frames = [f for f in frames if f.image_id not in ids] + chosen
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    run_frames(frames, measure, args.out, workers=args.workers)


def _main_fit(args: argparse.Namespace) -> None:
    """Run the ``fit`` subcommand: print each camera's fit as JSON.

    Parameters:
        args: The parsed arguments.
    """
    records = _load(args.records)
    summary: dict[str, Any] = {'library': _library_check(records)}
    all_rows: list[dict[str, Any]] = []
    for camera in ('NAC', 'WAC'):
        fit, rows = fit_camera(records, camera, filters=frozenset(args.filters))
        summary[camera] = asdict(fit)
        all_rows.extend({'camera': camera, **row} for row in rows)
    print(json.dumps(summary, indent=1))
    if args.stars_csv and all_rows:
        with args.stars_csv.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)


def main() -> None:
    """Measure Cassini ISS star fields, or fit the zero points from the measurements."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure', help='Measure frames into a JSON-lines file (resumes)')
    m.add_argument('out', type=Path)
    m.add_argument('--per-group', type=int, default=6)
    m.add_argument('--filters', nargs='*', default=[CLEAR])
    m.add_argument('--library', action='store_true', help="Add the library's Cassini frames")
    m.add_argument('--workers', type=int, default=4, help='At most 4 on the holdings share')
    m.add_argument('--limit', type=int, default=0)
    m.add_argument('--only', nargs='*', default=[])
    f = sub.add_parser('fit', help='Fit the zero points from measured records')
    f.add_argument('records', type=Path)
    f.add_argument('--filters', nargs='*', default=[CLEAR])
    f.add_argument('--stars-csv', type=Path, default=None)
    args = parser.parse_args()
    if args.command == 'measure':
        _main_measure(args)
    else:
        _main_fit(args)


if __name__ == '__main__':
    main()
