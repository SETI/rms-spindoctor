"""Measure Cassini ISS star fields for the photometric star limit and the star PSF.

The navigator reads the calibrated product (``*_CALIB.IMG``), which the archive's pipeline
has scaled to I/F, so a star's value there depends on the filter's solar flux as well as on
the exposure.  The camera's own signal is the raw DN of the raw product (``*_1.IMG``), and
the two share one geometry.  So this program finds the pointing and measures detection and
the PSF on the navigator's image, and measures each star's DN on the raw image at the same
point.  A raw image whose data were encoded through the camera's 12-to-8-bit lookup table
(``DATA_CONVERSION_TYPE`` ``TABLE``) is decoded with the inverse table the calibration
volume publishes (``COISS_0011/calib/lut/lut.tab``, from Appendix F1 of the ISS ground
calibration report); a ``12BIT`` image is read as it is, and an ``8LSB`` image, which kept
only the low 8 bits, is not sampled.

Frames come from the cumulative PDS3 index of the tour volumes (``COISS_2999``): images
inside the epochs the local SPICE kernels cover, whose index marks every ring as out of
the field (``RINGS_FLAG``) and whose inventory does not list Saturn or Jupiter (so the
field holds stars and at most moons), and not aimed at Titan, taken with the shutter
enabled and the calibration lamp not on, lossless or no compression, and no missing
lines, exposed at least a tenth of a second.  Through the clear pair (``CL1+CL2``),
which measures the rate, the gain states, the color term, and the limit of each readout
mode and gain state, up to ``--per-group`` frames are kept for each group of camera, gain
state, mode, and exposure octave, evenly spaced in the index.  Through every other filter
pair, whose frames check the calibration's filter offsets (see
:func:`systrans_filter_offsets`), up to ``--per-group-filter`` frames are kept for each
group of camera, filter pair, and exposure octave.  The gain state is the index's
``GAIN_MODE_ID`` (the nominal electrons per DN: 215, 95, 29, or 12), the readout mode
its ``INSTRUMENT_MODE_ID`` (``FULL``, ``SUM2``, or ``SUM4``), and the filter the pair
the index's ``FILTER_NAME`` gives, joined by ``+``.

Usage::

    python -m util.star_photometry.cassini_iss_star_limits OUT.jsonl [--per-group 6]
        [--per-group-filter 1] [--workers 4] [--limit N] [--only ID ...]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)
from vicar import VicarImage

from spindoctor.obs import ObsCassiniISS

__all__ = [
    'PARAMS',
    'cassini_frames',
    'decode_raw',
    'main',
    'measure',
    'read_lut',
    'saturation_dn',
    'systrans_filter_offsets',
]

FloatArray = NDArray[np.float64]

#: The volume set of the tour images, and its cumulative index.
VOLUME_SET = 'COISS_2xxx'
CUMULATIVE = 'COISS_2999'

#: Observation dates the local SPICE kernels cover (inclusive, as the index's day-of-year
#: ``IMAGE_TIME`` spells the date): 2004-01-01 to 2017-09-15, from
#: ``util/cohort_curation/spice_coverage.json``.
SPICE_WINDOW = ('2004-001', '2017-258')

#: Inventory entries that mean a planet is in the field.
PLANETS = frozenset({'SATURN', 'JUPITER'})

#: Targets whose frames rarely show a star field: Titan's haze fills much of the field of a
#: frame aimed at it.
EXCLUDED_TARGETS = frozenset({'TITAN'})

#: The reference filter pair: both wheels clear.
REFERENCE_FILTER = 'CL1+CL2'

#: Shortest exposure sampled (seconds).
MIN_EXPOSURE_S = 0.1

#: The raw DN at or above which a pixel is treated as saturated when the frame's full well
#: holds more: the 12-bit ADC's 4095, less a margin for the onset of nonlinearity.  A frame
#: whose full well (the index's VALID_MAXIMUM, which depends on the camera, gain state, and
#: readout mode) holds less saturates there instead (see :func:`saturation_dn`).
SATURATION_DN = 4000.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: The inverse lookup table of the calibration volume.
LUT_PATH = Path('volumes/COISS_0xxx/COISS_0011/calib/lut/lut.tab')

#: The index columns read, by name; the label gives each one's bytes.
_COLUMNS = (
    'FILE_SPECIFICATION_NAME',
    'VOLUME_ID',
    'DATA_CONVERSION_TYPE',
    'EXPOSURE_DURATION',
    'FILTER_NAME',
    'GAIN_MODE_ID',
    'IMAGE_TIME',
    'INSTRUMENT_ID',
    'INSTRUMENT_MODE_ID',
    'INST_CMPRS_TYPE',
    'TARGET_NAME',
    'SHUTTER_STATE_ID',
    'MISSING_LINES',
    'CALIBRATION_LAMP_STATE_FLAG',
    'RINGS_FLAG',
    'OBSERVATION_ID',
    'VALID_MAXIMUM',
)

_COMMON: dict[str, Any] = {
    'catalog_depth_mag': 4.0,
    'peak_threshold_snr': 5.0,
    'max_peaks': 300,
    'max_vote_stars': 60,
    'vote_bin_px': 1.5,
    'agree_px': 1.0,
    'min_agree': 3,
    'max_chance_probability': 1e-3,
    'box_half': 2,
    'blend_dmag': 2.5,
    'saturation_dn': SATURATION_DN,
    'max_smear_px': 1.0,
    'psf_min_snr': 20.0,
    'missing_value': 0.0,
    'max_peaks_per_line': 20,
}

#: Pointing search half-width per camera on a full frame (pixels): the NAC's pointing error
#: reaches about 140 px and the WAC's about 14 px (the navigator's offset windows); a summed
#: frame's is the full frame's divided by the summation.
_SEARCH_FULL_PX = {'NAC': 200.0, 'WAC': 30.0}

#: The summation of each readout mode.
_SUMMATION = {'FULL': 1, 'SUM2': 2, 'SUM4': 4}


def _params(camera: str, mode: str) -> MeasureParams:
    """Return the measurement parameters of a camera and readout mode.

    Parameters:
        camera: ``'NAC'`` or ``'WAC'``.
        mode: ``'FULL'``, ``'SUM2'``, or ``'SUM4'``.

    Returns:
        The parameters.
    """
    summation = _SUMMATION[mode]
    search = max(_SEARCH_FULL_PX[camera] / summation, 10.0)
    if mode == 'FULL':
        geometry: dict[str, Any] = {
            'highpass_size': 21,
            'smooth_sigma': 0.8,
            'edge_px': 16,
            'detect_search_px': 2.0,
            'noise_inner': 8.0,
            'noise_outer': 14.0,
            'aperture_px': 3.0,
            'big_aperture_px': 6.0,
            'sky_inner': 8.0,
            'sky_outer': 13.0,
            'blend_px': 8.0,
            'psf_half': 6,
            'ee_r_max': 6.0,
            'ee_n_bins': 12,
        }
    else:
        geometry = {
            'highpass_size': 11,
            'smooth_sigma': 0.7,
            'edge_px': 8,
            'detect_search_px': 1.5,
            'noise_inner': 5.0,
            'noise_outer': 9.0,
            'aperture_px': 2.5,
            'big_aperture_px': 4.0,
            'sky_inner': 6.0,
            'sky_outer': 9.0,
            'blend_px': 5.0,
            'psf_half': 4,
            'ee_r_max': 4.0,
            'ee_n_bins': 8,
        }
    return MeasureParams(**_COMMON, **geometry, search_px=search)


#: Measurement parameters per camera and readout mode.
PARAMS: dict[tuple[str, str], MeasureParams] = {
    (camera, mode): _params(camera, mode) for camera in _SEARCH_FULL_PX for mode in _SUMMATION
}


def _label_columns(label: str) -> dict[str, tuple[int, int, int]]:
    """Return each index column's start byte, byte count, and item count.

    Parameters:
        label: The index label's text.

    Returns:
        The zero-based start, the total bytes, and the number of items (1 for a column of
        one value), keyed by column name.

    Raises:
        ValueError: If a column object does not give its name, start byte, and bytes.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for block in re.findall(r'OBJECT\s*=\s*COLUMN(.*?)END_OBJECT', label, flags=re.S):
        name = re.search(r'\bNAME\s*=\s*(\S+)', block)
        start = re.search(r'\bSTART_BYTE\s*=\s*(\d+)', block)
        size = re.search(r'\bBYTES\s*=\s*(\d+)', block)
        if name is None or start is None or size is None:
            raise ValueError(f'index label column without a name, start byte, or bytes: {block}')
        items = re.search(r'\bITEMS\s*=\s*(\d+)', block)
        out[name.group(1)] = (
            int(start.group(1)) - 1,
            int(size.group(1)),
            int(items.group(1)) if items else 1,
        )
    return out


def _field(line: str, spec: tuple[int, int, int]) -> str | list[str]:
    """Return one column of an index line, as text, or a list of texts for an array.

    Parameters:
        line: The index line.
        spec: The column's start, bytes, and items (see :func:`_label_columns`).

    Returns:
        The stripped value, unquoted, or one per item.
    """
    start, size, n_items = spec
    text = line[start : start + size]
    if n_items == 1:
        return text.strip().strip('"').strip()
    values = next(csv.reader([text]))
    return [v.strip().strip('"').strip() for v in values]


def _exposure_octave(texp: float) -> int:
    """Return the base-2 octave of an exposure, the sampling group's exposure key.

    Parameters:
        texp: Exposure in seconds.

    Returns:
        ``floor(log2(texp))``.
    """
    return math.floor(math.log2(texp))


def cassini_frames(holdings: Path, per_group: int, per_group_filter: int) -> list[FrameInput]:
    """Return a stratified sample of Cassini ISS star-field frames from the PDS3 index.

    Parameters:
        holdings: The PDS3 holdings root (``$PDS3_HOLDINGS_DIR``).
        per_group: The most frames through the reference filter pair kept per camera, gain
            state, readout mode, and exposure octave.
        per_group_filter: The most frames through any other filter pair kept per camera,
            filter pair, and exposure octave.

    Returns:
        The sampled frames.
    """
    meta = holdings / 'metadata' / VOLUME_SET / CUMULATIVE
    bodies: dict[str, list[str]] = {}
    with (meta / f'{CUMULATIVE}_inventory.csv').open() as fh:
        for entry in csv.reader(fh):
            bodies[Path(entry[1].strip()).stem] = [b.strip() for b in entry[2:]]
    specs = _label_columns((meta / f'{CUMULATIVE}_index.lbl').read_text())
    columns = {name: specs[name] for name in _COLUMNS}
    groups: dict[tuple[str, str, str, str, int], list[FrameInput]] = defaultdict(list)
    with (meta / f'{CUMULATIVE}_index.tab').open() as fh:
        for line in fh:
            row = {name: _field(line, spec) for name, spec in columns.items()}
            if not SPICE_WINDOW[0] <= str(row['IMAGE_TIME'])[:8] <= SPICE_WINDOW[1]:
                continue
            if row['RINGS_FLAG'] != 'NO' or row['DATA_CONVERSION_TYPE'] == '8LSB':
                continue
            if row['SHUTTER_STATE_ID'] != 'ENABLED' or row['INST_CMPRS_TYPE'] == 'LOSSY':
                continue
            if row['CALIBRATION_LAMP_STATE_FLAG'] == 'ON':
                continue
            if row['MISSING_LINES'] not in ('0', ''):
                continue
            texp = float(str(row['EXPOSURE_DURATION'])) / 1000.0
            if texp < MIN_EXPOSURE_S:
                continue
            spec = str(row['FILE_SPECIFICATION_NAME'])
            image_id = Path(spec).stem
            if PLANETS & set(bodies.get(image_id, [])):
                continue
            if row['TARGET_NAME'] in EXCLUDED_TARGETS:
                continue
            camera = 'NAC' if row['INSTRUMENT_ID'] == 'ISSNA' else 'WAC'
            filters = row['FILTER_NAME']
            assert isinstance(filters, list)
            filter_name = '+'.join(filters)
            gain_state = str(row['GAIN_MODE_ID']).split()[0]
            mode = str(row['INSTRUMENT_MODE_ID'])
            volume = str(row['VOLUME_ID'])
            raw = holdings / 'volumes' / VOLUME_SET / volume / spec
            calib = (
                holdings / 'calibrated' / VOLUME_SET / volume / spec.replace('.IMG', '_CALIB.IMG')
            )
            frame = FrameInput(
                image_id=image_id,
                path=str(calib),
                camera=camera,
                filter_name=filter_name,
                gain_state=gain_state,
                mode=mode,
                texp=texp,
                extra={
                    'volume': volume,
                    'raw_path': str(raw),
                    'conversion': row['DATA_CONVERSION_TYPE'],
                    'target': row['TARGET_NAME'],
                    'observation_id': row['OBSERVATION_ID'],
                    'valid_maximum': [float(v) for v in row['VALID_MAXIMUM']],
                    'moons': bodies.get(image_id, [])[1:],
                },
            )
            if filter_name == REFERENCE_FILTER:
                key = (camera, gain_state, mode, filter_name, _exposure_octave(texp))
            else:
                key = (camera, '', '', filter_name, _exposure_octave(texp))
            groups[key].append(frame)
    sample: list[FrameInput] = []
    for key in sorted(groups):
        members = groups[key]
        n_keep = per_group if key[3] == REFERENCE_FILTER else per_group_filter
        if len(members) <= n_keep:
            sample += members
            continue
        picks = np.linspace(0, len(members) - 1, n_keep).round().astype(int)
        sample += [members[i] for i in sorted(set(picks.tolist()))]
    return sample


#: The filters of each camera's two wheels, as the labels spell them.
FILTER_WHEELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    'NAC': (
        ('CL1', 'RED', 'BL1', 'UV2', 'UV1', 'IRP0', 'P120', 'P60', 'P0', 'HAL', 'IR4', 'IR2'),
        ('CL2', 'GRN', 'UV3', 'BL2', 'MT2', 'CB2', 'MT3', 'CB3', 'MT1', 'CB1', 'IR3', 'IR1'),
    ),
    'WAC': (
        ('CL1', 'IR3', 'IR4', 'IR5', 'CB3', 'MT3', 'CB2', 'MT2', 'IR2'),
        ('CL2', 'RED', 'GRN', 'BL1', 'VIO', 'HAL', 'IRP90', 'IRP0', 'IR1'),
    ),
}

#: The calibration volume's directory, which holds the efficiency and correction tables.
CALIB_DIR = Path('volumes/COISS_0xxx/COISS_0011/calib')


def _data_rows(path: Path) -> FloatArray:
    """Return the numeric rows after a calibration table's ``\\begindata`` line.

    Parameters:
        path: The table.

    Returns:
        One row per data line, split on white space.
    """
    text = path.read_text().split('\\begindata', 1)[1]
    return np.array(
        [[float(x) for x in line.split()] for line in text.splitlines() if line.strip()]
    )


def _correction_factors(path: Path) -> dict[tuple[str, str, str], float]:
    """Return the absolute correction factor of each camera and filter pair.

    Parameters:
        path: The calibration volume's ``correctionfactors_qecorr.tab``.

    Returns:
        The factor, keyed by instrument (``ISSNA`` or ``ISSWA``) and the two filters.
    """
    text = path.read_text().split('\\begindata', 1)[1]
    out: dict[tuple[str, str, str], float] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4:
            out[(fields[0], fields[1], fields[2])] = float(fields[3])
    return out


def systrans_filter_offsets(calib_dir: Path, camera: str) -> dict[str, float]:
    """Return each filter pair's magnitude offset for a solar-type star, from the calibration.

    The calibration software (CISSCAL) predicts a pair's signal from a solar spectrum as
    ``C * sum(T * Q * F)`` over wavelength, where ``T`` is the pair's system transmission
    (quantum efficiency, optics, and both filters), ``F`` the solar photon flux (the two
    together are the third column of the pair's ``efficiency/systrans`` table), ``Q`` the
    camera's quantum-efficiency correction (``correction/*_qe_correction.tab``), and ``C``
    the pair's absolute correction factor from in-flight photometry of standard stars and
    satellites (``correction/correctionfactors_qecorr.tab``; 1 for a pair it does not list,
    and looked up with the clear filter in place of a polarizer, as CISSCAL does).  The
    offset of a pair is ``-2.5 log10`` of its signal over the clear pair's.  Starlight is
    unpolarized, so a pair holding a polarizer passes the mean of its parallel and
    perpendicular tables.

    Parameters:
        calib_dir: The calibration volume's ``calib`` directory.
        camera: ``'NAC'`` or ``'WAC'``.

    Returns:
        The offset of every pair the tables give, keyed by the pair as the labels spell it
        (``'CL1+GRN'``), the clear pair's zero included.
    """
    prefix, instrument = ('issna', 'ISSNA') if camera == 'NAC' else ('isswa', 'ISSWA')
    qe = _data_rows(calib_dir / 'correction' / f'{prefix[3:5]}c_qe_correction.tab')
    factors = _correction_factors(calib_dir / 'correction' / 'correctionfactors_qecorr.tab')
    systrans = calib_dir / 'efficiency' / 'systrans'

    def signal_of(path: Path) -> float:
        rows = _data_rows(path)
        return float(np.sum(rows[:, 2] * np.interp(rows[:, 0], qe[:, 0], qe[:, 1])))

    wheel1, wheel2 = FILTER_WHEELS[camera]
    signal: dict[str, float] = {}
    for first in wheel1:
        for second in wheel2:
            stem = f'{prefix}{first.lower()}{second.lower()}'
            parallel = systrans / f'{stem}_systrans.tab'
            if not parallel.exists():
                continue
            value = signal_of(parallel)
            perpendicular = systrans / f'{stem}_b_systrans.tab'
            if perpendicular.exists():
                value = 0.5 * (value + signal_of(perpendicular))
            lookup = (first, second)
            if 'P' in first + second:
                lookup = ('CL1', second) if camera == 'NAC' else (first, 'CL2')
            signal[f'{first}+{second}'] = value * factors.get((instrument, *lookup), 1.0)
    reference = signal[REFERENCE_FILTER]
    return {pair: -2.5 * math.log10(value / reference) for pair, value in signal.items()}


def read_lut(path: Path) -> FloatArray:
    """Return the inverse lookup table of table-encoded images: 12-bit DN per 8-bit DN.

    Parameters:
        path: The calibration volume's ``lut.tab``.

    Returns:
        The 256 12-bit values, indexed by the 8-bit value.

    Raises:
        ValueError: If the table does not hold 256 values.
    """
    text = path.read_text().split('\\begindata', 1)[1]
    values = np.array([float(v) for v in re.findall(r'\d+', text)])
    if values.size != 256:
        raise ValueError(f'{path} holds {values.size} values, not 256')
    return values


def decode_raw(data: NDArray[Any], conversion: str, lut: FloatArray) -> FloatArray:
    """Return a raw image's 12-bit DN.

    Parameters:
        data: The raw image as stored.
        conversion: The label's ``DATA_CONVERSION_TYPE``: ``TABLE`` or ``12BIT``.
        lut: The inverse lookup table (see :func:`read_lut`).

    Returns:
        The image in 12-bit DN, with a missing pixel (stored as 0) kept at 0.

    Raises:
        ValueError: If the conversion type is neither ``TABLE`` nor ``12BIT``.
    """
    values = np.asarray(data)
    if conversion == '12BIT':
        return values.astype(np.float64)
    if conversion != 'TABLE':
        raise ValueError(f'unsupported DATA_CONVERSION_TYPE {conversion!r}')
    codes = np.clip(values.astype(np.int64), 0, 255)
    decoded = lut[codes]
    decoded[codes == 0] = 0.0
    return decoded


def saturation_dn(frame: FrameInput) -> float:
    """Return the raw DN at or above which a pixel of a frame is treated as saturated.

    Parameters:
        frame: The frame, whose ``extra['valid_maximum']`` is the index's VALID_MAXIMUM: the
            ADC's ceiling (4095, or 255 for an image kept to 8 bits) and the full well in DN,
            the order the cumulative index gives them in.

    Returns:
        The lowest of the two and :data:`SATURATION_DN`.
    """
    return min(*(float(v) for v in frame.extra['valid_maximum']), SATURATION_DN)


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one Cassini ISS frame (the pool worker).

    Parameters:
        frame: The frame.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsCassiniISS.from_file(frame.path, config=config)
    raw = VicarImage.from_file(frame.extra['raw_path'])
    assert isinstance(raw, VicarImage)  # A tuple only when extraneous='include'.
    lut = read_lut(Path(os.environ['PDS3_HOLDINGS_DIR']) / LUT_PATH)
    data = raw.data_2d
    if data is None:
        raise ValueError(f'{frame.extra["raw_path"]} holds no image data')
    phot = decode_raw(data, str(frame.extra['conversion']), lut)
    if phot.shape != np.asarray(obs.data).shape:
        raise ValueError(
            f'raw image shape {phot.shape} differs from the calibrated {obs.data.shape}'
        )
    params = dataclasses.replace(
        PARAMS[(frame.camera, frame.mode)], saturation_dn=saturation_dn(frame)
    )
    record = measure_frame(obs, frame, params, config=config, phot_image=phot)
    record['obs_texp'] = float(obs.texp)
    record['obs_detector'] = str(obs.detector)
    return record


def main() -> None:
    """Measure a stratified sample of Cassini ISS star fields into a JSON-lines file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='JSON-lines output (appended; resumes)')
    parser.add_argument('--per-group', type=int, default=6)
    parser.add_argument('--per-group-filter', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Measure only the first N frames')
    parser.add_argument('--only', nargs='*', default=[], help='Measure only these image IDs')
    parser.add_argument('--list', action='store_true', help='Print the sample and stop')
    args = parser.parse_args()
    frames = cassini_frames(
        Path(os.environ['PDS3_HOLDINGS_DIR']), args.per_group, args.per_group_filter
    )
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    if args.list:
        for f in frames:
            print(
                f.image_id,
                f.camera,
                f.gain_state,
                f.mode,
                f.filter_name,
                f.texp,
                f.extra['conversion'],
                f.extra['target'],
            )
        return
    run_frames(frames, measure, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
