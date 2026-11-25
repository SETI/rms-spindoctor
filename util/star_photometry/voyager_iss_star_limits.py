"""Measure Voyager ISS star fields for the photometric star limit and the star PSF.

The navigator reads the geometrically corrected product (``*_GEOMED.IMG``, 1000 x 1000), which
the archive's pipeline has resampled and scaled to I/F, so a star's value there does not scale
with exposure.  The vidicon's own signal, which does, is the raw DN of the cleaned product
(``*_CLEANED.IMG``, 800 x 800: the raw image with the reseau marks, known blemishes and spikes
removed, in the raw geometry).  So this program finds the pointing and measures detection and
the PSF on the navigator's image, and measures each star's DN on the cleaned image at the same
point, carried across by the archive's own tiepoints (``*_GEOMA.TAB``: GEOMED line and sample
to CALIB line and sample, the CALIB and CLEANED geometries being the same).

Frames come from the cumulative PDS3 index of each Voyager ISS volume set: GEOMED products
inside the epochs the local SPICE kernels cover, whose target is not a dark, calibration-lamp
or plaque frame, and whose inventory lists no planet (so the field holds stars and at most
moons), exposed at least half a second (NAC) or a tenth of a second (WAC).  Every camera,
filter and exposure range present is sampled: within each group of encounter, camera, filter
and exposure octave, up to ``--per-group`` frames evenly spaced in the index.  The vidicon's
gain state comes from ``GAIN_MODE`` (every frame in these volumes says ``LOW``) and its readout
from ``SCAN_MODE``.

Usage::

    python -m util.star_photometry.voyager_iss_star_limits OUT.jsonl [--per-group 5]
        [--workers 4] [--limit N] [--only ID ...]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.interpolate
from numpy.typing import NDArray
from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)
from vicar import VicarImage

from spindoctor.obs import ObsVoyagerISS

__all__ = ['PARAMS', 'geoma_map', 'main', 'measure', 'voyager_frames']

FloatArray = NDArray[np.float64]

#: The volume sets of the four Voyager ISS planetary encounters.
VOLUME_SETS = ('VGISS_5xxx', 'VGISS_6xxx', 'VGISS_7xxx', 'VGISS_8xxx')

#: Observation dates (inclusive ISO dates) the local SPICE kernels cover, from
#: ``util/cohort_curation/spice_coverage.json``.
SPICE_WINDOWS = (
    ('1979-01-31', '1979-04-09'),
    ('1979-06-04', '1979-08-29'),
    ('1980-08-06', '1980-11-19'),
    ('1981-06-07', '1981-09-21'),
    ('1985-08-21', '1986-02-13'),
    ('1988-11-12', '1989-10-01'),
)

#: Targets that hold no star field.
EXCLUDED_TARGETS = frozenset({'DARK', 'CAL LAMPS', 'PLAQUE'})

#: Inventory entries that mean a planet is in the field.
PLANETS = frozenset({'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE'})

#: Shortest exposure sampled per camera (seconds): below it the camera records no star in the
#: field at the depth the navigator expects (a V 7-8 star in the NAC's 0.4 degree field), so
#: the frame cannot be fixed from stars.
MIN_EXPOSURE_S = {'NAC': 0.5, 'WAC': 0.1}

#: The 8-bit DN at which the vidicon's ADC saturates.
SATURATION_DN = 255.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: Columns of ``VGISS_n999_index.tab``.
_COL_VOLUME, _COL_PRODUCT, _COL_TARGET, _COL_TIME, _COL_INSTRUMENT = 0, 2, 6, 9, 11
_COL_SCAN, _COL_SHUTTER, _COL_GAIN, _COL_FILTER, _COL_EXPOSURE = 12, 13, 14, 16, 18

#: GEOMED product suffix, and the cleaned and tiepoint products beside it.
_GEOMED, _CLEANED, _GEOMA = '_GEOMED.IMG', '_CLEANED.IMG', '_GEOMA.TAB'

PARAMS = MeasureParams(
    catalog_depth_mag=4.0,
    highpass_size=21,
    smooth_sigma=1.0,
    peak_threshold_snr=5.0,
    edge_px=16,
    max_peaks=300,
    max_vote_stars=60,
    search_px=400.0,
    vote_bin_px=1.5,
    agree_px=1.5,
    min_agree=3,
    max_chance_probability=1e-3,
    detect_search_px=2.0,
    noise_inner=6.0,
    noise_outer=12.0,
    box_half=2,
    aperture_px=3.0,
    big_aperture_px=6.0,
    sky_inner=8.0,
    sky_outer=13.0,
    blend_px=10.0,
    blend_dmag=2.5,
    saturation_dn=SATURATION_DN,
    max_smear_px=1.0,
    psf_min_snr=20.0,
    psf_half=6,
    ee_r_max=8.0,
    ee_n_bins=16,
    highpass_subsample=2,
    missing_value=0.0,
)


def _in_spice_window(time: str) -> bool:
    """Return whether an index ``IMAGE_TIME`` lies in a SPICE-covered epoch.

    Parameters:
        time: The ISO time.

    Returns:
        True inside a window.
    """
    day = time[:10]
    return any(start <= day <= stop for start, stop in SPICE_WINDOWS)


def _exposure_octave(texp: float) -> int:
    """Return the base-2 octave of an exposure, the sampling group's exposure key.

    Parameters:
        texp: Exposure in seconds.

    Returns:
        ``floor(log2(texp))``, or -99 for a non-positive exposure.
    """
    return math.floor(math.log2(texp)) if texp > 0 else -99


def voyager_frames(holdings: Path, per_group: int) -> list[FrameInput]:
    """Return a stratified sample of Voyager ISS star-field frames from the PDS3 indexes.

    Parameters:
        holdings: The PDS3 holdings root (``$PDS3_HOLDINGS_DIR``).
        per_group: The most frames kept per encounter, camera, filter and exposure octave.

    Returns:
        The sampled frames.
    """
    groups: dict[tuple[str, str, str, int], list[FrameInput]] = defaultdict(list)
    for volume_set in VOLUME_SETS:
        stem = volume_set.replace('xxx', '999')
        meta = holdings / 'metadata' / volume_set / stem
        bodies: dict[str, list[str]] = {}
        with (meta / f'{stem}_inventory.csv').open() as fh:
            for row in csv.reader(fh):
                bodies[Path(row[1].strip()).name.split('_')[0]] = [b.strip() for b in row[3:]]
        with (meta / f'{stem}_index.tab').open() as fh:
            for row in csv.reader(fh):
                product = row[_COL_PRODUCT].strip()
                if not product.endswith(_GEOMED):
                    continue
                if row[_COL_TARGET].strip() in EXCLUDED_TARGETS:
                    continue
                if not _in_spice_window(row[_COL_TIME]):
                    continue
                image_id = product.split('_')[0]
                if PLANETS & set(bodies.get(image_id, [])):
                    continue
                texp = float(row[_COL_EXPOSURE])
                camera = 'NAC' if row[_COL_INSTRUMENT].strip().startswith('NARROW') else 'WAC'
                if texp < MIN_EXPOSURE_S[camera]:
                    continue
                volume = row[_COL_VOLUME].strip()
                path = (
                    holdings
                    / 'volumes'
                    / volume_set
                    / volume
                    / 'DATA'
                    / f'{image_id[:6]}XX'
                    / product
                )
                frame = FrameInput(
                    image_id=image_id,
                    path=str(path),
                    camera=camera,
                    filter_name=row[_COL_FILTER].strip(),
                    gain_state=row[_COL_GAIN].strip(),
                    mode=row[_COL_SCAN].strip(),
                    texp=texp,
                    extra={
                        'volume': volume,
                        'target': row[_COL_TARGET].strip(),
                        'shutter': row[_COL_SHUTTER].strip(),
                        'moons': bodies.get(image_id, []),
                    },
                )
                key = (volume_set, camera, frame.filter_name, _exposure_octave(texp))
                groups[key].append(frame)
    sample: list[FrameInput] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) <= per_group:
            sample += members
            continue
        picks = np.linspace(0, len(members) - 1, per_group).round().astype(int)
        sample += [members[i] for i in sorted(set(picks.tolist()))]
    return sample


def geoma_map(table: Path) -> scipy.interpolate.LinearNDInterpolator:
    """Return the GEOMED-to-CALIB position map of a Voyager image from its tiepoint table.

    The table's rows are ``index, output line, output sample, input line, input sample`` in
    the one-based coordinates of VICAR (the first pixel's center is line 1, sample 1); output
    is the GEOMED image and input the CALIB image.  The map is piecewise linear over the
    tiepoints' triangulation.

    Parameters:
        table: The ``*_GEOMA.TAB`` file.

    Returns:
        An interpolator taking ``(line, sample)`` GEOMED array-index positions to CALIB
        array-index positions (NaN outside the tiepoint mesh).
    """
    rows = np.loadtxt(table, delimiter=',')
    points = np.unique(rows[:, 1:5], axis=0)
    return scipy.interpolate.LinearNDInterpolator(points[:, 0:2] - 1.0, points[:, 2:4] - 1.0)


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one Voyager ISS frame (the pool worker).

    Parameters:
        frame: The frame.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsVoyagerISS.from_file(frame.path, config=config)
    cleaned = VicarImage.from_file(frame.path.replace(_GEOMED, _CLEANED))
    assert isinstance(cleaned, VicarImage)  # A tuple only when extraneous='include'.
    mapping = geoma_map(Path(frame.path.replace(_GEOMED, _GEOMA)))

    def to_cleaned(v: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        out = mapping(np.column_stack([v, u]))
        return out[:, 0], out[:, 1]

    record = measure_frame(
        obs,
        frame,
        PARAMS,
        config=config,
        phot_image=np.asarray(cleaned.data_2d, dtype=np.float64),
        phot_map=to_cleaned,
    )
    record['obs_texp'] = float(obs.texp)
    record['obs_detector'] = str(obs.detector)
    record['spacecraft'] = obs.spacecraft_digit
    return record


def main() -> None:
    """Measure a stratified sample of Voyager ISS star fields into a JSON-lines file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='JSON-lines output (appended; resumes)')
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Measure only the first N frames')
    parser.add_argument('--only', nargs='*', default=[], help='Measure only these image IDs')
    parser.add_argument('--list', action='store_true', help='Print the sample and stop')
    args = parser.parse_args()
    frames = voyager_frames(Path(os.environ['PDS3_HOLDINGS_DIR']), args.per_group)
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    if args.list:
        for f in frames:
            print(f.image_id, f.camera, f.filter_name, f.texp, f.mode, f.extra['target'])
        return
    run_frames(frames, measure, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
