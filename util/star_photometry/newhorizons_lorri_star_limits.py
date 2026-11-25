"""Measure New Horizons LORRI star fields for the photometric star limit and the star PSF.

The navigator reads LORRI's calibrated science product (``*_sci.fit``: bias-subtracted and
flat-fielded, still in DN), which is linear in the exposure, so the navigator's image is also
the photometric image.

Frames come from the cumulative supplemental index of the calibrated volumes
(``NHxxLO_2999``), inside the epochs the local SPICE kernels cover (before 2016), leaving out
the targets whose disk or rings fill much of the field (Jupiter and its Galilean moons, the
Jovian rings, the Sun), Pluto and Charon after 2015-07-04 (when Pluto spans more than a few
dozen pixels), and the cover-closed noise and bias tests.  What remains is star calibrations
(M7, NGC 3532), optical-navigation and satellite-search frames in which Pluto, a small moon or
a distant Kuiper-belt object is a point among the stars, and similar fields.  Every target
group, binning mode and exposure octave present is sampled: up to ``--per-group`` frames each,
evenly spaced in the index.  LORRI has one gain state and no filter.

Usage::

    python -m util.star_photometry.newhorizons_lorri_star_limits OUT.jsonl [--per-group 8]
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
from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)

from spindoctor.obs import ObsNewHorizonsLORRI

__all__ = ['PARAMS_BY_BINNING', 'lorri_frames', 'main', 'measure']

#: The end of the epochs the local SPICE kernels cover (exclusive ISO date).
SPICE_END = '2016-01-01'

#: The start of the epochs the local SPICE kernels cover (inclusive ISO date).
SPICE_START = '2006-01-19'

#: After this date Pluto and Charon fill too much of the field for a star field.
PLUTO_RESOLVED_FROM = '2015-07-04'

#: Targets whose disk or rings fill much of the field.
EXCLUDED_TARGETS = frozenset({'JUPITER', 'IO', 'EUROPA', 'GANYMEDE', 'CALLISTO', 'J RINGS', 'SUN'})

#: Targets resolved after ``PLUTO_RESOLVED_FROM``.
PLUTO_SYSTEM_RESOLVED = frozenset({'PLUTO', 'CHARON'})

#: Observation descriptions of cover-closed or dark-only frames.
EXCLUDED_DESCRIPTION_WORDS = ('NOISE TEST', 'BIAS', 'COVER DOOR CLOSED')

#: The calibrated DN above which a pixel is treated as saturated: the 12-bit ADC's 4095 less
#: the bias (about 540 DN), less a margin for the flat field and the onset of nonlinearity.
SATURATION_DN = 3000.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: Columns of ``NHxxLO_2999_supplemental_index.tab``.
_COL_VOLUME, _COL_FILE, _COL_TARGET, _COL_START = 0, 1, 6, 9
_COL_BINNING, _COL_EXPOSURE, _COL_COMPRESSION, _COL_DESCRIPTION = 15, 16, 17, 18

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
}

#: Measurement parameters for each binning mode.
PARAMS_BY_BINNING: dict[str, MeasureParams] = {
    '1x1': MeasureParams(
        **_COMMON,
        highpass_size=21,
        smooth_sigma=1.0,
        edge_px=16,
        search_px=100.0,
        detect_search_px=2.0,
        noise_inner=8.0,
        noise_outer=14.0,
        aperture_px=4.0,
        big_aperture_px=8.0,
        sky_inner=10.0,
        sky_outer=15.0,
        blend_px=10.0,
        psf_half=7,
        ee_r_max=8.0,
        ee_n_bins=16,
    ),
    '4x4': MeasureParams(
        **_COMMON,
        highpass_size=11,
        smooth_sigma=0.7,
        edge_px=10,
        search_px=40.0,
        detect_search_px=1.5,
        noise_inner=5.0,
        noise_outer=9.0,
        aperture_px=2.5,
        big_aperture_px=4.0,
        sky_inner=6.0,
        sky_outer=9.0,
        blend_px=5.0,
        psf_half=4,
        ee_r_max=4.0,
        ee_n_bins=8,
    ),
}


def _target_group(target: str) -> str:
    """Return the sampling group of a target: the target itself, Kuiper-belt objects pooled.

    Parameters:
        target: The index's ``TARGET_NAME``.

    Returns:
        The group name.
    """
    return 'KBO' if any(ch.isdigit() for ch in target) else target


def lorri_frames(holdings: Path, per_group: int) -> list[FrameInput]:
    """Return a stratified sample of LORRI star-field frames from the supplemental index.

    Parameters:
        holdings: The PDS3 holdings root (``$PDS3_HOLDINGS_DIR``).
        per_group: The most frames kept per target group, binning mode and exposure octave.

    Returns:
        The sampled frames.
    """
    meta = holdings / 'metadata' / 'NHxxLO_xxxx' / 'NHxxLO_2999'
    groups: dict[tuple[str, str, int], list[FrameInput]] = defaultdict(list)
    with (meta / 'NHxxLO_2999_supplemental_index.tab').open() as fh:
        for row in csv.reader(fh):
            day = row[_COL_START][:10]
            target = row[_COL_TARGET].strip()
            description = row[_COL_DESCRIPTION].strip().upper()
            texp = float(row[_COL_EXPOSURE])
            if not SPICE_START <= day < SPICE_END or texp <= 0.0:
                continue
            if target in EXCLUDED_TARGETS:
                continue
            if target in PLUTO_SYSTEM_RESOLVED and day >= PLUTO_RESOLVED_FROM:
                continue
            if any(word in description for word in EXCLUDED_DESCRIPTION_WORDS):
                continue
            spec = row[_COL_FILE].strip()
            volume = row[_COL_VOLUME].strip()
            binning = row[_COL_BINNING].strip()
            path = holdings / 'volumes' / 'NHxxLO_xxxx' / volume / spec.replace('.lbl', '.fit')
            frame = FrameInput(
                image_id=Path(spec).stem,
                path=str(path),
                camera='LORRI',
                filter_name='CLEAR',
                gain_state='SINGLE',
                mode=binning,
                texp=texp,
                extra={
                    'volume': volume,
                    'target': target,
                    'compression': row[_COL_COMPRESSION].strip(),
                    'description': description,
                },
            )
            key = (_target_group(target), binning, math.floor(math.log2(texp)))
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


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one LORRI frame (the pool worker).

    Parameters:
        frame: The frame.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsNewHorizonsLORRI.from_file(frame.path, config=config)
    record = measure_frame(obs, frame, PARAMS_BY_BINNING[frame.mode], config=config)
    record['obs_texp'] = float(obs.texp)
    return record


def main() -> None:
    """Measure a stratified sample of LORRI star fields into a JSON-lines file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='JSON-lines output (appended; resumes)')
    parser.add_argument('--per-group', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Measure only the first N frames')
    parser.add_argument('--only', nargs='*', default=[], help='Measure only these image IDs')
    parser.add_argument('--list', action='store_true', help='Print the sample and stop')
    args = parser.parse_args()
    frames = lorri_frames(Path(os.environ['PDS3_HOLDINGS_DIR']), args.per_group)
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    if args.list:
        for f in frames:
            print(f.image_id, f.mode, f.texp, f.extra['target'], f.extra['volume'])
        return
    run_frames(frames, measure, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
