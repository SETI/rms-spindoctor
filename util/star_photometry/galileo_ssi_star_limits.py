"""Measure Galileo SSI star fields for the photometric star limit and the star PSF.

Frames are those the PDS3 index names a star-field target (``BLACK_SKY``, ``STAR``, or a named
star or cluster such as ``PLEIADES``), on every volume or on the volumes named on the command
line, read through the navigator's own reader.  The cruise star calibrations of November 1990
(``GO_0002``) supply most measurable stars; the black-sky frames of ``GO_0008`` and ``GO_0009``
(December 1992) are the population whose independent fixes the star-detection work uses as
truth.  SSI images are raw 8-bit DN, so the navigator's image is also the photometric image.

Each frame's photometric state comes from the index: ``FILTER_NAME``, ``GAIN_MODE_ID`` (the
full-well setting ``10K``, ``40K``, ``100K`` or ``400K``), ``EXPOSURE_DURATION`` (milliseconds in
the index), the supplemental index's ``TELEMETRY_FORMAT_ID`` (the readout format; a summation
frame is 400 x 400) and ``COMPRESSION_TYPE``.

Usage::

    python -m util.star_photometry.galileo_ssi_star_limits OUT.jsonl [--volumes GO_0002 ...]
        [--workers 4] [--limit N]

writes one JSON line per frame (see :mod:`util.star_photometry.frame_measure`).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)

from spindoctor.obs import ObsGalileoSSI

__all__ = ['PARAMS', 'galileo_frames', 'main', 'measure']

#: PDS3 index targets of frames pointed at a star field.
STAR_FIELD_TARGETS = frozenset(
    {
        'BLACK_SKY',
        'STAR',
        'PLEIADES',
        'BET_ARI',
        'TAU_CET',
        'GAM_ORI',
        'DEL_PSC',
        'SIRIUS',
        'HD151288',
    }
)

#: Milliseconds per second (the index states exposures in milliseconds).
_MS_PER_S = 1000.0

#: The 8-bit DN at which SSI saturates.
SATURATION_DN = 255.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: Columns of the cumulative PDS3 index ``GO_0999_index.tab``.
_COL_VOLUME, _COL_FILE, _COL_TARGET = 0, 1, 9
_COL_FILTER, _COL_EXPOSURE, _COL_GAIN, _COL_COMPRESSION = 11, 13, 14, 47
#: Column of the supplemental index's ``TELEMETRY_FORMAT_ID`` (the readout format).
_COL_SUP_MODE = 2

PARAMS = MeasureParams(
    catalog_depth_mag=4.0,
    highpass_size=31,
    smooth_sigma=1.5,
    peak_threshold_snr=5.0,
    edge_px=16,
    max_peaks=300,
    max_vote_stars=60,
    search_px=350.0,
    vote_bin_px=1.5,
    agree_px=1.0,
    min_agree=3,
    max_chance_probability=1e-3,
    detect_search_px=1.5,
    noise_inner=8.0,
    noise_outer=15.0,
    box_half=2,
    aperture_px=4.0,
    big_aperture_px=8.0,
    sky_inner=10.0,
    sky_outer=15.0,
    blend_px=10.0,
    blend_dmag=2.5,
    saturation_dn=SATURATION_DN,
    max_smear_px=1.0,
    psf_min_snr=20.0,
    psf_half=8,
    ee_r_max=10.0,
    ee_n_bins=20,
)


def galileo_frames(holdings: Path, volumes: tuple[str, ...]) -> list[FrameInput]:
    """Return the star-field frames of the given volumes, from the cumulative PDS3 index.

    Parameters:
        holdings: The PDS3 holdings root (``$PDS3_HOLDINGS_DIR``).
        volumes: Volume IDs to include; every volume when empty.

    Returns:
        One entry per frame, in index order.
    """
    meta = holdings / 'metadata' / 'GO_0xxx' / 'GO_0999'
    with (meta / 'GO_0999_supplemental_index.tab').open() as fh:
        modes = {r[1].strip(): r[_COL_SUP_MODE].strip() for r in csv.reader(fh)}
    frames: list[FrameInput] = []
    with (meta / 'GO_0999_index.tab').open() as fh:
        for row in csv.reader(fh):
            volume = row[_COL_VOLUME].strip()
            target = row[_COL_TARGET].strip()
            if (volumes and volume not in volumes) or target not in STAR_FIELD_TARGETS:
                continue
            spec = row[_COL_FILE].strip()
            image = holdings / 'volumes' / 'GO_0xxx' / volume / spec.replace('.LBL', '.IMG')
            frames.append(
                FrameInput(
                    image_id=Path(spec).stem,
                    path=str(image),
                    camera='SSI',
                    filter_name=row[_COL_FILTER].strip(),
                    gain_state=row[_COL_GAIN].strip(),
                    mode=modes.get(spec, ''),
                    texp=float(row[_COL_EXPOSURE]) / _MS_PER_S,
                    extra={
                        'volume': volume,
                        'target': target,
                        'compression': row[_COL_COMPRESSION].strip(),
                    },
                )
            )
    return frames


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one Galileo SSI frame (the pool worker).

    Parameters:
        frame: The frame.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsGalileoSSI.from_file(frame.path, config=config)
    record = measure_frame(obs, frame, PARAMS, config=config)
    record['obs_texp'] = float(obs.texp)
    return record


def main() -> None:
    """Measure the Galileo SSI star-field frames and append the records to a JSON-lines file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='JSON-lines output (appended; resumes)')
    parser.add_argument('--volumes', nargs='*', default=[], help='Volumes (default: all)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Measure only the first N frames')
    parser.add_argument('--only', nargs='*', default=[], help='Measure only these image IDs')
    args = parser.parse_args()
    frames = galileo_frames(Path(os.environ['PDS3_HOLDINGS_DIR']), tuple(args.volumes))
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    run_frames(frames, measure, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
