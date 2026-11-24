"""Navigate every retrievable cohort frame and record what each one produced.

Runs the full autonomous pipeline (every model, every technique -- not a
pinned run) on each frame of ``titan_images.csv``, so the row carries both
what ``TitanHazeNav`` measured and what any INDEPENDENT technique on the same
frame measured.  The star techniques are the point of that: a star lock and a
haze lock on one frame are two measurements of the same scene-wide
translation, which is the strongest per-frame truth this campaign can get
without an operator eyeball.

Per-frame ``*_metadata.json``, ``*_summary.png``, and the per-image log land
in the campaign directory exactly as a production run writes them; this script
additionally distils one JSON line per frame into ``rows.jsonl`` for the
analyzer.

Run (from an activated project venv; ``source /seti/newnav/setup.sh``)::

    python util/titan_cohort/collect.py --workers 10 \\
        --campaign-dir _work/titan_cohort/run1

Campaign outputs are large and are not committed; the analyzer's report and
the campaign record are.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

# Pin native thread pools before the first numpy import (fork workers inherit
# the parent's already-configured runtime, so a pool initializer is too late).
for _thread_var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_thread_var, '1')

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from cohort import CohortFrame, holdings_root, resolved_cohort  # noqa: E402

TECHNIQUE = 'TitanHazeNav'

# Techniques whose offset is an INDEPENDENT witness of the same translation.
# StarRefineNav is deliberately absent: it is a pass-2 technique seeded by the
# pass-1 prior (which on a Titan frame is usually TitanHazeNav's own answer),
# so its agreement with Titan is not evidence of anything.
INDEPENDENT_WITNESSES = ('StarFieldFromCatalogNav', 'StarUniqueMatchNav')

DEFAULT_CAMPAIGN_DIR = REPO / '_work/titan_cohort/run1'


def _row_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Distil the curated per-image metadata into one analyzer row."""
    nav = metadata.get('navigation_result') or {}
    row: dict[str, Any] = {
        'status': metadata.get('status'),
        'nav_status': nav.get('status'),
        'status_reason': nav.get('status_reason'),
        'confidence': nav.get('confidence'),
        'confidence_rank': nav.get('confidence_rank'),
        'offset_px': nav.get('offset_px'),
        'sigma_px': nav.get('sigma_px'),
        'techniques': {},
        'features': [],
    }
    for entry in nav.get('per_technique', []):
        row['techniques'][entry['technique_name']] = {
            'offset_px': entry.get('offset_px'),
            'covariance_px2': entry.get('covariance_px2'),
            'confidence': entry.get('confidence'),
            'spurious': entry.get('spurious'),
            'at_edge': entry.get('at_edge'),
            'diagnostics': entry.get('diagnostics'),
        }
    for feature in nav.get('feature_inventory', []):
        row['features'].append(
            {
                'feature_id': feature.get('feature_id'),
                'feature_type': feature.get('feature_type'),
                'reliability': feature.get('reliability'),
                'gated': feature.get('gated'),
                'gate_reason': feature.get('gate_reason'),
                'reliability_reasons': feature.get('reliability_reasons'),
            }
        )
    return row


def _navigate_one(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Worker: navigate one cohort frame and return its distilled row."""
    frame_dict, campaign_dir = task
    row: dict[str, Any] = dict(frame_dict)
    from filecache import FCPath

    from spindoctor.dataset.dataset import ImageFile, ImageFiles
    from spindoctor.navigate_image_files import navigate_image_files
    from spindoctor.obs import ObsCassiniISS

    url = FCPath(str(holdings_root() / frame_dict['rel_path']))
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=url,
                label_file_url=url,
                results_path_stub=frame_dict['image_id'],
            )
        ]
    )
    start = time.monotonic()
    try:
        _success, metadata = navigate_image_files(
            ObsCassiniISS, image_files, FCPath(campaign_dir), write_output_files=True
        )
    except Exception as exc:
        # A frame that raises is a finding, not a crash: record it and go on.
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row
    row.update(_row_from_metadata(metadata))
    row['elapsed_s'] = round(time.monotonic() - start, 2)
    return row


def _init_worker() -> None:
    """Silence the per-image and main loggers in a pool worker."""
    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)


def _frame_payload(frame: CohortFrame) -> dict[str, Any]:
    """The cohort-side fields carried onto every row."""
    return {
        'image_id': frame.image_id,
        'flags': list(frame.flags),
        'notes': frame.notes,
        'camera': frame.camera,
        'filters': frame.filters,
        'image_time': frame.image_time,
        'target': frame.target,
        'rel_path': frame.rel_path,
    }


def _titan_config() -> dict[str, Any]:
    """The Titan config values a campaign's numbers depend on."""
    from spindoctor.config import DEFAULT_CONFIG

    navigation = DEFAULT_CONFIG.titan.navigation
    return {
        'atmosphere_height_km': DEFAULT_CONFIG.titan.atmosphere_height,
        'min_envelope_diameter_px': navigation.min_envelope_diameter_px,
        'max_occluded_fraction': navigation.max_occluded_fraction,
        'symmetry': dataclasses.asdict(navigation.symmetry),
        'arc': dataclasses.asdict(navigation.arc),
    }


def main(argv: list[str] | None = None) -> int:
    """Navigate the cohort in a worker pool and write ``rows.jsonl``."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--campaign-dir', type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument(
        '--only',
        default='',
        help='comma-separated image ids to run instead of the whole cohort',
    )
    args = parser.parse_args(argv)

    frames = [f for f in resolved_cohort() if f.rel_path is not None]
    if args.only:
        wanted = {i.strip() for i in args.only.split(',') if i.strip()}
        frames = [f for f in frames if f.image_id in wanted]
    campaign_dir = args.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    out_path = campaign_dir / 'rows.jsonl'

    tasks = [(_frame_payload(f), str(campaign_dir)) for f in frames]
    print(f'{len(tasks)} frames -> {campaign_dir}')
    start = time.monotonic()
    done = 0
    with out_path.open('w') as out:
        out.write(
            json.dumps(
                {
                    'manifest': True,
                    'n_frames': len(tasks),
                    'technique': TECHNIQUE,
                    'independent_witnesses': list(INDEPENDENT_WITNESSES),
                    'titan_config': _titan_config(),
                }
            )
            + '\n'
        )
        with multiprocessing.Pool(
            processes=args.workers, initializer=_init_worker, maxtasksperchild=20
        ) as pool:
            for row in pool.imap_unordered(_navigate_one, tasks):
                out.write(json.dumps(row, sort_keys=True) + '\n')
                out.flush()
                done += 1
                marker = row.get('error') or row.get('status_reason') or row.get('nav_status')
                print(f'{done}/{len(tasks)} {row["image_id"]} {marker}', flush=True)
    print(f'Wrote {out_path} in {(time.monotonic() - start) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
