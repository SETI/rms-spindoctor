"""Tabulate the independent multi-star pointing fixes of the Galileo SSI black-sky frames.

The black-sky frames of ``GO_0008`` and ``GO_0009`` have no outside pointing.  The star-limit
measurement (:mod:`util.star_photometry.galileo_ssi_star_limits`) finds each frame's pointing
from catalog stars alone, outside the navigator's star techniques; this program writes those
fixes as a table the star-detection work can use as truth.

A frame is truth only when its fix was accepted: at least three catalog stars agree within the
agreement radius, and the chance of a cluster that large arising from random star-peak pairs
anywhere in the search window is below 1e-3.  Every other frame is listed too, marked as not
truth, with the reason.

Columns:

* ``image_id``, ``volume``, ``filter``, ``gain_state``, ``texp_s``;
* ``truth``: ``yes`` or ``no``;
* ``offset_v``, ``offset_u``: the fix in pixels, in the navigator's convention (a star predicted
  at ``(v, u)`` is found at ``(v + offset_v, u + offset_u)``, after the rotation);
* ``rotation_deg``: rotation about the field center (line 399.5, sample 399.5 in array-index
  coordinates; positive turns +u toward +v), fitted when at least five stars agree, else 0;
* ``n_agree``: catalog stars agreeing with the fix; ``spread_px``: their RMS distance from it;
* ``chance_probability``; ``n_catalog_searched``; ``n_peaks``; ``reason`` (empty for truth).

Usage::

    python -m util.star_photometry.galileo_ssi_truth_table RECORDS.jsonl OUT.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

__all__ = ['TRUTH_VOLUMES', 'main', 'truth_rows']

#: Volumes whose black-sky frames make up the population.
TRUTH_VOLUMES = frozenset({'GO_0008', 'GO_0009'})

#: The population's PDS3 index target.
BLACK_SKY = 'BLACK_SKY'


def truth_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one table row per black-sky frame of the truth volumes.

    Parameters:
        records: Measurement records, including failed ones (``'error'`` present).

    Returns:
        The rows, sorted by image ID.
    """
    rows = []
    for record in records:
        frame = record['frame']
        extra = frame['extra']
        if extra['volume'] not in TRUTH_VOLUMES or extra['target'] != BLACK_SKY:
            continue
        row: dict[str, Any] = {
            'image_id': frame['image_id'],
            'volume': extra['volume'],
            'filter': frame['filter_name'],
            'gain_state': frame['gain_state'],
            'texp_s': frame['texp'],
        }
        if 'error' in record:
            last = record['error'].strip().splitlines()[-1]
            rows.append({**row, 'truth': 'no', 'reason': f'frame not measured: {last[:120]}'})
            continue
        fix = record['fix']
        rows.append(
            {
                **row,
                'truth': 'yes' if fix['accepted'] else 'no',
                'offset_v': round(fix['offset_v'], 3),
                'offset_u': round(fix['offset_u'], 3),
                'rotation_deg': round(fix['rotation_deg'], 4),
                'n_agree': fix['n_agree'],
                'spread_px': round(fix['spread_px'], 3),
                'chance_probability': f'{fix["chance_probability"]:.2g}',
                'n_catalog_searched': fix['n_candidates'],
                'n_peaks': fix['n_peaks'],
                'reason': fix['reason'],
            }
        )
    return sorted(rows, key=lambda r: r['image_id'])


def main() -> None:
    """Write the truth table of the Galileo SSI black-sky frames."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('records', type=Path, help='Galileo measurement records (JSON lines)')
    parser.add_argument('out', type=Path, help='The CSV table')
    args = parser.parse_args()
    with args.records.open() as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    rows = truth_rows(records)
    columns = [
        'image_id',
        'volume',
        'filter',
        'gain_state',
        'texp_s',
        'truth',
        'offset_v',
        'offset_u',
        'rotation_deg',
        'n_agree',
        'spread_px',
        'chance_probability',
        'n_catalog_searched',
        'n_peaks',
        'reason',
    ]
    with args.out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
