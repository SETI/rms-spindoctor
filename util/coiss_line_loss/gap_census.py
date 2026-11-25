"""Measure the groups of lost lines on real Cassini ISS frames, as the gap shape defaults do.

Reads every frame in ``gap_frames.txt`` (the losslessly compressed 12-bit frames of
COISS_2010, COISS_2023, COISS_2050, and COISS_2068 whose line ends the compressor did not
cut and that lose whole lines inside the frame) and finds, per frame:

* the groups: runs of lost lines that touch neither the first nor the last line;
* each group's head and tail lines: the line just above and just below it when that line
  kept its start and lost its end, with its kept samples.

It writes the groups to a JSON file for ``gap_fit.py`` and prints the tables the
``downlink_gaps`` shape defaults take, per camera:

* sizes, in lines of their own frame, for full frames and for summed frames;
* the count of groups per frame;
* the kept lines between neighboring groups, all of them and those of at most
  ``CLUSTER_GAP`` lines, which are the gaps inside a cluster;
* the share of lines bordering one group that lose their end, above and below, and the
  share of single kept lines between two groups that do;
* the kept shares of the head and tail lines.

A table gives the 0th, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th, and 100th
percentiles.  Size and gap tables are percentiles of the integers spread evenly over
each one's rounding interval (a size of 3 lines over 2.5 to 3.5), so that a draw rounded
to a whole line gives each integer its measured share; the 0th percentile is raised
where the smallest integer is rarer than the 5th percentile (see
:func:`spread_percentiles`).

Usage (the holdings root may be a local directory or a URL)::

    PDS3_HOLDINGS_DIR=/path/to/holdings python util/coiss_line_loss/gap_census.py OUT.json
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coiss_raw import RawFrame, read_raw_frame

from spindoctor.sim.artifact_modes import PERCENTILE_POINTS

FRAMES = Path(__file__).resolve().parent / 'gap_frames.txt'
# The longest run of kept lines between two groups of one cluster.
CLUSTER_GAP = 20


def frame_groups(frame: RawFrame) -> dict[str, Any]:
    """Return one frame's interior groups with their head and tail lines.

    Parameters:
        frame: The frame.

    Returns:
        The frame's name, readout mode, line count, and groups, each group as its first
        and last line and the kept samples of its head and tail lines (None where that
        line did not lose its end).
    """
    lines = frame.image.shape[0]
    lost = frame.first_start == 0
    end_lost = frame.end_lost
    groups: list[list[int | None]] = []
    v = 0
    while v < lines:
        if not lost[v]:
            v += 1
            continue
        first = v
        while v < lines and lost[v]:
            v += 1
        last = v - 1
        if first == 0 or last == lines - 1:
            continue
        head = int(frame.first_end[first - 1]) if end_lost[first - 1] else None
        tail = int(frame.first_end[last + 1]) if end_lost[last + 1] else None
        groups.append([first, last, head, tail])
    return {'name': frame.name, 'mode': frame.mode, 'lines': lines, 'groups': groups}


def spread_percentiles(values: list[int]) -> list[float]:
    """Return a table of integers spread evenly over each one's rounding interval.

    The simulator draws between a size or gap table's points in the logarithm of the
    value.  When the smallest integer is rarer than the table's 5th percentile, the
    stretch from the 0th to the 5th percentile holds it and part of the next integer,
    and a logarithmic draw over that stretch gives the smallest integer more than its
    share.  The 0th percentile is then raised to the value at which the logarithmic
    draw gives the smallest integer exactly its measured share.

    Parameters:
        values: The measured integers.

    Returns:
        The values at the table's percentiles.
    """
    ks, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    weights = counts / counts.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    table = []
    for point in PERCENTILE_POINTS:
        i = min(int(np.searchsorted(cumulative, point, side='right')) - 1, ks.size - 1)
        table.append(float(ks[i] - 0.5 + min(1.0, (point - cumulative[i]) / weights[i])))
    ratio = float(weights[0]) / PERCENTILE_POINTS[1]
    if ratio < 1.0:
        # Solve log(top / x) = ratio * log(table[1] / x) for x, where top is the half
        # line above the smallest integer, so the draws below top hold its share.
        top = math.log(float(ks[0]) + 0.5)
        table[0] = math.exp((top - ratio * math.log(table[1])) / (1.0 - ratio))
    return table


def plain_percentiles(values: list[float]) -> list[float]:
    """Return a table of values at the table's percentiles.

    Parameters:
        values: The measured values.

    Returns:
        The values at the table's percentiles.
    """
    points = 100.0 * np.asarray(PERCENTILE_POINTS)
    return [float(x) for x in np.percentile(np.asarray(values, dtype=np.float64), points)]


def gaps_of(frame: dict[str, Any]) -> list[int]:
    """Return the kept lines between each pair of neighboring groups in a frame.

    Parameters:
        frame: A frame as :func:`frame_groups` returns it.

    Returns:
        The gaps, top to bottom.
    """
    groups = frame['groups']
    return [b[0] - a[1] - 1 for a, b in itertools.pairwise(groups)]


def edge_rates(frames: list[dict[str, Any]]) -> tuple[float, float, float, int]:
    """Return the end-loss shares of lines bordering one group and of shared lines.

    Parameters:
        frames: The frames.

    Returns:
        The share of lines just above one group that lose their end, the share of lines
        just below one group that do, the share of single kept lines between two groups
        that do, and the number of such shared lines.
    """
    above: list[bool] = []
    below: list[bool] = []
    shared: list[bool] = []
    for frame in frames:
        groups = frame['groups']
        for i, (first, last, head, tail) in enumerate(groups):
            if not (i > 0 and groups[i - 1][1] == first - 2):
                above.append(head is not None)
            if i + 1 < len(groups) and groups[i + 1][0] == last + 2:
                shared.append(tail is not None)
            else:
                below.append(tail is not None)
    return float(np.mean(above)), float(np.mean(below)), float(np.mean(shared)), len(shared)


def report(label: str, frames: list[dict[str, Any]]) -> None:
    """Print one camera's tables.

    Parameters:
        label: The camera's name.
        frames: The camera's frames.
    """

    def fmt(table: list[float], digits: int = 1) -> str:
        return ', '.join(f'{x:.{digits}f}' for x in table)

    counts = np.array([len(f['groups']) for f in frames])
    print(f'== {label}: {len(frames)} frames, {int(counts.sum())} groups')
    print(
        f'count per frame: mean {counts.mean():.2f}, variance {counts.var():.1f}, '
        f'one group on {np.mean(counts == 1):.3f}, 90th percentile '
        f'{np.percentile(counts, 90):.1f}, largest {counts.max()}'
    )
    for name, summed in (('full', False), ('summed', True)):
        chosen = [f for f in frames if (f['mode'] != 'FULL') == summed]
        sizes = [g[1] - g[0] + 1 for f in chosen for g in f['groups']]
        if sizes:
            print(f'{name} sizes ({len(chosen)} frames, {len(sizes)} groups):')
            print(f'  [{fmt(spread_percentiles(sizes), 2)}]')
    gaps = [g for f in frames for g in gaps_of(f)]
    inner = [g for g in gaps if g <= CLUSTER_GAP]
    print(f'gaps ({len(gaps)}): {fmt(plain_percentiles(gaps))}')
    print(f'gaps of at most {CLUSTER_GAP} lines ({len(inner)}):')
    print(f'  [{fmt(spread_percentiles(inner), 2)}]')
    above, below, shared, n_shared = edge_rates(frames)
    print(
        f'lines bordering one group that lose their end: above {above:.3f}, below '
        f'{below:.3f}; single kept lines between two groups: {shared:.3f} of {n_shared}'
    )
    for name, index in (('head', 2), ('tail', 3)):
        kept = [g[index] / f['lines'] for f in frames for g in f['groups'] if g[index]]
        rate = len(kept) / counts.sum()
        print(f'{name} lines: {rate:.3f} of the groups; kept shares ({len(kept)}):')
        print(f'  [{fmt(plain_percentiles(kept), 4)}]')
    spans = [
        (f['groups'][-1][1] - f['groups'][0][0] + 1) / f['lines']
        for f in frames
        if len(f['groups']) > 1
    ]
    quartiles = np.percentile(spans, [25, 50, 75])
    print(f'span of frames with more than one group ({len(spans)}): {fmt(quartiles, 2)}')


def main() -> None:
    """Read the frames, write their groups, and print each camera's tables."""
    root = FCPath(os.environ['PDS3_HOLDINGS_DIR'])
    frames = []
    for line in FRAMES.read_text().split():
        frame = frame_groups(read_raw_frame(root / line))
        if frame['groups']:
            frames.append(frame)
    Path(sys.argv[1]).write_text(json.dumps(frames, indent=1))
    for label, initial in (('NAC', 'N'), ('WAC', 'W')):
        report(label, [f for f in frames if f['name'].startswith(initial)])


if __name__ == '__main__':
    main()
