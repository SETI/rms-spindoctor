"""Check the simulator's lossless line-pair coder against real compression-truncated frames.

For every frame in ``lossless_frames.txt`` the check codes each line pair whose first
line is whole with :func:`spindoctor.sim.render.telemetry.lossless.block_bits` (16-sample
blocks, 4 option bits, options 0 to 13) and compares the coded size with the pair's
budget, 16 bits per sample of one line:

* a pair whose second line kept its start and lost its end: the first line plus the
  second line's kept blocks, which the model says sits just under the budget;
* a whole pair: both lines, which the model says fits the budget;
* the budget left after the last kept block, in units of the mean cost of the last four
  kept blocks, which the model says is mostly under one (the next block would not fit).

Usage (the holdings root may be a local directory or a URL)::

    PDS3_HOLDINGS_DIR=/path/to/holdings python util/coiss_line_loss/lossless_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from filecache import FCPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coiss_raw import RawFrame, read_raw_frame

from spindoctor.sim.render.telemetry.lossless import block_bits

FRAMES = Path(__file__).resolve().parent / 'lossless_frames.txt'
BLOCK = 16
WORD_BITS = 16
ID_BITS = 4
MAX_SPLIT = 13


def pair_budget_use(frame: RawFrame) -> tuple[list[float], list[float], list[float]]:
    """Return the budget use of cut pairs and whole pairs, and the leftover at each cut.

    Parameters:
        frame: The frame.

    Returns:
        The coded size over the budget of each cut pair and of each whole pair, and the
        leftover at each cut in units of the line's recent block cost.
    """
    lines, samples = frame.image.shape
    budget = WORD_BITS * samples
    bits = block_bits(frame.image, block_samples=BLOCK, id_bits=ID_BITS, max_split=MAX_SPLIT)
    whole, end_lost = frame.whole, frame.end_lost
    cut_use: list[float] = []
    whole_use: list[float] = []
    leftover: list[float] = []
    for first in range(0, lines - 1, 2):
        if not whole[first]:
            continue
        first_bits = int(bits[first].sum())
        second = first + 1
        if whole[second]:
            whole_use.append((first_bits + int(bits[second].sum())) / budget)
        elif end_lost[second]:
            kept = int(frame.first_end[second]) // BLOCK
            used = first_bits + int(bits[second, :kept].sum())
            cut_use.append(used / budget)
            recent = float(bits[second, max(0, kept - 4) : kept].mean()) if kept else 1.0
            leftover.append((budget - used) / max(1.0, recent))
    return cut_use, whole_use, leftover


def _percentiles(values: list[float], points: tuple[float, ...]) -> str:
    """Format the percentiles of ``values``, or a dash for none."""
    if not values:
        return '-'
    return ' / '.join(f'{v:.3f}' for v in np.percentile(values, points))


def main() -> int:
    """Run the check over every listed frame and print per-frame and pooled results."""
    root = FCPath(os.environ['PDS3_HOLDINGS_DIR'])
    all_cut: list[float] = []
    all_whole: list[float] = []
    all_left: list[float] = []
    for relative in FRAMES.read_text().split():
        frame = read_raw_frame(root / relative)
        cut, whole, left = pair_budget_use(frame)
        all_cut += cut
        all_whole += whole
        all_left += left
        print(
            f'{frame.name} {frame.mode}: {len(cut)} cut pairs, use p5/50/95 '
            f'{_percentiles(cut, (5, 50, 95))}; {len(whole)} whole pairs, max '
            f'{max(whole) if whole else float("nan"):.3f}; leftover p50/95 '
            f'{_percentiles(left, (50, 95))}'
        )
    cut_array = np.asarray(all_cut)
    whole_array = np.asarray(all_whole)
    print(f'frames: {len(FRAMES.read_text().split())}')
    print(
        f'cut pairs: {cut_array.size}, use p0.1/1/50/99/99.9 '
        f'{_percentiles(all_cut, (0.1, 1, 50, 99, 99.9))}, outside [0.96, 1.01]: '
        f'{int(np.sum((cut_array < 0.96) | (cut_array > 1.01)))}'
    )
    print(
        f'whole pairs: {whole_array.size}, max {whole_array.max():.4f}, over the budget: '
        f'{int(np.sum(whole_array > 1.0))}'
    )
    print(
        f'leftover at the cut, in recent blocks: p50/95/99 {_percentiles(all_left, (50, 95, 99))}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
