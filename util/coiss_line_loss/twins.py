"""Compare compression truncation on simulated twins of real frames with the real frames.

A twin keeps a real frame's scene and draws its fine structure anew.  The real frame's
lost line ends are first filled from the line above (the whole first line of the
pair), and a 9 x 9 median filter of the result gives the scene's smooth brightness.
The twin is that brightness plus Gaussian noise whose local spread matches the real
frame's residual about it, measured in 33 x 33 windows, rounded and clipped to 12 bits.
The twin then runs through the simulator's telemetry stage with only
``compression_truncated_lines`` on, so the cuts come from the simulator's compressor and
the twin's content alone.

For the real frame and its twin the script prints the statistics the compressor sets:
the share of line pairs cut, the first and last cut row as fractions of the height,
the median lost share and its spread within the frame, the change in lost share
between neighboring cut lines, and, for the twin, the budget use of cut and whole pairs
and the budget left at each cut in units of the next block's cost.

Usage::

    PDS3_HOLDINGS_DIR=/path/to/holdings python util/coiss_line_loss/twins.py [NAME ...]

Without names it uses the ten frames listed in ``TWIN_FRAMES``.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coiss_raw import RawFrame, read_raw_frame

from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.lossless import block_bits
from spindoctor.sim.render.telemetry.stage import apply_telemetry

FRAMES = Path(__file__).resolve().parent / 'lossless_frames.txt'
TWIN_FRAMES = (
    'N1490874834_1.IMG',
    'N1488900664_1.IMG',
    'N1527054318_1.IMG',
    'N1530339570_1.IMG',
    'N1490874889_1.IMG',
    'N1490893099_1.IMG',
    'N1490874782_1.IMG',
    'N1605482571_1.IMG',
    'N1606906298_1.IMG',
    'W1683890970_1.IMG',
)


def filled(frame: RawFrame) -> np.ndarray:
    """Return the frame with every lost sample taken from the line above (below on line 0)."""
    image = frame.image.astype(np.float64)
    lines = image.shape[0]
    valid = np.zeros_like(image, dtype=bool)
    for line in range(lines):
        start, end = int(frame.first_start[line]), int(frame.first_end[line])
        if start >= 1 and end >= start:
            valid[line, start - 1 : end] = True
        second = int(frame.second_start[line])
        if second > 0:
            valid[line, second - 1 :] = True
    for line in [*range(1, lines), 0]:
        source = line - 1 if line > 0 else 1
        image[line, ~valid[line]] = image[source, ~valid[line]]
    return image


def twin_of(frame: RawFrame, seed: int) -> np.ndarray:
    """Return a simulated twin of the frame's content (see the module docstring)."""
    image = filled(frame)
    smooth = ndimage.median_filter(image, size=9, mode='nearest')
    residual = image - smooth
    step = np.abs(np.diff(residual, axis=1, append=residual[:, -1:]))
    spread = 1.2533 / np.sqrt(2.0) * ndimage.uniform_filter(step, size=33, mode='nearest')
    rng = np.random.default_rng(seed)
    twin = smooth + spread * rng.standard_normal(image.shape)
    result: np.ndarray = np.clip(np.rint(twin), 0.0, 4095.0)
    return result


def real_cuts(frame: RawFrame) -> dict[int, int]:
    """Return the kept samples of each odd line that kept its start and lost its end."""
    rows = np.nonzero(frame.end_lost)[0]
    return {int(r): int(frame.first_end[r]) for r in rows if r % 2 == 1}


def sim_cuts(twin: np.ndarray, camera: str) -> tuple[dict[int, int], dict[str, Any]]:
    """Run the twin through the telemetry stage; return its odd-line cuts and the record."""
    frame = SimFrame(signal=twin.copy(), point_e=np.zeros_like(twin), oversample=1, truth={})
    params = {
        'instrument': camera,
        'random_seed': 1,
        'artifacts': {'compression_truncated_lines': {'probability': 1.0}},
    }
    apply_telemetry(frame, params=params)
    record: dict[str, Any] = frame.truth['artifacts']['compression_truncated_lines']
    cuts = {c['row']: c['lost_from'] for c in record['cuts'] if c['row'] % 2 == 1}
    return cuts, record


def cut_statistics(cuts: dict[int, int], lines: int, samples: int) -> str:
    """Format the frame-level statistics of a set of cuts."""
    if not cuts:
        return 'no cuts'
    rows = sorted(cuts)
    shares = np.array([1.0 - cuts[r] / samples for r in rows])
    neighbors = [
        abs(cuts[a] - cuts[b]) / samples for a, b in itertools.pairwise(rows) if b == a + 2
    ]
    near = np.percentile(neighbors, [50, 75]) if neighbors else (np.nan, np.nan)
    return (
        f'pairs cut {len(rows) / (lines // 2):.2f}, rows {rows[0] / lines:.2f}-'
        f'{rows[-1] / lines:.2f}, lost share median {np.median(shares):.3f} sd '
        f'{np.std(shares):.3f}, neighbor change p50/75 {near[0]:.3f}/{near[1]:.3f}'
    )


def budget_statistics(twin: np.ndarray, record: dict[str, Any]) -> str:
    """Format the twin's budget use and the leftover at each cut, in next-block costs."""
    lines = twin.shape[0]
    budget = int(record['budget_bits'])
    bits = block_bits(twin.astype(np.int64), block_samples=16, id_bits=4, max_split=13)
    cut_rows = {c['row']: c for c in record['cuts']}
    used, whole, left = [], [], []
    for first in range(0, lines - 1, 2):
        second = first + 1
        if second in cut_rows and first not in cut_rows:
            cut = cut_rows[second]
            used.append(cut['pair_bits'] / budget)
            next_block = cut['lost_from'] // 16
            left.append((budget - cut['pair_bits']) / float(bits[second, next_block]))
        elif first not in cut_rows:
            whole.append(float(bits[first].sum() + bits[second].sum()) / budget)
    use = np.percentile(used, [5, 50, 95]) if used else (np.nan,) * 3
    return (
        f'cut-pair use p5/50/95 {use[0]:.3f}/{use[1]:.3f}/{use[2]:.3f}, whole pairs max '
        f'{max(whole) if whole else np.nan:.3f}, leftover max {max(left) if left else np.nan:.3f}'
        ' blocks'
    )


def main() -> int:
    """Build each twin and print the real and simulated statistics side by side."""
    root = FCPath(os.environ['PDS3_HOLDINGS_DIR'])
    wanted = set(sys.argv[1:]) or set(TWIN_FRAMES)
    for relative in FRAMES.read_text().split():
        if Path(relative).name not in wanted:
            continue
        frame = read_raw_frame(root / relative)
        lines, samples = frame.image.shape
        camera = 'coiss_wac' if frame.name.startswith('W') else 'coiss_nac'
        twin = twin_of(frame, seed=1)
        cuts, record = sim_cuts(twin, camera)
        print(f'{frame.name} {frame.mode}')
        print(f'  real: {cut_statistics(real_cuts(frame), lines, samples)}')
        print(f'  twin: {cut_statistics(cuts, lines, samples)}')
        print(f'  twin: {budget_statistics(twin, record)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
