"""Compare the simulator's downlink gaps with the census, and fit the two drawn parameters.

Reads the groups ``gap_census.py`` wrote and renders simulated frames with the
simulator's ``downlink_gaps`` applier, at the camera's shape defaults, on flat frames of
the census frames' heights and readout modes in turn.  Each simulated frame is measured as
the census measures a real one: a group is a run of lost lines that touches neither the
first nor the last line, and its head and tail lines are the lines just above and below
it that keep their start and lose their end.

The comparison prints every statistic three times: over all of the camera's frames, over
its full frames, and over its summed frames.  Each line gives the census value, the 90%
interval of that value over resampled census frames (the frames drawn with replacement,
each readout mode keeping its number of frames), the simulated value, and whether the
simulated value lies inside the interval.  It then splits each readout mode's frames into
sequences, frames less than a day apart on the spacecraft clock, and prints how much of the
mode the largest sequence holds.

``--fit`` fits the two parameters that the census does not give directly:

* ``cluster_probability``, at the value on a grid from 0.20 to 0.70 that minimizes the sum
  of the squared logarithmic errors of the gaps' 25th, 50th, 75th, and 90th percentiles
  and the squared errors of the span's quartiles in tenths of the height, at a quarter of
  the weight;
* ``mean_groups`` and ``dispersion``: the dispersion gives the census's share of frames
  with one group, and the mean, searched on a grid, gives the census's mean count after
  the frames too small for their draw lose groups.

Usage::

    python util/coiss_line_loss/gap_fit.py CENSUS.json coiss_nac [FRAMES] [key=value ...]
    python util/coiss_line_loss/gap_fit.py CENSUS.json coiss_nac --fit
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import brentq

from spindoctor.sim.instruments.resolvers import resolve_mode_with_catalog
from spindoctor.sim.render.feature_loci import FeatureLoci
from spindoctor.sim.render.telemetry.loss import apply_downlink_gaps

_BINNING = {'FULL': 1, 'SUM2': 2, 'SUM4': 4}
_NONE = np.zeros(0, dtype=np.int64)
_LOCI = FeatureLoci(rows=_NONE, pixel_v=_NONE, pixel_u=_NONE)
_SHARES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
_SIZE_POINTS = (10, 25, 50, 75, 90, 99)
_GAP_POINTS = (25, 50, 75, 90)
_SPAN_POINTS = (25, 50, 75)
# Resampled census sets per interval, and the generator's seed.
_RESAMPLES = 2000
_RESAMPLE_SEED = 1
# Frames of one camera less than this far apart on the spacecraft clock, in seconds,
# belong to one sequence.
_SEQUENCE_GAP = 86400

Item = tuple[int, list[tuple[int, int]], list[tuple[bool, bool]]]


def census_items(frames: list[dict[str, Any]]) -> list[Item]:
    """Return the census frames as (lines, groups, head and tail flags) items.

    Parameters:
        frames: The frames ``gap_census.py`` wrote.

    Returns:
        One item per frame.
    """
    return [
        (
            f['lines'],
            [(g[0], g[1]) for g in f['groups']],
            [(g[2] is not None, g[3] is not None) for g in f['groups']],
        )
        for f in frames
    ]


def measure_frame(signal: np.ndarray) -> Item:
    """Measure one simulated frame as the census measures a real one.

    Parameters:
        signal: The frame, with lost pixels at 0.

    Returns:
        The frame's lines, its groups as first and last lines, and each group's head and
        tail flags.
    """
    lines = signal.shape[0]
    good = signal != 0.0
    lost = ~good.any(axis=1)
    first_bad = np.argmin(good, axis=1)
    end_lost = good[:, 0] & ~good.all(axis=1)
    end_lost &= ~np.array([good[v, first_bad[v] :].any() for v in range(lines)])
    groups: list[tuple[int, int]] = []
    v = 0
    while v < lines:
        if not lost[v]:
            v += 1
            continue
        first = v
        while v < lines and lost[v]:
            v += 1
        if first != 0 and v != lines:
            groups.append((first, v - 1))
    flags = [(bool(end_lost[a - 1]), bool(end_lost[b + 1])) for a, b in groups]
    return lines, groups, flags


def statistics(items: list[Item], modes: list[str]) -> dict[str, float]:
    """Return the statistics the census and the simulation are compared on.

    Parameters:
        items: One (lines, groups, flags) item per frame with at least one group.
        modes: Each item's readout mode.

    Returns:
        The statistics by name.  A statistic with nothing to measure, such as the
        summed sizes of full frames, is left out.
    """
    counts = np.array([len(g) for _, g, _ in items])
    sizes: dict[bool, list[int]] = {False: [], True: []}
    gaps: list[int] = []
    spans: list[float] = []
    heads = tails = 0
    for (lines, groups, flags), mode in zip(items, modes, strict=True):
        for a, b in groups:
            sizes[mode != 'FULL'].append(b - a + 1)
        gaps += [b[0] - a[1] - 1 for a, b in itertools.pairwise(groups)]
        if len(groups) > 1:
            spans.append((groups[-1][1] - groups[0][0] + 1) / lines)
        heads += sum(h for h, _ in flags)
        tails += sum(t for _, t in flags)
    total = int(counts.sum())
    stats = {
        'count mean': float(counts.mean()),
        'count variance': float(counts.var()),
        'one group': float(np.mean(counts == 1)),
        'count p90': float(np.percentile(counts, 90)),
    }
    for name, summed in (('full', False), ('summed', True)):
        if sizes[summed]:
            values = np.percentile(sizes[summed], _SIZE_POINTS)
            stats |= {
                f'{name} sizes p{p}': float(x) for p, x in zip(_SIZE_POINTS, values, strict=True)
            }
    every_size = np.asarray(sizes[False] + sizes[True])
    stats['one-line groups'] = float(np.mean(every_size == 1))
    if gaps:
        values = np.percentile(gaps, _GAP_POINTS)
        stats |= {f'gaps p{p}': float(x) for p, x in zip(_GAP_POINTS, values, strict=True)}
        stats['one-line gaps'] = float(np.mean(np.asarray(gaps) == 1))
    if spans:
        values = np.percentile(spans, _SPAN_POINTS)
        stats |= {f'span p{p}': float(x) for p, x in zip(_SPAN_POINTS, values, strict=True)}
    stats['head rate'] = heads / total
    stats['tail rate'] = tails / total
    return stats


def resampling_intervals(items: list[Item], modes: list[str]) -> dict[str, tuple[float, float]]:
    """Return each statistic's 90% interval over resampled frames.

    Each resampled set draws the frames of every readout mode with replacement, as many
    as the mode has, and the interval runs from the 5th to the 95th percentile of the
    statistic over the sets.

    Parameters:
        items: One (lines, groups, flags) item per frame with at least one group.
        modes: Each item's readout mode.

    Returns:
        The interval of each statistic by name.
    """
    rng = np.random.default_rng(_RESAMPLE_SEED)
    by_mode: dict[str, list[int]] = {}
    for index, mode in enumerate(modes):
        by_mode.setdefault(mode, []).append(index)
    values: dict[str, list[float]] = {}
    for _ in range(_RESAMPLES):
        chosen = [int(i) for members in by_mode.values() for i in rng.choice(members, len(members))]
        stats = statistics([items[i] for i in chosen], [modes[i] for i in chosen])
        for key, value in stats.items():
            values.setdefault(key, []).append(value)
    return {
        key: (float(np.percentile(v, 5)), float(np.percentile(v, 95))) for key, v in values.items()
    }


def sequence_summary(frames: list[dict[str, Any]]) -> str:
    """Describe how a set of frames splits into sequences.

    Parameters:
        frames: Frames of one camera.

    Returns:
        The number of frames and sequences, and the largest sequence's frames, groups,
        and gaps against the set's.
    """
    ordered = sorted(frames, key=lambda f: int(f['name'][1:11]))
    sequences: list[list[dict[str, Any]]] = []
    previous = None
    for frame in ordered:
        clock = int(frame['name'][1:11])
        if previous is None or clock - previous >= _SEQUENCE_GAP:
            sequences.append([])
        sequences[-1].append(frame)
        previous = clock

    def groups(chosen: list[dict[str, Any]]) -> int:
        return sum(len(f['groups']) for f in chosen)

    largest = max(sequences, key=groups)
    first, last = largest[0]['name'][:11], largest[-1]['name'][:11]
    return (
        f'{len(frames)} frames in {len(sequences)} sequences; the largest, {first} to '
        f'{last}, holds {len(largest)} frames, {groups(largest)} of {groups(frames)} groups, '
        f'and {groups(largest) - len(largest)} of {groups(frames) - len(frames)} gaps'
    )


def simulate(
    frames: list[dict[str, Any]], instrument: str, count: int, overrides: dict[str, Any]
) -> tuple[list[Item], list[str]]:
    """Render and measure simulated frames at the census frames' heights and modes.

    Parameters:
        frames: The census frames, cycled for each simulated frame's shape.
        instrument: The simulated camera.
        count: The number of frames.
        overrides: Parameters that replace the camera's shape defaults.

    Returns:
        The measured items of the frames with at least one group, and their modes.
    """
    cfg = resolve_mode_with_catalog('downlink_gaps', {'probability': 1.0}, instrument)
    cfg.update(overrides)
    items: list[Item] = []
    modes: list[str] = []
    for seed in range(count):
        frame = frames[seed % len(frames)]
        lines = frame['lines']
        signal = np.full((lines, lines), 500.0)
        apply_downlink_gaps(
            signal,
            cfg,
            marker_dn=0.0,
            rng=np.random.default_rng(seed),
            loci=_LOCI,
            adversarial=False,
            binning=_BINNING[frame['mode']],
        )
        item = measure_frame(signal)
        if item[1]:
            items.append(item)
            modes.append(frame['mode'])
    return items, modes


def fit_loss(census: dict[str, float], simulated: dict[str, float]) -> float:
    """Return the cluster-share fit's loss.

    Parameters:
        census: The census statistics.
        simulated: The simulated statistics.

    Returns:
        The loss.
    """
    gaps = [np.log(simulated[f'gaps p{p}'] / census[f'gaps p{p}']) for p in _GAP_POINTS]
    spans = [(simulated[f'span p{p}'] - census[f'span p{p}']) / 0.1 for p in _SPAN_POINTS]
    return float(np.sum(np.square(gaps)) + 0.25 * np.sum(np.square(spans)))


def fit(frames: list[dict[str, Any]], instrument: str) -> None:
    """Fit the cluster share, then the count law, and print each step.

    Parameters:
        frames: The census frames of one camera.
        instrument: The simulated camera.
    """
    census = statistics(census_items(frames), [f['mode'] for f in frames])
    losses = []
    for share in _SHARES:
        simulated = statistics(*simulate(frames, instrument, 3000, {'cluster_probability': share}))
        losses.append(fit_loss(census, simulated))
        print(f'cluster_probability {share:.2f}: loss {losses[-1]:.3f}')
    share = _SHARES[int(np.argmin(losses))]
    one = census['one group']
    for mean in np.arange(0.8, 1.21, 0.05) * (census['count mean'] - 1.0) + 1.0:
        extra = mean - 1.0
        dispersion = brentq(lambda d, e=extra: (1.0 + d * e) ** (-1.0 / d) - one, 1e-3, 50.0)
        overrides = {
            'cluster_probability': share,
            'mean_groups': round(float(mean), 2),
            'dispersion': round(float(dispersion), 2),
        }
        simulated = statistics(*simulate(frames, instrument, 20000, overrides))
        print(
            f'{overrides}: count mean {simulated["count mean"]:.2f}, variance '
            f'{simulated["count variance"]:.1f}, one group {simulated["one group"]:.3f}, '
            f'p90 {simulated["count p90"]:.0f}'
        )


def compare(
    frames: list[dict[str, Any]], instrument: str, count: int, overrides: dict[str, Any]
) -> None:
    """Print the census, its intervals, and the simulated statistics, pooled and per mode.

    Parameters:
        frames: The census frames of one camera.
        instrument: The simulated camera.
        count: The number of simulated frames.
        overrides: Parameters that replace the camera's shape defaults.
    """
    items = census_items(frames)
    modes = [f['mode'] for f in frames]
    sim_items, sim_modes = simulate(frames, instrument, count, overrides)
    print(f'{instrument}, {count} simulated frames, {overrides or "shape defaults"}')
    for label in ('all', 'full', 'summed'):

        def chosen(mode: str, label: str = label) -> bool:
            return label == 'all' or (mode != 'FULL') == (label == 'summed')

        picked = [i for i, m in enumerate(modes) if chosen(m)]
        sim_picked = [i for i, m in enumerate(sim_modes) if chosen(m)]
        census = statistics([items[i] for i in picked], [modes[i] for i in picked])
        intervals = resampling_intervals([items[i] for i in picked], [modes[i] for i in picked])
        simulated = statistics(
            [sim_items[i] for i in sim_picked], [sim_modes[i] for i in sim_picked]
        )
        print(f'== {label} frames: {len(picked)} real, {len(sim_picked)} simulated')
        if label != 'all':
            print('   ' + sequence_summary([frames[i] for i in picked]))
        print(f'   {"statistic":18s} {"census":>9s} {"90% interval":>21s} {"simulated":>10s}')
        outside = []
        for key, value in census.items():
            low, high = intervals[key]
            sim = simulated.get(key, float('nan'))
            inside = low <= sim <= high
            if not inside:
                outside.append(key)
            print(
                f'   {key:18s} {value:9.3f} {low:9.3f} .. {high:9.3f} {sim:10.3f}'
                f'{"" if inside else "  outside"}'
            )
        print(f'   outside the interval: {", ".join(outside) or "none"}')


def main() -> None:
    """Compare or fit one camera."""
    frames = json.loads(Path(sys.argv[1]).read_text())
    instrument = sys.argv[2]
    initial = 'N' if instrument.endswith('nac') else 'W'
    frames = [f for f in frames if f['name'].startswith(initial)]
    if '--fit' in sys.argv:
        fit(frames, instrument)
        return
    rest = sys.argv[3:]
    count = int(rest.pop(0)) if rest and '=' not in rest[0] else 3000
    overrides = {k: yaml.safe_load(v) for k, v in (a.split('=', 1) for a in rest)}
    compare(frames, instrument, count, overrides)


if __name__ == '__main__':
    main()
