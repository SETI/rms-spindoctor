"""Compare the simulator's downlink gaps with the census, and fit the two drawn parameters.

Reads the groups ``gap_census.py`` wrote and renders simulated frames with the
simulator's ``downlink_gaps`` applier, at the camera's shape defaults, on flat frames of
the census frames' heights and readout modes in turn.  Each simulated frame is measured as
the census measures a real one: a group is a run of lost lines that touches neither the
first nor the last line, and its head and tail lines are the lines just above and below
it that keep their start and lose their end.

``--fit`` fits the two parameters that the census does not give directly:

* ``cluster_probability``, at the value that minimizes the sum of the squared logarithmic
  errors of the gaps' 25th, 50th, 75th, and 90th percentiles and the squared errors of the
  span's quartiles in tenths of the height, at a quarter of the weight;
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


def census_items(frames: list[dict[str, Any]]) -> list[tuple[int, list, list]]:
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


def measure_frame(signal: np.ndarray) -> tuple[int, list, list]:
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


def statistics(items: list[tuple[int, list, list]], modes: list[str]) -> dict[str, Any]:
    """Return the statistics the census and the simulation are compared on.

    Parameters:
        items: One (lines, groups, flags) item per frame with at least one group.
        modes: Each item's readout mode.

    Returns:
        The statistics by name.
    """
    counts = np.array([len(g) for _, g, _ in items])
    sizes = {False: [], True: []}
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
    return {
        'count mean': float(counts.mean()),
        'count variance': float(counts.var()),
        'one group': float(np.mean(counts == 1)),
        'count p90': float(np.percentile(counts, 90)),
        'full sizes p10/25/50/75/90/99': np.percentile(sizes[False], [10, 25, 50, 75, 90, 99]),
        'summed sizes p10/25/50/75/90/99': np.percentile(sizes[True], [10, 25, 50, 75, 90, 99]),
        'single-line share': float(np.mean(np.concatenate([sizes[False], sizes[True]]) == 1)),
        'gaps p10/25/50/75/90': np.percentile(gaps, [10, 25, 50, 75, 90]),
        'one-line gaps': float(np.mean(np.asarray(gaps) == 1)),
        'span p25/50/75': np.percentile(spans, [25, 50, 75]),
        'head rate': heads / total,
        'tail rate': tails / total,
    }


def simulate(
    frames: list[dict[str, Any]], instrument: str, count: int, overrides: dict[str, Any]
) -> tuple[list[tuple[int, list, list]], list[str]]:
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
    items: list[tuple[int, list, list]] = []
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


def fit_loss(census: dict[str, Any], simulated: dict[str, Any]) -> float:
    """Return the cluster-share fit's loss.

    Parameters:
        census: The census statistics.
        simulated: The simulated statistics.

    Returns:
        The loss.
    """
    gap_key, span_key = 'gaps p10/25/50/75/90', 'span p25/50/75'
    gaps = np.log(simulated[gap_key][1:] / census[gap_key][1:])
    spans = (simulated[span_key] - census[span_key]) / 0.1
    return float(np.sum(gaps**2) + 0.25 * np.sum(spans**2))


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
    census = statistics(census_items(frames), [f['mode'] for f in frames])
    simulated = statistics(*simulate(frames, instrument, count, overrides))
    print(f'{instrument}, {count} simulated frames, {overrides or "shape defaults"}')
    print(f'{"statistic":34s} {"census":>34s}   simulated')
    for key, value in census.items():
        if isinstance(value, float):
            print(f'{key:34s} {value:34.3f}   {simulated[key]:.3f}')
        else:
            text = ' '.join(f'{x:.1f}' if key.startswith('gap') else f'{x:.2f}' for x in value)
            sim = ' '.join(
                f'{x:.1f}' if key.startswith('gap') else f'{x:.2f}' for x in simulated[key]
            )
            print(f'{key:34s} {text:>34s}   {sim}')


if __name__ == '__main__':
    main()
