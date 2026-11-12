"""Regression bands for the realism match: what a fresh run is held to against the record.

The realism match (:mod:`tests.integration.sim_realism`) writes its summary to
``tests/integration/realism_results/realism_summary.json``, together with the commit of
the code that made it.  The regression check reruns the match on the LORRI cohort, whose
two frames take seconds, and compares every statistic the cohort supports with the
record's value.  A statistic is held when its support label, which counts the frames on
both sides, is ``limited`` or ``supported``.  The held values are each kind's W1/IQR
(when it is a number), the real and simulated medians, and each curve kind's W1/IQR.

Every band is two-sided.  A statistic that moves beyond its band fails whichever way it
moves, because a better-looking number can come from a worse simulator: a LORRI PSF core
three times too wide brought the star EE50 W1/IQR from 2.33 down to 1.04.

The measurement.  Navigation is not bit-for-bit repeatable, so each band rests on the
run-to-run spread of its statistic: the LORRI cohort was matched :data:`RUNS_MEASURED`
times on unchanged code, each run in its own process, and the spread is the largest value
minus the smallest over those runs.

The rule.  A statistic's band is a half-width around the recorded value:
max(:data:`SPREAD_MULTIPLE` times the spread, :data:`RELATIVE_FLOOR` times the magnitude
of the value measured).  The floor covers the arithmetic differences between machines,
which a spread measured on one machine cannot show.  :data:`REALISM_BANDS` holds the
measured value, the spread, and the band for every held statistic, and
:func:`band_rule` is the rule in code.

A record that carries no code commit predates the check and is not held to bands.  A
deliberate change to the simulator, or to the statistics, is accepted by regenerating
the record and then remeasuring the bands::

    python -m tests.integration.sim_realism
    python -m tests.integration.realism_regression
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tests.integration.checkout_guard import require_checkout_package

__all__ = [
    'INSTRUMENT',
    'REALISM_BANDS',
    'RELATIVE_FLOOR',
    'RUNS_MEASURED',
    'SPREAD_MULTIPLE',
    'RealismBand',
    'band_rule',
    'compare_with_record',
    'held_statistics',
]

INSTRUMENT = 'nhlorri'
"""The cohort the regression check reruns: two frames, matched in seconds."""

RUNS_MEASURED = 10
"""How many runs on unchanged code each spread was measured over."""

SPREAD_MULTIPLE = 3.0
"""A band is at least this many times its statistic's measured spread."""

RELATIVE_FLOOR = 0.02
"""A band is at least this fraction of its statistic's measured magnitude."""

REGENERATE_HINT = (
    'If the change is deliberate, accept it by regenerating the realism record '
    '(python -m tests.integration.sim_realism) and remeasuring the bands '
    '(python -m tests.integration.realism_regression).'
)

# The summary blocks whose entries are held, and the values held in each entry.
_HELD_FIELDS: dict[str, tuple[str, ...]] = {
    'divergences': ('w1_normalized', 'real_median', 'sim_median'),
    'curve_divergences': ('w1_normalized',),
}


@dataclass(frozen=True)
class RealismBand:
    """One held statistic's measurement and the band derived from it.

    Parameters:
        value: The statistic's median over the runs measured.
        spread: The largest minus the smallest value over the runs measured.
        band: The half-width a fresh value may lie from the recorded value,
            :func:`band_rule` of the spread and the value.
    """

    value: float
    spread: float
    band: float


def band_rule(spread: float, value: float) -> float:
    """The band for a statistic of measured spread ``spread`` and magnitude ``value``.

    Parameters:
        spread: The statistic's run-to-run spread.
        value: The statistic's measured value.

    Returns:
        max(SPREAD_MULTIPLE * spread, RELATIVE_FLOOR * abs(value)), rounded up at the
        third significant digit.
    """
    band = max(SPREAD_MULTIPLE * spread, RELATIVE_FLOOR * abs(value))
    if band <= 0.0:
        return 0.0
    scale = 10.0 ** (math.floor(math.log10(band)) - 2)
    return float(f'{math.ceil(band / scale - 1e-9) * scale:.3g}')


def _b(value: float, spread: float, band: float) -> RealismBand:
    """Return a :class:`RealismBand` from one table row, positionally for brevity."""
    return RealismBand(value=value, spread=spread, band=band)


# Columns: the median over the runs, the spread over the runs, the band.  Measured on
# 2026-11-11 over RUNS_MEASURED runs of the LORRI cohort on unchanged code.  Every
# statistic came out identical in all ten runs, so every spread is 0 and every band
# is the floor; a statistic whose value is 0 is held to exactly 0.
REALISM_BANDS: dict[str, RealismBand] = {
    'curve_divergences.star_profile.w1_normalized': _b(0.233617, 0.0, 0.00468),
    'divergences.artifact_missing_line_frac.real_median': _b(0.0, 0.0, 0.0),
    'divergences.artifact_missing_line_frac.sim_median': _b(0.0, 0.0, 0.0),
    'divergences.artifact_spike_frac.real_median': _b(0.00108337, 0.0, 2.17e-05),
    'divergences.artifact_spike_frac.sim_median': _b(0.0015564, 0.0, 3.12e-05),
    'divergences.frac_near_floor_ge_5s.real_median': _b(0.0455399, 0.0, 0.000911),
    'divergences.frac_near_floor_ge_5s.sim_median': _b(0.0927658, 0.0, 0.00186),
    'divergences.frac_saturated_ge_5s.real_median': _b(0.0, 0.0, 0.0),
    'divergences.frac_saturated_ge_5s.sim_median': _b(0.000213623, 0.0, 4.28e-06),
    'divergences.signal_p95_minus_p50_ge_5s.real_median': _b(4.48988, 0.0, 0.0898),
    'divergences.signal_p95_minus_p50_ge_5s.sim_median': _b(2.0, 0.0, 0.04),
    'divergences.signal_p99_minus_p50_ge_5s.real_median': _b(26.7749, 0.0, 0.536),
    'divergences.signal_p99_minus_p50_ge_5s.sim_median': _b(3.0, 0.0, 0.06),
    'divergences.star_ee50.real_median': _b(0.5869, 0.0, 0.0118),
    'divergences.star_ee50.sim_median': _b(0.485347, 0.0, 0.00971),
    'divergences.star_ee50.w1_normalized': _b(2.36477, 0.0, 0.0473),
    'divergences.star_ee80.real_median': _b(1.2307, 0.0, 0.0247),
    'divergences.star_ee80.sim_median': _b(0.656682, 0.0, 0.0132),
    'divergences.star_ee80.w1_normalized': _b(2.09637, 0.0, 0.042),
}
"""The band for every held statistic, keyed by its path in the instrument's summary."""


def held_statistics(entry: dict[str, Any]) -> dict[str, float]:
    """Every held statistic of one instrument's summary entry, keyed by its path.

    Parameters:
        entry: One instrument's entry in a realism summary.

    Returns:
        ``{'divergences.<kind>.<field>': value}`` for each kind whose support label is
        not ``unsupported`` and each held field whose value is a number, and the same
        for ``curve_divergences``.
    """
    held: dict[str, float] = {}
    for block, fields in _HELD_FIELDS.items():
        for kind, stats in sorted(entry.get(block, {}).items()):
            if stats.get('support', 'unsupported') == 'unsupported':
                continue
            for name in fields:
                value = stats.get(name)
                if value is not None:
                    held[f'{block}.{kind}.{name}'] = float(value)
    return held


def compare_with_record(
    fresh: dict[str, Any], record: dict[str, Any], bands: dict[str, RealismBand]
) -> list[str]:
    """Compare a fresh run's held statistics with the record's, within their bands.

    Parameters:
        fresh: The instrument's entry in the fresh run's summary.
        record: The instrument's entry in the committed record.
        bands: The band for each held statistic.

    Returns:
        One message per statistic that moved beyond its band, appeared, disappeared,
        or has no band; empty when every held statistic is within its band.
    """
    fresh_held = held_statistics(fresh)
    record_held = held_statistics(record)
    problems: list[str] = []
    for path in sorted(set(fresh_held) | set(record_held)):
        if path not in fresh_held:
            problems.append(
                f'{path} was {record_held[path]:.6g} in the record and is no longer held '
                '(its support dropped or its value is missing).'
            )
            continue
        if path not in record_held:
            problems.append(f'{path} is {fresh_held[path]:.6g} and the record does not hold it.')
            continue
        band = bands.get(path)
        if band is None:
            problems.append(f'{path} has no band; measure its spread and add it to REALISM_BANDS.')
            continue
        moved = fresh_held[path] - record_held[path]
        if abs(moved) > band.band:
            problems.append(
                f'{path} moved from {record_held[path]:.6g} to {fresh_held[path]:.6g}, '
                f'beyond its band of +-{band.band:.3g} (run-to-run spread '
                f'{band.spread:.3g}).'
            )
    return problems


def _run_once(out_dir: Path) -> None:
    """Match the regression cohort once and write its summary into ``out_dir``."""
    from tests.integration.sim_realism import run_realism_match
    from tests.integration.sim_realism_report import write_summary

    results = run_realism_match(instruments=[INSTRUMENT], skip_fom7=True)
    write_summary(results, results_root=out_dir)


def measure_bands(runs: int) -> dict[str, RealismBand]:
    """Match the regression cohort ``runs`` times, each in its own process, and band it.

    Parameters:
        runs: The number of runs.

    Returns:
        The band for every statistic held in all the runs.
    """
    values: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(runs):
            out_dir = Path(tmp) / f'run_{i}'
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'tests.integration.realism_regression',
                    '--run-once',
                    str(out_dir),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((out_dir / 'realism_summary.json').read_text())
            values.append(held_statistics(summary['instruments'][INSTRUMENT]))
    common = set.intersection(*(set(v) for v in values))
    bands: dict[str, RealismBand] = {}
    for path in sorted(common):
        sample = np.array([v[path] for v in values])
        value = float(np.median(sample))
        spread = float(sample.max() - sample.min())
        bands[path] = RealismBand(
            value=float(f'{value:.6g}'),
            spread=float(f'{spread:.3g}'),
            band=band_rule(spread, value),
        )
    return bands


def main() -> None:
    """CLI entry: measure the bands and print them as table rows."""
    parser = argparse.ArgumentParser(description='Measure the realism regression bands.')
    parser.add_argument('--runs', type=int, default=RUNS_MEASURED, help='runs to measure')
    parser.add_argument('--run-once', type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_once is not None:
        _run_once(args.run_once)
        return
    for path, band in measure_bands(args.runs).items():
        print(f"    '{path}': _b({band.value!r}, {band.spread!r}, {band.band!r}),")


if __name__ == '__main__':
    require_checkout_package(__file__)
    main()
