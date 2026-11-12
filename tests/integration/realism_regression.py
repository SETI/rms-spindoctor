"""Regression bands for the realism match: what a fresh run is held to against the record.

The realism match (:mod:`tests.integration.sim_realism`) writes its summary to
``tests/integration/realism_results/realism_summary.json``, together with the commit of
the code that made it and whether that checkout was dirty.  The regression check reruns
the match on each instrument listed in the band table, an instrument whose cohort
matches in seconds, and compares every statistic the instrument supports with the
record's value.  A statistic is held when its support label, which counts the frames on
both sides, is ``limited`` or ``supported``.  The held values are each kind's W1/IQR
(when it is a number), the real and simulated medians, and each curve kind's W1/IQR.

Every band is two-sided.  A statistic that moves beyond its band fails whichever way it
moves, because a better-looking number can come from a worse simulator.

The band table is :data:`BANDS_PATH`, a JSON file keyed by instrument.  Each entry gives
the number of runs measured and, for every held statistic, its measured value, its
run-to-run spread, and its band.  Navigation is not bit-for-bit repeatable, so each band
rests on that spread: the instrument was matched ``runs`` times on unchanged code, each
run in its own process, and the spread is the largest value minus the smallest.  A
statistic's band is a half-width around the recorded value: max(:data:`SPREAD_MULTIPLE`
times the spread, :data:`RELATIVE_FLOOR` times the magnitude of the value measured).
The floor covers the arithmetic differences between machines, which a spread measured
on one machine cannot show.  :func:`band_rule` is the rule in code.

The table and the record must describe the same run: the statistics the record holds
for an instrument are exactly the table's, and each recorded value lies within its band
of the table's measured value.  A record that carries no code commit predates the check
and is not held to bands.  A deliberate change to the simulator, or to the statistics,
is accepted by regenerating the record and then remeasuring the bands::

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
    'BANDS_PATH',
    'MIN_RUNS',
    'RELATIVE_FLOOR',
    'SPREAD_MULTIPLE',
    'InstrumentBands',
    'RealismBand',
    'band_rule',
    'check_record_against_bands',
    'compare_with_record',
    'held_statistics',
    'load_bands',
    'measure_bands',
    'write_bands',
]

BANDS_PATH = Path(__file__).parent / 'realism_regression_bands.json'
"""The committed band table, keyed by instrument."""

DEFAULT_RUNS = 10
"""How many runs the CLI measures each spread over unless told otherwise."""

MIN_RUNS = 2
"""The fewest runs a spread can be measured over."""

SPREAD_MULTIPLE = 3.0
"""A band is at least this many times its statistic's measured spread."""

RELATIVE_FLOOR = 0.02
"""A band is at least this fraction of its statistic's measured magnitude."""

REGENERATE_HINT = (
    'If the change is deliberate, accept it by regenerating the realism record '
    '(python -m tests.integration.sim_realism) and remeasuring the bands '
    '(python -m tests.integration.realism_regression).'
)

REMEASURE_HINT = (
    'The band table no longer describes the record: remeasure the bands on the code '
    'that made the record (python -m tests.integration.realism_regression).'
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


@dataclass(frozen=True)
class InstrumentBands:
    """One instrument's entry in the band table.

    Parameters:
        runs: How many runs on unchanged code the spreads were measured over.
        statistics: The band of every held statistic, keyed by its path in the
            instrument's summary entry.
    """

    runs: int
    statistics: dict[str, RealismBand]


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


def load_bands(path: Path = BANDS_PATH) -> dict[str, InstrumentBands]:
    """Read the band table.

    Parameters:
        path: The table's path.

    Returns:
        Each instrument's bands, keyed by instrument.
    """
    raw: dict[str, Any] = json.loads(path.read_text())
    return {
        instrument: InstrumentBands(
            runs=int(entry['runs']),
            statistics={
                stat: RealismBand(
                    value=float(row['value']),
                    spread=float(row['spread']),
                    band=float(row['band']),
                )
                for stat, row in entry['statistics'].items()
            },
        )
        for instrument, entry in raw.items()
    }


def write_bands(bands: dict[str, InstrumentBands], path: Path = BANDS_PATH) -> None:
    """Write the band table, sorted by instrument and by statistic.

    Parameters:
        bands: Each instrument's bands, keyed by instrument.
        path: The table's path.
    """
    payload = {
        instrument: {
            'runs': entry.runs,
            'statistics': {
                stat: {'band': band.band, 'spread': band.spread, 'value': band.value}
                for stat, band in entry.statistics.items()
            },
        }
        for instrument, entry in bands.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


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


def check_record_against_bands(
    record: dict[str, Any], instrument: str, bands: InstrumentBands
) -> list[str]:
    """Check that the band table describes the record for one instrument.

    The record must hold an entry for the instrument, the statistics it holds there
    must be exactly the table's, and each recorded value must lie within its band of
    the table's measured value.

    Parameters:
        record: The whole realism record.
        instrument: The instrument to check.
        bands: The instrument's entry in the band table.

    Returns:
        One message per mismatch; empty when the table describes the record.
    """
    entry = record.get('instruments', {}).get(instrument)
    if entry is None:
        return [f'The record holds no {instrument} entry, though the band table lists it.']
    if not bands.statistics:
        return [f'The band table holds no statistics for {instrument}.']
    held = held_statistics(entry)
    problems: list[str] = []
    for path in sorted(set(held) - set(bands.statistics)):
        problems.append(f'{path} is held in the record and has no band.')
    for path in sorted(set(bands.statistics) - set(held)):
        problems.append(f'{path} has a band and is not held in the record.')
    for path in sorted(set(held) & set(bands.statistics)):
        band = bands.statistics[path]
        if abs(held[path] - band.value) > band.band:
            problems.append(
                f'{path} is {held[path]:.6g} in the record and {band.value:.6g} in the '
                f'band table, beyond its band of +-{band.band:.3g}.'
            )
    return problems


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
            problems.append(f'{path} has no band in the band table.')
            continue
        moved = fresh_held[path] - record_held[path]
        if abs(moved) > band.band:
            problems.append(
                f'{path} moved from {record_held[path]:.6g} to {fresh_held[path]:.6g}, '
                f'beyond its band of +-{band.band:.3g} (run-to-run spread '
                f'{band.spread:.3g}).'
            )
    return problems


def _run_once(instrument: str, out_dir: Path) -> None:
    """Match one instrument once and write its summary into ``out_dir``."""
    from tests.integration.sim_realism import run_realism_match
    from tests.integration.sim_realism_report import require_code_provenance, write_summary

    provenance = require_code_provenance()
    results = run_realism_match(instruments=[instrument], skip_fom7=True)
    write_summary(results, provenance, results_root=out_dir)


def measure_bands(instrument: str, runs: int) -> InstrumentBands:
    """Match one instrument ``runs`` times, each in its own process, and band it.

    Parameters:
        instrument: The instrument to match.
        runs: The number of runs, at least :data:`MIN_RUNS`.

    Returns:
        The band for every statistic held in all the runs.

    Raises:
        ValueError: When ``runs`` is below :data:`MIN_RUNS`, or when no statistic is
            held in every run.
    """
    if runs < MIN_RUNS:
        raise ValueError(
            f'Measuring a run-to-run spread needs at least {MIN_RUNS} runs; got {runs}.'
        )
    values: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(runs):
            out_dir = Path(tmp) / f'run_{i}'
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'tests.integration.realism_regression',
                    instrument,
                    '--run-once',
                    str(out_dir),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((out_dir / 'realism_summary.json').read_text())
            values.append(held_statistics(summary['instruments'][instrument]))
    common = set.intersection(*(set(v) for v in values))
    if not common:
        raise ValueError(f'No statistic of {instrument} was held in all {runs} runs.')
    statistics: dict[str, RealismBand] = {}
    for path in sorted(common):
        sample = np.array([v[path] for v in values])
        value = float(np.median(sample))
        spread = float(sample.max() - sample.min())
        statistics[path] = RealismBand(
            value=float(f'{value:.6g}'),
            spread=float(f'{spread:.3g}'),
            band=band_rule(spread, value),
        )
    return InstrumentBands(runs=runs, statistics=statistics)


def main() -> None:
    """CLI entry: measure the bands of the named instruments and write the band table."""
    parser = argparse.ArgumentParser(description='Measure the realism regression bands.')
    parser.add_argument(
        'instruments',
        nargs='*',
        help='instruments to measure (default: every instrument the band table lists)',
    )
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS, help='runs to measure')
    parser.add_argument('--run-once', type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_once is not None:
        if len(args.instruments) != 1:
            parser.error('--run-once takes exactly one instrument')
        _run_once(args.instruments[0], args.run_once)
        return
    bands = load_bands() if BANDS_PATH.exists() else {}
    instruments = args.instruments or sorted(bands)
    if not instruments:
        parser.error('name the instruments to measure; the band table lists none')
    for instrument in instruments:
        bands[instrument] = measure_bands(instrument, args.runs)
        print(f'{instrument}: {len(bands[instrument].statistics)} statistics', flush=True)
    write_bands(dict(sorted(bands.items())))
    print(f'Wrote {BANDS_PATH}')


if __name__ == '__main__':
    require_checkout_package(__file__)
    main()
