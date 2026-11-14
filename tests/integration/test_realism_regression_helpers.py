"""Hermetic tests for the realism regression check's bands, table, and comparison.

The comparison reads nothing but its arguments, so this module needs neither an
environment guard nor the ``integration`` marker, and it runs in the plain unit tier.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tests.integration.realism_regression as realism_regression
import tests.integration.sim_realism_report as report
from tests.integration.realism_regression import (
    RELATIVE_FLOOR,
    InstrumentBands,
    RealismBand,
    band_rule,
    check_record_against_bands,
    compare_with_record,
    held_statistics,
    load_bands,
    measure_bands,
    write_bands,
)
from tests.integration.sim_realism import RealismResults

_EE50 = 'divergences.star_ee50.w1_normalized'
_EE80 = 'divergences.star_ee80.w1_normalized'
_BANDS = {_EE50: RealismBand(value=2.33, spread=0.0, band=0.0466)}
_TABLE = InstrumentBands(runs=10, statistics=_BANDS)
_COMMITTED_BANDS = load_bands()


def _entry(w1: float, *, support: str = 'limited', kind: str = 'star_ee50') -> dict[str, Any]:
    """An instrument summary entry holding one star divergence."""
    return {
        'divergences': {
            kind: {
                'w1_normalized': w1,
                'real_median': None,
                'sim_median': None,
                'support': support,
            }
        }
    }


def test_a_move_beyond_the_band_names_the_statistic_and_both_values() -> None:
    """A better-looking value beyond the band fails: the bands are two-sided."""
    problems = compare_with_record(_entry(1.04), _entry(2.33), _BANDS)
    assert problems == [
        f'{_EE50} moved from 2.33 to 1.04, beyond its band of +-0.0466 (run-to-run spread 0).'
    ]


def test_a_move_within_the_band_passes() -> None:
    """A value within its band of the record raises nothing."""
    assert compare_with_record(_entry(2.36), _entry(2.33), _BANDS) == []


def test_a_held_statistic_without_a_band_fails() -> None:
    """A statistic without a band in the table is reported, never passed."""
    problems = compare_with_record(_entry(2.33), _entry(2.33), {})
    assert problems == [f'{_EE50} has no band in the band table.']


def test_an_unsupported_statistic_is_not_held() -> None:
    """A statistic whose support label is unsupported is left out of the check."""
    assert held_statistics(_entry(2.33, support='unsupported')) == {}


def test_a_band_with_no_measured_spread_is_the_floor() -> None:
    """With no run-to-run spread, the band is the relative floor of the value."""
    assert band_rule(0.0, 2.5) == pytest.approx(RELATIVE_FLOOR * 2.5)


@pytest.mark.parametrize(
    ('instrument', 'path'),
    [(inst, path) for inst, entry in sorted(_COMMITTED_BANDS.items()) for path in entry.statistics],
)
def test_every_band_follows_the_rule(instrument: str, path: str) -> None:
    """Each committed band is the rule applied to its measured spread and value."""
    band = _COMMITTED_BANDS[instrument].statistics[path]
    assert band.band == band_rule(band.spread, band.value)


def test_a_record_without_the_instrument_fails_by_name() -> None:
    """A record lacking an instrument the table lists says so, rather than raising."""
    problems = check_record_against_bands({'instruments': {}}, 'nhlorri', _TABLE)
    assert problems == ['The record holds no nhlorri entry, though the band table lists it.']


def test_a_table_entry_without_statistics_fails() -> None:
    """An empty table entry fails, because a check over it would hold nothing."""
    table = InstrumentBands(runs=10, statistics={})
    record = {'instruments': {'nhlorri': _entry(2.33)}}
    problems = check_record_against_bands(record, 'nhlorri', table)
    assert problems == ['The band table holds no statistics for nhlorri.']


@pytest.mark.parametrize(
    ('entry', 'expected'),
    [
        (
            {
                'divergences': {
                    **_entry(2.33)['divergences'],
                    **_entry(2.1, kind='star_ee80')['divergences'],
                }
            },
            f'{_EE80} is held in the record and has no band.',
        ),
        (_entry(2.33, support='unsupported'), f'{_EE50} has a band and is not held in the record.'),
        (
            _entry(2.5),
            f'{_EE50} is 2.5 in the record and 2.33 in the band table, beyond its band of '
            '+-0.0466.',
        ),
    ],
    ids=['held-without-a-band', 'band-without-a-held-value', 'value-off-the-table'],
)
def test_a_record_the_table_does_not_describe_fails(entry: dict[str, Any], expected: str) -> None:
    """The record's held statistics are the table's, each within its band of the table."""
    record = {'instruments': {'nhlorri': entry}}
    assert check_record_against_bands(record, 'nhlorri', _TABLE) == [expected]


def test_the_band_table_reads_back_what_was_written(tmp_path: Path) -> None:
    """A table the measurement writes is the table the check reads."""
    path = tmp_path / 'bands.json'
    write_bands({'nhlorri': _TABLE}, path)
    assert load_bands(path) == {'nhlorri': _TABLE}


def _measure_two_runs(
    monkeypatch: pytest.MonkeyPatch, first: float, second: float
) -> InstrumentBands:
    """Measure LORRI's bands from two faked runs whose star EE50 W1/IQR are the values given.

    Parameters:
        monkeypatch: The test's monkeypatch fixture, which fakes each run's process.
        first: The statistic in the first run.
        second: The statistic in the second run.
    """
    runs = iter([first, second])

    def fake_run(args: list[str], **kwargs: Any) -> None:
        out_dir = Path(args[-1])
        out_dir.mkdir(parents=True)
        summary = {'instruments': {'nhlorri': _entry(next(runs))}}
        (out_dir / 'realism_summary.json').write_text(json.dumps(summary))

    monkeypatch.setattr(subprocess, 'run', fake_run)
    return measure_bands('nhlorri', 2)


def test_a_measured_band_table_reads_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A spread of 0.01236 is stored as 0.0124, and its band follows the stored spread.

    Three times 0.01236 rounds up to a band of 0.0371, narrower than the 0.0372 the rule
    gives the stored spread, so a band taken before rounding is refused on reading.
    """
    table = _measure_two_runs(monkeypatch, 1.0, 1.01236)
    path = tmp_path / 'bands.json'
    write_bands({'nhlorri': table}, path)
    assert load_bands(path) == {'nhlorri': table}


@pytest.mark.parametrize(
    ('first', 'second', 'stored'),
    [(1.0, 1.012349, 0.0124), (0.0, 0.0124, 0.0124), (1.0, 1.0, 0.0)],
    ids=['rounded-up', 'three-digits-kept', 'zero'],
)
def test_a_measured_spread_is_rounded_up_to_three_digits(
    monkeypatch: pytest.MonkeyPatch, first: float, second: float, stored: float
) -> None:
    """A spread is stored at three digits rounded up, so its band covers 3 times the spread.

    Rounded to the nearest, 0.012349 would be stored as 0.0123, whose band of 0.0369 is
    narrower than 3 times the spread measured.  A spread of exactly 0.0124 is kept, as
    is a spread of 0.
    """
    table = _measure_two_runs(monkeypatch, first, second)
    assert table.statistics[_EE50].spread == stored


def test_measuring_bands_needs_two_runs() -> None:
    """A spread can't be measured from fewer than two runs."""
    with pytest.raises(ValueError, match='needs at least 2 runs; got 1'):
        measure_bands('nhlorri', 1)


def test_the_code_provenance_names_a_full_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """In a checkout, the provenance names a full commit hash, SHA-1 or SHA-256."""
    monkeypatch.delenv('GIT_DIR', raising=False)
    monkeypatch.delenv('GIT_WORK_TREE', raising=False)
    provenance = report.code_provenance()
    assert provenance is not None
    assert re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', provenance.commit)


def test_the_provenance_is_refused_without_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a git checkout, the provenance is refused before any match runs."""
    monkeypatch.delenv('GIT_DIR', raising=False)
    monkeypatch.delenv('GIT_WORK_TREE', raising=False)
    monkeypatch.setattr(report, '_REPO_ROOT', tmp_path)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    with pytest.raises(RuntimeError, match='Cannot determine the code commit'):
        report.require_code_provenance()


def test_a_summary_records_the_provenance_it_is_given(tmp_path: Path) -> None:
    """The summary records the provenance taken before the run, never a later status.

    The match writes tracked figures, so a status taken while writing the summary would
    report them as changes.
    """
    provenance = report.CodeProvenance(commit='a' * 40, dirty=False)
    results = RealismResults(comparisons={}, runtime_sec=0.0)
    path = report.write_summary(results, provenance, results_root=tmp_path)
    payload = json.loads(path.read_text())
    assert payload['code_commit'] == 'a' * 40
    assert payload['code_dirty'] is False


_MATCH_OUTPUTS = (
    'tests/integration/realism_results/realism_summary.json',
    'docs/simulator_report/_figures/realism_nhlorri_noise.png',
)


def _checkout_after_changing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracked: tuple[str, ...],
    changed: tuple[str, ...],
) -> report.CodeProvenance | None:
    """The provenance of a fresh checkout of ``tracked`` once ``changed`` are rewritten.

    Parameters:
        tmp_path: The directory the checkout is made in.
        monkeypatch: Points the provenance at the checkout.
        tracked: Repository-relative paths committed in the checkout.
        changed: The tracked paths rewritten after the commit.
    """
    monkeypatch.delenv('GIT_DIR', raising=False)
    monkeypatch.delenv('GIT_WORK_TREE', raising=False)
    for name in tracked:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('first run\n')
    git = ['git', '-C', str(tmp_path), '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', 'files'], check=True)
    for name in changed:
        (tmp_path / name).write_text('second run\n')
    monkeypatch.setattr(report, '_REPO_ROOT', tmp_path)
    return report.code_provenance()


def test_the_match_outputs_leave_the_checkout_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncommitted figures and summary from an earlier run don't make the checkout dirty."""
    provenance = _checkout_after_changing(tmp_path, monkeypatch, _MATCH_OUTPUTS, _MATCH_OUTPUTS)
    assert provenance is not None
    assert provenance.dirty is False


@pytest.mark.parametrize(
    'changed',
    ['src/spindoctor/sim/realism/noise.py', 'docs/simulator_report/_figures/mismatch_axes.png'],
    ids=['code', 'another-tools-figure'],
)
def test_a_change_beside_the_match_outputs_makes_the_checkout_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """A tracked file the match does not write makes the checkout dirty when it changes.

    That holds for another tool's figure in the directory the match writes its own
    figures to.
    """
    provenance = _checkout_after_changing(
        tmp_path, monkeypatch, (*_MATCH_OUTPUTS, changed), (changed,)
    )
    assert provenance is not None
    assert provenance.dirty is True


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": NaN, "spread": 0, "band": 0}}}}',
            'the band table holds NaN',
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": 1, "spread": 0, "band": Infinity}}}}',
            'the band table holds Infinity',
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": 1e999, "spread": 0, "band": 0}}}}',
            "nhlorri s 'value' is inf, which is not finite",
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": 1, "spread": -0.1, "band": 0}}}}',
            "nhlorri s 'spread' is -0.1, which is negative",
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": 1, "spread": 0, "band": "0.1"}}}}',
            "nhlorri s 'band' is '0.1', which is not a number",
        ),
        (
            '{"nhlorri": {"runs": 1, "statistics": {}}}',
            'nhlorri: runs is 1; a spread needs a whole number of at least 2 runs',
        ),
        ('{"nhlorri": {"runs": 10}}', "nhlorri lacks 'statistics'"),
        (
            '{"nhlorri": {"runs": 10, "statistics": {}}}',
            'nhlorri holds no statistics, so it checks nothing',
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": {"value": 1, "band": 0.02}}}}',
            "nhlorri s lacks 'spread'",
        ),
        (
            '{"nhlorri": {"runs": 10, "statistics": {"s": '
            '{"value": 2.5, "spread": 0, "band": 0.04}}}}',
            "nhlorri s 'band' is 0.04, narrower than the rule's 0.05 for its spread and value",
        ),
    ],
    ids=[
        'nan',
        'infinity',
        'overflow',
        'negative-spread',
        'string-band',
        'one-run',
        'missing-statistics',
        'empty-statistics',
        'missing-spread',
        'band-below-the-rule',
    ],
)
def test_a_band_table_that_breaks_the_rules_is_refused(
    tmp_path: Path, text: str, message: str
) -> None:
    """A table holding a number no band can use is refused with a message naming it."""
    path = tmp_path / 'bands.json'
    path.write_text(text)
    with pytest.raises(ValueError, match=re.escape(message)):
        load_bands(path)


def test_a_band_table_listing_no_instruments_is_refused(tmp_path: Path) -> None:
    """An empty table is refused by its path: every check over it would collect nothing."""
    path = tmp_path / 'bands.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match=re.escape(f'the band table {path} lists no instruments')):
        load_bands(path)


def _run_band_cli(monkeypatch: pytest.MonkeyPatch, path: Path, instruments: list[str]) -> None:
    """Run the band CLI on the table at ``path``, measuring every instrument as ``_TABLE``.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.
        path: The band table the CLI reads and writes.
        instruments: The instruments named on the command line.
    """
    monkeypatch.setattr(realism_regression, 'BANDS_PATH', path)
    monkeypatch.setattr(realism_regression, 'measure_bands', lambda instrument, runs: _TABLE)
    monkeypatch.setattr(sys, 'argv', ['realism_regression', *instruments])
    realism_regression.main()


def test_the_band_cli_fills_an_empty_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A table that lists no instruments is the starting point for the instruments named."""
    path = tmp_path / 'bands.json'
    path.write_text('{}')
    _run_band_cli(monkeypatch, path, ['nhlorri'])
    assert load_bands(path) == {'nhlorri': _TABLE}


def test_the_band_cli_needs_instruments_for_an_empty_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With an empty table and no instruments named, the CLI says to name them."""
    path = tmp_path / 'bands.json'
    path.write_text('{}')
    with pytest.raises(SystemExit, match='2'):
        _run_band_cli(monkeypatch, path, [])
    assert 'name the instruments to measure; the band table lists none' in capsys.readouterr().err


def test_a_negative_statistic_value_is_read(tmp_path: Path) -> None:
    """A statistic's value may be negative, as the sky level less the floor can be."""
    table = InstrumentBands(
        runs=10, statistics={_EE50: RealismBand(value=-0.5, spread=0.0, band=0.01)}
    )
    path = tmp_path / 'bands.json'
    write_bands({'nhlorri': table}, path)
    assert load_bands(path) == {'nhlorri': table}
