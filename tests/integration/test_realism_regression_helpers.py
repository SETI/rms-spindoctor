"""Hermetic tests for the realism regression check's bands, table, and comparison.

The comparison reads nothing but its arguments, so this module carries no
environment guard and no ``integration`` marker and runs in the plain unit tier.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

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
    """A statistic the table has no band for is reported, never passed."""
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
    """A table entry that holds nothing can't pass: the check would hold nothing."""
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


def test_measuring_bands_needs_two_runs() -> None:
    """A spread can't be measured from fewer than two runs."""
    with pytest.raises(ValueError, match='needs at least 2 runs; got 1'):
        measure_bands('nhlorri', 1)


def test_the_code_provenance_names_a_full_commit() -> None:
    """In a checkout, the provenance names a full commit hash."""
    provenance = report.code_provenance()
    assert provenance is not None
    assert re.fullmatch(r'[0-9a-f]{40}', provenance.commit)


def test_a_summary_is_not_written_without_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a git checkout, the summary is refused rather than written without a commit."""
    monkeypatch.setattr(report, '_REPO_ROOT', tmp_path)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    results = RealismResults(comparisons={}, runtime_sec=0.0)
    with pytest.raises(RuntimeError, match='Cannot determine the code commit'):
        report.write_summary(results, results_root=tmp_path / 'out')
