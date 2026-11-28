"""Hermetic tests for the realism band tool's bands, table, and comparison.

The comparison reads nothing but its arguments, so this module needs neither an
environment guard nor the ``integration`` marker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import util.sim_realism.sim_realism_report as report
from util.sim_realism.realism_regression import (
    RELATIVE_FLOOR,
    InstrumentBands,
    RealismBand,
    band_rule,
    check_record_against_bands,
    compare_with_record,
    load_bands,
    write_bands,
)
from util.sim_realism.sim_realism import RealismResults

_EE50 = 'divergences.star_ee50.w1_normalized'
_EE80 = 'divergences.star_ee80.w1_normalized'
_BANDS = {_EE50: RealismBand(value=2.33, spread=0.0, band=0.0466)}
_TABLE = InstrumentBands(runs=10, statistics=_BANDS)


def _entry(
    w1: float | None, *, support: str = 'limited', kind: str = 'star_ee50'
) -> dict[str, Any]:
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


def test_a_band_with_no_measured_spread_is_the_floor() -> None:
    """With no run-to-run spread, the band is the relative floor of the value."""
    assert band_rule(0.0, 2.5) == pytest.approx(RELATIVE_FLOOR * 2.5)


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
