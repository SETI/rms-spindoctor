"""Regression check: a fresh realism match against the committed record.

The bands, their rule, and their measurement live in
:mod:`tests.integration.realism_regression`, and the band table in
``tests/integration/realism_regression_bands.json``.  Each check runs once for every
instrument the table lists.  The record-versus-table check reads only committed files;
the fresh match reads real frames, so it carries the ``integration`` marker.

The record is ``tests/integration/realism_results/realism_summary.json``.  The
environment variable ``SPINDOCTOR_TEST_REALISM_RECORD``, read only here, points the
checks at another summary, so that they can be shown to pass and fail against a record
made outside the repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.integration.realism_regression import (
    REGENERATE_HINT,
    REMEASURE_HINT,
    check_record_against_bands,
    compare_with_record,
    load_bands,
)
from tests.integration.sim_realism import run_realism_match
from tests.integration.sim_realism_report import summary_dict

# CI sets PDS3_HOLDINGS_DIR but not OOPS_RESOURCES.
_HAS_REAL_DATA = bool(os.environ.get('PDS3_HOLDINGS_DIR')) and bool(
    os.environ.get('OOPS_RESOURCES')
)

_RECORD_ENV = 'SPINDOCTOR_TEST_REALISM_RECORD'
_COMMITTED_RECORD = Path(__file__).parent / 'realism_results' / 'realism_summary.json'

_BANDS = load_bands()


@pytest.fixture
def realism_record() -> dict[str, Any]:
    """The realism record to hold the fresh run to, skipping one without a code commit."""
    path = Path(os.environ.get(_RECORD_ENV, str(_COMMITTED_RECORD)))
    record: dict[str, Any] = json.loads(path.read_text())
    if not record.get('code_commit'):
        pytest.skip(
            f'{path} carries no code commit, so it cannot be held to bands; the check '
            'arms itself when the record is regenerated'
        )
    return record


def _made_at(record: dict[str, Any]) -> str:
    """Name the code a record was made from, with its checkout's dirty flag."""
    dirty = record.get('code_dirty')
    state = 'unknown' if dirty is None else ('dirty' if dirty else 'clean')
    return f'{record["code_commit"]} (checkout {state})'


@pytest.mark.parametrize('instrument', sorted(_BANDS))
def test_the_band_table_describes_the_record(
    instrument: str, realism_record: dict[str, Any]
) -> None:
    """The record holds exactly the table's statistics, each within its band of the table."""
    problems = check_record_against_bands(realism_record, instrument, _BANDS[instrument])
    assert not problems, (
        f'The {instrument} band table and the record made at {_made_at(realism_record)} '
        'disagree:\n' + '\n'.join(problems) + f'\n{REMEASURE_HINT}'
    )


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_REAL_DATA, reason='PDS3_HOLDINGS_DIR or OOPS_RESOURCES unset')
@pytest.mark.parametrize('instrument', sorted(_BANDS))
def test_the_realism_match_holds_to_the_record(
    instrument: str, realism_record: dict[str, Any]
) -> None:
    """Every statistic the instrument supports lies within its band of the record."""
    record_entry = realism_record.get('instruments', {}).get(instrument)
    if record_entry is None:
        pytest.fail(
            f'The record made at {_made_at(realism_record)} holds no {instrument} entry, '
            f'though the band table lists it.\n{REMEASURE_HINT}'
        )
    results = run_realism_match(instruments=[instrument], skip_fom7=True)
    fresh = summary_dict(results)['instruments'][instrument]
    problems = compare_with_record(fresh, record_entry, _BANDS[instrument].statistics)
    assert not problems, (
        f'The {instrument} realism match moved from the record made at '
        f'{_made_at(realism_record)}:\n' + '\n'.join(problems) + f'\n{REGENERATE_HINT}'
    )
