"""Regression check: a fresh realism match against the committed record.

The bands, their rule, and their measurement live in
:mod:`tests.integration.realism_regression`, and the band table in
``tests/integration/realism_regression_bands.json``.  Each check runs once for every
instrument the table lists.  The record-versus-table check reads only committed files;
the fresh match reads real frames, so it carries the ``integration`` marker.

The committed record predates the check, so the record-versus-table check is expected to
fail on it until the record and the bands are regenerated together, and the fresh match
skips on a record that carries no code commit.

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


# The committed record predates the regression check: it was made before the record
# carried the support fields the band statistics are drawn from, so the band table
# cannot describe it.  The record and the band table are regenerated together at the end
# of the simulator realism stack, and that change removes this marker; strict, so the
# regeneration fails until it does.
_COMMITTED_RECORD_PREDATES_THE_CHECK = pytest.mark.xfail(
    _RECORD_ENV not in os.environ,
    reason=(
        'the committed realism record predates the regression check; it is regenerated, '
        'with the bands, at the end of the simulator realism stack, which removes this '
        'marker'
    ),
    strict=True,
)


@pytest.fixture
def realism_record() -> dict[str, Any]:
    """The realism record the checks read: the committed one, or the one the variable names."""
    path = Path(os.environ.get(_RECORD_ENV, str(_COMMITTED_RECORD)))
    record: dict[str, Any] = json.loads(path.read_text())
    return record


def _made_at(record: dict[str, Any]) -> str:
    """Name the code a record was made from, with its checkout's dirty flag."""
    dirty = record.get('code_dirty')
    state = 'unknown' if dirty is None else ('dirty' if dirty else 'clean')
    return f'{record.get("code_commit") or "an unrecorded commit"} (checkout {state})'


@_COMMITTED_RECORD_PREDATES_THE_CHECK
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
    if not realism_record.get('code_commit'):
        pytest.skip(
            'the realism record carries no code commit, so it predates the regression '
            'check and cannot be held to bands'
        )
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
