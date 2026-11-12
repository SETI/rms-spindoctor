"""Regression check: a fresh LORRI realism match against the committed record.

The bands, their rule, and their measurement live in
:mod:`tests.integration.realism_regression`.  The check reads real frames, so it
carries only the ``integration`` marker.

The record is ``tests/integration/realism_results/realism_summary.json``.  The
environment variable ``SPINDOCTOR_TEST_REALISM_RECORD``, read only here, points the
check at another summary, so that it can be shown to pass and fail against a record
made outside the repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.integration.realism_regression import (
    INSTRUMENT,
    REALISM_BANDS,
    REGENERATE_HINT,
    compare_with_record,
)
from tests.integration.sim_realism import run_realism_match
from tests.integration.sim_realism_report import summary_dict

pytestmark = pytest.mark.integration

# CI sets PDS3_HOLDINGS_DIR but not OOPS_RESOURCES.
_HAS_REAL_DATA = bool(os.environ.get('PDS3_HOLDINGS_DIR')) and bool(
    os.environ.get('OOPS_RESOURCES')
)

_RECORD_ENV = 'SPINDOCTOR_TEST_REALISM_RECORD'
_COMMITTED_RECORD = Path(__file__).parent / 'realism_results' / 'realism_summary.json'


@pytest.fixture
def realism_record() -> dict[str, Any]:
    """The realism record to hold the fresh run to, skipping one that has no provenance."""
    path = Path(os.environ.get(_RECORD_ENV, str(_COMMITTED_RECORD)))
    record: dict[str, Any] = json.loads(path.read_text())
    if not record.get('code_commit'):
        pytest.skip(
            f'{path} carries no code commit, so it cannot be held to bands; the check '
            'arms itself when the record is regenerated'
        )
    return record


@pytest.mark.skipif(not _HAS_REAL_DATA, reason='PDS3_HOLDINGS_DIR or OOPS_RESOURCES unset')
def test_the_lorri_realism_match_holds_to_the_record(realism_record: dict[str, Any]) -> None:
    """Every statistic the LORRI cohort supports lies within its band of the record."""
    results = run_realism_match(instruments=[INSTRUMENT], skip_fom7=True)
    fresh = summary_dict(results)['instruments'][INSTRUMENT]
    problems = compare_with_record(fresh, realism_record['instruments'][INSTRUMENT], REALISM_BANDS)
    commit = realism_record['code_commit']
    assert not problems, (
        f'The {INSTRUMENT} realism match moved from the record made at {commit}:\n'
        + '\n'.join(problems)
        + f'\n{REGENERATE_HINT}'
    )
