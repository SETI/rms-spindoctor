"""Regression check: a fresh realism match against the committed realism record.

The bands, their rule, and their measurement live in
:mod:`tests.integration.realism_regression`, and the band table in
``tests/integration/realism_regression_bands.json``.  The check runs once for every
instrument the table lists.  It reads real frames, so it carries the ``integration``
marker and stays out of CI.  The band table's agreement with the record is checked in
``tests/integration/test_realism_record_bands.py``, which reads only committed files.

The committed record predates the check, and the fresh match skips on a record that
lacks a code commit.  The environment variable ``SPINDOCTOR_TEST_REALISM_RECORD``
points the check at another summary, so that it can be shown to pass and fail against a
record made outside the repository.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.integration.realism_regression import (
    REGENERATE_HINT,
    REMEASURE_HINT,
    compare_with_record,
    load_bands,
    load_record,
    made_at,
)
from tests.integration.sim_realism import run_realism_match
from tests.integration.sim_realism_report import summary_dict

# CI sets PDS3_HOLDINGS_DIR but not OOPS_RESOURCES.
_HAS_REAL_DATA = bool(os.environ.get('PDS3_HOLDINGS_DIR')) and bool(
    os.environ.get('OOPS_RESOURCES')
)

_BANDS = load_bands()


@pytest.fixture
def realism_record() -> dict[str, Any]:
    """The realism record the check reads, as :func:`load_record` finds it."""
    return load_record()


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
            f'The record made at {made_at(realism_record)} holds no {instrument} entry, '
            f'though the band table lists it.\n{REMEASURE_HINT}'
        )
    results = run_realism_match(instruments=[instrument], skip_fom7=True)
    fresh = summary_dict(results)['instruments'][instrument]
    problems = compare_with_record(fresh, record_entry, _BANDS[instrument].statistics)
    assert not problems, (
        f'The {instrument} realism match moved from the record made at '
        f'{made_at(realism_record)}:\n' + '\n'.join(problems) + f'\n{REGENERATE_HINT}'
    )
