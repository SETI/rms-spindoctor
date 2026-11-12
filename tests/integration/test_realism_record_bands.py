"""The realism band table against the committed realism record.

The band table is ``tests/integration/realism_regression_bands.json``, and the record is
``tests/integration/realism_results/realism_summary.json``; the check reads nothing else,
so it runs in the plain unit tier, CI included.  For each instrument the table lists, the
record must hold exactly the table's statistics, each within its band of the table's
measured value (see :mod:`tests.integration.realism_regression`).
"""

from __future__ import annotations

import pytest

from tests.integration.realism_regression import (
    REMEASURE_HINT,
    check_record_against_bands,
    load_bands,
    load_record,
    made_at,
)

_BANDS = load_bands()

# The record the check reads: the committed one, or the one SPINDOCTOR_TEST_REALISM_RECORD
# names.
_RECORD = load_record()


# A record without a code commit predates the regression check: it was made before the
# record carried the support fields the band statistics are drawn from, so the band table
# cannot describe it.  The committed record is such a record until it and the band table
# are regenerated together at the end of the simulator realism stack; the regenerated
# record carries its commit, which arms the check.  Strict, and only a failed comparison
# counts, so a crash in the check still fails.
_RECORD_PREDATES_THE_CHECK = pytest.mark.xfail(
    not _RECORD.get('code_commit'),
    reason='the realism record carries no code commit, so it predates the regression check',
    raises=AssertionError,
    strict=True,
)


@_RECORD_PREDATES_THE_CHECK
@pytest.mark.parametrize('instrument', sorted(_BANDS))
def test_the_band_table_describes_the_record(instrument: str) -> None:
    """The record holds exactly the table's statistics, each within its band of the table."""
    problems = check_record_against_bands(_RECORD, instrument, _BANDS[instrument])
    assert not problems, (
        f'The {instrument} band table and the record made at {made_at(_RECORD)} '
        'disagree:\n' + '\n'.join(problems) + f'\n{REMEASURE_HINT}'
    )
