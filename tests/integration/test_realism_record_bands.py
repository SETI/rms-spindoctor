"""The realism band table against the committed realism record.

The band table is ``tests/integration/realism_regression_bands.json``, and the record is
``tests/integration/realism_results/realism_summary.json``; the check reads nothing else,
so it runs in the plain unit tier, CI included.  For each instrument the table lists, the
record must hold exactly the table's statistics, each within its band of the table's
measured value (see :mod:`tests.integration.realism_regression`).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.integration.realism_regression import (
    RECORD_ENV,
    REMEASURE_HINT,
    check_record_against_bands,
    load_bands,
    load_record,
    made_at,
)

_BANDS = load_bands()


# The committed record predates the regression check: it was made before the record
# carried the support fields the band statistics are drawn from, so the band table
# cannot describe it.  The record and the band table are regenerated together at the end
# of the simulator realism stack, and that change removes this marker; strict, so the
# regeneration fails until it does.
_COMMITTED_RECORD_PREDATES_THE_CHECK = pytest.mark.xfail(
    RECORD_ENV not in os.environ,
    reason=(
        'the committed realism record predates the regression check; it is regenerated, '
        'with the bands, at the end of the simulator realism stack, which removes this '
        'marker'
    ),
    strict=True,
)


@pytest.fixture
def realism_record() -> dict[str, Any]:
    """The realism record the check reads: the committed one, or the one the variable names."""
    return load_record()


@_COMMITTED_RECORD_PREDATES_THE_CHECK
@pytest.mark.parametrize('instrument', sorted(_BANDS))
def test_the_band_table_describes_the_record(
    instrument: str, realism_record: dict[str, Any]
) -> None:
    """The record holds exactly the table's statistics, each within its band of the table."""
    problems = check_record_against_bands(realism_record, instrument, _BANDS[instrument])
    assert not problems, (
        f'The {instrument} band table and the record made at {made_at(realism_record)} '
        'disagree:\n' + '\n'.join(problems) + f'\n{REMEASURE_HINT}'
    )
