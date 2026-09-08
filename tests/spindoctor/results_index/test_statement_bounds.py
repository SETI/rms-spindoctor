"""Every statement the index reader issues fits the smallest backend's parameter cap.

Each stub a statement names is a bind parameter, and SQLite before its 3.32
release caps a statement at 999 of them.  The bound the package batches on is
one constant, but a statement can name a batch more than once -- the named
listing answers from a union whose two arms each name every stub -- so what
has to hold is a property of the statements as issued, not of the constant.
These listen to the statements themselves.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.conftest import index_url, ingest_tree, metadata_document, write_metadata

from spindoctor.nav_records import Selection
from spindoctor.results_index import STUBS_PER_STATEMENT, open_record_source

_SMALLEST_CAP = 999
"""SQLite's own default cap on host parameters before 3.32, the smallest supported."""


@pytest.fixture
def parameter_counts() -> Iterator[list[int]]:
    """Count the bind parameters of every statement any engine issues.

    Yields:
        The list each statement's parameter count is appended to as it runs.
    """
    counts: list[int] = []

    def counting(
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        counts.append(len(parameters))

    sqlalchemy.event.listen(sqlalchemy.engine.Engine, 'before_cursor_execute', counting)
    try:
        yield counts
    finally:
        sqlalchemy.event.remove(sqlalchemy.engine.Engine, 'before_cursor_execute', counting)


def _index_over_one_document(tmp_path: Path) -> tuple[Path, str]:
    """Ingest a one-document tree and return its root and index URL.

    Parameters:
        tmp_path: Directory the tree and the index are written under.

    Returns:
        The results root and the index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=pdslogger.NullLogger())
    return root, url


_MANY_STUBS = tuple(
    f'VOL/N{1454725799 + index}_1_CALIB' for index in range(2 * STUBS_PER_STATEMENT + 1)
)
"""More names than two statements' worth, so every path batches more than once."""


@pytest.mark.parametrize('question', ['listing', 'records', 'facts'])
def test_a_question_about_many_named_stubs_fits_the_smallest_cap(
    tmp_path: Path, parameter_counts: list[int], question: str
) -> None:
    """No statement names more parameters than the smallest backend will carry.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
        parameter_counts: What every statement issued bound.
        question: Which of the three questions to ask about the named stubs.
    """
    root, url = _index_over_one_document(tmp_path)
    with open_record_source([root], results_index_db_url=url) as source:
        list(getattr(source, question)(Selection(stubs=_MANY_STUBS)))
    assert max(parameter_counts) <= _SMALLEST_CAP


def test_the_named_listing_really_is_batched(tmp_path: Path, parameter_counts: list[int]) -> None:
    """Without which the bound above could hold by the listing binding nothing at all."""
    root, url = _index_over_one_document(tmp_path)
    with open_record_source([root], results_index_db_url=url) as source:
        list(source.listing(Selection(stubs=_MANY_STUBS)))
    assert max(parameter_counts) > STUBS_PER_STATEMENT // 2
