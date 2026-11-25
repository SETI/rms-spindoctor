"""The merge of the three streams, against the order a backend hands rows back in.

An image and its child rows are read as three separate statements and paired up
in one pass, which can only be right while all three arrive in one order.  The
index the tests here build is written so that insertion order pairs every image
with the wrong one: a merge reading the streams off against each other by
position hands each image another image's children.

That the statements ask for that order is pinned by reading the statements
themselves rather than the source, because a backend is free to answer a join off
a child table's own index and hand the rows back in key order whether or not the
statement said to sort them.  And a child row belonging to no image the stream
yields has to fail the read: a merge that waits for one to the end of the pass
gives every image after it an empty list indistinguishable from an image with no
rows of its own.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import index_url
from tests.spindoctor.results_index.conftest import (
    COLUMNS,
    ERROR_STUB,
    FAILURE_STUB,
    FIRST_VALUES,
    MISSION,
    SUCCESS_STUB,
    UNLOADED_STUB,
    TwoRoots,
    child_name,
    facts_from_index,
    feature_source_row,
    image_row,
    stub_of,
    technique_row,
)

# The submodule is imported under its own qualified name, and the one test that
# replaces a statement inside it patches through that name.  Reading it off the
# package instead would bind whatever attribute the package holds, and a package
# free to re-export a function of the same name would hand the patch a function
# object rather than the module whose attribute the merge reads.
import spindoctor.results_index.facts_stream
from spindoctor.nav_records import ImageFacts, Selection, normalize_root_url
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    TECHNIQUES,
    IndexRecordSource,
    open_index,
    open_record_source,
)

REVERSED_STUBS = ('VOL/B_CALIB', 'VOL/A_CALIB')
"""Two images, written in the order a sorted read does not return them in."""


class ReversedIndex:
    """An index whose child rows are written in the opposite order to its images.

    What this shape shows is that the merge pairs a child row with its image by
    the key it carries: written back to front, insertion order pairs every image
    with the wrong one, so a merge reading the two streams off against each
    other by position hands each image another image's children.

    What it does not show is that the statements ask for an order.  The child
    read joins to ``images`` on the key, and a backend is free to answer that
    join off the child table's own unique index and hand the rows back in key
    order whether or not the statement said to sort them, which is what SQLite
    does here.  That the statements ask is pinned by reading the statements
    themselves, and what it costs when they do not is pinned on the tier that
    runs against a planner free to choose otherwise.

    Parameters:
        first: The root the tests read.
        second: A second root holding the same stubs with other children.
        url: The index holding both.
    """

    def __init__(self, first: Path, second: Path, url: str) -> None:
        self.first = first
        self.second = second
        self.url = url


@pytest.fixture
def reversed_index(tmp_path: Path) -> ReversedIndex:
    """Write an index whose rows defeat every order but the one the merge asks for.

    Parameters:
        tmp_path: Directory the index is written under.

    Returns:
        The two roots and the index holding both.
    """
    first = tmp_path / 'reversed-first'
    second = tmp_path / 'reversed-second'
    url = index_url(tmp_path / 'reversed.sqlite3')
    keys = [(normalize_root_url(root), stub) for root in (second, first) for stub in REVERSED_STUBS]
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                [
                    image_row(root_url=root_url, results_path_stub=stub, subtree='VOL')
                    for root_url, stub in keys
                ],
            )
            # The children go in back to front, so insertion order pairs every
            # image with the wrong one and only the sort puts them right.
            connection.execute(
                TECHNIQUES.insert(),
                [
                    technique_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        technique_name=child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
            connection.execute(
                FEATURE_SOURCES.insert(),
                [
                    feature_source_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        source_name=child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
    finally:
        engine.dispose()
    return ReversedIndex(first, second, url)


def _merged_children(held: ReversedIndex, root: Path, pick: str) -> dict[str, list[str]]:
    """Return the child names the merge gave each image of one root.

    Parameters:
        held: The index whose rows are written back to front.
        root: The root to read.
        pick: ``'techniques'`` or ``'feature_sources'``.

    Returns:
        The names, by stub.
    """
    engine = open_index(held.url)
    with IndexRecordSource(engine, [root], held.url, COLUMNS) as source:
        found = [one for one in source.facts(Selection()) if isinstance(one, ImageFacts)]
    if pick == 'techniques':
        return {
            stub_of(one): [str(row['technique_name']) for row in one.techniques] for one in found
        }
    return {stub_of(one): [str(row['source_name']) for row in one.feature_sources] for one in found}


def test_the_merge_gives_each_image_its_own_technique_rows(
    reversed_index: ReversedIndex,
) -> None:
    """Written back to front, so a positional merge pairs them all wrongly.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.first)
    assert _merged_children(reversed_index, reversed_index.first, 'techniques') == {
        stub: [child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def test_the_merge_gives_each_image_its_own_feature_source_rows(
    reversed_index: ReversedIndex,
) -> None:
    """The other child table, merged by the same rule.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.first)
    assert _merged_children(reversed_index, reversed_index.first, 'feature_sources') == {
        stub: [child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def test_the_merge_reads_the_selected_roots_children(reversed_index: ReversedIndex) -> None:
    """The other root holds the same stubs, with children named for itself.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    root_url = normalize_root_url(reversed_index.second)
    assert _merged_children(reversed_index, reversed_index.second, 'techniques') == {
        stub: [child_name(root_url, stub)] for stub in REVERSED_STUBS
    }


def _unordered_keys(url: str, table: sqlalchemy.Table) -> list[tuple[Any, Any]]:
    """Return the keys of one table in the order the server hands them back unasked.

    Every column is selected, because that is what the merge selects and a
    narrower statement is answered off an index whose order is the key's rather
    than the table's own.

    Parameters:
        url: The index to read.
        table: The table to read.

    Returns:
        One key per row, in the order they arrived.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return [
                (row.root_url, row.results_path_stub)
                for row in connection.execute(sqlalchemy.select(table))
            ]
    finally:
        engine.dispose()


def test_the_rows_really_do_arrive_in_two_different_orders(
    reversed_index: ReversedIndex,
) -> None:
    """Without which the three tests above hold whatever the merge does.

    Parameters:
        reversed_index: The index whose child rows are written in reverse.
    """
    images = _unordered_keys(reversed_index.url, IMAGES)
    techniques = _unordered_keys(reversed_index.url, TECHNIQUES)
    assert techniques != images


def _issued_by(url: str, root: Path, selection: Selection) -> list[str]:
    """Return the statements a stream of facts sent to the server.

    Parameters:
        url: The index to read.
        root: The root to read.
        selection: What to read.

    Returns:
        The SQL of each statement, in the order it was issued.
    """
    issued: list[str] = []
    engine = open_index(url)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: issued.append(statement),
    )
    with IndexRecordSource(engine, [root], url, COLUMNS) as source:
        list(source.facts(selection))
    return issued


def _ordering_of(statement: str) -> str:
    """Return what one statement sorts on.

    Parameters:
        statement: The SQL of a statement carrying an ``ORDER BY``.

    Returns:
        The sort terms, with their whitespace flattened.
    """
    return ' '.join(statement.split('ORDER BY')[1].split())


def test_every_statement_of_the_merge_orders_on_the_whole_key(two_roots: TwoRoots) -> None:
    """Adjacent rows in one order are the whole of what lets three streams merge.

    Read off the statements the source issued rather than off the source code.
    The merge never compares two keys for their order, so it can only be right
    while the three streams arrive in one order, and the sort is on the whole
    key because one index serves several roots: an image stream sorted on the
    stub alone interleaves two roots where a child stream sorted on the key does
    not, and every image whose rows are met out of turn is handed none of them.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    issued = _issued_by(two_roots.url, two_roots.first, Selection())
    assert sorted(_ordering_of(one) for one in issued if 'ORDER BY' in one) == [
        'feature_sources.root_url, feature_sources.results_path_stub',
        'images.root_url, images.results_path_stub',
        'techniques.root_url, techniques.results_path_stub',
    ]


def _child_statement_reading_every_row(
    table: sqlalchemy.Table, conditions: Sequence[sqlalchemy.ColumnElement[bool]]
) -> sqlalchemy.Select[Any]:
    """Return a child statement restricted to nothing at all.

    Stands in for any way a child stream could come to carry a key the image
    stream does not yield -- a write landing between the statements, a join
    dropped, a condition stated over one statement and not the others.

    Parameters:
        table: The child table to read.
        conditions: What the merge asked to restrict the images by, ignored.

    Returns:
        The statement, ordered by the key as the real one is.
    """
    del conditions
    return sqlalchemy.select(table).order_by(table.c.root_url, table.c.results_path_stub)


def test_a_child_row_belonging_to_no_yielded_image_fails_the_read(
    two_roots: TwoRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge that waits gives every later image nothing, and says nothing.

    A row the image stream never yields is one the merge waits for to the end of
    the pass, handing every image after it an empty list that is
    indistinguishable from an image with no rows of its own.  So the pass has to
    end by asking whether anything is still being waited for.

    Parameters:
        two_roots: The two ingested roots and their index.
        monkeypatch: Fixture the unrestricted child read is installed through.
    """
    monkeypatch.setattr(
        spindoctor.results_index.facts_stream,
        '_child_statement',
        _child_statement_reading_every_row,
    )
    with (
        open_record_source(
            [two_roots.first],
            results_index_db_url=two_roots.url,
            columns=COLUMNS,
            tuning=packaged_tree_tuning(),
        ) as source,
        pytest.raises(ValueError, match='did not answer from one state'),
    ):
        list(source.facts(Selection(instrument=MISSION)))


def test_a_read_whose_child_rows_are_all_claimed_does_not_fail(two_roots: TwoRoots) -> None:
    """Without which the guard above would be free to refuse every pass.

    Named rather than counted: a pass that raised nothing while yielding the
    wrong images would satisfy a check that the stream was not empty.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_from_index(two_roots, 'first', Selection(instrument=MISSION))
    assert [stub_of(one) for one in found if isinstance(one, ImageFacts)] == [
        SUCCESS_STUB,
        FAILURE_STUB,
        ERROR_STUB,
        UNLOADED_STUB,
    ]


def test_the_fixture_documents_are_what_the_ingest_read(two_roots: TwoRoots) -> None:
    """The tree half of every comparison is a tree an ingest could read.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    written = json.loads(
        (two_roots.first / f'{SUCCESS_STUB}_metadata.json').read_text(encoding='utf-8')
    )
    assert written['navigation_result']['covariance_px2'] == FIRST_VALUES.twist_covariance
