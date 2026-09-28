"""One state of the index over the three statements a stream of facts issues.

The merge behind :func:`spindoctor.results_index.facts_stream` waits rather than
advances, so a child row whose image the image statement never yielded stops
every image after it being given rows of its own.  The one reachable way for such
a row to exist is a write that lands between the statements, which an ingest
makes routinely: it commits per chunk rather than holding one lock for a whole
pass, exactly so that a reader and a writer can work at once.

The PostgreSQL half of this guarantee is pinned beside the rest of that tier.
This is the SQLite half, and it fails differently.  SQLite holds one state of the
database while a statement of its own is still stepping, and a buffered read
fetches ahead, so an image statement whose whole answer arrives in one fetch is
finished before either child statement is issued.  One image row is therefore the
shape that breaks, and one image row is what a stream restricted to a single
named stub asks for.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlalchemy
from sqlalchemy.engine import Connection
from tests.spindoctor.results_index.conftest import (
    feature_source_row,
    image_row,
    opened,
    sqlite_url_for,
    technique_row,
)

from spindoctor.nav_records import ImageFacts, Selection
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    SCHEMA_VERSION,
    TECHNIQUES,
    IndexRecordSource,
    open_index,
)

RACE_ROOT = '/data/race-nav-results'
"""The one root the images below are under."""

RACE_SUBTREE = 'COISS_2001'
"""The subtree they are all in."""

ALREADY_THERE_STUB = f'{RACE_SUBTREE}/data/N1294561200_1_CALIB'
"""The one image the index holds when the read starts.

One, not two: an image statement answering with two rows leaves its cursor
stepping while the child statements are issued, and a cursor still stepping is
its own reason for SQLite to hold the state of the database.  One row is
answered whole in the first fetch, which is what leaves the child statements
free to read a state of the index the image statement never saw.
"""

ARRIVING_STUB = f'{RACE_SUBTREE}/data/N1294561201_1_CALIB'
"""The image another connection commits part-way through the read.

Sorting after the image already there, so its technique row is the one left in
hand when the image stream ends: the merge waited for an image it was never
going to be handed.
"""

RACE_TECHNIQUE = 'star_field_from_catalog'
"""The technique name each image's one technique row carries."""

RACE_COLUMNS = (IMAGES.c.status, IMAGES.c.instrument)
"""A consumer's columns, which a stream of facts ignores."""

INGESTED_UTC = '2026-08-08T00:00:00+00:00'
"""When the root's one ingest run finished, which a read requires it to have."""


def _rows_of(stub: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return one image's three rows.

    Parameters:
        stub: The image's results path stub.

    Returns:
        Its ``images`` row, its ``techniques`` row and its ``feature_sources``
        row, in that order.
    """
    return (
        image_row(
            root_url=RACE_ROOT,
            results_path_stub=stub,
            subtree=RACE_SUBTREE,
            instrument='coiss',
        ),
        technique_row(root_url=RACE_ROOT, results_path_stub=stub, technique_name=RACE_TECHNIQUE),
        feature_source_row(root_url=RACE_ROOT, results_path_stub=stub, source_name=f'src:{stub}'),
    )


def _write_rows(connection: Connection, stubs: Sequence[str]) -> None:
    """Write the image, technique and feature-source rows of some images.

    Parameters:
        connection: The connection to write on, inside its own transaction.
        stubs: The images to write.
    """
    built = [_rows_of(stub) for stub in stubs]
    connection.execute(IMAGES.insert(), [one[0] for one in built])
    connection.execute(TECHNIQUES.insert(), [one[1] for one in built])
    connection.execute(FEATURE_SOURCES.insert(), [one[2] for one in built])


def _seed(url: str) -> None:
    """Create the index and write the image the racing read starts from.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        _write_rows(connection, [ALREADY_THERE_STUB])
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': RACE_ROOT,
                    'started_utc': INGESTED_UTC,
                    'finished_utc': INGESTED_UTC,
                    'schema_version': SCHEMA_VERSION,
                }
            ],
        )


def _facts_read_against_a_writer(url: str) -> tuple[bool, dict[str, list[str]]]:
    """Read the facts with another connection committing an image part-way in.

    The write is made when the first of the two child statements is issued,
    which is after the image statement and before either child statement has
    anything of its own: the one window in which a read free to answer each
    statement about its own state of the index puts an image into the child
    streams that is not in the image stream.  The writer is another engine, so
    the listener on this one does not fire again underneath itself.

    Parameters:
        url: The index to read and to write into.

    Returns:
        Whether the write landed in that window, and the technique names the
        merge gave each image it yielded.
    """
    landed: list[str] = []
    engine = open_index(url)

    def _write_between(conn: Any, cursor: Any, statement: str, *rest: Any) -> None:
        if landed or 'FROM techniques' not in statement:
            return
        landed.append(statement)
        with opened(url) as writer, writer.begin() as connection:
            _write_rows(connection, [ARRIVING_STUB])

    sqlalchemy.event.listen(engine, 'before_cursor_execute', _write_between)
    with IndexRecordSource(engine, [RACE_ROOT], url, RACE_COLUMNS) as source:
        found = {
            str(one.image['results_path_stub']): [
                str(row['technique_name']) for row in one.techniques
            ]
            for one in source.facts(Selection())
            if isinstance(one, ImageFacts)
        }
    return bool(landed), found


def test_an_image_committed_between_the_statements_leaves_the_read_whole_on_sqlite(
    tmp_path: Path,
) -> None:
    """An ingest commits per chunk, so a read shares the database with a writer.

    The three statements are answered about one state of the index, so an image
    that arrives between them is in all three or in none.  Answered about three
    states it is in the child streams and not in the image stream, and the pass
    then fails outright over a child row no image claimed.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _seed(url)
    _landed, found = _facts_read_against_a_writer(url)
    assert found == {ALREADY_THERE_STUB: [RACE_TECHNIQUE]}


def test_the_racing_write_really_lands_between_the_statements_on_sqlite(
    tmp_path: Path,
) -> None:
    """Without which the test above would hold over a read nothing raced.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _seed(url)
    landed, _found = _facts_read_against_a_writer(url)
    assert landed


def test_the_racing_write_really_reaches_the_database_on_sqlite(tmp_path: Path) -> None:
    """Without which the write could be landing in the window and doing nothing.

    A writer that failed to commit, or committed under some other key, would
    leave the read above whole for a reason that is not the one being pinned.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _seed(url)
    _facts_read_against_a_writer(url)
    with opened(url) as engine, engine.connect() as connection:
        stored: sqlalchemy.ScalarResult[str] = connection.execute(
            sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub)
        ).scalars()
        assert list(stored) == [ALREADY_THERE_STUB, ARRIVING_STUB]


def test_one_image_row_finishes_its_statement_before_the_children_on_sqlite(
    tmp_path: Path,
) -> None:
    """Which is what makes a one-image read the shape that needs the transaction.

    The read below is the merge's own: a buffered image statement, then a second
    statement issued after it.  With one image row the first statement is
    answered whole in its first fetch, so nothing of it is left stepping to hold
    the state of the database, and only an open transaction keeps the second
    statement reading the state the first one saw.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    _seed(url)
    with opened(url) as engine, engine.connect() as connection:
        images = connection.execution_options(yield_per=1000).execute(
            sqlalchemy.select(IMAGES).order_by(IMAGES.c.root_url, IMAGES.c.results_path_stub)
        )
        with opened(url) as writer, writer.begin() as writing:
            _write_rows(writing, [ARRIVING_STUB])
        seen = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count())
            .select_from(IMAGES)
            .where(IMAGES.c.results_path_stub == ARRIVING_STUB)
        ).scalar()
        images.close()
        assert seen == 1
