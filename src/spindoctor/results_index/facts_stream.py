"""The per-image facts of an index, merged out of the three tables that hold them.

One document becomes one ``images`` row, some ``techniques`` rows and some
``feature_sources`` rows, so putting the facts of one image back together means
reading three tables.  A reader of the documents gets the children for nothing --
they come out of the same file -- and this is what the index pays instead: two
more queries beside the one over ``images``, merged onto it by the key both child
tables carry.

Nothing here is fetched whole.  All three statements are streamed in server-side
chunks, and the merge holds one image's children at a time, so a pass over an
archive-scale root costs what one image costs.

How the merge stays honest
--------------------------

All three statements order on the key, which is what puts each image's child rows
next to each other and in the same sequence as the images they belong to.  That
ordering is safe because all three come out of one database under one collation:
the server compares its own text against its own text, and the answer is
self-consistent whatever locale it sorts under.

What is not safe is matching a server's text order to one computed anywhere else.
A locale collation orders a separator against an underscore differently from the
codepoint order a directory walk produces, so a merge that decided which stream
to advance by comparing keys in Python would pair one image's row with another
image's children on PostgreSQL and not on SQLite.  So nothing here compares two
keys for their order.  The only comparison made is for equality: a child row
belongs to the image being assembled or it does not, and one that does not waits
where it is until the image it belongs to comes round.

The three statements share one state of the index
-------------------------------------------------

Waiting is right only while a child row that does not match belongs to a *later*
image, and one order shared between the streams does not establish that on its
own: the three statements have to be answered about one state of the index as
well.  A server free to give each statement its own snapshot lets a write that
lands between them put an image into the child streams that is not in the image
stream, and the merge then waits for that key for the rest of the pass and hands
every image after it no rows at all.  A run has to expect that write: an ingest
commits per chunk rather than holding one lock for a whole pass, precisely so
that a reader and a writer can work at once.

So the connection is set to answer every statement from one snapshot before the
first of them is issued -- :func:`reading_one_snapshot` -- and a child row still
held when the image stream ends fails the read rather than being dropped
quietly.  The two together are what makes waiting safe: the first removes the
one reachable cause of a row nothing claims, and the second turns a pass that
handed its images back short into one that says so.

Neither backend holds that snapshot by default, and each withholds it
differently.  PostgreSQL reads at ``READ COMMITTED``, where the three statements
take three snapshots however long each of them stays open.  SQLite's reader
holds one state of the database only while a statement of its own is still
stepping, and the image cursor is not: a buffered read fetches ahead, so an
image statement whose whole answer arrives in that first fetch has finished
before either child statement is issued, and the child statements then read a
state of the index the image statement never saw.  That is not the rare case it
sounds like -- one image is what a stream restricted to a single named stub, or
to a subtree holding one image, asks for.

The child statements join to ``images`` under exactly the conditions the image
statement applies, so the keys they carry are the keys the image stream yields
and no child row is left holding at the end of a pass.  Without the join a stream
narrowed to one mission would drag back the children of every image under the
root, and the first of them that belonged to no yielded image would stall the
merge and leave every later image childless.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import sqlalchemy
from sqlalchemy.engine import Connection

from spindoctor.nav_records import ImageFacts
from spindoctor.results_index.schema import FEATURE_SOURCES, IMAGES, TECHNIQUES

__all__ = [
    'facts_stream',
    'reading_one_snapshot',
]

_SNAPSHOT_ISOLATION = {
    'postgresql': 'REPEATABLE READ',
}
"""The isolation level a backend needs to answer three statements alike.

PostgreSQL reads at ``READ COMMITTED`` unless told otherwise, and there every
statement takes a snapshot of its own, so two of them issued back to back on one
connection see two states of the database.  ``REPEATABLE READ`` fixes the
snapshot at the first statement of the transaction and every later one answers
from it.

SQLite is absent because its driver has no isolation level to set for this: it
reads at ``SERIALIZABLE`` already, and what a read of three statements needs
there is a transaction to be serializable *within* -- :data:`_OPEN_READ_TRANSACTION`.
A backend named in neither is read at whatever its driver defaults to, which is
why :mod:`spindoctor.results_index.engine` accepts only the two.
"""

_OPEN_READ_TRANSACTION = {
    'sqlite': 'BEGIN',
}
"""What a backend needs issued to put its reads inside one transaction.

SQLite's Python driver opens a transaction for a statement that writes and
leaves a statement that only reads to fend for itself, so a sequence of
``SELECT``\\ s is a sequence of transactions: the engine holds its read mark
only while some statement of the sequence is still stepping, and drops it the
moment they are all done.  A deferred ``BEGIN`` takes no lock and reads nothing
by itself; what it does is stop the read mark being dropped, so the first
statement to read fixes the state the rest of them answer from too.

Held for as long as the caller keeps the connection, which is a whole pass.  A
reader does not block a writer under write-ahead logging, but it does hold the
log at the point it started reading from, so the file grows until the pass ends
and can be checkpointed.  That is the cost of the guarantee, and it is the same
cost a single long-running cursor already carried.
"""


def reading_one_snapshot(connection: Connection) -> Connection:
    """Return the connection, reading every statement from one state of the index.

    Set before the first statement is issued, since a snapshot is fixed by the
    statement that opens the transaction and cannot be narrowed afterwards.

    Parameters:
        connection: The connection the merge is about to read on, on which
            nothing has been executed yet.

    Returns:
        The same connection, told to hold one snapshot where its backend would
        otherwise answer each statement from its own.  Closing it ends the
        transaction, which the caller does at the end of the pass.
    """
    dialect = connection.engine.dialect.name
    level = _SNAPSHOT_ISOLATION.get(dialect)
    if level is not None:
        connection = connection.execution_options(isolation_level=level)
    opener = _OPEN_READ_TRANSACTION.get(dialect)
    if opener is not None:
        connection.exec_driver_sql(opener)
    return connection


def _key_of(row: sqlalchemy.Row[*tuple[Any, ...]]) -> tuple[str, str]:
    """Return the image one row belongs to.

    Parameters:
        row: A row of any of the three tables, which all carry the pair.

    Returns:
        The root URL and the results path stub, which together are the key.
        Both halves, always: one index serves several roots, and two roots
        holding one stub are told apart by the root alone.
    """
    return str(row.root_url), str(row.results_path_stub)


class _ChildRows:
    """One child table's rows, ordered by image, handed out one image at a time.

    Holds a single row of lookahead, which is what lets the merge tell the end of
    one image's rows from the start of the next without reading past it.

    Parameters:
        rows: The table's rows, ordered by the image key and streamed.
    """

    def __init__(self, rows: Iterator[sqlalchemy.Row[*tuple[Any, ...]]]) -> None:
        self._rows = rows
        self._held: sqlalchemy.Row[*tuple[Any, ...]] | None = next(rows, None)

    def take(self, key: tuple[str, str]) -> list[dict[str, Any]]:
        """Return the rows belonging to one image, consuming them.

        Parameters:
            key: The image being assembled.

        Returns:
            Its rows, in the order the server returned them, each keyed by
            column name -- which is the shape a document reader builds too, so
            neither storage converts anything.  An image with no rows in this
            table gets an empty list, and so does one whose rows have not come
            round yet, which the shared snapshot and the join together rule out.
        """
        taken: list[dict[str, Any]] = []
        while self._held is not None and _key_of(self._held) == key:
            taken.append(dict(self._held._mapping))
            self._held = next(self._rows, None)
        return taken

    def unclaimed(self) -> tuple[str, str] | None:
        """Return the image a row nobody took belongs to.

        Returns:
            The key of the row this stream is holding, or None when it is
            holding none.  Asked once the image stream has ended, where a row
            still held is a row the merge waited for an image that never came:
            free to ask, because the lookahead is already in hand.
        """
        return None if self._held is None else _key_of(self._held)


def _refuse_rows_no_image_claimed(table: sqlalchemy.Table, rows: _ChildRows) -> None:
    """Fail a pass that reached the end of the images with a child row in hand.

    The merge waits rather than advances, so a row belonging to an image the
    image stream never yielded stops every later image being given its own rows
    -- silently, since a merge that waits cannot tell a row that has not come
    round yet from one that never will.  This is where it stops being silent.

    Parameters:
        table: The child table the row came out of, for the message.
        rows: The stream, once the image stream has ended.

    Raises:
        ValueError: If a row is still held, naming the image it belongs to.
            Reported rather than dropped: the images already yielded were handed
            back short, so a consumer that saw only the shortfall would report a
            root as having no techniques at all.
    """
    left = rows.unclaimed()
    if left is None:
        return
    root_url, stub = left
    raise ValueError(
        f'{stub}: the results index answered with {table.name} rows under {root_url} '
        f'for an image the same read did not yield, so the images after it were given '
        f'none of their own. The statements did not answer from one state of the index.'
    )


def _images_statement(
    conditions: Sequence[sqlalchemy.ColumnElement[bool]],
) -> sqlalchemy.Select[Any]:
    """Return the statement reading the image rows a selection covers.

    Every column of ``images``, whatever columns the source's consumer named for
    a record: the facts are the whole row, so a subset of them is a different
    question rather than a cheaper answer to this one.

    Parameters:
        conditions: What restricts the rows, stated over ``images``.

    Returns:
        The statement, ordered by the key so the child streams can be merged
        onto it.
    """
    return (
        sqlalchemy.select(IMAGES)
        .where(*conditions)
        .order_by(IMAGES.c.root_url, IMAGES.c.results_path_stub)
    )


def _child_statement(
    table: sqlalchemy.Table, conditions: Sequence[sqlalchemy.ColumnElement[bool]]
) -> sqlalchemy.Select[Any]:
    """Return the statement reading one child table's rows for those same images.

    Parameters:
        table: The child table, which carries the whole key.
        conditions: What restricts the images, stated over ``images``, which
            this applies through the join so that the rows read are exactly the
            rows of the images the stream yields.

    Returns:
        The statement, ordered by the key it will be merged on.
    """
    return (
        sqlalchemy.select(table)
        .join(
            IMAGES,
            sqlalchemy.and_(
                IMAGES.c.root_url == table.c.root_url,
                IMAGES.c.results_path_stub == table.c.results_path_stub,
            ),
        )
        .where(*conditions)
        .order_by(table.c.root_url, table.c.results_path_stub)
    )


def facts_stream(
    connection: Connection, conditions: Sequence[sqlalchemy.ColumnElement[bool]]
) -> Iterator[ImageFacts]:
    """Yield the per-image facts of the images a selection covers.

    Three statements on the caller's connection, read together rather than one
    after another, so the merge holds one image's children instead of a root's.

    Parameters:
        connection: The connection to read on, whose results the caller has
            already asked to arrive in server-side chunks and from one snapshot
            of the index.
        conditions: What restricts the images, stated over ``images``.

    Yields:
        One set of facts per image row, in the order the server sorts the key.
        Each mapping is keyed by column name, which is how a document reader
        keys them too.  None of them carries a record: these three statements
        select what the facts hold, and a record is rebuilt from a column set of
        its consumer's own, so a consumer wanting both asks twice.

    Raises:
        ValueError: If a child row belongs to no image the image stream yielded,
            naming it, since every image after it was given none of its own.
    """
    images = connection.execute(_images_statement(conditions))
    techniques = _ChildRows(iter(connection.execute(_child_statement(TECHNIQUES, conditions))))
    features = _ChildRows(iter(connection.execute(_child_statement(FEATURE_SOURCES, conditions))))
    for row in images:
        key = _key_of(row)
        yield ImageFacts(
            image=dict(row._mapping),
            techniques=techniques.take(key),
            feature_sources=features.take(key),
        )
    _refuse_rows_no_image_claimed(TECHNIQUES, techniques)
    _refuse_rows_no_image_claimed(FEATURE_SOURCES, features)
