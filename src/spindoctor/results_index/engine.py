"""Engine factory and version gate for the navigation results index.

:func:`open_index` is the opener every program that reads or writes the index
goes through.  It selects the backend from the connection URL, applies the
SQLite settings the concurrency model depends on, and refuses a database whose
schema version is not the one this code reads.  Every refusal is a
``ValueError`` naming the URL, including the ones a database driver raises, so a
caller that reports failures catches one type.  The URL is named with its
credentials masked, and so is anything the failure underneath it quoted back:
these messages are written to run logs, returned in cloud task results and
handed to operators, and a database password belongs in none of them.  The
masking rule itself lives in :mod:`spindoctor.results_index.masking`, because a
run log records a command line whose words may include one of these URLs.

:func:`open_database` is the one opener that stops before the version gate, and
it exists for the one operation that has to work on a database the gate refuses:
dropping the tables, which is the remedy the gate's own message prescribes.

Two URL forms are supported::

    sqlite:////data/nav-results/index.sqlite3
    postgresql+psycopg://user@host/spindoctor

A ``sqlite:`` URL names a **local filesystem path**.  It is the one path in this
system that is not a cloud-capable location: the C library opens it directly, so
a network filesystem that cannot honor its locking is refused at open rather than
corrupted later.  A read-only database is not that case and is not refused for
it: a consumer reads one, and only an opener meaning to write is turned away.
PostgreSQL is the option for sharing one index across machines, and its driver
ships as an optional extra.
"""

import contextlib
import datetime
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy
from sqlalchemy.engine import URL, Engine

from spindoctor.results_index.masking import masked_url, without_credentials
from spindoctor.results_index.schema import SCHEMA_META, SCHEMA_VERSION
from spindoctor.results_index.scope import (
    INDEX_TABLE_NAMES,
    carries_the_stamp_marks,
    creation_schema,
    index_tables_in,
    relations_in,
    resolved_schema,
)

__all__ = [
    'SQLITE_BUSY_TIMEOUT_MS',
    'open_database',
    'open_index',
    'reporting_a_failed_read',
]

STUBS_PER_STATEMENT = 500
"""How many stubs one statement names when it asks about a listed set of images.

Each stub is a bind parameter, and every backend limits how many one statement
may carry; the smallest cap among the supported backends is SQLite's own default
of 999 before its 3.32 release.  Every statement that names stubs -- a lookup, a
stream of named records, a prune -- names at most this many at a time, so the
bound is one number in one place rather than a different guess beside each
statement.  It is a property of the SQL dialects underneath, and so a constant
here and not a setting: nothing about how a tree is walked or retrieved has any
bearing on it.
"""

SQLITE_BUSY_TIMEOUT_MS = 30000
"""How long a SQLite connection waits for a competing writer before failing.

Multiple local writer processes are an ordinary case: several ingest workers on
one machine share one file.  With write-ahead logging and short transactions they
only contend briefly, so waiting is nearly always the right answer.
"""

_SQLITE_MEMORY = ':memory:'

_POSTGRES_BACKEND = 'postgresql'

_SQLITE_BACKEND = 'sqlite'

_SQLITE_READONLY_PREFIX = 'SQLITE_READONLY'
"""Prefix shared by every SQLite result code meaning "this database is read-only"."""

_SQLITE_NOT_A_DATABASE = 'SQLITE_NOTADB'
"""Result code SQLite gives for a file that is not a database at all."""

_SQLITE_CANNOT_OPEN = 'SQLITE_CANTOPEN'
"""Result code SQLite gives for a path it cannot open, whatever the reason."""

_SQLITE_BUSY_PREFIX = 'SQLITE_BUSY'
"""Prefix of the result codes that say somebody else holds the write lock.

A lock another process is holding is the ordinary reason: an ingest of this
index is running, or a session was left open on it.  A filesystem that cannot
honor SQLite locking produces this code too, which is why the remedy for that is
offered second rather than asserted first -- the common cause is a live writer,
and telling an operator to rebuild their deployment over one is telling them to
solve the wrong problem.
"""

_SQLITE_LOCK_REFUSED_PREFIXES = ('SQLITE_IOERR',)
"""Prefixes of the result codes a filesystem that cannot lock produces.

Distinct from a busy lock: this is the layer under SQLite failing to carry the
locking operation at all, which no waiting fixes and which moving the index to a
server does.
"""

_SUPPORTED_URL_FORMS = (
    'A results index is either a sqlite: URL naming a local path, or a '
    'postgresql+psycopg: URL naming a server.'
)


@dataclass(frozen=True)
class _Access:
    """What a caller means to do with the database it is opening.

    The three accesses differ in four independent ways, and spelling each one
    out is what keeps them from being inferred from one another.  A drop, in
    particular, writes a database it does not create and reads a version it does
    not require, which no combination of "create" and "read" describes.

    Parameters:
        creating: Whether missing tables and the version row are to be created.
        writing: Whether the database is going to be written, which is what a
            SQLite database on a read-only file or directory is refused for.
        must_exist: Whether a SQLite path that is not there is a refusal.  A
            caller that creates is the only one for which it is not.
        gated: Whether the stamped schema version must be the one this code
            reads.
        writer: What has to write the database, named in the refusal a
            read-only one raises.  Empty for an access that does not write,
            which never reaches that refusal.
        write_remedy: What to do about a read-only database file.  The three
            remedies below are a message table keyed by operation, because the
            same filesystem fact calls for different advice: working on a copy
            answers it for a pass that only wants the rows, and answers nothing
            for one whose whole purpose is to change this database.
        directory_remedy: What to do about a read-only directory, separately,
            because the file itself may be perfectly writable.
        absent_remedy: What to do about a SQLite path that is not there.  Empty
            for an access that creates one, which is never refused for it.
        opening: What this access is opening, as a failure names it, without an
            article so that a message is free to supply its own.  A drop is
            pointed at databases that are not indexes, which is the whole of why
            it exists, so reporting that one of those "could not be opened as
            the results index" would name a fault of its own making.
    """

    creating: bool
    writing: bool
    must_exist: bool
    gated: bool
    writer: str = ''
    write_remedy: str = ''
    directory_remedy: str = ''
    absent_remedy: str = ''
    opening: str = 'results index'


_READING = _Access(
    creating=False,
    writing=False,
    must_exist=True,
    gated=True,
    absent_remedy='Run sd_results_index to build one.',
)
"""Every consumer: the database must already be an index of this version."""

_INGESTING = _Access(
    creating=True,
    writing=True,
    must_exist=False,
    gated=True,
    writer='ingest',
    write_remedy='Ingest a writable copy; a consumer reads this one as it is.',
    directory_remedy='Ingest into a writable directory; a consumer reads this one as it is.',
)
"""The ingest programs: missing tables are created and the version row written."""

_DROPPING = _Access(
    creating=False,
    writing=True,
    must_exist=True,
    gated=False,
    writer='dropping the index',
    write_remedy=(
        'Make the file writable, or delete it: a SQLite index is one file, and removing '
        'it removes the index.'
    ),
    directory_remedy=(
        'Make the directory writable, or delete the database file: a SQLite index is one '
        'file, and removing it removes the index.'
    ),
    absent_remedy='Nothing was dropped.',
    opening='database',
)
"""The drop: a database that is there is opened whatever it holds.

Ungated on purpose.  A database stamped with another version, or carrying no
stamp at all, is exactly what the drop is pointed at -- the gate's own message
prescribes deleting such a database -- so requiring the gate to pass first would
withhold the remedy from the case that needs it.  Nothing is read from the
database through this access, so the columns the gate protects are never
touched: only the table names are, and those come from the schema metadata.
"""


class _IndexOpenError(ValueError):
    """A failure this module has already diagnosed.

    :func:`open_index` turns whatever escapes the builder into a ``ValueError``
    naming the URL, and that catch-all cannot tell a message this module wrote
    from a bare ``ValueError`` a driver raised deep inside itself.  Raising this
    subclass from the failures diagnosed here keeps a complete message from
    being wrapped in a second one, while a caller still catches the
    ``ValueError`` the contract promises.
    """


def _sqlite_error_name(exc: BaseException) -> str:
    """Return the SQLite result-code name a driver exception carries.

    Parameters:
        exc: The exception to read, either a driver exception or the wrapper
            SQLAlchemy raised around one.

    Returns:
        The result-code name, or an empty string when the exception carries
        none -- an exception from another driver, or one whose original is
        absent, which the caller diagnoses generically rather than crashing on.
    """
    original = getattr(exc, 'orig', exc)
    name = getattr(original, 'sqlite_errorname', '')
    return name if isinstance(name, str) else ''


def _is_read_only_error(error_name: str) -> bool:
    """Report whether a SQLite result code says the database will not be written.

    SQLite distinguishes a database it may never write -- a read-only file, or
    one whose directory it cannot create its side files in -- from a write it
    could not take at this moment, such as a busy lock or an I/O error.  Every
    result code in the first group shares one name prefix.

    Parameters:
        error_name: The result-code name to classify, possibly empty.

    Returns:
        True when the code says the database is read-only.
    """
    return error_name.startswith(_SQLITE_READONLY_PREFIX)


def _sqlite_on_connect(dbapi_connection: Any, connection_record: Any) -> None:
    """Apply the per-connection SQLite settings the index depends on.

    Registered as a dialect connect-time event rather than issued as a query, so
    every connection the pool hands out carries them, including ones opened long
    after :func:`open_index` returned.

    Foreign keys are off by default in SQLite, and without them the cascade that
    makes an image's child rows disappear with it does nothing.  Write-ahead
    logging lets a reader and a writer work at the same time, and the busy
    timeout lets two writers queue instead of failing.

    Selecting the journal mode writes the database header, so a database SQLite
    will never write refuses it.  That refusal is tolerated rather than raised,
    because such a database has no writers for a journal to protect and a
    consumer reading an archived copy is a deployment this index supports.
    Nothing is inferred from its absence: whether this caller needed a writable
    database is asked of the filesystem instead, in
    :func:`_require_writable_sqlite_database`.

    Parameters:
        dbapi_connection: The freshly opened DBAPI connection.
        connection_record: The pool's bookkeeping record, which these settings
            do not use.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute(f'PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}')
        try:
            cursor.execute('PRAGMA journal_mode = WAL')
        except sqlite3.Error as exc:
            if not _is_read_only_error(_sqlite_error_name(exc)):
                raise
    finally:
        cursor.close()


def _sqlite_path(parsed: URL) -> Path | None:
    """Return the local filesystem path a SQLite URL names.

    Parameters:
        parsed: A parsed URL whose backend is SQLite.

    Returns:
        The path, or None when the URL names an in-memory database.
    """
    database = parsed.database
    if database is None or database == '' or database == _SQLITE_MEMORY:
        return None
    # pathlib rather than FCPath: SQLite is opened by the C library, which knows
    # nothing of cloud locations, so a SQLite URL is always a local path.
    return Path(database)


def _make_engine(parsed: URL, url: str) -> Engine:
    """Create the engine for a parsed URL, or explain a missing driver.

    Parameters:
        parsed: The parsed connection URL.
        url: The URL as messages name it.

    Returns:
        The engine, not yet connected.

    Raises:
        ValueError: If the URL names a backend whose driver is not installed,
            distinguishing PostgreSQL -- whose driver SpinDoctor ships as an
            extra -- from a backend it does not support at all.
    """
    try:
        return sqlalchemy.create_engine(parsed)
    except ModuleNotFoundError as exc:
        if parsed.get_backend_name() == _POSTGRES_BACKEND:
            raise _IndexOpenError(
                f'{url}: the PostgreSQL driver is not installed. Install it with '
                f'"pip install rms-spindoctor[postgres]", or use a sqlite: URL.'
            ) from exc
        raise _IndexOpenError(
            f'{url}: the driver this URL names is not installed ({exc}), and SpinDoctor '
            f'ships no driver for that backend. {_SUPPORTED_URL_FORMS}'
        ) from exc


def _read_only_refusal(url: str, access: _Access) -> _IndexOpenError:
    """Return the refusal for a database the caller is never going to be able to write.

    Parameters:
        url: The URL as messages name it.
        access: What the caller means to do with the database, which names the
            writer and says what to do instead.

    Returns:
        The refusal to raise.
    """
    return _IndexOpenError(
        f'{url}: this SQLite database is read-only, and {access.writer} has to write it. '
        f'{access.write_remedy}'
    )


def _require_writable_sqlite_database(path: Path, url: str, access: _Access) -> None:
    """Verify that the caller can write the SQLite database a URL names.

    The question is put to the filesystem rather than to SQLite, because SQLite
    does not answer it at open.  A write-ahead-logged database -- the shape this
    opener always leaves behind, having selected that journal mode -- accepts
    both the journal-mode selection, which its header already records, and the
    write lock ``BEGIN IMMEDIATE`` takes, on a file it will never write.  It
    refuses only the first real write, in the middle of an ingest.

    The directory is asked about too: SQLite writes the write-ahead log and its
    shared-memory index beside the database, so a writable file in a directory
    that permits nothing is a database ingest still cannot write.

    Parameters:
        path: The database file's path.
        url: The URL as messages name it.
        access: What the caller means to do with the database, which names the
            writer and says what to do instead.

    Raises:
        ValueError: If the file exists and cannot be written, or if its
            directory exists and cannot be written.
    """
    if path.exists() and not os.access(path, os.W_OK):
        raise _read_only_refusal(url, access)
    directory = path.parent
    if directory.is_dir() and not os.access(directory, os.W_OK):
        raise _IndexOpenError(
            f'{url}: the directory {directory} is read-only, and {access.writer} has to '
            f'write the write-ahead log beside the database. {access.directory_remedy}'
        )


def _cannot_open_message(exc: sqlalchemy.exc.DBAPIError, url: str, path: Path | None) -> str:
    """Return the message for a path SQLite could not open.

    One result code covers a directory that was never created, a path naming
    something that is not a file, and a file this user may not touch.  Each has
    a different remedy, and none of them is PostgreSQL.

    Parameters:
        exc: The driver exception, for what SQLite itself said.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.

    Returns:
        The message.
    """
    if path is None:
        return f'{url}: SQLite could not open this database ({exc.orig}).'
    directory = path.parent
    if not directory.is_dir():
        return (
            f'{url}: the directory {directory} does not exist, so SQLite cannot open a '
            f'database in it. Create the directory first, or name a path inside one that '
            f'already exists.'
        )
    if path.exists() and not path.is_file():
        return (
            f'{url}: {path} is not a file, so it cannot be a SQLite database. Name the '
            f'database file itself.'
        )
    return (
        f'{url}: SQLite could not open {path} ({exc.orig}). Check that this user may read '
        f'and write both that file and its directory.'
    )


def _sqlite_probe_failure(
    exc: sqlalchemy.exc.DBAPIError, url: str, path: Path | None
) -> _IndexOpenError:
    """Return the refusal that fits what SQLite actually said.

    One exception type covers a file that is not a database, a path that cannot
    be opened at all, a lock a filesystem would not grant, and every other way a
    database can refuse a write.  Only the lock is a reason to move the index to
    a server, and prescribing that for the others sends an operator to rebuild a
    deployment over a directory they had not created yet, or over a disk that is
    merely full.  A code this classification does not know is reported as what
    SQLite said, with no remedy invented for it -- including an exception that
    carries no result code at all, which says nothing about locking and must not
    be answered as though it had.

    Codes are matched by prefix, because SQLite refines several of them into
    extended forms -- ``SQLITE_IOERR_WRITE``, ``SQLITE_CANTOPEN_ISDIR`` -- that
    name the same cause more precisely.

    Parameters:
        exc: The driver exception the probe raised.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.

    Returns:
        The refusal to raise.
    """
    error_name = _sqlite_error_name(exc)
    if error_name.startswith(_SQLITE_NOT_A_DATABASE):
        return _IndexOpenError(
            f'{url}: this file is not a SQLite database ({exc.orig}). Check the path: an '
            f'index is built by sd_results_index, and this file holds something else.'
        )
    if error_name.startswith(_SQLITE_CANNOT_OPEN):
        return _IndexOpenError(_cannot_open_message(exc, url, path))
    if error_name.startswith(_SQLITE_BUSY_PREFIX):
        return _IndexOpenError(
            f'{url}: could not take a SQLite write lock within {SQLITE_BUSY_TIMEOUT_MS} ms '
            f'({exc.orig}). Another process is holding it: an ingest of this index, or a '
            f'session left open on it. Wait for that to finish and run this again. If '
            f'nothing else is using this file, then its filesystem is not honoring SQLite '
            f'locking, and a postgresql+psycopg: URL is how one index is shared across '
            f'machines.'
        )
    if error_name.startswith(_SQLITE_LOCK_REFUSED_PREFIXES):
        return _IndexOpenError(
            f'{url}: could not take a SQLite write lock ({exc.orig}). A SQLite index '
            f'must live on a local filesystem that honors locking; use a '
            f'postgresql+psycopg: URL to share one index across machines.'
        )
    return _IndexOpenError(
        f'{url}: SQLite refused {"this database" if path is None else path} '
        f'({error_name or "no result code"}: {exc.orig}).'
    )


def _probe_sqlite_access(engine: Engine, url: str, path: Path | None, access: _Access) -> None:
    """Verify that a SQLite database can be locked, and read when that is all it is.

    Taking and releasing a write lock is the cheapest question that distinguishes
    a filesystem SQLite can be trusted on from one where concurrent writers
    silently corrupt the file.  Asking it at open turns a data-loss bug into a
    startup error.

    A read-only database answers this question in two ways depending on its
    journal mode, so neither answer is read as a verdict on whether the caller
    may write: that is settled from the filesystem before the engine is built.
    What matters here is that a read-only refusal is not reported as a locking
    failure, and that a database whose write-ahead logging makes it unreadable
    says so rather than failing a consumer's first query.  A refusal that
    reaches an ingest here is still reported as read-only, since a network
    filesystem answers the permission question from mode bits it does not
    itself enforce, and SQLite is the one that finds out.

    Parameters:
        engine: The engine to probe.
        url: The URL as messages name it.
        path: The database file's path, or None for an in-memory database.
        access: What the caller means to do with the database.

    Raises:
        ValueError: If the write lock cannot be taken, if the path is not a
            database this code can open, if the database is read-only and the
            caller means to write it, or if it is read-only and cannot be read
            either.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            connection.exec_driver_sql('ROLLBACK')
    except sqlalchemy.exc.DBAPIError as exc:
        if not _is_read_only_error(_sqlite_error_name(exc)):
            raise _sqlite_probe_failure(exc, url, path) from exc
        if access.writing:
            raise _read_only_refusal(url, access) from exc
        _require_sqlite_readable(engine, url)


def _require_sqlite_readable(engine: Engine, url: str) -> None:
    """Verify that a read-only SQLite database can at least be read.

    SQLite reads a write-ahead-logged database through a shared-memory index it
    creates beside the file, so such a database on a filesystem that permits no
    writes at all cannot be read either.  Saying that here beats letting the
    first query fail with a write error on what the caller asked to be a read.

    Parameters:
        engine: The engine to probe.
        url: The URL as messages name it.

    Raises:
        ValueError: If the database cannot be read.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('SELECT count(*) FROM sqlite_master')
    except sqlalchemy.exc.DBAPIError as exc:
        raise _IndexOpenError(
            f'{url}: this SQLite database is read-only and cannot be read either '
            f'({exc.orig}). A write-ahead-logged database creates an index file beside '
            f'itself even to be read, so copy it somewhere writable.'
        ) from exc


def _index_schema_refusal(url: str, schema: str, held: tuple[str, ...]) -> _IndexOpenError:
    """Return the refusal for a schema an index may not be created in.

    Two shapes reach this, and each is named for what it is.  A schema holding a
    relation the index does not own is one that belongs to something else: the
    index and its consumers own the schema they live in, so anything else in it
    says the URL, or the search path behind it, names somewhere it should not.
    A schema holding only relations of the index's own names, with no stamp of
    SpinDoctor's over them, says nothing about whose they are: those names are
    among the commonest there are, and a stamp written beside a stranger's table
    would make it this index's for every later reading.

    Parameters:
        url: The URL as messages name it.
        schema: The schema the index would have been created in.
        held: Every relation that schema holds, sorted.

    Returns:
        The refusal to raise.
    """
    foreign = tuple(name for name in held if name not in INDEX_TABLE_NAMES)
    if foreign:
        return _IndexOpenError(
            f'{url}: schema {schema} of this database holds table(s) the results index does '
            f'not own ({", ".join(foreign)}), so no index was created in it and nothing was '
            f'stamped. A results index owns the schema it lives in -- its own tables are the '
            f'whole of what belongs there -- so a table SpinDoctor did not create in that '
            f'schema means this URL, or the search path behind it, names a database or a '
            f"schema other than the index's. Check the URL, or name an empty schema with "
            f'options=-csearch_path=schemaname.'
        )
    return _IndexOpenError(
        f'{url}: schema {schema} of this database holds table(s) the results index also uses '
        f"({', '.join(held)}), but no schema_meta of SpinDoctor's stands over them, so "
        f'nothing there says they are an index of ours. They are either tables somebody else '
        f'created under names this index also uses, what an index whose stamp has gone left '
        f'behind, or an index another ingest is building at this moment, and the database '
        f'does not say which. No index was created and nothing was stamped, so they are '
        f'exactly as they were. Check the URL; run this again if another ingest was building '
        f'one; remove them by hand if they are an index of yours.'
    )


def _schema_to_create_in(engine: Engine, url: str) -> str:
    """Return the schema an index may be created in, refusing one that is not ours.

    An ingest never stamps a schema that already holds tables SpinDoctor did not
    create.  The stamp is what every later reading takes as proof that the tables
    beside it are the index's -- it is what the drop destroys on the strength of
    -- so writing one over tables of unknown provenance is what makes a
    stranger's table indistinguishable from ours.  The four answers:

    - A schema holding nothing is created in and stamped.
    - A schema carrying a stamp of SpinDoctor's is the index's own, whatever
      version it is stamped with, and is left to the version gate to accept or
      refuse.
    - A schema holding relations of the index's own names with no such stamp is
      refused, since a name is not evidence.  An index another ingest is part
      way through building looks like this for as long as that takes, and is
      answered the same way: running again once it has finished is what the
      refusal says to do.
    - A schema holding any relation the index does not own is refused, stamp or
      no stamp.

    The question is asked of one schema, not of the database: the one the index
    resolves to, which is where a stamp of ours was found, or, for a database
    that reaches no stamp at all, the one a table created without a schema name
    lands in.  Every other schema of the database belongs to whoever made it and
    is neither read nor named.

    Parameters:
        engine: The open engine.
        url: The URL as messages name it.

    Returns:
        The schema to create the index in.

    Raises:
        ValueError: If that schema holds anything this did not create, or if the
            connection reaches no schema a table could be created in.
    """
    with engine.connect() as connection:
        schema = resolved_schema(connection)
        if schema is None:
            schema = creation_schema(connection)
        if schema is None:
            raise _IndexOpenError(
                f'{url}: this connection reaches no schema a table can be created in, so '
                f'there is nowhere to build a results index. Name a schema that exists with '
                f'options=-csearch_path=schemaname, or create one on the server first.'
            )
        held = relations_in(connection, schema)
        if not held:
            return schema
        if not carries_the_stamp_marks(connection, schema) or any(
            name not in INDEX_TABLE_NAMES for name in held
        ):
            raise _index_schema_refusal(url, schema, held)
    return schema


def _create_schema(engine: Engine, schema: str) -> None:
    """Create every missing table and stamp the database with its version.

    Every table is named with its schema, so a name this creates cannot resolve
    through a search path onto a table of that name in another schema and be
    built over it.  The whole of the index therefore lands in one schema, which
    is the schema the drop later removes it from.

    Parameters:
        engine: The engine to create the schema in.
        schema: The schema to create the tables in, from
            :func:`_schema_to_create_in`.
    """
    bound = index_tables_in(schema)
    stamp_table = bound[SCHEMA_META.name]
    # Every table of that mapping shares one metadata container, which is what
    # creates them together and in dependency order.
    stamp_table.metadata.create_all(engine)
    with engine.begin() as connection:
        stamped = connection.execute(sqlalchemy.select(stamp_table.c.schema_version)).first()
        if stamped is None:
            connection.execute(
                stamp_table.insert().values(
                    singleton=1,
                    schema_version=SCHEMA_VERSION,
                    created_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                )
            )


def _stamped_version(engine: Engine) -> int | None:
    """Return the schema version the database is stamped with.

    Read by the version gate, and by nothing else: the drop reads its own,
    because it reads one out of a named schema and tolerates a value that is not
    a version, neither of which the gate does.

    Parameters:
        engine: The engine to inspect.

    Returns:
        The stamped version, or None when the database carries no
        ``schema_meta`` row -- because the table is absent, or because it is
        empty after an interrupted creation.
    """
    if not sqlalchemy.inspect(engine).has_table(SCHEMA_META.name):
        return None
    with engine.connect() as connection:
        row = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).first()
    return None if row is None else int(row.schema_version)


def _verify_schema_version(stamped: int | None, url: str) -> None:
    """Verify that a stamped version is the one this code reads.

    Parameters:
        stamped: The version read from the database, or None when it carries no
            ``schema_meta`` row.
        url: The URL as messages name it.

    Raises:
        ValueError: If the database carries no ``schema_meta`` row, or one whose
            version differs from
            :data:`~spindoctor.results_index.schema.SCHEMA_VERSION`.
    """
    if stamped is None:
        raise _IndexOpenError(
            f'{url}: this is not a results index (it has no schema_meta row). '
            f'Run sd_results_index first to build one.'
        )
    if stamped != SCHEMA_VERSION:
        raise _IndexOpenError(
            f'{url}: results index schema version {stamped} is not the '
            f'version {SCHEMA_VERSION} this code reads. There are no migrations: empty '
            f'the database with sd_results_index drop and re-run sd_results_index ingest.'
        )


def _sqlite_target(parsed: URL, url: str, access: _Access) -> Path | None:
    """Return the local path a SQLite URL names, refusing one it cannot serve.

    Parameters:
        parsed: The parsed connection URL, whose backend is SQLite.
        url: The URL as messages name it.
        access: What the caller means to do with the database.

    Returns:
        The database file's path, or None for an in-memory database.

    Raises:
        ValueError: If the URL carries a query string; if the caller means to
            write a database the filesystem will not let it write; or if the
            caller requires one that is not there.
    """
    if parsed.query:
        carried = ', '.join(sorted(parsed.query))
        raise _IndexOpenError(
            f'{url}: a SQLite index URL is a plain local path, and this one carries a '
            f'query string ({carried}). The driver would open a file named after the '
            f'query rather than the one named here, so name the file alone.'
        )
    path = _sqlite_path(parsed)
    if path is None:
        if access.must_exist:
            raise _IndexOpenError(
                f'{url}: this URL names an in-memory SQLite database, which holds nothing '
                f'when it is opened and is gone when it is closed, so nothing else can ever '
                f'have written one. Name the database file itself. {access.absent_remedy}'
            )
        return None
    if access.must_exist and not path.exists():
        raise _IndexOpenError(
            f'{url}: there is no {access.opening} at {path}. {access.absent_remedy}'
        )
    if access.writing:
        _require_writable_sqlite_database(path, url, access)
    return path


def _build_engine(url: str, access: _Access) -> Engine:
    """Open the database, letting a driver's own exceptions escape untranslated.

    :func:`_translated` wraps this so that every escape becomes a ``ValueError``.
    The separation keeps the translation in one place rather than repeated around
    each call a driver can fail inside.

    Parameters:
        url: The connection URL.
        access: What the caller means to do with the database.

    Returns:
        An open engine.

    Raises:
        ValueError: For the failures this module diagnoses itself.
    """
    parsed = sqlalchemy.engine.make_url(url)
    safe_url = masked_url(url)
    backend = parsed.get_backend_name()
    path = _sqlite_target(parsed, safe_url, access) if backend == _SQLITE_BACKEND else None
    engine = _make_engine(parsed, safe_url)
    try:
        if backend == _SQLITE_BACKEND:
            sqlalchemy.event.listen(engine, 'connect', _sqlite_on_connect)
            _probe_sqlite_access(engine, safe_url, path, access)
        if not access.gated:
            if backend != _SQLITE_BACKEND:
                # Reading the stamp is what reaches the server for every other
                # access.  An ungated open that returned here would hand back an
                # engine that has never connected to anything, and a URL naming
                # a database the server does not have, a password it refuses or
                # a host nothing answers on would surface later as some
                # statement failing rather than as this URL not opening -- which
                # is the difference between a diagnosis and a symptom.
                with engine.connect():
                    pass
            return engine
        # Which schema to build in is settled before a column of the database is
        # read, because a schema that is not this index's own is one whose
        # columns are nobody's business here: reading a stamp out of a stranger's
        # schema_meta first would answer a wrong URL with whatever that statement
        # fell over rather than with the diagnosis.
        schema = _schema_to_create_in(engine, safe_url) if access.creating else None
        # The stamped version is checked before anything is written: creating
        # this version's tables inside a database stamped with another version
        # would leave a mixture no single version number describes.
        stamped = _stamped_version(engine)
        if stamped is not None:
            _verify_schema_version(stamped, safe_url)
        if schema is not None:
            _create_schema(engine, schema)
            stamped = _stamped_version(engine)
        # The gate a non-creating open is refused by.  Reached after the create
        # branch too, where it re-reads the row that branch has just written and
        # so cannot fail; no test can observe that call refusing anything.
        _verify_schema_version(stamped, safe_url)
    except Exception:
        engine.dispose()
        raise
    return engine


def _translated(url: str, access: _Access) -> Engine:
    """Open a database, turning every way of failing into one exception type.

    Parameters:
        url: The connection URL.
        access: What the caller means to do with the database.

    Returns:
        An open engine.

    Raises:
        ValueError: For every failure, with the driver's own exception kept as
            the ``__cause__`` and every credential of the URL replaced, both in
            the URL the message names and in whatever the failure quoted back.
    """
    try:
        return _build_engine(url, access)
    except _IndexOpenError:
        raise
    except sqlalchemy.exc.NoSuchModuleError as exc:
        raise ValueError(
            f'{masked_url(url)}: there is no database driver for this URL scheme '
            f'({without_credentials(str(exc), url)}). {_SUPPORTED_URL_FORMS}'
        ) from exc
    except Exception as exc:
        # Everything else, not only SQLAlchemy's own exceptions: a dialect
        # reports a malformed port or an uncoercible connect argument as a bare
        # ValueError naming neither the URL nor the setting that supplied it.
        # What it does name is the piece of the URL it stopped on, which is why
        # the quoted message is cleaned as well as the URL beside it.
        raise ValueError(
            f'{masked_url(url)}: could not open the {access.opening} '
            f'({type(exc).__name__}: {without_credentials(str(exc), url)}).'
        ) from exc


def open_index(url: str, *, create: bool = False) -> Engine:
    """Open the results index named by a connection URL.

    Every program that reads or writes the index goes through this, so the
    version gate cannot be bypassed by one.  :func:`open_database` is the one
    other opener, and it reads and writes nothing: it exists so that the tables
    the gate refused can be dropped.

    With ``create`` false -- every consumer -- a database that does not exist, or
    that carries no ``schema_meta`` row, is an error naming ``sd_results_index``.
    A consumer pointed at a SQLite path that does not exist fails; it does not
    leave an empty database behind.  With ``create`` true -- the ingest programs
    -- missing tables are created and the version row is written, and a database
    the filesystem will not let this user write is refused before anything is
    opened.

    **A creating open never stamps a schema that already holds tables SpinDoctor
    did not create.**  The tables go into one schema: the one a stamp of
    SpinDoctor's was found in, or, for a database carrying no such stamp, the one
    a table created without a schema name lands in.  That schema is created in
    when it holds nothing, and gone on with when it carries a stamp of
    SpinDoctor's, whatever version that stamp names.  It is refused when it holds
    a table of one of the index's own names with no such stamp over it -- a name
    is not evidence, and a stamp written beside a stranger's table would make it
    this index's for every later reading -- and refused when it holds any table
    the index does not own, stamp or no stamp, since a results index owns the
    schema it lives in.  A refusal names the schema and the tables it found and
    creates nothing, stamps nothing and leaves that schema exactly as it was.
    Every other schema of the database is neither read nor named.

    Either way a database stamped with a different schema version is refused,
    naming both versions, because the index carries no migrations and rebuilding
    it is always available and always correct.  No table is created and no row is
    written in a database whose version is refused.

    Every failure is a ``ValueError`` naming the URL, including the ones a
    database driver raises: a consumer that wants to report the cause rather than
    crash catches one type, and the driver's own exception is kept as the
    ``__cause__``.  The URL is named with its credentials masked, and so is
    whatever the failure underneath it said, since a driver that could not read a
    URL reports the piece of it that stopped it.

    Parameters:
        url: A ``sqlite:`` URL naming a local filesystem path, or a
            ``postgresql+psycopg:`` URL naming a server.
        create: Whether to create missing tables and the version row.

    Returns:
        An open engine.  SQLite engines have foreign keys, write-ahead logging
        and a busy timeout applied to every connection.

    Raises:
        ValueError: If the URL cannot be parsed, carries a query string a SQLite
            index may not have, names a backend with no driver installed, or
            names a server that will not accept the connection; if SQLite
            refuses the file -- because its filesystem cannot honor write
            locking, or for any other cause it reports -- or the file cannot be
            written and ``create`` is true; if ``create`` is false and the
            database or its ``schema_meta`` row does not exist; if ``create`` is
            true and the schema the index resolves to holds tables no stamp of
            SpinDoctor's stands over, or any table the index does not own; or if
            the stamped schema version is not the one this code reads.
    """
    return _translated(url, _INGESTING if create else _READING)


def open_database(url: str) -> Engine:
    """Open the database a URL names, without requiring it to hold an index.

    The opener for the one operation that has to work on a database
    :func:`open_index` refuses: dropping the tables, which is the remedy the
    version gate's own message prescribes.  Requiring the gate to pass first
    would withhold that remedy from every database that needs it -- one stamped
    with a version this code does not read, and one left holding part of a
    schema by an interrupted creation.

    Everything else :func:`open_index` does is done here.  The URL is parsed,
    diagnosed and named the same way, a SQLite database is probed the same way
    and refused for the same causes, and every failure is the same ``ValueError``
    naming the same masked URL.  The database must already be there, on both
    backends alike: a SQLite path that is not there is refused rather than
    created, exactly as a PostgreSQL database that is not there is refused by
    the server.

    Nothing is read out of the database through this, so no column the version
    gate protects is ever touched.  What the caller may do with it is name
    tables, and the only names it has come from the schema metadata.  The
    connection is taken all the same, and taken here: reading the stamp is what
    reaches the server for the other opener, and an engine handed back without
    it would turn a database that is not there, or a password the server
    refuses, into whatever statement the caller ran first.

    Parameters:
        url: A ``sqlite:`` URL naming an existing local path, or a
            ``postgresql+psycopg:`` URL naming a server.

    Returns:
        An open engine.  SQLite engines have foreign keys, write-ahead logging
        and a busy timeout applied to every connection.

    Raises:
        ValueError: If the URL cannot be parsed, carries a query string a SQLite
            index may not have, names a backend with no driver installed, or
            names a server that will not accept the connection; or if SQLite
            refuses the file, including one that is not there and one the
            filesystem will not let this user write.
    """
    return _translated(url, _DROPPING)


@contextlib.contextmanager
def reporting_a_failed_read(url: str) -> Iterator[None]:
    """Report a database failure as the refusal every consumer already catches.

    :func:`open_index` goes to some length to make every way of failing to open
    the index a ``ValueError``, so that a consumer reporting the cause rather
    than crashing catches one type.  The queries issued afterwards are outside
    that guarantee on their own: a table the account may not read, a database
    holding part of the schema, or a connection lost between the open and the
    query raises the database layer's own exception, which a caller that
    deliberately never imports SQLAlchemy cannot name in an ``except`` clause.

    The whole family is caught rather than the operational failures alone: a
    missing table is one class on SQLite and another on PostgreSQL, a database
    whose file is damaged is a third, and a caller that cannot name any of them
    is not helped by a distinction between them.

    What the report carries is the driver's own sentence.  The database layer
    renders a failed statement with the SQL, the bound parameters and a link to
    its own documentation, which is eight lines and a thousand characters of
    machinery around the one sentence that says what to change -- and this
    wrapper exists precisely so that the sentence is what an operator reads.

    Parameters:
        url: The index URL, masked here so that the report names which index
            was asked without printing its password.

    Yields:
        Nothing; the queries run inside the translation.

    Raises:
        ValueError: If the block raises anything the database layer raised.
    """
    try:
        yield
    except sqlalchemy.exc.SQLAlchemyError as exc:
        # Compared against None rather than taken for its truth: str(None) is
        # 'None', which would read as a driver that said so.
        driver_error = getattr(exc, 'orig', None)
        detail = (str(driver_error).strip() if driver_error is not None else '') or str(exc)
        raise ValueError(
            f'{masked_url(url)}: the results index could not be read '
            f'({type(exc).__name__}: {detail}). Check that this URL names an index '
            f'sd_results_index wrote and that the account it is opened with may read '
            f'every table of it.'
        ) from exc
