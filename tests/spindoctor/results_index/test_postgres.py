"""Results-index tests that run against a real PostgreSQL server.

SQLite accepts almost anything: it stores a boolean as an integer, a double as
whatever the value looked like, and a foreign key as a comment unless asked
otherwise. A schema that behaves on SQLite therefore proves very little about
the backend an index shared across machines actually runs on. These re-ask the
questions that the type discipline exists to answer, against a server that
enforces them.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset, so a checkout with no server
still runs a green suite.

Each test gets a schema of its own from the ``postgres_url`` fixture, so a
repeated run, or two workers of a parallel run, never share a table.  A test
that reads the server's catalog takes ``postgres_schema`` as well and filters on
it: the catalog spans every schema on the server, so a lookup by table name
alone answers from whichever schema happens to hold a table of that name.
"""

import contextlib
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import pdslogger
import psycopg
import pytest
import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Connection
from tests.spindoctor.results_index.conftest import (
    STUB,
    feature_source_row,
    image_row,
    opened,
    technique_row,
)

from spindoctor.cli.ck.inputs import read_whole_mission
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import ResultsFilter, SelectionError
from spindoctor.nav_records import (
    ImageFacts,
    NavRecord,
    RecordSource,
    Selection,
    UnreadableFile,
)
from spindoctor.results_index import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
    IndexRecordSource,
    open_index,
    open_record_source,
)
from spindoctor.results_index.facts_stream import _child_statement

pytestmark = pytest.mark.postgres

OTHER_STUB = 'COISS_2002/data/1295221349_1296000000/N1294561202_1_CALIB'

FIFTEEN_DIGIT_OFFSET = -1234.56789012345

BOGUS_PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

UNDEFINED_FUNCTION = '42883'
"""SQLSTATE for an operator the server has no definition of.

Stable across every server locale, which the message text is not.
"""


def _password_of(url: str) -> str:
    """Return the password a URL carries.

    Parameters:
        url: The URL to read.

    Returns:
        The password, or an empty string when the URL carries none.
    """
    return sqlalchemy.engine.make_url(url).password or ''


def _with_password(url: str, password: str) -> str:
    """Return a URL carrying a different password.

    Parameters:
        url: The URL to rewrite.
        password: The password to put in it.

    Returns:
        The rewritten URL, with the password in plain text as a caller writes it.
    """
    rewritten = sqlalchemy.engine.make_url(url).set(password=password)
    return rewritten.render_as_string(hide_password=False)


def _refusal_of(url: str, message: str) -> str:
    """Open a URL, require the refusal it raises, and return that message.

    Parameters:
        url: The URL to open.
        message: Pattern the refusal message must match.

    Returns:
        The refusal message.
    """
    with pytest.raises(ValueError, match=message) as excinfo:
        open_index(url)
    return str(excinfo.value)


def _extra_password_appearances(message: str, url: str) -> list[str]:
    """Return the appearances of a URL's password a refusal does not account for.

    The bare password is searched for, rather than the ``:password@`` form alone,
    because a leak is a leak whatever shape it arrives in.  It is counted rather
    than merely looked for, because the password a server is configured with is
    also, on an ordinary local server, the role name and the database name, and
    those the message names legitimately: what proves masking is that the
    password appears no more often than the password-hiding rendering of the
    same URL accounts for.

    Parameters:
        message: The refusal message to read.
        url: The URL the refusal names.

    Returns:
        One entry per unaccounted appearance, empty when nothing leaked.
    """
    password = _password_of(url)
    masked = sqlalchemy.engine.make_url(url).render_as_string()
    extra = message.count(password) - masked.count(password)
    return [password] * max(extra, 0)


def _skip_without_a_password(url: str) -> None:
    """Skip the calling test when the configured URL carries no password.

    Parameters:
        url: The URL the test drives.
    """
    if not _password_of(url):
        pytest.skip('the configured server URL carries no password to mask')


def test_creating_the_schema_creates_every_table(postgres_url: str, postgres_schema: str) -> None:
    """The metadata emits DDL a server accepts, not just DDL SQLite accepts.

    The listing is scoped to this test's own schema rather than to whatever the
    connection's search path resolves to, so a table another worker left in
    ``public`` cannot answer for one this test was supposed to create.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True) as engine:
        found = sorted(sqlalchemy.inspect(engine).get_table_names(schema=postgres_schema))
    assert found == [
        'failed_files',
        'feature_sources',
        'images',
        'ingest_runs',
        'schema_meta',
        'techniques',
    ]


def test_creating_the_schema_stamps_the_version(postgres_url: str) -> None:
    """The stamp is what the version gate reads on every later open.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_a_consumer_refuses_a_schema_that_was_never_ingested(postgres_url: str) -> None:
    """An empty database is not an index, and absence is not "nothing navigated".

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with pytest.raises(ValueError, match='not a results index'):
        open_index(postgres_url)


def test_a_database_stamped_with_another_version_is_refused(postgres_url: str) -> None:
    """The gate is the whole migration strategy, so it has to hold here too.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match=f'schema version {SCHEMA_VERSION + 1}'):
        open_index(postgres_url)


def test_the_version_message_says_to_delete_and_re_ingest(postgres_url: str) -> None:
    """There are no migrations, so the instruction is the whole remedy.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match='empty the database with sd_results_index drop'):
        open_index(postgres_url)


def test_the_not_an_index_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """A server URL is the one that carries a password, and this gate names it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _skip_without_a_password(postgres_url)
    message = _refusal_of(postgres_url, 'not a results index')
    assert _extra_password_appearances(message, postgres_url) == []


def test_the_version_refusal_does_not_repeat_the_password(postgres_url: str) -> None:
    """Every route names the URL, so every route has to mask it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _skip_without_a_password(postgres_url)
    with opened(postgres_url, create=True) as engine, engine.begin() as connection:
        connection.execute(SCHEMA_META.update().values(schema_version=SCHEMA_VERSION + 1))
    message = _refusal_of(postgres_url, 'is not the version')
    assert _extra_password_appearances(message, postgres_url) == []


def test_a_masked_refusal_still_names_the_server(postgres_url: str) -> None:
    """Masking must not cost the identification the message exists for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    message = _refusal_of(postgres_url, 'not a results index')
    assert str(sqlalchemy.engine.make_url(postgres_url).host) in message


def test_a_rejected_password_is_not_repeated_in_the_refusal(postgres_server_url: str) -> None:
    """The failure most likely to carry a password is the one that is about it.

    Parameters:
        postgres_server_url: URL of the server the tier runs against.
    """
    with pytest.raises(ValueError, match='could not open the results index') as excinfo:
        open_index(_with_password(postgres_server_url, BOGUS_PASSWORD))
    assert BOGUS_PASSWORD not in str(excinfo.value)


def test_the_offset_round_trips_bit_for_bit(postgres_url: str) -> None:
    """Double precision on the server, not the single precision Float may give.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(offset_dv=FIFTEEN_DIGIT_OFFSET))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)).scalar()
    assert repr(stored) == repr(FIFTEEN_DIGIT_OFFSET)


SIGNED_ZERO_MATRIX = [[-0.0, 1e16], [1e16, 1e308]]
"""A matrix of the three values a jsonb column does not return as they were written.

Its number type is ``numeric``, which has no signed zero and no float: ``-0.0``
comes back ``0.0``, and a float of large magnitude comes back as an integer of
the same value.  Each of the three compares ``==`` to what was written, so the
comparison here is on ``repr`` -- the same technique the offset column's
bit-for-bit test uses, and the only one that can see any of these.
"""

OTHER_ROOT_MATRIX = [[-0.0, 2e16], [2e16, 1.5e308]]
"""The same three kinds of value under a second root, differing in every entry.

Two roots hold the one stub, so a select that compared the stub alone would
answer with whichever row the server returned first.
"""

ROUND_TRIP_ROOTS = ('file:///data/nav-results', 'file:///data/nav-results-second')
"""The two roots the JSON round-trip tests write one stub under."""


def _stored_json_under_each_root(
    postgres_url: str, column: sqlalchemy.Column[Any], values: Sequence[Any]
) -> list[Any]:
    """Write one value per root under one stub and read each back by whole key.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        column: The JSON column under test.
        values: What to store under each root, in root order.

    Returns:
        What each root's row holds, in root order.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            for root_url, value in zip(ROUND_TRIP_ROOTS, values, strict=True):
                connection.execute(
                    IMAGES.insert(), image_row(root_url=root_url, **{column.name: value})
                )
        with engine.connect() as connection:
            return [
                connection.execute(
                    sqlalchemy.select(column).where(
                        IMAGES.c.root_url == root_url, IMAGES.c.results_path_stub == STUB
                    )
                ).scalar()
                for root_url in ROUND_TRIP_ROOTS
            ]


def test_a_covariance_round_trips_bit_for_bit(postgres_url: str) -> None:
    """A jsonb column would return this matrix as three different numbers.

    The column is a JSON one because a covariance is a structure, and it holds
    floats, so it is declared plain ``json`` rather than ``jsonb``: the value
    travels as the JSON text the driver wrote and is parsed back by the rules
    that wrote it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    stored = _stored_json_under_each_root(
        postgres_url, IMAGES.c.covariance_px2, (SIGNED_ZERO_MATRIX, OTHER_ROOT_MATRIX)
    )
    assert [repr(one) for one in stored] == [repr(SIGNED_ZERO_MATRIX), repr(OTHER_ROOT_MATRIX)]


def test_a_recorded_attitude_round_trips_bit_for_bit(postgres_url: str) -> None:
    """A kernel is written from these nine numbers, and this project checks kernels.

    A ``-0.0`` in a recorded attitude that came back ``0.0`` would put a
    different rotation into a kernel written from an index than into one written
    from the same tree.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    first = [-0.0, 1e16, 1e308, 0.5, -0.5, 0.25, -0.25, 0.125, 1.0]
    second = [-0.0, 2e16, 1e308, 0.5, -0.5, 0.25, -0.25, 0.125, 2.0]
    stored = _stored_json_under_each_root(postgres_url, IMAGES.c.cmatrix, (first, second))
    assert [repr(one) for one in stored] == [repr(first), repr(second)]


def test_a_technique_covariance_round_trips_bit_for_bit(postgres_url: str) -> None:
    """The child table's matrix column is declared exactly as the image's is.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            for root_url, value in zip(
                ROUND_TRIP_ROOTS, (SIGNED_ZERO_MATRIX, OTHER_ROOT_MATRIX), strict=True
            ):
                connection.execute(IMAGES.insert(), image_row(root_url=root_url))
                connection.execute(
                    TECHNIQUES.insert(), technique_row(root_url=root_url, covariance_px2=value)
                )
        with engine.connect() as connection:
            stored = [
                connection.execute(
                    sqlalchemy.select(TECHNIQUES.c.covariance_px2).where(
                        TECHNIQUES.c.root_url == root_url,
                        TECHNIQUES.c.results_path_stub == STUB,
                    )
                ).scalar()
                for root_url in ROUND_TRIP_ROOTS
            ]
    assert [repr(one) for one in stored] == [repr(SIGNED_ZERO_MATRIX), repr(OTHER_ROOT_MATRIX)]


def test_a_boolean_column_round_trips_true(postgres_url: str) -> None:
    """A native boolean, not an integer flag that happens to read back as one.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row(spurious=True))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(TECHNIQUES.c.spurious)).scalar()
    assert stored is True


def test_comparing_a_boolean_column_to_an_integer_is_an_error(postgres_url: str) -> None:
    """This is why the integer spellings had to go, demonstrated rather than asserted.

    ``spurious = 0`` is ordinary SQLite and a type error here, so a query that
    kept the SQLite spelling would fail the moment an operator moved the index to
    a server.

    The SQLSTATE is what is asserted, not the message: a server translates its
    messages according to ``lc_messages``, so the word "boolean" is absent from
    the text on a server configured for another language, and the test would then
    fail for a reason that has nothing to do with the type discipline.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:  # noqa: SIM117
        with pytest.raises(sqlalchemy.exc.ProgrammingError) as excinfo:
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text('SELECT * FROM techniques WHERE spurious = 0'))
    original = excinfo.value.orig
    # SQLAlchemy types the wrapped exception as any exception at all, so the
    # driver's own class is stated before its result code is read.
    assert isinstance(original, psycopg.Error)
    # 42883 is "undefined function", which is how the server reports that no
    # boolean-to-integer equality operator exists.
    assert original.sqlstate == UNDEFINED_FUNCTION


def test_the_same_basename_in_two_volumes_produces_two_rows(postgres_url: str) -> None:
    """The composite key is a real constraint on the server, not a convention.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(IMAGES.insert(), image_row(results_path_stub=OTHER_STUB))
        with engine.connect() as connection:
            stubs = connection.execute(
                sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub)
            ).scalars()
            found = list(stubs)
    assert found == [STUB, OTHER_STUB]


def test_deleting_an_image_deletes_its_technique_rows(postgres_url: str) -> None:
    """The cascade is enforced natively here, with no pragma to remember.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
            connection.execute(IMAGES.delete().where(IMAGES.c.results_path_stub == STUB))
        with engine.connect() as connection:
            remaining = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(TECHNIQUES)
            ).scalar()
    assert remaining == 0


def test_a_second_row_for_one_technique_of_one_image_is_refused(postgres_url: str) -> None:
    """The uniqueness is a server constraint here, not a SQLite convention.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='uq_techniques_image_technique'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(TECHNIQUES.insert(), technique_row(confidence=0.5))


def test_a_second_row_for_one_feature_source_of_one_image_is_refused(postgres_url: str) -> None:
    """And the wider tuple the feature inventory is aggregated by.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(FEATURE_SOURCES.insert(), feature_source_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='uq_feature_sources_image_source'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(FEATURE_SOURCES.insert(), feature_source_row(n_features=7))


def test_a_json_column_round_trips_a_list(postgres_url: str) -> None:
    """The JSON columns are jsonb here, which the direct-SQL examples rely on.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(excluded_from_consensus=['ring_edge']))
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(IMAGES.c.excluded_from_consensus)
            ).scalar()
    assert stored == ['ring_edge']


def test_a_json_column_is_jsonb(postgres_url: str, postgres_schema: str) -> None:
    """A plain ``json`` column would reject ``jsonb_array_elements_text``.

    The catalog is server-wide, so the lookup is scoped to this test's own
    schema: filtered by table and column alone it would read whichever
    ``images`` row the catalog returned first, which under parallel workers is
    somebody else's.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_schema: Name of that schema.
    """
    with opened(postgres_url, create=True) as engine, engine.connect() as connection:
        found = connection.execute(
            sqlalchemy.text(
                'SELECT data_type FROM information_schema.columns '
                'WHERE table_schema = :schema AND table_name = :table '
                'AND column_name = :column'
            ),
            {'schema': postgres_schema, 'table': 'images', 'column': 'excluded_from_consensus'},
        ).scalar()
    assert found == 'jsonb'


def test_a_nanosecond_mtime_round_trips(postgres_url: str) -> None:
    """A 32-bit integer column would refuse this value outright.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    mtime_ns = 1755000000123456789
    with opened(postgres_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(mtime_ns=mtime_ns))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.mtime_ns)).scalar()
    assert stored == mtime_ns


SELECTION_ROOT = '/data/nav-results'
"""The results root the selection filters are asked about."""

SELECTION_OTHER_ROOT = '/data/other-nav-results'
"""A second ingested root, holding a row for the same stub."""

REFUSED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561203_1_CALIB'
"""A file the ingest refused, which is still a file that exists."""

OTHER_ROOT_REFUSED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561205_1_CALIB'
"""A file the other root refused, which the root under test holds nothing for.

The refusals are keyed by root and stub together exactly as the images are, and
this stub is the one that says so: it belongs to no image row anywhere, so a
refusals arm reading the stub without its root shows it up as a document of the
root under test.
"""

NAVIGATED_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561204_1_CALIB'
"""A document recording an outcome that is not a fatal error."""

SELECTION_SUBTREE = 'COISS_2001'
"""The subtree the selection reads."""

SELECTION_INGESTED = '2026-08-08T00:00:00+00:00'
"""When the pass over the root under test finished."""

SELECTION_OTHER_INGESTED = '2026-08-09T00:00:00+00:00'
"""When the pass over the other root finished, which is later and is not this one."""


def _seed_selection_rows(url: str) -> None:
    """Create the index and write the rows the selection filters read.

    The row under test records a fatal error and no ``status_error`` at all,
    which is the value SQL comparison handles differently from every other; the
    other root's row for the same stub records the SPICE error the filters tell
    apart, so a query that dropped the root would answer with it.  A second row
    of the root under test records a run that finished, so that the filter for
    a document recording no fatal error has something to select and is not
    satisfied by answering nothing.

    Both roots refuse a file, and the other root's refusal names a stub no
    image row anywhere carries, so the refusals arm is held to its root by the
    same evidence the images arm is: read without its root it adds a stub to
    what the root under test holds.

    The two roots' run rows differ the same way.  The other root is passed over
    second, so its run is the newest in the index and its finish time is not
    this root's: what the pass over this root recorded about itself is
    therefore visibly its own.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=SELECTION_ROOT,
                    results_path_stub=STUB,
                    subtree=SELECTION_SUBTREE,
                    status='error',
                    status_error=None,
                ),
                image_row(
                    root_url=SELECTION_ROOT,
                    results_path_stub=NAVIGATED_STUB,
                    subtree=SELECTION_SUBTREE,
                    status='failure',
                    status_error=None,
                ),
                image_row(
                    root_url=SELECTION_OTHER_ROOT,
                    results_path_stub=STUB,
                    subtree=SELECTION_SUBTREE,
                    status='error',
                    status_error='missing_spice_data',
                ),
            ],
        )
        connection.execute(
            FAILED_FILES.insert(),
            [
                {
                    'root_url': SELECTION_ROOT,
                    'results_path_stub': REFUSED_STUB,
                    'reason': 'not a current-schema navigation document',
                    'subtree': SELECTION_SUBTREE,
                    'mtime_ns': 1,
                    'size_bytes': 2,
                },
                {
                    'root_url': SELECTION_OTHER_ROOT,
                    'results_path_stub': OTHER_ROOT_REFUSED_STUB,
                    'reason': 'not a current-schema navigation document',
                    'subtree': SELECTION_SUBTREE,
                    'mtime_ns': 3,
                    'size_bytes': 4,
                },
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': stamp,
                    'finished_utc': stamp,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, stamp in (
                    (SELECTION_ROOT, SELECTION_INGESTED),
                    (SELECTION_OTHER_ROOT, SELECTION_OTHER_INGESTED),
                )
            ],
        )


SELECTION_CANDIDATES = (STUB, NAVIGATED_STUB, REFUSED_STUB, OTHER_ROOT_REFUSED_STUB)
"""Every stub either root records something for, in one fixed order.

The answers below are stated as the whole selection out of these four rather
than as one membership test each, so that a filter reading the other root's rows
is caught by what it gains as well as by what it loses.
"""


def _selecting(url: str, **flags: Any) -> list[str]:
    """Answer one filter combination over the seeded root, against the server.

    Parameters:
        url: The index to answer from.
        flags: The selection flags to apply.

    Returns:
        The stubs that passed, in the order the candidates are listed.
    """
    with ResultsFilter(
        [SELECTION_SUBTREE],
        SELECTION_ROOT,
        logger=pdslogger.NullLogger(),
        results_index_db_url=url,
        **flags,
    ) as results_filter:
        kept = [stub for stub in SELECTION_CANDIDATES if results_filter.passes(stub)]
        batch = [
            ImageFile(
                image_file_url=FCPath(f'{SELECTION_ROOT}/{stub}.IMG'),
                label_file_url=FCPath(f'{SELECTION_ROOT}/{stub}.LBL'),
                results_path_stub=stub,
            )
            for stub in kept
        ]
        return [image.results_path_stub for image in results_filter.filter_batch(batch)]


def test_the_selection_reads_a_document_and_a_refusal_on_postgresql(postgres_url: str) -> None:
    """The two tables are read as one, on the backend that types their columns.

    A file the ingest could not read is still a file the results root holds, so
    the presence filter reaches it as surely as it reaches a document, and the
    two tables it reaches are joined on the backend that will not reconcile
    columns of two types for a caller.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert _selecting(postgres_url, has_offset_file=True) == [
        STUB,
        NAVIGATED_STUB,
        REFUSED_STUB,
    ]


def test_another_roots_refusal_is_not_this_roots_document_on_postgresql(
    postgres_url: str,
) -> None:
    """The refusals are keyed by root and stub together, as the records are.

    The other root's refusal names a file the root under test holds nothing
    for, so a query reading the stub alone hands the enumeration a stub whose
    document is under a different root -- and the enumeration then reads it as
    an image of this root that has already been navigated.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert OTHER_ROOT_REFUSED_STUB not in _selecting(postgres_url, has_offset_file=True)


def test_the_error_filter_selects_on_postgresql(postgres_url: str) -> None:
    """A fatal error is told from a run that finished, on the backend that types both.

    The two tables are read as one here as well, and the refusals arm records no
    status at all: PostgreSQL refuses a union whose columns disagree, where
    SQLite reconciles them, so this is where a status column and a column
    holding nothing have to be one column of one type.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert _selecting(postgres_url, has_offset_error=True) == [STUB]


def test_the_negative_error_filter_selects_on_postgresql(postgres_url: str) -> None:
    """The filter phrased in the negative reaches the other side of the same column.

    A file the ingest refused is on neither side of it: it records no error and
    records no outcome either, so it is excluded here as surely as it is
    excluded from the filter above.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert _selecting(postgres_url, has_no_offset_error=True) == [NAVIGATED_STUB]


def test_a_fatal_error_with_no_cause_is_not_a_spice_error_on_postgresql(
    postgres_url: str,
) -> None:
    """NULL is not equal to anything and not unequal to anything either.

    A fatal error whose cause went unrecorded is the row an inequality drops
    rather than keeps, which silently loses it from the filter that asks for
    every fatal error that is not a SPICE one.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert _selecting(postgres_url, has_offset_nonspice_error=True) == [STUB]


def test_the_error_filter_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root's row for this stub is a SPICE error, and is not read.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_selection_rows(postgres_url)
    assert _selecting(postgres_url, has_offset_spice_error=True) == []


def test_the_snapshot_time_answers_for_one_root_on_postgresql(
    postgres_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """How old this answer is, on the backend that returns the stamp as it typed it.

    The other root was passed over afterwards, so a time read from the newest
    run in the index rather than from the newest run over this root dates the
    answer by an unrelated tree.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        capsys: Fixture the report is captured through.
    """
    _seed_selection_rows(postgres_url)
    ResultsFilter(
        [SELECTION_SUBTREE],
        SELECTION_ROOT,
        logger=pdslogger.PdsLogger(f'postgres_selection_{uuid.uuid4().hex}'),
        results_index_db_url=postgres_url,
        has_offset_file=True,
    )
    assert SELECTION_INGESTED in capsys.readouterr().out


def _seeded_without_the_refusals(url: str) -> None:
    """Seed the selection rows and then take the refusals table away.

    An index whose account was granted the rows it reports on and not the
    bookkeeping beside them is the case the refusal names, and dropping the
    table is how that account's view of it is reproduced.

    Parameters:
        url: The index to create, write into, and take a table from.
    """
    _seed_selection_rows(url)
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE {FAILED_FILES.name}'))


def test_a_table_this_account_cannot_read_is_reported_on_postgresql(postgres_url: str) -> None:
    """A missing relation is a different exception class here, and is still translated.

    An index whose account was granted the rows it reports on and not the
    bookkeeping beside them is the case the refusal names, and it is a
    PostgreSQL case: the class SQLite raises for the same query is another one,
    so a seam that caught only what SQLite raises would let this one out.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(SelectionError, match='could not be read'):
        _selecting(postgres_url, has_offset_file=True)


def test_a_failing_query_raises_no_database_exception_on_postgresql(postgres_url: str) -> None:
    """The consumer that never imports the database layer cannot name its types.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(SelectionError) as excinfo:
        _selecting(postgres_url, has_offset_file=True)
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_a_failing_query_is_reported_without_its_sql_on_postgresql(postgres_url: str) -> None:
    """The advice is what an operator reads, and a statement dump buries it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seeded_without_the_refusals(postgres_url)
    with pytest.raises(SelectionError) as excinfo:
        _selecting(postgres_url, has_offset_file=True)
    assert 'SELECT' not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The record source
# ---------------------------------------------------------------------------

RECORD_ROOT = '/data/record-nav-results'
"""The results root the record source below is opened over."""

RECORD_OTHER_ROOT = '/data/record-other-results'
"""A second ingested root, holding the same stubs with other values."""

RECORD_SUBTREE = 'COISS_2001'
"""The subtree the first two images live under."""

RECORD_OTHER_SUBTREE = 'COISS_2002'
"""The subtree the third lives under, so a subtree restriction has work to do."""

FIRST_RECORD_STUB = f'{RECORD_SUBTREE}/data/N1294561202_1_CALIB'
"""The image every per-image test below reads."""

SECOND_RECORD_STUB = f'{RECORD_SUBTREE}/data/N1294561203_1_CALIB'
"""A second image of the same subtree, exposed later."""

THIRD_RECORD_STUB = f'{RECORD_OTHER_SUBTREE}/data/N1294561204_1_CALIB'
"""An image of the other subtree."""

RECORD_REFUSED_STUB = f'{RECORD_SUBTREE}/data/junk'
"""A file this root's ingest refused, which is still a file that exists."""

ONLY_OTHER_ROOT_STUB = f'{RECORD_SUBTREE}/data/N1294561299_1_CALIB'
"""An image only the other root holds, which this root must never answer for."""

OTHER_REFUSED_STUB = f'{RECORD_OTHER_SUBTREE}/data/other_junk'
"""A file the other root's ingest refused, somewhere else and under another name."""

RECORD_OFFSET = (1.5, -2.5)
"""What this root's images record."""

OTHER_RECORD_OFFSET = (9.5, -8.5)
"""What the other root's rows for the same stubs record."""

FIRST_MIDTIME_ET = 100.0
"""The exposure midtime the first image records."""

SECOND_MIDTIME_ET = 300.0
"""The exposure midtime the second records, well after the first."""

RECORD_COLUMNS = (IMAGES.c.status, IMAGES.c.instrument, IMAGES.c.offset_dv, IMAGES.c.offset_du)
"""A consumer's columns, standing in for any consumer's."""


def _record_rows(root_url: str, offset: tuple[float, float]) -> list[dict[str, object]]:
    """Return one root's image rows, written in reverse of their path order.

    Reversed on purpose.  No statement the source issues asks for an order, so
    what a server hands back is its own; on a freshly written table that is the
    order the rows went in, and a run that needs path order has to impose it
    rather than inherit it.

    Parameters:
        root_url: The root these rows belong to.
        offset: The offset each of them records, which is what tells one root's
            rows from the other's.

    Returns:
        The rows, ready to insert.
    """
    return [
        image_row(
            root_url=root_url,
            results_path_stub=stub,
            subtree=stub.split('/')[0],
            instrument='coiss',
            offset_dv=offset[0],
            offset_du=offset[1],
            midtime_et=midtime,
            source_file=f'{root_url}/{stub}_metadata.json',
            mtime_ns=1_700_000_000_000_000_000,
            size_bytes=512,
        )
        for stub, midtime in (
            (THIRD_RECORD_STUB, SECOND_MIDTIME_ET),
            (SECOND_RECORD_STUB, SECOND_MIDTIME_ET),
            (FIRST_RECORD_STUB, FIRST_MIDTIME_ET),
        )
    ]


def _seed_record_rows(url: str) -> None:
    """Create the index and write the rows the record source reads.

    Two roots, holding the same three stubs with different offsets, each with a
    refused file of its own at a stub the other root has no file at.  A query
    that dropped the root half of the key would therefore answer with the wrong
    offsets, or fail an image this root has nothing to say about.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                *_record_rows(RECORD_ROOT, RECORD_OFFSET),
                *_record_rows(RECORD_OTHER_ROOT, OTHER_RECORD_OFFSET),
                # A stub only the other root holds, so a query that bound a key
                # and dropped the root is caught whichever order the server
                # returned its rows in.
                image_row(
                    root_url=RECORD_OTHER_ROOT,
                    results_path_stub=ONLY_OTHER_ROOT_STUB,
                    subtree=RECORD_SUBTREE,
                    instrument='coiss',
                    offset_dv=OTHER_RECORD_OFFSET[0],
                    offset_du=OTHER_RECORD_OFFSET[1],
                    midtime_et=SECOND_MIDTIME_ET,
                    source_file=f'{RECORD_OTHER_ROOT}/{ONLY_OTHER_ROOT_STUB}_metadata.json',
                    mtime_ns=1_700_000_000_000_000_000,
                    size_bytes=512,
                ),
            ],
        )
        connection.execute(
            FAILED_FILES.insert(),
            [
                {
                    'root_url': root_url,
                    'results_path_stub': stub,
                    'reason': 'not a current-schema navigation document',
                    'subtree': stub.split('/')[0],
                    'mtime_ns': 1_700_000_000_000_000_000,
                    'size_bytes': 64,
                }
                for root_url, stub in (
                    (RECORD_ROOT, RECORD_REFUSED_STUB),
                    (RECORD_OTHER_ROOT, OTHER_REFUSED_STUB),
                )
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url in (RECORD_ROOT, RECORD_OTHER_ROOT)
            ],
        )


@contextlib.contextmanager
def _record_source(url: str, *roots: str) -> Iterator[RecordSource]:
    """Open a record source over some of the seeded roots, and close it after.

    Parameters:
        url: The index URL.
        roots: The roots the source is to hold.

    Yields:
        The source.
    """
    with open_record_source(
        list(roots), results_index_db_url=url, columns=RECORD_COLUMNS
    ) as source:
        yield source


def test_the_listing_unions_both_tables_on_postgresql(postgres_url: str) -> None:
    """A server types the columns of a union, and one arm records no path at all.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(entry.stub for entry in source.listing(Selection()))
    assert found == sorted(
        [FIRST_RECORD_STUB, SECOND_RECORD_STUB, THIRD_RECORD_STUB, RECORD_REFUSED_STUB]
    )


def test_the_listing_answers_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root holds the same stubs and a refusal of its own.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.path.as_posix() for entry in source.listing(Selection())]
    assert [path for path in found if not path.startswith(RECORD_ROOT)] == []


def test_the_listing_narrows_to_a_subtree_on_postgresql(postgres_url: str) -> None:
    """Both arms of the union carry the restriction.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(entry.stub for entry in source.listing(Selection(subtrees=('COISS_2002',))))
    assert found == [THIRD_RECORD_STUB]


def test_a_listing_of_named_stubs_answers_on_postgresql(postgres_url: str) -> None:
    """The keyed form of the listing, on the server that types a union's columns.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        named = Selection(stubs=(THIRD_RECORD_STUB, FIRST_RECORD_STUB))
        found = [entry.stub for entry in source.listing(named)]
    assert found == [THIRD_RECORD_STUB, FIRST_RECORD_STUB]


def test_a_listing_of_named_stubs_reads_the_refusal_arm_on_postgresql(
    postgres_url: str,
) -> None:
    """A file the ingest refused is a file that is there, on either server.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.stub for entry in source.listing(Selection(stubs=(RECORD_REFUSED_STUB,)))]
    assert found == [RECORD_REFUSED_STUB]


def test_a_listing_of_named_stubs_reads_one_root_on_postgresql(postgres_url: str) -> None:
    """The other root holds this stub, and this root has nothing to say about it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.stub for entry in source.listing(Selection(stubs=(ONLY_OTHER_ROOT_STUB,)))]
    assert found == []


def test_a_listing_of_named_stubs_reads_one_roots_refusals_on_postgresql(
    postgres_url: str,
) -> None:
    """The refusal arm carries the root term as surely as the image arm does.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.stub for entry in source.listing(Selection(stubs=(OTHER_REFUSED_STUB,)))]
    assert found == []


def test_the_stream_reads_one_roots_values_on_postgresql(postgres_url: str) -> None:
    """The other root records another offset for every one of these stubs.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = {
            tuple(entry.metadata['offset'])
            for entry in source.records(Selection(instrument='coiss'))
            if isinstance(entry, NavRecord)
        }
    assert found == {RECORD_OFFSET}


def test_the_stream_bounds_time_on_postgresql(postgres_url: str) -> None:
    """A double comparison, against the epoch the document recorded.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = sorted(
            entry.stub
            for entry in source.records(Selection(stop_et=FIRST_MIDTIME_ET))
            if isinstance(entry, NavRecord)
        )
    assert found == [FIRST_RECORD_STUB]


def test_the_stream_reports_a_refusal_on_postgresql(postgres_url: str) -> None:
    """The refused file of this root, and not the one the other root refused.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [
            entry.stub for entry in source.records(Selection()) if isinstance(entry, UnreadableFile)
        ]
    assert found == [RECORD_REFUSED_STUB]


def test_named_stubs_come_back_in_the_order_named_on_postgresql(postgres_url: str) -> None:
    """Bound as a list of keys, and put back into the order a caller named them.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    named = (SECOND_RECORD_STUB, FIRST_RECORD_STUB, THIRD_RECORD_STUB)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [entry.stub for entry in source.records(Selection(stubs=named))]
    assert found == list(named)


def test_a_named_stub_reads_this_roots_row_on_postgresql(postgres_url: str) -> None:
    """The other root holds the same key, recording another offset.

    Read against the root whose rows are written *first*.  The batch read builds
    what it found with a dictionary update, so a query that dropped the root
    half of the key would be answered by whichever row came back last, and
    asking for the root that was written last would be answered correctly by the
    defect.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [
            tuple(entry.metadata['offset'])
            for entry in source.records(Selection(stubs=(FIRST_RECORD_STUB,)))
            if isinstance(entry, NavRecord)
        ]
    assert found == [RECORD_OFFSET]


def test_one_image_is_read_for_one_root_on_postgresql(postgres_url: str) -> None:
    """The per-image lookup joins both tables onto a key that carries the root.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_OTHER_ROOT) as source:
        found = source.record(FIRST_RECORD_STUB)
    assert tuple(found.metadata['offset']) == OTHER_RECORD_OFFSET


def test_the_other_roots_refusal_is_not_this_ones_on_postgresql(postgres_url: str) -> None:
    """A refusal lookup blind to the root would fail an image this root has no file for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with (
        _record_source(postgres_url, RECORD_ROOT) as source,
        pytest.raises(FileNotFoundError, match=OTHER_REFUSED_STUB),
    ):
        source.record(OTHER_REFUSED_STUB)


def test_a_run_puts_the_records_in_path_order_on_postgresql(postgres_url: str) -> None:
    """No statement sorts, so the order a run works in is the one it imposes itself.

    A server sorts text under its own collation, which is why nothing here asks
    it to; the rows are written in reverse of their path order, so a run that
    took the order it was handed would be caught.  Asserted through the function
    the kernel writer collects its mission with, because the ordering is that
    run's guarantee rather than the source's.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        records, _unreadable = read_whole_mission(source.records(Selection(instrument='coiss')))
    paths = [record.path.as_posix() for record in records]
    assert paths == sorted(paths)


def test_a_stream_over_a_table_this_account_cannot_read_is_reported_on_postgresql(
    postgres_url: str,
) -> None:
    """A missing relation is a different exception class here, and is still translated.

    The refusal reaches the caller out of the stream rather than out of the
    call that built it, which is where a failure of a lazily executed query
    arrives.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with opened(postgres_url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE {FAILED_FILES.name}'))
    with (
        _record_source(postgres_url, RECORD_ROOT) as source,
        pytest.raises(ValueError, match='could not be read'),
    ):
        list(source.records(Selection()))


def test_a_named_stub_only_the_other_root_holds_yields_nothing_on_postgresql(
    postgres_url: str,
) -> None:
    """Naming a key does not stop it being a key under one root.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = list(source.records(Selection(stubs=(ONLY_OTHER_ROOT_STUB,))))
    assert found == []


def test_a_named_stub_only_the_other_root_refused_yields_nothing_on_postgresql(
    postgres_url: str,
) -> None:
    """The refusal half of a named-stub read carries its own root term.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = list(source.records(Selection(stubs=(OTHER_REFUSED_STUB,))))
    assert found == []


# ---------------------------------------------------------------------------
# The per-image facts, merged out of three tables on a server
# ---------------------------------------------------------------------------

COLLATION_STUBS = (
    f'{RECORD_SUBTREE}/data/AB_CALIB',
    f'{RECORD_SUBTREE}/data/AB/CALIB',
    f'{RECORD_SUBTREE}/data/A_B_CALIB',
)
"""Three stubs differing only in where their separators fall.

Which is where a server's text order and the codepoint order a walk produces
can part company: a collation is free to weigh a separator against an underscore
however its locale says, and which locale a server was created under is not
this code's to know.  The merge never compares the server's order to one
computed anywhere else, and these are the stubs that would show it if it did.
"""

FACTS_TECHNIQUE = 'StarFieldFromCatalogNav'
"""The technique every seeded image reports, so a row is told apart by its key."""


def _facts_child_name(root_url: str, stub: str) -> str:
    """Return a source name naming the image whose row carries it.

    Parameters:
        root_url: The root half of the image's key.
        stub: The stub half.

    Returns:
        A name no other image's row carries, so a mis-paired row is visible.
    """
    return f'{root_url}|{stub}'


def _seed_facts_rows(url: str) -> None:
    """Create the index and write the images and child rows the facts read.

    Two roots holding the same three stubs, and the child rows written back to
    front, so that neither the insertion order nor the server's own unordered
    read pairs an image with its own children by accident.

    Parameters:
        url: The index to create and write into.
    """
    keys = [
        (root_url, stub)
        for root_url in (RECORD_OTHER_ROOT, RECORD_ROOT)
        for stub in COLLATION_STUBS
    ]
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=root_url,
                    results_path_stub=stub,
                    subtree=RECORD_SUBTREE,
                    instrument='coiss',
                    excluded_from_consensus=['StarRefineNav', 'BodyLimbNav'],
                    covariance_px2=[
                        [0.01, 0.002, 0.0003],
                        [0.002, 0.04, 0.0005],
                        [0.0003, 0.0005, 1e-06],
                    ],
                )
                for root_url, stub in keys
            ],
        )
        connection.execute(
            TECHNIQUES.insert(),
            [
                technique_row(
                    root_url=root_url,
                    results_path_stub=stub,
                    technique_name=FACTS_TECHNIQUE,
                    source_names=[_facts_child_name(root_url, stub)],
                    diagnostics={'iterations': 4},
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
                    source_name=_facts_child_name(root_url, stub),
                )
                for root_url, stub in reversed(keys)
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url in (RECORD_ROOT, RECORD_OTHER_ROOT)
            ],
        )


def _facts_by_stub(url: str, root_url: str) -> dict[str, ImageFacts]:
    """Read one root's facts off the server, keyed by stub.

    Parameters:
        url: The index URL.
        root_url: The root to read.

    Returns:
        The facts of each image.
    """
    with _record_source(url, root_url) as source:
        return {
            str(one.image['results_path_stub']): one
            for one in source.facts(Selection())
            if isinstance(one, ImageFacts)
        }


def _tables_cursored_part_way_through(url: str) -> list[str]:
    """Return the tables this session holds an open cursor over, part-way through.

    ``pg_cursors`` lists the cursors of the asking session and no other, so the
    question is put on the very connection the stream reads on, part-way through
    it.  A statement fetched whole has closed its cursor before the first image
    is yielded, and one issued without a server-side cursor never appears at all.

    Parameters:
        url: The index to read.

    Returns:
        The names of the tables cursored, sorted.
    """
    held: list[Connection] = []
    engine = open_index(url)
    sqlalchemy.event.listen(engine, 'engine_connect', held.append)
    with IndexRecordSource(engine, [RECORD_ROOT], url, RECORD_COLUMNS) as source:
        stream = source.facts(Selection())
        next(stream)
        open_cursors = list(
            held[0].execute(sqlalchemy.text('SELECT statement FROM pg_cursors')).scalars()
        )
        list(stream)
    return sorted(
        name
        for name in ('feature_sources', 'images', 'techniques')
        for statement in open_cursors
        if f'FROM {name}' in statement
    )


def test_the_facts_stream_holds_a_cursor_over_each_table_at_once_on_postgresql(
    postgres_url: str,
) -> None:
    """Three server-side cursors, opened together and read from as the merge goes.

    A merge that ran the three statements one after another, or that fetched any
    of them whole before reading the next, would have closed the cursor it had
    finished with by the time the first image came back.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    assert _tables_cursored_part_way_through(postgres_url) == [
        'feature_sources',
        'images',
        'techniques',
    ]


def test_the_facts_stream_reads_every_image_of_one_root_on_postgresql(
    postgres_url: str,
) -> None:
    """The three cursors put back together answer for the root that was asked for.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    assert sorted(_facts_by_stub(postgres_url, RECORD_ROOT)) == sorted(COLLATION_STUBS)


def test_the_merge_gives_each_image_its_own_children_on_postgresql(postgres_url: str) -> None:
    """Under the server's own collation, which is not the order a walk produces.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_ROOT)
    assert {
        stub: [row['source_names'] for row in one.techniques] for stub, one in found.items()
    } == {stub: [[_facts_child_name(RECORD_ROOT, stub)]] for stub in COLLATION_STUBS}


def test_the_merge_gives_each_image_its_own_feature_sources_on_postgresql(
    postgres_url: str,
) -> None:
    """The other child table, merged onto the same stream by the same rule.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_ROOT)
    assert {
        stub: [row['source_name'] for row in one.feature_sources] for stub, one in found.items()
    } == {stub: [_facts_child_name(RECORD_ROOT, stub)] for stub in COLLATION_STUBS}


def test_the_merge_reads_the_selected_roots_children_on_postgresql(postgres_url: str) -> None:
    """The other root holds the same stubs, with children named for itself.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_OTHER_ROOT)
    assert {
        stub: [row['source_name'] for row in one.feature_sources] for stub, one in found.items()
    } == {stub: [_facts_child_name(RECORD_OTHER_ROOT, stub)] for stub in COLLATION_STUBS}


def test_a_jsonb_matrix_comes_back_whole_on_postgresql(postgres_url: str) -> None:
    """The covariance is jsonb here and text on SQLite, and must read alike.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_ROOT)
    assert found[COLLATION_STUBS[0]].image['covariance_px2'] == [
        [0.01, 0.002, 0.0003],
        [0.002, 0.04, 0.0005],
        [0.0003, 0.0005, 1e-06],
    ]


def test_a_jsonb_list_keeps_the_order_it_was_written_in_on_postgresql(
    postgres_url: str,
) -> None:
    """A jsonb array preserves element order, which a jsonb object would not.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_ROOT)
    assert found[COLLATION_STUBS[0]].image['excluded_from_consensus'] == [
        'StarRefineNav',
        'BodyLimbNav',
    ]


def test_the_facts_carry_every_column_on_postgresql(postgres_url: str) -> None:
    """The consumer's columns narrow a record and never the facts.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    found = _facts_by_stub(postgres_url, RECORD_ROOT)
    assert set(found[COLLATION_STUBS[0]].image) == {column.name for column in IMAGES.columns}


def test_the_facts_report_a_refusal_on_postgresql(postgres_url: str) -> None:
    """A file the ingest refused is a shortfall whichever storage answered.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_record_rows(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [one.stub for one in source.facts(Selection()) if isinstance(one, UnreadableFile)]
    assert found == [RECORD_REFUSED_STUB]


def test_named_stubs_come_back_as_facts_in_the_order_named_on_postgresql(
    postgres_url: str,
) -> None:
    """A batched read, merged per batch and put back into the order named.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_facts_rows(postgres_url)
    named = (COLLATION_STUBS[2], COLLATION_STUBS[0], COLLATION_STUBS[1])
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = [
            str(one.image['results_path_stub'])
            for one in source.facts(Selection(stubs=named))
            if isinstance(one, ImageFacts)
        ]
    assert found == list(named)


# ---------------------------------------------------------------------------
# What the merge needs of the server that SQLite gives it for nothing
# ---------------------------------------------------------------------------

ARRIVING_STUB = f'{RECORD_SUBTREE}/data/N1294561200_1_CALIB'
"""An image another connection commits while the read below is under way.

It sorts ahead of both images already there, deliberately: its child rows are
then the first rows a child stream meets, and a merge holding a row against a
key its image stream never yields waits for that key for the rest of the pass
and hands every image after it none of its own rows.
"""

ALREADY_THERE_STUBS = (
    f'{RECORD_SUBTREE}/data/N1294561201_1_CALIB',
    f'{RECORD_SUBTREE}/data/N1294561202_1_CALIB',
)
"""The images the index holds when the read starts, both of them with children."""


def _concurrent_rows(stub: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return one image's three rows, for the seed and for the write that races it.

    Parameters:
        stub: The image's results path stub.

    Returns:
        Its image row, its technique row and its feature-source row.
    """
    return (
        image_row(
            root_url=RECORD_ROOT,
            results_path_stub=stub,
            subtree=RECORD_SUBTREE,
            instrument='coiss',
        ),
        technique_row(root_url=RECORD_ROOT, results_path_stub=stub, technique_name=FACTS_TECHNIQUE),
        feature_source_row(
            root_url=RECORD_ROOT,
            results_path_stub=stub,
            source_name=_facts_child_name(RECORD_ROOT, stub),
        ),
    )


def _write_rows(connection: Connection, stubs: Sequence[str]) -> None:
    """Write the image, technique and feature-source rows of some images.

    Parameters:
        connection: The connection to write on, inside its own transaction.
        stubs: The images to write.
    """
    built = [_concurrent_rows(stub) for stub in stubs]
    connection.execute(IMAGES.insert(), [one[0] for one in built])
    connection.execute(TECHNIQUES.insert(), [one[1] for one in built])
    connection.execute(FEATURE_SOURCES.insert(), [one[2] for one in built])


def _seed_concurrent_rows(url: str) -> None:
    """Create the index and write the images the racing read starts from.

    Parameters:
        url: The index to create and write into.
    """
    with opened(url, create=True) as engine, engine.begin() as connection:
        _write_rows(connection, ALREADY_THERE_STUBS)
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': RECORD_ROOT,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
            ],
        )


def _facts_read_against_a_writer(url: str) -> tuple[bool, dict[str, list[str]]]:
    """Read the facts with another connection committing an image part-way in.

    The write is made when the first of the two child statements is issued,
    which is after the image statement and before either child statement has
    anything of its own: the one window in which a server free to answer each
    statement from its own snapshot puts an image into the child streams that is
    not in the image stream.  The writer is another engine, so the listener on
    this one does not fire again underneath itself.

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
    with IndexRecordSource(engine, [RECORD_ROOT], url, RECORD_COLUMNS) as source:
        found = {
            str(one.image['results_path_stub']): [
                str(row['technique_name']) for row in one.techniques
            ]
            for one in source.facts(Selection())
            if isinstance(one, ImageFacts)
        }
    return bool(landed), found


def test_an_image_committed_between_the_statements_leaves_the_others_whole_on_postgresql(
    postgres_url: str,
) -> None:
    """An ingest commits per chunk, so a read shares the server with a writer.

    The three statements are answered from one snapshot, so an image that
    arrives between them is in all three or in none.  Answered from three
    snapshots it is in the child streams and not in the image stream, and the
    merge then gives every image it does yield no rows at all.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_concurrent_rows(postgres_url)
    _landed, found = _facts_read_against_a_writer(postgres_url)
    assert found == {stub: [FACTS_TECHNIQUE] for stub in ALREADY_THERE_STUBS}


def test_the_racing_write_really_lands_between_the_statements_on_postgresql(
    postgres_url: str,
) -> None:
    """Without which the test above would hold over a read nothing raced.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_concurrent_rows(postgres_url)
    landed, _found = _facts_read_against_a_writer(postgres_url)
    assert landed


def _tables_read_in_order(url: str) -> list[str]:
    """Return the table each statement of a stream of facts read, in order.

    Parameters:
        url: The index to read.

    Returns:
        One table name per statement issued, in the order they were issued.  The
        statement that reads the refusals names ``images`` in a subquery of its
        own, so only the statements of the merge itself are worth reading here.
    """
    issued: list[str] = []
    engine = open_index(url)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: issued.append(statement),
    )
    with IndexRecordSource(engine, [RECORD_ROOT], url, RECORD_COLUMNS) as source:
        list(source.facts(Selection()))
    return [
        name
        for statement in issued
        for name in ('images', 'techniques', 'feature_sources')
        if f'FROM {name}' in statement
    ]


def test_the_child_statements_are_issued_after_the_image_one_on_postgresql(
    postgres_url: str,
) -> None:
    """Which is what makes the window the write above lands in the deciding one.

    A write committed there is one the image statement was issued too early to
    see and the child statements are issued in time to see, so the two disagree
    about what the index holds unless they are made to answer from one snapshot
    of it.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_concurrent_rows(postgres_url)
    assert _tables_read_in_order(postgres_url)[:3] == ['images', 'techniques', 'feature_sources']


BOUNDARY_FIRST_STUBS = (f'{RECORD_SUBTREE}/data/A_CALIB', f'{RECORD_SUBTREE}/data/M_CALIB')
"""What the first of the two roots below holds."""

BOUNDARY_SECOND_STUBS = (f'{RECORD_SUBTREE}/data/M_CALIB', f'{RECORD_SUBTREE}/data/Z_CALIB')
"""What the second holds, beginning at the stub the first one ends on.

Two adjacent image groups sharing a stub is the shape a merge key that lost its
root half mis-pairs under: the first root's image takes the second root's rows
as well as its own, and the second root's image comes back with none.
"""


def _seed_boundary_rows(url: str) -> None:
    """Create the index and write two roots that meet on one stub.

    Parameters:
        url: The index to create and write into.
    """
    keys = [
        (root_url, stub)
        for root_url, stubs in (
            (RECORD_ROOT, BOUNDARY_FIRST_STUBS),
            (RECORD_OTHER_ROOT, BOUNDARY_SECOND_STUBS),
        )
        for stub in stubs
    ]
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=root_url,
                    results_path_stub=stub,
                    subtree=RECORD_SUBTREE,
                    instrument='coiss',
                )
                for root_url, stub in keys
            ],
        )
        connection.execute(
            TECHNIQUES.insert(),
            [
                technique_row(
                    root_url=root_url,
                    results_path_stub=stub,
                    technique_name=_facts_child_name(root_url, stub),
                )
                for root_url, stub in reversed(keys)
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url in (RECORD_ROOT, RECORD_OTHER_ROOT)
            ],
        )


def _children_over_both_roots(url: str) -> dict[tuple[str, str], list[str]]:
    """Return the technique names the merge gave each image of a two-root stream.

    Parameters:
        url: The index to read.

    Returns:
        The names, by the whole key of the image they were merged onto.
    """
    with _record_source(url, RECORD_ROOT, RECORD_OTHER_ROOT) as source:
        return {
            (str(one.image['root_url']), str(one.image['results_path_stub'])): [
                str(row['technique_name']) for row in one.techniques
            ]
            for one in source.facts(Selection())
            if isinstance(one, ImageFacts)
        }


def test_a_stub_two_adjacent_roots_share_keeps_each_ones_rows_on_postgresql(
    postgres_url: str,
) -> None:
    """One stream over two roots, meeting on a stub they both hold.

    The merge tells the two apart by the root alone, and there is no simpler
    shape in which it has to.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    _seed_boundary_rows(postgres_url)
    assert _children_over_both_roots(postgres_url) == {
        (root_url, stub): [_facts_child_name(root_url, stub)]
        for root_url, stubs in (
            (RECORD_ROOT, BOUNDARY_FIRST_STUBS),
            (RECORD_OTHER_ROOT, BOUNDARY_SECOND_STUBS),
        )
        for stub in stubs
    }


def test_the_two_roots_really_do_meet_on_one_stub() -> None:
    """Without which the test above would hold whatever the merge compared."""
    assert BOUNDARY_FIRST_STUBS[-1] == BOUNDARY_SECOND_STUBS[0]


ORDER_SENSITIVE_IMAGES = 300
"""How many images it takes for a server to answer a child read out of key order.

A statement that does not say how to sort comes back in whatever order the plan
the planner chose produces, and on a table this size that is the order the rows
were written -- which is written here as the reverse of the key.  Below it the
planner walks the child table's own unique index and hands back key order
whether or not the statement asked for one, so a smaller fixture cannot tell an
ordered read from an unordered one.
"""


def _seed_rows_written_back_to_front(url: str) -> tuple[str, ...]:
    """Create the index and write enough images, in reverse of their key order.

    Parameters:
        url: The index to create and write into.

    Returns:
        The stubs written, in key order.
    """
    stubs = tuple(
        f'{RECORD_SUBTREE}/data/N{index:05d}_CALIB' for index in range(ORDER_SENSITIVE_IMAGES)
    )
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=RECORD_ROOT,
                    results_path_stub=stub,
                    subtree=RECORD_SUBTREE,
                    instrument='coiss',
                )
                for stub in reversed(stubs)
            ],
        )
        connection.execute(
            TECHNIQUES.insert(),
            [
                technique_row(
                    root_url=RECORD_ROOT,
                    results_path_stub=stub,
                    technique_name=_facts_child_name(RECORD_ROOT, stub),
                )
                for stub in reversed(stubs)
            ],
        )
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': RECORD_ROOT,
                    'started_utc': SELECTION_INGESTED,
                    'finished_utc': SELECTION_INGESTED,
                    'schema_version': SCHEMA_VERSION,
                }
            ],
        )
    return stubs


def test_every_image_of_a_root_a_planner_scans_keeps_its_own_rows_on_postgresql(
    postgres_url: str,
) -> None:
    """What the child statement's own ordering is worth, where a plan does not give it.

    The merge pairs a child row with its image by holding it until that image
    comes round, which is right only while both streams arrive in one order.  A
    child read that left its order to the plan hands the rows back as they were
    written, and every image but the last is then given none of its own.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    stubs = _seed_rows_written_back_to_front(postgres_url)
    with _record_source(postgres_url, RECORD_ROOT) as source:
        found = {
            str(one.image['results_path_stub']): [
                str(row['technique_name']) for row in one.techniques
            ]
            for one in source.facts(Selection())
            if isinstance(one, ImageFacts)
        }
    assert found == {stub: [_facts_child_name(RECORD_ROOT, stub)] for stub in stubs}


def _child_stubs_a_plan_returns(url: str) -> list[str]:
    """Return the images the child statement names when its order is left to the plan.

    The merge's own statement with its ``ORDER BY`` cleared, so the plan is the
    one the merge would get and the only thing removed is what is under test.

    Parameters:
        url: The index to read.

    Returns:
        One stub per technique row, in whatever order the plan produced.
    """
    statement = _child_statement(TECHNIQUES, [IMAGES.c.root_url == RECORD_ROOT]).order_by(None)
    with opened(url) as engine, engine.connect() as connection:
        return [str(row.results_path_stub) for row in connection.execute(statement)]


def test_the_planner_really_answers_a_child_read_out_of_key_order_on_postgresql(
    postgres_url: str,
) -> None:
    """Without which the test above would hold over rows that arrived sorted anyway.

    How many images it takes before a plan stops walking the child table's own
    unique index depends on the statistics, on the server's cost settings and on
    its version, and this fixture analyzes none of them.  A server that chose an
    index scan at this size would leave the test above passing while the merge's
    ordering did nothing.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    stubs = _seed_rows_written_back_to_front(postgres_url)
    assert _child_stubs_a_plan_returns(postgres_url) != list(stubs)


def test_the_unordered_child_read_still_answers_with_every_row_on_postgresql(
    postgres_url: str,
) -> None:
    """Without which the control above would hold over a read that answered short.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
    """
    stubs = _seed_rows_written_back_to_front(postgres_url)
    assert sorted(_child_stubs_a_plan_returns(postgres_url)) == sorted(stubs)
