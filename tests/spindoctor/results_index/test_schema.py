"""Tests for the results-index schema.

The column set is the whole contract between ingest and every consumer, and it
carries no migrations: a column that quietly changed name, type or nullability
would be discovered by an operator whose report came out empty. These pin it
column by column, and pin the two properties the keying argument rests on --
that an image is identified by its root and stub rather than its basename, and
that a child row belongs to exactly one image.
"""

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite
from tests.spindoctor.results_index.conftest import (
    ROOT_URL,
    STUB,
    feature_source_row,
    image_row,
    opened,
    sqlite_url_for,
    technique_row,
)

from spindoctor.results_index import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
)

ColumnType = type[sqlalchemy.types.TypeEngine[Any]]

# (name, type, nullable), in schema order.
IMAGES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('root_url', sqlalchemy.Text, False),
    ('results_path_stub', sqlalchemy.Text, False),
    ('subtree', sqlalchemy.Text, True),
    ('image_name', sqlalchemy.Text, False),
    ('instrument', sqlalchemy.Text, False),
    ('camera', sqlalchemy.Text, True),
    ('shutter_mode', sqlalchemy.Text, True),
    ('image_path', sqlalchemy.Text, True),
    ('image_et', sqlalchemy.Double, True),
    ('image_date', sqlalchemy.Text, True),
    ('status', sqlalchemy.Text, False),
    ('status_error', sqlalchemy.Text, True),
    ('status_traceback', sqlalchemy.Text, True),
    ('status_reason', sqlalchemy.Text, True),
    ('offset_dv', sqlalchemy.Double, True),
    ('offset_du', sqlalchemy.Double, True),
    ('sigma_dv', sqlalchemy.Double, True),
    ('sigma_du', sqlalchemy.Double, True),
    ('covariance_px2', sqlalchemy.JSON, True),
    ('sigma_along_unobservable_px', sqlalchemy.Double, True),
    ('rotation_deg', sqlalchemy.Double, True),
    ('sigma_rotation_deg', sqlalchemy.Double, True),
    ('confidence', sqlalchemy.Double, True),
    ('confidence_rank', sqlalchemy.Text, True),
    ('n_techniques', sqlalchemy.Integer, False),
    ('excluded_from_consensus', sqlalchemy.JSON, True),
    ('image_class', sqlalchemy.Text, True),
    ('noise_sigma', sqlalchemy.Double, True),
    ('image_shape_v', sqlalchemy.Integer, True),
    ('image_shape_u', sqlalchemy.Integer, True),
    ('run_start', sqlalchemy.Text, True),
    ('run_end', sqlalchemy.Text, True),
    ('elapsed_s', sqlalchemy.Double, True),
    ('peak_memory_bytes', sqlalchemy.BigInteger, True),
    ('config_hash', sqlalchemy.Text, True),
    ('git_sha', sqlalchemy.Text, True),
    ('pipeline_run', sqlalchemy.Text, True),
    ('spice_kernels', sqlalchemy.JSON, True),
    ('image_number', sqlalchemy.BigInteger, True),
    ('start_et', sqlalchemy.Double, True),
    ('stop_et', sqlalchemy.Double, True),
    ('midtime_et', sqlalchemy.Double, True),
    ('exposure_s', sqlalchemy.Double, True),
    ('sclk_start', sqlalchemy.Text, True),
    ('sclk_midtime', sqlalchemy.Text, True),
    ('sclk_stop', sqlalchemy.Text, True),
    ('camera_frame', sqlalchemy.Text, True),
    ('camera_frame_id', sqlalchemy.Integer, True),
    ('ck_frame_id', sqlalchemy.Integer, True),
    ('cmatrix', sqlalchemy.JSON, True),
    ('cmatrix_original', sqlalchemy.JSON, True),
    ('source_file', sqlalchemy.Text, True),
    ('mtime_ns', sqlalchemy.BigInteger, True),
    ('size_bytes', sqlalchemy.BigInteger, True),
)

TECHNIQUES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('root_url', sqlalchemy.Text, False),
    ('results_path_stub', sqlalchemy.Text, False),
    ('technique_name', sqlalchemy.Text, False),
    ('offset_dv', sqlalchemy.Double, True),
    ('offset_du', sqlalchemy.Double, True),
    ('covariance_px2', sqlalchemy.JSON, True),
    ('confidence', sqlalchemy.Double, True),
    ('spurious', sqlalchemy.Boolean, False),
    ('at_edge', sqlalchemy.Boolean, False),
    ('source_names', sqlalchemy.JSON, True),
    ('diagnostics', sqlalchemy.JSON, True),
)

FEATURE_SOURCES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('root_url', sqlalchemy.Text, False),
    ('results_path_stub', sqlalchemy.Text, False),
    ('feature_type', sqlalchemy.Text, False),
    ('source_model', sqlalchemy.Text, False),
    ('source_name', sqlalchemy.Text, False),
    ('n_features', sqlalchemy.Integer, False),
    ('n_gated', sqlalchemy.Integer, False),
)

INGEST_RUNS_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('run_id', sqlalchemy.Integer, False),
    ('root_url', sqlalchemy.Text, False),
    ('started_utc', sqlalchemy.Text, False),
    ('finished_utc', sqlalchemy.Text, True),
    ('files_seen', sqlalchemy.Integer, True),
    ('files_ingested', sqlalchemy.Integer, True),
    ('files_skipped', sqlalchemy.Integer, True),
    ('files_failed', sqlalchemy.Integer, True),
    ('files_removed', sqlalchemy.Integer, True),
    ('schema_version', sqlalchemy.Integer, False),
)

FAILED_FILES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('root_url', sqlalchemy.Text, False),
    ('results_path_stub', sqlalchemy.Text, False),
    ('reason', sqlalchemy.Text, False),
    ('subtree', sqlalchemy.Text, True),
    ('mtime_ns', sqlalchemy.BigInteger, True),
    ('size_bytes', sqlalchemy.BigInteger, True),
)

SCHEMA_META_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
    ('singleton', sqlalchemy.Integer, False),
    ('schema_version', sqlalchemy.Integer, False),
    ('created_utc', sqlalchemy.Text, False),
)

JSON_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ('images.covariance_px2', 'JSON', 'JSON'),
    ('images.excluded_from_consensus', 'JSON', 'JSONB'),
    ('images.spice_kernels', 'JSON', 'JSONB'),
    ('images.cmatrix', 'JSON', 'JSON'),
    ('images.cmatrix_original', 'JSON', 'JSON'),
    ('techniques.covariance_px2', 'JSON', 'JSON'),
    ('techniques.source_names', 'JSON', 'JSONB'),
    ('techniques.diagnostics', 'JSON', 'JSON'),
)
"""Every JSON column, and the type each backend emits for it.

``type(column.type)`` is ``sqlalchemy.JSON`` for both of the declarations the
schema uses, so the per-table lists above cannot tell them apart and a column
that changed from one to the other would pass every test there.  The difference
is what the value reads back as: PostgreSQL's ``jsonb`` number type is
``numeric``, which has no signed zero and no float, so a column that can hold a
number is plain ``json`` and one holding text alone is ``jsonb``.  Every entry
here is ``(table.column, SQLite type, PostgreSQL type)``.
"""

JSON_COLUMN_CASES = [
    pytest.param(name, sqlite_type, postgres_type, id=name)
    for name, sqlite_type, postgres_type in JSON_COLUMNS
]
"""One case per JSON column, named so a failure says which column moved."""

SQLITE_DIALECT = SQLiteDialect_pysqlite()
"""The dialect a SQLite index compiles its DDL through.

The class an index's own URL selects, instantiated rather than taken off an
engine: constructing an engine imports the backend's driver, and a driver
SpinDoctor ships as an optional extra is then one the tests cannot be collected
without.  A dialect compiles DDL out of the type declarations alone and needs no
driver to do it.
"""

# The driver's dialect leaves its own constructor unannotated, so calling it is
# the one untyped call here; the SQLite one above needs no such exemption.
POSTGRES_DIALECT = PGDialect_psycopg()  # type: ignore[no-untyped-call]
"""The dialect a PostgreSQL index compiles its DDL through, likewise."""

DIALECT_CASES = [
    pytest.param(SQLITE_DIALECT, id='sqlite'),
    pytest.param(POSTGRES_DIALECT, id='postgres'),
]
"""The two backends a declaration is pinned against."""

COLUMN_SET_VERSION = 1
"""The schema version the column sets above are stamped with.

Written down here beside the columns and compared against
:data:`~spindoctor.results_index.SCHEMA_VERSION`, so that the stamp and the
column set it names cannot drift apart.  The stamp says which version wrote a
database rather than which columns it holds, so what keeps a column change from
being silent is the lists above, which is why they restate every name, type,
nullability and position rather than reading them off the schema they guard.
"""

TABLE_CASES = [
    pytest.param(IMAGES, IMAGES_COLUMNS, id='images'),
    pytest.param(TECHNIQUES, TECHNIQUES_COLUMNS, id='techniques'),
    pytest.param(FEATURE_SOURCES, FEATURE_SOURCES_COLUMNS, id='feature_sources'),
    pytest.param(INGEST_RUNS, INGEST_RUNS_COLUMNS, id='ingest_runs'),
    pytest.param(FAILED_FILES, FAILED_FILES_COLUMNS, id='failed_files'),
    pytest.param(SCHEMA_META, SCHEMA_META_COLUMNS, id='schema_meta'),
]

OTHER_STUB = 'COISS_2002/data/1295221349_1296000000/N1294561202_1_CALIB'

FIFTEEN_DIGIT_OFFSET = -1234.56789012345


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """Return a SQLite URL for a database file of this test's own.

    Parameters:
        tmp_path: Pytest-provided directory unique to the test.

    Returns:
        The URL.
    """
    return sqlite_url_for(tmp_path / 'index.sqlite3')


# ---------------------------------------------------------------------------
# The declared column set
# ---------------------------------------------------------------------------


def test_the_schema_stamps_the_version_this_column_set_belongs_to() -> None:
    """The stamp and the columns written beside it name one version.

    The stamp cannot tell one column set from another, so this pins the number
    rather than the column set: what guards the columns is the lists above, and
    what guards those is that they restate the schema instead of reading it.
    """
    assert SCHEMA_VERSION == COLUMN_SET_VERSION


@pytest.mark.parametrize(('table', 'expected'), TABLE_CASES)
def test_table_declares_exactly_these_columns_in_this_order(
    table: sqlalchemy.Table, expected: tuple[tuple[str, ColumnType, bool], ...]
) -> None:
    """The column set and its order are the contract, so both are pinned.

    Order matters because the CSV export writes columns in schema order.

    Parameters:
        table: The table under test.
        expected: The declared (name, type, nullable) triples in schema order.
    """
    assert tuple(table.columns.keys()) == tuple(name for name, _type, _null in expected)


@pytest.mark.parametrize(('table', 'expected'), TABLE_CASES)
def test_every_column_has_the_declared_type(
    table: sqlalchemy.Table, expected: tuple[tuple[str, ColumnType, bool], ...]
) -> None:
    """Types are compared exactly, not by subclass.

    ``Double`` is a subclass of ``Float``, so an ``isinstance`` check would
    accept the bare ``Float`` a dialect is free to map to single precision.

    Parameters:
        table: The table under test.
        expected: The declared (name, type, nullable) triples.
    """
    found = {name: type(table.columns[name].type) for name, _type, _null in expected}
    assert found == {name: column_type for name, column_type, _null in expected}


@pytest.mark.parametrize(('table', 'expected'), TABLE_CASES)
def test_every_column_has_the_declared_nullability(
    table: sqlalchemy.Table, expected: tuple[tuple[str, ColumnType, bool], ...]
) -> None:
    """A column that is NOT NULL rejects a document that omits it.

    Parameters:
        table: The table under test.
        expected: The declared (name, type, nullable) triples.
    """
    found = {name: table.columns[name].nullable for name, _type, _null in expected}
    assert found == {name: nullable for name, _type, nullable in expected}


def _json_columns_of_the_schema() -> set[str]:
    """Return every JSON column the schema declares, as ``table.column``.

    Returns:
        The qualified names.
    """
    return {
        f'{table.name}.{column.name}'
        for table in METADATA.tables.values()
        for column in table.columns
        if isinstance(column.type, sqlalchemy.JSON)
    }


def test_every_json_column_is_declared_here() -> None:
    """A JSON column left out of the list would have its declaration pinned nowhere.

    The per-table lists cannot tell the two JSON declarations apart, so this is
    what obliges a JSON column added to the schema to say which one it takes.
    """
    assert _json_columns_of_the_schema() == {name for name, _sqlite, _postgres in JSON_COLUMNS}


@pytest.mark.parametrize(('name', 'sqlite_type', 'postgres_type'), JSON_COLUMN_CASES)
def test_every_json_column_emits_the_declared_type_on_each_backend(
    name: str, sqlite_type: str, postgres_type: str
) -> None:
    """The compiled type is the one thing that tells the two declarations apart.

    A column that can hold a number and compiles to ``JSONB`` returns a stored
    ``-0.0`` as ``0.0`` and a large-magnitude float as an integer, and every
    equality comparison in the suite passes on all three.

    Parameters:
        name: Qualified name of the column under test.
        sqlite_type: Type the SQLite dialect emits for it.
        postgres_type: Type the PostgreSQL dialect emits for it.
    """
    table_name, column_name = name.split('.')
    column_type = METADATA.tables[table_name].columns[column_name].type
    found = (
        column_type.compile(dialect=SQLITE_DIALECT),
        column_type.compile(dialect=POSTGRES_DIALECT),
    )
    assert found == (sqlite_type, postgres_type)


@pytest.mark.parametrize(('name', '_sqlite_type', '_postgres_type'), JSON_COLUMN_CASES)
@pytest.mark.parametrize('dialect', DIALECT_CASES)
def test_every_json_column_stores_an_absent_value_as_sql_null(
    dialect: sqlalchemy.engine.Dialect, name: str, _sqlite_type: str, _postgres_type: str
) -> None:
    """Absent is not empty, and without this a missing value is the JSON value null.

    A column holding the JSON value ``null`` satisfies ``IS NOT NULL`` on every
    row ever written, so a query counting the rows that carry a matrix counts
    all of them.

    Parameters:
        dialect: The backend whose implementation of the type is under test.
        name: Qualified name of the column under test.
        _sqlite_type: Unused here; the case carries it for the type test.
        _postgres_type: Unused here likewise.
    """
    table_name, column_name = name.split('.')
    column_type = METADATA.tables[table_name].columns[column_name].type
    implementation = column_type.dialect_impl(dialect)
    assert isinstance(implementation, sqlalchemy.JSON)
    assert implementation.none_as_null is True


def test_no_column_anywhere_is_a_bare_float() -> None:
    """Bare ``Float`` may be single precision, which loses the stored offset.

    Checked across every table rather than per column, so a column added later
    cannot reintroduce it without this failing.
    """
    bare = sorted(
        f'{table.name}.{column.name}'
        for table in METADATA.tables.values()
        for column in table.columns
        if type(column.type) is sqlalchemy.Float
    )
    assert bare == []


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------


def test_images_is_keyed_by_root_and_stub() -> None:
    """The basename is not the key: two volumes may hold the same basename."""
    assert [column.name for column in IMAGES.primary_key.columns] == [
        'root_url',
        'results_path_stub',
    ]


@pytest.mark.parametrize('table', [TECHNIQUES, FEATURE_SOURCES], ids=['techniques', 'sources'])
def test_a_child_table_cascades_from_the_composite_key(table: sqlalchemy.Table) -> None:
    """Child rows are keyed on the same pair and go when their image goes.

    Parameters:
        table: The child table under test.
    """
    constraint = next(iter(table.foreign_key_constraints))
    assert constraint.ondelete == 'CASCADE'


@pytest.mark.parametrize('table', [TECHNIQUES, FEATURE_SOURCES], ids=['techniques', 'sources'])
def test_a_child_table_references_both_key_columns(table: sqlalchemy.Table) -> None:
    """Referencing the stub alone would let one root's child join another's.

    Parameters:
        table: The child table under test.
    """
    constraint = next(iter(table.foreign_key_constraints))
    assert [element.parent.name for element in constraint.elements] == [
        'root_url',
        'results_path_stub',
    ]


def test_the_stub_alone_carries_an_index() -> None:
    """A lookup that already knows the root still needs the stub indexed."""
    stub_indexes = [
        index
        for index in IMAGES.indexes
        if [c.name for c in index.columns] == ['results_path_stub']
    ]
    assert len(stub_indexes) == 1


@pytest.mark.parametrize(
    ('table', 'name'),
    [
        pytest.param(IMAGES, 'ix_images_root_url_subtree', id='images'),
        pytest.param(FAILED_FILES, 'ix_failed_files_root_url_subtree', id='failed-files'),
    ],
)
def test_the_root_and_subtree_together_carry_an_index(table: sqlalchemy.Table, name: str) -> None:
    """A selection restricted to some subtrees filters on the pair, in both tables.

    The primary key's second column is the stub, so without this the restriction
    has no ordered path to its rows and reads every row the root holds to find
    them.  Both tables carry it because one selection asks both the same
    question, and the shortfall it reports is counted from the half with no
    images row.

    Parameters:
        table: The table under test.
        name: The index it must carry.
    """
    # Compared as a set because a table's indexes are one: a second index on
    # the pair would otherwise make a list comparison pass or fail by iteration
    # order rather than by what the table declares.
    pairs = {
        index.name
        for index in table.indexes
        if [c.name for c in index.columns] == ['root_url', 'subtree']
    }
    assert pairs == {name}


def test_the_stub_index_is_not_unique() -> None:
    """One stub legitimately appears once per ingested root."""
    stub_index = next(
        index
        for index in IMAGES.indexes
        if [c.name for c in index.columns] == ['results_path_stub']
    )
    assert stub_index.unique is False


@pytest.mark.parametrize(
    ('table', 'expected'),
    [
        pytest.param(
            TECHNIQUES,
            [['root_url', 'results_path_stub', 'technique_name']],
            id='techniques',
        ),
        pytest.param(
            FEATURE_SOURCES,
            [
                [
                    'root_url',
                    'results_path_stub',
                    'feature_type',
                    'source_model',
                    'source_name',
                ]
            ],
            id='sources',
        ),
    ],
)
def test_a_child_table_declares_one_row_per_logical_key(
    table: sqlalchemy.Table, expected: list[list[str]]
) -> None:
    """A second row for one logical key is a contradiction, not more detail.

    The tuple leads with the image key, so the constraint's own index is also
    the index a lookup by image uses; a separate one would be redundant.

    Parameters:
        table: The child table under test.
        expected: The column names of the unique constraints it must declare.
    """
    found = [
        [column.name for column in constraint.columns]
        for constraint in sorted(table.constraints, key=lambda one: str(one.name))
        if isinstance(constraint, sqlalchemy.UniqueConstraint)
    ]
    assert found == expected


# ---------------------------------------------------------------------------
# Behavior against a real SQLite database
# ---------------------------------------------------------------------------


def test_creating_the_schema_creates_every_table(sqlite_url: str) -> None:
    """Opening with create writes the whole schema, not part of it.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        found = sorted(sqlalchemy.inspect(engine).get_table_names())
    assert found == [
        'failed_files',
        'feature_sources',
        'images',
        'ingest_runs',
        'schema_meta',
        'techniques',
    ]


def test_creating_the_schema_stamps_the_version(sqlite_url: str) -> None:
    """The stamp is what the version gate reads on every later open.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine, engine.connect() as connection:
        stamped = connection.execute(sqlalchemy.select(SCHEMA_META.c.schema_version)).scalar()
    assert stamped == SCHEMA_VERSION


def test_the_same_basename_in_two_volumes_produces_two_rows(sqlite_url: str) -> None:
    """The defect the composite key exists to fix: no silent overwrite.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(IMAGES.insert(), image_row(results_path_stub=OTHER_STUB))
        with engine.connect() as connection:
            stubs = connection.execute(
                sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub)
            ).scalars()
            found = list(stubs)
    assert found == [STUB, OTHER_STUB]


def test_two_roots_hold_the_same_stub_independently(sqlite_url: str) -> None:
    """A multi-root index serves each consumer only rows from its own root.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    other_root = 'gs://bucket/nav-results'
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(status='success'))
            connection.execute(IMAGES.insert(), image_row(root_url=other_root, status='error'))
        with engine.connect() as connection:
            status = connection.execute(
                sqlalchemy.select(IMAGES.c.status).where(IMAGES.c.root_url == other_root)
            ).scalar()
    assert status == 'error'


def test_deleting_an_image_deletes_its_technique_rows(sqlite_url: str) -> None:
    """Re-ingesting an image must not leave the previous run's children behind.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
            connection.execute(
                IMAGES.delete().where(IMAGES.c.results_path_stub == STUB),
            )
        with engine.connect() as connection:
            remaining = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(TECHNIQUES)
            ).scalar()
    assert remaining == 0


def test_a_child_row_without_its_image_is_rejected(sqlite_url: str) -> None:
    """The cascade only means anything with foreign keys enforced.

    SQLite leaves them off unless asked, so this is what proves the connect-time
    pragma reached the connection doing the work.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:  # noqa: SIM117
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='FOREIGN KEY constraint failed'):
            with engine.begin() as connection:
                connection.execute(TECHNIQUES.insert(), technique_row())


def test_a_second_row_for_one_technique_of_one_image_is_refused(sqlite_url: str) -> None:
    """A retried or duplicated ingest must not double what the table reports.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='UNIQUE constraint failed'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(TECHNIQUES.insert(), technique_row(confidence=0.5))


def test_two_techniques_of_one_image_are_accepted(sqlite_url: str) -> None:
    """The constraint binds the technique name, not the number of rows per image.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row())
            connection.execute(TECHNIQUES.insert(), technique_row(technique_name='ring_edge'))
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(TECHNIQUES)
            ).scalar()
    assert stored == 2


def test_a_second_row_for_one_feature_source_of_one_image_is_refused(sqlite_url: str) -> None:
    """The inventory is aggregated by this tuple, so it appears once.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(FEATURE_SOURCES.insert(), feature_source_row())
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='UNIQUE constraint failed'):  # noqa: SIM117
            with engine.begin() as connection:
                connection.execute(FEATURE_SOURCES.insert(), feature_source_row(n_features=7))


def test_two_feature_sources_of_one_image_are_accepted(sqlite_url: str) -> None:
    """One image legitimately reports several sources of one feature type.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(FEATURE_SOURCES.insert(), feature_source_row())
            connection.execute(FEATURE_SOURCES.insert(), feature_source_row(source_name='YBSC'))
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(FEATURE_SOURCES)
            ).scalar()
    assert stored == 2


def test_the_offset_round_trips_at_fifteen_significant_digits(sqlite_url: str) -> None:
    """The stored offset is the number every consumer applies, unrounded.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(offset_dv=FIFTEEN_DIGIT_OFFSET))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)).scalar()
    assert stored == FIFTEEN_DIGIT_OFFSET


def test_the_offset_round_trips_bit_for_bit(sqlite_url: str) -> None:
    """Equality could pass on a value that lost bits below the printed digits.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(offset_dv=FIFTEEN_DIGIT_OFFSET))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)).scalar()
    assert repr(stored) == repr(FIFTEEN_DIGIT_OFFSET)


def test_a_true_boolean_reads_back_as_true(sqlite_url: str) -> None:
    """A boolean column reads back as a Python bool, not as 1.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row(spurious=True))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(TECHNIQUES.c.spurious)).scalar()
    assert stored is True


def test_a_false_boolean_reads_back_as_false(sqlite_url: str) -> None:
    """And the false case, which an integer column would return as 0.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row())
            connection.execute(TECHNIQUES.insert(), technique_row(at_edge=False))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(TECHNIQUES.c.at_edge)).scalar()
    assert stored is False


def test_a_json_column_round_trips_a_list(sqlite_url: str) -> None:
    """JSON columns hand back parsed values, not the serialized text.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(excluded_from_consensus=['ring_edge']))
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(IMAGES.c.excluded_from_consensus)
            ).scalar()
    assert stored == ['ring_edge']


def test_a_nanosecond_mtime_round_trips(sqlite_url: str) -> None:
    """The incremental skip compares nanosecond epochs, which need 64 bits.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    mtime_ns = 1755000000123456789
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(IMAGES.insert(), image_row(mtime_ns=mtime_ns))
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.mtime_ns)).scalar()
    assert stored == mtime_ns


def test_a_stub_without_a_separator_stores_a_null_subtree(sqlite_url: str) -> None:
    """The simulated dataset's bare scene basenames are valid stubs.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(), image_row(results_path_stub='sim_scene_0001', subtree=None)
            )
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.subtree)).scalar()
    assert stored is None


def test_status_error_is_retrievable_verbatim(sqlite_url: str) -> None:
    """The SPICE-error selection filter matches this value exactly.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                image_row(
                    status='error',
                    status_error='missing_spice_data',
                    status_reason='no kernel covers the epoch',
                ),
            )
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.status_error)).scalar()
    assert stored == 'missing_spice_data'


def test_status_reason_is_stored_separately_from_status_error(sqlite_url: str) -> None:
    """The two are different vocabularies and must not be merged.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                image_row(
                    status='error',
                    status_error='missing_spice_data',
                    status_reason='no kernel covers the epoch',
                ),
            )
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.status_reason)).scalar()
    assert stored == 'no kernel covers the epoch'


def test_schema_meta_refuses_a_second_row(sqlite_url: str) -> None:
    """The table describes the database, so a second row is a contradiction.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:  # noqa: SIM117
        with pytest.raises(sqlalchemy.exc.IntegrityError, match='ck_schema_meta_singleton'):
            with engine.begin() as connection:
                connection.execute(
                    SCHEMA_META.insert().values(
                        singleton=2,
                        schema_version=SCHEMA_VERSION,
                        created_utc='2026-01-01T00:00:00',
                    )
                )


def test_an_ingest_run_starts_without_a_finish_time(sqlite_url: str) -> None:
    """A run in flight is exactly a row whose finish time is still NULL.

    Parameters:
        sqlite_url: URL of an empty database file.
    """
    with opened(sqlite_url, create=True) as engine:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.insert().values(
                    root_url=ROOT_URL,
                    started_utc='2026-01-01T00:00:00+00:00',
                    schema_version=SCHEMA_VERSION,
                )
            )
        with engine.connect() as connection:
            finished = connection.execute(sqlalchemy.select(INGEST_RUNS.c.finished_utc)).scalar()
    assert finished is None
