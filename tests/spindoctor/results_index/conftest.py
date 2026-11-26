"""Shared fixtures and row factories for the results-index tests.

The row factories exist so a test that cares about one column does not have to
restate the NOT NULL columns around it, and so a column gaining a constraint
fails in one place rather than in every test.

The PostgreSQL fixtures give each test its own schema.  Two workers of a parallel
run, or two runs against the same server, otherwise share one set of tables and
see one another's rows.

The two results roots and the accessors over a stream of facts are shared by the
several modules that read that one pair of roots from different angles.  Two
things are built into them because both are this codebase's own defect classes.

**Two roots, and the second differs in the value under test.**  The key is
``(root_url, results_path_stub)``, one index serves several roots, and a query
that filters on the stub alone passes every single-root test there is.  So both
roots hold the same stubs with different values, and the comparisons are made
against each of them in turn: a query that dropped the root answers out of
whichever row the server reached first, and one of the two directions catches it.

**Neither backend's order is trusted.**  Both streams are sorted in Python
before anything is compared, on the whole key, and so are the child rows within
one image, because the index stores no ordinal for a technique and a server
returns rows of one key in whatever order it likes.
"""

import builtins
import contextlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from sqlalchemy.engine import Engine
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import index_url, ingest_tree, metadata_document, write_metadata

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    ImageFacts,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)
from spindoctor.results_index import IMAGES, open_index, open_record_source

POSTGRES_URL_ENV_VAR = 'SPINDOCTOR_TEST_POSTGRES_URL'

ROOT_URL = 'file:///data/nav-results'

STUB = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'

AT_SIGN_USER = 'admin@pgsrv'
"""A user name carrying an at-sign, which is how a managed server names one.

``user@servername`` is the standard login form of a hosted PostgreSQL, and
SQLAlchemy's own parser accepts it. A rule that took the first at-sign as the
end of the credentials would find no password after it and leak the whole URL.
"""


EXPLODING_FACTORY_MESSAGE = 'the dialect exploded'
"""What the stand-in engine factory raises, standing for any escape from one."""


def sqlite_url_for(path: Path) -> str:
    """Return the SQLite URL naming a filesystem path.

    Parameters:
        path: The database file's path.

    Returns:
        The URL.
    """
    # as_posix rather than str: SQLAlchemy takes a URL, and a Windows path
    # separator in one is not a path separator.
    return f'sqlite:///{path.as_posix()}'


def without_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make one module unimportable, as it is on a machine without it installed.

    Asserting on a driver that merely happens to be absent from the current
    virtual environment is a test that stops testing the moment something pulls
    that driver in as a transitive dependency.

    Parameters:
        monkeypatch: Fixture the import hook is installed through.
        name: Dotted name of the module to hide, together with its submodules.
    """
    real_import = builtins.__import__

    def blocked(module_name: str, *args: Any, **kwargs: Any) -> Any:
        if module_name == name or module_name.startswith(f'{name}.'):
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)


def exploding_factory(*args: Any, **kwargs: Any) -> Engine:
    """Stand in for an engine factory that fails in a way nobody enumerated.

    A dialect coerces its own connect arguments and reports a bad one as a bare
    exception naming nothing, so the translation has to be a catch-all rather
    than a list of types.

    Parameters:
        args: Whatever the caller passed, all of it ignored.
        kwargs: Whatever the caller passed, all of it ignored.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(EXPLODING_FACTORY_MESSAGE)


def image_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``images`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to an ``images`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'image_name': 'N1294561202_1_CALIB.IMG',
        'instrument': 'COISS',
        'status': 'success',
        'n_techniques': 2,
    }
    row.update(overrides)
    return row


def technique_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``techniques`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to a ``techniques`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'technique_name': 'star_field_from_catalog',
        'spurious': False,
        'at_edge': False,
    }
    row.update(overrides)
    return row


def feature_source_row(**overrides: Any) -> dict[str, Any]:
    """Return a minimally valid ``feature_sources`` row, with overrides applied.

    Parameters:
        overrides: Column values replacing (or adding to) the defaults.

    Returns:
        A mapping ready to pass to a ``feature_sources`` insert.
    """
    row: dict[str, Any] = {
        'root_url': ROOT_URL,
        'results_path_stub': STUB,
        'feature_type': 'STAR',
        'source_model': 'NavModelStars',
        'source_name': 'UCAC4',
        'n_features': 41,
        'n_gated': 3,
    }
    row.update(overrides)
    return row


@contextlib.contextmanager
def opened(url: str, *, create: bool = False) -> Iterator[Engine]:
    """Open an index and dispose of the engine afterwards.

    Parameters:
        url: The connection URL to open.
        create: Whether to create missing tables and the version row.

    Yields:
        The open engine.
    """
    engine = open_index(url, create=create)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def postgres_server_url() -> str:
    """Return the PostgreSQL URL the postgres tier runs against.

    Returns:
        The URL named by the environment.
    """
    url = os.environ.get(POSTGRES_URL_ENV_VAR)
    if url is None:
        pytest.skip(f'{POSTGRES_URL_ENV_VAR} is not set')
    return url


@pytest.fixture
def postgres_schema() -> str:
    """Return the name of the private schema this test's tables live in.

    Requested alongside ``postgres_url`` by a test that reads the server's
    catalog, where a query scoped by table name alone would answer from whatever
    schema happened to hold a table of that name -- another worker's, or a
    leftover in ``public``.

    Returns:
        A schema name no other test uses.
    """
    return f'ri_test_{uuid.uuid4().hex}'


@pytest.fixture
def postgres_decoy_schema() -> str:
    """Return the name of a second schema, behind this test's own on the path.

    A search path of one entry is the one shape in which every unqualified table
    name resolves to the same schema whatever the code does with it, so a
    fixture that pins one cannot see a query that resolves two names into two
    schemas.  Every postgres test therefore runs with a schema behind its own.

    Returns:
        A schema name no other test uses.
    """
    return f'ri_decoy_{uuid.uuid4().hex}'


DECOY_TABLE = 'customers'
"""What the decoy schema holds: a table of a name the index never uses.

Deliberately not one of the index's own names, so that the decoy makes the
search path longer than one entry without also making a bare ``images``
resolve into it.  A creating open binds the schema it builds in and does not
adopt a table of one of its names from anywhere else, and the drop reports
every table of those names its connection reaches; the tests that need either
of those build the collision themselves.
"""


@pytest.fixture
def postgres_url(
    postgres_server_url: str, postgres_schema: str, postgres_decoy_schema: str
) -> Iterator[str]:
    """Yield a PostgreSQL URL scoped to a schema of this test's own.

    The schemas are created before the test and dropped after it, so repeated
    runs and parallel workers never see one another's tables.  The index's own
    schema leads the search path, so that is where a creating open builds it;
    the decoy behind it is what makes the path more than one entry long.

    Parameters:
        postgres_server_url: The server URL the schemas are created on.
        postgres_schema: Name of the schema to create for the index.
        postgres_decoy_schema: Name of the schema to create behind it.

    Yields:
        A URL whose search path names the private schema and then the decoy.
    """
    schema = postgres_schema
    decoy = postgres_decoy_schema
    scoped = sqlalchemy.engine.make_url(postgres_server_url).update_query_dict(
        {'options': f'-csearch_path={schema},{decoy}'}
    )
    admin = sqlalchemy.create_engine(postgres_server_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            connection.exec_driver_sql(f'CREATE SCHEMA "{decoy}"')
            connection.exec_driver_sql(f'CREATE TABLE "{decoy}".{DECOY_TABLE} (x INTEGER)')
        try:
            yield scoped.render_as_string(hide_password=False)
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
                connection.exec_driver_sql(f'DROP SCHEMA "{decoy}" CASCADE')
    finally:
        admin.dispose()


def url_scoped_to(server_url: str, *schemas: str) -> str:
    """Return the server URL with a search path naming these schemas in order.

    Parameters:
        server_url: The server URL, unscoped.
        schemas: The schemas to name, first one first.

    Returns:
        The scoped URL, with its password intact so it can be opened.
    """
    scoped = sqlalchemy.engine.make_url(server_url).update_query_dict(
        {'options': f'-csearch_path={",".join(schemas)}'}
    )
    return scoped.render_as_string(hide_password=False)


MISSION = 'coiss'
"""The instrument identity a mission-filtered read of these roots keeps."""

OTHER_MISSION = 'vgiss'
"""An instrument identity of another mission's documents in the same tree."""

COLUMNS = (IMAGES.c.status, IMAGES.c.offset_dv, IMAGES.c.offset_du)
"""A consumer's columns, which narrow a record and must not narrow the facts."""

OTHER_MISSION_STUB = 'VOL0/C1454725_CALIB'
"""An image of another mission, sorting ahead of every image a filter keeps.

Ahead deliberately: a mission-filtered stream drops it, and its technique rows
are then the first rows a child stream would meet.  A merge that did not restrict
the children to the images it yields would stall on them and hand every later
image no techniques at all.
"""

SUCCESS_STUB = 'VOL1/N1454725799_1_CALIB'
"""A navigated image, carrying everything a fully populated document carries."""

FAILURE_STUB = 'VOL1/N1454725800_1_CALIB'
"""An image the navigation reached and did not solve."""

ERROR_STUB = 'VOL2/N1454725801_1_CALIB'
"""An image whose run ended in a fatal error, naming it."""

UNLOADED_STUB = 'VOL2/N1454725802_1_CALIB'
"""An image that never loaded: no navigation result, and no outcome named.

It records no epoch either, because an epoch is the midtime of the observation
the load never built, so it is the shape that leaves the epoch columns NULL.
"""

REFUSED_STUB = 'VOL1/junk'
"""A JSON object of some other tool's shape, which no pass reads as a document.

The second root holds a navigable document at this same stub, so a refusal
excluded by the presence of a record under the *other* root would go unreported
here.
"""

TORN_STUB = 'VOL2/torn'
"""A file no JSON value comes out of at all."""

EXTRA_STUB = 'VOL2/N9999999999_1_CALIB'
"""An image only the second root holds, so a root-blind stream yields too much."""

OTHER_ROOT_REFUSED_STUB = 'VOL2/junk_second'
"""A file only the second root refused, at a stub the first root has nothing at.

Its own stub rather than a shared one: a read of the first root naming it has to
come back empty, and a stub both roots held would come back full whether or not
the refusal query knew about the root.
"""

REFUSED_DOCUMENT = '{"edges": []}'
"""What sits at the refused stub."""

TORN_DOCUMENT = '{"observation":'
"""What sits at the torn stub."""

NESTED_STUB = 'VOL2/nested'
"""A file whose nesting a decoder may give up on rather than fail to parse.

Its own class of refusal: how deep a decoder will follow nesting, and whether it
reports giving up as a decoding error or as something else entirely, is that
decoder's business.  Whatever it does, no value came out of the file, and the
two storages have to say that in the same words -- one of them naming the
exception it met and the other not is a difference an operator reads as the two
disagreeing about the file.
"""

NESTED_DOCUMENT = '{"a":' * 20000
"""What sits at the nested stub: twenty thousand opening braces and nothing else."""


@dataclass(frozen=True)
class RootValues:
    """What one root's documents record where the other root's record something else.

    Every value read off these roots is in here, so a query that dropped the root
    half of the key reads the wrong one of these rather than reading nothing.

    Parameters:
        offset: The authoritative offset every image of this root records.
        twist_covariance: The 3x3 covariance the navigated image records, whose
            rotation row and column no per-axis sigma states.
        technique_covariance: The 2x2 covariance each of its techniques records.
        excluded: The technique names the ensemble excluded, in an order that
            is not their sorted order, so a storage that sorted them is caught.
        camera_frame_id: The frame identifier the navigated image records, which
            the two roots record differently.
        midtime: The exposure midtime the navigated image records.
    """

    offset: list[float]
    twist_covariance: list[list[float]]
    technique_covariance: list[list[float]]
    excluded: list[str]
    camera_frame_id: int
    midtime: float


FIRST_VALUES = RootValues(
    offset=[1.5, -2.5],
    twist_covariance=[
        [0.010, 0.002, 0.0003],
        [0.002, 0.040, 0.0005],
        [0.0003, 0.0005, 0.000001],
    ],
    technique_covariance=[[0.11, 0.012], [0.012, 0.13]],
    excluded=['StarRefineNav', 'BodyLimbNav', 'RingEdgeNav'],
    camera_frame_id=-82360,
    midtime=100.0,
)
"""What the first root's documents record."""

SECOND_VALUES = RootValues(
    offset=[9.5, -8.5],
    twist_covariance=[
        [0.090, 0.008, 0.0007],
        [0.008, 0.070, 0.0009],
        [0.0007, 0.0009, 0.000004],
    ],
    technique_covariance=[[0.21, 0.022], [0.022, 0.23]],
    excluded=['RingEdgeNav', 'StarRefineNav', 'BodyLimbNav'],
    camera_frame_id=-84001,
    midtime=300.0,
)
"""What the second root's record instead, so the two are told apart by value."""

IDENTITY_CMATRIX = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
"""A recorded rotation, of no astronomical significance."""


def _technique(name: str, offset: list[float], values: RootValues) -> dict[str, Any]:
    """Build one ``per_technique`` entry.

    Parameters:
        name: The technique's class name, which is the identity of its row.
        offset: Its own estimate.
        values: The root's differing values, for the covariance it records.

    Returns:
        The entry.
    """
    return {
        'technique_name': name,
        'feature_ids': [f'{name.lower()}:IAPETUS'],
        'offset_px': offset,
        'covariance_px2': values.technique_covariance,
        'confidence': 0.7,
        'spurious': False,
        'at_edge': False,
        'diagnostics': {'iterations': 4},
    }


def _success_document(values: RootValues) -> dict[str, Any]:
    """Build the navigated image's document.

    Parameters:
        values: The root's differing values.

    Returns:
        The document.
    """
    document = metadata_document(
        image_name='N1454725799_1.IMG',
        instrument=MISSION,
        offset=values.offset,
        excluded=values.excluded,
        image_shape=[1024, 1024],
        per_technique=[
            _technique('StarFieldFromCatalogNav', [1.0, 2.0], values),
            _technique('BodyLimbNav', [1.1, 2.1], values),
        ],
        times={
            'midtime_et': values.midtime,
            'start_et': values.midtime - 1.0,
            'stop_et': values.midtime + 1.0,
            'exposure_s': 2.0,
        },
        pointing={
            'camera_frame': 'CASSINI_ISS_NAC',
            'camera_frame_id': values.camera_frame_id,
            'ck_frame_id': -82000,
            'cmatrix': IDENTITY_CMATRIX,
            'cmatrix_original': IDENTITY_CMATRIX,
        },
    )
    # A twist-fitted result, whose rotation row and column carry the terms a
    # per-axis sigma cannot state.  The builder writes a 2x2, so it is replaced
    # here rather than parameterized into a helper nothing else needs.
    document['navigation_result']['covariance_px2'] = values.twist_covariance
    document['navigation_result']['rotation_deg'] = 0.25
    document['navigation_result']['sigma_rotation_deg'] = 0.01
    return document


def _failure_document(values: RootValues) -> dict[str, Any]:
    """Build the document of an image the navigation did not solve.

    Parameters:
        values: The root's differing values, for the epoch it records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1454725800_1.IMG',
        instrument=MISSION,
        status='failed',
        status_reason='no technique produced a usable offset',
        offset=None,
        times={'midtime_et': values.midtime + 10.0},
    )


def _error_document(values: RootValues) -> dict[str, Any]:
    """Build the document of a run that ended in a fatal error.

    Parameters:
        values: The root's differing values, for the epoch it records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1454725801_1.IMG',
        instrument=MISSION,
        status='error',
        status_error='SPICE(SPKINSUFFDATA)',
        status_reason=None,
        offset=None,
        times={'midtime_et': values.midtime + 20.0},
    )


def _unloaded_document() -> dict[str, Any]:
    """Build the document of an image that never loaded.

    Returns:
        The document, which names no outcome and carries no navigation result,
        so it records neither an epoch nor anything a technique produced.
    """
    return {
        'observation': {
            'image_name': 'N1454725802_1.IMG',
            'instrument': MISSION,
            'image_path': '/holdings/N1454725802_1.IMG',
        }
    }


def _other_mission_document(values: RootValues) -> dict[str, Any]:
    """Build a document of the mission a filtered read drops.

    Parameters:
        values: The root's differing values.

    Returns:
        The document, which carries technique rows of its own so that dropping
        the image has to drop them too.
    """
    return metadata_document(
        image_name='C1454725.IMG',
        instrument=OTHER_MISSION,
        offset=values.offset,
        per_technique=[_technique('StarUniqueMatchNav', [3.0, 4.0], values)],
        times={'midtime_et': values.midtime},
    )


def _write_text(root: Path, stub: str, text: str) -> None:
    """Write one file where a navigation document belongs, whatever it holds.

    Parameters:
        root: The results root.
        stub: The document's results path stub under it.
        text: What to write.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _write_root(tmp_path: Path, name: str, values: RootValues, *, extra: bool) -> Path:
    """Write one results root holding every kind of file the tests read.

    Parameters:
        tmp_path: Directory the root is written under.
        name: The root's directory name.
        values: The values this root's documents record.
        extra: Whether this root also holds an image the other does not, and a
            navigable document where the other holds a refusal.

    Returns:
        The results root.
    """
    root = tmp_path / name
    write_metadata(root, OTHER_MISSION_STUB, _other_mission_document(values))
    write_metadata(root, SUCCESS_STUB, _success_document(values))
    write_metadata(root, FAILURE_STUB, _failure_document(values))
    write_metadata(root, ERROR_STUB, _error_document(values))
    write_metadata(root, UNLOADED_STUB, _unloaded_document())
    _write_text(root, TORN_STUB, TORN_DOCUMENT)
    _write_text(root, NESTED_STUB, NESTED_DOCUMENT)
    if extra:
        write_metadata(root, EXTRA_STUB, _failure_document(values))
        write_metadata(root, REFUSED_STUB, _failure_document(values))
        _write_text(root, OTHER_ROOT_REFUSED_STUB, REFUSED_DOCUMENT)
    else:
        _write_text(root, REFUSED_STUB, REFUSED_DOCUMENT)
    return root


class TwoRoots:
    """Two results roots differing in every value read, and the index of both.

    Parameters:
        first: The root whose documents record the first set of values.
        second: The root whose documents record the second set, holding one
            image the first does not and a navigable document where the first
            holds a refusal.
        url: The index both were ingested into.
    """

    def __init__(self, first: Path, second: Path, url: str) -> None:
        self.first = first
        self.second = second
        self.url = url

    def root(self, which: str) -> Path:
        """Return one of the two roots by name.

        Parameters:
            which: ``'first'`` or ``'second'``.

        Returns:
            That root.
        """
        return self.first if which == 'first' else self.second

    def values(self, which: str) -> RootValues:
        """Return the values one of the two roots was written to record.

        Parameters:
            which: ``'first'`` or ``'second'``.

        Returns:
            What every document under that root records, so a read parametrized
            over both roots names the values it expects rather than repeating
            the pairing at each call.
        """
        return FIRST_VALUES if which == 'first' else SECOND_VALUES


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that writes nowhere.

    Returns:
        The logger, so an ingest driven by a test says nothing.
    """
    return pdslogger.NullLogger()


@pytest.fixture
def two_roots(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> TwoRoots:
    """Write two results roots and ingest both into one index.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
        quiet_logger: Logger the ingest reports through.

    Returns:
        The two roots and the index holding both.
    """
    first = _write_root(tmp_path, 'results-first', FIRST_VALUES, extra=False)
    second = _write_root(tmp_path, 'results-second', SECOND_VALUES, extra=True)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    return TwoRoots(first, second, url)


BOTH_ROOTS = [pytest.param('first', id='first-root'), pytest.param('second', id='second-root')]
"""Read every comparison against each root, so a root-blind query fails one."""


def stub_of(found: ImageFacts | UnreadableFile) -> str:
    """Return the stub of one thing a facts stream yielded.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The image's results path stub, which both shapes carry.
    """
    if isinstance(found, UnreadableFile):
        return found.stub
    return str(found.image['results_path_stub'])


def technique_key(row: dict[str, Any]) -> str:
    """Return what identifies one technique row within its image.

    Parameters:
        row: The row.

    Returns:
        The technique's name.
    """
    return str(row['technique_name'])


def _root_of(found: ImageFacts | UnreadableFile) -> str:
    """Return the root one thing a facts stream yielded is held under.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The root URL.  An image carries it as a column of its own; a file no
        facts came out of carries only where its document is, which is that
        root joined to the stub it is a key under.
    """
    if isinstance(found, UnreadableFile):
        return str(found.path)[: -len(f'/{found.stub}{METADATA_SUFFIX}')]
    return str(found.image['root_url'])


def key_of(found: ImageFacts | UnreadableFile) -> tuple[str, str]:
    """Return the whole key of one thing a facts stream yielded.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The root URL and the results path stub.
    """
    return _root_of(found), stub_of(found)


def _in_order(found: Any) -> list[ImageFacts | UnreadableFile]:
    """Return what a stream yielded, sorted by the whole key in Python.

    Neither storage promises an order, and a server sorts text under its own
    collation, so the comparison is made in one order that belongs to neither.
    On the whole key rather than on the stub: a stream over two roots yields one
    stub twice, and a sort on half a key leaves those two in whichever order
    they arrived, which is a property of the backend rather than of the answer.

    Parameters:
        found: What the stream yielded.

    Returns:
        The same things, in key order.
    """
    return sorted(found, key=key_of)


def facts_from_tree(
    held: TwoRoots, which: str, selection: Selection
) -> list[ImageFacts | UnreadableFile]:
    """Read the facts of one root out of its documents.

    Parameters:
        held: The two roots and their index.
        which: Which root to read.
        selection: What to read.

    Returns:
        What the stream yielded, in stub order.
    """
    return _in_order(
        TreeRecordSource([held.root(which)], tuning=packaged_tree_tuning()).facts(selection)
    )


def facts_from_index(
    held: TwoRoots, which: str, selection: Selection
) -> list[ImageFacts | UnreadableFile]:
    """Read the facts of one root out of the index both roots were ingested into.

    Parameters:
        held: The two roots and their index.
        which: Which root to read.
        selection: What to read.

    Returns:
        What the stream yielded, in stub order.
    """
    with open_record_source(
        [held.root(which)],
        results_index_db_url=held.url,
        columns=COLUMNS,
        tuning=packaged_tree_tuning(),
    ) as source:
        return _in_order(source.facts(selection))


def named_facts_from_index(
    held: TwoRoots, which: str, selection: Selection
) -> list[ImageFacts | UnreadableFile]:
    """Read one root's facts out of the index for a selection naming its stubs.

    Sorted by nothing, unlike :func:`facts_from_index`: a read naming its own
    stubs is answered in the order it named them, and a sort of what came back
    is the one thing that could hide a read that was not.

    Parameters:
        held: The two roots and their index.
        which: Which root to read.
        selection: What to read, naming the stubs to read.

    Returns:
        What the stream yielded, in the order it yielded it.
    """
    with open_record_source(
        [held.root(which)],
        results_index_db_url=held.url,
        columns=COLUMNS,
        tuning=packaged_tree_tuning(),
    ) as source:
        return list(source.facts(selection))


def facts_of(found: list[ImageFacts | UnreadableFile], stub: str) -> ImageFacts:
    """Return one image's facts out of what a stream yielded.

    Parameters:
        found: What the stream yielded.
        stub: The image to pick out.

    Returns:
        Its facts.
    """
    picked = [one for one in found if isinstance(one, ImageFacts) and stub_of(one) == stub]
    assert len(picked) == 1
    return picked[0]


def child_name(root_url: str, stub: str) -> str:
    """Return a technique name naming the image it belongs to.

    Parameters:
        root_url: The root half of the image's key.
        stub: The stub half.

    Returns:
        A name no other image's row carries, so a mis-paired row is visible.
    """
    return f'{Path(root_url).name}:{stub}'
