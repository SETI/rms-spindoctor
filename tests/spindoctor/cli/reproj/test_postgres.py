"""Reading a navigation record from an index held on a real PostgreSQL server.

SQLite stores a JSON column as text, so a matrix written into one comes back as
the characters that were written.  PostgreSQL stores it as ``jsonb``, whose
numbers are decimal and are re-rendered on the way out, and a double is a native
type rather than whatever the value looked like.  A recorded attitude that
survives SQLite therefore proves nothing about the backend an index shared
across machines actually runs on, and the reader's flip gate holds the recovered
rotation to 1e-9.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset.  Each test gets a schema of its
own from the ``postgres_url`` fixture, so a repeated run, or two workers of a
parallel run, never share a table.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.reproj.conftest import (
    CMATRIX,
    CMATRIX_ORIGINAL,
    CMATRIX_STUB,
    MIDTIME_ET,
    OFFSET,
    POINTING,
    TIMES,
    UNNAVIGATED_STUB,
    build_tree,
    document,
    image_file,
)

from spindoctor.cli.reproj.offsets import PointingMechanism
from spindoctor.cli.reproj.pointing_source import IndexPointingSource
from spindoctor.cli.results_index import ingest_metadata_files
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import normalize_root_url, open_index

pytestmark = pytest.mark.postgres


def _ingested(url: str, roots: list[Path], *, logger: pdslogger.PdsLogger) -> None:
    """Create the schema on the server and ingest the given trees into it.

    Parameters:
        url: The PostgreSQL URL, scoped to this test's own schema.
        roots: The results roots to walk.
        logger: Logger the ingest reports through.
    """
    engine = open_index(url, create=True)
    try:
        ingest_metadata_files(
            engine, [root.as_posix() for root in roots], logger=logger, tuning=TreeTuning()
        )
    finally:
        engine.dispose()


@pytest.fixture
def cmatrix_source(
    tmp_path: Path, postgres_url: str, quiet_ingest_logger: pdslogger.PdsLogger
) -> Iterator[IndexPointingSource]:
    """Yield a source over one server-held record carrying a corrected attitude.

    Parameters:
        tmp_path: Directory the results root is written under.
        postgres_url: This test's own schema on the server.
        quiet_ingest_logger: Logger the ingest reports through.

    Yields:
        The source, whose engine is disposed of afterwards.
    """
    root = tmp_path / 'nav'
    build_tree(
        root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET, times=TIMES, pointing=POINTING)}
    )
    _ingested(postgres_url, [root], logger=quiet_ingest_logger)
    engine = open_index(postgres_url)
    try:
        yield IndexPointingSource(engine, normalize_root_url(root))
    finally:
        engine.dispose()


@pytest.fixture
def two_root_sources(
    tmp_path: Path, postgres_url: str, quiet_ingest_logger: pdslogger.PdsLogger
) -> Iterator[tuple[IndexPointingSource, IndexPointingSource]]:
    """Yield two sources over one index holding the same stub under two roots.

    The two documents differ in exactly the offset each records, so a query
    that dropped the root half of its key would answer one of the two tests
    below with the other's value.

    Parameters:
        tmp_path: Directory both roots are written under.
        postgres_url: This test's own schema on the server.
        quiet_ingest_logger: Logger the ingest reports through.

    Yields:
        The source for the first root and the source for the second.
    """
    first = tmp_path / 'nav_a'
    second = tmp_path / 'nav_b'
    build_tree(first, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[1.5, -2.5])})
    build_tree(second, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[9.25, 8.75])})
    _ingested(postgres_url, [first, second], logger=quiet_ingest_logger)
    engine = open_index(postgres_url)
    try:
        yield (
            IndexPointingSource(engine, normalize_root_url(first)),
            IndexPointingSource(engine, normalize_root_url(second)),
        )
    finally:
        engine.dispose()


def test_the_recorded_attitude_survives_jsonb_bit_for_bit(
    cmatrix_source: IndexPointingSource,
) -> None:
    """A rotation of full-mantissa float64 comes back exactly as recorded.

    Parameters:
        cmatrix_source: The source over the server-held record.
    """
    selection = cmatrix_source.load_pointing(image_file(CMATRIX_STUB))
    assert selection.cmatrix is not None
    assert np.array_equal(selection.cmatrix, np.asarray(CMATRIX).reshape(3, 3))


def test_the_recorded_baseline_survives_jsonb_bit_for_bit(
    cmatrix_source: IndexPointingSource,
) -> None:
    """So does the as-flown attitude the flip gate is computed against.

    Parameters:
        cmatrix_source: The source over the server-held record.
    """
    selection = cmatrix_source.load_pointing(image_file(CMATRIX_STUB))
    assert selection.cmatrix_original is not None
    assert np.array_equal(selection.cmatrix_original, np.asarray(CMATRIX_ORIGINAL).reshape(3, 3))


def test_the_recorded_midtime_survives_a_native_double(
    cmatrix_source: IndexPointingSource,
) -> None:
    """The epoch the reader gates to a microsecond comes back exactly.

    Parameters:
        cmatrix_source: The source over the server-held record.
    """
    assert cmatrix_source.load_pointing(image_file(CMATRIX_STUB)).midtime_et == MIDTIME_ET


def test_the_selection_still_takes_the_cmatrix_mechanism(
    cmatrix_source: IndexPointingSource,
) -> None:
    """Which is what a matrix that failed to round-trip would have cost.

    Parameters:
        cmatrix_source: The source over the server-held record.
    """
    selection = cmatrix_source.load_pointing(image_file(CMATRIX_STUB))
    assert selection.mechanism is PointingMechanism.CMATRIX


def test_a_lookup_answers_from_its_own_root(
    two_root_sources: tuple[IndexPointingSource, IndexPointingSource],
) -> None:
    """One index holds several roots, and each consumer sees only its own.

    Parameters:
        two_root_sources: The sources for the two ingested roots.
    """
    assert two_root_sources[0].load_pointing(image_file(CMATRIX_STUB)).offset == (1.5, -2.5)


def test_the_other_root_answers_for_itself(
    two_root_sources: tuple[IndexPointingSource, IndexPointingSource],
) -> None:
    """The other direction of the same assertion, which one row cannot satisfy.

    Parameters:
        two_root_sources: The sources for the two ingested roots.
    """
    assert two_root_sources[1].load_pointing(image_file(CMATRIX_STUB)).offset == (9.25, 8.75)


@pytest.fixture
def missing_row_message(
    tmp_path: Path, postgres_url: str, quiet_ingest_logger: pdslogger.PdsLogger
) -> str:
    """Return what the refusal says when the index holds no row for an image.

    Parameters:
        tmp_path: Directory the results root is written under.
        postgres_url: This test's own schema on the server, carrying the
            server's credentials.
        quiet_ingest_logger: Logger the ingest reports through.

    Returns:
        The refusal message.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    _ingested(postgres_url, [root], logger=quiet_ingest_logger)
    engine = open_index(postgres_url)
    try:
        source = IndexPointingSource(engine, normalize_root_url(root))
        with pytest.raises(FileNotFoundError) as excinfo:
            source.read_record(image_file(UNNAVIGATED_STUB))
        return str(excinfo.value)
    finally:
        engine.dispose()


def test_the_missing_row_message_hides_the_index_password(
    postgres_url: str, missing_row_message: str
) -> None:
    """The refusal names the index it asked without naming its password.

    It reaches a run log and whoever is sent one, and a connection URL to a
    server carries credentials that a SQLite path never does.  The authority
    form is what is looked for rather than the password on its own: this
    server's password is also its user name and its database name, so a bare
    substring search would report a leak that is not one, or miss one that is.

    Parameters:
        postgres_url: This test's own schema on the server.
        missing_row_message: What the refusal said.
    """
    parsed = sqlalchemy.engine.make_url(postgres_url)
    assert f'{parsed.username}:{parsed.password}@' not in missing_row_message


def test_the_missing_row_message_masks_where_the_password_was(
    postgres_url: str, missing_row_message: str
) -> None:
    """A guard on the assertion above, which a blank message would satisfy.

    Parameters:
        postgres_url: This test's own schema on the server.
        missing_row_message: What the refusal said.
    """
    parsed = sqlalchemy.engine.make_url(postgres_url)
    assert f'{parsed.username}:***@' in missing_row_message


def test_the_missing_row_message_still_names_the_index(
    postgres_url: str, missing_row_message: str
) -> None:
    """Everything but the credentials survives.

    Which of the three resolution levels supplied the URL is exactly what a
    reader of a failed run needs, and the host is what says which.

    Parameters:
        postgres_url: This test's own schema on the server.
        missing_row_message: What the refusal said.
    """
    parsed = sqlalchemy.engine.make_url(postgres_url)
    assert str(parsed.host) in missing_row_message
