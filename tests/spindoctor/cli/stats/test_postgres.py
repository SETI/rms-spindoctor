"""The statistics programs against a real PostgreSQL server.

SQLite accepts almost anything: it will subtract a boolean from an integer,
count distinct values of whatever it is handed, and treat a JSON column as
text.  A report that runs on SQLite therefore proves very little about the
backend an index shared across machines actually runs on.  These re-run the
frozen comparison, and the ingest that feeds it, against a server that enforces
the type discipline the schema declares.

The tier is opt-in: it is excluded by the default marker filter and skips itself
when ``SPINDOCTOR_TEST_POSTGRES_URL`` is unset, so a checkout with no server
still runs a green suite.  Each test gets a schema of its own from the
``postgres_url`` fixture, so a repeated run, or two workers of a parallel run,
never share a table.
"""

import csv
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.stats.conftest import (
    GOLDEN_DIR,
    GOLDEN_VARIANTS,
    build_tree,
    complete,
    fan_out,
    report_from_the_index,
    reported,
    run_rows,
)
from tests.spindoctor.conftest import (
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
)

from spindoctor.cli.results_index import (
    TaskResult,
    UnwritableRowError,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_task_share,
)
from spindoctor.cli.results_index.tasks import _LARGEST_RUN_ROW_COUNT
from spindoctor.cli.stats.report import main_report
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    TECHNIQUES,
    normalize_root_url,
    open_index,
    require_ingested_roots,
)

pytestmark = pytest.mark.postgres

_FULL_VARIANT: dict[str, Any] = GOLDEN_VARIANTS['full']
"""The unfiltered report invocation the frozen ``full`` output was produced by.

Read off the shared definition rather than restated beside it.  Every comparison
below is against output frozen for that variant, so a restatement that drifted
from it would have this module produce a report under one set of options and
measure it against output produced under another, and fail for a reason that has
nothing to do with the backend it is here to exercise.
"""

_TWIST_COVARIANCE = [
    [0.0961, 0.0100, 0.0025],
    [0.0100, 0.0784, -0.0050],
    [0.0025, -0.0050, 0.0009],
]
"""A 3x3 covariance of the kind a twist-fitted result records."""


def test_the_report_is_byte_identical_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same tree, the same report, on the backend the queries were written for."""
    out = report_from_the_index(
        postgres_url, tmp_path / 'full', logger=quiet_logger, **_FULL_VARIANT
    )
    frozen = (GOLDEN_DIR / 'full' / 'report.md').read_text(encoding='utf-8')
    assert (out / 'report.md').read_text(encoding='utf-8') == frozen


def test_the_csv_carries_the_same_images_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The export covers the same images, on the same key.

    Compared as sorted names: the rows are written where they are read, and a
    server sorts the key under its own collation, so line order is not something
    two backends are asked to agree on.
    """
    out = report_from_the_index(
        postgres_url, tmp_path / 'full', logger=quiet_logger, **_FULL_VARIANT
    )
    produced = sorted(
        row['image_name']
        for row in csv.DictReader((out / 'images.csv').read_text(encoding='utf-8').splitlines())
    )
    frozen_text = (GOLDEN_DIR / 'full' / 'images.csv').read_text(encoding='utf-8')
    frozen = sorted(row['image_name'] for row in csv.DictReader(frozen_text.splitlines()))
    assert produced == frozen


def test_the_json_columns_round_trip_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A JSONB column decodes to a Python value, and the CSV re-encodes one."""
    out = report_from_the_index(
        postgres_url, tmp_path / 'full', logger=quiet_logger, **_FULL_VARIANT
    )
    frozen_text = (GOLDEN_DIR / 'full' / 'images.csv').read_text(encoding='utf-8')
    frozen = {
        row['image_name']: row['excluded_from_consensus']
        for row in csv.DictReader(frozen_text.splitlines())
    }
    produced = {
        row['image_name']: row['excluded_from_consensus']
        for row in csv.DictReader((out / 'images.csv').read_text(encoding='utf-8').splitlines())
    }
    assert produced == frozen


def test_a_covariance_matrix_round_trips_through_jsonb(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A nested array of numbers is the shape most at risk from a backend.

    JSONB stores a number as an arbitrary-precision numeric and hands it back
    through the driver's own decoder, so a matrix that survives SQLite's text
    column proves nothing about the server the index is shared on.
    """
    root = tmp_path / 'results'
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = _TWIST_COVARIANCE
    write_metadata(root, 'VOL/N1454725799_1_CALIB', document)
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    try:
        with engine.connect() as connection:
            stored = connection.execute(sqlalchemy.select(IMAGES.c.covariance_px2)).scalar()
    finally:
        engine.dispose()
    assert stored == _TWIST_COVARIANCE


def test_the_exclusion_order_round_trips_through_jsonb(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """JSONB reorders the keys of an object; an array keeps its order.

    The column holds the list in the order the document recorded it, so a
    backend that sorted or re-ordered it would answer with a list no document
    holds.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(excluded=['StarRefineNav', 'BodyBlobNav', 'RingEdgeNav']),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    try:
        with engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.select(IMAGES.c.excluded_from_consensus)
            ).scalar()
    finally:
        engine.dispose()
    assert stored == ['StarRefineNav', 'BodyBlobNav', 'RingEdgeNav']


def test_the_unrounded_offset_survives_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Fifteen significant digits through a double-precision column."""
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(offset=[3.14159265358979, -2.71828182845905]),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(IMAGES.c.offset_dv)))
    engine.dispose()
    assert found[0][0] == 3.14159265358979


def test_a_boolean_column_holds_a_boolean_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An integer flag would be a type error the moment the report read it."""
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0), spurious=True)]),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(TECHNIQUES.c.spurious)))
    engine.dispose()
    assert found[0][0] is True


def test_an_unchanged_file_is_skipped_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The incremental skip reads the metrics it stored, on either backend."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    counts = ingest_tree(postgres_url, [root], logger=quiet_logger)
    assert counts.files_skipped == 1


def test_two_volumes_with_one_basename_produce_two_rows_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The composite primary key is a server-enforced constraint here."""
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document())
    write_metadata(root, 'COISS_2002/data/N1294561202_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(
            connection.execute(sqlalchemy.select(IMAGES.c.subtree).order_by(IMAGES.c.subtree))
        )
    engine.dispose()
    assert [row.subtree for row in found] == ['COISS_2001', 'COISS_2002']


def test_an_absent_cmatrix_is_sql_null_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A jsonb column holding the JSON value null satisfies IS NOT NULL on every row.

    That is the whole failure: a direct-SQL user asking which images carry a
    corrected attitude gets all of them.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(
            connection.execute(
                sqlalchemy.select(sqlalchemy.func.count())
                .select_from(IMAGES)
                .where(IMAGES.c.cmatrix.is_(None))
            )
        )
    engine.dispose()
    assert found[0][0] == 1


def test_a_recorded_cmatrix_is_not_sql_null_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the reverse, so the query tells the two cases apart in both directions."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', document)
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(IMAGES.c.cmatrix)))
    engine.dispose()
    assert found[0][0] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_an_empty_exclusion_set_is_stored_as_an_empty_list_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing excluded is a statement, and must not become nothing recorded."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document(excluded=[]))
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(IMAGES.c.excluded_from_consensus)))
    engine.dispose()
    assert found[0][0] == []


def test_a_refused_file_is_skipped_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The refusal bookkeeping is a table like any other, and the server holds it."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    counts = ingest_tree(postgres_url, [root], logger=quiet_logger)
    assert (counts.files_seen, counts.files_skipped) == (2, 2)


def test_a_deleted_document_loses_its_row_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The prune's delete cascades on a server that actually enforces the key."""
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))]),
    )
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    (root / 'VOL' / 'N1454725799_1_CALIB_metadata.json').unlink()
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    engine = open_index(postgres_url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(TECHNIQUES.c.technique_name)))
    engine.dispose()
    assert found == []


def test_the_report_cli_does_not_print_the_server_password(
    postgres_url: str,
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one refusal this program makes that names its index rather than a file.

    A SQLite URL has no password to leak, so this route can only be driven on a
    server, and the server URL a real deployment resolves carries one.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    ingest_tree(postgres_url, [root], logger=quiet_logger)
    parsed = sqlalchemy.engine.make_url(postgres_url)
    # The credentials, not the password alone: a deployment is free to name its
    # user, its database and its password the same word, and this one does, so
    # the bare password appears in the message for reasons that are not a leak.
    credentials = f'{parsed.username}:{parsed.password}@'
    with pytest.raises(SystemExit):
        main_report(
            [
                '--results-index-db',
                postgres_url,
                '--root',
                str(tmp_path / 'never-ingested'),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    error_text = capsys.readouterr().err
    assert credentials not in error_text
    assert f'{parsed.username}:***@' in error_text


def test_a_document_the_server_refuses_ends_the_pass(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A server enforces its column types, so this is where the refusal is real.

    An identifier larger than a ``bigint`` is rejected by the insert rather than
    by any check the ingest makes.  Charged to the file it would be counted and
    left out of both tables under a run stamped finished, after which absence of
    an ``images`` row -- which every consumer reads as "this image was never
    navigated" -- is an answer nobody can tell from the truth.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N9999999999_1_CALIB',
        metadata_document(image_name=f'N{"9" * 25}_1_CALIB.IMG'),
    )
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    with pytest.raises(UnwritableRowError, match='would not accept its rows'):
        ingest_tree(postgres_url, [root], logger=quiet_logger)


def test_the_shares_write_the_rows_and_the_run_a_single_pass_writes_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Cross-machine ingest is the case this backend exists for.

    A shared SQLite file is not an option there, so the workers connect to a
    server as ordinary clients -- and the rows they write between them, and the
    run row that says the root may be read, must be what one process writes over
    the same tree.  Both are read from one pass, since standing a schema up on
    the server twice to ask two questions of the same rows costs more than it
    tells.
    """
    root = tmp_path / 'results'
    for index in range(6):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
    engine = open_index(postgres_url, create=True)
    try:
        tasks = fan_out_ingest_tasks(
            engine, [root.as_posix()], share_size=2, logger=quiet_logger, tuning=TreeTuning()
        ).tasks
        results = [
            TaskResult(
                task_id=str(task['task_id']),
                result=ingest_task_share(
                    engine, task['data'], logger=quiet_logger, tuning=TreeTuning()
                ),
            )
            for task in tasks
        ]
        complete_ingest_tasks(engine, [root.as_posix()], results, logger=quiet_logger)
        with engine.connect() as connection:
            stubs = list(
                connection.execute(
                    sqlalchemy.select(IMAGES.c.results_path_stub).order_by(
                        IMAGES.c.results_path_stub
                    )
                )
            )
            runs = list(connection.execute(sqlalchemy.select(INGEST_RUNS.c.files_ingested)))
    finally:
        engine.dispose()
    assert len(stubs) == 6
    assert runs[0][0] == 6


def test_an_account_past_what_a_run_row_holds_is_refused_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The bound a run's shares are held to is this server's own column width.

    SQLite types a column by what is put into it and holds 64 bits, so the write
    that a run's total overflows can only be seen here: the count columns are
    32-bit integers on this server, and two shares each reporting a count the
    tally reader accepts on its own sum past what one column holds.  Unrefused,
    the write ends the whole completion in the database driver's own error --
    on the backend workers spread across machines connect to, which is where an
    event log gets concatenated from several sources in the first place.  The
    row afterwards carries the one share that was counted, at exactly the
    largest count it can hold.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    fan_out(postgres_url, [root], logger=quiet_logger)
    huge = [
        reported(
            f'ingest-1-00000{index}',
            {
                'status': 'ok',
                'run_id': 1,
                'root_url': normalize_root_url(root),
                'files_ingested': _LARGEST_RUN_ROW_COUNT,
                'files_skipped': 0,
                'files_failed': 0,
            },
        )
        for index in range(2)
    ]
    complete(postgres_url, [root], huge, logger=quiet_logger)
    assert run_rows(postgres_url)[0].files_ingested == _LARGEST_RUN_ROW_COUNT


def test_a_root_is_unreadable_until_its_shares_are_added_up_on_postgresql(
    postgres_url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Workers on several machines write into one index at once.

    Between the fan-out and the completion the index holds a part of the root,
    and a consumer must be told nobody has ingested it rather than be handed
    whatever has landed.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    engine = open_index(postgres_url, create=True)
    try:
        fan_out_ingest_tasks(
            engine, [root.as_posix()], share_size=2, logger=quiet_logger, tuning=TreeTuning()
        )
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(root)], url=postgres_url)
    finally:
        engine.dispose()
