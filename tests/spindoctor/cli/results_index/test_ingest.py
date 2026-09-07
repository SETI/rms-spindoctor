"""Tests for reading a navigation results tree into the results index.

What these pin is mostly what ingest does *not* do: it does not key an image by
its name, does not read a document whose file has not changed, does not stat a
file it already listed, and does not stop when a file turns out not to be a
navigation document at all.

Turning one document into the facts a row holds is the other half of the ingest
and is pinned separately, beside the module that does it.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from tests.spindoctor.cli.results_index.conftest import (
    run_rows,
)
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
    write_summary_png,
)

from spindoctor.cli.results_index import chunks as chunks_module
from spindoctor.cli.results_index import driver as driver_module
from spindoctor.cli.results_index import ingest_metadata_files
from spindoctor.nav_records import METADATA_SUFFIX, TreeTuning
from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    TECHNIQUES,
    normalize_root_url,
    open_index,
)

TWO_ROOT_STUB = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'
"""The one stub both roots of the two-root reads below hold."""

TWO_ROOT_COVARIANCE = [
    [0.0961, 0.0100, 0.0025],
    [0.0100, 0.0784, -0.0050],
    [0.0025, -0.0050, 0.0009],
]
"""A twist-fitted 3x3 covariance, for the root whose image was twist-fitted."""


def _rows(connection: sqlalchemy.Connection, statement: Any) -> list[Any]:
    """Execute a statement and return its rows.

    Parameters:
        connection: An open connection.
        statement: The statement to run.

    Returns:
        The rows.
    """
    return list(connection.execute(statement))


# ---------------------------------------------------------------------------
# The walk and the writer
# ---------------------------------------------------------------------------


def test_two_volumes_with_one_basename_produce_two_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Keying on the image name alone silently loses one of these."""
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document())
    write_metadata(root, 'COISS_2002/data/N1294561202_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        subtrees = _rows(connection, sqlalchemy.select(IMAGES.c.subtree).order_by(IMAGES.c.subtree))
    engine.dispose()
    assert [row.subtree for row in subtrees] == ['COISS_2001', 'COISS_2002']


def test_each_colliding_image_is_independently_retrievable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The pair is a key, so one of the two can be read without the other."""
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document(status='success'))
    write_metadata(
        root,
        'COISS_2002/data/N1294561202_1_CALIB',
        metadata_document(status='failed', status_reason='no_features', offset=None),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.status).where(
                IMAGES.c.results_path_stub == 'COISS_2002/data/N1294561202_1_CALIB'
            ),
        )
    engine.dispose()
    assert [row.status for row in found] == ['failed']


def _two_roots_holding_one_stub(
    tmp_path: Path,
    documents: tuple[dict[str, Any], dict[str, Any]],
    column: sqlalchemy.Column[Any],
    *,
    logger: pdslogger.PdsLogger,
) -> list[Any]:
    """Ingest one stub under two roots and read one column back, per root.

    The two roots hold the same stub and differ only in the column under test,
    so a read that filtered on the stub alone answers with whichever row it met
    first and passes on a fixture holding one root.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        documents: The document each root holds, first root first.
        column: The column to read back.
        logger: Logger the pass reports through.

    Returns:
        The column's value under each root, in the same order as the documents.
    """
    roots = [tmp_path / 'first', tmp_path / 'second']
    for root, document in zip(roots, documents, strict=True):
        write_metadata(root, TWO_ROOT_STUB, document)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, roots, logger=logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return [
                connection.execute(
                    sqlalchemy.select(column).where(
                        IMAGES.c.root_url == normalize_root_url(root),
                        IMAGES.c.results_path_stub == TWO_ROOT_STUB,
                    )
                ).scalar()
                for root in roots
            ]
    finally:
        engine.dispose()


def test_each_roots_covariance_matrix_survives_the_database(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The matrix goes through a JSON column whole, and per root.

    A twist-fitted 3x3 under one root beside an untwisted 2x2 under the other:
    a column that stored part of a matrix, or a read that found the wrong
    root's row, gives the wrong shape rather than a near-miss.
    """
    first = metadata_document()
    first['navigation_result']['covariance_px2'] = TWO_ROOT_COVARIANCE
    second = metadata_document()
    second['navigation_result']['covariance_px2'] = [[0.25, 0.0], [0.0, 0.25]]
    stored = _two_roots_holding_one_stub(
        tmp_path, (first, second), IMAGES.c.covariance_px2, logger=quiet_logger
    )
    assert stored == [TWO_ROOT_COVARIANCE, [[0.25, 0.0], [0.0, 0.25]]]


def test_each_roots_epoch_survives_the_database(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One root navigated the image; the other never loaded it.

    Whether a document carried an epoch at all is the difference between the
    two roots, so a read that answered from the wrong one reports an image that
    never loaded as one that did, and places it in time for a date filter.
    """
    navigated = metadata_document(image_et=170002800.0)
    unloaded = metadata_document(status='error', status_error='missing_spice_data', offset=None)
    unloaded['navigation_result'].pop('provenance')
    stored = _two_roots_holding_one_stub(
        tmp_path, (navigated, unloaded), IMAGES.c.image_et, logger=quiet_logger
    )
    assert stored == [170002800.0, None]


def test_each_roots_exclusion_order_survives_the_database(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The order is stored, so two roots hold two orders of the same names."""
    first = metadata_document(excluded=['StarRefineNav', 'BodyBlobNav'])
    second = metadata_document(excluded=['BodyBlobNav', 'StarRefineNav'])
    stored = _two_roots_holding_one_stub(
        tmp_path, (first, second), IMAGES.c.excluded_from_consensus, logger=quiet_logger
    )
    assert stored == [
        ['StarRefineNav', 'BodyBlobNav'],
        ['BodyBlobNav', 'StarRefineNav'],
    ]


def test_each_roots_frame_id_survives_the_database(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Each root's row keeps the identifier that root's document recorded."""
    first = metadata_document(pointing={'camera_frame_id': -82360})
    second = metadata_document(pointing={'camera_frame_id': -82361})
    stored = _two_roots_holding_one_stub(
        tmp_path, (first, second), IMAGES.c.camera_frame_id, logger=quiet_logger
    )
    assert stored == [-82360, -82361]


def test_a_bare_basename_stub_ingests_with_a_null_subtree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A simulated scene lives at the root of the tree and names no subtree."""
    root = tmp_path / 'results'
    write_metadata(root, 'sim_scene_000042', metadata_document(instrument='sim'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.subtree))
    engine.dispose()
    assert [row.subtree for row in found] == [None]


def _ingest_a_file_named_only_by_the_suffix(
    tmp_path: Path, logger: pdslogger.PdsLogger
) -> list[Any]:
    """Ingest a tree holding a file whose whole name is the document suffix.

    It is the last bracket case of the suffix test: a name that ends in the
    suffix and is nothing else, so trimming the suffix leaves an empty stub.
    The pass treats it as any other document, which puts it in under the empty
    stub with no subtree above it -- a row every subtree-restricted query passes
    over, which is what a selection is.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the pass reports through.

    Returns:
        The stub and subtree of every row the pass wrote.
    """
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / METADATA_SUFFIX).write_text(json.dumps(metadata_document()), encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub, IMAGES.c.subtree))
    engine.dispose()
    return found


def test_a_file_named_only_by_the_suffix_ingests_under_an_empty_stub(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Trimming a suffix off a name of exactly that length leaves nothing."""
    found = _ingest_a_file_named_only_by_the_suffix(tmp_path, quiet_logger)
    assert [row.results_path_stub for row in found] == ['']


def test_a_file_named_only_by_the_suffix_ingests_with_a_null_subtree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """So it is under no subtree, and no enumeration of subtrees reaches it."""
    found = _ingest_a_file_named_only_by_the_suffix(tmp_path, quiet_logger)
    assert [row.subtree for row in found] == [None]


def _tree_with_a_file_that_is_not_a_document(tmp_path: Path) -> Path:
    """Write a root holding one document and four files that are not one.

    A results root holds the summary PNG a navigation that reached a result
    drew, and whatever else an operator has left there.  None of them is a file
    the pass reads, and the walk has to pass over each without adding it to any
    tally.

    The clutter is chosen to bracket the suffix test rather than to be merely
    unlike a document.  ``notes.txt`` and the summary PNG are unlike one in
    every way; ``scene_index.json`` is JSON and is not a navigation document,
    which is what separates the document suffix from the file extension; and
    ``..._metadata.json.tmp`` -- a partial write, an editor's backup, a
    ``.json.gz`` -- carries the suffix without ending in it, which is what
    separates ending in the suffix from containing it.  A name that only
    contains it yields a stub with the suffix's length cut off the end of a
    longer name, naming nothing, which the pass then retrieves, fails on,
    records nothing for, and retrieves again on every pass afterwards.

    The remaining bracket case is a name that is the suffix and nothing else,
    which is a document as far as the walk is concerned; what it ingests as is
    two tests above.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_summary_png(root, 'VOL/N1454725799_1_CALIB')
    (root / 'VOL' / 'notes.txt').write_text('nothing to ingest here', encoding='utf-8')
    (root / 'VOL' / 'scene_index.json').write_text('{"not": "a document"}', encoding='utf-8')
    (root / 'VOL' / f'N1454725799_1_CALIB{METADATA_SUFFIX}.tmp').write_text(
        '{"half": "written"', encoding='utf-8'
    )
    return root


def test_a_file_that_is_not_a_document_is_not_a_file_this_pass_saw(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The walk counts the documents, and a tree holds far more than documents.

    Counted as one of them, a summary PNG, a JSON file that is not a navigation
    result, or a half-written document would be retrieved, refused for not being
    a navigation document, and tallied against the root on every pass.
    """
    root = _tree_with_a_file_that_is_not_a_document(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_seen, counts.files_ingested, counts.files_failed) == (1, 1, 0)


def test_a_file_that_is_not_a_document_is_no_part_of_what_the_walk_found(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """What the entry loop collects, over a tree of ordinary clutter.

    Every entry is a directory to descend into, a document to collect, or a
    file to pass over, and only the last leaves the listing as it was.  A name
    that ends in the document suffix is what says which, and neither half of
    that is enough on its own: a file that merely ends in ``.json`` is not a
    navigation document, and one that merely contains the suffix yields a stub
    with the suffix's length cut off the end of a longer name, naming nothing,
    which the pass then retrieves, fails on, records nothing for, and retrieves
    again on every pass afterwards.
    """
    root = _tree_with_a_file_that_is_not_a_document(tmp_path)
    listing = driver_module._listing_of_root(
        root.as_posix(), logger=quiet_logger, tuning=TreeTuning()
    )
    assert listing is not None
    assert [found.stub for found in listing.documents] == ['VOL/N1454725799_1_CALIB']


def test_re_ingesting_an_image_replaces_its_child_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A document that lost a technique must not leave the old row behind."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(
        root, stub, metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))])
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(root, stub, metadata_document(per_technique=[]))
    ingest_tree(url, [root], logger=quiet_logger, force=True)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(TECHNIQUES.c.technique_name))
    engine.dispose()
    assert [row.technique_name for row in found] == []


def test_an_unchanged_file_is_not_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second pass over an unchanged tree costs one listing and no reads."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    retrievals: list[Any] = []
    real_retrieve = FCPath.retrieve

    def counted(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        retrievals.append(sub_path)
        return real_retrieve(self, sub_path, **kwargs)

    monkeypatch.setattr(FCPath, 'retrieve', counted)
    ingest_tree(url, [root], logger=quiet_logger)
    assert retrievals == []


def _watch_per_file_questions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path ingest asks the storage layer about individually.

    ``exists`` is never needed at all, so it is forbidden outright.  ``stat``
    is recorded rather than forbidden: a local walk asks it about a directory
    it is entering, to recognize one it has already walked, and that question
    is asked once per directory and never on a cloud root.  What may not
    happen is asking it about a *file*, which on a cloud root is a paid round
    trip per image per run.

    Parameters:
        monkeypatch: Fixture the recorders are installed through.

    Returns:
        The list the paths accumulate in, in call order.
    """
    asked: list[str] = []
    real_stat = FCPath.stat

    def recorded(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        asked.append(self.as_posix())
        return real_stat(self, *args, **kwargs)

    def forbidden(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('ingest asked the backend whether one file exists')

    monkeypatch.setattr(FCPath, 'stat', recorded)
    monkeypatch.setattr(FCPath, 'exists', forbidden)
    return asked


def test_an_unchanged_file_is_not_stat_ed_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing already carried the metrics; asking again is a round trip."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    asked = _watch_per_file_questions(monkeypatch)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_skipped == 1
    assert [path for path in asked if path.endswith(METADATA_SUFFIX)] == []


def test_a_first_ingest_asks_about_no_single_file_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk feeds presence and both metrics, so a first pass is one listing."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    asked = _watch_per_file_questions(monkeypatch)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1
    assert [path for path in asked if path.endswith(METADATA_SUFFIX)] == []


def test_a_touched_file_is_read_again(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A document that changed on disk is what the second pass exists to catch."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(
        root, stub, metadata_document(status='failed', status_reason='no_features', offset=None)
    )
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


SAME_LENGTH_BEFORE = metadata_document(image_name='N1454725799_1_CALIB.IMG')
"""A document whose serialization is the same length as the one below."""

SAME_LENGTH_AFTER = metadata_document(image_name='N1454725798_1_CALIB.IMG')
"""The same document with one digit of the image name changed.

An edit of exactly the same byte length is what leaves the size half of the
incremental comparison saying nothing, so it is the only edit that asks the
modification time whether the file changed.
"""


def _rewrite_at_the_same_length(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest a document, rewrite it to the same length, and ingest again.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    path = write_metadata(root, stub, SAME_LENGTH_BEFORE)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    written = path.stat().st_mtime_ns
    path.write_text(json.dumps(SAME_LENGTH_AFTER), encoding='utf-8')
    # A rewrite this quick can land in the same nanosecond the first write did,
    # which would make the two passes agree for a reason the test is not about.
    os.utime(path, ns=(written + 1_000_000_000, written + 1_000_000_000))
    return url


def test_the_same_length_rewrite_really_is_the_same_length() -> None:
    """Otherwise the test below would be passing on the size half after all."""
    before = len(json.dumps(SAME_LENGTH_BEFORE).encode('utf-8'))
    after = len(json.dumps(SAME_LENGTH_AFTER).encode('utf-8'))
    assert before == after


def test_a_file_rewritten_to_the_same_length_is_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The modification time is the half of the comparison that catches this.

    A results tree is rewritten in place by a re-navigation, and a document
    that changed without changing length is the ordinary case: one status word
    for another, one digit of an offset for another. Comparing size alone would
    skip such a file for as long as it existed.
    """
    url = _rewrite_at_the_same_length(tmp_path, quiet_logger)
    counts = ingest_tree(url, [tmp_path / 'results'], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_file_rewritten_to_the_same_length_updates_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Re-reading is only worth anything if the row that comes back is the new one."""
    url = _rewrite_at_the_same_length(tmp_path, quiet_logger)
    ingest_tree(url, [tmp_path / 'results'], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.image_name))
    engine.dispose()
    assert [row.image_name for row in found] == ['N1454725798_1_CALIB.IMG']


def test_a_touched_file_updates_its_row(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Re-reading is only useful if the row that comes back is the new one."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(
        root, stub, metadata_document(status='failed', status_reason='no_features', offset=None)
    )
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.status))
    engine.dispose()
    assert [row.status for row in found] == ['failed']


def test_force_re_reads_an_unchanged_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The escape hatch for a tree whose metrics cannot be trusted."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_ingested == 1


def test_force_skips_nothing(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A forced pass reads everything, so nothing is counted as skipped."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_skipped == 0


def test_a_listing_without_metrics_re_reads_everything(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that cannot say whether a file changed gets no skip at all."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    real_iterdir = FCPath.iterdir_metadata

    def stripped(self: FCPath) -> Any:
        for path, entry in real_iterdir(self):
            if entry is not None and not entry['is_dir']:
                yield path, {'is_dir': False}
            else:
                yield path, entry

    monkeypatch.setattr(FCPath, 'iterdir_metadata', stripped)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_listing_without_metrics_warns(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently re-reading a whole archive every run would be a mystery."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    real_iterdir = FCPath.iterdir_metadata

    def stripped(self: FCPath) -> Any:
        for path, entry in real_iterdir(self):
            if entry is not None and not entry['is_dir']:
                yield path, {'is_dir': False}
            else:
                yield path, entry

    monkeypatch.setattr(FCPath, 'iterdir_metadata', stripped)
    warnings: list[str] = []
    monkeypatch.setattr(
        quiet_logger, 'warning', lambda message, *args: warnings.append(str(message))
    )
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert any('cannot be ingested incrementally' in message for message in warnings)


def test_a_malformed_document_is_counted_as_an_error(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A results tree holds files that were never navigation documents."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_failed == 1


def test_a_malformed_document_does_not_abort_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One unreadable file among hundreds must not cost the other hundreds."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'bad_metadata.json').write_text('not json at all', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_failures_are_tallied_by_reason(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Hundreds of files nobody wanted must read differently from a real fault."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'params_metadata.json').write_text('{"params": {}}', encoding='utf-8')
    (root / 'broken_metadata.json').write_text('not json at all', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {
        'not a current-schema navigation document (no observation.image_name)': 2,
        'not valid JSON': 1,
    }


def test_a_document_that_is_not_an_object_is_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Valid JSON that is a list is still not a navigation document."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'list_metadata.json').write_text('[1, 2, 3]', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'not a JSON object': 1}


def test_an_ingest_run_is_recorded_at_the_start(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer that looks mid-run must see a root that is not ready."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    seen: list[Any] = []

    real_listing = driver_module._listing_of_root

    def watching(listed_root: Any, **kwargs: Any) -> Any:
        with engine.connect() as connection:
            seen.extend(_rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc)))
        return real_listing(listed_root, **kwargs)

    monkeypatch.setattr(driver_module, '_listing_of_root', watching)
    ingest_metadata_files(engine, [root.as_posix()], logger=quiet_logger, tuning=TreeTuning())
    engine.dispose()
    assert [row.finished_utc for row in seen] == [None]


def test_an_ingest_run_is_completed_at_the_end(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The finish time is what makes absence of a row mean "not navigated"."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_an_ingest_run_records_what_it_covered(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The counts are the record of what a root's index actually contains."""
    root = tmp_path / 'results'
    root.mkdir()
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(
                INGEST_RUNS.c.files_seen, INGEST_RUNS.c.files_ingested, INGEST_RUNS.c.files_failed
            ),
        )
    engine.dispose()
    assert [tuple(row) for row in found] == [(2, 1, 1)]


def test_each_root_gets_its_own_run(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A run covers one root, because a consumer asks about one root."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_metadata(second, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.root_url))
    engine.dispose()
    assert sorted(row.root_url for row in found) == sorted([first.as_posix(), second.as_posix()])


_CHUNKS_OF_THREE = TreeTuning(retrieve_threads=1, retrieve_batch_size=1, ingest_commit_batches=3)
"""A tuning whose transactions cover three documents, passed the way a program passes one."""


def test_a_chunk_boundary_is_crossed_mid_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """More images than one transaction holds must all still arrive."""
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    url = index_url(tmp_path / 'index.sqlite3')
    counts = ingest_tree(url, [root], logger=quiet_logger, tuning=_CHUNKS_OF_THREE)
    assert counts.files_ingested == 7


def test_a_chunk_boundary_leaves_every_row_readable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Counting is not the same as having committed."""
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger, tuning=_CHUNKS_OF_THREE)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 7


def test_the_transactions_are_the_size_the_tuning_says(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk an ingest commits is the tuning's, not a number of the driver's own."""
    chunks: list[int] = []
    real_ingest_chunk = chunks_module._ingest_chunk

    def counting(engine: Any, root: Any, chunk: Any, **kwargs: Any) -> None:
        chunks.append(len(chunk))
        real_ingest_chunk(engine, root, chunk, **kwargs)

    monkeypatch.setattr(driver_module, '_ingest_chunk', counting)
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger, tuning=_CHUNKS_OF_THREE)
    assert chunks == [3, 3, 1]


def _cloud_style_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubs: list[str]) -> Path:
    """Make a local tree behave the way a cloud root behaves.

    On a cloud root, ``get_local_path`` names the file the cache *would* hold
    and does not put anything there; only ``retrieve`` downloads.  Here the
    documents are written to one directory and the root the ingest is handed
    lists them but holds no readable file, so a caller that names a file
    instead of retrieving it gets a path with nothing behind it.

    Parameters:
        tmp_path: Directory both trees live under.
        monkeypatch: Fixture the retrieval is redirected through.
        stubs: Results path stubs the tree holds.

    Returns:
        The root to hand the ingest.
    """
    origin = tmp_path / 'origin'
    root = tmp_path / 'results'
    for stub in stubs:
        write_metadata(origin, stub, metadata_document())
        # The listing sees a file of the right name, size and time; its
        # contents are not the document, so reading it in place fails.
        placeholder = root / f'{stub}{METADATA_SUFFIX}'
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.write_text('not the document', encoding='utf-8')

    def downloading(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        paths = sub_path if isinstance(sub_path, list) else [sub_path]
        return [Path(origin / str(one)) for one in paths]

    monkeypatch.setattr(FCPath, 'retrieve', downloading)
    return root


def test_a_cloud_style_root_downloads_its_files(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_local_path`` names a file on a cloud root; it does not fetch one.

    An ingest that named the file instead of retrieving it would read whatever
    happened to be at that path, which on a cloud root is nothing.
    """
    root = _cloud_style_root(tmp_path, monkeypatch, ['VOL/N1454725799_1_CALIB'])
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_cloud_style_root_reads_the_downloaded_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the row holds what the downloaded document said, not the placeholder."""
    root = _cloud_style_root(tmp_path, monkeypatch, ['VOL/N1454725799_1_CALIB'])
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.image_name))
    engine.dispose()
    assert [row.image_name for row in found] == ['N1454725799_1_CALIB.IMG']


def test_an_unretrievable_file_is_counted_not_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batched retrieval reports its failures rather than raising on one.

    The stand-in honors ``exception_on_fail`` rather than swallowing it, which
    is what makes this a test of the call and not only of the handling: the
    storage layer raises unless it is asked not to, so a caller that stops
    passing the keyword ends the run here instead of counting one file.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def failing(
        self: FCPath, sub_path: Any = None, *, exception_on_fail: bool = True, **kwargs: Any
    ) -> Any:
        errors = [FileNotFoundError('gone') for _ in sub_path]
        if exception_on_fail:
            raise errors[0]
        return errors

    monkeypatch.setattr(FCPath, 'retrieve', failing)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'could not be retrieved': 1}


def _tree_with_a_dangling_symlink(tmp_path: Path) -> Path:
    """Build a results tree holding one real document and one broken link.

    A dangling symlink is the ordinary way a file the walk listed cannot be
    retrieved: the listing names it, and the download finds nothing behind it.
    A re-navigation that moved its output and a partially restored backup both
    leave them.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    link = root / 'VOL' / f'N1454725800_1_CALIB{METADATA_SUFFIX}'
    link.symlink_to(tmp_path / 'nowhere' / f'gone{METADATA_SUFFIX}')
    return root


def test_a_dangling_symlink_costs_only_its_own_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same guarantee against a real broken file rather than a stand-in."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_dangling_symlink_is_counted_as_unretrievable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And it is counted, so a tree full of them does not read as a clean pass."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'could not be retrieved': 1}


def test_a_dangling_symlink_leaves_a_completed_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run that ended on one would leave every consumer refusing the root."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_a_missing_root_is_reported_rather_than_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A root that is not there holds no documents, and says so by counting none."""
    counts = ingest_tree(
        index_url(tmp_path / 'index.sqlite3'), [tmp_path / 'absent'], logger=quiet_logger
    )
    assert counts.files_seen == 0


def test_a_root_is_normalized_before_it_is_stored(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing separator must not make one root into two."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    ingest_metadata_files(engine, [f'{root.as_posix()}/'], logger=quiet_logger, tuning=TreeTuning())
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.root_url))
    engine.dispose()
    assert [row.root_url for row in found] == [root.as_posix()]


def _ingest_two_spellings_of_one_root(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest one root named twice, with and without a trailing separator.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    try:
        ingest_metadata_files(
            engine, [root.as_posix(), f'{root.as_posix()}/'], logger=logger, tuning=TreeTuning()
        )
    finally:
        engine.dispose()
    return url


def test_two_spellings_of_one_root_are_walked_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing separator is not another root, in this mode as in the others.

    Walked twice, every document under it is read twice and the tree is listed
    twice -- the most expensive thing an ingest does, and a paid round trip per
    directory on a cloud root.
    """
    url = _ingest_two_spellings_of_one_root(tmp_path, quiet_logger)
    assert len(run_rows(url)) == 1


def test_two_spellings_of_one_root_read_their_documents_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what the second pass over the same tree costs."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        counts = ingest_metadata_files(
            engine,
            [root.as_posix(), f'{root.as_posix()}/'],
            logger=quiet_logger,
            tuning=TreeTuning(),
        )
    finally:
        engine.dispose()
    assert counts.files_seen == 1


def _ingest_a_relative_root(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest a root named relatively to the working directory.

    A relative root is a documented spelling of the option, and the walk is
    handed the same normalized form the rows are keyed by rather than the
    string as typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL, for whatever the caller means to read from it.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        ingest_metadata_files(engine, ['results'], logger=logger, tuning=TreeTuning())
    finally:
        os.chdir(previous)
        engine.dispose()
    return url


def test_a_relative_root_is_ingested(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The walk is handed the normalized root, which is the absolute one.

    Handing it the string as typed asks the storage layer for a relative local
    URL, which it refuses outright -- a traceback out of a console entry point,
    over a spelling the option documents.
    """
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == ['VOL/N1454725799_1_CALIB']


def test_a_relative_root_completes_its_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run left unfinished is a root every consumer afterwards refuses."""
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_a_relative_root_records_an_absolute_source_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The source file names the same location the root does, not a shorter one."""
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.source_file))
    engine.dispose()
    assert [str(row.source_file).startswith(tmp_path.as_posix()) for row in found] == [True]


def test_a_second_ingest_of_the_same_root_adds_no_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Ingest is idempotent: the same tree twice is the same one row per image."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    ingest_tree(url, [root], logger=quiet_logger, force=True)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 1


def test_the_source_file_records_where_the_document_came_from(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Provenance: which file on which root produced this row."""
    root = tmp_path / 'results'
    path = write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.source_file))
    engine.dispose()
    assert [row.source_file for row in found] == [path.as_posix()]


def test_an_offset_survives_the_database_bit_for_bit(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Fifteen significant digits, through the column and back."""
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(offset=[3.14159265358979, -2.71828182845905]),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.offset_dv))
    engine.dispose()
    assert found[0][0] == 3.14159265358979


def test_a_deep_tree_is_walked_to_the_bottom(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A real results tree is volume, then range, then image."""
    root = tmp_path / 'results'
    stub = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == [stub]


def test_a_file_that_is_not_a_result_is_left_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only the two result suffixes are collected; a tree holds other files."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'VOL' / 'notes.txt').write_text('ignore me', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_seen == 1


def test_the_written_row_survives_a_plain_sqlite_reader(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The index is an ordinary database; opening it directly is supported."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    connection = sqlite3.connect(database)
    try:
        names = [row[0] for row in connection.execute('SELECT image_name FROM images')]
    finally:
        connection.close()
    assert names == ['N1454725799_1_CALIB.IMG']


def test_the_excluded_set_is_stored_as_json(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The column is JSON on both backends, so a direct query can reach inside."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document(excluded=['BodyBlobNav']))
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    connection = sqlite3.connect(database)
    try:
        stored = connection.execute('SELECT excluded_from_consensus FROM images').fetchone()[0]
    finally:
        connection.close()
    assert json.loads(stored) == ['BodyBlobNav']
