"""What the index holds between one ingest pass and the next.

A pass has to leave the index a true account of the tree, and has to leave the
next pass able to tell what it has already paid to read.  Three things follow.
A file that is not a navigation document is recorded as refused rather than
re-downloaded forever.  A document that has left the tree takes its row with
it, because absence of a row is what every consumer reads as "this image was
never navigated" and presence therefore has to mean the reverse.  And a root
the walk could not list at all is not a root that is empty, so its ingest run
is deliberately left unfinished -- as is the run of a root holding one
directory the walk could not list, which stops the pass where it meets it
rather than completing around the documents it could not see.

The transactions are pinned here too.  A crash costs one chunk rather than a
whole run, and one image's delete and inserts share one transaction; both are
properties of what survives a failure, and neither shows up in a count of rows
after a run that went well.
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
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
)

from spindoctor.cli.results_index import UnwritableRowError
from spindoctor.cli.results_index import driver as driver_module
from spindoctor.cli.results_index import store as store_module
from spindoctor.nav_records import (
    METADATA_SUFFIX,
    NOT_VALID_JSON,
    TreeTuning,
    UnlistableDirectoryError,
)
from spindoctor.nav_records import facts as facts_module
from spindoctor.nav_records.facts import (
    DocumentOrigin,
    MetadataDocumentError,
    facts_from_document,
)
from spindoctor.results_index import FAILED_FILES, IMAGES, INGEST_RUNS, TECHNIQUES, open_index
from spindoctor.support.nav_record import record_status

SOURCE = DocumentOrigin(
    root_url='/data/nav-results',
    results_path_stub='COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    source_file='/data/nav-results/x_metadata.json',
    mtime_ns=1234567890123456789,
    size_bytes=4096,
)


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
# Documents of some other shape
# ---------------------------------------------------------------------------


def _with_navigation(**fields: Any) -> dict[str, Any]:
    """Return a valid document with fields of its navigation result replaced.

    Parameters:
        fields: Keys of ``navigation_result`` to overwrite.

    Returns:
        The document.
    """
    document = metadata_document()
    document['navigation_result'].update(fields)
    return document


def _malformed_documents() -> dict[str, dict[str, Any]]:
    """Return one document per container shape the schema does not allow.

    Every one of these reached a bare ``AttributeError`` or ``TypeError`` out of
    the middle of the converter before the shapes were checked, which ended the
    whole run and cost every other file in the tree.

    Returns:
        A name for each shape, and the document carrying it.
    """
    string_observation = metadata_document()
    string_observation['observation'] = 'N1454725799_1_CALIB.IMG'
    listed_navigation = metadata_document()
    listed_navigation['navigation_result'] = ['BodyLimbNav']
    string_timing = metadata_document()
    string_timing['timing'] = '3.25 seconds'
    return {
        'observation-is-a-string': string_observation,
        'navigation-result-is-a-list': listed_navigation,
        'timing-is-a-string': string_timing,
        'per-technique-is-a-string': _with_navigation(per_technique='BodyLimbNav'),
        'per-technique-holds-scalars': _with_navigation(per_technique=[1, 2]),
        'feature-inventory-holds-scalars': _with_navigation(feature_inventory=['BODY_DISC']),
        'feature-inventory-is-a-string': _with_navigation(feature_inventory='BODY_DISC'),
        'excluded-mixes-types': _with_navigation(excluded_from_consensus=['BodyBlobNav', 3]),
        'excluded-is-a-string': _with_navigation(excluded_from_consensus='BodyBlobNav'),
        'provenance-is-a-list': _with_navigation(provenance=['abc1234']),
        'times-is-a-number': _with_navigation(times=170000000.5),
        'pointing-is-a-list': _with_navigation(pointing=[1.0, 0.0, 0.0]),
        'image-classifier-is-a-string': _with_navigation(image_classifier='clean'),
        'diagnostics-is-a-string': _with_navigation(
            per_technique=[{'technique_name': 'BodyLimbNav', 'diagnostics': 'fine'}]
        ),
        'two-techniques-of-one-name': _with_navigation(
            per_technique=[
                technique('BodyLimbNav', (1.0, 1.0)),
                technique('BodyLimbNav', (2.0, 2.0)),
            ]
        ),
        'two-techniques-with-no-name': _with_navigation(
            per_technique=[{'offset_px': [1.0, 1.0]}, {'offset_px': [2.0, 2.0]}]
        ),
        'one-technique-with-no-name': _with_navigation(per_technique=[{'offset_px': [1.0, 1.0]}]),
    }


_MALFORMED_PARAMS = [
    pytest.param(document, id=name) for name, document in _malformed_documents().items()
]


@pytest.mark.parametrize('document', _MALFORMED_PARAMS)
def test_a_document_of_another_shape_is_refused(document: dict[str, Any]) -> None:
    """A container the schema declares holds what the schema says, or nothing.

    Parameters:
        document: A document carrying one disallowed container shape.
    """
    with pytest.raises(MetadataDocumentError, match='not a current-schema navigation document'):
        facts_from_document(document, SOURCE)


@pytest.mark.parametrize('document', _MALFORMED_PARAMS)
def test_a_document_of_another_shape_costs_only_itself(
    document: dict[str, Any], tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One file of the wrong shape must not cost the rest of the tree.

    Parameters:
        document: A document carrying one disallowed container shape.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', document)
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_ingested, counts.files_failed) == (1, 1)


class _NobodyEnumeratedThisError(Exception):
    """An exception type the document reader has no way to name.

    Stands for a fault in the reader rather than a shape of the document, which
    is the case the policy under test is about.  It is defined here and imported
    nowhere, so nothing can be catching it by name.
    """


def _tree_of_two_documents(tmp_path: Path) -> Path:
    """Write a tree holding two documents this schema accepts.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', metadata_document())
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    return root


def _reader_that_faults_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the first document read raise a fault the reader cannot name.

    Raised from inside the reading rather than from the call to it, so that a
    guard put back anywhere along that path would catch it.

    Parameters:
        monkeypatch: Fixture one step of the reading is replaced through.
    """
    calls: list[int] = []
    real_status = record_status

    def occasionally_exploding(metadata: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise _NobodyEnumeratedThisError('a fault in this code')
        return real_status(metadata)

    monkeypatch.setattr(facts_module, 'record_status', occasionally_exploding)


def test_a_fault_in_the_reader_ends_the_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is recorded with the file's own metrics, so the next pass skips it.

    A fault in this code written down that way would outlive its own fix, and
    every later pass would report a clean run over a tree an image is missing
    from.  So it is not written down: it ends the pass where it happened.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the reader is replaced through.
    """
    root = _tree_of_two_documents(tmp_path)
    _reader_that_faults_once(monkeypatch)
    with pytest.raises(_NobodyEnumeratedThisError, match='a fault in this code'):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)


def test_a_fault_in_the_reader_records_no_refusal(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is the half that would survive the fix, so it is the half pinned.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the reader is replaced through.
    """
    root = _tree_of_two_documents(tmp_path)
    _reader_that_faults_once(monkeypatch)
    url = index_url(tmp_path / 'index.sqlite3')
    with pytest.raises(_NobodyEnumeratedThisError):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(FAILED_FILES.c.results_path_stub))
    engine.dispose()
    assert found == []


def test_a_fault_in_the_reader_stores_no_image(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass ends where the fault happened, so the document after it is unread.

    A pass that charged the fault to the one document and carried on would leave
    the second one stored and satisfy every other test here, while the image the
    fault struck sat in neither table under a run that raised.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the reader is replaced through.
    """
    root = _tree_of_two_documents(tmp_path)
    _reader_that_faults_once(monkeypatch)
    url = index_url(tmp_path / 'index.sqlite3')
    with pytest.raises(_NobodyEnumeratedThisError):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert found == []


def _tree_with_unparseable_nesting(tmp_path: Path) -> Path:
    """Write a tree holding one document and one file no JSON value comes out of.

    Twenty thousand opening braces and nothing else.  How a decoder gives up on
    them is its own business: one that recurses once per level of nesting
    exhausts the recursion limit part way down, and one that does not recurse
    reports the value that never arrived.  The guard under test names neither,
    because what it charges the file for is that nothing was parsed out of it.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'nested_metadata.json').write_text('{"a":' * 20000, encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    return root


def test_a_document_the_decoder_gives_up_on_costs_only_itself(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The file is charged and the run reads the rest, however the decoder gave up."""
    root = _tree_with_unparseable_nesting(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_ingested, counts.files_failed) == (1, 1)


def test_a_document_the_decoder_gives_up_on_is_charged_to_the_parse(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One reason covers every way a decoder ends with no value, so a tally reads as one.

    Compared whole rather than by prefix: however the decoder gave up, the file
    earns the one reason that says no value came out of it, and it is the same
    reason a reader of the results tree gives for the same file.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_unparseable_nesting(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    (reason,) = counts.failures_by_reason
    assert reason == NOT_VALID_JSON


def test_a_document_the_decoder_gives_up_on_leaves_a_completed_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unfinished run is worse than a lost file: every consumer refuses the root."""
    root = _tree_with_unparseable_nesting(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [True]


UNSTORABLE_IMAGE_NAME = f'N{"9" * 25}_1_CALIB.IMG'
"""An image name whose leading digit run does not fit in a 64-bit column.

``image_number`` is derived from it, and the driver refuses the value at the
insert rather than at any check this code makes -- which is what makes it a
database failure rather than a document-shape one.  What is under test is the
writer's answer to a row the database will not take, whatever produced it.
"""


EARLIER_STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the two below that a pass reaches first.

A pass reads a root in stub order, so which of two documents is written before
the other is settled by their names and not by the order the tree lists them
in.  Stated here because the tests below turn on it: what makes them evidence
is that the failure struck after a document had already been committed.
"""

LATER_STUB = 'VOL/N9999999999_1_CALIB'
"""The stub of the two that a pass reaches second."""


def _stubs_as_they_are_written(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the images a pass writes, in the order it writes them.

    Parameters:
        monkeypatch: Fixture the writer is wrapped through.

    Returns:
        The list the wrapper appends each image's stub to, which fills as the
        pass runs.
    """
    written: list[str] = []
    real_write = store_module._write_image

    def recording(connection: Any, rows: Any) -> Any:
        written.append(str(rows.image['results_path_stub']))
        return real_write(connection, rows)

    monkeypatch.setattr(store_module, '_write_image', recording)
    return written


def test_a_pass_writes_a_root_in_stub_order(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what settles what a pass has already written when one document fails.

    The two files are created in the opposite order, so a pass taking the order
    the tree happened to list them in would write them the other way round and
    the tests below would be measuring the filesystem.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the writer is wrapped through.
    """
    root = tmp_path / 'results'
    write_metadata(root, LATER_STUB, metadata_document())
    write_metadata(root, EARLIER_STUB, metadata_document())
    written = _stubs_as_they_are_written(monkeypatch)
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert written == [EARLIER_STUB, LATER_STUB]


def _tree_with_an_unstorable_document(tmp_path: Path) -> Path:
    """Write a tree holding one document the database will not accept.

    The unstorable one is the later stub, so the pass has already committed the
    other by the time it reaches it, and it is created first so that a pass that
    stopped ordering by stub would be seen to reach them the other way round.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, LATER_STUB, metadata_document(image_name=UNSTORABLE_IMAGE_NAME))
    write_metadata(root, EARLIER_STUB, metadata_document())
    return root


def test_a_document_the_database_refuses_ends_the_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Counted and passed over, it would be in neither table under a finished run.

    Absence of an ``images`` row is what every consumer reads as "this image was
    never navigated", so an image left out of both tables by a run that reported
    itself clean is an answer nobody can tell from the truth.  What refused it is
    this program's writer or its column set, not the file, and the next document
    of the same shape fails the same way.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_an_unstorable_document(tmp_path)
    with pytest.raises(UnwritableRowError, match='would not accept its rows'):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)


def test_such_a_failure_names_the_document_the_database_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One of several hundred thousand files, so the message has to say which.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_an_unstorable_document(tmp_path)
    with pytest.raises(UnwritableRowError) as excinfo:
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert LATER_STUB in str(excinfo.value)


def test_such_a_failure_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what makes every consumer refuse the root instead of reading it.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_an_unstorable_document(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    with pytest.raises(UnwritableRowError):
        ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [False]


def test_such_a_failure_keeps_the_documents_already_written(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Writing one image per transaction is what makes a rerun cheap after the fix.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_an_unstorable_document(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    with pytest.raises(UnwritableRowError):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == [EARLIER_STUB]


def test_such_a_failure_records_no_refusal(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal carries the file's own metrics, so the next pass would skip it.

    The document is a navigation result and this program would not store it;
    recorded as a refusal, that defect would survive its own fix.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
    """
    root = _tree_with_an_unstorable_document(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    with pytest.raises(UnwritableRowError):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(FAILED_FILES.c.results_path_stub))
    engine.dispose()
    assert found == []


def test_a_lost_connection_is_not_read_as_an_unwritable_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every later image would fail too, and a completed run would hide all of them.

    A consumer reads the absence of a row as "this image was never navigated",
    so a run that finished because the database went away is worse than one that
    did not finish at all.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def connection_lost(connection: Any, rows: Any) -> Any:
        raise sqlalchemy.exc.DBAPIError(
            'INSERT INTO images',
            {},
            OSError('server closed the connection'),
            connection_invalidated=True,
        )

    monkeypatch.setattr(store_module, '_write_image', connection_lost)
    with pytest.raises(sqlalchemy.exc.DBAPIError, match='server closed the connection'):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)


def test_a_reason_says_the_file_was_never_a_navigation_result(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Several hundred of these are ordinary, and the tally has to read that way."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {
        'not a current-schema navigation document (no observation.image_name)': 1
    }


def test_a_reason_carries_one_example_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A field-level diagnosis is only a judgement once one real file is named."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'edges_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert list(counts.example_by_reason.values()) == [path.as_posix()]


# ---------------------------------------------------------------------------
# What a refused file costs the next pass
# ---------------------------------------------------------------------------


def _counting_retrievals(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every batched retrieval the next ingest performs.

    Parameters:
        monkeypatch: Fixture the retrieval is wrapped through.

    Returns:
        The list each call appends its sub-path argument to.
    """
    retrievals: list[Any] = []
    real_retrieve = FCPath.retrieve

    def counted(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        retrievals.append(sub_path)
        return real_retrieve(self, sub_path, **kwargs)

    monkeypatch.setattr(FCPath, 'retrieve', counted)
    return retrievals


def _tree_with_a_refused_file(tmp_path: Path) -> Path:
    """Write a results tree holding one document and one file that is not one.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    return root


UNPARSEABLE_STUB = 'VOL/N1454725800_1_CALIB'
"""A file no reader gets a JSON value out of at all."""

NOT_AN_OBJECT_STUB = 'VOL/N1454725801_1_CALIB'
"""A file that parses to a JSON value of some other kind."""

ROOTLESS_STUB = 'N1454725802_1_CALIB'
"""A file above every subtree, which no enumeration walks or queries."""


def _tree_holding_one_of_each_refusal(root: Path) -> None:
    """Write a root holding one file of every kind a pass refuses.

    The four differ in why no record came out of them and in where they sit: a
    JSON object of the wrong schema, a half-written file no JSON value comes out
    of, a file holding a JSON list, and one above every subtree.

    Parameters:
        root: The results root to write under.
    """
    (root / 'VOL').mkdir(parents=True, exist_ok=True)
    (root / f'VOL/N1454725799_1_CALIB{METADATA_SUFFIX}').write_text(
        '{"edges": []}', encoding='utf-8'
    )
    (root / f'{UNPARSEABLE_STUB}{METADATA_SUFFIX}').write_text(
        '{"status": "error"', encoding='utf-8'
    )
    (root / f'{NOT_AN_OBJECT_STUB}{METADATA_SUFFIX}').write_text('[1, 2, 3]', encoding='utf-8')
    (root / f'{ROOTLESS_STUB}{METADATA_SUFFIX}').write_text('{"edges": []}', encoding='utf-8')


def test_every_kind_of_refused_file_is_recorded(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A file with no row is one the next pass pays to download and parse again.

    The record is what stops that, so it is written for every file no record
    came out of, whatever went wrong with it and wherever under the root it sits
    -- including one above every subtree, which no enumeration ever asks about
    but which a pass still meets and still has to remember refusing.
    """
    root = tmp_path / 'results'
    _tree_holding_one_of_each_refusal(root)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            recorded = connection.execute(sqlalchemy.select(FAILED_FILES.c.results_path_stub))
            stubs = sorted(str(row.results_path_stub) for row in recorded)
    finally:
        engine.dispose()
    assert stubs == sorted(
        ['VOL/N1454725799_1_CALIB', UNPARSEABLE_STUB, NOT_AN_OBJECT_STUB, ROOTLESS_STUB]
    )


def test_a_refused_file_is_not_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with no row is still a file this ingest has already paid to read.

    A real tree holds hundreds that are not navigation documents.  Re-reading
    them every pass is, on a cloud root, hundreds of paid downloads per run --
    exactly the cost the index exists to remove.
    """
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    retrievals = _counting_retrievals(monkeypatch)
    ingest_tree(url, [root], logger=quiet_logger)
    assert retrievals == []


def _listing_without_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the walk see a backend whose listing reports no size and no time.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
    """
    real_iterdir = FCPath.iterdir_metadata

    def without_metrics(self: FCPath) -> Any:
        for path, _entry_metadata in real_iterdir(self):
            yield path, None

    monkeypatch.setattr(FCPath, 'iterdir_metadata', without_metrics)


def test_a_metric_less_listing_retrieves_every_document_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that cannot say whether a file changed cannot be trusted that it did not.

    Both recorded metrics are then NULL, and a comparison of two pairs of NULLs
    finds them equal -- so a root whose listing reports neither would otherwise
    be read once and never updated again, silently and permanently.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', metadata_document())
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    _listing_without_metrics(monkeypatch)
    ingest_tree(url, [root], logger=quiet_logger)
    retrievals = _counting_retrievals(monkeypatch)
    ingest_tree(url, [root], logger=quiet_logger)
    assert [len(batch) for batch in retrievals] == [2]


def test_a_listing_that_does_not_say_what_is_a_directory_is_asked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend may report an entry's metrics and not whether it is a directory.

    Reading the key as though it were always there ends the walk on a
    ``KeyError``; reading a missing one as "not a directory" would silently drop
    the whole subtree under it.  The entry is asked instead.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    real_iterdir = FCPath.iterdir_metadata

    def without_the_kind(self: FCPath) -> Any:
        for path, entry_metadata in real_iterdir(self):
            reduced = None if entry_metadata is None else dict(entry_metadata)
            if reduced is not None:
                reduced.pop('is_dir', None)
            yield path, reduced

    monkeypatch.setattr(FCPath, 'iterdir_metadata', without_the_kind)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_metric_less_listing_skips_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the count says so, since that is what an operator reads."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725798_1_CALIB', metadata_document())
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    _listing_without_metrics(monkeypatch)
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert (counts.files_ingested, counts.files_skipped) == (2, 0)


def test_a_refused_file_is_counted_as_skipped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The pass has to account for it, since it accounted for it as seen."""
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert (counts.files_seen, counts.files_skipped) == (2, 2)


def test_force_re_reads_a_refused_file(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The escape hatch reaches every file, not only the ones that ingested."""
    root = _tree_with_a_refused_file(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_failed == 1


def test_a_changed_refusal_is_read_again(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A file that was refused and then rewritten is a file nobody has read."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'thing_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    path.write_text(json.dumps(metadata_document()), encoding='utf-8')
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_refusal_that_becomes_a_document_leaves_no_refusal_behind(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Otherwise the next pass would skip a file that now has a row to write."""
    root = tmp_path / 'results'
    root.mkdir()
    path = root / 'thing_metadata.json'
    path.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    path.write_text(json.dumps(metadata_document()), encoding='utf-8')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(FAILED_FILES.c.results_path_stub))
    engine.dispose()
    assert found == []


def test_a_document_that_stops_reading_loses_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A row nothing backs would answer confidently for an image nothing produced."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    (root / f'{stub}{METADATA_SUFFIX}').write_text('{"edges": []}', encoding='utf-8')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert found == []


# ---------------------------------------------------------------------------
# What leaving the tree costs
# ---------------------------------------------------------------------------


def test_a_deleted_document_loses_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Absence of a row means "never navigated", so presence must mean the reverse."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    gone = write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub),
        )
    engine.dispose()
    assert [row.results_path_stub for row in found] == ['VOL/N1454725799_1_CALIB']


def test_a_deleted_document_is_counted_as_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A silent deletion of rows is not something an operator should discover later."""
    root = tmp_path / 'results'
    gone = write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_removed == 1


def test_a_deleted_refusal_is_not_counted_as_a_removed_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is of rows an operator would otherwise go looking for.

    A stub that leaves the tree is forgotten from both tables, but only one of
    them held an answer about an image. Counting the refusals as well would
    report a tree of documents that were never navigation results as a mass
    deletion of navigation rows.
    """
    root = tmp_path / 'results'
    root.mkdir()
    refused = root / 'edges_metadata.json'
    refused.write_text('{"edges": []}', encoding='utf-8')
    ingested = write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    refused.unlink()
    ingested.unlink()
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_removed == 1


def test_a_deleted_document_takes_its_child_rows_with_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A technique row belonging to no image is a count nothing can explain."""
    root = tmp_path / 'results'
    gone = write_metadata(
        root,
        'VOL/N1454725800_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))]),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(TECHNIQUES.c.technique_name))
    engine.dispose()
    assert found == []


def test_another_roots_rows_are_left_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A pass over one root is evidence about that root and no other."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_metadata(second, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    ingest_tree(url, [first], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 2


def test_a_deleted_refusal_loses_its_recorded_refusal(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal outliving its file would skip a file written there later.

    The stub is recorded as refused; a file written at that stub afterwards has
    the size and modification time of a file nobody has read, but the refusal it
    would be compared against is the deleted file's.
    """
    root = tmp_path / 'results'
    root.mkdir()
    gone = root / 'edges_metadata.json'
    gone.write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(FAILED_FILES.c.results_path_stub))
    engine.dispose()
    assert found == []


UNLISTABLE_ERRORS = [
    pytest.param(FileNotFoundError, id='not-there'),
    pytest.param(NotADirectoryError, id='not-a-directory'),
    pytest.param(PermissionError, id='this-user-may-not-read-it'),
    pytest.param(TimeoutError, id='the-share-stopped-answering'),
]
"""Every way a real tree refuses to list a directory.

They are one thing to the walk -- it can see no result file there, which is not
evidence that there is none -- and enumerating only some of them lets the others
through as an empty directory.  A permission error is the commonest of all on a
shared tree.
"""


def _unlistable_subdirectory(monkeypatch: pytest.MonkeyPatch, error: type[OSError]) -> None:
    """Make one subdirectory of a two-volume tree refuse to be listed.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
        error: The exception type that directory raises.
    """
    real_iterdir = FCPath.iterdir_metadata

    def unlistable_vol2(self: FCPath) -> Any:
        if self.name == 'VOL2':
            raise error(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', unlistable_vol2)


def _two_volume_tree(tmp_path: Path) -> Path:
    """Write a results tree with one document in each of two volumes.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    return root


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_a_directory_that_cannot_be_listed_stops_the_pass(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass that finished around the gap would answer wrongly and go on doing so.

    Absence of a row is what every consumer reads as "this image was never
    navigated", and under a directory nobody listed that reading is simply
    false.  A stopped run costs an ingest; a completed one that skipped a
    directory costs a wrong answer no later pass corrects.

    Parameters:
        error: The exception type the unlistable directory raises.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    root = _two_volume_tree(tmp_path)
    _unlistable_subdirectory(monkeypatch, error)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_the_refusal_names_the_directory_that_would_not_list(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one thing an operator has to go and fix is the one thing it must say.

    Parameters:
        error: The exception type the unlistable directory raises.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    root = _two_volume_tree(tmp_path)
    _unlistable_subdirectory(monkeypatch, error)
    with pytest.raises(UnlistableDirectoryError) as excinfo:
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (root / 'VOL2').as_posix() in str(excinfo.value)


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_the_pass_stops_before_it_reads_a_document(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping where the walk finds it is the whole difference in what it costs.

    The same refusal noticed where the prune is skipped instead would discard a
    pass that had already read every document under the root, which on an
    archive-scale root is hours of retrieval thrown away to say what the first
    minute of the walk could have said.

    Parameters:
        error: The exception type the unlistable directory raises.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    root = _two_volume_tree(tmp_path)
    read: list[Any] = []
    monkeypatch.setattr(driver_module, '_ingest_chunk', lambda *args, **kwargs: read.append(args))
    _unlistable_subdirectory(monkeypatch, error)
    with pytest.raises(UnlistableDirectoryError):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert read == []


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_a_stopped_pass_leaves_its_run_unfinished(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No finish time is what makes every consumer refuse the root outright.

    Parameters:
        error: The exception type the unlistable directory raises.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    root = _two_volume_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _unlistable_subdirectory(monkeypatch, error)
    with pytest.raises(UnlistableDirectoryError):
        ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [False]


def _entries_the_listing_says_nothing_about(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    """Strip the listing's metadata and make asking about an entry fail.

    A backend whose listing carries no ``is_dir`` is asked about each entry
    directly, which is the only path on which the answer can fail at all.

    Parameters:
        monkeypatch: Fixture the listing and the inspection are wrapped through.
        error: The exception type asking about an entry raises.
    """
    real_iterdir = FCPath.iterdir_metadata

    def without_metadata(self: FCPath) -> Any:
        for path, _metadata in real_iterdir(self):
            yield path, None

    def refusing(self: FCPath) -> bool:
        raise error(self.as_posix())

    monkeypatch.setattr(FCPath, 'iterdir_metadata', without_metadata)
    monkeypatch.setattr(FCPath, 'is_dir', refusing)


def test_an_entry_the_storage_layer_will_not_classify_stops_the_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry nobody will answer about may be a directory full of documents.

    Passing it over as a file is the same gap as walking past a directory that
    would not list, arrived at one step earlier: the walk goes on, the run
    completes, and everything under it reads as never navigated.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing and the inspection are wrapped through.
    """
    root = _two_volume_tree(tmp_path)
    _entries_the_listing_says_nothing_about(monkeypatch, PermissionError)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)


def test_an_entry_that_has_gone_away_since_the_listing_is_passed_over(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deletion landing mid-walk is ordinary, and leaves nothing to have missed.

    The pass finishes and ingests what it did find, because an entry that is
    not there holds no document this pass failed to see.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing and the inspection are wrapped through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'N1454725799_1_CALIB', metadata_document())
    (root / 'VOL1').mkdir(exist_ok=True)
    _entries_the_listing_says_nothing_about(monkeypatch, FileNotFoundError)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_root_walked_before_the_one_that_stopped_keeps_its_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is already ingested and stamped is good, and is not rolled back.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    first = tmp_path / 'first'
    write_metadata(first, 'VOL1/N1454725801_1_CALIB', metadata_document())
    second = _two_volume_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _unlistable_subdirectory(monkeypatch, error=PermissionError)
    with pytest.raises(UnlistableDirectoryError):
        ingest_tree(url, [first, second], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [True, False]


def test_a_root_named_after_the_one_that_stopped_is_never_walked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass ends, so a later root gets no run of its own and no rows.

    A root left with no run at all is one every consumer refuses, which is the
    same answer it gives for the root that stopped.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    stopping = _two_volume_tree(tmp_path)
    later = tmp_path / 'later'
    write_metadata(later, 'VOL1/N1454725802_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    _unlistable_subdirectory(monkeypatch, error=PermissionError)
    with pytest.raises(UnlistableDirectoryError):
        ingest_tree(url, [stopping, later], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.root_url))
    engine.dispose()
    assert [row.root_url for row in found] == [stopping.as_posix()]


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_directory_the_filesystem_will_not_open_stops_the_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same thing again, against a real directory rather than a stand-in."""
    root = _two_volume_tree(tmp_path)
    closed = root / 'VOL2'
    closed.chmod(0o000)
    try:
        with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
            ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    finally:
        closed.chmod(0o755)


def test_a_root_the_walk_lists_completely_records_no_refusal(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The ordinary pass over the same tree, so the tests above pin the refusal.

    Every assertion above is about a tree one directory of which refuses; this
    is that tree with nothing wrong with it, and it completes.
    """
    root = _two_volume_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 2


# ---------------------------------------------------------------------------
# A tree that leads back into itself
# ---------------------------------------------------------------------------


def _tree_that_links_to_its_own_ancestor(tmp_path: Path) -> Path:
    """Build a results tree holding a link from a subdirectory to the root.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'VOL' / 'up').symlink_to(root)
    return root


def test_a_link_back_to_an_ancestor_writes_one_row_for_one_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Walking one directory twice writes the same document under two stubs.

    Nothing stops such a walk except the filesystem's own limit on how many
    links it will follow, so one document under a tree that links back to
    itself becomes as many rows as the limit allows, each under a stub no
    consumer will ever ask about. Every one of them answers for an image, and
    the count of navigated images is wrong by all of them but one.
    """
    root = _tree_that_links_to_its_own_ancestor(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == ['VOL/N1454725799_1_CALIB']


def test_a_link_back_to_an_ancestor_does_not_stop_the_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A directory reached a second way is not a directory nobody listed.

    Its documents are in the listing already, under the path the walk met
    first, so declining to walk it again leaves the root wholly accounted for
    and the pass completes.
    """
    root = _tree_that_links_to_its_own_ancestor(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [True]


def test_a_link_back_to_an_ancestor_still_prunes_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And such a pass has the evidence to remove a row, so it removes it.

    This is the difference between a directory that was reached twice and one
    that was not reached at all: the first leaves presence meaning what absence
    means, and the second is why a pass that meets one stops.
    """
    root = _tree_that_links_to_its_own_ancestor(tmp_path)
    write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    (root / f'VOL/N1454725800_1_CALIB{METADATA_SUFFIX}').unlink()
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_removed == 1


def test_a_link_to_a_directory_outside_the_walk_is_followed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only a directory already walked is skipped, not every linked one.

    A results tree that reaches a volume through a link is an ordinary
    deployment, and refusing to follow links at all would silently drop it.
    """
    root = tmp_path / 'results'
    root.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    write_metadata(elsewhere, 'N1454725799_1_CALIB', metadata_document())
    (root / 'VOL').symlink_to(elsewhere)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


# ---------------------------------------------------------------------------
# What licenses a prune
# ---------------------------------------------------------------------------


def test_a_pass_stopped_at_a_directory_removes_no_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prune is licensed by a listing of the whole root, and this pass has none.

    The rows of the volume that would not list are exactly the rows a pass that
    pruned on a partial listing would delete, and their images are still in the
    tree: deleting them would answer "never navigated" for images that were.
    """
    root = _two_volume_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    _unlistable_subdirectory(monkeypatch, PermissionError)
    with pytest.raises(UnlistableDirectoryError, match='VOL2 could not be listed'):
        ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.results_path_stub).order_by(IMAGES.c.results_path_stub),
        )
    engine.dispose()
    assert [row.results_path_stub for row in found] == [
        'VOL1/N1454725799_1_CALIB',
        'VOL2/N1454725800_1_CALIB',
    ]


def test_the_prune_deletes_only_under_the_root_it_listed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The key is ``(root_url, results_path_stub)``, and the stub half is not unique.

    Two roots holding one stub between them is the ordinary case -- a tree
    copied to a second location, a mirror being filled -- and a prune reading
    only the stub would delete the other root's row for every document that has
    left this one.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    stub = 'VOL/N1454725799_1_CALIB'
    gone = write_metadata(first, stub, metadata_document())
    write_metadata(second, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    gone.unlink()
    ingest_tree(url, [first], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.root_url))
    engine.dispose()
    assert [row.root_url for row in found] == [second.as_posix()]


# ---------------------------------------------------------------------------
# A root nobody can list
# ---------------------------------------------------------------------------


def _finish_times(url: str) -> list[Any]:
    """Return the finish time of every ingest run in an index.

    Parameters:
        url: The index URL.

    Returns:
        One entry per run, in insertion order.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return [
                row.finished_utc
                for row in _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
            ]
    finally:
        engine.dispose()


def test_a_root_that_is_not_there_leaves_its_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A completed run over a mistyped root answers "not navigated" for every image."""
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [tmp_path / 'absent'], logger=quiet_logger)
    assert _finish_times(url) == [None]


def test_a_root_that_is_not_there_is_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is what the driver reports, since nothing else went wrong."""
    counts = ingest_tree(
        index_url(tmp_path / 'index.sqlite3'), [tmp_path / 'absent'], logger=quiet_logger
    )
    assert counts.roots_unreadable == 1


def test_an_empty_root_completes_its_run(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A root that exists and holds nothing has been ingested, and holds nothing."""
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    assert [time is not None for time in _finish_times(url)] == [True]


def test_a_root_that_is_not_there_keeps_its_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unmounted root is not an emptied one, and must cost no row."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    for path in sorted(root.rglob('*_metadata.json')):
        path.unlink()
    (root / 'VOL').rmdir()
    root.rmdir()
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 1


# ---------------------------------------------------------------------------
# What one transaction holds
# ---------------------------------------------------------------------------


def _seven_images(tmp_path: Path) -> Path:
    """Write seven documents, enough to cross a chunk boundary twice.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    return root


class _TheWriterDiedError(BaseException):
    """A failure of the process rather than of the document being written.

    A document the database refuses is that document's own problem and is
    refused as one, so an ordinary exception no longer reaches the chunk's
    transaction.  What still does is a failure of everything -- the process
    being killed, the machine going down -- and that is what a chunk boundary
    bounds the cost of.  Deriving from ``BaseException`` is what makes this
    stand for one.
    """


def test_a_crash_mid_run_costs_one_chunk_and_no_more(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunking is what a crash costs, and counting rows afterwards is what says so.

    Seven images in chunks of three, failing on the fifth write: the first chunk
    is committed and the second is not, so exactly three rows survive.  One
    transaction for the whole run would leave none, and a commit per image would
    leave four.
    """
    chunks_of_three = TreeTuning(retrieve_threads=1, retrieve_batch_size=1, ingest_commit_batches=3)
    root = _seven_images(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    written: list[Any] = []
    real_write = store_module._write_image

    def failing(connection: Any, rows: Any) -> Any:
        written.append(rows)
        if len(written) == 5:
            raise _TheWriterDiedError('the writer died')
        return real_write(connection, rows)

    monkeypatch.setattr(store_module, '_write_image', failing)
    with pytest.raises(_TheWriterDiedError, match='the writer died'):
        ingest_tree(url, [root], logger=quiet_logger, tuning=chunks_of_three)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 3


def _statement_transactions(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, Any]]:
    """Ingest one image, recording the transaction each statement ran inside.

    The transaction objects are kept rather than their identities, so comparing
    two of them is a comparison of live objects rather than of addresses a
    garbage collector is free to reuse.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the execute wrapper is installed through.

    Returns:
        The opening line of each statement, with the transaction it ran inside.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))]),
    )
    seen: list[tuple[str, Any]] = []
    real_execute = sqlalchemy.Connection.execute

    def watching(self: Any, statement: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append((str(statement).split('\n')[0], self.get_transaction()))
        return real_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy.Connection, 'execute', watching)
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    return seen


def _transaction_of(seen: list[tuple[str, Any]], opening: str) -> Any:
    """Return the transaction the first statement with this opening ran inside.

    Parameters:
        seen: What :func:`_statement_transactions` recorded.
        opening: Text the statement begins with.

    Returns:
        The transaction object.
    """
    return next(transaction for text, transaction in seen if text.startswith(opening))


def test_an_image_write_runs_inside_a_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two statements outside any transaction would compare equal and prove nothing."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is not None


def test_the_image_delete_and_its_insert_share_one_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent worker must never see the gap between the two."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is _transaction_of(
        seen, 'INSERT INTO images'
    )


def test_the_image_delete_and_its_child_inserts_share_one_transaction(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delete cascades to the children, so their inserts belong with it."""
    seen = _statement_transactions(tmp_path, quiet_logger, monkeypatch)
    assert _transaction_of(seen, 'DELETE FROM images') is _transaction_of(
        seen, 'INSERT INTO techniques'
    )


# ---------------------------------------------------------------------------
# Absent is not empty
# ---------------------------------------------------------------------------


def _stored_json(database: Path, column: str) -> list[Any]:
    """Read one JSON column of every image row with a plain SQLite reader.

    A column name cannot be a bind parameter, so it is interpolated; it is
    checked against the table first, so no string a caller invents can reach the
    statement.

    Parameters:
        database: The index file.
        column: The column to read, which must be one ``images`` declares.

    Returns:
        One entry per row: the stored text, or None for SQL NULL.

    Raises:
        ValueError: If ``images`` declares no such column.
    """
    if column not in IMAGES.c:
        raise ValueError(f'images declares no column named {column}')
    connection = sqlite3.connect(database)
    try:
        return [row[0] for row in connection.execute(f'SELECT {column} FROM images')]
    finally:
        connection.close()


def test_an_absent_cmatrix_is_sql_null(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """``WHERE cmatrix IS NOT NULL`` has to find the rows that carry a matrix.

    Stored as the JSON value ``null`` the column would satisfy that test on every
    row ever written, and the CSV export would carry the text ``null``.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert _stored_json(database, 'cmatrix') == [None]


def test_a_recorded_cmatrix_is_not_sql_null(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the reverse, so the test tells the two cases apart in both directions."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', document)
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert json.loads(str(_stored_json(database, 'cmatrix')[0])) == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_an_empty_exclusion_set_is_stored_as_an_empty_list(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing excluded is a statement, and is not the same as nothing recorded."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document(excluded=[]))
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    assert json.loads(str(_stored_json(database, 'excluded_from_consensus')[0])) == []
