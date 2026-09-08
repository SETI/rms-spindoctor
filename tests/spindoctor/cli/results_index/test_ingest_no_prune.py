"""What a pass told not to remove rows leaves behind, and what it stops reading.

An ingest deletes the rows of one root whose documents the walk no longer found,
so that presence of a row means the tree still holds the document it stands for.
``--no-prune`` gives that up, and the tests here are written in pairs: the same
tree, the same deletion, and the row's fate asserted both ways, so that neither
half passes on code that has stopped telling the two apart.

Absence is the half nothing here can weaken.  Skipping a delete adds no row, so
every answer read from absence -- "this image was never navigated" among them --
means exactly what it meant, and there is no test of that here for the same
reason there is no code for it.

What the flag saves is also pinned, because it is the half an operator acts on
and it differs between the two passes that prune.  A pass that reads the
documents itself reads what the index already holds for the skip rule as well as
for the delete, so only a forced pass stops reading it; a fan-out reads it for
the delete alone, so it stops reading it either way.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.results_index.ingest_driver_helpers import (
    fanned_out,
    process,
    run_driver,
    tasks_of,
)
from tests.spindoctor.conftest import index_url, metadata_document, write_metadata

from spindoctor.cli.results_index import IngestCounts, fan_out_ingest_tasks, ingest_metadata_files
from spindoctor.cli.results_index import driver as driver_module
from spindoctor.cli.results_index import tasks as tasks_module
from spindoctor.cli.results_index.store import _RecordedFile
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import FAILED_FILES, IMAGES, open_index

KEPT = 'VOL/N1454725799_1_CALIB'
"""Stub of the document that stays in the tree between the two passes."""

LEFT = 'VOL/N1454725800_1_CALIB'
"""Stub of the document that leaves the tree between the two passes."""


def _tree_of_two(tmp_path: Path) -> tuple[Path, Path]:
    """Write a results tree holding two navigation documents.

    Parameters:
        tmp_path: Directory the tree is written under.

    Returns:
        The results root, and the document that is about to be deleted.
    """
    root = tmp_path / 'results'
    write_metadata(root, KEPT, metadata_document(image_name='N1454725799_1_CALIB.IMG'))
    leaving = write_metadata(root, LEFT, metadata_document(image_name='N1454725800_1_CALIB.IMG'))
    return root, leaving


def _ingest(
    url: str,
    root: Path,
    *,
    logger: pdslogger.PdsLogger,
    force: bool = False,
    prune: bool = True,
) -> IngestCounts:
    """Run one ingest pass over one root.

    Parameters:
        url: The index to create or add to.
        root: The results root to walk.
        logger: Logger the pass reports through.
        force: Whether to re-read every document.
        prune: Whether to remove the rows of documents that have left the tree.

    Returns:
        What the pass did.
    """
    engine = open_index(url, create=True)
    try:
        return ingest_metadata_files(
            engine,
            [root.as_posix()],
            force=force,
            prune=prune,
            logger=logger,
            tuning=TreeTuning(),
        )
    finally:
        engine.dispose()


def _fan_out(
    url: str, root: Path, *, logger: pdslogger.PdsLogger, prune: bool = True
) -> list[dict[str, Any]]:
    """Divide one root into ingest tasks.

    Parameters:
        url: The index to create or add to.
        root: The results root to list.
        logger: Logger the fan-out reports through.
        prune: Whether to remove the rows of documents that have left the tree.

    Returns:
        The task descriptions it cut.
    """
    engine = open_index(url, create=True)
    try:
        return fan_out_ingest_tasks(
            engine, [root.as_posix()], prune=prune, logger=logger, tuning=TreeTuning()
        ).tasks
    finally:
        engine.dispose()


def _stubs(url: str, table: sqlalchemy.Table) -> list[str]:
    """Return the stubs one table of an index holds, in order.

    Parameters:
        url: The index to read.
        table: The table to read them from.

    Returns:
        The stubs.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.select(table.c.results_path_stub).order_by(table.c.results_path_stub)
            )
            return [str(row.results_path_stub) for row in rows]
    finally:
        engine.dispose()


def _counting_recorded_files(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[str]:
    """Count the calls one module makes to the recorded-rows query.

    The query is a statement over every row a root holds, so whether it runs at
    all is the saving being claimed, and a test that reads only the rows left
    behind cannot tell a pass that skipped it from one that ran it and threw the
    answer away.

    Parameters:
        monkeypatch: Fixture the query is replaced through.
        module: The module whose binding is replaced.

    Returns:
        A list that gains the root of each call as it happens.
    """
    calls: list[str] = []
    original = module._recorded_files

    def counting(
        connection: sqlalchemy.Connection,
        root_url: str,
        *,
        stubs: Sequence[str] | None = None,
    ) -> dict[str, _RecordedFile]:
        calls.append(root_url)
        return dict(original(connection, root_url, stubs=stubs))

    monkeypatch.setattr(module, '_recorded_files', counting)
    return calls


# ---------------------------------------------------------------------------
# What becomes of the row of a document that has left the tree
# ---------------------------------------------------------------------------


def test_a_pass_that_does_not_prune_keeps_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The relaxation itself: a row is allowed to outlive its document."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _ingest(url, root, logger=quiet_logger, prune=False)
    assert _stubs(url, IMAGES) == [KEPT, LEFT]


def test_a_pass_that_prunes_removes_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The control: the same tree and the same deletion, pruned."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _ingest(url, root, logger=quiet_logger, prune=True)
    assert _stubs(url, IMAGES) == [KEPT]


def test_a_pass_that_does_not_prune_reports_no_removals(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is of rows this pass deleted, and it deleted none."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    counts = _ingest(url, root, logger=quiet_logger, prune=False)
    assert counts.files_removed == 0


def test_a_pass_that_prunes_reports_the_removal(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The control for the count."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    counts = _ingest(url, root, logger=quiet_logger, prune=True)
    assert counts.files_removed == 1


def test_a_pass_that_does_not_prune_keeps_the_refusal_of_a_deleted_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The prune reads both tables, so leaving it out has to leave both alone.

    A refusal that outlives its file is what makes the next pass go on skipping
    a file the tree no longer holds, which is the same relaxation seen from the
    bookkeeping side.
    """
    root, _leaving = _tree_of_two(tmp_path)
    refused = root / 'edges_metadata.json'
    refused.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    refused.unlink()
    _ingest(url, root, logger=quiet_logger, prune=False)
    assert _stubs(url, FAILED_FILES) == ['edges']


def test_a_pass_that_prunes_removes_the_refusal_of_a_deleted_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The control for the refusal."""
    root, _leaving = _tree_of_two(tmp_path)
    refused = root / 'edges_metadata.json'
    refused.write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    refused.unlink()
    _ingest(url, root, logger=quiet_logger, prune=True)
    assert _stubs(url, FAILED_FILES) == []


# ---------------------------------------------------------------------------
# What is no longer read, and what still is
# ---------------------------------------------------------------------------


def test_a_forced_pass_that_does_not_prune_reads_no_recorded_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is left to want the answer, so the query over the root is not run."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    calls = _counting_recorded_files(monkeypatch, driver_module)
    _ingest(url, root, logger=quiet_logger, force=True, prune=False)
    assert calls == []


def test_an_unforced_pass_that_does_not_prune_still_reads_the_recorded_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skip rule is the other reader, and it is still asking."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    calls = _counting_recorded_files(monkeypatch, driver_module)
    _ingest(url, root, logger=quiet_logger, prune=False)
    assert calls == [root.as_posix()]


def test_a_forced_pass_that_prunes_still_reads_the_recorded_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prune is the other reader, and a forced pass that prunes needs it."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    calls = _counting_recorded_files(monkeypatch, driver_module)
    _ingest(url, root, logger=quiet_logger, force=True, prune=True)
    assert calls == [root.as_posix()]


def test_a_forced_pass_that_prunes_still_removes_a_deleted_documents_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """What the reading is for: a forced pass deletes exactly as an ordinary one does."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _ingest(url, root, logger=quiet_logger, force=True, prune=True)
    assert _stubs(url, IMAGES) == [KEPT]


def test_a_pass_that_does_not_prune_still_skips_an_unchanged_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The saving must not reach the skip rule, which is what makes a re-ingest cheap."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    counts = _ingest(url, root, logger=quiet_logger, prune=False)
    assert counts.files_skipped == 2


# ---------------------------------------------------------------------------
# The fan-out, which reads the recorded rows for the prune alone
# ---------------------------------------------------------------------------


def test_a_fan_out_that_does_not_prune_keeps_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The one step of a queue-divided pass that removes a row does not."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _fan_out(url, root, logger=quiet_logger, prune=False)
    assert _stubs(url, IMAGES) == [KEPT, LEFT]


def test_a_fan_out_that_prunes_removes_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The control for the fan-out."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _fan_out(url, root, logger=quiet_logger, prune=True)
    assert _stubs(url, IMAGES) == [KEPT]


def test_a_fan_out_that_does_not_prune_reads_no_recorded_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fan-out reads them for the delete alone, so the query goes with it."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    calls = _counting_recorded_files(monkeypatch, tasks_module)
    _fan_out(url, root, logger=quiet_logger, prune=False)
    assert calls == []


def test_a_fan_out_that_prunes_reads_the_recorded_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: an unforced fan-out that prunes still runs the query."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    calls = _counting_recorded_files(monkeypatch, tasks_module)
    _fan_out(url, root, logger=quiet_logger, prune=True)
    assert calls == [root.as_posix()]


def test_a_fan_out_that_does_not_prune_still_cuts_every_document_into_a_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The shares come from the listing, which the flag does not touch."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = _fan_out(url, root, logger=quiet_logger, prune=False)
    assert len(tasks[0]['data']['files']) == 2


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    root: Path,
    *extra: str,
    command: str = 'ingest',
) -> tuple[int | None, list[str]]:
    """Run one ``sd_results_index`` subcommand over one root.

    Parameters:
        tmp_path: Directory the logs are written under.
        monkeypatch: Fixture the driver is run through.
        url: The index URL.
        root: The results root.
        *extra: Further arguments.
        command: The subcommand to run.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(
        [command, '--results-index-db', url, '--nav-results-root', root.as_posix(), *extra],
        monkeypatch,
        tmp_path,
    )


def test_the_command_line_leaves_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag reaches the pass rather than being parsed and dropped."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _run(tmp_path, monkeypatch, url, root, '--no-prune')
    assert _stubs(url, IMAGES) == [KEPT, LEFT]


def test_the_command_line_says_the_pass_did_not_prune(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log read later has to say which guarantee the index was built under."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _status, written = _run(tmp_path, monkeypatch, url, root, '--no-prune')
    assert any('left in place' in line for line in written)


def test_a_pass_that_prunes_says_how_many_rows_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the summary line, which the flag chooses between."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _status, written = _run(tmp_path, monkeypatch, url, root)
    assert any('Rows removed, their document gone from the tree: 1' in line for line in written)


def test_a_fan_out_from_the_command_line_says_it_did_not_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other pass that prunes reports the same way."""
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _status, written = _run(
        tmp_path,
        monkeypatch,
        url,
        root,
        '--tasks-file',
        str(tmp_path / 'tasks.json'),
        '--no-prune',
        command='divide',
    )
    assert any('left in place' in line for line in written)


def test_a_fan_out_from_the_command_line_leaves_the_row_of_a_deleted_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag has to reach the fan-out too, not only the summary it writes."""
    root, leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    leaving.unlink()
    _run(
        tmp_path,
        monkeypatch,
        url,
        root,
        '--tasks-file',
        str(tmp_path / 'tasks.json'),
        '--no-prune',
        command='divide',
    )
    assert _stubs(url, IMAGES) == [KEPT, LEFT]


def _a_completed_fan_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Divide a root into tasks, run them, and write the log that completes the run.

    A completion that would otherwise exit 0 is what makes a refusal of it worth
    asserting on: over a root nobody fanned out, or an index that is not there,
    the program exits 1 whether or not it refused anything.

    Parameters:
        tmp_path: Directory the tree, the index, the tasks and the log live
            under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The index URL.
    """
    url = fanned_out(tmp_path, monkeypatch, count=2)
    results = [process(task['data'], url)[1] for task in tasks_of(tmp_path / 'tasks.json')]
    events = [
        json.dumps({'event_type': 'task_completed', 'task_id': f'ingest-{n}', 'result': result})
        for n, result in enumerate(results)
    ]
    (tmp_path / 'events.log').write_text(
        ''.join(f'{event}\n' for event in events), encoding='utf-8'
    )
    return url


def _complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str, *extra: str
) -> tuple[int | None, list[str]]:
    """Add up what the workers did over the root ``_a_completed_fan_out`` divided.

    Parameters:
        tmp_path: Directory the tree, the index and the log live under.
        monkeypatch: Fixture the driver is run through.
        url: The index URL.
        *extra: Further arguments.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return _run(
        tmp_path,
        monkeypatch,
        url,
        tmp_path / 'results',
        '--events-log',
        str(tmp_path / 'events.log'),
        *extra,
        command='complete',
    )


def test_a_completion_of_a_covered_root_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completion whose tasks account for the root exits 0.

    The control for the subcommand surface's refusal of --no-prune here: over a
    root nobody fanned out, or an index that is not there, this exits nonzero of
    its own accord and a refusal could not be told from it.
    """
    url = _a_completed_fan_out(tmp_path, monkeypatch)
    status, _written = _complete(tmp_path, monkeypatch, url)
    assert status == 0


def test_a_completion_reports_the_removals_without_claiming_it_made_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count added up from a fan-out is not reported as this pass's own removal."""
    url = _a_completed_fan_out(tmp_path, monkeypatch)
    _status, written = _complete(tmp_path, monkeypatch, url)
    removals = [line for line in written if 'Rows removed before the fan-out' in line]
    assert len(removals) == 1


def test_a_completion_does_not_say_a_document_left_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing recorded at a fan-out says whether it was removing rows at all."""
    url = _a_completed_fan_out(tmp_path, monkeypatch)
    _status, written = _complete(tmp_path, monkeypatch, url)
    claims = [line for line in written if 'their document gone from the tree' in line]
    assert claims == []


def test_a_drop_of_a_real_index_runs(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drop of an index that is really there exits 0.

    The control for the subcommand surface's refusal of --no-prune here: a drop
    pointed at a database that is not there exits 1 of its own accord, and a
    refusal could not be told from it.
    """
    root, _leaving = _tree_of_two(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    _ingest(url, root, logger=quiet_logger)
    status, _written = run_driver(
        ['drop', '--results-index-db', url, '--yes'], monkeypatch, tmp_path
    )
    assert status == 0
