"""Tests for dividing an ingest pass into cloud tasks, and for one share of it.

The pass is split in three -- list the root and hand out shares, ingest a share,
add the shares up -- and each seam carries a guarantee the single-process pass
does not have to make.  This file covers the first two.  A share must write
exactly its own files and remove no row, because it knows nothing about the
stubs outside it; and the rows the workers write must be the rows one process
writes over the same tree, however they are scheduled.  Adding the shares up is
in ``test_ingest_cloud_tasks_completion``.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from tests.spindoctor.cli.results_index.conftest import (
    FIRST_STUB,
    build_tree,
    complete,
    cycle,
    fan_out,
    rows_of,
    run_rows,
    run_shares,
    write_metadata_in_each,
    write_refusal_matching,
)
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    technique,
    write_metadata,
    write_summary_png,
)

from spindoctor.cli.results_index import (
    TaskResult,
    fan_out_ingest_tasks,
    ingest_metadata_files,
    ingest_task_share,
)
from spindoctor.cli.results_index.store import _recorded_files
from spindoctor.nav_records import TreeTuning, UnlistableDirectoryError
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    STUBS_PER_STATEMENT,
    TECHNIQUES,
    normalize_root_url,
    open_index,
)

# ---------------------------------------------------------------------------
# Dividing a root up
# ---------------------------------------------------------------------------


def test_the_fan_out_writes_one_task_per_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Five documents in shares of two are three tasks, not one and not five."""
    root = tmp_path / 'results'
    build_tree(root, 5)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert len(tasks) == 3


def test_every_document_reaches_exactly_one_task(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A file in two shares is written twice; a file in none is never read."""
    root = tmp_path / 'results'
    stubs = build_tree(root, 5)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    handed = [entry['results_path_stub'] for task in tasks for entry in task['data']['files']]
    assert sorted(handed) == stubs


def _listing_in_reverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every directory listing arrive in the reverse of stub order.

    A directory listing arrives in whatever order the storage layer produced it,
    which for a local filesystem is the order the entries happen to sit in and
    for a bucket is the order the service returned.  A test that let the tree
    decide would therefore be asserting about this machine.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
    """
    real_iterdir = FCPath.iterdir_metadata

    def reversed_entries(self: FCPath) -> Any:
        yield from reversed(list(real_iterdir(self)))

    monkeypatch.setattr(FCPath, 'iterdir_metadata', reversed_entries)


def test_the_shares_are_cut_in_stub_order_however_the_storage_enumerated_the_tree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task file is an operator-visible artifact and describes one tree one way.

    The shares are cut straight off the listing, so the order the listing is
    held in decides which files are in which task.  Two fan-outs over one
    unchanged tree that cut different shares make a task file unreadable as a
    description of the tree and a re-run of one task a different piece of work.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, 5)
    _listing_in_reverse(monkeypatch)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    handed = [entry['results_path_stub'] for task in tasks for entry in task['data']['files']]
    assert handed == stubs


def test_each_share_holds_the_files_stub_order_puts_together(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cut, not just the concatenation: which task a file lands in is the promise."""
    root = tmp_path / 'results'
    stubs = build_tree(root, 5)
    _listing_in_reverse(monkeypatch)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    first = [entry['results_path_stub'] for entry in tasks[0]['data']['files']]
    assert first == stubs[:2]


def test_a_task_carries_the_metrics_the_walk_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The share is the walk's own evidence, so no worker stats a file itself."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    entry = tasks[0]['data']['files'][0]
    assert entry['size_bytes'] == (root / f'{FIRST_STUB}_metadata.json').stat().st_size


def test_a_file_that_is_not_a_document_is_handed_to_no_task(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The fan-out divides up the documents the walk found and nothing else.

    A summary PNG sits beside most documents a root holds, so a walk that
    counted one as a document would hand every worker a share of files it can
    only refuse.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, 2)
    write_summary_png(root, FIRST_STUB)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    handed = sorted(
        str(entry['results_path_stub']) for task in tasks for entry in task['data']['files']
    )
    assert handed == sorted(stubs)


def test_a_task_names_the_run_it_belongs_to(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Completion attributes a tally by the run rather than by guessing at it."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    assert tasks[0]['data']['run_id'] == run_rows(url)[0].run_id


def test_each_task_is_identified_separately(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A queue keys a task by its identifier, so two alike is one task lost."""
    root = tmp_path / 'results'
    build_tree(root, 5)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert len({task['task_id'] for task in tasks}) == len(tasks)


def test_the_tasks_of_two_roots_are_all_written(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One fan-out covers every root it was given, each with its own run."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 3)
    build_tree(second, 1)
    tasks = fan_out(index_url(tmp_path / 'index.sqlite3'), [first, second], logger=quiet_logger)
    assert len({task['data']['root_url'] for task in tasks}) == 2


def test_two_spellings_of_one_root_are_divided_up_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing separator is not another root, and one root is one listing.

    Listed twice, every document is handed out in two shares and read twice, and
    the first of the two runs is left unfinished for good.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    tasks = fan_out(
        index_url(tmp_path / 'index.sqlite3'),
        [root, f'{root.as_posix()}/'],
        logger=quiet_logger,
    )
    assert len(tasks) == 2


def test_two_spellings_of_one_root_are_one_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The run row is what a consumer reads, and a root has one of them per pass."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root, f'{root.as_posix()}/'], logger=quiet_logger)
    assert len(run_rows(url)) == 1


def test_a_share_of_no_files_is_refused(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A share size below one divides a root into no tasks and loses every file."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        with pytest.raises(ValueError, match='at least one file'):
            fan_out_ingest_tasks(
                engine, [root.as_posix()], share_size=0, logger=quiet_logger, tuning=TreeTuning()
            )
    finally:
        engine.dispose()


def test_a_root_that_is_not_there_is_counted_rather_than_fanned_out(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A mistyped root is not an empty one, so it yields no task and is reported."""
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        found = fan_out_ingest_tasks(
            engine,
            [str(tmp_path / 'absent')],
            share_size=2,
            logger=quiet_logger,
            tuning=TreeTuning(),
        )
    finally:
        engine.dispose()
    assert found.counts.roots_unreadable == 1


def test_a_root_that_is_not_there_yields_no_task(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing was listed under it, so there is nothing for a worker to read."""
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        found = fan_out_ingest_tasks(
            engine,
            [str(tmp_path / 'absent')],
            share_size=2,
            logger=quiet_logger,
            tuning=TreeTuning(),
        )
    finally:
        engine.dispose()
    assert found.tasks == []


# ---------------------------------------------------------------------------
# Removing the rows of documents that have left the tree
# ---------------------------------------------------------------------------


def test_the_fan_out_removes_a_row_whose_document_has_gone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The fan-out holds the one complete listing of the pass, so it prunes.

    Every stub a worker writes is one this listing held, and the prune deletes
    only stubs it did not hold, so nothing a worker is about to write can be
    deleted here however the workers are scheduled.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, 3)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    (root / f'{stubs[0]}_metadata.json').unlink()
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = list(connection.execute(sqlalchemy.select(IMAGES.c.results_path_stub)))
    engine.dispose()
    assert sorted(str(row.results_path_stub) for row in found) == stubs[1:]


def test_the_fan_out_reports_what_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is the only place a removal shows: no worker sees one."""
    root = tmp_path / 'results'
    stubs = build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    (root / f'{stubs[0]}_metadata.json').unlink()
    engine = open_index(url)
    try:
        found = fan_out_ingest_tasks(
            engine, [root.as_posix()], share_size=2, logger=quiet_logger, tuning=TreeTuning()
        )
    finally:
        engine.dispose()
    assert found.counts.files_removed == 1


def test_a_share_removes_no_row(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A worker holding part of a root would otherwise delete its peers' rows.

    The share here is one file of a root that holds three, so the other two are
    exactly the rows a worker that pruned on its own evidence would delete.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, 3)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    one_share = {
        'run_id': run_rows(url)[0].run_id,
        'root_url': normalize_root_url(root),
        'force': False,
        'has_file_metrics': True,
        'files': [
            {
                'results_path_stub': stubs[0],
                'mtime_ns': None,
                'size_bytes': None,
            }
        ],
    }
    engine = open_index(url)
    try:
        ingest_task_share(engine, one_share, logger=quiet_logger, tuning=TreeTuning())
        with engine.connect() as connection:
            found = list(connection.execute(sqlalchemy.select(IMAGES.c.results_path_stub)))
    finally:
        engine.dispose()
    assert sorted(str(row.results_path_stub) for row in found) == stubs


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_root_holding_a_directory_nobody_can_list_is_not_fanned_out(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A fan-out is the one moment a whole root is listed, so it has to be one.

    Shares cut from a listing of part of a root would be ingested, added up, and
    stamped as a completed pass over the root, and every stub the walk never
    reached would then read as an image nothing navigated.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    url = cycle(tmp_path, [root], logger=quiet_logger)
    closed = root / 'VOL2'
    closed.chmod(0o000)
    try:
        engine = open_index(url)
        try:
            with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
                fan_out_ingest_tasks(
                    engine,
                    [root.as_posix()],
                    share_size=2,
                    logger=quiet_logger,
                    tuning=TreeTuning(),
                )
        finally:
            engine.dispose()
    finally:
        closed.chmod(0o755)


# ---------------------------------------------------------------------------
# What a share writes
# ---------------------------------------------------------------------------


def test_the_shares_write_the_rows_a_single_pass_writes(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole point: the same tree, the same rows, however the work is split."""
    root = tmp_path / 'results'
    for index in range(4):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(
            root,
            f'VOL/{name}',
            metadata_document(
                image_name=f'{name}.IMG',
                per_technique=[technique('BodyLimbNav', (1.0, 2.0))],
            ),
        )
    write_summary_png(root, FIRST_STUB)
    (root / 'junk_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    divided = cycle(tmp_path, [root], logger=quiet_logger)
    serial = index_url(tmp_path / 'serial.sqlite3')
    engine = open_index(serial, create=True)
    try:
        ingest_metadata_files(engine, [root.as_posix()], logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert rows_of(divided, IMAGES) == rows_of(serial, IMAGES)


def test_the_shares_write_the_child_rows_a_single_pass_writes(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A per-technique row keys on the same pair and must arrive with its image."""
    root = tmp_path / 'results'
    for index in range(4):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(
            root,
            f'VOL/{name}',
            metadata_document(
                image_name=f'{name}.IMG',
                per_technique=[technique('BodyLimbNav', (1.0, 2.0))],
            ),
        )
    divided = cycle(tmp_path, [root], logger=quiet_logger)
    serial = index_url(tmp_path / 'serial.sqlite3')
    engine = open_index(serial, create=True)
    try:
        ingest_metadata_files(engine, [root.as_posix()], logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert rows_of(divided, TECHNIQUES) == rows_of(serial, TECHNIQUES)


def test_concurrent_shares_write_the_rows_a_single_pass_writes(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Several writers open the same SQLite file, which is what WAL is there for.

    Each share gets its own engine and runs at the same time as the others, so
    the write-ahead log and the busy timeout are what keep them from refusing
    one another.  The rows must come out as one process would have written them.
    """
    root = tmp_path / 'results'
    for index in range(12):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(
            root,
            f'VOL/{name}',
            metadata_document(
                image_name=f'{name}.IMG',
                per_technique=[technique('BodyLimbNav', (1.0, 2.0))],
            ),
        )
    divided = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(divided, [root], logger=quiet_logger, share_size=2)

    def one_worker(task: dict[str, Any]) -> TaskResult:
        engine = open_index(divided)
        try:
            return TaskResult(
                task_id=str(task['task_id']),
                result=ingest_task_share(
                    engine, task['data'], logger=quiet_logger, tuning=TreeTuning()
                ),
            )
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_worker, tasks))
    complete(divided, [root], results, logger=quiet_logger)
    serial = index_url(tmp_path / 'serial.sqlite3')
    engine = open_index(serial, create=True)
    try:
        ingest_metadata_files(engine, [root.as_posix()], logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert rows_of(divided, IMAGES) == rows_of(serial, IMAGES)


def test_concurrent_shares_lose_no_feature_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The child rows are written inside the image's own transaction.

    A worker that interleaved halves of an image would show up here rather than
    in the image table, since the child rows are the half that goes in second.
    """
    root = tmp_path / 'results'
    for index in range(12):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
    divided = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(divided, [root], logger=quiet_logger, share_size=2)

    def one_worker(task: dict[str, Any]) -> dict[str, Any]:
        engine = open_index(divided)
        try:
            return ingest_task_share(engine, task['data'], logger=quiet_logger, tuning=TreeTuning())
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one_worker, tasks))
    assert len(rows_of(divided, FEATURE_SOURCES)) == 24


def test_a_share_only_writes_its_own_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The stub is half of the key, so a lookup on it alone reaches two roots.

    Both trees here hold the same stub, and the second root's row is what a
    query filtering on the stub alone would overwrite.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, FIRST_STUB, metadata_document(status='success'))
    write_metadata(second, FIRST_STUB, metadata_document(status='failed', status_reason='blank'))
    url = cycle(tmp_path, [first, second], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = list(
            connection.execute(
                sqlalchemy.select(IMAGES.c.status).where(
                    IMAGES.c.root_url == normalize_root_url(second)
                )
            )
        )
    engine.dispose()
    assert [str(row.status) for row in found] == ['failed']


def test_a_share_reads_what_only_another_root_has_recorded(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The stub is half the key, so what a share has already read is per root.

    Both roots here hold the same stub, and the first is fully ingested.  A
    share of the second that asked about the stub alone would find a matching
    size and time, skip the file, and leave the second root with no row for an
    image it holds -- which every consumer reads as never navigated.  The two
    documents are given one modification time, so that "matching" holds whether
    or not the clock ticked between the two writes.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata_in_each([first, second], FIRST_STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    cycle_tasks = fan_out(url, [first], logger=quiet_logger)
    run_shares(url, cycle_tasks, logger=quiet_logger)
    tasks = fan_out(url, [second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    assert [found.result['files_ingested'] for found in results] == [1]


def test_a_share_reads_what_only_another_root_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal is recorded per root as well, and skipped on the same evidence.

    One root here holds a file that is not a navigation document; the other
    holds a real one at the same stub, of the same length and the same
    modification time.  A share of the second that asked about the stub alone
    would find the first root's refusal, decide nothing had changed, and skip a
    document it has never read -- leaving a navigated image with no row of its
    own, which every consumer reads as never navigated.
    """
    refusing = tmp_path / 'refusing'
    holding = tmp_path / 'holding'
    (document,) = write_metadata_in_each([holding], FIRST_STUB, metadata_document())
    write_refusal_matching(refusing, FIRST_STUB, document)
    url = index_url(tmp_path / 'index.sqlite3')
    run_shares(url, fan_out(url, [refusing], logger=quiet_logger), logger=quiet_logger)
    run_shares(url, fan_out(url, [holding], logger=quiet_logger), logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = list(connection.execute(sqlalchemy.select(IMAGES.c.root_url)))
    finally:
        engine.dispose()
    assert [str(row.root_url) for row in found] == [normalize_root_url(holding)]


def test_a_share_skips_what_the_index_has_already_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A retried task reads no document: its files match what is recorded.

    What it costs instead is a lookup over its own stubs, which is a bounded
    read of the index rather than a re-download of the share.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    assert [found.result['files_skipped'] for found in again] == [2]


def recorded_for(url: str, root: Path, stubs: Sequence[str] | None) -> list[str]:
    """Return the stubs the index answers about, for one root and one question.

    Parameters:
        url: The index URL.
        root: The results root to ask about.
        stubs: The stubs to ask about, or None to ask about the whole root.

    Returns:
        The stubs it answered with, in name order.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return sorted(_recorded_files(connection, normalize_root_url(root), stubs=stubs))
    finally:
        engine.dispose()


def test_a_share_asks_only_about_the_stubs_it_holds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A worker reads what the index records for its own files.

    Both tables answer, so both are narrowed: the root here holds two ingested
    documents and one file that was refused, and a share of one of them is
    entitled to hear about that one.  A lookup that ignored the stubs it was
    given would read every row of an archive-scale root, once per worker, to
    decide something about a few hundred files.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert recorded_for(url, root, stubs[:1]) == stubs[:1]


def test_a_share_of_no_files_asks_about_no_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming no stub asks about nothing, which is not asking about everything.

    The two differ in what they cost: answering a share that holds nothing with
    every row of its root is the one lookup a worker has no use for at all.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert recorded_for(url, root, []) == []


def test_a_share_wider_than_one_lookup_reads_none_of_it_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A share's stubs are asked about in batches, and every batch has to answer.

    The share here is one file wider than a batch, which is the size a
    production share crosses on every run.  A lookup answering from its first
    batch alone would leave a retried task re-reading every file past it -- the
    download the skip exists to avoid, on the tasks a queue redelivers.
    """
    root = tmp_path / 'results'
    stubs = build_tree(root, STUBS_PER_STATEMENT + 1)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger, share_size=len(stubs))
    run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    assert [found.result['files_skipped'] for found in again] == [len(stubs)]


def test_one_lookup_names_no_more_stubs_than_a_backend_will_carry() -> None:
    """A share is as wide as its fan-out made it, and one statement is not.

    Every stub of a batch is a bind parameter, and a backend caps how many one
    statement may carry: SQLite's own default cap has been 999, the smallest
    among the supported backends.  The batch is what holds a lookup under it
    whatever share size a fan-out was given, so what makes it a bound rather
    than a tuning choice is asserted.
    """
    assert STUBS_PER_STATEMENT <= 999


def test_a_forced_share_reads_everything_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """--force reaches the workers through the task rather than through a flag."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    forced = fan_out(url, [root], logger=quiet_logger, force=True)
    results = run_shares(url, forced, logger=quiet_logger)
    assert [found.result['files_ingested'] for found in results] == [2]


def test_a_share_with_no_metrics_reads_everything(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A listing that reports neither size nor time cannot answer "has it changed".

    The answer travels with the share, because it is a property of the one walk
    that produced it and no worker can find it out for itself.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    metric_less = dict(tasks[0]['data'], has_file_metrics=False)
    engine = open_index(url)
    try:
        result = ingest_task_share(engine, metric_less, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert result['files_ingested'] == 1


def test_a_share_names_every_file_it_could_not_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A worker has no run log, so the task result is where a refusal is named."""
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    named = [Path(name).name for name in results[0].result['failed_files']]
    assert sorted(named) == ['edges_metadata.json', 'rings_metadata.json']


def test_a_share_says_why_it_could_not_read_a_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The reason travels too, since nothing else can find it out afterwards.

    A file's name says nothing about what was wrong with it, and the program
    that adds the shares up never opens one.
    """
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    assert sum(results[0].result['failures_by_reason'].values()) == 1


def test_a_share_keeps_one_example_of_each_reason(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One reason with two files under it names one of them for the summary."""
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    examples = list(results[0].result['example_by_reason'].values())
    assert len(examples) == 1


def test_a_share_writes_its_rows_under_the_normalized_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A task file is an operator-visible artifact, and can be written by hand.

    The root is half of every row's key, so a share left to write under the
    spelling it was handed would produce rows no consumer's lookup matches --
    and the run would be stamped all the same, because the counts add up.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    handwritten = dict(tasks[0]['data'], root_url=f'{root.as_posix()}/')
    engine = open_index(url)
    try:
        ingest_task_share(engine, handwritten, logger=quiet_logger, tuning=TreeTuning())
        with engine.connect() as connection:
            found = list(connection.execute(sqlalchemy.select(IMAGES.c.root_url)))
    finally:
        engine.dispose()
    assert [str(row.root_url) for row in found] == [normalize_root_url(root)]


def test_a_share_counts_every_file_it_could_not_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The count is what makes a run's arithmetic add up, so it is asserted too."""
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    assert results[0].result['files_failed'] == 2


# ---------------------------------------------------------------------------
# A task nobody can read
# ---------------------------------------------------------------------------


def good_task(root: Path) -> dict[str, Any]:
    """Return a well-formed task's data for one file of a root.

    Parameters:
        root: The results root.

    Returns:
        The task data.
    """
    return {
        'run_id': 1,
        'root_url': normalize_root_url(root),
        'force': False,
        'has_file_metrics': True,
        'files': [
            {
                'results_path_stub': FIRST_STUB,
                'mtime_ns': 1,
                'size_bytes': 2,
            }
        ],
    }


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        ({'run_id': None}, 'run_id'),
        ({'root_url': None}, 'root_url'),
        ({'force': None}, 'force'),
        ({'has_file_metrics': None}, 'has_file_metrics'),
        ({'files': None}, 'files'),
        ({'run_id': True}, 'run_id'),
        ({'run_id': '1'}, 'run_id'),
        ({'root_url': 3}, 'root_url'),
        ({'force': 'yes'}, 'force'),
        ({'files': {}}, 'files'),
    ],
    ids=[
        'no-run-id',
        'no-root-url',
        'no-force',
        'no-metrics-flag',
        'no-files',
        'run-id-is-a-flag',
        'run-id-is-text',
        'root-url-is-a-number',
        'force-is-text',
        'files-is-an-object',
    ],
)
def test_a_task_of_another_shape_is_refused(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    mutation: dict[str, Any],
    message: str,
) -> None:
    """Every default available here would ingest the wrong files or the wrong root.

    ``None`` stands for the value's absence: each case removes the value the
    fan-out writes and asserts that the refusal names it.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    data = good_task(root)
    for key, value in mutation.items():
        if value is None:
            del data[key]
        else:
            data[key] = value
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        with pytest.raises(ValueError, match=message):
            ingest_task_share(engine, data, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        ('a string', 'not an object'),
        ({'mtime_ns': 1}, 'results_path_stub'),
        ({'results_path_stub': FIRST_STUB, 'mtime_ns': 1.5}, 'mtime_ns'),
        ({'results_path_stub': FIRST_STUB, 'mtime_ns': float('nan')}, 'mtime_ns'),
        ({'results_path_stub': FIRST_STUB, 'size_bytes': '2'}, 'size_bytes'),
        ({'results_path_stub': FIRST_STUB, 'size_bytes': True}, 'size_bytes'),
    ],
    ids=[
        'entry-is-text',
        'no-stub',
        'fractional-time',
        'time-is-not-a-number',
        'size-is-text',
        'size-is-a-flag',
    ],
)
def test_a_file_entry_of_another_shape_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, entry: Any, message: str
) -> None:
    """A stub with no metrics is ordinary; metrics of another type are not.

    A modification time that is not a whole number cannot be compared with the
    one recorded, and a value that is not a number at all -- NaN above all,
    which is unequal to itself and to what is stored -- would make every file
    look changed forever.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    data = good_task(root)
    data['files'] = [entry]
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        with pytest.raises(ValueError, match=message):
            ingest_task_share(engine, data, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()


@pytest.mark.parametrize('task_data', [None, 'a share', [1, 2]], ids=['none', 'text', 'a-list'])
def test_task_data_that_is_not_an_object_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, task_data: Any
) -> None:
    """A queue that delivers something else is a malformed task like any other.

    Reading a key out of it raises whatever the value's own type raises, which
    leaves the worker with an exception where its driver expects a refusal it
    can report as a task result.
    """
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        with pytest.raises(ValueError, match='not an object'):
            ingest_task_share(engine, task_data, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()


def test_a_file_entry_may_report_no_metrics(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A backend whose listing says neither is a documented case, not a bad task."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    data = good_task(root)
    data['has_file_metrics'] = False
    data['files'] = [
        {
            'results_path_stub': FIRST_STUB,
            'mtime_ns': None,
            'size_bytes': None,
        }
    ]
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        result = ingest_task_share(engine, data, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert result['files_ingested'] == 1


def test_a_file_carrying_no_metrics_is_read_again_whatever_its_task_claims(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An entry with no metrics has not been shown to be unchanged.

    Whether the listing reported metrics travels with the task rather than with
    each entry, so the two can disagree, and that is the one shape the skip
    cannot check against a file it never stats.  Compared as they stand, an
    entry carrying neither metric equals a row that recorded neither, and the
    file is then skipped by the pass after the one that wrote it and by every
    pass after that.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    data = good_task(root)
    data['files'] = [{'results_path_stub': FIRST_STUB, 'mtime_ns': None, 'size_bytes': None}]
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        ingest_task_share(engine, data, logger=quiet_logger, tuning=TreeTuning())
        again = ingest_task_share(engine, data, logger=quiet_logger, tuning=TreeTuning())
    finally:
        engine.dispose()
    assert again['files_ingested'] == 1
    assert again['files_skipped'] == 0
