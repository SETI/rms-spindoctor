"""Tree and index helpers for the results-index tests.

The document factories the trees here are built out of are shared with the rest
of the subtree and live in ``tests/spindoctor/conftest.py``; what is left here
is what a pass over a tree needs and the report does not.

The cloud-task helpers run the same pass in its three separate stages -- divide
a root into shares, ingest a share, add the shares up -- so that a test asserting
on one of them does not have to restate the other two.

The statistics tests beside these read the same helpers: a report over an index
is measured against an index something ingested, so the report's own fixtures
build one through the pass defined here rather than through a second definition
of it.
"""

import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.results_index.conftest import (
    postgres_decoy_schema,
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.results_index import (
    TaskCompletion,
    TaskResult,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_task_share,
)
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import INGEST_RUNS, open_index

# The results-index postgres tier runs against a schema of its own, exactly as
# the library tier does; re-exporting rather than restating keeps one definition
# of how that schema is created and dropped.
__all__ = ['postgres_decoy_schema', 'postgres_schema', 'postgres_server_url', 'postgres_url']


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, so raising its level cannot affect another test.
        The name carries a token that is unique for the life of the process:
        an object's address is not, since the object it belonged to is already
        collected and the next allocation is free to reuse it.
    """
    logger = pdslogger.PdsLogger(f'stats_test_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


PINNED_MTIME_NS = 1_700_000_000_000_000_000
"""Modification time given to documents whose metrics must not vary.

An arbitrary instant, in the past, with nanoseconds a filesystem storing whole
seconds would round away to the same value for every file given it.
"""


def write_metadata_in_each(
    roots: Sequence[Path], stub: str, document: dict[str, Any]
) -> list[Path]:
    """Write one document under several roots, indistinguishable but for its root.

    The rows of two roots holding one stub are told apart by the root half of
    the key alone, so a guard against a query that reads the stub alone has to
    hold when the two files match in every other respect -- same bytes, same
    size, same modification time.  Two writes microseconds apart usually do land
    on the same filesystem timestamp and occasionally do not, and a guard that
    depends on which is a guard that passes a root-blind lookup whenever the
    clock ticks between them.  So the time is set here rather than left to the
    clock.

    Parameters:
        roots: The results roots to write under.
        stub: The stub each of them holds.
        document: The document to write into each.

    Returns:
        The paths written.
    """
    written = []
    for root in roots:
        path = write_metadata(root, stub, document)
        os.utime(path, ns=(PINNED_MTIME_NS, PINNED_MTIME_NS))
        written.append(path)
    return written


_NOT_A_NAVIGATION_DOCUMENT = 'not_a_navigation_document'
"""The one key of a document that reads as JSON and holds no navigation result."""


def write_refusal_matching(root: Path, stub: str, document: Path) -> Path:
    """Write a document a pass refuses, matching another file's size and time.

    A refused file is recorded and skipped on the next pass on exactly the
    evidence an ingested one is: the size and the modification time the walk
    reports for it.  So the two halves of a two-root guard on the refusal table
    have to be indistinguishable but for their root, which means the length is
    asked for here rather than left to whatever a document happened to
    serialize to.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.
        document: The file whose size and modification time to match.

    Returns:
        The path written.

    Raises:
        ValueError: If the file to match is shorter than the smallest document
            of this shape, which cannot then be padded out to it.
    """
    metrics = document.stat()
    empty = json.dumps({_NOT_A_NAVIGATION_DOCUMENT: ''})
    if metrics.st_size < len(empty):
        raise ValueError(f'{document} holds {metrics.st_size} bytes, fewer than {len(empty)}')
    padding = 'x' * (metrics.st_size - len(empty))
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({_NOT_A_NAVIGATION_DOCUMENT: padding}), encoding='utf-8')
    os.utime(path, ns=(metrics.st_mtime_ns, metrics.st_mtime_ns))
    return path


def recorded_lines(
    logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch, *, level: str = 'info'
) -> list[str]:
    """Capture one level of a logger's output, rendered as it would be written.

    ``pdslogger`` writes through its own stream handler, so a test reads what a
    pass told an operator by standing in for the method rather than by capturing
    a stream.

    Parameters:
        logger: The logger to record.
        monkeypatch: Fixture the method is replaced through.
        level: Which method to record.

    Returns:
        The list the lines land in, which fills as the recorded code runs.
    """
    written: list[str] = []

    def recording(message: object, *args: object) -> None:
        written.append(str(message) % args if args else str(message))

    monkeypatch.setattr(logger, level, recording)
    return written


FIRST_STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the first document every fixture tree below writes."""


def build_tree(root: Path, count: int) -> list[str]:
    """Write a small results tree and return the stubs it holds.

    Parameters:
        root: The results root to write under.
        count: How many documents to write.

    Returns:
        The stubs, in the order the walk will report them.
    """
    stubs = []
    for index in range(count):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
        stubs.append(f'VOL/{name}')
    return sorted(stubs)


def root_strings(roots: Sequence[Path | str]) -> list[str]:
    """Render results roots as the strings a command line would carry.

    A root reaches a program as text, and two spellings of one root -- with and
    without a trailing separator -- are one root.  A test asking about that has
    to hand the spelling over untouched, which a ``Path`` cannot do: it drops a
    trailing separator the moment it is constructed.

    Parameters:
        roots: The roots, as paths or as the strings an operator typed.

    Returns:
        One string per root.
    """
    return [root.as_posix() if isinstance(root, Path) else root for root in roots]


def fan_out(
    url: str,
    roots: Sequence[Path | str],
    *,
    logger: pdslogger.PdsLogger,
    share_size: int = 2,
    **options: Any,
) -> list[dict[str, Any]]:
    """Create an index and divide the given roots into tasks.

    Parameters:
        url: The index URL to create.
        roots: The results roots to list.
        logger: Logger the fan-out reports through.
        share_size: How many files one task is handed.
        options: Further keyword arguments for the fan-out.

    Returns:
        The task descriptions.
    """
    engine = open_index(url, create=True)
    try:
        return fan_out_ingest_tasks(
            engine,
            root_strings(roots),
            share_size=share_size,
            logger=logger,
            tuning=TreeTuning(),
            **options,
        ).tasks
    finally:
        engine.dispose()


def run_shares(
    url: str, tasks: Sequence[dict[str, Any]], *, logger: pdslogger.PdsLogger
) -> list[TaskResult]:
    """Ingest every task's share, one after another, as one worker would.

    Parameters:
        url: The index URL, which must already carry the schema.
        tasks: The task descriptions.
        logger: Logger the shares report through.

    Returns:
        What each share returned, under the task that returned it, in task
        order.  A completion tells one task's report from another's by that
        identity, so the helper that runs the shares is where it is attached.
    """
    engine = open_index(url)
    try:
        return [
            TaskResult(
                task_id=str(task['task_id']),
                result=ingest_task_share(engine, task['data'], logger=logger, tuning=TreeTuning()),
            )
            for task in tasks
        ]
    finally:
        engine.dispose()


def reported(task_id: str, result: dict[str, Any]) -> TaskResult:
    """Return one hand-built task result under the task that reported it.

    Parameters:
        task_id: The identity the queue ran the task under.
        result: What that task returned.

    Returns:
        The pair a completion reads.
    """
    return TaskResult(task_id=task_id, result=result)


def complete(
    url: str,
    roots: Sequence[Path | str],
    results: Sequence[TaskResult],
    *,
    logger: pdslogger.PdsLogger,
) -> TaskCompletion:
    """Add up the shares of the given roots and stamp what they completed.

    Parameters:
        url: The index URL.
        roots: The results roots whose runs are being completed.
        results: What the shares returned.
        logger: Logger the completion reports through.

    Returns:
        The completion outcome.
    """
    engine = open_index(url)
    try:
        return complete_ingest_tasks(engine, root_strings(roots), results, logger=logger)
    finally:
        engine.dispose()


def rows_of(url: str, table: sqlalchemy.Table) -> list[tuple[Any, ...]]:
    """Return every row of one table, in a stable order.

    Parameters:
        url: The index URL.
        table: The table to read.

    Returns:
        The rows as tuples, ordered by their text columns so two indexes built
        by different routes compare equal when they hold the same rows.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = [tuple(row) for row in connection.execute(sqlalchemy.select(table))]
    finally:
        engine.dispose()
    return sorted(found, key=repr)


def run_rows(url: str) -> list[Any]:
    """Return every ingest run of an index, oldest first.

    Parameters:
        url: The index URL.

    Returns:
        The rows.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            return list(
                connection.execute(sqlalchemy.select(INGEST_RUNS).order_by(INGEST_RUNS.c.run_id))
            )
    finally:
        engine.dispose()


def cycle(
    tmp_path: Path, roots: Sequence[Path | str], *, logger: pdslogger.PdsLogger, share_size: int = 2
) -> str:
    """Fan out, ingest every share, and complete, over the given roots.

    Parameters:
        tmp_path: Directory the index is written into.
        roots: The results roots.
        logger: Logger every stage reports through.
        share_size: How many files one task is handed.

    Returns:
        The index URL.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, roots, logger=logger, share_size=share_size)
    results = run_shares(url, tasks, logger=logger)
    complete(url, roots, results, logger=logger)
    return url
