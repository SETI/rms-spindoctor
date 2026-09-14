"""Tests for the two command lines that divide an ingest up and put it together.

``sd_results_index`` gains two modes: one lists each root, removes the rows whose
documents have left it, and writes the shares out; the other reads the workers'
event log, adds their tallies up, and stamps the runs those tallies account for.
What is pinned here is what each mode does to the index and what its exit status
says -- a mode that reads no document, a completion that refuses a root its
tasks did not cover, and a refusal that names its cause rather than escaping as
a failure nobody enumerated.

The worker between the two is in ``test_ingest_cloud_tasks_worker``.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from tests.spindoctor.cli.results_index.ingest_driver_helpers import (
    STUB,
    fanned_out,
    fanned_out_with_a_refusal,
    process,
    run_driver,
    tasks_of,
)
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    write_metadata,
)

from spindoctor.results_index import IMAGES, INGEST_RUNS, normalize_root_url, open_index

# ---------------------------------------------------------------------------
# The command line that divides the work up
# ---------------------------------------------------------------------------


def test_the_driver_writes_the_tasks_it_divided_the_root_into(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is what an operator loads into a queue."""
    fanned_out(tmp_path, monkeypatch, count=3)
    handed = [
        entry['results_path_stub']
        for task in tasks_of(tmp_path / 'tasks.json')
        for entry in task['data']['files']
    ]
    assert len(handed) == 3


def test_the_driver_reads_no_document_when_it_is_dividing_the_work_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The work is the workers'; this mode lists, removes and hands out."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES)
            ).scalar()
    finally:
        engine.dispose()
    assert found == 0


def dividing_a_root_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Divide up a root the walk cannot list.

    Parameters:
        tmp_path: Directory the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(
        [
            'divide',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_dividing_a_root_that_is_not_there_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule as a pass that reads the documents: a listing failed."""
    status, _written = dividing_a_root_that_is_not_there(tmp_path, monkeypatch)
    assert status == 1


def test_dividing_a_root_that_is_not_there_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status alone is what the catch-all produces for anything at all.

    What tells this refusal from a failure nobody enumerated is the message
    naming the thing that went wrong, so the message is what is asserted.
    """
    _status, written = dividing_a_root_that_is_not_there(tmp_path, monkeypatch)
    assert any('Roots that could not be listed' in line for line in written)


def test_two_spellings_of_one_root_are_logged_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run opens by naming the roots it will work over, not the words typed.

    A trailing separator is not another root, and every later message names the
    normalized spelling; a run that opened with two and then accounted for one
    would read as a root having gone missing between them.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    _status, written = run_driver(
        [
            'divide',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
            '--nav-results-root',
            f'{root.as_posix()}/',
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert f'Roots: {normalize_root_url(root)}' in written


def naming_a_root_that_is_not_a_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> tuple[int | None, list[str]]:
    """Run the driver over a spelling that does not name a location.

    Parameters:
        tmp_path: Directory the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.
        spelling: The root as it reaches the command line.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(
        [
            'divide',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            spelling,
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )


# What each spelling costs if it is walked instead of refused: a bare UNC path
# raises out of the storage layer, an empty one is the working directory, and
# one carrying a null byte renders and then fails at the first listing call.
_ROOTS_THAT_ARE_NOT_LOCATIONS = ['//', '', '\x00bad']


@pytest.mark.parametrize(
    'spelling', _ROOTS_THAT_ARE_NOT_LOCATIONS, ids=['no-share-name', 'nothing-at-all', 'null-byte']
)
def test_a_root_that_is_not_a_location_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """A root nothing can be read from is a run that did not complete.

    An empty spelling is the one an operator reaches by accident: it is what
    ``--nav-results-root "$ROOT"`` hands the program when the variable is unset,
    and walked rather than refused it ingests the working directory under a root
    nobody named and reports a completed pass.
    """
    status, _written = naming_a_root_that_is_not_a_location(tmp_path, monkeypatch, spelling)
    assert status == 1


@pytest.mark.parametrize(
    'spelling', _ROOTS_THAT_ARE_NOT_LOCATIONS, ids=['no-share-name', 'nothing-at-all', 'null-byte']
)
def test_a_root_that_is_not_a_location_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """And is charged to the root, rather than escaping as a failure nobody named.

    Every root is rendered absolute before the pass begins, so a spelling that
    is not a location is refused there with a message about it -- rather than
    reaching the catch-all as a traceback naming a directory listing, which is
    where a null byte in a path is otherwise found out.
    """
    _status, written = naming_a_root_that_is_not_a_location(tmp_path, monkeypatch, spelling)
    assert any('is not a location that can be read' in line for line in written)


# ---------------------------------------------------------------------------
# The command line that puts it back together
# ---------------------------------------------------------------------------


def write_event_log_of_results(path: Path, results: list[Any]) -> Path:
    """Write a cloud-tasks event log holding the given task results.

    Named for what it takes: the sibling helper in
    ``test_ingest_cloud_tasks_reports`` writes whole events, because what that
    module varies is the event around a result, while what this one varies is
    the run the results describe.

    Parameters:
        path: Where to write it.
        results: What each task returned, each wrapped in an event saying its
            task completed.

    Returns:
        The path written.
    """
    lines = [
        json.dumps({'event_type': 'task_completed', 'task_id': f'ingest-{n}', 'result': result})
        for n, result in enumerate(results)
    ]
    path.write_text(''.join(f'{line}\n' for line in lines), encoding='utf-8')
    return path


def test_the_driver_completes_the_run_from_an_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full cycle through the two command lines and the worker between them."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    results = [process(task['data'], url)[1] for task in tasks_of(tmp_path / 'tasks.json')]
    write_event_log_of_results(tmp_path / 'events.log', results)
    status, _written = run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 0


def test_completing_a_run_the_tasks_did_not_cover_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stamped without every share would license a wrong answer."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log_of_results(tmp_path / 'events.log', [])
    status, _written = run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_completing_a_run_the_tasks_did_not_cover_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shortfall is named, with the root, so an operator knows what to re-run."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log_of_results(tmp_path / 'events.log', [])
    _status, written = run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('0 of 3 file(s) accounted for' in line for line in written)


def test_the_completion_summary_says_why_a_file_was_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker has no run log, so its reasons reach this summary or nowhere.

    A results tree holds many ``*_metadata.json`` files that were never
    navigation documents; several hundred of those are ordinary, and several
    hundred navigation results that would not parse are not. The tally with one
    example file per reason is what tells the two apart, and a divided ingest
    must not be the configuration that loses it.
    """
    url = fanned_out_with_a_refusal(tmp_path, monkeypatch)
    results = [process(task['data'], url)[1] for task in tasks_of(tmp_path / 'tasks.json')]
    write_event_log_of_results(tmp_path / 'events.log', results)
    _status, written = run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    examples = [line for line in written if 'for example' in line]
    assert any('edges_metadata.json' in line for line in examples)


def test_a_result_written_under_another_root_is_named_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run number is only unique inside the index that minted it.

    A task file that outlived its index names a run of whatever was built next,
    and its shares add up to that run's listing while having written their rows
    somewhere else entirely. Saying so is what tells an operator which of the
    two they are looking at.
    """
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log_of_results(
        tmp_path / 'events.log',
        [
            {
                'status': 'ok',
                'run_id': 1,
                'root_url': str(tmp_path / 'elsewhere'),
                'files_ingested': 3,
                'files_skipped': 0,
                'files_failed': 0,
            }
        ],
    )
    _status, written = run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('reporting rows under a different root' in line for line in written)


def completing_a_mistyped_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Divide up a root that is not there, then complete it from an empty log.

    Parameters:
        tmp_path: Directory the index, the tasks file and the log live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status of the completion, and one entry per line it wrote to
        the main log.
    """
    # The mistyping of 'results' is the subject of this test.
    mistyped = tmp_path / 'nav-offset-reslts'
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            'divide',
            '--results-index-db',
            url,
            '--nav-results-root',
            str(mistyped),
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    write_event_log_of_results(tmp_path / 'events.log', [])
    return run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            str(mistyped),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_completing_a_root_whose_listing_was_never_recorded_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sequence an operator reaches it by: a mistyped root, then a completion.

    The fan-out refuses the root and records nothing about it, so the completion
    has nothing to measure its tasks against. Read as zero files, the mistyped
    root completes as a fully ingested empty tree and every consumer then reports
    the images under the real one as never navigated.
    """
    status, _written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert status == 1


def test_completing_a_root_whose_listing_was_never_recorded_says_which_refusal_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status of 1 is what the catch-all produces for anything whatever.

    This refusal is one the pass enumerates, and what tells it apart is the
    message naming the root and the correction to make -- divide it up again,
    rather than re-run the outstanding tasks, which is what a shortfall needs.
    """
    _status, written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert any('never recorded what its listing found' in line for line in written)


def test_completing_a_root_whose_listing_was_never_recorded_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And it must not reach the catch-all, whose traceback replaces the message."""
    _status, written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_a_root_whose_listing_was_never_recorded_keeps_its_unfinished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what a consumer reads, and the reason the status is 1."""
    url = index_url(tmp_path / 'index.sqlite3')
    completing_a_mistyped_root(tmp_path, monkeypatch)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            finished = list(connection.execute(sqlalchemy.select(INGEST_RUNS.c.finished_utc)))
    finally:
        engine.dispose()
    assert [row.finished_utc for row in finished] == [None]


def missing_event_log_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a fanned-out root against an event log that is not there.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    url = fanned_out(tmp_path, monkeypatch)
    return run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'nowhere.log'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_an_event_log_that_is_not_there_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped path is an ordinary operator error, and is charged to the file.

    The message is what says which failure this is: a status of 1 alone is the
    same status the catch-all produces for a failure nobody enumerated.
    """
    _status, written = missing_event_log_run(tmp_path, monkeypatch)
    assert any('Cannot read the task event log' in line for line in written)


def test_an_event_log_that_is_not_there_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass charges every failure it expects to one file or one root.

    A path that names no file is one it can charge, so it must not reach the
    catch-all, whose message says the run could not complete and whose traceback
    is what an operator gets instead of a correction to make.
    """
    _status, written = missing_event_log_run(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_an_event_log_that_is_not_there_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the status says the run did not complete, as it does for any refusal."""
    status, _written = missing_event_log_run(tmp_path, monkeypatch)
    assert status == 1


def binary_event_log_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a fanned-out root against a file that is not text.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    url = fanned_out(tmp_path, monkeypatch)
    (tmp_path / 'events.log.gz').write_bytes(b'\x1f\x8b\x08\x00\xff\xfe\x00\x00')
    return run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--events-log',
            str(tmp_path / 'events.log.gz'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_an_event_log_that_is_not_text_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path naming a compressed log or a database is the same operator error.

    It is a path that names the wrong thing, which the pass charges to the file,
    and the decoding failure it raises is a ValueError rather than an OSError --
    so a guard written for the missing-file case alone lets this one past.
    """
    _status, written = binary_event_log_run(tmp_path, monkeypatch)
    assert any('Cannot read the task event log' in line for line in written)


def test_an_event_log_that_is_not_text_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch-all's message and traceback are what it must not produce."""
    _status, written = binary_event_log_run(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_an_event_log_that_is_not_text_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the run is not completed, since nothing was read to complete it with."""
    status, _written = binary_event_log_run(tmp_path, monkeypatch)
    assert status == 1


def completing_a_root_nobody_divided_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a root that no fan-out ever covered.

    Parameters:
        tmp_path: Directory the tree, the index and the log live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = fanned_out(tmp_path, monkeypatch)
    write_event_log_of_results(tmp_path / 'events.log', [])
    return run_driver(
        [
            'complete',
            '--results-index-db',
            url,
            '--nav-results-root',
            str(tmp_path / 'other-results'),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_completing_a_root_nobody_divided_up_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no run to stamp, and saying so is the whole of the diagnosis."""
    status, _written = completing_a_root_nobody_divided_up(tmp_path, monkeypatch)
    assert status == 1


def test_completing_a_root_nobody_divided_up_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the root and the step to run over it is that whole diagnosis.

    The status is the same one every other refusal exits with, so the message is
    the only thing that distinguishes this from a failure nobody enumerated.
    """
    _status, written = completing_a_root_nobody_divided_up(tmp_path, monkeypatch)
    assert any('No unfinished ingest run to complete' in line for line in written)


def test_completing_against_an_index_that_is_not_there_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creating an empty one would report every root as never divided up.

    The runs the operator meant to complete are sitting in the index they meant
    to name, and an empty index beside it answers the question wrongly.
    """
    write_event_log_of_results(tmp_path / 'events.log', [])
    database = tmp_path / 'index.sqlite3'
    status, _written = run_driver(
        [
            'complete',
            '--results-index-db',
            index_url(database),
            '--nav-results-root',
            str(tmp_path / 'results'),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_completing_against_an_index_that_is_not_there_creates_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And leaves nothing behind on the way out."""
    write_event_log_of_results(tmp_path / 'events.log', [])
    database = tmp_path / 'index.sqlite3'
    run_driver(
        [
            'complete',
            '--results-index-db',
            index_url(database),
            '--nav-results-root',
            str(tmp_path / 'results'),
            '--events-log',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert not database.exists()
