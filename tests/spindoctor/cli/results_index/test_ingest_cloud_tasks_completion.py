"""Tests for adding the shares of a divided ingest back up.

A root must stay unreadable until every share is accounted for, because a task
that never reported leaves documents unread and absence of their rows is what
every consumer reads as "this image was never navigated".  What licenses the
stamp is arithmetic: the listing found so many files, and this run's own shares
must say what became of exactly that many.  Dividing the root up and ingesting
one share are in ``test_ingest_cloud_tasks``; what a completion reads out of the
workers' own results -- why their files were refused, and the event log they
come back in -- is in ``test_ingest_cloud_tasks_reports``; and which values are
a share's tally at all is in ``test_ingest_cloud_tasks_tallies``.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from tests.spindoctor.cli.results_index.conftest import (
    build_tree,
    complete,
    cycle,
    fan_out,
    reported,
    run_rows,
    run_shares,
)
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)

from spindoctor.cli.results_index import TaskResult
from spindoctor.nav_records import UnlistableDirectoryError
from spindoctor.results_index import (
    INGEST_RUNS,
    SCHEMA_VERSION,
    normalize_root_url,
    open_index,
    require_ingested_roots,
)

# ---------------------------------------------------------------------------
# Adding the shares up
# ---------------------------------------------------------------------------


def test_a_root_is_unreadable_between_the_fan_out_and_the_completion(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Half-written rows must not be read as an account of a root.

    This is what the deferred finish time buys: while the workers are running,
    a consumer asking about the root is told nobody has ingested it rather than
    being handed the shares that happen to have landed.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(root)], url=url)
    finally:
        engine.dispose()


def test_a_completed_root_is_readable(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """And afterwards it is, which is the other half of the same guarantee."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, [normalize_root_url(root)], url=url)
    finally:
        engine.dispose()


def test_the_shares_are_added_into_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Three tasks of two files each are six ingested documents on one run row."""
    root = tmp_path / 'results'
    build_tree(root, 6)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 6


def test_the_failures_of_every_share_are_added_into_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal is part of the account, and is what makes the arithmetic add up."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_failed == 1


def test_the_run_keeps_what_only_the_fan_out_saw(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """No share knows how many files the root holds, so the fan-out records it."""
    root = tmp_path / 'results'
    build_tree(root, 5)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_seen == 5


def test_the_run_keeps_the_rows_the_fan_out_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The removal happens before any worker runs, and is reported on the run row."""
    root = tmp_path / 'results'
    stubs = build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    (root / f'{stubs[0]}_metadata.json').unlink()
    tasks = fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], run_shares(url, tasks, logger=quiet_logger), logger=quiet_logger)
    assert run_rows(url)[-1].files_removed == 1


def unlistable_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the volume named ``VOL2`` refuse to be listed.

    A directory the walk cannot enumerate is the ordinary case on a shared tree
    -- a permission the run does not hold, a mount that stopped answering -- and
    it is the case where absence of a row means nothing at all.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
    """
    real_iterdir = FCPath.iterdir_metadata

    def refusing_vol2(self: FCPath) -> Any:
        if self.name == 'VOL2':
            raise PermissionError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing_vol2)


def test_a_fan_out_that_cannot_list_a_directory_completes_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fan-out is the only step that sees the whole root, so it is where this ends.

    Its shares would otherwise be ingested and added up into a completed pass
    over a root nobody listed whole, and every stub under the directory it
    missed would read as an image nothing navigated.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    unlistable_volume(monkeypatch)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        fan_out(url, [root], logger=quiet_logger)


def test_a_share_that_never_reported_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A task that failed read none of its documents.

    Stamping the run anyway would tell every consumer that absence of those
    images' rows means they were never navigated, which is exactly the claim
    the run row exists to license.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[:-1], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_share_that_never_reported_names_its_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unfinished run with nothing said about it is nothing anyone can act on."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], results[:-1], logger=quiet_logger)
    assert any(normalize_root_url(root) in named for named in outcome.roots_unaccounted)


def test_a_worker_that_reported_an_error_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An error result carries no tally, so its files are unaccounted for."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    failure = reported('ingest-1-000009', {'status': 'error', 'status_error': 'index_unopenable'})
    broken = [*results[:-1], failure]
    complete(url, [root], broken, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_worker_error_is_counted(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """And is told apart from a result of a shape nobody recognizes."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(
        url,
        [root],
        [*results, reported('ingest-1-000009', {'status': 'error', 'status_error': 'no_db'})],
        logger=quiet_logger,
    )
    assert outcome.results_failed == 1


def test_a_share_counted_twice_does_not_short_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A retried task reports its share a second time, as skipped.

    Its later report stands in for the earlier one, so the account is still one
    report per task and still covers every file the walk saw.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    retried = run_shares(url, tasks[:1], logger=quiet_logger)
    complete(url, [root], [*results, *retried], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_share_reported_twice_does_not_cover_for_one_that_never_ran(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The arithmetic must not let over- and under-accounting cancel.

    A queue redelivers a task whenever it could not see the delivery
    acknowledged, so one share reported twice while another never ran is an
    ordinary sequence.  Summed by files alone it reaches the number the walk
    found, and the run is stamped with two documents nobody read -- which every
    consumer would then read as two images that were never navigated.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    first = run_shares(url, tasks[:1], logger=quiet_logger)
    again = run_shares(url, tasks[:1], logger=quiet_logger)
    complete(url, [root], [*first, *again], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_share_reported_twice_leaves_its_roots_shortfall_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the shortfall is the one the tasks that never ran left behind."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    first = run_shares(url, tasks[:1], logger=quiet_logger)
    again = run_shares(url, tasks[:1], logger=quiet_logger)
    outcome = complete(url, [root], [*first, *again], logger=quiet_logger)
    assert any('2 of 4 file(s)' in named for named in outcome.roots_unaccounted)


def test_a_repeat_of_one_task_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Counted once, and said out loud, so the log records that it happened."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], [*results, *again], logger=quiet_logger)
    assert outcome.results_superseded == 1


def test_the_later_report_of_a_task_is_the_one_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A task that failed and was re-run reports its failure first.

    Reading the earlier report would leave a run unfinished though its documents
    are in the index, so the last thing a task said is what counts.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    failed = reported(str(tasks[0]['task_id']), {'status': 'error', 'status_error': 'no_index'})
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], [failed, *results], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_result_naming_no_task_is_not_counted_toward_a_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing tells such a result from a repeat of another, so it counts nowhere.

    Counting it could only ever inflate an account, and an inflated account is
    what stamps a run whose documents were never read.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    anonymous = [TaskResult(task_id=None, result=found.result) for found in results]
    complete(url, [root], anonymous, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_result_naming_no_task_is_counted_as_one(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And is reported, since a log full of them means the run cannot complete."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    anonymous = [TaskResult(task_id=None, result=found.result) for found in results]
    outcome = complete(url, [root], anonymous, logger=quiet_logger)
    assert outcome.results_unidentified == 1


def test_a_root_with_no_unfinished_run_is_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Completing a root nobody fanned out is a mistake worth reporting."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    engine.dispose()
    outcome = complete(url, [root], [], logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]


def test_completing_a_root_twice_names_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The second completion has nothing outstanding to stamp."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results, logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]


def test_a_run_abandoned_under_a_newer_finished_one_is_not_completed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only the newest run of a root is a candidate for a completion.

    A fan-out given up on and then made good by an ordinary pass leaves an older
    unfinished run under a newer finished one.  Stamping the older one would put
    a finish time on a walk nothing came back from, and would date the root's
    ingest to a pass that was abandoned.  So the completion reports the root as
    one it found no run for, and the abandoned run goes on saying it never
    finished -- which is what an operator reading the table sees.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    ingest_tree(url, [root], logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]
    assert run_rows(url)[0].finished_utc is None


def test_two_spellings_of_one_root_are_completed_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing slash is not another root, here or at the fan-out.

    Completed twice, the second pass finds the run it has just stamped and
    reports the root as one nobody divided up.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root, f'{root.as_posix()}/'], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == []


def test_completion_leaves_another_fan_outs_results_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A result naming a run this completion is not waiting on is not thrown away.

    It belongs to whoever completes that root, and is counted here so that a
    completion pointed at the wrong event log says so.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [first], results, logger=quiet_logger)
    assert outcome.results_unclaimed == 1


def test_a_run_is_credited_only_with_its_own_shares(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A tally belongs to the run its task named, not to whichever run is being stamped.

    Both roots are fanned out together here and both sets of shares are handed
    to a completion of the first alone, so a run credited with everything it was
    shown would record twice the files its own listing found.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [first], results, logger=quiet_logger)
    ingested = {str(row.root_url): row.files_ingested for row in run_rows(url)}
    assert ingested[normalize_root_url(first)] == 2


def test_completion_finishes_only_the_root_it_was_given(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One root's completion says nothing about another's, however they were fanned out."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [first], results, logger=quiet_logger)
    finished = {str(row.root_url): row.finished_utc for row in run_rows(url)}
    assert finished[normalize_root_url(second)] is None


def test_an_empty_root_completes(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A root that holds nothing yields no task, and must still finish its run.

    Otherwise a consumer would refuse it forever, reporting a root that exists
    and is empty as one nobody has ingested.  This root was listed and found to
    hold nothing, which is what no shares can account for; a root whose listing
    was never recorded is the case below, and must not complete.
    """
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_shortfall_records_what_the_shares_did(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The run stays unfinished, but the work the shares did is not lost.

    Their documents are in the index, and an operator reading the run row after
    a partial pass needs to see how far it got rather than the zeros the fan-out
    left there.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[:-1], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 2


def test_a_shortfall_records_what_its_shares_skipped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A pass whose shares mostly skipped got just as far as one that ingested.

    Recording only the documents read would write a row of zeros for a re-run of
    an unchanged tree, which is the case the incremental skip makes ordinary.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], again[:-1], logger=quiet_logger)
    assert run_rows(url)[0].files_skipped == 2


def test_a_shortfall_records_what_its_shares_could_not_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the refusals, which are the half an operator is most likely to chase."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[1:], logger=quiet_logger)
    assert run_rows(url)[0].files_failed == 2


# ---------------------------------------------------------------------------
# An account that runs past the listing
# ---------------------------------------------------------------------------


def test_an_account_past_the_listing_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Each task counts once, so the sum can only exceed the listing wrongly.

    Whatever produced it -- a hand-edited log, a result from somewhere else --
    is not an account of this run, and stamping on it would license the one
    claim the run bookkeeping exists to license from evidence that cannot be
    right.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    inflated = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 1000000,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [inflated], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_an_account_past_the_listing_names_its_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """With both numbers, since the excess is the whole of what is wrong."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    inflated = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 1000000,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    outcome = complete(url, [root], [inflated], logger=quiet_logger)
    assert any('1000000 of 4 file(s)' in named for named in outcome.roots_unaccounted)


# ---------------------------------------------------------------------------
# Shares written under another root
# ---------------------------------------------------------------------------


def discard_index(path: Path) -> None:
    """Delete a SQLite index and the side files it writes beside itself.

    This is the documented remedy for an index stamped with another schema
    version, and it is the step that makes a surrogate run identifier start
    again at one.

    Parameters:
        path: The database file.
    """
    for name in (path.name, f'{path.name}-wal', f'{path.name}-shm'):
        (path.parent / name).unlink(missing_ok=True)


def stale_shares_of_a_rebuilt_index(
    tmp_path: Path, logger: pdslogger.PdsLogger
) -> tuple[str, Path, list[TaskResult]]:
    """Run one root's tasks against an index rebuilt around another root.

    The sequence takes no hand-editing of anything.  A root is divided up, the
    index is deleted and rebuilt -- which is what a schema version mismatch is
    remedied by -- another root is divided into the fresh index and takes the
    run identifier the first one had, and the queue still holds the first root's
    tasks.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        logger: Logger every stage reports through.

    Returns:
        The index URL, the root that was divided up second, and what the first
        root's tasks reported.
    """
    first = tmp_path / 'root-a'
    second = tmp_path / 'root-b'
    build_tree(first, 4)
    build_tree(second, 4)
    database = tmp_path / 'index.sqlite3'
    url = index_url(database)
    stale = fan_out(url, [first], logger=logger)
    discard_index(database)
    fan_out(url, [second], logger=logger)
    return url, second, run_shares(url, stale, logger=logger)


def test_a_run_is_not_stamped_by_shares_of_another_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run number is unique only inside the index that minted it.

    The shares wrote real rows and reported real counts, and they add up to
    exactly what this run's listing found -- under a different root. Credited by
    run number alone they stamp a root with nothing under it, and every consumer
    then reads absence of a row there as "this image was never navigated".
    """
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    complete(url, [second], results, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_left_to_the_shares_of_another_is_named_short(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Named with nothing accounted for, which is what its own tasks reported."""
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    outcome = complete(url, [second], results, logger=quiet_logger)
    assert any('0 of 4 file(s)' in named for named in outcome.roots_unaccounted)


def test_shares_written_under_another_root_are_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Told apart from a result belonging to a fan-out nobody is completing here.

    That one is somebody else's to complete; this one names the very run being
    completed and is still not its share, which is a different thing to say and
    a different thing to do about it.
    """
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    outcome = complete(url, [second], results, logger=quiet_logger)
    assert outcome.results_of_another_root == 2


def test_a_root_left_to_the_shares_of_another_stays_unreadable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what a consumer asks, and the whole reason for not stamping it."""
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    complete(url, [second], results, logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(second)], url=url)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# A run that never recorded what its root holds
# ---------------------------------------------------------------------------


def begin_a_run(url: str, root: Path) -> None:
    """Record that an ingest of a root began, and nothing else about it.

    This is the row a pass leaves behind when it dies between starting and
    listing, and the row a fan-out leaves for a root it could not list at all.

    Parameters:
        url: The index URL to create or add to.
        root: The results root the run covers.
    """
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.insert().values(
                    root_url=normalize_root_url(root),
                    started_utc='2026-01-01T00:00:00+00:00',
                    schema_version=SCHEMA_VERSION,
                )
            )
    finally:
        engine.dispose()


def test_a_root_whose_listing_was_never_recorded_is_not_stamped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run that never established what its root holds cannot be accounted for.

    No files seen is not zero files seen.  Read as zero, a mistyped root
    completes with nothing under it, and every consumer then reads absence of a
    row as "this image was never navigated" for a whole tree nobody listed.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    complete(url, [root], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_whose_listing_was_never_recorded_is_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Told apart from a shortfall, because what to do about it is different.

    A shortfall is re-run the outstanding tasks; this is divide the root up
    again, because no task was ever cut from it.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    outcome = complete(url, [root], [], logger=quiet_logger)
    assert outcome.roots_unlisted == [normalize_root_url(root)]


def test_a_root_whose_listing_was_never_recorded_records_what_its_shares_did(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Such a run can still hold real work, and the row is where it shows.

    The run is not stamped and the root stays unreadable, but a share may have
    written rows under it all the same -- a task file that outlived the listing
    it was cut from is exactly that -- and an operator reading a row of zeros
    would conclude nothing was written.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    share = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [share], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 2


def test_a_root_the_fan_out_could_not_list_is_not_stamped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same rule, reached the way an operator reaches it: a mistyped root.

    The fan-out refuses it and records nothing on the run, so the completion has
    nothing to measure against and must not stamp it -- otherwise the mistyped
    root reads as a fully ingested empty tree.
    """
    # The mistyping of 'results' is the subject of this test.
    absent = tmp_path / 'nav-offset-reslts'
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [absent], logger=quiet_logger)
    complete(url, [absent], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_the_fan_out_could_not_list_stays_unreadable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what a consumer asks about, and the whole point of not stamping."""
    # The mistyping of 'results' is the subject of this test.
    absent = tmp_path / 'nav-offset-reslts'
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [absent], logger=quiet_logger)
    complete(url, [absent], [], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(absent)], url=url)
    finally:
        engine.dispose()
