"""One ingest pass divided into cloud tasks, and put back together again.

An ingest of an archive-scale root is one listing followed by a great many
independent document reads, which is exactly the shape a task queue serves.
The pass is therefore split in three, and the split is what this module is:

* **Fan-out.**  One program lists each root once, removes the rows whose
  documents have left the tree, records what the walk found on the root's
  ingest run, and divides the files it found into shares.  It is the only
  point in the pass that sees a whole root.
* **A share.**  A worker is handed a list of files with the metrics the
  fan-out's walk reported for them.  It skips the ones the index has already
  read unchanged, reads the rest, writes their rows, and returns what it did.
  It has no run log, so its tally is its return value.
* **Completion.**  The shares' tallies are added up and written to the ingest
  run, which is what finally stamps the root as ingested.

Why a worker never removes a row
--------------------------------

Deleting the rows of documents that have left the tree is licensed by a
complete listing of the root, and a worker holding a share of one has no
evidence about the stubs outside its share: deleting on it would delete its
peers' rows.  Nothing here hands a worker a listing, and the prune refuses
anything that is not a complete one, so the restriction is a property of the
seam rather than a rule a worker has to remember.

Why the fan-out removes them, and completion does not
-----------------------------------------------------

The fan-out is the one moment in the pass when a complete listing exists, and
it is the listing the shares were cut from: every stub a worker writes is one
the listing held, and the prune deletes only stubs it did not hold, so the two
sets cannot intersect however the workers are scheduled.  Removing at
completion instead would mean listing the whole root a second time -- the most
expensive thing an ingest does, and a paid round trip per directory on a cloud
root -- to act on evidence no share ever came from.

A fan-out told not to remove them reads nothing about the root at all.  The
recorded rows have one reader here -- the prune; a worker decides for itself
what its own files are worth reading -- so declining the delete drops the query
with it, whether or not the shares are to be read in full.

That disjointness is a claim about one fan-out.  Two overlapping fan-outs
against one root can leave a stale row behind -- a worker of the first writes a
stub after the second has read what is recorded and before it deletes, for a
document that left the tree between the two listings -- which the next pass
removes.  The prune is also destructive before any document has been read, so an
abandoned fan-out shrinks the index, but only by rows whose documents have
genuinely left the tree; and the run is unfinished throughout either way, so no
consumer reads the root while it is happening.

What makes a run finishable
---------------------------

The fan-out records how many files it saw.  A run is stamped only when the
shares account for exactly that many files between them, because a task that
never reported leaves its documents unread, and a run stamped without them
would tell every consumer that absence of their rows means those images were
never navigated.

Three things make that arithmetic mean what it says.  Each task's report counts
once, however many times the queue delivered it: over- and under-accounting
would otherwise cancel, and one share reported twice would cover for a share
that never ran.  A report counts toward a run only when it names the root that
run covers as well as the run itself, because a run identifier starts again at
one in a fresh index and a task file outliving its index would otherwise stamp
a root that has nothing under it.  And a run whose listing was never recorded --
a root nothing could list, a pass that died before it had one -- is never
stamped at all, because a run that never established what its root holds has
nothing for its shares to be measured against.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.results_index.chunks import _batched, _ingest_chunk
from spindoctor.cli.results_index.counts import IngestCounts
from spindoctor.cli.results_index.driver import (
    _files_to_read,
    _listing_of_root,
    _prune_missing,
)
from spindoctor.cli.results_index.runs import (
    _finish_run,
    _record_fan_out,
    _record_shares,
    _start_run,
    _unfinished_run,
)
from spindoctor.cli.results_index.store import _recorded_files
from spindoctor.nav_records import (
    ListedRecord,
    TreeTuning,
    distinct_roots,
    document_path,
    normalize_root_url,
)

__all__ = [
    'INGEST_TASK_SHARE_SIZE',
    'FanOut',
    'TaskCompletion',
    'TaskResult',
    'TaskResults',
    'complete_ingest_tasks',
    'fan_out_ingest_tasks',
    'ingest_task_share',
    'task_results_from_event_log',
]

_ReasonValue = TypeVar('_ReasonValue', int, str)
"""What one entry of a share's per-reason maps holds: a count, or an example."""

INGEST_TASK_SHARE_SIZE = 512
"""How many metadata files one task is handed.

The share bounds what a single task costs and what re-running one costs, and it
is what makes naming every file a share could not read a bounded thing to put in
a task result.
"""

TASK_COMPLETED_EVENT = 'task_completed'
"""Event type under which a worker's return value is written to the event log."""

_LARGEST_RUN_ROW_COUNT = 2**31 - 1
"""Most files an ingest run records as ingested, skipped or refused.

This is what one of those columns holds on the narrowest backend the index
supports, where they are 32-bit integers.  A share is a list of files a fan-out
cut from one listing, so a report of more files than any archive holds is not a
count of anything, and a concatenated event log carrying a foreign or corrupted
line is exactly where one comes from.  What reaches the column is the sum over a
run's shares rather than any one of them, so the running total is held to the
same bound as each count that goes into it; bounding the counts alone would
leave two accepted lines to overflow it between them.  Past either bound the
write fails and the database driver's own error takes the whole completion down,
for one bad line.
"""

_SHARE_COUNT_NAMES = ('files_ingested', 'files_skipped', 'files_failed')
"""The counts a share reports, which are the counts a run row records."""

_TASK_EVENT_PREFIX = 'task_'
"""Prefix of every event type that reports the outcome of one task."""


@dataclass
class _ShareCounts(IngestCounts):
    """A share's tally, which names every file it could not read.

    A pass over a whole root keeps one example file per reason, because a tree
    holding several hundred thousand documents that were never navigation
    results would otherwise be summarized by a list of them.  A share is
    bounded by the fan-out, so naming all of its failures is bounded too -- and
    a worker has no run log to name them in instead.

    Parameters:
        failed_files: Every file this share could not turn into rows.
    """

    failed_files: list[str] = field(default_factory=list)

    def record_failure(self, reason: str, source_file: str) -> None:
        """Count one file that could not be ingested, and name it.

        Parameters:
            reason: What was wrong with it, with nothing file-specific in it.
            source_file: The file, kept both as this reason's example and in
                the share's own list.
        """
        super().record_failure(reason, source_file)
        self.failed_files.append(source_file)


@dataclass(frozen=True)
class _Share:
    """One task's worth of an ingest, as the task data describes it.

    Parameters:
        run_id: The ingest run this share belongs to, echoed back so that
            completion can attribute the tally without guessing.
        root_url: Normalized URL of the root the files live under.
        force: Whether to read every document regardless of what is recorded.
        has_file_metrics: Whether the fan-out's listing reported both a size
            and a modification time for every file of this share.
        files: The files, with the metrics the listing reported for them.
    """

    run_id: int
    root_url: str
    force: bool
    has_file_metrics: bool
    files: list[ListedRecord]


@dataclass
class FanOut:
    """What dividing an ingest into tasks produced.

    Parameters:
        tasks: The task descriptions, in the shape a cloud-tasks queue loads:
            each a ``task_id`` and the ``data`` one worker is handed.
        counts: What the fan-out itself did -- the files its walks found, the
            rows it removed, and the roots it could not list at all.
    """

    tasks: list[dict[str, Any]] = field(default_factory=list)
    counts: IngestCounts = field(default_factory=IngestCounts)


@dataclass(frozen=True)
class TaskResult:
    """What one task returned, with the identity the queue ran it under.

    The identity travels with the value because a task is delivered more than
    once as a matter of course -- a queue redelivers whatever it could not see
    acknowledged, and an operator re-runs a task file after a partial failure --
    and two reports of one share are still one share.

    Parameters:
        task_id: The task's identifier, as the fan-out minted it, or None when
            the event carried none.
        result: The value ``process_task`` returned.
    """

    task_id: str | None
    result: dict[str, Any]


@dataclass
class TaskResults:
    """What one worker event log holds.

    Parameters:
        results: The values ``process_task`` returned, each with the task it
            came from, newest last.
        lines_unread: Lines of the log that are not JSON objects.  An event log
            is appended to while it is being written, so a partial last line is
            ordinary; a great many of them says the file is not an event log.
        tasks_unfinished: Tasks the log records as having ended without
            returning a value -- an exception, a timeout, a worker that exited.
            Their documents were never read, so the run they belong to cannot be
            stamped.
    """

    results: list[TaskResult] = field(default_factory=list)
    lines_unread: int = 0
    tasks_unfinished: int = 0


@dataclass(frozen=True)
class _ShareTally:
    """One share's report, read out of the value its worker returned.

    Parameters:
        run_id: The ingest run the share names.
        root_url: The normalized root its rows were written under, which is the
            other half of the key every one of those rows carries.
        counts: What the share did.
    """

    run_id: int
    root_url: str
    counts: IngestCounts


@dataclass
class TaskCompletion:
    """What adding the shares up did.

    Parameters:
        counts: The shares' tallies, summed.
        runs_completed: Ingest runs stamped as finished.
        roots_unaccounted: Roots whose shares did not account for exactly the
            files the fan-out saw, each named with the account and the listing.
            Their runs keep their NULL finish times.
        roots_unlisted: Roots whose run never recorded what its listing found,
            each named.  Nothing says what such a root holds, so no account of
            it can be complete and its run is never stamped.
        roots_without_a_run: Roots with no unfinished run to complete, each
            named.
        results_unclaimed: Task results naming a run none of the given roots is
            waiting on.
        results_of_another_root: Task results naming a run that is being
            completed, but reporting rows written under a different root.  A run
            identifier is unique only within the index that minted it, so a task
            file outliving its index names a run of whatever was built next.
        results_failed: Task results in which a worker reported an error
            instead of a share.
        results_unreadable: Task results that are not the shape a worker
            returns at all, together with those whose counts cannot be added to
            their run's account: the run row records the sum over its shares,
            and a sum larger than that row holds is not an account of a listing.
        results_superseded: Task results replaced by a later report of the same
            task, and therefore counted once between them.
        results_unidentified: Task results carrying no task identity.  One of
            them cannot be told from a repeat of another, so none is counted
            toward a run.
    """

    counts: IngestCounts = field(default_factory=IngestCounts)
    runs_completed: int = 0
    roots_unaccounted: list[str] = field(default_factory=list)
    roots_unlisted: list[str] = field(default_factory=list)
    roots_without_a_run: list[str] = field(default_factory=list)
    results_unclaimed: int = 0
    results_of_another_root: int = 0
    results_failed: int = 0
    results_unreadable: int = 0
    results_superseded: int = 0
    results_unidentified: int = 0


def _task_files(files: Sequence[ListedRecord]) -> list[dict[str, Any]]:
    """Render one share's files as the task data carries them.

    Parameters:
        files: The files of this share.

    Returns:
        One JSON object per file, carrying the stub and the two metrics the
        fan-out's listing reported for it, which is everything a worker needs
        to decide whether it has to read the file.  Where the document lives is
        not among them: it is the stub joined onto the root, and both of those
        already travel in the task.
    """
    return [
        {
            'results_path_stub': listed.stub,
            'mtime_ns': listed.mtime_ns,
            'size_bytes': listed.size_bytes,
        }
        for listed in files
    ]


def fan_out_ingest_tasks(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool = False,
    prune: bool = True,
    share_size: int = INGEST_TASK_SHARE_SIZE,
    logger: PdsLogger,
    tuning: TreeTuning,
) -> FanOut:
    """List each root once and divide the documents under it into tasks.

    Each root gets an ingest run, a single walk, and the removal of the rows
    whose documents the tree no longer holds.  What the walk found is recorded
    on the run immediately, because nothing later in the pass can find it out
    again; the finish time is left for :func:`complete_ingest_tasks`, so until
    then every consumer treats the root as one nobody has ingested.

    A root the walk could not list yields no task and keeps its unfinished run,
    exactly as it does in a pass that reads the documents itself: a mistyped or
    unmounted root is not an empty one.

    A fan-out told not to remove those rows reads nothing about the root, since
    the prune is the only thing here that asks what the index already holds: a
    worker decides what its own files are worth reading from the metrics its
    share carries.  A row may then outlive its document, while absence of a row
    keeps exactly the meaning it had.

    Two spellings of one root are one root, and are listed and divided up once.

    Parameters:
        engine: The open index, which must already carry the schema.
        roots: Navigation results roots, each normalized to the form the rows
            record and consumers compare against.
        force: Have the workers read every document, ignoring what is recorded.
        prune: Delete the rows of one root whose documents the listing did not
            find.  False keeps them and reads nothing about the root.
        share_size: How many files one task is handed.
        logger: Logger for the per-root scan summary.
        tuning: How much of each root's listing runs at once, which the
            program resolved from its configuration.

    Returns:
        The tasks, and what the fan-out itself did.

    Raises:
        ValueError: If the share size is not at least one file, which would
            divide a root into no tasks at all and lose every document under it.
        UnlistableDirectoryError: If a directory under any root could not be
            listed.  The fan-out ends there and writes no tasks at all, so a
            queue is never loaded with shares of a root nobody listed whole.
    """
    if share_size < 1:
        raise ValueError(f'a task share holds at least one file, not {share_size}')
    fan_out = FanOut()
    for root_url in distinct_roots(roots):
        counts = IngestCounts()
        run_id = _start_run(engine, root_url)
        logger.info('Dividing %s into ingest tasks', root_url)
        listing = _listing_of_root(root_url, logger=logger, tuning=tuning)
        if listing is None:
            counts.roots_unreadable = 1
            fan_out.counts.add(counts)
            continue
        counts.files_seen = len(listing.documents)
        if prune:
            with engine.connect() as connection:
                recorded = _recorded_files(connection, root_url)
            counts.files_removed = _prune_missing(engine, listing, recorded, logger=logger)
        _record_fan_out(engine, run_id, counts)
        tasks_of_root = [
            {
                'task_id': f'ingest-{run_id}-{index:06d}',
                'data': {
                    'run_id': run_id,
                    'root_url': root_url,
                    'force': force,
                    'has_file_metrics': listing.has_file_metrics,
                    'files': _task_files(share),
                },
            }
            for index, share in enumerate(_batched(listing.documents, share_size))
        ]
        fan_out.tasks.extend(tasks_of_root)
        logger.info(
            'Divided %d file(s) under %s into %d task(s)',
            counts.files_seen,
            root_url,
            len(tasks_of_root),
        )
        fan_out.counts.add(counts)
    return fan_out


def _required(task_data: dict[str, Any], key: str, kind: type) -> Any:
    """Read one required value of a task's data, refusing anything else.

    Parameters:
        task_data: The task data.
        key: The value's name.
        kind: The type it must be.

    Returns:
        The value.

    Raises:
        ValueError: If the value is absent or of another type.  A task that does
            not say what to ingest is not something to guess at: every default
            available here would ingest the wrong files or the wrong root.
    """
    if key not in task_data:
        raise ValueError(f'the task carries no "{key}"')
    value = task_data[key]
    if kind is bool:
        acceptable = isinstance(value, bool)
    else:
        # A bool is an int to isinstance, so an identifier declared as a whole
        # number would otherwise accept True and be used as the number one.
        acceptable = isinstance(value, kind) and not isinstance(value, bool)
    if not acceptable:
        raise ValueError(f'the task\'s "{key}" is {type(value).__name__}, not {kind.__name__}')
    return value


def _share_from_task(task_data: dict[str, Any]) -> _Share:
    """Read one task's data into the share it describes.

    The root is normalized here as well as at fan-out.  The rows a share writes
    carry it as half of their primary key, and a task file is an operator-visible
    artifact that can be written or edited by hand: a share left to write under
    an unnormalized spelling would produce rows no consumer's lookup matches,
    and the run would still be stamped because the counts add up.

    A relative spelling is resolved against the worker's own working directory,
    which is the rule every program resolves a root by and the reason a fan-out
    writes the absolute form.  Two workers on two machines handed a relative root
    write their shares under two different roots, and the run is then left
    unfinished and the shares named, since a share counts toward a run only when
    it names that run's root.

    Each file's document path is rebuilt from that root rather than carried,
    which is the same join the retrieval makes: a task carries the two metrics
    that decide whether a file has to be read and the stub that names it, and
    nothing that could disagree with the root it declares.

    Parameters:
        task_data: The task data, as the fan-out wrote it.

    Returns:
        The share.

    Raises:
        ValueError: If the task data is not the shape a fan-out produces,
            naming what was wrong with it.  A worker that guessed at a
            malformed task would write rows for files nobody asked about, under
            a root nobody named, and a consumer cannot tell such a row from a
            correct one.
    """
    if not isinstance(task_data, dict):
        # A queue that delivered something else is a malformed task like any
        # other, and the driver turns this into a task result rather than a
        # traceback out of the worker.
        raise ValueError(f'the task data is {type(task_data).__name__}, not an object')
    run_id = int(_required(task_data, 'run_id', int))
    root_url = normalize_root_url(str(_required(task_data, 'root_url', str)))
    force = bool(_required(task_data, 'force', bool))
    has_file_metrics = bool(_required(task_data, 'has_file_metrics', bool))
    entries = _required(task_data, 'files', list)
    files: list[ListedRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f'a "files" entry is {type(entry).__name__}, not an object')
        stub = str(_required(entry, 'results_path_stub', str))
        mtime_ns = entry.get('mtime_ns')
        size_bytes = entry.get('size_bytes')
        if mtime_ns is not None and (isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int)):
            raise ValueError(f'the "mtime_ns" of {stub} is not a whole number')
        if size_bytes is not None and (
            isinstance(size_bytes, bool) or not isinstance(size_bytes, int)
        ):
            raise ValueError(f'the "size_bytes" of {stub} is not a whole number')
        files.append(
            ListedRecord(
                stub=stub,
                path=document_path(root_url, stub),
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
            )
        )
    return _Share(
        run_id=run_id,
        root_url=root_url,
        force=force,
        has_file_metrics=has_file_metrics,
        files=files,
    )


def ingest_task_share(
    engine: sqlalchemy.Engine,
    task_data: dict[str, Any],
    *,
    logger: PdsLogger,
    tuning: TreeTuning,
) -> dict[str, Any]:
    """Ingest one share of a root and report what it did.

    The share is skipped, read and written by exactly the rules a pass over a
    whole root uses, over the files the fan-out handed it and no others.  It
    removes no row: nothing here holds a listing, and a share is not evidence
    about the stubs outside it.

    A task re-run over a share it already ingested reads nothing, because every
    one of its files now matches what the index records, which is what makes a
    retried task cheap and harmless.

    Parameters:
        engine: The open index, which must already carry the schema.
        task_data: The task data, as :func:`fan_out_ingest_tasks` wrote it.
        logger: Logger for the per-file failures.  A cloud task has no run log,
            so this discards what it is given and the tally is returned instead.
        tuning: How much of the retrieval runs at once and how many documents
            one transaction covers, which the worker resolved from its
            configuration.

    Returns:
        The task result: the run and root it belongs to, how many files it
        ingested, skipped and could not read, the name of every file it could
        not read, and how many failed for each reason with one example of each.
        The reasons travel because the program that adds the shares up has no
        other way to report them, and a divided ingest would otherwise report a
        count of unreadable files with nothing to say about them.

    Raises:
        ValueError: If the task data is not the shape a fan-out produces.
    """
    share = _share_from_task(task_data)
    counts = _ShareCounts()
    root = FCPath(share.root_url)
    stubs = [listed.stub for listed in share.files]
    with engine.connect() as connection:
        recorded = _recorded_files(connection, share.root_url, stubs=stubs)
    to_read = _files_to_read(
        share.files,
        recorded,
        force=share.force,
        has_file_metrics=share.has_file_metrics,
    )
    counts.files_skipped = len(share.files) - len(to_read)
    for chunk in _batched(to_read, tuning.ingest_commit_chunk_size):
        _ingest_chunk(
            engine,
            root,
            chunk,
            root_url=share.root_url,
            counts=counts,
            logger=logger,
            tuning=tuning,
        )
    return {
        'status': 'ok',
        'run_id': share.run_id,
        'root_url': share.root_url,
        'files_ingested': counts.files_ingested,
        'files_skipped': counts.files_skipped,
        'files_failed': counts.files_failed,
        'failed_files': counts.failed_files,
        'failures_by_reason': counts.failures_by_reason,
        'example_by_reason': counts.example_by_reason,
    }


def task_results_from_event_log(path: FCPath) -> TaskResults:
    """Read the values the workers returned out of a cloud-tasks event log.

    The log is JSON Lines, one event per line, written as each task ends.  Only
    the events that carry a return value are of interest here; the rest are
    counted, because a task that ended without returning one read none of its
    documents.  Each value keeps the identifier of the task it came from, which
    is what lets a task delivered twice be counted once.

    The file is read a line at a time rather than whole: an archive-scale run's
    log carries a line per task naming every file that task could not read.

    Parameters:
        path: The event log.

    Returns:
        The returned values, and what else the log held.

    Raises:
        OSError: If the log cannot be read.  A path that names no file is the
            caller's to report, since only the caller knows how the operator
            spelled it.
        UnicodeDecodeError: If it is not text at all.  A gzipped log, a database
            file or an image named by mistake is the same operator error as a
            path that names nothing, and is charged to the file the same way
            rather than escaping as a failure nobody enumerated.
    """
    found = TaskResults()
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                found.lines_unread += 1
                continue
            if not isinstance(event, dict):
                found.lines_unread += 1
                continue
            event_type = event.get('event_type')
            if not isinstance(event_type, str) or not event_type.startswith(_TASK_EVENT_PREFIX):
                continue
            if event_type != TASK_COMPLETED_EVENT:
                found.tasks_unfinished += 1
                continue
            result = event.get('result')
            if not isinstance(result, dict):
                found.tasks_unfinished += 1
                continue
            task_id = event.get('task_id')
            found.results.append(
                TaskResult(task_id=task_id if isinstance(task_id, str) else None, result=result)
            )
    return found


def _reason_map(value: Any, kind: type[_ReasonValue]) -> dict[str, _ReasonValue]:
    """Read one of a share's per-reason maps, keeping the entries it can read.

    These maps are the diagnosis rather than the account.  What licenses a run's
    stamp is the three counts beside them, which are complete without these, so
    an entry of another shape costs its own reason and nothing else -- where a
    count of another shape refuses the whole tally and leaves the run unfinished.

    Parameters:
        value: The map as the task result carried it, whatever it turned out
            to be.
        kind: What one entry holds: a count, or an example file.

    Returns:
        The entries that are a reason and a value of that kind.
    """
    if not isinstance(value, dict):
        return {}
    found: dict[str, _ReasonValue] = {}
    for reason, entry in value.items():
        if isinstance(reason, str) and isinstance(entry, kind) and not isinstance(entry, bool):
            found[reason] = entry
    return found


def _share_tally(result: dict[str, Any]) -> _ShareTally | None:
    """Read one task result as the tally of a share, or refuse it.

    Parameters:
        result: One value a worker returned.

    Returns:
        Which share it is and what it did, or None when the value is not a
        share's tally at all.  Its files are then unaccounted for, which is what
        stops the run being stamped.  A count outside what a share could report
        is refused with the rest: below zero it is not a number of files and
        would let one result cancel another's, and above
        :data:`_LARGEST_RUN_ROW_COUNT` it is a number the run's own column cannot
        hold, which would end the whole completion at the write rather than
        costing the one result it came in on.
    """
    run_id = result.get('run_id')
    if isinstance(run_id, bool) or not isinstance(run_id, int):
        return None
    root_url = result.get('root_url')
    if not isinstance(root_url, str):
        return None
    try:
        root_url = normalize_root_url(root_url)
    except ValueError:
        return None
    counts = IngestCounts()
    for name in _SHARE_COUNT_NAMES:
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if value < 0 or value > _LARGEST_RUN_ROW_COUNT:
            return None
        setattr(counts, name, value)
    counts.failures_by_reason = _reason_map(result.get('failures_by_reason'), int)
    counts.example_by_reason = _reason_map(result.get('example_by_reason'), str)
    return _ShareTally(run_id=run_id, root_url=root_url, counts=counts)


def _fits_the_run_row(running: IngestCounts, counts: IngestCounts) -> bool:
    """Whether one more share's counts leave a run's account writable.

    The bound is on the running total because the total is what is written: a
    run row records the sum over its shares, and each of two lines a share could
    legitimately have reported can be inside the column while their sum is not.

    Parameters:
        running: What this run's shares have reported so far.
        counts: What one more of them reported.

    Returns:
        Whether every count of the total is one the run row holds.
    """
    return all(
        getattr(running, name) + getattr(counts, name) <= _LARGEST_RUN_ROW_COUNT
        for name in _SHARE_COUNT_NAMES
    )


def _latest_of_each_task(
    results: Sequence[TaskResult], completion: TaskCompletion
) -> list[dict[str, Any]]:
    """Return one value per task: the last one that task reported.

    A queue delivers a task again whenever it could not see the last delivery
    acknowledged, and an operator re-runs a task file after a partial failure,
    so a log holds several reports of one share as a matter of course.  Adding
    them all up would let over- and under-accounting cancel: one share reported
    twice covers for a share that never ran, and the run is stamped with its
    documents unread.

    The last report wins rather than the first, because a task that failed and
    was re-run reports its failure first and its share second, and the later
    report is the one that says what the index now holds.

    Parameters:
        results: The values the workers returned, in the order the log holds
            them.
        completion: Outcome the repeats and the unidentifiable are counted on.

    Returns:
        One value per task.
    """
    latest: dict[str, dict[str, Any]] = {}
    for found in results:
        if found.task_id is None:
            # Nothing tells this apart from a repeat of another result, and
            # counting it could only ever inflate an account, so it counts
            # toward no run at all.
            completion.results_unidentified += 1
            continue
        if found.task_id in latest:
            completion.results_superseded += 1
        latest[found.task_id] = found.result
    return list(latest.values())


def complete_ingest_tasks(
    engine: sqlalchemy.Engine,
    roots: list[str],
    results: Sequence[TaskResult],
    *,
    logger: PdsLogger,
) -> TaskCompletion:
    """Add the shares up and stamp the runs they completed.

    A run is stamped only when its shares account for exactly the files the
    fan-out's walk saw, counting each task's report once however many times it
    was delivered.  A share that never reported -- a task that failed, timed
    out, or was never run -- leaves its documents unread, and stamping the run
    anyway would tell every consumer that the absence of their rows means those
    images were never navigated.  Such a root keeps its unfinished run and is
    named instead, with what its shares did recorded on the run so that an
    operator can see how far it got.  An account that runs past the listing is
    refused for the same reason from the other side: each task counts once, so
    the sum can only exceed the listing on a report that is not this run's.

    A share is this run's when it names both the run and the root the run
    covers.  The identifier alone is not enough: it is a surrogate that starts
    again at one in a fresh index, which is exactly what the remedy for a schema
    version mismatch produces, so a task file that outlived its index names a
    run of whatever was built next.  Credited by identifier alone, those shares
    stamp a root that has nothing under it.

    A run whose walk was never recorded is not stamped either, whatever its
    shares say.  A root nothing could list and a pass that died before it had a
    listing both leave a run that never established what the root holds, and
    zero files seen is what a root that was listed and is genuinely empty
    records -- so the two must not be read the same way.

    A root whose run this stamps closes by reporting how many refused documents
    the index then holds under it, on the same terms an undivided pass reports
    it: the root's own total, which is what the shares' tallies are not.

    Parameters:
        engine: The open index.
        roots: The navigation results roots whose runs are being completed.  Two
            spellings of one root are one root, completed once.
        results: The values the workers returned, each with the task it came
            from.
        logger: Logger for the per-root outcome.

    Returns:
        What was completed and what was not.
    """
    completion = TaskCompletion()
    by_share: dict[tuple[int, str], IngestCounts] = {}
    results_of_share: dict[tuple[int, str], int] = {}
    for result in _latest_of_each_task(results, completion):
        if result.get('status') != 'ok':
            # A worker that could not open the index, or was handed a task it
            # could not read, reports that instead of a tally.  Its files were
            # never read, so the run it belonged to comes up short and is left
            # unfinished; naming the count here is what says why.
            completion.results_failed += 1
            logger.error(
                'A task reported no share: %s', result.get('status_error', '(no reason given)')
            )
            continue
        tally = _share_tally(result)
        if tally is None:
            completion.results_unreadable += 1
            continue
        key = (tally.run_id, tally.root_url)
        running = by_share.setdefault(key, IngestCounts())
        if not _fits_the_run_row(running, tally.counts):
            # Refused for the reason each count is refused on its own: the sum
            # is what the run row is written from, and one it cannot hold ends
            # the whole completion in the database driver's own error rather
            # than costing the line it arrived on.  The run then comes up short
            # and is named, as it is for every other result nobody can read.
            completion.results_unreadable += 1
            logger.error(
                'A task result would put the account of %s past the %d file(s) an ingest '
                'run records, so it is not counted: no share of a listing reports that '
                'many, and a log holding one is not an account of this run',
                tally.root_url,
                _LARGEST_RUN_ROW_COUNT,
            )
            continue
        running.add(tally.counts)
        results_of_share[key] = results_of_share.get(key, 0) + 1
    claimed: set[tuple[int, str]] = set()
    for root_url in distinct_roots(roots):
        with engine.connect() as connection:
            run = _unfinished_run(connection, root_url)
        if run is None:
            completion.roots_without_a_run.append(root_url)
            continue
        # Both halves of the share's own identity, because a run identifier is
        # a surrogate that starts again at one in a fresh index -- which the
        # documented remedy for a version mismatch creates -- and a task file
        # written before that rebuild names a run of whatever was built next.
        claimed.add((run.run_id, run.root_url))
        counts = by_share.get((run.run_id, run.root_url), IngestCounts())
        if run.files_seen is None:
            # What its shares did is recorded for the same reason a shortfall's
            # is: the run stays unreadable either way, and an operator reading
            # the row can see whether anything was written under this root.
            _record_shares(engine, run.run_id, counts)
            completion.roots_unlisted.append(root_url)
            logger.error(
                'The ingest run of %s never recorded what its listing found, so there is '
                'nothing its tasks can be measured against and the run is left unfinished: '
                'divide the root up again, and complete the run that fanned out',
                root_url,
            )
            completion.counts.add(counts)
            continue
        counts.files_seen = run.files_seen
        counts.files_removed = run.files_removed or 0
        accounted = counts.files_ingested + counts.files_skipped + counts.files_failed
        if accounted != counts.files_seen:
            # What the shares that did report did is written down without a
            # finish time: the run stays unreadable, and an operator inspecting
            # it can see how far the pass got rather than a row of zeros.
            _record_shares(engine, run.run_id, counts)
            completion.roots_unaccounted.append(
                f'{root_url} ({accounted} of {counts.files_seen} file(s) accounted for)'
            )
            logger.error(
                'The tasks of %s account for %d of the %d file(s) its listing found, so its '
                'ingest run is left unfinished: absence of a row under it is not evidence '
                'that its image was never navigated'
                if accounted < counts.files_seen
                else 'The tasks of %s account for %d file(s) where its listing found %d, so '
                'its ingest run is left unfinished: each task counts once, so an account '
                "can only run past a listing on results that are not this run's",
                root_url,
                accounted,
                counts.files_seen,
            )
            completion.counts.add(counts)
            continue
        _finish_run(engine, run.run_id, counts)
        completion.runs_completed += 1
        logger.info(
            'Completed the ingest of %s: %d ingested, %d skipped, %d failed of %d file(s)',
            root_url,
            counts.files_ingested,
            counts.files_skipped,
            counts.files_failed,
            counts.files_seen,
        )
        completion.counts.add(counts)
    claimed_runs = {run_id for run_id, _root_url in claimed}
    for key, count in results_of_share.items():
        if key in claimed:
            continue
        if key[0] in claimed_runs:
            completion.results_of_another_root += count
        else:
            completion.results_unclaimed += count
    return completion
