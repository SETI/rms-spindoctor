#!/usr/bin/env python3
"""Read one or more navigation results roots into the results index.

Dispatch script for the ``sd_results_index`` console entry point.  See
``spindoctor.cli.results_index`` for what one pass does and
``spindoctor.cli.stats`` for the statistics-system overview.

The program is a subject with four verbs under it, one of which every command
line names:

``ingest``
    Walk each named results root and read the documents under it into the
    index.
``divide``
    List each root once, remove the rows whose documents have left it, and
    write out the shares ``sd_results_index_cloud_tasks`` reads.
``complete``
    Read the event log those workers wrote, add their tallies up, and stamp
    each named root's ingest as finished.  Until that step a consumer treats
    the roots as ones nobody has ingested, which is what keeps absence of a row
    from being read as an answer while the workers are still writing.
``drop``
    Remove the index's own tables from the database and stop, reading no tree.

Every option belongs to the verbs that act on it and to no others, so a command
line asking for two different things is a usage error naming the option rather
than a program deciding which half of it to do.  ``--results-index-db``,
``--config-file`` and the logging options belong to all four; ``--nav-results-root``
to the three that read a tree; ``--force`` and ``--no-prune`` to the two that
read documents and remove rows; and ``--yes`` to the drop, which is the only
thing this program asks about.

The roots are resolved the way every consumer resolves a navigation results
root -- the command line, then ``environment.nav_results_root``, then
``NAV_RESULTS_ROOT`` -- because the root is half of the key every row is stored
under.  Ingesting a subdirectory of a results root would produce stubs no
consumer's lookup can match.

Ingest is never automatic: no batch driver runs it as a side effect, and the
index it writes is a snapshot of the tree as of this run.

A pass removes the rows whose documents have left the tree, which is what makes
presence of a row mean that the tree still holds the result it stands for.
``--no-prune`` gives that up and keeps them: absence of a row goes on meaning
that the image was never navigated, since skipping a delete adds nothing, and
what is saved is the deletes and, where nothing else wants it, the query that
reads what the index already holds about the root.

``drop`` is the opposite operation and shares only the URL with the rest: it
removes the index's own tables from the database that URL names and stops
there, walking no tree.  It is what makes emptying an index and starting over
something an operator can reach without hand-written SQL, on a shared
PostgreSQL server as well as on a file -- which the schema version gate depends
on, since a version bump is deliberately not migrated and rebuilding is the
whole of the remedy.  It drops from one schema, the one this database's own
stamp table was found in, and only after finding that stamp: a table called
``images`` is not evidence of anything.

The two halves rest on one rule.  An ingest builds an index only in a schema
that holds nothing, or one already carrying a stamp of SpinDoctor's, and refuses
a schema holding a table it did not create -- so a stamp never comes to stand
over somebody else's table, and the six names the drop removes from a stamped
schema are six tables SpinDoctor created.
"""

import argparse
import os
import sys

import sqlalchemy
from filecache import FCPath

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.cli.results_index import (
    IngestCounts,
    TaskCompletion,
    UnwritableRowError,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_metadata_files,
    task_results_from_event_log,
)
from spindoctor.cli.results_index.drop import drop_results_index
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_run_logging,
    get_nav_results_root,
    get_results_index_db_url,
    get_results_tree_tuning,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_RESULTS_INDEX
from spindoctor.nav_records import TreeTuning, UnlistableDirectoryError, distinct_roots
from spindoctor.results_index import open_index
from spindoctor.support.command_line import masked_command_line
from spindoctor.support.file import json_as_string

PROGRAM_NAME = SD_RESULTS_INDEX
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""


INGEST = 'ingest'
"""Subcommand that reads the documents under each results root into the index."""

DIVIDE = 'divide'
"""Subcommand that divides each results root into shares for a queue of workers."""

COMPLETE = 'complete'
"""Subcommand that adds up what those workers did and finishes each root's run."""

DROP = 'drop'
"""Subcommand that removes the results index's own tables and stops."""

_READS_A_TREE = (INGEST, DIVIDE, COMPLETE)
"""The subcommands a navigation results root is resolved for.

The drop is about the database alone and names no root, so requiring one would
refuse the command on a machine holding the index and not the tree.
"""


def _add_environment_arguments(parser: argparse.ArgumentParser, *, reads_a_tree: bool) -> None:
    """Add the options a subcommand resolves its surroundings from.

    Parameters:
        parser: The subcommand's parser.
        reads_a_tree: Whether this subcommand walks a navigation results tree.
            False leaves ``--nav-results-root`` off, so a subcommand that reads
            no tree refuses the option naming one instead of accepting it and
            reading nothing.
    """
    group = parser.add_argument_group('Environment')
    group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    if reads_a_tree:
        group.add_argument(
            '--nav-results-root',
            action='append',
            dest='nav_results_roots',
            default=None,
            metavar='ROOT',
            help="""Root directory of a navigation results tree to read (a local
            directory or any URL the filecache layer accepts); may be specified
            multiple times. Overrides NAV_RESULTS_ROOT and the nav_results_root
            configuration variable.""",
        )
    group.add_argument(
        '--results-index-db',
        default=None,
        metavar='URL',
        help="""Connection URL of the results index (a sqlite: URL naming a
        local path, or a postgresql+psycopg: URL naming a server); overrides the
        environment.results_index_db configuration variable and
        NAV_RESULTS_INDEX_DB. An ingest creates the tables if they are absent, in the
        schema this database's own schema_meta stamp was found in or, where
        there is no such stamp, the one a table created without a schema name
        lands in. That schema is refused, and nothing is created or stamped in
        it, when it already holds any table the index does not own or any table
        of the index's own names that no stamp of ours stands over.""",
    )


def _add_reading_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options saying what a pass reads and what it removes.

    Both belong to the two subcommands that read documents and remove rows.  A
    completion does neither -- the fan-out that cut the shares held the one
    listing of the root and removed there, and each worker reads its share from
    the metrics that share carries -- and a drop reads no tree at all, so
    neither of those offers either option.

    Parameters:
        parser: The subcommand's parser.
    """
    group = parser.add_argument_group('Ingest')
    group.add_argument(
        '--force',
        action='store_true',
        default=False,
        help="""Read every document, including ones whose recorded size and
        modification time still match the tree.""",
    )
    group.add_argument(
        '--no-prune',
        dest='prune',
        action='store_false',
        default=True,
        help="""Leave the rows whose documents have left the tree in place
        rather than removing them. A row then stops meaning that the tree still
        holds the document it stands for, so a consumer asking whether an image
        has been navigated is answered yes for one whose result the tree no
        longer has. Absence of a row is untouched and still means the image was
        never navigated, which is what makes this safe to offer at all. What it
        saves is the deletes, and, where nothing else wants it, the query that
        reads what the index already holds about the root: under ingest with
        --force as well, since the skip rule is then not consulting it either,
        and under divide whether or not --force is given, since a fan-out reads
        it for the removals alone. It saves no part of the walk.""",
    )


def parse_args(command_list: list[str]) -> argparse.Namespace:
    """Build the parser and read the command line.

    Each mode is a subcommand carrying the options that mode acts on, so two
    modes cannot be asked for at once and an option belonging to another mode is
    a usage error naming it rather than a request the program has to reconcile.

    Parameters:
        command_list: Arguments, without the program name.

    Returns:
        The parsed arguments, with the subcommand under ``command``.
    """
    cmdparser = argparse.ArgumentParser(
        description='Read navigation metadata documents into the results index.'
    )
    subcommands = cmdparser.add_subparsers(
        title='Commands',
        dest='command',
        required=True,
        metavar='COMMAND',
    )

    ingest = subcommands.add_parser(
        INGEST,
        help='Read the documents under each results root into the index',
        description="""Walk each named navigation results root and read every
        metadata document under it into the results index, removing the rows
        whose documents have left the tree.""",
    )
    _add_environment_arguments(ingest, reads_a_tree=True)
    _add_reading_arguments(ingest)
    add_logging_arguments(ingest, has_image_logger=False)

    divide = subcommands.add_parser(
        DIVIDE,
        help='Divide each results root into shares for a queue of workers',
        description="""List each named navigation results root once, remove the
        rows whose documents have left it, and write out the shares
        sd_results_index_cloud_tasks reads. No document is read here, and each
        root stays unfinished until sd_results_index complete adds up what the
        workers did.""",
    )
    _add_environment_arguments(divide, reads_a_tree=True)
    _add_reading_arguments(divide)
    divide_group = divide.add_argument_group('Cloud tasks')
    divide_group.add_argument(
        '--tasks-file',
        required=True,
        metavar='PATH',
        help="""Where to write the JSON task descriptions file, in the shape a
        cloud_tasks queue loads and sd_results_index_cloud_tasks reads.""",
    )
    add_logging_arguments(divide, has_image_logger=False)

    complete = subcommands.add_parser(
        COMPLETE,
        help="Add up what the workers did and finish each root's ingest run",
        description="""Read the cloud_tasks event log the workers wrote, add up
        what their tasks did, and record it against each named root's ingest
        run. A root whose tasks do not account for exactly the files its listing
        found is left unfinished and named. No document is read and no row is
        removed here, so --force and --no-prune belong to the sd_results_index
        divide that cut the shares and are not offered.""",
    )
    _add_environment_arguments(complete, reads_a_tree=True)
    complete_group = complete.add_argument_group('Cloud tasks')
    complete_group.add_argument(
        '--events-log',
        required=True,
        metavar='PATH',
        help="""The cloud_tasks event log the workers wrote, whose
        task_completed events carry what each share ingested, skipped and could
        not read.""",
    )
    add_logging_arguments(complete, has_image_logger=False)

    drop = subcommands.add_parser(
        DROP,
        help="Remove the results index's tables from the database and stop",
        description="""Remove the results index's own tables from the database
        --results-index-db names, and stop: no results root is read and no
        document is ingested. The tables that go are the index's own six names,
        from the one schema this database's own schema_meta stamp was found in;
        no other table of that schema, and no other schema, is touched. What
        makes those six SpinDoctor's own is that an ingest refuses to build an
        index in a schema holding anything it did not create, so a schema
        carrying that stamp holds this index and nothing else. A database
        holding none of those tables is left alone and said to be, and one
        holding tables of those names that no stamp of ours stands over is
        refused. It reads no tree and ingests nothing, so it offers none of the
        options that describe an ingest.""",
    )
    _add_environment_arguments(drop, reads_a_tree=False)
    drop_group = drop.add_argument_group('Drop')
    drop_group.add_argument(
        '--yes',
        action='store_true',
        default=False,
        help="""Drop without asking for confirmation, for a run with nobody at
        the terminal.""",
    )
    add_logging_arguments(drop, has_image_logger=False)

    return cmdparser.parse_args(command_list)


def _log_removals(files_removed: int, *, pruned: bool | None) -> None:
    """Say what became of the rows whose documents have left the tree.

    A pass that removed them reports how many, which is a number an operator
    would otherwise go looking for.  A pass that was told to leave them says so
    instead of reporting a removal of none, because the log is the only place
    that records which guarantee the index was built under and a zero reads
    exactly like a tree nothing has left.

    A count added up from somewhere else is reported as one, without saying
    which of the two produced it.  Adding up what the workers did reads the
    number off the run row the fan-out wrote it to, and nothing recorded there
    says whether that fan-out was removing rows at all, so a zero there means
    either that nothing had left the tree or that nobody looked.

    Parameters:
        files_removed: How many image rows were deleted.
        pruned: Whether the pass reporting this was removing them, or None where
            the count comes from a pass other than this one and which it was is
            not recorded.
    """
    if pruned is None:
        MAIN_LOGGER.info('Rows removed before the fan-out: %d', files_removed)
        return
    if pruned:
        MAIN_LOGGER.info('Rows removed, their document gone from the tree: %d', files_removed)
        return
    MAIN_LOGGER.info(
        'Rows whose document has left the tree: left in place, since --no-prune was given. '
        'A row under these roots no longer means the tree still holds the document it '
        'stands for. Absence of a row is unchanged and still means the image was never '
        'navigated.'
    )


def _log_outcome(counts: IngestCounts, *, pruned: bool | None) -> None:
    """Write the closing summary of an ingest pass to the main log.

    The failures are tallied by reason as well as counted, because a results
    tree holds many ``*_metadata.json`` files that were never navigation
    documents.  Several hundred of those are ordinary; several hundred
    navigation results that would not parse are not, and the tally is what
    tells the two apart at a glance.  Each reason names one file that carried
    it, because a reason is a field-level diagnosis and one look at a real file
    is what turns it into a judgement about the tree.

    Parameters:
        counts: What the pass did.
        pruned: Whether it removed the rows of documents that have left the
            tree, or None where the count was added up from a pass other than
            this one, which does not record which it was doing.
    """
    MAIN_LOGGER.info('Metadata files seen: %d', counts.files_seen)
    MAIN_LOGGER.info('Ingested: %d', counts.files_ingested)
    MAIN_LOGGER.info('Skipped as unchanged: %d', counts.files_skipped)
    _log_removals(counts.files_removed, pruned=pruned)
    MAIN_LOGGER.info('Not ingestible: %d', counts.files_failed)
    for reason in sorted(counts.failures_by_reason):
        MAIN_LOGGER.info(
            '    %s: %d file(s), for example %s',
            reason,
            counts.failures_by_reason[reason],
            counts.example_by_reason.get(reason, '(none recorded)'),
        )
    if counts.roots_unreadable:
        MAIN_LOGGER.error(
            'Roots that could not be listed and are therefore not ingested: %d',
            counts.roots_unreadable,
        )


def _log_completion(completion: TaskCompletion) -> None:
    """Write the closing summary of a cloud-task completion to the main log.

    Parameters:
        completion: What adding the shares up did.
    """
    _log_outcome(completion.counts, pruned=None)
    MAIN_LOGGER.info('Ingest runs completed: %d', completion.runs_completed)
    for root in completion.roots_unaccounted:
        MAIN_LOGGER.error(
            'Left unfinished, because its tasks did not account for exactly the files its '
            'listing found: %s',
            root,
        )
    for root in completion.roots_unlisted:
        MAIN_LOGGER.error(
            'Left unfinished, because its ingest run never recorded what its listing found, '
            'so nothing says what this root holds: %s. Divide the root up with '
            'sd_results_index divide, run its tasks, and complete that run.',
            root,
        )
    for root in completion.roots_without_a_run:
        MAIN_LOGGER.error(
            'No unfinished ingest run to complete for %s: run sd_results_index divide over '
            'that root first, and complete the run that fanned out',
            root,
        )
    if completion.results_failed:
        MAIN_LOGGER.error(
            'Tasks that reported an error instead of a share: %d', completion.results_failed
        )
    if completion.results_unreadable:
        MAIN_LOGGER.error(
            'Task results that are not the shape a share reports: %d',
            completion.results_unreadable,
        )
    if completion.results_unclaimed:
        MAIN_LOGGER.warning(
            'Task results naming an ingest run none of these roots is waiting on: %d. They '
            'belong to another fan-out, and are left for whoever completes it.',
            completion.results_unclaimed,
        )
    if completion.results_of_another_root:
        MAIN_LOGGER.error(
            'Task results naming a run being completed here but reporting rows under a '
            "different root: %d. They are not that run's shares and are not counted toward "
            'it: a run number is only unique within the index that minted it, so a task file '
            'that outlived its index names a run of whatever was built next.',
            completion.results_of_another_root,
        )
    if completion.results_superseded:
        MAIN_LOGGER.info(
            'Task results superseded by a later report of the same task: %d. A queue '
            'delivers a task again whenever it could not see the last delivery '
            'acknowledged, and one share reported twice is still one share.',
            completion.results_superseded,
        )
    if completion.results_unidentified:
        MAIN_LOGGER.error(
            'Task results naming no task: %d. One of them cannot be told from a repeat of '
            'another, so none of them is counted toward a run.',
            completion.results_unidentified,
        )


def _run_ingest(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool,
    prune: bool,
    tuning: TreeTuning,
) -> int:
    """Read every document under each root and write its rows.

    Parameters:
        engine: The open index.
        roots: The navigation results roots to walk.
        force: Whether to re-read every document.
        prune: Whether to remove the rows of documents that have left the tree.
        tuning: How much of the pass runs at once.

    Returns:
        The exit status: 0 when every named root was walked, 1 when one could
        not be listed.
    """
    counts = ingest_metadata_files(
        engine, roots, force=force, prune=prune, logger=MAIN_LOGGER, tuning=tuning
    )
    _log_outcome(counts, pruned=prune)
    # Whether the run completed, not what it found.  A count of documents flips
    # between two passes over one unchanged tree -- what one pass ingests the
    # next one skips, and what one pass refuses the next one skips too -- so a
    # status read from a count tells a scheduled run that a tree it has already
    # accounted for has gone wrong.  A root that could not be listed is the
    # failure: nothing under it was walked, and every later root of the same
    # pass is still walked, so the status is the only place it shows.
    return 1 if counts.roots_unreadable else 0


def _write_cloud_tasks(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool,
    prune: bool,
    path: str,
    tuning: TreeTuning,
) -> int:
    """List each root once and write out the shares its documents divide into.

    Parameters:
        engine: The open index.
        roots: The navigation results roots to walk.
        force: Whether the workers should re-read every document.
        prune: Whether to remove the rows of documents that have left the tree.
        path: Where to write the task descriptions.
        tuning: How much of each listing runs at once.

    Returns:
        The exit status: 0 when every named root was listed, 1 when one could
        not be.
    """
    MAIN_LOGGER.info('Writing cloud_tasks file to %s', path)
    fan_out = fan_out_ingest_tasks(
        engine, roots, force=force, prune=prune, logger=MAIN_LOGGER, tuning=tuning
    )
    with FCPath(path).open('w') as file:
        file.write(json_as_string(fan_out.tasks))
    MAIN_LOGGER.info('Wrote %d task(s) to %s', len(fan_out.tasks), path)
    MAIN_LOGGER.info('Metadata files seen: %d', fan_out.counts.files_seen)
    _log_removals(fan_out.counts.files_removed, pruned=prune)
    if fan_out.counts.roots_unreadable:
        MAIN_LOGGER.error(
            'Roots that could not be listed and are therefore not ingested: %d',
            fan_out.counts.roots_unreadable,
        )
    MAIN_LOGGER.info(
        'Each root stays unfinished until the workers have run and '
        'sd_results_index complete has added up what they did'
    )
    return 1 if fan_out.counts.roots_unreadable else 0


def _complete_cloud_tasks(engine: sqlalchemy.Engine, roots: list[str], *, path: str) -> int:
    """Add up what the workers did and stamp the runs they completed.

    Parameters:
        engine: The open index.
        roots: The navigation results roots whose runs are being completed.
        path: The cloud_tasks event log the workers wrote.

    Returns:
        The exit status: 0 when every named root's ingest run was completed, 1
        when one was not, and 1 when the event log could not be read.
    """
    MAIN_LOGGER.info('Reading task results from %s', path)
    try:
        found = task_results_from_event_log(FCPath(path))
    except (OSError, UnicodeDecodeError) as exc:
        # An ordinary mistyped path, which the pass enumerates and charges to
        # the file rather than letting out as a traceback.  A path naming a file
        # that is not text -- a gzipped log, a database, an image -- is the same
        # error and is charged the same way; it raises a UnicodeDecodeError,
        # which is a ValueError rather than an OSError.  Every named root keeps
        # its unfinished run, so no consumer reads absence under one of them as
        # an answer.
        MAIN_LOGGER.fatal('Cannot read the task event log %s: %s', path, exc)
        return 1
    MAIN_LOGGER.info('Task results read: %d', len(found.results))
    if found.lines_unread:
        MAIN_LOGGER.warning(
            'Lines of %s that are not events: %d. A log being appended to while it is read '
            'ends in a partial line; many of them say this is not an event log.',
            path,
            found.lines_unread,
        )
    if found.tasks_unfinished:
        MAIN_LOGGER.error(
            'Tasks that ended without returning a share: %d. Their documents were never '
            'read, so the roots they belong to cannot be completed.',
            found.tasks_unfinished,
        )
    completion = complete_ingest_tasks(engine, roots, found.results, logger=MAIN_LOGGER)
    _log_completion(completion)
    unfinished = (
        completion.roots_unaccounted + completion.roots_unlisted + completion.roots_without_a_run
    )
    return 1 if unfinished else 0


def main() -> None:
    """Console entry point for ``sd_results_index``.

    Resolves the index URL and, for every subcommand but the drop, the results
    roots as well; then does what the subcommand names -- removes the index's
    tables, reads every document under those roots, divides them into cloud
    tasks, or adds up what those tasks did.  The outcome goes to the main log
    whichever subcommand ran.

    Raises:
        SystemExit: Always, since this is a console entry point.  The status
            says whether the pass completed, not what it found: 0 when every
            named root was walked, whatever mix of documents was read, skipped
            and refused, and 1 when the run could not complete -- no index or no
            root could be resolved, a named root is not a location that can be
            read, the index could not be opened, a root could not be listed, or
            a directory under one could not be, which stops the pass where it
            is found and leaves every root from there on unfinished.
            A tree of files that are not navigation documents is a completed
            pass and exits 0, and exits 0 again on the next pass over the same
            tree, so a scheduled run's status means the same thing every time it
            is read.  Completing a fan-out exits 1 when the event log cannot be
            read, when a named root has no unfinished run, when its run never
            recorded what its listing found, or when its tasks did not account
            for exactly the files that listing found.  A drop exits 0 when the
            tables went and 0 again when the database held none of them, and 1
            when the database could not be opened or read, when it holds tables
            of those names that nothing proves are the index's, when a table
            would not drop, or when whoever was asked said anything but yes.
            A command line the parser will not take exits 2 with a usage error
            on standard error and does nothing at all: no subcommand, an unknown
            one, a subcommand missing the path it acts on, or an option
            belonging to one of the other three.
    """
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)
    if arguments.command in _READS_A_TREE:
        # One pass may cover several roots, which is what makes ingest different
        # from every other program; the shared root resolver reads one
        # attribute, so the first named root is the one this run is logged under
        # and the rest are further trees to walk.  With none named, the resolver
        # falls through to the configuration variable and the environment as it
        # does everywhere else, which is also how it answers for the drop, whose
        # command line has no root on it to read.
        arguments.nav_results_root = (arguments.nav_results_roots or [None])[0]

    # Read configuration files
    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)
    with reporting_configuration_errors():
        build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    # A level that names the index with an empty value is a different failure from
    # naming none at all, and its message names that level, so it is reported as
    # itself rather than as the "no index was named" refusal below.
    try:
        url = get_results_index_db_url(arguments, DEFAULT_CONFIG)
    except ValueError as exc:
        MAIN_LOGGER.fatal('%s', exc)
        sys.exit(1)
    if url is None:
        MAIN_LOGGER.fatal(
            'No results index was named. Give one with --results-index-db, the '
            'environment.results_index_db configuration variable, or NAV_RESULTS_INDEX_DB.'
        )
        sys.exit(1)

    if arguments.command == DROP:
        # Before the roots are resolved, and instead of them: a drop is about
        # the database alone, and requiring a results root for it would refuse
        # the command on a machine that has the index and not the tree.
        MAIN_LOGGER.info('Starting results index drop')
        MAIN_LOGGER.info('Arguments: %s', masked_command_line(command_list))
        sys.exit(drop_results_index(url, assume_yes=arguments.yes, logger=MAIN_LOGGER))

    try:
        named = arguments.nav_results_roots or [get_nav_results_root(arguments, DEFAULT_CONFIG)]
    except ValueError as exc:
        MAIN_LOGGER.fatal('No navigation results root was named: %s', exc)
        sys.exit(1)

    # Normalized and de-duplicated here rather than in each mode, so that the
    # roots this run reports on are the roots it works over: every later message
    # names the normalized spelling, and a command line naming one root two ways
    # would otherwise open by listing two and then account for one, which reads
    # as a root having gone missing.
    try:
        roots = distinct_roots(named)
    except ValueError as exc:
        MAIN_LOGGER.fatal('A navigation results root is not a location that can be read: %s', exc)
        sys.exit(1)

    # Completing a fan-out reads runs the fan-out already recorded, so an index
    # that is not there is a wrong URL rather than a first run: creating an
    # empty one would answer "no run to complete" for a root whose run is
    # sitting in the index the operator meant to name.
    completing = arguments.command == COMPLETE

    # The subcommand, not a fixed word: a run's log opens by saying which of the
    # four it is, and a divide or a completion reported as an ingest is a log
    # that names a pass nobody asked for.
    MAIN_LOGGER.info('Starting results index %s', arguments.command)
    MAIN_LOGGER.info('Roots: %s', ', '.join(roots))
    if not completing:
        MAIN_LOGGER.info('Force: %s', arguments.force)
    MAIN_LOGGER.info('Arguments: %s', masked_command_line(command_list))

    # Resolved once here and passed to whatever reads the tree.  The
    # configuration was validated when it was loaded, so this cannot refuse.
    tuning = get_results_tree_tuning(DEFAULT_CONFIG)

    try:
        engine = open_index(url, create=not completing)
    except ValueError as exc:
        MAIN_LOGGER.fatal('Cannot open the results index: %s', exc)
        sys.exit(1)
    try:
        if arguments.command == DIVIDE:
            status = _write_cloud_tasks(
                engine,
                roots,
                force=arguments.force,
                prune=arguments.prune,
                path=arguments.tasks_file,
                tuning=tuning,
            )
        elif completing:
            status = _complete_cloud_tasks(engine, roots, path=arguments.events_log)
        else:
            status = _run_ingest(
                engine, roots, force=arguments.force, prune=arguments.prune, tuning=tuning
            )
    except UnwritableRowError as exc:
        # The other failure a pass stops for.  The document read exactly as the
        # schema says and the writer or the column set would not take it, so
        # every document of that shape after it fails the same way; charged to
        # the file it would be left out of both tables and the run stamped
        # finished, after which absence reads as "never navigated".
        MAIN_LOGGER.fatal(
            'Ingest stopped: %s. The document is a navigation result and the index would not '
            'store it, which is a defect in this program rather than in the file. This root '
            'and any named after it have no completed ingest run, so no consumer reads '
            'absence under them as an answer.',
            exc,
        )
        status = 1
    except UnlistableDirectoryError as exc:
        # The one failure a pass stops for rather than charging to a file or a
        # root.  A directory nobody listed holds documents nobody recorded, and
        # absence of a row is what every consumer reads as "this image was
        # never navigated", so the alternative to stopping is a completed pass
        # that answers wrongly and goes on answering wrongly.
        MAIN_LOGGER.fatal(
            'Ingest stopped: %s. This root and any named after it have no completed ingest '
            'run, so no consumer reads absence under them as an answer. Make that directory '
            'readable and run the ingest again.',
            exc,
        )
        status = 1
    except Exception as exc:
        # The pass enumerates every failure it expects and charges it to one
        # file or one root.  Anything still escaping is a failure nobody
        # enumerated, and a console entry point owes its caller a message and a
        # status for one rather than a traceback: the run rows of the roots it
        # did not reach keep their NULL finish times, so no consumer reads
        # absence under them as an answer.
        MAIN_LOGGER.fatal('Ingest could not complete (%s: %s)', type(exc).__name__, exc)
        MAIN_LOGGER.exception(exc)
        sys.exit(1)
    finally:
        engine.dispose()
    sys.exit(status)


if __name__ == '__main__':
    main()
