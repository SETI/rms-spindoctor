#!/usr/bin/env python3
"""Ingest one share of a navigation results root, driven by a cloud-tasks queue.

Dispatch script for the ``sd_results_index_cloud_tasks`` console entry point.
Its interactive sibling ``sd_results_index divide`` divides a root into shares
and ``sd_results_index complete`` adds up what the workers did; this is what
reads one of those shares.

A worker is handed the files of its share, not a root to walk: the listing
happens once, where the work is divided up.  It therefore also removes no row.
Deleting the rows of documents that have left the tree is licensed by a complete
listing of the root, and a share is evidence about nothing outside itself, so
removing on it would delete another worker's rows.

A worker has no run log and no per-image scope, so what it did is returned in
the task result -- how many files it ingested, skipped and could not read, and
the name of every file it could not read.  The index it writes to must already
exist: only the program that divides the work up creates the schema, because a
worker that created one would answer a wrong URL by building an empty index
beside the real one.
"""

import argparse
import asyncio
import os
import sys
from typing import Any, cast

from cloud_tasks.worker import Worker, WorkerData

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.results_index import ingest_task_share
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_cloud_task_logging,
    get_results_index_db_url,
    get_results_tree_tuning,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_RESULTS_INDEX
from spindoctor.results_index import open_index

PROGRAM_NAME = SD_RESULTS_INDEX
"""Program identity: names the ``logging.programs`` configuration block this
worker shares with its interactive sibling."""


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Ingest one share of a results root into the index.

    Parameters:
        task_id: The ID of the task.
        task_data: The data for the task, carrying the ``run_id`` and
            ``root_url`` the share belongs to and the ``files`` to read.
        worker_data: The data for the worker.

    Returns:
        Tuple of ``(retry, result)``.  ``retry`` is always False.  ``result``
        names the error when the task could not run, and otherwise reports how
        many files were ingested, skipped and refused, with every refused file
        named.  It is the only channel a worker has: there is no run log for the
        program that completes the ingest to read this out of.
    """

    arguments = cast(argparse.Namespace, worker_data.args)
    try:
        load_default_and_user_config(arguments, DEFAULT_CONFIG)
        # Resolved once per share and passed down, as the interactive driver does.
        tuning = get_results_tree_tuning(DEFAULT_CONFIG)
    except (TypeError, ValueError) as exc:
        # A configuration no pass can run under is reported the way every other
        # refusal here is, so the program that adds the shares up can tally it
        # rather than reading a worker that raised as one that never ran.
        return False, {
            'status': 'error',
            'status_error': 'unusable_configuration',
            'status_exception': str(exc),
        }

    # Resolved the same way the interactive driver resolves it, which is also
    # what withholds the worker's terminal from everything logged below.  There
    # is no main log and no per-image log here: a share's outcome is its return
    # value.
    build_cloud_task_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    try:
        url = get_results_index_db_url(arguments, DEFAULT_CONFIG)
    except ValueError as exc:
        # A level that names the index with an empty value fails the share
        # naming that level, which is a different thing from naming no index at
        # all and is reported as its own status so a tally can tell them apart.
        return False, {
            'status': 'error',
            'status_error': 'unusable_results_index_db',
            'status_exception': str(exc),
        }
    if url is None:
        return False, {'status': 'error', 'status_error': 'no_results_index_db'}

    try:
        # create=False: the program that divided the work up made the schema.
        engine = open_index(url)
    except ValueError as exc:
        # The message names the URL with any password in it already masked, and
        # a task result is written to an event log like any other.
        return False, {
            'status': 'error',
            'status_error': 'index_unopenable',
            'status_exception': str(exc),
        }
    try:
        result = ingest_task_share(engine, task_data, logger=MAIN_LOGGER, tuning=tuning)
    except ValueError as exc:
        return False, {
            'status': 'error',
            'status_error': 'malformed_task',
            'status_exception': str(exc),
        }
    finally:
        engine.dispose()

    return False, result  # No retry under any circumstances


async def async_main() -> None:
    """Build the worker's argument parser and run it against the queue.

    The parser carries only what a share needs resolving: the configuration
    files and the index URL.  It carries no logging arguments, because a cloud
    task's logging is resolved for it rather than asked for on a command line.
    """
    argparser = argparse.ArgumentParser(description='Results index ingest (Cloud Tasks version)')

    environment_group = argparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--results-index-db',
        default=None,
        metavar='URL',
        help="""Connection URL of the results index to write (a sqlite: URL
        naming a local path, or a postgresql+psycopg: URL naming a server);
        overrides the environment.results_index_db configuration variable and
        NAV_RESULTS_INDEX_DB. The index must already exist.""",
    )

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    """Synchronous entry point; runs ``asyncio.run(async_main())``."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
