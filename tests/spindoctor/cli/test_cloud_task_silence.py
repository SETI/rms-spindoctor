"""Tests that a cloud task emits nothing on a real terminal.

The rest of the logging tests capture with ``capsys``, which replaces
``sys.stdout`` and ``sys.stderr`` inside the test process.  That is enough for
records written through Python, but the guarantee here is about the file
descriptors a worker's terminal actually is, and the handler that
``cloud_tasks`` installs on the root logger binds a stream at construction.
So this one runs a real subprocess, with ``logging.basicConfig`` installed
exactly as a worker installs it, and measures the bytes on each descriptor.
"""

import json
import subprocess
import sys
from typing import Any

import pytest
from filecache import FCPath
from tests.conftest import child_interpreter_environment

# Every level, so nothing is passing merely for being below a threshold, and
# an exception, whose traceback is the bulkiest thing a per-image log carries.
_CHILD = """
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

from filecache import FCPath
from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER
from spindoctor.config.config import Config
from spindoctor.config.logging_config import (
    build_cloud_task_logging,
    build_image_log_handlers,
)
import argparse

config = Config()
config.read_config()
log_root = FCPath(sys.argv[1])

run_logging = build_cloud_task_logging(
    'sd_offset',
    argparse.Namespace(log_root=log_root.as_posix(), log_level=['DEBUG']),
    config,
)
handlers, path = build_image_log_handlers(
    'nav', 'vol/N1', run_logging.sinks, run_logging.levels, timestamp='STAMP'
)
try:
    with IMAGE_LOGGER.open('IMAGE', handler=handlers):
        IMAGE_LOGGER.debug('CANARY-DEBUG')
        IMAGE_LOGGER.info('CANARY-INFO')
        IMAGE_LOGGER.warning('CANARY-WARNING')
        IMAGE_LOGGER.error('CANARY-ERROR')
        IMAGE_LOGGER.critical('CANARY-CRITICAL')
        try:
            raise RuntimeError('CANARY-EXCEPTION')
        except RuntimeError:
            IMAGE_LOGGER.exception('CANARY-EXCEPTION')
    MAIN_LOGGER.info('CANARY-MAIN')
    IMAGE_LOGGER.warning('CANARY-OUT-OF-SCOPE')
finally:
    for handler in handlers:
        handler.close()

FCPath(sys.argv[2]).write_text(path.as_posix())
"""


@pytest.fixture(scope='module')
def task_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str, str]:
    """Run one cloud task in a worker-like subprocess.

    Module-scoped: the subprocess costs a full interpreter start and every
    assertion below reads the same run.

    Parameters:
        tmp_path_factory: Fixture used to make the run's directory.

    Returns:
        Tuple of the child's stdout, its stderr, and its image log text.
    """
    directory = FCPath(tmp_path_factory.mktemp('cloud_task_silence'))
    script = directory / 'child.py'
    script.write_text(_CHILD)
    path_file = directory / 'log_path.txt'

    completed = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            (directory / 'logs').as_posix(),
            path_file.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env=child_interpreter_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    log_text = FCPath(path_file.read_text()).read_text()
    return completed.stdout, completed.stderr, str(log_text)


def test_a_cloud_task_writes_nothing_to_stdout(task_output: tuple[str, str, str]) -> None:
    """Not one byte reaches the descriptor the worker reports progress on."""
    assert task_output[0] == ''


def test_a_cloud_task_writes_nothing_to_stderr(task_output: tuple[str, str, str]) -> None:
    """Nor to stderr, where the root handler would otherwise re-emit it all."""
    assert task_output[1] == ''


@pytest.mark.parametrize(
    'canary',
    [
        'CANARY-DEBUG',
        'CANARY-INFO',
        'CANARY-WARNING',
        'CANARY-ERROR',
        'CANARY-CRITICAL',
        'CANARY-EXCEPTION',
    ],
)
def test_the_image_log_is_complete(task_output: tuple[str, str, str], canary: str) -> None:
    """Silence on the terminal is not silence everywhere.

    Every level reaches the per-image file, so what the terminal is spared is
    duplication rather than the record itself.
    """
    assert canary in task_output[2]


def test_the_exception_keeps_its_type(task_output: tuple[str, str, str]) -> None:
    """The exception survives, since the image log is where it now lives."""
    assert 'RuntimeError' in task_output[2]


def test_the_exception_keeps_its_traceback(task_output: tuple[str, str, str]) -> None:
    """And so does the traceback, which is the part worth having.

    Naming the exception type alone would be satisfied by a one-line record.
    The frame is what tells a reader where the image failed, and it is the
    bulkiest thing a per-image log carries -- so if isolation were going to
    lose anything to a size or formatting difference, it would lose this.
    """
    assert f'in <module>{chr(10)}' in task_output[2]


def test_the_traceback_names_the_failing_line(task_output: tuple[str, str, str]) -> None:
    """The frame carries the source line, not just the file it came from."""
    assert "raise RuntimeError('CANARY-EXCEPTION')" in task_output[2]


# ---------------------------------------------------------------------------
# The ingest worker, which has no per-image log to write into either
# ---------------------------------------------------------------------------

# An ingest worker logs where a navigation worker logs to an image file: its
# per-file notes have no per-image scope to go in, and its own tally comes back
# in the return value.  So the whole of what it says has to land nowhere, which
# is a stronger claim than the one above and is measured the same way -- on the
# descriptors, in a subprocess, with the root handler cloud_tasks installs.
# The share ingests one document, refuses one, and skips one, so every level the
# pass writes at is exercised.
_INGEST_CHILD = """
import json
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

from filecache import FCPath

from spindoctor.cli import sd_results_index_cloud_tasks
from spindoctor.cli.results_index import fan_out_ingest_tasks
from spindoctor.results_index import open_index
from spindoctor.nav_records import TreeTuning
import argparse

directory = FCPath(sys.argv[1])
root = directory / 'results'
document = {
    'status': 'success',
    'offset': [1.5, -2.5],
    'confidence': 0.8,
    'observation': {'image_name': 'N1_CALIB.IMG', 'instrument': 'coiss', 'camera': 'NAC'},
    'navigation_result': {'status': 'success', 'per_technique': [], 'feature_inventory': []},
}
with (root / 'VOL' / 'N1_CALIB_metadata.json').open('w') as file:
    file.write(json.dumps(document))
with (root / 'VOL' / 'other_metadata.json').open('w') as file:
    file.write('{"edges": []}')

url = 'sqlite:///' + (directory / 'index.sqlite3').as_posix()
engine = open_index(url, create=True)


class _Quiet:
    def info(self, *args):
        pass

    warning = error = debug = exception = info


tasks = fan_out_ingest_tasks(engine, [root.as_posix()], logger=_Quiet(), tuning=TreeTuning()).tasks
engine.dispose()


class _WorkerData:
    def __init__(self):
        self.args = argparse.Namespace(config_file=None, results_index_db=url)


results = [
    sd_results_index_cloud_tasks.process_task(task['task_id'], task['data'], _WorkerData())[1]
    for task in tasks
]
# The same share again, so the skip path logs too.
results += [
    sd_results_index_cloud_tasks.process_task(task['task_id'], task['data'], _WorkerData())[1]
    for task in tasks
]
with FCPath(sys.argv[2]).open('w') as file:
    file.write(json.dumps(results))
"""


@pytest.fixture(scope='module')
def ingest_task_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str, list[Any]]:
    """Run one ingest task in a worker-like subprocess.

    Module-scoped for the same reason as the navigation one: the subprocess
    costs a full interpreter start and every assertion below reads one run.

    Parameters:
        tmp_path_factory: Fixture used to make the run's directory.

    Returns:
        Tuple of the child's stdout, its stderr, and the task results it wrote.
    """
    directory = FCPath(tmp_path_factory.mktemp('ingest_task_silence'))
    script = directory / 'child.py'
    script.write_text(_INGEST_CHILD)
    results_file = directory / 'results.json'

    completed = subprocess.run(
        [
            sys.executable,
            script.as_posix(),
            (directory / 'run').as_posix(),
            results_file.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        env=child_interpreter_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout, completed.stderr, json.loads(results_file.read_text())


def test_an_ingest_task_writes_nothing_to_stdout(
    ingest_task_output: tuple[str, str, list[Any]],
) -> None:
    """Not one byte, though the pass logs a line per file it will not read.

    This is the half of the isolation that binds the null sink: a pdslogger left
    with no handlers at all does not go quiet, it prints every record to stdout
    whatever its level.
    """
    assert ingest_task_output[0] == ''


def test_an_ingest_task_writes_nothing_to_stderr(
    ingest_task_output: tuple[str, str, list[Any]],
) -> None:
    """Nor to stderr, where the root handler would otherwise re-emit it all.

    This is the other half, and it fails to a different break: a record that
    still propagated would be emitted a second time by the handler
    ``logging.basicConfig`` puts on the root logger, which the child installs
    above exactly as a worker does -- and that handler writes to stderr rather
    than stdout, so binding the null sink alone leaves this descriptor open.
    """
    assert ingest_task_output[1] == ''


def test_an_ingest_task_still_reports_what_it_ingested(
    ingest_task_output: tuple[str, str, list[Any]],
) -> None:
    """Silence on the terminal must not be silence about the work.

    An ingest worker has no log file to fall back on, so its return value is
    the whole record: if isolation cost the tally, nothing would carry it.
    """
    assert ingest_task_output[2][0]['files_ingested'] == 1


def test_an_ingest_task_still_names_what_it_refused(
    ingest_task_output: tuple[str, str, list[Any]],
) -> None:
    """The file that is not a navigation document is named, not merely counted."""
    named = [FCPath(name).name for name in ingest_task_output[2][0]['failed_files']]
    assert named == ['other_metadata.json']


def test_an_ingest_task_still_reports_what_it_skipped(
    ingest_task_output: tuple[str, str, list[Any]],
) -> None:
    """The second run of the same share reads nothing and says so."""
    assert ingest_task_output[2][1]['files_skipped'] == 2
