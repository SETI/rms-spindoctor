"""Helpers shared by the two modules that run the ingest command lines.

The interactive driver and the worker are two halves of one pass, and each
half's tests reach for the other's: a worker needs a root somebody fanned out,
and the mode that adds the shares up needs shares somebody ran.  What runs
either program lives here rather than in one of the two test modules, so that
neither imports the other.

Nothing here is a fixture, so importing it costs a test nothing it did not ask
for.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from cloud_tasks.worker import WorkerData
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    write_metadata,
)

from spindoctor.cli import sd_results_index, sd_results_index_cloud_tasks
from spindoctor.config import MAIN_LOGGER

STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the document every tree below holds."""


class _StubWorkerData:
    """Stands in for the cloud_tasks worker's data object."""

    def __init__(self, **kwargs: object) -> None:
        """Build worker data carrying only the given CLI arguments.

        Parameters:
            **kwargs: Argument names and values for the parsed namespace.
        """
        given: dict[str, object] = {'config_file': None, 'log_root': None}
        given.update(kwargs)
        self.args = argparse.Namespace(**given)


def worker_data(**kwargs: object) -> WorkerData:
    """Build the worker data a driver reads its CLI arguments from.

    Parameters:
        **kwargs: Argument names and values for the parsed namespace.

    Returns:
        The stub, typed as the worker data a driver expects.  A driver reads
        only ``args`` from it, so building the real thing would mean standing up
        a worker for no benefit.
    """
    return cast(WorkerData, _StubWorkerData(**kwargs))


def run_driver(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[int | None, list[str]]:
    """Run ``sd_results_index`` and return its exit status and its main log.

    Parameters:
        argv: Arguments, without the program name, beginning with the
            subcommand.
        monkeypatch: Fixture the argument vector and logger are replaced through.
        tmp_path: Directory the run's log files are written under.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    written: list[str] = []

    def recording(message: Any, *args: Any) -> None:
        written.append(str(message) % args if args else str(message))

    # The log root goes last, because every subcommand declares it and none of
    # them is reached until the subcommand itself has been read.
    monkeypatch.setattr(
        sys, 'argv', ['sd_results_index', *argv, '--log-root', str(tmp_path / 'logs')]
    )
    for level in ('info', 'warning', 'error', 'fatal', 'exception'):
        monkeypatch.setattr(MAIN_LOGGER, level, recording)
    with pytest.raises(SystemExit) as caught:
        sd_results_index.main()
    status = caught.value.code
    return (status if status is None or isinstance(status, int) else 1), written


def tasks_of(path: Path) -> list[dict[str, Any]]:
    """Read a written cloud-tasks file.

    Parameters:
        path: The file the driver wrote.

    Returns:
        The task descriptions.
    """
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding='utf-8')))


def process(task_data: dict[str, Any], url: str) -> tuple[bool, Any]:
    """Run one ingest task through the worker driver.

    Parameters:
        task_data: The task's data.
        url: The index URL to hand the worker.

    Returns:
        What ``process_task`` returned.
    """
    return sd_results_index_cloud_tasks.process_task(
        'ingest-1-000000', task_data, worker_data(results_index_db=url)
    )


def fanned_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 1) -> str:
    """Write a tree, fan it out through the driver, and return the index URL.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.
        count: How many documents to write.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    for index in range(count):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            'divide',
            '--results-index-db',
            url,
            '--nav-results-root',
            root.as_posix(),
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    return url


def fanned_out_with_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a tree holding one document and one file that is not one, and fan it out.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            'divide',
            '--results-index-db',
            url,
            '--nav-results-root',
            root.as_posix(),
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    return url
