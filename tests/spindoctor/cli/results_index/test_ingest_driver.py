"""Tests for the ``sd_results_index`` command line.

Three things the driver does are worth pinning on their own. It reports a
configuration failure -- no index named, no results root resolvable -- through
the log an operator is already reading, rather than as a traceback. Of what it
writes to that log, the index URL is masked and nothing else is: an index URL
can carry a database password and a run log is read by whoever is handed one,
while a results root carries no credentials and is the one word of the command
line the reader is there to correct. And its exit status says whether the run
completed, not what the run found, so a scheduled invocation reads the same
status from the same tree every time.

What the masking rule itself does with each spelling of an option, each shape of
a password, and each word that only looks like one is pinned where the rule
lives, in ``tests/spindoctor/support/test_command_line.py``; what is pinned here
is that this program's log goes through it.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    write_metadata,
)

from spindoctor.cli import sd_results_index
from spindoctor.cli.results_index import FanOut, IngestCounts
from spindoctor.config import MAIN_LOGGER
from spindoctor.nav_records import TreeTuning

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

SERVER_URL = f'postgresql+psycopg:/svc:{PASSWORD}@db.example/spindoctor'
"""An index URL carrying a password, in the one-slash form a parser rejects."""


def _run(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[int | None, list[str]]:
    """Run the driver and return its exit status and what it told the main log.

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
    monkeypatch.setattr(MAIN_LOGGER, 'info', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'warning', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'error', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'fatal', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'exception', recording)
    with pytest.raises(SystemExit) as caught:
        sd_results_index.main()
    status = caught.value.code
    return (status if status is None or isinstance(status, int) else 1), written


def test_no_navigation_root_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configuration failure is an operator's mistake, not a traceback."""
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    status, _written = _run(
        ['ingest', '--results-index-db', index_url(tmp_path / 'index.sqlite3')],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_no_navigation_root_says_which_settings_supply_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal that does not say what to set is a refusal nobody can act on."""
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    _status, written = _run(
        ['ingest', '--results-index-db', index_url(tmp_path / 'index.sqlite3')],
        monkeypatch,
        tmp_path,
    )
    assert any('--nav-results-root' in line for line in written)


def test_no_index_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest with nowhere to write is the other configuration failure.

    Both ambient sources of an index URL are closed for every test, so naming
    none on the command line is a program run with no index at all -- which for
    this one program is a mistake rather than the ordinary case it is everywhere
    else in the pipeline.  It is reported rather than raised, and names all
    three settings that supply one, since which of them an operator meant to
    set is theirs to know.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    status, written = _run(['ingest', '--nav-results-root', str(root)], monkeypatch, tmp_path)
    assert status == 1
    assert any('NAV_RESULTS_INDEX_DB' in line for line in written)


def test_an_index_named_with_an_empty_value_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value carrying no URL stops the ingest before it walks anything.

    Parameters:
        tmp_path: Directory the tree and the logs live under.
        monkeypatch: Fixture the argument vector and logger are replaced through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    status, _written = _run(
        ['ingest', '--nav-results-root', str(root), '--results-index-db', ''], monkeypatch, tmp_path
    )
    assert status == 1


def test_an_index_named_with_an_empty_value_says_which_level_named_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the level is the difference between a one-line fix and a hunt.

    Parameters:
        tmp_path: Directory the tree and the logs live under.
        monkeypatch: Fixture the argument vector and logger are replaced through.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        ['ingest', '--nav-results-root', str(root), '--results-index-db', ''], monkeypatch, tmp_path
    )
    assert any('--results-index-db is set to an empty value' in line for line in written)


def test_the_run_log_does_not_carry_a_database_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command line is logged, and a password can be one of its words."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        ['ingest', '--results-index-db', SERVER_URL, '--nav-results-root', root.as_posix()],
        monkeypatch,
        tmp_path,
    )
    assert not any(PASSWORD in line for line in written)


def test_the_run_log_still_names_the_index_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Masking must not cost the identification the log line exists for.

    Which of the command line, the configuration file and the environment
    supplied a bad URL is exactly what the logged arguments answer.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        ['ingest', '--results-index-db', SERVER_URL, '--nav-results-root', root.as_posix()],
        monkeypatch,
        tmp_path,
    )
    assert any('db.example/spindoctor' in line for line in written)


def test_the_run_log_names_the_roots_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary local root carries no credentials and reaches the log whole."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any(f'Roots: {root.as_posix()}' == line for line in written)


def test_a_root_that_is_not_there_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit status 1, because the run could not walk a root it was given."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    status, _written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_a_root_that_is_not_there_is_named_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped root reads as a root that is empty unless the summary says so."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    _status, written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('could not be listed' in line for line in written)


def _run_over_an_unlistable_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Run the driver over a root one directory of which will not be listed.

    Parameters:
        tmp_path: Directory the tree, the index and the logs live under.
        monkeypatch: Fixture the argument vector and logger are replaced through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    closed = root / 'VOL2'
    closed.chmod(0o000)
    try:
        return _run(
            [
                'ingest',
                '--results-index-db',
                index_url(tmp_path / 'index.sqlite3'),
                '--nav-results-root',
                root.as_posix(),
            ],
            monkeypatch,
            tmp_path,
        )
    finally:
        closed.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_directory_that_cannot_be_listed_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass ends rather than completing around documents it could not see.

    Every consumer of the index reads a missing row as "this image was never
    navigated", and under a directory nobody listed that reading is wrong, so
    the run an operator would otherwise act on is the thing that has to stop.
    """
    _status, written = _run_over_an_unlistable_directory(tmp_path, monkeypatch)
    assert any('Ingest stopped' in line for line in written)


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_the_directory_that_stopped_the_run_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is the one thing the operator has to go and fix."""
    _status, written = _run_over_an_unlistable_directory(tmp_path, monkeypatch)
    assert any((tmp_path / 'results' / 'VOL2').as_posix() in line for line in written)


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_run_stopped_by_a_directory_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduled ingest reads its status and nothing else."""
    status, _written = _run_over_an_unlistable_directory(tmp_path, monkeypatch)
    assert status == 1


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_the_directory_refusal_is_not_reported_as_an_unenumerated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch-all below exits 1 and says so too, which would hide this one.

    Status and traceback alike are the same for both, so what tells them apart
    is the message: this failure is one the pass enumerated, and it reads as the
    directory it is about rather than as something nobody expected.
    """
    _status, written = _run_over_an_unlistable_directory(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_a_failure_nobody_enumerated_exits_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A console entry point owes its caller a status, not a traceback.

    The pass charges every failure it expects to one file or one root. What is
    left is a failure nobody enumerated, and the driver's contract is that it
    exits either way -- otherwise a caller reading the status gets an exception
    instead, and the roots the pass never reached keep their unfinished runs
    with nothing said about why.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('a failure nobody enumerated')

    monkeypatch.setattr(sd_results_index, 'ingest_metadata_files', exploding)
    status, _written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_a_failure_nobody_enumerated_says_what_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit status with nothing in the log leaves nobody anything to act on."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('a failure nobody enumerated')

    monkeypatch.setattr(sd_results_index, 'ingest_metadata_files', exploding)
    _status, written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('a failure nobody enumerated' in line for line in written)


def test_a_failure_reason_names_one_example_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reason is a field-level diagnosis until one real file is named beside it."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        [
            'ingest',
            '--results-index-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    examples = [line for line in written if 'for example' in line]
    assert any('edges_metadata.json' in line for line in examples)


def _two_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[tuple[int | None, list[str]]]]:
    """Run the driver twice over a tree of files that are not navigation documents.

    This is the shape of the tree an operator measures a short selection
    against: every document under the root is one the ingest refuses, and the
    root has already been ingested, which is the only state a consumer accepts
    it in.  They sit under a subtree because that is where a selection looks:
    one enumerates the subtrees it was given, and a document above all of them
    is in no selection's answer whatever it records.

    Parameters:
        tmp_path: Directory the tree, the index and the logs live under.
        monkeypatch: Fixture the argument vector and logger are replaced through.

    Returns:
        The results root, and the exit status and main log of each pass.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    (root / 'VOL').mkdir(parents=True)
    (root / 'VOL' / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'VOL' / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    argv = [
        'ingest',
        '--results-index-db',
        index_url(tmp_path / 'index.sqlite3'),
        '--nav-results-root',
        root.as_posix(),
    ]
    return root, [_run(argv, monkeypatch, tmp_path), _run(argv, monkeypatch, tmp_path)]


def test_two_passes_over_one_tree_exit_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status that flips on an unchanged tree tells a scheduled run nothing.

    The first pass refuses both files and the second skips them as unchanged, so
    a status read from what was ingested or skipped reports a failure once and
    never again.  The pass completed both times, which is what the status says.
    """
    _root, passes = _two_passes(tmp_path, monkeypatch)
    assert [status for status, _written in passes] == [0, 0]


def test_a_second_pass_tallies_none_of_the_refusals_the_first_one_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass's own tally answers "what did this pass read", and nothing else.

    A refused file that has not changed is skipped without being read, so it
    never reaches the tally again.  The tally is therefore an account of one
    pass rather than of what the root holds, and an operator reading it as the
    second would conclude that a root full of refused files holds none.
    """
    _root, passes = _two_passes(tmp_path, monkeypatch)
    _status, written = passes[1]
    assert [line for line in written if line.startswith('Not ingestible')] == ['Not ingestible: 0']


def test_the_ingest_is_handed_the_tuning_the_configuration_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver resolves the tuning once, at its top, and the pass reads what it was handed."""
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    handed: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> IngestCounts:
        handed.append(kwargs.get('tuning'))
        return IngestCounts()

    monkeypatch.setattr(sd_results_index, 'get_results_tree_tuning', lambda config: configured)
    monkeypatch.setattr(sd_results_index, 'ingest_metadata_files', recording)
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    status, _ = _run(
        ['ingest', '--nav-results-root', str(root), '--results-index-db', url],
        monkeypatch,
        tmp_path,
    )
    assert status == 0
    assert handed == [configured]


def test_a_named_configuration_file_that_is_not_there_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mistyped --config-file is the operator's mistake, and is told as one."""
    status, _written = _run(
        ['ingest', '--config-file', str(tmp_path / 'absent.yaml')], monkeypatch, tmp_path
    )
    assert status == 1
    assert 'Invalid configuration' in capsys.readouterr().err


def test_a_divide_is_handed_the_same_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both subcommands that list a tree read the one resolution the driver made."""
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    handed: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> FanOut:
        handed.append(kwargs.get('tuning'))
        return FanOut()

    monkeypatch.setattr(sd_results_index, 'get_results_tree_tuning', lambda config: configured)
    monkeypatch.setattr(sd_results_index, 'fan_out_ingest_tasks', recording)
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    status, _ = _run(
        [
            'divide',
            '--nav-results-root',
            str(root),
            '--results-index-db',
            url,
            '--tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 0
    assert handed == [configured]
