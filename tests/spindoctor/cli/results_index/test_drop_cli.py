"""Tests for ``sd_results_index drop``.

The destructive command, so what is asked of it is mostly what it does *not* do:
it does not drop without an answer, it does not treat a closed standard input as
consent, it does not walk a tree, it does not print a password, and it does not
leave behind a database anything reads differently from one nobody ever built.

The last of those is asked of the real consumer entry points rather than
asserted about the schema, because "indistinguishable to every consumer" is a
statement about them.
"""

import io
import sqlite3
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.cli.results_index.conftest import (
    rows_of,
)
from tests.spindoctor.cli.results_index.ingest_driver_helpers import process, run_driver
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)

from spindoctor.cli.results_index.drop import _because
from spindoctor.cli.stats.report import main_report
from spindoctor.dataset.results_filter import ResultsFilter
from spindoctor.results_index import IMAGES, SCHEMA_VERSION, index_table_names
from spindoctor.results_index import engine as engine_module

STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the document the trees below hold."""

SECRET = 'p@ss:w/rd?x#y'
"""A password carrying every character that means something to a URL.

Each of them is a place a rule reading the URL by eye stops early, and the
at-sign in the user name beside it is the other one.  A message that carries
this string carries a working password.
"""

CREDENTIALED_URL = f'postgresql+psycopg://ad@min:{SECRET}@127.0.0.1:1/spindoctor'
"""A server URL nothing answers on, carrying credentials worth hiding.

Port 1 refuses at once, so the refusal this drives is the connection failing
rather than a name lookup waiting.
"""


class _CodedError(Exception):
    """A driver exception carrying the code a server would have returned.

    Parameters:
        sqlstate: The five-character code.
    """

    def __init__(self, sqlstate: str) -> None:
        super().__init__(f'the server said {sqlstate}')
        self.sqlstate = sqlstate


def _failure_with(sqlstate: str) -> sqlalchemy.exc.SQLAlchemyError:
    """Return the failure SQLAlchemy raises around a driver exception of one code.

    Every one of these is reachable from a real server, and two of them are
    reached in the postgres tier; asking each of them here is what keeps the
    table of causes from losing a row unnoticed.

    Parameters:
        sqlstate: The five-character code the server returns.

    Returns:
        The wrapper, carrying the driver exception as its ``orig``.
    """
    return sqlalchemy.exc.OperationalError('DROP TABLE images', {}, _CodedError(sqlstate))


def _sqlite_failure_with(error_name: str) -> sqlalchemy.exc.SQLAlchemyError:
    """Return the failure SQLAlchemy raises around a SQLite result code.

    Parameters:
        error_name: The result-code name the driver carries.

    Returns:
        The wrapper, carrying the driver exception as its ``orig``.
    """
    original = sqlite3.OperationalError(f'the driver said {error_name}')
    original.sqlite_errorname = error_name
    return sqlalchemy.exc.OperationalError('DROP TABLE images', {}, original)


def _tree_with_an_index(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Write a one-document results tree, ingest it, and return the index URL.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    return url


def _answering(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Put an answer where the confirmation will read one.

    Parameters:
        monkeypatch: Fixture standard input is replaced through.
        answer: What is typed, including any newline.  An empty string is a
            standard input at its end, which is what a scheduled run has.
    """
    monkeypatch.setattr('sys.stdin', io.StringIO(answer))


def _drop(
    url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> tuple[int | None, list[str]]:
    """Run a drop of one index and return its status and main log.

    Parameters:
        url: The index URL to drop.
        monkeypatch: Fixture the driver is run through.
        tmp_path: Directory the run's log files are written under.
        extra: Further arguments, such as ``--yes``.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(['drop', '--results-index-db', url, *extra], monkeypatch, tmp_path)


def _tables(url: str) -> list[str]:
    """Return every table a database holds.

    Parameters:
        url: The database URL.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The confirmation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('answer', ['y\n', 'yes\n', 'Y\n', ' yes \n'])
def test_an_answer_of_yes_drops_the_tables(
    answer: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case and surrounding space are not part of the answer.

    Parameters:
        answer: What is typed at the prompt.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, answer)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == []


@pytest.mark.parametrize('answer', ['n\n', '\n', 'yes please\n', 'drop\n'])
def test_any_other_answer_leaves_the_index_alone(
    answer: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything that is not yes is no, including nothing at all.

    Parameters:
        answer: What is typed at the prompt.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, answer)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == sorted(index_table_names())


def test_an_answer_that_is_not_yes_exits_nonzero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script must not read success from a drop that did not happen.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    status, _written = _drop(url, monkeypatch, tmp_path)
    assert status == 1


def test_a_declined_drop_says_what_the_answer_was(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run log has to distinguish an answer of no from a failure to ask.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('rather than yes' in line for line in written)


def test_a_drop_with_nobody_to_ask_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standard input at its end is not consent.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == sorted(index_table_names())


def test_a_drop_with_nobody_to_ask_names_the_flag_that_would_have_worked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduled run's operator reads the log to find out what to add.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('--yes' in line for line in written)


def test_yes_drops_without_reading_an_answer(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Told not to ask, it does not ask, and does not obey what was typed anyway.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path, '--yes')
    assert _tables(url) == []


def test_the_question_is_put_to_the_terminal(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary is read in the second before an answer is typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert 'Drop 6 table(s) and ' in capsys.readouterr().out


def test_the_question_names_the_index_it_is_about(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The question has to stand on its own, for a run whose log went to a file.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert f'from {url}, schema ' in capsys.readouterr().out


def test_the_account_of_each_table_is_logged(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is at stake is rows, and the tree above holds exactly one image.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('images: 1 row(s)' in line for line in written)


def test_a_run_told_not_to_ask_still_records_what_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary goes to the log whether or not anybody was shown it.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('images: 1 row(s)' in line for line in written)


def test_the_drop_names_the_tables_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Says what it removed" is a list of names, not a count.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    dropped = [line for line in written if line.startswith('Dropped from ')]
    assert dropped[0].endswith(', '.join(index_table_names()))


def test_an_unfinished_ingest_run_is_reported_before_the_question(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drop under a live pass is allowed, so the person answering is told.

    The run is left unfinished the way a real one is: a fan-out records what its
    listing found and waits for workers that have not run.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
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
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('1 ingest run(s) have begun and not finished' in line for line in written)


def test_an_unfinished_ingest_run_is_named_in_the_question_itself(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one fact a person answering could not have known from what they typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
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
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert '1 ingest run(s) have not finished.' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Dropping nothing
# ---------------------------------------------------------------------------


def test_a_database_holding_no_index_exits_zero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state asked for is the state arrived at, so it is not a failure.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 0


def test_a_database_holding_no_index_says_it_removed_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idempotent drop has to be visibly idempotent rather than silent.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('reaches none of the results index tables' in line for line in written)


def test_the_empty_answer_is_about_this_connection_rather_than_the_database(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What was looked at is what an unqualified name on this connection reaches.

    An index in a schema this connection's search path does not name, or in one
    this account may not look into, answers exactly as one that is not there
    does, so the sentence says which of the two was established.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('outside its search path' in line for line in written)


def test_nobody_is_asked_when_there_is_nothing_to_drop(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A question about nothing is noise, so it is not asked at all.

    Standard input is at its end, which would refuse a drop that had something
    to remove.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    _answering(monkeypatch, '')
    status, _written = _drop(url, monkeypatch, tmp_path)
    assert status == 0


# ---------------------------------------------------------------------------
# What a drop does not need, and what it will not do
# ---------------------------------------------------------------------------


def test_a_drop_needs_no_results_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drop is about the database alone, on a machine that may not hold the tree.

    Neither the option nor the environment variable names one here, which is
    what stops every other mode of this program.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    url = _tree_with_an_index(tmp_path, quiet_logger)
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 0


def test_a_drop_ingests_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It stops where it stops: a mistyped URL costs one command, not a walk.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert not any(line.startswith('Metadata files seen') for line in written)


def test_a_database_that_is_not_there_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed path gets the answer a server gives for a database that is absent.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    url = index_url(tmp_path / 'absent.sqlite3')
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 1


def test_the_index_being_dropped_is_recorded_before_it_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log saying only that a database would not open leaves out which one.

    The database here is one that cannot be opened, so a line naming it is a
    line written before the attempt.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    url = index_url(tmp_path / 'absent.sqlite3')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert f'Results index to drop the tables of: {url}' in written


def test_a_database_that_is_not_there_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped URL must not leave an empty database where it pointed.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    path = tmp_path / 'absent.sqlite3'
    _drop(index_url(path), monkeypatch, tmp_path, '--yes')
    assert not path.exists()


# ---------------------------------------------------------------------------
# The password
# ---------------------------------------------------------------------------


def _credentialed_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run a drop of an unreachable server URL and return everything it said.

    Parameters:
        monkeypatch: Fixture the driver is run through.
        tmp_path: Directory the run's log files are written under.
        capsys: Fixture the terminal output is captured through.

    Returns:
        The main log and the terminal output, joined.
    """
    _status, written = _drop(CREDENTIALED_URL, monkeypatch, tmp_path, '--yes')
    captured = capsys.readouterr()
    return '\n'.join([*written, captured.out, captured.err])


def test_the_password_survives_nowhere_a_drop_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asserted on the bytes, over a password carrying every character that ends one.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    assert SECRET not in _credentialed_drop(monkeypatch, tmp_path, capsys)


def test_no_run_of_the_password_survives_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A driver quotes back the fragment it stopped on, not the field it read.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    said = _credentialed_drop(monkeypatch, tmp_path, capsys)
    assert not any(SECRET[start : start + 4] in said for start in range(len(SECRET) - 3))


def test_the_masked_url_still_names_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Masking must not cost the identification the message exists for.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    assert '127.0.0.1' in _credentialed_drop(monkeypatch, tmp_path, capsys)


def test_the_command_line_the_drop_logs_is_masked_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run banner records the arguments, and one of those words is the URL.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    said = _credentialed_drop(monkeypatch, tmp_path, capsys)
    assert 'ad@min:***@127.0.0.1' in said


# ---------------------------------------------------------------------------
# A dropped index and an index that never existed
# ---------------------------------------------------------------------------


def _dropped_and_never_built(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """Return the URL of an index that was dropped and of one never built.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The dropped index's URL and the never-built one's.
    """
    dropped = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(dropped, monkeypatch, tmp_path, '--yes')
    return dropped, index_url(tmp_path / 'never-built.sqlite3')


def _selecting_against(url: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Ask one index for the images of a volume that have been navigated.

    Parameters:
        url: The index to answer from.
        tmp_path: Directory the results tree lives under.
        quiet_logger: Logger the filter reports through.
    """
    ResultsFilter(
        ['VOL'],
        tmp_path / 'results',
        logger=quiet_logger,
        results_index_db_url=url,
        has_offset_file=True,
        tuning=packaged_tree_tuning(),
    )


def test_neither_a_dropped_index_nor_an_absent_one_answers_a_selection(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filters read absence of a row as "never navigated", so both must refuse.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch):
        with pytest.raises(ValueError, match='sd_results_index'):
            _selecting_against(url, tmp_path, quiet_logger)


def test_both_send_the_reader_to_the_ingest(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two states, one remedy, so the two refusals prescribe the same thing.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    said = []
    for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch):
        with pytest.raises(ValueError) as excinfo:
            _selecting_against(url, tmp_path, quiet_logger)
        said.append('sd_results_index' in str(excinfo.value))
    assert said == [True, True]


def test_the_report_refuses_both(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one consumer that requires an index treats the two the same way.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    statuses = [
        main_report(['--results-index-db', url, '--output-dir', str(tmp_path / 'report')])
        for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch)
    ]
    assert statuses == [1, 1]


def test_an_ingest_over_a_dropped_index_writes_the_rows_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting from scratch, which is the operation the whole command is for.

    The rows are compared against the ones the first ingest wrote, so a drop
    that had left something behind, or one an ingest could only half-rebuild
    over, shows up as a difference rather than as a count.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    before = rows_of(url, IMAGES)
    _drop(url, monkeypatch, tmp_path, '--yes')
    ingest_tree(url, [root], logger=quiet_logger)
    assert rows_of(url, IMAGES) == before


def test_a_rebuilt_index_carries_this_schema_version(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp goes with the tables and comes back with them.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            stamped: Any = connection.exec_driver_sql(
                'SELECT schema_version FROM schema_meta'
            ).scalar()
    finally:
        engine.dispose()
    assert stamped == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# A database that does not prove it holds an index of ours
# ---------------------------------------------------------------------------


def _database_holding(tmp_path: Path, *statements: str) -> str:
    """Build a database that SpinDoctor did not create, and return its URL.

    Parameters:
        tmp_path: Directory the database file is written into.
        statements: The statements that create what is in it.

    Returns:
        The database URL.
    """
    url = index_url(tmp_path / 'theirs.sqlite3')
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()
    return url


def test_a_stranger_s_table_of_one_of_our_names_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A URL naming something that is not a SpinDoctor index is refused.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver is run through.
    """
    url = _database_holding(
        tmp_path,
        'CREATE TABLE images (id INTEGER PRIMARY KEY, caption TEXT)',
        "INSERT INTO images (caption) VALUES ('somebody elses cat'), ('their dog')",
        'CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)',
    )
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 1


def test_such_a_refusal_names_the_tables_it_would_not_account_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator needs to know which tables stopped it, to decide about them.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver is run through.
    """
    url = _database_holding(tmp_path, 'CREATE TABLE images (id INTEGER PRIMARY KEY)')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('images' in line and 'schema_meta' in line for line in written)


def test_such_a_database_keeps_every_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The finding itself: two rows of somebody else's data, destroyed at exit 0.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver is run through.
    """
    url = _database_holding(
        tmp_path,
        'CREATE TABLE images (id INTEGER PRIMARY KEY, caption TEXT)',
        "INSERT INTO images (caption) VALUES ('somebody elses cat'), ('their dog')",
        'CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)',
    )
    _drop(url, monkeypatch, tmp_path, '--yes')
    assert _tables(url) == ['customers', 'images']


def test_such_a_database_keeps_the_rows_of_that_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counted, because a table recreated empty passes a test that counts tables.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver is run through.
    """
    url = _database_holding(
        tmp_path,
        'CREATE TABLE images (id INTEGER PRIMARY KEY, caption TEXT)',
        "INSERT INTO images (caption) VALUES ('somebody elses cat'), ('their dog')",
    )
    _drop(url, monkeypatch, tmp_path, '--yes')
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            remaining: Any = connection.exec_driver_sql('SELECT count(*) FROM images').scalar()
    finally:
        engine.dispose()
    assert remaining == 2


def test_such_a_database_is_never_called_the_results_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The account said "the SpinDoctor results index tables" over a stranger's rows.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver is run through.
    """
    url = _database_holding(tmp_path, 'CREATE TABLE images (id INTEGER PRIMARY KEY)')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert not any(
        line.startswith('About to drop the SpinDoctor results index') for line in written
    )


def test_nobody_is_asked_about_a_database_that_proves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no question to ask: the answer does not turn on consent.

    Standard input holds a yes, which would drop an index that was there.

    Parameters:
        tmp_path: Directory the database file is written into.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _database_holding(tmp_path, 'CREATE TABLE images (id INTEGER PRIMARY KEY)')
    _answering(monkeypatch, 'yes\n')
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == ['images']


# ---------------------------------------------------------------------------
# What the messages name
# ---------------------------------------------------------------------------


def test_the_refusal_for_want_of_an_answer_names_the_index(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every message of a destructive command says which database it is about.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('Nothing was dropped from ' in line and url in line for line in written)


def test_the_refusal_of_an_answer_that_is_not_yes_names_the_index(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same, for the refusal a person makes by typing one.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('rather than yes' in line and url in line for line in written)


def test_the_reading_is_announced_before_it_is_done(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting the rows of a large index is a scan per table, and looks like a hang.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any(line.startswith('Reading what ') for line in written)


def test_the_summary_names_the_schema_the_tables_are_in(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One URL reaches several schemas of a server, and only one holds the index.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    about = [line for line in written if line.startswith('About to drop ')]
    assert about[0].endswith(', schema main')


# ---------------------------------------------------------------------------
# Ctrl-C at the question
# ---------------------------------------------------------------------------


def _interrupting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the confirmation raise ``KeyboardInterrupt``, as Ctrl-C does.

    Parameters:
        monkeypatch: Fixture the built-in reader is replaced through.
    """

    def interrupted(prompt: str = '') -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr('builtins.input', interrupted)


def test_ctrl_c_at_the_question_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal a person makes with a key rather than a word.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the reader are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting(monkeypatch)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == sorted(index_table_names())


def test_ctrl_c_at_the_question_exits_nonzero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script must not read success from a drop nobody agreed to.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the reader are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting(monkeypatch)
    status, _written = _drop(url, monkeypatch, tmp_path)
    assert status == 1


def test_ctrl_c_at_the_question_says_so_rather_than_raising(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other refusal prints a line, and this one is not a traceback either.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the reader are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting(monkeypatch)
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('the question was interrupted' in line for line in written)


def _interrupting_the_step(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make one step of the drop raise ``KeyboardInterrupt``, as Ctrl-C does.

    The three steps this stands in for all wait on something outside the
    process -- a server accepting a connection, a scan of every table, a lock
    another session holds -- so each of them is a real place for Ctrl-C to land.

    Parameters:
        monkeypatch: Fixture the step is replaced through.
        name: The name the drop module calls that step by.
    """

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(f'spindoctor.cli.results_index.drop.{name}', interrupted)


@pytest.mark.parametrize(
    ('step', 'says'),
    [
        ('open_database', 'opening it was interrupted'),
        ('index_contents', 'the reading of what it holds was interrupted'),
        ('drop_index_tables', 'The drop of'),
    ],
    ids=['the-open', 'the-reading', 'the-drop'],
)
def test_ctrl_c_during_a_step_says_so_rather_than_raising(
    step: str,
    says: str,
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destructive command owes an interrupt a line wherever it lands.

    Parameters:
        step: The step of the drop that is interrupted.
        says: What the line for that step has to say.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the step are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting_the_step(monkeypatch, step)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any(says in line for line in written)


@pytest.mark.parametrize(
    'step',
    ['open_database', 'index_contents', 'drop_index_tables'],
    ids=['the-open', 'the-reading', 'the-drop'],
)
def test_ctrl_c_during_a_step_exits_nonzero(
    step: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script must not read success from a drop that was cut short.

    Parameters:
        step: The step of the drop that is interrupted.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the step are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting_the_step(monkeypatch, step)
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 1


def test_ctrl_c_during_the_drop_leaves_every_table(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reassurance the line gives is the one the transaction makes true.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the step are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _interrupting_the_step(monkeypatch, 'drop_index_tables')
    _drop(url, monkeypatch, tmp_path, '--yes')
    assert _tables(url) == sorted(index_table_names())


# ---------------------------------------------------------------------------
# A results root the drop never reads
# ---------------------------------------------------------------------------


def test_a_results_root_from_the_environment_does_not_refuse_a_drop(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine's standing setting is not a request, so it is not a conflict.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the environment are set through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_ROOT', str(tmp_path / 'results'))
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 0


# ---------------------------------------------------------------------------
# What a failure is blamed on
# ---------------------------------------------------------------------------


def test_a_live_writer_is_not_reported_as_a_filesystem_that_cannot_lock(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running ingest is the ordinary cause, and rebuilding a deployment is not the fix.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the busy timeout are set through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    holder = sqlite3.connect(tmp_path / 'index.sqlite3')
    try:
        holder.execute('BEGIN IMMEDIATE')
        _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    finally:
        holder.rollback()
        holder.close()
    assert any('Another process is holding it' in line for line in written)


def test_a_live_writer_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is what makes it a refusal rather than a wait.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the busy timeout are set through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    monkeypatch.setattr(engine_module, 'SQLITE_BUSY_TIMEOUT_MS', 50)
    holder = sqlite3.connect(tmp_path / 'index.sqlite3')
    try:
        holder.execute('BEGIN IMMEDIATE')
        _drop(url, monkeypatch, tmp_path, '--yes')
    finally:
        holder.rollback()
        holder.close()
    assert _tables(url) == sorted(index_table_names())


def test_an_unrecognized_failure_gets_no_cause_invented_for_it() -> None:
    """A code this does not know is reported as the database worded it, and no more.

    Parameters:
        None.
    """
    assert _because(sqlalchemy.exc.SQLAlchemyError('something nobody enumerated')) == ''


@pytest.mark.parametrize(
    ('sqlstate', 'names'),
    [
        ('55P03', 'lock'),
        ('2BP01', 'depends on'),
        ('42501', 'does not own'),
    ],
    ids=['a-lock', 'a-dependent-object', 'a-privilege'],
)
def test_each_failure_a_server_reports_is_named_as_itself(sqlstate: str, names: str) -> None:
    """Blaming a lock for a dependent view sends an operator hunting a session.

    Parameters:
        sqlstate: The five-character code the server returns.
        names: What the message for it has to say.
    """
    assert names in _because(_failure_with(sqlstate))


@pytest.mark.parametrize(
    ('error_name', 'names'),
    [
        ('SQLITE_BUSY_SNAPSHOT', 'holding the write lock'),
        ('SQLITE_READONLY_DBMOVED', 'read-only'),
    ],
    ids=['a-busy-snapshot', 'a-moved-database'],
)
def test_an_extended_sqlite_code_is_named_as_the_cause_it_refines(
    error_name: str, names: str
) -> None:
    """SQLite refines its codes into extended forms that name the same cause.

    ``SQLITE_BUSY`` arrives as ``SQLITE_BUSY_SNAPSHOT`` from a connection whose
    snapshot has moved on, and matching those by equality would answer the
    precise code with no cause at all.

    Parameters:
        error_name: The extended result-code name the driver carries.
        names: What the message for the code it refines has to say.
    """
    assert names in _because(_sqlite_failure_with(error_name))


@pytest.mark.parametrize(
    'sqlstate',
    ['2BP01', '42501'],
    ids=['a-dependent-object', 'a-privilege'],
)
def test_a_failure_that_is_not_a_lock_is_not_blamed_on_one(sqlstate: str) -> None:
    """Which is what every drop failure was blamed on.

    Parameters:
        sqlstate: The five-character code the server returns.
    """
    assert 'Another session' not in _because(_failure_with(sqlstate))


# ---------------------------------------------------------------------------
# The five programs that open an index
# ---------------------------------------------------------------------------


def test_neither_state_lets_a_cloud_task_worker_ingest_a_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker opens without creating, so both states are the same refusal.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    outcomes = []
    for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch):
        _retry, result = process(
            {'root_url': str(tmp_path / 'results'), 'files': [], 'run_id': 1}, url
        )
        outcomes.append(result['status_error'])
    assert outcomes == ['index_unopenable', 'index_unopenable']


def test_neither_state_lets_a_fan_out_be_completed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completion opens without creating too, so a wrong URL is not a first run.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    events = tmp_path / 'events.log'
    events.write_text('', encoding='utf-8')
    statuses = [
        run_driver(
            [
                'complete',
                '--results-index-db',
                url,
                '--nav-results-root',
                str(tmp_path / 'results'),
                '--events-log',
                str(events),
            ],
            monkeypatch,
            tmp_path,
        )[0]
        for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch)
    ]
    assert statuses == [1, 1]


def test_both_states_are_rebuilt_by_the_ingest_that_creates(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one opener that creates treats them alike as well: it builds one.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    root = tmp_path / 'results'
    statuses = [
        run_driver(
            ['ingest', '--results-index-db', url, '--nav-results-root', root.as_posix()],
            monkeypatch,
            tmp_path,
        )[0]
        for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch)
    ]
    assert statuses == [0, 0]


def test_neither_state_lets_a_drop_find_anything_to_drop(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fifth opener is the drop's own, and it answers the two the same way.

    An index that was dropped is a database holding none of these tables; one
    that was never built is a SQLite path that is not there, which is refused on
    both backends alike.  Both leave nothing behind and neither creates one.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    dropped, never_built = _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch)
    assert _drop(dropped, monkeypatch, tmp_path, '--yes')[0] == 0
    assert not Path(never_built.removeprefix('sqlite:///')).exists()


# ---------------------------------------------------------------------------
# The same command line against a server
# ---------------------------------------------------------------------------


@pytest.mark.postgres
def test_a_server_database_holding_no_index_exits_zero(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same status a SQLite database holding none of them exits with.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    status, _written = _drop(postgres_url, monkeypatch, tmp_path, '--yes')
    assert status == 0


@pytest.mark.postgres
def test_a_server_database_that_is_not_there_is_refused(
    postgres_server_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As a SQLite path that is not there is: one flag, one answer, two backends.

    Parameters:
        postgres_server_url: URL of the server, unscoped.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    absent = sqlalchemy.engine.make_url(postgres_server_url).set(database='ri_no_such_database')
    status, _written = _drop(
        absent.render_as_string(hide_password=False), monkeypatch, tmp_path, '--yes'
    )
    assert status == 1


@pytest.mark.postgres
def test_a_server_database_that_is_not_there_is_not_called_a_permissions_problem(
    postgres_server_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It was reported as tables that could not be read, over a database never reached.

    Parameters:
        postgres_server_url: URL of the server, unscoped.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    absent = sqlalchemy.engine.make_url(postgres_server_url).set(database='ri_no_such_database')
    _status, written = _drop(
        absent.render_as_string(hide_password=False), monkeypatch, tmp_path, '--yes'
    )
    said = '\n'.join(written)
    assert 'may read every table' not in said


@pytest.mark.postgres
def test_a_server_database_that_is_not_there_says_it_could_not_be_opened(
    postgres_server_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what happened, and what names the thing to fix.

    Parameters:
        postgres_server_url: URL of the server, unscoped.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    absent = sqlalchemy.engine.make_url(postgres_server_url).set(database='ri_no_such_database')
    _status, written = _drop(
        absent.render_as_string(hide_password=False), monkeypatch, tmp_path, '--yes'
    )
    assert any('Cannot open the database' in line for line in written)


def _index_under_a_dependent_view(url: str) -> None:
    """Build a stamped index table with a view of somebody's standing over it.

    Parameters:
        url: The scoped server URL the tables are created through.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE schema_meta (singleton int primary key, schema_version int, '
                'created_utc text)'
            )
            connection.exec_driver_sql(
                f'INSERT INTO schema_meta VALUES (1, {SCHEMA_VERSION}, now()::text)'
            )
            connection.exec_driver_sql('CREATE TABLE images (root_url text)')
            connection.exec_driver_sql('CREATE VIEW their_view AS SELECT * FROM images')
    finally:
        engine.dispose()


@pytest.mark.postgres
def test_a_dependent_view_is_not_reported_as_a_lock(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_logger: Any
) -> None:
    """A view over ``images`` is a drop failure nobody's session is holding.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        quiet_logger: Logger the ingest reports through, unused here.
    """
    _index_under_a_dependent_view(postgres_url)
    _status, written = _drop(postgres_url, monkeypatch, tmp_path, '--yes')
    said = '\n'.join(written)
    assert 'Another session' not in said


@pytest.mark.postgres
def test_a_dependent_view_is_reported_as_what_it_is(
    postgres_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So that the next step is to look at the view rather than at pg_stat_activity.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    _index_under_a_dependent_view(postgres_url)
    _status, written = _drop(postgres_url, monkeypatch, tmp_path, '--yes')
    assert any('depends on one of these tables' in line for line in written)


@pytest.mark.postgres
def test_a_drop_a_dependent_view_refused_leaves_every_table(
    postgres_url: str,
    postgres_server_url: str,
    postgres_schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transaction takes them back, so a refused drop costs nothing.

    Parameters:
        postgres_url: URL of an empty schema of this test's own.
        postgres_server_url: URL of the server that schema lives on.
        postgres_schema: Name of that schema.
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
    """
    _index_under_a_dependent_view(postgres_url)
    _drop(postgres_url, monkeypatch, tmp_path, '--yes')
    remaining = sqlalchemy.create_engine(postgres_server_url)
    try:
        held = sorted(sqlalchemy.inspect(remaining).get_table_names(schema=postgres_schema))
    finally:
        remaining.dispose()
    assert held == ['images', 'schema_meta']


def test_the_tables_dropped_are_the_tables_the_question_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second reading between the question and the drop is a different list.

    The index here is missing one of its tables when the question is put, and
    that table is put back while the answer is being typed.  What goes is what
    the person was shown.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and the reader are replaced through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _run_sql(url, 'DROP TABLE ingest_runs')

    def answering_and_meddling(prompt: str = '') -> str:
        _run_sql(url, 'CREATE TABLE ingest_runs (run_id INTEGER PRIMARY KEY)')
        return 'yes'

    monkeypatch.setattr('builtins.input', answering_and_meddling)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == ['ingest_runs']


def _run_sql(url: str, *statements: str) -> None:
    """Run statements against a database from outside the command under test.

    Parameters:
        url: The database URL.
        statements: The statements to run, in order.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()
