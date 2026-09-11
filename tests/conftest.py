"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pdslogger
import pytest

import spindoctor
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    set_log_levels,
    set_strict_scope,
    strict_scope_override,
)
from spindoctor.config.log_scope import _reset_reported_call_sites

if TYPE_CHECKING:
    from tests.mini_nav_results.cohort import Cohort


@pytest.fixture(autouse=True)
def config_fixture() -> None:
    """Load bundled default config before each test if not already loaded."""
    DEFAULT_CONFIG.ensure_loaded()


USER_CONFIG_NAME = 'nav_default_config.yaml'
"""The user override file, which is resolved beside whatever process reads it."""


@pytest.fixture(scope='session')
def directory_naming_no_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a directory holding no user configuration file.

    One directory for the whole session rather than one per test: what makes it
    useful is what it does not hold, and it is emptied again after every test
    rather than trusted to stay that way.  What a test leaves in its working
    directory is what every later test of the same worker runs beside, so a
    configuration file left here would be resolved by all of them -- a failure
    landing arbitrarily far from the test that caused it.

    Parameters:
        tmp_path_factory: Factory the directory is made under.

    Returns:
        The directory.
    """
    return tmp_path_factory.mktemp('naming_no_index')


@pytest.fixture(scope='session', autouse=True)
def no_ambient_results_index_for_the_session(
    directory_naming_no_index: Path,
) -> Iterator[None]:
    """Close every ambient level for everything a session runs, not only tests.

    The per-test fixture below cannot reach a fixture of a broader scope: pytest
    builds a module- or session-scoped one before any function-scoped fixture of
    the test that first asked for it, so a fixture that ingests a tree or runs a
    report would run against the working directory and the environment the suite
    was started with.  Closing them here as well makes the guarantee one about
    the session rather than about test bodies.

    Parameters:
        directory_naming_no_index: The working directory to run under.

    Yields:
        Nothing; every level is closed for the life of the session.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(directory_naming_no_index)
        patch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
        patch.delenv('NAV_RESULTS_ROOT', raising=False)
        yield


@pytest.fixture(autouse=True)
def no_ambient_results_index(
    monkeypatch: pytest.MonkeyPatch, directory_naming_no_index: Path
) -> Iterator[None]:
    """Close every way a test could reach a results index or tree nobody named.

    A results index URL is resolved from three places in order: the argument,
    the ``environment.results_index_db`` configuration variable, and the
    ``NAV_RESULTS_INDEX_DB`` environment variable.  A test that names none of them is
    testing what a program does with no index, and on a machine that sets either
    ambient one it instead opens a real one -- for SQLite a write-lock probe
    against a file an ingest may be holding, and for a report a read of every
    row in it.  Both are closed here rather than in each test, because the level
    an author forgets is the level nothing then tests.

    The navigation results root is closed on the same terms, and for a sharper
    reason: a program that resolves one *walks* it.  ``sd_stats_report`` with no
    index reads every document under the tree it resolves, so a test naming
    neither an index nor a root would read whatever ``NAV_RESULTS_ROOT`` the
    machine exports -- several hundred thousand documents on a working machine,
    from a test that means to assert a refusal.

    The configuration level is closed by moving the working directory.  The user
    override file is ``nav_default_config.yaml`` beside the process, so a
    directory holding none is a configuration naming no index, whatever the
    directory the suite was started from holds.  A subprocess a test starts
    inherits both, so a test that gives one a working directory of its own names
    the directory it means.

    A test that wants either level sets it up for itself: what it does through
    the same fixture is undone before what is done here.

    The directory is shared by the whole session, so whatever a test writes into
    its working directory without moving there first is taken back out here, and
    a configuration file is reported as well: that one is not litter but a
    configuration every later test of this worker would resolve, and it has to
    fail the test that wrote it rather than one somewhere after it.

    Parameters:
        monkeypatch: Fixture the working directory and the environment are moved
            through.
        directory_naming_no_index: The working directory to run under.

    Yields:
        Nothing; the test runs with no ambient level reachable.
    """
    monkeypatch.chdir(directory_naming_no_index)
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    # A configuration merged before this test ran -- by a test that named its own
    # override file, or by one that ran before the working directory moved --
    # outlives it: the merge is into a process-global configuration and nothing
    # takes it back out.
    monkeypatch.delitem(DEFAULT_CONFIG.environment, 'results_index_db', raising=False)
    monkeypatch.delitem(DEFAULT_CONFIG.environment, 'nav_results_root', raising=False)
    yield
    left_behind = sorted(entry.name for entry in directory_naming_no_index.iterdir())
    for entry in directory_naming_no_index.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    if USER_CONFIG_NAME in left_behind:
        pytest.fail(
            f'This test wrote a {USER_CONFIG_NAME} into the directory the suite runs '
            'from, which is shared by every test of this worker and must name no results '
            "index. Move to a directory of the test's own -- monkeypatch.chdir(tmp_path) "
            '-- before writing anything relative to the working directory.'
        )


@pytest.fixture(autouse=True)
def restore_loggers_fixture() -> Iterator[None]:
    """Put both loggers back as they were found.

    Configuring a logger changes process state, and a cloud task deliberately
    reconfigures both of them for good -- a worker never un-isolates itself.
    Without this, one test resolving a cloud task's logging would silence every
    later test in the same worker, and the failures would land far from the
    cause.

    The restore is checked rather than assumed, and a test it could not undo
    fails here.  A logger holding an unexpected handler stops falling back to
    printing, so every later test in the same worker that reads its output
    through ``capsys`` sees nothing -- a failure with no visible connection to
    the test that caused it, appearing only in the worker that happened to run
    them in that order.  Failing at the source names the culprit instead.
    """
    main_handlers = list(MAIN_LOGGER.handlers)
    image_handlers = list(IMAGE_LOGGER.handlers)
    main_propagate = MAIN_LOGGER.propagate
    image_propagate = IMAGE_LOGGER.propagate
    main_level = MAIN_LOGGER.level
    image_level = IMAGE_LOGGER.level
    strict_override = strict_scope_override()
    yield
    left_behind: list[str] = []
    for name, logger, baseline in (
        ('main', MAIN_LOGGER, main_handlers),
        ('image', IMAGE_LOGGER, image_handlers),
    ):
        # remove_all_handlers only detaches, so a handler the test attached
        # would keep its log file open for the rest of the session.  Only what
        # the test added is closed; the baseline is put back as it was, and
        # NULL_HANDLER is a process-wide singleton nobody here owns.
        for handler in logger.handlers:
            if handler not in baseline and handler is not pdslogger.NULL_HANDLER:
                handler.close()
        logger.remove_all_handlers()
        for handler in baseline:
            logger.add_handler(handler)
        # NULL_HANDLER is excluded for the same reason the close loop above
        # excludes it: pdslogger can reinstate the process-wide singleton on
        # its own, and nobody here owns it, so finding it attached is not a
        # test leaving state behind.
        left_behind += [
            f'{name} logger: {handler!r}'
            for handler in logger.handlers
            if handler not in baseline and handler is not pdslogger.NULL_HANDLER
        ]
    MAIN_LOGGER.propagate = main_propagate
    IMAGE_LOGGER.propagate = image_propagate
    # Restored to what was found rather than to a level named here: a test that
    # sets one and puts back what it assumed the default was pins that
    # assumption on every test after it.
    MAIN_LOGGER.set_level(main_level)
    IMAGE_LOGGER.set_level(image_level)
    # The override, not the resolved value: saving the resolved boolean would
    # pin it and lose the deferral to the configuration.
    set_strict_scope(strict_override)
    # A test that logs out of scope otherwise leaves its call site in the
    # process-wide dedup set, so a later test asserting on that warning sees
    # nothing and fails somewhere unrelated to the cause.
    _reset_reported_call_sites()
    if len(left_behind) > 0:
        pytest.fail(
            'This test left a handler attached that could not be detached: '
            + '; '.join(left_behind)
            + '. One known cause: pdslogger identifies an open log file by the absolute '
            'path the working directory gives, so a handler built from a relative path '
            'cannot be found again once the working directory moves; build log handlers '
            'from an absolute path. A handler that was closed and then re-attached '
            'produces the same symptom.'
        )


@pytest.fixture(autouse=True)
def reset_log_levels_fixture() -> Iterator[None]:
    """Discard any resolved levels a test installs.

    The resolved set is process state, memoized on first use, so without this
    one test's levels would govern every later test in the same worker.
    """
    yield
    set_log_levels(None)


@pytest.fixture
def strict_log_scope() -> Iterator[None]:
    """Make an out-of-scope image log raise for the duration of a test.

    Opt-in rather than automatic.  A unit test exercising a model or technique
    in isolation calls it outside any image scope by design, which is correct
    practice and not the mis-binding this switch exists to catch, so enabling
    it for the whole suite would fail hundreds of legitimate tests.  Request it
    from a test that drives a real pipeline, where a scope genuinely should be
    open.

    Clears the override on exit rather than forcing it off, so behavior returns
    to whatever the configuration says.
    """
    set_strict_scope(True)
    yield
    set_strict_scope(None)


def child_interpreter_environment() -> dict[str, str]:
    """Return the environment a subprocess probe must run under.

    Several assertions in this suite are about what a *fresh* interpreter does --
    which modules an import pulls in, what a program writes to stdout -- and can
    only be made in a subprocess, because by the time any test runs this process
    has imported half the tree.  Every one of those probes has to be told which
    copy of SpinDoctor to import.

    Left to the inherited environment they are not: the suite runs each test from
    a directory of its own, so a relative ``PYTHONPATH=src`` resolves against
    that directory and the child imports whichever copy is installed instead.
    The probe then answers for somebody else's code, and it answers the same
    whatever the checkout under test does -- which is a test that cannot fail.
    This names the package by where *this* process imported it from.

    Returns:
        A copy of the environment with this checkout's source directory first on
        ``PYTHONPATH``, keeping whatever was already there behind it.
    """
    source_root = Path(spindoctor.__file__).resolve().parent.parent
    environment = dict(os.environ)
    inherited = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        f'{source_root}{os.pathsep}{inherited}' if inherited else str(source_root)
    )
    return environment


@pytest.fixture(scope='session')
def mini_nav_cohort(tmp_path_factory: pytest.TempPathFactory) -> Cohort:
    """Return the bundle cohort, built once for the whole session.

    The PDS4 phases assert against a navigation root and a backplane root that
    hold real products -- a FITS an astropy reader reopens, a PNG with a size
    and a checksum, and the documents beside them -- and building those per
    test would rewrite a FITS for every label a test looks at.  It is built
    into a temporary directory and torn down with the session, so nothing it
    produces reaches the working tree.

    Parameters:
        tmp_path_factory: Factory the cohort root is made under.

    Returns:
        The written cohort.
    """
    # Imported here rather than at the top: the cohort package pulls the whole
    # navigation stack in, and this file is loaded by every pytest process,
    # so a top-level import would cost every test the stack whether or not it
    # asks for the cohort.
    from tests.mini_nav_results.cohort import write_cohort

    return write_cohort(tmp_path_factory.mktemp('mini_nav_cohort'))
