"""Tests of the wait for a test's background renders to end.

Qt runs headless (offscreen platform).
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from tests.conftest import child_interpreter_environment

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

from tests.spindoctor.ui.sim_editor import sim_editor_waits
from tests.spindoctor.ui.sim_editor.sim_editor_waits import join_render_threads, render_threads

from spindoctor.ui.sim_editor.worker import RENDER_THREAD_NAME

# Two tests, as two editor tests would hold them: the first starts a render that ends a
# moment after the test does, and the second finds out whether it was waited for.
_TWO_TESTS = """
import threading
import time

from spindoctor.ui.sim_editor.worker import RENDER_THREAD_NAME

threads = []

def test_first_test_leaves_a_render_running(qapp):
    thread = threading.Thread(target=time.sleep, args=(1.0,), name=RENDER_THREAD_NAME)
    thread.start()
    threads.append(thread)

def test_second_test_finds_the_render_ended():
    assert not threads[0].is_alive()
"""


def _start_render_thread(release: threading.Event) -> threading.Thread:
    """Start a thread named as a background render, which ends when ``release`` is set."""
    thread = threading.Thread(target=release.wait, name=RENDER_THREAD_NAME)
    thread.start()
    return thread


def test_the_fixture_waits_for_the_renders_a_test_started(tmp_path: Path) -> None:
    """The editor fixture joins a render the test left running, before the next test."""
    here = Path(__file__).parent
    (tmp_path / 'conftest.py').write_text((here / 'conftest.py').read_text())
    module = tmp_path / 'test_two.py'
    module.write_text(_TWO_TESTS)
    env = child_interpreter_environment()
    root = Path(__file__).parents[4]
    env['PYTHONPATH'] = os.pathsep.join([str(root), env['PYTHONPATH']])
    result = subprocess.run(
        [
            *(sys.executable, '-m', 'pytest', '-q', '--rootdir', str(tmp_path)),
            *('-c', os.devnull, '-p', 'tests.spindoctor.ui.conftest'),
            *('-p', 'no:cacheprovider', str(module)),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_join_leaves_a_render_that_was_already_running() -> None:
    """A render an earlier test left behind is not this test's to wait for."""
    release = threading.Event()
    earlier = _start_render_thread(release)
    try:
        already_running = render_threads()
        later_release = threading.Event()
        later = _start_render_thread(later_release)
        later_release.set()
        join_render_threads(already_running)
        assert not later.is_alive()
        assert earlier.is_alive()
    finally:
        release.set()
        earlier.join()


def test_the_join_fails_for_a_render_that_outlives_its_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render still running when the wait ends fails the test."""
    monkeypatch.setattr(sim_editor_waits, 'TIMEOUT_SEC', 0.05)
    release = threading.Event()
    stuck = _start_render_thread(release)
    try:
        with pytest.raises(pytest.fail.Exception, match='outlived its test'):
            join_render_threads()
    finally:
        release.set()
        stuck.join()
