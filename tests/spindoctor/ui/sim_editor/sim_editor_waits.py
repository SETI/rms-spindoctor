"""Event-loop waits for the scene editor's background renders.

Import this only after the importing test module has made sure PyQt6 is usable.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from PyQt6.QtWidgets import QApplication

from spindoctor.ui.sim_editor.worker import RENDER_THREAD_NAME

# Generous: a gated render finishes as soon as it is released, and a real render in
# these tests is a small frame.
TIMEOUT_SEC = 20.0

# A full-render delay no test outlasts, for tests that start the background renders
# themselves: an idle timer firing mid-test would start renders the test never expected.
NEVER_IDLE_SEC = 3600.0


def process_until(condition: Callable[[], bool]) -> None:
    """Run the Qt event loop until ``condition`` holds, failing after the timeout.

    Parameters:
        condition: Called between rounds of event processing.
    """
    deadline = time.monotonic() + TIMEOUT_SEC
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail('timed out waiting for a background render')
        QApplication.processEvents()
        time.sleep(0.005)


def wait_for_background_renders(model: Any) -> None:
    """Run the Qt event loop until neither background render is running or waiting.

    Parameters:
        model: The editor window.
    """
    process_until(
        lambda: not model._verdict_runner.is_busy and not model._full_render_runner.is_busy
    )


def render_fully(model: Any) -> None:
    """Draw the fast preview, then run the full renders as the idle timer would.

    Parameters:
        model: The editor window.
    """
    model._update_image()
    model._start_full_renders()
    wait_for_background_renders(model)


def render_threads() -> frozenset[threading.Thread]:
    """The background render threads that are running now."""
    return frozenset(
        thread for thread in threading.enumerate() if thread.name == RENDER_THREAD_NAME
    )


def join_render_threads(already_running: frozenset[threading.Thread] = frozenset()) -> None:
    """Wait for the test's background renders to end, failing after the timeout.

    The event loop doesn't run meanwhile, so a request still waiting never starts.

    Parameters:
        already_running: Render threads that were running before the test began, which
            belong to no test that is ending now and are not waited for.
    """
    for thread in render_threads() - already_running:
        thread.join(TIMEOUT_SEC)
        if thread.is_alive():
            pytest.fail('a background render outlived its test')
