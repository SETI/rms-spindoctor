"""Pytest configuration and the shared Qt application for the UI tests (PyQt6).

``QT_QPA_PLATFORM`` must be set before any ``PyQt6`` import so headless Linux
hosts and CI runners do not require a display server.

A test process may have only one ``QApplication``, and only the fixture here makes it.
Once PyQt has destroyed an application, it no longer notices when Qt deletes an object
that Qt made itself, such as a spin box's line edit, so Python's wrapper for that object
stays in use.  Qt soon builds something else at the same address, PyQt takes it for the
deleted object, and a test sees the wrong widget or the process crashes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='session')
def _application() -> QApplication:
    """The one Qt application of the test process, held until the session ends."""
    # Imported here so that the UI tests that need no Qt still run where PyQt6 can't load.
    try:
        from PyQt6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
    except Exception as exc:
        pytest.skip(f'PyQt6 QApplication init failed: {exc}')
    return cast(QApplication, application)


@pytest.fixture
def qapp(_application: QApplication) -> Iterator[QApplication]:
    """The shared headless Qt application; the test's windows are deleted afterward.

    A window left to the garbage collector would keep its timers running in later tests,
    and would be deleted in the middle of one of them.
    """
    from PyQt6 import sip

    yield _application
    # Only what Python made: Qt owns the rest, such as a completer's popup, and deletes it
    # itself, so deleting it here could delete it twice.
    windows = [
        widget
        for widget in _application.topLevelWidgets()
        if widget.parent() is None and sip.ispycreated(widget)
    ]
    for window in windows:
        sip.delete(window)
