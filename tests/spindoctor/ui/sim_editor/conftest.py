"""Fixtures for the simulated-image editor's tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QApplication


@pytest.fixture
def qapp(qapp: QApplication) -> Iterator[QApplication]:
    """The shared Qt application; the test's background renders end before its windows go."""
    from tests.spindoctor.ui.sim_editor.sim_editor_waits import join_render_threads, render_threads

    already_running = render_threads()
    yield qapp
    join_render_threads(already_running)
