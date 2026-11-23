"""Tests for ``spindoctor.ui.mosaic_viewer.histogram_stretch``."""

import os

import numpy as np
import numpy.ma as ma
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

from spindoctor.ui.mosaic_viewer.histogram_stretch import HistogramStretchWidget


def _make_widget() -> tuple[HistogramStretchWidget, list[float], list[float]]:
    blacks: list[float] = []
    whites: list[float] = []
    widget = HistogramStretchWidget(
        on_black_changed=blacks.append,
        on_white_changed=whites.append,
    )
    return widget, blacks, whites


def test_initial_state_has_unit_range(qapp: QApplication) -> None:
    """Defaults map black=0, white=1 with no histogram data."""
    widget, _, _ = _make_widget()
    assert widget.get_values() == (0.0, 1.0)


def test_set_data_autoscales_to_data_extents(qapp: QApplication) -> None:
    """``set_data`` derives vmin/vmax from the valid pixels of the input."""
    widget, _, _ = _make_widget()
    img = ma.MaskedArray(np.linspace(5.0, 15.0, 100).reshape(10, 10))
    widget.set_data(img)
    # Indicators are clamped into the new range
    black, white = widget.get_values()
    assert 5.0 <= black <= 15.0
    assert 5.0 <= white <= 15.0


def test_set_data_handles_all_masked(qapp: QApplication) -> None:
    """All-masked input falls back to a unit range without crashing."""
    widget, _, _ = _make_widget()
    img = ma.MaskedArray(np.zeros((4, 4)), mask=np.ones((4, 4), dtype=bool))
    widget.set_data(img)
    assert widget.get_values() == (0.0, 1.0)


def test_set_data_with_none_clears(qapp: QApplication) -> None:
    """Passing ``None`` clears the histogram without changing the range."""
    widget, _, _ = _make_widget()
    img = ma.MaskedArray(np.linspace(0.0, 10.0, 100).reshape(10, 10))
    widget.set_data(img)
    widget.set_data(None)
    # No assertion on internal counts; just confirm the call did not raise


def test_set_values_does_not_fire_callbacks(qapp: QApplication) -> None:
    """``set_values`` updates positions silently."""
    widget, blacks, whites = _make_widget()
    widget.set_values(0.2, 0.8)
    assert widget.get_values() == (0.2, 0.8)
    assert blacks == []
    assert whites == []


def test_set_range_clamps_existing_values(qapp: QApplication) -> None:
    """Out-of-range black/white are clamped to the new range."""
    widget, _, _ = _make_widget()
    widget.set_values(0.0, 1.0)
    widget.set_range(5.0, 10.0)
    black, white = widget.get_values()
    assert 5.0 <= black <= 10.0
    assert 5.0 <= white <= 10.0


def test_set_range_with_degenerate_max_uses_unit_extension(qapp: QApplication) -> None:
    """``vmax <= vmin`` is widened to ``vmin + 1`` instead of dividing by zero."""
    widget, _, _ = _make_widget()
    widget.set_range(3.0, 3.0)
    widget.set_values(3.5, 3.8)
    assert widget.get_values() == (3.5, 3.8)


def test_drag_black_indicator_invokes_callback(qapp: QApplication) -> None:
    """Pressing on the black indicator and moving the mouse fires the callback."""
    widget, blacks, whites = _make_widget()
    widget.set_range(0.0, 1.0)
    widget.set_values(0.2, 0.8)
    widget.resize(400, 200)
    widget.show()
    qapp.processEvents()

    black_x = widget._value_to_x(0.2)
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(black_x, 50),
        QPointF(black_x, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(press)
    target_x = widget._value_to_x(0.4)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(target_x, 50),
        QPointF(target_x, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseMoveEvent(move)
    release = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(target_x, 50),
        QPointF(target_x, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseReleaseEvent(release)
    assert len(blacks) >= 1
    assert whites == []
    final_black, _ = widget.get_values()
    assert 0.3 < final_black < 0.5


def test_drag_does_not_push_black_past_white(qapp: QApplication) -> None:
    """Black indicator is clamped strictly below white during drag."""
    widget, blacks, _ = _make_widget()
    widget.set_range(0.0, 1.0)
    widget.set_values(0.2, 0.5)
    widget.resize(400, 200)

    black_x = widget._value_to_x(0.2)
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(black_x, 50),
        QPointF(black_x, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(press)
    far_right = widget._value_to_x(0.95)
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(far_right, 50),
        QPointF(far_right, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseMoveEvent(move)
    assert blacks
    final_black, final_white = widget.get_values()
    assert final_black < final_white
    assert final_white == 0.5


def test_paint_event_runs_without_data(qapp: QApplication) -> None:
    """``paintEvent`` is safe even before any histogram data has been set."""
    widget, _, _ = _make_widget()
    widget.resize(200, 150)
    widget.show()
    widget.repaint()
    qapp.processEvents()
