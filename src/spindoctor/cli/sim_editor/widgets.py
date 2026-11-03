"""Small reusable Qt helpers for the simulated-image editor.

``ImageLabel`` forwards raw mouse and wheel events to caller-supplied
callbacks so the main window can own the pan / zoom / selection logic,
``ParameterUpdater`` coalesces rapid parameter edits into a single debounced
re-render request, ``make_dspin`` builds a fully configured double spin box in
one call for the mixins that lay out many numeric fields, and
``kept_or_shown`` rebuilds a toggled group's map from its spins without
rounding a loaded value.
"""

from collections.abc import Callable, Mapping
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QDoubleSpinBox, QLabel, QWidget


def make_dspin(
    *,
    minimum: float,
    maximum: float,
    decimals: int,
    step: float,
    value: float,
    tooltip: str = '',
) -> QDoubleSpinBox:
    """Build a configured ``QDoubleSpinBox`` with its value set before wiring.

    Parameters:
        minimum: Lower bound of the spin range.
        maximum: Upper bound of the spin range.
        decimals: Displayed decimal places.
        step: Single-step increment.
        value: Initial value (set before any signal is connected).
        tooltip: Optional tooltip text.

    Returns:
        The configured spin box, not yet connected to any handler.
    """
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


def kept_or_shown(
    spins: Mapping[str, QDoubleSpinBox],
    removed: Mapping[str, Any],
    *,
    null_shows: Mapping[str, float] | None = None,
) -> dict[str, float | None]:
    """Return each spin's key with its value from a removed map, or from the spin.

    A group that is turned off keeps the map it removed, and turning it back on
    rebuilds the map from its spins.  A spin rounds to its displayed decimals and
    stops at its range, so each key takes the removed map's exact value while the
    spin still shows that value, and the spin's own value once it was edited.  A
    key in ``null_shows`` may be null in the removed map, and stays null while its
    spin still shows the value given there.

    Parameters:
        spins: The group's spin boxes by map key.
        removed: The map the group removed when it was turned off, or an empty
            mapping when it never was.
        null_shows: The value each spin shows for a null key, by map key.

    Returns:
        The rebuilt values by map key.
    """
    values: dict[str, float | None] = {}
    for key, spin in spins.items():
        shown = float(spin.value())
        kept = removed.get(key)
        kept_value: float | None
        if isinstance(kept, (int, float)):
            kept_value, kept_shows = float(kept), float(kept)
        elif kept is None and key in removed and null_shows is not None and key in null_shows:
            kept_value, kept_shows = None, null_shows[key]
        else:
            values[key] = shown
            continue
        # Compare as the spin displays both: Qt rounds an exact tie half-up, round() to even.
        still_shown = spin.textFromValue(kept_shows) == spin.textFromValue(shown)
        values[key] = kept_value if still_shown else shown
    return values


class ImageLabel(QLabel):
    """A ``QLabel`` that forwards mouse and wheel events to callbacks."""

    def __init__(
        self,
        parent: QWidget | None,
        on_press: Callable[[QMouseEvent], None],
        on_move: Callable[[QMouseEvent], None],
        on_release: Callable[[QMouseEvent], None],
        on_wheel: Callable[[QWheelEvent], None],
    ) -> None:
        """Store the event callbacks.

        Parameters:
            parent: The parent widget, or None.
            on_press: Called on a mouse-press event.
            on_move: Called on a mouse-move event.
            on_release: Called on a mouse-release event.
            on_wheel: Called on a wheel event.
        """
        super().__init__(parent)
        self._on_press = on_press
        self._on_move = on_move
        self._on_release = on_release
        self._on_wheel = on_wheel

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Forward a mouse-press event to the press callback."""
        if event is not None:
            self._on_press(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Forward a mouse-move event to the move callback."""
        if event is not None:
            self._on_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Forward a mouse-release event to the release callback."""
        if event is not None:
            self._on_release(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Forward a wheel event to the wheel callback."""
        if event is not None:
            self._on_wheel(event)


class ParameterUpdater(QObject):
    """Debounces parameter edits into a single ``update_requested`` signal."""

    update_requested = pyqtSignal()

    def __init__(self, delay_ms: int) -> None:
        """Create the debounce timer.

        Parameters:
            delay_ms: Idle delay in milliseconds before an update fires.
        """
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setInterval(delay_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_update)

    def request_update(self) -> None:
        """Restart the debounce timer, deferring the render."""
        self._timer.start()

    def immediate_update(self) -> None:
        """Cancel any pending debounce and emit the update now."""
        self._timer.stop()
        self.update_requested.emit()

    def _emit_update(self) -> None:
        """Emit the update signal when the debounce timer fires."""
        self.update_requested.emit()
