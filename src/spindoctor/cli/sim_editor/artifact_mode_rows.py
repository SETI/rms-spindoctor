"""Registry-driven artifact-mode rows for the simulated-image editor.

The Artifacts tab does not hand-code a section per artifact mode: it generates
one :class:`ModeRow` from each :class:`~spindoctor.sim.forward.artifact_modes.ArtifactMode`
in the registry, and each row builds its own widgets from the mode's
:class:`~spindoctor.sim.forward.artifact_modes.ModeParam` specs.  A registered
mode therefore acquires an editor row with no GUI change: the row's enable
checkbox maps to the mode key's presence, the incidence spin and one widget per
parameter map to that mode's map keys, and the parameter kinds pick the widget
(a spin, a checkbox, an enum combo, or a group of spins for a rect/window list).

The row never touches ``sim_params`` itself: every edit calls back into the
controller (the Artifacts-tab mixin), which owns the artifacts block and the
absent-key discipline -- an unchecked mode leaves its key absent, an enabled mode
starts as an empty map, and each parameter key appears only once its widget is
edited.  Availability filtering is per row: a mode unavailable on the scene's
instrument is disabled with the registry's reason as its tooltip.
"""

from __future__ import annotations

from typing import Any, Protocol

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

from spindoctor.sim.forward.artifact_modes import (
    ArtifactMode,
    ModeParam,
    mode_available,
    mode_unavailable_message,
)

# Generous numeric bounds: artifact parameters span pixel counts, electron
# amplitudes, and unit fractions, so the spins are not the validation surface
# (the scene validator is); they only need to not clip a legitimate value.
_NUMBER_MIN = -1.0e6
_NUMBER_MAX = 1.0e6
_INT_MAX = 1_000_000
_NUMBER_DECIMALS = 4


class ArtifactModeController(Protocol):
    """The seam a :class:`ModeRow` writes through (the Artifacts-tab mixin)."""

    _syncing: bool

    def _mode_map(self, mode_name: str) -> dict[str, Any] | None:
        """Return the mode's scene map, or None when the mode key is absent."""

    def _set_mode_enabled(self, mode_name: str, on: bool) -> None:
        """Insert an empty mode map, or remove the mode key entirely."""

    def _set_mode_param(self, mode_name: str, param_name: str, value: Any) -> None:
        """Write one parameter key into an enabled mode's map."""

    def _remove_mode_param(self, mode_name: str, param_name: str) -> None:
        """Remove one parameter key from an enabled mode's map."""


class ModeRow:
    """A checkable group of widgets editing one artifact mode's scene map.

    The group's checked state is the mode key's presence; the widgets inside it
    edit the mode's parameters.  Building the row wires every widget's change
    signal to the controller, so an edit writes straight through the absent-key
    discipline the controller enforces.
    """

    def __init__(self, mode: ArtifactMode, controller: ArtifactModeController) -> None:
        """Build the row's widgets from the mode's parameter specs.

        Parameters:
            mode: The registered artifact mode this row edits.
            controller: The Artifacts-tab mixin the row writes through.
        """
        self.mode = mode
        self._controller = controller
        self._scalar_widgets: dict[str, QWidget] = {}
        self._enum_choices: dict[str, tuple[Any, ...]] = {}
        self._list_checks: dict[str, QCheckBox] = {}
        self._list_spins: dict[str, list[QSpinBox]] = {}

        self.group = QGroupBox(f'{mode.name}  ({mode.stage})')
        self.group.setCheckable(True)
        form = QFormLayout(self.group)
        for param in mode.params:
            self._add_param_row(form, param)

        self._sync_widgets(self._controller._mode_map(mode.name))
        # Connect after the initial sync so building the row never writes.
        self.group.toggled.connect(self._on_group_toggled)
        self._connect_param_signals()

    # ---- Row construction ----

    def _add_param_row(self, form: QFormLayout, param: ModeParam) -> None:
        """Add one labeled parameter widget to the row's form."""
        if param.kind == 'int_list':
            form.addRow(f'{param.name}:', self._build_int_list_widget(param))
            return
        widget = self._build_scalar_widget(param)
        self._scalar_widgets[param.name] = widget
        form.addRow(f'{param.name}:', widget)

    def _build_scalar_widget(self, param: ModeParam) -> QWidget:
        """Build the single widget for a scalar (non-list) parameter kind."""
        if param.kind == 'bool':
            return QCheckBox()
        if param.kind == 'enum':
            combo = QComboBox()
            choices = param.choices or ()
            self._enum_choices[param.name] = choices
            combo.addItems([str(choice) for choice in choices])
            return combo
        if param.kind in ('nonneg_number', 'unit_interval'):
            spin = QDoubleSpinBox()
            spin.setDecimals(_NUMBER_DECIMALS)
            spin.setRange(0.0, 1.0 if param.kind == 'unit_interval' else _NUMBER_MAX)
            return spin
        # The integer kinds.
        spin_int = QSpinBox()
        low = (
            1
            if param.kind == 'positive_int'
            else 0
            if param.kind == 'nonneg_int'
            else int(_NUMBER_MIN)
        )
        spin_int.setRange(low, _INT_MAX)
        return spin_int

    def _build_int_list_widget(self, param: ModeParam) -> QWidget:
        """Build a checkable group of spins for a rect / window integer-list param.

        The list key stays absent until the small enable box is checked (a
        rect / window is optional on every mode that offers one), so the row
        keeps the absent-key discipline for its list parameters too.
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        check = QCheckBox('set')
        layout.addWidget(check)
        spins: list[QSpinBox] = []
        for _ in range(param.length or 0):
            spin = QSpinBox()
            # The validator accepts any integers in a list parameter (a ghost
            # offset such as residual_image.offset_px is legitimately negative),
            # so the spins must not clip below zero.
            spin.setRange(int(_NUMBER_MIN), _INT_MAX)
            layout.addWidget(spin)
            spins.append(spin)
        self._list_checks[param.name] = check
        self._list_spins[param.name] = spins
        return container

    # ---- Signal wiring ----

    def _connect_param_signals(self) -> None:
        """Connect every parameter widget's change signal to the controller."""
        for name, widget in self._scalar_widgets.items():
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(lambda checked, n=name: self._write_scalar(n, checked))
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda text, n=name: self._write_enum(n, text))
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(
                    lambda value, n=name: self._write_scalar(n, float(value))
                )
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(lambda value, n=name: self._write_scalar(n, int(value)))
        for name, check in self._list_checks.items():
            check.toggled.connect(lambda on, n=name: self._on_list_toggled(n, on))
        for name, spins in self._list_spins.items():
            for spin in spins:
                spin.valueChanged.connect(lambda _value, n=name: self._write_list(n))

    def _write_scalar(self, name: str, value: Any) -> None:
        """Write a scalar parameter through the controller."""
        self._controller._set_mode_param(self.mode.name, name, value)

    def _write_enum(self, name: str, text: str) -> None:
        """Write an enum parameter, mapping the display text back to its choice."""
        for choice in self._enum_choices.get(name, ()):
            if str(choice) == text:
                self._controller._set_mode_param(self.mode.name, name, choice)
                return

    def _on_list_toggled(self, name: str, on: bool) -> None:
        """Write or remove an integer-list key when its enable box toggles."""
        if on:
            self._write_list(name)
        else:
            self._controller._remove_mode_param(self.mode.name, name)

    def _write_list(self, name: str) -> None:
        """Write an integer-list parameter from its spins, if its box is checked."""
        if not self._list_checks[name].isChecked():
            return
        self._controller._set_mode_param(
            self.mode.name, name, [int(spin.value()) for spin in self._list_spins[name]]
        )

    def _on_group_toggled(self, on: bool) -> None:
        """Insert or remove the mode map when the row's enable box toggles."""
        self._controller._set_mode_enabled(self.mode.name, on)

    # ---- Display sync ----

    def sync(self) -> None:
        """Rebuild the row's widget state from the current scene map."""
        self._sync_widgets(self._controller._mode_map(self.mode.name))

    def _sync_widgets(self, mode_map: dict[str, Any] | None) -> None:
        """Set the row's checked state and widget values from a mode map.

        Display state only: signals are blocked so no write occurs, and a key
        absent from ``mode_map`` shows the registry default without authoring it.
        """
        enabled = mode_map is not None
        self.group.blockSignals(True)
        self.group.setChecked(enabled)
        self.group.blockSignals(False)
        current = mode_map or {}
        for param in self.mode.params:
            if param.kind == 'int_list':
                self._sync_int_list(param, current)
            else:
                self._sync_scalar(param, current)

    def _sync_scalar(self, param: ModeParam, current: dict[str, Any]) -> None:
        """Show one scalar parameter's scene value, or its registry default."""
        widget = self._scalar_widgets[param.name]
        value = current.get(param.name, param.default)
        widget.blockSignals(True)
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            index = widget.findText(str(value if value is not None else param.default))
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value) if value is not None else 0.0)
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value) if value is not None else widget.minimum())
        widget.blockSignals(False)

    def _sync_int_list(self, param: ModeParam, current: dict[str, Any]) -> None:
        """Show one integer-list parameter: check the box iff the key is present."""
        check = self._list_checks[param.name]
        spins = self._list_spins[param.name]
        value = current.get(param.name)
        check.blockSignals(True)
        check.setChecked(value is not None)
        check.blockSignals(False)
        if isinstance(value, (list, tuple)):
            for spin, entry in zip(spins, value, strict=False):
                spin.blockSignals(True)
                spin.setValue(int(entry))
                spin.blockSignals(False)

    # ---- Availability ----

    def apply_availability(self, instrument: str | None) -> None:
        """Enable or disable the row for the scene's instrument.

        An unavailable mode is disabled with the registry's reason as its
        tooltip; an available mode is enabled and carries its incidence semantics.
        """
        available = mode_available(self.mode.name, instrument)
        self.group.setEnabled(available)
        if available:
            self.group.setToolTip(f'incidence: {self.mode.incidence_semantics}')
        else:
            self.group.setToolTip(mode_unavailable_message(self.mode.name, instrument))
