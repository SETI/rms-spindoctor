"""Registry-driven artifact-mode rows for the simulated-image editor.

The Artifacts tab does not hand-code a section per artifact mode: it generates
one :class:`ModeRow` from each :class:`~spindoctor.sim.forward.artifact_modes.ArtifactMode`
in the registry, and each row builds its own widgets from the mode's
:class:`~spindoctor.sim.forward.artifact_modes.ModeParam` specs.  A registered
mode therefore acquires an editor row with no GUI change: the row's enable
checkbox maps to the mode key's presence, the activation-key spin
(``expected_events``, ``expected_events_per_sec``, ``probability``, or
``pixel_fraction``, as the registry names it) and one widget per parameter map
to that mode's map keys, and the parameter kinds pick the widget (a spin, a
checkbox, an enum combo, or a group of spins for a rect/window list).

The fixed ``count`` that ``dead_pixels`` and ``dead_columns`` take in place of
``expected_events`` carries a "set" box, like a list parameter, so a count of 0
is authorable and an unset count stays absent.  The two keys are mutually
exclusive in a scene, and the row keeps them so: setting the count removes
``expected_events``, and editing ``expected_events`` removes the count, so the
editor never authors a map the validator refuses.

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
    PIXEL_FRACTION,
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
# A hot-pixel fraction is a small density (1.6e-5 on the calibrated Cassini
# cameras, 2.75e-4 in the Titan truth scenes), so its spin keeps six decimals.
_PIXEL_FRACTION_DECIMALS = 6


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
        self._count_check: QCheckBox | None = None
        self._count_spin: QSpinBox | None = None

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
        if param.name == self.mode.count_alternative:
            form.addRow(f'{param.name}:', self._build_count_widget(param))
            return
        widget = self._build_scalar_widget(param)
        if param.name == self.mode.activation:
            widget.setToolTip(f'{param.name}: {self.mode.activation_semantics}')
        self._scalar_widgets[param.name] = widget
        form.addRow(f'{param.name}:', widget)

    def _build_count_widget(self, param: ModeParam) -> QWidget:
        """Build the "set" box and spin for the fixed-count alternative.

        The count key stays absent until the box is checked, and a checked box
        writes the spin's value, 0 included.
        """
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        check = QCheckBox('set')
        spin = QSpinBox()
        spin.setRange(0, _INT_MAX)
        tip = (
            f'a fixed event count in place of {self.mode.activation!r}; setting one '
            'removes the other'
        )
        check.setToolTip(tip)
        spin.setToolTip(tip)
        layout.addWidget(check)
        layout.addWidget(spin)
        self._count_check = check
        self._count_spin = spin
        self._scalar_widgets[param.name] = spin
        return container

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
            spin.setDecimals(
                _PIXEL_FRACTION_DECIMALS if param.name == PIXEL_FRACTION else _NUMBER_DECIMALS
            )
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
        count_key = self.mode.count_alternative
        if self._count_check is not None:
            self._count_check.toggled.connect(self._on_count_toggled)
        for name, widget in self._scalar_widgets.items():
            if name == count_key and isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._on_count_edited)
            elif isinstance(widget, QCheckBox):
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
        """Write a scalar parameter through the controller.

        Writing the activation key of a mode with a fixed-count alternative
        removes the count and clears its "set" box (the keys are exclusive).
        """
        self._controller._set_mode_param(self.mode.name, name, value)
        count_key = self.mode.count_alternative
        if name == self.mode.activation and count_key is not None and self._count_check is not None:
            self._controller._remove_mode_param(self.mode.name, count_key)
            self._count_check.blockSignals(True)
            self._count_check.setChecked(False)
            self._count_check.blockSignals(False)

    def _on_count_edited(self, _value: int) -> None:
        """Editing the count spin sets the count (checking its box if needed)."""
        if self._count_check is None:
            return
        if self._count_check.isChecked():
            self._write_count()
        else:
            self._count_check.setChecked(True)  # its toggled handler writes the count

    def _on_count_toggled(self, on: bool) -> None:
        """Write or remove the fixed count when its "set" box toggles."""
        count_key = self.mode.count_alternative
        if count_key is None:
            return
        if on:
            self._write_count()
        else:
            self._controller._remove_mode_param(self.mode.name, count_key)

    def _write_count(self) -> None:
        """Write the count and remove the activation key it replaces.

        The activation spin returns to its default display, since the key it
        shows is no longer in the map.
        """
        count_key = self.mode.count_alternative
        if count_key is None or self._count_spin is None:
            return
        self._controller._set_mode_param(self.mode.name, count_key, int(self._count_spin.value()))
        self._controller._remove_mode_param(self.mode.name, self.mode.activation)
        activation_param = self.mode.param_map[self.mode.activation]
        self._sync_scalar(activation_param, {})

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
        count_key = self.mode.count_alternative
        if self._count_check is not None and count_key is not None:
            self._count_check.blockSignals(True)
            self._count_check.setChecked(current.get(count_key) is not None)
            self._count_check.blockSignals(False)

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
        tooltip; an available mode is enabled and carries its activation key's
        semantics.
        """
        available = mode_available(self.mode.name, instrument)
        self.group.setEnabled(available)
        if available:
            self.group.setToolTip(f'{self.mode.activation}: {self.mode.activation_semantics}')
        else:
            self.group.setToolTip(mode_unavailable_message(self.mode.name, instrument))
