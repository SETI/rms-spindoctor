"""Artifacts tab for the simulated-image editor.

Authors the scene-level ``artifacts`` block: the ``instrument_defaults`` switch,
the ``adversarial`` switch, the ``detector`` override, and one generated row per
artifact mode in the registry.  ``instrument_defaults`` opts the whole scene into
the emulated camera's physical signal chain at catalog values (PSF, distortion
residual, shot noise, dark / hot / bloom / banding / bias); ``adversarial`` biases
each enabled mode's stochastic placement onto the navigation features.

The mode rows are not hand-coded: each is generated from an
:class:`~spindoctor.sim.forward.artifact_modes.ArtifactMode` (see
:mod:`spindoctor.cli.sim_editor.artifact_mode_rows`), so a registered mode gains
an editor row with no change here.  The whole tab keeps one discipline: an
unchecked switch or mode leaves its key absent, an enabled mode starts as an
empty map, and every parameter key -- including the detector override's -- appears
only once its widget is edited, so keys the operator never touched track the
per-instrument catalog defaults.  A mode unavailable on the scene's instrument is
disabled with the registry's reason as its tooltip.

The detector group is a checkable override for the electron-chain gain state, the
detector model, the exposure the well fraction references, and the ADC
quantization sub-mode, under the same per-key discipline (a vgiss scene stays
vidicon without ever writing ``detector_model``).
"""

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.artifact_mode_rows import ModeRow
from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES
from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults

_DETECTOR_MODELS: list[str] = ['ccd', 'vidicon']
_QUANTIZATION_MODES: list[str] = [
    'exact',
    '8bit',
    'uneven_12bit',
    'sqrt_lut',
    'ls8b',
    'contour_8bit',
]

# The rendering stages the mode rows are grouped under, in tab order.
_STAGE_TITLES: tuple[tuple[str, str], ...] = (
    ('telemetry', 'Telemetry-stage modes'),
    ('detector', 'Detector-stage modes'),
)


class ArtifactsTabMixin(SimEditorBase):
    """Builds and handles the Artifacts tab."""

    def _build_artifacts_tab(self) -> QWidget:
        """Build the scrollable Artifacts tab and return its container widget."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_artifacts_switch_group())
        layout.addWidget(self._build_detector_group())
        self._mode_rows = {}
        for stage, title in _STAGE_TITLES:
            layout.addWidget(self._build_mode_stage_group(stage, title))
        self._refresh_artifact_mode_availability()
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self._artifacts_tab = scroll
        return scroll

    # ---- Instrument-defaults and adversarial switches ----

    def _build_artifacts_switch_group(self) -> QGroupBox:
        """Build the physical-signal-chain opt-in and adversarial-placement group."""
        group = QGroupBox('Artifacts')
        form = QFormLayout(group)
        self._instrument_defaults_check = QCheckBox()
        self._instrument_defaults_check.setToolTip(
            "Turn on the emulated camera's physical signal chain at catalog "
            'values (PSF, distortion residual, shot noise, dark / hot / bloom '
            '/ banding / bias noise).'
        )
        self._instrument_defaults_check.setChecked(self._instrument_defaults_on())
        form.addRow('Instrument defaults:', self._instrument_defaults_check)
        self._instrument_defaults_check.toggled.connect(self._on_instrument_defaults)

        self._adversarial_check = QCheckBox()
        self._adversarial_check.setToolTip(
            "Bias every enabled mode's stochastic placement onto the navigation "
            'features (worst-case placement); uniform when off.'
        )
        self._adversarial_check.setChecked(self._adversarial_on())
        form.addRow('Adversarial placement:', self._adversarial_check)
        self._adversarial_check.toggled.connect(self._on_adversarial)
        return group

    def _instrument_defaults_on(self) -> bool:
        """Whether the scene opts into the instrument physical chain."""
        artifacts = self.sim_params.get('artifacts')
        return isinstance(artifacts, dict) and bool(artifacts.get('instrument_defaults', False))

    def _adversarial_on(self) -> bool:
        """Whether the scene requests adversarial artifact placement."""
        artifacts = self.sim_params.get('artifacts')
        return isinstance(artifacts, dict) and bool(artifacts.get('adversarial', False))

    def _on_instrument_defaults(self, checked: bool) -> None:
        """Set or clear the instrument-defaults switch, keeping other keys."""
        self._set_artifacts_switch('instrument_defaults', checked)

    def _on_adversarial(self, checked: bool) -> None:
        """Set or clear the adversarial switch, keeping other keys."""
        self._set_artifacts_switch('adversarial', checked)

    # ---- artifacts-block helpers (the absent-key discipline) ----

    def _artifacts_block(self) -> dict[str, Any]:
        """Return the artifacts block, inserting an empty one if absent."""
        block = self.sim_params.get('artifacts')
        if not isinstance(block, dict):
            block = {}
            self.sim_params['artifacts'] = block
        return block

    def _prune_artifacts(self) -> None:
        """Drop the artifacts block entirely once it holds no keys."""
        block = self.sim_params.get('artifacts')
        if isinstance(block, dict) and not block:
            self.sim_params.pop('artifacts', None)

    def _set_artifacts_switch(self, key: str, on: bool) -> None:
        """Write a boolean switch key, or remove it (unchecked = absent)."""
        if self._syncing:
            return
        if on:
            self._artifacts_block()[key] = True
        else:
            block = self.sim_params.get('artifacts')
            if isinstance(block, dict):
                block.pop(key, None)
            self._prune_artifacts()
        self._updater.request_update()

    def _mode_map(self, mode_name: str) -> dict[str, Any] | None:
        """Return a mode's scene map, or None when its key is absent."""
        block = self.sim_params.get('artifacts')
        if isinstance(block, dict):
            mode_map = block.get(mode_name)
            if isinstance(mode_map, dict):
                return mode_map
        return None

    def _set_mode_enabled(self, mode_name: str, on: bool) -> None:
        """Insert an empty mode map, or remove the mode key entirely."""
        if self._syncing:
            return
        if on:
            block = self._artifacts_block()
            if not isinstance(block.get(mode_name), dict):
                block[mode_name] = {}
        else:
            existing = self.sim_params.get('artifacts')
            if isinstance(existing, dict):
                existing.pop(mode_name, None)
            self._prune_artifacts()
        self._updater.request_update()

    def _set_mode_param(self, mode_name: str, param_name: str, value: Any) -> None:
        """Write one parameter key into an enabled mode's map."""
        if self._syncing:
            return
        mode_map = self._mode_map(mode_name)
        if mode_map is None:
            return
        mode_map[param_name] = value
        self._updater.request_update()

    def _remove_mode_param(self, mode_name: str, param_name: str) -> None:
        """Remove one parameter key from an enabled mode's map."""
        if self._syncing:
            return
        mode_map = self._mode_map(mode_name)
        if mode_map is None:
            return
        mode_map.pop(param_name, None)
        self._updater.request_update()

    # ---- Mode rows ----

    def _build_mode_stage_group(self, stage: str, title: str) -> QGroupBox:
        """Build the group of generated mode rows for one rendering stage."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        for mode in ARTIFACT_MODES.values():
            if mode.stage != stage:
                continue
            row = ModeRow(mode, self)
            self._mode_rows[mode.name] = row
            layout.addWidget(row.group)
        return group

    def _refresh_artifact_mode_availability(self) -> None:
        """Enable or disable each mode row for the scene's instrument."""
        instrument = self.sim_params.get('instrument')
        for row in self._mode_rows.values():
            row.apply_availability(instrument)

    # ---- Detector override group ----

    def _detector_catalog(self) -> dict[str, Any]:
        """The selected instrument's detector catalog defaults."""
        return resolve_detector_defaults(self.sim_params.get('instrument'))

    def _build_detector_group(self) -> QGroupBox:
        """Build the detector-override group (per-key discipline)."""
        group = QGroupBox('Detector override')
        group.setCheckable(True)
        form = QFormLayout(group)
        detector = self.sim_params.get('detector')

        self._detector_gain_state_spin = QSpinBox()
        self._detector_gain_state_spin.setRange(0, 3)
        self._detector_gain_state_spin.setToolTip(
            'Electron-chain gain state; must be cataloged for the instrument. '
            'Written to the scene only when edited.'
        )
        form.addRow('Gain state:', self._detector_gain_state_spin)

        self._detector_model_combo = QComboBox()
        self._detector_model_combo.addItems(_DETECTOR_MODELS)
        self._detector_model_combo.setToolTip(
            'ccd electron chain or vidicon DN chain. Written to the scene only when edited.'
        )
        form.addRow('Detector model:', self._detector_model_combo)

        self._detector_exposure_ref_spin = QDoubleSpinBox()
        self._detector_exposure_ref_spin.setRange(0.001, 10000.0)
        self._detector_exposure_ref_spin.setDecimals(4)
        self._detector_exposure_ref_spin.setSingleStep(0.1)
        self._detector_exposure_ref_spin.setToolTip(
            'Exposure (sec) the signal full-scale fraction references. '
            'Written to the scene only when edited.'
        )
        form.addRow('Exposure ref (sec):', self._detector_exposure_ref_spin)

        self._detector_quantization_combo = QComboBox()
        self._detector_quantization_combo.addItems(_QUANTIZATION_MODES)
        self._detector_quantization_combo.setToolTip(
            'ADC quantization sub-mode. Written to the scene only when edited.'
        )
        form.addRow('Quantization:', self._detector_quantization_combo)

        self._detector_group = group
        self._set_detector_widget_values()
        group.setChecked(isinstance(detector, dict))
        # Connect after the initial values so the build does not write.
        self._detector_gain_state_spin.valueChanged.connect(self._on_detector_gain_state)
        self._detector_exposure_ref_spin.valueChanged.connect(self._on_detector_exposure_ref)
        self._detector_model_combo.currentTextChanged.connect(self._on_detector_model)
        self._detector_quantization_combo.currentTextChanged.connect(self._on_detector_quantization)
        group.toggled.connect(self._on_detector_group_toggled)
        return group

    def _set_detector_widget_values(self) -> None:
        """Show each detector key's scene value, or the instrument catalog default.

        Only display state: nothing is written back to ``sim_params``, so a
        key the operator never edits stays absent and keeps tracking the
        catalog.
        """
        detector = self.sim_params.get('detector')
        block = detector if isinstance(detector, dict) else {}
        catalog = self._detector_catalog()
        self._detector_gain_state_spin.setValue(
            int(block.get('gain_state', catalog.get('default_gain_state', 0)))
        )
        model = str(block.get('detector_model', catalog.get('detector_model', 'ccd')))
        model_index = self._detector_model_combo.findText(model)
        if model_index >= 0:
            self._detector_model_combo.setCurrentIndex(model_index)
        self._detector_exposure_ref_spin.setValue(
            float(block.get('exposure_ref_sec', catalog.get('exposure_ref_sec', 1.0)))
        )
        quantization = str(block.get('quantization', catalog.get('quantization', 'exact')))
        quantization_index = self._detector_quantization_combo.findText(quantization)
        if quantization_index >= 0:
            self._detector_quantization_combo.setCurrentIndex(quantization_index)

    def _refresh_detector_catalog_defaults(self) -> None:
        """Refresh the displayed catalog defaults after an instrument change.

        Restores the caller's _syncing state on exit, so a refresh nested
        inside a wider sync never re-enables widget writes early.
        """
        was_syncing = self._syncing
        self._syncing = True
        try:
            self._set_detector_widget_values()
        finally:
            self._syncing = was_syncing

    def _set_detector_key(self, key: str, value: Any) -> None:
        """Write one detector key, preserving every other authored key."""
        if self._syncing or not self._detector_group.isChecked():
            return
        detector = self.sim_params.get('detector')
        if not isinstance(detector, dict):
            detector = {}
            self.sim_params['detector'] = detector
        detector[key] = value
        self._updater.request_update()

    def _on_detector_group_toggled(self, on: bool) -> None:
        """Insert an empty detector block, or remove the block entirely.

        The block starts empty so unedited keys stay absent (the catalog
        defaults keep applying); each widget edit then writes only its key.
        """
        if self._syncing:
            return
        if on:
            if not isinstance(self.sim_params.get('detector'), dict):
                self.sim_params['detector'] = {}
        else:
            self.sim_params.pop('detector', None)
        self._updater.request_update()

    def _on_detector_gain_state(self, value: int) -> None:
        """Write the gain state on a spin edit."""
        self._set_detector_key('gain_state', int(value))

    def _on_detector_exposure_ref(self, value: float) -> None:
        """Write the reference exposure on a spin edit."""
        self._set_detector_key('exposure_ref_sec', float(value))

    def _on_detector_model(self, text: str) -> None:
        """Write the detector model on a combo edit."""
        self._set_detector_key('detector_model', text or 'ccd')

    def _on_detector_quantization(self, text: str) -> None:
        """Write the quantization sub-mode on a combo edit."""
        self._set_detector_key('quantization', text or 'exact')

    # ---- Scene-load sync ----

    def _sync_artifacts_from_params(self) -> None:
        """Rebuild every Artifacts-tab widget from the current sim_params."""
        self._syncing = True
        try:
            self._instrument_defaults_check.setChecked(self._instrument_defaults_on())
            self._adversarial_check.setChecked(self._adversarial_on())
            self._set_detector_widget_values()
            self._detector_group.setChecked(isinstance(self.sim_params.get('detector'), dict))
            for row in self._mode_rows.values():
                row.sync()
        finally:
            self._syncing = False
        self._refresh_artifact_mode_availability()
