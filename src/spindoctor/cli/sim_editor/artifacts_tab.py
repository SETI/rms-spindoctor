"""Artifacts tab for the simulated-image editor.

Authors the scene-level ``artifacts`` block: the ``instrument_defaults`` switch,
the ``adversarial`` switch, the ``detector`` override, and one generated row per
artifact mode in the registry.  ``instrument_defaults`` opts the whole scene into
the emulated camera's physical signal chain at catalog values (PSF, distortion
residual, shot noise, dark / hot / bloom / banding / bias); ``adversarial`` biases
each enabled mode's stochastic placement onto the navigation features.

The mode rows are not hand-coded: each is generated from an
:class:`~spindoctor.sim.artifact_modes.ArtifactMode` (see
:mod:`spindoctor.cli.sim_editor.artifact_mode_rows`), so a registered mode gains
an editor row with no change here.  The whole tab keeps one discipline: an
unchecked switch or mode leaves its key absent, an enabled mode starts as an
empty map, and every parameter key -- including the detector override's -- appears
only once its widget is edited, so keys the operator never touched track the
per-instrument catalog defaults.  If the scene's instrument doesn't offer a mode,
or the scene's detector chain can't render it, the mode's row is disabled, and its
tooltip says why.  If the scene already includes such a mode, its row stays enabled
until the author turns the mode off.  The rows refresh when the instrument, the
detector model, or the binning changes.

The detector group is a checkable override for six keys: the gain state, the
binning, the detector model, the reference exposure, the quantization, and the
downlink table.  It too writes only the keys the author edits, so a vgiss scene
stays vidicon without ever writing ``detector_model``.  Choosing 'camera default'
for the downlink table removes the key, so the camera's default table applies.
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
from spindoctor.sim.artifact_modes import ARTIFACT_MODES, DETECTOR_MODELS, QUANTIZATION_MODES
from spindoctor.sim.instrument_resolvers import (
    cataloged_binnings,
    downlink_table_names,
    resolve_detector_defaults,
    resolve_detector_model,
    resolve_readout_mode,
)

# The rendering stages the mode rows are grouped under, in tab order.
_STAGE_TITLES: tuple[tuple[str, str], ...] = (
    ('telemetry', 'Telemetry-stage modes'),
    ('detector', 'Detector-stage modes'),
)

# The downlink-table choice that leaves ``detector.downlink_table`` out of the scene.
_CAMERA_DEFAULT_TABLE = 'camera default'


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
        """Enable or disable each mode row for the scene's instrument and detector model.

        The detector model is the one the scene renders with: its own
        ``detector.detector_model``, else the camera's default in the scene's
        readout mode, as the detector resolver reads it.  If the scene already
        includes a mode and the mode then becomes unavailable, as on an instrument
        change, the scene keeps it and scene validation reports it.  The mode's row
        stays enabled until the author turns the mode off.
        """
        instrument = self.sim_params.get('instrument')
        detector = self.sim_params.get('detector')
        model = resolve_detector_model(instrument, detector if isinstance(detector, dict) else None)
        for row in self._mode_rows.values():
            row.apply_availability(instrument, model)

    # ---- Detector override group ----

    def _detector_catalog(self, binning: int) -> dict[str, Any]:
        """The selected instrument's detector defaults in one readout mode.

        Where the readout mode sets a key, its value replaces the native one, as
        the detector resolver merges them, so a readout mode with its own default
        gain state or quantization shows that default.

        Parameters:
            binning: The scene's readout-mode binning.
        """
        instrument = self.sim_params.get('instrument')
        return {
            **resolve_detector_defaults(instrument),
            **resolve_readout_mode(instrument, binning),
        }

    def _detector_block(self) -> dict[str, Any]:
        """The scene's detector keys that carry a value; a null reads as absent."""
        detector = self.sim_params.get('detector')
        if not isinstance(detector, dict):
            return {}
        return {key: value for key, value in detector.items() if value is not None}

    def _build_detector_group(self) -> QGroupBox:
        """Build the detector-override group (per-key discipline)."""
        group = QGroupBox('Detector override')
        group.setCheckable(True)
        form = QFormLayout(group)
        detector = self.sim_params.get('detector')

        # The spin's range follows the instrument's cataloged gain states; see
        # _set_gain_state_range.
        self._detector_gain_state_spin = QSpinBox()
        self._detector_gain_state_spin.setToolTip(
            "Electron-chain gain state; the range spans the instrument's cataloged "
            'states. Written to the scene only when edited.'
        )
        form.addRow('Gain state:', self._detector_gain_state_spin)

        self._detector_binning_combo = QComboBox()
        self._detector_binning_combo.setToolTip(
            'Readout-mode binning: each frame pixel sums a b-by-b block of native pixels. '
            "Lists the instrument's cataloged modes. Written to the scene only when edited."
        )
        form.addRow('Binning:', self._detector_binning_combo)

        self._detector_model_combo = QComboBox()
        self._detector_model_combo.addItems(list(DETECTOR_MODELS))
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
        self._detector_quantization_combo.addItems(list(QUANTIZATION_MODES))
        self._detector_quantization_combo.setToolTip(
            'ADC quantization sub-mode. Written to the scene only when edited.'
        )
        form.addRow('Quantization:', self._detector_quantization_combo)

        self._detector_downlink_combo = QComboBox()
        self._detector_downlink_combo.setToolTip(
            'The table the sqrt_lut quantization decodes through. '
            f"Choose '{_CAMERA_DEFAULT_TABLE}' to use the camera's own table. "
            'Written to the scene only when edited.'
        )
        form.addRow('Downlink table:', self._detector_downlink_combo)

        self._detector_group = group
        self._set_detector_widget_values()
        group.setChecked(isinstance(detector, dict))
        # Connect after the initial values so the build does not write.
        self._detector_gain_state_spin.valueChanged.connect(self._on_detector_gain_state)
        self._detector_binning_combo.currentTextChanged.connect(self._on_detector_binning)
        self._detector_exposure_ref_spin.valueChanged.connect(self._on_detector_exposure_ref)
        self._detector_model_combo.currentTextChanged.connect(self._on_detector_model)
        self._detector_quantization_combo.currentTextChanged.connect(self._on_detector_quantization)
        self._detector_downlink_combo.currentIndexChanged.connect(self._on_detector_downlink_table)
        group.toggled.connect(self._on_detector_group_toggled)
        return group

    def _set_detector_widget_values(self) -> None:
        """Show each detector key's scene value, or the instrument catalog default.

        Only display state: nothing is written back to ``sim_params``, so a
        key the operator never edits stays absent and keeps tracking the
        catalog.
        """
        binning = int(self._detector_block().get('binning', 1))
        self._set_binning_choices(binning)
        self._show_readout_mode_values(binning)
        self._set_downlink_table_choices()

    def _show_readout_mode_values(self, binning: int) -> None:
        """Show each detector key other than the binning, under one readout mode.

        If the scene sets a key, its widget shows the scene's value; otherwise the
        widget shows the instrument's default in that readout mode.  Only display
        state.

        Parameters:
            binning: The readout-mode binning the defaults come from.
        """
        block = self._detector_block()
        catalog = self._detector_catalog(binning)
        gain_state = int(block.get('gain_state', catalog.get('default_gain_state', 0)))
        self._set_gain_state_range(catalog, gain_state)
        self._detector_gain_state_spin.setValue(gain_state)
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

    def _set_gain_state_range(self, catalog: dict[str, Any], gain_state: int) -> None:
        """Span the gain-state spin over the instrument's cataloged states.

        The catalog requires an instrument's gain states to be contiguous
        integers, so the range from the lowest to the highest holds exactly the
        cataloged states.  If the scene's state lies outside it, the range widens
        to hold it, so the widget shows what the scene says and validation reports
        it.  If the instrument has no gain table (a vidicon), its single state is 0.

        Parameters:
            catalog: The instrument's detector defaults.
            gain_state: The state the spin is about to show.
        """
        states = {*(catalog.get('gain_e_per_dn_by_state') or {0: 1.0}), gain_state}
        self._detector_gain_state_spin.setRange(min(states), max(states))

    def _set_binning_choices(self, binning: int) -> None:
        """List the instrument's cataloged binnings and select ``binning``.

        A scene binning the instrument doesn't catalog is listed too, so the
        widget shows what the scene says and validation reports it.
        """
        choices = list(cataloged_binnings(self.sim_params.get('instrument')))
        if binning not in choices:
            choices.append(binning)
        self._detector_binning_combo.clear()
        self._detector_binning_combo.addItems([str(choice) for choice in sorted(choices)])
        self._detector_binning_combo.setCurrentText(str(binning))

    def _set_downlink_table_choices(self) -> None:
        """List the instrument's downlink tables and select the scene's.

        The first choice stands for an absent key and is selected when the scene sets
        no table.  If the scene sets a table the instrument doesn't list, that table is
        listed too, so the widget shows what the scene says and validation reports it.
        """
        selected = self._detector_block().get('downlink_table')
        choices = list(downlink_table_names(self.sim_params.get('instrument')))
        if selected is not None and str(selected) not in choices:
            choices.append(str(selected))
        combo = self._detector_downlink_combo
        combo.clear()
        combo.addItems([_CAMERA_DEFAULT_TABLE, *choices])
        combo.setCurrentIndex(0 if selected is None else choices.index(str(selected)) + 1)

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

    def _remove_detector_key(self, key: str) -> None:
        """Remove one detector key, preserving every other authored key."""
        if self._syncing or not self._detector_group.isChecked():
            return
        detector = self.sim_params.get('detector')
        if isinstance(detector, dict) and key in detector:
            del detector[key]
            self._updater.request_update()

    def _on_detector_group_toggled(self, on: bool) -> None:
        """Insert an empty detector block, or remove the block entirely.

        The block starts empty so unedited keys stay absent (the catalog
        defaults keep applying); each widget edit then writes only its key.
        Turning the group on shows the new block's values, so no widget keeps
        an edit the removed block held.
        """
        if self._syncing:
            return
        if on:
            if not isinstance(self.sim_params.get('detector'), dict):
                self.sim_params['detector'] = {}
            self._refresh_detector_catalog_defaults()
        else:
            self.sim_params.pop('detector', None)
        self._refresh_artifact_mode_availability()
        self._updater.request_update()

    def _on_detector_gain_state(self, value: int) -> None:
        """Write the gain state on a spin edit."""
        self._set_detector_key('gain_state', int(value))

    def _on_detector_binning(self, text: str) -> None:
        """Write the readout-mode binning on a selection, then show that mode's defaults.

        Every other key's display is refreshed, so an unedited gain state or
        quantization shows what the scene renders with in the new mode.  The mode
        rows follow the detector model of the new readout mode.
        """
        if not text or self._syncing:
            return
        self._set_detector_key('binning', int(text))
        self._syncing = True
        try:
            self._show_readout_mode_values(int(self._detector_block().get('binning', 1)))
        finally:
            self._syncing = False
        self._refresh_artifact_mode_availability()

    def _on_detector_exposure_ref(self, value: float) -> None:
        """Write the reference exposure on a spin edit."""
        self._set_detector_key('exposure_ref_sec', float(value))

    def _on_detector_model(self, text: str) -> None:
        """Write the detector model on a combo edit, then refresh the mode rows.

        If the new model's chain can't render a mode, the mode's row is disabled;
        if it can, the row is enabled again.
        """
        self._set_detector_key('detector_model', text or 'ccd')
        self._refresh_artifact_mode_availability()

    def _on_detector_quantization(self, text: str) -> None:
        """Write the quantization sub-mode on a combo edit."""
        self._set_detector_key('quantization', text or 'exact')

    def _on_detector_downlink_table(self, index: int) -> None:
        """Write the downlink table on a selection, or remove it on the first choice.

        Removing the key lets the author drop a table once the scene no longer
        quantizes with ``sqrt_lut``, without clearing the whole detector block.  The
        choices are then rebuilt, so a table the instrument doesn't list is no longer
        offered.
        """
        if index < 0 or self._syncing:
            return
        if index == 0:
            self._remove_detector_key('downlink_table')
            self._syncing = True
            try:
                self._set_downlink_table_choices()
            finally:
                self._syncing = False
        else:
            self._set_detector_key('downlink_table', self._detector_downlink_combo.itemText(index))

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
