"""Stray-light panel, hosted in the Optics tab's stray-light group.

An additive low-frequency gradient the navigator's BANDPASS_DOG filter is meant
to suppress.  Writes the ``sim_params['optics']['stray_light']`` block;
amplitude 0 (default) means off.

The radial model's bump center is a position, so no value of it can double as
"unset": a separate enable box says whether the scene authors the center keys,
and with the box clear the spins show the frame center the renderer applies in
their absence.  A center is stated as a pixel corner, like every other position
in a scene.
"""

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout

from spindoctor.cli.sim_editor.base import SimEditorBase


class StrayLightMixin(SimEditorBase):
    """Builds and handles the stray-light panel."""

    def _build_stray_panel(self, gen_layout: QFormLayout) -> None:
        """Add the stray-light rows to the hosting group's form layout.

        Parameters:
            gen_layout: The form layout of the Optics tab's stray-light group.
        """
        self._stray_amplitude_spin = QDoubleSpinBox()
        self._stray_amplitude_spin.setRange(0.0, 1.0)
        self._stray_amplitude_spin.setDecimals(3)
        self._stray_amplitude_spin.setSingleStep(0.01)
        self._stray_amplitude_spin.setValue(float(self._stray_value('amplitude', 0.0)))
        self._stray_amplitude_spin.setToolTip('Stray-light amplitude (0 = off).')
        self._stray_amplitude_spin.valueChanged.connect(self._on_stray_amplitude)
        gen_layout.addRow('Stray light amplitude:', self._stray_amplitude_spin)

        self._stray_direction_spin = QDoubleSpinBox()
        self._stray_direction_spin.setRange(0.0, 360.0)
        self._stray_direction_spin.setDecimals(1)
        self._stray_direction_spin.setWrapping(True)
        self._stray_direction_spin.setValue(float(self._stray_value('direction_deg', 0.0)))
        self._stray_direction_spin.setToolTip('Gradient direction for the linear model, degrees.')
        self._stray_direction_spin.valueChanged.connect(self._on_stray_direction)
        gen_layout.addRow('Stray light direction (deg):', self._stray_direction_spin)

        self._stray_model_combo = QComboBox()
        self._stray_model_combo.addItems(['linear', 'radial'])
        stray_model = str(self._stray_value('model', 'linear'))
        stray_model_index = self._stray_model_combo.findText(stray_model)
        if stray_model_index >= 0:
            self._stray_model_combo.setCurrentIndex(stray_model_index)
        self._stray_model_combo.setToolTip('linear ramp or radial bump.')
        self._stray_model_combo.currentTextChanged.connect(self._on_stray_model)
        gen_layout.addRow('Stray light model:', self._stray_model_combo)

        # The bump center is an optional key pair: an absent key means the frame
        # center, and 0.0 is a real position on both axes, so a separate enable
        # box says whether the keys are authored at all.  A spin writes only its
        # own key, and only while the enable is on; unchecked drops the keys and
        # the spins show the effective default rather than a sentinel.
        has_center = self._stray_has_center()
        self._stray_center_check = QCheckBox('Set bump center')
        self._stray_center_check.setChecked(has_center)
        self._stray_center_check.setToolTip(
            'Enable authoring explicit radial-bump center keys; each spin edit '
            'writes only its own key.  Unchecked drops the keys (the renderer '
            'uses the frame center).'
        )
        gen_layout.addRow(self._stray_center_check)
        default_center_v, default_center_u = self._stray_center_defaults()

        self._stray_center_v_spin = QDoubleSpinBox()
        self._stray_center_v_spin.setRange(-10000.0, 20000.0)
        self._stray_center_v_spin.setDecimals(1)
        self._stray_center_v_spin.setValue(float(self._stray_value('center_v', default_center_v)))
        self._stray_center_v_spin.setEnabled(has_center)
        self._stray_center_v_spin.setToolTip(
            'Radial-model bump center V as a pixel corner; absent = frame center.'
        )
        self._stray_center_v_spin.valueChanged.connect(self._on_stray_center_v)
        gen_layout.addRow('Stray light center V (pixel corner):', self._stray_center_v_spin)

        self._stray_center_u_spin = QDoubleSpinBox()
        self._stray_center_u_spin.setRange(-10000.0, 20000.0)
        self._stray_center_u_spin.setDecimals(1)
        self._stray_center_u_spin.setValue(float(self._stray_value('center_u', default_center_u)))
        self._stray_center_u_spin.setEnabled(has_center)
        self._stray_center_u_spin.setToolTip(
            'Radial-model bump center U as a pixel corner; absent = frame center.'
        )
        self._stray_center_u_spin.valueChanged.connect(self._on_stray_center_u)
        gen_layout.addRow('Stray light center U (pixel corner):', self._stray_center_u_spin)

        self._stray_center_check.toggled.connect(self._on_stray_center_check)

    def _stray_has_center(self) -> bool:
        """True when the scene authors either radial-bump center key."""
        optics = self.sim_params.get('optics')
        stray = optics.get('stray_light') if isinstance(optics, dict) else None
        if not isinstance(stray, dict):
            return False
        return 'center_v' in stray or 'center_u' in stray

    def _stray_center_defaults(self) -> tuple[float, float]:
        """The effective bump-center default: the frame center, in pixel corner."""
        return (
            float(self.sim_params.get('size_v', 512)) / 2.0,
            float(self.sim_params.get('size_u', 512)) / 2.0,
        )

    def _stray_value(self, key: str, default: Any) -> Any:
        """Read a value from the optics.stray_light block, or a default."""
        optics = self.sim_params.get('optics')
        stray = optics.get('stray_light') if isinstance(optics, dict) else None
        if isinstance(stray, dict) and key in stray:
            return stray[key]
        return default

    def _set_stray(self, key: str, value: Any) -> None:
        """Write a value into the optics.stray_light block and re-render."""
        if self._syncing:
            return
        optics = self.sim_params.setdefault('optics', {})
        if not isinstance(optics, dict):
            optics = {}
            self.sim_params['optics'] = optics
        stray = optics.setdefault('stray_light', {})
        if not isinstance(stray, dict):
            stray = {}
            optics['stray_light'] = stray
        stray[key] = value
        self._updater.request_update()

    def _on_stray_amplitude(self, value: float) -> None:
        """Set the stray-light amplitude."""
        self._set_stray('amplitude', float(value))

    def _on_stray_direction(self, value: float) -> None:
        """Set the linear-model gradient direction."""
        self._set_stray('direction_deg', float(value))

    def _on_stray_model(self, text: str) -> None:
        """Set the stray-light model (linear or radial)."""
        self._set_stray('model', text or 'linear')

    def _on_stray_center(self, key: str, value: float) -> None:
        """Write one radial-model bump center coordinate, when the enable is on."""
        if self._stray_center_check.isChecked():
            self._set_stray(key, float(value))

    def _on_stray_center_check(self, checked: bool) -> None:
        """Enable the bump-center spins; unchecking drops the center keys.

        Checking writes nothing by itself: an absent center key stays absent (the
        frame center keeps applying) until its own spin is edited.
        """
        self._stray_center_v_spin.setEnabled(checked)
        self._stray_center_u_spin.setEnabled(checked)
        if self._syncing or checked:
            return
        optics = self.sim_params.get('optics')
        stray = optics.get('stray_light') if isinstance(optics, dict) else None
        if isinstance(stray, dict) and ('center_v' in stray or 'center_u' in stray):
            stray.pop('center_v', None)
            stray.pop('center_u', None)
            self._updater.request_update()
        # The keys are now absent, so show the effective default.  The spins are
        # disabled and their handler checks the enable, so this never writes the
        # keys back.
        default_center_v, default_center_u = self._stray_center_defaults()
        self._stray_center_v_spin.setValue(default_center_v)
        self._stray_center_u_spin.setValue(default_center_u)

    def _on_stray_center_v(self, value: float) -> None:
        """Write the radial-model bump center V on a spin edit, when enabled."""
        self._on_stray_center('center_v', value)

    def _on_stray_center_u(self, value: float) -> None:
        """Write the radial-model bump center U on a spin edit, when enabled."""
        self._on_stray_center('center_u', value)
