"""Detector-noise panel for the General tab.

Poisson shot noise, a Gaussian read-noise floor, and the cosmic-ray rate.  These write
the ``sim_params['noise']`` block the renderer consumes; physical defaults otherwise
come from the selected instrument.

The validator accepts every control on every instrument and detector model, but
some change nothing on some detector paths.  On the vidicon detector (``vgiss``,
or a detector model of ``vidicon``) the Poisson and read-noise controls change
nothing, because the vidicon's DN-domain read noise comes from the ``noise.vidicon``
keys.  The cosmic-ray control works everywhere.
"""

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout

from spindoctor.ui.sim_editor.base import SimEditorBase

_COSMIC_RAY_TOOLTIP = 'Expected cosmic-ray events per pixel per second of exposure.'


class NoiseMixin(SimEditorBase):
    """Builds and handles the detector-noise panel."""

    def _build_noise_panel(self, gen_layout: QFormLayout) -> None:
        """Add the detector-noise rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        # The controls that change nothing on some detector paths (see the module
        # docstring) are tracked as #804.
        self._poisson_check = QCheckBox()
        self._poisson_check.setToolTip('Signal-dependent Poisson shot noise (usually on).')
        self._poisson_check.toggled.connect(self._on_poisson)
        gen_layout.addRow('Poisson shot noise:', self._poisson_check)

        self._read_noise_spin = QDoubleSpinBox()
        self._read_noise_spin.setRange(0.0, 50.0)
        self._read_noise_spin.setDecimals(2)
        self._read_noise_spin.setToolTip('Gaussian read-noise floor in DN.')
        self._read_noise_spin.valueChanged.connect(self._on_read_noise)
        gen_layout.addRow('Read noise (DN):', self._read_noise_spin)

        self._cosmic_ray_spin = QDoubleSpinBox()
        self._cosmic_ray_spin.setRange(0.0, 0.01)
        self._cosmic_ray_spin.setDecimals(5)
        self._cosmic_ray_spin.setSingleStep(0.0001)
        self._cosmic_ray_spin.setToolTip(_COSMIC_RAY_TOOLTIP)
        self._cosmic_ray_spin.valueChanged.connect(self._on_cosmic_ray)
        gen_layout.addRow('Cosmic rays (/px/s):', self._cosmic_ray_spin)
        self._sync_noise_from_params()

    def _sync_noise_from_params(self) -> None:
        """Set every detector-noise control from the scene's ``noise`` block.

        A key the block leaves out shows the control's default.  The
        controls' signals are blocked, so syncing writes nothing back to the scene.
        """
        values: tuple[tuple[QCheckBox | QDoubleSpinBox, Any], ...] = (
            (self._poisson_check, bool(self._noise_value('poisson', True))),
            (self._read_noise_spin, float(self._noise_value('read_noise_dn', 4.0))),
            (
                self._cosmic_ray_spin,
                float(self._noise_value('cosmic_ray_events_per_pixel_per_sec', 0.0)),
            ),
        )
        for widget, value in values:
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)

    def _noise_value(self, key: str, default: Any) -> Any:
        """Read a value from the sim_params noise block, or a default when it is absent."""
        noise = self.sim_params.get('noise')
        value = noise.get(key) if isinstance(noise, dict) else None
        return default if value is None else value

    def _set_noise(self, key: str, value: Any) -> None:
        """Write a value into the sim_params noise block and re-render."""
        noise = self.sim_params.setdefault('noise', {})
        if not isinstance(noise, dict):
            noise = {}
            self.sim_params['noise'] = noise
        noise[key] = value
        self._updater.request_update()

    def _on_poisson(self, checked: bool) -> None:
        """Toggle Poisson shot noise."""
        self._set_noise('poisson', bool(checked))

    def _on_read_noise(self, value: float) -> None:
        """Set the read-noise floor in DN."""
        self._set_noise('read_noise_dn', float(value))

    def _on_cosmic_ray(self, value: float) -> None:
        """Set the expected cosmic-ray events per pixel per second."""
        self._set_noise('cosmic_ray_events_per_pixel_per_sec', float(value))
