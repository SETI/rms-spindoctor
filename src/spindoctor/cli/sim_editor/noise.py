"""Detector-noise panel for the General tab.

Poisson shot noise, a Gaussian read-noise floor, cosmic-ray rate, missing-data
rate, bias pedestal, saturation bloom, and signal full-scale fraction.  These
write the ``sim_params['noise']`` block the renderer consumes;
physical defaults otherwise come from the selected instrument.

The validator accepts every control on every instrument and detector model, but
some change nothing on some detector paths.  On the vidicon detector (``vgiss``,
or a detector model of ``vidicon``) the Poisson, read-noise, and bloom controls
change nothing: the vidicon's DN-domain noise comes from the ``noise.vidicon``
keys, and it has no electron well to bloom.  On a calibrated frame the bias
control changes nothing on the Cassini cameras and only the pixels near the
8-bit ceiling on ``vgiss``, because the calibration inverse subtracts the bias
again.  The cosmic-ray and missing-data controls work everywhere.
"""

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

from spindoctor.cli.sim_editor.base import SimEditorBase

_COSMIC_RAY_TOOLTIP = 'Expected cosmic-ray events per pixel per second of exposure.'
_MISSING_DATA_TOOLTIP = (
    "Fraction of pixels lost at random, each set to the instrument's missing-data "
    'marker (0 on every shipped instrument).'
)


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
        self._poisson_check.setChecked(bool(self._noise_value('poisson', True)))
        self._poisson_check.setToolTip('Signal-dependent Poisson shot noise (usually on).')
        self._poisson_check.toggled.connect(self._on_poisson)
        gen_layout.addRow('Poisson shot noise:', self._poisson_check)

        self._read_noise_spin = QDoubleSpinBox()
        self._read_noise_spin.setRange(0.0, 50.0)
        self._read_noise_spin.setDecimals(2)
        self._read_noise_spin.setValue(float(self._noise_value('read_noise_dn', 4.0)))
        self._read_noise_spin.setToolTip('Gaussian read-noise floor in DN.')
        self._read_noise_spin.valueChanged.connect(self._on_read_noise)
        gen_layout.addRow('Read noise (DN):', self._read_noise_spin)

        self._cosmic_ray_spin = QDoubleSpinBox()
        self._cosmic_ray_spin.setRange(0.0, 0.01)
        self._cosmic_ray_spin.setDecimals(5)
        self._cosmic_ray_spin.setSingleStep(0.0001)
        self._cosmic_ray_spin.setValue(
            float(self._noise_value('cosmic_ray_events_per_pixel_per_sec', 0.0))
        )
        self._cosmic_ray_spin.setToolTip(_COSMIC_RAY_TOOLTIP)
        self._cosmic_ray_spin.valueChanged.connect(self._on_cosmic_ray)
        gen_layout.addRow('Cosmic rays (/px/s):', self._cosmic_ray_spin)

        self._missing_data_spin = QDoubleSpinBox()
        self._missing_data_spin.setRange(0.0, 0.3)
        self._missing_data_spin.setDecimals(3)
        self._missing_data_spin.setSingleStep(0.005)
        self._missing_data_spin.setValue(float(self._noise_value('missing_data_rate', 0.0)))
        self._missing_data_spin.setToolTip(_MISSING_DATA_TOOLTIP)
        self._missing_data_spin.valueChanged.connect(self._on_missing_data)
        gen_layout.addRow('Missing data rate:', self._missing_data_spin)

        self._bias_spin = QDoubleSpinBox()
        self._bias_spin.setRange(0.0, 10000.0)
        self._bias_spin.setDecimals(2)
        self._bias_spin.setValue(float(self._noise_value('bias_dn', 20.0)))
        self._bias_spin.setToolTip('Additive bias pedestal in DN (lifts dark sky off zero).')
        self._bias_spin.valueChanged.connect(self._on_bias)
        gen_layout.addRow('Bias (DN):', self._bias_spin)

        self._bloom_spin = QSpinBox()
        self._bloom_spin.setRange(0, 200)
        self._bloom_spin.setValue(int(self._noise_value('bloom_length', 0)))
        self._bloom_spin.setToolTip('Saturation column-bloom half-length in pixels (0 = none).')
        self._bloom_spin.valueChanged.connect(self._on_bloom)
        gen_layout.addRow('Bloom length (px):', self._bloom_spin)

        self._signal_frac_spin = QDoubleSpinBox()
        self._signal_frac_spin.setRange(0.001, 1.0)
        self._signal_frac_spin.setDecimals(3)
        self._signal_frac_spin.setSingleStep(0.05)
        self._signal_frac_spin.setValue(float(self._noise_value('signal_full_scale_frac', 0.5)))
        self._signal_frac_spin.setToolTip(
            'Signal of 1.0 maps to this fraction of the camera full well.'
        )
        self._signal_frac_spin.valueChanged.connect(self._on_signal_frac)
        gen_layout.addRow('Signal full-scale frac:', self._signal_frac_spin)

    def _noise_value(self, key: str, default: Any) -> Any:
        """Read a value from the sim_params noise block, or a default.

        A key set to null reads as absent, as the renderer reads it.
        """
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

    def _on_missing_data(self, value: float) -> None:
        """Set the missing-data pixel fraction."""
        self._set_noise('missing_data_rate', float(value))

    def _on_bias(self, value: float) -> None:
        """Set the additive bias pedestal in DN."""
        self._set_noise('bias_dn', float(value))

    def _on_bloom(self, value: int) -> None:
        """Set the saturation column-bloom half-length."""
        self._set_noise('bloom_length', int(value))

    def _on_signal_frac(self, value: float) -> None:
        """Set the signal full-scale fraction."""
        self._set_noise('signal_full_scale_frac', float(value))
