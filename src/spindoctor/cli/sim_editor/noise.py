"""Detector-noise panel for the General tab.

Poisson shot noise, a Gaussian read-noise floor, cosmic-ray rate, missing-data
rate, bias pedestal, saturation bloom, and signal full-scale fraction.  These
write the ``sim_params['noise']`` block the renderer consumes;
physical defaults otherwise come from the selected instrument.

The scene validator refuses a positive cosmic-ray rate on the vidicon detector
and a positive missing-data rate on a calibrated frame.  The panel reads the
validator's own predicates (:func:`~spindoctor.sim.scene_checks.cosmic_ray_rate_refusal`
and :func:`~spindoctor.sim.scene_checks.missing_data_rate_refusal`): where one
refuses, its control is disabled with the reason as the tooltip.  A change of
instrument, detector model, or detector override that makes a set rate refused
moves the rate out of the scene into a stash, with a status-bar message, and a
change that lifts the refusal puts it back.
"""

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.scene_checks import cosmic_ray_rate_refusal, missing_data_rate_refusal

_COSMIC_RAY_KEY = 'cosmic_ray_events_per_pixel_per_sec'
_MISSING_DATA_KEY = 'missing_data_rate'

# Each control's own tooltip, shown while the scene's detector path accepts a
# positive value.
_COSMIC_RAY_TOOLTIP = (
    'Expected cosmic-ray events per pixel per second of exposure. CCD chain only: '
    'on the vidicon detector (vgiss, or detector model vidicon) this control is '
    'disabled, because validation refuses a non-zero rate there.'
)
_MISSING_DATA_TOOLTIP = (
    'Fraction of pixels marked as missing data. Raw-DN frames only: on a calibrated '
    'instrument (coiss_calib_nac, coiss_calib_wac, vgiss) this control is disabled, '
    'because validation refuses a non-zero rate there.'
)

# How long the status bar shows the stash message, in milliseconds.
NOISE_STASH_MESSAGE_MS = 15000


def noise_refusal_tooltip(reason: str) -> str:
    """The tooltip of a noise control the scene's detector path refuses.

    Parameters:
        reason: The validator's reason, from
            :func:`~spindoctor.sim.scene_checks.cosmic_ray_rate_refusal` or
            :func:`~spindoctor.sim.scene_checks.missing_data_rate_refusal`.

    Returns:
        ``Validation refuses a non-zero rate here: <reason>.``
    """
    return f'Validation refuses a non-zero rate here: {reason}.'


def noise_stash_message(keys: list[str]) -> str:
    """The status-bar sentence shown when a switch moves noise rates out of the scene.

    Parameters:
        keys: The ``noise`` keys the switch moved, at least one.

    Returns:
        A sentence naming the keys, saying that validation refuses them on the
        scene's detector path, that switching back restores them in this editor
        session, and that a saved scene file doesn't carry them.
    """
    names = ', '.join(f'noise.{key}' for key in keys)
    return (
        f'Moved {names} out of the scene, because validation refuses a non-zero rate '
        'on this detector path (hover the disabled control for why). Switching back '
        "restores it in this session; a saved scene file doesn't carry it."
    )


class NoiseMixin(SimEditorBase):
    """Builds and handles the detector-noise panel."""

    def _build_noise_panel(self, gen_layout: QFormLayout) -> None:
        """Add the detector-noise rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
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
        self._sync_noise_refusals()

    def _refusable_noise_controls(self) -> list[tuple[str, QDoubleSpinBox, str, str | None]]:
        """Each noise key a detector path can refuse, with its control and the verdict.

        Returns:
            ``(key, spin, own tooltip, reason)`` per key, where ``reason`` is the
            validator's refusal of a positive value in the current scene, or None
            when the scene accepts one.
        """
        instrument = str(self.sim_params.get('instrument') or 'generic')
        return [
            (
                _COSMIC_RAY_KEY,
                self._cosmic_ray_spin,
                _COSMIC_RAY_TOOLTIP,
                cosmic_ray_rate_refusal(self.sim_params.get('detector'), instrument=instrument),
            ),
            (
                _MISSING_DATA_KEY,
                self._missing_data_spin,
                _MISSING_DATA_TOOLTIP,
                missing_data_rate_refusal(
                    self.sim_params.get('instrument_config'), instrument=instrument
                ),
            ),
        ]

    def _sync_noise_refusals(self) -> None:
        """Gate the cosmic-ray and missing-data controls on the validator's rules.

        Call after any change to the instrument, ``detector.detector_model`` (or
        the detector override as a whole), or ``instrument_config``.  A control
        whose rate the scene's detector path refuses is disabled, with the
        validator's reason as its tooltip (:func:`noise_refusal_tooltip`); a
        positive rate the scene carries there moves into the editor's noise stash
        and the status bar names it (:func:`noise_stash_message`).  Where the path
        accepts the rate again, the control is enabled with its own tooltip and a
        stashed rate goes back into the scene, unless the scene set the key again.
        Each control then shows the scene's value.
        """
        moved: list[str] = []
        changed = False
        for key, spin, own_tooltip, reason in self._refusable_noise_controls():
            spin.setEnabled(reason is None)
            spin.setToolTip(own_tooltip if reason is None else noise_refusal_tooltip(reason))
            noise = self.sim_params.get('noise')
            if reason is not None:
                value = noise.get(key) if isinstance(noise, dict) else None
                if isinstance(noise, dict) and value is not None and float(value) > 0.0:
                    self._noise_key_stash[key] = noise.pop(key)
                    moved.append(key)
            elif key in self._noise_key_stash:
                stashed = self._noise_key_stash.pop(key)
                if not isinstance(noise, dict):
                    noise = {}
                    self.sim_params['noise'] = noise
                if noise.get(key) is None:
                    noise[key] = stashed
                    changed = True
            spin.blockSignals(True)
            spin.setValue(float(self._noise_value(key, 0.0)))
            spin.blockSignals(False)
        if moved:
            status_bar = self.statusBar()
            if status_bar is not None:
                status_bar.showMessage(noise_stash_message(moved), NOISE_STASH_MESSAGE_MS)
        if moved or changed:
            self._updater.request_update()

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
