"""Global scene fields for the General tab.

Image size, planted offset and camera roll, exposure, seed, the instrument
selector, the camera-rotation-fit override, midtime, closest planet, and the
ring timing fields.  ``GlobalFieldsMixin`` builds these rows and owns their
change handlers.
"""

from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.instruments import SIM_INSTRUMENTS

# Instrument options for the General-tab selector: the generic (instrument-
# agnostic) frame plus every per-instrument sim camera.
_INSTRUMENT_CHOICES: list[str] = ['generic', *sorted(SIM_INSTRUMENTS)]


class GlobalFieldsMixin(SimEditorBase):
    """Builds and handles the General tab's global scene fields."""

    def _build_global_fields(self, gen_layout: QFormLayout) -> None:
        """Add the global scene-field rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        self._size_v_spin = QSpinBox()
        self._size_v_spin.setRange(10, 9999)
        self._size_v_spin.setValue(self.sim_params['size_v'])
        self._size_v_spin.valueChanged.connect(self._on_size_v)
        gen_layout.addRow('Size V (height):', self._size_v_spin)

        self._size_u_spin = QSpinBox()
        self._size_u_spin.setRange(10, 9999)
        self._size_u_spin.setValue(self.sim_params['size_u'])
        self._size_u_spin.valueChanged.connect(self._on_size_u)
        gen_layout.addRow('Size U (width):', self._size_u_spin)

        self._offset_v_spin = QDoubleSpinBox()
        self._offset_v_spin.setRange(-10000.0, 10000.0)
        self._offset_v_spin.setDecimals(3)
        self._offset_v_spin.setValue(self.sim_params['offset_v'])
        self._offset_v_spin.setToolTip(
            'Offsets are saved in the model but not shown in the preview.'
        )
        self._offset_v_spin.valueChanged.connect(self._on_offset_v)
        gen_layout.addRow('Offset V:', self._offset_v_spin)

        self._offset_u_spin = QDoubleSpinBox()
        self._offset_u_spin.setRange(-10000.0, 10000.0)
        self._offset_u_spin.setDecimals(3)
        self._offset_u_spin.setValue(self.sim_params['offset_u'])
        self._offset_u_spin.setToolTip(
            'Offsets are saved in the model but not shown in the preview.'
        )
        self._offset_u_spin.valueChanged.connect(self._on_offset_u)
        gen_layout.addRow('Offset U:', self._offset_u_spin)

        self._offset_rotation_spin = QDoubleSpinBox()
        self._offset_rotation_spin.setRange(-180.0, 180.0)
        self._offset_rotation_spin.setDecimals(3)
        self._offset_rotation_spin.setValue(self.sim_params['offset_rotation_deg'])
        self._offset_rotation_spin.setToolTip(
            'Planted camera roll (deg) about the boresight; recovered by navigation.'
        )
        self._offset_rotation_spin.valueChanged.connect(self._on_offset_rotation)
        gen_layout.addRow('Offset rotation (deg):', self._offset_rotation_spin)

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(0.001, 100000.0)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setValue(float(self.sim_params['exposure_sec']))
        self._exposure_spin.setToolTip('Exposure time (sec); scales the cosmic-ray count.')
        self._exposure_spin.valueChanged.connect(self._on_exposure)
        gen_layout.addRow('Exposure (sec):', self._exposure_spin)

        # Random seed
        self._random_seed_spin = QSpinBox()
        self._random_seed_spin.setRange(0, 2147483647)
        self._random_seed_spin.setValue(self.sim_params['random_seed'])
        self._random_seed_spin.valueChanged.connect(self._on_random_seed)
        gen_layout.addRow('Random seed:', self._random_seed_spin)

        # Instrument selector: drives the per-instrument noise / saturation /
        # PSF / unit settings the renderer applies (see spindoctor.sim.instruments).
        self._instrument_combo = QComboBox()
        self._instrument_combo.addItems(_INSTRUMENT_CHOICES)
        instrument = str(self.sim_params.get('instrument', 'generic'))
        instrument_index = self._instrument_combo.findText(instrument)
        if instrument_index >= 0:
            self._instrument_combo.setCurrentIndex(instrument_index)
        self._instrument_combo.setToolTip(
            'Camera the sim emulates; sets noise, saturation, PSF, and units.'
        )
        self._instrument_combo.currentTextChanged.connect(self._on_instrument)
        gen_layout.addRow('Instrument:', self._instrument_combo)

        # Camera-rotation fit override: blank inherits the instrument default,
        # on/off force whether navigation solves for a camera roll on this scene.
        self._fit_rotation_combo = QComboBox()
        self._fit_rotation_combo.addItems(['(inherit)', 'on', 'off'])
        self._fit_rotation_combo.setToolTip(
            'Force whether navigation fits a camera roll; (inherit) uses the instrument default.'
        )
        self._fit_rotation_combo.currentTextChanged.connect(self._on_fit_rotation)
        gen_layout.addRow('Fit camera rotation:', self._fit_rotation_combo)

        # Midtime (informational ISO timestamp carried on the scene).
        self._midtime_edit = QLineEdit(str(self.sim_params.get('midtime_utc') or ''))
        self._midtime_edit.setToolTip('Optional ISO UTC timestamp recorded on the scene.')
        self._midtime_edit.textChanged.connect(self._on_midtime)
        gen_layout.addRow('Midtime UTC:', self._midtime_edit)

        # Closest planet (for ring models)
        self._closest_planet_combo = QComboBox()
        self._closest_planet_combo.setEditable(True)
        self._closest_planet_combo.addItems(
            [
                'MERCURY',
                'VENUS',
                'EARTH',
                'MARS',
                'JUPITER',
                'SATURN',
                'URANUS',
                'NEPTUNE',
                'PLUTO',
            ]
        )
        closest_planet = self.sim_params.get('closest_planet', 'SATURN')
        if closest_planet:
            index = self._closest_planet_combo.findText(closest_planet)
            if index >= 0:
                self._closest_planet_combo.setCurrentIndex(index)
            else:
                self._closest_planet_combo.setCurrentText(closest_planet)
        self._closest_planet_combo.currentTextChanged.connect(self._on_closest_planet)
        gen_layout.addRow('Closest planet:', self._closest_planet_combo)

        # Time (TDB seconds)
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(-1e10, 1e10)
        self._time_spin.setDecimals(1)
        self._time_spin.setValue(self.sim_params.get('time', 0.0))
        self._time_spin.setToolTip('Current time in TDB seconds for ring calculations')
        self._time_spin.valueChanged.connect(self._on_time)
        gen_layout.addRow('Time (TDB sec):', self._time_spin)

        # Ring epoch (TDB seconds)
        self._epoch_spin = QDoubleSpinBox()
        self._epoch_spin.setRange(-1e10, 1e10)
        self._epoch_spin.setDecimals(1)
        self._epoch_spin.setValue(self.sim_params.get('ring_epoch', 0.0))
        self._epoch_spin.setToolTip('Ring epoch time in TDB seconds for ring mode calculations')
        self._epoch_spin.valueChanged.connect(self._on_epoch)
        gen_layout.addRow('Ring Epoch (TDB sec):', self._epoch_spin)

    # ---- Sim param handlers ----
    def _on_size_v(self, value: int) -> None:
        """Update the image height."""
        self.sim_params['size_v'] = value
        self._updater.request_update()

    def _on_size_u(self, value: int) -> None:
        """Update the image width."""
        self.sim_params['size_u'] = value
        self._updater.request_update()

    def _on_offset_v(self, value: float) -> None:
        """Update the planted V offset."""
        self.sim_params['offset_v'] = value
        self._updater.request_update()

    def _on_offset_u(self, value: float) -> None:
        """Update the planted U offset."""
        self.sim_params['offset_u'] = value
        self._updater.request_update()

    def _on_offset_rotation(self, value: float) -> None:
        """Update the planted camera roll."""
        self.sim_params['offset_rotation_deg'] = float(value)
        self._updater.request_update()

    def _on_exposure(self, value: float) -> None:
        """Update the exposure time."""
        self.sim_params['exposure_sec'] = float(value)
        self._updater.request_update()

    def _on_fit_rotation(self, text: str) -> None:
        """Set, clear, or inherit the camera-rotation-fit override."""
        if text == 'on':
            self.sim_params['fit_camera_rotation'] = True
        elif text == 'off':
            self.sim_params['fit_camera_rotation'] = False
        else:
            self.sim_params.pop('fit_camera_rotation', None)
        self._updater.request_update()

    def _on_midtime(self, text: str) -> None:
        """Set or clear the informational midtime timestamp."""
        if text.strip():
            self.sim_params['midtime_utc'] = text.strip()
        else:
            self.sim_params.pop('midtime_utc', None)

    def _on_random_seed(self, value: int) -> None:
        """Update the render RNG seed."""
        self.sim_params['random_seed'] = value
        self._updater.request_update()

    def _on_instrument(self, text: str) -> None:
        """Update the emulated instrument and refresh the PSF preview."""
        self.sim_params['instrument'] = text or 'generic'
        self._update_psf_preview()
        # The Artifacts tab displays the instrument's detector catalog
        # defaults for keys the scene does not override, and its mode rows
        # enable or disable by the new instrument's availability.
        self._refresh_detector_catalog_defaults()
        self._refresh_artifact_mode_availability()
        self._updater.request_update()

    def _on_closest_planet(self, text: str) -> None:
        """Update the closest-planet ring reference (uppercased, or None)."""
        # Store as None if empty, otherwise store the text (uppercase)
        if text.strip():
            self.sim_params['closest_planet'] = text.strip().upper()
        else:
            self.sim_params['closest_planet'] = None
        self._updater.request_update()

    def _on_time(self, value: float) -> None:
        """Update the scene time in TDB seconds."""
        self.sim_params['time'] = value
        self._updater.request_update()

    def _on_epoch(self, value: float) -> None:
        """Update the ring epoch in TDB seconds."""
        self.sim_params['ring_epoch'] = value
        self._updater.request_update()
