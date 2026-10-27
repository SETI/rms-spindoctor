"""Scene file I/O glue: save, load, and apply a validated scene dict.

Wraps :func:`spindoctor.sim.scene.save_sim_scene` /
:func:`spindoctor.sim.scene.load_sim_scene` behind file dialogs and rebuilds
the whole editor state from a loaded ``sim_params`` mapping, syncing every
General-tab widget and rebuilding the object tabs.
"""

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.scene import load_sim_scene, save_sim_scene


class SceneIoMixin(SimEditorBase):
    """Save / load scenes and apply a loaded scene dict to the editor."""

    def _save_scene(self) -> None:
        """Prompt for a path and write the current scene as YAML."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Scene',
            'scene.yaml',
            'YAML Files (*.yaml *.yml)',
        )
        if not filename:
            return
        try:
            save_sim_scene(self.sim_params, Path(filename))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save scene:\n{e!s}')

    def _load_scene(self) -> None:
        """Prompt for a path and load a scene YAML into the editor."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Scene',
            '',
            'YAML Files (*.yaml *.yml)',
        )
        if not filename:
            return
        self.load_scene_file(Path(filename))

    def load_scene_file(self, path: Path) -> None:
        """Load a scene YAML into the editor, reporting failures in a dialog.

        The shared loader behind the Load Scene button and the command-line
        scene argument: a scene that fails to parse or validate raises an
        error dialog and leaves the editor's current state untouched.

        Parameters:
            path: Path to the ``<scene_name>.yaml`` file to load.
        """
        try:
            self._apply_params_dict(load_sim_scene(path))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load scene:\n{e!s}')

    def _apply_params_dict(self, params: dict[str, Any]) -> None:
        """Rebuild sim_params from a loaded params dict and sync every widget."""
        self.sim_params = {
            'size_v': int(params.get('size_v', 512)),
            'size_u': int(params.get('size_u', 512)),
            'offset_v': float(params.get('offset_v', 0.0)),
            'offset_u': float(params.get('offset_u', 0.0)),
            'offset_rotation_deg': float(params.get('offset_rotation_deg', 0.0)),
            'exposure_sec': float(params.get('exposure_sec', 1.0)),
            'random_seed': int(params.get('random_seed', 42)),
            'instrument': str(params.get('instrument', 'generic')),
            'time': float(params.get('time', 0.0)),
            'ring_epoch': float(params.get('ring_epoch', 0.0)),
            'closest_planet': params.get('closest_planet') or 'SATURN',
            'bodies': list(params.get('bodies', [])),
            'stars': list(params.get('stars', [])),
        }
        # Carry the block-valued schema keys through unchanged; the tab-specific
        # sync methods below then drive their widgets from these blocks (and the
        # instrument-config / midtime / fit-rotation keys the General tab reads).
        # sky_counts follows the absent-key discipline: a scene without the key
        # must not gain one, so it passes through rather than being defaulted.
        for passthrough_key in (
            'ring_system',
            'noise',
            'optics',
            'spk_error',
            'oversample',
            'detector',
            'artifacts',
            'instrument_config',
            'midtime_utc',
            'fit_camera_rotation',
            'sky_counts',
            'star_catalog_scatter_px',
            'expected',
        ):
            if passthrough_key in params:
                self.sim_params[passthrough_key] = params[passthrough_key]
        # Update general UI
        self._size_v_spin.setValue(self.sim_params['size_v'])
        self._size_u_spin.setValue(self.sim_params['size_u'])
        self._offset_v_spin.setValue(self.sim_params['offset_v'])
        self._offset_u_spin.setValue(self.sim_params['offset_u'])
        self._offset_rotation_spin.blockSignals(True)
        self._offset_rotation_spin.setValue(self.sim_params['offset_rotation_deg'])
        self._offset_rotation_spin.blockSignals(False)
        self._exposure_spin.blockSignals(True)
        self._exposure_spin.setValue(float(self.sim_params['exposure_sec']))
        self._exposure_spin.blockSignals(False)
        self._random_seed_spin.setValue(self.sim_params['random_seed'])
        # Update instrument selector
        self._instrument_combo.blockSignals(True)
        instrument_index = self._instrument_combo.findText(str(self.sim_params['instrument']))
        if instrument_index >= 0:
            self._instrument_combo.setCurrentIndex(instrument_index)
        self._instrument_combo.blockSignals(False)
        # Camera-rotation fit override and midtime
        fit_rotation = self.sim_params.get('fit_camera_rotation')
        fit_text = 'on' if fit_rotation is True else 'off' if fit_rotation is False else '(inherit)'
        self._fit_rotation_combo.blockSignals(True)
        fit_index = self._fit_rotation_combo.findText(fit_text)
        if fit_index >= 0:
            self._fit_rotation_combo.setCurrentIndex(fit_index)
        self._fit_rotation_combo.blockSignals(False)
        self._midtime_edit.blockSignals(True)
        self._midtime_edit.setText(str(self.sim_params.get('midtime_utc') or ''))
        self._midtime_edit.blockSignals(False)
        self._update_psf_preview()
        # Update time and epoch
        self._time_spin.setValue(self.sim_params.get('time', 0.0))
        self._epoch_spin.setValue(self.sim_params.get('ring_epoch', 0.0))
        # Update closest planet
        closest_planet = self.sim_params.get('closest_planet') or 'SATURN'
        index = self._closest_planet_combo.findText(closest_planet)
        if index >= 0:
            self._closest_planet_combo.setCurrentIndex(index)
        else:
            self._closest_planet_combo.setCurrentText(closest_planet)
        # Update detector-noise controls from the loaded noise block.
        self._poisson_check.blockSignals(True)
        self._poisson_check.setChecked(bool(self._noise_value('poisson', True)))
        self._poisson_check.blockSignals(False)
        self._read_noise_spin.blockSignals(True)
        self._read_noise_spin.setValue(float(self._noise_value('read_noise_dn', 4.0)))
        self._read_noise_spin.blockSignals(False)
        self._cosmic_ray_spin.blockSignals(True)
        self._cosmic_ray_spin.setValue(
            float(self._noise_value('cosmic_ray_events_per_pixel_per_sec', 0.0))
        )
        self._cosmic_ray_spin.blockSignals(False)
        self._missing_data_spin.blockSignals(True)
        self._missing_data_spin.setValue(float(self._noise_value('missing_data_rate', 0.0)))
        self._missing_data_spin.blockSignals(False)
        self._bias_spin.blockSignals(True)
        self._bias_spin.setValue(float(self._noise_value('bias_dn', 20.0)))
        self._bias_spin.blockSignals(False)
        self._bloom_spin.blockSignals(True)
        self._bloom_spin.setValue(int(self._noise_value('bloom_length', 0)))
        self._bloom_spin.blockSignals(False)
        self._signal_frac_spin.blockSignals(True)
        self._signal_frac_spin.setValue(float(self._noise_value('signal_full_scale_frac', 0.5)))
        self._signal_frac_spin.blockSignals(False)
        # The Optics-tab controls (PSF, smear, distortion, ghosts, stray
        # light, oversample, spk_error) and the Artifacts-tab controls
        # (instrument defaults, detector override) sync from their own blocks.
        self._sync_optics_from_params()
        self._sync_artifacts_from_params()
        # Update background-sky (sky_counts) controls.  Absent-key discipline:
        # a scene without the block leaves the enable checkbox unchecked (and
        # the value widgets disabled, showing the would-be defaults).
        has_sky = isinstance(self.sim_params.get('sky_counts'), dict)
        sky = self.sim_params.get('sky_counts') or {}
        self._sky_counts_check.blockSignals(True)
        self._sky_counts_check.setChecked(has_sky)
        self._sky_counts_check.blockSignals(False)
        for widget, value in (
            (self._sky_density_spin, float(sky.get('density_factor', 0.0))),
            (self._sky_a_spin, float(sky.get('a', -3.1))),
            (self._sky_b_spin, float(sky.get('b', 0.34))),
            (self._sky_diffuse_spin, float(sky.get('diffuse_e_per_px', 0.0))),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._sky_density_slider.blockSignals(True)
        self._sky_density_slider.setValue(int(float(sky.get('density_factor', 0.0)) * 10))
        self._sky_density_slider.blockSignals(False)
        self._set_sky_widgets_enabled(has_sky)
        # Sync the scene-level star-catalog-scatter control.
        has_scatter = self.sim_params.get('star_catalog_scatter_px') is not None
        self._star_scatter_check.blockSignals(True)
        self._star_scatter_check.setChecked(has_scatter)
        self._star_scatter_check.blockSignals(False)
        self._star_scatter_spin.blockSignals(True)
        self._star_scatter_spin.setValue(float(self.sim_params.get('star_catalog_scatter_px', 0.0)))
        self._star_scatter_spin.setEnabled(has_scatter)
        self._star_scatter_spin.blockSignals(False)
        # Sync the test-only expected-outcome block.
        expected = self.sim_params.get('expected')
        has_expected = isinstance(expected, dict)
        block: dict[str, Any] = expected if isinstance(expected, dict) else {}
        self._expected_group.blockSignals(True)
        self._expected_group.setChecked(has_expected)
        self._expected_group.blockSignals(False)
        self._expected_status_combo.blockSignals(True)
        status_index = self._expected_status_combo.findText(str(block.get('status', 'success')))
        if status_index >= 0:
            self._expected_status_combo.setCurrentIndex(status_index)
        self._expected_status_combo.blockSignals(False)
        self._expected_tier_combo.blockSignals(True)
        tier = block.get('confidence_tier')
        tier_index = self._expected_tier_combo.findText('(none)' if tier is None else str(tier))
        if tier_index >= 0:
            self._expected_tier_combo.setCurrentIndex(tier_index)
        self._expected_tier_combo.blockSignals(False)
        self._expected_reason_edit.blockSignals(True)
        self._expected_reason_edit.setText(str(block.get('status_reason') or ''))
        self._expected_reason_edit.blockSignals(False)
        # Rebuild tabs
        self._rebuild_dynamic_tabs()
        self._update_tab_titles()
        self._validate_ranges()
        self._updater.immediate_update()
