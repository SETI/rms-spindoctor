"""Optics tab for the simulated-image editor.

Authors the scene-level ``optics`` block and the two geometry knobs grouped with
it (``oversample`` and ``spk_error``).  Each optical sub-block is a checkable
group: unchecking it removes the key from ``sim_params`` entirely, so a disabled
sub-block is absent rather than zeroed and the absent-block semantics survive a
save / load round-trip.  The controls cover the whole-scene PSF (an explicit
core-plus-wing kernel or a navigator-matched form), the motion-smear entry list,
the residual distortion field, the ghost-reflection entry list, the stray-light
panel, the render oversampling factor, and the planted spacecraft-ephemeris
parallax error.  Every edit re-renders through the shared debounced updater.
"""

from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase

_SMEAR_CLASSES: list[str] = ['all', 'stars', 'bodies', 'rings']


@dataclass
class _SmearRow:
    """The widgets of one motion-smear entry row."""

    container: QWidget
    dv_spin: QDoubleSpinBox
    du_spin: QDoubleSpinBox
    class_combo: QComboBox


@dataclass
class _GhostRow:
    """The widgets of one ghost-reflection entry row."""

    container: QWidget
    dv_spin: QDoubleSpinBox
    du_spin: QDoubleSpinBox
    amplitude_spin: QDoubleSpinBox
    defocus_spin: QDoubleSpinBox


def _dspin(
    *,
    minimum: float,
    maximum: float,
    decimals: int,
    step: float,
    value: float,
    tooltip: str = '',
) -> QDoubleSpinBox:
    """Build a configured ``QDoubleSpinBox``."""
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


class OpticsTabMixin(SimEditorBase):
    """Builds and handles the Optics tab."""

    # ---- Optics block helpers (absent-key discipline) ----

    def _optics_map(self) -> dict[str, Any]:
        """Return the mutable ``optics`` sub-map, creating it if absent."""
        optics = self.sim_params.get('optics')
        if not isinstance(optics, dict):
            optics = {}
            self.sim_params['optics'] = optics
        return optics

    def _put_optics(self, key: str, value: Any) -> None:
        """Write one ``optics`` sub-block and request a re-render."""
        self._optics_map()[key] = value
        self._updater.request_update()

    def _drop_optics(self, key: str) -> None:
        """Remove one ``optics`` sub-block, pruning an empty ``optics`` map."""
        optics = self.sim_params.get('optics')
        if isinstance(optics, dict):
            optics.pop(key, None)
            if not optics:
                self.sim_params.pop('optics', None)
        self._updater.request_update()

    # ---- Tab construction ----

    def _build_optics_tab(self) -> QWidget:
        """Build the scrollable Optics tab and return its container widget."""
        self._smear_rows = []
        self._ghost_rows = []

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_psf_group())
        layout.addWidget(self._build_smear_group())
        layout.addWidget(self._build_distortion_group())
        layout.addWidget(self._build_ghosts_group())
        layout.addWidget(self._build_stray_group())
        layout.addWidget(self._build_scene_geometry_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self._optics_tab = scroll
        return scroll

    # ---- PSF group ----

    def _build_psf_group(self) -> QGroupBox:
        """Build the whole-scene PSF group (explicit kernel or navigator match)."""
        group = QGroupBox('Point-spread function (PSF)')
        group.setCheckable(True)
        form = QFormLayout(group)

        self._psf_match_nav_check = QCheckBox()
        self._psf_match_nav_check.setToolTip(
            'Match the navigator PSF exactly (a pure Gaussian at the instrument '
            'star_psf_sigma); disables the explicit kernel spins.'
        )
        form.addRow('Match navigator:', self._psf_match_nav_check)

        self._psf_sigma_v_spin = _dspin(
            minimum=0.01,
            maximum=20.0,
            decimals=3,
            step=0.05,
            value=0.55,
            tooltip='Gaussian core sigma along v, in detector pixels.',
        )
        form.addRow('Sigma V (px):', self._psf_sigma_v_spin)
        self._psf_sigma_u_spin = _dspin(
            minimum=0.01,
            maximum=20.0,
            decimals=3,
            step=0.05,
            value=0.55,
            tooltip='Gaussian core sigma along u, in detector pixels.',
        )
        form.addRow('Sigma U (px):', self._psf_sigma_u_spin)
        self._psf_w_spin = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=4,
            step=0.005,
            value=0.0,
            tooltip='Moffat wing energy fraction in [0, 1].',
        )
        form.addRow('Wing fraction w:', self._psf_w_spin)
        self._psf_r0_spin = _dspin(
            minimum=0.01,
            maximum=50.0,
            decimals=3,
            step=0.1,
            value=2.0,
            tooltip='Moffat core radius in detector pixels.',
        )
        form.addRow('Moffat r0 (px):', self._psf_r0_spin)
        self._psf_n_spin = _dspin(
            minimum=0.1,
            maximum=20.0,
            decimals=3,
            step=0.1,
            value=3.0,
            tooltip='Moffat index n.',
        )
        form.addRow('Moffat index n:', self._psf_n_spin)

        self._psf_optics_group = group
        psf = self._optics_sub('psf')
        group.setChecked(isinstance(psf, dict))
        self._apply_psf_match_state(bool(isinstance(psf, dict) and psf.get('match_navigator')))
        # Connect after the initial state so the build does not write.
        self._psf_match_nav_check.toggled.connect(self._on_psf_match_nav)
        for spin in (
            self._psf_sigma_v_spin,
            self._psf_sigma_u_spin,
            self._psf_w_spin,
            self._psf_r0_spin,
            self._psf_n_spin,
        ):
            spin.valueChanged.connect(self._on_psf_value)
        group.toggled.connect(self._on_psf_optics_group_toggled)
        return group

    def _optics_sub(self, key: str) -> Any:
        """Read one ``optics`` sub-block from sim_params, or None."""
        optics = self.sim_params.get('optics')
        if isinstance(optics, dict):
            return optics.get(key)
        return None

    def _apply_psf_match_state(self, match: bool) -> None:
        """Set the match-navigator checkbox and enable/disable the kernel spins."""
        self._psf_match_nav_check.setChecked(match)
        for spin in (
            self._psf_sigma_v_spin,
            self._psf_sigma_u_spin,
            self._psf_w_spin,
            self._psf_r0_spin,
            self._psf_n_spin,
        ):
            spin.setEnabled(not match)

    def _psf_block_from_widgets(self) -> dict[str, Any]:
        """Assemble the PSF block the widgets describe."""
        if self._psf_match_nav_check.isChecked():
            return {'match_navigator': True}
        return {
            'sigma_v': float(self._psf_sigma_v_spin.value()),
            'sigma_u': float(self._psf_sigma_u_spin.value()),
            'w': float(self._psf_w_spin.value()),
            'r0': float(self._psf_r0_spin.value()),
            'n': float(self._psf_n_spin.value()),
        }

    def _write_psf(self) -> None:
        """Write the PSF block when the group is enabled."""
        if self._syncing:
            return
        if self._psf_optics_group.isChecked():
            self._put_optics('psf', self._psf_block_from_widgets())

    def _on_psf_optics_group_toggled(self, on: bool) -> None:
        """Insert or remove the whole PSF block."""
        if self._syncing:
            return
        if on:
            self._put_optics('psf', self._psf_block_from_widgets())
        else:
            self._drop_optics('psf')

    def _on_psf_match_nav(self, checked: bool) -> None:
        """Toggle navigator matching, disabling the explicit kernel spins."""
        self._apply_psf_match_state(checked)
        self._write_psf()

    def _on_psf_value(self, _value: float) -> None:
        """Rewrite the PSF block from an explicit-kernel spin edit."""
        self._write_psf()

    # ---- Smear group ----

    def _build_smear_group(self) -> QGroupBox:
        """Build the motion-smear entry-list group."""
        group = QGroupBox('Motion smear')
        group.setCheckable(True)
        outer = QVBoxLayout(group)
        self._smear_rows_layout = QVBoxLayout()
        outer.addLayout(self._smear_rows_layout)
        add_btn = QPushButton('Add smear entry')
        add_btn.clicked.connect(self._on_add_smear_clicked)
        outer.addWidget(add_btn)

        self._smear_group = group
        smear = self._optics_sub('smear')
        if isinstance(smear, list):
            for entry in smear:
                self._add_smear_row(entry)
        group.setChecked(isinstance(smear, list))
        group.toggled.connect(self._on_smear_group_toggled)
        return group

    def _add_smear_row(self, entry: dict[str, Any] | None = None) -> None:
        """Append one smear-entry row, optionally seeded from an entry dict."""
        entry = entry or {}
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        dv_spin = _dspin(
            minimum=-200.0,
            maximum=200.0,
            decimals=3,
            step=0.5,
            value=float(entry.get('dv_px', 0.0)),
            tooltip='Drift along v (px).',
        )
        du_spin = _dspin(
            minimum=-200.0,
            maximum=200.0,
            decimals=3,
            step=0.5,
            value=float(entry.get('du_px', 0.0)),
            tooltip='Drift along u (px).',
        )
        class_combo = QComboBox()
        class_combo.addItems(_SMEAR_CLASSES)
        class_index = class_combo.findText(str(entry.get('object_class', 'all')))
        if class_index >= 0:
            class_combo.setCurrentIndex(class_index)
        remove_btn = QPushButton('Remove')
        row.addWidget(QLabel('dv'))
        row.addWidget(dv_spin)
        row.addWidget(QLabel('du'))
        row.addWidget(du_spin)
        row.addWidget(class_combo)
        row.addWidget(remove_btn)
        self._smear_rows_layout.addWidget(container)

        smear_row = _SmearRow(container, dv_spin, du_spin, class_combo)
        self._smear_rows.append(smear_row)
        dv_spin.valueChanged.connect(self._on_smear_value)
        du_spin.valueChanged.connect(self._on_smear_value)
        class_combo.currentTextChanged.connect(self._on_smear_text)
        remove_btn.clicked.connect(lambda: self._remove_smear_row(smear_row))

    def _remove_smear_row(self, row: _SmearRow) -> None:
        """Remove a smear-entry row and rewrite the block."""
        if row in self._smear_rows:
            self._smear_rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._write_smear()

    def _smear_list_from_rows(self) -> list[dict[str, Any]]:
        """Read the current smear rows into a schema list."""
        return [
            {
                'dv_px': float(row.dv_spin.value()),
                'du_px': float(row.du_spin.value()),
                'object_class': row.class_combo.currentText(),
            }
            for row in self._smear_rows
        ]

    def _write_smear(self) -> None:
        """Write the smear list when the group is enabled."""
        if self._syncing:
            return
        if self._smear_group.isChecked():
            self._put_optics('smear', self._smear_list_from_rows())

    def _on_add_smear_clicked(self) -> None:
        """Add a smear row from the button and rewrite the block."""
        self._add_smear_row()
        self._write_smear()

    def _on_smear_group_toggled(self, on: bool) -> None:
        """Insert or remove the whole smear list."""
        if self._syncing:
            return
        if on:
            self._put_optics('smear', self._smear_list_from_rows())
        else:
            self._drop_optics('smear')

    def _on_smear_value(self, _value: float) -> None:
        """Rewrite the smear list on a numeric edit."""
        self._write_smear()

    def _on_smear_text(self, _text: str) -> None:
        """Rewrite the smear list on an object-class edit."""
        self._write_smear()

    # ---- Distortion group ----

    def _build_distortion_group(self) -> QGroupBox:
        """Build the residual-distortion group."""
        group = QGroupBox('Residual distortion')
        group.setCheckable(True)
        form = QFormLayout(group)
        distortion = self._optics_sub('distortion')
        block = distortion if isinstance(distortion, dict) else {}
        self._distortion_k1_spin = _dspin(
            minimum=-1.0,
            maximum=1.0,
            decimals=5,
            step=0.001,
            value=float(block.get('k1', 0.0)),
            tooltip='Radial k1 coefficient.',
        )
        form.addRow('k1:', self._distortion_k1_spin)
        self._distortion_k2_spin = _dspin(
            minimum=-1.0,
            maximum=1.0,
            decimals=5,
            step=0.001,
            value=float(block.get('k2', 0.0)),
            tooltip='Radial k2 coefficient.',
        )
        form.addRow('k2:', self._distortion_k2_spin)
        # The optical center is an optional key pair: an absent key means the
        # frame center, so a center spin only authors its own key when the
        # enable is on and the spin itself is edited (a legitimate 0.0 center
        # is then expressible, and a partial block keeps its absent key).
        has_center = 'center_v' in block or 'center_u' in block
        self._distortion_center_check = QCheckBox('Set optical centre')
        self._distortion_center_check.setChecked(has_center)
        self._distortion_center_check.setToolTip(
            'Enable authoring explicit optical-centre keys; each spin edit '
            'writes only its own key.  Unchecked drops the keys (the '
            'renderer uses the frame center).'
        )
        form.addRow(self._distortion_center_check)
        default_center_v, default_center_u = self._distortion_center_defaults()
        self._distortion_center_v_spin = _dspin(
            minimum=0.0,
            maximum=20000.0,
            decimals=2,
            step=1.0,
            value=float(block.get('center_v', default_center_v)),
            tooltip='Optical-center v (px); absent = frame center.',
        )
        self._distortion_center_v_spin.setEnabled(has_center)
        form.addRow('Center V (px):', self._distortion_center_v_spin)
        self._distortion_center_u_spin = _dspin(
            minimum=0.0,
            maximum=20000.0,
            decimals=2,
            step=1.0,
            value=float(block.get('center_u', default_center_u)),
            tooltip='Optical-center u (px); absent = frame center.',
        )
        self._distortion_center_u_spin.setEnabled(has_center)
        form.addRow('Center U (px):', self._distortion_center_u_spin)
        self._distortion_nonradial_spin = _dspin(
            minimum=0.0,
            maximum=50.0,
            decimals=3,
            step=0.05,
            value=float(block.get('nonradial_rms_px', 0.0)),
            tooltip='Non-radial wander RMS in detector pixels.',
        )
        form.addRow('Non-radial RMS (px):', self._distortion_nonradial_spin)

        self._distortion_group = group
        group.setChecked(isinstance(distortion, dict))
        # Connect after the initial values so the build does not write.
        self._distortion_k1_spin.valueChanged.connect(self._on_distortion_k1)
        self._distortion_k2_spin.valueChanged.connect(self._on_distortion_k2)
        self._distortion_center_v_spin.valueChanged.connect(self._on_distortion_center_v)
        self._distortion_center_u_spin.valueChanged.connect(self._on_distortion_center_u)
        self._distortion_nonradial_spin.valueChanged.connect(self._on_distortion_nonradial)
        self._distortion_center_check.toggled.connect(self._on_distortion_center_check)
        group.toggled.connect(self._on_distortion_group_toggled)
        return group

    def _distortion_center_defaults(self) -> tuple[float, float]:
        """The effective optical-center default: the frame center, in pixels."""
        return (
            float(self.sim_params.get('size_v', 512)) / 2.0,
            float(self.sim_params.get('size_u', 512)) / 2.0,
        )

    def _set_distortion_key(self, key: str, value: Any) -> None:
        """Write one distortion key, preserving every other authored key."""
        if self._syncing or not self._distortion_group.isChecked():
            return
        optics = self._optics_map()
        block = optics.get('distortion')
        if not isinstance(block, dict):
            block = {}
            optics['distortion'] = block
        block[key] = value
        self._updater.request_update()

    def _on_distortion_group_toggled(self, on: bool) -> None:
        """Insert an empty distortion block, or remove the block entirely.

        The block starts empty so unedited keys stay absent (the renderer's
        defaults keep applying); each widget edit then writes only its key.
        """
        if self._syncing:
            return
        if on:
            optics = self._optics_map()
            if not isinstance(optics.get('distortion'), dict):
                optics['distortion'] = {}
            self._updater.request_update()
        else:
            self._drop_optics('distortion')

    def _on_distortion_center_check(self, checked: bool) -> None:
        """Enable the optical-center spins; unchecking drops the center keys.

        Checking writes nothing by itself: an absent center key stays absent
        (the frame center keeps applying) until its own spin is edited.
        """
        self._distortion_center_v_spin.setEnabled(checked)
        self._distortion_center_u_spin.setEnabled(checked)
        if self._syncing or checked:
            return
        optics = self.sim_params.get('optics')
        block = optics.get('distortion') if isinstance(optics, dict) else None
        if isinstance(block, dict) and ('center_v' in block or 'center_u' in block):
            block.pop('center_v', None)
            block.pop('center_u', None)
            self._updater.request_update()
        # The keys are now absent, so show the effective default.  The spins
        # are disabled and their handlers check the enable, so this never
        # writes the keys back.
        default_center_v, default_center_u = self._distortion_center_defaults()
        self._distortion_center_v_spin.setValue(default_center_v)
        self._distortion_center_u_spin.setValue(default_center_u)

    def _on_distortion_k1(self, value: float) -> None:
        """Write the radial k1 coefficient on a spin edit."""
        self._set_distortion_key('k1', float(value))

    def _on_distortion_k2(self, value: float) -> None:
        """Write the radial k2 coefficient on a spin edit."""
        self._set_distortion_key('k2', float(value))

    def _on_distortion_nonradial(self, value: float) -> None:
        """Write the non-radial wander RMS on a spin edit."""
        self._set_distortion_key('nonradial_rms_px', float(value))

    def _on_distortion_center_v(self, value: float) -> None:
        """Write the optical-centre v key on a spin edit, when enabled."""
        if self._distortion_center_check.isChecked():
            self._set_distortion_key('center_v', float(value))

    def _on_distortion_center_u(self, value: float) -> None:
        """Write the optical-centre u key on a spin edit, when enabled."""
        if self._distortion_center_check.isChecked():
            self._set_distortion_key('center_u', float(value))

    # ---- Ghosts group ----

    def _build_ghosts_group(self) -> QGroupBox:
        """Build the ghost-reflection entry-list group."""
        group = QGroupBox('Ghost reflections')
        group.setCheckable(True)
        outer = QVBoxLayout(group)
        self._ghosts_rows_layout = QVBoxLayout()
        outer.addLayout(self._ghosts_rows_layout)
        add_btn = QPushButton('Add ghost')
        add_btn.clicked.connect(self._on_add_ghost_clicked)
        outer.addWidget(add_btn)

        self._ghosts_group = group
        ghosts = self._optics_sub('ghosts')
        if isinstance(ghosts, list):
            for entry in ghosts:
                self._add_ghost_row(entry)
        group.setChecked(isinstance(ghosts, list))
        group.toggled.connect(self._on_ghosts_group_toggled)
        return group

    def _add_ghost_row(self, entry: dict[str, Any] | None = None) -> None:
        """Append one ghost row, optionally seeded from an entry dict."""
        entry = entry or {}
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        dv_spin = _dspin(
            minimum=-2000.0,
            maximum=2000.0,
            decimals=2,
            step=1.0,
            value=float(entry.get('dv_px', 0.0)),
            tooltip='Ghost offset along v (px).',
        )
        du_spin = _dspin(
            minimum=-2000.0,
            maximum=2000.0,
            decimals=2,
            step=1.0,
            value=float(entry.get('du_px', 0.0)),
            tooltip='Ghost offset along u (px).',
        )
        amplitude_spin = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=4,
            step=0.005,
            value=float(entry.get('amplitude', 0.0)),
            tooltip='Ghost amplitude fraction.',
        )
        defocus_spin = _dspin(
            minimum=0.0,
            maximum=100.0,
            decimals=2,
            step=0.5,
            value=float(entry.get('defocus_sigma', 0.0)),
            tooltip='Ghost defocus sigma (px).',
        )
        remove_btn = QPushButton('Remove')
        for label, widget in (
            ('dv', dv_spin),
            ('du', du_spin),
            ('amp', amplitude_spin),
            ('defocus', defocus_spin),
        ):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addWidget(remove_btn)
        self._ghosts_rows_layout.addWidget(container)

        ghost_row = _GhostRow(container, dv_spin, du_spin, amplitude_spin, defocus_spin)
        self._ghost_rows.append(ghost_row)
        for spin in (dv_spin, du_spin, amplitude_spin, defocus_spin):
            spin.valueChanged.connect(self._on_ghost_value)
        remove_btn.clicked.connect(lambda: self._remove_ghost_row(ghost_row))

    def _remove_ghost_row(self, row: _GhostRow) -> None:
        """Remove a ghost row and rewrite the block."""
        if row in self._ghost_rows:
            self._ghost_rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._write_ghosts()

    def _ghost_list_from_rows(self) -> list[dict[str, Any]]:
        """Read the current ghost rows into a schema list."""
        return [
            {
                'dv_px': float(row.dv_spin.value()),
                'du_px': float(row.du_spin.value()),
                'amplitude': float(row.amplitude_spin.value()),
                'defocus_sigma': float(row.defocus_spin.value()),
            }
            for row in self._ghost_rows
        ]

    def _write_ghosts(self) -> None:
        """Write the ghost list when the group is enabled."""
        if self._syncing:
            return
        if self._ghosts_group.isChecked():
            self._put_optics('ghosts', self._ghost_list_from_rows())

    def _on_add_ghost_clicked(self) -> None:
        """Add a ghost row from the button and rewrite the block."""
        self._add_ghost_row()
        self._write_ghosts()

    def _on_ghosts_group_toggled(self, on: bool) -> None:
        """Insert or remove the whole ghost list."""
        if self._syncing:
            return
        if on:
            self._put_optics('ghosts', self._ghost_list_from_rows())
        else:
            self._drop_optics('ghosts')

    def _on_ghost_value(self, _value: float) -> None:
        """Rewrite the ghost list on a numeric edit."""
        self._write_ghosts()

    # ---- Stray-light group ----

    def _build_stray_group(self) -> QGroupBox:
        """Build the stray-light group by reusing the stray-light panel."""
        group = QGroupBox('Stray light')
        group.setCheckable(True)
        form = QFormLayout(group)
        self._build_stray_panel(form)
        self._stray_group = group
        group.setChecked(isinstance(self._optics_sub('stray_light'), dict))
        group.toggled.connect(self._on_stray_group_toggled)
        return group

    def _on_stray_group_toggled(self, on: bool) -> None:
        """Insert the stray-light block from the panel, or remove it."""
        if self._syncing:
            return
        if on:
            self._set_stray('amplitude', float(self._stray_amplitude_spin.value()))
            self._set_stray('direction_deg', float(self._stray_direction_spin.value()))
            self._set_stray('model', self._stray_model_combo.currentText() or 'linear')
            self._on_stray_center('center_v', float(self._stray_center_v_spin.value()))
            self._on_stray_center('center_u', float(self._stray_center_u_spin.value()))
        else:
            self._drop_optics('stray_light')

    # ---- Scene optics & geometry group (oversample + spk_error) ----

    def _build_scene_geometry_group(self) -> QGroupBox:
        """Build the oversample spin and the spacecraft-ephemeris-error group."""
        group = QGroupBox('Scene optics & geometry')
        outer = QVBoxLayout(group)

        over_row = QHBoxLayout()
        self._oversample_check = QCheckBox('Oversample')
        self._oversample_check.setToolTip(
            'Pin the render oversampling factor; unchecked lets the renderer '
            'choose (4 with an active PSF, else 1).'
        )
        self._oversample_spin = QSpinBox()
        self._oversample_spin.setRange(1, 8)
        oversample = self.sim_params.get('oversample')
        self._oversample_check.setChecked(isinstance(oversample, int))
        self._oversample_spin.setValue(int(oversample) if isinstance(oversample, int) else 1)
        self._oversample_spin.setEnabled(isinstance(oversample, int))
        over_row.addWidget(self._oversample_check)
        over_row.addWidget(self._oversample_spin)
        over_row.addStretch()
        outer.addLayout(over_row)
        self._oversample_check.toggled.connect(self._on_oversample_check)
        self._oversample_spin.valueChanged.connect(self._on_oversample_value)

        spk_group = QGroupBox('Spacecraft ephemeris error (parallax)')
        spk_group.setCheckable(True)
        spk_form = QFormLayout(spk_group)
        spk = self.sim_params.get('spk_error')
        block = spk if isinstance(spk, dict) else {}
        self._spk_dv_spin = _dspin(
            minimum=-500.0,
            maximum=500.0,
            decimals=3,
            step=0.5,
            value=float(block.get('dv_px', 0.0)),
            tooltip='Parallax displacement along v at the reference range (px).',
        )
        spk_form.addRow('dv (px):', self._spk_dv_spin)
        self._spk_du_spin = _dspin(
            minimum=-500.0,
            maximum=500.0,
            decimals=3,
            step=0.5,
            value=float(block.get('du_px', 0.0)),
            tooltip='Parallax displacement along u at the reference range (px).',
        )
        spk_form.addRow('du (px):', self._spk_du_spin)
        self._spk_range_spin = _dspin(
            minimum=0.01,
            maximum=1.0e9,
            decimals=2,
            step=100.0,
            value=float(block.get('reference_range_km', 1000.0)),
            tooltip='Range (km) the displacement is quoted at; scales per object.',
        )
        spk_form.addRow('Reference range (km):', self._spk_range_spin)
        outer.addWidget(spk_group)

        self._spk_error_group = spk_group
        spk_group.setChecked(isinstance(spk, dict))
        for spin in (self._spk_dv_spin, self._spk_du_spin, self._spk_range_spin):
            spin.valueChanged.connect(self._on_spk_value)
        spk_group.toggled.connect(self._on_spk_group_toggled)
        return group

    def _on_oversample_check(self, checked: bool) -> None:
        """Pin or clear the oversample factor."""
        self._oversample_spin.setEnabled(checked)
        if self._syncing:
            return
        if checked:
            self.sim_params['oversample'] = int(self._oversample_spin.value())
        else:
            self.sim_params.pop('oversample', None)
        self._updater.request_update()

    def _on_oversample_value(self, value: int) -> None:
        """Update the pinned oversample factor when enabled."""
        if self._syncing:
            return
        if self._oversample_check.isChecked():
            self.sim_params['oversample'] = int(value)
            self._updater.request_update()

    def _spk_block_from_widgets(self) -> dict[str, Any]:
        """Assemble the spk_error block the widgets describe."""
        return {
            'dv_px': float(self._spk_dv_spin.value()),
            'du_px': float(self._spk_du_spin.value()),
            'reference_range_km': float(self._spk_range_spin.value()),
        }

    def _write_spk(self) -> None:
        """Write the spk_error block when the group is enabled."""
        if self._syncing:
            return
        if self._spk_error_group.isChecked():
            self.sim_params['spk_error'] = self._spk_block_from_widgets()
            self._updater.request_update()

    def _on_spk_group_toggled(self, on: bool) -> None:
        """Insert or remove the spk_error block."""
        if self._syncing:
            return
        if on:
            self.sim_params['spk_error'] = self._spk_block_from_widgets()
        else:
            self.sim_params.pop('spk_error', None)
        self._updater.request_update()

    def _on_spk_value(self, _value: float) -> None:
        """Rewrite the spk_error block on a spin edit."""
        self._write_spk()

    # ---- Scene-load sync ----

    def _sync_optics_from_params(self) -> None:
        """Rebuild every Optics-tab widget from the current sim_params."""
        self._syncing = True
        try:
            self._sync_psf()
            self._sync_entry_list_group(
                self._smear_group,
                self._optics_sub('smear'),
                self._smear_rows,
                self._add_smear_row,
            )
            self._sync_distortion()
            self._sync_entry_list_group(
                self._ghosts_group,
                self._optics_sub('ghosts'),
                self._ghost_rows,
                self._add_ghost_row,
            )
            self._sync_stray()
            self._sync_scene_geometry()
        finally:
            self._syncing = False

    def _sync_psf(self) -> None:
        """Sync the PSF group from sim_params."""
        psf = self._optics_sub('psf')
        block = psf if isinstance(psf, dict) else {}
        match = bool(block.get('match_navigator'))
        self._apply_psf_match_state(match)
        if not match:
            sigma_v = float(block.get('sigma_v', 0.55))
            self._psf_sigma_v_spin.setValue(sigma_v)
            # The renderer defaults sigma_u to sigma_v, so the widget shows
            # the same value for a block that omits sigma_u.
            self._psf_sigma_u_spin.setValue(float(block.get('sigma_u', sigma_v)))
            self._psf_w_spin.setValue(float(block.get('w', 0.0)))
            self._psf_r0_spin.setValue(float(block.get('r0', 2.0)))
            self._psf_n_spin.setValue(float(block.get('n', 3.0)))
        self._psf_optics_group.setChecked(isinstance(psf, dict))

    def _sync_entry_list_group(
        self,
        group: QGroupBox,
        entries: Any,
        rows: list[Any],
        add_row: Any,
    ) -> None:
        """Rebuild a dynamic entry-list group (smear or ghosts) from a list."""
        for row in list(rows):
            row.container.setParent(None)
            row.container.deleteLater()
        rows.clear()
        if isinstance(entries, list):
            for entry in entries:
                add_row(entry)
        group.setChecked(isinstance(entries, list))

    def _sync_distortion(self) -> None:
        """Sync the distortion group from sim_params."""
        distortion = self._optics_sub('distortion')
        block = distortion if isinstance(distortion, dict) else {}
        self._distortion_k1_spin.setValue(float(block.get('k1', 0.0)))
        self._distortion_k2_spin.setValue(float(block.get('k2', 0.0)))
        has_center = 'center_v' in block or 'center_u' in block
        self._distortion_center_check.setChecked(has_center)
        # An absent center key displays its effective default (frame center).
        default_center_v, default_center_u = self._distortion_center_defaults()
        self._distortion_center_v_spin.setValue(float(block.get('center_v', default_center_v)))
        self._distortion_center_v_spin.setEnabled(has_center)
        self._distortion_center_u_spin.setValue(float(block.get('center_u', default_center_u)))
        self._distortion_center_u_spin.setEnabled(has_center)
        self._distortion_nonradial_spin.setValue(float(block.get('nonradial_rms_px', 0.0)))
        self._distortion_group.setChecked(isinstance(distortion, dict))

    def _sync_stray(self) -> None:
        """Sync the stray-light group and its panel spins from sim_params."""
        stray = self._optics_sub('stray_light')
        self._stray_amplitude_spin.setValue(float(self._stray_value('amplitude', 0.0)))
        self._stray_direction_spin.setValue(float(self._stray_value('direction_deg', 0.0)))
        model_index = self._stray_model_combo.findText(str(self._stray_value('model', 'linear')))
        if model_index >= 0:
            self._stray_model_combo.setCurrentIndex(model_index)
        has_center = self._stray_has_center()
        self._stray_center_check.setChecked(has_center)
        # An absent center key displays its effective default (frame center).
        default_center_v, default_center_u = self._stray_center_defaults()
        self._stray_center_v_spin.setValue(float(self._stray_value('center_v', default_center_v)))
        self._stray_center_v_spin.setEnabled(has_center)
        self._stray_center_u_spin.setValue(float(self._stray_value('center_u', default_center_u)))
        self._stray_center_u_spin.setEnabled(has_center)
        self._stray_group.setChecked(isinstance(stray, dict))

    def _sync_scene_geometry(self) -> None:
        """Sync the oversample spin and the spk_error group from sim_params."""
        oversample = self.sim_params.get('oversample')
        self._oversample_check.setChecked(isinstance(oversample, int))
        self._oversample_spin.setEnabled(isinstance(oversample, int))
        if isinstance(oversample, int):
            self._oversample_spin.setValue(int(oversample))
        spk = self.sim_params.get('spk_error')
        block = spk if isinstance(spk, dict) else {}
        self._spk_dv_spin.setValue(float(block.get('dv_px', 0.0)))
        self._spk_du_spin.setValue(float(block.get('du_px', 0.0)))
        self._spk_range_spin.setValue(float(block.get('reference_range_km', 1000.0)))
        self._spk_error_group.setChecked(isinstance(spk, dict))
