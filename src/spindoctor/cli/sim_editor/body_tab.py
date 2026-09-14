"""Body tab builder and change handlers.

Builds the per-body editing tab (geometry, shape model, mesh parameters, pose,
lighting, crater relief, anti-aliasing, and the navigation-override group) and
owns the handlers that write body fields back into the data model.

The centre spin boxes write the scene's ``center_v`` / ``center_u``
unchanged, so they carry the scene's own convention: every position in a scene
is a pixel corner, and the centre of pixel ``N`` is ``N + 0.5``.  The row
labels name that convention rather than leaving the reader to infer it from a
rendered frame.
"""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase

# Tooltip for the centre spin boxes, which write the scene value directly.
_CENTER_TOOLTIP = (
    'Body centre {axis} position as a pixel corner: integer N is the boundary '
    'between pixel N-1 and pixel N, so the centre of pixel N is N + 0.5.'
)


class BodyTabMixin(SimEditorBase):
    """Builds and handles the per-body editing tab."""

    def _build_body_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the body at ``idx``."""
        p = self.sim_params['bodies'][idx]
        w = QWidget()
        w.setProperty('kind', 'body')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_body_name(i, t))
        fl.addRow('Name:', name_edit)

        center_v = QDoubleSpinBox()
        center_v.setRange(-10000.0, 20000.0)
        center_v.setDecimals(1)
        center_v.setValue(p.get('center_v', 0.0))
        center_v.setToolTip(_CENTER_TOOLTIP.format(axis='V'))
        center_v.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'center_v', v))
        fl.addRow('Center V (pixel corner):', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-10000.0, 20000.0)
        center_u.setDecimals(1)
        center_u.setValue(p.get('center_u', 0.0))
        center_u.setToolTip(_CENTER_TOOLTIP.format(axis='U'))
        center_u.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'center_u', v))
        fl.addRow('Center U (pixel corner):', center_u)
        # Keep references so drag updates can sync the UI
        w.center_v_spin = center_v  # type: ignore[attr-defined]
        w.center_u_spin = center_u  # type: ignore[attr-defined]

        # Range field (for layering/ordering)
        rng = QDoubleSpinBox()
        rng.setRange(-1e9, 1e9)
        rng.setDecimals(3)
        rng.setValue(p.get('range_km', idx + 1))
        rng.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'range_km', v, trigger_validate=True)
        )
        fl.addRow('Range:', rng)

        smaj = QDoubleSpinBox()
        smaj.setRange(1.0, 5000.0)
        smaj.setDecimals(1)
        smaj.setValue(p.get('axis1', 0.0))
        smaj.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'axis1', v))
        fl.addRow('Axis 1:', smaj)
        smin = QDoubleSpinBox()
        smin.setRange(1.0, 5000.0)
        smin.setDecimals(1)
        smin.setValue(p.get('axis2', 0.0))
        smin.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'axis2', v))
        fl.addRow('Axis 2:', smin)
        sc = QDoubleSpinBox()
        sc.setRange(1.0, 5000.0)
        sc.setDecimals(1)
        sc.setValue(p.get('axis3', 0.0))
        sc.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'axis3', v))
        fl.addRow('Axis 3:', sc)

        # Shape model (B7): ellipsoid or an irregular polyhedral mesh.  For a
        # mesh, lumpiness/seed pick the shape and pose_euler_deg orients it; the
        # axes above scale it.  The mesh fields are inert for an ellipsoid.
        shape_combo = QComboBox()
        shape_combo.addItems(['ellipsoid', 'polyhedral_mesh'])
        shape_index = shape_combo.findText(str(p.get('shape_model', 'ellipsoid')))
        if shape_index >= 0:
            shape_combo.setCurrentIndex(shape_index)
        shape_combo.currentTextChanged.connect(lambda t, i=idx: self._on_body_shape_model(i, t))
        fl.addRow('Shape model:', shape_combo)

        mesh_lump = QDoubleSpinBox()
        mesh_lump.setRange(0.0, 1.0)
        mesh_lump.setDecimals(3)
        mesh_lump.setSingleStep(0.01)
        mesh_lump.setValue(float(p.get('mesh_lumpiness', 0.3)))
        mesh_lump.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_lumpiness', v))
        fl.addRow('Mesh lumpiness:', mesh_lump)

        mesh_seed = QSpinBox()
        mesh_seed.setRange(0, 2147483647)
        mesh_seed.setValue(int(p.get('mesh_seed', 0)))
        mesh_seed.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_seed', v))
        fl.addRow('Mesh seed:', mesh_seed)

        mesh_n_lat = QSpinBox()
        mesh_n_lat.setRange(2, 256)
        mesh_n_lat.setValue(int(p.get('mesh_n_lat', 16)))
        mesh_n_lat.setToolTip('Mesh latitude bands (resolution).')
        mesh_n_lat.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_n_lat', v))
        fl.addRow('Mesh lat bands:', mesh_n_lat)

        mesh_n_lon = QSpinBox()
        mesh_n_lon.setRange(3, 512)
        mesh_n_lon.setValue(int(p.get('mesh_n_lon', 32)))
        mesh_n_lon.setToolTip('Mesh longitude divisions (resolution).')
        mesh_n_lon.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_n_lon', v))
        fl.addRow('Mesh lon divisions:', mesh_n_lon)

        pose = p.get('pose_euler_deg', [0.0, 0.0, 0.0])
        for axis_i, axis_name in enumerate(('X', 'Y', 'Z')):
            pose_spin = QDoubleSpinBox()
            pose_spin.setRange(0.0, 360.0)
            pose_spin.setDecimals(1)
            pose_spin.setWrapping(True)
            pose_spin.setValue(float(pose[axis_i]) if axis_i < len(pose) else 0.0)
            pose_spin.valueChanged.connect(lambda v, i=idx, a=axis_i: self._on_body_pose(i, a, v))
            fl.addRow(f'Mesh pose {axis_name} (deg):', pose_spin)

        rz = QDoubleSpinBox()
        rz.setRange(0.0, 360.0)
        rz.setDecimals(1)
        rz.setSuffix('°')
        rz.setValue(p.get('rotation_z', 0.0))
        rz.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'rotation_z', v))
        fl.addRow('Rotation Z:', rz)
        rt = QDoubleSpinBox()
        rt.setRange(0.0, 90.0)
        rt.setDecimals(1)
        rt.setSuffix('°')
        rt.setValue(p.get('rotation_tilt', 0.0))
        rt.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'rotation_tilt', v))
        fl.addRow('Rotation Tilt:', rt)

        illum = QDoubleSpinBox()
        illum.setRange(0.0, 360.0)
        illum.setDecimals(1)
        illum.setSuffix('°')
        illum.setValue(p.get('illumination_angle', 0.0))
        illum.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'illumination_angle', v))
        fl.addRow('Illumination angle:', illum)
        phase = QDoubleSpinBox()
        phase.setRange(0.0, 180.0)
        phase.setDecimals(1)
        phase.setSuffix('°')
        phase.setValue(p.get('phase_angle', 0.0))
        phase.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'phase_angle', v))
        fl.addRow('Phase angle:', phase)

        # Crater fill slider with min/max labels and spinbox
        cf_row = QHBoxLayout()
        cf_row.setSpacing(4)
        cf_row.setContentsMargins(0, 0, 0, 0)
        cf_min_label = QLabel('0.0')
        cf_min_label.setFixedWidth(35)
        cf_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cf_max_label = QLabel('10.0')
        cf_max_label.setFixedWidth(40)
        cf_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        cf_slider = QSlider(Qt.Orientation.Horizontal)
        cf_slider.setRange(0, 10000)  # 0.0 to 10.0 with 0.001 steps
        cf_slider.setValue(int(p.get('crater_fill', 0.0) * 1000))
        cf_slider.valueChanged.connect(lambda v, i=idx: self._on_body_crater_fill_slider(i, v))
        cf_spin = QDoubleSpinBox()
        cf_spin.setRange(0.0, 10.0)
        cf_spin.setDecimals(3)
        cf_spin.setSingleStep(0.01)
        cf_spin.setValue(p.get('crater_fill', 0.0))
        cf_spin.valueChanged.connect(lambda v, i=idx: self._on_body_crater_fill_spin(i, v))
        cf_row.addWidget(cf_min_label)
        cf_row.addWidget(cf_slider, stretch=1)
        cf_row.addWidget(cf_max_label)
        cf_row.addWidget(cf_spin)
        cf_holder = QWidget()
        cf_holder.setLayout(cf_row)
        fl.addRow('Crater fill (0-10):', cf_holder)
        # Store references for sync
        w.crater_fill_slider = cf_slider  # type: ignore[attr-defined]
        w.crater_fill_spin = cf_spin  # type: ignore[attr-defined]
        cmin = QDoubleSpinBox()
        cmin.setRange(0.01, 0.25)
        cmin.setDecimals(3)
        cmin.setSingleStep(0.005)
        cmin.setValue(p.get('crater_min_radius', 0.05))
        cmin.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_min_radius', v))
        fl.addRow('Crater min radius:', cmin)
        cmax = QDoubleSpinBox()
        cmax.setRange(0.01, 0.25)
        cmax.setDecimals(3)
        cmax.setSingleStep(0.005)
        cmax.setValue(p.get('crater_max_radius', 0.25))
        cmax.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_max_radius', v))
        fl.addRow('Crater max radius:', cmax)
        cexp = QDoubleSpinBox()
        cexp.setRange(1.1, 5.0)
        cexp.setDecimals(2)
        cexp.setSingleStep(0.05)
        cexp.setValue(p.get('crater_power_law_exponent', 3.0))
        cexp.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_power_law_exponent', v)
        )
        fl.addRow('Crater power-law exponent:', cexp)
        crs = QDoubleSpinBox()
        crs.setRange(0.0, 3.0)
        crs.setDecimals(3)
        crs.setSingleStep(0.01)
        crs.setValue(p.get('crater_relief_scale', 0.6))
        crs.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_relief_scale', v))
        fl.addRow('Crater relief scale:', crs)

        # Anti-aliasing slider with min/max labels and spinbox
        aa_row = QHBoxLayout()
        aa_row.setSpacing(4)
        aa_row.setContentsMargins(0, 0, 0, 0)
        aa_min_label = QLabel('0.0')
        aa_min_label.setFixedWidth(35)
        aa_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        aa_max_label = QLabel('1.0')
        aa_max_label.setFixedWidth(35)
        aa_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        aa_slider = QSlider(Qt.Orientation.Horizontal)
        aa_slider.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        aa_slider.setValue(int(p.get('anti_aliasing', 0.5) * 1000))
        aa_slider.valueChanged.connect(lambda v, i=idx: self._on_body_anti_aliasing_slider(i, v))
        aa_spin = QDoubleSpinBox()
        aa_spin.setRange(0.0, 1.0)
        aa_spin.setDecimals(3)
        aa_spin.setSingleStep(0.01)
        aa_spin.setValue(p.get('anti_aliasing', 0.5))
        aa_spin.valueChanged.connect(lambda v, i=idx: self._on_body_anti_aliasing_spin(i, v))
        aa_row.addWidget(aa_min_label)
        aa_row.addWidget(aa_slider, stretch=1)
        aa_row.addWidget(aa_max_label)
        aa_row.addWidget(aa_spin)
        aa_holder = QWidget()
        aa_holder.setLayout(aa_row)
        fl.addRow('Anti-aliasing:', aa_holder)
        # Store references for sync
        w.anti_aliasing_slider = aa_slider  # type: ignore[attr-defined]
        w.anti_aliasing_spin = aa_spin  # type: ignore[attr-defined]

        # Crater seed: -1 ("Auto") omits the key so the renderer hashes the
        # geometry; any other value pins the crater pattern.
        crater_seed = QSpinBox()
        crater_seed.setRange(-1, 2147483647)
        crater_seed.setSpecialValueText('Auto')
        crater_seed.setValue(int(p['seed']) if p.get('seed') is not None else -1)
        crater_seed.setToolTip('Crater RNG seed; Auto hashes the body geometry.')
        crater_seed.valueChanged.connect(lambda v, i=idx: self._on_body_seed(i, v))
        fl.addRow('Crater seed:', crater_seed)

        km_pp = QDoubleSpinBox()
        km_pp.setRange(0.0, 1.0e9)
        km_pp.setDecimals(4)
        km_pp.setValue(float(p.get('km_per_pixel', 0.0)))
        km_pp.setToolTip('Physical scale at the limb (0 = none); drives the irregularity factor.')
        km_pp.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'km_per_pixel', v))
        fl.addRow('km per pixel:', km_pp)

        # Navigation-override group: the renderer ignores it (it always draws the
        # true geometry), but the navigator predicts the body with these fields
        # overlaid, so a scene can render a mesh and predict a smooth ellipsoid or
        # a different pose.  Unchecking it removes the override.
        override = p.get('nav_override') if isinstance(p.get('nav_override'), dict) else {}
        nav_group = QGroupBox('Navigation override (predicted geometry)')
        nav_group.setCheckable(True)
        nav_group.setChecked('nav_override' in p)
        nav_form = QFormLayout(nav_group)
        nav_shape = QComboBox()
        nav_shape.addItems(['ellipsoid', 'polyhedral_mesh'])
        nav_shape_idx = nav_shape.findText(
            str(override.get('shape_model', p.get('shape_model', 'ellipsoid')))
        )
        if nav_shape_idx >= 0:
            nav_shape.setCurrentIndex(nav_shape_idx)
        nav_form.addRow('Predicted shape:', nav_shape)
        nav_lump = QDoubleSpinBox()
        nav_lump.setRange(0.0, 1.0)
        nav_lump.setDecimals(3)
        nav_lump.setSingleStep(0.01)
        nav_lump.setValue(float(override.get('mesh_lumpiness', p.get('mesh_lumpiness', 0.3))))
        nav_form.addRow('Predicted lumpiness:', nav_lump)
        nav_pose = override.get('pose_euler_deg', p.get('pose_euler_deg', [0.0, 0.0, 0.0]))
        nav_pose_spins = []
        for axis_i, axis_name in enumerate(('X', 'Y', 'Z')):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 360.0)
            sp.setDecimals(1)
            sp.setWrapping(True)
            sp.setValue(float(nav_pose[axis_i]) if axis_i < len(nav_pose) else 0.0)
            nav_form.addRow(f'Predicted pose {axis_name} (deg):', sp)
            nav_pose_spins.append(sp)

        def _update_override(_arg: Any = None, i: int = idx) -> None:
            if not nav_group.isChecked():
                self.sim_params['bodies'][i].pop('nav_override', None)
            else:
                self.sim_params['bodies'][i]['nav_override'] = {
                    'shape_model': nav_shape.currentText(),
                    'mesh_lumpiness': float(nav_lump.value()),
                    'pose_euler_deg': [float(s.value()) for s in nav_pose_spins],
                }
            self._updater.request_update()

        nav_group.toggled.connect(_update_override)
        nav_shape.currentTextChanged.connect(_update_override)
        nav_lump.valueChanged.connect(_update_override)
        for sp in nav_pose_spins:
            sp.valueChanged.connect(_update_override)
        fl.addRow(nav_group)

        # Truth-side appearance groups (relief, photometry, texture, transits,
        # mesh extras); each follows the absent-key discipline and stores its
        # widget refs on ``w``.
        self._build_body_appearance_groups(w, idx, main_layout)

        # Delete button at bottom
        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('body', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

        return w

    # ---- Field handlers ----
    def _on_body_field(
        self, idx: int, key: str, value: Any, *, trigger_validate: bool = False
    ) -> None:
        """Write a scalar body field into the data model and re-render."""
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()
            if trigger_validate and key == 'range_km':
                self._validate_ranges()

    def _on_body_shape_model(self, idx: int, text: str) -> None:
        """Set the shape model and gate the mesh-only appearance controls."""
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['shape_model'] = text
            tab_idx = self._find_tab_by_properties('body', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    self._sync_body_mesh_enabled(tab_w, text == 'polyhedral_mesh')
            self._updater.request_update()

    def _on_body_seed(self, idx: int, value: int) -> None:
        """Set an integer crater seed, or remove it (Auto) when value is -1."""
        if 0 <= idx < len(self.sim_params['bodies']):
            if value < 0:
                self.sim_params['bodies'][idx].pop('seed', None)
            else:
                self.sim_params['bodies'][idx]['seed'] = int(value)
            self._updater.request_update()

    def _on_body_pose(self, idx: int, axis: int, value: float) -> None:
        """Update one axis of a body's mesh pose (pose_euler_deg)."""
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            pose = list(body.get('pose_euler_deg', [0.0, 0.0, 0.0]))
            while len(pose) < 3:
                pose.append(0.0)
            pose[axis] = float(value)
            body['pose_euler_deg'] = pose
            self._updater.request_update()

    def _on_body_name(self, idx: int, text: str) -> None:
        """Rename a body and refresh the tab titles."""
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['name'] = text
            # update tab title
            self._update_tab_titles()
            self._updater.request_update()

    def _on_body_crater_fill_slider(self, idx: int, value: int) -> None:
        """Sync the crater-fill spin box from the slider and update."""
        fill_val = value / 1000.0
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin = tab_w.crater_fill_spin  # type: ignore[attr-defined]
                spin.blockSignals(True)
                spin.setValue(fill_val)
                spin.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['crater_fill'] = fill_val
            self._updater.request_update()

    def _on_body_crater_fill_spin(self, idx: int, value: float) -> None:
        """Sync the crater-fill slider from the spin box and update."""
        slider_val = int(value * 1000)
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                slider = tab_w.crater_fill_slider  # type: ignore[attr-defined]
                slider.blockSignals(True)
                slider.setValue(slider_val)
                slider.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['crater_fill'] = value
            self._updater.request_update()

    def _on_body_anti_aliasing_slider(self, idx: int, value: int) -> None:
        """Sync the anti-aliasing spin box from the slider and update."""
        aa_val = value / 1000.0
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin = tab_w.anti_aliasing_spin  # type: ignore[attr-defined]
                spin.blockSignals(True)
                spin.setValue(aa_val)
                spin.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['anti_aliasing'] = aa_val
            self._updater.request_update()

    def _on_body_anti_aliasing_spin(self, idx: int, value: float) -> None:
        """Sync the anti-aliasing slider from the spin box and update."""
        slider_val = int(value * 1000)
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                slider = tab_w.anti_aliasing_slider  # type: ignore[attr-defined]
                slider.blockSignals(True)
                slider.setValue(slider_val)
                slider.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['anti_aliasing'] = value
            self._updater.request_update()
