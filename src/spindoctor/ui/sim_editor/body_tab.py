"""Body tab builder and change handlers.

Builds the per-body editing tab (geometry, shape model, mesh parameters, pose,
lighting, crater relief, anti-aliasing, and the navigation-override group) and
owns the handlers that write body fields back into the data model.

The center spin boxes write the scene's ``center_v`` / ``center_u``
unchanged, so they carry the scene's own convention: every position in a scene
is a pixel corner, and the center of pixel ``N`` is ``N + 0.5``.  The row
labels name that convention rather than leaving the reader to infer it from a
rendered frame.
"""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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

from spindoctor.sim.scene.fields import default_body_name, default_body_range_km, scene_value
from spindoctor.sim.scene.idealized import DEFAULT_SHAPE_MODEL
from spindoctor.sim.scene.truth_body import (
    DEFAULT_ANTI_ALIASING,
    DEFAULT_CRATER_MAX_RADIUS,
    DEFAULT_CRATER_MIN_RADIUS,
    DEFAULT_CRATER_POWER_LAW_EXPONENT,
    DEFAULT_CRATER_RELIEF_SCALE,
)
from spindoctor.sim.shared.mesh import mesh_spec_from_params
from spindoctor.ui.sim_editor.base import SimEditorBase
from spindoctor.ui.sim_editor.body_mesh_keys import (
    STASH_MESSAGE_MS,
    refused_keys,
    stash_message,
)

# Tooltip for the center spin boxes, which write the scene value directly.
_CENTER_TOOLTIP = (
    'Body center {axis} position as a pixel corner: integer N is the boundary '
    'between pixel N-1 and pixel N, so the center of pixel N is N + 0.5.'
)


def _semi_c_px(body: dict[str, Any]) -> float:
    """The body's semi-axis c: its own, or the smaller of semi-axes a and b."""
    semi_a = float(scene_value(body, 'semi_a_px', 0.0))
    semi_b = float(scene_value(body, 'semi_b_px', 0.0))
    return float(scene_value(body, 'semi_c_px', min(semi_a, semi_b)))


class BodyTabMixin(SimEditorBase):
    """Builds and handles the per-body editing tab."""

    def _build_body_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the body at ``idx``."""
        # Built from the stash-merged view, so a tab rebuilt while the body is a
        # mesh shows the values switching it back to an ellipsoid restores.
        p = self._mesh_key_stash.view(self.sim_params['bodies'][idx])
        # Every control shows the value the renderer draws with: the body's own, or
        # the default the scene types declare for a key the body leaves out.
        mesh = mesh_spec_from_params(p)
        shape_model = str(scene_value(p, 'shape_model', DEFAULT_SHAPE_MODEL))
        w = QWidget()
        w.setProperty('kind', 'body')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(str(scene_value(p, 'name', default_body_name(idx))))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_body_name(i, t))
        fl.addRow('Name:', name_edit)

        center_v = QDoubleSpinBox()
        center_v.setRange(-10000.0, 20000.0)
        center_v.setDecimals(1)
        # If the center is unset, the renderer draws the body at the frame center.
        center_v.setValue(float(scene_value(p, 'center_v', self.sim_params['size_v'] / 2.0)))
        center_v.setToolTip(_CENTER_TOOLTIP.format(axis='V'))
        center_v.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'center_v', v))
        fl.addRow('Center V (pixel corner):', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-10000.0, 20000.0)
        center_u.setDecimals(1)
        center_u.setValue(float(scene_value(p, 'center_u', self.sim_params['size_u'] / 2.0)))
        center_u.setToolTip(_CENTER_TOOLTIP.format(axis='U'))
        center_u.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'center_u', v))
        fl.addRow('Center U (pixel corner):', center_u)
        # Keep references so drag updates can sync the UI
        w.name_edit = name_edit  # type: ignore[attr-defined]
        w.center_v_spin = center_v  # type: ignore[attr-defined]
        w.center_u_spin = center_u  # type: ignore[attr-defined]

        # Range (for layering/ordering).  If the range is unset, the spin shows the
        # renderer's default for the body's list position and stays disabled.  Two
        # overlapping bodies need an explicit range each, so checking the box writes
        # the range the spin shows, and clearing it removes the key.
        has_range = p.get('range_km') is not None
        rng_check = QCheckBox('Set range (km)')
        rng_check.setChecked(has_range)
        rng_check.setToolTip(
            'Write range_km on the body.  Two overlapping bodies each need it to set '
            'their depth order, and a scene with spk_error needs it on every body.  '
            "Clear the box to use the default for the body's position in the list."
        )
        rng = QDoubleSpinBox()
        rng.setRange(0.001, 1e9)
        rng.setDecimals(3)
        rng.setValue(float(scene_value(p, 'range_km', default_body_range_km(idx))))
        rng.setEnabled(has_range)
        rng_check.clicked.connect(
            lambda checked, i=idx, spin=rng: self._on_body_range_enabled(i, checked, spin)
        )
        rng.valueChanged.connect(lambda v, i=idx: self._on_body_range_value(i, v))
        fl.addRow(rng_check, rng)
        w.range_check = rng_check  # type: ignore[attr-defined]
        w.range_spin = rng  # type: ignore[attr-defined]

        # The three semi-axes, each half the body's width along its axis.  Two
        # decimals hold half of any width given to a tenth of a pixel, and the
        # half-pixel step changes the width by one pixel.
        semi_spins = []
        for key, label, value in (
            ('semi_a_px', 'Semi-axis a (px):', float(scene_value(p, 'semi_a_px', 0.0))),
            ('semi_b_px', 'Semi-axis b (px):', float(scene_value(p, 'semi_b_px', 0.0))),
            ('semi_c_px', 'Semi-axis c (px):', _semi_c_px(p)),
        ):
            semi_spin = QDoubleSpinBox()
            semi_spin.setRange(0.5, 2500.0)
            semi_spin.setDecimals(2)
            semi_spin.setSingleStep(0.5)
            semi_spin.setValue(value)
            semi_spin.valueChanged.connect(lambda v, i=idx, k=key: self._on_body_field(i, k, v))
            fl.addRow(label, semi_spin)
            semi_spins.append(semi_spin)
        w.semi_a_spin, w.semi_b_spin, w.semi_c_spin = semi_spins  # type: ignore[attr-defined]

        # Shape model: ellipsoid or an irregular polyhedral mesh.  For a
        # mesh, lumpiness/seed pick the shape and pose_euler_deg orients it; the
        # semi-axes above scale it.  The mesh fields are inert for an ellipsoid.
        shape_combo = QComboBox()
        shape_combo.addItems(['ellipsoid', 'polyhedral_mesh'])
        shape_index = shape_combo.findText(shape_model)
        if shape_index >= 0:
            shape_combo.setCurrentIndex(shape_index)
        shape_combo.currentTextChanged.connect(lambda t, i=idx: self._on_body_shape_model(i, t))
        fl.addRow('Shape model:', shape_combo)
        w.shape_combo = shape_combo  # type: ignore[attr-defined]

        mesh_lump = QDoubleSpinBox()
        mesh_lump.setRange(0.0, 1.0)
        mesh_lump.setDecimals(3)
        mesh_lump.setSingleStep(0.01)
        mesh_lump.setValue(mesh.lumpiness)
        mesh_lump.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_lumpiness', v))
        fl.addRow('Mesh lumpiness:', mesh_lump)
        w.mesh_lumpiness_spin = mesh_lump  # type: ignore[attr-defined]

        mesh_seed = QSpinBox()
        mesh_seed.setRange(0, 2147483647)
        mesh_seed.setValue(mesh.seed)
        mesh_seed.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_seed', v))
        fl.addRow('Mesh seed:', mesh_seed)

        mesh_n_lat = QSpinBox()
        mesh_n_lat.setRange(2, 256)
        mesh_n_lat.setValue(mesh.n_lat)
        mesh_n_lat.setToolTip('Mesh latitude bands (resolution).')
        mesh_n_lat.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_n_lat', v))
        fl.addRow('Mesh lat bands:', mesh_n_lat)

        mesh_n_lon = QSpinBox()
        mesh_n_lon.setRange(3, 512)
        mesh_n_lon.setValue(mesh.n_lon)
        mesh_n_lon.setToolTip('Mesh longitude divisions (resolution).')
        mesh_n_lon.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'mesh_n_lon', v))
        fl.addRow('Mesh lon divisions:', mesh_n_lon)
        w.mesh_seed_spin = mesh_seed  # type: ignore[attr-defined]
        w.mesh_n_lat_spin = mesh_n_lat  # type: ignore[attr-defined]
        w.mesh_n_lon_spin = mesh_n_lon  # type: ignore[attr-defined]

        pose = mesh.pose_euler_deg
        pose_spins = []
        for axis_i, axis_name in enumerate(('X', 'Y', 'Z')):
            pose_spin = QDoubleSpinBox()
            pose_spin.setRange(0.0, 360.0)
            pose_spin.setDecimals(1)
            pose_spin.setWrapping(True)
            pose_spin.setValue(pose[axis_i])
            pose_spin.valueChanged.connect(lambda v, i=idx, a=axis_i: self._on_body_pose(i, a, v))
            fl.addRow(f'Mesh pose {axis_name} (deg):', pose_spin)
            pose_spins.append(pose_spin)
        w.pose_spins = pose_spins  # type: ignore[attr-defined]

        rz = QDoubleSpinBox()
        rz.setRange(0.0, 360.0)
        rz.setDecimals(1)
        rz.setSuffix('°')
        rz.setValue(float(scene_value(p, 'rotation_z', 0.0)))
        rz.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'rotation_z', v))
        fl.addRow('Rotation Z:', rz)
        lat = QDoubleSpinBox()
        lat.setRange(-90.0, 90.0)
        lat.setDecimals(1)
        lat.setSuffix('°')
        lat.setValue(float(scene_value(p, 'sub_observer_lat_deg', 0.0)))
        lat.setToolTip(
            'Latitude of the point below the observer; positive tips the north pole '
            '(semi-axis a) toward the observer.'
        )
        lat.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'sub_observer_lat_deg', v))
        fl.addRow('Sub-observer lat:', lat)
        w.sub_observer_lat_spin = lat  # type: ignore[attr-defined]

        illum = QDoubleSpinBox()
        illum.setRange(0.0, 360.0)
        illum.setDecimals(1)
        illum.setSuffix('°')
        illum.setValue(float(scene_value(p, 'illumination_angle', 0.0)))
        illum.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'illumination_angle', v))
        fl.addRow('Illumination angle:', illum)
        phase = QDoubleSpinBox()
        phase.setRange(0.0, 180.0)
        phase.setDecimals(1)
        phase.setSuffix('°')
        phase.setValue(float(scene_value(p, 'phase_angle', 0.0)))
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
        crater_fill = float(scene_value(p, 'crater_fill', 0.0))
        cf_slider.setValue(int(crater_fill * 1000))
        cf_slider.valueChanged.connect(lambda v, i=idx: self._on_body_crater_fill_slider(i, v))
        cf_spin = QDoubleSpinBox()
        cf_spin.setRange(0.0, 10.0)
        cf_spin.setDecimals(3)
        cf_spin.setSingleStep(0.01)
        cf_spin.setValue(crater_fill)
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
        cmin.setValue(float(scene_value(p, 'crater_min_radius', DEFAULT_CRATER_MIN_RADIUS)))
        cmin.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_min_radius', v))
        fl.addRow('Crater min radius:', cmin)
        cmax = QDoubleSpinBox()
        cmax.setRange(0.01, 0.25)
        cmax.setDecimals(3)
        cmax.setSingleStep(0.005)
        cmax.setValue(float(scene_value(p, 'crater_max_radius', DEFAULT_CRATER_MAX_RADIUS)))
        cmax.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_max_radius', v))
        fl.addRow('Crater max radius:', cmax)
        cexp = QDoubleSpinBox()
        cexp.setRange(1.1, 5.0)
        cexp.setDecimals(2)
        cexp.setSingleStep(0.05)
        cexp.setValue(
            float(scene_value(p, 'crater_power_law_exponent', DEFAULT_CRATER_POWER_LAW_EXPONENT))
        )
        cexp.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_power_law_exponent', v)
        )
        fl.addRow('Crater power-law exponent:', cexp)
        crs = QDoubleSpinBox()
        crs.setRange(0.0, 3.0)
        crs.setDecimals(3)
        crs.setSingleStep(0.01)
        crs.setValue(float(scene_value(p, 'crater_relief_scale', DEFAULT_CRATER_RELIEF_SCALE)))
        crs.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'crater_relief_scale', v))
        fl.addRow('Crater relief scale:', crs)
        w.crater_min_radius_spin = cmin  # type: ignore[attr-defined]
        w.crater_max_radius_spin = cmax  # type: ignore[attr-defined]
        w.crater_exponent_spin = cexp  # type: ignore[attr-defined]
        w.crater_relief_scale_spin = crs  # type: ignore[attr-defined]

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
        anti_aliasing = float(scene_value(p, 'anti_aliasing', DEFAULT_ANTI_ALIASING))
        aa_slider.setValue(int(anti_aliasing * 1000))
        aa_slider.valueChanged.connect(lambda v, i=idx: self._on_body_anti_aliasing_slider(i, v))
        aa_spin = QDoubleSpinBox()
        aa_spin.setRange(0.0, 1.0)
        aa_spin.setDecimals(3)
        aa_spin.setSingleStep(0.01)
        aa_spin.setValue(anti_aliasing)
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

        km_pp = QDoubleSpinBox()
        km_pp.setRange(0.0, 1.0e9)
        km_pp.setDecimals(4)
        km_pp.setValue(float(scene_value(p, 'km_per_pixel', 0.0)))
        km_pp.setToolTip('Physical scale at the limb (0 = none); drives the irregularity factor.')
        km_pp.valueChanged.connect(lambda v, i=idx: self._on_body_field(i, 'km_per_pixel', v))
        fl.addRow('km per pixel:', km_pp)

        # Navigation-override group: the renderer ignores it (it always draws the
        # true geometry), but the navigator predicts the body with these fields
        # overlaid, so a scene can render a mesh and predict a smooth ellipsoid or
        # a different pose.  Turning it on writes an empty override, and an edit
        # writes only its own key.  Unchecking it removes the override.  If the
        # override leaves a key out, the navigator uses the body's own value,
        # and so does the control, which follows edits to the body's value.
        raw_override = p.get('nav_override')
        override: dict[str, Any] = raw_override if isinstance(raw_override, dict) else {}
        nav_group = QGroupBox('Navigation override (predicted geometry)')
        nav_group.setCheckable(True)
        nav_group.setChecked(isinstance(raw_override, dict))
        nav_form = QFormLayout(nav_group)
        nav_shape = QComboBox()
        nav_shape.addItems(['ellipsoid', 'polyhedral_mesh'])
        nav_shape_idx = nav_shape.findText(str(scene_value(override, 'shape_model', shape_model)))
        if nav_shape_idx >= 0:
            nav_shape.setCurrentIndex(nav_shape_idx)
        nav_form.addRow('Predicted shape:', nav_shape)
        nav_lump = QDoubleSpinBox()
        nav_lump.setRange(0.0, 1.0)
        nav_lump.setDecimals(3)
        nav_lump.setSingleStep(0.01)
        nav_lump.setValue(float(scene_value(override, 'mesh_lumpiness', mesh.lumpiness)))
        nav_form.addRow('Predicted lumpiness:', nav_lump)
        nav_pose = scene_value(override, 'pose_euler_deg', mesh.pose_euler_deg)
        nav_pose_spins = []
        for axis_i, axis_name in enumerate(('X', 'Y', 'Z')):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 360.0)
            sp.setDecimals(1)
            sp.setWrapping(True)
            sp.setValue(float(nav_pose[axis_i]) if axis_i < len(nav_pose) else 0.0)
            sp.valueChanged.connect(lambda v, i=idx, a=axis_i: self._on_body_override_pose(i, a, v))
            nav_form.addRow(f'Predicted pose {axis_name} (deg):', sp)
            nav_pose_spins.append(sp)
        nav_group.toggled.connect(lambda on, i=idx: self._on_body_override_toggled(i, on))
        nav_shape.currentTextChanged.connect(
            lambda t, i=idx: self._on_body_override_field(i, 'shape_model', t)
        )
        nav_lump.valueChanged.connect(
            lambda v, i=idx: self._on_body_override_field(i, 'mesh_lumpiness', float(v))
        )
        fl.addRow(nav_group)
        w.nav_override_group = nav_group  # type: ignore[attr-defined]
        w.nav_shape_combo = nav_shape  # type: ignore[attr-defined]
        w.nav_lump_spin = nav_lump  # type: ignore[attr-defined]
        w.nav_pose_spins = nav_pose_spins  # type: ignore[attr-defined]

        # Truth-side appearance groups (relief, photometry, texture, transits,
        # mesh extras); each follows the absent-key discipline and stores its
        # widget refs on ``w``.
        self._build_body_appearance_groups(w, idx, main_layout)
        self._sync_body_mesh_enabled(w, shape_model == 'polyhedral_mesh')

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
        """Write a scalar body field into the data model and re-render.

        The value keeps the type its control emits: a ``QDoubleSpinBox`` writes a
        float, and a ``QSpinBox`` (``mesh_seed``, ``mesh_n_lat``, ``mesh_n_lon``)
        writes an int, because the scene validator refuses a float for those keys.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx][key] = value
            self._updater.request_update()
            if trigger_validate and key == 'range_km':
                self._validate_ranges()
            if key in ('semi_a_px', 'semi_b_px'):
                self._refresh_body_derived_spins(idx)
            if key == 'mesh_lumpiness':
                self._refresh_body_override_controls(idx)

    def _refresh_body_derived_spins(self, idx: int) -> None:
        """Show the renderer's defaults in a body's unset semi-axis c and center spins.

        If ``semi_c_px`` is unset, the renderer draws it as the smaller of ``semi_a_px``
        and ``semi_b_px``.  If the center is unset, the renderer draws the body at the
        frame center.  An edit to a semi-axis or to the frame size therefore changes
        what these spins show.  If the body sets the key, its spin keeps its value.  The
        refresh never writes to the scene.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
        """
        tab_idx = self._find_tab_by_properties('body', idx)
        tab_w = self._tabs.widget(tab_idx) if tab_idx is not None else None
        if tab_w is None:
            return
        body = self.sim_params['bodies'][idx]
        derived = (
            (tab_w.semi_c_spin, 'semi_c_px', _semi_c_px(self._mesh_key_stash.view(body))),  # type: ignore[attr-defined]
            (tab_w.center_v_spin, 'center_v', self.sim_params['size_v'] / 2.0),  # type: ignore[attr-defined]
            (tab_w.center_u_spin, 'center_u', self.sim_params['size_u'] / 2.0),  # type: ignore[attr-defined]
        )
        for spin, key, shown in derived:
            if body.get(key) is None:
                spin.blockSignals(True)
                spin.setValue(shown)
                spin.blockSignals(False)

    def _on_body_range_enabled(self, idx: int, enabled: bool, spin: QDoubleSpinBox) -> None:
        """Write the range the spin shows, or remove the body's range.

        Checking the box writes the spin's value, which for an unset range is the
        renderer's default, so the author can make that default explicit without
        changing it.  Clearing the box removes ``range_km`` and puts the default
        back in the spin.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
            enabled: Whether the range box is checked.
            spin: The body tab's range spin.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            if enabled:
                body['range_km'] = float(spin.value())
            else:
                body.pop('range_km', None)
                spin.blockSignals(True)
                spin.setValue(default_body_range_km(idx))
                spin.blockSignals(False)
            spin.setEnabled(enabled)
            self._updater.request_update()
            self._validate_ranges()

    def _on_body_range_value(self, idx: int, value: float) -> None:
        """Write an edited range when the body sets one.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
            value: The range spin's value.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            if body.get('range_km') is not None:
                self._on_body_field(idx, 'range_km', float(value), trigger_validate=True)

    def _body_override(self, idx: int) -> dict[str, Any]:
        """Return the body's ``nav_override`` mapping, writing an empty one if it is unset.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.

        Returns:
            The mapping stored in the body, so a write to it edits the scene.
        """
        body = self.sim_params['bodies'][idx]
        override = body.get('nav_override')
        if not isinstance(override, dict):
            override = {}
            body['nav_override'] = override
        return override

    def _on_body_override_toggled(self, idx: int, on: bool) -> None:
        """Turn a body's navigation override on as an empty mapping, or remove it.

        An empty override leaves every key unset, so the navigator predicts the body
        with its own values, which is what the group's controls show.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
            on: Whether the group is checked.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            if on:
                self.sim_params['bodies'][idx]['nav_override'] = {}
            else:
                self.sim_params['bodies'][idx].pop('nav_override', None)
            self._refresh_body_override_controls(idx)
            self._updater.request_update()

    def _on_body_override_field(self, idx: int, key: str, value: Any) -> None:
        """Write one navigation-override key, leaving the others as they are.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
            key: The override key the edited control writes.
            value: The control's value.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            self._body_override(idx)[key] = value
            self._updater.request_update()

    def _on_body_override_pose(self, idx: int, axis: int, value: float) -> None:
        """Write the override pose with one axis edited.

        The other two axes keep the pose the navigator would use otherwise: the
        override's own pose if it sets one, else the body's.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
            axis: The edited axis (0, 1, or 2).
            value: The axis angle in degrees.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            mesh = mesh_spec_from_params(self._mesh_key_stash.view(body))
            override = self._body_override(idx)
            pose = [float(a) for a in scene_value(override, 'pose_euler_deg', mesh.pose_euler_deg)]
            pose[axis] = float(value)
            override['pose_euler_deg'] = pose
            self._updater.request_update()

    def _refresh_body_override_controls(self, idx: int) -> None:
        """Show the body's own values in its unset navigation-override controls.

        If the override leaves a key out, the navigator uses the body's own
        value, so an edit to the body's shape model, mesh lumpiness, or pose changes
        what that control shows.  A control whose override key is set keeps its
        value.  The refresh never writes to the scene.

        Parameters:
            idx: The body's index in the scene's ``bodies`` list.
        """
        tab_idx = self._find_tab_by_properties('body', idx)
        tab_w: Any = self._tabs.widget(tab_idx) if tab_idx is not None else None
        if tab_w is None:
            return
        body = self.sim_params['bodies'][idx]
        view = self._mesh_key_stash.view(body)
        mesh = mesh_spec_from_params(view)
        raw_override = body.get('nav_override')
        override = raw_override if isinstance(raw_override, dict) else {}
        if override.get('shape_model') is None:
            tab_w.nav_shape_combo.blockSignals(True)
            tab_w.nav_shape_combo.setCurrentText(
                str(scene_value(view, 'shape_model', DEFAULT_SHAPE_MODEL))
            )
            tab_w.nav_shape_combo.blockSignals(False)
        if override.get('mesh_lumpiness') is None:
            tab_w.nav_lump_spin.blockSignals(True)
            tab_w.nav_lump_spin.setValue(mesh.lumpiness)
            tab_w.nav_lump_spin.blockSignals(False)
        if override.get('pose_euler_deg') is None:
            for spin, angle in zip(tab_w.nav_pose_spins, mesh.pose_euler_deg, strict=True):
                spin.blockSignals(True)
                spin.setValue(float(angle))
                spin.blockSignals(False)

    def _on_body_shape_model(self, idx: int, text: str) -> None:
        """Set the shape model and gate the shape-specific controls and keys.

        The body moves the keys its new shape model refuses
        (:func:`~.body_mesh_keys.refused_keys`) into the editor's stash, so the
        scene keeps validating, and the status bar names the keys moved
        (:func:`~.body_mesh_keys.stash_message`); it takes back the keys its old
        shape model refused.
        """
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            old_refused = refused_keys(str(scene_value(body, 'shape_model', 'ellipsoid')))
            body['shape_model'] = text
            new_refused = refused_keys(text)
            if new_refused != old_refused:
                self._mesh_key_stash.give_back(body, old_refused)
                taken = self._mesh_key_stash.take(body, new_refused)
                status_bar = self.statusBar()
                if taken and status_bar is not None:
                    status_bar.showMessage(
                        stash_message(
                            str(scene_value(body, 'name', default_body_name(idx))), taken, text
                        ),
                        STASH_MESSAGE_MS,
                    )
            tab_idx = self._find_tab_by_properties('body', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    self._sync_body_mesh_enabled(tab_w, text == 'polyhedral_mesh')
            self._refresh_body_override_controls(idx)
            self._updater.request_update()

    def _on_body_pose(self, idx: int, axis: int, value: float) -> None:
        """Update one axis of a body's mesh pose (pose_euler_deg)."""
        if 0 <= idx < len(self.sim_params['bodies']):
            body = self.sim_params['bodies'][idx]
            pose = list(scene_value(body, 'pose_euler_deg', [0.0, 0.0, 0.0]))
            while len(pose) < 3:
                pose.append(0.0)
            pose[axis] = float(value)
            body['pose_euler_deg'] = pose
            self._refresh_body_override_controls(idx)
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
