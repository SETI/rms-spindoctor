"""Ring tab builder and change handlers.

Builds one editing tab per ``ring_system`` feature (kind, navigability,
optical depth, the kind-specific shape keys, the mode-1 catalog orbit, and
the photometric truth scalars) and owns the handlers that write ring fields
back into the data model.  The first feature's tab additionally carries the
system-level controls: the shared projection geometry (center, opening
angles, node), the phase angle, and the optional physical range / pixel
scale.  The orbit's m-mode / edge-wave perturbations, the truth-side
orbit_error / declared_orbit_sigma groups, and the system-level azimuthal /
moonlets blocks live in the sibling
:mod:`spindoctor.cli.sim_editor.ring_advanced` mixin, appended below the
groups built here.

The center spin boxes write the scene's shared ``center_v`` / ``center_u``
unchanged, so they carry the scene's own convention: every position in a scene
is a pixel corner, and the center of pixel ``N`` is ``N + 0.5``.  The row
labels name that convention rather than leaving the reader to infer it from a
rendered frame.
"""

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.sim.scene_schema import (
    RING_KIND_SHAPE_KEYS,
    RING_SHAPE_KEYS,
    RING_SIDE_DEFAULTS,
    ring_key_description,
)

# The kind vocabulary and which kinds take which shape keys come from the
# schema's table, the one the validator enforces: a stray shape key fails
# validation, so the handlers insert and remove keys as the kind changes.
_RING_KINDS = tuple(RING_KIND_SHAPE_KEYS)
# The value a kind switch writes for a shape key the new kind requires.
_SHAPE_KEY_DEFAULTS: dict[str, float] = {
    'width_px': 20.0,
    'wavelength_px': 8.0,
    'damping_px': 16.0,
}
# What each shape key holds, for its tooltip.
_SHAPE_KEY_DESCRIPTIONS: dict[str, str] = {
    key: ring_key_description('ring_system.features', key)
    for key in ('width_px', 'wavelength_px', 'damping_px')
} | {
    'side': (
        "which side of the orbit carries tau: 'in' fills inward of it, 'out' "
        f'outward (default {RING_SIDE_DEFAULTS["edge"]!r} on an edge, '
        f'{RING_SIDE_DEFAULTS["ramp"]!r} on a ramp)'
    ),
}

# Tooltip for the shared-center spin boxes, which write the scene value directly.
_CENTER_TOOLTIP = (
    'Ring system center {axis} position as a pixel corner: integer N is the '
    'boundary between pixel N-1 and pixel N, so the center of pixel N is N + 0.5.'
)


def _kind_takes(kind: str, key: str) -> bool:
    """Whether a feature of ``kind`` takes the shape key ``key``."""
    return key in RING_KIND_SHAPE_KEYS.get(kind, {})


def _shape_key_tooltip(key: str, kind: str) -> str:
    """The tooltip for a shape-key widget on a feature of ``kind``.

    An enabled widget describes its key.  A grayed-out one also says why: the
    kinds that take the key, read from the schema's per-kind table, and that
    the validator rejects the key on any other kind.

    Parameters:
        key: A shape key from ``RING_SHAPE_KEYS``.
        kind: The feature's kind.

    Returns:
        The tooltip text.
    """
    description = _SHAPE_KEY_DESCRIPTIONS[key]
    text = f'{key}: {description}.'
    if _kind_takes(kind, key):
        return text
    takers = [k for k, rules in RING_KIND_SHAPE_KEYS.items() if key in rules]
    if len(takers) > 2:
        names = ', '.join(takers[:-1]) + f', and {takers[-1]}'
    else:
        names = ' and '.join(takers)
    return (
        f'{text} Disabled for a {kind!r} feature: only {names} features take {key}, '
        f'and the validator rejects it on every other kind.'
    )


class RingTabMixin(SimEditorBase):
    """Builds and handles the per-feature ring_system editing tabs."""

    def _ring_features(self) -> list[dict[str, Any]]:
        """The live ring_system feature list ([] when the scene has none)."""
        ring_system = self.sim_params.get('ring_system')
        if not isinstance(ring_system, dict):
            return []
        features = ring_system.get('features')
        return features if isinstance(features, list) else []

    def _build_ring_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the ring_system feature at ``idx``."""
        features = self._ring_features()
        p = features[idx]
        w = QWidget()
        w.setProperty('kind', 'ring')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_ring_name(i, t))
        fl.addRow('Name:', name_edit)

        kind_combo = QComboBox()
        kind_combo.addItems(list(_RING_KINDS))
        kind_combo.setCurrentText(str(p.get('kind', 'ringlet')))
        kind_combo.currentTextChanged.connect(lambda t, i=idx: self._on_ring_kind(i, t))
        fl.addRow('Kind:', kind_combo)

        navigable_check = QCheckBox('Navigable (the navigator is told about this feature)')
        navigable_check.setChecked(bool(p.get('navigable', False)))
        navigable_check.setToolTip(
            'Unchecked features render as confounders the navigator has no '
            'knowledge of (dropped from nav_params).'
        )
        navigable_check.clicked.connect(lambda checked, i=idx: self._on_ring_navigable(i, checked))
        fl.addRow(navigable_check, QLabel(''))

        tau_spin = QDoubleSpinBox()
        tau_spin.setRange(0.0, 100.0)
        tau_spin.setDecimals(3)
        tau_spin.setValue(float(p.get('tau', 1.0)))
        tau_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_field(i, 'tau', v))
        fl.addRow('Optical depth (tau):', tau_spin)

        kind = str(p.get('kind', 'ringlet'))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.1, 10000.0)
        width_spin.setDecimals(1)
        width_spin.setSuffix(' px')
        width_spin.setValue(float(p.get('width_px', _SHAPE_KEY_DEFAULTS['width_px'])))
        width_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_shape(i, 'width_px', v))
        fl.addRow('Radial width (px):', width_spin)

        side_combo = QComboBox()
        side_combo.addItems(['in', 'out'])
        side_combo.setCurrentText(str(p.get('side', RING_SIDE_DEFAULTS.get(kind, 'out'))))
        side_combo.currentTextChanged.connect(lambda t, i=idx: self._on_ring_shape(i, 'side', t))
        fl.addRow('Side (edge/ramp):', side_combo)

        wavelength_spin = QDoubleSpinBox()
        wavelength_spin.setRange(0.1, 10000.0)
        wavelength_spin.setDecimals(1)
        wavelength_spin.setSuffix(' px')
        wavelength_spin.setValue(
            float(p.get('wavelength_px', _SHAPE_KEY_DEFAULTS['wavelength_px']))
        )
        wavelength_spin.valueChanged.connect(
            lambda v, i=idx: self._on_ring_shape(i, 'wavelength_px', v)
        )
        fl.addRow('Wave wavelength (px):', wavelength_spin)

        damping_spin = QDoubleSpinBox()
        damping_spin.setRange(0.1, 10000.0)
        damping_spin.setDecimals(1)
        damping_spin.setSuffix(' px')
        damping_spin.setValue(float(p.get('damping_px', _SHAPE_KEY_DEFAULTS['damping_px'])))
        damping_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_shape(i, 'damping_px', v))
        fl.addRow('Wave damping (px):', damping_spin)

        w.width_spin = width_spin  # type: ignore[attr-defined]
        w.side_combo = side_combo  # type: ignore[attr-defined]
        w.wavelength_spin = wavelength_spin  # type: ignore[attr-defined]
        w.damping_spin = damping_spin  # type: ignore[attr-defined]
        self._sync_ring_shape_widgets(w, kind)

        orbit = p.get('orbit') or {}
        orbit_block = 'ring_system.features.orbit'
        fl.addRow(QLabel('<b>Catalog orbit (mode 1)</b>'), QLabel(''))
        orbit_a = QDoubleSpinBox()
        orbit_a.setRange(0.1, 1.0e6)
        orbit_a.setDecimals(1)
        orbit_a.setSuffix(' px')
        orbit_a.setValue(float(orbit.get('a_px', 100.0)))
        orbit_a.setToolTip(f'a_px: {ring_key_description(orbit_block, "a_px")}.')
        orbit_a.valueChanged.connect(lambda v, i=idx: self._on_ring_orbit(i, 'a_px', v))
        fl.addRow('a (px):', orbit_a)
        orbit_ae = QDoubleSpinBox()
        orbit_ae.setRange(0.0, 1.0e5)
        orbit_ae.setDecimals(2)
        orbit_ae.setSuffix(' px')
        orbit_ae.setValue(float(orbit.get('ae_px', 0.0)))
        orbit_ae.setToolTip(f'ae_px: {ring_key_description(orbit_block, "ae_px")}.')
        orbit_ae.valueChanged.connect(lambda v, i=idx: self._on_ring_orbit(i, 'ae_px', v))
        fl.addRow('ae (px):', orbit_ae)
        orbit_long_peri = QDoubleSpinBox()
        orbit_long_peri.setRange(-360.0, 360.0)
        orbit_long_peri.setDecimals(1)
        orbit_long_peri.setSuffix(' deg')
        orbit_long_peri.setValue(float(orbit.get('long_peri_deg', 0.0)))
        orbit_long_peri.setToolTip(
            f'long_peri_deg: {ring_key_description(orbit_block, "long_peri_deg")}.'
        )
        orbit_long_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_orbit(i, 'long_peri_deg', v)
        )
        fl.addRow('long_peri (deg):', orbit_long_peri)
        orbit_rate_peri = QDoubleSpinBox()
        orbit_rate_peri.setRange(-1000.0, 1000.0)
        orbit_rate_peri.setDecimals(3)
        orbit_rate_peri.setSuffix(' deg/day')
        orbit_rate_peri.setValue(float(orbit.get('rate_peri_deg_per_day', 0.0)))
        orbit_rate_peri.setToolTip(
            f'rate_peri_deg_per_day: {ring_key_description(orbit_block, "rate_peri_deg_per_day")}.'
        )
        orbit_rate_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_orbit(i, 'rate_peri_deg_per_day', v)
        )
        fl.addRow('rate_peri (deg/day):', orbit_rate_peri)

        fl.addRow(QLabel('<b>Photometric truth</b>'), QLabel(''))
        albedo_spin = QDoubleSpinBox()
        albedo_spin.setRange(0.0, 10.0)
        albedo_spin.setDecimals(3)
        albedo_spin.setValue(float(p.get('albedo', 0.5)))
        albedo_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_field(i, 'albedo', v))
        fl.addRow('Albedo (A):', albedo_spin)
        phase_g_spin = QDoubleSpinBox()
        phase_g_spin.setRange(-0.99, 0.99)
        phase_g_spin.setDecimals(2)
        phase_g_spin.setSingleStep(0.05)
        phase_g_spin.setValue(float(p.get('phase_g', -0.3)))
        phase_g_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_field(i, 'phase_g', v))
        fl.addRow('Phase asymmetry (g):', phase_g_spin)

        if idx == 0:
            self._build_ring_system_group(w, fl)

        self._build_ring_advanced_groups(w, idx, main_layout)

        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('ring', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

        return w

    def _build_ring_system_group(self, w: QWidget, fl: QFormLayout) -> None:
        """Add the system-level controls (shared geometry, range, phase)."""
        ring_system = self.sim_params['ring_system']
        geometry = ring_system.setdefault('geometry', {})
        fl.addRow(QLabel('<b>Ring system (shared)</b>'), QLabel(''))

        center_v = QDoubleSpinBox()
        center_v.setRange(-1.0e6, 1.0e6)
        center_v.setDecimals(1)
        center_v.setValue(float(geometry.get('center_v', self.sim_params['size_v'] / 2.0)))
        center_v.setToolTip(_CENTER_TOOLTIP.format(axis='V'))
        center_v.valueChanged.connect(lambda v: self._on_ring_geometry('center_v', v))
        fl.addRow('Center V (pixel corner):', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-1.0e6, 1.0e6)
        center_u.setDecimals(1)
        center_u.setValue(float(geometry.get('center_u', self.sim_params['size_u'] / 2.0)))
        center_u.setToolTip(_CENTER_TOOLTIP.format(axis='U'))
        center_u.valueChanged.connect(lambda v: self._on_ring_geometry('center_u', v))
        fl.addRow('Center U (pixel corner):', center_u)
        # Keep references so drag updates can sync the UI.
        w.center_v_spin = center_v  # type: ignore[attr-defined]
        w.center_u_spin = center_u  # type: ignore[attr-defined]

        # The opening-angle spin ranges mirror the validator's (-90, 90].
        for key, label, low, high, default in (
            ('opening_deg_obs', 'Opening B (observer):', -89.9, 90.0, 90.0),
            ('opening_deg_sun', 'Opening B (sun):', -89.9, 90.0, 90.0),
            ('node_deg', 'Node angle:', -360.0, 360.0, 0.0),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(1)
            spin.setSuffix(' deg')
            spin.setValue(float(geometry.get(key, default)))
            spin.valueChanged.connect(lambda v, k=key: self._on_ring_geometry(k, v))
            fl.addRow(label, spin)
            # Keep references so tests can assert the validator-matched ranges.
            setattr(w, f'{key}_spin', spin)

        phase_spin = QDoubleSpinBox()
        phase_spin.setRange(0.0, 180.0)
        phase_spin.setDecimals(1)
        phase_spin.setSuffix(' deg')
        phase_spin.setValue(float(ring_system.get('phase_deg', 0.0)))
        phase_spin.setToolTip('phase_deg: the solar phase angle, in degrees.')
        phase_spin.valueChanged.connect(lambda v: self._on_ring_system_field('phase_deg', v))
        fl.addRow('Phase angle (deg):', phase_spin)

        # Physical range (km): optional-key discipline -- absent unless set.
        # Required whenever the system overlaps a body or the scene plants
        # spk_error; it is the system's per-pixel depth anchor.
        has_range_km = ring_system.get('range_km') is not None
        range_km_check = QCheckBox('Set physical range (km)')
        range_km_check.setChecked(has_range_km)
        range_km_check.setToolTip(
            'Write a physical range_km on the ring system (depth ordering '
            'against bodies and spk_error parallax); unchecked leaves the key '
            'absent.'
        )
        range_km_spin = QDoubleSpinBox()
        range_km_spin.setRange(0.01, 1.0e12)
        range_km_spin.setDecimals(1)
        range_km_spin.setValue(float(ring_system.get('range_km') or 1.0e6))
        range_km_spin.setEnabled(has_range_km)
        range_km_check.clicked.connect(
            lambda checked, spin=range_km_spin: self._on_ring_range_km_enabled(checked, spin)
        )
        range_km_spin.valueChanged.connect(self._on_ring_range_km_value)
        fl.addRow(range_km_check, range_km_spin)
        w.range_km_check = range_km_check  # type: ignore[attr-defined]
        w.range_km_spin = range_km_spin  # type: ignore[attr-defined]

        km_per_pixel_spin = QDoubleSpinBox()
        km_per_pixel_spin.setRange(0.001, 1.0e9)
        km_per_pixel_spin.setDecimals(3)
        km_per_pixel_spin.setValue(float(ring_system.get('km_per_pixel', 1.0)))
        km_per_pixel_spin.valueChanged.connect(
            lambda v: self._on_ring_system_field('km_per_pixel', v)
        )
        fl.addRow('km per pixel:', km_per_pixel_spin)

    # ---- Field handlers ----
    def _on_ring_field(self, idx: int, key: str, value: Any) -> None:
        """Write a scalar feature field into the data model and re-render."""
        features = self._ring_features()
        if 0 <= idx < len(features):
            features[idx][key] = float(value) if isinstance(value, (int, float)) else value
            self._updater.request_update()

    def _on_ring_shape(self, idx: int, key: str, value: Any) -> None:
        """Write a kind-specific shape field (only when the kind takes it)."""
        features = self._ring_features()
        if not 0 <= idx < len(features):
            return
        kind = str(features[idx].get('kind', 'ringlet'))
        if not _kind_takes(kind, key):
            return
        features[idx][key] = float(value) if isinstance(value, (int, float)) else value
        self._updater.request_update()

    def _on_ring_kind(self, idx: int, kind: str) -> None:
        """Change a feature's kind, adjusting its shape keys to the vocabulary."""
        features = self._ring_features()
        if not 0 <= idx < len(features):
            return
        feature = features[idx]
        feature['kind'] = kind
        self._apply_ring_kind_constraints(feature)
        # Enable exactly the shape widgets the new kind takes, showing the
        # values the renderer will use.
        tab_idx = self._find_tab_by_properties('ring', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                self._sync_ring_shape_widgets(tab_w, kind)
                self._show_ring_shape_values(tab_w, feature)
        self._updater.request_update()

    def _show_ring_shape_values(self, w: QWidget, feature: dict[str, Any]) -> None:
        """Show the shape values the renderer will use for ``feature``.

        After a kind switch the feature holds exactly its kind's required
        shape keys, and an absent ``side`` renders with the kind's default
        from ``RING_SIDE_DEFAULTS``.  Each widget the kind takes shows that
        value; a widget the kind rejects keeps its text.  Signals are blocked
        while the widgets change, so showing a default never writes it into
        the scene.

        Parameters:
            w: The feature's editing tab.
            feature: The feature mapping, its kind already set.
        """
        kind = str(feature.get('kind', 'ringlet'))
        spins: dict[str, QDoubleSpinBox] = {
            'width_px': w.width_spin,  # type: ignore[attr-defined]
            'wavelength_px': w.wavelength_spin,  # type: ignore[attr-defined]
            'damping_px': w.damping_spin,  # type: ignore[attr-defined]
        }
        for key, spin in spins.items():
            if key in feature:
                spin.blockSignals(True)
                spin.setValue(float(feature[key]))
                spin.blockSignals(False)
        if kind in RING_SIDE_DEFAULTS:
            combo: QComboBox = w.side_combo  # type: ignore[attr-defined]
            combo.blockSignals(True)
            combo.setCurrentText(str(feature.get('side', RING_SIDE_DEFAULTS[kind])))
            combo.blockSignals(False)

    def _sync_ring_shape_widgets(self, w: QWidget, kind: str) -> None:
        """Enable the shape widgets ``kind`` takes and say why the rest are gray."""
        widgets: dict[str, QWidget] = {
            'width_px': w.width_spin,  # type: ignore[attr-defined]
            'side': w.side_combo,  # type: ignore[attr-defined]
            'wavelength_px': w.wavelength_spin,  # type: ignore[attr-defined]
            'damping_px': w.damping_spin,  # type: ignore[attr-defined]
        }
        for key in RING_SHAPE_KEYS:
            widgets[key].setEnabled(_kind_takes(kind, key))
            widgets[key].setToolTip(_shape_key_tooltip(key, kind))

    def _apply_ring_kind_constraints(self, feature: dict[str, Any]) -> None:
        """Insert required shape keys and drop disallowed ones for the kind.

        The schema rejects a stray shape key on a kind that ignores it, so a
        kind switch must rewrite the feature's shape keys, defaulting any
        newly required one, per the schema's per-kind table.
        """
        rules = RING_KIND_SHAPE_KEYS.get(str(feature.get('kind', 'ringlet')), {})
        for key in RING_SHAPE_KEYS:
            rule = rules.get(key)
            if rule is None:
                feature.pop(key, None)
            elif rule == 'required':
                feature.setdefault(key, _SHAPE_KEY_DEFAULTS[key])

    def _on_ring_navigable(self, idx: int, checked: bool) -> None:
        """Toggle the feature's navigability flag."""
        features = self._ring_features()
        if 0 <= idx < len(features):
            features[idx]['navigable'] = bool(checked)
            self._updater.request_update()

    def _on_ring_orbit(self, idx: int, key: str, value: float) -> None:
        """Update a mode-1 catalog-orbit parameter."""
        features = self._ring_features()
        if 0 <= idx < len(features):
            orbit = features[idx].setdefault('orbit', {})
            orbit[key] = float(value)
            self._updater.request_update()

    def _on_ring_name(self, idx: int, text: str) -> None:
        """Rename a feature and refresh the tab titles."""
        features = self._ring_features()
        if 0 <= idx < len(features):
            features[idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _on_ring_geometry(self, key: str, value: float) -> None:
        """Update a shared projection-geometry field."""
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict):
            ring_system.setdefault('geometry', {})[key] = float(value)
            self._updater.request_update()

    def _on_ring_system_field(self, key: str, value: float) -> None:
        """Update a system-level scalar (phase_deg, km_per_pixel)."""
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict):
            ring_system[key] = float(value)
            self._updater.request_update()

    def _on_ring_range_km_enabled(self, enabled: bool, spin: QDoubleSpinBox) -> None:
        """Insert or remove the system's optional physical range_km key."""
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict):
            if enabled:
                ring_system['range_km'] = float(spin.value())
            else:
                ring_system.pop('range_km', None)
            spin.setEnabled(enabled)
            self._updater.request_update()

    def _on_ring_range_km_value(self, value: float) -> None:
        """Update the system's physical range_km when the key is enabled."""
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict) and 'range_km' in ring_system:
            ring_system['range_km'] = float(value)
            self._updater.request_update()
