"""Ring advanced groups: orbit perturbations, planted errors, and truth clutter.

Splits the ring tab's advanced controls out of the geometry/kind tab so neither
module runs long.  The per-feature groups cover the catalog orbit's m >= 2
``modes`` list and satellite ``edge_wave``, the truth-side planted
``orbit_error``, and the idealized ``declared_orbit_sigma`` error bars; the
first feature's tab additionally carries the system-level truth blocks --
``ring_system.azimuthal`` (brightness modulation, planet-shadow wedge, seeded
spokes) and the ``ring_system.moonlets`` list with its optional propeller
sub-blocks.

Every group follows the editor-wide absent-key discipline: an enable checkbox
writes its block only when active, an inactive control leaves its key absent,
and each widget writes only its own key.  List-valued keys (``modes``,
``moonlets``) follow the row discipline: rows drive the list, and removing the
last row removes the key.  Every widget reference the round-trip and per-widget
tests reach for is stored on the tab widget ``w`` (``w.edge_wave_group``,
``w.moonlet_rows``, and so on), so a test can drive one control and assert the
resulting ``sim_params`` edit.
"""

from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.cli.sim_editor.widgets import make_dspin as _dspin
from spindoctor.sim.scene_schema import ring_key_description


def _key_tooltip(block: str, key: str) -> str:
    """A ring widget's tooltip: its scene key and what the key holds.

    Parameters:
        block: The key's block path in the schema's rename table.
        key: The scene key the widget writes.

    Returns:
        ``'<key>: <description>.'``, the description read from the schema's
        rename table, the same one the validator's rename messages use.
    """
    return f'{key}: {ring_key_description(block, key)}.'


def _as_map(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, else an empty dict."""
    return value if isinstance(value, dict) else {}


@dataclass
class _ModeRow:
    """The widgets of one orbit m-mode entry row."""

    container: QWidget
    m_spin: QSpinBox
    amp_spin: QDoubleSpinBox
    peri_spin: QDoubleSpinBox


@dataclass
class _MoonletRow:
    """The widgets of one moonlet entry: placement spins plus a propeller group."""

    container: QWidget
    a_spin: QDoubleSpinBox
    lam_spin: QDoubleSpinBox
    radius_spin: QDoubleSpinBox
    amplitude_spin: QDoubleSpinBox
    propeller_group: QGroupBox
    prop_length_spin: QDoubleSpinBox
    prop_width_spin: QDoubleSpinBox
    prop_contrast_spin: QDoubleSpinBox


class RingAdvancedMixin(SimEditorBase):
    """Builds and handles the ring tab's orbit-perturbation and truth groups."""

    # ---- Shared helpers ----

    def _ring_feature(self, idx: int) -> dict[str, Any] | None:
        """Return the ring feature dict at ``idx``, or None when out of range."""
        features = self._ring_features()
        if 0 <= idx < len(features):
            feature: dict[str, Any] = features[idx]
            return feature
        return None

    def _ring_tab_widget(self, idx: int) -> QWidget | None:
        """Return the built tab widget for the ring feature at ``idx``, or None."""
        tab_idx = self._find_tab_by_properties('ring', idx)
        if tab_idx is None:
            return None
        return self._tabs.widget(tab_idx)

    def _ring_system_map(self) -> dict[str, Any] | None:
        """Return the live ``ring_system`` mapping, or None when absent."""
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict):
            return ring_system
        return None

    def _build_ring_advanced_groups(self, w: QWidget, idx: int, layout: QVBoxLayout) -> None:
        """Add the advanced groups to a ring tab, storing refs on ``w``.

        Parameters:
            w: The feature's tab widget (widget references are stored on it).
            idx: The feature's index in ``ring_system.features``.
            layout: The tab's main vertical layout.
        """
        p = self._ring_features()[idx]
        layout.addWidget(self._build_ring_modes_group(w, idx, p))
        layout.addWidget(self._build_ring_edge_wave_group(w, idx, p))
        layout.addWidget(self._build_ring_orbit_error_group(w, idx, p))
        layout.addWidget(self._build_ring_sigma_group(w, idx, p))
        if idx == 0:
            layout.addWidget(self._build_ring_azimuthal_group(w))
            layout.addWidget(self._build_ring_moonlets_group(w))

    # ---- Orbit m-modes ----

    def _build_ring_modes_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the orbit m-modes group (a list of ``{m, amp_px, long_peri_deg}`` rows)."""
        group = QGroupBox('Orbit m-modes')
        group.setToolTip(
            'Resonantly forced radial modes on the catalog orbit: '
            'r = a_px - amp_px*cos(m*(lam - long_peri_deg)), with m >= 2 (mode 1 '
            'is the base ellipse). No rows leaves the key absent.'
        )
        outer = QVBoxLayout(group)
        rows_layout = QVBoxLayout()
        outer.addLayout(rows_layout)
        add_btn = QPushButton('Add m-mode')
        outer.addWidget(add_btn)
        w.mode_rows = []  # type: ignore[attr-defined]
        w.mode_rows_layout = rows_layout  # type: ignore[attr-defined]
        orbit = _as_map(p.get('orbit'))
        for mode in orbit.get('modes') or []:
            self._add_ring_mode_row(w, idx, mode)
        add_btn.clicked.connect(lambda _c=False, i=idx: self._on_ring_add_mode(i))
        w.modes_group = group  # type: ignore[attr-defined]
        return group

    def _add_ring_mode_row(self, w: QWidget, idx: int, entry: dict[str, Any]) -> None:
        """Append one m-mode row (m, radial amplitude, pericenter longitude)."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        block = 'ring_system.features.orbit.modes'
        m_spin = QSpinBox()
        m_spin.setRange(2, 50)
        m_spin.setValue(int(entry.get('m', 2)))
        m_spin.setToolTip('m: the mode number, an integer of at least 2.')
        amp = _dspin(
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(entry.get('amp_px', 2.0)),
            tooltip=_key_tooltip(block, 'amp_px'),
        )
        peri = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(entry.get('long_peri_deg', 0.0)),
            tooltip=_key_tooltip(block, 'long_peri_deg'),
        )
        remove_btn = QPushButton('Remove')
        for label, widget in (('m', m_spin), ('amp (px)', amp), ('long_peri (deg)', peri)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addWidget(remove_btn)
        mode_row = _ModeRow(container, m_spin, amp, peri)
        w.mode_rows.append(mode_row)  # type: ignore[attr-defined]
        w.mode_rows_layout.addWidget(container)  # type: ignore[attr-defined]
        for widget in (m_spin, amp, peri):
            widget.valueChanged.connect(lambda _v, i=idx: self._rewrite_ring_modes(i))
        remove_btn.clicked.connect(
            lambda _c=False, i=idx, r=mode_row: self._remove_ring_mode_row(i, r)
        )

    def _mode_list_from_rows(self, rows: list[_ModeRow]) -> list[dict[str, Any]]:
        """Read m-mode rows into their schema list of dicts."""
        return [
            {
                'm': int(r.m_spin.value()),
                'amp_px': float(r.amp_spin.value()),
                'long_peri_deg': float(r.peri_spin.value()),
            }
            for r in rows
        ]

    def _on_ring_add_mode(self, idx: int) -> None:
        """Append an m-mode row and rewrite the modes list."""
        w = self._ring_tab_widget(idx)
        if w is None:
            return
        self._add_ring_mode_row(w, idx, {})
        self._rewrite_ring_modes(idx)

    def _rewrite_ring_modes(self, idx: int) -> None:
        """Rewrite the orbit's modes list from the rows (absent when empty)."""
        feature = self._ring_feature(idx)
        w = self._ring_tab_widget(idx)
        if feature is None or w is None:
            return
        orbit = feature.setdefault('orbit', {})
        modes = self._mode_list_from_rows(w.mode_rows)  # type: ignore[attr-defined]
        if modes:
            orbit['modes'] = modes
        else:
            orbit.pop('modes', None)
        self._updater.request_update()

    def _remove_ring_mode_row(self, idx: int, row: _ModeRow) -> None:
        """Remove an m-mode row and rewrite the modes list."""
        w = self._ring_tab_widget(idx)
        rows = getattr(w, 'mode_rows', None) if w is not None else None
        if rows is not None and row in rows:
            rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._rewrite_ring_modes(idx)

    # ---- Satellite edge wave ----

    def _build_ring_edge_wave_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the edge-wave group (amp_px, wavelength_px, damping_rad, lam0_deg)."""
        orbit = _as_map(p.get('orbit'))
        wave = _as_map(orbit.get('edge_wave'))
        block = 'ring_system.features.orbit.edge_wave'
        group = QGroupBox('Satellite edge wave')
        group.setCheckable(True)
        group.setToolTip(
            'A damped radial wave on the catalog orbit downstream of the '
            'perturbing moon at lam0_deg. Unchecked leaves the key absent.'
        )
        form = QFormLayout(group)
        amp = _dspin(
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(wave.get('amp_px', 1.0)),
            tooltip=_key_tooltip(block, 'amp_px'),
        )
        wavelength = _dspin(
            minimum=0.1,
            maximum=10000.0,
            decimals=1,
            step=1.0,
            value=float(wave.get('wavelength_px', 8.0)),
            tooltip=_key_tooltip(block, 'wavelength_px'),
        )
        damp = _dspin(
            minimum=0.01,
            maximum=2.0,
            decimals=3,
            step=0.1,
            value=float(wave.get('damping_rad', 0.5)),
            tooltip=(
                f'{_key_tooltip(block, "damping_rad")} The validator caps it at 2 '
                'to keep the modular wrap seam small.'
            ),
        )
        lam0 = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(wave.get('lam0_deg', 0.0)),
            tooltip=f"{_key_tooltip(block, 'lam0_deg')} The perturbing moon's longitude.",
        )
        form.addRow('Amplitude (px):', amp)
        form.addRow('Wavelength (px):', wavelength)
        form.addRow('Damping (rad):', damp)
        form.addRow('Launch lam0 (deg):', lam0)
        group.setChecked('edge_wave' in orbit)
        group.toggled.connect(lambda on, i=idx: self._on_ring_edge_wave_toggled(i, on))
        amp.valueChanged.connect(lambda v, i=idx: self._on_ring_edge_wave_value(i, 'amp_px', v))
        wavelength.valueChanged.connect(
            lambda v, i=idx: self._on_ring_edge_wave_value(i, 'wavelength_px', v)
        )
        damp.valueChanged.connect(
            lambda v, i=idx: self._on_ring_edge_wave_value(i, 'damping_rad', v)
        )
        lam0.valueChanged.connect(lambda v, i=idx: self._on_ring_edge_wave_value(i, 'lam0_deg', v))
        w.edge_wave_group = group  # type: ignore[attr-defined]
        w.edge_wave_amp_spin = amp  # type: ignore[attr-defined]
        w.edge_wave_wavelength_spin = wavelength  # type: ignore[attr-defined]
        w.edge_wave_damp_spin = damp  # type: ignore[attr-defined]
        w.edge_wave_lam0_spin = lam0  # type: ignore[attr-defined]
        return group

    def _on_ring_edge_wave_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the orbit's edge_wave map."""
        feature = self._ring_feature(idx)
        w = self._ring_tab_widget(idx)
        if feature is None or w is None:
            return
        orbit = feature.setdefault('orbit', {})
        if checked:
            orbit['edge_wave'] = {
                'amp_px': float(w.edge_wave_amp_spin.value()),  # type: ignore[attr-defined]
                'wavelength_px': float(w.edge_wave_wavelength_spin.value()),  # type: ignore[attr-defined]
                'damping_rad': float(w.edge_wave_damp_spin.value()),  # type: ignore[attr-defined]
                'lam0_deg': float(w.edge_wave_lam0_spin.value()),  # type: ignore[attr-defined]
            }
        else:
            orbit.pop('edge_wave', None)
        self._updater.request_update()

    def _on_ring_edge_wave_value(self, idx: int, key: str, value: float) -> None:
        """Update one edge-wave component when the map is present."""
        feature = self._ring_feature(idx)
        if feature is None:
            return
        wave = _as_map(feature.get('orbit')).get('edge_wave')
        if isinstance(wave, dict):
            wave[key] = float(value)
            self._updater.request_update()

    # ---- Planted orbit error (truth) ----

    def _build_ring_orbit_error_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the planted orbit-error group (truth-side radial model error)."""
        error = _as_map(p.get('orbit_error'))
        group = QGroupBox('Planted orbit error (truth)')
        group.setCheckable(True)
        group.setToolTip(
            'Displaces the RENDERED feature off its catalog orbit; the '
            'navigator predicts from the catalog and never sees these values. '
            'Unchecked leaves the key absent.'
        )
        form = QFormLayout(group)
        delta_a = _dspin(
            minimum=-1000.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(error.get('delta_a_px', 0.0)),
            tooltip='Semimajor-axis error in px (positive renders outward).',
        )
        delta_ae = _dspin(
            minimum=-1000.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(error.get('delta_ae_px', 0.0)),
            tooltip='Radial-eccentricity-amplitude error in px.',
        )
        delta_peri = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(error.get('delta_long_peri_deg', 0.0)),
            tooltip='Pericenter-longitude error in degrees.',
        )
        form.addRow('delta a (px):', delta_a)
        form.addRow('delta ae (px):', delta_ae)
        form.addRow('delta long_peri (deg):', delta_peri)
        group.setChecked('orbit_error' in p)
        group.toggled.connect(lambda on, i=idx: self._on_ring_orbit_error_toggled(i, on))
        delta_a.valueChanged.connect(
            lambda v, i=idx: self._on_ring_orbit_error_value(i, 'delta_a_px', v)
        )
        delta_ae.valueChanged.connect(
            lambda v, i=idx: self._on_ring_orbit_error_value(i, 'delta_ae_px', v)
        )
        delta_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_orbit_error_value(i, 'delta_long_peri_deg', v)
        )
        w.orbit_error_group = group  # type: ignore[attr-defined]
        w.orbit_error_a_spin = delta_a  # type: ignore[attr-defined]
        w.orbit_error_ae_spin = delta_ae  # type: ignore[attr-defined]
        w.orbit_error_peri_spin = delta_peri  # type: ignore[attr-defined]
        return group

    def _on_ring_orbit_error_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the feature's orbit_error map."""
        feature = self._ring_feature(idx)
        w = self._ring_tab_widget(idx)
        if feature is None or w is None:
            return
        if checked:
            feature['orbit_error'] = {
                'delta_a_px': float(w.orbit_error_a_spin.value()),  # type: ignore[attr-defined]
                'delta_ae_px': float(w.orbit_error_ae_spin.value()),  # type: ignore[attr-defined]
                'delta_long_peri_deg': float(w.orbit_error_peri_spin.value()),  # type: ignore[attr-defined]
            }
        else:
            feature.pop('orbit_error', None)
        self._updater.request_update()

    def _on_ring_orbit_error_value(self, idx: int, key: str, value: float) -> None:
        """Update one orbit-error component when the map is present."""
        feature = self._ring_feature(idx)
        if feature is None:
            return
        error = feature.get('orbit_error')
        if isinstance(error, dict):
            error[key] = float(value)
            self._updater.request_update()

    # ---- Declared orbit sigma (idealized error bars) ----

    def _build_ring_sigma_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the declared orbit-uncertainty group (idealized error bars)."""
        sigma = _as_map(p.get('declared_orbit_sigma'))
        group = QGroupBox('Declared orbit sigma')
        group.setCheckable(True)
        group.setToolTip(
            'The catalog orbit uncertainty the navigator IS told (error bars '
            'only; never the drawn error values). Unchecked leaves the key '
            'absent.'
        )
        form = QFormLayout(group)
        sigma_a = _dspin(
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(sigma.get('sigma_a_px', 0.0)),
            tooltip='One-sigma semimajor-axis uncertainty in px.',
        )
        sigma_ae = _dspin(
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(sigma.get('sigma_ae_px', 0.0)),
            tooltip='One-sigma radial-eccentricity-amplitude uncertainty in px.',
        )
        sigma_peri = _dspin(
            minimum=0.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(sigma.get('sigma_long_peri_deg', 0.0)),
            tooltip='One-sigma pericenter-longitude uncertainty in degrees.',
        )
        form.addRow('sigma a (px):', sigma_a)
        form.addRow('sigma ae (px):', sigma_ae)
        form.addRow('sigma long_peri (deg):', sigma_peri)
        group.setChecked('declared_orbit_sigma' in p)
        group.toggled.connect(lambda on, i=idx: self._on_ring_sigma_toggled(i, on))
        sigma_a.valueChanged.connect(lambda v, i=idx: self._on_ring_sigma_value(i, 'sigma_a_px', v))
        sigma_ae.valueChanged.connect(
            lambda v, i=idx: self._on_ring_sigma_value(i, 'sigma_ae_px', v)
        )
        sigma_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_sigma_value(i, 'sigma_long_peri_deg', v)
        )
        w.orbit_sigma_group = group  # type: ignore[attr-defined]
        w.orbit_sigma_a_spin = sigma_a  # type: ignore[attr-defined]
        w.orbit_sigma_ae_spin = sigma_ae  # type: ignore[attr-defined]
        w.orbit_sigma_peri_spin = sigma_peri  # type: ignore[attr-defined]
        return group

    def _on_ring_sigma_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the feature's declared_orbit_sigma map."""
        feature = self._ring_feature(idx)
        w = self._ring_tab_widget(idx)
        if feature is None or w is None:
            return
        if checked:
            feature['declared_orbit_sigma'] = {
                'sigma_a_px': float(w.orbit_sigma_a_spin.value()),  # type: ignore[attr-defined]
                'sigma_ae_px': float(w.orbit_sigma_ae_spin.value()),  # type: ignore[attr-defined]
                'sigma_long_peri_deg': float(w.orbit_sigma_peri_spin.value()),  # type: ignore[attr-defined]
            }
        else:
            feature.pop('declared_orbit_sigma', None)
        self._updater.request_update()

    def _on_ring_sigma_value(self, idx: int, key: str, value: float) -> None:
        """Update one declared-sigma component when the map is present."""
        feature = self._ring_feature(idx)
        if feature is None:
            return
        sigma = feature.get('declared_orbit_sigma')
        if isinstance(sigma, dict):
            sigma[key] = float(value)
            self._updater.request_update()

    # ---- System-level azimuthal structure (truth) ----

    def _build_ring_azimuthal_group(self, w: QWidget) -> QGroupBox:
        """Build the system-level azimuthal-structure group (three sub-blocks).

        The ``azimuthal`` key exists exactly while at least one sub-block is
        enabled: enabling a sub-group inserts its sub-map (creating the block),
        disabling it removes the sub-map and prunes an emptied block.
        """
        ring_system = self._ring_system_map() or {}
        azimuthal = _as_map(ring_system.get('azimuthal'))
        group = QGroupBox('Azimuthal structure (truth)')
        group.setToolTip(
            'Non-navigable intensity structure crossing the features '
            '(albedo/illumination only, never tau); the navigator is never '
            'told about any of it.'
        )
        outer = QVBoxLayout(group)

        modulation = _as_map(azimuthal.get('modulation'))
        mod_group = QGroupBox('Brightness modulation')
        mod_group.setCheckable(True)
        mod_form = QFormLayout(mod_group)
        mod_amplitude = _dspin(
            minimum=0.0,
            maximum=10.0,
            decimals=3,
            step=0.05,
            value=float(modulation.get('amplitude', 0.2)),
            tooltip='Fractional intensity modulation amplitude.',
        )
        mod_m = QSpinBox()
        mod_m.setRange(1, 50)
        mod_m.setValue(int(modulation.get('m', 1)))
        mod_m.setToolTip('Azimuthal wavenumber, an integer of at least 1 (1 = one bright arc).')
        mod_phase = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(modulation.get('lam_peak_deg', 0.0)),
            tooltip=_key_tooltip('ring_system.azimuthal.modulation', 'lam_peak_deg'),
        )
        mod_form.addRow('Amplitude:', mod_amplitude)
        mod_form.addRow('Wavenumber m:', mod_m)
        mod_form.addRow('Peak longitude (deg):', mod_phase)
        mod_group.setChecked('modulation' in azimuthal)
        outer.addWidget(mod_group)

        shadow = _as_map(azimuthal.get('shadow'))
        shadow_group = QGroupBox('Planet-shadow wedge')
        shadow_group.setCheckable(True)
        shadow_form = QFormLayout(shadow_group)
        shadow_start = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(shadow.get('start_deg', 0.0)),
            tooltip='Leading edge of the shadow wedge, in ring-plane degrees.',
        )
        shadow_extent = _dspin(
            minimum=0.1,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(shadow.get('extent_deg', 30.0)),
            tooltip='Angular extent of the wedge in degrees.',
        )
        shadow_darkness = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
            value=float(shadow.get('darkness', 0.8)),
            tooltip='Fractional darkening inside the wedge (1 = fully dark).',
        )
        shadow_form.addRow('Start (deg):', shadow_start)
        shadow_form.addRow('Extent (deg):', shadow_extent)
        shadow_form.addRow('Darkness:', shadow_darkness)
        shadow_group.setChecked('shadow' in azimuthal)
        outer.addWidget(shadow_group)

        spokes = _as_map(azimuthal.get('spokes'))
        spokes_group = QGroupBox('Spokes')
        spokes_group.setCheckable(True)
        spokes_form = QFormLayout(spokes_group)
        spokes_count = QSpinBox()
        spokes_count.setRange(1, 100)
        spokes_count.setValue(int(spokes.get('count', 5)))
        spokes_count.setToolTip('Number of seeded spoke wedges.')
        spokes_r_inner = _dspin(
            minimum=0.1,
            maximum=100000.0,
            decimals=1,
            step=5.0,
            value=float(spokes.get('r_inner_px', 50.0)),
            tooltip=(
                f'{_key_tooltip("ring_system.azimuthal.spokes", "r_inner_px")} '
                'It must be below r_outer_px.'
            ),
        )
        spokes_r_outer = _dspin(
            minimum=0.1,
            maximum=100000.0,
            decimals=1,
            step=5.0,
            value=float(spokes.get('r_outer_px', 80.0)),
            tooltip=(
                f'{_key_tooltip("ring_system.azimuthal.spokes", "r_outer_px")} '
                'It must exceed r_inner_px.'
            ),
        )
        spokes_width = _dspin(
            minimum=0.1,
            maximum=360.0,
            decimals=1,
            step=1.0,
            value=float(spokes.get('width_deg', 10.0)),
            tooltip='Azimuthal full width of each spoke in degrees.',
        )
        spokes_contrast = _dspin(
            minimum=-5.0,
            maximum=5.0,
            decimals=3,
            step=0.05,
            value=float(spokes.get('contrast', -0.5)),
            tooltip='Fractional intensity contrast (negative for dark low-phase spokes).',
        )
        spokes_form.addRow('Count:', spokes_count)
        spokes_form.addRow('r inner (px):', spokes_r_inner)
        spokes_form.addRow('r outer (px):', spokes_r_outer)
        spokes_form.addRow('Width (deg):', spokes_width)
        spokes_form.addRow('Contrast:', spokes_contrast)
        spokes_group.setChecked('spokes' in azimuthal)
        outer.addWidget(spokes_group)

        mod_group.toggled.connect(self._on_ring_modulation_toggled)
        mod_amplitude.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('modulation', 'amplitude', float(v))
        )
        mod_m.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('modulation', 'm', int(v))
        )
        mod_phase.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('modulation', 'lam_peak_deg', float(v))
        )
        shadow_group.toggled.connect(self._on_ring_shadow_toggled)
        shadow_start.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('shadow', 'start_deg', float(v))
        )
        shadow_extent.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('shadow', 'extent_deg', float(v))
        )
        shadow_darkness.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('shadow', 'darkness', float(v))
        )
        spokes_group.toggled.connect(self._on_ring_spokes_toggled)
        spokes_count.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('spokes', 'count', int(v))
        )
        spokes_r_inner.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('spokes', 'r_inner_px', float(v))
        )
        spokes_r_outer.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('spokes', 'r_outer_px', float(v))
        )
        spokes_width.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('spokes', 'width_deg', float(v))
        )
        spokes_contrast.valueChanged.connect(
            lambda v: self._on_ring_azimuthal_value('spokes', 'contrast', float(v))
        )
        w.azimuthal_modulation_group = mod_group  # type: ignore[attr-defined]
        w.modulation_amplitude_spin = mod_amplitude  # type: ignore[attr-defined]
        w.modulation_m_spin = mod_m  # type: ignore[attr-defined]
        w.modulation_phase_spin = mod_phase  # type: ignore[attr-defined]
        w.azimuthal_shadow_group = shadow_group  # type: ignore[attr-defined]
        w.shadow_start_spin = shadow_start  # type: ignore[attr-defined]
        w.shadow_extent_spin = shadow_extent  # type: ignore[attr-defined]
        w.shadow_darkness_spin = shadow_darkness  # type: ignore[attr-defined]
        w.azimuthal_spokes_group = spokes_group  # type: ignore[attr-defined]
        w.spokes_count_spin = spokes_count  # type: ignore[attr-defined]
        w.spokes_r_inner_spin = spokes_r_inner  # type: ignore[attr-defined]
        w.spokes_r_outer_spin = spokes_r_outer  # type: ignore[attr-defined]
        w.spokes_width_spin = spokes_width  # type: ignore[attr-defined]
        w.spokes_contrast_spin = spokes_contrast  # type: ignore[attr-defined]
        w.azimuthal_group = group  # type: ignore[attr-defined]
        return group

    def _set_ring_azimuthal_sub(self, sub: str, block: dict[str, Any] | None) -> None:
        """Insert or remove one azimuthal sub-map, pruning an emptied block."""
        ring_system = self._ring_system_map()
        if ring_system is None:
            return
        if block is not None:
            azimuthal = ring_system.setdefault('azimuthal', {})
            azimuthal[sub] = block
        else:
            azimuthal = ring_system.get('azimuthal')
            if isinstance(azimuthal, dict):
                azimuthal.pop(sub, None)
                if not azimuthal:
                    ring_system.pop('azimuthal', None)
        self._updater.request_update()

    def _on_ring_modulation_toggled(self, checked: bool) -> None:
        """Insert or remove the azimuthal brightness-modulation sub-map."""
        w = self._ring_tab_widget(0)
        if w is None:
            return
        block = (
            {
                'amplitude': float(w.modulation_amplitude_spin.value()),  # type: ignore[attr-defined]
                'm': int(w.modulation_m_spin.value()),  # type: ignore[attr-defined]
                'lam_peak_deg': float(w.modulation_phase_spin.value()),  # type: ignore[attr-defined]
            }
            if checked
            else None
        )
        self._set_ring_azimuthal_sub('modulation', block)

    def _on_ring_shadow_toggled(self, checked: bool) -> None:
        """Insert or remove the azimuthal planet-shadow sub-map."""
        w = self._ring_tab_widget(0)
        if w is None:
            return
        block = (
            {
                'start_deg': float(w.shadow_start_spin.value()),  # type: ignore[attr-defined]
                'extent_deg': float(w.shadow_extent_spin.value()),  # type: ignore[attr-defined]
                'darkness': float(w.shadow_darkness_spin.value()),  # type: ignore[attr-defined]
            }
            if checked
            else None
        )
        self._set_ring_azimuthal_sub('shadow', block)

    def _on_ring_spokes_toggled(self, checked: bool) -> None:
        """Insert or remove the azimuthal spokes sub-map."""
        w = self._ring_tab_widget(0)
        if w is None:
            return
        block = (
            {
                'count': int(w.spokes_count_spin.value()),  # type: ignore[attr-defined]
                'r_inner_px': float(w.spokes_r_inner_spin.value()),  # type: ignore[attr-defined]
                'r_outer_px': float(w.spokes_r_outer_spin.value()),  # type: ignore[attr-defined]
                'width_deg': float(w.spokes_width_spin.value()),  # type: ignore[attr-defined]
                'contrast': float(w.spokes_contrast_spin.value()),  # type: ignore[attr-defined]
            }
            if checked
            else None
        )
        self._set_ring_azimuthal_sub('spokes', block)

    def _on_ring_azimuthal_value(self, sub: str, key: str, value: Any) -> None:
        """Update one azimuthal sub-map component when that sub-map is present."""
        ring_system = self._ring_system_map()
        if ring_system is None:
            return
        block = _as_map(ring_system.get('azimuthal')).get(sub)
        if isinstance(block, dict):
            block[key] = value
            self._updater.request_update()

    # ---- System-level moonlets (truth) ----

    def _build_ring_moonlets_group(self, w: QWidget) -> QGroupBox:
        """Build the moonlets group (a list of embedded-disc rows)."""
        ring_system = self._ring_system_map() or {}
        group = QGroupBox('Moonlets (truth)')
        group.setCheckable(True)
        group.setToolTip(
            'Opaque discs embedded at the ring depth, each optionally carving '
            'a propeller tau disturbance: blob/star confounders the navigator '
            'is never told about. Unchecked leaves the key absent.'
        )
        outer = QVBoxLayout(group)
        rows_layout = QVBoxLayout()
        outer.addLayout(rows_layout)
        add_btn = QPushButton('Add moonlet')
        outer.addWidget(add_btn)
        w.moonlet_rows = []  # type: ignore[attr-defined]
        w.moonlet_rows_layout = rows_layout  # type: ignore[attr-defined]
        for moonlet in ring_system.get('moonlets') or []:
            self._add_ring_moonlet_row(w, moonlet)
        group.setChecked('moonlets' in ring_system)
        group.toggled.connect(self._on_ring_moonlets_toggled)
        add_btn.clicked.connect(lambda _c=False: self._on_ring_add_moonlet())
        w.moonlets_group = group  # type: ignore[attr-defined]
        return group

    def _add_ring_moonlet_row(self, w: QWidget, entry: dict[str, Any]) -> None:
        """Append one moonlet row with its optional propeller sub-group."""
        propeller_present = isinstance(entry.get('propeller'), dict)
        propeller = _as_map(entry.get('propeller'))
        container = QGroupBox('Moonlet')
        vbox = QVBoxLayout(container)
        form = QFormLayout()
        vbox.addLayout(form)
        a_spin = _dspin(
            minimum=0.1,
            maximum=100000.0,
            decimals=1,
            step=5.0,
            value=float(entry.get('a_px', 60.0)),
            tooltip=_key_tooltip('ring_system.moonlets', 'a_px'),
        )
        lam_spin = _dspin(
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(entry.get('lam_deg', 0.0)),
            tooltip='Ring-plane longitude in degrees from the ascending node.',
        )
        radius_spin = _dspin(
            minimum=0.1,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(entry.get('radius_px', 1.5)),
            tooltip='Disc radius in px.',
        )
        amplitude_spin = _dspin(
            minimum=0.0,
            maximum=10.0,
            decimals=3,
            step=0.05,
            value=float(entry.get('amplitude', 0.4)),
            tooltip='Emitted intensity in normalized signal units.',
        )
        form.addRow('a (px):', a_spin)
        form.addRow('lam (deg):', lam_spin)
        form.addRow('radius (px):', radius_spin)
        form.addRow('amplitude:', amplitude_spin)

        propeller_group = QGroupBox('Propeller')
        propeller_group.setCheckable(True)
        propeller_form = QFormLayout(propeller_group)
        prop_length = _dspin(
            minimum=0.1,
            maximum=360.0,
            decimals=1,
            step=1.0,
            value=float(propeller.get('length_deg', 20.0)),
            tooltip='Azimuthal length of each tau lobe in degrees.',
        )
        prop_width = _dspin(
            minimum=0.1,
            maximum=1000.0,
            decimals=2,
            step=0.5,
            value=float(propeller.get('width_px', 2.0)),
            tooltip='Radial width of each tau lobe in px.',
        )
        prop_contrast = _dspin(
            minimum=-5.0,
            maximum=5.0,
            decimals=3,
            step=0.05,
            value=float(propeller.get('contrast', -0.6)),
            tooltip='Fractional tau contrast of the lobes (negative carves partial gaps).',
        )
        propeller_form.addRow('length (deg):', prop_length)
        propeller_form.addRow('width (px):', prop_width)
        propeller_form.addRow('contrast:', prop_contrast)
        propeller_group.setChecked(propeller_present)
        vbox.addWidget(propeller_group)

        remove_btn = QPushButton('Remove moonlet')
        vbox.addWidget(remove_btn)

        moonlet_row = _MoonletRow(
            container,
            a_spin,
            lam_spin,
            radius_spin,
            amplitude_spin,
            propeller_group,
            prop_length,
            prop_width,
            prop_contrast,
        )
        w.moonlet_rows.append(moonlet_row)  # type: ignore[attr-defined]
        w.moonlet_rows_layout.addWidget(container)  # type: ignore[attr-defined]
        propeller_group.toggled.connect(lambda _v: self._rewrite_ring_moonlets())
        for spin in (
            a_spin,
            lam_spin,
            radius_spin,
            amplitude_spin,
            prop_length,
            prop_width,
            prop_contrast,
        ):
            spin.valueChanged.connect(lambda _v: self._rewrite_ring_moonlets())
        remove_btn.clicked.connect(lambda _c=False, r=moonlet_row: self._remove_ring_moonlet_row(r))

    def _moonlet_list_from_rows(self, rows: list[_MoonletRow]) -> list[dict[str, Any]]:
        """Read moonlet rows into the schema list of moonlet entries."""
        entries: list[dict[str, Any]] = []
        for r in rows:
            entry: dict[str, Any] = {
                'a_px': float(r.a_spin.value()),
                'lam_deg': float(r.lam_spin.value()),
                'radius_px': float(r.radius_spin.value()),
                'amplitude': float(r.amplitude_spin.value()),
            }
            if r.propeller_group.isChecked():
                entry['propeller'] = {
                    'length_deg': float(r.prop_length_spin.value()),
                    'width_px': float(r.prop_width_spin.value()),
                    'contrast': float(r.prop_contrast_spin.value()),
                }
            entries.append(entry)
        return entries

    def _on_ring_moonlets_toggled(self, checked: bool) -> None:
        """Insert or remove the system's moonlets list."""
        ring_system = self._ring_system_map()
        w = self._ring_tab_widget(0)
        if ring_system is None or w is None:
            return
        if checked:
            ring_system['moonlets'] = self._moonlet_list_from_rows(w.moonlet_rows)  # type: ignore[attr-defined]
        else:
            ring_system.pop('moonlets', None)
        self._updater.request_update()

    def _on_ring_add_moonlet(self) -> None:
        """Append a moonlet row and rewrite the list."""
        w = self._ring_tab_widget(0)
        if w is None:
            return
        self._add_ring_moonlet_row(w, {})
        self._rewrite_ring_moonlets()

    def _rewrite_ring_moonlets(self) -> None:
        """Rewrite the moonlets list when it is present."""
        ring_system = self._ring_system_map()
        w = self._ring_tab_widget(0)
        if ring_system is None or w is None:
            return
        if isinstance(ring_system.get('moonlets'), list):
            ring_system['moonlets'] = self._moonlet_list_from_rows(w.moonlet_rows)  # type: ignore[attr-defined]
            self._updater.request_update()

    def _remove_ring_moonlet_row(self, row: _MoonletRow) -> None:
        """Remove a moonlet row and rewrite the list."""
        w = self._ring_tab_widget(0)
        rows = getattr(w, 'moonlet_rows', None) if w is not None else None
        if rows is not None and row in rows:
            rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._rewrite_ring_moonlets()
