"""Body appearance groups: relief, photometry, surface texture, and mesh extras.

Splits the per-body tab's truth-side appearance controls out of the geometry
tab so neither module runs long.  Every group here follows the editor-wide
absent-key discipline: an enable checkbox (or a combo's default choice) writes
its block only when active, an inactive control leaves its key absent, and each
widget writes only its own key.  The groups cover the topographic renderer's
truth keys -- the limb-relief field, the photometric law and opposition surge,
the multiplicative albedo texture and giant-planet disc texture, transiting
moons and their shadows -- and the mesh-only shading / detail / pose-scatter
extras, which are enabled only for a polyhedral-mesh body.  The atmosphere
(haze) group lives in its own sibling module
(:mod:`spindoctor.cli.sim_editor.body_atmosphere`); this module's group
builder list places it among the others.

Every widget reference the round-trip and per-widget tests reach for is stored
on the tab widget ``w`` (``w.relief_group``, ``w.spot_rows``, and so on), so a
test can drive one control and assert the resulting ``sim_params`` edit.
"""

from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
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
from spindoctor.sim.forward.photometry import PHOTOMETRIC_LAWS

# The photometric-law vocabulary, Lambert first so the combo opens on the
# navigator-matched default when a scene enables the group without a law.
_LAW_ORDER = ('lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert')
_LAWS = tuple(law for law in _LAW_ORDER if law in PHOTOMETRIC_LAWS)


def _as_map(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list, else an empty list."""
    return value if isinstance(value, list) else []


@dataclass
class _SpotRow:
    """The widgets of one albedo-spot / storm-oval entry row."""

    container: QWidget
    lat_spin: QDoubleSpinBox
    lon_spin: QDoubleSpinBox
    radius_spin: QDoubleSpinBox
    factor_spin: QDoubleSpinBox


@dataclass
class _TransitRow:
    """The widgets of one transit entry: a moon sub-group and a shadow sub-group."""

    container: QWidget
    moon_group: QGroupBox
    moon_dv: QDoubleSpinBox
    moon_du: QDoubleSpinBox
    moon_radius: QDoubleSpinBox
    moon_factor: QDoubleSpinBox
    shadow_group: QGroupBox
    shadow_dv: QDoubleSpinBox
    shadow_du: QDoubleSpinBox
    shadow_radius: QDoubleSpinBox
    shadow_darkness: QDoubleSpinBox


class BodyAppearanceMixin(SimEditorBase):
    """Builds and handles the per-body appearance groups."""

    # ---- Shared helpers ----

    def _body(self, idx: int) -> dict[str, Any] | None:
        """Return the body dict at ``idx``, or None when out of range."""
        bodies = self.sim_params['bodies']
        if 0 <= idx < len(bodies):
            body: dict[str, Any] = bodies[idx]
            return body
        return None

    def _body_tab_widget(self, idx: int) -> QWidget | None:
        """Return the built tab widget for the body at ``idx``, or None."""
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is None:
            return None
        return self._tabs.widget(tab_idx)

    def _build_body_appearance_groups(self, w: QWidget, idx: int, layout: QVBoxLayout) -> None:
        """Add every appearance group to a body tab, storing refs on ``w``."""
        p = self.sim_params['bodies'][idx]
        layout.addWidget(self._build_limb_relief_group(w, idx, p))
        layout.addWidget(self._build_photometry_group(w, idx, p))
        layout.addWidget(self._build_opposition_surge_group(w, idx, p))
        layout.addWidget(self._build_albedo_texture_group(w, idx, p))
        layout.addWidget(self._build_disc_texture_group(w, idx, p))
        layout.addWidget(self._build_transits_group(w, idx, p))
        layout.addWidget(self._build_atmosphere_group(w, idx, p))
        mesh_group = self._build_mesh_extras_group(w, idx, p)
        layout.addWidget(mesh_group)
        w.mesh_extras_group = mesh_group  # type: ignore[attr-defined]
        is_mesh = str(p.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh'
        mesh_group.setEnabled(is_mesh)

    # ---- Limb relief ----

    def _build_limb_relief_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the limb-relief field group (rms + correlation length)."""
        group = QGroupBox('Limb relief')
        group.setCheckable(True)
        form = QFormLayout(group)
        rms = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=4,
            step=0.005,
            value=float(p.get('limb_relief_rms', 0.0)),
            tooltip='Fractional relief amplitude (0 disables the field).',
        )
        corr = _dspin(
            minimum=0.1,
            maximum=180.0,
            decimals=1,
            step=1.0,
            value=float(p.get('limb_relief_corr_deg', 15.0)),
            tooltip='Relief correlation length in degrees of surface arc.',
        )
        form.addRow('Relief RMS:', rms)
        form.addRow('Correlation (deg):', corr)
        group.setChecked('limb_relief_rms' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_relief_toggled(i, on))
        rms.valueChanged.connect(
            lambda v, i=idx: self._on_body_relief_value(i, 'limb_relief_rms', v)
        )
        corr.valueChanged.connect(
            lambda v, i=idx: self._on_body_relief_value(i, 'limb_relief_corr_deg', v)
        )
        w.relief_group = group  # type: ignore[attr-defined]
        w.relief_rms_spin = rms  # type: ignore[attr-defined]
        w.relief_corr_spin = corr  # type: ignore[attr-defined]
        return group

    def _on_body_relief_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's limb-relief keys."""
        body = self._body(idx)
        if body is None:
            return
        if checked:
            w = self._body_tab_widget(idx)
            body['limb_relief_rms'] = float(w.relief_rms_spin.value()) if w is not None else 0.0  # type: ignore[attr-defined]
            body['limb_relief_corr_deg'] = (
                float(w.relief_corr_spin.value()) if w is not None else 15.0  # type: ignore[attr-defined]
            )
        else:
            body.pop('limb_relief_rms', None)
            body.pop('limb_relief_corr_deg', None)
        self._updater.request_update()

    def _on_body_relief_value(self, idx: int, key: str, value: float) -> None:
        """Update one limb-relief component when the group is enabled."""
        body = self._body(idx)
        if body is not None and key in body:
            body[key] = float(value)
            self._updater.request_update()

    # ---- Photometric law ----

    def _build_photometry_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the photometric-law group (law combo + Minnaert exponent)."""
        group = QGroupBox('Photometric law')
        group.setCheckable(True)
        form = QFormLayout(group)
        combo = QComboBox()
        combo.addItems(list(_LAWS))
        law_idx = combo.findText(str(p.get('photometric_law', 'lambert')))
        if law_idx >= 0:
            combo.setCurrentIndex(law_idx)
        minnaert_k = _dspin(
            minimum=0.0,
            maximum=2.0,
            decimals=3,
            step=0.05,
            value=float(p.get('minnaert_k', 0.5)),
            tooltip='Minnaert exponent (k = 1 is Lambert); used only by the minnaert law.',
        )
        minnaert_k.setEnabled(combo.currentText() == 'minnaert')
        form.addRow('Law:', combo)
        form.addRow('Minnaert k:', minnaert_k)
        group.setChecked('photometric_law' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_photometry_toggled(i, on))
        combo.currentTextChanged.connect(lambda t, i=idx: self._on_body_photometric_law(i, t))
        minnaert_k.valueChanged.connect(lambda v, i=idx: self._on_body_minnaert_k(i, v))
        w.photometry_group = group  # type: ignore[attr-defined]
        w.photometry_law_combo = combo  # type: ignore[attr-defined]
        w.minnaert_k_spin = minnaert_k  # type: ignore[attr-defined]
        return group

    def _on_body_photometry_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's photometric-law keys."""
        body = self._body(idx)
        if body is None:
            return
        w = self._body_tab_widget(idx)
        if checked:
            law = w.photometry_law_combo.currentText() if w is not None else 'lambert'  # type: ignore[attr-defined]
            body['photometric_law'] = law
            if law == 'minnaert' and w is not None:
                body['minnaert_k'] = float(w.minnaert_k_spin.value())  # type: ignore[attr-defined]
        else:
            body.pop('photometric_law', None)
            body.pop('minnaert_k', None)
        self._updater.request_update()

    def _on_body_photometric_law(self, idx: int, law: str) -> None:
        """Write the chosen law and gate the Minnaert exponent to that law."""
        body = self._body(idx)
        if body is None or 'photometric_law' not in body:
            return
        body['photometric_law'] = law
        w = self._body_tab_widget(idx)
        if law == 'minnaert':
            if w is not None:
                w.minnaert_k_spin.setEnabled(True)  # type: ignore[attr-defined]
                body['minnaert_k'] = float(w.minnaert_k_spin.value())  # type: ignore[attr-defined]
        else:
            body.pop('minnaert_k', None)
            if w is not None:
                w.minnaert_k_spin.setEnabled(False)  # type: ignore[attr-defined]
        self._updater.request_update()

    def _on_body_minnaert_k(self, idx: int, value: float) -> None:
        """Update the Minnaert exponent when it is present."""
        body = self._body(idx)
        if body is not None and 'minnaert_k' in body:
            body['minnaert_k'] = float(value)
            self._updater.request_update()

    # ---- Opposition surge ----

    def _build_opposition_surge_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the opposition-surge group (amplitude + angular width)."""
        group = QGroupBox('Opposition surge')
        group.setCheckable(True)
        form = QFormLayout(group)
        surge = _as_map(p.get('opposition_surge'))
        amplitude = _dspin(
            minimum=0.0,
            maximum=10.0,
            decimals=3,
            step=0.05,
            value=float(surge.get('amplitude', 0.3)),
            tooltip='Surge amplitude at exact opposition (0 disables the factor).',
        )
        width = _dspin(
            minimum=0.1,
            maximum=90.0,
            decimals=2,
            step=0.5,
            value=float(surge.get('width_deg', 6.0)),
            tooltip='Angular e-folding width of the surge, in degrees of phase.',
        )
        form.addRow('Amplitude:', amplitude)
        form.addRow('Width (deg):', width)
        group.setChecked('opposition_surge' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_surge_toggled(i, on))
        amplitude.valueChanged.connect(
            lambda v, i=idx: self._on_body_surge_value(i, 'amplitude', v)
        )
        width.valueChanged.connect(lambda v, i=idx: self._on_body_surge_value(i, 'width_deg', v))
        w.surge_group = group  # type: ignore[attr-defined]
        w.surge_amplitude_spin = amplitude  # type: ignore[attr-defined]
        w.surge_width_spin = width  # type: ignore[attr-defined]
        return group

    def _on_body_surge_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's opposition-surge map."""
        body = self._body(idx)
        if body is None:
            return
        if checked:
            w = self._body_tab_widget(idx)
            body['opposition_surge'] = {
                'amplitude': float(w.surge_amplitude_spin.value()) if w is not None else 0.3,  # type: ignore[attr-defined]
                'width_deg': float(w.surge_width_spin.value()) if w is not None else 6.0,  # type: ignore[attr-defined]
            }
        else:
            body.pop('opposition_surge', None)
        self._updater.request_update()

    def _on_body_surge_value(self, idx: int, key: str, value: float) -> None:
        """Update one opposition-surge component when the map is present."""
        body = self._body(idx)
        if body is None:
            return
        surge = body.get('opposition_surge')
        if isinstance(surge, dict):
            surge[key] = float(value)
            self._updater.request_update()

    # ---- Albedo texture ----

    def _build_albedo_texture_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the albedo-texture group (noise field + a spots list)."""
        block = _as_map(p.get('albedo_texture'))
        group = QGroupBox('Albedo texture')
        group.setCheckable(True)
        outer = QVBoxLayout(group)
        form = QFormLayout()
        outer.addLayout(form)
        rms = _dspin(
            minimum=0.0,
            maximum=2.0,
            decimals=4,
            step=0.01,
            value=float(block.get('rms', 0.0)),
            tooltip='Global RMS of the multiplicative noise field (0 disables it).',
        )
        corr_px = _dspin(
            minimum=0.0,
            maximum=1000.0,
            decimals=2,
            step=1.0,
            value=float(block.get('corr_px', 20.0)),
            tooltip='Noise correlation length in detector pixels on the disc.',
        )
        form.addRow('Noise RMS:', rms)
        form.addRow('Correlation (px):', corr_px)

        rows_layout = QVBoxLayout()
        outer.addLayout(rows_layout)
        add_btn = QPushButton('Add albedo spot')
        outer.addWidget(add_btn)
        w.spot_rows = []  # type: ignore[attr-defined]
        w.spot_rows_layout = rows_layout  # type: ignore[attr-defined]
        for spot in block.get('spots') or []:
            self._add_spot_row(w, idx, spot, kind='spot')

        group.setChecked('albedo_texture' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_albedo_toggled(i, on))
        rms.valueChanged.connect(lambda v, i=idx: self._on_body_albedo_scalar(i, 'rms', v))
        corr_px.valueChanged.connect(lambda v, i=idx: self._on_body_albedo_scalar(i, 'corr_px', v))
        add_btn.clicked.connect(lambda _c=False, i=idx: self._on_body_add_spot(i))
        w.albedo_group = group  # type: ignore[attr-defined]
        w.albedo_rms_spin = rms  # type: ignore[attr-defined]
        w.albedo_corr_spin = corr_px  # type: ignore[attr-defined]
        return group

    def _on_body_albedo_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's albedo-texture map."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        if checked:
            body['albedo_texture'] = {
                'rms': float(w.albedo_rms_spin.value()),  # type: ignore[attr-defined]
                'corr_px': float(w.albedo_corr_spin.value()),  # type: ignore[attr-defined]
                'spots': self._spot_list_from_rows(w.spot_rows),  # type: ignore[attr-defined]
            }
        else:
            body.pop('albedo_texture', None)
        self._updater.request_update()

    def _on_body_albedo_scalar(self, idx: int, key: str, value: float) -> None:
        """Update one albedo-texture scalar when the map is present."""
        body = self._body(idx)
        if body is None:
            return
        block = body.get('albedo_texture')
        if isinstance(block, dict):
            block[key] = float(value)
            self._updater.request_update()

    def _on_body_add_spot(self, idx: int) -> None:
        """Append an albedo spot row and rewrite the spots list."""
        w = self._body_tab_widget(idx)
        if w is None:
            return
        self._add_spot_row(w, idx, {}, kind='spot')
        self._rewrite_spots(idx)

    # ---- Disc texture ----

    def _build_disc_texture_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the disc-texture group (latitude bands + a storms list)."""
        block = _as_map(p.get('disc_texture'))
        group = QGroupBox('Disc texture (bands and storms)')
        group.setCheckable(True)
        outer = QVBoxLayout(group)
        form = QFormLayout()
        outer.addLayout(form)
        amplitude = _dspin(
            minimum=0.0,
            maximum=2.0,
            decimals=4,
            step=0.01,
            value=float(block.get('band_amplitude', 0.0)),
            tooltip='Multiplicative contrast of the banded pattern (0 disables it).',
        )
        wavenumber = _dspin(
            minimum=0.0,
            maximum=64.0,
            decimals=2,
            step=0.5,
            value=float(block.get('band_wavenumber', 8.0)),
            tooltip='Cosine cycles per radian of body-polar latitude.',
        )
        phase = _dspin(
            minimum=0.0,
            maximum=360.0,
            decimals=1,
            step=5.0,
            value=float(block.get('band_phase_deg', 0.0)),
            tooltip='Phase offset of the band pattern, in degrees.',
        )
        form.addRow('Band amplitude:', amplitude)
        form.addRow('Band wavenumber:', wavenumber)
        form.addRow('Band phase (deg):', phase)

        rows_layout = QVBoxLayout()
        outer.addLayout(rows_layout)
        add_btn = QPushButton('Add storm')
        outer.addWidget(add_btn)
        w.storm_rows = []  # type: ignore[attr-defined]
        w.storm_rows_layout = rows_layout  # type: ignore[attr-defined]
        for storm in block.get('storms') or []:
            self._add_spot_row(w, idx, storm, kind='storm')

        group.setChecked('disc_texture' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_disc_toggled(i, on))
        amplitude.valueChanged.connect(
            lambda v, i=idx: self._on_body_disc_scalar(i, 'band_amplitude', v)
        )
        wavenumber.valueChanged.connect(
            lambda v, i=idx: self._on_body_disc_scalar(i, 'band_wavenumber', v)
        )
        phase.valueChanged.connect(
            lambda v, i=idx: self._on_body_disc_scalar(i, 'band_phase_deg', v)
        )
        add_btn.clicked.connect(lambda _c=False, i=idx: self._on_body_add_storm(i))
        w.disc_group = group  # type: ignore[attr-defined]
        w.disc_amplitude_spin = amplitude  # type: ignore[attr-defined]
        w.disc_wavenumber_spin = wavenumber  # type: ignore[attr-defined]
        w.disc_phase_spin = phase  # type: ignore[attr-defined]
        return group

    def _on_body_disc_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's disc-texture map."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        if checked:
            body['disc_texture'] = {
                'band_amplitude': float(w.disc_amplitude_spin.value()),  # type: ignore[attr-defined]
                'band_wavenumber': float(w.disc_wavenumber_spin.value()),  # type: ignore[attr-defined]
                'band_phase_deg': float(w.disc_phase_spin.value()),  # type: ignore[attr-defined]
                'storms': self._spot_list_from_rows(w.storm_rows),  # type: ignore[attr-defined]
            }
        else:
            body.pop('disc_texture', None)
        self._updater.request_update()

    def _on_body_disc_scalar(self, idx: int, key: str, value: float) -> None:
        """Update one disc-texture scalar when the map is present."""
        body = self._body(idx)
        if body is None:
            return
        block = body.get('disc_texture')
        if isinstance(block, dict):
            block[key] = float(value)
            self._updater.request_update()

    def _on_body_add_storm(self, idx: int) -> None:
        """Append a storm row and rewrite the storms list."""
        w = self._body_tab_widget(idx)
        if w is None:
            return
        self._add_spot_row(w, idx, {}, kind='storm')
        self._rewrite_storms(idx)

    # ---- Shared spot / storm row machinery ----

    def _add_spot_row(self, w: QWidget, idx: int, entry: dict[str, Any], *, kind: str) -> None:
        """Append one spot/storm row (lat, lon, radius, albedo factor)."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        lat = _dspin(
            minimum=-90.0,
            maximum=90.0,
            decimals=1,
            step=1.0,
            value=float(entry.get('lat_deg', 0.0)),
        )
        lon = _dspin(
            minimum=0.0,
            maximum=360.0,
            decimals=1,
            step=1.0,
            value=float(entry.get('lon_deg', 90.0)),
        )
        radius = _dspin(
            minimum=0.0,
            maximum=180.0,
            decimals=1,
            step=1.0,
            value=float(entry.get('radius_deg', 10.0)),
        )
        factor = _dspin(
            minimum=0.0,
            maximum=5.0,
            decimals=3,
            step=0.05,
            value=float(entry.get('albedo_factor', 0.7)),
        )
        remove_btn = QPushButton('Remove')
        for label, widget in (('lat', lat), ('lon', lon), ('r', radius), ('x', factor)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        row.addWidget(remove_btn)
        spot_row = _SpotRow(container, lat, lon, radius, factor)
        if kind == 'spot':
            w.spot_rows.append(spot_row)  # type: ignore[attr-defined]
            w.spot_rows_layout.addWidget(container)  # type: ignore[attr-defined]
            for widget in (lat, lon, radius, factor):
                widget.valueChanged.connect(lambda _v, i=idx: self._rewrite_spots(i))
            remove_btn.clicked.connect(
                lambda _c=False, i=idx, r=spot_row: self._remove_spot_row(i, r)
            )
        else:
            w.storm_rows.append(spot_row)  # type: ignore[attr-defined]
            w.storm_rows_layout.addWidget(container)  # type: ignore[attr-defined]
            for widget in (lat, lon, radius, factor):
                widget.valueChanged.connect(lambda _v, i=idx: self._rewrite_storms(i))
            remove_btn.clicked.connect(
                lambda _c=False, i=idx, r=spot_row: self._remove_storm_row(i, r)
            )

    def _spot_list_from_rows(self, rows: list[_SpotRow]) -> list[dict[str, Any]]:
        """Read spot/storm rows into their schema list of dicts."""
        return [
            {
                'lat_deg': float(r.lat_spin.value()),
                'lon_deg': float(r.lon_spin.value()),
                'radius_deg': float(r.radius_spin.value()),
                'albedo_factor': float(r.factor_spin.value()),
            }
            for r in rows
        ]

    def _rewrite_spots(self, idx: int) -> None:
        """Rewrite the albedo spots list when the map is present."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        block = body.get('albedo_texture')
        if isinstance(block, dict):
            block['spots'] = self._spot_list_from_rows(w.spot_rows)  # type: ignore[attr-defined]
            self._updater.request_update()

    def _remove_spot_row(self, idx: int, row: _SpotRow) -> None:
        """Remove an albedo spot row and rewrite the spots list."""
        w = self._body_tab_widget(idx)
        rows = getattr(w, 'spot_rows', None) if w is not None else None
        if rows is not None and row in rows:
            rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._rewrite_spots(idx)

    def _rewrite_storms(self, idx: int) -> None:
        """Rewrite the disc storms list when the map is present."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        block = body.get('disc_texture')
        if isinstance(block, dict):
            block['storms'] = self._spot_list_from_rows(w.storm_rows)  # type: ignore[attr-defined]
            self._updater.request_update()

    def _remove_storm_row(self, idx: int, row: _SpotRow) -> None:
        """Remove a storm row and rewrite the storms list."""
        w = self._body_tab_widget(idx)
        rows = getattr(w, 'storm_rows', None) if w is not None else None
        if rows is not None and row in rows:
            rows.remove(row)
        row.container.setParent(None)
        row.container.deleteLater()
        self._rewrite_storms(idx)

    # ---- Transits ----

    def _build_transits_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the transits group (a list of moon and/or shadow discs)."""
        group = QGroupBox('Transits')
        group.setCheckable(True)
        outer = QVBoxLayout(group)
        rows_layout = QVBoxLayout()
        outer.addLayout(rows_layout)
        add_btn = QPushButton('Add transit')
        outer.addWidget(add_btn)
        w.transit_rows = []  # type: ignore[attr-defined]
        w.transit_rows_layout = rows_layout  # type: ignore[attr-defined]
        entries = _as_list(p.get('transits'))
        for entry in entries:
            self._add_transit_row(w, idx, entry)

        group.setChecked('transits' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_transits_toggled(i, on))
        add_btn.clicked.connect(lambda _c=False, i=idx: self._on_body_add_transit(i))
        w.transits_group = group  # type: ignore[attr-defined]
        return group

    def _add_transit_row(self, w: QWidget, idx: int, entry: dict[str, Any]) -> None:
        """Append one transit row with a moon sub-group and a shadow sub-group."""
        moon_present = isinstance(entry.get('moon'), dict)
        shadow_present = isinstance(entry.get('shadow'), dict)
        moon = _as_map(entry.get('moon'))
        shadow = _as_map(entry.get('shadow'))
        container = QGroupBox('Transit')
        vbox = QVBoxLayout(container)

        moon_group = QGroupBox('Moon disc')
        moon_group.setCheckable(True)
        moon_form = QFormLayout(moon_group)
        moon_dv = _dspin(
            minimum=-5000.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(moon.get('dv_px', 0.0)),
        )
        moon_du = _dspin(
            minimum=-5000.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(moon.get('du_px', 0.0)),
        )
        moon_radius = _dspin(
            minimum=0.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(moon.get('radius_px', 5.0)),
        )
        moon_factor = _dspin(
            minimum=0.0,
            maximum=5.0,
            decimals=3,
            step=0.05,
            value=float(moon.get('albedo_factor', 1.0)),
        )
        moon_form.addRow('dv (px):', moon_dv)
        moon_form.addRow('du (px):', moon_du)
        moon_form.addRow('radius (px):', moon_radius)
        moon_form.addRow('albedo factor:', moon_factor)
        # A brand-new transit (neither sub-block present) defaults to a moon
        # disc so the entry is valid; an existing entry reflects what it carries.
        moon_group.setChecked(moon_present or not shadow_present)
        vbox.addWidget(moon_group)

        shadow_group = QGroupBox('Cast shadow')
        shadow_group.setCheckable(True)
        shadow_form = QFormLayout(shadow_group)
        shadow_dv = _dspin(
            minimum=-5000.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(shadow.get('dv_px', 0.0)),
        )
        shadow_du = _dspin(
            minimum=-5000.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(shadow.get('du_px', 0.0)),
        )
        shadow_radius = _dspin(
            minimum=0.0,
            maximum=5000.0,
            decimals=2,
            step=1.0,
            value=float(shadow.get('radius_px', 5.0)),
        )
        shadow_darkness = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
            value=float(shadow.get('darkness', 0.8)),
        )
        shadow_form.addRow('dv (px):', shadow_dv)
        shadow_form.addRow('du (px):', shadow_du)
        shadow_form.addRow('radius (px):', shadow_radius)
        shadow_form.addRow('darkness:', shadow_darkness)
        shadow_group.setChecked(shadow_present)
        vbox.addWidget(shadow_group)

        remove_btn = QPushButton('Remove transit')
        vbox.addWidget(remove_btn)

        transit_row = _TransitRow(
            container,
            moon_group,
            moon_dv,
            moon_du,
            moon_radius,
            moon_factor,
            shadow_group,
            shadow_dv,
            shadow_du,
            shadow_radius,
            shadow_darkness,
        )
        w.transit_rows.append(transit_row)  # type: ignore[attr-defined]
        w.transit_rows_layout.addWidget(container)  # type: ignore[attr-defined]
        for widget in (
            moon_group,
            shadow_group,
            moon_dv,
            moon_du,
            moon_radius,
            moon_factor,
            shadow_dv,
            shadow_du,
            shadow_radius,
            shadow_darkness,
        ):
            if isinstance(widget, QGroupBox):
                widget.toggled.connect(lambda _v, i=idx: self._rewrite_transits(i))
            else:
                widget.valueChanged.connect(lambda _v, i=idx: self._rewrite_transits(i))
        remove_btn.clicked.connect(
            lambda _c=False, i=idx, r=transit_row: self._remove_transit_row(i, r)
        )

    def _transit_list_from_rows(self, rows: list[_TransitRow]) -> list[dict[str, Any]]:
        """Read transit rows into the schema list of moon/shadow entries."""
        entries: list[dict[str, Any]] = []
        for r in rows:
            entry: dict[str, Any] = {}
            if r.moon_group.isChecked():
                entry['moon'] = {
                    'dv_px': float(r.moon_dv.value()),
                    'du_px': float(r.moon_du.value()),
                    'radius_px': float(r.moon_radius.value()),
                    'albedo_factor': float(r.moon_factor.value()),
                }
            if r.shadow_group.isChecked():
                entry['shadow'] = {
                    'dv_px': float(r.shadow_dv.value()),
                    'du_px': float(r.shadow_du.value()),
                    'radius_px': float(r.shadow_radius.value()),
                    'darkness': float(r.shadow_darkness.value()),
                }
            entries.append(entry)
        return entries

    def _on_body_transits_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's transits list."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        if checked:
            body['transits'] = self._transit_list_from_rows(w.transit_rows)  # type: ignore[attr-defined]
        else:
            body.pop('transits', None)
        self._updater.request_update()

    def _on_body_add_transit(self, idx: int) -> None:
        """Append a transit row and rewrite the list."""
        w = self._body_tab_widget(idx)
        if w is None:
            return
        self._add_transit_row(w, idx, {})
        self._rewrite_transits(idx)

    def _rewrite_transits(self, idx: int) -> None:
        """Rewrite the transits list when it is present."""
        body = self._body(idx)
        w = self._body_tab_widget(idx)
        if body is None or w is None:
            return
        if isinstance(body.get('transits'), list):
            body['transits'] = self._transit_list_from_rows(w.transit_rows)  # type: ignore[attr-defined]
            self._updater.request_update()

    def _remove_transit_row(self, idx: int, row: _TransitRow) -> None:
        """Remove a transit row and rewrite the list."""
        w = self._body_tab_widget(idx)
        if w is not None and row in w.transit_rows:  # type: ignore[attr-defined]
            w.transit_rows.remove(row)  # type: ignore[attr-defined]
        row.container.setParent(None)
        row.container.deleteLater()
        self._rewrite_transits(idx)

    # ---- Mesh extras (shading, detail octaves, pose scatter) ----

    def _build_mesh_extras_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the mesh-only group (shading mode, detail octaves, pose scatter).

        The whole group is enabled only for a polyhedral-mesh body; on an
        ellipsoid these keys have no rendering effect, so the group is grayed
        out and its keys stay absent.
        """
        group = QGroupBox('Mesh shading and detail')
        form = QFormLayout(group)
        shading = QComboBox()
        shading.addItems(['flat', 'gouraud'])
        shading_idx = shading.findText(str(p.get('shading', 'flat')))
        if shading_idx >= 0:
            shading.setCurrentIndex(shading_idx)
        shading.setToolTip('Rendered mesh shading; flat leaves the key absent.')
        form.addRow('Shading:', shading)
        octaves = QSpinBox()
        octaves.setRange(0, 6)
        octaves.setValue(int(p.get('mesh_detail_octaves', 0)))
        octaves.setToolTip('Higher-frequency mesh detail banks; 0 leaves the key absent.')
        form.addRow('Detail octaves:', octaves)

        scatter = _as_map(p.get('pose_scatter'))
        scatter_group = QGroupBox('Pose scatter')
        scatter_group.setCheckable(True)
        scatter_form = QFormLayout(scatter_group)
        sigma = _dspin(
            minimum=0.0,
            maximum=45.0,
            decimals=3,
            step=0.5,
            value=float(scatter.get('sigma_deg', 1.0)),
            tooltip='Per-frame Gaussian pose perturbation, sigma per Euler axis (deg).',
        )
        scatter_form.addRow('Sigma (deg):', sigma)
        scatter_group.setChecked('pose_scatter' in p)
        form.addRow(scatter_group)

        shading.currentTextChanged.connect(lambda t, i=idx: self._on_body_shading(i, t))
        octaves.valueChanged.connect(lambda v, i=idx: self._on_body_mesh_octaves(i, v))
        scatter_group.toggled.connect(lambda on, i=idx: self._on_body_pose_scatter_toggled(i, on))
        sigma.valueChanged.connect(lambda v, i=idx: self._on_body_pose_scatter_value(i, v))
        w.shading_combo = shading  # type: ignore[attr-defined]
        w.mesh_octaves_spin = octaves  # type: ignore[attr-defined]
        w.pose_scatter_group = scatter_group  # type: ignore[attr-defined]
        w.pose_scatter_sigma_spin = sigma  # type: ignore[attr-defined]
        return group

    def _on_body_shading(self, idx: int, mode: str) -> None:
        """Write gouraud shading, or drop the key for the flat default."""
        body = self._body(idx)
        if body is None:
            return
        if mode == 'flat':
            body.pop('shading', None)
        else:
            body['shading'] = mode
        self._updater.request_update()

    def _on_body_mesh_octaves(self, idx: int, value: int) -> None:
        """Write the mesh detail octaves, or drop the key at the default 0."""
        body = self._body(idx)
        if body is None:
            return
        if int(value) <= 0:
            body.pop('mesh_detail_octaves', None)
        else:
            body['mesh_detail_octaves'] = int(value)
        self._updater.request_update()

    def _on_body_pose_scatter_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's pose-scatter map."""
        body = self._body(idx)
        if body is None:
            return
        if checked:
            w = self._body_tab_widget(idx)
            body['pose_scatter'] = {
                'sigma_deg': float(w.pose_scatter_sigma_spin.value()) if w is not None else 1.0  # type: ignore[attr-defined]
            }
        else:
            body.pop('pose_scatter', None)
        self._updater.request_update()

    def _on_body_pose_scatter_value(self, idx: int, value: float) -> None:
        """Update the pose-scatter sigma when the map is present."""
        body = self._body(idx)
        if body is None:
            return
        scatter = body.get('pose_scatter')
        if isinstance(scatter, dict):
            scatter['sigma_deg'] = float(value)
            self._updater.request_update()

    def _sync_body_mesh_enabled(self, w: QWidget, is_mesh: bool) -> None:
        """Enable the mesh-extras group only for a polyhedral-mesh body."""
        group = getattr(w, 'mesh_extras_group', None)
        if group is not None:
            group.setEnabled(is_mesh)
