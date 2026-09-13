"""Star tab builder and change handlers.

Builds the per-star editing tab (position, magnitude, spectral class, PSF sigma,
smear vector, catalog name, and the V / U PSF-window sizes) and owns the
handlers that write star fields back into the data model.  It also carries the
truth-side information-asymmetry controls a scene plants on a star: the
``navigable`` flag, the ``catalog_error_v`` / ``catalog_error_u`` position
displacement, the unresolved-binary ``companion``, and the variable-star
``delta_mag``.  These render on the image side but are hidden from the
navigator, so the star technique must recover the pointing despite them.

Two absent-key conventions coexist on this tab.  Most controls follow the
editor-wide discipline: an enabled control writes its key, an unchecked control
leaves the key absent, and each widget writes only its own key.  The
``navigable`` checkbox is the deliberate exception: an absent ``navigable`` key
means navigable (true), so the box is checked by default and writes
``navigable: false`` (a present key) only when unchecked, matching the
validator's semantics.

The position spin boxes write the scene's ``v`` / ``u`` unchanged, so they
carry the scene's own convention: every position in a scene is a pixel corner,
and the centre of pixel ``N`` is ``N + 0.5``.  The row labels name that
convention rather than leaving the reader to infer it from a rendered star.

The PSF-window size is stored as a two-element list, never a tuple: the safe
YAML dumper is not guaranteed to accept tuples, and a tuple never compares
equal to the list the value reloads as, so a tuple would break a GUI-authored
psf_size round trip against its reloaded form.
"""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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

# Tooltip for the position spin boxes, which write the scene value directly.
_POSITION_TOOLTIP = (
    'Star {axis} position as a pixel corner: integer N is the boundary '
    'between pixel N-1 and pixel N, so the centre of pixel N is N + 0.5.'
)


class StarTabMixin(SimEditorBase):
    """Builds and handles the per-star editing tab."""

    def _build_star_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the star at ``idx``."""
        p = self.sim_params['stars'][idx]
        w = QWidget()
        w.setProperty('kind', 'star')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_star_name(i, t))
        fl.addRow('Name:', name_edit)

        v_spin = QDoubleSpinBox()
        v_spin.setRange(-10000.0, 20000.0)
        v_spin.setDecimals(1)
        v_spin.setValue(p.get('v', 0.0))
        v_spin.setToolTip(_POSITION_TOOLTIP.format(axis='V'))
        v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'v', v))
        fl.addRow('V (pixel corner):', v_spin)
        u_spin = QDoubleSpinBox()
        u_spin.setRange(-10000.0, 20000.0)
        u_spin.setDecimals(1)
        u_spin.setValue(p.get('u', 0.0))
        u_spin.setToolTip(_POSITION_TOOLTIP.format(axis='U'))
        u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'u', v))
        fl.addRow('U (pixel corner):', u_spin)
        # Keep references so drag updates can sync the UI
        w.v_spin = v_spin  # type: ignore[attr-defined]
        w.u_spin = u_spin  # type: ignore[attr-defined]

        vmag = QDoubleSpinBox()
        vmag.setRange(-10.0, 30.0)
        vmag.setDecimals(2)
        vmag.setValue(p.get('vmag', 3.0))
        vmag.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'vmag', v))
        fl.addRow('Magnitude (V):', vmag)
        sclass = QLineEdit(p.get('spectral_class', 'G2'))
        sclass.textChanged.connect(lambda t, i=idx: self._on_star_field(i, 'spectral_class', t))
        fl.addRow('Spectral class:', sclass)
        psf = QDoubleSpinBox()
        psf.setRange(0.1, 20.0)
        psf.setDecimals(2)
        psf.setValue(p.get('psf_sigma', 1.0))
        psf.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'psf_sigma', v))
        fl.addRow('PSF sigma:', psf)

        move_v_spin = QDoubleSpinBox()
        move_v_spin.setRange(-200.0, 200.0)
        move_v_spin.setDecimals(2)
        move_v_spin.setValue(float(p.get('move_v', 0.0)))
        move_v_spin.setToolTip('Star smear vector V (px) during the exposure.')
        move_v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'move_v', v))
        fl.addRow('Smear V (px):', move_v_spin)
        move_u_spin = QDoubleSpinBox()
        move_u_spin.setRange(-200.0, 200.0)
        move_u_spin.setDecimals(2)
        move_u_spin.setValue(float(p.get('move_u', 0.0)))
        move_u_spin.setToolTip('Star smear vector U (px) during the exposure.')
        move_u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'move_u', v))
        fl.addRow('Smear U (px):', move_u_spin)
        catalog_edit = QLineEdit(str(p.get('catalog_name', 'SIM')))
        catalog_edit.setToolTip('Source-catalog label carried on the star.')
        catalog_edit.textChanged.connect(lambda t, i=idx: self._on_star_field(i, 'catalog_name', t))
        fl.addRow('Catalog name:', catalog_edit)

        # Navigable flag (idealized): absent means navigable, so the box is
        # checked by default and writes ``navigable: false`` (a present key)
        # only when unchecked.  A non-navigable star renders but is dropped from
        # the navigator's filtered view entirely.
        navigable_check = QCheckBox('Navigable (unchecked plants a non-navigable star)')
        navigable_check.setChecked(bool(p.get('navigable', True)))
        navigable_check.setToolTip(
            'Checked leaves the key absent (navigable); unchecked writes '
            'navigable: false, dropping the star from the navigator view.'
        )
        navigable_check.clicked.connect(lambda checked, i=idx: self._on_star_navigable(i, checked))
        fl.addRow(navigable_check)
        w.navigable_check = navigable_check  # type: ignore[attr-defined]

        # Catalog error (truth): a fixed per-star position displacement (px) of
        # the RENDERED star off the catalog position the navigator predicts
        # from.  One enable checkbox gates both the v and u keys.
        has_catalog_error = p.get('catalog_error_v') is not None or (
            p.get('catalog_error_u') is not None
        )
        catalog_error_check = QCheckBox('Catalog error (v, u px)')
        catalog_error_check.setChecked(has_catalog_error)
        catalog_error_check.setToolTip(
            'Displace the rendered star off its catalog position; unchecked '
            'leaves both catalog_error keys absent.'
        )
        catalog_error_v_spin = QDoubleSpinBox()
        catalog_error_v_spin.setRange(-200.0, 200.0)
        catalog_error_v_spin.setDecimals(3)
        catalog_error_v_spin.setValue(float(p.get('catalog_error_v', 0.0)))
        catalog_error_v_spin.setEnabled(has_catalog_error)
        catalog_error_u_spin = QDoubleSpinBox()
        catalog_error_u_spin.setRange(-200.0, 200.0)
        catalog_error_u_spin.setDecimals(3)
        catalog_error_u_spin.setValue(float(p.get('catalog_error_u', 0.0)))
        catalog_error_u_spin.setEnabled(has_catalog_error)
        catalog_error_row = QHBoxLayout()
        catalog_error_row.setSpacing(4)
        catalog_error_row.setContentsMargins(0, 0, 0, 0)
        catalog_error_row.addWidget(catalog_error_v_spin)
        catalog_error_row.addWidget(catalog_error_u_spin)
        catalog_error_holder = QWidget()
        catalog_error_holder.setLayout(catalog_error_row)
        catalog_error_check.clicked.connect(
            lambda checked, i=idx: self._on_star_catalog_error_enabled(i, checked)
        )
        catalog_error_v_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_catalog_error_value(i, 'catalog_error_v', v)
        )
        catalog_error_u_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_catalog_error_value(i, 'catalog_error_u', v)
        )
        fl.addRow(catalog_error_check, catalog_error_holder)
        w.catalog_error_check = catalog_error_check  # type: ignore[attr-defined]
        w.catalog_error_v_spin = catalog_error_v_spin  # type: ignore[attr-defined]
        w.catalog_error_u_spin = catalog_error_u_spin  # type: ignore[attr-defined]

        # Variable star (truth): render at a brightness delta off the catalog
        # vmag (positive is fainter); the catalog vmag stays what the navigator
        # sees.
        has_delta_mag = p.get('delta_mag') is not None
        delta_mag_check = QCheckBox('Variable (delta mag)')
        delta_mag_check.setChecked(has_delta_mag)
        delta_mag_check.setToolTip(
            'Render the star off its catalog vmag; unchecked leaves delta_mag absent.'
        )
        delta_mag_spin = QDoubleSpinBox()
        delta_mag_spin.setRange(-10.0, 10.0)
        delta_mag_spin.setDecimals(2)
        delta_mag_spin.setValue(float(p.get('delta_mag', 0.0)))
        delta_mag_spin.setEnabled(has_delta_mag)
        delta_mag_check.clicked.connect(
            lambda checked, i=idx: self._on_star_delta_mag_enabled(i, checked)
        )
        delta_mag_spin.valueChanged.connect(lambda v, i=idx: self._on_star_delta_mag_value(i, v))
        fl.addRow(delta_mag_check, delta_mag_spin)
        w.delta_mag_check = delta_mag_check  # type: ignore[attr-defined]
        w.delta_mag_spin = delta_mag_spin  # type: ignore[attr-defined]

        # Companion (truth): an unresolved binary whose blended photocenter sits
        # off the catalog position -- a physical catalog error.  The checkable
        # group inserts or removes the whole companion map; each spin then
        # writes only its own sub-key.
        companion = p.get('companion') if isinstance(p.get('companion'), dict) else {}
        companion_group = QGroupBox('Companion (unresolved binary)')
        companion_group.setCheckable(True)
        companion_group.setChecked(bool(p.get('companion')))
        companion_layout = QFormLayout(companion_group)
        companion_sep_spin = QDoubleSpinBox()
        companion_sep_spin.setRange(0.0, 200.0)
        companion_sep_spin.setDecimals(3)
        companion_sep_spin.setValue(float(companion.get('sep_px', 2.0)))
        companion_sep_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_companion_value(i, 'sep_px', v)
        )
        companion_layout.addRow('Separation (px):', companion_sep_spin)
        companion_dmag_spin = QDoubleSpinBox()
        companion_dmag_spin.setRange(-10.0, 10.0)
        companion_dmag_spin.setDecimals(2)
        companion_dmag_spin.setValue(float(companion.get('delta_mag', 1.0)))
        companion_dmag_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_companion_value(i, 'delta_mag', v)
        )
        companion_layout.addRow('Delta mag:', companion_dmag_spin)
        companion_angle_spin = QDoubleSpinBox()
        companion_angle_spin.setRange(-360.0, 360.0)
        companion_angle_spin.setDecimals(1)
        companion_angle_spin.setValue(float(companion.get('angle_deg', 0.0)))
        companion_angle_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_companion_value(i, 'angle_deg', v)
        )
        companion_layout.addRow('Angle (deg):', companion_angle_spin)
        companion_group.toggled.connect(
            lambda checked, i=idx: self._on_star_companion_toggled(i, checked)
        )
        main_layout.addWidget(companion_group)
        w.companion_group = companion_group  # type: ignore[attr-defined]
        w.companion_sep_spin = companion_sep_spin  # type: ignore[attr-defined]
        w.companion_dmag_spin = companion_dmag_spin  # type: ignore[attr-defined]
        w.companion_angle_spin = companion_angle_spin  # type: ignore[attr-defined]

        # PSF size V slider with min/max labels and spinbox
        # Map slider positions 0-11 to odd values 1, 3, 5, ..., 23
        psf_size_v_row = QHBoxLayout()
        psf_size_v_row.setSpacing(4)
        psf_size_v_row.setContentsMargins(0, 0, 0, 0)
        psf_size_v_min_label = QLabel('1')
        psf_size_v_min_label.setFixedWidth(35)
        psf_size_v_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_v_max_label = QLabel('23')
        psf_size_v_max_label.setFixedWidth(40)
        psf_size_v_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_v_slider = QSlider(Qt.Orientation.Horizontal)
        psf_size_v_slider.setRange(0, 11)  # 12 positions for odd values 1-23
        psf_size_v_default = p.get('psf_size', [11, 11])[0]
        psf_size_v_default = self._ensure_odd_psf_size(psf_size_v_default)
        # Convert odd value to slider position: (value - 1) // 2
        psf_size_v_slider.setValue((psf_size_v_default - 1) // 2)
        psf_size_v_slider.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_v_slider(i, v)
        )
        psf_size_v_spin = QSpinBox()
        psf_size_v_spin.setRange(1, 23)
        psf_size_v_spin.setSingleStep(2)  # Step by 2 to keep odd
        psf_size_v_spin.setValue(psf_size_v_default)
        psf_size_v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_psf_size_v_spin(i, v))
        psf_size_v_row.addWidget(psf_size_v_min_label)
        psf_size_v_row.addWidget(psf_size_v_slider, stretch=1)
        psf_size_v_row.addWidget(psf_size_v_max_label)
        psf_size_v_row.addWidget(psf_size_v_spin)
        psf_size_v_holder = QWidget()
        psf_size_v_holder.setLayout(psf_size_v_row)
        fl.addRow('PSF size V:', psf_size_v_holder)
        # Store references for sync
        w.psf_size_v_slider = psf_size_v_slider  # type: ignore[attr-defined]
        w.psf_size_v_spin = psf_size_v_spin  # type: ignore[attr-defined]

        # PSF size U slider with min/max labels and spinbox
        # Map slider positions 0-11 to odd values 1, 3, 5, ..., 23
        psf_size_u_row = QHBoxLayout()
        psf_size_u_row.setSpacing(4)
        psf_size_u_row.setContentsMargins(0, 0, 0, 0)
        psf_size_u_min_label = QLabel('1')
        psf_size_u_min_label.setFixedWidth(35)
        psf_size_u_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_u_max_label = QLabel('23')
        psf_size_u_max_label.setFixedWidth(40)
        psf_size_u_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_u_slider = QSlider(Qt.Orientation.Horizontal)
        psf_size_u_slider.setRange(0, 11)  # 12 positions for odd values 1-23
        psf_size_u_default = p.get('psf_size', [11, 11])[1]
        psf_size_u_default = self._ensure_odd_psf_size(psf_size_u_default)
        # Convert odd value to slider position: (value - 1) // 2
        psf_size_u_slider.setValue((psf_size_u_default - 1) // 2)
        psf_size_u_slider.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_u_slider(i, v)
        )
        psf_size_u_spin = QSpinBox()
        psf_size_u_spin.setRange(1, 23)
        psf_size_u_spin.setSingleStep(2)  # Step by 2 to keep odd
        psf_size_u_spin.setValue(psf_size_u_default)
        psf_size_u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_psf_size_u_spin(i, v))
        psf_size_u_row.addWidget(psf_size_u_min_label)
        psf_size_u_row.addWidget(psf_size_u_slider, stretch=1)
        psf_size_u_row.addWidget(psf_size_u_max_label)
        psf_size_u_row.addWidget(psf_size_u_spin)
        psf_size_u_holder = QWidget()
        psf_size_u_holder.setLayout(psf_size_u_row)
        fl.addRow('PSF size U:', psf_size_u_holder)
        # Store references for sync
        w.psf_size_u_slider = psf_size_u_slider  # type: ignore[attr-defined]
        w.psf_size_u_spin = psf_size_u_spin  # type: ignore[attr-defined]

        # Delete button at bottom
        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('star', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

        return w

    # ---- Field handlers ----
    def _on_star_field(self, idx: int, key: str, value: Any) -> None:
        """Write a scalar star field into the data model and re-render."""
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_star_name(self, idx: int, text: str) -> None:
        """Rename a star and refresh the tab titles."""
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _on_star_navigable(self, idx: int, checked: bool) -> None:
        """Set or clear the navigable flag (absent means navigable)."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        if checked:
            star.pop('navigable', None)
        else:
            star['navigable'] = False
        self._updater.request_update()

    def _on_star_catalog_error_enabled(self, idx: int, enabled: bool) -> None:
        """Insert or remove the star's catalog_error_v / catalog_error_u keys."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        tab_idx = self._find_tab_by_properties('star', idx)
        tab_w = self._tabs.widget(tab_idx) if tab_idx is not None else None
        v_spin = getattr(tab_w, 'catalog_error_v_spin', None)
        u_spin = getattr(tab_w, 'catalog_error_u_spin', None)
        if enabled:
            star['catalog_error_v'] = float(v_spin.value()) if v_spin is not None else 0.0
            star['catalog_error_u'] = float(u_spin.value()) if u_spin is not None else 0.0
        else:
            star.pop('catalog_error_v', None)
            star.pop('catalog_error_u', None)
        if v_spin is not None:
            v_spin.setEnabled(enabled)
        if u_spin is not None:
            u_spin.setEnabled(enabled)
        self._updater.request_update()

    def _on_star_catalog_error_value(self, idx: int, key: str, value: float) -> None:
        """Update one catalog-error component when the key is enabled."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        if key in star:
            star[key] = float(value)
            self._updater.request_update()

    def _on_star_delta_mag_enabled(self, idx: int, enabled: bool) -> None:
        """Insert or remove the star's variable-brightness delta_mag key."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        tab_idx = self._find_tab_by_properties('star', idx)
        tab_w = self._tabs.widget(tab_idx) if tab_idx is not None else None
        spin = getattr(tab_w, 'delta_mag_spin', None)
        if enabled:
            star['delta_mag'] = float(spin.value()) if spin is not None else 0.0
        else:
            star.pop('delta_mag', None)
        if spin is not None:
            spin.setEnabled(enabled)
        self._updater.request_update()

    def _on_star_delta_mag_value(self, idx: int, value: float) -> None:
        """Update the star's delta_mag when the key is enabled."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        if 'delta_mag' in star:
            star['delta_mag'] = float(value)
            self._updater.request_update()

    def _on_star_companion_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the star's companion (unresolved-binary) map."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        if checked:
            tab_idx = self._find_tab_by_properties('star', idx)
            tab_w = self._tabs.widget(tab_idx) if tab_idx is not None else None
            sep = getattr(tab_w, 'companion_sep_spin', None)
            dmag = getattr(tab_w, 'companion_dmag_spin', None)
            angle = getattr(tab_w, 'companion_angle_spin', None)
            star['companion'] = {
                'sep_px': float(sep.value()) if sep is not None else 2.0,
                'delta_mag': float(dmag.value()) if dmag is not None else 1.0,
                'angle_deg': float(angle.value()) if angle is not None else 0.0,
            }
        else:
            star.pop('companion', None)
        self._updater.request_update()

    def _on_star_companion_value(self, idx: int, key: str, value: float) -> None:
        """Update one companion sub-key when the companion map is present."""
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        star = self.sim_params['stars'][idx]
        companion = star.get('companion')
        if isinstance(companion, dict):
            companion[key] = float(value)
            self._updater.request_update()

    def _ensure_odd_psf_size(self, value: int) -> int:
        """Ensure PSF size is an odd integer in the range [1, 23].

        Parameters:
            value: The value to normalize.

        Returns:
            Clamped odd integer value in [1, 23].
        """
        value = int(value)
        value = max(1, min(23, value))
        if value % 2 == 0:
            value = max(1, value - 1)
        return value

    def _on_star_psf_size_slider(self, idx: int, dimension: int, value: int) -> None:
        """Handle PSF size slider change for a star.

        Parameters:
            idx: Star index.
            dimension: 0 for V, 1 for U.
            value: Slider value (0-11).
        """
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        # Convert slider position (0-11) to odd value (1, 3, 5, ..., 23)
        odd_value = value * 2 + 1
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin_attr = 'psf_size_v_spin' if dimension == 0 else 'psf_size_u_spin'
                spin = getattr(tab_w, spin_attr, None)
                if spin is not None:
                    spin.blockSignals(True)
                    spin.setValue(odd_value)
                    spin.blockSignals(False)
        current_psf_size = self.sim_params['stars'][idx].get('psf_size', [11, 11])
        if dimension == 0:
            self.sim_params['stars'][idx]['psf_size'] = [odd_value, current_psf_size[1]]
        else:
            self.sim_params['stars'][idx]['psf_size'] = [current_psf_size[0], odd_value]
        self._updater.request_update()

    def _on_star_psf_size_spin(self, idx: int, dimension: int, value: int) -> None:
        """Handle PSF size spinbox change for a star.

        Parameters:
            idx: Star index.
            dimension: 0 for V, 1 for U.
            value: Spinbox value.
        """
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        # Coerce to nearest odd in [1, 23] and clamp
        odd_value = self._ensure_odd_psf_size(value)
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                # Update spinbox if value was adjusted
                if odd_value != value:
                    spin_attr = 'psf_size_v_spin' if dimension == 0 else 'psf_size_u_spin'
                    spin = getattr(tab_w, spin_attr, None)
                    if spin is not None:
                        spin.blockSignals(True)
                        spin.setValue(odd_value)
                        spin.blockSignals(False)
                # Convert odd value to slider position: (value - 1) // 2
                slider_attr = 'psf_size_v_slider' if dimension == 0 else 'psf_size_u_slider'
                slider = getattr(tab_w, slider_attr, None)
                if slider is not None:
                    slider.blockSignals(True)
                    slider.setValue((odd_value - 1) // 2)
                    slider.blockSignals(False)
        current_psf_size = self.sim_params['stars'][idx].get('psf_size', [11, 11])
        if dimension == 0:
            self.sim_params['stars'][idx]['psf_size'] = [odd_value, current_psf_size[1]]
        else:
            self.sim_params['stars'][idx]['psf_size'] = [current_psf_size[0], odd_value]
        self._updater.request_update()

    def _on_star_psf_size_v_slider(self, idx: int, value: int) -> None:
        """Handle PSF size V slider change for a star."""
        self._on_star_psf_size_slider(idx, 0, value)

    def _on_star_psf_size_v_spin(self, idx: int, value: int) -> None:
        """Handle PSF size V spinbox change for a star."""
        self._on_star_psf_size_spin(idx, 0, value)

    def _on_star_psf_size_u_slider(self, idx: int, value: int) -> None:
        """Handle PSF size U slider change for a star."""
        self._on_star_psf_size_slider(idx, 1, value)

    def _on_star_psf_size_u_spin(self, idx: int, value: int) -> None:
        """Handle PSF size U spinbox change for a star."""
        self._on_star_psf_size_spin(idx, 1, value)
