"""Body atmosphere group: the haze column plus an optional detached shell.

Splits the per-body ``atmosphere`` block's controls out of the appearance
module (:mod:`spindoctor.cli.sim_editor.body_appearance`) so neither module
runs long, following the same sibling-mixin pattern the ring tab uses for its
advanced groups.  The group follows the editor-wide absent-key discipline: the
enable checkbox writes the ``atmosphere`` map only when active, the optional
detached-shell altitude key exists only while its own checkbox is on, and each
widget writes only its own key.

Every widget reference the round-trip and per-widget tests reach for is stored
on the tab widget ``w`` (``w.atmosphere_group``, ``w.atmosphere_shell_check``,
and so on), so a test can drive one control and assert the resulting
``sim_params`` edit.
"""

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QWidget

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.cli.sim_editor.widgets import kept_or_shown
from spindoctor.cli.sim_editor.widgets import make_dspin as _dspin
from spindoctor.sim.forward.atmosphere import DEFAULT_HAZE_ALBEDO


def _as_map(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, else an empty dict."""
    return value if isinstance(value, dict) else {}


class BodyAtmosphereMixin(SimEditorBase):
    """Builds and handles the per-body atmosphere (haze) group."""

    def _build_atmosphere_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the atmosphere group (haze column plus an optional shell)."""
        group = QGroupBox('Atmosphere (haze)')
        group.setCheckable(True)
        form = QFormLayout(group)
        atmosphere = _as_map(p.get('atmosphere'))
        scale_height = _dspin(
            minimum=0.1,
            maximum=200.0,
            decimals=2,
            step=0.5,
            value=float(atmosphere.get('scale_height_px', 8.0)),
            tooltip='Haze e-folding scale height in pixels.',
        )
        tau_ref = _dspin(
            minimum=0.01,
            maximum=50.0,
            decimals=3,
            step=0.1,
            value=float(atmosphere.get('tau_ref', 1.0)),
            tooltip='Tangent optical depth at the reference altitude.',
        )
        ref_altitude = _dspin(
            minimum=0.0,
            maximum=200.0,
            decimals=2,
            step=0.5,
            value=float(atmosphere.get('ref_altitude_px', 0.0)),
            tooltip='Altitude where the tangent optical depth equals tau_ref, in pixels.',
        )
        asymmetry = _dspin(
            minimum=-0.99,
            maximum=0.99,
            decimals=3,
            step=0.05,
            value=float(atmosphere.get('g', 0.0)),
            tooltip='Henyey-Greenstein asymmetry (positive is forward-scattering).',
        )
        stated_albedo = atmosphere.get('albedo')
        albedo = _dspin(
            minimum=0.0,
            maximum=1.0,
            decimals=3,
            step=0.05,
            value=DEFAULT_HAZE_ALBEDO if stated_albedo is None else float(stated_albedo),
            tooltip='Haze brightness scale (0 adds no light; the halo still dims what is behind).',
        )
        shell_check = QCheckBox('Detached shell')
        detached = _dspin(
            minimum=0.1,
            maximum=200.0,
            decimals=2,
            step=0.5,
            value=float(atmosphere.get('detached_px', 8.0)),
            tooltip='Altitude of a detached haze shell above the surface, in pixels.',
        )
        shell_check.setChecked('detached_px' in atmosphere)
        detached.setEnabled(shell_check.isChecked())
        form.addRow('Scale height (px):', scale_height)
        form.addRow('Tau ref:', tau_ref)
        form.addRow('Ref altitude (px):', ref_altitude)
        form.addRow('Asymmetry g:', asymmetry)
        form.addRow('Albedo:', albedo)
        form.addRow(shell_check)
        form.addRow('Detached (px):', detached)
        group.setChecked('atmosphere' in p)
        group.toggled.connect(lambda on, i=idx: self._on_body_atmosphere_toggled(i, on))
        scale_height.valueChanged.connect(
            lambda v, i=idx: self._on_body_atmosphere_value(i, 'scale_height_px', v)
        )
        tau_ref.valueChanged.connect(
            lambda v, i=idx: self._on_body_atmosphere_value(i, 'tau_ref', v)
        )
        ref_altitude.valueChanged.connect(
            lambda v, i=idx: self._on_body_atmosphere_value(i, 'ref_altitude_px', v)
        )
        asymmetry.valueChanged.connect(lambda v, i=idx: self._on_body_atmosphere_value(i, 'g', v))
        albedo.valueChanged.connect(lambda v, i=idx: self._on_body_atmosphere_value(i, 'albedo', v))
        shell_check.toggled.connect(lambda on, i=idx: self._on_body_atmosphere_shell_toggled(i, on))
        detached.valueChanged.connect(
            lambda v, i=idx: self._on_body_atmosphere_value(i, 'detached_px', v)
        )
        w.atmosphere_group = group  # type: ignore[attr-defined]
        w.atmosphere_scale_height_spin = scale_height  # type: ignore[attr-defined]
        w.atmosphere_tau_ref_spin = tau_ref  # type: ignore[attr-defined]
        w.atmosphere_ref_altitude_spin = ref_altitude  # type: ignore[attr-defined]
        w.atmosphere_g_spin = asymmetry  # type: ignore[attr-defined]
        w.atmosphere_albedo_spin = albedo  # type: ignore[attr-defined]
        w.atmosphere_shell_check = shell_check  # type: ignore[attr-defined]
        w.atmosphere_detached_spin = detached  # type: ignore[attr-defined]
        return group

    def _on_body_atmosphere_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the body's atmosphere map.

        Removing the map keeps it on the tab.  Re-enabling the group restores its
        keys without spins, such as the haze structure, and the exact value of
        every key whose spin still shows it, so turning the group off and on
        leaves a loaded scene's values unrounded.  A key whose spin was edited in
        between takes the spin's value.
        """
        body = self._body(idx)
        if body is None:
            return
        w = self._body_tab_widget(idx)
        if not checked:
            removed = body.pop('atmosphere', None)
            if w is not None and isinstance(removed, dict):
                w.atmosphere_removed = removed  # type: ignore[attr-defined]
            self._updater.request_update()
            return
        if w is not None:
            spins = {
                'scale_height_px': w.atmosphere_scale_height_spin,  # type: ignore[attr-defined]
                'tau_ref': w.atmosphere_tau_ref_spin,  # type: ignore[attr-defined]
                'ref_altitude_px': w.atmosphere_ref_altitude_spin,  # type: ignore[attr-defined]
                'g': w.atmosphere_g_spin,  # type: ignore[attr-defined]
                'albedo': w.atmosphere_albedo_spin,  # type: ignore[attr-defined]
            }
            if w.atmosphere_shell_check.isChecked():  # type: ignore[attr-defined]
                spins['detached_px'] = w.atmosphere_detached_spin  # type: ignore[attr-defined]
            # The tab carries a removed map only after the group was turned off once.
            removed = getattr(w, 'atmosphere_removed', None) or {}
            restored: dict[str, Any] = {
                key: value
                for key, value in removed.items()
                if key not in spins and key != 'detached_px'
            }
            body['atmosphere'] = restored | kept_or_shown(spins, removed)
        else:
            body['atmosphere'] = {
                'scale_height_px': 8.0,
                'tau_ref': 1.0,
                'ref_altitude_px': 0.0,
                'g': 0.0,
                'albedo': DEFAULT_HAZE_ALBEDO,
            }
        self._updater.request_update()

    def _on_body_atmosphere_value(self, idx: int, key: str, value: float) -> None:
        """Update one atmosphere component when its key is present in the map."""
        body = self._body(idx)
        if body is None:
            return
        atmosphere = body.get('atmosphere')
        if not isinstance(atmosphere, dict):
            return
        # A detached-shell edit writes only while the shell is enabled, so the
        # spin cannot resurrect the optional key when the shell is off.
        if key == 'detached_px' and 'detached_px' not in atmosphere:
            return
        atmosphere[key] = float(value)
        self._updater.request_update()

    def _on_body_atmosphere_shell_toggled(self, idx: int, checked: bool) -> None:
        """Insert or remove the optional detached-shell altitude key."""
        w = self._body_tab_widget(idx)
        if w is not None:
            w.atmosphere_detached_spin.setEnabled(checked)  # type: ignore[attr-defined]
        body = self._body(idx)
        if body is None:
            return
        atmosphere = body.get('atmosphere')
        if not isinstance(atmosphere, dict):
            return
        if checked:
            atmosphere['detached_px'] = (
                float(w.atmosphere_detached_spin.value()) if w is not None else 8.0  # type: ignore[attr-defined]
            )
        else:
            atmosphere.pop('detached_px', None)
        self._updater.request_update()
