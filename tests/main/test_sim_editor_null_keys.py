"""Null scene keys in the ``sd_create_simulated_image`` editor.

The scene validator accepts null for every optional key and reads it as absent, and
so does the renderer.  These tests load scenes whose optional keys are null and check
that every editor control shows what it shows for the same scene without those keys,
and that a body without a name or center shows the renderer's name and position.

Qt runs headless (offscreen platform).
"""

import copy
import os
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QLineEdit,
        QMessageBox,
        QSlider,
        QSpinBox,
        QWidget,
    )
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(
        f'PyQt6 QApplication init failed: {exc}',
        allow_module_level=True,
    )

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.sim.scene import validate_sim_params

# Every key a scene body may carry, each null.  The tests below give a body's
# sub-blocks mappings whose own keys are null.
_NULL_BODY: dict[str, Any] = dict.fromkeys(
    (
        'albedo_texture',
        'anti_aliasing',
        'atmosphere',
        'axis1',
        'axis2',
        'axis3',
        'center_u',
        'center_v',
        'crater_fill',
        'crater_max_radius',
        'crater_min_radius',
        'crater_power_law_exponent',
        'crater_relief_scale',
        'disc_texture',
        'illumination_angle',
        'km_per_pixel',
        'limb_relief_corr_deg',
        'limb_relief_rms',
        'mesh_detail_octaves',
        'mesh_lumpiness',
        'mesh_n_lat',
        'mesh_n_lon',
        'mesh_seed',
        'minnaert_k',
        'name',
        'nav_override',
        'opposition_surge',
        'phase_angle',
        'photometric_law',
        'pose_euler_deg',
        'pose_scatter',
        'range_km',
        'rotation_tilt',
        'rotation_z',
        'seed',
        'shading',
        'shape_model',
        'transits',
    )
)

_NULL_SPOT: dict[str, Any] = dict.fromkeys(('lat_deg', 'lon_deg', 'radius_deg', 'albedo_factor'))

_NULL_BODIES_SCENE: dict[str, Any] = {
    'instrument': 'coiss_nac',
    'size_v': 300,
    'size_u': 400,
    'random_seed': 42,
    'bodies': [
        {
            **_NULL_BODY,
            'axis1': 40.0,
            'axis2': 30.0,
            'opposition_surge': dict.fromkeys(('amplitude', 'width_deg')),
            'albedo_texture': {'rms': None, 'corr_px': None, 'spots': [dict(_NULL_SPOT)]},
            'disc_texture': {
                **dict.fromkeys(('band_amplitude', 'band_wavenumber', 'band_phase_deg')),
                'storms': [dict(_NULL_SPOT)],
            },
            'transits': [
                {
                    'moon': dict.fromkeys(('dv_px', 'du_px', 'radius_px', 'albedo_factor')),
                    'shadow': dict.fromkeys(('dv_px', 'du_px', 'radius_px', 'darkness')),
                }
            ],
            'atmosphere': {
                'scale_height_px': 8.0,
                'tau_ref': 1.0,
                **dict.fromkeys(('ref_altitude_px', 'g', 'detached_px', 'albedo')),
            },
            'nav_override': dict.fromkeys(('shape_model', 'mesh_lumpiness', 'pose_euler_deg')),
        },
        {
            **_NULL_BODY,
            'center_v': 40.0,
            'center_u': 40.0,
            'axis1': 40.0,
            'axis2': 30.0,
            'shape_model': 'polyhedral_mesh',
            'pose_scatter': {'sigma_deg': None},
        },
    ],
}
"""Two bodies, an ellipsoid and a mesh, with every optional body key null.

Each body states its axes so that it draws, and the mesh body states a center away
from the frame center: two bodies that overlap need a range to order them."""

# The spacecraft-ephemeris block needs every body to state a range, so the frame
# blocks sit in a scene of their own.
_NULL_FRAME_SCENE: dict[str, Any] = {
    'instrument': 'coiss_nac',
    'size_v': 300,
    'size_u': 400,
    'random_seed': 42,
    'closest_planet': None,
    'midtime_utc': None,
    'fit_camera_rotation': None,
    'oversample': None,
    'bodies': None,
    'stars': None,
    'noise': dict.fromkeys(
        (
            'poisson',
            'read_noise_dn',
            'cosmic_ray_events_per_pixel_per_sec',
            'missing_data_rate',
            'bias_dn',
            'bloom_length',
            'signal_full_scale_frac',
        )
    ),
    'optics': {
        'psf': {'sigma_v': 1.0, **dict.fromkeys(('sigma_u', 'w', 'r0', 'n'))},
        'smear': [dict.fromkeys(('dv_px', 'du_px'))],
        'distortion': dict.fromkeys(('k1', 'k2', 'center_v', 'center_u', 'nonradial_rms_px')),
        'ghosts': [dict.fromkeys(('dv_px', 'du_px', 'amplitude', 'defocus_sigma'))],
        'stray_light': dict.fromkeys(
            ('amplitude', 'direction_deg', 'model', 'center_v', 'center_u')
        ),
    },
    'spk_error': dict.fromkeys(('dv_px', 'du_px', 'reference_range_km')),
    'detector': dict.fromkeys(
        ('gain_state', 'binning', 'detector_model', 'exposure_ref_sec', 'quantization')
    ),
    'artifacts': dict.fromkeys(('instrument_defaults', 'adversarial')),
    'sky_counts': dict.fromkeys(('a', 'b', 'density_factor', 'diffuse_e_per_px')),
}
"""The global, optics, stray-light, artifacts, and sky-count blocks, every key null."""


@pytest.fixture
def qapp(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    """The shared headless Qt application, with an error dialog raising instead."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


def _without_nulls(value: Any) -> Any:
    """Return ``value`` with every null-valued mapping key left out, at every depth.

    Parameters:
        value: A scene, or any value inside one.

    Returns:
        A copy without the null keys.
    """
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _control_state(widget: QWidget) -> Any:
    """Return what one control shows, or None for a widget that is not a control.

    Parameters:
        widget: Any widget inside a tab.

    Returns:
        The control's displayed value, with its type name so two controls never
        compare equal across kinds.
    """
    if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
        return (type(widget).__name__, widget.value())
    if isinstance(widget, QSlider):
        return ('QSlider', widget.value())
    if isinstance(widget, QCheckBox):
        return ('QCheckBox', widget.isChecked())
    if isinstance(widget, QGroupBox) and widget.isCheckable():
        return ('QGroupBox', widget.title(), widget.isChecked())
    if isinstance(widget, QComboBox):
        return ('QComboBox', widget.currentText())
    if isinstance(widget, QLineEdit) and not isinstance(widget.parent(), QAbstractSpinBox):
        return ('QLineEdit', widget.text())
    return None


def _shown(scene: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """Load a scene into a fresh editor and return every tab's controls, in order.

    Parameters:
        scene: The scene to load.

    Returns:
        One entry per tab: its title and the state of each control in it.
    """
    model: Any = CreateSimulatedImageModel()
    model._apply_params_dict(copy.deepcopy(scene))
    tabs: list[tuple[str, list[Any]]] = []
    for index in range(model._tabs.count()):
        tab: Any = model._tabs.widget(index)
        states = [_control_state(child) for child in tab.findChildren(QWidget)]
        tabs.append((model._tabs.tabText(index), [s for s in states if s is not None]))
    return tabs


@pytest.mark.parametrize('scene', [_NULL_BODIES_SCENE, _NULL_FRAME_SCENE], ids=['bodies', 'frame'])
def test_null_keys_show_what_absent_keys_show(qapp: QApplication, scene: dict[str, Any]) -> None:
    """Each control shows the same value for a null key as for the key left out."""
    validate_sim_params(copy.deepcopy(scene), source='null scene')
    assert _shown(scene) == _shown(_without_nulls(scene))


def test_body_without_name_or_center_shows_the_renderers(qapp: QApplication) -> None:
    """A body without a name or center shows the renderer's name and the frame center."""
    model: Any = CreateSimulatedImageModel()
    model._apply_params_dict(
        {
            'instrument': 'coiss_nac',
            'size_v': 300,
            'size_u': 400,
            'bodies': [{'axis1': 40.0, 'axis2': 30.0}],
        }
    )
    tab: Any = model._tabs.widget(model._find_tab_by_properties('body', 0))
    shown = (tab.name_edit.text(), tab.center_v_spin.value(), tab.center_u_spin.value())
    assert shown == ('SIM-BODY-1', 150.0, 200.0)


def test_spk_edit_keeps_an_absent_reference_range_absent(qapp: QApplication) -> None:
    """An spk_error edit leaves out a reference range the block left out, as 0 shows it."""
    model: Any = CreateSimulatedImageModel()
    model._apply_params_dict(
        {'instrument': 'coiss_nac', 'size_v': 300, 'size_u': 400, 'spk_error': {'dv_px': 1.0}}
    )
    model._spk_du_spin.setValue(2.0)
    assert model.sim_params['spk_error'] == {'dv_px': 1.0, 'du_px': 2.0}
