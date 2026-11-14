"""Null scene keys in the ``sd_create_simulated_image`` editor.

The scene validator and the renderer both accept null for any optional key and read it
as absent.  These tests load scenes whose optional keys are null.  They check that each
editor control shows the same value as when those keys are left out.  Further tests
check that a body whose name and center are unset shows the renderer's name and the
frame center, that unset derived values follow edits, and that loading a scene leaves
it unchanged.

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
from spindoctor.sim.forward.body_spec import body_spec_from_params
from spindoctor.sim.forward.star import SKY_COUNTS_DEFAULTS
from spindoctor.sim.scene import validate_sim_params

# Every key a scene body may carry, each set to null.  The scenes below replace some
# of these with sub-blocks whose own keys are null.
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
"""Two bodies, an ellipsoid and a mesh, whose optional body keys are all null.

Each body sets its axes so that it is drawn.  The mesh body sets a center away from
the frame center, because two overlapping bodies need ranges to order them."""

# The spacecraft-ephemeris block needs a range on every body, so the frame-wide
# blocks get a scene of their own.
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
"""The top-level keys and the frame-wide blocks, with every key set to null."""


@pytest.fixture
def qapp(monkeypatch: pytest.MonkeyPatch) -> QApplication:
    """Return the shared headless Qt application, with error dialogs patched to raise."""

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
        The control's type name and displayed value.  The type name keeps two
        different kinds of control from comparing equal.
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
    """An unnamed, uncentered body shows the renderer's name and the frame center."""
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
    """An edit to another spin keeps an unset reference range unset."""
    model: Any = CreateSimulatedImageModel()
    model._apply_params_dict(
        {'instrument': 'coiss_nac', 'size_v': 300, 'size_u': 400, 'spk_error': {'dv_px': 1.0}}
    )
    model._spk_du_spin.setValue(2.0)
    assert model.sim_params['spk_error'] == {'dv_px': 1.0, 'du_px': 2.0}


def _loaded(scene: dict[str, Any]) -> Any:
    """Return a fresh editor with ``scene`` loaded into it."""
    model: Any = CreateSimulatedImageModel()
    model._apply_params_dict(copy.deepcopy(scene))
    return model


def _body_tab(model: Any, idx: int) -> Any:
    """Return the built tab widget of the body at ``idx``."""
    return model._tabs.widget(model._find_tab_by_properties('body', idx))


_FRAME_300_BY_400: dict[str, Any] = {'instrument': 'coiss_nac', 'size_v': 300, 'size_u': 400}


def test_unset_keys_show_the_renderers_defaults(qapp: QApplication) -> None:
    """An unset key's control shows the value the renderer draws with."""
    body = {'axis1': 40.0, 'axis2': 30.0, 'opposition_surge': {}}
    spec = body_spec_from_params(body, index=0)
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [body], 'sky_counts': {}, 'spk_error': {}})
    tab = _body_tab(model, 0)
    shown = (
        tab.anti_aliasing_spin.value(),
        tab.axis3_spin.value(),
        tab.surge_amplitude_spin.value(),
        model._sky_density_spin.value(),
        model._spk_range_spin.value(),
    )
    assert shown == (
        spec.anti_aliasing,
        spec.axis3,
        spec.surge_amplitude,
        SKY_COUNTS_DEFAULTS['density_factor'],
        0.0,
    )


def test_loading_a_scene_leaves_it_unchanged(qapp: QApplication) -> None:
    """Loading a scene and touching nothing leaves the editor's scene equal to it."""
    frame_blocks = {
        key: value
        for key, value in _NULL_FRAME_SCENE.items()
        if key not in ('closest_planet', 'bodies', 'stars', 'spk_error')
    }
    scene = {
        **frame_blocks,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'offset_rotation_deg': 0.0,
        'exposure_sec': 1.0,
        'time': 0.0,
        'ring_epoch': 0.0,
        'closest_planet': 'SATURN',
        'bodies': _NULL_BODIES_SCENE['bodies'],
        'stars': [],
    }
    validate_sim_params(copy.deepcopy(scene), source='round-trip scene')
    assert _loaded(scene).sim_params == scene


def test_an_axis_edit_moves_an_unset_axis3(qapp: QApplication) -> None:
    """Editing axis1 moves what an unset axis3 shows, without writing axis3."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [{'axis1': 40.0, 'axis2': 30.0}]})
    tab = _body_tab(model, 0)
    tab.axis1_spin.setValue(20.0)
    shown = (tab.axis3_spin.value(), 'axis3' in model.sim_params['bodies'][0])
    assert shown == (20.0, False)


def test_a_frame_size_change_moves_an_unset_center(qapp: QApplication) -> None:
    """Changing the frame height moves what an unset center shows, without writing it."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [{'axis1': 40.0, 'axis2': 30.0}]})
    model._size_v_spin.setValue(600)
    shown = (_body_tab(model, 0).center_v_spin.value(), 'center_v' in model.sim_params['bodies'][0])
    assert shown == (300.0, False)


def test_spk_edit_keeps_a_reference_range_finer_than_the_spin_shows(
    qapp: QApplication,
) -> None:
    """An edit to another spin keeps a reference range that its own spin rounds to 0."""
    model = _loaded({**_FRAME_300_BY_400, 'spk_error': {'dv_px': 1.0, 'reference_range_km': 0.004}})
    model._spk_du_spin.setValue(2.0)
    assert model.sim_params['spk_error']['reference_range_km'] == 0.004


def test_an_empty_ring_modulation_starts_turned_off(qapp: QApplication) -> None:
    """An empty azimuthal modulation block starts turned off, as the renderer skips it."""
    model: Any = CreateSimulatedImageModel()
    model._add_ring_tab()
    model.sim_params['ring_system']['azimuthal'] = {'modulation': {}}
    model._rebuild_dynamic_tabs()
    tab = model._tabs.widget(model._find_tab_by_properties('ring', 0))
    assert tab.azimuthal_modulation_group.isChecked() is False
