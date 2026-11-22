"""Null scene keys in the ``sd_create_simulated_image`` editor.

The scene validator and the renderer both accept null for any optional key and read it
as absent.  These tests load scenes whose optional keys are null.  They check that each
editor control shows the same value as when those keys are left out.  Further tests
check that a body whose name and center are unset shows the renderer's name and the
frame center, that unset derived values follow edits, and that loading a scene leaves
it unchanged.  The remaining tests check that adding a ring to a null ring block or
feature list works, that a ring kind switch fills a null required shape key, that a
ring modulation can be turned on under a null azimuthal block, and that a body whose
range is unset counts at its default range when the editor picks a range for an added
body and when it warns about duplicate ranges.  The last tests check that the range box
writes the default range its spin shows, and that the navigation-override group writes
only the keys the author edits and shows the body's own values for the keys it leaves
unset.

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

from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.render.radiance.stars import SKY_COUNTS_DEFAULTS
from spindoctor.sim.scene.validate import validate_sim_params
from spindoctor.ui.sim_editor import CreateSimulatedImageModel

# Every key a scene body may carry, each set to null.  The scenes below replace some
# of these with sub-blocks whose own keys are null.
_NULL_BODY: dict[str, Any] = dict.fromkeys(
    (
        'albedo_texture',
        'anti_aliasing',
        'atmosphere',
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
        'semi_a_px',
        'semi_b_px',
        'semi_c_px',
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
            'semi_a_px': 20.0,
            'semi_b_px': 15.0,
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
            'semi_a_px': 20.0,
            'semi_b_px': 15.0,
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
"""The optional top-level keys and the frame-wide blocks, set to null."""


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
            'bodies': [{'semi_a_px': 20.0, 'semi_b_px': 15.0}],
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
    body = {'semi_a_px': 20.0, 'semi_b_px': 15.0, 'opposition_surge': {}}
    spec = body_spec_from_params(body, index=0)
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [body], 'sky_counts': {}, 'spk_error': {}})
    tab = _body_tab(model, 0)
    shown = (
        tab.anti_aliasing_spin.value(),
        tab.semi_c_spin.value(),
        tab.surge_amplitude_spin.value(),
        model._sky_density_spin.value(),
        model._spk_range_spin.value(),
    )
    assert shown == (
        spec.anti_aliasing,
        spec.semi_c_px,
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


def test_a_semi_axis_edit_moves_an_unset_semi_c(qapp: QApplication) -> None:
    """Editing semi_a_px moves what an unset semi_c_px shows, without writing it."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [{'semi_a_px': 20.0, 'semi_b_px': 15.0}]})
    tab = _body_tab(model, 0)
    tab.semi_a_spin.setValue(10.0)
    shown = (tab.semi_c_spin.value(), 'semi_c_px' in model.sim_params['bodies'][0])
    assert shown == (10.0, False)


def test_a_frame_size_change_moves_an_unset_center(qapp: QApplication) -> None:
    """Changing the frame height moves what an unset center shows, without writing it."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [{'semi_a_px': 20.0, 'semi_b_px': 15.0}]})
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


_RING_GEOMETRY: dict[str, Any] = {'opening_deg_obs': 90.0, 'opening_deg_sun': 90.0}


@pytest.mark.parametrize(
    'ring_system',
    [None, {'geometry': _RING_GEOMETRY, 'features': None}],
    ids=['null-block', 'null-features'],
)
def test_adding_a_ring_to_a_null_ring_system(
    qapp: QApplication, ring_system: dict[str, Any] | None
) -> None:
    """Adding a ring where the ring system or its feature list is null adds one feature."""
    scene = {**_FRAME_300_BY_400, 'random_seed': 42, 'ring_system': ring_system}
    validate_sim_params(copy.deepcopy(scene), source='null ring scene')
    model = _loaded(scene)
    model._add_ring_tab()
    assert len(model._ring_features()) == 1


def test_a_kind_switch_fills_a_null_required_shape_key(qapp: QApplication) -> None:
    """A switch to a kind that needs a null shape key writes the editor's starting width."""
    feature = {
        'name': 'Edge',
        'kind': 'edge',
        'tau': 1.0,
        'width_px': None,
        'orbit': {'a_px': 100.0},
    }
    scene = {
        **_FRAME_300_BY_400,
        'random_seed': 42,
        'ring_system': {'geometry': _RING_GEOMETRY, 'features': [feature]},
    }
    validate_sim_params(copy.deepcopy(scene), source='null width scene')
    model = _loaded(scene)
    model._on_ring_kind(0, 'ringlet')
    assert model.sim_params['ring_system']['features'][0]['width_px'] == 20.0


def test_turning_on_a_ring_modulation_under_a_null_azimuthal_block(
    qapp: QApplication,
) -> None:
    """Turning the modulation group on writes it when the azimuthal block is null."""
    feature = {'name': 'Ring', 'kind': 'ringlet', 'tau': 1.0, 'width_px': 20.0}
    scene = {
        **_FRAME_300_BY_400,
        'random_seed': 42,
        'ring_system': {
            'geometry': _RING_GEOMETRY,
            'features': [{**feature, 'orbit': {'a_px': 100.0}}],
            'azimuthal': None,
        },
    }
    validate_sim_params(copy.deepcopy(scene), source='null azimuthal scene')
    model = _loaded(scene)
    model._on_ring_modulation_toggled(True)
    assert 'modulation' in model.sim_params['ring_system']['azimuthal']


# The first body sits at the frame center with no range, so the renderer draws it at
# its default range of 1; the second sits apart from it.
_UNSET_RANGE_BODY: dict[str, Any] = {'semi_a_px': 10.0, 'semi_b_px': 10.0, 'range_km': None}
_APART_BODY: dict[str, Any] = {
    'semi_a_px': 10.0,
    'semi_b_px': 10.0,
    'center_v': 40.0,
    'center_u': 40.0,
}


def test_an_added_body_skips_an_unset_bodys_default_range(qapp: QApplication) -> None:
    """A body added beside one whose range is null takes a range the other doesn't use."""
    scene = {**_FRAME_300_BY_400, 'random_seed': 42, 'bodies': [_UNSET_RANGE_BODY]}
    validate_sim_params(copy.deepcopy(scene), source='unset range scene')
    model = _loaded(scene)
    model._add_body_tab()
    assert model.sim_params['bodies'][1]['range_km'] == 2.0


def test_an_unset_range_counts_toward_the_duplicate_warning(qapp: QApplication) -> None:
    """A body whose null range defaults to another body's explicit range raises the warning."""
    scene = {
        **_FRAME_300_BY_400,
        'random_seed': 42,
        'bodies': [_UNSET_RANGE_BODY, {**_APART_BODY, 'range_km': 1.0}],
    }
    validate_sim_params(copy.deepcopy(scene), source='duplicate range scene')
    assert _loaded(scene)._warning_label.text() == 'Warning: duplicate body ranges'


# The second body's range is unset, so its default is 2, which no other body's range
# or index default matches.
_SECOND_UNSET_SCENE: dict[str, Any] = {
    **_FRAME_300_BY_400,
    'random_seed': 42,
    'bodies': [_APART_BODY, _UNSET_RANGE_BODY],
}


@pytest.mark.parametrize(
    ('range_km', 'enabled'), [(None, False), (2.0, True)], ids=['unset', 'set']
)
def test_the_range_spin_is_enabled_only_for_a_set_range(
    qapp: QApplication, range_km: float | None, enabled: bool
) -> None:
    """The range spin starts disabled for an unset range and enabled for a set one."""
    scene = copy.deepcopy(_SECOND_UNSET_SCENE)
    scene['bodies'][1]['range_km'] = range_km
    assert _body_tab(_loaded(scene), 1).range_spin.isEnabled() is enabled


def test_setting_an_unset_range_writes_the_shown_default(qapp: QApplication) -> None:
    """Checking an unset body's range box writes the default its spin shows and enables it."""
    model = _loaded(_SECOND_UNSET_SCENE)
    tab = _body_tab(model, 1)
    tab.range_check.click()
    assert model.sim_params['bodies'][1]['range_km'] == pytest.approx(2.0)
    assert tab.range_spin.isEnabled()


def test_clearing_a_range_rechecks_the_duplicate_warning(qapp: QApplication) -> None:
    """Clearing a range that duplicates another body's puts the default back in its spin.

    The default ends the duplicate warning, and the spin is disabled.
    """
    scene = copy.deepcopy(_SECOND_UNSET_SCENE)
    scene['bodies'][0]['range_km'] = 1.0
    scene['bodies'][1]['range_km'] = 1.0
    model = _loaded(scene)
    tab = _body_tab(model, 1)
    assert model._warning_label.text() == 'Warning: duplicate body ranges'
    tab.range_check.click()
    assert 'range_km' not in model.sim_params['bodies'][1]
    assert model._warning_label.text() == ''
    assert tab.range_spin.value() == pytest.approx(2.0)
    assert not tab.range_spin.isEnabled()


# A mesh body whose override group is on and sets none of its keys.
_OVERRIDE_BODY: dict[str, Any] = {
    'semi_a_px': 20.0,
    'semi_b_px': 15.0,
    'shape_model': 'polyhedral_mesh',
    'nav_override': {},
}


def _set_control(control: Any, value: Any) -> None:
    """Set a combo box's text, a spin's value, or the first spin of a list of spins.

    Parameters:
        control: A combo box, a spin box, or a list of spin boxes.
        value: The text or value to set.
    """
    if isinstance(control, QComboBox):
        control.setCurrentText(value)
    elif isinstance(control, list):
        control[0].setValue(value)
    else:
        control.setValue(value)


def _control_value(control: Any) -> Any:
    """Return a combo box's text, a spin's value, or the first spin's value of a list.

    Parameters:
        control: A combo box, a spin box, or a list of spin boxes.

    Returns:
        The text or value the control shows.
    """
    if isinstance(control, QComboBox):
        return control.currentText()
    if isinstance(control, list):
        return control[0].value()
    return control.value()


@pytest.mark.parametrize(
    ('body_pose', 'override', 'attr', 'value', 'expected'),
    [
        (
            None,
            {'mesh_lumpiness': None},
            'nav_shape_combo',
            'ellipsoid',
            {'mesh_lumpiness': None, 'shape_model': 'ellipsoid'},
        ),
        (
            None,
            {'shape_model': None},
            'nav_lump_spin',
            0.5,
            {'shape_model': None, 'mesh_lumpiness': pytest.approx(0.5)},
        ),
        (
            [10.0, 20.0, 30.0],
            {'shape_model': None},
            'nav_pose_spins',
            40.0,
            {'shape_model': None, 'pose_euler_deg': pytest.approx([40.0, 20.0, 30.0])},
        ),
        (
            None,
            {'pose_euler_deg': [10.0, 20.0, 30.0]},
            'nav_pose_spins',
            40.0,
            {'pose_euler_deg': pytest.approx([40.0, 20.0, 30.0])},
        ),
    ],
    ids=['shape', 'lumpiness', 'pose', 'set-pose'],
)
def test_an_override_edit_writes_only_its_own_key(
    qapp: QApplication,
    body_pose: list[float] | None,
    override: dict[str, Any],
    attr: str,
    value: Any,
    expected: dict[str, Any],
) -> None:
    """An edit to one override control writes its key and leaves the others as stored.

    A pose edit keeps the other two axes of the pose the navigator would otherwise use:
    the override's own pose if it sets one, else the body's.
    """
    body = {**_OVERRIDE_BODY, 'nav_override': override}
    if body_pose is not None:
        body['pose_euler_deg'] = body_pose
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [body]})
    _set_control(getattr(_body_tab(model, 0), attr), value)
    assert model.sim_params['bodies'][0]['nav_override'] == expected


def test_turning_the_override_on_writes_an_empty_override(qapp: QApplication) -> None:
    """Turning the override group on writes an override that sets no key."""
    body = {key: value for key, value in _OVERRIDE_BODY.items() if key != 'nav_override'}
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [body]})
    _body_tab(model, 0).nav_override_group.setChecked(True)
    assert model.sim_params['bodies'][0]['nav_override'] == {}


def test_turning_the_override_back_on_shows_the_bodys_values(qapp: QApplication) -> None:
    """Turning the override off and on again shows the body's lumpiness, not the old one."""
    body = {**_OVERRIDE_BODY, 'mesh_lumpiness': 0.2, 'nav_override': {'mesh_lumpiness': 0.0}}
    tab = _body_tab(_loaded({**_FRAME_300_BY_400, 'bodies': [body]}), 0)
    tab.nav_override_group.setChecked(False)
    tab.nav_override_group.setChecked(True)
    assert tab.nav_lump_spin.value() == pytest.approx(0.2)


def test_a_body_shape_edit_moves_an_unset_predicted_shape(qapp: QApplication) -> None:
    """Switching the body's shape moves what an unset predicted shape shows."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [_OVERRIDE_BODY]})
    tab = _body_tab(model, 0)
    tab.shape_combo.setCurrentText('ellipsoid')
    assert tab.nav_shape_combo.currentText() == 'ellipsoid'
    assert model.sim_params['bodies'][0]['nav_override'] == {}


def test_a_body_lumpiness_edit_moves_an_unset_predicted_lumpiness(qapp: QApplication) -> None:
    """Editing the body's lumpiness moves what an unset predicted lumpiness shows."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [_OVERRIDE_BODY]})
    tab = _body_tab(model, 0)
    tab.mesh_lumpiness_spin.setValue(0.5)
    assert tab.nav_lump_spin.value() == pytest.approx(0.5)
    assert model.sim_params['bodies'][0]['nav_override'] == {}


def test_a_body_pose_edit_moves_an_unset_predicted_pose(qapp: QApplication) -> None:
    """Editing the body's pose moves what an unset predicted pose shows."""
    model = _loaded({**_FRAME_300_BY_400, 'bodies': [_OVERRIDE_BODY]})
    tab = _body_tab(model, 0)
    tab.pose_spins[0].setValue(30.0)
    assert tab.nav_pose_spins[0].value() == pytest.approx(30.0)
    assert model.sim_params['bodies'][0]['nav_override'] == {}


@pytest.mark.parametrize(
    ('override', 'attr', 'value', 'nav_attr', 'kept'),
    [
        (
            {'shape_model': 'polyhedral_mesh'},
            'shape_combo',
            'ellipsoid',
            'nav_shape_combo',
            'polyhedral_mesh',
        ),
        ({'mesh_lumpiness': 0.0}, 'mesh_lumpiness_spin', 0.5, 'nav_lump_spin', pytest.approx(0.0)),
        (
            {'pose_euler_deg': [0.0, 0.0, 0.0]},
            'pose_spins',
            30.0,
            'nav_pose_spins',
            pytest.approx(0.0),
        ),
    ],
    ids=['shape', 'lumpiness', 'pose'],
)
def test_a_body_edit_keeps_a_set_predicted_value(
    qapp: QApplication,
    override: dict[str, Any],
    attr: str,
    value: Any,
    nav_attr: str,
    kept: Any,
) -> None:
    """Editing a body value leaves the matching predicted value the override sets."""
    body = {**_OVERRIDE_BODY, 'nav_override': override}
    tab = _body_tab(_loaded({**_FRAME_300_BY_400, 'bodies': [body]}), 0)
    _set_control(getattr(tab, attr), value)
    assert _control_value(getattr(tab, nav_attr)) == kept
