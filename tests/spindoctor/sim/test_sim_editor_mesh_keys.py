"""The sim editor cannot author a mesh body its scene validator refuses.

The validator refuses a non-null value of every key in the schema's
``MESH_UNSUPPORTED_KEYS`` on a ``shape_model: polyhedral_mesh`` body.  The editor
reads the same table: switching a body to a mesh disables every control that
writes one of those keys, gives each the validator's own sentence as its
tooltip, and moves the keys the body carried out of the scene into the editor's
stash, so the mesh rule never refuses what the editor saves.  Switching the body
back to an ellipsoid restores the stashed keys and re-enables the controls.

The property itself is checked by operating every control a mesh body tab leaves
enabled and validating the scene that results.

Qt runs headless (offscreen platform).
"""

import copy
import os
from pathlib import Path
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import (
        QAbstractSlider,
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QWidget,
    )
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(f'PyQt6 QApplication init failed: {exc}', allow_module_level=True)

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.cli.sim_editor.body_mesh_keys import MESH_REFUSED_WIDGETS
from spindoctor.sim.scene import (
    SimSceneValidationError,
    load_sim_scene,
    save_sim_scene,
    validate_sim_params,
)
from spindoctor.sim.scene_schema import MESH_UNSUPPORTED_KEYS

# An ellipsoid body carrying every refused key with a non-default value, so a
# lost or defaulted key is visible after a round trip.
_FULL_ELLIPSOID: dict[str, Any] = {
    'name': 'FULL',
    'center_v': 256.0,
    'center_u': 256.0,
    'range_km': 1.0,
    'shape_model': 'ellipsoid',
    'axis1': 100.0,
    'axis2': 80.0,
    'axis3': 80.0,
    'atmosphere': {'scale_height_px': 4.0, 'tau_ref': 1.5, 'ref_altitude_px': 0.0, 'g': 0.2},
    'photometric_law': 'minnaert',
    'minnaert_k': 0.7,
    'opposition_surge': {'amplitude': 0.4, 'width_deg': 5.0},
    'albedo_texture': {
        'rms': 0.05,
        'corr_px': 12.0,
        'spots': [{'lat_deg': 10.0, 'lon_deg': 80.0, 'radius_deg': 8.0, 'albedo_factor': 0.6}],
    },
    'disc_texture': {
        'band_amplitude': 0.1,
        'band_wavenumber': 6.0,
        'band_phase_deg': 15.0,
        'storms': [],
    },
    'transits': [{'moon': {'dv_px': 3.0, 'du_px': -4.0, 'radius_px': 5.0, 'albedo_factor': 1.2}}],
    'crater_fill': 0.4,
    'crater_min_radius': 0.06,
    'crater_max_radius': 0.2,
    'crater_power_law_exponent': 2.5,
    'crater_relief_scale': 0.5,
}


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image editor model."""
    return CreateSimulatedImageModel()


def _tab(model: Any, idx: int = 0) -> QWidget:
    """The built tab widget of the body at ``idx``."""
    tab_idx = model._find_tab_by_properties('body', idx)
    assert tab_idx is not None
    return cast(QWidget, model._tabs.widget(tab_idx))


def _set_shape(model: Any, shape: str, idx: int = 0) -> None:
    """Pick ``shape`` in the body's shape-model combo, as an operator would."""
    _tab(model, idx).shape_combo.setCurrentText(shape)  # type: ignore[attr-defined]


def _full_ellipsoid() -> dict[str, Any]:
    """A deep copy of the ellipsoid body that carries every refused key."""
    return copy.deepcopy(_FULL_ELLIPSOID)


def _scene(body: dict[str, Any]) -> dict[str, Any]:
    """A minimal valid scene holding ``body`` as its only body."""
    return {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'bodies': [body],
    }


def _refusal(keys: list[str]) -> str:
    """The validator's sentence for a mesh body carrying ``keys``, without its prefix."""
    body = {key: _FULL_ELLIPSOID[key] for key in keys}
    body.update(name='MESH', axis1=30.0, shape_model='polyhedral_mesh')
    with pytest.raises(SimSceneValidationError) as excinfo:
        validate_sim_params(_scene(body), source='SRC')
    return str(excinfo.value).removeprefix('SRC: bodies[0]: ')


_MISSING = object()


def _perturb(widget: QWidget) -> None:
    """Change a control the way an operator would: toggle it or move it to an end stop."""
    if isinstance(widget, QGroupBox):
        widget.setChecked(not widget.isChecked())
    elif isinstance(widget, QAbstractSlider):
        low = widget.minimum()
        widget.setValue(widget.maximum() if widget.value() == low else low)
    elif isinstance(widget, QDoubleSpinBox):
        bottom = widget.minimum()
        widget.setValue(widget.maximum() if widget.value() == bottom else bottom)
    else:
        raise TypeError(f'no perturbation for {type(widget).__name__}')


@pytest.mark.parametrize(
    ('key', 'attr'),
    [(key, attr) for key in MESH_UNSUPPORTED_KEYS for attr in MESH_REFUSED_WIDGETS.get(key, ())],
)
def test_each_mapped_control_writes_its_key(model: Any, key: str, attr: str) -> None:
    """On an ellipsoid, operating a control the table names changes the key it backs."""
    model._add_body_tab(_full_ellipsoid())
    body = model.sim_params['bodies'][0]
    before = copy.deepcopy(body.get(key, _MISSING))
    _perturb(getattr(_tab(model), attr))
    assert body.get(key, _MISSING) != before


@pytest.mark.parametrize('key', MESH_UNSUPPORTED_KEYS)
def test_a_mesh_body_disables_every_control_backing_a_refused_key(model: Any, key: str) -> None:
    """Every refused key maps to controls that a mesh body disables."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    tab = _tab(model)
    assert key in MESH_REFUSED_WIDGETS
    for attr in MESH_REFUSED_WIDGETS[key]:
        assert not getattr(tab, attr).isEnabled(), attr


@pytest.mark.parametrize('key', MESH_UNSUPPORTED_KEYS)
def test_a_disabled_control_carries_the_validators_sentence(model: Any, key: str) -> None:
    """A disabled control's tooltip is the validator's refusal of the keys it writes."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    tab = _tab(model)
    for attr in MESH_REFUSED_WIDGETS[key]:
        backed = [k for k in MESH_UNSUPPORTED_KEYS if attr in MESH_REFUSED_WIDGETS[k]]
        assert getattr(tab, attr).toolTip() == _refusal(backed), attr


@pytest.mark.parametrize('key', MESH_UNSUPPORTED_KEYS)
def test_an_ellipsoid_body_reenables_the_controls(model: Any, key: str) -> None:
    """Switching back to an ellipsoid re-enables the controls and clears the refusal."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    _set_shape(model, 'ellipsoid')
    tab = _tab(model)
    for attr in MESH_REFUSED_WIDGETS[key]:
        widget = getattr(tab, attr)
        assert widget.isEnabled(), attr
        assert 'polyhedral_mesh' not in widget.toolTip(), attr


def test_a_new_body_built_as_a_mesh_starts_disabled(model: Any) -> None:
    """A body tab built for a mesh body opens with the refused controls disabled."""
    model._add_body_tab({'name': 'M', 'shape_model': 'polyhedral_mesh', 'axis1': 30.0})
    tab = _tab(model)
    enabled = [
        attr
        for attrs in MESH_REFUSED_WIDGETS.values()
        for attr in attrs
        if getattr(tab, attr).isEnabled()
    ]
    assert enabled == []


@pytest.mark.parametrize(
    'body',
    [None, _FULL_ELLIPSOID],
    ids=['default_new_body', 'every_refused_key_set'],
)
def test_the_editor_saves_a_mesh_body_its_validator_loads(
    model: Any, body: dict[str, Any] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save Scene writes a loadable file after a body is switched to a mesh."""
    model._add_body_tab(None if body is None else _full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    path = tmp_path / 'mesh_scene.yaml'
    errors: list[str] = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *_a, **_k: (str(path), ''))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **_k: errors.append(str(a[2])))
    model._save_scene()
    assert errors == []
    assert load_sim_scene(path)['bodies'][0]['shape_model'] == 'polyhedral_mesh'


def test_switching_to_a_mesh_moves_the_refused_keys_out_of_the_scene(model: Any) -> None:
    """The body the editor holds carries no refused key while it is a mesh."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    body = model.sim_params['bodies'][0]
    assert [key for key in MESH_UNSUPPORTED_KEYS if key in body] == []
    validate_sim_params(model.sim_params)


def test_switching_back_to_an_ellipsoid_restores_the_keys(model: Any) -> None:
    """A mesh round trip gives the body back every refused key it carried."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    _set_shape(model, 'ellipsoid')
    assert model.sim_params['bodies'][0] == _full_ellipsoid()


def test_the_restore_survives_a_tab_rebuild(model: Any) -> None:
    """Keys stashed by a mesh switch come back after the body tabs are rebuilt."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    model._add_body_tab()
    _set_shape(model, 'ellipsoid')
    assert model.sim_params['bodies'][0] == _full_ellipsoid()


def test_a_rebuilt_mesh_tab_shows_the_stashed_values(model: Any) -> None:
    """A tab rebuilt while the body is a mesh shows the values switching back restores."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    model._add_body_tab()
    tab = _tab(model)
    assert tab.photometry_group.isChecked()  # type: ignore[attr-defined]
    assert tab.crater_min_radius_spin.value() == pytest.approx(0.06)  # type: ignore[attr-defined]


def test_a_mesh_round_trip_adds_no_keys(model: Any) -> None:
    """A body that carried no refused key gains none from a mesh round trip."""
    model._add_body_tab({'name': 'BARE', 'shape_model': 'ellipsoid', 'axis1': 40.0})
    _set_shape(model, 'polyhedral_mesh')
    _set_shape(model, 'ellipsoid')
    assert [key for key in MESH_UNSUPPORTED_KEYS if key in model.sim_params['bodies'][0]] == []


def test_a_saved_mesh_scene_loads_back_unchanged(model: Any, tmp_path: Path) -> None:
    """The saved mesh body is the body the editor holds, so the file round-trips."""
    model._add_body_tab(_full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    path = tmp_path / 'mesh_scene.yaml'
    save_sim_scene(model.sim_params, path)
    assert load_sim_scene(path)['bodies'] == model.sim_params['bodies']


def test_the_widget_table_covers_exactly_the_refused_keys() -> None:
    """The editor's table names the validator's keys, each with at least one control."""
    assert set(MESH_REFUSED_WIDGETS) == set(MESH_UNSUPPORTED_KEYS)
    assert [key for key, attrs in MESH_REFUSED_WIDGETS.items() if not attrs] == []


def _operate(widget: QWidget) -> bool:
    """Operate one control the way an operator would; return whether it was one."""
    if isinstance(widget, QGroupBox):
        if not widget.isCheckable():
            return False
        # An odd number of toggles leaves it flipped, after writing both states.
        for _ in range(3):
            widget.setChecked(not widget.isChecked())
    elif isinstance(widget, QCheckBox):
        widget.setChecked(not widget.isChecked())
    elif isinstance(widget, QComboBox):
        for index in range(widget.count()):
            widget.setCurrentIndex(index)
    elif isinstance(widget, QAbstractSlider):
        widget.setValue(widget.maximum() // 2 + 1)
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(widget.minimum() + (widget.maximum() - widget.minimum()) * 0.37))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(widget.minimum() + (widget.maximum() - widget.minimum()) * 0.37)
    elif isinstance(widget, QPushButton):
        widget.click()
    elif isinstance(widget, QLineEdit) and not isinstance(widget.parentWidget(), QAbstractSpinBox):
        widget.setText(widget.text() + 'x')
    else:
        return False
    return True


@pytest.mark.parametrize(
    'body',
    [None, _FULL_ELLIPSOID],
    ids=['default_new_body', 'every_refused_key_set'],
)
def test_operating_every_enabled_control_on_a_mesh_body_keeps_the_scene_valid(
    model: Any, body: dict[str, Any] | None
) -> None:
    """Whatever an operator does with a mesh body's live controls, the scene validates.

    Every control the tab leaves enabled is operated (each check box and checkable
    group toggled, each combo stepped through its items, each slider and spin box
    moved, each button pressed), except Delete and the shape combo, which would
    end the mesh body.
    """
    model._add_body_tab(None if body is None else _full_ellipsoid())
    _set_shape(model, 'polyhedral_mesh')
    tab = _tab(model)
    skip = {id(tab.shape_combo)}  # type: ignore[attr-defined]
    operated = []
    for widget in tab.findChildren(QWidget):
        if not widget.isEnabled() or id(widget) in skip:
            continue
        if isinstance(widget, QPushButton) and widget.text() == 'Delete':
            continue
        if _operate(widget):
            operated.append(widget)
    assert operated != []
    validate_sim_params(copy.deepcopy(model.sim_params))


@pytest.mark.parametrize(
    ('attr', 'key'),
    [
        ('mesh_seed_spin', 'mesh_seed'),
        ('mesh_n_lat_spin', 'mesh_n_lat'),
        ('mesh_n_lon_spin', 'mesh_n_lon'),
    ],
)
def test_an_integer_mesh_spin_writes_an_integer(model: Any, attr: str, key: str) -> None:
    """The mesh seed, lat bands, and lon divisions spins write ints the validator accepts."""
    model._add_body_tab()
    _set_shape(model, 'polyhedral_mesh')
    spin = getattr(_tab(model), attr)
    spin.setValue(spin.value() + 1)
    assert type(model.sim_params['bodies'][0][key]) is int
    validate_sim_params(copy.deepcopy(model.sim_params))
