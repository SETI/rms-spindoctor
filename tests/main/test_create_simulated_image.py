"""GUI smoke tests for ``spindoctor.cli.sd_create_simulated_image``.

These cover the ``_load_scene`` YAML-load path, specifically that a
``ring_system`` block round-trips into the data model, and that a missing or
null ``closest_planet`` falls back to ``SATURN`` without raising
(``QComboBox.findText(None)`` would otherwise raise ``TypeError``).
"""

import importlib
import os
from pathlib import Path
from typing import Any, cast

import pytest
from ruamel.yaml import YAML

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
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

# The driver is an importable module under the ``spindoctor.cli`` package.
ncsi = importlib.import_module('spindoctor.cli.sd_create_simulated_image')


@pytest.fixture
def qapp() -> QApplication:
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image GUI model."""
    return ncsi.CreateSimulatedImageModel()


def _load_scene_with(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    """Merge ``payload`` into a minimal valid scene, write it, and drive ``_load_scene``."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'params',
        'instrument': 'generic',
        'size_v': 128,
        'size_u': 128,
        'random_seed': 42,
        **payload,
    }
    scene_path = tmp_path / 'params.yaml'
    yaml = YAML(typ='safe')
    with scene_path.open('w') as handle:
        yaml.dump(scene, handle)
    monkeypatch.setattr(
        QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *a, **k: (str(scene_path), 'YAML')),
    )

    # ``_load_scene`` swallows exceptions into a critical dialog; surface any
    # such failure as a test error instead so a regression is visible.
    def _fail_on_critical(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'_load_scene raised an error dialog: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_fail_on_critical))
    model._load_scene()


def test_load_ring_system_block_round_trips(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a ring_system block carries it into the data model verbatim."""
    ring_system = {
        'geometry': {
            'center_v': 64.0,
            'center_u': 64.0,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [
            {
                'name': 'R1',
                'kind': 'ringlet',
                'tau': 1.0,
                'width': 10.0,
                'navigable': True,
                'orbit': {'a': 30.0},
            }
        ],
    }
    _load_scene_with(monkeypatch, model, tmp_path, {'ring_system': ring_system})
    assert model.sim_params['ring_system'] == ring_system


def test_load_missing_closest_planet_defaults_to_saturn(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene with no ``closest_planet`` key falls back to ``SATURN``."""
    _load_scene_with(monkeypatch, model, tmp_path, {})
    assert model.sim_params['closest_planet'] == 'SATURN'


def test_load_null_closest_planet_defaults_to_saturn(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """An explicit ``closest_planet: null`` falls back to ``SATURN`` without raising."""
    _load_scene_with(monkeypatch, model, tmp_path, {'closest_planet': None})
    assert model.sim_params['closest_planet'] == 'SATURN'


def test_load_explicit_closest_planet_is_preserved(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A real ``closest_planet`` value is preserved in the data model."""
    _load_scene_with(monkeypatch, model, tmp_path, {'closest_planet': 'JUPITER'})
    assert model.sim_params['closest_planet'] == 'JUPITER'


def test_default_instrument_is_generic(model: Any) -> None:
    """A fresh model defaults to the generic (instrument-agnostic) frame."""
    assert model.sim_params['instrument'] == 'generic'


def test_instrument_combo_drives_sim_params(model: Any) -> None:
    """Selecting an instrument updates the data model's instrument."""
    model._instrument_combo.setCurrentText('coiss_nac')
    assert model.sim_params['instrument'] == 'coiss_nac'


def test_load_instrument_syncs_combo(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene with an instrument syncs both the model and the combo."""
    _load_scene_with(monkeypatch, model, tmp_path, {'instrument': 'gossi'})
    assert model.sim_params['instrument'] == 'gossi'
    assert model._instrument_combo.currentText() == 'gossi'


def test_load_missing_instrument_defaults_to_generic(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene without an instrument key falls back to generic."""
    _load_scene_with(monkeypatch, model, tmp_path, {'random_seed': 7})
    assert model.sim_params['instrument'] == 'generic'


def test_load_preserves_noise_block(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene round-trips the catalog-only noise block."""
    noise = {'poisson': False, 'read_noise_dn': 12.0}
    _load_scene_with(monkeypatch, model, tmp_path, {'noise': noise})
    assert model.sim_params['noise'] == noise


def test_load_preserves_stray_light_block(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene round-trips the catalog-only stray_light block."""
    stray = {'amplitude': 0.3, 'model': 'radial'}
    _load_scene_with(monkeypatch, model, tmp_path, {'optics': {'stray_light': stray}})
    assert model.sim_params['optics']['stray_light'] == stray


def test_default_has_no_dead_background_noise_key(model: Any) -> None:
    """The inert background_noise_intensity key is gone from the defaults."""
    assert 'background_noise_intensity' not in model.sim_params


def test_default_noise_block_present(model: Any) -> None:
    """A fresh model carries a detector-noise block with Poisson on."""
    assert model.sim_params['noise']['poisson'] is True


def test_poisson_toggle_updates_noise(model: Any) -> None:
    """Unchecking Poisson writes through to the noise block."""
    model._poisson_check.setChecked(False)
    assert model.sim_params['noise']['poisson'] is False


def test_read_noise_spin_updates_noise(model: Any) -> None:
    """The read-noise spin writes read_noise_dn into the noise block."""
    model._read_noise_spin.setValue(12.5)
    assert model.sim_params['noise']['read_noise_dn'] == 12.5


def test_cosmic_ray_spin_updates_noise(model: Any) -> None:
    """The cosmic-ray spin writes cosmic_ray_rate_per_sec into the noise block."""
    model._cosmic_ray_spin.setValue(0.002)
    assert model.sim_params['noise']['cosmic_ray_rate_per_sec'] == 0.002


def test_missing_data_spin_updates_noise(model: Any) -> None:
    """The missing-data spin writes missing_data_rate into the noise block."""
    model._missing_data_spin.setValue(0.1)
    assert model.sim_params['noise']['missing_data_rate'] == 0.1


def test_load_noise_block_syncs_widgets(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a noise block syncs the panel widgets."""
    _load_scene_with(
        monkeypatch, model, tmp_path, {'noise': {'poisson': False, 'read_noise_dn': 9.0}}
    )
    assert model._poisson_check.isChecked() is False
    assert model._read_noise_spin.value() == 9.0


def test_stray_light_defaults_off(model: Any) -> None:
    """A fresh model has no stray-light amplitude set (off)."""
    assert model._stray_amplitude_spin.value() == 0.0


def test_stray_amplitude_updates_block(model: Any) -> None:
    """The amplitude spin writes into the optics.stray_light block."""
    model._stray_amplitude_spin.setValue(0.4)
    assert model.sim_params['optics']['stray_light']['amplitude'] == 0.4


def test_stray_direction_updates_block(model: Any) -> None:
    """The direction spin writes direction_deg into the optics.stray_light block."""
    model._stray_direction_spin.setValue(45.0)
    assert model.sim_params['optics']['stray_light']['direction_deg'] == 45.0


def test_stray_model_updates_block(model: Any) -> None:
    """The model combo writes the optics.stray_light model."""
    model._stray_model_combo.setCurrentText('radial')
    assert model.sim_params['optics']['stray_light']['model'] == 'radial'


def test_load_stray_light_syncs_widgets(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading an optics.stray_light block syncs the panel widgets."""
    _load_scene_with(
        monkeypatch,
        model,
        tmp_path,
        {'optics': {'stray_light': {'amplitude': 0.25, 'direction_deg': 90.0, 'model': 'radial'}}},
    )
    assert model._stray_amplitude_spin.value() == 0.25
    assert model._stray_direction_spin.value() == 90.0
    assert model._stray_model_combo.currentText() == 'radial'


def test_saturation_overlay_defaults_off(model: Any) -> None:
    """The saturation overlay starts disabled."""
    assert model._show_saturation_overlay is False


def test_toggle_saturation_overlay_sets_flag(model: Any) -> None:
    """Checking the overlay box flips the display flag."""
    model._saturation_overlay_check.setChecked(True)
    assert model._show_saturation_overlay is True


def test_saturation_dn_for_raw_instrument(model: Any) -> None:
    """A raw-DN instrument reports its saturation DN."""
    model.sim_params['instrument'] = 'coiss_nac'
    assert model._current_saturation_dn() == 4095.0


def test_saturation_dn_none_for_calibrated_instrument(model: Any) -> None:
    """A calibrated-IF instrument has no saturation DN."""
    model.sim_params['instrument'] = 'vgiss'
    assert model._current_saturation_dn() is None


def test_saturation_overlay_updates_status(model: Any) -> None:
    """Enabling the overlay renders and reports a saturation fraction."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._update_image()
    model._saturation_overlay_check.setChecked(True)
    assert 'Saturated' in model._saturation_label.text()


def test_saturation_overlay_off_clears_status(model: Any) -> None:
    """Disabling the overlay clears the saturation status."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._update_image()
    model._saturation_overlay_check.setChecked(True)
    model._saturation_overlay_check.setChecked(False)
    assert model._saturation_label.text() == ''


def test_psf_preview_collapsed_by_default(model: Any) -> None:
    """The PSF preview pane starts collapsed with its inset hidden."""
    assert model._psf_group.isChecked() is False
    assert model._psf_image_label.isVisibleTo(model._psf_group) is False


def test_psf_preview_expands_and_annotates(model: Any) -> None:
    """Expanding the pane shows the PSF inset and its sigma / FWHM."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._psf_group.setChecked(True)
    assert model._psf_image_label.isVisibleTo(model._psf_group) is True
    assert not model._psf_image_label.pixmap().isNull()
    assert 'sigma' in model._psf_info_label.text()


def test_psf_preview_tracks_instrument(model: Any) -> None:
    """Switching instruments updates the PSF sigma shown."""
    model._psf_group.setChecked(True)
    model._instrument_combo.setCurrentText('coiss_nac')
    coiss_text = model._psf_info_label.text()
    model._instrument_combo.setCurrentText('gossi')
    gossi_text = model._psf_info_label.text()
    assert coiss_text != gossi_text


def test_new_body_defaults_to_ellipsoid(model: Any) -> None:
    """A newly added body uses the ellipsoid shape model by default."""
    model._add_body_tab()
    assert model.sim_params['bodies'][0]['shape_model'] == 'ellipsoid'


def test_new_body_has_mesh_fields(model: Any) -> None:
    """A newly added body carries the mesh shape parameters."""
    model._add_body_tab()
    body = model.sim_params['bodies'][0]
    assert body['mesh_lumpiness'] == 0.3
    assert body['pose_euler_deg'] == [0.0, 0.0, 0.0]


def test_body_shape_model_field_updates(model: Any) -> None:
    """Setting a body's shape_model writes through to the data model."""
    model._add_body_tab()
    model._on_body_field(0, 'shape_model', 'polyhedral_mesh')
    assert model.sim_params['bodies'][0]['shape_model'] == 'polyhedral_mesh'


def test_body_pose_handler_updates_axis(model: Any) -> None:
    """The pose handler writes one axis of the body's mesh pose."""
    model._add_body_tab()
    model._on_body_pose(0, 1, 90.0)
    assert model.sim_params['bodies'][0]['pose_euler_deg'] == [0.0, 90.0, 0.0]


def test_body_pose_handler_pads_missing_pose(model: Any) -> None:
    """The pose handler tolerates a body that lacks a pose list."""
    model.sim_params['bodies'].append({'name': 'B', 'center_v': 1.0, 'center_u': 1.0})
    model._on_body_pose(0, 2, 45.0)
    assert model.sim_params['bodies'][0]['pose_euler_deg'] == [0.0, 0.0, 45.0]


def _no_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any error dialog raise so failures surface as test errors."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


def test_save_scene_writes_valid_yaml(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving a scene writes a YAML the schema validates."""
    from spindoctor.sim.scene import load_sim_scene

    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    out = tmp_path / 'myscene.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    scene = load_sim_scene(out)
    assert scene['instrument'] == 'coiss_nac'
    assert scene['scene_name'] == 'myscene'
    assert scene['size_v'] == 128
    assert scene['size_u'] == 128


def test_load_scene_populates_model(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene YAML populates the data model from it."""
    scene_yaml = tmp_path / 'loadme.yaml'
    scene_yaml.write_text(
        'schema_version: 2\nscene_name: loadme\ninstrument: gossi\n'
        'size_v: 96\nsize_u: 96\nrandom_seed: 5\n'
    )
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(scene_yaml), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model.sim_params['instrument'] == 'gossi'
    assert model.sim_params['size_v'] == 96


def test_scene_round_trips_through_gui(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene saved then loaded preserves instrument, size, and a body."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 100
    model.sim_params['size_u'] = 100
    model.sim_params['bodies'] = [
        {'name': 'B', 'center_v': 50.0, 'center_u': 50.0, 'axis1': 40.0, 'axis2': 40.0}
    ]
    out = tmp_path / 'rt.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    model.sim_params['instrument'] = 'generic'
    model._load_scene()
    assert model.sim_params['instrument'] == 'coiss_nac'
    assert model.sim_params['size_v'] == 100
    assert model.sim_params['bodies'][0]['name'] == 'B'


def test_offset_rotation_handler_updates_param(model: Any) -> None:
    """The offset-rotation spin writes the planted roll."""
    model._on_offset_rotation(1.5)
    assert model.sim_params['offset_rotation_deg'] == 1.5


def test_exposure_handler_updates_param(model: Any) -> None:
    """The exposure spin writes exposure_sec."""
    model._on_exposure(2.5)
    assert model.sim_params['exposure_sec'] == 2.5


def test_fit_rotation_combo_tristate(model: Any) -> None:
    """The fit-camera-rotation combo maps to True / False / unset."""
    model._on_fit_rotation('on')
    assert model.sim_params['fit_camera_rotation'] is True
    model._on_fit_rotation('off')
    assert model.sim_params['fit_camera_rotation'] is False
    model._on_fit_rotation('(inherit)')
    assert 'fit_camera_rotation' not in model.sim_params


def test_midtime_handler_omits_when_blank(model: Any) -> None:
    """A blank midtime removes the key; a value sets it."""
    model._on_midtime('2010-01-01T00:00:00Z')
    assert model.sim_params['midtime_utc'] == '2010-01-01T00:00:00Z'
    model._on_midtime('   ')
    assert 'midtime_utc' not in model.sim_params


def test_noise_bias_and_bloom_handlers(model: Any) -> None:
    """The new noise spins write into the noise block."""
    model._on_bias(18.0)
    model._on_bloom(3)
    assert model.sim_params['noise']['bias_dn'] == 18.0
    assert model.sim_params['noise']['bloom_length'] == 3


def test_stray_center_writes_only_while_enabled(model: Any) -> None:
    """A stray-light center spin writes its key only while the enable is on."""
    model._on_stray_center_v(40.0)
    assert 'center_v' not in model.sim_params.get('optics', {}).get('stray_light', {})
    model._stray_center_check.setChecked(True)
    model._on_stray_center_v(40.0)
    assert model.sim_params['optics']['stray_light']['center_v'] == 40.0


def test_body_seed_auto_omits(model: Any) -> None:
    """A crater seed of -1 (Auto) removes the key; a value sets it."""
    model.sim_params['bodies'] = [{'name': 'B'}]
    model._on_body_seed(0, 11)
    assert model.sim_params['bodies'][0]['seed'] == 11
    model._on_body_seed(0, -1)
    assert 'seed' not in model.sim_params['bodies'][0]


def test_full_parameter_round_trip(
    model: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every newly exposed parameter survives a GUI save -> load scene cycle."""
    model.sim_params.update(
        {
            'instrument': 'coiss_nac',
            'size_v': 128,
            'size_u': 128,
            'offset_v': 1.4,
            'offset_u': -0.6,
            'offset_rotation_deg': 1.5,
            'exposure_sec': 2.5,
            'midtime_utc': '2010-01-01T00:00:00Z',
            'fit_camera_rotation': True,
            'noise': {'poisson': True, 'read_noise_dn': 4.0, 'bias_dn': 18.0, 'bloom_length': 3},
            'optics': {
                'stray_light': {
                    'amplitude': 0.3,
                    'direction_deg': 35.0,
                    'model': 'radial',
                    'center_v': 40.0,
                    'center_u': 50.0,
                }
            },
            'bodies': [
                {
                    'name': 'B',
                    'center_v': 64.0,
                    'center_u': 64.0,
                    'axis1': 90.0,
                    'axis2': 70.0,
                    'axis3': 60.0,
                    'shape_model': 'polyhedral_mesh',
                    'mesh_lumpiness': 0.4,
                    'mesh_seed': 3,
                    'mesh_n_lat': 20,
                    'mesh_n_lon': 40,
                    'pose_euler_deg': [10.0, 35.0, 0.0],
                    'seed': 11,
                    'km_per_pixel': 5.0,
                    'nav_override': {
                        'shape_model': 'ellipsoid',
                        'mesh_lumpiness': 0.0,
                        'pose_euler_deg': [10.0, 35.0, 0.0],
                    },
                }
            ],
            'stars': [
                {
                    'name': 'S',
                    'v': 30.0,
                    'u': 40.0,
                    'vmag': 6.0,
                    'move_v': 1.0,
                    'move_u': -2.0,
                    'catalog_name': 'UCAC4',
                }
            ],
        }
    )
    out = tmp_path / 'full_round_trip.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    model._apply_params_dict({'size_v': 64, 'size_u': 64})  # wipe state
    model._load_scene()
    p = model.sim_params
    assert p['offset_rotation_deg'] == 1.5
    assert p['exposure_sec'] == 2.5
    assert p['midtime_utc'] == '2010-01-01T00:00:00Z'
    assert p['fit_camera_rotation'] is True
    assert p['noise']['bias_dn'] == 18.0
    assert p['noise']['bloom_length'] == 3
    assert p['optics']['stray_light']['center_v'] == 40.0
    body = p['bodies'][0]
    assert body['mesh_n_lat'] == 20
    assert body['mesh_n_lon'] == 40
    assert body['seed'] == 11
    assert body['km_per_pixel'] == 5.0
    assert body['nav_override']['shape_model'] == 'ellipsoid'
    star = p['stars'][0]
    assert star['move_v'] == 1.0
    assert star['catalog_name'] == 'UCAC4'


def test_help_prints_usage_without_qt(capsys: pytest.CaptureFixture[str]) -> None:
    """--help exits before any Qt application is constructed."""
    from spindoctor.cli.sim_editor.main_window import main

    with pytest.raises(SystemExit) as excinfo:
        main(['--help'])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert 'sd_create_simulated_image' in out
    assert 'scene' in out


def test_arg_parser_accepts_optional_scene_path() -> None:
    """The positional scene argument parses to a Path (or None when omitted)."""
    from spindoctor.cli.sim_editor.main_window import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(['scene.yaml'])
    assert args.scene == Path('scene.yaml')
    args = parser.parse_args([])
    assert args.scene is None


def test_load_scene_file_opens_scene(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The command-line loader applies a scene through the Load Scene machinery."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))
    scene = {
        'schema_version': 2,
        'scene_name': 'cli_open',
        'instrument': 'coiss_nac',
        'size_v': 96,
        'size_u': 96,
        'random_seed': 5,
        'offset_v': 1.5,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 48.0,
                'center_u': 48.0,
                'axis1': 40.0,
                'axis2': 32.0,
                'axis3': 30.0,
            }
        ],
    }
    path = tmp_path / 'cli_open.yaml'
    yaml = YAML(typ='safe')
    with path.open('w') as handle:
        yaml.dump(scene, handle)
    model.load_scene_file(path)
    assert model.sim_params['size_v'] == 96
    assert model.sim_params['offset_v'] == 1.5
    assert model.sim_params['bodies'][0]['name'] == 'RHEA'


def test_load_scene_file_reports_invalid_scene(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """An invalid scene surfaces the error dialog and leaves the model as-is."""
    calls: list[str] = []
    monkeypatch.setattr(
        QMessageBox, 'critical', staticmethod(lambda *a, **k: calls.append(str(a[2])))
    )
    path = tmp_path / 'broken.yaml'
    path.write_text('schema_version: 2\nscene_name: broken\n')
    before = dict(model.sim_params)
    model.load_scene_file(path)
    assert len(calls) == 1
    assert 'Failed to load scene' in calls[0]
    assert model.sim_params['size_v'] == before['size_v']
