"""Round-trip acceptance tests for the ``sd_create_simulated_image`` editor.

These prove the editor's data model preserves loss-free scene round-tripping:

* A v2 scene exercising the full current key inventory loads into the editor,
  survives a single edit through the editor's data model, and re-saves with
  every other block byte-for-byte identical in meaning (loaded dicts compared,
  not raw bytes).
* A scene authored entirely through the GUI (added body / ring / star) is
  idempotent under save -> load -> save -> load.
* Loading a scene syncs every group / checkbox state, and a partial block
  survives an edit without gaining backfilled keys.

The per-tab widget-state tests (absent-key discipline, per-key edits) live in
``test_sim_editor_tabs``.  Qt runs headless (offscreen platform).
"""

import os
from pathlib import Path
from typing import Any, cast

import pytest

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

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.sim.scene import _ALLOWED_KEYS, load_sim_scene, save_sim_scene

# A v2 scene exercising the full current key inventory: every top-level key,
# and every per-object idealized and truth key across two bodies (one per
# render path), a ring, and a star.
_FULL_SCENE: dict[str, Any] = {
    'instrument': 'coiss_nac',
    'size_v': 128,
    'size_u': 128,
    'random_seed': 7,
    'exposure_sec': 2.5,
    'offset_v': 1.25,
    'offset_u': -0.75,
    'offset_rotation_deg': 3.5,
    'midtime_utc': '2010-01-01T00:00:00Z',
    'closest_planet': 'SATURN',
    'time': 100.0,
    'ring_epoch': 50.0,
    'sky_counts': {'a': -3.0, 'b': 0.35, 'density_factor': 8.0, 'diffuse_e_per_px': 2.5},
    'star_catalog_scatter_px': 0.4,
    'expected': {
        'status': 'success',
        'confidence_tier': 'high',
        'status_reason': 'ok',
    },
    'fit_camera_rotation': True,
    'noise': {
        'poisson': True,
        'read_noise_dn': 6.0,
        'cosmic_ray_rate_per_sec': 0.001,
        'missing_data_rate': 0.01,
        'bias_dn': 18.0,
        'bloom_length': 4,
        'signal_full_scale_frac': 0.6,
        'pixel_area_cm2': 1.5,
    },
    'oversample': 2,
    'optics': {
        'psf': {'sigma_v': 0.6, 'sigma_u': 0.5, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
        'smear': [
            {'dv_px': 1.0, 'du_px': 0.0, 'object_class': 'all'},
            {'dv_px': 0.0, 'du_px': 2.0, 'object_class': 'stars'},
        ],
        'distortion': {
            'k1': 0.01,
            'k2': 0.0,
            'center_v': 40.0,
            'center_u': 50.0,
            'nonradial_rms_px': 0.3,
        },
        'ghosts': [{'dv_px': 10.0, 'du_px': -5.0, 'amplitude': 0.01, 'defocus_sigma': 2.0}],
        'stray_light': {
            'amplitude': 0.3,
            'direction_deg': 35.0,
            'model': 'radial',
            'center_v': 40.0,
            'center_u': 50.0,
        },
    },
    'spk_error': {'dv_px': 0.5, 'du_px': -0.5, 'reference_range_km': 1000.0},
    'detector': {
        'gain_state': 2,
        'detector_model': 'ccd',
        'exposure_ref_sec': 1.0,
        'quantization': 'exact',
    },
    'artifacts': {
        'instrument_defaults': True,
        'adversarial': True,
        # One representative mode per stage family: a telemetry structured loss,
        # a detector-electronics mode, and a routed quantization mode -- all
        # available on coiss_nac -- so the round-trip exercises the generated
        # mode rows, not only the two switches.
        'missing_lines': {'incidence': 2.0, 'contiguous_run': True},
        'banding_coherent': {'incidence': 0.5, 'amplitude_e': 3.0},
        'quantization_lut': {'incidence': 1.0},
    },
    'instrument_config': {'inherit': 'coiss_nac'},
    # The full body key inventory is split across the two render paths it
    # belongs to: the mesh body carries the mesh-appearance keys (relief,
    # shading, detail, pose scatter), and the ellipsoid body carries the
    # ellipsoid-only appearance keys the validator rejects on a mesh body
    # (craters, photometric laws, surface texture, transits, atmosphere).
    'bodies': [
        {
            'name': 'Mimas',
            'shape_model': 'polyhedral_mesh',
            'center_v': 40.0,
            'center_u': 40.0,
            'axis1': 60.0,
            'axis2': 50.0,
            'axis3': 45.0,
            'illumination_angle': 45.0,
            'phase_angle': 30.0,
            'range_km': 1000.0,
            'km_per_pixel': 5.0,
            'mesh_lumpiness': 0.4,
            'mesh_n_lat': 20,
            'mesh_n_lon': 40,
            'mesh_seed': 3,
            'mesh_detail_octaves': 2,
            'pose_euler_deg': [10.0, 35.0, 0.0],
            'limb_relief_rms': 0.015,
            'limb_relief_corr_deg': 12.0,
            'shading': 'gouraud',
            'pose_scatter': {'sigma_deg': 1.5},
            'seed': 11,
            'anti_aliasing': 0.5,
            'nav_override': {
                'shape_model': 'ellipsoid',
                'mesh_lumpiness': 0.0,
                'pose_euler_deg': [10.0, 35.0, 0.0],
            },
        },
        {
            'name': 'Tethys',
            'center_v': 96.0,
            'center_u': 96.0,
            'axis1': 44.0,
            'axis2': 36.0,
            'axis3': 32.0,
            'rotation_z': 12.0,
            'rotation_tilt': 8.0,
            'illumination_angle': 60.0,
            'phase_angle': 35.0,
            'range_km': 3000.0,
            'crater_fill': 1.5,
            'crater_min_radius': 0.05,
            'crater_max_radius': 0.2,
            'crater_power_law_exponent': 3.0,
            'crater_relief_scale': 0.6,
            'photometric_law': 'minnaert',
            'minnaert_k': 0.6,
            'opposition_surge': {'amplitude': 0.4, 'width_deg': 5.0},
            'albedo_texture': {
                'rms': 0.05,
                'corr_px': 18.0,
                'spots': [
                    {'lat_deg': 10.0, 'lon_deg': 45.0, 'radius_deg': 8.0, 'albedo_factor': 0.7}
                ],
            },
            'disc_texture': {
                'band_amplitude': 0.1,
                'band_wavenumber': 8.0,
                'band_phase_deg': 0.0,
                'storms': [
                    {'lat_deg': -20.0, 'lon_deg': 120.0, 'radius_deg': 6.0, 'albedo_factor': 1.3}
                ],
            },
            'transits': [
                {
                    'moon': {'dv_px': 10.0, 'du_px': -5.0, 'radius_px': 6.0, 'albedo_factor': 0.9},
                    'shadow': {'dv_px': 12.0, 'du_px': -4.0, 'radius_px': 6.0, 'darkness': 0.8},
                }
            ],
            'atmosphere': {
                'scale_height_px': 4.0,
                'tau_ref': 1.5,
                'ref_altitude_px': 1.0,
                'g': 0.6,
                'detached_px': 8.0,
            },
        },
    ],
    'ring_system': {
        'geometry': {
            'center_v': 64.0,
            'center_u': 64.0,
            'opening_deg_obs': 35.0,
            'opening_deg_sun': 25.0,
            'node_deg': 20.0,
        },
        'range_km': 2000.0,
        'km_per_pixel': 100.0,
        'phase_deg': 40.0,
        'features': [
            {
                'name': 'RingA',
                'kind': 'ringlet',
                'tau': 1.2,
                'width': 20.0,
                'navigable': True,
                'orbit': {
                    'a': 100.0,
                    'ae': 2.0,
                    'long_peri': 10.0,
                    'rate_peri': 0.5,
                    'modes': [{'m': 2, 'amp': 1.5, 'peri': 70.0}],
                    'edge_wave': {'amp': 1.0, 'wavelength': 8.0, 'damp': 0.5, 'lam0': 90.0},
                },
                'declared_orbit_sigma': {'sigma_a_px': 0.5, 'sigma_ae_px': 0.2},
                'orbit_error': {'delta_a_px': 1.0, 'delta_long_peri_deg': 5.0},
                'albedo': 0.6,
                'phase_g': -0.2,
            },
            {
                'name': 'WaveTrain',
                'kind': 'wave',
                'tau': 0.4,
                'wavelength': 6.0,
                'damping': 12.0,
                'orbit': {'a': 130.0},
            },
        ],
        'azimuthal': {
            'modulation': {'amplitude': 0.2, 'm': 2, 'phase_deg': 30.0},
            'shadow': {'start_deg': 100.0, 'extent_deg': 40.0, 'darkness': 0.9},
            'spokes': {
                'count': 3,
                'r_inner': 90.0,
                'r_outer': 120.0,
                'contrast': -0.4,
                'width_deg': 12.0,
            },
        },
        'moonlets': [
            {
                'a': 110.0,
                'lam_deg': 45.0,
                'radius_px': 1.5,
                'amplitude': 0.4,
                'propeller': {'length_deg': 20.0, 'width_px': 2.0, 'contrast': -0.6},
            }
        ],
    },
    'stars': [
        {
            'name': 'S1',
            'catalog_name': 'UCAC4',
            'v': 30.0,
            'u': 40.0,
            'vmag': 6.0,
            'spectral_class': 'G2',
            'move_v': 1.0,
            'move_u': -2.0,
            'psf_size': [11, 11],
            'psf_sigma': 1.2,
        },
        {
            # A non-navigable confounder star carrying every truth-side star key:
            # a planted catalog-position error, an unresolved companion, and a
            # variable-brightness delta.
            'name': 'S2',
            'catalog_name': 'UCAC4',
            'v': 80.0,
            'u': 90.0,
            'vmag': 8.0,
            'navigable': False,
            'catalog_error_v': 0.6,
            'catalog_error_u': -0.4,
            'delta_mag': 0.5,
            'companion': {'sep_px': 2.5, 'delta_mag': 1.5, 'angle_deg': 30.0},
        },
    ],
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


def _no_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any error dialog raise so a failed save/load surfaces as a test error."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


def _comparable(scene: dict[str, Any]) -> dict[str, Any]:
    """Drop the file-identity metadata so two saved scenes compare on content."""
    return {k: v for k, v in scene.items() if k not in ('schema_version', 'scene_name')}


def test_full_inventory_round_trip_preserves_other_blocks(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Editing one field leaves every other block identical after a save cycle."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    original = load_sim_scene(src)

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()

    # Edit exactly one field through the editor's data model (drives _on_random_seed).
    model._random_seed_spin.setValue(2024)
    assert model.sim_params['random_seed'] == 2024

    out = tmp_path / 'edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)

    # The edited field changed; nothing else did.
    assert resaved['random_seed'] == 2024
    for key in _ALLOWED_KEYS - {'schema_version', 'scene_name', 'random_seed'}:
        assert resaved.get(key) == original.get(key), f'block {key!r} did not survive'


def test_full_inventory_edit_is_the_only_change(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The nested body/ring/star blocks survive with identical meaning."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    original = load_sim_scene(src)

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._random_seed_spin.setValue(2024)

    out = tmp_path / 'edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)

    assert resaved['bodies'] == original['bodies']
    assert resaved['ring_system'] == original['ring_system']
    assert resaved['stars'] == original['stars']


def test_gui_authored_scene_save_load_idempotent(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A GUI-authored scene is stable under save -> load -> save -> load."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_body_tab()
    model._add_ring_tab()
    model._add_star_tab()  # GUI-added star carries a list psf_size

    file1 = tmp_path / 'gui1.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(file1), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded1 = load_sim_scene(file1)

    # Feed the loaded scene back through the editor and re-save.
    model._apply_params_dict(loaded1)
    file2 = tmp_path / 'gui2.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(file2), 'YAML'))
    )
    model._save_scene()
    loaded2 = load_sim_scene(file2)

    assert _comparable(loaded1) == _comparable(loaded2)


def test_gui_added_star_scene_saves_without_error(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving a scene with a GUI-added star succeeds and preserves psf_size."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_star_tab()

    out = tmp_path / 'star.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['stars'][0]['psf_size'] == [11, 11]


def test_match_navigator_survives_save_and_load(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving persists the authored match-navigator form and never mutates it.

    The renderer resolves the navigator-matched PSF only when it builds the
    kernel, so both the live editor state and the file keep the authored form
    and the checkbox is still set after a reload.
    """
    model.sim_params['instrument'] = 'coiss_wac'
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)

    out = tmp_path / 'floor.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    # Saving does not rewrite the live editor state.
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}
    saved = load_sim_scene(out)
    assert saved['optics']['psf'] == {'match_navigator': True}

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._load_scene()
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}
    assert model._psf_match_nav_check.isChecked() is True


def test_psf_sigma_u_widget_defaults_to_sigma_v(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A PSF block omitting sigma_u displays the renderer's default (sigma_v)."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'psf': {'sigma_v': 1.3}},
    }
    src = tmp_path / 'psf.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._psf_sigma_u_spin.value() == 1.3


def test_partial_distortion_edit_leaves_center_u_absent(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A k1 edit on a partial {k1, center_v} block never backfills center_u."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'distortion': {'k1': 0.01, 'center_v': 40.0}},
    }
    src = tmp_path / 'partial_distortion.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._distortion_k1_spin.setValue(0.02)
    assert 'center_u' not in model.sim_params['optics']['distortion']

    out = tmp_path / 'partial_distortion_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['optics']['distortion'] == {'k1': 0.02, 'center_v': 40.0}


def test_partial_distortion_center_u_displays_frame_center(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The center-u spin shows the effective default (frame center) when absent."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'distortion': {'k1': 0.01, 'center_v': 40.0}},
    }
    src = tmp_path / 'partial_distortion.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._distortion_center_v_spin.value() == 40.0
    assert model._distortion_center_u_spin.value() == 32.0


def test_ring_spk_error_scene_authors_and_validates(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A ring + spk_error scene authored through the editor saves cleanly.

    spk_error requires range_km on the ring system; the first feature tab's
    physical-range control makes the key authorable (absent unless set).
    """
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    ring_tab = model._tabs.widget(tab_idx)
    ring_tab.range_km_check.click()
    ring_tab.range_km_spin.setValue(2.0e6)
    assert model.sim_params['ring_system']['range_km'] == 2.0e6
    model._spk_error_group.setChecked(True)

    out = tmp_path / 'ring_spk.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['ring_system']['range_km'] == 2.0e6
    assert loaded['spk_error']['reference_range_km'] > 0.0


def test_partial_detector_scene_edit_preserves_authored_keys(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Nudging one detector spin keeps an authored quantization key intact."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'detector': {'quantization': 'sqrt_lut'},
    }
    src = tmp_path / 'partial.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._detector_gain_state_spin.setValue(3)

    out = tmp_path / 'partial_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['detector'] == {'quantization': 'sqrt_lut', 'gain_state': 3}


def test_vgiss_scene_stays_vidicon_through_an_edit(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Editing a vgiss detector spin never backfills a ccd detector_model."""
    scene = {
        'instrument': 'vgiss',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'detector': {'exposure_ref_sec': 2.0},
    }
    src = tmp_path / 'vgiss.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    # The widgets display the vgiss catalog defaults for unauthored keys.
    assert model._detector_model_combo.currentText() == 'vidicon'
    model._detector_exposure_ref_spin.setValue(3.0)

    out = tmp_path / 'vgiss_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['detector'] == {'exposure_ref_sec': 3.0}

    from spindoctor.sim.forward.detector.params import resolve_detector_params

    assert resolve_detector_params(resaved).detector_model == 'vidicon'


def test_load_full_optics_syncs_group_states(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full-inventory scene checks every optics/artifacts group."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._psf_optics_group.isChecked() is True
    assert len(model._smear_rows) == 2
    assert model._ghosts_group.isChecked() is True
    assert model._spk_error_group.isChecked() is True
    assert model._detector_group.isChecked() is True
    assert model._instrument_defaults_check.isChecked() is True


def test_load_then_disable_optics_clears_blocks(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A loaded sub-block can be disabled back to absence after sync."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._detector_group.setChecked(False)
    assert 'detector' not in model.sim_params


def test_load_full_scene_syncs_mode_rows(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full-inventory scene checks its adversarial switch and modes."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._adversarial_check.isChecked() is True
    assert model._mode_rows['missing_lines'].group.isChecked() is True
    assert model._mode_rows['banding_coherent'].group.isChecked() is True
    assert model._mode_rows['quantization_lut'].group.isChecked() is True
    incidence = model._mode_rows['missing_lines']._scalar_widgets['incidence'].value()
    assert incidence == 2.0


def test_gui_authored_mode_scene_validates(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene authored through the mode rows saves and reloads cleanly."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(2.0)
    model._adversarial_check.setChecked(True)

    out = tmp_path / 'modes.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['artifacts']['missing_lines'] == {'incidence': 2.0}
    assert loaded['artifacts']['adversarial'] is True


def test_scene_without_sky_counts_does_not_gain_it(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """An opened and re-saved scene never gains a sky_counts block.

    The whole block follows the absent-key discipline: loading a scene without
    the key leaves the background-sky group unchecked, and saving writes no
    default block.
    """
    scene = {'instrument': 'coiss_nac', 'size_v': 64, 'size_u': 64, 'random_seed': 1}
    src = tmp_path / 'no_sky.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._sky_counts_check.isChecked() is False
    assert 'sky_counts' not in model.sim_params

    out = tmp_path / 'no_sky_resaved.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert 'sky_counts' not in resaved


def test_load_scene_with_sky_counts_checks_group(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene that authors sky_counts checks and populates the group."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._sky_counts_check.isChecked() is True
    assert model._sky_density_spin.value() == 8.0
    assert model._sky_diffuse_spin.value() == 2.5


def test_load_full_scene_syncs_star_asymmetry_controls(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full scene populates the scene-level star controls."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._star_scatter_check.isChecked() is True
    assert model._star_scatter_spin.value() == 0.4
    assert model._expected_group.isChecked() is True
    assert model._expected_status_combo.currentText() == 'success'
    assert model._expected_tier_combo.currentText() == 'high'


def test_load_full_scene_syncs_confounder_star_tab(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The confounder star's tab reflects its planted truth keys after load."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    # S2 is the non-navigable confounder; find its tab by name.
    s2_index = next(i for i, s in enumerate(model.sim_params['stars']) if s['name'] == 'S2')
    tab_idx = model._find_tab_by_properties('star', s2_index)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    assert tab.navigable_check.isChecked() is False
    assert tab.catalog_error_check.isChecked() is True
    assert tab.companion_group.isChecked() is True
    assert tab.delta_mag_check.isChecked() is True


def test_load_full_scene_syncs_ring_advanced_groups(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The first ring tab reflects the orbit-perturbation and truth blocks after load."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    assert len(tab.mode_rows) == 1
    assert tab.mode_rows[0].m_spin.value() == 2
    assert tab.edge_wave_group.isChecked() is True
    assert tab.edge_wave_damp_spin.value() == 0.5
    assert tab.orbit_error_group.isChecked() is True
    assert tab.orbit_error_a_spin.value() == 1.0
    assert tab.orbit_sigma_group.isChecked() is True
    assert tab.azimuthal_modulation_group.isChecked() is True
    assert tab.azimuthal_shadow_group.isChecked() is True
    assert tab.azimuthal_spokes_group.isChecked() is True
    assert tab.moonlets_group.isChecked() is True
    assert len(tab.moonlet_rows) == 1
    assert tab.moonlet_rows[0].propeller_group.isChecked() is True


def _key_paths(value: Any, prefix: str = '') -> set[str]:
    """Collect every nested mapping key of ``value`` as a dotted path.

    List entries contribute their element keys without an index component, so
    two features' key sets union into one path namespace.
    """
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, sub in value.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            paths.add(path)
            paths |= _key_paths(sub, path)
    elif isinstance(value, list):
        for entry in value:
            paths |= _key_paths(entry, prefix)
    return paths


def test_gui_authored_ring_system_reaches_every_schema_key(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Every ring_system schema key is authorable through the editor's widgets.

    Authors a three-feature ring system purely through widget handlers (a
    ringlet carrying every orbit perturbation and error block, a wave train,
    and an in-side ramp, plus every system-level truth block), saves it, and
    asserts the loaded scene's dotted key paths cover the schema's full
    ring_system inventory.
    """
    # The block / geometry / feature inventories come from the schema's
    # boundary classification (scene_checks_ring re-uses the same sets), so
    # a key added to the schema alone reaches this coverage assertion.
    from spindoctor.sim.scene_checks_ring import (
        _RING_AZIMUTHAL_KEYS,
        _RING_EDGE_WAVE_KEYS,
        _RING_FEATURE_ORBIT_KEYS,
        _RING_MODULATION_KEYS,
        _RING_MOONLET_KEYS,
        _RING_ORBIT_ERROR_KEYS,
        _RING_ORBIT_MODE_KEYS,
        _RING_ORBIT_SIGMA_KEYS,
        _RING_PROPELLER_KEYS,
        _RING_SHADOW_KEYS,
        _RING_SPOKES_KEYS,
    )
    from spindoctor.sim.scene_schema import (
        _RING_FEATURE_KEYS,
        _RING_SYSTEM_GEOMETRY_KEYS,
        _RING_SYSTEM_KEYS,
    )

    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_ring_tab()
    tab = model._tabs.widget(model._find_tab_by_properties('ring', 0))
    # Feature 0 (ringlet): every per-feature block.
    tab.edge_wave_group.setChecked(True)
    tab.orbit_error_group.setChecked(True)
    tab.orbit_sigma_group.setChecked(True)
    model._on_ring_add_mode(0)
    # Photometric truth scalars and the system scalars write on change, so
    # drive their handlers off the widget defaults.
    model._on_ring_field(0, 'albedo', 0.6)
    model._on_ring_field(0, 'phase_g', -0.2)
    model._on_ring_system_field('phase_deg', 40.0)
    model._on_ring_system_field('km_per_pixel', 100.0)
    # System-level: range, truth clutter, and a moonlet with a propeller.
    tab.range_km_check.click()
    tab.azimuthal_modulation_group.setChecked(True)
    tab.azimuthal_shadow_group.setChecked(True)
    tab.azimuthal_spokes_group.setChecked(True)
    tab.moonlets_group.setChecked(True)
    model._on_ring_add_moonlet()
    tab.moonlet_rows[0].propeller_group.setChecked(True)
    # Features 1 and 2 cover the remaining kind-specific shape keys.
    model._add_ring_tab()
    model._on_ring_kind(1, 'wave')
    model._add_ring_tab()
    model._on_ring_kind(2, 'ramp')
    ramp_tab = model._tabs.widget(model._find_tab_by_properties('ring', 2))
    ramp_tab.side_combo.setCurrentText('in')

    out = tmp_path / 'ring_full.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)

    authored = _key_paths(loaded['ring_system'])
    expected: set[str] = set(_RING_SYSTEM_KEYS)
    expected |= {f'geometry.{key}' for key in _RING_SYSTEM_GEOMETRY_KEYS}
    expected |= {f'features.{key}' for key in _RING_FEATURE_KEYS}
    expected |= {f'features.orbit.{key}' for key in _RING_FEATURE_ORBIT_KEYS}
    expected |= {f'features.orbit.modes.{key}' for key in _RING_ORBIT_MODE_KEYS}
    expected |= {f'features.orbit.edge_wave.{key}' for key in _RING_EDGE_WAVE_KEYS}
    expected |= {f'features.orbit_error.{key}' for key in _RING_ORBIT_ERROR_KEYS}
    expected |= {f'features.declared_orbit_sigma.{key}' for key in _RING_ORBIT_SIGMA_KEYS}
    expected |= {f'azimuthal.{key}' for key in _RING_AZIMUTHAL_KEYS}
    expected |= {f'azimuthal.modulation.{key}' for key in _RING_MODULATION_KEYS}
    expected |= {f'azimuthal.shadow.{key}' for key in _RING_SHADOW_KEYS}
    expected |= {f'azimuthal.spokes.{key}' for key in _RING_SPOKES_KEYS}
    expected |= {f'moonlets.{key}' for key in _RING_MOONLET_KEYS}
    expected |= {f'moonlets.propeller.{key}' for key in _RING_PROPELLER_KEYS}
    missing = expected - authored
    assert not missing, f'ring_system keys not reachable from the editor: {sorted(missing)}'
