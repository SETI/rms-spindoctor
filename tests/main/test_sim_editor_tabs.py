"""Per-tab widget-state tests for the ``sd_create_simulated_image`` editor.

These exercise the editor's data model directly through its widgets, with no
file I/O: every optics / detector / artifacts / star control follows the
absent-key discipline (an enable checkbox inserts its block; unchecked leaves
the key absent), edits write only their own key, and GUI-added objects carry
canonical value types.  The save / load / round-trip acceptance tests live in
``test_sim_editor_round_trip``.

Qt runs headless (offscreen platform).
"""

import os
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
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


def test_gui_added_star_psf_size_is_list(model: Any) -> None:
    """A GUI-added star stores psf_size as a list, not a tuple."""
    model._add_star_tab()
    psf_size = model.sim_params['stars'][0]['psf_size']
    assert isinstance(psf_size, list)
    assert psf_size == [11, 11]


def test_star_psf_size_edit_stays_a_list(model: Any) -> None:
    """Editing a star's PSF-window size keeps psf_size a list."""
    model._add_star_tab()
    model._on_star_psf_size_v_spin(0, 7)
    psf_size = model.sim_params['stars'][0]['psf_size']
    assert isinstance(psf_size, list)
    assert psf_size[0] == 7


def test_default_scene_has_no_optics_block(model: Any) -> None:
    """A fresh scene carries no optics block (the stage-activation floor)."""
    assert 'optics' not in model.sim_params


def test_psf_enable_inserts_block(model: Any) -> None:
    """Enabling the PSF group inserts an explicit kernel block."""
    model._psf_optics_group.setChecked(True)
    assert 'psf' in model.sim_params['optics']


def test_psf_disable_removes_optics_key(model: Any) -> None:
    """Disabling the only optics sub-block drops the optics key entirely."""
    model._psf_optics_group.setChecked(True)
    model._psf_optics_group.setChecked(False)
    assert 'optics' not in model.sim_params


def test_match_navigator_writes_canonical_form(model: Any) -> None:
    """The match-navigator checkbox writes the exclusive canonical PSF form."""
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}


def test_distortion_center_keys_absent_unless_enabled(model: Any) -> None:
    """The distortion block omits the optical-centre keys until enabled."""
    model._distortion_group.setChecked(True)
    block = model.sim_params['optics']['distortion']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_distortion_center_zero_is_authorable(model: Any) -> None:
    """An explicit 0.0 optical centre survives (no 0.0-to-absent flip)."""
    model._distortion_group.setChecked(True)
    model._distortion_center_check.setChecked(True)
    model._distortion_center_v_spin.setValue(0.0)
    model._distortion_center_u_spin.setValue(0.0)
    block = model.sim_params['optics']['distortion']
    assert block['center_v'] == 0.0
    assert block['center_u'] == 0.0


def test_distortion_center_uncheck_drops_both_keys(model: Any) -> None:
    """Unchecking the optical-centre enable removes both centre keys."""
    model._distortion_group.setChecked(True)
    model._distortion_center_check.setChecked(True)
    model._distortion_center_v_spin.setValue(40.0)
    model._distortion_center_u_spin.setValue(50.0)
    model._distortion_center_check.setChecked(False)
    block = model.sim_params['optics']['distortion']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_ring_range_km_unchecked_leaves_key_absent(model: Any) -> None:
    """Disabling the ring-system physical-range control removes the key."""
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    ring_tab = model._tabs.widget(tab_idx)
    ring_tab.range_km_check.click()
    assert 'range_km' in model.sim_params['ring_system']
    ring_tab.range_km_check.click()
    assert 'range_km' not in model.sim_params['ring_system']


def test_ring_kind_switch_rewrites_shape_keys(model: Any) -> None:
    """Switching a feature's kind swaps in exactly the kind's shape keys."""
    model._add_ring_tab()
    feature = model.sim_params['ring_system']['features'][0]
    assert 'width' in feature
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    model._on_ring_kind(0, 'wave')
    assert 'width' not in feature
    assert feature['wavelength'] > 0.0
    assert feature['damping'] > 0.0
    model._on_ring_kind(0, 'ringlet')
    assert 'wavelength' not in feature
    assert 'damping' not in feature
    assert feature['width'] > 0.0


def test_deleting_the_last_ring_feature_retires_the_block(model: Any) -> None:
    """Removing the only feature drops the whole ring_system key."""
    model._add_ring_tab()
    assert 'ring_system' in model.sim_params
    model._delete_tab_by_index('ring', 0)
    assert 'ring_system' not in model.sim_params


def test_deleting_the_last_ring_feature_blocked_while_system_blocks_exist(model: Any) -> None:
    """The last feature tab refuses deletion while moonlets are authored.

    The system-level azimuthal / moonlets widgets live on the first
    feature's tab, so deleting the last feature would leave those blocks
    with no widget to reach them; the deletion is refused (with a status-bar
    explanation) until the blocks are disabled.
    """
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    tab.moonlets_group.setChecked(True)
    model._on_ring_add_moonlet()
    model._delete_tab_by_index('ring', 0)
    assert len(model.sim_params['ring_system']['features']) == 1
    assert len(model.sim_params['ring_system']['moonlets']) == 1
    assert 'ring feature' in model.statusBar().currentMessage()


def test_deleting_the_last_ring_feature_allowed_after_blocks_removed(model: Any) -> None:
    """Disabling the system blocks unblocks the last feature's deletion."""
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    tab.azimuthal_modulation_group.setChecked(True)
    model._delete_tab_by_index('ring', 0)
    assert 'ring_system' in model.sim_params
    tab.azimuthal_modulation_group.setChecked(False)
    model._delete_tab_by_index('ring', 0)
    assert 'ring_system' not in model.sim_params


def test_ring_opening_angle_spins_match_the_validator_range(model: Any) -> None:
    """The opening-angle spins mirror the validator's (-90, 90] range."""
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    for key in ('opening_deg_obs', 'opening_deg_sun'):
        spin = getattr(tab, f'{key}_spin')
        assert spin.minimum() == -89.9
        assert spin.maximum() == 90.0


def test_match_navigator_disables_kernel_spins(model: Any) -> None:
    """Matching the navigator disables the explicit-kernel spins."""
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)
    assert model._psf_sigma_v_spin.isEnabled() is False


def test_smear_row_edit_updates_params(model: Any) -> None:
    """Editing a smear row's drift updates the smear list in sim_params."""
    model._smear_group.setChecked(True)
    model._on_add_smear_clicked()
    model._smear_rows[0].dv_spin.setValue(3.0)
    assert model.sim_params['optics']['smear'][0]['dv_px'] == 3.0


def test_ghost_enable_inserts_list(model: Any) -> None:
    """Enabling the ghost group with a row inserts a ghost list."""
    model._ghosts_group.setChecked(True)
    model._on_add_ghost_clicked()
    assert len(model.sim_params['optics']['ghosts']) == 1


def test_distortion_disable_removes_block(model: Any) -> None:
    """Disabling the distortion group removes its block."""
    model._distortion_group.setChecked(True)
    assert 'distortion' in model.sim_params['optics']
    model._distortion_group.setChecked(False)
    assert 'optics' not in model.sim_params


def test_oversample_checkbox_toggles_key(model: Any) -> None:
    """The oversample checkbox inserts and removes the top-level key."""
    model._oversample_check.setChecked(True)
    model._oversample_spin.setValue(4)
    assert model.sim_params['oversample'] == 4
    model._oversample_check.setChecked(False)
    assert 'oversample' not in model.sim_params


def test_spk_error_toggle_inserts_and_removes(model: Any) -> None:
    """The spk_error group inserts and removes the block with its three keys."""
    model._spk_error_group.setChecked(True)
    assert set(model.sim_params['spk_error']) == {'dv_px', 'du_px', 'reference_range_km'}
    model._spk_error_group.setChecked(False)
    assert 'spk_error' not in model.sim_params


def test_instrument_defaults_toggles_artifacts_key(model: Any) -> None:
    """The instrument-defaults checkbox inserts and removes the artifacts key."""
    model._instrument_defaults_check.setChecked(True)
    assert model.sim_params['artifacts'] == {'instrument_defaults': True}
    model._instrument_defaults_check.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_detector_group_toggles_key(model: Any) -> None:
    """The detector group inserts an empty block and removes it when disabled.

    The block starts empty (per-key discipline): unedited keys stay absent so
    the instrument's catalog defaults keep applying.
    """
    model._detector_group.setChecked(True)
    assert model.sim_params['detector'] == {}
    model._detector_group.setChecked(False)
    assert 'detector' not in model.sim_params


def test_detector_edit_writes_only_the_edited_key(model: Any) -> None:
    """A single spin edit writes its own key and nothing else."""
    model._detector_group.setChecked(True)
    model._detector_gain_state_spin.setValue(3)
    assert model.sim_params['detector'] == {'gain_state': 3}


def test_detector_quantization_is_authorable(model: Any) -> None:
    """The quantization combo writes the detector.quantization key."""
    model._detector_group.setChecked(True)
    model._detector_quantization_combo.setCurrentText('sqrt_lut')
    assert model.sim_params['detector'] == {'quantization': 'sqrt_lut'}


# ---- Registry-driven artifact-mode rows ----


def test_every_registry_mode_has_a_row(model: Any) -> None:
    """The tab generates one row per registered artifact mode."""
    from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES

    assert set(model._mode_rows) == set(ARTIFACT_MODES)


def test_adversarial_check_toggles_key(model: Any) -> None:
    """The adversarial checkbox inserts and removes the artifacts.adversarial key."""
    model._adversarial_check.setChecked(True)
    assert model.sim_params['artifacts'] == {'adversarial': True}
    model._adversarial_check.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_mode_enable_inserts_empty_map(model: Any) -> None:
    """Enabling a mode row inserts an empty map (absent-key discipline)."""
    model._mode_rows['missing_lines'].group.setChecked(True)
    assert model.sim_params['artifacts'] == {'missing_lines': {}}


def test_mode_param_edit_writes_only_edited_key(model: Any) -> None:
    """Editing one mode parameter writes only that key into the mode map."""
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(3.0)
    assert model.sim_params['artifacts']['missing_lines'] == {'incidence': 3.0}


def test_mode_disable_removes_key_and_prunes_block(model: Any) -> None:
    """Disabling the only enabled mode removes the artifacts block entirely."""
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(3.0)
    row.group.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_mode_enum_param_writes_native_choice(model: Any) -> None:
    """An enum row writes the choice in its native type (an int period)."""
    row = model._mode_rows['alternating_lines']
    row.group.setChecked(True)
    row._scalar_widgets['period'].setCurrentText('4')
    assert model.sim_params['artifacts']['alternating_lines']['period'] == 4


def test_mode_int_list_param_absent_until_set(model: Any) -> None:
    """A rect/window list key stays absent until its enable box is checked."""
    row = model._mode_rows['cutout_window']
    row.group.setChecked(True)
    assert 'rect' not in model.sim_params['artifacts']['cutout_window']
    row._list_checks['rect'].setChecked(True)
    for index, spin in enumerate(row._list_spins['rect']):
        spin.setValue(10 + index)
    assert model.sim_params['artifacts']['cutout_window']['rect'] == [10, 11, 12, 13]


def test_switches_and_modes_coexist_in_block(model: Any) -> None:
    """Instrument-defaults, adversarial, and a mode share one artifacts block."""
    model._instrument_defaults_check.setChecked(True)
    model._adversarial_check.setChecked(True)
    model._mode_rows['missing_lines'].group.setChecked(True)
    assert model.sim_params['artifacts'] == {
        'instrument_defaults': True,
        'adversarial': True,
        'missing_lines': {},
    }


def test_mode_availability_disables_unavailable_row(model: Any) -> None:
    """A mode unavailable on the instrument is disabled with the registry reason."""
    model.sim_params['instrument'] = 'nhlorri'
    model._refresh_artifact_mode_availability()
    row = model._mode_rows['hot_pixels']
    assert row.group.isEnabled() is False
    assert 'LORRI' in row.group.toolTip()


def test_mode_availability_enables_available_row(model: Any) -> None:
    """An available mode is enabled and carries its incidence semantics tooltip."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._refresh_artifact_mode_availability()
    row = model._mode_rows['hot_pixels']
    assert row.group.isEnabled() is True
    assert 'incidence' in row.group.toolTip()


# ---- Per-star information-asymmetry controls ----


def _star_tab(model: Any) -> Any:
    """Return the widget of the first star's tab."""
    tab_idx = model._find_tab_by_properties('star', 0)
    assert tab_idx is not None
    return model._tabs.widget(tab_idx)


def test_star_default_omits_navigable_key(model: Any) -> None:
    """A GUI-added star leaves the navigable key absent (absent means navigable)."""
    model._add_star_tab()
    assert 'navigable' not in model.sim_params['stars'][0]


def test_star_navigable_uncheck_writes_false(model: Any) -> None:
    """Unchecking navigable writes the present ``navigable: false`` key."""
    model._add_star_tab()
    _star_tab(model).navigable_check.click()
    assert model.sim_params['stars'][0]['navigable'] is False


def test_star_navigable_recheck_removes_key(model: Any) -> None:
    """Re-checking navigable drops the key back to absent."""
    model._add_star_tab()
    check = _star_tab(model).navigable_check
    check.click()
    check.click()
    assert 'navigable' not in model.sim_params['stars'][0]


def test_star_catalog_error_toggle_writes_both_keys(model: Any) -> None:
    """Enabling catalog error inserts both v and u keys; disabling removes them."""
    model._add_star_tab()
    check = _star_tab(model).catalog_error_check
    check.click()
    star = model.sim_params['stars'][0]
    assert 'catalog_error_v' in star
    assert 'catalog_error_u' in star
    check.click()
    assert 'catalog_error_v' not in star
    assert 'catalog_error_u' not in star


def test_star_catalog_error_spin_writes_value(model: Any) -> None:
    """A catalog-error spin edit writes its own key once enabled."""
    model._add_star_tab()
    tab = _star_tab(model)
    tab.catalog_error_check.click()
    tab.catalog_error_v_spin.setValue(1.5)
    assert model.sim_params['stars'][0]['catalog_error_v'] == 1.5


def test_star_delta_mag_toggle_inserts_and_removes(model: Any) -> None:
    """The variable-star enable inserts and removes the delta_mag key."""
    model._add_star_tab()
    check = _star_tab(model).delta_mag_check
    check.click()
    assert 'delta_mag' in model.sim_params['stars'][0]
    check.click()
    assert 'delta_mag' not in model.sim_params['stars'][0]


def test_star_companion_toggle_inserts_and_removes(model: Any) -> None:
    """The companion group inserts the map with its three keys and removes it."""
    model._add_star_tab()
    group = _star_tab(model).companion_group
    group.setChecked(True)
    assert set(model.sim_params['stars'][0]['companion']) == {'sep_px', 'delta_mag', 'angle_deg'}
    group.setChecked(False)
    assert 'companion' not in model.sim_params['stars'][0]


def test_star_companion_spin_writes_sub_key(model: Any) -> None:
    """A companion spin edit updates its own sub-key in the companion map."""
    model._add_star_tab()
    tab = _star_tab(model)
    tab.companion_group.setChecked(True)
    tab.companion_sep_spin.setValue(4.0)
    assert model.sim_params['stars'][0]['companion']['sep_px'] == 4.0


def test_sky_counts_absent_by_default(model: Any) -> None:
    """A fresh scene carries no sky_counts block (absent means no sky)."""
    assert 'sky_counts' not in model.sim_params
    assert model._sky_counts_check.isChecked() is False


def test_sky_counts_toggle_inserts_and_removes(model: Any) -> None:
    """The background-sky checkbox inserts and removes the whole block."""
    model._sky_counts_check.click()
    assert set(model.sim_params['sky_counts']) == {
        'a',
        'b',
        'density_factor',
        'diffuse_e_per_px',
    }
    model._sky_counts_check.click()
    assert 'sky_counts' not in model.sim_params


def test_sky_density_edit_writes_when_enabled(model: Any) -> None:
    """A density edit updates the block once the group is enabled."""
    model._sky_counts_check.click()
    model._sky_density_spin.setValue(12.0)
    assert model.sim_params['sky_counts']['density_factor'] == 12.0


def test_sky_value_edit_never_creates_the_block(model: Any) -> None:
    """A value-widget change with the group disabled inserts no key."""
    model._sky_density_spin.setValue(12.0)
    model._sky_a_spin.setValue(-2.0)
    assert 'sky_counts' not in model.sim_params


def test_star_scatter_toggle_inserts_and_removes(model: Any) -> None:
    """The star-catalog-scatter checkbox inserts and removes the top-level key."""
    model._star_scatter_check.click()
    model._star_scatter_spin.setValue(0.5)
    assert model.sim_params['star_catalog_scatter_px'] == 0.5
    model._star_scatter_check.click()
    assert 'star_catalog_scatter_px' not in model.sim_params


def test_expected_toggle_inserts_and_removes(model: Any) -> None:
    """The expected group inserts the block with a status and removes it."""
    model._expected_group.setChecked(True)
    assert model.sim_params['expected']['status'] == 'success'
    model._expected_group.setChecked(False)
    assert 'expected' not in model.sim_params


def test_expected_tier_none_writes_null(model: Any) -> None:
    """The confidence-tier (none) choice stores a null tier (assert status only)."""
    model._expected_group.setChecked(True)
    model._expected_tier_combo.setCurrentText('(none)')
    assert model.sim_params['expected']['confidence_tier'] is None


def test_expected_reason_line_edit_sets_and_clears(model: Any) -> None:
    """The status-reason edit writes a non-empty token and drops an empty one."""
    model._expected_group.setChecked(True)
    model._expected_reason_edit.setText('no_signal_in_image')
    assert model.sim_params['expected']['status_reason'] == 'no_signal_in_image'
    model._expected_reason_edit.setText('')
    assert 'status_reason' not in model.sim_params['expected']


# ---- Per-body appearance controls (relief, photometry, texture, mesh extras) ----


def _body_tab(model: Any) -> Any:
    """Add a body and return the widget of its tab."""
    model._add_body_tab()
    tab_idx = model._find_tab_by_properties('body', 0)
    assert tab_idx is not None
    return model._tabs.widget(tab_idx)


def _body(model: Any) -> Any:
    """Return the first body dict."""
    return model.sim_params['bodies'][0]


def test_body_relief_toggle_inserts_and_removes(model: Any) -> None:
    """The limb-relief group inserts both keys and removes them when disabled."""
    tab = _body_tab(model)
    tab.relief_group.setChecked(True)
    assert 'limb_relief_rms' in _body(model)
    assert 'limb_relief_corr_deg' in _body(model)
    tab.relief_group.setChecked(False)
    assert 'limb_relief_rms' not in _body(model)


def test_body_relief_value_edit_writes_key(model: Any) -> None:
    """A relief-RMS spin edit writes its own key once enabled."""
    tab = _body_tab(model)
    tab.relief_group.setChecked(True)
    tab.relief_rms_spin.setValue(0.03)
    assert _body(model)['limb_relief_rms'] == 0.03


def test_body_photometry_toggle_inserts_and_removes(model: Any) -> None:
    """The photometric-law group inserts and removes photometric_law."""
    tab = _body_tab(model)
    tab.photometry_group.setChecked(True)
    assert 'photometric_law' in _body(model)
    tab.photometry_group.setChecked(False)
    assert 'photometric_law' not in _body(model)


def test_body_minnaert_law_gates_k_key(model: Any) -> None:
    """Choosing minnaert inserts minnaert_k; a non-minnaert law drops it."""
    tab = _body_tab(model)
    tab.photometry_group.setChecked(True)
    tab.photometry_law_combo.setCurrentText('minnaert')
    assert 'minnaert_k' in _body(model)
    tab.photometry_law_combo.setCurrentText('lambert')
    assert 'minnaert_k' not in _body(model)


def test_body_minnaert_k_spin_disabled_off_minnaert(model: Any) -> None:
    """The Minnaert exponent spin is enabled only under the minnaert law."""
    tab = _body_tab(model)
    tab.photometry_group.setChecked(True)
    tab.photometry_law_combo.setCurrentText('lommel_seeliger')
    assert tab.minnaert_k_spin.isEnabled() is False


def test_body_surge_toggle_inserts_and_removes(model: Any) -> None:
    """The opposition-surge group inserts the map with both keys and removes it."""
    tab = _body_tab(model)
    tab.surge_group.setChecked(True)
    assert set(_body(model)['opposition_surge']) == {'amplitude', 'width_deg'}
    tab.surge_group.setChecked(False)
    assert 'opposition_surge' not in _body(model)


def test_body_atmosphere_toggle_inserts_and_removes(model: Any) -> None:
    """The atmosphere group inserts the four core keys and removes the map."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    assert set(_body(model)['atmosphere']) == {'scale_height_px', 'tau_ref', 'ref_altitude_px', 'g'}
    tab.atmosphere_group.setChecked(False)
    assert 'atmosphere' not in _body(model)


def test_body_atmosphere_value_edit_writes_key(model: Any) -> None:
    """A tau-ref spin edit writes its own key once the group is enabled."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    tab.atmosphere_tau_ref_spin.setValue(2.5)
    assert _body(model)['atmosphere']['tau_ref'] == 2.5


def test_body_atmosphere_scale_height_spin_writes_key(model: Any) -> None:
    """A scale-height spin edit writes its own key once the group is enabled."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    tab.atmosphere_scale_height_spin.setValue(9.0)
    assert _body(model)['atmosphere']['scale_height_px'] == 9.0


def test_body_atmosphere_g_spin_writes_key(model: Any) -> None:
    """An asymmetry spin edit writes the g key once the group is enabled."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    tab.atmosphere_g_spin.setValue(0.55)
    assert _body(model)['atmosphere']['g'] == 0.55


def test_body_atmosphere_ref_altitude_spin_writes_key(model: Any) -> None:
    """A reference-altitude spin edit writes its own key once enabled."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    tab.atmosphere_ref_altitude_spin.setValue(3.0)
    assert _body(model)['atmosphere']['ref_altitude_px'] == 3.0


def test_body_atmosphere_shell_gates_detached_key(model: Any) -> None:
    """The shell checkbox inserts detached_px; unchecking it drops the key."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    tab.atmosphere_shell_check.setChecked(True)
    assert 'detached_px' in _body(model)['atmosphere']
    tab.atmosphere_shell_check.setChecked(False)
    assert 'detached_px' not in _body(model)['atmosphere']


def test_body_atmosphere_detached_spin_disabled_without_shell(model: Any) -> None:
    """The detached-altitude spin is enabled only when the shell is on."""
    tab = _body_tab(model)
    tab.atmosphere_group.setChecked(True)
    assert tab.atmosphere_detached_spin.isEnabled() is False


def test_body_albedo_toggle_inserts_and_removes(model: Any) -> None:
    """The albedo-texture group inserts the map with a spots list and removes it."""
    tab = _body_tab(model)
    tab.albedo_group.setChecked(True)
    assert set(_body(model)['albedo_texture']) == {'rms', 'corr_px', 'spots'}
    assert _body(model)['albedo_texture']['spots'] == []
    tab.albedo_group.setChecked(False)
    assert 'albedo_texture' not in _body(model)


def test_body_albedo_add_and_remove_spot(model: Any) -> None:
    """Adding a spot appends a schema entry; removing it empties the list."""
    tab = _body_tab(model)
    tab.albedo_group.setChecked(True)
    model._on_body_add_spot(0)
    spots = _body(model)['albedo_texture']['spots']
    assert set(spots[0]) == {'lat_deg', 'lon_deg', 'radius_deg', 'albedo_factor'}
    model._remove_spot_row(0, tab.spot_rows[0])
    assert _body(model)['albedo_texture']['spots'] == []


def test_body_disc_toggle_inserts_and_removes(model: Any) -> None:
    """The disc-texture group inserts the band map with a storms list and removes it."""
    tab = _body_tab(model)
    tab.disc_group.setChecked(True)
    assert set(_body(model)['disc_texture']) == {
        'band_amplitude',
        'band_wavenumber',
        'band_phase_deg',
        'storms',
    }
    tab.disc_group.setChecked(False)
    assert 'disc_texture' not in _body(model)


def test_body_disc_add_storm_appends_entry(model: Any) -> None:
    """Adding a storm appends a schema entry to the storms list."""
    tab = _body_tab(model)
    tab.disc_group.setChecked(True)
    model._on_body_add_storm(0)
    storms = _body(model)['disc_texture']['storms']
    assert set(storms[0]) == {'lat_deg', 'lon_deg', 'radius_deg', 'albedo_factor'}


def test_body_transits_toggle_and_add(model: Any) -> None:
    """Enabling transits inserts an empty list; adding one seeds a moon disc."""
    tab = _body_tab(model)
    tab.transits_group.setChecked(True)
    assert _body(model)['transits'] == []
    model._on_body_add_transit(0)
    assert 'moon' in _body(model)['transits'][0]


def test_body_transit_shadow_sub_group_writes_shadow(model: Any) -> None:
    """Checking a transit's shadow sub-group adds a shadow map to the entry."""
    tab = _body_tab(model)
    tab.transits_group.setChecked(True)
    model._on_body_add_transit(0)
    tab.transit_rows[0].shadow_group.setChecked(True)
    assert 'shadow' in _body(model)['transits'][0]
    assert set(_body(model)['transits'][0]['shadow']) == {
        'dv_px',
        'du_px',
        'radius_px',
        'darkness',
    }


def test_body_shading_combo_absent_for_flat(model: Any) -> None:
    """Gouraud writes the shading key; flat drops it (absent means flat)."""
    tab = _body_tab(model)
    tab.shading_combo.setCurrentText('gouraud')
    assert _body(model)['shading'] == 'gouraud'
    tab.shading_combo.setCurrentText('flat')
    assert 'shading' not in _body(model)


def test_body_mesh_octaves_zero_leaves_key_absent(model: Any) -> None:
    """A positive octave count writes the key; zero drops it (absent means 0)."""
    tab = _body_tab(model)
    tab.mesh_octaves_spin.setValue(3)
    assert _body(model)['mesh_detail_octaves'] == 3
    tab.mesh_octaves_spin.setValue(0)
    assert 'mesh_detail_octaves' not in _body(model)


def test_body_pose_scatter_toggle_inserts_and_removes(model: Any) -> None:
    """The pose-scatter group inserts the sigma map and removes it."""
    tab = _body_tab(model)
    tab.pose_scatter_group.setChecked(True)
    assert set(_body(model)['pose_scatter']) == {'sigma_deg'}
    tab.pose_scatter_group.setChecked(False)
    assert 'pose_scatter' not in _body(model)


def test_body_mesh_extras_gated_by_shape(model: Any) -> None:
    """The mesh-extras group is disabled for an ellipsoid, enabled for a mesh."""
    tab = _body_tab(model)
    assert tab.mesh_extras_group.isEnabled() is False
    model._on_body_shape_model(0, 'polyhedral_mesh')
    assert tab.mesh_extras_group.isEnabled() is True


# ---- Ring advanced groups (orbit perturbations, planted errors, truth clutter) ----


def _ring_tab(model: Any) -> Any:
    """Add a ring feature and return its built tab widget."""
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    return model._tabs.widget(tab_idx)


def _ring(model: Any) -> Any:
    """Return the first ring_system feature dict."""
    return model.sim_params['ring_system']['features'][0]


def test_ring_mode_add_and_remove_row(model: Any) -> None:
    """Adding an m-mode row writes the modes list; removing the last drops the key."""
    tab = _ring_tab(model)
    model._on_ring_add_mode(0)
    modes = _ring(model)['orbit']['modes']
    assert set(modes[0]) == {'m', 'amp', 'peri'}
    model._remove_ring_mode_row(0, tab.mode_rows[0])
    assert 'modes' not in _ring(model)['orbit']


def test_ring_mode_row_edit_writes_value(model: Any) -> None:
    """Editing an m-mode row's m spin rewrites the modes list."""
    tab = _ring_tab(model)
    model._on_ring_add_mode(0)
    tab.mode_rows[0].m_spin.setValue(3)
    assert _ring(model)['orbit']['modes'][0]['m'] == 3


def test_ring_edge_wave_toggle_inserts_and_removes(model: Any) -> None:
    """The edge-wave group inserts the full map and removes it when disabled."""
    tab = _ring_tab(model)
    tab.edge_wave_group.setChecked(True)
    assert set(_ring(model)['orbit']['edge_wave']) == {'amp', 'wavelength', 'damp', 'lam0'}
    tab.edge_wave_group.setChecked(False)
    assert 'edge_wave' not in _ring(model)['orbit']


def test_ring_edge_wave_spin_writes_sub_key(model: Any) -> None:
    """An edge-wave amplitude edit writes its own key once enabled."""
    tab = _ring_tab(model)
    tab.edge_wave_group.setChecked(True)
    tab.edge_wave_amp_spin.setValue(2.5)
    assert _ring(model)['orbit']['edge_wave']['amp'] == 2.5


def test_ring_orbit_error_toggle_inserts_and_removes(model: Any) -> None:
    """The planted orbit-error group inserts all three keys and removes the map."""
    tab = _ring_tab(model)
    tab.orbit_error_group.setChecked(True)
    assert set(_ring(model)['orbit_error']) == {
        'delta_a_px',
        'delta_ae_px',
        'delta_long_peri_deg',
    }
    tab.orbit_error_group.setChecked(False)
    assert 'orbit_error' not in _ring(model)


def test_ring_orbit_error_spin_writes_own_key(model: Any) -> None:
    """An orbit-error delta_a edit writes its own key once enabled."""
    tab = _ring_tab(model)
    tab.orbit_error_group.setChecked(True)
    tab.orbit_error_a_spin.setValue(2.5)
    assert _ring(model)['orbit_error']['delta_a_px'] == 2.5


def test_ring_sigma_toggle_inserts_and_removes(model: Any) -> None:
    """The declared-sigma group inserts all three keys and removes the map."""
    tab = _ring_tab(model)
    tab.orbit_sigma_group.setChecked(True)
    assert set(_ring(model)['declared_orbit_sigma']) == {
        'sigma_a_px',
        'sigma_ae_px',
        'sigma_long_peri_deg',
    }
    tab.orbit_sigma_group.setChecked(False)
    assert 'declared_orbit_sigma' not in _ring(model)


def test_ring_sigma_spin_writes_own_key(model: Any) -> None:
    """A declared-sigma sigma_a edit writes its own key once enabled."""
    tab = _ring_tab(model)
    tab.orbit_sigma_group.setChecked(True)
    tab.orbit_sigma_a_spin.setValue(1.5)
    assert _ring(model)['declared_orbit_sigma']['sigma_a_px'] == 1.5


def test_ring_modulation_toggle_creates_and_prunes_azimuthal(model: Any) -> None:
    """The modulation sub-group creates the azimuthal block and prunes it empty."""
    tab = _ring_tab(model)
    tab.azimuthal_modulation_group.setChecked(True)
    modulation = model.sim_params['ring_system']['azimuthal']['modulation']
    assert set(modulation) == {'amplitude', 'm', 'phase_deg'}
    tab.azimuthal_modulation_group.setChecked(False)
    assert 'azimuthal' not in model.sim_params['ring_system']


def test_ring_shadow_toggle_inserts_sub_map(model: Any) -> None:
    """The planet-shadow sub-group writes its full sub-map."""
    tab = _ring_tab(model)
    tab.azimuthal_shadow_group.setChecked(True)
    shadow = model.sim_params['ring_system']['azimuthal']['shadow']
    assert set(shadow) == {'start_deg', 'extent_deg', 'darkness'}


def test_ring_spokes_toggle_inserts_sub_map(model: Any) -> None:
    """The spokes sub-group writes its full sub-map."""
    tab = _ring_tab(model)
    tab.azimuthal_spokes_group.setChecked(True)
    spokes = model.sim_params['ring_system']['azimuthal']['spokes']
    assert set(spokes) == {'count', 'r_inner', 'r_outer', 'width_deg', 'contrast'}


def test_ring_spokes_value_edit_writes_own_key(model: Any) -> None:
    """A spokes contrast edit writes only its own sub-key."""
    tab = _ring_tab(model)
    tab.azimuthal_spokes_group.setChecked(True)
    tab.spokes_contrast_spin.setValue(-0.4)
    assert model.sim_params['ring_system']['azimuthal']['spokes']['contrast'] == -0.4


def test_ring_disabling_one_azimuthal_sub_keeps_others(model: Any) -> None:
    """Removing one azimuthal sub-map leaves the block while another is enabled."""
    tab = _ring_tab(model)
    tab.azimuthal_modulation_group.setChecked(True)
    tab.azimuthal_spokes_group.setChecked(True)
    tab.azimuthal_modulation_group.setChecked(False)
    azimuthal = model.sim_params['ring_system']['azimuthal']
    assert 'modulation' not in azimuthal
    assert 'spokes' in azimuthal


def test_ring_moonlets_toggle_and_add(model: Any) -> None:
    """Enabling moonlets inserts an empty list; adding one seeds a disc entry."""
    tab = _ring_tab(model)
    tab.moonlets_group.setChecked(True)
    assert model.sim_params['ring_system']['moonlets'] == []
    model._on_ring_add_moonlet()
    moonlet = model.sim_params['ring_system']['moonlets'][0]
    assert set(moonlet) == {'a', 'lam_deg', 'radius_px', 'amplitude'}


def test_ring_moonlet_propeller_sub_group_writes_propeller(model: Any) -> None:
    """Checking a moonlet's propeller sub-group adds the propeller map."""
    tab = _ring_tab(model)
    tab.moonlets_group.setChecked(True)
    model._on_ring_add_moonlet()
    tab.moonlet_rows[0].propeller_group.setChecked(True)
    propeller = model.sim_params['ring_system']['moonlets'][0]['propeller']
    assert set(propeller) == {'length_deg', 'width_px', 'contrast'}


def test_ring_moonlets_toggle_off_removes_key(model: Any) -> None:
    """Disabling the moonlets group removes the key entirely."""
    tab = _ring_tab(model)
    tab.moonlets_group.setChecked(True)
    model._on_ring_add_moonlet()
    tab.moonlets_group.setChecked(False)
    assert 'moonlets' not in model.sim_params['ring_system']


def test_ring_moonlet_remove_row_empties_list(model: Any) -> None:
    """Removing the only moonlet row leaves an enabled empty list."""
    tab = _ring_tab(model)
    tab.moonlets_group.setChecked(True)
    model._on_ring_add_moonlet()
    model._remove_ring_moonlet_row(tab.moonlet_rows[0])
    assert model.sim_params['ring_system']['moonlets'] == []


def test_ring_advanced_groups_only_on_first_feature_tab(model: Any) -> None:
    """System-level truth groups appear on the first feature's tab only."""
    _ring_tab(model)
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 1)
    assert tab_idx is not None
    second = model._tabs.widget(tab_idx)
    assert hasattr(second, 'edge_wave_group') is True
    assert hasattr(second, 'moonlets_group') is False


# ---------------------------------------------------------------------------
# Scene positions: the editor's own coordinate system
# ---------------------------------------------------------------------------


def test_gui_added_body_defaults_to_the_frame_centre(model: Any) -> None:
    """A new body starts at the frame center in pixel corner coordinates."""
    model._add_body_tab()
    body = model.sim_params['bodies'][0]
    assert body['center_v'] == model.sim_params['size_v'] / 2.0


def test_gui_added_star_defaults_to_the_frame_centre(model: Any) -> None:
    """A new star starts at the frame center in pixel corner coordinates."""
    model._add_star_tab()
    star = model.sim_params['stars'][0]
    assert star['v'] == model.sim_params['size_v'] / 2.0


def test_gui_added_body_and_ring_system_are_concentric(model: Any) -> None:
    """A fresh scene's body and ring system share one center."""
    model._add_body_tab()
    model._add_ring_tab()
    body = model.sim_params['bodies'][0]
    geometry = model.sim_params['ring_system']['geometry']
    assert body['center_v'] == geometry['center_v']


def test_gui_added_star_and_ring_system_are_concentric(model: Any) -> None:
    """A fresh scene's star and ring system share one center."""
    model._add_star_tab()
    model._add_ring_tab()
    star = model.sim_params['stars'][0]
    geometry = model.sim_params['ring_system']['geometry']
    assert star['v'] == geometry['center_v']


def test_stray_center_keys_absent_unless_enabled(model: Any) -> None:
    """The stray-light block omits the bump-center keys until enabled."""
    model._stray_group.setChecked(True)
    block = model.sim_params['optics']['stray_light']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_stray_center_zero_is_authorable(model: Any) -> None:
    """An explicit 0.0 bump center survives (no 0.0-to-absent flip)."""
    model._stray_group.setChecked(True)
    model._stray_center_check.setChecked(True)
    model._stray_center_v_spin.setValue(0.0)
    model._stray_center_u_spin.setValue(0.0)
    block = model.sim_params['optics']['stray_light']
    assert block['center_v'] == 0.0
    assert block['center_u'] == 0.0


def test_stray_center_uncheck_drops_both_keys(model: Any) -> None:
    """Unchecking the bump-center enable removes both center keys."""
    model._stray_group.setChecked(True)
    model._stray_center_check.setChecked(True)
    model._stray_center_v_spin.setValue(40.0)
    model._stray_center_u_spin.setValue(50.0)
    model._stray_center_check.setChecked(False)
    block = model.sim_params['optics']['stray_light']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_stray_center_spin_shows_the_frame_centre_when_absent(model: Any) -> None:
    """With no center key the spin displays the center the renderer will use."""
    model._stray_group.setChecked(True)
    assert model._stray_center_v_spin.value() == model.sim_params['size_v'] / 2.0
