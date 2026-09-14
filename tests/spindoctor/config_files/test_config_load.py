"""Smoke tests for the shipped YAML configuration files.

Asserts the renumbered ``config_NNN_*.yaml`` set loads cleanly under
``Config.read_config`` and that bootstrap angles are now in degrees.
"""

from __future__ import annotations

import pathlib

from ruamel.yaml import YAML

import spindoctor.config.config
from spindoctor.config import Config


def test_default_config_loads_cleanly() -> None:
    """The full bundled config set merges without raising."""
    config = Config()
    config.read_config()
    assert config.is_loaded


def test_bootstrap_angles_are_degrees() -> None:
    """``config_070_bootstrap.yaml`` exposes degree-valued angle fields."""
    config = Config()
    config.read_config()
    bootstrap = config.bootstrap
    # Degree-keyed names plus magnitudes consistent with Cardinal Principle
    # #4 — values are in degrees, not radians.
    assert bootstrap.max_phase_angle_deg == 135.0
    assert bootstrap.max_incidence_angle_deg == 70.0
    assert bootstrap.max_emission_angle_deg == 70.0
    assert bootstrap.lon_resolution_deg == 0.5
    assert bootstrap.lat_resolution_deg == 0.5
    assert bootstrap.max_subsolar_dist_deg == 45.0


def _enables_rotation(node: object) -> bool:
    """Whether ``fit_camera_rotation`` is true anywhere in a loaded config tree.

    Parameters:
        node: A node of the parsed YAML, at any depth.

    Returns:
        True if any ``fit_camera_rotation`` key below this node is true.
    """
    if isinstance(node, dict):
        if node.get('fit_camera_rotation') is True:
            return True
        return any(_enables_rotation(value) for value in node.values())
    if isinstance(node, list):
        return any(_enables_rotation(value) for value in node)
    return False


def test_no_shipped_instrument_fits_camera_rotation() -> None:
    """No shipped instrument enables rotation fitting, and enabling one needs work.

    Nothing can currently use a fitted rotation.  Each technique measures it
    about its own center, so the translations reported alongside it are not in
    one convention, and a rotation without a recorded center cannot be carried
    into an attitude -- which suppresses the corrected C-matrix and omits the
    frame from the corrected kernels.  Enabling it therefore needs the rotation
    convention settled first, and needs the conflicted result path taught to
    carry a rotation, which today it silently drops.

    The instrument files are discovered rather than listed, so an instrument
    added later is covered without this test being edited.  It fails when a
    shipped configuration enables the flag, so that arrives as a decision.
    """
    config_dir = pathlib.Path(spindoctor.config.config.__file__).resolve().parent.parent
    inst_files = sorted((config_dir / 'config_files').glob('config_4[0-9]0_inst_*.yaml'))
    assert inst_files, 'no per-instrument configuration files were discovered'
    loader = YAML(typ='safe')
    enabling = [
        path.name for path in inst_files if _enables_rotation(loader.load(path.read_text()))
    ]
    assert enabling == []


def test_per_instrument_required_fields_present() -> None:
    """Every shipped 4N0 instrument block has the required Phase 3 keys."""
    config = Config()
    config.read_config()
    cassini = config.category('cassini_iss')
    for camera in (cassini['nac'], cassini['wac']):
        assert camera['data_units'] == 'raw_dn'
        assert camera['noise']['saturation_dn'] == 4095
        assert camera['fit_camera_rotation'] is False
        assert camera['max_rotation_deg'] == 5.0
        assert camera['mag_offset']['fallback_combo'] == 'CL1+CL2'
    voyager = config.category('voyager_iss')
    assert voyager['data_units'] == 'calibrated_if'
    assert voyager['noise']['saturation_dn'] == 255
    assert voyager['fit_camera_rotation'] is False
    assert voyager['max_rotation_deg'] == 5.0
    assert voyager['mag_offset']['fallback_combo'] == 'CL'
    galileo = config.category('galileo_ssi')
    assert galileo['data_units'] == 'raw_dn'
    assert galileo['noise']['saturation_dn'] == 255
    assert galileo['fit_camera_rotation'] is False
    assert galileo['max_rotation_deg'] == 5.0
    assert galileo['mag_offset']['fallback_combo'] == 'CL'
    nhlorri = config.category('newhorizons_lorri')
    assert nhlorri['data_units'] == 'raw_dn'
    assert nhlorri['noise']['saturation_dn'] == 4095
    assert nhlorri['fit_camera_rotation'] is False
    assert nhlorri['max_rotation_deg'] == 5.0
    assert nhlorri['mag_offset']['fallback_combo'] == '1'


def test_body_shape_section_loaded() -> None:
    """Body-shape entries are exposed via ``config.body_shape``."""
    config = Config()
    config.read_config()
    body_shape = config.body_shape
    assert 'MIMAS' in body_shape
    mimas = body_shape['MIMAS']
    # _sources keys are stripped at load time.
    assert '_sources' not in mimas
    assert 'shape_class_hint' in mimas
