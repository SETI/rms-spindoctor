"""Smoke tests for the shipped YAML configuration files.

Asserts the ``config_NNN_*.yaml`` set loads cleanly under ``Config.read_config``, and
checks the per-instrument blocks and the body-shape section it carries.
"""

from __future__ import annotations

import pathlib

import pytest
from ruamel.yaml import YAML

import spindoctor.config.config
from spindoctor.config import Config


def test_default_config_loads_cleanly() -> None:
    """The full bundled config set merges without raising."""
    config = Config()
    config.read_config()
    assert config.is_loaded


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


@pytest.mark.parametrize(
    ('section', 'camera', 'fraction'),
    [
        ('cassini_iss', 'nac', 0.6),
        ('cassini_iss', 'wac', 0.7),
        ('cassini_iss_calib', 'nac', 0.6),
        ('cassini_iss_calib', 'wac', 0.7),
        ('galileo_ssi', None, 0.6),
        ('newhorizons_lorri', None, 0.6),
        ('voyager_iss', None, 0.6),
        ('sim', None, 0.6),
    ],
)
def test_each_camera_sets_its_lone_star_ellipse(
    section: str, camera: str | None, fraction: float
) -> None:
    """The Cassini WAC believes a lone star out to 0.7 of its window; every other camera 0.6."""
    assert Config().camera_block(section, camera).one_star_ellipse_fraction == fraction


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
