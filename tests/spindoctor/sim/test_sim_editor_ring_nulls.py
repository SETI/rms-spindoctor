"""The sim editor reads a null optional ring key as unset, as the validator does.

Qt runs headless (offscreen platform).
"""

import os
from pathlib import Path
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(f'PyQt6 QApplication init failed: {exc}', allow_module_level=True)

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.sim.scene import load_sim_scene, validate_sim_params

_SCENES = Path(__file__).parents[2] / 'integration' / 'sim_scenes'


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


def _null_scene() -> dict[str, Any]:
    """The m-mode ringlet scene with every optional ring key its tabs read set to null."""
    scene = load_sim_scene(_SCENES / 'ring_system' / 'mmode_ringlet.yaml')
    ring_system = scene['ring_system']
    for key in ('center_v', 'center_u', 'node_deg'):
        ring_system['geometry'][key] = None
    ring_system['phase_deg'] = None
    ring_system['km_per_pixel'] = None
    feature = ring_system['features'][0]
    for key in ('albedo', 'phase_g'):
        feature[key] = None
    for key in ('ae_px', 'long_peri_deg', 'rate_peri_deg_per_day'):
        feature['orbit'][key] = None
    for key in ('amp_px', 'peri_deg'):
        feature['orbit']['modes'][0][key] = None
    ring_system['azimuthal'] = {
        'modulation': {'amplitude': 0.2, 'm': None, 'brightness_peak_longitude_deg': None},
        'shadow': {'start_deg': None, 'extent_deg': 30.0, 'darkness': 0.5},
    }
    return validate_sim_params(scene, source='case')


def test_the_editor_loads_a_ring_scene_whose_optional_keys_are_null(
    model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ring tab builds from null optional keys, renders, and keeps the scene's nulls."""
    errors: list[str] = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **_k: errors.append(str(a[2])))
    model._apply_params_dict(_null_scene())
    assert errors == []
    assert model._find_tab_by_properties('ring', 0) is not None
    assert model.sim_params['ring_system']['features'][0]['orbit']['ae_px'] is None
