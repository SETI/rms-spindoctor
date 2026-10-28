"""Every noise control in the sim editor is live and keeps the scene valid.

The validator accepts every noise key on every instrument and detector model, so
the editor's noise panel leaves each control enabled, writes what the operator
sets into the scene, and the scene that results always validates.

Qt runs headless (offscreen platform).
"""

import copy
import os
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QSpinBox, QWidget
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(f'PyQt6 QApplication init failed: {exc}', allow_module_level=True)

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.sim.instruments import SIM_INSTRUMENTS
from spindoctor.sim.scene import validate_sim_params

# Every control the noise panel builds.
_NOISE_CONTROLS = (
    '_poisson_check',
    '_read_noise_spin',
    '_cosmic_ray_spin',
    '_missing_data_spin',
    '_bias_spin',
    '_bloom_spin',
    '_signal_frac_spin',
)
_INSTRUMENTS = (*sorted(SIM_INSTRUMENTS), 'generic')


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


def _set_scene(model: Any, instrument: str, detector_model: str | None) -> None:
    """Pick ``instrument``, and turn the detector override on with ``detector_model``."""
    model._instrument_combo.setCurrentText(instrument)
    if detector_model is not None:
        model._detector_group.setChecked(True)
        model._detector_model_combo.setCurrentText(detector_model)


def _operate(widget: QWidget) -> None:
    """Operate one noise control the way an operator would, leaving a positive value."""
    if isinstance(widget, QCheckBox):
        widget.setChecked(not widget.isChecked())
        widget.setChecked(not widget.isChecked())
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(widget.minimum() + (widget.maximum() - widget.minimum()) * 0.37))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(widget.minimum() + (widget.maximum() - widget.minimum()) * 0.37)
    else:
        raise TypeError(f'no operation for {type(widget).__name__}')


@pytest.mark.parametrize('detector_model', [None, 'ccd', 'vidicon'])
@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_every_noise_control_is_enabled(
    model: Any, instrument: str, detector_model: str | None
) -> None:
    """No noise control is disabled on any instrument or detector model."""
    _set_scene(model, instrument, detector_model)
    disabled = [attr for attr in _NOISE_CONTROLS if not getattr(model, attr).isEnabled()]
    assert disabled == []


@pytest.mark.parametrize('detector_model', [None, 'ccd', 'vidicon'])
@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_operating_every_noise_control_keeps_the_scene_valid(
    model: Any, instrument: str, detector_model: str | None
) -> None:
    """Whatever an operator does with the noise controls, the scene validates.

    Every noise control is set to a positive value (each check box put through
    both states), then the scene is validated.
    """
    _set_scene(model, instrument, detector_model)
    for attr in _NOISE_CONTROLS:
        _operate(getattr(model, attr))
    validate_sim_params(copy.deepcopy(model.sim_params))
    noise = model.sim_params['noise']
    assert noise['cosmic_ray_events_per_pixel_per_sec'] > 0.0
    assert noise['missing_data_rate'] > 0.0


@pytest.mark.parametrize('instrument', ['vgiss', 'coiss_calib_nac'])
def test_a_rate_survives_a_switch_of_instrument(model: Any, instrument: str) -> None:
    """A rate set on coiss_nac stays in the scene after a switch to another instrument."""
    _set_scene(model, 'coiss_nac', None)
    model._cosmic_ray_spin.setValue(0.001)
    model._missing_data_spin.setValue(0.05)
    _set_scene(model, instrument, None)
    noise = model.sim_params['noise']
    assert noise['cosmic_ray_events_per_pixel_per_sec'] == pytest.approx(0.001)
    assert noise['missing_data_rate'] == pytest.approx(0.05)
