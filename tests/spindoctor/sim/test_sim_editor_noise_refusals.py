"""The sim editor cannot author a noise rate its scene validator refuses.

The validator refuses a positive ``noise.cosmic_ray_events_per_pixel_per_sec``
when the detector model resolves to the vidicon, and a positive
``noise.missing_data_rate`` on a ``calibrated_if`` frame.  The editor reads the
validator's own predicates: where one refuses, its control is disabled with the
reason as its tooltip.  A change of instrument, detector model, or detector
override that makes a set rate refused moves the rate into the editor's stash
with a status-bar message, so the scene keeps validating, and a change that lifts
the refusal restores it.

The property itself is checked by operating every noise control the panel leaves
enabled and validating the scene that results.

Qt runs headless (offscreen platform).
"""

import copy
import os
import re
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
from spindoctor.cli.sim_editor.noise import noise_refusal_tooltip, noise_stash_message
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params

_CR = 'cosmic_ray_events_per_pixel_per_sec'
_MD = 'missing_data_rate'
_SPIN = {_CR: '_cosmic_ray_spin', _MD: '_missing_data_spin'}
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


def _spin(model: Any, key: str) -> QDoubleSpinBox:
    """The control that writes ``noise.<key>``."""
    return cast(QDoubleSpinBox, getattr(model, _SPIN[key]))


def _set_instrument(model: Any, instrument: str) -> None:
    """Pick ``instrument`` in the General tab's combo, as an operator would."""
    model._instrument_combo.setCurrentText(instrument)


def _set_detector_model(model: Any, detector_model: str) -> None:
    """Turn the detector override on and pick ``detector_model``."""
    model._detector_group.setChecked(True)
    model._detector_model_combo.setCurrentText(detector_model)


def _validator_reason(scene: dict[str, Any], key: str) -> str:
    """The reason the validator gives for refusing a positive ``noise.<key>``.

    The scene's rate is set to 0.5 and validated; the reason is the message's
    clause between ``but`` and the remedy.
    """
    scene = copy.deepcopy(scene)
    scene.setdefault('noise', {})[key] = 0.5
    with pytest.raises(SimSceneValidationError) as excinfo:
        validate_sim_params(scene, source='SRC')
    match = re.fullmatch(
        rf'SRC: noise\.{key} is 0\.5, but (.*?); set it to 0 or remove it.*',
        str(excinfo.value),
    )
    assert match is not None, str(excinfo.value)
    return match.group(1)


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


# ---- Which controls are disabled, and why ----


@pytest.mark.parametrize(
    ('instrument', 'detector_model', 'refused'),
    [
        ('coiss_nac', None, set()),
        ('coiss_nac', 'vidicon', {_CR}),
        ('vgiss', None, {_CR, _MD}),
        ('coiss_calib_nac', None, {_MD}),
        ('coiss_calib_wac', None, {_MD}),
        ('generic', None, set()),
    ],
    ids=[
        'coiss-nac',
        'coiss-nac-vidicon',
        'vgiss',
        'coiss-calib-nac',
        'coiss-calib-wac',
        'generic',
    ],
)
def test_exactly_the_refused_controls_are_disabled(
    model: Any, instrument: str, detector_model: str | None, refused: set[str]
) -> None:
    """A noise control is disabled where, and only where, the validator refuses its rate."""
    _set_instrument(model, instrument)
    if detector_model is not None:
        _set_detector_model(model, detector_model)
    disabled = {key for key in _SPIN if not _spin(model, key).isEnabled()}
    assert disabled == refused


@pytest.mark.parametrize(
    ('instrument', 'detector_model', 'key'),
    [
        ('vgiss', None, _CR),
        ('vgiss', None, _MD),
        ('coiss_nac', 'vidicon', _CR),
        ('coiss_calib_nac', None, _MD),
    ],
    ids=['vgiss-cosmic', 'vgiss-missing', 'coiss-nac-vidicon-cosmic', 'coiss-calib-nac-missing'],
)
def test_a_disabled_control_carries_the_validators_reason(
    model: Any, instrument: str, detector_model: str | None, key: str
) -> None:
    """A disabled noise control's tooltip is the reason the validator gives."""
    _set_instrument(model, instrument)
    if detector_model is not None:
        _set_detector_model(model, detector_model)
    reason = _validator_reason(model.sim_params, key)
    assert _spin(model, key).toolTip() == noise_refusal_tooltip(reason)


def test_an_accepting_instrument_gives_each_control_its_own_tooltip(model: Any) -> None:
    """Back on a raw-DN CCD instrument the controls lose the refusal tooltip."""
    _set_instrument(model, 'vgiss')
    _set_instrument(model, 'coiss_nac')
    tooltips = [_spin(model, key).toolTip() for key in _SPIN]
    assert [t for t in tooltips if t.startswith('Validation refuses')] == []


def test_a_pinned_calibrated_frame_disables_the_missing_data_control(model: Any) -> None:
    """A loaded scene whose instrument_config pins calibrated_if disables the control."""
    model._apply_params_dict(
        {'instrument': 'coiss_nac', 'instrument_config': {'data_units': 'calibrated_if'}}
    )
    assert not _spin(model, _MD).isEnabled()


# ---- The property: the editor's noise controls keep the scene valid ----


@pytest.mark.parametrize('instrument', ['vgiss', 'coiss_calib_nac', 'coiss_nac'])
def test_operating_every_enabled_noise_control_keeps_the_scene_valid(
    model: Any, instrument: str
) -> None:
    """Whatever an operator does with the live noise controls, the scene validates.

    Every enabled noise control is set to a positive value (each check box put
    through both states), then the scene is validated.
    """
    _set_instrument(model, instrument)
    operated = []
    for attr in _NOISE_CONTROLS:
        widget = getattr(model, attr)
        if widget.isEnabled():
            _operate(widget)
            operated.append(attr)
    assert len(operated) >= len(_NOISE_CONTROLS) - 2
    validate_sim_params(copy.deepcopy(model.sim_params))


# ---- Switches that make a set rate refused ----


@pytest.mark.parametrize(
    ('key', 'value', 'refusing'),
    [(_CR, 0.001, 'vgiss'), (_MD, 0.05, 'coiss_calib_nac'), (_MD, 0.05, 'vgiss')],
    ids=['cosmic-to-vgiss', 'missing-to-coiss-calib-nac', 'missing-to-vgiss'],
)
def test_switching_to_a_refusing_instrument_moves_the_rate_out(
    model: Any, key: str, value: float, refusing: str
) -> None:
    """A rate set on coiss_nac leaves the scene on an instrument that refuses it."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, key).setValue(value)
    _set_instrument(model, refusing)
    assert key not in model.sim_params['noise']
    assert _spin(model, key).value() == 0.0
    validate_sim_params(copy.deepcopy(model.sim_params))


@pytest.mark.parametrize(
    ('key', 'value', 'refusing'),
    [(_CR, 0.001, 'vgiss'), (_MD, 0.05, 'coiss_calib_nac')],
    ids=['cosmic-via-vgiss', 'missing-via-coiss-calib-nac'],
)
def test_switching_back_restores_the_rate(
    model: Any, key: str, value: float, refusing: str
) -> None:
    """A round trip through a refusing instrument gives the scene its rate back."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, key).setValue(value)
    _set_instrument(model, refusing)
    _set_instrument(model, 'coiss_nac')
    assert model.sim_params['noise'][key] == pytest.approx(value)
    assert _spin(model, key).value() == pytest.approx(value)
    assert _spin(model, key).isEnabled()


def test_the_vidicon_detector_model_moves_the_cosmic_ray_rate_out(model: Any) -> None:
    """Choosing the vidicon model on coiss_nac moves the rate out; ccd brings it back."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, _CR).setValue(0.002)
    _set_detector_model(model, 'vidicon')
    assert _CR not in model.sim_params['noise']
    validate_sim_params(copy.deepcopy(model.sim_params))
    _set_detector_model(model, 'ccd')
    assert model.sim_params['noise'][_CR] == pytest.approx(0.002)


def test_dropping_the_detector_override_restores_the_rate(model: Any) -> None:
    """Unchecking a vidicon override returns coiss_nac to its CCD chain and the rate."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, _CR).setValue(0.002)
    _set_detector_model(model, 'vidicon')
    model._detector_group.setChecked(False)
    assert model.sim_params['noise'][_CR] == pytest.approx(0.002)
    assert _spin(model, _CR).isEnabled()


def test_a_switch_that_moves_a_rate_names_it_in_the_status_bar(model: Any) -> None:
    """The status bar names every rate the switch moved and says how it comes back."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, _CR).setValue(0.001)
    _spin(model, _MD).setValue(0.05)
    _set_instrument(model, 'vgiss')
    message = model.statusBar().currentMessage()
    assert message == noise_stash_message([_CR, _MD])
    assert f'noise.{_CR}, noise.{_MD}' in message
    assert 'Switching back restores it in this session' in message


def test_a_switch_that_moves_nothing_shows_no_message(model: Any) -> None:
    """A zero rate stays in the scene and the switch shows no message."""
    _set_instrument(model, 'vgiss')
    assert model.statusBar().currentMessage() == ''
    assert model.sim_params['noise'][_CR] == 0.0


def test_loading_a_scene_forgets_the_stash(model: Any) -> None:
    """A rate stashed before a load doesn't come back into the loaded scene."""
    _set_instrument(model, 'coiss_nac')
    _spin(model, _CR).setValue(0.001)
    _set_instrument(model, 'vgiss')
    model._apply_params_dict({'instrument': 'vgiss', 'noise': {'poisson': True}})
    _set_instrument(model, 'coiss_nac')
    assert _CR not in model.sim_params['noise']
