"""The artifact-mode activation keys: one name per quantity, validated per range.

Every artifact mode carries exactly one activation key, and the key names the
quantity it holds: ``expected_events`` (a Poisson expected event count per
frame), ``probability`` (a per-frame activation probability), or
``pixel_fraction`` (the hot-pixel spatial density).  The retired ``incidence``
key, which named all three, is refused with a message naming the mode's
replacement.  ``dead_pixels`` and ``dead_columns`` take a fixed ``count`` in
place of ``expected_events``, never both, and every mode is a no-op when its
activation key is 0.
"""

import re
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifact_modes import (
    ACTIVATION_KEYS,
    ARTIFACT_MODES,
    EXPECTED_EVENTS,
    PIXEL_FRACTION,
    PROBABILITY,
    RENAMED_MODE_PARAMS,
)
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params
from spindoctor.support.types import NDArrayFloatType

# The modes whose activation key is a Poisson expected event count: the line,
# block, garble, spike, and dead-pixel losses, plus the two detector count modes.
_COUNT_MODES = frozenset(
    {
        'missing_lines',
        'partial_lines',
        'missing_blocks',
        'line_garble',
        'pixel_spikes',
        'dead_pixels',
        'dead_columns',
        'radiation_transients',
        'bright_dark_pairs',
    }
)
_DENSITY_MODES = frozenset({'hot_pixels'})
_PROBABILITY_MODES = sorted(set(ARTIFACT_MODES) - _COUNT_MODES - _DENSITY_MODES)
_UNIT_INTERVAL_MODES = sorted(set(_PROBABILITY_MODES) | _DENSITY_MODES)
_DEAD_MODES = ('dead_pixels', 'dead_columns')


def _scene(artifacts: dict[str, Any], instrument: str = 'generic') -> dict[str, Any]:
    """A minimal valid scene carrying an artifacts block."""
    return {
        'schema_version': 2,
        'scene_name': 'probe',
        'instrument': instrument,
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'artifacts': artifacts,
    }


def _expected_key(mode_name: str) -> str:
    """The activation key the mode's mechanic calls for."""
    if mode_name in _COUNT_MODES:
        return EXPECTED_EVENTS
    if mode_name in _DENSITY_MODES:
        return PIXEL_FRACTION
    return PROBABILITY


# --- registry --------------------------------------------------------------


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_activation_key_names_its_mechanic(mode_name: str) -> None:
    """A count mode takes expected_events, hot_pixels pixel_fraction, the rest probability."""
    assert ARTIFACT_MODES[mode_name].activation == _expected_key(mode_name)


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_leads_with_its_activation_key_at_zero(mode_name: str) -> None:
    """The activation key is the mode's first parameter and defaults to 0 (off)."""
    mode = ARTIFACT_MODES[mode_name]
    first = mode.params[0]
    assert (first.name, first.default) == (mode.activation, 0.0)


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_carries_exactly_one_activation_key(mode_name: str) -> None:
    """No mode carries two of the three activation keys."""
    names = {param.name for param in ARTIFACT_MODES[mode_name].params}
    assert names & set(ACTIVATION_KEYS) == {ARTIFACT_MODES[mode_name].activation}


def test_renamed_map_retires_incidence_on_every_mode() -> None:
    """The old -> new table maps incidence to each mode's activation key."""
    expected = {name: {'incidence': mode.activation} for name, mode in ARTIFACT_MODES.items()}
    assert expected == RENAMED_MODE_PARAMS


# --- validation: the retired key -----------------------------------------------


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_retired_incidence_key_is_refused_naming_the_replacement(mode_name: str) -> None:
    """A scene still writing incidence fails, and the message names the new key."""
    new_key = _expected_key(mode_name)
    message = (
        f"artifacts.{mode_name}: 'incidence' was renamed; use '{new_key}' "
        f'({ACTIVATION_KEYS[new_key]})'
    )
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(_scene({mode_name: {'incidence': 0.5}}))


def test_retired_incidence_key_is_refused_beside_the_new_key() -> None:
    """Writing both the old and the new key still fails on the old one."""
    with pytest.raises(SimSceneValidationError, match="'incidence' was renamed"):
        validate_sim_params(_scene({'missing_lines': {'incidence': 1.0, 'expected_events': 1.0}}))


# --- validation: ranges ----------------------------------------------------------


@pytest.mark.parametrize('mode_name', _UNIT_INTERVAL_MODES)
def test_probability_and_pixel_fraction_above_one_are_refused(mode_name: str) -> None:
    """A probability or pixel fraction above 1 fails validation."""
    key = _expected_key(mode_name)
    with pytest.raises(SimSceneValidationError, match=rf'{mode_name}\.{key} must lie in \[0, 1\]'):
        validate_sim_params(_scene({mode_name: {key: 1.5}}))


@pytest.mark.parametrize('mode_name', _UNIT_INTERVAL_MODES)
def test_probability_and_pixel_fraction_of_one_validate(mode_name: str) -> None:
    """The closed upper bound 1 is a valid probability or pixel fraction."""
    validate_sim_params(_scene({mode_name: {_expected_key(mode_name): 1.0}}))


@pytest.mark.parametrize('mode_name', sorted(_COUNT_MODES))
def test_expected_events_above_one_validate(mode_name: str) -> None:
    """An expected event count is unbounded above (many lines may be lost)."""
    validate_sim_params(_scene({mode_name: {EXPECTED_EVENTS: 25.0}}))


@pytest.mark.parametrize('mode_name', sorted(_COUNT_MODES))
def test_negative_expected_events_are_refused(mode_name: str) -> None:
    """A negative expected event count fails validation."""
    with pytest.raises(SimSceneValidationError, match=rf'{mode_name}\.expected_events'):
        validate_sim_params(_scene({mode_name: {EXPECTED_EVENTS: -1.0}}))


# --- dead_pixels / dead_columns: count or expected_events ------------------------


@pytest.mark.parametrize('mode_name', _DEAD_MODES)
def test_count_and_expected_events_together_are_refused(mode_name: str) -> None:
    """A dead-pixel or dead-column map may set count or expected_events, never both."""
    with pytest.raises(SimSceneValidationError, match='mutually exclusive'):
        validate_sim_params(_scene({mode_name: {'count': 10, 'expected_events': 0.0}}))


@pytest.mark.parametrize('mode_name', _DEAD_MODES)
@pytest.mark.parametrize('key', ['count', 'expected_events'])
def test_count_or_expected_events_alone_validates(mode_name: str, key: str) -> None:
    """Either activation quantity alone is a valid dead-pixel or dead-column map."""
    validate_sim_params(_scene({mode_name: {key: 3}}))


def _render_dead(mode_name: str, config: dict[str, Any]) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render one dead-pixel or dead-column map on a flat 64 x 64 frame."""
    frame = SimFrame(
        signal=np.full((64, 64), 50.0, dtype=np.float64),
        point_e=np.zeros((64, 64), dtype=np.float64),
        oversample=1,
        truth={},
    )
    params = {'instrument': 'coiss_nac', 'random_seed': 7, 'artifacts': {mode_name: config}}
    validate_sim_params(_scene({mode_name: config}, instrument='coiss_nac'))
    apply_telemetry(frame, params=params, rng=np.random.default_rng(0))
    record: dict[str, Any] = frame.truth['artifacts'][mode_name]
    return frame.signal, record


@pytest.mark.parametrize('mode_name', _DEAD_MODES)
@pytest.mark.parametrize('key', ['count', 'expected_events'])
def test_dead_modes_are_a_no_op_at_zero(mode_name: str, key: str) -> None:
    """A zero count and a zero expected count both leave the frame untouched."""
    signal, _record = _render_dead(mode_name, {key: 0})
    assert np.all(signal == 50.0)


def test_dead_pixels_count_plants_exactly_that_many() -> None:
    """A fixed count plants exactly that many dead pixels."""
    signal, record = _render_dead('dead_pixels', {'count': 10})
    assert len(record['pixels']) == 10
    assert int(np.count_nonzero(signal == 0.0)) == 10


# --- detector activation reads the registry --------------------------------------


def test_detector_count_mode_is_active_at_any_positive_expected_events() -> None:
    """A detector count mode activates whenever expected_events is positive.

    The mechanic draws the event count, so even a tiny expected count must not
    pass through a per-frame probability draw (which would almost always fail).
    """
    dp = resolve_detector_params(
        _scene({'bright_dark_pairs': {'expected_events': 1.0e-9}}, instrument='coiss_nac')
    )
    assert 'bright_dark_pairs' in dp.detector_modes


def test_detector_probability_mode_is_inactive_at_zero() -> None:
    """A detector probability mode at probability 0 is a no-op."""
    dp = resolve_detector_params(
        _scene({'banding_coherent': {'probability': 0.0}}, instrument='coiss_nac')
    )
    assert 'banding_coherent' not in dp.detector_modes


def test_detector_probability_mode_is_active_at_one() -> None:
    """A detector probability mode at probability 1 always activates."""
    dp = resolve_detector_params(
        _scene({'banding_coherent': {'probability': 1.0}}, instrument='coiss_nac')
    )
    assert 'banding_coherent' in dp.detector_modes


def test_hot_pixels_pixel_fraction_sets_the_hot_pixel_fraction() -> None:
    """The hot_pixels mode's pixel_fraction is the resolved hot-pixel density."""
    dp = resolve_detector_params(
        _scene({'hot_pixels': {'pixel_fraction': 0.004}}, instrument='coiss_nac')
    )
    assert dp.hot_pixel_fraction == 0.004
