"""The artifact-mode activation keys: one name per quantity, validated per range.

Every artifact mode carries exactly one activation key, and the key names the
quantity it holds: ``expected_events`` (a Poisson expected event count per
frame), ``expected_events_per_sec`` (a Poisson expected event count per second of
exposure, on ``bright_dark_pairs``), ``probability`` (a per-frame activation
probability), or ``pixel_fraction`` (the hot-pixel spatial density).
``dead_pixels`` and ``dead_columns`` take a fixed ``count`` in place of
``expected_events``, never both set, and every mode is a no-op when its
activation key is 0, or its ``count`` where given is 0.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifact_modes import (
    ACTIVATION_KEYS,
    ARTIFACT_MODES,
    EXPECTED_EVENTS,
    EXPECTED_EVENTS_PER_SEC,
    PIXEL_FRACTION,
    PROBABILITY,
    resolve_mode_config,
)
from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params
from spindoctor.support.types import NDArrayFloatType

# The modes whose activation key is a Poisson expected event count per frame: the
# line, block, garble, spike, and dead-pixel losses, plus radiation_transients.
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
    }
)
# The mode whose expected count is a rate per second of exposure.
_RATE_MODES = frozenset({'bright_dark_pairs'})
_POISSON_MODES = sorted(_COUNT_MODES | _RATE_MODES)
_DENSITY_MODES = frozenset({'hot_pixels'})
_PROBABILITY_MODES = sorted(set(ARTIFACT_MODES) - _COUNT_MODES - _RATE_MODES - _DENSITY_MODES)
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
    if mode_name in _RATE_MODES:
        return EXPECTED_EVENTS_PER_SEC
    if mode_name in _DENSITY_MODES:
        return PIXEL_FRACTION
    return PROBABILITY


# --- registry --------------------------------------------------------------


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_activation_key_names_its_mechanic(mode_name: str) -> None:
    """Each mode's activation key names the quantity its mechanic reads.

    A per-frame count mode takes expected_events, bright_dark_pairs (whose count
    scales with exposure) expected_events_per_sec, hot_pixels pixel_fraction, and
    the rest probability.
    """
    assert ARTIFACT_MODES[mode_name].activation == _expected_key(mode_name)


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_leads_with_its_activation_key_at_zero(mode_name: str) -> None:
    """The activation key is the mode's first parameter and defaults to 0 (off)."""
    mode = ARTIFACT_MODES[mode_name]
    first = mode.params[0]
    assert (first.name, first.default) == (mode.activation, 0.0)


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_each_mode_carries_exactly_one_activation_key(mode_name: str) -> None:
    """No mode carries two of the activation keys."""
    names = {param.name for param in ARTIFACT_MODES[mode_name].params}
    assert names & set(ACTIVATION_KEYS) == {ARTIFACT_MODES[mode_name].activation}


def test_radiation_transients_semantics_state_its_scaling() -> None:
    """radiation_transients names the environment factor and the dwell it scales by."""
    semantics = ARTIFACT_MODES['radiation_transients'].activation_semantics
    words = ('environment_factor', 'readout_dwell_sec', 'exposure_sec')
    assert [word for word in words if word not in semantics] == []


@pytest.mark.parametrize('mode_name', sorted(ARTIFACT_MODES))
def test_plain_resolver_reads_a_null_parameter_as_unset(mode_name: str) -> None:
    """A parameter set to null resolves to its registry default, as the catalog resolver does."""
    params = ARTIFACT_MODES[mode_name].params
    resolved = resolve_mode_config(mode_name, {param.name: None for param in params})
    assert resolved == {param.name: param.default for param in params}


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


@pytest.mark.parametrize('mode_name', _POISSON_MODES)
def test_expected_events_above_one_validate(mode_name: str) -> None:
    """An expected event count is unbounded above (many lines may be lost)."""
    validate_sim_params(_scene({mode_name: {_expected_key(mode_name): 25.0}}))


@pytest.mark.parametrize('mode_name', _POISSON_MODES)
def test_negative_expected_events_are_refused(mode_name: str) -> None:
    """A negative expected event count fails validation."""
    key = _expected_key(mode_name)
    with pytest.raises(SimSceneValidationError, match=rf'{mode_name}\.{key} '):
        validate_sim_params(_scene({mode_name: {key: -1.0}}))


def test_rate_mode_refuses_the_per_frame_key() -> None:
    """bright_dark_pairs takes expected_events_per_sec; expected_events is unknown on it."""
    with pytest.raises(SimSceneValidationError, match=r"unknown keys: \['expected_events'\]"):
        validate_sim_params(_scene({'bright_dark_pairs': {EXPECTED_EVENTS: 5.0}}))


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


@pytest.mark.parametrize('mode_name', _DEAD_MODES)
@pytest.mark.parametrize(
    'config',
    [{'count': None, 'expected_events': 3.0}, {'count': 3, 'expected_events': None}],
    ids=['null-count', 'null-expected'],
)
def test_a_null_key_beside_the_other_validates(mode_name: str, config: dict[str, Any]) -> None:
    """A key set to null counts as unset, so it never conflicts with the other key."""
    validate_sim_params(_scene({mode_name: config}))


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


def test_dead_pixels_null_count_draws_from_expected_events() -> None:
    """A null count leaves expected_events in charge of the draw."""
    signal, record = _render_dead('dead_pixels', {'count': None, 'expected_events': 20.0})
    assert len(record['pixels']) > 0
    assert int(np.count_nonzero(signal == 0.0)) > 0


def test_dead_pixels_expected_events_plants_its_pixels() -> None:
    """A large expected_events plants dead pixels, each one dark in the frame."""
    signal, record = _render_dead('dead_pixels', {'expected_events': 20.0})
    planted = {(v, u) for v, u in record['pixels']}
    dark = {(int(v), int(u)) for v, u in zip(*np.nonzero(signal == 0.0), strict=True)}
    assert len(planted) >= 5
    assert dark == planted


def test_dead_columns_expected_events_plants_its_columns() -> None:
    """A large expected_events plants dead columns, each one dark top to bottom."""
    signal, record = _render_dead('dead_columns', {'expected_events': 20.0})
    dark = [int(u) for u in np.nonzero(np.all(signal == 0.0, axis=0))[0]]
    assert len(record['columns']) >= 5
    assert dark == record['columns']


# --- bright_dark_pairs: a rate per second of exposure -----------------------------


def _pair_count(rate_per_sec: float, exposure_sec: float) -> int:
    """Render bright_dark_pairs on Cassini NAC and count the planted pairs."""
    sim = _scene({'bright_dark_pairs': {EXPECTED_EVENTS_PER_SEC: rate_per_sec}}, 'coiss_nac')
    sim['exposure_sec'] = exposure_sec
    validate_sim_params(sim)
    _image, truth = render_combined_model(sim, ignore_offset=True)
    return len(truth['artifacts']['bright_dark_pairs']['pairs'])


def test_bright_dark_pairs_plant_pairs_at_a_large_rate() -> None:
    """At 400 pairs per second over a 1 s exposure the frame carries about 400 pairs."""
    assert 300 < _pair_count(400.0, 1.0) < 500


def test_bright_dark_pairs_count_scales_with_exposure() -> None:
    """The same rate over a 0.05 s exposure expects 20 pairs (Poisson), not 400."""
    assert 5 < _pair_count(400.0, 0.05) < 40


# --- detector activation reads the registry --------------------------------------


def test_detector_count_mode_is_active_at_any_positive_expected_events() -> None:
    """A detector count mode activates whenever expected_events is positive.

    The mechanic draws the event count, so even a tiny expected count must not
    pass through a per-frame probability draw (which would almost always fail).
    """
    dp = resolve_detector_params(
        _scene({'radiation_transients': {EXPECTED_EVENTS: 1.0e-9}}, instrument='coiss_nac')
    )
    assert 'radiation_transients' in dp.detector_modes


def test_detector_rate_mode_is_active_at_any_positive_rate() -> None:
    """bright_dark_pairs activates whenever its per-second rate is positive."""
    dp = resolve_detector_params(
        _scene({'bright_dark_pairs': {EXPECTED_EVENTS_PER_SEC: 1.0e-9}}, instrument='coiss_nac')
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
