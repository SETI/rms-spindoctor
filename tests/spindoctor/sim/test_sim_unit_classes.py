"""The missing-data marker of each sim instrument's frames.

Two unit classes exist.  The raw instruments render DN with a ``0`` missing-data
marker.  The calibrated instruments (the calibrated Cassini cameras and ``vgiss``)
render I/F with a ``0`` marker, because the Cassini CALIB and Voyager GEOMED
products the navigator reads fill lost pixels with zero.  In every class the
simulator writes the marker the navigator applies to the same frame.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instruments.catalog import sim_instrument_names
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.stage import apply_telemetry

_ALL_INSTRUMENTS = [*sim_instrument_names(), 'generic']


def _navigator_marker(instrument: str, overrides: dict[str, Any] | None = None) -> float:
    """The marker the navigator's instrument settings apply to a sim frame."""
    params = frame_scene(instrument=instrument, size=32)
    if overrides is not None:
        params['instrument_config'] = overrides
    obs = ObsSim.from_file('/tmp/sim_unit_classes.json', sim_params=params)
    return instrument_settings_from_obs(obs).marker_value


def _same(a: float, b: float) -> bool:
    """Equality that treats two NaNs as equal."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def _telemetry_frame(signal: np.ndarray, params: dict[str, Any]) -> SimFrame:
    """Run the telemetry stage over ``signal`` and return the frame."""
    frame = SimFrame(
        signal=signal,
        point_e=np.zeros(signal.shape, dtype=np.float64),
        oversample=1,
        truth={},
    )
    apply_telemetry(frame, params=params)
    return frame


def _lost_line_value(instrument: str, overrides: dict[str, Any] | None = None) -> float:
    """The value the telemetry stage writes on the first lost line of a flat frame."""
    params: dict[str, Any] = {
        'instrument': instrument,
        'random_seed': 7,
        'artifacts': {'missing_lines': {'expected_events': 4.0}},
    }
    if overrides is not None:
        params['instrument_config'] = overrides
    frame = _telemetry_frame(np.full((64, 64), 0.5, dtype=np.float64), params)
    lines = frame.truth['artifacts']['missing_lines']['lines']
    assert lines
    return float(frame.signal[lines[0], 0])


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_lost_lines_carry_the_navigator_marker(instrument: str) -> None:
    """Every instrument writes, on a lost line, the marker the navigator applies."""
    assert _same(_lost_line_value(instrument), _navigator_marker(instrument))


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_every_instrument_marks_a_lost_line_with_zero(instrument: str) -> None:
    """Every instrument's configuration marks a lost pixel with 0."""
    assert _lost_line_value(instrument) == 0.0


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_lost_lines_carry_a_nan_marker_as_the_navigator_reads_it(instrument: str) -> None:
    """A marker_value of NaN is NaN to the navigator, and on every lost line."""
    overrides = {'noise': {'marker_value': math.nan}}
    written = _lost_line_value(instrument, overrides)
    assert _same(written, _navigator_marker(instrument, overrides))
