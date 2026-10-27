"""The per-pixel missing-data rate is refused on the frames that would ignore it.

``noise.missing_data_rate`` drops random pixels to the missing-data marker in the
telemetry stage, on the raw-DN path only.  A calibrated (``calibrated_if``)
frame renders no dropouts at any rate, so the validator refuses a non-zero rate
on a scene whose resolved instrument block is calibrated, and on no other scene.
A zero rate passes everywhere.
"""

import re
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.instruments import SIM_INSTRUMENTS
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params

_INSTRUMENTS = (*sorted(SIM_INSTRUMENTS), 'generic')
# The data_units a property case pins through instrument_config: the
# instrument's own block, or each unit class forced over it.
_UNITS = (None, 'raw_dn', 'calibrated_if')


def _scene(instrument: str, rate: float, data_units: str | None) -> dict[str, Any]:
    """A small disc scene on ``instrument`` carrying ``noise.missing_data_rate``."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'missing_data_rate',
        'size_v': 32,
        'size_u': 32,
        'random_seed': 5,
        'instrument': instrument,
        'noise': {'missing_data_rate': rate},
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 16.0,
                'center_u': 16.0,
                'axis1': 10.0,
                'axis2': 10.0,
                'axis3': 10.0,
                'illumination_angle': 0.0,
                'phase_angle': 0.0,
            }
        ],
    }
    if data_units is not None:
        scene['instrument_config'] = {'data_units': data_units}
    return scene


def _refused(scene: dict[str, Any]) -> bool:
    """Whether the validator refuses ``scene``."""
    try:
        validate_sim_params(scene, source='scene')
    except SimSceneValidationError:
        return True
    return False


@pytest.mark.parametrize('data_units', _UNITS, ids=['own', 'raw', 'calibrated'])
@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_validator_refuses_exactly_the_frames_where_the_rate_is_inert(
    instrument: str, data_units: str | None
) -> None:
    """The validator refuses a rate if and only if the rendered frame ignores it."""
    without, _ = render_combined_model(_scene(instrument, 0.0, data_units))
    with_rate, _ = render_combined_model(_scene(instrument, 0.2, data_units))
    inert = bool(np.array_equal(without, with_rate, equal_nan=True))
    assert _refused(_scene(instrument, 0.2, data_units)) is inert


@pytest.mark.parametrize(
    ('instrument', 'data_units', 'via'),
    [
        ('vgiss', None, ''),
        ('coiss_calib_nac', None, ''),
        ('coiss_nac', 'calibrated_if', ' via instrument_config.data_units'),
    ],
    ids=['vgiss', 'coiss-calib-nac', 'coiss-nac-pinned'],
)
def test_nonzero_rate_on_a_calibrated_frame_names_the_reason(
    instrument: str, data_units: str | None, via: str
) -> None:
    """The refusal says the dropout is raw-DN only and points at the artifact modes."""
    message = (
        f'scene: noise.missing_data_rate is 0.2, but instrument {instrument!r} renders '
        f'calibrated_if{via}, and the per-pixel dropout applies to raw-DN frames only, so '
        'the rate would be silently ignored; set it to 0 or remove it, and plant loss on '
        'a calibrated frame with an artifact mode such as missing_lines'
    )
    with pytest.raises(SimSceneValidationError, match=f'^{re.escape(message)}$'):
        validate_sim_params(_scene(instrument, 0.2, data_units), source='scene')


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_zero_rate_is_accepted_everywhere(instrument: str) -> None:
    """A zero rate is the off state, and every instrument accepts it."""
    assert _refused(_scene(instrument, 0.0, None)) is False
