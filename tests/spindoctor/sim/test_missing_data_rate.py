"""The per-pixel missing-data rate works on every frame.

``noise.missing_data_rate`` drops random pixels to the missing-data marker in the
telemetry stage.  A lost pixel takes the marker the navigator applies to the frame,
0 on every shipped instrument, and the validator accepts the rate everywhere.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instrument_catalog import sim_instrument_names
from spindoctor.sim.instrument_navigator import navigator_marker_value, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params

# The shipped entries cover both unit classes: coiss_nac, gossi, nhlorri, and
# generic render raw DN, and coiss_calib_nac, coiss_calib_wac, and vgiss render I/F.
_INSTRUMENTS = (*sim_instrument_names(), 'generic')
_RATE = 0.2


def _scene(instrument: str, rate: float, *, marker: float | None = None) -> dict[str, Any]:
    """A small disc scene on ``instrument`` carrying ``noise.missing_data_rate``.

    ``marker``, when given, overrides the instrument block's missing-data marker
    through ``instrument_config``.
    """
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'missing_data_rate',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 5,
        'instrument': instrument,
        'noise': {'missing_data_rate': rate},
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 32.0,
                'center_u': 32.0,
                'axis1': 20.0,
                'axis2': 20.0,
                'axis3': 20.0,
                'illumination_angle': 0.0,
                'phase_angle': 0.0,
            }
        ],
    }
    if marker is not None:
        scene['instrument_config'] = {'noise': {'marker_value': marker}}
    return scene


def _lost(instrument: str, *, marker: float | None = None) -> tuple[np.ndarray, float]:
    """The values of the pixels the rate changed, and the frame's navigator marker."""
    without, _ = render_combined_model(_scene(instrument, 0.0, marker=marker))
    params = _scene(instrument, _RATE, marker=marker)
    with_rate, _ = render_combined_model(params)
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, instrument, params.get('instrument_config')
    )
    changed = with_rate[with_rate != without]
    return changed, navigator_marker_value(inst_config)


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_every_lost_pixel_carries_the_navigator_marker(instrument: str) -> None:
    """Every pixel the rate changes now reads as the frame's marker."""
    changed, marker = _lost(instrument)
    assert set(np.unique(changed).tolist()) == {marker}


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_the_lost_share_follows_the_rate(instrument: str) -> None:
    """About the rate's share of the 4096 pixels is lost: within 4 binomial sigma."""
    changed, _ = _lost(instrument)
    n_pixels = 64 * 64
    sigma = (n_pixels * _RATE * (1.0 - _RATE)) ** 0.5
    assert abs(changed.size - n_pixels * _RATE) < 4.0 * sigma


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac', 'vgiss'])
def test_a_calibrated_frame_marks_a_lost_pixel_with_zero(instrument: str) -> None:
    """On the calibrated instruments the lost pixels read exactly 0, their marker."""
    changed, _ = _lost(instrument)
    assert set(np.unique(changed).tolist()) == {0.0}


@pytest.mark.parametrize('instrument', ['coiss_nac', 'coiss_calib_nac', 'vgiss'])
def test_a_lost_pixel_carries_a_scene_marker_override(instrument: str) -> None:
    """A scene that moves the marker to 7 gets 7 on every lost pixel, calibrated or not."""
    changed, marker = _lost(instrument, marker=7.0)
    assert marker == 7.0
    assert set(np.unique(changed).tolist()) == {7.0}


@pytest.mark.parametrize('instrument', _INSTRUMENTS)
def test_the_validator_accepts_a_rate_everywhere(instrument: str) -> None:
    """A positive rate validates on every instrument."""
    scene = _scene(instrument, _RATE)
    assert validate_sim_params(scene, source='scene') is scene


@pytest.mark.parametrize('instrument', ['coiss_nac', 'coiss_calib_nac', 'vgiss'])
def test_a_nan_marker_override_puts_nan_on_exactly_the_lost_pixels(instrument: str) -> None:
    """Under a scene's NaN marker, the lost pixels read NaN, and no other pixel is NaN or 0."""
    without, _ = render_combined_model(_scene(instrument, 0.0, marker=math.nan))
    with_rate, _ = render_combined_model(_scene(instrument, _RATE, marker=math.nan))
    lost = np.isnan(with_rate)
    assert int(lost.sum()) > 0
    assert np.array_equal(with_rate[~lost], without[~lost])
    assert int((with_rate == 0.0).sum()) == 0
