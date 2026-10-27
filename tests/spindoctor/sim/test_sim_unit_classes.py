"""The unit class of each sim instrument, and how a scene author is told about it.

Three unit classes exist.  The raw instruments render DN with a ``0`` missing-data
marker.  The calibrated Cassini instruments render I/F with a ``NaN`` marker.
``vgiss`` renders I/F with a ``0`` marker, because the Voyager GEOMED products the
navigator reads fill lost pixels with zero.  In every class the simulator writes
the marker the navigator applies to the same frame, and
:func:`~spindoctor.sim.instruments.frame_units_summary` warns when that marker
covers enough of the frame for the navigator to refuse it.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_orchestrator.image_classifier import NavImageClassifier
from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry
from spindoctor.sim.instruments import (
    SIM_INSTRUMENTS,
    frame_units_summary,
    navigator_marker_value,
    resolve_sim_inst_config,
)
from spindoctor.sim.render import render_combined_model

_ALL_INSTRUMENTS = [*sorted(SIM_INSTRUMENTS), 'generic']


def _navigator_marker(instrument: str, overrides: dict[str, Any] | None = None) -> float:
    """The marker the navigator's instrument settings apply to a sim frame."""
    params: dict[str, Any] = {'size_v': 32, 'size_u': 32, 'random_seed': 1}
    params['instrument'] = instrument
    if overrides is not None:
        params['instrument_config'] = overrides
    obs = ObsSim.from_file('/tmp/sim_unit_classes.json', sim_params=params)
    return instrument_settings_from_obs(obs).marker_value


def _same(a: float, b: float) -> bool:
    """Equality that treats two NaNs as equal."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def _disc_scene(instrument: str) -> dict[str, Any]:
    """A 128 x 128 frame whose lit disc covers about a third of the pixels."""
    return {
        'size_v': 128,
        'size_u': 128,
        'instrument': instrument,
        'random_seed': 42,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 64.0,
                'center_u': 64.0,
                'axis1': 86.0,
                'axis2': 86.0,
                'axis3': 86.0,
                'illumination_angle': 18.0,
                'phase_angle': 30.0,
            }
        ],
    }


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_lost_lines_carry_the_navigator_marker(instrument: str) -> None:
    """Every instrument writes, on a lost line, the marker the navigator applies."""
    frame = SimFrame(
        signal=np.full((64, 64), 0.5, dtype=np.float64),
        point_e=np.zeros((64, 64), dtype=np.float64),
        oversample=1,
        truth={},
    )
    apply_telemetry(
        frame,
        params={
            'instrument': instrument,
            'random_seed': 7,
            'artifacts': {'missing_lines': {'expected_events': 4.0}},
        },
        rng=np.random.default_rng(0),
    )
    lines = frame.truth['artifacts']['missing_lines']['lines']
    assert lines
    written = float(frame.signal[lines[0], 0])
    assert _same(written, _navigator_marker(instrument))


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_marker_helper_agrees_with_the_navigator(instrument: str) -> None:
    """navigator_marker_value reads each instrument's marker as the navigator does."""
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    assert _same(navigator_marker_value(inst_config), _navigator_marker(instrument))


@pytest.mark.parametrize(
    ('instrument', 'overrides', 'expected'),
    [
        ('vgiss', {'noise': {'marker_value': None}}, math.nan),
        ('vgiss', {'noise': {'marker_value': 'nan'}}, math.nan),
        ('generic', {'noise': {'marker_value': 7}}, 7.0),
    ],
    ids=['calibrated-null', 'calibrated-nan-string', 'raw-explicit'],
)
def test_marker_helper_follows_an_overridden_marker(
    instrument: str, overrides: dict[str, Any], expected: float
) -> None:
    """An instrument_config marker override reads the same in the helper and navigator."""
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument, overrides)
    helper = navigator_marker_value(inst_config)
    assert _same(helper, _navigator_marker(instrument, overrides))
    assert _same(helper, expected)


def test_marker_helper_defaults_a_calibrated_block_to_nan() -> None:
    """A calibrated block that omits marker_value defaults to NaN, as in the navigator."""
    assert math.isnan(navigator_marker_value({'data_units': 'calibrated_if', 'noise': {}}))


def test_marker_helper_rejects_a_non_nan_string() -> None:
    """A marker string other than NaN is refused with the navigator's message."""
    with pytest.raises(ValueError, match='must be "NaN" or numeric'):
        navigator_marker_value({'noise': {'marker_value': 'zero'}})


@pytest.mark.parametrize(
    ('instrument', 'units'),
    [('coiss_nac', 'DN'), ('gossi', 'DN'), ('coiss_calib_nac', 'I/F'), ('vgiss', 'I/F')],
)
def test_summary_names_the_unit_class(instrument: str, units: str) -> None:
    """The summary opens with the instrument's units."""
    img, _ = render_combined_model(_disc_scene(instrument), ignore_offset=True)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    assert frame_units_summary(img, inst_config).split(' ')[0] == units


@pytest.mark.parametrize('instrument', ['coiss_nac', 'coiss_calib_nac', 'vgiss'])
def test_summary_warning_matches_the_classifier_verdict(instrument: str) -> None:
    """The summary warns of a refusal exactly when the navigator's classifier refuses."""
    params = _disc_scene(instrument)
    img, _ = render_combined_model(params, ignore_offset=True)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    settings = instrument_settings_from_obs(
        ObsSim.from_file('/tmp/sim_unit_classes.json', sim_params=params)
    )
    verdict = NavImageClassifier(thresholds=settings.thresholds).classify(img)
    warns = 'refuses' in frame_units_summary(img, inst_config)
    assert warns == (verdict.image_class == 'mostly_missing_data')


def test_vgiss_dark_sky_reads_as_missing_data() -> None:
    """A vgiss frame's exactly-zero dark sky is counted at the 0 marker and warned of."""
    img, _ = render_combined_model(_disc_scene('vgiss'), ignore_offset=True)
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss'))
    share = float((img == 0.0).mean()) * 100
    assert f'marker 0 on {share:.1f}% of pixels (over 30%' in summary


def test_summary_counts_nan_markers_on_a_calibrated_frame() -> None:
    """On a NaN-marker instrument the summary counts the NaN pixels."""
    img = np.full((10, 10), 0.25)
    img[0, :] = np.nan
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_calib_nac'))
    assert summary == 'I/F 0.25 to 0.25; marker NaN on 10.0% of pixels'


def test_summary_omits_the_marker_when_no_pixel_carries_it() -> None:
    """A frame without marker pixels reports only its units and range."""
    img = np.full((10, 10), 20.0)
    img[5, 5] = 3000.0
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac'))
    assert summary == 'DN 20 to 3000'


@pytest.mark.parametrize(
    ('marker_rows', 'warns'),
    [(20, False), (30, False), (31, True)],
    ids=['20pct', 'at-30pct-limit', '31pct'],
)
def test_summary_warns_only_above_the_missing_data_limit(marker_rows: int, warns: bool) -> None:
    """The refusal warning starts strictly above the 30% the navigator accepts."""
    img = np.full((100, 10), 0.5)
    img[:marker_rows, :] = 0.0
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss'))
    assert ('refuses' in summary) == warns
