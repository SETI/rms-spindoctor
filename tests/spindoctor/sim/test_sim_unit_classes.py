"""The unit class of each sim instrument, and how a scene author is told about it.

Two unit classes exist.  The raw instruments render DN with a ``0`` missing-data
marker.  The calibrated instruments (the calibrated Cassini cameras and ``vgiss``)
render I/F with a ``0`` marker, because the Cassini CALIB and Voyager GEOMED
products the navigator reads fill lost pixels with zero.  In every class the
simulator writes the marker the navigator applies to the same frame, and
:func:`~spindoctor.sim.instruments.read_frame_as_navigator` reads the frame the
navigator reads (planted offset applied) to the verdict the navigator's
classifier reaches, blank before missing data.
:func:`~spindoctor.sim.instruments.frame_units_summary` states that verdict.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.image_classifier import NavImageClassifier
from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry
from spindoctor.sim.instruments import (
    SIM_INSTRUMENTS,
    frame_units_name,
    frame_units_summary,
    has_planted_offset,
    navigator_marker_value,
    read_frame_as_navigator,
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


def _body_scene(
    instrument: str,
    axis: float,
    offset_vu: tuple[float, float],
    *,
    center_u: float = 64.0,
    light: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """A 128 x 128 frame with one Rhea-like body and a planted offset."""
    return {
        'size_v': 128,
        'size_u': 128,
        'instrument': instrument,
        'random_seed': 42,
        'offset_v': offset_vu[0],
        'offset_u': offset_vu[1],
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 64.0,
                'center_u': center_u,
                'axis1': axis,
                'axis2': axis,
                'axis3': axis,
                'illumination_angle': light[0],
                'phase_angle': light[1],
            }
        ],
    }


def _with_lost_lines(params: dict[str, Any], expected_events: float) -> dict[str, Any]:
    """``params`` with a ``missing_lines`` mode planting about that many lost lines."""
    return {**params, 'artifacts': {'missing_lines': {'expected_events': expected_events}}}


def _empty_scene(instrument: str) -> dict[str, Any]:
    """A 64 x 64 frame with nothing in it."""
    return {'size_v': 64, 'size_u': 64, 'instrument': instrument, 'random_seed': 1}


def _orchestrator_class(params: dict[str, Any]) -> str:
    """The class the navigator's orchestrator gives the frame it renders from ``params``."""
    obs = ObsSim.from_file('/tmp/sim_unit_classes.json', sim_params=params)
    return NavOrchestrator([]).prepare(obs).image_classifier.image_class


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


def _telemetry_frame(signal: np.ndarray, params: dict[str, Any]) -> SimFrame:
    """Run the telemetry stage over ``signal`` and return the frame."""
    frame = SimFrame(
        signal=signal,
        point_e=np.zeros(signal.shape, dtype=np.float64),
        oversample=1,
        truth={},
    )
    apply_telemetry(frame, params=params, rng=np.random.default_rng(0))
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
    """Raw and calibrated alike, the configuration's marker for a lost pixel is 0."""
    assert _lost_line_value(instrument) == 0.0


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_lost_lines_carry_a_null_marker_as_the_navigator_reads_it(instrument: str) -> None:
    """A marker_value of null is NaN to the navigator, and on every lost line."""
    overrides = {'noise': {'marker_value': None}}
    written = _lost_line_value(instrument, overrides)
    assert _same(written, _navigator_marker(instrument, overrides))


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


# Scenes whose navigated frame lands on each side of the classifier's gates.  The
# small body at the edge leaves the frame only once its offset is applied, so the
# preview is clean and the navigated frame blank; the lost lines put about 46%
# of the vgiss disc frame at the marker; an empty vgiss frame reads its sky
# residual, above the blank limit, while an empty calibrated Cassini frame's
# residual is below it; and a raw sky at a raised marker is blank only because
# the navigator fills marker pixels with 0 before it reads the peak.
_NAVIGATED_SCENES = {
    'coiss_nac-disc': _disc_scene('coiss_nac'),
    'coiss_calib_nac-disc': _disc_scene('coiss_calib_nac'),
    'vgiss-disc': _disc_scene('vgiss'),
    'vgiss-disc-lost-lines': _with_lost_lines(_disc_scene('vgiss'), 40.0),
    'calib-offset-leaves-frame': _body_scene('coiss_calib_nac', 6.0, (0.0, 20.0), center_u=118.0),
    'vgiss-offset': _body_scene('vgiss', 160.0, (3.0, -2.0), light=(18.0, 30.0)),
    'vgiss-empty': _empty_scene('vgiss'),
    'coiss_calib_nac-empty': _empty_scene('coiss_calib_nac'),
    'coiss_nac-empty': _empty_scene('coiss_nac'),
    'generic-sky-at-raised-marker': {
        **_empty_scene('generic'),
        'instrument_config': {'noise': {'marker_value': 20}},
    },
}


def _inst_config_of(params: dict[str, Any]) -> Any:
    """The resolved config block of a scene, with its instrument_config overrides."""
    return resolve_sim_inst_config(
        DEFAULT_CONFIG, params['instrument'], params.get('instrument_config')
    )


@pytest.mark.parametrize('name', sorted(_NAVIGATED_SCENES))
def test_reading_matches_the_orchestrator_on_the_navigated_frame(name: str) -> None:
    """The frame the navigator reads gets the class the orchestrator gives it."""
    params = _NAVIGATED_SCENES[name]
    img, _ = render_combined_model(params)
    reading = read_frame_as_navigator(img, _inst_config_of(params))
    assert reading.image_class == _orchestrator_class(params)


@pytest.mark.parametrize(
    ('name', 'image_class'),
    [
        ('vgiss-disc', 'clean'),
        ('vgiss-disc-lost-lines', 'mostly_missing_data'),
        ('calib-offset-leaves-frame', 'blank'),
        ('vgiss-offset', 'clean'),
        ('vgiss-empty', 'clean'),
        ('coiss_calib_nac-empty', 'blank'),
        ('coiss_nac-disc', 'clean'),
        ('generic-sky-at-raised-marker', 'blank'),
    ],
)
def test_navigated_scenes_land_where_the_comparison_needs_them(name: str, image_class: str) -> None:
    """Each comparison scene reaches the class it was chosen to exercise."""
    assert _orchestrator_class(_NAVIGATED_SCENES[name]) == image_class


def test_offset_moves_the_edge_scene_across_the_blank_limit() -> None:
    """The offset-free preview of the edge scene, its body still in the frame, is clean."""
    params = _NAVIGATED_SCENES['calib-offset-leaves-frame']
    img, _ = render_combined_model(params, ignore_offset=True)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_calib_nac')
    assert read_frame_as_navigator(img, inst_config).image_class == 'clean'


@pytest.mark.parametrize(
    ('image_class', 'phrase'),
    [
        ('mostly_missing_data', 'the navigator refuses it as missing data'),
        ('blank', 'the navigator classifies it blank'),
    ],
)
@pytest.mark.parametrize('name', sorted(_NAVIGATED_SCENES))
def test_summary_states_the_orchestrator_verdict(name: str, image_class: str, phrase: str) -> None:
    """The summary carries a verdict's phrase exactly when the orchestrator reaches it."""
    params = _NAVIGATED_SCENES[name]
    img, _ = render_combined_model(params)
    summary = frame_units_summary(img, _inst_config_of(params))
    assert (phrase in summary) == (_orchestrator_class(params) == image_class)


def test_empty_vgiss_summary_reads_the_sky_residual() -> None:
    """An empty vgiss frame reads its 1.5 DN sky residual and no marker pixel."""
    img, _ = render_combined_model(_empty_scene('vgiss'))
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss'))
    assert summary == 'I/F 0.01176 to 0.01176'


def _synthetic_inst_config(**thresholds: float) -> dict[str, Any]:
    """The generic raw block with ``image_quality_thresholds`` overrides."""
    inst_config = dict(resolve_sim_inst_config(DEFAULT_CONFIG, 'generic'))
    inst_config['image_quality_thresholds'] = {
        **inst_config['image_quality_thresholds'],
        **thresholds,
    }
    return inst_config


@pytest.mark.parametrize(
    ('saturated_rows', 'marker_rows', 'overrides'),
    [
        (85, 15, {'max_missing_frac_clean': 0.10}),
        (50, 40, {}),
        (85, 15, {}),
        (0, 100, {}),
        (0, 0, {}),
        (0, 31, {'max_missing_frac_clean': 0.50}),
    ],
    ids=[
        'overexposed-before-missing',
        'missing',
        'overexposed',
        'all-marker',
        'clean',
        'raised-limit',
    ],
)
def test_reading_follows_the_classifier_precedence(
    saturated_rows: int, marker_rows: int, overrides: dict[str, float]
) -> None:
    """Blank, then overexposed, then missing data, as the navigator's classifier decides."""
    inst_config = _synthetic_inst_config(**overrides)
    img = np.full((100, 10), 100.0)
    img[:saturated_rows, :] = 4095.0
    img[100 - marker_rows :, :] = 0.0
    settings = instrument_settings_from_obs(_InstConfigOnly(inst_config))
    expected = NavImageClassifier(thresholds=settings.thresholds).classify(img).image_class
    assert read_frame_as_navigator(img, inst_config).image_class == expected


def _boundary_frames(thresholds: Any, *, raw: bool) -> dict[str, np.ndarray]:
    """Frames that sit exactly on each of the classifier's limits."""
    lit = 2.0 * thresholds.blank_max_dn
    marker = thresholds.missing_data_marker_dn
    frames = {'peak-at-blank-limit': np.full((100, 10), thresholds.blank_max_dn)}
    at_missing = np.full((100, 10), lit)
    at_missing[: round(thresholds.max_missing_frac_clean * 100), :] = marker
    frames['marker-share-at-limit'] = at_missing
    if raw:
        at_saturation = np.full((100, 10), lit)
        rows = round(thresholds.max_saturation_frac_clean * 100)
        at_saturation[:rows, :] = thresholds.saturation_threshold_dn
        frames['saturated-share-at-limit'] = at_saturation
    return frames


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_reading_matches_the_classifier_exactly_at_each_limit(instrument: str) -> None:
    """On each limit's exact value the reading and the navigator's classifier agree."""
    inst_config = dict(resolve_sim_inst_config(DEFAULT_CONFIG, instrument))
    thresholds = instrument_settings_from_obs(_InstConfigOnly(inst_config)).thresholds
    classifier = NavImageClassifier(thresholds=thresholds)
    raw = inst_config.get('data_units', 'raw_dn') == 'raw_dn'
    frames = _boundary_frames(thresholds, raw=raw)
    mirror = {
        name: read_frame_as_navigator(img, inst_config).image_class for name, img in frames.items()
    }
    navigator = {name: classifier.classify(img).image_class for name, img in frames.items()}
    assert mirror == navigator


class _InstConfigOnly:
    """A stand-in observation carrying only an instrument config block."""

    def __init__(self, inst_config: dict[str, Any]) -> None:
        """Hold ``inst_config`` as the navigator's settings reader expects."""
        self.inst_config = inst_config


def test_raised_missing_limit_from_the_scene_is_honored() -> None:
    """A scene's max_missing_frac_clean override moves the summary's limit with the navigator's."""
    params = {
        **_NAVIGATED_SCENES['vgiss-disc-lost-lines'],
        'instrument_config': {'image_quality_thresholds': {'max_missing_frac_clean': 0.70}},
    }
    img, _ = render_combined_model(params)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss', params['instrument_config'])
    summary = frame_units_summary(img, inst_config)
    assert summary.endswith('of pixels')
    assert _orchestrator_class(params) == 'clean'


def test_lowered_missing_limit_from_the_scene_is_honored() -> None:
    """A lowered max_missing_frac_clean refuses a frame the default would accept."""
    params = {
        **_with_lost_lines(_NAVIGATED_SCENES['vgiss-offset'], 6.0),
        'instrument_config': {'image_quality_thresholds': {'max_missing_frac_clean': 0.02}},
    }
    img, _ = render_combined_model(params)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss', params['instrument_config'])
    assert '(over 2%: the navigator refuses it as missing data)' in frame_units_summary(
        img, inst_config
    )


@pytest.mark.parametrize(
    ('params', 'planted'),
    [
        ({}, False),
        ({'offset_v': 0.0, 'offset_u': 0.0}, False),
        ({'offset_v': None}, False),
        ({'offset_u': -0.5}, True),
        ({'offset_rotation_deg': 0.1}, True),
    ],
    ids=['none', 'zeros', 'null', 'u', 'roll'],
)
def test_planted_offset_is_any_non_zero_offset_or_roll(
    params: dict[str, Any], planted: bool
) -> None:
    """A scene plants an offset when any offset or roll key is non-zero."""
    assert has_planted_offset(params) is planted


@pytest.mark.parametrize('instrument', _ALL_INSTRUMENTS)
def test_units_name_matches_the_navigator_reading(instrument: str) -> None:
    """The units the editor names before a verdict arrives are the reading's units."""
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    reading = read_frame_as_navigator(np.ones((4, 4)), inst_config)
    assert frame_units_name(inst_config) == reading.units


@pytest.mark.parametrize(
    ('inst_config', 'units'),
    [({}, 'DN'), ({'data_units': 'raw_dn'}, 'DN'), ({'data_units': 'calibrated_if'}, 'I/F')],
    ids=['absent', 'raw_dn', 'calibrated_if'],
)
def test_units_name_follows_the_data_units(inst_config: dict[str, Any], units: str) -> None:
    """A raw_dn block (the default) is DN, a calibrated_if block is I/F."""
    assert frame_units_name(inst_config) == units


def test_vgiss_dark_sky_stays_out_of_the_marker_count() -> None:
    """A vgiss frame's dark sky sits above the 0 marker, so the summary names no marker."""
    img, _ = render_combined_model(_disc_scene('vgiss'), ignore_offset=True)
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss'))
    assert summary == f'I/F {float(img.min()):.4g} to {float(img.max()):.4g}'


def test_summary_counts_nan_markers_on_a_calibrated_frame() -> None:
    """Under a scene's NaN marker override the summary counts the NaN pixels."""
    img = np.full((10, 10), 0.25)
    img[0, :] = np.nan
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, 'coiss_calib_nac', {'noise': {'marker_value': 'NaN'}}
    )
    summary = frame_units_summary(img, inst_config)
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
