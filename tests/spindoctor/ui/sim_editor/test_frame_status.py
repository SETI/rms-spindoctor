"""The scene editor's status text for a frame of each sim instrument.

:func:`~spindoctor.ui.sim_editor.frame_status.read_frame_as_navigator` reads the
frame the navigator reads (planted offset applied) to the verdict the navigator's
classifier reaches, blank before missing data.
:func:`~spindoctor.ui.sim_editor.frame_status.frame_units_summary` states that
verdict.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene

# The editor package imports PyQt6, which needs the system's Qt libraries.
pytest.importorskip('PyQt6.QtWidgets', exc_type=ImportError)

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.config.sections_camera import CameraBlock
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.image_classifier import NavImageClassifier
from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instruments.catalog import sim_instrument_names
from spindoctor.sim.instruments.navigator_link import (
    resolve_sim_inst_config,
    scene_camera_block,
)
from spindoctor.sim.render.driver import render_scene
from spindoctor.ui.sim_editor.frame_status import (
    frame_units_name,
    frame_units_summary,
    has_planted_offset,
    read_frame_as_navigator,
)

_ALL_INSTRUMENTS = [*sim_instrument_names(), 'generic']


def _body_scene(
    instrument: str,
    semi_px: float,
    offset_vu: tuple[float, float],
    *,
    center_u: float = 64.0,
    light: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """A 128 x 128 frame with one Rhea-like body and a planted offset."""
    return frame_scene(
        instrument=instrument,
        size=128,
        seed=42,
        offset_v=offset_vu[0],
        offset_u=offset_vu[1],
        bodies=[
            disc_body(
                64.0,
                center_u,
                semi_px,
                name='RHEA',
                illumination_angle=light[0],
                phase_angle=light[1],
            )
        ],
    )


def _with_lost_lines(params: dict[str, Any], expected_events: float) -> dict[str, Any]:
    """``params`` with a ``missing_lines`` mode planting about that many lost lines."""
    return {**params, 'artifacts': {'missing_lines': {'expected_events': expected_events}}}


def _empty_scene(instrument: str) -> dict[str, Any]:
    """A 64 x 64 frame with nothing in it."""
    return frame_scene(instrument=instrument)


def _orchestrator_class(params: dict[str, Any]) -> str:
    """The class the navigator's orchestrator gives the frame it renders from ``params``."""
    obs = ObsSim.from_file('/tmp/sim_unit_classes.json', sim_params=params)
    return NavOrchestrator([]).prepare(obs).image_classifier.image_class


def _disc_scene(instrument: str) -> dict[str, Any]:
    """A 128 x 128 frame whose lit disc covers about a third of the pixels."""
    return frame_scene(
        instrument=instrument,
        size=128,
        seed=42,
        bodies=[
            disc_body(64.0, 64.0, 43.0, name='RHEA', illumination_angle=18.0, phase_angle=30.0)
        ],
    )


@pytest.mark.parametrize(
    ('instrument', 'units'),
    [('coiss_nac', 'DN'), ('gossi', 'DN'), ('coiss_calib_nac', 'I/F'), ('vgiss_1_nac', 'I/F')],
)
def test_summary_names_the_unit_class(instrument: str, units: str) -> None:
    """The summary opens with the instrument's units."""
    img, _ = render_scene(_disc_scene(instrument), ignore_offset=True)
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
    'vgiss-disc': _disc_scene('vgiss_1_nac'),
    'vgiss-disc-lost-lines': _with_lost_lines(_disc_scene('vgiss_1_nac'), 40.0),
    'calib-offset-leaves-frame': _body_scene('coiss_calib_nac', 3.0, (0.0, 20.0), center_u=118.0),
    'vgiss-offset': _body_scene('vgiss_1_nac', 80.0, (3.0, -2.0), light=(18.0, 30.0)),
    'vgiss-empty': _empty_scene('vgiss_1_nac'),
    'coiss_calib_nac-empty': _empty_scene('coiss_calib_nac'),
    'coiss_nac-empty': _empty_scene('coiss_nac'),
    'generic-sky-at-raised-marker': {
        **_empty_scene('generic'),
        'instrument_config': {'noise': {'marker_value': 20}},
    },
}


def _inst_config_of(params: dict[str, Any]) -> CameraBlock:
    """The navigator block of a scene, with its instrument_config merged over it."""
    return scene_camera_block(DEFAULT_CONFIG, params)


@pytest.mark.parametrize('name', sorted(_NAVIGATED_SCENES))
def test_reading_matches_the_orchestrator_on_the_navigated_frame(name: str) -> None:
    """The frame the navigator reads gets the class the orchestrator gives it."""
    params = _NAVIGATED_SCENES[name]
    img, _ = render_scene(params)
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
    img, _ = render_scene(params, ignore_offset=True)
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
    img, _ = render_scene(params)
    summary = frame_units_summary(img, _inst_config_of(params))
    assert (phrase in summary) == (_orchestrator_class(params) == image_class)


def test_empty_vgiss_summary_reads_the_sky_residual() -> None:
    """An empty vgiss frame reads its 1.5 DN sky residual and no marker pixel."""
    img, _ = render_scene(_empty_scene('vgiss_1_nac'))
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss_1_nac'))
    assert summary == 'I/F 0.01176 to 0.01176'


def _synthetic_inst_config(**thresholds: float) -> CameraBlock:
    """The generic raw block with ``image_quality_thresholds`` overrides."""
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, 'generic')
    limits = dataclasses.replace(inst_config.image_quality_thresholds, **thresholds)
    return dataclasses.replace(inst_config, image_quality_thresholds=limits)


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
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    thresholds = instrument_settings_from_obs(_InstConfigOnly(inst_config)).thresholds
    classifier = NavImageClassifier(thresholds=thresholds)
    raw = inst_config.data_units == 'raw_dn'
    frames = _boundary_frames(thresholds, raw=raw)
    mirror = {
        name: read_frame_as_navigator(img, inst_config).image_class for name, img in frames.items()
    }
    navigator = {name: classifier.classify(img).image_class for name, img in frames.items()}
    assert mirror == navigator


class _InstConfigOnly:
    """A stand-in observation carrying only a navigator block."""

    def __init__(self, inst_config: CameraBlock) -> None:
        """Hold ``inst_config`` as the navigator's settings reader expects."""
        self.inst_config = inst_config


def test_lowered_read_line_limit_from_the_scene_is_honored() -> None:
    """A scene's min_read_line_frac override moves the summary's limit with the navigator's.

    The lost lines are whole unread lines, so the read-out limit, and not
    max_missing_frac_clean, is what lets the frame through.
    """
    params = {
        **_NAVIGATED_SCENES['vgiss-disc-lost-lines'],
        'instrument_config': {'image_quality_thresholds': {'min_read_line_frac': 0.30}},
    }
    img, _ = render_scene(params)
    inst_config = scene_camera_block(DEFAULT_CONFIG, params)
    summary = frame_units_summary(img, inst_config)
    assert summary.endswith('of pixels')
    assert _orchestrator_class(params) == 'clean'


def test_raised_read_line_limit_from_the_scene_is_honored() -> None:
    """A raised min_read_line_frac refuses a frame with lost lines the default would accept.

    The lost lines are whole unread lines, so they count against the share of lines read
    out and not against max_missing_frac_clean.
    """
    params = {
        **_with_lost_lines(_NAVIGATED_SCENES['vgiss-offset'], 6.0),
        'instrument_config': {'image_quality_thresholds': {'min_read_line_frac': 1.0}},
    }
    img, _ = render_scene(params)
    inst_config = scene_camera_block(DEFAULT_CONFIG, params)
    expected = '(under 100% of lines read out: the navigator refuses it as missing data)'
    assert expected in frame_units_summary(img, inst_config)


@pytest.mark.parametrize(
    ('params', 'planted'),
    [
        ({}, False),
        ({'offset_v': 0.0, 'offset_u': 0.0}, False),
        ({'offset_u': -0.5}, True),
        ({'offset_rotation_deg': 0.1}, True),
    ],
    ids=['none', 'zeros', 'u', 'roll'],
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
    ('instrument', 'units'),
    [('coiss_nac', 'DN'), ('coiss_calib_nac', 'I/F')],
    ids=['raw_dn', 'calibrated_if'],
)
def test_units_name_follows_the_data_units(instrument: str, units: str) -> None:
    """A raw_dn block is DN, and a calibrated_if block is I/F."""
    assert frame_units_name(resolve_sim_inst_config(DEFAULT_CONFIG, instrument)) == units


def test_vgiss_dark_sky_stays_out_of_the_marker_count() -> None:
    """A vgiss frame's dark sky sits above the 0 marker, so the summary names no marker."""
    img, _ = render_scene(_disc_scene('vgiss_1_nac'), ignore_offset=True)
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss_1_nac'))
    assert summary == f'I/F {float(img.min()):.4g} to {float(img.max()):.4g}'


def test_summary_counts_nan_markers_on_a_calibrated_frame() -> None:
    """Under a scene's NaN marker override the summary counts the NaN pixels."""
    img = np.full((10, 10), 0.25)
    img[0, :] = np.nan
    overlay = {'noise': {'marker_value': 'NaN'}}
    scene = {'instrument': 'coiss_calib_nac', 'instrument_config': overlay}
    inst_config = scene_camera_block(DEFAULT_CONFIG, scene)
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
    summary = frame_units_summary(img, resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss_1_nac'))
    assert ('refuses' in summary) == warns
