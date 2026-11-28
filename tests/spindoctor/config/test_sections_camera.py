"""Tests for the typed camera blocks (:mod:`spindoctor.config.sections_camera`)."""

import json
import math
import re
from typing import Any

import pytest
from tests.packaged_sections import (
    PACKAGED_CAMERAS,
    TEST_OVERRIDE_LABEL,
    NavConfigWith,
    calibrated_camera,
    camera_mapping,
    packaged_sections,
    raw_camera,
)

from spindoctor.config import Config
from spindoctor.config.sections_camera import CameraBlock
from spindoctor.schema import SchemaError, parse


def test_packaged_camera_blocks_parse() -> None:
    """Every packaged camera section is typed, the simulated camera's included."""
    sections = packaged_sections()
    assert tuple(dict.fromkeys(name for name, _ in PACKAGED_CAMERAS)) == sections.CAMERA_SECTIONS
    for section, camera in PACKAGED_CAMERAS:
        assert isinstance(sections.camera_block(section, camera), CameraBlock)


def test_an_override_that_replaces_every_camera_is_refused(
    nav_config_with: NavConfigWith,
) -> None:
    """An override that leaves a camera section no camera block is still checked."""
    with pytest.raises(SchemaError, match=re.escape(f'{TEST_OVERRIDE_LABEL}: cassini_iss.nac')):
        nav_config_with({'cassini_iss': {'nac': False, 'wac': False}})


def test_a_camera_name_on_a_one_camera_section_is_refused() -> None:
    """A section that is one camera's block refuses a camera name."""
    with pytest.raises(KeyError, match="galileo_ssi is one camera's block; it has no camera 'nac'"):
        packaged_sections().camera_block('galileo_ssi', 'nac')


def test_a_bad_camera_value_in_an_override_is_refused(nav_config_with: NavConfigWith) -> None:
    """A bad camera value in an override is refused at load, with its source and key path."""
    expected = f'{TEST_OVERRIDE_LABEL}: cassini_iss.wac.one_star_ellipse_fraction: '
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with({'cassini_iss': {'wac': {'one_star_ellipse_fraction': 1.5}}})


@pytest.mark.parametrize(
    ('camera', 'block_key', 'key', 'value', 'message'),
    [
        (raw_camera, 'noise', 'saturation_dn', None, 'raw_dn must set this key'),
        (raw_camera, 'image_quality_thresholds', 'blank_max_if', 1.0e-4, 'does not read'),
        (calibrated_camera, 'noise', 'read_noise_dn', 4.0, 'does not read'),
        (calibrated_camera, 'image_quality_thresholds', 'noisy_threshold_if', None, 'must set'),
    ],
    ids=['raw-needs-saturation', 'raw-refuses-if', 'calibrated-refuses-dn', 'calibrated-needs-if'],
)
def test_a_camera_sets_the_keys_its_units_read(
    camera: Any, block_key: str, key: str, value: float | None, message: str
) -> None:
    """A camera sets every key its units read and leaves out the other units' keys."""
    raw = camera_mapping(camera())
    if value is None:
        del raw[block_key][key]
    else:
        raw[block_key][key] = value
    with pytest.raises(SchemaError, match=rf'camera: {block_key}\.{key}: .*{message}'):
        parse(CameraBlock, raw, source='camera')


def test_the_fallback_combination_has_an_offset() -> None:
    """The magnitude offset's fallback combination must be in its table."""
    raw = camera_mapping(raw_camera())
    raw['mag_offset']['fallback_combo'] = 'NOT_A_COMBO'
    with pytest.raises(
        SchemaError, match=r"mag_offset\.fallback_combo: 'NOT_A_COMBO' has no entry"
    ):
        parse(CameraBlock, raw, source='camera')


def test_the_bandpass_heavy_blur_is_the_wider() -> None:
    """A bandpass filter's heavy blur must be wider than its light blur."""
    raw = camera_mapping(raw_camera())
    raw['source_image_filter']['lo_sigma_px'] = 0.5
    with pytest.raises(SchemaError, match=r'source_image_filter\.lo_sigma_px: .* greater than'):
        parse(CameraBlock, raw, source='camera')


def test_a_block_read_through_json_parses_to_the_same_camera() -> None:
    """A block whose integer keys became strings, as JSON writes them, reads the same."""
    camera = raw_camera()
    through_json = json.loads(json.dumps(camera_mapping(camera)))
    assert parse(CameraBlock, through_json, source='camera') == camera


@pytest.mark.parametrize('marker', ['NaN', math.nan], ids=['string', 'float'])
def test_a_nan_marker_reads_as_nan(marker: str | float) -> None:
    """A marker written as NaN, or a float NaN in a mapping, reads as NaN."""
    raw = camera_mapping(raw_camera())
    raw['noise']['marker_value'] = marker
    assert math.isnan(parse(CameraBlock, raw, source='camera').noise.marker())


def test_the_margin_follows_the_image_height() -> None:
    """A margin table gives each image height its own margin."""
    camera = raw_camera(extfov_margin_vu={512: (25, 50), 1024: (50, 140)})
    assert camera.extfov_margin_for(512) == (25, 50)


def test_a_height_the_margin_table_leaves_out_is_refused() -> None:
    """A height the margin table has no entry for is refused, listing the table's."""
    camera = raw_camera(extfov_margin_vu={512: (25, 50), 1024: (50, 140)})
    with pytest.raises(ValueError, match=r'no entry for an image 256 pixels high.*\[512, 1024\]'):
        camera.extfov_margin_for(256)


def test_a_section_that_holds_no_camera_has_no_camera_names() -> None:
    """Asking a section that holds no camera block for its cameras is refused."""
    with pytest.raises(KeyError, match="'offset' is not a camera section"):
        Config().camera_names('offset')


def test_the_star_photometry_mode_tables_give_the_same_modes() -> None:
    """Mode tables that give different modes are refused."""
    raw = camera_mapping(raw_camera())
    raw['star_photometry'] = {
        'flux_unit': 'electrons',
        'flux_per_s_vmag0': 1.0e8,
        'gain': {'LOW': 20.0},
        'min_detectable_dn': {'FULL': 100.0},
        'psf_sigma': {'FULL': 0.9, 'SUMMED': 0.5},
        'search_depth_mag': 1.0,
    }
    with pytest.raises(
        SchemaError,
        match=r"camera: star_photometry\.psf_sigma: gives the modes \['FULL', 'SUMMED'\]",
    ):
        parse(CameraBlock, raw, source='camera')


def test_a_detection_limit_table_by_gain_state_gives_every_gain_state() -> None:
    """A mode's table by gain state that leaves out a gain state of ``gain`` is refused."""
    raw = camera_mapping(raw_camera())
    raw['star_photometry'] = {
        'flux_unit': 'electrons',
        'flux_per_s_vmag0': 1.0e8,
        'gain': {'LOW': 20.0, 'HIGH': 5.0},
        'min_detectable_dn': {'FULL': {'LOW': 100.0}},
        'psf_sigma': {'FULL': 0.9},
        'search_depth_mag': 1.0,
    }
    with pytest.raises(
        SchemaError,
        match=(
            r'camera: star_photometry\.min_detectable_dn\.FULL: gives the gain states '
            r"\['LOW'\] for the mode 'FULL', but gain gives \['HIGH', 'LOW'\]"
        ),
    ):
        parse(CameraBlock, raw, source='camera')
