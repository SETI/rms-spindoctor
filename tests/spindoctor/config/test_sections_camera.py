"""Tests for the typed camera blocks (:mod:`spindoctor.config.sections_camera`)."""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from tests.camera_sections import (
    PACKAGED_CAMERAS,
    calibrated_camera,
    camera_mapping,
    raw_camera,
)
from tests.packaged_sections import packaged_sections

from spindoctor.config import Config
from spindoctor.config.sections_camera import CameraBlock
from spindoctor.schema import SchemaError, parse


def test_packaged_camera_blocks_parse() -> None:
    """Every packaged camera section is typed, the simulated camera's included."""
    sections = packaged_sections()
    assert tuple(dict.fromkeys(name for name, _ in PACKAGED_CAMERAS)) == sections.CAMERA_SECTIONS
    for section, camera in PACKAGED_CAMERAS:
        assert isinstance(sections.camera_block(section, camera), CameraBlock)


def test_an_override_that_replaces_every_camera_is_refused(tmp_path: Path) -> None:
    """An override that leaves a camera section no camera block is still checked."""
    config = Config()
    override = tmp_path / 'override.yaml'
    override.write_text('cassini_iss:\n  nac: false\n  wac: false\n')
    with pytest.raises(SchemaError, match=re.escape(f'{override}: cassini_iss.nac')):
        config.update_config(override)


def test_a_camera_name_on_a_one_camera_section_is_refused() -> None:
    """A section that is one camera's block refuses a camera name."""
    with pytest.raises(KeyError, match="galileo_ssi is one camera's block; it has no camera 'nac'"):
        packaged_sections().camera_block('galileo_ssi', 'nac')


def test_a_bad_camera_value_in_an_override_is_refused(tmp_path: Path) -> None:
    """A bad camera value in an override is refused at load, with its file and key path."""
    config = Config()
    before = config.camera_block('cassini_iss', 'wac')
    override = tmp_path / 'override.yaml'
    override.write_text('cassini_iss:\n  wac:\n    one_star_ellipse_fraction: 1.5\n')
    expected = f'{override}: cassini_iss.wac.one_star_ellipse_fraction: '
    with pytest.raises(SchemaError, match=re.escape(expected)):
        config.update_config(override)
    assert config.camera_block('cassini_iss', 'wac') is before


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
