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
from spindoctor.config.sections_camera import CameraBlock, as_camera_block
from spindoctor.schema import SchemaError


def test_packaged_camera_blocks_parse() -> None:
    """Every packaged real camera section is typed, and each camera parses."""
    sections = packaged_sections()
    assert tuple(dict.fromkeys(name for name, _ in PACKAGED_CAMERAS)) == sections.CAMERA_SECTIONS
    for section, camera in PACKAGED_CAMERAS:
        assert isinstance(sections.camera_block(section, camera), CameraBlock)


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
        as_camera_block(raw, source='camera')


def test_the_fallback_combination_has_an_offset() -> None:
    """The magnitude offset's fallback combination must be in its table."""
    raw = camera_mapping(raw_camera())
    raw['mag_offset']['fallback_combo'] = 'NOT_A_COMBO'
    with pytest.raises(
        SchemaError, match=r"mag_offset\.fallback_combo: 'NOT_A_COMBO' has no entry"
    ):
        as_camera_block(raw, source='camera')


def test_the_bandpass_heavy_blur_is_the_wider() -> None:
    """A bandpass filter's heavy blur must be wider than its light blur."""
    raw = camera_mapping(raw_camera())
    raw['source_image_filter']['lo_sigma_px'] = 0.5
    with pytest.raises(SchemaError, match=r'source_image_filter\.lo_sigma_px: .* greater than'):
        as_camera_block(raw, source='camera')


def test_a_block_read_through_json_parses_to_the_same_camera() -> None:
    """A block whose integer keys became strings, as JSON writes them, reads the same."""
    camera = raw_camera()
    through_json = json.loads(json.dumps(camera_mapping(camera)))
    assert as_camera_block(through_json, source='camera') == camera


@pytest.mark.parametrize('marker', ['NaN', math.nan], ids=['string', 'float'])
def test_a_nan_marker_reads_as_nan(marker: str | float) -> None:
    """A marker written as NaN, or a float NaN in a mapping, reads as NaN."""
    raw = camera_mapping(raw_camera())
    raw['noise']['marker_value'] = marker
    assert math.isnan(as_camera_block(raw, source='camera').noise.marker())


def test_the_margin_follows_the_image_height() -> None:
    """A margin table gives each image height its own margin."""
    camera = raw_camera(extfov_margin_vu={512: (25, 50), 1024: (50, 140)})
    assert camera.extfov_margin_for(512) == (25, 50)


def test_a_height_the_margin_table_leaves_out_is_refused() -> None:
    """A height the margin table has no entry for is refused, listing the table's."""
    camera = raw_camera(extfov_margin_vu={512: (25, 50), 1024: (50, 140)})
    with pytest.raises(ValueError, match=r'no entry for an image 256 pixels high.*\[512, 1024\]'):
        camera.extfov_margin_for(256)


def test_an_untyped_camera_section_is_parsed_when_asked_for() -> None:
    """The simulator's camera section is left untyped at load and parsed on request."""
    config = Config()
    assert 'sim' not in config._typed_sections().CAMERA_SECTIONS
    assert isinstance(config.camera_block('sim'), CameraBlock)


def test_the_star_photometry_mode_tables_give_the_same_modes() -> None:
    """A mode the star width table gives and the faintest-signal table leaves out is refused."""
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
        as_camera_block(raw, source='camera')
