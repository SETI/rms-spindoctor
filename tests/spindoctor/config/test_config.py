"""Tests for the load-time validation in :class:`spindoctor.config.Config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spindoctor.config import Config


@pytest.mark.parametrize(
    'value', ['.nan', '-1.0', '2.0', 'abc'], ids=['nan', 'below_zero', 'above_one', 'text']
)
def test_malformed_blob_max_overflow_fraction_is_refused(tmp_path: Path, value: str) -> None:
    """An overflow limit that is not a finite fraction fails the load, naming the key."""
    override = tmp_path / 'override.yaml'
    override.write_text(f'bodies:\n  blob_max_overflow_fraction: {value}\n', encoding='utf-8')
    config = Config()
    with pytest.raises(ValueError, match=r'bodies\.blob_max_overflow_fraction must be finite'):
        config.update_config(override)


@pytest.mark.parametrize(
    ('section', 'path'),
    [
        ('extra_camera:\n  data_units: raw_dn\n', 'extra_camera'),
        ('extra_mission:\n  cam:\n    data_units: raw_dn\n', 'extra_mission.cam'),
    ],
    ids=['single_camera_section', 'camera_inside_a_section'],
)
def test_a_camera_without_a_lone_star_ellipse_is_refused(
    tmp_path: Path, section: str, path: str
) -> None:
    """A camera block that sets no lone-star ellipse fraction fails the load, naming it."""
    override = tmp_path / 'override.yaml'
    override.write_text(section, encoding='utf-8')
    config = Config()
    with pytest.raises(ValueError, match=rf'^{path} sets no one_star_ellipse_fraction$'):
        config.update_config(override)


@pytest.mark.parametrize(
    ('value', 'refused'),
    [('0.0', True), ('1.5', True), ('.nan', True), ('abc', True), ('1.0', False)],
    ids=['zero', 'above_one', 'nan', 'text', 'one'],
)
def test_a_lone_star_ellipse_outside_zero_to_one_is_refused(
    tmp_path: Path, value: str, refused: bool
) -> None:
    """A camera's lone-star ellipse fraction must lie in (0, 1]; 1 itself is accepted."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        f'cassini_iss:\n  wac:\n    one_star_ellipse_fraction: {value}\n', encoding='utf-8'
    )
    config = Config()
    if refused:
        with pytest.raises(
            ValueError, match=r'cassini_iss\.wac\.one_star_ellipse_fraction must lie in \(0, 1\]'
        ):
            config.update_config(override)
    else:
        config.update_config(override)
        assert config.category('cassini_iss')['wac']['one_star_ellipse_fraction'] == 1.0
