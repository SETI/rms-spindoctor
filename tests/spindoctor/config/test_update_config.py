"""Tests for how :meth:`Config.update_config` merges a user override.

A user override of one nested key must combine key-by-key with the packaged
defaults so sibling keys under the same sub-block survive (CODE-CFG-1).  The merge
itself is :func:`spindoctor.schema.layers.merge`, tested beside it; what is tested
here is what the loader adds to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.nav_technique import BodyBlobNav, RingEdgeNav
from spindoctor.nav_technique.confidence_config import shipped_techniques


def test_update_config_preserves_sibling_defaults(tmp_path: Path) -> None:
    """A user override of one leaf keeps unrelated sibling defaults."""
    config = Config()
    config.read_config()
    agreement_sigma = config.orchestrator.ensemble.agreement_sigma

    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('orchestrator:\n  ensemble:\n    min_confidence: 0.1\n', encoding='utf-8')
    config.update_config(user_path)

    assert config.orchestrator.ensemble.min_confidence == 0.1
    assert config.orchestrator.ensemble.agreement_sigma == agreement_sigma


def test_a_null_in_an_override_is_refused_naming_the_file_and_key(tmp_path: Path) -> None:
    """A null in a user override is refused at load, with its file and key path."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('orchestrator:\n  ensemble:\n    min_confidence: null\n', encoding='utf-8')
    config = Config()
    expected = f'{user_path}: orchestrator.ensemble.min_confidence: null is refused'
    with pytest.raises(ValueError, match=re.escape(expected)):
        config.update_config(user_path)


def test_a_section_given_as_a_scalar_is_refused(tmp_path: Path) -> None:
    """An override cannot replace a whole section with a value that is not a mapping."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('orchestrator: 3\n', encoding='utf-8')
    config = Config()
    expected = f'{user_path}: orchestrator: expected a mapping'
    with pytest.raises(ValueError, match=re.escape(expected)):
        config.update_config(user_path)


def test_an_override_refused_by_validation_changes_nothing(tmp_path: Path) -> None:
    """A refused merged result keeps the config, its overrides, and technique tuning."""
    config = Config()
    config.read_config()
    tuning = RingEdgeNav.tuning
    refused = tmp_path / 'refused.yaml'
    refused.write_text(
        'techniques:\n  ring_edge:\n    tuning:\n'
        '      rotation_at_edge_fraction: 0.5\n'
        'extra_camera:\n  data_units: raw_dn\n',
        encoding='utf-8',
    )
    valid = tmp_path / 'valid.yaml'
    valid.write_text('orchestrator:\n  ensemble:\n    min_confidence: 0.1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='extra_camera sets no one_star_ellipse_fraction'):
        config.update_config(refused)
    assert 'extra_camera' not in config._config_dict
    assert not config.override_paths
    assert RingEdgeNav.tuning is tuning
    config.update_config(valid)
    assert config.override_paths == (str(valid),)


def test_category_returns_cached_instance() -> None:
    """CODE-CFG-3: category() returns the same cached AttrDict across calls."""
    config = Config()
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 2}}}
    config._update_attrdicts()
    first = config.category('bodies')
    second = config.category('bodies')
    assert first is second


def test_category_cache_rebuilt_on_reload() -> None:
    """CODE-CFG-3: the category cache is dropped when the config is reloaded."""
    config = Config()
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 2}}}
    config._update_attrdicts()
    before = config.category('bodies')
    config._config_dict = {'bodies': {'MIMAS': {'albedo': 9}}}
    config._update_attrdicts()
    after = config.category('bodies')
    assert after is not before
    assert after['MIMAS']['albedo'] == 9


def test_reread_no_path_drops_removed_keys() -> None:
    """CODE-CFG-2: a reread starts again from the packaged files (no stale keys)."""
    config = Config()
    # Simulate a prior load that contains a key the bundled YAML does not.
    config._config_dict = {'general': {'stale_only_key': 123}}
    config._update_attrdicts()
    # Reread from the packaged config_files glob.
    config.read_config(reread=True)
    assert 'stale_only_key' not in config._config_dict.get('general', {})


def _apply_technique_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Apply a ``techniques`` override, restoring every technique class afterwards.

    Parameters:
        tmp_path: A directory for the override file.
        monkeypatch: Restores the technique class attributes when the test ends.
        text: The override file's contents.
    """
    for technique in shipped_techniques():
        monkeypatch.setattr(technique, 'tuning', technique.tuning)
        monkeypatch.setattr(technique, 'confidence_spec', technique.confidence_spec)
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(text, encoding='utf-8')
    Config().update_config(user_path)


def test_a_typed_tuning_override_reaches_its_technique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted override of a typed technique's tuning is what the class then holds."""
    _apply_technique_override(
        tmp_path,
        monkeypatch,
        'techniques:\n  ring_edge:\n    tuning:\n      gradient_ridge_refine: false\n',
    )
    assert RingEdgeNav.tuning.gradient_ridge_refine is False


def test_an_untyped_tuning_override_reaches_its_technique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted override of an untyped technique's tuning is in the class's mapping."""
    _apply_technique_override(
        tmp_path,
        monkeypatch,
        'techniques:\n  BodyBlobNav:\n    tuning:\n      at_edge_tolerance_px: 0.123\n',
    )
    assert BodyBlobNav.tuning['at_edge_tolerance_px'] == 0.123
