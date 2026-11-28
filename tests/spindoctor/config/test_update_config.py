"""Tests for how :func:`load_nav_config` and :meth:`Config.update_config` load overrides.

A user override of one nested key must combine key-by-key with the packaged
defaults so sibling keys under the same sub-block survive (CODE-CFG-1).  The merge
itself is :func:`spindoctor.schema.layers.merge`, tested beside it; what is tested
here is what the loader adds to it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spindoctor.config import Config, load_nav_config
from spindoctor.nav_technique import RingEdgeNav
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


def test_an_override_refused_by_a_typed_section_changes_nothing(tmp_path: Path) -> None:
    """A typed section refusing an override keeps the config, its overrides, and tuning."""
    config = Config()
    config.read_config()
    bodies = config.bodies
    tuning = RingEdgeNav.tuning
    refused = tmp_path / 'refused.yaml'
    refused.write_text(
        'techniques:\n  ring_edge:\n    tuning:\n'
        '      rotation_at_edge_fraction: 0.5\n'
        'bodies:\n  not_a_bodies_key: 1\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='not_a_bodies_key'):
        config.update_config(refused)
    assert config.bodies is bodies
    assert not config.override_paths
    assert RingEdgeNav.tuning is tuning


def test_an_unknown_section_is_refused_naming_the_file(tmp_path: Path) -> None:
    """A top-level section the navigator does not declare is refused at load."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('fuzzy_satellites:\n  SATURN: [PAN]\n', encoding='utf-8')
    expected = f"{user_path}: unknown keys: ['fuzzy_satellites']"
    with pytest.raises(ValueError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_reread_drops_the_files_merged_since(tmp_path: Path) -> None:
    """A reread starts again from the packaged files, so no override survives it."""
    config = Config()
    packaged = config.orchestrator.ensemble.min_confidence
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('orchestrator:\n  ensemble:\n    min_confidence: 0.1\n', encoding='utf-8')
    config.update_config(user_path)
    config.read_config(reread=True)
    assert config.orchestrator.ensemble.min_confidence == packaged
    assert config.override_paths == ()


def test_loading_a_configuration_leaves_the_techniques_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading builds a new object and changes nothing else, a technique's tuning included."""
    override = object()
    monkeypatch.setattr(RingEdgeNav, 'tuning', override)
    first = load_nav_config()
    assert load_nav_config() is not first
    assert RingEdgeNav.tuning is override


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


def test_a_tuning_override_reaches_its_technique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted override of a technique's tuning is what the class then holds."""
    _apply_technique_override(
        tmp_path,
        monkeypatch,
        'techniques:\n  ring_edge:\n    tuning:\n      gradient_ridge_refine: false\n',
    )
    assert RingEdgeNav.tuning.gradient_ridge_refine is False
