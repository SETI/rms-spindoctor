"""Tests for how the ``results_tree`` configuration section becomes a tuning.

The library's own rules about a tuning are tested beside it, in
``tests/spindoctor/nav_records/test_tuning.py``.  What is tested here is the one
door a program reads the section through,
:func:`spindoctor.config.get_results_tree_tuning`, and the loader every program
passes through, which validates the section before any of them has written
anything.
"""

import argparse
import re
from dataclasses import fields
from pathlib import Path

import pytest

from spindoctor.config import Config, get_results_tree_tuning, load_default_and_user_config
from spindoctor.nav_records import TreeTuning


def _config(tmp_path: Path, body: str = '') -> Config:
    """Build the shipped configuration plus one override file.

    Parameters:
        tmp_path: Directory the override file is written into.
        body: YAML text applied as an override; empty for the defaults alone.

    Returns:
        The configuration.
    """
    config = Config()
    config.read_config()
    if body:
        override = tmp_path / 'override.yaml'
        override.write_text(body)
        config.update_config(str(override))
    return config


def test_the_shipped_section_names_every_setting(tmp_path: Path) -> None:
    """A setting the shipped file omits is one nobody knows they can change."""
    shipped = set(_config(tmp_path).results_tree)
    assert shipped == {field.name for field in fields(TreeTuning)}


def test_the_shipped_section_is_what_the_defaults_say(tmp_path: Path) -> None:
    """The file and the dataclass agree, so neither is quietly the real one."""
    assert get_results_tree_tuning(_config(tmp_path)) == TreeTuning()


def test_an_override_of_one_setting_leaves_the_rest_at_their_defaults(tmp_path: Path) -> None:
    """An operator changing one number does not have to restate the rest."""
    tuning = get_results_tree_tuning(_config(tmp_path, 'results_tree:\n  walk_threads: 4\n'))
    assert tuning.walk_threads == 4
    assert tuning.retrieve_threads == TreeTuning().retrieve_threads


def test_a_null_setting_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    """A key written with no value is a mistake to report, not a default to fall back on."""
    config = _config(tmp_path, 'results_tree:\n  walk_threads: null\n')
    with pytest.raises(ValueError, match=re.escape('results_tree: walk_threads')):
        get_results_tree_tuning(config)


def test_a_setting_that_does_not_exist_is_refused_by_name(tmp_path: Path) -> None:
    """A misspelled key would otherwise tune nothing and say nothing."""
    config = _config(tmp_path, 'results_tree:\n  walk_thread: 4\n')
    with pytest.raises(ValueError, match=re.escape("no setting called 'walk_thread'")):
        get_results_tree_tuning(config)


def test_a_refusal_lists_the_settings_that_do_exist(tmp_path: Path) -> None:
    """The operator reading the failure is told what to write instead."""
    config = _config(tmp_path, 'results_tree:\n  walk_thread: 4\n')
    with pytest.raises(ValueError, match='walk_threads, walk_directories_at_once'):
        get_results_tree_tuning(config)


def test_a_value_no_pass_can_run_at_is_refused_naming_the_section(tmp_path: Path) -> None:
    """The library names the setting; the door a program reads it through adds the section."""
    config = _config(tmp_path, 'results_tree:\n  retrieve_batch_size: 8\n')
    with pytest.raises(ValueError, match=re.escape('results_tree: retrieve_batch_size')):
        get_results_tree_tuning(config)


def test_the_loader_refuses_a_section_no_pass_can_run_at(tmp_path: Path) -> None:
    """Every program loads its configuration before it writes anything, so this is startup."""
    override = tmp_path / 'override.yaml'
    override.write_text('results_tree:\n  walk_threads: 0\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    with pytest.raises(ValueError, match=re.escape('results_tree: walk_threads')):
        load_default_and_user_config(arguments, Config())


def test_the_loader_accepts_the_shipped_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defaults are a configuration every program starts under."""
    monkeypatch.chdir(tmp_path)
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert get_results_tree_tuning(config) == TreeTuning()
