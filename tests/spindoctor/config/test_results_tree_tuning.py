"""Tests for the typed ``results_tree`` configuration section.

The section's type is :class:`spindoctor.nav_records.TreeTuning`, whose own rules are
tested beside it in ``tests/spindoctor/nav_records/test_tuning.py``.  The schema core
checks each key's type and range, and is tested in its own suite.  What is tested here
is that the packaged section parses, and that the loader every program passes through
refuses a bad override, naming the file and the key, before any program has written
anything.
"""

import argparse
import re
from pathlib import Path

import pytest

from spindoctor.config import Config, load_default_and_user_config
from spindoctor.nav_records import TreeTuning
from spindoctor.schema import SchemaError


def test_packaged_results_tree_section_parses() -> None:
    """The packaged section parses into the tuning every tree reader is handed."""
    assert isinstance(Config().results_tree, TreeTuning)


def test_the_loader_refuses_a_section_no_pass_can_run_at(tmp_path: Path) -> None:
    """A round smaller than its pool is refused at load, and the error gives its key."""
    override = tmp_path / 'override.yaml'
    override.write_text('results_tree:\n  retrieve_threads: 64\n  retrieve_batch_size: 8\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    expected = f'{override}: results_tree.retrieve_batch_size: must be at least retrieve_threads'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        load_default_and_user_config(arguments, Config())
