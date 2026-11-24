"""Tests for the typed ``techniques`` configuration section.

The schema core checks each key's type and range, and is tested in its own suite.
What is tested here is that the packaged section parses, how its blocks are named, the
rule that a block names only attributes its technique declares, and that the loader
refuses a block by a name the section does not use.
"""

import re
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.nav_technique import BodyLimbNav
from spindoctor.nav_technique.confidence_config import shipped_techniques
from spindoctor.schema import SchemaError


def test_packaged_techniques_section_parses() -> None:
    """Every shipped technique has a block, and a typed technique's tuning has its type."""
    section = Config().techniques
    for technique in shipped_techniques():
        tuning = section.block(technique).technique_tuning()
        expected = dict if technique.Tuning is None else technique.Tuning
        assert isinstance(tuning, expected), technique.name


@pytest.mark.parametrize(
    ('text', 'path'),
    [
        ('    terms:\n      - feature: nope\n        alpha: 1.0\n', 'terms[0].feature'),
        ('    hard_zero_if:\n      nope: true\n', 'hard_zero_if.nope'),
    ],
    ids=['term', 'hard-zero'],
)
def test_an_undeclared_attribute_is_refused(tmp_path: Path, text: str, path: str) -> None:
    """A block that names an attribute its technique does not declare is refused at load.

    Parameters:
        tmp_path: A directory for the override file.
        text: The ``body_limb`` block's lines in the override.
        path: The key path the message gives inside the block.
    """
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(f'techniques:\n  body_limb:\n{text}', encoding='utf-8')
    expected = f'{user_path}: techniques.body_limb.{path}: BodyLimbNav declares no such attribute'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_typed_technique_block_under_its_class_name_is_refused(tmp_path: Path) -> None:
    """A block named by a typed technique's class name is an unknown key, and changes nothing."""
    config = Config()
    tuning = config.techniques.block(BodyLimbNav).technique_tuning()
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'techniques:\n  BodyLimbNav:\n    tuning:\n      spurious_dt_floor_px: 9.0\n',
        encoding='utf-8',
    )
    expected = f"{user_path}: techniques: unknown keys: ['BodyLimbNav']"
    with pytest.raises(SchemaError, match=re.escape(expected)):
        config.update_config(user_path)
    assert BodyLimbNav.tuning is tuning
