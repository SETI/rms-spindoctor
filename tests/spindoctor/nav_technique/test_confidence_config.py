"""Tests for the typed ``techniques`` configuration section.

The schema core checks each key's type and range, and is tested in its own suite.
What is tested here is that the packaged section parses, how its blocks are named, the
rule that a block names only attributes its technique declares, and that the loader
refuses a block by a name the section does not use.
"""

import re
from typing import Any

import pytest
from tests.packaged_sections import TEST_OVERRIDE_LABEL, NavConfigWith

from spindoctor.config import Config
from spindoctor.nav_technique import BodyLimbNav
from spindoctor.nav_technique.confidence_config import shipped_techniques
from spindoctor.schema import SchemaError


def test_packaged_techniques_section_parses() -> None:
    """Every shipped technique has a block, and its tuning has the technique's type."""
    section = Config().techniques
    for technique in shipped_techniques():
        tuning = section.block(technique).technique_tuning()
        assert isinstance(tuning, technique.Tuning), technique.name


@pytest.mark.parametrize(
    ('block', 'path'),
    [
        ({'terms': [{'feature': 'nope', 'alpha': 1.0}]}, 'terms[0].feature'),
        ({'hard_zero_if': {'nope': True}}, 'hard_zero_if.nope'),
    ],
    ids=['term', 'hard-zero'],
)
def test_an_undeclared_attribute_is_refused(
    nav_config_with: NavConfigWith, block: dict[str, Any], path: str
) -> None:
    """A block that names an attribute its technique does not declare is refused at load.

    Parameters:
        nav_config_with: Builds a configuration from an override mapping.
        block: The ``body_limb`` block in the override.
        path: The key path the message gives inside the block.
    """
    message = 'BodyLimbNav declares no such attribute'
    expected = f'{TEST_OVERRIDE_LABEL}: techniques.body_limb.{path}: {message}'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with({'techniques': {'body_limb': block}})


def test_a_technique_block_under_its_class_name_is_refused(nav_config_with: NavConfigWith) -> None:
    """A block under a technique's class name is an unknown key, and leaves the tuning alone."""
    tuning = Config().techniques.block(BodyLimbNav).technique_tuning()
    overrides = {'techniques': {'BodyLimbNav': {'tuning': {'spurious_dt_floor_px': 9.0}}}}
    expected = f"{TEST_OVERRIDE_LABEL}: techniques: unknown keys: ['BodyLimbNav']"
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(overrides)
    assert BodyLimbNav.tuning is tuning
