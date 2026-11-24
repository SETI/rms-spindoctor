"""Tests for the typed ``backplanes`` and ``pds4`` configuration sections.

The schema core checks each key's type and range, and is tested in its own suite.  What
is tested here is that each packaged section parses, each cross-field rule, the bundle
keys the ``pds4`` section accepts, and the lookup of one bundle's entry.  What the
shipped bundle entry holds is tested with its dataset.
"""

import re
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.config.sections_products import (
    BackplanesSection,
    Pds4BundleEntryError,
    Pds4Section,
    PlaneIndex,
    pds4_section_type,
)
from spindoctor.dataset import pds4_config_keys
from spindoctor.schema import SchemaError, parse

_COLUMN = {'name': 'minimum_x', 'description': 'The least x.'}
"""A stand-in index column entry."""


def test_packaged_backplanes_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().backplanes, BackplanesSection)


def test_packaged_pds4_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().pds4, Pds4Section)


def test_the_pds4_section_accepts_each_registered_dataset_s_bundle() -> None:
    """Each bundle key a registered dataset class gives is a field of the section."""
    fields = set(Config().pds4.__dataclass_fields__)
    assert fields == set(pds4_config_keys())


def test_a_pds4_key_no_dataset_names_is_refused_at_load(tmp_path: Path) -> None:
    """An entry for a bundle no dataset makes is refused, naming the file and the key."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('pds4:\n  no_such_bundle:\n    bundle_name: x\n', encoding='utf-8')
    expected = f"{user_path}: pds4: unknown keys: ['no_such_bundle']"
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_the_pds4_section_accepts_exactly_the_bundle_keys_of_the_datasets() -> None:
    """The keys are the ones the datasets that make bundles name, and no others."""
    assert set(pds4_config_keys()) == {'coiss_cruise', 'coiss_saturn'}


def test_an_entry_for_a_dataset_that_has_no_bundle_is_refused_at_load(tmp_path: Path) -> None:
    """An entry under a key that is no bundle's key, such as vgiss, is refused."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text('pds4:\n  vgiss:\n    bundle_name: x\n', encoding='utf-8')
    expected = f"{user_path}: pds4: unknown keys: ['vgiss']"
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_bundle_with_no_entry_is_refused() -> None:
    """Looking up a bundle the files give no entry raises, naming the bundle."""
    section = pds4_section_type(('a_bundle',))()
    with pytest.raises(Pds4BundleEntryError, match="no entry for the bundle 'a_bundle'"):
        section.bundle('a_bundle')


def test_a_lower_case_target_key_is_refused_at_load(tmp_path: Path) -> None:
    """A target key is a SPICE name, so one not in upper case is refused."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'backplanes:\n'
        '  target_lids:\n'
        '    saturn:\n'
        '      lid: urn:x\n'
        "      version: '1.0'\n"
        '      name: Saturn\n'
        '      type: Planet\n',
        encoding='utf-8',
    )
    expected = (
        f"{user_path}: backplanes.target_lids.saturn: 'saturn' is not in upper case; "
        'a target is written in upper case, as SPICE and the PDS spell it'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


def test_a_wrapped_range_given_one_end_is_refused() -> None:
    """A wrapped range is a pair of columns, so a block giving only one end is refused."""
    raw = {
        'data_type': 'ASCII_Real',
        'minimum': _COLUMN,
        'maximum': _COLUMN,
        'wrapped_minimum': _COLUMN,
    }
    expected = 'test.yaml: wrapped_minimum: wrapped_minimum is set and wrapped_maximum is not'
    with pytest.raises(SchemaError, match=re.escape(expected)):
        parse(PlaneIndex, raw, source='test.yaml')
