"""Tests for the typed ``general``, ``environment``, and ``offset`` configuration sections.

The schema core checks each key's type and range, and is tested in its own suite.  None
of these sections has a cross-field rule, so what is tested here is that each packaged
section parses.
"""

from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.config.sections import (
    MAX_UPSAMPLE_FACTOR,
    EnvironmentSection,
    GeneralSection,
    OffsetSection,
)
from spindoctor.schema import SchemaError


def test_packaged_general_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().general, GeneralSection)


def test_packaged_offset_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().offset, OffsetSection)


def test_packaged_environment_section_names_no_location() -> None:
    """No packaged file sets the section, so every location reads as None."""
    assert Config().environment == EnvironmentSection()


def test_an_upsample_factor_above_the_bound_is_refused_at_load(tmp_path: Path) -> None:
    """A factor past the bound would hang a correlation, so the load refuses it."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        f'offset:\n  correlation_fft_upsample_factor: {MAX_UPSAMPLE_FACTOR + 1}\n',
        encoding='utf-8',
    )
    with pytest.raises(SchemaError, match='correlation_fft_upsample_factor'):
        Config().update_config(user_path)
