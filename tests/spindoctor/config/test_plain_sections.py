"""Tests for the typed ``general``, ``environment``, and ``offset`` configuration sections.

The schema core checks each key's type and range, and is tested in its own suite.  None
of these sections has a cross-field rule, so what is tested here is that each packaged
section parses.
"""

from spindoctor.config import Config
from spindoctor.config.sections import EnvironmentSection, GeneralSection, OffsetSection


def test_packaged_general_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().general, GeneralSection)


def test_packaged_offset_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().offset, OffsetSection)


def test_packaged_environment_section_names_no_location() -> None:
    """No packaged file sets the section, so every location reads as None."""
    assert Config().environment == EnvironmentSection()
