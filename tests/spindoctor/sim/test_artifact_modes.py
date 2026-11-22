"""The artifact-mode parameter types the registry builds for each mode.

Each mode's map is checked against one dataclass built from its parameters.  A
parameter whose key is a Python keyword reads that key from a map and writes it
back.  A parameter's type tells the sim editor which values a choice offers and
how many items a list holds.
"""

import pytest

from spindoctor.schema import SchemaError
from spindoctor.sim.artifact_modes import ARTIFACT_MODES, mode_scene_map, parse_mode_params


def test_a_keyword_parameter_reads_and_writes_its_own_key() -> None:
    """truncated_frame's 'from' key parses and comes back as 'from'."""
    values = parse_mode_params('truncated_frame', {'from': 'top', 'lines': 4}, source='scene')
    assert mode_scene_map(values) == {'lines': 4, 'from': 'top'}


def test_a_keyword_parameter_refuses_its_field_name() -> None:
    """The field name a keyword parameter takes in Python is not a key a map may write."""
    with pytest.raises(SchemaError, match=r"scene: unknown keys: \['from_'\]"):
        parse_mode_params('truncated_frame', {'from_': 'top'}, source='scene')


def test_a_string_choice_lists_its_values() -> None:
    """A Literal parameter offers its values in the registry's order."""
    param = ARTIFACT_MODES['truncated_frame'].param_map['from']
    assert param.choices == ('bottom', 'top')


def test_an_integer_choice_lists_its_values() -> None:
    """An integer choice offers its values in the registry's order."""
    param = ARTIFACT_MODES['alternating_lines'].param_map['period']
    assert param.choices == (2, 4)


def test_a_number_parameter_offers_no_choices() -> None:
    """A parameter that is not a choice offers no values."""
    assert ARTIFACT_MODES['truncated_frame'].param_map['fraction'].choices is None


def test_a_list_parameter_states_its_length() -> None:
    """A list of integers states how many items it holds."""
    assert ARTIFACT_MODES['cutout_window'].param_map['rect'].length == 4


def test_a_scalar_parameter_has_no_length() -> None:
    """A parameter that is not a list states no length."""
    assert ARTIFACT_MODES['cutout_window'].param_map['probability'].length is None
