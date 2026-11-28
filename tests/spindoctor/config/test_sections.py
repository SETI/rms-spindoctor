"""Tests for the sections :mod:`spindoctor.config.sections` defines.

The schema core checks each key's type and range, and is tested in its own suite.  What
is tested here is that each packaged section parses, the ``orchestrator`` section's
cross-field rule, and that the loader refuses a bad value in these sections at load.
"""

import re

import pytest
from tests.packaged_sections import TEST_OVERRIDE_LABEL, NavConfigWith

from spindoctor.config import Config
from spindoctor.config.sections import (
    MAX_UPSAMPLE_FACTOR,
    EnvironmentSection,
    GeneralSection,
    OffsetSection,
    OrchestratorSection,
)
from spindoctor.schema import SchemaError


def test_packaged_general_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().general, GeneralSection)


def test_packaged_offset_section_parses() -> None:
    """The packaged section parses."""
    assert isinstance(Config().offset, OffsetSection)


def test_packaged_environment_section_names_no_location() -> None:
    """The packaged file leaves every location unset, so each reads as None."""
    assert Config().environment == EnvironmentSection()


def test_packaged_orchestrator_section_parses() -> None:
    """The packaged section parses, and the low tier's absent sigma bound reads as None."""
    section = Config().orchestrator
    assert isinstance(section, OrchestratorSection)
    assert section.ensemble.tier_thresholds.low.max_sigma_px is None


def test_an_upsample_factor_above_the_bound_is_refused_at_load(
    nav_config_with: NavConfigWith,
) -> None:
    """A factor past the bound would hang a correlation, so the load refuses it."""
    with pytest.raises(SchemaError, match='correlation_fft_upsample_factor'):
        nav_config_with({'offset': {'correlation_fft_upsample_factor': MAX_UPSAMPLE_FACTOR + 1}})


def test_an_inverted_body_witness_phase_window_is_refused(nav_config_with: NavConfigWith) -> None:
    """The shape-lock phase ceiling must be below the collapse phase floor, not equal to it."""
    expected = (
        f'{TEST_OVERRIDE_LABEL}: orchestrator.ensemble.body_witness_shape_lock_max_phase_deg: '
        'must be below body_witness_collapse_min_phase_deg (90), got 90'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        nav_config_with(
            {'orchestrator': {'ensemble': {'body_witness_shape_lock_max_phase_deg': 90}}}
        )


@pytest.mark.parametrize(
    ('override', 'expected'),
    [
        (
            {'orchestrator': {'ensemble': {'min_confidence': 2}}},
            'orchestrator.ensemble.min_confidence: expected a number in [0, 1], got 2',
        ),
        (
            {'orchestrator': {'reliability_gate': {'BODY_HALO': 0.5}}},
            "orchestrator.reliability_gate: unknown keys: ['BODY_HALO']",
        ),
    ],
    ids=['bad-value', 'unknown-key'],
)
def test_a_bad_orchestrator_override_is_refused(
    nav_config_with: NavConfigWith, override: dict[str, object], expected: str
) -> None:
    """A bad value or an unknown key is refused at load, naming the source and the key.

    Parameters:
        nav_config_with: Builds a configuration with an override.
        override: The override.
        expected: The message's key path and problem.
    """
    with pytest.raises(SchemaError, match=re.escape(f'{TEST_OVERRIDE_LABEL}: {expected}')):
        nav_config_with(override)
