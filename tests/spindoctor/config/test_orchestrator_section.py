"""Tests for the typed ``orchestrator`` configuration section.

The schema core checks each key's type and range, and is tested in its own suite.
What is tested here is that the packaged section parses, the section's cross-field
rule, and that the loader refuses a bad override at load.
"""

import re
from pathlib import Path

import pytest
from tests.packaged_sections import packaged_sections

from spindoctor.config import Config
from spindoctor.config.sections import OrchestratorSection
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import SchemaError


def test_packaged_orchestrator_section_parses() -> None:
    """The packaged section parses, and the low tier's absent sigma bound reads as None."""
    section = Config().orchestrator
    assert isinstance(section, OrchestratorSection)
    assert section.ensemble.tier_thresholds.low.max_sigma_px is None


def test_an_inverted_body_witness_phase_window_is_refused(tmp_path: Path) -> None:
    """The shape-lock phase ceiling must be below the collapse phase floor, not equal to it."""
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'orchestrator:\n  ensemble:\n    body_witness_shape_lock_max_phase_deg: 90\n',
        encoding='utf-8',
    )
    expected = (
        f'{user_path}: orchestrator.ensemble.body_witness_shape_lock_max_phase_deg: '
        'must be below body_witness_collapse_min_phase_deg (90), got 90'
    )
    with pytest.raises(SchemaError, match=re.escape(expected)):
        Config().update_config(user_path)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            'orchestrator:\n  ensemble:\n    min_confidence: 2\n',
            'orchestrator.ensemble.min_confidence: expected a number in [0, 1], got 2',
        ),
        (
            'orchestrator:\n  reliability_gate:\n    BODY_HALO: 0.5\n',
            "orchestrator.reliability_gate: unknown keys: ['BODY_HALO']",
        ),
    ],
    ids=['bad-value', 'unknown-key'],
)
def test_a_bad_override_is_refused_and_changes_nothing(
    tmp_path: Path, text: str, expected: str
) -> None:
    """A bad value or an unknown key is refused at load, naming the file and the key.

    Parameters:
        tmp_path: A directory for the override file.
        text: The override file's text.
        expected: The message's key path and problem.
    """
    config = Config()
    section = config.orchestrator
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(text, encoding='utf-8')
    with pytest.raises(SchemaError, match=re.escape(f'{user_path}: {expected}')):
        config.update_config(user_path)
    assert config.orchestrator is section
    assert config.override_paths == ()


def test_an_override_refused_by_validation_keeps_the_section(tmp_path: Path) -> None:
    """An override whose section parses but fails the load-time checks changes nothing."""
    config = Config()
    section = config.orchestrator
    user_path = tmp_path / 'user_config.yaml'
    user_path.write_text(
        'orchestrator:\n  ensemble:\n    min_confidence: 0.1\n'
        'bodies:\n  blob_max_overflow_fraction: 5\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='blob_max_overflow_fraction must be finite'):
        config.update_config(user_path)
    assert config.orchestrator is section
    assert not config.override_paths


def test_the_packaged_section_helper_keeps_technique_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the packaged section for tests leaves a technique's overridden tuning alone."""
    technique = next(cls for cls in NavTechnique._registry if not cls.name.startswith('_'))
    override = object()
    monkeypatch.setattr(technique, 'tuning', override)
    packaged_sections.cache_clear()
    packaged_sections()
    assert technique.tuning is override
