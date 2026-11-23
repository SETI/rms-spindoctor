"""Tests for the YAML plumbing of ensemble and reliability-gate settings.

Covers ``EnsembleConfig.from_mapping``, ``FeatureReliabilityGate.
from_mapping``, and the invariant that the bundled
``config_540_orchestrator.yaml`` section reproduces the code defaults
exactly.
"""

import pytest

from spindoctor.config.config import Config
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.reliability import (
    DEFAULT_RELIABILITY_THRESHOLDS,
    FeatureReliabilityGate,
)
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig


def test_ensemble_from_mapping_none_returns_defaults() -> None:
    assert EnsembleConfig.from_mapping(None) == EnsembleConfig()


def test_ensemble_from_mapping_empty_returns_defaults() -> None:
    assert EnsembleConfig.from_mapping({}) == EnsembleConfig()


def test_ensemble_from_mapping_scalar_override() -> None:
    cfg = EnsembleConfig.from_mapping({'min_confidence': 0.05})
    assert cfg.min_confidence == 0.05


def test_ensemble_from_mapping_scalar_override_keeps_other_defaults() -> None:
    cfg = EnsembleConfig.from_mapping({'min_confidence': 0.05})
    assert cfg.agreement_sigma == EnsembleConfig().agreement_sigma


def test_ensemble_from_mapping_partial_tier_override() -> None:
    cfg = EnsembleConfig.from_mapping({'tier_thresholds': {'low': {'min_confidence': 0.01}}})
    assert cfg.tier_thresholds['low']['min_confidence'] == 0.01


def test_ensemble_from_mapping_partial_tier_keeps_other_tiers() -> None:
    cfg = EnsembleConfig.from_mapping({'tier_thresholds': {'low': {'min_confidence': 0.01}}})
    assert cfg.tier_thresholds['high'] == EnsembleConfig().tier_thresholds['high']


def test_ensemble_from_mapping_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match=r'Unknown orchestrator\.ensemble config keys'):
        EnsembleConfig.from_mapping({'not_a_field': 1.0})


def test_ensemble_from_mapping_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tier 'extreme'"):
        EnsembleConfig.from_mapping({'tier_thresholds': {'extreme': {}}})


def test_ensemble_from_mapping_unknown_tier_key_raises() -> None:
    with pytest.raises(ValueError, match="key 'max_wobble'"):
        EnsembleConfig.from_mapping({'tier_thresholds': {'low': {'max_wobble': 1.0}}})


def test_gate_from_mapping_none_returns_defaults() -> None:
    gate = FeatureReliabilityGate.from_mapping(None)
    assert gate.thresholds == DEFAULT_RELIABILITY_THRESHOLDS


def test_gate_from_mapping_partial_override() -> None:
    gate = FeatureReliabilityGate.from_mapping({'BODY_DISC': 0.05})
    assert gate.thresholds[NavFeatureType.BODY_DISC] == 0.05


def test_gate_from_mapping_partial_override_keeps_other_types() -> None:
    gate = FeatureReliabilityGate.from_mapping({'BODY_DISC': 0.05})
    assert (
        gate.thresholds[NavFeatureType.STAR] == DEFAULT_RELIABILITY_THRESHOLDS[NavFeatureType.STAR]
    )


def test_gate_from_mapping_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown feature type 'BODY_HALO'"):
        FeatureReliabilityGate.from_mapping({'BODY_HALO': 0.5})


def test_bundled_yaml_ensemble_section_matches_code_defaults() -> None:
    config = Config()
    config.read_config()
    section = config.orchestrator.get('ensemble')
    assert EnsembleConfig.from_mapping(section) == EnsembleConfig()


def test_bundled_yaml_gate_section_matches_code_defaults() -> None:
    config = Config()
    config.read_config()
    section = config.orchestrator.get('reliability_gate')
    gate = FeatureReliabilityGate.from_mapping(section)
    assert gate.thresholds == DEFAULT_RELIABILITY_THRESHOLDS


def test_ensemble_from_mapping_non_numeric_scalar_names_key() -> None:
    with pytest.raises(
        ValueError, match=r'orchestrator\.ensemble\.min_confidence must be a number'
    ):
        EnsembleConfig.from_mapping({'min_confidence': 'high'})


def test_ensemble_from_mapping_non_numeric_tier_value_names_key() -> None:
    with pytest.raises(
        ValueError, match=r"tier_thresholds\['low'\]\.min_confidence must be a number"
    ):
        EnsembleConfig.from_mapping({'tier_thresholds': {'low': {'min_confidence': 'tiny'}}})


def test_gate_from_mapping_non_numeric_value_names_key() -> None:
    with pytest.raises(ValueError, match=r'orchestrator\.reliability_gate\.STAR must be a number'):
        FeatureReliabilityGate.from_mapping({'STAR': 'strict'})
