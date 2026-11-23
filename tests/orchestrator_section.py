"""The packaged ``orchestrator`` configuration section, for tests that need its values.

The section's types have no defaults, since every value lives in the packaged YAML.
A test that needs an ensemble configuration or a reliability gate takes the packaged
one from here and changes only the fields it is about, so it never restates a value.
"""

import dataclasses
import functools
from typing import Any

from spindoctor.config import Config
from spindoctor.config.sections import OrchestratorSection
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig, TierThresholds


@functools.cache
def packaged_orchestrator() -> OrchestratorSection:
    """Return the ``orchestrator`` section of the packaged configuration.

    The packaged files are read once per process, into a configuration of their own,
    so no override another test applies to the shared configuration reaches it.
    """
    return Config().orchestrator


def packaged_ensemble(**changes: Any) -> EnsembleConfig:
    """Return the packaged ensemble configuration, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The configuration.
    """
    return dataclasses.replace(packaged_orchestrator().ensemble, **changes)


def packaged_tiers() -> TierThresholds:
    """Return the packaged confidence tier requirements."""
    return packaged_orchestrator().ensemble.tier_thresholds


def packaged_gate(**changes: Any) -> FeatureReliabilityGate:
    """Return the packaged reliability gate, with the given thresholds changed.

    Parameters:
        **changes: Thresholds to change, keyed by feature type name.

    Returns:
        The gate.
    """
    return dataclasses.replace(packaged_orchestrator().reliability_gate, **changes)
