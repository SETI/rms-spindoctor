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
from spindoctor.nav_technique.nav_technique import NavTechnique

# The technique settings a configuration load assigns to every registered technique class.
_TECHNIQUE_SETTINGS = ('confidence_spec', 'tuning')


@functools.cache
def packaged_orchestrator() -> OrchestratorSection:
    """Return the ``orchestrator`` section of the packaged configuration.

    The packaged files are read once per process, into a configuration of their own,
    so no override another test applies to the shared configuration reaches it.

    Loading a configuration assigns each registered technique class its confidence
    specification and tuning, which every configuration shares. The load here puts
    back the settings the classes held before it, so a technique override another
    test applied survives.
    """
    saved = [
        (cls, name, cls.__dict__[name])
        for cls in NavTechnique._registry
        for name in _TECHNIQUE_SETTINGS
        if name in cls.__dict__
    ]
    unset = [
        (cls, name)
        for cls in NavTechnique._registry
        for name in _TECHNIQUE_SETTINGS
        if name not in cls.__dict__
    ]
    try:
        return Config().orchestrator
    finally:
        for cls, name, value in saved:
            setattr(cls, name, value)
        for cls, name in unset:
            if name in cls.__dict__:
                delattr(cls, name)


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
