"""The typed sections of the navigator configuration.

:class:`~spindoctor.config.config.Config` parses the merged configuration's typed
sections into a :class:`TypedSections` each time it installs a configuration, through
the schema core (:func:`spindoctor.schema.parse`).  A section here is a frozen
dataclass, and each of its fields gives a key's type and range.  A key that has no
default is required, and its value lives in the packaged YAML files only.

The module imports the types of the sections it holds, and those import the
configuration package, so :mod:`spindoctor.config.config` imports this module only
when it installs a configuration.
"""

from dataclasses import dataclass

from spindoctor.config.sections_models import (
    BodiesSection,
    FeatureEmissionSection,
    StarsSection,
    TitanSection,
)
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.schema import SCHEMA_CONFIG, Str

__all__ = [
    'OrchestratorSection',
    'TypedSections',
]


@dataclass(frozen=True)
class OrchestratorSection:
    """The ``orchestrator`` section: the settings the orchestrator applies to every image.

    Parameters:
        ensemble: The ensemble combine's tunables and confidence tiers.
        reliability_gate: The least reliability each feature type needs to reach a
            technique.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    ensemble: EnsembleConfig
    reliability_gate: FeatureReliabilityGate


@dataclass(frozen=True)
class TypedSections:
    """The sections of the navigator configuration that are parsed into types.

    Each field is one top-level section, named as the files write it.  So the key path
    of a problem found while parsing starts at the section's name, as in
    ``orchestrator.ensemble.min_confidence``.

    Parameters:
        orchestrator: The ``orchestrator`` section.
        bodies: The ``bodies`` section.
        stars: The ``stars`` section.
        titan: The ``titan`` section.
        satellites: Each planet's satellites, keyed by the planet's SPICE name.
        feature_emission: The ``feature_emission`` section.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    orchestrator: OrchestratorSection
    bodies: BodiesSection
    stars: StarsSection
    titan: TitanSection
    satellites: dict[Str, tuple[Str, ...]]
    feature_emission: FeatureEmissionSection
