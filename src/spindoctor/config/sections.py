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

from collections.abc import Iterable
from dataclasses import dataclass

from spindoctor.config.sections_models import (
    BodiesSection,
    FeatureEmissionSection,
    StarsSection,
    TitanSection,
)
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.schema import SCHEMA_CONFIG, RuleError, Str

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

    A mapping keyed by planet must name only planets of the ``planets`` section, as
    SPICE spells them, so a key in lower case or misspelled is refused rather than
    left unread.

    Parameters:
        planets: The SPICE names of the planets the navigator knows.
        orchestrator: The ``orchestrator`` section.
        bodies: The ``bodies`` section.
        stars: The ``stars`` section.
        titan: The ``titan`` section.
        satellites: Each planet's satellites, keyed by the planet's SPICE name.
        feature_emission: The ``feature_emission`` section.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    planets: tuple[Str, ...]
    orchestrator: OrchestratorSection
    bodies: BodiesSection
    stars: StarsSection
    titan: TitanSection
    satellites: dict[Str, tuple[Str, ...]]
    feature_emission: FeatureEmissionSection

    def __post_init__(self) -> None:
        """Refuse a key of a mapping keyed by planet that names no planet of ``planets``."""
        keyed_by_planet: tuple[tuple[tuple[str, ...], Iterable[str]], ...] = (
            (('satellites',), self.satellites),
            (('stars', 'ring_occlusion_radii_km'), self.stars.ring_occlusion_radii_km),
            (
                ('feature_emission', 'ring_annulus', 'planets'),
                self.feature_emission.ring_annulus.planets,
            ),
        )
        for at, planets in keyed_by_planet:
            for planet in planets:
                if planet not in self.planets:
                    raise RuleError(
                        f'{planet!r} is not a planet of the planets section; a planet is '
                        'written in upper case, as SPICE spells it',
                        at=(*at, planet),
                        reads=((*at, planet), ('planets',)),
                    )
