"""The typed sections of the navigator configuration.

:class:`~spindoctor.config.config.Config` parses the merged configuration's typed
sections into a :class:`TypedSections` each time it installs a configuration, through
the schema core (:func:`spindoctor.schema.parse`).  A section here is a frozen
dataclass, and each of its fields gives a key's type and range.  A key that has no
default is required, and its value lives in the packaged YAML files only.

The ``techniques`` section's fields come from the technique registry
(:mod:`spindoctor.nav_technique.confidence_config`), so the type to parse is
:func:`typed_sections_type`, which adds them.

The module imports the types of the sections it holds, and those import the
configuration package, so :mod:`spindoctor.config.config` imports this module only
when it installs a configuration.
"""

import dataclasses
import functools
from collections.abc import Iterable
from dataclasses import dataclass

from spindoctor.config.logging_keys import LoggingSection
from spindoctor.config.sections_models import (
    BodiesSection,
    FeatureEmissionSection,
    StarsSection,
    TitanSection,
)
from spindoctor.config.sections_rings import RingsSection
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.nav_technique.confidence_config import (
    TechniquesSection,
    shipped_techniques,
    techniques_section_type,
)
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import SCHEMA_CONFIG, RuleError, Str

__all__ = [
    'OrchestratorSection',
    'TypedSections',
    'typed_sections_type',
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

    The ``techniques`` field is declared here by its base type, which has no fields,
    so this class is never parsed itself: :func:`typed_sections_type` returns the type
    to parse.  That type is a subclass, so it inherits the planet-key rule below.

    Parameters:
        planets: The SPICE names of the planets the navigator knows.
        orchestrator: The ``orchestrator`` section.
        bodies: The ``bodies`` section.
        stars: The ``stars`` section.
        titan: The ``titan`` section.
        satellites: Each planet's satellites, keyed by the planet's SPICE name.
        feature_emission: The ``feature_emission`` section.
        rings: The ``rings`` section, with each planet's ring catalog.
        logging: The ``logging`` section.
        techniques: The ``techniques`` section: each technique's confidence formula and
            tuning.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    planets: tuple[Str, ...]
    orchestrator: OrchestratorSection
    bodies: BodiesSection
    stars: StarsSection
    titan: TitanSection
    satellites: dict[Str, tuple[Str, ...]]
    feature_emission: FeatureEmissionSection
    rings: RingsSection
    logging: LoggingSection
    techniques: TechniquesSection

    def __post_init__(self) -> None:
        """Refuse a planet or satellite name that is not in upper case, and a planet key
        that names no planet of ``planets``.

        Every reader looks a planet up by its upper-case SPICE name, so a lower-case name
        would be accepted and never read, even where ``planets`` lists it the same way.
        """
        for index, name in enumerate(self.planets):
            if name != name.upper():
                raise RuleError(
                    f'{name!r} is not in upper case; a planet is written in upper case, as '
                    'SPICE spells it',
                    at=('planets', index),
                )
        keyed_by_planet: tuple[tuple[tuple[str, ...], Iterable[str]], ...] = (
            (('satellites',), self.satellites),
            (('stars', 'ring_occlusion_radii_km'), self.stars.ring_occlusion_radii_km),
            (
                ('feature_emission', 'ring_annulus', 'planets'),
                self.feature_emission.ring_annulus.planets,
            ),
            (('rings', 'ring_features'), self.rings.ring_features),
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
        for planet, moons in self.satellites.items():
            for index, moon in enumerate(moons):
                if moon != moon.upper():
                    raise RuleError(
                        f'{moon!r} is not in upper case; a body is written in upper case, as '
                        'SPICE spells it',
                        at=('satellites', planet, index),
                    )


@functools.cache
def _typed_sections_type(
    techniques: tuple[type[NavTechnique], ...],
) -> type[TypedSections]:
    """Return the type to parse for a set of shipped techniques.

    The result is cached, so one set of techniques always gets the same type.

    Parameters:
        techniques: The shipped technique classes, in registry order.

    Returns:
        A subclass of :class:`TypedSections` whose ``techniques`` field has the section
        type for ``techniques``.
    """
    sections_type: type[TypedSections] = dataclasses.make_dataclass(
        'TypedSections',
        [('techniques', techniques_section_type(techniques))],
        bases=(TypedSections,),
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG},
    )
    return sections_type


def typed_sections_type() -> type[TypedSections]:
    """Return the type the typed sections are parsed into.

    It is :class:`TypedSections` with the ``techniques`` field typed by the section
    type of the shipped techniques
    (:func:`~spindoctor.nav_technique.confidence_config.techniques_section_type`).

    Returns:
        A subclass of :class:`TypedSections`.
    """
    return _typed_sections_type(shipped_techniques())
