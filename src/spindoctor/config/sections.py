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
from typing import Annotated

from annotated_types import Le

from spindoctor.config.logging_keys import LoggingSection
from spindoctor.config.sections_models import (
    BodiesSection,
    FeatureEmissionSection,
    StarsSection,
    TitanSection,
)
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.nav_records.tuning import TreeTuning
from spindoctor.nav_technique.confidence_config import (
    TechniquesSection,
    shipped_techniques,
    techniques_section_type,
)
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import SCHEMA_CONFIG, PositiveInt, RuleError, Str

__all__ = [
    'MAX_UPSAMPLE_FACTOR',
    'EnvironmentSection',
    'GeneralSection',
    'OffsetSection',
    'OrchestratorSection',
    'TypedSections',
    'typed_sections_type',
]

MAX_UPSAMPLE_FACTOR = 1000
"""The largest FFT upsample factor the ``offset`` section accepts.

The subpixel refinement evaluates a window of ``factor + 1`` samples on each axis, so its
allocation grows with the square of the factor.  At this bound, the window and the two DFT
matrices that produce it take about 65 MB for a 1024-pixel image; at a factor of a
million the window alone would need about 16 TB, which hangs the process rather than
failing it.  A factor of 1000 already resolves a thousandth of a pixel, far finer than any
correlation peak can be located, so the bound never clips a useful setting.
"""


@dataclass(frozen=True)
class GeneralSection:
    """The ``general`` section: settings that no single part of the pipeline owns.

    Parameters:
        truetype_font_dir: The directory searched for the TrueType fonts that annotation
            text is drawn in.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    truetype_font_dir: Str


@dataclass(frozen=True)
class EnvironmentSection:
    """The ``environment`` section: where a deployment keeps its files.

    Every key is optional, and a key left out reads as None; the packaged file leaves
    them all unset.  A reader resolves each location from its command line, from here, and from
    an environment variable.  Each reader applies its own order, and the helpers in
    :mod:`spindoctor.config.config_helper` define the usual one: command line, then this
    section, then the environment variable.

    Parameters:
        pds3_holdings_root: The root directory or URL of the PDS3 holdings that images
            are read from.
        nav_results_root: The root directory or URL navigation results are written to.
        backplane_results_root: The root directory or URL backplanes are written to.
        bundle_results_root: The root directory or URL PDS4 bundles are written to.
        log_root: The root directory or URL logs are written to.
        results_index_db: The connection URL of the results index, or ``none`` for no
            results index.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    pds3_holdings_root: Str | None = None
    nav_results_root: Str | None = None
    backplane_results_root: Str | None = None
    bundle_results_root: Str | None = None
    log_root: Str | None = None
    results_index_db: Str | None = None


@dataclass(frozen=True)
class OffsetSection:
    """The ``offset`` section: how finely an image correlation resolves its peak.

    Parameters:
        correlation_fft_upsample_factor: How many steps of the upsampled correlation
            grid make one pixel, from 1 to :data:`MAX_UPSAMPLE_FACTOR`.  The disc
            correlation, the ring annulus correlation, and the manual navigation
            dialog's automatic match all read it.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    correlation_fft_upsample_factor: Annotated[PositiveInt, Le(MAX_UPSAMPLE_FACTOR)]


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
        logging: The ``logging`` section.
        techniques: The ``techniques`` section: each technique's confidence formula and
            tuning.
        general: The ``general`` section.
        environment: The ``environment`` section.  The packaged file leaves its
            locations unset.
        offset: The ``offset`` section.
        results_tree: The ``results_tree`` section.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    planets: tuple[Str, ...]
    orchestrator: OrchestratorSection
    bodies: BodiesSection
    stars: StarsSection
    titan: TitanSection
    satellites: dict[Str, tuple[Str, ...]]
    feature_emission: FeatureEmissionSection
    logging: LoggingSection
    techniques: TechniquesSection
    general: GeneralSection
    offset: OffsetSection
    results_tree: TreeTuning
    environment: EnvironmentSection

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
