"""The navigator configuration as one typed object, and its sections.

:func:`~spindoctor.config.config.load_nav_config` parses the whole merged configuration
into a :class:`NavConfig` through the schema core (:func:`spindoctor.schema.parse`).
Each top-level section is a field of it.  A section here is a frozen dataclass, and each
of its fields gives a key's type and range.  A key that has no default is required, and
its value lives in the packaged YAML files only.

The ``techniques`` section's fields come from the technique registry
(:mod:`spindoctor.nav_technique.confidence_config`), and the ``pds4`` section's from the
dataset registry (:func:`spindoctor.dataset.pds4_config_keys`), so the type to parse is
:func:`nav_config_type`, which adds them.  It also adds a field for each section
of the configuration that holds camera blocks
(:func:`~spindoctor.config.sections_camera.camera_sections`).

The module imports the types of the sections it holds, and those import the
configuration package, so :mod:`spindoctor.config.config` imports this module only
when it loads a configuration.
"""

import dataclasses
import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from annotated_types import Le

from spindoctor.config.logging_keys import LoggingSection
from spindoctor.config.sections_camera import CameraBlock, camera_sections
from spindoctor.config.sections_models import (
    BodiesSection,
    BodyShapeEntry,
    FeatureEmissionSection,
    StarsSection,
    TitanSection,
)
from spindoctor.config.sections_products import BackplanesSection, Pds4Section, pds4_section_type
from spindoctor.config.sections_rings import RingsSection
from spindoctor.dataset import pds4_config_keys
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
    'NavConfig',
    'OffsetSection',
    'OrchestratorSection',
    'nav_config_type',
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
class NavConfig:
    """The navigator configuration, parsed: one field for each top-level section.

    Each field is named as the files write its section, so the key path of a problem
    found while parsing starts at the section's name, as in
    ``orchestrator.ensemble.min_confidence``.  The whole merged configuration is parsed
    into it, so a top-level section that is not a field is refused as an unknown key.

    A mapping keyed by planet must name only planets of the ``planets`` section, as
    SPICE spells them, so a key in lower case or misspelled is refused rather than
    left unread.  A mapping keyed by body, ``bodies.geometric_albedo`` or
    ``body_shape``, must spell each body in upper case, for the same reason.

    The ``techniques`` and ``pds4`` fields are declared here by their base types, which
    have no fields, so this class is never parsed itself: :func:`nav_config_type`
    returns the type to parse. That type is a subclass, so it inherits the key rules
    below.  It also has a field for each camera section, named as the files write it,
    and lists their names in :attr:`CAMERA_SECTIONS`; :meth:`camera_block` gives one
    camera's block.

    Parameters:
        planets: The SPICE names of the planets the navigator knows.
        orchestrator: The ``orchestrator`` section.
        bodies: The ``bodies`` section.
        stars: The ``stars`` section.
        titan: The ``titan`` section.
        satellites: Each planet's satellites, keyed by the planet's SPICE name.
        feature_emission: The ``feature_emission`` section.
        rings: The ``rings`` section, with each planet's ring catalog.
        body_shape: The body-shape table, keyed by each body's SPICE name.
        logging: The ``logging`` section.
        techniques: The ``techniques`` section: each technique's confidence formula and
            tuning.
        general: The ``general`` section.
        environment: The ``environment`` section.  The packaged file leaves its
            locations unset.
        offset: The ``offset`` section.
        results_tree: The ``results_tree`` section.
        backplanes: The ``backplanes`` section.
        pds4: The ``pds4`` section: each bundle's entry, keyed by its dataset class's
            ``PDS4_CONFIG_KEY``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    CAMERA_SECTIONS: ClassVar[tuple[str, ...]] = ()
    """The names of the sections that hold camera blocks, each a field of the type."""

    planets: tuple[Str, ...]
    orchestrator: OrchestratorSection
    bodies: BodiesSection
    stars: StarsSection
    titan: TitanSection
    satellites: dict[Str, tuple[Str, ...]]
    feature_emission: FeatureEmissionSection
    rings: RingsSection
    body_shape: dict[Str, BodyShapeEntry]
    logging: LoggingSection
    techniques: TechniquesSection
    general: GeneralSection
    offset: OffsetSection
    results_tree: TreeTuning
    environment: EnvironmentSection
    backplanes: BackplanesSection
    pds4: Pds4Section

    def __post_init__(self) -> None:
        """Refuse a planet, satellite, or body key not in upper case, and a planet key
        that names no planet of ``planets``.

        Every reader looks a planet or body up by its upper-case SPICE name, so a
        lower-case name would be accepted and never read, even where ``planets`` lists it
        the same way.
        """
        keyed_by_body: tuple[tuple[tuple[str, ...], Iterable[str]], ...] = (
            (('bodies', 'geometric_albedo'), self.bodies.geometric_albedo),
            (('body_shape',), self.body_shape),
        )
        for at, bodies in keyed_by_body:
            for body in bodies:
                if body != body.upper():
                    raise RuleError(
                        f'{body!r} is not in upper case; a body is written in upper case, as '
                        'SPICE spells it',
                        at=(*at, body),
                    )
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

    def camera_block(self, section: str, camera: str | None = None) -> CameraBlock:
        """Return one camera's block.

        Parameters:
            section: The camera section, one of :attr:`CAMERA_SECTIONS`.
            camera: The camera's name inside a section that holds several cameras; None
                for a section that is one camera's block.

        Returns:
            The camera's block.

        Raises:
            KeyError: If ``section`` is not a camera section, does not hold a camera
                named ``camera``, or is one camera's block and ``camera`` is given.
        """
        if section not in self.CAMERA_SECTIONS:
            raise KeyError(f'{section!r} is not a camera section of the configuration')
        value: CameraBlock | dict[str, CameraBlock] = getattr(self, section)
        if isinstance(value, CameraBlock):
            if camera is not None:
                raise KeyError(f"{section} is one camera's block; it has no camera {camera!r}")
            return value
        if camera not in value:
            raise KeyError(f'{section} has no camera {camera!r}; it has {sorted(value)}')
        return value[camera]


@functools.cache
def _nav_config_type(
    techniques: tuple[type[NavTechnique], ...],
    pds4_bundles: tuple[str, ...],
    cameras: tuple[tuple[str, bool], ...],
) -> type[NavConfig]:
    """Return the type to parse for a set of shipped techniques, bundles, and cameras.

    The result is cached, so one set of techniques, bundles, and camera sections always
    gets the same type.

    Parameters:
        techniques: The shipped technique classes, in registry order.
        pds4_bundles: The ``PDS4_CONFIG_KEY`` of each dataset class that makes bundles.
        cameras: ``(name, holds_several)`` for each camera section, as
            :func:`~spindoctor.config.sections_camera.camera_sections` gives them.

    Returns:
        A subclass of :class:`NavConfig` whose ``techniques`` field has the section
        type for ``techniques``, whose ``pds4`` field has the section type for
        ``pds4_bundles``, and which has a field for each camera section: a
        :class:`~spindoctor.config.sections_camera.CameraBlock`, or a mapping of camera
        name to one for a section that holds several cameras.
    """
    camera_fields: list[tuple[str, Any]] = [
        (name, dict[Str, CameraBlock] if holds_several else CameraBlock)
        for name, holds_several in cameras
    ]
    sections_type: type[NavConfig] = dataclasses.make_dataclass(
        'NavConfig',
        [
            ('techniques', techniques_section_type(techniques)),
            ('pds4', pds4_section_type(pds4_bundles)),
            *camera_fields,
        ],
        bases=(NavConfig,),
        frozen=True,
        namespace={
            '__pydantic_config__': SCHEMA_CONFIG,
            'CAMERA_SECTIONS': tuple(name for name, _ in cameras),
        },
    )
    return sections_type


def nav_config_type(
    config: Mapping[str, Any], *, known_cameras: frozenset[str] = frozenset()
) -> type[NavConfig]:
    """Return the type a merged configuration is parsed into.

    It is :class:`NavConfig` with the ``techniques`` field typed by the section
    type of the shipped techniques
    (:func:`~spindoctor.nav_technique.confidence_config.techniques_section_type`), the
    ``pds4`` field by the section type of the registered datasets' bundles
    (:func:`~spindoctor.config.sections_products.pds4_section_type`), and a field for
    each section of ``config`` that holds camera blocks
    (:func:`~spindoctor.config.sections_camera.camera_sections`).

    Parameters:
        config: The merged configuration.
        known_cameras: The camera sections of the configuration an override is merged
            over, which stay camera sections whatever the override puts in them.

    Returns:
        A subclass of :class:`NavConfig`.
    """
    static = frozenset(item.name for item in dataclasses.fields(NavConfig))
    cameras = camera_sections(config, typed=static, known=known_cameras)
    return _nav_config_type(shipped_techniques(), pds4_config_keys(), cameras)
