"""The typed sections of the navigator configuration.

:class:`~spindoctor.config.config.Config` parses the merged configuration's typed
sections into a :class:`TypedSections` each time it installs a configuration, through
the schema core (:func:`spindoctor.schema.parse`).  A section here is a frozen
dataclass, and each of its fields gives a key's type and range.  A key that has no
default is required, and its value lives in the packaged YAML files only.

The ``techniques`` section's fields come from the technique registry
(:mod:`spindoctor.nav_technique.confidence_config`), and the ``pds4`` section's from the
dataset registry (:func:`spindoctor.dataset.pds4_config_keys`), so the type to parse is
:func:`typed_sections_type`, which adds them.

The module imports the types of the sections it holds, and those import the
configuration package, so :mod:`spindoctor.config.config` imports this module only
when it installs a configuration.
"""

import dataclasses
import functools
from dataclasses import dataclass

from spindoctor.config.logging_keys import LoggingSection
from spindoctor.config.sections_products import BackplanesSection, Pds4Section, pds4_section_type
from spindoctor.dataset import pds4_config_keys
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.nav_technique.confidence_config import (
    TechniquesSection,
    shipped_techniques,
    techniques_section_type,
)
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import SCHEMA_CONFIG

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

    The ``techniques`` and ``pds4`` fields are declared here by their base types, which
    have no fields, so this class is never parsed itself: :func:`typed_sections_type`
    returns the type to parse.

    Parameters:
        orchestrator: The ``orchestrator`` section.
        logging: The ``logging`` section.
        techniques: The ``techniques`` section: each technique's confidence formula and
            tuning.
        backplanes: The ``backplanes`` section.
        pds4: The ``pds4`` section: each bundle's entry, keyed by the name its dataset
            class gives the bundle.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    orchestrator: OrchestratorSection
    logging: LoggingSection
    techniques: TechniquesSection
    backplanes: BackplanesSection
    pds4: Pds4Section


@functools.cache
def _typed_sections_type(
    techniques: tuple[type[NavTechnique], ...],
    pds4_bundles: tuple[str, ...],
) -> type[TypedSections]:
    """Return the type to parse for a set of shipped techniques and of bundles.

    The result is cached, so one set of techniques and bundles always gets the same type.

    Parameters:
        techniques: The shipped technique classes, in registry order.
        pds4_bundles: The names the dataset classes give their bundles.

    Returns:
        A subclass of :class:`TypedSections` whose ``techniques`` field has the section
        type for ``techniques``, and whose ``pds4`` field has the section type for
        ``pds4_bundles``.
    """
    sections_type: type[TypedSections] = dataclasses.make_dataclass(
        'TypedSections',
        [
            ('techniques', techniques_section_type(techniques)),
            ('pds4', pds4_section_type(pds4_bundles)),
        ],
        bases=(TypedSections,),
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG},
    )
    return sections_type


def typed_sections_type() -> type[TypedSections]:
    """Return the type the typed sections are parsed into.

    It is :class:`TypedSections` with the ``techniques`` field typed by the section
    type of the shipped techniques
    (:func:`~spindoctor.nav_technique.confidence_config.techniques_section_type`), and
    the ``pds4`` field by the section type of the registered datasets' bundles
    (:func:`~spindoctor.config.sections_products.pds4_section_type`).

    Returns:
        A subclass of :class:`TypedSections`.
    """
    return _typed_sections_type(shipped_techniques(), pds4_config_keys())
