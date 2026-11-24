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

from dataclasses import dataclass, field
from typing import Annotated

from annotated_types import Le

from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig
from spindoctor.nav_records.tuning import TreeTuning
from spindoctor.schema import SCHEMA_CONFIG, PositiveInt, Str

__all__ = [
    'MAX_UPSAMPLE_FACTOR',
    'EnvironmentSection',
    'GeneralSection',
    'OffsetSection',
    'OrchestratorSection',
    'TypedSections',
]

MAX_UPSAMPLE_FACTOR = 1_000_000
"""The largest FFT upsample factor the ``offset`` section accepts.

A factor above it would push the upsampled correlation peak grid into allocations of
several gigabytes, which hangs the process rather than failing it.  The bound is far above
any useful factor, so it never clips a real setting.
"""


@dataclass(frozen=True)
class GeneralSection:
    """The ``general`` section: settings that belong to no one part of the pipeline.

    Parameters:
        truetype_font_dir: The directory searched for the TrueType fonts that annotation
            text is drawn in.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    truetype_font_dir: Str


@dataclass(frozen=True)
class EnvironmentSection:
    """The ``environment`` section: where a deployment keeps its files.

    No packaged file sets this section, so every key is optional and an absent key reads
    as None.  A program resolves each location from its command line first, then from
    here, then from an environment variable; the helpers in
    :mod:`spindoctor.config.config_helper` apply that order.

    Parameters:
        pds3_holdings_root: The root directory or URL of the PDS3 holdings that images
            are read from.
        nav_results_root: The root directory or URL navigation results are written to.
        backplane_results_root: The root directory or URL backplanes are written to.
        bundle_results_root: The root directory or URL PDS4 bundles are written to.
        log_root: The root directory or URL logs are written to.
        results_index_db: The connection URL of the results index, or ``none`` to name
            no index.
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

    Parameters:
        orchestrator: The ``orchestrator`` section.
        general: The ``general`` section.
        environment: The ``environment`` section, which no packaged file sets; absent,
            it holds no locations.
        offset: The ``offset`` section.
        results_tree: The ``results_tree`` section.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    orchestrator: OrchestratorSection
    general: GeneralSection
    offset: OffsetSection
    results_tree: TreeTuning
    environment: EnvironmentSection = field(default_factory=EnvironmentSection)
