"""The packaged navigator configuration and its sections, for tests that need their values.

The section types have no defaults, since every value lives in the packaged YAML.  A test
that needs a section takes the packaged one from here and changes only the fields it is
about, so it never restates a value.  A test that needs a configuration whose files say
something else takes the ``nav_config_with`` fixture (``tests/conftest.py``), which
merges an override held in memory over the packaged files, as a user override file is
merged.  A test that needs its own ring catalog or feature writes it as the YAML would,
and the schema core parses it, so it is checked by the same rules as the packaged
catalogs.
"""

import contextlib
import dataclasses
import functools
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from spindoctor.config import Config, load_nav_config
from spindoctor.config.sections import NavConfig, OrchestratorSection
from spindoctor.config.sections_camera import CameraBlock
from spindoctor.config.sections_models import BodiesSection, StarsSection, TitanSection
from spindoctor.config.sections_products import (
    BackplanesSection,
    IndexColumnEntry,
    PlaneEntry,
    PlaneIndex,
)
from spindoctor.config.sections_rings import RingCatalog, RingFeatureEntry, RingsSection
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_model.rings import RingFeature
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig, TierThresholds
from spindoctor.nav_records import TreeTuning
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import parse
from spindoctor.support.image_class import ImageQualityThresholds, image_quality_thresholds

# The technique settings a configuration load assigns to every registered technique class.
_TECHNIQUE_SETTINGS = ('confidence_spec', 'tuning')

TEST_OVERRIDE_LABEL = 'the test override'
"""What a message gives as the source of a value the ``nav_config_with`` override sets."""

NavConfigWith = Callable[[Mapping[str, Any]], Config]
"""The type of the ``nav_config_with`` fixture: an override in, a loaded configuration out."""


@contextlib.contextmanager
def preserved_technique_settings() -> Iterator[None]:
    """Put the technique classes' settings back as they were when the block exits.

    Installing a configuration assigns each registered technique class its confidence
    specification and tuning, which every configuration shares.  A test that installs a
    configuration of its own wraps the load in this, so a technique override it applied
    does not outlive it, and one another test applied survives.
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
        yield
    finally:
        for cls, name, value in saved:
            setattr(cls, name, value)
        for cls, name in unset:
            if name in cls.__dict__:
                delattr(cls, name)


@functools.cache
def packaged_sections() -> NavConfig:
    """Return the packaged configuration, parsed.

    The packaged files are read and parsed once per process.  Loading a configuration
    changes nothing else, so the technique classes keep whatever settings they hold.
    """
    return load_nav_config()


def packaged_tree_tuning(**changes: Any) -> TreeTuning:
    """Return the packaged ``results_tree`` section, with the given settings changed.

    Parameters:
        **changes: Settings to change, by name.

    Returns:
        The tuning.
    """
    return dataclasses.replace(packaged_sections().results_tree, **changes)


def configure_tree_tuning(
    monkeypatch: pytest.MonkeyPatch, config: Config, tuning: TreeTuning
) -> None:
    """Make a configuration's ``results_tree`` section answer ``tuning`` for one test.

    Parameters:
        monkeypatch: Fixture that puts the configuration back after the test.
        config: The configuration to change.
        tuning: The section it answers until then.
    """
    nav_config = dataclasses.replace(config.nav_config, results_tree=tuning)
    monkeypatch.setattr(config, '_nav_config', nav_config)


def packaged_bodies(**changes: Any) -> BodiesSection:
    """Return the packaged ``bodies`` section, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The section.
    """
    return dataclasses.replace(packaged_sections().bodies, **changes)


def packaged_stars(**changes: Any) -> StarsSection:
    """Return the packaged ``stars`` section, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The section.
    """
    return dataclasses.replace(packaged_sections().stars, **changes)


def packaged_titan() -> TitanSection:
    """Return the packaged ``titan`` section."""
    return packaged_sections().titan


def packaged_orchestrator() -> OrchestratorSection:
    """Return the ``orchestrator`` section of the packaged configuration."""
    return packaged_sections().orchestrator


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


def packaged_rings(**changes: Any) -> RingsSection:
    """Return the packaged ``rings`` section, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The section.
    """
    return dataclasses.replace(packaged_sections().rings, **changes)


def ring_catalog(data: Mapping[str, Any]) -> RingCatalog:
    """Parse one planet's ring catalog, written as the YAML would write it.

    Parameters:
        data: The catalog.

    Returns:
        The parsed catalog.
    """
    return parse(RingCatalog, data, source='a test ring catalog')


def ring_feature(key: str, data: Mapping[str, Any]) -> RingFeature:
    """Parse one ring catalog entry and build its feature.

    Parameters:
        key: The feature's key in the catalog.
        data: The entry, written as the YAML would write it.

    Returns:
        The feature.
    """
    entry = parse(RingFeatureEntry, data, source='a test ring feature')
    return RingFeature.from_entry(key, entry)


def config_with_rings(rings: RingsSection) -> Config:
    """Return a configuration whose ``rings`` section is ``rings``.

    Parameters:
        rings: The section.

    Returns:
        A configuration with every other section packaged.
    """
    config = Config()
    config._nav_config = dataclasses.replace(config.nav_config, rings=rings)
    return config


def packaged_backplanes(**changes: Any) -> BackplanesSection:
    """Return the packaged ``backplanes`` section, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The section.
    """
    return dataclasses.replace(packaged_sections().backplanes, **changes)


def plane_entry(name: str, units: str, *, method: str | None = None) -> PlaneEntry:
    """Return a backplane entry, with the index block the index tables read.

    Parameters:
        name: The plane's name.
        units: The unit its array carries.
        method: The oops backplane method it is evaluated with; the plane's name when
            None.

    Returns:
        The entry, with an index block that gives its two columns ``minimum_<name>`` and
        ``maximum_<name>``, each ``ASCII_Real`` and described.
    """
    return PlaneEntry(
        name=name,
        method=name if method is None else method,
        units=units,
        index=PlaneIndex(
            data_type='ASCII_Real',
            minimum=IndexColumnEntry(name=f'minimum_{name}', description=f'The least {name}.'),
            maximum=IndexColumnEntry(name=f'maximum_{name}', description=f'The greatest {name}.'),
        ),
    )


# The sections of the packaged cameras, the simulated camera's included, and the cameras
# inside them; None for a section that is one camera's block.
PACKAGED_CAMERAS: tuple[tuple[str, str | None], ...] = (
    ('cassini_iss', 'nac'),
    ('cassini_iss', 'wac'),
    ('cassini_iss_calib', 'nac'),
    ('cassini_iss_calib', 'wac'),
    ('galileo_ssi', None),
    ('newhorizons_lorri', None),
    ('voyager_iss', 'vgiss_1_nac'),
    ('voyager_iss', 'vgiss_1_wac'),
    ('voyager_iss', 'vgiss_2_nac'),
    ('voyager_iss', 'vgiss_2_wac'),
    ('sim', None),
)


def packaged_camera(section: str, camera: str | None = None, **changes: Any) -> CameraBlock:
    """Return a packaged camera block, with the given fields changed.

    Parameters:
        section: The camera section.
        camera: The camera inside a section that holds several; None for a section that
            is one camera's block.
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    block = packaged_sections().camera_block(section, camera)
    return dataclasses.replace(block, **changes)


def raw_camera(**changes: Any) -> CameraBlock:
    """Return the packaged Cassini ISS narrow angle camera's block, which reads raw DN.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    return packaged_camera('cassini_iss', 'nac', **changes)


def calibrated_camera(**changes: Any) -> CameraBlock:
    """Return the packaged calibrated Cassini ISS narrow angle camera's block, in I/F.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    return packaged_camera('cassini_iss_calib', 'nac', **changes)


def camera_mapping(block: CameraBlock) -> dict[str, Any]:
    """Return a camera block as the mapping a file gives, with the keys it leaves out omitted.

    Parameters:
        block: The camera block.

    Returns:
        The mapping.
    """

    def strip(value: Any) -> Any:
        """Drop the keys whose value is None from every mapping inside ``value``."""
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if item is not None}
        return value

    mapping: dict[str, Any] = strip(dataclasses.asdict(block))
    return mapping


def raw_thresholds(**changes: Any) -> ImageQualityThresholds:
    """Return the image-quality thresholds of :func:`raw_camera`, with fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The thresholds.
    """
    return dataclasses.replace(image_quality_thresholds(raw_camera()), **changes)
