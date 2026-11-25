"""The packaged ``rings`` section, and ring catalogs and features built from mappings.

The section types have no defaults, since every value lives in the packaged YAML.  A
test that needs a ``rings`` section takes the packaged one from here and changes only
the fields it is about.  A test that needs its own catalog or feature writes it as the
YAML would, and the schema core parses it, so it is checked by the same rules as the
packaged catalogs.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

from spindoctor.config import Config
from spindoctor.config.sections_rings import RingCatalog, RingFeatureEntry, RingsSection
from spindoctor.nav_model.rings import RingFeature
from spindoctor.schema import parse
from tests.model_sections import packaged_sections


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
    config._sections = dataclasses.replace(config._typed_sections(), rings=rings)
    return config
