"""The packaged configuration sections the navigation models read, for tests.

The section types have no defaults, since every value lives in the packaged YAML.  A
test that needs a ``stars``, ``bodies``, or ``titan`` section takes the packaged one
from here and changes only the fields it is about, so it never restates a value.
"""

import dataclasses
import functools
from typing import Any

from spindoctor.config.config import _PACKAGED_CONFIG_DIR, _parse_typed_sections
from spindoctor.config.sections import TypedSections
from spindoctor.config.sections_models import BodiesSection, StarsSection, TitanSection
from spindoctor.schema import NullPolicy, merge, read_layer


@functools.cache
def packaged_sections() -> TypedSections:
    """Return the typed sections of the packaged configuration.

    The packaged files are read and parsed once per process, with no configuration
    built around them, so nothing a configuration load does to the registered
    techniques happens here.
    """
    layers = [read_layer(path) for path in sorted(_PACKAGED_CONFIG_DIR.glob('*.yaml'))]
    merged = merge(layers, nulls=NullPolicy.REFUSE)
    sections, _origins = _parse_typed_sections(
        merged.data, merged.origins, source='the packaged configuration'
    )
    return sections


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
