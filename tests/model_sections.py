"""The packaged configuration sections the navigation models read, for tests.

The section types have no defaults, since every value lives in the packaged YAML.  A
test that needs a ``stars``, ``bodies``, or ``titan`` section takes the packaged one
from here and changes only the fields it is about, so it never restates a value.
"""

import dataclasses
import functools
from typing import Any

from spindoctor.config import load_nav_config
from spindoctor.config.sections import NavConfig
from spindoctor.config.sections_models import BodiesSection, StarsSection, TitanSection


@functools.cache
def packaged_sections() -> NavConfig:
    """Return the typed sections of the packaged configuration.

    The packaged files are read and parsed once per process, with no configuration
    built around them, so nothing a configuration load does to the registered
    techniques happens here.
    """
    return load_nav_config()


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
