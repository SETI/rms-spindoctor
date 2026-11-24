"""The packaged ``backplanes`` section, and plane entries, for tests that need them.

The section's types have no defaults, since every value lives in the packaged YAML.  A
test that needs a backplanes section takes the packaged one from here and changes only
the fields it is about, so it never restates a value.
"""

import dataclasses
import functools
from typing import Any

from spindoctor.config import Config
from spindoctor.config.sections_products import (
    BackplanesSection,
    IndexColumnEntry,
    PlaneEntry,
    PlaneIndex,
)
from spindoctor.nav_technique.nav_technique import NavTechnique

# The technique settings a configuration load assigns to every registered technique class.
_TECHNIQUE_SETTINGS = ('confidence_spec', 'tuning')


@functools.cache
def _packaged_backplanes() -> BackplanesSection:
    """Return the ``backplanes`` section of the packaged configuration.

    The packaged files are read once per process, into a configuration of their own,
    so overrides that other tests apply to the shared configuration do not reach it.
    The load then restores the settings the technique classes held before it, so a
    technique override that another test applied survives.
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
        return Config().backplanes
    finally:
        for cls, name, value in saved:
            setattr(cls, name, value)
        for cls, name in unset:
            if name in cls.__dict__:
                delattr(cls, name)


def packaged_backplanes(**changes: Any) -> BackplanesSection:
    """Return the packaged ``backplanes`` section, with the given fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The section.
    """
    return dataclasses.replace(_packaged_backplanes(), **changes)


def plane_entry(name: str, units: str, *, method: str | None = None) -> PlaneEntry:
    """Return a backplane entry, with the index block the index tables read.

    Parameters:
        name: The plane's name.
        units: The unit its array carries.
        method: The oops backplane method it is evaluated with; the plane's name when
            None.

    Returns:
        The entry, with an index block naming its two columns ``minimum_<name>`` and
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
