"""The typed sections of the packaged configuration, for tests that need their values.

The section types have no defaults, since every value lives in the packaged YAML.  A test
that needs a section takes the packaged one from here and changes only the fields it is
about, so it never restates a value.
"""

import contextlib
import dataclasses
import functools
from collections.abc import Iterator
from typing import Any

import pytest

from spindoctor.config import Config
from spindoctor.config.sections import TypedSections
from spindoctor.nav_records import TreeTuning
from spindoctor.nav_technique.nav_technique import NavTechnique

# The technique settings a configuration load assigns to every registered technique class.
_TECHNIQUE_SETTINGS = ('confidence_spec', 'tuning')


@contextlib.contextmanager
def preserved_technique_settings() -> Iterator[None]:
    """Put the technique classes' settings back as they were when the block exits.

    Loading a configuration assigns each registered technique class its confidence
    specification and tuning, which every configuration shares.  A test helper that
    loads a configuration of its own wraps the load in this, so a technique override
    another test applied survives.
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
def packaged_sections() -> TypedSections:
    """Return the typed sections of the packaged configuration.

    The packaged files are read once per process, into a configuration of their own,
    so no override another test applies to the shared configuration reaches it.  The
    load leaves the technique classes' settings as it found them.
    """
    with preserved_technique_settings():
        return Config()._typed_sections()


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
        monkeypatch: Fixture that puts the configuration's sections back after the test.
        config: The configuration to change.
        tuning: The section it answers until then.
    """
    sections = dataclasses.replace(config._typed_sections(), results_tree=tuning)
    monkeypatch.setattr(config, '_sections', sections)
