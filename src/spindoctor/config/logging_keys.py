"""Key namespace and section type of the ``logging`` configuration section.

The ``logging`` section names each configurable module by a snake_case key.
For navigation techniques and models those keys are derived from the class
name by :func:`log_key_for`, so a key is always recoverable from the source
and never has to be maintained as a parallel list that can fall out of date.
A class may override its derived key by declaring a ``log_key`` class
attribute.

:class:`LoggingSection` is the section's type.  The configuration parses the
merged section into it each time a configuration is installed, so an
unrecognized module key, program name, or level name is refused at load with
the file and the key path, instead of silently having no effect.  The module
keys a category accepts come from the technique and model registries, and
those packages import ``spindoctor.config``, so the registry imports are
function-local: they run when a module key is checked, during a configuration
load.  No module loads the configuration while it is being imported, so by
then importing either package has registered every class it defines.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Annotated, Any

from pydantic import AfterValidator

from spindoctor.schema import SCHEMA_CONFIG, Bool, Str

from .program_names import PROGRAM_NAMES

__all__ = [
    'CATEGORY_KEYS',
    'LOG_LEVEL_NAMES',
    'LOG_LEVEL_VALUES',
    'OTHER_LOG_KEYS',
    'LoggingSection',
    'ProgramLogging',
    'log_key_for',
    'model_log_keys',
    'normalize_level',
    'technique_log_keys',
]


LOG_LEVEL_VALUES = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
    'NONE': 51,
}
"""Numeric severity of each level name, ordered least to most severe.

``NONE`` sits above ``CRITICAL`` so that selecting it suppresses every record
the library can emit, rather than relying on nothing happening to log at
``CRITICAL``.
"""

LOG_LEVEL_NAMES = frozenset(LOG_LEVEL_VALUES)
"""Level names accepted anywhere in the ``logging`` section."""

CATEGORY_KEYS = frozenset({'techniques', 'models', 'other'})
"""Categories that group per-module overrides."""

OTHER_LOG_KEYS = frozenset(
    {
        'annotate',
        'correlate',
        'ensemble',
        'image_derivatives',
        'obs',
        'orchestrator',
        'provenance',
    }
)
"""Image-scoped modules that are neither a technique nor a model.

A module earns a key here only once it opens a section of its own, because a
level is applied at ``logger.open()``; a key naming a component that never
opens one would validate cleanly and then do nothing.  Every key listed here
has one.

A per-image backend has no key here.  Each program drives at most one backend,
so that backend's verbosity is the program's ``image`` level.
"""

_CATEGORY_DEFAULT_KEY = 'default'

_CAMEL_BOUNDARY = re.compile(r'(?<!^)(?=[A-Z])')
_CLASS_NAME_PREFIXES = ('NavTechnique', 'NavModel')
_CLASS_NAME_SUFFIXES = ('Simulated', 'Nav')


def log_key_for(cls: type) -> str:
    """Return the ``logging`` configuration key naming ``cls``.

    The key is the class name with a ``NavTechnique`` or ``NavModel`` prefix
    removed, then a trailing ``Simulated`` and a trailing ``Nav`` removed, then
    converted from CamelCase to snake_case.  ``TitanHazeNav`` becomes
    ``titan_haze`` and ``NavTechniqueManual`` becomes ``manual``.  A simulated
    model shares its real sibling's key, because the two are one component
    differing only in where their inputs come from, so both ``NavModelRings``
    and ``NavModelRingsSimulated`` become ``rings``.

    A class whose derived key would be wrong may declare a ``log_key`` class
    attribute instead.  An inherited one counts, so a family that should share
    a key declares it once on their base -- which is what
    :attr:`~spindoctor.support.nav_base.NavBase.resolved_log_key` reads at run
    time.  The two have to agree: a key honored at run time but not here would
    name a component that the configuration rejects, while the key the
    configuration did accept silently governed nothing.

    Parameters:
        cls: The technique or model class to name.

    Returns:
        The snake_case configuration key for ``cls``.
    """
    declared = getattr(cls, 'log_key', None)
    if declared is not None:
        return str(declared)

    # A leading underscore would otherwise survive the CamelCase split and
    # double up, turning _StubNav into "__stub".
    name = cls.__name__.lstrip('_')
    for prefix in _CLASS_NAME_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix) :]
            break
    # Strip suffixes until none match rather than making one pass, so a name
    # carrying both (a simulated technique, FooSimulatedNav) reduces the same
    # way regardless of the order the suffixes appear in.
    stripping = True
    while stripping:
        stripping = False
        for suffix in _CLASS_NAME_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                stripping = True
    return _CAMEL_BOUNDARY.sub('_', name).lower()


def normalize_level(value: str) -> str:
    """Return the canonical spelling of a configured level name.

    Level names are accepted in any case and with surrounding whitespace, so
    every consumer must canonicalize the same way.  This is that one place;
    read a level through it rather than upper-casing at the point of use.

    Parameters:
        value: A level name as it appears in the configuration.

    Returns:
        The upper-case level name with surrounding whitespace removed.
    """
    return value.strip().upper()


def _is_shipped(cls: type) -> bool:
    """Whether ``cls`` is part of the distributed package.

    The technique and model registries are process-global, and test modules
    register their own subclasses into them.  Filtering on the defining module
    keeps the key namespace to what actually ships, so a configuration key set
    cannot vary with which tests have been imported.

    Parameters:
        cls: A registered technique or model class.

    Returns:
        True when ``cls`` is defined inside the ``spindoctor`` package.
    """
    return cls.__module__.startswith('spindoctor.')


def technique_log_keys() -> frozenset[str]:
    """Return the configuration key for every navigation technique.

    Covers the autonomous techniques in ``NavTechnique._registry`` plus the
    interactive ``NavTechniqueManual``, which logs like the others but is
    deliberately excluded from the autonomous registry.

    Returns:
        The set of valid keys for the ``logging.techniques`` category.
    """
    from spindoctor.nav_technique.nav_technique import NavTechnique
    from spindoctor.nav_technique.nav_technique_manual import NavTechniqueManual

    keys = {log_key_for(cls) for cls in NavTechnique._registry if _is_shipped(cls)}
    keys.add(log_key_for(NavTechniqueManual))
    return frozenset(keys)


def model_log_keys() -> frozenset[str]:
    """Return the configuration key for every navigation model.

    Returns:
        The set of valid keys for the ``logging.models`` category.  Simulated
        models collapse onto their real sibling's key, so the result names
        model families rather than classes.
    """
    from spindoctor.nav_model.nav_model import NavModel

    return frozenset(log_key_for(cls) for cls in NavModel._registry if _is_shipped(cls))


def _level(value: str) -> str:
    """Return the canonical spelling of a configured level, refusing an unknown one.

    Parameters:
        value: A level name as a file writes it.

    Returns:
        The level name as :func:`normalize_level` spells it.

    Raises:
        ValueError: If ``value`` names no known level.
    """
    level = normalize_level(value)
    if level not in LOG_LEVEL_NAMES:
        raise ValueError(f'{value!r} is not one of {sorted(LOG_LEVEL_NAMES)}')
    return level


def _module_key(key: str, valid_keys: frozenset[str], category: str) -> str:
    """Return ``key`` if a category accepts it; refuse it if not.

    Parameters:
        key: A key of the category's mapping.
        valid_keys: The module keys the category accepts, besides ``default``.
        category: The category's name, one of :data:`CATEGORY_KEYS`.

    Returns:
        ``key``.

    Raises:
        ValueError: If ``key`` is neither ``default`` nor one of ``valid_keys``.
    """
    if key != _CATEGORY_DEFAULT_KEY and key not in valid_keys:
        raise ValueError(
            f'{key!r} is not a known {category} key; '
            f'expected {_CATEGORY_DEFAULT_KEY!r} or one of {sorted(valid_keys)}'
        )
    return key


def _technique_key(key: str) -> str:
    """Return ``key`` if the ``techniques`` category accepts it; refuse it if not."""
    return _module_key(key, technique_log_keys(), 'techniques')


def _model_key(key: str) -> str:
    """Return ``key`` if the ``models`` category accepts it; refuse it if not."""
    return _module_key(key, model_log_keys(), 'models')


def _other_key(key: str) -> str:
    """Return ``key`` if the ``other`` category accepts it; refuse it if not."""
    return _module_key(key, OTHER_LOG_KEYS, 'other')


def _program_name(name: str) -> str:
    """Return ``name`` if it is a registered program identity; refuse it if not.

    Parameters:
        name: A key of the ``programs`` mapping.

    Returns:
        ``name``.

    Raises:
        ValueError: If ``name`` is not in
            :data:`~spindoctor.config.program_names.PROGRAM_NAMES`.
    """
    if name not in PROGRAM_NAMES:
        raise ValueError(
            f'{name!r} is not a known program; expected one of {sorted(PROGRAM_NAMES)}'
        )
    return name


_Level = Annotated[Str, AfterValidator(_level)]
_TechniqueKey = Annotated[Str, AfterValidator(_technique_key)]
_ModelKey = Annotated[Str, AfterValidator(_model_key)]
_OtherKey = Annotated[Str, AfterValidator(_other_key)]
_ProgramName = Annotated[Str, AfterValidator(_program_name)]


@dataclass(frozen=True)
class ProgramLogging:
    """One program's block under ``logging.programs``: the settings it overrides.

    Every key is optional, because a block names only what differs for its program.
    A key left out keeps the top-level value, and a category merges with the
    top-level category key by key, so a block can set one module without restating
    the rest.  ``strict_scope`` is a global switch and ``programs`` does not nest, so
    neither is a key here.

    Parameters:
        main: The main logger's level for this program.
        image: The image logger's level for this program.
        main_console: Whether the main logger writes to the terminal.
        image_console: Whether the image logger writes to the terminal.
        techniques: Per-technique levels, and an optional ``default``.
        models: Per-model levels, and an optional ``default``.
        other: Levels of the other image-scoped modules, and an optional ``default``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    main: _Level | None = None
    image: _Level | None = None
    main_console: Bool | None = None
    image_console: Bool | None = None
    techniques: Mapping[_TechniqueKey, _Level] | None = None
    models: Mapping[_ModelKey, _Level] | None = None
    other: Mapping[_OtherKey, _Level] | None = None


@dataclass(frozen=True)
class LoggingSection:
    """The ``logging`` section: the levels and terminal sinks of the two loggers.

    Each level is held in its canonical spelling, upper case, whatever case the
    file wrote it in.  A category maps module keys, and optionally ``default``, to
    levels.  Its module keys are those of the registered techniques
    (:func:`technique_log_keys`), the registered models (:func:`model_log_keys`),
    and :data:`OTHER_LOG_KEYS`.

    The two console keys are the only sink selection the configuration carries.
    Whether a log is written to a file is bound up with where the file goes, which
    is a per-run decision, while whether it appears on the terminal is the kind of
    preference a machine or an operator keeps.

    Parameters:
        strict_scope: Whether an image-scoped component that logs with no image
            scope open raises, rather than warns.
        main: The main logger's level.
        image: The image logger's level, and that of every image-scoped module
            with no level of its own.
        main_console: Whether the main logger writes to the terminal.
        image_console: Whether the image logger writes to the terminal.
        techniques: Per-technique levels, and an optional ``default``.
        models: Per-model levels, and an optional ``default``.
        other: Levels of the other image-scoped modules, and an optional ``default``.
        programs: Each program's overrides, keyed by its identity from
            :mod:`spindoctor.config.program_names`.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    strict_scope: Bool
    main: _Level
    image: _Level
    main_console: Bool
    image_console: Bool
    techniques: Mapping[_TechniqueKey, _Level]
    models: Mapping[_ModelKey, _Level]
    other: Mapping[_OtherKey, _Level]
    programs: Mapping[_ProgramName, ProgramLogging]

    def for_program(self, program_name: str) -> 'LoggingSection':
        """Return the settings in force for one program.

        The program's block under ``programs`` wins over the top-level value key by
        key, and its categories merge with the top-level categories key by key.

        Parameters:
            program_name: The program's identity, as declared by its dispatch
                module.  A name with no block, such as an empty one, selects the
                top-level settings alone.

        Returns:
            The merged settings, whose ``programs`` is empty.
        """
        block = self.programs.get(program_name, ProgramLogging())
        merged: dict[str, Any] = {
            field.name: getattr(self, field.name) for field in fields(LoggingSection)
        }
        for field in fields(ProgramLogging):
            value = getattr(block, field.name)
            if value is None:
                continue
            if field.name in CATEGORY_KEYS:
                merged[field.name] = {**merged[field.name], **value}
            else:
                merged[field.name] = value
        merged['programs'] = {}
        return LoggingSection(**merged)

    def categories(self) -> dict[str, Mapping[str, str]]:
        """Return each category's levels, keyed by the category's name.

        Returns:
            The ``techniques``, ``models``, and ``other`` mappings, keyed by
            :data:`CATEGORY_KEYS`.
        """
        return {'techniques': self.techniques, 'models': self.models, 'other': self.other}
