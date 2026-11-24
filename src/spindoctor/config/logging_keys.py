"""Key namespace and validation for the ``logging`` configuration section.

The ``logging`` section names each configurable module by a snake_case key.
For navigation techniques and models those keys are derived from the class
name by :func:`log_key_for`, so a key is always recoverable from the source
and never has to be maintained as a parallel list that can fall out of date.
A class may override its derived key by declaring a ``log_key`` class
attribute.

:func:`validate_logging_config` rejects an unrecognized module key, program
name, or level name when the configuration is loaded, so a typo fails at
startup instead of silently having no effect.  It lives here rather than on
:class:`~spindoctor.config.config.Config` because it needs the technique and
model registries, and those packages import ``spindoctor.config``; the
registry imports are therefore function-local and happen only when validation
runs.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    'CATEGORY_KEYS',
    'CONSOLE_KEYS',
    'LOGGER_KEYS',
    'LOG_LEVEL_NAMES',
    'LOG_LEVEL_VALUES',
    'OTHER_LOG_KEYS',
    'is_shipped',
    'log_key_for',
    'model_log_keys',
    'normalize_level',
    'technique_log_keys',
    'validate_logging_config',
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

LOGGER_KEYS = frozenset({'main', 'image'})
"""The two per-logger global defaults."""

CATEGORY_KEYS = frozenset({'techniques', 'models', 'other'})

CONSOLE_KEYS = frozenset({'main_console', 'image_console'})
"""Keys choosing whether each logger writes to the terminal.

Booleans rather than levels, and the only sink selection the configuration
carries: whether a log is *written to a file* is bound up with where the file
goes, which is a per-run decision, while whether it appears on the terminal is
the kind of preference a machine or an operator keeps.
"""
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
_PROGRAMS_KEY = 'programs'
_STRICT_SCOPE_KEY = 'strict_scope'

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


def is_shipped(cls: type) -> bool:
    """Whether ``cls`` is part of the distributed package.

    The technique and model registries are process-global, and test modules
    register their own subclasses into them.  Filtering on the defining module
    keeps the key namespace to what actually ships, so a configuration key set
    cannot vary with which tests have been imported.  The ``logging`` section's
    keys and the ``techniques`` section's blocks both follow this rule.

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

    keys = {log_key_for(cls) for cls in NavTechnique._registry if is_shipped(cls)}
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

    return frozenset(log_key_for(cls) for cls in NavModel._registry if is_shipped(cls))


def _validate_level(value: Any, location: str) -> None:
    """Raise if ``value`` is not a recognized level name.

    Parameters:
        value: The configured value, expected to be a level-name string.
        location: Dotted path of the key, used in the error message.

    Raises:
        ValueError: If ``value`` is not a string or names no known level.
    """
    if not isinstance(value, str):
        raise ValueError(f'{location} must be a level name string, got {type(value).__name__}')
    if normalize_level(value) not in LOG_LEVEL_NAMES:
        raise ValueError(f'{location} is {value!r}; expected one of {sorted(LOG_LEVEL_NAMES)}')


def _validate_category(
    block: Any, category: str, valid_keys: frozenset[str], location: str
) -> None:
    """Raise if a category block holds an unknown module key or a bad level.

    Parameters:
        block: The category's configured value, expected to be a mapping.
        category: Category name, one of :data:`CATEGORY_KEYS`.
        valid_keys: Module keys this category accepts.
        location: Dotted path of the category, used in error messages.

    Raises:
        ValueError: If ``block`` is not a mapping, names an unknown module, or
            holds a value that is not a level name.
    """
    if not isinstance(block, dict):
        raise ValueError(f'{location} must be a mapping, got {type(block).__name__}')
    for key, value in block.items():
        if key != _CATEGORY_DEFAULT_KEY and key not in valid_keys:
            raise ValueError(
                f'{location}.{key} is not a known {category} key; '
                f'expected {_CATEGORY_DEFAULT_KEY!r} or one of {sorted(valid_keys)}'
            )
        _validate_level(value, f'{location}.{key}')


def _validate_block(block: dict[str, Any], location: str, *, is_top_level: bool) -> None:
    """Validate one logging block: the top-level section or one program's.

    Parameters:
        block: The mapping to validate.
        location: Dotted path of the block, used in error messages.
        is_top_level: True for the ``logging`` section itself, False for a
            block under ``programs``.  Only the top-level block may carry
            ``strict_scope``, which is a global switch, or ``programs``, which
            does not nest.

    Raises:
        ValueError: If the block holds an unknown key or a malformed value.
    """
    categories = {
        'techniques': technique_log_keys(),
        'models': model_log_keys(),
        'other': OTHER_LOG_KEYS,
    }
    for key, value in block.items():
        if key == _PROGRAMS_KEY:
            if is_top_level:
                continue  # unpacked by validate_logging_config
            raise ValueError(
                f'{location}.{_PROGRAMS_KEY} does not nest; per-program overrides live '
                f'only in the top-level logging.{_PROGRAMS_KEY} block'
            )
        if key == _STRICT_SCOPE_KEY:
            if not is_top_level:
                raise ValueError(
                    f'{location}.{_STRICT_SCOPE_KEY} is a global setting and cannot be '
                    f'set for one program'
                )
            if not isinstance(value, bool):
                raise ValueError(
                    f'{location}.{_STRICT_SCOPE_KEY} must be true or false, '
                    f'got {type(value).__name__}'
                )
        elif key in LOGGER_KEYS:
            _validate_level(value, f'{location}.{key}')
        elif key in CONSOLE_KEYS:
            if not isinstance(value, bool):
                raise ValueError(
                    f'{location}.{key} must be true or false, got {type(value).__name__}'
                )
        elif key in CATEGORY_KEYS:
            _validate_category(value, key, categories[key], f'{location}.{key}')
        else:
            allowed = sorted(LOGGER_KEYS | CONSOLE_KEYS | CATEGORY_KEYS)
            raise ValueError(
                f'{location}.{key} is not a known logging key; expected one of {allowed}'
            )


def validate_logging_config(config: 'Config') -> None:
    """Validate the ``logging`` section of ``config``.

    Checks every key in the top-level block and in each ``programs`` block
    against the technique and model registries, the fixed set of other
    image-scoped modules, and the registered program identities, and checks
    that every level is a recognized name.

    Parameters:
        config: The configuration to validate.

    Raises:
        ValueError: On an unknown logging key, category, module key, program
            name, or level name.  The message names the offending dotted path.
    """
    from spindoctor.config.program_names import PROGRAM_NAMES

    section = config.logging
    if not isinstance(section, dict):
        raise ValueError(f'logging must be a mapping, got {type(section).__name__}')

    block = dict(section)
    _validate_block(block, 'logging', is_top_level=True)

    programs = block.get(_PROGRAMS_KEY, {})
    if not isinstance(programs, dict):
        raise ValueError(
            f'logging.{_PROGRAMS_KEY} must be a mapping, got {type(programs).__name__}'
        )
    for program_name, program_block in programs.items():
        location = f'logging.{_PROGRAMS_KEY}.{program_name}'
        if program_name not in PROGRAM_NAMES:
            raise ValueError(
                f'{location} is not a known program; expected one of {sorted(PROGRAM_NAMES)}'
            )
        if not isinstance(program_block, dict):
            raise ValueError(f'{location} must be a mapping, got {type(program_block).__name__}')
        _validate_block(dict(program_block), location, is_top_level=False)
