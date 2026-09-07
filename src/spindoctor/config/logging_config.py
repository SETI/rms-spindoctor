"""Resolution and construction of the two SpinDoctor loggers.

Two loggers exist.  The main logger covers one program run and reports what
the program is doing at the top level.  The image logger covers one image
inside one per-image backend and carries that image's processing detail.

Each logger writes to a console sink, a file sink, or both.  Both sinks always
share a level, so one level per module governs whatever sinks are enabled, and
a module's verbosity can be expressed as the ``level=`` argument of a single
``logger.open(...)`` call.

Making that work takes care.  A pdslogger section level is a floor applied
before the handlers, and each handler then applies its own level, so what
reaches a sink is the more severe of the two.  Handlers are therefore built at
the most verbose level any module could ask for, and the per-section floor
does all of the discrimination.  Building them at the plain image level would
silently drop every module configured more verbose than it -- and the section
summary would still count the dropped records, reporting output that was never
written.

Levels resolve in two steps.  The top-level ``logging`` block is first merged
with ``logging.programs.<program>``, the program block winning key by key;
then, against that merged result, the most specific setting wins:

    --log-level MODULE=LEVEL
      > <category>.<module>
      > <category>.default
      > --log-level-main / --log-level-image
      > --log-level LEVEL
      > main / image
      > INFO

Building a logger attaches exactly the enabled sinks.  When neither is
enabled the builder attaches ``NULL_HANDLER``, because a PdsLogger left with
no handlers does not go quiet -- it falls back to printing every record to
stdout regardless of level.
"""

import argparse
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import pdslogger
from filecache import FCPath
from pdslogger import PdsLogger

from .logger import MAIN_LOGGER
from .logging_keys import (
    CATEGORY_KEYS,
    LOG_LEVEL_NAMES,
    LOG_LEVEL_VALUES,
    normalize_level,
)

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    'BACKEND_NAMES',
    'DEFAULT_LEVEL',
    'LOG_TIMESTAMP_FORMAT',
    'SILENT_LEVEL',
    'LogLevels',
    'LogSinks',
    'RunLogging',
    'absolute_log_root',
    'build_cloud_task_logging',
    'build_image_log_handlers',
    'build_main_logger',
    'build_run_logging',
    'image_log_path',
    'isolate_cloud_task_logging',
    'log_levels',
    'main_log_path',
    'resolve_log_levels',
    'run_logging_for_root',
    'run_timestamp',
    'set_log_levels',
    'sinks_from_arguments',
]


DEFAULT_LEVEL = 'INFO'
"""Level used when nothing in the arguments or configuration says otherwise."""

LOG_TIMESTAMP_FORMAT = '%Y-%m-%dT%H-%M-%S'
"""Suffix format distinguishing one run's log file from the next, in UTC."""

BACKEND_NAMES = frozenset({'nav', 'backplanes', 'ck', 'reproj'})
"""Per-image backends, each owning a subtree of the log root."""

_OFF = 'NONE'
SILENT_LEVEL = LOG_LEVEL_VALUES[_OFF]
"""Numeric level suppressing every record, for a module configured ``NONE``.

pdslogger has no name for "off", so a section that must emit nothing is opened
with this number rather than a level name.
"""

_CATEGORY_DEFAULT_KEY = 'default'
_PROGRAMS_KEY = 'programs'


@dataclass(frozen=True)
class LogLevels:
    """Resolved level for each logger and each module that overrides it.

    Parameters:
        main: Level for the main logger.
        image: Level for the image logger, and for any image-scoped module
            with no override of its own.
        modules: Level per module key, for the modules that override the
            image level.  Keys are the snake_case names validated by
            :mod:`spindoctor.config.logging_keys`.
    """

    main: str = DEFAULT_LEVEL
    image: str = DEFAULT_LEVEL
    modules: dict[str, str] = field(default_factory=dict)

    def for_module(self, log_key: str) -> str:
        """Return the level name governing ``log_key``.

        Parameters:
            log_key: A module's snake_case configuration key.

        Returns:
            The module's own level if it has one, else the image level.
        """
        return self.modules.get(log_key, self.image)

    def section_level_for(self, log_key: str) -> str | int:
        """Return the value to pass as ``logger.open(level=...)`` for a module.

        pdslogger has no level named ``NONE``, so a module configured off is
        expressed as :data:`SILENT_LEVEL` instead of a name it would reject.

        Parameters:
            log_key: A module's snake_case configuration key.

        Returns:
            A pdslogger level name, or :data:`SILENT_LEVEL` for a module
            configured ``NONE``.
        """
        level = self.for_module(log_key)
        return SILENT_LEVEL if level == _OFF else level.lower()

    def main_section_level(self) -> str | int:
        """Return the value to pass as ``logger.open(level=...)`` for the run.

        The main-logger counterpart of :meth:`image_section_level`.  ``NONE`` is
        a level the configuration accepts and pdslogger rejects by name, so it
        is expressed as :data:`SILENT_LEVEL` here too.

        Returns:
            A pdslogger level name, or :data:`SILENT_LEVEL` when the main
            logger is configured ``NONE``.
        """
        return SILENT_LEVEL if self.main == _OFF else self.main.lower()

    def image_section_level(self) -> str | int:
        """Return the value to pass as ``logger.open(level=...)`` for an image.

        The counterpart of :meth:`section_level_for` for the section covering a
        whole image rather than one module inside it.  pdslogger has no level
        named ``NONE`` and rejects the name outright, so an image logger
        configured off is expressed as :data:`SILENT_LEVEL` instead.

        Returns:
            A pdslogger level name, or :data:`SILENT_LEVEL` when the image
            logger is configured ``NONE``.
        """
        return SILENT_LEVEL if self.image == _OFF else self.image.lower()

    def most_verbose_image_level(self) -> str:
        """Return the least severe level any image-scoped module can request.

        Image handlers are built at this level rather than at :attr:`image`,
        because a handler filters after the per-section floor: one built at
        :attr:`image` would discard every record from a module configured more
        verbose than it, while the section summary still counted them.

        Returns:
            The level name to build the image sinks at.
        """
        candidates = [self.image, *self.modules.values()]
        return min(candidates, key=lambda name: LOG_LEVEL_VALUES[name])


@dataclass(frozen=True)
class LogSinks:
    """Which sinks each logger writes to, and where files are written.

    Parameters:
        log_root: Root directory for every log file this run produces.
        main_console: Whether the main logger writes to stdout.
        main_file: Whether the main logger writes to a file.
        image_console: Whether the image logger writes to stdout.
        image_file: Whether the image logger writes to a file.
    """

    log_root: FCPath
    main_console: bool = True
    main_file: bool = True
    image_console: bool = False
    image_file: bool = True


def _level_or_none(value: Any, location: str) -> str | None:
    """Return a canonicalized level name, or None when nothing was configured.

    Parameters:
        value: A configured or command-line level, possibly absent.
        location: What supplied the value, used in the error message.

    Returns:
        The canonical level name, or None if ``value`` is absent.

    Raises:
        ValueError: If ``value`` names no known level.  The configuration side
            is checked at load; this catches the command-line side, which
            would otherwise reach pdslogger as a bare ``KeyError``.
    """
    if value is None:
        return None
    level = normalize_level(str(value))
    if level not in LOG_LEVEL_NAMES:
        raise ValueError(f'{location} is {value!r}; expected one of {sorted(LOG_LEVEL_NAMES)}')
    return level


def _first_set(*candidates: str | None) -> str:
    """Return the first candidate that was actually supplied.

    Written as an explicit None test rather than an ``or`` chain so that a
    value which happens to be falsy is not skipped in favor of a less specific
    source.

    Parameters:
        *candidates: Levels in order of decreasing precedence, the last of
            which must be a fallback rather than None.

    Returns:
        The first non-None candidate.
    """
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return DEFAULT_LEVEL


def _merged_logging_block(program_name: str, config: 'Config') -> dict[str, Any]:
    """Return the logging configuration in force for one program.

    The top-level block is merged with that program's block under
    ``programs``, the program's value winning key by key.  Categories merge a
    level deeper, so a program can override one module without restating the
    rest of the category.

    Parameters:
        program_name: The program's identity, as declared by its dispatch
            module.
        config: The loaded configuration.

    Returns:
        A mapping in the shape of the top-level ``logging`` block, with the
        ``programs`` key removed.
    """
    section = config.logging
    block: dict[str, Any] = {k: v for k, v in dict(section).items() if k != _PROGRAMS_KEY}
    for key in CATEGORY_KEYS:
        if isinstance(block.get(key), dict):
            block[key] = dict(block[key])

    overrides = dict(section).get(_PROGRAMS_KEY) or {}
    program_block = overrides.get(program_name) or {}
    for key, value in dict(program_block).items():
        if key in CATEGORY_KEYS and isinstance(value, dict):
            merged = dict(block.get(key) or {})
            merged.update(dict(value))
            block[key] = merged
        else:
            block[key] = value
    return block


def resolve_log_levels(
    program_name: str,
    arguments: argparse.Namespace | None,
    config: 'Config',
) -> LogLevels:
    """Resolve the level governing each logger and each overriding module.

    Applies the merge and precedence described in the module docstring.  A
    module named on the command line beats one named in the configuration; a
    module named anywhere beats its category default; a category default beats
    the per-logger default.

    Parameters:
        program_name: The program's identity, used to select its block under
            ``logging.programs``.
        arguments: Parsed command-line arguments, or None to resolve from the
            configuration alone.  Recognized attributes are ``log_level``,
            ``log_level_main``, and ``log_level_image``.
        config: The loaded configuration.

    Returns:
        The resolved :class:`LogLevels`.
    """
    block = _merged_logging_block(program_name, config)

    cli_global, cli_modules = _parse_log_level_arguments(arguments)
    cli_main = _level_or_none(getattr(arguments, 'log_level_main', None), '--log-level-main')
    cli_image = _level_or_none(getattr(arguments, 'log_level_image', None), '--log-level-image')

    main = _first_set(
        cli_main, cli_global, _level_or_none(block.get('main'), 'logging.main'), DEFAULT_LEVEL
    )
    image = _first_set(
        cli_image, cli_global, _level_or_none(block.get('image'), 'logging.image'), DEFAULT_LEVEL
    )

    modules: dict[str, str] = {}
    for category in CATEGORY_KEYS:
        entries = block.get(category)
        if not isinstance(entries, dict):
            continue
        category_default = _level_or_none(
            entries.get(_CATEGORY_DEFAULT_KEY), f'logging.{category}.default'
        )
        for key, value in entries.items():
            if key == _CATEGORY_DEFAULT_KEY:
                continue
            level = _level_or_none(value, f'logging.{category}.{key}')
            if level is not None:
                modules[key] = level
        if category_default is not None:
            # Applies to every module in the category that did not name itself.
            # Recorded per known key rather than as a category-wide entry so
            # LogLevels stays a flat lookup.
            for key in _category_member_keys(category):
                modules.setdefault(key, category_default)

    modules.update(cli_modules)
    return LogLevels(main=main, image=image, modules=modules)


def _all_module_keys() -> frozenset[str]:
    """Return every module key any category accepts.

    Returns:
        The union of the technique, model, and other key sets.
    """
    return frozenset().union(*(_category_member_keys(name) for name in CATEGORY_KEYS))


def _category_member_keys(category: str) -> frozenset[str]:
    """Return every module key belonging to ``category``.

    Parameters:
        category: One of the categories in
            :data:`spindoctor.config.logging_keys.CATEGORY_KEYS`.

    Returns:
        The category's valid module keys.
    """
    from .logging_keys import OTHER_LOG_KEYS, model_log_keys, technique_log_keys

    if category == 'techniques':
        return technique_log_keys()
    if category == 'models':
        return model_log_keys()
    return OTHER_LOG_KEYS


def _parse_log_level_arguments(
    arguments: argparse.Namespace | None,
) -> tuple[str | None, dict[str, str]]:
    """Split the repeatable ``--log-level`` values into global and per-module.

    A bare ``LEVEL`` sets the default for both loggers; a ``MODULE=LEVEL`` pair
    sets one module.  The last bare value wins, so a later flag overrides an
    earlier one rather than silently combining.

    Parameters:
        arguments: Parsed command-line arguments, or None.

    Returns:
        Tuple of the global level (or None) and a mapping of module key to
        level.
    """
    values = getattr(arguments, 'log_level', None) or []
    if isinstance(values, str):
        values = [values]

    known_keys = _all_module_keys()
    global_level: str | None = None
    modules: dict[str, str] = {}
    for value in values:
        text = str(value)
        if '=' in text:
            key, _, level = text.partition('=')
            key = key.strip()
            if not key:
                raise ValueError(f'--log-level {text!r} names no module before the "="')
            if key not in known_keys:
                raise ValueError(
                    f'--log-level {text!r} names unknown module {key!r}; '
                    f'expected one of {sorted(known_keys)}'
                )
            resolved = _level_or_none(level, f'--log-level {key}')
            if resolved is None:
                raise ValueError(f'--log-level {text!r} names no level after the "="')
            modules[key] = resolved
        else:
            resolved = _level_or_none(text, '--log-level')
            if resolved is not None:
                global_level = resolved
    return global_level, modules


_active_levels: LogLevels | None = None


def set_log_levels(levels: LogLevels | None) -> None:
    """Install the levels every component resolves its own level from.

    A driver resolves levels once and installs them here, so a component deep
    in the pipeline can ask for its own level without the resolved set being
    threaded through every constructor.

    This is process state, and cloud-task workers are spawned rather than
    forked, so a worker does not inherit what its parent installed.  A
    cloud-task driver installs inside the task it is processing, not once in
    the parent.

    Parameters:
        levels: The resolved levels, or None to fall back to resolving the
            configuration's global defaults on next use.
    """
    global _active_levels
    _active_levels = levels


def log_levels() -> LogLevels:
    """Return the levels in force.

    Falls back to the configuration's global defaults when no driver has
    installed a resolved set, so a component consulted outside a configured
    run still gets the shipped levels rather than nothing.  The fallback is
    memoized; :func:`set_log_levels` with None discards it.

    Returns:
        The active :class:`LogLevels`.
    """
    global _active_levels
    if _active_levels is None:
        from .config import DEFAULT_CONFIG

        _active_levels = resolve_log_levels('', None, DEFAULT_CONFIG)
    return _active_levels


def main_log_path(log_root: FCPath, program_name: str, *, timestamp: str) -> FCPath:
    """Return the path of a program run's main log file.

    Parameters:
        log_root: Root directory for this run's logs.
        program_name: The program's identity, which names its directory.
        timestamp: Run timestamp in :data:`LOG_TIMESTAMP_FORMAT`.

    Returns:
        ``{log_root}/{program_name}/main_{timestamp}.log``.
    """
    return log_root / program_name / f'main_{timestamp}.log'


def _validated_stub(results_path_stub: str) -> str:
    """Return ``results_path_stub`` if it names a location inside the log root.

    A stub reaches this from task data on the cloud-task path, so it is not
    necessarily trustworthy.  Subdirectories are legitimate -- a stub is
    normally ``{volume}/{filespec}`` -- but a stub that climbs out of the log
    root, names an absolute path, or carries a null byte is not, and would put
    a log file somewhere the caller never asked for.

    Parameters:
        results_path_stub: The image's results path stub.

    Returns:
        The stub, unchanged.

    Raises:
        ValueError: If the stub is absolute, empty, contains a null byte, or
            has any parent-directory component.
    """
    if not results_path_stub or '\x00' in results_path_stub:
        raise ValueError(f'Invalid results path stub for a log file: {results_path_stub!r}')
    normalized = results_path_stub.replace('\\', '/')
    if normalized.startswith('/') or PurePosixPath(normalized).is_absolute():
        raise ValueError(f'Results path stub must be relative, got {results_path_stub!r}')
    if any(part == '..' for part in PurePosixPath(normalized).parts):
        raise ValueError(
            f'Results path stub must stay within the log root, got {results_path_stub!r}'
        )
    return results_path_stub


def image_log_path(
    log_root: FCPath, backend: str, results_path_stub: str, *, timestamp: str
) -> FCPath:
    """Return the path of one image's log file for one backend.

    The subtree is keyed by backend rather than by program, so an image's log
    for a given stage is in one place whichever driver produced it.

    Parameters:
        log_root: Root directory for this run's logs.
        backend: One of :data:`BACKEND_NAMES`.
        results_path_stub: The image's results path stub.
        timestamp: Run timestamp in :data:`LOG_TIMESTAMP_FORMAT`.

    Returns:
        ``{log_root}/{backend}/{results_path_stub}_{timestamp}.log``.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    if backend not in BACKEND_NAMES:
        raise ValueError(f'Unknown backend {backend!r}; expected one of {sorted(BACKEND_NAMES)}')
    return log_root / backend / f'{_validated_stub(results_path_stub)}_{timestamp}.log'


def run_timestamp() -> str:
    """Return the current UTC time formatted for a log file name.

    UTC rather than local time so that log names sort chronologically and
    correlate across machines: cloud-task workers processing one batch may sit
    in different zones, and a local-time name is ambiguous across a
    daylight-saving fall-back.

    Returns:
        The timestamp in :data:`LOG_TIMESTAMP_FORMAT`.
    """
    return datetime.now(UTC).strftime(LOG_TIMESTAMP_FORMAT)


def _handlers_for(*, console: bool, to_file: bool, path: FCPath | None, level: str) -> list[Any]:
    """Build the handler list for one logger.

    Parameters:
        console: Whether to attach a stdout handler.
        to_file: Whether to attach a file handler.
        path: Destination for the file handler; required when ``to_file``.
        level: Level shared by both sinks.

    Returns:
        The handlers to attach.  Never empty: with no sink enabled the list is
        ``[NULL_HANDLER]``, because a PdsLogger with no handlers prints every
        record to stdout regardless of level.

    Raises:
        ValueError: If ``to_file`` is set without a ``path``.
    """
    handlers: list[logging.Handler] = []
    if level != _OFF:
        if console:
            handlers.append(pdslogger.stream_handler(level=level.lower()))
        if to_file:
            if path is None:
                raise ValueError('A file sink was requested without a destination path')
            # No mkdir here: pdslogger.file_handler creates missing parents
            # itself, and FCPath.mkdir raises NotImplementedError on a remote
            # root, which is exactly the cloud-task configuration.
            handlers.append(pdslogger.file_handler(path, level=level.lower()))
    if not handlers:
        handlers.append(pdslogger.NULL_HANDLER)
    return handlers


def build_main_logger(
    logger: PdsLogger,
    program_name: str,
    sinks: LogSinks,
    levels: LogLevels,
    *,
    timestamp: str | None = None,
) -> FCPath | None:
    """Attach this run's sinks to the main logger.

    Any handlers already present are removed first, so calling this twice does
    not duplicate output.

    Parameters:
        logger: The main logger to configure.
        program_name: The program's identity, which names its log directory.
        sinks: Which sinks to attach and where files go.
        levels: Resolved levels; the main level applies to both sinks.
        timestamp: Run timestamp for the file name; defaults to now.

    Returns:
        The path written to, or None when no file sink is enabled.
    """
    # remove_all_handlers detaches but does not close, so a rebuild would leak
    # the previous run's open log file.  NULL_HANDLER is a process-wide
    # singleton this module does not own, so it is left alone.
    for existing in list(logger.handlers):
        if existing is not pdslogger.NULL_HANDLER:
            existing.close()
    logger.remove_all_handlers()
    stamp = timestamp if timestamp is not None else run_timestamp()
    writes_a_file = sinks.main_file and levels.main != _OFF
    path = main_log_path(sinks.log_root, program_name, timestamp=stamp) if writes_a_file else None
    for handler in _handlers_for(
        console=sinks.main_console, to_file=sinks.main_file, path=path, level=levels.main
    ):
        logger.add_handler(handler)
    return path


def build_image_log_handlers(
    backend: str,
    results_path_stub: str,
    sinks: LogSinks,
    levels: LogLevels,
    *,
    timestamp: str | None = None,
) -> tuple[list[Any], FCPath | None]:
    """Build the handlers for one image, to be attached for that image only.

    The handlers are returned rather than attached because the image logger
    scopes them to a single ``logger.open(...)`` section.  Closing the section
    detaches them but does not close them, so the caller owns disposal and must
    close each handler once the image is done.

    Parameters:
        backend: The per-image backend, one of :data:`BACKEND_NAMES`.
        results_path_stub: The image's results path stub.
        sinks: Which sinks to attach and where files go.
        levels: Resolved levels; the image level applies to both sinks.
        timestamp: Run timestamp for the file name; defaults to now.

    Returns:
        Tuple of the handlers and the path written to, the latter None when no
        file sink is enabled.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    # Validated up front so a typo fails the same way whether or not a file
    # sink happens to be enabled.
    if backend not in BACKEND_NAMES:
        raise ValueError(f'Unknown backend {backend!r}; expected one of {sorted(BACKEND_NAMES)}')
    stamp = timestamp if timestamp is not None else run_timestamp()
    # Built at the most verbose level any module can ask for; the per-section
    # floor set from LogLevels.section_level_for then does the discrimination.
    handler_level = levels.most_verbose_image_level()
    writes_a_file = sinks.image_file and handler_level != _OFF
    path = (
        image_log_path(sinks.log_root, backend, results_path_stub, timestamp=stamp)
        if writes_a_file
        else None
    )
    handlers = _handlers_for(
        console=sinks.image_console, to_file=sinks.image_file, path=path, level=handler_level
    )
    return handlers, path


def absolute_log_root(log_root: FCPath) -> FCPath:
    """Return a local log root pinned to the working directory it names.

    A log root outlives the moment it is chosen: every per-image handler is
    built from it, one at a time, for as long as the run lasts.  pdslogger
    identifies an open log file by the absolute path its own working directory
    gives, so a relative root that outlives a change of working directory names
    one file when a handler is attached and another when it is detached -- and
    the handler is then never let go, leaving the logger writing into a file
    nobody is reading.  Resolving the root once, here, removes the dependence.

    A remote root is returned unchanged: it is already absolute, and there is
    no local working directory to resolve it against.

    Parameters:
        log_root: The log root as it was given.

    Returns:
        The same root, absolute if it is local.
    """
    if not log_root.is_local():
        return log_root
    # Path rather than FCPath.get_local_path: for a local path the latter
    # creates the parent directory, and resolving a root must not write
    # anything.
    return FCPath(Path(log_root.as_posix()).expanduser().resolve())


def sinks_from_arguments(
    arguments: argparse.Namespace | None,
    log_root: FCPath,
    *,
    program_name: str = '',
    config: 'Config | None' = None,
) -> LogSinks:
    """Build the sink selection from the command line and the configuration.

    Whether each logger reaches the terminal is settable in both places, so
    that a machine which should never chatter, or an operator who always wants
    to watch, can say so once.  The command line wins, being the more specific
    statement of the two; between the configuration's own levels, a
    ``programs`` block wins over the top-level value exactly as it does for a
    level.

    Whether each logger writes to a *file* is command-line only, because it is
    inseparable from where the file goes, and that is a per-run decision.

    This is where a log root becomes the one every handler is built from, so it
    is also where a local one is made absolute; see :func:`absolute_log_root`.

    Parameters:
        arguments: Parsed command-line arguments, or None for the defaults.
        log_root: Resolved log root for this run, absolute or not.
        program_name: The program's identity, for selecting its block under
            ``logging.programs``.  Empty selects the global block only.
        config: The loaded configuration, or None to consult the command line
            and the built-in defaults alone.

    Returns:
        The resolved :class:`LogSinks`, whose log root is absolute if it is
        local.
    """
    root = absolute_log_root(log_root)
    defaults = LogSinks(log_root=root)
    block = _merged_logging_block(program_name, config) if config is not None else {}

    def console(argument_name: str, config_key: str, default: bool) -> bool:
        from_arguments = getattr(arguments, argument_name, None)
        if from_arguments is not None:
            return bool(from_arguments)
        configured = block.get(config_key)
        return default if configured is None else bool(configured)

    def flag(name: str, default: bool) -> bool:
        value = getattr(arguments, name, None)
        return default if value is None else bool(value)

    return LogSinks(
        log_root=root,
        main_console=console('log_main_to_console', 'main_console', defaults.main_console),
        main_file=flag('log_main_to_file', defaults.main_file),
        image_console=console('log_image_to_console', 'image_console', defaults.image_console),
        image_file=flag('log_image_to_file', defaults.image_file),
    )


@dataclass(frozen=True)
class RunLogging:
    """What a run's logging resolved to, for the driver to pass on.

    Parameters:
        levels: The resolved levels, already installed for components to read.
        sinks: Which sinks are enabled, and the log root.
        timestamp: One stamp for the whole run, so a run's log files share it.
        main_log_path: Where the main log is being written, or None when the
            main logger has no file sink.
    """

    levels: LogLevels
    sinks: LogSinks
    timestamp: str
    main_log_path: FCPath | None


def run_logging_for_root(log_root: str | Path | FCPath, program_name: str = '') -> RunLogging:
    """Resolve logging against an explicit log root, with no arguments.

    For a library caller that already knows where its results go and has no
    command line to consult.  Levels come from the configuration alone, and
    are installed so that the sections a component opens are floored at the
    same levels the handlers were built at; leaving them uninstalled would let
    the two disagree, which silently drops records that the handlers would
    have accepted.

    Parameters:
        log_root: Root directory for this run's log files.
        program_name: The program's identity, for selecting its configuration
            block.  Empty selects the global block only.

    Returns:
        The resolved :class:`RunLogging`, with no main logger built.
    """
    from .config import DEFAULT_CONFIG

    root = FCPath(log_root)
    levels = resolve_log_levels(program_name, None, DEFAULT_CONFIG)
    set_log_levels(levels)
    return RunLogging(
        levels=levels,
        sinks=sinks_from_arguments(None, root, program_name=program_name, config=DEFAULT_CONFIG),
        timestamp=run_timestamp(),
        main_log_path=None,
    )


def build_run_logging(
    program_name: str,
    arguments: argparse.Namespace,
    config: 'Config',
    *,
    build_main: bool = True,
    fallback_log_root: str | Path | FCPath | None = None,
) -> RunLogging:
    """Resolve this run's logging and configure the main logger.

    Call once at startup, after the configuration has been loaded, before
    anything is logged.  The resolved levels are installed globally so a
    component deep in the pipeline can read its own level; the returned value
    carries what a driver needs to open per-image logs.

    Parameters:
        program_name: The program's identity, which selects its configuration
            block and names its main log directory.
        arguments: Parsed command-line arguments.
        config: The loaded configuration.
        build_main: False for a program that has no main logger of its own, so
            that levels and sinks are still resolved for its image logs.
        fallback_log_root: Where to put log files when nothing else names a log
            root.  For a driver that has somewhere sensible of its own -- a
            cloud task knows its output directory -- this keeps its logs rather
            than dropping them.

    Returns:
        The resolved :class:`RunLogging`.

    Raises:
        ValueError: If a level named on the command line or in the
            configuration is not a known level, or a module named on the
            command line is not a known component.
        TypeError: If a configured level is not a string.
    """
    # Local import: config_helper imports this module, so importing it back at
    # module level would close a cycle.
    from .config_helper import get_log_root

    fallback = FCPath(fallback_log_root) if fallback_log_root is not None else None
    levels = resolve_log_levels(program_name, arguments, config)
    try:
        log_root = FCPath(get_log_root(arguments, config))
    except ValueError as exc:
        if fallback is None:
            # A program with no results root of its own -- bundle summary, say
            # -- has nowhere to put log files.  That is not worth refusing to
            # run over: drop the file sinks, say so, and carry on.
            sinks = replace(
                sinks_from_arguments(
                    arguments, FCPath('.'), program_name=program_name, config=config
                ),
                main_file=False,
                image_file=False,
            )
            set_log_levels(levels)
            timestamp = run_timestamp()
            if build_main:
                build_main_logger(MAIN_LOGGER, program_name, sinks, levels, timestamp=timestamp)
            # Warned whether or not a main logger was built: a driver without
            # one still needs to know its image logs are going nowhere.
            MAIN_LOGGER.warning(
                'No log root could be determined (%s); logging to the terminal only. '
                'Pass --log-root to write log files.',
                exc,
            )
            return RunLogging(levels=levels, sinks=sinks, timestamp=timestamp, main_log_path=None)
        log_root = fallback
    sinks = sinks_from_arguments(arguments, log_root, program_name=program_name, config=config)
    set_log_levels(levels)
    timestamp = run_timestamp()
    main_log_path = None
    if build_main:
        main_log_path = build_main_logger(
            MAIN_LOGGER, program_name, sinks, levels, timestamp=timestamp
        )
    return RunLogging(levels=levels, sinks=sinks, timestamp=timestamp, main_log_path=main_log_path)


def isolate_cloud_task_logging() -> None:
    """Detach both loggers from the terminal a cloud-task worker owns.

    A worker's console is the worker's own, reporting task progress under
    cloud_tasks' configuration; per-image processing detail belongs in the
    per-image log file, not interleaved with it.  Two separate paths would put
    it on the console anyway, and closing one without the other leaves the
    output there, so both are closed here.

    Neither logger may be left with no handlers, because a PdsLogger with none
    does not go quiet -- it prints every record to stdout, whatever its level.
    The main logger has no sinks of its own in a cloud task, and the image
    logger has none between one image's section and the next, so both are bound
    to the null handler and the call sites that remain are inert instead.

    Call this outside any open image section.  It clears both loggers, and a
    section's handlers are attached to the image logger for as long as that
    section is open, so clearing one mid-image would discard the handlers that
    image's log is being written through.

    Both loggers otherwise propagate to the root logger, and cloud_tasks calls
    ``logging.basicConfig`` inside each worker subprocess, which puts a handler
    there.  Every record would be emitted a second time on stderr, formatted by
    the root handler rather than by pdslogger.  Turning propagation off ends
    the record at our own handlers.

    Call before anything is logged, and inside the task rather than in the
    parent: workers are spawned rather than forked, so a worker does not
    inherit what the parent configured.
    """
    # Imported here rather than at module level: log_scope imports this module
    # for the levels a section is opened at, so importing it back would close a
    # cycle.
    from .log_scope import IMAGE_LOGGER

    for logger in (MAIN_LOGGER, IMAGE_LOGGER):
        # Detached but not closed by remove_all_handlers, so anything real is
        # closed first; NULL_HANDLER is a process-wide singleton this module
        # does not own.  Both loggers are cleared, not just the main one: a
        # console handler left on either from an earlier configuration would
        # keep reaching the terminal, which turning propagation off does not
        # prevent.
        for existing in list(logger.handlers):
            if existing is not pdslogger.NULL_HANDLER:
                existing.close()
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)
        logger.propagate = False


def build_cloud_task_logging(
    program_name: str,
    arguments: argparse.Namespace,
    config: 'Config',
    *,
    fallback_log_root: str | Path | FCPath | None = None,
) -> RunLogging:
    """Resolve logging for one cloud task, isolated from the worker's console.

    The cloud-task counterpart to :func:`build_run_logging`: levels resolve
    identically, so an image's log reads the same whichever driver produced it,
    but no main logger is built and the image logger cannot reach the console
    however it was configured.  Isolation is applied first, so that even a
    failure to resolve the log root is reported into the null sink rather than
    printed.

    Call inside the task rather than once in the parent; see
    :func:`isolate_cloud_task_logging`.

    Parameters:
        program_name: The program's identity, which selects its configuration
            block.
        arguments: Parsed command-line arguments.
        config: The loaded configuration.
        fallback_log_root: Where to put log files when nothing else names a log
            root.

    Returns:
        The resolved :class:`RunLogging`, whose sinks name no console.

    Raises:
        ValueError: If a level named in the configuration is not a known
            level, or a module or program named there is not known.  Isolation
            is already applied when this propagates, so a driver reporting it
            does so without reaching the worker's terminal.
        TypeError: If a configured level is not a string.
    """
    isolate_cloud_task_logging()
    run_logging = build_run_logging(
        program_name,
        arguments,
        config,
        build_main=False,
        fallback_log_root=fallback_log_root,
    )
    # Forced rather than defaulted: --log-image-to-console is a reasonable
    # request of an interactive driver and an impossible one here, and a
    # configuration file is shared between the two.
    return replace(
        run_logging,
        sinks=replace(run_logging.sinks, main_console=False, main_file=False, image_console=False),
    )
