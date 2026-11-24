"""Routing of log records to the main or the image logger.

Two loggers exist, and every component belongs to exactly one.  Components
whose work spans a run -- enumerating a dataset, tallying totals -- belong to
the main logger.  Components working on one image belong to the image logger.

The image logger is reached through :data:`IMAGE_LOGGER`, a proxy rather than
a logger.  Opening a section on it enters an image scope; while that scope is
open the proxy forwards to the real logger underneath, and every nested
section, model and technique writes into the same per-image log.  Outside any
scope there is nothing sensible to forward to, so the proxy routes the record
to the main logger and warns once about the call site.  With
``logging.strict_scope`` enabled it raises instead; that is off by default,
because one mis-scoped call should not abort a batch over a log line.

Configuring the image logger -- setting a level, attaching a handler -- is not
the same as logging to it, and is a reasonable thing to do before any image is
open.  Those calls reach the image logger directly and never report a
violation.

Logging from an image-scoped component with no image open is a defect rather
than a mode to support.  Every occurrence found while this was written was
either correctly scoped or a component bound to the wrong logger.
"""

import functools
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, ParamSpec, TypeVar

import pdslogger
from pdslogger import PdsLogger

from .logger import MAIN_LOGGER

__all__ = [
    'IMAGE_LOGGER',
    'ImageLoggerProxy',
    'LogRole',
    'LogScopeError',
    'image_log_section',
    'image_scope',
    'image_scope_is_open',
    'logged_section',
    'set_strict_scope',
    'strict_scope',
    'strict_scope_override',
]


_P = ParamSpec('_P')
_R = TypeVar('_R')


class LogRole(Enum):
    """Which logger a component's records belong to."""

    MAIN = 'main'
    """For work spanning a whole run, such as enumeration and totals."""

    IMAGE = 'image'
    """For work on one image, such as models, techniques and annotation."""


class LogScopeError(RuntimeError):
    """An image-scoped component logged with no image scope open."""


_ACTIVE_IMAGE_LOGGER: ContextVar[PdsLogger | None] = ContextVar(
    'spindoctor_active_image_logger', default=None
)

_DEFAULT_IMAGE_LOGGER = pdslogger.PdsLogger('nav_image', lognames=False, digits=3)

_strict_scope_override: bool | None = None
_reported_call_sites: set[tuple[str, str, int]] = set()


def set_strict_scope(enabled: bool | None) -> None:
    """Override whether an out-of-scope image log raises or only warns.

    Production warns, so one mis-scoped call cannot abort a batch over a log
    line.  A caller that wants the stricter behavior for a bounded stretch --
    a test driving a real pipeline, say -- turns it on here.

    Parameters:
        enabled: True to raise, False to warn, or None to defer to
            ``logging.strict_scope`` in the configuration.
    """
    global _strict_scope_override
    _strict_scope_override = enabled


def strict_scope_override() -> bool | None:
    """Return the override in force, distinct from the resolved behavior.

    :func:`strict_scope` collapses "no override" into the configured value, so
    a caller saving and restoring state must read the override itself; saving
    the resolved boolean would pin it and lose the deferral.

    Returns:
        The override, or None when behavior defers to the configuration.
    """
    return _strict_scope_override


def strict_scope() -> bool:
    """Whether an out-of-scope image log currently raises.

    Falls back to ``logging.strict_scope`` in the configuration when no
    override is in force, so the shipped key is what actually governs
    behavior rather than being a setting nothing reads.

    Returns:
        True when strict scope is enabled.
    """
    if _strict_scope_override is not None:
        return _strict_scope_override
    from .config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.logging.strict_scope


def image_scope_is_open() -> bool:
    """Whether an image scope is currently open on this task.

    Returns:
        True when :func:`image_scope` is active.
    """
    return _ACTIVE_IMAGE_LOGGER.get() is not None


def _caller_site() -> tuple[str, str, int]:
    """Identify the first frame outside this module.

    Returns:
        Tuple of module name, function name, and line number of the code that
        logged, used both to name the offender and to deduplicate the warning.
    """
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get('__name__') == __name__:
        frame = frame.f_back  # type: ignore[assignment]
    if frame is None:  # pragma: no cover - a frame outside this module always exists
        return ('<unknown>', '<unknown>', 0)
    return (frame.f_globals.get('__name__', '<unknown>'), frame.f_code.co_name, frame.f_lineno)


def _report_out_of_scope() -> None:
    """Handle an image-scoped record logged with no image scope open.

    Raises:
        LogScopeError: When strict scope is enabled.
    """
    module, function, line = _caller_site()
    where = f'{module}.{function} (line {line})'
    if strict_scope():
        raise LogScopeError(
            f'{where} logged to the image logger with no image scope open. '
            f'A component scoped to one image must log inside an image scope; '
            f'if its work spans the run, bind it to the main logger instead.'
        )
    site = (module, function, line)
    if site not in _reported_call_sites:
        _reported_call_sites.add(site)
        MAIN_LOGGER.warning(
            'Log routed to the main logger: %s logged to the image logger with no '
            'image scope open. Reported once per call site.',
            where,
        )


def _reset_reported_call_sites() -> None:
    """Forget which call sites have been warned about.

    Exposed for tests, which need each case to warn independently rather than
    inheriting the deduplication state of an earlier one.
    """
    _reported_call_sites.clear()


@contextmanager
def image_scope(logger: PdsLogger | None = None) -> Iterator[PdsLogger]:
    """Make ``logger`` the image logger for the duration of the block.

    Nesting is a no-op: an inner scope keeps the logger the outer one
    established, so a model or technique opening its own section does not
    displace the image it is running under.

    Parameters:
        logger: The real logger to route image records to.  None uses the
            default image logger.

    Yields:
        The logger image records will reach.
    """
    active = _ACTIVE_IMAGE_LOGGER.get()
    if active is not None:
        yield active
        return
    target = logger if logger is not None else _DEFAULT_IMAGE_LOGGER
    token = _ACTIVE_IMAGE_LOGGER.set(target)
    try:
        yield target
    finally:
        _ACTIVE_IMAGE_LOGGER.reset(token)


def image_log_section(log_key: str, title: str, **kwargs: Any) -> Any:
    """Open an image-log section at the level configured for ``log_key``.

    The counterpart to :meth:`spindoctor.support.nav_base.NavBase.log_section`
    for components that are module-level functions rather than classes.  A
    component needs a section of its own to be independently configurable at
    all, because a level is applied when a section is opened; without one its
    records take the level of whatever section encloses them.

    Parameters:
        log_key: The component's snake_case configuration key.
        title: Title of the section.
        **kwargs: Passed to ``PdsLogger.open``.  An explicit ``level``
            overrides the configured one.

    Returns:
        The context manager returned by ``PdsLogger.open``.
    """
    from .logging_config import log_levels

    if 'level' not in kwargs:
        kwargs['level'] = log_levels().section_level_for(log_key)
    return IMAGE_LOGGER.open(title, **kwargs)


def logged_section(log_key: str, title: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a function so its records land in a section of its own.

    A component is only independently configurable if it opens a section,
    because that is where a level is applied.  This decorator gives one to a
    component that is a module-level function rather than a class, without
    reshaping the function around a ``with`` block.

    Parameters:
        log_key: The component's snake_case configuration key.
        title: Title of the section.

    Returns:
        A decorator opening the section around each call.
    """

    def decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with image_log_section(log_key, title):
                return func(*args, **kwargs)

        return wrapper

    return decorate


class ImageLoggerProxy:
    """Forwards image-scoped records to whichever logger the scope selected.

    Presents the part of the ``PdsLogger`` interface SpinDoctor uses, so a
    component can hold this object wherever it would have held a logger.
    Records arriving outside an image scope are routed to the main logger and
    reported; see the module docstring.
    """

    def _target(self) -> PdsLogger:
        """Return the logger this record should reach.

        Returns:
            The scope's logger, or the main logger when none is open.
        """
        active = _ACTIVE_IMAGE_LOGGER.get()
        if active is not None:
            return active
        _report_out_of_scope()
        return MAIN_LOGGER

    def _configured(self) -> PdsLogger:
        """Return the logger that configuration should act on.

        Configuring the image logger is not the same as logging to it.  A
        caller adjusting a level or attaching a handler is preparing the image
        logger, which is a sensible thing to do before any image is open, so
        this resolves to the default image logger rather than reporting a
        scope violation and redirecting the change onto the main logger --
        where it would silently fail to have the intended effect.

        Returns:
            The scope's logger if one is open, else the default image logger.
        """
        active = _ACTIVE_IMAGE_LOGGER.get()
        return active if active is not None else _DEFAULT_IMAGE_LOGGER

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at DEBUG. See :meth:`pdslogger.PdsLogger.debug`."""
        self._target().debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at INFO. See :meth:`pdslogger.PdsLogger.info`."""
        self._target().info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at WARNING. See :meth:`pdslogger.PdsLogger.warning`."""
        self._target().warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR. See :meth:`pdslogger.PdsLogger.error`."""
        self._target().error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at CRITICAL. See :meth:`pdslogger.PdsLogger.critical`."""
        self._target().critical(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with its stack trace.

        See :meth:`pdslogger.PdsLogger.exception`.
        """
        self._target().exception(message, *args, **kwargs)

    def log(self, level: Any, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at an explicit level. See :meth:`pdslogger.PdsLogger.log`."""
        self._target().log(level, message, *args, **kwargs)

    def set_level(self, level: Any) -> None:
        """Set the level of the current tier. See ``PdsLogger.set_level``."""
        self._configured().set_level(level)

    def add_handler(self, *handlers: Any, **kwargs: Any) -> None:
        """Attach handlers. See ``PdsLogger.add_handler``."""
        self._configured().add_handler(*handlers, **kwargs)

    def remove_handler(self, *handlers: Any) -> None:
        """Detach handlers. See ``PdsLogger.remove_handler``."""
        self._configured().remove_handler(*handlers)

    def remove_all_handlers(self) -> None:
        """Detach every handler. See ``PdsLogger.remove_all_handlers``."""
        self._configured().remove_all_handlers()

    def replace_handler(self, *handlers: Any, **kwargs: Any) -> None:
        """Replace the attached handlers. See ``PdsLogger.replace_handler``."""
        self._configured().replace_handler(*handlers, **kwargs)

    @property
    def level(self) -> Any:
        """The minimum level of the logger being configured."""
        return self._configured().level

    @property
    def propagate(self) -> Any:
        """Whether records also reach the ancestors of the underlying logger."""
        return self._configured().propagate

    @propagate.setter
    def propagate(self, value: Any) -> None:
        """Set whether records also reach the ancestors of the underlying logger.

        Propagation is a property of the logger rather than of a record, so it
        is configuration and resolves like the rest of it.  Turning it off
        matters where something outside SpinDoctor has attached a handler to
        the root logger, which would otherwise re-emit every record a second
        time on its own terms.

        Parameters:
            value: True to propagate to ancestor loggers, False to stop here.
        """
        self._configured().propagate = value

    @property
    def handlers(self) -> Any:
        """The handlers of the logger being configured.

        Reading this is introspection rather than logging, so it does not
        report a scope violation; a diagnostic must not be able to raise.
        """
        return self._configured().handlers

    def __getattr__(self, name: str) -> Any:
        """Reject a PdsLogger member this proxy does not implement.

        The proxy covers the interface SpinDoctor uses rather than all of
        ``PdsLogger``.  Failing here names the missing member and says why,
        instead of surfacing a bare AttributeError from an unexpected place.

        Parameters:
            name: The attribute that was requested.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(
            f'{type(self).__name__} does not implement {name!r}. The image logger is a '
            f'proxy over whichever image scope is open, and forwards only the interface '
            f'SpinDoctor uses; add {name!r} to it if it is genuinely needed.'
        )

    @contextmanager
    def open(self, title: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Open a section, entering an image scope if none is open.

        Opening a section on the image logger is what establishes an image
        scope, so the outermost such call selects the logger every nested
        section, model and technique will write to.  A nested call passes
        straight through to the scope's logger.

        Parameters:
            title: Title of the section.
            *args: Passed to ``PdsLogger.open``.
            **kwargs: Passed to ``PdsLogger.open``.

        Yields:
            Whatever ``PdsLogger.open`` yields.
        """
        with image_scope() as target, target.open(title, *args, **kwargs) as section:
            yield section


IMAGE_LOGGER = ImageLoggerProxy()
"""A proxy resolving to whichever image scope is currently open."""
