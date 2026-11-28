"""Tests for logging level resolution, sink selection, and logger construction."""

import argparse
import itertools
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pdslogger
import pytest
from filecache import FCPath
from tests.packaged_sections import NavConfigWith

from spindoctor.config import IMAGE_LOGGER, logging_config
from spindoctor.config.config import Config
from spindoctor.config.logging_config import (
    BACKEND_NAMES,
    DEFAULT_LEVEL,
    LOG_TIMESTAMP_FORMAT,
    SILENT_LEVEL,
    LogLevels,
    LogSinks,
    absolute_log_root,
    build_cloud_task_logging,
    build_image_log_handlers,
    build_main_logger,
    build_run_logging,
    image_log_path,
    main_log_path,
    resolve_log_levels,
    run_timestamp,
    sinks_from_arguments,
)
from spindoctor.config.logging_keys import log_key_for, technique_log_keys
from spindoctor.config.program_names import SD_MOSAIC, SD_OFFSET

_STAMP = '2026-07-29T12-00-00'
_RUN_STAMP = '2099-12-31T23-59-59'


def _args(**kwargs: object) -> argparse.Namespace:
    """Build a Namespace carrying only the given logging arguments.

    Parameters:
        **kwargs: Argument names and values to set.

    Returns:
        The populated Namespace.
    """
    return argparse.Namespace(**kwargs)


def _close(handlers: list[object]) -> None:
    """Close any handlers holding an open file.

    Parameters:
        handlers: Handlers returned by a builder.
    """
    for handler in handlers:
        close = getattr(handler, 'close', None)
        if close is not None:
            close()


def _sinks(tmp_path: Path, **kwargs: bool) -> LogSinks:
    """Build a LogSinks rooted at ``tmp_path``.

    Parameters:
        tmp_path: Directory used as the log root.
        **kwargs: Sink flags to override.

    Returns:
        The constructed LogSinks.
    """
    return LogSinks(log_root=FCPath(str(tmp_path)), **kwargs)


# ---------------------------------------------------------------------------
# Level resolution: defaults and configuration
# ---------------------------------------------------------------------------


def test_defaults_resolve_to_info() -> None:
    """With nothing configured, both loggers resolve to the fallback level."""
    levels = resolve_log_levels(SD_OFFSET, None, Config())
    assert levels.main == DEFAULT_LEVEL


def test_image_default_resolves_to_info() -> None:
    """The image logger shares the fallback level."""
    levels = resolve_log_levels(SD_OFFSET, None, Config())
    assert levels.image == DEFAULT_LEVEL


def test_config_sets_the_main_level(nav_config_with: NavConfigWith) -> None:
    """A configured main level is used."""
    levels = resolve_log_levels(SD_OFFSET, None, nav_config_with({'logging': {'main': 'WARNING'}}))
    assert levels.main == 'WARNING'


def test_config_level_is_canonicalized(nav_config_with: NavConfigWith) -> None:
    """A lower-case configured level resolves to its canonical spelling."""
    levels = resolve_log_levels(SD_OFFSET, None, nav_config_with({'logging': {'main': 'warning'}}))
    assert levels.main == 'WARNING'


def test_module_override_beats_the_image_default(nav_config_with: NavConfigWith) -> None:
    """A named module takes its own level rather than the image default."""
    config = nav_config_with(
        {'logging': {'image': 'WARNING', 'techniques': {'titan_haze': 'DEBUG'}}}
    )
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_unnamed_module_takes_the_image_default(nav_config_with: NavConfigWith) -> None:
    """A module with no override of its own follows the image level."""
    config = nav_config_with(
        {'logging': {'image': 'WARNING', 'techniques': {'titan_haze': 'DEBUG'}}}
    )
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('body_limb') == 'WARNING'


def test_category_default_applies_to_its_members(nav_config_with: NavConfigWith) -> None:
    """A category default governs every module in that category."""
    config = nav_config_with({'logging': {'techniques': {'default': 'DEBUG'}}})
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('body_limb') == 'DEBUG'


def test_category_default_does_not_cross_categories(nav_config_with: NavConfigWith) -> None:
    """A technique category default leaves models alone."""
    config = nav_config_with({'logging': {'techniques': {'default': 'DEBUG'}}})
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('rings') == DEFAULT_LEVEL


def test_module_override_beats_its_category_default(nav_config_with: NavConfigWith) -> None:
    """A named module wins over the default of its own category."""
    config = nav_config_with(
        {'logging': {'techniques': {'default': 'DEBUG', 'titan_haze': 'ERROR'}}}
    )
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('titan_haze') == 'ERROR'


def test_shipped_annotation_level_survives_resolution() -> None:
    """The shipped annotation override reaches the resolved levels."""
    levels = resolve_log_levels(SD_OFFSET, None, Config())
    assert levels.for_module('annotate') == 'ERROR'


# ---------------------------------------------------------------------------
# Level resolution: the program merge
# ---------------------------------------------------------------------------


def test_program_block_overrides_the_global_default(nav_config_with: NavConfigWith) -> None:
    """A program's block wins over the top-level value for that program."""
    config = nav_config_with(
        {'logging': {'main': 'INFO', 'programs': {SD_MOSAIC: {'main': 'WARNING'}}}}
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.main == 'WARNING'


def test_program_block_does_not_affect_another_program(nav_config_with: NavConfigWith) -> None:
    """One program's block leaves every other program untouched."""
    config = nav_config_with(
        {'logging': {'main': 'INFO', 'programs': {SD_MOSAIC: {'main': 'WARNING'}}}}
    )
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.main == 'INFO'


def test_program_block_inherits_unmentioned_keys(nav_config_with: NavConfigWith) -> None:
    """Overriding one key leaves the rest of the global block in force."""
    config = nav_config_with(
        {'logging': {'image': 'WARNING', 'programs': {SD_MOSAIC: {'main': 'ERROR'}}}}
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.image == 'WARNING'


def test_program_block_merges_within_a_category(nav_config_with: NavConfigWith) -> None:
    """A program overriding one module keeps the global entries for the others."""
    config = nav_config_with(
        {
            'logging': {
                'techniques': {'titan_haze': 'DEBUG', 'body_limb': 'ERROR'},
                'programs': {SD_MOSAIC: {'techniques': {'titan_haze': 'WARNING'}}},
            }
        }
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.for_module('body_limb') == 'ERROR'


def test_program_block_wins_within_a_category(nav_config_with: NavConfigWith) -> None:
    """The program's value for a module wins over the global one."""
    config = nav_config_with(
        {
            'logging': {
                'techniques': {'titan_haze': 'DEBUG'},
                'programs': {SD_MOSAIC: {'techniques': {'titan_haze': 'WARNING'}}},
            }
        }
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.for_module('titan_haze') == 'WARNING'


# ---------------------------------------------------------------------------
# Level resolution: command line
# ---------------------------------------------------------------------------


def test_bare_log_level_sets_both_loggers() -> None:
    """A bare --log-level sets the default for the main logger."""
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), Config())
    assert levels.main == 'DEBUG'


def test_bare_log_level_sets_the_image_logger_too() -> None:
    """A bare --log-level sets the default for the image logger as well."""
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), Config())
    assert levels.image == 'DEBUG'


def test_log_level_main_beats_the_bare_form() -> None:
    """The per-logger flag is more specific than the global one."""
    args = _args(log_level=['DEBUG'], log_level_main='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.main == 'ERROR'


def test_log_level_main_leaves_the_image_logger_alone() -> None:
    """Setting the main level does not change the image level."""
    args = _args(log_level=['DEBUG'], log_level_main='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.image == 'DEBUG'


def test_module_form_sets_one_module() -> None:
    """--log-level MODULE=LEVEL sets that module."""
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_module_form_leaves_other_modules_alone() -> None:
    """A per-module flag affects only the module it names."""
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.for_module('body_limb') == DEFAULT_LEVEL


def test_module_form_beats_the_configuration(nav_config_with: NavConfigWith) -> None:
    """A module named on the command line wins over the configured value."""
    config = nav_config_with({'logging': {'techniques': {'titan_haze': 'ERROR'}}})
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, config)
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_bare_and_module_forms_combine() -> None:
    """The documented "debug everywhere except one" invocation works."""
    args = _args(log_level=['debug', 'titan_haze=info'])
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.for_module('titan_haze') == 'INFO'


def test_bare_form_still_applies_to_other_modules() -> None:
    """The bare form governs every module the per-module form did not name."""
    args = _args(log_level=['debug', 'titan_haze=info'])
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.for_module('body_limb') == 'DEBUG'


def test_a_later_bare_form_wins() -> None:
    """Repeating the bare form takes the last value rather than combining."""
    args = _args(log_level=['debug', 'error'])
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.main == 'ERROR'


def test_command_line_beats_a_program_block(nav_config_with: NavConfigWith) -> None:
    """A per-logger flag wins over a per-program configured value."""
    config = nav_config_with({'logging': {'programs': {SD_MOSAIC: {'main': 'WARNING'}}}})
    levels = resolve_log_levels(SD_MOSAIC, _args(log_level_main='ERROR'), config)
    assert levels.main == 'ERROR'


def test_configured_module_beats_a_bare_command_line_level(nav_config_with: NavConfigWith) -> None:
    """A configured module is more specific than a global command-line default."""
    config = nav_config_with({'logging': {'techniques': {'titan_haze': 'ERROR'}}})
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), config)
    assert levels.for_module('titan_haze') == 'ERROR'


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_default_sinks_put_main_on_the_console(tmp_path: Path) -> None:
    """The main logger writes to the console by default."""
    assert sinks_from_arguments(None, FCPath(str(tmp_path))).main_console is True


def test_default_sinks_keep_image_off_the_console(tmp_path: Path) -> None:
    """The image logger stays off the console by default."""
    assert sinks_from_arguments(None, FCPath(str(tmp_path))).image_console is False


def test_default_sinks_write_both_log_files(tmp_path: Path) -> None:
    """Both loggers write files by default."""
    sinks = sinks_from_arguments(None, FCPath(str(tmp_path)))
    assert (sinks.main_file, sinks.image_file) == (True, True)


@pytest.mark.parametrize(
    ('flag', 'attribute'),
    [
        ('log_main_to_console', 'main_console'),
        ('log_main_to_file', 'main_file'),
        ('log_image_to_console', 'image_console'),
        ('log_image_to_file', 'image_file'),
    ],
)
def test_each_sink_flag_is_honored(tmp_path: Path, flag: str, attribute: str) -> None:
    """Each sink flag independently controls its own sink."""
    sinks = sinks_from_arguments(_args(**{flag: True}), FCPath(str(tmp_path)))
    assert getattr(sinks, attribute) is True


@pytest.mark.parametrize(
    ('flag', 'attribute'),
    [
        ('log_main_to_console', 'main_console'),
        ('log_main_to_file', 'main_file'),
        ('log_image_to_console', 'image_console'),
        ('log_image_to_file', 'image_file'),
    ],
)
def test_each_sink_flag_can_disable(tmp_path: Path, flag: str, attribute: str) -> None:
    """Each sink flag can turn its sink off."""
    sinks = sinks_from_arguments(_args(**{flag: False}), FCPath(str(tmp_path)))
    assert getattr(sinks, attribute) is False


# ---------------------------------------------------------------------------
# A log root outlives the working directory it was typed in
# ---------------------------------------------------------------------------


def test_a_relative_log_root_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the logs go is decided once, not re-decided at every use."""
    monkeypatch.chdir(tmp_path)
    sinks = sinks_from_arguments(None, FCPath('run/logs'))
    assert sinks.log_root.as_posix() == (tmp_path.resolve() / 'run' / 'logs').as_posix()


def test_an_absolute_log_root_is_unchanged(tmp_path: Path) -> None:
    """Resolving one that is already absolute does not move it.

    The expected value is built from ``tmp_path.resolve()`` because on macOS
    ``tmp_path`` itself sits behind the ``/var`` symlink that resolution
    removes.
    """
    sinks = sinks_from_arguments(None, FCPath(str(tmp_path / 'logs')))
    assert sinks.log_root.as_posix() == (tmp_path.resolve() / 'logs').as_posix()


def test_a_remote_log_root_is_left_as_it_was_given() -> None:
    """It is already absolute, and there is no working directory to use."""
    assert absolute_log_root(FCPath('gs://bucket/logs')).as_posix() == 'gs://bucket/logs'


def test_resolving_a_log_root_creates_nothing(tmp_path: Path) -> None:
    """Deciding where logs go must not itself write to the filesystem."""
    absolute_log_root(FCPath(str(tmp_path / 'logs')))
    assert not (tmp_path / 'logs').exists()


def test_a_main_handler_detaches_after_the_directory_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pdslogger finds an open log file by absolute path, so the root must be one.

    A handler built from a relative root is registered under the absolute path
    the working directory gave at the time.  Move the directory and it can no
    longer be found, so it is never detached and the logger goes on writing
    into a file nobody reads.
    """
    logger = pdslogger.PdsLogger('detach_after_chdir', lognames=False)
    monkeypatch.chdir(tmp_path)
    sinks = sinks_from_arguments(_args(log_main_to_console=False), FCPath('run/logs'))
    build_main_logger(logger, SD_OFFSET, sinks, LogLevels(), timestamp=_STAMP)
    monkeypatch.undo()
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert logger.handlers == []


# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------


def test_main_log_path_is_under_the_program_directory(tmp_path: Path) -> None:
    """A main log lands under a directory named for its program."""
    path = main_log_path(FCPath(str(tmp_path)), SD_OFFSET, timestamp=_STAMP)
    assert path.as_posix().endswith(f'{SD_OFFSET}/main_{_STAMP}.log')


@pytest.mark.parametrize('backend', sorted(BACKEND_NAMES))
def test_image_log_path_is_under_the_backend_directory(tmp_path: Path, backend: str) -> None:
    """An image log lands under a directory named for its backend."""
    path = image_log_path(
        FCPath(str(tmp_path)), backend, 'COISS_2001/data/N123_1', timestamp=_STAMP
    )
    assert path.as_posix().endswith(f'{backend}/COISS_2001/data/N123_1_{_STAMP}.log')


def test_image_log_path_rejects_an_unknown_backend(tmp_path: Path) -> None:
    """A backend that is not one of the three is rejected."""
    with pytest.raises(ValueError, match='Unknown backend'):
        image_log_path(FCPath(str(tmp_path)), 'wibble', 'stub', timestamp=_STAMP)


def test_two_backends_do_not_collide_for_one_image(tmp_path: Path) -> None:
    """The same image logged by two backends lands in two distinct files."""
    root = FCPath(str(tmp_path))
    nav = image_log_path(root, 'nav', 'vol/N123', timestamp=_STAMP)
    reproj = image_log_path(root, 'reproj', 'vol/N123', timestamp=_STAMP)
    assert nav.as_posix() != reproj.as_posix()


# ---------------------------------------------------------------------------
# Logger construction
# ---------------------------------------------------------------------------


def test_main_logger_gets_both_handlers(tmp_path: Path) -> None:
    """With both sinks enabled the main logger carries two handlers."""
    logger = pdslogger.PdsLogger('test_main_both', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    count = len(logger.handlers)
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert count == 2


def test_main_logger_writes_to_the_reported_path(tmp_path: Path) -> None:
    """The returned path is the file the main logger actually writes."""
    logger = pdslogger.PdsLogger('test_main_path', lognames=False)
    path = build_main_logger(
        logger, SD_OFFSET, _sinks(tmp_path, main_console=False), LogLevels(), timestamp=_STAMP
    )
    logger.info('a line')
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert path is not None
    with path.open('r') as handle:
        assert 'a line' in handle.read()


def test_main_logger_reports_no_path_without_a_file_sink(tmp_path: Path) -> None:
    """Disabling the file sink reports no path."""
    logger = pdslogger.PdsLogger('test_main_nofile', lognames=False)
    path = build_main_logger(
        logger, SD_OFFSET, _sinks(tmp_path, main_file=False), LogLevels(), timestamp=_STAMP
    )
    assert path is None


def test_rebuilding_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Building twice replaces the handlers rather than accumulating them."""
    logger = pdslogger.PdsLogger('test_main_twice', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    count = len(logger.handlers)
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert count == 2


def test_disabling_every_main_sink_attaches_the_null_handler(tmp_path: Path) -> None:
    """A logger with no sink still has a handler, so it cannot fall back to printing."""
    logger = pdslogger.PdsLogger('test_main_null', lognames=False)
    build_main_logger(
        logger,
        SD_OFFSET,
        _sinks(tmp_path, main_console=False, main_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_a_silenced_main_logger_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With every sink off, logging produces no terminal output at all."""
    logger = pdslogger.PdsLogger('test_main_quiet', lognames=False)
    build_main_logger(
        logger,
        SD_OFFSET,
        _sinks(tmp_path, main_console=False, main_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    logger.info('should not appear')
    logger.error('should not appear either')
    assert capsys.readouterr().out == ''


def test_level_none_silences_the_logger(tmp_path: Path) -> None:
    """A level of NONE attaches the null handler rather than a live sink."""
    logger = pdslogger.PdsLogger('test_main_none', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(main='NONE'), timestamp=_STAMP)
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_image_handlers_are_built_for_the_backend(tmp_path: Path) -> None:
    """Image handlers report the backend-scoped path they will write."""
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N123', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
    )
    _close(handlers)
    assert path is not None
    assert '/nav/' in path.as_posix()


def test_image_handlers_default_to_file_only(tmp_path: Path) -> None:
    """With default sinks the image logger gets exactly one handler."""
    handlers, _ = build_image_log_handlers(
        'nav', 'vol/N123', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
    )
    _close(handlers)
    assert len(handlers) == 1


def test_image_handlers_reject_an_unknown_backend(tmp_path: Path) -> None:
    """An unknown backend is rejected even when no file sink is enabled."""
    with pytest.raises(ValueError, match='Unknown backend'):
        build_image_log_handlers(
            'wibble', 'vol/N123', _sinks(tmp_path, image_file=False), LogLevels(), timestamp=_STAMP
        )


def test_image_handlers_fall_back_to_the_null_handler(tmp_path: Path) -> None:
    """An image logger with no sink still gets a handler."""
    handlers, _ = build_image_log_handlers(
        'nav',
        'vol/N123',
        _sinks(tmp_path, image_console=False, image_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    assert isinstance(handlers[0], logging.NullHandler)


def test_no_builder_ever_returns_an_empty_handler_list(tmp_path: Path) -> None:
    """Across every sink combination, a logger is never left able to print by fallback."""
    empty = []
    for console in (True, False):
        for to_file in (True, False):
            handlers, _ = build_image_log_handlers(
                'nav',
                'vol/N123',
                _sinks(tmp_path, image_console=console, image_file=to_file),
                LogLevels(),
                timestamp=_STAMP,
            )
            _close(handlers)
            if not handlers:
                empty.append((console, to_file))
    assert empty == []


# ---------------------------------------------------------------------------
# Per-module levels must actually reach a sink
# ---------------------------------------------------------------------------


def _emit_through_sections(
    tmp_path: Path, levels: LogLevels, emissions: list[tuple[str, str, str]]
) -> str:
    """Log through per-module sections and return what the image log file holds.

    Parameters:
        tmp_path: Directory used as the log root.
        levels: Resolved levels governing the sections.
        emissions: Tuples of module key, level-method name, and message.

    Returns:
        The text written to the image log.
    """
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', _sinks(tmp_path), levels, timestamp=_STAMP
    )
    logger = pdslogger.PdsLogger(f'emit_{tmp_path.name}', lognames=False)
    with logger.open('IMAGE', handler=handlers):
        for log_key, method, message in emissions:
            with logger.open(f'MODULE: {log_key}', level=levels.section_level_for(log_key)):
                getattr(logger, method)(message)
    _close(handlers)
    assert path is not None
    with path.open('r') as handle:
        return str(handle.read())


def test_a_module_raised_above_the_image_level_reaches_the_sink(tmp_path: Path) -> None:
    """A module configured more verbose than the image level actually writes."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit_through_sections(tmp_path, levels, [('titan_haze', 'debug', 'HAZE-DEBUG')])
    assert 'HAZE-DEBUG' in text


def test_an_unraised_module_still_honors_the_image_level(tmp_path: Path) -> None:
    """Raising one module does not make every other module verbose."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit_through_sections(tmp_path, levels, [('ring_edge', 'debug', 'RING-DEBUG')])
    assert 'RING-DEBUG' not in text


def test_a_module_lowered_below_the_image_level_is_suppressed(tmp_path: Path) -> None:
    """A module configured quieter than the image level drops its lesser records."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit_through_sections(tmp_path, levels, [('body_limb', 'info', 'LIMB-INFO')])
    assert 'LIMB-INFO' not in text


def test_a_lowered_module_still_reports_severe_records(tmp_path: Path) -> None:
    """Quieting a module does not lose its errors."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit_through_sections(tmp_path, levels, [('body_limb', 'error', 'LIMB-ERROR')])
    assert 'LIMB-ERROR' in text


def test_handlers_are_built_at_the_most_verbose_module_level(tmp_path: Path) -> None:
    """The sinks are opened wide enough for the most verbose module to pass."""
    levels = LogLevels(image='WARNING', modules={'titan_haze': 'DEBUG'})
    handlers, _ = build_image_log_handlers(
        'nav', 'vol/N1', _sinks(tmp_path), levels, timestamp=_STAMP
    )
    level = handlers[0].level
    _close(handlers)
    assert level == logging.DEBUG


def test_most_verbose_level_ignores_a_quieter_module() -> None:
    """A module quieter than the image level does not raise the handler level."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    assert levels.most_verbose_image_level() == 'INFO'


# ---------------------------------------------------------------------------
# NONE
# ---------------------------------------------------------------------------


def test_section_level_for_a_silent_module_is_not_the_name() -> None:
    """A NONE module resolves to a numeric level pdslogger will accept."""
    levels = LogLevels(modules={'titan_haze': 'NONE'})
    assert levels.section_level_for('titan_haze') == SILENT_LEVEL


def test_a_silent_module_can_open_a_section(tmp_path: Path) -> None:
    """Opening a section for a NONE module does not raise."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'NONE'})
    text = _emit_through_sections(tmp_path, levels, [('titan_haze', 'critical', 'HAZE-CRIT')])
    assert 'HAZE-CRIT' not in text


def test_a_silent_main_logger_reports_no_path(tmp_path: Path) -> None:
    """A main logger at NONE writes no file, so it reports no path."""
    logger = pdslogger.PdsLogger('test_none_path', lognames=False)
    path = build_main_logger(
        logger, SD_OFFSET, _sinks(tmp_path), LogLevels(main='NONE'), timestamp=_STAMP
    )
    logger.remove_all_handlers()
    assert path is None


def test_a_silent_image_logger_reports_no_path(tmp_path: Path) -> None:
    """An image logger silenced everywhere writes no file, so it reports no path."""
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', _sinks(tmp_path), LogLevels(image='NONE'), timestamp=_STAMP
    )
    _close(handlers)
    assert path is None


# ---------------------------------------------------------------------------
# Command-line validation
# ---------------------------------------------------------------------------


def test_an_unknown_command_line_level_is_rejected() -> None:
    """A misspelled level fails with a named error rather than a later KeyError."""
    with pytest.raises(ValueError, match='VERBOSE'):
        resolve_log_levels(SD_OFFSET, _args(log_level=['VERBOSE']), Config())


def test_an_unknown_per_logger_level_is_rejected() -> None:
    """A misspelled --log-level-main value is rejected."""
    with pytest.raises(ValueError, match='log-level-main'):
        resolve_log_levels(SD_OFFSET, _args(log_level_main='CHATTY'), Config())


def test_an_unknown_command_line_module_is_rejected() -> None:
    """A module key naming nothing is rejected rather than silently ignored."""
    with pytest.raises(ValueError, match='bogus_module'):
        resolve_log_levels(SD_OFFSET, _args(log_level=['bogus_module=DEBUG']), Config())


def test_a_hyphenated_module_key_is_rejected() -> None:
    """A near-miss spelling is rejected rather than quietly doing nothing."""
    with pytest.raises(ValueError, match='titan-haze'):
        resolve_log_levels(SD_OFFSET, _args(log_level=['titan-haze=DEBUG']), Config())


def test_an_empty_module_key_is_rejected() -> None:
    """A value with nothing before the equals sign is rejected."""
    with pytest.raises(ValueError, match='no module'):
        resolve_log_levels(SD_OFFSET, _args(log_level=['=DEBUG']), Config())


def test_an_empty_command_line_level_is_rejected() -> None:
    """A value with nothing after the equals sign is rejected."""
    with pytest.raises(ValueError, match='expected one of'):
        resolve_log_levels(SD_OFFSET, _args(log_level=['titan_haze=']), Config())


def test_an_empty_per_logger_level_is_rejected() -> None:
    """An empty level is rejected rather than falling through to a lesser source."""
    with pytest.raises(ValueError, match='log-level-main'):
        resolve_log_levels(SD_OFFSET, _args(log_level_main='', log_level=['ERROR']), Config())


# ---------------------------------------------------------------------------
# Coverage the first pass missed
# ---------------------------------------------------------------------------


def test_log_level_image_beats_the_bare_form() -> None:
    """The per-logger image flag is more specific than the global one."""
    args = _args(log_level=['DEBUG'], log_level_image='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.image == 'ERROR'


def test_log_level_image_leaves_the_main_logger_alone() -> None:
    """Setting the image level does not change the main level."""
    args = _args(log_level=['DEBUG'], log_level_image='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, Config())
    assert levels.main == 'DEBUG'


def test_a_models_category_default_applies(nav_config_with: NavConfigWith) -> None:
    """A models category default governs every model."""
    config = nav_config_with({'logging': {'models': {'default': 'DEBUG'}}})
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('rings') == 'DEBUG'


def test_a_shipped_module_key_beats_an_other_category_default(
    nav_config_with: NavConfigWith,
) -> None:
    """A module named in the category keeps its own level over the default.

    The shipped configuration sets ``other.annotate`` explicitly, so adding a
    category default must not override it.
    """
    config = nav_config_with({'logging': {'other': {'default': 'DEBUG'}}})
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('annotate') == 'ERROR'


def test_an_other_category_default_can_be_overridden_per_module(
    nav_config_with: NavConfigWith,
) -> None:
    """Naming the module alongside the default takes the module's value."""
    config = nav_config_with({'logging': {'other': {'default': 'DEBUG', 'annotate': 'WARNING'}}})
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('annotate') == 'WARNING'


def test_run_timestamp_matches_the_documented_format() -> None:
    """A generated timestamp parses back under the documented format."""
    datetime.strptime(run_timestamp(), LOG_TIMESTAMP_FORMAT)


def test_resolution_is_stable_across_repeated_calls(nav_config_with: NavConfigWith) -> None:
    """Resolving twice gives the same answer, so nothing is mutated in place."""
    config = nav_config_with({'logging': {'programs': {SD_MOSAIC: {'main': 'WARNING'}}}})
    first = resolve_log_levels(SD_MOSAIC, None, config)
    second = resolve_log_levels(SD_MOSAIC, None, config)
    assert first == second


def test_resolving_one_program_does_not_leak_into_another(nav_config_with: NavConfigWith) -> None:
    """A program's merge does not contaminate the shared configuration."""
    config = nav_config_with(
        {'logging': {'main': 'INFO', 'programs': {SD_MOSAIC: {'main': 'WARNING'}}}}
    )
    resolve_log_levels(SD_MOSAIC, None, config)
    assert resolve_log_levels(SD_OFFSET, None, config).main == 'INFO'


def test_module_keys_exclude_test_registered_classes() -> None:
    """A class a test registers does not enter the resolved key set.

    The registries are process-global and test modules add their own
    subclasses, so the key namespace would otherwise depend on which tests had
    been imported -- and a configuration valid under one selection would be
    rejected under another.

    Asserted against a class registered here, since the derived key for a stub
    is an ordinary snake_case name indistinguishable from a shipped one: an
    earlier version of this test looked for a leading underscore, which
    ``log_key_for`` strips, so it could not have failed.
    """
    from spindoctor.nav_technique.nav_technique import NavTechnique

    # Restored on the way out: the registry is process-global, and a class left
    # in it breaks the confidence-config validation every later configuration
    # load performs -- which is a sharper version of the same problem this
    # filter exists to solve.
    before = list(NavTechnique._registry)
    try:

        class _RegisteredByThisTestNav(NavTechnique):
            """A technique defined outside the package, as a test's would be."""

            Tuning = object

        assert _RegisteredByThisTestNav in NavTechnique._registry
        assert log_key_for(_RegisteredByThisTestNav) not in technique_log_keys()
    finally:
        NavTechnique._registry[:] = before


def test_a_shipped_technique_is_in_the_key_set() -> None:
    """The filter keeps what ships, rather than emptying the set.

    The counterpart to the exclusion above: a filter that excluded everything
    would satisfy it just as well.
    """
    assert 'titan_haze' in technique_log_keys()


def test_a_cloud_log_root_is_not_mkdired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building against a remote root never attempts a directory creation.

    FCPath.mkdir raises NotImplementedError on a remote path, and a cloud
    results root is exactly the cloud-task configuration, so the builder must
    leave parent creation to the handler factory.  The factory is stubbed so
    the test needs no credentials.
    """
    seen: list[str] = []

    def fake_file_handler(path: FCPath, level: object = None, **kwargs: object) -> logging.Handler:
        seen.append(FCPath(path).as_posix())
        return logging.NullHandler()

    def exploding_mkdir(*args: object, **kwargs: object) -> None:
        raise AssertionError('the builder must not mkdir a log root')

    monkeypatch.setattr(pdslogger, 'file_handler', fake_file_handler)
    monkeypatch.setattr(FCPath, 'mkdir', exploding_mkdir)

    sinks = LogSinks(log_root=FCPath('gs://example-bucket/run/logs'), main_console=False)
    logger = pdslogger.PdsLogger('test_cloud_root', lognames=False)
    build_main_logger(logger, SD_OFFSET, sinks, LogLevels(), timestamp=_STAMP)
    logger.remove_all_handlers()
    assert seen == [f'gs://example-bucket/run/logs/{SD_OFFSET}/main_{_STAMP}.log']


def test_rebuilding_reattaches_the_shared_null_handler(tmp_path: Path) -> None:
    """A rebuild leaves the process-wide NULL_HANDLER attached, not discarded."""
    logger = pdslogger.PdsLogger('test_null_reuse', lognames=False)
    silent = _sinks(tmp_path, main_console=False, main_file=False)
    build_main_logger(logger, SD_OFFSET, silent, LogLevels(), timestamp=_STAMP)
    build_main_logger(logger, SD_OFFSET, silent, LogLevels(), timestamp=_STAMP)
    attached = list(logger.handlers)
    logger.remove_all_handlers()
    assert attached == [pdslogger.NULL_HANDLER]


def test_a_silenced_logger_stays_silent_after_a_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rebuilding a fully silenced logger does not restore output."""
    logger = pdslogger.PdsLogger('test_null_quiet', lognames=False)
    silent = _sinks(tmp_path, main_console=False, main_file=False)
    build_main_logger(logger, SD_OFFSET, silent, LogLevels(), timestamp=_STAMP)
    build_main_logger(logger, SD_OFFSET, silent, LogLevels(), timestamp=_STAMP)
    logger.info('should not appear')
    logger.remove_all_handlers()
    assert capsys.readouterr().out == ''


# ---------------------------------------------------------------------------
# A results path stub reaches this from task data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'stub',
    [
        '../../../escaped/evil',
        'a/../../../evil',
        '/etc/passwd',
        '',
        'has\x00null',
        '..\\..\\evil',
    ],
)
def test_a_stub_leaving_the_log_root_is_rejected(tmp_path: Path, stub: str) -> None:
    """A stub that would put the log outside the log root is refused.

    Cloud-task drivers take this value from task data, so it is not
    necessarily trustworthy.
    """
    # Matched on the message: without it the test passes on any ValueError,
    # including one raised for a reason that has nothing to do with the stub.
    with pytest.raises(ValueError, match=r'[Rr]esults path stub'):
        image_log_path(FCPath(str(tmp_path)), 'nav', stub, timestamp=_STAMP)


def test_a_stub_with_subdirectories_is_accepted(tmp_path: Path) -> None:
    """A normal stub is a relative path with directories and must still work."""
    path = image_log_path(
        FCPath(str(tmp_path)), 'nav', 'COISS_2058/data/1635/N123_1', timestamp=_STAMP
    )
    assert path.as_posix().endswith(f'nav/COISS_2058/data/1635/N123_1_{_STAMP}.log')


def test_building_handlers_rejects_an_escaping_stub(tmp_path: Path) -> None:
    """The handler builder refuses the same stubs the path builder does."""
    with pytest.raises(ValueError, match='within the log root'):
        build_image_log_handlers(
            'nav', '../../evil', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
        )


# ---------------------------------------------------------------------------
# Falling back rather than losing logs
# ---------------------------------------------------------------------------


def _no_resolvable_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every source that would otherwise name a log root.

    Parameters:
        monkeypatch: Fixture used to clear the environment variables.
    """
    for name in ('NAV_LOG_ROOT', 'NAV_RESULTS_ROOT'):
        monkeypatch.delenv(name, raising=False)


def test_a_fallback_root_is_used_when_nothing_else_names_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver that knows its own output directory keeps its logs."""
    _no_resolvable_root(monkeypatch)
    run_logging = build_run_logging(
        SD_OFFSET,
        argparse.Namespace(),
        Config(),
        build_main=False,
        fallback_log_root=FCPath(str(tmp_path / 'task_out' / 'logs')),
    )
    assert run_logging.sinks.image_file is True


def test_the_fallback_root_is_where_logs_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback is used as the log root, not merely accepted."""
    _no_resolvable_root(monkeypatch)
    run_logging = build_run_logging(
        SD_OFFSET,
        argparse.Namespace(),
        Config(),
        build_main=False,
        fallback_log_root=FCPath(str(tmp_path / 'task_out' / 'logs')),
    )
    assert run_logging.sinks.log_root.as_posix().endswith('task_out/logs')


def test_without_a_fallback_the_file_sinks_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A program with nowhere to write still runs, without log files."""
    _no_resolvable_root(monkeypatch)
    run_logging = build_run_logging(SD_OFFSET, argparse.Namespace(), Config(), build_main=False)
    assert (run_logging.sinks.main_file, run_logging.sinks.image_file) == (False, False)


# ---------------------------------------------------------------------------
# No logger can reach the print fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('main_console', 'main_file', 'image_console', 'image_file'),
    list(itertools.product([True, False], repeat=4)),
)
@pytest.mark.parametrize('level', ['DEBUG', 'INFO', 'NONE'])
def test_image_handlers_are_never_empty(
    tmp_path: Path,
    main_console: bool,
    main_file: bool,
    image_console: bool,
    image_file: bool,
    level: str,
) -> None:
    """No combination of sinks and levels leaves a logger with no handlers.

    A PdsLogger with an empty handler list does not go quiet: it prints every
    record to stdout regardless of level.  Asking for no output must therefore
    produce the null handler rather than nothing, and that has to hold for
    every combination rather than the ones a driver happens to use.
    """
    sinks = LogSinks(
        log_root=FCPath(tmp_path),
        main_console=main_console,
        main_file=main_file,
        image_console=image_console,
        image_file=image_file,
    )
    handlers, _ = build_image_log_handlers(
        'nav',
        f'vol/{main_console}{main_file}{image_console}{image_file}{level}',
        sinks,
        LogLevels(main=level, image=level),
        timestamp=_STAMP,
    )
    try:
        assert handlers != []
    finally:
        _close(handlers)


@pytest.mark.parametrize(
    ('main_console', 'main_file'), list(itertools.product([True, False], repeat=2))
)
@pytest.mark.parametrize('level', ['DEBUG', 'INFO', 'NONE'])
def test_the_main_logger_is_never_left_empty(
    tmp_path: Path, main_console: bool, main_file: bool, level: str
) -> None:
    """The same holds for the main logger, however its sinks and level are set.

    ``NONE`` is the case worth having here: it creates no real sink even when a
    file sink was asked for, so it reaches the null handler by a different route
    than turning the sinks off does.
    """
    logger = pdslogger.PdsLogger.get_logger(
        f'never_empty_{main_console}_{main_file}_{level}', lognames=False
    )
    sinks = LogSinks(log_root=FCPath(tmp_path), main_console=main_console, main_file=main_file)
    build_main_logger(logger, SD_OFFSET, sinks, LogLevels(main=level), timestamp=_STAMP)
    try:
        assert logger.handlers != []
    finally:
        _close(list(logger.handlers))
        logger.remove_all_handlers()


# ---------------------------------------------------------------------------
# The logger registry does not grow with the batch
# ---------------------------------------------------------------------------


def test_the_registry_does_not_grow_with_the_image_count(tmp_path: Path) -> None:
    """Navigating many images leaves the logger registry the size it was.

    pdslogger keeps every logger it is asked for in a process-wide table and
    never frees one, so a logger per image would leak for the length of a
    batch.  One image logger is reused with its handlers swapped instead.
    """
    before = len(pdslogger._LOOKUP)
    for index in range(50):
        handlers, _ = build_image_log_handlers(
            'nav', f'vol/N{index}', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
        )
        try:
            with IMAGE_LOGGER.open(f'IMAGE {index}', handler=handlers):
                IMAGE_LOGGER.info('processing')
        finally:
            _close(handlers)
    assert len(pdslogger._LOOKUP) == before


def test_an_image_level_does_not_enable_the_image_console(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """Setting the image *level* does not also turn the image console on.

    The two are separate questions -- how much to log, and where it goes --
    and a level key that reached sink selection would quietly answer the
    second.
    """
    nav_config_with({'logging': {'image': 'DEBUG'}})
    assert sinks_from_arguments(_args(), FCPath(tmp_path)).image_console is False


# ---------------------------------------------------------------------------
# Console selection from the configuration
# ---------------------------------------------------------------------------


def _console_sinks(
    nav_config_with: NavConfigWith, root: FCPath, logging: dict[str, Any], **arguments: object
) -> LogSinks:
    """Resolve sinks for sd_offset from a logging override and any arguments.

    Parameters:
        nav_config_with: Builds a configuration from an override.
        root: Directory used as the log root.
        logging: The mapping placed under the ``logging`` key.
        **arguments: Command-line arguments to supply.

    Returns:
        The resolved sinks.
    """
    return sinks_from_arguments(
        _args(**arguments),
        root,
        program_name=SD_OFFSET,
        config=nav_config_with({'logging': logging}),
    )


def test_the_image_console_can_be_turned_on_from_the_configuration(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """An operator who always wants image detail on screen can say so once."""
    assert (
        _console_sinks(nav_config_with, FCPath(tmp_path), {'image_console': True}).image_console
        is True
    )


def test_the_main_console_can_be_turned_off_from_the_configuration(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """And a machine that should never chatter can say that once."""
    assert (
        _console_sinks(nav_config_with, FCPath(tmp_path), {'main_console': False}).main_console
        is False
    )


def test_the_command_line_beats_the_configured_image_console(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """The flag is the more specific statement, so it wins for that run."""
    sinks = _console_sinks(
        nav_config_with, FCPath(tmp_path), {'image_console': True}, log_image_to_console=False
    )
    assert sinks.image_console is False


def test_the_command_line_beats_the_configured_main_console(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """In the other direction too."""
    sinks = _console_sinks(
        nav_config_with, FCPath(tmp_path), {'main_console': False}, log_main_to_console=True
    )
    assert sinks.main_console is True


def test_a_program_block_scopes_the_console_to_that_program(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """A console set for one program is not set for another."""
    config = nav_config_with({'logging': {'programs': {SD_MOSAIC: {'image_console': True}}}})
    mosaic = sinks_from_arguments(_args(), FCPath(tmp_path), program_name=SD_MOSAIC, config=config)
    assert mosaic.image_console is True


def test_a_program_block_leaves_another_program_alone(
    tmp_path: Path, nav_config_with: NavConfigWith
) -> None:
    """The counterpart: the program that was not named keeps the default."""
    config = nav_config_with({'logging': {'programs': {SD_MOSAIC: {'image_console': True}}}})
    offset = sinks_from_arguments(_args(), FCPath(tmp_path), program_name=SD_OFFSET, config=config)
    assert offset.image_console is False


def test_the_shipped_defaults_are_unchanged(tmp_path: Path) -> None:
    """Adding the keys did not change what a run does without them."""
    sinks = sinks_from_arguments(_args(), FCPath(tmp_path), program_name=SD_OFFSET, config=Config())
    assert (sinks.main_console, sinks.image_console) == (True, False)


# ---------------------------------------------------------------------------
# What one timestamp groups
# ---------------------------------------------------------------------------


def test_one_run_stamps_every_file_it_writes_alike(tmp_path: Path) -> None:
    """A program stamps once at startup, so its files carry one timestamp.

    That is what makes a run identifiable afterwards: the logs of one pass
    over many images sort together and are distinguishable from the pass
    before it.
    """
    # The run's stamp is a sentinel that cannot coincide with "now": a builder
    # that ignored it and stamped itself would otherwise still match, because
    # both would fall in the same second.
    with mock.patch.object(logging_config, 'run_timestamp', return_value=_RUN_STAMP):
        run_logging = build_run_logging(
            SD_OFFSET, _args(log_root=str(tmp_path)), Config(), build_main=False
        )
    first_handlers, first_path = build_image_log_handlers(
        'nav', 'vol/N1', run_logging.sinks, run_logging.levels, timestamp=run_logging.timestamp
    )
    second_handlers, second_path = build_image_log_handlers(
        'nav', 'vol/N2', run_logging.sinks, run_logging.levels, timestamp=run_logging.timestamp
    )
    _close(first_handlers)
    _close(second_handlers)
    assert first_path is not None
    assert second_path is not None
    # The builder's own paths, not ones rebuilt here from the stamp that was
    # passed in -- that would only show image_log_path appends what it is given.
    assert first_path.name.endswith(f'{_RUN_STAMP}.log')
    assert second_path.name.endswith(f'{_RUN_STAMP}.log')


def test_a_cloud_task_stamps_itself(tmp_path: Path) -> None:
    """A worker stamps per task, because that is where logging is set up.

    There is no run-wide moment for tasks to share: they are handed to
    whichever workers are free, on machines that started at different times and
    may not have been running when the batch began.  Each image's log carries
    the time its own task was picked up, and UTC is what keeps those
    comparable across machines.
    """
    config = Config()
    first = build_cloud_task_logging(SD_OFFSET, _args(log_root=str(tmp_path)), config).timestamp
    with mock.patch.object(logging_config, 'run_timestamp', return_value='2099-01-01T00-00-00'):
        second = build_cloud_task_logging(
            SD_OFFSET, _args(log_root=str(tmp_path)), config
        ).timestamp
    assert first != second
