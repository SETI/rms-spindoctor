"""Tests for per-component log sections and the keys that configure them."""

import logging
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.annotation.annotations import Annotations
from spindoctor.config import (
    IMAGE_LOGGER,
    MAIN_LOGGER,
    LogLevels,
    LogRole,
    LogSinks,
    build_image_log_handlers,
    image_scope,
    logged_section,
    set_log_levels,
)
from spindoctor.config.logging_keys import OTHER_LOG_KEYS
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.simulated.rings import NavModelRingsSimulated
from spindoctor.nav_technique.nav_technique_body_limb import BodyLimbNav
from spindoctor.nav_technique.nav_technique_titan_haze import TitanHazeNav
from spindoctor.support.nav_base import NavBase

_STAMP = '2026-07-29T12-00-00'

# Every "other" component except annotation, which is a class with a declared
# log_key rather than a decorated function and is covered separately.
_DECORATED_OTHER_KEYS = sorted(OTHER_LOG_KEYS - {'annotate'})


def _emit(root: FCPath, levels: LogLevels, component: NavBase, method: str, message: str) -> str:
    """Log one record inside the component's section and return the log text.

    Parameters:
        root: Directory used as the log root.
        levels: Levels to install for the run.
        component: The component whose section the record is logged inside.
        method: Name of the level method to call.
        message: Text to log.

    Returns:
        The contents of the per-image log file.
    """
    set_log_levels(levels)
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', LogSinks(log_root=root), levels, timestamp=_STAMP
    )
    logger = pdslogger.PdsLogger.get_logger(f'section_{component.resolved_log_key}', lognames=False)
    logger.remove_all_handlers()
    logger.add_handler(pdslogger.NULL_HANDLER)
    try:
        with (
            image_scope(logger),
            IMAGE_LOGGER.open('IMAGE', handler=handlers),
            component.log_section('COMPONENT'),
        ):
            getattr(IMAGE_LOGGER, method)(message)
    finally:
        for handler in handlers:
            handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('component', 'expected'),
    [
        (TitanHazeNav, 'titan_haze'),
        (BodyLimbNav, 'body_limb'),
    ],
)
def test_a_technique_is_keyed_by_its_snake_case_name(
    component: type[NavBase], expected: str
) -> None:
    """A technique is configured under the snake_case form of its class name."""
    assert component().resolved_log_key == expected


def test_a_model_is_keyed_by_its_family() -> None:
    """A model is configured under its family rather than its class."""
    assert NavModelRings.__new__(NavModelRings).resolved_log_key == 'rings'


def test_a_simulated_model_shares_its_siblings_key() -> None:
    """A simulated model is configured with the model it stands in for."""
    real = NavModelRings.__new__(NavModelRings).resolved_log_key
    simulated = NavModelRingsSimulated.__new__(NavModelRingsSimulated).resolved_log_key
    assert real == simulated


def test_a_body_model_is_keyed_by_family() -> None:
    """Every body model shares one key regardless of which body it renders."""
    assert NavModelBody.__new__(NavModelBody).resolved_log_key == 'body'


def test_a_declared_key_wins_over_the_derived_one() -> None:
    """Annotation is configured as "annotate", not as its class name."""
    assert Annotations.__new__(Annotations).resolved_log_key == 'annotate'


def test_a_declared_key_is_inherited() -> None:
    """A family sharing one key declares it once on their base."""

    class _Base(NavBase):
        log_key = 'shared'

    class _Derived(_Base):
        pass

    assert _Derived().resolved_log_key == 'shared'


def test_an_underived_subclass_keys_by_its_own_name() -> None:
    """A subclass with no declared key derives one from its own class name."""

    class _SomeThingNav(NavBase):
        pass

    assert _SomeThingNav().resolved_log_key == 'some_thing'


# ---------------------------------------------------------------------------
# Levels reaching the log
# ---------------------------------------------------------------------------


def test_a_raised_module_writes_below_the_image_level(tmp_path: Path) -> None:
    """A component configured more verbose than the image level actually writes."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit(FCPath(tmp_path), levels, TitanHazeNav(), 'debug', 'HAZE-DEBUG')
    assert 'HAZE-DEBUG' in text


def test_an_unraised_module_is_unaffected(tmp_path: Path) -> None:
    """Raising one component does not make another verbose."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit(FCPath(tmp_path), levels, BodyLimbNav(), 'debug', 'LIMB-DEBUG')
    assert 'LIMB-DEBUG' not in text


def test_a_quieted_module_drops_its_lesser_records(tmp_path: Path) -> None:
    """A component configured quieter than the image level is suppressed."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit(FCPath(tmp_path), levels, BodyLimbNav(), 'info', 'LIMB-INFO')
    assert 'LIMB-INFO' not in text


def test_a_quieted_module_still_reports_errors(tmp_path: Path) -> None:
    """Quieting a component does not lose its errors."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit(FCPath(tmp_path), levels, BodyLimbNav(), 'error', 'LIMB-ERROR')
    assert 'LIMB-ERROR' in text


def test_a_module_with_no_override_follows_the_image_level(tmp_path: Path) -> None:
    """A component nobody configured takes the image level."""
    levels = LogLevels(image='DEBUG')
    text = _emit(FCPath(tmp_path), levels, BodyLimbNav(), 'debug', 'LIMB-DEBUG')
    assert 'LIMB-DEBUG' in text


def test_a_silenced_module_writes_nothing(tmp_path: Path) -> None:
    """A component configured NONE emits nothing, and does not raise doing so."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'NONE'})
    text = _emit(FCPath(tmp_path), levels, BodyLimbNav(), 'critical', 'LIMB-CRITICAL')
    assert 'LIMB-CRITICAL' not in text


def test_an_explicit_level_overrides_the_configured_one() -> None:
    """A caller passing level= keeps control of its own section."""
    levels = LogLevels(image='ERROR')
    set_log_levels(levels)
    logger = pdslogger.PdsLogger.get_logger('section_explicit', lognames=False)
    logger.remove_all_handlers()
    logger.add_handler(pdslogger.NULL_HANDLER)
    technique = BodyLimbNav()
    with image_scope(logger), technique.log_section('COMPONENT', level='debug'):
        assert logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# Role interaction
# ---------------------------------------------------------------------------


def test_a_main_role_section_takes_the_main_level() -> None:
    """A main-role component uses the main level, not the image level.

    The run's log is not divided by component, so a main-role component has no
    per-module key; taking the image level here would make the two loggers'
    verbosity move together.
    """
    set_log_levels(LogLevels(main='ERROR', image='DEBUG'))

    class _RunScoped(NavBase):
        log_role = LogRole.MAIN

    with _RunScoped().log_section('RUN SECTION'):
        assert MAIN_LOGGER.level == logging.ERROR


def test_a_main_role_section_opens_on_the_main_logger() -> None:
    """A main-role component sections the run's logger, not the image one."""
    set_log_levels(LogLevels(main='ERROR', image='DEBUG'))

    class _RunScoped(NavBase):
        log_role = LogRole.MAIN

    component = _RunScoped()
    assert component.logger is MAIN_LOGGER


def test_an_explicit_none_level_inherits() -> None:
    """Passing level=None inherits the enclosing level rather than the configured one."""
    set_log_levels(LogLevels(image='ERROR'))
    logger = pdslogger.PdsLogger.get_logger('section_inherit', lognames=False)
    logger.add_handler(pdslogger.NULL_HANDLER)
    with image_scope(logger):
        enclosing = logger.level
        with BodyLimbNav().log_section('COMPONENT', level=None):
            assert logger.level == enclosing


def test_omitting_the_level_applies_the_configured_one() -> None:
    """Saying nothing takes the configured level, unlike passing None."""
    set_log_levels(LogLevels(image='ERROR'))
    logger = pdslogger.PdsLogger.get_logger('section_configured', lognames=False)
    logger.add_handler(pdslogger.NULL_HANDLER)
    with image_scope(logger), BodyLimbNav().log_section('COMPONENT'):
        assert logger.level == logging.ERROR


# ---------------------------------------------------------------------------
# Components that are functions rather than classes
# ---------------------------------------------------------------------------


def _emit_in_decorated(root: FCPath, levels: LogLevels, log_key: str, message: str) -> str:
    """Log a DEBUG record inside a decorated section and return the log text.

    Parameters:
        root: Directory used as the log root.
        levels: Levels to install for the run.
        log_key: The component key whose section to open.
        message: Text to log.

    Returns:
        The contents of the per-image log file.
    """
    set_log_levels(levels)
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', LogSinks(log_root=root), levels, timestamp=_STAMP
    )
    logger = pdslogger.PdsLogger.get_logger(f'decorated_{log_key}', lognames=False)
    logger.remove_all_handlers()
    logger.add_handler(pdslogger.NULL_HANDLER)

    @logged_section(log_key, log_key.upper())
    def component() -> None:
        IMAGE_LOGGER.debug(message)

    try:
        with image_scope(logger), IMAGE_LOGGER.open('IMAGE', handler=handlers):
            component()
    finally:
        for handler in handlers:
            handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


@pytest.mark.parametrize(
    'log_key',
    _DECORATED_OTHER_KEYS,
)
def test_a_function_component_can_be_raised(tmp_path: Path, log_key: str) -> None:
    """Each function-shaped component can be made verbose on its own."""
    levels = LogLevels(image='INFO', modules={log_key: 'DEBUG'})
    assert 'RAISED' in _emit_in_decorated(FCPath(tmp_path), levels, log_key, 'RAISED')


@pytest.mark.parametrize(
    'log_key',
    _DECORATED_OTHER_KEYS,
)
def test_a_function_component_can_be_silenced(tmp_path: Path, log_key: str) -> None:
    """Each function-shaped component can be silenced on its own."""
    levels = LogLevels(image='DEBUG', modules={log_key: 'NONE'})
    assert 'SILENCED' not in _emit_in_decorated(FCPath(tmp_path), levels, log_key, 'SILENCED')


def test_raising_one_function_component_leaves_another_alone(tmp_path: Path) -> None:
    """Component keys are independent of one another."""
    levels = LogLevels(image='INFO', modules={'ensemble': 'DEBUG'})
    assert 'OTHER' not in _emit_in_decorated(FCPath(tmp_path), levels, 'provenance', 'OTHER')


def test_the_decorator_preserves_the_wrapped_return_value() -> None:
    """Opening a section does not disturb what the component returns."""

    @logged_section('ensemble', 'ENSEMBLE')
    def component(value: int) -> int:
        return value * 2

    with image_scope():
        assert component(21) == 42


def test_the_decorator_preserves_the_wrapped_identity() -> None:
    """The decorated component still reports its own name."""

    @logged_section('ensemble', 'ENSEMBLE')
    def component() -> None:
        return None

    assert component.__name__ == 'component'
