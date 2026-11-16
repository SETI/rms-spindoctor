"""The ``instrument_config`` block's validator for the sim-scene schema.

A scene's ``instrument_config`` overrides keys of the navigator block its instrument
resolves to.  The overrides are published knowledge: they cross the information
boundary, and the navigator's faint-star limit reads its zero point and read noise
from the ``noise`` block they leave.  This module checks that block, including the
binned-mode table ``noise.by_binning``, so that every value a scene pins is a value
some frame of that scene reads.  The scalar primitives it shares with the other block
checkers stay in :mod:`spindoctor.sim.scene_checks`, which also documents the overall
validation contract (every violation raises
:class:`spindoctor.sim.scene_schema.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate_sim_params` drives ``_check_instrument_config``
once per scene.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.config.config import BY_BINNING_VALUE_KEYS, check_by_binning
from spindoctor.sim.instrument_navigator import (
    resolve_sim_inst_config,
    with_integer_binning_keys,
)
from spindoctor.sim.instrument_resolvers import cataloged_binnings
from spindoctor.sim.scene_checks import _check_optional_mapping
from spindoctor.sim.scene_schema import SimSceneValidationError

_NOISE_PATH = 'instrument_config.noise'
"""The key path of the block these checks cover, for error messages."""

_TABLE_PATH = f'{_NOISE_PATH}.by_binning'
"""The key path of the binned-mode table, for error messages."""


def _refuse_nulls(value: Mapping[Any, Any], path: str, *, source: str) -> None:
    """Refuse a null at any depth of a mapping, naming its key path.

    Parameters:
        value: The mapping to search.
        path: The mapping's dotted key path.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If any value at any depth is None.
    """
    for key, item in value.items():
        if item is None:
            raise SimSceneValidationError(
                f'{source}: {path}.{key} is null; leave the key out to take its default'
            )
        if isinstance(item, Mapping):
            _refuse_nulls(item, f'{path}.{key}', source=source)


def _check_binning_keys(table: Mapping[Any, Any], *, source: str) -> None:
    """Refuse a binned-mode table whose digit-string keys don't each name one mode.

    A key written as a string of digits names the binning it spells, so a key with a
    leading zero would name a binning under a second spelling, and a table holding a
    binning both as an integer and as its digits would set one mode twice.

    Parameters:
        table: The scene's ``noise.by_binning`` mapping, keys as written.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If a key has a leading zero, or the table holds an
            integer key and its digits.
    """
    for key in table:
        if not (isinstance(key, str) and key.isascii() and key.isdigit()):
            continue
        if key != str(int(key)):
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH}: the key {key!r} has a leading zero; write it '
                f'as {int(key)}'
            )
        if int(key) in table:
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH} holds both {int(key)} and {key!r}, which name '
                f'the same binning; keep one'
            )


def _check_table_is_read(
    table: Mapping[Any, Any], resolved: Mapping[str, Any], *, instrument: str, source: str
) -> None:
    """Refuse a binned-mode table entry that no frame of the instrument reads.

    The star limit reads the table only on a raw-DN block, and there only the entry
    for the frame's binning.  A calibrated-unit block's limit is a constant that reads
    no noise value, so any table there is refused, and on a raw-DN block an entry for
    a binning the instrument's readout modes don't offer is refused.  A key that isn't
    a binning factor is left for the table's own check.

    Parameters:
        table: The scene's ``noise.by_binning`` mapping, keys read as integers.
        resolved: The navigator block the scene's overrides resolve to.
        instrument: The scene's (already validated) instrument name.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the table is set on a calibrated-unit block, or it
            holds an entry for a binning the instrument doesn't offer.
    """
    units = resolved['data_units']
    if units != 'raw_dn':
        raise SimSceneValidationError(
            f'{source}: {_TABLE_PATH} is never read: instrument {instrument!r} renders '
            f'{units} frames, whose star limit reads no noise value. Leave the table out.'
        )
    offered = cataloged_binnings(instrument)
    for key in table:
        if isinstance(key, int) and not isinstance(key, bool) and key not in offered:
            listed = ', '.join(str(binning) for binning in offered)
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH}.{key} is never read: instrument '
                f'{instrument!r} offers binnings {listed}. Leave the entry out.'
            )


def _check_instrument_config(value: Any, *, instrument: str, binning: int, source: str) -> None:
    """Validate the scene-level ``instrument_config`` block.

    The block can override any key of the navigator block except ``data_units``.  A
    frame's units come only from the instrument's config entry; for other units, a
    scene uses another instrument.

    When the block sets ``noise``, that must be a mapping with no null at any depth; a
    scene leaves a key out to take its default.  The binned-mode table
    ``noise.by_binning`` may write a key as an integer or as a string of digits, since
    a scene read through JSON carries every key as a string, but each binning only
    once and without a leading zero.  Each entry must be one that a frame of the scene's
    instrument reads: none on a calibrated-unit block, and only the binnings the
    instrument offers on a raw-DN block.  The navigator block the overrides resolve to
    (over the default configuration) must then hold a well-formed table, checked as the
    configuration loader checks it.  On a binned frame the star limit reads the mode's
    ``read_noise_dn`` and ``star_flux_dn_per_s_vmag0`` from that table, so a scene that
    sets either unbinned value must set the mode's value as well, or the value it sets
    would never be read.

    Parameters:
        value: The ``instrument_config`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name.
        binning: The scene's (already validated) readout binning.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the block isn't a mapping, it sets ``data_units``,
            its ``noise`` isn't a mapping or holds a null, its ``noise.by_binning``
            table spells a binning twice, holds an entry no frame reads, or is
            malformed, or it sets an unbinned noise value that a frame of this binning
            reads from the table.
    """
    _check_optional_mapping(value, 'instrument_config', source=source)
    if value is None:
        return
    if 'data_units' in value:
        raise SimSceneValidationError(
            f"{source}: instrument_config: a scene can't set data_units. Choose an "
            'instrument whose entry renders the units you want.'
        )
    if 'noise' not in value:
        return
    if value['noise'] is None:
        raise SimSceneValidationError(
            f'{source}: {_NOISE_PATH} is null; leave the key out to take its default'
        )
    if not isinstance(value['noise'], Mapping):
        raise SimSceneValidationError(f'{source}: {_NOISE_PATH} must be a mapping when present')
    _refuse_nulls(value['noise'], _NOISE_PATH, source=source)
    if isinstance(value['noise'].get('by_binning'), Mapping):
        _check_binning_keys(value['noise']['by_binning'], source=source)
    override = with_integer_binning_keys(value)
    noise = override['noise']
    resolved = resolve_sim_inst_config(DEFAULT_CONFIG, instrument, override)
    if 'by_binning' in noise:
        table = noise['by_binning']
        _check_table_is_read(
            table if isinstance(table, Mapping) else {},
            resolved,
            instrument=instrument,
            source=source,
        )
    try:
        check_by_binning('instrument_config', resolved)
    except ValueError as exc:
        raise SimSceneValidationError(f'{source}: {exc}') from exc
    if binning not in resolved['noise'].get('by_binning', {}):
        return
    mode = noise.get('by_binning', {}).get(binning, {})
    for key in sorted(BY_BINNING_VALUE_KEYS):
        if key in noise and key not in mode:
            raise SimSceneValidationError(
                f'{source}: {_NOISE_PATH}.{key} is set, but a frame binned by '
                f'{binning} reads {_TABLE_PATH}.{binning}.{key} '
                f'instead. Set {_TABLE_PATH}.{binning}.{key} too.'
            )
