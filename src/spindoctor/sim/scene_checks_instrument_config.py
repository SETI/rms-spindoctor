"""The ``instrument_config`` block's validator for the sim-scene schema.

A scene's ``instrument_config`` overrides keys of the navigator block its instrument
resolves to.  The overrides are published knowledge: they cross the information
boundary, and the navigator's faint-star limit reads its zero point, read noise, and PSF
width from the block they leave.  This module checks every one of those values a scene
sets, including the binned-mode table ``noise.by_binning``, so that every value a scene
pins is a value the limit can use and a frame of the scene's instrument reads.  A scene
is the only outside source of these values, so the scene validator is the one place
that checks them.  The scalar primitives it shares with the other block checkers stay in
:mod:`spindoctor.sim.scene_checks`, which also documents the overall validation contract
(every violation raises :class:`spindoctor.sim.scene_schema.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate_sim_params` drives ``_check_instrument_config``
once per scene.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instrument_navigator import (
    resolve_sim_inst_config,
    with_integer_binning_keys,
)
from spindoctor.sim.instrument_resolvers import cataloged_binnings
from spindoctor.sim.scene_checks import _check_optional_mapping
from spindoctor.sim.scene_schema import SimSceneValidationError

BY_BINNING_VALUE_KEYS = frozenset({'read_noise_dn', 'star_flux_dn_per_s_vmag0'})
"""The values every entry of a navigator block's ``noise.by_binning`` table states."""

_NOISE_PATH = 'instrument_config.noise'
"""The key path of the noise block these checks cover, for error messages."""

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


def _binning_of_key(key: Any) -> int | None:
    """The binning a binned-mode table key spells, or None when it spells none.

    Parameters:
        key: A key of the scene's ``noise.by_binning`` mapping, as written.

    Returns:
        The key itself when it is an integer, its value when it is a string of ASCII
        digits, and None for any other key.
    """
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    if isinstance(key, str) and key.isascii() and key.isdigit():
        return int(key)
    return None


def _check_binning_keys(table: Mapping[Any, Any], *, source: str) -> None:
    """Refuse a binned-mode table key that doesn't name exactly one binning.

    A key must be an integer or a string of ASCII digits, since a scene read through JSON
    carries every key as a string, and the binning it spells must be at least 2.  Any
    other key is refused here, before the scene's table is merged over the navigator's, so
    the key ``-1`` gets the same answer as the ``'-1'`` it becomes through JSON.  A float
    ``2.0`` equals and hashes as the integer ``2``, so the merge would fold it into the
    navigator's own key and the table's check would never see it.  Every key's form is
    checked before any two keys are compared, so a float that equals a digit-string key is
    refused for its form.  A key written as a string of digits names the binning it
    spells, so a key with a leading zero would name a binning under a second spelling, and
    a table holding a binning both as an integer and as its digits would set one mode
    twice.

    Parameters:
        table: The scene's ``noise.by_binning`` mapping, keys as written.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If a key is neither an integer nor a string of ASCII
            digits, a key spells a binning below 2, a key has a leading zero, or the
            table holds an integer key and its digits.
    """
    for key in table:
        binning = _binning_of_key(key)
        if binning is None or binning < 2:
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH}: a key must be an integer binning factor of at '
                f'least 2; got {key!r}'
            )
    for key in table:
        if not isinstance(key, str):
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


def _check_noise_is_read(
    noise: Mapping[Any, Any], resolved: Mapping[str, Any], *, instrument: str, source: str
) -> None:
    """Refuse a star-limit noise value that no frame of the instrument reads.

    The star limit reads noise values only on a raw-DN block, and there only the
    binned-mode entry for the frame's binning.  A calibrated-unit block's limit is a
    constant that reads no noise value, so a binned-mode table and an unbinned
    ``read_noise_dn`` or ``star_flux_dn_per_s_vmag0`` are refused there, and on a raw-DN
    block an entry for a binning the instrument's readout modes don't offer is refused.
    Every key of the table is a binning factor of at least 2, since the key check refuses
    any other key first.

    Parameters:
        noise: The scene's ``noise`` overrides, table keys read as integers.
        resolved: The navigator block the scene's overrides resolve to.
        instrument: The scene's (already validated) instrument name.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the table or an unbinned star-limit value is set on
            a calibrated-unit block, or the table holds an entry for a binning the
            instrument doesn't offer.
    """
    units = resolved['data_units']
    if units != 'raw_dn':
        unread = f'renders {units} frames, whose star limit reads no noise value'
        if 'by_binning' in noise:
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH} is never read: instrument {instrument!r} '
                f'{unread}. Leave the table out.'
            )
        for key in sorted(BY_BINNING_VALUE_KEYS):
            if key in noise:
                raise SimSceneValidationError(
                    f'{source}: {_NOISE_PATH}.{key} is never read: instrument '
                    f'{instrument!r} {unread}. Leave the key out.'
                )
        return
    offered = cataloged_binnings(instrument)
    for key in noise.get('by_binning', {}):
        if key not in offered:
            listed = ', '.join(str(binning) for binning in offered)
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH}.{key} is never read: instrument '
                f'{instrument!r} offers binnings {listed}. Leave the entry out.'
            )


def _check_finite_number(value: Any, path: str, *, positive: bool, source: str) -> None:
    """Refuse a value that isn't a finite number within its range.

    The value has no null meaning, so a null is refused with the way to take the
    default.  An integer too large for a float is refused as not finite.

    Parameters:
        value: The value as the scene sets it.
        path: The value's dotted key path, for error messages.
        positive: True when the value must be greater than 0, False when it must be at
            least 0.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the value is null, is a boolean, isn't an integer or
            a float, isn't finite as a float, or lies outside its range.
    """
    if value is None:
        raise SimSceneValidationError(
            f'{source}: {path} is null; leave the key out to take its default'
        )
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    try:
        is_finite = is_number and math.isfinite(value)
    except OverflowError:
        is_finite = False
    in_range = is_finite and (value > 0 if positive else value >= 0)
    if not in_range:
        rule = 'greater than 0' if positive else 'of at least 0'
        raise SimSceneValidationError(
            f'{source}: {path} must be a finite number {rule}; got {value!r}'
        )


def _check_noise_values(noise: Mapping[Any, Any], path: str, *, source: str) -> None:
    """Refuse a read noise or a zero point that the faint-star limit can't use.

    A read noise of 0 is an ideal detector, which the limit reads as its 1 DN noise
    floor; a zero point of 0 would leave no star above the noise.

    Parameters:
        noise: A noise block, or one entry of its binned-mode table.
        path: The block's dotted key path, for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If ``read_noise_dn`` is set and isn't a finite number
            of at least 0, or ``star_flux_dn_per_s_vmag0`` is set and isn't a finite,
            positive number.
    """
    if 'read_noise_dn' in noise:
        _check_finite_number(
            noise['read_noise_dn'], f'{path}.read_noise_dn', positive=False, source=source
        )
    if 'star_flux_dn_per_s_vmag0' in noise:
        _check_finite_number(
            noise['star_flux_dn_per_s_vmag0'],
            f'{path}.star_flux_dn_per_s_vmag0',
            positive=True,
            source=source,
        )


def _check_star_limit_values(override: Mapping[str, Any], *, source: str) -> None:
    """Refuse a scene override of a value the faint-star limit can't use.

    The limit reads ``star_psf_sigma``, which must be a finite, positive number on every
    block, since other navigator code reads it too.  On a raw-DN block it reads the
    unbinned ``read_noise_dn`` and ``star_flux_dn_per_s_vmag0``, and on a binned frame
    the two values from the ``noise.by_binning`` entry for the frame's binning, so each
    entry the scene sets must be a mapping that sets no other key, and every value under
    the same rule as the unbinned ones.  Every binning a raw-DN block offers has a
    complete entry in the navigator's table, so a scene's entry can't leave one
    incomplete.  A calibrated-unit block's noise values are refused as never read before
    this check.

    Parameters:
        override: The scene's ``instrument_config``, table keys read as integers.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If a value the limit reads is null or out of range, or
            an entry of the binned-mode table isn't a mapping or sets another key.
    """
    if 'star_psf_sigma' in override:
        _check_finite_number(
            override['star_psf_sigma'],
            'instrument_config.star_psf_sigma',
            positive=True,
            source=source,
        )
    noise = override.get('noise', {})
    _check_noise_values(noise, _NOISE_PATH, source=source)
    for key, entry in noise.get('by_binning', {}).items():
        path = f'{_TABLE_PATH}.{key}'
        if not isinstance(entry, Mapping) or not set(entry) <= BY_BINNING_VALUE_KEYS:
            raise SimSceneValidationError(
                f'{source}: {path} must be a mapping that sets only '
                f'{" and ".join(sorted(BY_BINNING_VALUE_KEYS))}; got {entry!r}'
            )
        _check_noise_values(entry, path, source=source)


def _check_noise_override(noise: Any, *, source: str) -> None:
    """Refuse a scene's ``instrument_config.noise`` block unless its form is sound.

    The block must be a mapping.  Its binned-mode table, when set, must be a mapping with
    no null at any depth, keyed under the rules of :func:`_check_binning_keys`.

    Parameters:
        noise: The scene's ``instrument_config.noise`` value, as written.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the block or its table is null or isn't a mapping,
            the table holds a null, or a key of the table is refused.
    """
    if noise is None:
        raise SimSceneValidationError(
            f'{source}: {_NOISE_PATH} is null; leave the key out to take its default'
        )
    if not isinstance(noise, Mapping):
        raise SimSceneValidationError(f'{source}: {_NOISE_PATH} must be a mapping when present')
    if 'by_binning' not in noise:
        return
    table = noise['by_binning']
    if table is None:
        raise SimSceneValidationError(
            f'{source}: {_TABLE_PATH} is null; leave the key out to take its default'
        )
    if not isinstance(table, Mapping):
        raise SimSceneValidationError(
            f'{source}: {_TABLE_PATH} must be a mapping keyed by binning factor'
        )
    _refuse_nulls(table, _TABLE_PATH, source=source)
    _check_binning_keys(table, source=source)


def _check_instrument_config(value: Any, *, instrument: str, binning: int, source: str) -> None:
    """Validate the scene-level ``instrument_config`` block.

    The block can override any key of the navigator block except ``data_units``.  A
    frame's units come only from the instrument's config entry; for other units, a
    scene uses another instrument.

    When the block sets ``noise``, that must be a mapping.  A null value of a navigator
    key there keeps the navigator's meaning where it has one (a null ``marker_value`` is
    NaN).  A value the faint-star limit reads has no null meaning, and neither has the
    binned-mode table, so a null there, or at any depth of the table, is refused: a scene
    leaves a key out to take its default.  The binned-mode table ``noise.by_binning`` may
    write a key only as an integer or as a string of ASCII digits, since a scene read
    through JSON carries every key as a string, and each binning only once and without a
    leading zero.  Each noise value the star limit reads must be one that a frame of the
    scene's instrument reads: none on a calibrated-unit block, and on a raw-DN block only
    the binnings the instrument offers.  The navigator block the overrides resolve to,
    over the default configuration, sets the units.  Every value the scene sets that the
    faint-star limit reads must then be one it can use: a read noise is a finite number of
    at least 0, a zero point and ``star_psf_sigma`` are finite, positive numbers, and each
    entry of the binned-mode table is a mapping that sets nothing but a read noise and a
    zero point.  On a binned frame the star limit reads the mode's ``read_noise_dn`` and
    ``star_flux_dn_per_s_vmag0`` from that table, so a scene that sets either unbinned
    value must set the mode's value as well, or the value it sets would never be read.

    Parameters:
        value: The ``instrument_config`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name.
        binning: The scene's (already validated) readout binning.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the block isn't a mapping, it sets ``data_units``,
            its ``noise`` isn't a mapping, its ``noise.by_binning`` table isn't a
            mapping, holds a null or a key that is neither an integer nor a string of
            ASCII digits, spells a binning twice, or holds an entry no frame reads, the
            block sets an unbinned noise value on a calibrated-unit block, a value the
            star limit reads is null or out of range, a binned-mode entry sets another
            key, or the block sets an unbinned noise value that a frame of this binning
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
    if 'noise' in value:
        _check_noise_override(value['noise'], source=source)
    override = with_integer_binning_keys(value)
    noise = override.get('noise', {})
    resolved = resolve_sim_inst_config(DEFAULT_CONFIG, instrument, override)
    _check_noise_is_read(noise, resolved, instrument=instrument, source=source)
    _check_star_limit_values(override, source=source)
    if binning not in (resolved.get('noise') or {}).get('by_binning', {}):
        return
    mode = noise.get('by_binning', {}).get(binning, {})
    for key in sorted(BY_BINNING_VALUE_KEYS):
        if key in noise and key not in mode:
            raise SimSceneValidationError(
                f'{source}: {_NOISE_PATH}.{key} is set, but a frame binned by '
                f'{binning} reads {_TABLE_PATH}.{binning}.{key} '
                f'instead. Set {_TABLE_PATH}.{binning}.{key} too.'
            )
