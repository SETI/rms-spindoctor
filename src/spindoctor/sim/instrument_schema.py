"""The schema of a simulated instrument's config file.

Every simulated camera is one YAML file in ``spindoctor/sim/instrument_configs/``
(see :mod:`spindoctor.sim.instrument_catalog`).  This module states what such a
file may hold: the key inventory, each key's type and range, and which keys an
entry must carry.  The cross-checks that tie one part of an entry to another live
in :mod:`spindoctor.sim.instrument_checks`.  Both run when the catalog loads, so a
typo, a value written as a string, a negative gain, or a missing table fails at
load with the file and the key path in the message instead of rendering with a
fallback.

The schema is a small tree of nodes: a :class:`Record` has named fields, each
required or optional; a :class:`MapOf` is a table with uniformly typed keys and
values (a gain table, a filter table); a :class:`ListOf` is a list; and a
:class:`Scalar` is a number, an integer, a boolean, or a string, optionally from
a fixed set of choices or within a range.  A number is a finite integer or float,
never a boolean or a string.

A file with a ``base`` may set a key to null, which removes the key the base
would otherwise pass on; a file without a ``base`` has nothing to remove, so a
null there fails.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, DETECTOR_MODELS, ModeParam

__all__ = [
    'CATALOG_BLIND_MODES',
    'DETECTOR_COMMON_REQUIRED',
    'GENERIC_INSTRUMENT',
    'INSTRUMENT_DEFAULTS',
    'NONNEG',
    'NON_INHERITED_KEYS',
    'OFFER_EVERY_MODE',
    'POSITIVE',
    'QUANTIZATION_MODES',
    'SHAPE_DEFAULTS',
    'UNIT',
    'InstrumentConfigError',
    'ListOf',
    'MapOf',
    'ModeMaps',
    'ModeNames',
    'Record',
    'Scalar',
    'check_file',
    'check_mode_name',
    'check_node',
    'config_error',
    'instrument_schema',
]

# The name of the instrument-agnostic entry, which a scene renders through when it
# sets no instrument.  It is a file like any other (``generic.yaml``).
GENERIC_INSTRUMENT = 'generic'

# The value of ``artifact_modes.offered`` that offers every registered mode.
OFFER_EVERY_MODE = 'all'

# Keys an entry does not pass on to an entry that sets it as its ``base``: the base
# reference itself, the alternative names, and the real cameras the realism match
# pairs with the entry, each of which belongs to one entry only.
NON_INHERITED_KEYS: frozenset[str] = frozenset({'base', 'aliases', 'realism_cohort'})

# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC modes
# reproduce a coarse or uneven ADC's histogram structure, and 'sqrt_lut' companding
# decodes through a camera's downlink table.  The order is the order the sim editor
# lists them in.
QUANTIZATION_MODES: tuple[str, ...] = (
    'exact',
    '8bit',
    'uneven_12bit',
    'sqrt_lut',
    'ls8b',
    'contour_8bit',
)

# The two artifact-mode tables a file may state: the shape parameters a scene
# inherits when it sets a mode without spelling out every parameter, and the
# detector-stage modes the instrument_defaults switch turns on.
SHAPE_DEFAULTS = 'shape_defaults'
INSTRUMENT_DEFAULTS = 'instrument_defaults'

# Modes whose shape the renderer takes from somewhere other than the per-instrument
# shape defaults, so a shape default for one would do nothing.  Each maps to where
# its shape comes from instead.
CATALOG_BLIND_MODES: dict[str, str] = {
    'hot_pixels': (
        'its shape comes from detector.hot_pixel_amplitude_e and detector.hot_pixel_column_factor'
    ),
}

# The range tags a number or integer may carry.
POSITIVE = 'positive'
NONNEG = 'nonneg'
UNIT = 'unit'
# How an error message describes a value of each range, by number or integer.
_BOUND_NAMES: dict[str, str] = {
    POSITIVE: 'a positive {noun}',
    NONNEG: 'a non-negative {noun}',
    UNIT: 'a {noun} in [0, 1]',
}

# The scalar type tags, and how an error message describes each.
_NUMBER = 'number'
_INT = 'int'
_BOOL = 'bool'
_STR = 'str'
_KIND_NAMES: dict[str, str] = {
    _NUMBER: 'a number',
    _INT: 'an integer',
    _BOOL: 'a boolean',
    _STR: 'a string',
}

# The message for a null that has nothing to remove.
_NULL_WITHOUT_BASE = 'null removes an inherited key, and this file sets no base'
_NULL_IN_LIST = (
    "a list item cannot be null: a list replaces the base's whole list, so an item has "
    'nothing to remove'
)


class InstrumentConfigError(ValueError):
    """A simulated instrument's config file breaks the schema."""


@dataclass(frozen=True)
class Scalar:
    """A scalar value.

    Parameters:
        kind: ``'number'`` (a finite integer or float), ``'int'``, ``'bool'``, or
            ``'str'``.
        choices: The permitted values, or None for any value of the kind.
        bound: The range of a number or integer: ``'positive'``, ``'nonneg'``,
            ``'unit'`` (in [0, 1]), or None for any finite value.
    """

    kind: str
    choices: frozenset[Any] | None = None
    bound: str | None = None


@dataclass(frozen=True)
class MapOf:
    """A table whose keys share one scalar kind and whose values share one node.

    Parameters:
        key_kind: The scalar kind of every key, ``'int'`` or ``'str'``.
        value: The node every value must match.
        key_bound: The range of an integer key (see :class:`Scalar`), or None.
    """

    key_kind: str
    value: Node
    key_bound: str | None = None


@dataclass(frozen=True)
class ListOf:
    """A list whose items share one node.

    Parameters:
        item: The node every item must match.
    """

    item: Node


@dataclass(frozen=True)
class Record:
    """A mapping with a fixed key inventory.

    Parameters:
        fields: Each key's node.
        required: The keys a complete entry must carry.  The check of a single file
            skips this, since a file with a ``base`` inherits the rest.
    """

    fields: Mapping[str, Node]
    required: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ModeNames:
    """The artifact modes an instrument offers: ``'all'`` or a list of registered modes."""


@dataclass(frozen=True)
class ModeMaps:
    """A table of artifact-mode scene maps keyed by registered mode name.

    Each map's keys must be parameters of its mode, and each value must have the type
    and range the parameter's kind calls for.  A shape-default table may not carry a
    mode's activation key or fixed count (a shape default never turns a mode on),
    nor a mode the renderer resolves without the shape defaults.

    Parameters:
        section: :data:`SHAPE_DEFAULTS` or :data:`INSTRUMENT_DEFAULTS`.
    """

    section: str


Node = Scalar | MapOf | ListOf | Record | ModeNames | ModeMaps

_FINITE = Scalar(_NUMBER)
_POSITIVE_NUMBER = Scalar(_NUMBER, bound=POSITIVE)
_NONNEG_NUMBER = Scalar(_NUMBER, bound=NONNEG)
_UNIT_NUMBER = Scalar(_NUMBER, bound=UNIT)
_STR_NODE = Scalar(_STR)

# A downlink table is either parametric (a quadratic decode curve over ``codes``
# codes) or tabulated (the decoded DN of each code, in code order, none negative);
# the cross-checks require exactly one form and at least two distinct decoded
# values, and the detector resolver refuses a table that decodes above the ADC
# ceiling when a frame is companded through it.
_DOWNLINK_TABLE = Record(
    fields={
        'codes': Scalar(_INT, bound=POSITIVE),
        'linear_dn': _FINITE,
        'quadratic_dn': _FINITE,
        'whole_dn': Scalar(_BOOL),
        'decoded_dn': ListOf(_NONNEG_NUMBER),
    }
)

_VIDICON = Record(
    fields={
        'read_noise_line_dn': _NONNEG_NUMBER,
        'read_noise_pixel_dn': _NONNEG_NUMBER,
        'coherent_amplitude_dn': _NONNEG_NUMBER,
        'coherent_period_px': _POSITIVE_NUMBER,
    },
    required=frozenset(
        {'read_noise_line_dn', 'read_noise_pixel_dn', 'coherent_amplitude_dn', 'coherent_period_px'}
    ),
)

# The detector keys every entry carries, whatever its model.
DETECTOR_COMMON_REQUIRED: frozenset[str] = frozenset(
    {
        'detector_model',
        'exposure_ref_sec',
        'bias_dn',
        'quantization',
        'cosmic_ray_events_per_pixel_per_sec',
    }
)

_DETECTOR_FIELDS: dict[str, Node] = {
    'detector_model': Scalar(_STR, frozenset(DETECTOR_MODELS)),
    'exposure_ref_sec': _POSITIVE_NUMBER,
    'bias_dn': _NONNEG_NUMBER,
    'quantization': Scalar(_STR, frozenset(QUANTIZATION_MODES)),
    'full_well_e': _POSITIVE_NUMBER,
    'gain_e_per_dn_by_state': MapOf(_INT, _POSITIVE_NUMBER, key_bound=NONNEG),
    'default_gain_state': Scalar(_INT, bound=NONNEG),
    'read_noise_e': _NONNEG_NUMBER,
    'dark_current_e_per_sec': _NONNEG_NUMBER,
    'readout_dark_e': _NONNEG_NUMBER,
    'hot_pixel_fraction': _UNIT_NUMBER,
    'hot_pixel_amplitude_e': _NONNEG_NUMBER,
    'hot_pixel_column_factor': _UNIT_NUMBER,
    'cosmic_ray_events_per_pixel_per_sec': _NONNEG_NUMBER,
    'cosmic_ray_readout_dwell_sec': _NONNEG_NUMBER,
    'cosmic_ray_amplitude_e': _NONNEG_NUMBER,
    'cosmic_ray_spread_px': _NONNEG_NUMBER,
    'banding_amplitude_e': _NONNEG_NUMBER,
    'banding_period_px': _POSITIVE_NUMBER,
    'bias_pedestal_sigma_dn': _NONNEG_NUMBER,
    'bias_row_gradient_dn': _NONNEG_NUMBER,
    'bias_col_gradient_dn': _NONNEG_NUMBER,
    'bloom_length': Scalar(_INT, bound=NONNEG),
    'calibrated_sky_if': _FINITE,
    'calibration_column_residual_e': _NONNEG_NUMBER,
    'calibration_pixel_residual_e': _NONNEG_NUMBER,
    'vidicon': _VIDICON,
}

# A readout mode overrides any detector key; the keys it leaves out keep the 1x1
# values, so none is required.  The cross-checks refuse the keys a readout mode
# cannot change (the detector model and the vidicon chain).
_READOUT_MODE = Record(fields=_DETECTOR_FIELDS)

_SCHEMA = Record(
    fields={
        'base': _STR_NODE,
        'aliases': ListOf(_STR_NODE),
        'navigator': Record(
            fields={'section': _STR_NODE, 'detector': _STR_NODE},
            required=frozenset({'section'}),
        ),
        'detector_id': _STR_NODE,
        'detector': Record(fields=_DETECTOR_FIELDS, required=DETECTOR_COMMON_REQUIRED),
        'readout_modes': MapOf(_INT, _READOUT_MODE, key_bound=POSITIVE),
        'native_frame_size_px': Scalar(_INT, bound=POSITIVE),
        'star_flux': Record(
            fields={'e_per_s_vmag0': _POSITIVE_NUMBER, 'dn_per_s_vmag0': _POSITIVE_NUMBER}
        ),
        'sky_pixel_scale_arcsec': _POSITIVE_NUMBER,
        'psf': Record(
            fields={
                'kernel': Record(
                    fields={
                        'sigma_v': _POSITIVE_NUMBER,
                        'sigma_u': _POSITIVE_NUMBER,
                        'w': _UNIT_NUMBER,
                        'r0': _POSITIVE_NUMBER,
                        'n': _FINITE,
                    },
                    required=frozenset({'sigma_v', 'sigma_u', 'w', 'r0', 'n'}),
                ),
                'truncation_px': Scalar(_INT, bound=NONNEG),
            },
            required=frozenset({'truncation_px'}),
        ),
        'distortion_residual': Record(
            fields={'k1': _FINITE, 'k2': _FINITE, 'nonradial_rms_px': _NONNEG_NUMBER},
            required=frozenset({'k1', 'k2', 'nonradial_rms_px'}),
        ),
        'downlink_tables': Record(
            fields={'default': _STR_NODE, 'tables': MapOf(_STR, _DOWNLINK_TABLE)},
            required=frozenset({'default', 'tables'}),
        ),
        'artifact_modes': Record(
            fields={
                'offered': ModeNames(),
                'unavailable_reasons': MapOf(_STR, _STR_NODE),
                INSTRUMENT_DEFAULTS: ModeMaps(INSTRUMENT_DEFAULTS),
                SHAPE_DEFAULTS: ModeMaps(SHAPE_DEFAULTS),
            },
            required=frozenset({'offered'}),
        ),
        'filters': Record(
            fields={
                'throughput_e_per_s_per_if': MapOf(_STR, _POSITIVE_NUMBER),
                'clear': _STR_NODE,
                'polarizer_clear': MapOf(_STR, _STR_NODE),
            },
            required=frozenset({'throughput_e_per_s_per_if', 'clear'}),
        ),
        'realism_cohort': ListOf(
            Record(
                fields={'mission': _STR_NODE, 'camera': _STR_NODE},
                required=frozenset({'mission', 'camera'}),
            )
        ),
    },
    required=frozenset(
        {
            'navigator',
            'detector',
            'readout_modes',
            'star_flux',
            'sky_pixel_scale_arcsec',
            'psf',
            'artifact_modes',
        }
    ),
)


def instrument_schema() -> Record:
    """The root node of an instrument config file's schema."""
    return _SCHEMA


def config_error(source: str, path: str, message: str) -> InstrumentConfigError:
    """Build the error for one key, with the file and the key path in its message.

    Parameters:
        source: The file name.
        path: The dotted key path, or empty for the file as a whole.
        message: What is wrong.

    Returns:
        The error, ready to raise.
    """
    location = f'{source}: {path}' if path else source
    return InstrumentConfigError(f'{location}: {message}')


def _join(path: str, key: Any) -> str:
    """Extend a dotted key path by one key."""
    return f'{path}.{key}' if path else str(key)


def _describe(value: Any) -> str:
    """Describe a value's type for an error message, with a hint for a string number."""
    text = f'{type(value).__name__} {value!r}'
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return text
        return f'{text} (a number written so YAML reads it as a string)'
    return text


def _scalar_ok(kind: str, value: Any) -> bool:
    """Whether ``value`` is a scalar of ``kind``; a number must also be finite."""
    if isinstance(value, bool):
        return kind == _BOOL
    if kind == _NUMBER:
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    if kind == _INT:
        return isinstance(value, int)
    if kind == _STR:
        return isinstance(value, str)
    return False


def _in_bound(bound: str | None, value: float) -> bool:
    """Whether a number or integer lies in the range that ``bound`` stands for."""
    if bound == POSITIVE:
        return value > 0
    if bound == NONNEG:
        return value >= 0
    if bound == UNIT:
        return 0 <= value <= 1
    return True


def _check_scalar(node: Scalar, value: Any, *, source: str, path: str) -> None:
    """Check one scalar's type, choices, and range."""
    if not _scalar_ok(node.kind, value):
        finite = node.kind == _NUMBER and isinstance(value, float)
        expected = 'a finite number' if finite else _KIND_NAMES[node.kind]
        raise config_error(source, path, f'expected {expected}, got {_describe(value)}')
    if node.choices is not None and value not in node.choices:
        raise config_error(source, path, f'{value!r} is not one of {sorted(node.choices)}')
    if node.bound is not None and not _in_bound(node.bound, value):
        expected = _BOUND_NAMES[node.bound].format(
            noun='number' if node.kind == _NUMBER else 'integer'
        )
        raise config_error(source, path, f'expected {expected}, got {value!r}')


def check_node(
    node: Node, value: Any, *, source: str, path: str, complete: bool, allow_null: bool = False
) -> None:
    """Check one value against a schema node.

    Parameters:
        node: The schema node.
        value: The value read from the file.
        source: The file name, for error messages.
        path: The dotted key path of ``value``, for error messages.
        complete: Whether to require each record's required keys.  False for a
            single file, which may inherit keys from its base; True for an entry
            after inheritance.
        allow_null: Whether a null value is accepted (it removes the key the base
            passes on).  True only for a file that sets a base.

    Raises:
        InstrumentConfigError: On an unknown key, a missing required key, a null in
            a file without a base or in a list, or a value of the wrong type or range.
    """
    if value is None:
        if allow_null:
            return
        raise config_error(source, path, _NULL_WITHOUT_BASE)
    if isinstance(node, ModeNames):
        _check_offered(value, source=source, path=path)
        return
    if isinstance(node, Scalar):
        _check_scalar(node, value, source=source, path=path)
        return
    if isinstance(node, ListOf):
        if not isinstance(value, list):
            raise config_error(source, path, f'expected a list, got {_describe(value)}')
        # A list replaces the base's whole list, so an item has nothing to remove.
        for index, item in enumerate(value):
            item_path = f'{path}[{index}]'
            if item is None:
                raise config_error(source, item_path, _NULL_IN_LIST)
            check_node(node.item, item, source=source, path=item_path, complete=complete)
        return
    if not isinstance(value, Mapping):
        raise config_error(source, path, f'expected a mapping, got {_describe(value)}')
    if isinstance(node, ModeMaps):
        for key, item in value.items():
            _check_mode_map(
                key,
                item,
                node.section,
                source=source,
                path=_join(path, key),
                allow_null=allow_null,
            )
        return
    if isinstance(node, MapOf):
        key_node = Scalar(node.key_kind, bound=node.key_bound)
        for key, item in value.items():
            key_path = _join(path, key)
            _check_scalar(key_node, key, source=source, path=key_path)
            check_node(
                node.value,
                item,
                source=source,
                path=key_path,
                complete=complete,
                allow_null=allow_null,
            )
        return
    unknown = sorted(str(key) for key in value if key not in node.fields)
    if unknown:
        raise config_error(source, path, f'unknown keys: {unknown}')
    if complete:
        missing = sorted(node.required - set(value))
        if missing:
            raise config_error(source, path, f'missing required keys: {missing}')
    for key, item in value.items():
        check_node(
            node.fields[key],
            item,
            source=source,
            path=_join(path, key),
            complete=complete,
            allow_null=allow_null,
        )


def _check_offered(value: Any, *, source: str, path: str) -> None:
    """Check ``artifact_modes.offered``: ``'all'`` or a list of registered modes."""
    if value == OFFER_EVERY_MODE:
        return
    if not isinstance(value, list):
        raise config_error(
            source,
            path,
            f'expected {OFFER_EVERY_MODE!r} or a list of modes, got {_describe(value)}',
        )
    for index, name in enumerate(value):
        check_mode_name(name, source=source, path=f'{path}[{index}]')


def check_mode_name(name: Any, *, source: str, path: str) -> None:
    """Check that ``name`` is a registered artifact mode.

    Parameters:
        name: The value that should be a mode's name.
        source: The file name, for error messages.
        path: The dotted key path, for error messages.

    Raises:
        InstrumentConfigError: If ``name`` is not a registered mode.
    """
    if not isinstance(name, str) or name not in ARTIFACT_MODES:
        raise config_error(source, path, f'{name!r} is not a registered artifact mode')


# The range each integer mode-parameter kind carries.
_INT_KIND_BOUNDS: dict[str, str | None] = {
    'int': None,
    'nonneg_int': NONNEG,
    'positive_int': POSITIVE,
}


def _param_ok(param: ModeParam, value: Any) -> bool:
    """Whether ``value`` has the type and range a mode parameter's kind calls for."""
    kind = param.kind
    if kind == 'bool':
        return isinstance(value, bool)
    if kind in ('nonneg_number', 'unit_interval'):
        bound = NONNEG if kind == 'nonneg_number' else UNIT
        return _scalar_ok(_NUMBER, value) and _in_bound(bound, value)
    if kind in _INT_KIND_BOUNDS:
        return _scalar_ok(_INT, value) and _in_bound(_INT_KIND_BOUNDS[kind], value)
    if kind == 'enum':
        return param.choices is not None and value in param.choices
    if kind == 'int_list':
        return (
            isinstance(value, list)
            and len(value) == param.length
            and all(_scalar_ok(_INT, item) for item in value)
        )
    return False


def _check_mode_map(
    name: Any, value: Any, section: str, *, source: str, path: str, allow_null: bool
) -> None:
    """Check one mode's map: the mode exists and each parameter has its type and range."""
    check_mode_name(name, source=source, path=path)
    if value is None and allow_null:
        return
    if not isinstance(value, Mapping):
        raise config_error(source, path, f'expected a mapping, got {_describe(value)}')
    mode = ARTIFACT_MODES[name]
    if section == SHAPE_DEFAULTS and name in CATALOG_BLIND_MODES:
        raise config_error(
            source, path, f'a shape default for {name!r} does nothing: {CATALOG_BLIND_MODES[name]}'
        )
    params = mode.param_map
    for key, item in value.items():
        key_path = _join(path, key)
        if key not in params:
            raise config_error(source, key_path, f'{name!r} has no parameter {key!r}')
        if section == SHAPE_DEFAULTS and key in (mode.activation, mode.count_alternative):
            raise config_error(
                source,
                key_path,
                f'{key!r} turns the mode on, and a shape default never does; a scene sets '
                f'it, or the mode goes in artifact_modes.{INSTRUMENT_DEFAULTS}',
            )
        if item is None:
            if allow_null:
                continue
            raise config_error(source, key_path, _NULL_WITHOUT_BASE)
        if not _param_ok(params[key], item):
            raise config_error(
                source, key_path, f'expected a {params[key].kind} value, got {_describe(item)}'
            )


def check_file(entry: Any, *, source: str) -> dict[str, Any]:
    """Check one config file on its own, before inheritance.

    Unknown keys, types, and ranges are checked; required keys are not, since a
    file with a ``base`` inherits them.  A null is accepted only in a file with a
    base, and never on a key a base does not pass on.

    Parameters:
        entry: The value the file holds.
        source: The file name, for error messages.

    Returns:
        ``entry``, which is a mapping.

    Raises:
        InstrumentConfigError: If the file breaks the schema.
    """
    if not isinstance(entry, dict):
        raise config_error(source, '', 'the file must hold a mapping')
    for key in sorted(NON_INHERITED_KEYS & set(entry)):
        if entry[key] is None:
            raise config_error(
                source, key, f'{key} is not inherited from a base, so null removes nothing'
            )
    has_base = entry.get('base') is not None
    check_node(_SCHEMA, entry, source=source, path='', complete=False, allow_null=has_base)
    return entry
