"""The schema of a simulated instrument's config file, and its checks.

Every simulated camera is one YAML file in ``spindoctor/sim/instrument_configs/``
(see :mod:`spindoctor.sim.instrument_catalog`).  This module states what such a
file may hold: the key inventory, each key's type, which keys an entry must
carry, and the cross-checks that tie one part of an entry to another (a gain
state, readout mode, filter, or artifact mode the entry names must be one it
defines).  The checks run when the catalog loads, so a typo, a value written as a
string, or a missing table fails at load with the file and the key path in the
message instead of rendering with a fallback.

The schema is a small tree of nodes: a :class:`Record` has named fields, each
required or optional; a :class:`MapOf` is a table with uniformly typed keys and
values (a gain table, a filter table); a :class:`ListOf` is a list; and a
:class:`Scalar` is a number, an integer, a boolean, or a string, optionally from
a fixed set of choices.  A number is an integer or a float, never a boolean or a
string.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, DETECTOR_MODE_ORDER, ModeParam

__all__ = [
    'DETECTOR_MODELS',
    'GENERIC_INSTRUMENT',
    'NON_INHERITED_KEYS',
    'OFFER_EVERY_MODE',
    'QUANTIZATION_MODES',
    'InstrumentConfigError',
    'ListOf',
    'MapOf',
    'ModeMaps',
    'ModeNames',
    'Record',
    'Scalar',
    'check_entry',
    'check_node',
]

# The name of the instrument-agnostic entry every other name falls back to when a
# scene names no instrument.  It is a file like any other (``generic.yaml``).
GENERIC_INSTRUMENT = 'generic'

# The value of ``artifact_modes.offered`` that offers every registered mode.
OFFER_EVERY_MODE = 'all'

# Keys an entry does not pass to an entry that names it as its ``base``: the base
# reference itself, and the alternative names, which belong to one entry only.
NON_INHERITED_KEYS: frozenset[str] = frozenset({'base', 'aliases'})

# The detector models the detector stage implements: the CCD electron chain and the
# vidicon DN chain.
DETECTOR_MODELS: frozenset[str] = frozenset({'ccd', 'vidicon'})

# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC modes
# reproduce the documented histogram structure of each camera.
QUANTIZATION_MODES: frozenset[str] = frozenset(
    {'exact', 'uneven_12bit', '8bit', 'sqrt_lut', 'ls8b', 'contour_8bit'}
)

# The scalar type tags, and how an error message names each.
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


class InstrumentConfigError(ValueError):
    """A simulated instrument's config file breaks the schema."""


@dataclass(frozen=True)
class Scalar:
    """A scalar value.

    Parameters:
        kind: ``'number'`` (an integer or a float), ``'int'``, ``'bool'``, or ``'str'``.
        choices: The permitted values, or None for any value of the kind.
    """

    kind: str
    choices: frozenset[Any] | None = None


@dataclass(frozen=True)
class MapOf:
    """A table whose keys share one scalar kind and whose values share one node.

    Parameters:
        key_kind: The scalar kind of every key, ``'int'`` or ``'str'``.
        value: The node every value must match.
    """

    key_kind: str
    value: Node


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
    the parameter's kind calls for.
    """


Node = Scalar | MapOf | ListOf | Record | ModeNames | ModeMaps

_NUMBER_NODE = Scalar(_NUMBER)
_INT_NODE = Scalar(_INT)
_STR_NODE = Scalar(_STR)

# The detector keys every entry carries, whatever its model.
_DETECTOR_COMMON_REQUIRED: frozenset[str] = frozenset(
    {
        'detector_model',
        'exposure_ref_sec',
        'bias_dn',
        'quantization',
        'cosmic_ray_events_per_pixel_per_sec',
    }
)

# The electron-chain keys a CCD entry carries.
_CCD_REQUIRED: frozenset[str] = frozenset(
    {
        'full_well_e',
        'gain_e_per_dn_by_state',
        'default_gain_state',
        'read_noise_e',
        'dark_current_e_per_sec',
        'hot_pixel_fraction',
        'hot_pixel_amplitude_e',
        'hot_pixel_column_factor',
        'banding_amplitude_e',
        'banding_period_px',
        'bias_pedestal_sigma_dn',
        'bias_row_gradient_dn',
        'bias_col_gradient_dn',
        'bloom_length',
    }
)

# The DN-chain key a vidicon entry carries.
_VIDICON_REQUIRED: frozenset[str] = frozenset({'vidicon'})

_SQRT_LUT = Record(
    fields={
        'codes': _INT_NODE,
        'linear_dn': _NUMBER_NODE,
        'quadratic_dn': _NUMBER_NODE,
        'whole_dn': Scalar(_BOOL),
    },
    required=frozenset({'codes', 'linear_dn', 'quadratic_dn', 'whole_dn'}),
)

_VIDICON = Record(
    fields={
        'read_noise_line_dn': _NUMBER_NODE,
        'read_noise_pixel_dn': _NUMBER_NODE,
        'coherent_amplitude_dn': _NUMBER_NODE,
        'coherent_period_px': _NUMBER_NODE,
    },
    required=frozenset(
        {'read_noise_line_dn', 'read_noise_pixel_dn', 'coherent_amplitude_dn', 'coherent_period_px'}
    ),
)

_DETECTOR_FIELDS: dict[str, Node] = {
    'detector_model': Scalar(_STR, DETECTOR_MODELS),
    'exposure_ref_sec': _NUMBER_NODE,
    'bias_dn': _NUMBER_NODE,
    'quantization': Scalar(_STR, QUANTIZATION_MODES),
    'full_well_e': _NUMBER_NODE,
    'gain_e_per_dn_by_state': MapOf(_INT, _NUMBER_NODE),
    'default_gain_state': _INT_NODE,
    'read_noise_e': _NUMBER_NODE,
    'dark_current_e_per_sec': _NUMBER_NODE,
    'readout_dark_e': _NUMBER_NODE,
    'hot_pixel_fraction': _NUMBER_NODE,
    'hot_pixel_amplitude_e': _NUMBER_NODE,
    'hot_pixel_column_factor': _NUMBER_NODE,
    'cosmic_ray_events_per_pixel_per_sec': _NUMBER_NODE,
    'cosmic_ray_readout_dwell_sec': _NUMBER_NODE,
    'cosmic_ray_amplitude_e': _NUMBER_NODE,
    'cosmic_ray_spread_px': _NUMBER_NODE,
    'banding_amplitude_e': _NUMBER_NODE,
    'banding_period_px': _NUMBER_NODE,
    'bias_pedestal_sigma_dn': _NUMBER_NODE,
    'bias_row_gradient_dn': _NUMBER_NODE,
    'bias_col_gradient_dn': _NUMBER_NODE,
    'bloom_length': _INT_NODE,
    'sqrt_lut': _SQRT_LUT,
    'calibrated_sky_if': _NUMBER_NODE,
    'calibration_column_residual_e': _NUMBER_NODE,
    'calibration_pixel_residual_e': _NUMBER_NODE,
    'vidicon': _VIDICON,
}

# A readout mode overrides any detector key; the keys it leaves out keep the 1x1
# values, so none is required.
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
        'detector': Record(fields=_DETECTOR_FIELDS, required=_DETECTOR_COMMON_REQUIRED),
        'readout_modes': MapOf(_INT, _READOUT_MODE),
        'native_frame_size_px': _INT_NODE,
        'star_flux': Record(fields={'e_per_s_vmag0': _NUMBER_NODE, 'dn_per_s_vmag0': _NUMBER_NODE}),
        'sky_pixel_scale_arcsec': _NUMBER_NODE,
        'psf': Record(
            fields={
                'kernel': Record(
                    fields=dict.fromkeys(('sigma_v', 'sigma_u', 'w', 'r0', 'n'), _NUMBER_NODE),
                    required=frozenset({'sigma_v', 'sigma_u', 'w', 'r0', 'n'}),
                ),
                'truncation_px': _INT_NODE,
            },
            required=frozenset({'truncation_px'}),
        ),
        'distortion_residual': Record(
            fields=dict.fromkeys(('k1', 'k2', 'nonradial_rms_px'), _NUMBER_NODE),
            required=frozenset({'k1', 'k2', 'nonradial_rms_px'}),
        ),
        'artifact_modes': Record(
            fields={
                'offered': ModeNames(),
                'unavailable_reasons': MapOf(_STR, _STR_NODE),
                'instrument_defaults': ModeMaps(),
                'shape_defaults': ModeMaps(),
            },
            required=frozenset({'offered'}),
        ),
        'filters': Record(
            fields={
                'throughput_e_per_s_per_if': MapOf(_STR, _NUMBER_NODE),
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


def _fail(source: str, path: str, message: str) -> InstrumentConfigError:
    """Build the error for one key, naming the file and the key path."""
    location = f'{source}: {path}' if path else source
    return InstrumentConfigError(f'{location}: {message}')


def _join(path: str, key: Any) -> str:
    """Extend a dotted key path by one key."""
    return f'{path}.{key}' if path else str(key)


def _describe(value: Any) -> str:
    """Name a value's type for an error message, with a hint for a string number."""
    text = f'{type(value).__name__} {value!r}'
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return text
        return f'{text} (a number written so YAML reads it as a string)'
    return text


def _scalar_ok(kind: str, value: Any) -> bool:
    """Whether ``value`` is a scalar of ``kind``."""
    if isinstance(value, bool):
        return kind == _BOOL
    if kind == _NUMBER:
        return isinstance(value, int | float)
    if kind == _INT:
        return isinstance(value, int)
    if kind == _STR:
        return isinstance(value, str)
    return False


def check_node(node: Node, value: Any, *, source: str, path: str, complete: bool) -> None:
    """Check one value against a schema node.

    Parameters:
        node: The schema node.
        value: The value read from the file.
        source: The file name, for error messages.
        path: The dotted key path of ``value``, for error messages.
        complete: Whether to require each record's required keys.  False for a
            single file, which may inherit keys from its base; True for an entry
            after inheritance.

    Raises:
        InstrumentConfigError: On an unknown key, a missing required key, or a value
            of the wrong type.
    """
    if isinstance(node, ModeNames):
        _check_offered(value, source=source, path=path)
        return
    if isinstance(node, Scalar):
        if not _scalar_ok(node.kind, value):
            raise _fail(source, path, f'expected {_KIND_NAMES[node.kind]}, got {_describe(value)}')
        if node.choices is not None and value not in node.choices:
            raise _fail(source, path, f'{value!r} is not one of {sorted(node.choices)}')
        return
    if isinstance(node, ListOf):
        if not isinstance(value, list):
            raise _fail(source, path, f'expected a list, got {_describe(value)}')
        for index, item in enumerate(value):
            check_node(node.item, item, source=source, path=f'{path}[{index}]', complete=complete)
        return
    if not isinstance(value, Mapping):
        raise _fail(source, path, f'expected a mapping, got {_describe(value)}')
    if isinstance(node, ModeMaps):
        for key, item in value.items():
            _check_mode_map(key, item, source=source, path=_join(path, key))
        return
    if isinstance(node, MapOf):
        for key, item in value.items():
            if not _scalar_ok(node.key_kind, key):
                raise _fail(
                    source,
                    _join(path, key),
                    f'expected {_KIND_NAMES[node.key_kind]} key, got {_describe(key)}',
                )
            check_node(node.value, item, source=source, path=_join(path, key), complete=complete)
        return
    unknown = sorted(str(key) for key in value if key not in node.fields)
    if unknown:
        raise _fail(source, path, f'unknown keys: {unknown}')
    if complete:
        missing = sorted(node.required - set(value))
        if missing:
            raise _fail(source, path, f'missing required keys: {missing}')
    for key, item in value.items():
        check_node(node.fields[key], item, source=source, path=_join(path, key), complete=complete)


def _check_offered(value: Any, *, source: str, path: str) -> None:
    """Check ``artifact_modes.offered``: ``'all'`` or a list of registered modes."""
    if value == OFFER_EVERY_MODE:
        return
    if not isinstance(value, list):
        raise _fail(
            source,
            path,
            f'expected {OFFER_EVERY_MODE!r} or a list of modes, got {_describe(value)}',
        )
    for index, name in enumerate(value):
        _check_mode_name(name, source=source, path=f'{path}[{index}]')


def _check_mode_name(name: Any, *, source: str, path: str) -> None:
    """Check that ``name`` is a registered artifact mode."""
    if not isinstance(name, str) or name not in ARTIFACT_MODES:
        raise _fail(source, path, f'{name!r} is not a registered artifact mode')


def _param_ok(param: ModeParam, value: Any) -> bool:
    """Whether ``value`` has the type a mode parameter's kind calls for."""
    if param.kind == 'bool':
        return isinstance(value, bool)
    if param.kind in ('nonneg_number', 'unit_interval'):
        return _scalar_ok(_NUMBER, value)
    if param.kind in ('int', 'nonneg_int', 'positive_int'):
        return _scalar_ok(_INT, value)
    if param.kind == 'enum':
        return param.choices is not None and value in param.choices
    if param.kind == 'int_list':
        return (
            isinstance(value, list)
            and len(value) == param.length
            and all(_scalar_ok(_INT, item) for item in value)
        )
    return False


def _check_mode_map(name: Any, value: Any, *, source: str, path: str) -> None:
    """Check one mode's map: the mode exists and each parameter has its type."""
    _check_mode_name(name, source=source, path=path)
    if not isinstance(value, Mapping):
        raise _fail(source, path, f'expected a mapping, got {_describe(value)}')
    params = ARTIFACT_MODES[name].param_map
    for key, item in value.items():
        if key not in params:
            raise _fail(source, _join(path, key), f'{name!r} has no parameter {key!r}')
        if not _param_ok(params[key], item):
            raise _fail(
                source,
                _join(path, key),
                f'expected a {params[key].kind} value, got {_describe(item)}',
            )


def check_entry(entry: Mapping[str, Any], *, source: str, complete: bool) -> None:
    """Check one instrument's config against the schema.

    With ``complete`` set, the entry is one after inheritance, and the
    cross-checks run as well: the detector keys its model calls for are present,
    the default gain state is in the gain table, the readout modes include binning
    1 and divide the native frame size, the star zero point's unit fits the detector
    model, the clear filter and each polarizer's clear element are cataloged, and
    every artifact mode the entry names is one it offers.

    Parameters:
        entry: The instrument's mapping.
        source: The file name, for error messages.
        complete: Whether ``entry`` is a whole entry after inheritance.

    Raises:
        InstrumentConfigError: On any violation, naming the file and the key path.
    """
    check_node(_SCHEMA, entry, source=source, path='', complete=complete)
    if complete:
        _check_detector(entry['detector'], source=source)
        _check_readout_modes(entry, source=source)
        _check_star_flux(entry, source=source)
        _check_filters(entry.get('filters'), source=source)
        _check_artifact_modes(entry['artifact_modes'], source=source)


def _check_detector(detector: Mapping[str, Any], *, source: str) -> None:
    """Check the model-specific detector keys and the default gain state."""
    model = detector['detector_model']
    required = _CCD_REQUIRED if model == 'ccd' else _VIDICON_REQUIRED
    missing = sorted(required - set(detector))
    if missing:
        raise _fail(source, 'detector', f'a {model} detector requires the keys {missing}')
    if 'default_gain_state' in detector:
        table = detector.get('gain_e_per_dn_by_state') or {}
        state = detector['default_gain_state']
        if state not in table:
            raise _fail(
                source,
                'detector.default_gain_state',
                f'state {state} is not in gain_e_per_dn_by_state {sorted(table)}',
            )


def _check_readout_modes(entry: Mapping[str, Any], *, source: str) -> None:
    """Check that binning 1 is cataloged and every binning divides the native size."""
    modes = entry['readout_modes']
    if 1 not in modes:
        raise _fail(source, 'readout_modes', 'binning 1 (the unbinned readout) is not defined')
    native = entry.get('native_frame_size_px')
    if native is None:
        return
    for binning in modes:
        if binning <= 0 or native % binning != 0:
            raise _fail(
                source,
                f'readout_modes.{binning}',
                f'binning {binning} does not divide native_frame_size_px {native}',
            )


def _check_star_flux(entry: Mapping[str, Any], *, source: str) -> None:
    """Check that the star zero point is stated once, in the detector model's unit."""
    star_flux = entry['star_flux']
    if len(star_flux) != 1:
        raise _fail(source, 'star_flux', 'state exactly one of e_per_s_vmag0 and dn_per_s_vmag0')
    expected = (
        'dn_per_s_vmag0' if entry['detector']['detector_model'] == 'vidicon' else ('e_per_s_vmag0')
    )
    if expected not in star_flux:
        raise _fail(
            source,
            'star_flux',
            f'a {entry["detector"]["detector_model"]} detector states {expected}',
        )


def _check_filters(filters: Mapping[str, Any] | None, *, source: str) -> None:
    """Check that the clear filter and each polarizer's clear element are cataloged."""
    if filters is None:
        return
    table = filters['throughput_e_per_s_per_if']
    if filters['clear'] not in table:
        raise _fail(
            source, 'filters.clear', f'{filters["clear"]!r} is not in throughput_e_per_s_per_if'
        )
    elements = {element for combination in table for element in combination.split('+')}
    for polarizer, clear in (filters.get('polarizer_clear') or {}).items():
        if clear not in elements:
            raise _fail(
                source,
                f'filters.polarizer_clear.{polarizer}',
                f'{clear!r} is not an element of any cataloged filter combination',
            )


def _check_artifact_modes(modes: Mapping[str, Any], *, source: str) -> None:
    """Check that every mode the entry names is one it offers (or, for a reason, not)."""
    offered = modes['offered']
    offered_set = set(ARTIFACT_MODES) if offered == OFFER_EVERY_MODE else set(offered)
    for section in ('shape_defaults', 'instrument_defaults'):
        for name in modes.get(section) or {}:
            if name not in offered_set:
                raise _fail(
                    source,
                    f'artifact_modes.{section}.{name}',
                    f'{name!r} is not in artifact_modes.offered',
                )
    for name in modes.get('instrument_defaults') or {}:
        if name not in DETECTOR_MODE_ORDER:
            raise _fail(
                source,
                f'artifact_modes.instrument_defaults.{name}',
                f'{name!r} is not a detector-stage mode the detector chain consults',
            )
    for name in modes.get('unavailable_reasons') or {}:
        _check_mode_name(name, source=source, path=f'artifact_modes.unavailable_reasons.{name}')
        if name in offered_set:
            raise _fail(
                source,
                f'artifact_modes.unavailable_reasons.{name}',
                f'{name!r} is offered, so it has no unavailability reason',
            )
