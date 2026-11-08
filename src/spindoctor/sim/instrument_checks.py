"""The cross-checks of a simulated instrument's config entry.

:mod:`spindoctor.sim.instrument_schema` checks each key on its own; this module
checks how the parts of one entry, after inheritance, fit together: the detector
keys the detector model calls for, the gain table and readout modes, the star zero
point's unit, the filter table's spelling and its clear elements, the downlink
tables, and the artifact modes an entry lists.  Each check refuses a setting that
the renderer could not honor or that would silently do nothing, with a message that
gives the file and the key path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, DETECTOR_MODE_ORDER
from spindoctor.sim.instrument_schema import (
    INSTRUMENT_DEFAULTS,
    OFFER_EVERY_MODE,
    SHAPE_DEFAULTS,
    check_mode_name,
    check_node,
    config_error,
    instrument_schema,
)

__all__ = ['canonical_filter_elements', 'check_entry']

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

# The detector keys a readout mode cannot change: a readout mode bins the same
# detector, so its model and its vidicon chain are the 1x1 mode's.
_READOUT_FIXED_KEYS: tuple[str, ...] = ('detector_model', 'vidicon')

# The keys of a parametric downlink table, and the key of a tabulated one.
_PARAMETRIC_TABLE_KEYS: frozenset[str] = frozenset(
    {'codes', 'linear_dn', 'quadratic_dn', 'whole_dn'}
)
_TABULATED_TABLE_KEY = 'decoded_dn'

# The fewest codes a downlink table may send.
_MIN_TABLE_CODES = 2


def canonical_filter_elements(filter_name: str) -> list[str]:
    """Split a filter combination into its elements as the catalog spells them.

    Each ``+``-separated element is stripped of surrounding whitespace and
    upper-cased, which is how a scene's ``detector.filter`` is read before lookup.

    Parameters:
        filter_name: A filter combination, such as ``'cl1 + grn'``.

    Returns:
        The elements, such as ``['CL1', 'GRN']``.
    """
    return [element.strip().upper() for element in filter_name.split('+')]


def check_entry(entry: Mapping[str, Any], *, source: str) -> None:
    """Check one instrument's whole entry, after inheritance.

    The entry must carry every required key, and its parts must fit together.  The
    detector keys its model calls for are present.  The gain states are contiguous
    and hold the default state.  The readout modes include binning 1, divide the
    native frame size, and keep the detector model.  The star zero point's unit
    fits the detector model.  A filter table belongs to a CCD, is spelled
    canonically, keys no polarizer, and holds the clear filter and each polarizer's
    clear element.  Each downlink table has one form and decodes to at least two
    levels, none negative, and the default is one of the tables.  Every artifact
    mode the entry lists is one it offers and one its detector model's chain
    applies.

    Parameters:
        entry: The instrument's mapping after inheritance.
        source: The file name, for error messages.

    Raises:
        ~spindoctor.sim.instrument_schema.InstrumentConfigError: On any violation, with
            a message that gives the file and the key path.
    """
    check_node(instrument_schema(), entry, source=source, path='', complete=True)
    _check_detector(entry['detector'], source=source)
    _check_readout_modes(entry, source=source)
    _check_star_flux(entry, source=source)
    model = entry['detector']['detector_model']
    _check_filters(entry.get('filters'), model=model, source=source)
    _check_downlink_tables(entry.get('downlink_tables'), source=source)
    _check_artifact_modes(entry['artifact_modes'], model=model, source=source)


def _check_detector(detector: Mapping[str, Any], *, source: str) -> None:
    """Check the model-specific detector keys and the gain table."""
    model = detector['detector_model']
    required = _CCD_REQUIRED if model == 'ccd' else _VIDICON_REQUIRED
    missing = sorted(required - set(detector))
    if missing:
        raise config_error(source, 'detector', f'a {model} detector requires the keys {missing}')
    if model != 'vidicon' and 'vidicon' in detector:
        raise config_error(
            source, 'detector.vidicon', f'a {model} detector has no vidicon chain to configure'
        )
    table = detector.get('gain_e_per_dn_by_state')
    if table is not None:
        _check_gain_states(table, source=source, path='detector.gain_e_per_dn_by_state')
    if 'default_gain_state' in detector:
        state = detector['default_gain_state']
        if state not in (table or {}):
            raise config_error(
                source,
                'detector.default_gain_state',
                f'state {state} is not in gain_e_per_dn_by_state {sorted(table or {})}',
            )


def _check_gain_states(table: Mapping[int, float], *, source: str, path: str) -> None:
    """Check that the gain states run contiguously upward from the lowest one."""
    states = sorted(table)
    expected = list(range(states[0], states[0] + len(states))) if states else []
    if states != expected:
        raise config_error(
            source,
            path,
            f'the gain states {states} must be contiguous integers from {states[0]}',
        )


def _check_readout_modes(entry: Mapping[str, Any], *, source: str) -> None:
    """Check the readout modes against the unbinned detector and the native size."""
    modes = entry['readout_modes']
    detector = entry['detector']
    if 1 not in modes:
        raise config_error(
            source, 'readout_modes', 'binning 1 (the unbinned readout) is not defined'
        )
    for binning, overrides in modes.items():
        path = f'readout_modes.{binning}'
        for key in _READOUT_FIXED_KEYS:
            if key in overrides:
                raise config_error(
                    source,
                    f'{path}.{key}',
                    f'a readout mode bins the same detector, so it cannot change detector.{key}',
                )
        if detector['detector_model'] == 'vidicon' and binning != 1:
            raise config_error(source, path, 'the vidicon DN chain has no binned readout')
        _check_readout_gain(detector, overrides, source=source, path=path)
    native = entry.get('native_frame_size_px')
    if native is None:
        return
    for binning in modes:
        if native % binning != 0:
            raise config_error(
                source,
                f'readout_modes.{binning}',
                f'binning {binning} does not divide native_frame_size_px {native}',
            )


def _check_readout_gain(
    detector: Mapping[str, Any], overrides: Mapping[str, Any], *, source: str, path: str
) -> None:
    """Check that a readout mode's gain table restates the unbinned table's states.

    A scene's gain state is validated against the unbinned table, so a readout
    mode may restate each state's gain but not add or drop a state, and its default
    state, if it sets one, must be in the table.
    """
    base_table = detector.get('gain_e_per_dn_by_state') or {}
    table = overrides.get('gain_e_per_dn_by_state', base_table)
    if sorted(table) != sorted(base_table):
        raise config_error(
            source,
            f'{path}.gain_e_per_dn_by_state',
            f'the states {sorted(table)} differ from detector.gain_e_per_dn_by_state '
            f'{sorted(base_table)}',
        )
    state = overrides.get('default_gain_state')
    if state is not None and state not in table:
        raise config_error(
            source,
            f'{path}.default_gain_state',
            f'state {state} is not in gain_e_per_dn_by_state {sorted(table)}',
        )


def _check_star_flux(entry: Mapping[str, Any], *, source: str) -> None:
    """Check that the star zero point is stated once, in the detector model's unit."""
    star_flux = entry['star_flux']
    if len(star_flux) != 1:
        raise config_error(
            source, 'star_flux', 'state exactly one of e_per_s_vmag0 and dn_per_s_vmag0'
        )
    model = entry['detector']['detector_model']
    expected = 'dn_per_s_vmag0' if model == 'vidicon' else 'e_per_s_vmag0'
    if expected not in star_flux:
        raise config_error(source, 'star_flux', f'a {model} detector states {expected}')


def _check_filters(filters: Mapping[str, Any] | None, *, model: str, source: str) -> None:
    """Check the filter table's detector, spelling, clear filter, and polarizers.

    A filter's throughput sets the CCD signal scale, which the vidicon DN chain
    does not have, so only a CCD carries a filter table.  A scene's polarizer folds
    onto its wheel's clear element before lookup, so a table key holding a
    polarizer could never be selected.
    """
    if filters is None:
        return
    if model == 'vidicon':
        raise config_error(
            source,
            'filters',
            'a filter sets the CCD signal scale; the vidicon DN chain has no reference '
            'exposure for it to set',
        )
    table = filters['throughput_e_per_s_per_if']
    polarizers = filters.get('polarizer_clear') or {}
    for combination in table:
        path = 'filters.throughput_e_per_s_per_if'
        _check_canonical(combination, source=source, path=path)
        for element in combination.split('+'):
            if element in polarizers:
                folded = '+'.join(polarizers.get(part, part) for part in combination.split('+'))
                raise config_error(
                    source,
                    f'{path}.{combination}',
                    f'the polarizer {element!r} folds onto {polarizers[element]!r}, so a scene '
                    f'that sets {combination!r} reads {folded!r}; key the throughput by that',
                )
    if filters['clear'] not in table:
        raise config_error(
            source, 'filters.clear', f'{filters["clear"]!r} is not in throughput_e_per_s_per_if'
        )
    elements = {element for combination in table for element in combination.split('+')}
    for polarizer, clear in polarizers.items():
        path = f'filters.polarizer_clear.{polarizer}'
        _check_canonical(polarizer, source=source, path='filters.polarizer_clear', element=True)
        if clear not in elements:
            raise config_error(
                source, path, f'{clear!r} is not an element of any cataloged filter combination'
            )


def _check_canonical(name: str, *, source: str, path: str, element: bool = False) -> None:
    """Refuse a filter spelling that a scene's filter could never fold onto.

    A scene's filter is stripped and upper-cased element by element before lookup,
    so a key spelled any other way could never be selected.
    """
    elements = canonical_filter_elements(name)
    canonical = '+'.join(elements)
    if element and len(elements) != 1:
        raise config_error(
            source, f'{path}.{name}', 'a polarizer is one filter element, without a +'
        )
    if '' in elements:
        raise config_error(source, f'{path}.{name}', f'{name!r} has an empty filter element')
    if canonical != name:
        raise config_error(
            source,
            f'{path}.{name}',
            f'{name!r} is not in canonical form; write {canonical!r} (a scene filter is read '
            'stripped and upper-cased)',
        )


def _check_downlink_tables(tables: Mapping[str, Any] | None, *, source: str) -> None:
    """Check each downlink table's form and decoded levels, and the default's name."""
    if tables is None:
        return
    for name, table in tables['tables'].items():
        path = f'downlink_tables.tables.{name}'
        keys = set(table)
        if keys == {_TABULATED_TABLE_KEY}:
            _check_decoded_levels(
                np.asarray(table[_TABULATED_TABLE_KEY], dtype=np.float64),
                source=source,
                path=f'{path}.{_TABULATED_TABLE_KEY}',
            )
            continue
        if keys != _PARAMETRIC_TABLE_KEYS:
            raise config_error(
                source,
                path,
                f'state either {_TABULATED_TABLE_KEY} alone or all of '
                f'{sorted(_PARAMETRIC_TABLE_KEYS)}; got {sorted(keys)}',
            )
        if table['codes'] < _MIN_TABLE_CODES:
            raise config_error(
                source,
                f'{path}.codes',
                f'a downlink table sends at least {_MIN_TABLE_CODES} codes, got {table["codes"]}',
            )
        _check_decoded_levels(_parametric_decoded_dn(table), source=source, path=path)
    if tables['default'] not in tables['tables']:
        raise config_error(
            source,
            'downlink_tables.default',
            f'{tables["default"]!r} is not one of the tables {sorted(tables["tables"])}',
        )


def _parametric_decoded_dn(table: Mapping[str, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """The DN each code of a parametric table decodes to, in code order.

    This is the curve :class:`~spindoctor.sim.forward.detector.sqrt_lut.SqrtLut`
    decodes through: ``linear_dn * c + quadratic_dn * c**2`` for code ``c``, rounded
    to whole DN when ``whole_dn`` is set.
    """
    codes = np.arange(table['codes'], dtype=np.float64)
    decoded = float(table['linear_dn']) * codes + float(table['quadratic_dn']) * codes**2
    return np.round(decoded) if table['whole_dn'] else decoded


def _check_decoded_levels(
    decoded: np.ndarray[Any, np.dtype[np.float64]], *, source: str, path: str
) -> None:
    """Refuse a downlink table with fewer than two decoded levels or a negative one.

    Codes that decode to one value are one quantization level, so a table needs as
    many distinct values as a working table needs codes; and the encoder snaps each
    DN, never negative, to the nearest level, so a negative level is DN no frame
    could hold.
    """
    distinct = len(np.unique(decoded))
    if distinct < _MIN_TABLE_CODES:
        raise config_error(
            source,
            path,
            f'a downlink table decodes to at least {_MIN_TABLE_CODES} distinct values, '
            f'got {distinct}',
        )
    lowest = float(decoded.min())
    if lowest < 0.0:
        raise config_error(
            source, path, f'a downlink table decodes to no negative DN, got {lowest:g}'
        )


def _check_artifact_modes(modes: Mapping[str, Any], *, model: str, source: str) -> None:
    """Check the entry's artifact modes against the modes it offers.

    A mode with shape defaults or an ``instrument_defaults`` entry must be offered,
    and a mode with an unavailability reason must not be.  If the entry offers a
    mode, gives it shape defaults, or turns it on, the detector model's chain must
    apply that mode, or a scene that enables it would render nothing.
    """
    offered = modes['offered']
    offered_set = set(ARTIFACT_MODES) if offered == OFFER_EVERY_MODE else set(offered)
    for section in (SHAPE_DEFAULTS, INSTRUMENT_DEFAULTS):
        for name in modes.get(section) or {}:
            if name not in offered_set:
                raise config_error(
                    source,
                    f'artifact_modes.{section}.{name}',
                    f'{name!r} is not in artifact_modes.offered',
                )
    named = [] if offered == OFFER_EVERY_MODE else [('offered', name) for name in offered]
    for section in (SHAPE_DEFAULTS, INSTRUMENT_DEFAULTS):
        named.extend((f'{section}.{name}', name) for name in modes.get(section) or {})
    for key, name in named:
        models = ARTIFACT_MODES[name].detector_models
        if model not in models:
            raise config_error(
                source,
                f'artifact_modes.{key}',
                f'{name!r} renders only in the {" and ".join(sorted(models))} detector chain; '
                f'a {model} detector never applies it',
            )
    for name, scene_map in (modes.get(INSTRUMENT_DEFAULTS) or {}).items():
        path = f'artifact_modes.{INSTRUMENT_DEFAULTS}.{name}'
        if name not in DETECTOR_MODE_ORDER:
            raise config_error(
                source, path, f'{name!r} is not a detector-stage mode the detector chain consults'
            )
        activation = ARTIFACT_MODES[name].activation
        if not scene_map.get(activation):
            raise config_error(
                source,
                path,
                f'an instrument_defaults entry turns the mode on, so it sets a positive '
                f'{activation!r}',
            )
    for name in modes.get('unavailable_reasons') or {}:
        path = f'artifact_modes.unavailable_reasons.{name}'
        check_mode_name(name, source=source, path=path)
        if name in offered_set:
            raise config_error(
                source, path, f'{name!r} is offered, so it has no unavailability reason'
            )
