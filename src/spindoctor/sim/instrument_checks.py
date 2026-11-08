"""The cross-checks of a simulated instrument's config entry.

:mod:`spindoctor.sim.instrument_schema` checks each key on its own; this module
checks how the parts of one entry, after inheritance, fit together: the detector
keys the detector model calls for, the gain table and readout modes, the star zero
point's unit, the filter table's spelling and its clear elements, the downlink
tables, and the artifact modes an entry names.  Each check refuses a setting that
the renderer could not honor or that would silently do nothing, naming the file
and the key path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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

    The entry must carry every required key, and its parts must fit together: the
    detector keys its model calls for are present, the gain states are contiguous
    and hold the default state, the readout modes include binning 1, divide the
    native frame size, and keep the detector model, the star zero point's unit
    fits the detector model, the filter table is spelled canonically and holds the
    clear filter and each polarizer's clear element, each downlink table has one
    form and the default names one of them, and every artifact mode the entry names
    is one it offers.

    Parameters:
        entry: The instrument's mapping after inheritance.
        source: The file name, for error messages.

    Raises:
        ~spindoctor.sim.instrument_schema.InstrumentConfigError: On any violation, naming
            the file and the key path.
    """
    check_node(instrument_schema(), entry, source=source, path='', complete=True)
    _check_detector(entry['detector'], source=source)
    _check_readout_modes(entry, source=source)
    _check_star_flux(entry, source=source)
    _check_filters(entry.get('filters'), source=source)
    _check_downlink_tables(entry.get('downlink_tables'), source=source)
    _check_artifact_modes(entry['artifact_modes'], source=source)


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
    state, if it names one, must be in the table.
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


def _check_filters(filters: Mapping[str, Any] | None, *, source: str) -> None:
    """Check the filter table's spelling, its clear filter, and the polarizers."""
    if filters is None:
        return
    table = filters['throughput_e_per_s_per_if']
    for combination in table:
        _check_canonical(combination, source=source, path='filters.throughput_e_per_s_per_if')
    if filters['clear'] not in table:
        raise config_error(
            source, 'filters.clear', f'{filters["clear"]!r} is not in throughput_e_per_s_per_if'
        )
    elements = {element for combination in table for element in combination.split('+')}
    for polarizer, clear in (filters.get('polarizer_clear') or {}).items():
        path = f'filters.polarizer_clear.{polarizer}'
        _check_canonical(polarizer, source=source, path='filters.polarizer_clear', element=True)
        if clear not in elements:
            raise config_error(
                source, path, f'{clear!r} is not an element of any cataloged filter combination'
            )


def _check_canonical(name: str, *, source: str, path: str, element: bool = False) -> None:
    """Refuse a filter spelling a scene's filter could never fold onto.

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
    """Check that each downlink table has one form and the default names a table."""
    if tables is None:
        return
    for name, table in tables['tables'].items():
        path = f'downlink_tables.tables.{name}'
        keys = set(table)
        if keys == {_TABULATED_TABLE_KEY}:
            if len(table[_TABULATED_TABLE_KEY]) < _MIN_TABLE_CODES:
                raise config_error(
                    source,
                    f'{path}.{_TABULATED_TABLE_KEY}',
                    f'a downlink table decodes at least {_MIN_TABLE_CODES} codes',
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
    if tables['default'] not in tables['tables']:
        raise config_error(
            source,
            'downlink_tables.default',
            f'{tables["default"]!r} is not one of the tables {sorted(tables["tables"])}',
        )


def _check_artifact_modes(modes: Mapping[str, Any], *, source: str) -> None:
    """Check that every mode the entry names is one it offers (or, for a reason, not)."""
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
