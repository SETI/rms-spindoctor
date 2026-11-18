"""Mapping from a sim instrument name to its per-instrument config block.

The simulator can render a frame as though it came from a specific camera, so
that a 'sim COISS NAC raw' frame goes through the same per-instrument noise,
saturation, marker, and unit settings the navigator applies to a real frame.
This module resolves a sim ``instrument`` name to the navigator config block that
the instrument's config entry points at (its ``navigator`` section and detector; see
:mod:`spindoctor.sim.instrument_catalog`); physical parameters then delegate to
that block, while sim-only knobs (signal full-scale, cosmic-ray rate, dropout
rate) stay in the ``sim`` config block.

A value of ``None`` (or the generic entry and its aliases) selects the standalone
``sim`` block, preserving the instrument-agnostic defaults.

A scene may also carry ``instrument_config`` overrides that are deep-merged over
the resolved block, so a scene can:

- **inherit** every physical parameter from a chosen instrument (no overrides),
- **inherit and override** individual parameters (override only those keys), or
- **fully self-specify** by choosing the generic block and overriding every key
  except ``data_units``, which a scene can't set.

Overriding a key pins it to the scene, so a later change to the real camera's
config cannot silently shift a sim test's behavior for that key; the non-overridden
keys continue to track the instrument.

A rendered frame's units come from the instrument's config entry (``data_units``).
The navigator reads the frame in the units its block sets, so resolving the block
fails if the two differ.  It also fails if the block doesn't set ``data_units``,
which the navigator requires.  A ``raw_dn`` instrument renders detector counts
with a ``0`` missing-data marker.  A ``calibrated_if`` instrument (the calibrated
Cassini cameras and Voyager ISS) renders I/F, also with a ``0`` marker, because the
archived Cassini CALIB and Voyager GEOMED products fill lost pixels with zero.
:func:`read_frame_as_navigator` classifies a frame with the navigator's own image
quality decision (:mod:`spindoctor.support.image_class`), and
:func:`frame_units_summary` states a rendered frame's units, range, marker share,
and that verdict for a scene author.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.config import Config
from spindoctor.sim.instrument_catalog import InstrumentCatalog, load_instrument_catalog
from spindoctor.sim.instrument_schema import config_error
from spindoctor.support.image_class import (
    decide_image_class,
    image_quality_thresholds,
    measure_image_quality,
    missing_data_mask,
)
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'NavigatorFrameReading',
    'frame_units_name',
    'frame_units_summary',
    'has_planted_offset',
    'navigator_matched_psf',
    'read_frame_as_navigator',
    'resolve_extfov_margin',
    'resolve_sim_inst_config',
    'with_integer_binning_keys',
]

# The offset keys a navigated frame applies and an editor preview zeroes.
_PLANTED_OFFSET_KEYS = ('offset_v', 'offset_u', 'offset_rotation_deg')


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` recursively merged in.

    Nested mappings merge key-by-key; any non-mapping value (scalar or list) in
    ``overrides`` replaces the base value outright. The result is a fresh dict, so
    the returned config is decoupled from the live config blocks it was built from.
    """
    merged: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value for key, value in base.items()
    }
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def with_integer_binning_keys(overrides: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``overrides`` with each digit-string ``noise.by_binning`` key read as an integer.

    The navigator block keys its binned readout modes by the binning factor, an
    integer.  JSON object keys are strings, so a scene that passes through JSON, as
    every validated scene does, carries a key ``2`` as ``'2'``; both spellings name the
    same mode.  Any other key is left for scene validation to refuse.

    Parameters:
        overrides: A scene's ``instrument_config`` overrides.

    Returns:
        ``overrides`` itself when it sets no ``noise.by_binning`` mapping, otherwise a
        copy whose table has integer keys where the scene wrote digits.
    """
    noise = overrides.get('noise')
    if not isinstance(noise, Mapping) or not isinstance(noise.get('by_binning'), Mapping):
        return overrides
    table: Mapping[Any, Any] = noise['by_binning']
    keyed = {
        int(key) if isinstance(key, str) and key.isascii() and key.isdigit() else key: value
        for key, value in table.items()
    }
    return {**overrides, 'noise': {**noise, 'by_binning': keyed}}


def resolve_sim_inst_config(
    config: Config,
    instrument: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Resolve a sim instrument name to its per-instrument config block.

    Parameters:
        config: The active configuration.
        instrument: A sim instrument name (see
            :func:`~spindoctor.sim.instrument_catalog.sim_instrument_names`), the
            generic entry or one of its aliases, or ``None`` for the
            instrument-agnostic block.
        overrides: Optional scene-level overrides deep-merged over the resolved
            block. Overridden keys are pinned to the scene; the rest continue to
            track the instrument. Any key except ``data_units`` can be overridden,
            so the generic block with every other key overridden is a fully
            self-specified scene config.  A ``noise.by_binning`` key written as a
            string of digits names the binning factor it spells.

    Returns:
        The resolved per-instrument config mapping (a fresh dict when ``overrides``
        are supplied, otherwise the live config block).

    Raises:
        ValueError: If ``instrument`` is unrecognized, the referenced config section
            or detector is missing, or ``overrides`` sets ``data_units``.
        ~spindoctor.sim.instrument_schema.InstrumentConfigError: If the referenced
            block doesn't set ``data_units``, or the entry's ``data_units`` differ
            from the block's.  The message gives the entry's file and name and the
            block; for a mismatch it also gives both units.
    """
    if overrides and 'data_units' in overrides:
        raise ValueError(
            "instrument_config: a scene can't set data_units. Choose an instrument whose "
            'entry renders the units you want.'
        )
    base = _resolve_base_block(config, instrument)
    if not overrides:
        return base
    return _deep_merge(base, with_integer_binning_keys(overrides))


def _resolve_base_block(config: Config, instrument: str | None) -> Mapping[str, Any]:
    """Resolve the unmodified per-instrument (or generic) config block."""
    catalog = load_instrument_catalog()
    navigator = catalog.entry(instrument)['navigator']
    section: str = navigator['section']
    detector: str | None = navigator.get('detector')
    category = config.category(section)
    if not category:
        raise ValueError(
            f'sim instrument {instrument!r} references config section {section!r}, '
            'which is missing or empty'
        )
    if detector is None:
        _check_units_agree(catalog, instrument, category, block_name=section)
        return category
    if detector not in category:
        raise ValueError(
            f'sim instrument {instrument!r} references detector {detector!r} in '
            f'section {section!r}, which has no such entry'
        )
    detector_block = category[detector]
    if not isinstance(detector_block, Mapping):
        raise ValueError(
            f'sim instrument {instrument!r} detector block {section}.{detector} '
            f'is not a mapping; got {type(detector_block).__name__}'
        )
    _check_units_agree(catalog, instrument, detector_block, block_name=f'{section}.{detector}')
    return detector_block


def _check_units_agree(
    catalog: InstrumentCatalog,
    instrument: str | None,
    block: Mapping[str, Any],
    *,
    block_name: str,
) -> None:
    """Refuse an entry whose frame units differ from the units its navigator block sets.

    The navigator reads a frame in its block's ``data_units`` and refuses a frame
    whose block doesn't set them, so a block without them is refused here as well.
    If the entry
    rendered I/F against a ``raw_dn`` block, the navigator would read every frame as
    DN, and the other way round.
    """
    name = catalog.canonical_name(instrument)
    source = catalog.sources[name]
    if 'data_units' not in block:
        raise config_error(
            source,
            'data_units',
            f"the navigator block {block_name} doesn't set data_units, which the "
            'navigator requires',
        )
    own = catalog.entry(name)['data_units']
    expected = str(block['data_units'])
    if own == expected:
        return
    raise config_error(
        source,
        'data_units',
        f'the entry renders {own} frames, but its navigator block {block_name} reads '
        f'frames as {expected}; the units must match',
    )


def navigator_matched_psf(
    config: Config,
    instrument: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return the navigator-matched whole-scene PSF block for an instrument.

    The self-consistency floor sets the image-side PSF equal to the navigator's
    own model: a pure Gaussian at the emulated instrument's configured
    ``star_psf_sigma``, with no Moffat wing and no field variation.  A scene
    authors this as ``optics.psf: {match_navigator: true}``; that authored form
    is preserved through save / load and in the editor, and the renderer calls
    this helper to resolve it into concrete kernel parameters when it builds
    the kernel.

    Parameters:
        config: The active configuration.
        instrument: The sim instrument name, a generic alias, or ``None``.
        overrides: Optional scene-level ``instrument_config`` overrides.

    Returns:
        A concrete PSF parameter mapping (``sigma_v``, ``sigma_u``, ``w``,
        ``r0``, ``n``) equal to the navigator's Gaussian.
    """
    inst_config = resolve_sim_inst_config(config, instrument, overrides)
    sigma = float(inst_config['star_psf_sigma'])
    return {'sigma_v': sigma, 'sigma_u': sigma, 'w': 0.0, 'r0': 2.0, 'n': 3.0}


def resolve_extfov_margin(
    inst_config: Mapping[str, Any],
    fallback_config: Mapping[str, Any],
    size_v: int,
) -> Any:
    """Resolve the extended-FOV margin for a sim image of a given size.

    A per-instrument margin table may be size-keyed and need not cover the sim
    image size; when it does not, the generic ``sim`` block's margin is used.

    Parameters:
        inst_config: The resolved per-instrument config block.
        fallback_config: The generic ``sim`` block to fall back to.
        size_v: The sim image height in pixels (the margin-table key).

    Returns:
        The ``(v, u)`` margin entry.

    Raises:
        ValueError: If the generic fallback table is itself size-keyed and
            carries no entry for ``size_v``.
    """
    entry = inst_config.get('extfov_margin_vu')
    if isinstance(entry, Mapping):
        if size_v in entry:
            return entry[size_v]
    elif entry is not None:
        return entry
    fallback_entry = fallback_config['extfov_margin_vu']
    if isinstance(fallback_entry, Mapping):
        if size_v not in fallback_entry:
            raise ValueError(
                f'no extfov_margin_vu entry for image size {size_v}: neither the '
                f'instrument table nor the generic sim fallback covers it '
                f'(fallback sizes: {sorted(fallback_entry)})'
            )
        return fallback_entry[size_v]
    return fallback_entry


@dataclass(frozen=True)
class NavigatorFrameReading:
    """How the navigator's image classifier reads a frame's marker and peak.

    Parameters:
        units: ``'DN'`` for a ``raw_dn`` block, ``'I/F'`` for a ``calibrated_if`` one.
        marker: The missing-data marker (see
            :func:`~spindoctor.support.image_class.missing_data_marker`).
        missing_frac: Fraction of pixels carrying the marker (NaN pixels when the
            marker is NaN).
        image_class: The classifier's outcome: ``'blank'``, ``'fully_overexposed'``,
            ``'mostly_missing_data'``, or ``'clean'``.
        blank_max: The peak below which the frame is blank.
        max_missing_frac: The marker share above which the frame is refused.
        max_overexposed_frac: The saturated share above which the frame is
            overexposed.
    """

    units: str
    marker: float
    missing_frac: float
    image_class: str
    blank_max: float
    max_missing_frac: float
    max_overexposed_frac: float


def read_frame_as_navigator(
    image: NDArrayFloatType, inst_config: Mapping[str, Any]
) -> NavigatorFrameReading:
    """Classify a frame's sensor pixels the way the navigator's image classifier does.

    The thresholds and the decision are the navigator's own
    (:func:`~spindoctor.support.image_class.image_quality_thresholds` and
    :func:`~spindoctor.support.image_class.decide_image_class`).  Like the
    navigator, this counts the pixels carrying the marker and fills them with 0
    before it measures the frame.  ``image`` is the sensor frame without
    extended-FOV padding, which the navigator masks out.

    Parameters:
        image: The frame the navigator reads, with the planted offset applied.
        inst_config: The resolved per-instrument config block of that frame.

    Returns:
        The reading, with the class and the limits that decided it.
    """
    arr = np.asarray(image, dtype=np.float64)
    thresholds = image_quality_thresholds(inst_config)
    marker = thresholds.missing_data_marker_dn
    missing = missing_data_mask(arr, marker)
    missing_frac = float(missing.mean()) if arr.size else 0.0
    filled = np.where(missing, 0.0, arr)
    statistics = measure_image_quality(filled, thresholds, missing_frac=missing_frac)
    return NavigatorFrameReading(
        units=frame_units_name(inst_config),
        marker=marker,
        missing_frac=missing_frac,
        image_class=decide_image_class(statistics, thresholds),
        blank_max=thresholds.blank_max_dn,
        max_missing_frac=thresholds.max_missing_frac_clean,
        max_overexposed_frac=thresholds.max_saturation_frac_clean,
    )


def frame_units_name(inst_config: Mapping[str, Any]) -> str:
    """Return the units in which the navigator reads this block's frames.

    Parameters:
        inst_config: A resolved per-instrument config block.

    Returns:
        ``'DN'`` for a ``raw_dn`` block, ``'I/F'`` for a ``calibrated_if`` one.
    """
    return 'DN' if str(inst_config['data_units']) == 'raw_dn' else 'I/F'


def has_planted_offset(sim_params: Mapping[str, Any]) -> bool:
    """Whether a scene plants a pointing offset or roll the navigated frame applies.

    Parameters:
        sim_params: The scene mapping.

    Returns:
        True when any of ``offset_v``, ``offset_u``, or ``offset_rotation_deg``
        is non-zero, so the navigated frame differs from an offset-free preview.
    """
    return any(float(sim_params.get(key) or 0.0) != 0.0 for key in _PLANTED_OFFSET_KEYS)


def frame_units_summary(image: NDArrayFloatType, inst_config: Mapping[str, Any]) -> str:
    """Describe a frame's units, value range, marker share, and the navigator's verdict.

    The text gives the frame's units and its finite value range, for example
    ``'I/F 0 to 1.48'``.  The units are ``DN`` for a ``raw_dn`` block and ``I/F`` for
    a ``calibrated_if`` one.  When any pixel equals the navigator's marker (see
    :func:`~spindoctor.support.image_class.missing_data_marker`; NaN pixels when the
    marker is NaN), it adds the marker and the share of pixels carrying it.  When
    the navigator's classifier
    (see :func:`read_frame_as_navigator`) would not call the frame clean, it adds
    the reason, in the classifier's order: below the blank limit, over the
    overexposed share, or over ``max_missing_frac_clean``, in which case the
    navigator refuses it as missing data.  Pass the frame the navigator reads, with
    the planted offset applied, for the verdict to match it.

    Parameters:
        image: The rendered frame.
        inst_config: The resolved per-instrument config block the frame was
            rendered with.

    Returns:
        One line of text for a status display.
    """
    arr = np.asarray(image, dtype=np.float64)
    reading = read_frame_as_navigator(arr, inst_config)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        text = f'{reading.units}: no finite pixels'
    else:
        text = f'{reading.units} {float(finite.min()):.4g} to {float(finite.max()):.4g}'
    if reading.missing_frac > 0.0:
        marker_text = 'NaN' if math.isnan(reading.marker) else f'{reading.marker:g}'
        text += f'; marker {marker_text} on {reading.missing_frac * 100:.1f}% of pixels'
    if reading.image_class == 'blank':
        text += f' (peak below {reading.blank_max:g}: the navigator classifies it blank)'
    elif reading.image_class == 'fully_overexposed':
        text += (
            f' (over {reading.max_overexposed_frac * 100:.0f}% saturated: '
            'the navigator classifies it overexposed)'
        )
    elif reading.image_class == 'mostly_missing_data':
        text += (
            f' (over {reading.max_missing_frac * 100:.0f}%: '
            'the navigator refuses it as missing data)'
        )
    return text
