"""Mapping from a sim instrument name to its per-instrument config block.

The simulator can render a frame as though it came from a specific camera, so
that a 'sim COISS NAC raw' frame goes through the same per-instrument noise,
saturation, marker, and unit settings the navigator applies to a real frame.
This module resolves a sim ``instrument`` name to the matching
``config_4N0_inst_*.yaml`` block; physical parameters then delegate to that
block, while sim-only knobs (signal full-scale, cosmic-ray rate, dropout rate)
stay in the ``sim`` config block.

A value of ``None`` (or the generic aliases) selects the standalone ``sim``
block, preserving the instrument-agnostic defaults.

A scene may also carry ``instrument_config`` overrides that are deep-merged over
the resolved block, so a scene can:

- **inherit** every physical parameter from a named instrument (no overrides),
- **inherit and override** individual parameters (override only those keys), or
- **fully self-specify** by naming the generic block and overriding everything.

Overriding a key pins it to the scene, so a later change to the real camera's
config cannot silently shift a sim test's behavior for that key; the non-overridden
keys continue to track the instrument.

The resolved block also fixes a rendered frame's unit class.  A ``raw_dn``
instrument renders detector counts with a ``0`` missing-data marker.  A
``calibrated_if`` instrument renders I/F, and its marker is whatever its block
declares: ``NaN`` for the calibrated Cassini cameras, ``0`` for Voyager ISS,
whose archived products fill lost pixels with zero.
:func:`navigator_marker_value` reads the marker the way the navigator does,
:func:`read_frame_as_navigator` classifies a frame the way the navigator's image
classifier does, and :func:`frame_units_summary` states a rendered frame's
units, range, marker share, and that verdict for a scene author.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.config import Config
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'GENERIC_INSTRUMENT_ALIASES',
    'SIM_INSTRUMENTS',
    'NavigatorFrameReading',
    'frame_units_summary',
    'has_planted_offset',
    'navigator_marker_value',
    'navigator_matched_psf',
    'read_frame_as_navigator',
    'resolve_extfov_margin',
    'resolve_sim_inst_config',
]

# Sim instrument name -> (config section, detector key or None for flat blocks).
SIM_INSTRUMENTS: dict[str, tuple[str, str | None]] = {
    'coiss_nac': ('cassini_iss', 'nac'),
    'coiss_wac': ('cassini_iss', 'wac'),
    'coiss_calib_nac': ('cassini_iss_calib', 'nac'),
    'coiss_calib_wac': ('cassini_iss_calib', 'wac'),
    'gossi': ('galileo_ssi', None),
    'nhlorri': ('newhorizons_lorri', None),
    'vgiss': ('voyager_iss', None),
}

# Names that resolve to the standalone, instrument-agnostic ``sim`` block.
GENERIC_INSTRUMENT_ALIASES = frozenset({'generic', 'sim'})

# The navigator's defaults for the image_quality_thresholds keys its classifier
# reads (nav_orchestrator.instrument_config and ImageQualityThresholds).  The
# blank and saturation limits are required keys there; every shipped block sets
# them, and the classifier's own defaults stand in for a block that omits them.
_DEFAULT_MAX_MISSING_FRAC = 0.30
_DEFAULT_MAX_OVEREXPOSED_FRAC = 0.80
_DEFAULT_BLANK_MAX = 5.0
_DEFAULT_SATURATION_THRESHOLD_DN = 4095.0
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


def resolve_sim_inst_config(
    config: Config,
    instrument: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Resolve a sim instrument name to its per-instrument config block.

    Parameters:
        config: The active configuration.
        instrument: A sim instrument name (see ``SIM_INSTRUMENTS``), one of the
            generic aliases, or ``None`` for the instrument-agnostic block.
        overrides: Optional scene-level overrides deep-merged over the resolved
            block. Overridden keys are pinned to the scene; the rest continue to
            track the instrument. Combined with the generic block this expresses a
            fully self-specified scene config.

    Returns:
        The resolved per-instrument config mapping (a fresh dict when ``overrides``
        are supplied, otherwise the live config block).

    Raises:
        ValueError: If ``instrument`` is unrecognized, or the referenced config
            section / detector is missing.
    """
    base = _resolve_base_block(config, instrument)
    if not overrides:
        return base
    return _deep_merge(base, overrides)


def _resolve_base_block(config: Config, instrument: str | None) -> Mapping[str, Any]:
    """Resolve the unmodified per-instrument (or generic) config block."""
    if instrument is None or instrument in GENERIC_INSTRUMENT_ALIASES:
        return config.category('sim')
    if instrument not in SIM_INSTRUMENTS:
        raise ValueError(
            f'unknown sim instrument {instrument!r}; '
            f'expected one of {sorted(SIM_INSTRUMENTS)} or a generic alias'
        )
    section, detector = SIM_INSTRUMENTS[instrument]
    category = config.category(section)
    if not category:
        raise ValueError(
            f'sim instrument {instrument!r} references config section {section!r}, '
            'which is missing or empty'
        )
    if detector is None:
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
    return detector_block


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


def navigator_marker_value(inst_config: Mapping[str, Any]) -> float:
    """Return the missing-data marker the navigator applies to a frame of this block.

    Reads ``noise.marker_value`` the way the navigator's instrument settings do: a
    number is taken as is, the string ``'NaN'`` (any case) or ``None`` is NaN, and an
    absent key defaults to ``0`` for a ``raw_dn`` block and to NaN for a
    ``calibrated_if`` block.  The telemetry stage writes this value on a lost pixel,
    so a simulated frame marks lost data exactly as the navigator expects.

    Parameters:
        inst_config: A resolved per-instrument config block.

    Returns:
        The marker value, possibly NaN.

    Raises:
        ValueError: If ``noise.marker_value`` is a string other than ``'NaN'``.
    """
    noise = inst_config.get('noise') or {}
    raw_dn = str(inst_config.get('data_units', 'raw_dn')) == 'raw_dn'
    value = noise.get('marker_value', 0 if raw_dn else None)
    if value is None:
        return math.nan
    if isinstance(value, str):
        if value.strip().lower() == 'nan':
            return math.nan
        raise ValueError(f'noise.marker_value string {value!r} must be "NaN" or numeric')
    return float(value)


@dataclass(frozen=True)
class NavigatorFrameReading:
    """How the navigator's image classifier reads a frame's marker and peak.

    Parameters:
        units: ``'DN'`` for a ``raw_dn`` block, ``'I/F'`` for a ``calibrated_if`` one.
        marker: The missing-data marker (see :func:`navigator_marker_value`).
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

    The navigator counts the pixels carrying the marker, fills them with 0, and
    then decides in a fixed order: a frame whose peak is below the blank limit
    (``image_quality_thresholds.blank_max_dn`` for a ``raw_dn`` block,
    ``blank_max_if`` for a ``calibrated_if`` one) is ``blank``; else a frame
    with more than ``max_overexposed_frac_clean`` (0.80) of its pixels at or
    above ``saturation_threshold_dn`` is ``fully_overexposed`` (never for a
    calibrated block, which has no saturation gate); else a frame with more than
    ``max_missing_frac_clean`` (0.30) of its pixels at the marker is
    ``mostly_missing_data``; else it is ``clean``.  ``image`` is the sensor
    frame without extended-FOV padding, which the navigator masks out.

    Parameters:
        image: The frame the navigator reads, with the planted offset applied.
        inst_config: The resolved per-instrument config block of that frame.

    Returns:
        The reading, with the class and the limits that decided it.
    """
    arr = np.asarray(image, dtype=np.float64)
    raw_dn = str(inst_config.get('data_units', 'raw_dn')) == 'raw_dn'
    thresholds = inst_config.get('image_quality_thresholds') or {}
    marker = navigator_marker_value(inst_config)
    missing = np.isnan(arr) if math.isnan(marker) else arr == marker
    missing_frac = float(missing.mean()) if arr.size else 0.0
    filled = np.where(missing, 0.0, arr)
    finite = filled[np.isfinite(filled)]
    peak = float(finite.max()) if finite.size else 0.0
    blank_key = 'blank_max_dn' if raw_dn else 'blank_max_if'
    blank_max = float(thresholds.get(blank_key, _DEFAULT_BLANK_MAX))
    max_missing = float(thresholds.get('max_missing_frac_clean', _DEFAULT_MAX_MISSING_FRAC))
    max_overexposed = float(
        thresholds.get('max_overexposed_frac_clean', _DEFAULT_MAX_OVEREXPOSED_FRAC)
    )
    saturated_frac = 0.0
    if raw_dn and arr.size:
        limit = float(thresholds.get('saturation_threshold_dn', _DEFAULT_SATURATION_THRESHOLD_DN))
        saturated_frac = float((filled >= limit).mean())
    if peak < blank_max:
        image_class = 'blank'
    elif saturated_frac > max_overexposed:
        image_class = 'fully_overexposed'
    elif missing_frac > max_missing:
        image_class = 'mostly_missing_data'
    else:
        image_class = 'clean'
    return NavigatorFrameReading(
        units='DN' if raw_dn else 'I/F',
        marker=marker,
        missing_frac=missing_frac,
        image_class=image_class,
        blank_max=blank_max,
        max_missing_frac=max_missing,
        max_overexposed_frac=max_overexposed,
    )


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

    The text names the units (``DN`` for a ``raw_dn`` block, ``I/F`` for a
    ``calibrated_if`` block) and the finite value range, for example
    ``'I/F 0 to 1.48'``.  When any pixel equals the navigator's marker (see
    :func:`navigator_marker_value`; NaN pixels when the marker is NaN), it adds the
    marker and the share of pixels carrying it.  When the navigator's classifier
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
