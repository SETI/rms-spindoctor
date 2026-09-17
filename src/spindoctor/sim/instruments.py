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
"""

from collections.abc import Mapping
from typing import Any

from spindoctor.config import Config

__all__ = [
    'GENERIC_INSTRUMENT_ALIASES',
    'SIM_INSTRUMENTS',
    'navigator_matched_psf',
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
