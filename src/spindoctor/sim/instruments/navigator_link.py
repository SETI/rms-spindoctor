"""What the simulator reads from the navigator's configuration.

The image side of a render takes a camera's physics from the simulator's own instrument
entries (:mod:`spindoctor.sim.instruments.resolvers`).  It reads the navigator's
configuration only where the simulated frame has to match it, and every such read is
here:

- :func:`navigator_matched_psf`: the navigator's star PSF, for a scene that asks for the
  same PSF on both sides.
- :func:`navigator_marker`: the value the navigator reads as missing data, which is
  therefore the value a lost pixel is written with.
- :func:`resolve_extfov_margin`: the extended-FOV margin the simulated observation pads
  its frame with.
- :func:`resolve_sim_inst_config`: the navigator block itself.  The simulated observation
  hands it to the navigator, which reads it as it reads a real camera's.

An instrument entry points at its navigator block by section and detector (its
``navigator`` key; see :mod:`spindoctor.sim.instruments.catalog`).  A value of ``None``
(or the generic entry and its aliases) selects the ``sim`` section, the block of the
instrument-agnostic simulated camera.

A scene may also carry ``instrument_config`` overrides that are deep-merged over
the resolved block, so a scene can:

- **inherit** every value from a chosen instrument (no overrides),
- **inherit and override** individual values (override only those keys), or
- **fully self-specify** by choosing the generic block and overriding every key
  except ``data_units``, which a scene can't set.

Overriding a key pins it to the scene, so a later change to the real camera's
config cannot silently shift a sim test's behavior for that key; the non-overridden
keys continue to track the instrument.  An override changes what the navigator believes
about the camera.  The frame keeps the physics of the instrument entry, and the scene's
``detector`` block changes that.

A rendered frame's units come from the instrument entry (``data_units``).
The navigator reads the frame in the units its block sets, so resolving the block
fails if the two differ.  It also fails if the block doesn't set ``data_units``,
which the navigator requires.  A ``raw_dn`` instrument renders detector counts
with a ``0`` missing-data marker.  A ``calibrated_if`` instrument (the calibrated
Cassini cameras and Voyager ISS) renders I/F, also with a ``0`` marker, because the
archived Cassini CALIB and Voyager GEOMED products fill lost pixels with zero.
"""

from collections.abc import Mapping
from typing import Any

from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.schema import Issue, SchemaError
from spindoctor.sim.instruments.catalog import InstrumentCatalog, load_instrument_catalog
from spindoctor.support.image_class import missing_data_marker

__all__ = [
    'navigator_marker',
    'navigator_matched_psf',
    'resolve_extfov_margin',
    'resolve_sim_inst_config',
    'with_integer_binning_keys',
]


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
    """Resolve a sim instrument name to its navigator block.

    Parameters:
        config: The active configuration.
        instrument: A sim instrument name (see
            :func:`~spindoctor.sim.instruments.catalog.sim_instrument_names`), the
            generic entry or one of its aliases, or ``None`` for the
            instrument-agnostic block.
        overrides: Optional scene-level overrides deep-merged over the resolved
            block. Overridden keys are pinned to the scene; the rest continue to
            track the instrument. Any key except ``data_units`` can be overridden,
            so the generic block with every other key overridden is a fully
            self-specified scene config.  A ``noise.by_binning`` key written as a
            string of digits names the binning factor it spells.

    Returns:
        The navigator block with ``overrides`` merged over it, as a fresh dict.
        Without ``overrides`` it is the live config block itself.

    Raises:
        ValueError: If ``instrument`` is unrecognized, the referenced config section
            or detector is missing, or ``overrides`` sets ``data_units``.
        ~spindoctor.schema.SchemaError: If the referenced
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
    """Resolve the unmodified navigator block of an instrument (or the generic one)."""
    catalog = load_instrument_catalog()
    navigator = catalog.entry(instrument).navigator
    section = navigator.section
    detector = navigator.detector
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
        message = (
            f"the navigator block {block_name} doesn't set data_units, which the navigator requires"
        )
        raise SchemaError([Issue(source, 'data_units', message)])
    own = catalog.entry(name).data_units
    expected = str(block['data_units'])
    if own == expected:
        return
    message = (
        f'the entry renders {own} frames, but its navigator block {block_name} reads '
        f'frames as {expected}; the units must match'
    )
    raise SchemaError([Issue(source, 'data_units', message)])


def _scene_block(sim_params: Mapping[str, Any]) -> Mapping[str, Any]:
    """The navigator block of a scene, its ``instrument_config`` merged over it."""
    return resolve_sim_inst_config(
        DEFAULT_CONFIG, sim_params.get('instrument'), sim_params.get('instrument_config')
    )


def navigator_matched_psf(sim_params: Mapping[str, Any]) -> dict[str, float]:
    """Return the navigator-matched whole-scene PSF block for a scene's instrument.

    The self-consistency floor sets the image-side PSF equal to the navigator's
    own model: a pure Gaussian at the emulated instrument's configured
    ``star_psf_sigma``, with no Moffat wing and no field variation.  A scene
    authors this as ``optics.psf: {match_navigator: true}``; that authored form
    is preserved through save / load and in the editor, and the renderer calls
    this helper to resolve it into concrete kernel parameters when it builds
    the kernel.

    Parameters:
        sim_params: The scene mapping.

    Returns:
        A concrete PSF parameter mapping (``sigma_v``, ``sigma_u``, ``w``,
        ``r0``, ``n``) equal to the navigator's Gaussian.
    """
    sigma = float(_scene_block(sim_params)['star_psf_sigma'])
    return {'sigma_v': sigma, 'sigma_u': sigma, 'w': 0.0, 'r0': 2.0, 'n': 3.0}


def navigator_marker(sim_params: Mapping[str, Any]) -> float:
    """Return the missing-data marker the navigator applies to a scene's frame.

    A lost pixel carries this value, and the calibration inverse keeps every delivered
    pixel of a calibrated frame off it.

    Parameters:
        sim_params: The scene mapping.

    Returns:
        The marker, possibly NaN (see
        :func:`~spindoctor.support.image_class.missing_data_marker`).
    """
    return missing_data_marker(_scene_block(sim_params))


def resolve_extfov_margin(
    inst_config: Mapping[str, Any],
    fallback_config: Mapping[str, Any],
    size_v: int,
) -> Any:
    """Resolve the extended-FOV margin for a sim image of a given size.

    A per-instrument margin table may be size-keyed and need not cover the sim
    image size; when it does not, the generic ``sim`` block's margin is used.

    Parameters:
        inst_config: The navigator block.
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
