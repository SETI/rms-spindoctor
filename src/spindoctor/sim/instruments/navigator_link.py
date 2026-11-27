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
- :func:`resolve_sim_inst_config` and :func:`scene_camera_block`: the navigator block
  itself, a :class:`~spindoctor.config.sections_camera.CameraBlock`.  The simulated
  observation hands it to the navigator, which reads it as it reads a real camera's.

An instrument entry points at its navigator block by section and detector (its
``navigator`` key; see :mod:`spindoctor.sim.instruments.catalog`).  A value of ``None``
(or the generic entry and its aliases) selects the ``sim`` section, the block of the
instrument-agnostic simulated camera.

A scene may also carry an ``instrument_config`` block that is merged over the resolved
block (see :mod:`spindoctor.sim.scene.idealized_camera`), so a scene can:

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
fails if the two differ.  A ``raw_dn`` instrument renders detector counts
with a ``0`` missing-data marker.  A ``calibrated_if`` instrument (the calibrated
Cassini cameras and Voyager ISS) renders I/F, also with a ``0`` marker, because the
archived Cassini CALIB and Voyager GEOMED products fill lost pixels with zero.
"""

from collections.abc import Mapping
from typing import Any

from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.config.sections_camera import CameraBlock, ExtfovMargin
from spindoctor.schema import Issue, SchemaError, parse
from spindoctor.sim.instruments.catalog import InstrumentCatalog, load_instrument_catalog
from spindoctor.sim.scene.idealized_camera import InstrumentConfig, overlay_camera_block
from spindoctor.support.image_class import missing_data_marker

__all__ = [
    'navigator_marker',
    'navigator_matched_psf',
    'resolve_extfov_margin',
    'resolve_sim_inst_config',
    'scene_camera_block',
]


def resolve_sim_inst_config(
    config: Config,
    instrument: str | None,
    overlay: InstrumentConfig | None = None,
) -> CameraBlock:
    """Resolve a sim instrument name to its navigator block.

    Parameters:
        config: The active configuration.
        instrument: A sim instrument name (see
            :func:`~spindoctor.sim.instruments.catalog.sim_instrument_names`), the
            generic entry or one of its aliases, or ``None`` for the
            instrument-agnostic block.
        overlay: A scene's ``instrument_config``, merged over the resolved block (see
            :func:`~spindoctor.sim.scene.idealized_camera.overlay_camera_block`).
            Overridden keys are pinned to the scene; the rest continue to track the
            instrument.

    Returns:
        The navigator block with ``overlay`` merged over it.  Without ``overlay`` it is
        the configuration's own block.

    Raises:
        ValueError: If ``instrument`` is unrecognized, or the referenced config section
            or detector is missing.
        ~spindoctor.schema.SchemaError: If the entry's ``data_units`` differ from the
            block's, or the merged block breaks the camera block's schema.  For a
            mismatch the message gives the entry's file and name, the block, and both
            units.
    """
    base, block_name = _resolve_base_block(config, instrument)
    if overlay is None:
        return base
    return overlay_camera_block(
        base, overlay, source=f'instrument_config merged over the navigator block {block_name}'
    )


def scene_camera_block(config: Config, sim_params: Mapping[str, Any]) -> CameraBlock:
    """The navigator block of a scene, its ``instrument_config`` merged over it.

    The scene's ``instrument_config`` mapping is parsed here, with the type the scene
    validator gives it.

    Parameters:
        config: The active configuration.
        sim_params: The scene mapping.

    Returns:
        The scene's navigator block.

    Raises:
        ~spindoctor.schema.SchemaError: If ``instrument_config`` breaks its schema, or the
            merged block breaks the camera block's.
    """
    raw = sim_params.get('instrument_config')
    overlay = None if raw is None else parse(InstrumentConfig, raw, source='instrument_config')
    return resolve_sim_inst_config(config, sim_params.get('instrument'), overlay)


def _resolve_base_block(config: Config, instrument: str | None) -> tuple[CameraBlock, str]:
    """Resolve the unmodified navigator block of an instrument (or the generic one).

    Returns:
        The block, and its name as ``section`` or ``section.detector``.
    """
    catalog = load_instrument_catalog()
    navigator = catalog.entry(instrument).navigator
    section = navigator.section
    detector = navigator.detector
    try:
        cameras = config.camera_names(section)
    except KeyError:
        raise ValueError(
            f'sim instrument {instrument!r} references config section {section!r}, '
            'which is not a camera section of the configuration'
        ) from None
    if detector not in cameras:
        raise ValueError(
            f'sim instrument {instrument!r} references detector {detector!r} in '
            f'section {section!r}, which has no such entry'
        )
    block_name = section if detector is None else f'{section}.{detector}'
    block = config.camera_block(section, detector)
    _check_units_agree(catalog, instrument, block, block_name=block_name)
    return block, block_name


def _check_units_agree(
    catalog: InstrumentCatalog,
    instrument: str | None,
    block: CameraBlock,
    *,
    block_name: str,
) -> None:
    """Refuse an entry whose frame units differ from the units its navigator block sets.

    If the entry rendered I/F against a ``raw_dn`` block, the navigator would read every
    frame as DN, and the other way round.
    """
    name = catalog.canonical_name(instrument)
    own = catalog.entry(name).data_units
    if own == block.data_units:
        return
    message = (
        f'the entry renders {own} frames, but its navigator block {block_name} reads '
        f'frames as {block.data_units}; the units must match'
    )
    raise SchemaError([Issue(catalog.sources[name], 'data_units', message)])


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
    sigma = scene_camera_block(DEFAULT_CONFIG, sim_params).star_psf_sigma
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
    return missing_data_marker(scene_camera_block(DEFAULT_CONFIG, sim_params))


def resolve_extfov_margin(
    margin: ExtfovMargin,
    fallback: ExtfovMargin,
    size_v: int,
) -> tuple[int, int]:
    """Resolve the extended-FOV margin for a sim image of a given size.

    A per-instrument margin table may be size-keyed and need not cover the sim
    image size; when it does not, the generic ``sim`` block's margin is used.

    Parameters:
        margin: The navigator block's ``extfov_margin_vu``.
        fallback: The generic ``sim`` block's ``extfov_margin_vu``, to fall back to.
        size_v: The sim image height in pixels (the margin-table key).

    Returns:
        The ``(v, u)`` margin entry.

    Raises:
        ValueError: If neither table covers ``size_v``.
    """
    if isinstance(margin, tuple):
        return margin
    if size_v in margin:
        return margin[size_v]
    if isinstance(fallback, tuple):
        return fallback
    if size_v not in fallback:
        raise ValueError(
            f'no extfov_margin_vu entry for image size {size_v}: neither the '
            f'instrument table nor the generic sim fallback covers it '
            f'(fallback sizes: {sorted(fallback)})'
        )
    return fallback[size_v]
