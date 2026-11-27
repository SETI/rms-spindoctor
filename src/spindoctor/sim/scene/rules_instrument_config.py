"""The rules of the scene's ``instrument_config`` block that read the scene's camera.

The block's type (:class:`~spindoctor.sim.scene.idealized_camera.InstrumentConfig`)
checks each value a scene sets against the camera block's own field type.  The rules
here need more than the block: the navigator block the scene's instrument points at,
and the readout modes the instrument offers.

- Only a camera whose frames are in raw DN reads noise values for its faint-star limit.
  A calibrated-unit camera's limit is a constant, so a scene for one sets no read noise,
  no zero point, and no binned-mode table.
- A binned-mode entry is for a binning the instrument offers, since no frame reads any
  other.
- The navigator block with the scene's block merged over it parses as a camera block,
  so every rule of that type holds for it.
- On a binned frame the faint-star limit reads the readout mode's read noise and zero
  point from the binned-mode table.  So a scene that sets either unbinned value sets the
  readout mode's value as well, or the value it sets would never be read.

:func:`spindoctor.sim.scene.validate.parse_scene` calls
:func:`check_instrument_config` once per scene, after the typed scene is parsed.
"""

from __future__ import annotations

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.schema import SchemaError
from spindoctor.sim.instruments.navigator_link import resolve_sim_inst_config
from spindoctor.sim.instruments.resolvers import cataloged_binnings
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.idealized_camera import InstrumentConfig, InstrumentConfigNoise

__all__ = ['STAR_LIMIT_NOISE_KEYS', 'check_instrument_config']

STAR_LIMIT_NOISE_KEYS = ('read_noise_dn', 'star_flux_dn_per_s_vmag0')
"""The noise values the faint-star limit reads, unbinned or from a binned-mode entry."""

_NOISE_PATH = 'instrument_config.noise'
"""The key path of the noise block these checks cover, for error messages."""

_TABLE_PATH = f'{_NOISE_PATH}.by_binning'
"""The key path of the binned-mode table, for error messages."""


def _check_noise_is_read(
    noise: InstrumentConfigNoise, units: str, *, instrument: str, source: str
) -> None:
    """Refuse a star-limit noise value that no frame of the instrument reads.

    The star limit reads noise values only on a raw-DN block, and there only the
    binned-mode entry for the frame's binning.  A calibrated-unit block's limit is a
    constant that reads no noise value, so a binned-mode table and an unbinned
    ``read_noise_dn`` or ``star_flux_dn_per_s_vmag0`` are refused there, and on a raw-DN
    block an entry for a binning the instrument's readout modes don't offer is refused.

    Parameters:
        noise: The scene's ``instrument_config.noise``.
        units: The navigator block's ``data_units``, which a scene can't change.
        instrument: The scene's (already validated) instrument name.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the table or an unbinned star-limit value is set on
            a calibrated-unit block, or the table holds an entry for a binning the
            instrument doesn't offer.
    """
    if units != 'raw_dn':
        unread = f'renders {units} frames, whose star limit reads no noise value'
        if noise.by_binning is not None:
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH} is never read: instrument {instrument!r} '
                f'{unread}. Leave the table out.'
            )
        for key in STAR_LIMIT_NOISE_KEYS:
            if getattr(noise, key) is not None:
                raise SimSceneValidationError(
                    f'{source}: {_NOISE_PATH}.{key} is never read: instrument '
                    f'{instrument!r} {unread}. Leave the key out.'
                )
        return
    offered = cataloged_binnings(instrument)
    for binning in noise.by_binning or {}:
        if binning not in offered:
            listed = ', '.join(str(mode) for mode in offered)
            raise SimSceneValidationError(
                f'{source}: {_TABLE_PATH}.{binning} is never read: instrument '
                f'{instrument!r} offers binnings {listed}. Leave the entry out.'
            )


def check_instrument_config(
    overlay: InstrumentConfig | None, *, instrument: str, binning: int, source: str
) -> None:
    """Check a scene's ``instrument_config`` against its camera.

    Parameters:
        overlay: The scene's typed ``instrument_config``, or None when the scene leaves
            the block out.
        instrument: The scene's (already validated) instrument name.
        binning: The scene's (already validated) readout binning.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If the block sets a noise value no frame of the
            instrument reads, the merged navigator block breaks the camera block's
            schema, or the block sets an unbinned noise value that a frame of this
            binning reads from the binned-mode table.
    """
    if overlay is None:
        return
    noise = overlay.noise or InstrumentConfigNoise()
    units = resolve_sim_inst_config(DEFAULT_CONFIG, instrument).data_units
    _check_noise_is_read(noise, units, instrument=instrument, source=source)
    try:
        merged = resolve_sim_inst_config(DEFAULT_CONFIG, instrument, overlay)
    except SchemaError as exc:
        raise SimSceneValidationError(f'{source}: {exc}') from exc
    if binning not in (merged.noise.by_binning or {}):
        return
    mode = (noise.by_binning or {}).get(binning)
    for key in STAR_LIMIT_NOISE_KEYS:
        if getattr(noise, key) is not None and (mode is None or getattr(mode, key) is None):
            raise SimSceneValidationError(
                f'{source}: {_NOISE_PATH}.{key} is set, but a frame binned by '
                f'{binning} reads {_TABLE_PATH}.{binning}.{key} '
                f'instead. Set {_TABLE_PATH}.{binning}.{key} too.'
            )
