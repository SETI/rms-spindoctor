"""The Voyager 1 ISS host: its camera frames, its exposure, and its clock.

What stamping an attitude solution onto a Voyager result needs to know, as the
:class:`~tests.mini_nav_results.shared.Host` :data:`VOYAGER_ISS`, and the reading
a Voyager image's shutter opened at, which its clock readings are counted from.
"""

from __future__ import annotations

import numpy as np

from spindoctor.support.types import NDArrayFloatType

from .shared import Host, elapsed_ticks

VGISS_KERNELS = (
    'naif0012.tls',
    'vg100019.tsc',
    'vg1_saturn.bsp',
    'vg1_super.bc',
)
"""Kernels loaded for the Voyager images."""


_VOYAGER_OOPS_FROM_SPICE: NDArrayFloatType = np.eye(3)
"""The constant rotation between the oops and SPICE Voyager ISS camera frames."""


_VOYAGER_CAMERA_FRAME_IDS = {'NAC': -31101, 'WAC': -31102}
"""SPICE frame id of each Voyager 1 ISS camera frame."""


_VOYAGER_EXPOSURE_S = 1.44
"""Exposure the Voyager images were taken with."""


_VOYAGER_SCLK_LINES_PER_MINOR = 800
"""Lines in one minor frame, the modulus of the Voyager clock's line field."""


_VOYAGER_SCLK_MINORS_PER_FRAME = 60
"""Minor frames in one FDS frame, the modulus of the clock's minor-frame field."""


_VOYAGER_SCLK_FIRST_LINE = 1
"""The line field's offset: it counts from one rather than from zero."""


_VOYAGER_SCLK_TICK_S = 0.06
"""Seconds in one line, which is the tick the Voyager clock counts in.

A minor frame is 800 lines and an FDS frame is 60 minor frames, which puts an
FDS frame at 2880 seconds, the rate the clock kernel records for it.
"""


def _voyager_sclk_reading(ticks: int) -> str:
    """Spell a Voyager clock tick count the way the conversion spells it.

    The three fields -- the FDS frame count, the minor frame within it and the
    line within that -- are written behind the clock partition and separated by
    colons, zero padded to five, two and three digits.  The line field counts
    from one, and a count that fills a field carries into the field above it
    rather than widening it.

    Parameters:
        ticks: The reading, as a count of line ticks.

    Returns:
        The clock string.
    """
    lines_per_frame = _VOYAGER_SCLK_MINORS_PER_FRAME * _VOYAGER_SCLK_LINES_PER_MINOR
    frame, within_frame = divmod(ticks, lines_per_frame)
    minor, line = divmod(within_frame, _VOYAGER_SCLK_LINES_PER_MINOR)
    return f'1/{frame:05d}:{minor:02d}:{line + _VOYAGER_SCLK_FIRST_LINE:03d}'


def voyager_sclk_open(frame: int, minor: int) -> int:
    """Return a Voyager image's clock reading at shutter open, as a tick count.

    A Voyager image is named for the frame and minor-frame fields of the
    reading its shutter closed at: a label carrying ``IMAGE_NUMBER = "13854.55"``
    carries ``SPACECRAFT_CLOCK_STOP_COUNT = "13854:55:001"`` beside it.  So the
    reading at shutter open is one exposure of ticks before the first line of
    the minor frame the image is named for.

    Parameters:
        frame: The FDS frame count the image is named for.
        minor: The minor frame within it the image is named for.

    Returns:
        The reading, as a count of ticks.
    """
    close = (frame * _VOYAGER_SCLK_MINORS_PER_FRAME + minor) * _VOYAGER_SCLK_LINES_PER_MINOR
    return close - elapsed_ticks(_VOYAGER_EXPOSURE_S, _VOYAGER_SCLK_TICK_S)


VOYAGER_ISS = Host(
    camera_frames={
        camera: (f'VG1_ISS{camera[0]}A', frame_id)
        for camera, frame_id in _VOYAGER_CAMERA_FRAME_IDS.items()
    },
    ck_frame_id=-31100,
    oops_from_spice=_VOYAGER_OOPS_FROM_SPICE,
    exposure_s=_VOYAGER_EXPOSURE_S,
    tick_s=_VOYAGER_SCLK_TICK_S,
    spell=_voyager_sclk_reading,
)
"""The Voyager 1 ISS host: its cameras' frames, its exposure, and its line clock."""
