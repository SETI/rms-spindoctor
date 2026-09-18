"""Reading a Voyager spacecraft clock string back.

The tick is the one the Voyager clock kernel records, a line of 0.06 seconds,
and the fields are how the conversion spells a reading: the FDS frame count, the
minor frame within it and the line within that, the line counting from one.
Written out here rather than taken from the builders, for the reason the package
gives.
"""

from .reader import SclkReader

_TICK_S = 0.06
"""How long one tick of the Voyager clock, one line, lasts."""

_MINORS_PER_FRAME = 60
"""Minor frames in one Voyager FDS frame."""

_LINES_PER_MINOR = 800
"""Lines in one Voyager minor frame; the line field counts from one."""


def _seconds(reading: str) -> float:
    """Read a Voyager clock string back as a number of seconds on that clock.

    Parameters:
        reading: The clock string, partition and all.

    Returns:
        The reading in seconds, on that clock's own origin.
    """
    frame, minor, line = reading.split('/', 1)[1].split(':')
    lines = (int(frame) * _MINORS_PER_FRAME + int(minor)) * _LINES_PER_MINOR
    return (lines + int(line) - 1) * _TICK_S


VOYAGER_READER = SclkReader(instrument='vgiss', tick_s=_TICK_S, seconds=_seconds)
"""The Voyager ISS clock, as a Voyager document records its readings."""
