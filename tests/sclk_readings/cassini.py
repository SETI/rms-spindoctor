"""Reading a Cassini spacecraft clock string back.

The tick is the one the Cassini clock kernel records, a 256th of a second, and
the fields are how the conversion spells a reading: whole seconds, a period, and
the ticks of the fraction.  Written out here rather than taken from the
builders, for the reason the package gives.
"""

from .reader import SclkReader

_TICK_S = 1.0 / 256.0
"""How long one tick of the Cassini clock lasts."""


def _seconds(reading: str) -> float:
    """Read a Cassini clock string back as a number of seconds on that clock.

    Parameters:
        reading: The clock string, partition and all.

    Returns:
        The reading in seconds, on that clock's own origin.
    """
    seconds, fraction = reading.split('/', 1)[1].split('.')
    return int(seconds) + int(fraction) * _TICK_S


CASSINI_READER = SclkReader(instrument='coiss', tick_s=_TICK_S, seconds=_seconds)
"""The Cassini ISS clock, as a Cassini document records its readings."""
