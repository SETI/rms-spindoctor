"""Reading a spacecraft clock string back, and holding a triple to its own epochs.

A document records three clock readings and the three epochs they were taken
at.  The readings are counted from the epochs, so the interval between two of
them is the interval between those epochs, to within the tick that clock counts
in; a triple that says the shutter was open for a fifth of the time the epochs
say is a hand-authored one, and every reader that subtracts two of its readings
measures that fifth.

Two document sets are held to that -- the statistics fixture tree and the PDS4
bundle cohort -- so what does the holding lives here rather than in either
suite.  What is written out here is each host's tick and how its fields are
spelled, deliberately: a test that asks the builders what their own clock counts
in agrees with every answer they give.  These are the ticks the mission clock
kernels record, a 256th of a second on Cassini and a Voyager line of 0.06
seconds.

Neither this nor anything reading it can report a conversion that is wrong the
same way at every epoch, since both sides of every comparison here come out of
the same conversion.  What reports that is a test that furnishes the real
kernel, which is in the integration tier because the kernel is not always there
to furnish.
"""

from collections.abc import Mapping
from typing import Any

SCLK_TICK_S = {'coiss': 1.0 / 256.0, 'vgiss': 0.06}
"""How long one tick of each host's spacecraft clock is."""

_VOYAGER_MINORS_PER_FRAME = 60
"""Minor frames in one Voyager FDS frame."""

_VOYAGER_LINES_PER_MINOR = 800
"""Lines in one Voyager minor frame; the line field counts from one."""


def sclk_seconds(instrument: str, reading: str) -> float:
    """Read a spacecraft clock string back as a number of seconds on that clock.

    Parameters:
        instrument: Which host recorded the reading.
        reading: The clock string, partition and all.

    Returns:
        The reading in seconds, on that clock's own origin.  Only differences
        between two readings of one clock mean anything.
    """
    count = reading.split('/', 1)[1]
    if instrument == 'coiss':
        seconds, fraction = count.split('.')
        return int(seconds) + int(fraction) * SCLK_TICK_S['coiss']
    frame, minor, line = count.split(':')
    lines = (int(frame) * _VOYAGER_MINORS_PER_FRAME + int(minor)) * _VOYAGER_LINES_PER_MINOR
    return (lines + int(line) - 1) * SCLK_TICK_S['vgiss']


def triples_disagreeing_with_their_epochs(documents: Mapping[str, dict[str, Any]]) -> list[str]:
    """Return one line for every clock reading its own epochs do not put it at.

    A document recording no times at all is passed over: an image whose load
    failed has no epoch to have read a clock at, and a simulated scene has no
    clock.

    Parameters:
        documents: The documents to read, keyed by whatever names them.

    Returns:
        One line per disagreeing reading, naming the document, and empty when
        every triple spans what its epochs span.
    """
    disagreeing: list[str] = []
    for name, document in documents.items():
        times = document.get('navigation_result', {}).get('times')
        if times is None:
            continue
        instrument = str(document['observation']['instrument'])
        tick_s = SCLK_TICK_S[instrument]
        opened = sclk_seconds(instrument, str(times['sclk_start']))
        for reading, epoch in (('sclk_midtime', 'midtime_et'), ('sclk_stop', 'stop_et')):
            on_the_clock = sclk_seconds(instrument, str(times[reading])) - opened
            between_the_epochs = float(times[epoch]) - float(times['start_et'])
            if abs(on_the_clock - between_the_epochs) > tick_s:
                disagreeing.append(
                    f'{name}: {reading} is {on_the_clock} s after sclk_start, against '
                    f'{between_the_epochs} s between the epochs'
                )
    return disagreeing
