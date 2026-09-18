"""Reading a spacecraft clock string back, and holding a triple to its own epochs.

A document records three clock readings and the three epochs they were taken
at.  The readings are counted from the epochs, so the interval between two of
them is the interval between those epochs, to within the tick that clock counts
in; a triple that says the shutter was open for a fifth of the time the epochs
say is a hand-authored one, and every reader that subtracts two of its readings
measures that fifth.

Two document sets are held to that -- the statistics fixture tree and every
bundle's cohort -- so what does the holding lives here rather than in either
suite.  How each host's clock is read back, its tick and how its fields are
spelled, is written out in a module of this package named for the host, and
deliberately not taken from the builders: a test that asks the builders what
their own clock counts in agrees with every answer they give.  The ticks are the
ones the mission clock kernels record.  :data:`SCLK_READERS` lists them.

Neither this nor anything reading it can report a conversion that is wrong the
same way at every epoch, since both sides of every comparison here come out of
the same conversion.  What reports that is a test that furnishes the real
kernel, which is in the integration tier because the kernel is not always there
to furnish.
"""

from collections.abc import Mapping
from typing import Any

from .cassini import CASSINI_READER
from .reader import SclkReader
from .voyager import VOYAGER_READER

__all__ = ['SCLK_READERS', 'SclkReader', 'triples_disagreeing_with_their_epochs']

SCLK_READERS: dict[str, SclkReader] = {
    reader.instrument: reader for reader in (CASSINI_READER, VOYAGER_READER)
}
"""Every host's clock reader, keyed by the instrument its documents record."""


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

    Raises:
        KeyError: If a document recording times comes from an instrument no
            reader in :data:`SCLK_READERS` reads.
    """
    disagreeing: list[str] = []
    for name, document in documents.items():
        times = document.get('navigation_result', {}).get('times')
        if times is None:
            continue
        reader = SCLK_READERS[str(document['observation']['instrument'])]
        opened = reader.seconds(str(times['sclk_start']))
        for reading, epoch in (('sclk_midtime', 'midtime_et'), ('sclk_stop', 'stop_et')):
            on_the_clock = reader.seconds(str(times[reading])) - opened
            between_the_epochs = float(times[epoch]) - float(times['start_et'])
            if abs(on_the_clock - between_the_epochs) > reader.tick_s:
                disagreeing.append(
                    f'{name}: {reading} is {on_the_clock} s after sclk_start, against '
                    f'{between_the_epochs} s between the epochs'
                )
    return disagreeing
