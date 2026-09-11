"""What reading one host's spacecraft clock back takes."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SclkReader:
    """How one host's clock strings are read back, and how long its tick lasts.

    Attributes:
        instrument: The instrument name a document from the host records.
        tick_s: How long one tick of the host's clock lasts, as its kernel records.
        seconds: Reads a clock string, partition and all, as a number of seconds
            on that clock's own origin.  Only differences between two readings of
            one clock mean anything.
    """

    instrument: str
    tick_s: float
    seconds: Callable[[str], float]
