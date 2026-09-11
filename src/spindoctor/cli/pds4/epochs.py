"""When a bundle label says an exposure was taken, and what it reads that from.

A navigation document records an exposure's epochs under ``navigation_result.times``:
``start_et``, ``stop_et`` and ``midtime_et``, in TDB seconds past J2000, beside the
spacecraft clock readings taken at them.  Every time a bundle label states comes from
these, written through :func:`~spindoctor.support.time.et_to_pds4_utc`: a data label
states its own exposure's start and stop, and the data collection label the earliest
start and the latest stop of the products the collection holds.
"""

from dataclasses import dataclass
from typing import Any

from spindoctor.support.time import et_to_pds4_utc

__all__ = ['RANGE_TIME_DIGITS', 'EpochRange', 'EpochRangeScan']

RANGE_TIME_DIGITS = 0
"""The decimals of a second a range of products' epochs is written to: none.

Whole seconds are what the reference bundle writes for its collection and bundle ranges.
The start is rounded down and the stop up, so the range written contains every product's
own start and stop as its data label writes them, at the nearest millisecond: the nearest
millisecond of an epoch is never before the whole second at or before the epoch, nor
after the one at or after it, since a whole second is itself a millisecond and rounding
to the nearest never carries an epoch past one.
"""


@dataclass(frozen=True)
class EpochRange:
    """The earliest exposure start and the latest exposure stop over a set of products.

    Attributes:
        start_et: The earliest start, in TDB seconds past J2000.
        stop_et: The latest stop, likewise.
    """

    start_et: float
    stop_et: float

    def template_variables(self) -> dict[str, str]:
        """Return the range as the two template variables a label states it with.

        Returns:
            ``EARLIEST_START_DATE_TIME``, the start rounded down to the whole second,
            and ``LATEST_STOP_DATE_TIME``, the stop rounded up to it, each in the PDS4
            UTC spelling with no decimals, as in ``2004-02-07T04:25:35Z``.  A range
            written this way contains every product's own start and stop as a data
            label writes them, at the nearest millisecond, which is never before the
            whole second at or before its epoch nor after the one at or after it.  A
            range rounded to the nearer second would not: it can begin after the
            first exposure opened and end before the last one closed.
        """
        return {
            'EARLIEST_START_DATE_TIME': et_to_pds4_utc(
                self.start_et, digits=RANGE_TIME_DIGITS, rounding='down'
            ),
            'LATEST_STOP_DATE_TIME': et_to_pds4_utc(
                self.stop_et, digits=RANGE_TIME_DIGITS, rounding='up'
            ),
        }


class EpochRangeScan:
    """The range of epochs over the supplemental files one scan reads, taken as it reads.

    The summary pass reads every supplemental file once, to build the global index,
    and the range is taken in that same read rather than by a second one: each file's
    navigation document is handed to :meth:`include` as the file is read.  The range
    is the least start and the greatest stop, so the order the files are read in
    cannot change it.
    """

    def __init__(self) -> None:
        """Start a scan that has read nothing."""
        self._start_et: float | None = None
        self._stop_et: float | None = None

    def include(self, navigation_document: dict[str, Any]) -> None:
        """Take one product's exposure into the range.

        Parameters:
            navigation_document: The product's navigation document, which records the
                exposure's ``start_et`` and ``stop_et`` under ``navigation_result.times``.
        """
        times = navigation_document['navigation_result']['times']
        start_et: float = times['start_et']
        stop_et: float = times['stop_et']
        self._start_et = start_et if self._start_et is None else min(self._start_et, start_et)
        self._stop_et = stop_et if self._stop_et is None else max(self._stop_et, stop_et)

    def result(self) -> EpochRange | None:
        """Return the range over every product included.

        Returns:
            The range, or None when no product was included.
        """
        if self._start_et is None or self._stop_et is None:
            return None
        return EpochRange(start_et=self._start_et, stop_et=self._stop_et)
