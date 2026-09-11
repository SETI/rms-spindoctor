"""When a bundle label says an exposure was taken, and what it reads that from.

A navigation document records an exposure's epochs under ``navigation_result.times``:
``start_et``, ``stop_et`` and ``midtime_et``, in TDB seconds past J2000, beside the
spacecraft clock readings taken at them.  Every time a bundle label states comes from
these, written through :func:`~spindoctor.support.time.et_to_pds4_utc`: a data label
states its own exposure's start and stop, and the data collection label the earliest
start and the latest stop of the products the collection holds.

A label cannot state a time the document does not record, and must not state an empty
one, so the labels pass holds every document it labels to recording all three as
finite numbers, the stop no earlier than the start, and fails an image whose document
does not before it writes anything for it.  A success document always records them,
since the navigation stamps them with the pointing it solved, so one that does not is a
broken input rather than an image with an unknown time.  The summary pass reads the
same documents again, out of the supplemental files the labels pass wrote, and takes
the collection's range from them under the same rule: one whose document the rule
refuses leaves the collection no range it can state.
"""

from dataclasses import dataclass
from typing import Any

from spindoctor.cli.pds4.statistic_checks import described_value
from spindoctor.support.nav_record import finite_float
from spindoctor.support.time import et_to_pds4_utc

__all__ = [
    'EPOCH_KEYS',
    'RANGE_TIME_DIGITS',
    'EpochRange',
    'EpochRangeScan',
    'NoEpochRange',
    'unrecorded_epoch',
]

EPOCH_KEYS = ('start_et', 'stop_et', 'midtime_et')
"""The exposure epochs a navigation document records, in the order they are checked."""

RANGE_TIME_DIGITS = 0
"""The decimals of a second a range of products' epochs is written to: none.

Whole seconds are what the reference bundle writes for its collection and bundle ranges.
The start is rounded down and the stop up, so the range written contains every product's
own start and stop as its data label writes them, at the nearest millisecond: the nearest
millisecond of an epoch is never before the whole second at or before the epoch, nor
after the one at or after it, since a whole second is itself a millisecond and rounding
to the nearest never carries an epoch past one.
"""

_NO_PRODUCTS = (
    'there are no products to take a range from: the data tree holds no supplemental file'
)
"""Why a scan that read nothing has no range."""


def _epochs(navigation_document: Any) -> dict[str, float] | str:
    """Read an exposure's epochs out of a navigation document, or say why they cannot be.

    Parameters:
        navigation_document: The navigation metadata document as read.

    Returns:
        The three epochs keyed by :data:`EPOCH_KEYS`, as floats, or the description
        :func:`unrecorded_epoch` gives when the document does not record them.
    """
    result = (
        navigation_document.get('navigation_result')
        if isinstance(navigation_document, dict)
        else None
    )
    if not isinstance(result, dict):
        return 'records no navigation_result block'
    times = result.get('times')
    if not isinstance(times, dict):
        return 'records no navigation_result.times block'
    epochs: dict[str, float] = {}
    for key in EPOCH_KEYS:
        if key not in times:
            return f'records no navigation_result.times.{key}'
        epoch = finite_float(times[key])
        if epoch is None:
            return f'records a navigation_result.times.{key} of {described_value(times[key])}'
        epochs[key] = epoch
    if epochs['stop_et'] < epochs['start_et']:
        return (
            f'records a navigation_result.times.stop_et of {times["stop_et"]!r}, earlier '
            f'than its start_et of {times["start_et"]!r}'
        )
    return epochs


def unrecorded_epoch(navigation_document: Any) -> str | None:
    """Find what a navigation document lacks for a label to state its exposure's times.

    Parameters:
        navigation_document: The navigation metadata document as read, which is also
            what the ``navigation`` member of a supplemental file holds.

    Returns:
        None when the document records, under ``navigation_result.times``, a
        ``start_et``, a ``stop_et`` and a ``midtime_et`` that are each a finite number
        within the range of a float, the stop no earlier than the start.  Otherwise
        the first thing it lacks, in the order the three keys are listed and the
        order of the stop after them, as a clause beginning with ``records`` that a
        message completes by putting the document before it: ``records no
        navigation_result block`` (also when what is there is not an object),
        ``records no navigation_result.times block`` (likewise), ``records no
        navigation_result.times.stop_et``, ``records a navigation_result.times.start_et
        of nan, which is not a finite number`` (a boolean, a string and a null are
        refused the same way, and an integer too large for a float is given by its
        length), or ``records a navigation_result.times.stop_et of 10.0, earlier than
        its start_et of 20.0``.
    """
    epochs = _epochs(navigation_document)
    return epochs if isinstance(epochs, str) else None


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


@dataclass(frozen=True)
class NoEpochRange:
    """Why a set of products has no range of epochs a label can state.

    Attributes:
        reason: Why, as a clause a message can end with: ``there are no products to
            take a range from: the data tree holds no supplemental file``, or one
            naming the supplemental file whose navigation document records no epochs
            a label can state, and saying that no range can be taken that contains
            every product.
    """

    reason: str


class EpochRangeScan:
    """The range of epochs over the supplemental files one scan reads, taken as it reads.

    The summary pass reads every supplemental file once, to build the global index,
    and the range is taken in that same read rather than by a second one: each file's
    navigation document is handed to :meth:`include` as the file is read.  The range
    is the least start and the greatest stop, so the order the files are read in
    cannot change it.  A single document whose epochs cannot be had leaves no range,
    since a range taken over the rest could leave that file's product outside it.
    """

    def __init__(self) -> None:
        """Start a scan that has read nothing."""
        self._start_et: float | None = None
        self._stop_et: float | None = None
        self._refusal: str | None = None

    def include(self, source: str, navigation_document: Any) -> None:
        """Take one product's exposure into the range.

        Parameters:
            source: What the document was read from, as a message names it:
                ``supplemental file <path>``.
            navigation_document: The product's navigation document as read.  One
                :func:`unrecorded_epoch` refuses leaves the scan no range, with its
                description as the reason; only the first such document is kept for
                the reason, since one is enough.
        """
        epochs = _epochs(navigation_document)
        if isinstance(epochs, str):
            if self._refusal is None:
                self._refusal = f'{source} {epochs}'
            return
        start_et = epochs['start_et']
        stop_et = epochs['stop_et']
        self._start_et = start_et if self._start_et is None else min(self._start_et, start_et)
        self._stop_et = stop_et if self._stop_et is None else max(self._stop_et, stop_et)

    def result(self) -> EpochRange | NoEpochRange:
        """Return the range over every product included, or why there is none.

        Returns:
            The range, when at least one product was included and none refused.
            Otherwise why there is none: the first document refused, as in
            ``supplemental file <path> records no navigation_result block, so no
            range can be taken that contains every product``; or, when nothing was
            read at all, that there are no products to take a range from.
        """
        if self._refusal is not None:
            return NoEpochRange(
                f'{self._refusal}, so no range can be taken that contains every product'
            )
        if self._start_et is None or self._stop_et is None:
            return NoEpochRange(_NO_PRODUCTS)
        return EpochRange(start_et=self._start_et, stop_et=self._stop_et)
