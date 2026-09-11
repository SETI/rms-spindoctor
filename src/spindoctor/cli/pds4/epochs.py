"""When a bundle label says an exposure was taken, and what it reads that from.

A navigation document records an exposure's epochs under ``navigation_result.times``:
``start_et``, ``stop_et`` and ``midtime_et``, in TDB seconds past J2000, beside the
spacecraft clock readings taken at them.  Every time a bundle label states is one of
these, written through :func:`~spindoctor.support.time.et_to_pds4_utc`: a data label
states its own exposure's start and stop.

A label cannot state a time the document does not record, and must not state an empty
one, so the labels pass holds every document it labels to recording all three as
finite numbers, the stop no earlier than the start, and fails an image whose document
does not before it writes anything for it.  A success document always records them,
since the navigation stamps them with the pointing it solved, so one that does not is a
broken input rather than an image with an unknown time.  The check lives here so that
anything else reading the same documents holds them to the same rule.
"""

from typing import Any

from spindoctor.cli.pds4.statistic_checks import described_value
from spindoctor.support.nav_record import finite_float

__all__ = ['EPOCH_KEYS', 'unrecorded_epoch']

EPOCH_KEYS = ('start_et', 'stop_et', 'midtime_et')
"""The epochs of an exposure a navigation document records, in the order they are checked."""


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
    return None
