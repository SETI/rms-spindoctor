"""Attitude read from a type 3 C-kernel without furnishing it.

A C-kernel furnished to SPICE joins the kernel pool, and every later pointing lookup for
the object it describes answers from it in preference to what was furnished before.  A
kernel that must answer one question and leave every other lookup as it was is therefore
read here, through the DAF routines, instead: the file is opened for reading, the
segments for one object are searched, and the file is closed again.  Nothing is loaded
into the pool, so ``pxform`` and ``ckgp`` answer after a read exactly as they did before
it.

Only type 3 segments whose reference frame is J2000 are read.  A type 3 segment holds
pointing instances, each a quaternion (and, optionally, an angular velocity) tagged with
an encoded spacecraft clock time, grouped into interpolation intervals.  Between two
instances of one interval the attitude turns at a constant rate about the axis of the
rotation that takes the first to the second, which is the rule SPICE's own type 3
evaluator applies; an epoch between two intervals, or outside every segment, has no
attitude.  Segments are searched last first, as SPICE searches them, so a later segment
covering an epoch answers for it.

The segment layout read is the one the NAIF C-kernel required reading gives for type 3:
the pointing instances, then their time tags, then a directory of every hundredth time
tag, then the interpolation intervals' start times and their directory, and last the
number of intervals and the number of instances.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cspyce
import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['ck_type3_attitudes']

_CK_TYPE_3 = 3
"""The C-kernel data type whose segments are read."""

_J2000_FRAME_CODE = 1
"""The SPICE frame code of J2000, the only reference frame read."""

_SUMMARY_DOUBLES = 2
"""Double precision components of a C-kernel segment descriptor."""

_SUMMARY_INTEGERS = 6
"""Integer components of a C-kernel segment descriptor."""

_DIRECTORY_STRIDE = 100
"""Number of time tags, or interval start times, each directory entry stands for."""

_QUATERNION_DOUBLES = 4
"""Doubles of a pointing instance that carries no angular velocity."""

_ANGULAR_VELOCITY_DOUBLES = 3
"""Doubles an angular velocity adds to a pointing instance."""


@dataclass(frozen=True)
class _Segment:
    """Where one type 3 segment's arrays lie in its file.

    Parameters:
        start_ticks: Encoded spacecraft clock time the descriptor says the segment begins.
        stop_ticks: Encoded spacecraft clock time the descriptor says it ends.
        begin: DAF address of the segment's first double.
        record_doubles: Doubles in one pointing instance.
        record_count: Number of pointing instances.
        interval_count: Number of interpolation intervals.
    """

    start_ticks: float
    stop_ticks: float
    begin: int
    record_doubles: int
    record_count: int
    interval_count: int

    @property
    def times_begin(self) -> int:
        """DAF address of the first time tag."""
        return self.begin + self.record_count * self.record_doubles

    @property
    def interval_starts_begin(self) -> int:
        """DAF address of the first interpolation interval's start time."""
        directory = (self.record_count - 1) // _DIRECTORY_STRIDE
        return self.times_begin + self.record_count + directory


def ck_type3_attitudes(
    path: Path, object_id: int, ticks: Sequence[float]
) -> list[NDArrayFloatType | None]:
    """Return the J2000-to-object rotation a type 3 C-kernel gives at each time.

    The kernel is read without being furnished, so every SPICE pointing lookup answers
    afterward as it did before.  At a time equal to a pointing instance's time tag the
    rotation is that instance's; between two instances of one interpolation interval it
    turns from the first toward the second at a constant rate about the axis of the
    rotation between them.  Segments of other objects, of other data types, and in
    reference frames other than J2000 are ignored.

    Parameters:
        path: Local path of the C-kernel.
        object_id: SPICE id of the object whose attitude is read, such as a spacecraft
            bus.
        ticks: Encoded spacecraft clock times to evaluate at.

    Returns:
        One entry per time, in order: the 3x3 rotation taking a J2000 vector into the
        object's frame, or None when no segment of the kernel gives the object an
        attitude at that time.

    Raises:
        OSError: if the file cannot be opened as a DAF.
    """
    handle = int(cspyce.dafopr(str(path)))
    try:
        segments = _object_segments(handle, object_id)
        return [_attitude_at(handle, segments, float(tick)) for tick in ticks]
    finally:
        cspyce.dafcls(handle)


def _object_segments(handle: int, object_id: int) -> list[_Segment]:
    """Return the type 3, J2000-referenced segments of one object in file order.

    Parameters:
        handle: Handle of the DAF, open for reading.
        object_id: SPICE id of the object.

    Returns:
        The segments, first in the file first.
    """
    segments: list[_Segment] = []
    cspyce.dafbfs(handle)
    while cspyce.daffna():
        summary = list(cspyce.dafgs())
        doubles, integers = cspyce.dafus(summary, _SUMMARY_DOUBLES, _SUMMARY_INTEGERS)
        seg_object, reference, data_type, has_rates, begin, end = (int(x) for x in integers)
        if seg_object != object_id or data_type != _CK_TYPE_3 or reference != _J2000_FRAME_CODE:
            continue
        interval_count, record_count = (int(x) for x in cspyce.dafgda(handle, end - 1, end))
        record_doubles = _QUATERNION_DOUBLES + (_ANGULAR_VELOCITY_DOUBLES if has_rates else 0)
        segments.append(
            _Segment(
                start_ticks=float(doubles[0]),
                stop_ticks=float(doubles[1]),
                begin=begin,
                record_doubles=record_doubles,
                record_count=record_count,
                interval_count=interval_count,
            )
        )
    return segments


def _attitude_at(handle: int, segments: list[_Segment], tick: float) -> NDArrayFloatType | None:
    """Return the attitude the last segment covering a time gives, or None.

    Parameters:
        handle: Handle of the DAF, open for reading.
        segments: The object's segments in file order.
        tick: Encoded spacecraft clock time.

    Returns:
        The J2000-to-object rotation, or None when no segment gives one at ``tick``.
    """
    for segment in reversed(segments):
        if not segment.start_ticks <= tick <= segment.stop_ticks:
            continue
        attitude = _segment_attitude(handle, segment, tick)
        if attitude is not None:
            return attitude
    return None


def _double(handle: int, address: int) -> float:
    """Return the double at one DAF address."""
    return float(cspyce.dafgda(handle, address, address)[0])


def _last_at_or_before(handle: int, begin: int, count: int, tick: float) -> int:
    """Return the index of the last of ``count`` ascending doubles not after ``tick``.

    Parameters:
        handle: Handle of the DAF, open for reading.
        begin: DAF address of the first double.
        count: Number of doubles.
        tick: The value searched for; the first double must not exceed it.

    Returns:
        The zero-based index.
    """
    low, high = 0, count - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _double(handle, begin + middle) <= tick:
            low = middle
        else:
            high = middle - 1
    return low


def _quaternion_matrix(handle: int, segment: _Segment, index: int) -> NDArrayFloatType:
    """Return the rotation one pointing instance holds.

    Parameters:
        handle: Handle of the DAF, open for reading.
        segment: The segment holding the instance.
        index: Zero-based index of the instance.

    Returns:
        The J2000-to-object rotation of the instance's quaternion.
    """
    first = segment.begin + index * segment.record_doubles
    quaternion = cspyce.dafgda(handle, first, first + _QUATERNION_DOUBLES - 1)
    return np.asarray(cspyce.q2m(quaternion), dtype=np.float64)


def _segment_attitude(handle: int, segment: _Segment, tick: float) -> NDArrayFloatType | None:
    """Return the attitude one segment gives at a time inside its descriptor's bounds.

    Parameters:
        handle: Handle of the DAF, open for reading.
        segment: The segment.
        tick: Encoded spacecraft clock time.

    Returns:
        The J2000-to-object rotation, or None when ``tick`` falls before the first
        instance, after the last, or between two interpolation intervals.
    """
    times = segment.times_begin
    count = segment.record_count
    if tick < _double(handle, times) or tick > _double(handle, times + count - 1):
        return None
    index = _last_at_or_before(handle, times, count, tick)
    before_ticks = _double(handle, times + index)
    if before_ticks == tick:
        return _quaternion_matrix(handle, segment, index)
    after_ticks = _double(handle, times + index + 1)
    starts = segment.interval_starts_begin
    interval = _last_at_or_before(handle, starts, segment.interval_count, tick)
    # An interval holds every instance from its own start time up to the next
    # interval's; the instance after ``tick`` opening the next interval means
    # ``tick`` lies in the gap between the two, where the kernel has no attitude.
    if interval + 1 < segment.interval_count and after_ticks >= _double(
        handle, starts + interval + 1
    ):
        return None
    before = _quaternion_matrix(handle, segment, index)
    after = _quaternion_matrix(handle, segment, index + 1)
    axis, angle = cspyce.raxisa(after @ before.T)
    fraction = (tick - before_ticks) / (after_ticks - before_ticks)
    step = np.asarray(cspyce.axisar(axis, fraction * angle), dtype=np.float64)
    return step @ before
