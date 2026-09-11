import datetime
import math
from typing import Literal, cast

import julian

Pds4Rounding = Literal['nearest', 'down', 'up']
"""How :func:`et_to_pds4_utc` takes an instant between two values of its last digit."""

_PDS4_ROUNDINGS: tuple[Pds4Rounding, ...] = ('nearest', 'down', 'up')
"""Every rounding :func:`et_to_pds4_utc` accepts."""


def now_iso() -> str:
    """Returns the current time as an ISO 8601 formatted string with timezone information.

    Returns:
        Current time as an ISO 8601 formatted string.
    """

    return datetime.datetime.now().astimezone().isoformat()


def now_dt() -> datetime.datetime:
    """Returns the current time as a datetime object with timezone information.

    Returns:
        Current time as a timezone-aware datetime object.
    """

    return datetime.datetime.now().astimezone()


def dt_delta_str(start_time: datetime.datetime, end_time: datetime.datetime) -> str:
    """Returns the difference between two datetime objects as a string representation.

    Parameters:
        start_time: The starting datetime.
        end_time: The ending datetime.

    Returns:
        String representation of the time difference.
    """

    return str(end_time - start_time)


def _tai_from_et(et: float) -> float:
    """Returns an ET as TAI seconds, the one step every UTC spelling here starts from.

    Parameters:
        et: The SPICE ET time (equivalent to TDB).

    Returns:
        The same instant in TAI seconds, through ``julian``'s model of TDB.
    """

    return cast(float, julian.tai_from_tdb(et))


def _utc_from_tai(tai: float, *, digits: int | None, suffix: str) -> str:
    """Returns a TAI time as a UTC calendar string, the one step every spelling ends with.

    Parameters:
        tai: The instant, in TAI seconds.
        digits: The decimals of a second to write, rounding to the nearer value of
            the last one, a half rounding up; None for whole seconds with no decimal
            point.
        suffix: What follows the time: ``Z`` or nothing.

    Returns:
        ``YYYY-MM-DDThh:mm:ss`` with the decimals and the suffix asked for, through
        ``julian``'s leap-second table.
    """

    return cast(str, julian.iso_from_tai(tai, digits=digits, suffix=suffix))


def _pds4_utc_from_tai(tai: float, digits: int) -> str:
    """Returns a TAI time in the PDS4 spelling, to the decimals given.

    Parameters:
        tai: The instant, in TAI seconds, rounded to the nearer value of the last digit
            written.
        digits: The number of decimals of a second, 0 for whole seconds.

    Returns:
        ``YYYY-MM-DDThh:mm:ss`` with the decimals asked for and a trailing ``Z``.
    """
    # julian writes zero decimals with a trailing point, which the PDS4 type does not
    # allow; None writes the same whole seconds without it.
    return _utc_from_tai(tai, digits=None if digits == 0 else digits, suffix='Z')


def _pds4_decimals(pds4_utc: str) -> int:
    """Returns the number of decimals of a second a PDS4 UTC time is written to.

    Parameters:
        pds4_utc: The time, as in ``2004-02-07T04:25:35.585Z`` or
            ``2004-02-07T04:25:35Z``.

    Returns:
        The digits after the seconds' decimal point, 0 when there is none.
    """
    seconds = pds4_utc.removesuffix('Z').rpartition(':')[2]
    return len(seconds.partition('.')[2])


def _tai_from_pds4_utc(pds4_utc: str) -> float:
    """Returns a time in the PDS4 spelling as TAI seconds, through ``julian``.

    Parameters:
        pds4_utc: The time, with its trailing ``Z``; a leap second is read as one.

    Returns:
        The instant, in TAI seconds.
    """
    return cast(float, julian.tai_from_iso(pds4_utc.removesuffix('Z')))


def et_to_utc(et: float, digits: int | None = 3) -> str:
    """Returns the UTC time for a given ET time.

    The time is written ``YYYY-MM-DDThh:mm:ss.sss``, with the seconds rounded to the
    nearer value of the last digit written, a half rounding up, so an instant in the
    last half of a day's last second can be written as the first of the next day.  A
    leap second is written as second 60.

    Parameters:
        et: The SPICE ET time (equivalent to TDB).
        digits: The number of digits to include after the decimal point.  None writes
            whole seconds, rounded to the nearer, with no decimal point; 0 rounds the
            same way and leaves the decimal point behind.

    Returns:
        The UTC time as a string.
    """

    return _utc_from_tai(_tai_from_et(et), digits=digits, suffix='')


def et_to_pds4_utc(et: float, *, digits: int = 3, rounding: Pds4Rounding = 'nearest') -> str:
    """Returns an ET the way a PDS4 label writes a UTC time.

    The form is PDS4's ``ASCII_Date_Time_YMD_UTC``: the date as year, month and day, a
    ``T``, the time of day with the seconds written to ``digits`` decimals, and the
    ``Z`` the type requires, as in ``2004-02-07T04:25:35.585Z``.  With ``digits`` of 0
    the seconds are whole and carry no decimal point, as in ``2004-02-07T04:25:35Z``.
    A leap second is written as second 60, which the type allows on the days one was
    inserted.  The conversion is the one :func:`et_to_utc` makes; only the spelling
    and the rounding differ.

    ``rounding`` decides an instant that falls between two values of the last digit
    written.  ``nearest`` writes the nearer of the two, a half rounding up, which is
    what SPICE's ``et2utc`` writes, and which gives back a time recorded to the digits
    written: an epoch computed from one lies within a few nanoseconds of it, on either
    side, where rounding down or up would move it a whole unit of the last digit
    whenever it lands on the far side.  ``down`` writes the one at or before the
    instant and ``up`` the one at or after it, which is how a range is written at a
    coarser precision than the times it bounds: its start rounded down and its stop
    up, it contains each of them written to the nearest at a finer precision, since
    every value of the coarser last digit is also one of the finer, and rounding to
    the nearest never carries an instant past one.  Since 1972 TAI and UTC differ by
    a whole number of seconds, so a value rounded in one is rounded in the other.

    Parameters:
        et: The epoch, as SPICE ET (TDB seconds past J2000).
        digits: The number of decimals of a second to write, 0 for whole seconds.
            The default of three is a millisecond, the precision a Cassini image's
            start and stop are recorded to in its PDS3 label and index.
        rounding: Which way an instant between two values of the last digit goes.

    Returns:
        The UTC time, in the form above.

    Raises:
        ValueError: If ``et`` is NaN or infinite, ``digits`` is negative, or
            ``rounding`` is not ``nearest``, ``down`` or ``up``.  The message names
            the value refused.
        TypeError: If ``et`` is not a number.
    """

    if not math.isfinite(et):
        raise ValueError(f'An epoch has to be a finite number; got {et!r}')
    if digits < 0:
        raise ValueError(f'The number of decimals has to be 0 or more; got {digits!r}')
    if rounding not in _PDS4_ROUNDINGS:
        raise ValueError(
            f'The rounding has to be one of {", ".join(_PDS4_ROUNDINGS)}; got {rounding!r}'
        )

    tai = _tai_from_et(et)
    if rounding != 'nearest':
        scaled = tai * 10**digits
        tai = (math.floor(scaled) if rounding == 'down' else math.ceil(scaled)) / 10**digits
    return _pds4_utc_from_tai(tai, digits)


def pds4_utc_midpoint(start: str, stop: str) -> str:
    """Returns the midpoint of two times a PDS4 label writes, in the same spelling.

    ``start`` and ``stop`` are two times :func:`et_to_pds4_utc` wrote, to the same number
    of decimals.  Each is a whole value of its last digit, so their midpoint is exact: a
    value of that digit, or halfway between two.  It is written to the same decimals,
    a half rounding up, to the later of the two.  That is how Cassini's PDS3 labels
    write an image's ``IMAGE_MID_TIME``, halfway through an exposure an odd number of
    milliseconds long, where rounding the midpoint of the two epochs instead would go
    whichever way the float of the half value happened to fall.

    Parameters:
        start: One time, in the PDS4 spelling, as in ``2009-08-24T04:55:38.824Z``.
        stop: The other, in the same spelling and to the same number of decimals.

    Returns:
        The midpoint, in the same spelling: ``2009-08-24T04:55:38.827Z`` for a start of
        ``.824`` and a stop of ``.829``.  A leap second between the two counts as one
        of the seconds between them.

    Raises:
        ValueError: If the two are written to different numbers of decimals.  The
            message gives both.
    """
    digits = _pds4_decimals(start)
    if _pds4_decimals(stop) != digits:
        raise ValueError(
            f'The two times have to be written to the same number of decimals; got '
            f'{start!r} and {stop!r}'
        )
    units = 10**digits
    start_units = round(_tai_from_pds4_utc(start) * units)
    stop_units = round(_tai_from_pds4_utc(stop) * units)
    # Twice the midpoint is a whole number of units, so adding one before halving takes
    # a half up and leaves a whole value as it is.
    return _pds4_utc_from_tai((start_units + stop_units + 1) // 2 / units, digits)


def utc_to_et(utc: str) -> float:
    """Returns the ET time (TDB seconds) for a given UTC time string.

    Parameters:
        utc: The UTC time as an ISO 8601 formatted string (e.g.,
            "2008-01-01 12:00:00" or "2008-01-01T12:00:00").

    Returns:
        The SPICE ET time (equivalent to TDB) in seconds as a float.
    """

    result = julian.tdb_from_tai(julian.tai_from_iso(utc))
    return float(result)
