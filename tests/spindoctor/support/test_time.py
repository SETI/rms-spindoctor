"""Tests for the one rule that turns an ET into a UTC time.

Every expected string here was written by SPICE's ``et2utc`` for the same epoch, with
the leapseconds kernel ``naif0012.tls`` furnished, and not by the function under test:
a nearer millisecond is ``et2utc``'s own three-decimal answer, and a floor or a ceiling
is read off its nine-decimal answer.  The epochs include the two leap seconds inserted
while Cassini was at Saturn.
"""

import math
import re
from typing import Any, cast

import pytest

from spindoctor.support.time import Pds4Rounding, et_to_pds4_utc, et_to_utc, pds4_utc_midpoint


@pytest.mark.parametrize(
    ('et', 'expected'),
    [
        (0.0, '2000-01-01T11:58:55.816Z'),
        (129399999.77, '2004-02-07T04:25:35.585Z'),
        (130700000.31, '2004-02-22T05:32:16.125Z'),
        (189345664.5839256, '2005-12-31T23:59:60.400Z'),
        (284040065.933932, '2008-12-31T23:59:60.750Z'),
        (141912064.183703, '2004-07-01T00:00:00.000Z'),
    ],
    ids=[
        'J2000',
        'a cohort epoch',
        'a cohort epoch on another day',
        'the 2005 leap second',
        'the 2008 leap second',
        'rounded into the next day',
    ],
)
def test_an_epoch_is_written_to_the_nearer_millisecond_with_a_z(et: float, expected: str) -> None:
    """The date, a T, the time to three decimals and the Z the PDS4 type requires.

    Parameters:
        et: The epoch.
        expected: What ``et2utc`` writes for it at three decimals, with the Z.
    """
    assert et_to_pds4_utc(et) == expected


@pytest.mark.parametrize(
    ('et', 'rounding', 'expected'),
    [
        (130700000.08, 'down', '2004-02-22T05:32:15.894Z'),
        (129400000.23, 'up', '2004-02-07T04:25:36.046Z'),
    ],
    ids=['down, where the nearer is later', 'up, where the nearer is earlier'],
)
def test_a_directed_rounding_takes_the_millisecond_on_its_side(
    et: float, rounding: Pds4Rounding, expected: str
) -> None:
    """A floor never states a later instant and a ceiling never an earlier one.

    Each epoch is one whose nearer millisecond is on the other side, which ``et2utc``
    writes as ``.895`` and ``.045`` respectively.

    Parameters:
        et: The epoch.
        rounding: Which way to round it.
        expected: The millisecond on that side of it.
    """
    assert et_to_pds4_utc(et, rounding=rounding) == expected


def test_whole_seconds_are_written_without_a_decimal_point() -> None:
    """Zero decimals write the seconds bare, the form the reference's ranges take."""
    assert et_to_pds4_utc(129399999.58493078, digits=0) == '2004-02-07T04:25:35Z'


@pytest.mark.parametrize(
    ('et', 'rounding', 'expected'),
    [
        (129399999.78493077, 'down', '2004-02-07T04:25:35Z'),
        (129399999.58493078, 'up', '2004-02-07T04:25:36Z'),
        (189345664.5839256, 'down', '2005-12-31T23:59:60Z'),
        (189345664.5839256, 'up', '2006-01-01T00:00:00Z'),
        (189345664.1829256, 'up', '2005-12-31T23:59:60Z'),
    ],
    ids=[
        'down from .6, which rounds to the second after',
        'up from .4, which rounds to the second before',
        'down inside a leap second',
        'up out of a leap second into the next year',
        'up into a leap second',
    ],
)
def test_a_directed_rounding_takes_the_whole_second_on_its_side(
    et: float, rounding: Pds4Rounding, expected: str
) -> None:
    """Whole seconds rounded down or up, the leap second counted as one of them.

    Parameters:
        et: The epoch.
        rounding: Which way to round it.
        expected: The whole second on that side of it.
    """
    assert et_to_pds4_utc(et, digits=0, rounding=rounding) == expected


def test_the_utc_spelling_carries_no_z() -> None:
    """The spelling the observation metadata records: three decimals, no suffix."""
    assert et_to_utc(0.0) == '2000-01-01T11:58:55.816'


def test_the_utc_spelling_with_no_digits_rounds_to_whole_seconds() -> None:
    """No digits write the nearer whole second, as ``et2utc`` writes at zero decimals."""
    assert et_to_utc(0.0, digits=None) == '2000-01-01T11:58:56'


@pytest.mark.parametrize('et', [math.nan, math.inf], ids=['NaN', 'infinity'])
def test_an_epoch_that_is_not_a_finite_number_is_refused(et: float) -> None:
    """No calendar time stands for one, and ``julian`` would write one anyway.

    Parameters:
        et: The epoch refused.
    """
    with pytest.raises(ValueError, match=re.escape(f'a finite number; got {et!r}')):
        et_to_pds4_utc(et)


def test_a_negative_number_of_decimals_is_refused() -> None:
    """A negative count names no precision a label can be written to."""
    with pytest.raises(ValueError, match=r'decimals has to be 0 or more; got -1'):
        et_to_pds4_utc(0.0, digits=-1)


def test_a_rounding_that_is_not_one_of_the_three_is_refused() -> None:
    """A spelling mistake is refused rather than read as rounding to the nearer."""
    with pytest.raises(ValueError, match=r"nearest, down, up; got 'sideways'"):
        et_to_pds4_utc(0.0, rounding=cast(Any, 'sideways'))


@pytest.mark.parametrize(
    ('start', 'stop', 'expected'),
    [
        ('2009-08-24T04:55:38.824Z', '2009-08-24T04:55:38.829Z', '2009-08-24T04:55:38.827Z'),
        ('2004-02-07T04:25:35Z', '2004-02-07T04:25:36Z', '2004-02-07T04:25:36Z'),
        ('2005-12-31T23:59:60.400Z', '2006-01-01T00:00:00.101Z', '2005-12-31T23:59:60.751Z'),
    ],
    ids=['a half millisecond', 'a half second', 'across a leap second'],
)
def test_the_midpoint_of_two_written_times_takes_a_half_up(
    start: str, stop: str, expected: str
) -> None:
    """The midpoint of two written times, in their spelling, a half going to the later.

    Each expected string is SPICE's midpoint taken up: ``et2utc`` writes the mean of the
    two epochs ``utc2et`` gives for the pair, at four decimals where the pair has three
    and at one where it has none, as ``04:55:38.8265``, ``04:25:35.5`` and
    ``23:59:60.7505``.  The first pair is W1629783475's PDS3 ``START_TIME`` and
    ``STOP_TIME``, whose ``IMAGE_MID_TIME`` is ``.827``.

    Parameters:
        start: The earlier time.
        stop: The later time.
        expected: Their midpoint, as it has to be written.
    """
    assert pds4_utc_midpoint(start, stop) == expected


def test_two_times_written_to_different_decimals_have_no_midpoint() -> None:
    """A midpoint is written to the precision of its two times, so they must share one."""
    with pytest.raises(ValueError, match=r'same number of decimals; got '):
        pds4_utc_midpoint('2004-02-07T04:25:35Z', '2004-02-07T04:25:36.500Z')
