"""Spacecraft clock counts as image labels record them.

A spacecraft clock reading is a sequence of fields, each counting in units of the field
before it, and a clock's SCLK kernel gives each field's modulus and offset.
:func:`fractional_count` turns one reading's fields into an exact number of units of its
leading field, and :func:`exposure_counts` assembles the start, midtime and end counts a
host publishes for one image.  Each instrument's host module holds its own clock's
moduli, offsets and label format; this module holds no instrument's constants.
"""

from collections.abc import Sequence
from fractions import Fraction

__all__ = ['exposure_counts', 'fractional_count']


def fractional_count(
    fields: Sequence[int], moduli: Sequence[int], offsets: Sequence[int]
) -> Fraction:
    """Return a spacecraft clock reading as an exact number of its leading field's units.

    With fields ``f``, moduli ``m`` and offsets ``o``, one per field of the clock, the
    result is ``(f[0] - o[0]) + (f[1] - o[1]) / m[1] + (f[2] - o[2]) / (m[1] * m[2])``
    and so on: the leading field, plus each finer field as a fraction of one unit of the
    field before it.  The leading field's modulus plays no part.  The result is exact, a
    whole number of ticks of the finest field over the ticks in one leading unit, so a
    sum or a mean of readings stays exact until it is written as a float.

    Parameters:
        fields: The reading's fields, leading field first, without its partition.
        moduli: Each field's modulus, as the clock's SCLK kernel gives them.
        offsets: Each field's offset, as the clock's SCLK kernel gives them.

    Returns:
        The reading in units of its leading field, as an exact fraction.

    Raises:
        ValueError: If the three sequences differ in length.
    """
    ticks = 0
    scale = 1
    for index, (field, modulus, offset) in enumerate(zip(fields, moduli, offsets, strict=True)):
        if index > 0:
            ticks *= modulus
            scale *= modulus
        ticks += field - offset
    return Fraction(ticks, scale)


def exposure_counts(
    start: Fraction | None, end: Fraction | None, *, bracketed: bool
) -> dict[str, float | None]:
    """Return the spacecraft clock counts a host publishes for one image.

    Each count is written as the float nearest its exact value, and the midtime count as
    the float nearest the exact mean of the two, which the mean of the two written floats
    is not always.

    Parameters:
        start: The label's start count, exactly, or None when the label carries none.
        end: The label's stop count, exactly, or None when the label carries none.
        bracketed: Whether the two counts bracket the exposure, so that the count halfway
            between them is the middle of the exposure.

    Returns:
        ``start_time_sclk`` and ``end_time_sclk``, the two counts, and ``midtime_sclk``,
        their mean.  The mean is None when the counts do not bracket the exposure, or
        when either count is None.
    """
    midtime = None
    if bracketed and start is not None and end is not None:
        midtime = float((start + end) / 2)
    return {
        'start_time_sclk': None if start is None else float(start),
        'midtime_sclk': midtime,
        'end_time_sclk': None if end is None else float(end),
    }
