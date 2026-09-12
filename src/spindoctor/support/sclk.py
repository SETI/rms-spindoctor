"""Spacecraft clock counts as image labels record them.

A spacecraft clock reading is a sequence of fields, each counting in units of the field
before it, and a clock's SCLK kernel gives each field's modulus and offset.
:func:`fractional_count` turns one reading's fields into a single number in units of its
leading field, and :func:`exposure_counts` assembles the start, midtime and end counts a
host publishes for one exposure.  :func:`pds3_label_clock_counts` reads the start and stop
counts from the PDS3 label beside an image.  Each instrument's host module holds its own
clock's moduli, offsets and label format; this module holds no instrument's constants.
"""

import re
from collections.abc import Sequence

from filecache import FCPath

__all__ = ['exposure_counts', 'fractional_count', 'pds3_label_clock_counts']

_PDS3_START_KEY = 'SPACECRAFT_CLOCK_START_COUNT'
"""The PDS3 label keyword naming the clock count at the start of an image."""

_PDS3_STOP_KEY = 'SPACECRAFT_CLOCK_STOP_COUNT'
"""The PDS3 label keyword naming the clock count at the end of an image."""

_PDS3_CLOCK_COUNT_RE = re.compile(
    r'^[ \t]*(?P<key>SPACECRAFT_CLOCK_(?:START|STOP)_COUNT)[ \t]*=[ \t]*'
    r'(?:"(?P<quoted>[^"]*)"|(?P<bare>(?:[^\s"/]|/(?!\*))+))',
    re.MULTILINE,
)
"""A PDS3 label's start or stop clock count keyword and its value, quoted or bare."""


def fractional_count(fields: Sequence[int], moduli: Sequence[int], offsets: Sequence[int]) -> float:
    """Return a spacecraft clock reading as a number of its leading field's units.

    With fields ``f``, moduli ``m`` and offsets ``o``, one per field of the clock, the
    result is ``(f[0] - o[0]) + (f[1] - o[1]) / m[1] + (f[2] - o[2]) / (m[1] * m[2]) + ...``:
    the leading field, plus each finer field as a fraction of one unit of the field before
    it.  The leading field's modulus plays no part.  The reading is summed in whole ticks of
    its finest field and divided once, so the result is the nearest float to the exact
    count.

    Parameters:
        fields: The reading's fields, leading field first, without its partition.
        moduli: Each field's modulus, as the clock's SCLK kernel gives them.
        offsets: Each field's offset, as the clock's SCLK kernel gives them.

    Returns:
        The reading in units of its leading field.

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
    return ticks / scale


def exposure_counts(
    start: float | None, end: float | None, *, bracketed: bool
) -> dict[str, float | None]:
    """Return the spacecraft clock counts a host publishes for one image.

    Parameters:
        start: The label's start count, or None when the label carries none.
        end: The label's stop count, or None when the label carries none.
        bracketed: Whether the two counts bracket the exposure, so that the count halfway
            between them is the middle of the exposure.

    Returns:
        ``start_time_sclk`` and ``end_time_sclk``, the two counts as given, and
        ``midtime_sclk``, their exact mean.  The mean is None when the counts do not
        bracket the exposure, or when either count is None.
    """
    midtime = None if not bracketed or start is None or end is None else (start + end) / 2
    return {'start_time_sclk': start, 'midtime_sclk': midtime, 'end_time_sclk': end}


def pds3_label_clock_counts(image: FCPath) -> tuple[str | None, str | None]:
    """Return the clock counts the PDS3 label beside an image records.

    The label is the file with the image's name and the suffix ``.LBL``, or ``.lbl`` when
    the image's own suffix is lower case.

    Parameters:
        image: The image file.

    Returns:
        The label's ``SPACECRAFT_CLOCK_START_COUNT`` and ``SPACECRAFT_CLOCK_STOP_COUNT``,
        each as the label writes it without its quotes, or None where the label carries
        no such keyword or there is no label beside the image.
    """
    label = image.with_suffix('.LBL' if image.suffix == image.suffix.upper() else '.lbl')
    try:
        text = label.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return None, None
    counts: dict[str, str] = {}
    for match in _PDS3_CLOCK_COUNT_RE.finditer(text):
        quoted = match.group('quoted')
        value = quoted if quoted is not None else match.group('bare')
        counts.setdefault(match.group('key'), value.strip())
    return counts.get(_PDS3_START_KEY), counts.get(_PDS3_STOP_KEY)
