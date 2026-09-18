"""Values derived from the fields of a navigation metadata document.

Each of these turns a recorded field into something a query can filter on: an
epoch into the calendar date the date filters compare, and an image name into
the number the image-range filters compare.  They run where the per-image facts
are built, so the derived values are columns of the results index and the
filters are ordinary column comparisons on any backend.
"""

import re

from spindoctor.support.time import et_to_utc

__all__ = ['date_from_image_et', 'datetime_from_image_et', 'image_number_from_name']

_IMAGE_NUMBER_RE = re.compile(r'\d+')


def image_number_from_name(image_name: str | None) -> int | None:
    """Numeric portion (first digit run) of an image name's basename.

    ``N1454725799_1_CALIB.IMG`` yields ``1454725799``;
    ``lor_0003103486_0x630_sci`` yields ``3103486`` (leading zeros drop in
    the integer).  This is the value the ``--min-image`` / ``--max-image``
    range filter compares, and it is stored as a column so that comparison is
    an ordinary one rather than a call into the process that opened the
    database.

    Parameters:
        image_name: Image name or path, or None.

    Returns:
        The integer value of the first digit run, or None when the name is
        None or contains no digits.
    """
    if image_name is None:
        return None
    match = _IMAGE_NUMBER_RE.search(image_name.rsplit('/', 1)[-1])
    if match is None:
        return None
    return int(match.group(0))


def date_from_image_et(image_et: float | None) -> str | None:
    """UTC calendar date (``YYYY-MM-DD``) for a SPICE ET epoch.

    The date of the instant rounded to the nearer whole second, so an epoch in
    the last half second of a day is dated the day after, as
    :func:`datetime_from_image_et` writes it.

    Parameters:
        image_et: TDB seconds past J2000 -- the epoch the document recorded for
            the image -- or None.

    Returns:
        The UTC date string, or None when ``image_et`` is None.
    """
    if image_et is None:
        return None
    return et_to_utc(image_et, digits=0)[:10]


def datetime_from_image_et(image_et: float | None) -> str | None:
    """UTC calendar date and time (``YYYY-MM-DDTHH:MM:SS``) for a SPICE ET epoch.

    The same instant as :func:`date_from_image_et`, rounded to the nearer whole
    second.  The report shows this where a bare date would collapse many
    images taken the same day into one indistinguishable bound.

    Parameters:
        image_et: TDB seconds past J2000 -- the epoch the document recorded for
            the image -- or None.

    Returns:
        The UTC timestamp string, or None when ``image_et`` is None.
    """
    if image_et is None:
        return None
    # No digits write whole seconds with no decimal point; zero digits round
    # the same way and leave a trailing '.' behind.
    return et_to_utc(image_et, digits=None)
