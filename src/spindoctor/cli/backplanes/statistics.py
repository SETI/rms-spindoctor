"""The summary a backplane plane is reduced to, and the unit that summary is in.

An angular backplane array is in radians.  Its statistics are in degrees, because
they become columns of the bundle's index tables, which people read, and each
statistic records its unit.  The rule is :func:`statistics_units`: a unit whose part
before any ``/`` is exactly ``rad`` becomes ``deg`` with the rest kept, so
``rad/pixel`` becomes ``deg/pixel``, and every other unit is left alone.  An angular
unit other than ``rad``, such as ``mrad`` or ``arcsec``, would need a change here.

A plane of longitudes is also summarized by its range wrapped at zero, the arc of the
circle its values cover, which :func:`wrapped_range` finds.
"""

from typing import NotRequired, TypedDict

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DEGREES',
    'FULL_CIRCLE',
    'RADIANS',
    'PlaneStatistics',
    'plane_statistics',
    'statistics_units',
    'wrapped_range',
]


RADIANS = 'rad'
"""How a plane in radians declares its measure, and what its FITS header says."""

DEGREES = 'deg'
"""How a statistic converted from radians declares its measure."""

FULL_CIRCLE = 360.0
"""The degrees in a circle, where a longitude wraps back to zero."""


class PlaneStatistics(TypedDict):
    """What one backplane plane is summarized as in the metadata document.

    Attributes:
        min: The smallest value the plane measured, in ``units``.
        max: The largest value the plane measured, in ``units``.
        units: The unit those two are in: the plane's own unit restated in
            degrees when it is in radians, as :func:`statistics_units` decides,
            and the plane's own unit otherwise.
        wrapped_min: For a plane of longitudes, where the arc of the circle its
            values cover starts, in ``units``, which are degrees, as
            :func:`wrapped_range` finds it: greater than ``wrapped_max`` when the
            arc crosses zero.
        wrapped_max: Where that arc ends.
    """

    min: float
    max: float
    units: str
    wrapped_min: NotRequired[float]
    wrapped_max: NotRequired[float]


def statistics_units(units: str) -> str:
    """Return the unit a statistic taken from a plane in these units is in.

    If the part of ``units`` before any ``/`` is exactly ``rad``, that part becomes
    ``deg`` and the rest is kept, so ``rad/pixel`` becomes ``deg/pixel``.  Every
    other unit, ``RAD`` and ``km/pixel`` among them, is returned unchanged.

    Parameters:
        units: The unit the plane's values carry, as the configuration declares
            it.

    Returns:
        The unit the plane's statistic is in.
    """
    measure, slash, qualifier = units.partition('/')
    if measure != RADIANS:
        return units
    return f'{DEGREES}{slash}{qualifier}'


def wrapped_range(longitudes: NDArrayFloatType, *, resolution: float) -> tuple[float, float]:
    """Return the arc of the circle some longitudes cover, as a range wrapped at zero.

    The longitudes are taken on the circle, from 0 up to 360 degrees.  The widest gap
    between two of them adjacent on it, the gap across zero among them, is the part of
    the circle they leave uncovered, and the arc runs from the longitude after that gap
    to the one before it.  When the widest gap is the one across zero, the arc does not
    cross zero and is the plain least and greatest; when it is another, the arc crosses
    zero and starts at a greater longitude than it ends at.  Of two gaps equally wide the
    one across zero is taken, so that the plain range is kept.  Longitudes leaving no gap
    wider than ``resolution`` cover the whole circle, which is stated as 0 to 360.

    Parameters:
        longitudes: The longitudes, in degrees; at least one.
        resolution: The widest gap, in degrees, that leaves the circle covered: the
            longitudinal size of the coarsest pixel the longitudes were measured at.

    Returns:
        The arc's start and its end, in degrees.
    """
    circle = np.unique(np.mod(np.asarray(longitudes, dtype=np.float64), FULL_CIRCLE))
    # The gap after each longitude, the last one's across zero to the first
    gaps = np.diff(circle, append=circle[0] + FULL_CIRCLE)
    # The last of the widest, so that a tie goes to the gap across zero
    widest = len(gaps) - 1 - int(np.argmax(gaps[::-1]))
    if gaps[widest] <= resolution:
        return 0.0, FULL_CIRCLE
    return float(circle[(widest + 1) % len(circle)]), float(circle[widest])


def plane_statistics(
    values: NDArrayFloatType, *, units: str, longitude_resolution: float | None = None
) -> PlaneStatistics:
    """Return the range one plane spans, in the unit a reader of it wants.

    Parameters:
        values: The values the plane measured, masked pixels already dropped.
            At least one, since a plane that measured nothing has no range and
            gets no statistics entry at all.
        units: The unit those values carry, as the configuration declares it.
        longitude_resolution: For a plane of longitudes, whose statistic is in
            degrees, the longitudinal size of its coarsest pixel, in degrees.  Given,
            the statistic also records the range wrapped at zero, as
            :func:`wrapped_range` finds it with this as its resolution.

    Returns:
        The plane's minimum and maximum, converted to degrees in double precision if
        the plane's unit is in radians, and the unit they are in; and, given a
        longitude resolution, its range wrapped at zero in the same unit.

    Raises:
        ValueError: If ``values`` is empty.
    """

    converted = statistics_units(units)
    # In double precision, whatever the plane's own type, so that a value restated in
    # degrees keeps every digit its unit's format writes: a float32 conversion loses the
    # eighth decimal of a size per pixel in degrees.
    values = np.asarray(values, dtype=np.float64)
    # A unit that changed is a radian one restated in degrees, and the values
    # move with it; every other unit is left alone in both.
    if converted != units:
        values = np.degrees(values)
    statistics: PlaneStatistics = {
        'min': float(np.nanmin(values)),
        'max': float(np.nanmax(values)),
        'units': converted,
    }
    if longitude_resolution is not None:
        statistics['wrapped_min'], statistics['wrapped_max'] = wrapped_range(
            values, resolution=longitude_resolution
        )
    return statistics
