"""The summary a backplane plane is reduced to, and the unit that summary is in.

An angular backplane array is in radians.  Its statistics are in degrees, because
they become columns of the bundle's index tables, which people read, and each
statistic records its unit.  The rule is :func:`statistics_units`: a unit whose part
before any ``/`` is exactly ``rad`` becomes ``deg`` with the rest kept, so
``rad/pixel`` becomes ``deg/pixel``, and every other unit is left alone.  An angular
unit other than ``rad``, such as ``mrad`` or ``arcsec``, would need a change here.
"""

from typing import TypedDict

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['DEGREES', 'RADIANS', 'PlaneStatistics', 'plane_statistics', 'statistics_units']


RADIANS = 'rad'
"""How a plane in radians declares its measure, and what its FITS header says."""

DEGREES = 'deg'
"""How a statistic converted from radians declares its measure."""


class PlaneStatistics(TypedDict):
    """What one backplane plane is summarized as in the metadata document.

    Attributes:
        min: The smallest value the plane measured, in ``units``.
        max: The largest value the plane measured, in ``units``.
        units: The unit those two are in, which is the degrees restatement of
            the plane's own unit wherever the plane is angular and the plane's
            own unit otherwise.
    """

    min: float
    max: float
    units: str


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

    Raises:
        TypeError: If ``units`` is not a string, as a configuration key written
            with no value is not.
        ValueError: If ``units`` is empty or blank.
    """

    if not isinstance(units, str):
        raise TypeError(f'units must be a string; got {type(units).__name__}')
    if not units.strip():
        raise ValueError('units must name a measure; got an empty value')

    measure, slash, qualifier = units.partition('/')
    if measure != RADIANS:
        return units
    return f'{DEGREES}{slash}{qualifier}'


def plane_statistics(values: NDArrayFloatType, *, units: str) -> PlaneStatistics:
    """Return the range one plane spans, in the unit a reader of it wants.

    Parameters:
        values: The values the plane measured, masked pixels already dropped.
            At least one, since a plane that measured nothing has no range and
            gets no statistics entry at all.
        units: The unit those values carry, as the configuration declares it.

    Returns:
        The plane's minimum and maximum, converted to degrees if the plane is
        angular, and the unit they are in.

    Raises:
        ValueError: If ``values`` is empty.
    """

    converted = statistics_units(units)
    # A unit that changed is a radian one restated in degrees, and the values
    # move with it; every other unit is left alone in both.
    if converted != units:
        values = np.degrees(values)
    return {
        'min': float(np.nanmin(values)),
        'max': float(np.nanmax(values)),
        'units': converted,
    }
