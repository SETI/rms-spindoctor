"""The summary a backplane plane is reduced to, and the unit that summary is in.

A backplane array and the statistics taken from it have different readers, and
each is stated in the unit its own reader wants.  An array is machine-read, so
it carries the unit the trigonometry that produced it is already in, which for
every angular plane is radians, and its FITS header says so.  The statistics
become the columns of a bundle's global index tables, which a person reads to
decide whether an image is worth opening, and a latitude range of -88 to 81
tells them something that -1.54 to 1.42 does not.  So an angular statistic is
converted to degrees, and every statistic records the unit it ended up in
beside its minimum and maximum rather than leaving a reader to work out which
of the two conventions it is looking at.

What is compared is the unit's measure -- everything up to the first solidus --
and not the whole string.  Radians per pixel is radians, and a plane declaring
itself in them converts exactly as a plane declaring bare radians does, its
qualifier carried through untouched: an equality against the whole string
recognises no compound unit, so a rule written that way leaves the one plane
the shipped configuration declares as ``rad/pixel`` stating radians per pixel
in a table every other angular column of which is degrees -- a table mixing
units without saying so, which for a product read by a person is worse than
any loss of precision in it.  ``km`` and ``km/pixel`` are not radians and keep
their values and their units untouched.

Radians is spelled ``rad`` and degrees ``deg``, which is what the configuration
writes and what the PDS4 units-of-angle vocabulary a label must draw from
names.  A measure spelled any other way is not converted, and one that is
angular all the same -- ``mrad``, ``microrad``, ``arcsec`` -- would need to be
scaled as well as renamed, so it is a change to this module rather than a
config entry this module already handles.  What stops such a unit reaching a
table unnoticed is a test over the shipped configuration, which is where a new
declaration appears.
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

    Parameters:
        units: The unit the plane's values carry, as the configuration declares
            it.

    Returns:
        The same unit with a radian measure restated in degrees, or the unit
        unchanged when its measure is not radians.
    """

    measure, solidus, qualifier = units.partition('/')
    if measure.strip().lower() != RADIANS:
        return units
    return f'{DEGREES}{solidus}{qualifier.strip()}'


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
