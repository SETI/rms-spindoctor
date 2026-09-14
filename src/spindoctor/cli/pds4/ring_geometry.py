"""The ring geometry a data label states, from the image's ring backplane statistics.

The rings dictionary, ``PDS4_RINGS_1O00_1F00``, describes the ring geometry of an image in
``rings:Reprojection_Geometry``, within ``rings:Ring_Reprojection``.  A data label of an
image with ring backplanes fills it from what the backplane stage records for the image:
the least and the greatest value of each ring plane, except that the ring longitude's
range is the one its statistic records wrapped at zero, as the dictionary defines a
longitude range, and the incidence angle of sunlight on the ring plane, one angle over
the whole image.  Each value is written as the global
index tables write its statistic, in the format
:data:`~spindoctor.cli.pds4.global_index.INDEX_VALUE_FORMATS` gives its unit, and stated
in the unit its attribute is defined in, a size per pixel in the length or the angle a
pixel spans.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.cli.pds4.global_index import INDEX_VALUE_FORMATS

__all__ = ['RING_GEOMETRY_ATTRIBUTES', 'RingAttribute', 'RingGeometry', 'ring_geometry']


RING_GEOMETRY_ATTRIBUTES: dict[str, str] = {
    'ring_phase_angle': 'phase_angle',
    'ring_emission_angle': 'emission_angle',
    'ring_longitude': 'inertial_ring_longitude',
    'ring_radius': 'ring_radius',
    'ring_radial_resolution': 'radial_resolution',
    'ring_longitudinal_resolution': 'longitudinal_resolution',
}
"""The rings dictionary attribute each configured ring plane's range is stated as.

Keyed by the plane's configured name; the attribute is given without its ``minimum_`` or
``maximum_``, which the least and the greatest value take.  The ring longitude is measured
from the ring plane's ascending node on the J2000 equator with no co-rotating frame, so it
is an inertial longitude, and its range is the arc of longitude the image covers: its
minimum is greater than its maximum where that arc crosses zero.
"""

_PER_PIXEL = '/pixel'
"""What follows the unit of a size per pixel, which a resolution attribute states without."""

_INCIDENCE_ATTRIBUTES = (
    'mean_incidence_angle',
    'minimum_incidence_angle',
    'maximum_incidence_angle',
)
"""The attributes the one incidence angle over an image is stated as, all three alike."""


@dataclass(frozen=True)
class RingAttribute:
    """One attribute of the ring geometry a data label states.

    Attributes:
        name: The rings dictionary attribute, without its prefix, as in
            ``minimum_ring_radius``.
        unit: The unit the label states it in.
        value: The value, as the label writes it.
    """

    name: str
    unit: str
    value: str


@dataclass(frozen=True)
class RingGeometry:
    """The ring geometry of one image, as its data label states it.

    Attributes:
        geometry: The attributes ``rings:Reprojection_Geometry`` holds before its grid
            parameters, in the order the schema gives them: the phase angle's range, the
            incidence angle, and the ranges of the emission angle, the inertial ring
            longitude and the ring radius.
        grid: The attributes ``rings:Reprojection_Grid_Parameters`` holds, in the order
            the schema gives them: the ranges of the radial and the longitudinal
            resolution.
    """

    geometry: tuple[RingAttribute, ...]
    grid: tuple[RingAttribute, ...]


def _written(value: float, units: str) -> str:
    """Write one value as the global index tables write a statistic in its unit.

    Parameters:
        value: The value.
        units: The unit it is in, as a statistic records it.

    Returns:
        The value in the format :data:`INDEX_VALUE_FORMATS` gives the unit.
    """
    return INDEX_VALUE_FORMATS[units].render(value)


_PLAIN = ('min', 'max')
"""The ends of a plane's range as its statistic records them, least and greatest."""

_WRAPPED = ('wrapped_min', 'wrapped_max')
"""The ends of a longitude's range as its statistic records them, wrapped at zero."""


def _range(
    statistics: Mapping[str, Any], plane: str, *, ends: tuple[str, str] = _PLAIN
) -> tuple[RingAttribute, ...]:
    """Return the two attributes one ring plane's statistic is stated as, or none.

    Parameters:
        statistics: The image's ring statistics, keyed by plane name.
        plane: The configured name of the plane.
        ends: The statistic's keys for the range's minimum and its maximum.

    Returns:
        The plane's minimum and maximum, under the attribute
        :data:`RING_GEOMETRY_ATTRIBUTES` gives it, each stated in the statistic's unit
        less any ``/pixel``; or none when the image has no statistic for the plane.
    """
    if plane not in statistics:
        return ()
    statistic = statistics[plane]
    attribute = RING_GEOMETRY_ATTRIBUTES[plane]
    unit = statistic['units'].removesuffix(_PER_PIXEL)
    minimum, maximum = ends
    return (
        RingAttribute(
            name=f'minimum_{attribute}',
            unit=unit,
            value=_written(statistic[minimum], statistic['units']),
        ),
        RingAttribute(
            name=f'maximum_{attribute}',
            unit=unit,
            value=_written(statistic[maximum], statistic['units']),
        ),
    )


def ring_geometry(backplane_metadata: Mapping[str, Any]) -> RingGeometry | None:
    """Return the ring geometry an image's data label states.

    An image's ring geometry is stated when its backplane metadata holds a ring
    statistic.  Each plane the metadata holds a statistic for gives its least and its
    greatest value, the ring longitude its range wrapped at zero; a plane with none gives
    neither, so its pair is left out.  The
    incidence angle the metadata's ``rings`` block records is stated as the mean, the
    minimum and the maximum incidence angle alike, since it is one angle over the image.

    Parameters:
        backplane_metadata: The image's backplane metadata, whose ``rings`` block holds
            the ring statistics under ``backplanes`` and the incidence angle as
            ``incidence_angle``, a value with its unit.

    Returns:
        The geometry, or None when the metadata holds no ring statistic, so that the
        image has no ring backplanes for its label to describe.
    """
    rings = backplane_metadata.get('rings', {})
    statistics = rings.get('backplanes', {})
    if len(statistics) == 0:
        return None
    incidence = rings['incidence_angle']
    incidence_value = _written(incidence['value'], incidence['units'])
    return RingGeometry(
        geometry=(
            *_range(statistics, 'ring_phase_angle'),
            *(
                RingAttribute(name=name, unit=incidence['units'], value=incidence_value)
                for name in _INCIDENCE_ATTRIBUTES
            ),
            *_range(statistics, 'ring_emission_angle'),
            *_range(statistics, 'ring_longitude', ends=_WRAPPED),
            *_range(statistics, 'ring_radius'),
        ),
        grid=(
            *_range(statistics, 'ring_radial_resolution'),
            *_range(statistics, 'ring_longitudinal_resolution'),
        ),
    )
