"""Tests for the ring geometry a data label states, over plumbing backplane metadata.

What the shipped Cassini template makes of it over the cohort is tested in
``test_ring_geometry_cassini_iss_saturn.py``.
"""

from typing import Any

from spindoctor.cli.pds4.ring_geometry import RING_GEOMETRY_ATTRIBUTES, RingGeometry, ring_geometry
from spindoctor.config import DEFAULT_CONFIG

from .conftest import ring_metadata


def _statistic(minimum: float, maximum: float, units: str) -> dict[str, Any]:
    """Return one ring plane's statistic, as the backplane metadata records it.

    Parameters:
        minimum: The plane's least value.
        maximum: Its greatest value.
        units: The unit the two are in.

    Returns:
        The statistic.
    """
    return {'min': minimum, 'max': maximum, 'units': units}


STATISTICS = {
    'ring_radius': _statistic(74658.04, 136780.0, 'km'),
    'ring_longitude': _statistic(0.5, 359.5, 'deg'),
    'ring_emission_angle': _statistic(10.0, 20.0, 'deg'),
    'ring_phase_angle': _statistic(30.0, 40.0, 'deg'),
    'ring_radial_resolution': _statistic(2.11, 9.04, 'km/pixel'),
    'ring_longitudinal_resolution': _statistic(1.4e-05, 3.9e-05, 'deg/pixel'),
}
"""A statistic for every configured ring plane, in an order other than the schema's."""


def _stated(statistics: dict[str, Any]) -> RingGeometry:
    """Return the ring geometry of backplane metadata holding these ring statistics.

    Parameters:
        statistics: The ring statistics, keyed by plane name.

    Returns:
        The geometry, which metadata holding a ring statistic always has.
    """
    geometry = ring_geometry({'bodies': {}, 'rings': ring_metadata(statistics)})
    assert geometry is not None
    return geometry


def test_an_image_with_no_ring_statistic_has_no_ring_geometry() -> None:
    """Rings with no statistic, beside a ring target and an incidence angle, state nothing."""
    assert ring_geometry({'bodies': {}, 'rings': ring_metadata({})}) is None


def test_the_geometry_states_each_range_and_the_incidence_angle_in_the_schema_s_order() -> None:
    """Phase, incidence, emission, longitude and radius, each written as the tables write it.

    The incidence angle is one value, stated as the mean, the minimum and the maximum.
    """
    stated = [(each.name, each.unit, each.value) for each in _stated(STATISTICS).geometry]
    assert stated == [
        ('minimum_phase_angle', 'deg', '30.000'),
        ('maximum_phase_angle', 'deg', '40.000'),
        ('mean_incidence_angle', 'deg', '64.600'),
        ('minimum_incidence_angle', 'deg', '64.600'),
        ('maximum_incidence_angle', 'deg', '64.600'),
        ('minimum_emission_angle', 'deg', '10.000'),
        ('maximum_emission_angle', 'deg', '20.000'),
        ('minimum_inertial_ring_longitude', 'deg', '0.500'),
        ('maximum_inertial_ring_longitude', 'deg', '359.500'),
        ('minimum_ring_radius', 'km', '74658.0'),
        ('maximum_ring_radius', 'km', '136780.0'),
    ]


def test_the_grid_states_each_resolution_in_the_size_a_pixel_spans() -> None:
    """A resolution per pixel is stated in the length or the angle a pixel spans."""
    stated = [(each.name, each.unit, each.value) for each in _stated(STATISTICS).grid]
    assert stated == [
        ('minimum_radial_resolution', 'km', '2.1100'),
        ('maximum_radial_resolution', 'km', '9.0400'),
        ('minimum_longitudinal_resolution', 'deg', '0.00001400'),
        ('maximum_longitudinal_resolution', 'deg', '0.00003900'),
    ]


def test_a_plane_with_no_statistic_leaves_out_its_pair() -> None:
    """A ring plane the metadata holds no statistic for states neither of its values."""
    statistics = {
        name: value for name, value in STATISTICS.items() if name != 'ring_emission_angle'
    }
    names = [each.name for each in _stated(statistics).geometry]
    assert [name for name in names if 'emission' in name] == []


def test_every_configured_ring_plane_has_a_place_in_the_ring_geometry() -> None:
    """Each ring plane the shipped configuration declares is stated under an attribute."""
    configured = sorted(entry['name'] for entry in DEFAULT_CONFIG.backplanes.rings)
    assert sorted(RING_GEOMETRY_ATTRIBUTES) == configured
