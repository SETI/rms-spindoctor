"""Spec-first tests for the unit a backplane statistic is stated in.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "Units" and
:mod:`spindoctor.cli.backplanes.statistics`): an angular plane's statistics are
converted to degrees while its array stays in radians, what is compared is the
unit's measure rather than the whole string, and every statistic records the
unit its minimum and maximum ended up in.

``rad/pixel`` is the case that matters.  It is a unit the shipped configuration
declares, it is radians, and an equality against the whole string does not
recognize it, so a rule written that way puts one column of a degrees table in
radians per pixel.
"""

import math

import numpy as np
import pytest

from spindoctor.cli.backplanes.statistics import (
    plane_statistics,
    statistics_units,
    wrapped_range,
)


def test_a_radian_unit_becomes_degrees() -> None:
    """A plane declared in bare radians is summarized in degrees."""
    assert statistics_units('rad') == 'deg'


def test_a_compound_radian_unit_converts_only_its_measure() -> None:
    """Radians per pixel is radians, so its statistic is degrees per pixel."""
    assert statistics_units('rad/pixel') == 'deg/pixel'


def test_a_unit_only_spelled_like_radians_is_left_alone() -> None:
    """Gradians are not radians, so a measure merely ending in rad does not convert."""
    assert statistics_units('grad') == 'grad'


def test_a_compound_non_radian_unit_is_left_alone() -> None:
    """A resolution in kilometers per pixel is not angular and does not convert."""
    assert statistics_units('km/pixel') == 'km/pixel'


def test_a_radian_unit_in_another_case_is_left_alone() -> None:
    """RAD is not the vocabulary's spelling, so it is not radians here and is kept as typed.

    Folding case would recognize it, and would corrupt the vocabulary's own
    upper-case tokens; the tests over the shipped configuration refuse the
    spelling instead.
    """
    assert statistics_units('RAD') == 'RAD'


def test_a_radian_unit_with_a_space_beside_it_is_left_alone() -> None:
    """A space beside the measure makes a spelling the vocabulary lacks, kept as typed."""
    assert statistics_units('rad ') == 'rad '


def test_a_qualifier_is_carried_through_untouched() -> None:
    """Only the measure is converted: a qualifier keeps its spelling, a space included."""
    assert statistics_units('rad/ pixel') == 'deg/ pixel'


def test_statistics_of_an_angular_plane_are_converted_and_say_so() -> None:
    """An angular plane's range is in degrees and the statistic names that unit."""
    stats = plane_statistics(np.array([1.4e-05, 3.9e-05]), units='rad/pixel')
    assert stats['min'] == pytest.approx(math.degrees(1.4e-05))
    assert stats['max'] == pytest.approx(math.degrees(3.9e-05))
    assert stats['units'] == 'deg/pixel'


def test_statistics_of_a_non_angular_plane_keep_their_values_and_unit() -> None:
    """A plane that is not angular is summarized in the unit its array carries."""
    stats = plane_statistics(np.array([1000.0, 2000.0]), units='km')
    assert stats['min'] == pytest.approx(1000.0)
    assert stats['units'] == 'km'


def test_a_statistic_is_restated_in_degrees_in_double_precision() -> None:
    """A float32 plane's radians become degrees without losing the eighth decimal.

    0.004730729 radians per pixel, as float32 holds it, is 0.27105080 degrees per pixel in
    double precision and 0.27105078 in single.
    """
    stats = plane_statistics(np.array([0.004730729], dtype=np.float32), units='rad/pixel')
    assert f'{stats["max"]:.8f}' == '0.27105080'


CROSSING = np.array([359.7, 0.1, 359.9, 0.4])
"""Longitudes in degrees either side of zero, an arc 0.7 degrees wide, in no order."""


def test_an_arc_crossing_zero_starts_at_a_greater_longitude_than_it_ends() -> None:
    """The arc runs from the longitude after the widest gap to the one before it."""
    assert wrapped_range(CROSSING, resolution=0.01) == (359.7, 0.4)


def test_an_arc_not_crossing_zero_is_the_plain_least_and_greatest() -> None:
    """When the widest gap is the one across zero, the arc is the plain range."""
    assert wrapped_range(np.array([150.0, 100.0, 200.0]), resolution=0.01) == (100.0, 200.0)


def test_of_two_widest_gaps_the_one_across_zero_keeps_the_plain_range() -> None:
    """Two longitudes half a circle apart leave two gaps alike, and the plain range stands."""
    assert wrapped_range(np.array([280.0, 100.0]), resolution=0.01) == (100.0, 280.0)


def test_a_longitude_of_360_degrees_is_the_one_at_zero() -> None:
    """A value rounded up to 360 is on the circle at zero, not beyond the last longitude."""
    assert wrapped_range(np.array([360.0, 10.0]), resolution=0.01) == (0.0, 10.0)


def test_longitudes_leaving_no_gap_wider_than_the_resolution_cover_the_circle() -> None:
    """A gap no wider than the coarsest pixel leaves the circle covered, 0 to 360."""
    assert wrapped_range(np.arange(0.0, 360.0, 1.0), resolution=1.0) == (0.0, 360.0)


def test_a_longitude_plane_s_statistic_records_its_wrapped_range_in_degrees() -> None:
    """Given a resolution, a plane's statistic holds its range wrapped at zero, converted."""
    stats = plane_statistics(np.radians(CROSSING), units='rad', longitude_resolution=0.01)
    assert stats == {
        'min': pytest.approx(0.1),
        'max': pytest.approx(359.9),
        'units': 'deg',
        'wrapped_min': pytest.approx(359.7),
        'wrapped_max': pytest.approx(0.4),
    }
