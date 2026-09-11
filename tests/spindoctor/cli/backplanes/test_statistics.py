"""Spec-first tests for the unit a backplane statistic is stated in.

Contract under test (docs/dev_guide/dev_guide_backplanes.rst "Units" and
:mod:`spindoctor.cli.backplanes.statistics`): an angular plane's statistics are
converted to degrees while its array stays in radians, what is compared is the
unit's measure rather than the whole string, and every statistic records the
unit its minimum and maximum ended up in.

``rad/pixel`` is the case that matters.  It is a unit the shipped configuration
declares, it is radians, and an equality against the whole string does not
recognise it, so a rule written that way puts one column of a degrees table in
radians per pixel.
"""

import math

import numpy as np
import pytest

from spindoctor.cli.backplanes.statistics import plane_statistics, statistics_units


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
    """A resolution in kilometres per pixel is not angular and does not convert."""
    assert statistics_units('km/pixel') == 'km/pixel'


def test_a_radian_unit_in_another_case_is_left_alone() -> None:
    """RAD is not the vocabulary's spelling, so it is not radians here and is kept as typed.

    Folding case would recognise it, and would corrupt the vocabulary's own
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
