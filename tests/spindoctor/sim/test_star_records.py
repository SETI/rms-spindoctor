"""The B-V color a simulated star record carries for its spectral class.

The record's Johnson B magnitude is its V magnitude plus the B-V color of its
spectral class.  An A0 star has a B-V of exactly zero by definition, so its B
and V magnitudes match; a class the star catalog cannot map to a color takes
the solar value instead.
"""

import pytest

from spindoctor.sim.star_records import star_record_from_params


def test_an_a0_star_has_equal_b_and_v_magnitudes() -> None:
    """An A0 star keeps its zero B-V rather than taking the solar fallback."""
    entry = {'name': 'S', 'vmag': 5.0, 'spectral_class': 'A0'}
    star = star_record_from_params(entry, index=0, default_v=0.0, default_u=0.0)
    assert star.johnson_mag_b == pytest.approx(star.johnson_mag_v, abs=1e-12)


def test_an_unknown_spectral_class_takes_the_solar_color() -> None:
    """A class with no catalog color gets the solar B-V of 0.63 on top of its V of 5.0."""
    entry = {'name': 'S', 'vmag': 5.0, 'spectral_class': 'XX'}
    star = star_record_from_params(entry, index=0, default_v=0.0, default_u=0.0)
    assert star.johnson_mag_b == pytest.approx(5.63, abs=1e-12)
