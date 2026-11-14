"""The B-V color a simulated star record carries for its spectral class.

The record's Johnson B magnitude is its V magnitude plus the B-V color of its
spectral class.  An A0 star has a B-V of exactly zero by definition, so its B
and V magnitudes match.  If the star catalog cannot map a class to a color, the
star takes the Sun's B-V instead.
"""

import pytest

from spindoctor.sim.star_records import star_record_from_params


def test_an_a0_star_has_equal_b_and_v_magnitudes() -> None:
    """An A0 star keeps its zero B-V rather than taking the solar fallback."""
    entry = {'name': 'S', 'vmag': 5.0, 'spectral_class': 'A0'}
    star = star_record_from_params(entry, index=0, default_v=0.0, default_u=0.0)
    assert star.johnson_mag_b == pytest.approx(star.johnson_mag_v, abs=1e-12)


def test_an_unknown_spectral_class_takes_the_solar_color() -> None:
    """If the catalog cannot map the class, B is V (5.0) plus the solar B-V (0.63)."""
    entry = {'name': 'S', 'vmag': 5.0, 'spectral_class': 'XX'}
    star = star_record_from_params(entry, index=0, default_v=0.0, default_u=0.0)
    assert star.johnson_mag_b == pytest.approx(5.63, abs=1e-12)
