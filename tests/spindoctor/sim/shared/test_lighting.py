"""The shared Henyey-Greenstein phase function the rings and the haze both call.

The function takes the phase angle in radians, is 1 at every phase for an
isotropic scatterer, peaks at opposition for a backscatterer and toward phase pi
for a forward scatterer, and holds its denominator at a caller's floor.
"""

import math

import pytest

from spindoctor.sim.shared.lighting import henyey_greenstein_phase


def test_henyey_greenstein_backscatter_peak() -> None:
    """P(g, phase) at opposition matches (1 - g^2)/(1 + g)^3 for g < 0."""
    g = -0.3
    assert henyey_greenstein_phase(g, 0.0) == pytest.approx((1 - g * g) / (1 + g) ** 3)


def test_henyey_greenstein_forward_scatter() -> None:
    """A dusty positive g brightens strongly toward phase pi."""
    assert henyey_greenstein_phase(0.6, math.pi) > 10.0 * henyey_greenstein_phase(0.6, 0.0)


def test_henyey_greenstein_is_unity_when_isotropic() -> None:
    """The phase function is 1 at every phase when g = 0."""
    assert henyey_greenstein_phase(0.0, math.radians(75.0)) == pytest.approx(1.0)


def test_henyey_greenstein_takes_radians() -> None:
    """At phase pi/2 radians cos(phase) is 0, so P is (1 - g^2)/(1 + g^2)^1.5."""
    g = 0.5
    expected = (1 - g * g) / (1 + g * g) ** 1.5
    assert henyey_greenstein_phase(g, math.pi / 2) == pytest.approx(expected, rel=1e-12)


def test_henyey_greenstein_denominator_floor_caps_the_value() -> None:
    """Near g = 1 at phase pi the denominator (1 - g)^3 sits below the floor."""
    g = 0.99999
    floor = 1e-12
    expected = (1 - g * g) / floor
    assert henyey_greenstein_phase(g, math.pi, denominator_floor=floor) == pytest.approx(expected)
