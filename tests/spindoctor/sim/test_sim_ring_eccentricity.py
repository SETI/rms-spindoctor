"""Ring edge eccentricity validation (SIM-9).

A mode 1 ring edge with ``ae / a >= 1`` does not describe a closed ellipse.
The edge-radius functions raise a ValueError naming the offending values
instead of silently clamping the eccentricity.
"""

import math

import numpy as np
import pytest

from spindoctor.sim.shared.ring import compute_edge_radii_array, compute_edge_radius_at_angle


def test_scalar_edge_radius_raises_on_impossible_eccentricity() -> None:
    """A scalar edge radius with ae > a raises and names the ae/a values."""
    with pytest.raises(ValueError, match=r'ae/a = 150\.0/100\.0 = 1\.5'):
        compute_edge_radius_at_angle(
            0.0, a=100.0, ae=150.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
        )


def test_scalar_edge_radius_raises_on_parabolic_eccentricity() -> None:
    """The boundary case e == 1 (parabolic, not an ellipse) also raises."""
    with pytest.raises(ValueError, match=r'ae/a = 100\.0/100\.0 = 1\.0'):
        compute_edge_radius_at_angle(
            0.0, a=100.0, ae=100.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
        )


def test_array_edge_radii_raise_on_impossible_eccentricity() -> None:
    """The vectorized edge radii with ae > a raise and name the ae/a values."""
    angles = np.linspace(0.0, 2.0 * np.pi, 8)
    with pytest.raises(ValueError, match=r'ae/a = 150\.0/100\.0 = 1\.5'):
        compute_edge_radii_array(
            angles, a=100.0, ae=150.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
        )


def test_error_message_calls_out_physical_impossibility() -> None:
    """The error explains that a closed elliptical edge requires e < 1."""
    with pytest.raises(ValueError, match='physically impossible'):
        compute_edge_radius_at_angle(
            0.0, a=100.0, ae=150.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
        )


def test_scalar_edge_radius_accepts_near_limit_eccentricity() -> None:
    """An extreme but valid eccentricity (e = 0.99) still computes a radius."""
    radius = compute_edge_radius_at_angle(
        0.0, a=100.0, ae=99.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
    )
    assert math.isfinite(radius)


def test_array_edge_radii_accept_circular_edge() -> None:
    """A circular edge (ae = 0) computes the constant semi-major radius."""
    angles = np.linspace(0.0, 2.0 * np.pi, 8)
    radii = compute_edge_radii_array(
        angles, a=100.0, ae=0.0, long_peri=0.0, rate_peri=0.0, epoch=0.0, time=0.0
    )
    assert np.allclose(radii, 100.0)
