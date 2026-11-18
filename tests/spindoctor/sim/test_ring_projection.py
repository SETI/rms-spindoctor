"""The shared ring-plane projection helpers: exact forms and identities.

Pins the normative opening-angle projection: the du/dv equations as written,
the longitude-frame convention (ring-plane longitude from the ascending node,
with the node angle entering only the final sky rotation), the flat-ring
regression identity at |B| = 90, and the line-of-sight depth convention
(nearest point at lam = 270 deg for B = 30, node = 0; ansae at zero depth).
"""

import math

import numpy as np
import pytest

from spindoctor.sim.shared.ring import (
    ring_los_depth,
    ring_plane_from_sky,
    ring_radial_scale,
    ring_sky_from_plane,
)
from spindoctor.support.types import NDArrayFloatType


def _arr(*values: float) -> NDArrayFloatType:
    """A float64 array literal for scalar-style geometry probes."""
    return np.asarray(values, dtype=np.float64)


def test_projection_matches_the_normative_equations() -> None:
    """dv/du reproduce the written forms at a generic (r, lam, B, node)."""
    r = 40.0
    lam = math.radians(35.0)
    b_obs = 25.0
    node = 55.0
    dv, du = ring_sky_from_plane(_arr(r), _arr(lam), opening_deg_obs=b_obs, node_deg=node)
    x = r * math.cos(lam)
    y = r * math.sin(lam)
    sin_b = math.sin(math.radians(b_obs))
    node_rad = math.radians(node)
    assert du[0] == pytest.approx(x * math.cos(node_rad) - y * sin_b * math.sin(node_rad))
    assert dv[0] == pytest.approx(-(x * math.sin(node_rad) + y * sin_b * math.cos(node_rad)))


def test_node_line_point_is_not_foreshortened() -> None:
    """A point at lam = 0 lands along the node direction independent of B.

    This is the longitude-frame arbiter: lam lives in the ring plane and the
    node angle enters only the sky rotation, so the ascending-node point
    (lam = 0) projects to the node's sky position angle at full radius for
    every opening angle.
    """
    r = 25.0
    node = 40.0
    node_rad = math.radians(node)
    for b_obs in (10.0, 30.0, 90.0):
        dv, du = ring_sky_from_plane(_arr(r), _arr(0.0), opening_deg_obs=b_obs, node_deg=node)
        assert du[0] == pytest.approx(r * math.cos(node_rad))
        assert dv[0] == pytest.approx(-r * math.sin(node_rad))


def test_round_trip_recovers_ring_plane_coordinates() -> None:
    """plane -> sky -> plane is the identity for a spread of points."""
    rng = np.random.default_rng(11)
    r = rng.uniform(5.0, 80.0, size=64)
    lam = rng.uniform(0.0, 2.0 * math.pi, size=64)
    dv, du = ring_sky_from_plane(r, lam, opening_deg_obs=27.0, node_deg=73.0)
    r_back, lam_back, _x, _y = ring_plane_from_sky(dv, du, opening_deg_obs=27.0, node_deg=73.0)
    np.testing.assert_allclose(r_back, r, rtol=1e-12)
    np.testing.assert_allclose(lam_back, lam, rtol=1e-12)


def test_face_on_projection_reduces_to_sky_plane_circles() -> None:
    """|B| = 90 is today's circle geometry: r equals the sky-plane radius."""
    rng = np.random.default_rng(5)
    dv = rng.uniform(-50.0, 50.0, size=128)
    du = rng.uniform(-50.0, 50.0, size=128)
    for b_obs in (90.0, -90.0):
        r, _lam, _x, _y = ring_plane_from_sky(dv, du, opening_deg_obs=b_obs, node_deg=33.0)
        np.testing.assert_allclose(r, np.hypot(dv, du), rtol=1e-12)


def test_edge_on_inverse_projection_raises() -> None:
    """B = 0 is edge-on: the inverse projection does not exist."""
    with pytest.raises(ValueError, match=r'not invertible.*edge-on'):
        ring_plane_from_sky(_arr(1.0), _arr(1.0), opening_deg_obs=0.0, node_deg=0.0)


def test_nearest_point_is_lam_270_at_b30_node0() -> None:
    """The pinned depth configuration: B = 30, node = 0.

    The depth maximum (nearest point) sits at lam = 270 deg and the ansae
    (lam = 0 and 180) have zero depth by construction.
    """
    lam = np.radians(np.arange(0.0, 360.0, 1.0))
    r = np.full_like(lam, 50.0)
    dv, du = ring_sky_from_plane(r, lam, opening_deg_obs=30.0, node_deg=0.0)
    _r, _lam, _x, y = ring_plane_from_sky(dv, du, opening_deg_obs=30.0, node_deg=0.0)
    dlos = ring_los_depth(y, opening_deg_obs=30.0)
    assert int(np.argmax(dlos)) == 270
    assert dlos[0] == pytest.approx(0.0, abs=1e-9)
    assert dlos[180] == pytest.approx(0.0, abs=1e-9)


def test_depth_magnitude_matches_the_written_form() -> None:
    """dlos = -y * cos(B): the nearest point of a radius-r ring is r*cos(B)."""
    dv, du = ring_sky_from_plane(
        _arr(50.0), _arr(math.radians(270.0)), opening_deg_obs=30.0, node_deg=0.0
    )
    _r, _lam, _x, y = ring_plane_from_sky(dv, du, opening_deg_obs=30.0, node_deg=0.0)
    dlos = ring_los_depth(y, opening_deg_obs=30.0)
    assert dlos[0] == pytest.approx(50.0 * math.cos(math.radians(30.0)))


def test_radial_scale_is_unity_face_on() -> None:
    """At |B| = 90 ring-plane and sky-plane radial distances coincide."""
    rng = np.random.default_rng(3)
    dv = rng.uniform(-40.0, 40.0, size=32)
    du = rng.uniform(-40.0, 40.0, size=32)
    r, _lam, x, y = ring_plane_from_sky(dv, du, opening_deg_obs=90.0, node_deg=0.0)
    scale = ring_radial_scale(r, x, y, opening_deg_obs=90.0)
    np.testing.assert_allclose(scale, np.ones_like(scale), rtol=1e-12)


def test_radial_scale_foreshortens_the_minor_axis() -> None:
    """On the projected minor axis the radial scale is 1/sin(B); ansae stay 1."""
    b_obs = 30.0
    minor = ring_radial_scale(_arr(50.0), _arr(0.0), _arr(50.0), opening_deg_obs=b_obs)
    ansa = ring_radial_scale(_arr(50.0), _arr(50.0), _arr(0.0), opening_deg_obs=b_obs)
    assert minor[0] == pytest.approx(1.0 / math.sin(math.radians(b_obs)))
    assert ansa[0] == pytest.approx(1.0)
