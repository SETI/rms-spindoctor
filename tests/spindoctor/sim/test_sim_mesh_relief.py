"""The irregular mesh's relief is a smooth, genuinely irregular surface.

``make_irregular_mesh`` must describe one solid body whose radius depends only on
the direction from the center.  A relief written in the mesh's own (latitude,
longitude) parametrization is discontinuous at the poles, where every longitude
meets: the rings around a pole keep their full spread of radii however close
they come to it, so refining the mesh turns the pole into a near-vertical pit or
spike and the body renders as banded shells around a hole.  Two tests rule that
out: the radius is continuous at the poles, and the steepest face slope
converges as the mesh is refined.

The relief must also stay irregular, and irregular in the right way.  Its
non-ellipsoidal part (spherical-harmonic degrees 3 and up) has an RMS of
``lumpiness / 8`` of the radius; it neither resizes the body (degree 0) nor moves
it (degree 1); and it varies around every circle of latitude, so the body is not
a body of revolution about the mesh's axis.  These are measured on the mesh
itself, by quadrature over a fine mesh's vertices against the polynomials of
degree 0 to 2, independent of how the relief is synthesized.
"""

import numpy as np
import pytest

from spindoctor.sim.mesh_geometry import Mesh, make_irregular_mesh
from spindoctor.support.types import NDArrayFloatType

_CASES = [(0.08, 0), (0.3, 0), (0.45, 7), (0.52, 3)]
_CASE_IDS = [f'lumpiness{lum}-seed{seed}' for lum, seed in _CASES]

# Fine mesh for the quadrature measurements, and the seeds they cover.
_FINE_LAT, _FINE_LON = 128, 256
_LUMPINESS = 0.3
_SEEDS = (0, 3, 5, 7, 11, 19)
# The documented calibration: the degree >= 3 RMS per unit lumpiness.
_NON_ELLIPSOIDAL_RMS_PER_UNIT = 1.0 / 8.0
# Circles of latitude (colatitude, degrees) whose radius spread is measured.
_RING_COLATITUDES_DEG = (45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0)


def _face_slopes_deg(mesh: Mesh) -> NDArrayFloatType:
    """Angle in degrees between each face normal and its centroid's radial direction."""
    p0 = mesh.vertices[mesh.faces[:, 0]]
    p1 = mesh.vertices[mesh.faces[:, 1]]
    p2 = mesh.vertices[mesh.faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centroids = (p0 + p1 + p2) / 3.0
    radial = centroids / np.linalg.norm(centroids, axis=1)[:, None]
    cosines = np.clip(np.sum(normals * radial, axis=1), -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(cosines)), dtype=np.float64)


def _pole_ring_spread(n_lat: int, lumpiness: float, seed: int, pole: str) -> float:
    """Largest radius difference between a pole vertex and its nearest ring."""
    n_lon = 2 * n_lat
    mesh = make_irregular_mesh(n_lat=n_lat, n_lon=n_lon, lumpiness=lumpiness, seed=seed)
    radii = np.linalg.norm(mesh.vertices, axis=1)
    if pole == 'north':
        return float(np.max(np.abs(radii[:n_lon] - radii[-2])))
    return float(np.max(np.abs(radii[-2 - n_lon : -2] - radii[-1])))


def _relief_degrees(lumpiness: float, seed: int) -> dict[str, float]:
    """Split a fine mesh's relief ``r - 1`` into its degree 0, 1, and >= 3 parts.

    The ring vertices sample the sphere on an equal-angle grid, so weighting each
    by ``sin(colatitude)`` makes the weighted mean a quadrature over the sphere.
    A weighted least-squares fit of the polynomials of degree 0 to 2 in the unit
    direction is then the relief's projection on degrees 0 to 2.

    Returns:
        ``mean`` (the degree-0 mean radius change), ``shift`` (the length of the
        degree-1 translation vector), and ``ge3_rms`` (the RMS left after the
        degree 0 to 2 fit), all in units of the unit radius.
    """
    mesh = make_irregular_mesh(n_lat=_FINE_LAT, n_lon=_FINE_LON, lumpiness=lumpiness, seed=seed)
    ring = mesh.vertices[:-2]
    radii = np.linalg.norm(ring, axis=1)
    x, y, z = (ring / radii[:, None]).T
    weights = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    weights /= weights.sum()
    relief = radii - 1.0
    ones = np.ones_like(x)
    degree_1 = np.stack([ones, x, y, z], axis=1)
    degree_2 = np.stack([ones, x, y, z, x * y, y * z, z * x, x * x - y * y, 3 * z * z - 1], axis=1)
    sqrt_w = np.sqrt(weights)

    def fit(basis: NDArrayFloatType) -> NDArrayFloatType:
        coeffs, *_ = np.linalg.lstsq(basis * sqrt_w[:, None], relief * sqrt_w, rcond=None)
        return np.asarray(coeffs, dtype=np.float64)

    # A degree-1 relief t . u is, to first order, a translation of the body by t.
    shift = fit(degree_1)[1:]
    residual = relief - degree_2 @ fit(degree_2)
    return {
        'mean': float(np.sum(weights * relief)),
        'shift': float(np.linalg.norm(shift)),
        'ge3_rms': float(np.sqrt(np.sum(weights * residual * residual))),
    }


@pytest.mark.parametrize(('lumpiness', 'seed'), _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize('pole', ['north', 'south'])
def test_mesh_radius_is_continuous_at_the_poles(lumpiness: float, seed: int, pole: str) -> None:
    """The ring nearest a pole closes on the pole's radius as the mesh is refined.

    A continuous relief differs from the pole's radius by about its slope times
    the ring's arc ``pi / n_lat``, so quadrupling ``n_lat`` cuts the spread about
    fourfold; a relief discontinuous at the pole keeps its spread.
    """
    coarse = _pole_ring_spread(32, lumpiness, seed, pole)
    fine = _pole_ring_spread(128, lumpiness, seed, pole)
    assert fine <= 0.4 * coarse


@pytest.mark.parametrize(('lumpiness', 'seed'), _CASES, ids=_CASE_IDS)
def test_steepest_face_slope_converges_under_refinement(lumpiness: float, seed: int) -> None:
    """Refining the mesh samples the same surface: its steepest slope barely moves.

    A relief that is singular anywhere grows ever steeper there as the faces
    shrink, toward a vertical wall; a smooth relief converges on its true slope.
    """
    coarse = make_irregular_mesh(n_lat=32, n_lon=64, lumpiness=lumpiness, seed=seed)
    fine = make_irregular_mesh(n_lat=128, n_lon=256, lumpiness=lumpiness, seed=seed)
    coarse_max = float(_face_slopes_deg(coarse).max())
    fine_max = float(_face_slopes_deg(fine).max())
    assert fine_max <= coarse_max + 2.0


@pytest.mark.parametrize('seed', _SEEDS)
def test_non_ellipsoidal_relief_rms_is_an_eighth_of_lumpiness(seed: int) -> None:
    """Every seed's degree >= 3 relief has an RMS of ``lumpiness / 8``.

    This is the irregularity a best-fit ellipsoid cannot absorb, so a scene's
    lumpiness means the same non-ellipsoidal shape error whatever its seed.
    """
    ge3_rms = _relief_degrees(_LUMPINESS, seed)['ge3_rms']
    assert ge3_rms / _LUMPINESS == pytest.approx(_NON_ELLIPSOIDAL_RMS_PER_UNIT, rel=0.03)


@pytest.mark.parametrize('seed', _SEEDS)
def test_relief_keeps_the_mean_radius(seed: int) -> None:
    """The relief averages to zero over the sphere: lumpiness never resizes the body."""
    mean = _relief_degrees(_LUMPINESS, seed)['mean']
    assert abs(mean) <= 0.002 * _LUMPINESS


@pytest.mark.parametrize('seed', _SEEDS)
def test_relief_does_not_move_the_body(seed: int) -> None:
    """The relief has no degree-1 part, the first-order shift of the body's center."""
    shift = _relief_degrees(_LUMPINESS, seed)['shift']
    assert shift <= 0.002 * _LUMPINESS


@pytest.mark.parametrize('seed', _SEEDS)
def test_radius_varies_around_circles_of_latitude(seed: int) -> None:
    """The body is not a body of revolution about the mesh's axis.

    The relief is isotropic, so a circle of latitude crosses the lumps like any
    other circle on the body: around each one the radius spreads by a fair share
    of the relief's RMS.
    """
    mesh = make_irregular_mesh(n_lat=_FINE_LAT, n_lon=_FINE_LON, lumpiness=_LUMPINESS, seed=seed)
    radii = np.linalg.norm(mesh.vertices[:-2], axis=1).reshape(_FINE_LAT - 1, _FINE_LON)
    rows = [round(colat / 180.0 * _FINE_LAT) - 1 for colat in _RING_COLATITUDES_DEG]
    spread = float(np.mean(radii[rows].std(axis=1)))
    assert spread >= 0.5 * _NON_ELLIPSOIDAL_RMS_PER_UNIT * _LUMPINESS
