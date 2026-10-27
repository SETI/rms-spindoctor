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
``lumpiness / 8`` of the radius at every lumpiness; its degree 2 part, an
ellipsoid-like elongation, is about two thirds of that; it neither resizes the
body (degree 0) nor moves it to first order (degree 1); its base shape stops at
degree 5; each detail octave ``k`` adds an RMS of ``0.5**k`` times the base's
degree >= 3 RMS in its own band of degrees; and it varies around every circle of
latitude, so the body is not a body of revolution about the mesh's axis.  These
are measured on the mesh itself, independent of how the relief is synthesized:
a Fourier transform around each ring and a Clenshaw-Curtis rule over the rings
project the vertex radii on SciPy's spherical harmonics, which is exact for a
relief whose degree is at most half the ring count.
"""

import math

import numpy as np
import pytest
from scipy.special import sph_legendre_p

from spindoctor.sim.mesh_geometry import Mesh, make_irregular_mesh
from spindoctor.support.types import NDArrayFloatType

_CASES = [(0.08, 0), (0.3, 0), (0.45, 7), (0.52, 3)]
_CASE_IDS = [f'lumpiness{lum}-seed{seed}' for lum, seed in _CASES]

# Mesh for the spectral measurements of the base shape (degrees <= 5; the
# projection is exact up to degree 16 at this size), and the seeds they cover.
_SPECTRUM_LAT, _SPECTRUM_LON = 32, 64
_LUMPINESS_LEVELS = (0.08, 0.3, 0.52)
_LUMPINESS = 0.3
_SEEDS = (0, 3, 5, 7, 11, 19)
# The documented calibration: the degree >= 3 RMS per unit lumpiness.
_NON_ELLIPSOIDAL_RMS_PER_UNIT = 1.0 / 8.0
_NON_ELLIPSOIDAL_DEGREE = 3
_TOP_BASE_DEGREE = 5
# Degree 2 RMS per unit lumpiness.  Over seeds 0 to 499 it ranges from 0.021 to
# 0.20 with a median of 0.080 (two thirds of the degree >= 3 RMS); every seed
# must stay inside the wider band, and the median of seeds 0 to 39 (0.084)
# inside the narrower one, about three standard errors of a 40-seed median.
_DEGREE_2_PER_UNIT_RANGE = (0.015, 0.25)
_DEGREE_2_MEDIAN_RANGE = (0.065, 0.10)
_DEGREE_2_POPULATION_SEEDS = range(40)
# Detail octaves: each band's RMS is 0.5**k of the base's degree >= 3 RMS.  The
# fine mesh projects degrees up to 64 exactly, past octave 3's top degree 40.
_OCTAVE_LAT, _OCTAVE_LON = 128, 256
_OCTAVES = 3
_OCTAVE_SEEDS = (0, 7)
# Circles of latitude (colatitude, degrees) whose radius spread is measured.
_FINE_LAT, _FINE_LON = 128, 256
_RING_COLATITUDES_DEG = (45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0)
# The projection is exact, so the calibrated quantities match to rounding.
_EXACT = 1e-9


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


def _clenshaw_curtis(n: int) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Colatitudes ``k pi / n`` (k = 0 to n) and their Clenshaw-Curtis weights.

    The weights integrate a polynomial in ``cos(colatitude)`` of degree <= ``n``
    exactly over ``[-1, 1]``; they sum to 2.
    """
    colat = np.pi * np.arange(n + 1) / n
    weights = np.ones(n + 1)
    for j in range(1, n // 2 + 1):
        factor = 1.0 if 2 * j == n else 2.0
        weights -= factor / (4.0 * j * j - 1.0) * np.cos(2.0 * j * colat)
    ends = np.full(n + 1, 2.0)
    ends[0] = ends[-1] = 1.0
    return colat, ends / n * weights


def _relief_spectrum(
    lumpiness: float,
    seed: int,
    *,
    n_lat: int = _SPECTRUM_LAT,
    n_lon: int = _SPECTRUM_LON,
    detail_octaves: int = 0,
    lmax: int = _TOP_BASE_DEGREE,
) -> tuple[NDArrayFloatType, float]:
    """Project a mesh's relief ``r - 1`` on the spherical harmonics, per degree.

    The mesh's rings sit at colatitudes ``k pi / n_lat`` and equal steps of
    longitude, with a vertex at each pole.  A Fourier transform around each ring
    gives the relief's order-``m`` part exactly for ``m < n_lon / 2``, and a
    Clenshaw-Curtis rule over the rings integrates it against the orthonormal
    associated Legendre function exactly while the degrees sum to at most
    ``n_lat``.

    Returns:
        The relief's mean square over the sphere in each degree 0 to ``lmax``
        (a degree's RMS squared), and its total mean square, which exceeds their
        sum by the power above ``lmax``.
    """
    mesh = make_irregular_mesh(
        n_lat=n_lat, n_lon=n_lon, lumpiness=lumpiness, seed=seed, detail_octaves=detail_octaves
    )
    relief = np.linalg.norm(mesh.vertices, axis=1) - 1.0
    grid = np.vstack(
        [
            np.full(n_lon, relief[-2]),
            relief[:-2].reshape(n_lat - 1, n_lon),
            np.full(n_lon, relief[-1]),
        ]
    )
    orders = np.fft.rfft(grid, axis=1) / n_lon
    colat, weights = _clenshaw_curtis(n_lat)
    mean_square = np.zeros(lmax + 1)
    for degree in range(lmax + 1):
        for order in range(degree + 1):
            legendre = sph_legendre_p(degree, order, colat)
            coeff = 2.0 * math.pi * np.sum(weights * orders[:, order] * legendre)
            mean_square[degree] += (1.0 if order == 0 else 2.0) * abs(coeff) ** 2
    total = 2.0 * math.pi * float(np.sum(weights * np.mean(grid * grid, axis=1)))
    return mean_square / (4.0 * math.pi), total / (4.0 * math.pi)


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


@pytest.mark.parametrize('lumpiness', _LUMPINESS_LEVELS)
@pytest.mark.parametrize('seed', _SEEDS)
def test_non_ellipsoidal_relief_rms_is_an_eighth_of_lumpiness(lumpiness: float, seed: int) -> None:
    """Every seed's degree >= 3 relief has an RMS of ``lumpiness / 8``, linear in lumpiness.

    This is the irregularity a best-fit ellipsoid cannot absorb, so a scene's
    lumpiness means the same non-ellipsoidal shape error whatever its seed, and
    doubling the lumpiness doubles it.
    """
    mean_square, _ = _relief_spectrum(lumpiness, seed)
    ge3_rms = math.sqrt(float(mean_square[_NON_ELLIPSOIDAL_DEGREE:].sum()))
    assert ge3_rms / lumpiness == pytest.approx(_NON_ELLIPSOIDAL_RMS_PER_UNIT, rel=_EXACT)


@pytest.mark.parametrize('seed', _SEEDS)
def test_degree_2_relief_stays_within_the_seed_range(seed: int) -> None:
    """Each seed's ellipsoid-like elongation is present and below the lumps' size."""
    mean_square, _ = _relief_spectrum(_LUMPINESS, seed)
    low, high = _DEGREE_2_PER_UNIT_RANGE
    assert low <= math.sqrt(float(mean_square[2])) / _LUMPINESS <= high


def test_median_degree_2_relief_is_two_thirds_of_the_lumps() -> None:
    """Over many seeds the degree 2 RMS centers on two thirds of ``lumpiness / 8``."""
    per_unit = [
        math.sqrt(float(_relief_spectrum(_LUMPINESS, seed)[0][2])) / _LUMPINESS
        for seed in _DEGREE_2_POPULATION_SEEDS
    ]
    low, high = _DEGREE_2_MEDIAN_RANGE
    assert low <= float(np.median(per_unit)) <= high


@pytest.mark.parametrize('seed', _SEEDS)
def test_base_relief_stops_at_degree_5(seed: int) -> None:
    """Without detail octaves the relief carries no power above degree 5."""
    mean_square, total = _relief_spectrum(_LUMPINESS, seed)
    above = total - float(mean_square.sum())
    assert abs(above) <= _EXACT * total


@pytest.mark.parametrize('seed', _OCTAVE_SEEDS)
@pytest.mark.parametrize('octave', range(_OCTAVES + 1))
def test_each_detail_octave_band_halves_the_rms(seed: int, octave: int) -> None:
    """Octave ``k``'s band of degrees has an RMS of ``0.5**k * lumpiness / 8``.

    Band 0 is the base's degrees 3 to 5, which the octaves leave untouched;
    octave ``k`` covers degrees ``5 * 2**(k-1) + 1`` to ``5 * 2**k``.
    """
    mean_square, _ = _relief_spectrum(
        _LUMPINESS,
        seed,
        n_lat=_OCTAVE_LAT,
        n_lon=_OCTAVE_LON,
        detail_octaves=_OCTAVES,
        lmax=_TOP_BASE_DEGREE * 2**_OCTAVES,
    )
    if octave == 0:
        low, high = _NON_ELLIPSOIDAL_DEGREE, _TOP_BASE_DEGREE
    else:
        low, high = _TOP_BASE_DEGREE * 2 ** (octave - 1) + 1, _TOP_BASE_DEGREE * 2**octave
    band_rms = math.sqrt(float(mean_square[low : high + 1].sum()))
    expected = 0.5**octave * _NON_ELLIPSOIDAL_RMS_PER_UNIT
    assert band_rms / _LUMPINESS == pytest.approx(expected, rel=_EXACT)


@pytest.mark.parametrize('seed', _OCTAVE_SEEDS)
def test_detail_octaves_stop_at_their_top_degree(seed: int) -> None:
    """Three octaves carry no power above degree 40."""
    mean_square, total = _relief_spectrum(
        _LUMPINESS,
        seed,
        n_lat=_OCTAVE_LAT,
        n_lon=_OCTAVE_LON,
        detail_octaves=_OCTAVES,
        lmax=_TOP_BASE_DEGREE * 2**_OCTAVES,
    )
    assert abs(total - float(mean_square.sum())) <= _EXACT * total


@pytest.mark.parametrize('seed', _SEEDS)
def test_relief_keeps_the_mean_radius(seed: int) -> None:
    """The relief averages to zero over the sphere: lumpiness never resizes the body."""
    mean_square, _ = _relief_spectrum(_LUMPINESS, seed)
    assert math.sqrt(float(mean_square[0])) <= _EXACT * _LUMPINESS


@pytest.mark.parametrize('seed', _SEEDS)
def test_relief_does_not_move_the_body(seed: int) -> None:
    """The relief has no degree-1 part, the first-order shift of the body's center.

    A degree-1 relief ``t . u`` is, to first order, a translation by ``t``, and
    its mean square over the sphere is ``|t|**2 / 3``.
    """
    mean_square, _ = _relief_spectrum(_LUMPINESS, seed)
    shift = math.sqrt(3.0 * float(mean_square[1]))
    assert shift <= _EXACT * _LUMPINESS


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
