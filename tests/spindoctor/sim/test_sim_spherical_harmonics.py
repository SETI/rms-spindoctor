"""Real spherical-harmonic synthesis: orthonormal basis, exact power, pole values.

The limb-relief field and the irregular mesh's shape both read their amplitude
from :func:`degree_mean_square`, so it must equal the field's actual mean square
over the sphere, degree by degree.  The check integrates
:func:`evaluate_at_directions` with a Gauss-Legendre rule in ``sin(lat)`` and an
equal-spaced rule in longitude, which is exact for these band-limited fields.
"""

import math

import numpy as np
import pytest

from spindoctor.sim.spherical_harmonics import (
    degree_mean_square,
    draw_sh_coefficients,
    evaluate_at_directions,
)
from spindoctor.support.types import NDArrayFloatType

_LMAX = 6


def _quadrature_grid() -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Directions and weights (summing to 1) of an exact rule for degree <= 2 * _LMAX."""
    nodes, gauss_weights = np.polynomial.legendre.leggauss(_LMAX + 2)
    n_lon = 2 * _LMAX + 3
    lon = 2.0 * math.pi * np.arange(n_lon) / n_lon
    sin_lat, lon_grid = np.meshgrid(nodes, lon, indexing='ij')
    cos_lat = np.sqrt(1.0 - sin_lat * sin_lat)
    directions = np.stack(
        [cos_lat * np.cos(lon_grid), cos_lat * np.sin(lon_grid), sin_lat], axis=-1
    ).reshape(-1, 3)
    weights = np.repeat(gauss_weights / 2.0, n_lon) / n_lon
    return directions, weights


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_degree_mean_square_matches_the_field(seed: int) -> None:
    """Each degree's reported power equals that degree's integrated square."""
    rng = np.random.default_rng(seed)
    coeffs = draw_sh_coefficients(rng, np.linspace(1.0, 0.3, _LMAX + 1))
    directions, weights = _quadrature_grid()
    expected = []
    for ell in range(_LMAX + 1):
        single = np.zeros_like(coeffs)
        single[ell, : ell + 1] = coeffs[ell, : ell + 1]
        values = evaluate_at_directions(single, directions)
        expected.append(float(np.sum(weights * values * values)))
    np.testing.assert_allclose(degree_mean_square(coeffs), expected, rtol=1e-10, atol=1e-14)


def test_real_basis_functions_are_orthonormal() -> None:
    """Order 0 at unit coefficient and orders >= 1 at ``sqrt(2)`` have unit norm."""
    directions, weights = _quadrature_grid()
    functions = []
    for ell in range(_LMAX + 1):
        for m in range(ell + 1):
            for part in (1.0,) if m == 0 else (math.sqrt(2.0), -1j * math.sqrt(2.0)):
                coeffs = np.zeros((_LMAX + 1, _LMAX + 1), dtype=np.complex128)
                coeffs[ell, m] = part
                functions.append(evaluate_at_directions(coeffs, directions))
    basis = np.array(functions)
    gram = (basis * weights) @ basis.T * (4.0 * math.pi)
    np.testing.assert_allclose(gram, np.eye(len(functions)), atol=1e-10)


def test_field_is_single_valued_at_the_poles() -> None:
    """Approaching a pole from any longitude gives the pole's own value."""
    coeffs = draw_sh_coefficients(np.random.default_rng(3), np.ones(_LMAX + 1))
    eps = 1e-7
    lon = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    for z in (1.0, -1.0):
        near = np.stack(
            [eps * np.cos(lon), eps * np.sin(lon), np.full(lon.size, z * math.sqrt(1 - eps**2))],
            axis=1,
        )
        pole = evaluate_at_directions(coeffs, np.array([[0.0, 0.0, z]]))
        np.testing.assert_allclose(evaluate_at_directions(coeffs, near), pole[0], atol=1e-5)


def test_zero_sigma_degree_draws_zero_coefficients() -> None:
    """A degree drawn at zero sigma is exactly zero and leaves the others' draws alone."""
    sigma = np.ones(_LMAX + 1)
    full = draw_sh_coefficients(np.random.default_rng(5), sigma)
    sigma[2] = 0.0
    gapped = draw_sh_coefficients(np.random.default_rng(5), sigma)
    assert not np.any(gapped[2])
    np.testing.assert_array_equal(np.delete(gapped, 2, axis=0), np.delete(full, 2, axis=0))
