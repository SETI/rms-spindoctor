"""Real spherical-harmonic synthesis shared by the simulator's surface fields.

Two simulator fields are random functions on the sphere built from real
spherical harmonics with independent Gaussian coefficients: the truth-side
limb-relief field (:mod:`spindoctor.sim.forward.relief`) and the irregular
mesh body's shape (:mod:`spindoctor.sim.mesh_geometry`, shared with the
navigator's predicted body).  This module holds the arithmetic they share --
drawing the coefficients, summing the orthonormal associated Legendre
functions per order, and evaluating the field -- so both fields use one
convention.  It reads no scene parameters.

**Convention.**  Latitude is measured from the field's equator, so a unit
direction ``(x, y, z)`` sits at ``sin(lat) = z`` and longitude
``atan2(y, x)``.  The coefficients are a complex ``(lmax + 1, lmax + 1)``
array ``c[l, m]`` (entries with ``m > l`` are unused) and the field is

    f(lat, lon) = sum over m of Re(Z_m(lat) * exp(i m lon)),
    Z_m(lat) = sum over l of c[l, m] * P_lm(sin(lat)),

with ``P_lm`` the associated Legendre functions normalized so that
``P_lm(sin(lat)) * exp(i m lon)`` has unit mean square times ``4 * pi`` over
the sphere.  The real functions ``P_l0``, ``sqrt(2) * P_lm * cos(m lon)``, and
``sqrt(2) * P_lm * sin(m lon)`` are then orthonormal on the sphere, so an order
``m >= 1`` coefficient ``c[l, m]`` puts ``|c[l, m]|**2 / 2`` into the field's
integrated square and an order-0 coefficient puts ``c[l, 0]**2``
(:func:`degree_mean_square`).
"""

import math

import numpy as np

from spindoctor.support.types import NDArrayComplexType, NDArrayFloatType

__all__ = [
    'degree_mean_square',
    'draw_sh_coefficients',
    'evaluate_at_directions',
    'legendre_order_sums',
]


def draw_sh_coefficients(rng: np.random.Generator, sigma: NDArrayFloatType) -> NDArrayComplexType:
    """Draw isotropic Gaussian real-harmonic coefficients, degree by degree.

    Every real orthonormal basis function of degree ``l`` gets an independent
    ``N(0, sigma[l]**2)`` coefficient, so the field is statistically isotropic
    and its expected mean square over the sphere is
    ``sum (2 l + 1) * sigma[l]**2 / (4 * pi)``.  The generator is consumed as
    two ``(lmax + 1, lmax + 1)`` standard-normal blocks (real parts, then
    imaginary parts), whatever ``sigma`` holds, so a zero ``sigma[l]`` leaves
    the other degrees' draws unchanged.

    Parameters:
        rng: The generator to draw from.
        sigma: ``(lmax + 1,)`` per-degree standard deviations.

    Returns:
        The complex ``(lmax + 1, lmax + 1)`` coefficient array ``c[l, m]``;
        the order-0 column is real, and the orders ``m >= 1`` carry the
        ``sqrt(2)`` of the real basis.
    """
    size = sigma.size
    coeffs = (rng.standard_normal((size, size)) - 1j * rng.standard_normal((size, size))) * sigma[
        :, np.newaxis
    ]
    coeffs[:, 1:] *= math.sqrt(2.0)
    coeffs[:, 0] = coeffs[:, 0].real
    return np.asarray(coeffs, dtype=np.complex128)


def legendre_order_sums(
    coeffs: NDArrayComplexType, sin_lat: NDArrayFloatType, cos_lat: NDArrayFloatType
) -> NDArrayComplexType:
    """Sum the coefficients against the Legendre functions, order by order.

    Computes ``Z_m(lat) = sum over l of c[l, m] * P_lm(sin(lat))`` for every
    order ``m`` at every latitude, advancing the orthonormal associated
    Legendre recurrence one degree at a time for all orders at once.  A degree
    whose coefficients are all zero is skipped.

    Parameters:
        coeffs: The ``(lmax + 1, lmax + 1)`` coefficient array.
        sin_lat: ``(n,)`` sines of the latitudes.
        cos_lat: ``(n,)`` cosines of the same latitudes (passed separately so a
            caller holding the latitude itself keeps full precision).

    Returns:
        The complex ``(lmax + 1, n)`` array ``Z[m, k]``.
    """
    lmax = coeffs.shape[0] - 1
    x = np.asarray(sin_lat, dtype=np.float64)
    s = np.asarray(cos_lat, dtype=np.float64)
    orders = np.arange(lmax + 1, dtype=np.float64)

    # p_prev[m] is degree l-1, p_prev2[m] degree l-2.
    zero = np.zeros((lmax + 1, x.size), dtype=np.float64)
    p_prev2 = zero.copy()
    p_prev = zero.copy()
    p_diag = np.full(x.size, 1.0 / math.sqrt(4.0 * math.pi))
    by_order = np.zeros((lmax + 1, x.size), dtype=np.complex128)
    for ell in range(lmax + 1):
        p_cur = zero.copy()
        if ell > 0:
            p_diag = -math.sqrt((2.0 * ell + 1.0) / (2.0 * ell)) * s * p_diag
            m = orders[: ell - 1]
            a = np.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - m * m))
            b = np.sqrt(((ell - 1.0) ** 2 - m * m) / (4.0 * (ell - 1.0) ** 2 - 1.0))
            p_cur[: ell - 1] = a[:, np.newaxis] * (
                x * p_prev[: ell - 1] - b[:, np.newaxis] * p_prev2[: ell - 1]
            )
            p_cur[ell - 1] = x * math.sqrt(2.0 * ell + 1.0) * p_prev[ell - 1]
        p_cur[ell] = p_diag
        if np.any(coeffs[ell, : ell + 1]):
            by_order[: ell + 1] += coeffs[ell, : ell + 1, np.newaxis] * p_cur[: ell + 1]
        p_prev2, p_prev = p_prev, p_cur
    return by_order


def evaluate_at_directions(
    coeffs: NDArrayComplexType, directions: NDArrayFloatType
) -> NDArrayFloatType:
    """Evaluate the real field at unit directions.

    The field is single-valued at the poles: every order ``m >= 1`` Legendre
    function carries a factor ``cos(lat)**m`` and vanishes there, so the
    longitude ``atan2(0, 0)`` assigns a pole does not matter.

    Parameters:
        coeffs: The ``(lmax + 1, lmax + 1)`` coefficient array.
        directions: ``(n, 3)`` unit vectors.

    Returns:
        The ``(n,)`` field values.
    """
    dirs = np.asarray(directions, dtype=np.float64)
    sin_lat = np.clip(dirs[:, 2], -1.0, 1.0)
    cos_lat = np.hypot(dirs[:, 0], dirs[:, 1])
    lon = np.arctan2(dirs[:, 1], dirs[:, 0])
    by_order = legendre_order_sums(coeffs, sin_lat, cos_lat)
    orders = np.arange(coeffs.shape[0], dtype=np.float64)
    phase = np.exp(1j * orders[:, np.newaxis] * lon[np.newaxis, :])
    return np.asarray(np.sum(by_order * phase, axis=0).real, dtype=np.float64)


def degree_mean_square(coeffs: NDArrayComplexType) -> NDArrayFloatType:
    """Each degree's contribution to the field's mean square over the sphere.

    Parameters:
        coeffs: The ``(lmax + 1, lmax + 1)`` coefficient array.

    Returns:
        ``(lmax + 1,)`` values ``(c[l, 0]**2 + sum over 1 <= m <= l of
        |c[l, m]|**2 / 2) / (4 * pi)``; their sum is the field's mean square,
        and degree 0 holds its squared mean.
    """
    power = np.abs(np.tril(coeffs)) ** 2
    power[:, 1:] /= 2.0
    return np.asarray(power.sum(axis=1) / (4.0 * math.pi), dtype=np.float64)
