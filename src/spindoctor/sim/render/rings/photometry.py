"""Single-scattering photometry of the ring system, at a point and along a ramp.

With ``mu = |sin B_obs|``, ``mu0 = |sin B_sun|``, the observer on the lit face iff
``sign(B_obs) == sign(B_sun)``, and the one-term Henyey-Greenstein ``P(g, alpha)``:

- lit:   ``I = A/4 * P * mu0/(mu0 + mu) * (1 - exp(-tau*(1/mu0 + 1/mu)))``
- unlit: ``I = A/4 * P * mu0/(mu0 - mu) * (exp(-tau/mu0) - exp(-tau/mu))``,
  with the limit ``A/4 * P * (tau/mu) * exp(-tau/mu)`` when ``|mu0 - mu| < 1e-6``.

The transmission is ``exp(-tau/mu)``.  Every form is a sum of exponentials in tau, or
tau times an exponential, so its mean over a stretch where tau changes linearly has a
closed form.  The means here take the stretch's smallest and largest tau and are exact
for any slope, including a steep one where tau changes by hundreds across a pixel; a
slope of zero gives the point value.
"""

import numpy as np

from spindoctor.sim.shared.inplace import result_array, ufunc_result
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'RING_ALBEDO_DEFAULT',
    'RING_PHASE_G_DEFAULT',
    'ring_reflection_factor',
    'ring_reflection_mean',
    'ring_transmission_mean',
]

RING_ALBEDO_DEFAULT: float = 0.5
"""Single-scattering albedo (times normalization); dusty features set higher values."""

RING_PHASE_G_DEFAULT: float = -0.3
"""Henyey-Greenstein asymmetry: main-ring backscatter; dusty features set about +0.6."""

# Below this |mu0 - mu| the unlit closed form is numerically indeterminate (0/0) and
# the analytic limit replaces it.
_UNLIT_MU_MATCH_TOL: float = 1e-6
# Below this argument the series of _exp_minus_tail is used; four terms leave a
# relative error under 2e-10 there.
_SERIES_LIMIT: float = 1e-2


def ring_reflection_factor(
    tau: NDArrayFloatType,
    mu: float,
    mu0: float,
    *,
    lit: bool,
) -> NDArrayFloatType:
    """The geometric factor of the single-scattering closed forms.

    Multiplied by ``A/4 * P`` this is the emitted intensity: the lit form
    saturates toward ``mu0/(mu0 + mu)`` at high tau, while the unlit form
    peaks at moderate tau and falls to zero for an opaque ring (nothing
    diffuses through) -- the real dark-side inversion.

    Parameters:
        tau: Normal optical depth per pixel (non-negative).
        mu: ``|sin B_obs|``, nonzero.
        mu0: ``|sin B_sun|``, nonzero.
        lit: Whether the observer sees the lit face
            (``sign(B_obs) == sign(B_sun)``).

    Returns:
        The per-pixel geometric factor.
    """
    # Each form is built in place on this call's own arrays, since the renderer
    # evaluates it over a whole oversampled frame.  Every step is the IEEE operation
    # of the plain closed form, so the factor is unchanged to the last bit, and the
    # result has the plain form's shape and type.
    factor = result_array(tau, mu, mu0)
    if lit:
        np.negative(tau, out=factor)
        factor *= 1.0 / mu0 + 1.0 / mu
        np.exp(factor, out=factor)
        np.subtract(1.0, factor, out=factor)
        factor *= mu0 / (mu0 + mu)
        return ufunc_result(factor)
    if abs(mu0 - mu) < _UNLIT_MU_MATCH_TOL:
        # The mu0 -> mu limit of the unlit form (the closed form is 0/0):
        # (tau / mu) * exp(-tau / mu).
        scaled = np.divide(tau, mu, out=np.empty_like(factor))
        np.negative(scaled, out=factor)
        np.exp(factor, out=factor)
        factor *= scaled
        return ufunc_result(factor)
    np.negative(tau, out=factor)
    factor /= mu0
    np.exp(factor, out=factor)
    behind = np.negative(tau, out=np.empty_like(factor))
    behind /= mu
    np.exp(behind, out=behind)
    factor -= behind
    factor *= mu0 / (mu0 - mu)
    return ufunc_result(factor)


def _mean_exp_fraction(x: NDArrayFloatType) -> NDArrayFloatType:
    """``(1 - exp(-x)) / x`` for ``x >= 0``, which is 1 at ``x = 0``."""
    positive = x > 0.0
    safe = np.where(positive, x, 1.0)
    result: NDArrayFloatType = np.where(positive, -np.expm1(-safe) / safe, 1.0)
    return result


def _exp_minus_tail(x: NDArrayFloatType) -> NDArrayFloatType:
    """``(1 - exp(-x)) / x - exp(-x)`` for ``x >= 0``, by its series near 0."""
    series = x * (0.5 + x * (-1.0 / 3.0 + x * (1.0 / 8.0 - x / 30.0)))
    direct = _mean_exp_fraction(x) - np.exp(-x)
    result: NDArrayFloatType = np.where(x < _SERIES_LIMIT, series, direct)
    return result


def _mean_exp(tau_lo: NDArrayFloatType, spread: NDArrayFloatType, rate: float) -> NDArrayFloatType:
    """The mean of ``exp(-rate * tau)`` as tau runs linearly over its range."""
    result: NDArrayFloatType = np.exp(-rate * tau_lo) * _mean_exp_fraction(rate * spread)
    return result


def ring_reflection_mean(
    tau_lo: NDArrayFloatType,
    tau_hi: NDArrayFloatType,
    mu: float,
    mu0: float,
    *,
    lit: bool,
) -> NDArrayFloatType:
    """The mean of :func:`ring_reflection_factor` where tau changes linearly.

    Parameters:
        tau_lo: The smallest optical depth of each stretch (non-negative).
        tau_hi: The largest optical depth of each stretch, at least ``tau_lo``.
        mu: ``|sin B_obs|``, nonzero.
        mu0: ``|sin B_sun|``, nonzero.
        lit: Whether the observer sees the lit face.

    Returns:
        The mean geometric factor of each stretch.
    """
    spread = tau_hi - tau_lo
    if lit:
        mean = 1.0 - _mean_exp(tau_lo, spread, 1.0 / mu0 + 1.0 / mu)
        return mu0 / (mu0 + mu) * mean
    if abs(mu0 - mu) < _UNLIT_MU_MATCH_TOL:
        # The mean of a * tau * exp(-a * tau) with a = 1/mu.
        rate = 1.0 / mu
        x = rate * spread
        result: NDArrayFloatType = np.exp(-rate * tau_lo) * (
            rate * tau_lo * _mean_exp_fraction(x) + _exp_minus_tail(x)
        )
        return result
    difference = _mean_exp(tau_lo, spread, 1.0 / mu0) - _mean_exp(tau_lo, spread, 1.0 / mu)
    return mu0 / (mu0 - mu) * difference


def ring_transmission_mean(
    tau_lo: NDArrayFloatType, tau_hi: NDArrayFloatType, mu: float
) -> NDArrayFloatType:
    """The mean of ``exp(-tau/mu)`` where tau changes linearly.

    Parameters:
        tau_lo: The smallest optical depth of each stretch (non-negative).
        tau_hi: The largest optical depth of each stretch, at least ``tau_lo``.
        mu: ``|sin B_obs|``, nonzero.

    Returns:
        The mean transmission of each stretch.
    """
    return _mean_exp(tau_lo, tau_hi - tau_lo, 1.0 / mu)
