"""The reflectance of a haze layer of finite optical depth, in I/F.

The haze of :mod:`spindoctor.sim.render.bodies.atmosphere` reflects sunlight by the
isotropic multiple-scattering approximation of Hapke: the single-scattering term
with the layer's own phase function, plus the light scattered more than once, taken
as isotropic and written with Chandrasekhar's H-functions.  For a semi-infinite
layer the radiance factor is

    ``I/F = (w / 4) * mu0 / (mu0 + mu) * [P + H(mu0) * H(mu) - 1]``

(Hapke 1981, J. Geophys. Res. 86, 3039; Hapke 2012, *Theory of Reflectance and
Emittance Spectroscopy*, 2nd ed., Cambridge University Press), where ``w`` is the
single-scattering albedo, ``P`` the phase function, ``mu0`` and ``mu`` the cosines of
the incidence and emission angles, and ``H(x) = (1 + 2x) / (1 + 2 x sqrt(1 - w))`` is
Hapke's closed-form approximation to the H-function (Hapke 1981; Chandrasekhar 1960,
*Radiative Transfer*, Dover).  ``I/F`` is 1 for a white Lambert surface lit and
seen face-on.

Three extensions fit the formula to a haze over a solid body:

- **A finite column.**  A layer of optical depth ``tau`` along the line of sight
  scatters the fraction ``f = 1 - exp(-tau * (1 + mu / mu0))`` of what a
  semi-infinite one would (the light that meets a particle on the way in or out).
  The single-scattering term takes ``f``, which makes it exact for single scattering
  in a plane-parallel layer.  The multiple-scattering term takes the square of that
  fraction (for the column of the next item), because light scattered twice meets
  the column twice, so a thin
  layer reflects ``(w / 4) * P * tau`` along its line of sight, the single-scattering
  result, while a thick one reflects the semi-infinite value.  The squared fraction
  is a modeling choice that interpolates between those two exact limits.
- **A forward-scattering haze.**  Light that scatters forward keeps traveling the
  way it went, so a forward-scattering layer reflects less of it back than an
  isotropic one with the same albedo.  The multiple-scattering term uses the
  similarity relations (van de Hulst 1980, *Multiple Light Scattering*, vol. 2,
  Academic Press): the H-functions take the albedo ``w (1 - g) / (1 - w g)`` and the
  column ``(1 - w g) tau``, ``g`` being the asymmetry parameter: the phase
  function's mean cosine, which for a two-term phase function is the weighted mean
  of its two terms' (:func:`mean_cosine`).
- **Grazing paths.**  On a sphere the column toward a horizon is finite, so each
  cosine enters the formula as the effective cosine ``sqrt(mu**2 (1 - c) + c)`` with
  ``c = 2 H / (pi R)``, ``H`` the scale height and ``R`` the radius.  It is ``mu``
  overhead and ``sqrt(c)`` at the horizon, where ``1 / sqrt(c)`` is the Chapman
  grazing-incidence enhancement ``sqrt(pi R / (2 H))`` of an exponential atmosphere
  (Chapman 1931, Proc. Phys. Soc. 43, 483).

What the model leaves out: the surface beneath the haze is lit only by the sunlight
that crosses the column unscattered and is seen only through the same direct beam
(no diffuse transmission, and no light bouncing between the surface and the haze);
the multiple-scattered light is isotropic; and the column is evaluated at the point
where the line of sight meets it, not integrated along a curved path.

**Titan's phase function.**  The scenes that render Titan's haze use a two-term
Henyey-Greenstein phase function, ``(1 - f) HG(g) + f HG(g2)``, fitted to the haze
phase function the Huygens DISR instrument measured above 80 km (Tomasko et al. 2008,
Planet. Space Sci. 56, 669, its tabulated phase functions for 80-200 km).  The
tabulation at 599.9 and 713.1 nm was interpolated linearly in wavelength to 650 nm,
the Cassini ISS clear band; normalized to average 1 over the sphere (a factor of
1.008); and fitted by least squares in the logarithm of the phase function, each
tabulated scattering angle weighted by the span of angle it stands for.  The fit is
``g`` = 0.886, ``g2`` = -0.164, and ``f`` = 0.091, so the mean cosine is 0.790 (the
tabulation's own is 0.787).  It matches the tabulation within 20% from 20 to 180 deg
of scattering angle, the phase angles 0 to 160 deg; nearer the forward direction it
runs up to 1.6 times high at 10 deg and 2.5 times low at 0 deg, because two terms
cannot follow the aggregates' narrow forward peak.
"""

import math

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'chandrasekhar_h',
    'effective_cosine',
    'grazing_constant',
    'haze_reflectance',
    'mean_cosine',
    'twilight_weight',
]


def mean_cosine(g: float, g2: float, g2_fraction: float) -> float:
    """The mean cosine of a two-term Henyey-Greenstein phase function.

    Parameters:
        g: The first term's asymmetry parameter.
        g2: The second term's asymmetry parameter.
        g2_fraction: The share of the scattered light in the second term.

    Returns:
        ``(1 - g2_fraction) * g + g2_fraction * g2``; ``g`` for a one-term function.
    """
    return (1.0 - g2_fraction) * g + g2_fraction * g2


def grazing_constant(
    scale_height_px: float | NDArrayFloatType, radius_px: float
) -> float | NDArrayFloatType:
    """The constant ``c = 2 H / (pi R)`` of the effective cosine.

    Parameters:
        scale_height_px: The haze scale height ``H``, a scalar or one per pixel.
        radius_px: The body radius ``R``, in the same pixels.

    Returns:
        ``c``, with the shape of ``scale_height_px``; ``1 / sqrt(c)`` is the slant
        column toward the horizon in units of the vertical one.
    """
    return 2.0 * np.asarray(scale_height_px, dtype=np.float64) / (np.pi * max(radius_px, 1e-6))


def effective_cosine(mu: NDArrayFloatType, c: float | NDArrayFloatType) -> NDArrayFloatType:
    """The cosine of a path's zenith angle, corrected for the curvature of the column.

    ``sqrt(mu**2 (1 - c) + c)``: ``mu`` overhead, ``sqrt(c)`` at the horizon, so
    ``1 / effective_cosine`` is the slant column in units of the vertical one, finite
    on a grazing path.

    Parameters:
        mu: The cosine of the zenith angle; values below 0 count as 0 (a path at or
            below the horizon).
        c: The grazing constant of :func:`grazing_constant`.

    Returns:
        The effective cosine, in ``[sqrt(c), 1]``.
    """
    mu_clipped = np.clip(mu, 0.0, 1.0)
    return np.sqrt(mu_clipped * mu_clipped * (1.0 - c) + c)


def chandrasekhar_h(x: NDArrayFloatType, albedo: float) -> NDArrayFloatType:
    """Hapke's closed-form approximation to Chandrasekhar's H-function.

    ``H(x) = (1 + 2x) / (1 + 2x sqrt(1 - w))``, good to a few percent for isotropic
    scattering at any albedo ``w`` (Hapke 1981).

    Parameters:
        x: A direction cosine, in [0, 1].
        albedo: The single-scattering albedo, in [0, 1].

    Returns:
        ``H(x)``, at least 1.
    """
    gamma = math.sqrt(max(1.0 - albedo, 0.0))
    h_value: NDArrayFloatType = (1.0 + 2.0 * x) / (1.0 + 2.0 * gamma * x)
    return h_value


def haze_reflectance(
    tau_los: NDArrayFloatType,
    mu: NDArrayFloatType,
    mu0: NDArrayFloatType,
    *,
    albedo: float,
    g: float,
    phase_function: float,
) -> NDArrayFloatType:
    """The radiance factor of a haze column, single and multiple scattering together.

    Parameters:
        tau_los: The optical depth of the column along the line of sight.
        mu: The effective emission cosine (:func:`effective_cosine`), above 0.
        mu0: The effective incidence cosine, above 0.
        albedo: The single-scattering albedo ``w``.
        g: The Henyey-Greenstein asymmetry parameter, in (-1, 1).
        phase_function: The phase function ``P`` at the scene's phase angle.

    Returns:
        The column's I/F, before the twilight weight.
    """
    geometry = mu0 / (mu0 + mu)
    path = tau_los * (1.0 + mu / mu0)
    single = phase_function * -np.expm1(-path)
    similar_albedo = albedo * (1.0 - g) / max(1.0 - albedo * g, 1e-12)
    multiple_fraction = -np.expm1(-(1.0 - albedo * g) * path)
    multiple = (
        chandrasekhar_h(mu0, similar_albedo) * chandrasekhar_h(mu, similar_albedo) - 1.0
    ) * (multiple_fraction * multiple_fraction)
    reflectance: NDArrayFloatType = 0.25 * albedo * geometry * (single + multiple)
    return reflectance


def twilight_weight(mu0: NDArrayFloatType, delta_wrap: float) -> NDArrayFloatType:
    """The fraction of the haze column still in sunlight past the terminator.

    1 wherever the sun is above the horizon, falling as ``exp(-(e / delta)**2)`` with
    the solar depression ``e`` below it, so the haze stays lit for about
    ``delta_wrap`` radians of arc past the terminator and the weight is smooth there.

    Parameters:
        mu0: The cosine of the incidence angle (negative past the terminator).
        delta_wrap: The depression angle over which the haze fades, radians.

    Returns:
        The weight, in (0, 1].
    """
    depression = np.arcsin(np.clip(-mu0, 0.0, 1.0))
    weight: NDArrayFloatType = np.exp(-((depression / delta_wrap) ** 2))
    return weight
