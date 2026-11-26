"""Photometric surface-scattering laws for the image-side body renderer.

The renderer shades body discs with a scene-selected law --
Lambert, Lommel-Seeliger, Minnaert(k), or a lunar-Lambert blend -- plus an
optional opposition-surge factor.  The law is a truth key: the navigator's
predicted-body model always shades with Lambert
(:func:`spindoctor.sim.shared.ellipsoid.lambert_from_normals`), so a
non-Lambert scene plants a photometric mismatch that moves the terminator
and the limb-darkening profile by a known amount.

All laws are normalized to 1 at disc center under head-on illumination
(``mu = mu0 = 1``) and expressed in ``mu0 = cos(incidence)`` and
``mu = cos(emission)``; for this renderer's orthographic view ``mu`` is the
surface normal's z component.  Forms:

- ``lambert``: ``I = mu0``.
- ``lommel_seeliger``: ``I = 2 * mu0 / (mu0 + mu)``, which rises toward the
  sunlit limb and reaches 2 there.
- ``minnaert``: ``I = mu0**k * mu**(k - 1)``; k = 1 is Lambert, k = 0.5
  is the classic limb-brightened lunar value.  The disc-center
  normalization holds.  Below k = 1 the law grows without bound toward the
  limb, so its value is capped at :data:`MINNAERT_CEILING`, the
  Lommel-Seeliger law's own ceiling.  Without the cap the brightest limb
  pixel would depend on the pixel size rather than on the surface.
- ``lunar_lambert``: ``I = 2 * L(alpha) * mu0 / (mu0 + mu)
  + (1 - L(alpha)) * mu0`` with the McEwen (1991) cubic blend
  ``L(alpha) = 1 - 0.019 * a + 2.42e-4 * a**2 - 1.46e-6 * a**3``
  (a = phase in degrees), clipped to [0, 1]: pure Lommel-Seeliger at
  alpha = 0, pure Lambert once the cubic reaches 0 (~119 deg).

A law's value may exceed 1, and the body stages carry it uncapped to the
detector, which saturates at its full well as a real camera does.

The opposition surge is a simple normalized exponential factor
``(1 + amplitude * exp(-alpha / width)) / (1 + amplitude)``: an interim
knob for the realism match, planting the surge's brightness-versus-phase
signature.  It is 1 at opposition, so a scene far from opposition renders
darker by ``1 / (1 + amplitude)``.

None of the laws has a night-side term, so the night side of the visible
hemisphere renders black, as does the far hemisphere.
"""

import math
from typing import cast

import numpy as np

from spindoctor.sim.shared.inplace import result_array, ufunc_result
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'MINNAERT_CEILING',
    'PHOTOMETRIC_LAWS',
    'incidence_cosines',
    'shade_surface',
    'surge_factor',
]

# The scene-selectable law vocabulary (schema-validated).
PHOTOMETRIC_LAWS: frozenset[str] = frozenset(
    {'lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert'}
)

MINNAERT_CEILING = 2.0
"""The largest value the Minnaert law takes: the Lommel-Seeliger law's limb value."""

# McEwen (1991) lunar-Lambert blend coefficients (phase in degrees).
_LL_A = -0.019
_LL_B = 2.42e-4
_LL_C = -1.46e-6


def incidence_cosines(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Cosine of the incidence angle for image-frame unit surface normals.

    Parameters:
        normal_v: V component of the unit surface normal in image coordinates.
        normal_u: U component of the unit surface normal in image coordinates.
        normal_z: Z (toward-observer) component of the unit surface normal.
        illumination_angle: In-plane light direction in radians; 0 is from
            the top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        ``cos(incidence)`` per pixel (negative on the night side).
    """
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )
    normals = (normal_v, normal_u, normal_z)
    if not any(isinstance(normal, np.ndarray) for normal in normals) or (
        len({np.result_type(normal, 1.0) for normal in normals}) > 1
    ):
        # Scalar normals, or normals of mixed types: the plain expression, which keeps
        # their type.  With a float32 v and u and a float64 z, it rounds v's and u's
        # terms and their sum in float32, which a sum built in float64 would not.
        return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + normal_z * illum_z)
    # v * illum_v + u * illum_u + z * illum_z, summed in place in that order, which
    # gives the same bits, shape, and type as the plain expression with two fewer
    # temporaries.
    cos_i = np.multiply(normal_v, illum_v, out=result_array(normal_v, normal_u, normal_z))
    term = np.multiply(normal_u, illum_u, out=np.empty_like(cos_i))
    cos_i += term
    np.multiply(normal_z, illum_z, out=term)
    cos_i += term
    return ufunc_result(cos_i)


def surge_factor(phase_angle: float, *, amplitude: float, width_deg: float) -> float:
    """The normalized opposition-surge brightness factor at a phase angle.

    ``(1 + amplitude * exp(-alpha / width)) / (1 + amplitude)``: 1 at exact
    opposition, ``1 / (1 + amplitude)`` far from it.

    Parameters:
        phase_angle: Phase angle in radians.
        amplitude: Surge amplitude (0 disables the factor).
        width_deg: Angular e-folding width of the surge in degrees of phase.

    Returns:
        The multiplicative brightness factor in (0, 1].
    """
    if amplitude <= 0.0:
        return 1.0
    width_rad = math.radians(max(width_deg, 1e-6))
    return (1.0 + amplitude * math.exp(-abs(phase_angle) / width_rad)) / (1.0 + amplitude)


def shade_surface(
    cos_incidence: NDArrayFloatType,
    cos_emission: NDArrayFloatType,
    *,
    law: str = 'lambert',
    phase_angle: float = 0.0,
    minnaert_k: float = 0.5,
    surge_amplitude: float = 0.0,
    surge_width_deg: float = 6.0,
) -> NDArrayFloatType:
    """Surface brightness under the selected photometric law.

    Parameters:
        cos_incidence: ``mu0`` per pixel (negative values are night side).
        cos_emission: ``mu`` per pixel; > 0 marks the visible hemisphere.
        law: One of :data:`PHOTOMETRIC_LAWS`.
        phase_angle: Phase angle in radians (drives the lunar-Lambert blend
            and the opposition surge).
        minnaert_k: Minnaert exponent (used by ``law='minnaert'`` only).
        surge_amplitude: Opposition-surge amplitude (0 = no surge).
        surge_width_deg: Opposition-surge angular width in degrees.

    Returns:
        The law value times the surge factor on the visible hemisphere, and 0 off
        it.  It is 0 on the night side and may exceed 1 toward a limb-brightened
        law's sunlit limb.

    Raises:
        ValueError: On an unknown law name.
    """
    if law not in PHOTOMETRIC_LAWS:
        raise ValueError(
            f'unknown photometric law {law!r}; expected one of {sorted(PHOTOMETRIC_LAWS)}'
        )

    visible = np.greater(cos_emission, 0.0)
    mu0 = np.clip(cos_incidence, 0.0, 1.0)

    if law == 'lambert':
        intensity = mu0
    else:
        mu = np.clip(cos_emission, 0.0, 1.0)
        if law == 'lommel_seeliger':
            intensity = _lommel_seeliger(mu0, mu)
        elif law == 'minnaert':
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                intensity = np.where(mu > 0.0, mu0**minnaert_k * mu ** (minnaert_k - 1.0), 0.0)
            intensity = np.minimum(np.nan_to_num(intensity, nan=0.0), MINNAERT_CEILING)
        else:  # lunar_lambert
            blend = _lunar_lambert_blend(phase_angle)
            intensity = blend * _lommel_seeliger(mu0, mu) + (1.0 - blend) * mu0

    # where(visible, intensity * surge, 0), which this matches to the last bit, and in
    # shape and type.  The intensity is a new array this call made, read nowhere after,
    # so it is built there when it has the result's shape; else in a new array.
    shaded = intensity
    if not isinstance(intensity, np.ndarray) or intensity.shape != np.broadcast_shapes(
        intensity.shape, visible.shape
    ):
        shaded = result_array(intensity, visible)
    np.multiply(
        intensity,
        surge_factor(phase_angle, amplitude=surge_amplitude, width_deg=surge_width_deg),
        out=shaded,
    )
    np.copyto(shaded, 0.0, where=np.logical_not(visible))
    return shaded


def _lommel_seeliger(mu0: NDArrayFloatType, mu: NDArrayFloatType) -> NDArrayFloatType:
    """The Lommel-Seeliger term ``2 * mu0 / (mu0 + mu)``, 0 where degenerate."""
    denominator = mu0 + mu
    # where(denominator > 0, 2 * mu0 / denominator, 0) with NaN set to 0, in place on
    # this call's own arrays, which gives the same bits as the plain expression.
    value = np.multiply(2.0, mu0, out=result_array(mu0, mu))
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(value, denominator, out=value)
    np.copyto(value, 0.0, where=np.logical_not(np.greater(denominator, 0.0)))
    return cast(NDArrayFloatType, np.nan_to_num(value, copy=False, nan=0.0))


def _lunar_lambert_blend(phase_angle: float) -> float:
    """The McEwen cubic blend L(alpha), clipped to [0, 1]."""
    alpha_deg = abs(math.degrees(phase_angle))
    blend = 1.0 + _LL_A * alpha_deg + _LL_B * alpha_deg**2 + _LL_C * alpha_deg**3
    return float(min(1.0, max(0.0, blend)))
