"""Photometric surface-scattering laws for the image-side body renderer.

The renderer shades body discs with a scene-selected law --
Lambert, Lommel-Seeliger, Minnaert(k), or a lunar-Lambert blend -- plus an
optional opposition-surge factor.  A body that leaves the law out shades with
:data:`PHOTOMETRIC_LAW_DEFAULT`, lunar-Lambert with McEwen's blend.  Real Cassini
frames of Saturn's icy moons follow that law: over 27 discs at phases from 1.5 to 136
degrees, the blend fitted to each real disc differs from McEwen's curve by an rms of
0.15 and a mean under 0.001, and the law reproduces both the brightness across the disc
and the position of the limb's brightness gradient peak.  A Lambert disc is too dark
toward the sunlit limb at low phase: on discs under 100 px across below 60 degrees, its
gradient peak sits 0.53 px inside the silhouette, where the real limbs' sits 0.06 px
inside and the default law's 0.01 px.

The law is a truth key: the navigator's predicted-body model always shades with
Lambert (:func:`spindoctor.sim.shared.ellipsoid.lambert_from_normals`), so a scene
that renders any other law, the default included, plants a photometric mismatch that
moves the terminator and the limb-darkening profile by a known amount.  Above about
104 degrees phase McEwen's blend reaches 0 and the default law is Lambert, so there
the default plants no mismatch.

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
  alpha = 0, pure Lambert once the cubic reaches 0 (~104 deg).

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

from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'MINNAERT_CEILING',
    'PHOTOMETRIC_LAWS',
    'PHOTOMETRIC_LAW_DEFAULT',
    'incidence_cosines',
    'lunar_lambert_blend',
    'shade_surface',
    'surge_factor',
]

# The scene-selectable law vocabulary (schema-validated).
PHOTOMETRIC_LAWS: frozenset[str] = frozenset(
    {'lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert'}
)

PHOTOMETRIC_LAW_DEFAULT = 'lunar_lambert'
"""The law a body shades with when its scene entry leaves ``photometric_law`` out."""

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
    return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + normal_z * illum_z)


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
    law: str = PHOTOMETRIC_LAW_DEFAULT,
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

    visible = cos_emission > 0.0
    mu0 = np.clip(cos_incidence, 0.0, 1.0)
    mu = np.clip(cos_emission, 0.0, 1.0)

    if law == 'lambert':
        intensity = mu0
    elif law == 'lommel_seeliger':
        intensity = _lommel_seeliger(mu0, mu)
    elif law == 'minnaert':
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            intensity = np.where(mu > 0.0, mu0**minnaert_k * mu ** (minnaert_k - 1.0), 0.0)
        intensity = np.minimum(np.nan_to_num(intensity, nan=0.0), MINNAERT_CEILING)
    else:  # lunar_lambert
        blend = lunar_lambert_blend(phase_angle)
        intensity = blend * _lommel_seeliger(mu0, mu) + (1.0 - blend) * mu0

    intensity = intensity * surge_factor(
        phase_angle, amplitude=surge_amplitude, width_deg=surge_width_deg
    )
    return cast(NDArrayFloatType, np.where(visible, intensity, 0.0))


def _lommel_seeliger(mu0: NDArrayFloatType, mu: NDArrayFloatType) -> NDArrayFloatType:
    """The Lommel-Seeliger term ``2 * mu0 / (mu0 + mu)``, 0 where degenerate."""
    denominator = mu0 + mu
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.where(denominator > 0.0, 2.0 * mu0 / denominator, 0.0)
    return cast(NDArrayFloatType, np.nan_to_num(value, nan=0.0))


def lunar_lambert_blend(phase_angle: float) -> float:
    """McEwen's blend L(alpha) at a phase angle, clipped to [0, 1].

    Parameters:
        phase_angle: Phase angle in radians.

    Returns:
        The Lommel-Seeliger share of the lunar-Lambert law at that phase.
    """
    alpha_deg = abs(math.degrees(phase_angle))
    blend = 1.0 + _LL_A * alpha_deg + _LL_B * alpha_deg**2 + _LL_C * alpha_deg**3
    return float(min(1.0, max(0.0, blend)))
