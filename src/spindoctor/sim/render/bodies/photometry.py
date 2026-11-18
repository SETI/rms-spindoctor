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
- ``lommel_seeliger``: ``I = 2 * mu0 / (mu0 + mu)`` (reaches 2 toward the
  limb; the [dark floor, 1] clip below caps it at the signal ceiling).
- ``minnaert``: ``I = mu0**k * mu**(k - 1)``; k = 1 is Lambert, k = 0.5
  is the classic limb-brightened lunar value.  The disc-center
  normalization holds; near the limb the law diverges and is capped by the
  [dark floor, 1] clip below.
- ``lunar_lambert``: ``I = 2 * L(alpha) * mu0 / (mu0 + mu)
  + (1 - L(alpha)) * mu0`` with the McEwen (1991) cubic blend
  ``L(alpha) = 1 - 0.019 * a + 2.42e-4 * a**2 - 1.46e-6 * a**3``
  (a = phase in degrees), clipped to [0, 1]: pure Lommel-Seeliger at
  alpha = 0, pure Lambert once the cubic reaches 0 (~119 deg).

The opposition surge is a simple normalized exponential factor
``(1 + amplitude * exp(-alpha / width)) / (1 + amplitude)``: an interim
knob for the realism match, planting the surge's brightness-versus-phase
signature while keeping the normalized signal plane within [0, 1] (a scene
at large phase renders darker by ``1 / (1 + amplitude)`` rather than the
opposition frame clipping at the full scale).

The dark-side floor keeps today's semantics: :func:`shade_surface` clips
every visible-hemisphere pixel to [0.01, 1] (none of the supported laws
defines a dark-side term, so geometric night renders at the floor), and
the far hemisphere is 0.  The floor is a guarantee on the *shading stage's
output only*: later multiplicative stages -- a transit's cast shadow in
particular -- may darken lit-disc pixels below it.
"""

import math
from typing import cast

import numpy as np

from spindoctor.sim.shared.ellipsoid import illumination_vector
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DARK_SIDE_FLOOR',
    'PHOTOMETRIC_LAWS',
    'incidence_cosines',
    'shade_surface',
    'surge_factor',
]

# The scene-selectable law vocabulary (schema-validated).
PHOTOMETRIC_LAWS: frozenset[str] = frozenset(
    {'lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert'}
)

# Visible-hemisphere intensity floor, matching the Lambert renderer's
# dark-side clamp: geometric night and terminator-march shadow render at
# this level in the shading stage.  Downstream multiplicative stages
# (transit shadows) may push lit-disc pixels below it.
DARK_SIDE_FLOOR = 0.01

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
        Brightness in [0, 1]: the law value times the surge factor, clipped
        to [:data:`DARK_SIDE_FLOOR`, 1] on the visible hemisphere, 0 off it.

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
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=np.inf)
    else:  # lunar_lambert
        blend = _lunar_lambert_blend(phase_angle)
        intensity = blend * _lommel_seeliger(mu0, mu) + (1.0 - blend) * mu0

    intensity = intensity * surge_factor(
        phase_angle, amplitude=surge_amplitude, width_deg=surge_width_deg
    )
    return cast(
        NDArrayFloatType,
        np.where(visible, np.clip(intensity, DARK_SIDE_FLOOR, 1.0), 0.0),
    )


def _lommel_seeliger(mu0: NDArrayFloatType, mu: NDArrayFloatType) -> NDArrayFloatType:
    """The Lommel-Seeliger term ``2 * mu0 / (mu0 + mu)``, 0 where degenerate."""
    denominator = mu0 + mu
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.where(denominator > 0.0, 2.0 * mu0 / denominator, 0.0)
    return cast(NDArrayFloatType, np.nan_to_num(value, nan=0.0))


def _lunar_lambert_blend(phase_angle: float) -> float:
    """The McEwen cubic blend L(alpha), clipped to [0, 1]."""
    alpha_deg = abs(math.degrees(phase_angle))
    blend = 1.0 + _LL_A * alpha_deg + _LL_B * alpha_deg**2 + _LL_C * alpha_deg**3
    return float(min(1.0, max(0.0, blend)))
