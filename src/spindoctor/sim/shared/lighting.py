"""Shared light direction and phase function for the simulator.

Every simulated body is lit by one light direction, derived here from the scene's
illumination and phase angles.  Both sides of the information boundary use it: the
image side's ellipsoid, mesh, and haze shading and the navigator side's predicted
bodies, so the rendered and the predicted bodies follow one illumination
convention.

The rings and the haze, both on the image side, use the one Henyey-Greenstein phase
function here, so a scattering asymmetry ``g`` means the same thing in both.

Every function takes explicit angles in radians and never reads a scene mapping.
"""

import math

import numpy as np

__all__ = [
    'henyey_greenstein_phase',
    'illumination_vector',
    'in_plane_light_direction',
]


def in_plane_light_direction(illumination_angle: float) -> tuple[float, float]:
    """The unit in-plane direction toward the light in image coordinates.

    Parameters:
        illumination_angle: In-plane light direction in radians; 0 is from the top
            of the image, pi/2 from the right.

    Returns:
        Tuple of (v, u) components.  The v component is negated because v
        increases downward.
    """
    return -math.cos(illumination_angle), math.sin(illumination_angle)


def illumination_vector(
    *, illumination_angle: float, phase_angle: float
) -> tuple[float, float, float]:
    """The unit body-to-sun direction in image coordinates.

    The in-plane direction comes from ``illumination_angle`` through
    :func:`in_plane_light_direction`.  The out-of-plane component encodes the
    phase angle -- the observer-body-sun angle: ``z = cos(phase_angle)`` so
    phase 0 (full) lights the visible face head-on and phase pi (new) lights
    it from behind, while the in-plane magnitude is ``sin(phase_angle)``.

    Parameters:
        illumination_angle: In-plane light direction in radians; 0 is from
            the top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Tuple of (v, u, z) components of the unit illumination direction;
        z points toward the observer.
    """
    illum_v_2d, illum_u_2d = in_plane_light_direction(illumination_angle)

    illum_z = np.cos(phase_angle)
    illum_scale_2d = np.sin(phase_angle)
    illum_v_3d = illum_v_2d * illum_scale_2d
    illum_u_3d = illum_u_2d * illum_scale_2d

    # Normalize the 3D illumination direction (already unit up to rounding;
    # the guard covers a degenerate zero vector only).
    illum_mag = np.sqrt(illum_v_3d**2 + illum_u_3d**2 + illum_z**2)
    if illum_mag > 1e-10:
        illum_v_3d /= illum_mag
        illum_u_3d /= illum_mag
        illum_z_norm = illum_z / illum_mag
    else:
        illum_z_norm = 1.0  # Directly toward observer
    return float(illum_v_3d), float(illum_u_3d), float(illum_z_norm)


def henyey_greenstein_phase(
    g: float, phase_angle: float, *, denominator_floor: float = 0.0
) -> float:
    """The one-term Henyey-Greenstein phase function at a phase angle.

    ``P = (1 - g**2) / (1 + g**2 + 2*g*cos(phase))**1.5`` with the phase angle 0 at
    opposition.  Single scattering turns light through ``pi - phase``, so a negative
    ``g`` backscatters (bright at low phase) and a positive ``g`` forward-scatters
    (bright toward phase pi).  ``P`` is 1 at every phase when ``g`` is 0, so it is a
    pure angular modulation of the scattered light.

    Parameters:
        g: Asymmetry parameter, strictly inside (-1, 1).
        phase_angle: Phase angle in radians, [0, pi].
        denominator_floor: Smallest value the denominator may take.  With the
            default 0 the function grows without bound as ``|g|`` approaches 1 and
            divides by zero once the denominator vanishes, which happens at phase pi
            for ``g`` near 1 and at phase 0 for ``g`` near -1.

    Returns:
        The phase function value (positive).
    """
    cos_phase = math.cos(phase_angle)
    denominator = (1.0 + g * g + 2.0 * g * cos_phase) ** 1.5
    return float((1.0 - g * g) / max(denominator, denominator_floor))
