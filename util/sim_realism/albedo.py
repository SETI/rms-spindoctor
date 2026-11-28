"""A body's normal albedo, measured from the I/F of its disc in a real frame.

The simulator draws a body's surface as its normal albedo times its photometric law,
which is 1 where the surface is lit and viewed head-on at zero phase.  A matched
simulated frame takes the real body's brightness from the real frame: each disc pixel's
I/F divided by the law's value at that pixel's lighting and viewing angles is one
estimate of the normal albedo, and :func:`disc_normal_albedo` returns their median.

Only pixels where the Sun and the camera are both within about 73 degrees of overhead
count (both cosines at least 0.3).  Near the terminator and the limb the law's value is
small, so the division magnifies the noise and any error in the pixel's angles.  The
caller leaves out pixels near the edge of the disc, where the optics blur the disc into
the sky.
"""

from __future__ import annotations

import numpy as np

from spindoctor.sim.render.bodies.photometry import shade_surface
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'MIN_ALBEDO_PIXELS',
    'MIN_DISC_COSINE',
    'disc_normal_albedo',
]

MIN_DISC_COSINE = 0.3
"""The smallest cosine of incidence or emission a disc pixel may have and still count."""

MIN_ALBEDO_PIXELS = 50
"""The fewest disc pixels that give a normal albedo."""


def disc_normal_albedo(
    disc_if: NDArrayFloatType,
    cos_incidence: NDArrayFloatType,
    cos_emission: NDArrayFloatType,
    *,
    law: str,
    phase_angle: float,
    minnaert_k: float = 0.5,
) -> float | None:
    """The median normal albedo of a body's disc pixels under a photometric law.

    Parameters:
        disc_if: The I/F of each disc pixel, any shape.
        cos_incidence: The cosine of the solar incidence angle at each pixel.
        cos_emission: The cosine of the emission angle at each pixel.
        law: The photometric law the albedo is measured against, one of
            :data:`~spindoctor.sim.render.bodies.photometry.PHOTOMETRIC_LAWS`.
        phase_angle: The phase angle in radians, which the lunar-Lambert law reads.
        minnaert_k: The Minnaert exponent, which the Minnaert law reads.

    Returns:
        The median over the pixels whose I/F is finite, whose two cosines are both at
        least 0.3, and where the law's value is positive and the I/F divided by it is
        finite, of that quotient.  None when fewer than 50 pixels count or the median
        is not positive.
    """
    values = np.asarray(disc_if, dtype=np.float64).ravel()
    mu0 = np.asarray(cos_incidence, dtype=np.float64).ravel()
    mu = np.asarray(cos_emission, dtype=np.float64).ravel()
    keep = np.isfinite(values) & (mu0 >= MIN_DISC_COSINE) & (mu >= MIN_DISC_COSINE)
    shaded = np.zeros(values.shape)
    shaded[keep] = shade_surface(
        mu0[keep], mu[keep], law=law, phase_angle=phase_angle, minnaert_k=minnaert_k
    )
    # A law value that underflows to zero (a large Minnaert exponent) does not give an
    # estimate, and neither does one so small that the division overflows.
    keep &= np.isfinite(shaded) & (shaded > 0.0)
    with np.errstate(over='ignore'):
        ratios = values[keep] / shaded[keep]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < MIN_ALBEDO_PIXELS:
        return None
    albedo = float(np.median(ratios))
    if not albedo > 0.0:
        return None
    return albedo
