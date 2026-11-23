"""The fixed pattern: the static response and offset structure of the detector.

The response (photo-response non-uniformity, vignetting, and dust) multiplies the
light before the shot noise.  The offsets (stitch combs and jail bars) ride on the DN.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_fixed_pattern_dn', 'add_fixed_pattern_response']


def add_fixed_pattern_response(
    electrons: NDArrayFloatType,
    *,
    prnu_rms: float,
    vignetting_frac: float,
    dust_donut_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Multiply the electron plane by the static per-pixel response in place.

    The multiplicative fixed pattern is a per-pixel response applied before
    Poisson: photo-response non-uniformity (a per-pixel gain jitter of
    ``prnu_rms``), corner vignetting (a radial falloff reaching ``vignetting_frac``
    at the corners), and ``dust_donut_count`` faint ring-shaped shadows.  The
    pattern is drawn from the mode's seeded stream, so it is the same every
    render of the scene.  All-zero parameters are a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        prnu_rms: RMS of the per-pixel response jitter.
        vignetting_frac: Corner response deficit (0 = none).
        dust_donut_count: Number of dust-donut shadow rings.
        rng: The mode's seeded generator.

    Returns:
        The realized response summary for the truth record.
    """
    if prnu_rms <= 0.0 and vignetting_frac <= 0.0 and dust_donut_count <= 0:
        return {'active': False}
    size_v, size_u = electrons.shape
    response = np.ones((size_v, size_u), dtype=np.float64)
    if prnu_rms > 0.0:
        response += rng.normal(0.0, prnu_rms, size=(size_v, size_u))
    if vignetting_frac > 0.0:
        vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
        cv, cu = (size_v - 1) / 2.0, (size_u - 1) / 2.0
        rho2 = ((vv - cv) / cv) ** 2 + ((uu - cu) / cu) ** 2
        response *= 1.0 - vignetting_frac * (rho2 / 2.0)
    donuts = 0
    for _ in range(max(0, dust_donut_count)):
        dv = int(rng.integers(0, size_v))
        du = int(rng.integers(0, size_u))
        radius = float(rng.uniform(3.0, 8.0))
        vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
        ring = np.exp(-(((stable_hypot(vv - dv, uu - du) - radius) / 1.5) ** 2))
        response *= 1.0 - 0.05 * ring
        donuts += 1
    np.maximum(response, 0.0, out=response)
    electrons *= response
    return {'active': True, 'prnu_rms': prnu_rms, 'dust_donuts': donuts}


def add_fixed_pattern_dn(
    dn: NDArrayFloatType,
    *,
    stitch_period_px: int,
    stitch_amplitude_dn: float,
    jail_bar_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add the static additive DN fixed pattern in place (combs and jail bars).

    The photolithography stitch comb is a set of bright columns every
    ``stitch_period_px`` pixels raised by ``stitch_amplitude_dn``; the jail bars
    are an even/odd column offset of ``jail_bar_dn`` whose sign is drawn once per
    seed (a power-cycle-dependent bias).  All-zero parameters are a no-op.

    Parameters:
        dn: The DN image, modified in place.
        stitch_period_px: Column period of the stitch comb (0 disables).
        stitch_amplitude_dn: Stitch-comb column amplitude (DN).
        jail_bar_dn: Even/odd column offset amplitude (DN).
        rng: The mode's seeded generator.

    Returns:
        The realized pattern summary for the truth record.
    """
    if (stitch_period_px <= 0 or stitch_amplitude_dn <= 0.0) and jail_bar_dn <= 0.0:
        return {'active': False}
    _size_v, size_u = dn.shape
    stitched: list[int] = []
    if stitch_period_px > 0 and stitch_amplitude_dn > 0.0:
        cols = np.arange(0, size_u, stitch_period_px)
        dn[:, cols] += stitch_amplitude_dn
        stitched = [int(c) for c in cols]
    if jail_bar_dn > 0.0:
        sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
        parity = (np.arange(size_u) % 2).astype(np.float64) * 2.0 - 1.0
        dn += (sign * jail_bar_dn * parity)[None, :]
    return {'active': True, 'stitch_columns': stitched, 'jail_bar_dn': jail_bar_dn}
