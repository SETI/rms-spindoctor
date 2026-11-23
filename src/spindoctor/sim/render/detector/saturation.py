"""Saturation: the full well, the column bloom above it, and the serial tail."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_serial_tail', 'apply_saturation']


def apply_saturation(
    electrons: NDArrayFloatType,
    *,
    full_well_e: float,
    bloom_length: int = 0,
) -> None:
    """Cap the electron image at the full well, optionally blooming along columns.

    Charge above ``full_well_e`` spills along the column (the v axis) up to
    ``bloom_length`` pixels each way, conserving the total excess, before every
    pixel is capped at the full well.  A saturated star therefore blooms into a
    vertical streak the way it does on cameras with column bleed, and the capped
    pixels read ``full_well_e / gain`` DN after conversion (below the ADC clip on
    an antiblooming-free camera such as Cassini's NAC).

    Parameters:
        electrons: The electron image, modified in place.
        full_well_e: The full-well ceiling in electrons.
        bloom_length: Column-bloom half-length in pixels; 0 disables bloom.
    """
    if bloom_length > 0:
        excess = np.maximum(electrons - full_well_e, 0.0)
        if float(excess.max()) > 0.0:
            width = 2 * bloom_length + 1
            spread = ndimage.uniform_filter1d(excess, size=width, axis=0, mode='constant')
            np.minimum(electrons, full_well_e, out=electrons)
            electrons += spread
    np.minimum(electrons, full_well_e, out=electrons)


def add_serial_tail(
    dn: NDArrayFloatType,
    *,
    saturation_dn: float,
    saturation_frac: float,
    amplitude_dn: float,
    length_px: int,
    direction: str,
) -> dict[str, Any]:
    """Add a horizontal bright-then-dark serial tail off saturated cores in place.

    An antiblooming CCD shows no column bloom; a hard-saturated compact source
    instead drives an amplifier undershoot along the readout (serial) direction:
    a short bright overshoot immediately after the source, then a longer dark
    undershoot.  Every pixel at or above ``saturation_frac`` of the ADC ceiling
    seeds a tail of ``length_px`` in ``direction``.  A zero amplitude or length,
    or no saturated pixel, is a no-op.

    Parameters:
        dn: The DN image, modified in place.
        saturation_dn: The ADC ceiling (DN).
        saturation_frac: Fraction of the ceiling that counts as saturated.
        amplitude_dn: Peak tail amplitude (DN).
        length_px: Tail length in pixels.
        direction: 'right' (+u readout) or 'left' (-u readout).

    Returns:
        The realized tail summary for the truth record.
    """
    if amplitude_dn <= 0.0 or length_px <= 0:
        return {'active': False, 'sources': 0}
    threshold = saturation_frac * saturation_dn
    vs, us = np.nonzero(dn >= threshold)
    if vs.size == 0:
        return {'active': False, 'sources': 0}
    size_u = dn.shape[1]
    step = 1 if direction == 'right' else -1
    # The pixel just past the source shows a bright overshoot; the pixels beyond
    # it show a dark undershoot decaying back to zero over the tail length.  The
    # profile is sized at least 2 so a 1-px tail (whose loop writes nothing past
    # the source) still builds without indexing past the array.
    profile = np.zeros(max(length_px, 2), dtype=np.float64)
    profile[1] = amplitude_dn
    if length_px > 2:
        decay = np.linspace(1.0, 0.0, length_px - 2, endpoint=False)
        profile[2:] = -amplitude_dn * decay
    for v, u in zip(vs.tolist(), us.tolist(), strict=True):
        for k in range(1, length_px):
            uu = u + step * k
            if 0 <= uu < size_u:
                dn[v, uu] += profile[k]
    return {'active': True, 'sources': int(vs.size), 'direction': direction}
