"""Banding: coherent interference the readout electronics pick up."""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_banding']


def add_banding(
    plane: NDArrayFloatType,
    *,
    amplitude_e: float,
    period_px: float,
    orientation: str,
    freq_step_factor: float,
    dark_step_dn: float,
    gain_e_per_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add coherent banding in place, horizontal, vertical, or both.

    Interference picked up during the readout varies with time, so it varies with the
    position of the pixel being read.  Noise slower than a line gives horizontal
    bands, constant along a row: a sinusoid in the row index with a random phase, plus
    an independent offset per row of 0.3 times the amplitude.  Noise in step with the
    line gives vertical bands, constant down a column: a sinusoid in the column index.
    ``both`` adds one of each.  A pause halfway down the readout multiplies the
    horizontal bands' spatial frequency by ``freq_step_factor`` from that line on, and
    ``dark_step_dn`` raises the dark level there.  A zero amplitude or period is a
    no-op.

    Parameters:
        plane: The image, modified in place: electrons on the CCD chain, DN on the
            vidicon chain.
        amplitude_e: The bands' amplitude, in the image's units.
        period_px: The bands' spatial period, in pixels.
        orientation: 'horizontal', 'vertical', or 'both'.
        freq_step_factor: Spatial-frequency factor of the horizontal bands below the
            mid-line.
        dark_step_dn: Dark-level step (DN) applied below the mid-line.
        gain_e_per_dn: Image units per DN, which converts the dark step: the gain on
            the CCD chain, 1 on the vidicon chain.
        rng: The effect's seeded generator.

    Returns:
        The realized banding geometry for the truth record.
    """
    if amplitude_e <= 0.0 or period_px <= 0.0:
        return {'active': False}
    size_v, size_u = plane.shape
    mid = size_v // 2
    freq = 2.0 * np.pi / period_px
    if orientation in ('horizontal', 'both'):
        # The phase stays continuous across the frequency step, so the band does
        # not jump there.
        freqs = np.full(size_v, freq, dtype=np.float64)
        if freq_step_factor != 1.0 and freq_step_factor > 0.0:
            freqs[mid:] = freq * freq_step_factor
        argument = np.zeros(size_v, dtype=np.float64)
        argument[1:] = np.cumsum(freqs[:-1])
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        band = amplitude_e * np.sin(argument + phase)
        band += 0.3 * amplitude_e * rng.standard_normal(size_v)
        plane += band[:, None]
    if orientation in ('vertical', 'both'):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        cols = np.arange(size_u, dtype=np.float64)
        comb = amplitude_e * np.sin(2.0 * np.pi * cols / period_px + phase)
        plane += comb[None, :]
    if dark_step_dn > 0.0 and gain_e_per_dn > 0.0:
        plane[mid:, :] += dark_step_dn * gain_e_per_dn
    return {
        'active': True,
        'orientation': orientation,
        'period_px': period_px,
        'dark_step_line': mid if dark_step_dn > 0.0 else None,
    }
