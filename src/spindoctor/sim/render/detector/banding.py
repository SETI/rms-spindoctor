"""Banding: interference the readout electronics pick up."""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_banding']


def add_banding(
    plane: NDArrayFloatType,
    *,
    amplitude: float,
    period_px: float,
    orientation: str,
    horizontal_pattern: str,
    freq_step_factor: float,
    dark_step_dn: float,
    ringing_amplitude: float,
    ringing_period_px: float,
    ringing_decay_px: float,
    binning: int,
    gain_e_per_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add banding in place: bands along the rows or columns, and column ringing.

    Interference picked up during the readout varies with time, so it varies with the
    position of the pixel being read.  Noise slower than a line gives horizontal
    bands, constant along a row.  With ``horizontal_pattern`` 'periodic' they are a
    sinusoid in the row index with a random phase, plus an independent offset per row
    of 0.3 times the amplitude; with 'random' each row takes its own offset, drawn
    uniformly between minus and plus the amplitude, and the period plays no part.
    Noise in step with the line gives vertical bands, constant down a column: a
    sinusoid in the column index.  ``both`` adds one of each.  A pause halfway down
    the readout multiplies the periodic horizontal bands' spatial frequency by
    ``freq_step_factor`` from that line on, and ``dark_step_dn`` raises the dark level
    there.

    An amplifier that rings at the start of each line adds the same damped wave to
    every row, whatever ``orientation`` says: ``ringing_amplitude`` times
    ``exp(-n / ringing_decay_px)`` times ``sin(2 pi n / ringing_period_px)`` at native
    column ``n``, counted from the frame's first column.  The wave is fixed to the
    line start, so every frame draws it the same.  Its period and decay are in native
    detector columns, and a binned frame column takes the mean over the ``binning``
    native columns it holds.

    A component with a zero amplitude or a zero period or decay is left out, and a
    frame with no component left is a no-op.

    Parameters:
        plane: The image, modified in place: electrons on the CCD chain, DN on the
            vidicon chain.
        amplitude: The bands' amplitude, in the image's units: electrons on the CCD
            chain, DN on the vidicon chain.
        period_px: The periodic bands' spatial period, in frame pixels.
        orientation: 'horizontal', 'vertical', or 'both'.
        horizontal_pattern: 'periodic' or 'random', the horizontal bands' pattern.
        freq_step_factor: Spatial-frequency factor of the periodic horizontal bands
            below the mid-line.
        dark_step_dn: Dark-level step (DN) applied below the mid-line.
        ringing_amplitude: The column ringing's amplitude at the first column, in the
            image's units.
        ringing_period_px: The column ringing's period, in native detector columns.
        ringing_decay_px: The native-column distance over which the column ringing
            falls by a factor of e.
        binning: The readout mode's binning factor: native columns per frame column.
        gain_e_per_dn: Image units per DN, which converts the dark step: the gain on
            the CCD chain, 1 on the vidicon chain.
        rng: The effect's seeded generator.

    Returns:
        The realized banding geometry for the truth record.
    """
    bands = amplitude > 0.0
    horizontal = (
        bands
        and orientation in ('horizontal', 'both')
        and (horizontal_pattern == 'random' or period_px > 0.0)
    )
    vertical = bands and orientation in ('vertical', 'both') and period_px > 0.0
    ringing = ringing_amplitude > 0.0 and ringing_period_px > 0.0 and ringing_decay_px > 0.0
    if not (horizontal or vertical or ringing):
        return {'active': False}
    size_v, size_u = plane.shape
    mid = size_v // 2
    if horizontal:
        if horizontal_pattern == 'random':
            plane += rng.uniform(-amplitude, amplitude, size=(size_v,))[:, None]
        else:
            plane += _periodic_rows(size_v, amplitude, period_px, freq_step_factor, rng)[:, None]
    if vertical:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        cols = np.arange(size_u, dtype=np.float64)
        comb = amplitude * np.sin(2.0 * np.pi * cols / period_px + phase)
        plane += comb[None, :]
    if ringing:
        native = np.arange(size_u * binning, dtype=np.float64)
        wave = (
            ringing_amplitude
            * np.exp(-native / ringing_decay_px)
            * np.sin(2.0 * np.pi * native / ringing_period_px)
        )
        plane += wave.reshape(size_u, binning).mean(axis=1)[None, :]
    if dark_step_dn > 0.0 and gain_e_per_dn > 0.0:
        plane[mid:, :] += dark_step_dn * gain_e_per_dn
    return {
        'active': True,
        'orientation': orientation,
        'horizontal_pattern': horizontal_pattern,
        'period_px': period_px,
        'dark_step_line': mid if dark_step_dn > 0.0 else None,
        'ringing': (
            {
                'period_px': ringing_period_px / binning,
                'decay_px': ringing_decay_px / binning,
            }
            if ringing
            else None
        ),
    }


def _periodic_rows(
    size_v: int,
    amplitude: float,
    period_px: float,
    freq_step_factor: float,
    rng: np.random.Generator,
) -> NDArrayFloatType:
    """The periodic horizontal bands' value on each row.

    Parameters:
        size_v: The number of rows.
        amplitude: The sinusoid's amplitude.
        period_px: The sinusoid's period, in rows.
        freq_step_factor: Spatial-frequency factor below the mid-line.
        rng: The effect's seeded generator.

    Returns:
        One value per row: the sinusoid plus an independent offset per row of 0.3
        times the amplitude.
    """
    freq = 2.0 * np.pi / period_px
    # The phase stays continuous across the frequency step, so the band does
    # not jump there.
    freqs = np.full(size_v, freq, dtype=np.float64)
    if freq_step_factor != 1.0 and freq_step_factor > 0.0:
        freqs[size_v // 2 :] = freq * freq_step_factor
    argument = np.zeros(size_v, dtype=np.float64)
    argument[1:] = np.cumsum(freqs[:-1])
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    band: NDArrayFloatType = amplitude * np.sin(argument + phase)
    band += 0.3 * amplitude * rng.standard_normal(size_v)
    return band
