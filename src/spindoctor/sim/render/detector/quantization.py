"""The quantization sub-modes: how the detector's DN become the frame's DN.

A scene selects one with ``detector.quantization``, and its camera's instrument entry
supplies the default.  ``sqrt_lut`` decodes through one of the camera's companding
tables (:mod:`~spindoctor.sim.render.detector.sqrt_lut`).
"""

from __future__ import annotations

import numpy as np

from spindoctor.sim.render.detector.sqrt_lut import DownlinkCurve, resolve_sqrt_lut
from spindoctor.sim.scene.truth_detector import DEFAULT_CONTOUR_STEP
from spindoctor.support.types import NDArrayFloatType

__all__ = ['quantize_dn']


def quantize_dn(
    dn: NDArrayFloatType,
    *,
    mode: str,
    saturation_dn: float,
    contour_step: int = DEFAULT_CONTOUR_STEP,
    sqrt_lut: DownlinkCurve | None = None,
) -> NDArrayFloatType:
    """Quantize a DN image by the selected ADC sub-mode.

    Parameters:
        dn: The DN image (float).
        mode: 'exact' (round to integer, uniform bins), '8bit' (integer bins
            with a hard 255 code ceiling), 'uneven_12bit' (integer bins with
            histogram spikes at the power-of-two bit boundaries), 'sqrt_lut'
            (square-root companding to 8 bits and back: each value snapped to
            the nearest decoded DN of ``sqrt_lut``, leaving a
            signal-dependent residual), 'ls8b' (low 8 bits kept: values above
            255 wrap modulo 256, the banded wraparound on bright targets), or
            'contour_8bit' (8-bit output posterized to multiples of
            ``contour_step``, the uneven-bin contouring of a coarse ADC).
        saturation_dn: The ADC ceiling, which scales the plain square-root
            curve when ``sqrt_lut`` is None.
        contour_step: The DN posterization step for the 'contour_8bit' mode.
        sqrt_lut: The companding curve for the 'sqrt_lut' mode; None selects
            the plain square-root curve whose top code decodes to
            ``saturation_dn``.

    Returns:
        The quantized DN image.

    Raises:
        ValueError: If ``mode`` is not a known quantization sub-mode.
    """
    if mode == 'exact':
        return np.rint(dn)
    if mode == 'ls8b':
        # LS8B telemetry mode: only the low 8 bits are kept, so a value above
        # 255 wraps modulo 256 -- the banded wraparound a bright target shows.
        return np.mod(np.rint(dn), 256.0)
    if mode == 'contour_8bit':
        # A coarse ADC with uneven bins is worst at DN multiples of the step:
        # posterizing an 8-bit word to multiples of contour_step reproduces the
        # visible contouring of a smooth low-contrast scene.
        clipped = np.clip(np.rint(dn), 0.0, 255.0)
        return np.clip(np.round(clipped / contour_step) * contour_step, 0.0, 255.0)
    if mode == '8bit':
        # An 8-bit output word: integer DN clipped at the 255 code ceiling.
        # The ceiling is the word width, not the scene's saturation_dn, so an
        # 8-bit mode on a deeper detector still tops out at 255.
        return np.clip(np.rint(dn), 0.0, 255.0)
    if mode == 'uneven_12bit':
        # Uneven bit weights concentrate codes at the power-of-two carry
        # boundaries: values within one DN of a 2^m boundary snap to it, which
        # is the histogram-spike signature of an ADC with unequal bit weights.
        quantized = np.rint(dn)
        max_bit = int(np.log2(max(saturation_dn, 2.0))) + 1
        for m in range(1, max_bit):
            boundary = float(2**m)
            near = np.abs(quantized - boundary) <= 1.0
            quantized[near] = boundary
        return quantized
    if mode == 'sqrt_lut':
        # Square-root companding: the table encodes each DN to the code whose
        # decoded value is nearest, and the product carries the decoded DN, so
        # the step tracks the photon noise and leaves a residual that grows with
        # signal.  A NaN pixel stays NaN, as np.rint keeps it in the other modes.
        if sqrt_lut is None:
            sqrt_lut = resolve_sqrt_lut(None, saturation_dn=saturation_dn)
        levels = sqrt_lut.levels()
        midpoints = 0.5 * (levels[1:] + levels[:-1])
        snapped = levels[np.searchsorted(midpoints, np.nan_to_num(dn, nan=0.0))]
        result: NDArrayFloatType = np.where(np.isnan(dn), np.nan, snapped)
        return result
    raise ValueError(f'unknown quantization mode {mode!r}')
