"""Telemetry-stage artifact modes beyond the structured loss modes.

Two families live here: the lossy DCT compression blockiness a codec plants on
the transmitted signal before any packet loss, and the Voyager GEOMED
archive-processing scars (reseau-removal smudges and the resample texture) the
ground pipeline leaves in the products the navigator consumes.  Each renders one
registry mode onto a detector-grid DN image in place and returns its realized
geometry for the frame's truth record; each is a no-op when its per-frame
``probability`` draw does not fire (the stage-activation rule) and draws from
its own seeded generator.

Order within the telemetry stage is physical: compression runs before the
structured loss (a lossy codec compresses, then packets drop), and the GEOMED
scars run after it (they emulate archive processing applied to the
already-loss-bearing raw frame).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn

from spindoctor.sim.forward.detector.params import CalibrationMap
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'apply_compression_dct',
    'apply_resample_texture',
    'apply_reseau_scars',
]

# The Voyager reseau grid carries 202 marks per camera; the lattice is thinned
# uniformly to that many points however large the frame, so the marks span the
# whole frame rather than crowding its top rows.
_RESEAU_MARK_COUNT = 202


def _activated(rng: np.random.Generator, probability: float) -> bool:
    """Whether a commanded / periodic telemetry mode fires this frame, with ``probability``."""
    if probability <= 0.0:
        return False
    return bool(rng.random() < probability)


def apply_compression_dct(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
    protect: tuple[int, int, int, int] | None,
    calibration: CalibrationMap | None = None,
) -> dict[str, Any]:
    """Quantize the frame's 8x8 DCT coefficients, planting blockiness and ringing.

    The lossy downlink codec (Galileo ICT, Cassini lossy, LORRI lossy) transforms
    each ``block`` x ``block`` tile to the DCT domain, quantizes the coefficients
    to a step of ``scale_factor``, and inverts -- coarser steps give visible
    block edges and ringing around high-contrast features.  Any commanded truth
    window (``protect``) is restored clean afterward, the losslessly-coded
    carve-out.  Frame rows/columns past the last whole block are left unchanged.
    A mode that does not fire is a no-op.

    The codec runs on board, on the DN the spacecraft transmits.  On a
    calibrated frame ``calibration`` maps the I/F image back to those DN, the
    coefficients are quantized there, and the result maps back to I/F, so the
    step is ``scale_factor`` DN on both kinds of frame and the pedestal the
    calibration subtracts takes part in the quantization as it does on board.

    Parameters:
        signal: The image, modified in place: DN, or I/F when ``calibration`` is
            given.
        cfg: The resolved mode config (``probability``, ``scale_factor``, ``block``).
        rng: The mode's seeded generator (for the activation draw).
        protect: The commanded truth-window rectangle to leave clean, or None.
        calibration: The calibrated frame's DN-to-I/F map, or None for a raw-DN
            frame.

    Returns:
        The realized compression summary for the truth record.
    """
    probability = float(cfg['probability'])
    if not _activated(rng, probability):
        return {'active': False}
    scale = float(cfg['scale_factor'])
    block = int(cfg['block'])
    if scale <= 0.0 or block <= 0:
        return {'active': False}
    size_v, size_u = signal.shape
    n_bv, n_bu = size_v // block, size_u // block
    if n_bv == 0 or n_bu == 0:
        return {'active': False}
    saved = None
    if protect is not None:
        pv0, pv1, pu0, pu1 = protect
        saved = signal[pv0:pv1, pu0:pu1].copy()
    region = signal[: n_bv * block, : n_bu * block]
    dn = region if calibration is None else calibration.to_dn(region)
    blocks = dn.reshape(n_bv, block, n_bu, block)
    coeffs = dctn(blocks, axes=(1, 3), norm='ortho')
    quantized = np.round(coeffs / scale) * scale
    restored = idctn(quantized, axes=(1, 3), norm='ortho').reshape(region.shape)
    region[:] = restored if calibration is None else calibration.to_if(restored)
    if protect is not None and saved is not None:
        pv0, pv1, pu0, pu1 = protect
        signal[pv0:pv1, pu0:pu1] = saved
    return {'active': True, 'scale_factor': scale, 'block': block}


def _reseau_lattice(size_v: int, size_u: int, spacing: int) -> list[tuple[int, int]]:
    """The triangular reseau lattice points, thinned uniformly to the 202-mark count.

    The full triangular lattice at ``spacing`` is generated first; when it holds
    more than the mark count, every-kth subsampling across the row-major
    ordering keeps exactly 202 marks spread over the whole frame (a truncation
    would instead crowd the marks into the top rows at archive frame sizes).
    """
    dv = max(1, round(spacing * np.sqrt(3.0) / 2.0))
    points: list[tuple[int, int]] = []
    for i, v in enumerate(range(0, size_v, dv)):
        offset = (spacing // 2) if (i % 2) else 0
        for u in range(offset, size_u, spacing):
            points.append((v, u))
    if len(points) <= _RESEAU_MARK_COUNT:
        return points
    indices = np.linspace(0, len(points) - 1, _RESEAU_MARK_COUNT).round().astype(int)
    return [points[i] for i in indices]


def apply_reseau_scars(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Smooth small patches on the reseau lattice, the reseau-removal scars.

    The archive removes each reseau mark by interpolating over it, leaving an
    anomalously smooth patch on the triangular lattice.  This blends the frame
    toward its locally-smoothed version inside a ``patch_radius_px`` disc at each
    lattice point, so a mark that sat on a limb or ring edge shows a smooth patch
    there.  A mode that does not fire is a no-op.

    Parameters:
        signal: The DN image, modified in place.
        cfg: The resolved mode config (``probability``, ``spacing_px``,
            ``patch_radius_px``).
        rng: The mode's seeded generator (for the activation draw).

    Returns:
        The realized scar geometry for the truth record.
    """
    probability = float(cfg['probability'])
    if not _activated(rng, probability):
        return {'active': False, 'marks': 0}
    spacing = int(cfg['spacing_px'])
    radius = int(cfg['patch_radius_px'])
    if spacing <= 0 or radius <= 0:
        return {'active': False, 'marks': 0}
    size_v, size_u = signal.shape
    smoothed = ndimage.gaussian_filter(signal, sigma=max(1.0, radius / 2.0))
    points = _reseau_lattice(size_v, size_u, spacing)
    for v, u in points:
        v0, v1 = max(0, v - radius), min(size_v, v + radius + 1)
        u0, u1 = max(0, u - radius), min(size_u, u + radius + 1)
        signal[v0:v1, u0:u1] = smoothed[v0:v1, u0:u1]
    return {'active': True, 'marks': len(points), 'spacing_px': spacing}


def apply_resample_texture(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Resample the frame with a subpixel warp, emulating GEOMED archive texture.

    The GEOMED geometric resample softens noise and edges and correlates the
    noise spatially the way the archive product's does.  This applies a smooth
    seeded subpixel displacement field of amplitude ``warp_amp_px`` (bilinear
    resample), optionally blanks an irregular ``blank_border_px`` border, and
    optionally replaces alternate lines by the mean of their neighbors
    (``missing_line_interp``, the interpolate-across-lines banding).  A mode that
    does not fire is a no-op.

    Parameters:
        signal: The DN image, modified in place.
        cfg: The resolved mode config (``probability``, ``warp_amp_px``,
            ``blank_border_px``, ``missing_line_interp``).
        rng: The mode's seeded generator.

    Returns:
        The realized resample summary for the truth record.
    """
    probability = float(cfg['probability'])
    if not _activated(rng, probability):
        return {'active': False}
    warp_amp = float(cfg['warp_amp_px'])
    border = int(cfg['blank_border_px'])
    interp_lines = bool(cfg['missing_line_interp'])
    size_v, size_u = signal.shape
    if warp_amp > 0.0:
        coarse_v = rng.normal(0.0, 1.0, size=(max(2, size_v // 16), max(2, size_u // 16)))
        coarse_u = rng.normal(0.0, 1.0, size=coarse_v.shape)
        zoom_v = size_v / coarse_v.shape[0]
        zoom_u = size_u / coarse_v.shape[1]
        disp_v = ndimage.zoom(coarse_v, (zoom_v, zoom_u), order=1)[:size_v, :size_u] * warp_amp
        disp_u = ndimage.zoom(coarse_u, (zoom_v, zoom_u), order=1)[:size_v, :size_u] * warp_amp
        vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
        signal[:] = ndimage.map_coordinates(
            signal, [vv + disp_v, uu + disp_u], order=1, mode='nearest'
        )
    if interp_lines:
        interp = signal.copy()
        interp[1:-1:2, :] = 0.5 * (signal[0:-2:2, :] + signal[2::2, :])
        signal[:] = interp
    if border > 0:
        signal[:border, :] = 0.0
        signal[-border:, :] = 0.0
        signal[:, :border] = 0.0
        signal[:, -border:] = 0.0
    return {
        'active': True,
        'warp_amp_px': warp_amp,
        'blank_border_px': border,
        'missing_line_interp': interp_lines,
    }
