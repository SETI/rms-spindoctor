"""Effects of how an exposure starts and stops.

A focal-plane shutter exposes some lines longer than others.  A frame-transfer camera,
which has no shutter, keeps collecting light while the image shifts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_frame_transfer_smear', 'apply_shutter_shading']


def apply_shutter_shading(
    signal: NDArrayFloatType,
    *,
    top_factor: float,
    bottom_factor: float,
) -> dict[str, Any]:
    """Scale the light by the shutter's line-dependent exposure in place.

    A focal-plane shutter opens and closes line by line, so the effective
    exposure varies from ``top_factor`` at line 0 to ``bottom_factor`` at the
    last line.  Equal factors are a no-op.

    Parameters:
        signal: The collected light, modified in place.
        top_factor: Exposure multiplier at line 0.
        bottom_factor: Exposure multiplier at the last line.

    Returns:
        The realized gradient for the truth record.
    """
    if top_factor <= 0.0 or bottom_factor <= 0.0 or top_factor == bottom_factor:
        return {'active': False}
    size_v, _size_u = signal.shape
    gradient = np.linspace(top_factor, bottom_factor, size_v) / top_factor
    signal *= gradient[:, None]
    return {'active': True, 'top_factor': top_factor, 'bottom_factor': bottom_factor}


def apply_frame_transfer_smear(
    electrons: NDArrayFloatType,
    *,
    t_scrub_sec: float,
    t_transfer_sec: float,
    exposure_sec: float,
) -> dict[str, Any]:
    """Add the frame-transfer vertical column pedestal (electrons) in place.

    A shutterless frame-transfer CCD keeps integrating while the image shifts
    through the bright rows during the pre-exposure scrub and the post-exposure
    transfer, so every column carries a pedestal proportional to its signal
    integral times ``(t_scrub + t_transfer) / t_exp``.  The pedestal differs on
    the two sides of the column's flux centroid: rows below the centroid in
    image coordinates (larger line numbers) receive the scrub share, rows above
    it receive the transfer share.  The scrub and transfer times are
    independent knobs, so swapping their values swaps the side assignment.  A
    zero transfer time or exposure is a no-op.

    The desmear-residual behavior (the ground pipeline's desmear failing through
    saturated columns and leaving residual banding) is not modeled here; the
    planted pedestal is the raw smear.

    Parameters:
        electrons: The electron image, modified in place.
        t_scrub_sec: Pre-exposure scrub time (seconds).
        t_transfer_sec: Post-exposure transfer time (seconds).
        exposure_sec: The time the frame integrates (seconds): the scene exposure plus
            the camera's extra integration time.

    Returns:
        The realized smear summary for the truth record.
    """
    if exposure_sec <= 0.0 or (t_scrub_sec <= 0.0 and t_transfer_sec <= 0.0):
        return {'active': False}
    size_v, _size_u = electrons.shape
    column_sum = electrons.sum(axis=0)
    ped_scrub = column_sum * t_scrub_sec / (exposure_sec * size_v)
    ped_transfer = column_sum * t_transfer_sec / (exposure_sec * size_v)
    rows = np.arange(size_v, dtype=np.float64)[:, None]
    weights = electrons - electrons.min(axis=0, keepdims=True)
    total = weights.sum(axis=0)
    safe = np.where(total > 0.0, total, 1.0)
    centroid = (rows * weights).sum(axis=0) / safe
    below_centroid = rows > centroid[None, :]
    electrons += np.where(below_centroid, ped_scrub[None, :], ped_transfer[None, :])
    return {
        'active': True,
        'mean_pedestal_e': float((ped_scrub + ped_transfer).mean() * 0.5),
    }
