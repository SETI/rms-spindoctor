"""Shared helpers for the ``spindoctor.support`` tests."""

import numpy as np


def gaussian_patch(
    shape: tuple[int, int],
    sigma: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a 2-D Gaussian patch with an optional sub-pixel offset.

    Parameters:
        shape: ``(v_size, u_size)`` of the patch.
        sigma: Gaussian sigma in pixels.
        offset: ``(dv, du)`` displacement of the peak from the patch center.

    Returns:
        The patch, peaking at 1.0 at its (offset) center.
    """
    v_size, u_size = shape
    cv = (v_size - 1) / 2.0
    cu = (u_size - 1) / 2.0
    vv, uu = np.meshgrid(np.arange(v_size), np.arange(u_size), indexing='ij')
    dv = vv - (cv + offset[0])
    du = uu - (cu + offset[1])
    return np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))
