"""Sub-pixel refinement of a correlation peak by a quadratic fit to the NCC surface.

:func:`~spindoctor.support.correlate.evaluate_candidate` turns a whole-pixel correlation
peak into a sub-pixel offset in one of two ways, named by :data:`SubpixelRefinement`:

- ``'cross_power'`` interpolates the raw cross-power of image and model with a localized
  upsampled DFT, within half a pixel of the whole-pixel peak, on a grid of
  ``1 / upsample_factor`` px.
- ``'ncc_quadratic'`` fits a quadratic surface to the 3x3 block of NCC values centered on
  the whole-pixel peak and moves to the fitted surface's maximum.  This module holds that
  fit.

The two answers differ when the raw cross-power and the NCC peak in different places.  The
NCC normalizes away the image variance under the shifted mask; the raw cross-power does
not, so its maximum can lie more than half a pixel from the NCC peak, and the cross-power
answer then stops at the edge of its half-pixel window.
"""

from typing import Literal, TypeAlias

import numpy as np

from spindoctor.support.types import NDArrayFloatType

SubpixelRefinement: TypeAlias = Literal['cross_power', 'ncc_quadratic']
"""The name of a sub-pixel refinement method; see the module docstring."""

SUBPIXEL_REFINEMENTS: tuple[SubpixelRefinement, ...] = ('cross_power', 'ncc_quadratic')
"""Every accepted :data:`SubpixelRefinement` value."""

NCC_QUADRATIC_MAX_SHIFT_PX: float = 1.0
"""Largest distance, per axis, the quadratic fit may move the offset from the whole pixel.

A fitted maximum farther away than this along an axis is moved back to this distance on
that axis.
"""


def ncc_quadratic_vertex(corr: NDArrayFloatType, rc: tuple[int, int]) -> tuple[float, float] | None:
    """Return the maximum of a quadratic fitted to the 3x3 NCC values around one cell.

    The fit is the least-squares surface ``c0 + cv*v + cu*u + cvv*v**2 + cuu*u**2 + cvu*v*u``
    over the nine cells whose row and column lie within one of ``rc``.  The surface is
    circular, as a correlation surface is, so a neighbor past an edge of the array is read
    from the opposite edge.  Each axis of the result is limited to
    ``NCC_QUADRATIC_MAX_SHIFT_PX`` in magnitude.

    Parameters:
        corr: The circular NCC surface.  It must be at least 3 cells along each axis.
        rc: Row and column of the whole-pixel peak.

    Returns:
        ``(dv, du)``, the fitted maximum's position relative to ``rc`` in pixels, each
        limited to ``[-NCC_QUADRATIC_MAX_SHIFT_PX, NCC_QUADRATIC_MAX_SHIFT_PX]``.  ``None``
        when the fit cannot place a maximum: when any of the nine values is not finite (an
        invalid lag reads ``-inf``), or when the fitted surface is not curved downward along
        both axes (a minimum, a saddle, or a flat or ridge-shaped surface).
    """
    rows = np.arange(rc[0] - 1, rc[0] + 2) % corr.shape[0]
    cols = np.arange(rc[1] - 1, rc[1] + 2) % corr.shape[1]
    block = np.asarray(corr[np.ix_(rows, cols)], np.float64)
    if not np.all(np.isfinite(block)):
        return None

    # The least-squares fit on a 3x3 grid of unit spacing has a closed form: each
    # coefficient is a fixed weighted sum of the nine values.  ``row_sums[k]`` is the sum
    # of the block's row at v = k - 1, and likewise for the columns.
    row_sums = block.sum(axis=1)
    col_sums = block.sum(axis=0)
    grad_v = (row_sums[2] - row_sums[0]) / 6.0
    grad_u = (col_sums[2] - col_sums[0]) / 6.0
    # Second derivatives of the fitted surface (twice the squared-term coefficients).
    hess_vv = (row_sums[0] - 2.0 * row_sums[1] + row_sums[2]) / 3.0
    hess_uu = (col_sums[0] - 2.0 * col_sums[1] + col_sums[2]) / 3.0
    hess_vu = (block[2, 2] - block[2, 0] - block[0, 2] + block[0, 0]) / 4.0
    det = hess_vv * hess_uu - hess_vu * hess_vu
    if not (hess_vv < 0.0 and det > 0.0):
        return None
    # The maximum is where the gradient of the fitted surface vanishes.
    dv = -(hess_uu * grad_v - hess_vu * grad_u) / det
    du = -(hess_vv * grad_u - hess_vu * grad_v) / det
    limit = NCC_QUADRATIC_MAX_SHIFT_PX
    return float(np.clip(dv, -limit, limit)), float(np.clip(du, -limit, limit))
