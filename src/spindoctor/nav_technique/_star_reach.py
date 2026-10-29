"""Where in the image a detected source can be one of the predicted stars.

A pointing error moves every star by one offset, and a technique answers only offsets
inside its search window.  A source farther from every predicted star than the window
reaches is therefore no star the technique can match at any offset it can answer, however
bright it is.  StarFieldFromCatalogNav keeps only its brightest sources, and on a frame
crossed by ring texture the brightest sources are mostly such unreachable peaks, which
crowd the faint stars out of the cohort.  The mask here marks the pixels a source must
lie in to be kept.

The submodule is private (leading underscore on the file name); its functions carry no
underscore prefix because the technique imports them, but they are not part of the
package's public API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from spindoctor.support.types import NDArrayBoolType

__all__ = ['reach_mask']


def reach_mask(
    shape: tuple[int, int],
    predictions_vu: Sequence[tuple[float, float]],
    *,
    reach_vu: tuple[float, float],
) -> NDArrayBoolType:
    """Return the pixels within reach of at least one predicted star.

    A pixel is within reach of a prediction when it lies within ``reach_vu[0]`` of it
    along v and within ``reach_vu[1]`` along u, the bounds inclusive.  A prediction off
    the image contributes the part of its rectangle that falls on it.

    Parameters:
        shape: ``(rows, cols)`` of the image the mask covers.
        predictions_vu: Predicted star positions ``(v, u)`` in that image's pixels.
        reach_vu: Half-widths ``(v, u)`` in pixels of the rectangle around each
            prediction; each must be finite and non-negative.

    Returns:
        A boolean array of ``shape``, True on every pixel within reach.

    Raises:
        ValueError: If a half-width is negative or not finite.
    """
    reach_v, reach_u = (float(x) for x in reach_vu)
    if not (math.isfinite(reach_v) and math.isfinite(reach_u)) or min(reach_v, reach_u) < 0.0:
        raise ValueError(f'reach_vu must be finite and >= 0; got {reach_vu!r}')
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.bool_)
    for v, u in predictions_vu:
        v_lo = max(0, math.ceil(v - reach_v))
        v_hi = min(rows, math.floor(v + reach_v) + 1)
        u_lo = max(0, math.ceil(u - reach_u))
        u_hi = min(cols, math.floor(u + reach_u) + 1)
        if v_lo < v_hi and u_lo < u_hi:
            mask[v_lo:v_hi, u_lo:u_hi] = True
    return mask
