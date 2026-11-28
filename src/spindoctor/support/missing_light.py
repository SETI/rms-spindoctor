"""How far lost pixels move the centroid of a source.

A frame that lost lines in transmission carries the instrument's missing-data marker in
them.  The navigator fills those pixels with 0.0 and marks them in
``NavContext.missing_data_mask_ext``.  A centroid that leaves them out is still measured
from only part of the source, so it moves away from the lost part by an amount that
depends on how much of the source's light fell there and where.
:func:`missing_light_shift_px` predicts that move from a model of the source's light, so a
technique can drop a source whose centroid the lost pixels would move too far.
"""

from __future__ import annotations

import math

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['missing_light_shift_px']


def missing_light_shift_px(
    model: NDArrayFloatType,
    origin_vu: tuple[int, int],
    missing_mask: NDArrayBoolType,
) -> float:
    """Return how far leaving the missing pixels out moves a model source's centroid.

    ``model`` is the predicted light of one source on a patch of the image, and the
    patch's pixel ``[0, 0]`` lies at pixel ``origin_vu`` of ``missing_mask``.  The shift
    is the distance between the light-weighted centroid of the patch's pixels that are
    not missing and the light-weighted centroid of all of them.  A patch pixel that lies
    beyond the bounds of ``missing_mask`` counts as present.

    Parameters:
        model: 2-D non-negative model of the source's light.
        origin_vu: ``(v, u)`` pixel of ``missing_mask`` under ``model[0, 0]``; it may be
            negative, or put part of the patch beyond the mask.
        missing_mask: ``True`` where a pixel of the image is missing.

    Returns:
        The shift in pixels: 0.0 when none of the source's light falls on a missing
        pixel, and ``math.inf`` when all of it does.
    """
    rows, cols = model.shape
    h, w = missing_mask.shape
    v0, u0 = origin_vu
    missing = np.zeros(model.shape, dtype=bool)
    mv_lo, mv_hi = max(0, v0), min(h, v0 + rows)
    mu_lo, mu_hi = max(0, u0), min(w, u0 + cols)
    if mv_lo < mv_hi and mu_lo < mu_hi:
        missing[mv_lo - v0 : mv_hi - v0, mu_lo - u0 : mu_hi - u0] = missing_mask[
            mv_lo:mv_hi, mu_lo:mu_hi
        ]
    light = np.asarray(model, np.float64)
    present = np.where(missing, 0.0, light)
    total = float(light.sum())
    present_total = float(present.sum())
    if present_total <= 0.0:
        return math.inf
    if present_total == total:
        return 0.0
    vs = np.arange(rows, dtype=np.float64)[:, None]
    us = np.arange(cols, dtype=np.float64)[None, :]
    shift_v = float((present * vs).sum()) / present_total - float((light * vs).sum()) / total
    shift_u = float((present * us).sum()) / present_total - float((light * us).sum()) / total
    return math.hypot(shift_v, shift_u)
