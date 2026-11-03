"""Local background of a star detection for the star techniques.

A star often sits on the sloping glare of a ring or a body.  A centroid weighted above the
median of its box leaves the slope's upper half in the weights and is pulled uphill, so the
star techniques subtract a plane fitted to the ring of pixels around the box instead.  The
submodule is private; its functions carry no underscore prefix because the star helpers
import them.
"""

from __future__ import annotations

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'MAD_TO_SIGMA',
    'box_background',
    'region_centroid',
    'robust_center_spread',
]

_BACKGROUND_ANNULUS_WIDTH_PX = 2
"""Width of the ring around a centroid box to which the plane background is fitted."""

MAD_TO_SIGMA = 1.4826
"""Scale from a median absolute deviation to a Gaussian standard deviation."""

_BACKGROUND_CLIP_SIGMAS = 3.0
"""Robust sigmas from the ring's median past which a ring pixel leaves the plane fit."""


def robust_center_spread(values: NDArrayFloatType) -> tuple[float, float]:
    """Return the median of ``values`` and their spread as a Gaussian sigma.

    Parameters:
        values: Pixel values (any shape).

    Returns:
        ``(median, MAD_TO_SIGMA * median absolute deviation from it)``.
    """
    center = float(np.median(values))
    return center, MAD_TO_SIGMA * float(np.median(np.abs(values - center)))


def box_background(
    image_ext: NDArrayFloatType, v_lo: int, v_hi: int, u_lo: int, u_hi: int
) -> NDArrayFloatType:
    """Return a plane background for the centroid box ``[v_lo:v_hi, u_lo:u_hi]``.

    The plane's slope is the least-squares fit to the ring of pixels
    ``_BACKGROUND_ANNULUS_WIDTH_PX`` wide around the box (pixels beyond the image left
    out), far enough from the star that its own light does not tilt the fit, less the ring
    pixels more than three robust sigmas from the ring's median (the light of a source
    larger than the box, or a cosmic ray), and its level is the median of the box less
    that slope, so a flat background gives the box median.  Under a star on a sloping
    glare of a ring or a body, the box median alone leaves the slope's upper half in the
    moment's weights and pulls the centroid uphill.  When the kept ring pixels cannot fix
    a plane (fewer than three, or all on one line, as a quantized background's median
    absolute deviation of zero can leave them), the background is the box median.

    Parameters:
        image_ext: 2-D extfov image array.
        v_lo: First row of the box.
        v_hi: Row after the box's last.
        u_lo: First column of the box.
        u_hi: Column after the box's last.

    Returns:
        The background at every pixel of the box.
    """
    # F ring frame N1571454567: a V 9.9 star on ring glare was measured 1.7 px from where
    # a high-passed image puts it while the background was the box median.
    h, w = image_ext.shape
    width = _BACKGROUND_ANNULUS_WIDTH_PX
    r_lo, r_hi = max(0, v_lo - width), min(h, v_hi + width)
    c_lo, c_hi = max(0, u_lo - width), min(w, u_hi + width)
    # Coordinates relative to the box's center, so the fitted slope carries no level.
    v_mid = 0.5 * (v_lo + v_hi - 1)
    u_mid = 0.5 * (u_lo + u_hi - 1)
    vv, uu = np.mgrid[r_lo:r_hi, c_lo:c_hi]
    ring = ~((vv >= v_lo) & (vv < v_hi) & (uu >= u_lo) & (uu < u_hi))
    box = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
    values = np.asarray(image_ext[r_lo:r_hi, c_lo:c_hi], np.float64)[ring]
    center, spread = robust_center_spread(values)
    sky = np.abs(values - center) <= _BACKGROUND_CLIP_SIGMAS * spread
    design = np.column_stack(
        [np.ones(int(sky.sum())), vv[ring][sky] - v_mid, uu[ring][sky] - u_mid]
    ).astype(np.float64)
    if int(np.linalg.matrix_rank(design)) < 3:
        return np.full(box.shape, float(np.median(box)))
    coeffs, *_ = np.linalg.lstsq(design, values[sky], rcond=None)
    bv, bu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
    slope = coeffs[1] * (bv - v_mid) + coeffs[2] * (bu - u_mid)
    return np.asarray(slope + float(np.median(box - slope)), np.float64)


def region_centroid(
    image_ext: NDArrayFloatType, region: tuple[int, int, int, int]
) -> tuple[float, float] | None:
    """Return the background-subtracted brightness-weighted centroid of a region.

    ``region`` is ``(v_lo, v_hi, u_lo, u_hi)``, upper bounds exclusive, inside the image.
    The background is the plane of :func:`box_background` over the region.  Returns
    ``None`` when nothing in the region rises above it.
    """
    v_lo, v_hi, u_lo, u_hi = region
    box = np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64)
    weights = np.clip(box - box_background(image_ext, v_lo, v_hi, u_lo, u_hi), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    vs = np.arange(v_lo, v_hi, dtype=np.float64)
    us = np.arange(u_lo, u_hi, dtype=np.float64)
    return (
        float(np.sum(vs[:, None] * weights) / total),
        float(np.sum(us[None, :] * weights) / total),
    )
