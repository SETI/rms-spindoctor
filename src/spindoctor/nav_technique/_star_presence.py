"""Whether a star is found where another star's offset puts it.

StarUniqueMatchNav corroborates a one-star match by looking for the field's other stars at
the offset the match implies.  The test here decides whether a star is present at one such
position, for a point source and for a streak, without fitting anything.

The submodule is private (leading underscore on the file name); its functions carry no
underscore prefix because the technique imports them, but they are not part of the
package's public API.
"""

from __future__ import annotations

import math

import numpy as np

from spindoctor.nav_technique._star_helpers import (
    _box_background,
    local_centroid,
    robust_center_spread,
)
from spindoctor.nav_technique._star_streak import is_streak, streak_half_extent_px
from spindoctor.support.types import NDArrayFloatType

__all__ = ['found_at']


_FOUND_RING_INNER_GAP_PX = 2
"""Gap in pixels between a centroid box's edge and the ring ``found_at`` measures."""

_FOUND_RING_OUTER_GAP_PX = 3
"""Distance in pixels from a centroid box's edge to the outside of that ring."""


def found_at(
    image_ext: NDArrayFloatType,
    target_vu: tuple[float, float],
    *,
    tolerance_px: float,
    box_half_px: int,
    noise_sigma: float,
    detection_sigma: float,
    streak_vu: tuple[float, float] = (0.0, 0.0),
) -> bool:
    """Return whether a point source lies within ``tolerance_px`` of ``target_vu``.

    The brightest pixel within ``tolerance_px`` of the target on each axis is the
    candidate.  It must stand ``detection_sigma`` times the local spread above the
    median of the ring of pixels 2 to 3 px outside its ``box_half_px`` box, the spread
    being the larger of the ring's scaled median absolute deviation and
    ``noise_sigma``, so a pixel of a body's limb or texture, or a quantized background
    step, is not taken for a star.  Its moment centroid over the box, above the plane
    background the local centroid uses, must then lie within ``tolerance_px`` of the
    target.

    On a streak frame the brightest pixel lands anywhere along the streak, so the star
    is instead taken as :func:`local_centroid` takes a streak (see
    :func:`_streak_found_at`).

    Parameters:
        image_ext: The extended-FOV image.
        target_vu: Where the star should be, ``(v, u)``.
        tolerance_px: Largest distance of the star from the target in pixels.
        box_half_px: Half-width of the centroid box in pixels.
        noise_sigma: Image noise sigma, the floor of the local spread.
        detection_sigma: Multiplier on the local spread the peak must clear.
        streak_vu: The frame's star streak ``(v, u)`` in pixels; ``(0, 0)`` for points.

    Returns:
        True when a star is found at the target.
    """
    if is_streak(streak_vu):
        return _streak_found_at(
            image_ext,
            target_vu,
            tolerance_px=tolerance_px,
            box_half_px=box_half_px,
            noise_sigma=noise_sigma,
            detection_sigma=detection_sigma,
            streak_vu=streak_vu,
        )
    rows, cols = image_ext.shape
    t_v, t_u = target_vu
    reach = math.ceil(tolerance_px)
    c_v, c_u = round(t_v), round(t_u)
    v_lo, v_hi = max(0, c_v - reach), min(rows, c_v + reach + 1)
    u_lo, u_hi = max(0, c_u - reach), min(cols, c_u + reach + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return False
    window = image_ext[v_lo:v_hi, u_lo:u_hi]
    w_v, w_u = np.unravel_index(int(np.argmax(window)), window.shape)
    p_v, p_u = v_lo + int(w_v), u_lo + int(w_u)
    outer = box_half_px + _FOUND_RING_OUTER_GAP_PX
    if not (outer <= p_v < rows - outer and outer <= p_u < cols - outer):
        return False
    patch = np.asarray(
        image_ext[p_v - outer : p_v + outer + 1, p_u - outer : p_u + outer + 1], np.float64
    )
    offsets = np.abs(np.arange(-outer, outer + 1))
    dist = np.maximum.outer(offsets, offsets)
    ring = patch[dist >= box_half_px + _FOUND_RING_INNER_GAP_PX]
    background, ring_spread = robust_center_spread(ring)
    spread = max(ring_spread, noise_sigma)
    if float(image_ext[p_v, p_u]) - background < detection_sigma * spread:
        return False
    box = patch[
        outer - box_half_px : outer + box_half_px + 1, outer - box_half_px : outer + box_half_px + 1
    ]
    plane = _box_background(
        image_ext,
        p_v - box_half_px,
        p_v + box_half_px + 1,
        p_u - box_half_px,
        p_u + box_half_px + 1,
    )
    weights = np.clip(box - plane, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return False
    offsets = np.arange(-box_half_px, box_half_px + 1, dtype=np.float64)
    cen_v = p_v + float(np.sum(offsets[:, None] * weights) / total)
    cen_u = p_u + float(np.sum(offsets[None, :] * weights) / total)
    return math.hypot(cen_v - t_v, cen_u - t_u) <= tolerance_px


def _streak_found_at(
    image_ext: NDArrayFloatType,
    target_vu: tuple[float, float],
    *,
    tolerance_px: float,
    box_half_px: int,
    noise_sigma: float,
    detection_sigma: float,
    streak_vu: tuple[float, float],
) -> bool:
    """Return whether a streak lies within ``tolerance_px`` of ``target_vu``.

    The streak is centroided as :func:`local_centroid` takes one, about the streak's
    center within ``tolerance_px`` of the target, and must lie within ``tolerance_px`` of
    it.  The local centroid accepts a streak whose brightest pixel near its segment stands
    ``detection_sigma`` times the noise above the plane fitted around its box; as for a
    point source, that noise is the larger of ``noise_sigma`` and the scaled median
    absolute deviation of the ring of pixels 2 to 3 px outside the centroid box, here
    grown by the streak's half extent on each axis, so a textured background is not
    taken for a streak.

    Parameters:
        image_ext: The extended-FOV image.
        target_vu: Where the star should be, ``(v, u)``.
        tolerance_px: Largest distance of the star from the target in pixels.
        box_half_px: Half-width of an unsmeared star's centroid box in pixels.
        noise_sigma: Image noise sigma, the floor of the local spread.
        detection_sigma: Multiplier on the local spread the streak must clear.
        streak_vu: The frame's star streak ``(v, u)`` in pixels.

    Returns:
        True when a streak is found at the target.
    """
    rows, cols = image_ext.shape
    ext_v, ext_u = streak_half_extent_px(streak_vu)
    inner_v, inner_u = box_half_px + ext_v, box_half_px + ext_u
    outer_v = inner_v + _FOUND_RING_OUTER_GAP_PX
    outer_u = inner_u + _FOUND_RING_OUTER_GAP_PX
    c_v, c_u = round(target_vu[0]), round(target_vu[1])
    if not (outer_v <= c_v < rows - outer_v and outer_u <= c_u < cols - outer_u):
        return False
    patch = np.asarray(
        image_ext[c_v - outer_v : c_v + outer_v + 1, c_u - outer_u : c_u + outer_u + 1],
        np.float64,
    )
    off_v = np.abs(np.arange(-outer_v, outer_v + 1))[:, None]
    off_u = np.abs(np.arange(-outer_u, outer_u + 1))[None, :]
    in_ring = (off_v >= inner_v + _FOUND_RING_INNER_GAP_PX) | (
        off_u >= inner_u + _FOUND_RING_INNER_GAP_PX
    )
    _, ring_spread = robust_center_spread(patch[in_ring])
    detection, _ = local_centroid(
        image_ext,
        target_vu,
        search_window_px=tolerance_px,
        centroid_box_half_px=box_half_px,
        image_noise_sigma=max(ring_spread, noise_sigma),
        detection_sigma=detection_sigma,
        streak_vu=streak_vu,
    )
    return detection is not None and (
        math.hypot(detection[0] - target_vu[0], detection[1] - target_vu[1]) <= tolerance_px
    )
