"""Streak geometry shared by the star techniques' centroids.

A star that moves across the detector during the exposure is a streak: its light spread
along the motion, with the point-spread function's width across it.  The predicted star
position is the star at mid-exposure, the streak's center.  Once the streak is long
enough to have a flat top, its brightest pixel falls at a point along it that the noise
picks, so a centroid box placed around that pixel measures a random point along the
streak, or cuts off part of it.  These helpers let a centroid find the streak instead: a
line kernel along the streak whose matched-filter response peaks at the streak's center,
the extra half-extent a centroid box needs on each axis to hold the whole streak, and the
capsule of pixels around the streak that holds its light.  The streak is the one the star
model measures on the frame (:mod:`spindoctor.nav_model.stars.streak`) and records on
every STAR feature.

The submodule is private (leading underscore on the file name); its helpers carry no
underscore prefix because every star technique imports them, but they are not part of
the package's public API.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate
from scipy.signal import convolve2d

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.flags import StarFlags
from spindoctor.nav_model.stars.streak import MIN_STREAK_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'frame_streak_vu',
    'is_streak',
    'streak_aperture',
    'streak_half_extent_px',
    'streak_kernel',
    'streak_matched_kernel',
    'streak_peak',
]

_STREAK_SAMPLE_STEP_PX = 0.25
"""Spacing in pixels of the points along the streak that the line kernel is built from."""


def is_streak(streak_vu: tuple[float, float]) -> bool:
    """Return whether a streak is long enough for a star to be centroided as one.

    Parameters:
        streak_vu: The streak ``(v, u)`` in pixels.

    Returns:
        True when its length is at least
        :data:`~spindoctor.nav_model.stars.streak.MIN_STREAK_PX`.
    """
    return math.hypot(streak_vu[0], streak_vu[1]) >= MIN_STREAK_PX


def streak_half_extent_px(streak_vu: tuple[float, float]) -> tuple[int, int]:
    """Return the pixels a streak reaches beyond its center on each axis.

    Parameters:
        streak_vu: The streak ``(v, u)`` in pixels.

    Returns:
        ``(ceil(|v| / 2), ceil(|u| / 2))``, the amount a centroid box centered on the
        streak's center must grow on each axis, beyond the box an unsmeared star needs,
        to hold the whole streak.
    """
    return math.ceil(abs(streak_vu[0]) / 2.0), math.ceil(abs(streak_vu[1]) / 2.0)


def streak_kernel(streak_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return a line kernel along the streak, centered and summing to one.

    Points every :data:`_STREAK_SAMPLE_STEP_PX` (at least two) along the segment from
    ``-streak_vu / 2`` to ``+streak_vu / 2`` each spread an equal weight over the four
    pixels around them by bilinear interpolation.  The kernel has an odd side on each
    axis, one pixel wider on each end than :func:`streak_half_extent_px`, and is
    symmetric about its center pixel, so the direction the streak is given in does not
    matter.

    Parameters:
        streak_vu: The streak ``(v, u)`` in pixels.

    Returns:
        The kernel, of shape ``(2 * (ev + 1) + 1, 2 * (eu + 1) + 1)`` for half-extents
        ``(ev, eu)``.
    """
    ext_v, ext_u = streak_half_extent_px(streak_vu)
    half_v, half_u = ext_v + 1, ext_u + 1
    kernel = np.zeros((2 * half_v + 1, 2 * half_u + 1), np.float64)
    length = math.hypot(streak_vu[0], streak_vu[1])
    n_points = max(2, math.ceil(length / _STREAK_SAMPLE_STEP_PX) + 1)
    weight = 1.0 / n_points
    for t in np.linspace(-0.5, 0.5, n_points):
        pos_v = half_v + t * streak_vu[0]
        pos_u = half_u + t * streak_vu[1]
        v0 = math.floor(pos_v)
        u0 = math.floor(pos_u)
        fv = pos_v - v0
        fu = pos_u - u0
        kernel[v0, u0] += weight * (1.0 - fv) * (1.0 - fu)
        kernel[v0 + 1, u0] += weight * fv * (1.0 - fu)
        kernel[v0, u0 + 1] += weight * (1.0 - fv) * fu
        kernel[v0 + 1, u0 + 1] += weight * fv * fu
    return kernel


def streak_matched_kernel(
    point_kernel: NDArrayFloatType, streak_vu: tuple[float, float]
) -> NDArrayFloatType:
    """Return a point-source matched-filter kernel spread along a streak.

    Parameters:
        point_kernel: Kernel for an unsmeared star, such as a Gaussian of the PSF's width.
        streak_vu: The streak ``(v, u)`` in pixels.

    Returns:
        ``point_kernel`` convolved with :func:`streak_kernel`, summing to one; each side
        is the sum of the two kernels' sides less one, so it stays odd when both are.
    """
    kernel = np.asarray(convolve2d(point_kernel, streak_kernel(streak_vu)), np.float64)
    return kernel / float(kernel.sum())


def streak_peak(
    image_ext: NDArrayFloatType,
    window_vu: tuple[int, int, int, int],
    streak_vu: tuple[float, float],
) -> tuple[int, int]:
    """Return the pixel inside a window where the streak line kernel responds most.

    The image is correlated with :func:`streak_kernel` less its mean over the kernel's
    footprint, over the window grown by the kernel's half-size on each side (clamped to
    the image), so a streak that crosses the window's edge is still seen whole, and the
    largest response inside the window is returned.  On an even streak the response is
    largest at the streak's center.  The kernel is symmetric about its center and sums
    to zero, so a uniform or evenly sloping background gives every pixel the same
    response and does not draw the peak uphill.

    Parameters:
        image_ext: 2-D extfov image array.
        window_vu: ``(v_lo, u_lo, v_hi, u_hi)`` of the window, ``hi`` exclusive; it must
            hold at least one pixel.
        streak_vu: The streak ``(v, u)`` in pixels.

    Returns:
        ``(v, u)`` image indices of the largest response; ties go to the first in row
        order.
    """
    kernel = streak_kernel(streak_vu)
    kernel -= float(kernel.mean())
    pad_v, pad_u = kernel.shape[0] // 2, kernel.shape[1] // 2
    v_lo, u_lo, v_hi, u_hi = window_vu
    h, w = image_ext.shape
    r_lo, r_hi = max(0, v_lo - pad_v), min(h, v_hi + pad_v)
    c_lo, c_hi = max(0, u_lo - pad_u), min(w, u_hi + pad_u)
    region = np.asarray(image_ext[r_lo:r_hi, c_lo:c_hi], np.float64)
    response = correlate(region, kernel, mode='nearest')
    inside = response[v_lo - r_lo : v_hi - r_lo, u_lo - c_lo : u_hi - c_lo]
    p_v, p_u = np.unravel_index(int(np.argmax(inside)), inside.shape)
    return v_lo + int(p_v), u_lo + int(p_u)


def streak_aperture(
    box_vu: tuple[int, int, int, int],
    center_vu: tuple[int, int],
    streak_vu: tuple[float, float],
    radius_px: float,
) -> NDArrayFloatType:
    """Return which pixels of a box lie within ``radius_px`` of a streak's segment.

    The segment runs from ``center - streak_vu / 2`` to ``center + streak_vu / 2``; the
    aperture is the capsule of pixels whose centers lie within ``radius_px`` of it, the
    streak's own pixels without the box corners a rectangle around a slanted streak
    adds, whose noise would only scatter the centroid.

    Parameters:
        box_vu: ``(v_lo, u_lo, v_hi, u_hi)`` of the box, ``hi`` exclusive.
        center_vu: ``(v, u)`` pixel at the segment's center.
        streak_vu: The streak ``(v, u)`` in pixels.
        radius_px: Largest distance of an aperture pixel's center from the segment.

    Returns:
        Boolean array of the box's shape, True inside the aperture.
    """
    v_lo, u_lo, v_hi, u_hi = box_vu
    vv, uu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
    dv = (vv - center_vu[0]).astype(np.float64)
    du = (uu - center_vu[1]).astype(np.float64)
    length_sq = streak_vu[0] ** 2 + streak_vu[1] ** 2
    if length_sq > 0.0:
        t = np.clip((dv * streak_vu[0] + du * streak_vu[1]) / length_sq, -0.5, 0.5)
        dv = dv - t * streak_vu[0]
        du = du - t * streak_vu[1]
    return np.asarray(dv * dv + du * du <= radius_px * radius_px)


def frame_streak_vu(features: list[NavFeature]) -> tuple[float, float]:
    """Return the per-axis median streak of the STAR features.

    A technique that searches the whole frame rather than one star's surroundings
    needs one streak for every detection.  The star model records the same streak on
    every STAR feature of a frame, so the median is that streak.

    Parameters:
        features: Features of any type; only those carrying :class:`StarFlags` count.

    Returns:
        The median ``(v, u)`` streak in pixels, or ``(0.0, 0.0)`` when no feature
        carries star flags.
    """
    streaks = [f.flags.streak_vu for f in features if isinstance(f.flags, StarFlags)]
    if len(streaks) == 0:
        return 0.0, 0.0
    arr = np.asarray(streaks, np.float64)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
