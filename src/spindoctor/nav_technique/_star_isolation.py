"""Source isolation for the star techniques: is a detection a point source standing alone.

A ridge of a ring or a limb, or a cosmic-ray track, is a peak of the matched-filter
response but runs through the annulus around it; a star falls to the background inside
it.  The star field's source detection, the unique match, and the window searches all
apply this one test.  The submodule is private; its functions carry no underscore prefix
because the star techniques import them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.ndimage import correlate, maximum_filter

from spindoctor.nav_model.stars.detection import matched_filter_image
from spindoctor.nav_technique._star_helpers import gaussian_psf_kernel
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'is_isolated_point_source',
    'isolated_response',
    'isolation_lookup',
    'isolation_radii',
]


def isolation_radii(inner_px: int, outer_px: int, smear_px: float) -> tuple[int, int]:
    """Return the isolation annulus radii widened for a smeared star.

    A star smeared over ``smear_px`` pixels reaches half that length either side of
    its center, so the annulus moves out by that much: the star's own streak does not
    cross it, while a ridge of a ring or a cosmic-ray track crosses an annulus of any
    radius.

    Parameters:
        inner_px: Inner radius for an unsmeared star, in pixels.
        outer_px: Outer radius for an unsmeared star, in pixels.
        smear_px: Expected smear length in pixels.

    Returns:
        ``(inner, outer)`` radii in whole pixels.
    """
    margin = math.ceil(max(smear_px, 0.0) / 2.0)
    return inner_px + margin, outer_px + margin


def _annulus_footprint(inner_px: int, outer_px: int) -> NDArrayBoolType:
    """Return a boolean annulus footprint of radii ``[inner_px, outer_px]``.

    Parameters:
        inner_px: Inner radius in pixels (inclusive).
        outer_px: Outer radius in pixels (inclusive); the footprint is
            ``2 * outer_px + 1`` pixels square.

    Returns:
        A ``(2 * outer_px + 1, 2 * outer_px + 1)`` boolean array, True on the
        annulus.
    """
    grid_v, grid_u = np.mgrid[-outer_px : outer_px + 1, -outer_px : outer_px + 1]
    radius = np.hypot(grid_v, grid_u)
    footprint: NDArrayBoolType = (radius >= inner_px) & (radius <= outer_px)
    return footprint


def isolated_response(
    response: NDArrayFloatType,
    *,
    inner_px: int,
    outer_px: int,
    max_ratio: float,
) -> NDArrayBoolType:
    """Return where a matched-filter response is an isolated peak.

    Over an annulus of radii ``[inner_px, outer_px]`` around each pixel, the
    brightest response may exceed the annulus mean by less than ``max_ratio``
    of the pixel's own height above that mean.  A point source falls to the
    background well inside the annulus, so its annulus is flat; a ridge along a
    ring or a limb, or a cosmic-ray track, crosses the annulus and lifts its
    maximum to nearly the peak.

    Parameters:
        response: Matched-filter response image.
        inner_px: Inner annulus radius in pixels.
        outer_px: Outer annulus radius in pixels.
        max_ratio: Largest allowed ratio of the annulus maximum's excess over
            the annulus mean to the pixel's excess over it.

    Returns:
        Boolean array of ``response``'s shape, True where the pixel is isolated.
    """
    annulus = _annulus_footprint(inner_px, outer_px)
    annulus_mean = correlate(
        response, annulus.astype(np.float64) / float(annulus.sum()), mode='reflect'
    )
    annulus_max = maximum_filter(response, footprint=annulus, mode='reflect')
    isolated: NDArrayBoolType = annulus_max - annulus_mean < max_ratio * (response - annulus_mean)
    return isolated


def is_isolated_point_source(
    image_ext: NDArrayFloatType,
    detection_vu: tuple[float, float],
    *,
    sigma_px: float,
    kernel_half_px: int,
    inner_px: int,
    outer_px: int,
    max_ratio: float,
) -> bool:
    """Return whether a detection is an isolated point source.

    Applies :func:`isolated_response` to the matched-filter response of a patch
    of the image around the detection, at the brightest response within a pixel
    of the detection's centroid.  This is the test the star field's source
    detection applies to every source, applied to one detection.

    Parameters:
        image_ext: 2-D extfov image array the detection was made in.
        detection_vu: ``(v, u)`` centroid of the detection.
        sigma_px: PSF sigma of the matched-filter kernel, in pixels.
        kernel_half_px: Half-width of the matched-filter kernel, in pixels.
        inner_px: Inner annulus radius in pixels.
        outer_px: Outer annulus radius in pixels.
        max_ratio: Largest allowed ratio of the annulus maximum's excess over
            the annulus mean to the peak's excess over it.

    Returns:
        True when the detection's peak is isolated.
    """
    reach = outer_px + kernel_half_px + 2
    h, w = image_ext.shape
    v_c = min(max(round(detection_vu[0]), 0), h - 1)
    u_c = min(max(round(detection_vu[1]), 0), w - 1)
    v_lo, v_hi = max(0, v_c - reach), min(h, v_c + reach + 1)
    u_lo, u_hi = max(0, u_c - reach), min(w, u_c + reach + 1)
    # A detection near the array edge gets a patch completed by reflection, as
    # the whole-image filter in the star field's detection does.
    patch = np.pad(
        np.asarray(image_ext[v_lo:v_hi, u_lo:u_hi], np.float64),
        (
            (v_lo - (v_c - reach), (v_c + reach + 1) - v_hi),
            (u_lo - (u_c - reach), (u_c + reach + 1) - u_hi),
        ),
        mode='reflect',
    )
    response = matched_filter_image(
        patch, kernel=gaussian_psf_kernel(sigma_px, size=2 * kernel_half_px + 1)
    )
    isolated = isolated_response(
        response, inner_px=inner_px, outer_px=outer_px, max_ratio=max_ratio
    )
    core = response[reach - 1 : reach + 2, reach - 1 : reach + 2]
    dv, du = np.unravel_index(int(np.argmax(core)), core.shape)
    return bool(isolated[reach - 1 + int(dv), reach - 1 + int(du)])


def isolation_lookup(
    image_ext: NDArrayFloatType,
    bounds: tuple[int, int, int, int],
    *,
    sigma_px: float,
    kernel_half_px: int,
    inner_px: int,
    outer_px: int,
    max_ratio: float,
) -> Callable[[tuple[float, float]], bool]:
    """Return :func:`is_isolated_point_source` for detections in one window, built once.

    The matched-filter response and its isolation map are computed once over the window
    grown by the reach the single-detection test reads, with the image completed by
    reflection beyond its edges as that test completes it, so each lookup returns what
    :func:`is_isolated_point_source` returns for a detection in the window, at the cost of
    one array read instead of one filter pass per detection.

    Parameters:
        image_ext: 2-D extfov image array.
        bounds: ``(v_lo, v_hi, u_lo, u_hi)`` of the window, from
            :func:`search_window_bounds`.
        sigma_px: PSF sigma of the matched-filter kernel, in pixels.
        kernel_half_px: Half-width of the matched-filter kernel, in pixels.
        inner_px: Inner annulus radius in pixels.
        outer_px: Outer annulus radius in pixels.
        max_ratio: Largest allowed ratio of the annulus maximum's excess over the annulus
            mean to the peak's excess over it.

    Returns:
        A function of a detection's ``(v, u)`` centroid, which must lie in the window,
        returning whether the detection is isolated.
    """
    reach = outer_px + kernel_half_px + 2
    h, w = image_ext.shape
    v_lo, v_hi, u_lo, u_hi = bounds
    pv_lo, pv_hi = v_lo - reach - 1, v_hi + reach + 1
    pu_lo, pu_hi = u_lo - reach - 1, u_hi + reach + 1
    patch = np.pad(
        np.asarray(
            image_ext[max(0, pv_lo) : min(h, pv_hi), max(0, pu_lo) : min(w, pu_hi)], np.float64
        ),
        (
            (max(0, pv_lo) - pv_lo, pv_hi - min(h, pv_hi)),
            (max(0, pu_lo) - pu_lo, pu_hi - min(w, pu_hi)),
        ),
        mode='reflect',
    )
    response = matched_filter_image(
        patch, kernel=gaussian_psf_kernel(sigma_px, size=2 * kernel_half_px + 1)
    )
    isolated = isolated_response(
        response, inner_px=inner_px, outer_px=outer_px, max_ratio=max_ratio
    )

    def lookup(detection_vu: tuple[float, float]) -> bool:
        v_c = min(max(round(detection_vu[0]), 0), h - 1) - pv_lo
        u_c = min(max(round(detection_vu[1]), 0), w - 1) - pu_lo
        core = response[v_c - 1 : v_c + 2, u_c - 1 : u_c + 2]
        dv, du = np.unravel_index(int(np.argmax(core)), core.shape)
        return bool(isolated[v_c - 1 + int(dv), u_c - 1 + int(du)])

    return lookup
