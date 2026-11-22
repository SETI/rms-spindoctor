"""Residual geometric distortion for the optics stage.

The navigator corrects each camera's known distortion model, so the quantity
actually present in the frames the pipeline consumes -- and the only quantity
this stage plants -- is the residual: a low-order radial polynomial about the
optical center plus an optional small non-radial wander.  A limb fitted at the
frame edge then disagrees with a ring fitted through the center by the
differential residual between their positions, which the navigator gets no
model to remove.

The warp maps each output pixel ``p`` to a source position
``center + (p - center) * (1 + k1*rho^2 + k2*rho^4)`` with
``rho = |p - center| / rho_ref`` and ``rho_ref`` half the image diagonal; the
scene is resampled there by cubic interpolation.  The non-radial term adds a
seeded, smooth 2-D displacement field (its own RNG stream) at the commanded
RMS amplitude.

The stage runs before the PSF, so a plane is usually zero away from the bodies,
rings, and stars, and the warp resamples only the lit parts.  Lit pixels are
grouped into clusters, and each cluster is warped on its own crop, with a margin
of dark pixels around it.  The warp is linear, so adding the clusters' results
gives the warp of the whole plane.

Before interpolating, the cubic spline computes a coefficient for every pixel of
the crop.  This step, the prefilter, spreads each lit pixel into the dark around
it.  The spread falls by a factor of 2 - sqrt(3), about 0.27, per pixel, so by
about 1e-37 across the margin.  Every lit pixel therefore comes out as the
whole-frame warp makes it, apart from rounding.  Where the whole-frame warp leaves
values below 1e-37 of the peak in the dark, the crops leave zero.

If the areas the crops are interpolated over would add up to at least the
frame's area, the plane is warped in one piece instead.  A plane that smear or a
diffuse sky has lit everywhere is warped that way, and gains nothing.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.sim.render.optics.lit_clusters import (
    CLUSTER_BLOCK_PX,
    cluster_crop,
    find_lit_clusters,
    widen_slice,
)
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.scene.fields import scene_value
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_distortion', 'clear_distortion_caches', 'distortion_reach_px']

# Control-grid resolution of the non-radial wander before it is smoothly
# upsampled to the render grid: a handful of points keeps the field low-order
# (crater-scale coherent, not per-pixel jitter).
_NONRADIAL_CONTROL = 4

# Dark render-grid pixels kept around each cluster when it is cropped.  The cubic
# prefilter's response falls by 2 - sqrt(3) per pixel, so by about 1e-37 across
# this margin.  A whole number of blocks, so every crop starts on a block edge.
_PREFILTER_MARGIN_PX = 2 * CLUSTER_BLOCK_PX

# The cubic spline reads up to two pixels past a source position.  An output crop
# therefore reaches this many render-grid pixels past its input crop, on top of the
# largest displacement rounded up to a whole pixel.
_SPLINE_REACH_PX = 2

# A plane lit up to its edge, as a frame-filling body lights its margin, is warped in
# one piece, and the prefilter continues the plane past its edge as its mirror image,
# which is not the scene there.  The error that makes in the coefficients falls by
# 2 - sqrt(3) per pixel away from the edge, so by 4e-16 across this many render-grid
# pixels, which the margin keeps between the plane's edge and the pixels the later
# stages read.
_EDGE_PREFILTER_PX = 27


@dataclass(frozen=True)
class _Warp:
    """The resolved warp from output pixel to source position on the render grid.

    Parameters:
        shape: The render-grid shape.
        center_v: Optical center v, pixel-centric on the render grid.
        center_u: Optical center u, pixel-centric on the render grid.
        rho_ref: The radius that normalizes ``rho``, half the render-grid diagonal.
        k1: Radial coefficient of ``rho^2``.
        k2: Radial coefficient of ``rho^4``.
        field_v: The non-radial wander along v over the render grid, or None.
        field_u: The non-radial wander along u over the render grid, or None.
        reach: How far, in render-grid pixels, an output crop extends past its
            input crop.
    """

    shape: tuple[int, int]
    center_v: float
    center_u: float
    rho_ref: float
    k1: float
    k2: float
    field_v: NDArrayFloatType | None
    field_u: NDArrayFloatType | None
    reach: int


def clear_distortion_caches() -> None:
    """Drop the cached non-radial displacement field (test/tooling helper)."""
    _nonradial_field.cache_clear()


# One entry: the driver sizes the margin from the field before the optics stage
# warps by it, so each render builds the field once.
@lru_cache(maxsize=1)
def _nonradial_field(
    shape: tuple[int, int], rms_px: float, seed: int
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Build a seeded, smooth non-radial displacement field at a target RMS.

    The field is cached, so its planes are read-only.

    Parameters:
        shape: The frame's render-grid ``(V*os, U*os)`` shape.
        rms_px: Target RMS displacement magnitude in render-grid pixels.
        seed: Seed for the field's own RNG stream.

    Returns:
        The ``(field_v, field_u)`` displacement planes.
    """
    rng = np.random.default_rng(seed)
    coarse_v = rng.normal(size=(_NONRADIAL_CONTROL, _NONRADIAL_CONTROL))
    coarse_u = rng.normal(size=(_NONRADIAL_CONTROL, _NONRADIAL_CONTROL))
    zoom = (shape[0] / _NONRADIAL_CONTROL, shape[1] / _NONRADIAL_CONTROL)
    field_v = ndimage.zoom(coarse_v, zoom, order=3)[: shape[0], : shape[1]]
    field_u = ndimage.zoom(coarse_u, zoom, order=3)[: shape[0], : shape[1]]
    magnitude_rms = float(np.sqrt(np.mean(field_v**2 + field_u**2)))
    if magnitude_rms > 0.0:
        scale = rms_px / magnitude_rms
        field_v *= scale
        field_u *= scale
    field_v.setflags(write=False)
    field_u.setflags(write=False)
    return field_v, field_u


def _largest_radial_displacement(
    shape: tuple[int, int], center_v: float, center_u: float, rho_ref: float, k1: float, k2: float
) -> float:
    """The largest radial displacement, in render-grid pixels, anywhere in the frame.

    The displacement at radius ``r`` is ``r * (k1*s + k2*s^2)`` with
    ``s = (r / rho_ref)^2``.  Between the center and the farthest pixel, its
    magnitude peaks either at that pixel or where the displacement's derivative
    vanishes, at ``s = -3*k1 / (5*k2)``.

    Parameters:
        shape: The render-grid shape.
        center_v: Optical center v on the render grid.
        center_u: Optical center u on the render grid.
        rho_ref: The normalizing radius.
        k1: Radial coefficient of ``rho^2``.
        k2: Radial coefficient of ``rho^4``.

    Returns:
        The largest magnitude of the radial displacement.
    """
    r_max = max(
        float(stable_hypot(v - center_v, u - center_u))
        for v in (0.0, shape[0] - 1.0)
        for u in (0.0, shape[1] - 1.0)
    )
    radii = [r_max]
    if k2 != 0.0 and -3.0 * k1 / (5.0 * k2) > 0.0:
        r_turn = rho_ref * math.sqrt(-3.0 * k1 / (5.0 * k2))
        if r_turn < r_max:
            radii.append(r_turn)
    largest = 0.0
    for r in radii:
        s = (r / rho_ref) ** 2
        largest = max(largest, abs(r * (k1 * s + k2 * s**2)))
    return largest


def _source_coords(
    warp: _Warp, out_v: slice, out_u: slice, origin_v: int, origin_u: int
) -> NDArrayFloatType:
    """Source positions, relative to an input crop, for a box of output pixels.

    Parameters:
        warp: The resolved warp.
        out_v: Output rows.
        out_u: Output columns.
        origin_v: The input crop's first row; the positions are measured from it.
        origin_u: The input crop's first column; the positions are measured from it.

    Returns:
        The ``(2, rows, columns)`` source positions.
    """
    rv = (np.arange(out_v.start, out_v.stop, dtype=np.float64) - warp.center_v)[:, np.newaxis]
    ru = (np.arange(out_u.start, out_u.stop, dtype=np.float64) - warp.center_u)[np.newaxis, :]
    rho2 = (rv**2 + ru**2) / (warp.rho_ref**2)
    factor = 1.0 + warp.k1 * rho2 + warp.k2 * rho2**2
    coords = np.empty((2, out_v.stop - out_v.start, out_u.stop - out_u.start))
    coords[0] = warp.center_v + rv * factor
    coords[1] = warp.center_u + ru * factor
    if warp.field_v is not None and warp.field_u is not None:
        coords[0] += warp.field_v[out_v, out_u]
        coords[1] += warp.field_u[out_v, out_u]
    # Subtracting a whole number from a coordinate is exact, so a crop indexes the
    # same positions the whole frame would.
    coords[0] -= origin_v
    coords[1] -= origin_u
    return coords


def _warp_crop(
    crop: NDArrayFloatType, in_v: slice, in_u: slice, warp: _Warp
) -> tuple[slice, slice, NDArrayFloatType]:
    """Warp a crop of a plane, treating the rest of the plane as zero.

    Parameters:
        crop: The plane's values at ``(in_v, in_u)``.
        in_v: The crop's rows in the plane.
        in_u: The crop's columns in the plane.
        warp: The resolved warp.

    Returns:
        The output rows and columns the crop's light can reach, and the warped
        values there.
    """
    out_v = widen_slice(in_v, warp.reach, warp.shape[0])
    out_u = widen_slice(in_u, warp.reach, warp.shape[1])
    coeffs = ndimage.spline_filter(crop, order=3, output=np.float64, mode='constant')
    coords = _source_coords(warp, out_v, out_u, in_v.start, in_u.start)
    values = ndimage.map_coordinates(
        coeffs, coords, order=3, mode='constant', cval=0.0, prefilter=False
    )
    return out_v, out_u, values


def _warp_plane(plane: NDArrayFloatType, warp: _Warp) -> None:
    """Warp one plane in place, resampling only its lit clusters.

    Parameters:
        plane: The render-grid plane, modified in place.
        warp: The resolved warp.
    """
    size_v, size_u = plane.shape
    clusters = find_lit_clusters(plane)
    if not clusters.boxes:
        return
    margin = _PREFILTER_MARGIN_PX
    crops = [
        (widen_slice(v, margin, size_v), widen_slice(u, margin, size_u)) for v, u in clusters.boxes
    ]
    # A crop's cost is the area it is interpolated over: the crop widened by the
    # reach.  Once those areas add up to the frame's, one whole-frame warp is cheaper.
    outputs = [
        (widen_slice(v, warp.reach, size_v), widen_slice(u, warp.reach, size_u)) for v, u in crops
    ]
    if sum((v.stop - v.start) * (u.stop - u.start) for v, u in outputs) >= size_v * size_u:
        whole_v, whole_u = slice(0, size_v), slice(0, size_u)
        _, _, values = _warp_crop(plane, whole_v, whole_u, warp)
        plane[:] = values
        return
    source = plane.copy()
    plane[:] = 0.0
    for index, (in_v, in_u) in enumerate(crops):
        crop = cluster_crop(clusters, source, index, in_v, in_u)
        out_v, out_u, values = _warp_crop(crop, in_v, in_u, warp)
        plane[out_v, out_u] += values


def apply_distortion(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    oversample: int,
    distortion: Mapping[str, Any] | None = None,
) -> None:
    """Warp the signal and point-source planes by the residual distortion.

    A disabled block (all-zero radial coefficients and no non-radial wander) is
    a true no-op, so a scene that names a distortion block without amplitude
    renders bit-identically to one without it.

    Parameters:
        frame: The frame whose planes are warped in place.
        params: The full scene mapping; supplies the scene ``random_seed`` for
            the non-radial field's stream and, when ``distortion`` is not passed,
            the ``optics.distortion`` block.
        oversample: The render-grid oversampling factor (center and amplitude
            are in detector pixels and scale to the render grid).
        distortion: An explicit distortion block; when None the block is read
            from ``params['optics']['distortion']`` (the instrument-defaults
            residual is passed here explicitly).
    """
    if distortion is None:
        optics = params.get('optics') or {}
        distortion = optics.get('distortion')
    warp = _resolve_warp(
        distortion,
        params,
        frame_shape=frame.frame_shape,
        margin=frame.margin,
        oversample=oversample,
    )
    if warp is None:
        return
    _warp_plane(frame.signal, warp)
    _warp_plane(frame.point_e, warp)


def distortion_reach_px(
    distortion: Mapping[str, Any] | None,
    params: Mapping[str, Any],
    *,
    frame_shape: tuple[int, int],
    margin: int,
    oversample: int,
) -> int:
    """How far the residual distortion moves light, in render-grid pixels.

    The warp moves light farther the farther it lies from the optical center, so the
    reach is measured over the frame widened by ``margin``: the pixels the stages
    after the warp read.

    Parameters:
        distortion: The resolved distortion block, or None.
        params: The full scene mapping (its ``random_seed`` seeds the non-radial field).
        frame_shape: The frame's rows and columns on the render grid.
        margin: How far past the frame the reach is measured, in render-grid pixels.
        oversample: The render-grid oversampling factor.

    Returns:
        The reach, including the pixels the cubic spline reads and the pixels that
        keep the prefilter's response to the plane's edge away from the frame; 0
        when the block is disabled.
    """
    warp = _resolve_warp(
        distortion,
        params,
        frame_shape=frame_shape,
        margin=margin,
        oversample=oversample,
        with_field=False,
    )
    return 0 if warp is None else warp.reach + _EDGE_PREFILTER_PX


def _resolve_warp(
    distortion: Mapping[str, Any] | None,
    params: Mapping[str, Any],
    *,
    frame_shape: tuple[int, int],
    margin: int,
    oversample: int,
    with_field: bool = True,
) -> _Warp | None:
    """Resolve a distortion block into the warp over a plane with a margin.

    The optical center, the normalizing radius, and the non-radial field belong to
    the frame.  The plane extends ``margin`` pixels past it on each side, where the
    radial term carries on by its formula and the non-radial field repeats its value
    at the nearest frame pixel.

    Parameters:
        distortion: The resolved distortion block, or None.
        params: The full scene mapping (its ``random_seed`` seeds the non-radial field).
        frame_shape: The frame's rows and columns on the render grid.
        margin: How many render-grid pixels the plane extends past the frame.
        oversample: The render-grid oversampling factor.
        with_field: Whether to spread the non-radial field over the plane; without it
            the warp carries only its reach, which is all the margin needs.

    Returns:
        The warp, or None when the block is absent or disabled.
    """
    if not isinstance(distortion, Mapping):
        return None
    k1 = float(scene_value(distortion, 'k1', 0.0))
    k2 = float(scene_value(distortion, 'k2', 0.0))
    nonradial_rms = float(scene_value(distortion, 'nonradial_rms_px', 0.0))
    if k1 == 0.0 and k2 == 0.0 and nonradial_rms == 0.0:
        return None

    size_v, size_u = frame_shape
    shape = (size_v + 2 * margin, size_u + 2 * margin)
    # A scene states a center in pixel-corner coordinates on the detector grid;
    # the sampling grid below is pixel-centric on the oversampled one.  Scaling
    # a corner coordinate is a plain multiply, and the half pixel takes it the
    # rest of the way.  The default is the frame's own center, which lands on
    # (size - 1) / 2 by the same route.  The frame starts ``margin`` pixels into
    # the plane.
    center_v = (
        float(scene_value(distortion, 'center_v', (size_v / oversample) / 2.0)) * oversample
        - PIXEL_CENTER_TO_CORNER_PX
        + margin
    )
    center_u = (
        float(scene_value(distortion, 'center_u', (size_u / oversample) / 2.0)) * oversample
        - PIXEL_CENTER_TO_CORNER_PX
        + margin
    )
    rho_ref = 0.5 * float(stable_hypot(size_v, size_u))
    largest = _largest_radial_displacement(shape, center_v, center_u, rho_ref, k1, k2)

    field_v: NDArrayFloatType | None = None
    field_u: NDArrayFloatType | None = None
    if nonradial_rms > 0.0:
        seed = derive_effect_seed(int(params['random_seed']), 'optics/distortion')
        frame_field_v, frame_field_u = _nonradial_field(
            (size_v, size_u), nonradial_rms * oversample, seed
        )
        if with_field:
            field_v = np.pad(frame_field_v, margin, mode='edge')
            field_u = np.pad(frame_field_u, margin, mode='edge')
        largest += float(np.sqrt(np.max(frame_field_v**2 + frame_field_u**2)))

    return _Warp(
        shape=shape,
        center_v=center_v,
        center_u=center_u,
        rho_ref=rho_ref,
        k1=k1,
        k2=k2,
        field_v=field_v,
        field_u=field_u,
        reach=math.ceil(largest) + _SPLINE_REACH_PX,
    )
