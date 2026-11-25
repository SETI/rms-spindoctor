"""Placement of a rendered body: draw it on a canvas, translate it into the frame.

A body renderer draws its body at a reference center, the frame center, so a shape
cache can serve the same body at any position.  The drawing is made on the
*reference plane*, the frame's pixel grid extended without bound in every
direction, and the box of that plane it is drawn on is the body's canvas
(:func:`body_canvas`).  :func:`finish_single_body` then translates the drawing to
the body's center by linear interpolation, composites the body over the frame, and
builds the body's inventory entry.  The ellipsoid path in
:mod:`spindoctor.sim.render.bodies.render` and the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`) both place their bodies here.

A body is composited by how much of each pixel it covers.  A renderer draws the
body's light already weighted by that coverage, so a pixel the outline covers by a
fraction ``c`` holds ``c`` times the light of the part it covers.  The frame pixel
becomes that light plus ``1 - c`` times whatever lay behind it: a farther body, the
ring system, the body's own haze glow, or the empty sky.  The translation moves the
weighted light and the coverage by the same interpolation, so the two stay
consistent at any sub-pixel position.

A body that fits in the frame, with a pixel to spare, when drawn at the frame center
is drawn on the frame itself.  A larger one, which a frame-sized drawing would cut
off, is drawn on a canvas that holds the part of it that lands in the frame, so a
planet larger than the frame shows its limb wherever its center lies, on the frame
or off it.

The image a body is painted into may extend past the frame by a margin on every
side, which the optics stage spreads into the frame (see
:mod:`spindoctor.sim.render.stages`).  The body is then painted across the margin
too: a canvas past the frame holds the part of the body that lands in the frame or
the margin.  The frame alone decides where the body is drawn and how its craters are
laid out, so the margin changes no pixel of the frame.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.atmosphere import HaloScreen
from spindoctor.sim.render.bodies.translate import nonzero_box, shifted_box, translate_box
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'PARTIAL_DRAWINGS_KEPT',
    'BodyCanvas',
    'BodyCoverage',
    'body_canvas',
    'composite_halo',
    'finish_single_body',
    'finish_unseen_body',
    'uncovered_fraction',
]

# Detector pixels a canvas keeps beyond the part of the body the frame reads.  The
# relief tilt is interpolated by a cubic spline, whose end conditions reach into a
# canvas by a fraction 0.27 per pixel, so 8 pixels hold their effect on the frame
# below 3e-5 of the tilt.
_CANVAS_MARGIN_DETECTOR_PX = 8

# The uncovered fraction below which a pixel counts as wholly covered.  Linear
# interpolation of a coverage of exactly 1 can return 1 less one unit in the last
# place, which would let 1e-16 of the background through the middle of a body.
_FULL_COVERAGE_TOLERANCE = 1e-12


PARTIAL_DRAWINGS_KEPT = 1
"""How many drawings on a canvas reaching past the frame a body renderer keeps.

Such a canvas changes whenever the body moves, so an offset sweep never reuses a
drawing, and each drawing is about the size of the frame. The radiance stage draws
every body twice for a scene that smears each layer apart, once for the composite and
once for the layers, so keeping one drawing serves a scene with one body this large.
A scene with two such bodies, which few scenes have, draws each of them twice.
"""


@dataclass(frozen=True)
class BodyCoverage:
    """How much of each frame pixel a placed body covers.

    Parameters:
        box_v: Frame rows outside which the body covers nothing.
        box_u: Frame columns outside which the body covers nothing.
        coverage: The box-sized fraction of each pixel the body's solid outline
            covers, in [0, 1].
    """

    box_v: slice
    box_u: slice
    coverage: NDArrayFloatType

    @cached_property
    def transmission(self) -> NDArrayFloatType:
        """The box-sized fraction of each pixel the body leaves uncovered.

        It is what shows through from behind the body: ``1 - coverage``, and exactly
        0 where the uncovered fraction is below 1e-12.  Starlight and everything the
        body passes in front of are scaled by it.  It is computed once, on first
        use, and is read-only.
        """
        uncovered = 1.0 - self.coverage
        uncovered[uncovered < _FULL_COVERAGE_TOLERANCE] = 0.0
        uncovered.setflags(write=False)
        return uncovered


def uncovered_fraction(
    coverages: Sequence[BodyCoverage],
) -> tuple[tuple[slice, slice], NDArrayFloatType]:
    """The fraction of each pixel no body covers, over the box the bodies cover.

    The fraction is the product of the bodies' transmissions, multiplied in the order
    given into an array of ones, and is exactly 1 outside the box, where a multiply
    by it leaves any value as it is.  Only the box is built, which on a frame with
    small bodies is far less than the frame.

    Parameters:
        coverages: The bodies' coverages, in the order they are multiplied.

    Returns:
        The box's rows and columns on the frame, and the box-sized fraction; an empty
        box when no body covers any pixel.
    """
    covering = [
        coverage
        for coverage in coverages
        if coverage.box_v.stop > coverage.box_v.start and coverage.box_u.stop > coverage.box_u.start
    ]
    if not covering:
        return (slice(0, 0), slice(0, 0)), np.ones((0, 0), dtype=np.float64)
    rows = slice(min(c.box_v.start for c in covering), max(c.box_v.stop for c in covering))
    cols = slice(min(c.box_u.start for c in covering), max(c.box_u.stop for c in covering))
    fraction = np.ones((rows.stop - rows.start, cols.stop - cols.start), dtype=np.float64)
    for coverage in covering:
        fraction[
            coverage.box_v.start - rows.start : coverage.box_v.stop - rows.start,
            coverage.box_u.start - cols.start : coverage.box_u.stop - cols.start,
        ] *= coverage.transmission
    return (rows, cols), fraction


@dataclass(frozen=True)
class BodyCanvas:
    """The box of the reference plane a body is drawn on.

    The reference plane is the frame's pixel grid extended without bound in every
    direction; its pixel ``(i, j)`` is the frame's pixel ``(i, j)`` wherever the frame
    has one.

    Parameters:
        rows: The canvas rows on the reference plane.
        cols: The canvas columns on the reference plane.
        ref_center_v: The reference center v, where the body is drawn, on the
            reference plane: the frame center.
        ref_center_u: The reference center u on the reference plane.
        plane_rows: The rows of the box the body's craters are laid out on: the
            frame, extended to hold the whole body.  It depends on the body's size
            only, never on its position, so a body's craters stay in place on its
            surface whichever part of it is drawn.
        plane_cols: The columns of the same box.
        on_frame: Whether the canvas is the frame itself.
        frame_margin: How many render-grid pixels the image the body is painted into
            extends past the frame on each side; the image's first pixel is the
            reference-plane pixel ``(-frame_margin, -frame_margin)``.
    """

    rows: slice
    cols: slice
    ref_center_v: float
    ref_center_u: float
    plane_rows: slice
    plane_cols: slice
    on_frame: bool
    frame_margin: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """The canvas rows and columns."""
        return self.rows.stop - self.rows.start, self.cols.stop - self.cols.start

    @property
    def origin(self) -> tuple[int, int]:
        """The reference-plane row and column of the canvas's first pixel."""
        return self.rows.start, self.cols.start

    @property
    def center(self) -> tuple[float, float]:
        """The reference center in the canvas's own pixel corner coordinates."""
        return self.ref_center_v - self.rows.start, self.ref_center_u - self.cols.start

    @property
    def is_empty(self) -> bool:
        """Whether the canvas holds no pixel, because the body misses the frame."""
        return bool(self.rows.start >= self.rows.stop or self.cols.start >= self.cols.stop)


def body_canvas(
    frame_shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    reach_v: float,
    reach_u: float,
    oversample: int,
    frame_margin: int = 0,
) -> BodyCanvas:
    """The canvas a body is drawn on before it is translated into the frame.

    A body that lies at least one pixel inside the frame edge when drawn at the frame
    center is drawn on the frame itself, wherever it is placed.  The translation
    reads nothing beyond the drawing, so a body lighting the drawing's outermost
    pixels would lose part of their light there.  Any other body is drawn on the
    part of the reference plane where the body and the pixels the translation into
    the frame and its margin reads overlap, each widened by 8 detector pixels, so the
    drawing never ends inside the frame or the margin.  That part changes with the
    body's position, so such a body is drawn again whenever it moves.  Every canvas
    edge lies on a detector pixel boundary.

    Parameters:
        frame_shape: The frame's rows and columns on the render grid.
        center_v: The body center v in the frame (offset already applied).
        center_u: The body center u in the frame (offset already applied).
        reach_v: How far the body, its relief, and its halo reach from its center
            along v, in render-grid pixels.
        reach_u: How far they reach along u.
        oversample: The render grid's oversampling factor.
        frame_margin: How many render-grid pixels the image the body is painted into
            extends past the frame on each side.

    Returns:
        The body's canvas; an empty one when the body misses the frame and its
        margin.
    """
    size_v, size_u = frame_shape
    ref_v = size_v / 2.0
    ref_u = size_u / 2.0
    frame_rows = slice(0, size_v)
    frame_cols = slice(0, size_u)
    if _fits(ref_v, reach_v, size_v) and _fits(ref_u, reach_u, size_u):
        return BodyCanvas(
            rows=frame_rows,
            cols=frame_cols,
            ref_center_v=ref_v,
            ref_center_u=ref_u,
            plane_rows=frame_rows,
            plane_cols=frame_cols,
            on_frame=True,
            frame_margin=frame_margin,
        )
    step = max(1, int(oversample))
    margin = _CANVAS_MARGIN_DETECTOR_PX * step
    body_rows = _aligned(
        math.floor(ref_v - reach_v) - margin, math.ceil(ref_v + reach_v) + margin, step
    )
    body_cols = _aligned(
        math.floor(ref_u - reach_u) - margin, math.ceil(ref_u + reach_u) + margin, step
    )
    read_rows = _read_span(center_v - ref_v, size_v, margin + frame_margin, step)
    read_cols = _read_span(center_u - ref_u, size_u, margin + frame_margin, step)
    return BodyCanvas(
        rows=_intersect(body_rows, read_rows),
        cols=_intersect(body_cols, read_cols),
        ref_center_v=ref_v,
        ref_center_u=ref_u,
        plane_rows=_union(frame_rows, body_rows),
        plane_cols=_union(frame_cols, body_cols),
        on_frame=False,
        frame_margin=frame_margin,
    )


def _fits(ref: float, reach: float, size: int) -> bool:
    """Whether a body drawn at ``ref`` with this reach lies within ``[1, size - 1]``.

    The pixel of clearance keeps the body off the drawing's outermost pixels, whose
    light the translation would partly drop: it reads nothing beyond the drawing.
    """
    return ref - reach >= 1.0 and ref + reach <= size - 1


def _aligned(start: int, stop: int, step: int) -> slice:
    """The span widened so both ends are multiples of ``step``."""
    return slice((start // step) * step, -((-stop) // step) * step)


def _read_span(delta: float, size: int, margin: int, step: int) -> slice:
    """The reference-plane pixels a translation by ``delta`` reads for the whole frame.

    Parameters:
        delta: The translation along this axis, in pixels.
        size: The frame extent along this axis.
        margin: The pixels added at each end.
        step: The alignment of the span's ends.

    Returns:
        The span, widened by ``margin`` and aligned to ``step``.
    """
    # Frame pixel i samples reference position i - delta, which reads the pixels
    # on each side of it; see the translate module.
    start = math.floor(-delta) - 1 - margin
    stop = math.ceil(size - 1 - delta) + 2 + margin
    return _aligned(start, stop, step)


def _intersect(first: slice, second: slice) -> slice:
    """The overlap of two spans (empty when they do not meet)."""
    start = max(first.start, second.start)
    return slice(start, max(start, min(first.stop, second.stop)))


def _union(first: slice, second: slice) -> slice:
    """The smallest span holding both spans."""
    return slice(min(first.start, second.start), max(first.stop, second.stop))


def finish_single_body(
    img: NDArrayFloatType,
    body_shape: NDArrayFloatType,
    silhouette: NDArrayFloatType,
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    canvas: BodyCanvas,
    range_km: float | None,
    halo: HaloScreen | None = None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Translate a body drawn on its canvas into place and composite it over the image.

    An atmospheric body's translucent halo goes over the image first, as a
    transmission screen (:func:`composite_halo`).  The body then goes over that by
    its coverage: each pixel becomes the body's coverage-weighted light plus the
    uncovered fraction of what was there.  A pixel the body covers wholly takes the
    body's light alone.

    The returned mask is every pixel the outline touches, however little, so the
    body claims its pixels in the truth maps there even where it renders black.  The
    halo is never part of the mask.

    Parameters:
        img: Image array to composite over in place.
        body_shape: Body intensity drawn on the canvas at the reference center,
            weighted by the coverage: 0 wherever the outline covers nothing.
        silhouette: The fraction of each canvas pixel the body's solid outline
            covers, in [0, 1].
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        canvas: The canvas the body was drawn on.
        range_km: The body's distance from the observer, recorded in the
            inventory.  None records 1.0.
        halo: The body's translucent halo screen on the canvas, or None for a
            body without an atmosphere.

    Returns:
        Tuple of (body_mask, body_info_dict).  The info holds the name, the
        inventory item, the body's :class:`BodyCoverage` on the image under
        ``'coverage'``, and, for a body that carries one, the positioned halo screen
        under ``'halo'``.
    """
    dv = center_v - canvas.ref_center_v
    du = center_u - canvas.ref_center_u
    positioned_body, positioned_outline, box = _position_body(
        body_shape, silhouette, canvas, img.shape, dv=dv, du=du
    )
    # The outline is 0 outside the box, so the mask is found inside it alone, which
    # leaves the rest of the outline plane unread.
    mask = np.zeros(positioned_outline.shape, dtype=np.bool_)
    mask[box] = positioned_outline[box] > 0
    coverage = BodyCoverage(
        box_v=box[0], box_u=box[1], coverage=np.clip(positioned_outline[box], 0.0, 1.0)
    )
    positioned_halo = (
        None if halo is None else _position_halo(halo, canvas, img.shape, dv=dv, du=du)
    )
    if positioned_halo is not None:
        composite_halo(img, positioned_halo)
    view = img[box]
    # positioned_body + transmission * view, with one box-sized temporary.
    behind = coverage.transmission * view
    np.add(positioned_body[box], behind, out=view)

    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': _inventory_item(
            center_v,
            center_u,
            half_extent_v=half_extent_v,
            half_extent_u=half_extent_u,
            range_km=range_km,
        ),
        'coverage': coverage,
    }
    if positioned_halo is not None:
        body_info['halo'] = positioned_halo
    return mask, body_info


def composite_halo(img: NDArrayFloatType, halo: HaloScreen) -> None:
    """Composite a halo screen over the image in place.

    Each pixel the screen differs from identity becomes the screen's emission plus
    its transmission times what was there.

    Parameters:
        img: Image array to composite over in place.
        halo: The halo screen, on the image's grid.
    """
    view = img[halo.box_v, halo.box_u]
    emission = halo.emission[halo.box_v, halo.box_u]
    transmission = halo.transmission[halo.box_v, halo.box_u]
    view[halo.mask] = emission[halo.mask] + transmission[halo.mask] * view[halo.mask]


def finish_unseen_body(
    frame_shape: tuple[int, int],
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    range_km: float | None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """The mask and inventory entry of a body that lands wholly outside the frame.

    Parameters:
        frame_shape: The rows and columns of the image the body would be painted
            into, the frame and its margin.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        range_km: The body's distance from the observer, or None to record 1.0.

    Returns:
        An all-False mask and the body info, as :func:`finish_single_body` returns
        them for a body that covers no pixel.
    """
    mask = np.zeros(frame_shape, dtype=np.bool_)
    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': _inventory_item(
            center_v,
            center_u,
            half_extent_v=half_extent_v,
            half_extent_u=half_extent_u,
            range_km=range_km,
        ),
        'coverage': BodyCoverage(
            box_v=slice(0, 0), box_u=slice(0, 0), coverage=np.zeros((0, 0), dtype=np.float64)
        ),
    }
    return mask, body_info


def _position_body(
    body_shape: NDArrayFloatType,
    silhouette: NDArrayFloatType,
    canvas: BodyCanvas,
    frame_shape: tuple[int, int],
    *,
    dv: float,
    du: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, tuple[slice, slice]]:
    """Translate a body's intensity and its outline from its canvas by ``(dv, du)``.

    Parameters:
        body_shape: Body intensity drawn on the canvas at the reference center.
        silhouette: The body's solid-outline coverage on the canvas.
        canvas: The canvas the body was drawn on.
        frame_shape: The rows and columns of the image the body is painted into,
            the frame and its margin.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The translated intensity and the translated outline on the image, and the
        image box outside which both are 0.  For a body drawn on a frame without a
        margin at a zero translation the two planes are the input arrays themselves,
        which the caller only reads.
    """
    lit_box = nonzero_box(body_shape, silhouette)
    # An exactly zero translation is the bitwise identity for the order-1
    # spline, so a body drawn on the frame at the reference center skips the
    # interpolation (the shape is only read, never written).  An image with a
    # margin is larger than the drawing, so the translation places it.
    if canvas.on_frame and canvas.frame_margin == 0 and dv == 0.0 and du == 0.0:
        return body_shape, silhouette, lit_box
    # Only the box the body covers is resampled; see the translate module.
    box_v, box_u = _on_reference_plane(lit_box, canvas)
    out_origin = (-canvas.frame_margin, -canvas.frame_margin)
    positioned_body = translate_box(
        body_shape,
        dv,
        du,
        box_v,
        box_u,
        shape=frame_shape,
        origin=canvas.origin,
        out_origin=out_origin,
    )
    positioned_outline = translate_box(
        silhouette,
        dv,
        du,
        box_v,
        box_u,
        shape=frame_shape,
        origin=canvas.origin,
        out_origin=out_origin,
    )
    # The boxes are found on the reference plane and stated on the image.
    first = -canvas.frame_margin
    ref_box_v = shifted_box(box_v, dv, frame_shape[0], low=first)
    ref_box_u = shifted_box(box_u, du, frame_shape[1], low=first)
    image_box = (
        slice(ref_box_v.start - first, ref_box_v.stop - first),
        slice(ref_box_u.start - first, ref_box_u.stop - first),
    )
    return positioned_body, positioned_outline, image_box


def _on_reference_plane(box: tuple[slice, slice], canvas: BodyCanvas) -> tuple[slice, slice]:
    """A box of canvas pixels as the reference-plane box it covers.

    Parameters:
        box: The box's rows and columns on the canvas.
        canvas: The canvas.

    Returns:
        The box's rows and columns on the reference plane.
    """
    rows, cols = box
    origin_v, origin_u = canvas.origin
    return (
        slice(rows.start + origin_v, rows.stop + origin_v),
        slice(cols.start + origin_u, cols.stop + origin_u),
    )


def _position_halo(
    halo: HaloScreen,
    canvas: BodyCanvas,
    frame_shape: tuple[int, int],
    *,
    dv: float,
    du: float,
) -> HaloScreen:
    """Translate a body's halo screen by ``(dv, du)``.

    The screen reaches a pixel into the body's outline (see
    :func:`~spindoctor.sim.render.bodies.atmosphere.apply_atmosphere`), so a pixel
    the translation blends from the limb and the sky reads glow from both of them.
    The body covers the screen where it lies over it.

    Parameters:
        halo: The halo screen on the canvas.
        canvas: The canvas the body was drawn on.
        frame_shape: The frame's rows and columns.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The positioned halo screen (``halo`` itself for a body drawn on a frame
        without a margin at a zero translation).
    """
    if canvas.on_frame and canvas.frame_margin == 0 and dv == 0.0 and du == 0.0:
        return halo
    size_v, size_u = frame_shape
    first = -canvas.frame_margin
    source_v, source_u = _on_reference_plane((halo.box_v, halo.box_u), canvas)
    emission = translate_box(
        halo.emission,
        dv,
        du,
        source_v,
        source_u,
        shape=frame_shape,
        origin=canvas.origin,
        out_origin=(first, first),
    )
    transmission = translate_box(
        halo.transmission,
        dv,
        du,
        source_v,
        source_u,
        cval=1.0,
        shape=frame_shape,
        origin=canvas.origin,
        out_origin=(first, first),
    )
    # The boxes are found on the reference plane and stated on the image.
    ref_box_v = shifted_box(source_v, dv, size_v, low=first)
    ref_box_u = shifted_box(source_u, du, size_u, low=first)
    box_v = slice(ref_box_v.start - first, ref_box_v.stop - first)
    box_u = slice(ref_box_u.start - first, ref_box_u.stop - first)
    emission_view = emission[box_v, box_u]
    transmission_view = transmission[box_v, box_u]
    halo_mask = (emission_view > 0.0) | (transmission_view < 1.0)
    return HaloScreen(
        emission=emission,
        transmission=transmission,
        box_v=box_v,
        box_u=box_u,
        mask=halo_mask,
    )


def _inventory_item(
    center_v: float,
    center_u: float,
    *,
    half_extent_v: float,
    half_extent_u: float,
    range_km: float | None,
) -> dict[str, float]:
    """The body's inventory entry: its unclipped bounding box, size, and range.

    Parameters:
        center_v: Body center V in the image.
        center_u: Body center U in the image.
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        range_km: The body's distance from the observer, or None to record 1.0.

    Returns:
        The inventory entry.
    """
    return {
        'v_min_unclipped': center_v - half_extent_v,
        'v_max_unclipped': center_v + half_extent_v,
        'u_min_unclipped': center_u - half_extent_u,
        'u_max_unclipped': center_u + half_extent_u,
        'v_pixel_size': 2 * half_extent_v,
        'u_pixel_size': 2 * half_extent_u,
        # The inventory mirrors the oops inventory contract for real frames,
        # whose distance key is named 'range'; the scene schema key is the
        # per-body 'range_km'.
        'range': 1.0 if range_km is None else range_km,
    }
