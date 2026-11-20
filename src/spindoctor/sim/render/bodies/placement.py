"""Placement of a rendered body: draw it on a canvas, translate it into the frame.

A body renderer draws its body at a reference center, the frame center, so a shape
cache can serve the same body at any position.  The drawing is made on the
*reference plane*, the frame's pixel grid extended without bound in every
direction, and the box of that plane it is drawn on is the body's canvas
(:func:`body_canvas`).  :func:`finish_single_body` then translates the drawing to
the body's center by linear interpolation, paints its opaque disc into the frame,
positions its translucent halo, and builds the body's inventory entry.  The
ellipsoid path in :mod:`spindoctor.sim.render.bodies.render` and the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`) both place their bodies here.

A body that fits in the frame, with a pixel to spare, when drawn at the frame center
is drawn on the frame itself.  A larger one, which a frame-sized drawing would cut
off, is drawn on a canvas that holds the part of it that lands in the frame, so a
planet larger than the frame shows its limb wherever its center lies, on the frame
or off it.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.atmosphere import HaloScreen
from spindoctor.sim.render.bodies.translate import nonzero_box, shifted_box, translate_box
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'PARTIAL_DRAWINGS_KEPT',
    'BodyCanvas',
    'body_canvas',
    'finish_single_body',
    'finish_unseen_body',
]

# Detector pixels a canvas keeps beyond the part of the body the frame reads.  The
# relief tilt is interpolated by a cubic spline, whose end conditions reach into a
# canvas by a fraction 0.27 per pixel, so 8 pixels hold their effect on the frame
# below 3e-5 of the tilt.
_CANVAS_MARGIN_DETECTOR_PX = 8


PARTIAL_DRAWINGS_KEPT = 4
"""How many drawings on a canvas reaching past the frame a body renderer keeps.

Such a canvas changes whenever the body moves, so an offset sweep never reuses a
drawing, and the cache holds at most this many, each about the size of the frame.
The radiance stage draws every body twice for a scene that smears each layer apart,
once for the composite and once for the layers, so a scene with up to four bodies
this large draws each of them once.
"""


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
    """

    rows: slice
    cols: slice
    ref_center_v: float
    ref_center_u: float
    plane_rows: slice
    plane_cols: slice
    on_frame: bool

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
) -> BodyCanvas:
    """The canvas a body is drawn on before it is translated into the frame.

    A body that lies at least one pixel inside the frame edge when drawn at the frame
    center is drawn on the frame itself, wherever it is placed.  The translation
    reads nothing beyond the drawing, so a body lighting the drawing's outermost
    pixels would lose part of their light there.  Any other body is drawn on the
    part of the reference plane where the body and the pixels the translation into
    the frame reads overlap, each widened by 8 detector pixels, so the drawing never
    ends inside the frame.  That part changes with the body's position, so such a
    body is drawn again whenever it moves.  Every canvas edge lies on a detector
    pixel boundary.

    Parameters:
        frame_shape: The render grid's rows and columns.
        center_v: The body center v in the frame (offset already applied).
        center_u: The body center u in the frame (offset already applied).
        reach_v: How far the body, its relief, and its halo reach from its center
            along v, in render-grid pixels.
        reach_u: How far they reach along u.
        oversample: The render grid's oversampling factor.

    Returns:
        The body's canvas; an empty one when the body misses the frame.
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
        )
    step = max(1, int(oversample))
    margin = _CANVAS_MARGIN_DETECTOR_PX * step
    body_rows = _aligned(
        math.floor(ref_v - reach_v) - margin, math.ceil(ref_v + reach_v) + margin, step
    )
    body_cols = _aligned(
        math.floor(ref_u - reach_u) - margin, math.ceil(ref_u + reach_u) + margin, step
    )
    return BodyCanvas(
        rows=_intersect(body_rows, _read_span(center_v - ref_v, size_v, margin, step)),
        cols=_intersect(body_cols, _read_span(center_u - ref_u, size_u, margin, step)),
        ref_center_v=ref_v,
        ref_center_u=ref_u,
        plane_rows=_union(frame_rows, body_rows),
        plane_cols=_union(frame_cols, body_cols),
        on_frame=False,
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
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    canvas: BodyCanvas,
    range_km: float | None,
    halo: HaloScreen | None = None,
    silhouette: NDArrayFloatType | None = None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Translate a body drawn on its canvas into place and composite it.

    Only the opaque body paints here (last writer wins); an atmospheric
    body's translucent halo is translated alongside it and returned on the
    info dict (key ``'halo'``) for the radiance stage to composite as a
    transmission screen, so the returned mask and the painted pixels are the
    solid silhouette only.  The mask is every pixel the silhouette touches,
    so the body paints, occludes stars and farther bodies, and claims its
    pixels in the truth maps there even where it renders black.

    Parameters:
        img: Image array to modify in-place.
        body_shape: Body intensity drawn on the canvas at the reference center.
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
        silhouette: The body's solid-outline pixel coverage on the canvas, or
            None when every pixel inside the body renders above zero, so
            ``body_shape > 0`` is the outline.

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        the name and the inventory item (plus the positioned halo screen under
        ``'halo'`` when the body carries one).
    """
    dv = center_v - canvas.ref_center_v
    du = center_u - canvas.ref_center_u
    positioned_body, positioned_outline = _position_body(
        body_shape, silhouette, canvas, img.shape, dv=dv, du=du
    )
    mask = positioned_outline > 0
    img[mask] = positioned_body[mask]

    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': _inventory_item(
            center_v,
            center_u,
            half_extent_v=half_extent_v,
            half_extent_u=half_extent_u,
            range_km=range_km,
        ),
    }
    if halo is not None:
        body_info['halo'] = _position_halo(halo, mask, canvas, dv=dv, du=du)
    return mask, body_info


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
        frame_shape: The render grid's rows and columns.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        range_km: The body's distance from the observer, or None to record 1.0.

    Returns:
        An all-False mask and the body info, as :func:`finish_single_body` returns
        them for a body that paints no pixel.
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
    }
    return mask, body_info


def _position_body(
    body_shape: NDArrayFloatType,
    silhouette: NDArrayFloatType | None,
    canvas: BodyCanvas,
    frame_shape: tuple[int, int],
    *,
    dv: float,
    du: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Translate a body's intensity and its outline from its canvas by ``(dv, du)``.

    Parameters:
        body_shape: Body intensity drawn on the canvas at the reference center.
        silhouette: The body's solid-outline coverage on the canvas, or None when
            ``body_shape > 0`` is the outline.
        canvas: The canvas the body was drawn on.
        frame_shape: The frame's rows and columns.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The translated intensity and the translated outline on the frame.  For a
        body drawn on the frame at a zero translation they are the input arrays
        themselves, which the caller only reads.
    """
    # An exactly zero translation is the bitwise identity for the order-1
    # spline, so a body drawn on the frame at the reference center skips the
    # interpolation (the shape is only read, never written).
    if canvas.on_frame and dv == 0.0 and du == 0.0:
        return body_shape, body_shape if silhouette is None else silhouette
    # Only the box the body covers is resampled; see the translate module.
    lit_planes = (body_shape,) if silhouette is None else (body_shape, silhouette)
    box_v, box_u = _on_reference_plane(nonzero_box(*lit_planes), canvas)
    positioned_body = translate_box(
        body_shape, dv, du, box_v, box_u, shape=frame_shape, origin=canvas.origin
    )
    positioned_outline = (
        positioned_body
        if silhouette is None
        else translate_box(
            silhouette, dv, du, box_v, box_u, shape=frame_shape, origin=canvas.origin
        )
    )
    return positioned_body, positioned_outline


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
    halo: HaloScreen, mask: NDArrayBoolType, canvas: BodyCanvas, *, dv: float, du: float
) -> HaloScreen:
    """Translate a body's halo screen by ``(dv, du)``, clear of the body's opaque paint.

    Parameters:
        halo: The halo screen on the canvas.
        mask: The positioned body's opaque mask on the frame.
        canvas: The canvas the body was drawn on.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The positioned halo screen (``halo`` itself for a body drawn on the frame at
        a zero translation).
    """
    if canvas.on_frame and dv == 0.0 and du == 0.0:
        # The screen already excludes every pixel the (unshifted) disc
        # painted, so it is positioned as-is.
        return halo
    size_v, size_u = mask.shape
    source_v, source_u = _on_reference_plane((halo.box_v, halo.box_u), canvas)
    emission = translate_box(
        halo.emission, dv, du, source_v, source_u, shape=mask.shape, origin=canvas.origin
    )
    transmission = translate_box(
        halo.transmission,
        dv,
        du,
        source_v,
        source_u,
        cval=1.0,
        shape=mask.shape,
        origin=canvas.origin,
    )
    box_v = shifted_box(source_v, dv, size_v)
    box_u = shifted_box(source_u, du, size_u)
    # The translation's linear interpolation bleeds the halo one pixel
    # into the painted rim; a screen never overlaps its own opaque
    # paint.  Outside the box the screen is identity already.
    mask_box = mask[box_v, box_u]
    emission_view = emission[box_v, box_u]
    emission_view[mask_box] = 0.0
    transmission_view = transmission[box_v, box_u]
    transmission_view[mask_box] = 1.0
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
