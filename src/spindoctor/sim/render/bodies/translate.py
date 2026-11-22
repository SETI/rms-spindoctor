"""Translate the content of a render-grid plane by a sub-pixel amount.

A body is drawn at a reference position and then translated into place by
linear (order-1) interpolation.  Its content covers a small box of the plane it
is drawn on, and the plane is constant outside it, so only the box is resampled.
The output is a box of the reference plane too: the frame, or the frame and the
margin around it that the render pipeline draws for the optics stage.

The plane a body is drawn on need not be the frame.  It is a box of the
*reference plane*, the frame's pixel grid extended without bound in every
direction, so a body larger than the frame can be drawn whole or in the part that
lands in the frame.  The plane's first pixel sits at ``origin`` on the reference
plane, and the translated content lands on a frame of its own size.  A plane drawn
on the frame itself has origin ``(0, 0)`` and the frame's size.

On a plane drawn on the frame itself the result is bit-identical to
``scipy.ndimage.shift(plane, (dv, du), order=1, mode='constant', cval=cval)``
inside the translated box.  ``shift`` samples output pixel ``i`` at source position
``i - dv``.  Here each position is computed the same way, from the pixel's index in
the whole frame, and then the first reference-plane index of the crop it reads is
subtracted.  Subtracting a whole number is exact, so the interpolation weights match
``shift``'s to the last bit, and they are the same wherever the plane lies on the
reference plane.  The crop includes every source pixel the interpolation reads, so
it never reaches a crop edge inside the frame unless the plane itself ends there.
Outside the translated box the result is exactly ``cval``, where ``shift`` gives
``cval`` up to rounding.

The box is resampled a strip of rows at a time, so the table of sample positions
stays small when a body fills the frame.  Each output pixel is computed on its
own, so the strips give the same bits as one pass over the whole box.
"""

import math

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = ['nonzero_box', 'shifted_box', 'translate_box']

# Output rows resampled per pass.  The table of sample positions for one pass holds
# two numbers per pixel of the strip, about 17 MB on a 4096-pixel-wide grid.
_STRIP_ROWS = 256


def nonzero_box(*planes: NDArrayFloatType) -> tuple[slice, slice]:
    """The bounding box of the pixels that are nonzero in any of the planes.

    Parameters:
        *planes: Planes of one shape.

    Returns:
        The box's rows and columns; empty slices when every pixel is zero.
    """
    lit = planes[0] != 0.0
    for plane in planes[1:]:
        lit |= plane != 0.0
    rows = np.flatnonzero(lit.any(axis=1))
    if rows.size == 0:
        return slice(0, 0), slice(0, 0)
    cols = np.flatnonzero(lit.any(axis=0))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def shifted_box(box: slice, delta: float, size: int, *, low: int = 0) -> slice:
    """The frame slice conservatively covering a box translated by ``delta``.

    The order-1 interpolation reads one source pixel to each side, so the
    translated content lies within the source span shifted by ``delta`` and
    widened by one pixel each way, clipped to the frame.

    Parameters:
        box: The source slice, on the reference plane.
        delta: The translation along this axis, in pixels.
        size: The frame extent along this axis.
        low: The reference-plane index of the frame's first pixel along this axis.

    Returns:
        The translated, widened, clipped slice, on the reference plane.
    """
    if box.start >= box.stop:
        return slice(0, 0)
    start = max(math.floor(box.start + delta) - 1, low)
    stop = min(math.ceil(box.stop + delta) + 1, low + size)
    if start >= stop:
        return slice(0, 0)
    return slice(start, stop)


def _source_span(out: slice, delta: float, low: int, high: int) -> slice:
    """The reference-plane pixels order-1 interpolation reads for an output span.

    Parameters:
        out: The output span on the frame.
        delta: The translation along this axis, in pixels.
        low: The first reference-plane index the source plane holds.
        high: One past the last reference-plane index the source plane holds.

    Returns:
        The span read, clipped to the source plane, on the reference plane.
    """
    return slice(
        max(math.floor(out.start - delta) - 1, low),
        min(math.ceil(out.stop - 1 - delta) + 2, high),
    )


def translate_box(
    plane: NDArrayFloatType,
    dv: float,
    du: float,
    box_v: slice,
    box_u: slice,
    *,
    cval: float = 0.0,
    shape: tuple[int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
    out_origin: tuple[int, int] = (0, 0),
) -> NDArrayFloatType:
    """Translate a plane whose content lies in a box, by order-1 interpolation.

    Each output pixel is computed from its reference-plane index, so a pixel comes out
    the same wherever the output starts on the reference plane.

    Parameters:
        plane: The plane; it must equal ``cval`` outside ``(box_v, box_u)``.  Its
            pixel ``(i, j)`` is the reference-plane pixel ``(i + origin[0], j +
            origin[1])``.
        dv: Translation along v, in pixels.
        du: Translation along u, in pixels.
        box_v: The rows that hold the content, on the reference plane; the plane
            equals ``cval`` outside them.
        box_u: The columns that hold the content, on the reference plane; the plane
            equals ``cval`` outside them.
        cval: The plane's value outside the box, and the fill beyond the plane.
        shape: The frame the content lands on; None for the plane's own shape.
        origin: The reference-plane row and column of the plane's first pixel.
        out_origin: The reference-plane row and column of the output's first pixel:
            ``(0, 0)`` for the frame, or minus the margin for a frame with a margin.

    Returns:
        A new frame-sized plane: the translated content, and ``cval`` everywhere
        else.
    """
    size_v, size_u = plane.shape if shape is None else shape
    origin_v, origin_u = origin
    first_v, first_u = out_origin
    out = np.full((size_v, size_u), cval, dtype=np.float64)
    out_v = shifted_box(box_v, dv, size_v, low=first_v)
    out_u = shifted_box(box_u, du, size_u, low=first_u)
    if out_v.start >= out_v.stop or out_u.start >= out_u.stop:
        return out
    out_cols = slice(out_u.start - first_u, out_u.stop - first_u)
    src_u = _source_span(out_u, du, origin_u, origin_u + plane.shape[1])
    cols = (np.arange(out_u.start, out_u.stop, dtype=np.float64) - du) - src_u.start
    plane_cols = slice(src_u.start - origin_u, src_u.stop - origin_u)
    for first in range(out_v.start, out_v.stop, _STRIP_ROWS):
        strip = slice(first, min(first + _STRIP_ROWS, out_v.stop))
        src_v = _source_span(strip, dv, origin_v, origin_v + plane.shape[0])
        rows = (np.arange(strip.start, strip.stop, dtype=np.float64) - dv) - src_v.start
        coords = np.empty((2, rows.size, cols.size))
        coords[0] = rows[:, np.newaxis]
        coords[1] = cols[np.newaxis, :]
        out[strip.start - first_v : strip.stop - first_v, out_cols] = ndimage.map_coordinates(
            plane[src_v.start - origin_v : src_v.stop - origin_v, plane_cols],
            coords,
            order=1,
            mode='constant',
            cval=cval,
        )
    return out
