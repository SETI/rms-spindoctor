"""Translate the content of a render-grid plane by a sub-pixel amount.

A body is drawn at a reference position and then translated into place by
linear (order-1) interpolation.  Its content covers a small box of the frame,
and the plane is constant outside it, so only the box is resampled.

The result is bit-identical to ``scipy.ndimage.shift(plane, (dv, du), order=1,
mode='constant', cval=cval)`` inside the translated box.  ``shift`` samples output
pixel ``i`` at source position ``i - dv``; here each position is computed the
same way from its whole-frame index, then the crop's first index is subtracted,
which is exact for a whole number.  The interpolation weights are therefore the
same bits.  The crop holds every source pixel those positions read, so a crop
edge inside the frame is never reached.  Outside the translated box the result is
``cval`` exactly, where ``shift`` gives ``cval`` up to rounding.
"""

import math

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = ['nonzero_box', 'shifted_box', 'translate_box']


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


def shifted_box(box: slice, delta: float, size: int) -> slice:
    """The frame slice conservatively covering a box translated by ``delta``.

    The order-1 interpolation reads one source pixel to each side, so the
    translated content lies within the source span shifted by ``delta`` and
    widened by one pixel each way, clipped to the frame.

    Parameters:
        box: The source slice.
        delta: The translation along this axis, in pixels.
        size: The frame extent along this axis.

    Returns:
        The translated, widened, clipped slice.
    """
    if box.start >= box.stop:
        return slice(0, 0)
    start = max(math.floor(box.start + delta) - 1, 0)
    stop = min(math.ceil(box.stop + delta) + 1, size)
    if start >= stop:
        return slice(0, 0)
    return slice(start, stop)


def _source_span(out: slice, delta: float, size: int) -> slice:
    """Source pixels the order-1 interpolation reads for an output span, clipped."""
    return slice(
        max(math.floor(out.start - delta) - 1, 0), min(math.ceil(out.stop - 1 - delta) + 2, size)
    )


def translate_box(
    plane: NDArrayFloatType,
    dv: float,
    du: float,
    box_v: slice,
    box_u: slice,
    *,
    cval: float = 0.0,
) -> NDArrayFloatType:
    """Translate a plane whose content lies in a box, by order-1 interpolation.

    Parameters:
        plane: The plane; it must equal ``cval`` outside ``(box_v, box_u)``.
        dv: Translation along v, in pixels.
        du: Translation along u, in pixels.
        box_v: Rows outside which the plane equals ``cval``.
        box_u: Columns outside which the plane equals ``cval``.
        cval: The plane's value outside the box, and the fill beyond the frame.

    Returns:
        A new plane: the translated content, and ``cval`` everywhere else.
    """
    size_v, size_u = plane.shape
    out = np.full(plane.shape, cval, dtype=np.float64)
    out_v = shifted_box(box_v, dv, size_v)
    out_u = shifted_box(box_u, du, size_u)
    if out_v.start >= out_v.stop or out_u.start >= out_u.stop:
        return out
    src_v = _source_span(out_v, dv, size_v)
    src_u = _source_span(out_u, du, size_u)
    coords = np.empty((2, out_v.stop - out_v.start, out_u.stop - out_u.start))
    coords[0] = ((np.arange(out_v.start, out_v.stop, dtype=np.float64) - dv) - src_v.start)[
        :, np.newaxis
    ]
    coords[1] = ((np.arange(out_u.start, out_u.stop, dtype=np.float64) - du) - src_u.start)[
        np.newaxis, :
    ]
    out[out_v, out_u] = ndimage.map_coordinates(
        plane[src_v, src_u], coords, order=1, mode='constant', cval=cval
    )
    return out
