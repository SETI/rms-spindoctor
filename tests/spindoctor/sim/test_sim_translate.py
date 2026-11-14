"""Box-limited sub-pixel translation of a render-grid plane.

The translation resamples only the box holding the content.  It must give the
same bits as ``scipy.ndimage.shift`` over the whole frame, apart from the fill
outside the box, which is exact where ``shift`` rounds a fill of one.
"""

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.forward.translate import (
    _STRIP_ROWS,
    nonzero_box,
    shifted_box,
    translate_box,
)


def _content(cval: float, rows: slice, cols: slice) -> np.ndarray:
    """A 300 x 280 plane equal to ``cval`` except for random values in a box."""
    plane = np.full((300, 280), cval, dtype=np.float64)
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    plane[rows, cols] = np.random.default_rng(4).uniform(0.0, 1.0, size=shape)
    return plane


@pytest.mark.parametrize(
    ('dv', 'du', 'rows', 'cols'),
    [
        pytest.param(0.37, -0.81, slice(100, 160), slice(90, 200), id='sub_pixel'),
        pytest.param(-23.6, 41.2, slice(100, 160), slice(90, 200), id='many_pixels'),
        pytest.param(5.5, -7.25, slice(0, 40), slice(250, 280), id='frame_edge'),
        pytest.param(-25.3, 40.6, slice(10, 60), slice(220, 280), id='partly_off_the_frame'),
    ],
)
def test_translation_is_the_whole_frame_shift(
    dv: float, du: float, rows: slice, cols: slice
) -> None:
    """With a fill of zero, every pixel of the frame has the whole-frame shift's bits."""
    plane = _content(0.0, rows, cols)
    expected = ndimage.shift(plane, (dv, du), order=1, mode='constant', cval=0.0)
    assert np.array_equal(translate_box(plane, dv, du, rows, cols), expected)


def test_a_box_taller_than_one_strip_is_the_whole_frame_shift() -> None:
    """A box resampled in several strips of rows has the whole-frame shift's bits."""
    rows, cols = slice(5, 295), slice(20, 260)
    assert shifted_box(rows, 0.62, 300).stop - shifted_box(rows, 0.62, 300).start > _STRIP_ROWS
    plane = _content(0.0, rows, cols)
    expected = ndimage.shift(plane, (0.62, -1.4), order=1, mode='constant', cval=0.0)
    assert np.array_equal(translate_box(plane, 0.62, -1.4, rows, cols), expected)


def test_translation_with_a_fill_of_one_is_the_whole_frame_shift() -> None:
    """With a fill of one, the frame matches the whole-frame shift to its last bit."""
    rows, cols = slice(100, 160), slice(90, 200)
    plane = _content(1.0, rows, cols)
    expected = ndimage.shift(plane, (1.3, 2.9), order=1, mode='constant', cval=1.0)
    out = translate_box(plane, 1.3, 2.9, rows, cols, cval=1.0)
    assert float(np.max(np.abs(out - expected))) <= 2.3e-16


def test_translation_outside_the_box_is_the_fill_value() -> None:
    """Outside the translated box the plane is exactly the fill value."""
    rows, cols = slice(100, 160), slice(90, 200)
    plane = _content(1.0, rows, cols)
    out = translate_box(plane, 1.3, 2.9, rows, cols, cval=1.0)
    box = (shifted_box(rows, 1.3, 300), shifted_box(cols, 2.9, 280))
    out[box] = 1.0
    assert np.all(out == 1.0)


def test_an_empty_box_translates_to_the_fill_value() -> None:
    """A plane with no content translates to a plane of the fill value."""
    plane = np.zeros((50, 60), dtype=np.float64)
    rows, cols = nonzero_box(plane)
    out = translate_box(plane, 3.2, -1.7, rows, cols)
    assert np.array_equal(out, plane)
