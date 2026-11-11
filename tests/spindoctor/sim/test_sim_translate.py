"""Box-limited sub-pixel translation of a render-grid plane.

The translation resamples only the box holding the content.  Inside the
translated box it must give the same bits as ``scipy.ndimage.shift`` over the
whole frame, and outside it the fill value exactly.
"""

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.forward.translate import nonzero_box, shifted_box, translate_box


def _content(cval: float, rows: slice, cols: slice) -> np.ndarray:
    """A 300 x 280 plane equal to ``cval`` except for random values in a box."""
    plane = np.full((300, 280), cval, dtype=np.float64)
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    plane[rows, cols] = np.random.default_rng(4).uniform(0.0, 1.0, size=shape)
    return plane


@pytest.mark.parametrize(
    ('dv', 'du', 'rows', 'cols', 'cval'),
    [
        pytest.param(0.37, -0.81, slice(100, 160), slice(90, 200), 0.0, id='sub_pixel'),
        pytest.param(-23.6, 41.2, slice(100, 160), slice(90, 200), 0.0, id='many_pixels'),
        pytest.param(5.5, -7.25, slice(0, 40), slice(250, 280), 0.0, id='frame_edge'),
        pytest.param(1.3, 2.9, slice(100, 160), slice(90, 200), 1.0, id='fill_of_one'),
    ],
)
def test_translation_inside_the_box_is_the_whole_frame_shift(
    dv: float, du: float, rows: slice, cols: slice, cval: float
) -> None:
    """Inside the translated box every pixel has the whole-frame shift's exact bits."""
    plane = _content(cval, rows, cols)
    expected = ndimage.shift(plane, (dv, du), order=1, mode='constant', cval=cval)
    out = translate_box(plane, dv, du, rows, cols, cval=cval)
    box = (shifted_box(rows, dv, plane.shape[0]), shifted_box(cols, du, plane.shape[1]))
    assert np.array_equal(out[box], expected[box])


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
