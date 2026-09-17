import math

import numpy as np
import pytest

from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.image import (
    apply_linear_gamma_stretch,
    array_unzoom,
    array_zoom,
    draw_circle,
    draw_line,
    draw_rect,
    filter_downsample,
    next_power_of_2,
    pad_array,
    pad_array_to_power_of_2,
    shift_array,
    unpad_array,
)


def test_shift_array() -> None:
    with pytest.raises(ValueError):
        shift_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert shift_array(arr, (0, 0)) is arr

    zeros = np.zeros(5)
    arr2 = zeros + 1
    shift1 = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    shiftn1 = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    assert np.all(shift_array(arr2, (1,)) == shift1)
    assert np.all(shift_array(arr2, (-1,)) == shiftn1)
    assert np.all(shift_array(arr2, (5,)) == zeros)

    zeros_2 = np.zeros((5, 5))
    arr_2 = zeros_2 + 1
    shift01_2 = np.array([shift1, shift1, shift1, shift1, shift1])
    shift0n1_2 = np.array([shiftn1, shiftn1, shiftn1, shiftn1, shiftn1])
    shift10_2 = np.array([zeros, arr2, arr2, arr2, arr2])
    shiftn10_2 = np.array([arr2, arr2, arr2, arr2, zeros])
    shift11_2 = np.array([zeros, shift1, shift1, shift1, shift1])
    assert np.all(shift_array(arr_2, (0, 1)) == shift01_2)
    assert np.all(shift_array(arr_2, (0, -1)) == shift0n1_2)
    assert np.all(shift_array(arr_2, (1, 0)) == shift10_2)
    assert np.all(shift_array(arr_2, (-1, 0)) == shiftn10_2)
    assert np.all(shift_array(arr_2, (1, 1)) == shift11_2)

    arr1d = np.arange(5.0) + 3
    arr2d = np.array([arr1d, arr1d + 1, arr1d + 2])
    arr3d = np.array([arr2d, arr2d + 10, arr2d + 20])
    exp = [
        [
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
        [
            [20.0, 20.0, 4.0, 5.0, 6.0],
            [20.0, 20.0, 5.0, 6.0, 7.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
        [
            [20.0, 20.0, 14.0, 15.0, 16.0],
            [20.0, 20.0, 15.0, 16.0, 17.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
    ]
    ret = shift_array(arr3d, (1, -1, 2), fill=20)
    assert np.all(ret == exp)


def test_pad_array() -> None:
    with pytest.raises(ValueError):
        pad_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert pad_array(arr, (0, 0)) is arr

    arr1 = np.array([1, 2, 3, 4, 5])
    pad1 = np.array([0, 0, 1, 2, 3, 4, 5, 0, 0])
    arr2 = np.array([arr1, arr1 + 10])
    zero1 = np.array([0, 0, 0, 0, 0])
    assert np.all(pad_array(arr1, (2,)) == pad1)

    pad2a = np.array([zero1, arr1, arr1 + 10, zero1])
    assert np.all(pad_array(arr2, (1, 0)) == pad2a)

    zero10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    pad10 = np.array([0, 0, 11, 12, 13, 14, 15, 0, 0])
    pad2b = np.array([zero10, pad1, pad10, zero10])
    assert np.all(pad_array(arr2, (1, 2)) == pad2b)

    pad2b[pad2b == 0] = 100
    assert np.all(pad_array(arr2, (1, 2), fill=100) == pad2b)


def test_unpad_array() -> None:
    with pytest.raises(ValueError):
        unpad_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert unpad_array(arr, (0, 0)) is arr

    arr1 = np.array([1, 2, 3, 4, 5])
    unpad1 = np.array([3])
    assert np.all(unpad_array(arr1, (2,)) == unpad1)

    arr2 = np.array([arr1, arr1 + 10, arr1 + 20, arr1 + 30, arr1 + 40, arr1 + 50, arr1 + 60])
    unpad2 = np.array([unpad1 + 10, unpad1 + 20, unpad1 + 30, unpad1 + 40, unpad1 + 50])
    assert np.all(unpad_array(arr2, (1, 2)) == unpad2)


def test_next_power_of_2() -> None:
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(4) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(6) == 8
    assert next_power_of_2(7) == 8
    assert next_power_of_2(8) == 8
    # CODE-SUPPORT-003: 0 -> 1 (smallest power of 2 >= 0), negatives rejected.
    assert next_power_of_2(0) == 1
    with pytest.raises(ValueError, match='non-negative'):
        next_power_of_2(-4)


def test_pad_array_to_power_of_2() -> None:
    ret = pad_array_to_power_of_2(np.array([[1], [2]]))
    assert np.all(ret[0] == np.array([[1], [2]]))
    assert ret[1] == (0, 0)

    ret = pad_array_to_power_of_2(np.array([[1, 2], [3, 4]]))
    assert np.all(ret[0] == np.array([[1, 2], [3, 4]]))
    assert ret[1] == (0, 0)

    with pytest.raises(ValueError):
        pad_array_to_power_of_2(np.array([[1, 2, 3], [4, 5, 6]]))

    ret = pad_array_to_power_of_2(
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8, 9],
            ]
        )
    )
    assert np.all(
        ret[0]
        == np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 0],
                [0, 2, 3, 4, 5, 6, 7, 0],
                [0, 3, 4, 5, 6, 7, 8, 0],
                [0, 4, 5, 6, 7, 8, 9, 0],
            ]
        )
    )
    assert ret[1] == (0, 1)

    ret = pad_array_to_power_of_2(
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 0],
                [6, 7, 8, 9, 0, 1],
            ]
        )
    )
    assert np.all(
        ret[0]
        == np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 0],
                [0, 2, 3, 4, 5, 6, 7, 0],
                [0, 3, 4, 5, 6, 7, 8, 0],
                [0, 4, 5, 6, 7, 8, 9, 0],
                [0, 5, 6, 7, 8, 9, 0, 0],
                [0, 6, 7, 8, 9, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    assert ret[1] == (1, 1)


def test_array_zoom() -> None:
    """One source cell fills the block of the zoomed array aligned to that cell."""
    array = np.zeros((4, 5))
    array[1, 2] = 7.0

    zoomed = array_zoom(array, (3, 4))

    assert zoomed.shape == (12, 20)
    marked = np.argwhere(zoomed == 7.0)
    # The source cell's replicated block starts where the cell starts, at rows
    # 1*3 and columns 2*4; it is not centered on the cell's own center.
    assert marked[:, 0].min() == 3
    assert marked[:, 0].max() == 5
    assert marked[:, 1].min() == 8
    assert marked[:, 1].max() == 11
    # Nothing outside that block is written.
    assert np.count_nonzero(zoomed) == 12
    # In pixel-corner coordinates the source cell's center is (1.5, 2.5) and
    # the zoom puts the replicated block's center at (4.5, 10.0), which is the
    # source center scaled by the zoom factor.
    zoomed_center_v = marked[:, 0].mean() + PIXEL_CENTER_TO_CORNER_PX
    zoomed_center_u = marked[:, 1].mean() + PIXEL_CENTER_TO_CORNER_PX
    assert zoomed_center_v == 4.5
    assert zoomed_center_u == 10.0


def test_array_unzoom() -> None:
    """Each unzoomed cell is the mean of its source block, and no other statistic."""
    # This block's mean (9.0) differs from its min (0.0), its max (50.0), its
    # median (4.0), its center sample (4.0) and its first sample (0.0), so only
    # a block mean produces the expected values.
    block = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 50.0], [6.0, 7.0, 8.0]])
    array = np.zeros((6, 6))
    array[0:3, 0:3] = block
    array[0:3, 3:6] = block + 1.0
    array[3:6, 0:3] = block * 2.0
    array[3:6, 3:6] = block - 5.0

    unzoomed = array_unzoom(array, (3, 3))

    assert unzoomed.shape == (2, 2)
    # Each source block lands in the cell at its own block position, so the
    # four distinct means cannot be exchanged.
    assert unzoomed[0, 0] == 9.0
    assert unzoomed[0, 1] == 10.0
    assert unzoomed[1, 0] == 18.0
    assert unzoomed[1, 1] == 4.0


def test_array_unzoom_factor_is_rows_then_columns() -> None:
    """The first unzoom factor divides rows and the second divides columns."""
    array = np.arange(24.0).reshape((4, 6))

    unzoomed = array_unzoom(array, (2, 3))

    # A factor read the other way round would be rejected outright, because 4
    # rows are not divisible by 3.
    assert unzoomed.shape == (2, 2)
    # Rows 0-1 by columns 0-2 holds 0, 1, 2, 6, 7, 8, whose mean is 4.0.
    assert unzoomed[0, 0] == 4.0


def test_filter_local_maximum() -> None:  # TODO: Implement
    ...


def test_filter_sub_median() -> None:  # TODO: Implement
    ...


def test_filter_downsample() -> None:
    """Each downsampled cell is the mean of its amt_y by amt_x block."""
    # The top-left block's mean (10.0) differs from its min (0.0), its max
    # (50.0), its median (2.5) and its first sample (0.0).
    array = np.array(
        [
            [0.0, 1.0, 2.0, 100.0, 100.0, 100.0],
            [3.0, 4.0, 50.0, 100.0, 100.0, 100.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    downsampled = filter_downsample(array, 2, 3)

    assert downsampled.shape == (2, 2)
    assert downsampled[0, 0] == 10.0
    assert downsampled[0, 1] == 100.0
    assert downsampled[1, 0] == 0.0
    # Rows 2-3 by columns 3-5 holds a single 6.0 among five zeros: mean 1.0.
    assert downsampled[1, 1] == 1.0


def test_draw_line() -> None:
    """A horizontal line marks its own row only, with both endpoints included."""
    img = np.zeros((9, 12), dtype=np.float64)

    draw_line(img, 1.0, 2, 3, 7, 3)

    marked = np.argwhere(img != 0)
    # The line's own row is the row named by the argument, so a whole-number
    # coordinate sits at the center of that pixel: it is pixel-centric.
    assert sorted(set(marked[:, 0].tolist())) == [3]
    assert marked[:, 1].min() == 2
    assert marked[:, 1].max() == 7
    # Six columns, 2 through 7 inclusive, with no gaps and nothing beyond.
    assert len(marked) == 6


def test_draw_line_truncates_a_fractional_coordinate() -> None:
    """A fractional coordinate marks the pixel that contains it, not the next one."""
    img = np.zeros((9, 12), dtype=np.float64)

    # In pixel-corner terms 3.9 lies inside pixel 3, nine tenths of the way
    # across it; rounding to the nearer pixel center would reach row 4.
    draw_line(img, 1.0, 2.0, 3.9, 7.0, 3.9)

    marked = np.argwhere(img != 0)
    assert sorted(set(marked[:, 0].tolist())) == [3]


def test_draw_rect_clips_off_image_center() -> None:
    """CODE-ANNO-1: an off-image center must not wrap and paint spurious pixels."""
    img = np.zeros((20, 20), dtype=np.float64)
    # Center far off the top-left corner; a negative slice index would wrap to
    # the opposite edge without clipping.
    draw_rect(img, 1.0, xctr=-50, yctr=-50, xhalfwidth=3, yhalfwidth=3)
    assert not np.any(img)


def test_draw_rect_draws_on_image() -> None:
    """A fully in-bounds rectangle paints its border."""
    img = np.zeros((20, 20), dtype=np.float64)
    draw_rect(img, 1.0, xctr=10, yctr=10, xhalfwidth=3, yhalfwidth=3)
    assert np.any(img)


def test_draw_rect_partial_off_edge_only_paints_in_bounds() -> None:
    """A rectangle straddling an edge paints only in-bounds pixels (no wrap)."""
    img = np.zeros((20, 20), dtype=np.float64)
    draw_rect(img, 1.0, xctr=1, yctr=1, xhalfwidth=5, yhalfwidth=5)
    # The far (bottom-right) edge of the image must remain untouched.
    assert not np.any(img[15:, 15:])


def _test_draw_rect_placeholder() -> None:  # TODO: Implement
    ...


def test_draw_circle() -> None:
    """A circle is centered on the named pixel itself: the center is pixel-centric."""
    x0 = 6
    y0 = 6
    radius = 4
    img = np.zeros((13, 13), dtype=np.float64)

    draw_circle(img, 1.0, x0, y0, radius)

    marked = np.argwhere(img != 0)
    # The marked figure's own centroid is the named pixel. Reading the center
    # as pixel-corner would put the figure half a pixel up and to the left, at
    # a centroid of (5.5, 5.5).
    assert marked[:, 0].mean() == 6.0
    assert marked[:, 1].mean() == 6.0
    # The figure is its own mirror image about row y0 and about column x0,
    # which only holds when those whole numbers are pixel centers.
    window = img[y0 - radius : y0 + radius + 1, x0 - radius : x0 + radius + 1]
    assert np.array_equal(window, window[::-1, :])
    assert np.array_equal(window, window[:, ::-1])
    # The four cardinal points lie exactly radius pixels from the center pixel.
    assert img[y0, x0 + radius] != 0
    assert img[y0, x0 - radius] != 0
    assert img[y0 + radius, x0] != 0
    assert img[y0 - radius, x0] != 0
    # Nothing is marked outside that box.
    assert marked[:, 0].min() == 2
    assert marked[:, 0].max() == 10
    assert marked[:, 1].min() == 2
    assert marked[:, 1].max() == 10


# ---------------------------------------------------------------------------
# apply_linear_gamma_stretch
# ---------------------------------------------------------------------------


def test_apply_linear_gamma_stretch_linear_gamma_one() -> None:
    """gamma=1.0 is a simple linear normalization."""
    data = np.array([0.0, 0.5, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_apply_linear_gamma_stretch_clips_below_black() -> None:
    """Verify values below ``black`` clip to 0.0; zero and mid-range values stay scaled."""
    data = np.array([-1.0, 0.0, 0.5])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.5])


def test_apply_linear_gamma_stretch_clips_above_white() -> None:
    """Verify values above ``white`` clip to 1.0; values at or below ``white`` stay scaled."""
    data = np.array([0.5, 1.0, 2.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.5, 1.0, 1.0])


def test_apply_linear_gamma_stretch_gamma_half_brightens_midtones() -> None:
    """gamma=0.5 maps 0.25 -> 0.5 (brightens relative to linear)."""
    data = np.array([0.0, 0.25, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=0.5)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-7)


def test_apply_linear_gamma_stretch_gamma_two_darkens_midtones() -> None:
    """gamma=2.0 maps 0.5 -> 0.25 (darkens relative to linear)."""
    data = np.array([0.0, 0.5, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=2.0)
    np.testing.assert_allclose(result, [0.0, 0.25, 1.0], atol=1e-7)


def test_apply_linear_gamma_stretch_white_equal_black_uses_epsilon_denominator() -> None:
    """``white == black`` is adjusted so the stretch never divides by zero."""
    data = np.array([5.0, 5.0, 6.0])
    result = apply_linear_gamma_stretch(data, black=5.0, white=5.0, gamma=1.0)
    assert np.all((result >= 0.0) & (result <= 1.0))
    np.testing.assert_array_less(result[:2], result[2])


def test_apply_linear_gamma_stretch_white_less_than_black_is_silently_clipped() -> None:
    """Inverted black/white is treated as ``white`` just above ``black``."""
    data = np.array([0.0, 0.5, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.5, white=0.25, gamma=1.0)
    assert np.all((result >= 0.0) & (result <= 1.0))
    expected = apply_linear_gamma_stretch(
        data,
        black=0.5,
        white=math.nextafter(0.5, math.inf),
        gamma=1.0,
    )
    np.testing.assert_allclose(result, expected)


def test_apply_linear_gamma_stretch_rejects_non_positive_gamma() -> None:
    data = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match='gamma must be greater than 0'):
        apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=0.0)


def test_apply_linear_gamma_stretch_rejects_negative_gamma() -> None:
    data = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match='gamma must be greater than 0'):
        apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=-0.5)


def test_apply_linear_gamma_stretch_rejects_non_finite_black() -> None:
    data = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match='black must be a finite number'):
        apply_linear_gamma_stretch(data, black=float('nan'), white=1.0, gamma=1.0)


def test_apply_linear_gamma_stretch_rejects_bool_black() -> None:
    data = np.array([0.0, 1.0])
    with pytest.raises(TypeError, match='black must be int or float, not bool'):
        apply_linear_gamma_stretch(data, black=True, white=1.0, gamma=1.0)
