"""Tests for the manual navigation dialog's cursor sampling.

The dialog prints a continuous cursor position and, beside it, the image and
model values at that position.  The position is pixel corner, because that is
what a Qt event position and the geometry layer both state, while the array it
samples is addressed pixel centric, so the sample has to be taken at the place
the printed number names and nowhere else.

Qt runs headless (offscreen platform).
"""

import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6 not available: {exc}', allow_module_level=True)

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(f'QApplication failed: {exc}', allow_module_level=True)

from spindoctor.ui.manual_nav_dialog import _bilinear_interpolate_fov


def _ramp(rows: int, cols: int) -> np.ndarray:
    """A plane whose value at pixel (v, u) is ``100 * v + u``."""
    v_grid, u_grid = np.mgrid[0:rows, 0:cols]
    ramp: np.ndarray = (100.0 * v_grid + u_grid).astype(np.float64)
    return ramp


def test_pixel_center_returns_that_pixel_value() -> None:
    """Sampling the center of a pixel returns that pixel's own value."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 3.5, 2.5) == pytest.approx(302.0, abs=1e-9)


def test_pixel_boundary_averages_the_two_pixels() -> None:
    """Sampling the boundary between two columns averages them evenly."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 3.5, 3.0) == pytest.approx(302.5, abs=1e-9)


def test_quarter_way_across_a_boundary_weights_the_nearer_pixel() -> None:
    """A position a quarter past a boundary weights the pixel it is in."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 3.5, 3.25) == pytest.approx(302.75, abs=1e-9)


def test_row_boundary_averages_the_two_rows() -> None:
    """Sampling the boundary between two rows averages them evenly."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 3.0, 2.5) == pytest.approx(252.0, abs=1e-9)


def test_first_half_pixel_holds_the_edge_value() -> None:
    """Inside the frame's first half pixel the edge value is held, not wrapped."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 0.1, 0.1) == pytest.approx(0.0, abs=1e-9)


def test_last_half_pixel_holds_the_edge_value() -> None:
    """Inside the frame's last half pixel the edge value is held."""
    arr = _ramp(6, 7)
    assert _bilinear_interpolate_fov(arr, 5.9, 6.9) == pytest.approx(506.0, abs=1e-9)
