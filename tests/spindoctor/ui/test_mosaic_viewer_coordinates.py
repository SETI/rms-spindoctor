"""Coordinate-system tests for the mosaic viewer's readouts and tick placement.

Both reprojectors define grid cell ``k`` as the point sample taken at
``origin + k * resolution``, and the viewer paints cell ``k`` into the display
rectangle ``[k, k + 1)``.  The cell's sample therefore sits at pixel corner
``k + 0.5``, so a coordinate read off the screen, a tick drawn on an axis and an
array lookup made beside them all have to agree about where that sample is.

Qt runs headless (offscreen platform).
"""

import os
from typing import cast

import numpy as np
import numpy.ma as ma
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

from spindoctor.ui.mosaic_viewer.tiled_image_widget import TiledImageWidget

# Ring grid: 5 radial rows of 10 km from 100 km, 8 longitude columns of 2 deg
# from 6 deg.  The radial span is an exact multiple of the resolution, so the
# array's middle row lands on the mean radius and the display's row-to-radius
# mapping is exact.
_N_ROWS = 5
_N_COLS = 8
_RAD_RES = 10.0
_LON_RES = 2.0
_RAD_INNER = 100.0
_LON_ORIGIN = 6.0
_RAD_MID = _RAD_INNER + (_N_ROWS - 1) / 2.0 * _RAD_RES

# Body grid: a full sphere at 45 deg, so 5 latitude rows and 9 longitude columns.
_BODY_RES = 45.0
_BODY_N_LAT = 5
_BODY_N_LON = 9


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def ring_widget(qapp: QApplication) -> TiledImageWidget:
    """A widget loaded with the ring grid described in the module docstring."""
    widget = TiledImageWidget()
    img = ma.MaskedArray(
        np.arange(_N_ROWS * _N_COLS, dtype=np.float64).reshape(_N_ROWS, _N_COLS), mask=False
    )
    widget.set_image(
        img,
        x_interval=_LON_RES,
        y_interval=_RAD_RES,
        y_flip=True,
        x_origin_deg=_LON_ORIGIN,
        x_axis_max=_LON_ORIGIN + _N_COLS * _LON_RES,
        ring_radial_axis_absolute=True,
        ring_radial_mid_km=_RAD_MID,
    )
    widget.set_axis_tick_options(True, True, _RAD_MID, y_tick_labels_absolute=True)
    return widget


@pytest.fixture
def body_widget(qapp: QApplication) -> TiledImageWidget:
    """A widget loaded with the full-sphere body grid described above."""
    widget = TiledImageWidget()
    img = ma.MaskedArray(
        np.arange(_BODY_N_LAT * _BODY_N_LON, dtype=np.float64).reshape(_BODY_N_LAT, _BODY_N_LON),
        mask=False,
    )
    widget.set_image(
        img,
        x_interval=_BODY_RES,
        y_interval=_BODY_RES,
        y_flip=False,
        body_full_sphere_canvas=True,
        body_lon_range_deg=(0.0, 360.0),
        body_lat_range_deg=(-90.0, 90.0),
    )
    return widget


def test_column_centre_reads_that_column_longitude(ring_widget: TiledImageWidget) -> None:
    """The center of a painted column reads the longitude that column samples."""
    for col in range(_N_COLS):
        lon, _rad = ring_widget.pixel_to_physical(col + 0.5, 0.5)
        assert lon == pytest.approx(_LON_ORIGIN + col * _LON_RES, abs=1e-9)


def test_column_boundary_reads_half_a_column_short(ring_widget: TiledImageWidget) -> None:
    """The boundary between two columns reads half a column below the right one."""
    lon, _rad = ring_widget.pixel_to_physical(3.0, 0.5)
    assert lon == pytest.approx(_LON_ORIGIN + 3 * _LON_RES - _LON_RES / 2.0, abs=1e-9)


def test_row_centre_reads_that_row_radius(ring_widget: TiledImageWidget) -> None:
    """The center of a painted row reads the radius that array row samples."""
    for arr_row in range(_N_ROWS):
        pixel_y = (_N_ROWS - 1 - arr_row) + 0.5
        _lon, rad = ring_widget.pixel_to_physical(0.5, pixel_y)
        assert rad == pytest.approx(_RAD_INNER + arr_row * _RAD_RES, abs=1e-9)


def test_readout_radius_belongs_to_the_row_sampled(ring_widget: TiledImageWidget) -> None:
    """The radius printed and the array row read for the value are the same row."""
    pixel_y = 1.25
    _lon, rad = ring_widget.pixel_to_physical(0.5, pixel_y)
    arr_row = ring_widget.pixel_y_to_arr_row(pixel_y)
    assert abs(rad - (_RAD_INNER + arr_row * _RAD_RES)) < _RAD_RES / 2.0


def test_column_sampled_is_the_one_under_the_cursor(ring_widget: TiledImageWidget) -> None:
    """The column read for the value is the column the cursor is painted over."""
    for col in range(_N_COLS):
        assert ring_widget.pixel_x_to_arr_col(col + 0.5, 0.5) == col


def test_column_lookup_holds_to_the_left_edge_of_a_column(ring_widget: TiledImageWidget) -> None:
    """Just inside a column's left edge the value still comes from that column."""
    assert ring_widget.pixel_x_to_arr_col(2.01, 0.5) == 2


def test_column_lookup_holds_to_the_right_edge_of_a_column(ring_widget: TiledImageWidget) -> None:
    """Just inside a column's right edge the value still comes from that column."""
    assert ring_widget.pixel_x_to_arr_col(2.99, 0.5) == 2


def test_y_tick_lands_on_the_centre_of_its_row(ring_widget: TiledImageWidget) -> None:
    """A radius tick is drawn through the middle of the row carrying that radius."""
    for arr_row in range(_N_ROWS):
        radius = _RAD_INNER + arr_row * _RAD_RES
        screen_y = ring_widget._y_physical_to_screen_y(radius, 4.0, 0)
        expected = ((_N_ROWS - 1 - arr_row) + 0.5) * 4.0
        assert screen_y == pytest.approx(expected, abs=1e-9)


def test_y_tick_mapping_inverts_the_y_readout(ring_widget: TiledImageWidget) -> None:
    """The tick placement is the exact inverse of the radius readout."""
    pixel_y = 2.375
    radius = ring_widget._pixel_y_to_y_physical(pixel_y)
    assert ring_widget._y_physical_to_screen_y(radius, 1.0, 0) == pytest.approx(pixel_y, abs=1e-9)


def test_x_tick_lands_on_the_centre_of_its_column(ring_widget: TiledImageWidget) -> None:
    """A longitude tick is drawn through the middle of the column carrying it."""
    for col in range(_N_COLS):
        lon = _LON_ORIGIN + col * _LON_RES
        assert ring_widget._x_physical_to_pixel_x(lon) == pytest.approx(col + 0.5, abs=1e-9)


def test_body_cell_centre_reads_that_cell_longitude(body_widget: TiledImageWidget) -> None:
    """The center of a body canvas cell reads the longitude that column samples."""
    lon, _lat = body_widget.pixel_to_physical(3.5, 0.5)
    assert lon == pytest.approx(3 * _BODY_RES, abs=1e-9)


def test_body_cell_centre_reads_that_cell_latitude(body_widget: TiledImageWidget) -> None:
    """The center of a body canvas cell reads the latitude that row samples."""
    _lon, lat = body_widget.pixel_to_physical(0.5, 2.5)
    assert lat == pytest.approx(90.0 - 2 * _BODY_RES, abs=1e-9)


def test_body_lookup_takes_the_nearest_latitude_row(body_widget: TiledImageWidget) -> None:
    """A latitude past the midpoint between two rows reads the nearer one."""
    _dc, dr, inside = body_widget.body_sphere_data_indices(0.0, -90.0 + 0.6 * _BODY_RES)
    assert inside
    assert dr == 1


def test_body_lookup_keeps_a_latitude_short_of_the_midpoint(
    body_widget: TiledImageWidget,
) -> None:
    """A latitude short of the midpoint between two rows reads the lower one."""
    _dc, dr, inside = body_widget.body_sphere_data_indices(0.0, -90.0 + 0.4 * _BODY_RES)
    assert inside
    assert dr == 0


def test_body_lookup_takes_the_nearest_longitude_column(body_widget: TiledImageWidget) -> None:
    """A longitude past the midpoint between two columns reads the nearer one."""
    dc, _dr, inside = body_widget.body_sphere_data_indices(2.6 * _BODY_RES, 0.0)
    assert inside
    assert dc == 3


def _painted_row_of_marked_latitude(qapp: QApplication, marked_row: int) -> list[int]:
    """Paint the body canvas with one white data row and return its gray column."""
    widget = TiledImageWidget()
    data = np.zeros((_BODY_N_LAT, _BODY_N_LON), dtype=np.float64)
    data[marked_row, :] = 1.0
    widget.set_image(
        ma.MaskedArray(data, mask=False),
        x_interval=_BODY_RES,
        y_interval=_BODY_RES,
        y_flip=False,
        body_full_sphere_canvas=True,
        body_lon_range_deg=(0.0, 360.0),
        body_lat_range_deg=(-90.0, 90.0),
    )
    qimg = widget.render_viewport_to_image()
    # The canvas loads unzoomed and unscrolled, so canvas cell (0, row) is
    # screen pixel (0, row).
    return [(qimg.pixel(0, row) >> 16) & 0xFF for row in range(_BODY_N_LAT)]


def test_canvas_top_row_paints_the_north_pole_row(qapp: QApplication) -> None:
    """The canvas's first row paints the data row at +90 deg, not the one below it."""
    grays = _painted_row_of_marked_latitude(qapp, _BODY_N_LAT - 1)
    assert grays[0] == 255


def test_canvas_bottom_row_paints_the_south_pole_row(qapp: QApplication) -> None:
    """The canvas's last row paints the data row at -90 deg rather than no data."""
    grays = _painted_row_of_marked_latitude(qapp, 0)
    assert grays[_BODY_N_LAT - 1] == 255


def test_cursor_position_scales_without_quantizing(ring_widget: TiledImageWidget) -> None:
    """Sub-pixel cursor motion is sub-pixel display motion, not nothing.

    Both measures are pixel corner and the zoom is a pure scale, so the display
    coordinate follows the cursor continuously; taking a whole viewport pixel on
    the way in would swallow the move entirely.
    """
    ring_widget.set_zoom(100.0, 100.0)
    x_zoom, _y_zoom = ring_widget.get_zoom()
    px_lo, _py = ring_widget.viewport_to_pixel(50.0, 0.0)
    px_hi, _py = ring_widget.viewport_to_pixel(50.5, 0.0)
    assert px_hi - px_lo == pytest.approx(0.5 / x_zoom, abs=1e-9)
