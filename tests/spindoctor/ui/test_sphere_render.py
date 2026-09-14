"""Tests for :func:`spindoctor.ui.mosaic_viewer.sphere_render.render_to_image`."""

import os

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

from PyQt6.QtGui import QImage

from spindoctor.ui.mosaic_viewer.sphere_render import render_to_image
from spindoctor.ui.mosaic_viewer.tiled_image_widget import _body_sphere_lon_bin_to_dc_map

# ---------------------------------------------------------------------------
# Shared grid constants
# ---------------------------------------------------------------------------

_LON_MIN = 0.0
_LON_MAX = 360.0
_D_LON = 45.0
_LAT_MIN = -90.0
_LAT_MAX = 90.0
_D_LAT = 45.0
_N_DATA_ROWS = 4
_N_DATA_COLS = 8

# Viewport dimensions
_VP = 20

# Longitude bin map for 0-360 deg at ``_D_LON`` with ``_N_DATA_COLS`` columns (matches helper).
_N_FULL_LON, _, _LON_BIN_TO_DC = _body_sphere_lon_bin_to_dc_map(
    _LON_MIN, _LON_MAX, _D_LON, _N_DATA_COLS
)


def _viewport_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon_deg, lat_deg, valid) for a 20x20 equirectangular viewport."""
    lon_1d = np.linspace(0.0, 359.0, _VP)
    lat_1d = np.linspace(89.0, -89.0, _VP)
    lon_deg = np.tile(lon_1d, (_VP, 1)).astype(np.float64)
    lat_deg = np.tile(lat_1d, (_VP, 1)).T.astype(np.float64)
    valid = np.ones((_VP, _VP), dtype=bool)
    return lon_deg, lat_deg, valid


def _render(
    image_ma: ma.MaskedArray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    valid: np.ndarray,
    *,
    color_tint: np.ndarray | None = None,
) -> QImage:
    """Call :func:`render_to_image` with shared test constants."""
    return render_to_image(
        image_ma,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        valid=valid,
        lon_min_deg=_LON_MIN,
        lat_min_deg=_LAT_MIN,
        d_lon_deg=_D_LON,
        d_lat_deg=_D_LAT,
        lon_bin_to_dc=_LON_BIN_TO_DC,
        n_full_lon=_N_FULL_LON,
        n_data_rows=_N_DATA_ROWS,
        n_data_cols=_N_DATA_COLS,
        black=0.0,
        white=1.0,
        gamma=1.0,
        color_tint=color_tint,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_render_produces_correct_width() -> None:
    """A 4x8 mosaic rendered to 20x20 produces a QImage with width 20."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)
    assert qimg.width() == _VP


def test_basic_render_produces_correct_height() -> None:
    """A 4x8 mosaic rendered to 20x20 produces a QImage with height 20."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)
    assert qimg.height() == _VP


def test_basic_render_produces_rgb888_format() -> None:
    """Output QImage must use Format_RGB888."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)
    assert qimg.format() == QImage.Format.Format_RGB888


def test_invalid_pixels_are_black_red() -> None:
    """Pixels where valid=False must have red channel 0."""
    lon_deg, lat_deg, _ = _viewport_grids()
    # Mark the left half of the viewport invalid
    valid = np.ones((_VP, _VP), dtype=bool)
    valid[:, : _VP // 2] = False
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    for row in range(_VP):
        for col in range(_VP // 2):
            pixel = qimg.pixel(col, row)
            r = (pixel >> 16) & 0xFF
            assert r == 0


def test_invalid_pixels_are_black_green() -> None:
    """Pixels where valid=False must have green channel 0."""
    lon_deg, lat_deg, _ = _viewport_grids()
    valid = np.ones((_VP, _VP), dtype=bool)
    valid[:, : _VP // 2] = False
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    for row in range(_VP):
        for col in range(_VP // 2):
            pixel = qimg.pixel(col, row)
            g = (pixel >> 8) & 0xFF
            assert g == 0


def test_invalid_pixels_are_black_blue() -> None:
    """Pixels where valid=False must have blue channel 0."""
    lon_deg, lat_deg, _ = _viewport_grids()
    valid = np.ones((_VP, _VP), dtype=bool)
    valid[:, : _VP // 2] = False
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    for row in range(_VP):
        for col in range(_VP // 2):
            pixel = qimg.pixel(col, row)
            b = pixel & 0xFF
            assert b == 0


def test_masked_data_renders_as_dark_red_r() -> None:
    """Data-masked pixels inside the projection must render with R=180."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    # Mask the entire first data row so projected pixels landing there turn dark-red
    mask = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=bool)
    mask[0, :] = True
    image_ma = ma.MaskedArray(data, mask=mask)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    # Find at least one pixel that maps to data row 0
    # data row 0 covers lat_min_deg .. lat_min_deg + d_lat_deg = -90 .. -45
    # viewport lat goes from 89 (top) to -89 (bottom), so bottom rows map there
    found_dark_red = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            if _LAT_MIN <= lat < _LAT_MIN + _D_LAT:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                if r == 180:
                    found_dark_red = True
                    break
        if found_dark_red:
            break
    assert found_dark_red


def test_masked_data_renders_as_dark_red_g() -> None:
    """Data-masked pixels inside the projection must render with G=0."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    mask = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=bool)
    mask[0, :] = True
    image_ma = ma.MaskedArray(data, mask=mask)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    found_dark_red_g = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            if _LAT_MIN <= lat < _LAT_MIN + _D_LAT:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                if r == 180 and g == 0:
                    found_dark_red_g = True
                    break
        if found_dark_red_g:
            break
    assert found_dark_red_g


def test_masked_data_renders_as_dark_red_b() -> None:
    """Data-masked pixels inside the projection must render with B=0."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    mask = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=bool)
    mask[0, :] = True
    image_ma = ma.MaskedArray(data, mask=mask)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    found_dark_red_b = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            if _LAT_MIN <= lat < _LAT_MIN + _D_LAT:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                b = pixel & 0xFF
                if r == 180 and b == 0:
                    found_dark_red_b = True
                    break
        if found_dark_red_b:
            break
    assert found_dark_red_b


def test_stretch_full_range_bright_pixels_near_255() -> None:
    """Checkerboard 1.0 cells stretch to ~255 with black=0, white=1, gamma=1."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    data[::2, ::2] = 1.0
    data[1::2, 1::2] = 1.0
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    # Sample a viewport pixel that maps to a bright data cell
    # data col 0 covers lon 0..45, data row 2 covers lat -45..0
    # Find a valid viewport pixel pointing at (dr=2, dc=0)
    found_bright = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            lon = lon_deg[row, col]
            dr = int((lat - _LAT_MIN) / _D_LAT)
            dc = int(lon / _D_LON)
            if 0 <= dr < _N_DATA_ROWS and 0 <= dc < _N_DATA_COLS and data[dr, dc] == 1.0:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                if r > 200:
                    found_bright = True
                    break
        if found_bright:
            break
    assert found_bright


def test_stretch_full_range_dark_pixels_near_0() -> None:
    """Checkerboard 0.0 cells stretch to ~0 with black=0, white=1, gamma=1."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    data[::2, ::2] = 1.0
    data[1::2, 1::2] = 1.0
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(image_ma, lon_deg, lat_deg, valid)

    found_dark = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            lon = lon_deg[row, col]
            dr = int((lat - _LAT_MIN) / _D_LAT)
            dc = int(lon / _D_LON)
            if 0 <= dr < _N_DATA_ROWS and 0 <= dc < _N_DATA_COLS and data[dr, dc] == 0.0:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                if r < 55:
                    found_dark = True
                    break
        if found_dark:
            break
    assert found_dark


def test_color_tint_top_half_red_dominant() -> None:
    """Top-half data rows with red tint produce output pixels where R > B.

    Regression test for the per-pixel color_tint refactor.
    """
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.full((_N_DATA_ROWS, _N_DATA_COLS), 0.5, dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    color_tint = np.zeros((_N_DATA_ROWS, _N_DATA_COLS, 3), dtype=np.float32)
    half = _N_DATA_ROWS // 2
    color_tint[:half, :, 0] = 1.0  # top half: red
    color_tint[half:, :, 2] = 1.0  # bottom half: blue

    qimg = _render(image_ma, lon_deg, lat_deg, valid, color_tint=color_tint)

    # Find a valid pixel that maps to a top-half data row (dr < half)
    found_red_dominant = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            dr = int((lat - _LAT_MIN) / _D_LAT)
            if 0 <= dr < half:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                b = pixel & 0xFF
                if r > b:
                    found_red_dominant = True
                    break
        if found_red_dominant:
            break
    assert found_red_dominant


def test_color_tint_bottom_half_blue_dominant() -> None:
    """Bottom-half data rows with blue tint produce output pixels where B > R.

    Regression test for the per-pixel color_tint refactor.
    """
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.full((_N_DATA_ROWS, _N_DATA_COLS), 0.5, dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    color_tint = np.zeros((_N_DATA_ROWS, _N_DATA_COLS, 3), dtype=np.float32)
    half = _N_DATA_ROWS // 2
    color_tint[:half, :, 0] = 1.0  # top half: red
    color_tint[half:, :, 2] = 1.0  # bottom half: blue

    qimg = _render(image_ma, lon_deg, lat_deg, valid, color_tint=color_tint)

    found_blue_dominant = False
    for row in range(_VP):
        for col in range(_VP):
            lat = lat_deg[row, col]
            dr = int((lat - _LAT_MIN) / _D_LAT)
            if half <= dr < _N_DATA_ROWS:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                b = pixel & 0xFF
                if b > r:
                    found_blue_dominant = True
                    break
        if found_blue_dominant:
            break
    assert found_blue_dominant


def _qimage_bytes(qimg: QImage) -> bytes:
    """Extract raw pixel bytes from a QImage in a PyQt6-compatible way."""
    bits = qimg.bits()
    assert bits is not None, 'QImage.bits() returned None for a non-null image'
    return bits.asstring(qimg.sizeInBytes())


def test_deterministic_rendering_same_bytes() -> None:
    """Two identical calls must produce byte-for-byte identical output."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.random.default_rng(42).random((_N_DATA_ROWS, _N_DATA_COLS)).astype(np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    qimg1 = _render(image_ma, lon_deg, lat_deg, valid)
    qimg2 = _render(image_ma, lon_deg, lat_deg, valid)

    assert _qimage_bytes(qimg1) == _qimage_bytes(qimg2)


def test_deterministic_rendering_same_size() -> None:
    """Two identical calls must produce output of the same byte length."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.random.default_rng(42).random((_N_DATA_ROWS, _N_DATA_COLS)).astype(np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    qimg1 = _render(image_ma, lon_deg, lat_deg, valid)
    qimg2 = _render(image_ma, lon_deg, lat_deg, valid)

    assert len(_qimage_bytes(qimg1)) == len(_qimage_bytes(qimg2))


def test_lon_bin_to_dc_map_helper_produces_correct_n_full_lon() -> None:
    """_body_sphere_lon_bin_to_dc_map returns ``n_full_lon`` consistent with shared constants."""
    n_full_lon, _lon_res_rad, _bin_to_dc = _body_sphere_lon_bin_to_dc_map(
        _LON_MIN, _LON_MAX, _D_LON, _N_DATA_COLS
    )
    assert n_full_lon == _N_FULL_LON


def test_lon_bin_to_dc_map_helper_produces_correct_array_length() -> None:
    """_body_sphere_lon_bin_to_dc_map bin_to_dc has length n_full_lon."""
    n_full_lon, _lon_res_rad, bin_to_dc = _body_sphere_lon_bin_to_dc_map(
        _LON_MIN, _LON_MAX, _D_LON, _N_DATA_COLS
    )
    assert len(bin_to_dc) == n_full_lon


def test_render_with_helper_lon_bin_to_dc_matches_manual() -> None:
    """render_to_image using the helper lon_bin_to_dc produces the same result as manual."""
    lon_deg, lat_deg, valid = _viewport_grids()
    data = np.ones((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    n_full_lon, _lon_res_rad, helper_bin_to_dc = _body_sphere_lon_bin_to_dc_map(
        _LON_MIN, _LON_MAX, _D_LON, _N_DATA_COLS
    )

    # Call render_to_image directly with typed arguments to satisfy mypy.
    qimg_helper = render_to_image(
        image_ma,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        valid=valid,
        lon_min_deg=_LON_MIN,
        lat_min_deg=_LAT_MIN,
        d_lon_deg=_D_LON,
        d_lat_deg=_D_LAT,
        lon_bin_to_dc=helper_bin_to_dc,
        n_full_lon=n_full_lon,
        n_data_rows=_N_DATA_ROWS,
        n_data_cols=_N_DATA_COLS,
        black=0.0,
        white=1.0,
        gamma=1.0,
        color_tint=None,
    )
    qimg_manual = render_to_image(
        image_ma,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        valid=valid,
        lon_min_deg=_LON_MIN,
        lat_min_deg=_LAT_MIN,
        d_lon_deg=_D_LON,
        d_lat_deg=_D_LAT,
        lon_bin_to_dc=_LON_BIN_TO_DC,
        n_full_lon=_N_FULL_LON,
        n_data_rows=_N_DATA_ROWS,
        n_data_cols=_N_DATA_COLS,
        black=0.0,
        white=1.0,
        gamma=1.0,
        color_tint=None,
    )

    assert _qimage_bytes(qimg_helper) == _qimage_bytes(qimg_manual)


def test_out_of_extent_valid_pixels_render_as_dark_red() -> None:
    """Valid-projection pixels outside the data's lat/lon extent render dark red.

    Uses a data grid that covers only the northern half (lat 0..90) with a
    viewport that includes southern latitudes.  Southern viewport pixels are
    valid in the projection (valid=True) but outside the data extent, so they
    must appear as (180, 0, 0) rather than black.
    """
    # Data covers lat 0..90 only (2 rows * 45 deg each)
    lat_min = 0.0
    d_lat = 45.0
    n_rows = 2
    n_cols = _N_DATA_COLS

    data = np.ones((n_rows, n_cols), dtype=np.float32)
    image_ma = ma.MaskedArray(data, mask=False)

    # Build a viewport where some pixels land at lat < 0 (outside data extent)
    lon_1d = np.linspace(0.0, 359.0, _VP)
    # lat goes from 80 (top) to -80 (bottom); bottom rows are outside the extent
    lat_1d = np.linspace(80.0, -80.0, _VP)
    lon_deg = np.tile(lon_1d, (_VP, 1)).astype(np.float64)
    lat_deg = np.tile(lat_1d, (_VP, 1)).T.astype(np.float64)
    valid = np.ones((_VP, _VP), dtype=bool)

    qimg = render_to_image(
        image_ma,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        valid=valid,
        lon_min_deg=_LON_MIN,
        lat_min_deg=lat_min,
        d_lon_deg=_D_LON,
        d_lat_deg=d_lat,
        lon_bin_to_dc=_LON_BIN_TO_DC,
        n_full_lon=_N_FULL_LON,
        n_data_rows=n_rows,
        n_data_cols=n_cols,
        black=0.0,
        white=1.0,
        gamma=1.0,
        color_tint=None,
    )

    # Find a viewport pixel whose latitude is below the data extent (lat < 0)
    found_dark_red = False
    for row in range(_VP):
        for col in range(_VP):
            if lat_deg[row, col] < 0.0:
                pixel = qimg.pixel(col, row)
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                b = pixel & 0xFF
                if r == 180 and g == 0 and b == 0:
                    found_dark_red = True
                    break
        if found_dark_red:
            break
    assert found_dark_red, 'Expected out-of-extent valid pixels to render as (180, 0, 0)'


def _one_pixel_grey(lon: float, lat: float, data: np.ndarray) -> int:
    """Render one ray at ``(lon, lat)`` over ``data`` and return its grey level."""
    image_ma = ma.MaskedArray(data, mask=False)
    qimg = _render(
        image_ma,
        np.full((1, 1), lon, dtype=np.float64),
        np.full((1, 1), lat, dtype=np.float64),
        np.ones((1, 1), dtype=bool),
    )
    return int((qimg.pixel(0, 0) >> 16) & 0xFF)


def _row_marked_data(marked_row: int) -> np.ndarray:
    """Data that is white on ``marked_row`` and black everywhere else."""
    data = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    data[marked_row, :] = 1.0
    return data


def _col_marked_data(marked_col: int) -> np.ndarray:
    """Data that is white on ``marked_col`` and black everywhere else."""
    data = np.zeros((_N_DATA_ROWS, _N_DATA_COLS), dtype=np.float32)
    data[:, marked_col] = 1.0
    return data


def test_latitude_past_the_midpoint_reads_the_nearer_row() -> None:
    """A ray past the midpoint between two rows samples the nearer row.

    Row ``r`` is the point sample taken at ``lat_min + r * d_lat``, so a ray 0.6
    of a row above row 0 is nearest row 1 and must read it.
    """
    grey = _one_pixel_grey(0.0, _LAT_MIN + 0.6 * _D_LAT, _row_marked_data(1))
    assert grey == 255


def test_latitude_short_of_the_midpoint_reads_the_lower_row() -> None:
    """A ray short of the midpoint between two rows samples the lower row."""
    grey = _one_pixel_grey(0.0, _LAT_MIN + 0.4 * _D_LAT, _row_marked_data(0))
    assert grey == 255


def test_longitude_past_the_midpoint_reads_the_nearer_column() -> None:
    """A ray past the midpoint between two columns samples the nearer column."""
    grey = _one_pixel_grey(1.6 * _D_LON, 0.0, _col_marked_data(2))
    assert grey == 255


def test_longitude_short_of_the_midpoint_reads_the_lower_column() -> None:
    """A ray short of the midpoint between two columns samples the lower column."""
    grey = _one_pixel_grey(1.4 * _D_LON, 0.0, _col_marked_data(1))
    assert grey == 255
