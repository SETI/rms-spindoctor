"""Software renderer for non-rectangular body-mosaic projections.

:func:`render_to_image` converts a (lon_deg, lat_deg, valid) coordinate grid
-- produced by :mod:`~spindoctor.ui.mosaic_viewer.projections` -- into a
``QImage`` (Format_RGB888) using nearest-neighbour texture lookup, the same
contrast-stretch pipeline as the existing rectangular render path, and optional
per-pixel metadata tinting.

The function is intentionally stateless and side-effect-free so it can be
called from any paint method without lock concerns.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.ma as ma
from PyQt6.QtGui import QImage

from spindoctor.support.image import apply_linear_gamma_stretch


def render_to_image(
    image_ma: ma.MaskedArray,
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    valid: np.ndarray,
    lon_min_deg: float,
    lat_min_deg: float,
    d_lon_deg: float,
    d_lat_deg: float,
    lon_bin_to_dc: np.ndarray,
    n_full_lon: int,
    n_data_rows: int,
    n_data_cols: int,
    black: float,
    white: float,
    gamma: float,
    color_tint: np.ndarray | None,
) -> QImage:
    """Render a mosaic texture onto an arbitrary projection grid.

    Parameters:
        image_ma: 2-D masked array (n_data_rows, n_data_cols) with the
            photometrically-corrected mosaic data.
        lon_deg: 2-D float64 array of longitudes (deg) for every output pixel,
            in [0, 360).  Each goes to the bin whose sample is nearest it; a pixel
            whose bin has ``lon_bin_to_dc[k] == -1`` has no data column and is
            off-grid (no-data when ``valid`` is True).
        lat_deg: 2-D float64 array of latitudes (deg), in [-90, 90].  Each goes to
            the row whose sample is nearest it, so a latitude more than half a row
            below ``lat_min_deg`` or above the last row's own latitude maps outside
            ``[0, n_data_rows)`` and is off-grid (no-data when ``valid`` is True).
        valid: 2-D bool array; False pixels are painted black (off-projection
            background).  True pixels with no usable data (outside the file's
            lat/lon extent or explicitly masked) are painted dark red.
        lon_min_deg: Longitude the data grid's first column samples (deg).
        lat_min_deg: Latitude the data grid's first row samples (deg).
        d_lon_deg: Longitude resolution of the data grid (deg/column).
        d_lat_deg: Latitude resolution of the data grid (deg/row).
        lon_bin_to_dc: 1-D int32 array mapping full-circle longitude bin number
            to data column number (-1 = absent).
        n_full_lon: Length of ``lon_bin_to_dc``.
        n_data_rows: Number of rows in ``image_ma``.
        n_data_cols: Number of columns in ``image_ma``.
        black: Stretch black point.
        white: Stretch white point.
        gamma: Stretch gamma (applied as power).
        color_tint: Optional per-pixel RGB tint, shape
            (n_data_rows, n_data_cols, 3), float32 in [0, 1].  None = greyscale.

    Returns:
        A QImage in Format_RGB888 matching the shape of ``lon_deg`` / ``lat_deg``.

    Raises:
        ValueError: If ``d_lon_deg`` or ``d_lat_deg`` is zero, if ``gamma`` is not
            positive, if ``len(lon_bin_to_dc) != n_full_lon``, or if ``lon_deg``,
            ``lat_deg``, and ``valid`` do not share the same shape.
    """
    if d_lon_deg == 0 or d_lat_deg == 0:
        raise ValueError(
            f'render_to_image requires non-zero d_lon_deg and d_lat_deg; '
            f'got d_lon_deg={d_lon_deg!r}, d_lat_deg={d_lat_deg!r}'
        )
    if gamma <= 0:
        raise ValueError(f'render_to_image requires gamma > 0; got gamma={gamma!r}')
    if lon_deg.shape != lat_deg.shape or lon_deg.shape != valid.shape:
        raise ValueError(
            'render_to_image: lon_deg, lat_deg, and valid must have identical shapes; '
            f'got lon_deg {lon_deg.shape}, lat_deg {lat_deg.shape}, valid {valid.shape}'
        )
    if len(lon_bin_to_dc) != n_full_lon:
        raise ValueError(
            f'render_to_image: len(lon_bin_to_dc) must equal n_full_lon; '
            f'got len(lon_bin_to_dc)={len(lon_bin_to_dc)}, n_full_lon={n_full_lon}'
        )

    out_h, out_w = lon_deg.shape

    # ------------------------------------------------------------------
    # Map (lon_deg, lat_deg) -> (dc, dr) data indices
    # ------------------------------------------------------------------
    lon_r = np.deg2rad(lon_deg)
    lon_res_rad = math.radians(d_lon_deg)
    twopi = 2.0 * math.pi

    # Grid bin ``k`` is the point sample taken at ``k * resolution``, so the
    # nearest-neighbour lookup this renderer promises is a round, not a floor:
    # a floor picks the bin below whenever the ray falls in the upper half of a
    # bin, which is half the sphere.
    lon_r_mod = np.mod(lon_r, twopi)
    k = np.round(lon_r_mod / lon_res_rad).astype(np.int64)
    k = np.clip(k, 0, n_full_lon - 1)
    dc = lon_bin_to_dc[k]  # data column, -1 if absent

    dr = np.round((lat_deg - lat_min_deg) / d_lat_deg).astype(np.int64)

    inside = valid & (dc >= 0) & (dc < n_data_cols) & (dr >= 0) & (dr < n_data_rows)

    # ------------------------------------------------------------------
    # Sample texture
    # ------------------------------------------------------------------
    tile_data = np.zeros((out_h, out_w), dtype=np.float32)
    tile_mask = np.ones((out_h, out_w), dtype=bool)

    if np.any(inside):
        dr_v = np.clip(dr[inside], 0, n_data_rows - 1)
        dc_v = np.clip(dc[inside], 0, n_data_cols - 1)
        sub = image_ma[dr_v, dc_v]
        tile_data[inside] = np.asarray(np.nan_to_num(sub.filled(0.0), nan=0.0), dtype=np.float32)
        tile_mask[inside] = ma.getmaskarray(sub)

    # ------------------------------------------------------------------
    # Apply stretch
    # ------------------------------------------------------------------
    stretched = apply_linear_gamma_stretch(tile_data, black=black, white=white, gamma=gamma).astype(
        np.float32
    )
    gray = (stretched * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Build RGB with optional per-pixel tint
    # ------------------------------------------------------------------
    if color_tint is not None and np.any(inside):
        dr_c = np.clip(dr, 0, n_data_rows - 1)
        dc_c = np.clip(dc, 0, n_data_cols - 1)
        # Default: neutral tint (1.0) everywhere
        tint = np.ones((out_h, out_w, 3), dtype=np.float32)
        tint[inside] = color_tint[dr_c[inside], dc_c[inside]]
        gray_f = gray[:, :, np.newaxis].astype(np.float32)
        rgb = np.clip(gray_f * tint, 0, 255).astype(np.uint8)
    else:
        rgb = np.stack([gray, gray, gray], axis=2)

    # Background (off-projection) pixels -> black
    background = ~valid
    if np.any(background):
        rgb[background, 0] = 0
        rgb[background, 1] = 0
        rgb[background, 2] = 0

    # Any valid projection pixel with no usable data (explicitly masked or
    # simply outside the file's lat/lon extent) -> dark red.
    no_data = valid & (~inside | tile_mask)
    if np.any(no_data):
        rgb[no_data, 0] = 180
        rgb[no_data, 1] = 0
        rgb[no_data, 2] = 0

    # ------------------------------------------------------------------
    # Build QImage from a persistent buffer (constructor keeps a pointer; do not
    # pass a temporary ``tobytes()`` result without pinning the backing bytes).
    # ------------------------------------------------------------------
    rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)
    buf = bytearray(rgb_c.tobytes())
    qimg = QImage(
        cast(bytes, buf),
        out_w,
        out_h,
        3 * out_w,
        QImage.Format.Format_RGB888,
    )
    qimg._buf = buf  # type: ignore[attr-defined]  # pin buffer lifetime for QImage ctor
    return qimg
