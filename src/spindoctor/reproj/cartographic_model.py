"""Cartographic model creation for planetary bodies.

Creates navigation model images by projecting a lat/lon body mosaic back
onto the image coordinate system for use in navigation correlation.

Thread safety: create_cartographic_model() creates a Backplane from the
provided observation and is not thread-safe. Do not call it from multiple
threads with the same observation.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.ma as ma
import oops
import oops.backplane
from scipy.ndimage import map_coordinates

from spindoctor.config import IMAGE_LOGGER
from spindoctor.reproj.bodies import BodyMosaicData
from spindoctor.support.types import NDArrayFloatType


@dataclass(frozen=True)
class CartographicModelResult:
    """Result returned by create_cartographic_model().

    This is a :func:`dataclasses.dataclass`; ``model_img`` is the model image in
    observation pixel coordinates ``[v, u]`` (0.0 outside mosaic coverage or
    where the body is not visible), and ``resolution_ratio`` is the ratio of the
    median mosaic effective resolution to the image-center resolution (both
    km/pixel; 1.0 means equal, greater than 1.0 means the mosaic is coarser).
    """

    model_img: NDArrayFloatType
    resolution_ratio: float


def create_cartographic_model(
    mosaic_data: BodyMosaicData,
    obs: Any,
    *,
    body_name: str,
    latlon_type: Literal['centric', 'graphic', 'squashed'] = 'centric',
    lon_direction: Literal['east', 'west'] = 'east',
) -> CartographicModelResult | None:
    """Create a navigation model by projecting a body mosaic onto image coordinates.

    For each pixel in the observation image, the corresponding latitude and
    longitude on the body surface is computed using the observation geometry.
    The mosaic is sampled at those coordinates via bilinear interpolation to
    produce a model image suitable for correlation with the actual image.

    Thread safety: Not thread-safe. Creates a Backplane from obs, which
    involves shared state. Do not call from multiple threads with the same
    observation.

    Precondition: the mosaic's grid must be anchored at its first row and
    column, so that ``mosaic_data.lat_range[0]`` is the latitude of row 0 and
    ``mosaic_data.lon_range[0]`` is the longitude of column 0, with row r at
    ``lat_range[0] + r * lat_resolution`` and column c at
    ``lon_range[0] + c * lon_resolution``. The sampling below inverts exactly
    that relation, so a mosaic whose ranges describe the outer edges of the
    grid, or any other origin, is sampled off by the difference between that
    origin and this one. BodyMosaic.to_bounded() and BodyMosaic.to_full() both
    satisfy the precondition.

    Parameters:
        mosaic_data: Body mosaic in lat/lon coordinates, typically obtained
            from BodyMosaic.to_bounded() or BodyMosaic.to_full(). Must satisfy
            the grid-anchoring precondition above.
        obs: The oops Observation for which the model is being created.
        body_name: Name of the planetary body (e.g. 'MIMAS').
        latlon_type: Coordinate system for latitude/longitude. One of
            'centric', 'graphic', or 'squashed'. Must match the type used
            to build the mosaic.
        lon_direction: Longitude direction. One of 'east' or 'west'. Must
            match the direction used to build the mosaic.

    Returns:
        CartographicModelResult with the model image and resolution ratio,
        or None if the mosaic contains no valid data.
    """
    logger = IMAGE_LOGGER

    if not isinstance(body_name, str):
        raise TypeError(f'body_name must be str, not {type(body_name).__name__}')
    if not body_name or not body_name.strip():
        raise ValueError('body_name must be a non-empty string')
    if latlon_type not in ('centric', 'graphic', 'squashed'):
        raise ValueError(
            f"latlon_type must be 'centric', 'graphic', or 'squashed', got {latlon_type!r}"
        )
    if lon_direction not in ('east', 'west'):
        raise ValueError(f"lon_direction must be 'east' or 'west', got {lon_direction!r}")

    mosaic_mask = ma.getmaskarray(mosaic_data.img)
    if mosaic_mask.all():
        logger.info('Mosaic is entirely masked; returning None')
        return None

    bp = oops.backplane.Backplane(obs)

    lat_scalar = bp.latitude(body_name, lat_type=latlon_type)
    lat_mvals = lat_scalar.mvals
    body_mask = ma.getmaskarray(lat_mvals)
    bp_latitude = lat_mvals.astype('float64')

    lon_scalar = bp.longitude(body_name, direction=lon_direction, lon_type=latlon_type)
    bp_longitude = lon_scalar.mvals.astype('float64')

    n_v, n_u = bp_latitude.shape
    n_lat, n_lon = mosaic_data.img.shape
    lat_min = mosaic_data.lat_range[0]
    lon_min = mosaic_data.lon_range[0]
    lat_res = mosaic_data.lat_resolution
    lon_res = mosaic_data.lon_resolution

    # Map each image pixel to fractional row/col in the mosaic. This inverts
    # the mosaic's rule that row r holds latitude lat_min + r * lat_res, which
    # holds only under the grid-anchoring precondition documented above.
    # All values are in radians: bp_latitude, bp_longitude, lat_min, lon_min,
    # lat_res, and lon_res. Longitude uses modular arithmetic for wraparound.
    row_coords = (bp_latitude - lat_min) / lat_res
    col_coords = ((bp_longitude - lon_min) % (2.0 * math.pi)) / lon_res

    # Determine which image pixels are on the body and within the mosaic.
    in_bounds = (
        ~body_mask
        & (row_coords >= 0.0)
        & (row_coords <= float(n_lat - 1))
        & (col_coords >= 0.0)
        & (col_coords <= float(n_lon - 1))
    )

    # Build a fill array from the mosaic (0 where masked) for interpolation.
    mosaic_fill = np.where(~mosaic_mask, mosaic_data.img.data, 0.0).astype(np.float64)

    # Bilinear interpolation of the mosaic at each image pixel's coordinates.
    # map_coordinates must not see NaNs from masked lat/lon; fill masked entries.
    row_samp = np.asarray(ma.filled(row_coords, 0.0), dtype=np.float64)
    col_samp = np.asarray(ma.filled(col_coords, 0.0), dtype=np.float64)
    coords = np.array([row_samp.ravel(), col_samp.ravel()])
    sampled = map_coordinates(mosaic_fill, coords, order=1, mode='constant', cval=0.0)
    model_img = sampled.reshape(n_v, n_u).astype(np.float32)
    model_img[~in_bounds] = 0.0

    # Compute resolution ratio: median mosaic eff_resolution / image center resolution.
    valid_mosaic_res = mosaic_data.eff_resolution.compressed()
    if len(valid_mosaic_res) > 0:
        median_mosaic_res = float(np.median(valid_mosaic_res))
        center_res = float(bp.center_resolution(body_name).vals)
        resolution_ratio = median_mosaic_res / center_res if center_res > 0.0 else 1.0
    else:
        resolution_ratio = 1.0

    logger.info(
        'Created cartographic model: shape %s, non-zero pixels %d, resolution ratio %.3f',
        model_img.shape,
        int(np.count_nonzero(model_img)),
        resolution_ratio,
    )

    return CartographicModelResult(model_img=model_img, resolution_ratio=resolution_ratio)
