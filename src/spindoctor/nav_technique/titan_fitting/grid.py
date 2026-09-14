"""Axis-aligned resampling and the shared array helpers built on it.

Both halves of the haze fit work in the frame of the symmetry axis, so the
rotated-grid resampler, the axis unit vectors, and the nearest-neighbor
boolean reads used to carry masks into that frame live here.
"""

import math
from typing import cast

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter1d

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'axis_vectors',
    'dilate_along_t',
    'grid_axis',
    'resample_rotated_grid',
    'rotated_sample_coords',
    'sample_bool_nearest',
    'validate_image',
]


def axis_vectors(theta_rad: float) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return the symmetry-axis unit vectors for an axis angle.

    Parameters:
        theta_rad: Symmetry-axis angle in radians.

    Returns:
        ``(c_hat, a_hat)``, each a length-2 float array in ``(v, u)``.
        ``a_hat = (sin theta, cos theta)`` points along the axis toward the
        sub-solar side; ``c_hat = (cos theta, -sin theta)`` is perpendicular
        to it and defines the positive cross-track direction.
    """
    sin_t = math.sin(theta_rad)
    cos_t = math.cos(theta_rad)
    c_hat = np.array([cos_t, -sin_t], dtype=np.float64)
    a_hat = np.array([sin_t, cos_t], dtype=np.float64)
    return c_hat, a_hat


def grid_axis(half_extent_px: float) -> NDArrayFloatType:
    """Return the integer sample offsets spanning ``+-half_extent_px``."""
    n = math.ceil(half_extent_px)
    return np.arange(-n, n + 1, dtype=np.float64)


def rotated_sample_coords(
    center_vu: tuple[float, float],
    theta_rad: float,
    s_vals: NDArrayFloatType,
    t_vals: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return the ``(v, u)`` image coordinates of every rotated grid point."""
    c_hat, a_hat = axis_vectors(theta_rad)
    ss = s_vals[:, np.newaxis]
    tt = t_vals[np.newaxis, :]
    vv = center_vu[0] + ss * c_hat[0] + tt * a_hat[0]
    uu = center_vu[1] + ss * c_hat[1] + tt * a_hat[1]
    return vv, uu


def sample_bool_nearest(
    mask: NDArrayBoolType, vv: NDArrayFloatType, uu: NDArrayFloatType
) -> NDArrayBoolType:
    """Return nearest-neighbor reads of a boolean image; off-frame reads False."""
    iv = np.rint(vv).astype(np.int64)
    iu = np.rint(uu).astype(np.int64)
    inside = (iv >= 0) & (iv < mask.shape[0]) & (iu >= 0) & (iu < mask.shape[1])
    out = np.zeros(vv.shape, dtype=bool)
    out[inside] = mask[iv[inside], iu[inside]]
    return out


def validate_image(image: NDArrayFloatType, valid_mask: NDArrayBoolType) -> None:
    """Raise if the image and its validity mask are not a compatible 2-D pair."""
    if image.ndim != 2:
        raise ValueError(f'image must be 2-D; got ndim={image.ndim}')
    if valid_mask.shape != image.shape:
        raise ValueError(
            f'valid_mask must match the image shape; got {valid_mask.shape} and {image.shape}'
        )


def dilate_along_t(mask: NDArrayBoolType, pad_px: float) -> NDArrayBoolType:
    """Return a grid mask dilated along the ``t`` axis by ``+-pad_px``."""
    k = math.ceil(pad_px)
    if k <= 0:
        return mask
    grown = maximum_filter1d(mask.astype(np.uint8), size=2 * k + 1, axis=1, mode='constant', cval=0)
    return cast(NDArrayBoolType, grown > 0)


def resample_rotated_grid(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    *,
    theta_rad: float,
    s_half_extent_px: float,
    t_half_extent_px: float,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Resample an image onto a grid aligned with the symmetry axis.

    The grid is centered on ``center_vu`` with axes ``(s, t)`` along
    ``c_hat`` and ``a_hat``, integer sample spacing, and half extents
    rounded up to whole pixels.  Samples are taken by cubic interpolation.

    Parameters:
        image: 2-D image to sample.
        valid_mask: Boolean array of the same shape marking usable pixels.
        center_vu: ``(v, u)`` image position of grid point ``(0, 0)``.
        theta_rad: Symmetry-axis angle in radians.
        s_half_extent_px: Half extent of the grid along ``c_hat``.
        t_half_extent_px: Half extent of the grid along ``a_hat``.

    Returns:
        ``(grid, grid_valid)``.  ``grid`` holds the interpolated samples
        with invalid entries set to zero; ``grid_valid`` is False wherever
        the sample fell outside the image or on a pixel ``valid_mask``
        rejects.  Both have shape ``(2 * ceil(s_half) + 1,
        2 * ceil(t_half) + 1)``.

    Raises:
        ValueError: if the image is not 2-D, ``valid_mask`` has a different
            shape, or either half extent is not positive and finite.
    """
    validate_image(image, valid_mask)
    for name, extent in (
        ('s_half_extent_px', s_half_extent_px),
        ('t_half_extent_px', t_half_extent_px),
    ):
        if not math.isfinite(extent) or extent <= 0.0:
            raise ValueError(f'{name} must be positive and finite; got {extent!r}')
    s_vals = grid_axis(s_half_extent_px)
    t_vals = grid_axis(t_half_extent_px)
    vv, uu = rotated_sample_coords(center_vu, theta_rad, s_vals, t_vals)
    samples = map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.stack([vv, uu], axis=0),
        order=3,
        mode='nearest',
    )
    in_frame = (vv >= 0.0) & (vv <= image.shape[0] - 1) & (uu >= 0.0) & (uu <= image.shape[1] - 1)
    grid_valid = in_frame & sample_bool_nearest(valid_mask, vv, uu)
    grid = np.where(grid_valid, samples, 0.0)
    return cast(NDArrayFloatType, grid), cast(NDArrayBoolType, grid_valid)
