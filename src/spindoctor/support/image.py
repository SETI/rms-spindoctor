import math
from typing import Any, cast

import numpy as np
import scipy.ndimage as ndimage
from numpy.fft import fft2, fftfreq, ifft2

from spindoctor.support.types import NDArrayComplexType, NDArrayFloatType, NDArrayType, NPType

# ==============================================================================
#
# ARRAY MANIPULATION
#
# ==============================================================================


def shift_array(
    array: NDArrayType[NPType], offset: int | list[int] | tuple[int, ...], fill: Any = 0
) -> NDArrayType[NPType]:
    """Shift an array by an offset, filling with zeros or a specified value.

    Parameters:
        array: The N-dimensional array to shift.
        offset: A list or tuple of length N of offsets in the same order as the array
            indices. A positive offset shifts to higher numbers on the axis.
        fill: The value used to fill the newly created array elements.

    Returns:
        The array shifted by the given amount.  Aliasing note: a zero offset
        returns the *same* array object (no copy); any non-zero offset returns
        a fresh copy.  Do not mutate the result in place assuming it is always
        independent of the input.

    Raises:
        ValueError: If the array shape and offset list have different lengths.
    """

    if isinstance(offset, int):
        offset = [offset]

    if array.ndim != len(offset):
        raise ValueError(
            f'Array ({array.ndim}) and offset ({len(offset)}) must have same number of dimensions'
        )

    if all(x == 0 for x in offset):
        return array

    array = array.copy()
    for axis in range(len(offset)):
        array = np.roll(array, offset[axis], axis)

    for axis in range(len(offset)):
        if offset[axis] != 0:
            rolled_array = np.rollaxis(array, axis)
            if offset[axis] < 0:
                rolled_array[offset[axis] :, ...] = fill
            else:
                rolled_array[: offset[axis], ...] = fill

    return array


def pad_array(
    array: NDArrayType[NPType], margin: list[int] | tuple[int, ...], fill: Any = 0
) -> NDArrayType[NPType]:
    """Pad an array by the given margin at each edge.

    Parameters:
        array: The N-dimensional array to pad.
        margin: A list or tuple of length N of margin amounts in the same order as the
            array indices. Each axis is padded to its original size plus 2*N and the
            original values are centered on the new axis.
        fill: The value used to fill the newly created array elements.

    Returns:
        The array padded by the given amount.  Aliasing note: an all-zero
        margin returns the *same* array object (no copy); a non-zero margin
        returns a fresh ``np.pad`` array.

    Raises:
        ValueError: If the array shape and margin list have different lengths.
    """

    if array.ndim != len(margin):
        raise ValueError(
            f'Array ({array.ndim}) and offset ({len(margin)}) must have same number of dimensions'
        )

    if all(x == 0 for x in margin):
        return array

    padding = [(x, x) for x in margin]
    return np.pad(array, padding, mode='constant', constant_values=fill)


def unpad_array(
    array: NDArrayType[NPType], margin: list[int] | tuple[int, ...]
) -> NDArrayType[NPType]:
    """Remove a padded margin from each edge.

    Parameters:
        array: The N-dimensional array to unpad.
        margin: A list or tuple of length N of margin amounts in the same order as the
            array indices. Each axis is unpadded to its original size minus 2*N and the
            original centered values are used on the new axis.

    Returns:
        The array unpadded by the given amount.  Aliasing note: this never
        copies -- an all-zero margin returns the *same* array object and a
        non-zero margin returns a *view* (slice) into the input.  Mutating the
        result therefore mutates the input; copy first if independence is
        needed.

    Raises:
        ValueError: If the array shape and margin list have different lengths.
    """

    if array.ndim != len(margin):
        raise ValueError(
            f'Array ({array.ndim}) and margin ({len(margin)}) must have same number of dimensions'
        )

    if all(x == 0 for x in margin):
        return array

    # See https://stackoverflow.com/questions/24806174/
    # is-there-an-opposite-inverse-to-numpy-pad-function
    reversed_padding = tuple(
        slice(pad, dim - pad) for (pad, dim) in zip(margin, array.shape, strict=True)
    )

    return array[reversed_padding]


def pad_top_left(array: NDArrayType[NPType], v_size: int, u_size: int) -> NDArrayType[NPType]:
    """Place array at (0,0) inside zeros(v_size, u_size).

    Parameters:
        array: A 2-D array to place.
        v_size: The height of the output array.
        u_size: The width of the output array.

    Returns:
        A (v_size, u_size) array with the input array at the top-left corner.
        If the input array is larger than the target size, it will be truncated.
    """

    if array.ndim != 2:
        raise ValueError(f'Array must be 2-dimensional, got {array.ndim}')
    ret = np.zeros((v_size, u_size), dtype=array.dtype)
    v, u = array.shape
    ret[: min(v, v_size), : min(u, u_size)] = array[: min(v, v_size), : min(u, u_size)]
    return ret


def crop_center(img: NDArrayType[NPType], out_shape: tuple[int, int]) -> NDArrayType[NPType]:
    """Center crop to out_shape (h,w).

    Parameters:
        img: A 2-D array to crop.
        out_shape: The target shape (height, width).

    Returns:
        The center-cropped array.

    Raises:
        ValueError: If img is not 2-dimensional or out_shape is larger than img.shape.
    """

    if img.ndim != 2:
        raise ValueError(f'Image must be 2-dimensional, got {img.ndim}')
    img_v, img_u = img.shape
    out_v, out_u = out_shape
    if out_v > img_v or out_u > img_u:
        raise ValueError(f'Output shape {out_shape} cannot be larger than image shape {img.shape}')
    sy = (img_v - out_v) // 2
    sx = (img_u - out_u) // 2
    return img[sy : sy + out_v, sx : sx + out_u]


def next_power_of_2(n: int) -> int:
    """Computes the smallest power of 2 that is greater than or equal to n.

    Parameters:
        n: A non-negative integer.

    Returns:
        The smallest power of 2 that is >= n.  ``0`` returns ``1`` (the
        smallest power of 2 that is >= 0).

    Raises:
        ValueError: If ``n`` is negative (a power of 2 >= a negative number is
            ill-defined, and ``bin()`` of a negative would be misparsed).
    """

    if n < 0:
        raise ValueError(f'next_power_of_2 requires a non-negative integer; got {n}')
    if n <= 1:
        return 1
    s = bin(n)[2:]
    if s.count('1') == 1:  # Already power of 2
        return n
    return 1 << len(s)


def pad_array_to_power_of_2(
    data: NDArrayType[NPType],
) -> tuple[NDArrayType[NPType], tuple[int, ...]]:
    """Zero-pad a 2-D array on all sides to be a power of 2 in each dimension.

    If the original array has a side that is not a power of 2, it must be even.

    Returns: A tuple containing the padded array and the amount of padding added.
    """

    p2 = [next_power_of_2(x) for x in data.shape]

    if all(s1 == s2 for s1, s2 in zip(p2, data.shape, strict=True)):
        return data, (0, 0)

    if any((s != 1) and (s & 1) for s in data.shape):
        raise ValueError(f'shape must be power of 2 or even: {data.shape}')

    padding = tuple([(s - dim) // 2 for s, dim in zip(p2, data.shape, strict=True)])
    return pad_array(data, padding), padding


# def compress_saturated_overlay(overlay):
#     """Compress a 2-D RGB array making each color a single bit."""
#     # Compress an RGB overlay assuming everything is either 0 or 255
#     ret = np.empty((overlay.shape[0]/2, overlay.shape[1]), dtype=np.uint8)
#     ret[:,:] = ( (overlay[ ::2,:,0] > 127) |
#                 ((overlay[ ::2,:,1] > 127) << 1) |
#                 ((overlay[ ::2,:,2] > 127) << 2) |
#                 ((overlay[1::2,:,0] > 127) << 3) |
#                 ((overlay[1::2,:,1] > 127) << 4) |
#                 ((overlay[1::2,:,2] > 127) << 5))
#     return ret

# def uncompress_saturated_overlay(overlay):
#     """Uncompress a 2-D RGB array."""
#     ret = np.empty((overlay.shape[0]*2, overlay.shape[1], 3), dtype=np.uint8)
#     ret[ ::2,:,0] =  overlay & 1
#     ret[ ::2,:,1] = (overlay & 2) >> 1
#     ret[ ::2,:,2] = (overlay & 4) >> 2
#     ret[1::2,:,0] = (overlay & 8) >> 3
#     ret[1::2,:,1] = (overlay & 16) >> 4
#     ret[1::2,:,2] = (overlay & 32) >> 5
#     ret *= 255
#     return ret

# def array_interpolate_missing_stripes(data):
#     """Interpolate missing horizontal data in an array.

#     This routine handles an array that has the right side of some lines missing
#     due to data transmission limitations. A pixel is interpolated if it is
#     missing (zero) and the pixels immediately above and below are present
#     (not zero)."""
#     zero_mask = (data == 0.)
#     data_up = np.zeros(data.shape)
#     data_up[:-1,:] = data[1:,:]
#     data_down = np.zeros(data.shape)
#     data_down[1:,:] = data[:-1,:]
#     up_mask = (data_up != 0.)
#     down_mask = (data_down != 0.)
#     good_mask = np.logical_and(zero_mask, up_mask)
#     good_mask = np.logical_and(good_mask, down_mask)
#     data_mean = (data_up+data_down)/2.
#     ret_data = data.copy()
#     ret_data[good_mask] = data_mean[good_mask]
#     return ret_data


def array_zoom(a: NDArrayType[NPType], factor: list[int] | tuple[int, ...]) -> NDArrayType[NPType]:
    """Zooms an array by the specified factor using array indexing.

    Parameters:
        a: The input array to zoom.
        factor: The zoom factor for each dimension of the array.

    Returns:
        The zoomed array with dimensions multiplied by the corresponding factors.
    """

    result: np.ndarray = np.asarray(a)
    for ax, f in enumerate(factor):
        if f > 1:
            result = np.repeat(result, f, axis=ax)
    return cast(NDArrayType[NPType], result)


def array_unzoom(
    array: NDArrayType[NPType],
    factor: int | list[int] | tuple[int, ...],
    method: str = 'mean',
) -> NDArrayType[NPType]:
    """Zoom down by an integer factor. Returned arrays are floating-point."""

    if len(array.shape) == 1:
        assert isinstance(factor, int)
        if factor == 1:
            return array

        shape1 = array.shape
        if shape1[0] % factor != 0:
            raise ValueError('array shape and unzoom factor are incompatible')

        new_shape1 = shape1[0] // factor
        reshaped1 = array.reshape((new_shape1, factor))

        if method == 'min':
            unzoomed = reshaped1.min(axis=1)
        elif method == 'mean':
            unzoomed = reshaped1.mean(axis=1)
        elif method == 'max':
            unzoomed = reshaped1.max(axis=1)
        else:
            raise ValueError(f'Unknown unzoom method "{method}"')
        return cast(NDArrayType[NPType], unzoomed.reshape((new_shape1,)))

    if not isinstance(factor, (tuple, list)):
        assert isinstance(factor, int)
        factor = (factor, factor)

    if factor[0] == 1 and factor[1] == 1:
        return array

    shape2 = array.shape
    if shape2[0] % factor[0] != 0 or shape2[1] % factor[1] != 0:
        raise ValueError('array shape and unzoom factor are incompatible')

    new_shape2 = (shape2[0] // factor[0], shape2[1] // factor[1])
    reshaped2 = array.reshape((new_shape2[0], factor[0], new_shape2[1], factor[1]))

    if method == 'min':
        unzoomed = reshaped2.min(axis=1)
        unzoomed = unzoomed.min(axis=2)
    elif method == 'mean':
        unzoomed = reshaped2.mean(axis=1)
        unzoomed = unzoomed.mean(axis=2)
    elif method == 'max':
        unzoomed = reshaped2.max(axis=1)
        unzoomed = unzoomed.max(axis=2)
    else:
        raise ValueError(f'Unknown unzoom method "{method}"')

    return cast(NDArrayType[NPType], unzoomed)


# ==============================================================================
#
# FILTERS
#
# ==============================================================================


def filter_local_maximum(
    data: NDArrayType[NPType],
    maximum_boxsize: int = 3,
    median_boxsize: int = 11,
    maximum_blur: int = 0,
    maximum_tolerance: float = 1.0,
    minimum_boxsize: int = 0,
    gaussian_blur: float = 0.0,
) -> NDArrayType[NPType]:
    """Filter an array to find local maxima.

    Process:
        1) Create a mask consisting of the pixels that are local maxima
           within maximum_boxsize
        2) Find the minimum value for the area around each array pixel
           using minimum_boxsize
        3) Remove maxima that are not at least maximum_tolerange times
           the local minimum
        4) Blur the maximum pixels mask to make each a square area of
           maximum_blur X maximum_blur
        5) Compute the median-subtracted value for each array pixel
           using median_boxsize
        6) Copy the median-subtracted array pixels to a new zero-filled
           array where the maximum mask is true
        7) Gaussian blur this final result

    Parameters:
        data: The array
        maximum_boxsize: The box size to use when finding the maximum value for
            the area around each pixel.
        median_boxsize: The box size to use when finding the median value for
            the area around each pixel.
        maximum_blur: The amount to blur the maximum filter. If a pixel is
            marked as a maximum, then the pixels in a blur X blur square will
            also be marked as a maximum.
        maximum_tolerance: The factor above the local minimum that a maximum
            pixel has to be in order to be included in the final result.
        minimum_boxsize: The box size to use when finding the minimum value for
            the area around each pixel.
        gaussian_blur: The amount to blur the final result.

    Returns:
        The filtered array.
    """

    assert maximum_boxsize > 0
    max_filter = ndimage.maximum_filter(data, maximum_boxsize)
    mask = data == max_filter

    if minimum_boxsize:
        min_filter = ndimage.minimum_filter(data, minimum_boxsize)
        tol_mask = data >= min_filter * maximum_tolerance
        mask = np.logical_and(mask, tol_mask)

    if maximum_blur:
        mask = ndimage.maximum_filter(mask, maximum_blur)

    if median_boxsize:
        flat = data - ndimage.median_filter(data, median_boxsize)
    else:
        flat = data

    filtered = np.zeros(data.shape, dtype=data.dtype)
    filtered[mask] = flat[mask]

    if gaussian_blur:
        filtered = ndimage.gaussian_filter(filtered, gaussian_blur)

    return filtered


_FOOTPRINT_CACHE: dict[int, Any] = {}


def filter_sub_median(
    data: NDArrayFloatType,
    median_boxsize: int = 11,
    gaussian_blur: float = 0.0,
    footprint: str = 'square',
) -> NDArrayFloatType:
    """Compute the median-subtracted value for each pixel.

    Parameters:
        data: The array
        median_boxsize: The box size to use when finding the median value for the
            area around each pixel.
        gaussian_blur: The amount to blur the median value before subtracting it
            from the array.
        footprint: The shape of footprint to use ('square', 'circle').

    Returns:
        The median-subtracted array.
    """

    if not median_boxsize and not gaussian_blur:
        return data

    sub = data

    if median_boxsize:
        if footprint == 'square':
            sub = ndimage.median_filter(sub, median_boxsize)
        else:
            assert footprint == 'circle'
            if median_boxsize in _FOOTPRINT_CACHE:
                footprint_arr = _FOOTPRINT_CACHE[median_boxsize]
            else:
                footprint_arr = np.zeros((median_boxsize, median_boxsize), dtype='bool')
                box_half = median_boxsize // 2
                x = np.tile(np.arange(median_boxsize) - box_half, median_boxsize).reshape(
                    (median_boxsize, median_boxsize)
                )
                y = np.repeat(np.arange(median_boxsize) - box_half, median_boxsize).reshape(
                    (median_boxsize, median_boxsize)
                )
                dist = np.sqrt(x**2 + y**2)
                footprint_arr[dist <= box_half + 0.5] = 1
                _FOOTPRINT_CACHE[median_boxsize] = footprint_arr
            sub = ndimage.median_filter(sub, footprint=footprint_arr)

    if gaussian_blur:
        sub = ndimage.gaussian_filter(sub, gaussian_blur)

    return data - sub


def filter_downsample(arr: NDArrayFloatType, amt_y: int, amt_x: int) -> NDArrayFloatType:
    """Downsamples an array by averaging blocks of pixels.

    Parameters:
        arr: The input array to downsample.
        amt_y: The vertical downsampling factor.
        amt_x: The horizontal downsampling factor.

    Returns:
        The downsampled array.

    Raises:
        AssertionError: If the array dimensions are not divisible by the downsampling
            factors.
    """

    assert arr.shape[0] % amt_y == 0
    assert arr.shape[1] % amt_x == 0
    ny = arr.shape[0] // amt_y
    nx = arr.shape[1] // amt_x
    ret = (
        np.swapaxes(arr.reshape(ny, amt_y, nx, amt_x), 1, 2)
        .reshape(ny, nx, amt_x * amt_y)
        .mean(axis=2)
    )
    return ret


def real_ifft2(spectrum: NDArrayComplexType) -> NDArrayFloatType:
    """Inverse-transform a spectrum and keep its real part, as its own array.

    ``np.real`` of a complex array is a view whose base is that array, so a
    real part taken that way keeps the whole complex inverse transform alive
    for as long as the view lives.  Copying the real part out lets the complex
    output die with the statement that made it, which on a wide-margin frame
    is one frame-sized buffer fewer for every result kept.

    Parameters:
        spectrum: The spectrum to inverse-transform.

    Returns:
        The real part of the inverse transform, contiguous and owning its data.
    """
    return np.ascontiguousarray(ifft2(spectrum).real)


def gaussian_blur_cov(img: NDArrayFloatType, sigma: NDArrayFloatType) -> NDArrayFloatType:
    """Blur by anisotropic Gaussian with covariance matrix in frequency domain.

    Parameters:
        img: A 2-D array to blur.
        sigma: A 2x2 covariance matrix [[Syy, Syx], [Sxy, Sxx]].

    Returns:
        The blurred image.

    Raises:
        ValueError: If img is not 2-dimensional or sigma is not 2x2.
    """

    if img.ndim != 2:
        raise ValueError(f'Image must be 2-dimensional, got {img.ndim}')
    if sigma.shape != (2, 2):
        raise ValueError(f'Sigma must be 2x2, got {sigma.shape}')
    fy = fftfreq(img.shape[0])[:, None]
    fx = fftfreq(img.shape[1])[None, :]
    Syy, Syx = sigma[0, 0], sigma[0, 1]
    Sxy, Sxx = sigma[1, 0], sigma[1, 1]
    q = Syy * (fy * fy) + (Syx + Sxy) * (fy * fx) + Sxx * (fx * fx)
    H = np.exp(-2.0 * (np.pi**2) * q)
    return real_ifft2(fft2(img) * H)


def normalize_array(a: NDArrayFloatType, eps: float = 1e-12) -> NDArrayFloatType:
    """Zero-mean, unit-std normalization (safe if nearly constant).

    Parameters:
        a: The input array to normalize.
        eps: Minimum standard deviation threshold. If std < eps, returns zeros.

    Returns:
        Normalized array with mean 0 and std 1, or zeros if nearly constant.
    """

    a = np.asarray(a, np.float64)
    m = a.mean()
    s = a.std()
    if s < eps:
        return np.zeros_like(a, dtype=np.float64)
    return cast(NDArrayFloatType, (a - m) / s)


def require_finite_int_or_float(name: str, value: object) -> None:
    """Validate stretch control numeric parameters at the public API boundary.

    Used by stretch helpers and the UI control-builders to reject ``bool``,
    non-numeric, and non-finite inputs with a clear error message tagged by
    the parameter name.

    Parameters:
        name: Parameter name to include in the error message (e.g.
            ``'black'``).
        value: Candidate value to validate.

    Raises:
        TypeError: When ``value`` is a ``bool`` or not an ``int``/``float``.
        ValueError: When ``value`` is not finite.
    """
    if isinstance(value, bool):
        raise TypeError(f'{name} must be int or float, not bool')
    if not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be int or float, not {type(value).__name__}')
    if not math.isfinite(float(value)):
        raise ValueError(f'{name} must be a finite number, got {value!r}')


def apply_linear_gamma_stretch(
    data: NDArrayFloatType,
    *,
    black: float,
    white: float,
    gamma: float,
) -> NDArrayFloatType:
    """Apply black/white/gamma stretch and return a float array in ``[0, 1]``.

    Uses the convention ``((clip(data, black, white) - black) / (white - black)) ** gamma``.
    A ``gamma`` of 1.0 is linear; values below 1.0 brighten the mid-tones
    (common for display) and values above 1.0 darken them.

    Parameters:
        data: Input float array (any shape).
        black: Black-point; input values at or below this map to 0.
        white: White-point; input values at or above this map to 1. If
            ``white <= black`` (including accidental UI equality), ``white`` is
            raised silently to the next representable float above ``black`` so
            the linear scale denominator is never zero.
        gamma: Exponent applied after linear normalisation; must be finite and
            strictly greater than zero.

    Returns:
        Float array of the same shape as ``data`` with values in ``[0, 1]``.

    Raises:
        TypeError: If ``black``, ``white``, or ``gamma`` is not an ``int`` or
            ``float``, or any of them is a ``bool``.
        ValueError: If any of ``black``, ``white``, or ``gamma`` is not finite;
            or if ``gamma <= 0``.
    """
    require_finite_int_or_float('black', black)
    require_finite_int_or_float('white', white)
    require_finite_int_or_float('gamma', gamma)
    b = float(black)
    w = float(white)
    g = float(gamma)
    if w <= b:
        w = math.nextafter(b, math.inf)
    if g <= 0.0:
        raise ValueError(f'gamma must be greater than 0, got {g!r}')
    normalized = np.clip((data - b) / (w - b), 0.0, 1.0)
    return cast(NDArrayFloatType, np.power(normalized, g))


def gradient_magnitude(img: NDArrayFloatType) -> NDArrayFloatType:
    """Compute isotropic gradient magnitude.

    Parameters:
        img: The input array.

    Returns:
        The gradient magnitude at each pixel, computed as sqrt(gx^2 + gy^2).
    """
    gy = np.gradient(img, axis=0)
    gx = np.gradient(img, axis=1)
    return cast(NDArrayFloatType, np.hypot(gy, gx))


# ==============================================================================
#
# DRAWING ROUTINES
#
# ==============================================================================


def draw_line_arrow(
    img: NDArrayType[NPType],
    color: int | float | list[int | float] | tuple[int | float, ...],
    x0: int | float,
    y0: int | float,
    x1: int | float,
    y1: int | float,
    thickness: float = 1.0,
    arrow_head_length: int = 10,
    arrow_head_angle: int = 45,
) -> None:
    """Draws a line with an arrow head at the end point.

    Parameters:
        img: The 2-D (or higher) array to draw on.
        color: The scalar (or higher) color to draw.
        x0: The x-coordinate of the starting point.
        y0: The y-coordinate of the starting point.
        x1: The x-coordinate of the ending point.
        y1: The y-coordinate of the ending point.
        thickness: The thickness of the line.
        arrow_head_length: The length of the arrow head lines.
        arrow_head_angle: The angle in degrees between the arrow head and the main line.
    """

    draw_line(img, color, x0, y0, x1, y1, thickness)

    line_angle = -np.atan2((y1 - y0), (x0 - x1))
    angle = line_angle + np.deg2rad(arrow_head_angle)
    x2 = x1 + np.round(arrow_head_length * np.cos(angle))
    y2 = y1 + np.round(arrow_head_length * np.sin(angle))
    draw_line(img, color, x1, y1, x2, y2, thickness)

    angle = line_angle - np.deg2rad(arrow_head_angle)
    x2 = x1 + np.round(arrow_head_length * np.cos(angle))
    y2 = y1 + np.round(arrow_head_length * np.sin(angle))
    draw_line(img, color, x1, y1, x2, y2, thickness)


def draw_line(
    img: NDArrayType[NPType],
    color: int | float | list[int | float] | tuple[int | float, ...],
    x0: int | float,
    y0: int | float,
    x1: int | float,
    y1: int | float,
    thickness: float = 1.0,
) -> None:
    """Draw a line using Bresenham's algorithm with the given thickness.

    The line is drawn by drawing each point as a line perpendicular to
    the main line.

    Parameters:
        img: The 2-D (or higher) array to draw on.
        x0, y0: The starting point.
        x1, y1: The ending point.
        color: The scalar (or higher) color to draw.
        thickness: The thickness (total width) of the line.
    """

    x0 = int(x0)
    x1 = int(x1)
    y0 = int(y0)
    y1 = int(y1)

    if thickness == 1:
        # Do the simple version
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if x0 < x1:
            sx = 1
        else:
            sx = -1
        if y0 < y1:
            sy = 1
        else:
            sy = -1
        err = dx - dy

        while True:
            if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
                img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x0 = x0 + sx
            if e2 < dx:
                err = err + dx
                y0 = y0 + sy
        return

    # Find the perpendicular angle
    angle = np.arctan2(y1 - y0, x1 - x0)
    x_offset = int(np.round(np.cos(angle)))
    y_offset = int(np.round(np.sin(angle)))
    perp_angle = angle + np.pi / 2
    perp_x1 = int(np.round(thickness / 2.0 * np.cos(perp_angle)))
    perp_x0 = -perp_x1
    perp_y1 = int(np.round(thickness / 2.0 * np.sin(perp_angle)))
    perp_y0 = -perp_y1
    if perp_x0 == perp_x1 and perp_y0 == perp_y1:
        draw_line(img, color, x0, y0, x1, y1)
        return

    # Compute the perpendicular offsets using one pass of Bresenham's
    perp_offsets_x = []
    perp_offsets_y = []

    dx = abs(perp_x1 - perp_x0)
    dy = abs(perp_y1 - perp_y0)
    if perp_x0 < perp_x1:
        sx = 1
    else:
        sx = -1
    if perp_y0 < perp_y1:
        sy = 1
    else:
        sy = -1
    err = dx - dy

    while True:
        # There's a better way to do this, but it's patented by IBM!
        # So just do something brute force instead
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0 + x_offset)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0 + y_offset)
        perp_offsets_x.append(perp_x0 + x_offset)
        perp_offsets_y.append(perp_y0 + y_offset)
        if perp_x0 == perp_x1 and perp_y0 == perp_y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            perp_x0 = perp_x0 + sx
        if e2 < dx:
            err = err + dx
            perp_y0 = perp_y0 + sy

    # Now draw the final line applying the offsets
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx - dy

    while True:
        for i in range(len(perp_offsets_x)):
            yy = y0 + perp_offsets_y[i]
            xx = x0 + perp_offsets_x[i]
            if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                img[yy, xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            err = err + dx
            y0 = y0 + sy


def draw_rect(
    img: NDArrayType[NPType],
    color: bool | int | float | list[int | float] | tuple[int | float, ...],
    xctr: int,
    yctr: int,
    xhalfwidth: int,
    yhalfwidth: int,
    thickness: int = 1,
    dot_spacing: int = 1,
) -> None:
    """Draw a rectangle with the given line thickness.

    Parameters:
        img: The 2-D (or higher) array to draw on.
        color: The scalar (or higher) color to draw.
        xctr: The horizontal (column) center of the rectangle.
        yctr: The vertical (row) center of the rectangle.
        xhalfwidth: The horizontal half-width, on each side of the center.
        yhalfwidth: The vertical half-width, on each side of the center.
        thickness: The thickness (total width) of the line.
        dot_spacing: The spacing between dots in the rectangle. 1 means dots are
            adjacent (solid lines).

    All slice bounds are clipped to the array extent so a center near or beyond
    the image edge cannot wrap a negative index around to the far side and paint
    a spurious rectangle; an entirely off-image rectangle draws nothing.
    """

    rows, cols = int(img.shape[0]), int(img.shape[1])

    def _cy(v: int) -> int:
        return max(0, min(int(v), rows))

    def _cx(v: int) -> int:
        return max(0, min(int(v), cols))

    # Top
    img[
        _cy(yctr - yhalfwidth - thickness + 1) : _cy(yctr - yhalfwidth + 1),
        _cx(xctr - xhalfwidth - thickness + 1) : _cx(xctr + xhalfwidth + thickness) : dot_spacing,
    ] = color
    # Bottom
    img[
        _cy(yctr + yhalfwidth) : _cy(yctr + yhalfwidth + thickness),
        _cx(xctr - xhalfwidth - thickness + 1) : _cx(xctr + xhalfwidth + thickness) : dot_spacing,
    ] = color
    # Left
    img[
        _cy(yctr - yhalfwidth - thickness + 1) : _cy(yctr + yhalfwidth + thickness),
        _cx(xctr - xhalfwidth - thickness + 1) : _cx(xctr - xhalfwidth + 1) : dot_spacing,
    ] = color
    # Right
    img[
        _cy(yctr - yhalfwidth - thickness + 1) : _cy(yctr + yhalfwidth + thickness),
        _cx(xctr + xhalfwidth) : _cx(xctr + xhalfwidth + thickness) : dot_spacing,
    ] = color


def draw_circle(
    img: NDArrayType[NPType],
    color: int | float | list[int | float] | tuple[int | float, ...],
    x0: int,
    y0: int,
    r: int,
    thickness: int = 1,
) -> None:
    """Draw a circle using Bresenham's algorithm with the given thickness.

    Parameters:
        img: The 2-D (or higher) array to draw on.
        x0, y0: The middle of the circle.
        r: The radius of the circle.
        color: The scalar (or higher) color to draw.
        thickness: The thickness (total width) of the circle.
    """

    def _draw_circle(
        img: NDArrayType[NPType],
        color: int | float | list[int | float] | tuple[int | float, ...],
        x0: int,
        y0: int,
        r: float | np.floating[Any],
        bigpixel: bool,
    ) -> None:
        x = -r
        y = 0
        err = 2 - 2 * r
        if bigpixel:
            off_list = [-1, 0, 1]
        else:
            off_list = [0]

        while x < 0:
            for xoff in off_list:
                for yoff in off_list:
                    if 0 <= y0 + y + yoff < img.shape[0] and 0 <= x0 - x + xoff < img.shape[1]:
                        img[int(y0 + y + yoff), int(x0 - x + xoff)] = color
                    if 0 <= y0 - x + yoff < img.shape[0] and 0 <= x0 - y + xoff < img.shape[1]:
                        img[int(y0 - x + yoff), int(x0 - y + xoff)] = color
                    if 0 <= y0 - y + yoff < img.shape[0] and 0 <= x0 + x + xoff < img.shape[1]:
                        img[int(y0 - y + yoff), int(x0 + x + xoff)] = color
                    if 0 <= y0 + x + yoff < img.shape[0] and 0 <= x0 + y + xoff < img.shape[1]:
                        img[int(y0 + x + yoff), int(x0 + y + xoff)] = color
            r = err
            if r <= y:
                y = y + 1
                err = err + y * 2 + 1
            if r > x or err > y:
                x = x + 1
                err = err + x * 2 + 1

    x0 = int(x0)
    y0 = int(y0)
    r = int(r)

    if thickness == 1:
        _draw_circle(img, color, x0, y0, r, False)
        return

    if thickness <= 3:
        _draw_circle(img, color, x0, y0, r, True)
        return

    # This is not perfect, but it's simple
    for r_offset in np.arange(-(thickness - 2) / 2.0, (thickness - 2) / 2.0 + 0.5, 0.5):
        _draw_circle(img, color, x0, y0, r + r_offset, True)


# ==============================================================================
#
# MISC
#
# ==============================================================================

# def hsv_to_rgb(data: NDArrayType[NPType]) -> NDArrayType[NPType]:
#     """Convert an array [...,3] of HSV values into RGB values.

#     This is the same as Python's colorsys.hsv_to_rgb but is vectorized.
#     """

#     # From http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
#     # C = V*S
#     # X = C*(1-|(h/60 deg) mod 2 - 1|)
#     # m = V-C

#     deg60 = oops.PI/3
#     h = data[..., 0] * oops.TWOPI
#     c = cast(NDArrayType[NPType], data[..., 1]) * cast(NDArrayType[NPType], data[..., 2])
#     x = c * (1 - np.abs((h / deg60) % 2 - 1))
#     m = data[..., 2] - c

#     ret = np.zeros(data.shape, dtype=data.dtype)

#     is_0_60 = h < deg60
#     is_60_120 = (deg60 <= h) & (h < deg60*2)
#     is_120_180 = (deg60*2 <= h) & (h < deg60*3)
#     is_180_240 = (deg60*3 <= h) & (h < deg60*4)
#     is_240_300 = (deg60*4 <= h) & (h < deg60*5)
#     is_300_360 = deg60*5 <= h

#     cond = is_0_60 | is_300_360
#     ret[cond, 0] = c[cond]
#     cond = is_60_120 | is_240_300
#     ret[cond, 0] = x[cond]

#     cond = is_0_60 | is_180_240
#     ret[cond, 1] = x[cond]
#     cond = is_60_120 | is_120_180
#     ret[cond, 1] = c[cond]

#     cond = is_120_180 | is_300_360
#     ret[cond, 2] = x[cond]
#     cond = is_180_240 | is_240_300
#     ret[cond, 2] = c[cond]

#     ret[..., 0] += m
#     ret[..., 1] += m
#     ret[..., 2] += m

#     return ret
