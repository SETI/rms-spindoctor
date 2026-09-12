"""Where a recorded attitude points, and how far that is from another answer.

A navigation record keeps the corrected attitude as a C-matrix.  Comparing two
navigations of one image means comparing where each of them put the camera
boresight, and the unit that comparison belongs in is pixels rather than
radians: a pointing difference only means something against the detector it was
measured on, and the same angle is a miss on one camera and a rounding error on
another.

A separation also has a direction, and the direction is what tells a systematic
apart from a mistake.  Two pipelines that disagree by the same small vector on
every frame of every observation are not disagreeing about where the camera
looked; they are using different pixel datums.  So the difference is also
offered resolved onto the camera's own axes, where a constant is visible as a
constant.

Nothing here is mission-specific.  The caller supplies the plate scale of the
camera whose frame is being compared.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    'boresight_from_cmatrix',
    'offset_in_camera_px',
    'separation_px',
    'vector_from_ra_dec',
]


def boresight_from_cmatrix(cmatrix: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    """The J2000 direction a C-matrix points the camera boresight.

    A C-matrix rotates J2000 into the camera frame, so its third row is the
    camera's Z axis written in J2000, which is the boresight.

    Parameters:
        cmatrix: The nine elements of the rotation, either flat or as three
            rows of three.

    Returns:
        The boresight as a unit vector in J2000.

    Raises:
        ValueError: If the value does not hold nine numbers, or if its third
            row has no length to normalize.
    """
    elements = np.asarray(cmatrix, dtype=float).reshape(-1)
    if elements.size != 9:
        raise ValueError(f'a C-matrix has nine elements, not {elements.size}')
    row = elements.reshape(3, 3)[2, :]
    norm = float(np.linalg.norm(row))
    if norm == 0.0:
        raise ValueError('the third row of this C-matrix has no direction to point')
    return row / norm


def vector_from_ra_dec(ra_deg: float, dec_deg: float) -> np.ndarray:
    """The unit vector a right ascension and declination name.

    Parameters:
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.

    Returns:
        The direction as a unit vector on the same axes as the right ascension
        and declination were measured against.
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def separation_px(
    first: np.ndarray,
    second: np.ndarray,
    *,
    plate_scale_urad: float,
) -> float:
    """The angle between two directions, in pixels of the camera that saw them.

    Parameters:
        first: One direction, as a vector of any length.
        second: The other direction, as a vector of any length.
        plate_scale_urad: The camera's plate scale in microradians per pixel.

    The angle comes from the cross product rather than from the arc cosine of
    the dot product, because the interesting separations here are a few
    millionths of a radian, where the cosine of the angle differs from one by
    less than the rounding of a double and the arc cosine resolves nothing
    finer than a hundredth of a pixel.

    Returns:
        The angular separation divided by the plate scale.

    Raises:
        ValueError: If either direction has no length, or if the plate scale is
            not positive.
    """
    if plate_scale_urad <= 0.0:
        raise ValueError(f'a plate scale is positive, not {plate_scale_urad}')
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError('a direction with no length has no angle to another')
    unit_a, unit_b = a / na, b / nb
    angle = math.atan2(
        float(np.linalg.norm(np.cross(unit_a, unit_b))), float(np.dot(unit_a, unit_b))
    )
    return angle / (plate_scale_urad * 1e-6)


def offset_in_camera_px(
    cmatrix: Sequence[float] | Sequence[Sequence[float]],
    direction: np.ndarray,
    *,
    plate_scale_urad: float,
) -> tuple[float, float]:
    """Where another answer falls on the detector, relative to this one's boresight.

    The C-matrix rotates J2000 into the camera frame, so it carries the other
    answer's direction onto the camera's own axes, where the boresight is the
    third axis and the first two are the detector.  The result is signed, which
    is the point: an offset that keeps its sign and its size across every frame
    of every observation is a difference of pixel datum rather than of pointing.

    The axes are the camera frame's, not the detector's rows and columns.  A
    constant common to a whole run is visible either way, which is what this is
    for; reading a single frame's sign as a row or a column direction would need
    the constant rotation between the two frames as well.

    Parameters:
        cmatrix: The attitude to measure from, as nine elements.
        direction: The other answer's direction, as a vector of any length.
        plate_scale_urad: The camera's plate scale in microradians per pixel.

    Returns:
        The displacement along the camera's first and second axes, in pixels.

    Raises:
        ValueError: If the C-matrix does not hold nine numbers, if the
            direction has no length, if the plate scale is not positive, or if
            the direction lies behind the camera, where a detector offset means
            nothing.
    """
    if plate_scale_urad <= 0.0:
        raise ValueError(f'a plate scale is positive, not {plate_scale_urad}')
    elements = np.asarray(cmatrix, dtype=float).reshape(-1)
    if elements.size != 9:
        raise ValueError(f'a C-matrix has nine elements, not {elements.size}')
    other = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(other))
    if norm == 0.0:
        raise ValueError('a direction with no length falls nowhere on a detector')
    in_camera = elements.reshape(3, 3) @ (other / norm)
    if in_camera[2] <= 0.0:
        raise ValueError('a direction behind the camera has no place on its detector')
    scale = plate_scale_urad * 1e-6
    return (
        math.atan2(float(in_camera[0]), float(in_camera[2])) / scale,
        math.atan2(float(in_camera[1]), float(in_camera[2])) / scale,
    )
