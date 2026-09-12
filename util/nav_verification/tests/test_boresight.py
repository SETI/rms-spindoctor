"""Unit tests for the boresight arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from util.nav_verification.boresight import (
    boresight_from_cmatrix,
    offset_in_camera_px,
    separation_px,
    vector_from_ra_dec,
)

NAC_URAD = 5.9946


def test_boresight_is_the_third_row() -> None:
    """A C-matrix points where its third row points."""
    cmatrix = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert boresight_from_cmatrix(cmatrix) == pytest.approx([1.0, 0.0, 0.0])


def test_boresight_accepts_three_rows() -> None:
    """The nine elements may arrive already shaped."""
    rows = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert boresight_from_cmatrix(rows) == pytest.approx([1.0, 0.0, 0.0])


def test_boresight_normalizes() -> None:
    """The direction comes back as a unit vector whatever the row's length."""
    cmatrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]
    assert float(np.linalg.norm(boresight_from_cmatrix(cmatrix))) == pytest.approx(1.0)


def test_boresight_refuses_the_wrong_count() -> None:
    """Eight numbers are not a rotation."""
    with pytest.raises(ValueError, match='nine elements, not 8'):
        boresight_from_cmatrix([0.0] * 8)


def test_boresight_refuses_a_row_with_no_direction() -> None:
    """A third row of zeros points nowhere."""
    with pytest.raises(ValueError, match='no direction to point'):
        boresight_from_cmatrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def test_vector_from_ra_dec_at_the_origin() -> None:
    """Zero right ascension and declination is the x axis."""
    assert vector_from_ra_dec(0.0, 0.0) == pytest.approx([1.0, 0.0, 0.0])


def test_vector_from_ra_dec_at_the_pole() -> None:
    """Ninety degrees of declination is the z axis."""
    assert vector_from_ra_dec(123.0, 90.0) == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)


def test_separation_of_a_direction_from_itself_is_zero() -> None:
    """The same direction twice is no separation at all."""
    direction = vector_from_ra_dec(83.99, 1.59)
    assert separation_px(direction, direction, plate_scale_urad=NAC_URAD) == 0.0


def test_separation_counts_pixels() -> None:
    """One plate scale of angle is one pixel."""
    one_pixel = math.degrees(NAC_URAD * 1e-6)
    separation = separation_px(
        vector_from_ra_dec(0.0, 0.0),
        vector_from_ra_dec(0.0, one_pixel),
        plate_scale_urad=NAC_URAD,
    )
    assert separation == pytest.approx(1.0, rel=1e-9)


def test_separation_resolves_a_hundredth_of_a_pixel() -> None:
    """The small-angle case the comparison exists for survives the arithmetic.

    An arc cosine of the dot product cannot do this: at this angle the cosine
    differs from one by less than a double's rounding.
    """
    tiny = math.degrees(NAC_URAD * 1e-6) * 0.01
    separation = separation_px(
        vector_from_ra_dec(0.0, 0.0),
        vector_from_ra_dec(0.0, tiny),
        plate_scale_urad=NAC_URAD,
    )
    assert separation == pytest.approx(0.01, rel=1e-6)


def test_separation_refuses_a_plate_scale_of_zero() -> None:
    """A camera with no plate scale measures nothing in pixels."""
    with pytest.raises(ValueError, match='a plate scale is positive'):
        separation_px(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), plate_scale_urad=0.0)


def test_separation_refuses_a_direction_with_no_length() -> None:
    """A zero vector has no angle to anything."""
    with pytest.raises(ValueError, match='no length'):
        separation_px(np.zeros(3), np.array([1.0, 0.0, 0.0]), plate_scale_urad=NAC_URAD)


IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_the_boresight_falls_at_the_origin_of_the_detector() -> None:
    """An answer agreeing with this one is no distance from its boresight."""
    offset = offset_in_camera_px(IDENTITY, np.array([0.0, 0.0, 1.0]), plate_scale_urad=NAC_URAD)
    assert offset == pytest.approx((0.0, 0.0))


def test_an_offset_along_the_first_axis_is_signed() -> None:
    """A direction displaced along camera x reports it on the first axis alone."""
    tilt = NAC_URAD * 1e-6
    direction = np.array([math.tan(tilt), 0.0, 1.0])
    offset = offset_in_camera_px(IDENTITY, direction, plate_scale_urad=NAC_URAD)
    assert offset[0] == pytest.approx(1.0, rel=1e-9)


def test_an_offset_along_the_first_axis_leaves_the_second_alone() -> None:
    """The two axes are reported independently."""
    tilt = NAC_URAD * 1e-6
    direction = np.array([math.tan(tilt), 0.0, 1.0])
    offset = offset_in_camera_px(IDENTITY, direction, plate_scale_urad=NAC_URAD)
    assert offset[1] == pytest.approx(0.0, abs=1e-12)


def test_the_sign_follows_the_direction() -> None:
    """A displacement the other way reports the other sign."""
    tilt = NAC_URAD * 1e-6
    direction = np.array([0.0, -math.tan(tilt), 1.0])
    offset = offset_in_camera_px(IDENTITY, direction, plate_scale_urad=NAC_URAD)
    assert offset[1] == pytest.approx(-1.0, rel=1e-9)


def test_the_cmatrix_carries_the_direction_into_the_camera_frame() -> None:
    """The attitude is what puts another answer on this camera's detector."""
    cmatrix = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    offset = offset_in_camera_px(cmatrix, np.array([1.0, 0.0, 0.0]), plate_scale_urad=NAC_URAD)
    assert offset == pytest.approx((0.0, 0.0))


def test_a_direction_behind_the_camera_is_refused() -> None:
    """Nothing behind the detector has a place on it."""
    with pytest.raises(ValueError, match='behind the camera'):
        offset_in_camera_px(IDENTITY, np.array([0.0, 0.0, -1.0]), plate_scale_urad=NAC_URAD)
