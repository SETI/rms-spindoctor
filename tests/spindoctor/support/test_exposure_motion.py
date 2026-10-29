"""Tests for ``spindoctor.support.exposure_motion``."""

import math

import cspyce
import numpy as np
import pytest

from spindoctor.support.exposure_motion import (
    motion_from_attitudes,
    motion_from_rate,
    rotate_ra_dec,
)
from spindoctor.support.types import NDArrayFloatType

_AXIS = np.array([0.2, -0.6, 0.77]) / np.linalg.norm([0.2, -0.6, 0.77])
"""The body-frame axis the test attitude turns about."""

_RATE = 8.0e-6
"""The test attitude's turning rate in radians per second."""


def _attitude(t: float) -> NDArrayFloatType:
    """Return a J2000-to-body rotation turning at a constant rate about a fixed axis."""
    start = np.asarray(cspyce.axisar(np.array([1.0, 0.4, -0.3]) / 1.1, 0.9), dtype=np.float64)
    turned = np.asarray(cspyce.axisar(_AXIS, _RATE * t), dtype=np.float64)
    return turned @ start


def test_rate_over_a_window_reproduces_a_constant_rate_motion() -> None:
    """For a camera turning steadily, the window's mean rate gives the exposure's motion."""
    exact = motion_from_attitudes(_attitude(-0.6), _attitude(0.0), _attitude(0.6), source='x')
    rate = motion_from_rate(
        _attitude(-8.0), _attitude(0.0), _attitude(8.0), window_s=16.0, exposure_s=1.2, source='x'
    )
    np.testing.assert_allclose(rate.start_rotation, exact.start_rotation, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(rate.end_rotation, exact.end_rotation, rtol=0.0, atol=1e-14)


def test_motion_rotations_carry_a_direction_to_where_it_lands_at_mid_exposure() -> None:
    """The end rotation turns a direction to the one landing where it lands at the end."""
    middle, end = _attitude(0.0), _attitude(0.6)
    motion = motion_from_attitudes(_attitude(-0.6), middle, end, source='x')
    direction = np.array([0.36, 0.48, 0.8])
    np.testing.assert_allclose(
        middle @ (motion.end_rotation @ direction), end @ direction, rtol=0.0, atol=1e-15
    )


def test_rate_window_must_outlast_the_exposure() -> None:
    """A rate measured over less than the exposure cannot describe the whole exposure."""
    with pytest.raises(ValueError, match=r'window_s \(16\.0\) must be longer than exposure_s'):
        motion_from_rate(np.eye(3), np.eye(3), np.eye(3), window_s=16.0, exposure_s=16.0, source='')


def test_rotate_ra_dec_about_the_pole_shifts_right_ascension() -> None:
    """A turn about the J2000 pole moves right ascension by the turn and keeps declination."""
    turn = 0.25
    rotation = np.array(
        [
            [math.cos(turn), -math.sin(turn), 0.0],
            [math.sin(turn), math.cos(turn), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ra, dec = rotate_ra_dec(np.array([6.2, 1.0]), np.array([0.3, -1.2]), rotation)
    np.testing.assert_allclose(ra, [6.45 - 2.0 * math.pi, 1.25], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(dec, [0.3, -1.2], rtol=0.0, atol=1e-14)
