"""How the sky moves across a camera during one exposure.

A camera that turns while its shutter is open records each star as a streak.  The motion
is described here by two rotations of J2000: a star whose direction is ``d`` lands, at
the start of the exposure, where the direction ``start_rotation @ d`` lands at the middle
of the exposure, and at the end where ``end_rotation @ d`` does.  Projecting the rotated
directions through the camera's attitude at mid-exposure therefore gives the ends of
every star's streak, with the camera's absolute pointing taken from one place and only
its turning during the exposure from another.

With ``C(t)`` the rotation taking a J2000 vector into the camera's (or the spacecraft's)
frame, the rotation for time ``t`` is ``C(t_mid).T @ C(t)``.  A constant rotation between
the spacecraft and the camera cancels from that product, so an attitude history of the
spacecraft bus gives the same motion as one of the camera.
"""

from dataclasses import dataclass

import cspyce
import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'ExposureMotion',
    'motion_from_attitudes',
    'motion_from_rate',
    'rotate_ra_dec',
]


@dataclass(frozen=True)
class ExposureMotion:
    """The turning of a camera during one exposure.

    Parameters:
        start_rotation: 3x3 rotation of J2000 taking a star's direction to the direction
            that lands, at mid-exposure, where the star lands at the start of the
            exposure.
        end_rotation: The same for the end of the exposure.
        source: What the motion was derived from, as recorded in a navigation's
            metadata.
        max_streak_to_predicted: The largest ratio of a streak measured on the image to
            the smear this motion predicts that the host has found on its images, when
            the host vouches for the predicted smear's length and direction; None when it
            does not.  A vouched-for motion lets the star model hold a measured streak to
            this ratio and stand the predicted streak in for one the image does not show
            (see :func:`~spindoctor.nav_model.stars.streak.streak_limits`).
    """

    start_rotation: NDArrayFloatType
    end_rotation: NDArrayFloatType
    source: str
    max_streak_to_predicted: float | None = None


def motion_from_attitudes(
    start: NDArrayFloatType,
    middle: NDArrayFloatType,
    end: NDArrayFloatType,
    *,
    source: str,
    max_streak_to_predicted: float | None = None,
) -> ExposureMotion:
    """Return the motion between attitudes at the start, middle, and end of an exposure.

    Parameters:
        start: J2000-to-frame rotation at the start of the exposure.
        middle: J2000-to-frame rotation at mid-exposure.
        end: J2000-to-frame rotation at the end of the exposure.
        source: What the attitudes were read from.
        max_streak_to_predicted: The motion's
            :attr:`~ExposureMotion.max_streak_to_predicted`.

    Returns:
        The motion, whose rotations are ``middle.T @ start`` and ``middle.T @ end``.
    """
    return ExposureMotion(
        start_rotation=middle.T @ start,
        end_rotation=middle.T @ end,
        source=source,
        max_streak_to_predicted=max_streak_to_predicted,
    )


def motion_from_rate(
    before: NDArrayFloatType,
    middle: NDArrayFloatType,
    after: NDArrayFloatType,
    *,
    window_s: float,
    exposure_s: float,
    source: str,
) -> ExposureMotion:
    """Return the motion of a camera turning at its mean rate over a longer window.

    The camera is taken to turn at a constant rate about a fixed axis: the rotation
    between the attitudes at the window's two ends, scaled by the fraction of the window
    that half the exposure is, takes the attitude at mid-exposure to the attitudes at the
    exposure's start and end.  The window must be longer than the exposure, since a rate
    measured over a shorter span cannot describe the turning across the whole exposure.
    The motion does not vouch for its predicted smear.

    Parameters:
        before: J2000-to-frame rotation at the start of the window.
        middle: J2000-to-frame rotation at mid-exposure, the window's center.
        after: J2000-to-frame rotation at the end of the window.
        window_s: Length of the window in seconds; longer than the exposure.
        exposure_s: Length of the exposure in seconds.
        source: What the attitudes were read from.

    Returns:
        The motion.

    Raises:
        ValueError: if ``window_s`` is not longer than ``exposure_s``.
    """
    if not window_s > exposure_s:
        raise ValueError(f'window_s ({window_s!r}) must be longer than exposure_s ({exposure_s!r})')
    axis, angle = cspyce.raxisa(after @ before.T)
    half_turn = 0.5 * exposure_s / window_s * float(angle)
    forward = np.asarray(cspyce.axisar(axis, half_turn), dtype=np.float64)
    return motion_from_attitudes(forward.T @ middle, middle, forward @ middle, source=source)


def rotate_ra_dec(
    ra: NDArrayFloatType, dec: NDArrayFloatType, rotation: NDArrayFloatType
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return sky directions turned by a rotation of J2000.

    Parameters:
        ra: Right ascensions in radians.
        dec: Declinations in radians, the same shape as ``ra``.
        rotation: 3x3 rotation applied to each direction's unit vector.

    Returns:
        The rotated directions' right ascensions, in ``[0, 2 pi)``, and declinations, in
        radians.
    """
    cos_dec = np.cos(dec)
    vectors = np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)
    turned = vectors @ rotation.T
    new_ra = np.mod(np.arctan2(turned[..., 1], turned[..., 0]), 2.0 * np.pi)
    # arctan2 rather than arcsin of z: arcsin loses precision near the poles, where a
    # small turn moves z by much less than the turn itself.
    new_dec = np.arctan2(turned[..., 2], np.hypot(turned[..., 0], turned[..., 1]))
    return new_ra, new_dec
