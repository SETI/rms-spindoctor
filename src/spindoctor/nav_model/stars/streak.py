"""Measure the streak a smeared frame's stars draw.

A star that moves across the detector during the exposure is recorded as a streak along
its motion.  The star techniques take the streak from the image: the bright stars of a
frame whose predicted smear is long enough to matter are each measured from the second
moments of their light, and the streak is the one most of them agree on.  The predicted
smear decides whether a frame is measured and whether a measured streak can be the
motion's, and where the observation's host vouches for its prediction it stands in for a
streak the image does not show (see :func:`streak_limits` and :func:`predicted_streak_vu`).

Each measurement finds the brightest pixel near a predicted star (any star serves, since
every star of a frame draws the same streak), keeps the light connected to that pixel
above the larger of three noise sigmas and a fraction of the peak, and turns the light's
second-moment tensor into the streak along its major axis: a uniform segment of length
``L`` has variance ``L**2 / 12`` along itself, and the point-spread function adds the
same variance across and along it, so the length is ``sqrt(12 * (major - minor))``.
A candidate is set aside when its light holds a saturated pixel (a bloom runs along a
column whatever the motion), reaches the edge of its box (the glare of a ring or a body),
or is too faint for its shape to be measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from spindoctor.support.misc import mad_std
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'MAX_STREAK_TO_PREDICTED',
    'MIN_PREDICTED_SMEAR_PX',
    'MIN_STREAK_PX',
    'StreakLimits',
    'StreakMeasurement',
    'consensus_streak',
    'measure_streak',
    'predicted_streak_vu',
    'star_streak_axis',
    'streak_limits',
]

MIN_STREAK_PX = 4.5
"""Measured streak length in pixels from which the star techniques centroid a streak.

Below it the 7 x 7 centroid box placed on a star's brightest pixel holds nearly the
whole streak and measures it as well as the streak centroid does or better; from it the
brightest pixel's random place along the flat-topped streak moves the box's centroid
along the streak by a growing fraction of its length.  Measured on Gaussian streaks
(sigma 0.54 and 0.77 px, total flux 60 to 1000 noise sigmas): at 4.5 px the two
centroids' errors are equal within 0.01 px rms and from 5 px the streak centroid's is
the smaller at every flux.  On F ring frames whose measured streak is 4.5 px or longer
the streak centroid gives the smaller star-to-star scatter along the streak on 76% of 21
frames, its median falling from 1.58 to 1.17 px; on frames whose streak is 2 to 4.5 px it
does so on 52% of 86 frames, which is no gain.
"""

MAX_STREAK_TO_PREDICTED = 2.0
"""Largest ratio of a measured streak's length to a predicted smear no host vouches for.

A longer measurement is the work of something other than the motion, such as the blooms
of saturated stars running along a column, which the saturation mask of a calibrated
image does not show.  A prediction that may overstate the streak several times but
rarely understates it by much is held to this ratio; the instrument chapters of the
developer guide give the evidence.  A host that vouches for its prediction sets its own
ratio (see :func:`streak_limits`).
"""

MIN_PREDICTED_SMEAR_PX = 2.0
"""Predicted smear length in pixels below which a frame's streak is not measured.

Applies to a prediction no host vouches for.  Below it no star of the frame is
centroided as a streak; this keeps every frame of negligible predicted motion exactly as
it is measured without a streak.
"""


@dataclass(frozen=True)
class StreakLimits:
    """How a frame's predicted smear bounds the streak recorded for it.

    Parameters:
        min_predicted_px: Median predicted smear in pixels below which the frame's streak
            is not measured.
        max_streak_to_predicted: Largest ratio of a measured streak's length to the
            median predicted smear that is recorded.
        prediction_stands_in: Whether the predicted streak is recorded when no streak is
            measured that the stars agree on.
    """

    min_predicted_px: float
    max_streak_to_predicted: float
    prediction_stands_in: bool


def streak_limits(vouched_ratio: float | None) -> StreakLimits:
    """Return the limits a frame's predicted smear sets on its streak.

    A prediction no host vouches for (``vouched_ratio`` None) is held to
    :data:`MIN_PREDICTED_SMEAR_PX` and :data:`MAX_STREAK_TO_PREDICTED`, and never stands
    in for a measurement.  A vouched-for prediction is held to its host's ratio, is
    measured from :data:`MIN_STREAK_PX` over that ratio (a frame predicted shorter could
    record a streak long enough to centroid on only by exceeding the ratio, so measuring
    it could not change how its stars are centroided), and stands in for a streak the
    image does not show.

    Parameters:
        vouched_ratio: The largest ratio of a measured streak to the predicted smear the
            host has found on its images, or None when the host does not vouch for its
            prediction.

    Returns:
        The limits.

    Raises:
        ValueError: if ``vouched_ratio`` is not greater than 1.
    """
    if vouched_ratio is None:
        return StreakLimits(MIN_PREDICTED_SMEAR_PX, MAX_STREAK_TO_PREDICTED, False)
    if not vouched_ratio > 1.0:
        raise ValueError(f'vouched_ratio must be greater than 1; got {vouched_ratio!r}')
    return StreakLimits(MIN_STREAK_PX / vouched_ratio, vouched_ratio, True)


_STARS_EXAMINED = 12
"""Number of the brightest unflagged stars whose neighborhoods are measured."""

_SEARCH_HALF_PX = 30
"""Half-width in pixels of the window searched for a star's brightest pixel.

Wide enough for the pointing errors of the frames the star techniques navigate; a
window that misses its own star but holds another serves as well, since every star of
the frame draws the same streak.
"""

_BOX_HALF_PX = 15
"""Half-width in pixels of the box whose light is measured around a brightest pixel.

Holds a streak of up to about 13 px whose brightest pixel lies at one end."""

_BORDER_PX = 3
"""Width in pixels of the box's border that gives the local background and spread."""

_MIN_PEAK_SIGMAS = 20.0
"""Local noise sigmas the brightest pixel must stand above the background."""

_THRESHOLD_SIGMAS = 3.0
"""Local noise sigmas a pixel must stand above the background to count as the star's."""

_PEAK_FRACTION = 0.1
"""Fraction of the peak a pixel must reach to count as the star's.

Keeps the noise of a quantized image, whose single-quantum pixels would otherwise join
up across the box, out of a faint star's light."""

_MIN_AGREEING = 3
"""Number of measured stars that must agree on the streak."""

_MAX_ANGLE_DEG = 20.0
"""Largest angle in degrees between an agreeing star's streak and the consensus."""

_MAX_LENGTH_FRACTION = 0.3
"""Largest fractional difference of an agreeing streak's length from the consensus."""


@dataclass(frozen=True)
class StreakMeasurement:
    """The streak a frame's stars agree on.

    Parameters:
        streak_vu: The streak ``(v, u)`` in pixels from one end to the other, with
            ``v >= 0`` (a streak has no direction of travel).
        n_measured: Number of stars whose streak could be measured.
        n_agreeing: Number of those that agree on ``streak_vu``.
    """

    streak_vu: tuple[float, float]
    n_measured: int
    n_agreeing: int


def _oriented(axis_v: float, axis_u: float) -> tuple[float, float]:
    """Return the axis pointing to ``v >= 0`` (``u > 0`` when ``v`` is zero)."""
    if axis_v < 0.0 or (axis_v == 0.0 and axis_u < 0.0):
        return -axis_v, -axis_u
    return axis_v, axis_u


def predicted_streak_vu(moves_vu: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the streak a frame's stars are predicted to draw.

    Parameters:
        moves_vu: Each examined star's predicted smear ``(v, u)`` in pixels, from the
            start of the exposure to its end; not empty.

    Returns:
        The component-wise median smear, oriented to ``v >= 0`` like a measured streak.
    """
    moves = np.asarray(moves_vu, dtype=np.float64)
    return _oriented(float(np.median(moves[:, 0])), float(np.median(moves[:, 1])))


def star_streak_axis(
    image: NDArrayFloatType,
    peak_vu: tuple[int, int],
    *,
    noise_sigma: float,
    saturation_mask: NDArrayBoolType,
) -> tuple[float, float] | None:
    """Return the streak drawn by the source whose brightest pixel is ``peak_vu``.

    The box ``_BOX_HALF_PX`` either side of the pixel must lie inside the image.  Its
    border, ``_BORDER_PX`` wide, gives the background (the median) and the spread (the
    larger of its scaled median absolute deviation and ``noise_sigma``).  The pixel must
    stand ``_MIN_PEAK_SIGMAS`` spreads above the background.  The source's light is the
    pixels connected to it that stand above the larger of ``_THRESHOLD_SIGMAS`` spreads
    and ``_PEAK_FRACTION`` of the peak; it must hold no saturated pixel and stay clear
    of the border.

    Parameters:
        image: 2-D image array.
        peak_vu: ``(v, u)`` indices of the source's brightest pixel.
        noise_sigma: Image noise sigma, the floor of the local spread.
        saturation_mask: True at saturated pixels; same shape as ``image``.

    Returns:
        The streak ``(v, u)`` in pixels along the light's major axis, of length
        ``sqrt(12 * (major - minor))`` from its second-moment eigenvalues and oriented to
        ``v >= 0``, or None when the source is set aside.
    """
    p_v, p_u = peak_vu
    half = _BOX_HALF_PX
    rows, cols = image.shape
    if not (half <= p_v < rows - half and half <= p_u < cols - half):
        return None
    rows = slice(p_v - half, p_v + half + 1)
    cols = slice(p_u - half, p_u + half + 1)
    box = np.asarray(image[rows, cols], np.float64)
    border = np.ones(box.shape, bool)
    border[_BORDER_PX:-_BORDER_PX, _BORDER_PX:-_BORDER_PX] = False
    background = float(np.median(box[border]))
    spread = max(mad_std(box[border]), noise_sigma)
    light = box - background
    peak = float(light[half, half])
    if peak < _MIN_PEAK_SIGMAS * spread:
        return None
    labels, _ = label(light > max(_THRESHOLD_SIGMAS * spread, _PEAK_FRACTION * peak))
    source = labels == labels[half, half]
    if bool((source & border).any()):
        return None
    if bool((source & saturation_mask[rows, cols]).any()):
        return None
    weights = np.where(source, light, 0.0)
    total = float(weights.sum())
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    gv, gu = np.meshgrid(offsets, offsets, indexing='ij')
    cv = float((weights * gv).sum()) / total
    cu = float((weights * gu).sum()) / total
    cvv = float((weights * (gv - cv) ** 2).sum()) / total
    cuu = float((weights * (gu - cu) ** 2).sum()) / total
    cvu = float((weights * (gv - cv) * (gu - cu)).sum()) / total
    evals, evecs = np.linalg.eigh(np.array([[cvv, cvu], [cvu, cuu]]))
    length = math.sqrt(max(0.0, 12.0 * float(evals[1] - evals[0])))
    return _oriented(float(evecs[0, 1]) * length, float(evecs[1, 1]) * length)


def measure_streak(
    image: NDArrayFloatType,
    stars_vu: list[tuple[float, float]],
    *,
    noise_sigma: float,
    saturation_mask: NDArrayBoolType,
) -> StreakMeasurement | None:
    """Return the streak the brightest stars of a frame agree on.

    For each of the first ``_STARS_EXAMINED`` predictions the brightest pixel within
    ``_SEARCH_HALF_PX`` of it on each axis, less the boxes already measured for
    earlier predictions, is measured by :func:`star_streak_axis`, and the streaks
    measured go to :func:`consensus_streak`.

    Parameters:
        image: 2-D image array.
        stars_vu: Predicted ``(v, u)`` of the frame's stars, brightest first.
        noise_sigma: Image noise sigma, the floor of each local spread.
        saturation_mask: True at saturated pixels; same shape as ``image``.

    Returns:
        The measurement, or None when fewer than ``_MIN_AGREEING`` stars, or fewer than
        half of those measured, agree.
    """
    rows, cols = image.shape
    # Pixels within a measured box are spent: a later window that holds the same star
    # looks past it to the next brightest source.
    spent = np.zeros(image.shape, bool)
    axes: list[tuple[float, float]] = []
    for pred_v, pred_u in stars_vu[:_STARS_EXAMINED]:
        v_lo = max(0, math.floor(pred_v) - _SEARCH_HALF_PX)
        u_lo = max(0, math.floor(pred_u) - _SEARCH_HALF_PX)
        v_hi = min(rows, math.floor(pred_v) + _SEARCH_HALF_PX + 1)
        u_hi = min(cols, math.floor(pred_u) + _SEARCH_HALF_PX + 1)
        if v_hi <= v_lo or u_hi <= u_lo:
            continue
        window = np.where(
            spent[v_lo:v_hi, u_lo:u_hi], -np.inf, np.asarray(image[v_lo:v_hi, u_lo:u_hi])
        )
        if not np.isfinite(window).any():
            continue
        w_v, w_u = np.unravel_index(int(np.argmax(window)), window.shape)
        p_v, p_u = v_lo + int(w_v), u_lo + int(w_u)
        spent[
            max(0, p_v - _BOX_HALF_PX) : p_v + _BOX_HALF_PX + 1,
            max(0, p_u - _BOX_HALF_PX) : p_u + _BOX_HALF_PX + 1,
        ] = True
        axis = star_streak_axis(
            image, (p_v, p_u), noise_sigma=noise_sigma, saturation_mask=saturation_mask
        )
        if axis is not None:
            axes.append(axis)
    return consensus_streak(axes)


def consensus_streak(axes: list[tuple[float, float]]) -> StreakMeasurement | None:
    """Return the streak that the measured stars' streaks agree on.

    The consensus direction is the major axis of the element-wise median of the streaks'
    outer products, and the consensus length their median length.  A streak agrees when
    it lies within ``_MAX_ANGLE_DEG`` of that direction and its length within
    ``_MAX_LENGTH_FRACTION`` of that length.  The streak returned is the mean of the
    agreeing streaks, each turned to the consensus direction.

    Parameters:
        axes: The measured streaks ``(v, u)`` in pixels, in either direction.

    Returns:
        The measurement, or None when fewer than ``_MIN_AGREEING`` streaks, or fewer than
        half of them, agree.
    """
    if len(axes) < _MIN_AGREEING:
        return None
    arr = np.asarray(axes, np.float64)
    outer = np.median(
        np.column_stack([arr[:, 0] ** 2, arr[:, 1] ** 2, arr[:, 0] * arr[:, 1]]), axis=0
    )
    _evals, evecs = np.linalg.eigh(np.array([[outer[0], outer[2]], [outer[2], outer[1]]]))
    direction = np.asarray(_oriented(float(evecs[0, 1]), float(evecs[1, 1])))
    lengths = np.hypot(arr[:, 0], arr[:, 1])
    length = float(np.median(lengths))
    agreeing: list[NDArrayFloatType] = []
    for row, row_len in zip(arr, lengths, strict=True):
        if row_len <= 0.0 or length <= 0.0:
            continue
        cos = float(np.dot(row, direction)) / float(row_len)
        angle = math.degrees(math.acos(min(1.0, abs(cos))))
        length_off = abs(float(row_len) / length - 1.0)
        if angle <= _MAX_ANGLE_DEG and length_off <= _MAX_LENGTH_FRACTION:
            agreeing.append(np.asarray(row if cos >= 0.0 else -row, np.float64))
    if len(agreeing) < _MIN_AGREEING or 2 * len(agreeing) < len(axes):
        return None
    mean = np.mean(np.asarray(agreeing), axis=0)
    return StreakMeasurement(
        streak_vu=_oriented(float(mean[0]), float(mean[1])),
        n_measured=len(axes),
        n_agreeing=len(agreeing),
    )
