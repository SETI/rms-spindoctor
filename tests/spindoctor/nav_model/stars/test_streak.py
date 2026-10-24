"""Tests for ``spindoctor.nav_model.stars.streak``."""

import math

import numpy as np
import pytest
from tests.spindoctor.star_streak_images import draw_streak

from spindoctor.nav_model.stars.streak import (
    MAX_STREAK_TO_PREDICTED,
    MIN_PREDICTED_SMEAR_PX,
    StreakLimits,
    consensus_streak,
    measure_streak,
    predicted_streak_vu,
    star_streak_axis,
    streak_limits,
)

_SHAPE = (128, 128)
_NO_SATURATION = np.zeros(_SHAPE, dtype=bool)


def _noise() -> np.ndarray:
    """Return a frame of unit Gaussian noise."""
    return np.random.default_rng(765).normal(0.0, 1.0, size=_SHAPE)


def _at(angle_deg: float, length_px: float = 6.0) -> tuple[float, float]:
    """Return the streak of ``length_px`` at ``angle_deg`` from the v axis."""
    return (
        length_px * math.cos(math.radians(angle_deg)),
        length_px * math.sin(math.radians(angle_deg)),
    )


@pytest.mark.parametrize(
    'axes',
    [
        [_at(0.0), _at(0.0), _at(0.0), _at(60.0), _at(60.0), _at(60.0)],
        [_at(0.0), _at(0.0), _at(0.0, 3.0), _at(0.0, 3.0)],
        [_at(0.0), _at(0.0), _at(60.0)],
        [_at(0.0)] * 3 + [_at(0.0, 4.0)] * 2 + [_at(0.0, 8.5)] * 2,
    ],
    ids=['split_direction', 'split_length', 'two_agree', 'three_of_seven_agree'],
)
def test_consensus_streak_needs_three_and_half_to_agree(axes: list[tuple[float, float]]) -> None:
    """Streaks that do not agree in direction and length, three and half of them, give none."""
    assert consensus_streak(axes) is None


def test_consensus_streak_is_the_mean_of_the_agreeing_streaks() -> None:
    """Agreeing streaks given in either direction average to one streak with v >= 0."""
    measured = consensus_streak([_at(10.0), _at(190.0), _at(12.0), _at(8.0), _at(70.0)])
    expected = np.mean([_at(10.0), _at(10.0), _at(12.0), _at(8.0)], axis=0)
    assert measured is not None
    assert measured.n_agreeing == 4
    assert measured.streak_vu[0] == pytest.approx(expected[0], abs=1e-9)
    assert measured.streak_vu[1] == pytest.approx(expected[1], abs=1e-9)


def test_star_streak_axis_measures_the_streak_length_without_the_psf() -> None:
    """The streak's length is its own, the point-spread function's width taken out."""
    image = _noise()
    draw_streak(image, (64.0, 64.0), (6.0, -1.0), flux=3000.0, sigma=1.2)
    axis = star_streak_axis(image, (64, 64), noise_sigma=1.0, saturation_mask=_NO_SATURATION)
    assert axis is not None
    assert math.hypot(axis[0], axis[1]) == pytest.approx(math.hypot(6.0, -1.0), abs=0.3)


def test_star_streak_axis_sets_aside_a_star_too_faint_to_measure() -> None:
    """A star whose peak stands less than the floor above the noise is not measured."""
    image = _noise()
    draw_streak(image, (64.0, 64.0), (6.0, -1.0), flux=100.0)
    peak = np.unravel_index(int(np.argmax(image[60:69, 60:69])), (9, 9))
    axis = star_streak_axis(
        image,
        (60 + int(peak[0]), 60 + int(peak[1])),
        noise_sigma=1.0,
        saturation_mask=_NO_SATURATION,
    )
    assert axis is None


def test_measure_streak_counts_each_star_once() -> None:
    """Predictions that all find the same star measure it once, which is too few to agree."""
    image = _noise()
    draw_streak(image, (64.0, 64.0), (6.0, -1.0), flux=3000.0)
    predictions = [(60.0, 60.0), (66.0, 70.0), (70.0, 58.0), (58.0, 66.0)]
    measured = measure_streak(image, predictions, noise_sigma=1.0, saturation_mask=_NO_SATURATION)
    assert measured is None


def test_star_streak_axis_sets_aside_light_that_reaches_its_box_edge() -> None:
    """A star on a band of glare that runs out of its box is not measured."""
    image = _noise()
    image[:, 60:63] += 200.0
    draw_streak(image, (64.0, 61.0), (6.0, -1.0), flux=3000.0)
    peak = np.unravel_index(int(np.argmax(image)), image.shape)
    axis = star_streak_axis(
        image,
        (int(peak[0]), int(peak[1])),
        noise_sigma=1.0,
        saturation_mask=_NO_SATURATION,
    )
    assert axis is None


def test_star_streak_axis_sets_aside_a_saturated_star() -> None:
    """A star whose light holds a saturated pixel is not measured."""
    image = _noise()
    draw_streak(image, (64.0, 64.0), (6.0, -1.0), flux=3000.0)
    saturation = np.zeros(_SHAPE, dtype=bool)
    saturation[64, 64] = True
    axis = star_streak_axis(image, (64, 64), noise_sigma=1.0, saturation_mask=saturation)
    assert axis is None


def test_streak_limits_of_a_prediction_no_host_vouches_for() -> None:
    """An unvouched prediction keeps the default floor and ceiling and never stands in."""
    assert streak_limits(None) == StreakLimits(
        MIN_PREDICTED_SMEAR_PX, MAX_STREAK_TO_PREDICTED, False
    )


def test_streak_limits_of_a_vouched_prediction() -> None:
    """A vouched prediction is held to its ratio, measured from 4.5 px over it, and stands in."""
    assert streak_limits(1.5) == StreakLimits(3.0, 1.5, True)


def test_streak_limits_refuse_a_ratio_that_cannot_hold_a_streak() -> None:
    """A ratio of 1 or less would refuse the prediction's own length."""
    with pytest.raises(ValueError, match=r'vouched_ratio must be greater than 1; got 1\.0'):
        streak_limits(1.0)


def test_predicted_streak_is_the_oriented_median_smear() -> None:
    """The predicted streak is the component-wise median, pointing to v >= 0."""
    assert predicted_streak_vu([(-6.0, 1.0), (-7.0, 2.0), (-8.0, 2.5)]) == (7.0, -2.0)
