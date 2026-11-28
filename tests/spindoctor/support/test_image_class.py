"""Tests for ``spindoctor.support.image_class``, the shared image-quality decision."""

import math
from dataclasses import replace

import numpy as np
import pytest
from tests.packaged_sections import calibrated_camera, raw_camera, raw_thresholds

from spindoctor.support.image_class import (
    ImageQualityStatistics,
    ImageQualityThresholds,
    decide_image_class,
    image_quality_thresholds,
    measure_image_quality,
    missing_data_marker,
)


def test_marker_is_the_camera_marker_value() -> None:
    """The missing-data marker is the camera block's marker value."""
    camera = raw_camera()
    camera = replace(camera, noise=replace(camera.noise, marker_value=7.0))
    assert missing_data_marker(camera) == 7.0


def test_raw_thresholds_are_in_dn() -> None:
    """A raw_dn camera's thresholds are its DN limits and its own marker."""
    camera = raw_camera()
    limits = replace(
        camera.image_quality_thresholds,
        max_missing_frac_clean=0.25,
        max_overexposed_frac_clean=0.75,
        partial_dropout_min_frac=0.1,
        blank_max_dn=6.0,
        saturation_threshold_dn=4000.0,
        noisy_threshold_dn=11.0,
    )
    camera = replace(
        camera, noise=replace(camera.noise, marker_value=2.0), image_quality_thresholds=limits
    )
    assert image_quality_thresholds(camera) == ImageQualityThresholds(
        saturation_threshold_dn=4000.0,
        missing_data_marker_dn=2.0,
        max_saturation_frac_clean=0.75,
        max_missing_frac_clean=0.25,
        partial_dropout_min_frac=0.1,
        blank_max_dn=6.0,
        noisy_threshold=11.0,
    )


def test_calibrated_thresholds_have_no_saturation_limit() -> None:
    """A calibrated camera reads its I/F limits and an infinite saturation threshold."""
    camera = calibrated_camera()
    limits = camera.image_quality_thresholds
    thresholds = image_quality_thresholds(camera)
    assert math.isinf(thresholds.saturation_threshold_dn)
    assert thresholds.blank_max_dn == limits.blank_max_if
    assert thresholds.noisy_threshold == limits.noisy_threshold_if


@pytest.mark.parametrize(
    ('max_dn', 'saturation_frac', 'missing_frac', 'expected'),
    [
        (4.0, 0.9, 0.9, 'blank'),
        (4095.0, 0.81, 0.9, 'fully_overexposed'),
        (100.0, 0.80, 0.31, 'mostly_missing_data'),
        (100.0, 0.80, 0.30, 'clean'),
        (5.0, 0.0, 0.0, 'clean'),
    ],
    ids=['blank-first', 'overexposed-before-missing', 'missing', 'at-limits', 'peak-at-blank'],
)
def test_decision_order_and_limits(
    max_dn: float, saturation_frac: float, missing_frac: float, expected: str
) -> None:
    """Blank, then overexposed, then missing data; a share at its limit is allowed."""
    statistics = ImageQualityStatistics(
        max_dn=max_dn, saturation_frac=saturation_frac, missing_frac=missing_frac
    )
    thresholds = raw_thresholds(
        blank_max_dn=5.0, max_saturation_frac_clean=0.80, max_missing_frac_clean=0.30
    )
    assert decide_image_class(statistics, thresholds) == expected


def test_measure_counts_a_nan_marker() -> None:
    """With a NaN marker the missing share counts NaN pixels, and the peak skips them."""
    sensor = np.array([[np.nan, 2.0], [3.0, np.nan]])
    statistics = measure_image_quality(sensor, raw_thresholds(missing_data_marker_dn=math.nan))
    assert statistics == ImageQualityStatistics(max_dn=3.0, saturation_frac=0.0, missing_frac=0.5)
