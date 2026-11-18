"""Tests for ``spindoctor.support.image_class``, the shared image-quality decision."""

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from spindoctor.support.image_class import (
    ImageQualityStatistics,
    ImageQualityThresholds,
    decide_image_class,
    image_quality_thresholds,
    measure_image_quality,
    missing_data_marker,
)

_RAW_BLOCK: dict[str, Any] = {
    'data_units': 'raw_dn',
    'noise': {'marker_value': 0},
    'image_quality_thresholds': {
        'saturation_threshold_dn': 4095,
        'blank_max_dn': 5.0,
        'noisy_threshold_dn': 10.0,
    },
}

_CALIBRATED_BLOCK: dict[str, Any] = {
    'data_units': 'calibrated_if',
    'image_quality_thresholds': {'blank_max_if': 1.0e-4, 'noisy_threshold_if': 0.005},
}


@pytest.mark.parametrize(
    ('block', 'expected'),
    [
        ({'data_units': 'raw_dn'}, 0.0),
        ({'data_units': 'calibrated_if', 'noise': {}}, math.nan),
        ({'data_units': 'calibrated_if', 'noise': {'marker_value': None}}, math.nan),
        ({'data_units': 'calibrated_if', 'noise': {'marker_value': 'nan'}}, math.nan),
        ({'data_units': 'raw_dn', 'noise': {'marker_value': 7}}, 7.0),
    ],
    ids=['raw-default', 'calibrated-default', 'null', 'nan-string', 'explicit'],
)
def test_marker_reading(block: dict[str, Any], expected: float) -> None:
    """The marker defaults by unit class, and null or 'NaN' reads as NaN."""
    marker = missing_data_marker(block)
    assert (math.isnan(marker) and math.isnan(expected)) or marker == expected


def test_marker_rejects_a_non_nan_string() -> None:
    """A marker string other than NaN is refused."""
    with pytest.raises(ValueError, match='must be "NaN" or numeric'):
        missing_data_marker({'data_units': 'raw_dn', 'noise': {'marker_value': 'zero'}})


def test_calibrated_thresholds_have_no_saturation_limit() -> None:
    """A calibrated block reads its I/F limits and an infinite saturation threshold."""
    # The NaN marker is replaced so the comparison does not depend on NaN equality.
    thresholds = replace(image_quality_thresholds(_CALIBRATED_BLOCK), missing_data_marker_dn=0.0)
    assert thresholds == ImageQualityThresholds(
        saturation_threshold_dn=math.inf, blank_max_dn=1.0e-4, noisy_threshold=0.005
    )


def test_raw_thresholds_require_the_blank_limit() -> None:
    """A raw block without blank_max_dn is refused, naming the field."""
    iqt = {
        key: value
        for key, value in _RAW_BLOCK['image_quality_thresholds'].items()
        if key != 'blank_max_dn'
    }
    with pytest.raises(ValueError, match="missing required field 'blank_max_dn'"):
        image_quality_thresholds({**_RAW_BLOCK, 'image_quality_thresholds': iqt})


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
    assert decide_image_class(statistics, ImageQualityThresholds()) == expected


def test_measure_counts_a_nan_marker() -> None:
    """With a NaN marker the missing share counts NaN pixels, and the peak skips them."""
    sensor = np.array([[np.nan, 2.0], [3.0, np.nan]])
    statistics = measure_image_quality(
        sensor, ImageQualityThresholds(missing_data_marker_dn=math.nan)
    )
    assert statistics == ImageQualityStatistics(max_dn=3.0, saturation_frac=0.0, missing_frac=0.5)
