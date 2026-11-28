"""The scene editor's status text for a rendered frame.

The status bar describes a frame in the terms the navigator reads it: its units, its
value range, the share of its pixels carrying the missing-data marker, and the class the
navigator's image classifier gives it.  :func:`read_frame_as_navigator` classifies the
frame with the navigator's own decision (:mod:`spindoctor.support.image_class`), and
:func:`frame_units_summary` puts the result into one line.  :func:`has_planted_offset`
says whether a scene's navigated frame differs from the offset-free picture on screen.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.config.sections_camera import CameraBlock
from spindoctor.sim.scene.boundary import PLANTED_POINTING_KEYS
from spindoctor.support.image_class import (
    decide_image_class,
    image_quality_thresholds,
    measure_image_quality,
    missing_data_mask,
)
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'NavigatorFrameReading',
    'frame_units_name',
    'frame_units_summary',
    'has_planted_offset',
    'read_frame_as_navigator',
]


@dataclass(frozen=True)
class NavigatorFrameReading:
    """How the navigator's image classifier reads a frame's marker and peak.

    Parameters:
        units: ``'DN'`` for a ``raw_dn`` block, ``'I/F'`` for a ``calibrated_if`` one.
        marker: The missing-data marker (see
            :func:`~spindoctor.support.image_class.missing_data_marker`).
        missing_frac: Fraction of pixels carrying the marker (NaN pixels when the
            marker is NaN).
        image_class: The classifier's outcome: ``'blank'``, ``'fully_overexposed'``,
            ``'mostly_missing_data'``, or ``'clean'``.
        blank_max: The peak below which the frame is blank.
        max_missing_frac: The marker share, outside the unread lines, above which the
            frame is refused.
        max_overexposed_frac: The saturated share above which the frame is
            overexposed.
    """

    units: str
    marker: float
    missing_frac: float
    image_class: str
    blank_max: float
    max_missing_frac: float
    max_overexposed_frac: float


def read_frame_as_navigator(
    image: NDArrayFloatType, inst_config: CameraBlock
) -> NavigatorFrameReading:
    """Classify a frame's sensor pixels the way the navigator's image classifier does.

    The thresholds and the decision are the navigator's own
    (:func:`~spindoctor.support.image_class.image_quality_thresholds` and
    :func:`~spindoctor.support.image_class.decide_image_class`).  Like the
    navigator, this counts the pixels carrying the marker and fills them with 0
    before it measures the frame.  ``image`` is the sensor frame without
    extended-FOV padding, which the navigator masks out.

    Parameters:
        image: The frame the navigator reads, with the planted offset applied.
        inst_config: The navigator block of that frame.

    Returns:
        The reading, with the class and the limits that decided it.
    """
    arr = np.asarray(image, dtype=np.float64)
    thresholds = image_quality_thresholds(inst_config)
    marker = thresholds.missing_data_marker_dn
    missing = missing_data_mask(arr, marker)
    missing_frac = float(missing.mean()) if arr.size else 0.0
    filled = np.where(missing, 0.0, arr)
    statistics = measure_image_quality(filled, thresholds, missing_mask=missing)
    return NavigatorFrameReading(
        units=frame_units_name(inst_config),
        marker=marker,
        missing_frac=missing_frac,
        image_class=decide_image_class(statistics, thresholds),
        blank_max=thresholds.blank_max_dn,
        max_missing_frac=thresholds.max_missing_frac_clean,
        max_overexposed_frac=thresholds.max_saturation_frac_clean,
    )


def frame_units_name(inst_config: CameraBlock) -> str:
    """Return the units in which the navigator reads this block's frames.

    Parameters:
        inst_config: A navigator block.

    Returns:
        ``'DN'`` for a ``raw_dn`` block, ``'I/F'`` for a ``calibrated_if`` one.
    """
    return 'DN' if inst_config.data_units == 'raw_dn' else 'I/F'


def has_planted_offset(sim_params: Mapping[str, Any]) -> bool:
    """Whether a scene plants a pointing offset or roll the navigated frame applies.

    Parameters:
        sim_params: The scene mapping.

    Returns:
        True when any of ``offset_v``, ``offset_u``, or ``offset_rotation_deg``
        is non-zero, so the navigated frame differs from an offset-free preview.
    """
    return any(float(sim_params.get(key, 0.0)) != 0.0 for key in PLANTED_POINTING_KEYS)


def frame_units_summary(image: NDArrayFloatType, inst_config: CameraBlock) -> str:
    """Describe a frame's units, value range, marker share, and the navigator's verdict.

    The text gives the frame's units and its finite value range, for example
    ``'I/F 0 to 1.48'``.  The units are ``DN`` for a ``raw_dn`` block and ``I/F`` for
    a ``calibrated_if`` one.  When any pixel equals the navigator's marker (see
    :func:`~spindoctor.support.image_class.missing_data_marker`; NaN pixels when the
    marker is NaN), it adds the marker and the share of pixels carrying it.  When
    the navigator's classifier
    (see :func:`read_frame_as_navigator`) would not call the frame clean, it adds
    the reason, in the classifier's order: below the blank limit, over the
    overexposed share, or over ``max_missing_frac_clean`` outside the unread lines,
    in which case the navigator refuses it as missing data.  A line is unread when
    every one of its pixels carries the marker.  Pass the frame the navigator reads, with
    the planted offset applied, for the verdict to match it.

    Parameters:
        image: The rendered frame.
        inst_config: The navigator block the frame was rendered with.

    Returns:
        One line of text for a status display.
    """
    arr = np.asarray(image, dtype=np.float64)
    reading = read_frame_as_navigator(arr, inst_config)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        text = f'{reading.units}: no finite pixels'
    else:
        text = f'{reading.units} {float(finite.min()):.4g} to {float(finite.max()):.4g}'
    if reading.missing_frac > 0.0:
        marker_text = 'NaN' if math.isnan(reading.marker) else f'{reading.marker:g}'
        text += f'; marker {marker_text} on {reading.missing_frac * 100:.1f}% of pixels'
    if reading.image_class == 'blank':
        text += f' (peak below {reading.blank_max:g}: the navigator classifies it blank)'
    elif reading.image_class == 'fully_overexposed':
        text += (
            f' (over {reading.max_overexposed_frac * 100:.0f}% saturated: '
            'the navigator classifies it overexposed)'
        )
    elif reading.image_class == 'mostly_missing_data':
        text += (
            f' (over {reading.max_missing_frac * 100:.0f}%: '
            'the navigator refuses it as missing data)'
        )
    return text
