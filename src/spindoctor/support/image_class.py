"""The image-quality decision: an instrument's thresholds in, an image class out.

Before it navigates an image, the navigator sorts it into one of four classes
and refuses the first three:

- ``blank``: the brightest sensor pixel is below the blank limit;
- ``fully_overexposed``: more than the allowed share of sensor pixels is at or
  above the saturation threshold;
- ``mostly_missing_data``: more than the allowed share of sensor pixels carries
  the missing-data marker;
- ``clean``: anything else.

The checks run in that order, so a dark frame whose marker is ``0`` is blank
rather than missing data.  This module holds that decision and the reading of
its thresholds from a camera block
(:class:`~spindoctor.config.sections_camera.CameraBlock`), so the navigator's image
classifier and the simulated-image editor's verdict make it the same way.  It
imports neither the navigator packages nor the simulator.

A ``raw_dn`` block quotes its limits in DN.  A ``calibrated_if`` block quotes
them in I/F and has no saturation check: the same full-well count maps to a
different I/F for every exposure, filter, and gain, so the saturation threshold
is infinite and no finite pixel counts as saturated.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from spindoctor.config.sections_camera import CameraBlock, as_camera_block
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ImageQualityClass',
    'ImageQualityStatistics',
    'ImageQualityThresholds',
    'decide_image_class',
    'image_quality_thresholds',
    'measure_image_quality',
    'missing_data_marker',
    'missing_data_mask',
]


ImageQualityClass = Literal['clean', 'blank', 'fully_overexposed', 'mostly_missing_data']
"""The classes :func:`decide_image_class` can return."""

_CAMERA_SOURCE = 'the camera block'
"""The label a message gives for a problem in a camera block held as a mapping."""


@dataclass(frozen=True)
class ImageQualityThresholds:
    """Per-instrument configuration for the image-quality classifier.

    Parameters:
        saturation_threshold_dn: Pixels at or above this DN are flagged
            saturated.
        missing_data_marker_dn: Pixels exactly equal to this value are
            treated as missing data (instrument-specific dropout marker).
        max_saturation_frac_clean: Above this fraction of saturated pixels
            the image is ``fully_overexposed``.
        max_missing_frac_clean: Above this fraction of missing pixels the
            image is ``mostly_missing_data``.
        partial_dropout_min_frac: Above this missing fraction (and at or
            below ``max_missing_frac_clean``) the ``partial_dropout`` advisory
            flag is set on the result.
        blank_max_dn: If the image's max DN is below this, the image is
            ``blank``.
        noisy_threshold: Above this MAD-noise sigma, the ``noisy`` flag is
            raised (image stays ``clean``).
    """

    saturation_threshold_dn: float
    missing_data_marker_dn: float
    max_saturation_frac_clean: float
    max_missing_frac_clean: float
    partial_dropout_min_frac: float
    blank_max_dn: float
    noisy_threshold: float


@dataclass(frozen=True)
class ImageQualityStatistics:
    """The three sensor-pixel measurements the decision reads.

    Parameters:
        max_dn: The brightest finite sensor pixel, or ``0.0`` when none is finite.
        saturation_frac: Fraction of sensor pixels at or above the saturation
            threshold.
        missing_frac: Fraction of sensor pixels carrying the missing-data marker.
    """

    max_dn: float
    saturation_frac: float
    missing_frac: float


def missing_data_marker(camera: CameraBlock | Mapping[str, Any]) -> float:
    """Return the missing-data marker of a camera block.

    Parameters:
        camera: The camera block.  A block held as a mapping is parsed first (see
            :func:`~spindoctor.config.sections_camera.as_camera_block`).

    Returns:
        The block's ``noise.marker_value``, possibly NaN.

    Raises:
        ~spindoctor.schema.SchemaError: If a mapping breaks the camera block's schema.
    """
    return as_camera_block(camera, source=_CAMERA_SOURCE).noise.marker()


def image_quality_thresholds(camera: CameraBlock | Mapping[str, Any]) -> ImageQualityThresholds:
    """Read the classifier's thresholds from a camera block.

    A ``raw_dn`` block gives its blank, noise, and saturation limits in DN.  A
    ``calibrated_if`` block gives its blank and noise limits in I/F, and its saturation
    threshold is infinite.  The marker is the block's ``noise.marker_value``, and the
    other limits are the block's own.

    Parameters:
        camera: The camera block.  A block held as a mapping is parsed first (see
            :func:`~spindoctor.config.sections_camera.as_camera_block`).

    Returns:
        The thresholds, in DN for a ``raw_dn`` block and in I/F for a
        ``calibrated_if`` one.

    Raises:
        ~spindoctor.schema.SchemaError: If a mapping breaks the camera block's schema.
    """
    block = as_camera_block(camera, source=_CAMERA_SOURCE)
    limits = block.image_quality_thresholds
    # The camera block sets every limit its units read, so none of these is None.
    if block.data_units == 'raw_dn':
        blank = limits.blank_max_dn
        noisy = limits.noisy_threshold_dn
        saturation = limits.saturation_threshold_dn
    else:
        blank = limits.blank_max_if
        noisy = limits.noisy_threshold_if
        saturation = math.inf
    return ImageQualityThresholds(
        saturation_threshold_dn=cast(float, saturation),
        missing_data_marker_dn=block.noise.marker(),
        max_saturation_frac_clean=limits.max_overexposed_frac_clean,
        max_missing_frac_clean=limits.max_missing_frac_clean,
        partial_dropout_min_frac=limits.partial_dropout_min_frac,
        blank_max_dn=cast(float, blank),
        noisy_threshold=cast(float, noisy),
    )


def missing_data_mask(image: NDArrayFloatType, marker: float) -> NDArrayBoolType:
    """Return the pixels carrying the missing-data marker.

    A NaN marker selects the NaN pixels, since NaN never equals itself.

    Parameters:
        image: The pixel array.
        marker: The missing-data marker, possibly NaN.

    Returns:
        A boolean array of the same shape as ``image``.
    """
    if math.isnan(marker):
        return np.isnan(image)
    return np.asarray(image == marker)


def measure_image_quality(
    sensor: NDArrayFloatType,
    thresholds: ImageQualityThresholds,
    *,
    missing_frac: float | None = None,
) -> ImageQualityStatistics:
    """Measure the brightest pixel and the saturated and missing shares of the sensor.

    Parameters:
        sensor: The sensor pixels, in any shape (extended-FOV padding excluded).
        thresholds: The instrument's thresholds.
        missing_frac: The missing-data share when the caller already knows it, for
            example because it replaced the marked pixels before this call.  When
            ``None`` the share is counted from ``thresholds.missing_data_marker_dn``.

    Returns:
        The measurements.
    """
    n_total = max(sensor.size, 1)
    saturation_frac = float((sensor >= thresholds.saturation_threshold_dn).sum()) / n_total
    if missing_frac is None:
        missing = missing_data_mask(sensor, thresholds.missing_data_marker_dn)
        missing_frac = float(missing.sum()) / n_total
    # ``np.nanmax`` ignores NaN markers that survive in the sensor pixels; with no
    # finite pixel the peak is 0.0, so the frame is blank.
    if sensor.size > 0 and np.isfinite(sensor).any():
        max_dn = float(np.nanmax(sensor))
    else:
        max_dn = 0.0
    return ImageQualityStatistics(
        max_dn=max_dn, saturation_frac=saturation_frac, missing_frac=missing_frac
    )


def decide_image_class(
    statistics: ImageQualityStatistics, thresholds: ImageQualityThresholds
) -> ImageQualityClass:
    """Return the image class the measurements and thresholds decide.

    The checks run in a fixed order: below the blank limit is ``blank``; else more
    than the allowed saturated share is ``fully_overexposed``; else more than the
    allowed missing share is ``mostly_missing_data``; else ``clean``.  A share
    exactly at its limit is allowed.

    Parameters:
        statistics: The sensor measurements (see :func:`measure_image_quality`).
        thresholds: The instrument's thresholds.

    Returns:
        The image class.
    """
    if statistics.max_dn < thresholds.blank_max_dn:
        return 'blank'
    if statistics.saturation_frac > thresholds.max_saturation_frac_clean:
        return 'fully_overexposed'
    if statistics.missing_frac > thresholds.max_missing_frac_clean:
        return 'mostly_missing_data'
    return 'clean'
