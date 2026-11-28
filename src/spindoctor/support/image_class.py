"""The image-quality decision: an instrument's thresholds in, an image class out.

Before it navigates an image, the navigator sorts it into one of four classes
and refuses the first three:

- ``blank``: the brightest sensor pixel is below the blank limit;
- ``fully_overexposed``: more than the allowed share of sensor pixels is at or
  above the saturation threshold;
- ``mostly_missing_data``: outside the unread lines, more than the allowed share
  of sensor pixels carries the missing-data marker, or fewer than the allowed share
  of the sensor's lines was read out;
- ``clean``: anything else.

A line is a row of the image, and it is unread when every one of its sensor pixels
carries the marker, as when a readout stops early or the lines are lost on the way to
the ground.  The lines that were read out hold good data, so the unread lines are
judged by the share of lines read out, and only the other missing pixels count
against the missing-pixel limit.

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
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from spindoctor.config.sections_camera import CameraBlock
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
        max_missing_frac_clean: Above this fraction of missing pixels outside
            the unread lines the image is ``mostly_missing_data``.
        partial_dropout_min_frac: Above this fraction of missing pixels, unread
            lines included, a clean image carries the ``partial_dropout`` advisory
            flag.
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
    """The sensor-pixel measurements the decision reads.

    Parameters:
        max_dn: The brightest finite sensor pixel that was read, or ``0.0`` when none
            is.
        saturation_frac: Fraction of sensor pixels at or above the saturation
            threshold.
        missing_frac: Fraction of sensor pixels carrying the missing-data marker.
        unread_line_frac: Fraction of the sensor's lines on which every sensor pixel
            carries the marker.
        scattered_missing_frac: Fraction of sensor pixels that carry the marker and
            lie outside the unread lines.
    """

    max_dn: float
    saturation_frac: float
    missing_frac: float
    unread_line_frac: float
    scattered_missing_frac: float


def missing_data_marker(camera: CameraBlock) -> float:
    """Return the missing-data marker of a camera block.

    Parameters:
        camera: The camera block.

    Returns:
        The block's ``noise.marker_value``, possibly NaN.
    """
    return camera.noise.marker()


def image_quality_thresholds(camera: CameraBlock) -> ImageQualityThresholds:
    """Read the classifier's thresholds from a camera block.

    A ``raw_dn`` block gives its blank, noise, and saturation limits in DN.  A
    ``calibrated_if`` block gives its blank and noise limits in I/F, and its saturation
    threshold is infinite.  The marker is the block's ``noise.marker_value``, and the
    other limits are the block's own.

    Parameters:
        camera: The camera block.

    Returns:
        The thresholds, in DN for a ``raw_dn`` block and in I/F for a
        ``calibrated_if`` one.
    """
    limits = camera.image_quality_thresholds
    # The camera block sets every limit its units read, so none of these is None.
    if camera.data_units == 'raw_dn':
        blank = limits.blank_max_dn
        noisy = limits.noisy_threshold_dn
        saturation = limits.saturation_threshold_dn
    else:
        blank = limits.blank_max_if
        noisy = limits.noisy_threshold_if
        saturation = math.inf
    return ImageQualityThresholds(
        saturation_threshold_dn=cast(float, saturation),
        missing_data_marker_dn=camera.noise.marker(),
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
    image: NDArrayFloatType,
    thresholds: ImageQualityThresholds,
    *,
    sensor_mask: NDArrayBoolType | None = None,
    missing_mask: NDArrayBoolType | None = None,
) -> ImageQualityStatistics:
    """Measure the peak, the saturated and missing shares, and the unread lines.

    A line is a row of ``image``.  It counts as one of the sensor's lines when it has at
    least one sensor pixel, and as unread when every one of its sensor pixels is
    missing.

    Parameters:
        image: The 2-D pixel array.
        thresholds: The instrument's thresholds.
        sensor_mask: The sensor pixels, of the same shape as ``image``, which leaves the
            extended-FOV padding out.  When ``None`` every pixel is a sensor pixel.
        missing_mask: The missing pixels, of the same shape as ``image``, when the
            caller already knows them, for example because it replaced the marked
            pixels before this call.  When ``None`` they are the pixels carrying
            ``thresholds.missing_data_marker_dn``.  Only the sensor pixels of the mask
            are read.

    Returns:
        The measurements.
    """
    if sensor_mask is None:
        sensor_mask = np.ones(image.shape, dtype=bool)
    if missing_mask is None:
        missing_mask = missing_data_mask(image, thresholds.missing_data_marker_dn)
    sensor = image[sensor_mask]
    sensor_missing = missing_mask & sensor_mask
    n_total = max(sensor.size, 1)
    saturation_frac = float((sensor >= thresholds.saturation_threshold_dn).sum()) / n_total
    missing_frac = float(sensor_missing.sum()) / n_total
    sensor_lines = sensor_mask.any(axis=1)
    unread_lines = sensor_lines & (sensor_missing == sensor_mask).all(axis=1)
    n_lines = max(int(sensor_lines.sum()), 1)
    n_unread = int(unread_lines.sum())
    scattered = float(sensor_missing[~unread_lines].sum()) / n_total
    # The peak is taken over the pixels that were read; with no finite one it is 0.0,
    # so a frame with no line read out is blank whatever its marker.
    read = image[sensor_mask & ~missing_mask]
    finite = read[np.isfinite(read)]
    max_dn = float(finite.max()) if finite.size > 0 else 0.0
    return ImageQualityStatistics(
        max_dn=max_dn,
        saturation_frac=saturation_frac,
        missing_frac=missing_frac,
        unread_line_frac=n_unread / n_lines,
        scattered_missing_frac=scattered,
    )


def decide_image_class(
    statistics: ImageQualityStatistics, thresholds: ImageQualityThresholds
) -> ImageQualityClass:
    """Return the image class the measurements and thresholds decide.

    The checks run in a fixed order: below the blank limit is ``blank``; else more
    than the allowed saturated share is ``fully_overexposed``; else more than the
    allowed share of missing pixels outside the unread lines is
    ``mostly_missing_data``; else ``clean``.  A share exactly at its limit is allowed.
    Unread lines never refuse an image by themselves: the techniques judge whether the
    lines that were read hold enough to navigate on.  An image with no line read out
    has a peak of 0.0 and is ``blank``.

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
    if statistics.scattered_missing_frac > thresholds.max_missing_frac_clean:
        return 'mostly_missing_data'
    return 'clean'
