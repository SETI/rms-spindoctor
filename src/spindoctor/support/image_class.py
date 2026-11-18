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
its thresholds from a per-instrument config block, so the navigator's image
classifier and the simulated-image editor's verdict make it the same way.  It
imports neither the navigator packages nor the simulator.

A ``raw_dn`` block quotes its limits in DN.  A ``calibrated_if`` block quotes
them in I/F and has no saturation check: the same full-well count maps to a
different I/F for every exposure, filter, and gain, so the saturation threshold
is infinite and such a frame is never ``fully_overexposed``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

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
        max_missing_frac_clean: Above this fraction of missing pixels the
            image is ``mostly_missing_data``.
        partial_dropout_min_frac: Below this missing fraction, no
            ``partial_dropout`` flag is raised.  At or above this fraction
            (and below ``max_missing_frac_clean``) the ``partial_dropout``
            advisory flag is set on the result.
        blank_max_dn: If the image's max DN is below this, the image is
            ``blank``.
        noisy_threshold: Above this MAD-noise sigma, the ``noisy`` flag is
            raised (image stays ``clean``).
    """

    saturation_threshold_dn: float = 4095.0
    missing_data_marker_dn: float = 0.0
    max_saturation_frac_clean: float = 0.80
    max_missing_frac_clean: float = 0.30
    partial_dropout_min_frac: float = 0.05
    blank_max_dn: float = 5.0
    noisy_threshold: float = 10.0


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


def _coerce_marker(value: Any) -> float:
    """Read a ``noise.marker_value`` entry as a float, ``NaN`` permitted."""
    if value is None:
        return math.nan
    if isinstance(value, str):
        if value.strip().lower() == 'nan':
            return math.nan
        raise ValueError(f'noise.marker_value string {value!r} must be "NaN" or numeric')
    return float(value)


def _required_float(block: Mapping[str, Any], key: str, *, location: str) -> float:
    """Read a required finite numeric field from a config mapping.

    Raises ``ValueError`` with a path-aware message when the key is missing,
    non-numeric, or non-finite (NaN or infinite), because the value becomes a
    threshold the classifier compares against pixel data.
    """
    if key not in block:
        raise ValueError(f'{location} missing required field {key!r}')
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{location}.{key} must be numeric; got {type(value).__name__}={value!r}')
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f'{location}.{key} must be a finite numeric value; got {value!r}')
    return coerced


def _data_units(inst_config: Mapping[str, Any]) -> str:
    """Return a block's ``data_units``, refusing anything but the two known kinds."""
    data_units = inst_config.get('data_units')
    if not isinstance(data_units, str) or data_units not in ('raw_dn', 'calibrated_if'):
        raise ValueError(
            'instrument config missing required field data_units '
            f"(must be 'raw_dn' or 'calibrated_if'); got {data_units!r}"
        )
    return data_units


def _noise_block(inst_config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return a block's ``noise`` mapping, or ``None`` when it has none."""
    noise = inst_config.get('noise')
    if noise is not None and not isinstance(noise, Mapping):
        raise TypeError(
            f'instrument config noise block must be a mapping; got {type(noise).__name__}'
        )
    return noise


def missing_data_marker(inst_config: Mapping[str, Any]) -> float:
    """Return the missing-data marker of a per-instrument config block.

    A number in ``noise.marker_value`` is taken as is, and the string ``'NaN'``
    (any case) or ``None`` is NaN.  An absent key, or an absent ``noise`` block,
    means ``0`` for a ``raw_dn`` block and NaN for a ``calibrated_if`` one.

    Parameters:
        inst_config: A per-instrument config block.

    Returns:
        The marker value, possibly NaN.

    Raises:
        ValueError: If ``data_units`` is not ``raw_dn`` or ``calibrated_if``, or
            ``noise.marker_value`` is a string other than ``'NaN'``.
        TypeError: If the ``noise`` entry is not a mapping.
    """
    raw_dn = _data_units(inst_config) == 'raw_dn'
    noise = _noise_block(inst_config) or {}
    return _coerce_marker(noise.get('marker_value', 0 if raw_dn else None))


def image_quality_thresholds(inst_config: Mapping[str, Any]) -> ImageQualityThresholds:
    """Read the classifier's thresholds from a per-instrument config block.

    A ``raw_dn`` block must set ``image_quality_thresholds.saturation_threshold_dn``,
    ``blank_max_dn``, and ``noisy_threshold_dn``.  A ``calibrated_if`` block must set
    ``blank_max_if`` and ``noisy_threshold_if`` and must not set
    ``saturation_threshold_if``; its saturation threshold is infinite.  Either kind
    may set ``max_overexposed_frac_clean`` (default 0.80), ``max_missing_frac_clean``
    (default 0.30), and ``partial_dropout_min_frac`` (default 0.05).  The marker comes
    from :func:`missing_data_marker`.

    Parameters:
        inst_config: A per-instrument config block.

    Returns:
        The thresholds, in DN for a ``raw_dn`` block and in I/F for a
        ``calibrated_if`` one.

    Raises:
        ValueError: If ``data_units`` is not recognized, the
            ``image_quality_thresholds`` block is missing, a required field is
            missing or not a finite number, or a ``calibrated_if`` block sets
            ``saturation_threshold_if``.
        TypeError: If the ``noise`` entry is not a mapping.
    """
    data_units = _data_units(inst_config)
    iqt_block = inst_config.get('image_quality_thresholds')
    if not isinstance(iqt_block, Mapping):
        raise ValueError(
            'instrument config missing required image_quality_thresholds block '
            f'(or block is not a mapping); got {type(iqt_block).__name__}'
        )
    marker = missing_data_marker(inst_config)
    location = 'image_quality_thresholds'
    if data_units == 'raw_dn':
        saturation = _required_float(iqt_block, 'saturation_threshold_dn', location=location)
        suffix = 'dn'
    else:
        if 'saturation_threshold_if' in iqt_block:
            raise ValueError(
                'calibrated_if instrument must not declare '
                'image_quality_thresholds.saturation_threshold_if; the '
                'saturation gate is off for calibrated images because I/F '
                'depends on exposure / filter / gain (see Phase 10 §F)'
            )
        saturation = math.inf
        suffix = 'if'
    return ImageQualityThresholds(
        saturation_threshold_dn=saturation,
        missing_data_marker_dn=marker,
        max_saturation_frac_clean=float(iqt_block.get('max_overexposed_frac_clean', 0.80)),
        max_missing_frac_clean=float(iqt_block.get('max_missing_frac_clean', 0.30)),
        partial_dropout_min_frac=float(iqt_block.get('partial_dropout_min_frac', 0.05)),
        blank_max_dn=_required_float(iqt_block, f'blank_max_{suffix}', location=location),
        noisy_threshold=_required_float(iqt_block, f'noisy_threshold_{suffix}', location=location),
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
