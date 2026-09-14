"""NavImageClassifier — quick-fail classifier for incoming images.

Operates on the entire sensor area and assigns an image to one of a small
set of classes.  Most "bad" classes never invoke an extractor —
corrupted images fail in milliseconds with a clear reason.

The classifier is global: no predicted feature positions are used.  Three
cheap statistics drive the decision:

    saturation_frac = fraction of pixels at saturation_threshold_dn or above
    missing_frac    = fraction of pixels equal to missing_data_marker_dn
    noise_sigma     = MAD-based image noise sigma

Per-instrument thresholds live in ``config_4N0_inst_*.yaml``; this module
takes them as constructor parameters so it stays pure-Python and unit-
testable without loading config.
"""

from dataclasses import dataclass, field

import numpy as np

from spindoctor.nav_orchestrator.image_classifier_result import (
    ImageFlag,
    NavImageClassifierResult,
)
from spindoctor.support.background_gradient import background_gradient_score
from spindoctor.support.noise_estimate import estimate_image_noise_sigma
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ImageQualityThresholds',
    'NavImageClassifier',
]


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


@dataclass
class NavImageClassifier:
    """Quick-fail image classifier consumed by the orchestrator.

    Parameters:
        thresholds: Per-instrument thresholds (see ``ImageQualityThresholds``).
    """

    thresholds: ImageQualityThresholds = field(default_factory=ImageQualityThresholds)

    def classify(
        self,
        image: NDArrayFloatType,
        sensor_mask: NDArrayBoolType | None = None,
        *,
        missing_frac: float | None = None,
    ) -> NavImageClassifierResult:
        """Run the classifier and return its verdict.

        Parameters:
            image: 2-D float image array (sensor + extfov padding).
            sensor_mask: Optional boolean mask selecting sensor pixels;
                if ``None``, every pixel is treated as sensor data.
            missing_frac: Optional pre-computed missing-data fraction over
                the sensor pixels.  The orchestrator supplies this from the
                true missing mask (which handles the calibrated-IF
                ``NaN`` sentinel before the array is sanitized for the
                finite-only derivative path); when ``None`` the classifier
                derives the fraction itself from ``missing_data_marker_dn``
                (using ``np.isnan`` when the marker is itself ``NaN``).

        Returns:
            NavImageClassifierResult.

        Raises:
            TypeError: if ``image`` is not 2-D.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f'NavImageClassifier requires a numpy.ndarray; got {type(image).__name__}'
            )
        if image.ndim != 2:
            raise TypeError(f'NavImageClassifier requires a 2-D image; got ndim={image.ndim}')
        if sensor_mask is None:
            sensor = image
        else:
            if not isinstance(sensor_mask, np.ndarray):
                raise TypeError(
                    f'sensor_mask must be a numpy ndarray; got {type(sensor_mask).__name__}'
                )
            if sensor_mask.dtype != np.bool_:
                raise TypeError(f'sensor_mask must have boolean dtype; got {sensor_mask.dtype}')
            if sensor_mask.shape != image.shape:
                raise ValueError(
                    f'sensor_mask shape {sensor_mask.shape} differs from image shape {image.shape}'
                )
            if sensor_mask.size == 0:
                raise ValueError('sensor_mask must not be empty')
            if not sensor_mask.any():
                raise ValueError('sensor_mask must select at least one sensor pixel')
            sensor = image[sensor_mask]
        # Compute statistics on the sensor pixels only.
        sat_mask = sensor >= self.thresholds.saturation_threshold_dn
        n_total = max(sensor.size, 1)
        saturation_frac = float(sat_mask.sum()) / float(n_total)
        # The missing-data marker is NaN for calibrated-IF instruments, so
        # ``sensor == marker`` can never match (NaN != NaN).  Detect the NaN
        # marker explicitly via ``np.isnan`` so missing/dropout detection is
        # not silently dead for calibrated images.  When the orchestrator
        # supplies a pre-computed ``missing_frac`` (from the true missing
        # mask before NaN sanitization), trust it.
        if missing_frac is None:
            marker = self.thresholds.missing_data_marker_dn
            if np.isnan(marker):
                miss_mask = np.isnan(sensor)
            else:
                miss_mask = sensor == marker
            missing_frac = float(miss_mask.sum()) / float(n_total)
        noise_sigma = estimate_image_noise_sigma(image, sensor_mask)
        # ``np.nanmax`` ignores any NaN missing-data markers that survive in
        # the sensor pixels; an all-NaN (or empty) sensor falls back to 0.0
        # so the blank short-circuit and noise estimate are not poisoned.
        if sensor.size > 0 and np.isfinite(sensor).any():
            max_dn = float(np.nanmax(sensor))
        else:
            max_dn = 0.0
        # Low-order brightness-ramp score over the 2-D frame (the block-median
        # affine fit needs spatial structure, so it runs on the image array
        # rather than the flattened sensor selection).  The sensor mask keeps
        # extfov padding out of the plane fit -- otherwise the zero-filled
        # border biases the score and can suppress it below the ensemble's
        # scattered-light threshold on full-frame images.  Recorded on every
        # verdict; the ensemble reads it when resolving disc/limb correlations.
        background_gradient = background_gradient_score(image, sensor_mask)
        flags: list[ImageFlag] = []
        # Outcome decision: blank check runs first so a near-zero image with
        # missing-data marker == 0 isn't mis-classified as "mostly_missing".
        if max_dn < self.thresholds.blank_max_dn:
            return NavImageClassifierResult(
                image_class='blank',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                background_gradient_score=background_gradient,
                flags=flags,
            )
        if saturation_frac > self.thresholds.max_saturation_frac_clean:
            return NavImageClassifierResult(
                image_class='fully_overexposed',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                background_gradient_score=background_gradient,
                flags=flags,
            )
        if missing_frac > self.thresholds.max_missing_frac_clean:
            return NavImageClassifierResult(
                image_class='mostly_missing_data',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                background_gradient_score=background_gradient,
                flags=flags,
            )
        # Otherwise the image is clean (with optional advisory flags).
        if missing_frac > self.thresholds.partial_dropout_min_frac:
            flags.append('partial_dropout')
        if noise_sigma > self.thresholds.noisy_threshold:
            flags.append('noisy')
        return NavImageClassifierResult(
            image_class='clean',
            saturation_frac=saturation_frac,
            missing_frac=missing_frac,
            noise_sigma=noise_sigma,
            max_dn=max_dn,
            background_gradient_score=background_gradient,
            flags=flags,
        )
