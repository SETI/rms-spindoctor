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
testable without loading config.  The thresholds type, the measurements, and
the decision itself live in :mod:`spindoctor.support.image_class`, which the
simulated-image editor's verdict shares.
"""

from dataclasses import dataclass, field

import numpy as np

from spindoctor.nav_orchestrator.image_classifier_result import (
    ImageFlag,
    NavImageClassifierResult,
)
from spindoctor.support.background_gradient import background_gradient_score
from spindoctor.support.image_class import (
    ImageQualityThresholds,
    decide_image_class,
    measure_image_quality,
)
from spindoctor.support.noise_estimate import estimate_image_noise_sigma
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'NavImageClassifier',
]


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
        # The sensor statistics and the class come from the shared decision in
        # spindoctor.support.image_class, which the simulated-image editor's verdict
        # also uses.  When the orchestrator supplies ``missing_frac`` (from the true
        # missing mask, before the NaN marker was filled), it is trusted.
        statistics = measure_image_quality(sensor, self.thresholds, missing_frac=missing_frac)
        noise_sigma = estimate_image_noise_sigma(image, sensor_mask)
        # Low-order brightness-ramp score over the 2-D frame (the block-median
        # affine fit needs spatial structure, so it runs on the image array
        # rather than the flattened sensor selection).  The sensor mask keeps
        # extfov padding out of the plane fit -- otherwise the zero-filled
        # border biases the score and can suppress it below the ensemble's
        # scattered-light threshold on full-frame images.  Recorded on every
        # verdict; the ensemble reads it when resolving disc/limb correlations.
        background_gradient = background_gradient_score(image, sensor_mask)
        image_class = decide_image_class(statistics, self.thresholds)
        # A clean image may carry advisory flags.
        flags: list[ImageFlag] = []
        if image_class == 'clean':
            if statistics.missing_frac > self.thresholds.partial_dropout_min_frac:
                flags.append('partial_dropout')
            if noise_sigma > self.thresholds.noisy_threshold:
                flags.append('noisy')
        return NavImageClassifierResult(
            image_class=image_class,
            saturation_frac=statistics.saturation_frac,
            missing_frac=statistics.missing_frac,
            noise_sigma=noise_sigma,
            max_dn=statistics.max_dn,
            background_gradient_score=background_gradient,
            flags=flags,
        )
