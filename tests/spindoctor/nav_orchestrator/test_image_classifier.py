"""Tests for ``spindoctor.nav_orchestrator.image_classifier.NavImageClassifier``."""

import numpy as np
import pytest

from spindoctor.nav_orchestrator.image_classifier import (
    ImageQualityThresholds,
    NavImageClassifier,
)


def test_classifier_clean_image_no_flags() -> None:
    """A normal image returns class 'clean' with empty flags."""
    rng = np.random.default_rng(seed=1)
    image = rng.standard_normal(size=(64, 64)) + 100.0
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert result.flags == []


def test_classifier_reports_background_gradient_on_ramp() -> None:
    """A large-scale brightness ramp populates a high background-gradient score."""
    rng = np.random.default_rng(seed=7)
    _yy, xx = np.mgrid[0:128, 0:128]
    image = 100.0 + 3.0 * xx + rng.normal(0.0, 2.0, size=(128, 128))
    result = NavImageClassifier().classify(image)
    assert result.background_gradient_score is not None
    assert result.background_gradient_score > 5.0


def test_classifier_background_gradient_low_on_flat_noise() -> None:
    """A flat noisy field scores well below the scattered-light threshold."""
    rng = np.random.default_rng(seed=8)
    image = 100.0 + rng.normal(0.0, 3.0, size=(128, 128))
    result = NavImageClassifier().classify(image)
    assert result.background_gradient_score is not None
    assert result.background_gradient_score < 5.0


def test_classifier_blank_image() -> None:
    """A near-zero image is classified as blank."""
    image = np.zeros((64, 64), np.float64)
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'blank'


def test_classifier_fully_overexposed() -> None:
    """An image saturated above the threshold yields fully_overexposed."""
    image = np.full((64, 64), 4095.0, np.float64)
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'fully_overexposed'
    assert result.saturation_frac == 1.0


def test_classifier_inf_saturation_threshold_disables_gate() -> None:
    """An ``inf`` saturation threshold disables the saturation gate.

    This is the calibrated_if path (Phase 10 §F): a single I/F
    saturation threshold is meaningless because calibrated values
    depend on exposure / filter / gain, so the loader installs an
    ``inf`` threshold and the classifier must report
    ``saturation_frac=0.0`` no matter what pixel values it sees.
    """
    image = np.full((64, 64), 4095.0, np.float64)
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            saturation_threshold_dn=float('inf'),
            blank_max_dn=1.0e-4,
            noisy_threshold=10.0,
        )
    )
    result = classifier.classify(image)
    assert result.saturation_frac == 0.0
    assert result.image_class != 'fully_overexposed'


def test_classifier_mostly_missing_data() -> None:
    """An image dominated by the missing-data marker yields mostly_missing_data."""
    image = np.full((64, 64), 0.0, np.float64)
    image[:48, :] = 100.0  # Some non-zero data, but most is zero (missing marker).
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            missing_data_marker_dn=0.0,
            max_missing_frac_clean=0.10,
        )
    )
    result = classifier.classify(image)
    assert result.image_class == 'mostly_missing_data'


def test_classifier_nan_marker_detects_missing_data() -> None:
    """A NaN missing-data marker (calibrated_if) is detected via np.isnan.

    For calibrated-IF instruments the missing-data sentinel is literally
    NaN.  ``sensor == NaN`` can never match (NaN != NaN), so the
    classifier must fall back to ``np.isnan`` when the marker is NaN.
    Here most of the image is NaN, so ``missing_frac`` must reflect that
    fraction and the image classifies as ``mostly_missing_data``.
    """
    image = np.full((64, 64), 0.5, np.float64)
    image[:48, :] = np.nan  # 75% NaN (missing) data
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            saturation_threshold_dn=float('inf'),
            missing_data_marker_dn=float('nan'),
            max_missing_frac_clean=0.30,
            blank_max_dn=1.0e-4,
        )
    )
    result = classifier.classify(image)
    assert result.missing_frac == pytest.approx(0.75)
    assert result.image_class == 'mostly_missing_data'


def test_classifier_nan_marker_max_dn_uses_nanmax() -> None:
    """max_dn ignores NaN markers so the blank short-circuit is not poisoned.

    ``np.max`` over a NaN-containing array returns NaN, which would make
    every comparison against the blank/saturation thresholds False and
    silently mis-classify.  ``np.nanmax`` returns the largest finite
    value instead.
    """
    image = np.full((64, 64), np.nan, np.float64)
    image[0, 0] = 123.0  # one finite pixel
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            saturation_threshold_dn=float('inf'),
            missing_data_marker_dn=float('nan'),
            max_missing_frac_clean=0.99,
            blank_max_dn=1.0e-4,
        )
    )
    result = classifier.classify(image)
    assert result.max_dn == pytest.approx(123.0)


def test_classifier_explicit_missing_frac_overrides_internal() -> None:
    """A caller-supplied missing_frac drives the verdict instead of the marker count.

    The orchestrator computes the true missing fraction from the raw
    image before NaN sanitization and threads it in; the classifier must
    use that value rather than recomputing from the (already-filled)
    sensor pixels.
    """
    image = np.full((64, 64), 100.0, np.float64)  # no NaN, no marker hits
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            missing_data_marker_dn=0.0,
            max_missing_frac_clean=0.30,
        )
    )
    result = classifier.classify(image, missing_frac=0.5)
    assert result.missing_frac == pytest.approx(0.5)
    assert result.image_class == 'mostly_missing_data'


def test_classifier_partial_dropout_flag() -> None:
    """A small fraction of missing data raises the partial_dropout flag."""
    rng = np.random.default_rng(seed=2)
    image = rng.standard_normal(size=(64, 64)) + 100.0
    image[:5, :] = 0.0  # ~7.8% missing
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert 'partial_dropout' in result.flags


def test_classifier_noisy_flag() -> None:
    """High image noise raises the noisy flag."""
    rng = np.random.default_rng(seed=3)
    image = rng.standard_normal(size=(64, 64)) * 50.0 + 100.0
    classifier = NavImageClassifier(thresholds=ImageQualityThresholds(noisy_threshold=10.0))
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert 'noisy' in result.flags


def test_classifier_rejects_non_2d() -> None:
    """3-D input raises TypeError."""
    image = np.zeros((4, 4, 4), np.float64)
    classifier = NavImageClassifier()
    with pytest.raises(TypeError, match='2-D'):
        classifier.classify(image)


def test_classifier_uses_sensor_mask() -> None:
    """Pixels outside the sensor mask are excluded from the classification."""
    image = np.zeros((64, 64), np.float64)
    image[:, :32] = 100.0
    image[:, 32:] = 4095.0  # extfov padding has bogus saturation values
    mask = np.zeros((64, 64), bool)
    mask[:, :32] = True  # sensor is the left half
    classifier = NavImageClassifier()
    result = classifier.classify(image, sensor_mask=mask)
    # Saturation fraction is computed on the sensor area only (zero saturated).
    assert result.saturation_frac == 0.0
    assert result.image_class == 'clean'
