"""Tests for the ``spindoctor.support.noise_estimate`` estimators."""

import numpy as np
import pytest

from spindoctor.support.noise_estimate import (
    estimate_background_and_sky_sigma,
    estimate_image_noise_sigma,
)


def test_estimate_image_noise_sigma_unit_normal() -> None:
    """MAD-based sigma on N(0, 1) data is close to 1."""
    rng = np.random.default_rng(seed=42)
    image = rng.standard_normal(size=(256, 256))
    sigma = estimate_image_noise_sigma(image)
    assert abs(sigma - 1.0) < 0.05


def test_estimate_image_noise_sigma_robust_to_bright_features() -> None:
    """A few bright outliers do not change the estimate appreciably."""
    rng = np.random.default_rng(seed=43)
    image = rng.standard_normal(size=(256, 256))
    image[100:110, 100:110] = 10000.0  # outlier block
    sigma = estimate_image_noise_sigma(image)
    assert abs(sigma - 1.0) < 0.1


def test_estimate_image_noise_sigma_uses_sensor_mask() -> None:
    """Pixels outside the sensor mask are excluded from the estimate."""
    rng = np.random.default_rng(seed=44)
    image = np.zeros((128, 128), np.float64)
    image[:, :64] = rng.standard_normal(size=(128, 64))
    image[:, 64:] = 1e6  # bogus values outside the sensor
    mask = np.zeros((128, 128), bool)
    mask[:, :64] = True
    sigma = estimate_image_noise_sigma(image, sensor_mask=mask)
    assert abs(sigma - 1.0) < 0.1


def test_estimate_image_noise_sigma_ignores_fill_over_most_of_frame() -> None:
    """Zero-filled lines covering 70% of the frame leave the estimate at the noise."""
    rng = np.random.default_rng(seed=46)
    image = np.zeros((256, 256), np.float64)
    image[:77] = rng.standard_normal(size=(77, 256))  # lines read out before truncation
    sigma = estimate_image_noise_sigma(image)
    assert abs(sigma - 1.0) < 0.1


def test_estimate_image_noise_sigma_constant_frame_is_zero() -> None:
    """A frame of one value has no pixel-to-pixel noise and returns 0."""
    image = np.full((64, 64), 7.0)
    sigma = estimate_image_noise_sigma(image)
    assert sigma == 0.0


def test_estimate_image_noise_sigma_rejects_empty_mask() -> None:
    """An all-False sensor mask raises ValueError."""
    image = np.zeros((4, 4), np.float64)
    mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match='no pixels'):
        estimate_image_noise_sigma(image, sensor_mask=mask)


def test_estimate_image_noise_sigma_shape_mismatch() -> None:
    """Mismatched sensor mask shape raises ValueError."""
    image = np.zeros((4, 4), np.float64)
    mask = np.ones((4, 5), bool)
    with pytest.raises(ValueError, match='shape'):
        estimate_image_noise_sigma(image, sensor_mask=mask)


def test_estimate_image_noise_sigma_rejects_non_2d() -> None:
    """Non-2D input is rejected at the entry point."""
    image = np.zeros((4, 4, 4), np.float64)
    with pytest.raises(TypeError, match='2-D'):
        estimate_image_noise_sigma(image)


def test_estimate_background_and_sky_sigma_recovers_pedestal() -> None:
    """The estimate recovers the pedestal and noise despite a bright body."""
    rng = np.random.default_rng(seed=45)
    image = 100.0 + 2.0 * rng.standard_normal(size=(256, 256))
    image[40:120, 40:120] = 500.0  # bright body covering ~10% of the frame
    background, sky_sigma = estimate_background_and_sky_sigma(
        image, np.ones(image.shape, bool), noise_sigma=2.0
    )
    assert abs(background - 100.0) < 0.5
    assert abs(sky_sigma - 2.0) < 0.5


def test_estimate_background_and_sky_sigma_empty_mask() -> None:
    """No valid pixels returns a zero background and the floored seed sigma."""
    image = np.zeros((8, 8), np.float64)
    background, sky_sigma = estimate_background_and_sky_sigma(
        image, np.zeros((8, 8), bool), noise_sigma=3.0
    )
    assert background == 0.0
    assert sky_sigma == 3.0


def test_estimate_background_and_sky_sigma_zero_frame() -> None:
    """A constant-zero frame yields a zero pedestal and the sigma floor."""
    image = np.zeros((32, 32), np.float64)
    background, sky_sigma = estimate_background_and_sky_sigma(
        image, np.ones((32, 32), bool), noise_sigma=0.0
    )
    assert background == 0.0
    assert sky_sigma <= 1e-9
