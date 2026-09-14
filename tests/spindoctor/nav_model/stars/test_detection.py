"""Tests for ``spindoctor.nav_model.stars.detection``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from psfmodel import PSF

from spindoctor.nav_model.stars.detection import (
    DAOPHOT_DEFAULT_DETECTION_SIGMA,
    DAOPHOT_DEFAULT_ROUNDNESS_BOUND,
    DAOPHOT_DEFAULT_SHARPNESS_MAX,
    DAOPHOT_DEFAULT_SHARPNESS_MIN,
    DetectedSource,
    _marginal_variance,
    _sharpness_roundness,
    apply_shape_cuts,
    centroid_gaussian_fit,
    centroid_saturated,
    detect_ccd_bloom_columns,
    detect_sources,
    matched_filter_image,
)


class _FakePSF:
    """Gaussian-like PSF stand-in exposing ``sigma``."""

    def __init__(self, sigma: float = 1.2) -> None:
        self.sigma = sigma


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Render a 2-D Gaussian PSF of the given pixel size and sigma."""
    coords = np.arange(size) - (size - 1) / 2.0
    g1 = np.exp(-(coords**2) / (2.0 * sigma**2))
    return np.outer(g1, g1)


def test_constants_have_documented_values() -> None:
    """Default cuts come straight from the module-level constants."""
    assert pytest.approx(4.0) == DAOPHOT_DEFAULT_DETECTION_SIGMA
    assert pytest.approx(0.2) == DAOPHOT_DEFAULT_SHARPNESS_MIN
    assert pytest.approx(1.0) == DAOPHOT_DEFAULT_SHARPNESS_MAX
    assert pytest.approx(1.0) == DAOPHOT_DEFAULT_ROUNDNESS_BOUND


def test_matched_filter_image_rejects_zero_kernel() -> None:
    """A flat kernel has zero energy after mean subtraction; the helper raises."""
    image = np.zeros((10, 10), dtype=np.float64)
    kernel = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='zero energy'):
        matched_filter_image(image, kernel=kernel)


def test_matched_filter_image_returns_input_shape() -> None:
    """The matched-filter response shares the input shape."""
    image = np.zeros((20, 30), dtype=np.float64)
    kernel = _gaussian_kernel(5, 1.0)
    out = matched_filter_image(image, kernel=kernel)
    assert out.shape == image.shape


def test_matched_filter_image_rejects_non_2d_image() -> None:
    """A non-2-D image raises ``TypeError`` mentioning the rank."""
    image = np.zeros(10, dtype=np.float64)
    with pytest.raises(TypeError, match='requires a 2-D image'):
        matched_filter_image(image, kernel=_gaussian_kernel(3, 1.0))


def test_matched_filter_image_rejects_non_2d_kernel() -> None:
    """A non-2-D kernel raises ``TypeError`` mentioning the rank."""
    image = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(TypeError, match='kernel must be 2-D'):
        matched_filter_image(image, kernel=np.ones(3, dtype=np.float64))


def test_detect_ccd_bloom_columns_marks_full_column() -> None:
    """A 5-row saturation run marks the entire column True."""
    image = np.zeros((10, 5), dtype=np.float64)
    image[2:7, 2] = 5000.0
    mask = detect_ccd_bloom_columns(image, full_well_dn=4095.0, min_run=5)
    assert mask[:, 2].all()
    expected_other = np.zeros_like(mask, dtype=bool)
    expected_other[:, 2] = True
    assert (mask == expected_other).all()


def test_detect_ccd_bloom_columns_skips_short_run() -> None:
    """A 4-row saturation run is below ``min_run=5`` and produces no bloom mask."""
    image = np.zeros((10, 5), dtype=np.float64)
    image[2:6, 2] = 5000.0
    mask = detect_ccd_bloom_columns(image, full_well_dn=4095.0, min_run=5)
    assert not mask.any()


def test_detect_ccd_bloom_columns_rejects_min_run_below_2() -> None:
    """``min_run < 2`` raises ``ValueError`` naming the value."""
    image = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match='min_run must be >= 2'):
        detect_ccd_bloom_columns(image, full_well_dn=4095.0, min_run=1)


def test_centroid_gaussian_fit_returns_zero_offset_for_centered_blob() -> None:
    """A symmetric Gaussian centered on the box returns ``(0, 0)``."""
    box = _gaussian_kernel(5, 1.0)
    dv, du = centroid_gaussian_fit(box)
    assert dv == pytest.approx(0.0, abs=1e-9)
    assert du == pytest.approx(0.0, abs=1e-9)


def test_centroid_gaussian_fit_offset_centroid() -> None:
    """A blob shifted by 1 pixel right returns positive ``du``."""
    full = np.zeros((5, 5), dtype=np.float64)
    full[2, 3] = 100.0
    full[2, 2] = 30.0
    full[2, 4] = 30.0
    dv, du = centroid_gaussian_fit(full)
    assert dv == pytest.approx(0.0, abs=1e-9)
    assert du > 0.0


def test_centroid_gaussian_fit_rejects_non_square_box() -> None:
    """An even-sided or non-square box raises ``ValueError`` with shape."""
    box = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match='must be square odd'):
        centroid_gaussian_fit(box)


def test_centroid_saturated_uses_annular_moment() -> None:
    """Saturated cores fall back to an annular brightness moment."""
    box = np.zeros((5, 5), dtype=np.float64)
    # Saturated center + lit annulus on the right side.
    box[2, 2] = 4096.0
    box[2, 3] = 1000.0
    box[1, 3] = 500.0
    box[3, 3] = 500.0
    dv, du = centroid_saturated(box, full_well_dn=4095.0, half_width_inner=1, half_width_outer=2)
    assert dv == pytest.approx(0.0, abs=1e-6)
    assert du > 0.0


def test_centroid_saturated_rejects_inverted_annulus() -> None:
    """``half_width_inner >= half_width_outer`` raises ``ValueError``."""
    box = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match='expected 0 <= half_width_inner'):
        centroid_saturated(box, full_well_dn=4095.0, half_width_inner=2, half_width_outer=2)


def test_apply_shape_cuts_keeps_clean_detection() -> None:
    """A typical PSF detection (sharp ~0.5, round ~0) passes the default cuts."""
    assert apply_shape_cuts(0.5, 0.0)


def test_apply_shape_cuts_rejects_hot_pixel() -> None:
    """A single-pixel spike (sharpness above max) is rejected."""
    assert not apply_shape_cuts(1.5, 0.0)


def test_apply_shape_cuts_rejects_extended_source() -> None:
    """A wide blob (sharpness below min) is rejected."""
    assert not apply_shape_cuts(0.05, 0.0)


def test_apply_shape_cuts_rejects_high_roundness() -> None:
    """A bloom-shaped detection (|roundness| > 1) is rejected."""
    assert not apply_shape_cuts(0.5, 1.5)


def test_detect_sources_finds_planted_star() -> None:
    """A single Gaussian planted in noise is recovered with sub-pixel centroid.

    The kernel is symmetric about its own middle sample, so planting it into
    rows 20 to 26 and columns 30 to 36 puts its center on the center of pixel
    ``(23, 33)``, which is the pixel centric position the centroid has to come
    back with.  The seeded noise realization moves it by about a thousandth of
    a pixel, and the bound is ten times that, two orders of magnitude below
    the half pixel that separates the two coordinate systems.
    """
    rng = np.random.default_rng(0)
    image = rng.normal(scale=0.5, size=(50, 50)).astype(np.float64)
    star = _gaussian_kernel(7, 1.2) * 200.0
    image[20:27, 30:37] += star
    psf = _FakePSF(sigma=1.2)
    smear = _gaussian_kernel(7, 1.2)
    sources = detect_sources(
        image,
        psf=psf,  # type: ignore[arg-type]
        image_noise_sigma=0.5,
        full_well_dn=10_000.0,
        smear_kernel=smear,
        bloom_mask=None,
        detection_sigma=4.0,
    )
    assert len(sources) >= 1
    best = max(sources, key=lambda s: s.peak_dn)
    assert best.v == pytest.approx(23.0, abs=0.01)
    assert best.u == pytest.approx(33.0, abs=0.01)
    assert isinstance(best, DetectedSource)
    assert best.saturated is False


def test_detect_sources_skips_bloom_columns() -> None:
    """Detections falling inside the bloom mask are suppressed."""
    image = np.zeros((30, 30), dtype=np.float64)
    star = _gaussian_kernel(5, 1.0) * 200.0
    image[10:15, 10:15] += star
    psf = _FakePSF(sigma=1.0)
    smear = _gaussian_kernel(5, 1.0)
    bloom = np.zeros_like(image, dtype=bool)
    bloom[:, 10:15] = True
    sources = detect_sources(
        image,
        psf=psf,  # type: ignore[arg-type]
        image_noise_sigma=0.5,
        full_well_dn=10_000.0,
        smear_kernel=smear,
        bloom_mask=bloom,
        detection_sigma=4.0,
    )
    assert sources == []


def test_centroid_gaussian_fit_returns_zero_when_box_is_below_background() -> None:
    """A box where every pixel is at the median returns ``(0, 0)``."""
    box = np.full((5, 5), 100.0, dtype=np.float64)
    assert centroid_gaussian_fit(box) == (0.0, 0.0)


def test_centroid_saturated_returns_zero_when_annulus_below_background() -> None:
    """A saturated box with no annular signal returns ``(0, 0)``."""
    box = np.full((5, 5), 100.0, dtype=np.float64)
    box[2, 2] = 5000.0  # saturated center, no annular signal
    out = centroid_saturated(box, full_well_dn=4095.0, half_width_inner=1, half_width_outer=2)
    assert out == (0.0, 0.0)


def test_centroid_saturated_rejects_even_sided_box() -> None:
    """A 4x4 box raises ``ValueError`` mentioning the shape."""
    box = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match='must be square odd'):
        centroid_saturated(box, full_well_dn=4095.0, half_width_inner=0, half_width_outer=2)


def test_psf_sigma_falls_back_to_fwhm_when_no_sigma_attribute() -> None:
    """``detect_sources`` uses ``psf.fwhm()`` when ``psf.sigma`` is absent."""

    class _FwhmOnlyPSF:
        def fwhm(self) -> float:
            return 2.3548200450309493  # sigma == 1.0

    image = np.zeros((30, 30), dtype=np.float64)
    smear = _gaussian_kernel(5, 1.0)
    sources = detect_sources(
        image,
        psf=cast('PSF', _FwhmOnlyPSF()),
        image_noise_sigma=0.5,
        full_well_dn=10_000.0,
        smear_kernel=smear,
    )
    assert sources == []


def test_detect_sources_finds_saturated_star() -> None:
    """A bright planted star drives ``saturated=True`` on the detection."""
    rng = np.random.default_rng(0)
    image = rng.normal(scale=0.5, size=(50, 50)).astype(np.float64)
    star = _gaussian_kernel(7, 1.2) * 50_000.0
    image[20:27, 30:37] += star
    sources = detect_sources(
        image,
        psf=cast('PSF', _FakePSF(sigma=1.2)),
        image_noise_sigma=0.5,
        full_well_dn=10_000.0,
        smear_kernel=_gaussian_kernel(7, 1.2),
    )
    assert any(s.saturated for s in sources)


def test_sharpness_roundness_returns_zero_for_dark_center() -> None:
    """A dark-center box returns ``(0, 0)`` directly via the ``center <= 0`` early-out."""
    box = np.full((5, 5), -1.0, dtype=np.float64)
    sharp, round_ = _sharpness_roundness(box)
    assert sharp == 0.0
    assert round_ == 0.0


def test_sharpness_roundness_returns_zero_round_when_marginals_are_flat() -> None:
    """Symmetric marginals collapse the variance to zero; roundness is 0."""
    # Uniform fill; col_var == row_var == 0 -> roundness = 0 by short-circuit.
    box = np.full((5, 5), 100.0, dtype=np.float64)
    box[2, 2] = 200.0  # bump the center so ``center > 0``
    sharp, round_ = _sharpness_roundness(box)
    assert round_ == 0.0
    assert sharp > 0.0


def test_sharpness_roundness_rejects_non_square_box() -> None:
    """A non-square box raises ``ValueError`` mentioning the shape."""
    box = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match='must be square odd'):
        _sharpness_roundness(box)


def test_marginal_variance_returns_zero_for_uniform_profile() -> None:
    """A uniform profile has zero unmasked weight; the helper returns 0."""
    assert _marginal_variance(np.full(5, 100.0)) == 0.0
