"""Tests for spindoctor.support.correlate, focused on masked NCC and pyramid navigation."""

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from spindoctor.config import IMAGE_LOGGER
from spindoctor.support.correlate import (
    _MAX_PEAK_RATIO,
    _ncc_quadratic_axis_offset,
    _residual_correlation_area,
    evaluate_candidate,
    fourier_shift,
    gradient_magnitude,
    masked_ncc,
    matched_filter_covariance,
    navigate_single_scale_kpeaks,
    navigate_with_pyramid_kpeaks,
    nms_topk,
    peak_to_runner_up_ratio,
)
from spindoctor.support.image import normalize_array, pad_top_left
from spindoctor.support.misc import mad_std

# =========================================================================
# Helpers
# =========================================================================


def _gaussian_patch(
    shape: tuple[int, int],
    sigma: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a 2-D Gaussian patch with optional subpixel offset."""
    v_size, u_size = shape
    cv = (v_size - 1) / 2.0
    cu = (u_size - 1) / 2.0
    vv, uu = np.meshgrid(np.arange(v_size), np.arange(u_size), indexing='ij')
    dv = vv - (cv + offset[0])
    du = uu - (cu + offset[1])
    return np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))


def _make_single_star(
    *,
    image_size: tuple[int, int] = (64, 64),
    model_size: tuple[int, int] = (64, 64),
    star_sigma: float = 2.0,
    mask_half: int = 15,
    image_offset: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic single-star scene with known offset."""
    ih, iw = image_size
    mh, mw = model_size
    psf_size = (2 * mask_half + 1, 2 * mask_half + 1)

    image = np.zeros(image_size, dtype=np.float64)
    icv, icu = ih // 2, iw // 2
    image[
        icv - mask_half : icv + mask_half + 1,
        icu - mask_half : icu + mask_half + 1,
    ] = _gaussian_patch(psf_size, star_sigma, offset=image_offset)

    model = np.zeros(model_size, dtype=np.float64)
    mask = np.zeros(model_size, dtype=bool)
    mcv, mcu = mh // 2, mw // 2
    model[
        mcv - mask_half : mcv + mask_half + 1,
        mcu - mask_half : mcu + mask_half + 1,
    ] = _gaussian_patch(psf_size, star_sigma, offset=(0.0, 0.0))
    mask[
        mcv - mask_half : mcv + mask_half + 1,
        mcu - mask_half : mcu + mask_half + 1,
    ] = True

    return image, model, mask


def _make_multi_star(
    *,
    image_size: tuple[int, int] = (64, 64),
    star_sigma: float = 2.0,
    mask_half: int = 5,
    image_offset: tuple[float, float] = (1.0, 0.0),
    positions: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic multi-star scene with all stars sharing the same offset."""
    if positions is None:
        positions = [(20, 20), (32, 40), (45, 15)]

    psf_size = (2 * mask_half + 1, 2 * mask_half + 1)
    image = np.zeros(image_size, dtype=np.float64)
    model = np.zeros(image_size, dtype=np.float64)
    mask = np.zeros(image_size, dtype=bool)

    for sv, su in positions:
        image[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] += _gaussian_patch(psf_size, star_sigma, offset=image_offset)
        model[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] += _gaussian_patch(psf_size, star_sigma, offset=(0.0, 0.0))
        mask[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] = True

    return image, model, mask


# =========================================================================
# masked_ncc unit tests
# =========================================================================


class TestMaskedNcc:
    """Tests for the masked_ncc function."""

    def test_ncc_perfect_match_peaks_at_one(self) -> None:
        """NCC at zero shift should be 1.0 when image == model."""
        size = (32, 32)
        model = np.zeros(size)
        mask = np.zeros(size, dtype=bool)
        model[10:20, 10:20] = _gaussian_patch((10, 10), 2.0)
        mask[10:20, 10:20] = True
        image = model.copy()

        ph, pw = size[0] * 2, size[1] * 2
        ip = pad_top_left(image, ph, pw)
        mp = pad_top_left(model, ph, pw)
        wp = pad_top_left(mask, ph, pw)
        ncc, _ = masked_ncc(ip, mp, wp)

        assert ncc[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_ncc_bounded(self) -> None:
        """NCC values must lie in [-1, 1] for a single-star scene."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc, _ = masked_ncc(ip, mp, wp)

        assert np.all(ncc >= -1.0 - 1e-6)
        assert np.all(ncc <= 1.0 + 1e-6)

    def test_ncc_peak_at_correct_offset(self) -> None:
        """Peak of NCC surface coincides with known single-star offset."""
        image, model, mask = _make_single_star(image_offset=(2.0, 0.0))
        mh, mw = model.shape
        ih, iw = image.shape
        ph, pw = ih + mh, iw + mw
        ip = pad_top_left(image, ph, pw)
        mp = pad_top_left(model, ph, pw)
        wp = pad_top_left(mask, ph, pw)
        ncc, _ = masked_ncc(ip, mp, wp)

        peak_idx = np.unravel_index(np.argmax(ncc), ncc.shape)
        assert int(peak_idx[0]) == 2
        assert int(peak_idx[1]) == 0

    def test_ncc_real_valued(self) -> None:
        """NCC output must be a real-valued (non-complex) array."""
        image, model, mask = _make_single_star()
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc, num = masked_ncc(ip, mp, wp)

        assert not np.iscomplexobj(ncc)
        assert not np.iscomplexobj(num)

    def test_numerator_peaks_at_correct_offset_sparse_template(self) -> None:
        """NCC numerator peaks correctly even when template is sparse in mask.

        When the PSF covers only a small fraction of the mask, the NCC
        itself can plateau near 1.0 at many shifts.  The numerator
        must still peak at the correct offset because it scales with
        the image variance under the mask.
        """
        image, model, mask = _make_single_star(
            image_size=(128, 128),
            model_size=(128, 128),
            star_sigma=2.0,
            mask_half=30,
            image_offset=(3.0, -2.0),
        )
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        _ncc, num = masked_ncc(ip, mp, wp)

        num_peak = np.unravel_index(np.argmax(num), num.shape)
        assert int(num_peak[0]) == 3
        assert int(num_peak[1]) == ip.shape[1] - 2

    def test_ncc_mask_excludes_padding(self) -> None:
        """Changing model values outside mask must not alter the NCC."""
        image, model, mask = _make_single_star()
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc1, num1 = masked_ncc(ip, mp, wp)

        model2 = model.copy()
        model2[~mask] = 999.0
        mp2 = pad_top_left(model2, ih + mh, iw + mw)
        ncc2, num2 = masked_ncc(ip, mp2, wp)

        np.testing.assert_allclose(ncc1, ncc2, atol=1e-10)
        np.testing.assert_allclose(num1, num2, atol=1e-10)


# =========================================================================
# Single-scale correlation tests
# =========================================================================


class TestSingleScale:
    """Tests for navigate_single_scale_kpeaks."""

    def test_single_star_integer_offset(self) -> None:
        """Single-star with integer offset converges to the correct shift."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            logger=None,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_single_star_subpixel_offset(self) -> None:
        """Single-star with subpixel offset converges within tolerance."""
        image, model, mask = _make_single_star(image_offset=(0.3, -0.7))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=64,
            metric='psr',
            logger=None,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(0.3, abs=0.05)
        assert dx == pytest.approx(-0.7, abs=0.05)

    def test_single_star_quality_above_threshold(self) -> None:
        """PSR quality for a clean single-star must be well above 6.0."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            logger=None,
        )
        assert result['quality'] > 6.0

    def test_no_candidates_result_carries_full_key_set(self) -> None:
        """When no peaks survive ``max_offset_vu`` the result still has every key.

        Regression: the empty-candidates early-return previously omitted
        ``peak_val`` and ``rc``, so callers that logged or read those
        keys (notably ``navigate_with_pyramid_kpeaks``) crashed with a
        ``KeyError`` whenever the search collapsed.  The contract now
        matches the populated path exactly.
        """
        # Empty image + empty model => no real peak; force the
        # max_offset_vu window down to (0, 0) so no candidate clears the
        # window and the early-return branch fires deterministically.
        image = np.zeros((16, 16), dtype=np.float64)
        model = np.zeros((16, 16), dtype=np.float64)
        mask = np.ones((16, 16), dtype=bool)
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=3,
            upsample_factor=8,
            metric='psr',
            max_offset_vu=(0, 0),
            logger=None,
        )
        for key in ('offset', 'cov', 'sigma_xy', 'quality', 'peak_val', 'rc', 'all_candidates'):
            assert key in result, f'no-candidates result missing key {key!r}'
        assert result['all_candidates'] == []
        assert result['quality'] == -np.inf


# =========================================================================
# Pyramid correlation tests
# =========================================================================


class TestPyramid:
    """Tests for navigate_with_pyramid_kpeaks."""

    def test_single_star_not_spurious(self) -> None:
        """Pyramid must not flag a clean single-star as spurious."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_single_star_subpixel(self) -> None:
        """Pyramid converges for a single-star with subpixel offset."""
        image, model, mask = _make_single_star(image_offset=(0.5, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=64,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(0.5, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_multi_star_converges(self) -> None:
        """Multi-star scene converges and is not flagged as spurious."""
        image, model, mask = _make_multi_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.1)
        assert dx == pytest.approx(0.0, abs=0.1)

    def test_zero_offset(self) -> None:
        """No offset between image and model returns approximately (0, 0)."""
        image, model, mask = _make_single_star(image_offset=(0.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=2,
            max_peaks=3,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(0.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)


# =========================================================================
# max_offset_vu tests
# =========================================================================


class TestMaxOffsetVu:
    """Tests that max_offset_vu correctly restricts the correlation search."""

    def test_nms_topk_excludes_out_of_bounds_peaks(self) -> None:
        """nms_topk must not return peaks outside max_offset_vu."""
        rng = np.random.default_rng(42)
        corr = rng.standard_normal((64, 64))
        # Plant a large value far outside the allowed region.
        corr[30, 30] = 1000.0  # signed offset (30-64, 30-64) = (-34, -34) -- outside (10,10)
        # Plant a smaller value inside the allowed region.
        corr[5, 3] = 500.0  # signed offset (5, 3) -- inside (10, 10)

        peaks = nms_topk(corr, k=3, radius=2, max_offset_vu=(10, 10))
        rows = [r for r, _c, _v in peaks]
        cols = [c for _r, c, _v in peaks]
        # The out-of-bounds peak at (30, 30) must not appear.
        for r, c in zip(rows, cols, strict=True):
            dv = r if r < 32 else r - 64
            du = c if c < 32 else c - 64
            assert abs(dv) <= 10
            assert abs(du) <= 10

    def test_nms_topk_no_limit_returns_out_of_bounds(self) -> None:
        """Without max_offset_vu the large out-of-bounds peak is selected first."""
        corr = np.zeros((64, 64))
        corr[30, 30] = 1000.0
        peaks = nms_topk(corr, k=1, radius=2, max_offset_vu=None)
        assert len(peaks) == 1
        assert int(peaks[0][0]) == 30
        assert int(peaks[0][1]) == 30

    def test_single_scale_with_max_offset_finds_correct_peak(self) -> None:
        """max_offset_vu forces the single-scale search to the correct small offset."""
        image, model, mask = _make_single_star(image_offset=(2.0, 0.0))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            max_offset_vu=(10, 10),
            logger=None,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(2.0, abs=0.1)
        assert dx == pytest.approx(0.0, abs=0.1)

    def test_pyramid_with_max_offset_not_spurious(self) -> None:
        """Pyramid with extfov-style max_offset_vu still converges cleanly."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
            max_offset_vu=(20, 20),
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.1)
        assert dx == pytest.approx(0.0, abs=0.1)

    def test_pyramid_peak_at_window_edge_is_spurious(self) -> None:
        """Pyramid result sitting within 1 pixel of the max-offset edge is spurious.

        Reason: a correlation peak at the boundary usually means the true peak
        is clipped outside the search window, so the reported offset cannot be
        trusted.
        """
        image, model, mask = _make_single_star(image_offset=(2.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=0.0,
            consistency_tol=10.0,
            max_offset_vu=(3, 10),
        )
        assert result['spurious']


# =========================================================================
# Gradient-mode tests
# =========================================================================


class TestGradientMode:
    """Gradient-mode NCC (``use_gradient=True``) for body-overflows-FOV scenes."""

    def test_gradient_magnitude_highlights_edges(self) -> None:
        """Gradient magnitude is ~0 on an interior flat region and large at an interior step."""
        # Check interior only: sobel in 'constant' mode picks up the array-border
        # as an edge, which is intentional (real images have a hard boundary to
        # the zero-padded extfov margin); the data_mask used by masked_ncc
        # excludes those pixels from the correlation at use time.
        flat = np.full((30, 30), 7.0)
        g_flat = gradient_magnitude(flat)
        assert float(g_flat[5:25, 5:25].max()) < 1e-9

        step = np.zeros((30, 30))
        step[:, 15:] = 1.0
        g_step = gradient_magnitude(step)
        # Sobel magnitude at a unit step is 4; check interior of each side stays flat.
        assert float(g_step[5:25, 14:16].max()) > 0.5
        assert float(g_step[5:25, 5:10].max()) < 1e-9
        assert float(g_step[5:25, 20:25].max()) < 1e-9

    def test_pyramid_gradient_converges_on_star_scene(self) -> None:
        """use_gradient=True still converges to the planted offset on a simple star scene."""
        image, model, mask = _make_single_star(image_offset=(1.5, -0.5))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=0.0,
            consistency_tol=10.0,
            max_offset_vu=(10, 10),
            use_gradient=True,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(1.5, abs=0.3)
        assert dx == pytest.approx(-0.5, abs=0.3)


class TestPyramidTopKPeaks:
    """``navigate_with_pyramid_kpeaks`` surfaces a sorted ``top_k_peaks`` list."""

    def test_top_k_peaks_present_and_sorted(self) -> None:
        """The returned dict carries the per-peak telemetry sorted by quality."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=3,
            upsample_factor=16,
            metric='psr',
            quality_thresh=0.0,
            consistency_tol=10.0,
            max_offset_vu=(10, 10),
        )
        peaks = result['top_k_peaks']
        assert isinstance(peaks, list)
        assert len(peaks) >= 1
        # First peak quality matches the headline ``quality`` value.
        assert peaks[0][0] == pytest.approx(result['quality'])
        # Sorted by quality descending.
        for prev, cur in pairwise(peaks):
            assert prev[0] >= cur[0]
        # Each entry has shape (quality, dv, du).
        for entry in peaks:
            assert len(entry) == 3


# =========================================================================
# peak_to_runner_up_ratio
# =========================================================================


def test_peak_to_runner_up_ratio_empty_and_single() -> None:
    """No peaks -> 0.0; a single surviving peak -> 1.0."""
    assert peak_to_runner_up_ratio([]) == 0.0
    assert peak_to_runner_up_ratio([(5.0, 0.0, 0.0)]) == 1.0


def test_peak_to_runner_up_ratio_ordinary() -> None:
    """A clear winner returns winner / runner-up quality."""
    ratio = peak_to_runner_up_ratio([(8.0, 0.0, 0.0), (2.0, 1.0, 1.0)])
    assert ratio == pytest.approx(4.0)


def test_peak_to_runner_up_ratio_caps_near_zero_runner_up() -> None:
    """CODE-NAV-022: a near-zero runner-up returns the cap, not ~1e9."""
    ratio = peak_to_runner_up_ratio([(1.0, 0.0, 0.0), (1e-12, 1.0, 1.0)])
    assert ratio == _MAX_PEAK_RATIO


def test_peak_to_runner_up_ratio_caps_independently_of_winner_magnitude() -> None:
    """The capped near-zero-runner result no longer scales with the winner."""
    small = peak_to_runner_up_ratio([(1e-6, 0.0, 0.0), (0.0, 1.0, 1.0)])
    large = peak_to_runner_up_ratio([(1e3, 0.0, 0.0), (0.0, 1.0, 1.0)])
    assert small == large == _MAX_PEAK_RATIO


def test_peak_to_runner_up_ratio_nonpositive_winner_is_zero() -> None:
    """A non-positive winner with a non-positive runner-up returns 0.0."""
    assert peak_to_runner_up_ratio([(0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)]) == 0.0


def test_peak_to_runner_up_ratio_clamps_large_ordinary_ratio() -> None:
    """An ordinary ratio above the cap is clamped to _MAX_PEAK_RATIO."""
    ratio = peak_to_runner_up_ratio([(1e9, 0.0, 0.0), (1.0, 1.0, 1.0)])
    assert ratio == _MAX_PEAK_RATIO


# =========================================================================
# Matched-filter (peak-curvature) covariance
# =========================================================================


def _gaussian_model(sigma: float = 4.0, size: int = 64) -> np.ndarray:
    """A centered unit-amplitude Gaussian bump with gradients on both axes."""
    return _gaussian_patch((size, size), sigma)


def test_matched_filter_covariance_matches_white_noise_crlb() -> None:
    """cov (area=1) equals sigma_n^2 * inv(sum grad grad^T) within a tight factor."""
    model = _gaussian_model()
    sigma_n = 0.05
    sx = np.gradient(model, axis=1)
    expected_var_u = sigma_n**2 / float(np.sum(sx * sx))
    cov = matched_filter_covariance(model, sigma_n, correlation_area=1.0)
    assert cov[1, 1] == pytest.approx(expected_var_u, rel=0.5)


def test_matched_filter_covariance_scales_with_noise_variance() -> None:
    """Doubling the residual noise quadruples the covariance."""
    model = _gaussian_model()
    cov_lo = matched_filter_covariance(model, 0.05, correlation_area=1.0)
    cov_hi = matched_filter_covariance(model, 0.10, correlation_area=1.0)
    assert cov_hi[0, 0] == pytest.approx(4.0 * cov_lo[0, 0], rel=1e-6)


def test_matched_filter_covariance_scales_with_correlation_area() -> None:
    """The covariance is linear in the correlation-area inflation factor."""
    model = _gaussian_model()
    cov_1 = matched_filter_covariance(model, 0.05, correlation_area=1.0)
    cov_10 = matched_filter_covariance(model, 0.05, correlation_area=10.0)
    assert cov_10[0, 0] == pytest.approx(10.0 * cov_1[0, 0], rel=1e-6)


def test_matched_filter_covariance_is_amplitude_invariant_when_normalized() -> None:
    """A bright and a faint image give the same sigma once both are normalized.

    This is the unit-consistency property the derivation restores: the
    covariance must not scale with the (arbitrary) template amplitude.
    """
    rng = np.random.default_rng(3)
    model = _gaussian_model()
    noise = rng.normal(0.0, 0.02, model.shape)
    sigmas = []
    for amplitude in (1.0, 1000.0):
        image = amplitude * model + amplitude * noise
        model_scaled = amplitude * model
        model_norm = normalize_array(model_scaled)
        resid = normalize_array(image) - model_norm
        cov = matched_filter_covariance(model_norm, mad_std(resid), correlation_area=1.0)
        sigmas.append(float(np.sqrt(cov[0, 0])))
    assert sigmas[1] == pytest.approx(sigmas[0], rel=0.05)


def test_matched_filter_covariance_controlled_surface_within_factor() -> None:
    """A known peak sharpness + known noise recovers the expected sigma (factor ~1.5).

    Feeds the covariance a normalized model with a known gradient sum and a
    controlled residual noise, and checks the reported per-axis sigma lands
    within a factor of 1.5 of the closed-form matched-filter bound.
    """
    model = _gaussian_model(sigma=3.0)
    model_norm = normalize_array(model)
    sigma_n = 0.1
    sx = np.gradient(model_norm, axis=1)
    expected_sigma_u = np.sqrt(sigma_n**2 / float(np.sum(sx * sx)))
    cov = matched_filter_covariance(model_norm, sigma_n, correlation_area=1.0)
    reported_sigma_u = float(np.sqrt(cov[1, 1]))
    assert 1.0 / 1.5 <= reported_sigma_u / expected_sigma_u <= 1.5


def test_residual_correlation_area_white_noise_is_near_one() -> None:
    """A white-noise residual has a correlation area close to 1 pixel."""
    rng = np.random.default_rng(1)
    white = rng.normal(0.0, 1.0, (96, 96))
    area = _residual_correlation_area(white)
    assert area < 2.5


def test_residual_correlation_area_smoothed_field_exceeds_white() -> None:
    """Spatially correlated residuals report a larger correlation area."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    white = rng.normal(0.0, 1.0, (96, 96))
    smoothed = gaussian_filter(white, sigma=3.0)
    assert _residual_correlation_area(smoothed) > 5.0 * _residual_correlation_area(white)


# =========================================================================
# Sub-pixel refinement saturation fallback
# =========================================================================


def test_ncc_quadratic_axis_offset_recovers_parabola_vertex() -> None:
    """The three-point fit returns the exact vertex of a sampled parabola."""
    vertex = 0.31
    samples = [-((t - vertex) ** 2) for t in (-1.0, 0.0, 1.0)]
    assert _ncc_quadratic_axis_offset(*samples) == pytest.approx(vertex)


def test_ncc_quadratic_axis_offset_nonfinite_neighbor_returns_zero() -> None:
    """An invalid-shift sentinel next to the peak disables the fit."""
    assert _ncc_quadratic_axis_offset(float('-inf'), 1.0, 0.5) == 0.0


def test_ncc_quadratic_axis_offset_flat_neighborhood_returns_zero() -> None:
    """A flat (curvature-free) neighborhood carries no sub-pixel information."""
    assert _ncc_quadratic_axis_offset(0.5, 0.5, 0.5) == 0.0


def test_ncc_quadratic_axis_offset_degenerate_curvature_returns_zero() -> None:
    """Rounding-noise curvature keeps the integer peak instead of exploding."""
    assert _ncc_quadratic_axis_offset(0.1, 0.2, 0.3) == 0.0


def test_ncc_quadratic_axis_offset_neighbor_tie_is_half_pixel() -> None:
    """A neighbor equal to the peak puts the vertex exactly halfway between them."""
    assert _ncc_quadratic_axis_offset(0.5, 1.0, 1.0) == pytest.approx(0.5)


def test_ncc_quadratic_axis_offset_neighbor_above_center_returns_zero() -> None:
    """A neighbor above the center is not a local maximum; keep the integer peak.

    A steeply tilted concave triple has negative curvature yet its highest
    sample is a neighbor; fitting it would report a clipped half-pixel shift
    away from the true maximum.
    """
    assert _ncc_quadratic_axis_offset(1.0, 0.9, 0.0) == 0.0
    assert _ncc_quadratic_axis_offset(0.0, 0.9, 1.0) == 0.0


def _evaluate_candidate_for_shift(
    true_shift_vu: tuple[float, float],
    corr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run evaluate_candidate on a Gaussian scene with integer peak (0, 0).

    The cross-power spectrum's true peak sits at ``true_shift_vu``; passing
    ``rc=(0, 0)`` makes the refinement window span ``[-0.5, +0.5]`` px, so a
    true shift beyond half a pixel saturates the window.  ``corr`` defaults
    to a surface whose quadratic vertex along v is ``+0.25``.

    Parameters:
        true_shift_vu: The ``(dv, du)`` shift in pixels planted between the
            image and the model via a Fourier shift.
        corr: Optional fabricated NCC surface handed to the candidate; the
            default puts the integer peak at ``(0, 0)`` with v-neighbors 0.8
            and 0.4 (quadratic vertex ``+0.25``) and a flat u axis.

    Returns:
        The candidate dictionary from :func:`evaluate_candidate`; the tests
        read its ``offset`` entry.
    """
    base = _gaussian_patch((64, 64), 3.0)
    shifted = fourier_shift(base, true_shift_vu[0], true_shift_vu[1])
    mask = np.ones_like(base, dtype=bool)
    if corr is None:
        corr = np.zeros_like(base)
        corr[0, 0] = 1.0
        corr[1, 0] = 0.8
        corr[-1, 0] = 0.4
    return evaluate_candidate(
        image_pad=shifted,
        model_pad=base,
        mask_pad=mask,
        corr=corr,
        rc=(0, 0),
        upsample_factor=64,
        model_shape=(64, 64),
        image_shape=(64, 64),
        logger=IMAGE_LOGGER,
    )


def test_evaluate_candidate_interior_peak_refines_on_cross_power() -> None:
    """A sub-half-pixel true shift is refined by the upsampled DFT as before."""
    result = _evaluate_candidate_for_shift((0.3, -0.2))
    assert result['offset'][0] == pytest.approx(0.3, abs=0.03)
    assert result['offset'][1] == pytest.approx(-0.2, abs=0.03)


def test_evaluate_candidate_saturated_axis_falls_back_to_ncc_quadratic() -> None:
    """A cross-power peak beyond the half-pixel window uses the NCC vertex.

    The fabricated NCC surface has its quadratic vertex at ``+0.25`` along v,
    so the saturated v axis reports that instead of pinning at exactly
    ``+0.5``; the u axis stays on the (interior) upsampled-DFT estimate.
    """
    result = _evaluate_candidate_for_shift((1.2, 0.0))
    assert result['offset'][0] == pytest.approx(0.25, abs=1e-6)
    assert result['offset'][1] == pytest.approx(0.0, abs=0.03)


def test_evaluate_candidate_saturated_axis_never_reports_half_pixel_pin() -> None:
    """With a flat NCC neighborhood the saturated axis keeps the integer peak."""
    corr = np.zeros((64, 64))
    corr[0, 0] = 1.0
    result = _evaluate_candidate_for_shift((1.2, 0.0), corr=corr)
    assert result['offset'][0] == pytest.approx(0.0, abs=1e-6)
