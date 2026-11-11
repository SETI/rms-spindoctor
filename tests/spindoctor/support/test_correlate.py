"""Tests for spindoctor.support.correlate, focused on masked NCC and pyramid navigation."""

import weakref
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import scipy.fft
from scipy.ndimage import gaussian_filter

import spindoctor.support.correlate as correlate_module
from spindoctor.support.correlate import (
    _MAX_PEAK_RATIO,
    _NCC_BIDIR_VAR_FRAC_MIN,
    _NCC_BIDIR_W_FRAC_MIN,
    _NCC_EPS,
    _correlate_from_spectra,
    _crop_circular,
    _masked_ncc_bidir,
    _residual_correlation_area,
    gradient_magnitude,
    masked_ncc,
    matched_filter_covariance,
    navigate_single_scale_kpeaks,
    navigate_with_pyramid_kpeaks,
    nms_topk,
    peak_to_runner_up_ratio,
    psr_metric,
)
from spindoctor.support.image import normalize_array, pad_top_left
from spindoctor.support.misc import mad_std
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

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

    def test_a_shift_longer_than_the_image_is_recovered(self) -> None:
        """A model much larger than the image recovers a shift longer than the image.

        The image is a window of the model 30 rows and 40 columns in, so the shift is
        (-30, -40), beyond the image's own height and width.  Such a lag lies where the
        surface's positive and negative lags meet, so it is found only if the surface
        is laid out with the image's extent on the positive side.
        """
        rng = np.random.default_rng(945)
        model = gaussian_filter(rng.normal(size=(64, 80)), 2.0)
        image = model[30:54, 40:69].copy()
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=np.ones(model.shape, dtype=bool),
            max_peaks=3,
            upsample_factor=16,
            logger=None,
        )
        assert result['offset'] == pytest.approx((-30.0, -40.0), abs=0.1)

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

    def test_no_refinement_spectrum_is_built_when_no_peak_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A search with no surviving peak returns before the refinement transforms."""

        def _refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError('the refinement spectrum was built with no peak to refine')

        monkeypatch.setattr(correlate_module, 'refinement_cross_spectrum', _refuse)
        result = navigate_single_scale_kpeaks(
            image=np.zeros((16, 16), dtype=np.float64),
            model=np.zeros((16, 16), dtype=np.float64),
            mask=np.ones((16, 16), dtype=bool),
            max_peaks=3,
            upsample_factor=8,
            metric='psr',
            max_offset_vu=(0, 0),
            logger=None,
        )
        assert result['all_candidates'] == []


def test_a_longer_transform_cropped_is_the_image_plus_model_surface() -> None:
    """Cropping a longer transform's surface puts every lag where image + model does.

    The model is larger than the image on both axes, so the split between
    positive and negative lags is not at the middle of the surface.
    """
    rng = np.random.default_rng(7)
    image = rng.normal(size=(23, 31))
    model = rng.normal(size=(29, 37))

    def correlate(rows: int, cols: int) -> NDArrayFloatType:
        """Circular correlation of the zero-padded image and model.

        Parameters:
            rows: Transform length on the first axis.
            cols: Transform length on the second axis.

        Returns:
            The correlation surface.
        """
        return _correlate_from_spectra(
            scipy.fft.rfft2(pad_top_left(image, rows, cols)),
            scipy.fft.rfft2(pad_top_left(model, rows, cols)),
            (rows, cols),
        )

    cropped = _crop_circular(correlate(60, 75), image.shape, model.shape)
    np.testing.assert_allclose(cropped, correlate(52, 68), rtol=0.0, atol=1e-12)


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

    def test_a_peak_with_no_sidelobe_is_spurious(self) -> None:
        """A NaN quality is spurious rather than passing the threshold test.

        On a 6 x 6 frame every shift with enough overlap lies within the PSR
        guard radius of zero shift, so the peak has no sidelobe and its PSR is NaN.
        """
        rng = np.random.default_rng(3)
        image = rng.normal(size=(6, 6))
        result = navigate_with_pyramid_kpeaks(
            image,
            image.copy(),
            np.ones((6, 6), dtype=bool),
            pyramid_levels=1,
            upsample_factor=8,
            data_mask=np.ones((6, 6), dtype=bool),
        )
        assert result['spurious'] is True


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
# Wrap-around on the circular correlation surface
# =========================================================================


def test_nms_topk_finds_a_peak_at_zero_shift_once() -> None:
    """A peak at zero shift is one candidate, not one per corner it wraps into."""
    corr = np.full((64, 64), -np.inf)
    # A valid neighborhood of radius 3 around zero shift, which wraps into all
    # four corners of the array.
    rows, cols = np.ogrid[:64, :64]
    dv = np.minimum(rows, 64 - rows)
    du = np.minimum(cols, 64 - cols)
    near = dv**2 + du**2 <= 9
    corr[near] = np.exp(-(dv**2 + du**2) / 4.0)[near]
    peaks = nms_topk(corr, k=5, radius=5)
    assert len(peaks) == 1


def test_psr_at_zero_shift_matches_the_same_peak_mid_array() -> None:
    """A peak at zero shift scores the PSR it scores mid-array.

    The mid-array surface is the same surface rolled, so every value keeps its
    neighbors and only the peak's position relative to the array edges changes.
    """
    rng = np.random.default_rng(930)
    rows, cols = np.ogrid[:64, :64]
    dv = np.minimum(rows, 64 - rows)
    du = np.minimum(cols, 64 - cols)
    corr = 0.05 * rng.standard_normal((64, 64)) + np.exp(-(dv**2 + du**2) / 4.0)
    mid = np.roll(corr, (32, 32), axis=(0, 1))
    assert psr_metric(corr, (0, 0)) == pytest.approx(psr_metric(mid, (32, 32)), rel=1e-12)


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


class TestMaskedNccBidirectional:
    """The data-mask-aware NCC against a direct spatial-domain evaluation."""

    @staticmethod
    def _brute_force_sums(
        image: NDArrayFloatType,
        model: NDArrayFloatType,
        mask: NDArrayBoolType,
        data_mask: NDArrayBoolType,
    ) -> dict[str, NDArrayFloatType]:
        """Compute every shift-wise sum by looping, using no transform.

        The correlation the transforms compute is circular, so the shifts
        wrap; this indexes with the same wrap rather than reproducing the
        transform itself. It is the independent oracle the transform-based
        implementation is checked against.

        Parameters:
            image: Image array, already zero outside ``data_mask``.
            model: Model array.
            mask: Where the model is painted.
            data_mask: Where the image carries data.

        Returns:
            The six sums, each indexed by shift.
        """
        rows, cols = image.shape
        mask_f = mask.astype(np.float64)
        dmask_f = data_mask.astype(np.float64)
        model_mask = model * mask_f
        model2_mask = (model * model) * mask_f
        out: dict[str, NDArrayFloatType] = {
            name: np.zeros((rows, cols), np.float64)
            for name in ('w_d', 'sum_iw', 'sum_i2w', 'sum_imw', 'sum_mw', 'sum_m2w')
        }
        for sv in range(rows):
            for su in range(cols):
                shifted_v = (np.arange(rows)[:, None] - sv) % rows
                shifted_u = (np.arange(cols)[None, :] - su) % cols
                m_s = mask_f[shifted_v, shifted_u]
                mm_s = model_mask[shifted_v, shifted_u]
                m2m_s = model2_mask[shifted_v, shifted_u]
                out['w_d'][sv, su] = float((dmask_f * m_s).sum())
                out['sum_iw'][sv, su] = float((image * m_s).sum())
                out['sum_i2w'][sv, su] = float((image * image * m_s).sum())
                out['sum_imw'][sv, su] = float((image * mm_s).sum())
                out['sum_mw'][sv, su] = float((dmask_f * mm_s).sum())
                out['sum_m2w'][sv, su] = float((dmask_f * m2m_s).sum())
        return out

    @classmethod
    def _brute_force_ncc(
        cls,
        image: NDArrayFloatType,
        model: NDArrayFloatType,
        mask: NDArrayBoolType,
        data_mask: NDArrayBoolType,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Assemble the published NCC from the looped sums.

        Parameters:
            image: Image array, already zero outside ``data_mask``.
            model: Model array.
            mask: Where the model is painted.
            data_mask: Where the image carries data.

        Returns:
            ``(ncc, num)``, matching what the implementation returns.
        """
        s = cls._brute_force_sums(image, model, mask, data_mask)
        w_d = s['w_d']
        w_floor = max(_NCC_BIDIR_W_FRAC_MIN * float(w_d.max()), _NCC_EPS)
        overlap_ok = w_d >= w_floor
        safe_w = np.where(overlap_ok, w_d, w_floor)
        num = s['sum_imw'] - (s['sum_iw'] / safe_w) * s['sum_mw']
        ssd_iw = np.maximum(s['sum_i2w'] - s['sum_iw'] ** 2 / safe_w, 0.0)
        ssd_mw = np.maximum(s['sum_m2w'] - s['sum_mw'] ** 2 / safe_w, 0.0)
        var_ok = (ssd_iw > _NCC_BIDIR_VAR_FRAC_MIN * float(ssd_iw.max())) & (
            ssd_mw > _NCC_BIDIR_VAR_FRAC_MIN * float(ssd_mw.max())
        )
        valid = overlap_ok & var_ok
        ncc = np.where(valid, num / np.maximum(np.sqrt(ssd_iw * ssd_mw), _NCC_EPS), 0.0)
        ncc = np.clip(ncc, -1.0, 1.0)
        ncc[~valid] = -np.inf
        return ncc, num

    def test_transforms_are_few_at_once_and_none_outlive_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The build order and the copied-out real parts are the whole point.

        Each transform output is a complex array of the padded frame, which on
        a wide-margin frame is the largest thing in the process.  Building a
        spectrum before the one it replaces has been dropped costs a whole
        extra copy, and so does keeping an inverse transform alive through a
        view of its real part; nothing about the result changes either way,
        so only a count of what is alive can catch it.
        """
        real_rfft2 = scipy.fft.rfft2
        real_irfft2 = scipy.fft.irfft2
        spectra: list[weakref.ref[Any]] = []
        made = {'forward': 0, 'inverse': 0}
        high = 0

        def counting_rfft2(a: Any, *args: Any, **kwargs: Any) -> Any:
            """Stand in for ``rfft2``, recording the spectrum it returns.

            Parameters:
                a: Array to transform.
                *args: Passed through.
                **kwargs: Passed through.

            Returns:
                The transform scipy would have returned.
            """
            nonlocal high
            made['forward'] += 1
            out = real_rfft2(a, *args, **kwargs)
            spectra.append(weakref.ref(out))
            high = max(high, sum(1 for ref in spectra if ref() is not None))
            return out

        def counting_irfft2(a: Any, *args: Any, **kwargs: Any) -> Any:
            """Stand in for ``irfft2``, counting the surfaces it produces.

            Its output is the surface itself rather than something a real part
            is copied out of, so it is counted but not weighed against the
            spectra: a surface is meant to outlive the transform that made it.

            Parameters:
                a: Spectrum to inverse-transform.
                *args: Passed through.
                **kwargs: Passed through.

            Returns:
                The transform scipy would have returned.
            """
            made['inverse'] += 1
            return real_irfft2(a, *args, **kwargs)

        monkeypatch.setattr('spindoctor.support.correlate.rfft2', counting_rfft2)
        monkeypatch.setattr('spindoctor.support.correlate.irfft2', counting_irfft2)
        rng = np.random.default_rng(11)
        rows, cols = 24, 20
        data_mask = rng.random((rows, cols)) > 0.25
        image = rng.normal(size=(rows, cols)) * data_mask
        model = rng.normal(size=(rows, cols))
        mask = rng.random((rows, cols)) > 0.35
        _masked_ncc_bidir(image, model, mask, data_mask)
        assert made['forward'] == 6
        assert made['inverse'] == 6
        # Three spectra at once: the pair one correlation is taken from, plus
        # the one being built to replace whichever of them it retires.  This
        # counts transform outputs; the conjugate the product is accumulated
        # into is one more half-spectrum again, which no reordering removes.
        assert high <= 3
        # A surface outlives the call -- the numerator is returned -- but no
        # spectrum does.
        assert all(ref() is None for ref in spectra)

    def test_the_scores_match_a_direct_evaluation(self) -> None:
        """Every shift's score is the one a transform-free evaluation gives."""
        rng = np.random.default_rng(4242)
        rows, cols = 10, 12
        data_mask = rng.random((rows, cols)) > 0.25
        image = rng.normal(size=(rows, cols)) * data_mask
        model = rng.normal(size=(rows, cols))
        mask = rng.random((rows, cols)) > 0.35
        ncc, _ = _masked_ncc_bidir(image, model, mask, data_mask)
        expected_ncc, _ = self._brute_force_ncc(image, model, mask, data_mask)
        assert np.allclose(ncc, expected_ncc, rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_the_numerator_matches_a_direct_evaluation(self) -> None:
        """The unnormalized cross term likewise survives the transform."""
        rng = np.random.default_rng(4242)
        rows, cols = 10, 12
        data_mask = rng.random((rows, cols)) > 0.25
        image = rng.normal(size=(rows, cols)) * data_mask
        model = rng.normal(size=(rows, cols))
        mask = rng.random((rows, cols)) > 0.35
        _, num = _masked_ncc_bidir(image, model, mask, data_mask)
        _, expected_num = self._brute_force_ncc(image, model, mask, data_mask)
        assert np.allclose(num, expected_num, rtol=1e-9, atol=1e-9)

    def test_a_perfect_match_is_the_strongest_shift(self) -> None:
        """An unshifted copy of the image scores highest at zero shift."""
        rng = np.random.default_rng(7)
        rows, cols = 16, 16
        data_mask = np.ones((rows, cols), bool)
        image = rng.normal(size=(rows, cols))
        mask = np.zeros((rows, cols), bool)
        mask[4:12, 4:12] = True
        ncc, _ = _masked_ncc_bidir(image, image, mask, data_mask)
        assert int(np.argmax(ncc)) == 0

    def test_every_finite_score_is_a_correlation(self) -> None:
        """No shift reports a normalized score outside [-1, 1]."""
        rng = np.random.default_rng(11)
        rows, cols = 14, 14
        data_mask = rng.random((rows, cols)) > 0.2
        image = rng.normal(size=(rows, cols)) * data_mask
        model = rng.normal(size=(rows, cols))
        mask = rng.random((rows, cols)) > 0.4
        ncc, _ = _masked_ncc_bidir(image, model, mask, data_mask)
        finite = ncc[np.isfinite(ncc)]
        assert finite.size > 0
        assert float(np.max(np.abs(finite))) <= 1.0


# =========================================================================
# Ranking a NaN quality
# =========================================================================


def _stub_quality_sequence(
    monkeypatch: pytest.MonkeyPatch, qualities: list[float]
) -> list[dict[str, Any]]:
    """Make evaluate_candidate return the given qualities, one per call, in order.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture.
        qualities: The quality for each successive candidate.

    Returns:
        The list the stub appends each returned evaluation to.
    """
    returned: list[dict[str, Any]] = []

    def _evaluate(**kwargs: Any) -> dict[str, Any]:
        quality = qualities[len(returned)]
        evaluation = {
            'offset': (float(len(returned)), 0.0),
            'cov': np.eye(2),
            'sigma_xy': (1.0, 1.0),
            'quality': quality,
            'peak_val': 0.5,
            'rc': kwargs['rc'],
        }
        returned.append(evaluation)
        return evaluation

    monkeypatch.setattr(correlate_module, 'evaluate_candidate', _evaluate)
    return returned


class TestNanQualityRanking:
    """A NaN quality ranks below every finite quality wherever candidates are ranked."""

    def _single_scale_with_nan_first(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Run one scale whose first candidate scores NaN and whose second scores 7.0.

        An ``nms_radius`` of 2, below the PSR's 5 px guard, is the setting under which
        a NaN-quality peak can stand beside other candidates.

        Parameters:
            monkeypatch: The pytest monkeypatch fixture, used to stub the candidate
                evaluation.

        Returns:
            The single-scale result.
        """
        rng = np.random.default_rng(943)
        image = gaussian_filter(rng.normal(size=(48, 48)), 1.0)
        evaluations = _stub_quality_sequence(monkeypatch, [float('nan'), 7.0, 3.0])
        result = navigate_single_scale_kpeaks(
            image=image,
            model=image.copy(),
            mask=np.ones(image.shape, dtype=bool),
            max_peaks=3,
            upsample_factor=4,
            nms_radius=2,
            logger=None,
        )
        assert len(evaluations) == 3
        return result

    def test_single_scale_winner_is_the_finite_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NaN candidate evaluated first does not win the scale."""
        result = self._single_scale_with_nan_first(monkeypatch)
        assert result['quality'] == 7.0

    def test_single_scale_lists_the_nan_candidate_last(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The candidate list puts a NaN quality after every finite one."""
        result = self._single_scale_with_nan_first(monkeypatch)
        assert np.isnan(result['all_candidates'][-1]['quality'])

    @staticmethod
    def _auto_mode_choice(
        monkeypatch: pytest.MonkeyPatch, raw_quality: float, grad_quality: float
    ) -> bool:
        """Run auto mode on stubbed raw and gradient results that share a spurious tier.

        Parameters:
            monkeypatch: The pytest monkeypatch fixture, used to stub the two passes.
            raw_quality: The quality of the raw-intensity result.
            grad_quality: The quality of the gradient result.

        Returns:
            Whether auto mode kept the gradient result.
        """
        auto_pyramid = correlate_module.navigate_with_pyramid_kpeaks

        def _mode_result(*_args: Any, use_gradient: bool, **_kwargs: Any) -> dict[str, Any]:
            quality = grad_quality if use_gradient else raw_quality
            return {'quality': quality, 'spurious': True, 'at_edge': False}

        monkeypatch.setattr(correlate_module, 'navigate_with_pyramid_kpeaks', _mode_result)
        model = np.zeros((8, 8))
        result = auto_pyramid(model, model, np.ones((8, 8), dtype=bool), use_gradient='auto')
        return bool(result['used_gradient'])

    def test_auto_mode_tie_break_prefers_a_finite_raw_quality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In the same tier, a finite raw quality beats a NaN gradient quality."""
        assert self._auto_mode_choice(monkeypatch, 3.0, float('nan')) is False

    def test_auto_mode_tie_break_keeps_gradient_when_both_are_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In the same tier, two NaN qualities keep the gradient result."""
        assert self._auto_mode_choice(monkeypatch, float('nan'), float('nan')) is True

    def test_peak_to_runner_up_ratio_skips_a_nan_runner_up(self) -> None:
        """A NaN runner-up is skipped in favor of the next finite one."""
        ratio = peak_to_runner_up_ratio(
            [(8.0, 0.0, 0.0), (float('nan'), 1.0, 1.0), (2.0, 2.0, 2.0)]
        )
        assert ratio == pytest.approx(4.0)

    def test_peak_to_runner_up_ratio_nan_runner_up_alone_is_a_lone_peak(self) -> None:
        """A NaN runner-up with no finite one after it gives the lone-peak ratio."""
        ratio = peak_to_runner_up_ratio([(8.0, 0.0, 0.0), (float('nan'), 1.0, 1.0)])
        assert ratio == 1.0
