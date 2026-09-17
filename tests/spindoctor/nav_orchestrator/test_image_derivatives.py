"""Unit tests for ``spindoctor.nav_orchestrator.image_derivatives``."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.nav_orchestrator.image_derivatives import (
    DEFAULT_DT_HALF_WIDTH_PX,
    DEFAULT_EDGE_THRESHOLD_K_SIGMA,
    DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
    ImageDerivativesConfig,
    build_image_edge_dt,
    compute_all_image_derivatives,
    compute_image_gradient_vu,
)


def _step_image(shape: tuple[int, int], step_v: int) -> np.ndarray:
    """Return an image with a single horizontal bright bar centered on ``step_v``.

    The bar is 8 pixels tall so the borders of the image stay in the same
    background as their interior neighbors, suppressing the
    sobel-with-constant-padding boundary artifact that would otherwise
    dominate the gradient image on a small test fixture.
    """
    img = np.zeros(shape, dtype=np.float64)
    img[step_v - 4 : step_v + 4, :] = 100.0
    return img


# A bar whose two edges sit at stated sub-pixel rows, one below its pixel's
# center and one above, so a half pixel added to or subtracted from the whole
# fixture moves one of the two zero rows whichever way it goes.
_BAR_SHAPE = (48, 48)
_BAR_LEADING_V = 12.25
_BAR_TRAILING_V = 28.75
_BAR_COLUMN = 24

# The gradient ridge reproduces each planted edge to within 1e-4 px here; the
# residue is the far edge's gaussian tail leaking into the profile window,
# which the wide bar keeps small.  The bound is two orders of magnitude below
# the half pixel these tests exist to resolve.
_RIDGE_TOLERANCE_PX = 0.005


def _subpixel_bar_image(leading_v: float, trailing_v: float) -> np.ndarray:
    """Return a bright bar whose edges lie at stated pixel-centric rows.

    Row ``i`` covers ``[i - 0.5, i + 0.5]`` in pixel-centric coordinates, so a
    row holds the fraction of its own extent that the bar covers.  The bar is
    dark at both ends of the v axis, keeping the sobel-with-constant-padding
    boundary artifact out of the frame, and is uniform along u.

    Parameters:
        leading_v: Pixel-centric row of the bar's low-v boundary.
        trailing_v: Pixel-centric row of the bar's high-v boundary.
    """
    v = np.arange(_BAR_SHAPE[0], dtype=np.float64)
    covered = np.clip(np.minimum(v + 0.5, trailing_v) - np.maximum(v - 0.5, leading_v), 0.0, 1.0)
    img: np.ndarray = np.repeat(covered[:, np.newaxis], _BAR_SHAPE[1], axis=1) * 100.0
    return img


def _ridge_centroid(gradient: np.ndarray, near_v: float) -> float:
    """Return the gradient ridge's pixel-centric row near a stated position.

    The magnitude profile down :data:`_BAR_COLUMN` is symmetric about the edge
    that produced it, so its weighted mean over a window of plus or minus five
    rows measures where that edge is without restating how the edge was drawn.

    Parameters:
        gradient: Gradient magnitude image.
        near_v: Pixel-centric row the window is centered on.
    """
    lo = round(near_v) - 5
    hi = round(near_v) + 6
    weights = gradient[lo:hi, _BAR_COLUMN]
    return float((np.arange(lo, hi, dtype=np.float64) * weights).sum() / weights.sum())


def test_image_derivatives_config_defaults_match_design() -> None:
    """Default ``ImageDerivativesConfig`` carries the documented constants."""
    cfg = ImageDerivativesConfig()
    assert cfg.image_gradient_sigma_px == DEFAULT_IMAGE_GRADIENT_SIGMA_PX
    assert cfg.edge_threshold_k_sigma == DEFAULT_EDGE_THRESHOLD_K_SIGMA
    assert cfg.dt_half_width_px == DEFAULT_DT_HALF_WIDTH_PX


def test_image_derivatives_config_rejects_zero_sigma() -> None:
    """A zero ``image_gradient_sigma_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_gradient_sigma_px'):
        ImageDerivativesConfig(image_gradient_sigma_px=0.0)


def test_image_derivatives_config_rejects_zero_threshold() -> None:
    """A zero ``edge_threshold_k_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='edge_threshold_k_sigma'):
        ImageDerivativesConfig(edge_threshold_k_sigma=0.0)


def test_image_derivatives_config_rejects_zero_half_width() -> None:
    """A zero ``dt_half_width_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='dt_half_width_px'):
        ImageDerivativesConfig(dt_half_width_px=0.0)


def test_build_image_edge_dt_returns_arrays_shaped_like_the_image() -> None:
    """Both products come back on the grid of the image they were built from."""
    img = _step_image((40, 40), step_v=16)
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    assert gradient.shape == (40, 40)
    assert edge_dt.shape == (40, 40)


def test_gradient_ridge_centers_on_the_planted_edge() -> None:
    """The smoothed gradient peaks where the bar's boundary was drawn.

    The bar states each boundary as a pixel-centric row and leaves the rows it
    crosses partly covered, so the ridge the gaussian and sobel pass produces
    is symmetric about that row and nothing about it is a restatement of the
    operator.  An operator biased by half a pixel, such as a forward difference
    in place of the centered one, moves the ridge off the number the fixture
    stated and fails here.
    """
    img = _subpixel_bar_image(_BAR_LEADING_V, _BAR_TRAILING_V)
    gradient, _edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    leading = _ridge_centroid(gradient, _BAR_LEADING_V)
    trailing = _ridge_centroid(gradient, _BAR_TRAILING_V)
    assert leading == pytest.approx(_BAR_LEADING_V, abs=_RIDGE_TOLERANCE_PX)
    assert trailing == pytest.approx(_BAR_TRAILING_V, abs=_RIDGE_TOLERANCE_PX)


def test_edge_dt_zero_locus_is_the_pixel_whose_center_is_nearest_the_edge() -> None:
    """The distance transform reads zero on one row per edge, at that row's center.

    Every DT-based residual in the pipeline is measured against this zero
    locus, so where it sits relative to the pixel grid is the reference the
    limb, terminator and ring-edge fits all inherit.  A bar boundary at pixel
    centric ``12.25`` puts it on row 12 and one at ``28.75`` on row 29: the
    pixel whose own center is nearest the boundary, not the pixel edge the
    boundary runs along.  Reading the boundary half a pixel high would move the
    first row to 13, half a pixel low would move the second to 28, so the pair
    fails whichever way a half pixel goes astray.
    """
    img = _subpixel_bar_image(_BAR_LEADING_V, _BAR_TRAILING_V)
    _gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    zero_rows = np.flatnonzero(edge_dt[:, _BAR_COLUMN] == 0.0)
    assert zero_rows.tolist() == [12, 29]


def test_build_image_edge_dt_zeros_edge_dt_on_thresholded_pixels() -> None:
    """Distance transform is exactly zero on retained edge pixels and positive elsewhere."""
    img = _step_image((40, 40), step_v=16)
    _, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    # Pixels along the leading edge (row 12) get DT = 0 exactly --
    # ``distance_transform_edt`` returns integer-zero on edge pixels of a
    # binary mask, so the assertion can be exact rather than approximate.
    assert float(edge_dt[12, 20]) == 0.0
    # A pixel far from any edge has positive DT.
    assert float(edge_dt[0, 0]) > 0.0


def test_build_image_edge_dt_falls_back_when_no_pixel_exceeds_threshold() -> None:
    """Empty edge mask saturates the DT at the configured half-width everywhere."""
    img = np.full((16, 16), 1.0, dtype=np.float64)
    cfg = ImageDerivativesConfig(
        edge_threshold_k_sigma=1000.0,
        dt_half_width_px=5.0,
    )
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=10.0, config=cfg)
    # No edge pixels survive the very high threshold; DT saturates at the
    # half width even though the gradient itself has small boundary
    # artifacts from the constant-padded Sobel.
    threshold = cfg.edge_threshold_k_sigma * 10.0
    assert (gradient <= threshold).all()
    assert np.allclose(edge_dt, cfg.dt_half_width_px, atol=1e-12)


def test_build_image_edge_dt_threshold_sweep_matches_expected_count() -> None:
    """Edge-pixel count is monotonically non-decreasing as the k-sigma threshold drops."""
    img = _step_image((40, 40), step_v=16)
    # Decreasing the k_sigma factor lets more pixels pass the threshold;
    # binarized edge counts must be monotonically non-decreasing.
    counts: list[int] = []
    for k in (8.0, 4.0, 2.0):
        cfg = ImageDerivativesConfig(edge_threshold_k_sigma=k)
        gradient, _ = build_image_edge_dt(img, image_noise_sigma=1.0, config=cfg)
        threshold = k * 1.0
        edge_count = int((gradient > threshold).sum())
        counts.append(edge_count)
    assert counts[0] <= counts[1]
    assert counts[1] <= counts[2]


def test_build_image_edge_dt_rejects_non_2d_input() -> None:
    """A non-2-D ``image_ext`` is rejected with a TypeError naming the field."""
    with pytest.raises(TypeError, match='image_ext must be 2-D'):
        build_image_edge_dt(np.zeros((4, 4, 4)), image_noise_sigma=1.0)


def test_build_image_edge_dt_rejects_negative_noise_sigma() -> None:
    """A negative ``image_noise_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_noise_sigma'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=-1.0)


def test_build_image_edge_dt_rejects_nan_noise_sigma() -> None:
    """A NaN ``image_noise_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_noise_sigma'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=float('nan'))


def test_compute_image_gradient_vu_horizontal_step_points_along_v() -> None:
    """Horizontal step edge produces a positive ``g_v`` and near-zero ``g_u``."""
    shape = (40, 40)
    img = _step_image(shape, step_v=16)
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert grad.shape == (40, 40, 2)
    # The leading edge of the bar (image rises from 0 to 100 with
    # increasing row index) sits near row 12: g_v is strictly positive
    # there because Sobel returns a positive derivative when image values
    # increase with row.
    assert float(grad[12, 20, 0]) > 0.0
    # The orthogonal u-axis gradient is essentially zero on a horizontal
    # bar.
    assert abs(float(grad[12, 20, 1])) < 1.0


def test_compute_image_gradient_vu_vertical_step_points_along_u() -> None:
    """Vertical step edge produces a positive ``g_u`` and near-zero ``g_v``."""
    img = np.zeros((32, 32), dtype=np.float64)
    img[:, 16:] = 100.0
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert float(grad[16, 16, 1]) > 0.0
    assert abs(float(grad[16, 16, 0])) < 1.0


def test_compute_image_gradient_vu_rejects_non_2d_input() -> None:
    """A non-2-D ``image_ext`` is rejected with a TypeError naming the field."""
    with pytest.raises(TypeError, match='image_ext must be 2-D'):
        compute_image_gradient_vu(np.zeros((4, 4, 2)))


def test_compute_image_gradient_vu_rejects_zero_sigma() -> None:
    """A zero ``sigma_px`` is rejected with a finite-positive-number message."""
    with pytest.raises(ValueError, match='sigma_px must be a finite positive number'):
        compute_image_gradient_vu(np.zeros((4, 4)), sigma_px=0.0)


def test_compute_image_gradient_vu_rejects_inf_sigma() -> None:
    """An infinite ``sigma_px`` is rejected with a finite-positive-number message."""
    with pytest.raises(ValueError, match='sigma_px must be a finite positive number'):
        compute_image_gradient_vu(np.zeros((4, 4)), sigma_px=float('inf'))


def test_build_image_edge_dt_rejects_inf_noise_sigma() -> None:
    """An infinite ``image_noise_sigma`` is rejected with a finite-required message."""
    with pytest.raises(ValueError, match='image_noise_sigma must be finite'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=float('inf'))


def test_image_derivatives_config_rejects_inf_sigma() -> None:
    """An infinite ``image_gradient_sigma_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_gradient_sigma_px'):
        ImageDerivativesConfig(image_gradient_sigma_px=float('inf'))


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


def test_build_image_edge_dt_handles_minimal_4x4_image() -> None:
    """Smallest meaningful image: 4x4 still produces correctly-shaped outputs.

    The Gaussian smooth, Sobel, NMS, and DT pipeline must all tolerate
    a 4x4 input without raising or producing degenerate-shape arrays.
    Sobel-with-constant-padding boundary artifacts may produce a few
    edge pixels at the corners; what we verify is that the pipeline
    completes and returns 2-D float64 arrays of the right shape with
    finite values throughout.
    """
    img = np.full((4, 4), 5.0, dtype=np.float64)
    cfg = ImageDerivativesConfig(dt_half_width_px=3.0)
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0, config=cfg)
    assert gradient.shape == (4, 4)
    assert edge_dt.shape == (4, 4)
    assert np.isfinite(gradient).all()
    assert np.isfinite(edge_dt).all()
    # DT entries are bounded by the configured half-width.
    assert float(edge_dt.max()) <= cfg.dt_half_width_px


def test_compute_image_gradient_vu_handles_minimal_4x4_image() -> None:
    """Gradient-vector helper produces a (4, 4, 2) output on a 4x4 input."""
    img = np.full((4, 4), 5.0, dtype=np.float64)
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert grad.shape == (4, 4, 2)


def test_build_image_edge_dt_threshold_boundary_exact() -> None:
    """A pixel whose gradient magnitude equals the threshold is excluded.

    The implementation uses a strict ``gradient > threshold`` comparison
    inside ``_directional_nms``; this test plants a gradient profile
    chosen so that exactly one pixel sits at the threshold value and
    verifies it is *not* kept in the edge mask.  The next-larger value
    is kept, confirming the "at boundary -> excluded" half-open
    convention.
    """
    img = _step_image((40, 40), step_v=16)
    # Probe a noise sigma that makes the gradient peak exactly on the
    # k_sigma boundary.  Compute the actual peak first to derive the
    # noise sigma that makes it fall right at the threshold.
    cfg = ImageDerivativesConfig(edge_threshold_k_sigma=4.0)
    gradient_only, _ = build_image_edge_dt(img, image_noise_sigma=1.0e-6, config=cfg)
    peak = float(gradient_only.max())
    # Set noise sigma so threshold == peak exactly: peak == 4 * sigma.
    boundary_sigma = peak / cfg.edge_threshold_k_sigma
    _, edge_dt_boundary = build_image_edge_dt(img, image_noise_sigma=boundary_sigma, config=cfg)
    # No pixel exceeds the threshold (strict ``>`` rejects the equality
    # case), so the DT saturates everywhere at the half-width.
    assert np.allclose(edge_dt_boundary, cfg.dt_half_width_px)
    # A noise sigma fractionally below the boundary lets the peak
    # through and produces at least one zero-DT pixel.
    _, edge_dt_below = build_image_edge_dt(img, image_noise_sigma=boundary_sigma * 0.99, config=cfg)
    assert float(edge_dt_below.min()) == 0.0


# ---------------------------------------------------------------------------
# Combined entry point — orchestrator-friendly single-pass derivative bundle
# ---------------------------------------------------------------------------


def test_compute_all_image_derivatives_matches_separate_calls() -> None:
    """Combined entry point produces the same arrays as the two split calls.

    The orchestrator switched from calling ``build_image_edge_dt`` and
    ``compute_image_gradient_vu`` separately to using the combined entry
    point.  Both paths must return byte-identical results because they
    share the same underlying gaussian + sobel pipeline.
    """
    shape = (40, 40)
    img = _step_image(shape, step_v=16)
    cfg = ImageDerivativesConfig()
    gradient_split, edge_dt_split = build_image_edge_dt(img, image_noise_sigma=1.0, config=cfg)
    gradient_vu_split = compute_image_gradient_vu(img, sigma_px=cfg.image_gradient_sigma_px)
    gradient_combo, edge_dt_combo, gradient_vu_combo = compute_all_image_derivatives(
        img, image_noise_sigma=1.0, config=cfg
    )
    np.testing.assert_array_equal(gradient_combo, gradient_split)
    np.testing.assert_array_equal(edge_dt_combo, edge_dt_split)
    np.testing.assert_array_equal(gradient_vu_combo, gradient_vu_split)


def test_compute_all_image_derivatives_rejects_non_2d_input() -> None:
    """A non-2-D ``image_ext`` is rejected with a TypeError naming the field."""
    with pytest.raises(TypeError, match='image_ext must be 2-D'):
        compute_all_image_derivatives(np.zeros((4, 4, 4)), image_noise_sigma=1.0)


def test_compute_all_image_derivatives_rejects_nan_image() -> None:
    """A non-finite ``image_ext`` is rejected before any heavy work runs."""
    img = np.zeros((8, 8), dtype=np.float64)
    img[3, 3] = np.nan
    with pytest.raises(ValueError, match='image_ext must contain only finite values'):
        compute_all_image_derivatives(img, image_noise_sigma=1.0)


def test_compute_all_image_derivatives_rejects_negative_noise_sigma() -> None:
    """A negative ``image_noise_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_noise_sigma'):
        compute_all_image_derivatives(np.zeros((4, 4)), image_noise_sigma=-1.0)


def test_compute_all_image_derivatives_uses_default_config_when_none() -> None:
    """``config=None`` resolves to the documented defaults."""
    img = _step_image((40, 40), step_v=16)
    gradient_default, edge_dt_default, gradient_vu_default = compute_all_image_derivatives(
        img, image_noise_sigma=1.0
    )
    gradient_explicit, edge_dt_explicit, gradient_vu_explicit = compute_all_image_derivatives(
        img, image_noise_sigma=1.0, config=ImageDerivativesConfig()
    )
    np.testing.assert_array_equal(gradient_default, gradient_explicit)
    np.testing.assert_array_equal(edge_dt_default, edge_dt_explicit)
    np.testing.assert_array_equal(gradient_vu_default, gradient_vu_explicit)
