"""End-to-end tests for ``StarRefineNav``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.diagnostics import StarRefineDiagnostics
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
)
from spindoctor.nav_technique.nav_technique_star_refine import StarRefineNav


def _attach_prior(context: NavContext, *, prior_offset_px: tuple[float, float]) -> NavContext:
    """Return ``context`` with the supplied pass-1 prior attached."""
    return context.with_prior(
        offset_px=prior_offset_px,
        covariance_px2=np.eye(2, dtype=np.float64),
    )


def test_star_refine_recovers_residual_correction(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A small residual after the prior is recovered as a delta."""
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 70.0)
    actual_b = (160.0, 130.0)
    actual_c = (110.0, 170.0)
    for c in (actual_a, actual_b, actual_c):
        draw_gaussian_star(image, c, peak_dn=200.0, sigma=1.2)
    # Pass-1 prior is 1 px off the true offset along V.
    true_offset = (3.0, -2.0)
    prior = (true_offset[0] - 1.0, true_offset[1])
    pred_a = (actual_a[0] - true_offset[0], actual_a[1] - true_offset[1])
    pred_b = (actual_b[0] - true_offset[0], actual_b[1] - true_offset[1])
    pred_c = (actual_c[0] - true_offset[0], actual_c[1] - true_offset[1])
    features = [
        make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=30.0),
        make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=30.0),
        make_star_feature('star:UCAC4:C', predicted_vu=pred_c, predicted_snr=30.0),
    ]
    technique = StarRefineNav()
    context = make_nav_context(image)
    context = _attach_prior(context, prior_offset_px=prior)
    feasibility = technique.is_feasible(features)
    assert feasibility.feasible is True
    result = technique.navigate(features, context)
    assert result.spurious is False
    # The technique reports the absolute offset (delta + prior); should
    # converge to the true planted value to sub-pixel.
    assert result.offset_px[0] == pytest.approx(true_offset[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(true_offset[1], abs=0.5)
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    assert result.diagnostics.n_stars_used == 3


def test_star_refine_drops_outlier_star(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star whose detection sits far from the shifted prediction is dropped."""
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 70.0)
    actual_b = (160.0, 130.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=200.0, sigma=1.2)
    # A wild prediction that would land far from any peak in the image.
    actual_c_predicted = (50.0, 200.0)
    draw_gaussian_star(image, (50.0, 60.0), peak_dn=200.0, sigma=1.2)
    true_offset = (1.0, 1.0)
    prior = (true_offset[0], true_offset[1])
    pred_a = (actual_a[0] - true_offset[0], actual_a[1] - true_offset[1])
    pred_b = (actual_b[0] - true_offset[0], actual_b[1] - true_offset[1])
    features = [
        make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=30.0),
        make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=30.0),
        make_star_feature(
            'star:UCAC4:C',
            predicted_vu=actual_c_predicted,
            predicted_snr=30.0,
        ),
    ]
    technique = StarRefineNav()
    context = make_nav_context(image)
    context = _attach_prior(context, prior_offset_px=prior)
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    # A and B contribute; the wild C prediction is dropped because no
    # bright peak sits inside its refine window.
    assert result.diagnostics.n_stars_used == 2


def test_star_refine_requires_prior(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """``StarRefineNav.requires_prior`` is True; navigate without prior fails."""
    assert StarRefineNav.requires_prior is True
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    draw_gaussian_star(image, (100.0, 100.0), peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=30.0)
    technique = StarRefineNav()
    context = make_nav_context(image)
    # Don't attach a prior.
    result = technique.navigate([feature], context)
    assert result.spurious is True
    assert result.confidence == 0.0


def test_star_refine_caps_single_inlier_confidence(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A 1-inlier refine on a near-perfect prior is capped at the configured ceiling.

    Pin the contract by reading the cap value from
    ``StarRefineNav.tuning`` rather than hard-coding 0.5.  A 1-inlier
    refine carries no independent cross-check, so its confidence must
    not exceed ``single_inlier_confidence_cap`` even when the
    underlying sigmoid would give a higher number.
    """
    cap = float(StarRefineNav.tuning['single_inlier_confidence_cap'])
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual = (110.0, 130.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    true_offset = (1.0, 1.0)
    pred = (actual[0] - true_offset[0], actual[1] - true_offset[1])
    feature = make_star_feature('star:UCAC4:lone', predicted_vu=pred, predicted_snr=40.0)
    technique = StarRefineNav()
    context = make_nav_context(image)
    context = _attach_prior(context, prior_offset_px=true_offset)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    assert result.diagnostics.n_stars_used == 1
    # The cap is a hard ceiling on the post-sigmoid confidence.
    assert result.confidence <= cap + 1e-12


def test_star_refine_skips_single_inlier_cap_log_for_multi_inlier(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A multi-inlier refine never logs the single-inlier-cap line.

    The structural cap only applies when ``n_stars_used == 1``; with
    two or more inliers the per-star residual scatter cross-checks the
    joint fit and the cap path is skipped.  Asserting on the log
    output confirms the cap branch did not fire.
    """
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 70.0)
    actual_b = (160.0, 130.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=200.0, sigma=1.2)
    true_offset = (1.0, 1.0)
    pred_a = (actual_a[0] - true_offset[0], actual_a[1] - true_offset[1])
    pred_b = (actual_b[0] - true_offset[0], actual_b[1] - true_offset[1])
    features = [
        make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=40.0),
        make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0),
    ]
    technique = StarRefineNav()
    context = make_nav_context(image)
    context = _attach_prior(context, prior_offset_px=true_offset)
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    assert result.diagnostics.n_stars_used == 2
    captured = capsys.readouterr()
    assert 'Single-inlier refine' not in captured.out


def test_star_refine_rejects_invalid_single_inlier_cap() -> None:
    """Construction rejects ``single_inlier_confidence_cap`` outside [0, 1]."""
    technique_class = StarRefineNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['single_inlier_confidence_cap'] = 1.5
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='single_inlier_confidence_cap'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


def test_star_refine_infeasible_on_empty_input() -> None:
    """``is_feasible([])`` reports infeasibility."""
    technique = StarRefineNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_usable_star_features' in report.reason


def test_star_refine_registered_with_navtechnique_registry() -> None:
    """``StarRefineNav`` self-registers on import."""
    assert StarRefineNav in NavTechnique._registry


def test_star_refine_3dof_recovers_planted_rotation_with_two_inliers(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two refined inliers under fit_camera_rotation recover a small rotation."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=180.0, sigma=1.2)
    prior = (0.5, -0.5)
    planted_theta = math.radians(0.6)
    pivot = (
        0.5 * (actual_a[0] + actual_b[0]) - prior[0],
        0.5 * (actual_a[1] + actual_b[1]) - prior[1],
    )
    cos_t = math.cos(-planted_theta)
    sin_t = math.sin(-planted_theta)

    def _rotate_back(p: tuple[float, float]) -> tuple[float, float]:
        rv = p[0] - pivot[0]
        ru = p[1] - pivot[1]
        return (
            pivot[0] + cos_t * rv - sin_t * ru,
            pivot[1] + sin_t * rv + cos_t * ru,
        )

    pred_a = _rotate_back((actual_a[0] - prior[0], actual_a[1] - prior[1]))
    pred_b = _rotate_back((actual_b[0] - prior[0], actual_b[1] - prior[1]))
    features = [
        make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=40.0),
        make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0),
    ]
    technique = StarRefineNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    context = _attach_prior(context, prior_offset_px=prior)
    result = technique.navigate(features, context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(planted_theta, abs=math.radians(0.3))
    assert result.sigma_rotation_rad is not None


def test_star_refine_3dof_single_inlier_unobservable(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single-inlier refine under fit_camera_rotation reports rotation as unobservable."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual = (100.0, 100.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    prior = (0.4, -0.3)
    pred = (actual[0] - prior[0], actual[1] - prior[1])
    feature = make_star_feature('star:UCAC4:A', predicted_vu=pred, predicted_snr=40.0)
    technique = StarRefineNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    context = _attach_prior(context, prior_offset_px=prior)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad == 0.0
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)


def test_star_refine_measures_a_star_past_a_brighter_spike_in_its_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single pixel outshining a star's peak does not take its refine window.

    The spike sits 5.8 px from star A's shifted prediction, inside the 6 px
    window and past the 4 px per-star residual limit, so a refinement that
    measured it would drop star A.
    """
    image = np.zeros((220, 220), dtype=np.float64)
    actual = [(60.0, 70.0), (160.0, 130.0), (110.0, 170.0)]
    for center in actual:
        draw_gaussian_star(image, center, peak_dn=15.0, sigma=1.2)
    image[55, 73] += 27.0
    true_offset = (3.0, -2.0)
    features = [
        make_star_feature(
            f'star:UCAC4:{i}',
            predicted_vu=(v - true_offset[0], u - true_offset[1]),
            predicted_snr=30.0,
        )
        for i, (v, u) in enumerate(actual)
    ]
    context = _attach_prior(make_nav_context(image), prior_offset_px=true_offset)
    result = StarRefineNav().navigate(features, context)
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    assert result.diagnostics.n_stars_used == 3


def test_star_refine_drops_a_star_whose_window_is_outshone_from_outside(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star that left its window is not replaced by a faint bump inside it.

    Star D lies 8 px from its shifted prediction, outside the 6 px window, and its
    flank outshines a faint source 2 px from the prediction, which a refinement that
    took the window's best peak would measure as D.
    """
    image = np.zeros((260, 260), dtype=np.float64)
    true_offset = (3.0, -2.0)
    actual = [(60.0, 70.0), (160.0, 130.0), (110.0, 170.0)]
    for center in actual:
        draw_gaussian_star(image, center, peak_dn=200.0, sigma=1.2)
    pred_d = (200.0, 60.0)
    shifted_d = (pred_d[0] + true_offset[0], pred_d[1] + true_offset[1])
    draw_gaussian_star(image, (shifted_d[0], shifted_d[1] + 8.0), peak_dn=1000.0, sigma=1.2)
    draw_gaussian_star(image, (shifted_d[0], shifted_d[1] - 2.0), peak_dn=20.0, sigma=1.0)
    preds = [(v - true_offset[0], u - true_offset[1]) for v, u in actual] + [pred_d]
    features = [
        make_star_feature(f'star:UCAC4:{i}', predicted_vu=pred, predicted_snr=30.0)
        for i, pred in enumerate(preds)
    ]
    context = _attach_prior(make_nav_context(image), prior_offset_px=true_offset)
    result = StarRefineNav().navigate(features, context)
    assert isinstance(result.diagnostics, StarRefineDiagnostics)
    assert result.diagnostics.n_stars_used == 3
