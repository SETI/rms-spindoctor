"""End-to-end tests for ``StarUniqueMatchNav``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique.diagnostics import StarUniqueMatchDiagnostics
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
)
from spindoctor.nav_technique.nav_technique_star_unique_match import StarUniqueMatchNav


def test_star_unique_match_one_star_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single uniquely-bright star recovers the planted translation."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (100.0, 100.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    planted = (2.0, -3.0)
    pred_vu = (actual_vu[0] - planted[0], actual_vu[1] - planted[1])
    feature = make_star_feature('star:UCAC4:1', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    feas = technique.is_feasible([feature])
    assert feas.feasible is True
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'one_star'
    assert result.confidence <= 0.7 + 1e-12


def test_the_matched_star_is_logged_in_the_image_frame(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The logged match names a pixel of the image, not of the padded array.

    The star is drawn on the center of row 100 and column 100 of the padded
    array, which sits 32 rows and columns of padding in from the image's own
    first row and column, so its light is on the center of image row 68.  A
    position stated to a person names a row's center by that row's number
    plus a half, so the line has to read 68.5000.  The offset beside it is a
    difference and stays as it is.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    drawn_vu = (100.0, 100.0)
    draw_gaussian_star(image, drawn_vu, peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:1', predicted_vu=drawn_vu, predicted_snr=40.0)
    StarUniqueMatchNav().navigate([feature], make_nav_context(image, extfov_margin_vu=(32, 32)))
    assert 'matched star:UCAC4:1 at (68.5000, 68.5000)' in capsys.readouterr().out


def test_star_unique_match_two_star_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The 2-star path picks the smaller-residual assignment."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=180.0, sigma=1.2)
    planted = (1.0, 1.5)
    pred_a = (actual_a[0] - planted[0], actual_a[1] - planted[1])
    pred_b = (actual_b[0] - planted[0], actual_b[1] - planted[1])
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.4)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'two_star'
    # No third predictable star, so the brightness margin diagnostic is +inf.
    assert result.diagnostics.brightness_margin_mag == float('inf')
    assert result.confidence <= 0.8 + 1e-12


def test_star_unique_match_one_star_fails_when_brightness_margin_too_small(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two near-equal-brightness stars fail the 1-star uniqueness test.

    With two stars whose predicted SNRs are close and the 2-star path
    fails (e.g. wrong-position prediction such that one detection misses
    its window entirely), the technique reports a spurious result with
    a brightness-margin reason rather than picking the wrong star.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual = (100.0, 100.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    planted = (1.0, 1.0)
    pred_a = (actual[0] - planted[0], actual[1] - planted[1])
    # Second prediction in a region with no signal so its detection
    # fails — forcing the one-star fallback.
    pred_b = (10.0, 10.0)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=20.0)
    feat_b = make_star_feature(
        'star:UCAC4:B',
        predicted_vu=pred_b,
        predicted_snr=18.0,  # too close to A — no margin
    )
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feat_a, feat_b], context)
    # Cannot trigger the 2-star path because pred_b has no detection.
    # The 1-star fallback rejects because brightness margin to feat_b
    # is below the 1.5 mag floor.
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)


def test_star_unique_match_infeasible_on_empty_input() -> None:
    """``is_feasible([])`` reports infeasibility."""
    technique = StarUniqueMatchNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_usable_star_features' in report.reason


def test_star_unique_match_skips_occluded_stars(
    make_star_feature: NavFeatureFactory,
) -> None:
    """STAR features marked in_body_silhouette are filtered out."""
    feature = make_star_feature(
        'star:UCAC4:occluded',
        predicted_vu=(100.0, 100.0),
        predicted_snr=30.0,
        in_body_silhouette=True,
    )
    technique = StarUniqueMatchNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False
    assert 'no_usable_star_features' in report.reason


def test_star_unique_match_registered_with_navtechnique_registry() -> None:
    """``StarUniqueMatchNav`` self-registers on import."""
    assert StarUniqueMatchNav in NavTechnique._registry


def test_star_unique_match_marks_at_edge_when_offset_hits_margin(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """An offset that hits the search-window edge is flagged + zero confidence."""
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (110.0, 110.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    margin = (8, 8)
    # Plant the offset exactly on the search-window edge.
    pred_vu = (actual_vu[0] - float(margin[0]), actual_vu[1])
    feature = make_star_feature('star:UCAC4:edge', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, extfov_margin_vu=margin)
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_star_unique_match_two_star_3dof_recovers_planted_rotation(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The 2-star Procrustes path recovers a planted rotation around the catalog midpoint.

    Translation is reported in the catalog frame: the test plants a 1
    deg rotation between the catalog cohort and the detection cohort,
    which the 2-star similarity-transform fit must recover within the
    expected sub-degree tolerance.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=180.0, sigma=1.2)
    # Choose catalog predictions such that rotating them by the planted
    # angle about their midpoint and applying a tiny translation lands
    # on the detected positions.  Pivot is the midpoint of the catalog
    # predictions.
    planted_offset = (0.5, 0.5)
    planted_theta = math.radians(1.0)
    cat_a = (actual_a[0] - planted_offset[0], actual_a[1] - planted_offset[1])
    cat_b = (actual_b[0] - planted_offset[0], actual_b[1] - planted_offset[1])
    pivot = (0.5 * (cat_a[0] + cat_b[0]), 0.5 * (cat_a[1] + cat_b[1]))
    cos_t = math.cos(-planted_theta)
    sin_t = math.sin(-planted_theta)

    def _rotate(p: tuple[float, float]) -> tuple[float, float]:
        rv = p[0] - pivot[0]
        ru = p[1] - pivot[1]
        return (
            pivot[0] + cos_t * rv - sin_t * ru,
            pivot[1] + sin_t * rv + cos_t * ru,
        )

    pred_a = _rotate(cat_a)
    pred_b = _rotate(cat_b)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(planted_theta, abs=math.radians(0.3))
    # The Procrustes translation is ``t = det_centroid - R @ cat_centroid``,
    # which equals ``planted_offset + (pivot - R @ pivot)`` for the test
    # setup.  When the pivot sits far from the image origin (here at
    # ~ (100, 100)) that pivot-rotation correction dominates the raw
    # ``planted_offset`` term, so the assertion compares against the
    # full expected Procrustes output rather than the planted shift.
    cos_t = math.cos(planted_theta)
    sin_t = math.sin(planted_theta)
    rotated_pivot_v = cos_t * pivot[0] - sin_t * pivot[1]
    rotated_pivot_u = sin_t * pivot[0] + cos_t * pivot[1]
    expected_offset_v = planted_offset[0] + (pivot[0] - rotated_pivot_v)
    expected_offset_u = planted_offset[1] + (pivot[1] - rotated_pivot_u)
    assert result.offset_px[0] == pytest.approx(expected_offset_v, abs=0.4)
    assert result.offset_px[1] == pytest.approx(expected_offset_u, abs=0.4)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'two_star'


def test_star_unique_match_one_star_3dof_rotation_unobservable(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A 1-star path with fit_camera_rotation reports rotation as unobservable."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual = (100.0, 100.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    feature = make_star_feature(
        'star:UCAC4:bright',
        predicted_vu=(99.0, 101.0),
        predicted_snr=60.0,
    )
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad == 0.0
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)


def test_star_unique_match_one_star_flags_ambiguous_detection(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two comparable faint peaks in the window make the match spurious.

    The single-detection premise of the 1-star path fails when the
    brightest peak barely beats its runner-up (a marginal star losing
    to noise spikes); the peak-to-runner-up gate must flag
    the result rather than hand the ensemble a confident wrong offset.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    # Both peaks clear detection_sigma (4 * noise_sigma 1.0) but sit
    # within ~10% of each other -- an ambiguous detection.
    draw_gaussian_star(image, (95.0, 105.0), peak_dn=6.0, sigma=1.2)
    draw_gaussian_star(image, (110.0, 90.0), peak_dn=5.5, sigma=1.2)
    feature = make_star_feature('star:UCAC4:1', predicted_vu=(100.0, 100.0), predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.detection_peak_ratio < 1.5
    assert result.diagnostics.detection_peak_ratio > 0.0


def test_star_unique_match_one_star_rejects_far_no_rival_detection(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """No-rival ``inf`` sentinels cannot carry a far-off lone detection.

    The failure signature: a lone artifact on a flat background
    reports ``inf`` for both ``detection_peak_ratio`` (no runner-up
    above background) and ``brightness_margin_mag`` (no rival catalog
    star), so both ratio gates pass vacuously.  The residual gate
    must reject the match when the detection sits outside the
    pointing-prior core (here 18.2 px, mirroring C0164392700R).
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (100.0, 100.0)
    # Lone bright source on a perfectly flat background: the runner-up
    # never clears the window median, so the peak ratio is ``inf``.
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    # Prediction 18.2 px away: inside the 30 px search window but beyond
    # the 10 px residual gate.
    planted = (12.0, -13.7)
    pred_vu = (actual_vu[0] - planted[0], actual_vu[1] - planted[1])
    # Single predictable star: the brightness margin is also ``inf``.
    feature = make_star_feature('star:UCAC4:far', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'one_star'
    assert result.diagnostics.brightness_margin_mag == float('inf')
    assert result.diagnostics.detection_peak_ratio == float('inf')
    assert result.diagnostics.residual_px == pytest.approx(math.hypot(*planted), abs=0.5)
    assert result.diagnostics.residual_px > 10.0


def test_star_unique_match_one_star_accepts_genuine_lone_bright_star(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A genuine lone bright star inside the residual gate still matches.

    The no-rival ``inf`` sentinels must keep passing the ratio gates: a
    single uniquely-predicted bright star on an otherwise empty frame is
    the technique's canonical use case, and the residual gate alone
    decides acceptance for it.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (100.0, 100.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    planted = (4.0, -3.0)
    pred_vu = (actual_vu[0] - planted[0], actual_vu[1] - planted[1])
    feature = make_star_feature('star:UCAC4:lone', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.brightness_margin_mag == float('inf')
    assert result.diagnostics.detection_peak_ratio == float('inf')
    assert result.confidence > 0.0


def test_star_unique_match_one_star_accepts_large_residual_with_finite_peak_ratio(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A measured (finite) peak ratio keeps large genuine offsets matchable.

    Operator-verified library frames carry genuine one-star offsets up
    to ~24 px (e.g. N1555145539_1 at 13.1 px): on a real background the
    runner-up is finite, the ambiguity gate measures actual uniqueness, and
    the residual gate must not fire.  Plants a faint rival blob so the
    peak ratio is finite yet far above ``one_star_min_peak_ratio``.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (100.0, 100.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    # Faint runner-up inside the search window but outside the
    # detection's exclusion box: the peak ratio becomes ~33 (finite).
    draw_gaussian_star(image, (85.0, 115.0), peak_dn=6.0, sigma=1.2)
    # Prediction 13.1 px away: beyond the 10 px no-rival gate, but the
    # gate does not apply because the ratio was measured.
    planted = (4.0, -12.5)
    pred_vu = (actual_vu[0] - planted[0], actual_vu[1] - planted[1])
    feature = make_star_feature('star:UCAC4:bright', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert math.isfinite(result.diagnostics.detection_peak_ratio)
    assert result.diagnostics.detection_peak_ratio >= 1.5
    assert result.diagnostics.residual_px > 10.0


def test_star_unique_match_one_star_reports_unambiguous_peak_ratio(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A lone bright star on a clean frame reports an unambiguous ratio."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    draw_gaussian_star(image, (100.0, 100.0), peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:1', predicted_vu=(98.0, 103.0), predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.detection_peak_ratio >= 1.5
