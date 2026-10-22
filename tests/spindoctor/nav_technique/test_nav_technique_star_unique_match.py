"""End-to-end tests for ``StarUniqueMatchNav``."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    FakeObs,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique._star_helpers import is_isolated_point_source
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
    # Prediction 18.2 px away: inside the search window but beyond
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


def test_star_unique_match_two_star_measures_each_star_past_a_brighter_spike(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single pixel outshining a star's peak does not take that star's window.

    Mirrors N1530185128_1: a cosmic ray 1.8 times the peak of the brighter
    star sits inside its search window.  Taking the brightest pixel measured
    the ray, the pair's residual went to tens of pixels and the two-star
    solution was declined; the star's light outweighs the ray's.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=15.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=12.0, sigma=1.2)
    image[35, 80] += 27.0
    planted = (5.0, -2.0)
    pred_a = (actual_a[0] - planted[0], actual_a[1] - planted[1])
    pred_b = (actual_b[0] - planted[0], actual_b[1] - planted[1])
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0)
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], make_nav_context(image))
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'two_star'
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.4)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)


def test_star_unique_match_one_star_refuses_a_window_with_a_brighter_spike(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """One star with a brighter single pixel in its window is not a unique match.

    The one-star path has no second star to check its detection against, so a
    window holding a pixel brighter than the detection's own peak is ambiguous
    and the result is spurious, rather than a confident offset to the spike.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (110.0, 100.0), peak_dn=15.0, sigma=1.2)
    image[80, 118] += 27.0
    feature = make_star_feature('star:UCAC4:1', predicted_vu=(100.0, 100.0), predicted_snr=40.0)
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.detection_peak_ratio < 1.0


def test_star_unique_match_pair_search_skips_a_star_not_in_the_image(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The two-star path finds the visible pair when the brightest prediction shows nothing.

    The brightest predicted star lies where the image is empty, as a star past
    the edge of the detector does, so the first pair tried has no detection
    for it; the next pairs are tried and the visible two match.
    """
    image = np.zeros((300, 300), dtype=np.float64)
    planted = (2.0, -3.0)
    actual_b = (100.0, 150.0)
    actual_c = (200.0, 220.0)
    draw_gaussian_star(image, actual_b, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_c, peak_dn=180.0, sigma=1.2)
    absent = make_star_feature('star:UCAC4:A', predicted_vu=(60.0, 40.0), predicted_snr=60.0)
    feat_b = make_star_feature(
        'star:UCAC4:B',
        predicted_vu=(actual_b[0] - planted[0], actual_b[1] - planted[1]),
        predicted_snr=55.0,
    )
    feat_c = make_star_feature(
        'star:UCAC4:C',
        predicted_vu=(actual_c[0] - planted[0], actual_c[1] - planted[1]),
        predicted_snr=50.0,
    )
    result = StarUniqueMatchNav().navigate([absent, feat_b, feat_c], make_nav_context(image))
    assert result.spurious is False
    assert result.feature_ids == ('star:UCAC4:B', 'star:UCAC4:C')
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.4)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)


def test_star_unique_match_refuses_pairs_that_disagree(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two star pairs that each pass the cross-check at different offsets are ambiguous."""
    image = np.zeros((400, 400), dtype=np.float64)
    first, second = (2.0, -3.0), (-9.0, 12.0)
    predictions = {
        'star:UCAC4:A': ((80.0, 80.0), first, 60.0),
        'star:UCAC4:B': ((80.0, 300.0), first, 55.0),
        'star:UCAC4:C': ((300.0, 80.0), second, 50.0),
        'star:UCAC4:D': ((300.0, 300.0), second, 45.0),
    }
    features = []
    for star_id, (pred, offset, snr) in predictions.items():
        draw_gaussian_star(image, (pred[0] + offset[0], pred[1] + offset[1]), peak_dn=200.0)
        features.append(make_star_feature(star_id, predicted_vu=pred, predicted_snr=snr))
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is True


def test_star_unique_match_refuses_a_pair_whose_stars_disagree_by_three_pixels(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two stars whose offsets differ by 3 px do not cross-check each other.

    The assignment residual is half that difference, 1.5 px, beyond the tolerance;
    two correctly matched stars agree to a fraction of a pixel.
    """
    image = np.zeros((300, 300), dtype=np.float64)
    draw_gaussian_star(image, (100.0, 100.0), peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, (203.0, 200.0), peak_dn=190.0, sigma=1.2)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(98.0, 103.0), predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(198.0, 203.0), predicted_snr=58.0)
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], make_nav_context(image))
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode != 'two_star'


def test_star_unique_match_refuses_the_cut_edge_of_a_source_outside_the_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A window whose brightest pixel is the wing of a source beyond its edge holds no star.

    The wing rises toward the window's edge, so it is no peak of the image and the window
    holds no detection.
    """
    image = np.zeros((300, 300), dtype=np.float64)
    technique = StarUniqueMatchNav()
    window_px = float(technique.tuning['search_window_px'])
    pred = (100.0, 150.0)
    draw_gaussian_star(image, (pred[0] + window_px + 3.0, pred[1]), peak_dn=500.0, sigma=2.0)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=pred, predicted_snr=60.0)
    result = technique.navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert 'no_detection_above_threshold' in capsys.readouterr().out


def test_star_unique_match_refuses_a_pair_of_tracks(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """Two cosmic-ray tracks at one offset from two predictions are not a star pair.

    Each window's brightest pixel lies on a track, and the two tracks sit at the same
    offset from their predictions, so the pair's residual is small; a track is not an
    isolated point source.
    """
    image = np.zeros((300, 300), dtype=np.float64)
    image[106, 90:110] = 300.0
    image[206, 190:210] = 300.0
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 95.0), predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(200.0, 195.0), predicted_snr=58.0)
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], make_nav_context(image))
    assert result.spurious is True


def test_star_unique_match_refuses_a_lone_star_beyond_the_inner_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lone detection inside the search window but beyond the inner window is refused."""
    image = np.zeros((300, 300), dtype=np.float64)
    technique = StarUniqueMatchNav()
    inner_px = float(technique.tuning['inner_search_window_px'])
    pred = (150.0, 150.0)
    draw_gaussian_star(image, (pred[0], pred[1] + inner_px + 5.0), peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, (40.0, 40.0), peak_dn=20.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=pred, predicted_snr=60.0)
    result = technique.navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert 'one_star_detection_beyond_inner_window' in capsys.readouterr().out


def test_is_isolated_point_source_accepts_a_star_and_refuses_a_track(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A Gaussian star is an isolated point source; a bright straight track is not."""
    image = np.zeros((120, 120), dtype=np.float64)
    draw_gaussian_star(image, (40.0, 40.0), peak_dn=100.0, sigma=1.2)
    image[80, 60:90] = 100.0
    star = is_isolated_point_source(
        image, (40.0, 40.0), sigma_px=1.0, kernel_half_px=3, inner_px=4, outer_px=6, max_ratio=0.5
    )
    track = is_isolated_point_source(
        image, (80.0, 75.0), sigma_px=1.0, kernel_half_px=3, inner_px=4, outer_px=6, max_ratio=0.5
    )
    assert star
    assert not track


def test_star_unique_match_pairs_found_only_in_the_inner_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A brighter star 40 px from one prediction spoils the wide window, not the inner one."""
    image = np.zeros((300, 300), dtype=np.float64)
    planted = (2.0, -3.0)
    draw_gaussian_star(image, (100.0 + planted[0], 100.0 + planted[1]), peak_dn=100.0, sigma=1.2)
    draw_gaussian_star(image, (200.0 + planted[0], 200.0 + planted[1]), peak_dn=100.0, sigma=1.2)
    draw_gaussian_star(image, (140.0, 100.0), peak_dn=400.0, sigma=1.2)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(200.0, 200.0), predicted_snr=58.0)
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], make_nav_context(image))
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.4)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)


def test_star_unique_match_pairs_found_only_in_the_wide_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A pair 40 px from its predictions lies outside the inner window and inside the wide one."""
    image = np.zeros((300, 300), dtype=np.float64)
    planted = (0.0, 40.0)
    draw_gaussian_star(image, (100.0 + planted[0], 100.0 + planted[1]), peak_dn=100.0, sigma=1.2)
    draw_gaussian_star(image, (200.0 + planted[0], 200.0 + planted[1]), peak_dn=100.0, sigma=1.2)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(200.0, 200.0), predicted_snr=58.0)
    context = make_nav_context(image, extfov_margin_vu=(60, 60))
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], context)
    assert result.spurious is False
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)


def test_star_unique_match_disagreeing_pairs_do_not_fall_back_to_one_star(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Passing pairs that disagree are spurious even when the one-star path would succeed.

    The brightest predicted star outshines the next by more than the 1.5 mag margin, so
    a one-star match on it would be accepted; the disagreement between pairs must stop
    the technique first.
    """
    image = np.zeros((400, 400), dtype=np.float64)
    first, second = (2.0, -3.0), (-9.0, 12.0)
    predictions = {
        'star:UCAC4:A': ((80.0, 80.0), first, 250.0),
        'star:UCAC4:B': ((80.0, 300.0), first, 60.0),
        'star:UCAC4:C': ((300.0, 80.0), second, 55.0),
        'star:UCAC4:D': ((300.0, 300.0), second, 50.0),
    }
    features = []
    for star_id, (pred, offset, snr) in predictions.items():
        draw_gaussian_star(image, (pred[0] + offset[0], pred[1] + offset[1]), peak_dn=200.0)
        features.append(make_star_feature(star_id, predicted_vu=pred, predicted_snr=snr))
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is True


@pytest.mark.parametrize(
    ('inst_config', 'accepted'), [(None, True), ({'star_one_star_radius_px': 9.0}, False)]
)
def test_star_unique_match_believes_a_lone_star_within_the_cameras_radius(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    inst_config: dict[str, float] | None,
    accepted: bool,
) -> None:
    """A lone star 9.6 px off is believed under the technique's radius, not a 9 px camera's."""
    image = np.zeros((300, 300), dtype=np.float64)
    draw_gaussian_star(image, (159.6, 150.0), peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(150.0, 150.0), predicted_snr=60.0)
    context = make_nav_context(image)
    cast(FakeObs, context.obs).inst_config = inst_config
    result = StarUniqueMatchNav().navigate([feature], context)
    assert (result.spurious is False) is accepted


def test_star_unique_match_searches_the_cameras_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A pair 20 px from its predictions lies beyond a camera's 13 px window."""
    image = np.zeros((300, 300), dtype=np.float64)
    draw_gaussian_star(image, (120.0, 100.0), peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, (220.0, 200.0), peak_dn=190.0, sigma=1.2)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(200.0, 200.0), predicted_snr=58.0)
    context = make_nav_context(image)
    cast(FakeObs, context.obs).inst_config = {
        'star_search_window_px': 13.0,
        'star_one_star_radius_px': 9.0,
    }
    result = StarUniqueMatchNav().navigate([feat_a, feat_b], context)
    assert result.spurious is True


def _camera(context: object, **inst_config: float) -> None:
    """Give a test context's observation stand-in a camera configuration."""
    cast(FakeObs, cast(NavContext, context).obs).inst_config = inst_config


@pytest.mark.parametrize(('offset_px', 'passes'), [(11.0, True), (15.0, False)])
def test_star_unique_match_window_only_camera_bounds_the_pair_search(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    offset_px: float,
    passes: bool,
) -> None:
    """A camera that names only a 13 px window finds a pair 11 px out and not 15 px out."""
    image = np.zeros((300, 300), dtype=np.float64)
    preds = ((100.0, 100.0), (200.0, 200.0))
    for pv, pu in preds:
        draw_gaussian_star(image, (pv + offset_px, pu), peak_dn=200.0, sigma=1.2)
    features = [
        make_star_feature(f'star:UCAC4:{k}', predicted_vu=pred, predicted_snr=60.0 - k)
        for k, pred in enumerate(preds)
    ]
    context = make_nav_context(image)
    _camera(context, star_search_window_px=13.0)
    result = StarUniqueMatchNav().navigate(features, context)
    assert (result.spurious is False) is passes


def test_star_unique_match_window_only_camera_keeps_the_default_radius(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A camera that widens only its window still believes a lone star within 30 px."""
    image = np.zeros((300, 300), dtype=np.float64)
    draw_gaussian_star(image, (175.0, 175.0), peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(150.0, 150.0), predicted_snr=60.0)
    context = make_nav_context(image)
    _camera(context, star_search_window_px=60.0)
    result = StarUniqueMatchNav().navigate([feature], context)
    assert result.spurious is True
    assert 'one_star_detection_beyond_inner_window' in capsys.readouterr().out


def test_star_unique_match_one_star_refuses_a_track(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lone prediction whose window holds a cosmic-ray track has no star."""
    image = np.zeros((300, 300), dtype=np.float64)
    for step in np.linspace(0.0, 20.0, 41):
        draw_gaussian_star(image, (155.0 + step, 155.0 + step), peak_dn=100.0, sigma=1.0)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(150.0, 150.0), predicted_snr=60.0)
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert 'window_peak_not_a_point_source' in capsys.readouterr().out


def _star_field(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    shape: tuple[int, int],
    stars: list[tuple[tuple[float, float], tuple[float, float] | None]],
) -> tuple[np.ndarray, list[NavFeature]]:
    """Draw each star at its prediction plus its offset (none drawn for ``None``).

    The stars are ranked by the order given, brightest first.
    """
    image = np.zeros(shape, dtype=np.float64)
    features = []
    for k, (pred, offset) in enumerate(stars):
        if offset is not None:
            draw_gaussian_star(image, (pred[0] + offset[0], pred[1] + offset[1]), peak_dn=200.0)
        features.append(
            make_star_feature(f'star:UCAC4:{k}', predicted_vu=pred, predicted_snr=60.0 - k)
        )
    return image, features


def test_star_unique_match_votes_in_the_narrow_window_the_pair_passed_in(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Sources 40 px from the other stars' predictions do not vote on a pair found at 30 px.

    The pair passes in the 30 px window first; the other three stars' windows of that
    size hold nothing.  Searched at 50 px they would hold the three sources and refuse
    the pair.
    """
    pair = (2.0, -3.0)
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        (500, 500),
        [
            ((80.0, 80.0), pair),
            ((80.0, 400.0), pair),
            ((400.0, 80.0), (40.0, 5.0)),
            ((400.0, 400.0), (-5.0, 40.0)),
            ((240.0, 240.0), (40.0, -40.0)),
        ],
    )
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is False


def test_star_unique_match_votes_only_among_the_searched_stars(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Stars below the five searched cast no vote, though they are found elsewhere."""
    pair = (2.0, -3.0)
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        (600, 600),
        [
            ((80.0, 80.0), pair),
            ((80.0, 500.0), pair),
            ((500.0, 80.0), None),
            ((500.0, 500.0), None),
            ((300.0, 300.0), None),
            ((80.0, 300.0), (20.0, 5.0)),
            ((300.0, 80.0), (-5.0, 20.0)),
            ((500.0, 300.0), (-20.0, -20.0)),
        ],
    )
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is False


def test_star_unique_match_pairs_within_twice_the_residual_are_not_rivals(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two passing pairs 1.8 px apart agree: rivals must differ by more than 2.4 px."""
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        (500, 500),
        [
            ((80.0, 80.0), (2.0, -3.0)),
            ((80.0, 400.0), (2.0, -3.0)),
            ((400.0, 80.0), (3.8, -3.0)),
            ((400.0, 400.0), (3.8, -3.0)),
        ],
    )
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is False


@pytest.mark.parametrize(('smear_px', 'two_star'), [(8.5, True), (0.0, False)])
def test_star_unique_match_a_smeared_pair_is_isolated_at_its_smear(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    smear_px: float,
    two_star: bool,
) -> None:
    """Two stars streaked over 8 rows pair up when their features carry that smear.

    A streak crosses the isolation annulus of an unsmeared star, as on the library
    frame N1530185128_1; the annulus moved out by half the expected smear clears it.
    """
    image = np.zeros((300, 300), dtype=np.float64)
    planted = (2.0, -3.0)
    preds = ((100.0, 100.0), (200.0, 200.0))
    for pv, pu in preds:
        for step in np.linspace(-4.0, 4.0, 17):
            draw_gaussian_star(image, (pv + planted[0] + step, pu + planted[1]), 25.0, 1.0)
    features = [
        make_star_feature(
            f'star:UCAC4:{k}', predicted_vu=pred, predicted_snr=60.0 - k, smear_length_px=smear_px
        )
        for k, pred in enumerate(preds)
    ]
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert (result.diagnostics.mode == 'two_star') is two_star


@pytest.mark.parametrize(('n_elsewhere', 'spurious'), [(3, True), (2, False)])
def test_star_unique_match_refuses_a_pair_the_other_stars_contradict(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
    n_elsewhere: int,
    spurious: bool,
) -> None:
    """Two stars that agree are refused when three others are all found elsewhere.

    The brightest two stars sit at one offset; the other searched stars sit at offsets
    of their own, as a catalog off by several pixels per star puts them.  Two such
    stars do not outvote the pair.
    """
    image = np.zeros((500, 500), dtype=np.float64)
    pair_offset = (2.0, -3.0)
    stars = [
        ((80.0, 80.0), pair_offset, 60.0),
        ((80.0, 400.0), pair_offset, 58.0),
        ((400.0, 80.0), (-15.0, 10.0), 50.0),
        ((400.0, 400.0), (12.0, 14.0), 48.0),
        ((240.0, 240.0), (-8.0, -20.0), 46.0),
    ][: 2 + n_elsewhere]
    features = []
    for k, (pred, offset, snr) in enumerate(stars):
        draw_gaussian_star(image, (pred[0] + offset[0], pred[1] + offset[1]), peak_dn=200.0)
        features.append(make_star_feature(f'star:UCAC4:{k}', predicted_vu=pred, predicted_snr=snr))
    result = StarUniqueMatchNav().navigate(features, make_nav_context(image))
    assert result.spurious is spurious
    assert ('pair_contradicted' in capsys.readouterr().out) is spurious


def test_star_unique_match_inner_window_is_a_radius(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lone star 25 px off on both axes is 35 px away, beyond a 30 px inner limit."""
    image = np.zeros((300, 300), dtype=np.float64)
    technique = StarUniqueMatchNav()
    inner_px = float(technique.tuning['inner_search_window_px'])
    side = inner_px * 0.85
    pred = (150.0, 150.0)
    draw_gaussian_star(image, (pred[0] + side, pred[1] + side), peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, (40.0, 40.0), peak_dn=20.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=pred, predicted_snr=60.0)
    result = technique.navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert 'one_star_detection_beyond_inner_window' in capsys.readouterr().out


# A 300 x 600 padded frame whose image spans columns 140 through 459, as a Cassini NAC
# frame's padding is wider across samples than the 50 px search window.  A star predicted
# at column 40 is 100 px outside the image, beyond the reach of any window.
_WIDE_MARGIN_VU = (32, 140)
_WIDE_MARGIN_SHAPE = (300, 600)


def test_star_unique_match_pairs_stars_past_unreachable_brighter_ones(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Five brighter stars out of reach do not take every slot of the pair search.

    The pair search tries the five brightest stars; here those five are predicted 100 px
    outside the image, so only once they are set aside are the two visible stars paired.
    """
    stars: list[tuple[tuple[float, float], tuple[float, float] | None]] = [
        ((60.0 + 40.0 * k, 40.0), None) for k in range(5)
    ]
    stars += [((100.0, 250.0), (2.0, -3.0)), ((200.0, 380.0), (2.0, -3.0))]
    image, features = _star_field(make_star_feature, draw_gaussian_star, _WIDE_MARGIN_SHAPE, stars)
    result = StarUniqueMatchNav().navigate(
        features, make_nav_context(image, extfov_margin_vu=_WIDE_MARGIN_VU)
    )
    assert result.feature_ids == ('star:UCAC4:5', 'star:UCAC4:6')


def test_star_unique_match_matches_a_lone_star_past_an_unreachable_brighter_one(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star predicted out of reach is neither the one-star candidate nor its rival."""
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        _WIDE_MARGIN_SHAPE,
        [((150.0, 40.0), None), ((150.0, 300.0), (2.0, -3.0))],
    )
    result = StarUniqueMatchNav().navigate(
        features, make_nav_context(image, extfov_margin_vu=_WIDE_MARGIN_VU)
    )
    assert result.feature_ids == ('star:UCAC4:1',)


def test_star_unique_match_reaches_the_image_through_the_wide_window(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A pair predicted beyond the inner window's reach but within the wide one's is kept.

    Both stars are predicted 40 px left of the image, farther than the 30 px inner window
    and nearer than the 50 px wide window, and a 45 px pointing error puts them in it.
    """
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        _WIDE_MARGIN_SHAPE,
        [((100.0, 100.0), (0.0, 45.0)), ((200.0, 100.0), (0.0, 45.0))],
    )
    result = StarUniqueMatchNav().navigate(
        features, make_nav_context(image, extfov_margin_vu=_WIDE_MARGIN_VU)
    )
    assert result.spurious is False
    assert result.offset_px[1] == pytest.approx(45.0, abs=0.4)


def test_star_unique_match_refuses_when_no_star_is_in_reach(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every star predicted out of reach of the image leaves nothing to search."""
    image, features = _star_field(
        make_star_feature,
        draw_gaussian_star,
        _WIDE_MARGIN_SHAPE,
        [((100.0, 40.0), None), ((200.0, 560.0), None)],
    )
    result = StarUniqueMatchNav().navigate(
        features, make_nav_context(image, extfov_margin_vu=_WIDE_MARGIN_VU)
    )
    assert result.spurious is True
    assert 'no_usable_star_within_reach_of_the_image' in capsys.readouterr().out
