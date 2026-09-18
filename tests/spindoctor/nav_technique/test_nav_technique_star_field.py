"""End-to-end and helper-level tests for ``StarFieldFromCatalogNav``."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from psfmodel import GaussianPSF
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.flags import StarFlags
from spindoctor.nav_technique.diagnostics import StarFieldDiagnostics
from spindoctor.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE, NavTechnique
from spindoctor.nav_technique.nav_technique_star_field import (
    StarFieldFromCatalogNav,
    _binomial_upper_tail,
    _canonical_triplet_order,
    _detect_image_sources,
    _enumerate_triplets,
    _hash_distance_sq,
    _optimal_inlier_assignment,
    _solve_translation,
    _star_photometry_saturated,
    _triplet_hash,
    _wide_offset_false_lock_expectation,
)


def _star_field_image(
    star_centers: list[tuple[float, float]],
    *,
    draw: DrawGaussianStarFactory,
    shape: tuple[int, int] = (300, 300),
    peak_dn: float = 200.0,
    sigma: float = 1.2,
) -> np.ndarray:
    """Return an image with planted Gaussian stars at the supplied centers."""
    image = np.zeros(shape, dtype=np.float64)
    for cv, cu in star_centers:
        draw(image, (cv, cu), peak_dn=peak_dn, sigma=sigma)
    return image


def _make_star_field_features(
    centers: list[tuple[float, float]],
    *,
    make_feature: NavFeatureFactory,
    offset: tuple[float, float],
    snrs: list[float] | None = None,
) -> list[NavFeature]:
    """Build a STAR feature per planted center, predicted at ``center - offset``.

    ``offset`` is the planted ``(dv, du)`` translation: predicted at
    position ``center - offset`` so that the technique recovers
    ``offset``.
    """
    if snrs is None:
        snrs = [40.0 - 0.5 * i for i in range(len(centers))]
    features: list[NavFeature] = []
    for i, (cv, cu) in enumerate(centers):
        pred = (cv - offset[0], cu - offset[1])
        features.append(
            make_feature(
                f'star:UCAC4:{i}',
                predicted_vu=pred,
                predicted_snr=snrs[i],
            )
        )
    return features


# ---------------------------------------------------------------------------
# Sub-piece 1 — source detection.
# ---------------------------------------------------------------------------


def test_detect_image_sources_finds_planted_stars(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The matched-filter detector recovers planted Gaussian peaks.

    The stars are planted at whole pixel centric positions on a noise-free
    frame, so each one is symmetric about the pixel it sits on and the moment
    centroid returns that position exactly.  The bound is what the finite
    centroid box leaves, and is far below the half pixel that separates the
    two coordinate systems.

    Parameters:
        draw_gaussian_star: Stamps a Gaussian source onto a frame in place.
    """
    centers = [(50.0, 60.0), (120.0, 80.0), (200.0, 220.0)]
    image = _star_field_image(centers, draw=draw_gaussian_star)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=30,
    )
    assert len(detected) == 3
    detected_centers = sorted((s.v, s.u) for s in detected)
    expected = sorted(centers)
    for got, want in zip(detected_centers, expected, strict=True):
        assert got[0] == pytest.approx(want[0], abs=0.01)
        assert got[1] == pytest.approx(want[1], abs=0.01)


def test_detect_image_sources_caps_at_max_sources(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """``max_sources`` keeps the brightest peaks first."""
    image = _star_field_image(
        [(50.0, 50.0), (100.0, 100.0), (150.0, 150.0), (200.0, 200.0)],
        draw=draw_gaussian_star,
        peak_dn=200.0,
    )
    # Plant a fainter fifth peak — it must rank below the four brighter
    # ones and get dropped when max_sources=4.
    draw_gaussian_star(image, (250.0, 250.0), peak_dn=80.0, sigma=1.2)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=4,
    )
    assert len(detected) == 4
    assert all(s.peak_dn >= detected[-1].peak_dn for s in detected)


def test_detect_image_sources_returns_empty_when_blank() -> None:
    """A blank image yields no detections."""
    image = np.zeros((100, 100), dtype=np.float64)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=30,
    )
    assert detected == []


# ---------------------------------------------------------------------------
# Sub-piece 2 — triplet hashing.
# ---------------------------------------------------------------------------


def test_triplet_hash_is_translation_invariant() -> None:
    """The hash does not change when every vertex is translated by the same offset."""
    a = (10.0, 20.0)
    b = (12.0, 35.0)
    c = (40.0, 22.0)
    h1 = _triplet_hash(a, b, c)
    h2 = _triplet_hash(
        (a[0] + 7.0, a[1] - 3.0),
        (b[0] + 7.0, b[1] - 3.0),
        (c[0] + 7.0, c[1] - 3.0),
    )
    assert h1 is not None
    assert h2 is not None
    for v1, v2 in zip(h1, h2, strict=True):
        assert v1 == pytest.approx(v2)


def test_triplet_hash_is_uniform_scale_invariant() -> None:
    """Multiplying every vertex by a constant scale leaves the hash unchanged."""
    a = (10.0, 20.0)
    b = (12.0, 35.0)
    c = (40.0, 22.0)
    s = 3.7
    h1 = _triplet_hash(a, b, c)
    h2 = _triplet_hash(
        (a[0] * s, a[1] * s),
        (b[0] * s, b[1] * s),
        (c[0] * s, c[1] * s),
    )
    assert h1 is not None
    assert h2 is not None
    for v1, v2 in zip(h1, h2, strict=True):
        assert v1 == pytest.approx(v2)


def test_triplet_hash_drops_collinear() -> None:
    """A triplet with two coincident points is rejected (None)."""
    h = _triplet_hash((10.0, 10.0), (10.0, 10.0), (20.0, 20.0))
    assert h is None


def test_enumerate_triplets_yields_one_per_unordered_set() -> None:
    """Each unordered triplet appears once, apex opposite the longest side."""
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)]
    brightness_rank = [0, 1, 2, 3]
    triplets = _enumerate_triplets(points, brightness_rank)
    # C(4,3) = 4 triplets.
    assert len(triplets) == 4
    seen_keys: set[tuple[int, int, int]] = set()
    for t in triplets:
        # A sits opposite the longest side (the side b--c is longest).
        pa = points[t.idx_a]
        pb = points[t.idx_b]
        pc = points[t.idx_c]
        opp_a = math.hypot(pb[0] - pc[0], pb[1] - pc[1])
        opp_b = math.hypot(pa[0] - pc[0], pa[1] - pc[1])
        opp_c = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        assert opp_a == max(opp_a, opp_b, opp_c)
        # B's opposite side is at least C's opposite side.
        assert opp_b >= opp_c
        # Unordered triplet seen once.
        ordered = sorted([t.idx_a, t.idx_b, t.idx_c])
        key = (ordered[0], ordered[1], ordered[2])
        assert key not in seen_keys
        seen_keys.add(key)


def test_enumerate_triplets_order_is_rotation_stable_on_equal_brightness() -> None:
    """Equal-brightness triplet canonicalisation is stable under rotation.

    A brightness-keyed apex would break the tie arbitrarily and flip the
    labeling when the field rotates; the geometric order canonicalises to
    the same physical vertex assignment for a triangle and any rotated copy.
    """
    points = [(0.0, 0.0), (30.0, 4.0), (10.0, 25.0)]
    equal_brightness = [0, 0, 0]
    base = _enumerate_triplets(points, equal_brightness)
    assert len(base) == 1
    theta = math.radians(37.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = [(cos_t * v - sin_t * u, sin_t * v + cos_t * u) for v, u in points]
    rotated_triplets = _enumerate_triplets(rotated, equal_brightness)
    assert len(rotated_triplets) == 1
    # The apex-first index order is identical before and after rotation.
    assert (base[0].idx_a, base[0].idx_b, base[0].idx_c) == (
        rotated_triplets[0].idx_a,
        rotated_triplets[0].idx_b,
        rotated_triplets[0].idx_c,
    )


def test_enumerate_triplets_order_independent_of_brightness_ranking() -> None:
    """Two brightness rankings produce the same scalene-triangle apex order.

    With a scalene triangle the geometry fixes the vertex order regardless
    of how brightness ranks the vertices, so a re-ranking (the equal-vmag
    seed lottery) cannot change which correspondences the matcher forms.
    """
    points = [(0.0, 0.0), (30.0, 4.0), (10.0, 25.0)]
    order_one = _enumerate_triplets(points, [0, 1, 2])
    order_two = _enumerate_triplets(points, [2, 0, 1])
    assert (order_one[0].idx_a, order_one[0].idx_b, order_one[0].idx_c) == (
        order_two[0].idx_a,
        order_two[0].idx_b,
        order_two[0].idx_c,
    )


def test_canonical_triplet_order_isosceles_breaks_tie_on_brightness() -> None:
    """An isosceles triangle breaks the equal-opposite-side tie on brightness.

    Vertices 0 and 1 have equal-length opposite sides, so geometry alone cannot
    order them; the documented fallback orders the tied pair by brightness rank
    ascending. With vertex 0 the brighter (lower rank), it precedes vertex 1.
    """
    points = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]
    order = _canonical_triplet_order(points, (0, 1, 2), [0, 1, 2])
    assert order == (2, 0, 1)


def test_canonical_triplet_order_isosceles_tie_follows_brightness_rank() -> None:
    """Reversing the tied pair's brightness rank swaps their canonical order.

    The apex (vertex 2, opposite the longest side) is fixed by geometry, but
    the two equal-opposite-side vertices are ordered by brightness rank, so
    ranking vertex 1 brighter than vertex 0 places it first.
    """
    points = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]
    order = _canonical_triplet_order(points, (0, 1, 2), [1, 0, 2])
    assert order == (2, 1, 0)


def test_hash_distance_sq_zero_for_identical_hashes() -> None:
    """``_hash_distance_sq(h, h, ...)`` is exactly zero."""
    h = (1.5, 2.5, 0.7)
    d_sq = _hash_distance_sq(h, h, ratio_weight=1.0, angle_weight=1.0)
    assert d_sq == 0.0


# ---------------------------------------------------------------------------
# Helper-level translation solve and inlier counting.
# ---------------------------------------------------------------------------


def test_solve_translation_recovers_constant_offset() -> None:
    """Constant per-correspondence offset is recovered as the weighted mean."""
    catalog = np.asarray([[10.0, 20.0], [30.0, 50.0], [70.0, 110.0]], dtype=np.float64)
    planted = (4.0, -2.5)
    detection = catalog + np.asarray(planted)
    weights = np.ones(3, dtype=np.float64)
    dv, du = _solve_translation(detection, catalog, weights)
    assert dv == pytest.approx(planted[0])
    assert du == pytest.approx(planted[1])


def test_optimal_inlier_assignment_under_perfect_offset() -> None:
    """Every detection is an inlier when the offset perfectly aligns the catalog."""
    catalog = np.asarray([[10.0, 20.0], [30.0, 50.0], [70.0, 110.0]], dtype=np.float64)
    offset = (4.0, -2.5)
    detections = catalog + np.asarray(offset)
    n_inliers, pairs = _optimal_inlier_assignment(
        detections, catalog, offset_vu=offset, tolerance_px=0.5
    )
    assert n_inliers == 3
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]


def test_optimal_inlier_assignment_does_not_double_match() -> None:
    """Each catalog star matches at most one detection; the closer detection wins."""
    catalog = np.asarray([[10.0, 20.0]], dtype=np.float64)
    detections = np.asarray([[10.0, 20.0], [10.5, 19.9]], dtype=np.float64)
    n_inliers, pairs = _optimal_inlier_assignment(
        detections, catalog, offset_vu=(0.0, 0.0), tolerance_px=2.0
    )
    assert n_inliers == 1
    assert pairs == [(0, 0)]


def test_optimal_inlier_assignment_resolves_competing_detections() -> None:
    """Two detections competing for one catalog star pair off one-to-one.

    Detection 0 at (2.5, 0) is nearest to catalog star 1 at (4, 0); a
    greedy sweep in detection order would consume star 1 for detection
    0 and leave detection 1 at (4.5, 0) with only star 0 at (0, 0),
    which is beyond tolerance -- one inlier.  The maximum-cardinality
    assignment pairs detection 0 with star 0 (distance 2.5) and
    detection 1 with star 1 (distance 0.5) -- two inliers.
    """
    catalog = np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    detections = np.asarray([[2.5, 0.0], [4.5, 0.0]], dtype=np.float64)
    n_inliers, pairs = _optimal_inlier_assignment(
        detections, catalog, offset_vu=(0.0, 0.0), tolerance_px=3.0
    )
    assert n_inliers == 2
    assert pairs == [(0, 0), (1, 1)]


def test_optimal_inlier_assignment_is_order_independent() -> None:
    """Reversing the detection ordering leaves the assignment unchanged.

    Same competing-detection geometry as above with the detection rows
    swapped: the inlier count and the (detection, catalog) pairing set
    must be identical up to the detection reindexing.
    """
    catalog = np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    detections = np.asarray([[4.5, 0.0], [2.5, 0.0]], dtype=np.float64)
    n_inliers, pairs = _optimal_inlier_assignment(
        detections, catalog, offset_vu=(0.0, 0.0), tolerance_px=3.0
    )
    assert n_inliers == 2
    assert pairs == [(0, 1), (1, 0)]


def test_optimal_inlier_assignment_prefers_min_residual_at_equal_cardinality() -> None:
    """Among maximum-cardinality assignments the lowest total residual wins."""
    catalog = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    detections = np.asarray([[0.1, 0.0], [0.9, 0.0]], dtype=np.float64)
    n_inliers, pairs = _optimal_inlier_assignment(
        detections, catalog, offset_vu=(0.0, 0.0), tolerance_px=2.0
    )
    assert n_inliers == 2
    assert pairs == [(0, 0), (1, 1)]


def test_optimal_inlier_assignment_empty_inputs() -> None:
    """Empty detection or catalog arrays produce zero inliers."""
    empty = np.zeros((0, 2), dtype=np.float64)
    catalog = np.asarray([[1.0, 2.0]], dtype=np.float64)
    n_inliers, pairs = _optimal_inlier_assignment(
        empty, catalog, offset_vu=(0.0, 0.0), tolerance_px=1.0
    )
    assert n_inliers == 0
    assert pairs == []


# ---------------------------------------------------------------------------
# Sub-piece 3 / 4 — RANSAC + verification (technique-level).
# ---------------------------------------------------------------------------


def test_star_field_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """RANSAC + Tukey refit recovers a planted translation on a clean field."""
    centers = [
        (60.0, 80.0),
        (110.0, 130.0),
        (180.0, 90.0),
        (220.0, 200.0),
        (90.0, 240.0),
        (250.0, 70.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (3.0, -2.5)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(32, 32))
    feasibility = technique.is_feasible(features)
    assert feasibility.feasible is True
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.n_inliers >= 6


def test_star_field_is_deterministic_across_two_invocations(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two back-to-back invocations on the same obs return bit-identical fits."""
    centers = [
        (60.0, 80.0),
        (110.0, 130.0),
        (180.0, 90.0),
        (220.0, 200.0),
        (90.0, 240.0),
        (250.0, 70.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (1.5, 2.0)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image)
    result_a = technique.navigate(features, context)
    result_b = technique.navigate(features, context)
    assert result_a.offset_px == result_b.offset_px
    assert np.array_equal(result_a.covariance_px2, result_b.covariance_px2)
    assert result_a.confidence == result_b.confidence
    assert result_a.diagnostics == result_b.diagnostics


def test_star_field_infeasible_with_fewer_than_three_features(
    make_star_feature: NavFeatureFactory,
) -> None:
    """``is_feasible`` requires >= 3 usable STAR features."""
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(50.0, 50.0), predicted_snr=40.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(80.0, 80.0), predicted_snr=35.0)
    technique = StarFieldFromCatalogNav()
    report = technique.is_feasible([feat_a, feat_b])
    assert report.feasible is False
    assert 'fewer_than_3_predicted_stars' in report.reason


def test_star_field_infeasible_when_all_stars_occluded(
    make_star_feature: NavFeatureFactory,
) -> None:
    """Stars marked in_body_silhouette are filtered out before feasibility."""
    features = [
        make_star_feature(
            f'star:UCAC4:{i}',
            predicted_vu=(50.0 + 30.0 * i, 60.0 + 25.0 * i),
            predicted_snr=30.0,
            in_body_silhouette=True,
        )
        for i in range(5)
    ]
    technique = StarFieldFromCatalogNav()
    report = technique.is_feasible(features)
    assert report.feasible is False
    assert 'fewer_than_3_predicted_stars' in report.reason


def test_star_field_returns_spurious_when_too_few_detections(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Image with only 2 bright peaks fails the >= 3 detected-sources check."""
    image = _star_field_image(
        [(50.0, 50.0), (120.0, 130.0)], draw=draw_gaussian_star, shape=(200, 200)
    )
    centers = [(50.0, 50.0), (120.0, 130.0), (160.0, 80.0)]
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=(0.0, 0.0))
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.spurious is True
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.n_detected_sources == 2


def test_star_field_marks_at_edge_when_offset_hits_margin(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """An offset that hits the search-window edge is flagged + zero confidence."""
    centers = [
        (80.0, 80.0),
        (110.0, 130.0),
        (140.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    margin = (8, 8)
    # Plant the offset on the V-axis margin.
    planted = (float(margin[0]), 0.0)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=margin)
    result = technique.navigate(features, context)
    # Translation hit the v-axis margin: at_edge fires and the
    # confidence formula's hard_zero_if pushes confidence to zero.
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_star_field_registered_with_navtechnique_registry() -> None:
    """``StarFieldFromCatalogNav`` self-registers on import."""
    assert StarFieldFromCatalogNav in NavTechnique._registry


def test_star_field_min_inliers_less_than_three_raises() -> None:
    """Construction rejects a ``pattern_match_min_inliers`` below 3."""
    technique_class = StarFieldFromCatalogNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['pattern_match_min_inliers'] = 2
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='pattern_match_min_inliers'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


def test_wide_offset_one_star_catalog_margin_not_positive_raises() -> None:
    """Construction rejects a non-positive ``wide_offset_one_star_catalog_margin_mag``."""
    technique_class = StarFieldFromCatalogNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['wide_offset_one_star_catalog_margin_mag'] = 0.0
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='wide_offset_one_star_catalog_margin_mag'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


def test_wide_offset_one_star_detection_ratio_below_one_raises() -> None:
    """Construction rejects a ``wide_offset_one_star_detection_margin_ratio`` below 1."""
    technique_class = StarFieldFromCatalogNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['wide_offset_one_star_detection_margin_ratio'] = 0.9
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='wide_offset_one_star_detection_margin_ratio'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


def test_wide_offset_one_star_confidence_cap_out_of_range_raises() -> None:
    """Construction rejects a ``wide_offset_one_star_confidence_cap`` outside [0, 1]."""
    technique_class = StarFieldFromCatalogNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['wide_offset_one_star_confidence_cap'] = 1.5
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='wide_offset_one_star_confidence_cap'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


# ---------------------------------------------------------------------------
# PSF-fit inlier refinement.
# ---------------------------------------------------------------------------


class _PSFProviderObs:
    """Obs stand-in exposing only ``star_psf`` (satisfies ``_StarPSFProvider``)."""

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    def star_psf(self) -> GaussianPSF:
        return GaussianPSF(sigma=self._sigma)


def _render_eval_rect_star(
    image: np.ndarray, center_vu: tuple[float, float], *, peak_dn: float, sigma: float
) -> None:
    """Stamp a pixel-integrated Gaussian whose centroid lands at pixel-center ``center``.

    Uses the same ``eval_rect(offset + 0.5)`` convention as the production
    renderer, so a ``find_position`` fit (which reports the ``eval_rect``
    position) recovers ``center`` after the technique's half-pixel correction.
    """
    psf = GaussianPSF(sigma=sigma)
    cv, cu = center_vu
    v_int, u_int = int(cv), int(cu)
    half = 8
    stamp = psf.eval_rect(
        (half * 2 + 1, half * 2 + 1),
        offset=(cv - v_int + 0.5, cu - u_int + 0.5),
        scale=1.0,
        movement=(0.0, 0.0),
        movement_granularity=1.0,
    )
    stamp = stamp / float(stamp.max()) * peak_dn
    image[v_int - half : v_int + half + 1, u_int - half : u_int + half + 1] += stamp


def test_box_snr_zero_for_flat_image() -> None:
    """A flat box has no net signal, so its integrated SNR is zero."""
    image = np.full((40, 40), 17.0, dtype=np.float64)
    assert StarFieldFromCatalogNav._box_snr(image, 20, 20, 5, 4.0) == pytest.approx(0.0)


def test_box_snr_increases_with_source_brightness() -> None:
    """A brighter source yields a larger integrated SNR in the same box."""
    faint = np.full((40, 40), 20.0, dtype=np.float64)
    _render_eval_rect_star(faint, (20.0, 20.0), peak_dn=100.0, sigma=1.0)
    bright = np.full((40, 40), 20.0, dtype=np.float64)
    _render_eval_rect_star(bright, (20.0, 20.0), peak_dn=1000.0, sigma=1.0)
    snr_faint = StarFieldFromCatalogNav._box_snr(faint, 20, 20, 5, 4.0)
    snr_bright = StarFieldFromCatalogNav._box_snr(bright, 20, 20, 5, 4.0)
    assert snr_bright > snr_faint


def test_psf_refine_corrects_seeded_centroid_error(
    make_nav_context: NavContextFactory,
) -> None:
    """PSF refinement pulls a deliberately-offset centroid back onto the star.

    Exercises the ``find_position`` path and the half-pixel ``eval_rect``
    convention: a noiseless star is planted at a sub-pixel center, the input
    centroid is seeded 0.3-0.4 px away, and the refined position must land back
    on the true center.
    """
    true_center = (30.37, 30.62)
    image = np.full((60, 60), 20.0, dtype=np.float64)
    _render_eval_rect_star(image, true_center, peak_dn=150.0, sigma=1.0)
    context = replace(make_nav_context(image), obs=_PSFProviderObs(1.0))
    technique = StarFieldFromCatalogNav()
    technique._psf_refine_snr_max = 1.0e18  # force refinement regardless of SNR
    seeded = np.array([[true_center[0] + 0.4, true_center[1] - 0.3]], dtype=np.float64)
    refined = technique._psf_refine_positions(seeded, context)
    assert refined[0, 0] == pytest.approx(true_center[0], abs=0.05)
    assert refined[0, 1] == pytest.approx(true_center[1], abs=0.05)


def test_psf_refine_keeps_moment_for_bright_source(
    make_nav_context: NavContextFactory,
) -> None:
    """A source above the SNR ceiling keeps its moment centroid unchanged."""
    image = np.full((60, 60), 20.0, dtype=np.float64)
    _render_eval_rect_star(image, (30.4, 30.6), peak_dn=150.0, sigma=1.0)
    context = replace(make_nav_context(image), obs=_PSFProviderObs(1.0))
    technique = StarFieldFromCatalogNav()
    technique._psf_refine_snr_max = 0.0  # every detection counts as "bright"
    seeded = np.array([[30.8, 30.3]], dtype=np.float64)
    refined = technique._psf_refine_positions(seeded, context)
    assert np.array_equal(refined, seeded)


def test_psf_refine_noop_when_obs_lacks_star_psf(
    make_nav_context: NavContextFactory,
) -> None:
    """Refinement is a no-op when the observation cannot supply a star PSF."""
    image = np.full((60, 60), 20.0, dtype=np.float64)
    _render_eval_rect_star(image, (30.4, 30.6), peak_dn=150.0, sigma=1.0)
    context = make_nav_context(image)  # obs is FakeObs, no star_psf method
    technique = StarFieldFromCatalogNav()
    seeded = np.array([[30.8, 30.3]], dtype=np.float64)
    refined = technique._psf_refine_positions(seeded, context)
    assert np.array_equal(refined, seeded)


# ---------------------------------------------------------------------------
# 3-DoF (Procrustes / similarity-transform) tests.
# ---------------------------------------------------------------------------


def _rotate_about(
    points: list[tuple[float, float]],
    pivot: tuple[float, float],
    theta_rad: float,
) -> list[tuple[float, float]]:
    """Rotate ``points`` about ``pivot`` by ``theta_rad``.

    Parameters:
        points: Sequence of ``(v, u)`` pixel coordinates to rotate.
        pivot: ``(v, u)`` pivot in the same coordinate frame as
            ``points``.  The rotation is applied about this point with
            no translation.
        theta_rad: Rotation angle in radians, counter-clockwise in the
            image-frame ``(v, u)`` axes (``v`` increasing downward,
            ``u`` increasing rightward) — i.e. the standard rotation
            matrix ``[[cos, -sin], [sin, cos]]`` applied to the
            ``(v, u)`` offset from the pivot.

    Returns:
        New list of ``(v, u)`` tuples, one per input point, in the
        same order as ``points``.
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    pv, pu = pivot
    rotated: list[tuple[float, float]] = []
    for v, u in points:
        rv = v - pv
        ru = u - pu
        rotated.append(
            (
                pv + cos_t * rv - sin_t * ru,
                pu + sin_t * rv + cos_t * ru,
            )
        )
    return rotated


def _rotated_star_field_fixture(
    *,
    draw: DrawGaussianStarFactory,
    make_feature: NavFeatureFactory,
    planted_offset: tuple[float, float],
    planted_theta: float,
) -> tuple[np.ndarray, list[NavFeature]]:
    """Build an image + catalog features for a planted rotation + translation.

    The detection-frame star centers are rotated back into the catalog frame
    about the (shifted) field centroid, so the technique should report
    ``(planted_offset, planted_theta)`` as the similarity transform that maps
    catalog to detection.

    Parameters:
        draw: Gaussian-star renderer fixture.
        make_feature: STAR-feature factory fixture.
        planted_offset: Planted ``(dv, du)`` translation in pixels.
        planted_theta: Planted camera roll in radians.

    Returns:
        ``(image, features)`` ready for ``StarFieldFromCatalogNav.navigate``.
    """
    centers = [
        (80.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
        (60.0, 60.0),
        (260.0, 250.0),
    ]
    image = _star_field_image(centers, draw=draw, shape=(320, 320))
    pivot = (
        sum(c[0] for c in centers) / len(centers),
        sum(c[1] for c in centers) / len(centers),
    )
    catalog_centers = _rotate_about(
        [(c[0] - planted_offset[0], c[1] - planted_offset[1]) for c in centers],
        pivot=(pivot[0] - planted_offset[0], pivot[1] - planted_offset[1]),
        theta_rad=-planted_theta,
    )
    snrs = [40.0 - 0.5 * i for i in range(len(catalog_centers))]
    features: list[NavFeature] = [
        make_feature(
            f'star:UCAC4:{i}',
            predicted_vu=catalog_centers[i],
            predicted_snr=snrs[i],
        )
        for i in range(len(catalog_centers))
    ]
    return image, features


def test_star_field_3dof_recovers_planted_rotation(
    draw_gaussian_star: DrawGaussianStarFactory,
    make_star_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An above-floor planted rotation + translation is recovered normally."""
    # 1.0 deg is above the 0.75 deg roll/translation separability floor, so
    # the rotation must be reported with a genuine (non-sentinel) variance.
    planted_theta = math.radians(1.0)
    image, features = _rotated_star_field_fixture(
        draw=draw_gaussian_star,
        make_feature=make_star_feature,
        planted_offset=(1.5, -2.0),
        planted_theta=planted_theta,
    )
    technique = StarFieldFromCatalogNav()
    # 1.0 deg of roll displaces the outermost stars by ~2.5 px relative to a
    # candidate triplet's pure-translation transform; widen the RANSAC inlier
    # tolerance so the rotation-induced residual does not evict inliers
    # before the Procrustes refit absorbs it.
    technique._inlier_tolerance_px = 4.0
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(planted_theta, abs=math.radians(0.2))
    assert result.sigma_rotation_rad is not None
    assert result.sigma_rotation_rad > 0.0
    assert result.covariance_px2[2, 2] < ROTATION_UNOBSERVABLE_VARIANCE
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.rotation_below_separability_floor is False


def test_star_field_3dof_sub_floor_rotation_reports_unobservable(
    draw_gaussian_star: DrawGaussianStarFactory,
    make_star_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A planted sub-floor roll is reported unobservable, never a confident zero.

    Below the ~0.75 deg roll/translation separability floor the fitted
    rotation is not separable from a translation (see the camera-roll
    section of the simulator report), so the rotation slot must carry the
    unobservable sentinel and the diagnostics flag -- not a small sigma.
    """
    planted_theta = math.radians(0.5)
    image, features = _rotated_star_field_fixture(
        draw=draw_gaussian_star,
        make_feature=make_star_feature,
        planted_offset=(1.5, -2.0),
        planted_theta=planted_theta,
    )
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)
    assert result.sigma_rotation_rad is not None
    assert result.sigma_rotation_rad == pytest.approx(math.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.rotation_below_separability_floor is True


def test_star_field_3dof_zero_rotation_path_remains_close_to_planted_offset(
    draw_gaussian_star: DrawGaussianStarFactory,
    make_star_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A zero-rotation scene under fit_camera_rotation still recovers the offset."""
    centers = [
        (80.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
        (60.0, 60.0),
        (260.0, 250.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (1.0, -2.5)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate(features, context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert result.rotation_rad == pytest.approx(0.0, abs=math.radians(0.5))
    # A fitted near-zero roll sits below the separability floor: it must be
    # flagged unobservable rather than reported as a confident zero.
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.rotation_below_separability_floor is True


def test_build_covariance_two_point_reduced_chi_square() -> None:
    """Translation-only covariance matches the analytic reduced-chi-square mean.

    Per-axis var = chi2_nu / sum(w), chi2_nu = sum(w r^2)/max(N - p, 1),
    p = 2.  w = [1, 3], sum(w) = 4, N = 2, dof = max(2 - 2, 1) = 1.

    V residuals = [-1.5, 0.5]: sum(w r^2) = 1*2.25 + 3*0.25 = 3.0,
        chi2_nu = 3.0/1 = 3.0, var = 3.0/4 = 0.75 (> floor 1/4 = 0.25).
    U residuals = [-1.5, 0.5]: identical -> var = 0.75.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 3.0], dtype=np.float64)
    residuals = np.array([[-1.5, -1.5], [0.5, 0.5]], dtype=np.float64)
    cov = technique._build_covariance(weights=weights, residuals=residuals)
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(0.75, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.75, abs=1e-9)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_n_point_reduced_chi_square() -> None:
    """Four-point translation covariance matches the analytic reduced chi-square.

    w = [1, 1, 1, 1], sum(w) = 4, N = 4, p = 2, dof = max(4 - 2, 1) = 2.

    V residuals = [-3, -1, 1, 3]: sum(w r^2) = 9 + 1 + 1 + 9 = 20,
        chi2_nu = 20/2 = 10, var = 10/4 = 2.5 (> floor 0.25).
    U residuals = [0, 0, 0, 0]: sum(w r^2) = 0, candidate var 0 ->
        floored at 1/sum(w) = 0.25.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    cov = technique._build_covariance(weights=weights, residuals=residuals)
    assert cov[0, 0] == pytest.approx(2.5, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)


def test_build_covariance_model_error_floor_inflates_by_square() -> None:
    """ORCH-001: model_error_floor_px>0 adds exactly its square to the diagonal.

    Reuses the two-point fixture (var 0.75/axis with no floor).  Setting
    model_error_floor_px = 2.0 adds 2.0**2 = 4.0 -> 4.75 on each axis;
    the off-diagonal stays zero.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 3.0], dtype=np.float64)
    residuals = np.array([[-1.5, -1.5], [0.5, 0.5]], dtype=np.float64)
    base = technique._build_covariance(weights=weights, residuals=residuals)
    technique._model_error_floor_px = 2.0
    floored = technique._build_covariance(weights=weights, residuals=residuals)
    assert floored[0, 0] == pytest.approx(4.75, abs=1e-9)
    assert floored[1, 1] == pytest.approx(4.75, abs=1e-9)
    assert floored[0, 0] - base[0, 0] == pytest.approx(4.0, abs=1e-9)
    assert floored[1, 1] - base[1, 1] == pytest.approx(4.0, abs=1e-9)
    assert floored[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_3dof_rotation_variance_analytic() -> None:
    """3-DoF rotation variance matches the analytic Fisher information.

    Four equal-weight points (w = 1), p = 3, dof = max(4 - 3, 1) = 1.

    Catalog positions (0,0),(0,4),(4,0),(4,4): weighted centroid (2,2),
    lever arms dv = [-2,-2,2,2], du = [-2,2,-2,2].

    Residuals: r_v = [1, -1, 1, -1], r_u = [0, 0, 0, 0].
        var_v = max(4/1, 1) = 4, var_u = max(0/1, 1) = 1 (floored),
        I_theta = sum(du^2/var_v + dv^2/var_u) = 4*(4/4 + 4/1) = 20,
        sigma_theta^2 = 1/20 = 0.05.

    (The isotropic pooled lever-arm form would report
    max(2.0/32, 1/32) = 0.0625 here -- the anisotropic residuals make
    the exact value differ.)

    Translation block (p = 3, dof = 1):
        V: chi2_nu = 4/1 = 4, var = 4/4 = 1.0 (> floor 0.25).
        U: chi2_nu = 0 -> floored at 0.25.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    cat_inliers = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 4.0]], dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov.shape == (3, 3)
    assert cov[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)
    assert cov[2, 2] == pytest.approx(0.05, abs=1e-9)
    # Cross-terms are zero by construction.
    assert cov[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert cov[1, 2] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_3dof_isotropic_limit_matches_lever_arm_form() -> None:
    """In the isotropic limit the Fisher form reduces to the classic lever-arm value.

    Same corner catalog (spread = 32, dof = 1) with isotropic
    residuals: r_v = [1,-1,0,0], r_u = [0,0,1,-1] give
    var_v = var_u = 2, so

        I_theta = sum(du^2 + dv^2)/2 = 32/2 = 16
        sigma_theta^2 = 1/16 = chi2_nu_residual/spread = 2/32.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64)
    cat_inliers = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 4.0]], dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov[2, 2] == pytest.approx(1.0 / 16.0, abs=1e-9)


def test_build_covariance_3dof_anisotropic_residuals_exact() -> None:
    """Anisotropic residuals give the exact tangential rotation variance.

    Five equal-weight points on the u axis: catalog
    (0,-4),(0,-2),(0,0),(0,2),(0,4), centroid (0,0), lever arms
    dv = 0, du = [-4,-2,0,2,4].  Every rotation Jacobian is purely
    along v (the tangential direction of a u-axis lever arm), so only
    the v-axis residual variance constrains theta.

    Residuals: r_v = [2,-2,2,-2,0], r_u = [1,-1,1,-1,0]; dof = 2.
        var_v = 16/2 = 8, var_u = 4/2 = 2 (no floor active),
        I_theta = sum(du^2)/var_v = 40/8 = 5,
        sigma_theta^2 = 0.2.

    The isotropic pooled formula would report
    0.5*(16+4)/2 / 40 = 0.125 -- a 1.6x underestimate; this is the
    regression the exact form corrects.  The value is cross-checked
    against a brute-force Fisher information built from a
    finite-difference rotation Jacobian.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.ones(5, dtype=np.float64)
    residuals = np.array(
        [[2.0, 1.0], [-2.0, -1.0], [2.0, 1.0], [-2.0, -1.0], [0.0, 0.0]], dtype=np.float64
    )
    cat_inliers = np.array(
        [[0.0, -4.0], [0.0, -2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0]], dtype=np.float64
    )
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov[2, 2] == pytest.approx(0.2, abs=1e-9)
    old_isotropic = (0.5 * (16.0 + 4.0) / 2.0) / 40.0
    assert cov[2, 2] != pytest.approx(old_isotropic, abs=1e-3)
    # Brute-force Fisher information: finite-difference the rotated
    # lever arm w.r.t. theta and accumulate per-axis precision.
    dof = 2.0
    var_v = float(np.sum(weights * residuals[:, 0] ** 2)) / dof
    var_u = float(np.sum(weights * residuals[:, 1] ** 2)) / dof
    centroid = np.sum(weights[:, None] * cat_inliers, axis=0) / float(np.sum(weights))
    lever = cat_inliers - centroid[None, :]
    eps = 1.0e-6

    def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float64,
        )
        return points @ rot.T

    jac = (_rotate(lever, eps) - _rotate(lever, -eps)) / (2.0 * eps)
    fisher_numeric = float(np.sum(weights * (jac[:, 0] ** 2 / var_v + jac[:, 1] ** 2 / var_u)))
    assert cov[2, 2] == pytest.approx(1.0 / fisher_numeric, rel=1e-9)


def test_build_covariance_3dof_zero_residuals_floor() -> None:
    """A noise-free fit floors both per-axis variances at 1 px^2.

    With zero residuals var_v = var_u = 1 (floored), so the rotation
    variance falls back to the inverse lever-arm spread 1/32.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.zeros((4, 2), dtype=np.float64)
    cat_inliers = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 4.0]], dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov[2, 2] == pytest.approx(1.0 / 32.0, abs=1e-12)


def test_build_covariance_3dof_coincident_catalog_is_rotation_unobservable() -> None:
    """Coincident catalog inliers (zero spread) report rotation unobservable.

    With all catalog points at the same location the lever-arm spread is
    zero, so the rotation slot collapses to the unobservable sentinel.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]], dtype=np.float64)
    cat_inliers = np.full((4, 2), 7.0, dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)


# ---------------------------------------------------------------------------
# Wide-offset asterism matching — false-lock significance model.
# ---------------------------------------------------------------------------


def test_binomial_upper_tail_full_range_is_unity() -> None:
    """``P(X >= 0)`` is 1 for any binomial."""
    assert _binomial_upper_tail(5, 0.3, 0) == pytest.approx(1.0)


def test_binomial_upper_tail_above_n_is_zero() -> None:
    """Asking for more successes than trials yields zero probability."""
    assert _binomial_upper_tail(3, 0.3, 4) == 0.0


def test_binomial_upper_tail_matches_closed_form() -> None:
    """The tail sum matches a hand-computed two-of-three value."""
    # P(X >= 2 | n=3, p=0.5) = P(2) + P(3) = 3/8 + 1/8 = 0.5.
    assert _binomial_upper_tail(3, 0.5, 2) == pytest.approx(0.5)


def test_wide_offset_expectation_three_inliers_is_significant() -> None:
    """A three-inlier anchor lock sits far below a one-percent budget."""
    expectation = _wide_offset_false_lock_expectation(
        n_inliers=3,
        n_seeds=15,
        n_catalog=18,
        n_detected=30,
        tolerance_px=2.0,
        image_area_px2=1500.0 * 1500.0,
    )
    assert expectation < 1.0e-3


def test_wide_offset_expectation_two_inliers_exceeds_budget() -> None:
    """A two-inlier lock (anchor plus one chance neighbor) is not significant."""
    expectation = _wide_offset_false_lock_expectation(
        n_inliers=2,
        n_seeds=15,
        n_catalog=18,
        n_detected=30,
        tolerance_px=2.0,
        image_area_px2=1500.0 * 1500.0,
    )
    assert expectation > 0.01


def test_wide_offset_expectation_grows_with_seed_count() -> None:
    """More seed trials raises the chance-lock expectation monotonically."""
    few = _wide_offset_false_lock_expectation(
        n_seeds=5,
        n_inliers=3,
        n_catalog=18,
        n_detected=30,
        tolerance_px=2.0,
        image_area_px2=2.25e6,
    )
    many = _wide_offset_false_lock_expectation(
        n_seeds=50,
        n_inliers=3,
        n_catalog=18,
        n_detected=30,
        tolerance_px=2.0,
        image_area_px2=2.25e6,
    )
    assert many > few


def test_wide_offset_expectation_degenerate_inputs_are_zero() -> None:
    """Non-positive or under-determined inputs return a zero expectation."""
    assert (
        _wide_offset_false_lock_expectation(
            n_inliers=0,
            n_seeds=5,
            n_catalog=18,
            n_detected=30,
            tolerance_px=2.0,
            image_area_px2=2.25e6,
        )
        == 0.0
    )
    assert (
        _wide_offset_false_lock_expectation(
            n_inliers=3,
            n_seeds=5,
            n_catalog=1,
            n_detected=30,
            tolerance_px=2.0,
            image_area_px2=2.25e6,
        )
        == 0.0
    )


def test_star_photometry_saturated_reads_flag(
    make_star_feature: NavFeatureFactory,
) -> None:
    """The helper reads the ``#284`` saturation flag off a STAR feature."""
    feature = make_star_feature('star:UCAC4:0', predicted_vu=(10.0, 10.0), predicted_snr=40.0)
    assert _star_photometry_saturated(feature) is False
    flagged = replace(
        feature, flags=replace(cast(StarFlags, feature.flags), photometry_saturated=True)
    )
    assert _star_photometry_saturated(flagged) is True


# ---------------------------------------------------------------------------
# Wide-offset asterism matching — technique-level.
# ---------------------------------------------------------------------------


def _wide_offset_field(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    *,
    offset: tuple[float, float],
    n_stars: int,
) -> tuple[np.ndarray, list[NavFeature]]:
    """Plant ``n_stars`` aligned stars at a wide ``offset`` on a 400 px field."""
    centers = [
        (70.0, 90.0),
        (300.0, 140.0),
        (160.0, 330.0),
        (240.0, 250.0),
        (110.0, 210.0),
    ][:n_stars]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(400, 400), peak_dn=400.0)
    snrs = [400.0, 120.0, 40.0, 20.0, 12.0][:n_stars]
    features = _make_star_field_features(
        centers, make_feature=make_star_feature, offset=offset, snrs=snrs
    )
    return image, features


def test_wide_offset_lock_recovers_sparse_three_star_field(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Three aligned stars at a wide offset lock via the asterism fallback."""
    planted = (55.0, -70.0)
    image, features = _wide_offset_field(
        make_star_feature, draw_gaussian_star, offset=planted, n_stars=3
    )
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.wide_offset_lock is True
    assert result.diagnostics.n_inliers == 3
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)


def test_wide_offset_lock_confidence_is_capped(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A wide-offset lock cannot exceed the configured confidence cap."""
    image, features = _wide_offset_field(
        make_star_feature, draw_gaussian_star, offset=(48.0, 60.0), n_stars=3
    )
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.confidence <= technique._wide_offset_confidence_cap + 1e-9


def test_wide_offset_lock_expectation_recorded(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The accepted lock records a positive, sub-budget false-lock expectation."""
    image, features = _wide_offset_field(
        make_star_feature, draw_gaussian_star, offset=(40.0, 52.0), n_stars=3
    )
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    exp = result.diagnostics.wide_offset_false_lock_expectation
    assert 0.0 < exp <= technique._wide_offset_false_lock_budget


def test_dense_field_uses_strong_tier_not_wide_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A six-plus-inlier field is a strong-tier match, not a wide-offset lock."""
    centers = [
        (60.0, 80.0),
        (110.0, 130.0),
        (180.0, 90.0),
        (220.0, 200.0),
        (90.0, 240.0),
        (250.0, 70.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    features = _make_star_field_features(
        centers, make_feature=make_star_feature, offset=(3.0, -2.5)
    )
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(32, 32))
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.wide_offset_lock is False
    assert result.diagnostics.n_inliers >= 6


def test_wide_offset_rejects_untrusted_bright_anchor(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """With every bright anchor photometry-saturated, no lock is trusted."""
    planted = (50.0, -60.0)
    image, features = _wide_offset_field(
        make_star_feature, draw_gaussian_star, offset=planted, n_stars=3
    )
    # Flag every star's brightness as an untrusted (saturated) lower bound so
    # the matcher has no trustworthy anchor to seed on.
    features = [
        replace(f, flags=replace(cast(StarFlags, f.flags), photometry_saturated=True))
        for f in features
    ]
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True


def test_wide_offset_rejects_incoherent_field(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Detections with no rigid alignment to the catalog do not lock."""
    # Three planted detections, but predictions scattered so that no single
    # translation aligns more than the seed anchor pair.
    image = _star_field_image(
        [(70.0, 90.0), (300.0, 140.0), (160.0, 330.0)],
        draw=draw_gaussian_star,
        shape=(400, 400),
        peak_dn=400.0,
    )
    features = [
        make_star_feature('star:UCAC4:0', predicted_vu=(30.0, 40.0), predicted_snr=400.0),
        make_star_feature('star:UCAC4:1', predicted_vu=(360.0, 60.0), predicted_snr=120.0),
        make_star_feature('star:UCAC4:2', predicted_vu=(40.0, 360.0), predicted_snr=40.0),
    ]
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True


def _one_star_wide_field(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    *,
    offset: tuple[float, float],
    second_detection_dn: float = 120.0,
) -> tuple[np.ndarray, list[NavFeature]]:
    """Plant one dominant star at a wide offset plus two off-pattern detections.

    Three predictable catalog stars and three detections satisfy the
    technique's minimum cohort, but only the bright anchor pairs (the two
    faint detections align with neither faint catalog star under the
    anchor's offset), so nothing above a single inlier can lock.
    ``second_detection_dn`` controls whether the brightest detection is
    uniquely dominant.
    """
    shape = (400, 400)
    image = np.zeros(shape, dtype=np.float64)
    anchor_pred = (40.0, 50.0)
    anchor_det = (anchor_pred[0] + offset[0], anchor_pred[1] + offset[1])
    draw_gaussian_star(image, anchor_det, peak_dn=400.0, sigma=1.2)
    draw_gaussian_star(image, (150.0, 150.0), peak_dn=second_detection_dn, sigma=1.2)
    draw_gaussian_star(image, (350.0, 350.0), peak_dn=120.0, sigma=1.2)
    features = [
        make_star_feature('star:UCAC4:anchor', predicted_vu=anchor_pred, predicted_snr=400.0),
        make_star_feature('star:UCAC4:faint1', predicted_vu=(210.0, 300.0), predicted_snr=30.0),
        make_star_feature('star:UCAC4:faint2', predicted_vu=(300.0, 210.0), predicted_snr=15.0),
    ]
    return image, features


def test_one_star_wide_offset_lock_recovers_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single uniquely-bright star locks a wide offset when enabled."""
    planted = (60.0, 80.0)
    image, features = _one_star_wide_field(make_star_feature, draw_gaussian_star, offset=planted)
    technique = StarFieldFromCatalogNav()
    technique._wide_offset_one_star_enabled = True
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.wide_offset_lock is True
    assert result.diagnostics.n_inliers == 1
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)


def test_one_star_wide_offset_lock_confidence_capped_low(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A one-star lock is capped below the >=3-inlier wide-offset cap."""
    image, features = _one_star_wide_field(
        make_star_feature, draw_gaussian_star, offset=(60.0, 80.0)
    )
    technique = StarFieldFromCatalogNav()
    technique._wide_offset_one_star_enabled = True
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.confidence <= technique._wide_offset_one_star_confidence_cap + 1e-9


def test_one_star_wide_offset_disabled_by_default(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """With the one-star lock off (the default), the same field stays failed."""
    image, features = _one_star_wide_field(
        make_star_feature, draw_gaussian_star, offset=(60.0, 80.0)
    )
    technique = StarFieldFromCatalogNav()
    assert technique._wide_offset_one_star_enabled is False
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True


def test_one_star_wide_offset_rejects_non_unique_detection(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A field with two comparably bright peaks fails the detection-margin gate.

    This is the C0059893300R regime (1694 vs 1240 DN): the brightest
    detection is not uniquely dominant, so pairing the anchor with it would
    be a guess and the lock is correctly refused.
    """
    image, features = _one_star_wide_field(
        make_star_feature, draw_gaussian_star, offset=(60.0, 80.0), second_detection_dn=400.0
    )
    technique = StarFieldFromCatalogNav()
    technique._wide_offset_one_star_enabled = True
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True


def test_one_star_wide_offset_rejects_untrusted_anchor(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A photometry-saturated brightest catalog anchor is not one-star locked."""
    image, features = _one_star_wide_field(
        make_star_feature, draw_gaussian_star, offset=(60.0, 80.0)
    )
    features[0] = replace(
        features[0], flags=replace(cast(StarFlags, features[0].flags), photometry_saturated=True)
    )
    technique = StarFieldFromCatalogNav()
    technique._wide_offset_one_star_enabled = True
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True


def test_wide_offset_disabled_falls_through_to_spurious(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """With the fallback disabled a sub-floor field returns spurious."""
    image, features = _wide_offset_field(
        make_star_feature, draw_gaussian_star, offset=(45.0, -55.0), n_stars=3
    )
    technique = StarFieldFromCatalogNav()
    technique._wide_offset_enabled = False
    context = make_nav_context(image, extfov_margin_vu=(150, 150))
    result = technique.navigate(features, context)
    assert result.spurious is True
