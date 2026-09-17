"""End-to-end tests for ``RingEdgeNav``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    ArcPolylineFactory,
    CirclePolylineFactory,
    DiscImageFactory,
    FlatPolylineFactory,
    HorizontalStepImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.dt_fitting import CoarseSearchResult
from spindoctor.nav_technique.nav_technique_ring_edge import (
    RingEdgeNav,
    aggregate_edge_normal_angle_deg,
)
from spindoctor.nav_technique.ring_edge_geometry import _RANK1_NULL_RELATIVE_THRESHOLD


def test_ring_edge_nav_recovers_planted_offset_curved(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    cv = 100.0
    cu = 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    # Plant model 0.7 v-down, 1.3 u-right.
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    feature = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(0.7, abs=0.3)
    assert result.offset_px[1] == pytest.approx(1.3, abs=0.3)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is False
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    assert float(eigvals.min()) > 1.0e-12


def test_ring_edge_nav_returns_rank_1_for_all_flat_input(
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    # A simple half-image step: bright above row 100, dark below.  The
    # gradient peak (one pixel wide) sits at row 100, giving a clean,
    # single minimum that the LM can converge to.
    image = horizontal_step_image(shape, 100.0)
    # Polyline sits 1.5 px off the actual edge.  The LM should drive
    # the offset back to ~ -1.5 in v but only along the radial axis;
    # along the edge tangent (u) the cost has no slope, so the
    # information matrix is rank-1.
    vertices, outward = flat_polyline(101.5, 20.0, 180.0, 120)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    null_eigval = float(eigvals.min())
    observed_eigval = float(eigvals.max())
    assert observed_eigval > 0.0
    # Null direction is along the edge tangent (u axis); rank-deficient
    # covariance is expressed via the eigenvalue ratio.  Use the same
    # threshold the production code's ``_is_rank_1`` classifier uses, so
    # this test exercises the same boundary the technique itself draws
    # rather than a looser independent constant.
    assert (
        null_eigval == pytest.approx(0.0, abs=1.0e-9)
        or null_eigval / observed_eigval < _RANK1_NULL_RELATIVE_THRESHOLD
    )


def test_ring_edge_nav_mixed_curved_and_flat_full_rank(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (220, 220)
    cv_disc, cu_disc = 60.0, 110.0
    image = disc_image(shape, (cv_disc, cu_disc), 24.0)
    # Add a horizontal step below the disc so the two image features do
    # not overlap; the disc occupies rows 36..84 and the step lives at
    # row 150.
    bar_image = np.zeros(shape, dtype=np.float64)
    bar_image[150:152, 30:190] = 100.0
    image = np.clip(image + bar_image, 0.0, 100.0)
    # Both polylines share offset (-1.5, 0): curved center at
    # (cv_disc + 1.5, cu_disc) and flat polyline at v = 151 + 1.5.
    curved_v, curved_n = circle_polyline((cv_disc + 1.5, cu_disc), 24.0, 120)
    flat_v, flat_n = flat_polyline(152.5, 40.0, 180.0, 80)
    curved_feature = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    flat_feature = make_ring_feature(
        'flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    result = technique.navigate([curved_feature, flat_feature], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is False
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    assert float(eigvals.min()) > 1.0e-12


def test_ring_edge_nav_infeasible_on_empty_input() -> None:
    technique = RingEdgeNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_ring_edge_features' in report.reason


def test_ring_edge_nav_marks_spurious_when_per_edge_rms_collapses_to_one(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Per-edge RMS exposes a single-edge mis-convergence Tukey would mask.

    Models the production failure mode (Cassini Tethys N1572471790):
    three RING_EDGE features at distinct radii feed the LM, and the
    fit walks onto one of them while the other two contribute pure
    outliers.  Tukey rejects the outliers so ``rms_px`` is near zero
    even though the offset is far from the true joint solution; the
    per-edge sum is the only signal that recovers the correct
    spurious flag.

    The test forges an LM result whose post-Tukey ``rms_px`` is
    zero but whose per-vertex residuals are bimodal so that the
    per-edge RMS averages to a value far above the spurious
    threshold.  ``RingEdgeNav.navigate`` must then mark the result
    spurious so the ensemble can drop it.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 60)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 60)
    vertices_c, normals_c = circle_polyline((100.0, 100.0), 50.0, 60)
    feat_a = make_ring_feature(
        'inner', vertices=vertices_a, outward_normals=normals_a, is_straight_line=False
    )
    feat_b = make_ring_feature(
        'middle', vertices=vertices_b, outward_normals=normals_b, is_straight_line=False
    )
    feat_c = make_ring_feature(
        'outer', vertices=vertices_c, outward_normals=normals_c, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Bimodal residuals: edge A fits cleanly (~0 px), edges B and C are
    # off by ~50 px so their per-edge RMS is large.  Tukey reweights
    # the bad edges to zero — ``rms_px`` collapses to ~0 but the raw
    # per-edge sum makes the mis-convergence visible.
    n_total = vertices_a.shape[0] + vertices_b.shape[0] + vertices_c.shape[0]
    residuals = np.zeros(n_total, dtype=np.float64)
    residuals[vertices_a.shape[0] :] = 50.0
    weights = np.zeros(n_total, dtype=np.float64)
    weights[: vertices_a.shape[0]] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-27.0, -18.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.0,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=vertices_a.shape[0],
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feat_a, feat_b, feat_c], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    # The forged fit anchors only edge A — one third of the model
    # vertices — so the inlier fraction sits far below the 0.5 gate.
    # The per-edge diagnostics record the misalignment signature.
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(50.0)
    assert result.diagnostics.per_edge_dt_rms_summed > 50.0
    assert result.spurious is True


def test_ring_edge_nav_multi_edge_with_undetected_dominant_edges_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Absent edges waive the aggregate inlier-fraction veto.

    Models the production false flag (Cassini C ring N1467344214): three
    RING_EDGE features feed the LM, one curved edge fits at sub-pixel
    RMS with every vertex an inlier, and the other two -- holding the
    majority of the model vertices -- are undetectable in the image:
    their vertices sit tens of pixels from every detected edge pixel and
    are fully Tukey-rejected.  The aggregate inlier fraction falls below
    the 0.5 gate even though the fused offset is correct.  Because the
    rejected edges are absent (median DT residual far above the waiver
    threshold) rather than mis-aligned against detected structure, the
    mis-convergence veto must be waived and the result kept; the
    per-edge diagnostics still record the missing edges' residual
    signature unchanged.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 120)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 120)
    vertices_c, normals_c = circle_polyline((100.0, 100.0), 50.0, 120)
    feat_a = make_ring_feature(
        'inner', vertices=vertices_a, outward_normals=normals_a, is_straight_line=False
    )
    feat_b = make_ring_feature(
        'middle', vertices=vertices_b, outward_normals=normals_b, is_straight_line=False
    )
    feat_c = make_ring_feature(
        'outer', vertices=vertices_c, outward_normals=normals_c, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Edge A fits at 0.2 px with every vertex an inlier; edges B and C
    # sit 17 and 46 px from every detected edge -- absent -- and are fully
    # Tukey-rejected.  Aggregate inlier fraction is 120 / 360 = 0.33,
    # below the 0.5 gate.
    n_per_edge = 120
    residuals = np.full(3 * n_per_edge, 0.2, dtype=np.float64)
    residuals[n_per_edge : 2 * n_per_edge] = 17.0
    residuals[2 * n_per_edge :] = 46.0
    weights = np.zeros(3 * n_per_edge, dtype=np.float64)
    weights[:n_per_edge] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=n_per_edge,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    # Pin the coarse seed next to the forged LM offset so the unrelated
    # LM-displacement gate stays quiet.
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, -1), score=1.0),
    )

    result = technique.navigate([feat_a, feat_b, feat_c], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    # The diagnostics still expose the missing edges (stats unchanged);
    # only the gate decision is robust to them.
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(46.0)
    assert result.spurious is False


def test_ring_edge_nav_rejected_edge_on_detected_structure_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A rejected edge lying ON a detected image edge blocks the waiver.

    Models the wrong-ring lock (Cassini Tethys N1572472169): the fused
    fit anchors one edge cleanly, but another rejected edge has a
    sub-pixel median DT residual -- it sits on a detected image edge the
    robust fit disagrees with.  That internal inconsistency is the
    mis-convergence signature the veto exists for, so the result must
    stay spurious even though one edge clears the per-edge gate.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 60)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 200)
    feat_a = make_ring_feature(
        'locked', vertices=vertices_a, outward_normals=normals_a, is_straight_line=False
    )
    feat_b = make_ring_feature(
        'disputed', vertices=vertices_b, outward_normals=normals_b, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Edge A: all 60 vertices inliers at 0.2 px.  Edge B: median DT
    # residual 0.4 px (it lies along detected structure) but only 60 of
    # its 200 vertices survive Tukey.  Aggregate inlier fraction is
    # 120 / 260 = 0.46, below the 0.5 gate; edge B is not well-fit and
    # not absent, so the veto must stand.
    residuals = np.full(260, 0.4, dtype=np.float64)
    residuals[:60] = 0.2
    weights = np.zeros(260, dtype=np.float64)
    weights[:120] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=120,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, -1), score=1.0),
    )

    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is True


def test_ring_edge_nav_rank1_well_fit_subset_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The waiver is rank-aware: a rank-1 constrained fit cannot carry it.

    When the surviving vertices constrain only one offset axis (the
    translation covariance is rank-1), the well-fit subset cannot vouch
    for the full 2-D offset on its own; the aggregate inlier-fraction
    veto must stand even when the rejected curved edge is absent from
    the image.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    flat_v, flat_n = flat_polyline(130.5, 20.0, 180.0, 60)
    curved_v, curved_n = circle_polyline((100.0, 100.0), 50.0, 200)
    flat_feat = make_ring_feature(
        'flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True
    )
    curved_feat = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # The straight edge fits cleanly (60 / 60); the curved edge is
    # absent (median 46 px, zero inliers).  Aggregate fraction is
    # 60 / 260 = 0.23, below the gate; the surviving vertices only
    # constrain the normal axis (rank-1 covariance), so the waiver
    # must not fire.
    residuals = np.full(260, 0.2, dtype=np.float64)
    residuals[60:] = 46.0
    weights = np.zeros(260, dtype=np.float64)
    weights[:60] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=60,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, -1), score=1.0),
    )

    result = technique.navigate([flat_feat, curved_feat], context)
    assert result.spurious is True


def test_ring_edge_nav_marks_spurious_when_every_edge_fits_poorly(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """No edge clears the per-edge gate, so the fraction veto stands.

    A genuinely mis-converged multi-edge fit where every edge retains
    only a few stray inliers must stay spurious: the well-fit-edge
    quorum is zero, so the aggregate inlier-fraction veto applies
    exactly as before the absent-edge exemption.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    features = []
    for name, radius in (('inner', 30.0), ('middle', 40.0), ('outer', 50.0)):
        vertices, normals = circle_polyline((100.0, 100.0), radius, 60)
        features.append(
            make_ring_feature(
                name, vertices=vertices, outward_normals=normals, is_straight_line=False
            )
        )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Every edge keeps only 5 of its 60 vertices as inliers: aggregate
    # fraction 15 / 180 = 0.083 and no edge reaches the 0.5 per-edge
    # gate (nor the 6-inlier per-edge minimum).
    n_per_edge = 60
    residuals = np.full(3 * n_per_edge, 25.0, dtype=np.float64)
    weights = np.zeros(3 * n_per_edge, dtype=np.float64)
    for edge_index in range(3):
        start = edge_index * n_per_edge
        residuals[start : start + 5] = 0.2
        weights[start : start + 5] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=15,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, -1), score=1.0),
    )

    result = technique.navigate(features, context)
    assert result.spurious is True


def test_ring_edge_nav_single_edge_low_inlier_fraction_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A single-edge fit below the inlier-fraction gate is spurious, unchanged.

    One edge can never reach the two-edge well-fit quorum, so the absent-edge
    exemption never applies to a single-edge fit: the aggregate
    inlier-fraction gate behaves exactly as before.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, normals = circle_polyline((100.0, 100.0), 30.0, 60)
    feature = make_ring_feature(
        'only', vertices=vertices, outward_normals=normals, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # 10 of 60 vertices survive Tukey: fraction 0.167, below the gate.
    residuals = np.full(60, 25.0, dtype=np.float64)
    residuals[:10] = 0.2
    weights = np.zeros(60, dtype=np.float64)
    weights[:10] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=10,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(1, -1), score=1.0),
    )

    result = technique.navigate([feature], context)
    assert result.spurious is True


def test_ring_edge_nav_flat_parallel_edges_with_minority_snaps_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A correct flat multi-edge fit with an outlier minority passes the gate.

    Models the ``ring_only_flat`` production regression (#203, Cassini
    N1863267799): parallel straight Keeler-gap edges fit cleanly while a
    minority of vertices (an edge too faint to detect, vertices snapping
    to a neighboring parallel edge 9-30 px away) are Tukey outliers.
    Those outliers inflate every raw per-edge residual statistic past any
    sigma-derived threshold even though the joint fit is correct — which is
    exactly what used to gate every flat ansa frame as spurious.  The fit
    still anchors 85% of the model vertices, so the inlier-fraction gate
    passes and the result survives with ``is_rank_1=True`` for the ensemble
    to surface as ``rank_1_only``.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    features = []
    for name, row in (('inner', 80.5), ('middle', 100.5), ('outer', 120.5)):
        vertices, outward = flat_polyline(row, 20.0, 180.0, 60)
        features.append(
            make_ring_feature(
                name, vertices=vertices, outward_normals=outward, is_straight_line=True
            )
        )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Per-edge residuals: 85% of vertices at the ~1 px fit residual, 15%
    # snapped to a parallel neighbor 20 px away.  Raw per-edge RMS is
    # sqrt(0.85*1 + 0.15*400) ~ 7.8 px — any sigma-derived residual gate
    # would fire — but the inlier fraction is 0.85, well above the gate.
    n_per_edge = 60
    per_edge = np.full(n_per_edge, 1.0, dtype=np.float64)
    per_edge[: int(0.15 * n_per_edge)] = 20.0
    residuals = np.concatenate([per_edge] * 3)
    weights = np.ones(residuals.size, dtype=np.float64)
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-1.5, 0.0),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=weights,
        rms_px=1.0,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=int(residuals.size * 0.85),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    # Pin the coarse seed next to the forged LM offset so the unrelated
    # LM-displacement gate stays quiet; this test is about the per-edge gate.
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(-2, 0), score=1.0),
    )

    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(1.0)
    assert result.diagnostics.per_edge_dt_rms_mean > 3.0
    assert result.spurious is False
    assert result.diagnostics.is_rank_1 is True


def test_ring_edge_nav_rank1_tangent_slide_not_gated(
    monkeypatch: pytest.MonkeyPatch,
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Tangent slide on a rank-1 scene trips neither at_edge nor displacement.

    Nothing constrains the along-edge axis of an all-straight fit, so the
    LM may drift to the search-window boundary along the tangent (Cassini
    N1863267979: dv slid to the margin while the observable normal
    component was mid-window).  Both the at-edge check and the
    LM-displacement spurious gate must therefore be evaluated on the
    edge-normal component only.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    vertices, outward = flat_polyline(101.5, 20.0, 180.0, 60)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)  # extfov margins (32, 32)

    # Horizontal edge: normal is +v, tangent is +u.  The forged fit slid
    # 31.5 px along the tangent — at the (32 - 1) px per-axis at-edge
    # boundary and far past the 4 px displacement gate — while the
    # observable normal component stays a benign -1.5 px.
    residuals = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-1.5, 31.5),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=np.ones(residuals.size, dtype=np.float64),
        rms_px=0.5,
        raw_rms_px=0.5,
        iterations=10,
        converged=True,
        inlier_count=int(residuals.size),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search_scored',
        lambda *_args, **_kwargs: CoarseSearchResult(offset_vu=(-2, 0), score=1.0),
    )

    result = technique.navigate([feature], context)
    assert result.at_edge is False
    assert result.spurious is False
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True


def test_aggregate_edge_normal_angle_all_straight_horizontal(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """Horizontal straight edges have a +v-aligned normal: angle 0 deg."""
    vertices, outward = flat_polyline(100.5, 20.0, 180.0, 60)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    angle = aggregate_edge_normal_angle_deg([feature])
    assert angle is not None
    assert angle == pytest.approx(0.0, abs=1.0e-6)


def test_aggregate_edge_normal_angle_polarity_sign_independent(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """Two parallel edges with opposite normal senses do not cancel."""
    vertices_a, outward_a = flat_polyline(80.5, 20.0, 180.0, 60)
    vertices_b, outward_b = flat_polyline(120.5, 20.0, 180.0, 60)
    feat_a = make_ring_feature(
        'inner', vertices=vertices_a, outward_normals=outward_a, is_straight_line=True
    )
    feat_b = make_ring_feature(
        'outer', vertices=vertices_b, outward_normals=-outward_b, is_straight_line=True
    )
    angle = aggregate_edge_normal_angle_deg([feat_a, feat_b])
    assert angle is not None
    assert angle == pytest.approx(0.0, abs=1.0e-6)


def test_aggregate_edge_normal_angle_none_when_any_edge_curved(
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """A mixed straight + curved scene is full-rank: no constraint seed."""
    curved_v, curved_n = circle_polyline((100.0, 100.0), 30.0, 60)
    flat_v, flat_n = flat_polyline(150.5, 20.0, 180.0, 60)
    curved = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    flat = make_ring_feature('flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True)
    assert aggregate_edge_normal_angle_deg([curved, flat]) is None


def test_aggregate_edge_normal_angle_none_without_ring_edges() -> None:
    """No ring-edge features means no seed."""
    assert aggregate_edge_normal_angle_deg([]) is None


def test_ring_edge_nav_registered_with_navtechnique_registry() -> None:
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert RingEdgeNav in NavTechnique._registry


def test_ring_edge_nav_raises_when_navcontext_lacks_derivatives(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image = disc_image(shape, (80.0, 80.0), 25.0)
    vertices, outward = circle_polyline((80.0, 80.0), 25.0, 80)
    feature = make_ring_feature(
        'mid', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    bare_context = NavContext(
        obs=context.obs,
        image_ext=context.image_ext,
        sensor_mask_ext=context.sensor_mask_ext,
        image_noise_sigma=context.image_noise_sigma,
        saturation_mask_ext=context.saturation_mask_ext,
        cosmic_ray_mask_ext=context.cosmic_ray_mask_ext,
        image_classifier=context.image_classifier,
        provenance=context.provenance,
    )
    with pytest.raises(RuntimeError, match='image_edge_dt_ext'):
        technique.navigate([feature], bare_context)


def test_ring_edge_nav_3dof_emits_3x3_covariance(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A curved ring edge with ``fit_camera_rotation=True`` produces a 3x3 covariance."""
    shape = (200, 200)
    cv = 100.0
    cu = 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    feature = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.sigma_rotation_rad is not None
    # No rotation planted; the LM should converge to near zero.  Allow
    # up to 3 sigma of the reported rotation uncertainty as the
    # tolerance so tighter sigma estimates also tighten the test.
    assert np.isclose(result.rotation_rad, 0.0, atol=3.0 * result.sigma_rotation_rad)


def test_ring_edge_nav_3dof_flat_edge_remains_rank_deficient(
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An all-flat scene under fit_camera_rotation reports rank-1 in the translation block.

    The 3x3 covariance is rank-deficient on multiple axes (along-edge
    plus rotation when geometry is uninformative); the technique still
    flags ``is_rank_1`` based on the 2x2 translation block so the
    diagnostic stays comparable to the 2-DoF case.
    """
    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    vertices, outward = flat_polyline(99.5, 30.0, 170.0, 200)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True


# ---------------------------------------------------------------------------
# Radial orbit-uncertainty channel
# ---------------------------------------------------------------------------


def test_ring_edge_orbit_sigma_widens_partial_arc_along_its_radial_axis(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """On a partial arc the declared sigma lands on the arc's radial axis.

    A short arc's normals are nearly parallel, so a coherent radial
    displacement is absorbed into the translation essentially one-for-one:
    the covariance difference is the rank-1 outer product of that radial
    direction with eigenvalue ``sigma_orbit**2``.
    """
    shape = (200, 200)
    cv, cu = 100.0, 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = arc_polyline((cv - 0.7, cu - 1.3), radius, 60, -0.4, 0.4)
    context = make_nav_context(image)
    technique = RingEdgeNav()
    plain = make_ring_feature(
        'arc', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    declared = make_ring_feature(
        'arc',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        sigma_orbit_radial_px=2.0,
    )
    result_plain = technique.navigate([plain], context)
    result_declared = technique.navigate([declared], context)
    diff = np.asarray(result_declared.covariance_px2) - np.asarray(result_plain.covariance_px2)
    eigvals = np.linalg.eigvalsh(diff)
    # The arc absorbs the displacement essentially one-for-one along its own
    # radial axis, and slightly MORE than one-for-one: a single translation
    # overshoots the middle of an arc to reduce the error at its ends, so the
    # derived sensitivity runs a little above 1 and the major eigenvalue a
    # little above sigma_orbit**2 = 4.0.
    assert float(eigvals.max()) == pytest.approx(4.22, rel=0.02)
    # With the sensitivity at or above 1 the isotropic complement is zero, so
    # the perpendicular axis takes nothing.
    assert float(eigvals.min()) == pytest.approx(0.0, abs=1.0e-9)


def test_ring_edge_orbit_sigma_widens_full_annulus_isotropically(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A closed annulus reports the orbit bound on BOTH axes, not one.

    A uniform semimajor-axis error dilates a closed ring rather than
    translating it, so the linearized fit absorbs almost none of it and no
    radial axis is distinguished.  The nonlinear acquisition can still lock
    onto a translation, and which direction it picks is exactly what cannot
    be predicted, so the inflation becomes isotropic: both axes carry the
    full ``sigma_orbit**2``.

    This is the regression guard for the earlier construction, which took the
    dominant eigenvector of the normals' outer-product sum.  On a closed ring
    that eigen-decomposition is degenerate (the two eigenvalues agree to ~1%,
    so the axis is set by rounding), and it widened one arbitrary axis by the
    full variance while leaving the perpendicular axis untouched -- letting a
    frame the channel should demote still pass the tier gate through the
    un-widened axis.  Both eigenvalues must now be inflated.
    """
    shape = (200, 200)
    cv, cu = 100.0, 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    context = make_nav_context(image)
    technique = RingEdgeNav()
    plain = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    declared = make_ring_feature(
        'outer',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        sigma_orbit_radial_px=2.0,
    )
    result_plain = technique.navigate([plain], context)
    result_declared = technique.navigate([declared], context)
    diff = np.asarray(result_declared.covariance_px2) - np.asarray(result_plain.covariance_px2)
    eigvals = np.linalg.eigvalsh(diff)
    # No axis is left un-widened: the SMALLEST eigenvalue of the added term
    # still carries essentially the whole sigma**2 = 4.0 px^2.
    assert float(eigvals.min()) == pytest.approx(4.0, rel=0.05)
    # And no axis is double-counted beyond the bound.
    assert float(eigvals.max()) == pytest.approx(4.0, rel=0.05)


def test_ring_edge_orbit_sigma_offset_unchanged(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The channel widens the reported uncertainty; it never moves the fit."""
    shape = (200, 200)
    cv, cu = 100.0, 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    context = make_nav_context(image)
    technique = RingEdgeNav()
    plain = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    declared = make_ring_feature(
        'outer',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        sigma_orbit_radial_px=2.0,
    )
    result_plain = technique.navigate([plain], context)
    result_declared = technique.navigate([declared], context)
    assert result_declared.offset_px[0] == pytest.approx(result_plain.offset_px[0])
    assert result_declared.offset_px[1] == pytest.approx(result_plain.offset_px[1])


def test_ring_edge_orbit_sigma_rank1_scene_stays_rank1(
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """On an all-straight scene the inflation follows the projection axis.

    The rank-1 projected covariance must stay exactly singular along the
    tangent after the orbit inflation (the term is added along the same
    aggregate normal the projection used), with the normal-axis variance
    increased by ``sigma_orbit**2``.
    """
    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    vertices, outward = flat_polyline(101.5, 20.0, 180.0, 120)
    context = make_nav_context(image)
    technique = RingEdgeNav()
    plain = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    declared = make_ring_feature(
        'flat',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=True,
        sigma_orbit_radial_px=2.0,
    )
    result_plain = technique.navigate([plain], context)
    result_declared = technique.navigate([declared], context)
    eig_declared = np.linalg.eigvalsh(result_declared.covariance_px2)
    null_eigval = float(eig_declared.min())
    observed_eigval = float(eig_declared.max())
    assert (
        null_eigval == pytest.approx(0.0, abs=1.0e-9)
        or null_eigval / observed_eigval < _RANK1_NULL_RELATIVE_THRESHOLD
    )
    eig_plain = np.linalg.eigvalsh(result_plain.covariance_px2)
    assert observed_eigval == pytest.approx(float(eig_plain.max()) + 4.0, rel=1.0e-6)


def test_ring_edge_orbit_sigma_recorded_in_diagnostics(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    cv, cu = 100.0, 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = circle_polyline((cv, cu), radius, 120)
    feature = make_ring_feature(
        'outer',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        sigma_orbit_radial_px=1.75,
    )
    technique = RingEdgeNav()
    result = technique.navigate([feature], make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.sigma_orbit_radial_px == pytest.approx(1.75)


def test_effective_orbit_sigma_is_weight_weighted_mean(
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """Two features' sigmas combine by their share of the final LM weight."""
    from spindoctor.nav_technique.ring_edge_geometry import _effective_orbit_sigma_px

    vertices_a, normals_a = circle_polyline((50.0, 50.0), 20.0, 10)
    vertices_b, normals_b = circle_polyline((50.0, 50.0), 30.0, 10)
    feat_a = make_ring_feature(
        'a',
        vertices=vertices_a,
        outward_normals=normals_a,
        is_straight_line=False,
        sigma_orbit_radial_px=1.0,
    )
    feat_b = make_ring_feature(
        'b',
        vertices=vertices_b,
        outward_normals=normals_b,
        is_straight_line=False,
        sigma_orbit_radial_px=3.0,
    )
    # Feature a carries 3x the total weight of feature b.
    weights = np.concatenate([np.full(10, 3.0), np.full(10, 1.0)])
    sigma = _effective_orbit_sigma_px([feat_a, feat_b], weights)
    assert sigma == pytest.approx((30.0 * 1.0 + 10.0 * 3.0) / 40.0)


def test_effective_orbit_sigma_zero_weights_returns_max(
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """A degenerate (all-zero-weight) fit falls back to the conservative max."""
    from spindoctor.nav_technique.ring_edge_geometry import _effective_orbit_sigma_px

    vertices, normals = circle_polyline((50.0, 50.0), 20.0, 10)
    feat = make_ring_feature(
        'a',
        vertices=vertices,
        outward_normals=normals,
        is_straight_line=False,
        sigma_orbit_radial_px=2.5,
    )
    sigma = _effective_orbit_sigma_px([feat], np.zeros(10))
    assert sigma == pytest.approx(2.5)


def test_effective_orbit_sigma_zero_when_undeclared(
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    from spindoctor.nav_technique.ring_edge_geometry import _effective_orbit_sigma_px

    vertices, normals = circle_polyline((50.0, 50.0), 20.0, 10)
    feat = make_ring_feature(
        'a', vertices=vertices, outward_normals=normals, is_straight_line=False
    )
    assert _effective_orbit_sigma_px([feat], np.ones(10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coarse-acquisition gate, end to end through navigate()
# ---------------------------------------------------------------------------


def test_ring_edge_nav_spurious_when_coarse_acquisition_finds_no_lock(
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A model with no matching image edge is rejected by the coarse gate.

    Drives the real acquisition (no monkeypatched score): the image carries a
    small disc far from the model ring, so no integer shift in the search
    window puts an appreciable fraction of the model's rasterized polyline on
    a detected edge pixel.  The fit has nothing to refine, and the technique
    must self-flag rather than report the LM's polish of noise.
    """
    shape = (300, 300)
    # A small bright disc in one corner: real edges exist (so the DT is not
    # degenerate) but almost none of them lie under the long model ring.
    image = np.zeros(shape, dtype=np.float64)
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    image[np.hypot(vs - 30.0, us - 30.0) <= 6.0] = 100.0
    vertices, outward = circle_polyline((170.0, 170.0), 80.0, 600)
    feature = make_ring_feature(
        'orphan', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    result = technique.navigate([feature], make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.coarse_peak_fraction < 0.05
    assert result.spurious is True


def test_ring_edge_nav_healthy_scene_clears_the_coarse_gate(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The coarse gate stays quiet on a normal acquisition (margin check)."""
    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 32.0)
    vertices, outward = circle_polyline((99.3, 98.7), 32.0, 120)
    feature = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    result = technique.navigate([feature], make_nav_context(image))
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.coarse_peak_fraction > 0.05
    assert result.spurious is False


def test_ring_edge_nav_fits_sub_pixel_when_the_model_sigma_is_tiny(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A catalog sigma far below a pixel does not collapse the robust scale.

    Ring orbits are solved to a fraction of a km, so on a coarse frame an
    edge's per-vertex sigma is thousandths of a pixel.  The fit measures
    distance to a binary edge mask quantized to the integer grid, so a
    residual scale that small leaves the Tukey redescender keeping only the
    vertices sitting exactly on a mask pixel -- reachable only at an integer
    offset.  Combining the catalog sigma with the mask's own localization
    scale keeps the sub-pixel plant recoverable.
    """
    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 32.0)
    vertices, outward = circle_polyline((100.0 - 0.4, 100.0 + 0.6), 32.0, 120)
    feature = make_ring_feature(
        'outer',
        vertices=vertices,
        outward_normals=outward,
        is_straight_line=False,
        sigma_radial_px=0.001,
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.offset_px[0] == pytest.approx(0.4, abs=0.05)
    assert result.offset_px[1] == pytest.approx(-0.6, abs=0.05)
    assert result.spurious is False


def test_ring_edge_nav_answer_does_not_depend_on_a_sub_mask_catalog_sigma(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Shrinking the catalog sigma below the mask scale changes nothing.

    The same scene fitted with a 0.5 px catalog sigma and with a 0.001 px one
    must return the same offset and a comparable reported sigma: below the
    mask's own localization scale the catalog number carries no information
    the evidence can support, so it must not move either the answer or the
    uncertainty claimed for it.
    """
    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 32.0)
    vertices, outward = circle_polyline((100.0 - 0.4, 100.0 + 0.6), 32.0, 120)
    context = make_nav_context(image)
    results = []
    for sigma_radial_px in (0.5, 0.001):
        feature = make_ring_feature(
            'outer',
            vertices=vertices,
            outward_normals=outward,
            is_straight_line=False,
            sigma_radial_px=sigma_radial_px,
        )
        results.append(RingEdgeNav().navigate([feature], context))
    coarse, fine = results
    assert fine.offset_px[0] == pytest.approx(coarse.offset_px[0], abs=0.01)
    assert fine.offset_px[1] == pytest.approx(coarse.offset_px[1], abs=0.01)
    sigma_coarse = float(np.sqrt(coarse.covariance_px2[0, 0]))
    sigma_fine = float(np.sqrt(fine.covariance_px2[0, 0]))
    assert sigma_fine == pytest.approx(sigma_coarse, rel=0.5)


def test_ring_edge_nav_rejects_a_negative_edge_localization_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative localization sigma fails at construction, not silently.

    ``hypot`` squares its arguments, so a negative value would behave as its
    absolute value and the misconfiguration would never surface.
    """
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['edge_localization_sigma_px'] = -0.5
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='edge_localization_sigma_px must be a finite number > 0'):
        RingEdgeNav()


def test_ring_edge_nav_rejects_a_zero_edge_localization_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero localization sigma fails at construction.

    Zero is physically meaningless against a half-pixel-quantized edge mask
    and lets a zero catalog sigma reach the LM refine as a zero residual
    scale.
    """
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['edge_localization_sigma_px'] = 0.0
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='edge_localization_sigma_px must be a finite number > 0'):
        RingEdgeNav()


def test_ring_edge_nav_rejects_a_null_edge_localization_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric localization sigma (YAML null) raises ValueError.

    The error names the config key rather than surfacing as a bare
    ``TypeError`` from ``float(None)`` far from the config that caused it.
    """
    bad_tuning: dict[str, Any] = dict(RingEdgeNav.tuning)
    bad_tuning['edge_localization_sigma_px'] = None
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='edge_localization_sigma_px must be a finite number > 0'):
        RingEdgeNav()


def test_ring_edge_nav_rejects_an_overflowing_edge_localization_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An int too large for a float raises ValueError naming the key.

    ``float()`` raises ``OverflowError`` for such an int, which would
    otherwise escape the documented ValueError contract.
    """
    bad_tuning: dict[str, Any] = dict(RingEdgeNav.tuning)
    bad_tuning['edge_localization_sigma_px'] = 10**10000
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='edge_localization_sigma_px must be a finite number > 0'):
        RingEdgeNav()
