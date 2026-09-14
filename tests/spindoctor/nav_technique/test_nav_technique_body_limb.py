"""End-to-end tests for ``BodyLimbNav``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    ArcPolylineFactory,
    CirclePolylineFactory,
    DiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.nav_technique_body_limb import BodyLimbNav

# Tests pull these values from the technique's loaded YAML tuning so a
# config edit retunes the assertions automatically.  The values must be
# read AFTER ``Config.read_config()`` runs (BodyLimbNav.__init__ calls
# it) so the class-level ``BodyLimbNav.tuning`` dict is populated.
BodyLimbNav()  # populates BodyLimbNav.tuning via Config.read_config
LIMB_MIN_ARC_PX = BodyLimbNav.tuning['min_arc_vertices']
SPURIOUS_MIN_INLIER_FRACTION = BodyLimbNav.tuning['spurious_min_inlier_fraction']
SPURIOUS_MAX_LM_DISPLACEMENT_PX = BodyLimbNav.tuning['spurious_max_lm_displacement_px']
SPURIOUS_DT_FLOOR_PX = BodyLimbNav.tuning['spurious_dt_floor_px']
SPURIOUS_DT_RMS_FACTOR = BodyLimbNav.tuning['spurious_dt_rms_factor']
SPURIOUS_UNCONVERGED_TRUST_BOUNDARY_FRACTION = BodyLimbNav.tuning[
    'spurious_unconverged_trust_boundary_fraction'
]
LM_TRUST_REGION_PX = BodyLimbNav.tuning['lm_trust_region_px']


def test_body_limb_nav_recovers_planted_offset_single_body(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    # Plant the model 1.5 px below and 2.5 px right of the actual disc:
    # the technique should report offset_px = (1.5, 2.5) so the model
    # gets shifted onto the image disc.
    model_center = (image_center[0] - 1.5, image_center[1] - 2.5)
    vertices, outward = circle_polyline(model_center, radius, 120)
    feature = make_limb_feature('moonA', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(1.5, abs=0.05)
    assert result.offset_px[1] == pytest.approx(2.5, abs=0.05)
    assert result.spurious is False
    assert result.at_edge is False
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    assert result.diagnostics.tukey_inlier_count == 120
    assert result.diagnostics.lm_iterations >= 1


def test_body_limb_nav_recovers_partial_arc(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    # 50 % visible arc starting at the right side of the body (angles
    # [-pi/2, pi/2]).
    model_center = (image_center[0] - 0.5, image_center[1] - 1.5)
    vertices, outward = arc_polyline(model_center, radius, 60, -np.pi / 2, np.pi / 2)
    feature = make_limb_feature(
        'moonB', vertices=vertices, outward_normals=outward, visible_arc_fraction=0.5
    )
    technique = BodyLimbNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(0.5, abs=0.1)
    assert result.offset_px[1] == pytest.approx(1.5, abs=0.1)
    assert result.confidence > 0.0


def test_body_limb_nav_recovers_multi_body_offset(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (240, 240)
    radius = 22.0
    centers = [(80.0, 80.0), (160.0, 90.0), (130.0, 170.0)]
    image = np.zeros(shape, dtype=np.float64)
    for cv, cu in centers:
        image += disc_image(shape, (cv, cu), radius)
    image = np.clip(image, 0.0, 100.0)
    planted_dv, planted_du = 1.0, -1.5
    features: list[NavFeature] = []
    for idx, (cv, cu) in enumerate(centers):
        model_center = (cv - planted_dv, cu - planted_du)
        vertices, outward = circle_polyline(model_center, radius, 80)
        features.append(
            make_limb_feature(f'moon_{idx}', vertices=vertices, outward_normals=outward)
        )
    technique = BodyLimbNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.05)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.05)
    # Single-body result on the same image, for covariance comparison.
    single_result = technique.navigate([features[0]], context)
    multi_diag_max = float(np.linalg.eigvalsh(result.covariance_px2).max())
    single_diag_max = float(np.linalg.eigvalsh(single_result.covariance_px2).max())
    assert multi_diag_max < single_diag_max


def test_body_limb_nav_infeasible_on_empty_input() -> None:
    technique = BodyLimbNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_limb_arc_features' in report.reason


def test_body_limb_nav_infeasible_on_short_arc(
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
) -> None:
    short_n = int(LIMB_MIN_ARC_PX) - 1
    vertices, outward = circle_polyline((50.0, 50.0), 12.0, short_n)
    feature = make_limb_feature('tiny_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False
    assert 'sufficient_visible_arc' in report.reason


def test_body_limb_nav_at_edge_when_offset_hits_search_window(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 25.0
    image = disc_image(shape, image_center, radius)
    margin_v, margin_u = 6, 6
    # Plant the model at exactly the search window boundary so LM can
    # only converge at the edge.
    model_center = (image_center[0] - float(margin_v), image_center[1] - float(margin_u))
    vertices, outward = circle_polyline(model_center, radius, 120)
    feature = make_limb_feature('atedge_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image, extfov_margin_vu=(margin_v, margin_u))
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    # Hard-zero gate via ``at_edge`` forces confidence to 0 per the spec.
    assert result.confidence == pytest.approx(0.0, abs=1e-12)


def test_body_limb_nav_at_edge_when_offset_walks_outside_search_window(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """LM converging at or beyond the extfov margin must register as at_edge.

    ``ObsSnapshot.extract_offset_array`` cannot honor an offset whose
    magnitude meets or exceeds the extfov margin without zero-filling
    part of the overlay slice.  The technique must flag at_edge=True so
    the ensemble can drop the result whenever an interior alternative
    exists, and so downstream consumers know the offset is at the edge
    of physically meaningful translations.

    The LM is unconstrained and may either stop at the boundary or
    walk past it; the regression we are guarding against is the
    earlier check that only fired when the offset was *near* the
    boundary within tolerance and missed offsets that overshot.
    """
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 25.0
    image = disc_image(shape, image_center, radius)
    margin_v, margin_u = 5, 10
    # Plant the model far enough from the image center that the
    # converged offset reaches the V-axis margin in either direction.
    model_center = (image_center[0] - 11.0, image_center[1] - 9.0)
    vertices, outward = circle_polyline(model_center, radius, 120)
    feature = make_limb_feature('past_edge_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image, extfov_margin_vu=(margin_v, margin_u))
    result = technique.navigate([feature], context)
    dv, _du = result.offset_px
    at_edge_tolerance_px = float(technique._at_edge_tolerance_px)
    # The fit converged at or past the V-axis margin (the regression case).
    assert abs(dv) >= float(margin_v) - at_edge_tolerance_px
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0, abs=1e-12)


def test_body_limb_nav_marks_spurious_when_image_lacks_limb(
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)  # No edge anywhere.
    image[10:20, 10:20] = 100.0  # a small unrelated bright square far from the model
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 120)
    feature = make_limb_feature('lonely_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True


def test_body_limb_nav_polarity_filters_wrong_polarity_vertices(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    model_center = (image_center[0] - 1.0, image_center[1] - 2.0)
    n_vertices = 120
    vertices, outward_normals = circle_polyline(model_center, radius, n_vertices)
    # Plant exactly half (60 of 120) the vertices with inverted outward
    # normals; the polarity filter must reject every one of them.  The
    # exact count is the load-bearing assertion: any silent change to the
    # polarity-rejection wiring (sign convention, Tukey constant,
    # ``_INFINITY_DT_PENALTY_PX`` substitute) shifts it.
    bad_indices = np.arange(0, n_vertices, 2)
    n_bad = int(bad_indices.size)
    n_good = n_vertices - n_bad
    outward_normals[bad_indices] *= -1.0
    feature = make_limb_feature('mixed_moon', vertices=vertices, outward_normals=outward_normals)
    technique = BodyLimbNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(1.0, abs=0.1)
    assert result.offset_px[1] == pytest.approx(2.0, abs=0.1)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    # Two complementary assertions so a failure points at which side broke:
    # the count must equal the planted ``n_good`` (no extra rejections from
    # convergence-tolerance drift, no extra acceptances from polarity-check
    # weakening).
    assert result.diagnostics.tukey_inlier_count <= n_good
    assert result.diagnostics.tukey_inlier_count == n_good


def test_body_limb_nav_registered_with_navtechnique_registry() -> None:
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert BodyLimbNav in NavTechnique._registry


def test_body_limb_nav_raises_when_navcontext_lacks_derivatives(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image = disc_image(shape, (80.0, 80.0), 25.0)
    vertices, outward = circle_polyline((80.0, 80.0), 25.0, 120)
    feature = make_limb_feature('moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)
    # Build a fresh context without the gradient / DT fields populated.
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


def test_body_limb_nav_marks_spurious_when_inlier_fraction_collapses(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An LM convergence that retains < 5 % of vertices as inliers is spurious.

    Models a real failure mode (Tethys N1716186428): the limb polyline
    finds a deep local minimum at internal-body crater rims, retaining a
    handful of inliers while rejecting the actual limb as outliers.  The
    fraction-based check rescues the ensemble from a wrong-offset
    moderate-confidence answer.

    The test is deterministic: ``lm_subpixel_refine`` is patched to
    return an LMResult whose inlier_count / vertex_count is below
    ``SPURIOUS_MIN_INLIER_FRACTION`` so the safety-net assertion is
    unconditionally exercised.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 1000)
    feature = make_limb_feature('confused_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Force a degenerate LM result: many vertices, very few inliers.
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(7.0, -3.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64),
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.zeros(vertices.shape[0], dtype=np.float64),
        rms_px=2.5,
        raw_rms_px=2.5,
        iterations=5,
        converged=True,
        inlier_count=10,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    inlier_fraction = result.diagnostics.tukey_inlier_count / float(vertices.shape[0])
    assert inlier_fraction < SPURIOUS_MIN_INLIER_FRACTION
    assert result.spurious is True


def test_body_limb_nav_3dof_emits_3x3_covariance(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """With ``fit_camera_rotation=True`` the result carries a 3x3 covariance + rotation."""
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    model_center = (image_center[0] - 1.0, image_center[1] - 1.0)
    vertices, outward = circle_polyline(model_center, radius, 120)
    feature = make_limb_feature('moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.sigma_rotation_rad is not None
    # No rotation planted; LM converges to within a small fraction of a
    # degree of zero on a clean disc (well below the 5 deg cap).
    assert abs(result.rotation_rad) < np.deg2rad(0.5)


def test_body_limb_nav_3dof_at_edge_when_rotation_saturates(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotation magnitude near the configured cap raises ``at_edge``."""
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 120)
    feature = make_limb_feature('moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    max_rotation_deg = 5.0
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=max_rotation_deg)

    # Force a converged LM rotation right at the configured at-edge
    # fraction of the cap so the test stays valid if calibration retunes
    # ``rotation_at_edge_fraction`` per technique.
    rotation_fraction = float(BodyLimbNav.tuning['rotation_at_edge_fraction'])
    forged_rotation_rad = float(np.deg2rad(rotation_fraction * max_rotation_deg))
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.0, 0.0),
        rotation_rad=forged_rotation_rad,
        covariance=np.diag([0.04, 0.04, 1.0e-4]).astype(np.float64),
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.1,
        raw_rms_px=0.1,
        iterations=5,
        converged=True,
        inlier_count=int(vertices.shape[0]),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.rotation_rad == pytest.approx(forged_rotation_rad)


def test_body_limb_nav_marks_spurious_when_lm_walks_far_from_coarse_seed(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """LM that walks multi-pixel from the integer coarse seed is spurious.

    Models the production failure mode (Cassini Tethys N1574928113):
    the coarse NCC search found the integer-precision basin (-1, 2)
    correctly, but the LM followed a DT gradient out of that basin
    onto a crater rim and converged at (6.66, 6.48) — a 9 px walk.
    The result retained > 50 % of vertices as inliers, so the
    inlier-fraction guard could not catch it.  The displacement
    guard caps trustworthy LM motion to a few pixels around the
    coarse seed.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    # Plant the model centered on the image — coarse NCC will report
    # offset (0, 0) for the test setup.
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 200)
    feature = make_limb_feature('walked_off', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Healthy inlier fraction and low RMS, but LM moved far from the
    # coarse seed; only the displacement guard should fire.
    forged_displacement = float(SPURIOUS_MAX_LM_DISPLACEMENT_PX) + 5.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(forged_displacement, 0.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.4,
        raw_rms_px=0.4,
        iterations=8,
        converged=True,
        inlier_count=int(vertices.shape[0] * 0.6),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    inlier_fraction = result.diagnostics.tukey_inlier_count / float(vertices.shape[0])
    # Sanity-check the test setup: inlier-fraction guard would NOT fire.
    assert inlier_fraction >= SPURIOUS_MIN_INLIER_FRACTION
    # The LM displacement was forced beyond the configured threshold.
    assert result.spurious is True


def test_body_limb_nav_marks_spurious_when_unconverged_at_trust_boundary(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An unconverged LM pinned at the trust-region boundary is spurious.

    Models the low-phase coarse-seed mis-lock: the coarse polarity search
    seeds the wrong basin, the trust region pins the LM against its
    boundary while it tries to escape, and it exits at the iteration cap
    without meeting the step tolerance.  The displacement stays inside the
    multi-pixel walk-off guard, the inliers stay healthy, and the RMS is
    low, so only the unconverged-at-boundary gate can catch it.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    # Coarse NCC reports (0, 0) for this centered disc.
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 200)
    feature = make_limb_feature('mislocked', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Displacement at the trust-region boundary from the (0, 0) seed, but
    # well inside the multi-pixel walk-off guard; unconverged.
    boundary_displacement = float(SPURIOUS_UNCONVERGED_TRUST_BOUNDARY_FRACTION * LM_TRUST_REGION_PX)
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(boundary_displacement, 0.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.4,
        raw_rms_px=0.4,
        iterations=30,
        converged=False,
        inlier_count=int(vertices.shape[0] * 0.6),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    # Sanity-check the test setup: the walk-off displacement guard would NOT fire.
    assert boundary_displacement < SPURIOUS_MAX_LM_DISPLACEMENT_PX
    assert result.spurious is True


def test_body_limb_nav_unconverged_near_seed_is_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The negative path: an unconverged LM that stayed near the seed is not spurious.

    A healthy dense-edge fit can miss the step tolerance while sitting well
    inside the trust region; the convergence flag alone must not reject it.
    Only a fit that is BOTH unconverged AND pinned at the boundary trips the
    mis-lock gate.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 1000)
    feature = make_limb_feature('near_seed', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Unconverged, but the displacement from the (0, 0) seed stays well
    # inside the trust-region boundary.
    near_seed_displacement = float(
        0.5 * SPURIOUS_UNCONVERGED_TRUST_BOUNDARY_FRACTION * LM_TRUST_REGION_PX
    )
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(near_seed_displacement, 0.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.4,
        raw_rms_px=0.4,
        iterations=30,
        converged=False,
        inlier_count=500,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert result.spurious is False


def test_body_limb_nav_does_not_mark_spurious_when_inlier_fraction_healthy(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The negative-path counterpart: a healthy inlier fraction does not trip spurious.

    Covers the boundary so a future widening of ``SPURIOUS_MIN_INLIER_FRACTION``
    (or a sign flip in the comparison) immediately fails this test.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 1000)
    feature = make_limb_feature('healthy_moon', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Healthy LM result: 50 % of vertices retained, sub-pixel RMS.
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.4,
        raw_rms_px=0.4,
        iterations=8,
        converged=True,
        inlier_count=500,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    inlier_fraction = result.diagnostics.tukey_inlier_count / float(vertices.shape[0])
    assert inlier_fraction >= SPURIOUS_MIN_INLIER_FRACTION
    assert result.spurious is False


def test_body_limb_nav_raw_rms_gate_catches_tukey_masked_bad_arc(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A bad arc that Tukey down-weights to zero is still flagged spurious.

    Models the mis-convergence the weighted ``rms_px`` gate cannot see:
    one limb arc fits cleanly (~0 px) while a second arc is offset by
    ~10 px.  Tukey reweights the bad arc's vertices to zero, so the
    *weighted* ``rms_px`` collapses to near zero and would slip past the
    ``rms_px > floor`` gate.  The *unweighted* ``raw_rms_px`` retains the
    offset arc and exceeds the threshold, so the raw-RMS gate is the only
    signal that recovers the spurious flag.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 200)
    feature = make_limb_feature('masked_bad_arc', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    # Bimodal residuals: half the vertices fit at ~0 px, the other half
    # are offset by ~10 px.  Weights are 1 on the clean half and 0 on the
    # offset half (Tukey rejected them), so the weighted ``rms_px`` is ~0
    # but the raw RMS over ALL vertices is ~7 px.
    n_total = vertices.shape[0]
    n_half = n_total // 2
    residuals = np.zeros(n_total, dtype=np.float64)
    residuals[n_half:] = 10.0
    weights = np.zeros(n_total, dtype=np.float64)
    weights[:n_half] = 1.0
    raw_rms = float(np.sqrt(np.mean(residuals**2)))
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        # Weighted RMS over the surviving (clean) half is ~0 — below the
        # floor, so the weighted gate alone would PASS this fit.
        rms_px=0.0,
        raw_rms_px=raw_rms,
        iterations=10,
        converged=True,
        inlier_count=n_half,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    # The weighted gate would not fire: rms_px is below the floor.
    assert forged_result.rms_px <= SPURIOUS_DT_FLOOR_PX
    # The raw RMS is large (well above the floor) — the raw-RMS gate fires.
    assert raw_rms > SPURIOUS_DT_FLOOR_PX
    assert result.spurious is True


def test_body_limb_nav_clean_fit_has_small_raw_rms_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A clean fit has a small ``raw_rms_px`` and is not flagged spurious.

    Negative-path counterpart: every vertex fits to within a fraction of
    a pixel, so the unweighted RMS stays well below the floor and the
    raw-RMS gate does not fire.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_body_limb

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = circle_polyline((100.0, 100.0), 30.0, 200)
    feature = make_limb_feature('clean_arc', vertices=vertices, outward_normals=outward)
    technique = BodyLimbNav()
    context = make_nav_context(image)

    n_total = vertices.shape[0]
    residuals = np.full(n_total, 0.3, dtype=np.float64)
    weights = np.ones(n_total, dtype=np.float64)
    raw_rms = float(np.sqrt(np.mean(residuals**2)))
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=raw_rms,
        raw_rms_px=raw_rms,
        iterations=8,
        converged=True,
        inlier_count=n_total,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_limb,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyLimbDiagnostics)
    assert raw_rms < SPURIOUS_DT_FLOOR_PX
    assert result.spurious is False
