"""End-to-end tests for ``BodyDiscCorrelateNav``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DiscImageFactory,
    NavContextFactory,
)

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags
from spindoctor.feature.geometry import BodyDiscGeometry
from spindoctor.nav_technique.diagnostics import BodyDiscDiagnostics
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NCCCovarianceTuning,
)
from spindoctor.nav_technique.nav_technique_body_disc import (
    BodyDiscCorrelateNav,
    _RotationCandidate,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType


def _disc_template(
    *,
    bbox_extfov_vu: tuple[int, int, int, int],
    center_in_template_vu: tuple[float, float],
    radius: float,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Build a postage-stamp BODY_DISC template (anti-aliased bright disc)."""
    h = bbox_extfov_vu[2] - bbox_extfov_vu[0]
    w = bbox_extfov_vu[3] - bbox_extfov_vu[1]
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    rr = np.hypot(vs - center_in_template_vu[0], us - center_in_template_vu[1])
    inside = rr <= radius - 0.5
    outside = rr >= radius + 0.5
    ramp = np.clip(radius + 0.5 - rr, 0.0, 1.0)
    template_img = np.where(inside, 100.0, np.where(outside, 0.0, 100.0 * ramp)).astype(np.float64)
    template_mask: NDArrayBoolType = template_img > 1e-6
    return template_img, template_mask


def _make_disc_feature(
    body_name: str,
    *,
    extfov_shape: tuple[int, int],
    image_center_vu: tuple[float, float],
    radius: float,
    planted_offset_vu: tuple[float, float] = (0.0, 0.0),
    subject_range_km: float = 1.0e6,
    overflow_fraction: float = 0.0,
) -> NavFeature:
    """Build a BODY_DISC feature whose template is shifted by a planted offset.

    The template is a bright anti-aliased disc placed inside a postage-stamp
    bbox.  ``planted_offset_vu`` shifts the predicted center relative to the
    actual image center, so the technique should report
    ``offset_px = planted_offset_vu`` (predicted + offset = actual).
    """
    pred_v = image_center_vu[0] - planted_offset_vu[0]
    pred_u = image_center_vu[1] - planted_offset_vu[1]
    half_extent = round(radius) + 8
    v_min = max(0, round(pred_v - half_extent))
    u_min = max(0, round(pred_u - half_extent))
    v_max = min(extfov_shape[0], round(pred_v + half_extent))
    u_max = min(extfov_shape[1], round(pred_u + half_extent))
    bbox = (v_min, u_min, v_max, u_max)
    center_in_template = (pred_v - v_min, pred_u - u_min)
    template_img, template_mask = _disc_template(
        bbox_extfov_vu=bbox,
        center_in_template_vu=center_in_template,
        radius=radius,
    )
    return NavFeature(
        feature_id=f'body_disc:{body_name}',
        feature_type=NavFeatureType.BODY_DISC,
        source_model='body',
        geometry=BodyDiscGeometry(
            bbox_extfov_vu=bbox,
            predicted_center_vu=(pred_v, pred_u),
            overflow_fraction=overflow_fraction,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.05,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.85,
        reliability_reasons=NavReliabilityBreakdown(
            visible_lit_fraction=1.0 - overflow_fraction,
            overflow_fraction=overflow_fraction,
        ),
        usable_types=frozenset({NavFeatureType.BODY_DISC}),
        flags=BodyDiscFlags(body_name=body_name, overflow_fov_fraction=overflow_fraction),
        template_img=template_img,
        template_mask=template_mask,
    )


def test_body_disc_correlate_recovers_planted_offset_single_body(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """One BODY_DISC against an anti-aliased disc image converges below 1 px."""
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 20.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(2.0, -3.0),
    )
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    assert feasibility.consumed_feature_count == 1
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(2.0, abs=1.0)
    assert result.offset_px[1] == pytest.approx(-3.0, abs=1.0)
    assert isinstance(result.diagnostics, BodyDiscDiagnostics)
    assert result.diagnostics.body_count == 1


def test_body_disc_correlate_multi_body_z_buffer_paint(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Two BODY_DISC features fuse via Z-buffer paint and recover one offset."""
    shape = (220, 220)
    radius = 18.0
    centers = [(60.0, 70.0), (150.0, 140.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_disc_feature(
            f'moon_{i}',
            extfov_shape=shape,
            image_center_vu=c,
            radius=radius,
            planted_offset_vu=planted,
            # Vary subject_range so depth ordering is well-defined
            subject_range_km=1.0e6 * (i + 1),
        )
        for i, c in enumerate(centers)
    ]
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=1.0)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=1.0)
    assert isinstance(result.diagnostics, BodyDiscDiagnostics)
    assert result.diagnostics.body_count == 2


def test_body_disc_correlate_infeasible_on_empty_input() -> None:
    technique = BodyDiscCorrelateNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_body_disc_features' in report.reason


def test_body_disc_correlate_infeasible_when_no_template(
    disc_image: DiscImageFactory,
) -> None:
    """A BODY_DISC feature without a template payload is rejected."""
    feature = NavFeature(
        feature_id='body_disc:no_template',
        feature_type=NavFeatureType.BODY_DISC,
        source_model='body',
        geometry=BodyDiscGeometry(
            bbox_extfov_vu=(0, 0, 10, 10),
            predicted_center_vu=(5.0, 5.0),
            overflow_fraction=0.0,
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.7,
        reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=1.0),
        usable_types=frozenset({NavFeatureType.BODY_DISC}),
        flags=BodyDiscFlags(body_name='no_template'),
    )
    technique = BodyDiscCorrelateNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False


@pytest.fixture
def at_edge_disc_result(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> NavTechniqueResult:
    """Build a BodyDiscCorrelateNav result whose offset hits the search-window edge.

    Plants a disc at exactly the search-window axis bound, runs
    ``BodyDiscCorrelateNav``, and returns the resulting
    :class:`NavTechniqueResult` for the at-edge / hard-zero assertions
    to consume.  Splitting into a fixture lets each property be
    asserted in its own test so a regression points at the failing
    branch directly.
    """
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 16.0
    image = disc_image(shape, image_center, radius)
    margin = 5
    feature = _make_disc_feature(
        'edge_moon',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(float(margin), 0.0),
    )
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(image, extfov_margin_vu=(margin, margin))
    return technique.navigate([feature], context)


def test_body_disc_correlate_marks_at_edge_when_offset_hits_window(
    at_edge_disc_result: NavTechniqueResult,
) -> None:
    """The pyramid wrapper flags the boundary peak as ``at_edge``."""
    assert at_edge_disc_result.at_edge is True


def test_body_disc_correlate_at_edge_forces_zero_confidence(
    at_edge_disc_result: NavTechniqueResult,
) -> None:
    """The ``hard_zero_if={'at_edge': True}`` gate drives confidence to 0."""
    assert at_edge_disc_result.confidence == pytest.approx(0.0)


def test_body_disc_correlate_registered_with_navtechnique_registry() -> None:
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert BodyDiscCorrelateNav in NavTechnique._registry


def test_body_disc_diagnostics_records_peak_to_runner_up_ratio(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A clean single-body scene reports peak-to-runner-up ratio > 1.0."""
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 20.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(1.0, 1.0),
    )
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyDiscDiagnostics)
    assert result.diagnostics.peak_to_runner_up_ratio > 1.0


def test_body_disc_diagnostics_records_consistency_and_quality(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The clean planted-offset case reports sub-pixel consistency and positive ncc_peak."""
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 20.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(1.0, 1.0),
    )
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyDiscDiagnostics)
    assert result.diagnostics.ncc_peak > 0.0
    assert result.diagnostics.consistency_px < 1.0


def test_body_disc_3dof_emits_3x3_covariance(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A planted offset under fit_camera_rotation produces a 3x3 covariance.

    NAV-010: the disc technique reports rotation as *unobservable* -- the
    NCC peak quality is a PSR/PMR separation ratio, not a log-likelihood,
    so its curvature carries no calibrated angular variance.  The
    rotation slot of the 3x3 covariance therefore carries the
    unobservable sentinel and ``sigma_rotation_rad`` is its square root.
    The translation block and rotation *estimate* are still produced by
    the 11+5+3 pyramid schedule.
    """
    shape = (120, 120)
    image_center = (60.0, 60.0)
    radius = 15.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(1.0, -1.0),
    )
    technique = BodyDiscCorrelateNav()
    context = make_nav_context(
        image,
        extfov_margin_vu=(8, 8),
        fit_camera_rotation=True,
        max_rotation_deg=5.0,
    )
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    # No rotation planted; the level-2 winner is centered on zero with the
    # 0.25 deg sample step, so |rotation| stays well inside one step.
    assert abs(result.rotation_rad) <= np.deg2rad(0.5)
    # Rotation is reported unobservable: the rotation-variance slot carries
    # the finite unobservable sentinel.
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)
    assert result.sigma_rotation_rad is not None
    assert result.sigma_rotation_rad == pytest.approx(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))
    # Wider abs tolerance than the 2-DoF case: the 11+5+3 rotation
    # search runs the NCC against rotated templates, and the 0.25 deg
    # level-2 sampling step plus integer-rounded pivot shift admits up
    # to ~ 1 px of translation jitter even on a circular disc planted
    # at zero rotation.
    assert result.offset_px[0] == pytest.approx(1.0, abs=1.5)
    assert result.offset_px[1] == pytest.approx(-1.0, abs=1.5)


def test_rotation_sigma_from_quality_reports_unobservable() -> None:
    """NAV-010: ``_rotation_sigma_from_quality`` always returns None.

    Even a sharply concave, well-centered quality parabola (the case the
    former curvature->variance map turned into a finite sigma) now
    returns None, routing the caller to the rotation-unobservable
    sentinel.  The PSR/PMR quality is not a log-likelihood, so no
    calibrated angular variance can be derived from its curvature.
    """
    candidates = [
        _RotationCandidate(theta_rad=-0.1, ncc_result={'quality': 8.0}),
        _RotationCandidate(theta_rad=0.0, ncc_result={'quality': 12.0}),
        _RotationCandidate(theta_rad=0.1, ncc_result={'quality': 8.0}),
    ]
    winner = candidates[1]
    sigma = BodyDiscCorrelateNav._rotation_sigma_from_quality(
        candidates=candidates, winner=winner, step_rad=0.1
    )
    assert sigma is None


def test_body_disc_model_error_floor_inflates_covariance(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """model_error_floor_px adds exactly its square to the covariance diagonal.

    The peak-curvature covariance measures statistical precision only; the
    floor (calibrated against the simulated-scene campaign) carries the
    size-independent pointing / interpolation model error.  Isolate the floor
    by zeroing the localization and size terms in both runs.
    """
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 20.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(2.0, -3.0),
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    bare = BodyDiscCorrelateNav()
    bare._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )
    floored = BodyDiscCorrelateNav()
    floored._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.8
    )
    cov_bare = bare.navigate([feature], context).covariance_px2
    cov_floored = floored.navigate([feature], context).covariance_px2
    for axis in (0, 1):
        assert cov_floored[axis, axis] == pytest.approx(cov_bare[axis, axis] + 0.8**2, rel=1e-9)


def test_body_disc_size_term_scales_covariance_with_diameter(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """model_error_size_frac adds (frac * body_diameter)**2 to the diagonal."""
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 20.0
    image = disc_image(shape, image_center, radius)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=image_center,
        radius=radius,
        planted_offset_vu=(2.0, -3.0),
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    bare = BodyDiscCorrelateNav()
    bare._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )
    sized = BodyDiscCorrelateNav()
    sized._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.05, model_error_floor_px=0.0
    )
    size_px = BodyDiscCorrelateNav._max_body_extent_px([feature])
    cov_bare = bare.navigate([feature], context).covariance_px2
    cov_sized = sized.navigate([feature], context).covariance_px2
    expected_extra = (0.05 * size_px) ** 2
    assert expected_extra > 0.0
    for axis in (0, 1):
        assert cov_sized[axis, axis] == pytest.approx(
            cov_bare[axis, axis] + expected_extra, rel=1e-9
        )
