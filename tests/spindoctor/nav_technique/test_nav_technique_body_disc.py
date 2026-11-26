"""End-to-end tests for ``BodyDiscCorrelateNav``."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.conftest import OVERRIDDEN_UPSAMPLE_FACTOR
from tests.spindoctor.nav_technique.conftest import (
    DiscImageFactory,
    NavContextFactory,
)

import spindoctor.nav_technique.nav_technique_body_disc as disc_module
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags
from spindoctor.feature.geometry import BodyDiscGeometry
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.diagnostics import BodyDiscDiagnostics
from spindoctor.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE
from spindoctor.nav_technique.nav_technique_body_disc import (
    BodyDiscCorrelateNav,
    _RotationCandidate,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.correlate import navigate_with_pyramid_kpeaks
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
        _RotationCandidate(theta_rad=-0.1, quality=8.0, offset_vu=(0, 0)),
        _RotationCandidate(theta_rad=0.0, quality=12.0, offset_vu=(0, 0)),
        _RotationCandidate(theta_rad=0.1, quality=8.0, offset_vu=(0, 0)),
    ]
    winner = candidates[1]
    sigma = BodyDiscCorrelateNav._rotation_sigma_from_quality(
        candidates=candidates, winner=winner, step_rad=0.1
    )
    assert sigma is None


def _navigate_with_cov_terms(
    monkeypatch: pytest.MonkeyPatch,
    feature: NavFeature,
    context: NavContext,
    *,
    size_frac: float,
    floor_px: float,
) -> NDArrayFloatType:
    """Navigate with the localization term off and the given size and floor terms.

    Parameters:
        monkeypatch: Installs the tuning on :class:`BodyDiscCorrelateNav`.
        feature: The disc feature.
        context: The navigation context.
        size_frac: The size-proportional model-error fraction.
        floor_px: The model-error floor in pixels.

    Returns:
        The result's covariance.
    """
    # The technique is built first, so the configuration is read and the tuning the test
    # replaces is the class's own, which is what ``monkeypatch`` restores.
    tuning = dataclasses.replace(
        BodyDiscCorrelateNav().tuning,
        localization_uncertainty_scale=0.0,
        model_error_size_frac=size_frac,
        model_error_floor_px=floor_px,
    )
    monkeypatch.setattr(BodyDiscCorrelateNav, 'tuning', tuning)
    return np.asarray(BodyDiscCorrelateNav().navigate([feature], context).covariance_px2)


def _consistency_tol(monkeypatch: pytest.MonkeyPatch, *, fraction: float, max_px: float) -> float:
    """Return the consistency cap for a 56 px disc under the given tuning.

    Parameters:
        monkeypatch: Installs the tuning on :class:`BodyDiscCorrelateNav`.
        fraction: The migration allowed per pixel of body extent.
        max_px: The smallest cap in pixels.

    Returns:
        The cap in pixels.
    """
    tuning = dataclasses.replace(
        BodyDiscCorrelateNav().tuning,
        consistency_max_fraction_of_diameter=fraction,
        consistency_max_px=max_px,
    )
    monkeypatch.setattr(BodyDiscCorrelateNav, 'tuning', tuning)
    feature = _make_disc_feature(
        'moonA', extfov_shape=(160, 160), image_center_vu=(80.0, 80.0), radius=20.0
    )
    return BodyDiscCorrelateNav()._consistency_tol_for([feature])


def test_consistency_tol_takes_the_scaled_cap_when_it_is_larger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large body's cap is its extent times the fraction, which exceeds the smallest cap."""
    assert _consistency_tol(monkeypatch, fraction=0.1, max_px=5.0) == pytest.approx(5.6)


def test_consistency_tol_takes_the_least_cap_when_it_is_larger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small body's cap is the smallest cap, which exceeds its extent times the fraction."""
    assert _consistency_tol(monkeypatch, fraction=0.01, max_px=5.0) == pytest.approx(5.0)


def test_body_disc_model_error_floor_inflates_covariance(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
    monkeypatch: pytest.MonkeyPatch,
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
    cov_bare = _navigate_with_cov_terms(monkeypatch, feature, context, size_frac=0.0, floor_px=0.0)
    cov_floored = _navigate_with_cov_terms(
        monkeypatch, feature, context, size_frac=0.0, floor_px=0.8
    )
    for axis in (0, 1):
        assert cov_floored[axis, axis] == pytest.approx(cov_bare[axis, axis] + 0.8**2, rel=1e-9)


def test_body_disc_size_term_scales_covariance_with_diameter(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
    monkeypatch: pytest.MonkeyPatch,
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
    size_px = BodyDiscCorrelateNav._max_body_extent_px([feature])
    cov_bare = _navigate_with_cov_terms(monkeypatch, feature, context, size_frac=0.0, floor_px=0.0)
    cov_sized = _navigate_with_cov_terms(
        monkeypatch, feature, context, size_frac=0.05, floor_px=0.0
    )
    expected_extra = (0.05 * size_px) ** 2
    assert expected_extra > 0.0
    for axis in (0, 1):
        assert cov_sized[axis, axis] == pytest.approx(
            cov_bare[axis, axis] + expected_extra, rel=1e-9
        )


_ROTATION_STUB_INPUTS: dict[str, Any] = {
    'image': np.zeros((8, 8)),
    'template_img': np.zeros((8, 8)),
    'template_mask': np.ones((8, 8), dtype=bool),
    'pivot_vu': (4.0, 4.0),
    'max_offset_vu': (2, 2),
    'data_mask': None,
    'upsample_factor': 4,
    'consistency_tol': 5.0,
}


def _stub_search_inputs(*, use_gradient: bool = False) -> disc_module._RotationSearchInputs:
    """Return rotation-search inputs for tests that stub the per-sample score.

    Parameters:
        use_gradient: The search mode the inputs carry.

    Returns:
        Inputs built from ``_ROTATION_STUB_INPUTS``.
    """
    return disc_module._RotationSearchInputs(
        image_surface=_ROTATION_STUB_INPUTS['image'],
        template_img=_ROTATION_STUB_INPUTS['template_img'],
        template_mask=_ROTATION_STUB_INPUTS['template_mask'],
        pivot_vu=_ROTATION_STUB_INPUTS['pivot_vu'],
        max_offset_vu=_ROTATION_STUB_INPUTS['max_offset_vu'],
        data_mask=None,
        use_gradient=use_gradient,
    )


def _stub_rotation_scores(
    monkeypatch: pytest.MonkeyPatch,
    technique: BodyDiscCorrelateNav,
    quality_for: Any,
) -> list[float]:
    """Replace the per-sample score with ``quality_for(theta_deg)``.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture.
        technique: The technique whose per-sample score is replaced.
        quality_for: Maps a rotation in degrees to the sample's quality.

    Returns:
        The list each scored rotation, in radians, is appended to.
    """
    scored: list[float] = []

    def _score_rotation(
        theta_rad: float, search: disc_module._RotationSearchInputs
    ) -> _RotationCandidate:
        scored.append(theta_rad)
        return _RotationCandidate(
            theta_rad=theta_rad, quality=quality_for(np.rad2deg(theta_rad)), offset_vu=(0, 0)
        )

    monkeypatch.setattr(technique, '_score_rotation', _score_rotation)
    return scored


def _stub_pyramids(
    monkeypatch: pytest.MonkeyPatch, technique: BodyDiscCorrelateNav, *, used_gradient: bool
) -> list[float]:
    """Replace the full NCC pyramid with a stub that records each rotation it runs at.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture.
        technique: The technique whose pyramid is replaced.
        used_gradient: The mode the stub reports picking.

    Returns:
        The list each pyramid's rotation, in radians, is appended to.
    """
    thetas: list[float] = []

    def _ncc_at_rotation(*, theta_rad: float, **_kwargs: Any) -> dict[str, Any]:
        thetas.append(theta_rad)
        return {'quality': 9.0, 'used_gradient': used_gradient, 'theta_rad': theta_rad}

    monkeypatch.setattr(technique, '_ncc_at_rotation', _ncc_at_rotation)
    return thetas


def test_rotation_samples_skip_a_nan_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN-quality first rotation sample does not block a later finite one."""
    technique = BodyDiscCorrelateNav()
    qualities = {-1.0: float('nan'), 0.0: 4.0, 1.0: 3.0}
    _stub_rotation_scores(monkeypatch, technique, lambda deg: qualities[round(deg)])
    best = technique._best_rotation_sample(
        list(np.deg2rad([-1.0, 0.0, 1.0])), _stub_search_inputs(), {}
    )
    assert best.theta_rad == 0.0


def test_rotation_samples_are_scored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rotation already scored in the search is taken from the record, not rescored."""
    technique = BodyDiscCorrelateNav()
    scored = _stub_rotation_scores(monkeypatch, technique, lambda deg: -abs(deg))
    record: dict[float, _RotationCandidate] = {}
    technique._best_rotation_sample([0.0, 0.01], _stub_search_inputs(), record)
    technique._best_rotation_sample([0.01, 0.02, 0.01], _stub_search_inputs(), record)
    assert scored == [0.0, 0.01, 0.02]


def test_rotation_search_returns_the_pyramid_at_the_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search pins the best sample and reports the full pyramid run at that rotation."""
    technique = BodyDiscCorrelateNav()
    # Peaked at +1.3 deg: level 0 picks 1, level 1 picks 1.5, and level 2 picks 1.25.
    _stub_rotation_scores(monkeypatch, technique, lambda deg: -abs(deg - 1.3))
    pyramid_thetas = _stub_pyramids(monkeypatch, technique, used_gradient=False)
    theta, ncc_result, sigma = technique._search_rotation(
        max_rotation_deg=5.0, **_ROTATION_STUB_INPUTS
    )
    assert theta == pytest.approx(np.deg2rad(1.25))
    assert pyramid_thetas == [0.0, theta]
    assert ncc_result['theta_rad'] == theta
    assert sigma is None


def test_rotation_search_reuses_the_unrotated_pyramid_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When zero rotation wins, the unrotated pyramid is the result and none reruns."""
    technique = BodyDiscCorrelateNav()
    _stub_rotation_scores(monkeypatch, technique, lambda deg: -abs(deg))
    pyramid_thetas = _stub_pyramids(monkeypatch, technique, used_gradient=False)
    theta, _, _ = technique._search_rotation(max_rotation_deg=5.0, **_ROTATION_STUB_INPUTS)
    assert theta == 0.0
    assert pyramid_thetas == [0.0]


@pytest.mark.parametrize('used_gradient', [False, True], ids=['raw', 'gradient'])
def test_rotation_search_scores_in_the_unrotated_pyramids_mode(
    monkeypatch: pytest.MonkeyPatch, used_gradient: bool
) -> None:
    """Every rotation sample is scored in the mode the unrotated pyramid picked."""
    technique = BodyDiscCorrelateNav()
    modes: list[bool] = []

    def _score_rotation(
        theta_rad: float, search: disc_module._RotationSearchInputs
    ) -> _RotationCandidate:
        modes.append(search.use_gradient)
        return _RotationCandidate(theta_rad=theta_rad, quality=-abs(theta_rad), offset_vu=(0, 0))

    monkeypatch.setattr(technique, '_score_rotation', _score_rotation)
    _stub_pyramids(monkeypatch, technique, used_gradient=used_gradient)
    technique._search_rotation(max_rotation_deg=5.0, **_ROTATION_STUB_INPUTS)
    assert set(modes) == {used_gradient}


def test_rotation_search_stays_within_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No rotation sample exceeds max_rotation_deg, even when the best lies at the cap."""
    technique = BodyDiscCorrelateNav()
    scored = _stub_rotation_scores(monkeypatch, technique, lambda deg: deg)
    _stub_pyramids(monkeypatch, technique, used_gradient=False)
    theta, _, _ = technique._search_rotation(max_rotation_deg=2.0, **_ROTATION_STUB_INPUTS)
    assert max(scored) == pytest.approx(np.deg2rad(2.0))
    assert theta == pytest.approx(np.deg2rad(2.0))


def _elongated_template(shape: tuple[int, int]) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Build a shaded ellipse template centered in ``shape``, long along u.

    Parameters:
        shape: The template's shape.

    Returns:
        The template image and its mask.
    """
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    dv = (vs - (shape[0] - 1) / 2.0) / 14.0
    du = (us - (shape[1] - 1) / 2.0) / 40.0
    inside = dv**2 + du**2 <= 1.0
    template = np.where(inside, 50.0 + 40.0 * du + 20.0 * dv, 0.0)
    return template.astype(np.float64), inside


def test_rotation_search_recovers_a_planted_rotation() -> None:
    """A frame showing the template turned by 1.5 deg and shifted is recovered in full."""
    shape = (120, 140)
    template, mask = _elongated_template(shape)
    pivot = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    rotated, _ = disc_module._rotate_template(template, mask, pivot, np.deg2rad(1.5))
    image = np.roll(rotated, (2, -3), axis=(0, 1))
    theta, ncc_result, _ = BodyDiscCorrelateNav()._search_rotation(
        image=image,
        template_img=template,
        template_mask=mask,
        pivot_vu=pivot,
        max_offset_vu=(10, 10),
        data_mask=None,
        upsample_factor=16,
        max_rotation_deg=5.0,
        consistency_tol=5.0,
    )
    assert np.rad2deg(theta) == pytest.approx(1.5)
    # The pyramid at the winner refines the gradient-mode peak on raw intensity, which
    # on this small sharp-edged template lands 0.125 px from the whole-pixel shift.
    assert ncc_result['offset'][0] == pytest.approx(2.0, abs=0.25)
    assert ncc_result['offset'][1] == pytest.approx(-3.0, abs=0.25)


def test_clamped_rotation_samples_clamp_to_the_cap() -> None:
    """Samples past the cap are moved onto it, in the order of the multiples."""
    samples = disc_module._clamped_rotation_samples(0.9, 0.5, (-2, -1, 0, 1, 2), 1.0)
    assert samples == pytest.approx([-0.1, 0.4, 0.9, 1.0, 1.0])


def _record_pyramid_refinements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the final-pass refinement of every pyramid the disc technique runs.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The list each pyramid call appends its refinement to; a call that names none
        records the default.
    """
    calls: list[str] = []

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs.get('final_refinement', 'cross_power'))
        return navigate_with_pyramid_kpeaks(*args, **kwargs)

    monkeypatch.setattr(disc_module, 'navigate_with_pyramid_kpeaks', _spy)
    return calls


def test_body_disc_translation_refines_on_the_highpass_cross_power(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The translation-only correlation refines its final pass on the high-pass surface."""
    calls = _record_pyramid_refinements(monkeypatch)
    shape = (160, 160)
    feature = _make_disc_feature(
        'moonA', extfov_shape=shape, image_center_vu=(80.0, 80.0), radius=20.0
    )
    context = make_nav_context(disc_image(shape, (80.0, 80.0), 20.0), extfov_margin_vu=(16, 16))
    BodyDiscCorrelateNav().navigate([feature], context)
    assert calls == ['highpass_cross_power']


def test_body_disc_rotation_samples_refine_on_the_highpass_cross_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rotation sample's correlation refines its final pass on the high-pass surface."""
    calls = _record_pyramid_refinements(monkeypatch)
    inputs = {**_ROTATION_STUB_INPUTS, 'image': np.ones((8, 8))}
    BodyDiscCorrelateNav()._ncc_at_rotation(theta_rad=0.0, **inputs)
    assert calls == ['highpass_cross_power']


def test_body_disc_correlation_reads_the_configured_upsample_factor(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
    upsample_override_config: Config,
) -> None:
    """The correlation runs at the upsample factor the configuration sets."""
    seen: list[int] = []

    def _record(**kwargs: Any) -> Any:
        seen.append(kwargs['upsample_factor'])
        return navigate_with_pyramid_kpeaks(**kwargs)

    monkeypatch.setattr(disc_module, 'navigate_with_pyramid_kpeaks', _record)
    shape = (160, 160)
    feature = _make_disc_feature(
        'moonA',
        extfov_shape=shape,
        image_center_vu=(80.0, 80.0),
        radius=20.0,
        planted_offset_vu=(2.0, -3.0),
    )
    context = make_nav_context(disc_image(shape, (80.0, 80.0), 20.0), extfov_margin_vu=(16, 16))
    BodyDiscCorrelateNav(config=upsample_override_config).navigate([feature], context)
    assert seen == [OVERRIDDEN_UPSAMPLE_FACTOR]
