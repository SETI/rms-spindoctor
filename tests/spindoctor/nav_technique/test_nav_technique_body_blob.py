"""End-to-end tests for ``BodyBlobNav``."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DiscImageFactory,
    NavContextFactory,
)

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyBlobFlags
from spindoctor.feature.geometry import BodyBlobGeometry
from spindoctor.nav_technique.diagnostics import BodyBlobDiagnostics
from spindoctor.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE
from spindoctor.nav_technique.nav_technique_body_blob import (
    BodyBlobNav,
    _centroid_model_error_var,
    _clamped_kernel_radius,
    _coarse_crescent_offset,
    _coarse_disc_offset,
    _collect_per_blob_residuals,
    _crescent_kernel,
    _disc_kernel,
    _joint_covariance,
    _kernel_centroid_offset,
    _shift_bbox,
)
from spindoctor.support.filters import NavFilterKind, NavFilterSpec


def _make_blob_feature(
    body_name: str,
    *,
    predicted_center_vu: tuple[float, float],
    predicted_diameter_px: float,
    bbox_pad: int = 4,
    phase_angle_deg: float = 0.0,
    phase_irregularity_factor: float = 0.0,
    sub_solar_dir_vu: tuple[float, float] = (0.0, 0.0),
) -> NavFeature:
    """Build a BODY_BLOB feature whose bbox tightly bounds the predicted disc."""
    radius = predicted_diameter_px / 2.0
    v_min = int(np.floor(predicted_center_vu[0] - radius - bbox_pad))
    u_min = int(np.floor(predicted_center_vu[1] - radius - bbox_pad))
    v_max = int(np.ceil(predicted_center_vu[0] + radius + bbox_pad))
    u_max = int(np.ceil(predicted_center_vu[1] + radius + bbox_pad))
    sigma_centroid = max(predicted_diameter_px / 6.0, 0.5)
    cov = (sigma_centroid * sigma_centroid) * np.eye(2, dtype=np.float64)
    return NavFeature(
        feature_id=f'body_blob:{body_name}',
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body',
        geometry=BodyBlobGeometry(
            predicted_center_vu=predicted_center_vu,
            bbox_extfov_vu=(v_min, u_min, v_max, u_max),
            predicted_diameter_px=predicted_diameter_px,
        ),
        subject_range_km=5.0e5,
        position_cov_px=cov,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.4,
        reliability_reasons=NavReliabilityBreakdown(
            blob_snr=0.5,
            blob_extent_px=predicted_diameter_px / 30.0,
        ),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(
            body_name=body_name,
            predicted_diameter_px=predicted_diameter_px,
            phase_angle_deg=phase_angle_deg,
            phase_irregularity_factor=phase_irregularity_factor,
            sub_solar_dir_vu=sub_solar_dir_vu,
        ),
    )


class _RecordingLogger:
    """Logger stand-in keeping the arguments of every debug call."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.debug_calls: list[tuple[str, tuple[Any, ...]]] = []

    def debug(self, message: str, *args: Any) -> None:
        """Record one debug call and its arguments."""
        self.debug_calls.append((message, args))


def test_the_blob_line_states_both_positions_in_the_image_frame(
    disc_image: DiscImageFactory,
) -> None:
    """The per-blob line names pixels of the image, not of the padded array.

    The disc is drawn on the center of row 100 and column 100 of the padded
    array, which sits 32 rows and columns of padding in from the image's own
    first row and column, so both the prediction and the centroid measured
    from it are on the center of image row 68.  A position stated to a person
    names a row's center by that row's number plus a half.  The coarse offset
    between them is a difference and stays as it is.
    """
    drawn_vu = (100.0, 100.0)
    image = disc_image((200, 200), drawn_vu, 8.0)
    feature = _make_blob_feature('MIMAS', predicted_center_vu=drawn_vu, predicted_diameter_px=16.0)
    logger = _RecordingLogger()
    _collect_per_blob_residuals(
        [feature],
        image,
        1.0,
        0.0,
        cast(Any, logger),
        image_signal=np.clip(image, 0.0, None),
        margin_vu=(32, 32),
        prior_offset_vu=None,
    )
    args = logger.debug_calls[-1][1]
    assert args[1:3] == (68.5, 68.5)
    assert args[5:7] == (68.5, 68.5)


def test_body_blob_recovers_planted_offset_single_blob(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A single bright disc + blob feature predicting an offset center recovers offset."""
    shape = (200, 200)
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 2.0, -3.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 1
    # Hard cap at 0.4 — the technique cannot dominate the ensemble
    # even with perfect inputs.
    assert result.confidence <= 0.4 + 1e-12


def test_body_blob_multi_body_least_squares_average(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Two blob features with the same planted offset average to that offset."""
    shape = (220, 220)
    radius = 7.0
    actual_centers = [(60.0, 70.0), (150.0, 140.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in actual_centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=(c[0] - planted[0], c[1] - planted[1]),
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(actual_centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 2


def test_body_blob_returns_zero_confidence_when_image_blank(
    make_nav_context: NavContextFactory,
) -> None:
    """A blob feature whose predicted bbox lies in a blank image is dropped."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=(100.0, 100.0),
        predicted_diameter_px=20.0,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 0


def test_body_blob_infeasible_on_empty_input() -> None:
    """``is_feasible([])`` reports infeasibility with a no-features reason."""
    technique = BodyBlobNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_body_blob_features' in report.reason


def test_body_blob_infeasible_on_zero_diameter() -> None:
    """A BODY_BLOB feature with ``predicted_diameter_px == 0`` is infeasible.

    Asserts the reason names the predicted-diameter requirement so a
    regression that returns a generic infeasibility message is caught.
    """
    feature = NavFeature(
        feature_id='body_blob:zero',
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body',
        geometry=BodyBlobGeometry(
            predicted_center_vu=(50.0, 50.0),
            bbox_extfov_vu=(40, 40, 60, 60),
            predicted_diameter_px=0.0,
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.1,
        reliability_reasons=NavReliabilityBreakdown(blob_snr=0.0),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(body_name='zero', predicted_diameter_px=0.0),
    )
    technique = BodyBlobNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False
    assert 'predicted_diameter' in report.reason


def test_body_blob_confidence_capped_at_0_4(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Even an ideal multi-blob fit cannot exceed 0.4 confidence."""
    shape = (240, 240)
    radius = 12.0
    centers = [(60.0, 60.0), (60.0, 180.0), (180.0, 60.0), (180.0, 180.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=c,
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.confidence == pytest.approx(0.4, abs=1e-12)


def test_body_blob_registered_with_navtechnique_registry() -> None:
    """``BodyBlobNav`` is auto-registered in ``NavTechnique._registry`` on import."""
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert BodyBlobNav in NavTechnique._registry


def test_body_blob_marks_at_edge_when_centroid_hits_window(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A converged offset within ``at_edge_tolerance_px`` of the search-window
    edge is flagged ``at_edge=True`` and forced to zero confidence by the
    ``hard_zero_if`` gate.
    """
    shape = (200, 200)
    margin_v = 6
    margin_u = 6
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    # Plant the predicted center exactly ``margin_v`` rows above the
    # actual center so the recovered offset lands on the search-window
    # boundary.
    pred_center = (actual_center[0] - float(margin_v), actual_center[1])
    feature = _make_blob_feature(
        'edge_moon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image, extfov_margin_vu=(margin_v, margin_u))
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_body_blob_diagnostics_records_residual_and_snr(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Multi-blob agreement reports sub-pixel residual scatter and high SNR.

    The two blobs share the same planted offset so per-blob residuals
    around the joint mean must be sub-pixel; the bright synthetic discs
    push mean SNR well above 5.
    """
    shape = (220, 220)
    radius = 7.0
    actual_centers = [(60.0, 70.0), (150.0, 140.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in actual_centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=(c[0] - planted[0], c[1] - planted[1]),
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(actual_centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.residual_px < 0.5
    assert result.diagnostics.body_snr_inside_predicted_bbox > 5.0


def test_body_blob_3dof_rotation_unobservable(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """When ``fit_camera_rotation=True`` the blob technique reports a 3x3 covariance.

    The rotation slot carries the ``ROTATION_UNOBSERVABLE_VARIANCE``
    sentinel because a centroid is rotation-invariant about itself; the
    ensemble combine treats this as no-information in the rotation
    direction.
    """
    shape = (200, 200)
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=(98.0, 102.0),
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image, fit_camera_rotation=True)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad == pytest.approx(0.0)
    assert result.sigma_rotation_rad == pytest.approx(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)


def test_body_blob_diagnostics_records_max_phase_irregularity(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The technique surfaces the worst per-blob irregularity factor.

    Two blobs in the same scene with different irregularity factors:
    the diagnostics record the maximum so the confidence-formula
    penalty is driven by the worst blob (regular bodies have factor ~
    0.005, irregular ~ 0.05+).
    """
    shape = (200, 200)
    image = np.maximum(
        disc_image(shape, (60.0, 60.0), 8.0),
        disc_image(shape, (140.0, 140.0), 8.0),
    )
    low_phase_blob = _make_blob_feature(
        'regular_moon',
        predicted_center_vu=(60.0, 60.0),
        predicted_diameter_px=16.0,
        phase_angle_deg=15.0,
        phase_irregularity_factor=0.003,
    )
    high_phase_blob = _make_blob_feature(
        'irregular_moon',
        predicted_center_vu=(140.0, 140.0),
        predicted_diameter_px=16.0,
        phase_angle_deg=130.0,
        phase_irregularity_factor=0.075,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([low_phase_blob, high_phase_blob], context)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.max_phase_angle_deg == pytest.approx(130.0)
    assert result.diagnostics.max_phase_irregularity_factor == pytest.approx(0.075)


def test_body_blob_flags_reject_phase_outside_valid_range() -> None:
    """``BodyBlobFlags`` validates ``phase_angle_deg`` is in [0, 180]."""
    with pytest.raises(ValueError, match='phase_angle_deg'):
        BodyBlobFlags(body_name='X', predicted_diameter_px=10.0, phase_angle_deg=-1.0)
    with pytest.raises(ValueError, match='phase_angle_deg'):
        BodyBlobFlags(body_name='X', predicted_diameter_px=10.0, phase_angle_deg=181.0)


def test_body_blob_flags_reject_negative_phase_irregularity() -> None:
    """``BodyBlobFlags`` validates ``phase_irregularity_factor`` is non-negative."""
    with pytest.raises(ValueError, match='phase_irregularity_factor'):
        BodyBlobFlags(
            body_name='X',
            predicted_diameter_px=10.0,
            phase_irregularity_factor=-0.01,
        )


def test_joint_covariance_two_point_reduced_chi_square() -> None:
    """Two-blob covariance matches the analytic reduced-chi-square weighted mean.

    Derivation (per-axis: var = chi2_nu / sum(w), with
    chi2_nu = sum(w * r^2) / max(N - p, 1), p = 2):

    w = [1, 3], sum(w) = 4, N = 2, dof = max(2 - 2, 1) = 1.

    V axis: offsets = [2, 4], mean = (1*2 + 3*4)/4 = 3.5,
            r = [-1.5, 0.5], sum(w r^2) = 1*2.25 + 3*0.25 = 3.0,
            chi2_nu = 3.0/1 = 3.0, var = 3.0/4 = 0.75 (> floor 1/4 = 0.25).
    U axis: offsets = [-1, 1], mean = (1*-1 + 3*1)/4 = 0.5,
            r = [-1.5, 0.5], sum(w r^2) = 3.0, var = 0.75 (identical).
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.5
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 0.5
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    assert cov[0, 0] == pytest.approx(0.75, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.75, abs=1e-9)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert cov[1, 0] == pytest.approx(0.0, abs=1e-9)


def test_joint_covariance_n_point_reduced_chi_square() -> None:
    """Four-blob covariance matches the analytic reduced-chi-square weighted mean.

    w = [1, 1, 1, 1], sum(w) = 4, N = 4, dof = max(4 - 2, 1) = 2.

    V axis: offsets = [0, 2, 4, 6], mean = 12/4 = 3.0,
            r = [-3, -1, 1, 3], sum(w r^2) = 9 + 1 + 1 + 9 = 20,
            chi2_nu = 20/2 = 10, var = 10/4 = 2.5 (> floor 0.25).
    U axis: offsets = [1, 1, 1, 1], mean = 1.0, r = 0 everywhere,
            sum(w r^2) = 0, chi2_nu = 0, candidate var = 0 -> floored at
            1/sum(w) = 1/4 = 0.25.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    offsets_v = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    offsets_u = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.0
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 1.0
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    assert cov[0, 0] == pytest.approx(2.5, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)


def test_joint_covariance_single_blob_is_large_not_overconfident() -> None:
    """NAV-005: a single blob yields the (large) inverse-precision floor.

    One point cannot constrain two translation parameters, so the
    covariance must be large rather than a tiny over-confident value.
    With a single small-weight blob (w = 0.01) the per-axis variance is
    the pure inverse precision 1/sum(w) = 1/0.01 = 100.0 -- there is no
    residual scatter (the lone residual is zero by construction).  The
    old ``sum(w r^2)/(sum w)^2`` form would have reported 0 here (no
    floor at the correct power), which the over-confidence fix corrects.
    """
    weights = np.array([0.01], dtype=np.float64)
    offsets_v = np.array([3.0], dtype=np.float64)
    offsets_u = np.array([-4.0], dtype=np.float64)
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=3.0,
        du=-4.0,
        extents_px=np.zeros_like(offsets_v),
    )
    # 1 / sum(w) = 1 / 0.01 = 100.0 on each axis.
    assert cov[0, 0] == pytest.approx(100.0, abs=1e-9)
    assert cov[1, 1] == pytest.approx(100.0, abs=1e-9)
    # Far larger than a well-determined multi-blob fit (~O(1)).
    assert cov[0, 0] > 1.0


def test_joint_covariance_model_error_floor_inflates_diagonal_by_square() -> None:
    """ORCH-001: model_error_floor_px>0 adds exactly its square to the diagonal.

    Reuses the two-point fixture (var = 0.75 per axis with no floor).
    With model_error_floor_px = 2.0 each diagonal grows by exactly
    2.0**2 = 4.0 -> 0.75 + 4.0 = 4.75; the off-diagonal stays zero.
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = 3.5
    du = 0.5
    base = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
    )
    floored = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        extents_px=np.zeros_like(offsets_v),
        model_error_floor_px=2.0,
    )
    assert floored[0, 0] == pytest.approx(4.75, abs=1e-9)
    assert floored[1, 1] == pytest.approx(4.75, abs=1e-9)
    assert floored[0, 0] - base[0, 0] == pytest.approx(4.0, abs=1e-9)
    assert floored[1, 1] - base[1, 1] == pytest.approx(4.0, abs=1e-9)
    assert floored[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_centroid_model_error_var_single_blob_is_radius_fraction_squared() -> None:
    """One blob's size-error variance is exactly (frac * radius)**2."""
    extents = np.array([100.0], dtype=np.float64)  # diameter -> radius 50
    var = _centroid_model_error_var(extents, 0.05)
    assert var == pytest.approx((0.05 * 50.0) ** 2, rel=1e-9)


def test_centroid_model_error_var_combines_by_inverse_variance() -> None:
    """Two equal blobs halve the single-blob size-error variance."""
    extents = np.array([100.0, 100.0], dtype=np.float64)
    single = _centroid_model_error_var(extents[:1], 0.05)
    both = _centroid_model_error_var(extents, 0.05)
    assert both == pytest.approx(single / 2.0, rel=1e-9)


def test_centroid_model_error_var_disabled_returns_zero() -> None:
    """A zero fraction disables the size term."""
    extents = np.array([100.0, 40.0], dtype=np.float64)
    assert _centroid_model_error_var(extents, 0.0) == 0.0


def test_joint_covariance_size_term_tracks_body_radius() -> None:
    """A single-blob covariance scales with body size through the size term.

    The reported per-axis sigma of a lone blob should land within a factor
    of 1.5 of the size-scaled centroid-model error (frac * radius) plus the
    absolute floor -- the calibration property the photon-only weight lacked.
    """
    weights = np.array([100.0], dtype=np.float64)  # bright: photon floor ~0.01 px
    offsets_v = np.array([0.0], dtype=np.float64)
    offsets_u = np.array([0.0], dtype=np.float64)
    extents = np.array([200.0], dtype=np.float64)  # radius 100 px
    frac = 0.05
    floor = 0.1
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=0.0,
        du=0.0,
        extents_px=extents,
        model_error_floor_px=floor,
        centroid_model_error_frac=frac,
    )
    reported_sigma = float(np.sqrt(cov[0, 0]))
    expected_sigma = float(np.sqrt((frac * 100.0) ** 2 + floor**2 + 1.0 / 100.0))
    assert 1.0 / 1.5 <= reported_sigma / expected_sigma <= 1.5


# ---------------------------------------------------------------------------
# Coarse blob-shaped-disc acquisition
# ---------------------------------------------------------------------------


def test_disc_kernel_is_a_filled_disc() -> None:
    """The matched-filter kernel is a filled disc of the given radius."""
    kernel = _disc_kernel(3.0)
    assert kernel.shape == (7, 7)
    assert kernel[3, 3] == 1.0  # center
    assert kernel[0, 0] == 0.0  # corner outside the disc
    assert kernel.sum() > 0.0


def test_coarse_disc_offset_locates_body_outside_predicted_center() -> None:
    """The correlation finds a bright disc displaced from the predicted center."""
    signal = np.zeros((120, 120), dtype=np.float64)
    vv, uu = np.mgrid[0:120, 0:120]
    # Bright disc centered at (80, 50), radius 6.
    signal[(vv - 80) ** 2 + (uu - 50) ** 2 <= 36] = 100.0
    dv, du = _coarse_disc_offset(
        signal, predicted_center_vu=(60.0, 60.0), predicted_diameter_px=12.0, margin_vu=(40, 40)
    )
    # Predicted (60, 60) -> body (80, 50): offset (+20, -10).
    assert dv == pytest.approx(20, abs=1)
    assert du == pytest.approx(-10, abs=1)


def test_coarse_disc_offset_returns_zero_on_blank_window() -> None:
    """With no signal in the window the coarse offset is zero (no relocation)."""
    signal = np.zeros((64, 64), dtype=np.float64)
    dv, du = _coarse_disc_offset(
        signal, predicted_center_vu=(32.0, 32.0), predicted_diameter_px=8.0, margin_vu=(10, 10)
    )
    assert (dv, du) == (0, 0)


# ---------------------------------------------------------------------------
# Phase-aware crescent coarse acquisition
# ---------------------------------------------------------------------------


def test_crescent_kernel_is_lit_only_on_the_sub_solar_side() -> None:
    """The crescent kernel lights the sub-solar limb and darkens the far side."""
    # Sun toward -v (top of the image): the lit limb is at the top edge.
    kernel = _crescent_kernel(10.0, math.radians(120.0), (-1.0, 0.0))
    mid = kernel.shape[0] // 2
    assert kernel[1, mid] > 0.0  # near the top (sun-side) limb: lit
    assert kernel[-2, mid] == 0.0  # near the bottom (anti-sun) limb: dark
    assert kernel[0, 0] == 0.0  # corner outside the projected disc: zero


def test_crescent_kernel_at_zero_phase_is_a_full_disc() -> None:
    """At phase 0 the crescent collapses to a symmetric full disc.

    Independent of the sub-solar direction, the kernel is limb-darkened but
    has no bright/dark asymmetry, so its brightness centroid sits at the
    middle and its interior is lit (the limb itself is zero where ``z == 0``).
    """
    kernel = _crescent_kernel(6.0, 0.0, (-1.0, 0.0))
    mid = kernel.shape[0] // 2
    radius = 6.0
    yy, xx = np.mgrid[-mid : mid + 1, -mid : mid + 1]
    interior = yy * yy + xx * xx <= (radius - 1.0) ** 2
    assert np.all(kernel[interior] > 0.0)  # every strict-interior pixel is lit
    off_v, off_u = _kernel_centroid_offset(kernel)
    assert off_v == pytest.approx(0.0, abs=1e-9)
    assert off_u == pytest.approx(0.0, abs=1e-9)


def test_kernel_centroid_offset_is_zero_for_a_disc() -> None:
    """A symmetric disc kernel has its brightness centroid at the middle."""
    off_v, off_u = _kernel_centroid_offset(_disc_kernel(8.0))
    assert off_v == pytest.approx(0.0, abs=1e-9)
    assert off_u == pytest.approx(0.0, abs=1e-9)


def test_kernel_centroid_offset_points_toward_the_bright_limb() -> None:
    """A crescent's brightness centroid is displaced toward the sub-solar limb.

    The sun points along +u alone, so the crescent is mirror-symmetric in v
    and its centroid cannot move off the kernel's own middle row at all: the
    v bound is exact rather than approximate, and a crescent built about the
    wrong row would fail it by however far it was built wrong.
    """
    # Sun toward +u (right): the lit centroid sits right of the geometric center.
    kernel = _crescent_kernel(10.0, math.radians(120.0), (0.0, 1.0))
    off_v, off_u = _kernel_centroid_offset(kernel)
    assert off_u > 1.0
    assert off_v == pytest.approx(0.0, abs=1e-12)


def test_coarse_crescent_offset_locates_a_displaced_crescent() -> None:
    """The crescent filter maps the predicted lit centroid onto the observed one.

    The observed body is a crescent of the modeled shape placed at geometric
    center (80, 95); the prediction is the unshifted geometric center (64, 64).
    Because the feature carries the *lit* centroid, the same brightness-centroid
    offset is added to both predicted and observed centers, so the recovered
    shift must equal the geometric displacement ``(16, 31)`` -- this pins the
    lit-vs-geometric-centroid bookkeeping the coarse stage depends on.
    """
    radius = 12.0
    sub_solar = (0.0, 1.0)  # sun toward +u
    phase_deg = 120.0
    kernel = _crescent_kernel(radius, math.radians(phase_deg), sub_solar)
    centroid_off = _kernel_centroid_offset(kernel)
    half = kernel.shape[0] // 2
    shape = (160, 160)
    body_center = (80, 95)
    signal = np.zeros(shape, dtype=np.float64)
    v0, u0 = body_center[0] - half, body_center[1] - half
    signal[v0 : v0 + kernel.shape[0], u0 : u0 + kernel.shape[1]] = 100.0 * kernel
    # Feature carries the lit centroid: geometric center + kernel centroid off.
    pred_lit_vu = (64.0 + centroid_off[0], 64.0 + centroid_off[1])
    dv, du = _coarse_crescent_offset(
        signal,
        predicted_center_vu=pred_lit_vu,
        predicted_diameter_px=2.0 * radius,
        phase_deg=phase_deg,
        sub_solar_dir_vu=sub_solar,
        margin_vu=(40, 40),
    )
    assert dv == pytest.approx(body_center[0] - 64, abs=1)
    assert du == pytest.approx(body_center[1] - 64, abs=1)


def test_body_blob_high_phase_crescent_relocates_with_sub_solar_dir(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A high-phase blob carrying a sub-solar direction is found beyond its bbox.

    With the direction known the coarse stage synthesizes a crescent template
    and correlates it, so a body displaced well outside its predicted bounding
    box is relocated -- the case the skipped disc template cannot handle.  A
    fully-lit disc stands in for the body here; the crescent template still
    peaks on it because the disc subsumes the crescent's lit region.
    """
    shape = (220, 220)
    actual_center = (110.0, 130.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 20.0, 24.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'crescentMoon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
        phase_angle_deg=120.0,
        sub_solar_dir_vu=(0.0, 1.0),
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)
    assert result.spurious is False


def test_shift_bbox_translates_all_corners() -> None:
    """``_shift_bbox`` adds the offset to both corners."""
    assert _shift_bbox((2, 3, 10, 11), 5, -4) == (7, -1, 15, 7)


def test_body_blob_recovers_offset_beyond_predicted_bbox(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A body displaced far outside its predicted bbox is recovered via correlation.

    The planted offset (22, -16) moves the body well outside the predicted
    bounding box (radius + a few px of slop), so the brightness-weighted
    centroid alone would clip and silently bias.  The blob-shaped-disc coarse
    correlation relocates the bbox onto the body, restoring sub-pixel recovery.
    """
    shape = (220, 220)
    actual_center = (110.0, 110.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 22.0, -16.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'farMoon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
        phase_angle_deg=20.0,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)
    assert result.spurious is False


def test_body_blob_high_phase_crescent_does_not_relocate(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A high-phase blob with no sub-solar direction keeps its predicted bbox.

    Above the phase ceiling a disc template would lock onto a crescent's bright
    arc rather than the body center, so it is skipped; the crescent template
    that replaces it needs the sub-solar direction, which this feature does not
    carry (default ``(0, 0)``).  The coarse stage therefore makes no relocation
    and, with a small in-bbox offset, the centroid still recovers it -- the
    point is that no spurious multi-pixel relocation occurs.
    """
    shape = (200, 200)
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 1.5, -1.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'crescentMoon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
        phase_angle_deg=120.0,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)


def test_body_blob_uses_installed_prior_to_seed_bbox(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An installed prior seeds the bbox even for a high-phase body.

    The prior is a measured offset, not a template match, so it bypasses the
    phase gate.  Here a high-phase body sits far outside its predicted bbox;
    only the prior (not the skipped disc correlation) can recover it.
    """
    shape = (220, 220)
    actual_center = (110.0, 110.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 20.0, -14.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'priorMoon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
        phase_angle_deg=120.0,
    )
    technique = BodyBlobNav()
    base_context = make_nav_context(image)
    context = base_context.with_prior(
        offset_px=(planted_dv, planted_du), covariance_px2=np.eye(2, dtype=np.float64)
    )
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)


# --- kernel-radius clamp (issue #202) ---


def test_clamped_kernel_radius_passes_small_bodies_through() -> None:
    """A body smaller than the frame keeps its predicted radius."""
    assert _clamped_kernel_radius(12.0, (256, 256)) == pytest.approx(6.0)


def test_clamped_kernel_radius_bounds_huge_bodies_to_frame() -> None:
    """A predicted diameter far beyond the frame clamps to the half-diagonal."""
    radius = _clamped_kernel_radius(50_000.0, (256, 256))
    assert radius == pytest.approx(math.hypot(256, 256) / 2.0)


def test_coarse_disc_offset_bounded_for_offframe_giant() -> None:
    """A mostly off-frame giant's blob must not allocate a body-sized kernel.

    Before the clamp, a predicted diameter of tens of thousands of pixels
    (an off-frame gas giant) built a kernel quadratic in the diameter and
    exhausted memory inside fftconvolve (issue #202: N1646315051 grew past
    61 GB RSS).  With the clamp the correlation runs against a frame-sized
    template and completes in bounded memory.
    """
    image = np.zeros((128, 128), dtype=np.float64)
    image[40:90, 30:80] = 50.0
    offset = _coarse_disc_offset(image, (64.0, 64.0), 40_000.0, (32, 32))
    assert isinstance(offset, tuple)
    assert abs(offset[0]) <= 32
    assert abs(offset[1]) <= 32


def test_coarse_crescent_offset_bounded_for_offframe_giant() -> None:
    """The crescent path clamps its kernel exactly like the disc path.

    ``_coarse_crescent_offset`` builds its own template, so removing the
    clamp there would regress independently of the disc caller: an
    off-frame giant at high phase would again allocate a kernel quadratic
    in the predicted diameter inside fftconvolve (issue #202).
    """
    image = np.zeros((128, 128), dtype=np.float64)
    image[40:90, 30:80] = 50.0
    offset = _coarse_crescent_offset(image, (64.0, 64.0), 40_000.0, 120.0, (0.0, -1.0), (32, 32))
    # Completing at all on a 128x128 frame proves the clamp: unclamped, the
    # 40,000 px diameter would allocate a 40k x 40k kernel before the FFT.
    # The returned shift is the margin-bounded correlation peak plus the
    # crescent kernel's lit-centroid compensation, so the bound includes it.
    radius = _clamped_kernel_radius(40_000.0, (128, 128))
    centroid_v, centroid_u = _kernel_centroid_offset(
        _crescent_kernel(radius, math.radians(120.0), (0.0, -1.0))
    )
    assert isinstance(offset, tuple)
    assert abs(offset[0]) <= 32 + abs(centroid_v) + 1.0
    assert abs(offset[1]) <= 32 + abs(centroid_u) + 1.0
