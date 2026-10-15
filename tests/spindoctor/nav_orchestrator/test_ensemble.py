"""Tests for ``spindoctor.nav_orchestrator.ensemble.ensemble`` and helpers."""

import dataclasses
import math

import numpy as np
import pytest

from spindoctor.nav_orchestrator.ensemble import (
    DEFAULT_MAX_ALLOWED_ROTATION_DEG,
    EnsembleConfig,
    _combine_confidence,
    _combine_precision_weighted,
    derive_confidence_rank,
    ensemble,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    NavTechniqueDiagnostics,
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    embed_rotation_unobservable,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.status_reason import NavStatusReason


def _classifier(*, background_gradient_score: float | None = None) -> NavImageClassifierResult:
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
        background_gradient_score=background_gradient_score,
    )


def _provenance() -> Provenance:
    return Provenance(
        spindoctor_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )


def _make_result(
    *,
    technique_name: str = 'BodyLimbNav',
    offset: tuple[float, float] = (0.0, 0.0),
    cov: np.ndarray | None = None,
    confidence: float = 0.8,
    spurious: bool = False,
    at_edge: bool = False,
    diagnostics: NavTechniqueDiagnostics | None = None,
) -> NavTechniqueResult:
    """Build a minimal NavTechniqueResult for ensemble tests."""
    if cov is None:
        cov = np.eye(2, dtype=np.float64) * 0.25
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=(f'{technique_name}:f1',),
        offset_px=offset,
        covariance_px2=cov,
        confidence=confidence,
        spurious=spurious,
        at_edge=at_edge,
        diagnostics=diagnostics if diagnostics is not None else BodyLimbDiagnostics(),
    )


def test_ensemble_empty_inputs_returns_failed_no_techniques() -> None:
    """An empty input list yields a no_feasible_techniques failure."""
    result = ensemble(
        [],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'no_feasible_techniques'


def test_ensemble_all_spurious_returns_failed() -> None:
    """If every input is spurious, the ensemble fails with the matching reason."""
    a = _make_result(technique_name='TechniqueA', spurious=True)
    b = _make_result(technique_name='TechniqueB', spurious=True)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'all_techniques_spurious'


def test_ensemble_two_agreeing_results_combine() -> None:
    """Two agreeing results combine to the precision-weighted mean."""
    a = _make_result(technique_name='TechniqueA', offset=(1.0, 2.0), confidence=0.85)
    b = _make_result(technique_name='TechniqueB', offset=(1.05, 2.05), confidence=0.85)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # Equal covariance + equal confidence -> arithmetic mean.
    assert np.isclose(result.offset_px[0], 1.025)
    assert np.isclose(result.offset_px[1], 2.025)


def test_ensemble_disagreement_yields_conflicted() -> None:
    """Two well-separated high-confidence results trigger the conflicted branch."""
    cov = np.eye(2, dtype=np.float64) * 0.01
    a = _make_result(technique_name='TechniqueA', offset=(1.0, 1.0), cov=cov, confidence=0.9)
    b = _make_result(technique_name='TechniqueB', offset=(50.0, 50.0), cov=cov, confidence=0.9)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.confidence_rank == 'conflicted'


def test_ensemble_pixel_floor_groups_crlb_tight_results_within_floor() -> None:
    """CRLB-tight covariances disagreeing within the pixel floor still group.

    Without the pixel-floor, two BodyDisc/Limb-class results reporting
    sigmas ~0.001 px register as thousands of sigmas apart even when
    their offsets agree to a few pixels — the production failure mode
    that turned ``status=ok`` into ``status=conflicted`` for many real
    images.  The default ``agreement_pixel_floor`` (5.0 px) recovers
    the consensus in that regime.
    """
    cov_tight = np.eye(2, dtype=np.float64) * (1.0e-3) ** 2
    cov_loose = np.eye(2, dtype=np.float64) * 0.15**2
    disc = _make_result(
        technique_name='BodyDiscCorrelateNav',
        offset=(5.09, -6.95),
        cov=cov_tight,
        confidence=0.36,
    )
    limb = _make_result(
        technique_name='BodyLimbNav',
        offset=(8.65, -6.46),
        cov=cov_loose,
        confidence=0.46,
    )
    result = ensemble(
        [disc, limb],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'


def test_ensemble_pixel_floor_disabled_falls_back_to_mahalanobis() -> None:
    """Setting ``agreement_pixel_floor=0.0`` restores Mahalanobis-only grouping."""
    cov_tight = np.eye(2, dtype=np.float64) * (1.0e-3) ** 2
    a = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov_tight, confidence=0.5)
    b = _make_result(technique_name='TechniqueB', offset=(3.0, 0.0), cov=cov_tight, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(agreement_pixel_floor=0.0),
    )
    assert result.status == 'conflicted'


def test_ensemble_pixel_floor_does_not_group_results_beyond_floor() -> None:
    """Two CRLB-tight results disagreeing well beyond the floor still conflict."""
    cov_tight = np.eye(2, dtype=np.float64) * (1.0e-3) ** 2
    a = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov_tight, confidence=0.5)
    # 20 px apart — far beyond the default 5.0 px floor.
    b = _make_result(technique_name='TechniqueB', offset=(20.0, 0.0), cov=cov_tight, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'


def test_ensemble_at_edge_dropped_when_interior_exists() -> None:
    """An at-edge result is dropped when an interior result exists."""
    a = _make_result(
        technique_name='TechniqueA',
        offset=(1.0, 1.0),
        confidence=0.9,
        at_edge=True,
    )
    b = _make_result(technique_name='TechniqueB', offset=(1.05, 1.05), confidence=0.85)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    # The at-edge result is dropped, leaving only the interior technique
    # which contributes its exact offset.
    assert result.offset_px == (1.05, 1.05)


def test_ensemble_at_edge_kept_when_only_one() -> None:
    """An at-edge result survives when no interior result exists."""
    a = _make_result(
        technique_name='TechniqueA',
        offset=(1.0, 1.0),
        confidence=0.9,
        at_edge=True,
    )
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'


def test_ensemble_below_min_confidence_fails() -> None:
    """A combined confidence below min_confidence yields failed."""
    cfg = EnsembleConfig(min_confidence=0.5)
    a = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), confidence=0.1)
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=cfg,
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'final_confidence_below_threshold'


def test_derive_confidence_rank_high() -> None:
    """High confidence + small sigma earns the 'high' tier."""
    rank = derive_confidence_rank(confidence=0.9, sigma_px=(0.3, 0.3))
    assert rank == 'high'


def test_derive_confidence_rank_medium_when_sigma_too_large_for_high() -> None:
    """High confidence but sigma above 'high' threshold falls back to 'medium'."""
    rank = derive_confidence_rank(confidence=0.85, sigma_px=(1.5, 1.5))
    assert rank == 'medium'


def test_derive_confidence_rank_low() -> None:
    """Low confidence + any sigma earns the 'low' tier when >= 0.35."""
    rank = derive_confidence_rank(confidence=0.4, sigma_px=(10.0, 10.0))
    assert rank == 'low'


def test_derive_confidence_rank_failed_below_threshold() -> None:
    """Confidence below the 0.35 low-tier boundary yields 'failed'."""
    rank = derive_confidence_rank(confidence=0.3, sigma_px=(0.3, 0.3))
    assert rank == 'failed'


def test_derive_confidence_rank_logs_when_sigma_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CODE-ORCH-003: a None sigma is logged and caps the rank at low."""
    rank = derive_confidence_rank(confidence=0.95, sigma_px=None)
    assert rank == 'low'
    captured = capsys.readouterr()
    assert 'sigma_px is None' in (captured.out + captured.err)


def test_ensemble_order_independent() -> None:
    """Reordering inputs produces the same combined offset (within tolerance)."""
    cov = np.eye(2, dtype=np.float64) * 0.1
    a = _make_result(technique_name='A', offset=(1.0, 1.0), cov=cov, confidence=0.8)
    b = _make_result(technique_name='B', offset=(1.1, 1.1), cov=cov, confidence=0.8)
    res1 = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    res2 = ensemble(
        [b, a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert res1.offset_px is not None
    assert res2.offset_px is not None
    assert abs(res1.offset_px[0] - res2.offset_px[0]) < 1e-9
    assert abs(res1.offset_px[1] - res2.offset_px[1]) < 1e-9


def test_ensemble_rank_1_only_for_single_axis_observable() -> None:
    """A single rank-1 covariance produces RANK_1_ONLY with infinite axis."""
    # Covariance with exactly one observable axis (rank 1).
    cov = np.array([[0.04, 0.0], [0.0, 1e12]], np.float64)
    a = _make_result(technique_name='RingEdgeNav', offset=(1.0, 2.0), cov=cov, confidence=0.7)
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    from spindoctor.support.status_reason import NavStatusReason

    assert result.status_reason == NavStatusReason.RANK_1_ONLY
    assert result.sigma_along_unobservable_px == float('inf')


def test_ensemble_unobservable_offset_when_all_share_null_direction() -> None:
    """If every input covariance shares one null direction, fail with the reason."""
    # Both inputs have an enormous along-axis variance (effectively unobservable
    # in u) and a tight v constraint, but with mutually inconsistent v means;
    # the precision-weighted combine is still well-defined (W > 0) so this
    # case demands a literal degenerate combine.  Use rcond large enough to
    # make pinvh project both inputs to the same rank-1 form, then have the
    # u-axis variance be infinite.
    big = 1e30
    cov_a = np.array([[0.04, 0.0], [0.0, big]], np.float64)
    cov_b = np.array([[0.04, 0.0], [0.0, big]], np.float64)
    a = _make_result(technique_name='A', offset=(0.0, 100.0), cov=cov_a, confidence=0.5)
    b = _make_result(technique_name='B', offset=(0.0, -100.0), cov=cov_b, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # v is observable, u is not; the ensemble reports RANK_1_ONLY with
    # ``sigma_along_unobservable_px`` set on the result.
    from spindoctor.support.status_reason import NavStatusReason

    assert result.status_reason == NavStatusReason.RANK_1_ONLY


def test_ensemble_3dof_combines_translation_and_rotation() -> None:
    """Two 3-DoF results agreeing in (dv, du, theta) merge into a 3-DoF NavResult."""
    cov_a = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    cov_b = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_a,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.01,
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.05, 1.95),
        covariance_px2=cov_b,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.012,
        sigma_rotation_rad=0.01,
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.rotation_rad is not None
    assert abs(result.rotation_rad - 0.011) < 1e-3
    assert result.sigma_rotation_rad is not None
    # Two equal-confidence inputs with sigma_theta = 0.01 rad each combine
    # via inverse-variance averaging to sigma = 1 / sqrt(1/0.01**2 + 1/0.01**2)
    # = 0.01 / sqrt(2) ~ 0.0070710678 rad.
    assert result.sigma_rotation_rad == pytest.approx(0.01 / np.sqrt(2.0), rel=1e-6)
    assert result.covariance_px2 is not None
    assert result.covariance_px2.shape == (3, 3)


def test_ensemble_3dof_with_rotation_unobservable_input() -> None:
    """Rotation-unobservable input contributes near-zero info to 3-DoF combine."""
    cov_observable = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    cov_unobservable = embed_rotation_unobservable(np.diag([0.04, 0.04]).astype(np.float64))
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_observable,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.01,
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_unobservable,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=float(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE)),
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.rotation_rad is not None
    # The unobservable input pulls rotation only marginally toward zero;
    # the observable input dominates.
    assert abs(result.rotation_rad - 0.01) < 1e-3


def test_ensemble_3dof_rotation_circular_mean_does_not_collapse_near_wrap() -> None:
    """Antipodal rotations combine on the circle, not to a spurious ~0.

    With a plain Euclidean average, ``+179 deg`` and ``-179 deg`` cancel
    to ~0; the precision-weighted *circular* mean instead reports ~180 deg.
    The ``max_allowed_rotation_deg`` bound is relaxed here so these
    deliberately-large stand-in angles pass the small-angle assertion while
    still exercising the wrap.
    """
    cov = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=np.radians(179.0),
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=np.radians(-179.0),
        sigma_rotation_rad=0.01,
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(max_allowed_rotation_deg=180.0),
    )
    assert result.rotation_rad is not None
    # Circular mean of +-179 deg is ~180 deg (|sin| ~ pi), nowhere near 0.
    assert abs(result.rotation_rad) > np.radians(170.0)


def test_ensemble_3dof_rotation_small_angle_circular_mean_matches_arithmetic() -> None:
    """For genuinely small angles the circular mean equals the arithmetic mean."""
    cov = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.02,
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.04,
        sigma_rotation_rad=0.01,
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(0.03, abs=1e-6)


def test_ensemble_3dof_rotation_over_bound_raises_contract_error() -> None:
    """A 3-DoF rotation beyond max_allowed_rotation_deg raises NavContractError."""
    cov = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=np.radians(10.0),
        sigma_rotation_rad=0.01,
    )
    with pytest.raises(NavContractError, match='small-angle bound'):
        ensemble(
            [a],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )


def test_combine_precision_weighted_rotation_over_bound_raises_contract_error() -> None:
    """The circular-mean combine's small-angle guard raises NavContractError directly."""
    cov = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=np.radians(10.0),
        sigma_rotation_rad=0.01,
    )
    with pytest.raises(NavContractError, match='violates small-angle bound'):
        _combine_precision_weighted(
            [a],
            rcond=1.0e-9,
            max_allowed_rotation_deg=DEFAULT_MAX_ALLOWED_ROTATION_DEG,
        )


def test_mahalanobis_null_space_groups_near_rank_deficient_agreement() -> None:
    """Near-rank-deficient covariances that genuinely agree still group (finite dist)."""
    # One eigenvalue ~1e-7 (nearly unobservable along v), tight along u.
    cov = np.array([[1.0e-7, 0.0], [0.0, 0.04]], np.float64)
    a = _make_result(technique_name='A', offset=(0.0, 0.0), cov=cov, confidence=0.7)
    # Small genuine agreement: 0.1 px apart in u (the observable axis).
    b = _make_result(technique_name='B', offset=(0.0, 0.1), cov=cov, confidence=0.7)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(agreement_pixel_floor=0.0),
    )
    assert result.status == 'success'


def test_mahalanobis_null_axis_junk_does_not_separate_agreeing_rank_1_pair() -> None:
    """Junk along a shared unobservable axis no longer manufactures a conflict.

    Two exactly-singular rank-1 results observe only u and agree there
    exactly; their v components are meaningless fit residue.  The
    agreement metric compares only the intersection of observable
    subspaces, so the 3 px of v junk is ignored and the
    pair groups as a rank-1 consensus.
    """
    cov = np.array([[0.0, 0.0], [0.0, 0.04]], np.float64)
    a = _make_result(technique_name='A', offset=(0.0, 0.0), cov=cov, confidence=0.5)
    b = _make_result(technique_name='B', offset=(3.0, 0.0), cov=cov, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(agreement_pixel_floor=0.0),
    )
    assert result.status == 'success'
    assert result.status_reason.value == 'rank_1_only'
    assert result.excluded_from_consensus == []


def test_mahalanobis_disagreement_along_shared_observable_axis_conflicts() -> None:
    """Two rank-1 results disagreeing along the axis they both observe conflict."""
    cov = np.array([[0.0, 0.0], [0.0, 0.04]], np.float64)
    a = _make_result(technique_name='A', offset=(0.0, 0.0), cov=cov, confidence=0.5)
    # 3 px displacement along u, the observable axis: a genuine standoff.
    b = _make_result(technique_name='B', offset=(0.0, 3.0), cov=cov, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(agreement_pixel_floor=0.0),
    )
    assert result.status == 'conflicted'


def _body_feature_result(
    *,
    technique_name: str,
    feature_id: str,
    offset: tuple[float, float] = (0.0, 0.0),
    cov: np.ndarray | None = None,
    confidence: float = 0.6,
    spurious: bool = False,
) -> NavTechniqueResult:
    """Build a NavTechniqueResult with a body-feature-shaped feature_id.

    Used by the fallback-tier filter tests.  ``source_bodies`` is derived
    from the ``<feature_kind>:<body_name>`` feature_id, mirroring what the
    live body techniques populate from each feature's structured
    ``body_name``.
    """
    if cov is None:
        cov = np.eye(2, dtype=np.float64) * 0.25
    body_name = feature_id.split(':', 1)[1] if ':' in feature_id else ''
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=(feature_id,),
        offset_px=offset,
        covariance_px2=cov,
        confidence=confidence,
        spurious=spurious,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        source_bodies=frozenset({body_name}) if body_name else frozenset(),
    )


def test_ensemble_drops_terminator_when_limb_succeeds_for_same_body() -> None:
    """Non-spurious BodyLimbNav supersedes BodyTerminatorNav on the same body.

    On the bug image this prevented a mis-converged 25-px-off
    terminator from out-voting a clean limb fit.
    """
    limb = _body_feature_result(
        technique_name='BodyLimbNav',
        feature_id='limb_arc:DIONE',
        offset=(-5.0, -16.0),
        confidence=0.5,
    )
    terminator = _body_feature_result(
        technique_name='BodyTerminatorNav',
        feature_id='terminator_arc:DIONE',
        offset=(-5.0, -40.0),
        confidence=0.7,
    )
    result = ensemble(
        [limb, terminator],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # The reported offset must come from limb, not the higher-confidence
    # but mis-converged terminator.
    assert abs(result.offset_px[1] - (-16.0)) < 1.0


def test_ensemble_fallback_drop_uses_source_bodies_not_feature_ids() -> None:
    """CODE-ORCH-004: supersession reads source_bodies, not the feature_id string.

    Both results carry an unparseable feature_id but a structured
    ``source_bodies={'DIONE'}``; the limb primary must still supersede the
    terminator fallback, proving the ensemble no longer depends on the
    feature-id format.
    """
    cov = np.eye(2, dtype=np.float64) * 0.25
    limb = NavTechniqueResult(
        technique_name='BodyLimbNav',
        feature_ids=('opaque-id-1',),
        offset_px=(-5.0, -16.0),
        covariance_px2=cov,
        confidence=0.5,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        source_bodies=frozenset({'DIONE'}),
    )
    terminator = NavTechniqueResult(
        technique_name='BodyTerminatorNav',
        feature_ids=('opaque-id-2',),
        offset_px=(-5.0, -40.0),
        covariance_px2=cov,
        confidence=0.7,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        source_bodies=frozenset({'DIONE'}),
    )
    result = ensemble(
        [limb, terminator],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - (-16.0)) < 1.0


def test_ensemble_keeps_terminator_when_limb_is_spurious() -> None:
    """A spurious primary does not supersede the fallback — terminator runs solo."""
    limb = _body_feature_result(
        technique_name='BodyLimbNav',
        feature_id='limb_arc:DIONE',
        offset=(-5.0, -16.0),
        confidence=0.5,
        spurious=True,
    )
    terminator = _body_feature_result(
        technique_name='BodyTerminatorNav',
        feature_id='terminator_arc:DIONE',
        offset=(-5.0, -40.0),
        confidence=0.7,
    )
    result = ensemble(
        [limb, terminator],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # Terminator is the only viable input; its offset is reported.
    assert abs(result.offset_px[1] - (-40.0)) < 1.0


def test_ensemble_keeps_terminator_for_different_body() -> None:
    """Limb success on body A does not drop terminator on body B."""
    limb = _body_feature_result(
        technique_name='BodyLimbNav',
        feature_id='limb_arc:DIONE',
        offset=(-5.0, -16.0),
        confidence=0.5,
    )
    terminator_other = _body_feature_result(
        technique_name='BodyTerminatorNav',
        feature_id='terminator_arc:RHEA',
        offset=(20.0, 30.0),
        confidence=0.7,
    )
    result = ensemble(
        [limb, terminator_other],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # Both results survive the fallback filter, end up in different
    # agreement groups, and so the ensemble flags conflicted.
    assert result.status in ('conflicted', 'success')
    # Cross-check by counting per-technique entries: both should be
    # preserved on the NavResult for diagnostics.
    names = {r.technique_name for r in result.per_technique}
    assert names == {'BodyLimbNav', 'BodyTerminatorNav'}


def test_ensemble_drops_blob_when_disc_succeeds_for_same_body() -> None:
    """BodyBlobNav is also fallback — a non-spurious BodyDiscCorrelateNav supersedes it."""
    disc = _body_feature_result(
        technique_name='BodyDiscCorrelateNav',
        feature_id='body_disc:MIMAS',
        offset=(2.0, 3.0),
        confidence=0.6,
    )
    # The blob sits far from the disc so a fused answer would be pulled off (2, 3);
    # its high phase keeps the shape-lock witness veto from firing, isolating the
    # supersession drop as the only reason it leaves the fuse.
    blob = NavTechniqueResult(
        technique_name='BodyBlobNav',
        feature_ids=('body_blob:MIMAS',),
        offset_px=(20.0, 30.0),
        covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
        confidence=0.4,
        spurious=False,
        at_edge=False,
        diagnostics=BodyBlobDiagnostics(body_extent_px=80.0, max_phase_angle_deg=155.0),
        source_bodies=frozenset({'MIMAS'}),
    )
    result = ensemble(
        [disc, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - 2.0) < 1.0


def test_ensemble_rejects_mixed_dof_inputs() -> None:
    """Mixing 2-DoF and 3-DoF results raises a ValueError."""
    cov_2d = np.diag([0.04, 0.04]).astype(np.float64)
    cov_3d = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    two = _make_result(technique_name='Two', cov=cov_2d)
    three = NavTechniqueResult(
        technique_name='Three',
        feature_ids=('three:f',),
        offset_px=(0.0, 0.0),
        covariance_px2=cov_3d,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=0.01,
    )
    with pytest.raises(ValueError, match='Mixed-DoF technique results'):
        ensemble(
            [two, three],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )


def test_combine_confidence_weight_ignores_rotation_precision() -> None:
    """CODE-NAV-013: the confidence weight uses positional precision only.

    Two 3-DoF results share an identical (v, u) covariance but differ wildly
    in rotation precision.  The rotation-tight result also carries the *lower*
    confidence.  With the old full-trace weight (px^-2 added to rad^-2) the
    rotation-tight result would dominate and drag the combined confidence down
    to ~its own value; the positional-only weight gives both results equal
    weight so the combine reflects both contributors.
    """
    cov_tight_rot = np.diag([0.25, 0.25, 1e-6]).astype(np.float64)
    cov_loose_rot = np.diag([0.25, 0.25, 1.0]).astype(np.float64)
    rot_tight = NavTechniqueResult(
        technique_name='RotTight',
        feature_ids=('rot_tight:f',),
        offset_px=(0.0, 0.0),
        covariance_px2=cov_tight_rot,
        confidence=0.2,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=1e-3,
    )
    rot_loose = NavTechniqueResult(
        technique_name='RotLoose',
        feature_ids=('rot_loose:f',),
        offset_px=(0.0, 0.0),
        covariance_px2=cov_loose_rot,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=1.0,
    )
    combined = _combine_confidence(
        [rot_tight, rot_loose],
        rcond=1e-6,
        disagreement_penalty=0.3,
        apply_disagreement_penalty=False,
    )
    # Equal positional weights -> weighted_avg = 0.5, n_significant = 2 ->
    # agreement_factor = 1 + 0.5*log2(2) = 1.5 -> combined = 0.75.  Far above
    # the ~0.2 the rotation-tight result would have forced under the old trace.
    assert combined == pytest.approx(0.75, abs=1e-9)


# --- Consensus outlier rejection (issue #124) ---


def _three_agreeing_plus_outlier() -> list[NavTechniqueResult]:
    """Three techniques agreeing at ~(10, 5) plus one mis-converged outlier."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    return [
        _make_result(technique_name='BodyLimbNav', offset=(10.0, 5.0), cov=cov, confidence=0.8),
        _make_result(
            technique_name='BodyDiscCorrelateNav', offset=(10.1, 5.1), cov=cov, confidence=0.8
        ),
        _make_result(technique_name='RingEdgeNav', offset=(10.0, 5.0), cov=cov, confidence=0.7),
        _make_result(
            technique_name='BodyTerminatorNav', offset=(-3.0, 12.0), cov=cov, confidence=0.8
        ),
    ]


def test_ensemble_rejects_lone_outlier_against_consensus() -> None:
    """Three agreeing results plus one far-off dissenter yield consensus, not conflict."""
    result = ensemble(
        _three_agreeing_plus_outlier(),
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.excluded_from_consensus == ['BodyTerminatorNav']
    assert result.offset_px is not None
    # The combined offset comes from the agreeing trio, unperturbed by the outlier.
    assert abs(result.offset_px[0] - 10.0) < 0.2
    assert abs(result.offset_px[1] - 5.0) < 0.2
    # All four techniques remain visible in per_technique for diagnostics.
    assert len(result.per_technique) == 4


def test_ensemble_outlier_rejection_applies_disagreement_penalty() -> None:
    """Excluding an outlier still applies the disagreement penalty."""
    with_outlier = ensemble(
        _three_agreeing_plus_outlier(),
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    unanimous = ensemble(
        _three_agreeing_plus_outlier()[:3],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert unanimous.excluded_from_consensus == []
    assert with_outlier.confidence < unanimous.confidence


def test_ensemble_two_pairs_disagreeing_across_pairs_conflict() -> None:
    """Two internally-agreeing pairs with comparable confidence conflict (no quorum winner)."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    results = [
        _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov, confidence=0.6),
        _make_result(technique_name='TechniqueB', offset=(0.5, 0.0), cov=cov, confidence=0.6),
        _make_result(technique_name='TechniqueC', offset=(30.0, 30.0), cov=cov, confidence=0.55),
        _make_result(technique_name='TechniqueD', offset=(30.5, 30.0), cov=cov, confidence=0.55),
    ]
    result = ensemble(
        results,
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.excluded_from_consensus == ['TechniqueC', 'TechniqueD']


def test_ensemble_dominant_pair_beats_weak_pair_without_conflict() -> None:
    """A runner-up pair far below the winning pair's summed confidence does not conflict."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    results = [
        _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov, confidence=0.9),
        _make_result(technique_name='TechniqueB', offset=(0.5, 0.0), cov=cov, confidence=0.9),
        _make_result(technique_name='TechniqueC', offset=(30.0, 30.0), cov=cov, confidence=0.3),
        _make_result(technique_name='TechniqueD', offset=(30.5, 30.0), cov=cov, confidence=0.3),
    ]
    result = ensemble(
        results,
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.excluded_from_consensus == ['TechniqueC', 'TechniqueD']


def test_ensemble_unanimous_consensus_reports_no_exclusions() -> None:
    """A fully-agreeing input set reports an empty excluded_from_consensus."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    results = [
        _make_result(technique_name='TechniqueA', offset=(1.0, 2.0), cov=cov, confidence=0.8),
        _make_result(technique_name='TechniqueB', offset=(1.05, 2.05), cov=cov, confidence=0.8),
    ]
    result = ensemble(
        results,
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.excluded_from_consensus == []


def test_ensemble_lone_dissenter_half_confidence_is_outlier_not_conflict() -> None:
    """A lone excluded dissenter at half the winner's confidence cannot veto it.

    A singleton 0.8-confidence winner against a singleton 0.4-confidence
    dissenter sits exactly on the relative-gap boundary (gap 0.4 ==
    agreement_gap 0.5 * best 0.8) and must resolve as outlier rejection,
    not a conflict.
    """
    cov = np.eye(2, dtype=np.float64) * 0.25
    best = _make_result(technique_name='TechniqueA', offset=(18.2, 0.1), cov=cov, confidence=0.8)
    dissenter = _make_result(
        technique_name='TechniqueB', offset=(-1.0, 3.5), cov=cov, confidence=0.4
    )
    result = ensemble(
        [best, dissenter],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.excluded_from_consensus == ['TechniqueB']
    assert result.offset_px is not None
    # The winner's offset carries through unperturbed by the dissenter.
    assert result.offset_px[0] == pytest.approx(18.2)
    assert result.offset_px[1] == pytest.approx(0.1)
    # The disagreement penalty still applies to the combined confidence.
    assert result.confidence == pytest.approx(0.8 * 0.7)


def test_ensemble_lone_vs_lone_comparable_confidence_still_conflicts() -> None:
    """A lone-vs-lone standoff with comparable confidences remains a conflict."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    best = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov, confidence=0.5)
    dissenter = _make_result(
        technique_name='TechniqueB', offset=(30.0, 30.0), cov=cov, confidence=0.45
    )
    result = ensemble(
        [best, dissenter],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.confidence_rank == 'conflicted'


def test_ensemble_single_inlier_refine_is_single_star_only() -> None:
    """A lone one-inlier refine is marked single_star_only and capped at low.

    The confidence is set above the high tier's min_confidence boundary and
    the refine's localization sigma is CRLB-tight, so without the mark the
    result would earn high.
    """
    res = _make_result(
        technique_name='StarRefineNav',
        offset=(3.06, -0.02),
        cov=np.eye(2, dtype=np.float64) * 0.01,
        confidence=0.9,
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
    )
    result = ensemble(
        [res],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.status_reason == NavStatusReason.SINGLE_STAR_ONLY
    assert result.confidence_rank == 'low'


def test_ensemble_one_star_unique_match_is_single_star_only() -> None:
    """A lone one-star unique match with tight sigma is single_star_only at low."""
    res = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(3.06, -0.02),
        cov=np.eye(2, dtype=np.float64) * 0.01,
        confidence=0.9,
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )
    result = ensemble(
        [res],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status_reason == NavStatusReason.SINGLE_STAR_ONLY
    assert result.confidence_rank == 'low'
    assert result.matched_star_count == 1


def test_ensemble_two_results_on_one_star_are_one_witness() -> None:
    """A one-star match and the refine of the same star are one witness, not two.

    They share the star, so they are collapsed: the fused sigma is the refine's
    own, not the tighter sigma an independent pair would earn, the confidence is
    the members' precision-weighted blend with no agreement boost, and the result
    is single_star_only.
    """
    match = dataclasses.replace(
        _make_result(
            technique_name='StarUniqueMatchNav',
            offset=(3.06, -0.02),
            cov=np.eye(2, dtype=np.float64) * 900.0,
            confidence=0.58,
            diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
        ),
        feature_ids=('star:UCAC4:1',),
    )
    refine = dataclasses.replace(
        _make_result(
            technique_name='StarRefineNav',
            offset=(3.06, -0.02),
            cov=np.eye(2, dtype=np.float64) * 0.08,
            confidence=0.5,
            diagnostics=StarRefineDiagnostics(n_stars_used=1),
        ),
        feature_ids=('star:UCAC4:1',),
        prior_source_techniques=frozenset({'StarUniqueMatchNav'}),
    )
    result = ensemble(
        [match, refine],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status_reason == NavStatusReason.SINGLE_STAR_ONLY
    assert result.confidence_rank == 'low'
    assert result.sigma_px == pytest.approx((math.sqrt(0.08), math.sqrt(0.08)))
    blend = (0.58 / 900.0 + 0.5 / 0.08) / (1 / 900.0 + 1 / 0.08)
    assert result.confidence == pytest.approx(blend)
    assert result.matched_star_count == 1


def test_ensemble_two_single_star_results_on_different_stars_are_two_stars() -> None:
    """Single-star results on two different stars are not single_star_only but stay medium."""
    cov = np.eye(2, dtype=np.float64) * 0.01
    first = dataclasses.replace(
        _make_result(
            technique_name='StarUniqueMatchNav',
            offset=(3.06, -0.02),
            cov=cov,
            confidence=0.9,
            diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
        ),
        feature_ids=('star:UCAC4:1',),
    )
    second = dataclasses.replace(
        _make_result(
            technique_name='StarRefineNav',
            offset=(3.10, -0.05),
            cov=cov,
            confidence=0.9,
            diagnostics=StarRefineDiagnostics(n_stars_used=1),
        ),
        feature_ids=('star:UCAC4:2',),
    )
    result = ensemble(
        [first, second],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status_reason == NavStatusReason.OK
    assert result.matched_star_count == 2
    assert result.confidence_rank == 'medium'


def test_ensemble_star_results_sharing_a_star_disagree_beyond_tolerance() -> None:
    """Two star results on one star 4 px apart are not fused by the general pixel floor.

    They found different detections of the same star, so one of them
    misidentified it; the frame is reported conflicted rather than fused.
    """
    cov = np.eye(2, dtype=np.float64) * 0.09
    field = dataclasses.replace(
        _make_result(
            technique_name='StarFieldFromCatalogNav',
            offset=(3.85, -35.14),
            cov=cov,
            confidence=0.6,
            diagnostics=StarFieldDiagnostics(n_inliers=3),
        ),
        feature_ids=('star:a', 'star:b', 'star:c'),
    )
    match = dataclasses.replace(
        _make_result(
            technique_name='StarUniqueMatchNav',
            offset=(4.63, -30.68),
            cov=cov,
            confidence=0.69,
            diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
        ),
        feature_ids=('star:b', 'star:c'),
    )
    result = ensemble(
        [field, match],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'


def test_ensemble_single_star_plus_body_technique_keeps_high() -> None:
    """A single-star result cross-checked by a non-star technique may earn high."""
    cov = np.eye(2, dtype=np.float64) * 0.01
    refine = _make_result(
        technique_name='StarRefineNav',
        offset=(3.06, -0.02),
        cov=cov,
        confidence=0.6,
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
    )
    limb = _make_result(
        technique_name='BodyLimbNav',
        offset=(3.10, -0.05),
        cov=cov,
        confidence=0.6,
        diagnostics=BodyLimbDiagnostics(),
    )
    result = ensemble(
        [refine, limb],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.confidence_rank == 'high'


def test_ensemble_multi_star_refine_keeps_high() -> None:
    """A refine with several inliers is independently constrained; high stays.

    The confidence sits above the high tier's 0.85 boundary so the tier
    outcome isolates the multi-inlier gate, not the boundary.
    """
    res = _make_result(
        technique_name='StarRefineNav',
        offset=(3.06, -0.02),
        cov=np.eye(2, dtype=np.float64) * 0.01,
        confidence=0.9,
        diagnostics=StarRefineDiagnostics(n_stars_used=4),
    )
    result = ensemble(
        [res],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.confidence_rank == 'high'


def test_ensemble_rank_1_ring_and_full_rank_blob_agreeing_radially_fuse() -> None:
    """A rank-1 ring edge and a full-rank blob that agree radially group.

    The ring result observes only v (its u variance is exactly zero under
    the Moore-Penrose convention) and carries junk in its u component;
    the blob is a full-rank absolute fix.  They agree in v, so the
    rank-aware metric groups them and the fusion combines the ring's
    radial precision with the blob's along-edge constraint into a
    full-rank result.
    """
    ring_cov = np.array([[0.04, 0.0], [0.0, 0.0]], np.float64)
    blob_cov = np.eye(2, dtype=np.float64) * 0.09
    ring = _make_result(
        technique_name='RingEdgeNav', offset=(1.0, 50.0), cov=ring_cov, confidence=0.9
    )
    blob = _make_result(
        technique_name='BodyBlobNav', offset=(1.1, 2.0), cov=blob_cov, confidence=0.4
    )
    result = ensemble(
        [ring, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.status_reason.value == 'ok'
    assert result.excluded_from_consensus == []
    assert result.offset_px is not None
    # v combines both (ring dominates); u comes from the blob alone, not
    # the ring's junk 50.0.
    assert abs(result.offset_px[0] - 1.03) < 0.05
    assert result.offset_px[1] == pytest.approx(2.0)
    assert result.sigma_along_unobservable_px is None


def test_ensemble_rank_1_ring_disagreeing_radially_does_not_group() -> None:
    """A rank-1 ring edge that disagrees along its observable axis stays excluded."""
    ring_cov = np.array([[0.04, 0.0], [0.0, 0.0]], np.float64)
    blob_cov = np.eye(2, dtype=np.float64) * 0.09
    ring = _make_result(
        technique_name='RingEdgeNav', offset=(9.0, 50.0), cov=ring_cov, confidence=0.9
    )
    blob = _make_result(
        technique_name='BodyBlobNav', offset=(1.1, 2.0), cov=blob_cov, confidence=0.4
    )
    result = ensemble(
        [ring, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # The 7.9 px radial disagreement is genuine; the ring wins the
    # consensus and the blob is excluded as an outlier.
    assert result.excluded_from_consensus == ['BodyBlobNav']


def test_blob_outlier_dropped_against_two_star_consensus() -> None:
    """A lone blob a corroborated two-star fix contradicts is dropped, not conflicted.

    Mirrors N1530185128: a two-star ``StarUniqueMatchNav`` on the truth and a
    ``BodyBlobNav`` brightness centroid pixels away.  Without the drop the lone
    blob's confidence is close enough to force the lone-vs-lone conflict; the
    drop removes it so the star fix commits, keeps the fused offset on the star
    solution, and leaves the blob out of the consensus.
    """
    star = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(18.0, 0.0),
        confidence=0.706,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    blob = _make_result(
        technique_name='BodyBlobNav',
        offset=(-1.0, 3.5),
        confidence=0.4,
        diagnostics=BodyBlobDiagnostics(body_extent_px=180.0, max_phase_angle_deg=28.0),
    )
    result = ensemble(
        [star, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # The blob is dropped, so the star fix commits instead of conflicting.
    assert result.status == 'success'
    # The dropped blob does not drag the committed offset off the star fix.
    assert result.offset_px is not None
    assert result.offset_px[0] == pytest.approx(18.0, abs=0.5)
    # The dropped blob is not a consensus member.
    assert result.consensus_techniques == ['StarUniqueMatchNav']


def test_blob_star_disagreement_floor_zero_disables_the_drop() -> None:
    """A zero floor disables the drop, so the contradicting blob still conflicts.

    Same scene as the drop test, but ``blob_star_disagreement_floor_px=0.0``
    turns the drop off: the lone blob is retained and forces the lone-vs-lone
    conflict, confirming the tunable's documented disable behavior.
    """
    star = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(18.0, 0.0),
        confidence=0.706,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    blob = _make_result(
        technique_name='BodyBlobNav',
        offset=(-1.0, 3.5),
        confidence=0.4,
        diagnostics=BodyBlobDiagnostics(body_extent_px=180.0, max_phase_angle_deg=28.0),
    )
    result = ensemble(
        [star, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=EnsembleConfig(blob_star_disagreement_floor_px=0.0),
    )
    assert result.status == 'conflicted'


def test_blob_agreeing_with_star_consensus_is_kept() -> None:
    """A blob that agrees with the star fix within the floor is not dropped."""
    star = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(18.0, 0.0),
        confidence=0.706,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    blob = _make_result(
        technique_name='BodyBlobNav',
        offset=(18.4, 0.3),
        confidence=0.4,
        diagnostics=BodyBlobDiagnostics(body_extent_px=180.0, max_phase_angle_deg=28.0),
    )
    result = ensemble(
        [star, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert 'BodyBlobNav' in result.consensus_techniques


def test_blob_outlier_kept_against_one_star_consensus_still_conflicts() -> None:
    """A one-star fix is not a corroborated consensus, so the blob is retained.

    A single-star unique match cannot cross-check its own identification, so it
    does not outrank the blob: the drop must not fire, and the lone-vs-lone
    conflict between the two weak, disagreeing witnesses stands.
    """
    star = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(18.0, 0.0),
        confidence=0.706,
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )
    blob = _make_result(
        technique_name='BodyBlobNav',
        offset=(-1.0, 3.5),
        confidence=0.4,
        diagnostics=BodyBlobDiagnostics(body_extent_px=180.0, max_phase_angle_deg=28.0),
    )
    result = ensemble(
        [star, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'


def test_blob_star_disagreement_floor_rejects_negative() -> None:
    """A negative blob/star disagreement floor is rejected at construction."""
    with pytest.raises(ValueError, match='blob_star_disagreement_floor_px must be non-negative'):
        EnsembleConfig(blob_star_disagreement_floor_px=-1.0)


def test_ensemble_rotation_sentinel_does_not_zero_translation_offset() -> None:
    """A 3-DoF result with the rotation-unobservable sentinel keeps its offset.

    pinvh's relative eigenvalue cutoff against the 1e15 rotation sentinel
    used to truncate the genuine translation information, collapsing the
    fused offset to (0, 0) and mislabeling the result rank_1_only (seen
    on a Galileo one-star frame and the flat-ring frames).
    """
    cov = embed_rotation_unobservable(np.diag([0.04, 0.04]).astype(np.float64))
    res = NavTechniqueResult(
        technique_name='StarUniqueMatchNav',
        feature_ids=('StarUniqueMatchNav:f1',),
        offset_px=(-12.3, -13.5),
        covariance_px2=cov,
        confidence=0.6,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=float(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE)),
    )
    result = ensemble(
        [res],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    assert result.offset_px[0] == pytest.approx(-12.3)
    assert result.offset_px[1] == pytest.approx(-13.5)
    # An unobservable rotation does not make the translation fix rank-1.
    assert result.status_reason.value == 'ok'
    assert result.sigma_along_unobservable_px is None


def test_ensemble_rank_deficient_fused_result_caps_at_medium() -> None:
    """A rank-1 fused covariance never earns the high tier."""
    ring_cov = np.array([[0.0004, 0.0], [0.0, 0.0]], np.float64)
    res = _make_result(
        technique_name='RingEdgeNav', offset=(0.0, 6.35), cov=ring_cov, confidence=0.95
    )
    result = ensemble(
        [res],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.status_reason.value == 'rank_1_only'
    assert result.confidence_rank == 'medium'


def _refine_result(
    *,
    offset: tuple[float, float],
    confidence: float = 0.5,
    prior_sources: frozenset[str] = frozenset(),
    cov: np.ndarray | None = None,
    n_stars_used: int = 1,
) -> NavTechniqueResult:
    """Build a pass-2 StarRefineNav result seeded by ``prior_sources``."""
    return NavTechniqueResult(
        technique_name='StarRefineNav',
        feature_ids=('StarRefineNav:f1',),
        offset_px=offset,
        covariance_px2=cov if cov is not None else np.eye(2, dtype=np.float64) * 0.25,
        confidence=confidence,
        spurious=False,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(n_stars_used=n_stars_used),
        prior_source_techniques=prior_sources,
    )


def test_ensemble_prior_descendant_does_not_boost_confidence() -> None:
    """A single-star refine seeded by its groupmate adds no agreement boost.

    The N1492091163 anatomy: a wrong pass-1 result seeds the prior, the
    1-star refine locks onto whatever sits near the (wrong) predicted
    position, and the pair's 'agreement' must not raise the combined
    confidence above what the pass-1 result carries alone.  The seeded
    single-star refine is dropped from the combine entirely, so
    the confidence is the pass-1 result's own, neither boosted nor diluted.
    """
    ring = _make_result(technique_name='RingEdgeNav', offset=(6.7, -118.6), confidence=0.9)
    tagged = _refine_result(offset=(6.8, -118.5), prior_sources=frozenset({'RingEdgeNav'}))
    untagged = _refine_result(offset=(6.8, -118.5))
    with_tag = ensemble(
        [ring, tagged],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    without_tag = ensemble(
        [ring, untagged],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert with_tag.status == 'success'
    assert without_tag.confidence > with_tag.confidence
    # The seeded single-star refine is dropped, leaving the ring result alone:
    # its own 0.9, with no 2-member agreement factor and no 0.5-dilution.
    assert with_tag.confidence == pytest.approx(0.9)
    assert without_tag.confidence == pytest.approx(0.99)


def test_ensemble_seeded_single_star_refine_does_not_drag_offset() -> None:
    """A seeded single-star refine no longer pulls the fused offset.

    The reopened N1572105349 anatomy: pass-1 techniques agree at the truth,
    and a single-inlier StarRefine seeded from that prior sits off to one side.
    Because it merely re-observes its own seed and adds no independent
    constraint, it is dropped from the combine, so the fused offset stays on
    the pass-1 result rather than being dragged toward the refine.
    """
    ring = _make_result(technique_name='RingEdgeNav', offset=(6.0, -118.0), confidence=0.9)
    refine = _refine_result(offset=(7.0, -119.0), prior_sources=frozenset({'RingEdgeNav'}))
    result = ensemble(
        [ring, refine],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # The refine is dropped: the offset is the ring result's, not the mean.
    assert result.offset_px[0] == pytest.approx(6.0)
    assert result.offset_px[1] == pytest.approx(-118.0)
    # The full consensus membership is still reported for transparency.
    assert result.consensus_techniques == ['RingEdgeNav', 'StarRefineNav']


def test_ensemble_seeded_multi_star_refine_still_refines_the_offset() -> None:
    """A seeded multi-star refine keeps contributing its precision.

    Only a *single*-star seeded refine is dropped: a multi-star refine carries
    independent catalog matches, so it still refines the fused offset even
    though the ensemble denies it a corroboration vote.
    """
    ring = _make_result(technique_name='RingEdgeNav', offset=(6.0, -118.0), confidence=0.9)
    refine = _refine_result(
        offset=(7.0, -119.0),
        prior_sources=frozenset({'RingEdgeNav'}),
        n_stars_used=5,
    )
    result = ensemble(
        [ring, refine],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # Equal covariances: the combined offset is the arithmetic mean.
    assert result.offset_px[0] == pytest.approx(6.5)
    assert result.offset_px[1] == pytest.approx(-118.5)


def test_ensemble_two_ring_techniques_are_one_witness_not_two() -> None:
    """RingEdge + RingAnnulus on one catalog earn no independence boost.

    Two ring techniques read the same catalog model, so their agreement is not
    independent corroboration.  Collapsing them to one witness must leave the
    combined confidence at a single witness's level, well below the two-witness
    boost an independent body+ring pair would earn on the same numbers.
    """
    tight = np.eye(2, dtype=np.float64) * 0.04
    edge = _make_result(
        technique_name='RingEdgeNav',
        offset=(5.0, -6.0),
        cov=tight,
        confidence=0.8,
        diagnostics=RingEdgeDiagnostics(),
    )
    annulus = _make_result(
        technique_name='RingAnnulusNav',
        offset=(5.05, -6.05),
        cov=tight,
        confidence=0.8,
        diagnostics=RingAnnulusDiagnostics(),
    )
    ring_pair = ensemble(
        [edge, annulus],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # Control: an independent body+ring pair on the same numbers does earn the
    # two-witness boost, so it lands at the confidence cap.
    body = _make_result(
        technique_name='BodyLimbNav',
        offset=(5.05, -6.05),
        cov=tight,
        confidence=0.8,
        diagnostics=BodyLimbDiagnostics(),
    )
    independent_pair = ensemble(
        [edge, body],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert ring_pair.status == 'success'
    # No agreement boost: the collapsed witness keeps its own 0.8, while the
    # independent pair is boosted to the cap.
    assert ring_pair.confidence == pytest.approx(0.8)
    assert independent_pair.confidence > ring_pair.confidence
    # And the collapsed covariance is not double-tightened: a single witness's
    # sigma, larger than the fused two-witness sigma.
    assert ring_pair.covariance_px2 is not None
    assert independent_pair.covariance_px2 is not None
    assert ring_pair.covariance_px2[0, 0] > independent_pair.covariance_px2[0, 0]


def test_ensemble_collapsed_ring_pair_has_no_spurious_quorum() -> None:
    """A collapsed ring pair counts as one witness in the conflict logic.

    RingEdge + RingAnnulus agree (and collapse to one witness), and a lone
    dissenter of comparable confidence is excluded. Because the collapsed pair
    is one witness, this is a lone-vs-lone standoff and must go conflicted --
    not be suppressed by the pair spuriously claiming a two-member quorum.
    """
    tight = np.eye(2, dtype=np.float64) * 0.04
    loose = np.eye(2, dtype=np.float64) * 0.25
    edge = _make_result(
        technique_name='RingEdgeNav',
        offset=(0.0, 0.0),
        cov=tight,
        confidence=0.6,
        diagnostics=RingEdgeDiagnostics(),
    )
    annulus = _make_result(
        technique_name='RingAnnulusNav',
        offset=(0.05, 0.05),
        cov=tight,
        confidence=0.6,
        diagnostics=RingAnnulusDiagnostics(),
    )
    dissenter = _make_result(
        technique_name='TechniqueB', offset=(30.0, 30.0), cov=loose, confidence=0.55
    )
    result = ensemble(
        [edge, annulus, dissenter],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.excluded_from_consensus == ['TechniqueB']


def test_ensemble_scattered_light_disc_limb_are_one_witness() -> None:
    """Disc + limb on a scattered-light frame do not inflate the tier.

    On a frame whose background-gradient score clears the scattered-light
    threshold, disc and limb measure the same veiling ramp, so their agreement
    is not independent.  Collapsing them must keep the tier honest, whereas the
    identical numbers on a clean frame legitimately earn the two-witness boost
    and a high tier.
    """
    tight = np.eye(2, dtype=np.float64) * 0.04
    disc = _make_result(
        technique_name='BodyDiscCorrelateNav',
        offset=(5.0, -6.0),
        cov=tight,
        confidence=0.8,
        diagnostics=BodyDiscDiagnostics(),
    )
    limb = _make_result(
        technique_name='BodyLimbNav',
        offset=(5.05, -6.05),
        cov=tight,
        confidence=0.8,
        diagnostics=BodyLimbDiagnostics(),
    )
    disc = dataclasses.replace(disc, source_bodies=frozenset({'MIMAS'}))
    limb = dataclasses.replace(limb, source_bodies=frozenset({'MIMAS'}))
    scattered = ensemble(
        [disc, limb],
        feature_inventory=[],
        image_classifier=_classifier(background_gradient_score=8.0),
        provenance=_provenance(),
    )
    clean = ensemble(
        [disc, limb],
        feature_inventory=[],
        image_classifier=_classifier(background_gradient_score=1.0),
        provenance=_provenance(),
    )
    assert scattered.status == 'success'
    assert clean.status == 'success'
    # Scattered: one witness, no boost -> medium tier at its own 0.8.
    assert scattered.confidence == pytest.approx(0.8)
    assert scattered.confidence_rank != 'high'
    # Clean: two independent witnesses, boosted to the cap and a high tier.
    assert clean.confidence > scattered.confidence
    assert clean.confidence_rank == 'high'


def test_ensemble_descendant_backed_winner_is_singleton_in_standoff() -> None:
    """A winner echoed only by its own refine has no quorum against a dissenter.

    Best subset {A 0.6, refine-of-A 0.5} versus a lone dissenter at 0.55:
    with the descendant's vote removed, this is a lone-vs-lone standoff
    with a relative gap of (0.6 - 0.55) / 0.6 << agreement_gap, so the
    result is conflicted; counting the refine as a second member would
    have declared quorum and outlier-rejected the dissenter.
    """
    cov = np.eye(2, dtype=np.float64) * 0.25
    a = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), cov=cov, confidence=0.6)
    refine = _refine_result(
        offset=(0.1, 0.1), confidence=0.5, prior_sources=frozenset({'TechniqueA'})
    )
    dissenter = _make_result(
        technique_name='TechniqueB', offset=(30.0, 30.0), cov=cov, confidence=0.55
    )
    result = ensemble(
        [a, refine, dissenter],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.excluded_from_consensus == ['TechniqueB']


def test_ensemble_descendant_backed_runner_up_has_no_quorum() -> None:
    """An excluded pair of (result, its own refine) is a lone dissenter, not a quorum."""
    cov = np.eye(2, dtype=np.float64) * 0.25
    a1 = _make_result(technique_name='TechniqueA1', offset=(0.0, 0.0), cov=cov, confidence=0.6)
    a2 = _make_result(technique_name='TechniqueA2', offset=(0.1, 0.0), cov=cov, confidence=0.6)
    b = _make_result(technique_name='TechniqueB', offset=(30.0, 30.0), cov=cov, confidence=0.5)
    b_refine = _refine_result(
        offset=(30.1, 30.0), confidence=0.4, prior_sources=frozenset({'TechniqueB'})
    )
    result = ensemble(
        [a1, a2, b, b_refine],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # Without the corroboration fix the excluded pair counts as a
    # two-member runner-up (summed 0.9 against best 1.2, gap 0.3 < 0.5)
    # and forces the conflicted branch; as a lone dissenter it is
    # outlier-rejected instead.
    assert result.status == 'success'
    assert result.excluded_from_consensus == ['TechniqueB', 'StarRefineNav']


def test_ensemble_shape_lock_veto_reports_conflicted() -> None:
    """A geometric consensus the low-phase blob contradicts is reported conflicted."""
    disc = NavTechniqueResult(
        technique_name='BodyDiscCorrelateNav',
        feature_ids=('disc:HYPERION',),
        offset_px=(-14.5, -6.5),
        covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyDiscDiagnostics(),
        source_bodies=frozenset({'HYPERION'}),
    )
    blob = NavTechniqueResult(
        technique_name='BodyBlobNav',
        feature_ids=('blob:HYPERION',),
        offset_px=(-3.8, -3.1),
        covariance_px2=np.eye(2, dtype=np.float64) * 4.0,
        confidence=0.4,
        spurious=False,
        at_edge=False,
        diagnostics=BodyBlobDiagnostics(body_extent_px=150.0, max_phase_angle_deg=30.0),
        source_bodies=frozenset({'HYPERION'}),
    )
    result = ensemble(
        [disc, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.status_reason.value == 'body_shape_lock_suspect'


def test_ensemble_shape_lock_suppressed_by_corroborating_star_fix() -> None:
    """A star fix corroborating the geometry commits, not conflicts (N1806609736).

    Iapetus's albedo dichotomy drags the brightness centroid off the true disc
    center, so the low-phase blob disagrees with a correct limb fit; a two-star
    ``StarUniqueMatchNav`` independently confirms the geometry within the
    grouping floor, so the blob is the outlier and the frame commits instead of
    reporting the shape-lock conflict.
    """
    limb = NavTechniqueResult(
        technique_name='BodyLimbNav',
        feature_ids=('limb:IAPETUS',),
        offset_px=(1.5, 12.0),
        covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        source_bodies=frozenset({'IAPETUS'}),
    )
    star = _make_result(
        technique_name='StarUniqueMatchNav',
        offset=(1.5, 11.6),
        confidence=0.7,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    blob = NavTechniqueResult(
        technique_name='BodyBlobNav',
        feature_ids=('blob:IAPETUS',),
        offset_px=(1.5, 25.0),
        covariance_px2=np.eye(2, dtype=np.float64) * 4.0,
        confidence=0.4,
        spurious=False,
        at_edge=False,
        diagnostics=BodyBlobDiagnostics(body_extent_px=150.0, max_phase_angle_deg=30.0),
        source_bodies=frozenset({'IAPETUS'}),
    )
    result = ensemble(
        [limb, star, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px is not None
    # The committed offset rests on the limb/star geometry near du=11.8, not on
    # the albedo-dragged blob centroid at du=25.0.
    assert result.offset_px[0] == pytest.approx(1.5, abs=0.5)
    assert result.offset_px[1] == pytest.approx(11.8, abs=1.0)
    # The contradicting blob is not a consensus member.
    assert 'BodyBlobNav' not in result.consensus_techniques


def test_ensemble_lone_blob_collapsed_regime_veto_reports_failed() -> None:
    """A lone blob with a spurious geometric sibling on the same body fails."""
    disc = NavTechniqueResult(
        technique_name='BodyDiscCorrelateNav',
        feature_ids=('disc:TITAN',),
        offset_px=(0.5, 0.5),
        covariance_px2=np.eye(2, dtype=np.float64) * 0.25,
        confidence=0.0,
        spurious=True,
        at_edge=False,
        diagnostics=BodyDiscDiagnostics(),
        source_bodies=frozenset({'TITAN'}),
    )
    blob = NavTechniqueResult(
        technique_name='BodyBlobNav',
        feature_ids=('blob:TITAN',),
        offset_px=(3.2, 29.3),
        covariance_px2=np.eye(2, dtype=np.float64) * 4.0,
        confidence=0.4,
        spurious=False,
        at_edge=False,
        diagnostics=BodyBlobDiagnostics(body_extent_px=170.0, max_phase_angle_deg=155.0),
        source_bodies=frozenset({'TITAN'}),
    )
    result = ensemble(
        [disc, blob],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'lone_blob_in_collapsed_regime'


def test_ensemble_untagged_results_unchanged_by_lineage_logic() -> None:
    """Results without prior_source_techniques keep full corroboration semantics."""
    a = _make_result(technique_name='TechniqueA', offset=(1.0, 2.0), confidence=0.85)
    b = _make_result(technique_name='TechniqueB', offset=(1.05, 2.05), confidence=0.85)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.consensus_techniques == ['TechniqueA', 'TechniqueB']
    # Two independent members still earn the 2-member agreement factor.
    assert result.confidence == pytest.approx(0.99)


def test_config_rejects_negative_body_witness_agreement_sigma() -> None:
    """A negative Mahalanobis threshold would veto nearly every separation."""
    with pytest.raises(ValueError, match='body_witness_agreement_sigma must be non-negative'):
        EnsembleConfig(body_witness_agreement_sigma=-1.0)


def test_config_rejects_non_finite_body_witness_disagreement_floor() -> None:
    """A non-finite disagreement floor is rejected rather than silently accepted."""
    with pytest.raises(ValueError, match='body_witness_disagreement_floor_px must be finite'):
        EnsembleConfig(body_witness_disagreement_floor_px=float('inf'))


def test_config_rejects_out_of_range_shape_lock_phase() -> None:
    """A phase ceiling outside [0, 180] deg is rejected."""
    with pytest.raises(ValueError, match=r'body_witness_shape_lock_max_phase_deg must lie'):
        EnsembleConfig(body_witness_shape_lock_max_phase_deg=200.0)


def test_config_rejects_inverted_phase_window() -> None:
    """The shape-lock ceiling must sit below the collapse floor."""
    with pytest.raises(ValueError, match='must be below'):
        EnsembleConfig(
            body_witness_shape_lock_max_phase_deg=95.0,
            body_witness_collapse_min_phase_deg=90.0,
        )


def test_ensemble_collapsed_star_set_reports_the_blended_confidence() -> None:
    """Collapsing results on shared stars cannot raise the tier above the members' blend.

    The seven-star refine represents the set, but the set's confidence is the
    precision-weighted mean of the members' confidences, as the fusion of the members
    would give before any agreement boost; the refine's own 0.95 alone would earn high.
    """
    stars = tuple(f'star:{i}' for i in range(7))
    field = dataclasses.replace(
        _make_result(
            technique_name='StarFieldFromCatalogNav',
            offset=(3.0, -1.0),
            cov=np.eye(2, dtype=np.float64) * 0.09,
            confidence=0.6,
            diagnostics=StarFieldDiagnostics(n_inliers=7),
        ),
        feature_ids=stars,
    )
    match = dataclasses.replace(
        _make_result(
            technique_name='StarUniqueMatchNav',
            offset=(3.1, -1.1),
            cov=np.eye(2, dtype=np.float64) * 0.05,
            confidence=0.8,
            diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
        ),
        feature_ids=stars[:2],
    )
    refine = dataclasses.replace(
        _make_result(
            technique_name='StarRefineNav',
            offset=(3.05, -1.05),
            cov=np.eye(2, dtype=np.float64) * 0.09,
            confidence=0.95,
            diagnostics=StarRefineDiagnostics(n_stars_used=7),
        ),
        feature_ids=stars,
    )
    result = ensemble(
        [field, match, refine],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    weights = (1 / 0.09, 1 / 0.05, 1 / 0.09)
    blend = (weights[0] * 0.6 + weights[1] * 0.8 + weights[2] * 0.95) / sum(weights)
    assert result.confidence == pytest.approx(blend)
    assert result.confidence_rank == 'medium'
