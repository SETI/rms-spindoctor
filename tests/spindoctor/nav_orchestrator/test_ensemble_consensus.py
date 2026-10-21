"""Tests for the consensus selection's fusion-compromise test and its use in the combine."""

import numpy as np
import pytest

from spindoctor.nav_orchestrator.ensemble import EnsembleConfig, ensemble
from spindoctor.nav_orchestrator.ensemble_consensus import (
    fusion_compromise_px,
    leave_out_compromising_fusions,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import (
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult


def _ring_result(along_edge_sigma_px: float) -> NavTechniqueResult:
    """A gently curved ring-edge answer 38 px along the edge (v) from the stars.

    Its sigma across the edge (u) is 0.2 px.
    """
    return NavTechniqueResult(
        technique_name='RingEdgeNav',
        feature_ids=('ring_edge:SATURN:a_ring:OER',),
        offset_px=(47.46, -11.01),
        covariance_px2=np.diag([along_edge_sigma_px**2, 0.2**2]),
        confidence=0.75,
        spurious=False,
        at_edge=False,
        diagnostics=RingEdgeDiagnostics(),
    )


def _one_star_result() -> NavTechniqueResult:
    """A one-star match whose sigma is the size of its own offset."""
    return NavTechniqueResult(
        technique_name='StarUniqueMatchNav',
        feature_ids=('star:UCAC4:1',),
        offset_px=(9.34, -11.12),
        covariance_px2=np.eye(2) * 14.5**2,
        confidence=0.70,
        spurious=False,
        at_edge=False,
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )


def _star_field_result() -> NavTechniqueResult:
    """A five-star refine located to a tenth of a pixel."""
    return NavTechniqueResult(
        technique_name='StarRefineNav',
        feature_ids=('star:UCAC4:1', 'star:UCAC4:2', 'star:UCAC4:3'),
        offset_px=(9.34, -11.12),
        covariance_px2=np.eye(2) * 0.1**2,
        confidence=0.9,
        spurious=False,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(n_stars_used=5),
    )


def _run(results: list[NavTechniqueResult]) -> NavResult:
    """Run the ensemble with default configuration."""
    return ensemble(
        results,
        feature_inventory=[],
        image_classifier=NavImageClassifierResult(
            image_class='clean',
            saturation_frac=0.0,
            missing_frac=0.0,
            noise_sigma=1.0,
            max_dn=10.0,
        ),
        provenance=Provenance(
            spindoctor_version='0.0.0',
            image_et=0.0,
            pipeline_run_iso8601='2026-10-19T00:00:00Z',
        ),
    )


def test_fusion_of_a_precise_and_a_vague_estimate_lands_on_the_precise_one() -> None:
    """With variances 0.01 and 100 the fusion is 1e-3 of the separation from the precise one."""
    compromise = fusion_compromise_px(np.array([10.0]), np.array([[0.01]]), np.array([[100.0]]))
    assert compromise == pytest.approx(10.0 * 0.01 / 100.01, rel=1e-9)


def test_fusion_of_two_equally_vague_estimates_lands_halfway() -> None:
    """Equal covariances put the fusion at the midpoint."""
    compromise = fusion_compromise_px(np.array([30.0, 40.0]), np.eye(2) * 200.0, np.eye(2) * 200.0)
    assert compromise == pytest.approx(25.0, rel=1e-9)


def test_fusion_of_complementary_estimates_compromises_neither() -> None:
    """A ring edge precise across it and a centroid precise along it each set their own axis.

    Along the edge (v) the centroid's 3.5 px beats the ring's 81 px; across it (u)
    the ring's 0.8 px beats the centroid's 3.5 px.  Per direction the fusion
    lands on the precise member, so the compromise is a fraction of a pixel.
    """
    delta = np.array([9.0, 2.0])
    compromise = fusion_compromise_px(delta, np.diag([81.0**2, 0.8**2]), np.eye(2) * 3.5**2)
    expected = np.hypot(9.0 * 3.5**2 / (81.0**2 + 3.5**2), 2.0 * 0.8**2 / (0.8**2 + 3.5**2))
    assert compromise == pytest.approx(expected, rel=1e-9)


def test_two_vague_answers_far_apart_are_a_conflict_not_a_compromise() -> None:
    """A ring answer 38 px along the edge and a one-star match are not fused.

    Their Mahalanobis distance is under 2, so they are statistically consistent,
    but their fusion would lie about 18 px from each.
    """
    result = _run([_ring_result(15.0), _one_star_result()])
    assert result.status == 'conflicted'


def test_a_precise_star_fix_sets_the_ring_edge_along_edge_component() -> None:
    """A ring answer as far off along a weaker edge fuses with a precise star fix.

    Each axis is the precision-weighted mean: along the edge (v) the star fix all
    but sets it, and across the edge (u) the two share it.
    """
    result = _run([_ring_result(25.0), _star_field_result()])
    expected_v = (47.46 / 25.0**2 + 9.34 / 0.1**2) / (1.0 / 25.0**2 + 1.0 / 0.1**2)
    expected_u = (-11.01 / 0.2**2 - 11.12 / 0.1**2) / (1.0 / 0.2**2 + 1.0 / 0.1**2)
    assert result.status == 'success'
    assert result.offset_px == pytest.approx((expected_v, expected_u), abs=1e-9)


def _two_star_result(offset: tuple[float, float], sigma_px: float) -> NavTechniqueResult:
    """A two-star match with an isotropic sigma."""
    return NavTechniqueResult(
        technique_name='StarUniqueMatchNav',
        feature_ids=('star:UCAC4:1', 'star:UCAC4:2'),
        offset_px=offset,
        covariance_px2=np.eye(2) * sigma_px**2,
        confidence=0.7,
        spurious=False,
        at_edge=False,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )


def _fused_v(ring: NavTechniqueResult, star: NavTechniqueResult) -> float:
    """The precision-weighted v of a ring and a star result with diagonal covariances."""
    w_ring = 1.0 / float(ring.covariance_px2[0, 0])
    w_star = 1.0 / float(star.covariance_px2[0, 0])
    return (w_ring * ring.offset_px[0] + w_star * star.offset_px[0]) / (w_ring + w_star)


def test_a_pair_past_the_floor_compromising_under_a_pixel_is_fused() -> None:
    """Two answers 8 px apart whose fusion compromises by 0.85 px still agree.

    The ring is 4 px uncertain along the edge (v) and the two-star match 1.38 px,
    so the pair is 1.89 sigma apart and the fusion lies 0.85 px from the match,
    inside the 1 px compromise limit.
    """
    ring = _ring_result(4.0)
    star = _two_star_result((ring.offset_px[0] - 8.0, ring.offset_px[1]), 1.9**0.5)
    compromise = fusion_compromise_px(
        np.array([8.0, 0.0]), ring.covariance_px2, star.covariance_px2
    )
    assert compromise == pytest.approx(8.0 * 1.9 / 17.9, rel=1e-9)
    result = _run([ring, star])
    assert result.offset_px is not None
    assert result.offset_px[0] == pytest.approx(_fused_v(ring, star), abs=1e-6)


def test_a_pair_inside_the_floor_agrees_however_vague() -> None:
    """Two answers 4 px apart agree by the pixel floor although fusing them compromises 2 px."""
    ring = _ring_result(15.0)
    star = _two_star_result((ring.offset_px[0] - 4.0, ring.offset_px[1]), 15.0)
    result = _run([ring, star])
    assert result.offset_px is not None
    assert result.offset_px[0] == pytest.approx(_fused_v(ring, star), abs=1e-6)


def test_fusion_of_estimates_with_crossed_axes_slides_off_both() -> None:
    """Two estimates whose precise axes differ fuse at a point several pixels from both.

    A ring edge precise across a slightly tilted edge (0.2 px) and a refine precise in u
    (0.32 px) are 2.8 px apart; the information-form fusion slides along the edge to
    about 7 px from each, which an axis-by-axis account (0.65 px) does not see.
    """
    tilt = np.array([[-1.0, -0.09], [-0.09, 1.0]]) / np.hypot(1.0, 0.09)
    cov_ring = tilt @ np.diag([7.18**2, 0.20**2]) @ tilt.T
    cov_refine = np.diag([2.65**2, 0.32**2])
    delta = np.array([-0.85, -2.64])
    info_ring, info_refine = np.linalg.inv(cov_ring), np.linalg.inv(cov_refine)
    fused = np.linalg.solve(info_ring + info_refine, info_refine @ delta)
    _, axes = np.linalg.eigh(cov_ring + cov_refine)
    expected = np.linalg.norm(np.minimum(np.abs(axes.T @ fused), np.abs(axes.T @ (fused - delta))))
    assert fusion_compromise_px(delta, cov_ring, cov_refine) == pytest.approx(expected, rel=1e-6)


def _edge(offset: tuple[float, float], sigma_px: float) -> NavTechniqueResult:
    """A ring-edge answer with an isotropic sigma."""
    return NavTechniqueResult(
        technique_name='RingEdgeNav',
        feature_ids=('ring_edge:SATURN:keeler_gap:IEG',),
        offset_px=offset,
        covariance_px2=np.eye(2) * sigma_px**2,
        confidence=0.65,
        spurious=False,
        at_edge=False,
        diagnostics=RingEdgeDiagnostics(),
    )


def _refine(sigma_px: float, *, seeds: frozenset[str] = frozenset()) -> NavTechniqueResult:
    """A multi-star refine at the origin with an isotropic sigma."""
    return NavTechniqueResult(
        technique_name='StarRefineNav',
        feature_ids=('star:UCAC4:1', 'star:UCAC4:2', 'star:UCAC4:3'),
        offset_px=(0.0, 0.0),
        covariance_px2=np.eye(2) * sigma_px**2,
        confidence=0.85,
        spurious=False,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(n_stars_used=8),
        prior_source_techniques=seeds,
    )


def _pruned(
    joiner: NavTechniqueResult, group: list[NavTechniqueResult]
) -> list[NavTechniqueResult]:
    """The results left out of a combine of ``joiner`` and a set ``group``, representative first."""
    estimators = [joiner, group[0]]
    return leave_out_compromising_fusions(
        estimators,
        estimators,
        [group],
        [joiner, *group],
        [],
        agreement_sigma=2.0,
        agreement_pixel_floor=5.0,
        agreement_compromise_px=1.0,
        rcond=1.0e-9,
    ).left_out


def test_a_joiner_less_precise_than_the_set_is_left_out() -> None:
    """A ring edge 3.6 px from a star set whose two-star member is 0.09 px is left out.

    The set's representative (0.6 px) is no more precise than the edge, but the set
    holds the 0.09 px measurement.
    """
    edge = _edge((0.0, -3.6), 0.6)
    assert _pruned(edge, [_refine(0.6), _two_star_result((0.0, 0.0), 0.09)]) == [edge]


def test_a_joiner_beside_a_precise_representative_is_fused() -> None:
    """The same ring edge beside a 0.1 px representative fuses onto it and stays."""
    assert (
        _pruned(_edge((0.0, -3.6), 0.6), [_refine(0.1), _two_star_result((0.0, 0.0), 0.09)]) == []
    )


def test_a_joiner_that_seeded_the_set_is_never_left_out() -> None:
    """A set the ring edge seeded throughout is held to it when the consensus is selected."""
    group = [_refine(0.6, seeds=frozenset({'RingEdgeNav'})), _two_star_result((0.0, 0.0), 0.09)]
    assert _pruned(_edge((0.0, -3.6), 0.6), group) == []


def test_a_joiner_a_member_seeded_is_never_left_out() -> None:
    """A refine the ring pair's prior seeded is held to that pair at selection instead."""
    refine = _refine(0.6, seeds=frozenset({'RingEdgeNav'}))
    assert _pruned(refine, [_edge((0.0, -3.6), 0.6), _edge((0.0, -3.6), 6.0)]) == []


def test_a_mahalanobis_joiner_past_the_floor_is_held_to_the_compromise() -> None:
    """Two answers 8 px apart and equally vague (6 px) agree statistically but do not fuse."""
    edge = _edge((0.0, 8.0), 6.0)
    assert _pruned(edge, [_refine(6.0), _two_star_result((0.0, 0.0), 6.0)]) == [edge]


def test_a_member_blind_along_the_separation_lends_the_set_no_precision() -> None:
    """A rank-1 member precise only across the separation does not make the set win.

    The member reports no variance along u (it does not observe it), so the set's
    precision along the u separation is its representative's 0.6 px, and the 0.5 px
    joiner keeps its place.
    """
    blind = NavTechniqueResult(
        technique_name='StarUniqueMatchNav',
        feature_ids=('star:UCAC4:9',),
        offset_px=(0.0, 0.0),
        covariance_px2=np.diag([0.01, 0.0]),
        confidence=0.7,
        spurious=False,
        at_edge=False,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    group = [_refine(0.6), blind]
    assert [r.technique_name for r in _pruned(_edge((0.0, -3.6), 0.5), group)] == [
        'StarRefineNav',
        'StarUniqueMatchNav',
    ]


def _ring_pair_and_a_star() -> list[NavTechniqueResult]:
    """A ring edge and annulus (one collapsed witness) and a 0.5 px star fit 3.5 px away."""
    annulus = NavTechniqueResult(
        technique_name='RingAnnulusNav',
        feature_ids=('ring_annulus:SATURN:a_ring',),
        offset_px=(0.0, 2.0),
        covariance_px2=np.eye(2) * 9.0,
        confidence=0.6,
        spurious=False,
        at_edge=False,
        diagnostics=RingAnnulusDiagnostics(),
    )
    star = NavTechniqueResult(
        technique_name='StarRefineNav',
        feature_ids=('star:UCAC4:1', 'star:UCAC4:2', 'star:UCAC4:3'),
        offset_px=(0.0, 3.5),
        covariance_px2=np.eye(2) * 0.25,
        confidence=0.85,
        spurious=False,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(n_stars_used=8),
    )
    return [_edge((0.0, 0.0), 0.6), annulus, star]


def test_ensemble_leaves_out_a_ring_pair_less_precise_than_a_star_fit() -> None:
    """The collapsed ring pair (best member 0.6 px) gives way to the 0.5 px star fit."""
    result = _run(_ring_pair_and_a_star())
    assert result.excluded_from_consensus == ['RingEdgeNav', 'RingAnnulusNav']


def test_ensemble_answer_is_the_star_fit_alone() -> None:
    """With the ring pair left out, the answer is the star fit's offset."""
    result = _run(_ring_pair_and_a_star())
    assert result.offset_px == pytest.approx((0.0, 3.5), abs=1e-9)


def test_ensemble_config_rejects_a_negative_compromise_distance() -> None:
    """A negative compromise distance fails at construction."""
    with pytest.raises(ValueError, match='agreement_compromise_px must be non-negative'):
        EnsembleConfig(agreement_compromise_px=-1.0)


def test_ensemble_config_rejects_a_negative_confirming_axis_sigma() -> None:
    """A negative confirming-axis sigma fails at construction."""
    with pytest.raises(ValueError, match='confirming_axis_max_sigma_px must be non-negative'):
        EnsembleConfig(confirming_axis_max_sigma_px=-1.0)
