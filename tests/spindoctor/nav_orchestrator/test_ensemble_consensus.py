"""Tests for the consensus selection's fusion-compromise test."""

import numpy as np
import pytest

from spindoctor.nav_orchestrator.ensemble import EnsembleConfig, ensemble
from spindoctor.nav_orchestrator.ensemble_consensus import fusion_compromise_px
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import (
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


def test_ensemble_config_rejects_a_negative_compromise_distance() -> None:
    """A negative compromise distance fails at construction."""
    with pytest.raises(ValueError, match='agreement_compromise_px must be non-negative'):
        EnsembleConfig(agreement_compromise_px=-1.0)
