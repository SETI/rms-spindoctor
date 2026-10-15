"""Tests for the consensus-group independence resolution.

Covers the four correlations the resolution corrects: a seeded single-star
refine (R1), two ring techniques on one catalog (R2), disc/limb on one
scattered-light gradient (R3), and two results that consumed a common feature
(R4).
"""

# R1 / R2 / R3 / R4 correspond to issues #222, #317, #339, #623 respectively.

import numpy as np
import pytest

from spindoctor.nav_orchestrator.ensemble_independence import (
    IndependenceResolution,
    is_single_star_result,
    resolve_independent_estimators,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_technique.diagnostics import (
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    NavTechniqueDiagnostics,
    RingAnnulusDiagnostics,
    RingEdgeDiagnostics,
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult

RCOND = 1.0e-9


def _classifier(*, gradient: float | None = None) -> NavImageClassifierResult:
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
        background_gradient_score=gradient,
    )


def _result(
    *,
    technique_name: str,
    offset: tuple[float, float] = (0.0, 0.0),
    cov: np.ndarray | None = None,
    confidence: float = 0.8,
    diagnostics: NavTechniqueDiagnostics | None = None,
    source_bodies: frozenset[str] = frozenset(),
    prior_sources: frozenset[str] = frozenset(),
    feature_ids: tuple[str, ...] | None = None,
) -> NavTechniqueResult:
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=feature_ids if feature_ids is not None else (f'{technique_name}:f1',),
        offset_px=offset,
        covariance_px2=cov if cov is not None else np.eye(2, dtype=np.float64) * 0.25,
        confidence=confidence,
        spurious=False,
        at_edge=False,
        diagnostics=diagnostics if diagnostics is not None else BodyLimbDiagnostics(),
        source_bodies=source_bodies,
        prior_source_techniques=prior_sources,
    )


def _resolve(
    group: list[NavTechniqueResult], *, gradient: float | None = None
) -> IndependenceResolution:
    return resolve_independent_estimators(
        group,
        image_classifier=_classifier(gradient=gradient),
        scattered_light_gradient_score=5.0,
        rcond=RCOND,
    )


def test_empty_group_raises() -> None:
    """Resolving an empty group is a programming error."""
    with pytest.raises(ValueError, match='empty group'):
        _resolve([])


def test_is_single_star_result_variants() -> None:
    """One-star matches and one-inlier refines are single-star; others are not."""
    one_star = _result(
        technique_name='StarUniqueMatchNav',
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )
    two_star = _result(
        technique_name='StarUniqueMatchNav',
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    one_inlier = _result(
        technique_name='StarRefineNav',
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
    )
    many_inlier = _result(
        technique_name='StarRefineNav',
        diagnostics=StarRefineDiagnostics(n_stars_used=6),
    )
    body = _result(technique_name='BodyLimbNav')
    one_star_lock = _result(
        technique_name='StarFieldFromCatalogNav',
        diagnostics=StarFieldDiagnostics(n_inliers=1),
    )
    three_star_lock = _result(
        technique_name='StarFieldFromCatalogNav',
        diagnostics=StarFieldDiagnostics(n_inliers=3),
    )
    assert is_single_star_result(one_star_lock)
    assert not is_single_star_result(three_star_lock)
    assert is_single_star_result(one_star)
    assert not is_single_star_result(two_star)
    assert is_single_star_result(one_inlier)
    assert not is_single_star_result(many_inlier)
    assert not is_single_star_result(body)


def test_r1_drops_seeded_single_star_refine() -> None:
    """A single-star refine seeded by a groupmate is dropped from the combine."""
    ring = _result(technique_name='RingEdgeNav', offset=(6.0, -118.0))
    refine = _result(
        technique_name='StarRefineNav',
        offset=(7.0, -119.0),
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_sources=frozenset({'RingEdgeNav'}),
    )
    res = _resolve([ring, refine])
    assert res.estimators == [ring]
    assert res.dropped_descendants == [refine]
    assert res.collapsed_groups == []


def test_r1_keeps_seeded_multi_star_refine() -> None:
    """A multi-star refine carries independent information and is kept."""
    ring = _result(technique_name='RingEdgeNav', offset=(6.0, -118.0))
    refine = _result(
        technique_name='StarRefineNav',
        offset=(7.0, -119.0),
        diagnostics=StarRefineDiagnostics(n_stars_used=6),
        prior_sources=frozenset({'RingEdgeNav'}),
    )
    res = _resolve([ring, refine])
    assert res.estimators == [ring, refine]
    assert res.dropped_descendants == []


def test_r1_keeps_single_star_refine_without_stronger_witness() -> None:
    """A pure single-star lock keeps its refine: no stronger witness to override.

    When the only other consensus member is itself single-star (a one-star
    unique match), the seeded refine is the sole refinement of a legitimately
    weak fix, not a redundant vote against a body or multi-star consensus, so
    it is kept.
    """
    match = _result(
        technique_name='StarUniqueMatchNav',
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )
    refine = _result(
        technique_name='StarRefineNav',
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_sources=frozenset({'StarUniqueMatchNav'}),
    )
    res = _resolve([match, refine])
    assert set(res.estimators) == {match, refine}
    assert res.dropped_descendants == []


def test_r1_keeps_unseeded_single_star_refine() -> None:
    """A single-star refine whose seed is absent from the group is not a descendant."""
    refine = _result(
        technique_name='StarRefineNav',
        offset=(7.0, -119.0),
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_sources=frozenset({'BodyLimbNav'}),
    )
    other = _result(technique_name='RingEdgeNav', offset=(7.0, -119.0))
    res = _resolve([refine, other])
    assert set(res.estimators) == {refine, other}
    assert res.dropped_descendants == []


def test_r1_never_empties_the_group() -> None:
    """If every member is a seeded single-star refine, none are dropped."""
    a = _result(
        technique_name='StarRefineNav',
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_sources=frozenset({'StarRefineNav'}),
    )
    res = _resolve([a])
    assert res.estimators == [a]
    assert res.dropped_descendants == []


def test_r2_collapses_two_ring_techniques() -> None:
    """RingEdge and RingAnnulus collapse to one representative witness."""
    tight = np.eye(2, dtype=np.float64) * 0.04
    loose = np.eye(2, dtype=np.float64) * 0.25
    edge = _result(
        technique_name='RingEdgeNav',
        cov=tight,
        diagnostics=RingEdgeDiagnostics(),
    )
    annulus = _result(
        technique_name='RingAnnulusNav',
        cov=loose,
        diagnostics=RingAnnulusDiagnostics(),
    )
    res = _resolve([edge, annulus])
    # One estimator; the higher-precision (tighter) edge result represents it.
    assert res.estimators == [edge]
    assert res.collapsed_groups == [[edge, annulus]]


def test_r3_collapses_disc_and_limb_only_when_scattered() -> None:
    """Disc and limb on one body collapse only above the scatter threshold."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        source_bodies=frozenset({'MIMAS'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    limb = _result(
        technique_name='BodyLimbNav',
        source_bodies=frozenset({'MIMAS'}),
        diagnostics=BodyLimbDiagnostics(),
    )
    # Below threshold (clean frame): both survive as independent estimators.
    clean = _resolve([disc, limb], gradient=1.0)
    assert set(clean.estimators) == {disc, limb}
    assert clean.collapsed_groups == []
    # Above threshold (scattered-light frame): collapsed to one witness.  Equal
    # precision, so the max() technique-name tie-break makes limb (the larger
    # name) the representative.
    scattered = _resolve([disc, limb], gradient=8.0)
    assert scattered.estimators == [limb]
    assert scattered.collapsed_groups == [[limb, disc]]


def test_r3_does_not_collapse_different_bodies() -> None:
    """Disc and limb on different bodies are independent even under scatter."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        source_bodies=frozenset({'MIMAS'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    limb = _result(
        technique_name='BodyLimbNav',
        source_bodies=frozenset({'TETHYS'}),
        diagnostics=BodyLimbDiagnostics(),
    )
    res = _resolve([disc, limb], gradient=8.0)
    assert set(res.estimators) == {disc, limb}
    assert res.collapsed_groups == []


def test_no_gradient_score_never_collapses_disc_limb() -> None:
    """A missing background-gradient score defaults to not-scattered."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        source_bodies=frozenset({'MIMAS'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    limb = _result(
        technique_name='BodyLimbNav',
        source_bodies=frozenset({'MIMAS'}),
        diagnostics=BodyLimbDiagnostics(),
    )
    res = _resolve([disc, limb], gradient=None)
    assert set(res.estimators) == {disc, limb}


def test_uncorrelated_group_passes_through_unchanged() -> None:
    """A body and a star witness share no error source and both survive."""
    body = _result(technique_name='BodyLimbNav', source_bodies=frozenset({'RHEA'}))
    star = _result(
        technique_name='StarFieldFromCatalogNav',
        diagnostics=StarRefineDiagnostics(n_stars_used=8),
    )
    res = _resolve([body, star], gradient=8.0)
    assert res.estimators == [body, star]
    assert res.dropped_descendants == []
    assert res.collapsed_groups == []


def test_estimators_preserve_input_order() -> None:
    """Surviving estimators keep the input group's order."""
    star = _result(
        technique_name='StarFieldFromCatalogNav',
        offset=(1.0, 1.0),
        diagnostics=StarRefineDiagnostics(n_stars_used=8),
    )
    edge = _result(technique_name='RingEdgeNav', diagnostics=RingEdgeDiagnostics())
    annulus = _result(technique_name='RingAnnulusNav', diagnostics=RingAnnulusDiagnostics())
    # Ring pair collapses to one; the star stays. Order: star (idx 0) then ring.
    res = _resolve([star, edge, annulus])
    assert res.estimators[0] == star
    assert len(res.estimators) == 2


def test_r4_collapses_two_results_on_one_star() -> None:
    """A one-star match and the one-inlier refine of that star are one witness."""
    match = _result(
        technique_name='StarUniqueMatchNav',
        cov=np.eye(2, dtype=np.float64) * 900.0,
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
        feature_ids=('star:UCAC4:1',),
    )
    refine = _result(
        technique_name='StarRefineNav',
        cov=np.eye(2, dtype=np.float64) * 0.08,
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_sources=frozenset({'StarUniqueMatchNav'}),
        feature_ids=('star:UCAC4:1',),
    )
    res = _resolve([match, refine])
    # The refine is the tighter centroid of the same detection, so it represents both.
    assert res.estimators == [refine]
    assert res.collapsed_groups == [[refine, match]]


def test_r4_collapses_partly_overlapping_feature_sets() -> None:
    """A two-star match and a three-star refine sharing both stars are one witness."""
    match = _result(
        technique_name='StarUniqueMatchNav',
        cov=np.eye(2, dtype=np.float64) * 0.5,
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
        feature_ids=('star:a', 'star:b'),
    )
    refine = _result(
        technique_name='StarRefineNav',
        cov=np.eye(2, dtype=np.float64) * 0.1,
        diagnostics=StarRefineDiagnostics(n_stars_used=3),
        feature_ids=('star:a', 'star:b', 'star:c'),
    )
    res = _resolve([match, refine])
    assert res.estimators == [refine]


def test_r4_keeps_results_on_disjoint_features() -> None:
    """Two star results that measured different stars stay independent."""
    first = _result(
        technique_name='StarUniqueMatchNav',
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
        feature_ids=('star:a', 'star:b'),
    )
    second = _result(
        technique_name='StarFieldFromCatalogNav',
        diagnostics=StarFieldDiagnostics(n_inliers=3),
        feature_ids=('star:c', 'star:d', 'star:e'),
    )
    res = _resolve([first, second])
    assert res.estimators == [first, second]
    assert res.collapsed_groups == []
