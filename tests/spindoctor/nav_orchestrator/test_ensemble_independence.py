"""Tests for the consensus-group independence resolution.

Covers the three correlations the resolution corrects: a seeded single-star
refine (R1), two ring techniques on one catalog (R2), and disc/limb on one
scattered-light gradient (R3).
"""

# R1 / R2 / R3 correspond to issues #222, #317, #339 respectively.

import dataclasses

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
) -> NavTechniqueResult:
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=(f'{technique_name}:f1',),
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
        agreement_sigma=2.0,
        confirming_axis_max_sigma_px=1.0,
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


def _ring_edge_across(radial_sd: float, along_sd: float) -> np.ndarray:
    """Covariance of a ring edge whose radial direction is (0.247, -0.969)."""
    radial = np.array([0.247, -0.969])
    radial = radial / np.linalg.norm(radial)
    along = np.array([radial[1], -radial[0]])
    return radial_sd**2 * np.outer(radial, radial) + along_sd**2 * np.outer(along, along)


def _one_star_pair(
    *,
    refine_offset: tuple[float, float] = (2.21, 20.70),
    refine_star: str = 'star:T1',
    match_mode: str = 'one_star',
    match_prior_sources: frozenset[str] = frozenset(),
) -> tuple[NavTechniqueResult, NavTechniqueResult]:
    """A one-star match and a single-star refine seeded by the ring and the match."""
    match = NavTechniqueResult(
        technique_name='StarUniqueMatchNav',
        feature_ids=('star:T1',),
        offset_px=(2.21, 20.70),
        covariance_px2=np.eye(2, dtype=np.float64) * 20.8**2,
        confidence=0.66,
        spurious=False,
        at_edge=False,
        diagnostics=StarUniqueMatchDiagnostics(mode=match_mode),
        prior_source_techniques=match_prior_sources,
    )
    refine = NavTechniqueResult(
        technique_name='StarRefineNav',
        feature_ids=(refine_star,),
        offset_px=refine_offset,
        covariance_px2=np.eye(2, dtype=np.float64) * 0.32**2,
        confidence=0.47,
        spurious=False,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(n_stars_used=1),
        prior_source_techniques=frozenset({'RingEdgeNav', 'StarUniqueMatchNav'}),
    )
    return match, refine


def test_r1_refine_replaces_one_star_match_the_ring_confirms() -> None:
    """A refine on the one-star match's star, confirmed across a ring edge, replaces it."""
    ring = _result(
        technique_name='RingEdgeNav',
        offset=(-1.16, 19.91),
        cov=_ring_edge_across(0.755, 11.33),
        diagnostics=RingEdgeDiagnostics(),
    )
    match, refine = _one_star_pair()
    res = _resolve([ring, match, refine])
    assert res.replaced_matches == [(refine, match)]
    assert res.dropped_descendants == []
    assert [r.technique_name for r in res.estimators] == ['RingEdgeNav', 'StarRefineNav']
    assert res.estimators[1].prior_source_techniques == frozenset()


def test_r1_drops_refine_off_the_one_star_match_position() -> None:
    """A refine that did not land on the one-star match's star is dropped as before."""
    ring = _result(
        technique_name='RingEdgeNav',
        offset=(-1.16, 19.91),
        cov=_ring_edge_across(0.755, 11.33),
        diagnostics=RingEdgeDiagnostics(),
    )
    match, refine = _one_star_pair(refine_offset=(1.21, 20.45))
    res = _resolve([ring, match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [refine]
    assert res.estimators == [ring, match]


def test_r1_drops_refine_on_a_different_star() -> None:
    """A refine on another star than the one-star match's is dropped as before."""
    ring = _result(
        technique_name='RingEdgeNav',
        offset=(-1.16, 19.91),
        cov=_ring_edge_across(0.755, 11.33),
        diagnostics=RingEdgeDiagnostics(),
    )
    match, refine = _one_star_pair(refine_star='star:T2')
    res = _resolve([ring, match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [refine]


def test_r1_drops_refine_the_ring_contradicts_across_the_edge() -> None:
    """A refine the ring places 3 px off radially is not confirmed, so it is dropped."""
    radial = np.array([0.247, -0.969]) / np.linalg.norm([0.247, -0.969])
    ring_offset = np.array([2.21, 20.70]) - 3.0 * radial
    ring = _result(
        technique_name='RingEdgeNav',
        offset=(float(ring_offset[0]), float(ring_offset[1])),
        cov=_ring_edge_across(0.755, 11.33),
        diagnostics=RingEdgeDiagnostics(),
    )
    match, refine = _one_star_pair()
    res = _resolve([ring, match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [refine]


def _ring_across() -> NavTechniqueResult:
    """A ring edge on the star across itself (0.755 px) and 3.4 px off it along itself."""
    return _result(
        technique_name='RingEdgeNav',
        offset=(-1.16, 19.91),
        cov=_ring_edge_across(0.755, 11.33),
        diagnostics=RingEdgeDiagnostics(),
    )


def test_r1_vague_witness_does_not_confirm_a_refine() -> None:
    """A blob 20 px off with sigma 15 px agrees statistically but confirms nothing."""
    blob = _result(
        technique_name='BodyBlobNav',
        offset=(14.0, 35.0),
        cov=np.eye(2, dtype=np.float64) * 15.0**2,
    )
    match, refine = _one_star_pair()
    res = _resolve([blob, match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [refine]


def test_r1_refine_within_two_sigma_of_its_match_replaces_it() -> None:
    """A refine 1.7 sigma (0.54 px) from its match's centroid still replaces the match."""
    match, refine = _one_star_pair(refine_offset=(2.21 + 0.544, 20.70))
    res = _resolve([_ring_across(), match, refine])
    assert res.replaced_matches == [(refine, match)]


def test_r1_refine_does_not_replace_a_seeded_match() -> None:
    """A one-star match seeded by a groupmate is not the wide search, so it is not replaced."""
    match, refine = _one_star_pair(match_prior_sources=frozenset({'RingEdgeNav'}))
    res = _resolve([_ring_across(), match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [match, refine]


def test_r1_refine_does_not_replace_a_two_star_match() -> None:
    """Only a one-star match is replaced, even when a two-star match lists the same star."""
    match, refine = _one_star_pair(match_mode='two_star')
    res = _resolve([_ring_across(), match, refine])
    assert res.replaced_matches == []
    assert res.dropped_descendants == [refine]


def test_r1_two_refines_cannot_replace_one_match() -> None:
    """When two refines qualify for one match, the first replaces it and the second is dropped."""
    match, refine = _one_star_pair()
    second = dataclasses.replace(refine, technique_name='StarRefineNavB')
    res = _resolve([_ring_across(), match, refine, second])
    assert res.replaced_matches == [(refine, match)]
    assert res.dropped_descendants == [second]


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
