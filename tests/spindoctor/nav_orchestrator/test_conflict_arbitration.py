"""Tests for ``spindoctor.nav_orchestrator.conflict_arbitration``."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from spindoctor.nav_orchestrator.conflict_arbitration import (
    DEFAULT_MIN_SUPPORT,
    DEFAULT_MIN_SUPPORT_RATIO,
    ConflictArbitrationConfig,
    arbitrate_conflict,
    confirmed_candidate,
    pass2_support,
)
from spindoctor.nav_orchestrator.ensemble import ensemble
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import (
    ArbitrationCandidate,
    ConflictArbitrationRecord,
    NavResult,
)
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics, StarRefineDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

_CONFIG = ConflictArbitrationConfig(min_support=5, min_support_ratio=3.0)
_STAR_OFFSET = (1.0, 2.0)
_ALIAS_OFFSET = (-18.0, -46.0)


def _result(
    name: str,
    offset: tuple[float, float],
    *,
    feature_ids: tuple[str, ...] = (),
    confidence: float = 0.8,
    spurious: bool = False,
    at_edge: bool = False,
) -> NavTechniqueResult:
    """Build a per-technique result with a tight covariance."""
    return NavTechniqueResult(
        technique_name=name,
        feature_ids=feature_ids or (f'{name}:f1',),
        offset_px=offset,
        covariance_px2=np.eye(2, dtype=np.float64) * 0.01,
        confidence=confidence,
        spurious=spurious,
        at_edge=at_edge,
        diagnostics=BodyLimbDiagnostics(),
    )


def _refine(offset: tuple[float, float], n_stars: int) -> NavTechniqueResult:
    """Build a pass-2 refine result that consumed ``n_stars`` stars."""
    return NavTechniqueResult(
        technique_name='RefineStandIn',
        feature_ids=tuple(f'star:{i}' for i in range(n_stars)),
        offset_px=offset,
        covariance_px2=np.eye(2, dtype=np.float64) * 0.04,
        confidence=0.9,
        spurious=n_stars == 0,
        at_edge=False,
        diagnostics=StarRefineDiagnostics(
            n_stars_used=n_stars, median_pos_err_px=0.2, residual_scatter_px=0.2
        ),
    )


def _combine(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> Callable[[list[NavTechniqueResult]], NavResult]:
    """Return an ensemble closure over the shared fixtures."""

    def combine(group: list[NavTechniqueResult]) -> NavResult:
        return ensemble(
            group, feature_inventory=[], image_classifier=classifier, provenance=provenance
        )

    return combine


def _pass2_by_offset(
    support: dict[tuple[float, float], int],
) -> Callable[[tuple[float, float], NDArrayFloatType], list[NavTechniqueResult]]:
    """Return a pass-2 runner whose refine finds ``support[offset]`` stars."""

    def run(offset: tuple[float, float], cov: NDArrayFloatType) -> list[NavTechniqueResult]:
        del cov
        key = min(support, key=lambda k: float(np.hypot(k[0] - offset[0], k[1] - offset[1])))
        return [_refine(offset, support[key])]

    return run


@pytest.fixture
def conflict(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> tuple[list[NavTechniqueResult], NavResult]:
    """A lone-vs-lone pass 1: a star match against a ring alias 50 px away."""
    star = _result('StarMatchStandIn', _STAR_OFFSET, confidence=0.80)
    alias = _result('RingStandIn', _ALIAS_OFFSET, confidence=0.755)
    results = [star, alias]
    pass1 = ensemble(
        results, feature_inventory=[], image_classifier=classifier, provenance=provenance
    )
    return results, pass1


def test_the_fixture_pass_1_is_conflicted(
    conflict: tuple[list[NavTechniqueResult], NavResult],
) -> None:
    """The fixture reproduces the lone-vs-lone standoff the arbitration resolves."""
    assert conflict[1].status == 'conflicted'


def test_support_counts_distinct_features_of_usable_results() -> None:
    """Spurious and at-edge results add nothing; a shared feature counts once."""
    results = [
        _result('A', (0.0, 0.0), feature_ids=('s1', 's2')),
        _result('B', (0.0, 0.0), feature_ids=('s2', 's3')),
        _result('C', (0.0, 0.0), feature_ids=('s4',), spurious=True),
        _result('D', (0.0, 0.0), feature_ids=('s5',), at_edge=True),
    ]
    assert pass2_support(results) == 3


@pytest.mark.parametrize(
    ('supports', 'expected'),
    [
        ([30, 1], 0),
        ([1, 30], 1),
        ([5, 0], 0),
        ([4, 0], None),
        ([15, 6], None),
        ([15, 5], 0),
        ([0, 0], None),
    ],
    ids=['first', 'second', 'at-minimum', 'below-minimum', 'ratio-short', 'at-ratio', 'none'],
)
def test_confirmed_candidate(supports: list[int], expected: int | None) -> None:
    """The confirmed candidate reaches the minimum and the ratio over the other."""
    assert confirmed_candidate(supports, config=_CONFIG) == expected


def test_a_zero_minimum_confirms_nothing() -> None:
    """``min_support`` 0 disables the arbitration."""
    config = ConflictArbitrationConfig(min_support=0, min_support_ratio=3.0)
    assert confirmed_candidate([30, 0], config=config) is None


def test_the_test_confirms_the_consensus_answer(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """Stars found at the consensus answer only: the runner-up is refuted."""
    results, pass1 = conflict
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 30, _ALIAS_OFFSET: 1}),
        config=_CONFIG,
    )
    assert outcome is not None
    assert outcome.refuted == [results[1]]
    assert outcome.refuted[0] is results[1]
    assert [r.offset_px for r in outcome.pass2_results] == [_STAR_OFFSET]
    assert outcome.pass2_results[0].prior_source_techniques == frozenset({'StarMatchStandIn'})


def test_the_test_confirms_the_runner_up_answer(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """Stars found at the runner-up answer only: the consensus group is refuted."""
    results, pass1 = conflict
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 0, _ALIAS_OFFSET: 12}),
        config=_CONFIG,
    )
    assert outcome is not None
    assert outcome.refuted == [results[0]]
    assert outcome.pass2_results[0].prior_source_techniques == frozenset({'RingStandIn'})


def test_stars_the_answer_already_used_count_toward_its_support(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> None:
    """A right answer whose fit already used every star wins over chance detections."""
    stars = tuple(f'star:{i}' for i in range(6))
    results = [
        _result('StarMatchStandIn', _STAR_OFFSET, confidence=0.80, feature_ids=stars),
        _result('RingStandIn', _ALIAS_OFFSET, confidence=0.755),
    ]
    pass1 = ensemble(
        results, feature_inventory=[], image_classifier=classifier, provenance=provenance
    )
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 6, _ALIAS_OFFSET: 3}),
        config=ConflictArbitrationConfig(min_support=4, min_support_ratio=2.0),
    )
    assert outcome is not None
    assert outcome.refuted == [results[1]]


def test_comparable_support_declines(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """Support at both answers within the ratio confirms neither."""
    results, pass1 = conflict
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 12, _ALIAS_OFFSET: 6}),
        config=_CONFIG,
    )
    assert outcome is not None
    assert outcome.confirmed is False
    assert outcome.refuted == []
    assert outcome.pass2_results == []
    assert outcome.record.refuted_techniques == ()


def test_the_record_names_both_answers_and_the_refuted(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """The record carries each answer's techniques, offset, support, and verdict."""
    results, pass1 = conflict
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 30, _ALIAS_OFFSET: 1}),
        config=_CONFIG,
    )
    assert outcome is not None
    assert outcome.record == ConflictArbitrationRecord(
        candidates=(
            ArbitrationCandidate(
                techniques=('StarMatchStandIn',),
                offset_px=_STAR_OFFSET,
                support=30,
                confirmed=True,
            ),
            ArbitrationCandidate(
                techniques=('RingStandIn',),
                offset_px=_ALIAS_OFFSET,
                support=1,
                confirmed=False,
            ),
        ),
        refuted_techniques=('RingStandIn',),
    )


def _recording_runner(
    calls: list[tuple[float, float]],
) -> Callable[[tuple[float, float], NDArrayFloatType], list[NavTechniqueResult]]:
    """Return a pass-2 runner that records each seed and finds nothing."""

    def run(offset: tuple[float, float], cov: NDArrayFloatType) -> list[NavTechniqueResult]:
        del cov
        calls.append(offset)
        return []

    return run


def test_a_body_witness_conflict_is_not_tested(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """A conflict the body-witness veto reported is not a consensus conflict to test."""
    results, pass1 = conflict
    vetoed = dataclasses.replace(pass1, status_reason=NavStatusReason.BODY_SHAPE_LOCK_SUSPECT)
    calls: list[tuple[float, float]] = []
    outcome = arbitrate_conflict(
        results,
        vetoed,
        combine=_combine(classifier, provenance),
        run_pass2=_recording_runner(calls),
        config=_CONFIG,
    )
    assert outcome is None
    assert calls == []


def test_a_runner_up_without_an_answer_is_not_tested(
    conflict: tuple[list[NavTechniqueResult], NavResult],
    classifier: NavImageClassifierResult,
    provenance: Provenance,
) -> None:
    """When the runner-up group's own ensemble is itself conflicted, nothing is tested."""
    results, pass1 = conflict

    def combine(group: list[NavTechniqueResult]) -> NavResult:
        del group
        return NavResult.conflicted(
            offset_px=_ALIAS_OFFSET,
            covariance_px2=np.eye(2, dtype=np.float64) * 0.01,
            confidence=0.2,
            per_technique=[],
            feature_inventory=[],
            image_classifier=classifier,
            provenance=provenance,
        )

    calls: list[tuple[float, float]] = []
    outcome = arbitrate_conflict(
        results, pass1, combine=combine, run_pass2=_recording_runner(calls), config=_CONFIG
    )
    assert outcome is None
    assert calls == []


def test_a_three_way_conflict_refutes_only_the_tested_runner_up(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> None:
    """An excluded result outside the runner-up answer was never tested: not refuted."""
    star = _result('StarMatchStandIn', _STAR_OFFSET, confidence=0.80)
    alias = _result('RingStandIn', _ALIAS_OFFSET, confidence=0.755)
    other = _result('OtherStandIn', (60.0, -30.0), confidence=0.3)
    results = [star, alias, other]
    pass1 = ensemble(
        results, feature_inventory=[], image_classifier=classifier, provenance=provenance
    )
    assert pass1.status == 'conflicted'
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 30, _ALIAS_OFFSET: 1}),
        config=_CONFIG,
    )
    assert outcome is not None
    assert outcome.refuted == [alias]


def test_a_successful_pass_1_is_not_tested(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> None:
    """Only a conflicted pass 1 runs pass 2 at more than one answer."""
    results = [_result('A', (1.0, 2.0)), _result('B', (1.1, 2.0))]
    pass1 = ensemble(
        results, feature_inventory=[], image_classifier=classifier, provenance=provenance
    )
    calls: list[tuple[float, float]] = []

    def run(offset: tuple[float, float], cov: NDArrayFloatType) -> list[NavTechniqueResult]:
        del cov
        calls.append(offset)
        return []

    arbitrate_conflict(
        results, pass1, combine=_combine(classifier, provenance), run_pass2=run, config=_CONFIG
    )
    assert calls == []


def test_a_name_in_both_groups_declines(
    classifier: NavImageClassifierResult, provenance: Provenance
) -> None:
    """Two results of one technique on opposite sides cannot be assigned by name."""
    results = [
        _result('A', _STAR_OFFSET, confidence=0.5),
        _result('Same', _STAR_OFFSET, confidence=0.5),
        _result('B', _ALIAS_OFFSET, confidence=0.4),
        _result('Same', _ALIAS_OFFSET, confidence=0.4, feature_ids=('Same:f2',)),
    ]
    pass1 = ensemble(
        results, feature_inventory=[], image_classifier=classifier, provenance=provenance
    )
    assert pass1.status == 'conflicted'
    outcome = arbitrate_conflict(
        results,
        pass1,
        combine=_combine(classifier, provenance),
        run_pass2=_pass2_by_offset({_STAR_OFFSET: 30, _ALIAS_OFFSET: 0}),
        config=_CONFIG,
    )
    assert outcome is None


def test_config_defaults() -> None:
    """An absent section yields the code defaults."""
    config = ConflictArbitrationConfig.from_mapping(None)
    assert (config.min_support, config.min_support_ratio) == (
        DEFAULT_MIN_SUPPORT,
        DEFAULT_MIN_SUPPORT_RATIO,
    )


def test_config_reads_the_section() -> None:
    """Section values override the defaults."""
    config = ConflictArbitrationConfig.from_mapping({'min_support': 7, 'min_support_ratio': 4})
    assert config == ConflictArbitrationConfig(min_support=7, min_support_ratio=4.0)


@pytest.mark.parametrize(
    ('mapping', 'message'),
    [
        ({'min_supports': 5}, 'Unknown orchestrator.conflict_arbitration config keys'),
        ({'min_support': 2.5}, 'min_support must be a whole number'),
        ({'min_support': True}, 'min_support must be a whole number'),
        ({'min_support': -1}, 'min_support must be >= 0'),
        ({'min_support_ratio': 'x'}, 'min_support_ratio must be a number'),
        ({'min_support_ratio': 0.5}, 'min_support_ratio must be a finite number greater'),
        ({'min_support_ratio': 1.0}, 'min_support_ratio must be a finite number greater'),
        ({'min_support_ratio': math.inf}, 'min_support_ratio must be a finite number greater'),
        ({'min_support_ratio': math.nan}, 'min_support_ratio must be a finite number greater'),
    ],
    ids=[
        'unknown-key',
        'fractional',
        'bool',
        'negative',
        'not-a-number',
        'ratio-below-one',
        'ratio-one',
        'ratio-infinite',
        'ratio-nan',
    ],
)
def test_config_rejects(mapping: dict[str, Any], message: str) -> None:
    """A malformed section is refused at load with the offending key named."""
    with pytest.raises(ValueError, match=message):
        ConflictArbitrationConfig.from_mapping(mapping)


def test_config_constructor_rejects_a_fractional_support() -> None:
    """The constructor holds the whole-number rule as the section reader does."""
    with pytest.raises(ValueError, match='min_support must be a whole number'):
        ConflictArbitrationConfig(min_support=2.5)  # type: ignore[arg-type]
