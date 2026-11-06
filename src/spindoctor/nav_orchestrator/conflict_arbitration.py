"""Arbitrate a conflicted pass 1 by testing each candidate answer in pass 2.

A pass-1 ensemble reports ``conflicted`` when two answers of comparable vote stand
apart: for example a two-star unique match and a ring edge locked on an alias tens of
pixels away.  Seeding pass 2 with either answer alone would lock the prior-required
techniques onto one arbitrary mode.  Yet the prior-required techniques are a test of a
candidate answer, not only a polish of it: a star refine seeded at the right offset
finds the catalog stars where the offset puts them, while one seeded at a wrong offset
finds noise or nothing.

This module runs that test.  The two candidates are the pass-1 consensus group and the
runner-up group: the excluded results ensembled on their own, of which the members of
that ensemble's consensus are the runner-up answer.  Pass 2 is run once seeded at each;
a candidate's *support* is the number of distinct features the non-spurious, interior
pass-2 results consumed at it.  The features the candidate's own pass-1 results
consumed are counted with the rest: leaving them out would leave a right answer whose
pass-1 star fit already holds every visible star with no support at all, against the
chance detections a refine makes at a wrong answer.  ``min_support`` is set so that a
two-star match needs stars beyond its own two.  A candidate is confirmed when its
support reaches ``min_support`` and is at least ``min_support_ratio`` times the other
candidate's.  The confirmed candidate's pass-2 results join the final ensemble as the
pass-2 results of an ordinary seeded run, and the other candidate's pass-1 results are
reported to the final ensemble as refuted: they are named among the excluded techniques
but cast no vote, since the test run at both answers held up at only one of them.  An
excluded result outside both candidates is left to the final ensemble as an ordinary
outlier.  When neither candidate is confirmed (no pass-2 technique is feasible, both
find comparable support, or neither finds enough) the arbitration declines and pass 2
runs unseeded.

The test runs only on a ``conflicted_techniques`` pass 1 whose consensus and runner-up
groups are told apart by technique name, and only when the runner-up group has a
successful answer of its own.  A pass 1 in which one technique name sits in both groups
is left to the unseeded path, since the names the ensemble reports cannot say which of
that technique's results belongs where.  Whenever the test runs, what it found is
returned as a :class:`ConflictArbitrationRecord` for the metadata document.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from spindoctor.config import IMAGE_LOGGER
from spindoctor.nav_orchestrator.nav_result import (
    ArbitrationCandidate,
    ConflictArbitrationRecord,
    NavResult,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DEFAULT_MIN_SUPPORT',
    'DEFAULT_MIN_SUPPORT_RATIO',
    'ConflictArbitration',
    'ConflictArbitrationConfig',
    'arbitrate_conflict',
    'confirmed_candidate',
    'pass2_support',
]

#: Fewest distinct features the pass-2 results must consume at a candidate for
#: the test to confirm it.
DEFAULT_MIN_SUPPORT = 4

#: Factor by which the confirmed candidate's support must exceed the other's.
DEFAULT_MIN_SUPPORT_RATIO = 2.0


@dataclass(frozen=True)
class ConflictArbitrationConfig:
    """Thresholds of the pass-2 test of a conflicted pass 1.

    Parameters:
        min_support: Fewest distinct features the non-spurious, interior pass-2 results
            must consume at a candidate for it to be confirmed.  ``0`` disables the
            arbitration.
        min_support_ratio: Factor by which the confirmed candidate's support must
            exceed the other candidate's.
    """

    min_support: int = DEFAULT_MIN_SUPPORT
    min_support_ratio: float = DEFAULT_MIN_SUPPORT_RATIO

    def __post_init__(self) -> None:
        """Reject a support that is not a whole number >= 0 and a ratio not above one.

        A ratio of one would confirm one of two equal supports arbitrarily, and an
        infinite or NaN ratio would confirm nothing, silently disabling the test.

        Raises:
            ValueError: If ``min_support`` is not a whole number or is negative, or
                ``min_support_ratio`` is not a finite number greater than ``1.0``.
        """
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, int):
            raise ValueError(
                f'orchestrator.conflict_arbitration.min_support must be a whole number; '
                f'got {self.min_support!r}'
            )
        if self.min_support < 0:
            raise ValueError(
                f'orchestrator.conflict_arbitration.min_support must be >= 0; '
                f'got {self.min_support}'
            )
        if not 1.0 < self.min_support_ratio < math.inf:
            raise ValueError(
                f'orchestrator.conflict_arbitration.min_support_ratio must be a finite '
                f'number greater than 1; got {self.min_support_ratio}'
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> ConflictArbitrationConfig:
        """Build the thresholds from the ``orchestrator.conflict_arbitration`` section.

        Parameters:
            mapping: The config section; ``None`` or empty returns the code defaults.

        Returns:
            The thresholds, mapping values overriding the defaults key by key.

        Raises:
            ValueError: If the mapping holds an unknown key, a ``min_support_ratio`` that
                is not a number, or a value the constructor rejects.
        """
        if not mapping:
            return cls()
        unknown = set(mapping) - {'min_support', 'min_support_ratio'}
        if unknown:
            raise ValueError(
                f'Unknown orchestrator.conflict_arbitration config keys: {sorted(unknown)}'
            )
        kwargs: dict[str, Any] = {}
        if 'min_support' in mapping:
            kwargs['min_support'] = mapping['min_support']
        if 'min_support_ratio' in mapping:
            value = mapping['min_support_ratio']
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f'orchestrator.conflict_arbitration.min_support_ratio must be a '
                    f'number; got {value!r}'
                )
            kwargs['min_support_ratio'] = float(value)
        return cls(**kwargs)


@dataclass(frozen=True)
class ConflictArbitration:
    """The outcome of the pass-2 test of a conflicted pass 1.

    Parameters:
        pass2_results: The pass-2 results of the run seeded at the confirmed candidate,
            stamped with the techniques whose consensus formed that candidate; empty
            when the test confirmed neither candidate.
        refuted: The pass-1 results of the other candidate's consensus, the same
            objects as in the pass-1 result list; empty when the test confirmed
            neither candidate.
        record: What the test found, for the metadata document.
    """

    pass2_results: list[NavTechniqueResult]
    refuted: list[NavTechniqueResult]
    record: ConflictArbitrationRecord

    @property
    def confirmed(self) -> bool:
        """True when the test confirmed one of the two candidates."""
        return any(c.confirmed for c in self.record.candidates)


def pass2_support(results: Sequence[NavTechniqueResult]) -> int:
    """Return how many distinct features the usable pass-2 results consumed.

    Parameters:
        results: Pass-2 results of one seeded run.

    Returns:
        The number of distinct ``feature_ids`` over the results that are neither
        spurious nor at the edge of their search.
    """
    return len({fid for r in results if not (r.spurious or r.at_edge) for fid in r.feature_ids})


def confirmed_candidate(
    supports: Sequence[int], *, config: ConflictArbitrationConfig
) -> int | None:
    """Return the index of the one candidate the pass-2 test confirms.

    Parameters:
        supports: Support of each candidate, in candidate order (at least two).
        config: The thresholds.

    Returns:
        The index of the candidate whose support is at least ``config.min_support`` and
        at least ``config.min_support_ratio`` times every other candidate's support;
        ``None`` when no candidate qualifies or ``config.min_support`` is ``0``.
    """
    if config.min_support == 0:
        return None
    for i, support in enumerate(supports):
        others = [s for j, s in enumerate(supports) if j != i]
        if support >= config.min_support and all(
            support >= config.min_support_ratio * s for s in others
        ):
            return i
    return None


def _split_groups(
    pass1_results: Sequence[NavTechniqueResult], pass1: NavResult
) -> tuple[list[NavTechniqueResult], list[NavTechniqueResult]] | None:
    """Return the consensus and runner-up groups of a conflicted pass 1.

    Returns:
        ``(consensus, runner_up)`` non-spurious pass-1 results, or ``None`` when either
        is empty or a technique name is reported in both groups.
    """
    consensus_names = set(pass1.consensus_techniques)
    excluded_names = set(pass1.excluded_from_consensus)
    if consensus_names & excluded_names:
        return None
    consensus = [r for r in pass1_results if not r.spurious and r.technique_name in consensus_names]
    runner_up = [r for r in pass1_results if not r.spurious and r.technique_name in excluded_names]
    if not consensus or not runner_up:
        return None
    return consensus, runner_up


def arbitrate_conflict(
    pass1_results: Sequence[NavTechniqueResult],
    pass1: NavResult,
    *,
    combine: Callable[[list[NavTechniqueResult]], NavResult],
    run_pass2: Callable[[tuple[float, float], NDArrayFloatType], list[NavTechniqueResult]],
    config: ConflictArbitrationConfig,
) -> ConflictArbitration | None:
    """Test each answer of a conflicted pass 1 with the pass-2 techniques.

    Parameters:
        pass1_results: Every pass-1 result.
        pass1: The pass-1 ensemble result.
        combine: Ensemble over a list of results; used to form the runner-up group's
            answer.
        run_pass2: Runs the prior-required techniques seeded at an offset and
            covariance and returns their results.
        config: The thresholds.

    Returns:
        The test's outcome: when it confirms a candidate, that candidate's pass-2
        results and the other candidate's refuted pass-1 results, else neither; in
        both cases the record of what it found.  ``None`` when the test does not run:
        the arbitration is disabled, the pass 1 is not a ``conflicted_techniques``
        result, its groups cannot be told apart, or the runner-up group has no
        successful answer of its own.
    """
    if config.min_support == 0:
        return None
    if pass1.status != 'conflicted':
        return None
    if pass1.status_reason != NavStatusReason.CONFLICTED_TECHNIQUES:
        return None
    if pass1.offset_px is None or pass1.covariance_px2 is None:
        return None
    groups = _split_groups(pass1_results, pass1)
    if groups is None:
        return None
    consensus, runner_up = groups
    runner = combine(runner_up)
    if runner.status != 'success' or runner.offset_px is None or runner.covariance_px2 is None:
        IMAGE_LOGGER.info(
            'Conflict arbitration declined: the runner-up group (%s) has no answer of its '
            'own (status %s)',
            ', '.join(r.technique_name for r in runner_up),
            runner.status,
        )
        return None
    # The runner-up answer is the runner-up ensemble's own consensus; an excluded
    # result outside it was never tested and stays an ordinary outlier.
    runner_names = set(runner.consensus_techniques)
    candidates = [
        (consensus, pass1.offset_px, pass1.covariance_px2, tuple(pass1.consensus_techniques)),
        (
            [r for r in runner_up if r.technique_name in runner_names],
            runner.offset_px,
            runner.covariance_px2,
            tuple(runner.consensus_techniques),
        ),
    ]
    runs: list[list[NavTechniqueResult]] = []
    supports: list[int] = []
    for members, offset, cov, _names in candidates:
        IMAGE_LOGGER.info(
            'Conflict arbitration: testing the answer of %s at (%.4f, %.4f) px in pass 2',
            ', '.join(r.technique_name for r in members),
            offset[0],
            offset[1],
        )
        results = run_pass2(offset, cov)
        runs.append(results)
        supports.append(pass2_support(results))
    winner = confirmed_candidate(supports, config=config)
    IMAGE_LOGGER.info(
        'Conflict arbitration: pass-2 support %s (minimum %d, ratio %.1f); %s',
        ' vs '.join(str(s) for s in supports),
        config.min_support,
        config.min_support_ratio,
        'no answer confirmed' if winner is None else f'answer {winner + 1} confirmed',
    )
    loser = [] if winner is None else candidates[1 - winner][0]
    record = ConflictArbitrationRecord(
        candidates=tuple(
            ArbitrationCandidate(
                techniques=names,
                offset_px=(float(offset[0]), float(offset[1])),
                support=support,
                confirmed=i == winner,
            )
            for i, ((_members, offset, _cov, names), support) in enumerate(
                zip(candidates, supports, strict=True)
            )
        ),
        refuted_techniques=tuple(r.technique_name for r in loser),
    )
    if winner is None:
        return ConflictArbitration(pass2_results=[], refuted=[], record=record)
    seeds = frozenset(candidates[winner][3])
    IMAGE_LOGGER.info(
        'Conflict arbitration: %s refuted by the pass-2 test',
        ', '.join(r.technique_name for r in loser),
    )
    return ConflictArbitration(
        pass2_results=[_stamp(r, seeds) for r in runs[winner]],
        refuted=list(loser),
        record=record,
    )


def _stamp(result: NavTechniqueResult, seeds: frozenset[str]) -> NavTechniqueResult:
    """Return ``result`` with ``prior_source_techniques`` set to ``seeds``."""
    return replace(result, prior_source_techniques=seeds)
