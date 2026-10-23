"""Consensus-subset selection for the ensemble combine.

Given the viable per-technique results, decide which subset the ensemble
fuses and which results are outliers.  Every result sponsors the subset
that agrees with it pairwise (Mahalanobis distance or pixel floor, both
measured only in the observable-subspace intersection, with a Mahalanobis
agreement past the pixel floor kept only when fusing the pair would not
compromise between them); the highest summed-confidence subset wins.  This
machinery lives apart from ``ensemble`` so that module stays under the
1000-line limit; ``ensemble`` imports :func:`consensus_selection` and
:func:`result_param_vector`.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.nav_orchestrator.ensemble_independence import STAR_CONSENSUS_TECHNIQUES
from spindoctor.nav_orchestrator.ensemble_observability import (
    mahalanobis_distance,
    observable_basis,
    observable_intersection_basis,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

#: Squared length of a unit direction's projection onto a result's observable
#: subspace at or above which the result observes that direction.
_FULLY_OBSERVED = 0.999

#: Factor by which one star result's distinct matched stars must outnumber another's
#: for it to leave the other out when the two contradict each other.  Below it the two
#: are comparable evidence, and the combine weighs them by their covariances.
_OUTNUMBERING_FACTOR = 2

__all__ = [
    'ConsensusSelection',
    'FusionPruning',
    'consensus_selection',
    'corroborating_confidence',
    'corroborating_members',
    'fusion_compromise_px',
    'leave_out_compromising_fusions',
    'result_param_vector',
]


def corroborating_members(group: list[NavTechniqueResult]) -> list[NavTechniqueResult]:
    """Members of a group whose agreement is an independent opinion.

    A pass-2 result searches a small window around the pass-1 prior, so
    its offset is conditionally dependent on whichever techniques seeded
    that prior (``prior_source_techniques``).  When any of those
    techniques sits in the same group, the pass-2 result's agreement
    with the group corroborates nothing — it re-observed its own seed —
    so it is dropped from every quorum, summed-confidence, and
    agreement-boost computation.  It still contributes its
    precision to the combined offset.

    Parameters:
        group: Results forming one agreement subset.

    Returns:
        The members whose vote counts, in input order.
    """
    names = {r.technique_name for r in group}
    return [r for r in group if not (r.prior_source_techniques & (names - {r.technique_name}))]


def corroborating_confidence(group: list[NavTechniqueResult]) -> float:
    """Summed confidence over a group's corroborating members only."""
    return sum(r.confidence for r in corroborating_members(group))


def result_param_vector(res: NavTechniqueResult) -> NDArrayFloatType:
    """Return the parameter vector for a per-technique result.

    Two-DoF results emit ``(dv, du)``; 3-DoF results emit
    ``(dv, du, rotation_rad)``.  The vector length always matches the
    covariance shape — a 3x3 covariance with ``rotation_rad=None`` would be
    inconsistent and raises here.
    """
    cov = np.asarray(res.covariance_px2, np.float64)
    if cov.shape == (3, 3):
        if res.rotation_rad is None:
            raise ValueError(
                f'{res.technique_name}: 3x3 covariance requires rotation_rad to be set'
            )
        return cast(
            NDArrayFloatType,
            np.array([res.offset_px[0], res.offset_px[1], res.rotation_rad], np.float64),
        )
    return cast(NDArrayFloatType, np.array([res.offset_px[0], res.offset_px[1]], np.float64))


@dataclass(frozen=True)
class ConsensusSelection:
    """Output of :func:`consensus_selection`.

    Parameters:
        best: The winning consensus subset (never empty).
        excluded: Viable results outside the winning subset, in input order.
        runner_up_confidence: Highest corroborating summed confidence of
            any consensus subset formed within ``excluded`` alone, leaving out
            a pass-2 result the seeded-pair test keeps apart from its seed in
            the winning subset; ``0.0`` when nothing else was excluded.
        runner_up_has_quorum: True when that runner-up subset has at least
            two *corroborating* members -- a genuine alternative consensus
            rather than a lone dissenter (or a dissenter echoed only by
            its own prior-descendant).
    """

    best: list[NavTechniqueResult]
    excluded: list[NavTechniqueResult]
    runner_up_confidence: float
    runner_up_has_quorum: bool


def fusion_compromise_px(
    delta_vu: NDArrayFloatType, cov_a: NDArrayFloatType, cov_b: NDArrayFloatType
) -> float:
    """How far the fusion of two estimates lies from what either one supports, in pixels.

    The information-form fusion of two estimates ``a`` and ``b`` lies at
    ``a + C_a (C_a + C_b)^-1 (b - a)``: on the more precise one along a
    direction where the two differ in precision, halfway between them where
    they are equally vague.  The fused point's offsets from ``a`` and from
    ``b`` are resolved onto the principal axes of ``C_a + C_b`` (the axes along
    which the pair's disagreement is measured), the smaller of the two is kept
    per axis, and the result is the length of those parts together.

    So a precise estimate beside a vague one scores near zero, and so do two
    that complement each other, each precise along an axis where the other is
    vague (a ring edge fixing the radial direction and a body centroid the
    direction along the edge); two that are both vague along their separation
    score about half of it.  The fused point is computed from the full
    matrices, not axis by axis: when the two estimates' principal axes differ,
    a disagreement along one axis moves the fusion along the other, and two
    estimates a few pixels apart can fuse at a point several pixels from both.

    Parameters:
        delta_vu: ``(k,)`` separation ``b - a`` expressed in a basis both
            estimates observe.
        cov_a: ``(k, k)`` covariance of ``a`` in that basis.
        cov_b: ``(k, k)`` covariance of ``b`` in that basis.

    Returns:
        The compromise in pixels.
    """
    total = cov_a + cov_b
    _, axes = np.linalg.eigh(total)
    fused = cov_a @ (np.linalg.pinv(total) @ delta_vu)
    from_a = axes.T @ fused
    from_b = axes.T @ (fused - delta_vu)
    return float(np.linalg.norm(np.minimum(np.abs(from_a), np.abs(from_b))))


def _agreement_adjacency(
    results: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    agreement_compromise_px: float,
    rcond: float,
) -> tuple[NDArrayBoolType, NDArrayBoolType]:
    """Pairwise agreement matrix: Mahalanobis threshold with a pixel-floor fallback.

    Two results agree iff *either* their pairwise Mahalanobis distance is
    at most ``agreement_sigma`` *or* their Euclidean translation distance
    (in pixels, ignoring the rotation component for 3-DoF results) is at
    most ``agreement_pixel_floor``.

    A Mahalanobis agreement between results farther apart than the pixel
    floor is kept only when, along each principal axis of the pair, fusing the
    two would land on whichever is the more precise there, to within
    ``agreement_compromise_px`` over all axes (see :func:`fusion_compromise_px`).
    Two results that are each uncertain by tens of pixels along their
    separation are statistically consistent however far apart they lie, yet
    their fusion is a compromise between two answers at most one of which is
    right; such a pair is a disagreement.  A precise result beside a vague
    one passes, and so do two that complement each other, each precise where
    the other is vague.

    A pass-2 result and a technique that seeded its prior are held to the same
    compromise test when they agree by the pixel floor but not by Mahalanobis
    distance.  The refine searched small windows the seed placed, so landing
    several standard deviations from the seed means it found something other
    than what the seed predicted there, and fusing the two would put the
    answer at a point neither supports: a fusion of two estimates with crossed
    principal axes can slide along one's weak axis to several pixels from both.
    Seeds are recorded by technique name, so a seeded pair is recognized by name.

    The Mahalanobis distance differences the rotation component of two
    3-DoF results linearly; that is correct here because every input
    rotation is bounded to the small-angle window enforced by the caller,
    so the pairwise angle difference never approaches the ``+-pi`` wrap.

    The pixel floor compensates for per-technique covariances that
    report only a CRLB-style precision (FFT subpixel localization or
    M-estimator information) and so under-estimate the actual position
    uncertainty by orders of magnitude.  See ``EnsembleConfig`` for the
    motivation.

    Parameters:
        results: List of two or more per-technique results.
        agreement_sigma: Maximum pairwise Mahalanobis distance for agreement.
        agreement_pixel_floor: Maximum pairwise Euclidean translation
            distance (in pixels) for agreement; ``0.0`` disables.
        agreement_compromise_px: Largest compromise (pixels, see
            :func:`fusion_compromise_px`) the fusion of a Mahalanobis-agreeing
            pair past the pixel floor may make; ``0.0`` disables the test.
        rcond: rcond passed to ``pinvh``.

    Returns:
        ``(adj, seed_split)``: ``adj`` is the symmetric boolean ``(n, n)``
        agreement matrix with a True diagonal; ``seed_split[i, j]`` is True when
        result ``j`` seeded result ``i``'s prior and the seeded-pair test above
        is what keeps them apart.

    Raises:
        ValueError: if 2-DoF and 3-DoF results are mixed.
    """
    n = len(results)
    adj = np.eye(n, dtype=bool)
    seed_split = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mu_i = result_param_vector(results[i])
        cov_i = np.asarray(results[i].covariance_px2, np.float64)
        for j in range(i + 1, n):
            mu_j = result_param_vector(results[j])
            cov_j = np.asarray(results[j].covariance_px2, np.float64)
            if mu_i.shape != mu_j.shape:
                # 2-DoF and 3-DoF results never coexist within one image.
                # If the orchestrator routed mismatched shapes through the
                # ensemble, fail loudly rather than coerce.
                raise ValueError(
                    'Mixed-DoF technique results in one ensemble: '
                    f'{results[i].technique_name} produced {mu_i.shape} parameters; '
                    f'{results[j].technique_name} produced {mu_j.shape}.'
                )
            dist = mahalanobis_distance(mu_i, cov_i, mu_j, cov_j, rcond=rcond)
            mahal_match = dist <= agreement_sigma
            # The pixel floor is likewise measured only in the translation
            # directions both results observe: a rank-1 result's junk
            # along-edge component must not push a genuine radial agreement
            # past the floor.
            translation_delta = mu_i[:2] - mu_j[:2]
            basis_t = observable_intersection_basis(
                np.ascontiguousarray(cov_i[:2, :2]), np.ascontiguousarray(cov_j[:2, :2])
            )
            delta_t = basis_t.T @ translation_delta
            pixel_dist = float(np.linalg.norm(delta_t))
            pixel_match = agreement_pixel_floor > 0.0 and pixel_dist <= agreement_pixel_floor
            # A pass-2 result inside its own seed's floor but statistically
            # apart from it is held to the compromise test as well.
            i_seeded_j = results[i].technique_name in results[j].prior_source_techniques
            j_seeded_i = results[j].technique_name in results[i].prior_source_techniques
            floor_only = (i_seeded_j or j_seeded_i) and pixel_match and not mahal_match
            if ((mahal_match and not pixel_match) or floor_only) and agreement_compromise_px > 0.0:
                compromise = fusion_compromise_px(
                    delta_t,
                    basis_t.T @ cov_i[:2, :2] @ basis_t,
                    basis_t.T @ cov_j[:2, :2] @ basis_t,
                )
                if compromise > agreement_compromise_px:
                    seed_split[j, i] = floor_only and i_seeded_j
                    seed_split[i, j] = floor_only and j_seeded_i
                    mahal_match = False
                    pixel_match = False
            if mahal_match or pixel_match:
                adj[i, j] = True
                adj[j, i] = True
    return adj, seed_split


def _best_neighborhood(
    results: list[NavTechniqueResult],
    adj: NDArrayBoolType,
    candidate_indices: list[int],
) -> list[int]:
    """The candidate neighborhood with the highest corroborating confidence.

    Each candidate ``i`` sponsors the subset of ``candidate_indices`` that
    agree with it pairwise (including itself).  Subsets are scored by the
    summed confidence of their *corroborating* members (a prior-descendant
    whose seeding technique is in the subset adds no vote mass); ties break
    toward the larger subset, then toward the earlier candidate, so the
    selection is deterministic.

    Parameters:
        results: The full per-technique result list (for confidences).
        adj: Pairwise agreement matrix over ``results``.
        candidate_indices: Indices eligible for this selection round.

    Returns:
        The winning subset as a list of indices into ``results``.
    """
    best: list[int] = []
    best_key = (float('-inf'), 0)
    for i in candidate_indices:
        subset = [j for j in candidate_indices if adj[i, j]]
        key = (corroborating_confidence([results[j] for j in subset]), len(subset))
        if key > best_key:
            best = subset
            best_key = key
    return best


def consensus_selection(
    results: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    agreement_compromise_px: float,
    rcond: float,
    max_allowed_rotation_deg: float,
) -> ConsensusSelection:
    """Select the consensus subset the ensemble combines, and its outliers.

    Every result sponsors a candidate subset -- the results that agree
    with it pairwise (Mahalanobis distance within ``agreement_sigma`` or
    translation distance within ``agreement_pixel_floor``, a Mahalanobis
    agreement past the floor counting only when the pair's fusion compromises
    by at most ``agreement_compromise_px``; see :func:`fusion_compromise_px`).  The subset
    with the highest summed confidence wins; results outside it are
    excluded from the combine rather than forcing a conflict, and the
    strongest consensus formable among the excluded results alone is
    reported as the runner-up so the caller can distinguish a lone
    mis-converged dissenter from a genuine alternative answer.

    Unlike single-link transitive-closure grouping, a result that agrees
    with only one member of the winning subset does not drag the whole
    subset toward it: membership requires pairwise agreement with the
    sponsoring result.

    Parameters:
        results: Non-empty list of per-technique results.
        agreement_sigma: Maximum pairwise Mahalanobis distance for agreement.
        agreement_pixel_floor: Maximum pairwise Euclidean translation
            distance (in pixels) for agreement; ``0.0`` disables.
        agreement_compromise_px: Largest compromise (pixels, see
            :func:`fusion_compromise_px`) a Mahalanobis-agreeing pair's fusion
            may make when they are farther apart than the pixel floor;
            ``0.0`` disables.
        rcond: rcond passed to ``pinvh``.
        max_allowed_rotation_deg: Small-angle bound; each 3-DoF input's
            ``rotation_rad`` magnitude must stay strictly below this many
            degrees for the linear rotation differencing to be valid.

    Returns:
        The selection: winning subset, excluded results, and runner-up
        summary.

    Raises:
        NavContractError: if a 3-DoF input's rotation magnitude is at or
            above the small-angle bound (an upstream programming error).
        ValueError: if 2-DoF and 3-DoF results are mixed.
    """
    n = len(results)
    max_rotation_rad = math.radians(max_allowed_rotation_deg)
    for res in results:
        cov = np.asarray(res.covariance_px2, np.float64)
        # Small-angle assumption: a 3-DoF rotation outside +-max_rotation_deg
        # is an upstream error (every technique clamps its rotation fit to
        # this bound), and would break the linear rotation differencing.
        # Raised as NavContractError (not a bare assert) so the check
        # survives python -O and is never swallowed as an ordinary
        # technique failure.
        if (
            cov.shape == (3, 3)
            and res.rotation_rad is not None
            and abs(res.rotation_rad) >= max_rotation_rad
        ):
            raise NavContractError(
                f'{res.technique_name}: rotation '
                f'{math.degrees(res.rotation_rad):.3f} deg violates small-angle '
                f'bound +-{max_allowed_rotation_deg:.3f} deg'
            )
    if n == 1:
        return ConsensusSelection(
            best=list(results),
            excluded=[],
            runner_up_confidence=0.0,
            runner_up_has_quorum=False,
        )
    adj, seed_split = _agreement_adjacency(
        results,
        agreement_sigma=agreement_sigma,
        agreement_pixel_floor=agreement_pixel_floor,
        agreement_compromise_px=agreement_compromise_px,
        rcond=rcond,
    )
    best_indices = _best_neighborhood(results, adj, list(range(n)))
    best_set = set(best_indices)
    excluded_indices = [i for i in range(n) if i not in best_set]
    runner_up_confidence = 0.0
    runner_up_has_quorum = False
    # A pass-2 result kept apart from its seed in the winning subset only by the
    # seeded-pair test is not an alternative to that subset: its windows were
    # placed by the seed, so it cannot form a runner-up.
    contenders = [i for i in excluded_indices if not seed_split[i, sorted(best_set)].any()]
    if contenders:
        runner_indices = _best_neighborhood(results, adj, contenders)
        runner_group = [results[j] for j in runner_indices]
        runner_up_confidence = corroborating_confidence(runner_group)
        runner_up_has_quorum = len(corroborating_members(runner_group)) >= 2
    return ConsensusSelection(
        best=[results[i] for i in sorted(best_set)],
        excluded=[results[i] for i in excluded_indices],
        runner_up_confidence=runner_up_confidence,
        runner_up_has_quorum=runner_up_has_quorum,
    )


@dataclass(frozen=True)
class FusionPruning:
    """Output of :func:`leave_out_compromising_fusions`.

    Parameters:
        combine_set: The estimators the combine fuses, those left out removed.
        best_group: The consensus subset, every left-out result removed.
        excluded: The excluded results, every left-out result appended.
        left_out: The results left out, in the order they were found.
    """

    combine_set: list[NavTechniqueResult]
    best_group: list[NavTechniqueResult]
    excluded: list[NavTechniqueResult]
    left_out: list[NavTechniqueResult]


def leave_out_compromising_fusions(
    estimators: list[NavTechniqueResult],
    combine_set: list[NavTechniqueResult],
    collapsed_groups: list[list[NavTechniqueResult]],
    best_group: list[NavTechniqueResult],
    excluded: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    agreement_compromise_px: float,
    rcond: float,
) -> FusionPruning:
    """Leave out of the combine the less precise side of a fusion that would compromise.

    A result joins the consensus by agreeing with the subset's sponsor, which may
    be one member of a collapsed set, while the combine fuses the set through its
    representative, which may be less precise than that member.  So each other
    estimator that agrees with some member of a collapsed set (the selection's
    pairwise test) is held again to the test against the set's representative: a
    Mahalanobis agreement inside the pixel floor stands, and otherwise the pair
    must agree by Mahalanobis distance or the pixel floor and fuse with a
    compromise of at most ``agreement_compromise_px``
    (:func:`fusion_compromise_px`).  Two estimates a few pixels apart at several
    standard deviations are not one answer, and fusing them lands between them.
    The estimator must also lie within ``agreement_sigma`` (Mahalanobis distance) of
    at least one member of the set.  One that agrees only through the pixel floor,
    several standard deviations from every member, contradicts the set: fusing it
    moves the answer toward it by as much as the compromise limit when the two are
    about equally precise, and lets it outweigh the set when it is the more precise.

    When the pair fails, the side less precise along the pair's separation is left
    out: the estimator, or the whole set (its representative and every member).
    A set is one measurement reported by several techniques, and its precision is
    taken as its most precise member's: the joiner may disagree with any of them,
    and the representative's own covariance can be the looser view (a fit chosen
    for the features it consumed rather than its covariance).
    A pair in which either side seeded the other's prior is not tested here: a
    seeded refine is held to its seed when the consensus is selected.

    Two star results (:data:`STAR_CONSENSUS_TECHNIQUES`) that consumed no star in
    common and whose translations lie more than ``agreement_sigma`` apart
    (Mahalanobis distance) disagree about which detections are the catalog stars:
    each places the whole star field.  When one matched at least
    ``_OUTNUMBERING_FACTOR`` times as many distinct stars as the other, and no star
    result on still more stars contradicts it, the other is left out, with every
    member of its set when it represents one; closer counts are left to the combine.
    Results are decided from the most stars down, so the outcome does not depend on
    their order.  Seeding does not exempt such a pair: a
    refine seeded at a two-star match that finds nine stars elsewhere has tested the
    match and found the stars somewhere else.

    Parameters:
        estimators: The resolved independent estimators; each collapsed set's
            representative is the same object as the set's first member.
        combine_set: The set the combine fuses, position for position with
            ``estimators`` (a representative may carry a blended confidence).
        collapsed_groups: Each collapsed set, representative first.
        best_group: The consensus subset.
        excluded: The results already excluded from the consensus.
        agreement_sigma: Maximum pairwise Mahalanobis distance for agreement.
        agreement_pixel_floor: Translation distance (pixels) of the pixel-floor
            agreement; ``0.0`` disables it.
        agreement_compromise_px: Largest compromise (pixels) of a fusion; ``0.0``
            disables both tests (nothing is left out).
        rcond: rcond passed to ``pinvh``.

    Returns:
        The pruned sets.  ``best_group`` and ``excluded`` are matched by result
        equality (technique and consumed features), because a representative in
        ``estimators`` is a copy of the consensus member it stands for.
    """
    if agreement_compromise_px <= 0.0:
        return FusionPruning(list(combine_set), list(best_group), list(excluded), [])
    index = {id(est): k for k, est in enumerate(estimators)}
    representatives = {id(group[0]) for group in collapsed_groups}
    dropped: set[int] = set()
    left_out: list[NavTechniqueResult] = []
    for group in collapsed_groups:
        rep_k = index[id(group[0])]
        for k, est in enumerate(estimators):
            if rep_k in dropped:
                break
            if k in dropped or id(est) in representatives or _seeded_either_way(est, group):
                continue
            terms = _pair_terms(
                est, group[0], agreement_pixel_floor=agreement_pixel_floor, rcond=rcond
            )
            if terms.fuses(
                agreement_sigma=agreement_sigma, compromise_px=agreement_compromise_px
            ) and _consistent_with_a_member(
                est,
                group,
                agreement_sigma=agreement_sigma,
                agreement_pixel_floor=agreement_pixel_floor,
                rcond=rcond,
            ):
                continue
            set_variance = min(_variance_along(member, terms.separation_vu) for member in group)
            if _variance_along(est, terms.separation_vu) >= set_variance:
                dropped.add(k)
                left_out.append(est)
            else:
                dropped.add(rep_k)
                left_out.extend(group)
    members = {id(group[0]): group for group in collapsed_groups}
    for k in _outnumbered_star_results(
        estimators,
        members,
        dropped,
        agreement_sigma=agreement_sigma,
        agreement_pixel_floor=agreement_pixel_floor,
        rcond=rcond,
    ):
        dropped.add(k)
        left_out.extend(members.get(id(estimators[k]), [estimators[k]]))
    return FusionPruning(
        combine_set=[r for k, r in enumerate(combine_set) if k not in dropped],
        best_group=[r for r in best_group if r not in left_out],
        excluded=[*excluded, *left_out],
        left_out=left_out,
    )


def _outnumbered_star_results(
    estimators: list[NavTechniqueResult],
    members: dict[int, list[NavTechniqueResult]],
    dropped: set[int],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    rcond: float,
) -> list[int]:
    """Return the star estimators outnumbered by a star estimator they contradict.

    Two star estimators contradict each other when they consumed no star in common
    and their translations lie more than ``agreement_sigma`` apart (Mahalanobis
    distance in the directions both observe).  An estimator is outnumbered by one it
    contradicts that matched at least ``_OUTNUMBERING_FACTOR`` times as many distinct
    stars, provided that one is uncontested: no surviving estimator on more stars
    contradicts it.  Estimators are decided from the most stars down, so the outcome
    does not depend on the order of ``estimators``.

    Parameters:
        estimators: The resolved independent estimators.
        members: Each collapsed set's members keyed by ``id`` of its representative;
            a representative's stars are those of every member.
        dropped: Indices into ``estimators`` already left out; they neither outnumber
            nor are returned.
        agreement_sigma: Maximum Mahalanobis distance for agreement.
        agreement_pixel_floor: Translation distance (pixels) of the pixel-floor
            agreement, passed through to :func:`_pair_terms`.
        rcond: rcond passed to ``pinvh``.

    Returns:
        The indices into ``estimators`` to leave out, in ascending order.
    """
    stars = {
        k: {
            feature_id
            for member in members.get(id(est), [est])
            if member.technique_name in STAR_CONSENSUS_TECHNIQUES
            for feature_id in member.feature_ids
        }
        for k, est in enumerate(estimators)
        if k not in dropped and est.technique_name in STAR_CONSENSUS_TECHNIQUES
    }

    def contradicts(a: int, b: int) -> bool:
        if stars[a] & stars[b]:
            return False
        terms = _pair_terms(
            estimators[a], estimators[b], agreement_pixel_floor=agreement_pixel_floor, rcond=rcond
        )
        return terms.translation_mahalanobis > agreement_sigma

    order = sorted(stars, key=lambda k: (-len(stars[k]), k))
    out: set[int] = set()
    for i, k in enumerate(order):
        if k in out:
            continue
        if any(
            len(stars[m]) > len(stars[k]) and contradicts(m, k) for m in order[:i] if m not in out
        ):
            continue
        for j in order[i + 1 :]:
            if j in out or len(stars[k]) < _OUTNUMBERING_FACTOR * len(stars[j]):
                continue
            if contradicts(k, j):
                out.add(j)
    return sorted(out)


def _consistent_with_a_member(
    est: NavTechniqueResult,
    group: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    rcond: float,
) -> bool:
    """Return whether a result lies within ``agreement_sigma`` of some member of a set.

    The distance is the Mahalanobis distance of :func:`_pair_terms`.

    Parameters:
        est: The result tested against the set.
        group: The collapsed set's members.
        agreement_sigma: Maximum Mahalanobis distance for agreement.
        agreement_pixel_floor: Translation distance (pixels) of the pixel-floor agreement,
            passed through to :func:`_pair_terms`.
        rcond: rcond passed to ``pinvh``.

    Returns:
        True when the Mahalanobis distance to at least one member is at most
        ``agreement_sigma``.
    """
    for member in group:
        terms = _pair_terms(est, member, agreement_pixel_floor=agreement_pixel_floor, rcond=rcond)
        if terms.mahalanobis <= agreement_sigma:
            return True
    return False


def _seeded_either_way(est: NavTechniqueResult, group: list[NavTechniqueResult]) -> bool:
    """Return whether ``est`` seeded the set's prior or a member of the set seeded ``est``'s."""
    names = {member.technique_name for member in group}
    return bool(
        est.technique_name in group[0].prior_source_techniques
        or est.prior_source_techniques & names
    )


def _variance_along(result: NavTechniqueResult, direction_vu: NDArrayFloatType) -> float:
    """Return a result's translation variance along a direction (px^2).

    Parameters:
        result: The result.
        direction_vu: ``(2,)`` unit direction in ``(v, u)``.

    Returns:
        ``d^T C d`` for the result's translation covariance ``C``, or infinity when
        the result does not observe the whole direction (a rank-1 ring edge along
        its tangent).
    """
    cov = np.asarray(result.covariance_px2, np.float64)[:2, :2]
    observed = observable_basis(cov).T @ direction_vu
    if float(observed @ observed) < _FULLY_OBSERVED:
        return math.inf
    return float(direction_vu @ cov @ direction_vu)


@dataclass(frozen=True)
class _PairTerms:
    """The agreement and fusion measures of two results.

    Parameters:
        mahalanobis: Mahalanobis distance between the two.
        pixel_match: Whether their observable separation is within the pixel floor.
        compromise_px: How far their fusion lies from what either supports
            (:func:`fusion_compromise_px`).
        separation_vu: ``(2,)`` unit direction from the first to the second in
            ``(v, u)``, within the directions both observe (zero when they coincide
            there).
        translation_mahalanobis: Mahalanobis distance of the translations alone,
            within the directions both observe (a rotation difference does not
            count).
    """

    mahalanobis: float
    pixel_match: bool
    compromise_px: float
    separation_vu: NDArrayFloatType
    translation_mahalanobis: float

    def agrees(self, *, agreement_sigma: float, compromise_px: float) -> bool:
        """Return the selection's pairwise agreement (the floor, or Mahalanobis without compromise).

        Parameters:
            agreement_sigma: Maximum Mahalanobis distance for agreement.
            compromise_px: Largest compromise of a Mahalanobis agreement past the floor.

        Returns:
            True when the pair agrees.
        """
        mahal_match = self.mahalanobis <= agreement_sigma
        return self.pixel_match or (mahal_match and self.compromise_px <= compromise_px)

    def fuses(self, *, agreement_sigma: float, compromise_px: float) -> bool:
        """Return whether the pair agrees and its fusion does not compromise.

        A Mahalanobis agreement inside the pixel floor fuses; otherwise the pair
        must agree by either test and compromise by at most ``compromise_px``.

        Parameters:
            agreement_sigma: Maximum Mahalanobis distance for agreement.
            compromise_px: Largest compromise of the fusion.

        Returns:
            True when the pair may be fused.
        """
        mahal_match = self.mahalanobis <= agreement_sigma
        if mahal_match and self.pixel_match:
            return True
        return (mahal_match or self.pixel_match) and self.compromise_px <= compromise_px


def _pair_terms(
    a: NavTechniqueResult,
    b: NavTechniqueResult,
    *,
    agreement_pixel_floor: float,
    rcond: float,
) -> _PairTerms:
    """Return the agreement and fusion measures of two results.

    Every translation measure is taken in the intersection of the two results'
    observable subspaces, as the selection's pairwise test takes it.

    Parameters:
        a: The first result.
        b: The second result.
        agreement_pixel_floor: Translation distance (pixels) of the pixel-floor
            agreement; ``0.0`` disables it.
        rcond: rcond passed to ``pinvh``.

    Returns:
        The pair's measures.

    Raises:
        ValueError: if a 2-DoF and a 3-DoF result are mixed.
    """
    mu_a, mu_b = result_param_vector(a), result_param_vector(b)
    cov_a = np.asarray(a.covariance_px2, np.float64)
    cov_b = np.asarray(b.covariance_px2, np.float64)
    if mu_a.shape != mu_b.shape:
        raise ValueError(
            'Mixed-DoF technique results in one ensemble: '
            f'{a.technique_name} produced {mu_a.shape} parameters; '
            f'{b.technique_name} produced {mu_b.shape}.'
        )
    basis_t = observable_intersection_basis(
        np.ascontiguousarray(cov_a[:2, :2]), np.ascontiguousarray(cov_b[:2, :2])
    )
    delta_t = basis_t.T @ (mu_b[:2] - mu_a[:2])
    cov_a_t = basis_t.T @ cov_a[:2, :2] @ basis_t
    cov_b_t = basis_t.T @ cov_b[:2, :2] @ basis_t
    separation = float(np.linalg.norm(delta_t))
    direction = basis_t @ delta_t / separation if separation > 0.0 else np.zeros(2)
    translation_d2 = float(delta_t @ np.linalg.pinv(cov_a_t + cov_b_t, rcond=rcond) @ delta_t)
    return _PairTerms(
        mahalanobis=float(mahalanobis_distance(mu_a, cov_a, mu_b, cov_b, rcond=rcond)),
        pixel_match=agreement_pixel_floor > 0.0 and separation <= agreement_pixel_floor,
        compromise_px=fusion_compromise_px(delta_t, cov_a_t, cov_b_t),
        separation_vu=cast(NDArrayFloatType, direction),
        translation_mahalanobis=math.sqrt(max(translation_d2, 0.0)),
    )
