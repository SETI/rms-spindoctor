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

from spindoctor.nav_orchestrator.ensemble_observability import (
    mahalanobis_distance,
    observable_intersection_basis,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ConsensusSelection',
    'consensus_selection',
    'corroborating_confidence',
    'corroborating_members',
    'fusion_compromise_px',
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
            any consensus subset formed within ``excluded`` alone; ``0.0``
            when nothing was excluded.
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

    Along a single direction the information-form fusion of two estimates
    ``a`` and ``b`` lies ``delta * s_a**2 / (s_a**2 + s_b**2)`` from ``a`` and
    ``delta * s_b**2 / (s_a**2 + s_b**2)`` from ``b``: on the more precise one
    when the two differ, halfway between them when they are equally vague.
    The separation is resolved onto the principal axes of ``C_a + C_b`` (the
    axes along which the pair's disagreement is measured), each axis is
    treated as such a one-dimensional fusion using the two estimates'
    variances along it, the smaller of the two distances is kept per axis,
    and the result is the length of those parts together.

    So a precise estimate beside a vague one scores near zero, and so do two
    that complement each other, each precise along an axis where the other is
    vague (a ring edge fixing the radial direction and a body centroid the
    direction along the edge); two that are both vague along their separation
    score about half of it.

    Parameters:
        delta_vu: ``(k,)`` separation ``b - a`` expressed in a basis both
            estimates observe.
        cov_a: ``(k, k)`` covariance of ``a`` in that basis.
        cov_b: ``(k, k)`` covariance of ``b`` in that basis.

    Returns:
        The compromise in pixels.
    """
    _, axes = np.linalg.eigh(cov_a + cov_b)
    along = axes.T @ delta_vu
    var_a = np.einsum('ik,ij,jk->k', axes, cov_a, axes)
    var_b = np.einsum('ik,ij,jk->k', axes, cov_b, axes)
    total = var_a + var_b
    nearer = np.where(
        total > 0.0, np.minimum(var_a, var_b) / np.where(total > 0.0, total, 1.0), 0.0
    )
    return float(np.linalg.norm(along * nearer))


def _agreement_adjacency(
    results: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    agreement_compromise_px: float,
    rcond: float,
) -> NDArrayBoolType:
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
        Symmetric boolean ``(n, n)`` matrix with a True diagonal.

    Raises:
        ValueError: if 2-DoF and 3-DoF results are mixed.
    """
    n = len(results)
    adj = np.eye(n, dtype=bool)
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
            if mahal_match and not pixel_match and agreement_compromise_px > 0.0:
                compromise = fusion_compromise_px(
                    delta_t,
                    basis_t.T @ cov_i[:2, :2] @ basis_t,
                    basis_t.T @ cov_j[:2, :2] @ basis_t,
                )
                mahal_match = compromise <= agreement_compromise_px
            if mahal_match or pixel_match:
                adj[i, j] = True
                adj[j, i] = True
    return adj


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
    adj = _agreement_adjacency(
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
    if excluded_indices:
        runner_indices = _best_neighborhood(results, adj, excluded_indices)
        runner_group = [results[j] for j in runner_indices]
        runner_up_confidence = corroborating_confidence(runner_group)
        runner_up_has_quorum = len(corroborating_members(runner_group)) >= 2
    return ConsensusSelection(
        best=[results[i] for i in sorted(best_set)],
        excluded=[results[i] for i in excluded_indices],
        runner_up_confidence=runner_up_confidence,
        runner_up_has_quorum=runner_up_has_quorum,
    )
