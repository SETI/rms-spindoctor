"""Resolve a consensus group into mutually independent estimators.

The ensemble fuses agreeing per-technique results with an information-form
(Kalman-style) merge whose every step assumes the inputs are *independent*
measurements: their precisions add, and their mutual agreement earns a
confidence boost.  That assumption is false whenever two results draw on the
same underlying error source.  Fusing correlated results as independent both
over-tightens the combined covariance (so the reported sigma is too small and
the tier too high) and lets a redundant member drag the fused offset.

This module resolves a winning consensus group into the set of estimators the
combine may treat as independent, correcting three known correlations:

* **Seeded second-pass refine (R1).**  A pass-2 ``StarRefineNav`` searches a
  small window around the pass-1 prior, so its offset is conditionally
  dependent on whichever techniques seeded that prior
  (``prior_source_techniques``).  A refine that rests on a *single* star adds
  no independent constraint of its own -- it merely re-observes its seed near
  the predicted position -- and its CRLB-tight covariance would otherwise let
  it dominate the precision-weighted merge.  So a seeded single-star refine is
  dropped from the combine when the group also holds a stronger, non-single-
  star witness (a body or ring fix, or a multi-star result) that it would
  otherwise drag off; a genuine single-star lock, where the refine is the
  only refinement of a legitimately weak fix and no stronger witness is
  present, keeps it.  A multi-star refine always carries independent
  information and is kept (the ensemble separately denies it a corroboration
  vote).

  One seeded single-star refine is not a re-observation of its seed: the
  polish of an unseeded one-star match.  When the group holds a one-star
  ``StarUniqueMatchNav`` result on the same star, the refine lands on that
  match (within ``agreement_sigma`` of it under the refine's own covariance),
  no stronger witness contradicts the refine, and at least one stronger
  witness confirms it at sub-pixel scale -- along each direction in which that
  witness's standard deviation is at most ``confirming_axis_max_sigma_px``,
  the two agree within ``agreement_sigma`` of their combined standard
  deviation there (a ring edge across itself, say) -- the star was found by
  the wide one-star search, the prior only told the refine where to look, and
  the refine replaces the one-star match in the combine: the match is dropped
  instead, and the refine carries the match's independent vote.  A witness
  vague in every direction confirms nothing.  The one-star match reports a
  covariance as large as its own offset, so without the replacement the star
  could not fix the direction the stronger witness leaves open (along a ring
  edge).

* **Two ring techniques on one catalog (R2).**  ``RingEdgeNav`` and
  ``RingAnnulusNav`` observe the same predicted ring geometry from the same
  catalog model, so a radially misplaced catalog puts both wrong in the same
  direction by the same amount.  They are collapsed to a single representative
  witness.

* **Disc and limb on one veiling gradient (R3).**  On a scattered-light frame
  ``BodyDiscCorrelateNav`` and ``BodyLimbNav`` both lock onto the same
  large-scale brightness ramp rather than the body, so their errors correlate.
  When the image's background-gradient score exceeds the scattered-light
  threshold, the disc and limb results on the same body are collapsed to a
  single representative.

R1 is a directional drop (the descendant is subordinate to its seeds, which
remain independent of one another); R2 and R3 are symmetric collapses (neither
member is subordinate, so the pair is replaced by one representative).  The
representative of a collapsed set is its highest-positional-precision member,
with a deterministic tie-break, so the fused offset is the single best view of
the shared measurement rather than a spuriously tightened average of two.

The representative votes for the whole set, so it carries the set's seed
dependence rather than its own: a technique seeded the set only when it seeded
every member.  A set holding a pass-1 member found its answer without any prior,
so its vote is independent even when the member standing in for it is a pass-2
refine whose prior another technique of the group helped form; denying that vote
leaves the group without the star witness it holds.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from spindoctor.nav_orchestrator.ensemble_observability import (
    mahalanobis_distance,
    mixed_scale_pinvh,
    observable_basis,
    positional_precision,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_technique.diagnostics import (
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'STAR_CONSENSUS_TECHNIQUES',
    'IndependenceResolution',
    'is_single_star_result',
    'resolve_independent_estimators',
]

#: Star techniques whose non-single-star result is a corroborated, feature-
#: matched fix (a multi-star pattern match, a two-star unique match, or a
#: multi-star refine).  Such a fix cross-checks its identification against more
#: than one star, so it outranks a lone brightness-centroid blob as a position
#: witness: a blob that disagrees with it is the outlier.  Paired with
#: :func:`is_single_star_result` to admit only the corroborated members.
STAR_CONSENSUS_TECHNIQUES = frozenset(
    {'StarFieldFromCatalogNav', 'StarUniqueMatchNav', 'StarRefineNav'}
)

#: Ring techniques that read one shared ``NavModelRings`` catalog per frame.
#: Any two of these observing the same frame are correlated witnesses of one
#: model.
_RING_TECHNIQUES = frozenset({'RingEdgeNav', 'RingAnnulusNav'})

#: Intensity-based body techniques that both measure a scattered-light veiling
#: gradient when one is present, correlating their errors.
_SCATTER_SENSITIVE_BODY_TECHNIQUES = frozenset({'BodyDiscCorrelateNav', 'BodyLimbNav'})


def is_single_star_result(res: NavTechniqueResult) -> bool:
    """Return True when a result rests on a single star with no cross-check.

    A one-star ``StarUniqueMatchNav`` match and a one-inlier
    ``StarRefineNav`` refine each localize the offset from a single
    detection: nothing in the solution corroborates the identification
    itself.  Multi-star solutions and every non-star technique return False.

    Parameters:
        res: The per-technique result to classify.

    Returns:
        True when ``res`` is a ``StarUniqueMatchNav`` result whose match mode
        is ``'one_star'`` or a ``StarRefineNav`` result that used at most one
        star; False for every other result, including multi-star star results
        and all non-star techniques.
    """
    diag = res.diagnostics
    if isinstance(diag, StarUniqueMatchDiagnostics):
        return diag.mode == 'one_star'
    if isinstance(diag, StarRefineDiagnostics):
        return diag.n_stars_used <= 1
    return False


@dataclass(frozen=True)
class IndependenceResolution:
    """Outcome of resolving a consensus group into independent estimators.

    Parameters:
        estimators: The results the combine may treat as independent, in the
            input group's order.  One member per independence class: seeded
            single-star refines removed, each correlated peer set replaced by
            its representative.  Never empty when the input group is non-empty.
        dropped_descendants: Seeded single-star refines removed by R1 (kept for
            logging / provenance; they cast no vote and lend no precision).
        collapsed_groups: One entry per correlated peer set that was collapsed
            (R2 / R3), representative first, then the members it stood in for.
            Singletons are not listed.
        replaced_matches: One ``(refine, match)`` pair per one-star match
            that a confirmed single-star refine replaced: the match is absent
            from ``estimators``, and the refine stands in its place with an
            empty ``prior_source_techniques``, so it carries the match's
            independent vote.
    """

    estimators: list[NavTechniqueResult]
    dropped_descendants: list[NavTechniqueResult]
    collapsed_groups: list[list[NavTechniqueResult]]
    replaced_matches: list[tuple[NavTechniqueResult, NavTechniqueResult]] = dataclasses.field(
        default_factory=list
    )


def _is_seeded_descendant(res: NavTechniqueResult, group_names: set[str]) -> bool:
    """Return True when a result was seeded by another technique in the group.

    A result is a seeded descendant when at least one of its
    ``prior_source_techniques`` (the techniques whose pass-1 offset seeded its
    search prior) is also present in the group, other than the result's own
    technique.  A result seeded only by itself, or only by techniques absent
    from the group, is not a descendant.

    Parameters:
        res: The result whose seeding provenance is examined.
        group_names: The technique names of every member of the group ``res``
            belongs to.

    Returns:
        True when ``res.prior_source_techniques`` intersects ``group_names``
        after removing ``res``'s own technique name; False otherwise.
    """
    return bool(res.prior_source_techniques & (group_names - {res.technique_name}))


def _is_one_star_match(res: NavTechniqueResult) -> bool:
    """Return True for a one-star ``StarUniqueMatchNav`` result."""
    diag = res.diagnostics
    return isinstance(diag, StarUniqueMatchDiagnostics) and diag.mode == 'one_star'


def _has_precise_axis(cov: NDArrayFloatType, *, max_sigma_px: float) -> bool:
    """Return True when a translation covariance pins down some direction to ``max_sigma_px``.

    The covariance is resolved into principal axes within its observable
    subspace, so a ring edge's unobservable along-edge direction (zero variance
    by the null-space convention) is not an axis at all.

    Parameters:
        cov: ``(2, 2)`` translation covariance.
        max_sigma_px: Largest standard deviation (px) of a precise axis.

    Returns:
        True when some observable principal axis has a standard deviation of at
        most ``max_sigma_px``.
    """
    basis = observable_basis(cov)
    if basis.shape[1] == 0:
        return False
    variances = np.linalg.eigvalsh(basis.T @ cov @ basis)
    return bool(np.any(variances <= max_sigma_px * max_sigma_px))


def _confirmed_one_star_match(
    refine: NavTechniqueResult,
    group: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    confirming_axis_max_sigma_px: float,
    rcond: float,
) -> int | None:
    """Return the index of the one-star match a single-star refine may replace.

    The refine stands in for a one-star ``StarUniqueMatchNav`` result when all
    of the following hold:

    - the match is not itself seeded by a groupmate;
    - the two rest on the same star (equal ``feature_ids``);
    - the refine's offset lies within ``agreement_sigma`` of the match's under
      the refine's own translation covariance (both centroid one detection, so
      they coincide to within the refine's precision);
    - every member of the group that is not a single-star result agrees with
      the refine by Mahalanobis distance within ``agreement_sigma``;
    - at least one such member is precise in some direction: a principal axis
      of its observable covariance has a standard deviation of at most
      ``confirming_axis_max_sigma_px``.  The Mahalanobis bound then holds the
      separation along that axis within ``agreement_sigma`` of the pair's
      combined standard deviation there, which is sub-pixel, so the refine is
      confirmed at sub-pixel scale in at least one direction.  A ring edge
      confirms across itself; a witness vague in every direction confirms
      nothing, however small its Mahalanobis distance.

    Parameters:
        refine: A seeded single-star ``StarRefineNav`` result.
        group: The consensus group holding ``refine``.
        agreement_sigma: Threshold of every agreement test, in standard
            deviations.
        confirming_axis_max_sigma_px: Largest standard deviation (px) of a
            witness axis that counts as precise.
        rcond: rcond for the generalized inverses.

    Returns:
        The index into ``group`` of the match the refine replaces, or None when
        no member qualifies (the first qualifying index when several do).
    """
    group_names = {r.technique_name for r in group}
    strong = [r for r in group if not is_single_star_result(r)]
    refine_mu = np.asarray(refine.offset_px[:2], np.float64)
    refine_cov = np.asarray(refine.covariance_px2, np.float64)[:2, :2]
    for res in strong:
        dist = mahalanobis_distance(
            refine_mu,
            refine_cov,
            np.asarray(res.offset_px[:2], np.float64),
            np.asarray(res.covariance_px2, np.float64)[:2, :2],
            rcond=rcond,
        )
        if dist > agreement_sigma:
            return None
    if not any(
        _has_precise_axis(
            np.asarray(res.covariance_px2, np.float64)[:2, :2],
            max_sigma_px=confirming_axis_max_sigma_px,
        )
        for res in strong
    ):
        return None
    refine_info = mixed_scale_pinvh(refine_cov, rcond=rcond)
    for i, res in enumerate(group):
        if not _is_one_star_match(res) or _is_seeded_descendant(res, group_names):
            continue
        if set(res.feature_ids) != set(refine.feature_ids):
            continue
        delta = refine_mu - np.asarray(res.offset_px[:2], np.float64)
        if float(delta @ refine_info @ delta) <= agreement_sigma * agreement_sigma:
            return i
    return None


def _correlation_adjacency(
    group: list[NavTechniqueResult],
    *,
    scattered_light: bool,
) -> list[set[int]]:
    """Return undirected correlation edges among group indices (R2 and R3).

    Two results are joined by an edge when they are correlated witnesses of
    one shared error source: R2 joins any two ring techniques (both in
    ``_RING_TECHNIQUES``), which read one shared catalog model; R3 joins the
    disc and limb techniques (``_SCATTER_SENSITIVE_BODY_TECHNIQUES``) observing
    a shared body, but only when ``scattered_light`` is True.

    Parameters:
        group: The results to test pairwise, indexed positionally.
        scattered_light: Whether the frame carries a scattered-light veiling
            gradient; gates the R3 disc/limb edge.

    Returns:
        An adjacency list of length ``len(group)``; entry ``i`` is the set of
        indices ``j`` correlated with result ``i``.  The relation is symmetric,
        so ``j in adj[i]`` iff ``i in adj[j]``, and no index is adjacent to
        itself.
    """
    n = len(group)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = group[i], group[j]
            names = {ri.technique_name, rj.technique_name}
            # R2: two ring techniques share one catalog model.
            ring_pair = ri.technique_name in _RING_TECHNIQUES and (
                rj.technique_name in _RING_TECHNIQUES
            )
            # R3: disc and limb on the same body share a veiling gradient when
            # the frame carries a scattered-light ramp.
            scatter_pair = (
                scattered_light
                and names <= _SCATTER_SENSITIVE_BODY_TECHNIQUES
                and ri.technique_name != rj.technique_name
                and bool(ri.source_bodies & rj.source_bodies)
            )
            if ring_pair or scatter_pair:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _collapsed_seed_sources(members: list[NavTechniqueResult]) -> frozenset[str]:
    """Return the techniques that seeded every member of a collapsed set.

    The set's representative votes for all of its members, so its vote depends on
    a pass-1 prior only when every member's does.  Only a pass-2 result carries
    seeds, and only the shared-feature collapse of the star techniques puts one in
    a set, beside the pass-1 star fits that seeded it.

    Parameters:
        members: The collapsed set's members, in any order.

    Returns:
        The intersection over the members of their ``prior_source_techniques``;
        empty when any member is prior-free.
    """
    return frozenset.intersection(*(m.prior_source_techniques for m in members))


def _connected_components(adj: list[set[int]]) -> list[list[int]]:
    """Return the connected components of an undirected adjacency list.

    Parameters:
        adj: A symmetric adjacency list; ``adj[i]`` holds the indices adjacent
            to ``i``.

    Returns:
        A list of components covering every index ``0 .. len(adj) - 1`` exactly
        once.  Each component is sorted in ascending index order, and an index
        with no edges forms its own singleton component.  The components
        themselves are ordered by their smallest member.
    """
    n = len(adj)
    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adj[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(nbr)
        components.append(sorted(comp))
    return components


def resolve_independent_estimators(
    group: list[NavTechniqueResult],
    *,
    image_classifier: NavImageClassifierResult,
    scattered_light_gradient_score: float,
    agreement_sigma: float,
    confirming_axis_max_sigma_px: float,
    rcond: float,
) -> IndependenceResolution:
    """Resolve a consensus group into mutually independent estimators.

    Parameters:
        group: The winning consensus subset (non-empty).
        image_classifier: The image-quality verdict; its
            ``background_gradient_score`` gates the R3 scattered-light rule.
        scattered_light_gradient_score: Background-gradient score at or above
            which a frame is treated as scattered-light for R3.
        agreement_sigma: Threshold, in standard deviations, of the tests that
            confirm a single-star refine replacing a one-star match (see the
            module docstring).
        confirming_axis_max_sigma_px: Largest standard deviation (px) of a
            witness axis precise enough to confirm such a refine.
        rcond: rcond for the pseudo-inverse used to weigh precision.

    Returns:
        An :class:`IndependenceResolution`.  ``estimators`` is the deduplicated,
        correlation-collapsed set the combine treats as independent; it is
        never empty when ``group`` is non-empty.

    Raises:
        ValueError: if ``group`` is empty.
    """
    if not group:
        raise ValueError('empty group passed to resolve_independent_estimators')

    # R1: drop a seeded single-star refine, but only when a stronger,
    # non-single-star witness is present for it to override.  A pure
    # single-star lock (no stronger witness) keeps its refine, which is then
    # the only refinement of a legitimately weak fix rather than a redundant
    # vote against a body or multi-star consensus.  Guard against emptying the
    # set defensively (the strong-witness condition already prevents it).
    # Exclusion is by index, not value: NavTechniqueResult equality keys only
    # on (technique_name, feature_ids), so a value-membership test could drop a
    # distinct result that happened to share those.
    group_names = {r.technique_name for r in group}
    strong_witness_present = any(not is_single_star_result(r) for r in group)
    dropped_idx: set[int] = set()
    replaced_idx: dict[int, int] = {}
    if strong_witness_present:
        for i, r in enumerate(group):
            if not (_is_seeded_descendant(r, group_names) and is_single_star_result(r)):
                continue
            match_idx = _confirmed_one_star_match(
                r,
                group,
                agreement_sigma=agreement_sigma,
                confirming_axis_max_sigma_px=confirming_axis_max_sigma_px,
                rcond=rcond,
            )
            if match_idx is None or match_idx in replaced_idx.values():
                dropped_idx.add(i)
            else:
                replaced_idx[i] = match_idx
    survivors_idx = [
        i for i in range(len(group)) if i not in dropped_idx and i not in replaced_idx.values()
    ]
    if not survivors_idx:
        # Defensive: strong_witness_present already guarantees a non-single-star
        # survivor, so this cannot fire; kept so the combine never sees an empty
        # set even if the drop rule changes.
        dropped_idx = set()
        replaced_idx = {}
        survivors_idx = list(range(len(group)))
    # A refine standing in for a one-star match carries the match's
    # independence: the wide search found the star, so its vote is not a
    # re-observation of the prior.
    survivors = [
        dataclasses.replace(group[i], prior_source_techniques=frozenset())
        if i in replaced_idx
        else group[i]
        for i in survivors_idx
    ]
    dropped = [group[i] for i in sorted(dropped_idx)]
    replaced = [(group[i], group[m]) for i, m in sorted(replaced_idx.items())]

    # R2 / R3: collapse each correlated peer set to a single representative.
    score = image_classifier.background_gradient_score
    scattered_light = score is not None and score >= scattered_light_gradient_score
    adj = _correlation_adjacency(survivors, scattered_light=scattered_light)
    components = _connected_components(adj)

    keep: set[int] = set()
    collapsed_groups: list[list[NavTechniqueResult]] = []
    for comp in components:
        if len(comp) == 1:
            keep.add(comp[0])
            continue
        # Representative: highest positional precision, then a deterministic
        # tie-break so the choice never depends on input order or hashing.
        rep_idx = max(
            comp,
            key=lambda i: (
                positional_precision(survivors[i].covariance_px2, rcond=rcond),
                survivors[i].technique_name,
                survivors[i].feature_ids,
            ),
        )
        keep.add(rep_idx)
        rep = dataclasses.replace(
            survivors[rep_idx],
            prior_source_techniques=_collapsed_seed_sources([survivors[i] for i in comp]),
        )
        survivors[rep_idx] = rep
        others = [survivors[i] for i in comp if i != rep_idx]
        collapsed_groups.append([rep, *others])

    estimators = [survivors[i] for i in range(len(survivors)) if i in keep]
    return IndependenceResolution(
        estimators=estimators,
        dropped_descendants=dropped,
        collapsed_groups=collapsed_groups,
        replaced_matches=replaced,
    )
