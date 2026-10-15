"""Resolve a consensus group into mutually independent estimators.

The ensemble fuses agreeing per-technique results with an information-form
(Kalman-style) merge whose every step assumes the inputs are *independent*
measurements: their precisions add, and their mutual agreement earns a
confidence boost.  That assumption is false whenever two results draw on the
same underlying error source.  Fusing correlated results as independent both
over-tightens the combined covariance (so the reported sigma is too small and
the tier too high) and lets a redundant member drag the fused offset.

This module resolves a winning consensus group into the set of estimators the
combine may treat as independent, correcting four known correlations:

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

* **Two techniques on one feature (R4).**  Two results that consumed a common
  feature measured the same thing: the three star techniques centroid the same
  detections of the same catalog stars, so a one-star unique match and the
  one-inlier refine of that star agree to the last digit because they are one
  measurement reported twice, and a misidentified star moves both by the same
  amount.  Any two results whose ``feature_ids`` share an entry are collapsed
  to a single representative.  Every non-star technique consumes a feature type
  no other technique reads, so in practice the rule acts on the star family; it
  is stated over features rather than technique names so that it holds for any
  technique that comes to share one.

R1 is a directional drop (the descendant is subordinate to its seeds, which
remain independent of one another); R2, R3, and R4 are symmetric collapses
(neither member is subordinate, so the pair is replaced by one representative).  The
representative of a collapsed set is its highest-positional-precision member,
with a deterministic tie-break, so the fused offset is the single best view of
the shared measurement rather than a spuriously tightened average of two.  A set
joined by shared features is represented first by the member that consumed the
most features, which holds every measurement the others made; a fit to fewer
stars can carry the tighter covariance without being the better view.
"""

from dataclasses import dataclass

from spindoctor.nav_orchestrator.ensemble_observability import positional_precision
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_technique.diagnostics import (
    StarFieldDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult

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

    A one-star ``StarUniqueMatchNav`` match, a one-inlier ``StarRefineNav``
    refine, and a one-inlier ``StarFieldFromCatalogNav`` lock each localize the
    offset from a single detection: nothing in the solution corroborates the
    identification itself.  Multi-star solutions and every non-star technique
    return False.

    Parameters:
        res: The per-technique result to classify.

    Returns:
        True when ``res`` is a ``StarUniqueMatchNav`` result whose match mode
        is ``'one_star'``, a ``StarRefineNav`` result that used at most one
        star, or a ``StarFieldFromCatalogNav`` result with at most one inlier;
        False for every other result, including multi-star star results and
        all non-star techniques.
    """
    diag = res.diagnostics
    if isinstance(diag, StarUniqueMatchDiagnostics):
        return diag.mode == 'one_star'
    if isinstance(diag, StarRefineDiagnostics):
        return diag.n_stars_used <= 1
    if isinstance(diag, StarFieldDiagnostics):
        return diag.n_inliers <= 1
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
            (R2 / R3 / R4), representative first, then the members it stood in for.
            Singletons are not listed.
    """

    estimators: list[NavTechniqueResult]
    dropped_descendants: list[NavTechniqueResult]
    collapsed_groups: list[list[NavTechniqueResult]]


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


def _correlation_adjacency(
    group: list[NavTechniqueResult],
    *,
    scattered_light: bool,
) -> list[set[int]]:
    """Return undirected correlation edges among group indices (R2, R3, and R4).

    Two results are joined by an edge when they are correlated witnesses of
    one shared error source: R2 joins any two ring techniques (both in
    ``_RING_TECHNIQUES``), which read one shared catalog model; R3 joins the
    disc and limb techniques (``_SCATTER_SENSITIVE_BODY_TECHNIQUES``) observing
    a shared body, but only when ``scattered_light`` is True; R4 joins any two
    results whose ``feature_ids`` have an entry in common, since a feature both
    consumed is one measurement whichever technique reports it.

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
            # R4: a feature both consumed is one measurement reported twice.
            shared_feature = bool(set(ri.feature_ids) & set(rj.feature_ids))
            if ring_pair or scatter_pair or shared_feature:
                adj[i].add(j)
                adj[j].add(i)
    return adj


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
    rcond: float,
) -> IndependenceResolution:
    """Resolve a consensus group into mutually independent estimators.

    Parameters:
        group: The winning consensus subset (non-empty).
        image_classifier: The image-quality verdict; its
            ``background_gradient_score`` gates the R3 scattered-light rule.
        scattered_light_gradient_score: Background-gradient score at or above
            which a frame is treated as scattered-light for R3.
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
    if strong_witness_present:
        dropped_idx = {
            i
            for i, r in enumerate(group)
            if _is_seeded_descendant(r, group_names) and is_single_star_result(r)
        }
    survivors = [r for i, r in enumerate(group) if i not in dropped_idx]
    if not survivors:
        # Defensive: strong_witness_present already guarantees a non-single-star
        # survivor, so this cannot fire; kept so the combine never sees an empty
        # set even if the drop rule changes.
        dropped_idx = set()
        survivors = list(group)
    dropped = [group[i] for i in sorted(dropped_idx)]

    # R2 / R3 / R4: collapse each correlated peer set to a single representative.
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
        # Representative: in a set joined by shared features (R4), the member
        # that consumed the most features, since it holds every measurement the
        # others made and more; then the highest positional precision; then a
        # deterministic tie-break so the choice never depends on input order or
        # hashing.  Feature count comes first because a fit to fewer stars can
        # report the tighter covariance -- a two-star match carries the
        # predicted centroid bound, a seven-star fit its measured scatter.
        shares_features = any(
            set(survivors[i].feature_ids) & set(survivors[j].feature_ids)
            for i in comp
            for j in comp
            if i < j
        )
        rep_idx = max(
            comp,
            key=lambda i: (
                len(set(survivors[i].feature_ids)) if shares_features else 0,
                positional_precision(survivors[i].covariance_px2, rcond=rcond),
                survivors[i].technique_name,
                survivors[i].feature_ids,
            ),
        )
        keep.add(rep_idx)
        rep = survivors[rep_idx]
        others = [survivors[i] for i in comp if i != rep_idx]
        collapsed_groups.append([rep, *others])

    estimators = [survivors[i] for i in range(len(survivors)) if i in keep]
    return IndependenceResolution(
        estimators=estimators,
        dropped_descendants=dropped,
        collapsed_groups=collapsed_groups,
    )
