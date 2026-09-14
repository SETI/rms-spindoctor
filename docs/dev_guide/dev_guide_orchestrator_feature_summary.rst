==========================================================
Per-Feature Post-Mortem (NavFeatureSummary)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary` is the small frozen
dataclass that records one entry per emitted
:class:`~spindoctor.feature.feature.NavFeature` in the per-image
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.feature_inventory` list. The summary
carries enough information about each feature for the curator to write a post-mortem entry
into the per-image JSON sidecar — the per-feature identifier, its type, its source model,
the gate decision (kept vs. dropped, with a stable reason), and its bounding box. It does
not carry the heavy bits of a full
:class:`~spindoctor.feature.feature.NavFeature` (templates, polylines, covariance) because those
would bloat the metadata.

Theory
======

The summary is a deliberately narrow record. Heavy-bit features (BODY_DISC templates,
LIMB_ARC polylines) routinely run to tens of kilobytes per feature; surfacing every byte
of every feature in the per-image JSON would balloon the sidecar to multi-megabyte sizes
without operationally useful content. The summary keeps the post-mortem structure (one
entry per emitted feature) without the bulk.

The :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.gated` flag and the
:attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.gate_reason` together
describe the per-feature reliability-gate decision: ``gated=False`` means the feature
reached the technique-feasibility loop; ``gated=True`` means the gate dropped it before
any technique saw it. The reason string is stable across images so reviewer tooling can
correlate gate refusals across an entire imaging campaign.

Restrictions and assumptions
----------------------------

- Every field is required at construction time. The dataclass enforces invariants in
  ``__post_init__``: feature id and source model must be non-empty strings; reliability
  must lie in :math:`[0, 1]`; the bounding box must be a length-4 tuple of ints.
- The gate-reason field is required non-empty when ``gated`` is ``True``.
- The dataclass is frozen.

Sources of uncertainty
----------------------

The summary reports no uncertainty. The reliability score is the per-feature value the
producing :class:`~spindoctor.nav_model.nav_model.NavModel` reported; the gate decision is
deterministic given the per-image context.

Configuration
=============

The dataclass carries no YAML configuration of its own. The per-feature reliability
floor that drives the gate decision lives on the
:class:`~spindoctor.feature.reliability.FeatureReliabilityGate` (per-type floor) rather than on
this summary dataclass.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/feature_summary.py`` —
:class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary`.

Public class :class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary`, frozen
dataclass. Public fields (autodocumented at
:doc:`/api_reference/api_nav_orchestrator`):

- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.feature_id` —
  unique identifier matching the producing
  :attr:`~spindoctor.feature.feature.NavFeature.feature_id`. Non-empty string.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.feature_type` — one of
  the :class:`~spindoctor.feature.feature_type.NavFeatureType` enum values.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.source_model` — name of
  the producing :class:`~spindoctor.nav_model.nav_model.NavModel` (``'stars'``,
  ``'body:DIONE'``, ``'rings:SATURN'``). Non-empty string.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.reliability` — per-feature
  reliability score in :math:`[0, 1]`.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.gated` — bool. ``True``
  when the reliability gate dropped this feature before any technique saw it.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.gate_reason` — str or
  ``None``. Stable English reason when ``gated`` is ``True``; ``None`` otherwise.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.bbox_extfov_vu` —
  half-open ``(v_min, u_min, v_max, u_max)`` bounding box in extfov coordinates;
  4-tuple of ints.
- :attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.reliability_reasons` —
  the producing feature's
  :class:`~spindoctor.feature.feature.NavReliabilityBreakdown`, carried through to the
  per-image JSON so a gated feature's cause is readable without re-running navigation. The
  breakdown's fields are per-feature-type and optional; a missing field means "not applicable
  to this feature type", not zero.

The dataclass enforces invariants in ``__post_init__``: every field must have the
declared type; ``reliability`` must lie in :math:`[0, 1]`; ``gate_reason`` must be a
non-empty string when ``gated`` is ``True``; ``bbox_extfov_vu`` must be a length-4 tuple
of real :class:`int` (numpy int / bool are rejected).

The orchestrator's
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate` builds the per-image
inventory by iterating over the kept and gated cohorts and constructing one
:class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary` per feature via the
private ``_summary_from_feature`` helper, which projects the heavy
:class:`~spindoctor.feature.feature.NavFeature` down to the summary's narrow shape.

Examples
========

**Kept feature on body_partial_overflow.**  The body model emits a LIMB_ARC for Rhea with
reliability 0.82 and bounding box ``(180, 700, 540, 1024)``. The orchestrator's gate
keeps it, so the inventory entry is::

    NavFeatureSummary(
        feature_id='limb_arc:RHEA',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body:RHEA',
        reliability=0.82,
        gated=False,
        gate_reason=None,
        bbox_extfov_vu=(180, 700, 540, 1024),
    )

**Gated feature.**  The body model emits a LIMB_ARC for a fully-lit body whose
reliability score falls below the per-type floor (the textbook ``body_full_fov`` case).
The gate drops it::

    NavFeatureSummary(
        feature_id='limb_arc:DIONE',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body:DIONE',
        reliability=0.14,
        gated=True,
        gate_reason='reliability_0.140_below_threshold_0.300',
        bbox_extfov_vu=(364, 364, 644, 644),
    )

The reason string embeds the feature's reliability and the per-type threshold
(``'reliability_{r:.3f}_below_threshold_{t:.3f}'``); its ``reliability_`` prefix is
stable across images (the embedded numbers vary per feature), so a campaign-level
reviewer who counts LIMB_ARC entries whose ``gate_reason`` starts with
``'reliability_'`` across 1,000 ``body_full_fov``-class images learns directly how
often LIMB_ARC features fall below the reliability gate on full-disc bodies. The
reason does not say which formula component drove the score down; that attribution
comes from the feature's
:attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.reliability_reasons`
breakdown.

**Per-image inventory size.**  A multi-body Cassini fly-by image with 3 bodies plus 50
predictable stars produces an inventory of 50 STAR plus up to 12 body-derived features
(LIMB_ARC + BODY_DISC + TERMINATOR_ARC per body) — about 62 entries totaling ~6 KB of
JSON. The same scene's heavy-bit
:class:`~spindoctor.feature.feature.NavFeature` objects total tens of megabytes; the summary's
narrow shape is what makes the per-image sidecar tractable to read in a reviewer's text
editor.
