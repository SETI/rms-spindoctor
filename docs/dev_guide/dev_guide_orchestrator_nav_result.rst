==========================================================
Final Output (NavResult)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` is the frozen dataclass the
orchestrator returns from
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`. It carries the
headline answer (offset plus per-axis sigma plus a five-bucket confidence rank) alongside
full diagnostic information about every technique that ran, every feature that was
extracted, and the per-image provenance. The dataclass is the in-memory object the
orchestrator hands back; the curator
(:doc:`dev_guide_orchestrator_curator`) builds a JSON-friendly subset for the per-image
sidecar.

Theory
======

A :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` encodes one of three top-level
outcomes:

- ``'success'`` — the ensemble combine produced an offset above the per-image confidence
  threshold.
- ``'failed'`` — a short-circuit gate fired, every technique was spurious, or the ensemble
  combine fell below the confidence threshold. The result carries no offset.
- ``'conflicted'`` — multiple non-overlapping technique groups had similar summed
  confidences and the ensemble could not select a winner confidently (the best
  group's offset is still reported).

The dataclass enforces consistency invariants in ``__post_init__``:

- ``status='failed'`` must have ``offset_px=None``.
- ``status='success'`` must have a non-``None`` ``offset_px``.
- ``confidence_rank='failed'`` requires ``status='failed'``.
- ``confidence`` must lie in :math:`[0, 1]`.
- ``covariance_px2``, when set, must be square and 2-D.

Three classmethod constructors centralize common shapes:
:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.success`,
:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed`, and
:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.conflicted`. Direct instantiation is
also supported when the caller already knows every field.

Restrictions and assumptions
----------------------------

- The dataclass is **frozen**; the constructors return new instances rather than mutating
  in place.
- The ``feature_inventory``, ``model_metadata``, and ``annotations`` fields are **always
  populated**; even a failure path attaches whatever was extracted before the gate fired
  so the per-image JSON sidecar always has full diagnostic content.
- The covariance is read-only after construction; the dataclass marks the array
  non-writable in ``__post_init__``.

Sources of uncertainty
----------------------

The ``sigma_px`` per-axis 1-sigma marginal is derived from the diagonal of the
``covariance_px2``. When the covariance is rank-1 (a flat-ring-only scene), the
``sigma_along_unobservable_px`` field carries the unconstrained-axis sigma so reviewer
tooling can distinguish a rank-1 from a full-rank result.

Configuration
=============

The dataclass carries no YAML configuration of its own. The five-bucket confidence rank
and the per-tier sigma thresholds are configured on the
:class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` (documented at
:doc:`dev_guide_orchestrator_ensemble`); the rank assignment runs inside
:func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank`.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/nav_result.py`` —
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` plus the ``Status`` and
``ConfidenceRank`` Literal aliases.

Public class :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`, frozen dataclass.

Public fields (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status` — one of ``'success'``,
  ``'failed'``, ``'conflicted'``.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.offset_px` — ``(dv, du)`` offset;
  ``None`` on failure.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_px` — per-axis 1-sigma marginal
  uncertainty; ``None`` on failure.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` — set
  when the covariance is rank-1 (e.g. flat-ring-only scenes); ``None`` for full-rank
  results.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.confidence_rank` — five-bucket rank
  (``'high'`` / ``'medium'`` / ``'low'`` / ``'conflicted'`` / ``'failed'``).
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.confidence` — the underlying
  calibrated confidence in :math:`[0, 1]`.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status_reason` — discrete
  :class:`~spindoctor.support.status_reason.NavStatusReason` value.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.covariance_px2` — full 2x2 (or 3x3
  with rotation) covariance; ``None`` on failure.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.per_technique` — list of every
  technique's :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` (kept or
  dropped by the ensemble).
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.excluded_from_consensus` —
  technique names of viable results the ensemble left out of the reported combine
  (outliers rejected against a multi-technique consensus, or the runner-up alternative
  on a conflicted result); empty when every viable result contributed. Also emitted in
  the metadata JSON.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.consensus_techniques` — technique
  names of the results the ensemble actually combined into the reported offset (the
  winning consensus subset), in input order; empty on failure. The orchestrator uses
  this to stamp pass-2 results with the techniques that seeded their prior.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.feature_inventory` — list of
  per-feature :class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary` entries.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.image_classifier` — the
  :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult` verdict.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.provenance` — the
  :class:`~spindoctor.nav_orchestrator.provenance.Provenance` envelope.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.model_metadata` —
  per-:class:`~spindoctor.nav_model.nav_model.NavModel`
  diagnostic dicts keyed by model name.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.annotations` — composite
  :class:`~spindoctor.annotation.annotations.Annotations` collection assembled from every
  registered :class:`~spindoctor.nav_model.nav_model.NavModel`.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.rotation_rad` — Optional fitted camera
  rotation in radians.
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_rotation_rad` — Optional
  1-sigma rotation uncertainty.

Public classmethod constructors:

- :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.success` — build a ``success``-status
  result from the ensemble's offset and covariance.
- :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed` — build a ``failed``-status
  result from a status reason.
- :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.conflicted` — build a
  ``conflicted``-status result when multiple technique groups disagreed.

The dataclass enforces consistency in ``__post_init__``:
``status='failed'`` requires ``offset_px=None``;
``status='success'`` requires a non-``None`` offset;
``confidence_rank='failed'`` requires ``status='failed'``;
``confidence`` must lie in :math:`[0, 1]`;
``covariance_px2`` must be square 2-D and is marked read-only after construction.

Examples
========

**``success`` result on body_partial_overflow.**  After
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` reports
``(12.06, 30.53)`` px and :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` accepts the
result, the orchestrator returns::

    NavResult(
        status='success',
        offset_px=(12.06, 30.53),
        sigma_px=(0.125, 0.122),
        confidence_rank='high',
        confidence=0.794,
        status_reason=NavStatusReason.OK,
        covariance_px2=array([[0.0156, 0.0017], [0.0017, 0.0148]]),
        ...
    )

**``failed`` result on a blank image.**  When the image classifier returns
``image_class='blank'`` the orchestrator's hard-failure short-circuit invokes
:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed`::

    NavResult(
        status='failed',
        offset_px=None,
        sigma_px=None,
        confidence_rank='failed',
        confidence=0.0,
        status_reason=NavStatusReason.NO_SIGNAL_IN_IMAGE,
        ...
    )

**``conflicted`` result on multi_body.**  When pass-1 techniques produce two non-
overlapping high-confidence groups (the disc + limb cohort agreeing around
:math:`(7.0, -18.0)` px and the terminator latching at :math:`(11.6, 12.6)` px), the
ensemble's agreement-gap test fails and the orchestrator returns the best group's
offset (and its covariance-derived sigma) under the conflicted status::

    NavResult(
        status='conflicted',
        offset_px=(11.6, 12.6),
        sigma_px=(0.4, 0.4),
        confidence_rank='conflicted',
        confidence=0.222,
        status_reason=NavStatusReason.CONFLICTED_TECHNIQUES,
        consensus_techniques=['BodyTerminatorNav'],
        excluded_from_consensus=['BodyDiscCorrelateNav', 'BodyLimbNav'],
        ...
    )

:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.conflicted` requires an offset
and covariance — the conflicted result still reports the winning group's answer, with
its confidence multiplied down and the rank hard-set to ``'conflicted'`` so downstream
consumers refuse it without explicit opt-in.

Every result carries a populated
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.feature_inventory` and
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.model_metadata` so the curator can write
the per-image JSON sidecar identically across the three status paths.
