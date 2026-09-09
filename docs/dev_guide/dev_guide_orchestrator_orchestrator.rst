==========================================================
Orchestrator (NavOrchestrator)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` is the top-level driver that
turns one observation into one
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`. The driver builds the per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`, runs every registered
:class:`~spindoctor.nav_model.nav_model.NavModel`'s
:meth:`~spindoctor.nav_model.nav_model.NavModel.create_model` and
:meth:`~spindoctor.nav_model.nav_model.NavModel.to_features`, applies the feature-reliability
gate, runs every feasible prior-free
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` (pass 1), reconciles those results
into a prior via :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`, runs every feasible
prior-required technique against that prior (pass 2), and reconciles the union of pass-1
and pass-2 results into the final
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Two passes, four short-circuit gates, and an exception-sandbox discipline keep the driver
from raising through to its caller — every failure mode surfaces on the returned
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` instead. The four short-circuit gates
are:

- ``_HARD_FAILURE_TO_REASON`` — the
  :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier` reports ``blank``,
  ``fully_overexposed``, ``mostly_missing_data``, or ``corrupt``.
- ``NO_FEATURES_EXTRACTED`` — every :class:`~spindoctor.nav_model.nav_model.NavModel` emitted
  zero features.
- ``ALL_FEATURES_GATED`` — the reliability gate dropped every emitted feature.
- ``NO_FEASIBLE_TECHNIQUES`` — pass 1 produced zero technique results (no technique was
  feasible on the gated cohort).

Theory
======

The orchestrator is a deterministic pipeline. Each per-image run is a sequence of stages:
build context, build models, extract features, gate features, run pass 1, fuse pass 1 into a
prior, run pass 2, fuse the union, return a
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Pipeline stages
---------------

1. **Provenance and context.**  Build a reproducibility envelope (git SHA, loaded SPICE
   kernels, static-data hashes) and a per-image global-state container. The context
   carries the extfov image, the per-image masks (sensor / saturation / cosmic-ray), the
   per-image noise sigma, the image-quality classifier verdict, the shared image-side
   derivatives (gradient magnitude, gradient vector, edge distance transform), and the
   per-instrument settings (data units, rotation flag, max rotation cap).
2. **Model construction.**  Iterate the registered
   :class:`~spindoctor.nav_model.nav_model.NavModel` set, calling each model's
   :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model` per the operator's
   ``only_models`` glob filter. An exception in any model is logged and the model is
   dropped from the per-image set.
3. **Feature extraction.**  Each surviving model's
   :meth:`~spindoctor.nav_model.nav_model.NavModel.to_features` is invoked; an exception is
   logged and the model contributes zero features.
4. **Reliability gate.**  The per-feature reliability gate (an instance of
   :class:`~spindoctor.feature.reliability.FeatureReliabilityGate`) drops features whose
   per-feature ``reliability`` score falls below the per-type floor.
5. **Pass 1.**  Run the registered techniques with
   :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior` ``False`` whose
   :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types` overlaps the
   surviving feature set and whose ``is_feasible`` reports feasible. Primary-tier
   techniques run first; fallback-tier techniques then run, skipping any technique whose
   source body already has a non-spurious primary result, so a fallback runs only when
   the primary fails on that body.
6. **Pass-1 ensemble.**  Reconcile the pass-1 results via
   :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`. A ``'success'`` ensemble's
   offset and covariance become the pass-2 prior; a ``'conflicted'`` ensemble installs no
   prior (pass 2 still runs); when the ensemble fails the
   technique pipeline returns a failed
   :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` immediately.
7. **Pass 2.**  Run every prior-required technique with the pass-2 context (a copy of the
   pass-1 context with the prior offset attached via
   :meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior`).
8. **Final ensemble.**  Reconcile the union of pass-1 and pass-2 results into the final
   :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Exception-sandbox discipline
----------------------------

Every :class:`~spindoctor.nav_model.nav_model.NavModel` callback (``create_model``,
``to_features``, ``to_annotations``) and every
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` callback (``navigate``)
runs inside a broad ``except Exception`` block. An
exception is logged with full traceback and the offending model or technique is treated as
if it produced no output. This is intentional — the orchestrator must never raise through
to its caller; failures land on the
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult.status` field instead so the per-image
JSON sidecar always exists.

Glob-pattern model and technique filters
----------------------------------------

The constructor accepts ``only_models`` and ``only_techniques`` glob patterns (or pattern
lists) so an operator can restrict which models or techniques run for debugging. The
syntax matches gitignore-style globs with leading ``!`` for exclusion. The model-name
convention is ``prefix:VALUE`` (``rings:SATURN``, ``body:DIONE``, plain ``stars`` for the
catalog star model); a pattern missing a colon expands to ``prefix:*`` so the common
shorthand (``rings``) matches every namespaced ring model under that prefix.

Restrictions and assumptions
----------------------------

- The orchestrator assumes
  :class:`~spindoctor.nav_model.nav_model.NavModel` and
  :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclasses are well-formed and
  non-malicious — the exception sandbox catches programmer bugs but cannot recover from
  state-corrupting plugins.
- The pass-2 prior is seeded only when the pass-1 ensemble's status is ``'success'``. A
  conflicted pass 1 still runs pass 2 — but with no prior installed, since a conflicted
  best-group offset would lock pass 2 toward one arbitrary mode instead of letting pass-2
  evidence break the tie. When pass 1 fails, the orchestrator returns immediately and
  pass 2 does not run.
- The image-quality classifier's hard-failure short-circuit fires before any model is
  constructed; a ``blank`` or ``fully_overexposed`` image never invokes a
  :class:`~spindoctor.nav_model.nav_model.NavModel`.

Sources of uncertainty
----------------------

The orchestrator itself reports no uncertainty. Per-technique uncertainty is captured in
each technique's
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2`; the final
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult.covariance_px2` is the
precision-weighted-merge result from
:func:`~spindoctor.nav_orchestrator.ensemble.ensemble`.

Configuration
=============

The orchestrator's runtime knobs come from three places: per-instrument YAML
(``config_4N0_inst_*.yaml`` via
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`), construction-time
overrides on the constructor, and the per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`'s default values.

Per-instrument YAML keys (read via
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs`):

- ``data_units`` — ``raw_dn`` or ``calibrated_if``. Drives the saturation-mask and
  noise-sigma branches.
- ``noise.saturation_dn`` — saturation DN threshold (``raw_dn`` only).
- ``noise.marker_value`` — missing-data marker DN.
- ``image_quality_thresholds.*`` — the per-instrument
  :class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds` values.
- ``fit_camera_rotation`` — bool, top-level in the per-camera block; turns on 3-DoF
  technique fits.
- ``max_rotation_deg`` — float, top-level in the per-camera block; rotation cap.

Constructor-level overrides:

- ``models`` — list of constructed
  :class:`~spindoctor.nav_model.nav_model.NavModel` instances. The caller builds these per-image.
- ``config`` — optional :class:`~spindoctor.config.config.Config` override.
- ``only_models`` — glob pattern or list selecting which models run. Default ``'*'``.
- ``only_techniques`` — glob pattern or list selecting which techniques run. Default
  ``'*'``.
- ``ensemble_config`` — optional
  :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` override.
- ``image_quality_thresholds`` — optional
  :class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds` override.
- ``image_derivatives_config`` — optional
  :class:`~spindoctor.nav_orchestrator.image_derivatives.ImageDerivativesConfig` override.
- ``spindoctor_version`` — string written into provenance.

Implementation
==============

Source files:

- ``src/spindoctor/nav_orchestrator/orchestrator.py`` —
  :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator`, the
  ``_HARD_FAILURE_TO_REASON`` mapping, the ``_ModelRegistry`` glob-filter helper, and the
  per-stage private methods.
- ``src/spindoctor/nav_orchestrator/nav_context.py`` —
  :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`; documented at
  :doc:`dev_guide_orchestrator_nav_context`.
- ``src/spindoctor/nav_orchestrator/nav_result.py`` —
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`; documented at
  :doc:`dev_guide_orchestrator_nav_result`.
- ``src/spindoctor/nav_orchestrator/ensemble.py`` —
  :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`; documented at
  :doc:`dev_guide_orchestrator_ensemble`.
- ``src/spindoctor/nav_orchestrator/image_classifier.py`` —
  :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier`; documented at
  :doc:`dev_guide_orchestrator_image_classifier`.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` —
  :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives`;
  documented at :doc:`dev_guide_techniques_image_derivatives`.
- ``src/spindoctor/nav_orchestrator/instrument_config.py`` —
  :class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`; documented at
  :doc:`dev_guide_orchestrator_instrument_config`.
- ``src/spindoctor/nav_orchestrator/provenance.py`` —
  :class:`~spindoctor.nav_orchestrator.provenance.Provenance`; documented at
  :doc:`dev_guide_orchestrator_provenance`.
- ``src/spindoctor/nav_orchestrator/status_reason_info.py`` —
  ``STATUS_REASON_INFO_TEMPLATE``, the per-status-reason operator log lines.
- ``src/spindoctor/feature/reliability.py`` —
  :class:`~spindoctor.feature.reliability.FeatureReliabilityGate` and
  :class:`~spindoctor.feature.reliability.GatedFeatureRecord`.

Public class :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator`, base
:class:`~spindoctor.support.nav_base.NavBase`.

Public methods (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.prepare` — build per-image
  state without running any technique. Returns an
  :class:`~spindoctor.nav_orchestrator.orchestrator.OrchestratorPrep` dataclass carrying
  the context, provenance, image-classifier verdict, features, feature inventory,
  per-model metadata, and merged annotations. The ``apply_gate`` keyword (default
  ``True``) controls whether the reliability gate runs; the manual-nav dialog passes
  ``apply_gate=False`` so the operator sees every emitted feature.
- :meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate` — run the full
  pipeline on one observation. Returns one
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate`:

1. Build the :class:`~spindoctor.nav_orchestrator.provenance.Provenance` envelope.
2. Build the per-image :class:`~spindoctor.nav_orchestrator.nav_context.NavContext` and the
   image classifier verdict. Log the verdict at INFO.
3. **Hard-failure short-circuit.**  When the classifier's ``image_class`` is in
   ``_HARD_FAILURE_TO_REASON`` (``blank`` /
   ``fully_overexposed`` / ``mostly_missing_data`` / ``corrupt``), return a failed
   :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` with the matching
   :class:`~spindoctor.support.status_reason.NavStatusReason`.
4. Build every :class:`~spindoctor.nav_model.nav_model.NavModel` by invoking
   :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model` (exception-sandboxed).
5. Extract features from each surviving model
   (:meth:`~spindoctor.nav_model.nav_model.NavModel.to_features`, exception-sandboxed) and apply
   the reliability gate.
6. Build the feature inventory, the model metadata snapshot, and the merged annotations.
7. **Two short-circuit gates** on the gated cohort:

   - ``NO_FEATURES_EXTRACTED`` when every model emitted zero features.
   - ``ALL_FEATURES_GATED`` when the gate dropped every feature.

   Each returns a failed
   :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` via the ``_fail`` helper.
   Either gate reports ``BODY_FILLS_FOV`` instead when a body model declined its body
   as covering the extended frame with neither limb nor terminator inside it and every
   feature emitted was a star behind it, so that a statistics report can tell an image
   that could not have been navigated from one that failed; see
   :class:`~spindoctor.support.status_reason.NavStatusReason`.
8. **Pass 1.**  Run the prior-free techniques whose feature types overlap the gated
   cohort and whose ``is_feasible`` returns feasible: primary-tier techniques first,
   then fallback-tier techniques excluding any body already covered by a non-spurious
   primary result. An exception in any technique is
   logged and the technique contributes no result.
9. **NO_FEASIBLE_TECHNIQUES** when pass 1 produced zero results — return a failed
   :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
10. Reconcile pass-1 results via :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`. When
    the ensemble's ``status`` is ``'failed'`` return the failed
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
11. Build the pass-2 context via
    :meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior` carrying the pass-1
    ensemble's offset and covariance.
12. **Pass 2.**  Run every prior-required technique against the pass-2 context.
13. Reconcile the union of pass-1 and pass-2 results via
    :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` to produce the final
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
14. Log the final offset / confidence / per-technique summary and the per-status-reason
    operator INFO lines from ``STATUS_REASON_INFO_TEMPLATE``.

The ``_fail`` helper wraps
:meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed` and emits the per-status-reason
INFO log lines defined in ``STATUS_REASON_INFO_TEMPLATE``. Every short-circuit gate
listed above invokes ``_fail`` to keep the operator log consistent.

Examples
========

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right with about 22 % of the disc off-frame. The
    orchestrator builds the body model, extracts LIMB_ARC + BODY_DISC + TERMINATOR_ARC
    features, runs pass 1 with
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
    :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`, and
    :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` (the latter
    two flag spurious; only BodyLimb produces a usable result). The pass-1 ensemble
    selects BodyLimb's offset of ``(12.06, 30.53)`` px, the pass-2 prior is set, and pass 2
    runs no prior-required techniques (no STAR features in this scene). The final
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` reports
    :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status` ``'success'`` against the
    operator-verified offset :math:`(\Delta v, \Delta u) = (11.0, 29.5)` px.

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single bright star (Vega) in an otherwise empty FOV. The orchestrator's pass-1
    :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
    produces a 1-star prior; pass 2 runs
    :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` against that
    prior, polishes the offset, and returns it with
    :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.confidence` capped at
    0.5 by the single-inlier post-sigmoid cap. The final ensemble fuses both technique
    results and reports
    :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status` ``'success'`` against the
    operator-verified offset :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px.

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV. Pass 1 runs
    :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`, which
    produces a multi-star prior via the triplet hash; pass 2 runs
    :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`, which polishes
    the offset using the full predictable cohort. The final ensemble fuses both and
    reports the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px.

**Hard-failure short-circuit example.**  An over-exposed Cassini calibration target
returns the classifier's ``fully_overexposed`` class. The orchestrator returns the failed
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` with
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.status_reason`
``IMAGE_OVEREXPOSED`` before any :class:`~spindoctor.nav_model.nav_model.NavModel`
constructs. The per-image JSON sidecar still
emits with the classifier verdict and the provenance envelope so reviewer tooling can
correlate the failure across an entire campaign.
