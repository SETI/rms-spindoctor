==========================================
Star Refinement (StarRefineNav)
==========================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` is the pass-2 star
refinement technique: given a prior offset produced by the
pass-1 ensemble, it polishes that offset by re-centroiding every predicted catalog star against
the image and averaging the per-star residuals. Each predicted catalog star is shifted by the
prior offset, re-detected inside a tight refinement window, and its sub-pixel centroid compared
to the shifted prediction; the inverse-variance weighted average of the per-star residuals is
the refined delta the technique reports. When the per-instrument camera-rotation flag is on
and at least two stars survive the per-star quality gates, a Procrustes / similarity-transform
refit recovers a translation plus rotation instead of a translation alone.

Feasibility passes when at least one
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature survives the shared usability
gates — predictable and not occluded by a body silhouette or ring annulus. Feasibility
fails when no usable star is offered.

Theory
======

The technique is a per-star centroid average rather than a global pattern-matcher. Pass-1 has
already aligned the catalog to the image to within a few pixels (the prior offset); the
remaining task is to extract the residual misalignment plus, on instruments that support it, the
camera rotation.

Per-star centroid model
-----------------------

For a single predicted catalog star at position :math:`x_{\mathrm{pred}}` with covariance
:math:`\Sigma_{i}` (the Cramer-Rao lower bound carried on the feature), the refinement step
takes the brightness-weighted moment of a small box around the predicted brightness peak in a
search window centered on :math:`x_{\mathrm{pred}} + \mathbf{p}` (where :math:`\mathbf{p}` is the
prior offset). The centroid is rejected when the brightest pixel in the window is below the
detection threshold or the implied per-star residual exceeds a configured cap; the surviving
correspondences are the inlier set.

Translation-only fit
--------------------

For each surviving star :math:`i` the per-star residual is

.. math::

    r_{i} = \mathrm{centroid}_{i} - (x_{\mathrm{pred},\,i} + \mathbf{p}).

The reported delta is the inverse-variance weighted mean

.. math::

    \Delta = \frac{\sum_{i} w_{i} \, r_{i}}{\sum_{i} w_{i}},
    \qquad w_{i} = \frac{1}{\mathrm{tr}(\Sigma_{i})}

where :math:`\mathrm{tr}(\Sigma_{i})` is the trace of the per-star feature covariance; a
zero-or-missing covariance falls back to a unit weight. The reported translation on the result
is :math:`\mathbf{p} + \Delta` (absolute, not delta-from-prior); the orchestrator's ensemble
combine treats every per-technique offset as absolute. The orchestrator also stamps every
pass-2 result with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.prior_source_techniques`
— the technique names whose pass-1 consensus seeded the prior — so the final ensemble lets a
refine polish those techniques' offset without letting it *corroborate* them (a refine's
vote is excluded from corroboration and quorum counts against its own prior's sources).

Rotation-aware fit
------------------

When :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true and the inlier count is at least two, the technique runs
a weighted Kabsch / orthogonal-Procrustes solve on the catalog-vs-detection point clouds to
extract a rotation about the catalog-side weighted centroid plus a translation. The
similarity-transform fitter computes the weighted cross-covariance, decomposes it via SVD, and
applies a determinant correction so that one cohort being a numerical mirror image of the other
cannot produce a reflection masquerading as a rotation. The translation read off the
similarity-transform output is the reported absolute offset for the rotation-fitting
branch; the inverse-variance-weighted mean is used only when rotation fitting is
disabled or there are too few inliers to fit a rotation.

The rotation variance is derived from the inlier residual scatter scaled by the catalog spread:

.. math::

    \sigma_{\theta}^{2} = \max\!\left(\frac{\sigma_{r}^{2}}{S},\; \frac{1}{S}\right),
    \qquad S = \sum_{i} w_{i} \, \lVert x_{\mathrm{cat},\,i} - \bar{x}_{\mathrm{cat}}\rVert^{2}

with :math:`\sigma_{r}^{2}` the per-axis-averaged weighted residual variance. A vanishing
catalog spread (every inlier on top of the centroid) collapses :math:`S` to zero and the
rotation is reported with the project-wide rotation-unobservable sigma sentinel; the ensemble
combine treats that variance as effectively unconstrained.

Restrictions and assumptions
----------------------------

- The technique requires a prior offset on the per-image
  :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`; absent that, it returns a spurious
  zero-confidence result rather than running the per-star centroid loop.
- Predicted catalog positions are assumed to be in the same extended-FOV pixel frame the image
  uses; a SPICE pose error larger than the refinement window radius pushes the prediction
  outside the search window and the per-star detection misses.
- The per-star CRLB covariance carried on each feature is taken at face value as the inlier
  weight; an over-confident CRLB will tighten the reported confidence. The pass-1 model is
  responsible for grounding the CRLB physically.
- The 1-inlier path provides no independent cross-check beyond the prior it was handed: the
  same single observation that drove the pass-1 fit just gets polished. A documented
  post-sigmoid cap (default 0.5) prevents the technique from outranking
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` on the same
  one-star scene. The cap is not the only safeguard: every pass-2 result is stamped with
  the pass-1 consensus techniques that seeded its prior
  (:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.prior_source_techniques`),
  and the ensemble's ``corroborating_members`` filter excludes a refine's vote from
  corroboration and quorum against those source techniques, so a refine cannot
  self-corroborate the prior it was handed.

Sources of uncertainty
----------------------

The reported translation covariance is one of two forms:

- **Single inlier.**  The (2, 2) covariance is the inlier feature's CRLB (a copy of
  :attr:`~spindoctor.feature.feature.NavFeature.position_cov_px`). A missing CRLB falls back to the
  identity matrix.
- **Multi-inlier.**  The (2, 2) covariance is a diagonal of the per-axis weighted residual
  variances about the converged delta, floored at the inverse total weight so the diagonals do
  not collapse to zero on a coincidentally-tight inlier set.

When camera rotation is fit, the (2, 2) translation block is embedded in a (3, 3) covariance
whose rotation diagonal is the variance derived above; with a single inlier the rotation
diagonal is the project-wide rotation-unobservable sentinel and the rotation angle is reported
as zero. At-edge and spurious flags hard-zero the confidence on top of the linear formula.

Configuration
=============

All numeric tunables for this technique live in ``techniques.StarRefineNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``refine_window_px`` — float, default ``6.0`` px. Half-width of the per-star refinement
  window centered on each shifted prediction. Tighter than the unique-match search window
  because the pass-1 prior already put the prediction within a few pixels of the brightness
  peak.
- ``centroid_box_half_px`` — int, default ``3`` px. Half-width of the
  brightness-weighted-moment box around each accepted peak.
- ``max_per_star_residual_px`` — float, default ``4.0`` px. A star whose detected centroid
  sits more than this many pixels from the shifted prediction is dropped before the average;
  almost certainly a wrong peak.
- ``detection_sigma`` — float, default ``4.0`` (dimensionless). Detection-threshold
  multiplier on the per-image noise sigma; the brightest pixel in the refinement window must
  clear ``detection_sigma * image_noise_sigma`` to be accepted.
- ``min_inliers`` — int, default ``1`` (count). Below this many surviving inliers the
  technique reports a spurious zero-confidence result instead of attempting the average.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. The reported absolute translation
  is flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` when its absolute distance from any search-window axis bound falls
  within this tolerance.
- ``single_inlier_confidence_cap`` — float, default ``0.5`` (dimensionless). Post-sigmoid
  confidence cap when the refine fit had only one inlier. A 1-star refine carries no
  independent cross-check beyond the prior it was handed; the cap keeps the technique from
  outranking a 1-star unique-match on the same observation.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). Fraction of
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` at which the converged rotation magnitude trips :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Only the
  multi-inlier Procrustes path uses this threshold; a 1-inlier refine always reports rotation
  as unobservable.
- ``model_error_floor_px`` — float, default ``0.27`` px. Calibrated model-error floor added
  in quadrature to the covariance's translation diagonal: the 2026-07-18 calibration
  campaign measured 2-sigma coverage 0.83 at a 0.22 px floor and 0.87 at this value,
  against the 0.865 reference.

Per-instrument overrides
------------------------

The nine keys above are global; the per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override any of them. The
camera-rotation flag (:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`) and the rotation cap (:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg`)
that this technique reads off the per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext` come from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared sigmoid
combination, see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic and
:doc:`dev_guide_techniques` for the family-level overview. The formula spec is
``techniques.StarRefineNav`` in the same YAML file and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics` plus :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.n_stars_used` — alpha = 1.228,
  offset = 0.0, divisor = 5.0, cap at 1.0. Number of stars that survived the per-star
  quality gates. More inliers earn confidence up to a five-star saturation point.
- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.median_pos_err_px` —
  alpha = -0.81, offset = 0.0, divisor = 1.0, no cap. Median per-star Euclidean residual
  (observed centroid vs. shifted prediction). Larger residuals pull confidence down.
- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.residual_scatter_px` —
  alpha = 0.0, offset = 0.0, divisor = 1.0, no cap. Per-axis weighted RMS of the per-star
  residuals about the fitted delta. Sign-bounded to zero on the calibration campaign (the
  median per-star error already carries the same signal on those rows); kept wired for a
  future real-anchored calibration.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces confidence to zero before the
sigmoid is evaluated. The constant baseline is :math:`\alpha_{0} = 1.795`. No post-sigmoid
``hard_cap`` is applied at the spec level; the ``single_inlier_confidence_cap`` above is
applied by the technique itself only when exactly one inlier survived.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_star_refine.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` and its private
  collect-residuals / build-covariance / Procrustes-refit / fail-shape helpers.
- ``src/spindoctor/nav_technique/_star_helpers.py`` — shared star-feature helpers
  (``usable_stars``,
  ``local_centroid``,
  ``similarity_transform_fit``). The leading underscore
  marks the module as package-private; every star technique imports it.
- ``src/spindoctor/nav_technique/confidence.py`` — the shared sigmoid-combination evaluator.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.name` —
  ``'StarRefineNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.accepts_feature_types`
  — ``frozenset({STAR})``.
- :attr:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.requires_prior` —
  ``True``. The orchestrator only invokes this technique on pass 2, after the pass-1
  ensemble has produced a prior offset.
- :attr:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'n_stars_used', 'median_pos_err_px', 'residual_scatter_px'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.n_stars_used` — count of stars
  that survived the per-star quality gates. Consumed by the confidence formula and by the
  single-inlier confidence cap.
- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.median_pos_err_px` — median
  per-star Euclidean residual. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.residual_scatter_px` —
  per-axis weighted RMS of the per-star residuals about the fitted delta. Consumed by the
  confidence formula.

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav.navigate`:

1. Open a logged section. Filter the offered features down to
   ``usable_stars`` (predictable and not occluded by a body
   silhouette or ring annulus).
2. **No prior on the context.**  Return early with a spurious zero-confidence result via the
   private ``_fail`` helper, with ``reason='no_prior_offset_on_context'``.
3. Read the prior offset, the extfov image, and the per-image noise sigma off the context.
   Loop over the usable stars: for each, call
   ``local_centroid`` against the shifted prediction; drop
   stars with no peak above the detection threshold or whose centroid sits more than
   ``max_per_star_residual_px`` from the shifted prediction. Compute the per-star residuals
   and inverse-trace weights.
4. **Too few inliers.**  When the surviving inlier count is below ``min_inliers``, return a
   spurious zero-confidence result via ``_fail``.
5. Compute the inverse-variance weighted delta, the median per-star residual distance, and
   the per-axis weighted scatter about the delta. Build a
   :class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics`.
6. Read the search-window margin via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs` and apply the at-edge test
   to the absolute offset (prior + delta). Evaluate the confidence spec via
   :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination` and log the per-term
   breakdown.
7. **Single-inlier post-sigmoid cap.**  When exactly one inlier survived and the raw
   confidence exceeds ``single_inlier_confidence_cap``, the confidence is clipped to the cap
   and a log line records the clamp.
8. Build the (2, 2) translation covariance via the private ``_build_covariance`` helper:
   single-inlier returns the feature's CRLB; multi-inlier returns the diagonal of per-axis
   residual variances floored at the inverse total weight.
9. Result-shape branches on the rotation flag and the inlier count:

   - **Translation only**
     (:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` false). The
     reported covariance is the (2, 2) block;
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit, two or more inliers.**  Call the private ``_fit_rotation_3dof`` helper,
     which reconstructs the per-star detection and shifted-prediction positions from the
     residuals and runs ``similarity_transform_fit`` to extract the rotation and translation.
     The Procrustes refit can pull the absolute offset outside the search window even when
     the delta-only path was inside, so
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` is re-evaluated
     against the updated absolute offset and the rotation magnitude. The
     reported covariance is the (3, 3) block with the per-axis translation variances on the
     leading diagonal and the rotation variance on the bottom-right.
   - **Rotation fit, single inlier.**  No rotation evidence exists; the technique embeds the
     (2, 2) translation block in a (3, 3) covariance via
     :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`, sets
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` to ``0.0``,
     and reports
     :func:`~spindoctor.nav_technique.nav_technique.rotation_unobservable_sigma_rad` as the sigma.

10. Assemble and return the
    :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.
    :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` preserves the
    inlier set; the orchestrator's curator uses it to attribute per-star contributions at
    audit time. The orchestrator then stamps the result with
    :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.prior_source_techniques`
    (the pass-1 consensus techniques that seeded the prior) so the final ensemble excludes
    the refine's vote from corroboration and quorum against its own prior's sources.

Examples
========

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single bright star (Vega) in an otherwise empty field. The pass-1 ensemble's
    :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` recovers a
    1-star prior; :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` runs on
    pass 2, finds that single star at the shifted
    prediction, and reports a polished offset within sub-pixel of the prior. The result fires
    the single-inlier confidence cap (one inlier, no cross-check) and is reported with
    confidence at most 0.5. Operator-verified ground-truth offset is
    :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px.

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field, no body in FOV. When the pass-1
    :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` produces a
    multi-star prior, :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`
    re-centroids every predicted catalog star against the
    image and averages the per-star residuals. Multi-inlier inputs do not fire the
    single-inlier cap; the reported covariance is the diagonal of per-axis weighted residual
    variances rather than the CRLB floor. Operator-verified ground-truth offset is
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px.
