================================================
Body Terminator Fit (BodyTerminatorNav)
================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` recovers a single
translation from one or more body terminator polylines by aligning each polyline against the
image's edge-distance-transform. The technique consumes every
:data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` feature offered by the
orchestrator, weights every vertex of a given body uniformly by an inverse-variance derived
from that body's mean per-vertex normal sigma, and runs the same polarity-weighted coarse
acquisition plus Tukey-reweighted Levenberg-Marquardt refinement that
:doc:`dev_guide_techniques_dt_fitting` describes for the limb fit. The output is the joint
translation that minimizes the summed weighted squared distance from the model polylines to
the image edges, plus a covariance derived from the M-estimator information matrix at
convergence.

Feasibility passes when at least one offered ``TERMINATOR_ARC`` carries enough surviving vertices
to constrain a 2-D translation; feasibility fails when every offered ``TERMINATOR_ARC`` has fewer
than the minimum-arc-length floor (a body whose phase angle is too low for a usable
terminator, or whose terminator is mostly hidden by the FOV boundary).

Theory
======

The technique belongs to the distance-transform family of polyline fitters: predicted
polylines are shifted as a rigid body until their vertices lie as close as possible to the
nearest image edge, where "as close as possible" is measured against a precomputed image-edge
distance transform. The shared algorithmic infrastructure handles the heavy lifting; this
section describes the cost function and conventions specific to the terminator fit.

Cost function
-------------

The technique minimizes

.. math::

    C(\Delta v, \Delta u, \theta) = \sum_{i} w_{i}(\Delta v, \Delta u, \theta) \,
        \mathrm{DT}\bigl[\,R(\theta)\,(x_{i} - x_{p}) + x_{p} + (\Delta v, \Delta u)\,\bigr]^{2}

where :math:`x_{i}` are the input vertices concatenated across every consumed ``TERMINATOR_ARC``,
:math:`x_{p}` is the rotation pivot (the centroid of the concatenated vertices),
:math:`R(\theta)` is the in-plane rotation matrix, :math:`\mathrm{DT}` is the bilinearly
sampled image-edge distance transform, and the per-vertex weight :math:`w_{i}` is the product
of a per-body uniform precision (the inverse mean variance across the body's terminator
polyline) and a Tukey biweight evaluated at the scaled DT residual. When the per-instrument
camera-rotation flag is off the parameter vector collapses to :math:`(\Delta v, \Delta u)`
and the rotation term is dropped.

Per-body uniform weighting
--------------------------

Unlike the limb fit, every vertex of a given body's terminator shares one inverse-variance
weight derived from that body's mean per-vertex normal sigma. Cross-body weighting reflects
the structural fact that low-albedo bodies provide tighter terminators than high-albedo ones
(an albedo gradient blurs the apparent terminator more than a smooth surface would), so the
mean per-vertex sigma already captures the per-body terminator quality and a single per-body
weight avoids over-weighting a long terminator from a low-confidence body.

Search strategy
---------------

The fit proceeds in the same two stages as the limb fit: a coarse integer cross-correlation
between the rendered terminator polyline mask and the thresholded image edge mask, then a
sub-pixel Levenberg-Marquardt refinement against the bilinearly sampled DT. See
:doc:`dev_guide_techniques_dt_fitting` for the per-iteration mechanics.

Robustness
----------

The Tukey biweight redescender drops vertices whose scaled residuals exceed the Holland-Welsch
cutoff. Polarity filtering — comparing the model normal at each vertex against the local
image gradient direction — is enabled with the model normals oriented so the gradient should
point from the dark side toward the lit side at the terminator. Polarity-rejected vertices
contribute a synthetic near-infinite residual on every iteration so the Tukey biweight zeroes
their weight on the first reweighting; this keeps the terminator fit from latching onto the
opposite-polarity edge of a neighboring body's limb.

Restrictions and assumptions
----------------------------

- The orchestrator must supply both an image-edge distance transform and a per-pixel gradient
  vector image on the per-image
  :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`; in their absence the navigation
  aborts with a :exc:`RuntimeError`.
- The terminator is assumed to be photometrically distinguishable from the limb. At phase
  angles below approximately 3 degrees the terminator is too close to the limb to be
  usefully separated; the model-side
  :data:`~spindoctor.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR` gate suppresses emission
  in that regime so the technique never sees those features.
- The albedo penalty per body is supplied by the model side via the per-body
  :class:`~spindoctor.nav_model.body_shape.BodyShape`; the technique reads it off the
  per-feature reliability breakdown
  (:attr:`~spindoctor.feature.feature.NavFeature.reliability_reasons`'s ``albedo_penalty``,
  not the feature flags) and does not re-derive it.
- Multi-body inputs are fused into a single translation by concatenating their per-vertex
  arrays. The joint-translation parameterization cannot represent disagreement between
  bodies about the offset; if SPICE relative geometry is wrong the joint fit walks toward the
  higher-vertex-count body and the lower-vertex body's residuals appear as outliers the
  Tukey weight zeroes out.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence, scaled by the per-vertex Tukey weights, with the calibrated
``model_error_floor_px`` (4.32 px) added in quadrature to the translation diagonal.
It does not capture
systematic biases (an under-modeled per-body albedo gradient propagates straight into the
covariance) and it does not capture model-side uncertainty in the SPICE prediction itself.
When the converged offset sits within a small tolerance of any axis bound of the search
window, or when the rotation parameter is at the configured fraction of its cap, the result
is flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
confidence formula's hard-zero gate forces confidence to zero. The spurious tests gate on
the Tukey-weighted DT residual RMS, the unweighted (raw) DT residual RMS against the same
threshold (so a fit where Tukey rejects a wholly mis-aligned arc cannot pass on its
collapsed weighted RMS), the degenerate flag, the inlier count and fraction, and the LM
displacement from the coarse seed; when any of these fails, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` and similarly forced
to zero.

A fit that passes every residual metric can still be a mis-convergence onto a rival
alignment -- a crater shadow, an albedo boundary, or a second body's edge -- because a
local DT minimum fits the polyline as cleanly as the true one. The *basin second-opinion*
(:func:`~spindoctor.nav_technique.dt_fitting.find_secondary_dt_minimum`) closes this hole:
after LM convergence the mean per-vertex DT cost is sampled at every integer shift in the
coarse search window, and when the best shift farther than ``basin_exclude_radius_px``
from the converged offset scores below ``basin_cost_ratio_threshold`` times the converged
cost, the DT surface is not unimodal and the result is flagged spurious. The rival's
distance and cost ratio are recorded on the diagnostics
(:attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_distance_px` /
:attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_cost_ratio`).
This matters most for
the lone-primary case: the ensemble's fallback-tier drop protects against a mis-converged
terminator only when a limb or disc result exists on the same body to supersede it.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyTerminatorNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``min_arc_vertices`` — float, default ``30.0``. Minimum surviving vertex count per
  ``TERMINATOR_ARC`` for feasibility. Arcs with fewer vertices do not constrain a 2-D
  translation enough to be worth the LM iteration.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many terminator-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``4.0`` px. Floor of the spurious-detection
  threshold; terminators are softer than limbs so the floor sits one pixel wider than the
  limb fit's.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``spurious_min_inlier_fraction`` — float, default ``0.20`` (dimensionless). Below this
  inlier fraction the LM has almost certainly walked off the true terminator onto crater
  shadows or surface boundaries; the result is flagged spurious. Same motivation as the
  limb fit's identically-named knob.
- ``spurious_max_lm_displacement_px`` — float, default ``4.0`` px. If the LM moves more
  than this many pixels from the integer coarse-NCC seed, flag spurious. Defensive: with
  the trust region below the LM cannot leave the coarse basin; the guard catches any
  future regression that bypasses the trust region.
- ``lm_trust_region_px`` — float, default ``1.0`` px. Maximum LM displacement from the
  integer coarse-NCC seed; the LM rejects any trial step that would land outside this
  radius. See :doc:`dev_guide_techniques_body_limb` for the rationale.
- ``lm_tikhonov_alpha`` — float, default ``0.0`` (dimensionless). Tikhonov anchor strength
  toward the coarse-NCC seed; 0 disables the anchor (the trust region is the harder
  bound).
- ``model_error_floor_px`` — float, default ``4.32`` px. Model-error floor added in
  quadrature to the covariance's translation diagonal, solved on the technique's own
  calibration-campaign rows: the DT fit's reported sigma (~1.2 px) sits far below its
  actual recovery error (median 5.2 px), so the floor carries nearly all of the honest
  uncertainty. The campaign measured 2-sigma coverage 0.04 before flooring and 0.87
  after. A terminator fix is therefore always a low-tier result, consistent with its
  fallback-only ensemble role.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the
  bilinear-DT half-cell width.
- ``basin_cost_ratio_threshold`` — float, default ``1.2`` (dimensionless). A competing
  DT basin whose mean per-vertex cost falls below this multiple of the converged cost
  marks the fit spurious; ``0`` disables the basin second-opinion.
- ``basin_exclude_radius_px`` — float, default ``5.0`` px. Shifts within this distance of
  the converged offset belong to the converged basin and are never counted as rivals.
- ``basin_cost_epsilon_px`` — float, default ``0.25`` px. Floor on the converged cost in
  the ratio's denominator, so a near-perfect fit is not vetoed by numerical noise; a rival
  must genuinely match a near-zero converged cost to fire the gate.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses this
  fraction of the per-image
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

The keys above are global; the per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override any of them. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` rather than from this
block.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic.
The formula spec is ``techniques.BodyTerminatorNav`` in the same YAML file and consumes
attributes off :class:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics` plus the
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` flags.

The spec is fitted against the simulated-scene calibration campaign, whose terminator rows
are single-class: zero of 116 usable rows recovered within 1 px (median error 5.2 px,
p5 2.6 px). On randomized crescents the DT fit settles several pixels from the planted
truth, systematically toward the bright limb-side gradient, and no diagnostic separates a
closer result from a farther one, so the L2-regularized fit converges to an honest low
plateau (confidence ~0.03-0.05): a terminator fix cannot pass the ensemble's 0.35 gate on
its own or outweigh a calibrated technique. The arc-fraction, arc-length, and phase terms
are sign-bounded to zero by the single-class rows (no success mass to reward); they stay
wired for a future real-anchored calibration.

- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_terminator_arc_fraction`
  — alpha = 0.0, offset = 0.0, divisor = 1.0, no cap. Vertex-weighted average of the
  per-feature visible-arc fraction across consumed ``TERMINATOR_ARC`` features.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.dt_fit_rms_px` —
  alpha = -1.533, offset = 0.0, divisor = 1.0, no cap. Final root-mean-square DT residual
  after LM convergence; smaller is sharper.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_arc_px` —
  alpha = 0.0, offset = 0.0, divisor = 100.0, cap at 1.0. Total surviving polyline length in
  pixels, capped after normalization.
- ``mean_phase_angle_factor`` — alpha = 0.0, offset = 0.0, divisor = 1.0, no cap. Mean of
  :math:`\sin(\phi)` across consumed terminators (read off the per-feature
  :class:`~spindoctor.feature.flags.TerminatorArcFlags`).
- ``mean_albedo_penalty`` — alpha = -0.216, offset = 0.0, divisor = 1.0, no cap. Mean of the
  per-body albedo penalty (read off the per-feature reliability breakdown). High-albedo
  bodies pull confidence down; uniform low-albedo bodies leave it unchanged.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = -1.93`. No post-sigmoid ``hard_cap`` is applied.

The confidence trust label stays provisional even though the technique is sim-fitted: the
realism match that anchors the calibration basis computed its limb-rise figure of merit on
the limb side only, so no terminator-side rise-width divergence verdict exists yet.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_body_terminator.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` and its private
  aggregation / polyline-mask helpers.
- ``src/spindoctor/nav_technique/dt_fitting.py`` — the shared coarse-NCC and LM-refinement helpers
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` — the per-image gradient / DT derivatives
  attached to :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`; documented at
  :doc:`dev_guide_techniques_image_derivatives`.
- ``src/spindoctor/nav_technique/confidence.py`` — the shared sigmoid-combination formula evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class
:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so the orchestrator's ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.name` —
  ``'BodyTerminatorNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.accepts_feature_types`
  — ``frozenset({TERMINATOR_ARC})``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.requires_prior` —
  ``False``. Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.tier` —
  ``'fallback'``. The terminator is a photometric feature modulated by phase, albedo, and
  local topography, and its failure modes (DT-fit locking onto crater shadows or other
  local minima) have no per-technique signal that admits them; when a non-spurious primary
  fit (limb or disc) is available for the same body the ensemble drops the terminator
  result rather than risk a clean-looking mis-convergence overriding the geometric
  techniques.
- :attr:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'visible_terminator_arc_fraction', 'visible_arc_px',
  'dt_fit_rms_px', 'lm_iterations', 'tukey_inlier_count', 'mean_phase_angle_factor',
  'mean_albedo_penalty'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_terminator_arc_fraction`
  — vertex-weighted average of the per-feature visible-arc fraction across consumed
  ``TERMINATOR_ARC`` features. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_arc_px` — total
  surviving polyline arc length in pixels. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.dt_fit_rms_px` — weighted
  RMS DT residual at the converged pose. Consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.lm_iterations` — number of
  LM iterations actually performed.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.tukey_inlier_count` —
  number of vertices that retained a strictly positive Tukey weight at the final estimate.
  Consumed by the spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.mean_phase_angle_factor`
  — vertex-weighted mean ``sin(phase)`` factor across the consumed features. Consumed by
  the confidence formula and recorded so the calibration fit sees the same inputs.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.mean_albedo_penalty` —
  vertex-weighted mean catalog albedo-variation penalty across the consumed features.
  Consumed by the confidence formula; recorded alongside it.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_distance_px`
  — distance from the converged offset to the best competing DT-cost basin in the search
  window; ``None`` when the scan did not run or found no eligible shift.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_cost_ratio`
  — the competing basin's mean per-vertex DT cost divided by the (epsilon-guarded)
  converged cost; values below ``basin_cost_ratio_threshold`` mark the fit spurious.
  ``None`` when not measured.

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.navigate`:

1. Open a logged section. Fail fast (:exc:`RuntimeError`) if either
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing.
2. Filter the offered features down to ``TERMINATOR_ARC`` polylines whose surviving vertex count
   is at least ``min_arc_vertices``, then concatenate the per-feature vertex arrays via the
   private aggregation helper. The helper also computes per-body sigma scalars (mean of the
   per-vertex normal sigmas) and the per-body phase / albedo flags.
3. Threshold the edge DT into an edge mask and pull the search-window margin off the
   observation via :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`. Run
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored` on the edge mask,
   the gradient image, and the polyline vertices / polarity normals to obtain an integer seed
   offset. The polarity weighting keeps the lit disc's limb, rings, or a neighboring body from
   out-scoring the true terminator arc and mis-seeding the LM.
4. Decide whether to fit camera rotation by reading
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`. When rotation
   is fit, the rotation pivot is set to the centroid of the concatenated vertices and the
   pivot-to-image-center distance is computed via
   :func:`~spindoctor.nav_technique.nav_technique.rotation_pivot_distance_px`.
5. Call :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` with the polyline,
   per-vertex sigmas, the edge DT, the gradient image, the integer seed, and the rotation
   options.
6. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block. Any non-(2, 2) covariance returned by the refiner is logged
     at WARNING and truncated.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (3, 3) translation + rotation information matrix.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is the
     converged angle and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` is the
     square root of its diagonal. An unexpected covariance shape raises
     :exc:`RuntimeError`.

7. Apply the at-edge tests against translation axis bounds and the rotation cap, plus the
   spurious tests: the degenerate flag, the Tukey-weighted RMS and the unweighted (raw)
   RMS — both against the threshold ``max(spurious_dt_floor_px,
   spurious_dt_rms_factor * sigma_min)`` where ``sigma_min`` is the smallest per-body
   sigma — the inlier count, the inlier fraction, and the LM displacement from the coarse
   seed. A fit that passes all of those still runs the basin second-opinion scan, which
   marks it spurious when a rival DT basin's cost ratio falls below
   ``basin_cost_ratio_threshold``.
8. Build a :class:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the per-term breakdown via
   :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``high_phase_terminator`` (Cassini ISS NAC, image ``N1597846115_2``)
    A high-phase terminator arc with no other features in the FOV.
    :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` consumes the
    single ``TERMINATOR_ARC`` feature offered by the body model and converges within ~1 px of
    the operator-verified offset :math:`(\Delta v, \Delta u) = (5.19, 1.30)` px.

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right with about 22 % of the disc off-frame. The body model
    emits a ``TERMINATOR_ARC`` feature alongside ``LIMB_ARC`` and ``BODY_DISC``; on this scene the
    terminator fit reports zero Tukey inliers (the LM does not iterate) and the spurious
    gate forces the confidence to zero. The
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` fit is the technique that
    actually drives the orchestrator's primary on this image; see
    :doc:`dev_guide_techniques_body_limb` for that walk-through.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees.
    Two ``TERMINATOR_ARC`` features are offered;
    :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` fuses them
    into a joint translation. On this scene the multi-body crescent geometry produces a
    coarse-NCC seed at a wrong local minimum (~31 px off-axis); the LM converges near that
    seed and reports sub-pixel RMS with most vertices as inliers — a clean-looking
    mis-convergence that no residual metric flags, exactly the failure mode the calibration
    campaign measured. The calibrated formula holds the result to its low plateau, and the
    ensemble's fallback-tier supersession drops it in favor of the limb and disc fits,
    which agree around the operator-verified offset
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px.
