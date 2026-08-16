===========================
Body Limb Fit (BodyLimbNav)
===========================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` recovers a single translation
from one or more body limb polylines by aligning
each polyline against the image's edge-distance-transform. The technique consumes every
:data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` feature offered by the orchestrator,
weights each vertex by its prior-precision sigma, and runs a coarse normalised-cross-correlation
search followed by a Tukey-reweighted Levenberg-Marquardt refinement. The output is the
joint translation that minimises the summed weighted squared distance from the model polylines
to the image edges, plus a covariance derived from the M-estimator information matrix at
convergence.

Feasibility passes when at least one offered ``LIMB_ARC`` carries enough surviving vertices to
constrain a 2-D translation; feasibility fails when every offered ``LIMB_ARC`` has fewer than the
minimum-arc-length floor (a body whose limb is mostly hidden by the FOV boundary, occluded by
another body, or lost to shadow).

Theory
======

The technique belongs to the distance-transform family of polyline fitters: predicted polylines
are shifted as a rigid body until their vertices lie as close as possible to the nearest image
edge, where "as close as possible" is measured against a precomputed image-edge distance
transform. Shared algorithmic infrastructure handles the heavy lifting; this section describes
the cost function and conventions specific to the limb fit.

Cost function
-------------

The technique minimises

.. math::

    C(\Delta v, \Delta u, \theta) = \sum_{i} w_{i}(\Delta v, \Delta u, \theta) \,
        \mathrm{DT}\bigl[\,R(\theta)\,(x_{i} - x_{p}) + x_{p} + (\Delta v, \Delta u)\,\bigr]^{2}

where :math:`x_{i}` are the input vertices concatenated across every consumed ``LIMB_ARC``,
:math:`x_{p}` is the rotation pivot (the centroid of the concatenated vertices),
:math:`R(\theta)` is the in-plane rotation matrix, :math:`\mathrm{DT}` is the bilinearly
sampled image-edge distance transform, and the per-vertex weight :math:`w_{i}` is the product
of the prior precision :math:`1 / \sigma_{i}^{2}` (with :math:`\sigma_{i}` the per-vertex normal
sigma supplied by the body model) and a Tukey biweight evaluated at the scaled DT residual
:math:`\mathrm{DT}_{i} / \sigma_{i}`. When the per-instrument camera-rotation flag is off the
parameter vector collapses to :math:`(\Delta v, \Delta u)` and the rotation term is dropped.

Search strategy
---------------

The fit proceeds in two stages:

1. **Coarse integer search.**  The model polyline is rendered into a binary mask, the image
   edges are thresholded into a binary mask of their own (the truncated DT thresholded at
   half a pixel), and an integer-pixel cross-correlation is evaluated over a search window
   bracketing the per-instrument SPICE pointing-error envelope. The argmax of the correlation
   is the seed translation.
2. **Sub-pixel Levenberg-Marquardt refinement.**  Starting from the integer seed, the refiner
   evaluates the cost above, its parameter Jacobian (central differences against the bilinear
   DT), and an LM-damped normal-equation step. After each accepted step the Tukey weights are
   recomputed against the new residuals (iteratively reweighted least squares), so vertices
   that drifted onto an unrelated edge during refinement progressively lose weight.

Robustness
----------

The Tukey biweight is the redescender used by the LM reweighting; its asymptotic 95 % efficiency
constant is the documented default. Vertices whose model normal disagrees with the local image
gradient direction (a *polarity* mismatch — for a bright body on a dark background the gradient
points outward, into the silhouette's exterior) are assigned a near-infinite synthetic residual
on every iteration so the Tukey biweight zeroes their weight on the first reweighting; this
keeps the limb fit from latching onto the body's interior crater rims.

Restrictions and assumptions
----------------------------

- The orchestrator must supply both an image-edge distance transform and a per-pixel gradient
  vector image on the per-image
  :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`; in their absence the navigation aborts
  with a runtime error.
- The vertices and per-vertex normal sigmas must be physically meaningful — vertices with zero
  or negative sigma are rejected by the LM refiner.
- The fit assumes the body is bright against a dark background. The polarity rule is hard-coded
  by inverting the geometric outward normal so that the sign of the test matches the
  bright-on-dark gradient direction; this is correct for every supported instrument's body
  scenes.
- Multi-body inputs are fused into a single translation by concatenating their per-vertex
  arrays. The joint-translation parameterisation cannot represent disagreement between bodies
  about the offset; if SPICE relative geometry is wrong (a body misidentification, a stale SPK)
  the joint fit walks toward the higher-vertex-count body and the lower-vertex body's residuals
  appear as outliers the Tukey weight zeroes out.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence, scaled by the per-vertex Tukey weights, with the calibrated
model-error floor added in quadrature to the translation diagonal
(``add_model_error_floor(cov, model_error_floor_px)`` with the calibrated 2.61 px value —
the Tukey-weighted DT covariance under-reports the limb model error, which is dominated by
physics the fit cannot see: the limb-relief field, albedo texture, and non-Lambert shading
all displace the rendered limb from the smooth predicted silhouette. The floor restores
the campaign's 2-sigma coverage to the 2-D-Gaussian reference). The covariance
reflects the *shape* of the cost surface near the minimum and the surviving inlier population;
it does not capture systematic biases (e.g. an inflation of the per-vertex sigma due to
unmodelled crater roughness) and it does not capture model-side uncertainty in the SPICE
prediction itself (the search-window margin is what bounds that).

One such systematic is a **model-vs-image edge-localization bias floor of ~0.1 px** in the
recovered offset (median ~0.09-0.14 px over a dense sub-pixel sweep depending on the cross-axis
phase, up to ~0.25 px at the worst two-axis phase). It is *independent of SNR* -- it persists
on a clean, high-signal frame -- so it is not a noise effect. Its origin is the mismatch
between what the ``LIMB_ARC`` model predicts and what the image edge actually is: the model
predicts the *geometric silhouette*, but a body limb is a one-sided brightness transition
(sky outside, limb-darkened surface rising from ~0 at the silhouette inside), so after PSF
convolution the gradient-magnitude peak -- the feature any edge-fit locks onto -- sits a
fraction of the PSF width *inside* the silhouette. On a clean, zero-planted-offset render the
limb fit therefore recovers ~(0.06, 0.08) px instead of (0, 0), and a direct measurement of
the signed normal distance from the model silhouette to the gradient peak gives a median of
~+0.10 px inward.

The model polyline itself is not a quantization source: its vertices are probe-refined onto
the sub-pixel geometric silhouette (see the sub-pixel boundary refinement section of
:doc:`dev_guide_navigation_models_body`), so the polyline translates continuously with the
predicted pointing and the fit returns a planted rigid shift to within the DT optimizer's
own ~0.04 px floor (measured 0.01-0.06 px across planted shifts on real frames). The
integer-quantized distance transform (see :doc:`dev_guide_techniques_dt_fitting`) adds its
own sub-pixel-phase jitter on top, and its quantization + Tukey + trust-region machinery
*accidentally* pulls the fit ~0.1 px back toward truth, so the observed ~0.1 px floor is the
partial cancellation of a larger (~0.16 px) inward model-image offset against that pull. This
matters for the remedy: fitting the final sub-pixel offset against the *continuous* gradient
field (the ``gradient_ridge_refine`` stage in :doc:`dev_guide_techniques_dt_fitting`, wired but
held off via the ``gradient_ridge_refine`` tuning flag) converges precisely onto the gradient
peak, which removes the lucky cancellation and *sharpens* the bias rather than removing it --
clean planted-(0,0) recovery worsens from ~0.10 px to ~0.17 px. The ring edge, being a
symmetric transition whose gradient peak coincides with the geometric edge, does not have this
offset and is already at its floor (~0.016 px) without the refine. The genuine remedy is a
*model-side* fix that predicts the limb at the gradient-peak location (limb-darkening- and
PSF-aware); that fix has not yet been made. Until then the limb fit is the least precise of
the point-feature techniques on a well-resolved body, though it stays well inside the
navigability bound. When the converged offset
sits within a small tolerance of any axis bound of the search window, or when the rotation
parameter is at the configured fraction of its cap, the result is flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
confidence formula's hard-zero gate forces confidence to zero. The spurious tests gate on
the Tukey-weighted DT residual RMS, the unweighted (raw) DT residual RMS against the same
threshold (so a fit where Tukey rejects a wholly mis-aligned arc cannot pass on its
collapsed weighted RMS), the degenerate flag, the inlier count and fraction, the LM
displacement from the coarse seed, and a coarse-seed mis-lock test; when any of these
fails, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` and similarly
forced to zero.

The mis-lock test targets a distinct low-phase failure. Near zero phase (below ~15 deg) the
lit arc spans almost the whole silhouette and the limb-darkening roll-off is nearly
symmetric, so the across-limb gradient that constrains the fit is weakest exactly where the
geometry offers the least leverage. In that regime the polarity-weighted coarse search can
find a false overlap peak several pixels from the true limb and seed the LM in the wrong
basin. The trust region then holds the LM within ~1 px of that wrong seed, so the fit cannot
reach the true limb and instead sits pinned against the trust-region boundary, exiting at the
iteration cap without meeting the step tolerance. The residuals stay low and every vertex
stays an inlier, so the RMS, inlier, and displacement guards all pass and the fit would
otherwise report a confident multi-pixel offset. The mis-lock test catches it by the joint
signature: a fit that BOTH failed to converge AND ended at least
``spurious_unconverged_trust_boundary_fraction`` of the trust region away from the coarse
seed is a wrong-basin lock and is flagged spurious. A healthy sub-pixel refinement lands
within ~0.71 px of its integer seed and converges, so it clears the test on both counts; a
converged fit that walked the full trust region to a good limb (common just outside the
low-phase band) also clears it, because convergence is the discriminator.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyLimbNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``min_arc_vertices`` — float, default ``30.0``. Minimum surviving vertex count per
  ``LIMB_ARC`` for feasibility. Arcs with fewer vertices do not constrain a 2-D translation
  enough to be worth the LM iteration.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many limb-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``3.0`` px. Floor of the spurious-detection
  threshold; the threshold is the larger of the floor and the per-feature sigma multiple.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``spurious_min_inlier_fraction`` — float, default ``0.20`` (dimensionless). Below this
  inlier fraction the LM has almost certainly walked off the true limb onto internal-body
  features (crater rims, terminator, surface boundaries); the result is flagged spurious.
  A clean limb fit on a fully-visible body retains 50 %+ of vertices as inliers, and image
  dropouts and cosmic rays account for at most a few percent more rejection, so below 20 %
  is a strong mis-convergence signature.
- ``spurious_max_lm_displacement_px`` — float, default ``4.0`` px. If the LM moves more
  than this many pixels from the integer coarse-NCC seed, flag spurious. Defensive: with
  the trust region below the LM cannot leave the coarse basin, so this guard normally
  never fires; it catches any future regression that bypasses the trust region. Set
  slightly larger than ``lm_trust_region_px``.
- ``spurious_unconverged_trust_boundary_fraction`` — float, default ``0.9`` (dimensionless).
  Coarse-seed mis-lock gate. A fit that BOTH failed to converge AND ended at least this
  fraction of the trust region away from the integer coarse seed was pinned against the
  trust-region boundary trying to escape a wrong-basin seed; it is flagged spurious rather
  than allowed to emit a confident multi-pixel offset. This is the low-phase
  (below ~15 deg) under-conditioned failure where a false overlap peak seeds the wrong basin.
  A healthy sub-pixel fit sits within ~0.71 px of its integer seed and converges, so it
  clears the gate. Convergence is the decisive condition: a fit that legitimately walks to
  the trust-region boundary but still converges is not flagged, because the gate requires
  both a failure to converge and a boundary-pinned displacement.
- ``lm_trust_region_px`` — float, default ``1.0`` px. Maximum LM displacement from the
  integer coarse seed; the LM rejects any trial step that would land outside this
  radius without committing. The coarse search returns the integer-pixel polarity-weighted
  match maximum, so the true sub-pixel optimum is within ~0.71 px of the seed — 1.0 px gives
  legitimate sub-pixel refinement headroom while denying the LM the runway to reach a
  2-4 px-away spurious minimum.
- ``lm_tikhonov_alpha`` — float, default ``0.0`` (dimensionless). Tikhonov anchor strength
  toward the coarse-NCC seed. Zero by default: an anchor strong enough to prevent
  multi-pixel walks on textured bodies also prevents legitimate sub-pixel refinement on
  clean ones, and the trust region is the harder bound.
- ``gradient_ridge_refine`` — int flag, default ``0`` (OFF). Final continuous
  gradient-ridge sub-pixel refinement after the DT LM converges. Held off for the limb:
  the dominant limb error is the model-vs-image edge offset discussed above, not DT
  quantization, and refining onto the gradient peak sharpens that offset. See
  :doc:`dev_guide_techniques_dt_fitting`.
- ``model_error_floor_px`` — float, default ``2.61`` px. Calibrated model-error floor
  added in quadrature to the covariance's translation diagonal (see "Sources of
  uncertainty"). The 2026-07-18 calibration campaign measured 2-sigma coverage 0.68 at
  a 0.92 px floor and 0.86 at this value. The floor sits above the medium tier's 2.0 px
  sigma cap, so a limb-only fix surfaces as a low-tier result — the honest price of a
  limb whose position error is set by unmodeled terrain.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the bilinear-DT half-cell width.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the converged rotation magnitude trips :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it
  crosses this fraction of the per-image :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

The thirteen keys above are global; the per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override any of them. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` rather than from this block.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared sigmoid
combination, see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic and
:doc:`dev_guide_techniques` for the family-level overview of confidence. The formula spec is
``techniques.BodyLimbNav`` in the same YAML file and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` plus the :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` flags carried on the result.

- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction` —
  alpha = 1.068, offset = 0.0, divisor = 1.0, no cap.
  Fraction of the polyline (weighted by surviving vertex count across consumed ``LIMB_ARC``
  features) whose vertices were not pre-rejected by the model-side shadow / FOV gates. One
  means every offered vertex is usable. The sim limb reports its honest visible-arc
  fraction (surviving polyline over the unclipped whole-body silhouette boundary, net of
  frame clipping and sibling-body occlusion), so the partial-arc penalty is fitted rather
  than frozen at a design prior (calibration campaign raw p5/p50/p95 = 0.33/0.76/1.0).
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px` — alpha = -1.303,
  offset = 0.0, divisor = 1.0, no cap. Final root-mean-square DT residual after LM
  convergence; smaller is sharper.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px` — alpha = 0.776,
  offset = 0.0, divisor = 440.0, cap at 1.0. Total surviving polyline length in pixels,
  capped after normalisation. More polyline earns confidence up to a 440-pixel saturation
  point (calibration campaign raw p5/p50/p95 = 150/280/433).

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces the confidence to zero before
the sigmoid is evaluated. The constant baseline is :math:`\alpha_{0} = 0.132`. No post-sigmoid
``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_body_limb.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` and its private aggregation /
  polyline-mask helpers.
- ``src/spindoctor/nav_technique/dt_fitting.py`` — the shared coarse-NCC and LM-refinement helpers
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` — the per-image gradient / DT derivatives
  attached to :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`.
- ``src/spindoctor/nav_technique/confidence.py`` — the shared sigmoid-combination formula evaluator.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so the orchestrator's
``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.name` — ``'BodyLimbNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.accepts_feature_types` —
  ``frozenset({LIMB_ARC})``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.requires_prior` — ``False``.
  The technique runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.confidence_attributes` — the
  names of every attribute the spec is allowed to read, validated at config-load time:
  ``{'at_edge', 'spurious', 'visible_limb_arc_fraction', 'visible_arc_px', 'dt_fit_rms_px',
  'lm_iterations', 'tukey_inlier_count'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` is the typed dataclass attached to
the result. Every field is named in the call path or in the confidence formula above:

- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction` —
  vertex-weighted average of the per-feature visible-arc fraction; consumed by the confidence
  formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px` — total
  surviving polyline arc length in pixels; consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px` — weighted RMS DT
  residual at the converged pose; consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.lm_iterations` — number of LM
  iterations actually performed.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.tukey_inlier_count` — number of
  vertices that retained a strictly positive Tukey weight at the final estimate; consumed by
  the spurious-detection gate.

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate`:

1. Open a logged section. Fail fast if either
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing from
   the context — the orchestrator's per-image setup
   is responsible for populating both via
   :func:`~spindoctor.nav_orchestrator.image_derivatives.compute_all_image_derivatives`; see
   :doc:`dev_guide_techniques_dt_fitting` for the surface those products expose.
2. Filter the offered features down to ``LIMB_ARC`` polylines whose surviving vertex count is at
   least ``min_arc_vertices``, then concatenate the per-feature vertex / normal / sigma arrays via
   the private ``_aggregate_limb_features`` helper. The geometric outward normals are negated
   in this step so that the polarity test in the LM refiner expects the image gradient to
   point *into* the body silhouette.
3. Threshold the edge DT into an edge mask and pull the search-window margin off the
   observation via :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`. Run
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored` on the edge mask,
   the gradient image, and the polyline vertices / polarity normals to obtain an integer seed
   offset. The polarity weighting keeps a cluttered scene's competing edge population (a ring
   behind the disc, the terminator, a second moon) from out-scoring the true limb arc and
   seeding the LM in the wrong basin.
4. Decide whether to fit camera rotation by reading
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`. When rotation
   is fit, the rotation pivot is set to the
   centroid of the concatenated vertices and the pivot-to-image-centre distance is computed
   via :func:`~spindoctor.nav_technique.nav_technique.rotation_pivot_distance_px` for the convergence
   test.
5. Call :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` with the polyline,
   per-vertex sigmas, the edge DT, the gradient image, the integer seed, and the rotation
   options. The refiner returns a converged
   :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult`.
6. Compute the result-shape branches:

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
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` is
     the square root of its diagonal. An unexpected covariance shape raises
     :exc:`RuntimeError` — a programmer error in the refiner contract is not silently
     absorbed.

7. Apply the at-edge tests against both the translation axis bounds and the rotation cap, and
   the spurious tests against the final RMS, the inlier count, and the inlier fraction.
8. Build a :class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`, log
   the per-term breakdown via
   :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field on the
result preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can attribute
each contribution at audit time.

Examples
========

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right, about 22 % of the disc off-frame. The body model emits a
    ``LIMB_ARC`` feature; the technique consumes it and converges to
    :math:`(\Delta v, \Delta u) = (12.06, 30.53)` px against an operator-verified ground truth
    of ``(11.0, 29.5)`` px. Feasibility passes (one ``LIMB_ARC``, surviving vertex count well
    above ``min_arc_vertices``), neither :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` nor :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` fires, and the technique
    becomes the orchestrator's primary on this image.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees. Two
    ``LIMB_ARC`` features are offered;
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` fuses them into a joint
    translation and
    converges to :math:`(\Delta v, \Delta u) = (7.00, -18.00)` px against an operator-verified
    ground truth of ``(7.03, -18.42)`` px. The technique reports a reduced confidence — the
    sigmoid is drawn down by the modest visible-arc length of each limb on its own — but the
    fit is geometrically correct.

``body_full_fov`` (Cassini ISS NAC, image ``N1572105349_1``)
    Dione fills the FOV, predicted disc diameter approximately 155 px. The body model emits a
    ``LIMB_ARC`` feature, but the upstream feature-reliability gate drops it before
    :meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.is_feasible` is consulted
    (the textbook full-disc, fully-lit limb saturates
    the model-side reliability formula's incidence-factor penalty). The technique therefore
    reports zero consumed features and skips
    :meth:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate` for this scene.
