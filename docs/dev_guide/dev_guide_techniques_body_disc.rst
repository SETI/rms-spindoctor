==========================================================
Body Disc Correlate (BodyDiscCorrelateNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` recovers a single
translation by full-disc normalized cross-correlation against a composite template fused
from every offered ``BODY_DISC`` feature. Per-body templates are Z-buffer painted into a single
postage stamp (the closer body's pixels overwrite the farther body's), the result is run
through the shared pyramid-NCC machinery in :mod:`spindoctor.support.correlate`, and the chosen
peak is returned with a Cramer-Rao-lower-bound covariance derived from the local correlation
curvature. Multi-body composites improve the peak's signal-to-noise as roughly
:math:`\sqrt{N}` for ``N`` bodies and remove the "swap moon assignments" mode-failure that
plagues per-body solo correlation.

Feasibility passes when at least one offered ``BODY_DISC`` feature carries a template payload;
feasibility fails when no offered feature has one (a body model that emitted only
``LIMB_ARC`` / ``BODY_BLOB`` without an accompanying disc).

Theory
======

The technique fits a per-image translation by maximizing the normalized cross-correlation
between the composite template and the observed image. It supports an optional in-plane
rotation parameter that runs an outer NCC pyramid over a rotation schedule.

Composite template construction
-------------------------------

When more than one body emits a ``BODY_DISC`` feature, the per-body templates are fused into a
single composite by Z-buffer paint: at each pixel covered by more than one body's bounding
box, the closer body's template value (and the closer body's mask True) overwrite the
farther body's, so an in-front body occludes an in-behind body in the composite. The fused
template carries one combined bounding box, one fused brightness image, and one fused mask;
the orchestrator's
:func:`~spindoctor.feature.composition.compose_template_features` helper does the work.

Cost function
-------------

Let :math:`T` be the composite template image and :math:`M` the composite template mask.
Let :math:`I` be the observed image (or a mode-selected gradient of it; see below). The
technique maximises the normalized cross-correlation

.. math::

    \mathrm{NCC}(\Delta v, \Delta u) =
        \frac{\langle T, I_{\Delta v, \Delta u} \rangle_{M}}
             {\sqrt{\langle T, T \rangle_{M} \cdot \langle I_{\Delta v, \Delta u},
                                          I_{\Delta v, \Delta u} \rangle_{M}}}

over the integer offsets :math:`(\Delta v, \Delta u)` in the per-instrument search window,
where :math:`\langle \cdot, \cdot \rangle_{M}` denotes the masked inner product (only pixels
where :math:`M = \mathrm{True}` contribute). Sub-pixel refinement comes from a localized
upsampled-DFT (Guizar-Sicairos) of the cross-power spectrum around the integer peak,
reaching a 1/128 px resolution. The refinement is evaluated on the **raw-intensity**
surfaces even when the integer peak was chosen on the gradient surfaces, so a gradient-mode
peak does not carry the gradient-magnitude rectification bias (the magnitude rectifies the
signal, making the cross-power peak non-smooth at its apex) into the reported offset; the
template gradient structure provides the matched-filter statistical covariance term (see
"Sources of uncertainty").

The refinement window spans half a pixel either side of the integer NCC peak (exactly so
for the even upsample factors in use; an odd factor would offset one boundary by a single
upsampled step). When
the upsampled-DFT argmax lands on the window boundary, the cross-power surface's own peak
lies farther than half a pixel from the NCC peak -- the two surfaces disagree, which happens
when the template constrains an axis poorly (a disc that overflows the FOV constrains the
overflow axis only weakly) and the raw surface's scale bias takes over. Following the raw
surface farther is unsafe (its peak can sit pixels from the true alignment), and reporting
the boundary value would pin the offset at exactly half a pixel from the integer peak, an
answer that does not move when the true alignment does. Each saturated axis instead falls
back to the vertex of the quadratic through the three NCC samples around the peak -- the
surface that selected the peak -- which is bounded to the same half pixel and consistent
with the peak choice. Interior (non-saturated) refinements are untouched.

Sub-pixel refinement band-limit (``refine_lowpass_sigma_px``)
-------------------------------------------------------------

Before the cross-power spectrum is formed, the full-resolution refinement low-passes both
surfaces -- the image and the (already mask-multiplied) template -- with a Gaussian of
``refine_lowpass_sigma_px`` (default 1.0 px).

**Why.** The template is an oversampled but otherwise *sharp* Lambert silhouette, and a real
or simulated body has an equally sharp (PSF-aside) limb whose edge profile does not match the
template's. Cross-correlating two sharp surfaces with differing edge profiles aliases the
cross-power peak: the high-frequency edge content beats against the sampling grid and shifts
the peak by an amount that depends on the sub-pixel phase. The result is an odd S-curve in the
recovered offset -- zero at integer and half-pixel offsets, growing to a peak near the
quarter-pixel -- which is *per-axis separable* (a pure-``v`` offset biases ``v`` only). The
matched low-pass removes the high-frequency mismatch that drives the aliasing, so the peak
lands on the true shift. The localized upsampled DFT still resolves the smoothed peak to
1/128 px, so the band-limit costs no usable resolution.

**What happens if you don't (set it to 0).** The disc carries a sub-pixel-phase-dependent
residual of roughly ``±0.03`` px on a single axis, rising to ``~0.1`` px at the worst
two-axis phase, *independent of SNR* (it is a deterministic correlation artifact, not noise).
Because the simulator's body and the navigator's template are both sharp, a round-number
planted offset (``0``, ``0.5``) lands on the bias null and hides it entirely -- which is why
the per-technique characterization plants an off-grid offset. With the default 1.0 px
low-pass the residual drops to ``~0.01`` px across every sub-pixel phase. The band-limit is
applied **only** to the final full-resolution refine; the coarse pyramid levels keep their
sharp surfaces so integer-peak selection is unaffected. It must low-pass the *mask-multiplied*
template, not the bare template -- low-passing before the mask multiply lets the hard mask
edge re-sharpen the model and makes the bias worse, not better.

**Alternative considered and rejected: a render-time PSF.** Since the bias comes from sharp
edges, an obvious idea is to convolve the simulated body (and, to match, the template) with
the instrument point-spread function so the edges are realistically soft. This does not work
as a fix. Blurring only the rendered image at the camera's own ~0.54 px sigma leaves it
mismatched against the still-sharp template -- a soft image against a sharp template is a
worse mismatch than two matched sharp surfaces -- and it destabilizes the integer-peak choice
at some phases; a larger blur is monotonically worse. It is also the wrong layer: the bias is
in the navigator's correlation, which applies to real PSF-blurred images too, so it must be
fixed in the correlator (the band-limit above) rather than in the simulator. The band-limit
removes the bias for both simulated and real images and changes neither the rendered image nor
the template.

Mode selection (auto / raw / gradient)
--------------------------------------

The shared NCC pyramid evaluates the correlation in two modes — raw image vs. raw template,
and gradient image vs. gradient template — and ``auto`` mode runs both and picks whichever
peak scores higher quality. Raw mode wins on smooth Lambert-shaded discs that fill the FOV
where the per-pixel intensity carries unique-alignment signal; gradient mode wins when only
the limb edge carries useful signal (a bright disc on a bright background, an over-exposed
or saturated body).

Search strategy
---------------

The technique runs the shared pyramid-NCC entry point
(:func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`). At each pyramid level the
NCC is evaluated coarse-to-fine, the top ``k`` peaks are kept, and consistency is measured
between levels — a peak that drifts more than a documented threshold across levels is
flagged spurious. The per-image quality metric is the peak-to-side-lobe ratio (PSR) at the
finest level; the per-image consistency metric is the maximum Euclidean drift of the
chosen peak across pyramid levels.

Rotation-aware schedule
-----------------------

When the per-instrument camera-rotation flag is on, the technique runs the rotation-aware
schedule: the template is rotated about the body-centroid pivot at 11 angles spanning
:math:`\pm \mathrm{max\_rotation\_deg}` at the coarsest pyramid level, the best three
rotations advance to a 5-sample refinement at the next level, and the best one advances to a
3-sample fine refinement at the finest level. The (dv, du, theta) triple at the global
maximum is returned with a 3x3 covariance whose translation block is the matched-filter
covariance (plus model-error terms) from the chosen rotation. The technique reports its
rotation as
unobservable: the NCC-peak quality is a PSR/PMR separation ratio rather than a
log-likelihood, so it carries no calibrated Fisher information about rotation, and the
rotation diagonal holds the
:data:`~spindoctor.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE` sentinel. The
ensemble's ``pinvh`` combine maps that to a near-zero rotation contribution, so the disc
technique constrains translation while abstaining on rotation.

Restrictions and assumptions
----------------------------

- The orchestrator must populate
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` and
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.sensor_mask_ext` on the per-image
  context; in their absence the technique cannot evaluate the NCC.
- The composite template must have non-empty support inside the search window. An empty
  template (every body off-frame) collapses the NCC to a constant; the spurious gate flags
  the result.
- Multi-body composites assume the per-body templates are coherent — the per-body
  ``BODY_DISC`` emission gates upstream
  (:doc:`dev_guide_navigation_models_body`) ensure each contributing body has a sharp,
  high-contrast template.
- The rotation-aware schedule assumes the rotation pivot is well-defined. When every
  consumed body has a degenerate centroid (single-pixel bodies) the pivot collapses to the
  image centre.

Sources of uncertainty
----------------------

The reported covariance is a matched-filter (peak-curvature) statistical term plus three
model-error terms added in quadrature on the translation diagonal.

The statistical term is ``sigma_n**2 * inv(sum grad M grad M^T)``, evaluated on the aligned
template ``M``. Both the residual noise ``sigma_n`` and the template gradients are measured
on the **same** zero-mean, unit-std normalized surfaces, so the covariance is independent of
the (arbitrary) template amplitude -- measuring the noise on a normalized residual while
taking gradients on the raw template would shrink the covariance by the template variance and
under-report the sigma by one to two orders of magnitude. A residual-correlation-area factor
inflates this white-noise bound so spatially correlated residuals (camera PSF, anti-alias and
sub-pixel-refine low-pass, coherent model error) are not counted as thousands of independent
samples.

On a well-matched template that statistical term is far smaller than the true registration
error, which is dominated by silhouette / photometric model error rather than photon noise.
Three model-error terms restore calibration so the reported sigma **tracks** the error across
body size and resolution instead of pinning to a constant floor:

- a **localization-spread** term ``(localization_uncertainty_scale * consistency)**2``, where
  ``consistency`` is the inter-pyramid-level peak migration in pixels -- a peak that walks
  between levels is empirically the less well-localized one;
- a **silhouette** term ``(model_error_size_frac * body_diameter_px)**2`` -- a coherent shape
  mismatch displaces the peak by a roughly fixed fraction of the body's extent, so a large
  body's uncertainty is larger in absolute pixels than a small body's;
- an absolute ``model_error_floor_px`` floor for the size-independent residual (pointing,
  camera distortion, sub-pixel interpolation).

The covariance still does not model the Z-buffer compositing bias when bodies are
near-occluding. When the chosen peak sits within the at-edge tolerance of any axis bound,
or when the rotation parameter is at the configured fraction of its cap, the result is
flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
hard-zero gate forces confidence to zero. The pyramid-consistency check flags peaks that
drift across levels as
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyDiscCorrelateNav.tuning``
in ``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses
  this fraction of the per-image
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.
- ``consistency_max_fraction_of_diameter`` / ``consistency_max_px`` — floats, defaults
  ``0.025`` / ``4.0`` px. Diameter-scaled and absolute floors on the accepted pyramid-level
  peak migration; the applied spurious cap is the larger of the two.
- ``refine_lowpass_sigma_px`` — float, default ``1.0`` px. Gaussian low-pass applied to both
  surfaces in the full-resolution sub-pixel refine (see "Sub-pixel refinement band-limit").
  Removes the sharp-edge cross-power aliasing that otherwise leaves a ~0.03 px (up to ~0.1 px
  at the worst two-axis phase) sub-pixel-phase S-curve in the recovered offset; ``0.0``
  disables it and restores that bias.
- ``localization_uncertainty_scale`` — float, default ``1.0`` (dimensionless). Multiplies the
  inter-pyramid-level peak migration (``consistency``, px) into an added translation sigma (see
  "Sources of uncertainty"). ``0.0`` disables it.
- ``model_error_size_frac`` — float, default ``0.005`` (fraction of the body diameter). The
  silhouette model error as a fraction of the body's ext-FOV extent. ``0.0`` disables it.
- ``model_error_floor_px`` — float, default ``0.05`` px. Absolute size-independent floor added
  in quadrature. ``0.0`` disables it.

The remaining numeric thresholds (NCC peak quality, consistency tolerance, top-k count) are
shared across every pyramid-NCC technique and live as module-level constants in
:mod:`spindoctor.support.correlate`; consuming techniques pass overrides at call time when needed.

Per-instrument overrides
------------------------

The per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not
override ``rotation_at_edge_fraction``. The search-window margin used by the
at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. The formula spec is
``techniques.BodyDiscCorrelateNav`` and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.ncc_peak` — alpha = 0.051,
  offset = 6.0, divisor = 14.0, cap at 1.0. PSR-style quality measure of the chosen NCC
  peak. The calibration campaign's raw p5/p50/p95 is 6.2/9.9/20.5, so the offset-6
  divisor-14 transform spans that range in [0, 1]. The fitted weight is small: with
  textured and banded discs in the cohort a sharp peak no longer separates success from
  failure — the peak-to-runner-up ratio and consistency carry the discrimination.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.consistency_ratio` —
  alpha = -1.365, offset = 0.0, divisor = 1.0, no cap. Pyramid-level consistency
  normalized by the per-image diameter-scaled spurious threshold: a ratio below 1.0 means
  the result is within budget, and a ratio of exactly 1.0 sits at the spurious edge, where
  the term contributes its full negative alpha to the sigmoid argument. Using the ratio
  rather than the raw pixel value lets the divisor stay at 1.0 while the underlying budget
  tracks body diameter.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.body_count` — alpha = 0.4,
  offset = 0.0, divisor = 3.0, cap at 1.0. Number of ``BODY_DISC`` features fused into the
  composite. More bodies sharpen the joint geometric constraint up to a 3-body saturation.
  Not identifiable in the single-body sim campaign; retained at the design prior.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.peak_to_runner_up_ratio` —
  alpha = 1.121, offset = 0.0, divisor = 2.0, cap at 1.0. Ratio of the winning peak's
  quality to the next-best peak outside the exclusion radius.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 0.429`. No post-sigmoid ``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_body_disc.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` and the rotation
  / template-rotate / composite helpers.
- ``src/spindoctor/feature/composition.py`` —
  :func:`~spindoctor.feature.composition.compose_template_features`, the Z-buffer paint helper
  that fuses per-body templates into a single composite.
- ``src/spindoctor/support/correlate.py`` —
  :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`, the shared pyramid-NCC entry
  point.
- ``src/spindoctor/nav_technique/confidence.py`` — shared sigmoid-combination evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.name` —
  ``'BodyDiscCorrelateNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.accepts_feature_types`
  — ``frozenset({BODY_DISC})``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.requires_prior` —
  ``False``. Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'ncc_peak', 'peak_to_runner_up_ratio', 'consistency_px',
  'consistency_ratio', 'used_gradient', 'body_count'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.ncc_peak` — peak NCC quality at
  the finest pyramid level. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.peak_to_runner_up_ratio` —
  ratio of the winning peak's quality to the next-best peak's outside the exclusion radius.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.consistency_px` — maximum
  Euclidean drift across pyramid levels. Consumed by the spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.consistency_ratio` —
  ``consistency_px`` normalized by the per-image diameter-scaled spurious threshold, so a
  healthy fit on a large body is not penalized by the same divisor that suits a small
  body. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.used_gradient` — True when
  ``auto`` mode picked the gradient pass. Diagnostic only; not in the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.body_count` — number of
  ``BODY_DISC`` features fused.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav.navigate`:

1. Open a logged section. Filter the offered features down to ``BODY_DISC`` entries that carry
   a template payload via the private filter helper.
2. Fuse the per-body templates into a single composite via
   :func:`~spindoctor.feature.composition.compose_template_features`. The composite carries a
   single bounding box, brightness image, and mask.
3. Read the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`.
4. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**  Run :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`
     once on the unrotated composite. The pyramid returns the chosen peak's
     ``(dv, du)``, the 2x2 CRLB covariance, ``quality``, ``consistency``, ``spurious``,
     ``at_edge``, ``used_gradient``, and the top-``k`` peak telemetry.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``; the (2, 2) covariance is reported.
   - **Rotation fit.**  Run the private 3-DoF pyramid: 11 rotation samples at the coarsest
     pyramid level, top 3 advance to 5 samples at the next level, top 1 advances to 3
     samples at the finest level. At each rotation sample the template is rotated about the
     body-centroid pivot via the private template-rotate helper and the pyramid evaluates as
     usual. The (dv, du, theta) triple at the global maximum is returned with a 3x3
     covariance: the translation block is the CRLB from the winning rotation's correlation
     curvature, and the rotation diagonal is always the rotation-unobservable sentinel —
     the PSR/PMR-style peak quality is not a log-likelihood, so its curvature across the
     rotation schedule carries no Fisher information about rotation, and no calibrated
     curvature-to-variance map exists. The disc technique contributes a translation
     estimate while abstaining on rotation (the theory section above discusses why).

5. Apply the at-edge tests against the search-window axis bounds and the rotation cap, and
   the spurious tests using the pyramid wrapper's ``spurious`` and ``consistency``
   readouts.
6. Build a :class:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics` from the pyramid
   wrapper's quality / consistency / used-gradient / runner-up-ratio readouts plus the
   composite body count, evaluate the confidence spec via
   :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`, log the per-term
   breakdown via :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and
   assemble the :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``body_full_fov`` (Cassini ISS NAC, image ``N1572105349_1``)
    Dione fills the FOV at predicted disc diameter approximately 155 px. The body model
    emits a ``BODY_DISC`` feature; the technique converges to (9.17, -17.01) px against the
    operator-verified ground truth :math:`(\Delta v, \Delta u) = (8.68, -17.37)` px (within
    0.5 px of true). The pyramid wrapper's ``consistency`` reports 2.78 px disagreement
    between coarse and fine pyramid levels in the auto-gradient pass, exceeding the
    consistency tolerance and tripping the spurious gate; the result is therefore reported
    with confidence zero even though the offset is geometrically correct.

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right with about 22 % of the disc off-frame. The body model
    emits a ``BODY_DISC`` feature; the disc-template NCC peak collapses against the
    heavily-cropped silhouette and the technique correctly flags itself spurious.
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` carries the navigation
    on this image; the operator-verified offset is
    :math:`(\Delta v, \Delta u) = (11.0, 29.5)` px.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees.
    The body model emits two ``BODY_DISC`` features;
    :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` Z-buffer paints
    them into a composite (the closer body's pixels overwriting the farther body's) and the
    pyramid NCC converges to (6.76, -17.71) px against the operator-verified ground truth
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px. The composite's joint geometric
    constraint sharpens the peak relative to a per-body solo correlation; the multi-body
    body_count term contributes a positive offset to the confidence sum.
