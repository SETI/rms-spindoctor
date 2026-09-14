==========================================================
Body Blob Centroid (BodyBlobNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` recovers a single translation
from one or more body brightness centroids. For each offered ``BODY_BLOB`` feature the technique
computes a brightness-weighted moment inside the predicted bounding box, compares the
observed centroid to the predicted lit-weighted centroid carried on the feature, and runs an
inverse-variance-weighted joint fit across every consumed body to recover the per-image
translation. With ``N >= 2`` blobs the fit is over-determined and the joint solution is
robust to centroid errors on any single body.

Feasibility passes when at least one offered ``BODY_BLOB`` carries a non-zero predicted
diameter; feasibility fails when every blob has degenerate geometry (a sub-pixel body that
collapses the brightness-weighted moment).

The technique reports a confidence intrinsically capped at 0.4 via the spec's
:attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap` — a brightness-weighted
centroid is much weaker evidence than a limb fit, so even an ideal blob match cannot dominate
the ensemble combine.

Theory
======

The technique fits a per-image translation by minimizing the inverse-variance-weighted
squared residual between the per-blob observed and predicted centroids.

Coarse acquisition (lit-shape matched filter)
---------------------------------------------

A brightness-weighted moment only sees the body when it already sits inside the predicted
bounding box, so the bare centroid's capture range is just the box -- a few pixels of
per-body slop. Once the SPICE pointing error exceeds that slop the body drifts out of the
box, the moment is taken over a clipped fragment, and the technique reports a *silently*
biased centroid (no spurious or at-edge flag fires). To extend the capture range to the full
extended-FOV search window, each blob first runs a coarse acquisition that re-centres its
bounding box on the body before the centroid is taken:

- If a pass-1 prior offset is installed on the context (another technique already located the
  body), the box is shifted by the rounded prior. The prior is a measured offset, so it
  applies regardless of phase.
- Otherwise the technique correlates a matched-filter template of the predicted *lit
  silhouette* against the lit-signal image (background subtracted, clipped at zero,
  sky-masked) over ``predicted_center +/- margin``. The response peaks where a body of that
  shape is best centered; the integer peak offset re-centres the box. The template depends on
  phase (:data:`~spindoctor.nav_technique.nav_technique_body_blob._COARSE_CORRELATION_MAX_PHASE_DEG`,
  90 deg):

  - **At or below half phase** the lit silhouette is a near-full disc, so the kernel is a
    filled disc of the predicted body radius.
  - **Above half phase** the sunlit region is a thin crescent whose bright pixels sit a
    fraction of a radius off the body center; a disc kernel would lock onto the crescent arc
    rather than the center. The kernel is instead a *synthesized crescent* -- a Lambertian
    ``max(0, cos(incidence))`` rendering of a sphere of the predicted radius at the body's
    phase, lit from the sub-solar direction the ``BODY_BLOB`` feature carries
    (``sub_solar_dir_vu``, the projected body-to-Sun direction; see
    :doc:`dev_guide_navigation_models_body`). Correlating the crescent puts the template
    *center* on the body center instead of the bright arc.

  The kernel is flipped before the FFT so the operation is a cross-correlation, and the
  template's own brightness-centroid offset is added back to the peak: the feature carries the
  body's *lit* centroid (which on a crescent sits off the geometric center), so the recovered
  shift is expressed in lit-centroid terms and matches the residual the centroid step forms.

Both templates clamp their radius to the frame's half-diagonal
(``_clamped_kernel_radius``): a template
larger than the frame adds no localization information, while the kernel array and its FFT
convolution allocate memory quadratically in the predicted diameter — a mostly off-frame gas
giant predicts tens of thousands of pixels, enough to exhaust RAM without the clamp.

The crescent template needs the sub-solar direction. It is undefined near full phase (the lit
and geometric centroids coincide), where the disc kernel is used anyway, and is reported as
``(0, 0)`` then. If a body is past half phase yet carries no direction (its illumination
geometry was not populated), the coarse stage makes no relocation and keeps the predicted box
(an installed prior still applies). The coarse offset is integer; the sub-pixel precision
comes entirely from the brightness-weighted moment below, computed inside the re-centred box,
so the recovered ``observed - predicted`` residual already includes the coarse shift.

Per-blob centroid
-----------------

For each consumed body, the technique computes the brightness-weighted moment over the
(coarse-re-centred) predicted bounding box:

.. math::

    \bar{x}_{\mathrm{obs}} =
        \frac{\sum_{(v, u) \in \mathrm{bbox}} I(v, u) \,(v, u)}
             {\sum_{(v, u) \in \mathrm{bbox}} I(v, u)}.

The predicted centroid carried on the feature is the lit-weighted moment of the rendered
model (see :doc:`dev_guide_navigation_models_body`); the per-blob residual is

.. math::

    r_{i} = \bar{x}_{\mathrm{obs},\,i} - \bar{x}_{\mathrm{pred},\,i}.

Per-blob weight
---------------

Each blob is weighted by its photon-limited centroid precision, the standard CRLB scaling for
a uniform-brightness disc:

.. math::

    \sigma_{\mathrm{centroid}} \approx
        \frac{D_{\mathrm{px}}}{2 \, \sqrt{N_{\mathrm{lit}}} \, \mathrm{SNR}},
    \qquad w_{i} = \frac{1}{\sigma_{\mathrm{centroid},\,i}^{2}}

where :math:`D_{\mathrm{px}}` is the predicted disc diameter in pixels,
:math:`N_{\mathrm{lit}}` is the number of lit pixels inside the predicted bounding box, and
SNR is the per-pixel signal-to-noise ratio.

Joint translation fit
---------------------

The joint translation minimizes

.. math::

    C(\Delta v, \Delta u) = \sum_{i} w_{i} \,\bigl\| r_{i} - (\Delta v, \Delta u) \bigr\|^{2},
    \qquad w_{i} = \frac{1}{\sigma_{\mathrm{centroid},\,i}^{2}}.

The closed-form minimum is the inverse-variance-weighted mean of the per-blob residuals:

.. math::

    (\Delta v, \Delta u)^{*} = \frac{\sum_{i} w_{i} \, r_{i}}{\sum_{i} w_{i}}.

Joint covariance
----------------

The photon-only weights above make the bare precision-weighted-mean variance far too tight:
a brightness-weighted centroid is a biased estimate of the geometric center (lit-hemisphere
offset, shape irregularity), and that bias -- not photon noise -- dominates the true error. It
scales with the body radius, an error the per-blob CRLB is blind to. The reported per-axis
variance is therefore the reduced-chi-square weighted mean (with a ``1 / sum(w_i)``
inverse-precision floor) plus two model-error terms added in quadrature:

- a size-scaled centroid-model-error variance ``(model_error_size_frac * R_i)**2`` per blob,
  combined across blobs by inverse-variance (so a single blob reports exactly
  ``(model_error_size_frac * R)**2`` and a multi-blob fit tightens correctly), which makes the
  reported sigma track body size the way the photon-only weight cannot;
- an absolute ``model_error_floor_px`` floor for the size-independent residual (pointing).

The size term uses ``R_i = D_{\mathrm{px},\,i} / 2``. The
:attr:`~spindoctor.feature.flags.BodyBlobFlags.phase_irregularity_factor` still feeds the
confidence formula (down-weighting irregular high-phase scenes) but is not folded into the
covariance.

Restrictions and assumptions
----------------------------

- Per-blob centroids assume the (coarse-re-centred) bounding box truly contains the body's
  flux. When a cosmic-ray hit, an in-band stellar source, or a neighboring body's halo lands
  inside the box, the moment skews and the technique reports a wrong centroid. The upstream
  ``BODY_BLOB`` emission gates filter pathological cases (see
  :doc:`dev_guide_navigation_models_body`).
- The coarse lit-shape acquisition extends the capture range from the bounding box to the full
  search window at any phase: a disc template at or below half phase, a synthesized crescent
  above it. The only residual gap is a body past half phase whose illumination geometry was
  not populated (no ``sub_solar_dir_vu``), where the crescent cannot be oriented; such a body
  is then recovered only via the bounding-box centroid (small offsets) or an installed prior.
- A vanishing total flux (an entirely-in-shadow body whose predicted bounding box happens to
  cover the right part of the FOV) collapses the moment; the technique drops such blobs
  before the joint fit and reports a no-signal failure when every blob is dropped.
- **Undetected bodies are gated out; small detected ones are not.** The ``BODY_BLOB``
  feature's reliability is driven by a *measured* detection SNR: the model computes the
  flux-weighted effective count ``N`` of its *observable* lit pixels (the Kish effective
  sample size of the rendered brightnesses over on-sensor, unsaturated pixels -- a Lambert
  crescent's near-terminator tail and any off-sensor overhang do not inflate it), takes the
  median of the ``N`` brightest valid pixels in the search window
  (the predicted bbox expanded by the extfov margins, since the pointing error is unknown),
  and subtracts the level pure noise's top-``N`` order statistics would produce. The SNR
  sigmoid is centered at the technique's own 3-sigma lit-pixel threshold, so a window with
  no body-scale signal above the noise floor sits decisively below the keep threshold at any
  size, while a bright body only a little above the 5 px emission floor is admitted (the
  extent term applies a mild near-floor discount, not a second size gate — the reliability
  crosses the 0.20 gate at detection SNR ~3.1-3.3 across the emitted size range). Whether
  the centroid is then *precise* is the per-blob covariance's and the confidence formula's
  job, not the gate's.
- The technique carries no rotation evidence — a brightness-weighted centroid is rotation-
  invariant about itself. When the per-instrument
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  technique returns the rank-deficient 3x3 covariance from
  :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable` and reports
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` as zero with
  the rotation-unobservable sentinel sigma.

Sources of uncertainty
----------------------

The reported covariance is the reduced-chi-square weighted-mean variance plus the size-scaled
centroid-model-error and absolute-floor terms described above. It does
not capture systematic biases from a body whose true rotational orientation differs from the
rendered ellipsoid (the
:attr:`~spindoctor.feature.flags.BodyBlobFlags.phase_irregularity_factor` term tracks this so the
confidence formula can down-weight the technique on irregular high-phase scenes, but the
centroid itself remains biased). When the converged offset sits within
:math:`\mathtt{at\_edge\_tolerance\_px}` of any axis bound, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyBlobNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``model_error_size_frac`` — float, default ``0.05`` (fraction of the body radius). The
  size-scaled centroid-model-error term (see "Joint covariance"). ``0.0`` disables it.
- ``model_error_floor_px`` — float, default ``0.1`` px. Absolute size-independent floor added
  in quadrature. ``0.0`` disables it.

The remaining numeric thresholds (the per-blob CRLB scaling constants, the noise-floor
detection threshold) are derived from the per-image
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_noise_sigma` and the per-blob
geometry; no YAML knob is exposed.

Per-instrument overrides
------------------------

The ``at_edge_tolerance_px`` knob is global; per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override it. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic.
The formula spec is ``techniques.BodyBlobNav`` in the same YAML file and consumes attributes
off :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — alpha = 0.895, offset = 0.0, divisor = 600.0, cap at 1.0. Per-image SNR inside the
  predicted bounding box. Brightness-weighted centroid uncertainty shrinks with SNR; the
  calibration campaign's raw p5/p50/p95 is 18/61/517 (a heavy tail).
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` —
  alpha = -1.063, offset = 8.0, divisor = 130.0, cap at 1.0. Predicted body's longer-axis
  extent in pixels. The alpha is *negative*, reversing the design's more-signal prior: the
  lit-weighted-centroid model error grows with apparent size, so the absolute probability
  of recovering within 1 px falls as the blob gets bigger. Small blobs remain the
  technique's tightest regime in absolute pixels. Campaign raw p5/p50/p95 is 8/25/69.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — alpha = 0.4,
  offset = 0.0, divisor = 3.0, cap at 1.0. Number of ``BODY_BLOB`` features fused.
  Multi-body geometry over-determines the joint translation up to a 3-blob saturation.
  Not identifiable in the single-body sim campaign; retained at the design prior.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor`
  — alpha = -0.545, offset = 0.0, divisor = 0.35, cap at 1.0. Maximum phase-and-irregularity
  factor across the consumed blobs (see :doc:`dev_guide_navigation_models_body` for the
  formula); the divisor spans the campaign's raw p5/p50/p95 of 0.003/0.019/0.369. With
  surface texture and relief in the rendered cohort the irregularity penalty carries
  substantial weight: an irregular body's unpredictable shadowing is model error the
  ellipsoidal render cannot correct.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
firing forces confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 1.465`. A post-sigmoid ``hard_cap`` of ``0.4`` clamps the result: a
brightness-weighted centroid cannot drive the ensemble past 0.4 confidence even when every
term saturates.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_body_blob.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`, the per-blob residual
  collector, and the joint-translation helper.
- ``src/spindoctor/nav_technique/confidence.py`` — shared sigmoid-combination evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.name` — ``'BodyBlobNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.accepts_feature_types` —
  ``frozenset({BODY_BLOB})``.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.requires_prior` — ``False``.
  Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.confidence_attributes` —
  ``{'at_edge', 'body_snr_inside_predicted_bbox', 'body_extent_px', 'blob_count',
  'residual_px', 'max_phase_angle_deg', 'max_phase_irregularity_factor'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`
  — per-image SNR inside the predicted bounding box. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px` — predicted
  body's longer-axis extent in pixels. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` — number of
  ``BODY_BLOB`` features fused. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.residual_px` — joint-fit RMS
  residual after solving the precision-weighted mean. Diagnostic only; not in the formula.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_angle_deg` — maximum
  raw phase angle across consumed blobs. Diagnostic only; the formula consumes
  ``max_phase_irregularity_factor`` instead because raw phase understates the centroid
  uncertainty for an irregular body.
- :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor` —
  maximum
  :math:`(\sigma_{\mathrm{ellipsoid}} / R_{\mathrm{body}}) \cdot (1 + 2 \sin^{2}(\phi/2))`
  across consumed blobs, where :math:`\sigma_{\mathrm{ellipsoid}}` is the body's ellipsoid
  RMS shape residual in km and :math:`R_{\mathrm{body}}` its radius in km. Consumed by the
  confidence formula.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav.navigate`:

1. Open a logged section. Filter the offered features down to ``BODY_BLOB`` entries with a
   non-zero predicted diameter via the private eligibility helper.
2. Read the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`, the extfov image off
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext`, and the per-image noise
   sigma off :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_noise_sigma`
   (clamped at a tiny floor so the noise-floor test stays well-defined on near-blank
   inputs).
3. For each eligible blob, the private residual collector slices the predicted bounding
   box from the extfov image, evaluates the brightness-weighted-moment centroid, drops the
   blob when total flux falls below the noise floor, computes the per-blob residual against
   the predicted lit-weighted centroid, and accumulates the per-blob inverse-variance
   weight.

   - **No surviving blobs.**  The technique returns a spurious zero-confidence result via
     the private fail helper, with the corresponding
     :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics`.

4. The private joint-fit helper computes the inverse-variance-weighted-mean translation and
   the reduced-chi-square covariance with the size-scaled and absolute model-error terms.
5. Apply the at-edge test against the search-window axis bounds.
6. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**  The technique embeds the (2, 2) translation block in a (3, 3)
     covariance via
     :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`, sets
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` to ``0.0``,
     and reports
     :func:`~spindoctor.nav_technique.nav_technique.rotation_unobservable_sigma_rad` as the
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad`.
     A brightness-weighted centroid is rotation-invariant about itself, so the technique
     carries no rotation evidence.

7. Build a :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics` from the per-blob
   residuals (max-SNR, max-extent, blob-count, RMS residual, max raw phase, max
   phase-irregularity factor), evaluate the confidence spec via
   :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`, log the per-term
   breakdown via :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and
   assemble the :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas approximately 20 px in diameter in the lower left, at phase angle 72 degrees. The
    body model emits a single ``BODY_BLOB`` feature (the per-pixel ellipsoid uncertainty exceeds
    :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX` so ``LIMB_ARC`` is
    suppressed in favor of the centroid path).
    :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` consumes the blob and
    converges within ~1 px of the operator-verified offset
    :math:`(\Delta v, \Delta u) = (6.08, -1.53)` px. The post-sigmoid hard cap of 0.4
    keeps the technique from outranking a hypothetical limb fit on a similar but
    well-resolved scene.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible at phase angle approximately 90 degrees. When the body
    model emits ``BODY_BLOB`` features for both bodies (or one body's limb fails the uncertainty
    gate), :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` fuses the two
    centroids into a joint translation. The 3-blob saturation in the confidence formula is
    not reached, but the multi-body
    :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count` term still
    contributes a positive offset. Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px.
