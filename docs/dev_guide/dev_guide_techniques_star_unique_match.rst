==========================================================
Star Unique Match (StarUniqueMatchNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` is the pass-1
1-star or 2-star unique-match technique. It searches a small window around the prediction of
the brightest predictable star (or the brightest two), takes the brightest peak above a
noise threshold, and reports the implied translation (1-star path) or the implied similarity
transform (2-star path); the 1-star path additionally requires the star to be a configured
magnitude margin brighter than every other predictable star in the field. It runs without a prior — its job is to seed the pass-2
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` on scenes where the SPICE
pointing error is small enough that the brightest predictable star sits inside the
per-instrument search window.

Feasibility passes when at least one usable
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature is in the input set;
feasibility fails when none are usable (every predictable star is occluded by a body
silhouette or ring annulus).

Theory
======

The technique exploits a structural fact: when one predictable star dominates the field's
predicted SNR by a comfortable margin, the brightest peak inside a small search window
around that star's prediction is almost certainly the matched detection — there is no other
predictable source bright enough to be confused with it. Two stars dominating the field
similarly support a 2-star unique-match path that recovers translation plus rotation by
Procrustes alignment.

Brightness-margin gate
----------------------

The matcher computes a magnitude difference between the brightest predictable star and the
next-brightest predictable star in the FOV (using the standard relation
:math:`\Delta m = 2.5 \log_{10}(\mathrm{SNR}_{1} / \mathrm{SNR}_{2})`). The margin floor
(``brightness_margin_to_next_catalog_star_mag``) gates only the 1-star path: a star whose
brightness margin falls below it is not "uniquely bright" and the 1-star path fails. The
2-star path is attempted first whenever at least two usable stars exist, regardless of the
margin; on that path the margin (to the next-brightest *unmatched* star) is computed as a
diagnostic only.

Local centroid
--------------

For each candidate match, the technique slices a window of half-width ``search_window_px``
around the predicted position, takes the brightest pixel above
``detection_sigma * image_noise_sigma``, and fits a brightness-weighted moment over a
``centroid_box_half_px``-half-width box around that peak to recover the sub-pixel centroid.
A peak that fails the noise threshold collapses the candidate; on the 2-star path a
best-assignment residual exceeding ``max_residual_px`` triggers the fall back to the 1-star
path, and on the 1-star path a no-rival detection whose prediction-to-detection distance
exceeds ``one_star_max_residual_px`` marks the result spurious.

1-star path
-----------

The 1-star path reports the per-star residual (observed centroid minus predicted position)
as the translation. The reported covariance is the per-feature Cramer-Rao lower bound from
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px`, inflated by ``residual_px**2``
on the diagonal so a noisy match honestly reports its scatter rather than the noise-free
CRLB floor. Rotation is unobservable on a single point match.

The path rests on the premise that the search window holds a *single unambiguous*
detection, and the absolute ``detection_sigma`` threshold alone cannot enforce that: the
maximum of the window's several thousand noise pixels sits at 3.5-4.5 sigma, so with a
marginal star the brightest pixel is routinely a noise spike that clears the threshold.
The detection therefore also passes a peak-to-runner-up ambiguity gate
(``one_star_min_peak_ratio``): the background-subtracted peak must exceed the window's
runner-up (outside the detection's own centroid box) by the configured ratio, or the
result is spurious with an ``ambiguous_detection`` reason. A real star detection towers
over the window's noise order statistics while a matched spike sits within a few percent
of its runner-up; the ratio is unit-free, so no DN-scale or photometric calibration enters
the gate. The measured ratio is surfaced as the
:attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
diagnostic.

The peak-ratio and brightness-margin gates both report an ``inf`` sentinel when no rival
exists (no runner-up detection above the window background, no other predictable catalog
star). The sentinels pass the ratio checks by design so a genuinely unique bright star
stays matchable — but an infinite peak ratio also means the ambiguity gate measured
nothing: on a real noisy background the runner-up is always finite, so the sentinel is the
signature of a lone hot pixel or artifact on a flat or quantized frame. In
that vacuous case the detection must additionally sit within ``one_star_max_residual_px``
of the prediction. The residual on this path is the claimed offset itself, and with no
rival statistics to lean on, a no-rival detection far from the prediction is far more
likely an unrelated artifact than a large pointing error; the default bound is one third
of ``search_window_px``, the ~1-sigma core of the pointing envelope the window is sized to
bracket. A finite peak ratio leaves acceptance with the measured ambiguity gate: genuine
one-star matches with offsets up to ~24 px exist in the operator-verified library, so no
uniform residual cut below the search window is possible.

2-star path
-----------

When at least two usable stars exist, the technique first attempts the 2-star path: a
Procrustes / similarity-transform fit on the two correspondences that reports the recovered
translation plus rotation. The rotation variance is derived analytically as

.. math::

    \sigma_{\theta}^{2} = \frac{2 \,(\sigma_{v}^{2} + \sigma_{u}^{2})}{L^{2}}

where :math:`L` is the catalog-pair angular separation and the per-axis sigmas are the
average of the two stars' CRLB floors. When the 2-star path's best assignment residual
exceeds ``max_residual_px`` the technique falls back to the 1-star path.

Per-mode confidence cap
-----------------------

The 1-star path cannot self-check (a single observation has nothing to validate against), so
the technique applies a post-sigmoid cap of ``one_star_confidence_cap`` (default 0.7). The
2-star path can validate the assignment via the per-correspondence residual, so its cap is
``two_star_confidence_cap`` (default 0.8). The caps are applied by the technique itself,
not by the YAML spec's :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap`, so the
two paths can share a single spec.

Restrictions and assumptions
----------------------------

- The brightness-margin gate is driven by the catalog visual magnitudes: each star's
  ``predicted_snr`` is the magnitude-margin effective SNR derived from how far the star
  sits below the per-observation limiting magnitude, so the SNR ratio between two stars
  recovers their catalog magnitude difference. A wrong catalog magnitude shifts the
  brightness-margin computation and may erroneously accept or reject the match.
- The search window assumes the SPICE pointing error is bounded by ``search_window_px``
  (default 30 px). When the per-instrument pointing envelope exceeds this width, the
  brightest peak in the window is not the matched detection and the technique reports a
  wrong centroid — this is the regime where the multi-star
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` triplet
  matcher takes over.
- Both paths assume the matched detection is unsaturated. A saturated star whose centroid
  has been clipped will produce a biased centroid; in practice the brightest catalog stars
  are held off by the magnitude gate. The predicted-position saturation / cosmic-ray mask is
  not consulted to reject stars (a star's predicted pixel is not special once an offset is
  present), so among the predictable stars occlusion is the only thing that makes one
  unusable.

Sources of uncertainty
----------------------

The 1-star covariance is the per-feature CRLB floor from
:attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` inflated by ``residual_px**2``
on the diagonal. The 2-star translation covariance is the average of the two stars'
residual-inflated matrices, plus the analytic rotation variance derived above when rotation
is fitted.
When the converged offset sits within the at-edge tolerance of any axis bound, or when the
2-star rotation parameter is at the configured fraction of its cap, the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. When a no-rival 1-star detection's
prediction-to-detection distance exceeds ``one_star_max_residual_px`` the result is
flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.StarUniqueMatchNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``brightness_margin_to_next_catalog_star_mag`` — float, default ``1.5`` (mag). Minimum
  magnitude difference between the brightest predictable star and the next-brightest for
  the 1-star path to fire.
- ``search_window_px`` — float, default ``30.0`` px. Half-width of the search window
  around each prediction. Should bracket the per-instrument SPICE pointing-error
  envelope.
- ``centroid_box_half_px`` — int, default ``3`` px. Half-width of the brightness-weighted
  moment box around the brightness peak.
- ``max_residual_px`` — float, default ``4.0`` px. Maximum allowed best-assignment
  residual in the 2-star path before falling back to the 1-star path.
- ``detection_sigma`` — float, default ``4.0`` (dimensionless). Detection-threshold
  multiplier on the per-image noise sigma.
- ``one_star_confidence_cap`` — float, default ``0.7`` (dimensionless). Post-sigmoid
  confidence cap when the 1-star path fires. A single match cannot self-check.
- ``two_star_confidence_cap`` — float, default ``0.8`` (dimensionless). Post-sigmoid
  confidence cap when the 2-star path fires. Per-correspondence residual cross-checks the
  assignment.
- ``one_star_min_peak_ratio`` — float, default ``1.5`` (dimensionless). Minimum
  background-subtracted peak-to-runner-up ratio for a 1-star detection; below it the
  result is spurious with an ``ambiguous_detection`` reason. Values below 1 are invalid
  (the peak is by definition at least the runner-up).
- ``one_star_max_residual_px`` — float, default ``10.0`` px. Maximum
  prediction-to-detection distance accepted by the 1-star path when the peak-ratio gate
  is vacuous (its no-rival ``inf`` sentinel); beyond it the result is spurious with a
  ``no_rival_detection`` reason. Default is one third of ``search_window_px`` (the
  ~1-sigma core of the pointing envelope the window brackets). Not consulted when the
  peak ratio is finite. Values at or below 0 are invalid.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). Fraction of
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` at which the
  converged 2-star rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Only the 2-star
  Procrustes path uses it; the 1-star path always reports rotation as unobservable.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not
override any of these knobs.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. Spec is
``techniques.StarUniqueMatchNav``; consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr` —
  alpha = 0.809, offset = 0.0, divisor = 250.0, cap at 1.0. Predicted SNR of the brightest
  catalog star. Higher SNR shrinks the centroid CRLB; saturates at SNR=250 (calibration
  campaign raw p5/p50/p95 = 15/85/252 — the sim's flux-normalized star deposit brightens
  the healthy range, so the divisor spans it).
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`
  — alpha = 1.002, offset = 1.5, divisor = 1.5, cap at 1.0. Magnitude margin to the
  next-brightest predictable star in the extfov. Below the 1.5 mag floor the 1-star path
  short-circuits before reaching the formula; above the floor, additional margin earns
  confidence up to a 1.5 mag span.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px` —
  alpha = -0.112, offset = 0.0, divisor = 2.0, no cap. Detection-vs-prediction residual.
  Larger residuals pull confidence down.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 0.781`. No post-sigmoid ``hard_cap`` is applied at the spec level; the
per-mode caps above are applied by the technique itself.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_star_unique_match.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` and the
  brightness-margin / 1-star / 2-star helpers.
- ``src/spindoctor/nav_technique/_star_helpers.py`` — package-private helpers ``usable_stars``,
  ``local_centroid``, ``predicted_snr``, ``predicted_vu``, ``brightness_margin_mag``, and
  ``similarity_transform_fit``.
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
base :class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.name` — ``'StarUniqueMatchNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset({STAR})``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` —
  ``{'at_edge', 'spurious', 'predicted_snr', 'brightness_margin_mag', 'residual_px'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.is_feasible`
and
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode` —
  ``'one_star'`` or ``'two_star'``. Diagnostic only; the confidence formula does not
  consume strings, but the per-mode cap branches inside
  :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`
  read it.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr` —
  predicted SNR of the brightest catalog star. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`
  — magnitude difference to the next-brightest unmatched catalog source predictable in
  extfov; ``+inf`` when no unmatched star exists. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px` —
  detection-vs-prediction residual. Consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`
  — background-subtracted peak over window runner-up for the 1-star path (``0.0`` when
  not measured; ``inf`` when the runner-up sits at or below the background). Diagnostic
  only; the ambiguity gate consumes it before the confidence formula runs.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`:

1. Open a logged section. Filter the offered features down to ``usable_stars``
   (predictable and not occluded by a body silhouette or ring annulus).
2. Sort the usable cohort by predicted SNR (brightest first) and compute the brightness
   margin between consecutive entries via ``brightness_margin_mag``.
3. Branch on the usable-star count:

   - **2-star path.**  Attempted first whenever at least two usable stars exist (no
     brightness-margin precondition). For each of the brightest two stars, slice a
     ``search_window_px``-half-width window around its prediction in
     :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` and call
     ``local_centroid`` to extract the sub-pixel detection. When both detections clear
     the noise threshold, run a Procrustes / similarity refit via
     ``similarity_transform_fit`` to recover translation and rotation. When the
     per-correspondence residual exceeds ``max_residual_px``, fall back to the 1-star
     path.
   - **1-star path.**  When fewer than two stars are usable (or the 2-star fallback
     fired), require the brightest to clear the brightness-margin floor, slice a window
     around its prediction, extract the sub-pixel
     centroid, pass the peak-to-runner-up ambiguity gate (plus the
     ``one_star_max_residual_px`` residual gate when that ratio is the no-rival
     sentinel), and report the per-star residual as the translation.

4. Result-shape branches on the chosen path and
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **1-star, no rotation.**  (2, 2) covariance: the CRLB floor from
     :attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` inflated by
     ``residual_px**2`` on the diagonal;
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **1-star, rotation fit.**  Rank-deficient (3, 3) covariance via
     :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`;
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is
     ``0.0`` and the sigma is the rotation-unobservable sentinel.
   - **2-star, no rotation.**  (2, 2) covariance from the average of the two stars'
     residual-inflated
     :attr:`~spindoctor.feature.feature.NavFeature.position_cov_px` matrices.
   - **2-star, rotation fit.**  Full (3, 3) covariance: the (2, 2) translation block is
     the average of the two residual-inflated per-star matrices, the rotation diagonal is
     the analytic
     :math:`2 (\sigma_{v}^{2} + \sigma_{u}^{2}) / L^{2}` form derived from the catalog-pair
     separation :math:`L`.
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is the
     Procrustes-fit angle and the sigma is the square root of the rotation diagonal.

5. Apply the at-edge test against the search-window axis bounds and the rotation cap.
6. Build a :class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`,
   apply the per-mode cap (``one_star_confidence_cap`` or ``two_star_confidence_cap``)
   inside the technique itself, log the breakdown, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each match at audit time.

Examples
========

``one_bright_star_no_body`` (Cassini ISS WAC, image ``W1449079117_1``)
    Single star (Vega) in an otherwise empty FOV. The stars model emits a single ``STAR``
    feature with a brightness margin of ``+inf`` to the (nonexistent) next-brightest.
    :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav` runs the
    1-star path: a small search window around Vega's prediction returns the brightest peak
    above the noise threshold, the brightness-weighted moment recovers the centroid, and
    the per-star residual is reported as the translation. The 1-star post-sigmoid cap of
    0.7 holds the technique's confidence below 0.7 even when every term saturates.
    Operator-verified offset is :math:`(\Delta v, \Delta u) = (3.06, -0.02)` px. The
    pass-2 :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` consumes
    this prior and polishes it; see :doc:`dev_guide_techniques_star_refine`.

``two_bright_stars_no_body`` (Cassini ISS / NHLORRI star-calibration scene class)
    Catalog yields exactly two unambiguous stars in the FOV, each at least 1.5 magnitudes
    brighter than its next-brightest competing candidate. The stars model emits two
    ``STAR`` features; the technique falls into the 2-star path, runs the per-star local
    centroid against each prediction, and — when both detections clear the per-mode
    photometric and at-image-edge gates — invokes
    ``similarity_transform_fit`` on the catalog-vs-detection point pair to recover a
    translation plus rotation simultaneously. The reported covariance is the
    closed-form 3x3 from the similarity-transform residual scatter; the post-sigmoid
    cap is the 2-star ``two_star_confidence_cap`` (default 0.8), allowing higher
    final confidence than the 1-star path. The
    :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode` field on the
    diagnostics carries ``'two_star'`` so the curator surfaces which path fired.

``faint_stars`` (Galileo SSI / Voyager outer-leg scene class)
    Every catalog star in the FOV is fainter than the per-observation limiting magnitude
    ``obs.star_max_usable_vmag()``. The stars model emits no ``STAR`` features that clear
    the magnitude gate, so the orchestrator does not offer any usable star feature to
    :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.is_feasible`.
    Feasibility fails with reason ``no_usable_star_features`` and the technique skips its
    :meth:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`
    pass entirely. The orchestrator surfaces the infeasibility on the per-image
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` via the per-technique
    :attr:`~spindoctor.nav_technique.feasibility.NavFeasibilityReport.reason` so the curator
    records which gate fired without the technique having to allocate a result
    record.
