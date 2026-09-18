==========================================================
Ring Annulus Correlate (RingAnnulusNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` recovers a single
translation by full-template normalized cross-correlation against a composite annulus
fused from every offered ``RING_ANNULUS`` feature. Per-planet annulus templates are Z-buffer
painted into a single postage stamp (the closer ring system's pixels overwrite the farther
one's), the result is run through the shared pyramid-NCC machinery in
:mod:`spindoctor.support.correlate`, and the chosen peak is returned with a matched-filter
covariance plus model-error terms (see "Sources of uncertainty").

Multi-planet annulus composites improve disambiguation in the same way as multi-body disc
composites: each annulus contributes its own translational constraint, the geometric
alignment between ring systems removes the "swap planet assignments" ambiguity, and the SNR
of the combined peak grows roughly as :math:`\sqrt{N}` if backgrounds are independent.
Multi-planet scenes are rare but real (the Cassini approach phase imaged Jupiter and Saturn
together; New Horizons imaged Jupiter from Pluto distance).

Feasibility passes when at least one offered ``RING_ANNULUS`` carries a template payload;
feasibility fails when no offered feature has a template (a ring system whose per-edge
polylines were emitted as ``RING_EDGE`` features instead).

Theory
======

The technique fits a per-image translation by maximizing the normalized cross-correlation
between the composite annulus template and the observed image.

Composite template construction
-------------------------------

When more than one planet emits a ``RING_ANNULUS`` feature (rare but supported), the per-planet
templates are fused via Z-buffer paint: at each pixel covered by more than one ring system,
the closer planet's template value (and its mask True) overwrite the farther one's so an
in-front ring system occludes an in-behind ring system in the composite. The fused
template carries one combined bounding box, one fused brightness image, and one fused mask;
the orchestrator's
:func:`~spindoctor.feature.composition.compose_template_features` helper does the work.

Cost function
-------------

The technique maximizes the masked normalized cross-correlation between the composite
template and the observed image (or a mode-selected gradient of it):

.. math::

    \mathrm{NCC}(\Delta v, \Delta u) =
        \frac{\langle T, I_{\Delta v, \Delta u} \rangle_{M}}
             {\sqrt{\langle T, T \rangle_{M} \cdot \langle I_{\Delta v, \Delta u},
                                          I_{\Delta v, \Delta u} \rangle_{M}}}

over the integer offsets in the per-instrument search window, with the masked inner product
restricted to pixels where the annulus mask is True. Sub-pixel refinement comes from a
localized upsampled DFT around the integer peak (the Guizar-Sicairos efficient
subpixel-registration method, ``upsampled_dft`` in ``src/spindoctor/support/correlate.py``,
applied by :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`); the template
gradient structure provides the matched-filter statistical covariance term (see "Sources of
uncertainty").

Mode selection (auto / raw / gradient)
--------------------------------------

The shared pyramid evaluates the correlation in two modes — raw vs. raw and gradient vs.
gradient — and ``auto`` picks whichever peak scores higher quality. Raw wins on
broad-brightness-gradient ring geometries (a low-resolution Saturn ring system where the
C-ring is uniformly dim and the A/B rings dominate the per-pixel intensity); gradient wins
when sharp ringlet edges dominate the per-pixel signal.

Search strategy
---------------

The shared pyramid-NCC entry point
(:func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`) runs coarse-to-fine, keeps the
top ``k`` peaks at each level, and reports the per-level consistency. See
:doc:`dev_guide_techniques_body_disc` for the pyramid-NCC mechanics that the body-disc
technique shares. Rotation fitting is disabled for ring annuli (the rotation pivot
of a ring system is its planet-center, which is well outside the rendered template's
support; an outer rotation search would have to rotate the template about a far-off-template
pivot and the NCC peak's rotation curvature is degenerate for symmetric annuli).

Restrictions and assumptions
----------------------------

- The orchestrator must populate
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` and
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.sensor_mask_ext`.
- The composite template must have non-empty support in the search window. An empty
  template (every annulus off-frame) collapses the NCC to a constant; the spurious gate
  flags the result.
- The upstream rings model decides whether to emit ``RING_ANNULUS`` or ``RING_EDGE`` features per
  the per-planet ``feature_emission.ring_annulus`` block in
  ``config_510_techniques.yaml`` — the technique only sees the path the model chose.

Sources of uncertainty
----------------------

The reported covariance is a matched-filter (peak-curvature) statistical term plus
model-error terms added in quadrature, following the same derivation as
:doc:`dev_guide_techniques_body_disc`: the statistical term measures noise and gradient
content on unit-std-normalized surfaces (amplitude-invariant, inflated by the residual
correlation area), and a ``localization_uncertainty_scale * consistency`` localization-spread
term plus a ``model_error_floor_px`` absolute floor carry the template model error that
dominates the true registration error. The ring annulus size (``model_error_size_frac``)
term is disabled by default because the ring bulk error does not track the annulus radial
span.

Radial orbit-uncertainty channel
---------------------------------

The matched-filter covariance describes the statistical lock onto the *modeled* annulus. A
catalog-orbit error displaces the whole predicted annulus coherently, and the correlation
absorbs that displacement into its reported offset without any residual scatter to reveal it
-- the same systematic-error hazard the ring-edge fit carries. The technique therefore
widens its covariance by the declared radial orbit uncertainty along the translation the
correlation would absorb such a displacement into, so a tight lock on a misplaced annulus is
not reported as a tight pointing fix. This is the correlation-side counterpart of the
ring-edge channel documented at :doc:`dev_guide_techniques_ring_edge`; both ring techniques
consequently price the same declared hazard, which is what lets the fused radial error bar
cover the residual bias rather than being carried by whichever member left it un-widened.

The absorbed direction comes from the annulus fit's own radial geometry: the outward normals
of the constituent ring edges painted into the composite template, which the upstream rings
model concatenates onto
:attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.orbit_normals_vu` with an effective
:attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.sigma_orbit_radial_px`. The technique
feeds those normals through the same absorbed-translation solve the ring-edge fit runs on its
fit vertices (``_absorbed_orbit_sensitivity`` in
:mod:`spindoctor.nav_technique.ring_edge_geometry`), so both techniques price an identical
annulus geometry identically: a short visible arc absorbs the radial error one-for-one along
its normal (a directional widening), while a closed ring barely absorbs it -- a uniform
radial error of a full ring is a dilation, not a translation -- and the bound is reported
isotropically, since which direction the nonlinear acquisition locks is exactly what cannot
be predicted. The effective sigma is echoed on
:attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.sigma_orbit_radial_px`.

When the chosen peak sits within the at-edge tolerance of any axis bound the
result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. The pyramid consistency check flags peaks that drift across
levels as :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

The technique's only per-technique ``tuning`` knobs in ``config_510_techniques.yaml`` are the
covariance model-error terms (the shared pyramid-NCC machinery owns the numeric thresholds for
peak quality, consistency, and top-k count):

- ``localization_uncertainty_scale`` — float, default ``1.0`` (dimensionless). Multiplies the
  inter-pyramid-level peak migration (``consistency``, px) into an added translation sigma.
- ``model_error_size_frac`` — float, default ``0.0`` (fraction of the annulus radial span);
  disabled because the ring bulk error does not track span.
- ``model_error_floor_px`` — float, default ``0.13`` px. Absolute floor added in quadrature.

Feature-emission tunables (per-planet)
--------------------------------------

The upstream rings model decides whether to emit ``RING_ANNULUS`` or ``RING_EDGE`` features based on
``feature_emission.ring_annulus`` in ``src/spindoctor/config_files/config_510_techniques.yaml``.
Two gates route an edge to the annulus template. The system-level gate is inclusive:
``km_per_pixel_radial >= kmpp_threshold`` forces every surviving edge in the scene into the
composite, so a scene exactly at the threshold is annulus-class. Below the threshold a per-edge
gate applies: an edge joins the composite when ``radial_extent_px <= max_radial_px and not
straight`` -- a compressed polyline that is flagged straight-line stays on the per-edge path as a
rank-1 ``RING_EDGE`` rather than joining the composite. A sub-threshold scene can therefore emit
a mix: ``RING_EDGE`` features for well-extended or straight edges alongside one ``RING_ANNULUS``
composite for the edges that compressed below ``max_radial_px``.

For Saturn the km/px threshold routes by measured reliability, not by pixel span: at or above
25 km/px radial resolution the whole Saturn system routes to the annulus composite, and the
per-edge distance-transform fit receives only sub-25 km/px scenes. A 131-frame clean-truth
head-to-head measured the per-edge fit wrong-when-accepted at 5% / 13% / 56% / 100% in the
300-1000 / 100-300 / 25-100 / 0-25 km/px bands, while the annulus fit was wrong on zero
accepted answers at every band -- so the annulus route covers the regimes that comparison
validates, everything at or above 25 km/px. The per-edge fit degrades toward fine resolution, where the ring alias lattice
resolves into many distinct similar concentric edges and a shape-only fit locks onto the wrong
one; the rendered-brightness template disambiguates the lattice because relative ring
brightness is part of the match. Below 25 km/px the trustworthy evidence is three frames on
which neither technique is validated (the edge path re-locked on all three; the annulus
near-missed twice and failed once), so the threshold stops where the evidence stops. The
threshold is Saturn-measured only; the other planets' entries keep geometric values.

- ``feature_emission.ring_annulus.default.max_radial_px`` — float, default ``5.0`` px.
  Maximum per-edge polyline radial extent below which the rings model emits a ``RING_ANNULUS``
  template instead of per-edge polylines. Per-polyline gate; fires when a single ring edge
  has compressed below this width.
- ``feature_emission.ring_annulus.default.kmpp_threshold`` — float, default ``1000.0`` km/px.
  Ring-radial km/px threshold above which the entire ring system is annulus-class
  regardless of any per-polyline metric.
- ``feature_emission.ring_annulus.planets.SATURN.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.SATURN.kmpp_threshold`` — float, default
  ``25.0`` km/px. Measured routing threshold: at or above 25 km/px the whole Saturn
  system is annulus-class; the per-edge fit receives only sub-25 km/px scenes.
- ``feature_emission.ring_annulus.planets.JUPITER.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.JUPITER.kmpp_threshold`` — float, default
  ``200.0`` km/px. Jupiter's main ring is ~10x narrower than Saturn's, so the kmpp
  threshold is correspondingly tighter.
- ``feature_emission.ring_annulus.planets.URANUS.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.URANUS.kmpp_threshold`` — float, default
  ``300.0`` km/px.
- ``feature_emission.ring_annulus.planets.NEPTUNE.max_radial_px`` — float, default
  ``5.0`` px.
- ``feature_emission.ring_annulus.planets.NEPTUNE.kmpp_threshold`` — float, default
  ``500.0`` km/px.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not
override any ``feature_emission`` keys. The search-window margin used by the
at-edge test comes from the per-instrument
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. The formula spec is
``techniques.RingAnnulusNav`` in the same YAML file and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak` — alpha = 1.168,
  offset = 6.0, divisor = 45.0, cap at 1.0. PSR-style quality measure of the chosen NCC
  peak. The calibration campaign's raw p5/p50/p95 is 10/27/42, so the offset-6 divisor-45
  transform spans that range in [0, 1].
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio` —
  alpha = 1.059, offset = 0.0, divisor = 2.0, cap at 1.0. Ratio of the winning peak's
  quality to the next-best peak's outside the exclusion radius.
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` —
  alpha = 0.593, offset = 0.0, divisor = 2.0, cap at 1.0. Number of ``RING_ANNULUS`` features
  fused. Multi-planet scenes saturate at 2 (vs ``body_count``'s 3).

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero. The constant baseline is :math:`\alpha_{0} = 0.647`. No post-sigmoid
``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_ring_annulus.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` and the per-feature
  filter / composite helpers.
- ``src/spindoctor/feature/composition.py`` —
  :func:`~spindoctor.feature.composition.compose_template_features`.
- ``src/spindoctor/support/correlate.py`` —
  :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`, the shared pyramid-NCC entry
  point.
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.name` —
  ``'RingAnnulusNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.accepts_feature_types`
  — ``frozenset({RING_ANNULUS})``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.requires_prior` —
  ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'ncc_peak', 'peak_to_runner_up_ratio', 'used_gradient',
  'annulus_count'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak` — peak NCC
  quality. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio` —
  ratio of winning peak's quality to next-best. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` — number of
  ``RING_ANNULUS`` features fused.
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.used_gradient` — True when
  ``auto`` mode picked the gradient pass.
- :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.sigma_orbit_radial_px` —
  effective declared radial orbit-uncertainty sigma added to the reported covariance through
  the absorbed translation direction. Auditability only; not consumed by the confidence
  formula.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav.navigate`:

1. Open a logged section. Filter the offered features to ``RING_ANNULUS`` entries that carry a
   template payload via the private filter helper.
2. Fuse the per-planet templates via
   :func:`~spindoctor.feature.composition.compose_template_features` into a single composite.
3. Read the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`.
4. Run :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks` on the composite
   template against the extfov image. The pyramid returns the chosen peak's
   ``(dv, du)``, the 2x2 CRLB covariance, ``quality``, ``consistency``, ``spurious``,
   ``at_edge``, ``used_gradient``, and the top-``k`` peak telemetry.
5. Widen the (2, 2) covariance by the radial orbit-uncertainty channel: aggregate the
   consumed annuli's :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.orbit_normals_vu`
   and effective :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.sigma_orbit_radial_px`,
   derive the absorbed-translation sensitivity, and add the declared variance along it (see
   "Radial orbit-uncertainty channel"). The covariance shape is always (2, 2); rotation
   fitting is disabled for ring annuli. When
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true the
   technique embeds the (2, 2) translation block in a (3, 3) covariance via
   :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable` and reports the
   rotation as unobservable.
6. Apply the at-edge tests against the search-window axis bounds.
7. Build a :class:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the breakdown, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A high-resolution Saturn-ring scene whose individual ring edges resolve into separable
    polylines. On this scene the rings model emits ``RING_EDGE`` features rather than
    ``RING_ANNULUS``, so :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`
    is not feasible. The annulus path fires on lower-resolution / approach-phase scenes
    where the per-planet km/px exceeds the configured threshold.

A multi-planet illustration: a hypothetical fly-by image that catches both Jupiter's main
ring and Saturn's ring system (compressed below their per-planet kmpp thresholds) emits two
``RING_ANNULUS`` features. The technique Z-buffer paints them into a composite (the closer
planet's annulus overwriting the farther one's) and the pyramid NCC's joint geometric
constraint produces a sharper peak than either annulus alone. The
:attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count` term in the
confidence formula contributes a positive offset.
