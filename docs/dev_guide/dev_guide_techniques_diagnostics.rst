============================================================
Per-Technique Diagnostics (Shared Dataclass Family)
============================================================

Overview
========

Per-technique diagnostics are the typed dataclasses every navigation technique returns on its
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.diagnostics` field. Each
technique declares its own diagnostics dataclass — a frozen, narrow record of the per-fit
quantities that the confidence formula consumes and the curator surfaces in the JSON sidecar.
Centralizing every diagnostics dataclass in one module lets the curator's allow-list
discipline catch a programmer who adds a new diagnostic field without updating its JSON
schema, and lets the
:func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` walk verify at
config-load time that every YAML-driven confidence formula references only attributes the
technique actually emits.

Theory
======

A diagnostics dataclass is a frozen record whose fields are exactly the per-fit quantities
that downstream systems read. Two consumers exist:

The confidence formula
----------------------

Each :class:`~spindoctor.nav_technique.confidence.ConfidenceTerm` references a diagnostic-attribute
name; the shared evaluator reads that attribute off the diagnostics object and feeds it
through the offset / divisor / cap normalization before applying the linear coefficient. See
:doc:`dev_guide_techniques_confidence` for the sigmoid math. The technique's
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` allow-list spans
both the diagnostic-attribute names *and* any side-channel flags the spec is allowed to read
(``at_edge``, ``spurious``, etc., which live on the result rather than the diagnostics
object); the validation walk verifies that every term and every hard-zero key falls inside the
allow-list.

The curator
-----------

The orchestrator's curator
(:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict`) walks every diagnostics dataclass's
``CURATOR_FIELDS`` class attribute — a mapping of dataclass-field name to JSON-key name (or
``None`` to skip) — and emits exactly those fields into the per-image JSON sidecar. The
mapping format lets the JSON schema use a different name than the Python field (e.g. an
internal ``mode`` could surface as ``"path"`` in the JSON), but the conventional usage is
identity (the dataclass field name and the JSON key match).
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` runs at startup and
fails the build when a new dataclass field is added without updating
``CURATOR_FIELDS``.

Restrictions and assumptions
----------------------------

- Every diagnostics dataclass is frozen (``@dataclass(frozen=True)``); the technique builds
  one instance per fit and the orchestrator passes it on to the curator without mutation.
- Every dataclass declares a ``CURATOR_FIELDS`` :class:`typing.ClassVar` mapping covering
  every public field. Fields the curator deliberately omits are mapped to ``None``;
  every other field maps to its JSON key name. CI fails if any field is unmapped.
- All numeric fields are plain Python floats / ints. Numpy scalars are coerced before
  storage so the JSON serialiser does not encounter non-native types.
- Every per-technique confidence formula references only attributes that exist on the
  technique's diagnostics dataclass plus the four side-channel flags carried on the
  :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` itself
  (:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`,
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`, plus the
  technique's own internal flags exposed via an adapter object).

Sources of uncertainty
----------------------

Diagnostics are the *outputs* of the per-fit numerics — they record what the technique
measured rather than uncertainty about the measurement. Any uncertainty quoted on the
diagnostic value (e.g. an LM RMS residual) is the technique's own number.

Configuration
=============

Diagnostics carry no YAML configuration of their own. Each technique's confidence formula —
which references diagnostic-attribute names by string — lives under
``techniques.<TechniqueName>`` in
``src/spindoctor/config_files/config_510_techniques.yaml``; see :doc:`dev_guide_techniques_confidence`
for the YAML schema.

Implementation
==============

Source file: ``src/spindoctor/nav_technique/diagnostics.py``.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :class:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.ncc_peak`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.peak_to_runner_up_ratio`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.consistency_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.consistency_ratio`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.used_gradient`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyDiscDiagnostics.body_count`.

- :class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.lm_iterations`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.tukey_inlier_count`.

- :class:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`. Same shape as
  :class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` with
  ``visible_terminator_arc_fraction`` substituted for
  ``visible_limb_arc_fraction``, plus the basin second-opinion pair
  :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_distance_px`
  and
  :attr:`~spindoctor.nav_technique.diagnostics.BodyTerminatorDiagnostics.secondary_basin_cost_ratio`.

- :class:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_snr_inside_predicted_bbox`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.body_extent_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.blob_count`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.residual_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_angle_deg`,
  :attr:`~spindoctor.nav_technique.diagnostics.BodyBlobDiagnostics.max_phase_irregularity_factor`.

- :class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_summed`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_mean`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_median_max`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.edge_count`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1`.

- :class:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.ncc_peak`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.peak_to_runner_up_ratio`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.annulus_count`,
  :attr:`~spindoctor.nav_technique.diagnostics.RingAnnulusDiagnostics.used_gradient`.

- :class:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_triplets_evaluated`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.rotation_below_separability_floor`.

- :class:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarUniqueMatchDiagnostics.detection_peak_ratio`.

- :class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.n_stars_used`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.median_pos_err_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics.residual_scatter_px`.

- :class:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`. Fields:
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.sun_angle_deg`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.axis_degenerate`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.phase_deg`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.envelope_diameter_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.cross_track_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.along_track_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_peak_score`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_valid_fraction`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.symmetry_second_peak_ratio`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.theta_refined_deg`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_rays_total`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_rays_inlier`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_inlier_fraction`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.arc_residual_rms_px`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.fitted_haze_radius_km`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.filters`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.recentered`,
  :attr:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics.gate_failed`. The peak score
  and the residual RMS are ``None`` rather than zero when the fit could not measure them, so a
  falling confidence sigmoid cannot read an unmeasured quantity as a perfect one.

- :class:`~spindoctor.nav_technique.diagnostics.ManualNavDiagnostics` — emitted by
  :class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual`. Single field:
  :attr:`~spindoctor.nav_technique.diagnostics.ManualNavDiagnostics.operator_accepted`, kept
  explicit so the JSON metadata records that a human, not an autonomous technique, set the
  offset.

The module also exports the
:data:`~spindoctor.nav_technique.diagnostics.NavTechniqueDiagnostics` union type spanning every
per-technique dataclass; the orchestrator's curator and
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` both consume this union.
Adding a new technique means adding both its diagnostics dataclass and a new entry into the
union.

Examples
========

**Curator allow-list discipline.**  Each diagnostics dataclass declares a class-level
``CURATOR_FIELDS`` mapping that the curator walks at JSON-emit time. For
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics` the mapping is::

    CURATOR_FIELDS = {
        'visible_limb_arc_fraction': 'visible_limb_arc_fraction',
        'visible_arc_px': 'visible_arc_px',
        'dt_fit_rms_px': 'dt_fit_rms_px',
        'lm_iterations': 'lm_iterations',
        'tukey_inlier_count': 'tukey_inlier_count',
    }

A new field added to the dataclass without a corresponding entry trips
:func:`~spindoctor.nav_orchestrator.curator.assert_diagnostic_fields_present` at startup, which
raises :exc:`AssertionError` and fails the build before any image is processed.

**Confidence-formula reference.**  The YAML stanza for ``BodyLimbNav`` declares::

    techniques:
      BodyLimbNav:
        alpha0: 0.132
        terms:
          - feature: visible_limb_arc_fraction
            alpha: 1.068
          - feature: dt_fit_rms_px
            alpha: -1.303
          - feature: visible_arc_px
            alpha: 0.776
            divisor: 440.0
            cap_at: 1.0
        hard_zero_if:
          at_edge: true
          spurious: true

Each ``feature`` value, and each ``hard_zero_if`` key, names an attribute on
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics`. At config-load time
:func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` walks the spec
and confirms every name appears in
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`'s
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` allow-list.

**JSON sidecar field-by-field.**  A successful ``BodyLimbNav`` fit on a Cassini image
produces a per-technique block in the per-image JSON sidecar of the form::

    {
      "technique_name": "BodyLimbNav",
      "feature_ids": ["limb_arc:DIONE"],
      "offset_px": [11.0, 29.5],
      "covariance_px2": [[0.0156, 0.0017], [0.0017, 0.0148]],
      "confidence": 0.675,
      "spurious": false,
      "at_edge": false,
      "diagnostics": {
        "visible_limb_arc_fraction": 0.85,
        "visible_arc_px": 120.0,
        "dt_fit_rms_px": 0.4,
        "lm_iterations": 5,
        "tukey_inlier_count": 118
      }
    }

Every key under ``"diagnostics"`` corresponds to a non-``None``-valued entry in the
``CURATOR_FIELDS`` mapping for
:class:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics`; nothing else surfaces in the
sidecar.
