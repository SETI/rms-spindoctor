================================================================
Image library — operator guide for adding sidecar entries
================================================================

This directory is the registry. Every YAML file at
images/<scene_class>/<image_id>.yaml enrolls one image in the
regression library. The directory tree IS the registry — there is
no manifest file.

Two test layers consume the library:

  1. A fast structural-invariants test that validates every YAML
     against the schema below.
  2. A per-image regression test that runs the autonomous
     orchestrator against the live PDS3 holdings and asserts the
     result against the expected.* fields. This requires
     PDS3_HOLDINGS_DIR to be set; otherwise it is skipped.

A separate baseline layer at images/../baselines/<image_id>.json
records the exact rounded (offset_dv_px, offset_du_px, confidence)
the orchestrator produces; baselines are mechanical and refreshed
via the developer tool at tests/integration/update_baselines.py
(invoke as `python -m tests.integration.update_baselines`).

----------------------------------------------------------------
Workflow
----------------------------------------------------------------

  1. Pick a candidate image and figure out which scene class it
     belongs to (see "Scene classes" below). When in doubt, pick
     the class that exercises the technique you most want to test.
     Each class directory has its own README.txt describing what
     scenes belong there.

  2. Open the manual-navigation dialog on the candidate:

         sd_offset [args] --manual

     where [args] are whatever selection / dataset / config flags
     pin the run down to a single image (dataset id, an image-list
     file, --config, --pds3-holdings-dir). The flag fails fast if
     the selection resolves to more than one image.

  3. Pick the offset by hand. Drag the overlay until limbs,
     terminators, ring edges, or star centroids line up at sub-
     pixel precision. The Auto button runs the same masked-NCC
     pyramid that the autonomous correlate techniques use; you can
     accept it as a starting point.

     Convention: the *predicted* position (v, u) plus the chosen
     offset (dv, du) equals the *actual* position in the image.

  4. Click "Save as Library Entry...". Point the file-save dialog
     at images/<scene_class>/<image_id>.yaml. The dialog also
     writes a companion <image_id>.png next to the YAML showing
     the red-image / green-model overlay at the chosen (dv, du);
     it is an orientation aid and is not consumed by any test.

  5. Open the saved YAML and replace every TODO_REPLACE_*
     placeholder using the field reference below.

  6. Validate the schema (no holdings needed):

         pytest tests/integration/test_image_library.py -k <image_id>

     Iterate until it passes.

  7. (Optional, requires holdings) Run the autonomous regression:

         pytest tests/integration/test_autonomous_nav.py -k <image_id>

     Tier-check failures during library expansion are *expected*
     until the calibration sweep tunes the per-technique alpha
     coefficients against the full library. Treat the failures as
     informational — they are the calibration input.

  8. (Optional, requires holdings) Seed or refresh the baseline:

         python -m tests.integration.update_baselines --image-id <image_id>

----------------------------------------------------------------
YAML schema
----------------------------------------------------------------

  schema_version: 1
  mission: <one of the ALLOWED_MISSIONS values below>
  camera:  <one of the ALLOWED_CAMERAS values below>
  image_id: <opaque string, must match filename stem>
  image_datetime_utc: <UTC ISO 8601 string | omitted>
                                   # from et_to_utc(obs.midtime)
  exposure_time_sec: <float | omitted>   # seconds; from obs.texp
  filter_combo: <sorted, '+'-joined filter list, e.g. 'CL1+CL2'>
  image_url: <opaque URL, e.g. pds3://... or https://...>

  scene_tags:                        # ordered list, length >= 1
    - <primary_class>                # MUST equal directory name
    - <optional secondary tag>       # body name, morphology, etc.

  ground_truth:
    offset_dv_px: <float>            # operator-verified offset, V
    offset_du_px: <float>            # operator-verified offset, U
    offset_uncertainty_px: <float>   # 1-sigma; > 0
    source: operator_verified
    operator: <username>             # required, non-empty string
    verified_date: YYYY-MM-DD
    ui_version: <spindoctor version>
    notes: |                         # optional multi-line string
      <one-line rationale: which feature you used to verify>

  expected:
    status: <ok | failed | conflicted>
    confidence_tier: <high | medium | low | failed | conflicted>
    primary_technique: <NavTechnique class name>
    techniques_must_run:             # optional list of technique
      - <name>                       # names that MUST appear in
                                     # per_technique results
    techniques_must_skip:             # optional list of technique
      - <name>                       # names that MUST NOT appear

  camera_rotation_expected:          # optional whole block
    rotation_deg: <float | null>
    uncertainty_deg: <float | null>

----------------------------------------------------------------
Field reference
----------------------------------------------------------------

image_id
  Opaque identifier; must match the filename stem (foo.yaml ->
  image_id 'foo'). Conventionally the PDS3 product ID, with any
  pipeline suffix kept (e.g. '_CALIB' for Cassini calibrated I/F).

mission     ALLOWED_MISSIONS
  COISS | VGISS | GOSSI | NHLORRI

  These match the dataset names registered in the rms-nav CLI
  (``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri``) upper-cased so a
  sidecar's mission is unambiguous against an invocation like
  ``sd_offset --dataset coiss``.

camera      ALLOWED_CAMERAS
  NAC | WAC | SSI | NA | WA | LORRI

filter_combo
  Filters applied for the exposure, sorted alphabetically and
  joined with '+'. Examples: 'CL1+CL2', 'CL+CL', 'BL1+CL2'. Use
  the same canonicalization that mag_offset_table keys use.

image_datetime_utc
  Optional. UTC ISO 8601 timestamp of the observation midtime,
  derived from obs.midtime via et_to_utc. When present, must be
  a non-empty string. Legacy sidecars written before the field
  was introduced may omit it.

exposure_time_sec
  Optional. Exposure duration in seconds, taken from obs.texp.
  When present, must be a finite positive number. Legacy sidecars
  written before the field was introduced may omit it (the field
  is permitted to be absent or null).

image_url
  Opaque URL. The 'pds3://' scheme is implicit relative to
  PDS3_HOLDINGS_DIR; 'https://' / 'gs://' / 'file://' work for
  one-off frames. Resolution happens at test time. The manual-nav
  dialog auto-populates this with the 'pds3://' prefix whenever the
  image lives under PDS3_HOLDINGS_DIR (or
  config.environment.pds3_holdings_root).

scene_tags
  scene_tags[0] MUST exactly equal the directory the file lives
  in. Subsequent entries are free-form and used by reviewers
  (typical second tag: body name; typical third tag: morphology
  qualifier like 'crescent', 'edge_on', 'ansa').

ground_truth.offset_dv_px / offset_du_px
  Operator-verified pixel offset. Convention: predicted_vu +
  offset = actual_vu.

ground_truth.offset_uncertainty_px
  1-sigma operator uncertainty in pixels (per axis). Practical
  values:
    1.0  - sharp limb, bright unsaturated stars, definitive edges.
    2.0  - soft features (high-phase terminator, BLOB-regime body,
           faint stars, rank-1 ring edge).
    3.0+ - extremely soft fits; if you would write 4 px, reconsider
           whether the image belongs in the library.
  CI tolerance is offset_uncertainty_px + 0.5 px slack on each
  axis.

ground_truth.source
  Always operator_verified. Cross-image inference is not
  supported; every offset comes from manually navigating that
  specific image.

ground_truth.operator
  Username of the human who picked the offset.

ground_truth.verified_date
  ISO date (YYYY-MM-DD) the offset was last verified.

ground_truth.ui_version
  rms-nav version string at the time of verification.

ground_truth.notes
  Optional one-line human-readable rationale: which feature you
  used to verify the alignment ("limb fit on Rhea limb, no
  rings"; "star centroid on HIP 12345 at lower-right").

expected.status     ALLOWED_STATUSES
  ok          - scene is navigable; offset is the load-bearing
                output. Use for almost every sidecar.
  failed      - scene is unnavigable. ONLY for the
                negative_cases/ directory.
  conflicted  - two disjoint ensembles disagree by more than their
                combined uncertainty; the orchestrator hard-sets
                rank='conflicted'. Reach for this only when the
                scene is *designed* to make ensembles fight.

expected.confidence_tier     ALLOWED_TIERS
  high        - sharp limb fully in frame, dense bright stars,
                multiple unambiguous features. Calibrated sigma
                expected below 0.5 px.
  medium      - partial limb, soft terminator, a few stars.
                Calibrated sigma 0.5-1.5 px.
  low         - barely navigable: tiny body, faint stars, marginal
                SNR. Calibrated sigma above 1.5 px but a clear fit.
  failed      - MUST pair with status='failed'.
  conflicted  - MUST pair with status='conflicted'.

  Tier is a calibration TARGET, not a description of current
  pipeline behavior. The calibration sweep tunes the per-
  technique alpha coefficients to make the autonomous result land
  the tier you write here. Pick conservatively: a 'medium' that
  comes back 'high' after calibration is a free win you can
  revise upward; a 'high' that the calibrated pipeline can never
  reach biases the fit across every other image. If you would
  be unsurprised either way, write 'medium'.

expected.primary_technique     NAV_TECHNIQUES
  The technique you expect to win pass-1. The autonomous
  orchestrator's tie-break is (-confidence, technique_name)
  ascending, so when two techniques produce comparable confidence
  the alphabetically earlier name wins.

  Quick map (scene -> primary):
    body fills FOV               -> BodyDiscCorrelateNav
    body partly off-frame, limb  -> BodyLimbNav
    crescent (phase > 90)        -> BodyTerminatorNav
    irregular body close-in      -> BodyBlobNav
    body diameter < 15 px        -> BodyBlobNav
    curved ring edge             -> RingEdgeNav
    flat ring edge (rank-1 only) -> RingEdgeNav (or RingAnnulusNav)
    >= 3 catalog stars           -> StarFieldFromCatalogNav
    1 unambiguous star           -> StarUniqueMatchNav
    2 unambiguous stars          -> StarUniqueMatchNav
    body + >= 3 stars            -> highest of body / star
                                    techniques (often
                                    StarFieldFromCatalogNav after
                                    StarRefineNav pass 2)
    rings + body                 -> highest of RingEdgeNav vs body
                                    technique

  All recognized technique names:

    BodyDiscCorrelateNav
    BodyLimbNav
    BodyTerminatorNav
    BodyBlobNav
    RingEdgeNav
    RingAnnulusNav
    StarFieldFromCatalogNav
    StarUniqueMatchNav
    StarRefineNav
    TitanHazeNav

  Names not yet implemented (do not use):

    CartographicNav     (deferred; cartographic-mosaic correlate)

expected.techniques_must_run
  Optional list. Each name MUST appear in the per_technique
  result list when the autonomous test runs; otherwise it fails.
  Default: [<primary_technique>] only. Add a technique only when
  the scene clearly enables it (e.g. limb + correlate on a 70%-
  visible body, two ring edges plus a moon). Each addition is an
  independent regression-failure mode.

expected.techniques_must_skip
  Optional list. Each name MUST NOT appear in the per_technique
  result list — i.e. the feasibility gate must reject the
  technique outright. Use to assert infeasibility, not low
  confidence. Examples:
    body_full_fov            -> [StarFieldFromCatalogNav]
    one_bright_star_no_body  -> [BodyLimbNav, BodyDiscCorrelateNav]
    ring_only_curved         -> [BodyLimbNav]
    negative_cases           -> typically empty

  techniques_must_run and techniques_must_skip cannot overlap.

camera_rotation_expected
  Optional block; only meaningful when the per-camera config has
  fit_camera_rotation: true (currently Galileo SSI and Voyager
  ISS). Either subfield may be null.

----------------------------------------------------------------
Scene classes
----------------------------------------------------------------

The directory under images/ that an entry lives in MUST be one of
the following 17 declared scene classes. Each has its own README
under that directory describing in detail what scenes belong
there.

  star_dominated              >= 3 catalog stars, no body, no rings
  body_full_fov               regular body fills >= 70% of FOV
  body_partial_overflow       regular body 70-90% in frame
  body_mostly_offscreen       regular body 50-90% off frame, limb
                              arc visible
  body_irregular              irregular body in BLOB-uncertainty
                              regime
  multi_body                  >= 2 separable bodies in FOV
  ring_only_curved            curved ring edge, no body
  ring_only_flat              flat ring edge (rank-1 fit), no body
  ring_plus_body              ring edge + >= 1 moon
  stars_plus_body             body + >= 3 catalog stars
  one_bright_star_no_body     exactly 1 unambiguous star, no body,
                              no rings
  two_bright_stars_no_body    exactly 2 unambiguous stars, no body,
                              no rings
  faint_stars                 every catalog star below SNR threshold
  scattered_light             Galileo / Voyager stray-light gradient
  high_phase_terminator       crescent geometry, phase > 90 degrees
  below_resolution_body       body diameter < 15 px (BLOB regime)
  negative_cases              expected.status='failed' on purpose

If a candidate sits between two classes, pick the one that
exercises the *expected primary technique* and document the
judgment in ground_truth.notes.

----------------------------------------------------------------
Schema constraints enforced at validation time
----------------------------------------------------------------

  - schema_version must equal 1.
  - mission, camera, status, confidence_tier, source must each be
    one of the enum values listed above.
  - scene_tags must be a non-empty list of strings; first entry
    must equal the parent directory name; no duplicates.
  - offset_uncertainty_px must be > 0.
  - status='failed' requires confidence_tier='failed', and
    vice-versa.
  - status='conflicted' requires confidence_tier='conflicted',
    and vice-versa.
  - techniques_must_run and techniques_must_skip cannot share any
    entry.
  - filename stem must equal image_id.
  - All numeric fields must be finite (no NaN, no Infinity).
  - verified_date must parse as a real calendar date.
