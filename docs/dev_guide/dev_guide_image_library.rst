==================
Image Library
==================

The image library is the operator-curated regression suite at
``tests/integration/image_library/``. It serves three roles at once:

- **Regression cohort** — every shipping commit re-navigates every library
  image and asserts the orchestrator's reported offset stays within the
  per-image tolerance budget. Drift is caught at commit time.
- **Calibration cohort** — the per-technique confidence formulas in
  ``config_510_techniques.yaml`` are tuned against the library so that the
  orchestrator's reported confidence_tier matches the operator-assigned
  tier on every image. Without the library there is no objective signal to
  tune the confidence formulas against.
- **Coverage map** — the scene-class taxonomy (``body_full_fov``,
  ``ring_only_curved``, ``star_dominated``, ``below_resolution_body``,
  ``high_phase_terminator``, etc.) makes coverage gaps visible at a glance:
  a regime with zero entries is a regime nobody has hand-calibrated, and
  the orchestrator's behavior there is unverified.

Each entry is a YAML *sidecar* that records:

* the image's mission / camera / filter combo,
* an opaque ``pds3://`` URL resolved through ``PDS3_HOLDINGS_DIR``,
* the operator-verified ground-truth offset and its 1sigma uncertainty,
* the expected ``status``, ``confidence_tier``, and primary technique,
* and the set of techniques that must (or must not) run on the scene.

Two test layers consume it: a fast structural-invariants test
(:file:`tests/integration/test_image_library.py`) and a slow per-image
regression test (:file:`tests/integration/test_autonomous_nav.py`),
gated by the ``integration`` pytest marker.

Layout
======

The directory tree IS the registry — there is no ``manifest.yaml``::

   tests/integration/image_library/
     images/
       body_mostly_offscreen/
         W1521598221_1_CALIB.yaml
       ring_only_curved/
         N1601122100_1.yaml
       ...

Adding a sidecar to the right scene-class directory enrolls it in CI
automatically; removing a sidecar stops its tests. The set of
scene-class subdirectories is checked against the master list in
:data:`tests.integration.sidecar.DECLARED_SCENE_CLASSES`, so
typos like ``body_overflow`` vs ``body_partial_overflow`` fail loudly
at collection time.

Sidecar schema (schema_version 1)
=================================

.. code-block:: yaml

   schema_version: 1
   mission: COISS                     # COISS | VGISS | GOSSI | NHLORRI
   camera: WAC                        # NAC | WAC | SSI | NA | WA | LORRI
   image_id: W1521598221_1_CALIB
   image_datetime_utc: '2006-04-26T08:32:14.123Z'
                                      # UTC ISO 8601; from et_to_utc(obs.midtime)
   exposure_time_sec: 0.46            # seconds; from obs.texp
   filter_combo: 'CL+VIO'             # canonicalized: filters sorted, '+'-joined
   image_url: 'pds3://volumes/COISS_2xxx/COISS_2021/.../W1521598221_1_CALIB.IMG'

   scene_tags: [body_mostly_offscreen, rhea]
                                      # First tag is the primary class;
                                      # must match the directory name.

   ground_truth:
     offset_dv_px: 12.5
     offset_du_px: -3.25
     offset_uncertainty_px: 1.0       # 1sigma; the test's tolerance budget
     source: operator_verified
     operator: rfrench
     verified_date: 2026-04-28
     ui_version: 'rms-spindoctor 0.1.dev0'
     notes: |
       Hand-verified limb fit, no rings in the FOV.

   expected:
     status: success                  # success | failed | conflicted
     confidence_tier: high            # high | medium | low | failed
     primary_technique: BodyLimbNav
     techniques_must_run: [BodyLimbNav]
     techniques_must_skip: [StarFieldFromCatalogNav]

The full validator lives in :mod:`tests.integration.sidecar`; malformed
fields raise :class:`~tests.integration.sidecar.SidecarValidationError`
at collection time.

The calibration process
=======================

The library is the project's calibration substrate. This section describes
the end-to-end calibration loop that the library participates in.

Why the library is the right substrate
--------------------------------------

The autonomous navigation pipeline produces three pieces of output per
image: an offset, a covariance, and a calibrated confidence rank
(``high`` / ``medium`` / ``low`` / ``failed``). The first two are
testable against ground truth; the third is **defined** by ground truth
— a "high"-confidence verdict means "in the regime where the operator
hand-verified the offset to ~1 px and the technique reported
self-consistent diagnostics, the answer is right ≥ 95 % of the time."
That definition has no meaning without a cohort of images on which the
operator actually hand-verified the offset. The library is that cohort.

The same cohort drives:

- **Per-technique confidence-formula coefficients.**  Each
  :class:`~spindoctor.nav_technique.nav_technique.NavTechnique`
  carries a small set of calibrated coefficients in
  ``techniques.<TechniqueName>.confidence`` in
  ``config_510_techniques.yaml``. The coefficients map per-technique
  diagnostics (correlation peak height, NCC margin, fit residuals, ...)
  into a [0, 1] confidence score. The coefficients are tuned offline so
  that, across the library, the score correlates monotonically with
  per-image correctness.
- **Per-instrument photometric thresholds.**  Per-instrument
  ``mag_offset``, ``noise.sigma_floor``, ``image_quality_thresholds``,
  and ``source_image_filter`` defaults are sanity-checked against
  library images for each instrument; values that fail to reproduce the
  hand-verified offset on a representative cohort are revisited. See
  :doc:`dev_guide_config_and_static_data` for the citation discipline.
- **Per-technique runtime tunables.**  Spurious-detection thresholds,
  at-edge tolerances, minimum arc lengths, ring-edge detectability
  cutoffs — every numeric knob in ``config_510_techniques.yaml`` was
  picked because it was the value that made the library pass while
  staying conservative on regimes outside the library's coverage.

The calibration loop
--------------------

A change to the navigation pipeline that affects per-image output goes
through this loop:

1. **Land the change behind the regression suite.**  The change is
   developed on a branch. The author runs ``pytest -m integration -k
   <relevant_image_ids>`` against ``$PDS3_HOLDINGS_DIR`` to see which
   library entries it shifts. A pure refactor should shift nothing; a
   tuning change shifts a known cohort.
2. **Inspect the diff per image.**  For each shifted image, the author
   re-runs ``sd_offset --manual <image>`` and visually verifies that
   the recomputed offset still overlays the limb / star field / ring
   edge. When the recomputed offset is *better* than the operator-stored
   ground truth — for example a fix produces a sub-pixel offset where
   the prior implementation reported 1.5 px — the operator-stored
   ground truth is updated in the same PR (a fresh
   ``Save as Library Entry...`` from the manual nav dialog rewrites the
   sidecar's ``ground_truth`` block; ``operator``, ``verified_date``,
   and ``ui_version`` get refreshed automatically).
3. **Update the regression baselines.**  Once the per-image
   ground-truth review is complete, the author runs
   ``python -m tests.integration.update_baselines --image-id <ids>`` to
   refresh the byte-stable baseline JSONs for the images that shifted.
   The diff that update emits goes into the same PR as the code change.
4. **Re-tune calibrated confidence if needed.**  If the change shifts
   the per-image confidence score (not just the offset), the author
   re-runs the offline confidence-tuning script. (Automated tooling
   under ``tests/integration/calibrate_confidence.py`` is reserved as a
   slot pending a wider library cohort; the workflow uses ad-hoc
   per-technique tuning.) The per-technique ``confidence`` coefficient
   block in ``config_510_techniques.yaml`` updates accordingly.
5. **Reviewer sign-off.**  PRs that touch any of (a) library sidecars,
   (b) baseline JSONs, or (c) per-technique coefficients require a
   reviewer to manually open at least one shifted image's summary PNG
   and verify the overlay still tracks the data. This is the
   "operator-in-the-loop" gate the calibration substrate cannot replace.

Coverage taxonomy
-----------------

The directory names under ``tests/integration/image_library/images/``
constitute the scene-class taxonomy. Each name encodes one
information-bearing regime the orchestrator must handle. The shipping
classes (mirrored in
:data:`tests.integration.sidecar.DECLARED_SCENE_CLASSES`) include:

- **Body geometry** — ``body_full_fov``, ``body_partial_overflow``,
  ``body_mostly_offscreen``, ``body_overlapping``,
  ``below_resolution_body``, ``multi_body``,
  ``high_phase_terminator``.
- **Ring geometry** — ``ring_only_curved``, ``ring_only_straight``,
  ``ring_with_body``, ``ring_below_resolution`` (annulus regime).
- **Star regimes** — ``one_bright_star_no_body``, ``star_dominated``,
  ``star_field_with_body``, ``star_field_with_ring``.
- **Failure regimes** — ``empty_fov``, ``saturated``,
  ``mostly_dropouts``, ``cosmic_ray_dense``.

A regime with **zero** library entries is a regime where the
orchestrator's behavior is unverified. When a code path is added —
say, a new technique — the contributor adds at least one library image
per regime that exercises the path. The
:doc:`dev_guide_extending` chapter's checklist enumerates which regimes
each subsystem needs.

Confidence-tier semantics
-------------------------

The four tiers are calibrated to operator expectations:

- ``high`` — operator-verified offset to ≤ 1 px on a sharp-feature
  image; orchestrator reports a sharp NCC peak and tight covariance;
  multiple techniques agree. Bound: the per-image error is below the
  operator-supplied ``offset_uncertainty_px`` essentially every time.
- ``medium`` — operator-verified offset to ≤ 2 px on a
  soft-feature / star-poor image, or an image where one technique
  dominates without cross-technique corroboration.
- ``low`` — operator-verified offset to ≤ 4 px on a degenerate-geometry
  scene (rank-1 ring fits, partial limb arcs without curvature).
  The pipeline reports the offset and the operator decides whether
  to trust it; the bundle annotates the data label accordingly.
- ``failed`` — no usable techniques fired, or every technique reported
  spurious / at-edge. No offset is reported.

The library's ``expected.confidence_tier`` field is the calibration
target; a tier mismatch on regression always fails (no slack — tier
*is* the calibration target). The per-axis offset budget is
``offset_uncertainty_px + 0.5 px`` slack.

Per-instrument calibration anchors
----------------------------------

Each per-instrument config (``config_4N0_inst_*.yaml``) includes a
small set of calibration-anchor library images that exercise the
instrument's per-image quirks. Examples:

- COISS NAC: at least one calibrated-IF image, one CALIB-stage image,
  one heavily-saturated image (to exercise the saturation mask), one
  cosmic-ray-dense image.
- VGISS NA / WA: at least one image with the known per-camera
  geometric distortion residual (the per-instrument
  ``geometric_distortion`` correction in the per-instrument config
  was tuned against this image).
- GOSSI / NHLORRI: at least one calibrated image and one raw image to
  cross-check the ``calibration=False`` default in the LORRI loader.

These anchors are the minimum coverage; the broader per-regime cohort
above is what catches drift. When a per-instrument calibration value
changes, every per-instrument anchor must still pass.

Adding a new entry
==================

The recommended path is the manual-navigation dialog's
**Save as Library Entry...** button:

1. Run the manual-nav dialog on the candidate image with the
   ``sd_offset [args] --manual`` CLI flag, where ``[args]`` are the
   selection / dataset / config flags that pin the run down to a
   single image (e.g. dataset id, an image-list file, ``--config`` for
   a non-default bundle).
2. Pick the offset by hand (or accept the **Auto** result).
3. Click **Save as Library Entry...**. A file-save dialog suggests
   ``<image_id>.yaml`` as the filename — point it at the right
   scene-class directory under
   ``tests/integration/image_library/images/<class>/``. The dialog
   also drops a companion ``<image_id>.png`` next to the YAML showing
   the red-image / green-model overlay at the chosen ``(dv, du)``;
   it's an orientation aid for future reviewers and is not consumed
   by any test.
4. Open the saved YAML and replace every ``TODO_REPLACE_*`` placeholder
   (scene_tags, primary_technique, notes, etc.).
5. Re-run ``pytest tests/integration/test_image_library.py`` to check
   the schema; iterate until it passes.
6. With ``PDS3_HOLDINGS_DIR`` set, run
   ``pytest tests/integration/test_autonomous_nav.py -k <image_id>`` to
   check the offset assertion against the live orchestrator.

Tolerance regimes
=================

.. list-table::
   :header-rows: 1
   :widths: 50 20 30

   * - Source
     - Typical uncertainty (px)
     - Use when
   * - ``operator_verified``, sharp limb / bright stars
     - 1.0
     - majority of cases
   * - ``operator_verified``, soft features / star-poor
     - 2.0
     - high-phase / soft-edge / faint-star

The CI test tolerance is ``offset_uncertainty_px + 0.5 px`` slack on
each axis. ``confidence_tier`` mismatches always fail (no slack — tier
is part of the calibration target).

Regression baselines
====================

In addition to the per-sidecar tolerance test, a separate baseline
layer records the *exact* rounded ``(offset_dv_px, offset_du_px,
confidence)`` triple in
``tests/integration/baselines/<image_id>.json``. Comparison is
exact-equal on rounded values (``offset`` to 4 decimals, ``confidence``
to 3); the baseline schema deliberately omits
``pipeline_run_iso8601`` because that is the only provenance field that
is *not* byte-identical between identical runs.

What checks the baselines
-------------------------

Two tests under ``tests/integration/test_baselines.py``:

- ``test_every_baseline_cites_a_sidecar`` — runs in the fast suite (no
  holdings needed). Asserts that every ``baselines/<image_id>.json``
  has a matching sidecar at
  ``image_library/images/*/<image_id>.yaml``, and that the file's stem
  matches the baseline's ``image_id`` field. Catches the common drift
  where a sidecar is renamed or deleted but its baseline lingers.
- ``test_regression_baseline_exact_match`` — gated by the
  ``integration`` pytest marker and skipped when ``PDS3_HOLDINGS_DIR``
  is unset. Parametrized one case per ``(baseline, sidecar)`` pair;
  runs the orchestrator against the real holdings, calls
  :meth:`tests.integration.baseline.Baseline.from_run` to round the
  fresh outputs, and asserts ``actual == expected`` (exact equality on
  all four keys). The failure message tells the operator to update
  the JSON in the same PR if the diff is intended.

Plus a handful of round-trip / serialization unit tests on
:meth:`tests.integration.baseline.Baseline.from_run` and
:meth:`tests.integration.baseline.Baseline.to_json` that pin the rounding
rule and confirm byte-stable JSON (sorted keys, trailing newline).

How a baseline is created or updated
------------------------------------

Use the developer tool at
:file:`tests/integration/update_baselines.py`. It is intentionally
not packaged as a user-facing CLI — invoke it from a project checkout
as a Python module so the test stack imports resolve naturally. It
refuses to run without ``PDS3_HOLDINGS_DIR`` set.

.. code-block:: bash

   python -m tests.integration.update_baselines --image-id <image_id>      # one image
   python -m tests.integration.update_baselines --image-id A --image-id B  # hand-picked batch
   python -m tests.integration.update_baselines --all                      # every sidecar
   python -m tests.integration.update_baselines --all --dry-run            # preview only

For each image the tool runs
:func:`~spindoctor.navigate_image_files.navigate_image_files` against the live
holdings, rounds the result via
:meth:`tests.integration.baseline.Baseline.from_run`, compares against
any existing baseline, and reports one of:

* ``CREATE`` — no on-disk baseline; new file written.
* ``UPDATE`` — baseline drifted; old → new diff printed and the file
  overwritten.
* ``UNCHANGED`` — bytes match; file untouched (mtime preserved).
* ``FAILED`` — orchestrator returned no offset, or the requested
  ``--image-id`` matched no sidecar.

The exit code is ``0`` when every selected image succeeded (regardless
of write/update/unchanged), ``1`` when at least one ``FAILED`` line
was emitted, ``2`` on argument-parsing errors or when
``PDS3_HOLDINGS_DIR`` is unset.

Sidecars must land first — the
``test_every_baseline_cites_a_sidecar`` invariant refuses orphan
baselines. Baseline updates always require explicit operator review
on the PR; the CLI is the mechanical step, but the human review of
the resulting diff (does the new offset still overlay the limb?  does
the new confidence still match ``expected.confidence_tier``?) is what
keeps the regression layer trustworthy.
