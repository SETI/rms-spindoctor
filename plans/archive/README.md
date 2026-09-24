# Archived plans

Historical planning records, kept for reference. The live plans are in the
parent `plans/` directory: `PROGRAM_PLAN.md` is the top-level plan of record;
`VALIDATION_AND_CALIBRATION_PLAN.md` (Track A science detail),
`ENGINEERING_PLAN.md` (Tracks B-F implementation detail), and
`COHORT_CURATION_PLAN.md` (image-library growth playbook) are its detail
layers, and `OPERATOR_PLAYBOOK.md` is the operator's dispatch sheet over all
of them. Archived documents may reference each other and pre-archive paths
(e.g. `plans/ROADMAP.md`, `plans/PHASE10_CURATION.md`); resolve such
references against the dated filenames in this directory.

- `AUTONAV_PLAN_2026-06-19.md` — design record and phase plan for the
  autonomous-navigation core rewrite (merged as of commit 77a90f4). Phases 0-9
  shipped as specified; the remaining work migrated to GitHub issues. Parts
  1-16 remain the design reference for the shipped architecture (read with the
  document's own precedence rules: the Part 0 header overrides stale body
  text).
- `SIM_IMPROVEMENT_PLAN_2026-06-19.md` — simulator improvement plan, executed
  except for items deferred with issue numbers (#151, #152, #153). Section 0.1
  is the authoritative status board.
- `SIM_REALISM_PLAN_2026-07-18.md` — the simulator realism and
  de-circularization design (rev 2.1), fully executed 2026-07-16 to
  2026-07-18: all ten phases (A-J) plus the final sweep merged through the
  `rf_sim_realism` integration branch, all acceptance criteria (its
  Section 11) verified met. The as-built system is documented in
  `docs/dev_guide/dev_guide_simulator.rst` (including the capability
  envelope), the calibration in `util/calibration/CAMPAIGN_20260718.md`,
  and the independent assessment in
  `critiques/archive/SIM_REALISM_CRITIQUE_2026-07-18.md`; follow-up work moved to
  GitHub issues (#301, #309, #310, #311).
- `TITAN_NAV_PLAN_2026-07-25.md` — the haze solar-symmetry navigation design
  (the French method), revision 12, fully executed 2026-07-25 to 2026-07-29:
  all six phases (A-F) merged through the `rf_titan_nav` branch as PR #408,
  closing #60, with every acceptance criterion in its Section 8 met. Titan
  frames navigate autonomously to a published bound of 1 px cross-track and
  3 px along-track, validated on an 82-frame Cassini cohort and a 700-scene
  planted-truth campaign. The as-built system is documented in
  `docs/dev_guide/dev_guide_techniques_titan_haze.rst` and the two
  `dev_guide_navigation_models_titan*.rst` pages; the method analysis that
  selected the approach is frozen at
  `critiques/archive/TITAN_NAV_CONCEPT_2026-07-25.md` and the seven review
  rounds at `critiques/archive/TITAN_NAV_PLAN_CRITIQUE_2026-07-25*.md` and
  `critiques/archive/TITAN_NAV_COLLATERAL_SWEEP_2026-07-25.md`. Its Section 9
  deferred work is filed as issues (#397, #398, #399, #400, #401, #402,
  #403, #404, #405), with the operator ratification bundle on #407.
- `LOGGING_REDESIGN_PLAN_2026-08-04.md` — the two-logger design: one main
  logger per program run and one image logger per image, a single per-module
  level system, a top-level `logging` configuration section with per-program
  overrides, and one identical command-line surface across every pipeline
  program. Fully executed through the `rf_logging_redesign` integration
  branch and merged 2026-08-04 as PR #425 (commit `ac690ea`), all nine
  phases plus a whole-branch adversarial review, every acceptance criterion
  in its Section 5 met — including cloud-task drivers writing zero bytes to
  the worker terminal and no PdsLogger anywhere able to reach pdslogger's
  `print()` fallback. The as-built system is documented in
  `docs/user_guide/user_guide_logging.rst` and
  `docs/dev_guide/dev_guide_logging.rst`. Its Section 7 follow-ups are filed
  as issues (#424, #427, #428, #429), with the defects the work surfaced on
  #418, #423 and #426.
- `CK_KERNEL_PLAN_2026-08-04.md` — the corrected-pointing C-kernel design,
  fully executed and merged 2026-08-09, closing #188. Its sections 1-3 remain
  the design reference for what shipped: the navigator records `cmatrix` /
  `cmatrix_original`, the frame identities and the exposure times in every
  image's metadata, and `sd_create_ck` writes one type-3 segment per navigated
  exposure into files mirroring the originals they correct, beside a
  meta-kernel and a per-mission CSV report. The as-built system is documented
  in `docs/user_guide/user_guide_ck_kernels.rst` and
  `docs/dev_guide/dev_guide_ck_kernels.rst`, which is what a reader wanting
  current behavior should read first. Its Section 7 follow-ups are filed as
  issues (#433, #434, #435, #436, #437, #440, #443, #444, #446, #448, #455,
  #468); its acceptance criterion 7 (90% suite coverage) did not hold at
  merge and is now tracked as #548.
- `CK_KERNEL_DESIGN_NOTE_2026-07-30.md` — the pre-decision design analysis
  for corrected-pointing C-kernels: six candidate designs ranked, the
  cross-cutting problems, and the decision record that adopted the overlay
  type-3 design. Superseded by `CK_KERNEL_PLAN_2026-08-04.md`, which carries
  the decided design forward as an implementation plan; this note remains the
  record of the alternatives considered and why they lost. One caution for
  a future reader: its offset-to-rotation sketch predates the discovery
  that the oops observation frames differ from the SPICE camera frames by
  constant flips, which the executed plan's frame handling addresses.
- `CMATRIX_READERS_PLAN_2026-08-09.md` — the reading half of #50, fully
  executed in one pull request over four phases and merged 2026-08-09: the
  backplane and reprojection stages apply the recorded C-matrix by frame
  replacement (`apply_cmatrix_to_obs` in `spindoctor/support/cmatrix.py`),
  with the pixel offset as the documented fallback for fitted-rotation,
  offset-only and malformed records, and uncorrected pointing with the reason
  recorded when neither is usable. Adversarially reviewed before
  implementation (`critiques/archive/CMATRIX_READERS_PLAN_CRITIQUE_2026-08-07.md`);
  its section 0 records where execution refined the letter of the plan. The
  pointing selection and application code it left in the reprojection CLI
  package is tracked for relocation as #520.
- `RESULTS_INDEX_PLAN_2026-08-04.md` — the optional, rebuildable index over
  the navigation results tree (authored as `RESULTS_DB_PLAN.md` and renamed
  with the program it builds), fully executed 2026-08-09 to 2026-08-25 across
  seven phases and twenty-one individually reviewed pull requests, merged to
  `main` as PR #551 (commit `e63a0d51`, a merge commit so the reviewed PRs
  survive in the history), closing #430, #487 and #507. What shipped: a
  stub-keyed schema, a SQLAlchemy Core layer over SQLite and PostgreSQL,
  incremental and cloud-task ingest, a `--results-index-db` opt-in on every
  consuming program, and one record seam (`spindoctor/nav_records/`,
  `spindoctor/results_index/`) that answers the same four questions over the
  results tree and over the index. The as-built system is documented in
  `docs/user_guide/user_guide_results_index.rst` and
  `docs/dev_guide/dev_guide_results_index.rst`. Two of its acceptance
  criteria openly do not hold and are tracked as #547 (no written product is
  compared between the two storages, because the one integration frame that
  would do it no longer navigates) and #548 (suite coverage is 79% against a
  stated floor of 90%, and nothing enforces it); its Section 7 follow-ups are
  filed as issues (#462, #464, #465, #466, #467, #472, #486, #493, #497,
  #501, #512, #513, #514, #515, #516, #524, #528, #531, #533, #534, #536,
  #538, #540, #542). One caution for an editor: this document is not inert.
  `tests/spindoctor/results_index/test_enumeration_lists.py` reads its Phase 5
  entry and its acceptance criterion 1 and compares both against the module
  docstring of `spindoctor.dataset.results_filter` and the navigation guide,
  so a member added to or removed from the index-versus-tree enumeration is
  edited into all four statements or into none.
- `RESULTS_DB_REUSE_NOTE_2026-07-31.md` — the pre-decision analysis for
  reusing the statistics database as a shared results index: the consumer
  survey, the gaps in the schema as it stood, four reuse designs, and the
  backend-selection options with the decision record. Superseded by
  `RESULTS_INDEX_PLAN_2026-08-04.md`, which carries the decided design
  (rebuildable index, JSON authoritative, SQLAlchemy Core with
  SQLite/PostgreSQL) forward as an implementation plan; this note remains the
  record of the alternatives and the operational trade-offs (ship-the-file vs
  server) behind them.
- `PDS4_DRAFT_BUNDLE_PLAN_2026-09-08.md` — the first PDS4 bundle the project
  can hand to the RMS Node for review: one Cassini ISS Saturn backplanes
  bundle, complete enough that the NASA PDS `validate` tool's only findings
  are the `TODO DOI` placeholders. Executed 2026-09-09 to 2026-09-18 across
  ten phases on the `rf_pds4_draft_bundle` integration branch, merged to
  `main` as PR #721 (commit `3e26383a`), closing #47, #66, #69, #71, #72,
  #73, #74, #75, #76, #265, #519, #601, #602 and #678. What shipped: the
  bundle product, the readme, the context, document, schema, miscellaneous
  and `spice_kernels` collections, conforming inventories, the backplane
  FITS as an archived file with a described data object, epochs and targets
  in the data labels, global index tables labeled in their own collection,
  the bundle name and version taken from configuration, and a `--check-only`
  integrity pass gated by a test. Its section 0.1 status board is the record
  of what each phase landed. The operator ruled on 2026-09-15 that the work
  finishes as a prototype over the synthetic cohort: the draft run over a
  real COISS volume, with registered DOIs and a fresh navigation, is #708,
  and it carries the cohort choice and the user-guide PDF with it. Part B of
  Phase 8, the `cassini:ISS_Specific_Attributes` mission area, is #717, now
  that #698 records the label metadata in the observation block. This plan
  is the "finish and validate the Cassini path" half of #53; generalizing to
  Voyager, Galileo and New Horizons is the other half and begins from the
  reference template tree this work produced. Its other open follow-ups are
  #600, #611, #614, #677, #687, #706, #710, #716 and #720.
- `ROADMAP_2026-07-12.md` — the issue-ordered, Cassini-first pipeline
  build-out that served as the plan of record until 2026-07-12; consolidated
  into `plans/PROGRAM_PLAN.md` together with the post-stack task inventory.
- `FULL_PROGRAM_AFTER_MANUAL_WORK_2026-07-11.md` — the six-track task
  inventory written after the 2026-07 PR stack (#208-#220) was prepared;
  consolidated into `plans/PROGRAM_PLAN.md` the following day. Its track
  letters (A-F) carried over to the live plan.
- `PHASE10_CURATION_2026-07-12.md` — the original operator playbook for the
  first-stage (49-image) library build. Its durable content moved to
  `docs/dev_guide/dev_guide_image_library.rst` (sidecar schema, tier rubric,
  baselines) and the appendix of `plans/COHORT_CURATION_PLAN.md` (scene-class
  budget, selection guide, mission hints); the automated Stage A-E workflow in
  that plan superseded its manual front half.
