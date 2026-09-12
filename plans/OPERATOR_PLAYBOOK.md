# Operator Playbook: driving the agreement study and the calibration finish

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` as of 2026-08-27. Work through Section 0 first; after
it, independent and unblocked work can be dispatched in parallel as agent
sessions. Where a section states an order or a dependency -- Section 0.3's
groups, or anything gated on #288 -- that order governs.
Environment for every command below: `source /seti/newnav/setup.sh` from
`/seti/newnav/rms-nav` (the venv is `venv/`). A venv installed before
2026-08-25 needs `venv/bin/pip install -e ".[dev]"` once, because
`sqlalchemy>=2.0` became a runtime dependency.*

## 0. Right now (operator-only, minutes)

### 0.1 The pending decisions (comment on the issues)

Each is a scope commitment the downstream work waits on:

- **#316 ring orbit-uncertainty severity**: ship at the conservative
  default, ratchet `rings.orbit_radial_sigma_correlated_fraction`, or
  implement the wander decomposition. It demotes five operator-verified
  Keeler frames to low, and it must be settled before the #230
  recalibration reads their tiers.
- **#407 Titan haze navigation ratification**: the method is implemented and
  validated to a stated 1 px cross-track / 3 px along-track bound. What needs
  your decision is a bundle: five mid-implementation specification changes
  (each recorded with the measurement behind it), three acceptance bounds the
  evidence argues with (a unit-test noise bound, the planted-truth z-score
  band, and the >= 90% consistency-pair bound measured at 83.3%), and three
  staged curation artifacts — a twenty-frame overlay review batch under
  `util/titan_cohort/review_batch/` with every vote null, six library
  nominations with draft sidecars under `util/titan_cohort/nominations/`, and
  a recommendation to add a `titan_haze` scene class. The issue enumerates
  each one.
- **#338 highly-irregular terminator fit (N1853392805)**: accept the
  2-px-class ground truth, keep TERMINATOR_ARC for SPICE-known synchronous
  rotators, or wait for shape models (#23).
- **#548 the coverage floor**: the suite measures 79%; two shipped plans'
  acceptance criteria assert 90%; nothing enforces either. Raise coverage to
  the stated floor, or ratify a lower one and gate CI on that. The shortfall
  is almost entirely PyQt6 widget code. Leaving both the claim and the
  absent gate in place is the one option that keeps asserting something
  untrue.
- **#547 the uncompared written product**: acceptance criterion 1 of the
  results-index work compares a built product between the two storages, and
  the one integration frame that would do it no longer navigates. Pick a
  frame that still navigates, or close it behind the image-library
  regression (#288).
- **#459 Cassini predicted kernels**: whether Cassini navigation should run
  from the predicted rather than the reconstructed kernels. There is a
  branch, `origin/rf_ck_cassini_predicted`, nine commits, last touched
  2026-08-07, with no pull request ever opened. Say whether it is live
  before it drifts further from `main`.
- **#468 New Horizons pointing family**: declare the merged family
  reconstructed on its comment-area evidence, or leave it `UNCLASSIFIED`.
- **#466 getting the index to cloud workers**: publish the SQLite file to
  the results bucket and have each worker download it once, or run a
  PostgreSQL instance. This one also decides whether #464's document column
  is affordable, since it roughly doubles the index size.

```bash
gh issue comment 316 --body "Decision: <ship default | ratchet fraction | wander decomposition>"
gh issue comment 407 --body "Decisions: <per the enumerated list>"
gh issue comment 338 --body "Decision: <accept 2px GT | keep TERMINATOR_ARC | shape models>"
gh issue comment 548 --body "Decision: <raise to 90% | ratify <N>% and gate CI on it>"
gh issue comment 547 --body "Decision: <pick frame <name> | close behind #288>"
gh issue comment 459 --body "Decision: <rf_ck_cassini_predicted is live | abandon the branch>"
gh issue comment 468 --body "Decision: <declare reconstructed | leave UNCLASSIFIED>"
gh issue comment 466 --body "Decision: <ship the SQLite file | run PostgreSQL>"
```

- **#557 where the confidence scale should saturate**: the combined
  confidence caps at 0.99 once two significant corroborating techniques agree
  at a precision-weighted mean of 0.66 with no post-cap disagreement penalty,
  which is the ordinary good case rather than an exceptional one. Above that
  the scale carries no information, and a third or fourth corroborating
  technique earns nothing. WS-5 fits a monotonic calibration map, which cannot
  separate values the cap has already collapsed onto one number, so this is
  not something calibration settles later. An agent can measure where the cap
  binds across the cohort and what the tiers would look like at other
  saturation points; choosing the point is yours.

### 0.1b Merge or close the one open pull request

PR #484 (#447, the round-trip residual) has been open since 2026-08-09,
is green and mergeable, and is fifty commits behind `main`. Rebase and
merge it, or say what it is waiting on. Nothing else is in flight.

### 0.1c Labels and assignees: check five inferred priorities

The tracker is fully labeled. Every open issue carries exactly one Priority
and one Effort label, at least one each of A-type and B-location, and
`rfrenchseti` as an assignee. The `Priority TBD` label is unused and can be
deleted from the repository if you want it gone.

Five priorities were assigned by inference from these plans rather than by
you, and are the ones worth a glance: **#53** PDS4 bundle generator parent as
Essential (the plans call output bundles required for all four instruments,
and none of it works end to end); **#28** backplane generator parent as
Important (a scope decision gating #54/#55/#57/#63/#77); **#34** PDS4 input
as Defer (the plans say it is not required for project completion); and
**#23** body shape models and **#78** CraterMaker as Defer (both sit with the
far-off Track F items). Override any of them in one command.

### 0.2 Adopt the calibration's falsification criterion (#334)

The confidence calibration has no armed falsification criterion and its
real-frame regression gate is suspended. Edit
`util/calibration/CAMPAIGN_20260718.md`: change the "Transfer watch
(proposed)" heading to "Transfer watch (adopted YYYY-MM-DD)", adjusting the
thresholds if you disagree with the proposal. That gives the calibration a
criterion that can fail. Tracked by #334.

### 0.3 What runs while the decisions wait

Everything in this section is dispatchable now, needs nothing from you until
its pull request, and is ordered by leverage rather than by size. Hand out
the first group before the second: the whole of Track A reads what the first
group produces, so work done before it is work measured through a corrupted
instrument.

**Group 1 — evidence integrity.** Nothing above this in the whole project.

1. **#288 resolution** — every one of the 10 reds is attributed and owned.
   Four are the coarse-lock family (#346's three, plus `N1633925572_1` whose
   tier the wrong ring lock moves); two are the Galileo star fields whose
   ground truth was captured under three degrees of freedom and no longer
   describes a two-DoF fit; one is the standing #24 exclusion; one is a real
   limb bias deliberately pinned red; and two are stale or wrong
   `primary_technique` pins batched onto #483. Nothing is unattributed, and
   no `expected.*` field is moved to match current behavior.

**Group 2 — the coarse-lock family.** One family, best done as one campaign:
**#476** (RingEdgeNav re-locking under a planted shift), **#346** (three
frames locked onto the wrong ring feature) and **#373** (the coarse seed
against competing edge populations). These gate the Track A study, which
consumes ensemble output at scale. The Saturn routing decision (annulus
composite at and above 25 km/px radial resolution, per-edge fit below)
reduces the family's exposure to the sub-25 km/px regime.

**Group 3 — parallel fill, any order, any number at once.**

- **Results index:** #515 with #516 (the root-blind share write and the test
  that cannot catch it); #501, #512, #514, #536 as one seam-cleanup
  batch; #528 with #531 as a metadata-provenance pair; #493 and #496 (run-level
  conditions reported per image); then the tail #472, #497, #524,
  and #533, #534, #535, #538, #540, #541.
- **Test debt:** #241 and #242 first, because the plan wants tested ground
  under any serious PDS4 or backplane work; then #243, #177, #524,
  #525, #530, #473.
- **CI:** #324, #336 and #426 — the agreement-estimator tests, the
  data-independent simulator suites and the stale committed render, none of
  which run in Actions today. #391 pins the lint tools so a release cannot
  turn `main` red on its own.
- **Products:** what is left of #265 (the dev-guide output-layout mismatch
  and the inventory-filename mismatch; its swallowed-label-write part is
  fixed); #520
  (move the pointing selection out of the reprojection CLI package); #495
  (raw-product dataset names for Cassini ISS).
- **Docs and cleanup:** #545, #549, #470, #471, #494, #518.

**What looks dispatchable and is not.** #483 and #547 wait on #288. #129
(Sphinx nitpicky-clean) wants #443 settled first, since that decides whether
`spindoctor.cli` subpackages get autodoc pages at all. #418 is a policy
question before it is a coding one. #464's affordability depends
on #466. #552 waits on an upstream oops fix. #239's cohort scan is agent work but the
sidecars it produces need your votes. And the two largest Track A
items, #225 and #172/#235, are gated on your frame approvals and batch votes, which
is what makes Group 1 worth starting today.

## 1. The library regression, and the deliberately-red set

**Read this before trusting a library run.** In the local integration
environment 10 of 75 sidecars disagree on `main` (#288). That is not the
pinned set below; it is a broken regression instrument, and until it is
reconciled the only gate a navigation-affecting branch can honestly clear is
**no new failures against `main`** — run the suite on `main` first, then on
the branch, and account for the difference. Do not re-ratchet a sidecar to
match current behavior, and do not read a green-looking subset as a pass.

Reconciling #288 is prerequisite to two other things: #483 (re-ratcheting the
pins the shift-equivariance fix moves) and #547 (the one place a built
product is compared between the results tree and the results index, whose
frame no longer navigates).

The set below is the *intended* steady state — the frames that should stay
red once #288 is reconciled, each owned by an open navigation issue. These
are pins, not regressions; do not re-ratchet them until the owning issue
closes.

| frame(s) | owner |
|---|---|
| N1492091163, N1867601758, N1867602424 (wrong ring-feature locks) | #346 |
| N1853392805 (highly-irregular exclusion discards the terminator fit) | #338 |
| N1484593951, N1686349893 (resolved-body ~2 px offset misses) | #350 |
| N1487595731_1 (multi_body: expects BodyDiscCorrelateNav primary, gets BodyLimbNav) | #483 |
| N1633925572_1 (ring_plus_body: expects the medium tier, gets low) | #476 |

**After any navigation-affecting merge, compare against `main` rather than
against the table:**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pytest tests/integration/test_autonomous_nav.py -m '' -n auto --dist=loadfile
# Expect, while #288 is open: the same red set as main. Any delta either way
# must be attributed in the merging PR.
# Once #288 is reconciled: red set = only the frames above.
```

The thread pins are not optional and are set nowhere in the repository —
not in `setup.sh`, not in `conftest.py`, not in CI, not in
`scripts/run-all-checks.sh` — so every script and every session exports them
itself. Without them a parallel run saturates the machine.

## 2. Track A critical path (dispatch as agent sessions, in this order)

The estimator (WS-0, `util/agreement/`) and the distortion tool (WS-17,
`util/fov_distortion/`) are built and proven on sims; the study now needs
the cohorts and the bulk run. The findings below constrain how the study is
scoped — read them before 2.2.

**Estimator findings that gate the study:**

- **limb-DT and ring-DT are not bias-independent** through the shared
  preprocessing layer. That pair must be declared or excluded from joint
  solves; body+ring is the common Cassini composition, so this hits the
  main cohort.
- **limb-DT vs disc-NCC holds as an anchor against symmetric PSF error**: a
  symmetric rendered-PSF mismatch opens no shared edge bias (the disc-NCC is
  ~16x less PSF-sensitive than the limb-DT). Not a clean-on-all-PSF verdict
  — the asymmetric/coma kernel that most directly matches the mechanism is
  unrenderable by the sim (#359), and the disc's sub-pixel NCC resolution
  floors detectability (#361).
- **Multi-body frames are not two independent measurements** (#322):
  cross-body limb errors correlate at +0.72 and the naive solve
  misattributes the coupling onto disc.
- **A ~2 px inward bias on partial-arc limb fits** (#321) is a navigation
  finding in its own right.
- The reliability gate *filters* rather than *shifts*, so its common-mode
  effect is a survivorship selection, not a bias — it can only make
  agreement look better, never worse, and cannot manufacture cross-technique
  coupling. That result is conditional on a separable/monotonic admission
  model; the real score-vs-error coupling is deferred to #358 (the sim
  cannot supply it), and whether the solve needs a survivorship correction
  is #360.
- Estimator tests do not run in CI (#324).

### 2.1 WS-3 — library growth (continuous; your votes are the bottleneck)

Next concrete step is the batch-006 manual-nav pass (7 frames voted "m",
class changes recorded in `_work/cohort_curation/batch_006_followups.yaml`).

**Prompt:**

> Run the batch-006 manual-nav pass: the 7 frames in
> _work/cohort_curation/batch_006_followups.yaml. Apply the operator's
> recorded class changes (C3479608 -> two_bright_stars_no_body;
> C4337947/C4401900 -> one_bright_star_no_body). Use sd_offset with the
> manual technique for each frame; do not trust the triage offset for
> C0164400400R (bloom-biased). Produce sidecars for the frames that
> navigate, present the results for operator review before committing.
> One PR per the sidecar-batch convention.

Then resume normal batch generation (`util/cohort_curation/`) and vote as
batches arrive.

### 2.2 WS-1 — the agreement study (#225, #226); waits on 2.1 cohorts

Your role at the gate: approve the frame selection. Apply these scoping
gates from the estimator findings above:

- The **limb-ring pair is correlated**, so it may not carry per-technique
  covariance claims; declare it or exclude it.
- The **limb-disc pair holds as an anchor against symmetric PSF error**:
  carry a declared limb-disc covariance where precision demands (a mild
  intrinsic negative coupling is real but its sign is unreliable), and treat
  the asymmetric-PSF channel as still open (#359).
- **Multi-body cohorts** declare the limb-limb pair (#322) and should be cut
  by illumination geometry, since part of the coupling is illumination-locked.
- Blob and disc correlate at +0.83 on partial bodies; never share a solve
  there.
- Cohorts are already filtered by the reliability gate; its selection effect
  is bounded in-sim but its real-frame size is unknown until #358, so the
  study's covariances describe *navigable* frames rather than frames, and
  the report must say so.
- A healthy identifiability report is **not** evidence that independence
  holds; all-positive recovered variances are necessary but not sufficient.

**Prompt:**

> Execute the agreement study's bulk layer (WS-1, #225) per
> plans/VALIDATION_AND_CALIBRATION_PLAN.md: run the pipeline over the
> approved real-frame cohorts with two or more independent fiducials per
> frame, compute the pairwise agreement statistics with the WS-0-proven
> estimator, and produce the report. Do not start the per-technique
> separation layer (it waits on WS-0's solvability map saying where it is
> meaningful). Operator approves the frame selection before any bulk run.

### 2.3 The finish line (dispatch after 2.2 produces data)

- **#229 / WS-4** — real images in CI: "Wire a small cached real-image
  tier into every-PR CI and the full suite on a schedule, per WS-4." Related:
  the data-independent sim suites still never run in Actions (#336) and there
  is no canonical environment for the committed sim baselines (#335).
- **#230 / WS-5** — re-anchor confidence on real evidence. The correlated
  ring-witness fix (#317) is done, so the calibration no longer trains
  against rows where two ring techniques on one catalog were fused as
  independent witnesses. **Still settle #316 before reading the Keeler
  tiers:** the tooling fits tier boundaries from the fused confidence
  scalar, and the orbit-uncertainty severity call moves five
  operator-verified frames across a boundary. Then: "Re-run the
  calibration tooling against the agreement study's measurements per WS-5;
  retire the confidence_provisional marker where the evidence supports it;
  re-bless tiers with the operator." This is where the terminator's
  provisional label and the sim-anchored coefficients get their real-world
  upgrade.
- **Accuracy tail** — #233, #150/#128 (design first; see Section 3),
  plus #234 and #232.

## 3. Parallel fill (independent agent sessions, any order)

Copy the line as the session prompt, prepending: "Work in
/seti/newnav/rms-nav. Read CLAUDE.md and the named issue first.
Independent review before done; all CI gates; one PR."

- **Ring ensemble follow-ups**: #319 (no library coverage for
  opposed-ansae geometry, so the conditioning guard is unvalidated); #380
  (fit an explicit per-family cross-covariance instead of collapsing
  correlated witnesses to a representative — gated on real-frame rho
  measurements from #225). #316 is
  an operator decision (Section 0.1), reversible by config either way.
- **#150/#128 (photometric limb redesign)** *(Fable-required — see 3b;
  the physics is subtle enough that a wrong premise survives review)*:
  "Produce the DESIGN ONLY for the photometric-limb fit that removes the
  ~0.1 px limb-darkening bias, per the diagnosis on #150/#128. No
  implementation until the design is operator-approved; validation must be
  against real images per WS-10. Address whether the same model-vs-image
  bias applies to non-step (gradual / shouldered) ring edges, not only the
  limb."
- **#373 (coarse-lock calibration pass)**: "Make the RingEdgeNav coarse
  seed robust against competing edge populations per #373, folding in the
  wrong-lock datapoints from #346."
- **#130**: "Calibrate the star limiting-magnitude model against real
  fields per #130."
- **#394 (shape-lock veto residual)**: the veto is suppressed when a trusted
  star fix agrees with the geometric consensus, which leaves the corner
  where the star fix is itself wrong-locked — a safe `conflicted` becomes a
  confident-wrong `success`. Sequence with #230/WS-5.
- **Rotation, in order**: #434 first (every technique reports its rotation
  about the image center, converting at the technique boundary). Until that
  lands, the distortion study's optical-center convention and navigation's
  image-origin one are not comparable, so #561 (the distortion cohorts sample
  a single sequence for some instruments) and any decision to fit rotation
  again both wait on it. #521 waits on fitting being enabled at all.
- **CK kernel follow-ups**: the kernel-side follow-ups (#433, #434,
  #437, #440/#444/#455, #446, #513) are independent of each other; #435/#436 are a
  pair and #436 waits on #435. #448 (locate C-kernel inputs through
  `spyceman` instead of a kernel directory tree) has a practical face worth
  weighing when scheduling it: running `sd_create_ck` today takes several
  `--kernel-dir` flags and still misses kernels. #459 (predicted vs
  reconstructed Cassini kernels, with an orphan branch behind it) and #468
  (the New Horizons pointing family) are operator decisions, not
  dispatchable work — Section 0.1.
- **Navigation correctness, in order**: #476 (RingEdgeNav
  re-locks onto the wrong ring edge under a planted shift), which belongs
  with #346 and #373 as one coarse-lock family, now scoped to sub-25 km/px
  Saturn scenes by the routing decision. `N1633925572_1_CALIB` is the
  cleanest reproducer in the library: two techniques agree to 0.08 px while
  RingEdgeNav lands 39 px away on 6.7% of its points, so a fix is verifiable
  without operator adjudication and the frame's tier returns on its own.
- **Results-index follow-ups** (the index shipped 2026-08-25; none of these
  blocks anything): #515 and #516 together (a cloud-share ingest can write
  another root's document into this root's rows, and the test that should
  catch it cannot fail); #501, #512, #514, #536 as a cleanup batch over the
  seam; #493 and #496 with #418 as one "what does a task's status owe a
  retrying queue" batch; #531 and #528 as a metadata-provenance pair. #466
  and #462 are decisions first (Section 0.1). #464, #465, #467, #486, #542
  are capability extensions to schedule when someone wants them.
- **Logging follow-ups** (small, independent, no sequencing): #424 (remove
  `sd_create_bundle_cloud_tasks` — it is unwired and leaks to the worker
  terminal); #418 (decide whether a mosaic cloud task's `status` should
  reflect its per-image failures, not only its counts — a policy question
  before it is a coding one); #423 (the GUI viewers print library log
  records to stdout; about ten tests capture that fallback and need their
  capture strategy changed first); #429 (give the `util/` tooling the same
  logging surface). #427 (reorganize the config namespace) is larger and
  should precede #118.
- **Sim realism residual (#227)**: the de-circularization is done and on
  main; #227 stays open only for the realism proof and closes at the
  operator's realism-verdict gate, itself gated on #309. #309
  (realism-configured multi-instrument campaign — biggest
  calibration-credibility win available; consumes the fidelity gaps #325,
  #329-#333, #341-#345, #290, #377) is the load-bearing step, with #310
  (structural boundary enforcement) and #311 (mirror-parity guard) hardening
  the partition. Each issue body is a prompt basis.
- **#355 (Voyager sim distortion per camera)**: re-measure and split the
  Voyager distortion defaults once the star-lock rate improves.

## 3b. Model-tier guidance (where a top-tier model is truly needed)

Reserve the top-tier (Fable-class) model for work where a
plausible-but-wrong answer survives review by looking right; a
mid-tier (Opus-class) implementer is the efficient default everywhere
else. Applied to the open items:

- **Top-tier required:** the #230/WS-5 calibration-fit adjudication and #309
  (calibration on messy evidence); the #358/#360 survivorship-correction
  math and the #359/#361 asymmetric-PSF coupling probes; the **#150/#128
  photometric-limb redesign** (both the design and its adjudication — the
  physics is subtle and a plausible-but-wrong premise rides straight through
  review: e.g. "rings are unaffected" holds only for sharp step-edges, but a
  gradual or shouldered ring edge carries the same model-vs-image photometric
  bias the limb does); and the independent-review pass on anything
  statistical, boundary-touching, or calibration-touching, regardless of who
  implemented it.
- **Mid-tier drafts, top-tier adjudicates:** #310 (the boundary
  restructuring — the guard tests catch mechanical regressions, the review
  catches new leak shapes).
- **Mid-tier or below suffices:** library growth, the agreement study's
  bulk execution (once WS-0 hands it a proven estimator), #229, #311, #373,
  #130, the logging follow-ups (#418, #423, #424, #429), and the
  documentation/engineering items.

## 3c. Tracking-issue register

Open issues grouped by theme so none is lost to a PR body; the sequencing
hooks reference the sections above. All carry A/B/Priority/Effort labels
with assignee rfrenchseti.

**Evidence integrity (do first; everything else reads what these produce):**

- **#288** 10 of 75 library sidecars disagree on `main` locally, so the
  regression instrument cannot tell a regression from the standing state;
  20 are one dependency defect and 3 are genuine; blocks #483 and #547
- **#548** suite coverage is 79% against a stated 90% floor with nothing
  enforcing either (Section 0.1 decision)
- **#547** the one place a built product is compared between the results tree
  and the results index runs against a frame that no longer navigates

**Kernel-facing navigation defects:** none open. Rotation fitting is off for
every instrument, so the fitted-rotation omission costs nothing and Galileo
gets the corrected kernels it previously got none of. #521 — a conflicted
result dropping its fitted rotation — cannot fire while that holds, and is
deferred until an instrument fits rotation again; a test fails if one does.

**Results index (shipped 2026-08-25; none blocking):**

- **#515 / #516** a cloud-share ingest can write another root's document into
  this root's rows, and the test that should catch it asserts on the row a
  root-blind write would keep
- **#501, #512, #514, #536** seam cleanup: bind queries to the resolved
  schema, one document-to-column placement, fold the ingest's parse loop into
  the seam, stop overstating what a missing row means
- **#493, #496** run-level conditions reported per image (batch with #418)
- **#528, #531** metadata provenance: no format version, and documents record
  where the image was cached rather than where it came from
- **#462, #466** decisions (Section 0.1)
- **#464, #465, #467, #486, #542** capability extensions, schedule on demand
- **#472, #497, #524, #533, #534, #535, #538, #540, #541** the small tail

**Confident-wrong / ensemble honesty (sequence with #230/WS-5):**

- **#346** three library frames lock confidently onto the wrong ring feature
  (owns the N1492091163 / N1867601758 / N1867602424 red pins)
- **#476** RingEdgeNav is not shift-equivariant: a planted shift re-locks it
  onto the wrong ring edge — the same coarse-lock family as #346 and #373,
  seen from the round-trip side; measured as an alias lattice that polarity
  does not break, so it waits on #373; exposure is limited to sub-25 km/px
  Saturn scenes by the routing decision
- **#482** BodyDiscCorrelateNav misses by up to ~1 px on a weakly-constrained
  axis, the residual left after PR #484 closes #447
- **#394** shape-lock veto suppression trusts a star fix that could itself be
  wrong-locked, turning a safe `conflicted` into a confident-wrong `success`
- **#380** correlated-witness fusion collapses to a representative at rho=1;
  fit an explicit cross-covariance once #225 measures real-frame rho
- **#400** the ensemble merge and tier logic have never been exercised on the
  strongly anisotropic covariance the Titan haze fit reports

**Simulator fidelity gaps (feed #309 and the sim follow-ups in Section 3):**

- **#325** simulated stars shine through dark limbs; star-technique success is
  optimistic
- **#329** simulated calibrated products floor at 1 LSB; real products dither
  below it (WAC diverges 8x)
- **#330** instrument chains render cosmic-ray transients at zero
- **#331** simulated hot pixels are per-scene, not per-detector
- **#332** PSF catalog has one kernel per instrument; binned/summed readout
  modes are inexpressible
- **#333** four physical error axes are unmodeled
- **#290** body renderer exceeds the sim render-time budget on oversampled grids
- **#377** sim rings are single annuli; build realistic nested-ringlet scenes
  and tests
- **#341** the campaign's scene mixture is authored, unvalidated against real
  frames
- **#342** star_psf_sigma is a 3.0 placeholder on Galileo, Voyager, LORRI
- **#343** the tuned NAC PSF wing may be absorbing operator registration error
- **#344** haze brightness is a module constant
- **#345** a scene can echo truth-side noise into instrument_config with no
  validator warning

**Calibration governance / CI (gate WS-5 and the CI tier in Section 2.3):**

- **#334** calibration has no armed falsification criterion and its real-frame
  gate is suspended (owns the Section 0.2 transfer-watch step)
- **#335** no canonical environment for committed sim baselines (0.99 vs
  0.81-0.84 across machines)
- **#336** data-independent simulator integration suites never run in Actions
  (relates to #229/WS-4)
- **#340** library_crosscheck records only a yes/no primary-technique flag,
  not the winning technique
- **#426** a committed sim render is stale on `main` and the test that would
  say so is integration-marked, so nothing catches it per PR

**Titan haze refinements (the method ships; these are measured limits):**

- **#403** the arc ray reach is sized by the search window rather than by
  where the limb can be, costing rays on large well-framed frames
- **#404** the flat arc-residual cap behaves as a size-dependent gate, and
  the measurements say it must not simply be raised
- **#401** the extreme-phase (> 150 deg) edge of the working range is
  uncharacterized
- **#402** the main rings are masked opaque, refusing frames visible through
  the C ring or the gaps
- **#397** a self-calibrated haze-radius table would remove the dominant
  along-track error; **#398** CB3 cartographic refinement; **#399** a
  Voyager validation cohort; **#405** library growth through the standard
  curation pipeline

**Logging follow-ups (all small and independent; Section 3):**

- **#418** a mosaic cloud task reports success when every image in it failed
  (policy decision first)
- **#423** the GUI viewers print library log records to stdout through
  pdslogger's handler-less fallback
- **#424** remove `sd_create_bundle_cloud_tasks`
- **#427** the config namespace is organized on no stated axis; sequence
  before #118
- **#428** upstream registry-eviction request to `rms-pdslogger`
- **#429** give the `util/` tooling the same logging surface

**Library-frame reds and decisions (Section 1):**

- **#338** highly-irregular exclusion discards the ground-truth terminator fit
  on N1853392805 (decision)
- **#350** two resolved-body frames miss offset tolerance by ~2 px
  (N1484593951, N1686349893)
**Agreement estimator real-frame follow-ups (sequence with #225/WS-1 and
#230/WS-5):**

- **#358** measure the real reliability-vs-error coupling and run the
  stratified estimator on the real #225 cohorts — the size of the gate's
  selection optimism the sim cannot supply (Important)
- **#360** decide, after #358, whether the agreement solve needs a
  selection-aware (survivorship) correction (a decision issue)
- **#359** probe limb-disc PSF coupling under the asymmetric/coma/field-varying
  PSF error the sim cannot render (Important)
- **#361** disc-NCC sub-pixel resolution floors the smallest limb-disc coupling
  detectable (Important)
- **#321** partial-arc limb fits carry an undiagnosed inward radial bias of
  about 2 px (navigation finding)
- **#322** cross-body limb errors correlate at +0.72; multi-body frames are
  not independent measurements
- **#324** agreement estimator tests do not run in CI

## 4. Standing practices for every session you dispatch

- Environment: `source /seti/newnav/setup.sh`.
- The controller pattern works: one session as controller, implementer
  subagents per phase/slice, an independent fresh-context review of every
  deliverable, fix rounds until the critique is clean, full CI
  (`./scripts/run-all-checks.sh -i`), then one PR. Ask for it explicitly in
  the prompt if you want it.
- CI expectations: `run-all-checks.sh -i` is the pre-merge gate; the library
  suite's red set must equal the documented pinned set (Section 1) or every
  delta must be attributed in the PR.
- Issues: every new issue carries A-type, B-location, Priority, Effort
  labels and assignee rfrenchseti.
- **Never leave future work, a deferred fix, a known limit, or a pending
  decision recorded only in a PR body, a comment, a campaign record, or a
  docstring — file a tracking issue and reference it from the prose.** PRs
  get merged and scroll away; an item that lives only in prose is an item
  that will be lost.
- Sidecar changes: one PR per review batch; per-frame dated notes in the
  sidecar, never only in gitignored files.
- Perf tests (`tests/integration/test_sim_perf.py`): serial only, never
  under a parallel battery.

## 5. Sequencing summary

```text
0.1 decisions (#316, #407, #338, #548, #547, #459, #468, #466, #557)  (operator, minutes)
0.1b merge or close PR #484                                     (operator, minutes)
0.2 adopt transfer watch (#334)                                 (operator, minutes)
--- then, before Track A collects anything at scale
--- #288 library regression reconcile    (agent session; unblocks #483 and #547)
--- then ---
2.1 library growth (batch-006 + continued)   (agent session; your votes gate it)
2.2 agreement study bulk   (after 2.1 cohorts; you approve frames)
2.3 CI tier, re-anchor confidence, accuracy tail (after 2.2)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal named in Section 1 of the program plan.
