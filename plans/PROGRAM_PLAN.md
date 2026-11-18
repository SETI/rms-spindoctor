# SpinDoctor Program Plan

*The top-level plan of record for all remaining work. It is written to be
readable without knowledge of the code internals or the statistical
methodology; the detail lives in the sub-plans it points to. Last
reconciled 2026-11-17.*

**Document map** (what to read for what):

| Document | Role |
|---|---|
| `plans/PROGRAM_PLAN.md` (this file) | What remains, why, in what order, and what can run in parallel. Every open issue is accounted for in the index at the end. |
| `plans/VALIDATION_AND_CALIBRATION_PLAN.md` | Full methodology for Track A (the science): what "validated" and "calibrated" mean, the statistical machinery, acceptance criteria per workstream. Binding where it overlaps anything else. |
| `plans/SIM_IMPROVEMENT_PLAN.md` | The simulator plan (Track S): the order of simulator work toward realistic frames for every supported camera, the restructuring of its code, and the phase each simulator issue lands in. |
| `plans/ENGINEERING_PLAN.md` | Full implementation detail for Tracks B-F: per-item context, file pointers, constraints, and acceptance criteria sufficient to hand any item to a developer (human or model) cold. |
| `plans/COHORT_CURATION_PLAN.md` | The operational playbook for growing the curated image library: metadata-driven discovery, operator review votes, sidecar generation. |
| `plans/OPERATOR_PLAYBOOK.md` | The operator's dispatch sheet: which decisions are pending, which sessions to launch, and in what order. |
| `plans/archive/` | Superseded and fully-executed plans, kept as historical records with dates in their filenames. Nothing in there is current work, though the executed designs remain the reference for what shipped — the C-kernel writing and reading halves and the results index among them. Its `README.md` says what each one delivered and where the follow-ups went. |
| `critiques/` | Point-in-time reviews, frozen as written. The two the simulator plan works from are in `critiques/` itself (the simulator realism critique of 2026-11-06 and the simulator code-structure critique of 2026-11-16); older reviews are in `critiques/archive/`. |

---

## 1. The goal

One sentence: **an end user can take raw archival images from Cassini ISS,
Voyager ISS, Galileo SSI, or New Horizons LORRI and produce — locally or in
the cloud — precisely navigated pointing, reprojected mosaics, per-pixel
geometry backplanes, archive-quality PDS4 bundles, preview images and
metadata, and updated SPICE pointing kernels, with every accuracy and
confidence number backed by published evidence and every capability
documented.**

The end-of-project deliverable therefore has two halves:

1. **A finished pipeline** — every stage works for every supported
   instrument, runs at campaign scale, and is protected by tests.
2. **A defensible accuracy story** — the pointing errors and confidence
   values the pipeline reports are calibrated against evidence a reviewer
   can check, and a capability matrix states exactly what is supported and
   how well, per instrument.

The second half is not polish. This system's outputs are scientific
measurements; a navigation system that reports offsets without defensible
uncertainties is not finished, it is merely running.

## 2. Where we stand

*Status as of 2026-11-17.*

The engineering core is built and healthy: the full navigation architecture
(ten autonomous techniques plus manual), reprojection, backplanes,
simulation, rank-1 (single-axis) ring support, corrected-pointing SPICE
C-kernels as a delivered product, the statistics system, and strict quality
gates (typing, linting, the full unit suite) are in place. Every program that
reads a navigation record reads it through one seam over two storages — the
results tree of JSON documents, and an optional rebuildable index
(`sd_results_index` / `sd_stats_report`) that no program requires. The
cross-technique agreement estimator exists and is proven on known-truth sims.
Every reported position is stated in one declared pixel convention across the
navigator, the simulator, the reprojection and mosaic products, the bundle
labels and the operator displays, with tests that fail on a wrong convention;
the residuals that remain are named in the tracks below. PDS4 bundle
generation runs end to end for Cassini ISS over a synthetic cohort: a bundle
whose collections, inventories, index tables and data labels satisfy the PDS4
schemas, the Schematron rules and their own tables under `sd_create_bundle
check`, held there by a test. What it does not yet have is a run over a real
volume (#708), the Cassini mission area (#717), or any of the other three
instruments (see Track D). Cassini ISS frames are read through
rms-host-cassini (since #952, on rms-oops 0.4.0), whose current release
times 2,812 Cassini frames one exposure late until an upstream fix ships
(#949).

The simulator is de-circularized and realism-tuned, and the first phase of its
improvement plan is done and the second nearly so (Track S): a calibrated
Cassini noise floor, fitted cosmic rays, sweeps through the full instrument
chain, per-pixel noise, scene validation on every entry point, truth-based
assertions, per-instrument tables in validated configuration, and renders fast
enough for full-frame scenes. The largest item left in that phase is the
regeneration of the realism record, which is in progress. A code-structure
review (2026-11-16) found the simulator's core well designed and much of the
rest accreted, and the simulator plan schedules the restructuring beside the
physics.

Two facts about the evidence behind that health are worth stating plainly,
because both are load-bearing and neither is comfortable. Suite coverage is
79%, not the 90% two shipped plans' acceptance criteria claim, and nothing
enforces any floor (#548). And 6 of 75 curated library frames disagree with
their sidecars on `main` in the local integration environment (measured
2026-11-16, when PRs #999 and #1003 each ran the library suite on `main`
beside their branches: 6 failed and 69 passed, the same frames and messages
both times): the two Galileo frames whose recorded truth is the
rotation-fitted answer that the change in #560 retired (#744), two stale
`primary_technique` pins under #483, a deliberate limb-bias pin under #350,
and the highly-irregular exclusion under #338. The ten star-frame pins the
sparse-star-field fixes moved were ratified by the operator on 2026-10-31 and
re-pinned with #754, and #346's two Encke gap frames now pass. Every one of
the six is attributed to an open issue or an operator decision, so the
regression instrument reads as a known state rather than an unknown one; but
the two under #744 and the two under #483 are judgments still owed rather than
settled pins, so "no new failures against `main`" remains the only gate a
navigation-affecting branch can clear.

The single largest remaining block is Track A: **the validation program is
designed and partly built, but not yet run on real frames.** The confidence
formulas are anchored against simulated planted-truth scenes, so confidence
values are not arbitrary defaults, but every output still carries a
`confidence_provisional` marker until the real-frame agreement study
re-anchors them. Turning that provisional story into a defensible one is the
critical path.

The ring, star, and ensemble fixes that followed the sparse-star-field work
sit in a stack of seven pull requests (#766, #773, #774, #779, #782, #784,
and #789) that the operator froze on 2026-11-06: their rules were tuned on F
ring and spokes bundle frames, about 5% of Cassini, and there is no
cross-mission benchmark to judge them on (#1033 builds it). No further work
goes into them until that benchmark exists, and the issues they address stay
open (Track B).

The curated image library — the raw material for both regression testing and
validation — holds 75 sidecars covering 16 of its 17 scene classes, short of
the first-stage per-class minima (#172), and only Cassini has reached the
final per-instrument target of at least 20 images, toward at least 120
across all four instruments (#235). It seeds regression baselines and the
calibration cohort, and supplies the agreement study.

## 3. Why validation dominates the remainder (plain-language version)

No independent record exists of where these cameras were truly pointing —
not to the hundredth-of-a-pixel level the system reports. So accuracy
cannot be checked against an answer key. The program substitutes three
mutually reinforcing sources of evidence:

1. **Simulation with planted truth** — render a synthetic image where the
   correct answer is known by construction, and measure recovery error.
   Trustworthy only if the simulator is (a) independent of the navigator's
   own models and (b) demonstrably realistic compared to real images.
   Making it realistic for every camera, and proving that simulated error
   predicts real error, is the simulator's track (Track S); the realism
   verdict is Track A's.
2. **Agreement between independent methods on real images** — when a star
   field and a moon's edge independently yield the same pointing
   correction to within a fraction of a pixel, that agreement is evidence
   both are right to about that level. This is the only accuracy signal
   available from real archival frames, and extracting it requires real
   statistical care (the methodology plan covers the traps).
3. **A large, deliberately diverse library of operator-verified images**
   — feeding both of the above and serving as the permanent regression
   safety net.

Everything else in the program — new instruments' quirks, PDS4 bundle
generalization, performance, documentation — is real work but
conventional engineering.
This is the part that makes the numbers mean something.

## 4. The tracks

Work is organized into seven tracks: six lettered A-F, and Track S, the
simulator, which has its own plan. A track is a stream of related work that
can proceed largely independently of the others; the parallelism notes say
where they touch.

### Track A — Validation and calibration (the science half)

**Goal:** every accuracy and confidence number the pipeline emits is
backed by published evidence; the `confidence_provisional` marker is
retired.

**Why:** section 3. This is the critical path to the deliverable's second
half and the majority of the remaining effort.

**Shape of the work** (methodology and acceptance criteria in
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`; the workstream codes below are
its section names):

1. **Grow the image library** (#172 first stage, #235 growth; WS-3) —
   continuous background work: automated candidate discovery, operator
   votes in batches, sidecar generation. Feeds everything below. Re-pinning
   the 19 sidecars whose tier or primary technique the frozen ring and star
   stack changes waits until it merges (#866). Checking three recorded
   truths the image stars contradict against a star overlay (#867) is the
   operator's and can run any time. No sidecar yet expects the ring annulus
   technique as its primary (#871).
2. **Prove the simulator realistic** (#227, with #223, #309, #341, #377,
   #644, #646; WS-2) — the de-circularization is done and on main: an
   independent forward model, the truth/idealized information partition with
   an import-graph guard, model-error sweep axes, full detector noise
   including the I/F path, and the rewritten simulator report. #227 stays
   open only for the *realism* residual: the calibration is not yet fitted on
   the realism-anchored renderer configuration, so the realism evidence does
   not underwrite the shipped calibration (#309, the biggest
   calibration-credibility win available, which the simulator plan repairs in
   its Phase 3 together with the generator defects under it); the terminator
   deliverable is degenerate with no realism verdict (#223); and the authored
   scene mixture is unvalidated (#341). Closing #227 is the operator's
   realism-verdict gate, itself gated on #309; the simulator plan completes it
   in Phase 5, when simulated frames are rendered from real frames' geometry
   and navigated by the production models. The simulator-fidelity gaps this
   item listed on 2026-10-14 are closed: stars shining through dark limbs
   (#325), the 1-LSB calibrated-product floor (#329), zero cosmic-ray
   transients (#330), per-scene hot pixels (#331), one PSF kernel per
   instrument (#332), the render-time budget (#290), the constant haze
   brightness (#344), the truth-side-noise echo (#345), the four scene
   features that rendered something other than what they name (#625, #626,
   #627), the scene-language keys (#629 to #633, and the star-index versus
   body-corner split, #641 with #409), and the simulated body's overflow
   handling (#750). What remains of the fidelity work is scheduled in Track
   S: a planted roll that turns positions but not smear, companions, sky, or
   a body's lit side (#644, with #825), the limb-bias harness measuring a
   simulated ridge half a pixel from the deployed one (#646, with #842), the
   unmodeled physical error axes (#333), the NAC PSF wing (#343), dense ring
   scenes (#377), and the structural information-boundary and mirror-parity
   guards (#310, #311).
3. **The agreement-estimator residual** (WS-0) — the estimator, the
   identifiability map, and the known-truth validation are delivered and on
   main under `util/agreement/`. Findings that constrain item 5: the
   **limb-ring pair is measured as correlated** through the shared
   preprocessing layer, so it cannot carry per-technique covariance claims;
   the **limb-disc pair holds as an anchor against symmetric PSF error** but
   its asymmetric/coma channel is unrenderable by the sim (#359, which the
   simulator plan places in its Phase 4d) and the disc's sub-pixel NCC
   resolution floors detectability (#361); **multi-body frames are not
   independent measurements** (#322); a **~2 px inward bias on partial-arc
   limb fits** (#321) surfaced as a navigation finding in its own right. The
   open residual, all feeding the study in item 5: measure the real
   reliability-vs-error coupling on the #225 cohorts (#358, which gates the
   reliability-gate lower-bound claim on real frames), decide whether the
   solve needs a survivorship correction (#360), probe the asymmetric-PSF
   limb-disc channel (#359), quantify the disc-NCC resolution floor (#361),
   and wire the estimator tests into CI (#324).
4. **Camera distortion residual** (WS-17) — the star-field distortion tool
   is delivered and on main, its measured residuals populate the sim
   defaults, and distortion is applied per feature-position in the agreement
   study so it does not masquerade as navigation error. The open residual is
   #355: re-measure and split the Voyager sim distortion per camera once the
   star-lock rate improves. The cameras' small residual distortions are to be
   fixed in the upstream camera models rather than dropped from the simulator
   (#970, decided 2026-11-14).
5. **The agreement study itself** (#225, corroborated by #226; WS-1, WS-1b) —
   the flagship: run the pipeline over hundreds to thousands of real frames
   with two or more independent fiducials and publish the agreement
   statistics. Its bulk (pairwise) layer needs only the library and distortion
   validation; it must not wait for items 2-3, which gate only the finer
   per-technique separation. Two outside answers corroborate it: the published
   F ring bundle's per-frame pointing, and the B ring spokes bundle (#637).
   The F ring bundle's WAC frames shared a 0.76 px difference that was the
   bundle's own; with it restored, 2270 of 2448 WAC frames agree within 1 px
   (#759; the cause is filed against the bundle). Two calibration
   recommendations that a sidecar note is the wrong home for -- the agreement
   gap and the terminator coarse search -- are carried on #675.
6. **Wire real images into CI** (#229; WS-4) — a small cached tier on
   every PR, the full suite on a schedule. The data-independent simulator
   suites run in Actions with a canonical environment for the committed sim
   baselines (#336 and #335, both done by #834), and the committed sim render
   that had gone stale on `main` without anyone noticing (#426, fixed by
   #680) showed what a deselected integration tier costs.
7. **Re-anchor confidence on real evidence** (#230; WS-5) — re-run the
   existing calibration tooling against the agreement study's
   measurements; retire the provisional marker. The correlated-ring-witness
   fix (#317) is done, so the recalibration no longer trains against those
   rows; the ring orbit-uncertainty severity decision (#316) still sequences
   before it, since it pushes tier boundaries the wrong way, and adopt the
   calibration's armed falsification criterion (#334). Whether the low tier
   carries a sigma ceiling is an operator decision calibrated with it (#751):
   today any answer that clears the low tier's confidence floor is reported
   `success` however large its uncertainty, while legitimate blob-only answers
   on irregular bodies reach a sigma of 18.85 px. The reported offset sigma is
   too large at the core and too small in the tail (#788), which waits on the
   ensemble fix for #767 in the frozen stack. The simulator plan's Track R
   restates this item from the simulator's side: the real-frame part of #230
   proceeds once #334 is adopted and #225 supplies anchors, and the part that
   needs single-technique accuracy waits for the simulator's transfer check.
8. **Close the accuracy tail** (#233 measured star SNR and constant
   sensitivity, WS-9; #150/#128 the known ~0.1 px limb bias, WS-10; #232
   end-product accuracy for backplanes/mosaics/PDS4 values, WS-18). Realistic
   noise for calibrated images (#234, WS-13) is done, closed by #888.

**Confident-wrong families that poison the validation data** (they belong to
Track B but gate the study, because it consumes navigator output at scale): the
ring-lock family (#346, #373, #755), whose remaining error is along nearly
straight ring edges (a few pixels, reported at the medium tier with a
sub-pixel sigma) and, on a few frames, a seed on the neighboring cataloged
edge, and the sparse star field family -- a one- or two-star solution
accepted as success and landing up to 23 px wrong, two techniques reading the
same single star counted as independent corroboration, and a centroid that
can measure a brighter neighbor instead of the star it was asked for, all
three closed by the sparse-star-field work (#754); what is left is the frames
with one usable star or none, refused rather than solved (#621). The fixes
for both families' remainders are in the frozen stack. The
body-witness-veto
family -- a high-phase haze crescent returning a gate-passing success ~30 px
wrong (#328) and the disc technique locking on at extreme shape mismatch (#291)
-- and the body-body occlusion pair -- the disc template (#326) and the
visible-arc report (#327) -- are closed by the cross-technique body-witness veto
and the occlusion-aware body model. The ensemble-independence family -- a seeded
single-star refine dragging a body fix (#222), two ring techniques fused as
independent witnesses (#317), and scattered-light disc/limb errors fused as
independent (#339) -- is closed by the consensus independence resolution.

**Operator's role:** batch votes on library candidates (ongoing); bless the
realism verdict; approve agreement-study frame selection; make the ring
orbit-uncertainty (#316) call; re-bless tiers after item 7. Everything
else is agent-executable.

**Parallelism:** items 1-4 all run concurrently; item 5's bulk layer starts
as soon as 1 and 4 give it cohorts; 6 rides alongside; 7-8 close out in
sequence after 5.

### Track S — The simulator

**Goal:** synthetic frames for every supported camera, as realistic as they
need to be to check navigation accuracy and no more, produced by code that
looks designed rather than accreted. The detail, the phase each issue lands
in, and the exit criterion of each phase are in
`plans/SIM_IMPROVEMENT_PLAN.md`.

**What it is for:** the simulator renders frames whose correct pointing is
planted, so the navigator's error can be measured directly, which no real
frame allows. It characterizes the navigator; it never tunes it. The navigator
is tuned on real images only, and where a simulated frame and a real one
disagree, the simulator is what gets fixed.

**Its two purposes:**

1. **Sensitivity sweeps** — how navigation responds along one axis (noise,
   exposure, smear, cosmic-ray rate, a defect mode). These scenes stay simple
   on purpose, and everything they use stays supported throughout.
2. **Accuracy scenes** — realistic, compound scenes: real frame sizes and
   offsets, every camera, several kinds of content at once, and interleaved
   detector defects. The goal is that these stand in for most of the real
   image library. A library class is retired from the regression suite only
   when simulated frames of that class pass a transfer check (planted offsets
   on real frames and on their simulated twins give matching errors, frame by
   frame) and the validation catalog covers it. Until then the real frames
   stay the reference.

**Where it stands (2026-11-17):** Phase 0 is done: the calibrated Cassini
noise floor, fitted cosmic rays, and full-chain sweeps are on main, and the
calibrated disc correlation's sub-pixel bias is attributed. Phase 1, the
foundations, is done except for these items, and #970 waits on the upstream
camera-model fixes. The realism-record regeneration (with the report's sweep
figures, #961) is in progress; the realism star twins it needed, built from
each real frame's catalog stars, merged on 2026-11-17 (#989, by #1004), and
the regeneration arms the realism match's regression bands (#903). The
simulator's tripwires still rest on one noise draw (#960); the disc
correlation's rotation search and candidate refinement are still too costly
for full-frame scenes (#126, #945); simulated star saturation (#854) is on
hold by the operator; and the realism match's star-count criterion (#925)
closes with the star-limit and radiometry fixes of Phase 4a (#899, #895).
Phase 1 also found and fixed navigator defects in the shared correlator
(peak suppression ignoring wrap-around, the ring annulus refinement clamp,
the overlap floor), and filed the rest in Track B (#791, #941, #997, #1002).

**What comes next**, in the plan's order:

- **Phase 1b, restructuring that moves no pixel** — shared scene fixtures
  first (#923), then the code laid out by concern, with the rendering
  code's directory renamed for what it does (#1005), then exact copies,
  names, the oversized body model, model selection, the light-direction
  convention, and stage seeding (#1006 to #1010, #1030). It lands while the
  committed baselines are still, so they prove each change neutral. It starts
  when the realism-record regeneration has landed.
- **Phase 2, rendering correctness** — defects that put pixels in the wrong
  place: retire the classic body renderer (#901, with #931), one ellipsoid
  projection for both sides (#1011), frame-filling bodies (#890), what a
  planted roll turns (#825, #644), ring-edge positions (#891, #873),
  edge-of-frame blur (#892), and the simulated limb's gradient peak checked
  against real limbs (#842, #646), among others.
- **Phase 2a, settle the scene format** — refuse null scene keys (#988 and
  its kin), build one typed schema core for configuration and scene files
  (#1021; pydantic, operator decision 2026-11-16), type the instrument
  catalog (#1022), and give each detector effect one model and one source
  (#1017, #1023), so the scenes written next are written once.
- **Phase 2b, realistic guard scenes** — every camera at real frame sizes
  and offsets, compound content, dense rings, interleaved defects (the
  authorable parts of #904 and #377), each asserting against its planted
  truth, so they guard the physics changes that follow.
- **Phase 3, repair what the simulated confidence values were fitted on** —
  a calibration generator that draws catalog bodies at half size and fails
  on four families (#898, #837), the realism-anchored configuration (#309),
  and the refits that follow (#793, #874, #942, #794).
- **Phase 3b, restructure the renderer before the physics** — parse every
  scene into typed records (#1024), resolve each render's settings once
  (#1028), make every stage read the typed scene (#1027), and rebuild the
  editor from the scene's field declarations (#1020), among others, so each
  new key in Phase 4 is declared in one place. It runs alongside Phases 2b
  and 3.
- **Phases 4 to 7, the physics and the per-camera twins** — physics in order
  of how many real frames need it (radiometry and shadows, ring geometry,
  bodies the navigator does not know about, per-camera detectors and optics,
  Voyager, and filters); rendering from real frames' geometry and navigating
  with the production models (#918, #897, #843, #311), which completes
  #227; the transfer check (Phase 6); and the validation catalog, which
  publishes a per-instrument accuracy number for each class that passed and
  retires those library classes (Phase 7). Three navigator defects the
  simulator found are listed in Phase 7's table and held as expected
  failures until fixed: #790 (the shape-lock veto missing confident wrong
  answers on irregular mesh bodies, Priority 2; no phase waits on it), #835,
  and #975.
- **Phase 8, code and docs** — the simulator docs and the editor's
  remaining defects, alongside the rest and finished last (#914, #915,
  #856, #1000).

**Operator decisions:** #923's three choices are decided (2026-11-18): both
sides import a neutral module for the image classifier, the test-only
helper moves under the tests, and the old plan-phase labels outside the
simulator are tracked separately (#1035). Before Phase 2a's detector work,
whether each pair #1017 lists (banding, dark signal, hot pixels) is one
effect or two; and in Phase 2, the ruling #644 asks for on what a planted
roll turns. #854 stays on hold until the operator lifts it.

**The navigator config is deferred** (operator, 2026-11-17). Typing the
navigator's camera blocks (#1029) and every other section (#1032), and the
config namespace question (#427), wait in Track D. Nothing in the simulator
plan changes navigator code; #1023 edits the navigator's config files only to
delete camera-block keys nothing reads, and until #1029 lands the typed
scene gives the camera settings a scene may override their own type
(#1024), repeating the camera keys' ranges.

**Parallelism:** Phase 1b waits only on the realism-record regeneration.
Phase 3b runs alongside Phases 2b and 3, and Phase 4 waits for both 3b and
2b; within Phase 4, the per-camera parts run in parallel with Phase 5. Track
A's bulk agreement study does not wait for the simulator; the simulator gates
only the realism verdict (#227) and the single-technique accuracy part of the
recalibration (#230). The simulator plan's two parallel tracks belong to
other tracks here: real-frame confidence (its Track R) is Track A item 7, and
the outer-planet ring models (its Track O: #908, #81, #82, #83) are Track F.

### Track B — Navigation correctness (remaining)

**Goal:** no known case where the navigator returns a confidently wrong
answer or fails on a navigable scene.

**Why:** these defects poison the validation data (Track A consumes the
navigator's output at scale) and are exactly what a user hits first.
The known open defects:

- **#949** (Priority 2, upstream) — 2,812 of 443,177 Cassini frames are
  timed one exposure late, because rms-host-cassini starts each exposure at a
  label time that equals the shutter-close time on those frames. None of the
  75 library frames is affected. The fix is upstream
  (SETI/rms-host-cassini#3); when it ships, raise the floor and re-navigate
  any of those frames navigated in the meantime.
- **#346** — three library frames (N1492091163, N1867601758, N1867602424)
  that once locked confidently onto the wrong ring feature. None is a wrong
  lock any longer, and all three pass on their pins since the operator's
  2026-10-31 re-pins (applied with #754); the issue stays open as the record
  of the wrong-lock failure mode. The coarse seed now checks the mask-overlap
  seed against the strength of the edges under it, which removed the 94 px
  lattice lock on N1492091163: RingEdgeNav converges 1.1 px from operator
  truth, and a two-star StarRefineNav moves the fused answer to 1.70 px in du
  from truth (the sparse-star family's behavior). The two Encke gap frames
  were never wrong locks: the error entered through the rank-1 axis, which
  the ring model's 45-degree normal quantization tilted by up to 22.5
  degrees; with the axis taken from the line they land on operator truth at
  the `low` tier, the ring orbit channel's.
  Exposure was limited to sub-25 km/px Saturn scenes by the ring routing
  decision: at and above 25 km/px radial resolution the whole Saturn system
  feeds the annulus composite, which a 131-frame clean-truth head-to-head
  measured wrong on zero accepted answers while the per-edge fit ran
  5% / 13% / 56% / 100% wrong-when-accepted in the 300-1000 / 100-300 /
  25-100 / 0-25 km/px bands (measured before the edge-strength check).
  Below 25 km/px, against the published F ring bundle's pointing on 1,041
  held-out frames, the seed check took right answers from 692 to 877 and
  wrong ones from 283 to 129, the alias locks across the edge among them
  from 188 to 4; the 129 left are mostly along nearly straight edges (115,
  median 3.0 px, almost all at the medium tier), with 5 seeds on the
  neighboring cataloged edge (7 to 49 px), 4 alias locks across the edge
  (3.8 to 45 px, under #373) and 5 answers carried by star techniques. A
  same-code control on 150 of those frames reproduced `main` exactly, so the
  counts carry no run-to-run noise. Two remedies for the
  along-edge error (a nearest-prediction seed band, and inflating the
  along-edge variance in the covariance the ensemble fuses) failed on the
  263-frame tuning sample, taking it from 179 right and 68 wrong to 170 and
  80; the untried alternative, lowering only RingEdgeNav's own confidence, is
  tracked as #755. The annulus-gate recalibration (accepting
  more of its correct answers, #566) and a correlation-picks-basin /
  edge-polishes hybrid (#567) are filed follow-ons; measured as a basin
  selector below 25 km/px, the annulus correlation was right across the edge
  on F ring movie frames but 30 to 115 px wrong on four of the library's ring
  frames, so the hybrid as filed would regress them. The 25 km/px cut is no
  longer the intended routing: on 2026-10-28 the operator ruled on #822
  (Priority 2) that resolved catalog edges are to be navigated at coarse
  scales too, after the full spokes run refused 601 of 936 NAC frames with
  sharp catalog edges in view. That work waits with the frozen stack.
- **#755** — RingEdgeNav reports along-edge error on nearly straight edges
  at the medium tier with a sub-pixel sigma: 115 of the 1,041 held-out F
  ring frames, a median 3.0 px along the edge. Two remedies failed on the
  263-frame tuning sample (under #346 above); the untried one lowers only
  RingEdgeNav's own confidence on such an edge. #766, in the frozen stack,
  carries a fix (the edge's covariance taken from the arc's own geometry).
  It also carries the along-edge part of the planted-shift sweep (raw
  RingEdgeNav rows median 4.2 / 1.7 px v/u on frames whose edge sets are
  straight or nearly so). #476, the planted-shift round trip, closed on its
  criterion as the operator restated it on 2026-10-27 (done, #756), the
  across-edge component, which is met: RingEdgeNav's own round trip is a
  median 0.03 px and, with the rank-1 axis taken from the line, the
  ensemble's is 0.04 px (it was 0.39 px, worst 17 px). The five
  `ring_only_flat` constraint pins that recorded the quantized angle were
  re-pinned along the line's own normal on 2026-10-27.
- **#482** — BodyDiscCorrelateNav misses by up to ~1 px on a
  weakly-constrained axis. It is the residual the shift-equivariance fix
  (#447) leaves once it closes the coarse-grid and boundary-pinning halves.
- **#350** — N1484593951 misses the offset tolerance by ~2 px after the
  recalibration. The issue names a second frame, N1686349893, whose
  disagreement the measurement attributes to a stale pin on #483 instead.
- **#373** — the RingEdgeNav coarse seed against competing edge
  populations (polarity-blind). The edge-strength seed check covers the
  populations weaker than the cataloged edge (ring texture, glow contours).
  Still open: a stronger competitor such as a body limb, the held-out frames
  still wrong across the edge (5 neighboring-edge and 4 alias locks),
  `N1467344214_2`, not re-measured against the issue's gate, and whole movies
  locked across the edge (a 2007 spokes movie, #778). #779, in the frozen
  stack, checks the seed against nearby edges.
- **The frozen ring, star, and ensemble stack** — seven pull requests (#766,
  #779, #773, #774, #782, #784, and #789), frozen by the operator on
  2026-11-06 until a cross-mission benchmark exists; on revisit the
  principled pieces (star photometry, smear, ring-edge axes) are re-landed and
  measured on it, and each ensemble rule earns its place there. Every issue
  the stack addresses stays open until code that fixes it reaches `main`. By
  subject: ring edges and the annulus (#822, #761 a ring edge 34 km off the
  catalog, #778, #831, #770 spoke movies 2.5-3.6 px from the spokes bundle,
  #760 WAC spoke movies barely better than SPICE); stars (#762, #764, #765,
  #768 star smear from repeated C-kernel attitudes, #769, #771, #772, #780,
  #781, #785, #787, #852, #862, #864, #865, #868, and the operator choice in
  #880 of which stars judge a lone star's absent-star test); the ensemble
  (#767, #783, #786, #869); and the visible limb that #789 places (#858 and
  #878, the altitude as a property of the body and wavelength calibrated from
  a full Cassini run, then #857, #855, #859, #860, #861).
- **#128 / #150** — the strategic limb-navigation redesign and the ~0.1 px
  limb systematic (shared with Track A's WS-10; design first, validate
  against real images before touching). The measured bias attributes the
  systematic to limb-darkening / photometric roll-off: the edge DT localizes
  the gradient ridge, which sits ~0.5 px inside the geometric silhouette by
  an illumination-dependent amount. The strongest fix is a photometric-limb
  fit (predict the limb-darkened-disc-convolved-with-PSF profile), tracked
  on #150; #128 is the fuller redesign across all body types and
  illuminations. On real frames the fitter contributes only ~0.1 px while
  spacecraft-position / ephemeris error dominates (0.4-1.7 px), so the
  higher-leverage pointing-kernel side (Track D #50) was built first and is
  delivered. #128 also carries landing the held shift-equivariance fix
  (#447): the edge-offset correction that keeps its limb answers from moving
  inward, the simulated body model adopting the same sub-pixel vertex rule,
  and the library pins the fix moves (#483). The simulated side of the limb
  question, checking a simulated limb's gradient peak against real limbs and
  separating the simulated body model from the renderer, is Track S's (#842,
  #843).
- **#282** — a ~0.05 px, one-pixel-period sub-pixel ripple rides on top of
  the directional bias; a higher-order / matched-filter sub-pixel edge
  estimator would remove it. Precision refinement, secondary to #150.
- **#283** (done, #840) — the pixel-center convention mismatch between the
  simulated `BODY_DISC` predicted-center metadata and the renderer.
- **#25** — model blurring for very-high-resolution bodies (investigation).
- **#239** — sub-5 px body policy: decided as expected-failure curation;
  what remains is a targeted cohort scan (predicted diameter at or below
  ~5 px, single-body scenes) to find a qualifying frame, as library-growth
  work.
- **#338** — decision: the highly-irregular exclusion discards a
  ground-truth terminator fit on N1853392805; choose among accepting the
  2-px-class ground truth, keeping TERMINATOR_ARC for SPICE-known
  synchronous rotators, or shape models (#23).
- **Sparse star fields** — measured on all 1030 frames of a Cassini F ring
  movie against the published bundle's own per-frame pointing. After #754
  star navigation navigates 74.9% of them, none more than 2 px wrong, against
  the bundle's 96.6%, and the refusals come in runs of up to 25 frames
  (#621); most refused frames show one usable star or none. On the frozen
  stack's code the movie navigates 1006 of its 1030 frames, through the fixes
  for cosmic-ray hits crowding the detections (#764) and faint stars against
  ring texture (#769). Open beside them: Galileo's broad stars failing an
  isolation test sized for Cassini (#753), a wide-sigma one-star answer
  merging disagreeing witnesses (#841), and grounding the lone-star brightness
  bound in photometry (#844).
- **#130** — star limiting-magnitude calibration against real fields
  (coordinate with #233's measured-SNR work — same frames, same tooling).
- **#836 / #994** — the star model sometimes emits one catalog star twice
  under one feature ID; a likely cause is a catalog query whose magnitude
  bins are inclusive at both ends (#994). **#830** sets one offset window
  per camera for every technique. **#863** — the star pair search neither
  bounds nor compares a fitted rotation; it waits with #521 on the rotation
  convention.
- **#591** — the sparse 16x16 ring pre-check can skip a frame whose rings
  fall between its samples, so the frame is never offered a ring model.
- **The correlation techniques** — the disc and ring annulus correlations
  share one correlator. The disc's sub-pixel refinement cannot leave a
  +-0.5 px window and reports the window edge unflagged (#791, Priority 2;
  the ring annulus half was fixed by #1003, and the disc stays as it is by
  decision until a refinement beats it on real frames); an annulus answer the
  quadratic fit cannot refine stays on the whole pixel with nothing marking
  it (#1002); a constant brightness offset can make the masked correlation
  reject every shift (#997); the quality metric measures its background over
  a window 20x the search window, which is a question about what the metric
  means before it is a defect (#585, which also carries the missing minimum
  background size, #941); and RingAnnulusNav's answer on one frame flips
  under a 1e-8 s time change (#951).
- **#577** — the observation center RA/DEC accessor the star model reports
  through; **#579** — the logged star-smear summary is a frame-center vector,
  which reads zero for a rolling frame.
- **#824** — a calibrated camera block defaults its missing-pixel marker to
  NaN where every real calibrated product uses 0; **#815** — the image
  classifier, called without the missing fraction, reads the peak before the
  marker pixels are filled.
- **Titan haze fit refinements** — Titan navigates autonomously via the
  haze solar-symmetry method, validated on an 82-frame Cassini cohort. Four
  measured follow-ups remain: the arc ray reach is sized by the full search
  window rather than by where the limb can be, costing rays on large
  well-framed frames (#403); the flat arc-residual cap is a size-dependent
  gate that cannot simply be raised (#404); the extreme-phase edge of the
  working range is uncharacterized (#401); and the main rings are masked as
  opaque, refusing frames visible through the C ring or the gaps (#402). The
  symmetry axis is found by searching a backplane for the sunward direction
  where it can be computed (#594). Two capability extensions are deferred:
  the self-calibrated haze-radius table that would remove the dominant
  along-track error and answer the wavelength-dependent-haze-top question
  (#397), and methane surface-window (CB3) cartographic correlation as a
  refinement stage (#398).
- **Gates and the body-witness veto** — the combined polarity gate keys on an
  LM convergence flag that reads false on most healthy fits and flips with
  sub-pixel details of the model, so it rejects or keeps fits on an
  effectively arbitrary signal (#747); the multi-body exclusion counts bodies
  that have results rather than bodies in the frame (#749); and whether a
  body's witness blob needs its own overflow limit, separate from its limit as
  an answer, is a decision (#748).
- **#400** — the ensemble merge and tier logic have never been exercised on
  a strongly anisotropic covariance; the haze fit reports a 0.36 px by
  1.02 px oblique ellipse whose orientation varies with the sun direction.
- **#447** — body techniques do not return the shift they are given, which is
  the round-trip residual every corrected-pointing consumer inherits. The fix
  (a sub-pixel silhouette probe, and a per-axis NCC-quadratic fallback for
  saturated refinement) is ported onto `main` as the branch
  `rf_shift_equivariance_v2` and held there, by operator ruling, as the base
  for #128 rather than merged. Over 66 library frames it cuts
  `BodyLimbNav`'s median round-trip residual from 0.118 / 0.082 px to
  0.051 / 0.047 px (v / u). But on the five library frames that have both an
  independent star answer and a limb result, it moves the limb answer inward
  by a mean 0.27 px and farther from the stars on four of them. The model
  limb now sits on the geometric limb, and the image edge sits inside it
  (#150); `main`'s whole-pixel vertices were partly canceling that offset.
  Merged alone it would put the bias into every limb answer, so it lands
  with #128. The rest of the work is filed as #476 (since closed), #482 and #483.

**Parallelism:** parallel with Track A, except that the ring-lock family
(#346, #373, #755) and the sparse-star-field family (#621)
gate the agreement study -- a technique that reports a
confidently wrong answer moves the cross-technique disagreement WS-1 measures
-- and #288 gates reading any Track A cohort with confidence. The frozen
stack's issues wait for the cross-mission benchmark.

### Track C — Statistics, QA, and the accuracy checkpoint

**Goal:** navigation quality is continuously measured, not assumed.

The statistics system is built, tested, and documented in the user guide, and
it now reports over a stream of records rather than over a database, so it
runs with or without an index. The library coverage-matrix invariant is done
(#240, by #833). Remaining: the cross-check records only a yes/no
primary-technique flag (#340); two reporting defects the seam work surfaced —
a file that could not be retrieved is counted by a tree-backed report and by
no index-backed one (#533), and the report retains more than it prints under
`--top-n` (#535); and the standing practice of re-running the library
cross-check after every calibration-affecting change, which can only be read
as "no new deltas against `main`" while 6 of 75 library frames are red.
Small track, mostly done, listed separately because it is the program's QA
instrument.

### Track D — Capability completion (decision gates first)

**Goal:** every capability the docs imply either works and is validated,
or is explicitly scoped out in the capability matrix.

Titan navigation is delivered: the haze solar-symmetry method ships as a
model, a technique, and a simulated-Titan renderer, validated against an
82-frame Cassini cohort and a 700-scene planted-truth campaign, with the
published bound (1 px cross-track, 3 px along-track) confirmed by
star-anchored evidence. Its two deferred capability extensions, the
self-calibrated haze-radius table (#397) and methane surface-window
cartographic correlation (#398), are listed with the haze fit's other
refinements under Track B, where the engineering plan carries them.

Logging is delivered across every pipeline program: one main logger per
program run and one image logger per image, per-module levels, four
independently-controlled sinks, and one identical flag set whose defaults also
live in the configuration file, so learning the logging surface for one
program teaches it for all of them. Cloud-task workers are silent at the
terminal and complete in their log files. Three defects the work surfaced are
open: a mosaic cloud task still reports success when every image in it failed
(#418), the GUI viewers print library log records to stdout through
pdslogger's handler-less fallback (#423), and `sd_create_bundle_cloud_tasks`
is unwired and should be removed rather than fixed (#424). A log's file name
is stamped in UTC while the records inside it are stamped in local time, and
nothing in the file says so (#731). Three follow-ups the design deferred are
also open: the config namespace that `logging` was lifted out of is still
organized on no stated axis (#427, deferred with the rest of the navigator
config work, below), an upstream registry-eviction request to
`rms-pdslogger` (#428), and extending the same surface to the `util/` tooling
(#429).

The headline capability item is delivered in both halves: the navigator
records each image's corrected attitude as a C-matrix and `sd_create_ck`
ships updated-pointing SPICE C-kernels as a product, and the backplane and
reprojection consumers apply the recorded C-matrix whenever a usable one
exists — with the pixel offset as the documented fallback for
fitted-rotation, offset-only and malformed records, and uncorrected pointing
with the reason recorded when neither is usable — closing #50 and #188. The
designs are archived at `plans/archive/CK_KERNEL_PLAN_2026-08-04.md` and
`plans/archive/CMATRIX_READERS_PLAN_2026-08-09.md`; current behavior is
`docs/user_guide/user_guide_ck_kernels.rst` and its dev-guide companion.
What that leaves is the kernel follow-ups: the remaining half of the oops
API swap (#433), whose offset-to-rotation construction waits on
SETI/rms-oops#213 now that the convention conversion goes through oops, one
rotation convention about the image
center (#434) -- which is upstream of the wider distortion cohorts (#561) and
of any decision to fit rotation again -- with the static-twist FK/IK pair
behind it (#435, #436), SPICE database registration
(#437), the interior-epoch fidelity bound through an adaptive record cadence
(#440, #444) with its per-instrument characterization (#455), the
kernel-input items (#446, #448, #468), and a memory bound on `sd_create_ck`
(#513). One of them is a navigation defect rather than kernel work and is
listed under Track B (#521), deferred while no instrument fits rotation.

The results index is delivered too (#430, #487, #507). It is an optional,
rebuildable index over the results tree, so programs stop reading one JSON
document per image on a cloud root, and no program requires it: a run that
names none reads the tree exactly as it always has. What it actually changed
is broader than the index — every program that reads a navigation record now
reads it through one seam over both storages, with one rebuild, one opener and
one enumerated list of where the two storages answer differently. The design
is archived at `plans/archive/RESULTS_INDEX_PLAN_2026-08-04.md`; current
behavior is `docs/user_guide/user_guide_results_index.rst` and its dev-guide
companion. Its follow-ups divide into three groups: capability extensions
nobody is blocked on (a document column for bundle generation #464, a schema
wide enough for the curation triage tool #465, a `--since` selector #467,
`sd_offset` writing its own rows #486, a pruned-rows question #542, and the
three programs that take the results filters but cannot name a results index
#736), operational questions with a decision in them (a documented workflow
for getting the index to cloud workers #466, the consumer open that takes a
write lock it never needs #462), and correctness or hygiene items (#472, #493,
#496, #497, #501, #512, #514, #515, #516, #528, and #531, #533, #534, #536,
#538, #540, #541, plus a stale index that passes the version gate and fails on
the first read #587, a results tree reached through a symlink getting a
different stub on each pass #578, and a URL with no scheme that should mean a
local SQLite file #572).

Some items start with an operator decision, because each is a scope
commitment:

| Decision | Then the work is |
|---|---|
| **Backplane content** (#28 family): finalize the backplane set and formats | #55, #54, #57, #77, then the generator hardening (including the product-correctness defects #251, #252, #253 found by the #241 test suite). |
| **Metadata document precision** (#730): whether any value is rounded when the metadata document is written | Store every value at full precision and leave rounding to reports and logs, or name what makes a rounded value different; either way the two fields holding the offset agree, and the metadata chapter's rounding table changes with the answer. |

**PDS4 output bundles are required for all four instruments** — not a
scope decision. The Cassini ISS path runs end to end over a synthetic
cohort and is the reference the other three are generalized from: the bundle
product, the readme, and the context, document, schema, miscellaneous,
`spice_kernels` and index collections, with conforming inventories, real
epochs and named targets in the data labels, the backplane FITS described as
an archived file, the bundle name and version taken from configuration, and
an integrity pass (`sd_create_bundle check`) that holds the whole tree to the
PDS4 schemas, the Schematron rules and its own tables, gated by a test. The
design that produced it is archived at
`plans/archive/PDS4_DRAFT_BUNDLE_PLAN_2026-09-08.md`. Voyager, Galileo and
New Horizons hit not-implemented walls.

What the Cassini path still owes: the run over a real COISS volume, which the
PDS `validate` tool judges and which carries the cohort choice, registered
DOIs and a fresh navigation with it (#708); the
`cassini:ISS_Specific_Attributes` mission area, filled from the observation
block (#717), which is empty today because the fourteen names the mapping
asks of a PDS3 index row are not columns of one; a
holdings tree for the synthetic cohort so `sd_create_bundle` can enumerate it
(#609); a global navigation index in the bundle, deferred by the operator on
2026-09-15 (#710); and the spelling gate reading the schemas the package
ships (#705). Then the generalization — per-mission label templates, LID
builders, and collection machinery (#53 with #67, #79, #30, #63). Beside them
are the bundle's user guides, one LaTeX template and one guide per instrument
(#595-#599); the scope questions of what a bundle says about an image that
did not navigate (#600), one with no backplanes (#720), and which kernels the
metakernel lists (#677); citing the calibrated PDS4 product once its bundle
exists (#687); checking a bundle at a remote results root (#716); a dataset
without PDS4 support ending `sd_create_bundle` in a traceback
(#614); and the backplane viewer showing rad/pixel planes in radians where
every other angular plane is in degrees (#611). Distinct from this, **PDS4
*input*** (#34) — reading PDS4-archived data instead of PDS3 — is treated
like any other future instrument: the archives do not exist yet, their
creation is external development outside our control, and input support is
*not* required for project completion; when an archive appears, its support
replaces the PDS3 source for that instrument.

The reprojection and mosaic products have their own open list, all of it
independent of the decisions above: mosaics show brightness steps between
images because phase is not modeled and nothing levels the seams (#616), the
Minnaert exponent is fixed at 0.5 where the icy satellites need another value
(#617), each planet's ring target is a literal rather than configuration
(#618), the products record sub-solar and sub-observer longitude west while
their grid is east (#634), reading a product from remote storage never fetches
it (#638), body backplanes read silhouette or visible extent depending on
which truth field is present (#701), and ring `--zoom` averages sub-samples of
the output grid where body `--zoom` interpolates the source image (#718).
`sd_mosaic` carries handling for a missing navigation results root that it can
never reach, and an unset root ends in a traceback rather than the one-line
message the configuration-error reporting exists to print (#733); and
`--resolution-threshold` accepts a value below the 1.0 its contract states as
the minimum (#738).

Plus, not gated on decisions: the capability matrix itself (#231),
cloud-operation audit (#108, #67, #141, #142, and the cloud-task items the
logging work left open, #418 and #424), performance and safe parallelism for
campaign scale (#236, #103, #134, #126), and config validation (#118: its
foundation, the shared schema core #1021, is in the simulator plan's Phase 2a;
#1029 and #1032, which type the sections and delete the packaged keys nothing
reads, are deferred by the operator (2026-11-17) with the namespace
reorganization #427, and the navigator config is left alone for now; the
loader's one rule for which keys it checks, #1001, closes with #1032's last
step), and two command-line inconsistencies: `--profile`
produces no profile in `sd_offset` and does nothing in `sd_backplanes` (#734),
and `sd_create_ck` and `sd_stats_report` cannot select images with the options
the other six programs share (#735). The user-guide completion items are
delivered.

**Parallelism:** decisions can be made any time; the resulting work is
independent of Tracks A-B except that #232 (end-product accuracy) wants
the backplane-content decisions settled and the PDS4 bundle
generalization done.

### Track E — Test and documentation debt

**Goal:** no shipped stage without tests; no doc a future maintainer
cannot follow.

- Zero-coverage stages: backplane CLI (#241), PDS4 CLI (#242).
- Untested star-conflict logic (#243); real-image baselines beyond one
  frame (#174).
- Summary-PNG unit tests (#177).
- The image-library regression reconciliation (#288). In the local
  integration environment it is not reduced to the deliberately-red pins:
  6 of 75 frames disagree on `main` (measured 2026-11-16, when PRs #999 and
  #1003 each ran the library suite on `main` beside their branches), each
  attributed to an open issue or an operator decision, which is the state
  Track A's evidence has to be read against, and it is what blocks the only
  place a built product is compared between the results tree and the index
  (#547). Two are Galileo frames whose recorded truth is the rotation-fitted
  answer that #560 retired; re-recording that truth from an outside
  measurement, or pinning both red under #434, is an operator decision
  (#744). Two are `primary_technique` pins on #483 that name the answer the
  code correctly declines to prefer, one is #350's limb bias, and one is
  #338's highly-irregular exclusion. The ten star frames whose pins the
  sparse-star-field and star-centroid fixes moved were ratified by the
  operator on 2026-10-31 and re-pinned with #754, as was the
  `primary_technique` pin of each of #346's two Encke gap frames, and every
  frame re-pinned on 2026-10-27 passes.
- Suite coverage is 79% and unenforced against a stated 90% (#548). The
  shortfall is almost entirely PyQt6 widget code. This is an operator call
  before it is an implementer's: raise the number, or ratify a lower floor
  and gate on that one.
- Tests that cannot fail, which this project has now hit repeatedly: #516
  asserts on the row a root-blind write would keep. Re-ratcheting the library
  pins the shift-equivariance fix moves is #483; those deltas arrive with the
  held fix branch when #128 lands it.
- CI reliability: test jobs that hang for six hours or segfault in the Qt
  tests (#832), dataset unit tests that download from the live archive and
  fail CI when it is slow (#820, Priority 2), and a ring performance test
  running close to its 8 s limit (#814).
- Test-suite hygiene: 29 test modules over the 1000-line cap, counted
  2026-11-17 (#525), POSIX-only constructs behind a Windows-support claim
  (#473), fixture clock seconds that do not follow from their epochs (#530),
  consolidating `test_record_source.py` onto the shared fixtures (#524), no
  check in the repository reaching `cloud_support/` (#574), whether mypy's
  `warn_unreachable` is worth its seven suppressions (#576), Python 3.15 not
  declared or tested (#928), no test of the cross-volume early stop or its
  Voyager opt-out (#870), and shared-directory test cleanup failing on a
  directory symlink (#953).
- Docs: Sphinx nitpicky-clean CI (#129, with the gate the rules already
  require but nothing runs, #438) and terminator-doc verification (#122).
  Smaller doc defects: the README's Features entries link to no guide (#545),
  four technique guides name a `dt_fitting.py` that is a package (#549),
  config placeholder comments still carry an internal codename (#470, #471),
  a wrong `marker_value` docstring for Voyager ISS (#807), and a LORRI
  filespec check that describes the wrong directory format (#939).
  Python source carries em dashes and arrows the docstring rule prohibits with
  nothing gating them (#714), the Qt modules document 7% of their method
  arguments where the rest of `src` documents 65% (#715), 62 of 64 dev-guide
  chapters lack the class diagram the rules require (#562), and
  `library_entry` resolves the PDS3 holdings root in a different order than
  `DataSetPDS3` while saying it does not (#612). Renumbering the configuration
  files, whose numbering has grown with odd gaps, is left for later discussion
  (#712). The project rules and checks have fallen behind the `rms-devenv`
  template they came from, and some now contradict this repository's own
  configuration (#723). The open-file clamp on a retrieval batch is a
  workaround to remove once filecache bounds its own lock batches (#725). The
  per-instrument chapters are written, one per instrument in each guide, as is
  the metadata-JSON format chapter (docs/user_guide/user_guide_metadata.rst,
  with a staleness-guard test), along with the filters / uncertainty /
  troubleshooting dev-guide pages, the mosaic-viewer API-reference pages, and
  the curation-tooling language pass.
- Tooling parity: give the `util/` programs the logging surface the
  pipeline programs now have (#429) — several run for hours and report
  through bare `print()`.

**Parallelism:** entirely parallel with everything; good filler between
larger items. #241/#242 should precede any serious PDS4/backplane work in
Track D so that work lands on tested ground.

### Track F — Remaining instruments, features, and hardening

**Goal:** the other three instruments reach Cassini's proven level; the
enhancement backlog and code-quality tail are burned down.

- **Instruments** (after Track A proves the Cassini spine): Voyager star
  navigation (#19), Galileo star navigation (#18),
  LORRI PSF and product policy (#2, #138, #33), outer-planet ring models
  (#82, #81, #83, with the sub-pixel ring feature type Uranus needs, #908),
  rotation-pyramid cost (#126), degradation classifier
  (#181), per-instrument calibration extension (re-run #230 per
  instrument as library frames land).
- **Features:** BOTSIM (#27), star streaks (#22), backplane-reader repo
  (#107), PDS4 input when external archives exist (#34 — replaces the
  PDS3 source per instrument; not required for project completion),
  cartographic/bootstrap navigation
  (#184 — explicitly far off), polarity-aware ring matching (#183),
  chaotic-rotator poses (#187), manual-nav dialog redesign (#186),
  gated-feature PNG styling (#185), stop-after-features flag (#182),
  body shape models (#23), sim polish (#78 CraterMaker craters and #152
  diffraction spikes, both out of the simulator plan's scope; #157 and #158
  are done, and #84 and #151 are scheduled in Track S).
- **Hardening/cleanup** (any time, mostly small): #15, #21, #38, #39, #65, #92, #96-#103, #105, #109, #110, #119, #135, #137, #140, #143, #144, #147, #155, #212 (done, #834), holding the model stage's backplane cache to one model's worth (#584), hoisting the remaining function-level imports or recording why each stays (#590), a second config load in one process keeping the first (#604), `draw_line` truncating a fractional coordinate instead of rounding to the containing pixel (#709), plus the GUI viewers printing library log records to stdout (#423), the upstream `rms-pdslogger` registry-eviction request (#428), Cassini BOTSIM pairs defined two different ways (#494), stating the encoding wherever a document or text file is read or written (#518), hashing into a navigation document's provenance only the instrument configuration it used, where every instrument's is hashed now (#711), removing the `AttrDict` `_IS_IMMUTABLE` marker once oops stops writing its mutability bookkeeping onto foreign objects (#552), a body-occluder bounding-box pre-filter (#388), choosing the source of the observation block's spacecraft clock counts (#628), and naming the configuration files in the wheel's package data (#706).

**Parallelism:** hardening is permanent filler. Instrument work waits for
Track A's Cassini verdict only in the sense that there is no point
calibrating three more instruments with an unproven method; the
star-navigation bug fixes (#19, #18) can start any time.

## 5. Suggested global order

0. **First:** the evidence-integrity work — #288's six red library frames
   (closing it is closing #483, #744, and the frames' owning issues), and the
   confidently-wrong families: the ring-lock family (#346, #373, #755) and
   what is left of the sparse-star-field family (#621), frames showing one
   usable star or none. Like the coarse-lock family, the star family moves
   what the agreement study measures. The fixes for #373, #755, and #621's
   refusals sit in the frozen stack, so this step needs the cross-mission
   benchmark the operator asked for before any further work goes into them.
1. **Then, in parallel:** Track A items 1-4 (library growth, sim
   realism campaign, agreement-estimator real-frame follow-ups, distortion
   feed-in), with the gates above respected -- the coarse-lock family before
   the agreement study collects, and #288 before any cohort is read as
   evidence rather than as a comparison against `main`. Beside them, Track S
   finishes the realism-record regeneration and runs Phase 1b (the
   restructuring that moves no pixel), then Phases 2 and 2a. The operator
   decisions in section 6 go across as a batch — they cost nothing to decide
   early and unblock scoping; #923's three choices are decided.
2. **Next:** Track A item 5 (agreement study, bulk layer first), with
   Track E test-debt and Track B remainder as parallel fill. Track S runs
   Phase 2b (guard scenes), Phase 3 (the generator repair and refit), and
   Phase 3b (the renderer restructuring) alongside.
3. **Then:** Track A items 6-8 (CI tiers, real-anchored recalibration,
   accuracy tail). This is the calibration finish line: confidence and
   uncertainty defensible against reality. Track S's Phases 4 to 6 (physics,
   rendering from real geometry, the transfer check) close the realism
   verdict (#227) and supply the single-technique accuracy the last part of
   the recalibration (#230) needs.
4. **Then:** Track D build-out per the decisions; Track F instruments,
   re-running the now-proven calibration per instrument.
5. **Last:** capability matrix finalized (#231), documentation
   completion, end-product accuracy (#232), and the simulator's validation
   catalog with a per-instrument accuracy number for each library class that
   passed the transfer check (Track S Phase 7) — the deliverable's evidence
   package.

**Effort estimate:** Track A is multi-week at agent pace; its two largest
remaining items (sim-realism campaign, agreement study) are serialized only
through the estimator's real-frame follow-ups between them. Track S is the
next-largest block: its plan adds about 25 restructuring issues before the
physics, most of them small and pixel-neutral, and the physics and per-camera
twins after them are multi-week. Tracks B+C+E are one to two weeks of
interleavable small/medium items, apart from the frozen stack, which waits for
its benchmark. Track D depends on scope decisions; Track F is another
multi-week block dominated by per-instrument calibration repeats. Operator
hands-on time is dominated by library votes and the decision gates, not by any
implementation.

## 6. Operator decision gates (collected)

1. **Ring orbit-uncertainty severity (#316)** — the ring radial channel
   prices catalog fit residual as a fully correlated whole-edge
   displacement, which the catalog's own mode-fitted epochs suggest is
   several-fold over-severe. It demotes five operator-verified Keeler
   frames to low. Decide: ship at the conservative default, ratchet
   `rings.orbit_radial_sigma_correlated_fraction`, or implement the
   wander decomposition. Reversible by config either way.
2. **Titan haze navigation ratification (#407)** — the method is
   implemented and validated; what needs a decision is a bundle of
   mid-implementation specification changes, three acceptance bounds the
   evidence argues with, and three staged curation artifacts (an overlay
   review batch, six library nominations, and a `titan_haze` scene-class
   recommendation).
3. **Highly-irregular terminator fit (#338)** — choose among accepting the
   2-px-class ground truth for resolved highly-irregular bodies, keeping
   TERMINATOR_ARC for SPICE-known synchronous rotators, or shape models
   (#23), then a session implements the choice on N1853392805.
4. **Simulator realism verdict (#227)** — close #227 once the calibration is
   fitted on the realism-anchored renderer configuration (#309) and the
   realism evidence underwrites the shipped calibration; gated on #309, and
   in the simulator plan completed by Phase 5.
5. **Coverage floor (#548)** — the suite measures 79% against a 90% floor two
   shipped plans' acceptance criteria assert, and nothing enforces either
   number. Raise coverage to the stated floor, or ratify a lower one and gate
   CI on it. Leaving both the claim and the absence of a gate in place is the
   one option that keeps saying something untrue.
6. **Cassini predicted kernels (#459)** — whether Cassini navigation should
   run from the predicted rather than the reconstructed kernels. A branch
   (`origin/rf_ck_cassini_predicted`, nine commits, last touched 2026-08-07)
   exists with no pull request; say whether it is live before it drifts
   further from `main`.
7. **New Horizons pointing family (#468)** — whether the merged family should
   be declared reconstructed on its comment-area evidence rather than staying
   `UNCLASSIFIED`.
8. **The written-product comparison (#547)** — the one place a built product
   is compared between the results tree and the index runs against no frame;
   whether to repoint it or to accept the gap.
9. **Getting the index to cloud workers (#466)** — which documented workflow
   to adopt, since it decides how #464 is deployed.
10. **Where the confidence scale saturates (#557)** — the combined confidence
    caps at 0.99 on the ordinary two-technique agreement, and WS-5's monotonic
    calibration map cannot separate what the cap collapsed, so calibration
    does not settle it later.
11. **What a bundle says about an image it cannot fully describe** (#600 one
    that did not navigate, #720 one with no backplanes, #677 which kernels
    the metakernel lists) — three scope commitments that decide what the
    generalization to the other three instruments is generalizing.
12. **Metadata document precision (#730)** — whether the metadata document
    should round any value at all. Today pixel values are stored to four
    decimal places and confidences to three, while the top-level offset beside
    them is stored at full precision, so two fields holding one quantity can
    disagree in the last place. Decide whether every value is stored at full
    precision, with rounding left to reports and logs.
13. **The witness blob's overflow limit (#748)** — whether a body's witness
    blob in the body-witness veto keeps its own overflow limit, derived from
    the veto's tolerance, separate from the limit that withholds it as an
    answer.
14. **A sigma ceiling on the low tier (#751)** — whether the lowest confidence
    tier carries a sigma ceiling, and at what value. It is a calibration
    choice rather than a derivation, between the 18.85 px that legitimate
    blob-only answers on irregular bodies reach and the far-off blobs a
    rejected limb would expose, and it interacts with WS-5.
15. **The two Galileo library frames' truth (#744)** — `C0059894800R` and
    `C0059899900R` record the rotation-fitted answer that #560 retired.
    Re-record their truth from an outside measurement, or pin them red under
    #434.
16. **The SQLite write lock a consumer takes (#462)** — probe a read-only
    open with a read, or keep the rule that refuses in both modes.
17. **The ten star-frame library pins** — decided 2026-10-31 as recommended,
    and applied with #754 (done).
18. **The frozen ring, star, and ensemble stack** — frozen 2026-11-06 until a
    cross-mission benchmark exists. One decision waits with it: which stars
    judge a lone star's absent-star test (#880). Checking three library
    truths the image stars contradict against a star overlay (#867) is the
    operator's and does not wait: it can run any time.
19. **Ring edges at coarse scales (#822)** — decided 2026-10-28: navigate
    resolved catalog edges at coarse scales too (done; the work waits with
    the frozen stack).
20. **Simulator decisions** (asked one at a time, in the simulator plan's
    order): #923's three choices, decided 2026-11-18; for #1017, before Phase
    2a's detector work, whether each pair (banding, dark signal, hot pixels)
    is one effect or two; and in Phase 2, the ruling #644 asks for on what a
    planted roll turns. Decided: pydantic as the engine of the shared schema
    core (#1021, 2026-11-16), and fixing the cameras' residual distortion in
    the upstream camera models (#970, 2026-11-14). On hold: simulated star
    saturation (#854).
21. **The navigator config** (#427, #1029, #1032) — deferred by the operator
    on 2026-11-17. When the work resumes, which sections the namespace
    reorganization covers, and whether it lands before they are typed, is
    the decision to record on #427.
22. Recurring: library batch votes; agreement-study frame selection; tier
    re-blessing after #230.

## 7. Issue index (open work by track)

Every open issue, listed exactly once by the track that owns it: 439 issues
as of 2026-11-17. Of the 304 the previous index listed, 28 have closed
since; they are listed after the table with the change that closed each,
together with the four the previous index had already removed (#476, #622,
#623, #639). 162 open issues were added, most of them the simulator's
(Track S) and the frozen stack's (Track B). The count is given so a reader
can tell a stale index from a current one at a glance.

| Track | Count | Issues |
|---|---|---|
| A — validation & calibration | 44 | #153, #172, #174, #176, #223, #225, #226, #227, #229, #230, #232, #233, #235, #309, #316, #319, #321, #322, #324, #334, #341, #342, #355, #358, #359, #360, #361, #377, #380, #399, #405, #407, #561, #637, #644, #646, #675, #727, #751, #759, #788, #866, #867, #871 |
| S — the simulator | 103 | #84, #151, #310, #311, #333, #343, #790, #793, #794, #804, #806, #816, #825, #835, #837, #842, #843, #848, #850, #854, #856, #873, #874, #875, #877, #881, #882, #883, #884, #885, #890, #891, #892, #893, #894, #895, #896, #897, #898, #899, #901, #902, #903, #904, #905, #906, #907, #909, #910, #914, #915, #918, #919, #920, #921, #922, #923, #925, #931, #935, #942, #945, #947, #956, #957, #960, #961, #967, #970, #975, #977, #979, #986, #987, #988, #990, #995, #1000, #1005, #1006, #1007, #1008, #1009, #1010, #1011, #1012, #1013, #1014, #1015, #1016, #1017, #1018, #1019, #1020, #1021, #1022, #1023, #1024, #1025, #1026, #1027, #1028, #1030 |
| B — navigation correctness | 85 | #25, #128, #130, #150, #239, #282, #338, #346, #350, #373, #394, #397, #398, #400, #401, #402, #403, #404, #447, #482, #521, #557, #558, #566, #567, #577, #579, #585, #591, #594, #621, #747, #748, #749, #753, #755, #760, #761, #762, #764, #765, #767, #768, #769, #770, #771, #772, #778, #780, #781, #783, #785, #786, #787, #791, #815, #822, #824, #830, #831, #836, #841, #844, #852, #855, #857, #858, #859, #860, #861, #862, #863, #864, #865, #868, #869, #878, #880, #941, #949, #951, #994, #997, #1002, #1033 |
| C — statistics & QA | 3 | #340, #533, #535 (plus the standing cross-check and campaign-report practice) |
| D — capability completion | 98 | #28, #30, #53, #54, #55, #57, #63, #67, #77, #79, #108, #118, #126, #141, #142, #231, #236, #251, #252, #253, #411, #418, #424, #427, #433, #434, #435, #436, #437, #440, #444, #446, #448, #455, #459, #462, #464, #465, #466, #467, #468, #472, #486, #493, #495, #496, #497, #501, #512, #513, #514, #515, #520, #528, #531, #534, #536, #538, #540, #541, #542, #572, #578, #587, #595, #596, #597, #598, #599, #600, #609, #611, #614, #616, #617, #618, #634, #638, #677, #687, #701, #705, #708, #710, #716, #717, #718, #720, #730, #731, #733, #734, #735, #736, #738, #1001, #1029, #1032 |
| E — test & docs debt | 44 | #122, #129, #177, #241, #242, #243, #288, #379, #391, #429, #438, #443, #470, #471, #473, #483, #516, #524, #525, #530, #545, #547, #548, #549, #554, #562, #574, #576, #612, #712, #714, #715, #723, #725, #744, #807, #814, #820, #832, #870, #928, #939, #953, #1035 |
| F — instruments, features, hardening | 62 | #2, #15, #18, #19, #21, #22, #23, #27, #33, #34, #38, #39, #65, #78, #81, #82, #83, #92, #96, #97, #98, #99, #100, #101, #102, #103, #105, #107, #109, #110, #119, #134, #135, #137, #138, #140, #143, #144, #147, #152, #155, #181, #182, #183, #184, #185, #186, #187, #388, #423, #428, #494, #518, #552, #584, #590, #604, #628, #706, #709, #711, #908 |

Priority census across all seven tracks: no Critical, 39 Essential, 164
Important, 171 Useful, 50 Minor, 15 Defer, which accounts for all 439. Every
open issue carries exactly one Priority and one Effort label and at least one
each of A-type and B-location.

Cross-listed items (listed once above, noted here): #150/#128 sit in Track B
and also serve Track A's limb-bias workstream (WS-10); the confident-wrong
ring-lock family (#346, #373, #755) and the sparse-star-field family (#621)
sit in Track B but gate the Track A study; #103/#134/#126 serve both Track D
performance and Track F hardening, and #126 is also on the simulator's Phase 1
list, with #945; #513 and #520 are results-index work in Track D that lands in
the kernel writer and the reprojection package respectively; #521 is a
kernel-facing defect owned by Track B because the fix is in the navigator; the
per-instrument guide chapters exist and each Track F instrument workstream
extends its own pair in the same change; #174 baselines are Track A
infrastructure delivered as Track E test work; #288 and #547 are one
dependency seen from two tracks. The simulator plan also schedules Track A
issues (#153, #223, #227, #309, #322, #341, #342, #355, #359, #377, #644,
#646), Track B navigator defects its Phase 1 found (#791, #941, #997, #1002),
and its parallel tracks' issues (Track A's #225, #230, #334, #788; Track F's
#81, #82, #83, #908); each is listed once, under the track named here. #1021,
#1023, and #1024 are Track S issues that the engineering plan's config entries
(Track D) build on; #1029 and #1032 are Track D issues the simulator plan
defers. Track S also holds three navigator defects the simulator found, kept
as expected failures until fixed (#790, #835, #975).

**Closed since the 2026-10-14 reconcile** (removed from the open lists above,
kept here with the change that closed each):

- #157 simulated missing data as whole lines — closed 2026-11-04 with no
  pull request, already covered by the telemetry artifact modes
- #158 smooth-shaded mesh bodies — closed 2026-11-01, landed in #313
- #212 a flaky sim baseline under xdist — closed 2026-11-02 by #834
- #234 detector noise for the calibrated (I/F) render path — closed
  2026-11-08 by #888
- #240 the library coverage-matrix invariant — closed 2026-11-02 by #833
- #283 the simulated `BODY_DISC` pixel-center mismatch — closed 2026-11-02
  by #840
- #290 the body render-time budget — closed 2026-11-01 with no pull
  request, already met by the topographic renderer's split-resolution path
- #325 simulated stars shining through dark limbs — closed 2026-11-03 by #846
- #329 the 1-LSB calibrated-product floor — closed 2026-11-07 by #886
- #330 zero cosmic-ray transients — closed 2026-11-07 by #887
- #331 per-scene hot pixels — closed 2026-11-04, landed in #846
- #332 one PSF kernel per instrument — closed 2026-11-06 by #849
- #335 no canonical environment for committed sim baselines — closed
  2026-11-02 by #834
- #336 simulator suites never running in Actions — closed 2026-11-02 by #834
- #344 the constant haze brightness — closed 2026-11-03 by #846
- #345 the truth-side-noise echo into `instrument_config` — closed
  2026-11-02 by #838
- #409 one scene-coordinate convention — closed 2026-11-02 by #840
- #426 a stale committed sim render — closed 2026-10-31, fixed by #680
- #476 RingEdgeNav's planted-shift round trip — closed 2026-10-29 by #756
- #622, #623, #639 the sparse-star-field defects — closed 2026-11-03 by #754
- #625, #626, #627 scene features that rendered wrong — closed 2026-10-28
  by #796
- #629, #630, #631, #632, #633 scene keys that said one thing and meant
  another — closed 2026-10-28 by #819
- #641 stars on pixel indices, bodies on pixel corners — closed 2026-11-02
  by #840, as a duplicate of #409
- #750 the simulated body's overflow handling — closed 2026-11-02 by #845

---

*History: this plan supersedes `plans/archive/ROADMAP_2026-07-12.md` (the
issue-ordered pipeline build-out) and
`plans/archive/FULL_PROGRAM_AFTER_MANUAL_WORK_2026-07-11.md` (the
six-track task inventory), consolidating both into one top-level view.
The validation methodology and the curation playbook were already single
documents and remain in place as the detail layer.*
