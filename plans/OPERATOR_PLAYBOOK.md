# Operator Playbook: driving the agreement study and the calibration finish

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` and `plans/SIM_IMPROVEMENT_PLAN.md`. Last reconciled
against the tracker on 2026-11-17. Work through Section 0 first; after
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
- **The star-frame library pins (ten frames, Section 1)** -- **decided
  2026-10-31, as recommended, and applied (PR #754, merged 2026-11-03):** the
  nine tier pins took the new tiers, `N1571691490_1` took its truth from the
  three stars, `N1492091163_1` was pinned as a success at the stars' offset,
  and the two #346 Encke gap frames' `primary_technique` pins were moved with
  their tiers. The question as it was put: the sparse-star-field
  fixes and the star-centroid fix moved ten star-frame pins. Nine keep
  their offsets: three one-star answers are capped at low, four frames whose
  star techniques share their stars drop from high to medium (within 0.12 px
  of `main`; in `N1489603610_2` the two-star search now answers too, 0.12 px
  from two stars the pin is 0.3 to 0.5 px from), and two whose refinement now
  keeps fewer stars drop from medium to low (0.14 and 0.27 px from truth,
  against 0.81 and 0.66 px on `main`, which is inside the truth's own
  resolution). In `N1571691490_1` a two-star solution replaces the body-blob
  answer the sidecar pins, 9.4 px away, and three stars confirm it. Ratify
  the new tiers and re-record `N1571691490_1`'s truth, or name the tier
  changes you reject.
- **#744 the two Galileo library frames' truth**: `C0059894800R` and
  `C0059899900R` record the rotation-fitted answer that #560 retired, so both
  are red against the translation `main` now reports. Re-record their truth
  from an outside measurement, or pin them red under #434. The frames carry a
  real twist of about -0.39 deg, so a translation alone cannot match their
  stars everywhere in the frame.
- **#600 / #720 / #677 what a bundle says about what it cannot describe**:
  an image that did not navigate (#600), a navigated image with no
  backplanes (#720), and which SPICE kernels the metakernel lists (#677).
  Each is a scope commitment, and together they decide what the
  generalization to Voyager, Galileo and New Horizons is generalizing.
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
- **#730 whether the metadata document rounds anything**: pixel values are
  stored to four decimal places and confidences to three, while the top-level
  offset beside them is stored at full precision, so two fields holding one
  quantity can disagree in the last place. Store every value at full
  precision and leave rounding to reports and logs, or say what makes a
  rounded value different.
- **#748 the witness blob's overflow limit**: a body's witness blob is
  withheld above the overflow fraction derived for its role as an answer,
  while its role as a witness in the body-witness veto survives more
  clipping. Keep one limit, or give the witness its own, derived from the
  veto's tolerance.
- **#751 a sigma ceiling on the low tier**: any answer that clears the low
  tier's confidence floor is reported `success` however large its
  uncertainty. Legitimate blob-only answers on irregular bodies reach 18.85
  px, so any ceiling is a calibration choice; it interacts with the #230
  recalibration.
- **#462 the SQLite write lock a consumer takes**: every open of a SQLite
  results index probes with a write transaction, so a reader beside a
  running ingest waits 30 s and then fails with a message that prescribes a
  server. Probe a read-only open with a read instead, or keep the plan's
  rule that refuses in both modes.
- **#880 which stars judge a lone star's absent-star test**: one faint star
  showing nothing refutes a lone star today; three measured judge sets each
  recover answers and each add wrong lone answers. Choose the set. It
  matters once the photometric star limit of the frozen ring/star PR stack
  lands (#766), so it waits with that stack.
- **#867 three library truths the image stars contradict**
  (`N1597846115_2`, `N1550270436_1`, `N1633925572_1`): check each against a
  star overlay and re-record it if the stars confirm the new position. Found
  while validating the frozen stack, but it does not wait for it: the check
  needs only a star overlay and can run any time.

**The simulator's decisions** (order and context in
`plans/SIM_IMPROVEMENT_PLAN.md`; ask them one at a time):

- **#923, before Phase 1b starts** -- **decided 2026-11-18.** The three
  choices were: (1) have the simulator call the navigator's image
  classifier, or move the pure decision to a neutral module both import;
  (2) move the test-only `artifact_counts.py` beside `realism/` or under
  `tests/`; (3) whether the "Phase 10 §" labels outside the simulator are
  in scope. Ruling: (1) a neutral module; (2) under `tests/`; (3) out of
  scope, tracked in #1035.
- **#1017, before Phase 2a's detector work** -- for each effect modeled
  twice (banding, dark signal, and hot pixels), whether the scene's `noise`
  knobs and the artifact mode are one effect or two.
- **#854 simulated star saturation** -- on hold by the operator
  (2026-11-10). Do not start it, or file the real-frame gain-2 question
  separately, until the hold is lifted.
- **#970 residual distortion on tiny-residual cameras** -- **decided
  2026-11-14:** fix the camera models upstream instead
  (SETI/rms-host-galileo#3, SETI/rms-host-cassini#4,
  SETI/rms-host-new-horizons#1, SETI/rms-host-voyager#2, all unassigned).
  Each camera keeps its simulated residual until its updated model is in the
  host SpinDoctor uses.
- **#427 the config namespace** -- deferred with the navigator config work
  (2026-11-17), together with #1029 and #1032; nothing in the simulator plan
  waits on it (Section 3, logging follow-ups).

```bash
gh issue comment 316 --body "Decision: <ship default | ratchet fraction | wander decomposition>"
gh issue comment 407 --body "Decisions: <per the enumerated list>"
gh issue comment 338 --body "Decision: <accept 2px GT | keep TERMINATOR_ARC | shape models>"
gh issue comment 548 --body "Decision: <raise to 90% | ratify <N>% and gate CI on it>"
gh issue comment 547 --body "Decision: <pick frame <name> | close behind #288>"
gh issue comment 744 --body "Decision: <re-record truth from <measurement> | pin red under #434>"
gh issue comment 459 --body "Decision: <rf_ck_cassini_predicted is live | abandon the branch>"
gh issue comment 468 --body "Decision: <declare reconstructed | leave UNCLASSIFIED>"
gh issue comment 466 --body "Decision: <ship the SQLite file | run PostgreSQL>"
gh issue comment 600 --body "Decision: <per the enumerated options>"
gh issue comment 720 --body "Decision: <per the enumerated options>"
gh issue comment 677 --body "Decision: <per the enumerated options>"
gh issue comment 748 --body "Decision: <one limit | a separate witness limit derived from the veto tolerance>"
gh issue comment 751 --body "Decision: <no ceiling | ceiling of <N> px on the low tier>"
gh issue comment 730 --body "Decision: <store full precision | keep rounding for <values>, because <reason>>"
gh issue comment 462 --body "Decision: <probe a read-only open with a read | keep the write probe in both modes>"
gh issue comment 880 --body "Decision: <judge set>"
gh issue comment 923 --body "Decisions: <classifier home>; <artifact_counts.py home>; <Phase 10 labels in or out of scope>"
gh issue comment 1017 --body "Decisions: banding <one | two>; dark signal <one | two>; hot pixels <one | two>"
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

### 0.1c Labels and assignees: check five inferred priorities

The tracker is fully labeled. Every open issue carries exactly one Priority
and one Effort label and at least one each of A-type and B-location, and
every one but #928 (Python 3.15 is not declared or tested) has `rfrenchseti`
as an assignee; assign that one. The `Priority TBD` label is unused and can be
deleted from the repository if you want it gone.

Five priorities were assigned by inference from these plans rather than by
you, and are the ones worth a glance: **#53** PDS4 bundle generator parent as
Essential (the plans call output bundles required for all four instruments,
and three of the four are not started); **#28** backplane generator parent as
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

1. **#288 resolution** — 6 of 75 frames are red on `main` (measured
   2026-11-16 and 2026-11-17 by PRs #1003 and #999, the same six with the
   same messages on each branch and its base), and every one is attributed
   and owned. The ten star-frame pins, `N1492091163`, and the two Encke gap
   frames' `primary_technique` pins were re-pinned on the operator's
   2026-10-31 decisions (PR #754), and every frame re-pinned on 2026-10-27
   now passes (Section 1). Of the six, two are the Galileo frames awaiting
   the #744 decision (Section 0.1); one is `N1853392805`, held out by the
   highly-irregular-body policy and owned by the #338 decision; one is
   `N1484593951`, a real limb bias deliberately pinned red (#350); and two
   are stale or wrong `primary_technique` pins batched onto #483. Nothing is
   unattributed, and no `expected.*` field is moved to match current behavior.
   Closing the issue is therefore closing #483 and the frames' owning issues,
   not a separate investigation.

**Group 2 — the confidently-wrong families.** Two campaigns, each best done
as one piece. Both gate the Track A study, which consumes ensemble output at
scale.

*The ring-lock family:* **#346** (three library frames once locked onto the
wrong ring feature; all three now pass on their pins, and it stays open as
the record of the confident wrong-lock failure mode, with the two F ring
observations its 2026-09-12 comment adds), **#373** (the coarse seed
against competing edge populations) and **#755** (along-edge error on
nearly straight edges);
**#476** (RingEdgeNav's round trip under a planted shift) closed with the
work described here.
The coarse seed now checks the overlap seed against edge strength
and rejects fits on faint scattered-light contours, which removed most of the
lattice locks (alias locks across the edge went from 188 to 4 of 1,041
held-out F ring frames). The rank-1 axis now comes from a straight edge's own line rather than
the ring model's 45-degree per-vertex normals, which removes the along-edge
leak into the ensemble's answer on flat ring frames; the five
`ring_only_flat` pins that recorded the old axis were re-pinned along the
line's own normal (Section 1). What remains is the along-edge error of nearly straight edges (115 of
1,041 held-out F ring frames, a median 3.0 px along the edge, almost all
reported at the medium tier with a sub-pixel sigma), a seed on the
neighboring cataloged edge (5 of the 1,041, 7 to 49 px), and the 4 alias
locks left (3.8 to 45 px, under #373). Two remedies for
the first were measured on the tuning sample and failed; the untried one,
lowering only RingEdgeNav's own confidence on such an edge, is tracked as
#755. #476's criterion was restated on 2026-10-27 as the across-edge
component, where it is met; the along-edge part of the raw sweep rows is
#755's.
The Saturn
routing decision (annulus composite at and above 25 km/px radial resolution,
per-edge fit below) limits the family's exposure to the sub-25 km/px regime
today, but it is no longer the intended routing: on 2026-10-28 the operator
ruled on #822 that resolved catalog ring edges are to be navigated at coarse
scales too, after the full spokes run refused 601 of 936 NAC frames with
sharp catalog edges in view. #822's work waits with the frozen stack.
**Not dispatchable while frozen:** the fixes for #755 (PR #766) and #373
(PR #779) sit in the ring/star PR stack (#766, #773, #774, #779, #782, #784,
and #789), which the operator froze on 2026-11-06 pending a cross-mission
benchmark. The issues stay open; start that benchmark before any new work
on them.

*The sparse-star-field family:* **#621** (star navigation refused half the
F ring movie `ISS_006RI_LPHRLFMOV001_PRIME`, in runs of up to 30 frames).
PR #754 (merged 2026-11-03) did not meet the issue's criterion on `main`:
there star navigation navigates 771 of the movie's 1030 frames (74.9%),
every one within 2 px of the published F ring bundle, against the bundle's
96.6% by stars, and the refusals still come in runs of up to 25 frames. The
2026-10-22 run that navigated 1006 frames (97.7%, longest refusal run two
frames) used code a263e966, which combined #754 with the frozen #766 branch
and with #756 (merged 2026-10-29), so nothing it measured is on `main`. The
remainder, cosmic-ray hits crowding the star field's detections (#764) and
faint stars against ring texture (#769, PR #782), waits with the frozen
stack. It is measured against the published F ring bundle's own per-frame
pointing, so it has an outside answer to fix against.

**Group 3 — parallel fill, any order, any number at once.**

- **Results index:** #515 with #516 (the root-blind share write and the test
  that cannot catch it); #501, #512, #514, #536 as one seam-cleanup
  batch; #528 with #531 as a metadata-provenance pair; #493 and #496 (run-level
  conditions reported per image); #587 and #578 (a stale index that passes
  the version gate, and a symlinked tree that gets a different stub on each
  pass); #736 (three programs that take the results filters cannot name a
  results index); then the tail #472, #497, #524, #572,
  and #533, #534, #535, #538, #540, #541.
- **PDS4 bundle:** #717 (fill the Cassini mission area from the observation
  block; it is empty today because fourteen of the names it maps are not
  PDS3 index columns); #609
  (give the synthetic cohort a holdings tree so `sd_create_bundle` can
  enumerate it); #705 (the spelling gate reads the shipped schemas); #614 (a
  dataset without PDS4 support ends in a traceback); #716 (check a bundle at
  a remote results root). #708, the draft run over a real COISS volume,
  carries the cohort choice and needs your frame approval first.
- **Simulator:** dispatch in the order of `plans/SIM_IMPROVEMENT_PLAN.md`.
  The next sessions are (1) the end-of-stack realism-record regeneration,
  with #961's sweep figures, in progress in another session (its
  prerequisite #989 landed in PR #1004 on 2026-11-17), and then (2) Phase 1b,
  the pixel-neutral restructuring, starting with #923's shared scene fixtures
  and #1005's package layout (`sim/forward/` becomes `sim/render/`); #923's
  three choices are decided (Section 0.1). Phase 1b starts only after the
  regeneration lands.
- **Simulator scene fidelity** (the batch this group listed): four features
  rendered something other than what they name — #625 (`limb_relief_rms`
  carved radial wedges out of the disc), #626 (`polyhedral_mesh` bodies
  rendered as banded shells), and #627 (the two haze terms seamed the halo at
  the equator), **done (PR #796)**, and #644 (a planted roll turns positions
  but not smear, companions or sky field; it also asks for a ruling on what a
  roll turns), open — plus #646 (the limb-bias harness measures a ridge half
  a pixel from the deployed one), open; #750 (the simulated body model kept
  the disc and witness blob that production withholds on a body overflowing
  the frame), **done (PR #845)**; and the scene-language batch #629, #630,
  #631, #632, #633, **done (PR #819)**, and #641 with #409, **done (PR #840;
  #641 closed as a duplicate of #409)**. #644 and #646 sit in the simulator
  plan's Phase 2.
- **Reprojection and mosaic products:** #634 (longitudes recorded west on an
  east grid), #638 (a remote product is never fetched), #701 (body backplanes
  read silhouette or visible extent depending on which truth field is
  present), #617 (Minnaert k fixed at 0.5), #618 (ring target from
  configuration), #616 (unleveled mosaic seams), #718 (the two `--zoom`
  meanings), #733 (`sd_mosaic`'s unreachable handling of a missing navigation
  results root, and the traceback that an unset root produces), #738
  (`--resolution-threshold` accepts a value below 1.0).
- **Test debt:** #241 and #242 first, because the plan wants tested ground
  under any serious PDS4 or backplane work; then #243, #177, #524,
  #525, #530, #473.
- **Image library curation:** #727 (land the ring-truth curation campaign:
  branch `origin/rf_ring_curation_campaign` holds 954 rows of your screening
  verdicts and a 75-frame ring shortlist and was never opened as a PR; it
  needs a check against `main` and a developer-guide paragraph on the
  staging directory).
- **CI:** #324, the agreement-estimator tests, which do not run in Actions.
  #336 (the data-independent simulator suites) is **done (PR #834)**, and
  #426 (the stale committed render) is **done (fixed by PR #680)**. #391 pins
  the lint tools so a release cannot turn `main` red on its own.
- **Products:** #520 (move the pointing selection out of the reprojection
  CLI package); #495 (raw-product dataset names for Cassini ISS).
- **Command-line programs:** #734 (`--profile` produces no profile in
  `sd_offset` and does nothing in `sd_backplanes`); #735 (`sd_create_ck` and
  `sd_stats_report` cannot take the image-selection options the other six
  programs share).
- **Docs and cleanup:** #545, #549, #470, #471, #494, #518, #714, #715,
  #562, #612.

**What looks dispatchable and is not.** #483 and #547 wait on #288. #129
(Sphinx nitpicky-clean) wants #443 settled first, since that decides whether
`spindoctor.cli` subpackages get autodoc pages at all. #418 is a policy
question before it is a coding one. #464's affordability depends
on #466. #552 waits on an upstream oops fix. The work in the frozen
ring/star PR stack (#755, #373, #321, and the star issues it addresses)
waits on the cross-mission benchmark. #239's cohort scan is agent work but the
sidecars it produces need your votes. And the two largest Track A
items, #225 and #172/#235, are gated on your frame approvals and batch votes, which
is what makes Group 1 worth starting today.

## 1. The library regression, and the deliberately-red set

**Read this before trusting a library run.** In the local integration
environment 6 of 75 sidecars disagree on `main` (measured 2026-11-16 and
2026-11-17 by PRs #1003 and #999): `C0059894800R` and `C0059899900R` under
#744, `N1484593951_2` under #350, `N1487595731_1` and `N1686349893_1` under
#483, and `N1853392805_1` under #338, exactly the rows of the table below
not marked re-pinned. PR #1304 (Cassini's measured star limit) adds three
more, so 9 of 75 disagree once it lands: `N1669606966_1` and
`N1597846115_2` under #1302, and `N1461997416_1` under #1303. Every one is
owned by an open issue or by your decision. The two under #744 await your decision in Section 0.1. (On
2026-10-31 the count was 19: the ten star-frame pins, `N1492091163`, and the
two Encke gap frames' `primary_technique` pins have since been re-pinned on
your decisions, PR #754.) Three of the six are
unresolved judgments rather than settled pins — the two
`primary_technique` flips on #483, and `N1484593951`'s limb bias — which
is why #288 is open, and why the gate a navigation-affecting branch
clears is **no new failures against `main`**: run the suite on `main`
first, then on the branch, and account for the difference. Do not
re-ratchet a sidecar to match current behavior, and do not read a
green-looking subset as a pass.

Reconciling #288 is prerequisite to two other things: #483 (re-ratcheting the
pins the shift-equivariance fix moves) and #547 (the one place a built
product is compared between the results tree and the results index, whose
frame no longer navigates).

| frame(s) | disagreement | owner |
|---|---|---|
| N1492091163 | **Re-pinned 2026-10-31 (PR #754): pinned as a success at the stars' offset; passes.** Expected `failed`; navigates to (5.21, -26.02) at medium tier, 1.70 px in du from operator truth: RingEdgeNav converges 1.1 px from truth and a two-star StarRefineNav moves the fused answer 0.7 px in du (the sparse-star family). The expectation predated a working ring fit | #346 |
| N1669606966_1, N1597846115_2 | **Expected failures from PR #1304.** A lone-star match on a non-star source, refined to confidence 0.92, outweighs the body techniques: 16.7 px off on N1669606966 (Hyperion, 0.38 s CL1+GRN), 3.1 px off in v on N1597846115 (0.07 s clear) | #1302 |
| N1461997416_1 | **Expected failure from PR #1304.** The methane pair CL1+MT1's offset puts the 0.26 s frame's limit at 7.2, so only the V 6.5 star is predicted and its match is judged spurious; the frame fails | #1303 |
| N1465178827_2, N1486991973_1, N1492091163_1, N1506004925_1, N1519300016_2, N1572105349_1, N1598065103_1, N1828122657_1, N1503576135_2, N1548432342_1, N1555145539_1, N1633925572_1, N1777325846_1, W1449079117_1, N1644757096_1 | **Re-pinned 2026-11-28 (PR #1304, operator decision to keep Cassini's measured star limit): pass.** More predicted stars move the primary technique to a star technique, or raise the tier, or both; every offset stays within tolerance. PR #1304 lists each old and new value | Section 0.1 (decided) |
| N1853392805 | highly-irregular exclusion discards the terminator fit; the ring fit is right across the edge but about 8 px off along it, so the ensemble reports `conflicted` | #338 |
| N1484593951 | dv error 2.687 px, a limb bias pinned red on purpose | #350 |
| N1487595731_1 | expects BodyDiscCorrelateNav primary, gets BodyLimbNav; the pin is stale | #483 |
| N1686349893_1 | expects BodyLimbNav primary, gets BodyDiscCorrelateNav, which matches operator truth to 1e-4 px; the pin names the worse answer | #483 |
| N1867601758_1, N1867602424_1 | **Re-pinned 2026-10-31 (PR #754): RingEdgeNav primary; both pass.** Expected BodyBlobNav primary, got RingEdgeNav; their tiers were re-pinned on 2026-10-27, their primaries were not | #346 |
| C0059894800R, C0059899900R | du error 5.709 px and dv error 3.051 px: the recorded truth is the rotation-fitted answer #560 retired; the star techniques, StarUniqueMatchNav now among them, agree with one another 6 px from it | #744 |
| N1461997416_1, N1555145539_1, W1449079117_1 | **Re-pinned 2026-10-31 (PR #754) to low; pass.** Expected the medium tier, got low: the offset rests on one star, so it is `single_star_only` and capped at low; offsets unchanged | Section 0.1 (decided) |
| N1459552012_1, N1488823805_1, W1580760393_1, N1489603610_2 | **Re-pinned 2026-10-31 (PR #754) to medium; pass.** Expected high, got medium: the star techniques on the same stars are one witness, whose blended confidence is under the high tier's 0.85; offsets within 0.12 px of `main` and within 0.42 px of truth | Section 0.1 (decided) |
| N1484688342_1, N1489256001_1 | **Re-pinned 2026-10-31 (PR #754) to low; pass.** Expected medium, got low: refinement no longer measures the sources that had pulled each answer 0.81 and 0.66 px from truth, so each lands 0.14 and 0.27 px from it, inside the truth's own resolution, on fewer stars with a sigma over 2 px | Section 0.1 (decided) |
| N1571691490_1 | **Truth re-recorded 2026-10-31 from the three stars (PR #754); passes.** Expected low at the body blob's offset, got medium 9.4 px away: three stars measure 0.24 to 0.75 px from the new offset and show nothing at the pinned one | Section 0.1 (decided) |

`N1530185128_1` left this table when #639 closed: StarUniqueMatchNav pairs
its two smeared stars again and the answer lies 0.1 px from truth.

`N1633925572_1` left this table on 2026-10-27. It was held red on its
`medium` tier pin on the reading that a wrong RingEdgeNav lock set its tier,
but since the Saturn routing decision (150.6 km/px routes it to the annulus
composite) it no longer runs RingEdgeNav. Its `low` tier now comes from
BodyLimbNav and a one-star StarUniqueMatchNav, fused to a sigma of 2.49 px
(2.50 px with the ring lock), so the operator re-pinned the tier to `low`;
whether that sigma is sized right is a sigma-calibration matter (#788). It
passes on `main` (it is not among the six). Whether its recorded truth
carries a half-pixel datum error is part of #867 (Section 0.1). The frames in
the table not marked re-pinned are pins, not regressions; do not re-ratchet
them until the owning issue closes.

**Proposed re-pins for the photometric Cassini star limit.** #866 tracks the
library sidecars the frozen ring/star PR stack moves (19 frames). Its table,
"Proposed re-pins for the photometric Cassini star limit", is in this file
on the stack's branch (`rf_ring_edge_along_755`, PR #766) and arrives with
it; it is not on `main`. No sidecar has been edited for it, and nothing is
re-pinned until the stack's last PR merges and the library is run on
`main`.

**Re-pins for the rank-1 axis fix (operator decision, applied 2026-10-27).**
RingEdgeNav now takes a straight ring edge's rank-1 axis from the line itself
instead of the rings model's 45-degree per-vertex normals. Five
`ring_only_flat` sidecars recorded the quantized angle as their constraint
normal, so their pins encoded the old axis, and the answer the old code
reported on four of them was 14 to 19 px off across the real edge. The new
pins keep the raw fit each frame was verified at and re-express it along the
line's own normal (from the fixed `aggregate_edge_normal_angle_deg`;
regenerated independently by the reviewer to within 0.004 px); each frame's
raw fit projects onto the new normal within 0.004 px of the new value, and
the pinned `(dv, du)` is the foot of the new constraint line. The original
votes were cast on overlays rendered at the old representative point, 14 to
19 px across the edge on four of the five, so a fresh operator look at each
raw fit is still welcome. All five pass on `main` (none is among the six red
frames measured on 2026-11-16).

| frame | old pin: normal / along-normal | new pin: normal / along-normal | measured answer vs new pin |
|---|---|---|---|
| N1863267861_1 | 45.0 deg / -27.4128 px | 50.69 deg / -23.2398 px | 0.003 px |
| N1863268207_1 | 90.0 deg / -20.4941 px | 68.38 deg / -36.1736 px | 0.014 px |
| N1863269418_1 | 90.0 deg / -3.5032 px | 69.24 deg / -19.0444 px | 0.010 px |
| N1863269466_1 | 90.0 deg / 1.5777 px | 68.77 deg / -12.8190 px | 0.001 px |
| N1863269574_1 | 90.0 deg / 4.4777 px | 67.49 deg / -14.8007 px | 0.004 px |

The two Encke gap frames of #346 move the other way: with the axis fixed the
ensemble keeps the body blob and lands on operator truth (N1867601758_1 at
(2.90, -0.39) against (2.96, -0.80); N1867602424_1 at (2.48, 11.48) against
(2.50, 11.34)), and each then reported the `low` tier against a `medium`
pin, because the ring orbit channel puts the fused sigma just over 2 px.
Both tier pins were re-pinned to `low` on 2026-10-27, as N1867602424's own
sidecar note asked, and their primaries to RingEdgeNav on 2026-10-31 (PR
#754); both pass on `main`. N1492091163, #346's third frame, was pinned as a
success at the stars' offset on 2026-10-31 and passes too; #346 stays open as
the record of the wrong-lock failure mode (Section 0.3, Group 2).

**After any navigation-affecting merge, compare against `main` rather than
against the table:**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pytest tests/integration/test_autonomous_nav.py -m '' -n 4 --dist=loadfile
# Expect, while #288 is open: the same red set as main — the six frames
# above. Any delta either way must be attributed in the merging PR.
```

The thread pins are not optional and are not set anywhere in the repository —
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
  tier into every-PR CI and the full suite on a schedule, per WS-4." Related,
  both **done (PR #834)**: the data-independent sim suites run in Actions
  (#336), and the committed sim baselines have a canonical environment
  (#335).
- **#230 / WS-5** — re-anchor confidence on real evidence. The correlated
  ring-witness fix (#317) is done, so the calibration no longer trains
  against rows where two ring techniques on one catalog were fused as
  independent witnesses. **Still settle #316 before reading the Keeler
  tiers:** the tooling fits tier boundaries from the fused confidence
  scalar, and the orbit-uncertainty severity call moves five
  operator-verified frames across a boundary. **And the anchor itself is
  contaminated while the confidently-wrong ring locks stand (#558):** WS-5
  calibrates against WS-1, so the coarse-lock family has to close first or
  the recalibration trains on its errors. Then: "Re-run the
  calibration tooling against the agreement study's measurements per WS-5;
  retire the confidence_provisional marker where the evidence supports it;
  re-bless tiers with the operator." This is where the terminator's
  provisional label and the sim-anchored coefficients get their real-world
  upgrade.
- **Accuracy tail** — #233, #150/#128 (design first; see Section 3),
  plus #232; #234 (the detector-noise model for the calibrated render path)
  is **done (PR #888)**.

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
  limb." The implementation starts from `rf_shift_equivariance_v2`, the
  #447 fix held as #128's base, and lands it: the edge-offset correction, the
  simulated body model adopting the same sub-pixel vertex rule, and the pins
  the fix moves, which go to #483. The measurements are on #447.
- **#373 (coarse-lock remainder)** *(frozen: PR #779, in the ring/star
  stack, waits on the cross-mission benchmark; do not dispatch until then)*:
  "Make the RingEdgeNav coarse seed
  robust against competing edge populations per #373 where the edge-strength
  check does not reach: a competitor stronger than the cataloged edge, such
  as a body limb in the search window, and the held-out frames still wrong
  across the edge (5 neighboring-edge locks; 4 alias locks, N1492091385,
  N1851236541, N1492064782 and N1851226701). Re-measure `N1467344214_2`
  against the issue's gate."
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
- **Navigation correctness, in order**: the ring-lock family, #346, #373
  and #755, scoped to sub-25 km/px Saturn scenes by the routing decision,
  which the operator's 2026-10-28 ruling on #822 overrules (navigate
  resolved catalog ring edges at coarse scales too; #822's work waits with
  the frozen stack).
  #476 (RingEdgeNav's round trip under a planted shift) has closed on its
  criterion as restated on 2026-10-27, the across-edge component: measured
  across the edge normal, RingEdgeNav's own round trip on the library's ring
  frames is at the DT floor (median 0.03 px), and with the rank-1 axis taken
  from the line so is the ensemble's (0.04 px, worst 0.31 px; it was 0.39 px,
  worst 17 px). The raw sweep rows still carry the along-edge part (median
  4.2 / 1.7 px v/u), lying along straight or nearly straight edges, which
  #755 tracks. The five `ring_only_flat` constraint pins whose recorded
  normal angle was the old quantized axis were re-pinned (Section 1). The
  fixes for #755 and #373 are in the frozen ring/star stack (PRs #766 and
  #779), so this line waits on the cross-mission benchmark.
- **Gates and the body-witness veto**: #747 (the combined polarity gate keys
  on an LM convergence flag that reads false on most healthy fits; re-derive
  its discriminant against labeled healthy and mis-locked fits, adding the
  healthy frames it cites to the library first) and #749 (the multi-body
  exclusion counts bodies with results rather than bodies in the frame). #748
  is an operator decision first (Section 0.1).
- **Sparse star fields**: #621 is not met on `main`, where PR #754 (merged
  2026-11-03) navigates 771 of the movie's 1030 frames (74.9%) with refusal
  runs of up to 25 frames; the 97.7% run of 2026-10-22 combined #754 with
  the frozen #766 branch. The remainder, #764 (cosmic-ray hits crowding the
  star field's detections) and #769 (faint stars against ring texture, PR
  #782), waits with the frozen ring/star stack. All
  1030 frames of `ISS_006RI_LPHRLFMOV001_PRIME` are the measurement harness,
  and the published F ring bundle's per-frame pointing is the outside
  answer, so a fix is verifiable without operator adjudication. A
  single-star rule loosened to close #621 is where silent wrong answers come
  from; measure the wrong count beside the rate.
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
  logging surface); #731 (log file names are stamped in UTC and log records
  in local time). The navigator config work is deferred (operator,
  2026-11-17): #427 (reorganize the config namespace), #1029 (the
  navigator camera blocks), and #1032 (the remaining sections) wait. When
  it resumes, #427 needs a decision first: which sections it covers, and
  whether it lands before #1029 and #1032 type them. #1032 then asks
  #411's snake_case question before its technique PRs, and it deletes the
  unimplemented `bootstrap` section's placeholder keys unless the operator
  objects.
- **Sim realism residual (#227)**: the de-circularization is done and on
  main; #227 stays open only for the realism proof and closes at the
  operator's realism-verdict gate, itself gated on #309. #309
  (realism-configured multi-instrument campaign — biggest
  calibration-credibility win available) is the load-bearing step. Of the
  fidelity gaps it consumes, #325, #329, #330, #331, #332, #344, #345, and
  #290 are **done** (PRs #846, #886, #887, #846, #849, #846, #838, and the
  split-resolution body renderer), and #333, #341, #342, #343, and #377 are
  open. #310 (structural boundary enforcement) and #311 (mirror-parity guard)
  harden the partition. All of these are placed and ordered in
  `plans/SIM_IMPROVEMENT_PLAN.md` (#309 in Phase 3, #310 in Phase 3b);
  dispatch from there. Each issue body is a prompt basis.
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
  catches new leak shapes); **#621** (where the line between too little
  evidence and a solvable frame sits is a calibration judgment, and moving it
  the wrong way trades a visible failure for a silent one).
- **Mid-tier or below suffices:** library growth, the agreement study's
  bulk execution (once WS-0 hands it a proven estimator), #229, #311, #373,
  #130, the PDS4 bundle items, the simulator scene-fidelity batch, the
  logging follow-ups (#418, #423, #424, #429, #731), and the
  documentation/engineering items.

## 3c. Tracking-issue register

Open issues grouped by theme so none is lost to a PR body; the sequencing
hooks reference the sections above. All carry A/B/Priority/Effort labels
with assignee rfrenchseti.

**Evidence integrity (do first; everything else reads what these produce):**

- **#288** 6 of 75 library sidecars disagree on `main` (measured
  2026-11-16), all attributed: two to #744, two stale `primary_technique`
  pins on #483, one limb bias that stays red (#350), and the frame owned by
  #338; the ten star-frame pins and #346's frames were re-pinned on the
  operator's 2026-10-31 decisions (PR #754); blocks #483 and #547
- **#744** two Galileo library frames record the rotation-fitted answer #560
  retired, so both are red; re-record their truth from an outside measurement
  or pin them red under #434 (Section 0.1 decision)
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
- **#736** three programs that take the results filters cannot name a results
  index, so their filters always walk the tree
- **#472, #497, #524, #533, #534, #535, #538, #540, #541** the small tail

**Confident-wrong answers and the ensemble (sequence with #230/WS-5):**

- **#346** three library frames once locked onto the wrong ring feature;
  all three now pass: N1492091163 was pinned as a success at the stars'
  offset and the two Encke gap frames' tier and primary pins were moved
  (2026-10-27 and 2026-10-31, PRs #756 and #754); it stays open as the
  record of the wrong-lock failure mode, with two F ring observations wrong
  nearly everywhere RingEdgeNav carried them; exposure is limited to
  sub-25 km/px Saturn scenes by the routing decision
- **#822** NAC spoke frames show sharp catalog ring edges but navigate on
  none of them; the operator's 2026-10-28 ruling on it overrules the
  sub-25 km/px routing cut (navigate resolved catalog ring edges at coarse
  scales too), and its work waits with the frozen ring/star stack
- **#755** RingEdgeNav reports along-edge error on nearly straight edges at
  the medium tier with a sub-pixel sigma (115 of 1,041 held-out F ring
  frames, a median 3.0 px along the edge); it also carries the along-edge
  part of the planted-shift sweep that #476 left when it closed on the
  across-edge criterion; its fix is PR #766, frozen with the ring/star stack
- **#482** BodyDiscCorrelateNav misses by up to ~1 px on a weakly-constrained
  axis, the residual the #447 fix leaves (held on `rf_shift_equivariance_v2`
  for #128)
- **#394** shape-lock veto suppression trusts a star fix that could itself be
  wrong-locked, turning a safe `conflicted` into a confident-wrong `success`
- **#380** correlated-witness fusion collapses to a representative at rho=1;
  fit an explicit cross-covariance once #225 measures real-frame rho
- **#400** the ensemble merge and tier logic have never been exercised on the
  strongly anisotropic covariance the Titan haze fit reports
- **#621** star navigation refused half the F ring movie it was measured
  on; after PR #754 (merged 2026-11-03) `main` navigates 74.9% of it, with
  refusal runs of up to 25 frames; the 97.7% run combined #754 with the
  frozen #766 branch, and the remainder (#764, #769) waits with the frozen
  stack
- **#591** the sparse 16x16 ring pre-check can skip a frame whose rings fall
  between its samples
- **#747** the combined polarity gate keys on an LM convergence flag that most
  healthy fits fail, so it rejects or keeps fits on an arbitrary signal
- **#749** the multi-body exclusion counts bodies that have results rather
  than bodies in the frame
- **#748**, **#751** decisions (Section 0.1): the witness blob's own overflow
  limit, and a sigma ceiling on the low tier

**Simulator fidelity gaps (feed #309 and the sim follow-ups in Section 3;
every simulator issue is placed and ordered in
`plans/SIM_IMPROVEMENT_PLAN.md`, which is the register for the simulator):**

- **#325** simulated stars shone through dark limbs -- **done (PR #846)**
- **#329** simulated calibrated products floored at 1 LSB -- **done
  (PR #886)**
- **#330** instrument chains rendered cosmic-ray transients at zero --
  **done (PR #887)**
- **#331** simulated hot pixels were per-scene, not per-detector -- **done
  (PR #846)**
- **#332** PSF catalog had one kernel per instrument; binned/summed readout
  modes were inexpressible -- **done (PR #849)**
- **#333** four physical error axes are unmodeled
- **#290** body renderer exceeded the sim render-time budget on oversampled
  grids -- **done** (the split-resolution body renderer; closed 2026-11-01)
- **#377** sim rings are single annuli; build realistic nested-ringlet scenes
  and tests
- **#341** the campaign's scene mixture is authored, unvalidated against real
  frames
- **#342** star_psf_sigma is a 3.0 placeholder on Galileo, Voyager, LORRI
- **#343** the tuned NAC PSF wing may be absorbing operator registration error
- **#344** haze brightness was a module constant -- **done (PR #846)**
- **#345** a scene could echo truth-side noise into instrument_config with no
  validator warning -- **done (PR #838)**
- **#625** `limb_relief_rms` carved radial wedges out of the disc instead of
  roughening the limb; **#626** `polyhedral_mesh` bodies rendered as banded
  shells and the shipped Hyperion scene as a torus; **#627** the two haze
  terms seamed the halo at the equator -- all three **done (PR #796)**;
  **#644** a planted roll turns positions but not smear, companions or sky
  field (open; asks for a ruling on what a roll turns)
- **#646** the limb-bias harness measures a simulated ridge half a pixel from
  the deployed one
- **#750** the simulated body model had no silhouette overflow handling --
  **done (PR #845)**
- **#641** with **#409** stars sat on pixel indices beside bodies on pixel
  corners -- **done (PR #840; #641 closed as a duplicate of #409)**;
  **#629** one artifact key named three quantities; **#630** a fluence per
  cm^2 was documented per pixel; **#631** a Voyager scene rendered in I/F
  under a raw-DN marker; **#632** ring scene keys carried no units; **#633**
  the editor authored mesh scenes its own validator refused -- all five
  **done (PR #819)**
- **#1017** decision (Section 0.1) before the simulator plan's Phase 2a
  (#923 decided 2026-11-18); **#854** on hold (operator); **#970** decided (fix the
  camera models upstream)

**Calibration governance / CI (gate WS-5 and the CI tier in Section 2.3):**

- **#334** calibration has no armed falsification criterion and its real-frame
  gate is suspended (owns the Section 0.2 transfer-watch step)
- **#335** no canonical environment for committed sim baselines (0.99 vs
  0.81-0.84 across machines) -- **done (PR #834)**
- **#336** data-independent simulator integration suites never ran in
  Actions (relates to #229/WS-4) -- **done (PR #834)**
- **#340** library_crosscheck records only a yes/no primary-technique flag,
  not the winning technique
- **#426** a committed sim render was stale on `main` -- **done (fixed by
  PR #680; closed 2026-10-31)**

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
- **#427** the config namespace is organized on no stated axis; decide
  which sections it covers and whether it precedes their typing (#1029,
  #1032), when the deferred navigator config work resumes
- **#428** upstream registry-eviction request to `rms-pdslogger`
- **#429** give the `util/` tooling the same logging surface
- **#731** log file names are stamped in UTC while the records inside them are
  stamped in local time

**Library-frame reds and decisions (Section 1):**

- **#338** highly-irregular exclusion discards the ground-truth terminator fit
  on N1853392805 (decision)
- **#350** N1484593951's limb fit misses the offset tolerance by 2.687 px in
  `dv`, a real accuracy gap pinned red on purpose
- **#483** two `primary_technique` pins name a technique the current code
  correctly declines to prefer (N1487595731_1, N1686349893_1); re-ratchet
  them, or stop asserting `primary_technique` where two techniques agree
  within tolerance; the pins the #447 fix moves arrive with
  `rf_shift_equivariance_v2` when #128 lands it

**PDS4 bundle (the Cassini path runs end to end over the synthetic cohort;
these are what it owes and what generalizes it):**

- **#708** the draft run over a real COISS volume, judged by the PDS
  `validate` tool; carries the cohort choice, the DOIs and a fresh navigation
- **#717** the Cassini mission area, filled from the observation block: it is
  empty today because fourteen of the names it maps are not index columns
- **#609** the synthetic cohort has no holdings tree, so `sd_create_bundle`
  cannot enumerate it; **#705** the spelling gate would read the shipped
  schemas; **#614** a dataset without PDS4 support ends in a traceback;
  **#716** check a bundle at a remote bundle results root
- **#600**, **#720**, **#677** scope decisions (Section 0.1); **#710** (a
  global navigation index in the bundle) **decided 2026-09-15: deferred**,
  the prototype bundle finishes without it
- **#53** with **#67**, **#79**, **#30**, **#63** the generalization to
  Voyager, Galileo and New Horizons; **#595**-**#599** the user guides;
  **#687** cite the calibrated PDS4 product once its bundle exists

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
  about 2 px (navigation finding); PR #789 addresses it and is frozen with
  the ring/star stack
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
0.1 decisions (#316, #407, #338, #548, #547, #459, #468, #466, #462, #557,
    #600, #720, #677, #730, #744, #748, #751, #867; #880 with the
    frozen stack; simulator #1017; #923 decided)               (operator, minutes)
    star-frame pins: decided 2026-10-31 and applied (PR #754)   (done)
0.2 adopt transfer watch (#334)                                 (operator, minutes)
--- then, before Track A collects anything at scale
--- #288 library regression reconcile    (agent session; unblocks #483 and #547)
--- the confidently-wrong families: #346/#373/#755, and #621's re-check on
    main (#373 and #755 wait on the frozen ring/star stack's benchmark)
--- simulator, in parallel: realism-record regeneration with #961
    (in progress) -> Phase 1b, starting with #923 and #1005
--- then ---
2.1 library growth (batch-006 + continued)   (agent session; your votes gate it)
2.2 agreement study bulk   (after 2.1 cohorts; you approve frames)
2.3 CI tier, re-anchor confidence, accuracy tail (after 2.2)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal the program plan opens with.
