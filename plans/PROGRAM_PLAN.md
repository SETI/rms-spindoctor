# SpinDoctor Program Plan

*The order in which to work every open issue outside the simulator plan, grouped into phases, with
the condition that ends each phase. Last reconciled 2026-11-22 against the open issue list.*

---

## Goal

An end user can take raw archival images from Cassini ISS, Voyager ISS, Galileo SSI, or New Horizons
LORRI and produce, locally or in the cloud, precisely navigated pointing, reprojected mosaics,
per-pixel geometry backplanes, PDS4 bundles, preview images and metadata, and updated SPICE pointing
kernels. Every accuracy and confidence number is backed by published evidence, and every capability
is documented.

## How to read this plan

- Each open issue outside the simulator has one row, in the order it should be worked. `Exit` marks an issue the phase's
  exit condition needs. The other issues in a phase belong to its subject and can land later without
  holding the phase open.
- A finished row stays in place and is marked **Done**, with its pull request.
- Phases marked parallel can run beside the others. Otherwise a phase starts when its predecessors'
  exit conditions are met, except that any row that waits on nothing can start early.
- The priority is the issue's own GitHub label.
- Simulator work is tracked separately, in `plans/SIM_IMPROVEMENT_PLAN.md`, and has no row here.
  That covers changes to the simulator, its tests, and its report. 20 issues whose fix is
  navigator or calibration work have a row in both plans, because the simulator plan also names
  them. They are marked below.

| Document | Role |
|---|---|
| `plans/SIM_IMPROVEMENT_PLAN.md` | The simulator's issues, in phases |
| `plans/ENGINEERING_PLAN.md` | Implementation detail for the issues here |
| `plans/VALIDATION_AND_CALIBRATION_PLAN.md` | Methodology and acceptance criteria for Phases 7 to 9. Binding where it overlaps this plan |
| `plans/COHORT_CURATION_PLAN.md` | How the image library grows |
| `plans/OPERATOR_PLAYBOOK.md` | The operator's pending decisions and sessions |
| `plans/archive/PROGRAM_PLAN_2026-11-17.md` | The previous program plan, by track, with the history of each issue |

The detail plans name the tracks of the previous plan. Track A is Phases 2 and 7 to 9, Track B is
Phases 2b to 6, Track C is spread over Phases 2 and 14, Track D is Phases 11 to 15, Track E is Phases
1 and 16, Track F is Phases 10, 10b, and 17, and Track S is the simulator plan.

## Principles

- **A trustworthy test comes before the change it judges.** Phases 1 and 2 come before the navigator
  changes that move answers.
- **Confidently wrong answers come before the agreement study.** A technique that reports a wrong
  answer with high confidence moves the disagreement the study measures.
- **The navigator is tuned on real frames only.** The simulator characterizes it.
- **The frozen stack waits for the benchmark.** The seven pull requests frozen on 2026-11-06 are
  measured on it before any of them merges.
- **No new library failures.** A change that moves answers is compared with `main` on the 75 library
  frames on the same machine. Six fail on `main`, each under an issue in Phase 1, 3, or 5.

## Issue count

450 issues were open on 2026-11-22. 348 of them have a row here, and the other 102 are
simulator work tracked separately. 5 finished issues are kept as rows.

## Phase 0: read every frame at the right time, pointing, and brightness

The host session owns this phase.

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #1088 | 2 Essential | yes |  | Install the rms-oops type 1 inverse in CI and fresh environments |
| #1085 | 2 Essential | yes |  | Voyager corrected C-kernels not seen through the host package |
| #949 | 2 Essential | yes |  | Cassini frames timed one exposure late (waits on the rms-host-cassini rebuild) |
| #1087 | 2 Essential | yes |  | Restore released version floors for rms-oops and the hosts (last, after the releases) |
| #1039 | 3 Important | yes |  | Voyager 1 SODIUM images 40 times too dim in I/F |
| #1089 | 3 Important |  |  | Galileo dataset drops GO_0022 GARBLED and REPAIRED images |
| #1090 | 4 Useful |  |  | Check the host packages' published values in CI |
| #628 | 4 Useful |  |  | Choose the source of the observation block's clock counts |

**Exit:** every camera is read through a released host package, the Cassini frames of #949 carry the right time, and a corrected kernel moves the pointing of the image it corrects.

## Phase 1: make a test failure mean something

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #240 | 4 Useful |  | **Done (PR #833)** | Library coverage-matrix invariant |
| #820 | 2 Essential | yes |  | Dataset unit tests download from the live archive |
| #832 | 3 Important | yes |  | CI jobs hang for six hours or segfault in Qt tests |
| #744 | 3 Important | yes |  | Two Galileo truths record the rotation-fitted answer (operator decision) |
| #867 | 3 Important | yes |  | Re-verify three truths the image stars contradict (operator check) |
| #338 | 3 Important | yes |  | Highly-irregular exclusion discards a good terminator fit (operator decision) |
| #547 | 3 Important |  |  | Written-product comparison runs against no frame |
| #516 | 4 Useful |  |  | A root-ownership test that cannot fail |
| #814 | 5 Minor |  |  | Ring performance test runs close to its limit |
| #1043 | 4 Useful |  |  | run-all-checks runs pytest at -n auto |
| #391 | 4 Useful |  |  | Pin ruff and the lint tools |
| #324 | 4 Useful |  |  | Run the agreement-estimator tests in CI |

**Exit:** the unit suite needs no network, no CI job hangs or crashes, and each of the 6 red library frames is fixed, re-pinned, or pinned red under an issue that owns it.

## Phase 2: build the benchmark that judges navigator changes

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #727 | 3 Important | yes |  | Land the ring-truth curation campaign |
| #637 | 3 Important | yes |  | Add the B ring spokes bundle as a second answer |
| #1033 | 3 Important | yes |  | Cross-mission navigation benchmark |
| #340 | 4 Useful |  |  | Cross-check records only a yes/no primary technique |
| #612 | 4 Useful |  |  | library_entry resolves the holdings root in its own order |

**Exit:** one command scores any branch against main on a fixed frame set split into tuning and held-out frames, for all four cameras, and the plans name it.

## Phase 2b (parallel): fix the navigator defects that wait on nothing

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #591 | 3 Important | yes |  | Sparse ring pre-check can skip a frame with rings |
| #787 | 3 Important | yes |  | Star model emits nothing when its noise estimate is zero |
| #400 | 2 Essential | yes |  | Ensemble merge and tiers on anisotropic covariances |
| #994 | 4 Useful |  |  | Catalog lists a star on a magnitude-bin edge twice |
| #815 | 4 Useful |  |  | Classifier reads the peak before filling marker pixels |
| #824 | 4 Useful |  |  | Calibrated blocks default their marker to NaN, products use 0 |
| #749 | 4 Useful |  |  | Multi-body exclusion counts results, not bodies in frame |
| #577 | 4 Useful |  |  | Restore the observation center RA/Dec accessor |
| #579 | 5 Minor |  |  | Logged smear summary reads zero for a rolling frame |

**Exit:** the issues marked exit are closed, and each change is measured on the library with no new failure.

## Phase 3: make the disc and ring correlations right, or flagged when they are not

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #791 | 2 Essential | yes | PR #1053 open | Disc refinement stops at its half-pixel window |
| #483 | 3 Important | yes |  | Re-pin the frames moved by the shift-equivariance fix (operator decision; after #791) |
| #1044 | 3 Important | yes |  | Auto mode prefers a pass that found no peak |
| #1048 | 3 Important | yes |  | Gradient correlation reads the sensor margin |
| #1061 | 2 Essential | yes |  | Flag answers that stop at the refinement window edge |
| #1055 | 3 Important | yes |  | Disc locks 145 px off on an Iapetus frame |
| #1065 | 3 Important | yes |  | Noise lock reported as a success on a dark frame |
| #1050 | 3 Important | yes |  | Unmodeled bright structure pulls the disc correlation (also in the simulator plan) |
| #482 | 3 Important |  |  | Disc misses by up to 1 px on a weak axis |
| #997 | 3 Important |  |  | A constant brightness offset can reject every shift |
| #1002 | 3 Important |  |  | Annulus whole-pixel answers left unmarked |
| #585 | 4 Useful |  |  | Quality metric's background is 20x the search window |
| #941 | 3 Important |  |  | Quality metric has no minimum background size (with #585; also in the simulator plan) |
| #951 | 4 Useful |  |  | Annulus answer flips under a 1e-8 s time change |
| #126 | 2 Essential |  |  | Rotation search cost (mostly done by #1042; also in the simulator plan) |
| #1040 | 3 Important | yes |  | Rotation search misses a planted roll |
| #945 | 3 Important |  |  | Refine candidates on a window at a fast length (after #791; also in the simulator plan) |
| #1049 | 5 Minor |  |  | Signed lag misreads the midpoint of an odd surface |

**Exit:** an answer that stops at its window edge is flagged, the known wrong locks (#1055, #1065) are refused, the rotation search finds a planted roll, and the frames of #483 are re-pinned.

## Phase 4a: navigate on every star the frame shows (frozen stack)

Phases 4a to 4c hold the seven PRs frozen on 2026-11-06 (#766, #773, #774, #779, #782, #784, #789). They wait for Phase 2.

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #622 | 2 Essential | yes | **Done (PR #754)** | Sparse star field accepts a thin solution |
| #623 | 2 Essential | yes | **Done (PR #754)** | One star counted as two witnesses |
| #639 | 2 Essential | yes | **Done (PR #754)** | Star centroid measures a brighter neighbor |
| #621 | 2 Essential | yes |  | Star techniques refuse half the F ring frames |
| #764 | 3 Important | yes | PR #766 open | Cosmic-ray hits crowd out the stars |
| #836 | 3 Important | yes | PR #766 open | Catalog stars emitted twice under one feature ID |
| #768 | 3 Important | yes | PR #774 open | Predicted star smear wrong in length and direction |
| #765 | 3 Important | yes | PR #773 open | Centroids land at random points along a streak |
| #769 | 3 Important | yes | PR #782 open | WAC spoke frames refused with measurable faint stars |
| #785 | 3 Important |  |  | WAC answers near the search margin zeroed |
| #830 | 3 Important |  |  | One offset window per camera for every technique |
| #852 | 3 Important |  |  | One clipped star decides the StarRefineNav fit |
| #880 | 3 Important |  |  | Which stars judge the absent-star test (operator decision) |
| #844 | 3 Important |  |  | Ground the lone-star brightness bound in photometry |
| #835 | 3 Important |  |  | Success at 4.9 px error under 6 px catalog scatter (also in the simulator plan) |
| #862 | 4 Useful |  |  | Coincidental two-star pair answered 35.7 px wrong |
| #780 | 4 Useful |  |  | Faint ring-background star refused with a near rival |
| #771 | 4 Useful |  |  | Pass 2 credits stars a streak box reaches from a wrong prior |
| #781 | 4 Useful |  |  | Record the star field's set-aside source count |
| #772 | 4 Useful |  |  | Split _star_helpers.py |

**Exit:** each star change is measured on the benchmark, the frozen star PRs are merged or closed, and the F ring movie refusals of #621 are solved or shown unsolvable.

## Phase 4b: navigate resolved ring edges without confident wrong locks (frozen stack)

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #476 | 3 Important | yes | **Done (PR #756)** | Ring edges return the shift they are given |
| #755 | 2 Essential | yes | PR #766 open | Along-edge error reported at sub-pixel sigma |
| #373 | 3 Important | yes | PR #779 open | Coarse seed robust to competing edges |
| #346 | 2 Essential | yes |  | Wrong-ring-feature locks (closes when the family does) |
| #822 | 2 Essential | yes |  | NAC spoke frames navigate on no sharp edge |
| #778 | 3 Important | yes |  | Cassini Division edges aliased onto B ring texture |
| #566 | 2 Essential | yes |  | Recalibrate the annulus acceptance for 10-1000 km/px |
| #316 | 2 Essential | yes |  | Ring orbit uncertainty priced as fully correlated (operator decision) |
| #1092 | 3 Important |  |  | Ring-edge vertices at border-pixel centers (also in the simulator plan) |
| #761 | 3 Important |  |  | A ring outer edge 34 km off the catalog |
| #770 | 3 Important |  |  | 2009 NAC spoke movies 2.5-3.6 px from the bundle |
| #760 | 4 Useful |  |  | WAC spoke movies barely better than SPICE |
| #567 | 3 Important |  |  | Correlation picks the basin, edge polishes |
| #831 | 4 Useful |  |  | Bound the ring-edge basin tolerance |

**Exit:** no held-out benchmark frame reports a confident wrong ring answer, and resolved catalog edges are navigated at coarse scales too.

## Phase 4c: fuse only answers that agree (frozen stack)

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #762 | 3 Important | yes | PR #766 open | Conflicted pass 1 keeps StarRefineNav from testing |
| #767 | 3 Important | yes | PR #784 open | Results fused only through the pixel floor |
| #783 | 3 Important | yes |  | Distant pairs still fused inside the pixel floor |
| #786 | 3 Important | yes |  | One- or two-star geometry-seeded refine counts as a witness |
| #841 | 3 Important | yes |  | Wide-sigma one-star answer merges disagreeing witnesses |
| #864 | 3 Important | yes |  | Collapsed star set represented by a looser fit |
| #865 | 4 Useful |  |  | Three F ring frames refused on ring-star conflicts |
| #394 | 4 Useful |  |  | Shape-lock veto trusts a star fix that may be wrong |
| #869 | 5 Minor |  |  | Leave-out log line does not say which rule acted |
| #866 | 3 Important | yes |  | Re-pin the sidecars the stack moves (last) |

**Exit:** no benchmark frame fuses statistically distant answers or counts one witness twice, and the library pins match the merged code.

## Phase 5: remove the limb bias and the confident wrong body answers

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #1110 | 2 Essential | yes |  | Lambert template biases body techniques (also in the simulator plan) |
| #1099 | 3 Important | yes |  | Overlapping body templates leave a dark outline (also in the simulator plan) |
| #790 | 2 Essential | yes |  | Shape-lock veto misses confident wrong answers (also in the simulator plan; after #1062 there) |
| #321 | 2 Essential | yes | PR #789 open | Partial-arc limb fits biased inward by 2 px |
| #747 | 3 Important | yes |  | Polarity check keys on an unreliable convergence flag |
| #150 | 3 Important | yes |  | Limb fit's offset from the image edge |
| #447 | 3 Important | yes |  | Body techniques do not return the shift they are given (held for #128) |
| #128 | 3 Important | yes |  | Robust limb navigation redesign |
| #223 | 2 Essential | yes |  | Terminator fit is degenerate and unvalidated |
| #1103 | 4 Useful | yes |  | Confident limb answer a pixel wrong under blur (also in the simulator plan) |
| #859 | 3 Important |  |  | Vertices beside missing-data edges still fitted |
| #860 | 3 Important |  |  | Saturn limbs 4-9 px off on rev 36-37 frames |
| #350 | 4 Useful |  |  | Resolved-body library frames 2 px off |
| #288 | 2 Essential |  |  | Library regression: 6 of 75 frames disagree (closes when the others do) |
| #858 | 3 Important |  |  | Visible-limb altitude by body and wavelength |
| #878 | 3 Important |  |  | Calibrate that altitude from a full Cassini run |
| #857 | 4 Useful |  |  | Saturn's visible-limb table misses filter, latitude, season |
| #855 | 4 Useful |  |  | Summary overlay draws the silhouette, not the placed limb |
| #861 | 4 Useful |  |  | Tethys limb 2-3 km inside its ellipsoid |
| #975 | 4 Useful |  |  | BodyBlobNav up to 36 px off, not flagged (also in the simulator plan) |
| #25 | 3 Important |  |  | Blur the model for high-resolution bodies |
| #748 | 4 Useful |  |  | Separate overflow limit for a witness blob (operator decision) |
| #282 | 5 Minor |  |  | Limb sub-pixel ripple |

**Exit:** limb answers agree with strong star solutions without a systematic inward offset on the benchmark, and no body technique reports a confident wrong answer there.

## Phase 6 (parallel): finish Titan haze navigation

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #407 | 2 Essential | yes |  | Ratify the Titan decisions and curation artifacts (operator decision) |
| #403 | 3 Important | yes |  | Size the arc ray reach by where the limb can be |
| #404 | 3 Important | yes |  | Size-relative arc-residual limit |
| #399 | 3 Important | yes |  | Voyager Titan validation cohort |
| #405 | 3 Important | yes |  | Titan library growth |
| #401 | 4 Useful |  |  | Extreme phase above 150 degrees |
| #402 | 4 Useful |  |  | Ring-occlusion translucency |
| #594 | 4 Useful |  |  | Compute the sunward direction directly |
| #397 | 3 Important |  |  | Self-calibrated haze-radius table |
| #398 | 4 Useful |  |  | Methane surface-window refinement |

**Exit:** #407 is ratified, large well-framed Titans keep their rays, and the Voyager cohort and the library both cover Titan.

## Phase 7 (parallel): grow the image library and run real images in CI

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #172 | 2 Essential | yes |  | First-stage curated library |
| #174 | 2 Essential | yes |  | Autonomous-nav integration tests and baselines |
| #229 | 3 Important | yes |  | Real-image CI tier per PR, full suite on a schedule |
| #235 | 2 Essential | yes |  | Library: 20 per camera, 120 in all |
| #871 | 4 Useful |  |  | No sidecar expects RingAnnulusNav as primary |
| #319 | 3 Important |  |  | Opposed-ansae ring geometry coverage |
| #239 | 3 Important |  |  | Curate sub-5 px bodies as expected failures |
| #494 | 4 Useful |  |  | Cassini BOTSIM pairs defined two ways |

**Exit:** at least 20 library images per camera and 120 in all, and a real-image tier runs on every PR.

## Phase 8: measure accuracy from independent methods that agree

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #558 | 2 Essential | yes |  | Keep confidently wrong locks out of the anchor (needs Phase 4b) |
| #322 | 3 Important | yes |  | Cross-body limb errors correlate |
| #361 | 3 Important | yes |  | Disc resolution floors the limb-disc coupling |
| #358 | 3 Important | yes |  | Selection effect of the reliability checks |
| #360 | 4 Useful | yes |  | Decide on a survivorship correction |
| #225 | 2 Essential | yes |  | Cross-technique agreement study (also in the simulator plan) |
| #226 | 3 Important | yes |  | Reprojection-consistency check |
| #759 | 3 Important |  |  | WAC difference from the F ring bundle |
| #561 | 3 Important |  |  | FOV twist cohorts |

**Exit:** pairwise agreement statistics over real frames are published, with the confidently wrong families closed or their frames left out.

## Phase 9: anchor confidence and uncertainty in real frames

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #334 | 2 Essential | yes |  | Arm the calibration's falsification criteria (also in the simulator plan; start now) |
| #176 | 2 Essential | yes |  | Move every tuning constant into configuration |
| #557 | 3 Important | yes |  | Combined confidence saturates at 0.99 (operator decision) |
| #751 | 3 Important | yes |  | Sigma ceiling on the low tier (operator decision) |
| #788 | 3 Important | yes |  | Reported sigma wrong at the core and tail (also in the simulator plan) |
| #230 | 2 Essential | yes |  | Real-anchored recalibration (also in the simulator plan) |
| #233 | 3 Important |  |  | Star SNR from measured photometry |
| #675 | 4 Useful |  |  | Two calibration recommendations from sidecar notes |
| #380 | 4 Useful |  |  | Explicit cross-covariance for correlated witnesses |

**Exit:** the confidence values are refitted on real agreement, the falsification criteria pass, and the provisional marker is gone.

## Phase 10: bring Voyager, Galileo, and New Horizons to Cassini's level

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #130 | 2 Essential | yes |  | Star limiting magnitudes per camera |
| #342 | 3 Important | yes |  | Placeholder star PSF on three cameras (also in the simulator plan) |
| #18 | 4 Useful | yes |  | Galileo star navigation |
| #19 | 4 Useful | yes |  | Voyager star navigation |
| #753 | 3 Important | yes |  | Galileo stars fail isolation |
| #2 | 4 Useful | yes |  | LORRI PSF sigma |
| #868 | 4 Useful |  |  | Predicted smear for the other cameras |
| #181 | 4 Useful |  |  | Image-degradation classes |
| #138 | 4 Useful |  |  | LORRI ENG frames policy (operator decision) |
| #22 | 4 Useful |  |  | Star streaks |
| #27 | 4 Useful |  |  | BOTSIM navigation |
| #495 | 3 Important |  |  | Cassini raw-product dataset names |

**Exit:** each camera navigates its star fields and bodies with a calibrated PSF and magnitude limit, and has its own confidence calibration.

## Phase 10b (parallel): navigate outer-planet rings

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #908 | 4 Useful | yes |  | Ring feature narrower than a pixel (also in the simulator plan) |
| #82 | 4 Useful | yes |  | Uranus ring models (also in the simulator plan) |
| #81 | 4 Useful | yes |  | Jupiter ring models (also in the simulator plan) |
| #83 | 4 Useful | yes |  | Neptune ring models (also in the simulator plan) |

**Exit:** Uranus, Jupiter, and Neptune rings navigate on real Voyager frames.

## Phase 11: make corrected pointing kernels exact

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #434 | 3 Important | yes |  | One rotation convention about the FOV center |
| #440 | 3 Important | yes |  | Interior-epoch attitude fidelity |
| #444 | 5 Minor |  |  | Adaptive record cadence |
| #455 | 4 Useful |  |  | Interior interpolation error per camera |
| #863 | 5 Minor |  |  | Bound and compare a fitted star rotation (after #434) |
| #435 | 4 Useful |  |  | Static twist in an FK/IK correction |
| #436 | 4 Useful |  |  | CK writer with corrected instrument kernels |
| #433 | 4 Useful |  |  | C-matrix through an oops API |
| #513 | 4 Useful |  |  | Bound sd_create_ck's memory |
| #437 | 5 Minor |  |  | Register corrected kernels for oops |
| #459 | 3 Important |  |  | Cassini predicted kernels (operator decision) |
| #468 | 4 Useful |  |  | New Horizons pointing family (operator decision) |
| #446 | 4 Useful |  |  | Exercise the remote kernel-index path |

**Exit:** rotation is fitted about the FOV center everywhere, and a corrected segment reproduces the attitude at interior epochs.

## Phase 12: make backplanes and mosaics correct

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #241 | 3 Important | yes |  | Backplane CLI tests (check what is left) |
| #55 | 4 Useful | yes |  | Final set of backplanes (operator decision) |
| #57 | 4 Useful | yes |  | Contents of the FITS HDUs (operator decision) |
| #251 | 3 Important | yes |  | Ring-won pixels carry no body ID |
| #252 | 4 Useful | yes |  | Occluding ring does not own the body planes |
| #253 | 4 Useful | yes |  | FITS sidecar missing promised content |
| #701 | 4 Useful | yes |  | Silhouette or visible extent depends on a truth field |
| #634 | 3 Important | yes |  | Sub-solar longitude recorded west on an east grid |
| #638 | 3 Important | yes |  | Remote products are never fetched |
| #28 | 3 Important | yes |  | Backplane generator |
| #733 | 3 Important |  |  | Missing results root handling unreachable |
| #616 | 3 Important |  |  | Brightness steps between mosaic images |
| #617 | 3 Important |  |  | Minnaert exponent fixed at 0.5 |
| #520 | 3 Important |  |  | Move pointing selection out of the CLI package |
| #418 | 4 Useful |  |  | Mosaic task reports success when all images failed |
| #496 | 4 Useful |  |  | Backplane task lets a per-image failure escape |
| #493 | 4 Useful |  |  | Report a lost index once |
| #738 | 4 Useful |  |  | Resolution threshold below its minimum |
| #611 | 4 Useful |  |  | Viewer shows rad/pixel in radians |
| #143 | 4 Useful |  |  | Backplane viewer read-out after pan |
| #54 | 4 Useful |  |  | Backplane cropping |
| #77 | 4 Useful |  |  | Optional backplane arguments |
| #618 | 5 Minor |  |  | Ring target from configuration |
| #63 | 5 Minor |  |  | Bodies far from planets |
| #718 | 5 Minor |  |  | Ring and body zoom differ |

**Exit:** the backplane set is decided, every plane and map names the right owner, and remote and local products read alike.

## Phase 13: deliver PDS4 bundles for all four cameras

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #705 | 3 Important | yes |  | Spelling check reads the packaged schemas |
| #609 | 3 Important | yes |  | Holdings tree for the synthetic cohort |
| #242 | 3 Important | yes |  | PDS4 CLI tests (check what is left) |
| #717 | 2 Essential | yes |  | Fill cassini:ISS_Specific_Attributes |
| #600 | 3 Important | yes |  | Images that did not navigate (operator decision) |
| #720 | 4 Useful | yes |  | Images with no backplanes (operator decision) |
| #677 | 4 Useful | yes |  | Kernels the metakernel lists (operator decision) |
| #53 | 2 Essential | yes |  | Bundle generator for every camera |
| #67 | 3 Important | yes |  | Cloud-aware bundle generation |
| #79 | 3 Important | yes |  | Context products for targets |
| #30 | 4 Useful | yes |  | PDS4 labels for backplane files |
| #595 | 3 Important | yes |  | LaTeX template for the user guides |
| #596 | 3 Important | yes |  | User guide: Cassini |
| #597 | 4 Useful | yes |  | User guide: Voyager |
| #598 | 4 Useful | yes |  | User guide: Galileo |
| #599 | 4 Useful | yes |  | User guide: New Horizons |
| #614 | 4 Useful |  |  | Dataset without PDS4 support ends in a traceback |
| #716 | 4 Useful |  |  | Check a bundle at a remote root |
| #687 | 4 Useful |  |  | Cite the calibrated PDS4 product |
| #424 | 3 Important |  |  | Remove sd_create_bundle_cloud_tasks |
| #107 | 5 Minor |  |  | Backplane reader repository |

**Exit:** sd_create_bundle builds and checks a bundle for each camera, with the user guides written. The real-volume run (#708) is deferred.

## Phase 14: run whole missions in the cloud

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #108 | 2 Essential | yes |  | Audit CLI logging and cloud operation |
| #236 | 3 Important | yes |  | Per-frame cost and batch throughput |
| #466 | 3 Important | yes |  | Getting the index to cloud workers (operator decision) |
| #538 | 3 Important | yes |  | Mistyped results root reads as nothing navigated |
| #587 | 3 Important | yes |  | Stale index passes the version check |
| #531 | 3 Important | yes |  | Documents record the cache path, not the source |
| #528 | 3 Important | yes |  | Format version in metadata documents |
| #541 | 3 Important | yes |  | One rule for which cache a file lands in |
| #574 | 3 Important | yes |  | Check cloud_support/ |
| #736 | 3 Important |  |  | Name an index in three programs |
| #515 | 4 Useful |  |  | Ingest can write another root's rows |
| #578 | 4 Useful |  |  | Symlinked root gets a different stub |
| #501 | 4 Useful |  |  | Bind queries to the resolved schema |
| #536 | 4 Useful |  |  | Seam overstates what a missing row means |
| #533 | 4 Useful |  |  | Unretrieved files counted by one backend only |
| #534 | 5 Minor |  |  | SQLite drops signed zero |
| #540 | 4 Useful |  |  | Narrow selection lists a whole volume |
| #462 | 4 Useful |  |  | Consumer open takes a write lock (operator decision) |
| #464 | 4 Useful |  |  | Document column for bundle generation |
| #465 | 4 Useful |  |  | Schema for the curation tool |
| #486 | 4 Useful |  |  | sd_offset writes its own rows |
| #542 | 4 Useful |  |  | Record whether a root was pruned |
| #572 | 4 Useful |  |  | URL with no scheme means a local file |
| #512 | 4 Useful |  |  | Document-to-column placement written twice |
| #514 | 4 Useful |  |  | Ingest has its own parse loop |
| #141 | 4 Useful |  |  | Deduplicate the CLI preamble and task loop |
| #103 | 4 Useful |  |  | Thread-unsafe module caches |
| #584 | 4 Useful |  |  | Bound the model stage's backplane cache |
| #142 | 5 Minor |  |  | Task export disagrees on ImageFiles count |
| #467 | 5 Minor |  |  | --since selector |
| #472 | 5 Minor |  |  | Ingest roots as FCPath |
| #497 | 5 Minor |  |  | Integration case for a refused document |
| #725 | 5 Minor |  |  | Remove the open-file clamp |
| #535 | 5 Minor |  |  | Report retains more than it prints |

**Exit:** every program runs as cloud tasks and logs properly, per-frame cost and throughput are published, and the results index reaches cloud workers.

## Phase 15: validate configuration and give every program one interface

#427, #1032, and #1001 are deferred by the operator (2026-11-17). #1029 is in the simulator plan.

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #427 | 4 Useful | yes |  | Reorganize the config namespace (deferred by the operator) |
| #1032 | 3 Important | yes |  | Type every navigator config section (deferred by the operator) |
| #1001 | 4 Useful | yes |  | One rule for which keys the loader checks (closes with #1032) |
| #118 | 3 Important | yes |  | Config validation system |
| #735 | 4 Useful | yes |  | sd_create_ck and sd_stats_report image selection |
| #730 | 3 Important |  |  | Rounding in the metadata document (operator decision) |
| #706 | 4 Useful |  |  | Wheel built outside git ships no config |
| #604 | 4 Useful |  |  | Second load keeps the first configuration |
| #711 | 4 Useful |  |  | Hash only the instrument configuration used |
| #731 | 4 Useful |  |  | Log names in UTC, records in local time |
| #734 | 4 Useful |  |  | --profile produces no profile |
| #411 | 4 Useful |  |  | snake_case technique keys |
| #552 | 4 Useful |  |  | Remove the AttrDict workaround |
| #100 | 4 Useful |  |  | One root-path getter |
| #101 | 4 Useful |  |  | ArgumentParser.error for CLI errors |
| #102 | 4 Useful |  |  | No module globals in CLI drivers |
| #38 | 5 Minor |  |  | Filecache location options |
| #471 | 5 Minor |  |  | Internal codename in config placeholders |

**Exit:** a bad configuration key is refused at startup in every section, and every program selects images the same way.

## Phase 16: publish the evidence, the docs, and the coverage floor

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #232 | 3 Important | yes |  | End-product geometric accuracy (after Phases 12 and 13) |
| #231 | 3 Important | yes |  | Capability matrix (last) |
| #548 | 4 Useful | yes |  | Coverage floor (operator decision) |
| #243 | 4 Useful | yes |  | Star-conflict tests |
| #177 | 4 Useful | yes |  | Summary-PNG tests |
| #438 | 4 Useful |  |  | Run the nitpicky Sphinx check |
| #129 | 4 Useful |  |  | Zero nitpicky warnings |
| #122 | 4 Useful |  |  | Verify the terminator guide's rationale |
| #545 | 4 Useful |  |  | README links to the guides |
| #549 | 4 Useful |  |  | Guides name a module that is a package |
| #562 | 4 Useful |  |  | Class diagrams in dev-guide chapters (operator decision) |
| #723 | 4 Useful |  |  | Catch up with the rms-devenv template |
| #928 | 4 Useful |  |  | Declare and test Python 3.15 |
| #576 | 4 Useful |  |  | mypy warn_unreachable (operator decision) |
| #530 | 4 Useful |  |  | Fixture clock seconds |
| #524 | 4 Useful |  |  | Share the record-source fixtures |
| #525 | 5 Minor |  |  | Test modules over 1000 lines |
| #473 | 5 Minor |  |  | Windows claim, POSIX constructs |
| #554 | 5 Minor |  |  | FCPath wraps Path in str() |
| #870 | 5 Minor |  |  | Cross-volume early-stop test |
| #953 | 5 Minor |  |  | Cleanup fails on a directory symlink |
| #714 | 5 Minor |  |  | Em dashes and arrows in source |
| #715 | 5 Minor |  |  | Qt method argument docs |
| #807 | 5 Minor |  |  | Voyager marker_value docstring |
| #939 | 5 Minor |  |  | LORRI filespec docstring |
| #443 | 5 Minor |  |  | API reference scope (operator decision) |
| #470 | 5 Minor |  |  | Codename in config comments |
| #379 | 5 Minor |  |  | Star-saturation wording |
| #1035 | 5 Minor |  |  | Old plan-phase labels |

**Exit:** the capability matrix and end-product accuracy are published and test-verified, and a coverage floor is enforced.

## Phase 17 (parallel): add viewer features and clean up the code

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #186 | 3 Important |  |  | Manual-nav dialog with a result panel |
| #185 | 3 Important |  |  | Style gated-out features on the PNG |
| #15 | 4 Useful |  |  | Overlay does not hide the background model |
| #119 | 4 Useful |  |  | Move PNG creation out of navigate_image_files |
| #182 | 5 Minor |  |  | Stop-after-features option |
| #183 | 5 Minor |  |  | Polarity-aware ring matching |
| #65 | 4 Useful |  |  | Exception hierarchy |
| #98 | 4 Useful |  |  | One instrument registry |
| #135 | 4 Useful |  |  | One from_file margin block |
| #97 | 4 Useful |  |  | Split modules over 1000 lines |
| #96 | 4 Useful |  |  | Prune dead code |
| #92 | 4 Useful |  |  | Optional dependency groups |
| #109 | 4 Useful |  |  | Safe-path helpers |
| #110 | 4 Useful |  |  | Scalar validation helpers |
| #423 | 4 Useful |  |  | Viewers print library logs to stdout |
| #429 | 4 Useful |  |  | Shared logging for util/ |
| #518 | 4 Useful |  |  | State file encodings |
| #39 | 4 Useful |  |  | AttrDict missing attributes |
| #21 | 4 Useful |  |  | Inventory in public metadata |
| #99 | 4 Useful |  |  | Orphan report_profile.py |
| #105 | 5 Minor |  |  | Typed interop boundaries |
| #147 | 5 Minor |  |  | Deduplicate confidence contexts |
| #140 | 5 Minor |  |  | Geometry union access |
| #137 | 5 Minor |  |  | Dead selection validation |
| #144 | 5 Minor |  |  | QApplication never released |
| #155 | 5 Minor |  |  | Duplicate display scaling |
| #388 | 5 Minor |  |  | Body occluder pre-filter |
| #590 | 5 Minor |  |  | Function-level imports |
| #709 | 5 Minor |  |  | draw_line truncates coordinates |
| #1082 | 5 Minor |  |  | Read index tables a few rows at a time |
| #428 | 5 Minor |  |  | Logger eviction API from rms-pdslogger |

**Exit:** none; each item lands when its fix is ready.

## Deferred

| Issue | Priority | Exit | Status | Subject |
|---|---|---|---|---|
| #708 | Defer |  |  | PDS4 bundle over a real COISS volume |
| #710 | Defer |  |  | Global navigation index in the bundle |
| #521 | Defer |  |  | Conflicted result drops its fitted rotation |
| #134 | Defer |  |  | RingMosaic changes oops precision globally |
| #712 | Defer |  |  | Renumber the configuration files |
| #34 | Defer |  |  | PDS4 Cassini input |
| #448 | Defer |  |  | Kernel inputs through spyceman |
| #23 | Defer |  |  | Body shape models |
| #187 | Defer |  |  | Hyperion pose |
| #184 | Defer |  |  | Cartographic navigation |
| #33 | Defer |  |  | LORRI instrument kernel |

**Exit:** none; an item moves into a phase when its Defer priority is lifted.

## Dependencies at a glance

- Phase 0 (released hosts) comes before the Cassini re-navigation that #949 needs.
- #791 comes before #1061, #945, and the two disc pins of #483.
- Phase 2 (#1033) comes before Phases 4a, 4b, and 4c.
- The simulator's #1062 comes before #790. #128 carries #447.
- Phases 3 to 5 come before Phase 8, and #558 waits for Phase 4b.
- Phase 8 (#225) comes before Phase 9 (#230), and #334 comes before #230 and can start now.
- #434 comes before #435, #561, #863, and any decision to fit rotation again.
- The decisions on #55 and #57 come before the rest of Phase 12, and #241 and #242 before Phases 12
  and 13.
- Phases 12 and 13 come before #232, and #232 before #231.
- #427 comes before #1032, and #1032 before #1001.

## Operator decisions

| Issue | Decision | Phase |
|---|---|---|
| #744 | Truth of the two Galileo library frames | 1 |
| #867 | Three library truths the image stars contradict | 1 |
| #338 | Terminator fits on highly irregular bodies | 1 |
| #483 | The re-pins, after #791 | 3 |
| #880 | Which stars judge a lone star's absent-star test | 4a |
| #316 | Ring orbit uncertainty severity | 4b |
| #748 | Overflow limit for a witness blob | 5 |
| #407 | Titan haze ratification | 6 |
| #557, #751 | Where confidence saturates, and a sigma ceiling on the low tier | 9 |
| #138 | LORRI ENG frames | 10 |
| #459, #468 | Cassini predicted kernels, and the New Horizons pointing family | 11 |
| #55, #57 | Backplane set and HDU contents | 12 |
| #600, #677, #720 | What a bundle says about images it cannot describe, and which kernels it lists | 13 |
| #462, #466 | SQLite write lock, and getting the index to cloud workers | 14 |
| #730 | Rounding in the metadata document | 15 |
| #443, #548, #562, #576 | API reference scope, coverage floor, class diagrams, `warn_unreachable` | 16 |

Deferred by the operator on 2026-11-17: the navigator config (#427, #1029, #1032). #822 was decided
on 2026-10-28 (navigate resolved catalog edges at coarse scales) and waits with the frozen stack.
