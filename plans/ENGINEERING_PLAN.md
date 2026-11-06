# SpinDoctor Engineering Plan (Tracks B-F)

*The implementation-detail companion to `plans/PROGRAM_PLAN.md` for
everything outside the validation and calibration science (Track A, which
has its own detail plan in `plans/VALIDATION_AND_CALIBRATION_PLAN.md`).
Each item below carries enough context — current behavior, files, fix
direction, constraints, acceptance — to be handed to a developer or an
implementing model with no other briefing beyond `/seti/newnav/CLAUDE.md`.
Issue numbers are the tracking source of truth; where this plan and an
issue disagree, update whichever is stale.*

Conventions that apply to every item here: project rules in
`CLAUDE.md` (line length, mypy strict, pdslogger-only logging, pytest
style, Conventional Commits); after any change that can move navigation
output, run the library cross-check
(`util/calibration/library_crosscheck.py`) and the sim integration suites
and account for every delta; never edit a library sidecar's `expected.*`
fields to match current behavior.

---

## Track B — Navigation correctness

Ordering within the track: the confidently-wrong defects (#346, #755, #350), because the agreement study consumes
ensemble output at scale and several curated library frames pin these as
standing red regressions. Then the coarse-lock remainder (#373), then the
investigation/design items (#25, #128/#150), with the smaller decision items
(#130, #239, #338) as fill.

The rotation work sits outside that ordering and has its own sequence: one
convention about the image center first (#434), then wider distortion cohorts
(#561), and only then any decision to fit rotation again. A conflicted result
dropping its fitted rotation (#521) waits on the same convention and cannot
fire until fitting is enabled somewhere.

The ensemble-independence family (#222 seeded single-star refine, #317 two
ring techniques on one catalog, #339 scattered-light disc/limb) is closed by
`src/spindoctor/nav_orchestrator/ensemble_independence.py`
(`resolve_independent_estimators`), which resolves the winning consensus group
into mutually independent estimators before the merge: a seeded single-star
refine is dropped when a stronger non-single-star witness is present; two ring
techniques and (on a scattered-light frame) disc and limb are each collapsed to
a single representative. The scattered-light signal is a runtime
background-gradient score
(`src/spindoctor/support/background_gradient.py`, surfaced on
`NavImageClassifierResult`) ported from the cohort-curation prescan. See the
"resolve independent estimators" step in the ensemble dev guide. The collapse
uses the conservative rho=1 (fully-correlated) model; #380 tracks the
follow-up to fit an explicit per-family cross-covariance from the agreement
study and recover the partially-correlated precision, gated on real-frame rho
measurements (#225).

The confident-wrong body-nav family (#328 high-phase haze crescent, #291
extreme-shape-mismatch disc/limb consensus) is closed by the cross-technique
body-witness veto: it declines the high-phase haze-crescent gate-passing ~30 px
success and reports the extreme-shape-mismatch disc/limb consensus conflicted
rather than a confident-wrong 10-17 px lock. The body-body occlusion pair (#326
BODY_DISC template, #327 `visible_arc_fraction` / `visible_lit_fraction` report)
is closed by making `NavModelBody` body-body-occlusion-aware: the correlation
template is trimmed against nearer-body occlusion and the visible-arc and
visible-lit fractions no longer over-credit an occluded limb. The star-matcher
robustness items are closed as well: triplet canonicalization is geometric
(opposite-side-length), deterministic and rotation-stable on equal-brightness
fields (#337); the widened saturation match prefers the brightest qualifying
reference rather than the nearest (#376); and a single-bright-star wide-offset
lock exists, gated behind two uniqueness margins and default-off pending an
image-library false-lock sweep (#367). The low-phase limb gate (#281) is closed:
a `BodyLimbNav` coarse-seed mis-lock below 15 deg phase is flagged spurious by an
unconverged-at-trust-boundary gate. The ring-annulus occlusion trim (#378) is
closed: the RING_ANNULUS correlation template is trimmed for planet occlusion,
the annulus-side analog of the ring-edge trim. The post-recalibration library
reds are closed in turn: a lone brightness-centroid blob that a corroborated
star consensus contradicts is now dropped from the ensemble math, so a correct
multi-star fix commits instead of conflicting on the blob outlier (#351); and
the star-navigation hardening (two-star unique-match assignment plus multi-star
refine) locks the small-offset WAC cruise field that previously self-flagged
all-techniques-spurious (#352). The Iapetus shape-lock veto misfire (#392) is
closed the same way on the geometric side: the shape-lock verdict is suppressed
when a trusted (non-spurious, non-single-star) star fix agrees with the
geometric consensus offset the blob disputes, so an albedo-dichotomy-biased
centroid no longer vetoes a star-confirmed limb fit. The residual that a wrong
trusted star fix could itself corroborate a wrong geometry -- downgrading a safe
`conflicted` to a confident-wrong `success` in that corner -- is tracked in #394.

### #521 — the fitted rotation a conflicted result drops (deferred)

`NavResult.conflicted` declares no `rotation_rad`, so a conflicted result on
a rotation-fitting instrument drops the twist its own ensemble measured, and
everything downstream reads it as 2-DoF: `rotation_fitted` computes `False`,
a corrected `cmatrix` is built from the translation alone, and the image is
eligible for a segment. The outcomes invert with respect to trust — a
*success* result is refused a segment as `rotation_unsupported` while a
*conflicted* one is written into the kernel with an attitude wrong by exactly
the twist thrown away. The document is internally inconsistent too: a 3x3
`covariance_px2` with no `rotation_deg`, where the metadata chapter documents
matrix shape as the signal for degrees of freedom.

**Unreachable, and deferred until it is not.** No instrument fits rotation,
so no result carries a rotation to drop. What re-opens it is any instrument
setting `fit_camera_rotation: true`, which
`test_no_shipped_instrument_fits_camera_rotation` fails on, naming this
precondition. It is not worth doing alone: fitting every technique about the
FOV center (#434) removes the per-technique pivot entirely and needs no pivot
field, so the conflicted path's rotation parameters should be designed with
that rather than bolted onto the present scheme. When it is picked up, 35
test sites already build a `NavContext` with `fit_camera_rotation=True`, so a
regression test needs no configuration change, and the missing case is the
conflicted counterpart of
`test_a_fitted_rotation_of_zero_is_still_a_fitted_rotation`.

### #434 — one rotation convention, about the image center

Rotation is fitted per technique, and each technique picks its own center: a
vertex centroid for the distance-transform family, a composite template
centroid for the disc, and for the star family the *image origin* -- pixel
(0, 0). `_star_helpers` records a `pivot_vu` of the weighted catalog centroid
but returns `translation = det_c - R * cat_c`, which is a rotation about the
origin, and `nav_technique_star_field` takes that value as the offset. The
recorded pivot is not the pivot the reported translation uses.

Because theta is pivot-invariant, this never shows in the angle. It shows in
the translation, which is what the ensemble fuses and what the metadata
reports. Measured on two Galileo star fields, turning fitting off moved the
reported offset by 5.90 and 5.65 px -- `(I - R) * cat_c` for a centroid some
780 px from the origin on an 800x800 frame. So **`offset` means a different
quantity depending on whether a rotation was fitted**, by several pixels, and
any expectation recorded under one convention cannot judge a result measured
under the other.

The design is in the issue: every technique reports its rotation about one
common center, the center of the field of view, converting at the technique
boundary. No pivot field is needed and no fit has to change. What has to be
decided is which point that center is -- the FOV center and the boresight
differ, and the distortion models make the difference real -- and then the
conversion has to be applied and tested per solver.

**This is upstream of the rest of the rotation work.** While the field-of-view
distortion study pivots at the optical center and navigation pivots at the
image origin, the two cannot be compared, so neither a wider distortion cohort
(#561) nor any decision to enable fitting again rests on solid ground. Do the
convention first.

### #561 — the distortion cohorts sample too little

Per-instrument twist verdicts come from cohorts that vary from 225 images over
10 sequences down to 18 images over one, and one instrument's is a single
frame. Running the same tool on a second Galileo sequence returns the opposite
sign, three times the magnitude, and the opposite recommendation. The verdicts
that recommend a static kernel correction versus per-frame fitting therefore
rest on samples too narrow to carry them. Widen the cohorts, and state each
verdict's sampling in the report so a mission-wide result is distinguishable
from a single-sequence one. Gated on #434 for the reason above.

### #447 — the round-trip residual (fixed on a held branch; lands with #128)

A technique re-measuring a frame whose pointing was corrected by its own
previous answer should measure zero, and the body techniques do not:
`BodyLimbNav` fell 0.14 px short on `W1637520502_1_CALIB` and
`BodyDiscCorrelateNav` 0.50 px short on `C3446143_GEOMED`. The fix measures
the non-equivariance across the whole library with a sweep harness
(`util/calibration/shift_equivariance.py`, baseline committed), then removes
both body-side causes: a sub-pixel silhouette probe
(`src/spindoctor/nav_model/silhouette_probe.py`) that stops the discrete
polyline extraction from re-rasterizing a ridge instead of translating it,
and a per-axis NCC-quadratic fallback in `evaluate_candidate` for the case
where the upsampled-DFT refinement argmax lands on the window boundary and
the old code reported a pinned +-0.5 px.

The fix is ported onto `main` as the branch `rf_shift_equivariance_v2` (head
519250b4) and held there, by operator ruling, as the base for #128; it is not
merged on its own. Over 66 library frames with six planted shifts each it
cuts `BodyLimbNav`'s median round-trip residual from 0.118 / 0.082 px to
0.051 / 0.047 px (v / u), and `BodyDiscCorrelateNav`'s reference offsets stop
landing on exact half pixels (13 of 21 on `main`, none on the branch). Ring,
star and blob techniques are bit-identical between the two trees. The cost is
absolute accuracy at the limb: on the five library frames that have both an
independent star-only answer and a limb result, the limb answer moves inward
along the lit-limb direction by 0.21 to 0.41 px (mean 0.27 px) and ends
farther from the stars on four of them. The probe moves the model's lit-limb
vertices outward by 0.11 to 0.18 px, onto the geometric limb, and the image
edge sits inside that limb (#150); `main`'s ridge-center vertices sat inside
it too and partly canceled the offset. Merged alone, the fix would put that
bias into every limb answer and product, so it lands with #128 (below). The
measurements are on #447, which stays open until then.

What the work filed
rather than fixed: RingEdgeNav's planted-shift round trip (#476, same
family as #346 and #373; measured across the edge normal the technique is at the DT
floor, the ensemble's error there was the rank-1 axis quantization fixed
below, and #476 closed on that across-edge criterion; the along-edge part
is reported honestly but not reduced, under #755); `BodyDiscCorrelateNav` still misses by
up to ~1 px on a weakly-constrained axis (#482); and the library pins the fix
moves need re-ratcheting when it lands (#483, which cannot be done honestly
until #288 is reconciled).

### The ring routing decision, and #346 — the remaining confident-wrong ring-lock

Ring routing for Saturn is decided and shipped: every scene at or above
25 km/px radial resolution feeds the annulus composite
(`feature_emission.ring_annulus.planets.SATURN.kmpp_threshold`), and the
per-edge DT fit receives only sub-25 km/px scenes. The basis is a 131-frame
clean-truth head-to-head (operator-audited, bundle defect lists applied)
that measured the edge fit wrong-when-accepted at 5% / 13% / 56% / 100% in
the 300-1000 / 100-300 / 25-100 / 0-25 km/px bands while the annulus fit was
wrong on zero accepted answers at every band, so the annulus route covers the
regimes that comparison validates, everything at or above 25 km/px. The edge
fit degrades toward fine
resolution, where the ring alias lattice resolves into many distinct similar
concentric edges and the shape-only fit locks the wrong one -- which the
rendered-brightness template disambiguates. Below 25 km/px the trustworthy
evidence is three frames on which neither technique is validated (the edge
path re-locks catastrophically on all three; the annulus near-misses twice
and fails once), so the threshold stops where the evidence stops and the
sub-25 confident-wrong exposure stays open under #346/#755. The threshold is
Saturn-measured only. Two follow-ons are filed: recalibrating the annulus
gates so more of its correct answers are accepted (#566 -- today they veto
many right answers, which costs coverage, not correctness), and a hybrid in
which the correlation fit picks the basin and the edge fit polishes within
it (#567). Measured as a basin selector below 25 km/px, the annulus
correlation was right across the edge on F ring movie frames but 30 to 115 px
wrong on N1492091163, N1467344214, N1867602424 and the Keeler gap frames,
where the edge fit is right, so the hybrid as filed would regress them.

The lattice lock itself is addressed in the coarse seed. The mask-overlap
seed counts every detected edge pixel alike, so where a ring's interior
texture or a scattered-light glow's faint contours run parallel to the
cataloged edge it settles on an alias; the seed is now checked against the
strength of the edges under it, and the strongest alignment replaces it when
the overlap seed sits on edges below 0.8 of that strength. A converged fit
whose matched edges are under 0.005 of the frame's strongest is rejected as
spurious, which catches frames whose cataloged edges lie outside the sensor at
the true pointing. Measured against the published F ring bundle's per-frame
pointing on 1,041 held-out frames below 25 km/px: right answers (within 2 px)
went from 692 to 877, wrong answers from 283 to 129 (alias locks across the
edge from 188 to 4), refusals from 66 to 35, and the median error from 0.84
to 0.69 px. A same-code control, `main` rerun on 150 of those frames,
reproduced the earlier `main` run exactly (status, tier and every fused and
per-technique offset identical on all 150), so these counts carry no
run-to-run noise. No frame `main` answers right is
answered wrong; six frames `main` refuses are answered wrong, all of them
along the edge (2.0 to 19 px). On the 263-frame tuning sample the counts went
from 147 right and 94 wrong to 179 and 68. The axis fix below changes none of
those answers (no held-out frame has an all-straight edge set). On the
library the seed check alone leaves the ring frames' answers where they were
except N1492091163 (below); the axis fix is what moves seven of them.

What remains of the family:

- **The rank-1 axis (fixed).** The rank-1 axis was the dominant direction
  of the rings model's pixel-neighbor normals, which point along multiples of
  45 degrees, off by up to 22.5 degrees
  for a straight edge at any other orientation. It now comes from the line's
  own vertices: the ensemble's planted-shift round trip across the edge fell
  from a median 0.39 px (worst 17 px) to 0.04 px (worst 0.31 px), and the two
  Encke gap frames of #346 navigate to their operator truth. Five
  `ring_only_flat` sidecars recorded the quantized angle as their constraint
  normal and went red; on 2026-10-27 the operator re-pinned them along the
  line's own normal (the values are recorded in the Operator Playbook), and
  re-pinned the two Encke gap frames' tier from `medium` to `low`, the ring
  orbit channel's. All seven are expected to pass; the library has not been
  run since the re-pins.
- **Along-edge position of gently curved edges (#755, open).**
  RingEdgeNav's covariance carries the weakness (several pixels along such
  an edge), so a star fix sets that component, a ring-only answer is
  reported at the low tier, and the ensemble refuses rather than averages a
  ring answer and an equally vague witness far from it (the compromise
  test). The position the coarse seed picks along the edge is unchanged,
  but few frames now depend on it: the Cassini star limit is photometric
  (filter pair, exposure, gain state; NAC 14.75 and WAC 11.66 at 1 s
  through the clear filters, measured on F ring star detections), so on
  the 1,614-frame F ring sample 422 of the 424 NAC ring-only answers gain
  a star fix. Against the bundle pointing, NAC answers within 1 px went
  from 561 to 1,020 and those over 2 px from 320 to 98; WAC from 86 to 210,
  with refusals from 133 to 28 and one answer over 2 px. Supporting
  changes: a star refine drops a star the frame edge cuts, a catalog star
  on a magnitude-bin edge is kept once, a detection must stand above its
  local background, a refine star that disagrees with four or more others
  is dropped, a single-star refine replaces the one-star match it
  confirms across a precise witness, the WAC annulus carries a 0.9 px
  floor, and a body brightness center's bias lies along the Sun and
  includes the ephemeris. 83 answers moved away from the bundle pointing:
  star centroids in the image show the pointing wrong on 62; N1600250613
  is refused where the pointing is right (a RingEdgeNav fit 2.1 sigma off
  a tighter star fix, refused at the baseline too); 20 have no two
  agreeing stars to decide (8 of them have one star that agrees with the
  new answer and none with the pointing, and 4 of their 5 WAC frames lie
  within 0.9 px of the pointing once its measured (0.43, 0.69) px WAC
  offset is removed). So strictly better than the bundle is not shown and
  the issue stays open. Stepping the fit along the radius-gradient
  normals (Jacobian and gradient-ridge pass) was measured and not
  shipped; a future attempt at the along-edge position starts from that
  measurement. The library failed 31 of 75 against the sixteen then red on
  `main`: the fifteen new are tier and primary-technique pins recorded with
  the shallower limit (re-pins proposed in the Operator Playbook). That was
  measured before the 2026-10-27 re-pins of eight of the sixteen. Seven are
  expected to pass but have not been run on this branch; the eighth,
  N1633925572_1, reaches the high tier here against its new `low` pin, so it
  stays red on this branch (Operator Playbook).
  The full F ring bundle run on this branch combined with #754 (20,584
  frames, 2026-10-20) took NAC answers within 1 px of the bundle pointing
  from 6,726 to 14,934 and over 2 px from 4,299 to 679, and WAC within
  1 px from 723 to 1,767; of 556 frames that got worse, stars showed the
  pointing wrong on 383, 162 could not be decided, and 11 were ours. Five
  fixes followed: the one-star brightness margin counts only stars that can
  enter the star's search window, and a match whose star is uniquely bright
  only there must be corroborated by two other stars at its offset; a
  collapsed set of star witnesses keeps its vote when one member is a
  seeded refine; a result fused with a collapsed set is held to the
  compromise test against the set's representative, the side less precise
  along their separation left out; a refine that agrees with its own seed
  only by the pixel floor takes the compromise test; and a star's moment
  centroid sits above a plane fitted to the ring around its box. A rerun of
  the 556 worse frames, 2,000 right controls and the 14 ringcause frames on
  the combination: 1,992 controls stay right; of the 536 frames worse than
  either full run, stars show the pointing wrong on 375 and side with it on
  none, and 161 cannot be decided (55 WAC frames within 1 px once the WAC
  offset is removed, 88 within 0.1 px of the full run's answer). Still
  worse than one of the two full runs: N1604017452 refused where 19 stars confirm the answer (a lone ring
  alias 50 px off conflicts with the two-star pass-1 answer, so the refine
  never runs, #762); N1862698331 refused (its pair search draws on stars no
  window can reach, which only #754's pair search can skip, #763); the
  cosmic-ray crowding of the star-field detector is #764; W1762900979
  refused (a three-star refine 1.5 px from its two-star seed); and
  N1605530694 and three ring-versus-star conflicts refused. So strictly
  better than the bundle is still not shown and the issue stays open.
  Three more fixes followed. A conflicted pass 1 is now tested: pass 2
  runs seeded at the consensus answer and at the runner-up's, and an answer
  whose pass-2 results find at least four stars, twice as many as at the
  other, is confirmed while the other's pass-1 results are refuted (#762);
  of 247 frames the full run refused that way, 228 now answer within 1 px of
  the bundle and the 12 marginal or wrong are bundle wrong by stars. The
  star-field detector sets aside a peak sharper than the camera's narrowest
  star, a cosmic-ray hit, before its 30-source cut (#764). And a precise,
  full-rank pass-1 answer breaks a 2-to-1 split of a three-star refine in its
  own favor. A rerun of the same 2,570 frames on the combination with #754
  (whose pair search now sets aside stars no window can reach, #763): against
  the earlier rerun, 12 frames better and 6 worse, refusals 9 to 3; of the
  529 frames worse than one of the two full runs or the earlier rerun, stars
  show the pointing wrong on 376 and side with it on none, and 153 cannot be
  decided (152 already undecided before). Still refused: `N1605530694`,
  whose 6.5 px smear puts every peak at a random point along its streaks
  (#765), and the ring-versus-star conflicts `N1543177345` and
  `N1743090421`, where the previous run's answer is 5 to 7 px off the stars.
  The library changes only pins: on this branch two frames' primary technique
  becomes the star field and one frame's tier rises (closer to its verified
  offset); non-Cassini frames are unchanged. Strictly better than the bundle
  is still not shown: `N1605530694` was answered within 1 px before.
  A full run of the 20,584 frames on that combination (2026-10-22) found
  three answers ours by stars against the run before it, each a star answer
  pulled just past 1 px, and one wrong at the high tier (`N1584292832`).
  Two fixes followed. A result fused with a collapsed set must agree
  statistically with at least one of its members: a ring edge inside the
  5 px floor but 4.6 to 8.6 standard deviations from every star of the set
  had passed whenever its fusion moved the answer under a pixel. And of two
  star results that share no star and disagree, the one on at most half as
  many stars is left out unless a result on still more stars contradicts the
  larger. A third, proposed for #754, lets a three-star fit within a pixel
  stand for a star set whose sixteen-star refine scatters over more than one.
  Rerun of the 2,570 frames plus 342 the offline replay moved: against the
  full run 12 better and 10 worse, every worse one bundle wrong by 2 to 45
  stars; of the 532 frames worse than the full run or the earlier one, stars
  show the pointing wrong on 382 and side with it on none, and 150 cannot be
  decided (148 already undecided). The four losses above answer within
  0.52 px. The symmetric star check, which also looks for stars at the
  bundle's answer, changes no verdict of the full run's worse frames.
  Still refused: `N1543177345` and `N1743090421` (the previous answers are
  5 px off the stars) and `N1493641551` (no usable star either way). Single
  results inside the floor are still fused however far apart statistically
  (#767). The image library is unchanged on this branch (75 of 75
  identical); on the combination only the
  Galileo frame `C0059899900R` moves, 1 px farther from an operator truth that
  three catalog stars put 5.3 to 7 px off, and closer to those stars (mean
  0.85 px against 1.15 px). The confidence tiers keep their meaning over the
  replayed full run: at the high tier 86.6% of NAC answers lie within 0.5 px
  of the bundle (85.7% before) and 292 over 2 px (299).
  Stars on a smeared frame are then centroided at their streaks' centers
  (#765). The predicted smear is a poor guide to the streak (on frames
  predicted to smear 4.5 to 8 px the streak is a median 0.6 times the
  predicted length and 10 to 20 degrees off), so the star model measures the
  streak the frame's bright stars draw and every star technique centroids on
  it when it is at least 4.5 px long. `N1605530694`, whose peaks split 2.4 px
  apart along its 6 px streaks, answers 0.01 px from the bundle and
  `N1605536089` 0.18 px. A rerun on the round-5 combination of the 560
  smeared frames of the 2,570, a 500-frame control and the final full run's
  119 further streaked frames makes no verdict worse than the earlier rerun or
  the final full run; 3 refusals and 2 marginal answers become right, the
  streaked frames' median distance from the bundle falls from 0.14 to
  0.09 px, and every unstreaked frame and the library are unchanged.
  Strictly better than the bundle is still not shown: 150 frames cannot be
  decided by stars.
  The predicted smear then comes from the as-flown predicted C-kernels
  rather than the reconstructed ones, whose 4 s records repeat an
  attitude on 31% of intervals (#768): over 1,796 measured frames the
  prediction moves from 0.58 times the streak, 20.8 degrees off and 15%
  within 1 px to 0.96 times, 2.9 degrees and 99%. The as-flown kernel is
  read without being furnished, so the a priori C-matrix is bit-identical
  on every frame rerun. The as-flown motion vouches for a 1.5 streak
  ratio, so Cassini frames are measured from 3 px and held to 1.5 times
  the prediction, and where no star's streak can be measured a predicted
  streak of 4.5 px or more stands in (on 202 of the 212 F ring frames
  predicted that long whose stars measure no streak). A rerun of 1,699
  frames (the 1,179 above, a 207-frame low-smear control, 178 frames newly
  predicted over 2 px and the 212): against the previous rerun 1 better
  (`N1723506317`, marginal 1.19 to right 0.19 px) and 1 worse; against the
  final full run 14 better (9 refused and 4 marginal become right, 1
  refused becomes marginal; 5 of them the previous rerun's) and 2 worse.
  `N1860785342` (right 0.61 to marginal 1.40 px) is ours by stars, 5
  streak centroids to none and 2 peaks to none. `N1748760905` (marginal to
  refused) is a loss: stars agree with both answers within 0.12 px, but
  pass-2 arbitration counts 11 stars refined from a prior 2 px off, since
  the streak-grown box still reaches them, against 12 from the consensus,
  so neither is confirmed; the support count should require the refined
  stars to lie near the tested answer (#771). The control frames move at
  most 0.008 px, the library's non-Cassini frames are unchanged (one
  Cassini frame moves 1.98 px onto its verified offset, one rises to its
  expected tier, one moves 0.13 px within its verified uncertainty, the
  rest under 0.03 px), and of 122 frames whose predicted streak moved the
  answer by more than 0.05 px, stars side with the new answer on 71, the
  old on 35 (median separation 0.08 px) and 16 cannot be decided.
  StarUniqueMatchNav's one-star ambiguity test masks a square grown by the
  streak's half extent, so a streak of 12 px or more is not its own
  runner-up; in both reruns one streaked frame reached that test, and it
  passed either way.
  StarFieldFromCatalogNav then counts a source only within reach of a
  catalog star of its cohort: on each axis, no farther from the star's
  prediction than that axis's search-window half-width plus the inlier
  tolerance plus one pixel (#769). On WAC spoke frames crossed by ring
  texture the 30 brightest sources were mostly not stars, so faint stars
  measurable against the rings never reached the matcher: W1726706510 now
  matches 21 inliers 0.04 px from the star consensus with #754's source
  isolation test (20 inliers, 0.05 px without it), where it was refused.
  The wide-offset false-lock model spreads the kept sources at the lower of
  the reach's and the frame's source density, so a sparse field's own stars
  do not set the density they are tested against. On a rerun of 7,091
  frames (the 2,110 of a 3,544-frame study set whose star techniques
  changed, the 4,962 of the ring-alias validation lists and 19 frames the
  density rule moved), 289 verdicts improve and 33 worsen against the
  bundles. Of the 631 refused WAC spoke frames, 95 now answer (70 within
  1 px of the spokes bundle, 25 within 2 px) and none wrongly; the rest
  hold no usable star. Of the 33 worse, 10 are bundle wrong by two or more
  agreeing stars. Eighteen are WAC frames that go from within 1 px of the F
  ring bundle to within 1.26 px: the star answers there sit about
  (0.47, 0.69) px from that bundle on every WAC frame, and stars counted at
  any signal side with the new answer or tie. Two NAC frames go from right
  to 1.61 and 1.22 px (`N1735588543`, `N1786899933`), `N1735601719` from
  1.62 to 2.47 px, and `W1619755013` from 1.68 to 2.08 px from the spokes
  bundle; stars of any signal favor the new answer on all four (5 to 12
  against none). `W1614941856` is the one loss: the star field locks
  0.26 px from the bundle's star position, but pass-2 StarRefineNav moves
  3.2 px onto ring peaks and is fused inside the 5 px grouping floor
  (#767), right at 0.96 px to wrong at 2.45 px. `W1600577960` (two-star
  consensus) stays refused. Of the 1,000 control frames no verdict changes.
  On the combination no library frame changes status or tier, and seven
  answers move closer to their verified offsets (by up to 0.35 px). On this
  branch alone, without the sparse-star work, two library tiers rise to
  high, `W1444747627` falls to its expected medium, and `N1571691490`
  (Hyperion) locks 9.4 px from its operator-verified offset, where four
  stars of signal 7 to 9 lie within 0.7 px of the new answer and none near
  the verified one: that truth needs the operator's review.
- **One strong step, two cataloged edges (open).** Where cataloged edges lie
  a few tens of pixels apart, the strongest alignment can lay the wrong one
  on the strong step: 5 of the 1,041 held-out frames, 7 to 49 px across the
  edge (N1851240381 at 49 px), all at the medium tier.
- **Alias locks across the edge (open, #373).** Answers displaced across the
  edge by a distance that matches no spacing between cataloged edges went
  from 188 to 4 on the 1,041 held-out frames: N1492091385 (45.0 px),
  N1851236541 (43.8 px), N1492064782 (34.4 px) and N1851226701 (3.8 px),
  all at the medium tier. Of the 14 frames the along-edge diagnosis used,
  two stay `conflicted` on a radial lock: N1492071146 (a 42 px alias of
  this kind) and N1601496278 (6 px, the rev 087 A ring outer edge catalog
  error, #761).
- **Few stars against a ring edge (for the sparse-star work).** With the
  along-edge covariance honest, N1746561771 and N1579830292 (F ring movie
  frames the bundle navigated on stars) land 0.08 and 0.44 px from the
  published pointing. N1538201454 is still 4.4 px off, at the low tier: its
  one-star answer carries a sigma of its own offset size and lies within
  the 5 px grouping floor of the ring answer, so the ring edge, 2.7 px
  uncertain along itself, sets the position. Where the single star is the
  right one but lies farther than that from a ring answer as vague as it
  is, the frame is now refused rather than averaged (N1596689242,
  N1789926660, N1866222516). These belong with #621/#622/#623.

- **#346** — three library frames (N1492091163, N1867601758, N1867602424),
  none on a wrong lock. N1492091163's 94 px alias is gone:
  RingEdgeNav converges 1.1 px from operator truth, and a two-star
  StarRefineNav then moves the fused answer 0.7 px in du, to 1.70 px in du
  from truth at the medium tier (the sparse-star family's behavior); its
  `failed` pin wants a decision, and the issue stays open on it. The two
  Encke gap frames are the rank-1 axis above: their offsets now land on
  operator truth at the `low` tier, and on 2026-10-27 the operator re-pinned
  their tier from `medium` to `low`, so both are expected to pass (the
  library has not been run since). Exposure is limited to the sub-25 km/px
  regime by the routing decision above.
- **#476** (closed) — its acceptance criterion read: "any fix should drive
  the RingEdgeNav rows of the shift-equivariance sweep toward the
  DT-technique floor (~0.06 px) on the `ring_only_*` and `ring_plus_body`
  library sets, using the committed baseline as the before-measurement". On
  2026-10-27 the operator restated it as the across-edge component, which is
  met: on those frames RingEdgeNav's round trip along the edge normal is a
  median 0.03 px (worst 0.31 px), and the ensemble's 0.04 px (worst
  0.31 px). The raw rows still carry the along-edge part: RingEdgeNav's
  median |residual| is 4.2 / 1.7 px (v/u), against 3.6 / 1.6 px on the
  earlier `main`. Every frame in those rows has a straight or nearly
  straight edge set, and the remainder lies along the edge, where a straight
  edge carries no information and a nearly straight one very little. The
  along-edge follow-up (#755) makes the reported covariance honest along
  the edge but leaves the offsets, and so these rows, unchanged; that part
  is #755's.

### #350 — post-recalibration resolved-body red

- **#350** — the issue names two frames, and the measurement attributes one
  of them elsewhere: N1484593951 misses the offset tolerance by ~2 px after
  the recalibration, while N1686349893's disagreement is a stale
  `primary_technique` pin on #483. One debugging session against the first
  frame; its sidecar pins it red until resolved.

### #373 — DT coarse-prior search vs competing edge populations

The coarse NCC search over the distance-transform image can lock onto the
wrong edge population and hand the Levenberg-Marquardt refine an
unrecoverable prior (`src/spindoctor/nav_technique/dt_fitting/coarse.py`).
The polarity-weighted seed serves the techniques whose polarity is
predictable; RingEdgeNav stays polarity-blind. Its seed is checked
against edge strength (above): the mask-overlap seed is kept unless the edges
under it are below 0.8 of the strongest alignment in the window, and a fit
whose matched edges are under 0.005 of the frame's strongest is spurious.
That covers competing populations weaker than the cataloged edge -- a ring's
interior texture and a scattered-light glow's faint contours -- which were
most of the lattice locks (alias locks across the edge from 188 to 4 on the
1,041 held-out frames). What remains:

- a competing population stronger than the cataloged edge, such as a body
  limb in the search window: the check prefers the strongest step whatever
  produced it, so it does not separate them;
- the held-out frames still wrong across the edge: 5 neighboring-edge locks
  (7 to 49 px) and 4 alias locks (N1492091385, N1851236541, N1492064782,
  N1851226701; 3.8 to 45 px), all at the medium tier;
- the issue's own calibration datapoint, `N1467344214_2`, has not been
  re-measured against the issue's gate (the #179 no-net-new-failures gate
  over the `ring_only_*` library set).

### #621 / #622 / #623 — the star gates and what the ensemble counts

Three defects measured on the 1030-frame `ISS_006RI_LPHRLFMOV001_PRIME` F ring
block, where stars are the only technique that can navigate at all -- the ring
catalogs carry no F ring, so RingEdgeNav has no feature to fit there. They are
one family: what the star techniques accept, and what the ensemble does with
what they return.

- **#621** — star navigation refuses about half the frames that are
  demonstrably navigable by stars, and not for want of stars: the acceptance
  gates refuse sparse fields. The cost is coverage.
- **#622** — the converse, and the more serious of the two: a solution built
  on one or two matched stars is accepted as `success` and lands five to
  twenty-three pixels wrong. Two stars cannot constrain a two-dimensional
  offset against misidentification, so an accepted two-star answer is a
  confident-wrong exposure. Fix #622 before widening #621's gates.
- **#623** — the ensemble counts two techniques consuming one feature as two
  witnesses agreeing, when they are one measurement reported twice. Same shape
  as the family `resolve_independent_estimators` already handles for seeded
  refines and paired ring techniques; the shared-feature case is the one it
  does not cover. #557 (combined confidence saturating at 0.99 whenever two
  techniques agree) is the same arithmetic seen from the confidence side.

### #128 / #150 — Limb navigation redesign and the ~0.1 px systematic

The strategic pair behind several symptoms (the terminator
mis-convergence class, #187 chaotic rotators). #150 is the measured
~0.09-0.13 px limb bias: the model predicts the geometric silhouette while
the image's gradient ridge sits ~0.1 px inside it (PSF), so
`gradient_ridge_refine` is disabled for the limb technique
(`config_510_techniques.yaml`) while ring edges run with it on. The measured
diagnosis attributes the dominant term to the photometric roll-off, not DT
quantization, and ranks the fixes: (1) fit a photometric limb (predict the
limb-darkened-disc-convolved-with-PSF brightness profile and match it, #150);
(2) a matched-filter sub-pixel edge estimator to remove the interpolation
ripple (#282); (3) a pixel-center convention audit (#283). **Constraint:** do
not enable a fitter change until
the real-image measurement exists (Track A #225 provides it) — the current
partial cancellation is accidental and a well-meaning "fix" can make real
accuracy worse. On real frames the fitter contributes only ~0.1 px while
spacecraft-position / ephemeris error dominates (0.4-1.7 px), so the
higher-leverage pointing-kernel side (#50) was built first and is
delivered. #128 is the fuller redesign (all body types and illuminations)
and starts with a design document, not code.

**#128 now carries landing the held #447 fix.** The redesign starts from
`rf_shift_equivariance_v2` rather than from `main`, and landing that branch
takes three things from it: the model-side correction for the image edge
offset (#150), so the limb answer's absolute position improves rather than
worsens; the simulated body model adopting the same sub-pixel vertex rule,
since `NavModelBodySimulated` still places limb and terminator vertices at
ridge-pixel centers and the limb-bias study, the calibration campaign, and the
limb and terminator sim scenes would otherwise stop describing the production
model (the probe's bracketing helpers are geometry-agnostic, and the simulated
model supplies the inside/outside classifier from its own predicted geometry,
never the planted truth); and the tier and primary-technique pins the fix
moves (`N1572105349_1`, `N1488823805_1`, `N1855878123_1`, `N1487595731_1`),
which are #483's.

**#646 gates the measurement.** The harness that exists to calibrate the limb
bias puts its simulated limb ridge in a different place from the catalog model
the real frames use, about half a pixel away, so it measures the harness rather
than the fitter. Fix it before reading any number it produces.

### Smaller Track B items

- **#25** — high-resolution bodies: the model renders sharper than the
  PSF-blurred image; investigate blur-matching the model
  (`nav_model_body.py`) at high resolution. Investigation first: measure
  whether it actually moves offsets on library close-flyby frames.
- **#130** — per-instrument star limiting magnitudes measured from real
  star fields (a small campaign over the library's star classes;
  coordinate with #233's measured-SNR work — same frames, same tooling).
  Cassini ISS is done on the #755 branch: photometric (CISSCAL filter
  throughput, exposure, gain state) with the minimum detectable DN measured
  on 517 F ring frames (CL1+CL2 full frames; other filters, gains and
  summing rest on the CISSCAL scaling). Galileo, Voyager and New Horizons
  keep their exposure-only anchors.
- **#239** — implement the sub-5 px body policy the operator settled on
  (expected-failure curation); the open work is the targeted diameter-
  filtered cohort scan for a qualifying single-body frame.
- **#591** — `NavModelRings._render`'s cheap 16x16 visibility pre-check can
  skip the whole ring model on a frame whose rings fall between its samples,
  so a navigable ring frame silently offers no ring feature.
- **#577** — `compute_smear_vector_px` calls `obs.boresight_ra()` and
  `obs.boresight_dec()`, which exist on no observation class, so the star
  smear path raises wherever it is reached. Restore the accessor.
- **#579** — the logged star-smear summary projects one sky direction, the
  frame center, so it reads zero on a rolling frame where the smear is
  largest at the corners. A log line, not a navigation input.
- **#338** — decision: the highly-irregular exclusion discards a
  ground-truth terminator fit on N1853392805; implement whichever option the
  operator picks (accept the 2-px-class ground truth, keep TERMINATOR_ARC for
  SPICE-known synchronous rotators, or shape models per #23).
- **#635** — the star pixel datum: a star record carries oops uv (a pixel's
  corner) and every star technique measures array indices (a pixel's center),
  so every star-derived offset was short by half a pixel in both axes. Fixed
  by declaring the convention on `MutableStar` and converting at each point of
  use (`STAR_UV_DATUM_PX`). The library's star-derived ground truths carried
  the same bias and 17 of them were corrected by the datum, each one a pin the
  sidecar records as a star technique's own offset or as a run in which only
  star techniques emitted. Four were left: three fused pins whose correction is
  the datum times a star weight that can only be reconstructed, and
  N1633925572, whose sidecar then carried an operator ruling not to
  re-ratchet. That ruling ended on 2026-10-27, when the tier was re-pinned to
  `low`, but the truth (the one-star match's own offset) still carries the
  uncorrected datum, and correcting it is an open operator decision.
  W1444747627 was pinned by hand against an overlay the notes describe as star
  and body together, so which half the drag followed is not recorded. One
  consequence remains open: N1530185128 flips to a different star lock
  (18.2 px -> 1.2 px, confidence 0.72 -> 0.40) because
  `StarUniqueMatchNav`'s 30-px search window takes the brightest peak inside a
  slab whose integer bounds move with the prediction — a fragility the datum
  change tips rather than causes.
- **#641** — a simulator scene states every position that places something in
  it as a pixel corner, which is the same oops uv a star record carries, so
  the record builder copies a scene position through. The shipped scenes moved
  by half a pixel in the files, and every one of the 64 renders byte-identical
  to what it rendered before.
- **#642** — the simulator turns a planted roll about one point: the frame's
  center in pixel corner coordinates, where the body and ring paths already
  pivoted. `render_stars` was reading `size / 2` as a pixel centric coordinate,
  half a detector pixel away, so a rolled scene planted a star truth its own
  bodies and rings disagreed with. Moving a pivot by `d` shifts every rendered
  position by `(I - R) d`, a rigid `2 |d| sin(theta / 2)` the same near the
  center as far from it, so the one shipped scene with a roll moved its whole
  star field by 0.0169 px and the other 63 render byte-identical. That scene's
  baseline flipped from `conflicted` to `success`: `StarUniqueMatchNav` had been
  confidently answering it 11 px wrong and now declines, which is the #639
  window fragility tipping on a 0.01 px render change in either direction, not a
  navigation improvement this earned. One finding from the same audit is filed
  and left out of that change: a roll rotates positions and poses but not a
  star's smear vector, a companion's position angle, or the background sky field
  (#644).
- **#640** — the star edge cull expresses the four extended-FOV bounds in uv
  before it tests six uv positions against them, so the gate no longer sits
  half a pixel inside the edge it names. Across the 75 library frames one
  star changes state, on `lor_0030713597_0x633_sci`; it lands ten rows into
  the zero-filled extfov margin, so no frame's navigation moves.
- **The pixel-convention audit** — every use of a pixel coordinate in the
  repository was read against the two systems the code names with
  `PIXEL_CENTER_TO_CORNER_PX`: pixel corner, which the `oops` geometry calls
  answer in and which everything a person writes or reads is stated in, and
  pixel centric, which anything addressing an array works in -- including every
  position a feature payload carries. Neither system is universal to a layer;
  each field declares its own, and `feature/geometry.py` states positions
  pixel-centric while its half-open bounding boxes are pixel-corner. Nav core, every GUI
  program, every command-line program, and the documentation were all in
  scope.

  The measured fixes: the Titan haze fit converts the inventory position it
  starts from, worth ~0.71 px on every real Titan frame, and its sub-solar
  direction no longer mixes the two systems, so its degeneracy gate can fire
  (#648, #649). The simulator reduces a truth mask by coverage rather than by
  its center subsample, taking the mask centroid error from 0.142 px to zero
  (#650). A zoomed ring mosaic puts a cell's sub-samples around that cell's own
  coordinate, so its radius and longitude metadata are no longer up to half a
  cell high (#651). The mosaic viewer's readouts and tick labels, the scene
  editor's markers and hit tests, and every position reported to an operator
  are stated where the operator supplies them (#652, #653, #654).

  Two structural changes stand behind those. Feature geometry payloads declare
  their coordinate system on the field, and `predicted_center_vu` means pixel
  centric on every payload that carries one rather than differing between them
  (#655). The sixteen conversions that were
  spelled as a bare `0.5` are named (#660).

  The rest is instrumentation and record. The documentation agrees with the
  code, and the observations guide says where each system is used and where
  the conversion happens (#657). The tests that looked like convention tests
  but passed under every candidate convention now fail on a wrong one (#659).
  The YAML records carry no issue numbers or history (#661). #656 records why
  none of the three instruments positioned to catch a half-pixel offset did:
  a navigated offset is a difference, so an error shared by prediction and
  measurement cancels everywhere except the absolute answer, and the library
  gate's tolerance carries the defect as its pad.

  Two findings of the same class are open. `draw_line` converts a fractional
  coordinate with `int()`, which truncates toward zero rather than rounding to
  the pixel that contains it, so an overlay line is drawn in the wrong pixel
  for negative coordinates and biased low elsewhere (#709). And
  `create_body_backplane` treats two simulator truth fields as interchangeable
  sources for one question when they are the silhouette and the visible extent,
  which are different quantities (#701).
- **Titan haze fit** — the haze solar-symmetry method ships and is validated;
  four measured refinements remain: the arc ray reach sized by the search
  window rather than by where the limb can be (#403), the flat arc-residual
  cap that behaves as a size-dependent gate (#404), the uncharacterized
  extreme-phase edge (#401), and opaque-ring masking that refuses frames
  visible through the C ring or the gaps (#402). The ensemble has never been
  exercised on the oblique covariance the fit reports (#400).
- **#747** — `lm_unconverged_with_polarity_rejection` hard-rejects a
  `BodyLimbNav` or `BodyTerminatorNav` fit when `LMRefineResult.converged` is
  False and the polarity rejection fraction is at least 0.10, and below 0.35
  rejection the convergence flag is the only discriminant. The flag is set
  only by an accepted LM step below 1e-3 px, so at a non-smooth minimum of the
  bilinear-sampled DT cost every trial is rejected, the 30-iteration cap is
  reached, and a stationary fit reads unconverged: on `main` 17 of 48 library
  DT fits report converged though 44 are stationary, and the flag flips in
  both directions with sub-pixel model details. Counting a rejected trial step
  below tolerance as convergence would disarm the gate on four fits it rejects
  today, so the discriminant has to be re-derived against labeled healthy and
  mis-locked fits; the healthy frames the gate cites (`N1702389073`,
  `N1702389662`, `N1702390017`) have to join the library first.
- **#748** — decision: a body's witness blob serves the body-witness veto as
  well as standing as an answer, and the veto's 2-sigma tolerance survives
  more clipping than the answer's 0.05 R centroid-bias allowance. On a
  constructed case (R = 200 px, a limb locked 30 px wrong, 72 directions) a
  witness kept up to an overflow fraction of about 0.105 still catches the
  wrong limb, and above that its own bias raises false conflicts. Decide
  whether the witness role takes its own overflow limit, derived from the
  veto's tolerance, separate from `bodies.blob_max_overflow_fraction`. No
  library frame falls in the band with a running blob.
- **#749** — the ensemble's multi-body exclusion asks `_distinct_body_count`,
  which counts bodies that have results rather than bodies in the frame. A
  body that yields no result reads as absent, a two-body frame reads as
  single-body, and the shape-lock veto can fire on the companion although the
  other body still corrupts its centroid. Count the bodies the models placed
  in the field of view instead. No library frame is known to reach it.

## Track C — Statistics and QA

- **#240** — coverage-matrix invariant test: every scene class >= 2
  sidecars, every autonomous technique the expected primary somewhere.
  Lives with the library structural tests
  (`tests/integration/test_image_library.py`); runs in the deliberate
  tier, marked expected-incomplete until #235 fills `faint_stars` and
  `ring_only_flat`.
- **#533** — a file that could not be retrieved is counted by a tree-backed
  report and by no index-backed one. The ingest deliberately records no
  refusal for a retrieval that failed, because a retrieval that failed once
  is worth trying again; that is right for the ingest and it makes the two
  storages disagree about one of the six refusal kinds. Close it by deciding
  which count is the honest one, not by making the ingest lie.
- **#535** — the statistics report retains more than it prints under
  `--top-n`. Small; the fix is to bound what the accumulator keeps.
- **#340** — `library_crosscheck` records only a yes/no primary-technique
  flag, not the winning technique, so a cross-check delta cannot say which
  technique took over. Cheap, and it makes the standing practice below
  actually diagnostic.
- **Standing practice** — after any calibration- or technique-affecting
  merge: `util/calibration/library_crosscheck.py` over the full library,
  every per-image delta accounted for; `sd_stats_report` over campaign
  outputs as the accuracy checkpoint. Note that the practice is currently
  degraded: with 8 of 75 library frames expected red locally (#288, #744,
  #639; 16 at the last run, before eight frames were re-pinned on
  2026-10-27 and not run since), a cross-check can only be read as "no
  *new* deltas against `main`".

## Track D — Capability completion

### PDS4 output bundles (required for all four instruments)

PDS4 *output* (bundle generation) and PDS4 *input* (reading
PDS4-archived data as a dataset source) are different things. Output
bundles are mandatory for every instrument. Input is
availability-contingent: no PDS4 archive of these datasets exists yet,
producing one is external development outside this project's control,
and input support (#34, `dataset_pds4.py`) is not required for project
completion — when an archive appears, implementing its `DataSetPDS4`
replaces the PDS3 source for that instrument.

Output current state: the Cassini ISS Saturn path is finished and validated
as a prototype. The per-dataset hook pattern (template dir, LID/LIDVID
builders, template variables) lives on `DataSetPDS3CassiniISS`; the
collection machinery writes the data, browse, context, document, SPICE
kernel and miscellaneous collections with conforming inventories; the data
labels state real exposure times, name their targets, describe the backplane
FITS beside them and cite their source products; the global bodies and rings
index tables state each image's ring and body ranges; and `sd_create_bundle
check` holds a written bundle to the PDS4 schemas, the Schematron rules and
its own tables. All of it is gated by tests over a synthetic cohort of
navigation and backplane products. The design of record is
`plans/archive/PDS4_DRAFT_BUNDLE_PLAN_2026-09-08.md`, which finished as a
prototype over that cohort.

Two things remain. **#708** is the draft run over a real COISS volume and a
pass of the NASA `validate` tool over its output; it needs registered DOIs
for the bundle and its user guide, and a fresh navigation of the chosen
volume. And the other three instruments still hit `NotImplementedError` walls
in their `pds4_*` DataSet hooks, which is the second half of #53: per-mission
template trees plus hook implementations, mechanical but voluminous.

Work items, in dependency order:

1. **#708** — the real-volume run. What blocks it is external (DOIs) and
   operational (a fresh navigation), not code.
2. **The real-run gaps the prototype does not hit** — the synthetic cohort
   writes no holdings tree, so `sd_create_bundle` cannot enumerate it and an
   operator pointed at it gets an empty bundle at exit 0 (#609); and the
   `cassini:ISS_Specific_Attributes` mission area is empty, because every
   `cassini:*` variable is mapped from a PDS3 index row and fourteen of the
   names it asks for are not columns of one; the writer takes them from the
   observation block instead (#717). These are the items to clear
   before #708 is worth attempting.
3. **The decisions still open on the bundle's content** — what a bundle says
   about an image that did not navigate (#600) and about a navigated image
   whose backplanes hold nothing worth describing, which today is dropped
   with no record (#720); which SPICE kernels the metakernel lists (#677); and
   whether to add a global navigation index beside the bodies and rings
   indexes (#710).
4. **#595 through #599** — the backplanes user guides. Every bundle's document
   collection declares a `Product_Document` naming a PDF that does not exist;
   #595 is the shared LaTeX template and #596-#599 are the per-instrument
   guides.
5. **#30** — backplane label design (couples to the #55 backplane-set
   decision).
6. **#79** — scrape the PDS4 context products to maintain the targets
   table, `backplanes.target_lids`, which Phase 8 of the PDS4 plan filled by
   hand from the PDS registry.
7. **#67** — cloud-aware bundle generation (with the Track D cloud
   audit). #716 is the near half of it: the check pass walks the tree with
   `pathlib` and reads schemas with lxml, so it refuses a remote
   `--bundle-results-root` that the labels and summary passes accept.
8. **#614** — `sd_create_bundle labels` and `summary` end in a traceback
   rather than a clean refusal for five of the seven registered dataset
   families. Cheap, and it is what an operator meets first when the answer is
   "this dataset has no PDS4 support yet".
9. **#705** — the spelling gate would read the NASA PDS4 schema copies the
   package and the test suite ship, which are NASA's files byte for byte and
   carry NASA's own misspellings. Exclude them by path; do not edit them.
10. **#687** — cite each image's calibrated product by its PDS4 LID once a
    calibrated Cassini ISS bundle is registered; today the
    `Source_Product_External` names the PDS3 product by volume ID and label
    path, because no such bundle exists.
11. Schema-validate generated `.lblx` against the PDS4 schemas in CI for
    all four instruments (acceptance for the whole family). Cassini's is
    done: the test gating `sd_create_bundle check` over its cohort. Each
    other instrument's bundle is held to the same check over a cohort of its
    own.

### Backplane family (decision: #28 scope)

Issues #55 (final backplane set) and #57 (FITS HDU content) are
decisions that gate #54 (cropping), #77 (optional args), and #63 (bodies far from
planets). The generator machinery exists
(`src/spindoctor/cli/backplanes/`); tests are Track E #241. End-product
value correctness is Track A #232.

Product-correctness defects found by the #241 test suite, each pinned
by a strict xfail in `tests/spindoctor/cli/backplanes/`:

- **#251** — ring-won pixels carry no BODY_ID_MAP entry, so a
  rings-only image ships no ID map at all and viewer masking treats
  ring pixels as invalid. Needs an ID-source decision first (the dev
  guide's `bodn2c('SATURN_RINGS')` suggestion raises).
- **#252** — an occluding ring never takes ownership of body planes or
  the ID map, against the dev guide's nearest-source rule; plausibly
  intentional (translucent rings), so this is a code-or-doc decision.
- **#253** — the FITS sidecar lacks dev-guide-promised content
  (per-plane mean and valid-pixel count, per-body NAIF IDs and
  bounding boxes, an observation metadata block); couples to the
  #55/#57 decisions.

Two further items in this family, neither found by that suite:

- **#496** — `sd_backplanes_cloud_tasks` lets a per-image failure escape the
  task handler, so the worker's failure mode is the handler's rather than
  the pipeline's. Same family as #418: decide what a task's status owes a
  retrying queue.
- **#611** — `sd_backplane_viewer` decides whether a plane is angular with an
  exact string match on the FITS `BUNIT`, so a plane whose unit is a compound
  radian unit is displayed and read out in radians while every other angular
  plane is in degrees.
- **#618** — `create_ring_backplanes` builds the ring target name as
  `f'{closest_planet}_RING_SYSTEM'` with a hardcoded exception routing Saturn
  to `SATURN_MAIN_RINGS`. Each planet's ring target belongs in configuration.
- **#520** — move the pointing selection and application code
  (`src/spindoctor/cli/reproj/offsets.py`: metadata pointing selection and
  the C-matrix/offset application ladder) out of the reprojection CLI
  package. It already has two consumers, the reprojection and backplane
  stages, which is the condition the code's own comment names for promoting
  it into the library package proper.

### Reprojection and mosaics

The machinery ships (`src/spindoctor/reproj/`, `sd_mosaic`,
`sd_mosaic_display`). The open items are product correctness and
photometry:

- **#634** — every body reprojection and body mosaic records `sub_solar_lon`
  and `sub_observer_lon` in west longitude while the grid they sit on is
  east, and the same file says `lon_direction: east`. Nothing in the product
  distinguishes the two, so a consumer reading them together is wrong by
  `360 - lon`.
- **#638** — three read paths in `reproj/_serialization.py` call
  `get_local_path()` where they need `retrieve()`, so reading a reprojection
  or mosaic product from remote storage never fetches it.
- **#616** — mosaics show brightness steps at image boundaries: the
  photometric models correct per pixel for incidence and emission and none of
  them models phase, and nothing levels the seams afterward.
- **#617** — `MinnaertModel.k` is fixed at 0.5 and nothing can change it,
  which is the wrong limb-darkening exponent for the icy satellites. Pairs
  with #616; a configurable exponent is the cheaper half.
- **#718** — `--zoom` names two different operations: the ring path averages
  sub-samples of the output grid, the body path interpolates the source
  image. Decide whether one flag should mean one thing.
- **#733** — `sd_mosaic` is written for an optional navigation results root
  (`str | None` annotations and two `is not None` guards in
  `src/spindoctor/cli/sd_mosaic.py`), but `get_nav_results_root` raises when
  no root is set and never returns `None`, so those branches are unreachable.
  Decide whether a mosaic on uncorrected pointing should be possible: if so,
  catch the refusal and leave the path `None`; if not, remove the `None`
  handling. Either way, move the call inside the
  `reporting_configuration_errors()` block, so an unset root prints the
  one-line message rather than a `ValueError` traceback.
- **#738** — `--resolution-threshold` takes any float
  (`src/spindoctor/cli/reproj/args.py`), and `BodyMosaic.add` applies it
  unchecked, although its contract says the value should be at least 1.0.
  Below 1.0 a coarser pixel can replace a finer one. Refuse the value where it
  is parsed, naming the option and the bound.
- **#493** — `sd_mosaic` reports a lost results index as N failed images
  rather than as one run-level condition (see the cloud items below).

### CK kernels: what remains after the deliverable (follow-ups)

The "updated pointing" deliverable is built, both halves. The navigator
records a corrected camera C-matrix beside the pixel offset in the metadata
of each eligible image (a fitted-rotation result records only the
uncorrected attitude, and a simulated image records neither), and a
cspyce-only `sd_create_ck` writes one type-3 segment
per eligible navigated exposure into files that mirror the originals they
correct -- an exposure whose baseline no candidate reproduces, or that
yields to its simultaneous partner, is reported omitted rather than
written --
with a meta-kernel and a per-mission CSV report beside them. The designs of
record are `plans/archive/CK_KERNEL_PLAN_2026-08-04.md` (the writing half) and
`plans/archive/CMATRIX_READERS_PLAN_2026-08-09.md` (the reading half: the backplane and
reprojection readers apply the recorded C-matrix by frame replacement, with
the offset as the documented fallback); the consumer-facing and
developer-facing documentation is `docs/user_guide/user_guide_ck_kernels.rst`
and `docs/dev_guide/dev_guide_ck_kernels.rst`. Validation is a closed loop:
navigate, generate a kernel, furnish it, re-navigate, and confirm the
second offset is approximately zero and the C-matrix matches, which the
round trip does on Cassini NAC, Cassini WAC, Voyager and LORRI; the reader
comparison (`tests/integration/test_cmatrix_readers.py`) holds the two
consumption paths together through the switched consumers themselves.

The round trip also measured something that belongs to Track B rather than
here: a technique re-measuring a corrected frame does not return exactly
the negative of the shift it was given, which costs up to 0.49 px on frames
carried by the correlation and distance-transform body techniques (#447).
The fix for both body-side causes is held on `rf_shift_equivariance_v2` to
land with the limb redesign (#128).

Two of the plan's follow-ups turned out to be navigator defects rather than
kernel work. Rotation fitting is now off for Galileo SSI, so no instrument
fits rotation, the fitted-rotation omission costs nothing, and Galileo gets
the corrected kernels it previously got none of. The conflicted result that
drops its fitted rotation (#521) is thereby unreachable and is deferred until
an instrument fits rotation again.

The rest of the plan's follow-ups are filed and none blocks use of the
kernels: the remaining half of the oops API swap (#433) -- the conversion
between the oops and SPICE conventions now goes through oops, and what is
left is the offset-to-rotation construction, waiting on SETI/rms-oops#213 --
fitted-twist support (#434) with the static-twist FK/IK question behind it
(#435, #436), SPICE database registration (#437), the interior-epoch
fidelity bound through an adaptive record cadence (#440, #444) with its
per-instrument characterization (#455), the kernel-input handling items
(#446, #448, #468), and a memory bound: `sd_create_ck` holds a whole mission
in memory where a time-ordered stream would not (#513, which is also the
consumer that would use a `Selection` order parameter).

Operationally, the program is still hard to run: locating the kernels it
needs takes several `--kernel-dir` flags and still misses some. That is the
practical face of #448 (locate C-kernel inputs through `spyceman` instead of
a kernel directory tree) and should be weighed when #448 is scheduled.

### The results index: delivered, and what it left open

Delivered and merged to `main` on 2026-08-25 (#430, #487, #507). An optional,
rebuildable database index over the results tree, so that programs needing a
few fields per image stop reading one JSON document each -- one paid cloud
round trip per image per program otherwise, at order 400,000 images for a
Cassini-scale run. The JSON documents stay authoritative, **no program
requires the index**, and the file-reading paths remain the default: a run
that names no index reads the tree exactly as it always has.

What it changed beyond the index itself is the more important half. Every
program that reads a navigation record now reads it through one seam,
`spindoctor/nav_records/`, over both storages, with one row-to-record
rebuild, one opener, and one enumerated statement of where the two storages
answer differently. A program becomes index-backed by *declaring*
`--results-index-db`, never by inheriting an exported environment variable.
The design is archived at `plans/archive/RESULTS_INDEX_PLAN_2026-08-04.md`;
current behavior is `docs/user_guide/user_guide_results_index.rst` and
`docs/dev_guide/dev_guide_results_index.rst`.

Two of its acceptance criteria openly do not hold, which is stated rather
than papered over: no written product is compared between the two storages,
because the one integration frame that would do it no longer navigates
(#547, blocked behind the image-library regression #288), and suite coverage
is 79% against a stated floor of 90% with nothing enforcing it (#548, an
operator decision -- raise the number or ratify a lower floor).

The ingest is fast enough over a cloud root to be routine: a bucket of 52,101
documents ingests cold in 240 seconds and re-ingests, with nothing changed, in
13. Both halves of a pass are latency rather than bandwidth on such a root -- a
listing is one round trip and a document is another, and neither moves enough
bytes to matter -- so the walk lists directories in parallel, a bounded round at
a time, and retrieval passes a thread count with a batch large enough to keep
that pool full. How much of either runs at once is the `results_tree`
configuration section, resolved once per program and passed down; every
program that reads a tree honors it. What is left is the service's own round
trip: streaming the documents without storing them measures no faster than
retrieving them.

Its follow-ups, none blocking:

- **Capability extensions:** a document column so bundle generation stops
  reading one file per image (#464, likely a separate optional database
  since it roughly doubles the index size); a schema wide enough to serve
  the curation triage tool (#465, wanted only if triage's ten rglobs per
  frame remain a practical pain); a `--since` selector so a re-scan stops
  paying for the whole listing (#467); `sd_offset` writing each result into
  the index as it navigates (#486); and an index that can say whether a
  root's rows were pruned (#542); and a URL with no scheme taken to mean a
  local SQLite file, so that `--results-index-db nav_index.sqlite3` means what
  it looks like it means rather than being refused (#572); and
  `--results-index-db` declared by the three programs that take the results
  filters without it -- `sd_create_bundle`, `sd_consolidate_metadata`, and
  `sd_backplane_viewer` -- so their filters can read a results index rather
  than always walking the tree (#736). The fix is for each program to declare
  the option; the seam must not start reading `NAV_RESULTS_INDEX_DB` on their
  behalf.
- **Operational questions with a decision in them:** a documented workflow
  for getting the index to cloud workers (#466 -- publish the SQLite file to
  the results bucket, or run PostgreSQL); and the lockability probe that
  takes a SQLite write lock a consumer never needs (#462), which cloud-task
  ingest makes routine rather than occasional.
- **Correctness and hygiene:** a cloud-share ingest that can write another
  root's document into this root's rows (#515) with the test that cannot
  catch it (#516) -- the same threat model, to be answered together; queries
  bound to the search path rather than the resolved schema (#501); the
  document-to-column placement still written twice (#512); the ingest's own
  retrieve-and-parse loop beside the seam's (#514); the seam overstating what
  a missing row means (#536); documents recording where the image was cached
  rather than where it came from (#531); no format version on the metadata
  documents (#528); a mistyped `--nav-results-root` reading as a tree in
  which nothing was navigated (#538); a narrow selection still listing a
  whole volume (#540); signed zero not surviving SQLite (#534); roots taken
  as `str` rather than the `FCPath` union (#472); and the FileCache/FCPath
  construction audit (#541); a results tree reached through a symlink giving
  a document a different stub on each pass, because `_claim` keeps whichever
  path the walk met first (#578); and a stale index passing the version gate
  and failing on the first read, because `SCHEMA_VERSION` is a column-set
  version and does not move when the column contents change (#587).
- **Reported through other programs:** `sd_mosaic` reports a lost index as N
  failed images rather than one run-level condition (#493), and there is no
  integration-tier case for a document the ingest refused (#497).

One dependency rather than a follow-up: `FileCacheSourceS3.iterdir_metadata`
issues one `list_objects_v2` and reads that single response, so an S3 prefix
holding more than a thousand objects lists short and says nothing
(SETI/rms-filecache#65). The seam's completeness guarantee assumes a listing
is complete or raises, and there is deliberately no workaround here. GS
auto-paginates and is unaffected.

### Metadata document precision (#730, a decision)

`src/spindoctor/nav_orchestrator/curator.py` rounds as it assembles the
metadata document: `PIXEL_DECIMALS = 4` for offsets, sigmas, and covariance
entries, `CONFIDENCE_DECIMALS = 3` for confidences, reliability components, and
classifier scores, and `ET_DECIMALS = 6` for `provenance.image_et`. The
top-level `offset` and `confidence`, the C-matrices, and `times.start_et`,
`times.stop_et`, and `times.midtime_et` are written unrounded, so `navigation_result.offset_px` and the top-level `offset` can
disagree in the last place. The operator decides whether every stored value
is written at full precision with rounding left to reports and logs, or names
what makes a rounded value different. Whatever is decided, the two fields
holding one quantity agree, and the rounding table in
`docs/user_guide/user_guide_metadata.rst` changes with it.

### Command-line consistency

- **#734** — `--profile` is declared by three programs and works in one.
  `sd_mosaic` prints its profile; `sd_offset` starts a profiler and never
  stops or reports it, so a run pays the overhead for nothing; and
  `sd_backplanes` declares the flag and never reads it. One small context
  manager shared by all three, printing on exit or writing to a named file,
  keeps a fourth program from acquiring the flag without the report.
- **#735** — `sd_create_ck` selects only by `--start-time` / `--stop-time`,
  and `sd_stats_report` by its own `--instrument`, `--start-date`,
  `--min-image`, and `--max-image`; neither takes the volume, image-list, and
  image-number options the other six programs share through
  `add_selection_arguments`. For `sd_create_ck` adding them is additive; for
  `sd_stats_report` it carries a choice between aliasing and replacing its own
  spellings. Writing kernels or a report for one volume is the commonest
  request neither program can serve today.

### Capability matrix (#231)

Generated/test-verified matrix; see the issue for the two-axis design
(feature support x validation status). Implementation home:
`docs/user_guide/` page generated from the registries
(`spindoctor.dataset`, `spindoctor.obs`, technique registry) plus a
static validation-status table that the WS reports update; a test
asserts the generated half matches the registries.

### Cloud and scale

- **Navigation memory** -- a navigation's resident size is bounded
  by the extended frame, which is the detector plus twice the instrument's
  search margin, and `oops` sizes its intermediates by the meshgrid it is
  handed. The wide-margin instruments are therefore the expensive ones: a
  Voyager frame extends to 3.24 Mpx against a 1000x1000 image. Bounding
  Titan's two backplane boxes, striping the ring, body and Titan backplanes,
  and collecting each strip's transient memory and returning the freed arenas
  to the operating system where the C library can, brought Voyager Saturn
  frames from 16.7-26.0 GB to 7.47-7.72 GB; dropping the observation's
  backplanes where the model stage ends and correlating through half spectra
  then brought the six frames that still exceeded the 8 GB working target a
  cloud worker's instance size implies from 8.02-9.82 GB to 6.39-7.66 GB,
  with every offset and status unchanged and roughly half the runtime on the
  heaviest. Two of those findings cost more to reach than they should have.
  Striping alone bought nothing measurable until the release was added,
  because oops intermediates live in reference cycles and glibc retains freed
  arenas, so a striped pass grew by the sum of its strips rather than the
  largest. And the residue a ring render leaves was recorded as unreturnable
  fragmentation on a probe that cleared one cache on one of the three live
  backplanes; it is held, by the observation's own backplane caches, and
  dropping them takes settled resident size from 6.80 GB to 1.77 GB. The
  model stage now sets the peak, at 7.66 GB on the heaviest frame, 0.34 GB
  of headroom against a run-to-run spread of about 0.1 GB; #584 is the
  measured way to widen it and #585 is the question the padding raises. Each
  navigation records the peak it reached, in the metadata document and in the
  results index, and the statistics report summarizes it per instrument with
  a histogram and the hungriest images, so the next regression is read off a
  pass's own numbers rather than found by a cloud pool dying. The record is
  null where the kernel publishes no peak or will not let the mark be reset
  ahead of the image, `/proc` being Linux-only. The release placements
  measured to be worth nothing are recorded in
  `docs/dev_guide/dev_guide_memory.rst` so they are not tried again. The ring
  radius, radial resolution, `border_atop` and `radial_mode` backplanes
  remain whole-frame through the extended backplane and set the ring model's
  remaining floor until striping lands in oops: striping belongs in `oops`
  rather than in each consumer, which is SETI/rms-oops#222. A striped answer
  holds to the photon solver's convergence tolerance rather than to the last
  bit, because oops converges on the largest light-time change anywhere on
  the meshgrid; the measured agreement, now including the body quantities, is
  recorded in `docs/dev_guide/dev_guide_memory.rst`. Titan's envelope box is
  undersampled rather than striped because its work, not just its memory, is
  unbounded; #594 would remove that box altogether by projecting the
  sub-solar direction instead of searching a backplane for it.

- **#108** — audit every `sd_*` CLI for logging, cloud operation, and
  working `cloud_tasks` variants; fix what the audit finds. The logging
  third is done: every pipeline program takes the same flags with the same
  defaults, resolves per-module levels the same way, and writes a main log
  plus per-image logs to the same path scheme, with the statistics and GUI
  programs deliberately excluded and on `print()`. What the audit found and
  did not fix is now three issues — #418, #423, #424 below — and what
  remains under #108 itself is the cloud-operation half: whether each
  driver's cloud path actually works end to end, which is untested for
  several of them. The task files such a run needs, and the instance
  startup script it runs under, are scripted in `cloud_support/`.
- **#574** — `cloud_support/` is outside every check the repository runs:
  `ruff`, `ruff format --check` and `mypy` are invoked over `src tests` and
  `pytest` collects `tests/`, so the four task generators and their shared
  module are unlinted, untyped and untested. Sequence it with #108, which is
  the work that will exercise them.
- **#706** — the 24 files in `src/spindoctor/config_files/` reach a built
  wheel only through setuptools-scm's git file finder, so a wheel built from
  an unpacked sdist or a tarball ships no configuration at all. Name them in
  `[tool.setuptools.package-data]`.
- **#418** — a `sd_mosaic_cloud_tasks` task returns `status: success` no
  matter how many of its images failed. The counts are in the result now
  (`n_uncorrected`, `pointing_reasons`, `rejected_stubs`), so the
  information exists; the question the issue records is whether `status`
  should reflect it, given that a queue keys retry off `status` and
  "retry the whole task because one image had no offset file" is usually
  the wrong response. Decide the policy before coding it.
- **#423** — the mosaic and backplane GUI viewers import library code that
  logs, but construct no handlers, so pdslogger's handler-less fallback
  prints those records to stdout at every level. Pre-existing, and the fix
  is small (bind the null handler, or give the viewers a real logger); the
  cost is that about ten tests currently capture that fallback's output via
  `capsys` and would need their capture strategy changed first. Do that
  first, then the fix.
- **#424** — remove `sd_create_bundle_cloud_tasks` and its `pyproject.toml`
  entry point. Bundle assembly is a packaging step over an
  already-processed collection, not per-image work suited to a task queue;
  the module is unwired for logging and leaks to the worker terminal, and
  fixing it would be maintaining something that should not exist.
- **#731** — log file names are stamped in UTC (`run_timestamp()` in
  `src/spindoctor/config/logging_config.py`) while the records inside them
  are stamped in naive local time by pdslogger, and nothing in the file says
  which clock each half is on. Stamping the records in UTC as well, through a
  pdslogger change or a formatter installed here, is the option that makes a
  log comparable against other timestamped records.
- **#411** — migrate the technique config keys to snake_case, so the
  configuration spells component names the way `log_key_for` and the rest
  of the config system already do.
- **#427** — the config top-level namespace mixes domain objects, pipeline
  stages, instruments, output products, and infrastructure on no stated
  axis, with `general` as the catch-all. Lifting `logging` out of `general`
  fixed the worst instance and left the pattern. Sequence this **before**
  #118 (config validation): a schema per section is easier to write and to
  keep honest once the sections are organized, and doing it the other way
  freezes the current grouping into a schema. #712 (renumber the
  configuration files, so the `config_NN_*` merge order carries the same
  axis) is the filename half of the same question and moves with it.
- **#141 / #142** — dedup the CLI driver preamble and cloud-task loop;
  fix the dropped `extra_params` and the ImageFiles cardinality
  disagreement.
- **#493** — `sd_mosaic` reports a lost results index as N failed images
  rather than as one run-level condition. Same shape of defect as #418
  and #496 and worth fixing with them: a run-level fault reported per image
  reads to an operator as N unrelated failures.
- **#515 / #516** — a cloud-share ingest can write another root's document
  into this root's rows, because `_share_from_task` reads the stub straight
  out of the task JSON and never builds one; the fix is to call
  `stub_refusal` in its existing per-entry loop. The test that should catch
  it asserts on the row a root-blind write would keep, so it cannot fail.
  Same threat model as a hand-written task file naming a stub outside its
  root; answer them together.
- **#466** — a documented workflow for getting the index to cloud workers:
  publish the SQLite file to the results bucket and have each worker
  download it once, or run PostgreSQL. This is a decision before it is
  work, and it is the item that decides whether #464's document column
  (which roughly doubles the index size) is affordable.
- **#495** — add raw-product dataset names for Cassini ISS (`coiss_raw` and
  family), so a run can name the raw products the way it names the
  calibrated ones.
- **#236** — profiling + supported batch-parallel path (issue has the
  breakdown; respects #103/#134 thread-safety constraints — per-thread
  `Backplane` objects, no shared `obs`).
- **#126** — rotation-pyramid cost (~10 min on 1024^2); only bites
  rotation-fitting instruments (Galileo, Voyager); coordinate with
  Track F instrument enablement.
- **#118** — config validation system: schema per section, unknown-key
  rejection, type/range checks at load
  (`src/spindoctor/config/config.py`); pairs well with #176's
  constants-into-config completion.

## Track E — Test and documentation debt

- **#241 / #242** — unit tests for `spindoctor.cli.backplanes` and
  `spindoctor.cli.pds4`. The backend halves are covered (hermetic,
  spec-first); the remaining scope is the `sd_backplanes.py` /
  `sd_create_bundle.py` driver arg-parsing layer, which should fold into
  the broader sd_*-driver test effort. The backplane suite carries strict
  xfails for #251, #252, #253, ready to flip when each fix lands.
- **#288** — image-library regression reconciliation, and the piece of test
  debt with the widest blast radius. In the local integration environment 8
  of 75 sidecars are expected to disagree, each one attributed and owned, but
  the set is still wider than the deliberately-pinned one below, so the
  regression instrument cannot yet be read as clean. The last library run
  found 16; eight of those were re-pinned on 2026-10-27 (five
  `ring_only_flat` frames whose pins recorded the old quantized rank-1 axis,
  the tier pins of #346's two Encke gap frames, and the tier pin of
  `N1633925572_1`, which no longer runs RingEdgeNav; the values are recorded
  in the Operator Playbook) and are expected to pass, but the library has not
  been run since. Two, the Galileo
  frames `C0059894800R` and `C0059899900R`, record the rotation-fitted
  answer that the change in #560 retired; re-recording their truth from an
  outside measurement, or pinning them red under #434, is the operator
  decision in #744. Another, `N1530185128_1`, is owned by the star-centroid
  defect in #639. Two consequences to hold onto: a navigation-affecting
  branch can only be gated on *no new failures against `main`*, and #547
  (the one place a built product is compared between the results tree and
  the results index) is blocked because the frame it needs no longer
  navigates. Reconciling this is prerequisite to #483's re-ratchet and to
  reading the Track A cohorts with confidence.

  The intended steady state, and what the pins mean when it is reached: the
  standing red set reduced to the deliberately-pinned frames, each owned by an
  open navigation issue (#338, #346, #350). W1444747627 (#352) is re-ratcheted
  to success/medium and green: the star-navigation hardening locks it.
  N1530185128 was re-ratcheted to success/medium under #351, once the ensemble
  blob-outlier drop let the multi-star fix commit, but it has been red since
  the star datum correction (#643), with a dv error of 17.472 px, and is owned
  by the star-centroid defect in #639. N1806609736 (#392) is green as well:
  the shape-lock veto no longer misfires when a star fix corroborates the
  limb. N1572105349's pin was owned by #222 (now closed by the independence
  resolution); its autonomous regression should flip green on the next
  integration run against real holdings (that suite does not run in PR CI).
  Keep the pins accurate as issues close and re-ratchet only what they
  authorize.
- **#243** — direct tests for `nav_model/stars/conflicts.py`
  (`_check_one_star`, `mark_body_and_ring_conflicts`) with synthetic
  geometry; the existing per-pixel occlusion tests cover only the
  occlusion-fraction path.
- **#174** — regression baselines beyond the single frame: seed
  baselines for the full library (`python -m
  tests.integration.update_baselines --all`, which requires
  `PDS3_HOLDINGS_DIR`) and commit them with the per-image diff
  accounted for.
- **#177** — unit tests for `spindoctor.support.summary_png`.
- **#129** — drive Sphinx nitpicky warnings to zero, then add `-n` to
  the CI docs build.
- **#122** — verify the albedo/terminator-sharpness rationale in the
  body-terminator dev guide against the shipped implementation.
- **#429** — give the `util/` tooling the logging surface the pipeline
  programs have. Several of these programs run for hours over hundreds of
  images and report through bare `print()`: no level control, no file
  record of a campaign, no way to raise one component's verbosity. Not all
  of them qualify — a script that prints four lines does not need a logger.
  Scope to settle when picked up: which programs qualify, whether they
  declare program names alongside the shipped drivers (they are not `sd_*`
  entry points, so `logging.programs` would gain non-shipped keys), and
  where their log files belong, since `util/` output does not live under a
  results root.
- **#391** — pin the lint tools (`ruff`, `mypy`, `pymarkdownlnt`) in the
  `dev` group, or move linting to pinned `pre-commit` hooks, so a new
  release cannot turn `main` red without a deliberate code change. A ruff
  0.16 release promoting RUF036 did exactly that mid-batch; the offending
  lines were fixed, but the unpinned-linter exposure remains.
- **#548** — suite coverage measures 79%, two shipped plans' acceptance
  criteria assert 90%, and nothing enforces either figure. The shortfall is
  almost entirely PyQt6 widget code. This is an operator decision before it
  is an implementer's: raise coverage to the stated floor, or ratify a lower
  one and gate CI on that. Leaving the claim and the absent gate both in
  place is the one option that keeps asserting something untrue. Related
  cautionary note: `pytest --cov` was recorded as broken here for months and
  runs fine, which is how the figure went unmeasured for so long.
- **#547** — acceptance criterion 1 of the results-index work compares a
  written product between the two storages, and the one integration frame
  that would do it no longer navigates, so that half of the criterion runs
  against no frame. Either pick a frame that still navigates, or close it
  behind #288.
- **Tests that cannot fail.** This project has now hit the same failure
  three times, and it deserves a named place in the plan rather than a line
  in a PR body: a test whose docstring promises more than its body checks
  passes against the very defect it exists to catch. Open instances: #516
  (`test_a_share_only_writes_its_own_root` asserts on the row a root-blind
  write would keep). When writing a guard, verify it by breaking the source
  and confirming the test fails; a test that stays green against a
  deliberately broken implementation is a defect in the test and is reported
  as one.
- **#483** — re-ratchet the library pins the shift-equivariance fix (#447)
  moves. Those deltas arrive with `rf_shift_equivariance_v2` when #128 lands
  it. Gated on #288: pins cannot be re-ratcheted honestly
  against a baseline that is itself 10/75 red.
- **#524** — consolidate `test_record_source.py` onto the shared
  results-index fixtures, which already exist and are built by the writer
  that makes real results trees.
- **#525** — seventeen test modules exceed the 1000-line module cap the
  rulebook sets for source.
- **#473** — the test suite claims Windows support while using POSIX-only
  constructs. Decide which is true and make the suite match it.
- **#530** — Cassini fixture clock seconds do not follow from their epochs,
  so a fixture that looks self-consistent is not.
- **#438** — the Sphinx nitpicky gate is required by the project rules and
  is not run; pairs with #129, which is the work of getting to zero.
- **#545** — the README's Features entries carry no link to the user-guide
  chapter behind each. The program mentions and the Quick Start invocations
  the issue also asked for are in place. Cheap, and it is the first page
  anyone reads.
- **#549** — four technique guides name a `dt_fitting.py` that is a package.
- **#470 / #471** — config placeholder comments and config placeholders
  themselves still carry an internal phase codename; documentation and
  configuration should be self-contained.
- **#562** — 62 of 64 dev-guide chapters carry no class diagram, which the
  developer-guide rules require of every chapter. Decide whether the rule
  holds for every chapter or only where a class structure is the subject,
  then make the guides and the rule agree.
- **#714** — Python source carries 332 em dashes and 23 arrows that the
  docstring rule prohibits in `.py` files, and no check gates them. The gate
  is the point; the sweep without one only resets the count.
- **#715** — the Qt modules document 7% of their method arguments where the
  rest of `src/` documents 65%. Same exemption question as the mypy and
  coverage carve-outs those modules already have.
- **#576** — `mypy warn_unreachable` would have caught a `continue` left
  behind after a `raise`, at the cost of seven suppressions elsewhere.
  Decide whether the trade is worth taking.
- **#723** — the rules in `.cursor/rules/` came from the `rms-devenv`
  repository template, which has moved on. Some of the differences contradict
  this repository's own configuration or its other rules, some are checks the
  template runs and this repository does not, and the rest is improved rule
  text; the issue carries the checklist.
- **#725** — `spindoctor/nav_records/descriptors.py` raises the open-file
  limit and clamps `retrieve_threads` and `retrieve_batch_size` because
  `FileCache.retrieve()` holds a lock file for every path in a batch until the
  last download finishes. Remove the module once filecache bounds its own lock
  batches (SETI/rms-filecache#67).
- **#443** — decide whether `spindoctor.cli` subpackages belong in the API
  reference at all. It is what decides whether the C-kernel writer package
  gets an autodoc page rather than the nitpick-ignore every other
  `spindoctor.cli` subpackage has, so it sequences before #129/#438.

## Track F — Instruments, features, hardening

### Instrument enablement (Phase-2 of the old roadmap)

Start after Track A's Cassini verdict; per instrument the pattern is:
fix ingest/navigation defects, add library frames (#235), extend the
calibration (#230), and update that instrument's two chapters under
`docs/user_guide/instruments/` and `docs/dev_guide/instruments/` in the
same change.

- **Voyager ISS:** #19 — star navigation broken; overlaps the per-camera
  Voyager distortion split (#355) and limiting-magnitude (#130) work, so
  schedule together. Rotation fitting is currently off for cost (#126).
- **Galileo SSI:** #18 — star navigation broken (same cluster).
- **New Horizons LORRI:** #2 — PSF sigma calibration; #138 — decide and
  enforce the `_eng` product policy; #33 (deferred) — new instrument
  kernel.
- **Ring models:** #82 Jupiter, #81 Uranus, #83 Neptune — extend the
  per-planet ring catalogs (`config_3N0_*_rings.yaml`) and the ring
  model's edge selection; Voyager/Galileo frames exist in the archive
  scan for all three.
- **#181** — image-degradation classifier classes; taxonomy design
  first, patterns are largely Voyager/Galileo-specific.

### Features

In rough priority order: #27 BOTSIM (NAC/WAC simultaneous), #22 star
streaks, #107 backplane-reader companion repo, #34 PDS4 input (when
external archives exist — replaces the PDS3 source per instrument; not
required for project completion), #184
cartographic/bootstrap navigation (crater-mapping correlation of
overlapping navigated frames — explicitly far off; design record in the
issue), #183 polarity-aware ring matching, #187 chaotic-rotator
(Hyperion) pose handling, #186 manual-nav dialog redesign, #185
gated-feature styling on summary PNGs, #182 stop-after-features
flag, #23 body shape models (topographic meshes; also feeds Track A's sim
realism), sim polish: #84 ring edges/gaps overwrite, #78 CraterMaker
craters, #151 flux-correct star smear, #152 diffraction spikes, #157
line-based missing data, #158 smooth-shaded meshes.

### Simulator scene authoring

An audit of what a scene file can say, and what the renderer does with it,
found three keys that render garbage and six that document something other
than what they mean. None of them affects a shipped scene's navigation
baseline, and all of them mislead the next person to author a scene.

- **Renders wrong:** `limb_relief_rms` carves radial wedges out of the disc
  at every value instead of roughening the limb (#625); `polyhedral_mesh`
  bodies render as banded shells, and the shipped Hyperion scene renders as a
  torus (#626); and the north and south haze terms seam the halo at the
  equator instead of blending across it (#627).
- **Says one thing and means another:** the artifact key `incidence` names
  three different quantities and collides with the illumination angle
  (#629); `cosmic_ray_rate_per_sec` is a fluence per cm^2 and two places
  document it per pixel (#630); a `vgiss` scene renders in I/F with a raw-DN
  marker and full well, matching neither documented unit class (#631); and
  the ring scene keys carry no units beside keys in the same block that do
  (#632).
- **Tooling:** the scene editor authors mesh body scenes its own validator
  refuses to load (#633).
- **#644** — a planted roll turns the scene's positions and poses but not a
  star's smear vector, a companion's position angle, or the background sky
  field. Filed out of the pivot audit (#642) and deliberately left out of it.

### Hardening / cleanup tail

Mostly small, any time, no ordering: #15 overlay
occlusion of background models, #21 metadata inventory cleanup, #38
filecache config, #39 AttrDict, #65 exception class, #92 dependency groups, #96 dead code, #97 oversized
modules, #98 registry consolidation, #99 orphan report_profile, #100
root-path getters, #101 ArgumentParser.error, #102 CLI globals, #103
thread-unsafe caches, #105 typed interop
boundaries, #109 safe-path helpers, #110 scalar validation helpers, #119 PNG
creation location, #135 from_file dedup, #137 dead validation
helper, #140 geometry-union access, #143 viewer cursor after pan, #144
QApplication lifetime, #147 confidence-context dedup, #155
display-scaling consolidation, #388 body-occluder bbox pre-filter (skip
the full `where_in_front` depth test for siblings whose predicted bbox
does not overlap, avoiding the O(bodies^2) grid pass in busy frames),
#212 xdist worker nondeterminism
(software-only scope; the faulty CPU cores are permanently offlined and
nothing gates on hardware), #428 the upstream `rms-pdslogger`
registry-eviction request (nothing in this repo depends on it; the
constraint is worth recording where the next person to reach for a
logger-per-image design will find it), #494 Cassini BOTSIM pairs defined two
different ways (one definition, before #27 builds on either), #518 state
the encoding wherever a document or text file is read or written, and #552
remove the `AttrDict` `_IS_IMMUTABLE` marker once oops confines its
mutability bookkeeping to its own objects, #554 the 82 test call sites that
wrap a `Path` in `str()` to build an `FCPath`, #590 the remaining
function-level imports that do not qualify for the GUI exemption, #604 a
second `load_default_and_user_config` in one process silently keeping the
first load, #612 `library_entry` resolving the PDS3 holdings root in a
different order from `DataSetPDS3` while its docstring says the two agree,
#711 hashing every static-data configuration file into a navigation's
provenance rather than the instrument configuration it actually used.

#552 is worth reading before touching `AttrDict` (#39), because it records
a live constraint rather than a wish: an `AttrDict` is its own instance
dictionary, so any attribute a library sets on one becomes a configuration
key. On the oops `mark-2026-02` branch, `oops.mutable._get_info` walks
everything reachable from an Observation — which reaches the shared `Config`
through the observation and so every section of it — and writes
`_MUTABLE_info` onto each. `validate_logging_config` then refused the
logging section and `sd_create_ck` and the `_cloud_tasks` drivers exited 1.
The marker is the opt-out that stops it; it is a workaround in another
package's private namespace, and #552 is the tracking issue for removing it.
