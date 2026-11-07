# Simulator Improvement Plan

*The order in which to work the simulator issues raised by
`critiques/SIMULATOR_REALISM_CRITIQUE_2026-11-06.md`, with the reason for each step and the
condition that ends it.*

---

## Goal

The simulator serves two purposes:

1. **Sensitivity sweeps.** Measure how navigation responds along one configuration axis, such as
   noise, exposure, smear, cosmic-ray rate, or a defect mode. These scenes stay simple on purpose.
2. **Accuracy validation.** Measure navigation accuracy and find failures on scenes that look like
   real frames: simple and realistic scenes, with no, single, or multiple features, interleaved
   instrument defects, compound content, real frame sizes and offsets, and every camera.

The goal of the second purpose is that simulated scenes, done properly, stand in for most of the
real image library.

**Terms.**

- The *image library* is the curated set of real frames with operator-verified pointing that the
  library regression suite navigates. A *library class* is one of its groups by instrument and
  content (for example Cassini NAC star fields, or WAC ring frames).
- The *realism match* renders a simulated frame for each library frame, with the same instrument
  chain, exposure, and class of content, and compares image statistics between the two.
- A *twin* is a simulated frame built from a specific real frame's geometry (Phase 5), so that it
  shows the same bodies, rings, and stars where the real frame shows them.

A library class is retired from the regression suite when its class passes the transfer check
(Phase 6) and the validation catalog covers it (Phase 7).

**What must keep working.** Everything the simulator does for the first purpose stays supported
throughout: idealized scenes in pixel units with no real image behind them, the
`NavModel*Simulated` models, the sweeps, and the scene generators in `util/calibration`,
`util/agreement`, and `util/titan_truth`. Where a change replaces an idealized mechanism, the
idealized form stays available beside the physical one: #909 keeps a pixel-unit ring mode beside
kilometer radii, and #896 keeps an authored shadow beside the geometric one.

## Principles

- **Fix what makes scenes wrong before writing scenes that depend on it.** A baseline committed on
  a broken renderer has to be rewritten when the renderer is fixed, and until then it certifies
  the wrong answer.
- **Make the tests robust before the churn.** Nearly every fix in this plan moves renders and
  baselines. The noise stream and the assertions are fixed first, so each later change moves only
  what it should, and every moved baseline is listed and explained in its PR.
- **Real frames decide.** No simulated number is called accuracy until the transfer check
  (Phase 6) shows that simulated error predicts real error for that class of frame.
- **Production values come from real frames where real frames exist.** The real-frame part of the
  confidence work runs in parallel with all of this (Track R).
- **One fact, one issue.** Each phase names the issues it closes; anything found along the way is
  filed, not folded silently into a PR.

## Phase 0: land the open stack

| Issue or PR | Subject |
|---|---|
| #886 | Calibrated Cassini noise floor: filter scale, readout dark, calibration residuals, and the measured Cassini 12-to-8-bit decode table |
| #887 | Fitted cosmic-ray hits on the Cassini cameras and Galileo SSI |
| #888 | Calibrated sweeps through the full instrument chain |
| #916 | Attribute the calibrated disc correlation's sub-pixel bias that #888 reports |

Merge #886, #887, and #888 in order and confirm that #329, #330, and #234 close. #916 follows
directly. The rest of the plan assumes this stack.

**Exit:** the three PRs are on main, and #916 names the source of the bias or the stage it is
confined to.

## Phase 1: foundations

These make every later phase cheaper or safer. Most are small.

| Issue | Subject | Why now |
|---|---|---|
| #914, framing part | Docs present self-consistency as accuracy | Stop the overclaim immediately: simulated results are reproducibility and self-consistency on the Cassini NAC, and confidence values are provisional. Docs only. |
| #911, including per-pixel variates | One frame-wide Poisson stream | Every later fix would otherwise turn into a new noise field across the whole frame. |
| #905, truth-check part | Assertions are change detectors and 1 px bounds | Give validation scenes truth-based assertions before the renders start moving, so a move is judged against the planted truth. |
| #872 | The baseline regenerator imports whichever installed package it finds | Regenerated baselines must come from the checkout under test before the churn starts. |
| #900, #813 | Validation skips the render path; null body keys crash; the navigator's simulated ring model reads nullable keys unsafely | Typed per-block specs on every entry point, with null meaning absent; the first slice of #915. |
| #913 | `axis1`, `axis2`, and `axis3` are full widths named like semi-axes | Rename before more scenes are written in the old names. |
| #876 | Per-instrument tables in one validated YAML file | Adding a camera or a readout mode becomes a configuration entry; Phase 4's instrument work depends on it. Lands after #886, as its text asks. |
| #912, #126 | Correlation FFTs at slow sizes; the disc correlation's rotation search runs 19 full pyramid evaluations | Make full-frame scenes affordable in the integration tier; #904 and #908 both need it. |
| #854 | Simulated stars clip below the navigator's saturation level | Small; star brightness is wrong until it lands. |
| #903, regression part | The realism match cannot fail | Compare a fresh run with the committed summary within per-statistic bands, so later physics changes are caught when they make realism worse. The hold-out and transfer parts come in Phase 6. |

**Exit:** changing the signal in one region of a frame leaves the noise elsewhere in that frame
unchanged; validation scenes assert against their planted truth; a null body key behaves exactly
as an absent one on every entry point; the realism match fails on a planted regression; adding a
camera's tables is a YAML edit.

## Phase 2: rendering correctness

These are defects that put pixels in the wrong place. They come before any new physics, because
new physics on a wrong canvas is wasted work. #901 goes first: until the classic body renderer is
retired, every body fix has to be made twice.

| Issue | Subject |
|---|---|
| #901 | Classic and topographic body renderers disagree; retire the classic path (part of #915) |
| #890 | A body larger than half the frame and off center is clipped or vanishes |
| #825, #644 | A planted roll turns ellipsoids the wrong way, mesh bodies and the lit side not at all, and leaves smear, companions, and sky unturned; #644 also asks for a ruling on what a roll turns |
| #891, #873 | Ring-edge anti-aliasing on optical depth shifts lone edges up to 0.7 px; a rank-1 ring scene misses its planted shift by 0.52 px |
| #892 | PSF and smear zero-pad the frame: darkened edges, no light from outside it |
| #893 | Limb seam, coverage-blind star extinction, and `rotation_tilt` stretch |
| #894 | Night-side 1% floor and clipped photometric laws |
| #875 | Relief reads disc terrain at the smooth sphere's direction |
| #842, #646 | Simulated limbs are not checked for where the brightness gradient peaks; the limb-bias harness measures a simulated ridge half a pixel from the deployed one |

**Exit:** a frame-filling planet renders at its true position with light from beyond the frame
edge; a rolled scene behaves as the ruling on #644 decides; a lone ring edge lands within 0.05 px
of its authored radius; the simulated limb's gradient peak is checked against real limbs; every
moved baseline is explained.

## Phase 2b: realistic guard scenes

Once pixels land where they should, write the validation scenes that YAML can express today, so
they guard the churn of Phases 3 and 4. These are the authorable parts of #904 and the ring-only
part of #377:

- every camera the simulator catalogs, at real frame sizes and real offsets, with twist;
- star fields with a body, and rings with a body;
- dense ring profiles at real scale and low opening, with only catalog edges navigable and
  unmodeled structure beside them;
- interleaved defect modes on one frame.

Each scene asserts against its planted truth (#905). Scenes that expose a navigator failure are
committed as expected failures with the issue they reproduce.

**Exit:** the guard set runs in the integration tier, and each scene's result is either a pass
against truth or an expected failure tied to an issue.

## Phase 3: repair what the simulated confidence values were fitted on

Several production confidence values were fitted on one simulated campaign, seed 20260718, whose
generator drew the bodies of four families at half size and fails on four others. Repair the
generator here, so the comparison in Track R has a correct simulated baseline and so Phase 5's
scenes are not built on the same slip.

| Issue | Subject |
|---|---|
| #837 | The calibration generator fails on four families |
| #898 | The calibration generator draws catalog bodies at half size |
| #309 | The campaign does not run the realism-anchored configuration (chain off, NAC only) |
| #793, #874 | Refit on the repaired renderers and generator, under the overflow limits, and re-ratchet the moved coefficients |
| #341 | The campaign's scene mixture is authored, with no data that it matches real frames |

Track R replaces simulation-fitted values with real-anchored ones wherever real anchors exist;
this phase keeps the remaining simulation-fitted values correct until then.

**Exit:** the campaign regenerates on main with bodies at their catalog size and the full
instrument chain on, the refit coefficients are re-ratcheted as #793 and #874 require, and each
move is recorded in `plans/VALIDATION_AND_CALIBRATION_PLAN.md`.

## Phase 4: physics, in order of how many real frames need it

Ordered by the share of real frames each gap affects, from the critique's population counts.

**4a. Radiometry and shadows (every frame with more than one object type):**

| Issue | Subject |
|---|---|
| #895 | Stars, bodies, and rings share no radiometric scale |
| #896 | No ring-body shadows; the planet's shadow is a fixed wedge |
| #921 | Background star counts 3 to 10 times too sparse |
| #899 | Navigator star limit: 30.0 on calibrated frames, filter ignored on raw ones (the Cassini part here; the Voyager part follows #907, and the Galileo, Voyager, and LORRI limits follow #342) |

**4b. Ring geometry and structure (the largest real population: 177,540 Cassini ring frames):**

| Issue | Subject |
|---|---|
| #909 | Rings in pixels, orthographic, no thickness; add kilometer radii, perspective, and the catalog, keeping the pixel-unit mode |
| #910 | Ring envelopes, arcs, per-feature azimuthal structure, and withheld shapes past mode 1 |

Features the navigator does not model are rendered on the image side only; the navigator's ring
model does not gain them.

**4c. Objects the navigator does not know about:**

| Issue | Subject |
|---|---|
| #922 | A body the navigator does not know about cannot be planted; no sub-pixel moons |

**4d. Detector, readout, and optics per camera:**

| Issue | Subject |
|---|---|
| #906 | Hot pixels ignore exposure; bloom discards charge; LORRI bloom contradiction |
| #920 | WAC and Galileo gain states, Galileo ICT, and LORRI frame-transfer smear and calibrated mode |
| #919 | Ghosts translate with the scene; dust donuts move between scenes |
| #850 | The DCT codec rebuilds values outside the ADC word |
| #848, #343 | The LORRI PSF wing is too weak; the tuned NAC PSF wing may absorb registration error |
| #359 | PSF error forms the simulator cannot render, for the agreement estimator |
| #877, simulator part | Windowed frames |
| #333 | Four physical error axes are unmodeled |
| #151 | Flux-correct star smear and smear authoring (deferred until the upstream PSF model supports it) |

**4e. Voyager:**

| Issue | Subject |
|---|---|
| #806 | One `vgiss` key for four cameras; split per spacecraft and camera |
| #907 | GEOMED plate scale, beam-bend sign, reseau lattice, and a fixed DN scale with no exposure term |
| #804 | Noise keys inert on the vidicon and on calibrated frames |
| #355 | Distortion and twist per camera once the split keys exist (deferred until star navigation locks enough frames per camera) |

**4f. Filters beyond Cassini throughput:**

| Issue | Subject |
|---|---|
| #885 | Galileo and Voyager filter tables |
| #884, #883, #882, #881 | Filter dependence of body photometry, PSF width, star color, and haze |

4a and 4b come first because Phase 5 needs them. 4c through 4f can run in parallel with Phase 5
once 4a and 4b land; within them, work the camera whose library classes are next in line for
retirement.

**Exit:** for each item, the realism match moves toward the real frames, or the issue records why
it cannot be measured there.

## Phase 5: render from real geometry and navigate with production models

| Issue | Subject |
|---|---|
| #918 | Render simulated frames from a real frame's geometry and navigate them with production models |
| #897, #843 | Production ring, star, Titan, and body models never run on simulated frames |
| #311 | Parity guard between the simulated models and the production models |
| #227 | Umbrella: de-circularize the simulator |

This is the step that makes simulated results mean accuracy. #918 builds a twin from a library
frame's real geometry, renders it with the simulator, plants an offset, adds content the
navigator does not model, and runs the production pipeline unchanged. The `NavModel*Simulated`
models stay for idealized scenes and sweeps; #897 and #843 close for the real-geometry path when
production models navigate its frames, and #311 guards whatever mirrors remain. #843 also depends
on #842 (Phase 2).

Prerequisites: #876 (Phase 1); #901, #890, #825, and #842 (Phase 2); #898 and #837 (Phase 3);
#895, #896, #909, and #910 (Phase 4a and 4b).

**Exit:** every library class has a twin generator; twins navigate through production code in the
integration tier; idealized scenes and every existing baseline are unchanged.

## Phase 6: prove the simulation predicts real error

| Issue | Subject |
|---|---|
| #903, hold-out and transfer parts | In-sample tuning, no hold-out, censored noise comparison for Galileo and Voyager, no transfer check |
| #153 | Calibration test layer: bootstrap and real-versus-simulated diagnostics (follows #230) |

The transfer check plants matched offsets on real frames and on their twins and compares simulated
error with real error per library class. A class passes when the two error distributions agree
within bands set before the comparison. Tuning uses frames held out from the comparison.

**Exit:** a per-class table of simulated against real error, with the classes that pass named.
Those classes are eligible for retirement in Phase 7.

## Phase 7: the validation catalog, and retiring library classes

| Issue | Subject |
|---|---|
| #904 | Validation scenes cover only small, clean, single-family Cassini NAC frames (the rest, beyond Phase 2b) |
| #905 | Per-instrument accuracy numbers (the rest, beyond Phase 1) |
| #377 | Dense, realistic ring scenes (the rest, beyond Phase 2b) |
| #835 | Star navigation reports success at 4.9 px error under 6 px catalog scatter |

Build the validation catalog from twins and from idealized scenes that the transfer check
supports: every camera, real frame sizes and offsets, twist, compound content, frame-filling
bodies, interleaved defects, and unmodeled ring structure and bodies.

The catalog reproduces the navigator failures already filed and guards them against regression:
#755 (along-edge ring lock), #346 (wrong-feature ring locks), and #434 (rotation about the field of
view's center). It also repeats two measurements on twins, where truth is known: #322 (correlated
limb errors across bodies) and #223 (the terminator's realism).

A library class is retired from the regression suite when it passed Phase 6 and the catalog covers
it. Library frames stay as the reference for classes that have not passed.

**Exit:** a published per-instrument accuracy number for each class that passed, and a list of
retired and retained library classes.

## Phase 8: code and docs

| Issue | Subject |
|---|---|
| #915 | Restructure below the core: typed specs (begun in #900), one body path (begun in #901), and smaller stages |
| #923 | Duplicate classifier, test-only module, missing fixtures, and codenames |
| #902 | 26 scene keys without editor controls |
| #856 | The simulated-image editor segfaults intermittently (fix open as PR #917) |
| #914, the rest | User chapter around a complete scene, developer guide split, and one schema reference |
| #310 | Structural information-boundary guard |

Each phase updates the docs for what it changes; this phase finishes the restructuring and the
guide split once the schema has settled. #856 can land whenever its PR is ready.

## Track R: real-frame confidence (parallel)

| Issue | Subject |
|---|---|
| #334 | Adopt the calibration's falsification criteria and restore its real-frame regression check |
| #225 | Multi-object cross-technique agreement study on real frames |
| #342 | `star_psf_sigma` is a 3.0 px placeholder on Galileo, Voyager, and LORRI (navigator configuration, affects real frames today) |
| #230 | Real-anchored confidence recalibration |
| #788 | Reported offset sigma too large at the core and too small in the tail |

#334 is small and blocks #230, so it starts now, as do #225 and #342. #230 lists #225 and #227 as
prerequisites. This plan proposes splitting #230: its real-frame part (the tier boundaries and the
0.35 success threshold first, then RingEdgeNav confidence before edge routing widens further, then
the limb floor and coefficients, the terminator plateau, and the disc size term) proceeds once
#334 is adopted and #225 supplies anchors; the part that needs single-technique accuracy waits
for #227 (Phase 6). The F ring and spoke comparisons and the audit's evidence on #230 supply the
real frames. #788 waits on the ensemble fix for #767, which is in the frozen ring and star stack.

## Track O: outer-planet rings (parallel, navigator work)

| Issue | Subject |
|---|---|
| #908 | A ring feature type narrower than a pixel |
| #81, #82, #83 | Ring models for Uranus, Jupiter, and Neptune |

Production cannot navigate these rings yet, so the simulator cannot validate them. #908 also needs
#126 (Phase 1). Once #908 and #81 land, add Voyager Uranus twins under Phase 5, which needs #907's
exposure-scaled DN.

## Out of scope

- #152, diffraction spikes: the issue marks itself lowest priority, and no navigation result
  depends on it yet.

## Dependencies at a glance

```text
Phase 0 -> Phase 1 -> Phase 2 -> Phase 2b
                        |
                        +-> Phase 3 ----------------+
                        +-> Phase 4a, 4b -----------+-> Phase 5 -> Phase 6 -> Phase 7
                        +-> Phase 4c-4f (parallel with Phase 5)
Phase 1 (#876) -> Phase 4d-4f, Phase 5
Track R: #334, #225, #342 now -> #230 real-frame part; #230 remainder after Phase 6
Track O: needs #126 (Phase 1); joins Phase 5 for Voyager Uranus twins
Phase 8: alongside, finished last
```

## Risks

- **Baseline churn.** Phases 1 and 2 move most renders. Mitigation: #911 and #905 first, the
  guard scenes of Phase 2b before the physics, and one logical change per PR with its moved
  baselines listed.
- **Real-geometry cost.** #918 scenes need SPICE kernels and the holdings, so they run only in the
  integration tier. Mitigation: #912 and #126 first, and a small per-class twin set in the
  integration tier with the full set run on demand.
- **The transfer check may fail for a class for reasons outside the simulator.** That is a result,
  not a setback: the class keeps its library frames and the gap is filed.
- **Scope.** Phases 4c through 4f are long. Work them by which library class is next in line, not
  in table order.
