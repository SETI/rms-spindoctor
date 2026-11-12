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

## Phase 0: land the open stack (done)

| Issue or PR | Subject |
|---|---|
| #886 | **Done.** Calibrated Cassini noise floor: filter scale, readout dark, calibration residuals, and the measured Cassini 12-to-8-bit decode table |
| #887 | **Done.** Fitted cosmic-ray hits on the Cassini cameras and Galileo SSI |
| #888 | **Done.** Calibrated sweeps through the full instrument chain |
| #916 | **Done (PR #933).** Attribute the calibrated disc correlation's sub-pixel bias that #888 reports |

Merge #886, #887, and #888 in order and confirm that #329, #330, and #234 close. #916 follows
directly. The rest of the plan assumes this stack.

**Done:** the three PRs are on main, and #329, #330, and #234 are closed. #916's steady part came
from shading a smooth body at detector resolution and interpolating it onto the oversampled grid;
the simulator now shades a smooth body on the grid it renders it on. The larger error at three
offsets is the correlation refinement clamp, a navigator defect scheduled in Phase 1 as #929.

**Exit (met):** the three PRs are on main, and #916 names the source of the bias or the stage it
is confined to.

## Phase 1: foundations

These make every later phase cheaper or safer. Most are small.

| Issue | Subject | Why now |
|---|---|---|
| #914, framing part | **Done (PR #932).** Docs present self-consistency as accuracy | Stop the overclaim immediately: simulated results are reproducibility and self-consistency on the Cassini NAC, and confidence values are provisional. Docs only. |
| #924 | **Done (PR #932).** The simulator developer guide mixes per-camera values into the mechanism descriptions | Move each camera's measured values to its instrument page so each is stated once, before Phases 3 and 4 re-measure them and every change has to edit two copies. Docs only. |
| #911, including per-pixel variates | **Done (PR #962).** One frame-wide Poisson stream | Every later fix would otherwise turn into a new noise field across the whole frame. |
| #954 | A last-bit geometry change can reseed every crater on a simulated body | The same fragility as #911, in the crater field instead of the noise; fixing it moves every cratered render once, so it is cheapest alongside another regeneration. |
| #960 | Simulator tripwires depend on a single noise draw | The #911 redraw moved three of them; until each rate or band is measured over several noise seeds, the next redraw can move them again. |
| #961 | The simulator report quotes the sweeps' earlier noise draw | Regenerate the sweep figures and their prose together in the report's end-of-stack regeneration. |
| #905, truth-check part | Assertions are change detectors and 1 px bounds | Give validation scenes truth-based assertions before the renders start moving, so a move is judged against the planted truth. |
| #872 | **Done (PR #927).** The baseline regenerator imports whichever installed package it finds | Regenerated baselines must come from the checkout under test before the churn starts. |
| #900, #813 | **Done: #900 (PR #959); #813 (PR #927).** Validation skips the render path; null body keys crash; the navigator's simulated ring model reads nullable keys unsafely | Typed per-block specs on every entry point, with null meaning absent; the first slice of #915. |
| #940 | **Done (PR #959).** An in-memory scene reaches the renderer without validation | The one entry point that renders a scene validation would refuse; closing it with #900 makes every path validate. |
| #958 | The editor's tabs crash or misread a scene with null keys | The editor is the last entry point where a null key does not act as an absent one; the render path and the navigator side read nulls as absent since #900. |
| #913 | `axis1`, `axis2`, and `axis3` are full widths named like semi-axes | Rename before more scenes are written in the old names. |
| #876 | **Done (PR #934).** Per-instrument tables in one validated YAML file | Adding a camera or a readout mode becomes a configuration entry; Phase 4's instrument work depends on it. Lands after #886, as its text asks. |
| #912 | **Done (PR #946).** Correlation FFTs at slow sizes | The correlator runs at fast transform lengths: every library frame and simulated scene gives the same answer, and a 1024 px disc correlates 2.1 to 2.3 times faster. A lag whose overlap ties the floor can still change in contrived small geometries (#944). |
| #126, #945 | The disc correlation's rotation search runs 19 full pyramid evaluations; each candidate is refined on the whole surface at a slow length | Make full-frame scenes affordable in the integration tier; #904 and #908 both need it. Refinement is now most of the correlator's time. |
| #963, #964, #965, #966 | Mesh bodies are drawn one triangle at a time in Python; the distortion warp, the body placement shift, and the PSF blur each process the whole fine render grid | Mesh drawing is 40% of the catalog's render time. On a full-size calibrated frame the distortion warp is half the render and the shift and blur a quarter between them. The full-frame guard scenes and twins need renders that cost little. |
| #854 | Simulated stars clip below the navigator's saturation level | Small; star brightness is wrong until it lands. |
| #955 | Simulated A0 stars get B-V 0.63 instead of 0.00 | Small; every A0 star has the wrong color until it lands. |
| #925 | **Content match and star ladder done (PR #978); the star-count criterion stays open with #899 (record regenerated at the end of the stack).** The realism match compares mismatched content on content-filled and narrowband frames | **Done:** compare sky noise only on patches that lie on dark sky in both frames, chosen by level alone against a per-camera dark-sky ceiling (2.5e-3 I/F above the calibrated sky residual on both Cassini CALIB cameras, in the instrument config), and scale the matched star ladder by filter throughput, so the statistics the regression check holds to measure only the simulator. **Open with #899:** N1459552012 still predicts 25 simulated stars against the real frame's 4, because the navigator's star prediction ignores the filter. **Moved to #977:** whether the TABLE path is too quiet; on the 12 NAC TABLE frames compared, the simulated-to-real sky sigma ratio has a median of 0.37 (quartiles 0.25 to 0.59) against 0.88 on the 26 12-bit frames, and the real TABLE skies sit 4.5e-4 to 2.3e-3 I/F above the residual, which the twins don't reproduce. |
| #930 | **Done (PR #946).** Peak suppression and PSR ignore wrap-around | Disc and ring confidence rose about 0.08; no library frame's final offset, status, or tier moved. |
| #929 | The correlation refinement clamps at ±0.5 px from the integer peak | A navigator defect the simulator found; committed baselines record clamped answers, so it lands before the guard scenes and the Phase 3 re-ratchet. It is measured on real frames first, and its fix keeps the expected failures failing. |
| #941, #943, #944 | The PSR has no minimum background size; a NaN quality makes candidate selection order-dependent; the overlap floor flips a lag's validity on rounding ties | Since #930, a 4 px body navigates on a PSR from about five background cells and a NaN quality is reachable for small bodies; the smallest range-sweep body is an expected failure until #941 lands. #944 shows only in contrived small geometries. |
| #903, regression part | **Done; its bands arm at the end-of-stack regeneration of the realism record, the first record to carry a code commit.** The realism match cannot fail | **Done:** compare a fresh run of each instrument in the committed band table (LORRI) with the committed summary within two-sided per-statistic bands (the larger of 3 times the run-to-run spread over 10 runs and 2% of the value), so later physics changes are caught when they move realism either way; a second check requires that the band table describes the record. The summary records its code commit and whether the checkout was dirty, leaving the run's own `realism_*.png` figures and summary out of the dirty check; both are taken before the match, which stops when git cannot name the commit. The record-versus-table check is a strict expected failure, counting only a failed comparison, on a record without a code commit, which predates the check; the end-of-stack regeneration writes a record with its commit, which arms the check without editing the test, and the fresh match skips a record without a commit. Support counts frames on both sides, so a statistic with no simulated samples is unsupported. Quantized sky patches are measured by paired differences, read from their whole-step histogram, where that estimate is at least two steps and the patch is not clipped, and otherwise by a maximum-likelihood fit of the quantization (a quadratic sky surface plus rounded Gaussian noise, each pixel conditioned on an iterated window three fitted sigmas wide); the fit counts pixels cut off from the bulk of the levels by empty levels as censored and gives NaN when censored pixels exceed 0.27% by 1% of the patch plus three pixels. So a camera whose simulated sky noise is below one DN step contributes, texture is not read as noise once the noise spans two steps, and zero-filled blocks cut off from the sky make a tile unmeasurable; a contaminant at the patch's lowest or highest level that joins the bulk with no empty level between, within about three sigmas of the sky, is still fitted as a clip (two zero rows under a sky 3 sigma above zero read 9% high, five rows 16% low). On the 2026-11-11 frames, Galileo's matched frames give 435 simulated sky patches on 8 frames (none before), and Voyager's 730 on 3; the Cassini and LORRI figures are unchanged. Galileo: simulated 0.215 DN against real 0.960 DN (835 real patches, W1/IQR 7.9). Voyager: simulated 3.09e-3 against real 1.86e-3 I/F (620 real patches, W1/IQR 0.44); per frame, real against simulated, C1205021 6.86e-3 against 3.16e-3 (202 against 241 patches), C3446143 1.69e-3 against 3.09e-3 (210 against 241), and C3448723 1.71e-3 against 2.98e-3 (208 against 248). The Voyager comparison is not like for like: the real frames are resampled (GEOMED), which correlates neighboring pixels, so neighbor differences read their noise about a third low (on C3446143, 1.4e-3 I/F from adjacent pixels against 2.1e-3 from pixels two apart, and 2.2e-3 from the fit), while the simulator does not model the resampling; and C1205021 is filled by a cloud deck, so its darkest quarter of tiles is cloud, not sky. The hold-out and transfer parts come in Phase 6. |

**Exit:** changing the signal in one region of a frame leaves the noise elsewhere in that frame
unchanged; validation scenes assert against their planted truth; a null body key behaves exactly
as an absent one on every entry point; the realism match fails on a planted regression; adding a
camera's tables is a YAML edit (met by #876).

## Phase 2: rendering correctness

These are defects that put pixels in the wrong place. They come before any new physics, because
new physics on a wrong canvas is wasted work. #901 goes first: until the classic body renderer is
retired, every body fix has to be made twice.

| Issue | Subject |
|---|---|
| #901 | Classic and topographic body renderers disagree; retire the classic path (part of #915) |
| #931 | Coarse-grid shading shifts relief, textured, and cratered bodies at oversample above 1 (smooth bodies fixed by #916) |
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
| #942 | Disc and ring confidence was fitted on runner-up ratios that wrap-around depressed before #930; the refit covers them |
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
| #935 | The vidicon chain renders no fixed-pattern structure, so `vgiss` does not offer `fixed_pattern`; measure the residual pattern on calibrated frames before adding a stage |
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

This step prepares simulated results for the transfer check (Phase 6); accuracy claims wait for a
class to pass that check. #918 builds a twin from a library frame's real geometry, renders it with
the simulator, plants an offset, adds content the navigator does not model, and runs the production
pipeline unchanged. The `NavModel*Simulated` models stay for idealized scenes and sweeps; #897 and
#843 close for the real-geometry path when production models navigate its frames, and #311 guards
whatever mirrors remain. #843 also depends on #842 (Phase 2).

Prerequisites: #876 (Phase 1, done); #901, #890, #825, and #842 (Phase 2); #898 and #837 (Phase 3);
#895, #896, #909, and #910 (Phase 4a and 4b).

**Exit:** every library class has a twin generator; twins navigate through production code in the
integration tier; idealized scenes and every existing baseline are unchanged.

## Phase 6: prove the simulation predicts real error

| Issue | Subject |
|---|---|
| #903, hold-out and transfer parts | In-sample tuning, no hold-out, no transfer check (the censored noise comparison for Galileo and Voyager is done with the regression part) |
| #153, real-versus-simulated diagnostics part | Per-technique comparison of simulated and real diagnostics, run beside the transfer check |

The transfer check plants matched offsets on real frames and on their twins and compares simulated
error with real error per library class. A class passes only when both comparisons, with bands set
before the comparison, agree: the two error distributions, and the paired results frame by frame,
so that a twin fails where its real frame fails and its error tracks the real frame's. Matching
distributions alone never pass a class. Tuning uses frames held out from the comparison. #153's
bootstrap and calibration-validation parts follow the rest of #230 in Track R.

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
| #947 | The simulated-image editor shows 0 for parameters the renderer fills from the detector block or another key |
| #967 | The simulated-image editor's controls can build scenes that validation refuses: ranges wider than the rules, conflicting settings, settings left behind by a camera switch, and duplicate names |
| #956 | Unnamed simulated bodies get different default names in the renderer and the navigator |
| #856 | The simulated-image editor segfaults intermittently (reopened after the fix in PR #917) |
| #914, the rest | User chapter around a complete scene, developer guide split, and one schema reference |
| #310 | Structural information-boundary guard |

Each phase updates the docs for what it changes, with per-camera values on the instrument pages
after #924 (done); this phase finishes the restructuring and the guide split once the schema has settled.
#856 can land whenever its fix is ready.

## Track R: real-frame confidence (parallel)

| Issue | Subject |
|---|---|
| #334 | Adopt the calibration's falsification criteria and restore its real-frame regression check |
| #225 | Multi-object cross-technique agreement study on real frames |
| #342 | `star_psf_sigma` is a 3.0 px placeholder on Galileo, Voyager, and LORRI (navigator configuration, affects real frames today) |
| #230 | Real-anchored confidence recalibration |
| #153, bootstrap and calibration-validation parts | Starting guess for the confidence fit, and the sweep check of the fitted formulas |
| #788 | Reported offset sigma too large at the core and too small in the tail |

#334 is small and blocks #230, so it starts now, as do #225 and #342. #230 lists #225 and #227 as
prerequisites. This plan proposes splitting #230: its real-frame part (the tier boundaries and the
0.35 success threshold first, then RingEdgeNav confidence before edge routing widens further, then
the limb floor and coefficients, the terminator plateau, and the disc size term) proceeds once #334
is adopted and #225 supplies anchors; the part that needs single-technique accuracy waits for #227
(Phase 5) and the transfer check (Phase 6), and #153's bootstrap and calibration-validation parts
follow it. The F ring and spoke comparisons and the audit's evidence on #230 supply the real frames.
#788 waits on the ensemble fix for #767, which is in the frozen ring and star stack.

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
Phase 1 (#876, done) -> Phase 4d-4f, Phase 5
Track R: #334, #225, #342 now -> #230 real-frame part;
         #230 remainder, then #153's other parts, after Phase 6
Track O: needs #126 (Phase 1); joins Phase 5 for Voyager Uranus twins
Phase 8: alongside, finished last
```

## Risks

- **Baseline churn.** Phases 1 and 2 move most renders. Mitigation: #911 and #905 first, the
  guard scenes of Phase 2b before the physics, and one logical change per PR with its moved
  baselines listed.
- **Real-geometry cost.** #918 scenes need SPICE kernels and the holdings, so they run only in the
  integration tier. Mitigation: #912 (done) and #126 first, and a small per-class twin set in the
  integration tier with the full set run on demand.
- **The transfer check may fail for a class for reasons outside the simulator.** That is a result,
  not a setback: the class keeps its library frames and the gap is filed.
- **Scope.** Phases 4c through 4f are long. Work them by which library class is next in line, not
  in table order.
