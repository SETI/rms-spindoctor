# Simulator Improvement Plan

*The order in which to work the simulator issues raised by
`critiques/SIMULATOR_REALISM_CRITIQUE_2026-11-06.md` and
`critiques/SIMULATOR_CODE_STRUCTURE_CRITIQUE_2026-11-16.md`, with the reason for each step and the
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
real image library. That means synthetic frames for every supported camera, as realistic as they
need to be to check navigation accuracy, and no more.

The code that produces them should look designed rather than accreted:
- one declaration for each scene key
- one implementation of each effect
- one path for each body type
- modules split by concern, never at the line limit
- no special cases, compatibility layers, or tests of behavior that no longer exists
- documentation that describes the code as it is

Phases 1b, 2 (#1011, #1012), 2a, 3b, and 8 carry that work. The other phases follow it as they go.

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
- **Restructure where it is cheapest and safest.**
  - A refactor that should move no pixel lands while the baselines are still, so the baselines prove
    it.
  - A change to the scene format lands before scenes are written in it.
  - A change to the renderer's shape lands before the physics that would otherwise be written into
    the old shape.

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
#929 changed only the ring annulus correlation; the disc correlation keeps the clamp (#791), so
this calibrated-disc error remains.

**Exit (met):** the three PRs are on main, and #916 names the source of the bias or the stage it
is confined to.

## Phase 1: foundations

These make every later phase cheaper or safer. Most are small.

**Status (2026-11-18):** every row below is done, decided, or moved, except #126 and #945, and
#854 (on hold). #960 is done (PR #1067). The realism-record regeneration with #961 is done (PR #1034), and #925's
star-count criterion moved to #899 and #895 in Phase 4a. #970 waits on the upstream camera-model
fixes.

| Issue | Subject | Why now |
|---|---|---|
| #914, framing part | **Done (PR #932).** Docs present self-consistency as accuracy | Stop the overclaim immediately: simulated results are reproducibility and self-consistency on the Cassini NAC, and confidence values are provisional. Docs only. |
| #924 | **Done (PR #932).** The simulator developer guide mixes per-camera values into the mechanism descriptions | Move each camera's measured values to its instrument page so each is stated once, before Phases 3 and 4 re-measure them and every change has to edit two copies. Docs only. |
| #911, including per-pixel variates | **Done (PR #962).** One frame-wide Poisson stream | Every later fix would otherwise turn into a new noise field across the whole frame. |
| #954 | **Done (PR #998): each crater draws from its own stream, and its center from a search that does not depend on the body's exact shape.** A last-bit geometry change can reseed every crater on a simulated body | The same fragility as #911, in the crater field instead of the noise; fixing it moves every cratered render once, so it is cheapest alongside another regeneration. |
| #960 | **Done (PR #1067): the optional scene key `noise_seed` redraws only the detector's per-frame noise; the rate and band checks are measured over noise seeds, and every other single-draw check records its measured spread.** Simulator tripwires depend on a single noise draw | The #911 redraw moved three of them; until each rate or band is measured over several noise seeds, the next redraw can move them again. |
| #961 | **Done (PR #1034): the two sweep figures and the irregularity and pose-disagreement passages were regenerated from one run on 2026-11-17, the population claims included.** The simulator report quotes the sweeps' earlier noise draw | Regenerate the sweep figures and their prose together in the report's end-of-stack regeneration. |
| #905, truth-check part | **Done (PR #976).** Assertions are change detectors and 1 px bounds | Give validation scenes truth-based assertions before the renders start moving, so a move is judged against the planted truth. |
| #872 | **Done (PR #927).** The baseline regenerator imports whichever installed package it finds | Regenerated baselines must come from the checkout under test before the churn starts. |
| #900, #813 | **Done: #900 (PR #959); #813 (PR #927).** Validation skips the render path; null body keys crash; the navigator's simulated ring model reads nullable keys unsafely | Typed per-block specs on every entry point, with null meaning absent; the first slice of #915. |
| #940 | **Done (PR #959).** An in-memory scene reaches the renderer without validation | The one entry point that renders a scene validation would refuse; closing it with #900 makes every path validate. |
| #958 | **Done (PR #973).** The editor's tabs crash or misread a scene with null keys | The editor is the last entry point where a null key does not act as an absent one; the render path and the navigator side read nulls as absent since #900. |
| #913 | **Done (PR #992): the keys are now `semi_a_px`, `semi_b_px`, and `semi_c_px`, semi-axes in pixels.** `axis1`, `axis2`, and `axis3` are full widths named like semi-axes | Rename before more scenes are written in the old names. |
| #876 | **Done (PR #934).** Per-instrument tables in one validated YAML file | Adding a camera or a readout mode becomes a configuration entry; Phase 4's instrument work depends on it. Lands after #886, as its text asks. |
| #912 | **Done (PR #946).** Correlation FFTs at slow sizes | The correlator runs at fast transform lengths: every library frame and simulated scene gives the same answer, and a 1024 px disc correlates 2.1 to 2.3 times faster. A lag whose overlap tied the floor could change with FFT rounding in contrived small geometries; #944 (done, PR #999) counts the overlap exactly. |
| #126, #945 | The disc correlation's rotation search runs 19 full pyramid evaluations; each candidate is refined on the whole surface at a slow length | Make full-frame scenes affordable in the integration tier; #904 and #908 both need it. Refinement is now most of the correlator's time. |
| #963, #964, #965, #966 | #964 **done (PR #968).** #965 and #966 **done (PR #971).** #963 **done (PR #991).** Mesh bodies are drawn one triangle at a time in Python; the distortion warp, the body placement shift, and the PSF blur each process the whole fine render grid | Mesh drawing is 40% of the catalog's render time. On a full-size calibrated frame the distortion warp was half the render; warping only the lit clusters took that frame from 3.2 s to 1.8 s, and shifting and blurring only the lit parts took it to 1.1 s. Drawing the mesh triangles in batches, with byte-identical output, took the six catalog mesh renders from about 1.3 s to 0.15 s and a full-frame mesh body from about 5.5 s to 3 s. The full-frame guard scenes and twins need renders that cost little. |
| #969 | **Done (PR #985).** The scene editor renders the whole pipeline on every edit, freezing for a second or more on a full-size frame | Editing should feel immediate: a fast approximate preview while the user edits, replaced by the full render once they pause. |
| #970 | **Decided: fix the camera models instead (SETI/rms-host-galileo#3, SETI/rms-host-cassini#4, SETI/rms-host-new-horizons#1, SETI/rms-host-voyager#2).** Decide whether to keep the simulated residual distortion for cameras where it is tiny (Cassini NAC and WAC at most 0.05 px, LORRI 0.2 px, Galileo 0.8 px, in the corners) | Measured on full-frame scenes, the residual moves fused answers by 0.002 px (WAC) to 0.09 px (Galileo SSI), as its coefficients predict, and changes no status or tier. Each camera keeps its simulated residual until its updated model is in the host SpinDoctor uses; then the star-field campaign is rerun and that camera's residual is removed or replaced, regenerating the scenes that use camera defaults. |
| #984 | **Decided: not implemented; documented (PR #985).** A simulated body moved by a fraction of a pixel has a softer limb than one drawn there | Measured: drawing at the exact sub-pixel position changes no technique's offset error by more than 0.025 px and no confidence by more than 0.003, at or below the noise scatter, so the cached drawing and interpolated move stay and the developer guide explains them. Original rationale: In scenes without a PSF the limb's sharpness changes with the planted fractional offset, which no real image does. Measure the effect on navigation first; then draw each body at its true sub-pixel position in every render that leaves the editor, keeping the cached drawing and interpolated move only for dragging in the editor (#969). Fixing the canvas also fixes #890. |
| #854 | **On hold (operator).** Simulated stars clip below the navigator's saturation level | Star brightness is wrong until it lands. Not started unless the operator lifts the hold. |
| #955 | **Done (PR #972).** Simulated A0 stars get B-V 0.63 instead of 0.00 | Small; every A0 star has the wrong color until it lands. |
| #925 | **Done (PR #1034; operator, 2026-11-18). Content match and star ladder done (PR #978). Star twins built from catalog stars with #989. The realism record was regenerated at the end of the stack (PR #1034, 2026-11-17, commit ea3c5442). The star-count criterion moved to #899 and #895, with its target corrected to the real frames' 25 NAC star cutouts (the earlier 79 counted the made-up ladder).** The realism match compares mismatched content on content-filled and narrowband frames | **Done:** on a camera with a dark-sky ceiling, sky noise is compared only on patches that lie on dark sky in both frames. A patch lies on dark sky when its mean is at most a per-camera dark-sky ceiling above the calibrated sky residual, and the test uses the level alone. The ceiling is 2.5e-3 I/F on both Cassini CALIB cameras and is set in the instrument config. A camera without a ceiling compares every sky patch. On a filtered frame, the matched star ladder starts from brighter stars by the filter's throughput, so the renderer's filter dimming leaves them detectable (#989 replaces the ladder with catalog stars). With both changes, the statistics the regression check holds for the cameras with a ceiling measure only the simulator. **Moved to #899 and #895 (star count):** N1459552012's twin showed 25 stars against the real frame's 4, because its made-up ladder held 25 stars bright enough to see through the filter. With #989 a star twin holds the catalog stars of the real field: N1459552012's twin plants the field's 50 catalog stars down to its clear-filter cutoff, of which 3 are bright enough to show through HAL+CL2 (at least 5 times the noise over the star's core), and it gives 2 star cutouts against the real frame's 3. Two things kept the criterion open. On the twin, the navigator's star-field fit ranks 30 predicted stars (`n_catalog_predicted`, capped at the 30 it takes) against the real frame's 4, because the calibrated simulated frame's star limit is a placeholder (#899). Over the 13 NAC star frames the twins give 9 star cutouts against 25 real (73 with the ladder), because the simulated NAC stars are about 3.5 mag fainter than real ones at the same catalog magnitude (#895). The regenerated record (2026-11-17) confirms the pooled counts: 9 simulated NAC star cutouts from 6 frames against 25 real from 11, and on N1459552012's twin the star-field fit finds 2 inliers among its 30 and doesn't lock. The record also reports the sky comparison's frame counts for each Cassini camera, which a CLI run had written as zeros until this regeneration fixed it: on the NAC, 26 of 32 12-bit and 12 of 26 TABLE frames are compared; on the WAC, 3 of 4. The binned LORRI twins hold all 311 of their stars, and with #993 the navigator on a binned twin keeps the stars it shows: the LORRI twins give 16 star cutouts against the real frames' 16. The navigator's star prediction also ignores the filter (#899), so the star SNR and brightness-margin fields can't be compared on filtered frames until it is fixed. **Moved to #977:** whether the TABLE path is too quiet. On the 12 NAC TABLE frames compared, the simulated-to-real sky sigma ratio has a median of 0.37 (quartiles 0.25 to 0.59), against 0.88 on the 26 12-bit frames. The real TABLE skies also sit 4.5e-4 to 2.3e-3 I/F above the residual, which the twins don't reproduce. |
| #989 | **Done (PR #1004).** Realism star twins hold a made-up magnitude ladder, not the real frame's stars | Each star twin holds every catalog star in the real frame down to a clear-filter cutoff, at the real frame's pointing moved by its verified offset, and the ladder shift by filter is gone. The filter then decides how many of the twin's stars show, as it does in the real frame. The twins do not close #925's star-count criterion, which moved to #899 and #895. |
| #993 | **Done (PR #996).** The simulated frame's faint-star limit ignores readout binning | **Done:** the navigator's view of a simulated scene states its readout binning, which a real frame's label states, and no other detector key. On a binned frame the star limit reads the zero point and read noise from the navigator block's `noise.by_binning` entry for the mode, which each raw-DN camera's block carries for every binned mode the simulator offers, from the camera's published gain and read noise; a mode without an entry keeps the unbinned values. Scene validation checks every value of a scene's `instrument_config` that the limit reads, unbinned and binned: a read noise is a finite number of at least 0, a zero point and `star_psf_sigma` are finite and positive, none of them is null, and each `by_binning` entry sets nothing but the two values. On a calibrated-unit camera, whose limit reads no noise value, validation refuses both the table and the unbinned values as never read. The configuration loader doesn't check the packaged table, which tests hold to the simulator's tables. Validation also refuses a binned scene that pins an unbinned noise value without pinning the mode's value too. The limit divides `star_psf_sigma` by the binning and counts the core as at least one pixel. It reads none of the simulator's camera tables. A 4x4-binned LORRI frame's limit moves from 11.41 to 15.83 at 5 s; an unbinned frame's limit is unchanged. A scene's own `optics.psf`, `match_navigator` included, is not shrunk on a binned frame, so such a scene renders a wider core than the limit assumes; that, and the navigator's other star values on a binned frame, are #995 (Phase 4d). #989's LORRI smoke test needs it. |
| #930 | **Done (PR #946).** Peak suppression and PSR ignore wrap-around | Disc and ring confidence rose about 0.08; no library frame's final offset, status, or tier moved. |
| #941, #943, #944 | **#943 done (PR #974). #944 done (PR #999). #941 moved to #585 (operator, 2026-11-15): the minimum background size is settled with the background region.** The PSR has no minimum background size; a NaN quality makes candidate selection order-dependent; the overlap floor flips a lag's validity on rounding ties | Since #930, a 4 px body navigates on a PSR from about five background cells and a NaN quality is reachable for small bodies; the smallest range-sweep body is an expected failure until #585 settles the background, #941 included. #944 showed only in contrived small geometries; a constant brightness offset that empties the variance floor is #997. |
| #929 | **Done for the ring annulus correlation (PR #1003): its final full-resolution pass refines on a 3x3 quadratic fit of the NCC surface, and the coarser pyramid levels keep the cross-power refinement. The disc correlation is unchanged by decision: no refinement measured beat the cross-power one against strong star solutions on real disc frames; its clamp stays open as #791. An answer the fit cannot refine stays on the whole pixel with nothing to mark it (#1002). No library frame's final offset, status, or tier moved, and no simulated scene's or sweep step's final answer moved; the annulus's own answer moved on the six catalog scenes that run it.** The correlation refinement clamps at ±0.5 px from the integer peak | A navigator defect the simulator found; committed baselines record clamped answers, so it lands before the guard scenes and the Phase 3 re-ratchet. It is measured on real frames first, and its fix keeps the expected failures failing. |
| #903, regression part | **Done (PR #980); its bands armed at the end-of-stack regeneration of the realism record (PR #1034, 2026-11-17), the first record to carry a code commit. The LORRI band table was remeasured on the regenerated code: all 19 statistics repeat exactly over 10 runs, so every band is still the 2% floor, and both checks run and pass.** The realism match cannot fail | **Done:** a fresh run of each instrument in the committed band table (LORRI) is compared with the committed summary. Each statistic is held to a two-sided band: the larger of 3 times its run-to-run spread over 10 runs and 2% of its value. So later physics changes are caught when they move realism either way. A second check requires that the band table describes the record. The summary records its code commit and whether the checkout was dirty; the run's own `realism_*.png` figures and summary don't count toward the dirty flag. Both are read before the match, which stops when git cannot tell the commit. On a record without a code commit, which predates the check, the record-versus-table check is a strict expected failure that counts only a failed comparison, and the fresh match skips. The end-of-stack regeneration writes a record with its commit, which arms both checks without editing the tests. Support counts frames on both sides, so a statistic that lacks simulated samples is unsupported. Quantized sky patches are measured by paired differences, read from their whole-step histogram, when that estimate is at least two steps and the patch is not clipped. Otherwise they are measured by a maximum-likelihood fit of the quantization: a quadratic sky surface plus rounded Gaussian noise, each pixel conditioned on an iterated window three fitted sigmas wide. Pixels beyond their windows, or cut off from the bulk of the levels by empty levels, count as censored. The fit gives NaN when censored pixels exceed 0.27% by more than 1% of the patch plus three pixels. It also gives NaN when the bulk's lowest or highest level holds more pixels than the fitted model predicts by more than four times the prediction's square root plus 0.27% of the patch and three pixels. An unclipped patch whose differences read below two steps and whose fit reads 2.5 steps or more gives NaN too. As a result, a camera whose simulated sky noise is below one DN step contributes sky samples, texture is not read as noise once the noise spans two steps, and zero-filled blocks cut off from the sky make a tile unmeasurable. Dropout rows that pile onto the floor level of a floored sky also make a tile unmeasurable (399 of 400 test patches under a sky at 9 DN of 1.5 to 3.5 DN, where a fit that followed them read 1.2 to 2 times the true sigma). So do the zero-filled border tiles of resampled Voyager frames (33 of the 37 fitted sky tiles on C3446143 and C3448723). The levels between the end levels are not checked, because real Galileo sky puts about 1.4 times the predicted pixels on its fullest level. Dropout rows inside the sky's levels, where the sky's noise reaches below zero, are measured by the differences and read low: about 8% for two rows and 21% for five, at 3 to 4 DN. On the 2026-11-11 frames, Galileo's matched frames give 435 simulated sky patches on 8 frames (0 before), and Voyager's give 730 on 3; the Cassini and LORRI figures are unchanged. Galileo: simulated 0.215 DN against real 0.960 DN (822 real patches, W1/IQR 8.2). Voyager: simulated 3.09e-3 against real 1.85e-3 I/F (611 real patches, W1/IQR 0.45). Per frame, real against simulated: C1205021 6.86e-3 against 3.16e-3 (202 against 241 patches), C3446143 1.69e-3 against 3.09e-3 (205 against 241), and C3448723 1.70e-3 against 2.98e-3 (204 against 248). The Voyager comparison is not like for like. The real frames are resampled (GEOMED), which correlates neighboring pixels, so paired differences read their noise about a third low (on C3446143's 205 sky patches, 1.68e-3 I/F against 2.55e-3 from the fit on the same patches). The matched scene does not turn on the simulator's resampling mode (`resample_texture`), which the Voyager config describes as correlating noise the way GEOMED does. And C1205021 is filled by a cloud deck, so its darkest quarter of tiles is cloud, not sky. The hold-out and transfer parts come in Phase 6. |

**Exit:** changing the signal in one region of a frame leaves the noise elsewhere in that frame
unchanged (met by #911); validation scenes assert against their planted truth (met by #905's
truth-check part); a null body key behaves exactly as an absent one on every entry point (met by
#900, #940, and #958); the realism match fails on a planted regression (arms with the
regeneration below, done by PR #1034); adding a camera's tables is a YAML edit (met by #876).

The last step of Phase 1 is the realism-record regeneration. **Done (PR #1034, record of
2026-11-17 on commit ea3c5442):** #989 landed first (PR #1004); the regenerated record arms the
regression bands of #903 and so meets the "realism match fails on a planted regression" criterion,
and #961's sweep figures were regenerated in the same pass.

Phase 1b starts now: the realism-record regeneration (with #961) and #989, its prerequisites,
are done. #960 is done (PR #1067). #854 (on hold) and #126/#945 do not hold it up; a later PR among
them that moves renders follows the rebase rule in the Risks section.

## Phase 1b: pixel-neutral restructuring

The code-structure critique (`critiques/SIMULATOR_CODE_STRUCTURE_CRITIQUE_2026-11-16.md`) found the
simulator's core well designed: the stage pipeline, the artifact-mode registry, the instrument
catalog, and the information boundary. It found much of the rest accreted. The restructuring that
changes no pixel lands now, while the committed baselines are still and can prove each change
neutral. From Phase 2 on, nearly every PR moves renders, and a refactor can no longer be told
apart from a physics change.

| Issue | Subject | Why now |
|---|---|---|
| #923 | **Done (PRs #1036, #1037, and #1038).** Duplicate image classifier, test-only `artifact_counts.py`, no shared scene fixtures, plan codenames, and the navigator-side `sim_params` name. **The classifier part is done (PR #1036):** the thresholds, their reading from a camera block, and the ordered decision are in `spindoctor.support.image_class`, which the navigator's classifier and the editor's verdict both call. **The test-only module, codename, and rename parts are done (PR #1037).** **The shared scene fixtures are done (PR #1038).** | Shared scene fixtures first, so every later schema change edits one place instead of 50 test files. |
| #1005 | **Done (PRs #1045, #1046, and #1047).** Lay out the simulator package by concern; the image side's `forward/` package becomes `render/`, with the `render.py` entry point inside it (operator, 2026-11-17). **The navigator-side part is done (PR #1045).** **The `sim` package part is done (PR #1046).** **The editor part is done (PR #1047).** | Every later issue creates its modules in their final places, and the boundary guard checks packages instead of file names. Lands right after Phase 1's realism-record regeneration, when no other simulator PR is open. |
| #1006 | **Done (PR #1051).** Exact copies and dead defaults | Each later change would otherwise have to find and edit every copy. |
| #1007 | **Done (PR #1054).** Names that describe history or one of several meanings | One glossary before the typed scene model fixes names into types. |
| #1008 | **Done (PR #1059).** The simulated body model is over 1,000 lines, and its limb and terminator arc code is two near-copies | The one module over the limit. |
| #1030 | **Done (PR #1064).** The simulated-or-real model choice is spread over eight guards; each model class declares the inputs it needs instead | One place to choose models, and the shape Phase 5 needs, where a twin provides both kinds of input. Gated also on the library regression tier, because it touches model selection on real frames. |
| #1009 | **Done (PR #1060).** Light direction and phase function derived twice | A shared convention with two copies can plant an unmeasured systematic. If a render moves, the move is that systematic and is explained. |
| #1010 | **Done (PR #1058).** The stage protocol says each stage is seeded by its generator; most stages discard it | Seeding by name is the rule every later stage must follow. |

#923's fixtures and #1005 go first. The others can land in any order after them.

#923's three choices are decided (operator, 2026-11-18): the classifying decision moves to a
neutral module both sides import, as its own small PR first, because the navigator's classifier
is in a file the navigator work also edits; `artifact_counts.py` moves under `tests/`; and the
plan-phase labels outside the simulator are out of scope, tracked in #1035.

**Exit:** the issues above are closed. Every committed render and baseline reproduced unchanged
through them, or a move under #1009 is explained in its PR. No simulator module is over 1,000 lines.

## Phase 2: rendering correctness

These are defects that put pixels in the wrong place. They come before any new physics, because
new physics on a wrong canvas is wasted work. #901 goes first: until the classic body renderer is
retired, every body fix has to be made twice. #931 lands with #901, so the topographic baselines
regenerate once. #1011 follows, so the one body path also has one projection, built for both of the
topographic renderer's grids. #1012 follows #901, because #901 removes the classic renderer from the
module #1012 splits. The render grid that removes hand-threaded oversampling (#1015) waits for the
typed scene, in Phase 3b.

**Done (PR #1102).** Of the rows not yet started, #893 goes first (operator, 2026-11-21). Its limb
seam paints a dark one-pixel outline wherever a body's edge crosses a pixel, because a partly
covered limb pixel keeps the body's share and drops what lies behind it; that outline is visible
around the haze of the Titan scenes, sharpens every occluding limb a limb technique fits, and
loosened two mutual-event tolerances in the disc refinement work (#1053). It follows #890, whose
coverage of a body too large for the frame the seam fix shares.

With #893 done and #892, #894, and #875 under way in their own PRs, the next row not yet started
is #842 with #646: no check compares where a simulated limb's brightness gradient peaks with a
real limb, and the limb-bias harness measures a simulated ridge half a pixel from the deployed
one. This phase's exit asks for that check.

| Issue | Subject |
|---|---|
| #901 | **Done (PR #1069).** One renderer, `bodies/ellipsoid.py`, now draws every ellipsoid; the classic path and the path choice are gone. A setting at an effectively zero value renders the smooth body: a relief of 1e-9 of the radius now changes no pixel by more than 1.2e-8 of peak, where it changed 628 pixels by up to 1%. A small body with relief at oversample 4 matches an 8x finer render as closely as a smooth one. Classic and topographic body renderers disagree; retire the classic path (part of #915) |
| #1011 | **Done (PR #1071).** The image-side renderer builds its working and detector grids with the shared `project_ellipsoid`, and the navigator's predicted body is the shared `render_lambert_ellipsoid`; the renderer's private grid copy is gone. The predicted body is bit-identical. An image-side render can move only when a coordinate of the body center, scaled to the grid (times the anti-aliasing factor on the working grid, divided by the oversample on the detector grid), is below 0.25 px. There the two copies rounded the centered coordinates differently by one unit in the last place. The largest move measured was 1.2e-14 of peak. The navigator-side predicted-body module is gone, since its one function only called the two shared helpers, and the predicted-ring module and function are now `predicted_ring` and `create_predicted_ring_feature`. The topographic body renderer projects through its own copy of the ellipsoid projection, while the navigator predicts through the shared one |
| #1012 | **Done (PR #1073).** Crater carving and body placement have their own modules, the body modules import one another only at the top of a file and never upward, and each render cache has a public clearer. One function, `body_identity_seed`, seeds every body, each caller hashing the key it hashed before, and no scene render moves. The body renderer modules import each other through hidden cycles, and the body identity seed is written out in several places |
| #931 | **Done (PR #1069).** Every ellipsoid is shaded on its working grid. The relief tilt is interpolated from the detector grid by a cubic spline, except in a limb band, along the relief's day-night edge, and wherever its cast shadows fall, and the crater slopes bilinearly; textures and transits are drawn on the working grid. Against working-grid shading the light of 14 test bodies with 2% relief at a 12-degree correlation length lands within 0.0015 px, and that of a 150 px body with 3% to 5% relief at 5 degrees within 0.0007 px. That holds while the relief's correlation length spans at least about 5 detector pixels; finer relief lands up to 0.011 px off at 1.5 to 4 pixels and 0.023 px at 1 pixel (#1068). Coarse shading had moved it up to 0.026 px on cratered and 0.008 px on textured bodies. Three regression baselines moved toward their planted offsets. Coarse-grid shading shifts relief, textured, and cratered bodies at oversample above 1 (smooth bodies fixed by #916); lands with #901 |
| #1013 | **Done (PR #1075).** The oversampling choice asks the optics stage whether the scene has a PSF, through `effective_psf`, the function the optics stage takes its kernel from, so a generic-camera scene with instrument defaults renders at oversample 1; no committed scene or render moves, since every committed scene with instrument defaults uses a camera with a kernel. A scene with instrument defaults oversamples 4x even when its camera has no PSF |
| #1014 | **Done (PR #1078).** One navigator-side module, `nav_model/simulated/predicted_body.py`, draws every predicted body for the body and haze models, so the haze model predicts a mesh occluder as its mesh, puts a body that states no center at the frame center, and boxes each sibling by its predicted silhouette on the extended frame, rotation included; no committed scene has a Titan beside another body, so no render or baseline moves. The simulated Titan model predicts a mesh occluder as an ellipsoid, and at pixel (0, 0) when no center is given |
| #890 | **Done (PR #1081).** A body, relief, or halo that reaches past the frame from the frame center is drawn on a canvas past the frame edge and moved into place as before (the interpolated move stays, per the #984 ruling), so it matches the unclipped body exactly at whole-pixel positions and in light and light center at fractional ones; only the three Titan halo scenes moved, gaining the halo light the move had left out of a one- to two-pixel strip at the frame edge. A body larger than half the frame and off center is clipped or vanishes |
| #825, #644 | **Done (PR #1086).** Per the 2026-11-20 ruling, a planted roll turns everything in view about the frame center in the sense the navigator's rotation fits report, where before the ellipsoid outline turned the wrong way and the mesh pose, lit side, craters, smear, companions, sky, and stray-light ramp not at all; the camera's PSF, distortion, ghosts, and radial stray light stay put. No committed scene or sweep step moved, since the only rolled ones are star fields that carry none of these. A planted roll turns ellipsoids the wrong way, mesh bodies and the lit side not at all, and leaves smear, companions, and sky unturned; #644 also asks for a ruling on what a roll turns |
| #1157 | **Done (PR #1159).** A planted roll now turns the scene about the boresight, the frame center after the planted offset, which moved every star of `planted_rotation_star_field` by 0.037 px and its fused error from 4.984 to 4.972 px, and every star of the six `star_rotation` sweep steps (the only other frames that plant both) by 2 * abs(d) * sin(theta / 2), about 0.014 to 0.054 px. Turn a planted roll about the boresight, after the offset |
| #891, #873 | **Done (PR #1093).** A pixel that a ring boundary crosses is split at the boundary and each piece is lit at its own optical depth; a lone edge lands within 0.002 px of its authored radius at every optical depth, pixel phase, oversampling, and view tested (it was off by up to 0.67 px). A pixel across a ramp is integrated in closed form. The rank-1 scene's remaining 0.46 px miss comes from the navigator's pixel-centered edge vertices along its grid-aligned edges. Ring-edge anti-aliasing on optical depth shifts lone edges up to 0.7 px; a rank-1 ring scene misses its planted shift by 0.52 px |
| #893 | **Done (PR #1102).** Each body now goes over what lies behind it by how much of each pixel it covers, and dims starlight by the same fraction, so the dark one-pixel outline at limbs over a farther body, a ring, a haze glow, or the sky is gone (seven scenes moved; `mutual_deep` and `titan_crescent_horns` now hold known errors from #1050); `rotation_tilt` is replaced by `sub_observer_lat_deg`, a true rotation of the ellipsoid that the image, the haze, and the navigator's predicted body share, so a sphere keeps its outline and bands curve, with no committed scene moving. Limb seam, coverage-blind star extinction, and `rotation_tilt` stretch |
| #84 | **Done (PR #1098).** Confirmed with no code change on a planet circled by an inclined ring, against the closed form: the near arm dims the disc by exp(-tau/mu) and adds its own light, a cleared gap and a faint ring show the disc through, and the far arm is hidden where the disc covers it and shows beside it; no committed scene puts a ring in front of a body. Simulated rings were copied in place, so ring edges overwrote brighter objects behind them and gaps erased what lay in them. The ring system has composited by optical depth since PR #313, so a gap or a faint ring shows what lies behind it; confirm on a ring-over-body scene and close |
| #892 | PSF and smear zero-pad the frame: darkened edges, no light from outside it |
| #894 | **Done (PR #1104).** Night sides render black in the image and in the simulated template (the production template keeps a 0.05 pedestal there, tracked in #1106), the predicted silhouette comes from the outline, brightness above 1 reaches the detector uncapped (Minnaert capped at 2), and the simulated limb is the lit outline by production's mean-incidence rule, which also closes #1056; 40 scene renders and 40 baselines moved, the low-phase limb mis-lock vanished with the night glow, the Titan crescents' known error went from 5.74 to 1.84 px on a haze #1101 finds four times too bright, and the #790 lock persists (mesh seed 10 still ends 12.8 px off on 24 of 25 draws, now at 0.89 confidence). Current state after #1053 and #1101: the Titan crescents' known error is 10.19 px, the disc correlation locked on the ring of light at medium tier, and mesh seed 10 is conflicted (body_shape_lock_suspect) on 25 of 25 draws, 13.62-13.76 px off at confidence 0.27, so the veto declines it. Night-side 1% floor and clipped photometric laws |
| #875, #1068 | **Done (PR #1107).** Each pixel of a relief body reads the terrain where its line of sight first meets the relief surface, ridge tops a quarter-cell step could pass over included, so the share of the disc read more than half a correlation length from the point shown drops from 9% to none at 5% relief and 15 degrees and from 63% to 0.008% at 2 degrees, and no normal is mirrored; cells where the point seen jumps behind a ridge or stretches across an edge-on slope are shaded on the working grid, and relief finer than 5 detector pixels is shaded wholly there, so the light lands within 0.00142 px of whole-working-grid shading on the bodies measured. The relief regression scene's fused error moved from 0.840-0.847 px to 0.878 px, where the disc correlation stops at the edge of its window (#791), and its blob band from 2.35-2.64 px to 1.79-1.85 px; the RMS cap, a scene-format limit, rises from 0.05 to 0.1, where the surface stays about half the radius out from the center; the guide's outline accuracy is corrected to 7e-4 on 99.9% of the azimuths at 5% relief, 2e-3 at worst. A body seen from a sub-observer latitude is read the same way, about its turned line of sight. Current state after #1053: the relief scene's fused error is 0.882-0.900 px, and the disc correlation no longer stops at the edge of its window; its answer is the high-pass surface's own maximum, pulled by the relief the smooth template lacks. The blob band, 1.79-1.85 px, is unchanged. Relief reads disc terrain at the smooth sphere's direction; fine limb relief is interpolated from too coarse a grid |
| #842, #646 | Simulated limbs are not checked for where the brightness gradient peaks; the limb-bias harness measures a simulated ridge half a pixel from the deployed one |
| #1163 | **Affects every calibrated Cassini realism row; found when the realism record was regenerated on current render code.** Simulated limbs now measure narrower than real on both calibrated cameras: NAC 2.19 px against 2.54 real (2.54 at 87bd9d68, W1/IQR 0.12 to 0.54), WAC 2.75 against 3.24 (W1/IQR 0.57 to 1.26). Bisect over the body render PRs merged since 87bd9d68, decide from the physics which value is right, and fix the simulator against the real frames |
| #1100 | **Done (PR #1141).** The cast-shadow march starts where each pixel's line of sight crosses the surface and counts only terrain between that point and the Sun, ground right at the point that turns from the Sun is shadowed though the smoothed normal faces it, a pixel whose normal faces away is marked shadowed so craters cannot light it, and a normal leaning away is laid just in front of the sky plane; against a brute-force truth on 18 bodies at phases 0 to 150 the renderer darkens no pixel at phase 0 (main 48), lights none of the pixels whose ground turns from the Sun (main 364), shadows none the truth leaves lit (main 3), and misses 19 of the 24,245 cast-shadowed sunward pixels (main 21), and the line-of-sight reading of #1107 had already cut the black pixels of an 80 px disc at 2% relief from 344 to none; against main, `relief_psf_stray_light` darkens 12 pixels by one display level and none of its 16 blob answers moves by more than 0.00002 px, and the two relief gallery images darken where ground turns from the Sun (388 pixels by up to 10 levels and 124 by up to 2). Relief terrain hidden from both the Sun and the camera is drawn as black shadow inside the lit disc, because the cast-shadow march has no visibility test; at phase 0 a 2% relief body has 344 black pixels on an 80 px sphere |
| #1101 | **Done (PR #1137).** The haze source carries the factor of 1/4, the haze reflects as a finite column by Hapke's isotropic multiple-scattering approximation with the surface seen through the column's direct transmission, and Titan's haze is a body property (`haze_model: titan`) that depends on wavelength: the DISR optical-depth power law, the albedo above 144 km, and two-term Henyey-Greenstein fits to the DISR phase functions from 430 to 1078 nm, averaged over the scene filter's solar-weighted passband from the new Cassini passband tables. The haze hides the surface, and a line of sight past the limb sees the far half of the tangent column through the near half, so a haze thick along the grazing path hides the solid limb (a line-of-sight integration agrees to 0.4%); against seven real Cassini Titan frames in their own filters the disc is 16 to 44% darker and the forward ring 1.5 to 3.0 times fainter, and against Monte Carlo the multiple scattering runs about 22% low (#1191). The Titan scenes are exposed so the haze peak fills the 12-bit range as on real frames at similar phase (3.6 s at 155 deg, 7.3 and 7.5 s near 60 deg) and carry the camera's 12 e- read noise in place of 4 DN, which ends the draw-to-draw flips: on 16 distinct draws each, `titan_haze_limb` went from a medium success 0.03 px off to a medium success 7.35 to 7.55 px off at 6 to 7 sigma (#1190), the crescents from a low-tier 2.87 px to a medium-tier 10.19 px on every draw, the disc correlation locking on the ring of light (#1050), `titan_haze` from 0.04-0.06 to 0.49-0.54 px, and the generic-haze `haze_limb_base` from 0.030 to 1.0 px; the haze sweep checks were re-derived. The haze albedo sweep walks the haze's brightness on a body whose surface is black, so the navigator sees only the haze's own edge: up to albedo 0.3 no draw comes within 1 px (the frame fails or locks 41 to 59 px off), and at albedo 0.6 every draw lands 0.25 to 0.44 px off. Haze over a flattened body takes its emission angle from the surface normal. The simulated haze is about four times too bright: its single-scattering source lacks the factor of 1/4 |
| #1189 | Titan's haze in methane-band and methane-window filters (MT1-MT3, CB1-CB3, and any passband overlapping a strong methane band): methane absorption with altitude and wavelength, surface through the windows; scenes naming those filters are refused until then |
| #1190 | **Affects `titan_haze_limb`, a medium-confidence success 7.35 to 7.55 px off at 6 to 7 sigma.** A body whose thick haze the navigator does not model navigates 7 to 10 px off at medium confidence: its disc and limb techniques read the top of the haze as the limb (navigator) |
| #1191 | **Affects the brightness of every Titan scene.** Simulated Titan haze is 16 to 44% too dark on the disc from 60 to 107 deg phase and its forward ring 1.5 to 3.0 times too faint at 149 to 167 deg; the isotropic multiple-scattering term runs about 22% low against Monte Carlo and carries no forward-scattered light along the limb |
| #1219 | **Affects no current scene: a Titan scene in a violet or ultraviolet filter is refused.** The haze table starts at 430 nm, so a passband with more than a quarter of its light below it is refused; the real CL1+VIO frame W1552216646 puts the limb about 90 km higher than CL1+RED |
| #1220 | **Affects the limb height of every Titan scene in RED.** The simulated sunward limb falls to half its peak 51 km lower than real in CL1+RED (209 against 260 km on W1552216540), and it moves 56 km between BL1 and RED where the real limb does not move |
| #1224 | **Affects `haze_limb_body` and the guide image of it, not Titan: under a moderate haze at low phase the night-side solid limb steps up about 10%.** The far half of the tangent column is lit as the near half for the light scattered more than once, so that light does not see the limb the way a line of sight does |
| #1192 | **Affects no current scene.** Only the Cassini cameras have filter passbands, so a Titan haze on a Voyager, Galileo, or New Horizons camera is refused |
| #1193 | **Affects the `util/` Titan truth and calibration campaigns; no committed scene.** Their scene generators give Titan a generic thin haze through which the surface shows, in place of Titan's published haze |
| #1196 | **Affects the draw counts of every scene whose only random content is the detector noise; no tolerance or band moves.** The truth procedure's noise draws repeat its random draws there, so such a scene is measured on 16 distinct draws, not 24 |
| #1197 | **Affects the albedo of every Titan scene, by an amount not yet known.** The Titan haze albedo table is a transcription of Tomasko et al. 2008 Table 2 that has not been checked against the paper, and its notes disagree with its rows |
| #1200 | **Affects `titan_crescent_horns` only when noise swamps the frame; at real exposure and read noise every draw gives the 10.19 px answer.** The disc correlation's spurious flag hinges on a near-tie between two peaks at its coarsest level, so noise alone can switch its answer between 2.4 and 10.2 px off (navigator) |
| #1201 | **Affects the 59 scenes that carry it.** Simulator scenes override read noise to 4 DN, 4 to 6 times the 0.7 to 1.1 DN of sky noise on real 12-bit Cassini frames; the Titan scenes take the camera's own |
| #1106 | **Affects every scene navigated with the simulated body model, by an amount not yet measured; resolved by #843 if simulated frames navigate with the production model.** The simulated body template is 0 on the night side, where the production template carries a 0.05 pedestal under a lit maximum of 1.05 (operator 2026-11-22: keep 0 for now) |
| #1076 | **Affects no current scene.** The simulated Titan model draws a nearer moon on the frame alone, so its part in the margins hides none of the envelope |
| #1077 | **Done (PR #1129).** The overflow and visible-arc canvases now reach as far as the mesh's farthest projected vertex, and the overflow shortcut uses the same reach. **Affects no current scene.** The simulated body model's overflow and visible-arc fractions draw a mesh body on a canvas of half again its longest semi-axis, which a very lumpy mesh reaches past, and the overflow check can report 0 for it without drawing; the tilted-ellipsoid case is fixed by #893, whose true tilt keeps every ellipsoid inside that canvas |
| #1076 | **Done (PR #1128).** The simulated Titan model and the simulated body model now draw a nearer moon on the extended frame, so its part in the margins hides the envelope and the body as the rest of it does; no committed scene puts a moon at the frame edge, so no render moved. |
| #1077 | **Affects no current scene.** The simulated body model's overflow and visible-arc fractions draw a mesh body on a canvas of half again its longest semi-axis, which a very lumpy mesh reaches past, and the overflow check can report 0 for it without drawing; the tilted-ellipsoid case is fixed by #893, whose true tilt keeps every ellipsoid inside that canvas |
| #1080 | **Done (PR #1130).** The mesh renderer now measures faces and pixel centers from the body's own position on the pixel grid, so a mesh body draws the same pixels on a canvas of any size; no committed scene, sweep step, or population scene moved. A mesh body with the default pose draws up to 3% of peak differently near its outline on frames of different size |
| #1084 | **Done (PR #1140).** The sky field now turns about the boresight with every other object, moved after the roll by the planted offset turned by the roll, drawing the same stars and keeping each by where it lands. **Affects 10 current frames, but only which random clutter stars they show, not their statistics.** The sky field turns with a planted roll but does not move with the planted offset: in the 10 committed frames with a sky field, every sky star sits 1.55 px from where it belongs |
| #1068 | **Done (PR #1107), with #875.** **Affects no current scene.** Fine limb relief (a correlation length under about 5 detector pixels) is interpolated from too coarse a grid, moving the light up to 0.023 px; the fix is shading it wholly on the working grid |
| #1094 | **Affects no current scene.** A ring ramp under a density wave renders about 35 times slower than without it (69 s against 1.9 s for a 300 px ramp at 1,024 px and oversample 4), because every band pixel takes the 32-cell curved path |
| #1057 | **Affects no current scene.** A Henyey-Greenstein asymmetry within rounding of 1 or -1 divides by zero in the ring renderer; cap it in the ring and haze rules |
| #1115 | **Affects no current scene.** The outline of a relief body seen at a latitude is rebuilt on every render (5.9 s at a 1-degree correlation length); cache it by view |
| #1127 | **Done in part (PR #1179).** Relief bodies render bit for bit as before, 2.2 to 3.0 times faster on one core (the 150 px 5% body 18.8 s to 7.5 s, the 280 px fine-relief body 195 s to 66 s, the render-budget body 4.9 s to 3.7 s); the 2 s and 20 s targets wait on compiling the walk and the march with numba (#1182). **Affects the relief scene and every relief sweep by render time only.** A body with relief takes far too long to render (17 s for a 150 px rough body, over 3 minutes for fine relief) |

**Exit:** a frame-filling planet renders at its true position with light from beyond the frame
edge; a rolled scene behaves as the ruling on #644 decides; a lone ring edge lands within 0.05 px
of its authored radius; the simulated limb's gradient peak is checked against real limbs; every
moved baseline is explained. One ellipsoid projection serves both sides, one function seeds every
body, and no body module imports another inside a function.

## Phase 2a: settle the scene format

Phase 2b writes realistic guard scenes, and Phase 3 repairs the calibration generator, which writes
scenes too. The changes that move or remove scene keys land first, so those scenes are written
once, in the format that stays. The same changes build the shared schema core that the typed scene
(Phase 3b) uses. Renders move here only where an issue says so, and each moved baseline is
explained.

The navigator config work resumed on 2026-11-23 (operator) and is tracked in
`PROGRAM_PLAN.md`: the camera blocks (#1029), and every other section with the deletion of the
keys nothing reads (#1032). The operator ruled on 2026-11-23 that the navigator config will
refuse null keys, as scenes do. No issue in this plan changes navigator code; #1023 edits the
navigator's config files only to delete the camera-block keys nothing reads.

| Issue | Subject | Why now |
|---|---|---|
| #988, #986, #987, #990 | **Done (PR #1116): the validator refuses a null anywhere in a scene with its key path, and each nav_override value passes the check of the body key it replaces.** Refuse null keys in simulated scenes (operator ruling, 2026-11-15); a null noise value crashes the renderer; values inside `nav_override` are not validated | The typed scene needs "unset" to mean absent, never null. A null inside `nav_override` is refused like any other. |
| #1021 | **Done (PR #1109).** The core is the `spindoctor/schema/` package: schemas are frozen dataclasses that pydantic parses, each problem is one line naming the file and key path, and a layer merge applies the REMOVE and REFUSE null policies. One typed schema core for configuration and scene files: plain frozen dataclasses parsed by pydantic (operator decision, 2026-11-16), and one layered YAML merge with a null policy per family | Every later typing step builds on it. Its first consumer is the instrument catalog (#1022). It changes nothing in the navigator config. |
| #1022 | **Done (PR #1114).** Catalog entries are typed frozen dataclasses parsed through the core, with their cross-checks as `__post_init__` rules, and each artifact mode's parameters are annotated types checked by one parse for scenes and catalog entries alike; every committed render reproduces unchanged. The instrument catalog and the artifact-mode parameters, parsed through the core | Typed detector fields, which the scene's noise and detector blocks reuse. One mode-parameter interpreter, not two. |
| #806, naming part | **Done (PR #1118).** `vgiss.yaml` holds `vgiss_1_nac`, `vgiss_1_wac`, `vgiss_2_nac`, and `vgiss_2_wac`, each stating every value itself with the same numbers, so nothing renders differently; `instrument: vgiss` now fails as unknown. Split the `vgiss` entry into four per-camera entries, renaming the scene's instrument value | Renames a scene value before Phase 2b writes Voyager scenes. The per-camera values stay in Phase 4e. |
| #1017 | **Done (PR #1124).** Banding, hot pixels, and the readout dark are one effect each, set by one artifact mode; dark current stays a separate effect in the noise block, shutter shading becomes a mode of its own, bias structure, bloom, and quantization each keep one setting, and the detector modules are grouped by physics. No committed scene, sweep, or baseline moves. Under `instrument_defaults` Galileo's bands turn vertical and Voyager's take a new phase, and LORRI's banding is drawn as published (random row striping within 1 DN plus a damped amplifier ringing from the first column), which moves the realism match's Galileo, Voyager, and LORRI statistics; the realism record and LORRI's regression bands are regenerated. Detector effects modeled twice, once in the noise block and once as an artifact mode | Removes or merges scene keys. |
| #1023 | **Done (PR #1138, with #1139).** The instrument entries are the only image-side source of detector physics, a scene's `detector` block sets the ceiling, well, bias, signal scale, and loss rate, the `noise` block holds only noise, the renderer reads the navigator config only for the marker and a scene's matched PSF, and the camera-block keys nothing reads are deleted; no committed render moves. Image-side detector physics comes from two configuration trees, and the noise block holds keys that are not noise | Moves scene keys, and lets a scene plant a mismatch between the true and believed saturation ceilings. It also deletes the camera-block keys nothing reads, the image-side keys it stops reading among them. |
| #1072 | **Done (PR #1111).** The key, the control, and the fallback seeds are gone, the body renderers require the scene's seed, and the simulator integration tier passes with no committed render or baseline moved. **Affects no current scene.** A body's own `seed` key, and the editor's "Crater seed" control, never change a scene render; remove them and the fallback seeds behind them. A body renderer called directly without a scene seed, as some tests and tools do, then needs a seed passed in, and those direct renders change | Removes a scene key and an editor control before the typed scene declares every key. |

Order: #988 and #1021 first, then #1022, #1017, and #1023, one at a time; #806's naming part and
#1072 land whenever they are ready.

Decisions needed before the work starts, asked one at a time: for #1017, whether each pair (banding,
dark signal, hot pixels) is one effect or two.

**Exit:** the issues above are closed. No
scene key is renamed or moved after this phase except by an issue that adds new physics. The
simulator's instrument catalog is typed. Image-side detector physics has one source.

## Phase 2b: realistic guard scenes

Once pixels land where they should and the scene format has settled (Phase 2a), write the
validation scenes that YAML can express today, so they guard the churn of Phases 3 and 4. These
are the authorable parts of #904 and the ring-only part of #377:

- every camera the simulator catalogs, at real frame sizes and real offsets, with twist;
  **Done (PR #1154; Voyager in PR #1218, after the four-camera split, #806, PR #1118):**
  the `full_frame_cameras` class holds ten full-frame scenes on the Cassini NAC and WAC (raw
  and calibrated), Galileo SSI (257 and 286 px offsets), and LORRI, each with its camera's
  measured twist unfit, navigated once per run by `test_sim_full_frame_cameras.py`; three of
  them are expected failures tied to #342, #785, and #1153. Four Voyager scenes, one per
  camera at the 1000 px GEOMED size and 32 to 49 px offsets, follow real frames: a
  Voyager 1 NAC moon (0.06 to 0.57 px off), a Voyager 1 WAC Saturn at the real frame's
  scale, ring radii, ring brightness, and limb profile with unmodeled rings, where the disc
  correlation carries the frame at medium tier as on the real frame (0.18 to 0.69 px off;
  the planet is too small for a simulated limb, #1209), a Voyager 2 NAC Pleiades field
  (0.03 to 0.26 px, the star-field fit locking normally as on the real frame), and a
  Voyager 2 WAC Scorpius field whose unfit +0.358 deg twist puts the answer 1.0 to 1.4 px
  off. The same PR sets the Voyager 2 NAC and WAC star zero points from their own star
  frames (8.7e4 and 4.6e3 raw DN/s for V = 0);
- star fields with a body, and rings with a body; **Done (PR #1155).** The `mixed_content` class
  holds stars with a body 1.81 px off its predicted position (fused 0.16-0.26 px off, pulled
  from the stars toward the body), stars with a crescent on a summed 512 px frame (fused
  0.01-0.18 px off, the disc 0.01 px), a moon over a ring sheet, which conflicts because
  RingEdgeNav slides 8.4-9.3 px along the arcs at a 0.1 px sigma (#755), and a distant Saturn
  with its rings on the WAC (fused 0.03-0.10 px off, within 2.6 reported sigmas); a nearer
  planet with rings waits on #896;
- dense ring profiles at real scale and low opening, with only catalog edges navigable and
  unmodeled structure beside them; **Done (PR #1156).** The `dense_rings` class holds five
  full-size NAC scenes (a navigated Keeler-gap profile, the F ring beside the A ring's edge,
  the same rings at an opening of 2.5 degrees, a hidden Maxwell-ringlet eccentricity, a
  saturated B ring), each held to a known-error band (#755 for the Keeler gap, #1143 for the
  unfit twist, #1152 for the low opening, #1144 for the hidden eccentricity, where the frame
  conflicts, #1145 for the saturated edge);
- interleaved defect modes on one frame. **Done (PR #1198).** The `mixed_defects` class holds a
  lossy table NAC moon, a NAC stars-and-moon frame with groups of lost lines down the frame, a
  truncated bottom, and 2 px of smear, a Galileo moon with compression and lost 8-line slices,
  and a raw LORRI moon with frame-transfer smear; all four navigate within 0.12 px over 24
  distinct draws (16 for the lossy moon, which seeds only its noise). The Galileo frame's lost
  slices leave its answer within 0.08 px.

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
| #837 | **Done (PR #1171): the regular moons render as ellipsoids carrying the photometric and surface keys, the irregular moons as meshes carrying only mesh keys, and all eight families generate and validate.** The calibration generator fails on four families |
| #898 | **Done (PR #1171): every catalog body is drawn with semi-axis a equal to its drawn radius, and its rendered width matches twice that radius.** The calibration generator draws catalog bodies at half size |
| #1174 | **Affects the refit of BodyTerminatorNav.** At full body size the disc correlation covers the crescent, so the terminator family runs BodyTerminatorNav (a fallback) in about 1 of 30 scenes against 12 of 30 at the old half size; decide before the refit whether the family measures the technique in isolation or changes its geometry |
| #309 | The campaign does not run the realism-anchored configuration (chain off, NAC only) |
| #793, #874 | Refit on the repaired renderers and generator, under the overflow limits, and re-ratchet the moved coefficients |
| #942 | Disc and ring confidence was fitted on runner-up ratios that wrap-around depressed before #930; the refit covers them |
| #794 | The Titan haze calibration (`cross_sigma_scale` and the `TitanHazeNav` confidence anchors) was fitted on the haze render with the equator seam that #627 (done) removed; rerun the Titan campaign on the shipping render and refit both; the fit also predates the haze's single-scattering factor of 1/4 (#1101), so the refit runs after that fix |
| #341 | The campaign's scene mixture is authored, with no data that it matches real frames |

Track R replaces simulation-fitted values with real-anchored ones wherever real anchors exist;
this phase keeps the remaining simulation-fitted values correct until then.

**Exit:** the campaign regenerates on main with bodies at their catalog size and the full
instrument chain on, the refit coefficients are re-ratcheted as #793 and #874 require, and each
move is recorded in `plans/VALIDATION_AND_CALIBRATION_PLAN.md`.

## Phase 3b: restructure the renderer before the physics

Phase 4 adds most of the simulator's remaining physics, and with it many new scene keys and stages.
Under the current structure, each new key is written in up to six places, each new artifact mode
needs its own dispatch, and every stage reads the scene dict with its own defaults. This phase
gives the renderer the shape Phase 4 builds on. It runs alongside Phases 2b and 3, and must be done
before Phase 4 starts. It changes no pixel, except where #1015 finds a hand-scaled key that was
scaled wrongly, and where #1074 stops a scene that asks for oversampling with no PSF from pulling
its stars toward pixel centers (no committed scene does); each such move is explained in its PR.

| Issue | Subject | Why now |
|---|---|---|
| #1024 | Parse every scene into typed scene dataclasses that are its field table, and make that parse the only validation (3 to 5 PRs, block by block) | Each Phase 4 key is then declared once: type, range, default, boundary class, and editor hint. The boundary is structural: truth types extend idealized types. Until the navigator camera blocks are typed (navigator config work in `PROGRAM_PLAN.md`), the scene's `instrument_config` overlay gets its own type, which repeats the camera keys' ranges (the operator accepted the duplication), and replaces the simulator's 378-line overlay validator. |
| #1028 | Resolve each render's settings once into a `RenderSettings` record on the frame (after #1024) | One place interprets switches such as `instrument_defaults`; the detector parameters resolve once per render. |
| #1027 | Every render stage reads the typed scene (after #1024 and #1028; a few PRs, by stage) | Deletes about 200 raw scene reads in the image-side stages and the render-time validation backstops. |
| #1015 | Render through a grid object that owns oversampling (after #1028) | Removes the factor from 14 modules before Phase 4 adds pixel-unit keys. |
| #1026 | A typed idealized view of the scene for the navigator side (after #1027) | The navigator's simulated models accept only idealized types, checked by mypy. The shared helpers take typed records, and `scene_value` and the per-object parse functions go. |
| #310 | Move the truth attributes off `ObsSim` into a truth record (after #1026) | Completes the structural boundary. |
| #1025 | One shared per-kind layout for ring features on both sides (after #1026) | #909 and #910 add ring kinds. |
| #1016 | Every artifact mode applied through its registry entry, one loop per stage (after #1017 and #1028, which owns activation) | #906, #919, #935, and #850 add or change modes. |
| #1018 | The radiance stage split into named sub-steps (after #1027 and #1028) | #895 and #896 change this stage. |
| #1019 | Each render logs its `RenderSettings` (after #1028) | Phase 4's realism work needs to see which values a render used. |
| #1020 | The simulated-image editor rebuilt from per-block panels generated from the typed scene's field metadata (after #1024 and #1028) | Each Phase 4 key then gets its editor control from its declaration, not from hand wiring that a later rewrite throws away. Closes #902, #947, and #967. |
| #1074 | **Affects no current scene.** A scene that asks for oversampling with no PSF pulls its stars up to 0.43 px toward pixel centers (with #1015) | #1015 owns how the oversampling factor is chosen. |
| #1070 | **Affects no current scene.** The haze layer places its body through the shared ellipsoid projection (with #1015, or just before it) | The haze layer keeps its own copy of the projection and the normals; the grid object is where a projection of a box of the frame belongs. |
| #1066 | **Affects no scene; a test-budget fix.** Two render-time checks exceed their CPU budgets on main (the 1,024-pixel ring and haze body) | The restructuring in this phase must not slow rendering, and a budget check that always fails cannot show it. |

Order: #1024, then #1028, then #1027, then #1026. #1015, #1016, #1018, #1019, and #1020 follow in
any order their own dependencies allow; #310 and #1025 follow #1026. #1066 goes first, so the
budgets can guard the rest; #1070 goes with #1015.

**Exit:** the issues above are closed. No render module reads the scene dict. The navigator's
simulated models accept only the idealized types, and `ObsSim` holds no truth attribute. Each stage
applies its artifact modes in one loop over the registry. No function in the radiance, atmosphere,
ring, body, or detector-parameter modules is longer than about 100 lines. Every scene key has an
editor control generated from its declaration, or a documented exception. Every committed render
reproduced unchanged through the phase, apart from #1015's and #1074's explained moves.

## Phase 4: physics, in order of how many real frames need it

Phase 4 starts once Phase 3b's exit is met, including the editor (#1020), so each new key gets its
type, validation, default, and editor control from one declaration, and once Phase 2b's guard
scenes are committed, so they guard Phase 4's changes. Its items are ordered by the
share of real frames each
gap affects, from the critique's population counts.

**4a. Radiometry and shadows (every frame with more than one object type):**

| Issue | Subject |
|---|---|
| #895 | Stars, bodies, and rings share no radiometric scale. **Carries #925's star-count criterion (operator, 2026-11-18):** the matched NAC star twins should give a star cutout count comparable to the real frames' 25; the 2026-11-17 record gives 9, because simulated NAC stars are about 3.5 mag fainter than real ones at the same catalog magnitude. |
| #1153 | **Done (PR #TBD).** Measured on the 1992 Pleiades full frames, the Galileo star zero point is 2.3e8 electrons per second at V 0 (2.07e8 within 4 px at 10K on 118 stars, 1.95e8 at 100K on 62, 0.16 to 0.18 mag rms), up from 4.0e6; the 1990 frames read 1.3 mag fainter (#1211) and summed frames read 1.33 times the DN of full frames, and `gossi_star_field` now fails on all 24 draws until the navigator's copy of the zero point follows (1.23e6 DN/s at 187 e-/DN). **Affects `gossi_star_field_short_exposure`, held as an expected failure.** At the real Pleiades frames' 0.1 s the navigator predicts only the V 2.8 star of a simulated Galileo field, where the real frames lock on three stars: simulated Galileo stars are about 12 times fainter than real ones of the same magnitude (4e7 to 7e7 electrons per second at V 0 on the real frames against 4.0e6), and the simulated-frame star limit spreads that zero point over the 3.0 px placeholder blur (#342), reaching V 3.2 against V 7.8 on the real frames |
| #1213 | **Done (PR #TBD).** Measured on unbinned and 4x4 sci frames, the LORRI star zero point is 7.0e8 electrons per second at V 0 (6.8e8 unbinned on 173 stars, 7.2e8 binned on 444, 0.11 and 0.18 mag rms), up from 7.0e7; `nhlorri_star_field` lands 0.93 to 0.94 px off over 24 draws, and `nhlorri_star_field_short_exposure` keeps failing until the navigator's copy follows (3.33e7 DN/s unbinned, 3.61e7 binned). **Affects `nhlorri_star_field` and `nhlorri_star_field_short_exposure`.** Simulated LORRI stars are about 11 times fainter than real ones, and the navigator's simulated-frame copy of the zero point is equally low |
| #896 | No ring-body shadows; the planet's shadow is a fixed wedge |
| #921 | Background star counts 3 to 10 times too sparse |
| #899 | Navigator star limit: 30.0 on calibrated frames, filter ignored on raw ones (the Cassini part here; the Voyager part follows #907, and the Galileo, Voyager, and LORRI limits follow #342). **Carries #925's star-count criterion (operator, 2026-11-18):** the navigator on the NAC star twin N1459552012 should predict a number of stars comparable to the real frame's 4; in the 2026-11-17 record its star-field fit ranks 30, the most it takes. |

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
| #906 | **Hot-pixel part done (PR #1168): raw Cassini hot pixels scale with exposure and follow the truncated power law of the 2008 dark frames, Galileo's are per-mode constants, and calibrated Cassini products carry their own residual population fitted to the cohort's stationary spikes; the bloom, anti-blooming, and LORRI parts remain.** Hot pixels ignore exposure; bloom discards charge; LORRI bloom contradiction |
| #977 | Simulated Cassini TABLE-downlink skies are about a third as noisy as real ones (NAC median ratio 0.37 against 0.88 at 12 bits) and lack the real skies' offset; split from #925 |
| #979 | Simulated Galileo SSI skies are about a quarter as noisy as real ones (0.22 DN against 0.96 DN on 8 frames each), measured with #903's quantized-noise fit; the Voyager comparison waits on modeling the GEOMED resampling in the twins |
| #995 | On a binned simulated frame the navigator's star blur, star box sizes, and per-pixel thresholds stay in unbinned pixels, and a `match_navigator` PSF is not divided by the binning. #993 (done) fixed only the faint-star limit, so none of these is covered yet. No committed scene or sweep sets binning; the realism twins do, with the catalog kernel |
| #920 | WAC and Galileo gain states, Galileo ICT, and LORRI frame-transfer smear and calibrated mode |
| #1121 | **Affects no current scene.** Galileo's shutter shading scales the light at every exposure, where the shutter adds a fixed exposure time that differs across the frame |
| #1125 | **Affects no current scene.** The vidicon residual image is shifted with wraparound, so a source at one edge puts its ghost on the opposite edge |
| #1126 | **Affects no current scene; the truth record only.** The frame-transfer smear records the mean of both pedestals where each pixel receives one |
| #1131 | **Affects no current scene.** The banding mode skips its dark step whenever the band amplitudes are zero, so a scene asking for the step alone gets nothing |
| #1132 | **Done (PR #1168): the streak is flat on the rows read after the hot pixel, as raw Cassini and Galileo frames show: on Cassini the pixel's rate times its 0.0146 s per-row dwell, on Galileo a per-mode constant.** **Affects every frame drawn with the Cassini or Galileo camera defaults.** A hot pixel's warm streak sits on the side of the column read before it and is strongest at line 0, where the readout puts it on the other side; which process it models needs a ruling first |
| #1169 | Simulated Galileo hot pixels ignore the camera's frame mode and the date: real 10K darks hold about 8x more hot pixels in 1992 than 1990 in the same mode, and about 6x more in the 8 2/3 s mode than the 60 2/3 s mode |
| #1170 | Simulated raw Cassini dark current, 5 e-/s, is about 20 times what real dark frames show (at most 0.1-0.26 e-/s) |
| #1177 | Simulated calibrated Cassini frames lack the dark pits calibration leaves at raw hot pixels (real calibrated frames sit 0.2 to 1.3 times the raw charge below their neighbors there) |
| #1178 | The realism match counts a cluster shared by two calibrated NAC frames (192 of 957 stationary spikes, in rows and columns 512 to 767) as hot pixels, inflating the real-side target the simulated calibrated hot pixels are fitted on |
| #1186 | **Affects no current scene.** The hot-pixel power-law sampler rounds to the minimum amplitude for an index far below 1e-15 (the instrument entries use 0.94 to 1.0); compute the inverse CDF in log form |
| #1133 | **Affects no current scene.** Simulated bright and dark pixel pairs are placed independently, so two pairs can overlap and cancel while the truth record lists both |
| #1134 | **Affects LORRI frames drawn with its defaults, which draw 7 tail pixels of 8.** The serial tail writes one pixel fewer than its length, so a one-pixel tail draws nothing yet reports itself active |
| #1135 | **Affects no current scene.** The 8-bit contour quantization rounds 252 to 255 DN up and clips them to 255, a level off its step grid |
| #1136 | **Affects no render; the code and its description disagree.** A cosmic-ray streak or splatter deposits the sampled charge on every pixel of its footprint, where the docstrings and the guide call it the charge per event, and the catalog amplitudes were fitted with the per-pixel behavior |
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
| #1161 | **Affects the Voyager realism rows, which compare two of the four cameras on one or two frames each.** Curate at least 3 library frames for each Voyager camera, star frames among them, with verified pointing and a sidecar stating the spacecraft, so each camera's values can be measured and checked against its own frames |
| #806, per-camera values | One `vgiss` key for four cameras; the split of the entry and the scene value is Phase 2a; this part gives each camera its own values. Operator ruling 2026-11-23: optical values (PSF, distortion, plate scale, star zero point, vidicon values) are per camera and never shared or inherited, because each vidicon and telescope is unique; PR #1118 states every value in each entry, with the same numbers until a camera is measured. Operator ruling, second part: every analysis (realism, field-of-view distortion, rotation) is per camera; the realism cohort of each entry holds only its own camera's frames, and a sidecar gives its spacecraft |
| #907 | GEOMED plate scale, beam-bend sign, reseau lattice, and a fixed DN scale with no exposure term; each is measured and stated per camera |
| #804 | Noise keys inert on the vidicon and on calibrated frames |
| #935 | The vidicon chain renders no fixed-pattern structure, so `vgiss` does not offer `fixed_pattern`; measure the residual pattern on calibrated frames before adding a stage |
| #355 | Distortion and twist per camera once the split keys exist: distortion, twist (rotation), and every other analysis are measured per camera (deferred until star navigation locks enough frames per camera) |

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
| #1209 | **Affects `vgiss_1_wac_saturn` at its real scale: a Saturn of 94.6 px emits no simulated limb, where the production model fits it.** The simulated body model emits no limb for a body under 100 px across (`_MIN_LIMB_DIAMETER_PX` in `nav_model/simulated/body_arcs.py`), where the production model gates on the uncertainty of its ellipsoid fit. On the real frame C3446143 the production model emits the limb (BodyLimbNav, 168 vertices, visible-arc fraction 1.0, confidence 0.61, 0.33 px from the disc answer); on the simulated frame the limb fit does not run, and the disc correlation and the blob are the only body techniques |
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
| #975 | BodyBlobNav answers up to 36 px off on validation scenes and does not flag the answers spurious; the truth checks hold them in bands |
| #790 | The shape-lock veto misses confident wrong solutions on irregular mesh bodies: successes at 0.9 or more confidence 3 to 13 px off on most mesh seeds. A navigator fix the simulator found (Priority 2), listed here with the other navigator failures the simulator found; no phase waits on it, and a strict expected failure over mesh seeds holds it until it lands |
| #1062 | **Affects the test that holds #790.** Once the disc refinement work (#791) lands, no mesh seed reproduces the shape-lock veto miss; find a scene that does and hold it as a strict expected failure |
| #1050 | **Affects `mutual_deep`, `titan_crescent_horns`, and its noiseless twin, held as known errors.** The disc correlation is pulled by bright structure its template leaves out (navigator work) |
| #1092 | **Affects `planted_rank1_ring_flat`, held as a known error.** Ring-edge vertices sit at border-pixel centers, which biases the fit for straight edges (navigator work) |
| #1099 | **Affects `mutual_grazing` and `mutual_half`, held as known errors.** Overlapping body templates are painted by a yes-or-no mask, which leaves a dark outline at the occluding limb (navigator work) |
| #1103 | **Affects the widest step of the PSF sweep.** The limb fit reports a confident answer a pixel wrong under a blur the navigator does not model (navigator work) |
| #1146 | **Affects `coiss_nac_star_field`, where the other star techniques carry the answer.** StarFieldFromCatalogNav passes only its 30 brightest sources to the matcher, and on a raw frame cosmic rays take those places, so it returns spurious (navigator work) |
| #1147 | **Affects `coiss_wac_disc`, held at low tier.** The disc correlation discards any answer within 2 px of its search limit, so with the WAC's 13 px window a correct answer past 11 px is thrown away (navigator work) |
| #1148 | **Affects no current scene: the Galileo guard scenes plant the measured sky noise of #979, because real Galileo skies are that noisy.** A quantized 8-bit sky measures zero noise, and the star model then draws no stars. The Galileo star field at the real Pleiades frames' 0.1 s exposure and the catalog read noise reproduces it (navigator work) |
| #1149 | **Affects no current scene: with the hot pixels scaled to the exposure (#906, PR #1168), `saturn_rings_wac` sits at 0.9 to 2.6 sigma and needs no sigma band.** On a whole-system WAC frame RingEdgeNav answered 0.15 to 0.23 px off at a sigma of about 0.04 px, 3 to 5 sigma, with the simulated hot pixels near full well whatever the exposure. Re-measured on PR #1168 it answers 0.03 to 0.10 px off at a sigma of 0.04 to 0.05 px; the planet in view still adds to the error (0.10 against 0.08 px on the committed seed) |
| #1150 | **Affects no current scene: with the hot pixels scaled to the exposure (#906, PR #1168), `stars_body_ephemeris_error` shows no such draw.** StarUniqueMatchNav answered 114 px off at 0.70 confidence on one draw of 16 and did not flag it (navigator work). Re-measured on PR #1168 it flags 5 of its 25 answers spurious, and the rest lie within 0.25 px of the truth |
| #1143 | **Affects `a_ring_edge_f_ring`, held as a known error, and the other `dense_rings` scenes in part.** The NAC's unfit twist of 0.0115 degrees slides ring answers along curved edges by about 0.3 px, and moves weakly determined answers by more, and the reported sigma leaves it out (navigator work) |
| #1144 | **Affects `maxwell_hidden_eccentricity`, held as a conflicted frame with a known-error band on each technique.** RingEdgeNav follows an eccentric ringlet whose catalog orbit is circular and answers 6.8 px off at 120 sigma; RingAnnulusNav, pulled by the same ringlet, answers 11.8 px off without flagging it, so the frame conflicts (navigator work) |
| #1145 | **Affects `b_ring_saturated_plateau`, held as a known error.** A saturated ring edge pulls the answer 0.06 px farther out across the edges than the same frame unsaturated, about five times the reported sigma (navigator work) |
| #1152 | **Affects `a_ring_low_opening`, held as a known error.** At a low ring opening a narrow gap is resolved only near the ansa, yet both its edges are fitted along the whole arc, and the answer lands 0.84 px off at 16 sigma (navigator work) |
| #1164 | **Affects no current scene.** When lost 8-line slices of zero data cross a body's disc, the disc correlation does not exclude them or flag the answer. It now lands 0.020 to 0.046 px off on `gossi_ict_moon`, including the six seeds whose lost band crosses the disc, where the same scene once held it up to 0.91 px off. Exclude missing rows from the correlation, or flag the answer when they cover enough of the disc |
| #1204 | The simulator cannot draw Cassini's compression-truncated lines (alternating lines that lose their ends, 19% of the NAC frames with loss) or lost-line groups of varying size at any start line; `coiss_nac_stars_moon_lost_lines` uses fixed 7-line groups as the nearest form |
| #1205 | **Done (PR #1212): fitted on 14 raw LORRI frames of Io, Europa, Ganymede, and Callisto at 3 to 16 ms (residuals at most 0.09 ms, side ratios 1.090 to 1.093), the entry now gives the larger transfer time (12.17 ms) to the rows above a source and the smaller (11.15 ms) to the rows below, and states a 0.63 ms extra integration time (0.70 ms with four faint Charon frames added) that every charge the frame collects follows.** LORRI's catalog frame-transfer smear timings are about 10% long and on the wrong sides against lor_0034849319 and the instrument paper; `nhlorri_moon_raw` sets measured timings itself |

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
| #915 | **Now tracked through its slices, each its own issue: #1005 to #1030, #901, #923, and #310, in Phases 1b, 2, 2a, and 3b, and the navigator camera blocks (#1029), which the navigator config work in `PROGRAM_PLAN.md` carries.** It closes when its slices close. Restructure below the core: typed specs (begun in #900), one body path (begun in #901), and smaller stages |
| #1020 | **Moved to Phase 3b.** Build the simulated-image editor from per-block panels driven by the typed scene's field metadata (#1024) |
| #923 | **Moved to Phase 1b.** Duplicate classifier, test-only module, missing fixtures, and codenames |
| #902 | **Closes with #1020 (Phase 3b).** 26 scene keys without editor controls |
| #947 | **Closes with #1020 (Phase 3b).** The simulated-image editor shows 0 for parameters the renderer fills from the detector block or another key |
| #967 | **Closes with #1020 (Phase 3b).** The simulated-image editor's controls can build scenes that validation refuses: ranges wider than the rules, conflicting settings, settings left behind by a camera switch, and duplicate names |
| #956 | **Fixed by #1024 and #1026**: one default name, read on both sides. Unnamed simulated bodies get different default names in the renderer and the navigator |
| #856 | **Done (PR #1123).** The simulated-image editor segfaults intermittently (reopened after the fix in PR #917); recheck it against #1020's structure. The cause was in the tests: each destroyed its own Qt application, after which PyQt no longer notices Qt deleting its own objects (the segfaults), and some left windows whose timers fired in later tests (the aborts), so the UI tests now share one application and delete their windows, and the editor's debounce timer is deleted with its window. |
| #1000 | **Done (PR #1123).** The editor's null-keys test can see body controls that are pending deletion, so it fails intermittently in CI; collect the controls only after pending deletions run. The cause was #856's: PyQt took a combo box's line edit for a deleted widget, so #856's fix removes it. |
| #1119 | **Affects no current scene.** The editor never deletes replaced tab pages, so its memory grows with every rebuild |
| #1120 | **Affects no current scene.** A scene refused only while it renders, such as overlapping bodies with no range, opens a modal dialog on each editor redraw |
| #1122 | **Affects no scene; test scheduling.** With one Qt application per test session, the tests may no longer need `--dist=loadfile`; check on CI before dropping the rule |
| #957 | **Done (PR #1046).** The `sqrt_lut` detector module has no entry in the simulator API reference; after #1005 its path is under `render/` |
| #816 | The developer guide's detector quantization list omits `ls8b` and `contour_8bit` |
| #1052 | **Affects the report only.** The simulator report describes the disc correlation's old half-pixel window and the bias a haze puts on it; rewrite it in the end-of-stack regeneration, after the disc refinement work (#791) lands |
| #914, the rest | User chapter around a complete scene, developer guide split, and one schema reference. The schema reference page is generated from the typed scene's field metadata by #1024. |
| #310 | **Moved to Phase 3b.** Advanced by #1005 (package-level import guard) and #1026 (the navigator view as types); its own change moves the truth attributes off `ObsSim`. Structural information-boundary guard |
| #1182 | Survey the simulator's and navigator's slow routines (the relief walk and march, the density-wave ring ramp, the turned outline, the disc correlation's rotation search, and others found by profiling) for compiling with numba; decide the accuracy rule (bit for bit or a tested tolerance), platform and Python-version support, and a fallback |
| #1184 | **Done (PR #1185).** Committed baselines and contact sheets drifted from main's own output by up to 0.008 px (below the 0.02 px check); regenerated |

Each phase updates the docs for what it changes, with per-camera values on the instrument pages
after #924 (done). This phase finishes the guide split once the schema has settled. The
restructuring is placed earlier, where it is cheapest: Phase 1b while the baselines are still,
Phase 2 beside the body fixes, Phase 2a before scenes are written, and Phase 3b before the physics.

Code-structure critique items not taken, with the reason for each:

- **Making the validator's private primitives public (critique step 4, O8).** #1024 deletes the
  hand-written checkers, so publishing their primitives first would be work thrown away.
- **The two private `nav_model_stars` helpers used outside their module (O8).** #1026 replaces the
  navigator side's dict reads; recheck after it.
- **`realism/noise.py` at 984 lines (F2).** It is under the limit. The rule in `CLAUDE.md` splits it
  by concern when Phase 6's work would push it over; no issue is needed before then.
- **The parallel `full_render.py` and `verdict.py` modules (D6).** #1020 rebuilds the editor around
  them and decides whether one module serves both.
- **The critique's layout names `artifacts/registry.py`, `forward/frame.py`, and
  `schema/nodes.py`.**
  The artifact registry stays one module (`artifact_modes.py`), `stages.py` keeps `SimFrame`, and
  the schema core is the top-level `spindoctor/schema/` package (#1021), which the navigator config
  will share once its typing (#1029 and #1032) lands.

#856, #1000, #957, and #816 can land whenever their fixes are ready.

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
- #78, craters drawn with the CraterMaker package: deferred by priority; the simulator's own crater
  field serves navigation testing.

## Dependencies at a glance

```text
Phase 0 -> Phase 1 -> Phase 1b -> Phase 2 -> Phase 2a -> Phase 2b
Phase 2a -> Phase 3 -> Phase 5
Phase 2a -> Phase 3b
Phase 2b and Phase 3b -> Phase 4a, 4b -> Phase 5 -> Phase 6 -> Phase 7
Phase 4a, 4b -> Phase 4c-4f (parallel with Phase 5)
Phase 3b runs alongside Phases 2b and 3; Phase 4 waits for both Phase 3b and Phase 2b.
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
- **Restructuring delays the physics.** Phases 1b, 2a, and 3b add about 25 issues before Phase 4.
  Mitigation: each is pixel-neutral or moves only what it names, so each lands on the baselines'
  evidence alone. Phase 3b runs alongside Phases 2b and 3. Phase 4 then adds keys, modes, and
  ring kinds at one place each.
- **A refactor hides a physics change.** Mitigation: every pixel-neutral issue requires the
  integration tier to reproduce with nothing regenerated. The work that touches the most code
  lands in Phase 1b, before the baselines start moving. Phase 3b lands after Phase 2's moves and
  before Phase 4's, when renders are stable again. A pixel-neutral PR (Phase 3b, #1012) is rebased
  onto main after the last merge that regenerated baselines, and the integration tier is rerun on it
  with nothing regenerated, before it merges.
