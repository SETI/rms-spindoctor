# Simulator Realism and Fitness Critique

**Date:** 2026-11-06
**Baseline:** `origin/main` @ 01b04002 ("refactor(sim): name artifact occurrence rates rates, not
incidence (#879)").
**Question:** how realistic are the simulated images, can the simulator measure navigation accuracy,
is it enough to write new scenes or must the simulator change, is the simulator isolated from the
navigator, and are its code, tests, and documentation in good shape?
**Method:** fourteen independent audits, each by a separate agent working read-only in a worktree
of the baseline, followed by two adversarial reviews by agents that wrote none of the audits. The
areas were: instruments, filters, and detector defects; objects and radiance; realism tests and
calibration; the navigation scene catalog; sim/navigator isolation; ring images (three audits:
the committed scenes against seven library frames, 32 Cassini NAC and WAC frames drawn from the
full tour index, and the rings of Uranus, Neptune, and Jupiter); code architecture;
documentation; production values that rest on simulation; cost and determinism at full frame
size; measured test coverage with mutation checks; and rendering mechanics per object type. A
third adversarial review covered the outer-planet ring audit.
Every load-bearing claim was verified against the source, most by a probe render or a single
navigation. Claims about real cameras that rest on general knowledge rather than a file are
marked as such. Where a reviewer overturned an auditor, the reviewed result is the one given
here; section 14 lists what changed. This is a frozen snapshot; nothing in it is maintained.

---

## Verdict

**The simulator is a careful, well-built regression instrument for the navigation techniques on
one idealized camera. It is not yet an accuracy instrument for any camera, and the production
confidence layer that was fitted on it inherits that limitation.**

- **The core is sound.** Stars land on their truth position to 1e-4 px at every pixel phase,
  every object family shares one pixel convention and one roll pivot, the stage order is
  physically right, renders are deterministic and reproduce bit for bit, the truth/idealized key
  partition is default-deny and complete, and the old star-list leak is gone. The ring renderer,
  the relief field, and the haze model are better than the scenes that use them.
- **The scenes test the easy case.** All 64 catalog scenes are Cassini NAC at a 1.0 s exposure on
  128-360 px frames, with planted offsets of at most 3.6 px (63 of 64) in mostly one quadrant;
  real operator-verified NAC offsets in the library have a median of 16 px. 63 of 64 leave the
  instrument defect chain off. No scene has stars with a body, rings with a body, or a body
  filling or overflowing the frame. Every ring scene is a whole, uniform, face-on-ish donut on
  black sky, which resembles no Cassini ring frame.
- **The navigator's model is drawn by the renderer.** Simulated frames never run the production
  `NavModelBody`, `NavModelRings`, or star catalog path; the `NavModel*Simulated` siblings take
  over and share the renderer's geometry, shading, PSF, ring projection, and star-flux code.
  About 40 of the 64 scenes are the navigator's own template plus a rigid shift and noise. Body
  and ring "accuracy" from the catalog is self-consistency.
- **Radiance has no physical anchor.** Every body is an albedo-1 surface at a fixed per-instrument
  well fraction, with no solar-distance or filter term. Relative to bodies, simulated stars are
  about 34x too bright on the WAC and 17-30x too faint on LORRI at Pluto and Arrokoth and on
  Voyager beyond Saturn. The simulator has no concept of a filter, though 33% of NAC, 66% of WAC,
  and 54% of Galileo frames use a non-clear one. Rings render about 4x too faint against Saturn.
- **The realism checks cannot fail and are in-sample.** No realism comparison has a threshold; a
  LORRI PSF 3x too wide passes and scores better. The PSF and hot-pixel "matches" were tuned on
  the same 75 frames that report them. No measurement links simulated accuracy to real accuracy;
  where both are measured, simulated star fits are 4-5x tighter and ring fits about 20x tighter
  than real ones.
- **Production confidence rests on it.** One simulated campaign (seed 20260718) set every
  per-technique confidence coefficient except Titan's, both tier boundaries, and the 0.35 success
  threshold; every covariance floor is simulation-derived or uncalibrated, and the floors also set
  each technique's weight in the fused offset. That campaign drew every catalog body at half its
  intended size (a newly found generator slip), rendered NAC only with the chain off, and ran on
  renderers since found broken. Where real evidence exists it disagrees: RingEdgeNav is accepted
  wrong on 5-56% of real frames by resolution band while the simulation rates it 0.90 successful.
- **The simulator can already reproduce the hard real failures**, once realistic scenes are
  written. A dense Keeler-gap profile at real scale made RingEdgeNav 29.6 px wrong along the edge
  at a claimed 0.24 px sigma (the #755 failure); an unmodeled ringlet 120 km outside the A-ring
  edge produced a 40 px wrong lock at high confidence; matched unlit and low-opening WAC scenes
  gave 5-7 px errors at sigma below 0.1 px. Much of the gap is scene authoring, not physics.
- **Some cases need the simulator to change first.** A body larger than half the frame and off
  center is clipped to a square or vanishes (24% of real ring frames at most, and the commonest
  real limb geometry). A planted roll turns a body's shape the wrong way and its lit side not at
  all. The planet's shadow on the rings is a constant-longitude wedge that turned a matched
  success into total failure. Edge-on and equinox rings render nothing. Voyager's four cameras
  share one key with the wrong plate scale.
- **Code and documentation are good at the core and accreted below it.** The code is a
  dict-driven renderer whose validation runs only on YAML load and save; null body keys pass
  validation and then crash the renderer. The developer guide is one 3,784-line file with three
  drifting copies of the schema and 10-15% lab-notebook content; the user chapter does not show a
  newcomer how to build a scene.

**Answer to the operator's question:** both. Write the authorable scenes now, because they
immediately expose behavior the catalog hides, and fix a short list of simulator defects before
any body, ring, Voyager, or twisted-frame result can be called accuracy. Then re-derive the
production confidence layer on real frames, using simulation only for what it can establish.
Section 11 sorts every gap into those bins; section 12 gives the order.

---

## 1. What the simulator is, and what it does well

The image simulator (`spindoctor.sim`, about 18k lines) renders synthetic frames of stars, bodies,
rings, and haze through a detector model, from operator-authored geometry rather than SPICE, so
that every frame's true pointing offset is known. A simulated frame reaches navigation through
`ObsSim` (`obs/obs_inst_sim.py`), which renders in memory into a dummy snapshot (a flat FOV, fake
kernels) and passes the navigator a filtered view of the scene, `nav_params`.

What is done well, verified:

- **Placement is exact.** Star centroid bias is 0.0000 px at pixel phases 0, 0.25, 0.5, and 0.75,
  at oversample 1 and 4, with and without roll; flux is conserved through deposit and downsample.
  Stars, bodies, rings, the distortion center, and the stray-light center all use the
  pixel-corner convention and rotate about one pivot. Body anti-aliasing matches the navigator's
  template exactly at an integer center; ellipsoid flux is within 0.01% of 2/3 pi R^2.
- **The stage order is physically right and matches the developer guide:** occlusion, then smear,
  distortion, PSF, ghosts, and stray light, then the box downsample, then the detector (dark,
  Poisson, hot pixels, saturation and bloom, cosmic rays after the cap, read noise, gain, bias,
  quantization), then telemetry.
- **Determinism.** SHA-256 sub-seeds are derived per effect name, so adding an effect never
  reseeds another. With the SIMD pin, all 64 renders are bit-identical within and across
  processes, and every committed PNG and baseline reproduced under a different package set. The
  one realism number that would not reproduce (LORRI EE50) was traced to a commit inside #849
  that changed binned rendering after the summary was written; the summary is stale, the
  rendering is deterministic.
- **The information boundary.** `build_nav_params` copies only keys classified as idealized; every
  schema key is classified, with an import-time completeness check and a test that iterates the
  truth inventory. `nav_override` is validated to idealized keys. The old `sim_star_list` leak is
  gone and pinned absent by two tests. No technique, the ensemble, or the orchestrator branches on
  "simulated", apart from not recording pointing. Ring features default to non-navigable, so
  structure the navigator never learns about is easy to render.
- **Honest provenance.** Catalog values carry TUNED, RETAINED, MEASURED, or interim tags with
  dates and cohort sizes; each retained-zero cosmic-ray rate records its measured value and why it
  was not adopted; the report labels clean numbers as a floor, not accuracy; the calibration
  record discloses its chain-off basis.
- **Physics worth keeping.** Single-scattering ring photometry with the lit/unlit inversion,
  Henyey-Greenstein per feature, `exp(-tau/mu)` transmission, and per-pixel depth compositing; a
  sphere-consistent relief field with an exact shadow march and low degrees removed so it cannot
  alias a translation; a haze model with explicit symmetry-breaking keys aimed at the navigator's
  own assumption; a broad, data-driven artifact registry (31 modes) with per-instrument
  availability.
- **The ring renderer scales.** A 223-feature profile renders at 1024 x 1024 in 1.2 s, and an
  `a_px` of 400,000 is numerically fine.
- **Tests.** 2,187 simulator unit tests pass in 24 s and execute 85-100% of every `sim/` module;
  the 190-test simulator integration tier passes in 6.4 min. Six of eight deliberate mutations of
  numerically important lines were caught.

---

## 2. Realism: instruments, filters, and detector defects

### 2.1 What the tests and calibration actually exercise

Every navigation scene and the whole confidence-calibration campaign run on one camera in one
configuration: Cassini NAC, `exposure_sec: 1.0` (the reference exposure), mostly with the
navigator's own Gaussian PSF (`optics.psf: {match_navigator: true}` on 20 scenes) or no PSF at
all (41 body and ring scenes), and with `instrument_defaults` on in exactly one scene
(`star_confounder/saturated_star.yaml`). No scene uses ghosts, distortion, binning, compression,
hot pixels, cosmic rays, or radiation transients; one uses stray light and one uses smear. The
WAC, Voyager, Galileo, and LORRI chains are unit-tested for mechanics and never navigated in CI.

For scale, the Cassini index (COISS_2010-2090, 14,005 frames; an early-tour sample, not the whole
mission) shows anti-blooming on for 51% of NAC frames, the 12-to-8-bit table on 48%, summing on
37%, missing lines on 9% of NAC and 29% of WAC frames, and a non-clear filter pair on 33% of NAC
and 66% of WAC frames. Galileo's tour (GO_0017-0023, 4,003 frames) is 74% ICT-compressed and 54%
non-clear.

### 2.2 Radiometry

- **No physical anchor (high).** Electrons are `clip(signal, 0, 1) * 0.5 * full_well_e *
  binning^2 * exposure/exposure_ref` (`detector/chain.py:213-226`). Signal 1.0 is half a well at
  the reference exposure on every camera; there is no per-body albedo key, no heliocentric
  distance, and no filter throughput. Real DN per unit I/F differs by orders of magnitude across
  filters, cameras, and targets. The simulator report's own known gap 1 records the WAC sky noise
  8x low for this reason.
- **Star-to-body ratio is right only by accident.** Comparing each instrument's star zero point
  with its own body scale through solar-flux physics (reviewed and corrected; R > 1 means
  simulated stars are too bright relative to bodies): NAC at Saturn 1.14, WAC at Saturn 34,
  Galileo at Jupiter 0.89, LORRI at Jupiter 2.3, LORRI at Pluto 0.058, LORRI at Arrokoth 0.034,
  Voyager at 1 s at Jupiter, Saturn, Uranus, Neptune 1.18, 0.35, 0.087, 0.035 (times the exposure
  in seconds). WAC star navigation is flattered; LORRI encounter and Voyager outer-planet star
  navigation is under-represented.
- **A hidden [0, 1] clip ahead of the detector (medium).** Anything that pushes signal above 1
  (stray light on a lit disc, a ghost, Lommel-Seeliger limb brightening, a bright ring) is
  flattened at half the well and never flagged. A body saturates only all at once, and only when
  the exposure reaches twice the reference.
- **Voyager exposure (unfiled).** The vidicon emits I/F, so a body's I/F correctly ignores
  exposure, but the star's I/F grows with exposure and the body's DN, SNR, quantization, and
  saturation never respond to it.
- **Star zero points are unmeasured, and the navigator reads the same number** (`config_400`
  holds the catalog value divided by gain), so a wrong zero point can never show in simulated
  results. Star flux has no color or filter term (#882).

### 2.3 Filters

The simulator has no concept of a filter: no scene key, catalog table, or stage reads one. (#883
describes a scene `filter` key "added by the fix for #329"; no such key exists.) Throughput, PSF
width, star flux by color, body albedo and limb darkening, haze optical depth and altitude, and
polarizer behavior are all filter-independent (#881-#885). Methane-band frames, where the planet
is dark and the rings or haze bright, have no simulated counterpart. Importance: high; a third of
NAC and two thirds of WAC frames sit outside the regime the simulator renders.

### 2.4 Optics

- **PSF.** A Gaussian core plus Moffat wing per instrument, field- and wavelength-invariant. NAC
  is tuned against real stars (EE50/EE80 about 0.9/1.7 px), the WAC on 9 cutouts from 2 frames;
  Voyager, Galileo, and LORRI are interim (LORRI's wing too weak, #848). The navigator's
  `star_psf_sigma` is a 3.0 px placeholder on Galileo, Voyager, and LORRI against simulated
  kernels of 0.8-1.1 px (#342).
- **Ghosts** are a fixed translation of the whole frame; real internal-reflection ghosts are
  point-reflected through the optical axis (general knowledge). No defaults, no scene.
- **Stray light** is an authored linear ramp or radial bump with no source; bodies outside the
  frame contribute nothing (section 4.4).
- **Smear** is a straight constant-rate line, authored per scene, with no link to SPICE rates or
  exposure (#151, #333).
- **Distortion residuals** are measured from star-field FOV analysis for Cassini; one Voyager WAC
  block stands for four cameras (#806). Applied only under `instrument_defaults`.
- **Dust donuts** are drawn per scene, not per detector, so they move between scenes on one
  camera (the analog of the fixed #331).

### 2.5 Detector and electronics

- **NAC saturation disagrees with the navigator (#854).** The simulated well is 110000/30 + 20 =
  3687 DN; the navigator flags saturation at 4095, so clipped simulated stars are never flagged.
  The catalog's "bright" star field is heavily saturated (39 pixels at the clip) and its "dim"
  field has one clipped star; scene comments and the report still say 100-150 DN and
  1000-2000 DN.
- **Bloom** spreads the excess over a fixed 2L+1 window and discards charge on already-saturated
  pixels: a pixel 50x over full well keeps 18% of its charge and the streak never exceeds 9 px.
  There is no anti-blooming state. LORRI contradicts itself: the registry disables bloom, the
  catalog sets `bloom_length: 2`, and `instrument_defaults` applies it.
- **Cosmic rays** are rate 0 on every chain, each with a documented measured rate (#330). When
  enabled, every hit deposits at least the full well, so the navigator's saturation mask removes
  it trivially. Real hits are mostly below saturation and look like faint stars, the main false
  source for the star matcher. Importance: high for star techniques.
- **Hot pixels** are a fixed per-detector map (fixed in #331) with a tuned fraction, but their
  amplitude does not scale with exposure; long-exposure star frames get too few star-like hot
  pixels and short ones too many.
- **Gain states:** the WAC catalogs one state though 20% of real WAC frames use 12 or 95 e-/DN;
  Galileo defaults to 40K though 100K is the most common tour state.
- **Opt-in and never in a scene:** the 12-to-8-bit table (a pure square-root compander stands in
  for the real piecewise table), 8LSB, uneven-bit ADC, the 2 Hz banding, dark ramp and RBI,
  frame-transfer smear (LORRI, on by default there), summation and binning, and the generic
  8x8 DCT codec (no frequency weighting; Galileo ICT is mostly what the real tour used). The DCT
  rebuild is not clipped to the ADC word (#850).
- **Telemetry loss** has 13 modes; only missing and partial lines appear in any navigated scene.

### 2.6 Per mission

- **Voyager:** one `vgiss` key for NA and WA on both spacecraft (#806); the plate scale is the
  raw NA value 1.910"/px, while navigated GEOMED frames are 1.617"/px (NA) and 12.1"/px (WA), so
  background-star density is off by 1.4x and about 40x (it affects only the sky-field star
  count); reseau scars sit on a synthetic lattice, not the real positions; beam bend moves pixels
  along one axis with a sign taken from noise inside bright regions; the vidicon noise keys are
  inert (#804).
- **Galileo:** radiation, comb banding, and stitch surrogates exist, all opt-in; no star-frame
  PSF calibration; ICT compression absent in practice; the real negative frames' glow is not
  modeled.
- **LORRI:** frame-transfer smear is a step at the column centroid rather than a cumulative
  integral, and there is no `_sci`-style calibrated product mode though the navigator accepts
  both `_sci` and `_eng`.
- **Cassini:** the best-covered camera, but only in its clear-filter, unsummed, 12-bit,
  anti-blooming-agnostic configuration.

---

## 3. Realism: objects and radiance

- **Night side (defect, unfiled).** `DARK_SIDE_ILLUM_STRENGTH = 0.01` floors every pixel of the
  visible hemisphere, night side included (`ellipsoid_geometry.py:39,365`; mesh and non-Lambert
  paths alike). On `mimas_phase_120` the night side reads 18.5 DN above a sky of 20 ± 4 DN, a
  4.6-sigma edge per pixel along the whole dark limb. Production's template is zero there (oops
  `lambert_law` masked, filled with 0). The simulated template carries the same floor, so it
  largely cancels: with the floor removed from both sides, a probe navigated no worse (at 120°
  phase the fused error went from 0.28 px to 0.001 px). It is a realism defect of unmeasured
  effect, not a demonstrated optimism. Real night sides are black apart from directional
  planetshine.
- **Photometric clip (defect, unfiled).** Lommel-Seeliger and Minnaert are clipped to [0.01, 1]
  (`photometry.py:179`); at 60° phase, 33% (Lommel-Seeliger) and 21% (Minnaert k = 0.7) of the
  lit disc sits on a flat plateau, removing the limb brightening these laws exist to plant. The
  calibration campaign draws them on 40% of body scenes.
- **Lambert against Lambert.** Of 64 scenes, 1 sets a non-Lambert law, 1 relief, 1 an albedo
  texture, 0 craters, and 0 an opposition surge. Hapke is absent, and phase has no darkening
  beyond geometry.
- **`rotation_tilt` stretches the body instead of tilting it** (a sphere at 60° renders twice as
  tall), and textures put the pole on the image axis, so giant-planet belts are always straight.
  Unused, unfiled.
- **Craters** are image-plane circles with no foreshortening or shadows, used by no scene.
  Irregular meshes cast no shadows on themselves and refuse photometric laws.
- **Relief** is a single-correlation-length Gaussian field, not fitted to real limbs (#875 open).
- **Shadows between objects** are missing: no ring shadow on the planet; the planet's shadow on
  the rings is an authored constant-longitude wedge (the real one is a strip bounded by lines
  parallel to the anti-sun direction, half-width asin(R/r) in longitude); moon shadows on a planet
  are hand-placed textures; no eclipses or planetshine.
- **Haze** is physically motivated but wavelength-blind (#881), and the real Titan campaign
  already shows it optimistic (0.08 px simulated sigma floor against 0.70 px rms on real frames).
- **Stars:** color is ignored; the navigator's catalog is exactly the scene's star list (exact
  positions at pixel centers, exact magnitudes, no proper motion, no conflicts, no smear, no
  saturation flag); only 3 of 64 scenes perturb it. The background star-count law looks 3-10x too
  sparse (general knowledge) and no calibration star family uses it. Aberration and proper motion
  are never exercised.
- **Missing objects:** a body the navigator does not know about cannot be planted (`navigable` is
  rejected on bodies); no sub-pixel moons; no frame-filling or overflowing planet in the catalog.
- **Off-frame light:** none; see 4.4.

---

## 4. Rendering mechanics

Placement in the frame interior is exact (section 1). The defects are at frame edges and in
partly covered pixels.

1. **A body larger than the frame and off center is truncated or vanishes (high, unfiled).** Each
   body and halo is drawn into a frame-sized cache centered on the frame, then moved with
   `ndimage.shift(..., cval=0)` (`body.py:543,912`). A 120 px body centered 40 px above a 64 px
   frame renders nothing (21 rows should show); 80 px at -25 loses 30% of its flux. No scene check
   refuses it. The commonest real limb geometry, a planet larger than the NAC frame with its limb
   across a corner, cannot be simulated; a scene that tries gets a spurious failure, never a quiet
   bias. Fixing the calibration generator's half-size slip (section 7) would push its limb family
   straight into this defect, so the renderer must be fixed first.
2. **A planted roll turns a body the wrong way (medium, unfiled).** The center rotates by +roll,
   but `rotation_z += roll` rotates the sampling grid, so the pattern turns by -roll: at +10° roll
   a star and the ring pattern turn +10° and an elongated body's major axis -10°. The lit side does
   not turn at all (#825). `test_sim_rotation.py` checks only stars and rings. The bodies of #644
   and #825 state the sign wrongly.
3. **Ring-edge anti-aliasing is applied to optical depth, not intensity (medium, unfiled).** A
   partly covered pixel renders as a thinner ring, so each sharp edge sits 0.28 px outward at tau 2
   (oversample 4), up to 0.39 px at oversample 1 with a pixel-phase ripple, and up to 0.67 px at
   tau 5. It cancels for a ringlet seen with both edges, which is why the shipped scenes hide it; a
   single visible edge carries the full bias. At oversample 4 the edge is also blurred twice.
4. **The PSF and smear zero-pad the frame (medium).** A frame-filling body's edge row reads 72% of
   the interior and its corner 52% under the NAC kernel; 12 px of smear makes the edge 53%. Light
   from outside never enters: the PSF wing of a bright limb 2 px off the frame (0.8% of its
   brightness in row 0), most of a star centered 0.2 px off the edge, and ghosts of off-frame
   sources.
5. **Bilinear planting (low).** Every body is rendered at the frame center and shifted bilinearly,
   softening a fractional-offset limb by about 20% and biasing a template fit by at most 0.04 px.
6. **Smaller items:** a near body over a far body or ring leaves a dark one-pixel seam (0.145
   where blending gives 0.63); the topographic path at oversample 4 gives small bodies up to 9%
   flux error and a 0.1 px lit-centroid bias; distortion resamples star splats without conserving
   flux (negligible at catalog values); the body anti-aliasing ramp in `ellipsoid_geometry.py` is
   inert and misdocumented; reseau patches are square though documented as discs.

---

## 5. Realism: rings

Rings got three audits at the operator's request: the seven committed ring scenes and twelve
calibration ring scenes against seven library NAC frames; 32 Cassini frames (16 WAC, 16 NAC)
drawn from the full tour index; and Voyager, Galileo, and LORRI frames of the rings of Uranus,
Neptune, and Jupiter.

### 5.1 The real ring-frame population

Joining the PDS3 index with the ring and Saturn geometry tables for all 116 tour volumes gives
177,540 frames with the main rings or the F ring in view:

| property | share |
|---|---|
| WAC | 37.5% (16.6% of ring-discipline frames) |
| Saturn in view | 38.6% (70.8% of WAC ring frames) |
| unlit face | 43.0% |
| edge-on, \|B\| < 1° | 12.9% |
| phase > 120° | 19.2% |
| equinox, \|B_sun\| < 1° | 4.0% |
| summed | 29.3% |
| non-clear filter | 42.6% |
| WAC at 30-300 km/px, main-ring edges blending | 70% of WAC |

Half of these are not ring observations; the rings are clutter in them.

### 5.2 What the scenes show

Every committed and campaign ring scene is one or two uniform annuli on black sky, centered, with
2-4 edges, no PSF (edges drop in one pixel), opening 50-90° (4 of 7 at 90°), lit side, phase 0,
NAC, and no planet. Real frames show 3-73 radial transitions each, straight or gently curved
edges across 1024 px (never a whole ellipse), mostly low elevation, often the unlit face, and the
planet or its shadow. Every committed ring result is a trivial 2-D lock at about 0.01 px.

### 5.3 What realistic ring scenes do

Realistic ring-only frames are authorable today with YAML alone. A 223-feature Keeler-gap scene
built from a real frame's radial profile at 0.34 km/px and the real unlit geometry validates and
looks broadly like the real frame (it lacks the gap-edge rim, glow into the gap, wake texture,
and azimuthal streaks). Navigated:

- **Along-edge wrong lock (#755).** On a near-straight edge near the ansa at B = -12.8°,
  RingEdgeNav reports success at high tier, 29.6 px wrong along the edge, with a largest-axis
  sigma of 0.24 px; across-edge error is 0.01-0.12 px. An offset sweep gives 1.4-2.8 px along-edge
  errors at sigma about 0.25 px. This is the failure #755 measured on 1,041 real frames, and no
  committed scene reaches the regime.
- **Unknown structure.** Only `navigable: true` features reach the navigator (2 of 105-110 in
  every demonstration). Dense unknown A-ring structure and four unmodeled F-ring-like strands did
  not pull a strongly curved edge fit; an unmodeled narrow ringlet 120 km outside the A-ring edge
  produced a 40 px wrong lock at confidence 0.89; a hidden eccentricity (model circular, truth
  ae = 30 km plus an m = 7 term) came out as a 0.96 px error, 15 sigma beyond the claimed
  uncertainty.
- **Matched WAC and NAC frames** (planted offset 1.3, -0.7 px): a whole-system lit WAC frame with
  a correctly sized planet navigated to 0.12 px; an unlit dense WAC arc without the shadow wedge
  gave a (-1.31, -5.39) px error at sigma 0.08; a low-opening WAC frame gave (-6.04, 3.39) px at
  sigma 0.03-0.04; the same unlit WAC frame with the wedge failed outright (all techniques
  spurious).
- **Profiles from real frames carry that frame's pointing error.** N1557141567's profile sits
  about 125 km from catalog radii (10.8 px); uncorrected it navigates 10 px off, corrected
  0.36 px. Profile-derived scenes must be built from navigated geometry.

### 5.4 What the ring simulator cannot do

- **The simulated ring model masks neither planet occlusion nor shadow** and fixes reliability at
  1.0; production `NavModelRings` masks both and filters features for resolvability. Any scene
  with a planet tests a ring model production does not use. (One audit's Saturn-plus-rings error
  figures were measured with Saturn drawn at half size; the mechanism stands, the magnitudes need
  re-measuring at low opening.)
- **Planet clipping** (section 4, item 1) affects every close planet-plus-ring frame: Saturn's
  radius exceeds half the frame in 56% of WAC and 75% of NAC ring frames with Saturn in view, an
  upper bound of 24% of all ring frames.
- **The shadow wedge** cannot fit a real shadow edge and turned a matched success into failure.
  No ring shadow falls on the planet.
- **Orthographic projection** misplaces whole-system WAC geometry by 7-23 px RMS (fitted to real
  backplanes); the radial scale cannot vary across a frame (real: 64-160 km/px in one WAC frame).
- **Features are in pixels** at one scale; nothing reads `config_310_saturn_rings.yaml`, and the
  simulator accepts static m >= 2 modes only.
- **Brightness:** the B ring to disc ratio is 0.149 simulated against 0.59 real (a default ring
  albedo against bodies with no albedo).
- **Absent:** vertical thickness and inclination (B = 0 or B_sun = 0 renders nothing: 12.9% and
  4.0% of frames), dusty envelopes (F ring halo, E and G glow), ring opposition surge, wake
  texture, per-feature azimuthal structure (arcs, clumps, kinks), partial arcs.
- **Partial knowledge:** mode amplitudes, edge waves, and widths on a navigable feature are passed
  to the navigator, so they are known, never model errors. Only `orbit_error` (semimajor axis,
  eccentricity, pericenter) can be withheld. Other hidden shape errors need an undocumented decoy
  pair (a navigable zero-tau copy beside a hidden true copy), which works but is untested.
- **An F-ring-like distractor** is renderable as a non-navigable feature the navigator never
  receives, which is the legitimate test for the F ring; no committed scene does it. The F ring is
  not in the navigator's ring catalog and must not be added.

### 5.5 Rings of other planets

The audit read 15 Voyager 2 Uranus frames with ring content, 5 Neptune frames, and 4 Jupiter
frames, all GEOMED through `ObsVoyagerISS`, and viewed each. Three (the Uranus dust-band frame
C2685219 and the Jupiter frames C1636819 and C2070817) fail to load with `SPICE(CKINSUFFDATA)`.

- **The navigator cannot navigate these rings.** `config_300_jupiter_rings.yaml`,
  `config_320_uranus_rings.yaml`, and `config_330_neptune_rings.yaml` are 0 bytes (#82, #81, #83,
  each a one-line issue). A catalog alone would not be enough: there is no feature type for a ring
  narrower than a pixel; the resolvability filter's per-planet `min_feature_pixels` (2.0 in the
  Saturn config) would, if copied, drop every Uranus ring but epsilon, which passes only barely at
  28 km/px; and the catalog accepts inclination modes that the code ignores, with no inclination
  in the base orbit. oops already carries the French et al. (1991) elements for all ten Uranus
  rings (`body.py:176-206`).
- **Real signal.** On narrow-angle 15 s frames at 10-28 km/px, the epsilon ring is about 6e-3 I/F,
  a per-pixel SNR of roughly 14-32; the other rings are 1e-4 to 7e-4 I/F (per-pixel SNR about
  0.2-3.8, lower bounds) and stand out only after averaging along the arc. On approach frames no
  ring reaches a per-pixel SNR of 0.4, and the high-phase narrow-angle frames show no narrow rings.
  (Measured with a crude twist-plus-shift ring matcher, not the navigator, on the three frames
  where it locked.)
- **Real pointing.** On those three frames SPICE pointing is off by 6, 65, and 224 px with up to
  0.5° of twist, and Voyager ships with rotation fitting off (`config_430_inst_vgiss.yaml:15`). A
  twist that size moves narrow rings by several pixels.
- **Simulator geometry works; photometry does not.** The simulator's orthographic ring projection
  matches oops geometry for C2676527 to 0.07 px rms, epsilon's ellipse included (registration to
  the real image itself is only the crude matcher's 1-2 px). But the vidicon chain has no exposure
  term, so 1 DN is always 7.84e-3 I/F and the sky floor 0.0118 I/F, against about 4.1e-4 I/F per
  raw DN (about 19x finer) and a sky of about 9.4e-4 I/F on the real frame. At a physical ring
  albedo the render has three distinct pixel values and the narrow rings come out as 1-DN dashes;
  with `instrument_defaults` on, banding dominates; continuous rings needed an albedo of 1.0, 20x
  physical.
- **Not expressible:** epsilon's width varying with longitude, inclination, Neptune's arcs (ring
  brightness can vary with longitude only for all rings at once), and the Jupiter halo's thickness.
- **Navigation of the matched scene** with only epsilon navigable gave about 1 px errors
  ((-0.97, -0.32) px, reproduced; one seed, about half of it along the arc, and the fused answer
  took RingEdgeNav at confidence 0.46 over RingAnnulusNav at 0.94). That exercises only the
  simulated ring model, never production.

No simulated outer-planet ring result says anything about real accuracy today. Real Uranus ring
navigation looks feasible on the narrow-angle 10-35 km/px frames once #81 adds a narrow-ring
feature type and the twist is fitted.

---

## 6. Isolation between simulator and navigator

- **Information: sound.** No truth reaches the navigator through `nav_params`, attributes, caches,
  labels, or the shared helpers. The consumer-side guard (`test_boundary_static_guard.py`) is a
  lexical name scan of `nav_model`, `nav_technique`, and `nav_orchestrator` only; it skips the
  navigator-facing methods inside `obs_inst_sim.py` (which read `instrument`,
  `instrument_config`, `fit_camera_rotation`, and `closest_planet` straight from `sim_params`;
  idealized today) and the four `sim/` helpers navigator models call (#310, whose body cites the
  critique at its pre-archive path).
- **Models: weak.** All four production models return nothing for a simulated observation
  (`nav_model_body.py:685`, `nav_model_rings.py:318`, `nav_model_stars.py:125`,
  `nav_model_titan.py:502`). The simulated siblings make their own choices: a limb only at
  diameter >= 100 px and phase <= 60°, fixed per-vertex sigmas, limb vertices on integer pixels,
  no star smear or conflicts, a 1% night side, no ring occlusion. None of the production catalog
  reduction, oops limb and terminator, ring catalog filter, or SPICE Titan geometry runs in
  simulation. #843 covers bodies only; #227 is the umbrella.
- **Model sharing.** Both sides shade through `sim/ellipsoid_geometry.py` and one mesh rasterizer;
  ring edges are placed at byte-identical positions by design; the PSF is absent or the
  navigator's own Gaussian on all but three scenes; catalog flux and rendered flux use the same
  color-free law. With integer offsets the noise-free render equals the template to 4e-4 of peak.
  About 40 of 64 catalog scenes plant no geometry error and no appearance mismatch.
- **The planted error is exactly what the solver fits:** a rigid translation plus roll about the
  frame center. Only 5 scenes plant a geometry error. `spk_parallax_body` has no stars, so its
  parallax shift cannot be told from pointing error and the test is tautological. No scene has
  stars and a body together, so cross-technique disagreement, the main real failure the ensemble
  exists for, is never tested.
- **Geometry is independent of oops** (dummy FOV), which is correct isolation, at the cost that
  light time, aberration, twist, mid-time, and frame chains cannot be represented.
- **The reader layer is never exercised:** simulated frames never pass through a file writer or a
  per-instrument `Obs*` class (label parsing, empirical star limits, LORRI `_sci`/`_eng`, GEOMED
  borders).
- **Calibrated-product path defects:** `ObsSim.star_max_usable_vmag` returns 30.0 for every
  calibrated instrument (`obs_inst_sim.py:226-229`), giving a V = 7 star a predicted SNR of
  8 x 2.512^23 = 1.27e10, exactly what the realism runner records; the method's own docstring
  warns against such a placeholder. A quantized calibrated sky probably zeroes BodyBlobNav's noise
  estimate (6.7e8 SNR), which is inferred, not traced. Only one catalog scene is calibrated, so
  the catalog is unaffected.
- **Docs overclaim:** the developer guide says the simulated side predicts "exactly the way the
  SPICE-backed models predict a real frame" and calls a planted-truth recovery "an exact statement
  about the navigation algorithms" (`dev_guide_simulator.rst:131-132, 274-282`).
- **Layering:** `sim` imports nothing from the navigator, but `nav_model` and `obs` import four
  `sim` modules, and scene validation imports the renderer.

---

## 7. Realism tests and calibration against real frames

- **No threshold, no baseline, no CI comparison.** "No pass/fail threshold is attached"
  (`divergence.py:9-11`, a recorded plan decision). The by-hand runner (`sim_realism.py`, about
  21 min) pairs each of the 75 library frames with one matched simulated frame and reports
  Wasserstein-1/IQR per figure of merit. The LORRI smoke test asserts only structure; with the
  LORRI PSF core 3x too wide or read noise 10x too high it still passes, and the wrong PSF scores
  better (EE50 2.33 to 1.04). Nothing reads the committed summary JSON except its writer.
- **In-sample.** The NAC PSF, WAC PSF, calibrated and Galileo hot-pixel fractions, and vidicon
  read noise were tuned on the same frames that report the match; nothing is held out.
- **The reference set** was chosen for navigation regression, not to represent cameras: 58 NAC,
  4 WAC, 8 Galileo (all negative or scattered-light frames), 3 Voyager (one key for all), 2 LORRI
  (both 4x4-binned). Filter is ignored in the matching.
- **The noise comparison is censored.** Simulated sky noise is below one quantization step, so the
  patch finder rejects most simulated patches; Galileo and Voyager have zero simulated samples yet
  are labeled "limited", and the NAC "medians match" compares a lattice-valued sample with a
  continuous one. #329's rationale (simulated sky noisier than real) has the direction wrong: the
  WAC simulated sky is 8x quieter.
- **The star-profile comparison is biased:** real cutouts are centered on catalog position plus
  sidecar offset without re-centroiding (#343), so real stars measure wider.
- **Matched twins share little with the real frame** (class, exposure, size, diameter, phase), so
  the dynamic-range and sky-level measures compare scene content. Saturation, bloom, reseau,
  compression, and smear are never compared; Voyager's real 1.4% line loss against 0 simulated is
  not mentioned.
- **No sim-to-real transfer check.** The library cross-check compares real navigation with
  operator labels only; the transfer watch is unarmed (#334). The technique diagnostics recorded
  on matched pairs say the simulation is optimistic: StarFieldFromCatalogNav median residual
  0.057/0.016/0.063 px simulated against 0.278/0.062/0.325 px real (NAC, WAC, LORRI);
  StarRefineNav scatter 4-11x tighter; RingEdgeNav per-edge DT RMS 1.46 px against 28.0 px.
- **The accuracy runs use a configuration the realism match never examined:** 1 of 64 scenes and
  0 of 30 sweeps use the realism-tuned chain.
- **The report and guide overstate.** The NAC dynamic-range table shows only the better exposure
  stratum (the 0.5-5 s stratum's p99 stretch is 57x short); the LORRI residual is 0.063 px, not
  "0.01-0.02"; the Voyager pair where the simulation fails and the real frame succeeds is omitted;
  "23 star-bearing frames" is 11; "ring-edge widths agree at the few-tenths level" is 0.7 px
  (23%); "real-frame transfer is measured" is not true; tier behavior "validated" on 46/75 tier
  agreement is too strong. The LORRI figures and paragraph are stale since #849.

---

## 8. The navigation scene catalog

| Group | n | What is asserted |
|---|---|---|
| algorithmic_invariants | 18 | per-axis error against truth at 1.0 px, roll at 0.3° |
| artifact_sweep | 4 | per-axis error at 0.5 px (missing and partial lines only) |
| atmosphere | 4 | expected status and tier only |
| expected_fail | 2 | failure asserted |
| model_mismatch | 4 | sweep floors and trends |
| mutual_event, multi_body, noise, phase, range | 13 | baseline only, or status; one at 0.5 px |
| regression | 7 | 3 at 0.1 px, the rest status or spurious flag |
| ring_system | 5 | expected status; one honest pin at 3.2 ± 1.0 px |
| smear_sweep, star_confounder | 7 | sweep floor; fused error at 0.5-1.0 px |

Plus an exact-match baseline of status, offset, and confidence rounded to 0.01 for every scene, and
a byte-identical committed render. Findings:

- **One instrument.** 63 `coiss_nac`, 1 `coiss_calib_nac`; all three randomized campaigns are
  NAC. Retargeting catalog scenes changes outcomes: the disc scene on Galileo and LORRI comes back
  conflicted (BodyBlobNav 4.0-4.6 px off; 1.00 px fused under defaults), and the six-star field
  fails on `vgiss`. Unexplained and unpinned.
- **Tiny, one-directional offsets.** 63 of 64 at most 3.6 px, 12 exactly zero, 47 of 52 nonzero in
  the +v/-u quadrant. Real library offsets: NAC median 16 px (90th percentile 36, max 60), Galileo
  near 290 px, Voyager 28-49 px. The wide sweeps go along +v only and assert nothing.
- **Frame size.** 128-360 px; no real size; most scenes fall back to a generic 50 px margin.
- **Twist.** One roll scene and one roll sweep, both with roll fitting forced on; every shipped
  instrument has it off, so the configuration real navigation uses never meets a twisted frame.
  With roll fitted, translation is reported about the extended-FOV corner (#434; no shipped run is
  affected).
- **No compound scenes:** no body filling (largest 0.78 of the frame) or overflowing the frame, no
  stars with a body, no rings with a body, though the library's commonest classes are exactly
  these.
- **Weak assertions.** Invariant tolerance 1.0 px per axis against measured errors of 0.00-0.27 px;
  19 scenes are never compared with their truth; of 24 expected-outcome blocks only one checks an
  offset (`relief_psf_stray_light` passes at 0.82 px); baselines pin wrong answers alongside right
  ones; per-technique failures of 7-36 px under a good fused answer go unasserted; the offset
  sweeps assert nothing.
- **Stale and inverted star numbers.** The report's star figures predate the 2026-07-20 zero point:
  measured today, the bright field's median error is 0.059 px (report: 0.005, "the most accurate
  of any technique") and the dim field's 0.014 px; the ranking is inverted.
- **No per-instrument accuracy number anywhere;** sweep results are not committed, so the report's
  figures cannot be regenerated from the repository.
- **BodyTerminatorNav never runs on the catalog** by design: it is a fallback skipped whenever the
  disc correlation succeeds, which on template-matched scenes is always. Forced, it is 17.8 px off
  (#223).
- **Thinly sampled:** smear (one star scene), saturation (one deliberate star), artifacts (lost
  lines only), exposure (always 1.0 s), low signal (read-noise steps only).

---

## 9. Production values that rest on simulation

Three simulated campaigns feed production values: confidence calibration (seed 20260718), Titan
planted truth (seed 20260725, constrained by an 82-frame real cohort), and the agreement-estimator
study (no shipped value).

- **What 20260718 set.** Every per-technique confidence alpha except Titan's
  (`config_510_techniques.yaml:19-55`), the 0.85 high-tier boundary, the 0.35 success threshold
  (`config_540_orchestrator.yaml:74,121`), and the limb (2.61 px), terminator (4.32 px), and
  StarRefineNav (0.27 px) floors. The NCC floors and size terms come from earlier simulated
  campaigns (2026-07-12 and 07-14); StarFieldFromCatalogNav's floor is an uncalibrated 0.0;
  Titan's 0.2 rests on real-frame reasoning. So every floor is simulation-derived or uncalibrated.
- **How far they reach.** The 0.35 threshold decides success or failure on every frame; the tier
  boundaries and floors set every tier; the ensemble weights each technique by inverse covariance,
  so the floors also set each technique's weight in the fused offset; the 2026-07-18 refit changed
  the primary technique on 3 library frames. `confidence_provisional: True` is written into every
  metadata document.
- **What it was fitted on.** NAC only, chain off, navigator-matched PSF on star families and no PSF
  on body and ring families, 128-220 px frames, offsets within ±4 px (70%) or ±10 px, an authored
  error mixture (#341), broken relief and mesh renderers (#625, #626; refit #793), a simulated body
  model that kept disc and blob features production drops (#874), and, newly found, **every
  catalog body at half its intended size**: `util/calibration/scene_gen.py:256` passes the
  apparent radius as `axis1`, which is a full width (a "radius" of 23 renders 24-25 px across).
  Every sibling generator uses `2*r`, so it is a slip. `km_per_pixel` is therefore 2x too large,
  the blob irregularity term saw twice its production value, the size alphas were fitted on
  half-size bodies, and the disc size term extrapolates about tenfold (not fivefold) to a 1000 px
  disc. The same full-width-named-like-a-semi-axis trap caught one of this audit's own scene
  scripts.
- **Real evidence disagrees where it exists.** RingEdgeNav: simulated base rate 0.90; a 131-frame
  real comparison recorded in `config_510_techniques.yaml:1404-1430` found it wrong-when-accepted
  5%, 13%, 56%, and 100% across the 300-1000, 100-300, 25-100, and 0-25 km/px bands (the last bin
  is three frames, and the two metrics differ). Star floor: real F ring data (#788) say star
  answers are 0.06-0.13 px off while reporting 0.23-0.31 px. Limb floor: three real limb-vs-star
  pairs give about 0.95 px per axis against 2.61 (n = 3, selection-biased; the upper bound still
  supports "too wide"). The one real test of whether the pipeline's own numbers separate good
  answers from bad found they do not (`util/nav_verification/README.md:45`).
- **Values with no real-frame evidence at all:** the terminator alphas and floor (the technique is
  effectively off on simulation alone: 0 of 116 rows within 1 px), the disc and blob NCC terms, the
  annulus floor and alphas, the unique-match no-floor decision and its SNR transform, the
  PSF-refine SNR ceiling of 30 (chosen on scenes using the navigator's PSF), the limb mis-lock
  boundary and ridge setting, the body-witness veto thresholds (designed on two simulated failure
  modes), and the tier boundaries themselves.
- **Correlated errors fused as independent.** The ensemble collapses only four known pairings
  (`ensemble_independence.py`). Limb fits on two bodies correlate at +0.72 and blob with disc at
  +0.83 on partial bodies (#322); limb and ring-edge fits share the gradient layer (an injected-bias
  result, #225). The fused sigma on those frames is too tight if the effect is real.
- **Mostly outside the blast radius:** the star, ring-edge, and Titan technique gates, which were
  tuned on real F ring, spoke, and Titan frames.

---

## 10. Cost, determinism, and coverage

### 10.1 Cost at real frame size (one core, i9-13900K)

| Scene at 1024 px | Navigate | Render | Peak memory |
|---|---|---|---|
| Disc | about 52 s | about 5 s | 2.2 GB |
| Ring | 24-29 s | under 1 s | 0.95 GB |
| Star field | 0.7 s | about 3 s | 1.2 GB |
| Star field with `instrument_defaults` | 0.6 s | about 7 s | 2.0 GB |
| Disc with 0.4° roll, rotation fitted | 1260 s | about 8 s | 2.2 GB |

About 85% of navigation time is the correlation in `navigate_with_pyramid_kpeaks`, not backplane
or model rendering as the validation plan assumes. Its FFT sizes are poor (2248 x 2608 has prime
factors 281 and 163; `correlate.py:839` does no fast-length padding); padding to `next_fast_len`
with `scipy.fft` measured about 4.7x faster per transform, and each candidate transforms the whole
frame where only a window is needed. The roll case is #126's brute-force rotation search. The
simulator integration tier takes about 414 s, of which sweeps are 315 s; 66 navigating
algorithmic-invariant tests carry no marker and run in the default unit tier. About ten mixed
full-frame scenes are affordable now, about twice the disc count after the FFT fix; body twist at
full size is not affordable until #126.

### 10.2 Determinism

All renders and baselines reproduce bit for bit with the SIMD pin. Without it, 18 of 64 renders
change almost everywhere and 4 of 64 baselines move by 0.01 px. The amplifier is the frame-wide
Poisson draw (`detector/chain.py:271`): a mean flipping between exactly 0 (no uniforms consumed)
and a tiny positive value, as FFT tails cross zero under `np.maximum(electrons, 0)`, desynchronizes
the stream for every later pixel. The pin covers NumPy on x86-64 only; NumPy does not promise
Generator distribution streams across versions; the render-diff test compares PNG bytes, which
also depend on the encoder; offsets are compared exactly after rounding to 0.01 px while
disc-correlation answers move in 1/128 px steps. Cheap mitigation: zero means below about 1e-9
before the draw. Durable fix: per-pixel variates, unrounded baselines compared with a ±0.02 px
tolerance, decoded-pixel comparison of noise-free renders, and statistical checks of noisy ones.

### 10.3 Coverage

Unit tests execute 85-100% of every `sim/` module; adding the integration tier moves the total
only from 81.4/63.6% to 82.7/65.2% (line/branch). Below 70% in the unit tier alone: two simulated
navigator models (body 56%, stars 42%) and ten editor modules (all but two rise above 85% with the
rest of the unit tests). Six detector modes, one vidicon branch, and two telemetry modes
(`reseau_scars`, `resample_texture`) never run in a rendered frame in CI; LORRI frame-transfer
smear renders under `instrument_defaults` but no test asserts it and no CI tier renders LORRI.
Only 2 of 31 artifact modes appear in any navigated scene. Mutations: the night-side floor (killed,
two of three tests only because the template shares the code), floor constants, photometric clip,
binned and full-well caps, and star flux law were killed; a beam-bend sign flip was killed only by
accident by one seed of a cosmic-ray test; a reseau-scar no-op survived.

---

## 11. Code quality and architecture

**Sound at the core:** a fixed physical stage order, named deterministic seeds, a typed
`DetectorParams` resolved once with documented precedence, a data-driven artifact registry, the
data-classified boundary with mechanical guards, about 2,600 precise behavioral tests, 412 of 415
definitions with docstrings, complete annotations, no shims, no `print`, no stdlib logging, and
only one source file over 1,000 lines.

**Accreted below it:**

- **Validation is off the render path (high).** `validate_sim_params` runs only on YAML load and
  save; `render_combined_model` and the in-memory `ObsSim` path do not call it (the calibration
  generators do validate, and the editor validates on load and save; only its preview render does
  not).
- **Null body keys crash the renderer (medium).** 15 of 19 body keys set to null pass validation
  and then raise `TypeError` or `ValueError`; the null-safe reader exists but only the ring code
  uses it (the body-block sequel to the closed #809).
- **Two image-side body renderers.** The "classic" path is kept byte-for-byte for old baselines;
  the topographic path takes over when any extra knob is set. A 1e-12 surge amplitude changes
  about 500-700 limb pixels by up to 1-1.8% of peak. No test compares the paths.
- **`compose_scene_radiance` is one 451-line function;** stages pass data through an untyped truth
  side channel; three of five stages discard their own generator; the seed default of 42 is
  restated 9 times; scenes and objects are untyped dicts; crater defaults appear in 4 places; one
  function takes 24 parameters.
- **About 80 instrument-name literals in generically named files,** three separate alias-folding
  mechanisms, and two parameter stores for star flux and read noise (Python catalog and navigator
  YAML) that agree today with no test tying them (#876 covers the tables, not the branches).
- **The editor lacks controls for 19 schema keys** (9 noise, 4 vidicon noise, 6 atmosphere),
  though the guide says every parameter is reachable from all three entry points; the editor is
  one class of 16 mixins with 194 `type: ignore` comments.
- **Smaller:** the navigator re-implements the image classifier; `artifact_counts.py` is called
  only by tests; the sim tests have no shared fixtures (32 files build their own scene); plan
  codenames ("5.2 interim" 18 times, "B7") sit in code; the navigator-side body model names its
  filtered input `sim_params`, the name used everywhere else for the full truth scene; the
  simulator logs nothing, so nothing records which defaults or modes a render resolved.
- **The `axis1` naming trap.** A full width named like a semi-axis, beside the ring `a_px`
  semi-axis, caught the calibration generator and an auditor.

---

## 12. Documentation

- **The user chapter does not teach a newcomer to build a scene or read the result (high).** It has
  no complete scene (only an artifacts and a rings fragment), never mentions the editor's Save
  Scene, Load Scene, and Save Image buttons, and never says the planted offset is not in the
  metadata document. It carries developer framing, per-instrument sky values in a shared chapter,
  and a "validated on your behalf" line the confidence warning in the outputs chapter contradicts.
  The README and quick start describe `sd_create_simulated_image` as something that renders and
  compares; it is an editor.
- **The developer guide is one 3,784-line file (high).** About 935 lines are a parameter reference
  that `sim_scenes/README.txt` repeats; the copies have drifted (README says `closest_planet`
  defaults to SATURN; the code and guide have no default). "Instruments" is a bold pseudo-heading
  inside "The stage pipeline"; test-harness topics sit inside the artifact framework; the realism
  match is the last section though cited from line 249. Proposed split: architecture and boundary
  (about 500 lines); rendering (about 1,300); scene format and parameter reference as the single
  source, with the README reduced to a pointer; testing with the simulator (about 450); the editor
  (about 170).
- **Lab-notebook content (medium):** roughly 400-550 lines of cohort numbers, a 64-seed table, haze
  measurement logs, scene outcome narratives, editor lock timings, runtime history, a plan
  codename ("Section 8"), critique pointers, and a rhetorical register ("honest" 10 times, "full
  stop"). Per-value provenance belongs beside the constants, campaign numbers in the report, scene
  numbers in the scene YAML, the rest in plans and critiques.
- **Claims that do not match the code:** every catalog value called "interim" though five are
  TUNED; "23 star-bearing frames" (11); the default `pytest` run said to include the simulator
  integration tests (they are also marked `integration`, which the default excludes); the
  invariant scenes said to run in the integration tier (they have no marker); the README naming
  the wrong schema module and omitting three scene classes; "all eight feature techniques" where
  two of the eight are scene types; a nonexistent `DISTORTION_RESIDUAL_RMS_PX`.
- **Terminology:** "catalog" has five meanings and "registry" three; the idealized view has three
  names; "navigation document" appears where "metadata document" is required; FOM, cohort, sidecar,
  "honest pin", and "three peers" are never defined; I/F is not defined in the user guide.
- **The report** is readable and defines each table, but "Purpose and scope" names two measurements
  across about 14 sections, it has no generation date or commit for the sweeps, and two passages
  compare Cassini with the other instruments.
- **check-doc-prose.sh --warnings:** 0 errors, 174 warnings (83 Oxford comma, about 10% real; 44
  X-not-Y, about a third noise; 21 "gate", of which the verb uses in the simulated body-model
  chapter should go; 15 "so"-clause).

---

## 13. New scenes or simulator changes?

**Authorable now, with YAML only, and worth doing immediately:**

- Scenes on the WAC, Galileo, LORRI, and (crudely) Voyager, with `instrument_defaults` on.
- Large offsets in every direction (tens to hundreds of pixels) on full-size frames; about ten
  full-frame scenes fit in CI today.
- Unfit twist on star fields at full size, with roll fitting off as shipped.
- Stars plus a body with an independent ephemeris error (`spk_error` or a per-body `nav_override`
  of the center), which is the ensemble's real arbitration problem.
- Dense ring profiles at real scale and opening, on the unlit face, at real phase, near the ansa
  and at low opening, with only catalog edges navigable; an A-ring edge with an unmodeled F-ring
  distractor at the real separation; a hidden eccentricity; saturated ring plateaus. (Build
  profiles from navigated geometry.)
- Whole-system WAC frames with a distant, correctly sized planet (no wedge).
- Unsaturated star fields at realistic magnitudes; terminator-dominated crescents; artifacts
  beyond lost lines (cosmic rays at a realistic rate once their amplitude is fixed, hot pixels,
  bloom, compression, smear).
- A truth check on every scene with a planted offset, 2-D error, per-scene tolerances from
  measured error plus a stated margin, nonzero off-grid offsets on the zero-offset scenes, error
  asserted against the reported sigma, and an assertion layer over the offset sweeps.

**Authorable but unrealistic until the simulator changes:**

| Case | Blocking gap |
|---|---|
| Any planet larger than half the frame (frame-filling limbs, close planet-plus-ring frames) | body canvas sized to the frame (section 4, item 1) |
| Planet plus rings | simulated ring model lacks occlusion and shadow masks and honest reliability; wedge shadow; no ring shadow on the planet |
| Twist on bodies | roll turns the body pattern the wrong way and the lit side not at all (#825, #644) |
| Single sharp ring edges | anti-aliasing on tau (0.3-0.7 px per edge) |
| Frame-edge content | zero-padded PSF and smear; no off-frame light |
| Whole-system WAC within about 3e6 km | orthographic projection (7-23 px RMS) |
| Non-clear filters | no filter model (#881-#885) |
| Physically scaled brightness | no albedo, solar distance, or filter term |
| Realistic star false sources | cosmic rays always saturate; hot pixels ignore exposure |
| Voyager NA vs WA | one `vgiss` key, wrong plate scale (#806, #804) |
| Mutual shadows | no cast shadows between bodies (#333) |
| Partial knowledge of ring shapes beyond mode 1 | no withheld mode, edge-wave, or width error; no ring `nav_override` |

**Not authorable:** edge-on and equinox rings (no thickness), inclined rings, dusty envelopes, wake
texture, partial arcs, pointing jitter, image mid-time error, Voyager readout shear (#333), and
accuracy of the production body and ring models (#843, #227).

---

## 14. Recommendations, in order

1. **Stop calling simulated numbers accuracy.** Correct the developer guide, the report, and the
   user chapter: simulated results are reproducibility and self-consistency on the Cassini NAC,
   and confidence values are provisional, fitted on simulation. Regenerate the stale star and
   LORRI figures. (Docs only.)
2. **Fix the defects that make realistic scenes impossible or wrong**, in this order: the body
   canvas (render at true position on a canvas covering the body's extent plus a PSF and smear
   margin); validation on the render path with typed per-block specs (fixes the null crash); the
   roll sign and lit side for bodies; ring anti-aliasing on intensity; a padded canvas for PSF and
   smear; the 30.0 calibrated star limit; the NAC clip level (#854).
3. **Write the authorable scenes** (section 13), with real assertions, and commit sweep summaries
   so report figures can be regenerated and checked.
4. **Give the simulated ring model production parity** (occlusion, shadow, reliability,
   resolvability), or better, run production `NavModelBody`, `NavModelRings`, and `NavModelStars`
   on simulated frames through an oops-compatible `ObsSim` (#843 extended; #227). Until then, label
   body and ring results as technique precision against a perfect model.
5. **Re-derive the production confidence layer on real frames,** starting with the tier boundaries
   and the 0.35 threshold, then RingEdgeNav confidence (before edge routing widens under #822), the
   limb floor and alphas, the star sigma (#788), the terminator plateau, and the disc size term. The
   F ring and spoke comparisons already give hundreds of frames with independent pointing. Fix the
   calibration generator's half-size slip and refit together with #793 and #874, after item 2.
6. **Make the realism match a test:** a fresh run compared with the committed summary within
   per-statistic bands, held-out frames, a quantization-aware noise estimator with support counted
   on both sides, re-centroided star cutouts (#343), and a sim-to-real transfer check (#334) that
   plants offsets in matched pairs and compares simulated error with real error per class.
7. **Add the physics in population order:** a common I/F scale (albedo, solar distance, filter
   throughput) for bodies and rings; geometric shadows from one sun vector; perspective,
   km-based, catalog-importing ring geometry; realistic cosmic rays and exposure-scaled hot pixels;
   charge-conserving bleed and an anti-blooming state; filter-dependent PSF, flux, and photometry;
   per-camera Voyager keys; ring thickness; envelopes and texture.
8. **Make the regression tests robust:** zero tiny Poisson means now; per-pixel variates, unrounded
   baselines with tolerances, and decoded-pixel comparison later. Fix the correlation FFT sizes,
   which makes full-frame scenes cheaper everywhere.
9. **Restructure code and docs** as sections 11 and 12 propose: typed frozen specs, retire the
   classic body path, split `compose_scene_radiance`, land #876, move the shared helpers out of
   `sim/`, split the developer guide in five, make one schema reference, and rewrite the user
   chapter around a complete scene.

---

## 15. Issue reconciliation

**Unfiled; should be filed:**

- Body canvas: an off-center body larger than the frame is clipped or vanishes.
- Roll turns a body's orientation opposite to stars and rings.
- Ring-edge anti-aliasing applied to tau biases single edges 0.3-0.7 px.
- PSF and smear zero-pad the frame; no off-frame light.
- Near-body-over-far-body seam; coverage-blind star extinction.
- Night-side 1% floor on every body (realism; effect unmeasured).
- Lommel-Seeliger and Minnaert clipped at 1.
- `rotation_tilt` stretches instead of tilting.
- No per-body albedo, solar-distance, or vidicon exposure term (the half of #884/#885 they do not
  cover).
- No shadows between rings and bodies; wedge planet shadow.
- Simulated ring model lacks occlusion, shadow, and reliability parity; production ring and star
  models never run on simulated frames (extend #843 or #227).
- Calibration generator draws catalog bodies at half size (`scene_gen.py:256`); refit with #793.
- `ObsSim.star_max_usable_vmag` returns 30.0 on calibrated instruments.
- Validation off the render path; 15 null body keys crash the renderer.
- Classic and topographic body paths disagree.
- 19 schema keys without editor controls.
- Realism match has no regression check; in-sample tuning with no hold-out; censored noise
  comparison and one-sided support labels.
- Stale realism summary and LORRI report figures since #849; stale star sweep figures; selected
  NAC stratum.
- Catalog: no non-NAC scenes; tiny one-quadrant offsets; no real frame sizes; no compound scenes;
  weak assertions; no per-instrument accuracy number; sweep results not committed.
- Cosmic-ray amplitude always saturating; hot-pixel amplitude independent of exposure; bloom cap
  discards charge; LORRI bloom contradiction.
- Voyager plate scale wrong for GEOMED; beam-bend sign from noise; reseau on a synthetic lattice;
  the vidicon's fixed photometric scale (1 DN about 7.8e-3 I/F at any exposure, about 19x coarser
  than a real Uranus frame), which turns faint rings into 1-DN dashes.
- For #81 (and #82, #83): production has no feature type for a ring narrower than a pixel, a
  Saturn-like `min_feature_pixels` would drop every Uranus ring but epsilon, and inclination modes
  are accepted by the catalog but ignored; a catalog alone will not make the rings navigable.
- Ring geometry: no km/catalog mode, orthographic only, no thickness (B = 0 renders nothing), no
  envelopes, no per-feature azimuthal structure, no withheld shape errors beyond mode 1.
- Poisson stream desynchronization; correlation FFT sizes.
- The `axis1` naming trap.
- Docs: user chapter, developer-guide split, schema drift, test-selection claims, terminology.

**Filed, and mis-tagged "unfiled" by an auditor:** #434 (rotation pivot), #755 (along-edge ring
lock), #322 (cross-body limb correlation), #342 (3.0 px PSF placeholder), #223 (terminator), #227
(de-circularize; the umbrella for model sharing), #153 and #334 (real-vs-sim comparison).

**Filed issues whose text needs correcting:**

- #329: its rationale says the simulated calibrated sky is noisier than real; the data say the WAC
  is 8x quieter and the real defect is that the noise cannot be measured.
- #330: says the vidicon has no transient stage (it now does), cites hot pixels per scene (fixed in
  #331), and uses an old key name.
- #883: describes a scene `filter` key that does not exist.
- #355: largely duplicates #806 now.
- #377: the renderer-scaling question is answered yes; add the ring-model parity blocker.
- #644 and #825: state that adding roll to `rotation_z` turns the pattern with the frame; it turns
  it the opposite way.
- #843: understates its scope (rings, stars, and Titan share the substitution; the night-side floor
  is a further shared assumption).
- #884: there is no body albedo for the filter to change.
- #310: cites the critique at its pre-archive path.

**Confirmed fixed or correctly closed:** #331 (per-detector hot pixels), #332 (binned modes), #325
(stars through dark limbs), #344 (haze albedo key), #809 (null top-level scalars; the body block is
the open sequel).

---

## 16. What the reviews changed

The adversarial reviews overturned or corrected these audit claims; the text above uses the
corrected versions.

- The star-to-body brightness table was inverted: WAC stars are too bright, LORRI and outer-planet
  Voyager stars too faint.
- The night-side floor was called optimistic for crescents; removing it from both sides made
  navigation no worse.
- The 1e10 calibrated star SNR was attributed to a unit mismatch; it is the 30.0 constant.
- "About 45" template-plus-shift scenes is about 40.
- The rotation pivot, along-edge ring lock, and cross-body correlation are filed (#434, #755, #322).
- BodyTerminatorNav's absence is by design, not a defect.
- "Every floor from one campaign" is overstated: three floors come from 20260718, the rest from
  earlier simulated campaigns or are uncalibrated.
- The calibration half-size slip was an aside in one audit; it is real and high severity, and it
  invalidates one ring audit's planet-plus-ring magnitudes as well.
- The LORRI realism mismatch is one commit (79b1cb37), confirmed by bisection.
- LORRI frame-transfer smear does render (under `instrument_defaults`); one realism test in the
  integration tier does render a frame; the calibration generators do validate their scenes.
- The planet's shadow edge on the rings is straight, not curved.
- The Poisson desynchronization is triggered by zero-mean flips, not arbitrary last-bit changes.
- The null-key crash is loud, so medium severity rather than high.
- One audit stated that Cassini ISS has no anti-blooming; the index shows the anti-blooming state
  on for 51% of NAC frames, so the bloom finding should be read with that correction.
- Outer-planet rings: the real Voyager scale is about 4.1e-4 I/F per raw DN (the simulator is 19x
  coarser, not 16x); the 2 px resolvability minimum is a per-planet config value, not code; the
  catalog accepts inclination modes the code ignores; real pointing errors on the three locked
  frames are 6-224 px with up to 0.5° twist; and the 0.07 px figure compares the simulator's
  projection with oops geometry, not with the image.
