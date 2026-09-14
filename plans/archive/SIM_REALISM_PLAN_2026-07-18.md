# Simulator Realism and De-circularization Plan

Design for issue #227 (WS-2 of `plans/VALIDATION_AND_CALIBRATION_PLAN.md`):
de-circularize the simulator, make the simulated image side dramatically more
realistic than the navigator's model, and prove that realism against real
frames. Absorbs the sim-side scope of #223 (terminator emission), #153
(real-vs-sim diagnostics), #84 (ring edge/gap compositing), #78 (CraterMaker),
and #158 (mesh smooth shading); supplies the truth-known scene machinery
WS-0 (#224) needs (Section 8); consumes the per-instrument residual
distortion measurements of #228 (Section 4.4).

**Status: design, revision 2 (2026-07-15) — under operator review; nothing
below is built.** Revision 2 redefines independence as information asymmetry
(Section 1), adopts the multi-PR integration-branch delivery shape
(Section 10), and folds in the fixes from the 2026-07-15 four-lens review.

---

## 1. Problem and goals

Today the image being navigated and the model navigating it describe the same
ideal world: a perfect ellipse is rendered where a perfect ellipse is
predicted, stars sit exactly on their catalog positions, and ring features are
exactly where their orbit models say. With nothing separating what the image
contains from what the navigator knows, every simulated-image accuracy number
is a self-consistency measurement, not an accuracy measurement.

The fix is **information asymmetry** (operator definition, 2026-07-15). The
sim plays the role of nature plus SPICE: it decides the true scene — ragged
limbs, ring features scattered within their orbital-element errors, stars
over- or under-exposed through a non-ideal PSF, background light, bad pixels,
cosmic rays, lost lines, distortion residuals, SPK offsets — and renders the
image from that truth. The navigator receives exactly two things:

1. the image, and
2. the idealized information the production pipeline would retrieve from oops
   backplanes, config files, and star catalogs: catalog star positions and
   magnitudes, ellipsoidal body shapes and predicted locations, ring-feature
   predicted geometry with its *declared* uncertainty, the published PSF.

The variance parameters and the unnavigable contaminants never cross to the
navigator's side. Recovery error then measures the navigator against the same
information asymmetry it faces on real frames, and because the planted gap
between truth and idealized information is known exactly, the error curves
are honest by construction.

Two properties carry the program:

1. **The information boundary.** No truth parameter — planted error values,
   variance fields, artifact placements, unnavigable content — is readable by
   any navigator-side model. Enforced mechanically at the `ObsSim` boundary
   (Section 3.2), not by authorship rules: shared geometry helpers between
   the renderer and the navigator-side models are legal and desirable, since
   shared conventions make the planted error the *only* error in the
   measurement.
2. **Realism.** The image side is *more* realistic than the navigator's best
   model — oversampled, PSF-convolved, topographic, photometrically richer,
   with the real detector and telemetry artifacts of each supported camera —
   so the residual recovery error is the genuine model error the real
   pipeline also incurs. The distributional match against real frames
   (Section 7) is the sole bridge from sim-world numbers to real-world
   claims, and is load-bearing accordingly.

The navigator is never handicapped: it keeps its best available model, and the
reported error is a function of the *remaining* mismatch.

Scope of the claim: these numbers measure the *techniques and the ensemble*
under model error, driving the same technique code the real pipeline runs.
The navigator-side simulated NavModels stand in for the real models' geometry
(pixel-space here, backplane-based on real frames); the real models' geometry
path is validated by the real-image baselines, not by this program.

### Non-goals

- Radiative-transfer-grade photometry (Hapke with coherent backscatter,
  polarization). The realism bar is "distributionally indistinguishable in the
  statistics the techniques consume," not photometric-pipeline grade.
- Shadows of one body on another (eclipses) and planet-shine. Mutual events
  in the *occultation* sense — one body partially occluding another — are IN
  scope (Section 4.1): they are common in Cassini frames and directly stress
  limb fitting. Only the cast-shadow/illumination coupling between bodies is
  excluded.
- Replacing the navigator-side simulated NavModels. They stay as-is (they are
  the "best model" under test), gaining only feature emission work (terminator
  arcs, navigable-subset selection) — never renderer changes driven by the
  image side.

---

## 2. Current-state audit

What the image side renders today, and why it is not realistic:

| Domain | Today | Realism gaps |
|---|---|---|
| Stars | Gaussian PSF, peak-normalized `2.512^-(vmag-4)`, optional linear motion smear per star | No PSF wings/undersampling; peak (not flux) scaling; background stars drawn from an ad-hoc power law, not a `log N(m)` star-count law; no field-position PSF variation |
| Bodies | Ellipsoid, Lambert with a 0.01 dark-side floor, craters perturb *shading normals only*, hard AA edge | Silhouette is a perfect ellipse — crater/terrain relief never reaches the limb or terminator line; single photometric law; no albedo texture; no atmosphere (Titan/Saturn limbs are hard edges); no limb darkening; terminator position is the Lambert zero-crossing of a smooth sphere |
| Mesh bodies | Flat per-face shading, polygon silhouette | Faceted appearance (#158); relief is low-frequency modes only |
| Rings | Solid annulus (or gap) between mode-1 eccentric precessing edges, drawn as circles about a pixel center; brightness = fade from edge; ringlets add with a clip, gaps subtract the background toward black | No projected opening angle; no radial optical-depth structure; no transmission model — a gap erases what is behind it instead of revealing it, and a clipped sum is not a transparency screen (#84's surviving half; bodies, separately, overwrite by mask); no lit/unlit asymmetry; no phase behavior; no azimuthal structure; one navigated feature = the whole rendered feature; feature positions are exact (no inter-feature orbit error) |
| Optics | Stars get a Gaussian PSF; bodies/rings get anti-aliasing only; stray light is a smooth ramp/bump | Every edge in a real frame is softened by the camera PSF; no whole-scene smear; no geometric distortion; no ghosts/structured scattered light |
| Detector | Poisson on DN, Gaussian read noise, flat bias, per-pixel dropouts, single-pixel cosmic rays, symmetric column bloom | No gain/electrons distinction; no dark current or hot pixels; no coherent/banding noise; no 12-to-8-bit quantization; no compression artifacts; no residual image; no reseaus (Voyager); dropouts are per-pixel, which is not a failure mode any of the four cameras has |
| Telemetry | `missing_data_rate` marks random pixels | Real losses are whole lines, partial lines, alternating lines, and truncated frame bottoms/tops (Cassini, Voyager), and 8x8 compression-block dropouts (Galileo ICT); per-pixel loss essentially never occurs |
| Calibrated (I/F) path | No noise at all | Real calibrated frames carry propagated shot/read noise and calibration residuals |

---

## 3. Architecture: an independent forward model

### 3.1 Package layout

New package `src/spindoctor/sim/forward/` — the image side. The existing
`sim/sim_body.py`, `sim/sim_body_polyhedral.py`, `sim/sim_ring.py` become the
**navigator's** property (they are what `nav_model_body_simulated.py` and
`nav_model_rings_simulated.py` consume; the star model's plumbing is fixed in
3.2) and move under `nav_model/` in phase A, so the package layout states
which side owns what. Geometry helpers may still be shared across the
boundary (Section 3.2) — the split is about information, not lineage.

Phase A gives `forward/body.py` and `forward/ring.py` their initial content
by **porting the current rendering logic at present fidelity** — the image
side gets its own working copy in A, so every scene still renders — and
phases E and F then *replace* those ports with the topographic and
optical-depth renderers. The ports may call the shared geometry helpers;
they are not required to be rewrites.

```text
src/spindoctor/sim/forward/
    scene_radiance.py    # composes the oversampled, noise-free radiance image
    body.py              # topographic ellipsoid renderer
    body_mesh.py         # smooth-shaded mesh renderer
    ring.py              # optical-depth ring-system renderer
    star.py              # flux-normalized star renderer with empirical PSF
    atmosphere.py        # haze-limb / terminator-wrap layer for Titan-class bodies
    optics.py            # PSF convolution, smear, distortion, ghosts/stray light
    detector.py          # electrons, dark, hot pixels, read noise, banding, ADC
    telemetry.py         # compression artifacts and structured data loss
    artifacts_catalog.py # per-instrument artifact defaults (Section 5 tables)
```

`render.py`'s `render_combined_model` keeps its signature (its callers are
`ObsSim`, the scene editor, and `sim/png_export.py`, which the doc-gallery
and sweep runners drive) but becomes a thin driver over `sim/forward/`.

### 3.2 The information boundary

Independence is a property of what information crosses to the navigator, not
of who wrote the code (operator definition, 2026-07-15). Enforcement is
mechanical:

- The scene schema separates, for every object, the **idealized block** —
  what the navigator may know: catalog star positions and magnitudes,
  ellipsoid shape and predicted location, ring-feature predicted orbits and
  their *declared* uncertainties — from the **truth block**, which only the
  renderer may read: `catalog_error_*`, `orbit_error`, `spk_error`,
  `limb_relief_*`, `albedo_texture`, `artifacts`, `optics`, and every
  non-navigable object. Section 15.6 tags every key with its side.
- `ObsSim` exposes navigator-side models a **filtered view**
  (`obs.nav_params`) built by dropping every truth key and every
  non-navigable object. The renderer consumes the full scene; the navigator
  side structurally cannot read what is not there. A unit test constructs a
  scene exercising every truth key in the 15.6 inventory and asserts none is
  reachable through the filtered view — this test is the independence
  guarantee, and any PR that adds a truth key must extend it in the same PR.
- The star path is the standing violation this section exists to fix:
  `NavModelStarsSimulated` today reads `obs.sim_star_list`, which `ObsSim`
  takes from the *renderer's return metadata* (`obs_inst_sim.py`) — rendered
  stars with planted errors applied and rendered DN attached. Phase A
  replumbs it: the navigator-side star model reads catalog entries from the
  filtered view, and the renderer's output objects never cross.
- Shared geometry helpers are legal on both sides, deliberately: with shared
  conventions (pixel centers, sign of `dv`, edge rasterization) the planted
  error is the *only* error in the recovery measurement. Independent
  implementations would each carry their own conventions, and any delta
  between them would land as an unknown systematic inside the measured
  error — contaminating the truth reference the whole program rests on.
  Shared helpers take explicit geometry arguments; they never read the
  scene dict, so they cannot smuggle truth across.
- The residual blind spot, stated honestly: a bug in a shared helper that
  makes rendered features subtly unlike *real* features leaves both sides
  consistent and recovery clean — scenes wrong, measurement blind. Its
  detectors are the Section 7 distributional match and the reviewed render
  diffs (15.11), not any structural test.

### 3.3 Rendering pipeline

The forward model renders in physically ordered stages. Every stage is a pure
function on an oversampled float image plus a seeded RNG; scenes enable and
parameterize stages; per-instrument defaults come from the artifact catalog.

```text
1. Scene radiance   oversample x4 (configurable); bodies, rings, stars, sky
                    composed with correct occlusion AND transparency
                    (ring transmission, Section 4.2)
2. Optics           whole-scene PSF convolution (empirical kernel: core +
                    wings), pointing smear (shared by every scene element),
                    optional geometric distortion, ghosts, structured stray
                    light
3. Downsample       to detector grid (box filter)
4. Detector         signal -> electrons (gain), dark current + hot pixels,
                    Poisson(e-), residual image, read noise, coherent/banding
                    noise, bias structure, ADC: gain state -> DN, 12-bit
                    (or 8-bit via LUT) quantization, full-well bloom + clip
5. Telemetry        lossy-compression artifacts, then structured data loss
                    (lines / partial lines / alternating lines / truncation /
                    blocks), reseau marks (vidicon), missing-data markers
```

Key contrast with today: the PSF applies to the *whole composed scene* — the
limb-gradient profile, ring-edge profile, and star shape all inherit it — and
motion smear applies to the scene, not to stars individually.

The navigator-side simulated NavModels are untouched by all five stages: they
continue to predict the ideal geometry. Everything the forward model adds is
therefore genuine model error, which is the point.

---

## 4. Realism upgrades by scene domain

### 4.1 Bodies: topographic limbs and terminators

The operator's core observation: craters on the disc are irrelevant to
navigation (we fit limbs and terminators), but crater *relief on the edge* is
exactly what the navigator's perfect-ellipsoid model cannot capture. The
forward body renderer therefore makes the silhouette and terminator honest:

- **Limb topography field.** The rendered silhouette is the ellipse radius
  modulated by a seeded 1-D random field around limb azimuth:
  `r(theta) = r_ellipse(theta) * (1 + delta(theta))`, with `delta` specified
  by an RMS relief fraction (`limb_relief_rms`, physically h/R: ~0.1% for
  Enceladus-class, ~1-3% for small cratered moons) and a correlation length in
  degrees (`limb_relief_corr_deg`, set by crater-diameter statistics). The
  field is band-limited noise, not per-pixel jitter, so the limb is *bumpy the
  way terrain is bumpy* — coherent over crater-scale arcs.
- **Terminator raggedness.** The same relief, carried into the incidence
  computation, displaces the terminator locally by `H * tan(incidence)`
  (H = h * R, the *absolute* height in px — the fractional field times the
  local radius) — growing toward the terminator, where incidence
  approaches 90 deg, and capped there by the horizon limit
  `sqrt(2 * R * H)` since the tangent diverges — and casts approximate
  local shadows (a shadow march along the projected sun direction at the
  oversampled grid — cheap and sufficient; full ray-traced shadowing is
  out of scope). The relief field's domain and the march are pinned
  exactly in Section 15.5. High-phase crescents then have the ragged inner
  edge real ones do, which is the feature `BodyTerminatorNav` actually
  fits.
- **True topographic option (CraterMaker, #78) — deferred (operator
  decision 2026-07-15).** The statistical limb field above is the mechanism;
  a CraterMaker-based global height map remains a possible future upgrade
  for a ground-truth-grade shape axis, but is not part of this plan's scope
  and adds no dependency now.
- **Photometric laws.** The forward renderer supports Lambert,
  Lommel-Seeliger, Minnaert(k), and a lunar-Lambert blend, plus a simple
  opposition surge term; the navigator keeps Lambert. Law mismatch is a sweep
  axis (it moves the terminator and the limb-darkening profile).
- **Albedo texture.** Optional multiplicative albedo field (band-limited noise
  plus discrete albedo spots) so disc correlation sees realistic contrast.
- **Mesh bodies** (irregulars, chaotic rotators like Hyperion): Gouraud/Phong
  shading (#158), higher mesh frequency content, and the same limb-relief
  modulation applied in mesh space. Chaotic rotation needs no dynamics — pose
  is per-frame ground truth from the scene — but multi-frame scene sequences
  get an optional per-frame pose scatter to emulate an unmodelable rotation
  state (the navigator's mesh pose is then wrong by a known amount: a sweep
  axis).
- **Mutual events (partial occlusion).** The renderer already composites
  bodies near-over-far, so an overlapping pair renders correctly; what is
  missing is everything around that. A `mutual_event` scene class puts two
  bodies in partial overlap (grazing to near-total, swept), with truth
  bookkeeping for the visible fraction and occluded limb arc of the far
  body. The navigator-side simulated body model predicts each body's full
  limb — the occluded arc is genuine model error unless the model masks it,
  which makes this both a realism scene and a defect detector: limb/DT
  fitting must either downweight the hidden arc or degrade gracefully, and
  the ensemble must not double-count two bodies whose features overlap.
  Cassini frames hit this constantly (satellite pairs against the rings).
- **Planet disc texture, satellite transits, and shadow transits.** Giant
  planets get an azimuthally structured albedo texture (banded zones/belts
  plus discrete GRS-class storm ovals) — the disc-correlation confound
  Jupiter and Saturn actually present. On top of it, a `transit` scene
  variant renders a moon in front of the disc (bright or dark against the
  bands) and its cast shadow: a sharp, high-contrast circular *false
  crater* that disc correlation and blob techniques can lock onto. (The
  shadow here is texture on a rendered disc, not body-on-body illumination
  coupling, which stays out of scope.)
- **Atmospheric bodies (Titan, Saturn).** New `atmosphere` block on a body:
  exponential haze layer above the surface with scale height (px), optical
  depth, single-scattering asymmetry (forward-scattering brightening at high
  phase), optional detached haze shell. Consequences the navigator model must
  live with: the limb is a soft exponential ramp, its apparent radius is
  phase- and wavelength-dependent (the known Titan altitude-vs-phase
  problem, issue #60), and the terminator brightens past 90 deg incidence
  instead of cutting
  off. This is the sim substrate Titan navigation (#60) will eventually be
  tested on.

### 4.2 Rings: an optical-depth ring system

Replace "annulus with an inner and outer edge" with a **ring-system spec**: a
list of radial features, each with its own orbit model, optical depth, and
navigability flag. The renderer draws the whole system; the navigator is only
told about the flagged features.

- **Radial structure.** Each feature contributes a radial optical-depth
  profile tau(r): `ringlet` (flat or ramped tau between edges), `gap` (tau
  suppression), `edge` (one-sided step), `ramp`, `wave` (damped sinusoid for
  density-wave trains — visual clutter, typically non-navigable). A realistic
  Saturn scene composes tens of features; today's scenes compose one or two.
- **Per-feature orbit models and planted orbit error.** Every feature keeps
  the mode-1 eccentric precessing model and gains optional m>=2 modes
  (resonantly forced edges, e.g. the B-ring outer edge m=2) and short-
  wavelength satellite edge waves (Daphnis/Pan-style sinusoidal perturbation
  with wavelength, amplitude, damping). Each feature additionally takes a
  **planted ephemeris error** (`delta_a_px`, `delta_ae_px`,
  `delta_long_peri_deg`) applied on the *render* side only — the navigator
  predicts from the catalog values, so features are slightly misplaced
  relative to each other and to the model, exactly as real ring features are
  relative to their published orbit solutions. This is a first-class sweep
  axis (it is the ring analog of the body ephemeris-error axis).
- **Photometry: lit vs unlit side and transparency.** Brightness derives
  from tau and the viewing/lighting geometry via the single-scattering
  closed forms pinned in Section 15.4 — one normative equation set, stated
  once. The unlit branch produces the real inversion: moderate-tau regions
  (C ring, Cassini Division) bright from the dark side, high-tau (B ring)
  nearly black. The scene supplies the shared `geometry` block
  (Section 15.4) and a one-parameter Henyey-Greenstein phase function
  (dusty features brighten strongly forward). This kills the "solid
  annulus" model: a feature's contrast, and even its sign against the
  background sheet, follows from tau and geometry.
- **Compositing (#84).** Rings composite as a transmission screen, not paint:
  `img = ring_brightness + exp(-tau/mu) * background` per pixel, evaluated in
  range order. Stars, bodies, and the planet show through low-tau features;
  gaps reveal the background instead of erasing it; anti-aliased edges blend
  instead of overwriting.
- **Projection (in scope — operator decision 2026-07-15).** The ring system
  is rendered in ring-plane coordinates and projected through an opening
  angle B and node longitude, so an inclined view produces the foreshortened
  ellipse geometry, radial resolution gradient, and near/far-arm asymmetry
  real ring images have. The projection parameters live in the scene's
  shared ring `geometry` block — an idealized block, so both sides consume
  it: the forward renderer draws the full system through it, and
  `NavModelRingsSimulated` predicts the navigable features' edges through
  the same projection helper (shared by design, per Section 3.2), so
  predicted edges land in projected positions too.
  The current sky-plane-circle mode is the degenerate B = 90 deg case and is
  kept for the existing regression scenes.
- **Azimuthal structure.** Optional low-frequency azimuthal brightness
  modulation (self-gravity-wake quadrant asymmetry), clump fields
  (F-ring-style beads along an eccentric core), and a planet-shadow boundary
  (an azimuthal darkening wedge crossing every feature at a specified
  longitude — a strong non-navigable edge cutting across navigable ones,
  common in real Cassini geometry) for realistic clutter.
- **Spokes.** Transient wedge-shaped azimuthal albedo features on the
  B-ring sheet — dark at low phase, bright at high phase — radially broad
  and azimuthally sharp: a moving, non-navigable feature crossing navigable
  edges. Implemented as an azimuthal-structure variant.
- **Embedded moonlets and propellers.** Point sources *inside* the ring
  system (Pan/Daphnis in their gaps, with their local edge waves) and
  propeller-shaped local disturbances — realistic blob/star confounders
  sitting exactly on navigable features.
- **Navigable subset.** Each feature carries `navigable: true|false`.
  `NavModelRingsSimulated` builds `RingFeature`s only for navigable features
  (predicting from catalog orbits, without the planted error). The rendered
  frame is full of structure the navigator was never told about — the
  false-lock / distractor regime the DT techniques face on real A-ring and
  C-ring frames, which today's sim cannot produce at all.

### 4.3 Stars

The same render-everything / navigate-a-subset principle as rings. Today the
navigator is told about every star in the scene's `stars:` list and the only
confounders are the random `background_stars_num` field stars; that cannot
express the regimes the star techniques must be proven in.

- **Navigable subset.** Each star gains `navigable: true|false`: rendered
  either way, but only navigable stars reach the simulated star NavModel's
  catalog. This directly builds the target regimes: lock onto a single
  bright star, a pair, or a three-star triangle match, embedded in a field
  of comparably bright rendered stars the navigator has no knowledge of.
  Scenes sweep the confounder density and brightness ratio until unique
  matching breaks — the measured breakdown point is a deliverable, not a
  failure.
- **Planted catalog error.** Per-star `catalog_error_v/u` (and a scene-level
  random catalog-scatter sigma) displace the *rendered* star from where the
  catalog tells the navigator it is. Small values sweep astrometric-error
  tolerance; large values build the expected-to-fail scenes — every star in
  the wrong place, where the correct behavior is a failed/low-confidence
  result, not a confident wrong offset. These scenes carry a scene-level
  `expected` block (Section 15.6), asserted by new sim-suite machinery
  (Section 15.8), and belong in the false-positive characterization
  alongside the scattered-light criteria work.
- **Star occultation by rings and bodies.** With transmission compositing
  (Section 4.2) a catalog star behind the C ring renders attenuated by
  `exp(-tau/mu)`, and a star behind the B ring or a body simply is not
  there — while the navigator is still told about it. Tests
  predicted-but-absent and predicted-but-dimmed star handling, a regime
  every Cassini ring image has. Essentially free once compositing exists.
- **Saturated navigable stars.** A scene can plant a navigable star bright
  enough to clip (plus bloom on cameras that bloom), whose centroid is
  biased by the clipped core — testing the star techniques' saturation
  handling directly rather than only as background clutter.
- **Double and variable stars.** A per-star `companion` (separation, delta
  magnitude) renders an unresolved binary whose photocenter sits off the
  catalog position by a magnitude-weighted amount — a *physical* catalog
  error, distinct from the planted kind; a per-star `delta_mag` renders a
  variable at a different brightness than cataloged, exercising the
  brightness-margin diagnostics.
- **Flux normalization.** Star DN scales as integrated flux for the magnitude
  and exposure (through the instrument's photometric zero point), not
  peak = f(vmag); the PSF then dictates the peak. Undersampled cameras
  (LORRI 4x4, Voyager) get sub-pixel-phase-dependent peaks, which is what
  their real centroiding noise looks like.
- **Empirical PSF.** Per-instrument kernel: Gaussian-ish core with measured
  FWHM plus power-law wings and (optionally) mild astigmatic ellipticity and a
  field-position-dependent width. Shared with the whole-scene optics stage.
- **Background sky.** Star counts drawn from a cumulative star-count law
  `log10 N(<m) = a + b*m` per square degree, scaled by the FOV area:
  interim mid-galactic-latitude V-band values `a = -3.1`, `b = 0.34`
  (provenance-tagged; reproduces ~2/deg2 at V=10, ~100 at V=15, ~5000 at
  V=20 against tabulated counts), with a `density_factor` multiplier
  (default 1; ~5-10 emulates low galactic latitude) — replacing the ad-hoc
  exponent. Keys and defaults: `sky_counts` in Section 15.6. Plus an
  optional diffuse sky/zodiacal floor.
- **Streaked fields.** Smear applies to the scene (4.4), so long-exposure
  star fields streak coherently with the bodies, not per-star.

### 4.4 Scene-level optics

- **Whole-scene PSF** (core + wings) at the oversampled grid. This is the
  single highest-leverage realism change for the DT techniques: real limb and
  ring-edge gradient profiles are PSF-shaped, and the realism match
  (Section 7) explicitly compares limb-gradient profiles.
- **Smear.** A pointing-drift track (linear or short arc, px and direction)
  convolves the whole scene: Voyager long exposures, LORRI 3-axis-off
  sequences, Cassini long-exposure WAC frames. **Differential smear** is in
  scope: each object class can carry its own motion vector relative to the
  frame (tracked body sharp while stars trail, or vice versa — the fast
  flyby regime), which stresses star-technique vs body-technique agreement
  on a single frame.
- **Spacecraft ephemeris (SPK) error.** A scene-level planted spacecraft
  position error displaces bodies and rings by parallax (scaled by
  1/range, so near bodies move more than far ones) while stars stay put.
  The star solution and the body solution then genuinely disagree — an
  internally inconsistent frame no single `(dv, du)` can fix, and the sim
  analog of the real SPK blind spot the validation plan repeatedly flags.
  The ensemble's behavior on such frames (detect the inconsistency, don't
  average it away) becomes testable for the first time.
- **Ghosts and structured stray light.** Beyond the current smooth ramp:
  out-of-field bright-source gradient with structure, and optional ghost
  images (displaced, defocused copies at low amplitude) for the cameras that
  document them.
- **Geometric distortion — residual only.** oops already implements each
  instrument's geometric distortion model and applies it wherever pixel
  coordinates and look vectors are interconverted (backplanes, predicted
  feature positions), so the navigator corrects the *known* distortion on
  real frames as a matter of course. The quantity that is actually present
  in the frames the pipeline consumes — and therefore the only quantity
  this stage plants — is the **residual**: the error remaining after the
  model is applied (~1 px internal error documented for Voyager GEOMED
  products; sub-pixel for Cassini's corrected field; to be *measured*
  per instrument by the formalized star-field residual analysis of
  issue #228, which promotes `experiments/fov_twist/find_fov_twist.py` into the
  supported suite). The stage warps the oversampled scene by a low-order
  field (radial polynomial about the optical center plus an optional small
  random non-radial term for the Voyager per-image wander); the navigator
  gets no distortion model for sim frames, so the warp is planted,
  field-position-dependent geometric error — a limb fitted at the frame
  edge disagrees with a ring fitted through the center by the differential
  residual between their positions, which is the WS-17 confound at its
  true amplitude. Uses: (a) ON in each instrument's realism defaults at
  the measured residual amplitude once #228 delivers it (interim
  amplitudes ship meanwhile, Section 15.7; #228's method needs star
  frames, which exist for all four instruments — the FOV-twist experiment
  lists are the Voyager/Galileo source, Section 7 FOM 2); (b) swept
  amplitude as a tolerance
  study (how much residual the techniques and ensemble absorb before
  cross-technique disagreement inflates); (c) OFF in the Section 8
  accuracy sweeps so mismatch curves stay unconfounded. Plate-scale error
  — promoted out of the Section 13 triage into this stage — is its
  first-order term and shares the implementation.

---

## 5. Detector and telemetry artifacts, per instrument

> Instrument numbers below come from the instrument and calibration papers
> and PDS documentation cited in Section 12 (Instrument references).

### 5.1 Common artifact framework

Every artifact is a `detector.py`/`telemetry.py` stage with a per-instrument
default block in `artifacts_catalog.py` (delegating to the
`config_4N0_inst_*.yaml` files where a number already lives there) and
per-scene overrides. All stages are individually seeded via
`derive_effect_seed` so adding one never perturbs another.

The current `missing_data_rate` per-pixel dropout **stays** as a generic
stress knob (it is useful precisely because it is worst-case), but it stops
being the only data-loss model and is no longer presented as an instrument
artifact. Structured loss replaces it in instrument-named scenes:

| Loss mode | Shape | Cameras |
|---|---|---|
| `missing_lines` | N whole image lines (random or contiguous run) | Cassini, Voyager, Galileo |
| `partial_lines` | lines truncated from column k to end | Cassini, Voyager |
| `alternating_lines` | every other (or every 4th) line absent | Cassini lossless under severe entropy; Galileo HMA/HCA formats |
| `edited_frame` | only a centered vertical band of each line present, or a half-height frame | Voyager edited modes (IM4-IM15); Galileo cut-out windows |
| `truncated_frame` | bottom (or top) M lines absent | Cassini (readout-window truncation, lossy packet loss), Galileo |
| `missing_blocks` | zero-filled bands quantized to compression-block rows (8 lines for Galileo ICT), often starting mid-line | Galileo ICT slice loss, Cassini lossy |
| `line_garble` | from a bit-error column onward, line replaced by noise/garbage rather than marker | Voyager IDC (difference-encoded) lines; Galileo Reed-Solomon overflow |
| `pixel_spikes` | isolated pixels flipped to wrong values (salt-and-pepper), not zeroed | Voyager uncompressed-era bit errors |
| `hot_pixels` | fixed per-seed set of high-DN singletons (and, on CCDs read through them, warm columns) | Cassini, Galileo, future instruments (HST-class); explicitly *disabled* for LORRI, which has none |
| `dead_pixels/columns` | fixed zero/low-response singletons or columns | LORRI, generic CCDs |

Each mode takes an incidence parameter (expected lines/blocks lost per frame,
or a per-frame probability of any loss), so the realism match can compare
*artifact incidence* distributions against the WS-3 cohort, per instrument.

**Mode key registry.** The `artifacts` block (15.6) is keyed by exactly these
names; the validator rejects any other. Loss modes, from the table above:
`missing_lines`, `partial_lines`, `alternating_lines`, `edited_frame`,
`truncated_frame`, `missing_blocks`, `line_garble`, `pixel_spikes`,
`hot_pixels`, `dead_pixels`, `dead_columns` (the table's last row is two
keys). Detector/electronics modes, whose shapes are defined in 5.2-5.6:
`banding_coherent` (the Cassini 2 Hz / Galileo 42-px / LORRI striping
family), `bias_structure`, `dark_ramp` (RBI, vidicon, and shutter-shading
line gradients), `bright_dark_pairs`, `bloom` (electron-domain full-well
bleed; unavailable for LORRI), `quantization_lut`, `quantization_ls8b`,
`contouring_8bit`, `fixed_pattern` (stitch combs, dust donuts, vignetting,
jail bars, PRNU), `radiation_transients` (the generalized cosmic-ray model
of 5.6), `frame_transfer_smear` (LORRI), `serial_tail` (LORRI saturation
undershoot), `beam_bend` (Voyager brightness-dependent limb bias),
`residual_image`, `reseau_scars` (Voyager GEOMED), `resample_texture`
(Voyager GEOMED), `compression_dct` (Cassini lossy / Galileo ICT / LORRI
lossy blockiness), `truth_window`, `cutout_window`, and `embedded_header`
(LORRI row 0). Ghosts and structured stray light are `optics` blocks, not
artifact modes. A mode key with no catalog entry for an instrument is
unavailable there and fails validation on that instrument's scenes (e.g.
`hot_pixels` on nhlorri).

**Adversarial placement.** By default every stochastic artifact is placed
uniformly. An optional `adversarial: true` placement mode seeds artifacts
preferentially on the navigation features instead — cosmic ray on the limb,
hot pixel at a predicted star position, missing line through the ring edge,
compression-slice dropout across the terminator — turning the artifact
stress sweep from average-case into worst-case. Nearly free: the renderer
already knows where the features are.

### 5.2 Cassini ISS

Sources: Porco et al. 2004 (SSRv 115), ISS Data User's Guide (Knowles 2018,
PDS), West et al. 2010 (PSS 58). Full-frame 1024x1024 CCD with a mechanical
shutter — there is **no** frame-transfer smear; vertical streaks come from
blooming and hot pixels, not readout.

Data-loss modes (the atomic loss unit is a partial-line segment, because one
image line spans up to 4 telemetry packets and missing segments are
zero-filled):

| Mode | Shape | Notes |
|---|---|---|
| Lost packets, uncompressed | isolated partial-line segments / whole lines, zero-filled | up to two good segments per line survive |
| Lost packets, lossless | a *pair* of adjacent lines lost per error | Huffman codes lines in pairs |
| Lost packets, lossy | image truncated from first lost packet (bottom gone), resync only at group-of-blocks boundaries | |
| Readout-window truncation | clean full-width cut at some line: bottom 1/8, 1/4... of frame absent | commanded readout index shorter than actual compression outcome; common and clean |
| Huffman 2:1 line-pair truncation | right-hand tails of **alternating lines** blanked; severe case = every other line lost | entropy-driven: triggers in noisy/busy image regions, not dark sky — this is the operator's "every other line" artifact |
| Lossy (DCT) compression | 8x8 blockiness, ringing, high-frequency loss; 8-bit data only | 4 quantization-matrix pages, scale factor B (0-15) |

Detector/electronics artifacts:

- **2 Hz coherent noise**: horizontal banding, constant amplitude in
  electrons (~30 e- NAC at 2.1+2.5 Hz beating; ~6 e- WAC at 4.0 Hz), spatial
  period set by line readout rate — including a mid-image *frequency change*
  when the camera buffer fills and readout pauses (which also steps the dark
  level at that line). Appears as a diagonal pattern in 2x2/4x4 summed frames.
- **Bias**: per-image pedestal (tens of DN) plus the 2 Hz line-correlated
  variation; low-level irregular *vertical* banding family in some telemetry
  modes.
- **Dark/RBI**: dominant "dark" signal is residual bulk image from the
  pre-exposure light flood; increases with line number (readout gradient),
  enhanced columns above hot pixels.
- **Hot pixels**: 0.15% of pixels (2004) growing to ~0.28% (2008/9), each
  contaminating the column read out over it (vertical streak); population
  drifts (some anneal away). Time-tagged per-seed sets.
- **Bright/dark pixel pairs**: anti-blooming mode ON produces scattered
  vertical 2-pixel bright/dark pairs in unsummed long exposures.
- **Blooming**: full well (~110k e- NAC / ~95k e- WAC = ~3600/3400 DN at
  gain 2 — i.e. saturation *below* DN 4095) bleeds along columns; extreme
  cases wipe whole columns. DN saturation (255/4095) clips without bleeding.
- **Quantization**: 12-bit ADC with uneven bit weights (histogram spikes at
  DN = 2^m boundaries); **LUT mode** (square-root-like 4096->256 encoding,
  quantization ~ photon noise, residual after inversion grows with signal);
  **LS8B mode** (low 8 bits kept: values > 255 wrap modulo 256 — banded
  wraparound on bright targets).
- **PSF**: FWHM 1.2-1.6 px (NAC) / 1.1-1.8 px (WAC), undersampled, with
  measured non-diffraction wings extending hundreds of pixels (dynamic range
  1e7-1e8); 4 diffraction spikes NAC / 6 WAC; ~1%-amplitude ghost a few tens
  of px from the peak in NAC GRN filters; WAC IR2/IR1 badly defocused
  (FWHM 4.7 px).
- **Stray light**: structured — streaks perpendicular to the frame edge
  nearest an off-frame source, diagonal corner streaks (NAC), diffuse curving
  bands (WAC, ~1000x worse than NAC); dust "donut" rings (NAC flat field,
  <1% each, accumulating over mission).
- **Read noise** ~12 e-; gain states ~233/95/30/13 e-/DN (NAC). The sim's
  default gain state is 2 (~30 e-/DN NAC, ~28 e-/DN WAC — interim, from the
  calibration-report ratios), the tour-standard science state; a scene
  selects another via `detector.gain_state` (15.6), and the per-state
  `gain_e_per_dn` tables live in `artifacts_catalog.py`. Only state 2 is
  cataloged for the WAC until its full table is sourced; selecting
  another WAC state is a validation error, not a silent guess. Cosmic rays:
  no published ISS rate; use the standard interplanetary ~1-2 events/cm2/min
  over the 1.51 cm2 chip (order 1.5-3 hits per 60 s), flagged as an assumed
  default to be tuned by the Section 7 incidence match.

Sim priorities for navigation realism: partial-line/truncation loss modes,
Huffman alternating-line truncation, 2 Hz banding, hot-pixel columns, PSF
wings, LUT quantization, blooming. (Shutter-shading, CTE, and absolute-flux
drift are photometric effects below navigation relevance.)

### 5.3 Voyager ISS (vidicon)

Sources: Benesh & Jepsen 1978 (JPL 618-802 calibration report), PDS RMS Node
instrument catalogs and volume processing docs, EDR volume documentation,
Owen (DESCANSO optical-navigation monograph), Voyager Neptune Travel Guide
(JPL 89-24). The only non-CCD camera set: a magnetic-deflection vidicon,
whose artifacts are geometric and scene-dependent in ways no CCD's are.

Data-loss / telemetry modes:

| Mode | Shape | Notes |
|---|---|---|
| Missing minor frames | zero-filled whole lines and partial lines (per-line first/last-valid-pixel structure) | common at Uranus/Neptune rates |
| Bit-error spikes (uncompressed era) | isolated single pixels flipped by powers of 2 — salt-and-pepper | the one real per-pixel artifact among the four cameras; a *spike*, not a zeroed marker |
| IDC bit error (Uranus/Neptune) | remainder of the compressed line garbled or lost from the error onward | first-difference variable-length code cannot resync mid-line |
| IDC line clipping | high-entropy lines truncated near the frame edges | fixed per-line bit budget |
| Edited modes (IM4-IM14) | only a centered vertical band of each line downlinked (608/440/272/160/80 of 800 px); rest zero | Jupiter/Saturn realtime |
| IM5/IM15 (2:1 scan) | top half of the frame in one downlink frame, bottom half in the next | half-frame products |
| RAM corruption | spurious values across the frame (`RAM_DATA_CORRUPTION`) | rare |

Vidicon artifacts (all in every raw frame the navigator sees):

- **Reseau marks**: 202 light-insensitive ~3 px squares per camera on a
  ~46 px triangular lattice (dark bars in the extreme corners); their
  measured positions move image-to-image with the distortion field.
- **Geometric distortion**: barrel, RMS ~2.8 px, up to ~14 px in corners,
  and *scene- and time-dependent* (not a fixed map — per-image reseau
  solutions are how the archive corrects it); "pinched" corners the scan
  never reaches.
- **Beam bending**: the readout beam deflects toward stored charge — limb
  positions of a bright disc shift by 1-4 px depending on brightness and
  position, the region above a bright target darkens, and a halo of light
  surrounds saturated regions (electron diffusion). This is a *geometric
  navigation error source*, unique to vidicons and directly relevant to
  limb-fitting accuracy claims for Voyager.
- **Residual image**: ghost of the prior frame when the 14-erase-frame
  light-flood cycle is shortened (fast shuttering sequences).
- **Dark current ramp**: dark signal accumulates during the 48 x n s
  readout, so it grows with line number (and a WAC BOTSIM frame waits a
  full extra readout); nonlinear in wait time; per scan rate (1:1 to 10:1).
- **Shading**: radial sensitivity gradient (brighter response toward frame
  top), filter-dependent flats; fixed blemish/hot-spot population
  (cosmic-ray anode damage) in every image.
- **Noise**: readout-chain noise ~0.3-0.75 DN RMS (low gain) / ~2.2-2.6 DN
  (high gain); faint coherent periodic noise (2.4 kHz vertical component,
  ~0.5 DN peak-to-peak); 8-bit quantization. Not Poisson-dominated —
  per-line-correlated readout noise is the right model.
- **Smear**: limit-cycle drift ~10 urad/s (~1 NAC px/s) dominates the
  15.36 + 48n s Neptune exposures; nodding/maneuver image-motion
  compensation leaves characteristic residuals (stepper-motor jitter in
  MIMC); star streaks with a sharp tracked target, or vice versa.
- **Archive-processing scars** (present in the GEOMED/cleaned products the
  pipeline may consume): reseau-removal smudges, missing-line interpolation
  banding, irregular blank border on the 1000x1000 geometric products.

**What the navigator actually sees (operator decision 2026-07-15):** the
pipeline consumes GEOMED products — reseaus removed, geometric correction
applied, resampled to 1000x1000. The sim therefore targets *GEOMED-level*
Voyager realism, not raw vidicon frames:

- **No reseau marks rendered.** What remains of them is the
  reseau-removal scar: ~3-5 px interpolated smudges on the reseau lattice,
  anomalously smooth patches that sit wherever a mark crossed a limb or
  ring edge.
- **Residual geometry, not raw distortion.** GEOMA correction leaves ~1 px
  internal geometric error (per-image, low-order) rather than the 3-14 px
  raw barrel field — this is the natural amplitude for the optics
  distortion stage (Section 4.4) on Voyager scenes.
- **Beam bending survives partially.** The correction is anchored to
  measured reseau positions, but a bright disc bends the beam differently
  at the limb than at the nearest reseaus, so a brightness-dependent limb
  bias of order a pixel remains. Modeled as a planted limb-position bias vs
  disc brightness.
- **Resampling texture.** GEOMED interpolation softens noise and edges;
  the detector noise should pass through a matching resample so its spatial
  correlation looks like the archive product, plus the irregular blank
  border and missing-line interpolation bands.

Sim priorities (GEOMED-level): line/partial-line loss and interpolation
bands, edited-mode centered bands, beam-bending limb bias, dark-current
line ramp, ~1 px residual geometry, high-gain noise with resampling
correlation, long-exposure smear, reseau-removal scars. Bit-error spikes
justify keeping a salt-and-pepper *spike* mode (distinct from the
zero-marker dropout knob).

### 5.4 Galileo SSI

Sources: Belton et al. 1992 (SSRv 60), Klaasen JPL D-5880 Calibration Report,
PDS REDR SIS and dataset catalogs, Costa & Tong 1994 (ICT), MIT/JPL SSI
electron-sensor study. 800x800 virtual-phase CCD, 8-bit ADC, mechanical
shutter (no readout smear), preflash/light-flood before every exposure.

Data-loss / compression modes (two distinct mission eras):

| Mode | Shape | Notes |
|---|---|---|
| BARC rate-controlled (pre-1991) | DN posterization changing at 64-px horizontal block boundaries (LSB truncation) | entropy-driven |
| BARC information-preserving | lines truncated at the right end — ragged variable-length right margins | Gaspra-era frames |
| Reed-Solomon overflow | whole line garbled/flagged bad | |
| ICT lossy (LGA era, 1996-2002) | 8x8 integer-DCT blocking, ringing around limbs/craters, contouring at high quantization step | ratios ~6:1 to 60:1 |
| ICT/Huffman slice loss | zero-filled horizontal bands exactly **8 lines tall**, typically starting mid-line at the bit error and running right, then whole 8-line stripes | packets are 8-line slices of 8x8 blocks |
| Truth window | one 96x96 losslessly-coded square with visibly cleaner texture than the ICT remainder | |
| Cut-out window | scene confined to a commanded rectangle, hard zero border elsewhere | REDRs stay 800x800 |
| HMA / HCA formats | valid data on every 2nd / every 4th line only — literal jail bars | vertical decimation without summation |
| Summation-readout fault (I24/G28) | left/right halves of the scene folded onto alternating lines with a 13-sample offset | late-mission summation frames |

Detector/environment artifacts:

- **Radiation transients** — the dominant stochastic artifact and a regime
  the generic cosmic-ray model must scale to: ~1e4 spikes/frame (~1.6% of
  pixels) at Ganymede distance in the 8-2/3 s readout, average spike ~4 DN
  in low gain; at Io the radiation background alone reaches ~8 DN in
  summation mode and forced the 4x-faster summation readout. Morphology:
  1-3 pixel hits (secondaries), plus grazing streaks and alpha tracks;
  amplitudes few DN to >120 DN, steeply falling. Rate scales with exposure
  *plus readout dwell* and with distance from Jupiter.
- **Fixed pattern**: bright columns every 33 px (photolithography stitch
  comb); dust donuts; corner vignetting; nine low-full-well columns
  (30-85k e- vs 108k) that bloom early and can spill frame-to-frame;
  blemish population concentrated on the frame edges plus column 170;
  growing hot-pixel ("dark spike") population from RTG neutrons.
- **Bias/dark structure**: coherent vertical bands every 42 px (2400 Hz
  supply noise, ~0.35 DN raw); summation mode adds a left-to-right shading
  ramp; blemish-protection and inverted-clock modes add diagonal shading and
  much higher offsets; <8 Hz horizontal coherent noise in high-gain frames.
- **Quantization**: 8-bit ADC with uneven bin widths, worst at DN multiples
  of 8 — visible contouring in smooth low-contrast scenes (8 bits is poorly
  matched to the CCD's dynamic range).
- **Noise/gain**: read noise 31 e- (full-res) / 44 e- (summation); gain
  states ~1822/377/187/39 e-/DN, sim default state 2 (~187 e-/DN — interim,
  the common science state); full well ~108k e- (432k summed).
- **PSF/optics**: sharp core (sigma ~0.8 px, mildly field-dependent),
  scattered-light wings 1-4% of peak at 100 px from a bright edge, ~2%
  ghost pedestal; pincushion distortion r^3, max ~1.2 px in corners.
- **On-chip mosaics**: multiple exposures / single readout share one
  radiation-hit accumulation, one bias, and can bleed into each other.
- **Shutter**: line-dependent exposure offset (~1.5 -> ~1.05 ms from line 1
  to 800); ~1e-6 closed-shutter light leak.

Sim priorities: radiation-transient regime (rate x distance x readout mode),
ICT blocking + 8-line slice loss, cut-out/truth windows, jail-bar formats,
33-px and 42-px fixed-pattern combs, early-blooming columns, 8-bit
contouring. (BARC-era, OCM, and the I24/G28 fold are documented but
lower-priority scene classes.)

### 5.5 New Horizons LORRI

Sources: Cheng et al. 2008 (SSRv 140), Weaver et al. 2020 (PASP 132), NH
SOC-to-Instrument ICD (PDS), Lauer et al. 2021/2022. Frame-transfer
1024x1024 CCD, no shutter, antiblooming; thruster-only pointing.

Data-loss / downlink modes:

| Mode | Shape | Notes |
|---|---|---|
| Packet dropouts | horizontal zero-filled strips ("gaps") | flagged missing; ground pipeline may median-fill |
| Windowed downlink | only commanded rectangular subframes present; the rest zero-filled | the banded Pluto quicklook look |
| Lossy (DCT) downlink | 8x8 blockiness/ringing; embedded-header corruption spreads over the first 40 px x 8 rows corner | many encounter frames exist in lossy then lossless re-downlink versions |
| Embedded header | first 34 pixels of row 0 are binary housekeeping in *every* image | never scene data |

Detector/optics artifacts:

- **Frame-transfer smear**: the defining LORRI artifact. Vertical
  full-column pedestals through bright targets from the ~12 ms pre-exposure
  scrub and ~11 ms post-exposure transfer; pedestal differs above vs below
  the target; comparable to the scene signal for exposures near 10 ms; ~10%
  column pedestal at the nominal 100 ms. Desmear fails through saturated
  columns, leaving residual vertical banding.
- **Saturation morphology**: antiblooming CCD, so *no* column bloom; a
  hard-saturated compact source instead shows a clipped 4095 DN core, a
  bright-then-dark **horizontal** serial tail (amplifier undershoot, up to
  ~12 DN), plus the vertical smear column.
- **Pointing smear**: no reaction wheels — deadband drift ~5 arcsec/s
  (~1 px per 100 ms; multi-pixel trailing at longer exposures) in random
  direction; ~2 arcsec jitter kernel in relative-control mode (65 s 4x4
  deep frames); 10 ms exposures during scan-platform support.
- **PSF**: undersampled and asymmetric — in-flight X-FWHM 2.06 px,
  Y-FWHM 2.65 px, persistent lower-right excess flux; peak pixel holds only
  ~14% of the flux (1x1) so sub-pixel phase matters; stable over the
  mission.
- **Hot/dead pixels: none** — all PDS hot/dead maps are zeroes. Transient
  "warm" pixels (few DN, <1% of pixels, epoch-variable) appear only in long
  4x4 exposures. Dark current negligible (~0.04 e-/s/px). No measurable CTE
  degradation.
- **Fixed pattern**: ~4% corner vignetting, ~1% dust donuts, ~62 dark
  single-pixel specks (~20% deep), 0.9% rms PRNU; jail bars (<=0.5 DN
  even/odd column offset whose sign flips per power cycle); <=1 DN random
  horizontal striping; ~0.8 DN vertical banding near low column numbers.
- **Ghosts/stray light**: arc-shaped ghosts from bright sources just outside
  the FOV (within ~0.37 deg); ~1.5% central lens-group ghost over a ~200 px
  radius; solar stray light floods frames at solar elongation below
  ~13-25 deg (distance-dependent).
- **Noise/ADC**: gain ~21 e-/DN (1x1) / 19.4 (4x4), read noise ~1.1 DN
  (~23 e-), bias ~545 DN estimated per-image from 4 dark columns
  (temperature-dependent), 12-bit clip at 4095.
- **Cosmic rays**: ~16 +/- 13 hits per full frame per (readout-dominated)
  short exposure; mostly single pixels, occasional grazing streaks.

Sim priorities: frame-transfer smear + desmear-residual behavior, pointing
drift trailing, undersampled asymmetric PSF, windowed/gap zero-fill,
saturated-source serial tails. LORRI is also the clean counterexample for
the framework: an instrument whose artifact block *disables* hot pixels and
column bloom.

### 5.6 Cosmic rays (all CCDs)

The current model (single-pixel lognormal spikes) is replaced by morphological
events: point hits, short streaks with random angle and length drawn from an
incidence-angle distribution, and rare multi-pixel splatters; rate scales with
exposure and per-instrument environment (Galileo's magnetospheric rates are a
separate regime). Single-pixel spikes remain a degenerate case.

---

## 6. Scene schema changes

Schema version bumps to 2, and version 2 is the only version the loader
accepts. The scene catalog is converted in place (Section 15.1); nothing
reads or writes version 1 afterwards. Keys are partitioned into
**idealized** and **truth** blocks per Section 3.2, and the 15.6 inventory
tags every key with its side. Three things author scenes, and each converts
in the phase that changes its keys: the catalog under
`tests/integration/sim_scenes/`, the `_GUI_GALLERY` / `_REPORT_SCENES`
definitions in `tests/integration/sim_doc_images.py`, and the calibration
campaign generator `util/calibration/scene_gen.py` — which today feeds its
dicts straight to the renderer, bypassing the validator, and starts routing
them through `load_sim_scene`'s validation in phase A so campaign scenes
cannot silently drift from the schema. New/changed keys:

- body: `limb_relief_rms`, `limb_relief_corr_deg`, `photometric_law` (+ law
  params), `albedo_texture` block, `atmosphere` block, `shape_model:
  cratermaker` variant with its parameters, mesh smooth-shading params.
- star: `navigable`, `catalog_error_v`, `catalog_error_u`, `companion`
  (separation + delta mag), `delta_mag`; scene-level
  `star_catalog_scatter_px`.
- top level: `spk_error` block (planted spacecraft-position error applied
  render-side to bodies/rings, scaled 1/range); per-object-class `smear`
  motion vectors (differential smear).
- rings: `ring_system` block (list of features with `kind`, `tau`, orbit
  modes, `edge_wave`, planted `orbit_error`, `navigable`, `phase_function`,
  shared `geometry` with opening angle) replaces the `rings:` list. On the
  integration branch, `rings:` remains valid until phase F, which converts
  the two `rings:` scenes (and `scene_gen.py`'s ring family) to two-edge
  ringlet features with `navigable: true` and deletes the key; `main`
  never sees a v2 schema that still accepts `rings:`.
- top level: `optics` block (`psf`, `smear`, `ghosts`, `distortion`),
  `artifacts` block (per-mode incidence overrides), `oversample`.
- `noise` keeps its meaning (detector stage parameters).

Scene classes gain `artifact_sweep`, `ring_system`, `star_confounder`,
`mutual_event`, and `expected_fail` families in `DECLARED_SIM_SCENE_CLASSES`;
the sweep runner gains the new axes (Section 8). The complete key
inventory with types and defaults is Section 15.6; this section is the
overview.

---

## 7. Proving realism (the validation half of #227)

Realism is demonstrated distributionally against the WS-3 image-library
cohort, per instrument, using statistics the techniques actually consume. No
pointing truth is needed, so Voyager/Galileo qualify — for them this match is
the *only* absolute-accuracy evidence.

Figures of merit (each reported per instrument, sim vs real, as
distribution overlays plus a scalar divergence):

1. **Noise statistics**: sky-region histogram (mean/sigma/skew), noise vs
   signal level estimated by local differencing — paired row/pixel
   differences within near-uniform patches, since science frames have no
   flat pairs and naive signal-binning conflates scene texture with noise —
   and the spatial power spectrum of a sky region (catches banding/coherent
   noise).
2. **PSF / encircled energy**: star-cutout radial profiles where star frames
   exist. All four instruments have them: Cassini and LORRI in the library
   already, and Galileo and Voyager via the star-field frames of the #228
   FOV-twist residual analysis (`experiments/fov_twist/config/` lists 18
   GOSSI star-calibration frames and ~70 twist-verified VGISS frames
   spanning both spacecraft and cameras), a selection of which joins the
   library as `star_dominated` frames (curation batch 006). Where a cohort
   still lacks star frames for some camera/mode, fall back to the
   limb-gradient-based proxy and say so.
3. **Limb gradient profile**: normalized profile across real vs sim limbs,
   binned by resolution and phase (this is what BodyLimbNav's DT actually
   sees). This is a *joint* match of PSF + limb topography + photometric
   law: FOM 2 pins the PSF independently where star frames exist, and the
   report states the attribution limit where it cannot.
4. **Ring-edge gradient profile** and radial brightness profile across known
   features, lit and unlit side (Cassini; Voyager where usable).
5. **Dynamic range / exposure statistics**: fraction saturated, fraction near
   bias, signal percentiles — compared on exposure/gain-stratified or
   matched-pair samples; an unstratified cohort comparison measures what the
   spacecraft pointed at, not the forward model.
6. **Artifact incidence**: measured rates of missing lines / partial lines /
   truncation / hot pixels / cosmic rays in the cohort vs the catalog
   defaults (this both tunes the defaults and documents them).
7. **Technique-diagnostic distributions** (#153's real-vs-sim comparison):
   for matched scene/frame pairs, the per-technique confidence diagnostics
   (inlier fractions, residual scatters, SNR proxies) drawn from sim vs real
   frames. **Read-only diagnostic — never a tuning target** (see the
   tuning-loop rule below).

All figures of merit are computed in the cohort's units — I/F for the CALIB
cohorts — which is why the calibrated-path detector chain (Section 15.2) is
phase B scope rather than deferred work.

Deliverable: `tests/integration/sim_realism.py` (runner producing the match
report + figures) and a `realism` section in the simulator report presenting
the per-instrument match quality. Cohort reality as of 2026-07-15: the
library holds 62 Cassini / 8 Galileo / 2 Voyager / 2 LORRI frames, with a
9-frame GOSSI/VGISS `star_dominated` selection (curation batch 006, from
the FOV-twist lists) in operator review — so this is a Cassini match
first, with FOM 2 reachable on every instrument once batch 006 lands. The runner ships for all four instruments, reports
each against whatever cohort WS-3 has landed, and — where the cohort cannot
support a statistic (an IQR of two frames is not a statistic) or no
independent PSF/shape source exists — says so and labels that instrument's
sim accuracy as bounded by unverified forward-model fidelity, per the WS-2
acceptance criteria.

Tuning loop: adjust forward-model defaults (PSF wings, noise, artifact
incidence) until FOMs 1-6 match; every tuned value is recorded in
`artifacts_catalog.py` with its provenance. **FOM 7 is excluded from the
loop**: FOMs 1-6 are navigator-independent, but FOM 7 is built from the
navigator's own outputs, and tuning the image side until the navigator's
diagnostics agree would re-admit circularity through parameter fitting —
invisible to every structural test. FOM 7 is reported, not fitted.

---

## 8. Sweep axes and tests

`tests/integration/sim_sweep*` gains controlled model-error axes; the
recovery error vs mismatch curve is the product (issue #227's core ask):

| Axis | Render side | Navigate side |
|---|---|---|
| PSF mismatch | empirical PSF, wings, jitter | navigator PSF sigma |
| Shape mismatch | limb topography / CraterMaker / mesh | ellipsoid (or wrong-pose mesh) |
| Photometric mismatch | Lommel-Seeliger / Minnaert / surge | Lambert |
| Ephemeris error (body) | planted center offset | catalog center |
| Ephemeris error (ring) | per-feature planted orbit error | catalog orbits |
| Ephemeris error (spacecraft) | SPK error: bodies/rings parallax-shifted by 1/range, stars unmoved | consistent geometry |
| Differential smear | per-object-class motion vectors (tracked target sharp, stars trailed) | unsmeared predictions |
| Ring clutter | full ring system rendered | navigable subset only |
| Star clutter | full star field rendered (1/2/3-star lock regimes) | navigable subset only |
| Star catalog error | stars rendered off-catalog (small: tolerance sweep; large: expected-fail) | catalog positions |
| Mutual event | overlapping bodies, swept occultation fraction | full limbs of both bodies |
| Atmosphere | haze limb (Titan-class) | hard limb at reference radius |
| Detector noise | full stage 4/5 stack on every scene, including the I/F path | n/a (noise is never modeled) |
| Artifact stress | structured loss at swept incidence | n/a |

Zero-mismatch columns are labeled "self-consistency floor (not accuracy)"
everywhere they appear. A zero-mismatch scene is defined by *equality with
the navigator's configuration*, not by the 15.3 empirical kernels: the
image-side PSF is a **pure Gaussian** (w = 0, sigma_v = sigma_u, no field
variation — exactly the navigator's model) at the instrument's configured
`star_psf_sigma` (WAC's configured 0.77 vs the empirical 0.64 — and the 3.0
placeholders on gossi/vgiss/nhlorri — would otherwise plant a hidden PSF
mismatch into the floor itself, and empirical wings the navigator does not
model are themselves a mismatch).

Section 13 holds the axes reviewed but left unscoped in the 2026-07-15
triage (mid-time error, field-rotation rate, rolling-readout shear,
pole/rotation-phase/radius errors, and others); they are recorded there for
a later pass.

Test layers:

- Unit: each forward stage (deterministic given seed; artifact geometry
  checks — a missing line is a *line*, blocks are aligned, hot pixels are
  stable per seed).
- Structural: the information-boundary test (Section 3.2) — every truth key
  in the 15.6 inventory unreachable through the filtered view — plus
  ordinary import hygiene.
- Integration: scene-class sweeps as today, plus `artifact_sweep` (navigation
  quality vs artifact incidence — this quantifies how much structured loss
  the techniques tolerate, which per-pixel dropouts overstate),
  `ring_system` and `star_confounder` scenes (clutter/false-lock regimes),
  `mutual_event` scenes (occluded-limb degradation), and `expected_fail`
  scenes (asserting failure/low confidence, never a confident wrong offset).
- Realism: the Section 7 runner.

`#223` sequencing inside this plan: `NavModelBodySimulated` emits
`TERMINATOR_ARC` in phase I (feature work, ungated), and phase J refits
`BodyTerminatorNav` alongside the rest. The terminator realism verdict —
Section 7 figure-of-merit 3 computed on the terminator side achieving the
same divergence band the limb side achieves for Cassini, a phase-H
deliverable — gates *trust*, not emission: until it passes, the refit
technique's confidences stay `confidence_provisional`, the same interim
safeguard every uncalibrated confidence already carries. H therefore still
does not block the branch.

### Support for WS-0 (#224): estimator validation scenes

WS-0 validates the cross-technique agreement estimator on truth-known
frames before any real-image per-technique claim is trusted. This plan must
supply its raw material:

- **Composition families.** Scene sets matching the identifiability map's
  rows — limb+disc, limb+disc+ring, multi-body — where several techniques
  run on one frame and each technique's true error is known by
  construction. The existing `multi_body_geometry` class extends to cover
  these; the ring-system and mutual-event classes add the compositions the
  current sim cannot build.
- **Seed ensembles.** The covariance-components solve needs a *population*
  of frames per composition: the sweep runner gains an ensemble mode that
  replicates one geometry across N seeds (noise, artifacts, confounders
  redrawn; geometry fixed) with per-technique truth recorded per frame.
- **Stage sequencing.** WS-0 Stage 0a (recover-known-error, identifiability)
  needs only a truth-known sim and can run on today's renderer in parallel
  with this plan. Stage 0b (shared-bias injection) injects through the
  navigator's shared preprocessing layer (`image_derivatives`, the noise
  sigma estimate, the reliability gate) — an orchestrator-side harness, not
  an image-side one — but its scenes must sit in *sensitive* regimes (noise
  near gate thresholds, gradients near DT-convergence limits, shared-layer
  products actually load-bearing) or the induced correlation is
  unmeasurable. Phases A-B are Stage 0b's prerequisite; the ensemble mode
  and composition families are deliverables of phase A (runner) plus the
  domain phases that own each composition.

---

## 9. Documentation and GUI

Everything in Sections 4-8 changes what an operator authors and what a
developer maintains, so the scene editor and the prose ship with the work
rather than after it. Each phase lands its own GUI controls and doc edits;
the phase is not done until they are (Section 15.10).

### 9.1 Simulator report

`docs/simulator_report/simulator_report.rst` is restructured so that:

- every accuracy number is presented as a function of model mismatch (the
  Section 8 curves);
- the zero-mismatch column is explicitly labeled the self-consistency floor;
- the per-instrument realism match (Section 7) is presented first, as the
  precondition for reading the mismatch curves as accuracy;
- artifact incidence defaults and their cohort provenance are tabulated.

### 9.2 Developer guide

`docs/dev_guide/dev_guide_simulator.rst` is the chapter of record for the
simulator and needs the largest revision, because the thing it describes
changes shape:

- the two-sided architecture (Section 3): what `sim/forward/` is, what the
  navigator-side models are, what the information boundary means — which
  scene keys the navigator may see and why sharing code is fine while
  sharing information is not — and the boundary test that enforces it. A
  developer who does not understand this distinction will destroy the
  property the plan exists for, so it leads the chapter.
- the stage pipeline (Section 3.3) and the stage interface (15.2), including
  the unit chain from scene signal to DN and the per-stage seeding rule.
- the schema v2 key inventory (15.6), replacing the v1 documentation.
- per-instrument artifact catalogs (Section 5) and how to add an instrument.
- the standing prohibition on reading navigator-side renderers while
  implementing `sim/forward/` (15.11), stated as a rule with its rationale.

`docs/dev_guide/_sim_images/` (11 committed PNGs, regenerated by
`python -m tests.integration.sim_doc_images` from `_GUI_GALLERY`) is
re-rendered as the renderers change, and gains panels for the new
ingredients: topographic limb, ragged terminator, ring system at a non-polar
opening angle, structured telemetry loss, mutual event. The same generator
also writes `docs/simulator_report/_scene_images/` (10 PNGs, from
`_REPORT_SCENES`, rendered from the catalog YAML) — both galleries
re-render whenever the catalog or the renderers change, as deliverables of
the phase that changed them. The generator's scene definitions move to
schema v2 with the catalog.

`docs/api_reference/api_sim.rst` gains the `spindoctor.sim.forward.*`
modules as they land.

### 9.3 User guide

`docs/user_guide/user_guide_simulated_images.rst` is short and orienting,
and its framing survives, but two claims in it do not:

- it says the simulator exists to test and validate the navigation pipeline
  because a simulated frame's offset is known by construction. That stays
  true, but it must no longer imply the resulting numbers are accuracy: the
  page gains the self-consistency-floor distinction in plain language, and
  points at the realism match as what makes sim numbers credible.
- it points readers at the dev guide for the scene formats and the GUI;
  those pointers stay, but the scene-format story is now schema v2.

The four instrument appendices (`user_guide_appendix_*.rst`, currently
placeholders tracked as #93) are where the per-instrument artifact
descriptions of Section 5 belong if #93 is executed in this window; this
plan does not depend on that and does not block on it.

### 9.4 Scene editor GUI

`src/spindoctor/cli/sd_create_simulated_image.py` (3078 lines) is the scene
editor: per-parameter spin boxes and per-body tabs wired to a live render.
Schema v2 roughly doubles the authorable surface, so the GUI is **in
scope**, with three rules:

- **Every scene key is reachable from the GUI.** A key that can only be
  authored by hand-editing YAML is a key operators will not use, which
  wastes the realism work that produced it.
- **The editor round-trips v2 without loss** — load, edit one field,
  re-save, and every other block survives byte-identical in meaning. This is
  a test, not an aspiration.
- **Controls land with their phase**, grouped to mirror the schema: an
  `Optics` tab (PSF, smear, distortion, ghosts), an `Artifacts` tab (per-mode
  incidences + `instrument_defaults`), new body-tab groups (limb relief,
  photometric law, atmosphere, texture), a ring-system tree replacing the
  flat ring list, and star-block additions.

The file is already 3x the project's 1000-line module ceiling, and this work
makes that worse. Phase A splits it into a `cli/sim_editor/` package
(widgets per schema block, one module each) as the precondition for the
per-phase control additions. That split is mechanical but large; it is
called out here so it is budgeted rather than discovered.

---

## 10. Phasing and delivery shape

The series lands as one PR per phase against an integration branch
(`rf_sim_realism`), mirroring the `rf_core_rewrite` series that delivered
the navigation core: each phase PR keeps the full suite green on the
branch, the controller (15.11) reviews and merges each one, and the whole
branch squash-merges to `main` once — so `main` never sees an intermediate
schema state, and the intra-branch staging below is ordinary sequencing,
not a compatibility shim. Baselines, tier files, and doc galleries are
regenerated inside whichever PR changes their renders. Each phase also
lands its own GUI controls (Section 9.4) and its own dev-guide and
user-guide edits (Sections 9.2-9.3) — the feature, the control, and the
prose are one unit of work, not three.

- **A. Skeleton + boundary.** `sim/forward/` package with stage
  interfaces, its body/ring renderers seeded by porting the current logic
  at present fidelity (3.1 — E and F replace the ports);
  `render_combined_model` re-plumbed as a thin driver;
  `sim_body*`/`sim_ring` re-homed under `nav_model/`; the
  information-boundary filter (`obs.nav_params`) with its structural test
  and the star-list replumb (3.2); the `ring_epoch` fix (15.6); the
  render-diff contact-sheet generator
  (`tests/integration/render_contact_sheet.py`, 15.11) built and its
  first sheet committed; scene catalog converted to schema v2 (ring
  scenes keep `rings:` until F) and re-rendered with the diff reviewed
  via that sheet; `scene_gen.py` routed through the validator; the
  calibration-campaign timing baseline recorded (15.9);
  `sd_create_simulated_image.py` split into `cli/sim_editor/`; dev-guide
  architecture chapter rewritten.
- **B. Scene-level optics + detector core.** Whole-scene PSF, smear
  (including differential per-object-class smear), SPK-error geometry knob,
  electrons/gain with the exposure-referenced unit chain (15.2), dark +
  hot pixels, banding, quantization, electron-domain bloom, cosmic-ray
  morphology, and the calibrated (I/F) path: `data_units: calibrated_if`
  scenes render through the full DN chain and then the calibration
  transform, so calibrated products carry propagated noise (closes the
  Section 2 last-row gap; required by Section 7's I/F cohorts). Schema v2
  `optics`/`artifacts`/`spk_error`/`detector` blocks.
- **C. Telemetry artifacts.** Structured loss modes + per-instrument
  catalog defaults from Section 5 tables; adversarial placement mode;
  `artifact_sweep` scene class; incidence figures of merit.
- **D. Stars.** Flux normalization, empirical PSF hookup, star-count sky,
  navigable subset + planted catalog error, saturated navigable stars,
  double/variable stars; `star_confounder` and `expected_fail` scene
  classes (1/2/3-star lock regimes and wrong-catalog failure scenes); the
  `expected`-block assertion machinery (15.8); `scene_gen.py` star
  families converted. Star occultation lands with phase F's compositing.
- **E. Bodies.** 2-D relief field, limb application + terminator
  raggedness + local shadowing (15.5), photometric laws, albedo texture,
  planet disc texture + satellite/shadow transits, mesh smooth shading
  (#158), `mutual_event` scene class (partial occlusion sweeps).
- **F. Rings.** Ring-system spec, ring-plane projection with the 15.4
  longitude and depth conventions, tau photometry lit/unlit, transparency
  compositing (#84 — which also delivers star occultation), per-feature
  orbit errors, edge modes/waves, spokes, embedded moonlets/propellers,
  navigable subset; `ring_system` scene class; the two `rings:` scenes
  (`planted_offset_ring`, `planted_rank1_ring_flat`) and `scene_gen.py`'s
  ring family converted, `rings:` and `shade_solid_rings` removed.
- **G. Atmospheres.** Haze-limb bodies (Titan-class), the sim substrate
  for issue #60.
- **H. Realism match.** Section 7 runner + tuning + per-instrument match
  report. The only phase gated on data outside this plan's control: it
  needs the WS-3 cohort (operator-supplied), ships the runner and the
  Cassini match, and reports the other instruments against whatever cohort
  exists, labeled per Section 7.
- **I. Sweeps + report.** Section 8 axes wired into sim_sweep, simulator
  report rewrite (9.1), user-guide revision (9.3), both doc galleries
  re-rendered, `TERMINATOR_ARC` emission (#223's feature half).
- **J. Recalibration.** The shipped confidence calibration is
  sim-anchored: the alphas and normalizations in
  `config_510_techniques.yaml`, the `model_error_floor_px` tunables, and
  the tier boundaries in `config_540_orchestrator.yaml` were fitted
  (2026-07-09/10, `util/calibration`, campaign seed 20260709) against the
  renderer this plan replaces, and are invalid the moment phases B-F
  change star brightness, limb gradients, and ring edges. Phase J re-runs
  the calibration campaign on the new renderer and refits in the
  `util/calibration` README's documented order: per-technique alphas
  (`fit.py`), written into the YAML by hand with curated comments, then
  re-collect (fused confidences depend on the alphas), then the
  model-error floors (`fit_floors.py`, the ~0.865 2-sigma coverage
  solve), then the tier boundaries and final gate (`fit_gates.py` — one
  tool, one step). It updates the provenance headers and includes the
  #223 `BodyTerminatorNav` refit once phase I emits its features. **The
  branch does not merge to `main` with stale sim-anchored coefficients.**

Dependencies: A then B; C-G in any order after B; I after C-G; H whenever
the cohort allows (it does not block the branch — the terminator verdict
it produces gates trust labels, not emission; Section 8); J after
everything else on the branch (it recalibrates against the renderer B-G
built and refits the technique I feeds). Per-phase acceptance criteria are
Section 15.10.

Phase A is deliberately the heaviest: it carries the architecture, the
boundary, the conversion, and the GUI split at once. They are one phase
because each is churn over the same files and the same 30 scenes, and
sequencing them means converting the catalog twice.

---

## 11. Acceptance criteria (from #227, restated against this design)

1. No truth parameter — planted error value, variance field, artifact
   placement, or unnavigable object — is reachable by any navigator-side
   model: enforced by the Section 3.2 information-boundary test. (Shared
   geometry helpers are permitted by design; the boundary is informational,
   not authorial.)
2. The simulator report presents error vs PSF/shape/photometric/ephemeris
   mismatch, realism-match evidence per instrument, and a labeled
   self-consistency floor.
3. The simulated-vs-real distributional match is quantified and reported per
   instrument for the features each technique consumes; sim accuracy is
   presented as credible only to the degree the match supports.
4. Structured telemetry/detector artifacts exist for all four instruments
   with cohort-derived incidence defaults; per-pixel dropout remains only as
   a generic stress knob.
5. Bodies can render non-ellipsoidal limbs, ragged terminators, and partial
   mutual occultations; rings can render multi-feature systems with planted
   inter-feature orbit error and a navigable subset; star fields can render
   confounder stars and off-catalog stars the navigator is not told about;
   all are sweep axes.
6. Expected-to-fail scenes (wrong catalog, overwhelming clutter) exist and
   the navigator's correct behavior on them — failure or low confidence,
   never a confident wrong offset — is asserted in the integration suites.
7. Every schema v2 key is authorable from the scene editor, which
   round-trips v2 scenes without loss; the developer guide describes the
   two-sided architecture and the schema as built; the user guide no longer
   lets a reader mistake a self-consistency number for accuracy. Exactly one
   scene schema version exists in the tree.
8. The confidence calibration that ships on `main` is fitted against the
   renderer that ships with it (phase J) — never against the replaced one.

---

## 12. Instrument references

Sources consulted for the Section 5 artifact catalogs. Citations in the
per-instrument sections refer to these entries.

### Cassini ISS

- Porco, C. C., et al. 2004, "Cassini Imaging Science: Instrument
  Characteristics and Anticipated Scientific Investigations at Saturn,"
  Space Science Reviews 115, 363-497.
  <https://ciclops.org/sci/docs/CassiniImagingScience.pdf>
  (Sec. 3.9 compression/8-bit conversion, 3.10 noise, 3.11 dark/RBI/coherent
  noise, 3.12 performance, 3.13 calibration; Tables VII, XII.)
- Knowles, B., 2018, "Cassini Imaging Science Subsystem (ISS) Data User's
  Guide," PDS Imaging Node.
  <https://pds-imaging.jpl.nasa.gov/documentation/iss_data_user_guide_180916.pdf>
  (Sec. 2.3 instrument, 3.3 line prefix/partial-line segments, 3.6 anomalies
  incl. truncated images and the Double Bit Error anomaly, 4.2-4.4
  calibration/CISSCAL; Appendix A tables incl. per-filter PSF FWHM.)
- West, R., et al. 2010, "In-flight calibration of the Cassini imaging
  science sub-system cameras," Planetary and Space Science 58, 1475-1488.
  <https://isis.astrogeology.usgs.gov/8.1.0/Application/presentation/Tabbed/cisscal/assets/Cal_paper_submitted_5_24_2010.pdf>
  (Sec. 3 RBI/hot-pixel populations, 4 flats/dust rings, 5 CTE, 9 PSF wings
  and ghosts, 10 stray-light morphology; Tables 1, 7, 8.)
- Knowles, B., et al. 2020, "End-of-mission calibration of the Cassini
  Imaging Science Subsystem," Planetary and Space Science.
  <https://www.sciencedirect.com/science/article/abs/pii/S003206331930460X>
  (Sensitivity decline and end-of-mission hot-pixel/flat/PSF updates.)

### New Horizons LORRI

- Cheng, A. F., et al. 2008, "Long-Range Reconnaissance Imager on New
  Horizons," Space Science Reviews 140, 189-215.
  <https://arxiv.org/abs/0709.4278>
  (Detector/readout parameters, frame-transfer timing and desmear, bias
  structure, flat-field features, central ghost, bench PSF, early cosmic-ray
  rate.)
- Weaver, H. A., et al. 2020, "In-Flight Performance and Calibration of the
  LOng Range Reconnaissance Imager (LORRI) for New Horizons," PASP 132,
  035003. <https://arxiv.org/abs/2001.03524>
  (In-flight PSF X/Y FWHM and asymmetry, smear timing, jail bars/striping,
  warm pixels, solar stray-light thresholds, pointing modes.)
- "New Horizons SOC to Instrument Pipeline ICD," SwRI 05310-SOCINST-01,
  Section 9 (LORRI), PDS Small Bodies Node.
  <https://pds-smallbodies.astro.umd.edu/holdings/nh-j-lorri-3-jupiter-v1.0/document/soc_inst_icd/soc_inst_icd.pdf>
  (Raw formats/dark columns, amplifier undershoot, desmear matrix, downlink
  APIDs, embedded 34-pixel header, QUALITY_MAP bits.)
- "New Horizons LORRI Instrument Overview" and the LORRI Pluto-encounter
  calibrated collection overview, PDS4, Small Bodies Node.
  <https://pdssbn.astro.umd.edu/holdings/pds4-nh_documents-v4.2/lorri/documents/lorri_inst_overview.pdf>
  (Hot/dead-pixel maps all zeroes, windowing/gap handling, lossy-header
  corruption, dual lossy/lossless downlink versions.)
- Lauer, T. R., et al. 2021, ApJ 906, 77 <https://arxiv.org/abs/2011.03052>
  and 2022 <https://arxiv.org/abs/2212.07449>
  (PRNU, power-on transient, dark-current measurement, off-axis ghost/scatter
  quantification, cosmic-ray handling in deep 4x4 stacks.)
- JHU/APL SOC, "About the LORRI images" artifact gallery.
  <https://pluto.jhuapl.edu/soc/Pluto-Encounter/lorri_about.php>
  (Saturated-source serial tails, ghost arcs, cosmic-ray morphology.)

### Voyager ISS

- Benesh, M., & Jepsen, P. 1978, "Voyager Imaging Science Subsystem
  Calibration Report," JPL 618-802.
  <https://pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5101/DOCUMENT/REPORTS/BENESH_JEPSEN_1978.PDF>
  (Reseau grid, distortion magnitudes, beam-bending measurements, shutter
  behavior, random/coherent noise levels, scan rates.)
- PDS RMS Node Voyager ISS instrument catalogs.
  <https://pds-rings.seti.org/voyager/iss/inst_cat_na1.html> (and wa1/na2/wa2)
  (Camera parameters, instrument modes and edited-frame pixel widths,
  shading, shutter modes.)
- PDS RMS Node volume processing documentation: VGISS_5101 (Jupiter),
  VGISS_7201 (Uranus), VGISS_8201 (Neptune) `DOCUMENT/PROCESSING.TXT` and
  `TUTORIAL.TXT` under <https://pds-rings.seti.org/holdings/volumes/>.
  (Dark-current wait-time model, missing/partial-line handling and
  interpolation, reseau/blemish removal, calibration residuals.)
- Voyager EDR CD documentation, VG_0001 `DOCUMENT/VOLINFO.TXT`.
  (Per-line engineering records, edit-mode IDs, data-anomaly types,
  compressed .IMQ format.)
- Owen, W. M., "Spacecraft Optical Navigation," JPL DESCANSO monograph.
  <https://descanso.jpl.nasa.gov/monograph/series15/Spacecraft-Optical-Navigation.pdf>
  (Vidicon beam bending, pinched corners, saturation halos, blemish
  hot spots, vidicon astrometric limits.)
- Ludwig, R., & Taylor, J., "Voyager Telecommunications," DESCANSO.
  <https://voyager.gsfc.nasa.gov/Library/DeepCommo_Chapter3--141029.pdf>
  and Kohlhase (ed.), "The Voyager Neptune Travel Guide," JPL 89-24.
  (Image data compression design, bit-error behavior, Reed-Solomon,
  image-motion-compensation modes and their smear signatures.)
- Danielson, G. E., et al. 1981, JGR 86, 8683 (radiometric baseline) and
  the RMS Node 1986-1990 recalibration memos.
- Per-image reseau/geometry tables, e.g. VGISS_5101
  `DATA/C13854XX/C1385455_RESLOC.TAB` / `_GEOMA.TAB` (202-mark counts and
  measured-vs-nominal displacements verified directly).

### Galileo SSI

- Belton, M. J. S., et al. 1992, "The Galileo Solid-State Imaging
  experiment," Space Science Reviews 60, 413-455.
  <https://articles.adsabs.harvard.edu/pdf/1992SSRv...60..413B>
  (Detector/gain/full-well/read-noise parameters, BARC compression,
  radiation-spike predictions at 15 Rj and 5 Rj, fixed-pattern features,
  preflash/residual-image design, PSF/MTF, distortion.)
- Klaasen, K. P., "Galileo SSI Calibration Report Part 2," JPL D-5880,
  1993 (PDS).
  <https://planetarydata.jpl.nasa.gov/img/data/go-cal-ssi-6-v1.0/go_0001/document/calrpt2.pdf>
  (Zero-exposure offset structure and mode dependence, coherent-noise combs,
  low-full-well column table, uneven ADC bins, scattered light/ghost,
  shutter offset and light leak, calibrated-noise tables.)
- SSI REDR Software Interface Specification and CD volume SIS (White 1997;
  Mortensen 1997), PDS.
  <https://planetarydata.jpl.nasa.gov/img/data/go-j_jsa-ssi-2-redr-v1.0/go_0023/document/redrsis.htm>
  (Line-prefix loss bookkeeping, ICT 8-line slice loss, truth/cut-out
  windows, HMA/HCA line-skip formats, bad-data record classes.)
- PDS dataset catalog GO-J/JSA-SSI-2-REDR-V1.0 (dataset.cat, inst.cat) and
  "Tracking GLL SSI Bad-Data Values" (Yagi, JPL IOM 384-91-3, baddata.txt).
  (Phase-2 telemetry formats, dark-current rates, dropout handling.)
- Costa, M., & Tong, K. 1994, "A Simplified Integer Cosine Transform and
  its Application in Image Compression," TDA Progress Report 42-119, JPL.
  (ICT matrix, quantization, observed artifact behavior on radiation-noisy
  scenes.)
- "Galileo SSI G28 HIS Anomaly Description and Recovery Algorithm," PDS
  go_0023 document volume. (The I24/G28 summation-readout fold artifact.)
- Klaasen, K. P., et al. 1997, Opt. Eng. 36, 3001; 1999, Opt. Eng. 38(7);
  2003, Opt. Eng. 42, 494 (abstract-level: in-flight radiation transients
  "at about the expected levels," stable calibration; full texts paywalled).
- Carlton, A., et al. (MIT/JPL), "Using the Galileo SSI as a Sensor of
  Jovian Energetic Electrons." <https://dspace.mit.edu/handle/1721.1/114747>
  (Gain conversions, hit-to-pixel multiplicity, in-flight DN histograms of
  radiation hits.)

---

## 13. Candidate additional effects (brainstorm remainder — not scoped)

Everything in this section is a way a real frame differs from a perfect
model that Sections 4-5 do not cover. A triage pass (operator, 2026-07-15)
promoted the following into scope, and they now live in their host
sections: spacecraft SPK error and differential smear (Section 4.4), star
occultation / saturated navigable stars / double and variable stars
(Section 4.3), planet disc texture and satellite/shadow transits
(Section 4.1), ring spokes and embedded moonlets/propellers (Section 4.2),
adversarial artifact placement (Section 5.1), and plate-scale error (the
distortion stage's first-order term, Section 4.4). The Voyager GEOMED
processing scars were already in scope (Section 5.3).

What remains below was reviewed and deliberately left unscoped — kept as a
recorded menu for a later pass, not deleted, because each item is real.

### 13.1 Geometry, pointing, and timing (remainder)

These produce *internally inconsistent* scenes — geometry errors no single
`(dv, du)` can fix. The SPK-error axis now in scope covers the most
important case; these are the others.

- **Planted mid-time error.** A wrong exposure mid-time displaces every
  object along its own apparent-motion vector: rings by their orbital rate,
  each body by its relative rate, stars by the pointing drift only. Distinct
  signature from SPK error (displacement along track vs parallax). Real
  causes: SCLK/SCET drift, shutter-offset error (documented for all four
  cameras), label rounding.
- **Field rotation during exposure.** A roll rate about the boresight turns
  every point source into an arc whose length grows with distance from the
  roll center — the corners smear while the center stays sharp. Distinct
  from (and composable with) translational smear. The static planted roll
  (`offset_rotation_deg`) exists today; the *rate* version is a new smear
  kernel that varies across the field, which also breaks the current
  "convolve the whole scene with one kernel" assumption — the optics stage
  needs a spatially varying kernel path anyway for field-dependent PSFs, so
  these two share machinery.
- **Structured jitter tracks.** Real smear is not a straight line: LORRI's
  deadband produces sawtooth drift-and-kick tracks (thruster firing every
  few seconds), Voyager's MIMC slews are stepper-motor staircases, and
  reaction-wheel spacecraft (future instruments) add narrowband sinusoidal
  jitter. Parameterize the smear kernel as a *track* (list of segments or a
  drift+impulse process) rather than a vector. A double-exposure variant
  (shutter bounce — documented for Voyager below -5 C) is the degenerate
  two-point track: a faint displaced second image of the whole scene.
- **Rolling-readout shear (vidicon).** A Voyager frame is read over 48 x n
  seconds: the top of the frame is a snapshot at t0, the bottom at t0+48n s.
  A target moving during readout is *sheared*, not smeared — its limb is
  displaced progressively down the frame. No CCD has this; it is a
  Voyager-only geometric effect on fast-moving/close targets and interacts
  with the mid-time question (each line has its own effective epoch).
- **Pole / rotation-phase / shape-radius errors.** Planted errors in the
  things SPICE gets slightly wrong: ring-plane pole error (tilts every ring
  ellipse coherently — a Saturn pole error rotates all edges together),
  body rotation-phase error (albedo features and terrain misplaced on the
  disc while the limb stays put — matters once disc correlation is in
  play), and body reference-radius error (the rendered body is a few km
  larger/smaller than the navigator's ellipsoid — a pure radial limb bias,
  the body analog of the ring orbit-error axis).

### 13.2 Scene content and appearance (remainder)

- **Extended and moving interlopers.** Galaxies (small Gaussian blobs with
  non-stellar profiles), an uncataloged moonlet moving through the field,
  or another spacecraft — point-like confounders that are *not* in any
  catalog and move between frames of a sequence. Star matching should
  reject them; blob detection may lock onto them.
- **High-phase crescent extension.** Forward-scattering haze extends a
  crescent's horns past the geometric terminator (Titan's ring of light at
  phase > 150 deg as the limit case). Falls out of the atmosphere layer for
  hazy bodies; for airless bodies a small horn-extension term would be a
  deliberate wrongness the ellipsoid model cannot express.

### 13.3 Backgrounds and diffuse light

- **Astronomical backgrounds.** Zodiacal light and galactic (Milky Way)
  gradients: smooth, very low amplitude, direction-dependent — mostly they
  set the noise floor for faint-star work. Cheap (reuse the stray-light
  ramp with astronomical amplitudes).
- **Diffuse dust environments.** E-ring haze near Enceladus, ring-plane
  scattered light near the ansa, cometary coma (future instruments): a
  broad, low-contrast diffuse component *with structure* (radial/vertical
  falloff), brighter at high phase. This is the fog the scattered-light
  cohort criteria fight; the sim currently cannot produce it at all.
- **Combined off-frame source.** One knob that places a bright body just
  outside the FOV and derives the *consistent set* of consequences: stray
  light gradient + ghost arc + elevated background + (LORRI) smear
  pedestal. Realistic frames have these correlated, not independent.

### 13.4 Metadata lies

A distinct class from image artifacts: the *label* the navigator reads can
be wrong while the image is fine. Planted metadata errors — exposure time
(brightness model off by 2x), filter (wrong PSF/zero point), gain state,
mid-time (Section 13.1), even image dimensions/summation mode — exercise
every place the pipeline trusts a header. Real archives contain all of
these (Voyager's impossible scan rates in the index files; Cassini's
corrupted-header Double Bit Error images). Cheap to implement (the scene
carries `lie_to_navigator: {exposure_sec: ...}` overrides applied to the
ObsSim metadata only), and unlike most of this section it tests *error
handling*, not accuracy.

### 13.5 Calibrated-product processing scars (general class)

The calibrated (I/F) path — whose propagated detector noise is phase B
scope (Section 15.2) — should eventually carry the artifacts *calibration
itself* introduces, since the navigator often consumes calibrated frames:
flat-field residual texture, dark/bias subtraction banding, desmear
residuals through saturated columns (LORRI), despike scars (cosmic rays
removed, leaving interpolated holes — sometimes taking a star with them),
reseau-removal smudges and missing-line interpolation bands (Voyager
GEOMED), and resampling softening from geometric correction. These are the
difference between "raw + noise" and what the pipeline actually sees for
Voyager/Galileo, where archive products are heavily preprocessed. The
Voyager GEOMED subset is in scope (Section 5.3); the triage decision keeps
the general class here, to be revisited after the phase-B calibrated-path
noise lands.

---

## 14. Decisions and open questions

Resolved (operator, 2026-07-15):

1. **Ring projection**: in scope — the ring system renders in ring-plane
   coordinates through an opening angle (Section 4.2), phase F.
2. **CraterMaker** (#78): not required at this time; the statistical limb
   field is the mechanism (Section 4.1).
3. **Voyager reseau marks**: not rendered — the navigator consumes GEOMED
   products with reseaus removed; the sim targets GEOMED-level realism
   (residual geometry, removal scars, resampling texture; Section 5.3).
4. **Geometric distortion**: residual-only. oops applies each instrument's
   distortion model throughout the geometry pipeline already, so the sim
   stage plants only post-correction residuals (Section 4.4); the
   per-instrument residual amplitudes come from #228, reframed as
   formalizing the `experiments/fov_twist` star-field residual analysis
   into the supported suite with documentation and results under `docs/`.
   Disposition: ON at measured residual amplitude in realism defaults,
   swept as a tolerance study, OFF in accuracy sweeps.
5. **Section 13 triage** (operator, 2026-07-15, item by item): promoted
   into scope — SPK error, differential smear, star occultation, saturated
   navigable stars, double/variable stars, planet disc texture,
   satellite/shadow transits, ring spokes, embedded moonlets/propellers,
   adversarial artifact placement. Left unscoped (recorded in Section 13)
   — mid-time error, field-rotation rate, structured jitter tracks,
   rolling-readout shear, pole/rotation-phase/radius errors,
   moving/extended interlopers, crescent horn extension, astronomical and
   dust backgrounds, combined off-frame source, metadata lies, and the
   general calibrated-product scars class (Voyager GEOMED subset remains
   in scope).

6. **Independence redefined (operator, 2026-07-15): information asymmetry,
   not authorship.** The sim acts as nature plus SPICE; the navigator
   receives only the image and the idealized information (Section 1).
   Sealed-room dual implementation, the import-graph-as-independence test,
   and the implementer reading prohibition were removed from the plan;
   enforcement is the Section 3.2 boundary filter and its structural test.
7. **Delivery shape (operator, 2026-07-15): one PR per phase against an
   integration branch** (`rf_sim_realism`), squash-merged to `main` once,
   on the `rf_core_rewrite` pattern; Fable acts as controller and
   Opus/Fable as per-phase implementers (Section 15.11).
8. **Calibrated-path (I/F) detector noise promoted into phase B** —
   required by the Section 7 figures of merit (the CALIB cohorts are I/F)
   and by Voyager GEOMED realism; previously deferred.
9. **Recalibration is in scope as phase J.** The sim-anchored confidence
   coefficients are invalidated by the renderer change and are refit
   before the branch merges.

Open, and owned outside this plan: the WS-3 cohort sizes for
Voyager/Galileo/LORRI (phase H ships its runner regardless, reports against
what exists, and labels the gaps — narrowed 2026-07-15 by the operator
directing a selection of GOSSI/VGISS star frames from the FOV-twist lists
into the library, which gives FOM 2 star cutouts on all four instruments;
limb/ring-class cohort growth for Voyager/Galileo/LORRI remains WS-3
work); the #228 residual-distortion measurements (interim amplitudes ship
meanwhile, Section 15.7).

---

## 15. Implementation contract

Everything an implementer needs that Sections 3-10 state only in prose.
Where a value is marked **interim**, it is a defensible starting point to
be replaced by the phase H realism match (or by #228 for distortion); every
interim value ships with a provenance comment saying exactly that.

### 15.1 Stage activation and scene conversion (governs every phase)

**A stage whose scene block is absent is disabled**, and contributes
nothing to the rendered image.

This is a design property, not a compatibility guarantee. It exists because
Sections 8 and 9 depend on it: a single-variable sweep can only attribute
error to one effect if every other effect is off, and the self-consistency
floor is by definition the render with every mismatch stage disabled. A
stage that contributes something when unconfigured makes both meaningless.
Concretely:

- Whole-scene PSF, smear, distortion, SPK error, atmosphere, ring-system
  mode, star `companion`/`delta_mag`, artifacts, adversarial placement —
  all activate only when their key is present in the scene YAML.
- Per-instrument artifact defaults are *not* implied by naming an
  instrument. A scene opts in with `artifacts: {instrument_defaults: true}`
  (then per-mode overrides apply on top). Naming an instrument selects a
  geometry and a detector, not a set of defects.

**The scene catalog is converted, not preserved.** There is no compatibility
requirement in any direction, per the project's no-compatibility-shims rule:

- Scene authors are exactly three: the 30 scene files under
  `tests/integration/sim_scenes/`, the `_GUI_GALLERY` / `_REPORT_SCENES`
  definitions in `tests/integration/sim_doc_images.py`, and the calibration
  campaign generator `util/calibration/scene_gen.py` (whose dicts bypass
  the file loader today; phase A routes them through the validator). All of
  it is internal, all of it is in this repo, and nothing outside the repo
  authors scenes. There is no population of v1 files to protect.
- Phase A converts every scene to schema v2 (ring scenes keep the `rings:`
  key until phase F converts them to `ring_system`) and updates every
  program and YAML that reads them. If a v1 file turns up later, it is
  converted, not accommodated.
- **Rendered output may change**, and small differences are expected
  wherever a rewritten renderer discretizes differently or a stage reseeds.
  The bar is that each scene still renders what it asks for: the same
  ingredients with the same geometry and the same planted truth, and the
  navigation result the scene exists to assert still holds. It is not
  pixel equality, and no phase is blocked on reproducing a prior render.
- Baselines, tier files, and the doc gallery are regenerated when the render
  changes, and the diff is reviewed rather than rubber-stamped — a converted
  scene that recovers its planted offset but looks wrong is a conversion
  bug, and the review is the only thing that catches it.
- Stage names seed their own RNG streams (15.2), so renaming a stage changes
  that scene's noise realization. That is allowed; it regenerates baselines.

### 15.2 Stage interface and unit chain

```python
@dataclass
class SimFrame:
    signal: NDArrayFloatType    # (V*os, U*os) float64, intensive scene
                                # signal units (I/F-like): bodies, rings,
                                # diffuse backgrounds
    point_e: NDArrayFloatType   # (V*os, U*os) float64, ELECTRONS:
                                # point sources (stars, moonlets), kept
                                # separate because one array cannot carry
                                # two unit systems through the detector
                                # stage's signal->electron conversion
    oversample: int             # os >= 1; detector grid is (V, U)
    truth: dict[str, Any]       # feature truth accumulated by radiance stages
class Stage(Protocol):
    def __call__(self, frame: SimFrame, *, params: Mapping[str, Any],
                 rng: np.random.Generator) -> None: ...
```

The optics stage convolves and warps `signal` and `point_e` identically
(they share every geometric transform); the detector stage converts
`signal` to electrons and then adds `point_e` into the electron image
before Poisson — so stars never pass through the intensive conversion.

- Stages mutate `frame` in place, in the fixed order of Section 3.3; each
  stage receives its own `np.random.Generator` seeded via
  `derive_effect_seed(random_seed, '<stage-name>')`, so each stage's noise
  is independent of which other stages are enabled. (The current code seeds
  `np.random.RandomState`; the switch to `Generator` reseeds every noise
  realization, which the phase-A conversion absorbs.) A stage name is
  therefore part of its scenes' noise realization: renaming one reseeds it
  and regenerates those baselines (15.1), which is allowed but should not
  be done idly.
- **Unit chain.** Scene signal units are normalized I/F-like values in
  [0, ~1] (unchanged from today; intensive). The detector stage converts to
  electrons through the exposure, so extended sources and stars share one
  radiometric footing:
  `electrons = signal * signal_full_scale_frac * full_well_e *
  (exposure_sec / exposure_ref_sec)`, where:
  - `signal_full_scale_frac` is the existing sim config key
    (`config_440_sim.yaml`, default 0.5): the well fraction a signal of
    1.0 fills at the reference exposure. It keeps its meaning; only the
    well it references changes from DN to electrons.
  - `exposure_ref_sec` is a new per-instrument catalog value — interim
    values, provenance-tagged as such: coiss_nac 1.0, coiss_wac 1.0,
    gossi 0.2, nhlorri 0.1 s (each a typical science exposure, so a
    converted scene at its instrument's typical exposure keeps today's
    brightness scale); vgiss n/a (DN path below).
  - `full_well_e` is a new per-instrument value (Section 5: coiss_nac
    110e3, coiss_wac 95e3, gossi 108e3, nhlorri 86e3 [ADC-limited:
    4095 DN x 21 e-/DN, antiblooming CCD], vgiss n/a).
  Chain order: convert `signal` to electrons, add `point_e` (stars —
  already electrons, never converted), then Poisson, then electron-domain
  full-well bleed (the `bloom` mode — charge spills during integration,
  before the read amplifier; Cassini saturates at ~3600 DN at gain 2,
  *below* the ADC clip, per 5.2), then read noise (added at readout),
  then `DN = electrons / gain_e_per_dn + bias_dn`, quantized, clipped at
  `saturation_dn`. `gain_e_per_dn` comes from the instrument's gain-state
  table in `artifacts_catalog.py` (Section 5 values), selected by the
  scene key `detector.gain_state` (defaults per Section 5). The
  *image-side* well in DN is derived (`full_well_e / gain_e_per_dn`); the
  navigator-side config key `full_well_dn` is NOT redefined — it feeds the
  real pipeline's star detection and saturation masks
  (`nav_model/stars/detection.py`, `support/image_quality.py`) and remains
  the published ADC-referenced value the navigator believes.
- **Calibrated (I/F) path.** `data_units: calibrated_if` scenes render
  through the same DN chain and then apply the calibration transform the
  real pipeline applies:
  `IF = (DN - bias_dn - dark_dn) / (calibration_scale_dn_per_s_per_if *
  exposure_sec)` — bias and dark subtracted *before* the divide, exactly
  as real calibration does, or every calibrated frame carries a spurious
  pedestal (~545 DN for LORRI) that grows as 1/exposure and poisons the
  Section 7 sky statistics. Calibrated products then carry propagated
  shot/read noise and quantization texture in I/F units. This replaces
  the noise-free calibrated branch (`render.py` gates the detector stack
  on `raw_dn` today) and is phase B scope. Voyager GEOMED scenes use this
  path with the vidicon noise model below.
- **Point-source deposition and the downsample.** Stars are deposited
  into `point_e` in stage 1 as sub-pixel-positioned point masses of
  weight `total_electrons * os^2` — **not** pre-spread over any PSF; the
  whole-scene optics stage is the ONLY PSF application, or stars would be
  convolved twice (~sqrt(2) too wide) and their shape would desync from
  the limb and ring-edge profiles that FOMs 2 and 3 cross-check. Floor
  and converted legacy scenes therefore carry an explicit `optics.psf`
  block (Section 8) so stars keep a finite PSF. The box downsample
  (pipeline step 3) is a MEAN over the os^2 subsamples: intensive
  `signal` passes through unchanged, and the point-mass weighting makes
  the detector-grid electron sum exactly `total_electrons` after the
  mean.
- **Voyager exception.** The vidicon is not photon-noise dominated; its
  detector stage skips the electron conversion and applies the Section 5.3
  noise model directly in DN (line-correlated read noise + coherent
  component). Config key `detector_model: ccd | vidicon` selects the path.
  Vidicon star flux is specified directly in DN:
  `total_dn = star_flux_dn_per_s_vmag0 * 10**(-0.4 * vmag) * exposure_sec`,
  interim vgiss zero point 3e3 DN/s (provenance-tagged; sized so the 5.3
  limiting magnitudes land at the matched-filter boundary), deposited as a
  point mass like the CCD path (the optics PSF shapes it).
- **Star flux (CCD).** `total_electrons = star_flux_e_per_s_vmag0 *
  10**(-0.4 * vmag) * exposure_sec`, deposited as a point mass into
  `point_e` (flux normalization, not peak — the optics stage's PSF then
  dictates the shape and the peak). Interim zero points, derived
  from the Section 5 limiting magnitudes at SNR ~5 (provenance-tagged):
  coiss_nac 1.0e7, coiss_wac 2.6e6, nhlorri 7e7, gossi 4e6 e-/s. This
  replaces the peak-based `2.512^-(vmag-4)` path, which is removed; star
  scenes are converted and re-rendered, and their brightness changes (a
  flux-normalized star of a given magnitude does not have the same peak DN
  as a peak-normalized one). The scenes assert recovered offsets and
  detection outcomes, not DN values, so the conversion is checked against
  those and against the reviewed diff.

### 15.3 PSF kernel specification

Per-instrument kernel, all radii in detector pixels, evaluated on the
oversampled grid and normalized to unit sum over the truncation window:

```text
K(r) = (1 - w) * G_norm(r; sigma_v, sigma_u) + w * M_norm(r; r0, n)
```

G_norm is an elliptical Gaussian (sigma_v = sigma_u unless stated) and
M_norm the Moffat wing `(1 + (r/r0)^2)^(-n/2)`, **each separately
normalized to unit sum over its truncation window**, so `w` is exactly the
wing energy fraction (an unnormalized Moffat term with n=3, r0=2 integrates
to ~8*pi, which would make a nominal w mean ~25x more wing energy than it
says). Truncation radius 16 px (Cassini: 32 px — documented long wings);
the measured Cassini wings extend hundreds of px (5.2), and the truncated,
renormalized kernel conserves flux at the cost of the far halo — the halo
beyond the window is `optics` stray-light scope, not kernel scope. The
truncation radius is tunable in phase H alongside w/r0/n. Interim
parameters:

| Instrument | sigma (px, from FWHM/2.355) | w (wing energy frac) | r0 | n |
|---|---|---|---|---|
| coiss_nac | 0.55 | 2.5e-2 | 2.0 | 3 |
| coiss_wac | 0.64 | 2.5e-2 | 2.0 | 3 |
| vgiss | 0.85 | 1.2e-2 | 2.0 | 3 |
| gossi | 0.80 | 1.2e-2 | 2.0 | 3 |
| nhlorri | sigma_u 0.87, sigma_v 1.13 | 1.2e-2 | 2.0 | 3 |

(gossi's 0.80 is a directly published sigma, not FWHM/2.355; vgiss's 0.85
is an interim estimate — 5.3 publishes no Voyager FWHM, and GEOMED
resampling broadens whatever the vidicon delivered.) Cores come from the
Section 5 measured FWHMs; wing parameters are **interim** (re-expressed as
energy fractions from an unnormalized draft so the delivered kernels are
unchanged). The truncated kernel's in-window core-to-wing dynamic range is
~2e6 for coiss_nac; the 1e7-1e8 of 5.2 lives in the far halo beyond the
window, which is stray-light scope by the rule above. Wing parameters are
the first thing phase H tunes. `star_psf_sigma` in existing
configs remains the navigator's model; these kernels are the image side's;
floor scenes use neither and set the image sigma equal to the navigator's
configured value (Section 8).

### 15.4 Ring photometry and projection (exact forms)

One normative equation set; the shared projection helper (3.2) implements
it once and both sides call it.

- Geometry block: `{center_v, center_u, opening_deg_obs, opening_deg_sun,
  node_deg}`. B_obs = opening_deg_obs, B_sun = opening_deg_sun, both in
  (-90, 90], positive north; mu = |sin B_obs|, mu0 = |sin B_sun|; lit side
  iff sign(B_obs) == sign(B_sun) (B of exactly 0 renders nothing).
- **Longitude convention.** `lam` is ring-plane longitude measured from the
  ascending node, *in the ring plane*, increasing counterclockwise viewed
  from the north. Every orbital angle — `peri`, `delta_long_peri_deg`,
  edge-wave `lam0`, azimuthal-structure longitudes — lives in this frame.
  `node_deg` is the sky position angle of the ascending node, measured
  counterclockwise from +u toward -v; it enters only the final sky
  rotation, never the orbit model. (Conflating the two frames is the
  single likeliest implementation error here; this paragraph is the
  arbiter.)
- Projection of ring-plane point (r, lam):
  `x = r*cos(lam)`, `y = r*sin(lam)` (in-plane, node-aligned axes), then
  `du = x*cos(node) - y*sin(B_obs)*sin(node)`,
  `dv = -(x*sin(node) + y*sin(B_obs)*cos(node))`,
  pixel = (center_v + dv, center_u + du). |B_obs| = 90 reduces to today's
  sky-plane circles (regression identity).
- **Line-of-sight depth.** A point's depth relative to the ring center is
  `dlos = -y * cos(B_obs)`, positive toward the observer: for B_obs > 0
  the near arm is the y < 0 half, and the ring's *nearest point* sits at
  lam = 270 deg when node = 0 (a unit test pins exactly this
  configuration at B_obs = 30; note the ansae — the projected major-axis
  tips at lam = 0/180 — have zero depth by construction). Per-pixel
  compositing between the ring and bodies orders by observer distance
  `range_km - dlos_km` where `range_km` is set (15.6) — dlos positive
  toward the observer means *subtracting* it gives the nearer object the
  smaller distance; the far arm composites behind the planet disc, the
  near arm in front.
- Single-scattering photometry with one-term Henyey-Greenstein
  `P(g, alpha)` (default g = -0.3 for main-ring backscatter; dusty features
  set g ~ +0.6):
  - lit: `I = A/4 * P * mu0/(mu0 + mu) * (1 - exp(-tau*(1/mu0 + 1/mu)))`
  - unlit: `I = A/4 * P * mu0/(mu0 - mu) * (exp(-tau/mu0) - exp(-tau/mu))`,
    with the limit `(A/4) * P * (tau/mu) * exp(-tau/mu)` when
    |mu0 - mu| < 1e-6.
  - A (single-scattering albedo x normalization) defaults to 0.5.
- Compositing, far-to-near: `img = I_ring + exp(-tau/mu) * img_behind`.
- Edge waves: radial perturbation
  `dr(lam) = amp * exp(-dlam/damp) * sin(2*pi*dlam*a/wavelength)` with
  `dlam = (lam - lam0) mod 2*pi` in [0, 2*pi), so the wave exists only
  downstream of lam0 (the exponential grows without bound if evaluated
  for lam < lam0, so the clamp is load-bearing, not cosmetic). The
  modular form is the clamp's periodic implementation — rings are
  periodic and the wave physically wraps the full turn — and leaves an
  upstream residual of `amp * exp(-2*pi/damp)` just before lam0; the
  validator caps `damp <= 2` radians, bounding that wrap-seam residual
  at `exp(-pi)` (~4.3% of amp). `damp` is in radians and `a` the
  feature's semimajor axis (so the sine argument is arc length over
  wavelength, dimensionless);
  m-modes: `r(lam) = a - amp_m*cos(m*(lam - peri))` with `amp_m` = a*e in
  the same radial units as `a` and `peri` in the longitude frame above.

### 15.5 Body relief field: limb and terminator (exact form)

**The relief is a 2-D field on the body surface, not a 1-D field on the
limb**, so the limb perturbation and the terminator shadowing are slices of
one consistent surface:

- `h(lat, lon)`: fractional relief (height / local radius), a periodic 2-D
  Gaussian random field synthesized by 2-D FFT on a (lat, lon) grid:
  independent complex Gaussian spectral coefficients with variance
  `S(k) proportional to exp(-(|k| * corr_rad / 2)^2)`, where |k| is total
  angular wavenumber and `corr_rad` = `limb_relief_corr_deg` in radians of
  surface arc; band-limit at `kmax = ceil(8 / corr_rad)` (S there is ~1e-7
  of peak). The randomness is in the coefficients — fresh Gaussian draws
  per seed, so a new seed is a new terrain; `S(k)` is their variance (the
  power spectrum), whose Gaussian shape in k yields a Gaussian
  autocorrelation of width ~corr_rad on the surface. Seeded from the
  body's crater seed derivation.
- **Low-order modes are zeroed** (total wavenumber < 3): the degree-1
  content of a radius perturbation is, to first order, a *translation* of
  the body — at the default 15-deg correlation length it would carry ~38%
  of the field RMS as a planted, untruthed center offset that no limb fit
  could distinguish from the pointing error under recovery — and degree-2
  content aliases ellipsoid shape error, which is its own sweep axis. Both
  belong to other knobs; the relief field starts at degree 3.
- **Normalization.** After zeroing, the field is rescaled so the *limb
  slice's* standard deviation equals `limb_relief_rms` per-realization
  (this is what phase E's "within 10% over 100 seeds" acceptance
  measures).
- **Limb application.** `delta(theta)` = h sampled along the sub-observer
  horizon circle, with theta the image azimuth about the body center. The
  renderer's normalized ellipse radial function e(p) — homogeneous of
  degree 1 in the pixel offset p from body center, equal to 1 exactly at
  the unperturbed limb — becomes `e_adj(p) = e(p) / (1 + delta(theta))` at
  the oversampled grid, placing the perturbed limb at radius
  `r_ellipse * (1 + delta)`. Shading normals keep the unperturbed e:
  relief moves the silhouette and the terminator, not the low-frequency
  disc shading.
- **Terminator march.** All march arithmetic uses *absolute* heights
  `H = h * R` in px (R the local body radius in px) — the fractional
  field h (~0.01) and surface distances (px) must never be compared
  directly; the factor of R is load-bearing. For disc points within the
  march cap of the terminator, march from the point toward the sun in
  *surface* distance (step = 1 oversampled px of surface arc, corrected
  for foreshortening near the limb — an image px near the limb spans a
  large surface step), sampling H along the ray. The point is shadowed
  iff any upstream sample at surface distance d satisfies
  `H_up - H_pt > d / tan(i_pt)`, where i_pt is the local incidence (sun
  elevation = 90 deg - i). March cap:
  `d_max = min((H_max - H_min) * tan(i_pt), sqrt(2 * R * H_max))` — the
  first term is the longest shadow the terrain can cast, the second the
  horizon limit that caps the tangent's divergence at the terminator
  itself — so cost is bounded and the geometry stays physical.
- **Synthesis caveat.** A plain (lat, lon) FFT grid holds `corr_rad` in
  latitude but over-corrugates high latitudes in longitude (a longitude
  cycle spans `~cos(lat)` less surface arc), and the sub-observer limb
  circle passes near the poles for equatorial views. At the default
  15-deg correlation length the affected caps are small; if the phase-E
  acceptance statistics miss because of it, substitute an equal-area or
  spherical-harmonic synthesis — the contract fixes the spectrum and the
  limb-slice statistics, not the synthesis mesh.
- Defaults: `limb_relief_rms 0.0` (off); guidance values in scene
  comments — icy midsize moons 0.001-0.003, small cratered bodies
  0.01-0.03; `limb_relief_corr_deg 15` (degrees of surface arc; image
  azimuth and surface azimuth differ under foreshortening, and the field
  lives on the surface).

### 15.6 Schema v2 (complete key inventory)

`CURRENT_SCHEMA_VERSION` (`sim/scene.py`) becomes 2. The loader keeps its
current strict-equality check and so accepts only 2 — there is no v1 branch
to write, because today's loader already rejects any version that is not the
current one, and phase A converts the catalog in the same change. Any v1
file encountered later is converted with a throwaway script, not read.
`save_sim_scene` always writes 2. New keys (types; defaults in parentheses):

Every key is tagged **[I]** (idealized: exposed through `obs.nav_params`)
or **[T]** (truth: renderer-only, stripped by the boundary filter, 3.2).

- top level: `oversample` int [T] (4 whenever a PSF is active — from an
  `optics` block or `instrument_defaults` — else 1); `optics` map [T];
  `artifacts` map [T]; `detector` map [T] `{gain_state int,
  detector_model str ('ccd' | 'vidicon'), exposure_ref_sec float}`
  (per-instrument defaults from `artifacts_catalog.py`; scene keys
  override); `spk_error` map [T] `{dv_px float, du_px float,
  reference_range_km float}` — an object at physical range R km displaces
  by `(dv, du) * reference_range_km / R`; `sky_counts` map [T]
  `{a float (-3.1), b float (0.34), density_factor float (1)}`;
  `expected` map `{status str ('success' | 'failed' | 'conflicted'),
  status_reason str, confidence_tier str | null}` (scene-level expected
  outcome; consumed by the 15.8 assertion machinery, not by either
  renderer).
- `optics`: `psf` map `{sigma_v/sigma_u/w/r0/n floats (15.3
  per-instrument)}`; `smear` LIST of maps `{dv_px, du_px float,
  object_class str ('all' | 'stars' | 'bodies' | 'rings')}` (one entry =
  whole-scene smear; several = differential smear, and bodies and rings
  are separately addressable); `distortion` map `{k1, k2 float, center_v,
  center_u float, nonradial_rms_px float (0)}` — warp
  `p -> p * (1 + k1*rho^2 + k2*rho^4)` about the center, with
  rho = |p - center| / rho_ref and rho_ref = half the image diagonal; the
  15.7 per-instrument amplitudes are RMS displacement over the frame,
  mapped to k1 with k2 = 0; `nonradial_rms_px` adds the Voyager per-image
  wander as a seeded low-order 2-D field; `ghosts` list of maps
  `{dv_px, du_px, amplitude, defocus_sigma}`.
- `artifacts`: `instrument_defaults` bool (false); per-mode maps keyed by
  the 5.1 mode key registry, each `{incidence float, ...mode params}`;
  `adversarial` bool (false).
- body: geometry/photometry the navigator may know — center, axes, pose,
  `range_km` float [I] (physical range; REQUIRED on every object in a
  scene that carries `spk_error` or overlapping bodies, and the
  compositing depth when present; replaces the v1 per-body `range` z-order
  hint) — plus truth keys: `limb_relief_rms` float (0) [T],
  `limb_relief_corr_deg` float (15) [T], `photometric_law` str ('lambert'
  | 'lommel_seeliger' | 'minnaert' | 'lunar_lambert') [T], `minnaert_k`
  float (0.5) [T], `opposition_surge` map [T], `albedo_texture` map
  `{rms float, corr_px float, spots list}` [T], `atmosphere` map
  `{scale_height_px, tau_ref, ref_altitude_px, g, detached_px}` [T]
  (tau_ref is the tangent optical depth at ref_altitude_px),
  `disc_texture` map (bands/storms) [T], `transits` list of maps (moon +
  shadow discs) [T]; mesh keys unchanged [I] plus `shading` str ('flat' |
  'gouraud') [T] and per-frame `pose_scatter` [T].
- star: catalog position and `vmag` [I], `navigable` bool (true; the
  filter drops false entries from `nav_params` entirely),
  `catalog_error_v/u` float (0) [T], `companion` map `{sep_px float,
  delta_mag float, angle_deg float}` [T], `delta_mag` float (0) [T];
  top-level `star_catalog_scatter_px` float (0) [T].
- ring feature (inside `ring_system.features`): `kind` str ('ringlet' |
  'gap' | 'edge' | 'ramp' | 'wave') [I], `tau` float [I], `navigable`
  bool (false; false entries dropped from `nav_params`), `orbit` map [I]
  (mode-1 params + `modes` list `{m, amp, peri}` + `edge_wave` map),
  `orbit_error` map `{delta_a_px, delta_ae_px, delta_long_peri_deg}` [T],
  `declared_orbit_sigma` map [I] (the uncertainty the navigator is
  entitled to know; the drawn `orbit_error` values are truth),
  `phase_g` float [T], `albedo` float (0.5) [T];
  `ring_system.geometry` per 15.4 [I]; `ring_system.phase_deg` float [I]
  (the phase angle the ring photometry evaluates at) and
  `ring_system.km_per_pixel` float [I] (physical pixel scale for the
  per-pixel depth conversion) — both SPICE knowledge, like the geometry
  block; `ring_system.azimuthal` map (modulation/spokes/shadow wedge)
  [T]; `ring_system.moonlets` list [T].

**v1 key dispositions** (all 25 v1 top-level keys of `_ALLOWED_KEYS` —
this is what makes the inventory complete rather than additive):

| v1 key | v2 disposition |
|---|---|
| `schema_version`, `scene_name`, `instrument`, `instrument_config`, `size_v`, `size_u`, `random_seed`, `exposure_sec`, `offset_v`, `offset_u`, `offset_rotation_deg`, `midtime_utc`, `closest_planet`, `time`, `ring_epoch`, `fit_camera_rotation`, `noise` | retained unchanged |
| `bodies`, `stars` | retained; per-object keys gain the additions and [I]/[T] tags above (per-body `range` becomes `range_km`) |
| `rings` | valid on the integration branch until phase F, then removed (converted to `ring_system`) |
| `background_stars_num`, `background_stars_distribution_exponent` | removed in phase D, replaced by `sky_counts` |
| `background_stars_psf_sigma` | removed in phase D (background stars go through the scene PSF like everything else) |
| `stray_light` | retained; structured modes move under `optics` in phase B, the smooth-ramp params unchanged |
| `shade_solid_rings` | removed in phase F (meaningless under tau compositing) |

Phase A also fixes a latent v1 defect found in review: `ObsSim` reads
`sim_params['epoch']` — a key the schema does not allow — so
`obs.sim_epoch` is always 0.0 for every validated scene while the renderer
honors `ring_epoch`; any precessing ring scene today carries an
undocumented planted epoch error. The filtered view exposes `ring_epoch`
[I] and the navigator-side ring model reads it.

The machine-readable truth-key set lives as a `TRUTH_KEYS` frozenset (with
per-object-block sub-maps) beside `_ALLOWED_KEYS` in `sim/scene.py`: the
`ObsSim` boundary filter strips exactly this set, the 3.2 structural test
iterates exactly this set, and a key added to the inventory without a
`TRUTH_KEYS` / idealized classification fails a completeness assertion —
that is what "extend the boundary test in the same PR" means concretely.

The validator enforces this inventory exactly (unknown keys still fail).
Every key above is authorable from the scene editor (Section 9.4); the
editor's round-trip test asserts that loading and re-saving a scene using
the full inventory preserves it.

### 15.7 Interim artifact incidences and residual distortion

`instrument_defaults: true` switches on the instrument's *physical signal
chain*: detector noise, bias, quantization, the instrument's 15.3 PSF
kernel (an explicit `optics.psf` block overrides it; `oversample` defaults
to 4 whenever a PSF is active from either source), and, for LORRI,
`frame_transfer_smear`. This is consistent with 15.1 because
`instrument_defaults` is itself a present key — a scene with neither an
`artifacts` nor an `optics` block renders with every stage disabled, and
that is the floor configuration (Section 8).

All *loss-mode* incidences default to 0 even under `instrument_defaults`.
Loss modes are planted by `artifact_sweep` scenes explicitly, and phase H
replaces the zeros with cohort-measured rates. This keeps "realism
defaults" honest until they are measured, per the acceptance criteria.
Residual-distortion interim amplitudes (ON under `instrument_defaults`):
coiss 0.1 px, vgiss 1.0 px, gossi 0.05 px, nhlorri 0.05 px —
provenance-tagged pending #228, which the FOV-twist star-frame lists make
measurable for all four instruments (Sections 4.4, 7).

### 15.8 Navigator-side plumbing and truth records

- `ObsSim` holds the full `sim_params` for the renderer and builds the
  filtered idealized view `obs.nav_params` (3.2): truth keys stripped,
  non-navigable objects dropped, catalog values only. Navigator-side
  models consume ONLY `nav_params`. `NavModelStarsSimulated` reads catalog
  star entries from it — replacing today's `obs.sim_star_list`, which is
  the renderer's output metadata and the flagship boundary violation
  (3.2). `NavModelRingsSimulated` reads navigable features' catalog
  orbits, `declared_orbit_sigma`, `ring_epoch`, and the shared `geometry`
  block. `NavModelBodySimulated` reads the idealized ellipsoid/mesh
  geometry and `range_km`. The drawn `catalog_error_*` / `orbit_error` /
  `spk_error` values are absent from `nav_params` by construction;
  declared uncertainties (the error *bars* the navigator is entitled to
  know) are idealized keys.
- Scene truth (metadata additions): planted offset (existing), per-body
  `visible_fraction` and `occluded_limb_arc_deg` (mutual events), per-star
  rendered-vs-catalog delta, per-feature applied orbit error, and the
  scene-level `expected` block. `expected` is asserted by NEW machinery in
  the sim integration suite (phase D), modeled on
  `tests/integration/sidecar.py`'s field taxonomy (`status`,
  `status_reason`, `confidence_tier`, and its cross-field rules) but
  separate from it — sim scenes are not image-library sidecars, and no
  sim test imports that module today.

### 15.9 Performance and determinism budget

- A 512x512 scene with `optics.psf` + full detector stack at oversample 4
  renders in < 2 s single-core; a 1024x1024 Cassini-class scene < 8 s.
  "Single-core" is asserted by the perf harness itself
  (`OMP_NUM_THREADS=1` plus an affinity mask of one core) — an unpinned
  numpy FFT silently multithreads and fakes the budget. The PSF kernel
  gets its own small cache keyed by kernel params (new in phase B; the
  existing render caches are `lru_cache(maxsize=1)` except the two
  shape caches at 30, and stay that way).
- The `_render_combined_model_cached` params-JSON caching contract is
  preserved: all new blocks are JSON-serializable scalars/lists/maps; no
  wall-clock, no global `np.random` state, no non-deterministic iteration
  order anywhere in `sim/forward/`. Two renders of the same scene file are
  bit-identical on one machine.
- The WS-5-style campaign throughput bound: a 4200-scene calibration run
  with default stages stays within 2x the recorded baseline. The
  "~8 minutes (14 workers, thread-pinned)" figure is an unrecorded
  operator observation: phase A measures and records the baseline (command
  line + elapsed time, in `util/calibration/README.md`) BEFORE any phase
  changes the renderer, so the 2x budget has a reproducible reference.
  Campaign runs source `setup.sh` first — worker affinity on this machine
  is load-bearing, not cosmetic.

### 15.10 Per-phase acceptance

- **A**: catalog converted to v2 and re-rendered, every scene's assertions
  still hold; the contact-sheet generator exists
  (`tests/integration/render_contact_sheet.py` — one
  before/after/|difference| panel per scene, "before" sourced from the
  prior commit's committed sheet PNGs) and its phase-A sheet is reviewed
  panel-by-panel; information-boundary test green (every `TRUTH_KEYS`
  entry unreachable through `nav_params`); star-list replumb landed
  (`sim_star_list` gone from the navigator path); `sim_body*`/`sim_ring`
  re-homed and `forward/` ports render the catalog at present fidelity;
  `ring_epoch` fix landed; `scene_gen.py` validates; timing baseline
  recorded (15.9); stage registry + SimFrame landed; `cli/sim_editor/`
  split done with the round-trip test; dev-guide architecture chapter
  rewritten.
- **B**: PSF/smear/SPK/detector stages meet 15.2-15.3, including the
  calibrated (I/F) path carrying propagated noise; stage-disabled test
  (15.1) covers every new block; floor-equality rule (Section 8)
  implemented and tested on one instrument with a non-matching configured
  PSF (coiss_wac); performance budget met under the pinned harness;
  Optics and Artifacts tabs authorable.
- **C**: every mode in the 5.1 key registry has a geometry unit test (a
  missing line is a full line, blocks align to their grid, adversarial
  mode hits the declared features); `artifact_sweep` runs in CI.
- **D**: 1/2/3-star lock scenes navigate; confounder-density breakdown
  curve produced; `expected` assertion machinery lands and
  `expected_fail` scenes assert failure/low confidence through it;
  `scene_gen.py` star families converted.
- **E**: limb-relief field matches 15.5 statistics (limb-slice RMS within
  10% of commanded over 100 seeds; degree < 3 power measured at zero);
  terminator march respects its cap; mutual-event truth recorded;
  disc-texture/transit scenes render.
- **F**: projection reduces to circles at |B| = 90 (regression identity);
  photometry matches the 15.4 closed forms on analytic cases; the
  depth-convention unit test passes (ring's nearest point at lam = 270
  for B_obs = 30, node = 0, and it composites in front of a body at the
  ring center's range); star behind the B ring (tau = 2) at B_obs = 30
  attenuated > 98% (exp(-4)); #84 compositing test (gap reveals
  background); `rings:` and `shade_solid_rings` removed; the two `rings:`
  scenes and `scene_gen.py`'s ring family converted.
- **G**: Titan-class limb profile's tangent optical depth falls
  exponentially with the commanded scale height (asserted on the
  optically thin portion — above tau ~ 1 the intensity profile saturates
  by design); terminator brightens past 90 deg incidence.
- **H**: per-instrument match report exists with every figure of merit
  from Section 7 its cohort can support (unsupportable ones labeled, per
  Section 7); interim values replaced or explicitly retained with
  reasons; divergence statistic is the **Wasserstein-1 distance on
  quantile-clipped data (1st-99th percentile), normalized by the real
  distribution's IQR** — W1 is a transport metric in the variable's own
  units, which is the actual reason to use it, but it is *not*
  outlier-robust (it grows linearly with displaced-tail distance), so the
  clip keeps FOM 1 from silently measuring FOM 6's artifact tails;
  reported per figure of merit, no pass/fail threshold.
- **I**: sweep tables regenerate; report restructured per 9.1; user guide
  revised per 9.3; both doc galleries re-rendered with panels for the new
  ingredients; `TERMINATOR_ARC` emitted.
- **J**: calibration campaign re-collected on the new renderer; refit in
  the README's order — alphas (`fit.py`), hand-written YAML, re-collect,
  floors (`fit_floors.py`), then tiers + gate (`fit_gates.py`) — run to
  convergence; 2-sigma fused coverage re-verified (~0.865); provenance
  headers in `config_510` / `config_540` updated; `BodyTerminatorNav`
  refit included (#223). No sim-anchored coefficient on the branch
  predates the renderer it ships with.
- **Every phase**: its schema keys are reachable from the editor, and its
  dev-guide section describes what it built. A phase that lands a renderer
  without its controls and its prose is not done.

### 15.11 Execution: controller and implementers

The series is executed by AI agents on the `rf_core_rewrite` pattern: a
frontier-class **controller** session (Fable) that owns the integration
branch, and one **implementer** session per phase (Opus-class or
Fable-class per the tiering below) that delivers that phase's PR. The
operator reviews whatever they choose on the integration branch; the plan
requires operator input at exactly two points — supplying/approving the
WS-3 cohort for phase H, and final approval to merge the branch to
`main`.

Controller responsibilities:

- Dispatch each phase to an implementer with this plan, the phase's 15.10
  acceptance list, and the current branch state. Phases C-G may run as
  parallel implementers on separate PRs where file overlap allows; the
  controller sequences merges.
- Review every phase PR against its acceptance list before merging: run
  the full suite, check that any new truth key extends the boundary test
  in the same PR, check that any render change regenerated its baselines
  and galleries in the same PR, and read the render-diff contact sheet
  panel by panel.
- Maintain the render-diff **contact sheet** as a standing artifact: a
  before/after/difference panel per scene, regenerated by any PR that
  changes renders, committed on the branch (`tests/integration/`
  `render_diffs/`). "The scene still renders what it asks for — same
  ingredients, same geometry, same planted truth; differences confined to
  discretization and reseeding" is the review criterion, and 15.1 makes
  this review the only thing that catches a conversion bug that still
  recovers its planted offset.
- Independence is informational (3.2), so there is no reading prohibition
  and the controller may hold both sides in context. The discipline the
  controller enforces is the boundary: no PR merges that reads truth keys
  on the navigator side or ships a truth key untagged in 15.6.

Implementer tiering (standard-class = Opus; frontier-class = Fable):

- **Standard-class OK: phases B, C, D, G, I.** Well-specified
  transformations with mechanical verification: the stage interface, unit
  chain, PSF kernel form, schema inventory, artifact geometries, and
  per-phase acceptance criteria are pinned, and the stage-disabled,
  boundary, and geometry unit tests catch implementation mistakes. Phase C
  in particular is a long list of small, independently testable features.
- **Frontier-class: phases A, E, F, H, J.**
  - **A** (skeleton + boundary + conversion): an architecture task whose
    failure mode is not a red test but a structure that taxes every later
    phase, plus the conversion judgment — deciding whether a changed
    render is a correct rewrite or a conversion bug is exactly what no
    test in this plan can do. The `cli/sim_editor/` split is mechanical
    and may be delegated to a standard-class implementer as a sub-PR,
    provided it lands before the phase-B controls.
  - **E** (bodies): the 2-D relief field, its zeroed low-order modes, and
    the foreshortening-corrected shadow march are the plan's most
    geometry-laden spec; errors produce plausible-looking wrong terrain.
  - **F** (rings): the longitude/depth conventions, the mu0 -> mu limit,
    and per-pixel transmission compositing interact, and a frame
    conflation produces images that look right at B = 90 and are wrong
    everywhere else.
  - **H** (realism match): judgment work by construction — deciding
    whether a mismatch is a wrong parameter, a missing effect, or a
    cohort artifact cannot be pinned by any contract.
  - **J** (recalibration): the fit / curated-YAML-edit / re-collect loop
    plus judgment on every coefficient proposal against its provenance
    history.
- **Every standard-class phase PR gets the controller's frontier-class
  review before merge**, focused on what tests cannot check: whether a
  new key crossed the boundary untagged, whether a render change is
  degradation dressed as conversion, whether an acceptance criterion was
  satisfied in letter but not substance.
