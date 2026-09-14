# Simulator improvement plan — backend, GUI, and calibration test layer

This document is a self-contained plan for upgrading the rms-nav
simulator system so it can serve as a useful adjunct to the
operator-curated real-image library for navigation calibration and
verification work.  It is written so that an AI or developer with no
prior context can read this single file and understand both what to
build and why.

> **Issue triage (2026-06-19).** A pass over this plan found that every
> unimplemented phase is already tracked: **B3/G3** (flux-correct star smear) ->
> #151, **B8** (diffraction spikes) -> #152, and **T5/T6/T7** (alpha-bootstrap,
> real-vs-sim diagnostics, calibration validation) -> #153 -- all Priority
> Defer. No new issues were needed. The implemented phases (B0-B7 except B3, all
> GUI phases, T1-T4) live on branch `core_rewrite_phase10_sim`. See
> `plans/ROADMAP.md` for how these fit the overall plan.

---

## Reading-order guide

If you are picking this up cold, read sections in this order:

1. **Context and problem statement** — what motivates this plan.
2. **System inventory** — what exists today.
3. **Cardinal principles** — non-negotiable design constraints.
4. **Backend improvement plan** — what to change in
   `src/nav/sim/`, in dependency order.
5. **GUI improvement plan** — how the simulator GUI in
   `src/main/nav_create_simulated_image.py` follows the backend.
6. **Calibration test strategy** — how the upgraded sim feeds into
   confidence-formula calibration and regression coverage.
7. **Phase ordering** — which work unblocks which.
8. **Out of scope / explicit non-goals** — things that look good but
   aren't worth doing for the calibration mission.
9. **Acceptance criteria** — how to know each phase is done.

The plan is intentionally not prescriptive about the exact PR
boundaries; it gives a phase ordering with clean handoffs and lets
the implementer batch them.

---

## 0. Current status and next steps (session handoff)

This section is the live status board; the phase sections below are the durable
plan. A new contributor should read this first, then the phase it points to.

### 0.1 Plan-phase status

- **Backend:** B0 (determinism), B1 (noise), B2 (per-instrument coupling),
  B4 (saturation), B5 (PSF), B6 (stray light), B7 (irregular bodies) are
  **done**. B7 covers the procedural mesh render, navigator prediction,
  end-to-end navigation, the four mesh-vs-ellipsoid / pose scenarios (via the
  `nav_override` render-vs-navigation channel), and the named-mesh sourcing
  decision (the procedural generator is the source; named `.obj` meshes are a
  documented future hook, not built -- resolving open question 3). B3 (smear) is
  **deferred** on upstream `psfmodel` (issue #151); B8 (diffraction) is
  **deferred** as a lowest-priority future enhancement, only relevant if a
  spike-producing instrument is ever supported (issue #152).
- **GUI:** G0, G1, G2, G4, G5, G6, G7, G8 are **done**. G3 (smear) is **deferred**
  with B3 (issue #151).
- **Test layer:** T1 (scene catalog), T2 (regression baselines), T3 (sweeps) are
  **done**. T4 (algorithmic invariants) is **done** -- disc, blob (incl. the
  displaced high-phase crescent), limb, ring edge, star field, camera roll, the
  `StarUniqueMatchNav` one-star path, and the `StarRefineNav` pass-2
  prior-refinement path all navigate from catalog scenes and recover planted
  transforms. T5 (alpha bootstrap), T6 (real-vs-sim diagnostics), and T7
  (calibration validation) are **deferred** until the project is ready to run the
  real-data confidence calibration; they are specified together in issue #153.

### 0.2 Navigation-accuracy work completed alongside the plan

A separate track of navigation/accuracy fixes was driven by the T3 sweeps and the
per-technique characterization (`docs/simulator_report/`). Completed:

1. **Star reliability gate** -- the star model no longer gates a star on the
   saturation / cosmic-ray mask at its predicted position (a star's predicted
   pixel is not special, and a sharp peak self-flags as a cosmic ray); gating is
   occlusion-only. See `dev_guide_techniques_star_field` and the star model
   chapters.
2. **Star PSF-fit refinement (SNR-adaptive)** -- after the RANSAC match,
   `StarFieldFromCatalogNav` re-centroids each matched inlier with a
   maximum-likelihood PSF fit when the star is faint and keeps the
   brightness-weighted moment when it is bright (the moment beats the PSF fit's
   sub-pixel-phase bias floor above a configurable integrated-SNR ceiling,
   `psf_refine_snr_max`). Dim-field offset error dropped ~0.05 -> ~0.025 px,
   bright field ~0.005 px.
3. **Disc correlator sub-pixel band-limit** -- `evaluate_candidate` low-passes both
   refine surfaces before the cross-power (`refine_lowpass_sigma_px`, default
   1.0 px), removing a sharp-edge aliasing S-curve (~0.03 px, up to ~0.1 px at the
   worst 2-D phase). See `dev_guide_techniques_body_disc`.
4. **Off-grid invariant offsets** -- every fixed algorithmic-invariant scene plants
   one common off-grid offset `(1.43, -0.61)` px (sweeps keep round grids), so a
   technique cannot land on a sub-pixel-bias null; the per-scene errors are at one
   phase and directly comparable.
5. **Per-technique characterization** -- `technique_snr_characterization.py` and
   `star_snr_characterization.py` (runner-only) sweep SNR and injected offset
   across backgrounds and emit the report figures.

**Documentation split (cardinal):** the simulator report
(`docs/simulator_report/`) is *ephemeral* -- current statistics, figures, and
conclusions only, no decisions or change history. All mechanisms, decisions,
rejected alternatives, and "what breaks if you don't" live in the **developer
guide** (`docs/dev_guide/`). Keep new findings on that split.

### 0.3 Next steps, in order

**Done (this track):** *Sim instrument config -- inherit / override / fully
self-specify.* A scene may now carry an ``instrument_config`` mapping deep-merged
over the resolved instrument block (:func:`resolve_sim_inst_config` gained an
``overrides`` arg). Omit it to inherit; override individual keys to pin them;
name ``generic`` and override everything to self-specify. Pinned keys are
decoupled from the live camera config, so a later camera-config edit cannot shift
a sim scene's pinned behavior. Wired through the scene schema (`src/nav/sim/scene.py`),
both consumers (`obs_inst_sim.py`, `render.py`), and the GUI load/save round-trip;
documented in `dev_guide_observations`, the simulated-image user guide, and the
scene README. Resolves open question 2.

**Investigated and re-scoped (this track):** *Staged limb / ring sub-pixel refine against the
continuous gradient field.* The premise -- that the limb/ring floor is a removable
quantized-DT artifact -- was disproved on the full sim pipeline. The continuous gradient-ridge
refine (`gradient_ridge_refine` in `dt_fitting`, wired but held off via the per-technique
tuning flag) does not help: the **ring** is already at its floor (~0.016 px, below the 0.02 px
target) because its edges are symmetric, and the **limb** gets *worse* (clean planted-(0,0)
recovery worsens from ~0.10 px to ~0.17 px). The limb floor is a **model-vs-image edge offset**
-- the gradient peak of a limb-darkened, PSF-blurred limb sits ~0.1 px inside the geometric
silhouette the `LIMB_ARC` predicts, and the DT-LM only hits ~0.10 px because its quantization
accidentally pulls back toward truth. The genuine remedy is a model-side limb-edge prediction
(limb-darkening- and PSF-aware), tracked in **issue #150**. See the corrected mechanism note in
`dev_guide_techniques_body_limb` and `dev_guide_techniques_dt_fitting`.

**Done (this track):** *Large-offset acquisition for the blob centroid.* The per-technique
navigable-ceiling characterization showed the bbox-limited technique was **BodyBlobNav**
(~6 px capture range, set by the predicted bounding box); the DT edge techniques (limb, ring,
disc) already reach ~extfov, so they did not need this. `BodyBlobNav` now runs a coarse
acquisition before the brightness-weighted centroid: it uses an installed pass-1 prior when
present, otherwise correlates a blob-shaped disc (filled-disc matched filter) over the search
window to re-center each blob's box on the body. This extends the blob capture range from
~6 px to ~extfov (recovery to a few hundredths of a pixel out to the margin on the low-phase
`small_sphere_base` sweep), with the disc correlation gated to bodies at least half-lit (a
high-phase crescent past its box still needs a prior, since a disc template cannot match a
crescent). Self-contained in `nav_technique_body_blob.py`; no orchestrator change. See
`dev_guide_techniques_body_blob` and the simulator report's navigable-range table.

**Done (this track):** *High-phase blob acquisition (phase-aware crescent template).* The
remaining blob gap -- a high-phase crescent displaced beyond its bounding box -- is closed with
a phase-aware coarse template. Above half phase `BodyBlobNav` now correlates a synthesized
Lambertian crescent (oriented along the sub-solar direction the `BODY_BLOB` feature carries in
`sub_solar_dir_vu`) instead of skipping the coarse stage; at or below half phase it keeps the
filled disc. The crescent's center sits on the body center rather than the bright arc, and the
template's brightness-centroid offset is added back so the recovered shift is expressed against
the lit centroid the feature carries. Recovery holds to a few hundredths of a pixel out to the
extfov margin across seeds and illumination directions; the
`planted_offset_blob_crescent_displaced` invariant scene guards it. The sub-solar direction is
derived model-agnostically from the rendered lit-centroid offset, so a SPICE-backed body model
populates it without extra plumbing. The only residual gap is a high-phase body whose
illumination geometry is unpopulated (no direction), which still relies on the bounding-box
centroid or a prior. Self-contained in `nav_technique_body_blob.py` plus the shared
`_build_blob_feature`; no orchestrator change. See `dev_guide_techniques_body_blob`,
`dev_guide_navigation_models_body`, and the simulator report's navigable-range table.

The per-technique navigable-ceiling gaps identified by the T3 sweeps are now closed (disc,
limb, ring at ~extfov; blob at ~extfov for both disc and crescent regimes).

**Done (this track): B7's remaining increments.** The four irregular-body scenarios
are in, each pinned by a catalog scene with a regression baseline and a recovery /
behavioral assertion, and the named-mesh sourcing decision is resolved. The enabling
mechanism is the body `nav_override` mapping -- the renderer always draws the true
geometry, while `NavModelBodySimulated` builds its predicted body from the body params
with `nav_override` overlaid, so the navigation geometry can diverge from the render
geometry without touching the rendered image (it never moves the center, so the
predicted body stays at the unshifted position the planted offset is measured from).
Both scenarios 2 and 3 keep the predicted body on the *same* mesh renderer as the
rendered body (the ellipsoidal prediction is realized as the zero-relief limit of the
mesh), so the residual is pure shape or pose mismatch rather than a renderer-convention
skew between the mesh and ellipsoid renderers:

1. **Scenario 1 (mesh vs mesh, same pose)** -- `planted_offset_limb_mesh` pins the
   resolved-mesh limb to exact planted-offset recovery.
2. **Scenario 2 (mesh vs ellipsoid, same pose)** -- `planted_offset_shapemismatch`
   renders a lumpy mesh and predicts its smooth (ellipsoidal) limit; it still recovers
   under a pixel, and the `irregularity_shape_mismatch` sweep walks the rendered relief
   up to show the recovered centroid bias growing and the confidence falling.
3. **Scenario 3 (mesh vs ellipsoid, disagreeing pose)** -- the `pose_disagreement`
   degradation sweep walks the predicted pose off the rendered one (limb error grows,
   then the limb self-flags spurious), and `test_sim_irregular_pose` asserts the
   per-technique decision: the wrong-pose limb degrades far off while the pose-free blob
   stays accurate. Per-technique, NOT on the fused tier (the confidence alphas are
   uncalibrated placeholders).
4. **Scenario 4 (centroid-only)** -- `planted_offset_blob_mesh_crescent` pins a
   high-phase mesh crescent to `BodyBlobNav`, the mesh variant of the ellipsoid crescent
   blob scenes.

**Named meshes / `shape_meshes/` sourcing decision (section 12.3, resolved):** the
procedural generator stays the sim's mesh source; named `.obj`/`.ply` meshes are NOT
bundled. The sim's job is controlled sensitivity and regression, for which a
parameterized irregular body (seed, lumpiness, pose) spans the irregularity axis
continuously -- more useful than one fixed Hyperion shape, which a single committed mesh
cannot. Real irregular-body accuracy is calibrated against the real Cassini
Hyperion/Phoebe images in the operator library (cardinal principle 3.1), so a bundled
mesh would not move the calibration needle, and it would add repo bloat or
network-at-test-time. The renderer already accepts any `Mesh` (vertices + faces), so a
`shape_model: named_mesh` + `mesh_file` loader can be added behind that hook if a
specific body is ever needed; it is left unbuilt until proven necessary.

**The only remaining sim items are deferred**, parked in GitHub issues: smear rendering +
GUI controls (B3/G3, #151, pending upstream `psfmodel`), diffraction spikes (B8, #152, a
lowest-priority future enhancement), and the calibration test layer (T5/T6/T7, #153, gated
on the real-data confidence calibration, which the now-complete B7 `phase_irregularity_factor`
diagnostics feed when it runs).

**Validation discipline (carried throughout):** validate accuracy changes only with
simulated images; use ensemble statistics across seeds and offsets, not single
scenes; keep the full sweep out of the normal pytest run; guard found defects with
fast bug-specific regression scenes. The worktree has its own `.venv` -- run
`/seti/newnav/rms-nav-sim/.venv/bin/python` and, for the `tests.*` runners,
`PYTHONPATH=/seti/newnav/rms-nav-sim`. Integration tests and real-image
re-blessing need the holdings env vars (`OOPS_RESOURCES`, `PDS3_HOLDINGS_DIR`,
`PDS4_HOLDINGS_DIR`, `UCAC4_PATH`, `YBSC_PATH`).

---

## 1. Context and problem statement

### 1.1 What this plan is for

The rms-nav navigation pipeline (`src/nav/nav_orchestrator/`) maps
real spacecraft images to per-image pixel-offset corrections plus a
calibrated `(offset, sigma, confidence_tier)` triple.  The
confidence formula for each technique is `sigmoid(α₀ + Σ αᵢ × xᵢ)`
where the `αᵢ` are coefficients tuned once against an
operator-curated library of ~50 real-holdings images
(`tests/integration/image_library/images/<scene_class>/<image_id>.yaml`,
documented in `tests/integration/image_library/images/README.txt` and
`PHASE10_CURATION.md`).

The Phase 10 curation work is operator-bottlenecked.  Finding real
images with specific scene attributes (a body at a specific phase
angle, a body of a specific irregularity, a frame with exactly N
catalog stars, a frame with stray-light contamination) is hard.
Manually navigating each found image to set the ground-truth offset
is labor-intensive.  The set of scenes that *exist* in the real
archives is fixed by what spacecraft happened to take during their
missions; there are gaps that no amount of curation can fill.

The simulator system at `src/nav/sim/` can synthesize images on
demand for arbitrary geometry.  The natural question — and the
question that motivates this plan — is *can we use the simulator
to make the calibration work easier or more thorough?*

### 1.2 The cardinal answer: augment, do not substitute

A separate analysis (preserved here so this document is
self-contained) concluded:

- **Sim cannot substitute for real-data calibration.** The sim has
  its own model assumptions (ellipsoidal bodies, Gaussian PSF,
  additive Gaussian noise, no scattered light, no instrument
  artifacts).  Calibrating the per-technique α coefficients against
  sim-image diagnostic distributions tunes the formula for the sim's
  idealized world — and on real images the same diagnostics
  distribute differently, so the calibrated tiers underflow or
  overflow systematically.  The whole class of failure modes the
  operator-curated library exists to characterize (irregular bodies,
  real PSF wings, real noise structure, real scattered light) is
  fundamentally absent from sim today.
- **Sim can augment the verification, sensitivity, and
  regression layers.** Once calibrated against real data, the
  sim is the right tool for: single-variable sensitivity analysis
  (sweep phase angle continuously, verify confidence response is
  smooth and monotonic), unit testing the technique algorithms
  (planted-offset recovery), regression coverage for combinations
  that don't exist in real archives, and bootstrap pre-tuning to
  give the curve-fit a sensible α starting point.
- **The gap between sim and real is closeable.** Specific
  improvements to the sim — realistic noise, per-instrument PSF
  coupling, non-ellipsoidal bodies, smear as a true convolution —
  bring sim diagnostics close enough to real that the sim's
  verification layer becomes load-bearing.  This plan enumerates
  those improvements.

The plan is therefore organized around two goals:

1. **Close the realism gap** between sim and real, in priority order
   of which gaps most distort the diagnostics the calibration cares
   about.
2. **Build the test infrastructure** that lets the sim carry weight
   in regression / sensitivity / bootstrap work without having to
   become the calibration target itself.

### 1.3 Why both backend and GUI must move together

The sim GUI (`src/main/nav_create_simulated_image.py`, a single ~2500
LOC PyQt6 application) is operator-facing: it lets a human dial in
geometry parameters and watch the rendered image update live.  Every
backend parameter that the GUI doesn't expose is invisible to the
operator and effectively dead.  Conversely, every GUI control that
isn't backed by a meaningful backend parameter is a lie.

The plan therefore phases the backend and GUI together: each backend
addition lands with a corresponding GUI tab / control / preview pane,
and each GUI affordance is backed by a real backend parameter that is
also reachable from a YAML scene spec (so the GUI is one of three
peers — Python API, YAML, GUI — not the sole control surface).

---

## 2. System inventory (what exists today)

This is the state of the world at the time this plan was written.
Everything below is grep-verifiable in the repo.

### 2.1 Backend: `src/nav/sim/`

Three Python modules, ~1800 LOC total:

- **`render.py`** (~840 LOC) — top-level scene composition.
  - `render_combined_model(...)` is the entry point that composes
    bodies, rings, stars, noise, and background-stars into a single
    image array.
  - `render_stars(...)` rasterizes a list of stars with a single
    `psfmodel.GaussianPSF` (configurable `psf_sigma`).  Smear is
    implemented as a per-star translation `(move_v, move_u)` —
    **not** a true line-integral convolution, so smeared stars are
    rendered as translated points rather than line segments.
  - `render_background_noise(img, noise_level, seed)` adds **purely
    additive Gaussian noise** with `np.random.default_rng(seed)`.
    There is no Poisson term, no read-noise floor, no cosmic-ray
    injection, no missing-data marker injection.
  - `render_background_stars(...)` synthesizes a uniform random
    background-star field (not from a real catalog).
  - Caching is via `functools.lru_cache` on parameter hashes; same
    inputs → same outputs in-process, but cross-process determinism
    relies on the seed parameter being passed explicitly.

- **`sim_body.py`** (~480 LOC) — single-body silhouette renderer.
  - `create_simulated_body(axis1, axis2, ...)` renders an
    **elliptical** silhouette (two axes only — there is no third
    axis or 3D shape).  No polyhedral mesh support, no DSK
    integration, no actual irregular silhouette.
  - Lambertian shading via `_lambertian_shading` with optional
    crater overlay via `_add_craters_and_shading` (random craters
    drawn into the brightness map; craters do *not* perturb the
    silhouette).
  - The body is positioned by the caller via `(dv, du)` translation
    of the rendered patch.

- **`sim_ring.py`** (~440 LOC) — geometric ring rendering with
  per-pixel anti-aliasing of the radial edge.  Computes ring-edge
  radii from mode-1 / mode-N analytical models.  Independent of the
  per-planet ring catalog the navigator consumes.

### 2.2 GUI: `src/main/nav_create_simulated_image.py`

Single 2509-LOC PyQt6 application:

- `CreateSimulatedImageModel` is a `QMainWindow` with a tabbed
  parameter pane (General + per-body + per-ring + a `'+'` add-tab)
  and a live preview panel with zoom/pan.
- General tab carries: image size (V/U), offset, random seed,
  closest planet (for ring geometry), time + epoch (TDB seconds),
  background noise intensity (single Gaussian-amplitude slider),
  background star count + PSF sigma + radial-distribution exponent.
- Per-body tabs (added via `_add_body_tab`) carry the
  `create_simulated_body` parameters: axis1/axis2, position, crater
  density, albedo, etc.
- Per-ring tabs (added via `_add_ring_tab`) carry mode amplitudes,
  inner/outer radii, etc.
- The GUI saves a model spec as JSON plus the rendered image as a
  PNG / FITS pair via `nav_create_simulated_image` from the CLI
  side.

### 2.3 Navigator-side glue

- `src/nav/obs/obs_inst_sim.py` — `ObsSim` subclass of
  `ObsSnapshotInst`; loads a sim image and presents it through the
  same interface as Cassini / Voyager / Galileo / NHLORRI obs.
- `src/nav/dataset/dataset_sim.py` — `DataSetSim` registered under
  the `'sim'` dataset name.
- `src/nav/nav_model/nav_model_body_simulated.py` — body NavModel
  variant that renders the predicted body silhouette using the same
  sim renderer; used so the navigator's body overlay matches the
  sim image when both are derived from the same parameters.
- `src/nav/nav_model/nav_model_rings_simulated.py` — analogous for
  rings.

### 2.4 Configuration

- `src/nav/config_files/config_440_sim.yaml` (32 lines).  This is a
  *separate* config from the per-instrument configs
  (`config_400_inst_coiss.yaml`, `config_410_inst_gossi.yaml`,
  `config_420_inst_nhlorri.yaml`, `config_430_inst_vgiss.yaml`).
  The sim's noise model, PSF, ADC ceiling, etc. are sim-specific
  rather than mirrored from the instrument it's pretending to be.

### 2.5 Test coverage

A handful of unit tests in `tests/nav/nav_model/test_nav_model_body_simulated.py`
and similar for rings.  No scene-spec catalog, no rendered-image
regression baselines, no sensitivity-sweep harness, no algorithmic-
invariant test set keyed off planted offsets.

### 2.6 Known limitations relative to real holdings

Direct consequences of section 2.1:

| Real-image phenomenon | Sim today | Gap impact on calibration |
|---|---|---|
| Body silhouette deviates from ellipsoid | Always ellipsoidal | `phase_irregularity_factor` always 0; can't exercise irregular-body regime |
| Body rotational pose | Scene-supplied rotation angles (not SPICE) | Pose is ground truth shared by renderer and navigator; for chaotic rotators (Hyperion) this is an advantage -- any true pose can be planted, and the navigator can be given an agreeing, disagreeing, or absent pose (see B7) |
| Star PSF | Single Gaussian | Per-instrument wings absent (no diffraction spikes to model: the supported telescopes have no support vanes) |
| Star centroid offset from catalog | Zero by construction | Can't exercise unique-match assignment ambiguity |
| Per-pixel noise | Additive Gaussian, signal-independent | No Poisson, read floor, cosmic rays, dropouts; MAD-noise estimator overfits to a regime that doesn't exist in real frames |
| Smear on long exposure | Per-star translation | Real smear is a line integral; centroid bias and SNR distribute differently |
| Saturation / bloom | Not modeled | Saturated bright stars don't get flagged in sim; no full-well clipping |
| Stray-light gradient | Not modeled | `scattered_light` scene class can't be sim'd |
| Missing-data markers | Not modeled | Partial-dropout classifier can't be exercised |

---

## 3. Cardinal principles

These principles are non-negotiable for the rest of the plan.

### 3.1 Sim is verification, real data is calibration

The `αᵢ` coefficients in `config_510_techniques.yaml` are tuned
*only* against the operator-curated real-holdings library.  Sim
images are used to *verify* that calibration (sensitivity sweeps,
algorithmic invariants, regression coverage) and to *bootstrap*
α to a sensible starting point — never to determine the final
calibrated values.  Any phase of this plan that crosses this line is
out of scope.

### 3.2 Per-instrument coupling

Today the sim has its own
`config_440_sim.yaml`, separate from the per-instrument configs.
This means a sim image isn't a "Cassini NAC raw frame", it's a "sim
frame".  The plan converges on **the sim consuming the same
per-instrument configs the navigator uses**, so a "sim COISS NAC raw"
frame goes through the same noise-sigma branch, the same
`signal_dn_to_image_unit_scale` path, and the same predicted-SNR
formula as a real CISS frame.  Without this, sim sensitivity analysis
isn't transferable to real-image interpretation.

### 3.3 Determinism

Every random call (noise, cosmic rays, star jitter, dropout markers)
takes a single per-scene seed and produces byte-identical output for
identical inputs.  No module-level RNG state, no `time.time()`
seeding fallbacks.  Required so sim images can participate in the
rounded-baseline regression layer.

### 3.4 Three peers, not one

The sim's parameter surface has **three equally-valid entry points**:

1. **Python API** (`render_combined_model`, etc.).
2. **YAML scene specs** (under
   `tests/integration/sim_scenes/<class>/<name>.yaml`, mirroring the
   real-image library's structure).
3. **GUI** (`nav_create_simulated_image`).

Every backend parameter must be reachable from all three.  Adding a
new physical effect means adding the parameter to the rendering
function, the YAML schema, *and* a GUI control.  The GUI is the
operator-friendly view; the YAML is the durable catalog artifact;
the API is what tests call.

### 3.5 Phase backend before GUI

GUI controls that aren't backed by real backend parameters are lies.
Backend parameters that the GUI doesn't expose are operator-invisible.
Each phase pairs a backend addition with the corresponding GUI work,
and the GUI work cannot start until the backend lands (otherwise
the controls have nothing real to drive).

### 3.6 No backwards-compat shims

Per the project's cardinal principles (Part 0 of `AUTONAV_PLAN.md`):
when sim parameters are renamed or restructured, the YAML / GUI / API
all change in the same PR.  No `if old_param: ... elif new_param:`
adapters.  Existing sim scenes either get migrated in the same PR or
are deleted.

---

## 4. Backend improvement plan

Phases are listed in *dependency order* — earlier phases unblock
later ones — not in priority order.  Within each phase the work is
small enough to land as one or two PRs.

### Phase B0: Determinism and seed audit (prerequisite for everything)

**Goal:** establish that every random call in the sim is seeded
through a single per-scene seed parameter and produces byte-identical
output for identical inputs.

**Scope:**
- Audit every `np.random.*`, `random.*`, and `default_rng(...)` call
  in `src/nav/sim/`.  Any that doesn't take a seed traceable to the
  scene's `random_seed` parameter is a determinism bug; fix it.
- Add a unit test (`tests/nav/sim/test_sim_determinism.py`) that
  renders the same scene twice and asserts byte-identical output.
- Add a per-effect sub-seed derivation:
  `noise_seed = hash((scene_seed, 'noise'))`, etc., so adding a new
  randomized effect (cosmic rays in B1) doesn't perturb the noise
  output of pre-existing scenes.

**Why first:** every later regression test relies on this.  Without
it, "the sim image changed" is ambiguous — was it the seed walking,
or did B5's PSF change really affect this scene?

**Files touched:** `src/nav/sim/render.py`, `src/nav/sim/sim_body.py`,
`src/nav/sim/sim_ring.py`, new `tests/nav/sim/test_sim_determinism.py`.

### Phase B1: Realistic noise model

**Goal:** replace pure additive Gaussian noise with the real
camera-noise structure.

**Scope:**

- **Poisson shot noise on the signal.** After all bodies / rings /
  stars are composed, sample each pixel from a Poisson distribution
  with mean equal to the noise-free signal in DN.  Required so that
  noise grows with brightness — the regime real cameras have.  Use
  `numpy.random.Generator.poisson` with the per-effect seed from B0.
- **Gaussian read-noise floor** added on top of Poisson at
  `read_noise_dn` from the per-instrument config (B2's coupling makes
  this pull from the right config block).  Until B2 lands, default
  to `config_440_sim.yaml`'s value.
- **Cosmic-ray hits** as sparse single-pixel spikes at a configurable
  fluence (events / cm² / sec × exposure_sec × pixel area).  Use a
  Poisson-distributed count of hits, each placed at a uniformly
  random pixel with a single-pixel intensity drawn from a long-tail
  distribution (Pareto or log-normal).  Spike intensity exceeds
  saturation by design — this exercises the orchestrator's
  `cosmic_ray_mask` path.
- **Missing-data markers** at the instrument's marker value (0 for
  raw CISS, NaN for CALIB) at a configurable rate.  Spread as
  isolated pixels (most common in real archives) plus optionally a
  contiguous-block "row dropout" mode for the partial-dropout
  classifier.

**Backend interface:**

```python
def render_background_noise(
    img: NDArrayFloatType,
    *,
    read_noise_dn: float,         # additive Gaussian sigma
    cosmic_ray_rate_per_sec: float = 0.0,  # events / cm² / sec
    exposure_sec: float = 1.0,
    pixel_area_cm2: float = 1.0,
    missing_data_marker_dn: float = 0.0,
    missing_data_rate: float = 0.0,
    seed: int,
) -> None:
    ...
```

The Poisson term is applied implicitly to the signal already in
`img` (no separate parameter — Poisson is always on whenever signal
is present).

**Why now:** this is the single biggest lever in the plan.  Without
it, every per-feature SNR diagnostic is in the wrong regime, and
sim-derived sensitivity analysis doesn't transfer to real frames.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/config_files/config_440_sim.yaml`, new
`tests/nav/sim/test_sim_noise.py` (asserts the noise distribution
matches the parameters).

### Phase B2: Per-instrument coupling

**Goal:** the sim consumes the same `config_4N0_inst_*.yaml` files
the navigator uses, so a "sim COISS NAC raw" frame goes through the
same code paths a real CISS NAC raw frame does.

**Scope:**

- The sim takes an `instrument` parameter (string: `'coiss_nac'`,
  `'coiss_wac'`, `'coiss_calib_nac'`, `'coiss_calib_wac'`,
  `'gossi'`, `'nhlorri'`, `'vgiss'`).  This selects which
  per-instrument config block drives the rendering: noise sigma /
  read-noise / saturation_dn / marker_value / data_units / star
  PSF / mag_offset / extfov_margin.
- `config_440_sim.yaml` becomes a small wrapper that defines the
  *defaults* used when no instrument is specified, plus
  sim-specific knobs that don't belong on real instruments (e.g.
  `closest_planet` for ring geometry).  All physical parameters
  delegate to the per-instrument config.
- `obs_inst_sim.py` exposes `inst_config` populated from the
  selected per-instrument block, so the orchestrator's
  `instrument_settings_from_obs(obs)` returns the right
  `InstrumentSettings` (noise, saturation_dn, signal scale,
  rotation flag).

**Why now:** every later phase that adds physics (smear, saturation,
PSF, etc.) needs to look up the right per-instrument value.  Doing
B2 first means each later phase's parameter wiring is a one-line
config lookup, not a duplicate sim-side knob.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/sim/sim_body.py`, `src/nav/obs/obs_inst_sim.py`,
`src/nav/dataset/dataset_sim.py`, `src/nav/config_files/config_440_sim.yaml`,
new `tests/nav/sim/test_sim_instrument_coupling.py`.

### Phase B3: Smear as a true convolution

**Status: deferred -- tracked in issue #151** (filed self-contained, with the
backend rendering work and the GUI/scene controls of G3 together).  `psfmodel`
is being extended to produce smeared PSFs directly.  Once that lands, the sim
renders smear by asking `psfmodel` for the smeared kernel rather than carrying
its own line-integral code, so this phase waits on that upstream work instead of
building a parallel implementation now.

**Goal:** rendered smeared stars become line integrals of the PSF,
matching the navigator's `smeared_psf.py` model.

**Scope:**

- `render_stars` no longer translates the PSF by `(move_v, move_u)`.
  Instead, build the smeared kernel via the same
  `nav.nav_model.stars.smeared_psf.compute_smeared_psf(psf, move_v,
  move_u)` the navigator uses, and rasterize that.
- The kernel's total flux is preserved (peak DN drops as smear
  length grows), matching the real photometry.
- Smear length is per-star (different stars in the FOV have
  different smear vectors when the camera rotates during the
  exposure).  Today the GUI applies one global smear; backend should
  accept per-star.

**Why now:** unblocks honest calibration of `StarUniqueMatchNav` and
`StarFieldFromCatalogNav` against scenes with non-trivial smear.
Today's translated-PSF approximation gives star centroids the wrong
sub-pixel position and the wrong SNR, which would systematically
mistune the star-related α coefficients if sim were used in
calibration.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_smear.py` (asserts a smeared star's
brightness-weighted centroid lies along the smear track).

### Phase B4: Saturation + bloom

**Goal:** clip pixels at the per-instrument full-well DN, with
column-direction bloom on cameras that have it.

**Scope:**

- After all noise is added, clip pixel values at
  `saturation_dn` from the per-instrument config (B2's coupling).
- For cameras with documented column-bloom behavior (Cassini NAC at
  high SNR), spread saturated-pixel excess flux into the column
  direction up to a configurable bloom-length.
- Saturated pixels should land on `saturation_mask_ext` when the
  navigator processes the sim image — the orchestrator's
  `_build_saturation_mask` already does the comparison; this phase
  just makes sure the pixels actually hit the threshold.

**Why now:** unblocks the saturated-bright-star reliability gate
(STAR features whose predicted pixel is in the saturation mask get
zeroed reliability per `_reliability_from_snr`).  Without B4, sim
star-cal scenes always navigate cleanly because no star ever
saturates.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_saturation.py`.

### Phase B5: Realistic per-instrument PSF [done]

**Goal:** sim stars use the same `psfmodel.PSF` instance the
navigator uses, with instrument-specific wings.

**Scope:**

- Replace `GaussianPSF(sigma=...)` in `render.py` with the PSF
  returned by `obs.star_psf()` for the selected instrument (B2 makes
  this lookup possible).  For Cassini NAC this means a real
  measured-from-data PSF if one is in `psfmodel`; for other
  cameras, the configured `star_psf_sigma` parameterized PSF.
- No diffraction spikes.  The Cassini telescope has no secondary-mirror
  support vanes, so its stars carry no cross-shaped diffraction pattern
  no matter how bright; the other supported cameras are likewise modeled
  without spikes.

**Why now:** brings star-centroid diagnostics into the same
distribution as real CISS frames.  Less load-bearing than B1 and B2,
but matters for `StarFieldFromCatalogNav` calibration verification.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_psf.py`.

### Phase B6: Stray-light gradient [done]

**Goal:** optional per-frame stray-light contribution that exercises
the BANDPASS_DOG source-image filter.

**Scope:**

- A new optional `stray_light` scene parameter taking
  `(amplitude, gradient_direction_deg, model='linear' | 'radial')`.
  Linear = a brightness ramp across the frame; radial = a brightness
  bump centered at a configurable point (mimics a sun-near-FOV
  flare).
- Applied multiplicatively to the noise-free image before noise is
  added.

**Why now:** the only way to test that the navigator's
source-image filter (`config_4N0_inst_*.yaml` `source_image_filter:
BANDPASS_DOG`) actually mitigates scattered light without finding a
real Galileo or Voyager outer-leg frame.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/config_files/config_440_sim.yaml`, new
`tests/nav/sim/test_sim_stray_light.py`.

### Phase B7: Non-ellipsoidal bodies [done]

**Goal:** render at least one canonical irregular body
(Hyperion-like, Phoebe-like) from a polyhedral mesh rather than as
an ellipsoid silhouette.

**Progress:**

- *Increment 1 (render side):* `src/nav/sim/sim_body_polyhedral.py`
  provides a procedural irregular-mesh generator and a z-buffered
  polyhedral renderer (orthographic projection, flat Lambertian shading
  matching the ellipsoid's light convention, supersampled limb).  A body
  with `shape_model: polyhedral_mesh` plus a `pose_euler_deg` orientation
  routes through it in `render.py`.
- *Increment 2 (navigator-side prediction + geometry separation):*
  `NavModelBodySimulated` predicts a mesh silhouette when its own params
  say `shape_model: polyhedral_mesh`, via the shared `MeshBodySpec` /
  `render_mesh_body_image` primitive.  Because the model reads its own
  params, the predicted shape and pose can differ from what was rendered
  -- this realizes the render-geometry / navigation-geometry separation.
  Verified: the predicted mesh reproduces the rendered shape when params
  agree (scenario 1/2 base), differs from an ellipsoid prediction
  (scenario 2), and changes under a disagreeing pose (scenario 3).  The
  mesh seed/pose are explicit body params (not the scene noise seed) so
  both sides reproduce the same shape.

- *Increment 3 (end-to-end navigation wiring):* the model-selection
  layer **does** exist (`build_models_for_obs` is a registry factory over
  each NavModel's `instances_for_obs`).  The gap was that the simulated
  NavModels opted out of it.  Now `NavModelBodySimulated` /
  `NavModelRingsSimulated` build one instance per body/ring from the sim
  params the obs carries, gated on `obs.is_simulated`; the real
  `NavModelBody` / `NavModelRings` / `NavModelStars` skip a simulated obs.
  A sim image now navigates end to end (`navigate_image_files` /
  `NavOrchestrator`) and recovers a planted offset -- the first rung of
  the T4 layer (`tests/integration/test_sim_navigation.py`).  This also
  required a **bias pedestal** in the detector model (`bias_dn`, default
  20): signal-free sky rendered at exactly 0 DN collided with the
  missing-data marker (0) and the frame was misclassified as
  `mostly_missing_data`.

- *Increment 4 (render-vs-navigation separation channel + pose scenarios):* a
  body may carry an optional `nav_override` mapping.  The renderer ignores it and
  always draws the true geometry; `NavModelBodySimulated` builds its predicted
  body from the body params with `nav_override` overlaid (`_nav_params`), so the
  navigation geometry diverges from the render geometry at the scene level without
  touching the rendered image.  The override never moves the center, so the
  predicted body stays at the unshifted position the planted offset is measured
  from.  Both scenarios 2 and 3 keep the predicted body on the same mesh renderer
  as the rendered body -- the ellipsoidal prediction is the zero-relief
  (`mesh_lumpiness: 0.0`) limit of the mesh -- so the residual is pure shape or
  pose mismatch, not a renderer-convention skew (the standalone ellipsoid renderer
  transposes axis1/axis2 relative to the mesh renderer, so mixing the two would
  inject a spurious 90 deg rotation).  The four scenarios each land as a catalog
  scene with a regression baseline:

  1. **Scenario 1 (mesh vs mesh, same pose) -- invariant.**
     `planted_offset_limb_mesh` pins the resolved-mesh limb to exact planted-offset
     recovery (a few tenths of a pixel).
  2. **Scenario 2 (mesh vs ellipsoid, same pose) -- invariant + sweep.**
     `planted_offset_shapemismatch` renders a lumpy mesh and predicts its smooth
     limit; it still recovers under a pixel.  The `irregularity_shape_mismatch`
     sweep walks the rendered relief up and shows the recovered centroid bias
     growing (~0 to several px) and the fused confidence falling -- the
     `phase_irregularity_factor` regime.
  3. **Scenario 3 (mesh vs ellipsoid, disagreeing pose) -- degradation sweep +
     behavioral test.**  The `pose_disagreement` sweep walks the predicted pose off
     the rendered one; the limb error grows and then the limb self-flags spurious
     (the navigator declining a confidently-wrong limb).  `test_sim_irregular_pose`
     asserts the per-technique decision: the wrong-pose limb degrades far off while
     the pose-free blob stays accurate (the body's bulk shape is a
     centrally-symmetric triaxial ellipsoid, so its lit-weighted centroid barely
     moves under rotation).  Asserted per-technique, NOT on the fused tier (the
     confidence alphas are uncalibrated placeholders).
  4. **Scenario 4 (centroid-only) -- invariant.**
     `planted_offset_blob_mesh_crescent` pins a high-phase (120 deg) mesh crescent
     to `BodyBlobNav`, the mesh variant of the ellipsoid crescent blob scenes.

**Named meshes / `shape_meshes/` sourcing (section 12.3, resolved):** the procedural
generator stays the sim's mesh source; named `.obj`/`.ply` meshes are not bundled.
A parameterized irregular body (seed, lumpiness, pose) spans the irregularity axis
continuously, which is what the controlled sensitivity / regression role needs --
more useful than one fixed Hyperion shape; and real irregular-body accuracy is
calibrated against the real Cassini images in the operator library (principle 3.1),
so a bundled mesh would not move the calibration needle and would add repo bloat or
network-at-test-time.  The renderer already accepts any `Mesh`, so a
`shape_model: named_mesh` + `mesh_file` loader can be added behind that hook if a
specific body is ever needed; it is left unbuilt until proven necessary.

**Scope:**

This phase is paired with `AUTONAV_PLAN.md` Part 13b §7 (resolved
irregular-body LIMB_ARC via real shape models).

`oops` will not gain DSK kernel support in the foreseeable future, so
the DSK-via-oops path is not available.  The only path is the
**standalone polyhedral renderer**: the sim grows a small renderer that
projects triangle-mesh vertices through the body's pose into image
space and draws the silhouette.  The mesh itself is generated
procedurally (`make_irregular_mesh`: seed + lumpiness + pose), not loaded
from a bundled file -- see the named-mesh sourcing resolution above and in
section 12.3.  A `shape_model: named_mesh` + `mesh_file` loader under a
`src/nav/sim/shape_meshes/<BODY>.obj` (or .ply) hook is a documented future
extension, left unbuilt until a specific body is proven necessary.

#### Orientation is always an input; the scenarios differ only in shape

The navigator never estimates body orientation from the pixels.  In real
navigation the pose (the body-fixed -> camera rotation) is an *input*
from SPICE; the navigator renders its predicted body at that pose and
solves only for the pointing offset.  In the sim there is no SPICE
(`ObsSim` fabricates the geometry), so the pose is supplied instead as
**scene ground truth** carried on the obs and read by the sim-aware body
NavModel -- exactly the channel `NavModelBodySimulated` already uses to
read `rotation_z` / `rotation_tilt` today.

Orientation is therefore a *required* input to draw the navigator's
predicted body in every case, because a triaxial ellipsoid's silhouette
and terminator both depend on its orientation.  What varies across the
test scenarios is the **shape model** the navigator predicts with, and
**whether the assumed pose agrees with the rendered (true) pose**.  This
means the sim must separate two things that are usually identical:

- the **render geometry** -- the true mesh and pose used to draw the
  image (ground truth), and
- the **navigation geometry** -- the shape and pose the navigator's
  predicted body is built from.

Default: the two are identical (the navigator knows the truth).  When
they diverge, the divergence is the thing under test.

#### The scenario matrix (maps onto the technique ladder)

1. **mesh vs mesh, same pose** -- navigator predicts the true mesh at the
   true pose; recovers the planted offset exactly.  Tests the resolved
   mesh LIMB technique.
2. **mesh vs ellipsoid, same pose** -- navigator predicts an *ellipsoid*
   at the *true* pose; the only error is shape.  This cleanly isolates
   and exercises `phase_irregularity_factor` (correct pose, wrong shape).
3. **mesh vs ellipsoid, disagreeing pose** -- the chaotic-rotator case
   for a body whose orientation we cannot know (Hyperion).  Render the
   mesh at the true pose but give the navigator an ellipsoid at a
   *different* assumed pose, so the predicted silhouette is both the
   wrong shape and the wrong orientation.  Measures how badly limb
   fitting degrades when the pose is untrustworthy -- and confirms the
   navigator should demote to a pose-free technique here.
4. **centroid only (no oriented model)** -- the realistic Hyperion path.
   There is no usable orientation, so the navigator carries *no*
   ellipsoid or mesh and falls back to the lit-weighted centroid
   (`BodyBlobNav`), which is orientation-independent.  Render the true
   irregular lit shape and assert the BLOB centroid still recovers the
   planted offset.  This is valuable precisely because it is the correct
   technique for chaotic rotators, where scenarios 1-3 do not apply.

Scenarios 3 and 4 are why the render/navigation-geometry separation
matters: a chaotic rotator's pose is genuinely unknown, so the useful
tests are the ones where the navigator either guesses the orientation
wrong (3) or declines to use it at all (4).

#### What form each remaining scenario takes (test vs sweep)

The scenarios are not all the same kind of artifact.  This repo has two tracks
-- exact-recovery **invariant tests** (`tests/integration/test_sim_algorithmic_invariants.py`,
in the normal pytest run) and **characterization sweeps**
(`tests/integration/sim_sweeps/*.yaml` + `sim_sweep_runner` -> report figures,
with coarse response-shape assertions in `test_sim_sweeps.py`).  Map the
remaining scenarios onto them as follows:

- **Scenario 1 (mesh vs mesh, same pose) -> invariant test.** A scene under
  `sim_scenes/algorithmic_invariants/` whose render and navigation geometry are
  the same mesh + pose; assert the resolved-mesh limb path recovers the planted
  offset to a tight tolerance.  This is the standard planted-offset invariant
  shape.  (`planted_offset_irregular` already exercises a mesh body via the disc
  correlator; the new work is pinning the *mesh limb* technique.)
- **Scenario 2 (mesh vs ellipsoid, same pose) -> invariant test + a
  characterization sweep.**  A pinned invariant scene confirms the body still
  navigates with a shape-mismatched ellipsoid prediction; a sweep over phase (or
  over a shape-irregularity parameter) characterizes how `phase_irregularity_factor`
  and the resulting centroid bias grow, feeding the simulator report.  Correct
  pose, wrong shape -- this is the clean isolation of the irregularity term.
- **Scenario 3 (mesh vs ellipsoid, disagreeing pose) -> a sweep AND a behavioral
  test; NOT an exact-recovery invariant.**  The point is that recovery
  *degrades*, so do not assert "limb recovers the planted offset".  Instead:
  (a) a **sweep** that walks the navigation pose away from the render pose and
  records the limb technique's offset error and confidence as the disagreement
  grows (a degradation curve in the report); and (b) a **behavioral test** that
  asserts the *decision* -- on a wrong-pose body the limb per-technique error is
  much larger than on the correct-pose body while the pose-free `BodyBlobNav`
  stays accurate, i.e. the system should not trust the confidently-wrong limb.
  Because the ensemble's demote-to-pose-free choice is confidence-driven and the
  confidence alphas are uncalibrated placeholders, assert this **per-technique**
  (the geometry/error gap is real and placeholder-independent), not on the fused
  tier; the fused "prefers the pose-free answer" assertion only becomes clean
  after the real-data calibration (issue #153).
- **Scenario 4 (centroid-only) -> invariant test.**  An irregular-mesh body with
  the navigator carrying no oriented model, pinned to `BodyBlobNav`, asserting
  exact planted-offset recovery.  Largely already covered in ellipsoid form by
  the high-phase crescent blob scenes; the delta is a mesh-rendered variant.

**Prerequisite (done):** `NavModelBodySimulated` is wired into the live
model-selection path -- `build_models_for_obs` builds it (and
`NavModelRingsSimulated`) for a simulated obs, gated on `obs.is_simulated`, while
the SPICE-backed models skip a simulated obs.  So sim body navigation (mesh or
ellipsoid) is available; the remaining work is the scenarios above, not the
wiring.

**Why now:** the `phase_irregularity_factor` term added in Phase 10
§F is identically zero on every sim frame today.  Without B7, that
term cannot be calibrated, sensitivity-tested, or regression-covered
on sim.

**Files touched:** new `src/nav/sim/sim_body_polyhedral.py`,
`src/nav/sim/render.py`, the sim-aware body NavModel selection path
(`src/nav/nav_model/nav_model_body_simulated.py`, including the
`nav_override` render-vs-navigation channel), the four
`tests/integration/sim_scenes/.../planted_offset_*`/`hyperion_pose_disagree`
scenes + baselines, the `irregularity_shape_mismatch` and `pose_disagreement`
sweeps, `tests/integration/test_sim_irregular_pose.py`, and
`tests/nav/sim/test_sim_irregular_body.py`.  No `shape_meshes/` directory is
created: the mesh source is procedural (section 12.3, resolved).

### Phase B8: Diffraction spikes -- deferred (future enhancement, lowest priority)

**Status: deferred -- tracked in issue #152.**  None of the supported cameras
have diffraction spikes.  The Cassini telescope in particular has no
secondary-mirror support vanes, so bright stars never show a cross-shaped
diffraction pattern, and there is nothing to model today.  Rather than drop it
permanently, it is parked as a lowest-priority future enhancement: implement it
only if the navigation system is ever extended to support an instrument whose
optics genuinely produce diffraction spikes, at which point the simulator should
reproduce the spike signature for star-detection / artifact-rejection testing.
Until such an instrument lands, no work is scheduled against it.

---

## 5. GUI improvement plan

Each GUI phase consumes one or more backend phases.  The phasing
mirrors the backend phasing — never start a GUI phase before its
backend is in.

### Phase G0: Three-peer audit [done]

**Goal:** confirm every existing GUI control is also reachable from
the YAML scene-spec schema (Phase T1 below) and the Python API.
Where there's drift, fix it.

**Scope:** for each control in `nav_create_simulated_image.py`
(general tab, per-body tab, per-ring tab), verify the corresponding
field exists in the YAML schema and in the API.  Where it doesn't,
add it.

**Audit outcome:** the drift found was (a) the GUI had no `instrument`
control -- closed by G1; (b) the GUI's `background_noise_intensity`
slider is inert since B1 replaced the additive-Gaussian model -- to be
replaced by the noise panel in G2; (c) the GUI load path dropped the
catalog-only `noise` / `stray_light` / `exposure_sec` blocks -- now
preserved on load (round-trip parity) pending their own controls in
G2 / G6.  Per-body / per-ring controls already map to the API and YAML.

**Why now:** keeps the three peers from drifting out of sync as the
later phases land.  Cheap to do once; expensive to retrofit later.

**Files touched:** `src/main/nav_create_simulated_image.py`, scene
schema docs.

### Phase G1: Instrument selector [done]

**Goal:** add a top-level "Instrument" combo box that drives the
per-instrument config used for the rest of the rendering.

**Done:** a General-tab "Instrument" combo (`generic` plus every sim
camera) writes `sim_params['instrument']`, which the renderer already
consumes for noise / saturation / PSF / units (B2).  It round-trips
through save / load and defaults to `generic`.  The "update General-tab
defaults on instrument change" affordance (showing the instrument's PSF
sigma / dimensions) is deferred -- the operator's fields stay as set and
the renderer applies the instrument settings regardless; revisit if
operators want the fields auto-populated.

**Backend dependency:** B2 must land first.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G2: Noise-model controls [done]

**Goal:** expose the B1 noise model in the GUI.

**Done:** the inert "Background noise intensity" slider is removed and
replaced by a detector-noise panel writing the `sim_params['noise']`
block: a Poisson shot-noise checkbox, a read-noise (DN) spin, a
cosmic-ray rate spin (events / cm^2 / sec), and a missing-data rate
spin.  Values round-trip through save / load.  Two drafted items are
deferred: the missing-data block-size (row-dropout) control waits on
that render mode, and auto-populating the panel from the selected
instrument's defaults on instrument change is the same deferral as G1
(the renderer applies instrument defaults when the scene leaves a field
unset).

**Backend dependency:** B1, B2.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G3: Smear controls

**Status: deferred -- tracked in issue #151** (specified together with the B3
rendering work, since the GUI/scene smear controls and the flux-correct renderer
land as one unit).

**Goal:** expose B3's per-star smear correctly.

**Scope:** today the "Background stars" tab has a single PSF sigma
spin and no smear control (smear is only applied via the body's
motion vector).  Add:

- "Camera smear (global)" two-spinner row: smear vector v / u in px.
  Applied to every star and to the catalog-star overlay.
- For per-body tabs, the existing motion vector becomes a true
  smear (per B3) rather than a translation.
- Live preview shows the smear track on a representative star.

**Backend dependency:** B3.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G4: Saturation visualization [done]

**Goal:** show the operator which pixels saturate.

**Done:** a "Saturation overlay" checkbox in the visual-options row.
When checked, the preview renders in RGB and pixels at or above the
selected instrument's `saturation_dn` are painted red, and a status-bar
label shows the per-image saturation fraction.  The overlay is a display
toggle (re-displays without re-rendering) and is inert for
calibrated_if instruments, which have no saturation DN.

**Backend dependency:** B4.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G5: PSF preview pane [done]

**Goal:** let the operator see what PSF the sim is using.

**Done:** a collapsible "PSF preview" group (initially closed) on the
General tab renders the selected instrument's Gaussian star PSF as an
inset image and annotates its sigma / FWHM.  It refreshes when the
instrument selector changes (and on load).  The smeared-PSF toggle is
omitted -- it depends on B3, which is deferred (issue #151).

**Backend dependency:** B5.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G6: Stray-light controls [done]

**Goal:** expose B6's stray-light gradient.

**Done:** a stray-light panel on the General tab writes the
`sim_params['stray_light']` block: an amplitude spin (0 = off, the
default), a direction spin (degrees, wrapping), and a linear/radial
model dropdown.  Values round-trip through save / load.  A spin is used
for direction instead of the drafted dial widget (consistent with the
rest of the General tab and testable).

**Backend dependency:** B6.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G7: Shape-file selector for irregular bodies

**Goal:** let the operator pick "Hyperion (polyhedral mesh)"
instead of just "Hyperion (ellipsoid)" for a per-body tab.

**Done:** each per-body tab has a "Shape model" dropdown
(`ellipsoid` / `polyhedral_mesh`) plus mesh-lumpiness, mesh-seed, and
three mesh-pose (X/Y/Z degrees) controls, wired to the body params B7
consumes.  New bodies default to `ellipsoid`.  Two deviations from the
draft: the axes are *not* grayed out for a mesh (they scale the
procedural mesh, so they remain meaningful), and the dropdown lists the
generic `polyhedral_mesh` rather than named `HYPERION` / `PHOEBE`
meshes, since the sim uses a procedural generator until named meshes are
sourced (section 12.3).

### Phase G8: Scene catalog browser [done]

**Goal:** the GUI becomes a first-class editor for the YAML scene
catalog (Phase T1 below).

**Done:** the GUI has "Load Scene (YAML)" and "Save Scene (YAML)"
buttons alongside the JSON parameter buttons.  Loading a scene validates
it and populates every tab (via a shared `_apply_params_dict` the JSON
loader also uses); saving writes a schema-valid YAML derived from the
current scene.  So an operator can render a scene in the GUI and save it
as a catalog artifact, and load a catalog scene to edit it.

**Schema promotion:** to let production GUI code consume the schema
without depending on a test module, the scene schema moved from
`tests/integration/sim_scene.py` to **`src/nav/sim/scene.py`** (the
importable peer named in principle 3.4).  It gained
`scene_dict_from_sim_params` / `save_sim_scene` (the inverse mapping for
saving); the T1 structural test now imports from `nav.sim.scene` and
passes the catalog root explicitly.  Buttons are used rather than a File
menu, consistent with the existing parameter buttons; "Save & Render" is
unneeded since saving does not change the live preview.

---

## 6. Calibration test strategy

This is the section that turns the upgraded sim into useful
calibration / verification infrastructure.  Phases T1–T7.

### Phase T1: Scene-spec YAML catalog [done]

**Goal:** durable artifact directory for sim scenes, mirroring the
real-image library's layout.

**Done:** `tests/integration/sim_scene.py` validates the schema and maps
a scene to sim params (`SimScene.to_sim_params`); `tests/integration/
sim_scenes/<class>/<name>.yaml` holds the initial catalog with a
`README.txt` schema doc; `tests/integration/test_sim_scenes.py` enforces
the structural invariants (every scene validates, directories are
declared classes, names unique, every scene renders) and runs in the
default suite.  One addition to the drafted schema: `image_size_vu` is a
required field (the renderer needs it).  The `smear_sweep` class is
declared but unpopulated pending B3.

**Scope:**

- New directory `tests/integration/sim_scenes/`.
- Subdirectories per scene class:
  `tests/integration/sim_scenes/<scene_class>/<scene_name>.yaml`.
  Scene classes are scoped to "what we're calibration-testing", not
  the real-image library's `DECLARED_SCENE_CLASSES`.  Initial set:
  - `phase_sweep_regular_body/` — same body at increasing phase
    angles (0, 30, 60, 90, 120, 150°).
  - `phase_sweep_irregular_body/` — same Hyperion-mesh body at
    increasing phase angles.
  - `noise_sweep/` — same scene at increasing read-noise levels.
  - `smear_sweep/` — same scene at increasing smear lengths.
  - `range_sweep/` — same body at increasing distances (resolves
    the BLOB threshold transition).
  - `multi_body_geometry/` — controlled multi-body arrangements.
  - `algorithmic_invariants/` — clean planted-offset scenes for
    technique unit tests.
- YAML schema documented in
  `tests/integration/sim_scenes/README.txt` (analogous to
  `images/README.txt` for the real library).  Key fields:

  ```yaml
  schema_version: 1
  scene_name: <stem must match filename>
  instrument: coiss_nac      # see G1
  random_seed: 42
  midtime_utc: '2010-01-01T00:00:00Z'
  exposure_sec: 1.0

  bodies:
    - name: HYPERION
      shape_model: polyhedral_mesh   # or 'ellipsoid'
      ...
  rings: [...]
  stars:
    catalog: ucac4
    smear_v_px: 0.0
    smear_u_px: 0.0
  noise:
    poisson: true
    read_noise_dn: 4.0
    cosmic_ray_rate_per_sec: 0.0
    missing_data_rate: 0.0
  stray_light:
    amplitude: 0.0
    direction_deg: 0.0
    model: linear

  ground_truth:
    planted_offset_dv_px: 0.0   # see T4
    planted_offset_du_px: 0.0
    planted_rotation_deg: 0.0
  ```

- Validator at `tests/integration/sim_scene.py` (parallel to
  `tests/integration/sidecar.py`) that loads + validates each YAML.
- Structural-invariants test
  (`tests/integration/test_sim_scenes.py`) that asserts every
  YAML validates and every directory name is a known scene class.

**Why now:** turns sim scenes from one-off Python fixtures into
catalog artifacts that the GUI (G8), the test harness (T2–T7), and
human reviewers can all consume.

**Files touched:** new `tests/integration/sim_scenes/`, new
`tests/integration/sim_scene.py`, new
`tests/integration/test_sim_scenes.py`.

### Phase T2: Deterministic regression baselines [done]

**Done:** `tests/integration/sim_baseline.py` defines the `SimBaseline`
schema (scene_name, status, rounded offset, confidence -- with `status`
and nullable offsets so failed scenes are recorded too, unlike the
real-image baseline) plus `baseline_for_scene` (render + navigate).
`tests/integration/sim_baselines/<scene>.json` holds the catalog's
baselines; `tests/integration/update_sim_baselines.py` regenerates them
(`python -m ...`); `tests/integration/test_sim_baselines.py` re-navigates
every scene and asserts an exact rounded match.

Two adjustments from the draft, both forced by reality: the technique
solvers carry sub-millipixel floating-point jitter across processes under
parallel load, so (a) the test is `@pytest.mark.integration` -- the
deliberate layer, exactly like the real-image baseline test, rather than
the fast default suite -- and (b) rounding is coarsened to 2 decimals
(0.01 px / 0.01 confidence), still a real tripwire but immune to the
jitter.  The `range_sweep/small_body` scene was shrunk so it robustly
fails (it sat at the navigability boundary and flipped success/fail); its
baseline records `failed` and will be re-blessed when BLOB lands.

**Goal:** sim scenes get a regression-baseline layer parallel to
the real-image library's `tests/integration/baselines/`.

**Scope:**

- New directory `tests/integration/sim_baselines/<scene_name>.json`.
- Each baseline records the rounded
  `(offset_dv_px, offset_du_px, confidence)` the orchestrator
  produces when navigating the sim'd scene.
- Test layer
  `tests/integration/test_sim_baselines.py` parametrized one case
  per `(scene, baseline)` pair; renders the scene, runs the
  navigator, asserts exact-equal rounded match.
- Updater under
  `tests/integration/update_sim_baselines.py` (parallel to
  `tests/integration/update_baselines.py` from earlier work);
  invoked as `python -m tests.integration.update_sim_baselines …`.

**Why now:** a sim scene catalog without baselines doesn't gate
regressions — anyone can rerun and get any result.  Baselines turn
"this scene navigates to X" into a tripwire.

**Files touched:** new
`tests/integration/sim_baselines/`, new
`tests/integration/test_sim_baselines.py`, new
`tests/integration/update_sim_baselines.py`.

### Phase T3: Single-variable parameter sweeps [done]

**Goal:** harness that renders a sweep of scenes varying one
parameter, navigates each, and emits a per-parameter response curve.

**Done:** `tests/integration/sim_sweep.py` defines the `SweepSpec` schema
(`base_scene`, a list of dotted `parameters` moved together, `values`) plus
`run_sweep`, which drives one catalog scene by overriding the parameter(s) in its
sim-params and navigating each step into a `SweepRow` (status, offset error vs the
planted offset, confidence, primary technique).  `tests/integration/
sim_sweeps/<name>.yaml` holds the specs; `sim_sweep_runner.py` (`python -m`)
writes per-step JSON response curves under `sim_sweeps/results/` (a regenerable
diagnostic, gitignored, not a re-blessed baseline); `test_sim_sweeps.py` asserts
the invariants and is `@pytest.mark.integration` (each sweep navigates several
frames).  Three sweeps over one resolved-sphere base scene
(`phase_sweep_regular_body/regular_sphere_base.yaml`):

- **noise** (`read_noise_dn` 1 -> 64): recovers within tolerance at low noise and
  fails at the navigability cliff (the highest level), capturing the
  success-to-failure degradation.
- **phase** (0 -> 150 deg): the resolved body navigates to success and recovers
  within 0.5 px at every phase (disc at moderate phase, blob at the extremes).
- **range** (diameter 130 -> 12 px): the primary technique transitions
  `BodyLimbNav -> BodyDiscCorrelateNav -> BodyBlobNav` as resolution falls, and
  the smallest body is unnavigable -- the technique ladder the range regime is
  meant to exercise.
- **offset (blob / disc)**: the planted offset swept across whole-pixel,
  near-boundary (0.99), perfect-fraction, and arbitrary (0.12783, 0.73912) values
  on a blob body and a disc body.  The blob is quantization-free (~0.006 px at
  every offset); the disc shows a periodic ~0.3 px error (zero at the half-pixel)
  that the sweep traces to the gradient-magnitude NCC sub-pixel refinement (raw
  NCC reaches ~1/128 px).  This is a core-correlator defect in
  `navigate_with_pyramid_kpeaks`, surfaced by the sweep and flagged for a fix
  validated against the real-image library -- the clearest case of the sim layer
  catching a real navigation defect.
- **star roll** (0.5 -> 2 deg): the field recovers the roll in the working window;
  below ~0.75 deg `StarFieldFromCatalogNav`'s RANSAC cannot separate a small roll
  from a translation and collapses to zero (the two-star path extends the floor to
  ~0.5 deg).

The harness reads the planted ground truth from the post-override sim params, so a
sweep over the offset or the roll itself tracks correctly (`SweepRow` carries both
`offset_error_px` and `rotation_error_deg`).  The invariants are trends and bounds
(a transition, a degradation, recovery within tolerance), not the placeholder-alpha
confidence curve, which barely moves on these clean frames -- so the
confidence-monotonicity checks the draft below imagined wait on the Phase 10
calibration and are deferred to T7.

Two navigation findings the sweeps surfaced are documented in the standalone
performance report (`docs/simulator_report/`): the gradient-NCC sub-pixel bias
above, and that the small-body navigation floor (~16-24 px) is set by the
`BODY_BLOB` reliability gate, not the centroid algorithm (the centroid is exact
well below it).  Navigation is also confirmed unit-agnostic -- an I/F-calibrated
`coiss_calib_nac` scene (`planted_offset_disc_if`) recovers identically to raw DN.

**Scope:**

- Sweep specs are themselves YAML files at
  `tests/integration/sim_sweeps/<sweep_name>.yaml`.  Each sweep
  declares a base scene (referencing a `sim_scenes/` template), a
  parameter name, and a list of values.
- Harness at
  `tests/integration/sim_sweep_runner.py` (also runnable as
  `python -m`) that renders the sweep, navigates each frame, and
  emits a JSON or CSV of `(parameter_value, offset_error_px,
  confidence, primary_technique)` per row.
- Test layer
  `tests/integration/test_sim_sweeps.py` that asserts invariants
  per sweep.  Examples:
  - `phase_sweep_regular_body` — confidence should be roughly flat
    across phase 0–60° and decrease modestly above 60°.
  - `phase_sweep_irregular_body` — confidence should drop more
    sharply with phase (the `phase_irregularity_factor` term is
    doing its job).
  - `noise_sweep` — confidence monotonically decreases with
    `read_noise_dn`.
  - `range_sweep` — primary technique transitions from BodyDisc to
    BodyLimb to BodyBlob at the right diameter thresholds.

**Why now:** this is the verification layer the calibration sweep
relies on.  After fitting α against the real-image library, run
these sweeps and assert the calibrated formulas respond smoothly to
single-parameter changes.  A non-monotonic or jagged response
signals a calibration bug.

**Files touched:** new `tests/integration/sim_sweeps/`, new
`tests/integration/sim_sweep_runner.py`, new
`tests/integration/test_sim_sweeps.py`.

### Phase T4: Algorithmic invariants (planted-offset / planted-rotation) [done]

**Goal:** unit-test the techniques against ground truth that's
correct *by construction*.

**Progress:** the harness exists -- `tests/integration/
test_sim_algorithmic_invariants.py` renders every
`sim_scenes/algorithmic_invariants/` scene into an `ObsSim`, navigates
it, and asserts success plus offset recovery within tolerance (correct by
construction, so no baseline to bless).  Initial scenes:
`planted_offset_disc` (ellipsoid), `planted_offset_small` (small offset),
`planted_offset_irregular` (mesh body).  All navigate via
`BodyDiscCorrelateNav` and recover to a few tenths of a pixel; the
invariant bound is set to 1.0 px (the disc/correlation technique is not
sub-0.1 px on these scenes, so the drafted 0.1 px target is loosened).

`BodyBlobNav` coverage is now in: `NavModelBodySimulated` emits a
`BODY_BLOB` feature (the lit-weighted, orientation-independent centroid)
in addition to `BODY_DISC`, so a simulated body can navigate by the blob
path.  The `planted_offset_blob` scene plus three dedicated tests pin
`only_techniques='BodyBlobNav'` and assert the blob alone recovers the
planted offset.  One constraint shaped the scene: a small body sits at the
disc-spurious boundary where the full-ensemble outcome flips across
processes (the same BLAS jitter the T2 baselines hit), so the scene body
is large enough that the full ensemble is stably disc-dominated while the
blob-only tests carry the blob proof.

High-phase blob recovery is now fixed end to end, by three coupled
corrections (all of which make the sim behave like a real frame, so the
fix transfers to real-image navigation):

1. **`NavModelBodySimulated` emits a *tight* bbox** around the body
   silhouette (plus slop), matching the SPICE-backed `NavModelBody`.  It
   previously emitted the whole frame, so `BodyBlobNav` integrated its
   observed centroid over the entire image and the few hundred lit
   crescent pixels were swamped by the ~0.1% of sky pixels that randomly
   clear the noise threshold across thousands of background pixels.
2. **`BodyBlobNav` subtracts a frame-global background pedestal** (bias +
   dark) before the moment, estimated once over the valid sky region
   (`_estimate_frame_background_and_sigma`) rather than per-bbox -- a
   per-bbox estimate over a body-filled high-phase box overestimates the
   sky and over-thresholds the dim crescent.
3. **`BodyBlobNav` thresholds against the *sky* noise**, not the global
   MAD `image_noise_sigma`, which a bright body inflates (its brightness
   gradient widens the whole-sensor spread).  The inflated sigma raised
   the threshold, cut the dim crescent, and made the thresholded observed
   centroid diverge from the model's continuous predicted centroid.  For a
   small body (the typical resolved-enough-for-a-blob case) the sky sigma
   equals the global sigma, so this only bites when a large body would
   otherwise inflate it.

Together these take blob-only recovery on a clean coiss_nac moon from
failing above ~50 deg phase to sub-pixel through 160 deg (e.g. ~0.05 px at
90 deg, ~0.23 px at 120 deg, ~0.8 px at 140 deg).  The `planted_offset_blob`
(moderate) and `planted_offset_blob_crescent` (120 deg, disc-spurious)
scenes plus blob-only tests guard it.

Star coverage is now in.  `NavModelStarsSimulated` (a thin subclass of
`NavModelStars` registered for simulated obs) sources its star list from the
sim renderer's output (`obs.sim_star_list`, the unshifted `MutableStar`
records) and reuses the catalog-driven model's STAR feature emission, CRLB
covariance, reliability, and annotations unchanged -- so a simulated star
field is navigated by exactly the same `StarFieldFromCatalogNav` /
`StarUniqueMatchNav` / `StarRefineNav` code a real frame is.  Closing this
required a half-pixel rendering fix that is general to real-image fidelity:
`render_stars` passed `psfmodel.eval_rect` the sub-pixel fraction directly,
but `eval_rect` measures its offset from the pixel's lower edge (`offset=0`
centers the PSF half a pixel low), whereas the detection centroid and the
star NavModel's predicted position both use the pixel-center convention.  The
rendered star therefore sat half a pixel from where the model predicted it,
and every star navigation carried a constant -0.5 px bias.  Adding 0.5 to the
eval offset renders the star centroid exactly at its predicted `(v, u)`, so a
star the model predicts at `(v, u)` lands there in the image and a technique
recovers the planted offset without bias (recovery is now ~0.02-0.05 px on a
clean coiss_nac field).  The `planted_offset_star_field` scene (a six-star
field) plus full-ensemble recovery tests guard it: the field routes through
`StarFieldFromCatalogNav`'s RANSAC pattern match, `StarUniqueMatchNav`'s
two-star path, and `StarRefineNav`, and the fused offset is recovered stably.
A lone-star scene was deliberately not added: a single star is correctly a
low-confidence (rank-low) fix, so its full-ensemble *status* sits on the
success/fail boundary and flips under parallel BLAS jitter -- which would flake
every full-ensemble layer (the generic navigate assertion, the recovery
assertion, and the T2 baseline, whose recorded status would be marginal).  The
one-star path is therefore left to a later non-status-gated harness.

Planted-rotation recovery is now in.  The renderer applies `planted_rotation_deg`
as a camera roll about the boresight -- rotating each star (and each body's
center and line-of-sight pose) before the translation offset -- while the star
record keeps its unrolled catalog `(v, u)`, so the NavModel predicts the
unrolled geometry and `StarFieldFromCatalogNav`'s similarity (rotation +
translation) fit recovers the roll.  A design decision made this tractable:
rotation fitting is gated per-instrument (`fit_camera_rotation`, true only for
Galileo SSI among the real cameras), but Galileo's broad PSF and shallow 8-bit
well make a clean star field hard to construct.  Rather than couple the sim's
rotation testing to one finicky camera -- or invent a fake `sim_rotation`
instrument that emulates nothing (violating principle 3.2) -- a scene carries an
optional `fit_camera_rotation` flag that `ObsSim.from_file` injects into the
resolved instrument config.  This decouples "does the navigator solve for a
roll" from the camera's optics, so `planted_rotation_star_field` keeps Cassini
NAC's sharp PSF (precise centroids) while exercising the 3-DoF path, and
StarField recovers a 1.5 deg roll to ~0.05 deg.  The roll is asserted on the
StarField per-technique result: the technique recovers it geometrically
(non-spurious), but on a clean sim field its placeholder-alpha confidence holds
the *fused* status at a stable `failed`, so the scene is held out of the
full-ensemble navigate assertion and the roll is checked per-technique.  The
roll is kept small (1.5 deg) so the outer stars stay inside the matcher's inlier
tolerance; larger rolls drop the inlier count below the quorum.

`BodyLimbNav` and `RingEdgeNav` coverage is now in.  Both techniques consume a
polyline feature the simulated models did not previously emit, so each model
grew one:

* `NavModelBodySimulated` emits a `LIMB_ARC` -- the silhouette boundary
  (`_compute_limb_mask_from_body_mask`) sampled into a vertex polyline with
  outward normals.  `BodyLimbNav`'s distance-transform fit aligns it to the
  image edge and recovers the planted offset to ~0.02 px on a well-resolved
  sphere.  Emission is gated on a minimum diameter (100 px) and low phase
  (<= 60 deg): the catalog body gates the limb on its ellipsoid-fit uncertainty,
  which small / low-resolution / high-phase bodies fail, and -- because the limb
  is an LM-refined fit -- emitting it on a marginal body injects cross-process
  jitter into the fused offset.  The diameter floor sits above every existing
  catalog scene's body, so no existing baseline shifts; the dedicated
  `planted_offset_limb` scene uses a 130 px body.
* `NavModelRingsSimulated` emits one `RING_EDGE` per rendered edge -- the
  `compute_border_atop_simulated` edge mask sampled into a polyline with outward
  radial normals and a curvature-based `is_straight_line` flag.  `RingEdgeNav`
  fits two curved arcs and recovers the planted offset in both axes (RMS
  ~0.4 px, rank_1 false).  Like rotation, the offset is asserted on the
  per-technique result: the technique is non-spurious but its placeholder-alpha
  confidence holds the fused status at a stable `failed`.

The two remaining technique-pinned cases are now in.  `StarUniqueMatchNav`'s
one-star path is covered by `planted_unique_star_single` (a single bright star,
too sparse for the field matcher) pinned with `only_techniques='StarUniqueMatchNav'`.
`StarRefineNav`'s prior-refinement is covered by `planted_refine_star_field` (a
multi-star field whose pass-1 ensemble reaches `success` and installs a prior, so
the pass-2 refine runs), read from the full-ensemble per-technique result since a
prior-requiring pass-2 technique cannot be isolated with `only_techniques`.  The
displaced high-phase crescent (`planted_offset_blob_crescent_displaced`, a 120 deg
body planted ~20 px past its bounding box) was also added once `BodyBlobNav` gained
its phase-aware crescent coarse-acquisition.

**Done.**  Every technique on the ladder navigates from a catalog scene and recovers
its planted transform.  The only residual item is *not* a T4 gap: lifting the
StarField / RingEdge confidence off their placeholder-alpha floors so a clean
rotation or ring field can navigate to a fused `success` is part of the real-data
confidence calibration (issue #153), and until then those two are asserted
per-technique.

**Scope:**

- Scene specs in `sim_scenes/algorithmic_invariants/` carry
  `ground_truth.planted_offset_dv_px` / `planted_offset_du_px` /
  `planted_rotation_deg`.
- The sim renders at
  `(predicted_pose ± planted_offset, predicted_rotation ± planted_rotation)`.
- The test asserts the navigator recovers the planted values to
  within a tight tolerance (e.g. 0.1 px, 0.01°).
- Coverage targets:
  - `BodyDiscCorrelateNav` — clean disc, planted (dv, du), assert
    recovery to <0.1 px.
  - `BodyLimbNav` — limb fit on planted offset.
  - `BodyBlobNav` — BLOB on planted offset; assert lit-weighted
    centroid recovers correctly even at high phase.
  - `RingEdgeNav` — curved-ring planted offset.
  - `StarFieldFromCatalogNav` — N planted stars + planted rotation,
    assert both translation and rotation recovered.
  - `StarUniqueMatchNav` — single bright star, planted offset.
  - `StarRefineNav` — pre-existing prior + small planted refinement.

**Why now:** these tests replace many of the inline-constructed
fixtures in `tests/nav/nav_technique/` with catalog-driven scenes,
and they grow the test suite faster than operator labor can grow
the real library.

**Files touched:** new
`tests/integration/sim_scenes/algorithmic_invariants/*.yaml`, new
`tests/integration/test_sim_algorithmic_invariants.py`.

### Phase T5: α-bootstrap pre-fit [deferred]

**Status: deferred -- tracked in issue #153** (specified together with T6 and
T7 as the calibration test layer), until the operator is ready to run the
real-data WS-5 calibration (`plans/VALIDATION_AND_CALIBRATION_PLAN.md`)
against the curated image library.  T5 has no real-data dependency of its
own -- it runs sim scenes whose recovery error is known by construction --
but its only consumer is the real calibration's starting guess, so it is
sequenced to run immediately before that calibration rather than now.
Deferring it also lets the irregular-body diagnostics
(`phase_irregularity_factor`, B7 scenarios 2-3) come online first, so the
bootstrap fits over a representative diagnostic spread.

**Goal:** use sim-derived diagnostic distributions to give the
real-data WS-5 calibration a sensible starting point for the α
coefficients.

**Scope:**

- Run a representative set of sim scenes through the navigator.
- For each scene, record the per-technique diagnostics and the known
  recovery error (planted offset minus recovered offset).
- Pre-fit the α coefficients so reported confidence tracks the
  sim recovery error — the sim-anchored arm of WS-5's reliability
  calibration, run early.  The sim anchors are approximate by design
  (sim realism is itself under validation), but they rule out
  obviously-wrong α regions.
- The output α is **not** committed to
  `config_510_techniques.yaml`.  It's used as the initial guess for
  the real-data WS-5 calibration, replacing the current
  arithmetic-illustrative default coefficients.
- Documented as a separate one-off helper script under
  `tests/integration/sim_alpha_bootstrap.py`.

**Why deferred, not dropped:** when run, it shrinks the optimization basin
for the real-data calibration without committing to sim's diagnostic
distributions.  It is optional even then -- skip if the operator finds the
default coefficients already close enough -- which is why it waits until
the calibration is actually about to happen.

**Files touched:** new
`tests/integration/sim_alpha_bootstrap.py`.

### Phase T6: Real-vs-sim diagnostic distribution comparison

**Status: deferred -- tracked in issue #153** (specified together with T5 and
T7 as the calibration test layer); needs the real-image library navigable
(holdings + SPICE env) and is sequenced around the real-data calibration.

**Goal:** quantify how close sim has come to real, per technique.

**Scope:**

- For each technique, harvest the diagnostic values reported on:
  - Every real-image library sidecar (run the navigator over the
    real library).
  - Every sim-scene catalog entry (run the navigator over the sim
    scenes).
- Plot per-diagnostic histograms side-by-side (real vs. sim).
- Report Kolmogorov–Smirnov distances per diagnostic.
- The output is a Markdown report at
  `tests/integration/sim_vs_real_report.md` regenerable on demand.

**Why now:** this is the empirical answer to "how good is the sim
*now*".  Before any backend phase lands, run T6 to get a baseline.
After each backend phase lands, re-run T6 and report the change.
The KS distance per diagnostic measures whether a backend
improvement actually closed the gap it was supposed to.

**Files touched:** new
`tests/integration/sim_vs_real_diagnostics.py`,
`tests/integration/sim_vs_real_report.md`.

### Phase T7: Calibration validation harness

**Status: deferred -- tracked in issue #153** (specified together with T5 and
T6 as the calibration test layer); hard-gated on the real-data calibration
having landed, so it is the last of the three.

**Goal:** after the real-data WS-5 calibration lands, verify the
calibrated formulas pass the sim sweeps from T3.

**Scope:**

- Same sweep infrastructure as T3.
- Assertions tightened:
  - Monotonicity in the expected direction for each parameter.
  - Smoothness (no per-sample jumps > some configurable epsilon).
  - The three regimes (high / medium / low confidence) appear in
    the right portions of each sweep — e.g. the noise sweep should
    cross from high to medium at the noise level corresponding to
    the threshold, not somewhere arbitrary.
- Failures are *advisory* (informational on the PR, not
  CI-blocking) until the operator certifies the calibration is
  stable enough that sim sensitivity tests can gate.

**Why now:** turns the sim sweeps from "useful diagnostic" into
"calibration-validation tripwire".  Catches calibration
regressions that don't show up on the real-image library because
the real library doesn't sample the parameter space densely enough.

**Files touched:** extends
`tests/integration/test_sim_sweeps.py`.

---

## 7. Phase ordering and dependencies

The phases map onto a dependency graph.  Earlier phases unblock
later ones:

```
B0 (determinism) [done]
 │
 ├──→ B1 (noise model) [done]
 │     │
 │     └──→ B4 (saturation) [done], B5 (PSF) [done], B6 (stray light) [done], B7 (irregular) [done]
 │
 └──→ B2 (per-instrument coupling) [done]
       │
       └──→ B3 (smear) [deferred: psfmodel, #151], B4 [done], B5 [done], B6 [done], B7 [done]

T1 (scene catalog)
 │
 ├──→ T2 (regression baselines)
 ├──→ T3 (sweeps)
 ├──→ T4 (algorithmic invariants)
 │
 └──→ G8 (GUI catalog browser)

G0 (3-peer audit) before any GUI phase
G1 follows B2
G2 follows B1, B2
G3 follows B3 (both deferred: psfmodel, #151)
G4 follows B4
G5 follows B5
G6 follows B6
G7 follows B7

T5 (α bootstrap) — deferred (#153); run immediately before the real-data WS-5 calibration
T6 (real-vs-sim) — deferred (#153); needs the real-image library navigable; run around the calibration
T7 (calibration validation) — deferred (#153); last, depends on real-data calibration landing
```

### Minimum viable cut

If the work has to be sequenced over multiple releases, the smallest
useful self-contained slice is:

1. **B0** (determinism)
2. **B1** (noise model)
3. **B2** (per-instrument coupling)
4. **G0** (3-peer audit)
5. **G1, G2** (instrument selector + noise GUI)
6. **T1** (scene catalog)
7. **T2** (regression baselines)
8. **T4** (planted-offset invariants)
9. **T6** (real-vs-sim baseline report)

With this slice, the sim becomes a deterministic, instrument-aware,
realistic-noise simulator with a YAML-catalog-driven test layer that
can grow algorithmic-invariant coverage without operator labor.
Items B3 / B4 / B5 / B6 / B7 / T3 / T5 / T7 / G3–G8 each add a
specific incremental capability that can be deferred.

### Recommended interleaving

Within a single release the natural interleaving alternates one
backend phase with its matching GUI phase:

1. B0 → tests pass, no GUI change
2. B1 → G2 (noise controls)
3. B2 → G0 + G1 (3-peer audit + instrument selector)
4. T1 → G8 (catalog browser)
5. T2, T4, T6 (test infrastructure; no GUI change)
6. B4 → G4 (saturation)
7. B5 → G5 (PSF preview)
8. B6 → G6 (stray light)
9. B7 → G7 (irregular bodies)
10. T3, T5, T7 (sweeps, bootstrap, validation)

B3 / G3 (smear) are deferred pending `psfmodel`'s smear support (issue #151);
B8 (diffraction) is deferred as a lowest-priority future enhancement (issue
#152).

---

## 8. Out of scope (and why)

These items sound like they belong in a sim-improvement plan but
explicitly do *not* move the calibration needle, so they are
deferred indefinitely:

- **Hapke / Lommel-Seeliger / Akimov photometric models.** The
  navigator works on gradients and silhouettes; replacing Lambert
  with a more accurate photometric function gives better-looking
  pictures but doesn't shift any diagnostic the calibration depends
  on.  Polish, not load-bearing.
- **Fixed-pattern noise / detector banding.** Captured by the
  noise-sigma estimator as elevated noise; the calibration
  response is essentially the same as B1's contribution.
- **JPEG / lossy compression artifacts.** Real archives don't
  ship JPEGs; this is moot.
- **Color rendering.** The navigator ignores color (per-camera
  `mag_offset` already absorbs band conversion).  Color is
  presentation polish.
- **Multi-frame trajectory rendering.** Render a sequence of frames
  along an encounter trajectory.  Useful for testing batch behavior
  (worker pool, cloud_tasks queue dynamics) but not for per-image
  calibration.  Deferred to a separate plan if the cloud-tasks layer
  needs it.
- **Atmospheric body navigation (TitanNav stub).** Tracked in
  `AUTONAV_PLAN.md` Part 13b §3 and won't be exercised by sim until
  the navigator side is implemented.

---

## 9. Acceptance criteria

A phase is considered done when **all** of these are true:

1. Code changes land in the project's standard format
   (`ruff check`, `ruff format`, `mypy`, `sphinx-build -W` all
   clean).
2. New unit tests cover the new functionality with the same density
   as adjacent code (target ≥ 90% line coverage on new files).
3. Backend phases: `tests/nav/sim/test_sim_*` covers the
   determinism guarantee, the parameter ranges, and at least one
   "renders without exception across realistic input ranges" smoke
   test.
4. GUI phases: at least one UI smoke test that constructs the
   `CreateSimulatedImageModel` and exercises the new control's
   change-handler.  GUI tests are inherently fragile; the bar is
   "the control wires to the right backend parameter", not "the
   pixmap looks right".
5. Test phases: the harness runs as `python -m
   tests.integration.<module>` from a clean checkout and emits the
   expected JSON / Markdown / per-test summary.
6. Documentation:
   - Backend phases update
     `tests/integration/sim_scenes/README.txt` (after T1) with the
     new YAML field if applicable.
   - GUI phases update screenshots in the developer guide if one
     exists; otherwise leave a one-line note in the GUI's docstring.
   - This `SIM_IMPROVEMENT_PLAN.md` file gets a checkmark next to
     the completed phase in section 7.
7. The phase's output is consumable by the next phase that depends
   on it (verify by spot-running one of the dependent phase's
   acceptance tests against the new state).

---

## 10. Concrete file additions / changes summary

For a quick scan of what this plan touches:

### New files
```
tests/integration/sim_scenes/                            (T1)
tests/integration/sim_scenes/README.txt                  (T1)
tests/integration/sim_scenes/<class>/<name>.yaml         (T1+)
tests/integration/sim_baselines/                         (T2)
tests/integration/sim_baselines/<name>.json              (T2+)
tests/integration/sim_sweeps/                            (T3)
tests/integration/sim_sweeps/<name>.yaml                 (T3+)
tests/integration/sim_sweep.py                           (T3: schema + runner core)
tests/integration/sim_scene.py                           (T1)
tests/integration/sim_sweep_runner.py                    (T3)
tests/integration/sim_alpha_bootstrap.py                 (T5)
tests/integration/sim_vs_real_diagnostics.py             (T6)
tests/integration/sim_vs_real_report.md                  (T6 output)
tests/integration/test_sim_scenes.py                     (T1)
tests/integration/test_sim_baselines.py                  (T2)
tests/integration/test_sim_sweeps.py                     (T3, T7)
tests/integration/test_sim_algorithmic_invariants.py     (T4)
tests/integration/update_sim_baselines.py                (T2)
src/nav/nav_model/stars/nav_model_stars_simulated.py     (T4)
tests/nav/nav_model/stars/test_nav_model_stars_simulated.py (T4)
tests/nav/sim/test_sim_rotation.py                       (T4: camera roll)
tests/nav/nav_model/test_nav_model_rings_simulated.py    (T4: RING_EDGE)
tests/nav/sim/test_sim_determinism.py                    (B0)
tests/nav/sim/test_sim_noise.py                          (B1)
tests/nav/sim/test_sim_instrument_coupling.py            (B2)
tests/nav/sim/test_sim_smear.py                          (B3)
tests/nav/sim/test_sim_saturation.py                     (B4)
tests/nav/sim/test_sim_psf.py                            (B5)
tests/nav/sim/test_sim_stray_light.py                    (B6)
tests/nav/sim/test_sim_irregular_body.py                 (B7)
src/nav/sim/sim_body_polyhedral.py                       (B7)
tests/integration/test_sim_irregular_pose.py             (B7: pose scenario 3)
tests/integration/sim_sweeps/irregularity_shape_mismatch.yaml  (B7: scenario 2)
tests/integration/sim_sweeps/pose_disagreement.yaml      (B7: scenario 3)
(no src/nav/sim/shape_meshes/ -- mesh source is procedural; section 12.3)
```

### Modified files
```
src/nav/sim/render.py              (B0, B1, B4, B5, B6, B7, T4; B3 deferred #151, B8 deferred #152)
src/nav/nav_model/__init__.py      (T4: register NavModelStarsSimulated)
src/nav/nav_model/stars/__init__.py  (T4: register NavModelStarsSimulated)
src/nav/sim/sim_body.py            (B0, B2, B7)
src/nav/sim/sim_ring.py            (B0)
src/nav/obs/obs_inst_sim.py        (B2, T4: fit_camera_rotation override)
src/nav/dataset/dataset_sim.py     (B2)
src/nav/config_files/config_440_sim.yaml  (B1, B2, B6)
src/main/nav_create_simulated_image.py  (G0, G1, G2, G3, G4, G5, G6, G7, G8)
```

### No changes
```
The operator-curated real-image library at
tests/integration/image_library/ — by cardinal principle 3.1, the
sim work does not modify or replace the real library.  The
calibration target stays anchored to real data.
```

---

## 11. Discussion notes preserved for context

This section captures the reasoning that led to the plan above,
including dead-ends and explicit rejections, so a future reader can
see why specific decisions were made and which alternatives were
considered.

### 11.1 Why not just calibrate against sim?

The thought experiment that motivated this plan was: "the sim can
produce hundreds of controlled images; why curate real ones at
all?"  The rejection rests on three observations:

1. The α coefficients in the confidence formula encode "what
   diagnostic value `x` corresponds to what tier".  If sim
   diagnostics distribute differently from real (and they do —
   ellipsoidal bodies, perfect-Gaussian PSFs, additive Gaussian
   noise, no irregularity, no scattered light), α tuned against sim
   gives the wrong answer on real.
2. Verifying that "α tuned against sim works on real" requires the
   real-image library anyway.  So sim doesn't replace the labor,
   it just front-loads it.
3. The tier labels themselves (`high` / `medium` / `low` / `failed`)
   are judgments about real-world performance.  What does `high`
   mean on a sim'd Hyperion?  The sim's idealized-Hyperion-as-an-
   ellipsoid would always navigate `high`; the real-Hyperion is
   `low`.  Tier targets assigned to sim scenes don't reflect the
   real-world performance the calibration is supposed to map onto.

The compromise is the augment-not-substitute architecture in this
plan.  Sim covers what it's good at (controlled sweeps,
algorithmic invariants, regression coverage); the real library
remains the calibration target.

### 11.2 Why a YAML scene catalog instead of inline Python?

Today's sim tests construct scenes inline.  The reasons to move to
YAML:

- **Reviewability.**  A reviewer can read a YAML scene and
  understand it without reading Python.  Inline-Python scenes hide
  parameters in helper-function defaults.
- **GUI/test/library parity.**  The same YAML file can be loaded by
  the GUI for visualization, by the test harness for assertions,
  and by the regression layer for baselining.  Inline Python forces
  the GUI to re-implement the scene by hand.
- **Catalog growth.**  The real-image library's directory-as-registry
  pattern works well; the sim catalog mirrors it.  Adding a scene
  is one YAML file.
- **Versioning.**  YAML scenes are diffable in git.  Inline Python
  scene constructors couple to refactors of the helper functions.

### 11.3 Why instrument-coupling (B2) is so high in the order

The original draft had B2 much later — "nice-to-have polish".  On
review, every later phase that adds physics needs to look up the
right per-instrument value (noise sigma, PSF, full-well, marker
value, signal scale).  Without B2 each later phase grows a
duplicate sim-side knob, then a second pass has to reconcile the
two.  Doing B2 second (after determinism) means the later phases
each pull from one place.

### 11.4 Why determinism (B0) is the first phase

Anything that adds randomness later (cosmic rays in B1, stray-light
phase noise) silently changes the output of every previously-
existing test.  If the determinism contract isn't established
first, B0 becomes a forensic exercise: "this baseline drifted —
why?".  Doing it first makes B1 and beyond auditable: a baseline
change is either explainable by a documented physics addition or
it's a bug.

### 11.5 Why the GUI work is mandatory rather than optional

The GUI exists because operators want to dial parameters and see
the result live.  Backend additions that the GUI can't show are
operator-invisible.  If we land B1 (realistic noise) without G2,
the only way an operator sees the new noise model is to write a
Python script — which they won't.  Bundling the GUI work with the
backend work means each release lands a *complete* operator-visible
capability, not a backend half-feature.

The cost of this discipline is real: every backend PR also touches
the GUI file, which is large and PyQt-heavy.  The alternative —
stockpile backend PRs and do the GUI later — looks attractive
short-term but produces a GUI file that's permanently out of sync.

### 11.6 Why the irregular-body phase (B7) is so far down

Three reasons:

1. **`oops` will not gain DSK support in the foreseeable future**, so
   the only path is the standalone polyhedral renderer -- doable but
   significant: ~500 LOC of mesh loading, projection, silhouette
   extraction, anti-aliasing, plus sourcing the body's pose in space
   at the scene midtime to orient the mesh.
2. **The `phase_irregularity_factor` term it would exercise is
   already in place in the navigator** (Phase 10 §F's BLOB work).
   It's calibration-relevant but the term defaults to 0 in
   sim-ellipsoid scenes, which means existing calibration work isn't
   blocked on B7 — it just doesn't get to *test* the irregular path
   on sim until B7 lands.
3. **One real Cassini Phoebe / Hyperion close-flyby image in the
   real-image library would cover the irregular-body case for
   calibration purposes.**  B7 is the right answer when the gap
   between "what the real library covers" and "what we want to
   sweep with sim" specifically includes irregular-body
   sensitivity, which is a Phase 11+ concern.

If irregular-body sensitivity becomes a hot calibration question
sooner, B7 can be promoted up the order.

### 11.7 Why diffraction spikes (B8) are deferred, not built

The supported cameras have no diffraction spikes.  The Cassini
telescope has no secondary-mirror support vanes, so its stars carry no
cross-shaped diffraction pattern at any brightness.  There is nothing
to model today, so B8 is parked as a lowest-priority future enhancement
(issue #152): build it only if the navigation system is ever extended
to support an instrument whose optics genuinely produce spikes.

### 11.8 What about the existing simulator users?

`nav_create_simulated_image` today is used (rarely) for one-off
illustrative renders.  The plan's GUI changes are all additive —
existing controls stay where they are, new tabs / panels appear
alongside.  The biggest behavior change for existing users is the
instrument selector (G1), which defaults to `(generic)` so existing
workflows are untouched unless the operator opts in.

### 11.9 Estimated effort

Rough order-of-magnitude scoping (single full-time developer,
including tests + docs + reviews):

| Phase | Estimated effort |
|---|---|
| B0 | 1 day (done) |
| B1 | 3 days (done) |
| B2 | 4 days (done) |
| B3 | deferred (psfmodel smear; #151) |
| B4 | 1 day (done) |
| B5 | 2 days (done) |
| B6 | 1 day (done) |
| B7 | 2 weeks |
| B8 | deferred (future, #152) |
| G0 | 1 day |
| G1 | 1 day |
| G2 | 2 days |
| G3 | 1 day (deferred with B3; #151) |
| G4 | 1 day |
| G5 | 2 days |
| G6 | 1 day |
| G7 | 1 day |
| G8 | 3 days |
| T1 | 2 days |
| T2 | 2 days |
| T3 | 3 days |
| T4 | 3 days |
| T5 | 2 days (deferred; #153) |
| T6 | 3 days (deferred; #153) |
| T7 | 2 days (deferred; #153) |

**Minimum viable cut total**: ~16 days.
**Full plan total**: ~50 days, dominated by B7 (irregular-body
mesh renderer).  Skipping B7 brings the full plan to ~36 days.

These are rough — the actual numbers depend heavily on how
ergonomic the existing code is to extend, how hard `oops`
integration turns out to be for B2, and how much new GUI design
work G2 / G8 require.  Use as a relative-ranking guide, not a
schedule commitment.

---

## 12. Open questions — operator decisions before starting

Items the implementer should get explicit answers on before
starting:

1. **Cardinal-principle confirmation.** Does the
   "augment-not-substitute" stance (section 3.1) reflect the
   project's intent?  If the operator wants sim to be the primary
   calibration target, the plan needs to be reframed entirely.
2. **Per-instrument config split** (B2).  *Resolved.*  The two options
   (auto-sync vs. isolate) are reconciled per scene rather than globally: a
   scene inherits live from the named instrument by default (auto-sync), and an
   optional ``instrument_config`` mapping deep-merged over the resolved block
   pins individual keys (or all of them, with the ``generic`` block) to isolate
   the scene from instrument-config drift.  See section 0.3 and
   `dev_guide_observations`.
3. **Polyhedral mesh source for B7.**  *Resolved (neither).*  The procedural
   generator (``make_irregular_mesh``: seed + lumpiness + pose) is the sim's mesh
   source; named ``.obj`` meshes are not bundled and not fetched.  A parameterized
   irregular body spans the irregularity axis continuously, which the controlled
   sensitivity / regression role needs, and real irregular-body accuracy is
   calibrated against the real Cassini Hyperion/Phoebe images in the operator
   library (principle 3.1), so a fixed bundled mesh would not move the calibration
   needle while adding ~5 MB of repo size or network-at-test-time.  The renderer
   accepts any ``Mesh``, so a ``shape_model: named_mesh`` + ``mesh_file`` loader is
   a documented future hook, left unbuilt until a specific body is proven
   necessary.  See the B7 section and section 0.3.
4. **Sim baseline policy** (T2).  Should sim baselines be
   regenerated automatically when the sim's noise model changes
   (B1), or should each backend change explicitly call out which
   sim baselines need re-blessing?  The latter is more
   conservative; the former is faster.
5. **Sweep failure policy** (T3, T7).  Should sweep-invariant
   failures CI-block immediately, or be advisory until the
   real-data calibration is also stable?  The plan assumes
   advisory-then-promote-to-blocking; operator can flip this.

When these are decided, the corresponding sections of this plan
should be updated with the operator's chosen direction so the
implementer has a clear contract.
