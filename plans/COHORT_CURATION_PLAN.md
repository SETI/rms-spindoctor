# Cohort Curation Plan: metadata-driven image discovery for validation and calibration

**Audience.** A fresh AI session with no prior context, working in this repo
with an operator (rfrench) available only for brief review votes. Read
`/seti/newnav/CLAUDE.md` first, then `plans/VALIDATION_AND_CALIBRATION_PLAN.md`
(the methodology this plan feeds) and
`docs/dev_guide/dev_guide_image_library.rst` (the sidecar schema, tier
semantics, and baseline mechanics). The scene-class budget and per-class
selection guide are in the appendix at the end of this file.

**Goal.** Populate every image cohort the validation and calibration program
needs — the 47-image curated first stage (#172), its WS-3 growth to >=20 per
instrument / >=120 total, and the WS-1 agreement-study cohorts — by
**automated search over the published PDS geometry metadata**, so the operator
does nothing but look at overlay PNGs and vote yes/no. Arbitrary or convenient
images are not acceptable: every cohort must span a **wide variety** of moons,
rings, planets, illumination, viewing geometry, resolution, and filters
(quotas below), because the validation math bins by exactly those axes and a
narrow cohort silently invalidates it.

---

## Status board (2026-09-21)

The five-stage workflow of section 4 is built and has been run end to end
for review batches 001 through 006. What is unfinished is the curation
itself: the library covers 16 of the 17 declared scene classes but has
not reached the first-stage per-class minima, and the WS-3 growth target
is untouched for three of the four instruments.

| Stage | State | What is left |
|---|---|---|
| **A — query** | Done, reusable | Nothing; `scan_stage_a.py` covers every class, `scatter_prescan.py` covers the pixel-level `scattered_light` gate, and `spice_coverage.json` filters out-of-coverage epochs |
| **B — triage** | Done, reusable | Nothing structural; the ten-glob-per-frame results-tree scan is the pain the results index does not yet serve (#465) |
| **C — review** | Done, reusable | Batch 006's seven manual-nav follow-ups are still queued (`_work/cohort_curation/batch_006_followups.yaml`), with three class reassignments to apply during that pass |
| **D — sidecars** | Partly done | 75 sidecars committed; `faint_stars` is empty and `body_mostly_offscreen` holds 1 of its 4 (#172), and the per-class minima are not enforced by any test (#240) |
| **E — baselines and consumption** | Partly done | Baselines exist and are re-ratcheted per change, but eight of the 75 frames disagree with their sidecars locally (#288); no consumer study has been run (#230, #225, #355) |

**Library state.** `tests/integration/image_library/images/` holds 75
sidecars. Per class against the first-stage budget in the appendix:
`star_dominated` 8, `body_irregular` 8, `negative_cases` 7,
`stars_plus_body` 6, `body_full_fov` 5, `ring_only_flat` 5,
`ring_plus_body` 12, `scattered_light` 4, `high_phase_terminator` 3,
`multi_body` 3, `ring_only_curved` 3, `body_partial_overflow` 3,
`two_bright_stars_no_body` 3, `below_resolution_body` 2,
`one_bright_star_no_body` 2, `body_mostly_offscreen` 1, `faint_stars` 0.
Two classes are therefore below their minima and the first-stage budget
(#172) is not met, even though the total exceeds 47.

**Mission spread.** COISS 62 frames (58 Saturn, 4 Jupiter), GOSSI 8,
VGISS 3, NHLORRI 2. The WS-3 target of >=20 per instrument and >=120
total (#235) is met by no instrument but Cassini.

**Staging areas outside the library.**
`tests/integration/image_library/_uncurated/` holds two `body_irregular`
frames held back from the tree, and `util/titan_cohort/nominations/`
holds the six Titan frames awaiting the class decision (#407).

---

## 1. Data sources (all local, no network needed)

**Canonical environment: `source /seti/newnav/setup.sh`.** It activates the
project venv (`venv/` at the repo root — not `.venv`, which is stale) and
exports every variable below. Do not use the `/mnt/ganymede/SPICE` or
`/mnt/ganymede/UCAC4` shares for navigation runs: the SPICE share has no
`SPICE.db` (which spicedb requires) and the UCAC4 mountpoint is empty.

| Resource | Path |
|---|---|
| PDS3 holdings (images) | `/mnt/ganymede/PDS/holdings` (`PDS3_HOLDINGS_DIR`) |
| Geometry metadata tables | `/mnt/ganymede/PDS/holdings/metadata/<VOLSET>/<VOLUME>/` |
| OOPS resources (incl. SPICE) | `/home/rfrench/DS/Shared/OOPS-Resources` (`OOPS_RESOURCES`; `SPICE.db` lives at `$OOPS_RESOURCES/SPICE/SPICE.db`, no `SPICE_PATH` needed) |
| UCAC4 star catalog | `/data/external-data/star-catalogs/UCAC4` (`UCAC4_PATH`) |
| YBSC star catalog | `/data/external-data/star-catalogs/YBSC` (`YBSC_PATH`) |
| Nav results (default root) | `/data/nav-offset-results` (`NAV_RESULTS_ROOT`) |

**Tooling:** the Stage A through D automation lives in
`util/cohort_curation/`: `scan_stage_a.py` (query), `scatter_prescan.py`
(the pixel-level gradient score for `scattered_light`),
`triage_stage_b.py` (autonomous-pipeline triage, with
`rescue_config.yaml` relaxing the uncalibrated gates so a navigable
scene yields an offset to look at), `build_review_batch.py` and
`star_check_pngs.py` (review PNGs and `votes.yaml`), and
`build_sidecars.py` (sidecars from the voted file), plus the support
files `pdsmeta.py`, `body_radii.json` and `spice_coverage.json`. See its
README for usage and the metadata-format gotchas.
Generated artifacts — candidate manifests, triage results, review batches,
votes — go under `_work/`, which is gitignored; only the tooling is
tracked.

Volume sets by instrument: `COISS_1xxx` (Cassini Jupiter leg) and `COISS_2xxx`
(Cassini Saturn) for COISS; `VGISS_5xxx`/`6xxx`/`7xxx`/`8xxx` (Voyager
Jupiter/Saturn/Uranus/Neptune); `GO_0xxx` (Galileo SSI); `NHxxLO_xxxx`
(New Horizons LORRI). Use the unversioned directories (e.g. `COISS_2xxx`, not
`COISS_2xxx_v1.0`).

Each volume directory carries five table pairs (`.lbl` describes columns,
`.tab`/`.csv` is the data). **Always parse the `.lbl` to get authoritative
column positions per volume set** — layouts differ between missions. Verified
for COISS_2xxx:

- `*_moon_summary.tab` — one row per (image, moon). Key columns (1-based):
  2 `FILE_SPECIFICATION_NAME`, 4 `TARGET_NAME`, 5-10 planetocentric/graphic
  lat and IAU lon min/max, 15-18 surface resolution min/max, 21-22 phase
  angle min/max, 23-26 incidence/emission min/max, 33 `CENTER_RESOLUTION`
  (km/px), 34 `CENTER_DISTANCE`, 35 `CENTER_PHASE_ANGLE`. **`-999` means the
  quantity is undefined for that image (typically: the body is in the FOV
  inventory but not resolved/on-disk)** — filter on it deliberately: lat/lon
  columns at -999 with a valid center distance means an unresolved point
  target; valid lat/lon ranges mean resolved surface in frame.
- `*_ring_summary.tab` — one row per image with rings in FOV. Key columns:
  RA/dec min/max (4-7), `MINIMUM/MAXIMUM_RING_RADIUS` (8-9), radial
  resolution, ring longitude min/max, phase/incidence/emission, and the
  solar/observer **ring opening angles** (last columns).
- `*_saturn_summary.tab` (or the mission's planet summary) — same shape as
  moon rows, for the planet disc.
- `*_inventory.csv` — per image, the full list of bodies in the FOV: the
  fastest way to require or exclude bodies (multi-body scenes, "no body"
  star frames).
- `*_index.tab` — per image: exposure, filters, camera, pointing RA/dec.
  Use it for filter/exposure diversity and for star-frame pointing.

Image path construction: `FILE_SPECIFICATION_NAME` gives
`data/.../N1454725799_1.LBL`; the raw image swaps `.LBL` for `.IMG` under
`volumes/<VOLSET>/<VOLUME>/`, and the Cassini calibrated variant lives under
`calibrated/<VOLSET>/<VOLUME>/...<name>_CALIB.IMG`. Sidecar `image_url` uses
the `pds3://` form (see any existing sidecar in
`tests/integration/image_library/images/`). Voyager must use the geometrically
corrected (`GEOMED`/calibrated) products — see the mission-specific hints
in the appendix.

Body radii for apparent-size computation: use the values already in
`src/spindoctor/config_files/config_220_body_shape.yaml` and oops, not a
hand-typed table. Apparent diameter in pixels = 2 * R_body /
`CENTER_RESOLUTION`.

Ring edge radii for "does an edge cross this frame": read the per-planet ring
catalogs `src/spindoctor/config_files/config_3N0_*_rings.yaml` — do not
hardcode edge radii in the search script.

**Proven feasible:** a 20-line Python scan of 35 COISS volumes' moon summaries
found 2,223 `high_phase_terminator` candidates (phase 110-155 deg, apparent
diameter 150-800 px) in seconds. The same pattern covers ~15 of the 17 scene
classes; the star classes additionally need a UCAC4 count at the index-table
pointing (predicted detectable stars given the instrument magnitude limit —
reuse `spindoctor.nav_model.stars` catalog plumbing rather than reimplementing).

---

## 2. Cohorts to build and what each feeds

One physical library (`tests/integration/image_library/`), tagged per image;
cohorts are queries over the tags, not separate directories beyond the
existing scene classes. Consult VALIDATION_AND_CALIBRATION_PLAN WS-0/WS-1/WS-3
for the full rationale.

| Cohort | Definition (metadata query sketch) | Feeds |
|---|---|---|
| **Scene-class library, 47 images** | the 17 classes in the appendix budget table, found per class as in section 3 | #172 regression seed, diagnostics for the confidence recalibration (#230), #174 baselines |
| **WS-3 growth** | same classes continued to >=20/instrument, >=120 total | WS-1 statistics, WS-4 CI tiers |
| **Route 1: intra-body** | one resolved moon, apparent diameter 150-900 px, full limb or >=30% arc, phase < 90 for limb+disc pairs and > 90 to add terminator; **restricted to round, photometrically bland bodies** per the config_220 shape gate (`ellipsoid_rms_residual_km`, `crater_scale_km`, `albedo_variation`) | WS-1 Route 1 (technique pairs on one body; SPICE cancels) |
| **Route 2: body + ring** | inventory has >=1 moon AND ring_summary row with a catalog edge radius inside [MIN,MAX]_RING_RADIUS; the bulk cohort — collect widely | WS-1 Route 2 (ring-radial axis) |
| **Route 3: star tie-points** | star count >= 3 at pointing AND (resolved moon OR ring edge) in frame; scarce — sweep exhaustively, keep every hit | WS-1 Route 3 (absolute attitude, ephemeris probe) |
| **Multi-body** | inventory has >=2 resolved moons (each diameter > 50 px), non-occluding | WS-1 orthogonal axis, inter-moon ephemeris |
| **Over-determined** | >=2 resolved moons AND a ring edge in one frame (rarest; sweep everything, keep every hit) | WS-1 closure test — the only real-photon assumption check |
| **WS-1b sequences** | 5-20 consecutive frames of one body with overlapping footprints (sort moon_summary by time within a volume; same target, overlapping lat/lon boxes), no second fiducial needed | WS-1b reprojection consistency |
| **WS-17 star fields** | >=20 predicted stars across the FOV, no body/ring (empty inventory); Cassini (and LORRI if counts allow) only | WS-17 distortion validation via plate-solve residuals |
| **WS-2 realism set** | any representative real frames per instrument spanning exposure/filter/noise regimes (no geometry requirement — it is a statistics match) | WS-2 sim-realism validation |

## 3. Diversity requirements (mandatory, not aspirational)

The WS-1 solve bins by resolution / phase / lit-fraction /
limb-orientation-vs-radial, assumes within-bin stationarity, and needs
populated bins on every axis. A cohort of thirty Enceladus frames from one
flyby is worthless. Enforce at query time:

- **Targets:** every resolved moon the archive offers per mission (Saturn
  system: at minimum Mimas, Enceladus, Tethys, Dione, Rhea, Iapetus; plus
  irregulars Phoebe/Hyperion/Janus/Epimetheus for the shape-contaminated
  bucket — kept separate, never in Route 1). Jupiter: Galilean four. Include
  the planet discs. Cover **all four ring systems** where the mission saw
  them (Saturn primary; Jupiter/Uranus/Neptune rings are Phase-2 models —
  collect candidates now, navigate later).
- **Illumination:** phase bins at least {<30, 30-60, 60-90, 90-120, >120}
  deg with several images per populated bin per target class; both
  low-incidence and terminator-dominated lighting.
- **Viewing geometry:** for rings, spread ring opening angle (near edge-on
  through wide-open) and ring longitude; for moons, spread
  limb-orientation-vs-ring-radial (compute from pointing + geometry at query
  time or post-hoc) because WS-1 bins on it.
- **Resolution:** at least three decades (e.g. <1 km/px close flybys,
  1-10 km/px mid-range, >10 km/px distant) per mission where available.
- **Filters/exposure:** do not let one filter dominate; include the CALIB and
  RAW Cassini variants per the appendix saturation policy; include long and
  short exposures for the star classes.
- **Time:** spread over the mission (different SPK/CK eras), not one
  encounter.
- **Stratified sampling:** when a query returns thousands (it will — 2,223
  hits for one class in 35 volumes), sample the strata (target x phase-bin x
  resolution-decade x filter), never take the top-N rows. Seed any random
  sampling and record the seed.
- **Provenance:** every sidecar's `notes` records the query criteria and
  metadata values that selected it, so cohort composition is reproducible.

## 4. Workflow: minimal operator interaction

The operator's entire job is: look at a PNG, vote yes/no, optionally comment.
Target under 30 seconds per image, in batches of up to 100 (operator
preference, 2026-07-08). Everything else is
automated.

**Stage A — query.** Scripted scan of the metadata tables per cohort
(section 2), stratified sampling (section 3), producing a candidate manifest
(CSV/YAML: image id, path, class, selecting metadata values).

**Stage B — automated triage (no operator).** For each candidate, run the
autonomous pipeline locally (`sd_offset` with the local mounts; ~35 s/frame,
so batch overnight if needed). Auto-drop, without operator review: frames the
feasibility gates reject, majority-missing-data frames, saturated-bloom
frames, and frames whose scene class the actual geometry contradicts. Keep
the per-frame `_metadata.json` and the summary PNG. A candidate is only
promoted to review when the pipeline produced a proposed offset AND the
overlay looks self-consistent by machine checks (fit residuals, technique
agreement) — or when the frame is scarce (Route 3 / over-determined), in
which case promote it even on failure and flag it for manual navigation.

**Stage C — operator review.** Produce one review batch at a time:
`_work/cohort_review/batch_NNN/` containing, per image, a single composite
PNG (red image / green model overlay AT the proposed offset — the same
rendering the manual-nav "Save as Library Entry" writes, plus the proposed
`(dv, du)`, class, and any triage warnings burned into the image margin),
plus a pre-filled `votes.yaml` listing every image with `vote: null` and an
empty `comment:`. The operator opens the PNGs, edits `votes.yaml` to `y`/`n`
(+ optional comment), and hands it back — or simply votes in chat
("1-14 y; 15 n, limb misaligned"). Never ask the operator to run tools, drag
overlays, or fill schemas; that is the AI's job. Reserve `sd_offset --manual`
for the scarce-cohort frames flagged in Stage B, and queue them so the
operator does all manual work in one sitting.

**Stage D — sidecar generation (no operator).** `build_sidecars.py`
consumes the voted `votes.yaml` plus the Stage B triage report. For each
`y`: write the
sidecar into the right scene-class directory with `ground_truth` from the
reviewed offset (`source: operator_verified`, the vote date, the reviewed
PNG kept beside the YAML), auto-filled `expected.*` per the sidecar rubric in
`docs/dev_guide/dev_guide_image_library.rst` (conservative tier — `medium`
when unsure; tier labels are plausibility cross-checks, never calibration fit
targets), and the selection provenance in `notes`. Run the structural suite —
`pytest tests/integration/test_image_library.py`, which needs no holdings
access and no marker filter — until clean; then, with `PDS3_HOLDINGS_DIR`
set, spot-run each added frame with
`pytest tests/integration/test_autonomous_nav.py -m "" -k <id>`. A frame
whose class or ground truth is unsettled goes to
`tests/integration/image_library/_uncurated/<class>/` instead, where it
carries no regression weight. Submit one PR per review batch (operator preference,
2026-07: reviewer cost is dominated by `ground_truth` spot-checks, and one
batch per PR keeps the vote-to-merge mapping clean).

**Stage E — baselines and consumption.** After sidecar PRs merge, seed
regression baselines (`python -m tests.integration.update_baselines
--image-id <id>`, or `--all` for a whole re-ratchet), then
hand off per consumer: calibration diagnostics collection for the
real-anchored confidence recalibration (#230 — reliability diagrams
against measured error anchors, never tier-midpoint fitting),
agreement-study runs per WS-1's harness plan (#225),
plate solves on the star-field cohort for the per-camera Voyager distortion
split (#355). CI stays tiered per WS-4's
"Library consumers and CI tiers" note: the full library and all offline
analyses never run per-PR.

## 5. Order of work

1. **In progress.** Fill the empty scene classes of the first-stage budget
   first (per-class state: compare the appendix budget table against
   `tests/integration/image_library/images/*/`), one review batch. Two
   classes remain: `faint_stars` (0 of 2) and `body_mostly_offscreen`
   (1 of 4). Clear the batch-006 manual-nav queue in the same sitting —
   seven frames, three of which change class on the way in.
2. **Not started.** Top up all classes to the per-class minima; verify
   mission spread (>=1 image from each of the four missions; >=1 Cassini
   `_CALIB`). All four missions and a `_CALIB` frame are present, so this
   step is now only the per-class top-up.
3. **Not started.** Sweep for the scarce cohorts (Route 3 star tie-points,
   over-determined frames) across ALL volumes — these are kept whenever
   found, and they gate the most valuable science (absolute attitude,
   closure test).
4. **Not started.** Build Route 1 / Route 2 / multi-body / WS-1b cohorts to
   the WS-3 target (>=20/instrument, >=120 total), stratified per section 3.
   Voyager, Galileo and LORRI carry the whole gap.
5. **Not started.** WS-17 star fields and the WS-2 realism set (no
   ground-truth votes needed for WS-2 — it is a distributional match, so it
   can be fully automated).

A sixth stream runs alongside rather than after: the library that exists is
red. Ten of the 75 frames disagree with their sidecars in the local
integration environment (#288), which makes "no new failures against `main`"
the only gate a navigation-affecting branch can clear, and several of the
reds are curation decisions rather than code defects (#346, #350, #483,
#563). Reconciling them is curation work and belongs to this plan.

Checkpoint with the operator between numbered steps, not within them.

## 6. Known traps

- `-999` sentinels in every summary table; never treat them as values.
- A body in `inventory.csv` may be unresolved or behind the planet; always
  cross-check the moon_summary row before classifying.
- Metadata tables describe the PREDICTED geometry from archived SPICE — the
  same kernels the navigator corrects. Fine for finding scenes; never treat
  metadata geometry as ground truth for offsets.
- Scene classes are decided by the expected PRIMARY technique
  (appendix per-class table), not by what happens to be in frame;
  when a frame straddles classes, pick the class that exercises the primary
  technique and record the judgment.
- Do not let the autonomous proposal bias ground truth on frames where it is
  systematically wrong (the known ~0.1 px limb bias, WS-10): the vote
  verifies pixel-level alignment, and sub-pixel systematics are exactly what
  WS-1/WS-2 measure — record `offset_uncertainty_px` honestly (1.0 px
  default, 2.0 px for soft features), never 0.1 px.
- Voyager: only geometrically corrected products; Galileo/Voyager carry
  camera-rotation fitting (slow) — budget triage time accordingly.

## 7. Open issues this plan carries

| Issue | What it is | Where it lands |
|---|---|---|
| #172 | First-stage budget of 47 images across 17 scene classes | Order of work step 1 |
| #235 | Growth to >=20 per instrument / >=120 total | Order of work steps 3-4 |
| #240 | Coverage-matrix and per-class-minimum invariants in the structural test | Appendix budget table |
| #288 | Ten of the 75 frames disagree with their sidecars locally | Order of work, the sixth stream |
| #346 | Three library frames lock confidently onto the wrong ring feature | Same |
| #350 | Two resolved-body frames miss offset tolerance by ~2 px | Same |
| #483 | Re-ratchet library pins moved by the shift-equivariance fix | Same |
| #563 | A Galileo frame's offset moved 5.6 px, with two reader tests | Same |
| #239 | Decision: sub-5 px bodies — relaxed disc or expected-failure curation | Class assignment for `below_resolution_body` / `negative_cases` |
| #319 | Opposed-ansae ring geometry has no library coverage | A wide-field Cassini WAC both-ansae sweep, Stage A |
| #399 | Voyager ISS validation cohort for the haze-symmetry fit | Titan curation |
| #405 | Titan library growth through this pipeline | Titan curation, after #407 |
| #407 | Operator ratification of the Titan decisions and the staged nominations | The `titan_haze` class decision |
| #340 | `library_crosscheck` records a yes/no primary-technique flag, not the winner | Stage E diagnostics |
| #465 | Widen the results-index schema far enough to serve Stage B triage | Stage B |
| #612 | `library_entry` resolves the PDS3 holdings root in its own order | Stage C/D save path |
| #229 | Fast per-PR real-image tier plus a scheduled full-library run | Stage E, CI tiers |

Consumers waiting on the cohorts: #230 (confidence recalibration), #174
(autonomous-nav tests and baselines), #225 and #358 (the agreement study),
#355 (Voyager distortion defaults).

---

## Appendix: scene-class budget, selection guide, and mission hints

Distilled from the retired first-stage playbook
(`plans/archive/PHASE10_CURATION_2026-07-12.md`); the sidecar schema, field
rubric, tier semantics, and baseline workflow live in
`docs/dev_guide/dev_guide_image_library.rst`.

### First-stage budget (47 images across 17 scene classes)

| Scene class (directory name)   | Min images | What it exercises                                              |
|--------------------------------|-----------:|----------------------------------------------------------------|
| `star_dominated`               |          4 | `StarFieldFromCatalogNav` primary; many catalog stars, no body |
| `body_full_fov`                |          3 | `BodyDiscCorrelateNav` primary; regular body fills FOV         |
| `body_partial_overflow`        |          3 | `BodyDiscCorrelateNav` (gradient mode); body 70-90% in frame   |
| `body_mostly_offscreen`        |          4 | `BodyLimbNav` primary; only a limb arc in FOV                  |
| `body_irregular`               |          3 | `BodyBlobNav` primary; close-range irregular body              |
| `multi_body`                   |          3 | Multi-feature joint fit; >=2 separable bodies                  |
| `ring_only_curved`             |          3 | `RingEdgeNav` full 2-D                                         |
| `ring_only_flat`               |          3 | `RingEdgeNav` rank-1 (single-axis) result                      |
| `ring_plus_body`               |          3 | Ensemble: rings + >=1 moon                                     |
| `stars_plus_body`              |          3 | Ensemble: body + >=3 visible catalog stars                     |
| `one_bright_star_no_body`      |          2 | `StarUniqueMatchNav` 1-star primary                            |
| `two_bright_stars_no_body`     |          2 | `StarUniqueMatchNav` 2-star primary                            |
| `faint_stars`                  |          2 | Predicted SNR < 3 for every catalog star (Galileo / Voyager)   |
| `scattered_light`              |          2 | Galileo / Voyager stray-light gradient -> DoG bandpass         |
| `high_phase_terminator`        |          2 | `BodyTerminatorNav` primary; phase > 90 deg, crescent          |
| `below_resolution_body`        |          2 | `BodyBlobNav` (detection-SNR gate); body < 15 px               |
| `negative_cases`               |          3 | Expected `status='failed'`: unnavigable scenes                 |

The per-class minima above sum to 47 and are the authoritative first-stage
budget (#172). They are a budget only: `tests/integration/test_image_library.py`
enforces the tree's shape — every directory is a declared scene class, every
sidecar validates, the primary scene tag matches its directory, `image_id` is
unique, and the filename matches the `image_id` — and asserts no per-class
count. The coverage-matrix and per-class-minimum invariants are #240. The
WS-3 growth target (>=20 per instrument, >=120 total; #235) continues the
same classes.

**Pending operator recommendation: an eighteenth class, `titan_haze`**
(`TitanHazeNav` primary; a hazy body whose navigable feature is a haze
envelope). It is a recommendation, not a decision: neither the
`DECLARED_SCENE_CLASSES` entry nor a budget row is added, and the six
candidate frames are staged under `util/titan_cohort/nominations/` with draft
sidecars rather than committed to the library, because a library sidecar's
ground truth must be `operator_verified` and these carry an autonomous fix.
The argument for a dedicated class: the existing body classes are defined by
what the BODY model predicts (a hard limb, a terminator, a disc), a Titan
frame's navigable feature is a haze envelope with no hard limb at all, its
primary technique is one no other class uses, and its characteristic failure
-- the along-track/radius degeneracy -- is not a failure any body class
exercises; filing Titan frames as `body_full_fov` would make that class's
realism figure of merit average a hard limb and a haze limb together. The
class decision rides #407; the pipeline wiring it would need is #405.

### Per-class selection guide

When a candidate sits between two classes, pick the one that exercises the
expected primary technique and record the judgment in the sidecar `notes`.

| Class                          | Geometric requirement                                                        | Best sources                                                     | Avoid                                                        |
|--------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------|
| `star_dominated`               | >=3 catalog stars predicted detectable in extfov; no body silhouette         | Cassini NAC star-cal frames; NHLORRI cruise                       | Smear > 30 px; saturated bloom across frame                   |
| `body_full_fov`                | Regular body >= 70% of FOV; full limb in frame; >=30% lit                    | Cassini NAC mid-range satellites; Galileo flybys; NHLORRI Pluto   | Crescent > 50% terminator (use `high_phase_terminator`)       |
| `body_partial_overflow`        | Body 70-90% in frame; visible limb arc > 30%                                 | Cassini close encounters; Galileo flybys                          | <50% in-frame (use `body_mostly_offscreen`)                   |
| `body_mostly_offscreen`        | Body 50-90% off-frame; limb arc >=10% visible                                | Cassini closest-approach NAC; Galileo Io/Europa                   | No limb at all in FOV (use `negative_cases`)                  |
| `body_irregular`               | Irregular body where limb uncertainty > 3 px (blob regime)                   | Cassini Phoebe / Hyperion / small inner moons                     | Body so close the blob centroid is also ambiguous             |
| `multi_body`                   | >=2 separable bodies in FOV; not occluding                                   | Cassini family portraits; Galileo multi-moon shots                | Overlapping bodies (occlusion is tested separately)           |
| `ring_only_curved`             | Edge polyline max-deviation > 0.5 px from straight; no bodies                | Cassini Saturn rings, mid-range                                   | Bodies in FOV (use `ring_plus_body`)                          |
| `ring_only_flat`               | Ring-edge polyline curvature < 0.5 px                                        | Cassini ansa shots; long-range ring-only frames                   | Curved enough to reach full rank (defeats the rank-1 test)    |
| `ring_plus_body`               | Rings + >=1 moon in FOV                                                      | Cassini Saturn + shepherd moons                                   | Edge-on rings + body (hard to characterize)                   |
| `stars_plus_body`              | Body + >=3 visible stars                                                     | Cassini, NHLORRI long-exposure frames                             | Accidentally also `multi_body` — check before saving          |
| `one_bright_star_no_body`      | Exactly 1 unambiguous star (next-brightest >= 1.5 mag fainter)               | Cassini + NHLORRI star-cal frames                                 | Next-brightest within 1.5 mag (use the two-star class)        |
| `two_bright_stars_no_body`     | Exactly 2 unambiguous stars; no body, no rings                               | Cassini + NHLORRI star-cal frames                                 | Saturated/faint pair (assignment ambiguous)                   |
| `faint_stars`                  | Predicted SNR < 3.0 for every catalog star in FOV                            | Galileo SSI science frames; Voyager outer-leg                     | A clean frame where stars do show                             |
| `scattered_light`              | Stray-light gradient AND navigable content (score >=5 plus a resolved ring/limb, or >=6 star-field inliers — raised from 3 after the C00598xx quintet, which had only ~2 bright stars each, was correctly declined) | Galileo Earth/Moon outer fields; Voyager encounter outer-leg | Already-flat frames; gradient-only frames with nothing to fit |
| `high_phase_terminator`        | Crescent body, phase > 90 deg                                                | Cassini approach phases; Galileo Earth-departure crescent         | Crescent so thin no terminator pixels rise above noise        |
| `below_resolution_body`        | Body diameter < 15 px (distant body, any mission)                            | Voyager / Cassini long-range satellite frames                     | Body so distant the centroid is sub-noise (use `negative_cases`) |
| `negative_cases`               | Unnavigable: distant tiny body + sensor-limited stars; empty frames          | Spread across all four missions                                   | A scene that *barely* navigates (that is a `low`-tier entry)  |

### Mission-specific hints

- **Cassini ISS (NAC + WAC):** the richest source for every body/ring class
  and the most usable star fields. Both `_RAW.IMG` and `_CALIB.IMG` variants
  exist; the suffix in `image_id` selects the config block. **Saturation
  policy:** the saturation gate is intentionally off for calibrated I/F
  frames (a single I/F threshold cannot identify saturated DN); when accurate
  saturation flags matter (e.g. a Pleiades star-cal with saturating stars),
  curate the matching `_RAW.IMG` instead — Cassini calibration is not
  geometric, so raw and calibrated share pixel coordinates and the same
  ground-truth offset. Keep >=1 `_CALIB` frame in the library so the
  `signal_dn_to_image_unit_scale` path stays exercised.
- **New Horizons LORRI:** strong star-cal coverage (panchromatic, filter slot
  `1`); Pluto/Charon for body classes; `raw_dn` only.
- **Galileo SSI:** the canonical source of `faint_stars` and
  `scattered_light`; carries camera-rotation fitting
  (`fit_camera_rotation: true`), which makes triage slower — budget for it.
- **Voyager ISS:** the second source of `faint_stars` / `scattered_light`;
  must be navigated against the geometrically corrected (GEOMED) products
  because raw Voyager frames carry uncorrected distortion; also
  rotation-fitting. Usable epochs are constrained by the fixed SPICE coverage
  windows (`util/cohort_curation/spice_coverage.json`).

Aim for all four missions across the full library so the calibration never
silently overfits one camera.
