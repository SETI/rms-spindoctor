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

Ordering within the track: the confidently-wrong defects first
(#346, #350), because the agreement study consumes ensemble output
at scale and several curated library frames pin these as standing red
regressions. Then the coarse-lock calibration (#373), then the
investigation/design items (#25, #128/#150), with the smaller decision items
(#130, #239, #338) as fill.

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

### #346 — the remaining confident-wrong ring-lock

- **#346** — three library frames (N1492091163, N1867601758, N1867602424)
  lock confidently onto the wrong ring feature; standing library reds, tied
  to the coarse-lock calibration (#373).

### #350 — post-recalibration resolved-body red

- **#350** — two resolved-body frames (N1484593951, N1686349893) miss the
  offset tolerance by ~2 px after the recalibration. One debugging session
  against its named frames; the sidecars pin them red until resolved.

### #373 — DT coarse-prior search vs competing edge populations

The coarse NCC search over the distance-transform image can lock onto the
wrong edge population (e.g. a ring edge when fitting a limb) and hand the
Levenberg-Marquardt refine an unrecoverable prior.
`src/spindoctor/nav_technique/dt_fitting.py` (`coarse_ncc_search`). The
polarity-weighted coarse seed already landed; what remains (#373) is making
the RingEdgeNav coarse seed robust against competing edge populations even
when polarity is blind — a calibration pass over the library that
characterizes when the coarse search's top basins are ambiguous, and either
widens the second-opinion gate to the other DT techniques or adds
per-feature-type edge masking. Needs the library cohort; coordinate with
Track A so the fix is measured, not guessed. #346 supplies the concrete
wrong-lock datapoints, and #476 quantifies the round-trip cost: under
planted pointing shifts RingEdgeNav re-locks onto the wrong ring edge
(median residual 2.67 / 1.19 px v/u, growing with the shift), with the
shift-equivariance sweep as the before-measurement any fix must move.

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
ripple (#282); (3) a pixel-centre convention audit (#283). **Constraint:** do
not enable a fitter change until
the real-image measurement exists (Track A #225 provides it) — the current
partial cancellation is accidental and a well-meaning "fix" can make real
accuracy worse. On real frames the fitter contributes only ~0.1 px while
spacecraft-position / ephemeris error dominates (0.4-1.7 px), so the
higher-leverage pointing-kernel side (#50) was built first and is
delivered. #128 is the fuller redesign (all body types and illuminations)
and starts with a design document, not code.

### Smaller Track B items

- **#25** — high-resolution bodies: the model renders sharper than the
  PSF-blurred image; investigate blur-matching the model
  (`nav_model_body.py`) at high resolution. Investigation first: measure
  whether it actually moves offsets on library close-flyby frames.
- **#130** — per-instrument star limiting magnitudes measured from real
  star fields (a small campaign over the library's star classes;
  coordinate with #233's measured-SNR work — same frames, same tooling).
- **#239** — implement the sub-5 px body policy the operator settled on
  (expected-failure curation); the open work is the targeted diameter-
  filtered cohort scan for a qualifying single-body frame.
- **#338** — decision: the highly-irregular exclusion discards a
  ground-truth terminator fit on N1853392805; implement whichever option the
  operator picks (accept the 2-px-class ground truth, keep TERMINATOR_ARC for
  SPICE-known synchronous rotators, or shape models per #23).
- **Titan haze fit** — the haze solar-symmetry method ships and is validated;
  four measured refinements remain: the arc ray reach sized by the search
  window rather than by where the limb can be (#403), the flat arc-residual
  cap that behaves as a size-dependent gate (#404), the uncharacterized
  extreme-phase edge (#401), and opaque-ring masking that refuses frames
  visible through the C ring or the gaps (#402). The ensemble has never been
  exercised on the oblique covariance the fit reports (#400), and two
  pre-existing library reds found while attributing its cross-check need an
  owner (#406).

## Track C — Statistics and QA

- **#240** — coverage-matrix invariant test: every scene class >= 2
  sidecars, every autonomous technique the expected primary somewhere.
  Lives with the library structural tests
  (`tests/integration/test_image_library.py`); runs in the deliberate
  tier, marked expected-incomplete until #235 fills `faint_stars` and
  `ring_only_flat`.
- **Standing practice** — after any calibration- or technique-affecting
  merge: `util/calibration/library_crosscheck.py` over the full library,
  every per-image delta accounted for; `sd_stats_ingest` /
  `sd_stats_report` over campaign outputs as the accuracy checkpoint
  (both from the statistics system).

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

Output current state: nothing works end to end yet. The Cassini path is
partially implemented — the per-dataset hook pattern (template dir,
LID/LIDVID builders, template variables) exists on
`DataSetPDS3CassiniISS` and the collection machinery runs — but it has
no final templates, zero tests (#242), and no schema validation, so its
output is unvalidated. The other three instruments additionally hit
`NotImplementedError` walls in their `pds4_*` DataSet hooks. The work
is therefore: finish and validate Cassini first (final templates,
tests, schema validation), then generalize — per-mission template trees
plus hook implementations, mechanical but voluminous.

Work items, in dependency order:

1. **#265 — swallowed label-write errors and the output-layout mismatch**
   — every `template.write` ignores pdstemplate's error/warning counts
   (an unresolved variable silently drops the label while the run reports
   success), and the dev-guide "Output layout" section describes a layout
   neither the code nor the user guide matches:
   `src/spindoctor/cli/pds4/collections.py` and the surrounding writer
   path. Fix the write path to fail loudly on a dropped label and
   reconcile the layout documentation.
2. **Template finalization acceptance list** — the items recorded
   on #53: schema validation, the unreferenced `cassini:*` variables and
   hardcoded placeholders, TITLE/DESCRIPTION wording, collection date
   ranges, unrendered bundle-level products, variable-less global-index
   labels, FITS placement (#69/#30), missing-value sentinels,
   non-navigated-image handling, and the `.tab`/`.csv` + directory-layout
   decision. These are the acceptance criteria for "final templates" in
   the paragraph above.
3. **#69, #30** — backplane FITS description in data labels; backplane
   label design (couples to the #55 backplane-set decision).
4. **#79** — scrape PDS4 context products for targets (feeds #73).
5. **#71-#76, #47** — label/collection completeness items, each small:
   parameterized bundle name/version, target handling, ring geometry
   class fields, global-index labels, collection CSVs, ring incidence
   angle.
6. **#66** — integrity-checking pass over a generated bundle.
7. **#67** — cloud-aware bundle generation (with the Track D cloud
   audit).
8. Schema-validate generated `.lblx` against the PDS4 schemas in CI for
   all four instruments (acceptance for the whole family).

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
record are `plans/CK_KERNEL_PLAN.md` (the writing half) and
`plans/CMATRIX_READERS_PLAN.md` (the reading half: the backplane and
reprojection readers apply the recorded C-matrix by frame replacement, with
the offset as the documented fallback); the consumer-facing and
developer-facing documentation is `docs/user_guide/user_guide_ck_kernels.rst`
and `docs/dev_guide/dev_guide_ck_kernels.rst`. Validation is a closed loop:
navigate, generate a kernel, furnish it, re-navigate, and confirm the
second offset is approximately zero and the C-matrix matches, which the
round trip does on Cassini NAC, Cassini WAC, Voyager and LORRI; the reader
comparison (`tests/integration/test_cmatrix_readers.py`) holds the two
consumption paths together through the switched consumers themselves.

The round trip also measured something that belonged to Track B rather than
here: a technique re-measuring a corrected frame did not return exactly the
negative of the shift it was given, costing up to 0.49 px on frames carried
by the correlation and distance-transform body techniques. Both body-side
causes are fixed (#447): the limb / terminator polyline vertices are
probe-refined onto the sub-pixel geometric boundary, and the correlator's
sub-pixel refinement falls back to the NCC quadratic vertex when the
upsampled-DFT window saturates instead of pinning at half a pixel. The
`util/calibration/shift_equivariance.py` sweep is the standing measurement
(baseline in `util/calibration/shift_equivariance_baseline_20260808.md`).
What remains is filed: RingEdgeNav's wrong-edge re-locks under planted
shifts (#476), the disc correlator's ~0.5-1 px miss on a weakly-constrained
axis now that the pinning no longer hides it (#482), and the library-pin
re-ratchet for the five frames the fix legitimately moved (#483).

The plan's own follow-ups are filed: the oops API replacing the hand-derived
derivation (#433), fitted-twist support (#434) with the static-twist FK/IK
question behind it (#435, #436), SPICE database registration (#437), the
interior-epoch fidelity bound through an adaptive record cadence
(#440, #444) with its per-instrument characterization (#455), and the
kernel-input handling items (#446, #448, #468).

### The results index (#430)

An optional, rebuildable database index over the results tree, so that
programs needing a few fields per image stop reading one JSON document
each -- one paid cloud round trip per image per program today, at order
400,000 images for a Cassini-scale run. Design is settled and detailed in
`plans/RESULTS_DB_PLAN.md`, which is self-contained. The JSON documents
stay authoritative, no program ever requires the index, and the
file-reading paths remain the default.

### Capability matrix (#231)

Generated/test-verified matrix; see the issue for the two-axis design
(feature support x validation status). Implementation home:
`docs/user_guide/` page generated from the registries
(`spindoctor.dataset`, `spindoctor.obs`, technique registry) plus a
static validation-status table that the WS reports update; a test
asserts the generated half matches the registries.

### Cloud and scale

- **#108** — audit every `sd_*` CLI for logging, cloud operation, and
  working `cloud_tasks` variants; fix what the audit finds. The logging
  third is done: every pipeline program takes the same flags with the same
  defaults, resolves per-module levels the same way, and writes a main log
  plus per-image logs to the same path scheme, with the statistics and GUI
  programs deliberately excluded and on `print()`. What the audit found and
  did not fix is now three issues — #418, #423, #424 below — and what
  remains under #108 itself is the cloud-operation half: whether each
  driver's cloud path actually works end to end, which is untested for
  several of them.
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
- **#411** — migrate the technique config keys to snake_case, so the
  configuration spells component names the way `log_key_for` and the rest
  of the config system already do.
- **#427** — the config top-level namespace mixes domain objects, pipeline
  stages, instruments, output products, and infrastructure on no stated
  axis, with `general` as the catch-all. Lifting `logging` out of `general`
  fixed the worst instance and left the pattern. Sequence this **before**
  #118 (config validation): a schema per section is easier to write and to
  keep honest once the sections are organized, and doing it the other way
  freezes the current grouping into a schema.
- **#141 / #142** — dedup the CLI driver preamble and cloud-task loop;
  fix the dropped `extra_params` and the ImageFiles cardinality
  disagreement.
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
- **#288** — image-library regression reconciliation: the standing red set
  is now reduced to the deliberately-pinned frames, each owned by an open
  navigation issue (#338, #346, #350). N1530185128 (#351) and W1444747627
  (#352) are now re-ratcheted to success/medium and green: the ensemble
  blob-outlier drop lets the multi-star fix commit on the former, and the
  star-navigation hardening already locks the latter. N1806609736 (#392) is
  green as well: the shape-lock veto no longer misfires when a star fix
  corroborates the limb. N1572105349's pin was
  owned by #222 (now closed by the independence resolution); its autonomous
  regression should flip green on the next integration run against real
  holdings (that suite does not run in PR CI). Keep the pins accurate as
  issues close and re-ratchet only what they authorize.
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
- **Galileo SSI:** #18 — star navigation broken (same cluster); #17 —
  REDO product handling in the dataset layer.
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

### Hardening / cleanup tail

Mostly small, any time, no ordering: #13 SCET strings, #15 overlay
occlusion of background models, #21 metadata inventory cleanup, #38
filecache config, #39 AttrDict, #43 `--pds3-holdings-root`
placement, #65 exception class, #92 dependency groups, #96 dead code, #97 oversized
modules, #98 registry consolidation, #99 orphan report_profile, #100
root-path getters, #101 ArgumentParser.error, #102 CLI globals, #103
thread-unsafe caches, #104 broad excepts, #105 typed interop
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
logger-per-image design will find it).
