# SpinDoctor Program Plan

*The top-level plan of record for all remaining work. It is written to be
readable without knowledge of the code internals or the statistical
methodology; the detail lives in the three sub-plans it points to. Last
reconciled 2026-08-25.*

**Document map** (what to read for what):

| Document | Role |
|---|---|
| `plans/PROGRAM_PLAN.md` (this file) | What remains, why, in what order, and what can run in parallel. Every open issue is accounted for in the index at the end. |
| `plans/VALIDATION_AND_CALIBRATION_PLAN.md` | Full methodology for Track A (the science): what "validated" and "calibrated" mean, the statistical machinery, acceptance criteria per workstream. Binding where it overlaps anything else. |
| `plans/ENGINEERING_PLAN.md` | Full implementation detail for Tracks B-F: per-item context, file pointers, constraints, and acceptance criteria sufficient to hand any item to a developer (human or model) cold. |
| `plans/COHORT_CURATION_PLAN.md` | The operational playbook for growing the curated image library: metadata-driven discovery, operator review votes, sidecar generation. |
| `plans/OPERATOR_PLAYBOOK.md` | The operator's dispatch sheet: which decisions are pending, which sessions to launch, and in what order. |
| `plans/archive/` | Superseded and fully-executed plans, kept as historical records with dates in their filenames. Nothing in there is current work, though the executed designs remain the reference for what shipped — the C-kernel writing and reading halves and the results index among them. Its `README.md` says what each one delivered and where the follow-ups went. |
| `critiques/archive/` | Point-in-time reviews, frozen as written. `critiques/` itself is empty while no review is open. |

---

## 1. The goal

One sentence: **an end user can take raw archival images from Cassini ISS,
Voyager ISS, Galileo SSI, or New Horizons LORRI and produce — locally or in
the cloud — precisely navigated pointing, reprojected mosaics, per-pixel
geometry backplanes, archive-quality PDS4 bundles, preview images and
metadata, and updated SPICE pointing kernels, with every accuracy and
confidence number backed by published evidence and every capability
documented.**

The end-of-project deliverable therefore has two halves:

1. **A finished pipeline** — every stage works for every supported
   instrument, runs at campaign scale, and is protected by tests.
2. **A defensible accuracy story** — the pointing errors and confidence
   values the pipeline reports are calibrated against evidence a reviewer
   can check, and a capability matrix states exactly what is supported and
   how well, per instrument.

The second half is not polish. This system's outputs are scientific
measurements; a navigation system that reports offsets without defensible
uncertainties is not finished, it is merely running.

## 2. Where we stand

The engineering core is built and healthy: the full navigation architecture
(nine autonomous techniques plus manual), reprojection, backplanes,
simulation, rank-1 (single-axis) ring support, corrected-pointing SPICE
C-kernels as a delivered product, the statistics system, and strict quality
gates (typing, linting, the full unit suite) are in place. Every program that
reads a navigation record reads it through one seam over two storages — the
results tree of JSON documents, and an optional rebuildable index
(`sd_results_index` / `sd_stats_report`) that no program requires. The
de-circularized, realism-tuned simulator and the cross-technique agreement
estimator both exist and are proven on known-truth sims. PDS4 bundle
generation exists only as partially implemented machinery (see Track D): a
spec-tested generator backend, but no final templates and no schema
validation.

Two facts about the evidence behind that health are worth stating plainly,
because both are load-bearing and neither is comfortable. Suite coverage is
79%, not the 90% two shipped plans' acceptance criteria claim, and nothing
enforces any floor (#548). And 10 of 75 curated library frames disagree with
their sidecars in the local integration environment (#288), so the regression
instrument that is supposed to catch a navigation change is itself red; until
it is reconciled, "no new failures against `main`" is the only gate a
navigation-affecting branch can honestly clear.

The single largest remaining block is Track A: **the validation program is
designed and partly built, but not yet run on real frames.** The confidence
formulas are anchored against simulated planted-truth scenes, so confidence
values are not arbitrary defaults, but every output still carries a
`confidence_provisional` marker until the real-frame agreement study
re-anchors them. Turning that provisional story into a defensible one is the
critical path.

The curated image library — the raw material for both regression testing and
validation — continues to grow toward the first-stage budget of 47 images
spanning 17 scene classes (#172) and the final target of at least 120 across
all four instruments (#235). It seeds regression baselines and the
calibration cohort, and supplies the agreement study.

## 3. Why validation dominates the remainder (plain-language version)

No independent record exists of where these cameras were truly pointing —
not to the hundredth-of-a-pixel level the system reports. So accuracy
cannot be checked against an answer key. The program substitutes three
mutually reinforcing sources of evidence:

1. **Simulation with planted truth** — render a synthetic image where the
   correct answer is known by construction, and measure recovery error.
   Trustworthy only if the simulator is (a) independent of the navigator's
   own models and (b) demonstrably realistic compared to real images.
   Proving the realism at campaign scale, and closing the catalogued
   fidelity gaps, is the remaining sim work in Track A.
2. **Agreement between independent methods on real images** — when a star
   field and a moon's edge independently yield the same pointing
   correction to within a fraction of a pixel, that agreement is evidence
   both are right to about that level. This is the only accuracy signal
   available from real archival frames, and extracting it honestly
   requires real statistical care (the methodology plan covers the traps).
3. **A large, deliberately diverse library of operator-verified images**
   — feeding both of the above and serving as the permanent regression
   safety net.

Everything else in the program — new instruments' quirks, PDS4 bundle
generalization, performance, documentation — is real work but
conventional engineering.
This is the part that makes the numbers mean something.

## 4. The tracks

Work is organized into six tracks. A track is a stream of related work
that can proceed largely independently of the others; the parallelism
notes say where they touch.

### Track A — Validation and calibration (the science half)

**Goal:** every accuracy and confidence number the pipeline emits is
backed by published evidence; the `confidence_provisional` marker is
retired.

**Why:** section 3. This is the critical path to the deliverable's second
half and the majority of the remaining effort.

**Shape of the work** (methodology and acceptance criteria in
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`; the workstream codes below are
its section names):

1. **Grow the image library** (#172 first stage, #235 growth; WS-3) —
   continuous background work: automated candidate discovery, operator
   votes in batches, sidecar generation. Feeds everything below.
2. **Prove the simulator realistic** (#227, with #223, #153, #84; WS-2) —
   the de-circularization is done and on main: an independent forward model,
   the truth/idealized information partition with an import-graph guard,
   model-error sweep axes, full detector noise including the I/F path, and
   the rewritten simulator report. #227 stays open only for the *realism*
   residual: the calibration is not yet fitted on the realism-anchored
   renderer configuration, so the realism evidence does not underwrite the
   shipped calibration (#309, the biggest calibration-credibility win
   available); the terminator deliverable is degenerate with no realism
   verdict (#223); and realism is Cassini-only with the authored scene
   mixture unvalidated (#309, #341). Closing #227 is the operator's
   realism-verdict gate, itself gated on #309. The catalogued
   simulator-fidelity gaps feed #309: stars shining through dark limbs
   (#325), the 1-LSB calibrated-product floor (#329), zero cosmic-ray
   transients (#330), per-scene rather than per-detector hot pixels (#331),
   one PSF kernel per instrument (#332), four unmodeled physical error axes
   (#333), the render-time budget on oversampled grids (#290), single-annulus
   rings vs realistic nested ringlets (#377), the placeholder
   `star_psf_sigma` (#342), the possibly registration-absorbing NAC PSF wing
   (#343), the constant haze brightness (#344), the truth-side-noise echo
   into `instrument_config` (#345). The structural information-boundary guard
   (#310) and the sim-navigator mirror-parity guard (#311) harden the
   partition.
3. **The agreement-estimator residual** (WS-0) — the estimator, the
   identifiability map, and the known-truth validation are delivered and on
   main under `util/agreement/`. Findings that constrain item 5: the
   **limb-ring pair is measured as correlated** through the shared
   preprocessing layer, so it cannot carry per-technique covariance claims;
   the **limb-disc pair holds as an anchor against symmetric PSF error** but
   its asymmetric/coma channel is unrenderable by the sim (#359) and the
   disc's sub-pixel NCC resolution floors detectability (#361); **multi-body
   frames are not independent measurements** (#322); a **~2 px inward bias on
   partial-arc limb fits** (#321) surfaced as a navigation finding in its own
   right. The open residual, all feeding the study in item 5: measure the
   real reliability-vs-error coupling on the #225 cohorts (#358, which gates
   the reliability-gate lower-bound claim on real frames), decide whether the
   solve needs a survivorship correction (#360), probe the asymmetric-PSF
   limb-disc channel (#359), quantify the disc-NCC resolution floor (#361),
   and wire the estimator tests into CI (#324).
4. **Camera distortion residual** (WS-17) — the star-field distortion tool
   is delivered and on main, its measured residuals populate the sim
   defaults, and distortion is applied per feature-position in the agreement
   study so it does not masquerade as navigation error. The open residual is
   #355: re-measure and split the Voyager sim distortion per camera once the
   star-lock rate improves.
5. **The agreement study itself** (#225, corroborated by #226; WS-1,
   WS-1b) — the flagship: run the pipeline over hundreds to thousands of
   real frames with two or more independent fiducials and publish the
   agreement statistics. Its bulk (pairwise) layer needs only the library
   and distortion validation; it must not wait for items 2-3, which gate
   only the finer per-technique separation.
6. **Wire real images into CI** (#229; WS-4) — a small cached tier on
   every PR, the full suite on a schedule. Related: the data-independent
   simulator suites still never run in Actions (#336), there is no
   canonical environment for the committed sim baselines (#335), and a
   committed sim render is already stale on `main` without anyone noticing
   (#426) — which is what a deselected integration tier costs.
7. **Re-anchor confidence on real evidence** (#230; WS-5) — re-run the
   existing calibration tooling against the agreement study's
   measurements; retire the provisional marker. The correlated-ring-witness
   fix (#317) is done, so the recalibration no longer trains against those
   rows; the ring orbit-uncertainty severity decision (#316) still sequences
   before it, since it pushes tier boundaries the wrong way, and adopt the
   calibration's armed falsification criterion (#334).
8. **Close the accuracy tail** (#233 measured star SNR and constant
   sensitivity, WS-9; #150/#128 the known ~0.1 px limb bias, WS-10; #234
   realistic noise for calibrated images, WS-13; #232 end-product
   accuracy for backplanes/mosaics/PDS4 values, WS-18).

**Confident-wrong families that poison the validation data** (they belong to
Track B but gate the study, because it consumes navigator output at scale): the
three frames that lock onto the wrong ring feature (#346). The body-witness-veto
family -- a high-phase haze crescent returning a gate-passing success ~30 px
wrong (#328) and the disc technique locking on at extreme shape mismatch (#291)
-- and the body-body occlusion pair -- the disc template (#326) and the
visible-arc report (#327) -- are closed by the cross-technique body-witness veto
and the occlusion-aware body model. The ensemble-independence family -- a seeded
single-star refine dragging a body fix (#222), two ring techniques fused as
independent witnesses (#317), and scattered-light disc/limb errors fused as
independent (#339) -- is closed by the consensus independence resolution.

**Operator's role:** batch votes on library candidates (ongoing); bless the
realism verdict; approve agreement-study frame selection; make the ring
orbit-uncertainty (#316) call; re-bless tiers after item 7. Everything
else is agent-executable.

**Parallelism:** items 1-4 all run concurrently; item 5's bulk layer starts
as soon as 1 and 4 give it cohorts; 6 rides alongside; 7-8 close out in
sequence after 5.

### Track B — Navigation correctness (remaining)

**Goal:** no known case where the navigator returns a confidently wrong
answer or fails on a navigable scene.

**Why:** these defects poison the validation data (Track A consumes the
navigator's output at scale) and are exactly what a user hits first.
The known open defects:

- **#346** — three library frames (N1492091163, N1867601758, N1867602424)
  lock confidently onto the wrong ring feature. Standing library reds.
  Exposure is reduced to sub-25 km/px Saturn scenes by the ring routing
  decision: at and above 25 km/px radial resolution the whole Saturn system
  feeds the annulus composite, which a 131-frame clean-truth head-to-head
  measured wrong on zero accepted answers while the per-edge fit ran
  5% / 13% / 56% / 100% wrong-when-accepted in the 300-1000 / 100-300 /
  25-100 / 0-25 km/px bands; below 25 km/px the trusted evidence is three
  frames validating neither technique, so the threshold stops where the
  measurement stops.
  The annulus-gate recalibration (accepting more of its correct answers,
  #566) and a correlation-picks-basin / edge-polishes hybrid (#567) are
  filed follow-ons.
- **#476** — RingEdgeNav is not shift-equivariant either: a planted shift
  re-locks onto the wrong ring edge. Same family as #346 and #373 seen
  from the round-trip side; the coarse score surface is an alias lattice
  whose members score within 1% of each other, and polarity does not
  separate them. Same reduced sub-25 km/px exposure as #346.
- **#482** — BodyDiscCorrelateNav misses by up to ~1 px on a
  weakly-constrained axis. The residual left after the shift-equivariance
  fix (#447) closes the coarse-grid and boundary-pinning halves.
- **#350** — two resolved-body frames (N1484593951, N1686349893) miss the
  offset tolerance by ~2 px after the recalibration.
- **#373** — the RingEdgeNav coarse seed is not robust against competing
  edge populations (polarity-blind); the coarse-lock family that a
  calibration pass against the library must close.
- **#128 / #150** — the strategic limb-navigation redesign and the ~0.1 px
  limb systematic (shared with Track A's WS-10; design first, validate
  against real images before touching). The measured bias attributes the
  systematic to limb-darkening / photometric roll-off: the edge DT localizes
  the gradient ridge, which sits ~0.5 px inside the geometric silhouette by
  an illumination-dependent amount. The strongest fix is a photometric-limb
  fit (predict the limb-darkened-disc-convolved-with-PSF profile), tracked
  on #150; #128 is the fuller redesign across all body types and
  illuminations. On real frames the fitter contributes only ~0.1 px while
  spacecraft-position / ephemeris error dominates (0.4-1.7 px), so the
  higher-leverage pointing-kernel side (Track D #50) was built first and is
  delivered.
- **#282** — a ~0.05 px, one-pixel-period sub-pixel ripple rides on top of
  the directional bias; a higher-order / matched-filter sub-pixel edge
  estimator would remove it. Precision refinement, secondary to #150.
- **#283** — pixel-centre convention mismatch (simulated `BODY_DISC`
  predicted-centre metadata at `center` vs renderer at `center - 0.5`);
  latent, does not affect the limb path today, fold into #128.
- **#25** — model blurring for very-high-resolution bodies (investigation).
- **#239** — sub-5 px body policy: decided as expected-failure curation;
  what remains is a targeted cohort scan (predicted diameter at or below
  ~5 px, single-body scenes) to find a qualifying frame, as library-growth
  work.
- **#338** — decision: the highly-irregular exclusion discards a
  ground-truth terminator fit on N1853392805; choose among accepting the
  2-px-class ground truth, keeping TERMINATOR_ARC for SPICE-known
  synchronous rotators, or shape models (#23).
- **#130** — star limiting-magnitude calibration against real fields
  (coordinate with #233's measured-SNR work — same frames, same tooling).
- **Titan haze fit refinements** — Titan navigates autonomously via the
  haze solar-symmetry method, validated on an 82-frame Cassini cohort. Four
  measured follow-ups remain: the arc ray reach is sized by the full search
  window rather than by where the limb can be, costing rays on large
  well-framed frames (#403); the flat arc-residual cap is a size-dependent
  gate that cannot simply be raised (#404); the extreme-phase edge of the
  working range is uncharacterized (#401); and the main rings are masked as
  opaque, refusing frames visible through the C ring or the gaps (#402).
- **#400** — the ensemble merge and tier logic have never been exercised on
  a strongly anisotropic covariance; the haze fit reports a 0.36 px by
  1.02 px oblique ellipse whose orientation varies with the sun direction.
- **#447** — body techniques do not return the shift they are given, which is
  the round-trip residual every corrected-pointing consumer inherits. This is
  the one piece of open work in flight: PR #484 fixes both body-side causes
  (a sub-pixel silhouette probe, and a per-axis NCC-quadratic fallback for
  saturated refinement) and files the rest as #476, #482 and #483. It is
  mergeable and green, and it is fifty commits behind `main`.

**Parallelism:** parallel with Track A, except that the coarse-lock family
(#346, #476, #373) gates the agreement study -- a technique that reports a
confidently wrong answer moves the cross-technique disagreement WS-1 measures
-- and #288 gates reading any Track A cohort with confidence.

### Track C — Statistics, QA, and the accuracy checkpoint

**Goal:** navigation quality is continuously measured, not assumed.

The statistics system is built, tested, and documented in the user guide, and
it now reports over a stream of records rather than over a database, so it
runs with or without an index. Remaining: the library coverage-matrix
invariant (#240); two reporting defects the seam work surfaced — a file that
could not be retrieved is counted by a tree-backed report and by no
index-backed one (#533), and the report retains more than it prints under
`--top-n` (#535); and the standing practice of re-running the library
cross-check after every calibration-affecting change. Small track, mostly
done, listed separately because it is the program's QA instrument.

### Track D — Capability completion (decision gates first)

**Goal:** every capability the docs imply either works and is validated,
or is explicitly scoped out in the capability matrix.

Titan navigation is delivered: the haze solar-symmetry method ships as a
model, a technique, and a simulated-Titan renderer, validated against an
82-frame Cassini cohort and a 700-scene planted-truth campaign, with the
published bound (1 px cross-track, 3 px along-track) confirmed by
star-anchored evidence. Two capability extensions are deferred as issues:
the self-calibrated haze-radius table that would remove the dominant
along-track error and answer the wavelength-dependent-haze-top question
(#397), and methane surface-window (CB3) cartographic correlation as a
refinement stage (#398).

Logging is delivered across every pipeline program: one main logger per
program run and one image logger per image, per-module levels, four
independently-controlled sinks, and one identical flag set whose defaults
also live in the configuration file, so learning the logging surface for one
program teaches it for all of them. Cloud-task workers are silent at the
terminal and complete in their log files. Three defects the work surfaced are
open: a mosaic cloud task still reports success when every image in it failed
(#418), the GUI viewers print library log records to stdout through
pdslogger's handler-less fallback (#423), and `sd_create_bundle_cloud_tasks`
is unwired and should be removed rather than fixed (#424). Three follow-ups
the design deferred are also open: the config namespace that `logging` was
lifted out of is still organized on no stated axis (#427), an upstream
registry-eviction request to `rms-pdslogger` (#428), and extending the same
surface to the `util/` tooling (#429).

The headline capability item is delivered in both halves: the navigator
records each image's corrected attitude as a C-matrix and `sd_create_ck`
ships updated-pointing SPICE C-kernels as a product, and the backplane and
reprojection consumers apply the recorded C-matrix whenever a usable one
exists — with the pixel offset as the documented fallback for
fitted-rotation, offset-only and malformed records, and uncorrected pointing
with the reason recorded when neither is usable — closing #50 and #188. The
designs are archived at `plans/archive/CK_KERNEL_PLAN_2026-08-04.md` and
`plans/archive/CMATRIX_READERS_PLAN_2026-08-09.md`; current behavior is
`docs/user_guide/user_guide_ck_kernels.rst` and its dev-guide companion.
What that leaves is the kernel follow-ups: the remaining half of the oops
API swap (#433), whose offset-to-rotation construction waits on
SETI/rms-oops#213 now that the convention conversion goes through oops, one
rotation convention about the image
center (#434) -- which is upstream of the wider distortion cohorts (#561) and
of any decision to fit rotation again -- with the static-twist FK/IK pair
behind it (#435, #436), SPICE database registration
(#437), the interior-epoch fidelity bound through an adaptive record cadence
(#440, #444) with its per-instrument characterization (#455), the
kernel-input items (#446, #448, #468), and a memory bound on `sd_create_ck`
(#513). One of them is a navigation defect rather than kernel work and is
listed under Track B (#521), deferred while no instrument fits rotation.

The results index is delivered too (#430, #487, #507). It is an optional,
rebuildable index over the results tree, so programs stop reading one JSON
document per image on a cloud root, and no program requires it: a run that
names none reads the tree exactly as it always has. What it actually changed
is broader than the index — every program that reads a navigation record now
reads it through one seam over both storages, with one rebuild, one opener
and one enumerated list of where the two storages answer differently. The
design is archived at `plans/archive/RESULTS_INDEX_PLAN_2026-08-04.md`;
current behavior is `docs/user_guide/user_guide_results_index.rst` and its
dev-guide companion. Its follow-ups divide into three groups: capability
extensions nobody is blocked on (a document column for bundle
generation #464, a schema wide enough for the curation triage tool #465, a `--since`
selector #467, `sd_offset` writing its own rows #486, a pruned-rows
question #542), operational questions with a decision in them (a documented workflow
for getting the index to cloud workers #466, the consumer open that takes a
write lock it never needs #462), and correctness or hygiene items
(#472, #493, #496, #497, #501, #512, #514, #515, #516, #528,
and #531, #533, #534, #536, #538, #540, #541).

Some items start with an operator decision, because each is a scope
commitment:

| Decision | Then the work is |
|---|---|
| **Backplane content** (#28 family): finalize the backplane set and formats | #55, #54, #57, #77, then the generator hardening (including the product-correctness defects #251, #252, #253 found by the #241 test suite). |

**PDS4 output bundles are required for all four instruments** — not a
scope decision — and **none of it works end to end today**. The Cassini path
is partially implemented machinery with no final templates, no schema
validation, and only a spec-tested backend; Voyager, Galileo, and New
Horizons additionally hit not-implemented walls. The work is: finish and
validate the Cassini path (final templates — acceptance list recorded on
#53; schema validation; the swallowed `template.write` errors and the
dev-guide output-layout mismatch tracked by #265), then generalize —
per-mission label templates, LID builders, and collection machinery (#53
with #66, #67, #69, #71-#76, #79, #47, #30, #63). Distinct from this, **PDS4
*input*** (#34) — reading PDS4-archived data instead of PDS3 — is treated
like any other future instrument: the archives do not exist yet, their
creation is external development outside our control, and input support is
*not* required for project completion; when an archive appears, its support
replaces the PDS3 source for that instrument.

Plus, not gated on decisions: the capability matrix itself (#231),
cloud-operation audit (#108, #67, #141, #142, and the cloud-task items the
logging work left open, #418 and #424), performance and safe
parallelism for campaign scale (#236, #103, #134, #126), and config
validation (#118, which should follow the namespace reorganization #427
rather than freeze the current grouping into a schema). The user-guide
completion items are delivered.

**Parallelism:** decisions can be made any time; the resulting work is
independent of Tracks A-B except that #232 (end-product accuracy) wants
the backplane-content decisions settled and the PDS4 bundle
generalization done.

### Track E — Test and documentation debt

**Goal:** no shipped stage without tests; no doc a future maintainer
cannot follow.

- Zero-coverage stages: backplane CLI (#241), PDS4 CLI (#242).
- Untested star-conflict logic (#243); real-image baselines beyond one
  frame (#174).
- Summary-PNG unit tests (#177).
- The image-library regression reconciliation (#288). In the local
  integration environment it is not reduced to the deliberately-red pins:
  10 of 75 frames disagree, which is the state Track A's evidence has to be
  read against, and it is what blocks the only place a built product is
  compared between the results tree and the index (#547).
- Suite coverage is 79% and unenforced against a stated 90% (#548). The
  shortfall is almost entirely PyQt6 widget code. This is an operator call
  before it is an implementer's: raise the number, or ratify a lower floor
  and gate on that one.
- Tests that cannot fail, which this project has now hit
  repeatedly: #516 asserts on the row a root-blind write would keep, and re-ratcheting
  the library pins the shift-equivariance fix moves is #483.
- Test-suite hygiene: seventeen test modules over the 1000-line cap (#525),
  POSIX-only constructs behind a Windows-support claim (#473), fixture clock
  seconds that do not follow from their epochs (#530), and consolidating
  `test_record_source.py` onto the shared fixtures (#524).
- Docs: Sphinx nitpicky-clean CI (#129, with the gate the rules already
  require but nothing runs, #438) and terminator-doc verification (#122).
  Smaller doc defects: the README names two of fourteen command-line
  programs and links to no guide (#545), four technique guides name a
  `dt_fitting.py` that is a package (#549), and config placeholder comments
  still carry an internal codename (#470, #471). The per-instrument
  chapters are written, one per instrument in
  each guide, as is the metadata-JSON format chapter
  (docs/user_guide/user_guide_metadata.rst, with a staleness-guard test),
  along with the filters / uncertainty / troubleshooting dev-guide pages,
  the mosaic-viewer API-reference pages, and the curation-tooling language
  pass.
- Tooling parity: give the `util/` programs the logging surface the
  pipeline programs now have (#429) — several run for hours and report
  through bare `print()`.

**Parallelism:** entirely parallel with everything; good filler between
larger items. #241/#242 should precede any serious PDS4/backplane work in
Track D so that work lands on tested ground.

### Track F — Remaining instruments, features, and hardening

**Goal:** the other three instruments reach Cassini's proven level; the
enhancement backlog and code-quality tail are burned down.

- **Instruments** (after Track A proves the Cassini spine): Voyager star
  navigation (#19), Galileo star navigation (#18),
  LORRI PSF and product policy (#2, #138, #33), outer-planet ring models
  (#82, #81, #83), rotation-pyramid cost (#126), degradation classifier
  (#181), per-instrument calibration extension (re-run #230 per
  instrument as library frames land).
- **Features:** BOTSIM (#27), star streaks (#22), backplane-reader repo
  (#107), PDS4 input when external archives exist (#34 — replaces the
  PDS3 source per instrument; not required for project completion),
  cartographic/bootstrap navigation
  (#184 — explicitly far off), polarity-aware ring matching (#183),
  chaotic-rotator poses (#187), manual-nav dialog redesign (#186),
  gated-feature PNG styling (#185), stop-after-features flag (#182),
  body shape models (#23), sim polish (#84, #78, #151, #152, #157, #158).
- **Hardening/cleanup** (any time, mostly small): #15, #21, #38, #39, #65, #92, #96-#105, #109, #110, #119, #135, #137, #140, #143, #144, #147, #155, #212, plus the GUI viewers printing library log records to stdout (#423), the upstream `rms-pdslogger` registry-eviction request (#428), Cassini BOTSIM pairs defined two different ways (#494), stating the encoding wherever a document or text file is read or written (#518), and removing the `AttrDict` `_IS_IMMUTABLE` marker once oops stops writing its mutability bookkeeping onto foreign objects (#552).

**Parallelism:** hardening is permanent filler. Instrument work waits for
Track A's Cassini verdict only in the sense that there is no point
calibrating three more instruments with an unproven method; the
star-navigation bug fixes (#19, #18) can start any time.

## 5. Suggested global order

0. **First, and small:** land PR #484 (#447, the round-trip residual), which
   is green and fifty commits behind `main`.
1. **Then, in parallel:** Track A items 1-4 (library growth, sim
   realism campaign, agreement-estimator real-frame follow-ups, distortion
   feed-in), with the gates above respected -- the coarse-lock family before
   the agreement study collects, and #288 before any cohort is read as
   evidence rather than as a comparison against `main`. The operator
   decisions in section 6 go across as a batch — they cost nothing to decide
   early and unblock scoping.
2. **Next:** Track A item 5 (agreement study, bulk layer first), with
   Track E test-debt and Track B remainder as parallel fill.
3. **Then:** Track A items 6-8 (CI tiers, real-anchored recalibration,
   accuracy tail). This is the calibration finish line: confidence and
   uncertainty defensible against reality.
4. **Then:** Track D build-out per the decisions; Track F instruments,
   re-running the now-proven calibration per instrument.
5. **Last:** capability matrix finalized (#231), documentation
   completion, end-product accuracy (#232) — the deliverable's
   evidence package.

**Effort honesty:** Track A is multi-week at agent pace; its two largest
remaining items (sim-realism campaign, agreement study) are serialized only
through the estimator's real-frame follow-ups between them. Tracks B+C+E are
one to two weeks of interleavable small/medium items. Track D depends on
scope decisions; Track F is another multi-week block dominated by
per-instrument calibration repeats. Operator hands-on time is dominated by
library votes and the decision gates, not by any implementation.

## 6. Operator decision gates (collected)

1. **Ring orbit-uncertainty severity (#316)** — the ring radial channel
   prices catalog fit residual as a fully correlated whole-edge
   displacement, which the catalog's own mode-fitted epochs suggest is
   several-fold over-severe. It demotes five operator-verified Keeler
   frames to low. Decide: ship at the conservative default, ratchet
   `rings.orbit_radial_sigma_correlated_fraction`, or implement the
   wander decomposition. Reversible by config either way.
2. **Titan haze navigation ratification (#407)** — the method is
   implemented and validated; what needs a decision is a bundle of
   mid-implementation specification changes, three acceptance bounds the
   evidence argues with, and three staged curation artifacts (an overlay
   review batch, six library nominations, and a `titan_haze` scene-class
   recommendation).
3. **Highly-irregular terminator fit (#338)** — choose among accepting the
   2-px-class ground truth for resolved highly-irregular bodies, keeping
   TERMINATOR_ARC for SPICE-known synchronous rotators, or shape models
   (#23), then a session implements the choice on N1853392805.
4. **Simulator realism verdict (#227)** — close #227 once the calibration is
   fitted on the realism-anchored renderer configuration (#309) and the
   realism evidence underwrites the shipped calibration; gated on #309.
5. **Coverage floor (#548)** — the suite measures 79% against a 90% floor two
   shipped plans' acceptance criteria assert, and nothing enforces either
   number. Raise coverage to the stated floor, or ratify a lower one and gate
   CI on it. Leaving both the claim and the absence of a gate in place is the
   one option that keeps saying something untrue.
6. **Cassini predicted kernels (#459)** — whether Cassini navigation should
   run from the predicted rather than the reconstructed kernels. A branch
   (`origin/rf_ck_cassini_predicted`, nine commits, last touched 2026-08-07)
   exists with no pull request; say whether it is live before it drifts
   further from `main`.
7. **New Horizons pointing family (#468)** — whether the merged family should
   be declared reconstructed on its comment-area evidence rather than staying
   `UNCLASSIFIED`.
8. **The written-product comparison (#547)** — the one place a built product
   is compared between the results tree and the index runs against no frame;
   whether to repoint it or to accept the gap.
9. **Getting the index to cloud workers (#466)** — which documented workflow
   to adopt, since it decides how #464 is deployed.
10. **Where the confidence scale saturates (#557)** — the combined confidence
    caps at 0.99 on the ordinary two-technique agreement, and WS-5's monotonic
    calibration map cannot separate what the cap collapsed, so calibration
    does not settle it later.
11. Recurring: library batch votes; agreement-study frame selection; tier
    re-blessing after #230.

## 7. Issue index (open work by track)

Every open issue, listed exactly once by the track that owns it. 240 issues
as of 2026-09-03; the counts are given so a reader can tell a stale index
from a current one at a glance.

| Track | Count | Issues |
|---|---|---|
| A — validation & calibration | 51 | #84, #153, #172, #174, #176, #223, #225, #226, #227, #229, #230, #232, #233, #234, #235, #290, #309, #310, #311, #316, #319, #321, #322, #324, #325, #329, #330, #331, #332, #333, #334, #335, #336, #341, #342, #343, #344, #345, #355, #358, #359, #360, #361, #377, #380, #399, #405, #407, #409, #426, #561 |
| B — navigation correctness | 25 | #25, #128, #130, #150, #239, #282, #283, #338, #346, #350, #373, #394, #400, #401, #402, #403, #404, #447, #476, #482, #521, #557, #558, #566, #567 |
| C — statistics & QA | 4 | #240, #340, #533, #535 (plus the standing cross-check and campaign-report practice) |
| D — capability completion | 73 | #28, #30, #47, #53, #54, #55, #57, #63, #66, #67, #69, #71, #72, #73, #74, #75, #76, #77, #79, #108, #118, #126, #141, #142, #231, #236, #251, #252, #253, #265, #397, #398, #411, #418, #424, #427, #433, #434, #435, #436, #437, #440, #444, #448, #455, #459, #462, #464, #465, #466, #467, #468, #472, #486, #493, #495, #496, #497, #501, #512, #513, #514, #515, #519, #520, #528, #531, #534, #536, #538, #540, #541, #542 |
| E — test & docs debt | 28 | #122, #129, #177, #241, #242, #243, #288, #379, #391, #429, #438, #443, #446, #470, #471, #473, #483, #516, #524, #525, #530, #545, #547, #548, #549, #554, #562, #563 |
| F — instruments, features, hardening | 59 | #2, #15, #18, #19, #21, #22, #23, #27, #33, #34, #38, #39, #65, #78, #81, #82, #83, #92, #96, #97, #98, #99, #100, #101, #102, #103, #104, #105, #107, #109, #110, #119, #134, #135, #137, #138, #140, #143, #144, #147, #151, #152, #155, #157, #158, #181, #182, #183, #184, #185, #186, #187, #212, #388, #423, #428, #494, #518, #552 |

Priority census across all six tracks: no Critical, 22 Essential, 66
Important, 107 Useful, 33 Minor, 12 Defer. Every open issue carries exactly
one Priority and one Effort label and at least one each of A-type and
B-location.

Cross-listed items (listed once above, noted here): #150/#128 sit in Track B
and also serve Track A's limb-bias workstream (WS-10); the confident-wrong
ring-lock family (#346, #476) sits in Track B but gates the Track A
study; #103/#134/#126 serve both Track D performance and Track F
hardening; #513 and #520 are results-index work in Track D that lands in the kernel
writer and the reprojection package respectively; #521 is a kernel-facing
defect owned by Track B because the fix is in the navigator; the
per-instrument guide chapters exist and each Track F instrument workstream
extends its own pair in the same change; #174 baselines are Track A
infrastructure delivered as Track E test work; #288 and #547 are one
dependency seen from two tracks.

---

*History: this plan supersedes `plans/archive/ROADMAP_2026-07-12.md` (the
issue-ordered pipeline build-out) and
`plans/archive/FULL_PROGRAM_AFTER_MANUAL_WORK_2026-07-11.md` (the
six-track task inventory), consolidating both into one top-level view.
The validation methodology and the curation playbook were already single
documents and remain in place as the detail layer.*
