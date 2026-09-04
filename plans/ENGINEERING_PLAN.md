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

Ordering within the track: the confidently-wrong defects (#346, #476, #350), because the agreement study consumes
ensemble output at scale and several curated library frames pin these as
standing red regressions. Then the coarse-lock calibration (#373), then the
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

### #447 — the round-trip residual (PR #484 open)

A technique re-measuring a frame whose pointing was corrected by its own
previous answer should measure zero, and the body techniques do not:
`BodyLimbNav` fell 0.14 px short on `W1637520502_1_CALIB` and
`BodyDiscCorrelateNav` 0.50 px short on `C3446143_GEOMED`. PR #484 measures
the non-equivariance across the whole library with a sweep harness
(`util/calibration/shift_equivariance.py`, baseline committed), then fixes
both body-side causes: a sub-pixel silhouette probe
(`src/spindoctor/nav_model/silhouette_probe.py`) that stops the discrete
polyline extraction from re-rasterizing a ridge instead of translating it,
and a per-axis NCC-quadratic fallback in `evaluate_candidate` for the case
where the upsampled-DFT refinement argmax lands on the window boundary and
the old code reported a pinned +-0.5 px.

The PR is green, mergeable, and fifty commits behind `main`. What it filed
rather than fixed: RingEdgeNav is not
shift-equivariant either and a planted shift re-locks it onto the wrong ring
edge (#476, same family as #346 and #373 seen from the round-trip side, and
now measured down to an alias lattice whose members score within 1% of each
other and which polarity does not separate -- with exposure reduced to
sub-25 km/px Saturn scenes now that coarser ones route to the annulus
composite); `BodyDiscCorrelateNav` still misses by
up to ~1 px on a weakly-constrained axis (#482); and the library pins the fix
moves need re-ratcheting (#483, which cannot be done honestly until #288 is
reconciled).

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
sub-25 confident-wrong exposure stays open under #346/#476. The threshold is
Saturn-measured only. Two follow-ons are filed: recalibrating the annulus
gates so more of its correct answers are accepted (#566 -- today they veto
many right answers, which costs coverage, not correctness), and a hybrid in
which the correlation fit picks the basin and the edge fit polishes within
it (#567).

- **#346** — three library frames (N1492091163, N1867601758, N1867602424)
  lock confidently onto the wrong ring feature; standing library reds, tied
  to the coarse-lock calibration (#373). Exposure is reduced to the
  sub-25 km/px regime by the routing decision above.

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
wrong-lock datapoints.

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
  exercised on the oblique covariance the fit reports (#400).

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
  degraded: with 10 of 75 library frames red locally (#288), a cross-check
  can only be read as "no *new* deltas against `main`".

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
2. **#519 — labels carry empty `START_DATE_TIME`, `STOP_DATE_TIME` and
   `IMAGE_MID_TIME`.** An archive-quality label with empty time fields is
   not archive-quality, and it is exactly the class of defect #265's
   swallowed write errors let through, so fix #265 first and this becomes
   visible rather than silent. Also worth knowing when picking this up:
   `sd_create_bundle` crashes inelegantly on a missing metadata file, which
   is the same area.
3. **Template finalization acceptance list** — the items recorded
   on #53: schema validation, the unreferenced `cassini:*` variables and
   hardcoded placeholders, TITLE/DESCRIPTION wording, collection date
   ranges, unrendered bundle-level products, variable-less global-index
   labels, FITS placement (#69/#30), missing-value sentinels,
   non-navigated-image handling, and the `.tab`/`.csv` + directory-layout
   decision. These are the acceptance criteria for "final templates" in
   the paragraph above.
4. **#69, #30** — backplane FITS description in data labels; backplane
   label design (couples to the #55 backplane-set decision).
5. **#79** — scrape PDS4 context products for targets (feeds #73).
6. **#71-#76, #47** — label/collection completeness items, each small:
   parameterized bundle name/version, target handling, ring geometry
   class fields, global-index labels, collection CSVs, ring incidence
   angle.
7. **#66** — integrity-checking pass over a generated bundle.
8. **#67** — cloud-aware bundle generation (with the Track D cloud
   audit).
9. Schema-validate generated `.lblx` against the PDS4 schemas in CI for
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

Two further items in this family, neither found by that suite:

- **#496** — `sd_backplanes_cloud_tasks` lets a per-image failure escape the
  task handler, so the worker's failure mode is the handler's rather than
  the pipeline's. Same family as #418: decide what a task's status owes a
  retrying queue.
- **#520** — move the pointing selection and application code
  (`src/spindoctor/cli/reproj/offsets.py`: metadata pointing selection and
  the C-matrix/offset application ladder) out of the reprojection CLI
  package. It already has two consumers, the reprojection and backplane
  stages, which is the condition the code's own comment names for promoting
  it into the library package proper.

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
PR #484 fixes both body-side causes and is open.

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

Its follow-ups, none blocking:

- **Capability extensions:** a document column so bundle generation stops
  reading one file per image (#464, likely a separate optional database
  since it roughly doubles the index size); a schema wide enough to serve
  the curation triage tool (#465, wanted only if triage's ten rglobs per
  frame remain a practical pain); a `--since` selector so a re-scan stops
  paying for the whole listing (#467); `sd_offset` writing each result into
  the index as it navigates (#486); and an index that can say whether a
  root's rows were pruned (#542).
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
  construction audit (#541).
- **Reported through other programs:** `sd_mosaic` reports a lost index as N
  failed images rather than one run-level condition (#493), and there is no
  integration-tier case for a document the ingest refused (#497).

One dependency rather than a follow-up: `FileCacheSourceS3.iterdir_metadata`
issues one `list_objects_v2` and reads that single response, so an S3 prefix
holding more than a thousand objects lists short and says nothing
(SETI/rms-filecache#65). The seam's completeness guarantee assumes a listing
is complete or raises, and there is deliberately no workaround here. GS
auto-paginates and is unaffected.

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
  several of them. The task files such a run needs, and the instance
  startup script it runs under, are scripted in `cloud_support/`.
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
  debt with the widest blast radius. In the local integration environment 10
  of 75 sidecars disagree, each one attributed and owned, but the set is
  still wider than the deliberately-pinned one below, so the regression
  instrument cannot yet be read as clean. Two consequences to hold
  onto: a navigation-affecting branch can only be gated on *no new failures
  against `main`*, and #547 (the one place a built product is compared
  between the results tree and the results index) is blocked because the
  frame it needs no longer navigates. Reconciling this is prerequisite
  to #483's re-ratchet and to reading the Track A cohorts with confidence.

  The intended steady state, and what the pins mean when it is reached: the
  standing red set reduced to the deliberately-pinned frames, each owned by
  an open navigation issue (#338, #346, #350). N1530185128 (#351) and W1444747627
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
- **#483** — re-ratchet the library pins the shift-equivariance fix (#447 /
  PR #484) moves. Gated on #288: pins cannot be re-ratcheted honestly
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
- **#545** — the README names two of fourteen command-line programs and
  links to no guide. Cheap, and it is the first page anyone reads.
- **#549** — four technique guides name a `dt_fitting.py` that is a package.
- **#470 / #471** — config placeholder comments and config placeholders
  themselves still carry an internal phase codename; documentation and
  configuration should be self-contained.
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
logger-per-image design will find it), #494 Cassini BOTSIM pairs defined two
different ways (one definition, before #27 builds on either), #518 state
the encoding wherever a document or text file is read or written, and #552
remove the `AttrDict` `_IS_IMMUTABLE` marker once oops confines its
mutability bookkeeping to its own objects.

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
