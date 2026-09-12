# RMS-NAV Validation and Calibration Plan

*A concrete, sequenced plan to close every issue raised in
`critiques/archive/SCIENTIST_REVIEW_CRITICAL_2026-06-19.md`. Each workstream states the problem, the
goal, the specific tasks (with the files involved), the acceptance criteria that
define "closed," dependencies, and risk. A traceability matrix at the end shows
each critical-report finding mapped to the workstream(s) that retire it.*

---

**Premise.** The core architecture is stable; this plan is *calibration and
validation*, not architecture change, so the validation effort targets stable
code. The remaining open content — growing the curated image library and the
real-anchored confidence calibration — is exactly the WS-3/WS-5 territory
below, sequenced by `plans/PROGRAM_PLAN.md` Track A. There is **no fixed accuracy specification** —
the goal is to characterize the best accuracy the system actually achieves,
honestly and with its uncertainty, not to clear a numeric bar.

## Guiding principles

1. **Validation is the highest-value work, and the hardest.** Absolute accuracy
 can only be characterized in a de-circularized, realism-validated simulation; on
 real data we can measure *consistency*, not accuracy. This is the core of the
 plan — but it does **not** block the independent honesty/ops fixes (docs,
 provenance, degenerate-rotation, performance), which run in parallel and depend
 on nothing here.
2. **Claim only what the statistics support.** Report pairwise *combined* error by
 default; claim *per-technique* accuracy only where the system is identifiable and
 the techniques' bias-independence has been *verified* (WS-0), never assumed. Every
 reported number carries an uncertainty interval. (Note on notation: the recovered
 error quantity is a 2x2 **covariance**, not a scalar — see WS-0/WS-1. Where this
 document writes "σ" it is shorthand for the relevant covariance element, e.g. the
 radial-radial element on the ring axis; it never implies an isotropic scalar.)
3. **No output field without a calibration behind it.** Anything that looks like a
 probability or a quality grade must either be calibrated against real data or
 be explicitly labelled provisional.
4. **One source of truth for "what works."** A capability matrix, enforced by
 tests, replaces scattered prose claims.
5. **CI must exercise what we ship.** If real-image navigation is the product,
 real-image navigation must run automatically.
6. **Scope honestly.** Where a capability won't be built soon (e.g. PDS4 input,
 which waits on archives that do not exist yet), the claim is removed from the
 user-facing docs and demoted to a roadmap, not left as an implied feature.
7. **Validation can fail.** It may reveal accuracy worse than hoped. With no spec
 to pass/fail against, a poor result is not a gate — but it feeds the algorithm
 workstreams (e.g. WS-10), it is not merely reported and shrugged off.

---

## Relationship to `plans/PROGRAM_PLAN.md`

`plans/PROGRAM_PLAN.md` is the top-level plan of record across all remaining
work. This plan is its Track A detail layer: PROGRAM_PLAN says *when* the
validation and calibration work runs relative to everything else; this plan
says *how* and defines what "validated" and "calibrated" mean. Where the two
overlap, the methodology here is binding and the cross-track ordering there is
binding.

**Current baseline.** The confidence formulas and tier boundaries are
*sim-anchored* — fitted against simulated planted-truth recovery by the
`util/calibration/` tooling — and `confidence_provisional` stays true
pending the real-anchored pass (#230). The remaining workstreams below build
on that baseline; each has a tracking issue (cross-map below).

**One prerequisite for measuring anything at scale.** It is Track E work
rather than methodology, and it silently corrupts what every workstream below
reads:

- **#288** — 10 of 75 curated library sidecars disagree in the local
  integration environment, so the regression instrument that is supposed to
  detect a navigation change cannot currently do so. Until it is reconciled,
  a navigation-affecting change can only be gated on *no new failures against
  `main`*, and the library's own tiers cannot be read as verified
  expectations.

It does not block writing methodology or building tooling. It blocks
collecting the numbers.

**Two shared decisions, declared once:**

- **Confidence-calibration methodology (binding for #230).** Confidence is
  calibrated per WS-5: per-technique reliability diagrams against measured error
  anchors (WS-1 per-technique covariance where identifiable, pairwise combined
  covariance elsewhere, WS-2 sim recovery error where no multi-object cohort
  exists), a monotonic calibration map, tiers defined by error percentiles, and
  the calibration basis recorded per emitted value. Operator-assigned sidecar
  tiers are plausibility cross-checks and regression expectations, never fit
  targets.
- **Library-size target (binding for #172 and WS-3).** One library, two stages:
  the 47-image curated library (#172, workflow `plans/COHORT_CURATION_PLAN.md`) is
  the first stage — it seeds regression baselines and the initial calibration
  cohort; WS-3 then grows the same library to >=20 images per instrument, >=120
  total, which is the size the WS-1 agreement study needs. The 47-image stage is
  a milestone inside the WS-3 target, not a competing number.

**Workstream-to-issue cross-map** (issues carry the trackable work; workstreams
carry the methodology and acceptance criteria):

| Workstream | GitHub issue(s) | Note |
|---|---|---|
| WS-0 | #358, #359, #360, #361 | Estimator proven on sims (identifiability + bias-independence); residual is the real-frame reliability-vs-error coupling and PSF-coupling probes feeding WS-1. |
| WS-1 | #225 | The agreement study. The statistics system reports metadata statistics and consumes WS-1's per-frame disagreement metric; it is not the agreement study. |
| WS-1b | #226 | Reprojection consistency. |
| WS-2 | #227 (+ #223, #309, #341, #377, #409) | De-circularization done; #227 open only for the realism residual (realism-anchored calibration #309, terminator verdict #223, sim-fidelity gaps, and the scene-coordinate convention split #409 where stars are the outlier). |
| WS-17 | #355 | Distortion measured from star fields; residual is the per-camera Voyager sim split. |
| WS-3 | #172, #174, #235, #288 | 47-image stage first (#172), then the >=120 growth target (#235); discovery/review workflow in `plans/COHORT_CURATION_PLAN.md`. #288 is the standing regression: 10 of 75 sidecars disagree locally, so the library's tiers are not currently verified expectations. |
| WS-4 | #229, #426, #324, #336, #335, #340 | CI integration tiers, plus what does not run in Actions today: the agreement-estimator tests (#324), the data-independent simulator suites (#336), the committed sim baselines that have no canonical environment (#335), and the cross-check's yes/no primary-technique flag (#340). |
| WS-5 | #230, #176 | Real-anchored recalibration once WS-1 anchors exist. |
| WS-6 | #231 | Capability matrix. |
| WS-7 | #397, #398, #399, #400, #401, #402, #403, #404, #405, #407 | Titan haze navigation delivered and validated; open items are the deferred refinements and the operator ratification bundle (#407). |
| WS-8 | #53, #67 | Output bundles for all four instruments (required). PDS4 input (#34) is availability-contingent and not required for completion. |
| WS-9 | #233, #130, #176 | Measured star SNR + sensitivity tests (#233); constants inventory (#176); limiting magnitudes (#130). |
| WS-10 | #150, #128 | Limb bias root cause and redesign. |
| WS-12 | -- | Per-instrument guide chapters. Delivered; each instrument's workstream now updates its own pair. |
| WS-13 | #234 (+ #153) | Detector-noise model for the I/F render path. |
| WS-15 | #236, #103, #134, #126 | Thread safety, profiling, batch-parallel throughput. |
| WS-18 | #232 (+ #28, #66 partial) | End-product accuracy checks. |

---

## Phase 0 — Establish what validation is actually possible

> **Premise (the hard truth).** Absolute, subpixel ground truth for the real
> pointing of an archival Cassini/Voyager/Galileo/NH frame **does not exist**. No
> independent instrument measured where that camera truly looked to a hundredth of
> a pixel, and the obvious proxies do not supply it:
>
> - **Star plate-solving** (Gaia/UCAC) recovers camera *attitude* only. Stars are
> at infinity, so they are blind to the spacecraft-relative-to-body geometry
> (SPK + target ephemeris) that places bodies and rings — i.e. blind to exactly
> the techniques that most need validating. It also fails on the bright-body
> frames where those techniques run (no usable stars), and Voyager/Galileo
> distortion turns the solve into a fight where the distortion model is the
> confound.
> - **Repeat/overlap "predict B from A"** does not work: inter-frame jitter means
> the two frames have *different* true pointings, so there is no shared known
> quantity. Only a *physical* reprojection-agreement variant survives (below),
> and only as a differential check.
> - **Legacy comparison** is same-lineage and the old subpixel layer was never
> trusted (it is why the rewrite exists). It catches pixel-scale regressions; it
> has no authority on subpixel truth.
> - **An independently navigated archive answer** exists for a few observations:
> the published F ring bundle records, beside every reprojected product, the
> boresight that project navigated the frame to and how it navigated it, and it
> is the one real-frame comparison this pipeline had no hand in. It is not truth
> either. Its own file labels each frame Stars, Ring and/or Satellite Models, or
> Manual, and only the first is worth believing to well under a pixel, so a
> disagreement with one of the others is a disagreement rather than an error; and
> it covers a handful of observations rather than a cohort. What it does supply
> is a per-frame check that a run's own confidence cannot give, which is why it
> is worth running where it reaches: `util/nav_verification/` holds the
> comparison. Any comparison against it has to separate the constant from the
> per-frame part before quoting a number, because the two pipelines disagree by
> a fixed half pixel in both camera axes on every frame of every observation
> checked -- a difference of pixel datum rather than of pointing, which left in
> makes the per-frame disagreement look ten times larger than it is.
>
> Therefore the strategy is: **measure absolute accuracy only in a simulation that
> is (a) independent of the navigator and (b) proven realistic against real
> images; and on real frames, measure *consistency*, not accuracy** — stating
> plainly the common-mode error that consistency cannot rule out. The honest
> deliverable is "recovery error on a realism-validated, de-circularized sim,
> corroborated by real-image consistency, with a stated residual common-mode
> bound" — not a single "X px" accuracy claim.

### WS-0: Validate the validator (identifiability + bias-independence)
**Closes:** the unproven pillars under WS-1 — that the variance system is solvable
in the form it is actually used (full 2x2 covariances, not scalars), and that the
compared techniques have independent biases through their *shared preprocessing* —
before any real-image per-technique number is trusted.

**Problem.** WS-1's per-technique covariance rests on (a) the variance-components
system being identifiable, (b) the compared techniques having uncorrelated errors,
and (c) the recovered quantity being a covariance, not a scalar σ. If any fails,
"per-technique σ" repeats the simulator's original sin: agreement masking a
*shared* bias. The estimator math can be checked on any truth-known sim now; the
*real-world* correlation verdict additionally needs the realistic sim (WS-2) and a
real-data cross-check (see WS-1 over-determination).

**Stage 0a — estimator + identifiability (needs only a truth-known sim; runs in
parallel with WS-2, on the existing sim).**
- **Recover-known-error test.** With each technique's true error known, run the
 WS-1 covariance-components solve and confirm it recovers per-technique
 *covariances* (full 2x2, see WS-1 fix below), for each cohort composition.
- **Identifiability map.** Enumerate, per composition, what is recoverable:
 limb+disc *alone* gives only the combined covariance (one matrix equation, two
 unknown matrices — **not** separable); **limb+disc+ring** recovers only the
 **radial-radial element** of each body technique's covariance (the ring is rank-1)
 and only after the limb's anisotropic covariance is projected onto the radial
 axis — which **rotates frame to frame**, so the bin must also be cut by
 limb-orientation-relative-to-radial, or the solve carried in full matrix form;
 **multi-body** frames add the orthogonal axis. This map governs where WS-1 may
 report per-technique covariance vs only combined pairwise covariance.

**Stage 0b — bias-independence (needs the realistic WS-2 sim).**
- **Inject a shared bias *through the shared preprocessing layer*.** The DT
 techniques consume common per-image products (`image_derivatives` gradient/edge
 distance-transform, the single noise-σ estimate, the reliability gate); a bias in
 that layer shifts limb, terminator, **and** ring-edge together. WS-0 must inject
 the bias *there*, not into each technique separately, or it will not reproduce
 the real coupling. Confirm the solve detects the induced correlation rather than
 hiding it.
- **Two pivotal pairs, not one.** `Var(limb−disc)` is the *base* equation, so
 **limb-DT vs disc-NCC** independence is as load-bearing as **limb-DT vs
 ring-DT**: if disc and limb share a PSF edge bias (an NCC disc-template peak also
 tracks the PSF-blurred edge), per-technique covariance is unrecoverable
 *everywhere*, not just on the ring leg. Measure cov for **both** pairs; treat
 *no* pair as "clean" until measured.
- **Caveat — this verdict is the sim's.** It is only as real as WS-2's realism;
 the independent real-data check is WS-1's over-determined-frame closure test.

**Acceptance criteria.**
- The covariance solve provably recovers known per-technique covariances on the sim.
- A published identifiability map states, per composition, what is recoverable
 (which covariance elements, on which axes) vs only combined pairwise covariance.
- cov is measured for limb-disc *and* limb-ring (and other joined pairs) by
 injection *through the shared preprocessing layer*; no pair is assumed clean;
 failing pairs are excluded from joint solves.

**Dependencies:** Stage 0a needs only a truth-known sim (parallel with WS-2);
Stage 0b needs WS-2 (realistic sim). Gates the per-technique claims in WS-1 and
WS-5. **Risk:** medium-high — the binding outcome is whether the
DT techniques are bias-independent *through the shared preprocessing*; if limb-disc
or limb-ring are correlated, WS-1 falls back to pairwise combined covariance for
most regimes (an honest outcome, not a failure).

### WS-1: Multi-object cross-technique agreement (the primary real-image accuracy test)
**Closes:** "no real-image accuracy assessment." This is the one method that
yields a real-image accuracy signal without external ground truth, which does not
exist for these frames.

**Per-instrument applicability (decides what is even possible).**
- **Cassini ISS:** body+ring common, resolved-moon frames common, and *dozens* of
 star+body / star+ring tie-points exist ⇒ both relative/algorithm σ at scale and
 an absolute-attitude anchor are achievable. The strong case.
- **NH LORRI:** star-frame count to be verified (WS-3 task). LORRI optics are
 low-distortion, so even if star tie-points are few, distortion is a minor
 worry; absolute-attitude anchoring depends on that count.
- **Voyager ISS / Galileo SSI:** effectively *zero* usable star frames. There is
 therefore **no real-data inertial reference**: only relative/algorithm agreement
 (intra-body, body+ring) is available, which is blind to common-mode CK/distortion.
 **Absolute attitude accuracy for these two is sim-only (WS-2)** and must be
 reported as such; distortion is taken from the literature (see WS-17).

**Core idea.** Find real frames navigable by two or more fiducials — a star field
**and** a moon, a moon **and** a ring edge, two moons, or the same moon by limb
**and** disc **and** terminator. Compute each technique's offset and compare. The
shared quantity is the pointing offset itself (the signal); a small disagreement is
evidence both are accurate to roughly that level. Two independence caveats keep
this honest, and both are real:

- **The techniques are not as independent as "different fiducials" suggests.** The
 DT techniques share preprocessing — the `image_derivatives` gradient/edge
 distance-transform, the single noise-σ estimate, the reliability gate — built once
 per `NavContext`. A bias in that shared layer moves limb, terminator, and
 ring-edge together. "Run each technique independently" (not just taking the
 orchestrator's fused pick) is necessary but **not sufficient**: the shared
 preprocessing is a common-mode coupling that WS-0 must probe and that the report
 must treat as the prime suspect for any agreement-masking-bias. Where feasible,
 recompute the agreement with independent preprocessing per technique to break the
 coupling for the measurement.
- **Agreement is evidence of accuracy only between *verified* bias-independent
 techniques** (WS-0); a shared PSF edge bias can move two techniques the same way.

This is why WS-1 reports plain pairwise disagreement everywhere, and per-technique
covariance only on the narrow footing WS-0 certifies.

**Two interpretations, depending on whether the SPICE geometry is shared:**

- **Same target, different technique** (limb vs disc vs terminator on one moon;
 two edges of one ring): SPK + body ephemeris are identical and *cancel* in the
 difference, isolating the pure **algorithm/fitting error** — the cleanest
 technique-accuracy measure.
- **Different fiducial class** (star vs moon, moon vs ring): the star correction
 touches attitude only while the moon correction also touches SPK + ephemeris, so
 their disagreement additionally contains the **ephemeris error**. This measures
 end-to-end "is the body really where we put it?" accuracy and *bounds SPK error*
 — but an SPK-driven disagreement must not be misread as a technique being
 inaccurate.

**Archive reality (drives the design).** In the Cassini set, **body + ring frames
are common; frames with stars alongside a body or ring are uncommon; star + body +
ring together is rare.** So per-frame triple collocation cannot be the workhorse,
and the absolute-attitude reference (stars) is the scarce resource. Per-technique
σ is therefore separated three ways, leaning on the common cohorts and spending
the scarce stellar frames only where nothing else will do:

- **Route 1 — intra-body (common, free of ephemeris).** A single resolved moon is
 navigable by *limb*, *disc*, and (at phase) *terminator* / *blob*. Same body ⇒
 identical SPK + ephemeris ⇒ they **cancel**, leaving pure algorithm error.
 Available on every resolved-moon frame, including the common body+ring ones. The
 candidate base pair is **disc (NCC) vs limb (DT)** — *if* WS-0 certifies them
 bias-independent (not assumed; an NCC disc peak can track the same PSF-blurred
 edge as the DT limb). limb vs terminator are correlated (shared DT/PSF bias) and
 never separate it. **Only valid on round, photometrically bland bodies** — see
 the body-shape caveat below.
- **Route 2 — body + ring pairwise (the bulk cohort).** Common ⇒ large N. Compare
 along the (usually rank-1) ring-radial axis. A moon and Saturn's rings do **not**
 share SPK, so the disagreement is σ_body_algo² + σ_ring_algo² **plus a per-moon
 ephemeris bias** — the ephemeris term is a fixed offset, not a variance (see
 below), and is handled as a nuisance parameter, not folded into a σ.
- **Route 3 — star + body / star + ring (scarce tie-points).** The only
 attitude-only, inertial reference, and it does two jobs. (1) It *anchors absolute
 attitude*: stars give true δ_CK. (2) On a star+body frame, `(body − star)`
 cancels δ_CK and so **directly measures that moon's ephemeris error**
 δ_SPK(moon) (up to the body-algorithm noise) — which is exactly the per-moon
 nuisance bias Route 2 needs. So the scarce star+body frames are the cleanest
 *ephemeris* probe, not only an attitude anchor. What a single such frame *cannot*
 do is cleanly split body-algorithm error from SPK without importing
 σ_body_algo from Route 1/WS-0; treat the per-moon ephemeris estimate as carrying
 the body-algorithm noise until that import is available.

**Estimation strategy — pairwise by default, per-technique only where earned.** The
default output is the *pairwise* disagreement of two techniques on one frame. It is
the least assumption-laden product — but **not** assumption-free: reporting one
combined covariance per bin still assumes within-bin stationarity (below), so
binning a continuous dependence inflates it. It bounds only the *combined* error
(Σ_A + Σ_B), not either alone. Separating per-technique error needs a
covariance-components / three-cornered-hat solve, reported **only where WS-0
certifies the cohort**:

- **The recovered quantity is a 2x2 covariance, not a scalar σ.** A limb fit's
 covariance is strongly *anisotropic* (tight perpendicular to the arc, loose along
 its tangent); a disc fit is near-isotropic. The identities therefore live in
 matrix form: `Cov(limb−disc) = Σ_limb + Σ_disc` (clean, same body), and the
 three-way solve recovers covariance *matrices*. Writing them as scalars is wrong.
- **Identifiability.** A resolved moon *alone* gives only `Σ_limb + Σ_disc` — one
 matrix equation, two unknown matrices — **not** separable. A **body+ring frame
 with a resolved moon** carries three estimators, but the ring is **rank-1
 (radial)**, so it constrains only the **radial-radial element** of each body
 technique's covariance. Worse, the limb's anisotropy axis rotates relative to the
 fixed radial direction *frame to frame*, so the projected radial variance is not a
 bin-constant: the bin must additionally be cut by **limb-orientation-relative-to-
 radial**, or the solve carried in full matrix form across frames. The clean
 scalar story (`Var(limb−ring)−Var(disc−ring)=σ_limb²−σ_disc²`) holds only as the
 radial-radial component, after that geometry is controlled. Both the
 moon-ephemeris term and Σ_ring cancel in that subtraction (verified for δ_SPK
 uncorrelated with the algorithm errors). The **orthogonal axis** needs a second
 body (multi-body) or a star.
- **Bias independence — two pivotal pairs.** `Cov(limb−disc)` is the *base*
 equation, so **limb-DT vs disc-NCC** independence is as load-bearing as **limb-DT
 vs ring-DT**: if disc and limb share a PSF edge bias, per-technique covariance is
 unrecoverable everywhere, not just on the ring leg. WS-0 measures cov for both
 pairs *through the shared preprocessing layer*; **no pair is assumed clean.**
 limb-DT/terminator-DT are correlated and never share a solve.

Within a qualifying bin, frames are pooled assuming each technique's error
covariance is stationary across the bin (binned by resolution / phase /
lit-fraction **and** limb-orientation-vs-radial). **Stationarity is assumed by the
pairwise product too**, not only the separation — so it is the broad limiter, not a
per-technique-only one; spot-check each bin (e.g. that its disagreements look like
one distribution, not a mix), and where it fails, shrink the bin or report the
disagreement distribution itself rather than a single covariance. Every estimate
carries an **uncertainty interval** reflecting the bin's sample size; cohorts grow
until those intervals are as tight as the available frames allow (no target
precision to size against — see Premise).

**Ephemeris error is a per-target bias, not noise.** A moon's SPK error at a given
epoch is a fixed offset, not a zero-mean random variable, so it cannot be a
variance term. In cross-class comparisons it is modelled as a **per-moon, per-epoch
bias** and either (a) estimated and removed using several frames of the same moon
over a short span where the bias is ~constant, before the variance solve, or
(b) carried as a separately-reported bias term — never silently absorbed into a
technique's σ.

**Body-shape caveat — different techniques measure different "centers."** The solve
assumes limb, disc, and terminator estimate the *same* offset. On a perfect uniform
sphere they do. On a real body with topography, limb-darkening, or albedo patches,
the geometric-limb center (limb fit) and the brightness-weighted template center
(disc NCC) converge to points that differ by a *real physical amount* — not an
error in either. That genuine difference lands in `Cov(limb−disc)` and is misread as
algorithm error, inflating the recovered covariances in a target-dependent way. So
the intra-body separation is valid **only on round, photometrically bland bodies**;
irregular or high-contrast bodies are excluded from per-technique separation (their
pairwise disagreement is still reported, flagged as shape-contaminated). The
qualification gate is **not a judgment call**: derive it from the existing static
body-shape table (`ellipsoid_rms_residual_km`, `crater_scale_km`, `albedo_variation`),
with explicit thresholds, so "round and bland enough" is a reproducible,
data-sourced criterion. This shrinks the qualifying cohort further. WS-2's
shape/photometric sweeps quantify the effect on the sim, but on a real frame it
cannot be separated from algorithm error.

**Over-determination is the one real-data check on all of the above.** A frame with
limb+disc+ring+second-body makes the covariance system *over-determined*. The
residual of that over-fit is the only test of bias-independence, stationarity, and
the ephemeris model that runs on **real photons** (WS-0's verdict is the sim's, and
inherits WS-2's realism). If the over-determined equations do not close within
their uncertainty intervals, an assumption is violated — and the report must say
which numbers are thereby suspect. Curate and exploit over-determined frames
specifically for this closure test; it is the real-data counterpart to WS-0.
**Caveat: this check may itself be data-starved.** Four-fiducial frames
(limb+disc+ring+second body) are rarer than the star triples already called rare,
and for Voyager/Galileo (no stars, perhaps few multi-moon+ring frames) there may be
too few — or none — to run it. Where the closure test cannot be populated, the
assumptions remain **sim-verified only** (WS-0), and the report must say so for that
instrument rather than implying a real-data check that did not happen.

**Robustness — variances are outlier- and failure-dominated.** Covariance estimates
are exquisitely sensitive to a few mis-navigated frames, and a technique that
*occasionally fails catastrophically* (returns garbage) produces a wildly different
"covariance" than one with consistent small error. Use robust covariance estimation
with explicit outlier rejection, and **report each technique's failure rate
separately from its error covariance** — a covariance without a companion failure
rate is misleading. A "failure" (no/garbage result) and a "large error" are
different products and are reported as such.

**Population, not per-frame.** WS-1 yields *bin-level* statistics pooled over many
frames; the navigator emits a *per-frame* covariance from each fit's information
matrix. So WS-1 validates the **average scaling** of the reported error bars over a
regime — does the per-frame covariance, aggregated over the bin, match the empirical
spread? — not whether any *single* image's error bar is correct. A scientist quoting
one image's uncertainty therefore gets population-level assurance for that regime,
not per-image verification. State this explicitly; it bounds what the whole method
can promise.

**The irreducible blind spot.** Even with bias independence verified, the method is
blind to error that shifts *all* compared features *identically* — a true
common-mode CK shift. Because body+ring agreement is purely *relative*, a
CK error common to a moon and a ring edge is invisible to it; only the scarce
inertial (star) frames and the sim catch that. (Distortion is **not** in this
blind spot — it acts *differently* at different field positions, so it shows up in
the disagreement and is handled as a contaminant in WS-17, not as a hidden term.)
**Net: relative / algorithm accuracy is characterized at scale; absolute attitude
accuracy rests on the small stellar tie-point set plus the de-circularized sim
(WS-2).** The report must say exactly this and not let the large body+ring N imply
absolute accuracy.

**Coverage caveat (state it).** This characterizes only regimes where ≥2
techniques are *simultaneously feasible*. A limb that fills the FOV and drowns the
stars, or a lone faint moon with no second fiducial, is never cross-checked; those
single-technique regimes remain characterized only by the sim (WS-2).

**Tasks.**
- Curate cohorts in `tests/integration/image_library/` (extends WS-3), each tagged
 by fiducial set, SPICE-sharing class, and per-technique bias mechanism:
 resolved-moon frames (Route 1, large), body+ring frames (Route 2, large — the
 ring-radial separation route), **multi-body frames (two+ resolved moons,
 common)** — which supply the *orthogonal* axis the rank-1 ring cannot and a
 direct probe of inter-moon ephemeris error (note two limbs share the limb bias,
 so they complete axes and measure relative ephemeris, they do not reveal a shared
 limb bias), and star+body / star+ring tie-points (Route 3, small but deliberately
 collected, for absolute attitude). The richest single frame is **two resolved
 moons + a ring edge**. Genuine star+body+ring triples are recorded when found but
 not depended on.
- Build `tests/integration/agreement/` harness: per frame, run every feasible
 technique independently (and, where feasible, on independent preprocessing to
 break the shared-`image_derivatives` coupling for the measurement) and record all
 pairwise disagreements as **2x2 covariances** with their SPICE-sharing class,
 body-shape class (round/bland vs irregular/high-contrast), and
 limb-orientation-vs-radial. Report **pairwise combined covariance everywhere**;
 run the covariance-components solve for **per-technique** covariance only on the
 bins WS-0 qualifies, excluding bias-dependent pairs and shape-contaminated
 bodies, with the per-moon ephemeris modelled separately. Use **robust** estimation
 with outlier rejection, and record each technique's **failure rate** separately.
 Run the **over-determined-frame closure test** as the real-data assumption check.
 Attach an uncertainty interval to every estimate.
- Produce `docs/agreement_report/agreement_report.rst`: per-instrument, per-regime
 pairwise disagreement distributions (always); per-technique covariance with
 uncertainty intervals only where identifiable and shape-clean; per-technique
 failure rates; the over-determination closure result; the relative-vs-absolute
 split; and the identifiability, anisotropy/projection, stationarity,
 bias-independence (per WS-0), body-shape, ephemeris-bias, and coverage caveats.
- **Wire it into the live product (not just offline):** surface per-frame
 cross-technique disagreement in `_metadata.json` as an empirical uncertainty / QA
 flag (within expected combined σ → trust; beyond it → flag for review), computed
 from the `NavResult.per_technique` results the orchestrator already produces.
 This per-frame signal is also the real-data anchor for WS-5.

**Acceptance criteria.**
- A published agreement report whose every number comes from real frames: pairwise
 combined covariance everywhere; per-technique covariance (with uncertainty
 intervals) only on WS-0-qualified, shape-clean bins; per-technique failure rates;
 an explicit relative-vs-absolute split.
- No per-technique covariance is reported for a non-identifiable bin, a
 bias-dependent pair, or a shape-contaminated body; ephemeris is reported as a
 separate bias, never inside a covariance.
- The over-determined-frame closure test passes (or the violated assumption and the
 numbers it taints are named).
- Absolute attitude accuracy is reported only from the stellar tie-points + sim,
 never implied from the body+ring cohort.
- Multi-object frames emit a per-frame cross-technique disagreement metric into the
 metadata, consumed by WS-5.

**Dependencies (split — this matters).** The **pairwise combined-covariance**
product — the bulk, most-trustworthy deliverable — needs only **WS-3** (cohort) and
**WS-17** (distortion correction); it does **not** depend on WS-0 or WS-2 and should
ship without waiting for them. Only the **per-technique separation** gates on
**WS-0** (identifiability + bias-independence) and, through it, the realistic
**WS-2** sim. Do not hold the pairwise product hostage to the riskiest workstreams.
Feeds WS-5. **Risk:** medium-high for the *per-technique* layer — the compounding
risks are that WS-0 finds the DT techniques bias-correlated through shared
preprocessing, that the shape-clean + geometry-controlled qualifying cohort is
small, and that star tie-points are too few to anchor absolute attitude; the
realistic end state may be "pairwise combined covariance at scale, per-technique
covariance only on a curated set of round uniform moons in favorable
ring/multi-body geometry." The pairwise layer carries low risk and no external-data
need.

**The anchor must not be contaminated before it is measured (#558).** This
workstream's product is cross-technique *disagreement*, and a technique that
reports a confidently wrong answer contributes a disagreement that is a property of
the defect rather than of the scene. The confidently-wrong ring locks
(#346, #476) are the known case: on `N1633925572_1_CALIB`, measured 2026-08-26, a body fit
and a star fit agree to 0.08 px while `RingEdgeNav` converges 39 px away on 452 of
6786 inliers, is not flagged spurious, and enters the combine. Feeding that frame to
this workstream does not merely add noise -- it moves the anchor WS-5 then
calibrates against, so a calibration fitted before those defects are closed learns
to distrust ring evidence everywhere and has to be redone once they are. Either fix
that family first, or exclude from the agreement product any frame whose
contributing technique self-reports an inlier fraction below a stated floor, and say
which was chosen.

### WS-1b: Reprojection consistency across overlapping frames (secondary corroboration)
**Closes:** the one regime WS-1 cannot reach — a body imaged repeatedly with **no
second in-frame fiducial** — so it is not redundant with WS-1.

For overlapping real images of one body, navigate each independently and check that
a fixed surface feature reprojects to the same body-fixed lat/lon (rings: same
radius/longitude). Inter-frame jitter is fine — each frame carries its own pointing;
we check surface-fixed agreement, never one frame's offset predicted from another.

**Unique value vs WS-1 and WS-18 (stated to avoid double-counting):** WS-1 needs ≥2
fiducials *in one frame*; WS-1b works across frames that each have only one body, so
it covers single-fiducial sequences WS-1 skips. It is the *navigation* counterpart
of WS-18's *mosaic-product* seam check — same arithmetic, different object: WS-1b
checks the navigated offsets, WS-18 checks the assembled mosaic. They share the
implementation (build it once, apply twice). *Blind spot:* error common to both
frames (shared SPK / timing) cancels undetected, so this corroborates WS-1 and never
substitutes for it.

**Acceptance criteria.**
- A reprojection-consistency harness exists that, given overlapping frames of one
 body (or ring region), navigates each frame independently and reports the
 body-fixed (lat/lon or radius/longitude) scatter of matched surface features,
 with an uncertainty interval per sequence.
- The published results cover at least one single-fiducial sequence per
 instrument where the archive provides one (documented as absent where it does
 not), binned by resolution, and state the shared-SPK/timing blind spot
 alongside every number.
- Sequences whose scatter exceeds the combined reported per-frame σ beyond the
 expected rate are flagged and fed into WS-1/WS-10 as inputs, not dropped.
- WS-18's mosaic seam check consumes the same implementation (asserted by a
 shared-module test), so the two checks cannot drift apart.

**Dependencies:** WS-3. Shares implementation with WS-18. **Risk:** medium.

### WS-2: Prove the de-circularized simulator realistic (primary accuracy instrument)
**Closes:** "the headline accuracy numbers are circular." This is the *primary*
source of any absolute accuracy number, since real truth is unobtainable.
**Tracked by:** #227, open only for the realism residual below.

**Delivered (the de-circularization half, on main).** The image side no longer
shares a rendering routine with the navigator: an independent forward model, the
truth/idealized information partition guarded by an import/call-graph test,
model-error sweep axes, full detector noise including the I/F path, and the
rewritten simulator report. Sim recovery error is therefore no longer pure
self-consistency.

**The realism residual (why #227 stays open).** Independence is necessary but not
sufficient — the image side must also be **realistic**, and that is not yet
underwritten: the calibration is not fitted on the realism-anchored renderer
configuration (#309), the terminator deliverable is degenerate with no realism
verdict (#223), realism is Cassini-only with the authored scene mixture
unvalidated (#309, #341), and the catalogued sim-fidelity gaps (#325-#345) and
single-annulus rings vs realistic nested ringlets (#377) remain. Closing #227 is
the operator's realism-verdict gate, gated on #309.

**Approach for the residual.** The navigator uses its *best available* model — do
**not** preserve a known-worse model just to manufacture a gap. The image side is
made *more* realistic, and the residual recovery error is the genuine model error
the real pipeline would also incur. Report error as a function of the *remaining*
mismatch, and separately validate that the simulated images are statistically
indistinguishable from real ones.

**Tasks.**
- **Independent, more-realistic image model.** Render the image with a forward
 model that is (a) implemented independently of the nav model (no shared rendering
 routine for any quantity under test) and (b) *more* realistic than the
 navigator's best model — higher oversampling, measured/empirical PSF (Airy +
 jitter + filter-dependent wings), topographic shape, real detector noise. The
 navigator uses its best model, not a deliberately handicapped one; the gap under
 test is the residual the navigator genuinely cannot model, not a hand-set delta.
- **Inject controlled model error** as sweep axes in `tests/integration/sim_sweep*`:
 PSF mismatch; shape mismatch (render with topography/high-res polyhedral
 `sim_body_polyhedral.py` + craters, navigate with the ellipsoid); photometric
 mismatch (render Lommel-Seeliger, navigate Lambert, etc.); ephemeris error
 (perturb planted body position to emulate SPK error); **full detector noise on
 every scene, including the I/F path** (WS-13).
- **Validate sim realism.** Tune the image forward model until simulated frames
 match real frames distributionally — noise statistics, PSF / encircled-energy,
 the gradient profile across a real limb, dynamic range, and artifact incidence —
 using the WS-3 real cohort as the reference. This match uses *any* real frames
 (it is a statistics comparison, not a pointing comparison), so it needs **no star
 frames** and is therefore achievable for Voyager / Galileo too — which matters,
 because for those instruments the sim is the *only* absolute-accuracy basis (see
 WS-1 applicability), making their realism match the load-bearing evidence. Where
 no accurate independent PSF/shape exists for an instrument (a real risk for
 Voyager/Galileo), say so and treat that instrument's sim accuracy as bounded by
 the unverified forward-model fidelity, not as measured.
- Recovery accuracy on the sim is only credible to the extent the realism match
 holds; report the match quality alongside the accuracy.
- **Rewrite `simulator_report.rst`** so each accuracy number is reported *as a
 function of model mismatch*, the zero-mismatch column is explicitly labelled
 "self-consistency floor (not accuracy)," and the realism-match evidence is
 presented as the precondition for reading the mismatch curves as accuracy.

**Acceptance criteria.**
- No quantity reported as "accuracy" is computed with image and model sharing a
 rendering function (asserted by a test on the import/call graph).
- The report presents error vs PSF/shape/photometric/ephemeris mismatch, the
 realism-match evidence, and the labelled self-consistency floor.
- The simulated-vs-real distributional match is *quantified and reported* per
 instrument for the features each technique consumes (a described figure of merit,
 not a pass/fail threshold — there is no spec; see Premise). Sim accuracy numbers
 are presented as credible only to the degree that reported match supports.

**Dependencies:** WS-3 (real frames to match against — the cohort, not the WS-1
analysis, so there is no cycle: WS-3 → WS-2 → WS-0 → WS-1). Gates WS-0. **Risk:** high — a credible independent + realistic image forward
model is the crux of the whole validation story.

### WS-17: Geometric distortion model validation (prerequisite for real-image agreement)
**Closes:** the confound that would otherwise inflate WS-1 disagreements (and any
plate solve), called out in the critical-report discussion of Voyager/Galileo
distortion.

**Problem.** WS-1 compares features at *different field positions*; ring/limb fits
span the frame. If the per-instrument geometric distortion model is wrong, two
*correct* techniques at different field positions disagree purely from distortion.
This is **not** a common-mode (hidden) term — distortion acts *differently* across
the field, so it shows up in the measured disagreement and *inflates apparent
technique error*. It must therefore be **corrected per feature-position before the
differences are formed** — not subtracted as a single aggregate "budget." Distortion
enters the key identity `Var(limb−ring)−Var(disc−ring)` as a per-frame vector
(distortion-at-the-limb-position minus distortion-at-the-ring-position); the clean
way to remove it is to apply the distortion model to each feature's pixel
coordinates first (same per-position rigor the plan applies to the ephemeris bias),
leaving only the model's *residual* uncertainty in the budget. Voyager and Galileo
are badly distorted; this must be pinned before their agreement numbers mean
anything.

**Tasks (split by whether star frames exist).**
- **Cassini (and NH LORRI if star frames exist):** validate/refine the distortion
 model from real star fields — a plate solve's residuals *as a function of field
 position* expose distortion directly (the one job star solves are good at, no
 spacecraft-position truth needed). Note the residual is not pure distortion: it
 also carries the star catalog's astrometric error (field-position-independent, so
 separable from distortion's smooth field pattern) **and the star centroiding error,
 which is *not* field-position-independent** — the PSF degrades toward the edges and
 corners, exactly where distortion is largest, so the two grow together and are
 *not* cleanly separable by their field dependence. Model the edge-dependent
 centroiding error explicitly (from the PSF map) and propagate its uncertainty
 rather than assuming it away. Then apply the distortion model per feature-position
 in WS-1 (above) and feed the field map into the per-instrument chapters.
- **Voyager / Galileo (zero star frames — the star route is impossible for exactly
 the most-distorted instruments):** there is no in-house way to validate
 distortion. **Adopt the published Voyager ISS / Galileo SSI geometric-distortion
 solutions from the calibration literature as given** (cited per the static-data
 policy), and use **body+ring agreement vs field position** as a coarse residual
 check only — if disagreement grows toward the frame edges, the adopted model is
 suspect. This check is entangled with other errors and is a sanity test, not a
 calibration.
- NH LORRI: low-distortion optics, so even absent star frames the distortion risk
 is small; document the adopted model and move on.

**Acceptance criteria.** Cassini (and LORRI if possible) distortion is quantified
from star residuals (with catalog error separated out and the edge-dependent
centroiding error modelled, not assumed flat) and applied **per feature-position**
in WS-1, leaving only its residual uncertainty in the budget. Voyager/Galileo
document the adopted literature distortion model with a field-position agreement
sanity check, and their agreement/accuracy claims are scoped to "literature
distortion assumed, unvalidated in-house."

**Dependencies:** WS-3 (star-field frames where they exist). **Risk:** high for Voyager/Galileo — distortion cannot be independently validated
there, which directly limits how far their real-image agreement numbers can be
trusted.

---

## Phase 1 — Make the safety net real

### WS-3: Expand the real-image regression cohort
**Closes:** "real-image regression rests on ~13 hand-blessed images."
**Tracked by:** #172 (the 47-image first stage; workflow
`plans/COHORT_CURATION_PLAN.md`) and #174 (integration tests + baselines);
the growth beyond that stage is #235.

**Tasks.**
- Complete the 47-image curated stage (#172), then grow
 `tests/integration/image_library/` to the target of **≥20 per instrument, ≥120
 total** — the size the WS-1 agreement study needs — spanning the geometry
 taxonomy already present (full-FOV body, partial overflow, below-resolution,
 high-phase terminator, multi-body, ring curved/flat, ring+body, star-dominated,
 faint stars, scattered light, negative cases).
- Expand `image_library/README.md` (currently a minimal schema/registry note)
 with the curation workflow, blessing/re-blessing procedure, and the
 ground-truth provenance for each entry.
- Add explicit **negative/failure cases** (unnavigable frames) and assert the
 system fails cleanly with the right status reason.

**Acceptance criteria.**
- Cohort size and per-category coverage meet the documented targets (47-image
 stage first, then ≥20 per instrument / ≥120 total).
- `README.md` documents schema + curation + blessing + provenance; every sidecar
 records its ground-truth source.

**Dependencies:** feeds WS-1, WS-7. **Risk:** low.

### WS-4: Run real-image tests in CI
**Closes:** "CI never runs integration tests."

**Problem.** `.github/workflows/run-tests.yml:94` runs `-m "not integration"`;
nothing real is exercised automatically.

**Tasks.**
- **Fast integration tier:** cache a small set of real images + the minimal SPICE
 subset (or stand up a read-through holdings cache) so a curated subset runs on
 every PR in a bounded time budget. Mark these `integration_fast`.
- **Nightly/weekly full tier:** run the complete integration + accuracy suite on a
 schedule (the repo already has a weekly cron) with the holdings/catalog env vars,
 uploading the accuracy report as an artifact and failing on regression beyond a
 tolerance band.
- Add an **accuracy-regression gate**: compare per-technique median/95th error
 against a committed baseline; fail if it degrades beyond threshold.
- Cover the committed sim renders. `test_render_diffs.py` is
 integration-marked, so nothing checks the committed `current/` gallery on a
 default run — and one of those renders is already stale on `main` without
 anyone noticing (#426). Fix the stale render, then decide which tier owns
 that check; it needs no network, so the per-PR sim tier can carry it.

**Acceptance criteria.**
- Every PR runs at least the `integration_fast` tier with real images.
- A scheduled job runs the full accuracy suite and gates on regression.
- A documented, reproducible mechanism supplies images + kernels to CI.

**Library consumers and CI tiers (binding note).** The image library has four
distinct consumers, and only the smallest of them ever runs per PR:

| Consumer | What runs | When |
|---|---|---|
| Sim tier (baselines + sim navigation/invariant tests) | fast, no network | every default `pytest` and every PR |
| `integration_fast` real-image subset (~5-10 cached frames) | bounded minutes | every PR |
| Full library regression + accuracy suite | the whole library | nightly/weekly schedule only |
| Calibration (WS-5) and agreement study (WS-1) | offline analysis producing reports and coefficients | on demand, never CI-gated per PR (only the cheap calibration-drift check joins the scheduled tier) |

The arithmetic forbids anything else: at ~35 s/frame a 120-image library is
over an hour of compute before downloads. Do not wire the full library, the
calibration sweep, or the agreement analysis into per-PR CI; the per-PR gate
is the sim tier plus the small cached `integration_fast` subset.

**Dependencies:** WS-1, WS-3. **Risk:** medium — kernel/holdings
provisioning in CI is the hard part; a cached fixture bundle is the mitigation.

---

## Phase 2 — Stop shipping unbacked numbers

### WS-5: Calibrate (or quarantine) confidence and tiers
**Closes:** "ships a confidence it admits is meaningless."
**Tracked by:** #230 (the real-anchored calibration — this workstream defines
its methodology, binding per the relationship section above) and #176
(constants into config, which lands before calibration writes coefficients).
The curated library's operator-assigned `confidence_tier` labels serve as
plausibility cross-checks and regression expectations for the calibrated
output; they are never fit targets.

**Problem.** `confidence`/`confidence_tier` are emitted per image; the sigmoid
coefficients and tiers are sim-anchored but not yet real-anchored
(`nav_technique/confidence*.py`, `confidence_config.py`,
`config_510_techniques.yaml`).

**Real-data anchor.** There is no per-frame "achieved pixel error" on real images
(no external truth). The anchors are, in order of strength: per-technique covariance
from WS-1 *where WS-0 says it is identifiable, bias-independent, and shape-clean*;
pairwise combined covariance elsewhere; per-frame cross-technique disagreement; and
the de-circularized sim (WS-2) for single-technique regimes WS-1 cannot reach.

**Calibration is therefore patchy by regime — say so.** A *per-technique* confidence
can be calibrated against a real anchor only where per-technique covariance exists
(the narrow WS-0-qualified, shape-clean, geometry-controlled set). Everywhere else
the only real anchor is *combined* pairwise covariance, from which a per-technique
confidence is under-determined — there the calibration falls back to the sim (WS-2)
and is labelled sim-anchored, not real-anchored. The result is a confidence whose
*calibration basis* varies by regime; the metadata records which basis each value
used. **Avoid the feedback loop:** the per-frame disagreement used to calibrate must
come from independently-run techniques, and the calibrated confidence must not then
re-weight those same techniques' ensemble fusion in a way that feeds the calibration
target.

**Tasks.**
- Build per-technique **reliability diagrams** against the anchor: reported
 confidence (and reported σ) vs the WS-1 per-technique error variance *on
 identifiable bins*, the pairwise combined σ on non-identifiable bins, and sim
 recovery error (WS-2) where no multi-object cohort exists. Carry the anchor's own
 uncertainty interval into the diagram.
- Fit a **monotonic calibration map** (isotonic regression or temperature/Platt
 scaling on the existing sigmoid) so reported confidence tracks empirical
 reliability, and so each technique's reported σ is consistent with its WS-1/sim
 error estimate (coverage check: does cross-technique disagreement fall within the
 combined reported σ at the expected rate?).
- **Redefine tiers** by WS-1/sim error percentiles, not by hand.
- Add a **calibration regression test** that recomputes the reliability diagram and
 fails if calibration drifts.
- **Interim safeguard (ship immediately, before calibration lands):** mark the
 field `confidence_provisional` in `_metadata.json` and document it as
 uncalibrated, so no user mistakes it for a probability in the meantime.

**Acceptance criteria.**
- Confidence is calibrated against the WS-1 anchor where per-technique covariance
 exists and sim-anchored elsewhere, with a published reliability diagram carrying
 the anchor's uncertainty; per-frame cross-technique disagreement falls within
 combined covariance at the expected rate.
- Each emitted confidence records its **calibration basis** (real-anchored vs
 sim-anchored) in the metadata; the regime patchiness is documented, not hidden.
- Calibration does not consume a target that depends on the calibrated confidence
 (no feedback loop), and tier boundaries map to stated error percentiles.
- A test guards calibration drift.

**Two limits on the method itself, to be carried rather than discovered.**
A *monotonic* calibration map cannot separate values that were collapsed onto one
number before it sees them, and `COMBINED_CONFIDENCE_CAP` does exactly that above a
threshold the ordinary good case reaches (#557): two results whose precision-weighted
mean confidence is 0.66 report the ceiling when both count as significant
corroborators and no second agreement group applied the disagreement penalty, which
is applied after the cap. So the confidence axis is censored from above over a range
that ordinary agreement reaches, and the map is fitted with that understood rather
than fitted through it. And the anchor is only as good as the techniques feeding it: a
confidently-wrong contributor moves the cross-technique disagreement this workstream
calibrates against (#558).

**Dependencies:** WS-0 (which bins yield per-technique σ), WS-1 (anchor, and
see #558 on keeping it uncontaminated), WS-2 (single-technique regimes). **Risk:**
medium — may reveal that some techniques' covariances are mis-scaled (feeds WS-9),
and the confidence axis may prove too censored to calibrate above the cap (#557).

---

## Phase 3 — Close the capability gaps (or scope them out honestly)

### WS-6: Reconcile claims with reality (capability matrix)
**Closes:** "unfinished rewrite wearing a finished manual," forward-looking
vapor.

**Tasks.**
- Author a single **capability matrix** (instrument × {navigate, backplanes,
 mosaic, PDS4 input, PDS4 bundle, manual}) with states {supported, partial,
 not-supported}, generated/verified from the registries and a test so it cannot
 drift from code. **A binary "navigate: supported" cell is not enough** — it would
 flatten the very nuance the rest of this plan establishes (Cassini's validated
 relative covariance + absolute attitude on dozens of frames vs Voyager/Galileo's
 sim-only absolute + literature distortion). Add a second **validation-status**
 axis per (instrument, technique): {relative-σ validated at scale, per-technique-σ
 on curated set, absolute-attitude anchored, sim-only}. The matrix must carry the
 accuracy-evidence tier, not just feature existence, and link to the WS-1/WS-2
 reports for the numbers.
- Sweep `docs/` for "placeholder / reserved for / pending / not yet implemented /
 future enhancement"; move each to a **roadmap** page or delete it. The user
 guide describes only what exists.
- Audit **reserved-but-unwired config keys** (curvature/roughness limb filters,
 ring fiducial promotion, `remove_body_shadows`, `min_emission_ring_body`,
 curvature reductions): either implement, or remove from shipped YAML and the
 docs so the config surface stops advertising absent features.

**Acceptance criteria.**
- A capability matrix exists, is test-verified, and is the single referenced
 source for "what works."
- No shipped config key is documented as functional unless the code consumes it.

**Dependencies:** light coupling to WS-8. **Risk:** low.

### WS-7: Titan navigation
**Closes:** "Titan is a no-op." **Status:** delivered and validated; the open
items are refinements and an operator ratification bundle.

**What ships.** Titan's opaque haze hides the surface, so shape-based
navigation is systematically wrong on it and the body model excludes it. In
its place, the haze solar-symmetry method (Hanson, French, Waugh, Barth &
Anderson, 2025, GRL, doi:10.1029/2024GL113415): absent clouds or visible
surface features a hazy atmosphere is mirror-symmetric about the image-plane
line through the body centre and the sub-solar point, so the shift
perpendicular to that line is the one maximising mirror symmetry, and a
FREE-radius circle fit to the sunward limb arc gives the shift along it
without assuming a haze altitude. The free radius is what makes the method
filter-independent — a wavelength-dependent haze top moves the fitted radius,
not the fitted centre — and no per-filter or per-phase training data exists
anywhere in the implementation. Delivered as a model emitting one `TITAN_LIMB`
feature, the `TitanHazeNav` technique wrapping a pure fitting library, a
simulated-Titan renderer with symmetry-breaking haze structure, and the
standing sweeps and campaigns that measure them.

**Deliberate design points.** Frame quality is reliability, never a refusal to
emit: a clipped, occluded, or too-small envelope scores exactly zero and the
standard per-type gate removes it, so a marginal Titan resolves through the
same statuses as any other marginal scene and carries an attributing gate
record. There is no Titan-specific status reason. The reported covariance is
strongly anisotropic by construction, because the cross-track direction is
genuinely far better determined than the along-track one.

**Stated bound (the acceptance criterion's answer).** Single-frame accuracy is
1 px or better cross-track and 3 px or better along-track. The bound is the
planted-truth clean-scene P95 (0.17 px cross-track, 0.82 px along-track on
the clean family of a 700-scene randomized campaign; artifact-injected
families run wider) confirmed by real-frame evidence: against an independent
star lock on the same frame the haze fit disagrees by 0.99 px rms cross-track
and 1.50 px rms along-track over nine pairs, implying about 0.70 and 1.06 px
per frame; repeat frames of one target through one filter agree to 0.34 px
and 0.33 px. Measured on an 82-frame Cassini cohort: 73.5% of `clean` frames
commit, every `clean`-frame refusal is attributed to a named gate, and no
adverse frame produces a confident-wrong lock that a witness contradicts.

**Open items.** The refinements are issues, not prose: the haze-radius table
that would remove the along-track/radius degeneracy and settle whether the
haze top is measurably wavelength-dependent (#397), CB3 surface-window
cartographic correlation (#398), a Voyager validation cohort (#399), ensemble
handling of the oblique covariance (#400), extreme phase (#401), ring
translucency (#402), arc ray reach (#403), a size-relative residual gate
(#404), and library growth through the standard curation pipeline (#405).
Three acceptance bounds the evidence argues with, five mid-implementation
specification changes, and three staged curation artifacts await operator
ratification (#407).

**Dependencies:** WS-1 for the agreement channel that would graduate the
accuracy claim from this workstream's evidence to published statistics (#225).
**Risk:** realized — the physics was hard and the delivered method sidesteps
the hardest part of it by never assuming a haze altitude.

### WS-8: PDS4 — generalize output bundle generation (input is separate and external-dependent)
**Closes:** "PDS4 is largely fictional."
**Tracked by:** #53 (bundle generator parent — output bundles, required for all
four instruments), #67 (cloud-aware bundles), #34 (PDS4 input — availability-
contingent, not required for completion).

**Scope split (binding).** PDS4 *output* (bundle generation) and PDS4 *input*
(reading PDS4-archived data as a dataset source) are different deliverables.
Output bundles are **mandatory for all four instruments**. Input is treated
like any other future instrument: no PDS4 archive of these datasets exists
yet, producing one is external development outside this project's control,
and input support is **not required for project completion** — when an
archive becomes available, implementing its `DataSetPDS4` replaces the PDS3
source for that instrument.

**Problem.** No instrument's bundle output works today: the Cassini path is
partially implemented (hook pattern and collection machinery exist) but has
no final templates, no tests, and no schema validation; the `pds4_*` bundle
hooks raise `NotImplementedError` for Voyager, Galileo, NH.
`dataset_pds4.py` (input) raises "not yet implemented" for all methods,
correctly, since no input archive exists to read.

**Tasks:**
- **Finish and validate the Cassini path (required):** final templates,
 tests, and schema validation for the partially implemented reference
 implementation.
- **Bundle generalization (required):** implement `pds4_*` hooks (template
 dir, LID/LIDVID, template variables) for Voyager/Galileo/NH using the
 completed Cassini path as the reference, with per-mission template trees.
- Add bundle-validation tests (schema-validate generated `.lblx` against PDS4
 schemas) for all four instruments.
- **PDS4 input (deferred until archives exist):** implement `DataSetPDS4`
 enumeration/reading per instrument as each external archive appears (#34);
 until then the capability matrix lists input as pending external archive
 availability.

**Acceptance criteria.** Generated bundles pass PDS4 schema validation in CI
for all four instruments; the capability matrix records PDS4 input as pending
external archive availability (or supported, once an archive exists and is
implemented).

**Dependencies:** WS-6. **Risk:** medium.

### WS-12: Per-instrument guide chapters — delivered

The four placeholder appendices are gone. Each instrument now has one chapter
in `docs/user_guide/instruments/` and one in `docs/dev_guide/instruments/`,
both copied from a `_template.rst` that fixes the section list, both discovered
by a glob toctree, and both required by
`tests/spindoctor/test_instrument_chapters.py`, which walks the instrument
registry and fails on a missing chapter or a missing template section. The
chapters carry what SpinDoctor decides, configures, measures or does
differently per instrument; they deliberately do **not** restate the instrument
teams' own optics, filter or plate-scale documentation, which the References
section of each chapter points at instead.

**Remaining work is per instrument, not per document.** Every value a chapter
records as a placeholder or a nominal-optics derivation is closed by the
workstream that measures it -- WS-9 for the limiting magnitudes and constants,
WS-1 for the distortion field, WS-13 for the noise model -- and each of those
updates the affected chapters in the same change.

---

## Phase 4 — Make the numbers defensible

### WS-9: Justify, derive, or measure the magic constants
**Closes:** "the numbers a scientist would quote are built on hand-picked
constants," and the fabricated star SNR.
**Tracked by:** #176 (constants into config) and #130 (star limiting-magnitude
calibration).

**Tasks.**
- Inventory the load-bearing constants: `ROTATION_UNOBSERVABLE_VARIANCE = 1e15`
 (`nav_technique.py`), `DEFAULT_PINVH_RCOND = 1e-9` (`ensemble.py`),
 `SNR_REF = 8.0` / `SNR_FLOOR = 0.1` (`nav_model/stars/nav_model_stars.py:77-78`),
 `COMBINED_CONFIDENCE_CAP = 0.99` and `AGREEMENT_FACTOR_CAP = 1.5`
 (`feature/constants.py`), blob noise thresholds, MAD factor, edge thresholds. For
 each: document its derivation, sensitivity, and the regime where it holds, next to
 its definition.
- **Decide where the confidence scale should saturate** (#557). The two caps above
 censor the combined confidence from above, and they bind far earlier than
 "exceptionally confident": the agreement factor `1 + 0.5*log2(n_significant)`
 already equals its own cap at `n_significant = 2`, so two significant
 corroborating techniques whose precision-weighted mean confidence reaches 0.66
 report exactly 0.99 unless a second agreement group applies the disagreement
 penalty, which is applied after the cap. Four distinct library frames measured
 2026-08-26 report that one value.
 Refusing to emit 1.0 is defensible -- confidence is a proxy, not a probability --
 but a scale that saturates on the ordinary good case carries no information above
 the cap, and gives no credit for a third or fourth corroborating technique.
- **Measure star SNR from the image, not the magnitude.** Replace (or cross-check)
 the synthesized `snr_eff = SNR_REF * 2.512**(mag_limit - vmag)` with a
 photometrically measured per-star SNR from the actual frame for gating and
 covariance, keeping the magnitude estimate only as a *prior* where no pixels are
 available. Validate that recovered star covariances pass the WS-5 coverage test.
- Add a **sensitivity test** that perturbs each constant and asserts results are
 stable within the documented tolerance (flags hidden over-fitting to a constant).

**Acceptance criteria.** Every load-bearing constant has a documented derivation
and a sensitivity bound; star SNR used for covariance derives from measured
photometry; covariance coverage passes WS-5; the confidence scale's saturation
point is a decision with a recorded rationale rather than an inherited constant.

**Dependencies:** WS-1/WS-5 for coverage validation. **Risk:**
medium — measured SNR may change star gating behavior and need re-tuning.

### WS-10: Fix the limb systematic bias
**Closes:** "known systematic, no working fix."
**Tracked by:** #150 (model-vs-image edge offset) and #128 (limb-navigation
redesign).

**Problem.** Limb bias ~0.09–0.13 px (≤0.25 px two-axis); the implemented
gradient-ridge refine is disabled for the limb technique
(`techniques.BodyLimbNav.tuning.gradient_ridge_refine: 0` in
`config_510_techniques.yaml`) because it worsens limb fits there, while the
ring-edge technique runs with it enabled (`RingEdgeNav` sets `1`). The limb's current partial
cancellation (integer DT quantization + Tukey) is accidental.

**Diagnosis (the measured basis for the redesign).** The instrumented
diagnosis confirms the mechanism and quantifies it: the genuine algorithmic
bias is 0.05–0.14 px, directional (points from the lit limb toward the
interior — a limb-darkening / photometric roll-off signature, the geometric
edge matching a gradient ridge ~0.5 px inside the true limb), varies with
illumination direction, and is roughly flat with body size; a ~0.05 px
sub-pixel interpolation ripple rides on top; below ~15 deg phase the fit is
poorly conditioned. The simulator's own limb render is bias-free (<2e-5 px),
so it is trustworthy ground truth. **The finding that reframes the
redesign:** on real limb+star frames the limb-vs-star gap is 0.5–1.8 px, an
order of magnitude larger than the 0.1 px algorithmic bias — so the limb
fitter explains only ~0.1 px and the remaining 0.4–1.7 px is
spacecraft-position / body-ephemeris error (isolable only because the sim
geometry is exact). Fixing the fitter buys ~0.1 px; the dominant real-frame
error is on the pointing-kernel side. Ranked fixes: (1) fit a photometric
limb (predict the limb-darkened-disc-convolved-with-PSF brightness profile
and match it) rather than aligning a geometric edge to the gradient ridge
(#150); (2) a matched-filter edge estimator to remove the interpolation
ripple (#282); (3) gate low-phase (<~15 deg) fits (#281, shipped: a
`BodyLimbNav` coarse-seed mis-lock below 15 deg phase is flagged spurious by
an unconverged-at-trust-boundary gate); (4) a minor
pixel-centre-convention audit (#283). Harness and full report:
`util/calibration/limb_bias/limb_navigation_bias_diagnosis.md`.

**Tasks.**
- Implement the diagnosis's remaining ranked fixes, in order: the
 photometric-limb fit
 (#150 — the dominant, illumination-tracking term), the matched-filter
 sub-pixel edge estimator (#282), and the
 pixel-centre-convention audit (#283). The low-phase gate (#281) is
 already shipped. The earlier candidate fixes (modeling
 a PSF-inward offset in `nav_model_body.py`; a continuous sub-pixel DT) are
 superseded: the diagnosis showed the dominant term is the photometric
 roll-off, not DT quantization.
- Re-measure limb accuracy on the WS-2 (mismatched-model) and WS-1 (real) cohorts;
 enable `gradient_ridge_refine` only if it demonstrably reduces real bias.

**Acceptance criteria.** Limb median bias reduced to a stated target (e.g.
≤0.03 px) on de-circularized synthetic and real cohorts, with the fix enabled by
default; no reliance on accidental cancellation documented as a feature.

**Dependencies:** WS-1, WS-2. **Risk:** medium-high — this is a
genuine algorithmic problem the team has already struggled with.

### WS-13: Make the calibrated (I/F) path realistic and tested
**Closes:** "the calibrated (I/F) path is essentially untested."

**Tasks.** Add a detector-noise model to the I/F render path in `sim/render.py`
(Poisson shot noise in electrons before conversion, read noise, full-well
saturation, bias pedestal, missing-data/CR markers), so I/F scenes exercise a
realistic noise regime. Add I/F frames to the WS-2 sweeps and the WS-1
consistency study (real calibrated products).

**Acceptance criteria.** I/F scenes carry a realistic detector model; the
accuracy and sweep reports include calibrated-path results comparable to the
raw-DN path.

**Dependencies:** WS-2. **Risk:** low-medium.

---

## Phase 5 — Operational hardening

### WS-15: Performance and safe parallelism
**Closes:** "it is slow," "the obvious fix for slowness is mined."
**Tracked by:** #103 (thread-unsafe caches), #134 (oops precision mutated
process-globally), #126 (rotation-pyramid cost).

**Tasks.**
- **Thread-safety:** remove the global `oops` precision mutation in
 `reproj/rings.py` (`_reduced_oops_precision`) in favor of a per-instance/per-
 thread setting, and make `BodyMosaic.reproject` / `create_cartographic_model`
 build per-thread `Backplane` objects. Add a concurrency test that runs
 reprojection on shared geometry across threads and checks bitwise-identical
 results vs serial.
- **Profile** the ~35 s/1024-px navigation (`--profile` already exists); attack the
 dominant cost (likely backplane/model rendering and DT construction). Target a
 documented per-frame budget.
- **Star matching:** bound or improve the O(M³) triplet hash (cap `max_sources`
 with brightness pre-selection, or switch to a smarter matcher) and document the
 complexity.
- Provide a **supported batch-parallel path** (process- or task-level) with a
 documented throughput figure for a realistic campaign size.

**Acceptance criteria.** Reprojection is thread-safe with a proving test;
per-frame navigation cost is reduced against a documented baseline; a supported
parallel batch path exists with a published throughput number.
**Dependencies:** none. **Risk:** medium — `oops` global state may
be deep.

### WS-18: End-product geometric accuracy (backplanes, mosaics, PDS4)
**Closes:** a gap the critical report implied but no other workstream covers — the
deliverables are not the offset, they are the *products built on it*, and their
correctness is unvalidated.

**Problem.** WS-1/WS-2 validate the navigation *offset*. But the science outputs
are backplanes (per-pixel geometry), mosaics (reprojected grids), and PDS4 bundles.
A correct offset can still feed a wrong backplane angle or a mis-registered mosaic
seam through a downstream bug; none of that is tested for accuracy today.

**Tasks.**
- **Backplanes:** on the WS-2 sim, where per-pixel
 lon/lat/incidence/emission/phase/radius/resolution are known by construction,
 assert the generated backplanes match to a stated tolerance. **The sim-truth
 geometry must be computed independently of the production backplane generator**
 (do not let both come from the same `oops.Backplane` calls), or "backplane matches
 sim truth" is the same self-consistency circularity WS-2 exists to break, one layer
 down. On real frames, cross-check internal consistency (e.g. ring-radius backplane
 vs the navigated ring geometry).
- **Mosaics:** quantify seam registration where overlapping reprojections meet
 (this is WS-1b's reprojection-consistency check applied to the mosaic product),
 and confirm the BEST_RESOLUTION/coverage merge picks the right pixel. A first
 measurement of this exists for ring mosaics in
 `util/nav_verification/measure_core_radius.py`, which reads the ring core's
 radial placement out of the assembled product against the orbit model it was
 built on: a displacement common to every column is a systematic to chase, and
 one that steps between adjacent columns is two neighbouring frames navigated
 differently from each other, with the step's longitude naming the frame. It
 needs nothing outside the mosaic, so it reaches observations no other project
 has navigated, but it reads one known feature rather than matching surface
 features across a seam, which is what the task above asks for.
- **PDS4:** beyond schema validation (WS-8), spot-check that label geometry values
 match the backplane metadata they are derived from.

**Acceptance criteria.** Backplane values match sim ground truth within a stated
tolerance and are internally consistent on real frames; mosaic seam registration is
quantified; PDS4 label geometry matches its source metadata.

**Dependencies:** WS-1b, WS-2, WS-8. **Risk:** low-medium.

---

## Sequencing and milestones

Dependency lists (each line: workstream — what it needs). No box-art, so it
survives reflow.

- **WS-0a** (estimator math + identifiability map) — a truth-known sim only; starts
 now, parallel with everything.
- **WS-3** (cohort) — nothing; starts now.
- **WS-2** (de-circularized + realism-validated sim) — WS-3.
- **WS-0b** (bias-independence verdict) — WS-2.
- **WS-17** (distortion) — WS-3 (star frames where they exist).
- **WS-1 pairwise layer** (combined covariance — the bulk product) — WS-3 + WS-17
 only. **Does not wait on WS-0/WS-2.**
- **WS-1 per-technique layer** (separation) — additionally WS-0 (+ WS-2 through it).
- **WS-1b** — WS-3; shares its implementation with WS-18.
- **WS-4** (CI) — WS-3 (+ WS-1 for the accuracy-regression gate).
- **WS-5** (confidence) — WS-0, WS-1, WS-2.
- **WS-6** (capability matrix) — light coupling to WS-8.
- **WS-8** (PDS4) — decision gates first. **WS-7** (Titan) and **WS-12**
  (per-instrument chapters) are delivered; their remaining issues are
  independent.
- **WS-9 / WS-10 / WS-13** (constants / limb bias / I/F) — WS-1 and/or WS-2.
- **WS-15** (performance) — independent; start anytime.
- **WS-18** (end-product accuracy) — WS-1b + WS-2 + WS-8.

WS-1 gates WS-5, WS-9, WS-10 (WS-18 gates on WS-1b, not WS-1 directly).

Note: WS-0 splits — **0a** (estimator code + identifiability map) needs only a
truth-known sim and starts immediately in parallel with WS-2; **0b** (the
bias-independence verdict) needs the realistic WS-2 sim and is cross-checked on real
data by WS-1's over-determination test.

Note: the chain is **WS-3 → WS-2 → WS-0 → WS-1** (no cycle). WS-2 (sim) carries the
absolute-accuracy number and is also where WS-0 proves the WS-1 estimator before any
real-image per-technique σ is believed. WS-1/WS-17 supply real-image *agreement*;
neither alone is "the" accuracy — sim accuracy + agreement + the stated
common-mode/identifiability/bias caveats together are the honest result.

- **Milestone B — "Measured-as-far-as-possible":** WS-3 + WS-2 + WS-0 + WS-17 +
 WS-1 (+WS-1b) + WS-4. WS-0 first proves the agreement estimator on the sim and
 maps where per-technique σ is even recoverable; WS-1 then reports pairwise
 combined σ everywhere and per-technique σ only on the qualified bins, absolute
 attitude from the scarce star tie-points plus the realism-validated sim. CI
 exercises real frames. The honest output is "pairwise agreement at scale +
 per-technique σ where identifiable + absolute attitude on a small stellar sample
 + sim, with stated caveats," not a single absolute px figure.
- **Milestone C — "Calibrated & defensible":** WS-5 (full), WS-9, WS-10, WS-13.
 Confidence and covariance mean something; the limb bias is fixed; the I/F path is
 real.
- **Milestone D — "Complete & scalable":** WS-8, WS-12, WS-15, WS-18.
 Capability gaps closed or honestly scoped; end products validated; performance and
 parallelism fit a campaign.

---

## Traceability: every critical-report finding → workstream

| # | Critical-report finding | Workstream(s) | "Closed" when |
|---|---|---|---|
| 1 | Validation is circular (shared forward model) | WS-2 | navigator uses its best model; image side independent + more-realistic; accuracy reported vs *residual* mismatch with realism-match evidence; no shared render fn under test |
| 2 | No real-image accuracy assessment | WS-1 (+WS-0, WS-1b, WS-17) | agreement report from real frames: pairwise combined σ everywhere, per-technique σ (with uncertainty intervals) only on WS-0-qualified identifiable, bias-independent bins; ephemeris carried as a separate bias; absolute attitude from scarce star tie-points + sim; explicit relative-vs-absolute split (external truth unobtainable) |
| 2′ | Per-technique σ unproven (identifiability + bias independence) | WS-0 | estimator recovers known sim σ; identifiability map published; bias independence established by injection test, not assumed |
| 3 | ~13-image cohort; CI skips integration | WS-3, WS-4 | cohort targets met; real images run in CI + nightly accuracy gate |
| 4 | Ships uncalibrated confidence/tiers | WS-5 | calibrated where per-technique covariance exists, sim-anchored elsewhere (basis recorded per value); reliability diagram; interim provisional label |
| 5 | Mid-rewrite; docs ≠ code; vapor; dead config | WS-6 | test-verified capability matrix; docs/config reconciled |
| 6a | Titan is a no-op | WS-7 | closed: haze solar-symmetry navigation implemented and validated on an 82-frame Cassini cohort to a stated 1 px cross-track / 3 px along-track bound, star-anchored |
| 6b | PDS4 input absent; bundles Cassini-only | WS-8 | bundles schema-validate for all four instruments; input recorded as pending external archive availability |
| 6c | Empty instrument appendices | WS-12 | closed: one user-guide and one dev-guide chapter per instrument, template-conformant and registry-enforced by test |
| 7a | Covariance shaped by magic constants | WS-9 | each constant derived + sensitivity-bounded |
| 7b | Star SNR fabricated from magnitude | WS-9 | measured-photometry SNR; covariance coverage passes |
| 8 | Limb bias, fix disabled | WS-10 | bias ≤ target on real+mismatched data, fix enabled |
| 9a | Slow (~35 s/frame), O(M³) matcher | WS-15 | per-frame budget cut; matcher bounded; throughput published |
| 9b | Reprojection thread-unsafe | WS-15 | per-thread state + proving concurrency test |
| 9d | I/F path noise-light/untested | WS-13 | realistic detector model + I/F in accuracy report |
| 10 | End products (backplanes/mosaics/PDS4) accuracy untested | WS-18 | backplanes match sim truth; mosaic seams quantified; PDS4 geometry matches source |

---

## Assumptions and notes

- These workstreams assume continued access to PDS holdings, SPICE kernels, and star
 catalogs for the target instruments; provisioning these to CI (WS-4) is itself a
 task and a risk.
- **Absolute subpixel ground truth for real frames does not exist** (star solves
 measure attitude only and fail on bright-body frames; inter-frame jitter kills
 repeat-frame prediction; legacy code shares this lineage and its subpixel layer
 was never trusted). The plan therefore measures absolute accuracy only in a
 de-circularized, realism-validated sim (WS-2), and uses real-image *agreement*
 (WS-1, with distortion removed per WS-17) as corroboration, with the residual
 common-mode CK error stated as an explicit bound. Any "accuracy" claim that cannot
 be backed this way is downgraded in the capability matrix rather than invented.
- WS-2 is now the critical path and highest-risk item: the validity of every
 absolute accuracy number rests on the image-side forward model being both
 independent of the navigator and provably realistic against real frames. If
 realism cannot be demonstrated for an instrument, its accuracy numbers are
 reported as sim-only, mismatch-bounded estimates, not measured accuracy.
- Common-mode error — a CK shift that moves *all* compared features *identically* —
 is the irreducible blind spot of real-image agreement (it cancels in the
 difference). It is caught only by the scarce star frames and the sim. Note this is
 distinct from distortion, which acts *differently* across the field, shows up in
 the disagreement, and is *corrected per feature-position* (WS-17) rather than being
 a hidden term.

**Statistical assumptions the per-technique numbers depend on (each stated in the
report, none silently assumed):**

- **Bias independence** of any two techniques entering a joint solve — **including
 through their shared preprocessing** (the `image_derivatives` DT/gradient images,
 noise-σ, reliability gate built once per `NavContext`, which couple all DT
 techniques). Verified by WS-0's injection test *through that shared layer*, not
 assumed; *no* pair (not even disc-NCC vs limb-DT) is treated as clean until
 measured. The two load-bearing pairs are limb-disc (the base equation) and
 limb-ring.
- **The recovered quantity is a 2x2 covariance, not a scalar.** Limb error is
 anisotropic and its axis rotates relative to the fixed ring-radial direction
 frame to frame, so per-technique error is recovered only as covariance elements on
 controlled geometry (bin also by limb-orientation-vs-radial, or solve in matrix
 form). Scalar-σ framing is wrong (WS-0/WS-1).
- **Techniques estimate the *same* offset** — true only for round, photometrically
 bland bodies. On topographic/high-contrast bodies the limb-center and disc-center
 differ by a real physical amount that inflates the apparent disagreement, so such
 bodies are excluded from per-technique separation by a reproducible gate derived
 from the static body-shape table (`ellipsoid_rms_residual_km`, `crater_scale_km`,
 `albedo_variation`), not a judgment call (WS-1 body-shape caveat).
- **Identifiability** of the covariance system per bin (WS-0). The common limb+disc
 frame is *not* separable; per-technique covariance is reported only on qualified,
 shape-clean, geometry-controlled bins, combined pairwise covariance elsewhere.
- **Errors are well-behaved enough for variance estimation** — robust estimation
 with outlier rejection is used, and *failure rate* is reported separately from
 error covariance (a catastrophic-failure technique and a large-error technique are
 different products) (WS-1).
- **The assumptions are checkable on real data only via over-determined frames**
 (limb+disc+ring+second body); their closure is the one real-photon test of
 bias-independence/stationarity/ephemeris, since WS-0's verdict is the sim's (WS-1).
- **Within-bin stationarity** of each technique's error covariance (WS-1) — assumed
 by the *pairwise* product too, not only the separation, so it is the broad limiter.
 Inherited by WS-5's calibration and WS-2's realism binning. Bin on resolution /
 phase / lit-fraction / limb-orientation and spot-check; if it fails, report the
 disagreement distribution rather than a single covariance.
- **Moon ephemeris error is a per-target bias, not zero-mean noise** — modelled as a
 separate nuisance parameter, never folded into a technique's σ (WS-1).
- **An accurate, independent PSF/shape model exists** to drive WS-2's realistic
 image. Solid for Cassini, doubtful for Voyager/Galileo; where it does not, that
 instrument's sim accuracy is bounded by unverified forward-model fidelity, not
 measured.
- **Saturn's ephemeris error is negligible at subpixel scale** (used in the
 star+ring relation) — to be quantified, not assumed.
- **Star-catalog astrometric error is separable from distortion** in WS-17 (it is
 field-position-independent; distortion is a smooth field pattern). **Centroiding
 error is *not* separable that way** — it grows toward the edges with the PSF, where
 distortion is also largest — so it is modelled explicitly from the PSF map and its
 uncertainty propagated, not assumed flat. Distortion is then applied per
 feature-position in WS-1, leaving only its residual in the budget.
- **Realism validation needs only real frames, not star frames** (it is a
 statistics match), so WS-2 realism — hence the sole absolute-accuracy basis for
 Voyager/Galileo — is achievable for every instrument.
- **Validation reach is sharply instrument-dependent and must be reported per
 instrument, not pooled.** *Cassini ISS:* relative/algorithm σ at scale **plus**
 an absolute-attitude anchor from dozens of star tie-points — the only fully
 validatable instrument. *NH LORRI:* star-frame count to be verified (WS-3);
 low-distortion optics reduce the risk regardless. *Voyager ISS / Galileo SSI:*
 effectively zero star frames, so (a) absolute attitude accuracy is **sim-only**
 (no real-data inertial reference exists) and (b) geometric distortion cannot be
 validated in-house and is **adopted from the literature** (WS-17) — and these are
 the most-distorted instruments, so their real-image numbers carry the weakest
 backing. The agreement report must scope every number to its instrument's reach.
- (Within-bin stationarity, covered above, is the broad limiter — it constrains the
 *pairwise* fallback too, which is *less* assumption-laden than the separation but
 not assumption-free.)
- Titan navigation is built and validated (WS-7), so the capability matrix
 records it as supported with its bound rather than as a scope question; what
 remains open there is the ratification bundle (#407) and the deferred
 refinements. PDS4 output bundles are required for all four instruments
 (WS-8); only PDS4 *input* is external-dependent and out of the completion
 scope.
- **Validation can come back bad.** There is no fixed accuracy spec to pass (goal is
 best-achievable, honestly characterized), so a poor result is not a go/no-go gate
 — but it is not merely reported either: a worse-than-hoped σ routes into the
 algorithm workstreams (WS-10 for the limb; WS-9 for covariance scaling; new work
 if a technique is structurally weak). The plan does not assume validation merely
 confirms.
- Nothing here changes code or docs yet; this is the plan of record.
