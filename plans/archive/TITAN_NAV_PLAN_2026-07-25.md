# Titan Haze Navigation Plan (the French method)

*Implementation plan of record for the "implement" branch of program
decision #60 (Titan navigation; WS-7 in
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`, Track B/D in
`plans/PROGRAM_PLAN.md`). Written 2026-07-25. The method analysis
that selected this approach is frozen at
`critiques/TITAN_NAV_CONCEPT_2026-07-25.md`; this plan is
self-contained and does not require reading any critique.*

*Revised through revision 12 (2026-07-25): seven independent review
rounds — three adversarial/verification passes, three Opus-model
cold reads, and a repo-wide collateral-surface sweep of tests, utilities,
and documentation — plus operator-directed corrections, most
recently the deletion of the Titan-specific status reason in favor
of the uniform reliability-gate path; the frozen records
(`critiques/TITAN_NAV_PLAN_CRITIQUE_2026-07-25*.md`,
`critiques/TITAN_NAV_COLLATERAL_SWEEP_2026-07-25.md`) say what each
revision changed and why. All normative behavior lives in Sections
1-10 below; nothing in the revision history adds to or overrides
them.*

*This plan is written to be executed by an implementing model with no
briefing beyond `/seti/newnav/CLAUDE.md` and the repo itself. Every
phase names its files, its algorithm section, its tests, and its
acceptance gate. Where a value is tunable it appears in the config
schema (Section 5) with a default; an implementer changes defaults
only through the tuning procedure in Phase E, never ad hoc. Code
references below use file paths plus symbol names, not line numbers;
verify against the current source when implementing.*

---

## 0. Scope

**Goal.** Titan frames navigate autonomously: a new technique produces
a `(dv, du)` offset with honest anisotropic covariance and calibrated
provisional confidence from the haze's solar-symmetry geometry, for
all supported instruments, with no per-filter or per-phase training
data.

**Method in one paragraph** (Hanson, French, Waugh, Barth & Anderson,
2025, GRL, doi:10.1029/2024GL113415 — "the French method"): absent
clouds or visible surface features, a hazy atmosphere is
mirror-symmetric about the image-plane line through the body center
and the sub-solar point. Finding the image offset perpendicular to
that line that maximizes observed symmetry pins the cross-axis
coordinate even in the presence of minor variations. The limb region
in the direction of the sub-solar point is approximately circular, so
a circle fit with free radius to that sunward limb arc pins the
along-axis coordinate without knowing the haze altitude. The two
constraints together give the navigated offset.

**In scope:** the fitting library; `NavModelTitan` emitting a real
`TITAN_LIMB` feature; a new `TitanHazeNav` technique; feasibility,
body-occlusion, and ring-occlusion gating; annotations; config; a
simulated-Titan extension of the existing body `atmosphere` block and
a planted-truth harness; real-frame validation; documentation and plan
reconciliation.

**Out of scope** (tracking issues filed at merge, Section 9): the
legacy profile-database method and the breakpoint method (rejected in
the frozen analysis); a self-calibrated haze-radius table; methane
surface-window cartographic correlation; Voyager-cohort validation
depth; ensemble treatment of strongly anisotropic covariance beyond
what the existing merge already supports; final (non-provisional)
confidence calibration, which belongs to the #230 real-evidence
re-anchoring.

**Execution framework** (binding; from `plans/OPERATOR_PLAYBOOK.md`
Section 4): one controller session; one implementer subagent per
phase; an independent fresh-context adversarial review of every phase
deliverable; fix rounds until the critique is clean; full CI
(`./scripts/run-all-checks.sh -i`) before merge; a single PR from the
integration branch `rf_titan_nav`. Model requirement: every subagent
this plan is executed with — implementer and reviewer alike — runs
on an Opus-class model (model override `opus`), NOT Fable; the plan
was written and validated against Opus-cold-read executability, and
the model choice is part of the contract. Details in Section 7.

## 1. Conventions binding on every phase

- Project rules: `/seti/newnav/CLAUDE.md` and `.cursor/rules/*.mdc`
  (read both before writing code). Highlights that recur here: line
  length 100; mypy strict; Google docstrings with `Parameters:`; no
  unicode punctuation in `.py`; pdslogger only (never stdlib
  `logging`) in core packages; every `NavTechnique.navigate` body
  opens `with self.logger.open(f'TECHNIQUE: {self.name}'):`; issue
  numbers in `#`-comments only, never docstrings; modules under 1000
  lines; tests annotated, one assert per condition; Conventional
  Commits.
- Offset convention: predicted position `(v, u)` means actual position
  is `(v + dv, u + du)`. `v` is row (down), `u` is column (right).
- Pixel coordinates in Section 2 are all in the extended-FOV frame:
  the image is `NavContext.image_ext`
  (`src/spindoctor/nav_orchestrator/nav_context.py`), and the model
  computes its predicted center in the same frame (nominal-frame
  coordinates plus the extfov margin, exactly as
  `NavModelBody.create_model` does for its bbox math). Do not mix
  nominal-frame and extended-frame coordinates.
- Environment: `source /seti/newnav/setup.sh` (provides venv, SPICE
  under `$OOPS_RESOURCES/SPICE`, holdings, star catalogs). Integration
  data is always available locally.
- Multi-test integration runs use `pytest -n auto --dist=loadfile`.
- After any change that can move navigation output, run the library
  cross-check (`util/calibration/library_crosscheck.py`) and account
  for every delta against the documented pinned red set.
- Never edit a library sidecar's `expected.*` fields to match current
  behavior.
- Config access: `Config` section properties return an `AttrDict`
  that wraps only the top level of the section; nested mappings are
  plain dicts. Read `config.titan['navigation']['symmetry']` style
  below the first level (matching how technique `tuning` dicts are
  consumed), not attribute chains.

## 2. Method specification (normative)

This section defines the algorithm the code must implement. Symbols:

- `theta`: symmetry-axis angle.
- `a_hat = (sin theta, cos theta)` in `(v, u)`: unit vector along the
  axis, pointing toward the sub-solar side.
- `c_hat = (cos theta, -sin theta)` in `(v, u)`: the perpendicular
  unit vector. This exact formula is the definition; the diagnostics
  sign convention is that `cross_track_px` is positive along `c_hat`.
  (The final offset is invariant to the sign choice; tests must use
  this one.)
- `r_solid_px`, `r_env_px`: apparent solid-body and haze-envelope
  radii in pixels.
- `W`: pointing search half-window in pixels.

### 2.1 Geometry inputs (computed by the model, at predicted pointing)

1. **Predicted center** `p0 = (v0, u0)`: the geometric disc center —
   Titan's oops inventory `center_uv`, converted to extended-frame
   coordinates by adding the extfov margins
   (`p0 = (center_uv[1] + extfov_margin_v, center_uv[0] +
   extfov_margin_u)`). Deliberately NOT the lit-weighted centroid
   (`_lit_weighted_centroid_vu` in `nav_model_body_base.py`) that body
   features use as their predicted center: that centroid is
   phase-biased along the sun direction, which is exactly the axis
   this method fits.
   [Revised during Phase B review; pending operator ratification. The
   original text derived `p0` from the midpoint of the inventory
   bounding box, the way `NavModelBody.create_model` does. That
   midpoint is quantized: the box edges are integers, so it lands on a
   whole or half pixel regardless of where the body actually is —
   measured 0.343 px of error in `u` on `W1822132529_1`, a third of
   this method's entire 1 px cross-track budget, spent before the fit
   starts. `center_uv` is the exact projected center and costs
   nothing. The box midpoint remains correct for the envelope and mask
   boxes, which are integer pixel boxes bounding where backplanes are
   evaluated, and those keep using it.
   Frame convention: `center_uv` is a field-of-view coordinate, and
   adding the margin with no half-pixel adjustment is the
   pipeline-wide convention for a predicted position — a catalog
   star's extfov position is `star.v + extfov_margin_v`
   (`nav_model_stars.py`). Holding to it is what makes a haze offset
   and a star offset on the same frame directly comparable, which the
   Phase E evidence tier (a) agreement test depends on.]
2. **Scale and radii.** `km_per_px` = the mean of
   `float(bp.center_resolution('TITAN', axis='u').vals)` and the same
   call with `axis='v'` — the method signature is
   `center_resolution(event_key, axis='u')`, one scalar per axis;
   `src/spindoctor/reproj/cartographic_model.py` shows single-axis
   usage with the default axis, and the dual-axis average here is
   this plan's own convention. Then
   `r_solid_px = R_TITAN_km / km_per_px` and
   `r_env_px = (R_TITAN_km + atmosphere_height) / km_per_px`, with
   `R_TITAN_km` from the oops body and `atmosphere_height` (km) from
   config. Do not register a modified oops Body: nothing here needs an
   inflated body in the SPICE inventory (the legacy system did this;
   it mutates oops global registry state, which is process-wide), and
   radii are plain numbers used to build masks and windows.
3. **Symmetry angle** `theta`, from the incidence-angle backplane
   (there is no sub-solar-point image projection helper in the repo,
   and `_sub_solar_dir_vu` in `nav_model_body_base.py` must NOT be
   used — it is a brightness-centroid heuristic that intentionally
   returns `(0, 0)` at low phase): build the incidence backplane over
   the envelope bounding box at predicted pointing
   (`oops.Backplane(...).incidence_angle('TITAN')`, the same call
   family `NavModelBody` uses for its backplane scalars; construct
   the bbox-restricted meshgrid/backplane the way
   `NavModelBody.create_model` builds its `restr_bp` — but over the
   UNCLIPPED envelope bbox, not the extfov-clipped one `restr_bp`
   uses: oops backplanes evaluate fine at off-detector pixel
   coordinates, and a clipped bbox can contain zero
   surface-intercept pixels on exactly the off-edge frames the
   hard-zero visibility condition targets — and embed subgrid
   results into the full frame at the bbox origin), mask to
   the incidence backplane's valid pixels — the surface-intercept
   mask the backplane itself carries, not the envelope disc — and
   take the pixel `(ve, ue)` of MINIMUM incidence
   — at every phase, not just below 90 degrees. (Express `(ve, ue)` in
   the SAME frame as `p0`: the meshgrid's field-of-view coordinate plus
   the extfov margin, with no half-pixel adjustment, exactly as item 1
   builds `p0` from `center_uv`. `theta` is a difference, so only
   consistency matters — but converting one end to pixel indices and
   not the other tilts the axis by half a pixel over the disc radius.
   Noted during Phase B.) The minimum-incidence
   visible pixel always projects in the sunward image direction; the
   maximum-incidence pixel is the anti-solar surface point, which
   becomes visible above 90 degrees phase and points the wrong way
   (do not branch on phase here — a max-incidence branch flips
   `theta` by 180 degrees and aims the arc fit at the dark limb
   exactly on the high-phase frames). Then
   `theta = atan2(ve - v0, ue - u0)`. Degenerate case: if
   `hypot(ve - v0, ue - u0) < axis_min_offset_px`, the phase is near
   zero and the disc is nearly rotationally symmetric; set
   `theta = 0.0`, skip angle refinement (Section 2.2 step 5), and set
   the `axis_degenerate` diagnostic flag — any axis is equally valid
   there, and the arc fit still constrains the center because the
   whole limb is circular. Second degenerate case: if the
   surface-intercept set is empty or any geometry computation fails,
   do NOT raise — set `theta = 0.0`, `axis_degenerate = True`, and
   defensible defaults, and let the Section 2.5 never-raise rule
   carry the frame to an attributable gate.
4. **Phase angle** at Titan center, `phase_deg`.
5. **Contaminant mask** — pixels the fits must ignore, built
   UNDILATED at predicted geometry as the union of four components.
   Source domain: the *mask bbox* = the envelope bounding box
   dilated by `annulus_outer_pad_px + 2 * W` on every side. The
   fits sample out to `r_env + pad + W` from centers hypothesized up
   to `W` away, so the mask must cover everything they can touch —
   an envelope-bbox-only mask leaves contaminants in the outer
   margin unmaskable on exactly the large-offset frames. Components
   are painted over the mask bbox and the model ships the result
   embedded in a full extended-frame-shaped boolean array (one bool
   per pixel; a shared coordinate frame beats a bbox-origin
   parameter on every fitting signature), or None when empty.
   Alignment principle: the offset is a scene-wide translation (an
   attitude error displaces Titan, moons, rings, and stars
   identically), so every relative position is exact, and the mask
   is applied *shifted by the current center hypothesis*, never
   statically. In the pass-1 symmetry scan at candidate `c`, mask
   validity for grid cell `(s, t)` is read at `(s - c, t)` from the
   mask dilated along `a_hat` only, by `+-W` (the along-track
   component is the one not yet solved); the pass-1 arc step uses
   the same c-shifted, t-dilated mask. In the recenter pass the
   grid moves to `p0 + d * a_hat` but the mask stays ANCHORED AT
   `p0`: the mask is read at the sample position MINUS the
   accumulated shift, so the residual misalignment is the
   along-track error still left after the first pass rather than
   the full `d`. (Resampling the mask on the moved grid instead
   would leave it misaligned by the whole `d`, which the
   `recenter_threshold_px` dilation is not sized for.) The pass-2
   symmetry scan rides the NEW candidate `c'` (not pass-1's
   `c_sub`), the pass-2 arc step shifts by the accumulated
   `(c, d)`, and the along-track dilation shrinks to
   `recenter_threshold_px` everywhere. The model ships the
   undilated mask; the fitting code owns alignment and dilation.
   - *Body occluders* — nearer bodies covering Titan.
     `NavModelBody`'s existing body-body occlusion computation
     (`_compute_occluder_local` and its sibling-inventory plumbing in
     `nav_model_body.py`) is private instance state; Phase B extracts
     it into a module-level helper in `nav_model_body.py` that takes
     exactly the state the current code consumes — the pre-built
     oversampled restricted backplane, the sibling list, the subject
     range, and the oversample factors:
     `occluder_mask_for_body(restr_bp, body_name, siblings,
     subject_range_km, *, oversample_v, oversample_u) ->
     NDArrayBoolType | None`. `NavModelBody` passes the objects it
     already builds — that is what makes bit-identical behavior
     achievable; a helper that rebuilt its own backplane could not
     guarantee it — and `NavModelTitan` builds its own restricted
     backplane over the mask bbox, following the `restr_bp`
     construction in `NavModelBody.create_model`. `NavModelBody`
     behavior must be bit-identical after the refactor (its existing
     tests are the guard).
   - *Ring occluders* (new; nothing in the repo occludes by rings
     today): pixels of the mask bbox where the Saturn ring-plane
     intercept has radius inside
     `titan.navigation['ring_occlusion_radii_km']` (default
     `[74490.0, 140500.0]`, C-ring inner edge to just outside the F
     ring) AND the ring intercept range is less than the range to
     Titan's center. Both quantities come from the standard oops
     `Backplane` ring methods evaluated on the bbox subgrid. This
     deliberately treats the main rings as opaque; translucency
     refinement is out of scope. Note the mask is what makes the many
     "edge-on rings occluding" frames in the Phase E cohort gate out
     at hard-zero reliability instead of fitting through ring
     stripes.
   - *Sibling footprints* — the inventory bbox of every other in-FOV
     body regardless of range order (enumerated by the module-level
     `bodies_in_extfov(obs, config)` in `nav_model_body.py`). A moon
     behind Titan occludes nothing, but its visible sliver beside the
     limb sits exactly in the symmetry annulus and in the arc rays.
     Bbox masking is deliberately conservative: a moon hidden
     entirely behind Titan costs a bbox-sized patch of valid pairs,
     which the coverage gates meter.
   - *Bright point sources* — predicted positions of catalog stars
     brighter than `star_mask_vmag_limit`, from the module-level
     `stars_in_extfov` (`src/spindoctor/nav_model/stars/catalog.py`
     — its docstring notes the reduction is exposed as free
     functions so `NavModelStars` composes them without subclassing;
     it is equally callable here with the `(obs, config)` a model
     already holds). It takes ONE catalog per call and filters on
     raw catalog magnitude with no photometry correction, so query
     the photometry-reference catalogs — `catalog_name='ybsc'` for
     `mag_max=6.5` and `catalog_name='tycho2'` for `mag_min=6.5,
     mag_max=star_mask_vmag_limit` (`mag_min` is keyword-only with
     no default, so the YBSC call must pass one too:
     `catalog_name='ybsc', mag_min=-2.0, mag_max=6.5`) — and NEVER
     bright-end UCAC4:
     `UCAC4_SATURATION_VMAG_LIMIT` is 8.0 (`stars/saturation.py`)
     and merged UCAC4 magnitudes can run several magnitudes too
     faint exactly in the mask's range. Duplicates across the two
     queries are harmless (overlapping mask discs). Returned
     `star.u` / `star.v` are nominal-frame; add the extfov margins
     before painting each disc of radius `star_mask_radius_px`.
     Because the mask rides the offset hypothesis, a star's mask
     lands on the star to within the along-track dilation.
6. **Search half-window** `W`: `max(search_window_for_obs(context))`
   (`search_window_for_obs` in
   `src/spindoctor/nav_technique/nav_technique.py` returns the
   per-axis extfov margins). Note what this IS: the fixed
   per-instrument extended-FOV margin every technique uses as its
   offset search bound and at-edge reference — a constant of the
   instrument configuration, not a per-frame pointing-uncertainty
   estimate. Do not go looking for a dynamic uncertainty quantity;
   none exists in the repo, and "search half-window" here means only
   "the offset range this technique searches." The scalar `max` of
   the two margins is deliberate: `c` and `d` live in the rotated
   `c_hat` / `a_hat` frame, where per-axis image margins cannot be
   applied cleanly. The model computes `W` in `create_model` by
   reading `obs.extfov_margin_vu` directly — `search_window_for_obs`
   is the technique-side accessor of the same values, and no
   `NavContext` exists at model-build time. The geometry payload
   carries BOTH the scalar `W` and the per-axis margins it came from:
   the scalar is what the rotated-frame search uses, and the per-axis
   pair is what the Section 2.5 visibility condition uses, that test
   running in image axes where per-axis margins DO apply cleanly.
   [Added during Phase E; pending operator ratification — see the
   visibility bullet in Section 2.5.]

`occluded_fraction` = pixels of the UNDILATED occluder + ring
components (not the sibling or star components) inside the envelope
disc
(distance <= `r_env_px` from `p0`) divided by the total pixel count of
that disc (clipped to the extended frame). It is a geometric estimate
of true occlusion, feeding the reliability formula and its
hard-zero conditions; the
dilation and the sibling component are search-robustness devices and
do not count toward it.

Mask and annulus per pass — the normative one-glance summary of the
rules stated across Sections 2.1-2.3 (where prose and this table
disagree, fix the prose):

| Pass / step | Center | Mask shift | Along-track dilation | Annulus |
|---|---|---|---|---|
| 1, symmetry scan | `p0` | candidate `c` | `+-W` | capsule (band t-dilated by `W`) |
| 1, arc step | rays from `p0 + c_sub * c_hat` | `c_sub` | `+-W` | n/a (per-ray windows span `+-W`) |
| recenter, symmetry scan | `p0 + d * a_hat` | new candidate `c'` | `+-recenter_threshold_px` | tight annulus |
| recenter, arc step | rays from the recentered origin `+ c_sub * c_hat` | accumulated `(c_sub, d)` | `+-recenter_threshold_px` | n/a |

**Deliberately not masked: stars fainter than the mask limit, cosmic
rays, hot pixels.** Faint point sources are a few low-amplitude
pixels against thousands of mirror pairs in the Pearson score and a
handful of samples in a median-filtered radial profile; cosmic rays
and hot pixels additionally have no predicted position to shift with
the hypothesis. Phase A test 11 and the Phase D star / cosmic-ray
injections verify the insensitivity and guard the chosen
`star_mask_vmag_limit`.

### 2.2 Step S — cross-track offset by mirror correlation

1. Resample the image onto a rotated grid `G(s, t)` centered on `p0`,
   with both `t` (along `a_hat`) and `s` (along `c_hat`) spanning
   `[-(r_env_px + pad + W), +(r_env_px + pad + W)]`, 1 px spacing,
   cubic interpolation (`scipy.ndimage.map_coordinates`, `order=3`;
   out-of-frame samples marked invalid;
   `pad = symmetry['annulus_outer_pad_px']`). The along-track extent
   includes `W` so the whole disc is in-grid wherever Titan actually
   sits within the pointing window: truncation in `t` is symmetric in
   `s` and so cannot bias the peak, but it discards exactly the limb
   pairs that carry the signal. Resample the contaminant mask with
   `order=0` and pre-dilate it along `t` by the pass pad — `+-W` in
   pass 1, `+-recenter_threshold_px` in the recenter pass; during
   scoring it is read c-shifted per the Section 2.1 alignment
   principle (mask validity for the pair member at `(s, t)` comes
   from the dilated mask at `(s - c, t)`). Out-of-frame samples are
   invalid unconditionally.
2. Annulus domain for a candidate shift `c` — pass-aware, because the
   along-track position is unknown in pass 1:
   - *Pass 1 (capsule):* grid points whose distance from the axis
     segment `{(c, t0) : |t0| <= W}` lies in
     `[annulus_inner_fraction * r_env_px, r_env_px + pad]` — the
     tight annulus dilated along `t` by `W`, the same
     hypothesis-riding treatment the contaminant mask gets. This is
     not an optimization: with a small Titan and a large true
     along-track error (`r_env_px` well below `W` is a legal frame),
     a t=0-centered annulus can miss the disc entirely and produce
     no signal at any `c`.
   - *Recenter pass:* the tight annulus,
     `rho = hypot(s - c, t)` in the same band.
   The band (not the full disc) is used so structured content in the
   disc interior — surface-window features, cloud fields — cannot
   bias the symmetry estimate. Both shapes are symmetric in `s`
   about the candidate axis, so neither can bias the peak position;
   the capsule only admits more diluting samples, which pass 2
   removes.
3. For each integer `c` in `[-W, +W]`: form the mirror pairs
   `(G(c + q, t), G(c - q, t))` for all `q > 0, t` with both samples
   valid and both grid points in the annulus domain; `score(c)` =
   Pearson correlation over those pairs;
   `valid_fraction(c)` = (number of pairs used) / (total number of
   `(q, t)` pairs in the annulus domain, valid or not). (`q` is the
   mirror displacement — distinct from the along-track shift `d` of
   Section 2.3.) Pearson — not SSD, not unnormalized correlation —
   is load-bearing: it is invariant to an affine brightness relation
   between the two halves, so Titan's hemispheric north-south
   asymmetry (whose boundary runs roughly along the sun axis, i.e.
   the mirror maps north onto south) is invisible to the score as
   long as it scales or offsets one side uniformly; only
   structural, non-affine asymmetry costs correlation. Do not
   substitute another similarity metric.
4. `c* = argmax score`. Sub-pixel refinement: fit
   `score(c) = s_pk + a (c - c_sub)^2` through the three points
   `(c*-1, c*, c*+1)` and take `c_sub` (require `a < 0`; if the peak
   is at the window boundary, skip refinement and set `at_edge`).
   `sigma_cross = cross_sigma_scale * sqrt((1 - s_pk) / (2 * |a|))`,
   with `(1 - s_pk)` clamped below at 0 (parabolic refinement can
   push `s_pk` above 1) and any non-finite result replaced by `W`
   — the widest uncertainty this fit can express — then clamped to
   `[sigma_floor_cross_px, W]`. [Revised during Phase A: the
   original text said "replaced by the floor", which reports the
   TIGHTEST possible sigma for exactly the two cases that produce a
   non-finite estimate, an `at_edge` peak and a flat score curve.
   Pending operator ratification.] The formula is a
   noise-deficit heuristic; `cross_sigma_scale` (0.10 as set by the
   Phase D free-row solve; the pre-calibration default was 1.0) is
   calibrated in Phase D so planted-truth cross-track z-scores are
   unit-normal, which is what makes the reported sigma meaningful.
5. Optional angle refinement (skipped when `axis_degenerate`): repeat
   steps 3-4 for `theta` offsets in `+-angle_refine_deg` at
   `angle_refine_step_deg` spacing; adopt the refined `(theta, c)`
   only if its peak beats the SPICE-`theta` peak by more than
   `angle_refine_min_gain`; else keep SPICE `theta`. (Titan's
   atmospheric symmetry axis is known to be tilted a few degrees from
   the spin axis; this absorbs it without trusting a noisy fit.)
6. Gates, evaluated ONLY on the final pass (Section 2.3.8):
   intermediate-pass estimates are carried forward ungated, because a
   pass-1 score diluted by the capsule must not kill a frame the
   recenter pass was designed to rescue, and a garbage intermediate
   estimate cannot survive the final pass's tightened annulus and
   realigned mask. `gate_failed` always names a final-pass gate. In
   this order; on failure the technique returns a spurious result
   with the gate name in `diagnostics.gate_failed`:
   - `valid_fraction(c*) >= min_valid_fraction` (gate name
     `'valid_fraction'`);
   - `score(c*) >= min_peak_score` (`'peak_score'`);
   - competing-peak check (`'second_peak'`): normalize
     `s_norm(c) = (score(c) - s_min) / (score(c*) - s_min)` where
     `s_min` is the window minimum; competing peaks are local maxima
     of `score` at distance >= 3 px from `c*`; the gate fails if any
     has `s_norm > max_second_peak_ratio`; no competing local maxima
     means the gate passes;
   - `|c*| < W`, else set `at_edge` (an `at_edge` result is returned
     non-spurious with the flag set; the ensemble already treats
     `at_edge` conservatively).

### 2.3 Step A — along-track offset by sunward arc fit

1. Fix the cross-track shift: rays emanate from
   `p1 = p0 + c_sub * c_hat`.
2. Ray set: angles `phi` in `[theta - sector_half_angle_deg,
   theta + sector_half_angle_deg]` at `ray_step_deg` spacing. Drop
   any ray with an out-of-frame sample, or a contaminant-masked
   sample (mask aligned per Section 2.1: shifted by the accumulated
   center hypothesis, t-dilated by `W` in pass 1 and by
   `recenter_threshold_px` in the recenter pass), at radius
   `> r_solid_px` (the limb region must be clean; interior samples
   may be masked without harm).
3. Per-ray radial profile: sample the image at radii `r` from
   `radial_inner_fraction * r_solid_px` (note the base: this scales
   the SOLID radius; the symmetry `annulus_inner_fraction` scales
   `r_env_px` — do not conflate the two inner fractions) to
   `r_env_px + radial_outer_pad_px + W`, step `radial_step_px`, along
   direction `(sin phi, cos phi)` from `p1`, cubic interpolation;
   median-filter the profile with `median_filter_samples` taps.
4. Per-ray limb radius `rho_phi`: within the window given by
   intersecting `[r_solid_px - W, r_env_px + W]` with the sampled
   range shrunk by half the median-filter width at each end, find the
   most negative outward gradient (steepest falloff into sky) of the
   filtered profile; refine by parabolic interpolation on the three
   gradient samples around the minimum, with the refined vertex
   clamped to `+-0.5` sample of that minimum (a vertex further out
   than half a sample comes from noise flattening the curvature, not
   from the limb). The window's width — not a haze-altitude
   assumption — is what bounds where the limb may sit.
   Drop the ray unless
   `|g_min| >= min_gradient_snr * MAD(gradient over the window)`
   (raw MAD, no scale factor — this is a same-units SNR test).
   ALSO drop the ray when the minimum lands on the first or last
   sample of the window: that is the search saturating against its
   own bound, not a detected extremum, and the true limb may lie
   outside the window. (Without this rule a body displaced past the
   window returns a cluster of rays pinned at exactly the window
   bound, whose mutual agreement then wins the robust fit and
   produces a gate-passing, floor-sigma answer that is wrong by the
   whole excess — the `arc_radius` gate cannot see it, because the
   saturation radius is inside the gate band by construction.)
   Contaminated interior samples cost the ray nothing, but the
   gradient samples they touch are excluded from the search so a
   masked-out hole cannot masquerade as the steepest falloff.
5. Robust circle fit with center constrained to the symmetry axis:
   over scalar `d` (along-track center shift) and radius `R`,
   minimize `sum_phi rho_Tukey(e_phi / s_mad)` with
   `e_phi = |x_phi - (p1 + d * a_hat)| - R` and
   `x_phi = p1 + rho_phi * (sin phi, cos phi)`. IRLS: at each outer
   iteration, `s_mad = 1.4826 * MAD(e_phi over current inliers)`,
   Tukey biweight with tuning constant `tukey_c` in units of `s_mad`
   (reuse `tukey_biweight_weights` from
   `src/spindoctor/nav_technique/dt_fitting/weights.py` — it is a
   pure importable function); the inner step is one Gauss-Newton
   update of `(d, R)` on the weighted residuals, taken in FULL —
   truncating the step to a maximum length breaks the fit outright,
   because a clipped `d` step paired with an unclipped `R` step
   leaves a uniform residual offset that `s_mad` (a spread, not an
   offset) reads as an all-outlier population, and the next
   reweighting zeroes every ray. Initialize
   `d = 0`, `R = median(rho_phi)`; iterate until `|delta d| < 0.01`
   px or 25 iterations. Floor `s_mad` at `1e-3` px so a noiseless
   arc, whose MAD is exactly zero, does not make every Tukey
   argument non-finite.
6. Uncertainty: with final weights `w_phi`, residuals `e_phi`, and
   Jacobian `J` of the residual vector with respect to `(d, R)`:
   `s2 = sum(w e^2) / max(1, sum(w) - 2)` and
   `Cov(d, R) = s2 * inv(J^T diag(w) J)`;
   `sigma_along = along_sigma_scale * sqrt(Cov[0, 0])`, clamped to
   `[sigma_floor_along_px, W]`. `along_sigma_scale` (default 1.0) is
   calibrated in Phase D alongside `cross_sigma_scale`.
7. Gates (spurious + named gate on failure, order as listed): ray
   yield — rays surviving steps 2-4 `>= min_rays` (`'ray_yield'`);
   inlier count (final `w > 0`) `>= min_rays` and inlier fraction of
   surviving rays `>= min_inlier_fraction` (`'arc_inliers'`); fitted
   `R` in `[0.98 * r_solid_px, 1.05 * (r_env_px + W)]`
   (`'arc_radius'`); inlier residual RMS `<= max_residual_rms_px`
   (`'arc_residual'`); `|d| < W`, else `at_edge` as in Section 2.2.
8. **Recenter pass.** If `|d| > recenter_threshold_px`, repeat
   Section 2.2 exactly once with the grid centered on
   `p0 + d * a_hat` (`theta` and `W` unchanged; angle refinement per
   its own rules; the pass-2 scan rides its own new candidate `c'`
   and its `c_sub` replaces pass-1's — see Section 2.4), then repeat
   steps 1-7 of this section from the new `c_sub`; set the
   `recentered` diagnostic. At most one repeat: the
   first pass bounds the residual along-track error to the fit-noise
   scale, so the second pass's annulus and ray windows are well
   placed. Gates are evaluated on the final pass.

### 2.4 Result assembly

- Offset: `(dv, du)` = the FINAL pass's `c_sub * c_hat` plus the sum
  of the along-track shifts `d` over both passes. Each pass's
  symmetry scan re-measures the full cross-track offset (the
  recenter moves the grid along `a_hat` only), so pass-2 `c_sub`
  REPLACES pass-1's — summing both would double-count cross-track —
  while the `d` contributions genuinely accumulate. Worked form for
  a recentered run:
  `(dv, du) = c_sub_pass2 * c_hat + (d_pass1 + d_pass2) * a_hat`.
  This is the measured position of Titan relative to prediction,
  matching the repo convention (actual = predicted + offset).
- Covariance:
  `Sigma_vu = M diag(sigma_cross^2, sigma_along^2) M^T` where `M` has
  columns `c_hat` and `a_hat` expressed in `(v, u)`; then apply the
  technique's configured model-error floor via `add_model_error_floor`
  (`src/spindoctor/nav_technique/nav_technique.py`), whose
  `model_error_floor_px` lives with the rest of the technique tuning
  (Section 3).
- Confidence: a sigmoid-combination `ConfidenceSpec` LOADED from
  `config_510_techniques.yaml` via `load_confidence_spec` onto the
  class's `confidence_spec` attribute — the base class defaults it
  to None and no technique hardcodes a spec; `BodyBlobNav`
  (`src/spindoctor/nav_technique/nav_technique_body_blob.py`) shows
  the consumption pattern inside `navigate`. Declare the companion
  `confidence_attributes` ClassVar naming the diagnostics fields the
  terms may reference (`validate_registered_confidence_specs`
  enforces this). Terms:
  `symmetry_peak_score`, `symmetry_valid_fraction`,
  `arc_inlier_fraction`, `arc_residual_rms_px`,
  `envelope_diameter_px`. Anchors start at Phase-A placeholders and
  are set by the Phase-E procedure; confidence stays under the
  program-wide `confidence_provisional` marker either way (final
  re-anchoring is #230's job).
- Diagnostics: new `TitanHazeDiagnostics` (Section 4) carries every
  gate input plus `fitted_haze_radius_km = R * km_per_px` and the
  filter names — recorded so a future haze-radius table (deferred,
  Section 9) can be built from production output.
- Rotation: `rotation_rad = None` and a 2x2 covariance (rotation is
  unobservable from a single quasi-circular feature) on the
  instruments that do not fit camera rotation — Cassini and LORRI,
  which is the whole Phase-E cohort. [Revised during Phase B: on an
  instrument where `NavContext.fit_camera_rotation` is True (VGISS,
  GOSSI) the result instead carries the rank-deficient `(3, 3)` form
  from `embed_rotation_unobservable` with `rotation_rad = 0.0` and the
  unobservable sigma, exactly as `BodyBlobNav` does for its equally
  rotation-blind centroid. The reason is not a crash — the
  mixed-DoF `ValueError` `_combine_precision_weighted` raises IS
  caught by `ensemble`, which converts it into
  `NavResult.failed(UNOBSERVABLE_OFFSET)` — but that outcome throws
  away a whole Titan-plus-star Voyager or Galileo frame under a status
  that misdescribes it: the offset was perfectly observable, the two
  results just could not be fused. Matching the fleet's DoF convention
  costs nothing and keeps those frames navigable. The physical claim —
  no rotation evidence — is unchanged; only its encoding is.]
- Technique attributes: `tier = 'primary'` (Titan has no other
  estimator; supersession semantics of the fallback tier would be
  wrong here) and `accepts_feature_types =
  frozenset({NavFeatureType.TITAN_LIMB})`.

### 2.5 Required failure behavior

- **Frame-quality problems are reliability, not statuses.** When
  Titan is in the extfov inventory (real obs), the model ALWAYS
  emits its `TITAN_LIMB` feature; there is no separate decline path
  and no Titan-specific status reason — a marginal Titan is treated
  exactly like a marginal star field or a five-pixel moon.
  Reliability encodes frame quality (the Phase B
  sigmoid-times-occlusion formula) and is forced to exactly 0.0
  under the hard conditions: the envelope disc dilated PER IMAGE AXIS
  by that axis's own extfov margin does not fit inside the extended
  frame (full visibility is a property
  of Titan's TRUE position, which can sit anywhere in the window — a
  predicted-visible but actually-clipped frame would fit sky);
  `occluded_fraction > max_occluded_fraction`; or envelope diameter
  below `min_envelope_diameter_px`.
  [Revised during Phase E; pending operator ratification. The
  original text dilated by the scalar `W`, which is the LARGER of the
  two margins. Since the extended frame is the detector plus those
  two margins, an axis-matched dilation makes this test say exactly
  "the envelope clears the detector" — the physical statement
  intended — while the scalar version says "clears the detector,
  shrunk on the tighter axis by the difference between the margins",
  which is 90 px per side on a Cassini NAC (50 rows against 140
  columns) and states nothing physical. Measured on the Phase E
  cohort: eleven `clean` frames were hard-zeroed by this condition,
  and eight of them have Titan entirely inside the detector. The
  scalar `W` is unchanged everywhere else; see Section 2.1 item 6.]
  A 0.0 can never pass the 0.30
  TITAN_LIMB type threshold, so a hard condition is exactly as
  strong as a hard decline — but it flows through the EXISTING
  reliability-gate machinery (`FeatureReliabilityGate` /
  `GatedFeatureRecord` in `src/spindoctor/feature/reliability.py`)
  and the frame resolves through the standard statuses: a Titan-only
  frame with a gated feature ends `ALL_FEATURES_GATED`, like any
  other marginal scene. Attribution lives where it does for every
  family: the typed reliability breakdown
  (`titan_envelope_diameter_px`, `titan_occluded_fraction`) on the
  feature and its gate record.
- **`geometry_from_obs` never raises — the always-emit invariant
  depends on it.** The orchestrator's plugin sandbox DROPS a model
  whose `create_model` throws and treats a raising `to_features` as
  zero features (`_build_models` / `_extract_features` in
  `orchestrator.py` swallow exceptions), which would end a
  Titan-only frame `NO_FEATURES_EXTRACTED` with no gate record — an
  unattributable failure, on exactly the clipped/off-edge frames
  the hard-zero visibility condition exists for. On any geometry
  pathology (empty surface-intercept set, backplane failure,
  degenerate radii), the model still emits the feature with
  `axis_degenerate = True`, `theta = 0.0`, defensible default
  geometry, and reliability forced to 0.0. Phase B includes a
  pathological-geometry unit test asserting emit-not-raise.
- **`TITAN_UNSUPPORTED` is DELETED, not renamed.** Titan is
  supported after this work, and a per-body status reason is a
  special case whose only justification — attributing a categorical
  capability hole — this plan removes; other bodies get no such
  status and neither does Titan. Delete the enum member from
  `src/spindoctor/support/status_reason.py` (count 20 -> 19; update
  the name list and count assertion in
  `tests/spindoctor/support/test_status_reason.py`), its template in
  `src/spindoctor/nav_orchestrator/status_reason_info.py`, and the
  whole orchestrator special case (`_titan_in_models` and the
  `titan_present` branch in
  `src/spindoctor/nav_orchestrator/orchestrator.py`); the model's
  `titan_in_fov` property goes with them. The deletion is clean, not
  shimmed: the repo convention is no backwards-compat shims, and
  nothing pins the old string (no library sidecar contains Titan,
  `ALLOWED_STATUS_REASONS` derives from the enum, the stats layer
  stores status reasons as free text, and historical per-image JSONs
  keep the old value as inert data).
- **Metadata reachability (generic, not Titan-specific).** Gate
  attribution must be readable from the per-image JSON. Verified
  scope: it is NOT serialized today, and the extension is
  three-file, not one — `NavResult` retains only `NavFeatureSummary`
  rows, which carry scalar `reliability`, `gated`, and the
  `gate_reason` string but not the breakdown. Phase B therefore
  adds a `reliability_reasons` field to `NavFeatureSummary`
  (`src/spindoctor/nav_orchestrator/feature_summary.py` — the path
  this plan gave as `src/spindoctor/feature/feature_summary.py` does
  not exist; corrected during Phase B), populates it where
  the orchestrator builds the feature inventory, and serializes it
  in `_curate_feature_summary` (`curator.py`) — generically, for
  ALL feature types. No bespoke `titan` metadata block.
- **Technique-side failures** (any Section 2.2/2.3 gate): a spurious
  `NavTechniqueResult` via `NavTechnique._spurious_result` with the
  failed gate named in diagnostics. The frame then resolves through
  the standard generic reasons: a Titan-only frame with a spurious
  result ends `ALL_TECHNIQUES_SPURIOUS`. These are deliberate,
  correct outcomes — the frame got its chance and failed measurably
  — and are not special-cased.
- Net invariant: a Titan frame never produces an unattributable
  failure. Every path ends in a committed result, a technique-gate
  name in diagnostics, or a `TITAN_LIMB` gate record whose
  reliability breakdown says why — all under the standard status
  reasons, with no Titan-specific status in the vocabulary.

## 3. Architecture: files

This section is the complete file manifest with one-line change
summaries; the phase narratives (Section 6) carry the normative
detail for each. A file a phase touches that is missing here is a
plan bug — fix the manifest in the same commit (Section 7 item 3).

New files:

| File | Contents |
|---|---|
| `src/spindoctor/nav_technique/titan_fitting/` | Pure fitting library, split into a package under the sizing note below because the single module ran past 1000 lines: `grid.py` (axis unit vectors, rotated-grid resample, the shared array helpers), `symmetry.py` (mirror-correlation scan with angle refinement, its params/result), `arc.py` (radial profiles, limb-gradient extraction, constrained robust circle fit, its params/result), `driver.py` (`fit_titan_center`, the two-pass sequence), and `__init__.py` re-exporting the whole surface so consumers import `spindoctor.nav_technique.titan_fitting`. No oops, no NavContext, no config reads — plain functions on arrays plus the Section 4 parameter/result dataclasses. Everything unit-testable on synthetic arrays. |
| `src/spindoctor/nav_technique/nav_technique_titan_haze.py` | `TitanHazeNav(NavTechnique)`: `is_feasible`, `navigate` (Sections 2.2-2.4, math delegated to `titan_fitting`), `confidence_spec` + `confidence_attributes`, `tier`, `accepts_feature_types`, tuning load; plus the Phase-C gate table logged inside the `TECHNIQUE: TitanHazeNav` section (one info line per Section 2.2.6 / 2.3.7 gate with its measurement, its threshold, and `PASS` / `FAIL` / `EDGE` / `SKIP`). |
| `src/spindoctor/nav_model/titan_geometry.py` | Added during Phase B under the sizing note below: `TitanGeometryInputs` plus `geometry_from_obs` and every oops / star-catalog helper behind it, because the model plus its geometry extraction ran past 1000 lines in one file. `nav_model_titan.py` keeps the reliability formula, the feature build, and the NavModel class, all pure functions of `TitanGeometryInputs`. |
| `tests/spindoctor/nav_technique/test_titan_fitting.py` | Phase-A unit tests. |
| `tests/spindoctor/nav_model/test_nav_model_titan.py` | Phase-B model tests (reliability, feature payload, contaminant mask, never-raise); Phase C adds the overlay-rasterization and annotation tests. Added to this manifest during Phase C, which touched it: the file shipped with Phase B but the manifest never listed it. |
| `tests/spindoctor/nav_technique/test_nav_technique_titan_haze.py` | Phase-B technique tests (use the `FakeObs` fixture pattern from `tests/spindoctor/nav_technique/conftest.py`; do not instantiate a real `ObsSnapshot` for unit tests); Phase C adds the gate-table tests -- capsys over a navigated frame, plus a table-driven pass over the row builders with a synthetic fit per named gate. |
| `tests/integration/test_titan_haze_nav.py` | Phase-B integration tests (marked, holdings-fetched): model emission and technique execution on `W1822132529_1`, including the real YBSC / Tycho-2 star-mask queries. |
| `util/titan_cohort/` | Phase E: the real-frame campaign. `titan_images.csv` (the legacy cohort list vendored into the repo, Section 6 Phase E step 1), `cohort.py` (flags plus holdings/epoch/filter resolution from the PDS3 volume indexes), `collect.py` (full-pipeline batch run), `analyze.py` (the four acceptance bounds plus the evidence tiers, resolved onto the technique's own axes), `build_review_batch.py` and `review_batch/` (the pending operator overlay review), `build_nominations.py` and `nominations/` (draft library sidecars, pending operator votes), `final_run_summary.csv` (one committed row per frame of the shipped configuration's run), `README.md`, `CAMPAIGN_20260726.md`. |

Modified files:

| File | Change |
|---|---|
| `src/spindoctor/nav_model/nav_model_titan.py` | `create_model` computes Section 2.1 geometry (preserving the existing `TITAN MODEL` logger section); `to_features` ALWAYS emits one `TITAN_LIMB` feature with Section 2.5 reliability (hard-zero conditions included); the `titan_in_fov` property is deleted; `to_annotations` (Phase C); docstrings rewritten — they currently say the model always declines. The `is_simulated -> []` branch stays until Phase D. |
| `src/spindoctor/nav_model/nav_model_body.py` | Extract the body-occluder computation into the module-level `occluder_mask_for_body` helper (Section 2.1 item 5); `NavModelBody` behavior bit-identical, guarded by its existing tests. |
| `src/spindoctor/feature/geometry.py` | New `TitanHazeGeometry` dataclass (Section 4); add to the geometry union at the bottom of the file. |
| `src/spindoctor/feature/flags.py` | New `TitanHazeFlags` (Section 4); add to the flags union. |
| `src/spindoctor/feature/feature.py` | Two new optional `NavReliabilityBreakdown` fields: `titan_envelope_diameter_px: float \| None = None`, `titan_occluded_fraction: float \| None = None` (this dataclass lives here, not in `reliability.py`). |
| `src/spindoctor/feature/feature_type.py` | `TITAN_LIMB` docstring: no longer "never emitted". |
| `src/spindoctor/nav_technique/diagnostics.py` | New `TitanHazeDiagnostics` incl. its `CURATOR_FIELDS` tuple (every diagnostics class declares one); add to the union. |
| `src/spindoctor/nav_technique/__init__.py` | Import + `__all__` for `TitanHazeNav`, `TitanHazeDiagnostics` (importing registers the class via `__init_subclass__`). |
| `src/spindoctor/feature/__init__.py`, `src/spindoctor/nav_model/__init__.py` | Exports, mirroring existing entries. |
| `src/spindoctor/config_files/config_060_titan.yaml` | Section 5 schema (complete replacement; `atmosphere_height` currently has no code consumer, so the replacement is safe). |
| `src/spindoctor/config_files/config_510_techniques.yaml` | `techniques.TitanHazeNav.tuning`: `model_error_floor_px` plus the confidence-spec coefficients, alongside the existing techniques' entries. (NOT config_540; nothing there reads technique tuning.) |
| `src/spindoctor/support/status_reason.py`, `src/spindoctor/nav_orchestrator/status_reason_info.py`, `src/spindoctor/nav_orchestrator/orchestrator.py`, `tests/spindoctor/support/test_status_reason.py` | DELETE `TITAN_UNSUPPORTED` (enum member, template entry, the `_titan_in_models` + `titan_present` orchestrator special case, and the test name list + count assertion AND its "Exactly 20 values" docstring, 20 -> 19), per Section 2.5. Clean deletion, no shim; nothing pins the old string. |
| `src/spindoctor/sim/forward/atmosphere.py` + sim schema/boundary files (Phase D) | Extend the existing body `atmosphere` block (haze is an atmosphere block on a body element — do NOT invent a new top-level scene element). Phase D added the sibling `src/spindoctor/sim/forward/haze_structure.py` under the Section 3 sizing note: the spec, its parsing, and the per-pixel field builders for the six symmetry-breaking keys did not fit in `atmosphere.py` alongside the column arithmetic. `atmosphere.py` keeps `apply_atmosphere` and calls into it. |
| `tests/integration/sim_sweeps/titan_offset_fine.yaml`, `.../titan_offset_wide.yaml` (Phase D, new) | Dense sub-pixel + wide-range offset sweep specs pinning `TitanHazeNav`, matching every other technique's entries. Corrected during Phase D: this plan said the entries go in `tests/integration/sim_sweep.py`, which is the harness (schema, loader, runner) and hard-codes no sweeps; every sweep is its own YAML under `sim_sweeps/`. |
| `tests/integration/sim_scenes/atmosphere/titan_haze.yaml` + `sim_baselines/titan_haze.json` (Phase D, new) | The `titan_haze` base scene both sweeps drive, and its regression baseline (`test_every_scene_has_a_baseline` / `test_no_orphan_baselines` fire without it). |
| `tests/integration/render_diffs/current/titan_haze.png`, `.../sheet_atmosphere.png` (Phase D) | The standing render-review artifacts, regenerated with `python -m tests.integration.render_contact_sheet`. `test_render_diffs` byte-compares every `current/` PNG against a fresh render, so a new scene without one is a red; the class sheet gains its review row. Added to this manifest during Phase D. |
| `docs/simulator_report/_figures/offset_accuracy_fine.png`, `.../offset_accuracy_wide.png` (Phase D) | Regenerated so the report's per-technique offset curves carry `TitanHazeNav`. |
| `tests/spindoctor/sim/test_sim_haze_structure.py` (Phase D, new) | One render-level test per new atmosphere key asserting the effect is measurably present, the gating contract (a block naming no structure key leaves the spec structureless), and the schema validation of the structure keys. |
| `tests/spindoctor/nav_model/test_nav_model_titan_simulated.py` (Phase D, new) | Geometry translation, the contaminant-mask components, and the inherited emission path of `NavModelTitanSimulated`. |
| `tests/spindoctor/nav_model/test_sim_model_selection.py` (Phase D) | Titan routing in both directions (`TITAN` -> the haze model only; every other body unaffected), plus the unconfigured-scene cases. |
| `tests/integration/sim_sweep_plots.py` (Phase D) | Added to this manifest during Phase D. Its `_OFFSET_TECHNIQUES` list is hand-enumerated exactly like `technique_snr_characterization.py`'s `_TECHNIQUES`, and it is what draws the simulator report's `offset_accuracy_fine` / `offset_accuracy_wide` figures; omission silently drops Titan from them while every sweep still runs green. |
| `src/spindoctor/nav_model/nav_model_titan_simulated.py` (Phase D, new) | `NavModelTitanSimulated`, a subclass of `NavModelTitan` that replaces only how `TitanGeometryInputs` is obtained (operator parameters instead of `oops`), so feature emission, reliability, the hard-zero conditions, and the overlay are inherited rather than reimplemented. |
| `src/spindoctor/nav_model/titan_geometry.py` (Phase D, E) | Phase D: `_paint_disc` / `_occluded_fraction` promoted to the public `paint_disc` / `occluded_disc_fraction` so the simulated model builds its contaminant mask and occluded fraction with the same code the real one does. Phase E: `TitanGeometryInputs` gains `extfov_margin_vu` and `_frame_bounds` returns it, so the Section 2.5 visibility test can dilate per image axis (Section 2.1 item 6). |
| `src/spindoctor/nav_model/nav_model_titan.py`, `.../nav_model_titan_simulated.py` (Phase E) | `_envelope_fits_in_frame` dilates each image axis by that axis's own extfov margin instead of by the scalar search half-window; the simulated model passes its margins through. Two unit tests cover the asymmetric-margin case in both directions. |
| `tests/integration/image_library/images/README.txt`, `tests/integration/sim_realism.py` (Phase E) | `TitanHazeNav` moves from the not-yet-implemented list to the usable one; the model-glob map gains `titan:*` / `titan_sim:*` under a `titan_haze` key. The key is inert until a scene class is declared, so it costs nothing if the operator files Titan frames under an existing class instead. |
| `util/titan_truth/` (Phase D, new; extended in Phase E) | The planted-truth campaign: `scene_gen.py`, `collect.py`, `analyze.py`, `README.md`. Phase E adds the full per-row diagnostics payload to `collect.py` (a candidate confidence spec cannot be scored offline without the terms it reads) and the sibling `fit_confidence.py`, which fits the technique's confidence anchors on those rows and verifies the shipped ones against the no-confident-wrong checks. |
| `docs/simulator_report/simulator_report.rst` (Phase F) | Titan base scene, `TitanHazeNav` row in the per-technique offset-sweep table, regenerated response curves. |
| `src/spindoctor/nav_orchestrator/feature_summary.py`, `src/spindoctor/nav_orchestrator/orchestrator.py` (inventory builders), `src/spindoctor/nav_orchestrator/curator.py` (Phase B) | Breakdown serialization, three files (Section 2.5 verified scope): `reliability_reasons` field on `NavFeatureSummary`, populated at inventory build, serialized by `_curate_feature_summary` — generic for ALL feature types; acceptance criterion 3 depends on it. No Titan-specific block. |
| `tests/spindoctor/nav_orchestrator/test_orchestrator.py` (Phase B) | Two tests exercise the deleted path (`test_orchestrator_titan_only_yields_titan_unsupported`, `test_orchestrator_titan_plus_stars_navigates_normally`) via a `_FakeTitanModel` exposing `titan_in_fov`; rewrite both (and the fake) to the Section 2.5 status matrix (`ALL_FEATURES_GATED` with a `TITAN_LIMB` gate record / normal navigation). |
| `src/spindoctor/feature/composition.py`, `src/spindoctor/nav_technique/nav_technique_manual.py` (Phase C) | Manual-nav support: add a `TitanHazeGeometry` branch to `compose_dialog_overlay` (envelope-circle outline at `r_env_px` around `predicted_center_vu`, following the `BodyBlobGeometry` branch pattern) and to `NavTechniqueManual.is_feasible`'s renderable-feature count. Both enumerate geometry types by hand; without both branches, manual navigation is impossible on a Titan-only frame. |
| `tests/spindoctor/feature/test_composition.py`, `tests/spindoctor/nav_technique/test_nav_technique_manual.py` (Phase C) | Manual-nav coverage for the two branches above: the composed circle's radius and clipping, and a Titan-only feature set that is feasible and paints a non-empty drag overlay. |
| `src/spindoctor/nav_technique/titan_fitting/arc.py`, `.../__init__.py` (Phase C) | Promote the arc-radius band constants to the package surface (`ARC_RADIUS_MIN_FRACTION` / `ARC_RADIUS_MAX_FRACTION`) so the technique's gate table quotes the band the gate itself tests, from one definition. Behavior unchanged. |
| `tests/shims/obs.py`, `tests/shims_tests/test_shims_self.py` (Phase C) | `FakeObs.extract_offset_array`, mirroring `ObsSnapshot`'s semantics (window extraction at an offset, zero-filled past the extfov), so the overlay-render test can composite an annotation the way the summary PNG does; plus its shim self-tests. |
| `src/spindoctor/nav_model/nav_model_body_simulated.py` (Phase D) | Exclude `TITAN` from simulated body-model selection (mirror of the real path's exclusion) once `NavModelTitanSimulated` exists — today it builds a model for EVERY body, so both models would claim a sim Titan. |
| `tests/integration/sim_scenes/atmosphere/titan_haze_limb.yaml`, `.../titan_crescent_horns*.yaml`, `tests/integration/sim_scenes/model_mismatch/haze_limb_base.yaml` + their `sim_baselines/*.json` (Phase D) | Rename the body away from `TITAN` (e.g. `HAZEMOON`) and regenerate baselines via `update_sim_baselines` with diff review. These scenes are body-navigation fidelity records premised on a haze-BLIND navigator; without the rename the new routing flips them to `TitanHazeNav`, breaking exact-match baselines, invalidating the crescent-horns rationale comment, and destroying the `atmosphere_haze` model-mismatch sweep's purpose. |
| `tests/spindoctor/nav_technique/test_diagnostics.py` (Phase B) | Add `TitanHazeDiagnostics` to BOTH hard-coded parametrize lists (defaults-construct + `CURATOR_FIELDS` completeness) — the lists are not auto-discovered, and `curator.assert_diagnostic_fields_present` raises at metadata-build time on any mismatch. |
| `tests/spindoctor/nav_technique/conftest.py` (Phase B) | Add a `make_titan_feature` factory alongside the existing per-type feature factories. |
| `tests/integration/technique_snr_characterization.py` (Phase D) | Add the Titan entry to the hard-coded `_TECHNIQUES` list (entries are `(label, base_scene_path, technique_name, marker_char)` 4-tuples) — a second report-feeding harness beyond `sim_sweep.py`; omission silently drops Titan from the response curves. |
| `util/calibration/fit.py`, `util/calibration/fit_floors.py`, `util/calibration/scene_gen.py` (Phase E) | Enroll `TitanHazeNav` in the hand-enumerated `TECHNIQUES` tuples of both fitters (plus `SIGN_BY_FEATURE` entries: `arc_residual_rms_px` negative, the quality terms positive) and add a `titan` scene family + generator — otherwise the calibration campaign yields zero Titan rows and the Phase-B placeholder coefficients are never recalibrated. |
| `docs/user_guide/...`, `docs/dev_guide/...` | Phase F — the itemized checklist in Phase F is normative, not the generic row here. |

Reliability gating context the implementer must know: the `TITAN_LIMB:
0.30` entries under `reliability_gate:` in
`config_540_orchestrator.yaml` AND in `DEFAULT_RELIABILITY_THRESHOLDS`
(`src/spindoctor/feature/reliability.py`) are a per-feature-type
minimum reliability — features below it are gated out before any
technique runs (ending `ALL_FEATURES_GATED` if nothing else is
navigable). They are not an ensemble weight; no per-feature-type
ensemble weight exists. Leave both entries at 0.30 (they must stay in
sync), and see the reliability formula in Phase B, which is aligned to
this threshold.

Sizing note: keep `titan_fitting.py` and the technique module each
under 1000 lines; if the fitting library grows past that, split into a
`titan_fitting/` package (symmetry / arc modules). The same cap
applies to `nav_model_titan.py`, which is why Phase B moved the
observation-side geometry into the sibling `titan_geometry.py` — with
`to_annotations` still to come in Phase C, one file could not hold both
halves.

## 4. New dataclasses (signatures)

```python
@dataclass(frozen=True)
class SymmetryFitParams:
    annulus_inner_fraction: float
    annulus_outer_pad_px: float
    angle_refine_deg: float
    angle_refine_step_deg: float
    angle_refine_min_gain: float
    min_peak_score: float
    min_valid_fraction: float
    max_second_peak_ratio: float
    cross_sigma_scale: float
    sigma_floor_cross_px: float

@dataclass(frozen=True)
class SymmetryFitResult:
    cross_track_px: float          # positive along c_hat
    sigma_cross_px: float
    theta_rad: float               # possibly refined
    peak_score: float
    valid_fraction: float
    second_peak_ratio: float       # 0.0 when no competing peak
    at_edge: bool
    gate_failed: str | None        # None, or a Section 2.2 gate name

@dataclass(frozen=True)
class ArcFitParams:
    sector_half_angle_deg: float
    ray_step_deg: float
    radial_step_px: float
    radial_inner_fraction: float
    radial_outer_pad_px: float
    median_filter_samples: int
    min_gradient_snr: float
    min_rays: int
    min_inlier_fraction: float
    max_residual_rms_px: float
    tukey_c: float
    along_sigma_scale: float
    sigma_floor_along_px: float

@dataclass(frozen=True)
class ArcFitResult:
    along_track_px: float          # positive along a_hat
    sigma_along_px: float
    radius_px: float
    n_rays_total: int
    n_rays_inlier: int
    residual_rms_px: float         # NaN when zero rays survive
                                   # reweighting (never 0.0, which a
                                   # falling confidence sigmoid would
                                   # read as maximally good); Phase B
                                   # maps NaN to None at the
                                   # diagnostics boundary so strict
                                   # JSON serialization never sees a
                                   # bare NaN
    at_edge: bool
    gate_failed: str | None        # None, or a Section 2.3 gate name
```

Fitting-library function signatures (all re-exported from the
`titan_fitting/` package; pure, array-in/dataclass-out):

The leading `(image, valid_mask, center_vu)` group is positional;
everything after it is keyword-only, per the project signature rule
(groups larger than ~5 go keyword-only):

```python
def resample_rotated_grid(image, valid_mask, center_vu, *,
                          theta_rad, s_half_extent_px,
                          t_half_extent_px): ...
    # -> (grid, grid_valid); grid axes (s, t) as defined in 2.2.1

def symmetry_scan(image, valid_mask, center_vu, *, contaminant_mask,
                  theta0_rad, r_env_px, window_px, pass_pad_px,
                  capsule_half_extent_px=0.0,
                  mask_shift_vu=(0.0, 0.0), params): ...
    # -> SymmetryFitResult; owns the resampling internally (calls
    # resample_rotated_grid per candidate theta), so no callback
    # parameter is needed.  valid_mask is static validity
    # (out-of-frame, detector defects); contaminant_mask (None
    # allowed) is hypothesis-riding per Section 2.1, dilated along
    # t by pass_pad_px (W in pass 1, recenter_threshold_px in the
    # recenter pass) and read c-shifted.
    # capsule_half_extent_px selects the Section 2.2.2 annulus shape
    # (W in pass 1 for the capsule, 0.0 in the recenter pass for the
    # tight annulus): pass_pad_px alone cannot express both, since it
    # must be recenter_threshold_px for the pass-2 mask.
    # mask_shift_vu is the displacement already applied to center_vu
    # relative to the geometry the mask was built at, so the mask
    # stays anchored at the predicted center per Section 2.1.

def radial_profiles(image, valid_mask, center_vu, *, contaminant_mask,
                    mask_shift_vu, axis_dir_vu, pass_pad_px,
                    phi_rad_list, r_start_px, r_stop_px, r_step_px): ...
    # -> (profiles, profile_valid); contaminant_mask is shifted by
    # mask_shift_vu (the accumulated center hypothesis) and dilated
    # along a_hat by pass_pad_px before sampling.  axis_dir_vu names
    # that a_hat: the ray angles alone do not determine it, and an
    # isotropic dilation of radius W would over-mask badly.

def limb_radii_from_profiles(profiles, profile_valid, *, r_start_px,
                             r_step_px, r_solid_px, window_px_lo,
                             window_px_hi, params): ...
    # -> (rho_px per ray, ray_ok mask); r_solid_px is what makes the
    # Section 2.3.2 rule "interior samples may be masked without
    # harm" expressible here -- it separates the interior from the
    # limb region the ray-drop rule polices.

def constrained_circle_fit(points_vu, axis_origin_vu, axis_dir_vu, *,
                           r_solid_px, r_env_px, window_px, params): ...
    # -> ArcFitResult; the three radii/window scalars carry the
    # Section 2.3.7 arc_radius band, the |d| < W at_edge test, and
    # the sigma clamp, none of which live in ArcFitParams.

def fit_titan_center(image, valid_mask, center_vu, *,
                     contaminant_mask, theta0_rad, r_solid_px,
                     r_env_px, window_px, sym_params, arc_params,
                     recenter_threshold_px): ...
    # -> (SymmetryFitResult, ArcFitResult, offset_vu, recentered)
    # The full Section 2.2-2.3 sequence including the recenter pass;
    # the technique is a thin wrapper around this driver, so the
    # complete algorithm is testable without oops.
```

`TitanHazeGeometry` (in `geometry.py`): `predicted_center_vu:
tuple[float, float]` (extended frame), `sun_angle_rad: float`,
`axis_degenerate: bool`, `phase_deg: float`, `r_solid_px: float`,
`r_env_px: float`, `km_per_px: float`, `contaminant_mask:
NDArrayBoolType | None` (the UNDILATED four-component union of
Section 2.1 item 5 at predicted geometry, shipped as a full
extended-frame-shaped boolean array so the fitting signatures need
no bbox-origin parameter; None when nothing is masked — hypothesis
alignment and along-track dilation are the fitting code's job, not
the model's), `filters: tuple[str, ...]`,
`bbox_extfov_vu: tuple[int, int, int, int]` (the envelope bbox).
[The bbox field was added during Phase B: `_bbox_from_geometry` in
`orchestrator.py` reads `bbox_extfov_vu` off EVERY geometry variant
when building the feature inventory, and that read is not sandboxed,
so a payload without it would break the orchestrator's never-raise
contract on the first Titan frame.]

`TitanHazeFlags` (in `flags.py`): `body_name: str = ''` (set to
`'TITAN'`; the field is required for feature-level body attribution —
`NavFeature.body_name` reads it off the flags dataclass like every
body-feature flags class), `surface_window_filter: bool = False`,
`high_phase: bool = False`.

`TitanHazeDiagnostics` (in `diagnostics.py`, docstring style of
`BodyBlobDiagnostics`, with `CURATOR_FIELDS`): `sun_angle_deg`,
`axis_degenerate`, `phase_deg`, `envelope_diameter_px`,
`cross_track_px`, `along_track_px`, `symmetry_peak_score`,
`symmetry_valid_fraction`, `symmetry_second_peak_ratio`,
`theta_refined_deg` (delta vs SPICE), `arc_rays_total`,
`arc_rays_inlier`, `arc_inlier_fraction`, `arc_residual_rms_px`,
`fitted_haze_radius_km`, `filters`, `recentered`, `gate_failed`.

`TitanGeometryInputs` (in `nav_model/titan_geometry.py`; the plan
originally placed it in `nav_model_titan.py`, and Phase B moved it with
the rest of the observation-side half under the Section 3 sizing note):
the frozen dataclass separating oops access from reliability/feature
logic — fields and testing role specified in Phase B.

Feature identity: `feature_id = 'titan_limb:TITAN'` (the documented
`<type_lc>:<scope>` format for `TITAN_LIMB`);
`usable_types = frozenset({NavFeatureType.TITAN_LIMB})`;
`source_model` the model name; the technique sets
`source_bodies = frozenset({'TITAN'})` on its result.

## 5. Config schema (`config_060_titan.yaml`, complete replacement)

```yaml
titan:
  # Haze envelope above the solid radius, km. Bounds the search
  # annulus and windows; the fit itself does not assume a haze
  # altitude.
  atmosphere_height: 700
  navigation:
    # Hard-zero reliability floors (Section 2.5): beyond these the
    # feature is still emitted but with reliability 0.0, so the
    # standard TITAN_LIMB type gate always removes it.
    min_envelope_diameter_px: 40.0
    max_occluded_fraction: 0.10
    ring_occlusion_radii_km: [74490.0, 140500.0]
    axis_min_offset_px: 3.0
    recenter_threshold_px: 8.0
    star_mask_vmag_limit: 8.0
    star_mask_radius_px: 4.0
    # Reliability sigmoid (Phase B): chosen so reliability crosses
    # the 0.30 TITAN_LIMB type gate near the hard-zero floor above,
    # keeping the two gates consistent.
    reliability_diameter_midpoint_px: 52.0
    reliability_diameter_scale_px: 14.0
    # Cassini filters that see through to the surface; sets the
    # surface_window flag (diagnostic + future refinement only).
    surface_window_filters: [CB3]
    # Phase angle (deg) setting the high_phase feature flag. [Added
    # post-review: originally a module constant; moved to config with
    # the titan.annotation styling keys below.]
    high_phase_deg: 150.0
    symmetry:
      annulus_inner_fraction: 0.55
      annulus_outer_pad_px: 6.0
      angle_refine_deg: 5.0
      angle_refine_step_deg: 0.5
      angle_refine_min_gain: 0.02
      min_peak_score: 0.60
      min_valid_fraction: 0.50
      max_second_peak_ratio: 0.90
      # 0.10 set by the Phase D free-row solve (was 1.0 pre-calibration).
      cross_sigma_scale: 0.10
      sigma_floor_cross_px: 0.30
    arc:
      # 60.0 through Phase D; widened by Phase E on real-frame evidence.
      sector_half_angle_deg: 80.0
      ray_step_deg: 2.0
      radial_step_px: 0.5
      radial_inner_fraction: 0.80
      radial_outer_pad_px: 6.0
      median_filter_samples: 5
      # 4.0 through Phase D; raised by Phase E (#396) at no real-frame cost.
      min_gradient_snr: 8.0
      min_rays: 20
      min_inlier_fraction: 0.50
      max_residual_rms_px: 2.0
      tukey_c: 4.685
      along_sigma_scale: 1.0
      sigma_floor_along_px: 1.00
  annotation:
    # Overlay styling. [Added post-review: originally module constants
    # in nav_model_titan.py; configurable because bodies.outline_thicken
    # interacts with the dot gaps.]
    gated_dot_spacing: 4
    center_marker_half_px: 4
```

Access via `config.titan` (top level is an `AttrDict`; everything
below it is plain dicts — see Section 1). The technique's
`model_error_floor_px` and confidence coefficients go in
`config_510_techniques.yaml` under `techniques.TitanHazeNav.tuning`,
not here. Defaults above are Phase-A engineering estimates; Phase E
owns re-tuning them, and every change lands as a reviewed config
commit with the sweep evidence linked.

## 6. Phases

Each phase is one implementer-subagent slice with its own independent
review (Section 7; both subagents on the Opus model per the Section 0
requirement). A phase is done when its tests pass, `ruff check
src tests`, `ruff format --check src tests`, and `mypy src tests` are
clean, and its acceptance line is demonstrably true.

### Phase A — fitting library

Files: the `titan_fitting/` package (Section 3),
`test_titan_fitting.py`. Implement the Section 4 signatures per the
Section 2.2-2.3 math.

Tests (synthetic arrays, no oops, fast):

1. A rendered symmetric haze disc (logistic falloff at radius R, ramp
   along the axis) displaced by known sub-pixel `(dv, du)` at several
   angles theta: recovered cross-track within 0.05 px, along-track
   within 0.2 px, over a grid of 5 offsets x 4 angles.
2. Same with additive Gaussian noise at SNR ~ 20: cross-track within
   0.15 px, along-track within 1.5 px. [bound revised from 0.5 px
   during Phase A: unreachable with the prescribed estimator
   (measured P95 1.01 px at SNR 20, confirmed independently by
   implementer and reviewer); consistent with the Phase D P95 <= 3.0
   px and Section 8 <= 3 px bounds; pending operator ratification]
3. A Gaussian "cloud" blob injected off-axis (amplitude 30% of disc):
   both recoveries degrade by less than 2x the case-2 bounds.
4. A north-south brightness gradient across the disc interior: the
   annulus restriction keeps cross-track error within case-2 bounds.
5. Void guard: disc half off the grid -> `gate_failed ==
   'valid_fraction'`, not a wrong answer.
6. Second-peak guard: two discs side by side -> `gate_failed ==
   'second_peak'`.
7. Circle fit: exact circle points + 20% outliers -> center recovered
   within 0.05 px, outliers zero-weighted; degenerate short arc
   (sector 10 degrees) -> a gate failure or `sigma_along` > 3 px,
   never a confident wrong `d`.
8. Every named gate in Sections 2.2.6 and 2.3.7 exercised at least
   once (one test per gate), asserting the exact `gate_failed`
   string.
9. Sign conventions: a disc displaced purely along `+c_hat` yields
   positive `cross_track_px` and near-zero `along_track_px`, and
   symmetrically for `+a_hat`; assembled `(dv, du)` per Section 2.4
   equals the planted displacement within case-1 bounds.
10. Beside-limb intruder: a second small disc (diameter ~0.2x, same
    peak brightness) beside the main disc's limb, with the WHOLE
    scene (both discs) displaced by a planted offset while the
    intruder's entry in `contaminant_mask` stays at its predicted
    (undisplaced) position. Hypothesis-riding alignment must still
    exclude it: recovery meets case-2 bounds. With no mask at all,
    recovery stays within 2x the case-2 bounds or fails a named gate
    — never an unflagged wrong answer.
11. Point sources: ~20 single-pixel spikes at 5x disc brightness
    scattered uniformly (unmasked, per the Section 2.1 policy):
    recovery within case-2 bounds.
12. Recenter: planted along-track displacement of `0.8 * W` through
    `fit_titan_center`: recovered within case-2 bounds and
    `recentered` is True; a displacement below
    `recenter_threshold_px` leaves `recentered` False. Include one
    case with `W = 3 * r_env_px` (small disc, large window): only
    the pass-1 capsule annulus gives that case any signal, so it is
    the direct regression test for the capsule. Include one case
    planting BOTH components (`c_true != 0` with the recenter
    triggered): the assembled cross-track must equal the final
    pass's `c_sub` alone — the double-count regression of Section
    2.4.

Acceptance: the twelve test families above pass; the package imports
nothing from spindoctor beyond `support.types` and the two
`dt_fitting` weighting helpers (`tukey_biweight_weights` and
`information_matrix_to_covariance`, both from
`dt_fitting/weights.py`; reusing the second avoids duplicating the
pseudoinverse covariance the robust fit needs).

### Phase B — model feature + technique (the vertical slice)

Files: `nav_model_titan.py` and `titan_geometry.py`, `nav_model_body.py`
(helper extraction), `geometry.py`, `flags.py`, `feature.py`,
`feature_type.py`, `diagnostics.py`, `nav_technique_titan_haze.py`,
registrations, `config_060_titan.yaml`, `config_510_techniques.yaml`,
plus the new test files.

- Model, structured for testability: `create_model` splits into
  `geometry_from_obs(obs, config) -> TitanGeometryInputs` (ALL oops
  access lives here, in the sibling `titan_geometry.py`; covered by the
  integration-marked real-frame test) and pure logic operating on
  `TitanGeometryInputs` (a frozen
  dataclass carrying center, radii, `km_per_px`, phase, theta,
  `axis_degenerate`, `occluded_fraction`, the contaminant mask, and
  the frame bounds). The reliability unit tests (hard-zero
  conditions included) construct `TitanGeometryInputs` directly — no
  oops, no `FakeObs` extension; `FakeObs` supplies only
  technique-side needs (it exposes an extfov margin, not inventories
  or backplanes) and cannot drive these conditions. Then: Section
  2.1 geometry; the Section 2.5 hard-zero conditions; reliability =
  `sigmoid((D - reliability_diameter_midpoint_px) /
  reliability_diameter_scale_px) * (1 - occluded_fraction)` where `D`
  is the envelope diameter in px and `sigmoid(x) = 1/(1+exp(-x))`,
  with the two new breakdown fields populated. With the Section 5
  defaults, reliability crosses the 0.30 TITAN_LIMB type threshold
  near the hard-zero floor — at D of about 40.1 px with zero occlusion,
  moving up to about 42.3 px at the maximum permitted
  `occluded_fraction` of 0.10. Frames in that narrow band emit a
  feature that the type gate then removes, ending
  `ALL_FEATURES_GATED`; this is a sanctioned terminal state
  (Section 2.5), not a defect — do not write a boundary test
  asserting that emit-then-gate cannot happen.
- Technique: `is_feasible` counts TITAN_LIMB features (infeasible
  with reason `'no TITAN_LIMB features'` when zero); `navigate` wires
  `NavContext.image_ext` (the raw extended image — NOT the
  gradient/DT planes the DT techniques consume) through Sections
  2.2-2.4 into a `NavTechniqueResult`. Recenter-pass totals: the
  assembled offset can legitimately exceed `W` with per-pass
  `at_edge` False (Section 2.3.7 gates per pass; Section 2.4 sums
  the passes — a truth just beyond `W` is recovered correctly via
  the recenter). The technique sets `at_edge` on its RESULT when
  either assembled component's magnitude reaches `W`, so the
  ensemble's conservative at-edge treatment applies to totals
  beyond the declared search bound. [Added during Phase A review;
  the fitting library reports per-pass at_edge only.]
- Confidence spec + `confidence_attributes` registered;
  `validate_registered_confidence_specs()` passes. The config_510
  entry must exist at Phase B with placeholder anchors (copy the
  structural shape of the `BodyBlobNav` entry): rising sigmoids on
  `symmetry_peak_score` (midpoint 0.70), `symmetry_valid_fraction`
  (0.60), `arc_inlier_fraction` (0.60), `envelope_diameter_px` (80),
  and a falling sigmoid on `arc_residual_rms_px` (midpoint 1.5),
  equal weights. These are placeholders in the literal sense — Phase
  E owns the real anchors — but they must load, validate, and
  produce monotone-sane values from day one.
- Status wording updates per Section 2.5.

Tests: reliability unit tests (hard-zero conditions) via directly
constructed `TitanGeometryInputs` (no oops); technique-path tests via
the `FakeObs` conftest pattern for the end-to-end run (planted
offset on a Phase-A rendered scene recovered through the full
NavTechnique interface within Phase-A case-2 bounds; covariance major
axis within 5 degrees of `a_hat`); an occlusion-helper refactor test
plus the existing `NavModelBody` suite unchanged; orchestrator-level
tests asserting the Section 2.5 status matrix — a hard-zero
condition on a Titan-only frame -> `ALL_FEATURES_GATED` with a
`TITAN_LIMB` gate record; emitted usable feature + all technique gates
failed -> `ALL_TECHNIQUES_SPURIOUS`; registry/config/result-validity
tests. Mask tests: a sibling moon beside Titan enters the
contaminant mask (undilated bbox) without counting toward
`occluded_fraction`; a nearer moon covering Titan counts toward
both; a catalog star brighter than `star_mask_vmag_limit` inside the
window contributes a masked disc, a fainter one does not (unit
tests monkeypatch `stars_in_extfov`; one integration-marked case
exercises the real YBSC/Tycho-2 queries and the nominal-to-extfov
conversion). Collateral (Section 3 table): `TitanHazeDiagnostics`
added to both `test_diagnostics.py` parametrize lists; a
`make_titan_feature` factory in the nav_technique conftest; a
metadata test asserting the `TITAN_LIMB` gate record, with its
reliability breakdown, appears in the emitted JSON for a gated frame
(Section 2.5 reachability, the three-file serialization extension);
a pathological-geometry test (clipped bbox with empty
surface-intercept set) asserting the model emits an
`axis_degenerate`, reliability-0.0 feature rather than raising; the
two rewritten orchestrator Titan tests (Section 3 table). Integration
(marked, holdings-fetched, no local cache exists):
model emission and technique execution on Cassini frame
`W1822132529_1` (unoccluded Titan, from the Phase E cohort list).

Acceptance: `sd_offset` on `W1822132529_1` runs the technique end to
end and produces a non-spurious result (offset quality is Phase E's
concern, not Phase B's); every hard-zero condition yields
`ALL_FEATURES_GATED` with an attributing gate record; every
technique gate yields
the correct generic status.

### Phase C — annotations and operator surface

`to_annotations` overlay: predicted envelope circle, symmetry axis,
disc-center mark + arc sector, styled by the feature's reliability; PNG
snapshot test following `tests/spindoctor/support/test_summary_png.py`;
keep the existing `TITAN MODEL` logger section and add the gate table
at info level inside the technique's `TECHNIQUE: TitanHazeNav`
section.
[Revised during Phase C; pending operator ratification. The original
text said "fitted circle center + arc sector, styled by
accept/spurious". Neither the fitted center nor the technique's
accept-or-spurious verdict exists when this runs:
`_collect_annotations` merges every model's annotations in
`_navigate_pipeline` BEFORE `_run_pass`, and only NavModels produce
annotations at all. Two substitutions, both faithful to the intent.
(a) The fit is shown by POSITION, not by a second drawn circle:
`write_summary_png` combines annotations at `result.offset_px`, so the
drawn envelope and the center cross land on the fitted center on a
committed frame and stay at the prediction when no offset was
committed — which is how every other model's overlay already reports
its technique's answer. (b) The style encodes the accept/reject
quantity the model DOES know, the feature's reliability against the
per-type gate threshold: solid curves at or above it, dotted plus a
`TITAN (low reliability)` label below it. The label says reliability
rather than "gated" because manual navigation renders the same overlay
with the gate deliberately skipped (`apply_gate=False`); reliability
is also what decides whether the fit is attempted at all in an
autonomous run, so a dotted overlay and a spurious result are never
confusable. Showing the technique's own verdict would need a
post-technique annotation pass, a cross-cutting orchestrator change
outside this phase's scope. Residual, accepted: on a frame where other
techniques also contribute, the overlay lands on the ENSEMBLE offset
rather than on `TitanHazeNav`'s own — exactly as every other model's
overlay behaves, and the per-technique offsets remain in the metadata
JSON.]

Manual navigation (Section 3 table): `compose_dialog_overlay` and
`NavTechniqueManual.is_feasible` both enumerate geometry types by
hand and know nothing of `TitanHazeGeometry` — as shipped today a
Titan-only frame cannot be manually navigated at all (`feasible=False,
'no_renderable_features_for_manual_nav'`). Add the
`TitanHazeGeometry` branch to both (envelope-circle outline, the
`BodyBlobGeometry` pattern), with a unit test that a Titan-only
feature set is manual-nav feasible and composes a non-empty overlay.
Manual nav is the curation fallback when the autonomous technique
fails; Phase E depends on it working for Titan.

Acceptance: preview PNG for a navigated Titan frame shows the
overlay; a capsys-based log test asserts the gate-table lines; a
Titan-only frame is manual-nav feasible with a rendered drag overlay.

### Phase D — simulated Titan and planted truth

- Scene schema: extend the existing body `atmosphere` block in the
  forward model (`src/spindoctor/sim/forward/atmosphere.py`; a body
  carrying an `atmosphere` block gains a haze layer — this is the
  established convention, do not add a new top-level element) with
  optional truth-side keys: `interior_ramp_amplitude`,
  `ns_asymmetry_amplitude` (affine hemispheric scaling),
  `ns_falloff_ratio` (NON-affine: different limb falloff scales per
  hemisphere — probes past the Pearson affine invariance),
  `axis_tilt_deg` (true symmetry axis rotated from the SPICE sun
  direction), `sector_sharpness_gradient` (limb falloff scale
  varying with ray angle across the sector — drives the
  edge-localization bias risk), `cloud_blobs` (list of `{center_vu,
  sigma_px, amplitude}`), plus whatever falloff-shape key the block
  already uses. The symmetry-breaking axes are the point: the
  navigator's core assumption must be attacked by the sweep, not
  built into every scene. Rendering forms (normative; each key gets
  a render-level unit test asserting its effect is measurably
  present in the output image): `ns_falloff_ratio` multiplies the
  haze falloff scale for the southern-latitude half of the disc (a
  genuinely non-affine hemispheric difference); `axis_tilt_deg`
  rotates the renderer's illumination axis relative to the geometric
  sun direction the navigator will compute; `sector_sharpness_gradient
  g` scales the local falloff length by
  `(1 + g * |phi - theta| / sector_half_angle)` around the limb;
  `interior_ramp_amplitude` adds a linear brightness ramp along the
  axis inside the disc; `cloud_blobs` add Gaussians. Be clear about
  the cost: this is new rendering math in `atmosphere.py` (whose
  current model is a single haze profile), not key plumbing — budget
  the phase accordingly, because the Phase D acceptance bounds are
  only meaningful if these effects genuinely render. Schema documentation and defaults follow the sim
  package's existing key-inventory conventions.
- Information boundary: truth-side keys must be filtered exactly like
  the existing haze truth keys. The enforcement lives in the boundary
  whitelist exercised by
  `tests/spindoctor/sim/test_information_boundary.py` and
  `test_boundary_static_guard.py`; extend the whitelist and both
  tests — and note the trap: the new keys live inside
  `bodies.atmosphere`, which the completeness check treats as ONE
  atomic truth block, so omitting them from
  `_TRUTH_SAMPLES['bodies.atmosphere']` fails nothing and leaves
  them silently un-exercised for leakage. Add each new key to that
  sample explicitly. Idealized keys visible to the navigator:
  predicted center, radii, sun angle only.
- Routing protection for existing scenes (Section 3 table): the sim
  side has no Titan exclusion today — `NavModelBodySimulated` builds
  for every body, and three existing scenes use a body literally
  named `TITAN` as a body-navigation fidelity fixture. Before
  `NavModelTitanSimulated` lands: rename the body in
  `titan_haze_limb`, `titan_crescent_horns(_noiseless)`, and
  `haze_limb_base` (e.g. to `HAZEMOON`), regenerate their baselines
  via `python -m tests.integration.update_sim_baselines` with diff
  review (model-id renames only; behavior identical), and keep the
  `atmosphere_haze` sweep pointing at the renamed haze-BLIND
  fixture. Then add the `TITAN` exclusion to
  `NavModelBodySimulated` and selection tests in
  `test_sim_model_selection.py` covering both directions (`TITAN` ->
  `NavModelTitanSimulated` only; any other body unaffected).
- Performance gating: the new atmosphere rendering math must be
  strictly gated behind key presence — `test_sim_perf.py` carries
  cold-render budgets (2 s at 512, 8 s at 1024) for the EXISTING
  haze path, and per-pixel overhead leaking into scenes without the
  new keys silently regresses them.
- Bookkeeping for the new `titan_haze` base scene: a matching
  baseline JSON (`test_every_scene_has_a_baseline` /
  `test_no_orphan_baselines` both fire otherwise), a committed
  render-review artifact (`render_diffs/current/titan_haze.png` plus
  the regenerated `sheet_atmosphere.png`, from
  `python -m tests.integration.render_contact_sheet`;
  `test_render_diffs` byte-compares every `current/` PNG against a
  fresh render, so a new scene without one is a red), and a Titan
  entry in the hard-coded `_TECHNIQUES` list of
  `tests/integration/technique_snr_characterization.py` (Section 3
  table).
- `NavModelTitanSimulated` in `nav_model/`, mirroring
  `NavModelBodySimulated`'s operator-parameter pattern; it replaces
  the `is_simulated -> []` branch and owns the
  unconfigured case (a simulated Titan scene without the required
  operator parameters builds no model and resolves through the
  standard generic reasons rather than crashing). Sim inventory
  contract: `src/spindoctor/sim/forward/body.py` builds inventories
  WITHOUT the `center_uv` key the real-frame geometry path now
  requires (Phase B center revision); a missing key degrades to a
  hard-zero feature, not a raise, so a sim Titan would silently
  gate out. If the simulated path reuses `geometry_from_obs`, add
  `center_uv` to the sim inventory (it mirrors the oops contract);
  if it reads operator parameters directly, no change is needed —
  decide explicitly, do not rediscover this as a silent hard-zero.
  [Noted during Phase B review. DECIDED in Phase D: the simulated
  model reads operator parameters directly and does NOT call
  `geometry_from_obs`, so the sim inventory is unchanged and gains no
  `center_uv`. It could not have reused that function in any case —
  every branch of it needs `oops` backplanes a simulated observation
  does not carry. What the simulated model does reuse is everything
  downstream of the geometry dataclass, by subclassing
  `NavModelTitan`. A separate coordinate-convention finding came out
  of the same work and is recorded in
  `BODY_CENTER_INDEX_OFFSET_PX`: the sim BODY renderer treats a stated
  body center as a corner coordinate (index center = `center - 0.5`)
  while the sim STAR renderer uses pixel centers, so the simulated
  haze model applies the half-pixel shift and the simulated star model
  does not. Measured directly: without the shift every sim frame
  carried a flat 0.500 px cross-track error, half the clean-scene
  budget.]
- Standing per-technique sweeps: enroll `TitanHazeNav` in the
  EXISTING sweep framework alongside every other technique — add a
  `titan_haze` base scene to the sim scene catalog and dense
  sub-pixel + wide-range offset sweep specs (pinning `TitanHazeNav`)
  as their own YAML files under `tests/integration/sim_sweeps/`,
  runnable via `python -m tests.integration.sim_sweep_runner` like the
  rest. (`sim_sweep.py` is the harness — schema, loader, runner — and
  hard-codes no sweeps; corrected during Phase D, which is also when
  the Section 3 row was fixed.) These feed the simulator report's
  per-technique table and response curves (Phase F), which additionally
  means enrolling the technique in the hand-enumerated lists of
  `sim_sweep_plots.py` and `technique_snr_characterization.py`; the
  `util/` campaign below is the separate randomized multi-axis
  harness, not a replacement for them.
- Planted-truth harness: a sweep script under `util/` (mirroring the
  existing calibration-campaign layout) running N >= 200 randomized
  scenes across offset, angle, phase, size, noise, cloud injection,
  the symmetry-breaking axes above (axis tilt, non-affine
  hemispheric falloff, sector sharpness gradient), star fields, and
  cosmic-ray / hot-pixel artifacts (reuse the sim's existing star
  and artifact machinery — faint point sources are unmasked by
  design and must be swept, not assumed harmless); emits
  recovery-error percentiles per axis and the z-score distributions
  used to set `cross_sigma_scale` / `along_sigma_scale`.

Acceptance: the standing dense/wide offset sweeps run for
`TitanHazeNav` end to end and record a measured navigable ceiling,
like every other technique's entries; the planted-truth sweep
reports cross-track P95 <= 1.0 px and
along-track P95 <= 3.0 px on clean scenes; sigma scales set so
planted-truth z-scores have standard deviation in [0.8, 1.25] per
axis. The confidence-conditioned no-confident-wrong bound (P99 of
error among results with confidence >= 0.5 stays <= 2x the P95 bound)
is evaluated here provisionally with placeholder anchors and becomes
binding only at the Section 7 item 4 re-run, after Phase E sets the
anchors. If the clean-scene bounds cannot be met, stop and report to
the operator rather than loosening silently.

Phase D outcome, as measured (700 scenes, seven families of 100,
campaign seed 20260725; `util/titan_truth`). The sweeps run end to end
with every step committed: sub-pixel recovery to 0.29 px total error and
a navigable ceiling at the 45 px top of the wide grid, which is the
extfov search margin rather than a limit of the method. Clean-scene
bounds hold with large margin — cross-track P95 **0.169 px** against
1.0, along-track P95 **0.820 px** against 3.0. Overall commit rate is
66% (460 of 700). Five findings came out of the same run and are
carried forward rather than smoothed over:

1. **The sigma floors, not the sigma scales, set the reported
   uncertainty, and NEITHER axis reaches the z-score band.**
   **[Pending operator ratification, both axes.]**
   `sigma_floor_cross_px` clamps 94% of rows and `sigma_floor_along_px`
   99%, so the multipliers this phase owns are nearly inert.
   `cross_sigma_scale` goes to 0.10 — the free-row unit-normal solve
   measured 0.101 on this draw (0.088 and 0.098 on earlier ones) — and
   `along_sigma_scale` stays at **1.0**. Measured all-row z standard
   deviations are 0.580 cross and 0.512 along, both below the
   [0.8, 1.25] band, and no value of either multiplier reaches it:
   they saturate at 0.594 and 0.650 once every row sits on its floor.

   The along axis was briefly set to 0.4 on an earlier draw because
   that raised its all-row statistic into the band; that change was
   REVERTED, and the reason is worth recording because it is a trap
   the same evidence will set again. The few rows whose fit sigma
   exceeds the floor are not over-conservative outliers — they are the
   rows where the arc fit knew it had done badly, and on this draw
   they are exactly the campaign's four worst along-track errors
   (8.05, 6.34, 4.72, 2.92 px). Their own z-scores run to an rms of
   3.774: the estimator reports sigma far too NARROW there, not too
   wide. Shrinking the multiplier would have narrowed the reported
   uncertainty further on precisely the frames that are wrong, buying
   a better-looking aggregate by making the honest minority dishonest.

   So both axes carry the same disposition: the floor is the binding
   term and only re-tuning it can settle the statistic, which is Phase
   E's with real-frame evidence. The targets the campaign implies are
   `sigma_floor_along_px` near 0.63 px and `sigma_floor_cross_px` near
   0.08 px (each the value that, in quadrature with the 0.20 px
   model-error floor, equals the measured per-axis rms of 0.663 and
   0.214 px). Both saturation figures are population-dependent —
   bootstrapped over the committed rows the cross value runs 0.640 at
   n = 454 (10-90 pct 0.544-0.755) and 0.580 at n = 100 (0.412-0.823)
   — so the honest statement is not "the band is unreachable" but
   **the floors pin the achievable z-std near the band's lower edge,
   and only lowering them can settle it**. The z-versus-scale curve
   the analyzer prints, anchored to the reported all-row z, is the
   evidence.
2. **The competing-peak gate is what bounds the working phase range,
   not the estimator.** A haze disc near full illumination is close to
   rotationally symmetric, so the mirror-correlation scan grows side
   lobes about 15 px either side of the true axis. Measured on a
   152 px envelope, the strongest rival runs 0.91 of the peak at phase
   20, 0.89 at phase 40, 0.74 at phase 60, and vanishes by phase 90 —
   straddling the 0.90 `max_second_peak_ratio` threshold. Where it
   fires the recovered cross-track offset is nonetheless exact
   (measured 8.001 px against a planted 8.0), so the gate is refusing
   frames it could have navigated. Commit rate by phase bin runs
   65 / 61 / 64 / 68 / 74 / 62% over 10-30 / 30-50 / 50-70 / 70-90 /
   90-110 / 110-140 deg — flatter than the rival-lobe numbers alone
   would suggest, because the arc-side gates take over where this one
   relents (`arc_radius` alone accounts for 85 of the 240 refusals,
   against 31 for `second_peak`). Phase E owns the threshold; the base
   scene sits at phase 60 so the standing sweeps characterize the
   estimator rather than the gate.
3. **The along-track tail is a small-body-at-high-phase property of
   the estimator itself, present in CLEAN scenes, and it is not the
   tilted symmetry axis.** Only 5 of 460 committed rows carry an
   along-track error above 2 px, and every one of them has an
   apparent solid radius in the bottom sixth of the drawn range
   (30.6-38.1 px against a 28-78 px draw) at a phase of 84-132 deg.
   Split that way the population separates cleanly: bodies with
   `r_solid >= 40 px` give an along-track P95 of 0.717 px and a
   maximum of 1.655 px across ALL families and phases, while bodies
   below 40 px at phase above 60 deg give P95 **3.009 px** and a
   maximum of **8.050 px** — and the same small bodies below 60 deg
   phase give P95 0.369 px. A small disc at high phase leaves the
   sunward arc its least support, which is the mechanism. Two things
   it is NOT: contamination (the worst row, 8.050 px, is a `clean`
   scene) and the tilted axis (over the asymmetry family the
   along-track error is essentially uncorrelated with `axis_tilt_deg`,
   Spearman +0.149, and +0.043 on the cross axis).

   The asymmetry family's own cost is cross-track (P95 0.345 px
   against the clean family's 0.169) but DIFFUSE: at these draw
   strengths no single structure key dominates it — Spearman against
   |cross| runs +0.232 for `interior_ramp_amplitude`, +0.183 for
   `ns_asymmetry_amplitude`, -0.266 for |`ns_falloff_ratio` - 1|, and
   -0.005 for `sector_sharpness_gradient`, all weak and not all of one
   sign. An earlier draw that pinned the sun to two directions showed
   `ns_falloff_ratio` as a clean monotone driver (cross P95 0.204 ->
   0.779 by departure from unity); that ordering does not survive
   uniform sun coverage, so it is recorded as draw-dependent rather
   than as a mechanism. Phase E should aim the arc-side knobs
   (`sector_half_angle_deg`, `min_gradient_snr`) and the apparent-size
   floor at the small-body high-phase regime, and should not expect
   `axis_tilt_deg` to be the cross-track stressor.
4. **Unmodelled point sources are the single largest degrader, and the
   stress family overstates the operational case by design.** Matched
   -scene ablation over the same 100 clean geometries, three
   conditions: no artifacts 77% commit / cross P95 0.169 / along P95
   0.820; the campaign's STRESS artifact draw 37% commit / cross P95
   0.806 / along P95 1.486; the instrument's own realism-matched
   population (`instrument_defaults` alone, nothing overridden) 79%
   commit / cross P95 0.235 / along P95 1.444. So the operational
   prediction is no commit-rate cost at all (79% against 77%, inside
   the draw's own noise) and a cross-track P95 comfortably inside the
   clean bound, while the stress condition halves the commit rate and
   quintuples the cross-track P95 — a bound on the regime, not a
   forecast. The stress
   ranges are deliberate and their provenance is stated in
   `_artifact_blocks`: hot-pixel incidence is drawn over 2e-4 to 2e-3
   where the realism match measures a 2.75e-4 transient spike fraction
   on the CALIB NAC cohort, and the cosmic-ray rate is drawn strictly
   positive although the realism recalibration RETAINED zero for it
   (the tuned hot-pixel fraction already carries that population, so a
   nonzero rate double-counts on purpose). The `artifacts_nominal`
   family is the realism-matched condition and is the one to quote;
   Phase E consumes both.
5. **The no-confident-wrong criterion PASSES as a percentile and FAILS
   as an existence statement, and the existence reading is the one
   Section 8 makes.** Section 8 criterion 2 says results wrong by more
   than 2x the stated bound with confidence >= 0.5 "do not occur";
   Phase D's acceptance line evaluates a P99, which is a different and
   weaker question. Both readings are recorded here so the difference
   cannot be lost. As a percentile the check passes on both axes —
   cross P99 0.799 px against a 2.0 px limit, along P99 2.010 px
   against 6.0 px. As an existence statement it does not: three
   committed rows exceed 2x their axis bound while carrying confidence
   >= 0.5, and they are the concrete population Phase E's anchors have
   to separate:

   | row | axis error | reported sigma | z | confidence | family | phase |
   |---|---|---|---|---|---|---|
   | `clean_0052` | along +8.050 px | 1.372 | +5.87 | 0.783 | clean | 100.8 |
   | `clouds_0006` | along +6.340 px | 1.919 | +3.30 | 0.753 | clouds | 99.5 |
   | `artifacts_0013` | cross -2.602 px | 0.361 | -7.22 | 0.733 | artifacts | 83.7 |

   Every one of them is small-body-at-high-phase (finding 3), and
   every one of them reports a sigma its own error dwarfs — so both
   the confidence spec and the reported covariance have a shot at
   catching them. The placeholder spec catches neither: it scores all
   460 committed rows at confidence >= 0.5, so it currently filters
   nothing at all. The criterion becomes binding at the Section 7
   item 4 re-run, after Phase E sets the anchors; these three rows are
   the acceptance test for them.

### Phase E — real frames, tuning, confidence anchors

- Step 1, vendor the cohort: copy
  `/seti/all_repos/rms-csmithing/tests/titan_images.txt` (87 lines;
  legacy annotated Titan test list) into
  `util/titan_cohort/titan_images.csv` with columns
  `image_id, flags, notes` — `flags` a semicolon list drawn from
  `{rings_occluding, moon_occluding, high_phase, near_edge,
  off_edge, known_bad, clean}` assigned by reading each line's
  freeform annotation (unannotated lines -> `clean`); keep the
  original text in `notes`. Provenance comment at the top names the
  source path and date.
- Run `sd_offset` over all retrievable Cassini frames in the list
  (batch conventions per Section 1).
- Evidence tiers (offsets are measured against each frame's own
  SPICE prediction, so commanded pointing differences are already
  removed):
  (a) star-anchored truth — stars share the frame's translation, so
  any cohort frame where a star technique independently locks gives
  an absolute per-frame anchor: per-axis
  `|offset_titan - offset_star| <= 2 * sqrt(sigma_t^2 + sigma_s^2)`.
  Scan the whole cohort for star locks first; every such frame is
  the strongest evidence available and seeds the WS-1 agreement
  channel (Section 9);
  (b) within-sequence consistency — pairs of `clean` frames of the
  same target within 30 minutes: per-axis
  `|offset_1 - offset_2| <= 2 * sqrt(sigma_1^2 + sigma_2^2)`;
  (c) cross-filter consistency — near-simultaneous (<= 10 minutes)
  different-filter pairs, same test; this directly checks the
  method's filter-independence claim;
  (d) operator review of overlays for a stratified sample of ~20
  frames (filters x phase bins), dispatched as a normal
  curation-style review batch.
- Levers handed over from Phase D, each with its measured evidence in
  the Phase D outcome block above. They are listed here, not merely
  cross-referenced, so the Section 7 item 2 dispatch carries them
  inline:
  - `sigma_floor_cross_px` (0.30): the binding term in the reported
    cross-track uncertainty, clamping 94% of planted-truth rows. No
    value of `cross_sigma_scale` brings the all-row z-score standard
    deviation past 0.594, so this floor is the only lever that can
    settle it inside [0.8, 1.25]. Measured cross-track errors have an
    rms of 0.214 px and a clean-scene P95 of 0.169 px; a floor near
    0.08 px is what that rms implies in quadrature with the 0.20 px
    model-error floor.
  - `sigma_floor_along_px` (1.00): the same story on the other axis —
    it clamps 99% of rows, caps the all-row z-score standard deviation
    at 0.650, and the measured along-track rms of 0.663 px implies a
    floor near 0.63 px. It carries a second consequence beyond
    calibration: `hypot(1.00, 0.20)` is 1.02 px against the `high`
    tier's `max_sigma_px: 0.5` in `config_540_orchestrator.yaml`, so
    **`TitanHazeNav` can never reach the high confidence tier** on a
    Titan-only frame however good the fit. Every such frame caps at
    `medium`. Decide deliberately whether that is the intended
    statement about a one-feature estimator or an artifact of a
    placeholder floor.
    Lower it on real-frame evidence, not on the planted-truth rms
    alone: the campaign's own worst rows are the ones whose fit sigma
    already EXCEEDS this floor (finding 1), so a floor cut narrows the
    reported uncertainty on the majority while leaving the honest
    minority untouched, and the confident-wrong rows in finding 5
    would get no easier to catch.
  - Apparent-size floor for the arc fit: every along-track error above
    2 px in the campaign came from a body under 40 px solid radius at
    phase above 60 deg (finding 3). Consider whether
    `min_envelope_diameter_px` (40.0, which a 30 px solid radius
    clears at 76 px) is the right gate for that regime, or whether the
    arc fit needs its own size condition.
  - `max_second_peak_ratio` (0.90) at low phase: the gate refuses
    frames whose cross-track answer is exact (measured 8.001 px
    against a planted 8.0) because a near-fully-lit haze disc is
    genuinely near-rotationally-symmetric. Raising it trades that
    conservatism for false-lock exposure; the phase-versus-rival-lobe
    numbers in Phase D finding 2 are the input.
  - Confidence-anchor separation target: the placeholder spec scores
    all 460 committed planted-truth rows at confidence >= 0.5, so the
    no-confident-wrong check currently filters nothing. Finding 5
    names the three concrete rows the anchors must separate
    (`clean_0052`, `clouds_0006`, `artifacts_0013`); all three are
    small-body-at-high-phase and all three report a sigma their own
    error dwarfs, so both the spec and the covariance have a lever.
  - Point-source contamination: the realism-matched condition costs no
    measurable commit rate and leaves cross-track P95 inside the clean
    bound (Phase D finding 4), while the stress condition halves the
    commit rate. Confirm on real frames, and treat the campaign's
    stress family as a regime bound rather than a forecast.
- Tune Section 5 defaults from failures; set confidence-spec anchors
  so the planted-truth confidence-vs-error curve is monotone and the
  Phase-D no-confident-wrong bound holds; document the sweep in a
  campaign record under `util/` with a tracking issue for anything
  deferred. Known lever carried from Phase A review (#396): with
  the shipped defaults, 8/96 (8%) of frames with truth planted
  OUTSIDE the search window (16-26 px at SNR 20, W = 10) lock
  confident-wrong instead of gating; `min_gradient_snr: 4.0 -> 8.0`
  measured 0/96 at no in-window cost (worst in-window error 1.319
  -> 1.373 px). Evaluate that lever here; also keep
  `radial_outer_pad_px` well above ~1.5 px so the boundary-argmin
  drop rule retains its margin (#396).
- Enroll Titan in the calibration tooling (Section 3 table): the
  fitters enumerate techniques BY HAND — add `TitanHazeNav` to
  `util/calibration/fit.py`'s `TECHNIQUES` tuple (with
  `SIGN_BY_FEATURE` entries: `arc_residual_rms_px` negative, the
  quality terms positive) and to `util/calibration/fit_floors.py`'s
  tuple (its `model_error_floor_px` is fitted there, not guessed),
  and add a `titan` family + generator to
  `util/calibration/scene_gen.py` so the campaign renders Titan
  rows at all. Without these three, the placeholders installed at
  Phase B are never recalibrated and the data-driven gate reports
  simply never see Titan.
- Library: nominate 3-6 cohort frames (spanning filter/phase classes)
  through the standard curation flow (operator votes gate admission;
  sidecars per the cohort-curation plan; one PR per review batch).
  Scene-class decision at nomination time (with the operator): add a
  `titan_haze` class to `tests/integration/sidecar.py`
  `DECLARED_SCENE_CLASSES` (the structural-invariants test rejects
  an undeclared subdirectory) and to the COHORT_CURATION_PLAN budget
  table, or file the frames under an existing class; either way,
  update the deferred-technique name in
  `tests/integration/image_library/images/README.txt` (`TitanNav` ->
  `TitanHazeNav`) and extend `tests/integration/sim_realism.py`'s
  model-glob map with `titan:*` / `titan_sim:*` so the realism FOM
  exercises the Titan model on the new frames.

Acceptance: >= 70% of `clean` frames produce accepted (non-spurious)
results; >= 90% of the (a)+(b) consistency pairs pass the 2-sigma
test; every `clean`-frame failure is attributed to a named gate or a
filed issue; every `rings_occluding` / `moon_occluding` / `off_edge` /
`known_bad` frame is type-gated (hard-zero reliability) or fails a
named technique gate rather than producing a
confident-wrong lock.

Phase E outcome, as measured (82 Cassini frames, `util/titan_cohort`;
the full record with every sweep is
`util/titan_cohort/CAMPAIGN_20260726.md`).

| criterion | bound | measured | verdict |
|---|---|---|---|
| clean frames accepted | >= 70% | 36/49 = 73.5% | PASS |
| (a)+(b) pairs within 2-sigma | >= 90% | 10/12 = 83.3% | FAIL [pending operator ratification] |
| clean-frame failures attributed | all | 13/13 | PASS |
| adverse frames: no confident-wrong lock | all | 18/27 gate or fail a named gate, 2 emit no feature at all, 7 commit; under the criterion's own physical screen every witness places all seven inside twice a stated axis bound except `BodyBlobNav` on three, and two frames have no witness | PARTIAL |

The pair bound is marked pending ratification because finding 6 argues it
is not reachable by an honestly calibrated estimator, and changing an
acceptance bound is the operator's call (Section 7 item 3), not this
phase's. The measurement stands either way; what needs a decision is
whether >= 90% on a two-axis 2-sigma conjunction remains the criterion.

1. **Real frames confirm the published bound.** Against an
   independent star lock on the same frame — nine such pairs — the
   cross-track disagreement runs 0.99 px rms (worst 1.84) and the
   along-track 1.50 px rms (worst 3.84), implying about 0.70 and
   1.06 px per frame against the plan's <= 1 px and <= 3 px targets.
   Repeat frames of one target through one filter agree to 0.34 px
   cross-track, 0.33 px along-track, and 4 km of fitted haze radius.
2. **The sunward-sector circle fit's `(d, R)` degeneracy is the
   dominant real-frame error; the haze top is EXPECTED to be
   wavelength-dependent, which is why the degeneracy matters (the
   campaign could not itself confirm the wavelength dependence).** Widening `sector_half_angle_deg` from 60
   to 80 degrees (correlation between `d` and `R` 0.984 -> 0.942,
   a factor 3.5 in the variance the degeneracy adds to `d`) cut the
   star-anchored along-track pair-difference rms from 4.23 to 1.50 px
   and removed a confident-wrong lock on `N1647091889_1` (10.6 px from
   its star anchor at confidence 0.85). It did NOT measurably tighten
   the red-versus-violet fitted-radius differences themselves: over the
   six pairs committing on both sides those span 35.0-131.2 km at
   60 deg and 11.8-97.8 km at 80 deg, with the spread about the mean
   essentially unchanged, so nothing there separates a physical
   haze-top difference from fit noise. It costs simulated commit rate
   (65% -> 59%),
   where a rendered haze limb IS the fitted circle and the degeneracy
   has nothing to correct; the simulator's clean bounds are unmoved.
   This sharpens Section 9 item 1 from "enables small-disc Titan" to
   "is the fix for the along-track error generally".
3. **The `arc_residual` gate must NOT be raised.** It is the largest
   single cause of clean-frame refusal and the real-frame residual
   distribution has an inviting gap (a continuum to 3.03 px, then
   8.99 and 57.9 px), but at 3.5 px two admitted frames lock
   measurably wrong — one by 12.7 px against its own star anchor and
   8.7 px against its cross-filter twin. The frames between 2 and 3 px
   of residual are wrong, not merely noisy.
4. **Both sigma floors are held on real-frame evidence, against the
   simulator's advice.** Phase D implied 0.08 px cross and 0.63 px
   along; real frames measure 0.70 and 1.06 px per frame against the
   reported 0.36 and 1.02 px. Cutting either would report a tighter
   uncertainty than real frames support. The along floor's consequence
   is accepted deliberately: a Titan-only frame caps at the `medium`
   confidence tier. The planted-truth z-score band settles the same
   way and for a reason the aggregate hides: over all 413 committed
   rows the along z standard deviation is 1.56, above the band, but
   over the rows the confidence spec calls confident it is 0.32 and at
   confidence >= 0.7 it is 0.26. The whole excess is the small-disc
   tail the anchors reject, so the reported sigma is conservative on
   the population the ensemble consumes; raising the floor to move the
   all-row statistic would widen the trusted rows to cover the
   rejected ones.
5. **Confidence anchors separate the failure population.** Fitted over
   the 413 committed rows of the Phase D campaign re-run at the final
   configuration: 0 of 7 rows wrong by more than twice an axis bound
   keep confidence >= 0.5 (they score 0.106-0.384), and among
   confident rows the along-track maximum is 3.80 px against the 6 px
   limit — the no-confident-wrong criterion holds as an existence
   statement, not only as a percentile. Real-frame confidence lands in
   0.53-0.88 (median 0.76). The anchors are produced and verified by
   `util/titan_truth/fit_confidence.py`, which exists because the
   generic fitter's label has a 0.975 base rate on the Titan
   calibration family and so discriminates nothing; the residual
   coefficient is bounded at -2.5 in that fit's own configuration
   against the -15.12 the unconstrained solve returns, which would
   have been a near-hard gate at 0.5 px of residual — fine in a
   simulator whose limb is a perfect circle, ruinous on real frames
   whose median residual is 1.1 px.
6. **The pair bound sits at the ceiling of what an honest calibration
   can reach.** A per-axis 2-sigma test on two axes passes at 91.1%
   with an exactly-right covariance, so >= 90% leaves no margin; 10 of
   12 has a binomial probability of about 0.29 under correct
   calibration. Neither failure is a lock error (the offsets agree
   with the star anchor to 1.4 and 1.9 px); both are the reported
   sigma being tight, and widening a floor to convert them would
   contradict finding 4.
7. **Two gate mechanisms are worth follow-ups.** Radial sampling
   reaches `r_env + pad + W`, and `W` is 140 px on a Cassini NAC, so a
   large Titan loses whole rays to out-of-frame outer samples: of the
   81 rays an 80 deg half-sector at 2 deg spacing offers,
   `N1481452791_1` keeps 16 and `N1686939958_1` keeps 5. And the arc
   residual scales with apparent size (+0.315 correlation over the 49
   committed frames, whose median envelope is 518.4 px against the
   569-847 px of the frames the gate refuses), so a fixed 2.0 px cap is
   a size-dependent gate; it cannot be raised (finding 3) but a
   size-relative form is worth measuring.

### Phase F — docs, reconciliation, deferred issues

- User guide: Titan section — capability, bounds from Phase D/E, the
  gate behavior on marginal frames, config knobs. Dev guide:
  technique page (method
  distilled from Section 2, current-state wording), diagnostics and
  reliability field docs, API-reference stubs. Simulator report
  (`docs/simulator_report/simulator_report.rst`): the `titan_haze`
  base scene, a `TitanHazeNav` row in the per-technique offset-sweep
  table, and the regenerated dense/wide response curves — Titan must
  not be the one technique absent from the report's sweep coverage. Docs describe shipped
  behavior only — no references to prior interim behavior or to this
  plan's phase labels; provenance is the published paper citation.
- Stale-statement checklist (normative — each item is verified
  changed, or consciously confirmed unaffected, in this phase's
  review): `/seti/newnav/CLAUDE.md` ("a registered placeholder that
  emits no features"; the technique enumeration gains
  `TitanHazeNav`); `README.md` technique-families bullet;
  `docs/user_guide/user_guide_navigation.rst` (the
  Titan-always-fails passage, the model/technique enumerations and
  family count, and a `titan.navigation.*` key table matching the
  star/body/ring list-tables); every documentation and plan
  occurrence of `titan_unsupported` is REMOVED — the status reason
  is deleted, not renamed (Section 2.5); grep the docs and plans
  trees, do not rely on this checklist's named spots alone; `docs/introduction_configuration.rst`
  `--nav-techniques` name list;
  `docs/user_guide/user_guide_statistics.rst` feature-family list
  gains `titan`. Dev guide:
  `dev_guide_navigation_models_titan.rst` and
  `dev_guide_navigation_models_titan_simulated.rst` are FULL
  rewrites (both describe the decline-only world; the latter says no
  source file exists); `dev_guide_techniques_body_titan.rst`
  ("Titan (unsupported)", and it names the wrong planned method) is
  replaced by the new technique page with the
  `dev_guide_techniques.rst` toctree fixed (the family is Titan, not
  Body); `dev_guide_navigation_models.rst`,
  `dev_guide_navigation_models_bodies.rst`,
  `dev_guide_navigation_models_body.rst` no-result clauses;
  `dev_guide_class_hierarchy.rst` (mermaid technique diagram +
  inheritance edge, the Titan paragraphs, and the geometry/flags
  variant enumerations gain `TitanHazeGeometry` /
  `TitanHazeFlags`); `dev_guide_annotations.rst` empty-collection
  claim; `dev_guide_familiarization.rst` tour entries; the
  orchestrator page's "emits no navigable features by design"
  passages (deleted along with the orchestrator special case they
  describe);
  `dev_guide_config_and_static_data.rst` (config_060 carries the
  full consumed schema, not a single reserved value);
  `dev_guide_simulator.rst` capability envelope + the
  sim-atmosphere section (new keys) + a caveat that its
  haze-blind-navigator mismatch prose describes body navigation, not
  `TitanHazeNav`. API reference: `api_nav_technique.rst` automodule
  for `nav_technique_titan_haze`; `api_nav_model.rst` automodule for
  `nav_model_titan_simulated`.
- Plan reconciliation beyond the named spots: in the VALIDATION
  plan, the "scope honestly (e.g. Titan)" example, the WS-7 body
  text — which describes a per-filter-haze-profile DT method, NOT
  the French method; rewrite it to the shipped method rather than
  merely marking it done — the capability-matrix row 6a, and the two
  "decision gates first" passages; the PROGRAM_PLAN /
  ENGINEERING_PLAN / OPERATOR_PLAYBOOK interim-wording sentences.
  One deliberate NON-change: issue #344 (sim haze brightness is a
  module constant) stays OPEN — Phase D adds shape/symmetry-breaking
  rendering, not photometric brightness variation; confirm it, do
  not close it.
- Reconcile the five plan files as if the PR merged: PROGRAM_PLAN
  (#60 rows in Track B/D, decision-gates list), ENGINEERING_PLAN
  (#60 item), VALIDATION_AND_CALIBRATION_PLAN (WS-7 status + the
  stated bound), OPERATOR_PLAYBOOK (decision 0.1 list, pinned-red set
  if it changes), COHORT_CURATION_PLAN (Titan frame classes if the
  library nominations land).
- File the deferred issues (Section 9) with A-type, B-location,
  Priority, Effort labels and assignee `rfrenchseti`; reference them
  from the PR body; one `Closes #60.` sentence; REST API
  (`gh api -X PATCH`) for any PR/issue edits.

Acceptance: `sphinx-build -W` clean; `pymarkdown scan` targets clean;
plans contain no stale Titan-decision language.

Phase F outcome, as executed. Every checklist item was changed or
consciously confirmed unaffected. `sphinx-build -W` and `sphinx-build -n`
both exit clean; nitpicky warnings fall 290 -> 277 with ZERO Titan-related
warnings remaining and none added elsewhere. `pymarkdown scan docs/ .cursor/
README.md CONTRIBUTING.md` is clean (0, unchanged). `ruff check`, `ruff
format --check`, `mypy src tests`, and the 4434-test unit suite are green.
Six things went beyond the itemized checklist and are recorded here rather
than left as surprises:

1. **A stale docstring was corrected in code.** `TitanHazeGeometry`'s
   `predicted_center_vu` still described "the midpoint of the predicted
   bounding box", which the Phase B center revision replaced with the
   projected field-of-view center. The API reference publishes that
   docstring, so it is a documentation defect and was fixed in
   `src/spindoctor/feature/geometry.py`.
2. **The API reference gained more than the two named entries.**
   `nav_technique_titan_haze` and `nav_model_titan_simulated` were the named
   ones; `nav_model.titan_geometry`, the `titan_fitting` package and its four
   submodules, and `sim.forward.haze_structure` were also required, because
   without them nine nitpicky cross-references from shipped docstrings had no
   resolvable target.
3. **Two stale spots outside the checklist.**
   `dev_guide_orchestrator_feature_summary.rst` never listed the
   `reliability_reasons` field Phase B added to `NavFeatureSummary`, and
   `dev_guide_rotation.rst`'s rank-deficient-rotation list did not include
   `TitanHazeNav`. Both fixed.
4. **`#406` was filed outside the Section 9 list.** The playbook's
   pinned-red table gives every red an owning issue; the two pre-existing
   reds Phase E documented had none, so adding the rows without an issue
   would have created ownerless pins.
5. **`#396` is discharged, not deferred.** Phase E evaluated its lever and
   shipped it (`min_gradient_snr` 8.0, `radial_outer_pad_px` 6.0), so the
   merging PR closes it.
6. **`/seti/newnav/CLAUDE.md` is outside the repository** (it lives one
   directory up and is not under version control here), so its two required
   edits are made but do not appear in the PR diff.

## 7. Execution protocol (controller contract)

1. Branch `rf_titan_nav` off current `main`; one commit series per
   phase (Conventional Commits; `feat:` / `test:` / `docs:` as
   appropriate).
2. Per phase: dispatch an implementer subagent — Opus model
   override, per the Section 0 model requirement — whose prompt
   embeds the phase's section of this plan verbatim plus Sections
   1-5 (the subagent must not need to locate this file); then
   dispatch an independent fresh-context adversarial reviewer (also
   Opus, never Fable) with the diff, the
   same plan sections, and instructions to (a) verify each normative
   statement of Section 2 against the code line by line, (b) run the
   phase's tests plus `ruff check src tests`, `ruff format --check
   src tests`, `mypy src tests`, (c) hunt for convention violations
   (Section 1) and unstated deviations from this plan. Fix rounds
   until the review is clean; the controller, not the implementer,
   judges cleanliness.
3. Deviations from this plan discovered mid-phase (an API that does
   not exist, a wrong assumption about the orchestrator) are recorded
   in the phase commit message and reconciled into this plan file in
   the same commit — the plan must stay true as built. Scope changes
   (dropping a phase, changing an acceptance bound) go to the
   operator instead.
4. Final sweep before PR: `./scripts/run-all-checks.sh -i`;
   `util/calibration/library_crosscheck.py` with every delta vs the
   documented pinned red set attributed in the PR body (expected
   deltas: Titan-bearing frames only); the Phase-D planted-truth
   sweep re-run on the final revision with the Phase-E confidence
   anchors — this re-run is where the no-confident-wrong bound
   (Phase D acceptance) becomes binding.
5. One PR to `main`: summary, phase map, sweep evidence links,
   `Closes #60.`, deferred-issue references, plan/guide
   reconciliation included. Squash-merge per repo convention.

## 8. Acceptance criteria (whole plan, maps to WS-7)

1. Real Cassini Titan frames navigate within a stated bound: the
   bound published in the user guide is the Phase-D planted-truth P95
   pair confirmed by the Phase-E consistency evidence (target: <= 1
   px cross-track, <= 3 px along-track).
2. No confident-wrong behavior: at the Section 7 item 4 re-run, and
   on every cohort frame flagged `known_bad` / occluded / off-edge,
   results wrong by more than 2x the stated bound with confidence >=
   0.5 do not occur.
3. Every Titan frame is attributable under the STANDARD status
   vocabulary: a committed result; a technique-failure status
   (`all_techniques_spurious`, or a downstream consensus status)
   with the failed gate named in diagnostics; or
   `all_features_gated` with a `TITAN_LIMB` gate record whose
   reliability breakdown names the cause — all verifiable in the
   emitted JSON. No Titan-specific status reason exists anywhere in
   the codebase, and no Titan frame produces an unattributable
   failure.
4. The full check suite (`run-all-checks.sh -i`) is green; the
   library red set equals the documented pinned set plus only
   operator-accepted Titan deltas.
5. Docs and the five plan files reflect the shipped state; #60
   closed; deferred work exists as labeled issues, not prose.

## 9. Deferred follow-ups (filed as issues)

1. (#397) Self-calibrated haze-radius table per (instrument, filter, phase
   bin) accumulated from production `fitted_haze_radius_km`
   diagnostics; enables small-disc Titan via known-radius circle fit.
   Phase E raised the value of this: the sunward-sector fit determines
   the limb position `d + R` far better than it separates `d` from
   `R`, so pinning `R` from a table removes the dominant real-frame
   error on the along-track axis and not only on small discs. Note
   what Phase E could NOT show: the fitted radii of matched
   cross-filter pairs span 35.0-131.2 km apart at a 60 degree
   half-sector and 11.8-97.8 km at 80 degrees, which is too scattered
   to separate a physical wavelength-dependent haze top from fit
   noise. Building the table is also how that question gets answered.
2. (#398) Methane surface-window (CB3) cartographic correlation as a
   refinement stage on Phase-1 solutions.
3. (#399) Voyager ISS Titan validation cohort (the method is
   instrument-independent; evidence depth is Cassini-only after
   Phase E).
4. (#400) Ensemble handling of strongly anisotropic covariances: verify the
   merge and tier logic treat a 1 px x 3 px result correctly; extend
   if Titan results expose gaps.
5. Titan confidence re-anchoring rides #230 (real-evidence
   recalibration) like every other technique; no separate issue, and
   #400 carries the covariance half of the same question.
6. (#401) High-phase (> 150 degrees) Titan: measure where the sector fit
   actually fails; consider a full-ring fit at extreme phase.
7. (#402) Ring-occlusion translucency: the Section 2.1 mask treats the main
   rings as opaque; frames where Titan is visible through the C ring
   or gaps currently gate out at hard-zero reliability.
8. (#225) Titan-vs-star agreement channel: register the TitanHazeNav /
   star-technique pair in the WS-1 agreement study (#225), so Titan
   accuracy claims graduate from this plan's acceptance evidence to
   the program's published agreement statistics. Concrete shape:
   `util/agreement/analyze.py` `_TECH_TO_INSTANCE` (currently maps
   four body/ring techniques and no star technique), the
   rotating-basis and pivotal-pair wiring in the same file, a
   Titan+star family in `util/agreement/scene_gen.py`, and a run key
   in `util/agreement/collect.py`. Register the Titan+body pair in
   the same pass: a co-visible moon's shape-feature lock anchors the
   haze fit exactly the way a star lock does, and the Phase E cohort
   measured it as the second-strongest corroboration (BodyLimbNav
   11/12 at 2-sigma; recorded on #225).
9. (#403) Ray reach versus the search window: radial profiles are sampled
   out to `r_env + radial_outer_pad_px + W`, and `W` is 140 px on a
   Cassini NAC, so a large Titan loses whole rays to out-of-frame
   outer samples even when its limb sits comfortably inside the
   detector (Phase E measured, of the 81 rays an 80 degree
   half-sector at 2 degree spacing offers, 16 surviving on
   `N1481452791_1` and 5 on `N1686939958_1`). The ray-drop rule is
   right; sizing the reach by the full search window rather than by
   where the limb can actually be is what costs those frames.
10. (#404) Size-relative arc-residual gate: on real frames the inlier
    residual RMS scales with apparent size (+0.315 correlation over the
    49 committed frames; the refused frames run 569-847 px of envelope
    against an accepted-population median of 518.4 px), so the fixed
    `max_residual_rms_px` is a size-dependent gate that refuses large
    well-resolved Titans.
    Phase E showed it must NOT simply be raised — the frames it
    refuses between 2 and 3 px are measurably wrong — so the question
    is whether a size-relative form separates the two populations
    the flat cap conflates.
11. (#405) Titan library growth through the standard curation pipeline: a
    `titan_haze` scene class in the cohort-curation taxonomy (the
    `DECLARED_SCENE_CLASSES` enum, the COHORT_CURATION_PLAN budget
    table and structural-invariants minima, candidate-discovery scan
    builders in `util/cohort_curation/scan_stage_a.py`, and the
    primary-technique rubric in `build_sidecars.py`) — Phase E
    deliberately bypasses this pipeline via the vendored legacy
    cohort, so growing Titan coverage beyond those frames is
    follow-up work.

Filed alongside these, outside the Section 9 list:

- **#407** collects every operator decision this plan defers -- the five
  mid-implementation specification changes marked `[pending operator
  ratification]` in Sections 2 and 6, the three acceptance bounds the
  evidence argues with (the Phase A noisy-scene along-track bound, the
  Phase D z-score band, the Phase E consistency-pair bound), and the three
  staged curation artifacts (the twenty-frame overlay review batch, the six
  library nominations, the `titan_haze` scene-class recommendation).
- **#406** owns the two pre-existing library reds Phase E documented
  (`N1487595731_1_CALIB`, `N1633925572_1_CALIB`); both fail identically on
  `main` and neither contains Titan, so they are pins the playbook's table
  had not caught up with rather than deltas from this work.
- **#396** (the `min_gradient_snr` lever carried from the Phase A review)
  is DISCHARGED by Phase E: `min_gradient_snr` ships at 8.0 and
  `radial_outer_pad_px` at 6.0, measured at negligible real-frame cost, so
  the merging PR closes it.
- **#344** (sim haze brightness is a module constant) stays OPEN
  deliberately: Phase D added shape and symmetry-breaking rendering, not
  photometric brightness variation.

## 10. Risks and prescribed responses

- **Symmetry bias from clouds / seasonal asymmetry** — the annulus
  restriction plus Tukey weighting are the designed mitigations;
  Phase A tests 3-4 and Phase D injections quantify them. If the
  planted-truth bound fails, first raise `annulus_inner_fraction`
  (more limb-dominated), then escalate to the operator.
- **Axis degeneracy near zero phase** — handled explicitly by the
  `axis_min_offset_px` branch in Section 2.1 item 3 (any axis is
  valid on a rotationally symmetric disc; refinement is skipped).
  Phase D includes a low-phase cell to confirm the branch behaves.
- **Oops geometry cost** — Section 2.1 needs only the inventory bbox,
  center resolution, the incidence backplane on the envelope-bbox
  subgrid, and the occlusion backplanes on the mask-bbox subgrid
  (envelope bbox + pad + 2W); no full-frame backplanes. The shipped
  full-frame boolean array is memory, not computation.
  If profiling shows the ring-occlusion backplane dominating, skip it
  when no ring plane crosses the bbox.
- **`map_coordinates` cost on 4k images** — the rotated grid is
  bounded by `r_env_px + pad + W`, not the frame size; assert the
  grid stays under ~2000 x 4000 samples for the largest supported
  Titan.
- **Sector-asymmetric edge localization (the Titan analog of the
  #150 limb systematic)** — limb sharpness varies with ray angle
  (softening toward the terminator sides of the sector), so the
  detected max-gradient ridge shifts radially as a function of
  `phi`. The free radius absorbs only the uniform part; the
  `phi`-dependent part biases `d` along the axis. The
  `sector_sharpness_gradient` sweep axis measures it; if it exceeds
  the along-track budget, narrow `sector_half_angle_deg` (trading
  random error for bias) before considering per-ray sharpness
  modeling.
- **The plan is wrong somewhere** — Section 7 item 3: the plan is
  reconciled as built, in-commit, so the document the reviewer holds
  is never stale.
