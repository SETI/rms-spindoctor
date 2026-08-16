# SpinDoctor Corrected-Pointing C-Kernel Plan

*The design of record for recording each navigated image's corrected camera
attitude as a C-matrix in its metadata, and for generating SPICE C-kernels
from those matrices so that any SPICE-based tool can load navigated pointing
with a `furnsh`. Sections 1-3 are the as-built specification: they describe
the system that exists, and a change to the code is a change to them.*

---

## 0. Status

The design in sections 1-3 is implemented and merged: the navigator records
`cmatrix` / `cmatrix_original`, the frame identities and the exposure times
in every image's metadata, and `sd_create_ck` writes one type-3 segment per
navigated exposure into files mirroring the originals they correct, beside a
meta-kernel and a per-mission CSV report. The as-built system is documented
in `docs/user_guide/user_guide_ck_kernels.rst` (what the kernels claim, how
to load them, the report columns, the program's arguments) and
`docs/dev_guide/dev_guide_ck_kernels.rst` (the frame relations, the
derivation, the angular-velocity rationale, the writer's structure).

Section 4 records what each phase delivered and what it measured. Section 5
records the acceptance criteria and their status. What remains:

- **Suite coverage is below criterion 8's bound**, at 77% against the 90%
  the criterion asks for. That is the project's standing figure rather than
  a regression from this work: measured over the unit suite, every module in
  `spindoctor/cli/ck/` is at 99-100% and `spindoctor/spice_ids.py` at 100%,
  while `spindoctor/support/cmatrix.py` measures 80% there because its
  SPICE-dependent paths are exercised by the integration tier that a plain
  `pytest` skips. Nothing in this plan's scope raises the suite figure; the
  gap is the GUI and other long-standing untested surfaces.
- The follow-ups in section 7, none of which block use of the kernels.

The omission-reason set is five members, in the code and in section 3.3
alike. `DEGENERATE_EXPOSURE` never took the fifth place: the case it named
is handled with a single-record segment and no run could emit it, and a
published set with a member no run emits asks every consumer to write dead
code against a case that never arrives. `BASELINE_COVERAGE_GAP` holds it
instead, for an exposure whose baseline reproduced at the midtime and then
supplied no pointing at another of its record epochs.

---

## 1. Purpose and scope

Navigation measures where a camera was actually pointing. Today that
measurement leaves the pipeline as a pixel offset in a JSON file, which only
SpinDoctor knows how to apply. Expressed instead as a C-matrix and written
into a C-kernel, the same measurement loads into oops, ISIS, or a plain
`spiceypy` script with one `furnsh`, and every geometry computation those
tools perform is then correct without anyone touching an offset.

This plan does two things:

1. **Records the corrected attitude in the metadata**, as a C-matrix
   computed at navigation time. The pixel offset stays exactly as it is; the
   C-matrix is recorded beside it.
2. **Generates C-kernels from those C-matrices**, as an overlay: one type-3
   segment per navigated image, covering only that image's exposure, in
   files that mirror the coverage of the original kernels they correct.

The kernel writer consumes the recorded C-matrix. It performs no
offset-to-rotation conversion of its own, and it does not import oops.

**In scope:** the C-matrix computation and its metadata fields; the exposure
and clock fields the writer needs; the `sd_create_ck` program and its writer
core; file mirroring and naming; the per-mission frame handling; the BOTSIM
and Voyager rules; the comment area, meta-kernel, and CSV report; and the
round-trip validation that closes the loop from navigation through kernel
and back.

**Out of scope, deliberately:**

- Replacing the pixel offset. Every existing consumer of `offset` continues
  to work unchanged.
- Correcting images that did not navigate. They get no segment and fall
  through to the original kernels.
- New camera frames. The correction is written at the bus or platform level
  the existing kernels already describe.
- **Fitted twist.** The per-image `rotation_deg` a technique fits is a
  rotation about a per-technique pivot (a feature centroid), and that pivot
  is not recorded anywhere in the result or the metadata; a twist about the
  boresight and a twist about an off-center pivot differ by a translation
  of order several pixels. The correction is therefore not expressible from
  the recorded data. When a result carries a fitted rotation, no `cmatrix`
  is recorded and the generator omits the image with a reason (sections
  2.2, 3.2). Only Galileo SSI fits rotation today
  (`fit_camera_rotation: true` appears only in `config_410_inst_gossi.yaml`;
  pin behavior to the config key, not to prose). Twist support is a
  follow-up (section 7).
- Registration of the corrected kernels in the SPICE database oops selects
  from. That database is scheduled for replacement; a plain `furnsh` of the
  meta-kernel is the supported loading path, and registration is a
  follow-up (section 7).
- Ingesting metadata written by an older schema. The generator reads the
  current schema only; results are regenerated before kernel production.

---

## 2. The C-matrix in the metadata

### 2.1 Definition, and the frame trap this section exists to avoid

A C-matrix is the rotation taking a vector expressed in J2000 to the same
vector expressed in a frame:

```text
v_frame = C . v_J2000
```

which is what `cspyce.pxform('J2000', frame_name, et)` returns.

Two are recorded per navigated image, both in the **SPICE camera frame
convention** and both at the exposure **midtime**:

- `cmatrix_original` -- the attitude the furnished kernels gave at
  navigation time, before any correction.
- `cmatrix` -- the corrected attitude, the one a kernel should carry.

Recording both makes the correction self-contained: their difference is the
correction, and `cmatrix_original` is what lets the writer verify the
baseline kernels have not changed since navigation ran.

**The trap: the oops observation frame is not the SPICE camera frame.** The
navigated offset lives in the oops observation frame, and for three of the
four instruments oops builds that frame with a constant flip on top of the
SPICE frame:

| Instrument | oops frame vs SPICE frame | Where |
|---|---|---|
| Cassini ISS (NAC, WAC) | `R = diag(-1, -1, +1)` (a 180 degree flip) | `oops/hosts/cassini/iss.py` (`rot180`) |
| New Horizons LORRI | `R = diag(+1, -1, -1)` (SPICE boresight is -Z) | `oops/hosts/newhorizons/lorri.py` |
| Galileo SSI | `R = I` (oops uses `GLL_SCAN_PLATFORM` directly) | `oops/hosts/galileo/ssi.py` |
| Voyager ISS | no SPICE-frame relation at runtime: oops freezes a `Cmatrix` built from a tolerance-snapped `ckgp` (see 2.2) | `oops/hosts/voyager/iss.py` |

where `R` relates the two J2000-referenced attitudes as
`C_oops = R . C_spice`. A correction rotation built in the oops frame and
composed onto a `pxform`-derived matrix without conjugating through `R` is a
proper rotation of the right magnitude pointing the **wrong way** (for
Cassini, both tangent-plane components negated) -- and every hermetic test
that never touches SPICE passes anyway. `R` is therefore computed at
runtime, never assumed, and asserted against this table (section 2.2).

### 2.2 How the corrected matrix is derived

This is the part a future oops call will replace, so it lives in one module
behind one function and nothing else computes it.

**Step 1 -- the corrected boresight in the oops frame.** The navigated
offset is applied downstream as `oops.fov.OffsetFOV(fov, uv_offset=(du,
dv))` (note the order: the metadata `offset` is `[dv, du]`, and both
existing consumers construct the FOV with `(du, dv)`). That FOV maps pixels
to camera tangent-plane coordinates as

```text
xy_from_uv(uv) = fov.xy_from_uv(uv) - xy_offset
xy_offset      = fov.xy_from_uv(fov.uv_los + (du, dv))
```

Under the corrected pointing, the true direction seen by pixel `uv` in the
original oops frame is `fov.los_from_xy(fov.xy_from_uv(uv) - xy_offset)`.
The corrected frame is the one in which the *unmodified* FOV holds, so the
rotation `M` (oops-frame coordinates, original to corrected) satisfies

```text
M . los_from_xy(xy - xy_offset) = los_from_xy(xy)     for all xy
```

Evaluating at the boresight (`xy = xy_from_uv(uv_los)`, which need not be
zero for a subarray or distorted FOV) gives the constraint

```text
d = los_from_xy(xy_from_uv(uv_los) - xy_offset)
M . d = los_from_xy(xy_from_uv(uv_los))
```

**Step 2 -- construct `M`.** `M` is the minimal rotation taking `d` to the
boresight direction `b = los_from_xy(xy_from_uv(uv_los))`: axis
`d x b` normalized, angle `arctan2(|d x b|, d . b)`, realized with
`cspyce.axisar(axis, angle)` -- which is the **active** vector rotation
(`M . d = b`); `cspyce.rotate` is the frame rotation and gives the
transpose. Guard: when `|d x b| < 1e-12` (a zero or sub-nanoradian offset),
`M = I`. `M` is exact by construction; nothing is orthonormalized. A fitted
`rotation_deg` on the result means no `cmatrix` is computed at all (section
1); there is no twist term in `M`.

The angle is `arctan2(|d x b|, d . b)` and not the mathematically
equivalent `arccos(d . b)` this plan first specified, because for unit
vectors `arccos` loses all relative precision as the angle goes to zero,
which is exactly the regime a sub-pixel offset lives in. Measured against a
Cassini NAC pixel scale of 6e-6 rad/px: at 0.01 px `arccos` returns
5.9605e-08 rad against a true 6.0000e-08 (0.7% low), and at 0.001 px it
returns **exactly 0.0** -- `cos` of 6e-9 rad rounds to 1.0 in float64, so
the correction would be silently dropped altogether. `arctan2` returns the
true angle in every one of those cases. This is a correction of a defect in
the formula, not a change of method: the two forms agree wherever `arccos`
has any precision left.

An exact rigid rotation is not exactly a uniform tangent-plane shift; the
difference is second order in field angle. Measured over a 17x17 pixel grid
across each full frame, worst case over eight offset directions, for **50
pixels of total boresight displacement**, comparing `M` applied to oops's
own `OffsetFOV` line of sight against the unmodified FOV:

| Instrument | worst residual (rad) | in tangent-plane px | in pixel space |
|---|---|---|---|
| Cassini NAC | 6.01e-9 | 1.00e-3 px | 1.24e-3 px |
| Cassini WAC | 5.91e-6 | 9.89e-2 px | 7.86e-2 px |
| New Horizons LORRI | 1.62e-8 | 8.15e-4 px | 1.23e-3 px |
| Galileo SSI | 1.82e-8 | 1.79e-3 px | 1.79e-3 px |
| Voyager 2 NAC | 1.29e-8 | 1.64e-3 px | 1.64e-3 px |

These replace this plan's earlier figures of "about 1e-9 radians on a
Cassini NAC" and "at most ~0.04 px at the corner of a WAC", which were low
by roughly 6x and 2.5x respectively; the 0.04 px figure corresponds to an
offset closer to 20 px than to 50.

**The residual is essentially linear in the offset**, not quadratic.
Measured across 12.5 / 25 / 50 / 100 px of total displacement, each
doubling multiplies it by 2.034, 2.067, 2.129 -- on both the NAC and the
WAC, identically. The two figures above confirm it independently: 8.72e-9
rad at 70.71 px against 6.01e-9 at 50 px is a ratio of 1.452, where linear
predicts 1.414 and quadratic 2.000. The term is second order in *field
angle* and first order in the offset -- it goes as offset times field angle
squared -- and an earlier revision of this plan conflated the two
variables. Quoting the residual without the offset it was measured at is
still meaningless, but halving the offset only halves the residual.

**The WAC case sits essentially at Phase D's 0.1 px decision boundary.**
9.89e-2 px of total displacement at a 50 px total offset, against a target
stated as 0.1 px *per axis* -- so the comparison is conservative (a total
displacement is at most sqrt(2) times the larger per-axis component), but
the two are not in the same units and Phase D must convert before deciding.
That is a fact to confront with the rule as written, not a reason to move
the bound: a WAC round trip at a 50 px offset should be expected to consume
nearly the whole budget from this term alone. Because the term is linear,
buying headroom costs proportionally more offset than a quadratic rule
would suggest: a Phase D WAC frame at about 10 px of total offset lands
near a fifth of budget, where a quadratic reading would have promised that
at ~22 px.

**Step 3 -- express both attitudes in the SPICE convention.** With
`C_oops` the observation frame's J2000-to-camera matrix at midtime
(evaluated from the observation's own frame object) and, for Cassini,
Galileo and LORRI,

```text
cmatrix_original = pxform('J2000', camera_frame, et_mid)
R                = C_oops . cmatrix_original^T
cmatrix          = (R^T . M . R) . cmatrix_original
```

`R` is asserted equal (to 1e-9) to the section 2.1 constant for the
instrument, and asserted epoch-independent by recomputing at `start_et` and
`stop_et`; a violation raises rather than being absorbed. For **Voyager**,
oops builds the observation frame as
`P . ckgp(ck_id, sce2t(scid, et_mid), 800 + texp/48, 'J2000')` with
`P = pxform('VGn_SCAN_PLATFORM', camera_frame, 0)` -- frozen,
time-independent, and tolerance-snapped, so `pxform` at midtime does not
reproduce it. For Voyager, `cmatrix_original = C_oops` itself (already the
SPICE convention built on the snapped platform attitude), `R = I` by
construction, and the writer must reproduce the baseline with the same
snapped `ckgp` call (section 3.3) -- which is why `exposure_s` is a
recorded field.

The tick conversion is `sce2t`, not the `sce2c` this plan first named:
`oops/hosts/voyager/iss.py` calls `cspyce.sce2t(scid, tstart + texp/2.)`.
The two differ. On `C1205021_GEOMED` (Voyager 2, texp 0.48 s), `sce2t`
returns 6349696766.0 and `sce2c` returns 6349696766.186253 -- 0.19 ticks
apart against a tolerance of 800.01, so on this frame both find the same
pointing record (found tick 6349696815.0). It is nonetheless the wrong
call, and section 3.3's reproduction step must use `sce2t` to be sure of
matching oops on every frame.

Sanity check on the result: `|det(cmatrix) - 1| < 1e-9`,
`max|cmatrix . cmatrix^T - I| < 1e-9`, and every element finite, all three
raising on violation -- a non-proper rotation here is a defect, not
something to repair. The finiteness check is not redundant with the other
two: `NaN` fails every inequality, so a `NaN` matrix passes both tolerance
guards silently, and the metadata writer emits the C-matrices and epochs
unrounded (they must reproduce to 1e-9 rad, which rounding would break),
bypassing the rounding helper that maps non-finite floats onto the JSON
sentinel. An unchecked `NaN` therefore reaches the file as a bare `NaN`
token, which is not valid JSON and which the results-index ingest rejects.

### 2.3 Metadata fields

Added under `navigation_result` (so they exist exactly when a `NavResult`
exists; a load-error document has no `navigation_result` and none of
these):

```text
pointing:
  cmatrix:            [9 floats, row-major]   # corrected, J2000 -> camera, SPICE convention;
                                              # absent when rotation_deg was fitted
  cmatrix_original:   [9 floats, row-major]   # uncorrected, same convention
  camera_frame:       "CASSINI_ISS_NAC"       # SPICE frame name
  camera_frame_id:    -82360
  ck_frame_id:        -82000                  # the object a CK targets
times:
  start_et:           float                   # TDB seconds past J2000
  stop_et:            float
  midtime_et:         float
  exposure_s:         float
  sclk_start:         "1/1484573293.055"      # sce2s string forms
  sclk_midtime:       "1/1484573295.118"
  sclk_stop:          "1/1484573297.181"
```

and, under `observation`, `shutter_mode` (the instrument's shutter mode
string when the host exposes one, e.g. Cassini's `BOTSIM`; the BOTSIM rule
in section 3.2 is undetectable without it).

`cmatrix_original` and the `times` and frame fields are written whenever a
`NavResult` exists; `cmatrix` additionally requires an offset and no fitted
rotation. `ck_frame_id` is derived at navigation time from the observation's
host identity -- for Voyager, from the spacecraft number
(`scid * 1000 - 100`) -- and recorded per image so the writer never infers
it. It does not belong in the instrument config files: the config is keyed
per instrument, and Voyager needs a different id per spacecraft under one
key.

These field names and shapes are also the ones the results-index schema
declares in advance (`plans/RESULTS_DB_PLAN.md` section 2.3); changing them
here means changing them there.

### 2.4 Where it is computed

The computation lives in `spindoctor/support/cmatrix.py`, a single entry
point taking the observation, the offset, and the fitted-rotation flag, and
returning a frozen dataclass carrying both matrices, the frame identities,
and the times block. It imports oops -- it must, to read the observation's
frame and FOV -- which is fine on the pipeline side; the constraint that
bans oops applies to the kernel writer (section 3.6). When oops gains its
own corrected-attitude API, this module's body is replaced and its
interface stays. A `#` comment records that intent.

`NavResult` is constructed inside the ensemble, which never sees the
observation, and it is a frozen dataclass -- so the wiring is: `NavResult`
gains an optional `pointing` field (default None), populated via
`dataclasses.replace` where the observation is in hand, and
`curator.build_metadata_dict` serializes it. This is stated explicitly so
the implementer does not go looking for an `obs` argument inside
`ensemble()` that does not exist.

The stamping site is `NavOrchestrator.navigate` in
`nav_orchestrator/orchestrator.py`, not `_navigate_pipeline` as this plan
first said. `_navigate_pipeline` has five early failure returns and
`navigate` has two more of its own -- the hard-failure image-class
short-circuit, which never enters the pipeline at all, and the
`NavContractError` path -- so stamping the pipeline's final return alone
would leave the uncorrected matrix and the times off every failed result,
against this section's own "whenever a `NavResult` exists". `navigate`
routes all three of its returns through one `with_pointing(result, obs)`
method instead.

That method is **public**, because the manual-navigation driver needs it:
`run_manual_nav` in `nav_technique/nav_technique_manual.py` builds its
`NavResult` directly from the operator's pick and never calls `navigate`,
so it calls `with_pointing` itself. Operator-ratified offsets are the
highest-quality pointing in the corpus; leaving them unstamped would make
them the one subset excluded from every generated kernel.

`with_pointing` absorbs only `NavPointingError`; everything else
propagates. A pointing solution is recorded metadata rather than the
navigation itself, so an expected failure is reported and the field is left
unset -- no wrong C-matrix is ever recorded, which is the property the
raises in section 2.2 exist to guarantee. That absorption is the one
qualification on `navigate`'s otherwise unconditional guarantee not to
raise through to its caller: a defect beneath the attitude computation
does reach the caller, deliberately. Two constraints on how it absorbs:

- The caught set is exactly `NavPointingError`, the typed exception in
  `support/exceptions.py` that the computation raises for every failure it
  expects: its own guards, and the frame, kernel and clock lookups SPICE
  cannot answer, each converted at the call site with `raise ... from` so
  the original traceback survives. Everything else propagates. Catching the
  untyped `LookupError` / `OSError` / `RuntimeError` / `ValueError` family
  instead would make a `ValueError` or `RuntimeError` from a defect inside
  the computation indistinguishable from an expected SPICE failure, and
  quietly drop pointing from a 50,000-image batch while every image still
  reports `status=success`.
- Anything that degrades or omits a solution goes to **both** logs: detail
  to the image log, one line to the run log. An operator watching a batch
  must not have to open every per-image log to learn that pointing stopped
  being recorded. The same applies to a registered instrument that reaches
  navigation with no entry in the frame table (section 2.1): that is a
  build defect and warns to both logs, where a simulated image, which has
  no spacecraft and no furnished camera frame, is expected and logs at
  debug.

---

## 3. The kernel generator

### 3.1 Frames, per mission

The correction is measured at the camera and written at the object the
existing kernels describe, with reference frame J2000:

| Mission | CK object | Camera-to-CK-object relation |
|---|---|---|
| Cassini ISS | -82000 (bus) | fixed FK rotation, **not** small |
| Voyager ISS | -31100 / -32100 (scan platform) | fixed FK rotation (the platform-to-camera `pxform` oops applies; not identity) |
| Galileo SSI | -77001 (scan platform) | identity (SSI reads the platform frame) |
| New Horizons LORRI | -98000 (spacecraft) | fixed FK rotation |

`F = pxform(ck_frame, camera_frame, et_mid)` is always computed, never
assumed -- Cassini's `F` is a permutation-like matrix nowhere near
identity. The corrected attitude of the CK object and the correction as a
rotation in the CK frame's own coordinates are

```text
C_ck_corrected(mid) = F^-1 . cmatrix
delta               = C_ck_corrected(mid) . C_ck_original(mid)^T
                    = F^-1 . (R^T M R) . F        (the oops-frame M conjugated into the CK frame)
```

with `C_ck_original(mid)` read from the original kernels. Across the
exposure the correction is held **body-fixed**:

```text
C_ck_corrected(t) = delta . C_ck_original(t)      for t in [start, stop]
```

That is the physical model -- the spacecraft is pointed slightly wrong and
the error turns with it -- so the *correction* is right at every epoch in
the window. What the segment reproduces between its records is a separate
question: it carries records at start, midtime and stop, plus a 1 s cadence
above 10 s of exposure (section 3.3), and interpolates between them, so
smear geometry is right only to the fidelity of that interpolation. That
fidelity is measured in Phase D's interior note and is weaker than this
paragraph once claimed; bounding it needs a denser, adaptive cadence
(#444). **Exception: Voyager.** The
navigated model assumed a constant, tolerance-snapped attitude (section
2.2), so a Voyager segment carries that single corrected attitude,
constant across its window; writing time-varying pointing there would
disagree with what was navigated.

### 3.2 What gets a segment

- **Eligible:** any image whose status is `success` or `conflicted` and
  which has a recorded `cmatrix`. No confidence or rank threshold. The
  consequence is that the report and the segment comments must carry
  status, status reason, confidence, and rank, since they are then the only
  way a consumer can filter low-confidence or conflicted pointing.
- **Not eligible:** every other image -- including a result with a fitted
  `rotation_deg`, which has no `cmatrix` (section 1) and is reported as
  `rotation_unsupported`. No segment; pointing falls through to the
  originals. No uncorrected copies are written.
- **BOTSIM pairs** (simultaneous Cassini NAC and WAC exposures) would ask
  one bus attitude to honor two corrections at once. Pairing predicate: two
  eligible Cassini images with `observation.shutter_mode == 'BOTSIM'`,
  opposite cameras, and `|stop_et_a - stop_et_b| <= 2.0 s` -- the shutter
  close is the commanded simultaneous event, and unequal exposure
  durations push the starts of a genuine pair whole seconds apart. One
  correction per pair: NAC wins; WAC keeps its own correction when the NAC
  member is not eligible or when no candidate kernel reproduces the NAC
  baseline, since a winner that writes nothing suppresses nothing. The
  loser is reported as `botsim_loser`.
- **Voyager** corrected files pair only with the per-encounter SEDR
  platform kernels (the platform object is the only one Voyager ISS
  pointing reads; the decades-spanning bus kernels cover a different object
  and an image served only by them cannot have navigated in the first
  place). An eligible image whose baseline no candidate reproduces --
  Voyager or otherwise -- is reported as `no_reproducing_baseline`
  (section 3.3).

### 3.3 Segments and files

One **type 3** segment per eligible image, interpolation interval exactly
`[start_et, stop_et]`.

**Records.** At exposure start, midtime, and stop. When the span from
start to stop reaches 10 s, additional records at a 1 s cadence, capped at
10,000 records for the segment. Both the cadence gate and the record count
read that one span, and `ImagePointing` requires the recorded `exposure_s`
to agree with it, so the two cannot answer for different exposures; the cap
is there because the arithmetic that expands a span has no bound of its own,
and a recorded span of 1e9 s exhausts memory rather than being recognized as
epochs that are not an exposure. The boundary is inclusive because ten
seconds exactly is an ordinary commanded ISS exposure: measured over 40
random epochs on `04002_04009ra.bc` in NAC pixels, a 10.000 s exposure at
three records leaves mean 0.43 px, p99 5.16 px, max 9.86 px and 23.0% of
samples beyond 0.1 px, where 10.001 s at the 13-record cadence leaves 0.02,
0.44 and 5.92 px and 3.0%.

Those cadence records earn their place, and it is worth saying how, because
nothing asserts them. The segment declares one interpolation interval
spanning `[start_et, stop_et]`, so SPICE interpolates between bracketing
records for **every** epoch in the window -- an epoch between records is
interpolated, not fallen through to the original kernels, which happens
only outside the window. Each added record therefore shortens every
interpolation span it touches and improves accuracy continuously across the
window rather than at instants. Measured on a real Cassini reconstructed
kernel, a 60 s exposure carrying only the three mandatory records has a
worst interior error of 25.98 px; at the 1 s cadence it is 1.07 px. The
cadence reduces interior error by more than twenty-fold and **bounds
nothing** -- 1.07 px is still well outside any tolerance this plan states,
which is why a consumer is told (Phase D's interior note) that only the
record epochs are claimed. Replacing this fixed cadence with an adaptive
one that does bound the error is #444.

(SPICE offers no API to enumerate a type-3 segment's interior records, so
the earlier idea of copying the original's record times is dropped.) Time
tags must be
**strictly increasing in encoded SCLK**; an exposure so short that start,
mid and stop collapse to one tick produces a single-record segment at
midtime, which type 3 permits and Phase B tests. Records are quaternions
from `cspyce.m2q` with **sign continuity enforced**: `m2q` fixes the scalar
component non-negative, which flips the sign between adjacent records
whenever the attitude's rotation angle passes 180 degrees; each record is
negated as needed to keep a non-negative dot product with its predecessor.

Phase B measured two things about that paragraph. First, `sce2c` returns
*continuous* encoded SCLK -- a tick with a fractional part -- so the
single-record collapse happens only when the three epochs are
indistinguishable as doubles (a nanosecond exposure near ET 5e8), not
merely when the exposure is shorter than one tick: measured against a
1/256 s clock, a 1 ms exposure is 0.256 ticks and still produces three
records. **The single-record path is therefore unreachable for any real
exposure** -- the shortest Cassini ISS exposure is 5 ms -- and it exists as
a guard, not as a case the corpus contains. When it does fire the midtime
record is bit-identical to the start record, so "at midtime" is a statement
of intent rather than an observable. Second, SPICE's own type-3
reader restores quaternion sign on the way out: a segment written with a
sign-discontinuous sequence reads back attitudes identical to a repaired
one (0.0 rad difference at an interior epoch, measured). The enforcement
stays, because the file should say what it means, but the test guarding it
must assert on the written records; no read-back assertion can see the
difference.

**Clocks.** Encoded SCLK comes from `cspyce.sce2c(sclk_id, et)`. The
spacecraft clock ID (-82, -31, -77, -98) is **not** derivable from the CK
object id by integer division (`-31100 // 1000` is -32 in Python: wrong
spacecraft, silently). The SCLK kernel furnished must be the one navigation
used, resolved from `provenance.spice_kernels`; a different SCLK is a silent
time-tag error.

**How it is resolved, exactly.** Not by name and not by version: the
basenames say nothing (`cas00172.tsc`, `vg200022.tsc`,
`new-horizons_1280.tsc`), and a Voyager run's provenance names both
spacecraft's clock kernels. Each candidate the image's provenance names is
furnished on its own and asked whether it defines the clock the image's own
`ck_frame_id` resolves to -- the pool variables `SCLK_DATA_TYPE_<n>` and
`SCLK_PARTITION_START_<n>`, either of which is enough, both present in every
kernel in the holdings. Exactly one must; **zero and several are both refused
rather than resolved by picking**, since two versions of one clock disagree
about the very thing being encoded. The probe needs the clock not to be
defined already, or a kernel furnished earlier would answer for every
candidate alike, so that is checked rather than assumed.

**The frame kernels get the same discipline, differently shaped.** A mission
furnishes several frame kernels at once by design -- Cassini's dynamic, rocks
and status kernels beside its main one -- so they are not resolved to one.
What is refused is two of them defining a frame *this run's images name*: the
camera frame the reproduction test asks through, or the frame naming a
corrected object. A text kernel's last assignment wins, and two versions of a
mission's frame kernel differ in exactly the assignment that fixes the camera
to the spacecraft, so a corpus navigated across a frame kernel upgrade would
be reproduced entirely against whichever version sorted last. Every image
navigated under the other then fails to reproduce, the run writes nothing, and
it reports the whole corpus as `no_reproducing_baseline` -- the drift verdict,
for a pool the run built itself. That is the misdiagnosis item 4 below exists
to prevent, arriving by a different route, and it is refused the same way:
before any candidate is tried, naming the two kernels and the frame they
disagree about.

The choice is per image and the pool is per run, so the run's images must
agree: two images of one spacecraft whose records name different versions of
its clock kernel cannot both be encoded correctly by one pool, and the run
refuses rather than encoding one of them against the other's kernel. (This
run-level requirement is a consequence of the per-image rule, not a second
rule; it is stated because it is a refusal an operator can meet.)

**The shared table is the resolver; `ckmeta` is only the cross-check.**
`sclk_id` comes from the CK-object-to-clock mapping in
`spindoctor/spice_ids.py` (section 3.6), and `cspyce.ckmeta(ck_frame_id,
'SCLK')` is then required to agree with it. It is deliberately not the
other way around, because `ckmeta` computes rather than validates:
`ckmeta(-999999, 'SCLK')` returns `-999` and `ckmeta(-12345, 'SCLK')`
returns `-12`, neither raising. A `ck_frame_id` that is wrong for any
reason would otherwise yield a plausible-looking clock id, a successful
`sce2c`, and silently wrong time tags on every record. Both call sites --
the writer's `resolve_sclk_id` and the attitude computation's own resolver
-- return the **recorded** id rather than the one `ckmeta` computed, even
though the check has just proved them equal, so that weakening the check
later cannot quietly promote `ckmeta` back to being the source.

**Both attitudes are evaluated at the exposure midtime**, and Phase A's
integration tier pins that against a moving attitude on LORRI alone -- the
only frame in the cohort with an exposure (5 s) long enough for the
attitude to move measurably between start and midtime. That is adequate but
thin. It matters most here: a midtime/start mix-up in the reproduction step
would fail every baseline at the 1e-9 rad bound, so an unexplained
across-the-board `no_reproducing_baseline` should be checked against this
first.

**Angular velocity: copied unchanged, never rotated, and never declared
absent.** CK angular velocity is expressed in the segment's **base reference
frame** (J2000), per the CK Required Reading, and the corrected frame differs
from the original by a constant body-fixed rotation -- two frames rigidly
attached to each other have identical angular velocity in the base frame. So
when the original segment carries AV (`ckgpav` succeeds), the corrected records
carry the same vectors bit-identically and `avflag = 1`. Rotating AV through
`delta` -- superficially the "thorough" treatment -- is the wrong-frame error,
and Phase B's test must fail if it is introduced.

**`avflag = 0` is never written.** A segment with `avflag = 0` is not a segment
whose angular velocity is unknown: SPICE skips it entirely for `ckgpav` and for
`sxform`, falling through to the next loaded kernel that does carry AV for that
object and epoch and answering with **that kernel's uncorrected attitude**.
Measured on a real reconstructed Cassini kernel (`04002_04009ra.bc`, all 2645
local -82000 segments carry AV), holding everything but `avflag` constant with
a 1.000e-04 rad correction applied:

| corrected segment | `ckgp` vs `ckgpav` | `sxform` vs corrected |
|---|---|---|
| `avflag = 0` | 1.000e-04 rad | 1.000e-04 rad |
| `avflag = 1` | 0.000e+00 rad | 0.000e+00 rad |

That is decisive because oops reads pointing through `sxform`:
`oops/frame/spiceframe.py` defaults `omega_type='tabulated'` and calls
`cspyce.sxform` in that mode, and no oops host overrides it. An `avflag = 0`
corrected segment would deliver its correction to `ckgp` and `pxform` and
withhold it from `sxform`, with which of the two a consumer saw depending on
what else was in their pool. Exposure across the local baselines: Cassini
-82000 2645/2645 segments carry AV and New Horizons -98000 4346/4346 do, so
those are immune today; **Galileo -77001 has 150 segments of which 38 carry
none**; Voyager -31100/-32100 have 9 segments, none with AV.

So the writer applies **all records or none, and none means refuse**. SPICE
reports the want-of-AV failure exactly as it reports pointing that is not
covered at all -- `found` false from `ckgpav`, one answer for two conditions
that mean opposite things here, since one refuses the run and the other omits
one image; the sampling pass over every record decides, and when it comes back
empty-handed a second pass reads attitude alone, so a genuine coverage gap
surfaces as itself (`BaselineCoverageGapError`) and an exposure that had only
its AV missing is refused with a `ValueError` naming that. Both lookups use the
flag-returning form of the SPICE call rather than the raising one, so "nothing
here" is told apart from a pool with no C-kernel furnished, an undefined
reference frame, or a kernel that cannot be read -- each of which still raises
and still stops the run, and none of which may be read as a data condition.
Choosing the form per call does not touch the process-wide `use_errors` regime.
Refusal rather than zeros
for the records that lack AV: the attitude would be right, but a scan platform
genuinely parks, so an invented zero is indistinguishable from a measured one,
and the overlay would start answering `sxform` at epochs where the pool has no
answer at all (measured on Galileo: at an epoch covered only by an AV-less
segment, `ckgp` and `pxform` succeed while `ckgpav` and `sxform` raise).

A **frozen (Voyager) segment writes zeros with `avflag = 1`**: the segment's
attitude is constant, so its angular velocity is zero -- a measurement, not an
invention -- and writing it that way is what makes `sxform` return the corrected
attitude. The baseline's own vectors are still not copied there, since the
rigid-attachment argument does not hold for a segment that deliberately drops
the baseline's time variation.

**Files mirror the originals.** Each output `.bc` corresponds to exactly
one original CK file and carries the segments of the images whose corrected
attitude that original supplied. Output size grows with the images
corrected, not with the originals' own size; the baseline pairing is
legible from the filename; and regeneration is per original file. Naming: the original basename with
`_nav` before the extension -- `03236_04002ra.bc` becomes
`03236_04002ra_nav.bc` (a real reconstructed-kernel name; use real names in
tests). No label files are written.

**Assignment is by reproduction, not by guessing.** `provenance.
spice_kernels` records sorted basenames only (deliberately, for hash
determinism), it does not preserve load order, and in a batch process it
accumulates kernels from earlier images -- a superset. So:

1. **Pre-index once per run:** every CK under the mission's kernel
   directories is scanned with `ckobj` and `ckcov` (SEGMENT level, TDB), so
   candidates are filtered by CK object and midtime coverage before any
   furnishing. Basenames from provenance resolve against this index; a
   basename found in more than one directory contributes each file as a
   candidate. A basename whose stem ends in `_nav` is **not** indexed:
   writing the corrections back beside the originals is the natural
   workflow, a corrected kernel reproduces its own baseline exactly
   wherever the correction was the identity (which the section 2.2 `M = I`
   guard makes bit-exact for a zero-offset image), and its name sorts after
   the original's, so it would win the tie-break and then abort the run at
   output naming. Coverage for a **frozen-attitude (Voyager) object is read
   with `ckcov`'s tolerance set to the widest tolerance its navigated
   lookup could have used (80000 ticks)**, which lengthens each interval by
   that much at both ends: the snapped lookup answers with a record up to
   its tolerance away from the epoch asked for, so an image navigated
   through the fallback tolerance has a midtime outside the segment window
   by construction, and a tol-zero filter would drop the only candidate
   that reproduces it. The filter is only a filter -- reproduction decides
   -- so it is widened rather than tightened.

   **An object whose spacecraft clock no kernel defines is recorded as
   unreadable rather than stopping the scan.** `ckcov` reports coverage in
   TDB, which needs that clock, and a real kernel can name an object that
   has none: `nh_scispi_2015_recon.bc` in the New Horizons holdings
   describes object **-1** beside -98000, `ckmeta` computes its clock as 0,
   and no SCLK kernel supplies one, so `ckcov` raises
   `SPICE(KERNELVARNOTFOUND)`. Refusing the scan there makes the whole
   mission unindexable for the sake of an object no image will ever ask
   about -- the defect Phase D hit on the first LORRI frame it ran. Such an
   object contributes no coverage and is therefore never offered as a
   candidate; an image that *does* correct one is refused before any
   candidate is tried, naming the missing clock, exactly as a missing frame
   kernel is (item 4 below), rather than being reported as a baseline that
   drifted. The driver warns once in its run log, naming the skipped
   objects, so a kernel set missing a clock says so even when no image
   needs it.
2. **Per image** (grouped by candidate set so the kernel pool is switched
   per group, not per image): furnish the supporting kernels (LSK, SCLK,
   FK) plus one candidate CK at a time, evaluate the original attitude at
   midtime -- for Voyager via the snapped
   `ckgp(ck_id, sce2t(scid, mid), 800 + exposure_s/48, 'J2000')`, composed
   with `P`, matching section 2.2 (`sce2t`, not `sce2c`; see the evidence
   there); otherwise via `pxform` -- and keep
   candidates that reproduce the recorded `cmatrix_original` to within
   **1e-9 radians** (an angular tolerance on the rotation between the two
   matrices; this is a reproduction bound, far tighter than any navigation
   tolerance).
3. **Tie-break** when several reproduce (expected: the holdings carry
   reconstructed, gapfill and predicted sets with overlapping coverage, and
   oops loads gapfill by default): prefer by kernel class **read from the
   kernel's own basename** -- reconstructed over gapfill over predicted --
   then the lexicographically greatest basename. The reproducing candidates
   agree on the attitude by construction, so the choice affects only which
   output file carries the segment; it must merely be deterministic. The
   class comes from the basename rather than from the directory holding it
   for three reasons. The directory is a redundant restatement of what the
   name already says, so it can only ever agree or be wrong. Two of the six
   real Cassini CK directories name no class while every basename in them
   does -- `CK-cruise` and `CK-jup`, 95 reconstructed kernels that a
   directory reading ranks below predicted. And the basename is the form
   the eventual kernel manager works from, so classifying by name is the
   same fact expressed once instead of twice. Measured over the holdings,
   all 1316 Cassini kernels classify (1093 reconstructed, 15 gapfill, 208
   predicted). The encoding is mission-specific and is declared one mission
   at a time: New Horizons marks only the two kernels released in both
   forms, with a trailing `_recon` or `_pred`; Voyager and Galileo encode
   nothing, so every kernel of theirs is `UNCLASSIFIED`, the class rank
   ties for every candidate and the tie-break falls through to the basename
   exactly as before. A kernel whose name declares no class ranks last, so
   a name that does say is always preferred over one that does not.
4. An eligible image **no** candidate reproduces gets no segment, reason
   `no_reproducing_baseline`. This is also the baseline-drift detector: if
   the kernel set changed since navigation, reproduction fails and the
   image is refused rather than corrected against a baseline that no
   longer exists. **Which is why a frame the images name but the pool does
   not define is refused before any candidate is tried**, rather than
   arriving as a failed `pxform` and being reported as drift: a frame
   kernel that was never furnished defeats the same lookup for every image
   alike, so a run that forgot it would write nothing, blame the holdings,
   and say so 50,000 times.

**Voyager has a second tolerance to try.** When the snapped `ckgp` above
raises `LookupError`, `oops/hosts/voyager/iss.py` does not fail the image:
it warns and falls back to the FK-registered frame
`VOYAGER<n>_ISS_<NAC|WAC>`, which chains on
`SpiceType1Frame('VG<n>_SCAN_PLATFORM', -3<n>, TOL_TICKS)` with
`TOL_TICKS = 80000.` (raised from 800 in oops "to deal with very long
exposures"). The recorded `cmatrix_original` is the observation frame's
attitude at midtime either way, so Phase A needs no special case, but a
Voyager image navigated through that fallback is reproduced only at the
80000-tick tolerance. The reproduction step should try `800 + exposure_s/48`
first and 80000 second; an image that reproduces under neither is a genuine
`no_reproducing_baseline`. **Each attempt encodes the epoch the way the call
it reproduces encodes it**: `sce2t` (a whole tick) for the first, because
that is what `voyager/iss.py` calls, and `sce2c` (a fractional tick) for the
second, because `SpiceType1Frame.transform_at_time` calls `sce2c`. The two
are not interchangeable on a baseline that interpolates between records --
measured on a 1/256 s clock, a midtime 3 ms off a tick boundary reads two
attitudes 2.7e-6 rad apart, three orders above the reproduction bound -- and
on a type 1 baseline, which is what the real Voyager kernels are, they
usually find the same record.

**Type 1 is why the tolerance exists at all, and a type 3 test cannot see
it.** A type 3 segment interpolates, so it answers any epoch inside its
window whatever tolerance is asked for; a type 1 segment answers only
within the tolerance of a record it holds. Measured on
`vg2_sat_version1_type1_iss_sedr.bc` over 200 epochs spread across its
window, a lookup finds pointing at tolerance 0 from 2 of them, at 800 from
25, and at 80000 from 130 -- and the records do not sit on whole clock
ticks, so tolerance 0 misses a record even when asked at that record's own
epoch. Any hermetic test of the tolerances therefore has to write a type 1
baseline (`ckw01`), and the two attempts have to be observable
individually: the wider one subsumes the narrower, so no reproduce-or-not
outcome can tell them apart.

One consequence of the fallback frame to know rather than to fix:
`SpiceType1Frame` caches its transform for `tick_tolerance` converted to
seconds, measured at 4799.995 s for Voyager. Two Voyager images navigated
through the fallback within eighty minutes of each other in one process
share one attitude, and the second then records an attitude no lookup at
its own midtime reproduces, so it is honestly refused.

**Omission reasons**, the complete set: `not_eligible`, `botsim_loser`,
`rotation_unsupported`, `no_reproducing_baseline`, `baseline_coverage_gap`.

`degenerate_exposure` was declared and removed rather than kept in
reserve. The case it named -- three epochs that encode to one tick -- is
*handled*, with a single-record segment, and no path could ever emit it, so
publishing it obliged every report consumer to handle a value that could
not arrive. `baseline_coverage_gap` takes the fifth place for a case that
does occur: an exposure whose midtime reproduces against a baseline, which
is what paired the image with it, but one of whose record epochs the
baseline cannot answer, which happens near a coverage gap.

### 3.4 Companion outputs

- **Comment area**, per file: generator version, configuration hash, the
  baseline kernel basenames and the SCLK kernel used, and one line per
  image (name, time, offset, sigma, confidence, rank, status, status
  reason). Mechanics: segments are written through `ckopn` (with `ncomch`
  reserved generously up front -- estimate the comment size and reserve
  with slack) / `ckw03` / `ckcls`; the comment text then goes in via
  `dafopw` / `dafac` / `dafcls`. Reading it back (acceptance) uses
  `dafopr` / `dafec`.

  Three measurements against the installed toolkit (CSPICE N0067) shape
  what that reservation buys and what a comment line may hold, and only
  the first was expected. **The comment area is grown when it has to
  be:** a `dafac` that overflows what `ncomch` reserved does not fail, and
  does not truncate -- SPICE extends the area by shifting every data
  record in the file, and the comments read back complete. Measured on a
  one-segment kernel with 2130 characters of comment: reserving 2130 or
  3130 leaves the first data record at 5 and 6 respectively and unmoved by
  the write; reserving 1065 moves it from 4 to 5, and reserving 0 moves it
  from 2 to 5. So the reservation buys a file that is not rewritten
  rather than a comment that is not lost, the failure is silent either
  way, and the only observable is that displacement -- which is what the
  Phase E test asserts, since no read-back can see the difference.
  **A comment line longer than 255 characters is stored and then cannot be
  read back at all:** `dafec` reads into a 255-character buffer and raises
  `SPICE(COMMENTTOOLONG)` on the first line that overflows it, which
  loses the whole comment area rather than one line's tail. **Trailing
  whitespace does not survive and a non-printing character is refused**
  by `dafac` outright -- after the segments have been written, so the file
  is left with a comment area it was meant to have and does not. All three
  are refused before a file is opened.
- **Meta-kernel** per file set, furnishing originals first and corrections
  after, so precedence is explicit rather than an ordering a user has to
  know. A SPICE text kernel holds at most 80 characters per string value
  and **truncates a longer one silently** rather than refusing it, so a
  kernel path over 80 characters -- which every path in the holdings tree
  is -- is written through SPICE's `+` continuation, as several strings of
  at most 79 characters each. The one path that cannot be expressed is one
  ending in `+`, whose last character is indistinguishable from the marker
  that would join it to the next kernel; it is refused by name.

  Blanks need the same care and get less of it from SPICE. A **trailing**
  blank is trimmed from a text kernel string, so a path whose last character
  is a blank comes back one character shorter than it went in -- measured
  through a real `furnsh`, a 119-character path ending in a space loads a
  118-character name, with no error at all -- and such a path is refused. A
  blank *inside* a path survives at a join on the toolkit installed here:
  swept across every position of a 119-character path (two pieces) and a
  200-character path (three pieces), the only position that loses its blank
  is the last character of the path itself, and every piece boundary
  round-trips exactly. The joins are nonetheless chosen to fall on
  non-blanks, because it costs nothing, the holdings tree holds names with
  spaces in them, and a name that lost a character in the middle would reach
  the consumer as a file it never asked for.
- **CSV report, one per mission.** Columns: `image_name`, `utc`, `et`,
  `sclk`, `offset_dv`, `offset_du`, `sigma_dv`, `sigma_du`, `confidence`,
  `confidence_rank`, `status`, `status_reason`, `source_bc`,
  `omission_reason`. Every image considered appears exactly once:
  `source_bc` names the file carrying its segment, or is empty with
  `omission_reason` set. Sources, precisely: `offset` (top level, `[dv,
  du]`, unrounded) for the offset pair; `sigma_px` from
  `navigation_result` (rounded there, and reported as recorded);
  `confidence` top level; `confidence_rank` and `status_reason` from
  `navigation_result`; `status` top level; `sclk` from
  `times.sclk_midtime`. The column set is expected to evolve; treat it as
  version 1.

### 3.5 The program

`sd_create_ck`, a new dispatch module in `src/spindoctor/cli/`, entry point
in `pyproject.toml`, arguments: mission, time range, input roots, output
directory, and the shared logging surface. Add `SD_CREATE_CK =
'sd_create_ck'` to `spindoctor/config/program_names.py` and
`PROGRAM_NAMES`; declare `PROGRAM_NAME`; call `add_logging_arguments(
parser)` (image flags on -- it processes images individually); wrap the
configuration load in `reporting_logging_errors()` with the literal
adjacency the logging test suite asserts. No entry is needed under
`logging.programs` in `config_015_logging.yaml` (it ships empty and keys
are optional). Anything that degrades or omits a result goes to both logs:
per-image detail to the image log, one line plus a count to the run log.
The per-image logs go under a `ck` backend, which is added to
`BACKEND_NAMES` beside `nav`, `backplanes` and `reproj`.

The arguments as built: a positional mission (`coiss`, `gossi`, `nhlorri`,
`vgiss`, spelled in the driver rather than read from the observation
registry, which is behind an oops import); `--nav-results-root`, whose tree
is walked for `*_metadata.json` and filtered on `observation.instrument`;
`--kernel-dir`, repeatable and required, which is both the set of
directories indexed for candidate C-kernels and the set that resolves the
basenames provenance records; `--start-time` / `--stop-time` as UTC, applied
to the recorded exposure midtime, with an image that recorded none -- or
recorded a non-finite one -- ignored whenever either bound is given, since
it cannot be placed in time; and `--output-dir`.

**Some failures deliberately stop the run rather than being reported as an
omission**, because the omission-reason set is closed and none of them has an
entry in it: a metadata document that cannot be read as a navigated image at
all, a baseline supplying angular velocity at only some of an exposure's
records, and a window too long for a segment's records. Each names the image in
**the run log only** -- not in both, which the "anything that degrades or omits
a result" rule above would otherwise ask for. That rule is about a result the
run goes on to report; these end the run, and the per-image log they would be
written to is opened by the reporting pass that then never runs. Logging
through the image logger with no image scope open is a defect rather than a
fallback, and under `logging.strict_scope` the `LogScopeError` it raises
would *replace* the failure worth reading and demote it to a `__context__`
nobody prints -- so a documented configuration setting would turn a
diagnosable failure into a logging crash.

**A run stopped for any reason it can know in advance writes nothing at all.**
Every segment of every output file is built, and every destination judged,
before the first file is opened, so such a refusal leaves no corrected kernels,
no meta-kernel and no report -- rather than a partial set with no report saying
what is in it, which the refusal to overwrite an existing corrected kernel
would then block a rerun on. Judging the destinations is a second, separate
thing from separating the phases: whether an output path can be written is a
property of the output directory that the build never looks at, so a run whose
second output path was occupied would otherwise build cleanly and then refuse
with the first file already on disk. `check_output_paths` judges the whole set
at once and names every path that failed; `check_ck_file` judges one file's
contents and is what `write_ck_file` itself calls, so the per-file refusal
cannot drift from the set-level one. The cost is that every segment of the run
is resident at once: measured on this writer's own segments, 678 bytes for the
three records an ordinary exposure carries and 1826 for the 21 a twenty-second
one does, so about 34 MB for the 50,000-image batch this tool is sized for,
against the per-image metadata such a batch already holds.

**That is a bounded guarantee, not atomicity, and it is not written down as
one.** What stays possible is what no check made beforehand can see: the device
filling up, a path or a permission changing between the check and the write,
and a record set `ckw03` refuses once the file is open. The file being written
is removed; the files written before it are not, and the meta-kernel and the
report are never reached. The user guide names that as the one failure that
needs the output directory cleared before a rerun.

A metadata *file* that is not readable as JSON is different again: it names
no image, so there is nothing for the report to say about it and nothing an
omission reason could attach to. It is counted and named in the run log, the
run continues on what it could read, and **the run exits non-zero**, so a
batch wrapper can tell a clean run from one that silently skipped its input.

**The output directory and the kernel directories are resolved to absolute
paths** before anything is written. The meta-kernel names the kernels it
furnishes by these paths and SPICE resolves a relative name against the
*consumer's* working directory, so a meta-kernel written from a relative
`--output-dir` would work only from the directory that generated it, and
elsewhere would fail on the first correction -- after the originals had
already loaded, leaving the consumer with an uncorrected pool rather than an
empty one.

### 3.6 The writer's imports, and cspyce

The writer core lives in `spindoctor/cli/ck/` and imports `cspyce` and
nothing from oops -- and nothing from `spindoctor.support`, which is where
the oops-importing `cmatrix` module lives; one careless helper import there
would drag oops in transitively. The guarantee is asserted on `sys.modules`
after importing the writer package in a fresh interpreter, not by scanning
source text.

The one fact the writer and the attitude computation must agree on -- which
spacecraft clock each CK object's time tags are encoded against -- therefore
lives in `spindoctor/spice_ids.py`, a top-level constants module importing
only the standard library, which both sides read. It is deliberately not
under `spindoctor/support/`, which the writer may not import at all. That
mapping is the check against `ckmeta` computing a clock id rather than
validating one, so a second copy of it would be a silent way for the check
to rot on one side while it kept passing on the other; `cmatrix` derives
each instrument's clock from it, and the writer's `resolve_sclk_id`
validates against it. Both keep their own error type and message.

The same rule covers the widest snapped lookup tolerance (80000 ticks),
for the same reason: the index widens a frozen-attitude object's coverage
by exactly what that lookup reaches, so the two are one declared constant
read twice rather than two constants that agree today. They agree to
within half a tick at the extreme edge, because the filter measures from
the exposure midtime and the lookup from that midtime rounded to a whole
tick; an image that far from any pointing record is refused rather than
corrected, which is the safe direction.

The `cspyce` surface the writer needs, all present in `cspyce` >= 2.3.5
(the pinned minimum):
`furnsh`, `unload`, `kclear`, `pxform`, `frmnam`, `namfrm`, `ckmeta`,
`sce2c`, `sce2t`, `sce2s`, `ckobj`, `ckcov`, `ckgp`, `ckgpav`, `ktotal`,
`m2q`, `ckopn`, `ckw03`, `ckcls`, `dafopw`, `dafac`, `dafcls`, `dafopr`,
`dafec`. One
global to respect: `cspyce.use_errors()` / `use_flags()` is process-wide
and shared with oops; the writer assumes the exceptions regime
(`use_errors`, the package default) and must never flip it. `ktotal('CK')`
is how the assignment step refuses to run with a stray C-kernel furnished,
which would answer the reproduction lookups alongside the candidate under
test.

---

## 4. What the phases delivered

All six phases are implemented and merged. Each subsection below records
what that phase delivered, the tests that guard it, and the measurements it
produced; the measurements are the reason this section is kept rather than
dropped, since several of them bound what the product may claim and would
otherwise have to be made again.

### Phase A — C-matrix in the metadata

`spindoctor/support/cmatrix.py` per section 2.2; the `NavResult.pointing`
field and the `navigate` / `run_manual_nav` wiring per section 2.4; curator
serialization of `pointing`, `times`, and `observation.shutter_mode` per
section 2.3.

Hermetic tests (synthetic FOV and frames):

- A planted offset produces a `cmatrix` whose correction, inverted,
  recovers that offset; the test fails if the sign of `xy_offset` flips.
- A zero offset produces `cmatrix == cmatrix_original` exactly, through
  the `M = I` guard. **This holds only for a FOV whose boresight pixel maps
  to `xy` exactly `(0, 0)`**, which a synthetic `FlatFOV` does, and which
  Galileo SSI and Voyager ISS also do. It does not hold on a real
  `PolynomialFOV`: `oops.fov.OffsetFOV(fov, uv_offset=(0, 0))` is itself
  not the identity there, because its `xy_offset` is `fov.xy_from_uv(
  fov.uv_los)` rather than zero. Measured `|xy_from_uv(uv_los)|`, which is
  exactly the residual correction angle a zero offset produces, each
  converted at its own frame's pixel scale:

  | Frame | scale (rad/px) | residual (rad) | in px |
  |---|---|---|---|
  | Cassini WAC, 1024x1024 | 5.977e-5 | 5.427585e-08 | 9.08e-4 |
  | LORRI, 256x256 (4x4 binned) | 1.986e-5 | 6.842953e-10 | 3.45e-5 |
  | Cassini NAC, 1024x1024 | 5.992e-6 | 1.149029e-10 | 1.92e-5 |
  | Galileo SSI | 1.015e-5 | exactly 0 | 0 |
  | Voyager 2 NAC | 7.842e-6 | exactly 0 | 0 |

  The WAC is the worst case by a factor of ~470 over the NAC and must not
  be left off this list. The LORRI conversion uses the binned scale of the
  actual test frame; quoting it at the unbinned 4.96e-6 rad/px understates
  it by 4x. The non-zero cases are faithful, not a defect -- the recorded
  attitude reproduces what a consumer applying the offset through
  `OffsetFOV` actually sees -- but Phase D must not assume `offset == 0`
  implies `cmatrix == cmatrix_original` on a real frame.
- A small-but-real offset (0.05 px) still produces a non-identity
  correction that recovers that offset, so the `|d x b| < 1e-12` guard is
  pinned from below as well as above.
- The Cassini-style flip: with a synthetic `R = diag(-1,-1,1)` between the
  "oops" and "SPICE" frames, the recorded `cmatrix` reproduces the offset
  in the SPICE convention; the test fails if the `R` conjugation is
  dropped (the un-conjugated result negates both tangent components).
- The conjugation **direction** with a synthetic non-involutory `R` (a
  quarter turn about Z). Every `R` in the section 2.1 table is diagonal
  and therefore its own inverse, so `R^T M R` and `R M R^T` agree on all
  real data and no real-frame test can tell them apart.
- A result carrying a fitted rotation records `cmatrix_original` but no
  `cmatrix`.
- Determinant, orthonormality and finiteness violations raise.
- A measured flip that is not the instrument's constant raises.

Integration tests (real frames, marked `integration`): for one Cassini
NAC, one Cassini WAC, one LORRI, and one Galileo frame, the measured `R`
equals the section 2.1 constant to 1e-9 and is identical at start, mid and
stop; for one Voyager frame, `cmatrix_original` equals the frozen oops
attitude.

Every one of those frames must additionally recover its planted `(dv, du)`
**exactly**, by inverting the recorded `cmatrix` back through the recorded
flip on the instrument's own distorted FOV. A magnitude check -- the
rotation angle of `cmatrix . cmatrix_original^T` against the offset's field
angle -- is **not** sufficient and must not be substituted: a rotation
magnitude is invariant under a sign flip, an `R` conjugation, and a
reversed composition, so such a test passes unchanged against every
directional error this section exists to catch. The recovery tolerance
follows from the step-2 error budget above; 1e-2 px is comfortable at a
~19 px offset, and a directional error is off by twice the offset.

### Phase B — Writer core, one image

The type-3 segment writer for a single image: `F` and `delta` per section
3.1, records, SCLK, quaternion sign continuity, AV policy, the
single-record degenerate path.

Tests are hermetic: the suite writes its own minimal LSK, SCLK and FK text
kernels (small text files `furnsh` accepts) -- the FK because `F` is
`pxform(frmnam(ck_frame_id), camera_frame, mid)` and both of those frames
have to be defined for the call to resolve -- plus an original CK produced
by the writer's own primitives, so no holdings are needed. Write a kernel
from a constructed attitude history and correction, furnish, query back
with `ckgpav` at `tol = 0`:

- Attitude at start, mid, stop and an interior time matches the composed
  truth within 1e-9 radians.
- AV is bit-identical to the original's; the test **fails if AV is rotated
  through `delta`**.
- An AV-less original is refused, and so is an exposure straddling one
  original segment that carries AV and one that does not: a segment has one
  flag for all its records, and one declaring none would be skipped by
  `ckgpav` and `sxform` in favor of another kernel's uncorrected attitude.
- With the original still furnished underneath it, the corrected segment is
  what `ckgp`, `ckgpav` **and** `sxform` all answer with. The `sxform` half
  is the one that catches `avflag = 0`; the `ckgp` half alone does not.
- A sign-discontinuous quaternion sequence is repaired, asserted on the
  written records rather than on a read-back attitude (section 3.3: SPICE
  restores the sign when it interpolates, so the read-back cannot see it);
  interpolated attitude mid-record stays continuous.
- Exposure epochs that collapse to a single encoded tick produce a valid
  single-record segment. Per section 3.3 that needs three epochs equal as
  doubles, which no real exposure produces, so a second test pins the
  reachable neighbour: a sub-tick (1 ms) exposure still produces three
  records.
- The written file's `ckcov` window is exactly `[start_et, stop_et]` in
  encoded SCLK. Asserting on the record array instead cannot see a segment
  descriptor that advertises coverage the records do not have.
- Baseline pointing is read at the record epoch with tolerance zero: an
  exposure the original does not cover is refused rather than corrected
  against the nearest attitude within some tolerance.
- Sub-spacecraft-clock ids come from `ckmeta`, asserted for all four CK
  objects.

### Phase C — Grouping, mirroring, naming, and the coverage rules

The candidate pre-index (`ckobj`/`ckcov`), assignment-by-reproduction with
its tolerance and tie-break, basename resolution, output naming, the
BOTSIM predicate, the Voyager snapped-baseline reproduction, and the
omission-reason set.

Tests: assignment keeps exactly the reproducing candidate and refuses when
none reproduces; the tie-break is deterministic and class-ordered; a BOTSIM
pair yields one segment and a `botsim_loser` row; a fitted-rotation result
yields `rotation_unsupported`; filenames derive from real-shaped original
names.

### Phase D — Round-trip validation

The acceptance test of the whole plan, as an integration test, run as
**three subprocesses** because oops caches frames and manages its own
kernel pool, and a mid-process `furnsh` is not guaranteed to take effect:

1. *Navigate* a real image normally; record offset and `cmatrix`.
2. *Generate* the corrected kernel from that result.
3. *Re-navigate* in a fresh process in which the corrected kernel is
   furnished immediately **after** the host's `from_file` returns (after
   its lazy CK furnishing) and before any geometry is computed. In that
   process, first assert the pointing actually changed: `ckgp` /
   `pxform` at midtime differs from the uncorrected value by the expected
   correction. This distinguishes "kernel took effect" from "kernel was
   silently buried", which the offset alone cannot.
4. Assert the re-navigation's offset is approximately `(0, 0)` and its
   `cmatrix` matches run 1's as a rotation. The comparison is valid only
   when both runs commit the same winning technique set; the test records
   both sets and fails as *inconclusive-mismatch* (not as a pass) when
   they differ.
5. Assert the corrected attitude at **start, midtime and stop only**.
   Those three are the records every segment carries, and they are the
   epochs this plan claims. A segment for an exposure longer than 10 s
   additionally carries records at a 1 s cadence (section 3.3); those are
   reproduced exactly too, but they are deliberately **not** asserted,
   because they exist only on long exposures and asserting them would make
   the validation's coverage depend on the cohort's exposure lengths.
   Interior epochs -- anything between records -- are not asserted either,
   because the record scheme does not bound them (see the limitation
   below); testing them would pin a number this plan does not undertake to
   hold.

Tolerances: the target is at or below **0.1 px per axis**, and the
C-matrix target is that offset's angular equivalent at the instrument's
pixel scale. Decision rule: when the measured residual is at or below
0.1 px, pin the test at measured-plus-margin; when it is above, **stop and
diagnose** -- a larger residual is a defect (most likely a section 2
convention error), never a tolerance to raise. The section 2.2
rotation-vs-shift bound is part of the budget, and on a WAC it very nearly
**is** the budget: 9.89e-2 px of total displacement, worst case across the
frame, at a 50 px total offset, against a 0.1 px per-axis target -- convert
before comparing. Choose the WAC cohort frame accordingly, remembering the
bound is **linear** in the offset, not quadratic: about 10 px of total
offset buys a fifth of budget.

**Interior epochs are outside what this plan claims, and that is a
measured limitation rather than an oversight.** A segment reproduces its
record epochs exactly and interpolates between them, so an epoch inside
the exposure carries the reconstruction error of that interpolation.
Its size is instrument-dependent and is deliberately **not** quoted here as
a single figure. The error is an angle, and the same angle is a different
number of pixels on each camera -- a Cassini WAC pixel subtends about ten
times a NAC pixel -- so a figure measured on one camera does not transfer
to another, and how much baseline rate structure a segment interpolates
across differs by mission and by how the platform was slewing. Several
measurements taken during execution disagreed by factors of two to six
precisely because they used different kernels, sampling and statistics; a
figure quoted without all three is not a bound. Characterizing it per
instrument, with a stated sampling scheme, is #455.

What is invariant is the shape: the error is zero at every record epoch,
grows between them, is largest where the baseline's rate changes inside the
window, shrinks as records are added, and is present at the same size when
the correction is zero -- which is how it is known to be interpolation loss
rather than an error in the correction.

Bounding it is deferred to a denser, adaptive record cadence (#444). Until
that lands, the round trip asserts the three record epochs and nothing
between them, and the user guide states the limitation plainly rather than
implying interior fidelity the kernels do not provide. A consumer that
evaluates geometry at the midtime -- which is what the backplane and
reprojection stages do -- is unaffected and exact; the cost falls only on a
consumer integrating smear across the exposure.

Cohort: one star-navigated Cassini NAC frame (best-constrained truth), one
Cassini WAC frame, and one frame from each other instrument that has a
navigated library frame. This is where acceptance criterion 2's
per-instrument claim is earned. Local-only (needs local binary kernels);
marked `integration`.

**Galileo SSI cannot be in that cohort, and the reason is the plan's own
rule rather than an omission.** `config_410_inst_gossi.yaml` sets
`fit_camera_rotation: true`, and both Galileo library frames that navigate
successfully fit a real rotation -- `C0059894800R` reports -0.432 deg and
`C0059899900R` -0.431 deg -- so by section 1 they record `cmatrix_original`
and no `cmatrix`, and the generator omits them as `rotation_unsupported`.
The library's other six Galileo frames are `negative_cases` that do not
navigate at all. There is therefore no Galileo image anywhere in the corpus
that a corrected kernel can be written for, and the round-trip cohort is
Cassini NAC, Cassini WAC, Voyager and LORRI. Phase D pins that as a
measurement rather than leaving it implied: it navigates a Galileo frame and
asserts the fitted rotation, the absent `cmatrix`, the present
`cmatrix_original` and the `rotation_unsupported` omission, so that on the
day twist support (section 7) lands, the test says so. Acceptance criterion 3 is
consequently claimed for four instruments, not five; criterion 2's Galileo
claim stands on Phase A's planted-offset recovery on a real Galileo frame,
which does not need a `cmatrix` in a result to be measured.

**What the round-trip residual is made of, measured.** The end-to-end
residual has two independent parts, and only the first belongs to this plan:

| Part | Measured over eleven real frames |
|---|---|
| The pointing chain: offset to `cmatrix` to segment to kernel to readback | 0 to 5.6e-17 rad against what the segment says; at most 1.5e-15 rad against the recorded `cmatrix`; the pool's pointing moves by the measured offset to within 1.2e-3 px |
| The navigation re-measuring a nearly-zero offset | 0.0001 to 0.4853 px per axis, depending on which techniques carry the ensemble |

The first is floating-point noise: a corrected kernel gives back the
recorded attitude bit for bit, on every frame tried. The second is what the
0.1 px target actually spends, and it is a property of the navigation
techniques rather than of any instrument or of the size of the offset. A
frame navigated by the star techniques alone lands within 0.017 px of zero
(measured: 0.0001 to 0.0170 px per axis over seven such frames, on all four
instruments, at offsets from 1.86 to 49.2 px). A frame whose ensemble is
carried by the correlation and distance-transform body techniques does not:
`W1637520502_1_CALIB` (Cassini WAC, 1.86 px offset) leaves **0.1022 px** on
`dv`, and `C3446143_GEOMED` (Voyager 1 WA, 28.8 px offset) leaves
**0.4853 px** on `du`. Both are above the target and neither is a pointing
defect -- on those two frames the record epochs read back to 2.6e-17 and
0 rad, the midtime reads back the recorded `cmatrix` to 8.8e-16 and
1.7e-16 rad, and the pool's pointing moves by the measured offset to within
5.7e-5 and 5.1e-5 px. What is left is that those techniques do not return
the shift they were given. Measured as the difference between a technique's
two answers against the correction actually applied: `BodyLimbNav` falls
0.139 px short on `W1637520502_1_CALIB`, and `BodyDiscCorrelateNav` falls
0.504 px short on `C3446143_GEOMED`, where it answers `du` on a 0.25 px grid
(-5.75 px in the first run and -0.5 px in the second) and the ensemble
weights it at 0.73. The techniques were not exactly shift-equivariant, and
re-measuring after a shift did not return exactly the negative of that
shift. The body-technique causes are since fixed under #447 (probe-refined
polyline vertices; the correlator's NCC-quadratic fallback when its
sub-pixel refinement saturates), with the standing measurement in
`util/calibration/shift_equivariance.py`; RingEdgeNav's larger wrong-edge
re-locks remain open as #476.

The decision rule stands as written. What these measurements add is how to
tell its two outcomes apart: a section 2 convention error leaves about
*twice* the original offset (3.7 to 98 px on these frames) **and** a
readback that disagrees, where technique non-equivariance leaves a fraction
of a pixel and a readback that is exact to floating point. The round trip
therefore asserts the chain on a body-navigated frame as well as on the
star-navigated cohort, and pins the end-to-end offset only where the
measurement is a pointing measurement.

### Phase E — Comment area, meta-kernel, report, driver

Section 3.4 and the `sd_create_ck` driver per section 3.5.

Tests: every considered image appears exactly once in the CSV, with either
`source_bc` or `omission_reason`; the comment area reads back via
`dafopr`/`dafec` and names the baseline and SCLK kernels; the meta-kernel
furnishes originals before corrections; the driver passes the shared
logging-surface assertions.

### Phase F — Documentation and reconciliation

`docs/user_guide/user_guide_ck_kernels.rst`: what the kernels are, what they
claim and do not claim -- corrected only inside a navigated exposure with the
originals still required, exact at the record epochs and interpolated
between them, and eligible without any confidence threshold, all three as
sections of their own rather than as footnotes -- loading with and without
the meta-kernel, the naming convention, the report columns, and every
`sd_create_ck` argument.
`docs/dev_guide/dev_guide_ck_kernels.rst`: the frame relations of section
2.1 including the oops-flip table, the derivation of section 2.2, why
angular velocity is copied verbatim, why the writer imports no oops and
nothing from `spindoctor.support`, why assignment is by reproduction, and
the writer's module structure. Both are in the Sphinx toctrees;
`spindoctor.cli.ck` gets no API-reference page, matching every other
`spindoctor.cli` subpackage, and is nitpick-ignored in `docs/conf.py` the
same way.

---

## 5. Acceptance criteria

1. Every navigated image with an offset and no fitted rotation carries
   `cmatrix`, `cmatrix_original`, the frame identities and the `times`
   block; both matrices are proper rotations; a fitted-rotation result
   carries `cmatrix_original` only.
2. The recorded `cmatrix` reproduces the navigated offset (inverting the
   correction recovers `(dv, du)` to a stated sub-pixel tolerance) on real
   frames from each instrument that has a navigated library frame, and the
   measured oops-vs-SPICE `R` matches the section 2.1 table on each.
3. Round-trip: navigate, generate, furnish, re-navigate yields an offset
   at or below the pinned tolerance with a matching C-matrix, the
   pointing-actually-changed assertion passing, and matching technique
   sets between runs.
4. A kernel written by this tool loads in a plain `furnsh` session with no
   SpinDoctor code present; `ckgp`, `ckgpav`, `pxform` and `sxform` all
   return the expected attitude inside a navigated exposure and fall through
   to the original outside it; every written segment carries angular
   velocity for `ckgpav` to return -- the baseline's own vectors, or zeros
   for a frozen-attitude segment, whose constant attitude that is.
5. Every image considered appears exactly once in its mission's CSV, with
   either a `source_bc` or an `omission_reason` from the section 3.3 set.
6. No image whose recorded baseline no candidate kernel reproduces
   receives a segment.
7. Importing the writer package in a fresh interpreter loads no oops
   module and nothing from `spindoctor.support`, asserted on
   `sys.modules`.
8. `ruff check`, `ruff format --check`, `mypy --strict`, `sphinx-build -W`
   and `pymarkdown scan` all pass; suite coverage stays at or above 90%,
   with the writer core covered by the hermetic self-written-kernel tests
   of Phase B, not only by integration runs.

### Status

Criteria 1-7 are met, each by the tests named in section 4: 1 and 2 by
Phase A's hermetic and per-instrument integration tests, 3 by the Phase D
round trip on Cassini NAC, Cassini WAC, Voyager and LORRI (Galileo is
excluded by this plan's own rotation rule, and Phase D pins that as a
measurement), 4 by the Phase B lookup tests over a furnished baseline
including the `sxform` half, 5 and 6 by the Phase C and E report and
assignment tests, and 7 by the fresh-interpreter `sys.modules` probe.

Criterion 8 is met except for its coverage bound: the lint, type, docs and
markdown gates all pass, and the writer core is covered hermetically rather
than only by integration runs, but suite coverage measures 77% against the
90% asked for. Section 0 records what that figure is made of; the bound
stands as written rather than being lowered to what was measured.

---

## 6. Risks and constraints

**Frame-convention errors are the dominant risk, and hermetic tests cannot
see the worst one.** The oops-vs-SPICE flip (section 2.1) only bites when
real host frames meet real kernels, which is exactly what unit tests mock
away. That is why Phase A carries per-instrument integration assertions on
`R` and why Phase D exists. The sign conventions that *are* hermetically
testable (`xy_offset`, `axisar` vs `rotate`, composition order) each have
a test that fails when flipped.

**The AV trap points the other way than intuition.** The plausible-looking
"thorough" treatment -- rotating AV through the correction -- is the
wrong-frame error, because CK AV lives in the base frame and the
correction is a constant rigid rotation of the body. The correct treatment
is verbatim copy. Phase B pins this in the correct direction.

**One bus attitude cannot serve two cameras.** The BOTSIM rule manages the
case that exists today. If per-camera conflicts matter more broadly, the
answer is corrected camera frames in a supplemental FK -- a larger change
with an adoption cost, not in this plan.

**Segment count per file.** Thousands of segments make CK lookups a linear
scan. Existing NH kernels carry hundreds; measure the largest Cassini file
produced and report the number in the PR.

**Consumers need the originals.** An overlay claims pointing only inside
navigated exposures; a user loading only the corrections gets nothing
between images. Prominent in the user guide.

**The correction is a midtime snapshot held body-fixed across the
exposure** (constant outright for Voyager, matching what navigation
assumed). It is an approximation for an exposure during which the pointing
error itself changed; nothing available today measures that.

---

## 7. Follow-ups

Each is filed as a tracking issue; the number is given with the item. None
blocks use of the kernels.

- **Consumers switch from the offset to the C-matrix** -- delivered.
  `plans/CMATRIX_READERS_PLAN.md` is the design of record: the backplane
  and reprojection readers apply the recorded `cmatrix` by frame
  replacement (`apply_cmatrix_to_obs` in `spindoctor/support/cmatrix.py`),
  with the offset as the documented fallback for records that carry no
  usable C-matrix. This was the reading half of #50; this plan delivered
  the writing half.
- **Replace the C-matrix derivation with the oops API** when oops gains
  one (#433): `spindoctor/support/cmatrix.py` keeps its interface, its body
  goes.
- **Fitted-twist support** (#434): record the rotation pivot
  (`rotation_pivot_vu`) on the technique result and through the ensemble,
  define the boresight-referenced conversion -- including the sign flip
  from the `(v, u)`-to-`(x, y)` axis swap, under which a positive
  image-plane rotation is a negative rotation about camera +Z -- and lift
  the `rotation_unsupported` omission. Costs only Galileo today.
- **Static per-instrument twist** (#435) belongs in an FK/IK correction
  rather than per-image CK records; the measured LORRI and SSI values are
  candidates.
- **Adapt the CK writer when corrected instrument kernels exist** (#436). If
  FK/IK kernels encoding the measured static rotations are produced (the
  previous item), the writer's world changes: `F` and the camera frame
  definitions come from the new kernels, corrections shrink by the static
  twist the FK now carries, and every previously-generated overlay kernel
  is stale against the new baseline. The issue covers deciding whether the
  adaptation is feasible per mission (an FK change alters what
  `cmatrix_original` reproduction means, so recorded metadata may need
  regeneration first), and regenerating the overlay set against the new
  kernels.
- **SPICE database registration** for oops kernel selection (#437), if
  wanted before that database is replaced: rows and load priority for the
  corrected kernels, plus the per-mission furnish-policy story. The
  meta-kernel path works without it.
- **Bound the interior-epoch error** (#440) by choosing the record cadence
  adaptively rather than at a fixed one second (#444), with the error
  characterized per instrument first (#455). Until that lands, only the
  record epochs are claimed, which is what the user guide says.
- **Kernel-input handling**: decide whether the merged New Horizons
  pointing family should be declared reconstructed on its comment-area
  evidence rather than staying `UNCLASSIFIED` (#468) -- the scan itself
  indexes all 32 New Horizons files, skipping and reporting the one
  object it cannot clock -- locate C-kernel inputs through `spyceman`
  instead of a kernel directory tree (#448), and cover the remote
  kernel-index path, which no test exercises (#446).

Broader than this plan and blocking nothing: whether `spindoctor.cli`
subpackages belong in the API reference at all (#443), which is what
decides whether the writer package gets an autodoc page rather than the
nitpick-ignore every other `spindoctor.cli` subpackage has. The
documentation chapter specifying the metadata JSON format -- including the
`pointing` and `times` blocks this plan added -- is written
(docs/user_guide/user_guide_metadata.rst).
