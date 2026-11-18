# Simulator Code-Structure Critique

**Date:** 2026-11-16

**Baseline:** `main` @ `5207f7b7` ("Draw each simulated crater from its own random stream (#998)")

**Question:** Looking only at code structure (organization, duplication of effort, logical
structure, function and file naming, and file length), where does the image simulator deviate
from what a careful original design would have produced, as opposed to code that accreted over
time? Physics correctness, realism, test coverage, and performance are out of scope except where
a structural problem causes them.

**Method:** Read in place on the baseline. Scope: `src/spindoctor/sim/` (63 modules, 24,447
lines of Python, plus 5 instrument YAML files), `src/spindoctor/cli/sim_editor/` (25 modules,
10,076 lines) with its entry point `cli/sd_create_simulated_image.py`, and the navigator-side glue
(`nav_model/nav_model_{body,rings,titan}_simulated.py`, `nav_model/sim_body.py`,
`nav_model/sim_ring.py`, `nav_model/stars/nav_model_stars_simulated.py`, `obs/obs_inst_sim.py`,
`dataset/dataset_sim.py`, `config_files/config_440_sim.yaml`; 2,923 lines of Python in eight
modules, plus the 60-line YAML). Surveyed first with
tools: an `ast` script listing every module's top-level functions and classes with line counts,
an import graph built from every `import` statement including function-level ones, an exact and
a name-normalized AST hash of every function body, a sliding 7-statement window hash for repeated
code blocks, grep for repeated idioms (seeding, `params.get(key, default)`, deep merges, light
direction, phase functions, key inventories, private-name imports), and `git log` for the commits
that created split modules. Then read in depth where the survey pointed. Every claim below was
checked against the source; `path:line` citations are to the baseline. The stated design was
taken from `docs/dev_guide/dev_guide_simulator.rst`, `CLAUDE.md`, and `.cursor/rules/python.mdc`.
Drafted without reading the earlier realism critique or the archived audits; the closing note
compares afterward. A second agent, which wrote none of the draft, then reviewed it
adversarially: it recounted every count, rechecked every citation, and looked for a design reason
behind each claimed duplication. Its corrections are incorporated (the D3 duplicate was
reframed, O7 narrowed, N4 downgraded, the layout mapping fixed, and several counts corrected).

This is a frozen snapshot of the code at the baseline. It is not updated as the code changes;
dispositions belong in plans and issues.

---

## Verdict

The simulator has a well-designed spine: a fixed five-stage pipeline over a `SimFrame`, stable
per-stage seeding, a declarative artifact-mode registry, a declarative instrument catalog with
mission-named YAML files, and an information boundary enforced by a default-deny filter and an
import-time classification check. Around that spine, nearly everything else shows accretion. The
largest structural gap is that there is no scene model: the scene stays a raw dict that more than
a dozen modules read with their own literal defaults, its key inventory is written out in up to
six places in three different schema languages, and the GUI peer has silently fallen behind (it
has no control for 16 of the 23 `noise` keys). The renderers repeat that pattern at smaller
scale: two image-side ellipsoid renderers that do not share a projection, a body seed chain
written out in full twice and split across callers elsewhere, three dispatch mechanisms for one
artifact-mode registry, and an oversampling factor threaded by hand through 14 modules. One module is over the 1000-line limit and eleven more sit
between 800 and 1000, several of them split at the limit rather than at a seam.

---

## Scope at a glance

| Area | Modules | Lines | Largest module |
| --- | ---: | ---: | --- |
| Scene schema and validation (`scene*.py`) | 7 | 3,504 | `scene_checks.py` 866 |
| Instrument catalog (`instrument_*.py`, `artifact_modes.py`) | 6 | 3,311 | `instrument_schema.py` 669 |
| Shared geometry (`*_geometry.py`, `star_records.py`, `spherical_harmonics.py`) | 5 | 1,968 | `mesh_geometry.py` 646 |
| Entry points and utilities (`render.py`, `png_export.py`, `seeds.py`) | 3 | 421 | `render.py` 208 |
| Forward model (`forward/`, `forward/detector/`) | 35 | 13,403 | `forward/body.py` 974 |
| Realism measurement (`realism/`) | 6 | 1,839 | `realism/noise.py` 984 |
| Scene editor (`cli/sim_editor/`) | 25 | 10,076 | `optics_tab.py` 962 |
| Navigator-side glue | 8 | 2,923 | `nav_model_body_simulated.py` 1,084 |

---

## Findings

Severity says how much a careful original design would differ: **High** means the planned
version would have a different shape (a missing layer, a missing abstraction, a second source of
truth); **Medium** means the same shape with a different seam or a removed copy; **Low** is local
tidying. Within each theme the findings are in descending severity.

### 1. Organization and layering

#### O1 (High). There is no scene model; every consumer re-reads the raw dict with its own defaults

The scene module says a validated scene file "IS the `sim_params` mapping with no translation layer"
(`sim/scene.py:10-11`). That choice is the root of most of the duplication below. Only one block got
a parse-once layer, and late: `forward/body_spec.py` (created in #959, 2026-11-11) turns a body
entry into a frozen `BodySpec` "so the renderer reads typed fields and never meets a null"
(`forward/body_spec.py:1-8`). Everything else is read as a dict at the point of use:

- `random_seed` is read with the literal default `42` in seven places
  (`forward/pipeline.py:49`, `forward/scene_radiance.py:163`, `forward/telemetry.py:106`,
  `forward/distortion.py:313`, `forward/detector/params.py:727` and `:800`,
  `cli/sim_editor/scene_io.py:110`), although the validator requires the key
  (`sim/scene.py:330`), so the default is dead in every one of them.
  The `DetectorParams.random_seed = 42` dataclass default (`forward/detector/params.py:197`) is an
  eighth.
- `exposure_sec` is read with `1.0` five times in four modules, `time` with `0.0` in five
  modules, `ring_epoch` with `0.0` in four, and `size_v`/`size_u` with `512` in three editor
  modules.
- `resolve_detector_params` (333 lines) runs twice per render, at `forward/detector/chain.py:727`
  and again at `forward/telemetry.py:119`; nothing passes its result along on the `SimFrame`.
- Ring features have no spec at all: `scene_value(feature, 'width_px', 0.0)` and
  `scene_value(feature, 'side', RING_SIDE_DEFAULTS[...])` are repeated per kind in
  `forward/ring_system.py:695-708` and `:799-831` and again in `nav_model/sim_ring.py:275-310`.
- The scene's body geometry is parsed three times with different defaults, once on the image
  side and twice on the navigator side: `BodySpec` on the image side defaults an unset center to
  the frame center; `NavModelBodySimulated._render_body_image` also uses the frame center
  (`nav_model/nav_model_body_simulated.py:438-442`); `nav_model_titan_simulated._render_silhouette`
  uses `0.0` (`nav_model/nav_model_titan_simulated.py:108`). Each also re-derives
  `semi_c_px = min(semi_a_px, semi_b_px)` by hand. `_render_silhouette` also ignores
  `shape_model`, so a mesh occluder is predicted as an ellipsoid there but as a mesh by
  `NavModelBodySimulated._render_body_image` (`:443`).
- `BodySpec` itself is shadowed by `TopoBodySpec` (`forward/body_topo.py:84-153`), which repeats
  22 of its fields with literal defaults (`0.05`, `0.25`, `3.0`, `0.6`, `15.0`, `'lambert'`,
  `0.5`, `6.0`) that `body_spec.py:39-54` names as constants, and with a different
  `anti_aliasing` default (`0.0` against `BodySpec`'s `1.0`). Those defaults are reached only by
  tests: the one construction in `src` passes every field, and `TopoBodySpec` is a
  units-converted record (radians, render-grid pixels, derived seeds) rather than a second
  scene parse. Its field list is still a hand copy. `render_single_body` then copies
  `BodySpec` fields into about 20 locals (`forward/body.py:643-667`) and passes them into
  `TopoBodySpec(...)` one by one (`:762-788`).

A planned design would parse and validate a scene once into a typed `Scene` (top level, bodies,
rings, stars, optics, noise and detector, artifacts), with every default applied in one place;
stages, the navigator-side filter, and the editor would read that object. `BodySpec` shows the
pattern already works here; it was applied to one block of the scene. The navigator side gets its
own typed view: an idealized-only type built by `build_nav_params` after the `nav_override`
overlay, never the full `Scene`, so the default-deny filter and the boundary tests keep covering
every field the navigator can reach (`dev_guide_simulator.rst:222-249`).

#### O2 (High). Image-side detector truth comes from two configuration trees

The forward model reads physical detector values from the navigator's configuration, not only
from the simulator's own instrument catalog. `resolve_detector_params` takes the ADC ceiling and
DN full well from the navigator block (`forward/detector/params.py:545-546`:
`inst_noise.get('saturation_dn', sim_noise['saturation_dn'])`), the signal scale from the
navigator block or the navigator's `sim` block (`:538-539`), and falls back to the `sim` block for
`bias_dn`, `bloom_length`, and the cosmic-ray rate (`:543`, `:592`, `:601`). The telemetry stage
reads the ceiling and the missing-data rate the same way (`forward/telemetry.py:184-185`,
`:243-248`). Meanwhile the electron-side values (full well in electrons, gain table, read noise)
come from `sim/instrument_configs/*.yaml`. The two trees also repeat values: `generic.yaml` sets
`bias_dn: 20.0` and `star_flux.e_per_s_vmag0: 1.3e+4`, and `config_440_sim.yaml` sets
`bias_dn: 20.0` and `star_flux_dn_per_s_vmag0: 1.3e4`. Four forward modules import
`sim.instrument_navigator` (`scene_radiance`, `optics`, `telemetry`, `detector/params`) to do this.

The strongest consequence: `saturation_dn` and `full_well_dn` are not scene `noise` keys
(`scene_checks.py:286-322`), so the only way to change the image's ceiling is
`instrument_config`, an idealized key the navigator also reads. A scene therefore cannot plant a
mismatch between the true ceiling and the ceiling the navigator believes.

The dev guide says the `instrument` field selects "which instrument config entry ... and which
navigator configuration block drive the detector model and PSF", so this is documented, but it gives
one camera two homes for its physics. A planned design would make the simulator's instrument entry
the only image-side source of physical values and consult the navigator block only where a scene
deliberately asks for the navigator's belief (`match_navigator`, the marker value, the extended-FOV
margin).

#### O3 (Medium). The schema layer depends on the renderer, not the reverse

Scene key inventories and defaults live in renderer modules, so the validator and the editor
import `forward/`:

- `sim/scene_checks_body.py` imports `CRATER_*_DEFAULT` from `forward/body_spec.py`,
  `HAZE_STRUCTURE_KEYS` and `CLOUD_BLOB_KEYS` from `forward/haze_structure.py:148,165`,
  `PHOTOMETRIC_LAWS` from `forward/photometry.py`, and relief limits from `forward/relief.py`.
- The editor imports `forward/body_spec`, `body_texture`, `photometry`, `relief`, `atmosphere`,
  and `star` for defaults (`body_appearance.py:40-46`, `body_atmosphere.py:24`,
  `body_tab.py:39`, `background_stars.py:25`).
- Key inventories are spread over five modules: `scene_schema.py` (top level, bodies, stars, ring
  system and features), `scene_checks.py` (16 sets: optics, PSF, smear, distortion, ghosts, stray
  light, noise, vidicon, detector, sky counts, spk error, expected, companion),
  `scene_checks_body.py` (9 sets including `ATMOSPHERE_KEYS`), `scene_checks_ring.py` (11 sets),
  and `forward/haze_structure.py` (2 sets). `scene_schema.py:3` nonetheless says "Every top-level
  and per-object key a scene may carry is inventoried here".

At the same time, eleven forward modules import `sim.scene_schema`, so the dependency runs
both ways between the two layers (no module-level cycle, but no layer either). In a planned
design the schema owns every key, type, range, and default; the renderer imports it; nothing in
the schema imports the renderer.

#### O4 (Medium). Body-renderer import cycles are hidden by function-level imports

`forward/body.py` imports `body_topo`, which needs `body.carve_crater_heights` and
`body.perturb_normals_by_height` and imports them inside functions "to keep the import graph
acyclic" (`forward/body_topo.py:474-476`, `:735-738`). `body.py` dispatches to
`body_mesh.render_single_mesh_body` with another function-level import
(`forward/body.py:695-697`) because `body_mesh` imports `body.finish_single_body`
(`forward/body_mesh.py:33`). `body.py` therefore holds four unrelated things: the classic
ellipsoid renderer, the crater carver both ellipsoid paths share, the per-body dispatcher, and the
placement finisher. Separately, `render.clear_render_caches` reaches into five private caches in
three modules by attribute (`render.py:57-66`: `star` twice, `body` twice, `body_mesh` once);
`relief` already exposes `clear_relief_caches()`, the pattern each module should follow. A
planned design has `craters.py`, a placement module, renderer modules that import only those, and
a dispatcher on top; each cached module would expose its own clear function, as `relief` does.

#### O5 (Medium). The top level of `sim/` is a flat mix of four concerns

`sim/` holds 21 modules side by side: seven scene-schema and validation modules, six
instrument-catalog modules, five shared geometry helpers, and three entry points or utilities.
There is a real rule behind which modules sit outside `forward/` (the dev guide's "shared
geometry helpers" that both sides may call, `dev_guide_simulator.rst:280-301`), and the
`*_geometry` suffix is its documented marker (`forward/__init__.py:10-12`). The marker is
imperfect: `png_export.py`, `seeds.py`, and the validators sit next to the shared helpers,
`star_records.py` and `spherical_harmonics.py` are shared but do not carry the suffix, and the
`*_geometry` modules hold rendering: `mesh_geometry.py` holds the polyhedral
rasterizer (`_rasterize_faces`, 159 lines, `:281`) and a full renderer (`render_mesh_body_image`,
`:603`), and `ellipsoid_geometry.py` holds Lambert shading (`lambert_from_normals`, `:319`).
`forward/` is equally flat: 27 modules, ten of them body rendering (`body`, `body_topo`,
`body_mesh`, `body_spec`, `body_texture`, `relief`, `photometry`, `atmosphere`,
`haze_structure`, `translate`). `sim/__init__.py` is one line describing the package as "common
utilities for simulated bodies and rendering"; neither it nor `forward/__init__.py` declares an
`__all__`, against
`.cursor/rules/python.mdc` section 3. A planned layout is sketched in section 6.

#### O6 (Medium). `instrument_navigator.py` is a grab-bag with a copy of the navigator's classifier

The module resolves the navigator block for a scene (`resolve_sim_inst_config`, `:95`), builds the
navigator-matched PSF (`:203`), the extended-FOV margin (`:232`), and the marker value (`:272`),
and then re-implements the navigator's image classifier (`read_frame_as_navigator`,
`:328-388`), copying its thresholds as module constants (`_DEFAULT_MAX_MISSING_FRAC = 0.30`, `:68`)
from `nav_orchestrator/image_classifier.py:63` and its decision order by hand, to produce the
editor's status-bar sentence (`frame_units_summary`, `:415-459`). `has_planted_offset` (`:402`)
is an editor helper too. A planned design calls the navigator's own classifier, keeps
scene-to-navigator config resolution in an instrument module, and puts status-bar text in the
editor.

#### O7 (Medium). A test-only estimator module lives in the renderer package

`forward/artifact_counts.py` (228 lines) holds `planted_artifact_counts` and five `measured_*` image
estimators. It is not a stage. Its docstring says it serves "the realism match" (`:3-4`), but
nothing under `src/` or `tests/integration/` imports it; its only importers are two unit tests
(`tests/spindoctor/sim/test_sim_artifact_counts.py` and `test_sim_telemetry.py`). By contrast,
`sim/realism/` is already a measurement package outside `forward/`, and its docstring gives the
reason it lives under `src/` (pure functions unit-tested on synthetic distributions,
`realism/__init__.py:20-23`). Its line-loss estimator
(`realism/artifact_rates.measure_artifact_rates`, `:64-127`, heuristics for real archive frames)
and `artifact_counts.measured_missing_lines` (`:116`, exact marker runs on simulated frames)
measure line loss on different inputs and should stay distinct. A planned design keeps the
simulated-frame estimators beside `realism/` or with the tests that use them, not in `forward/`.

#### O8 (Medium). Private names cross module boundaries 57 times

`sim/scene.py` imports 20 private `_check_*`/`_require_*` functions from `scene_checks.py`, four
more from the body, ring, and artifact checkers, and four private inventories from `scene_schema.py`
(`sim/scene.py:46-89`); the three `scene_checks_*` modules import 18 private primitives from
`scene_checks.py` and four private inventories from `scene_schema.py`. Counting attribute reach-ins
(five in `render.py`) and the two `nav_model_stars` helpers, 57 private names are used outside
their module, 50 of them the validator's.
The modules that hold them are not themselves private (`scene_checks_ring` is imported by tests).
Either the primitives are a shared API (public names, one module) or the checkers are one private
package; the current form is a single validator split across files after the fact (see F3).

#### O9 (Low). The editor is a 10,000-line PyQt package under `cli/`

`CLAUDE.md` says PyQt6 widgets live in `src/spindoctor/ui/`, and most GUIs follow that
(`ui/mosaic_viewer/`, `ui/manual_nav_dialog.py`; `cli/sd_mosaic_display.py` is a thin entry), but
not all: `cli/sd_backplane_viewer.py` is a 1,519-line PyQt dialog under `cli/`. The editor follows
that second precedent, and `CLAUDE.md`'s "CLI programs and their CLI-only helpers" under
`spindoctor.cli` is a defensible reading for it. A planned layout would still pick one rule and
put the editor at `ui/sim_editor/` with `cli/sd_create_simulated_image.py` as the dispatch shim.

### 2. Duplication

#### D1 (High). The scene key inventory is written out in up to six places

For a body key such as `crater_relief_scale`, the name and its meaning appear in:
`scene_schema._BODY_TRUTH_KEYS` (`scene_schema.py:188`), the type list in
`scene_checks_body._check_body_object` (`:103-121`), `BodySpec` and `body_spec_from_params`
(`body_spec.py:58-231`), `TopoBodySpec` (`body_topo.py:84-153`), `create_simulated_body`'s and
`_render_body_shape_cached`'s parameter lists (`body.py:45-63`, `:534-551`), and the editor's
control and handler. At the top level, `_ALLOWED_KEYS` (`scene_schema.py:44-74`) is a
hand-maintained copy of the union of the three classification sets (verified equal), and the
import-time assertion `_assert_boundary_classification_complete` (`:573`) exists to catch the two
copies drifting. The editor adds its own list of 14 pass-through keys
(`cli/sim_editor/scene_io.py:123-138`).

The consequence is visible in the GUI peer. The dev guide says "Every parameter is reachable from
all three" (`dev_guide_simulator.rst:36`) and "The GUI exposes the full scene parameter surface"
with two named exceptions (`:4334`). The editor has no control for 16 of the 23 `noise` keys
(every dark, hot-pixel, banding, bias-structure, calibration-residual, cosmic-ray shape, and
`vidicon` key; no module under `cli/sim_editor/` reads or writes any of them). It passes the block
through on load, which is why the round-trip test still passes. (The wider control gap is filed
as #902.) The editor's own defaults have
drifted too: with no `noise` block, `noise.py:43,51` and `scene_io.py:181,184` show Poisson
checked and read noise 4.0 DN, while the renderer draws neither
(`forward/detector/params.py:570-584`: read noise 0 and Poisson off unless `instrument_defaults`
is on).

A planned design has one field table (key, block, type, range, default, boundary class,
description, editor hint) from which the inventory, the classification, the type checks, the
typed scene model, and most editor controls are derived.

#### D2 (High). Three schema languages, one of them interpreted twice

The package validates structured data in three unrelated styles:

1. The instrument catalog uses declarative nodes, `Scalar`/`MapOf`/`ListOf`/`Record`, walked by
   one `check_node` (`instrument_schema.py:131-206`, `:463`).
2. The scene uses frozensets of names plus hand-written functions such as
   `_check_optional_nonnegative_number` (`scene_checks.py:765-866` holds 12 of them).
3. Artifact-mode parameters use `ModeParam.kind` strings (`'nonneg_number'`, `'unit_interval'`,
   ...; `artifact_modes.py:81-100`), and two separate interpreters type-check them:
   `scene_checks_artifacts._check_mode_param` (`:175-204`, for scenes) and
   `instrument_schema._param_ok` (`:586-605`, for instrument entries).

The same detector parameters are then inventoried in two of these languages. The scene `noise`
block repeats 18 keys of the instrument schema's `_DETECTOR_FIELDS` plus the whole `vidicon`
sub-block (verified by set comparison; `scene_checks.py:286-322` against
`instrument_schema.py:254-283`), and the scene `detector` block repeats three more. The copies
already disagree on ranges: the instrument schema bounds `hot_pixel_fraction` and
`hot_pixel_column_factor` to [0, 1] and requires `banding_period_px` and
`vidicon.coherent_period_px` to be positive, while the scene validator accepts any non-negative
number for all four. `noise: {hot_pixel_fraction: 5}`, `hot_pixel_column_factor: 3`,
`banding_period_px: 0`, and `vidicon: {coherent_period_px: 0}` all validate and render in a scene
but would be refused in an instrument entry. A third copy of the same field list is the
`DetectorParams` dataclass with its 333-line hand merge (L2), and `instrument_checks._CCD_REQUIRED`
(`:38-55`) restates which of them a CCD entry must carry.

A planned design picks one node vocabulary (the instrument schema's is the best of the three),
uses it for scenes and mode parameters too, and has the scene's detector-noise block reuse the
instrument's detector field nodes.

#### D3 (High). Two ellipsoid projections and a split body seed chain

The image side has two ellipsoid renderers: the classic `create_simulated_body`
(`forward/body.py:45-209`) and `body_topo.render_topographic_body` (`forward/body_topo.py:305`),
which handles every other case. The split exists to preserve old output, as the dispatcher says:
"The classic smooth-Lambert path stays byte-for-byte for the scenes it has always rendered"
(`forward/body.py:722-733`). The two do not share a projection. The classic path and the
navigator's predicted body (`nav_model/sim_body.py:76`) both project through
`ellipsoid_geometry.project_ellipsoid` (`ellipsoid_geometry.py:87-212`). The topographic path
builds its own `_Grid` (`forward/body_topo.py:156-229`), a hand copy of the same corner offset,
rotation, tilt compression, and radial function `e2`, whose docstring says it is written to
"match the classic renderer exactly" (`:194`). Every oversampled, relief, textured, or non-Lambert
body is therefore drawn through a private copy of the projection, while the navigator predicts it
through the shared one. The dev guide's rule is that conventions both sides must share go in the
shared helpers, because "any delta between them would land as an unknown systematic"
(`dev_guide_simulator.rst:280-301`); this is the one duplication in the body path that can plant
an unmeasured systematic. The mesh path already does it right
(`mesh_geometry.render_mesh_body_image`, `:603`, called by both sides).

The navigator-side `sim_body.create_simulated_body` is not a renderer copy: it is a deliberate,
roughly 30-line composition of the shared helpers (`project_ellipsoid`,
`ellipsoid_image_normals`, `lambert_from_normals`), documented as "the navigator's *best model*"
(`nav_model/sim_body.py:1-10`). What is duplicated there is the composition itself (normals,
Lambert, 2x2 box downsample), which a shared `render_lambert_ellipsoid` would absorb.

The per-body identity seed (scene seed mixed with body index and name, else the explicit body
seed, else a geometry digest) is written out in full twice (`forward/body.py:744-757` and
`forward/body_mesh.py:119-142`, whose docstring accurately says it "mirrors the ellipsoid path's
chain exactly"). The crater variant is split between the caller (`body.py:675-683`, which has no
geometry digest) and a geometry fallback repeated in both ellipsoid renderers (`body.py:138-148`,
`body_topo.py:744-751`). The fallbacks hash different keys (`center` and `grid.center` against
`body_index, body_name`), and committed seed-less scenes depend on each key (#913), so one
`body_identity_seed()` must take the fallback key as a parameter or reseed those scenes. A
planned design has one projection type both ellipsoid paths use, one shared
`render_lambert_ellipsoid`, and one seed function.

#### D4 (Medium). Navigator-side models repeat their own logic and the renderer's

- `NavModelBodySimulated._limb_visible_arc_fraction` (`nav_model_body_simulated.py:798-839`) and
  `_terminator_visible_arc_fraction` (`:952-992`) are the same 20 lines with a different edge
  extractor; `_build_limb_arc_feature` (`:841`) and `_build_terminator_arc_feature` (`:994`)
  share their emission tail (`:899` and `:1047`).
- Ringlet and gap band coverage is computed identically on both sides
  (`nav_model/sim_ring.py:275-283` and `forward/ring_system.py:799-804`: two
  `compute_antialiasing_shade` calls and a `np.minimum`) even though `ring_geometry.py` exists to
  hold exactly such shared conventions; the per-kind radial layout is an if/elif ladder over kinds
  in four functions (`ring_system.py:326`, `:695-708`, `:799-831`; `sim_ring.py:275-320`).
- Four `instances_for_obs` class methods carry the same simulated-observation guard
  (`if not getattr(obs, 'is_simulated', False)`: `nav_model_body_simulated.py:352`,
  `nav_model_rings_simulated.py:150`, `nav_model_titan_simulated.py:192`,
  `stars/nav_model_stars_simulated.py:72`), and the four production models carry the inverse
  guard (`nav_model_body.py:686`, `nav_model_rings.py:318`, `nav_model_titan.py:504`,
  `stars/nav_model_stars.py:127`), so the simulated/real dispatch is spread over eight
  `instances_for_obs` methods.

#### D5 (Medium). Shared conventions and predicates written twice

- Light direction: `ellipsoid_geometry.illumination_vector` (`:275`) says "Both simulator sides
  derive the light direction here, so their illumination conventions cannot diverge", yet
  `mesh_geometry._light_direction` (`:251-266`) derives it again in its own axis order.
- Henyey-Greenstein phase: `ring_system.henyey_greenstein_phase(g, alpha_deg)` (`:138`) and
  `atmosphere.hg_phase_factor(g, phase_angle)` (`:269`) compute the same
  `(1 - g**2) / (1 + g**2 + 2*g*cos(alpha))**1.5`, one in degrees and one in radians with a floor.
  `forward/photometry.py` exists but holds only the body laws.
- `_deep_merge` twice (`instrument_catalog.py:193`, `instrument_navigator.py:76`); the planted
  pointing keys twice under two names (`render.py:47` `_PLANTED_POINTING_KEYS`,
  `instrument_navigator.py:72` `_PLANTED_OFFSET_KEYS`); the detector catalog merge
  `{**resolve_detector_defaults(instrument), **resolve_readout_mode(instrument, binning)}` twice
  (`forward/scene_radiance.py:121`, `forward/detector/params.py:509`); binning resolved twice
  (`scene_schema.scene_binning`, `:470`; `detector/params._resolve_binning`, `:283`).
- The `instrument_defaults` switch: `optics.instrument_defaults_on` (`:47`) exists, and
  `render.py:124-126`, `detector/params.py:517`, and `cli/sim_editor/artifacts_tab.py:115`
  re-implement it.
- "Is a PSF active?" twice, disagreeing: `render.resolve_oversample` (`:103-127`) treats any
  `instrument_defaults` scene as having a PSF and oversamples 4x, while `optics.effective_psf`
  (`:79-84`) returns no kernel for an instrument whose entry has none, as `generic.yaml` states.
  A generic scene with `instrument_defaults` therefore renders 16 times the pixels for no blur.
- Render-time re-validation: `DetectorParamError` is raised at 9 sites in
  `forward/detector/params.py` that repeat validator checks "as a backstop for scenes that bypass
  validation"
  (`scene_checks.py:428-430`), although every render validates first (`render.py:172`).

#### D6 (Low). Small copies

`optics_tab._dspin` (`:70`) is `widgets.make_dspin` (`:20`) verbatim while three sibling modules
import `make_dspin as _dspin`; `_as_map` is defined three times (`body_appearance.py:61`,
`body_atmosphere.py:41`, `ring_advanced.py:58`); `_stray_center_defaults` and
`_distortion_center_defaults` are identical (`stray_light.py:107`, `optics_tab.py:471`);
`_on_body_surge_value`, `_on_body_albedo_scalar`, and `_on_body_disc_scalar` differ only in the
block key (`body_appearance.py:328`, `:405`, `:507`); the noise widgets are synced in both
`noise.py:43-51` and `scene_io.py:181-187`; `full_render.py` and `verdict.py` are parallel
modules that each render, resolve the navigator block, and summarize; `_activated` is defined in
both `telemetry_loss.py:67` and `telemetry_artifacts.py:36`.

### 3. Logical structure

#### L1 (High). One declarative registry, three activation and dispatch mechanisms

`artifact_modes.py` is a good registry (see section 7), but what it registers is applied in three
different ways:

- **Telemetry modes** run in three near-identical loops (`forward/telemetry.py:118-168`), one per
  sub-stage, each with its own dispatch style: a direct call (`compression_dct`), a table
  (`LOSS_APPLIERS`), and an `if mode_name == 'reseau_scars' ... else` branch. To fit the table,
  all 12 loss appliers take `**_ignored: Any` (`telemetry_loss.py:115`, `:151`, ... `:533`).
  Activation (`_activated`, `_poisson_count`) is decided inside each applier.
- **Detector modes** come in three kinds. Chain-dispatched modes are tested one by one with
  `if 'name' in modes:` inside `_apply_ccd` (`forward/detector/chain.py:248-396`) and
  `_apply_vidicon`, with activation decided centrally (`params._mode_active`, `:356`). "Routed"
  modes are folded into flat `DetectorParams` fields, and their names are listed twice
  (`params._ROUTED_ONLY_MODES`, `:345-353`; `chain._record_routed_modes`, `:113-119`).
  `hot_pixels` is routed by its own block of special code (`params.py:666-692`) through a
  different resolver (`artifact_modes.resolve_mode_config`, its only caller) than every other mode
  (`instrument_resolvers.resolve_mode_with_catalog`). That special case needs a second special case
  in the instrument schema, `CATALOG_BLIND_MODES` (`instrument_schema.py:89-93`), so that an entry
  cannot give `hot_pixels` shape defaults the renderer would ignore.
- Each mode's parameter defaults are filled by the resolver, and the chain then applies its own
  fallbacks anyway (`cfg.get('orientation', 'horizontal')` at `chain.py:344` and `:559`,
  `cfg.get('saturation_frac', 0.99)` at `:380`), duplicating the registry's defaults
  (`artifact_modes.py:389`, `:456`).

In a planned design each registry entry names its stage, its slot in the chain, and its applier;
one loop per stage walks the registry; activation is decided in one place; and a noise-block knob
and an artifact mode that model the same effect (noise-block banding and `banding_coherent`, the
noise hot-pixel knobs and `hot_pixels`) are one effect with two ways to set it.

#### L2 (High). The central stage functions are monoliths

| Function | Lines | What it does |
| --- | ---: | --- |
| `forward/scene_radiance.py:130` `compose_scene_radiance` | 475 | seeds four sub-streams, resolves the PSF and zero point, renders stars, prepares every body (parallax, roll, oversample scaling, default ranges), depth-sorts, paints bodies, renders and composites the ring system and halos, captures per-class layers for the optics stage, and records truth |
| `forward/detector/params.py:481` `resolve_detector_params` | 333 | a key-by-key merge of scene, instrument catalog, navigator block, and `sim` block, with the `hot_pixels` and routed-mode special cases |
| `forward/atmosphere.py:454` `apply_atmosphere` | 304 | haze geometry, optical depth, phase, structure, disc and halo composition |
| `forward/body.py:594` `render_single_body` | 259 | seeds, dispatch to mesh, the `use_topo` decision, spec rebuilding, atmosphere, finish |
| `forward/ring_system.py:189` `render_ring_system` | 242 | per-feature tau, photometry, azimuthal clutter, moonlets, compositing maps |
| `forward/body.py:45` `create_simulated_body` | 165 | the classic renderer, including crater seeding |

`compose_scene_radiance` in particular reads as layers added in sequence: the pointing transform
(offset, roll, and spacecraft-ephemeris parallax, `:240-255`) is written inline rather than as one
function applied to every object, and the radiance stage reaches forward into the optics stage's
needs (`_optics_needs_layers`, `:794`, which inspects `optics.smear` so the stage knows whether to
save per-class layers that `smear.py:131` later pops). A planned design splits radiance into
sub-steps (pointing, stars, sky, bodies, rings, compositing, truth), each a function of a typed
scene and a frame.

#### L3 (Medium). Oversampling was bolted onto renderers written in detector pixels

The oversampling factor is threaded through 14 forward modules, most heavily `stages.py`,
`scene_radiance.py`, and `ring_system.py`. Each object scales its own pixel-space keys by hand,
by different rules: `_scale_star_params` keeps an explicit key list (`scene_radiance.py:813-847`), body centers and
axes are scaled in `compose_scene_radiance`, atmosphere lengths are scaled inside
`atmosphere_spec_from_params(oversample=...)`, surface-texture and transit lengths deliberately
stay in detector pixels (`body.py:613-616`), ring widths are multiplied by `os` per kind
(`ring_system.py:696-707`, `:800-823`), and `stages._downsample_truth` (`:254-302`) rescales the
truth records on the way back. A planned design renders through a grid or transform object so
that no renderer knows the factor, or states every scene length in one unit and converts at one
boundary.

#### L4 (Medium). A returned dataclass is unpacked and passed back field by field

`project_ellipsoid` returns an `EllipsoidProjection` (`ellipsoid_geometry.py:43-85`), the RORO
pattern the project rules ask for. Every caller then unpacks it into about 15 locals
(`forward/body.py:113-129`) and passes them back individually: `ellipsoid_image_normals` takes 9
of its fields as separate parameters (`ellipsoid_geometry.py:213-224`), and
`_add_craters_and_shading` takes 23 parameters (`forward/body.py:212`), 14 of them projection
fields. The planned version passes the projection object. Its second caller, `body_topo`, passes
fields of its own grid type (`_Grid`, `body_topo.py:440-450`), so the fix is one projection type
for both paths (D3).

#### L5 (Medium). Cross-cutting switches are interpreted separately in each layer

`artifacts.instrument_defaults` changes the oversampling (`render.py:124`), the PSF and distortion
(`optics.py:79`, `:106`), the detector chain (`params.py:517` and every `_default_or_zero` call),
the instrument-default artifact modes (`params._resolve_detector_modes`), validation
(`scene_checks.py:527`, `scene_checks_artifacts.py:90`), and the editor (`artifacts_tab.py:115`).
No single function says what the switch turns on. Likewise the "classic or topographic" decision
is a five-term predicate in the dispatcher (`body.py:727-733`) whose terms partly restate the
renderer's own capabilities. A planned design resolves such switches once, while building the
typed scene, into explicit per-stage settings.

#### L6 (Medium). The editor is one window object made of 16 mixins

`CreateSimulatedImageModel` (`main_window.py:63-81`) inherits 16 mixins plus `SimEditorBase`.
The base declares about 110 shared widget and state attributes and 34 stub methods that exist
"only to let mypy resolve the cross-mixin calls" (`base.py:1-11`, `:43-326`). Every mixin can
touch every widget, and some do: `scene_io._apply_params_dict` (169 lines, `scene_io.py:96`)
syncs widgets owned by the global-fields, noise, background-stars, and expected-outcome mixins.
Controls are wired one key at a time: 156 `_on_*` handlers across the package. The dev guide
calls the layout "one mixin per schema block" (`dev_guide_simulator.rst:4180`), but bodies span
four modules, rings two, the `detector` block lives in the Artifacts tab, and `spk_error` (an
ephemeris error) lives in the Optics tab (`optics_tab.py:700-729`). A planned design composes
the window from per-block panel widgets, each owning its controls and its load and save, with the
flat blocks generated from the field table (D1).

#### L7 (Low). Indirection that serves only tests

`full_render._render` (`:71`) and `verdict._summarize` (`:51`) are one-line wrappers that exist,
per their docstrings, "so a test can replace" the real function. A runner that takes its work
function as a parameter (it already does, `worker.py:88`) can be given a fake directly.

### 4. Naming

#### N1 (Medium). One word, several concepts

- **Instrument config** means three things: a simulator catalog entry
  (`sim/instrument_configs/*.yaml`, read by `instrument_catalog`), a navigator config block
  (`resolve_sim_inst_config` returns it, and callers name it `inst_config`), and the scene's
  `instrument_config` key, which overrides the navigator block, not the catalog entry.
- **Catalog** means the instrument catalog, the star catalog, the scene catalog of YAML files,
  a ring feature's catalog orbit, and, as a local variable, the merged detector defaults
  (`params.py:509`, `scene_radiance.py:121`).
- **Mode** means an artifact mode, a readout mode, an orbit's m-mode, a quantization mode, and a
  shading mode. The editor has `ModeRow` for an artifact-mode row (`artifact_mode_rows.py:89`) and
  `_ModeRow` for an orbit m-mode row (`ring_advanced.py:64`).
- **Truth** means the scene's truth keys (`TRUTH_KEYS`) and also `SimFrame.truth`, the renderer's
  output metadata, which carries idealized items such as star records and masks.
- **The scene dict** is called `sim_params`, `params`, `scene`, `p`, and `body_params`, depending
  on the module.

#### N2 (Medium). Names that describe history, not content

- The scene `noise` block holds `bias_dn` and `signal_full_scale_frac`, which are not noise, and
  `missing_data_rate`, which the telemetry stage applies (`forward/telemetry.py:243-248`).
- `render_combined_model` names neither the scene nor the pipeline; `CreateSimulatedImageModel` is
  a `QMainWindow`, not a model.
- `create_simulated_body` names two different functions (`forward/body.py:45`,
  `nav_model/sim_body.py:27`).
- In `nav_model/`, the navigator-side renderers are `sim_body.py` and `sim_ring.py` while their
  models are `nav_model_*_simulated.py`, and the star model alone sits in the `stars/`
  subpackage.
- `ring_advanced.py` and `body_appearance.py` say in their docstrings that they split the tab
  "so neither module runs long" (`ring_advanced.py:3`, `body_appearance.py:3-4`).
- `render_single_body`'s `seed` parameter is documented as the "Scene-level crater seed"
  (`body.py:618`) but is also the root of the relief, albedo, and mesh streams (`:747-761`).
  `chain._mode_seed` returns a generator, not a seed (`chain.py:101`).
- `forward/detector/__init__.py` describes the package without `electronics_stages.py` or
  `sqrt_lut.py`.

#### N3 (Low). Base-first naming applied unevenly

Module names mostly follow the base-first rule (`body_mesh`, `body_topo`, `telemetry_loss`,
`scene_checks_ring`). Class names often do not: `TopoBodySpec`, `MeshBodySpec`, `TopoBodyRender`,
`AlbedoTextureSpec`, `DiscTextureSpec`, and `TransitSpec` beside `BodySpec`; `TabulatedLut` beside
`SqrtLut`; `PredictedRingEdge` and `PredictedRingFeature`. The editor mixes tab-named modules
(`body_tab`, `optics_tab`, `artifacts_tab`) with panel-named ones (`noise`, `stray_light`,
`background_stars`, `global_fields`, `expected_outcome`), and `artifacts_tab` with
`artifact_mode_rows`. Three unrelated modules are named for noise: the editor's General-tab panel
(`cli/sim_editor/noise.py`), the detector effects (`forward/detector/noise_stages.py`), and the
sky-noise estimators (`realism/noise.py`).

#### N4 (Low). A mission named in a generic module's docstring

The simulator follows the mission-file rule: instrument data is in mission-named YAML files, and
no code branches on an instrument name. The reseau and resample mechanics in
`forward/telemetry_artifacts.py:107-227` are a parameterized triangular-lattice mechanic with no
mission constants; their Voyager values (202 marks, 59 px spacing) come from
`instrument_configs/vgiss.yaml:116-117` and `:139-145`. Only the module docstring is
mission-specific ("the Voyager GEOMED archive-processing scars"); it should describe the mechanic
and leave the mission to the YAML.

### 5. File length

#### F1. Over the limit

`nav_model/nav_model_body_simulated.py` is 1,084 lines, against the 1000-line rule in `CLAUDE.md`
and `python.mdc`. The natural seam is the limb and terminator arc machinery (`:694-1068`, about
375 lines including the twin visible-fraction methods of D4).

#### F2. Near the limit

Five modules sit between 900 and 999 lines: `realism/noise.py` 984,
`cli/sim_editor/ring_advanced.py` 983, `forward/body.py` 974, `cli/sim_editor/optics_tab.py` 962,
and `cli/sim_editor/body_appearance.py` 957. Six more sit between 800 and 899: `scene_checks.py`
866, `forward/scene_radiance.py` 847, `forward/ring_system.py` 833, `forward/detector/params.py`
813, `forward/atmosphere.py` 809, and `forward/body_topo.py` 805. Any further feature in these areas
forces another split.

#### F3. Splits made at the limit rather than at a seam

- **Editor tabs.** `ring_tab.py` (509) and `ring_advanced.py` (983), and `body_tab.py` (715) and
  `body_appearance.py` (957), split one tab each "so neither module runs long". The remaining
  halves are still large: `_build_body_tab` is 354 lines, `_build_ring_azimuthal_group` 191.
- **Scene checkers.** Bodies, rings, and artifacts got their own modules; stars, optics, noise,
  detector, downlink tables, spk error, sky counts, and the expected block stayed in
  `scene_checks.py` with the shared primitives. The split follows size, not the schema.
- **Detector effects.** `noise_stages.py` and `electronics_stages.py` are split by which scene
  block turns an effect on (the `noise` block versus an artifact mode, as
  `electronics_stages.py:1-4` says), not by physics: banding is
  in both (`add_banding`, `noise_stages.py:243`; `add_coherent_banding`,
  `electronics_stages.py:46`), as is dark signal (`add_dark_current`, `noise_stages.py:55`;
  `add_dark_ramp`, `electronics_stages.py:111`).
- **`forward/body.py`** holds the classic renderer, the crater carver used by both ellipsoid
  paths, the dispatcher, and the finisher (O4).
- **`instrument_catalog`, `instrument_schema`, `instrument_checks`, and `instrument_resolvers`**
  (created together in #934) are split along a principled seam (loading, schema, cross-checks,
  read accessors) and are not a problem.

Long functions are listed under L2; in the editor, `_build_star_tab` (278 lines), `_setup_ui`
(147), and `_rebuild_dynamic_tabs` (142) are the next largest.

---

## 6. What a planned layout would look like

The same functionality, laid out from a plan. Names are illustrative.

```text
spindoctor/sim/
  __init__.py              public API with __all__: render, load/save/validate, Scene types
  schema/
    nodes.py               one node vocabulary (Scalar, MapOf, ListOf, Record) and check_node,
                           used by scenes, instrument entries, and artifact-mode parameters
  scene/
    fields.py              the one field table: key, block, node, default, boundary class,
                           description, editor hint
    model.py               typed Scene, Body, RingFeature, Star, Optics, DetectorNoise, Artifacts;
                           parse_scene() applies every default once
    validate.py            field-table checks plus cross-field rules
    rules_body.py, rules_ring.py, rules_artifacts.py   cross-field rules only
    boundary.py            TRUTH_KEYS derived from fields.py; build_nav_params
    truth_echo.py
    io.py                  load, save, catalog paths and scene classes
  instruments/
    catalog.py, schema.py, checks.py, resolvers.py
    navigator_link.py      the few deliberate reads of navigator config
    configs/*.yaml         mission-named entries, the only image-side source of detector physics
  artifacts/
    registry.py            each mode: stage, chain slot, parameter nodes, applier
  shared/                  both sides; reads no scene dict except idealized per-object
                           records, as star_record_from_params and mesh_spec_from_params do now
    ellipsoid.py           projection, normals, and the smooth Lambert render
    mesh.py                mesh construction, rasterizer, mesh render
    ring.py                orbits, projection, per-kind radial layout and band coverage
    lighting.py            illumination vector, Henyey-Greenstein phase
    stars.py, spherical_harmonics.py
  forward/
    pipeline.py, frame.py  SimFrame, Stage, and a Grid that owns oversampling
    feature_loci.py        shared by the detector and telemetry stages
    radiance/              compose.py, pointing.py, stars.py, sky.py, compositing.py
    bodies/                render.py (dispatch), ellipsoid.py, mesh.py, craters.py, relief.py,
                           photometry.py, texture.py, atmosphere.py, haze.py, placement.py,
                           identity.py (one seed chain)
    rings/                 render.py, azimuthal.py, moonlets.py
    optics/                stage.py, smear.py, distortion.py, psf.py, ghosts.py, stray_light.py
    detector/              params.py, ccd.py, vidicon.py, calibration.py, quantization.py,
                           effects by physics (dark, bias, banding, pixels, transients),
                           shot_noise.py, sqrt_lut.py
    telemetry/             stage.py, loss.py, codec.py, vgiss_geomed.py
  render.py, png_export.py, seeds.py
  realism/                 unchanged; also takes the simulated-frame artifact counts

spindoctor/nav_model/simulated/    the navigator owns its renderers, per the dev guide
  body.py, body_arcs.py, rings.py, titan.py, stars.py,
  predict_body.py, predict_ring.py  thin callers of shared/

spindoctor/ui/sim_editor/
  window.py                composes per-block panels; no mixins
  panels/                  general.py, noise.py, optics.py, detector.py, artifacts.py, body.py,
                           ring.py, star.py, expected.py; flat blocks generated from fields.py
  render_display.py, workers.py, scene_io.py
```

| Current | Planned |
| --- | --- |
| `scene_schema.py`, the inventories in `scene_checks*.py` and `forward/haze_structure.py` | `scene/fields.py`, `scene/boundary.py` |
| `scene.py` (load, save, validate, nav params) | `scene/io.py`, `scene/validate.py`, `scene/boundary.py` |
| `scene_checks.py`, `scene_checks_body.py`, `scene_checks_ring.py`, `scene_checks_artifacts.py` | `scene/validate.py` plus `scene/rules_*.py` |
| `scene_truth_echo.py` | `scene/truth_echo.py` |
| `forward/body_spec.py`, `TopoBodySpec`, the specs in `body_texture.py`, `atmosphere.py`, `haze_structure.py`, `mesh_geometry.py` | `scene/model.py` |
| `instrument_schema.py` nodes, `artifact_modes.ModeParam` kinds | `schema/nodes.py` |
| `instrument_schema.py` entry schema (`_SCHEMA`, `ModeMaps`, cross-entry rules) | `instruments/schema.py` |
| `instrument_catalog.py`, `instrument_checks.py`, `instrument_resolvers.py`, `instrument_configs/` | `instruments/` |
| `instrument_navigator.py` | `instruments/navigator_link.py`; classifier call and status text to the editor |
| `artifact_modes.py`, `telemetry_loss.LOSS_APPLIERS`, the routed-mode lists | `artifacts/registry.py` |
| `ellipsoid_geometry.py`, the composition in `nav_model/sim_body.py`, `body_topo._Grid` | `shared/ellipsoid.py` |
| `mesh_geometry.py` | `shared/mesh.py` |
| `ring_geometry.py`, the band-coverage and kind-layout code in `ring_system.py` and `nav_model/sim_ring.py` | `shared/ring.py` |
| `illumination_vector`, `_light_direction`, both HG functions | `shared/lighting.py` |
| `star_records.py`, `spherical_harmonics.py` | `shared/stars.py`, `shared/spherical_harmonics.py` |
| `render.py`, `png_export.py`, `seeds.py` | unchanged, at the top of `sim/` |
| `forward/body.py`, `body_topo.py`, `body_mesh.py`, `translate.py`, `photometry.py`, `relief.py`, `body_texture.py`, `atmosphere.py`, `haze_structure.py` | `forward/bodies/` |
| `forward/pipeline.py`, `forward/stages.py` (`SimFrame`, downsampling) | `forward/pipeline.py`, `forward/frame.py` |
| `forward/scene_radiance.py`, `scene_compositing.py`, `star.py` | `forward/radiance/` |
| `forward/ring_system.py` | `forward/rings/` |
| `forward/optics.py`, `smear.py`, `distortion.py`, `psf.py`, `ghosts.py`, `lit_clusters.py` | `forward/optics/` |
| `forward/detector/*` | `forward/detector/`, effects regrouped by physics |
| `forward/telemetry*.py` | `forward/telemetry/` |
| `forward/feature_loci.py` | `forward/feature_loci.py` (shared by detector and telemetry) |
| `forward/artifact_counts.py` | `realism/` or the unit tests that use it |
| `realism/` | unchanged |
| `nav_model/nav_model_*_simulated.py`, `sim_body.py`, `sim_ring.py` (as thin callers), `stars/nav_model_stars_simulated.py` | `nav_model/simulated/` |
| `obs/obs_inst_sim.py`, `dataset/dataset_sim.py`, `config_440_sim.yaml` | unchanged |
| `cli/sim_editor/*` | `ui/sim_editor/` |

## 7. What is already well designed and should be kept

- **The stage pipeline.** `forward/pipeline.py` is 52 lines: a `STAGES` tuple, a `Stage` protocol,
  a mutable `SimFrame`, and one loop. Stage order is data, and each stage gets an independent
  generator from `seeds.derive_effect_seed` (a stable digest, never the salted `hash`). New stages
  and sub-streams slot in without reseeding others. This is the part of the simulator that looks
  most like a plan.
- **The artifact-mode registry as data.** `artifact_modes.py` describes each mode once (stage,
  parameters, activation key, detector models), and the validator, the instrument schema, and the
  editor read it. L1 is about the dispatch around it, not the table.
- **The instrument catalog.** Mission-named YAML entries with `base` inheritance and null-removal,
  checked by a declarative schema and a separate cross-check module at load time, read through
  small named accessors. It follows the mission-file rule and is the right model for D2.
- **The information boundary.** `build_nav_params` is default-deny, the classification is
  asserted complete at import, and a static AST guard keeps truth attributes out of navigator
  code. Of the structures in the package, this is the one most clearly designed before it was
  built.
- **One table, many consumers, where it exists.** `MESH_UNSUPPORTED_KEYS` and
  `mesh_unsupported_message` drive both the validator and the editor's disabled controls
  (`body_mesh_keys.py`); `RING_KIND_SHAPE_KEYS` drives both the ring validator and the editor's kind
  switch (`scene_checks_ring.py:373`, `ring_tab.py:76`). These are small examples of what D1 asks
  for at full scale.
- **Small, focused modules.** `translate.py`, `lit_clusters.py`, `seeds.py`,
  `detector/shot_noise.py`, `detector/sqrt_lut.py`, `ghosts.py`, and `cli/sim_editor/worker.py`
  (the thread hand-off) each do one thing and are easy to read.
- **The shared mesh render.** `mesh_geometry.render_mesh_body_image` is called by both sides, as
  the dev guide's rule intends; D3 asks for the ellipsoid to get the same treatment.

## 8. Recommended restructuring steps

Ordered from smallest risk to largest. Sizes are rough: XS is under a day, S one to two days, M
up to a week, L more than a week. "Baselines" means the step can change rendered pixels and must
be checked against the render-diff sheet and the committed baselines.

1. **Remove exact copies (XS).** One `make_dspin`, one `_as_map`, one `_activated`, one
   `_deep_merge` (with the catalog's null rule as an option), one planted-pointing tuple,
   `instrument_defaults_on` everywhere, read `random_seed` without the dead default, one routed-mode
   list. No behavior change.
2. **Rename for content (S).** `CreateSimulatedImageModel` to a window name; the navigator-side
   `create_simulated_body` to a prediction name; `ModeRow` and `_ModeRow` to `ArtifactModeRow`
   and `OrbitModeRow`; base-first spec classes; fix the two stale package docstrings and add
   `sim/__init__.py`'s `__all__`. Mechanical, but touches tests.
3. **Split `nav_model_body_simulated.py` (S).** Move the arc builders out along the
   limb/terminator seam and merge the twin visible-fraction methods. Brings the one over-limit
   module under 1000 lines.
4. **Make the validator's primitives public and split checkers by block (S to M).** Move stars,
   optics, and detector checks out of `scene_checks.py` so each schema block has one checker
   module; give the shared primitives public names in one module. Removes 50 of the 57 private
   reach-ins, which are the validator's.
5. **Move `artifact_counts.py` out of `forward/` (XS).** Put it beside `realism/` or with the unit
   tests that use it, and correct its docstring's claim that it serves the realism match.
6. **Shared lighting and one ellipsoid projection (M, baselines).** Put the light direction and
   the Henyey-Greenstein phase in one shared module and have the mesh path use it; have the
   topographic path project through `project_ellipsoid` instead of its own `_Grid`; make the
   smooth Lambert composition a shared function both sides call; replace the two full seed chains
   and the two crater fallbacks with one function that takes the fallback key as a parameter, so
   committed seed-less scenes keep their seeds. The classic path's output must stay bit-identical
   or its baselines regenerate.
7. **Registry-driven artifact dispatch (M, baselines).** Give each registry entry its applier and
   chain slot, run one loop per stage, decide activation in one place, route `hot_pixels` through
   the common resolver, and drop `CATALOG_BLIND_MODES`. Keep every per-mode stream name so no
   scene reseeds.
8. **Untangle the body package (M).** Separate craters, placement, and dispatch so the
   function-level imports go away; give each cached module a `clear_cache()`.
9. **One source of image-side detector physics (L, baselines).** Move the ADC ceiling, DN full
   well, signal scale, and the `sim`-block fallbacks into the simulator's instrument entries; keep
   navigator config reads to the deliberate matches.
10. **A typed scene model (L).** Extend the `BodySpec` approach to every block, parse once after
    validation, pass the typed scene to stages, and delete `TopoBodySpec`'s copied field list and
    the raw `.get(key, default)` reads. The navigator side gets its own idealized-only type,
    built by `build_nav_params` after the `nav_override` overlay, never the full `Scene`, so the
    default-deny filter and the boundary tests keep covering every field the navigator can
    reach.
11. **One field table and one schema vocabulary (L).** Derive the inventories, the boundary
    classification, the type checks, and the defaults from one table built on the instrument
    schema's node types; let the scene's detector-noise block reuse the instrument's detector
    nodes, which also settles the range disagreements of D2.
12. **Compose the editor from panels (L).** Replace the mixins with per-block panel widgets that own
    their controls and their load and save; generate flat blocks from the field table, which closes
    the 16 missing `noise` controls; move the package to `ui/sim_editor/`.
13. **A grid object for oversampling (L, baselines).** Replace the per-key `* os` scaling with one
    transform the renderers draw through.

Steps 10 and 11 are the ones that remove the cause rather than the symptoms; most of D1, D2, O1,
O3, and the editor drift in L6 follow from their absence.

## 9. Note: overlap with the realism critique

Written after the findings above were drafted. Section 11 of
`critiques/SIMULATOR_REALISM_CRITIQUE_2026-11-06.md` ("Code quality and architecture", on an
earlier baseline) already raised these, in shorter form:

- the two image-side body renderers kept for old baselines (part of D3);
- `compose_scene_radiance` as one long function, the restated seed default of 42, untyped scene
  dicts, crater defaults in several places, and a function with more than 20 parameters (L2, O1,
  L4);
- two parameter stores for detector values, the simulator catalog and the navigator YAML (part of
  O2);
- the editor's missing controls (then 26 keys across several blocks; this read checked only the
  `noise` block and finds 16) and its single class of 16 mixins (D1, L6);
- the sim-side copy of the navigator's image classifier and the test-only `artifact_counts.py`
  (O6, O7);
- the editor's missing controls are also filed as #902 (D1);
- the filtered navigator input named `sim_params` (part of N1).

Two of its items have since been addressed and are not repeated here: validation now runs on
every render (#959), and the per-instrument tables moved into the mission-named YAML catalog
(#934); this read finds no instrument-name branches left in code, which is why N4 is only a
docstring. That critique counted `DetectorParams` as sound because it is resolved once; this one
finds it resolved twice per render (O1) and counts its 333-line resolver and its third copy of the detector field list against it
(L2, D2).

New here: the three schema languages and their range disagreement (D2), the inverted dependency
between schema and renderer (O3), the hidden import cycles (O4), the flat layout (O5), the private
reach-ins (O8), the three artifact dispatch mechanisms (L1), oversampling threaded through 14
modules (L3), switches interpreted per layer (L5), the topographic path's private copy of the
ellipsoid projection and the split seed chain (D3), the duplicated light direction and phase
function (D5), the two config trees' consequence for the saturation ceiling (O2), the doubled
detector-parameter resolve (O1), and the file-length and split-point findings (F1-F3).
