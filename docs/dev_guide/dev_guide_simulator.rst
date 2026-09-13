===================
The Image Simulator
===================

Overview
========

The image simulator (the ``spindoctor.sim`` package) renders synthetic spacecraft
frames -- stars, planetary bodies, and rings, with a realistic detector model --
from operator-supplied geometry rather than from SPICE. It exists to **test and
validate the navigation pipeline**, not as an end-user product: because every
simulated frame's true pointing offset is known by construction, a simulated
image is the only frame whose navigation answer can be checked exactly. The
simulator drives the algorithmic-invariant tests, the regression baselines, the
single-variable sensitivity sweeps, and the sensitivity report
(:doc:`/simulator_report/simulator_report`).

The simulator has three equally valid entry points (the "three peers"):

- the **Python API** (:func:`spindoctor.sim.render.render_combined_model`, a thin
  cached driver over the stage pipeline in :mod:`spindoctor.sim.forward`),
- the **YAML scene catalog** under ``tests/integration/sim_scenes/`` (validated by
  :mod:`spindoctor.sim.scene`), which is the durable test artifact, and
- the **GUI** ``sd_create_simulated_image`` (the ``spindoctor.cli.sim_editor``
  package), an interactive editor for the same parameters.

Every parameter is reachable from all three; adding a physical effect means
adding it to the renderer, the scene schema, and a GUI control together. The
navigator side -- how a simulated frame is turned into ``NavFeature`` objects and
navigated -- is documented in the simulated-model chapters
(:doc:`dev_guide_navigation_models_body_simulated`,
:doc:`dev_guide_navigation_models_ring_simulated`,
:doc:`dev_guide_navigation_models_star_simulated`); this chapter documents the
architecture, the rendering side, and the scene formats.

.. _sim-two-sided:

The two-sided architecture
==========================

The simulator is split into two sides that exchange information through exactly
one channel. Understanding this split comes before everything else in this
chapter, because a change that quietly bridges it destroys the property the
simulator exists to provide: that a recovery measurement's only error is the
error the scene planted.

**The image side** renders the frame the navigator is given, reading the *full*
scene -- including every planted error, noise knob, and contaminant. It lives in
:mod:`spindoctor.sim.forward`:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Role
   * - ``stages.py``
     - The :class:`~spindoctor.sim.forward.stages.SimFrame` dataclass and the
       :class:`~spindoctor.sim.forward.stages.Stage` protocol (see
       :ref:`sim-stage-pipeline`).
   * - ``pipeline.py``
     - The fixed-order stage registry and ``run_pipeline`` driver.
   * - ``scene_radiance.py``
     - Composes the noise-free signal image: background stars, catalog stars,
       and the body/ring stack depth-sorted far to near.
   * - ``scene_compositing.py``
     - The radiance stage's translucent-screen machinery: orders the ring
       system and body halos far to near for compositing and enforces the
       explicit-``range_km`` depth contract on every overlap.
   * - ``body.py`` / ``body_topo.py`` / ``body_texture.py`` / ``body_mesh.py``
     - The ellipsoid body renderer and its topographic path (relief, photometric
       laws, surface texture, transits), and the polyhedral-mesh renderer (see
       :ref:`sim-body-renderer`).
   * - ``ring_system.py``
     - The optical-depth ring-system renderer: radial tau features on
       perturbed orbits, projected through the shared opening-angle
       geometry, lit by the single-scattering closed forms, and composited
       as a transmission screen (see :ref:`sim-ring-system`).
   * - ``star.py``
     - Star and background-star-field renderers.
   * - ``optics.py``
     - The optical-path stage: it runs motion smear, residual distortion, the
       whole-scene PSF, ghost reflections, and the stray-light field in a fixed
       internal order (see :ref:`sim-optics-stage`). The smear, distortion,
       PSF, and ghost sub-stages live in the sibling modules ``smear.py``,
       ``distortion.py``, ``psf.py``, and ``ghosts.py``.
   * - ``detector/``
     - The detector stage as a package: the CCD electron unit chain
       (``chain.py``), the resolved-parameter view (``params.py``), and the
       stochastic / structured noise sub-effects (``noise_stages.py`` -- dark,
       hot pixels, banding, bias structure, cosmic rays). Carries the vidicon
       DN path and the calibrated-I/F inversion (see :ref:`sim-detector-stage`).
   * - ``telemetry.py``
     - Transmission loss; carries the per-pixel missing-data markers and
       drives the structured loss modes in their fixed physical order.
   * - ``telemetry_loss.py``
     - The structured telemetry-loss appliers (missing / partial /
       alternating lines, missing blocks, garble, dead pixels, commanded
       frame shapes), each recording its realized geometry into the frame
       truth.
   * - ``telemetry_artifacts.py``
     - Telemetry-stage artifacts beyond packet loss: lossy DCT compression
       blockiness and the Voyager GEOMED archive-processing scars.
   * - ``artifact_modes.py``
     - The artifact-mode registry: each mode's stage, parameter schema,
       instrument availability, and implementation status, read by the
       validator, both stages, and the scene editor (see
       :ref:`sim-artifact-framework`).
   * - ``incidence.py``
     - Artifact-incidence measurement: the planted event counts read from
       the frame truth and the marker-based image estimators that recover
       them from the DN image alone (the realism match compares the two).
   * - ``feature_loci.py``
     - Navigation-feature loci (body / ring rows, limb and star pixels)
       extracted from the frame truth for adversarial artifact placement.
   * - ``atmosphere.py``
     - The haze-limb layer for atmospheric (Titan-class) bodies: an
       exponential haze column composited onto the disc, giving a soft limb, a
       terminator that brightens past 90 deg, and a forward-scattering ring of
       light at high phase (see :ref:`sim-atmosphere`).
   * - ``haze_structure.py``
     - The optional symmetry-breaking structure of that haze layer: axis tilt,
       hemispheric and azimuthal falloff variation, an interior brightness
       ramp, and cloud blobs. Bypassed entirely when a body's ``atmosphere``
       block names none of its keys (see :ref:`sim-atmosphere`).
   * - ``artifacts_catalog.py``
     - Per-instrument PSF, distortion-residual, and detector-chain defaults the
       ``instrument_defaults`` switch turns on (see
       :ref:`sim-artifacts-catalog`).

**The navigator side** predicts what the scene *should* look like, exactly the
way the SPICE-backed models predict a real frame. Its renderers live under
``nav_model/`` so the package layout states who owns them:
:mod:`spindoctor.nav_model.sim_body` and :mod:`spindoctor.nav_model.sim_ring`
draw the predicted templates that
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` and
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
emit as features, and
:class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`
builds its catalog star records directly.

.. _sim-information-boundary:

The information boundary
------------------------

Every key in the scene schema is classified into one of three disjoint classes.
**Idealized** keys are information the production pipeline could know from
catalogs, SPICE, labels, or configuration: instrument identity, image size,
exposure, body ellipsoid/mesh geometry and pose, ring orbits and epochs, star
catalog positions and magnitudes, and the per-star ``navigable`` flag.
**Truth** keys are nature's values and the test's contaminants: the planted
pointing offset and roll, the RNG realization, the noise and stray-light blocks,
crater terrain and the limb-relief field, the surface photometric law and
opposition surge, the albedo and disc textures and transiting moons and their
shadows, the mesh shading mode and per-frame pose scatter, the per-star PSF
anomaly and planted catalog error, the background sky, and the ``nav_override``
channel. **Test-only** keys are the
scene's declared navigation outcome (the ``expected`` block): read only by the
integration suite's assertion machinery, and by neither the renderer nor the
navigator. :data:`spindoctor.sim.scene.TRUTH_KEYS` is the machine-readable truth
inventory; an import-time assertion in :mod:`spindoctor.sim.scene` fails if any
schema key is left unclassified or lands in two classes, so a schema change
cannot dodge the classification. The boundary filter is default-deny, so the
test-only keys are stripped from the navigator's view alongside the truth keys.

The boundary's enforcement has two distinct strengths, and it matters which
applies where. The *channel* is structural:

- :class:`~spindoctor.obs.obs_inst_sim.ObsSim` exposes the navigator-side
  models a **filtered view**, ``obs.nav_params``, built by
  :func:`spindoctor.sim.scene.build_nav_params`: every truth key is stripped
  (the filter is default-deny -- an unclassified or unknown key stays behind),
  a body's ``nav_override`` mapping is overlaid onto its idealized view and
  then dropped (the navigator sees what it *believes*, never the true values
  underneath), and all values are deep copies. Nothing reachable *through
  the filtered view* carries truth.
- The navigator-side models read **only** ``obs.nav_params``. None of the
  renderer's output -- rendered star records, body masks, z-order maps, all
  accumulated on the frame's ``truth`` metadata -- ever crosses.
- Navigator-side quantities that depend on the detector are derived from the
  emulated instrument's **published** configuration, never from the scene's
  truth-side blocks. The star detection limit
  (:meth:`~spindoctor.obs.obs_inst_sim.ObsSim.star_max_usable_vmag`) is the
  worked example: it comes from the resolved per-instrument config block
  (including any idealized ``instrument_config`` overrides) and is independent
  of the scene's ``noise`` block, so a scene that plants noise different from
  the published values produces an honestly-wrong detection limit -- planted
  model error, exactly what such a scene is for.
- ``tests/spindoctor/sim/test_information_boundary.py`` constructs a scene
  exercising every ``TRUTH_KEYS`` entry and asserts none is reachable through
  the filtered view. This test is the independence guarantee: any change that
  adds a truth key must extend it in the same change (the test iterates the
  frozenset, so an unextended sample table fails loudly).

The *consumer* side of the guarantee is enforced by tests and review, not by
structure. The full scene -- planted offset included -- still rides the same
observation object every navigator-side model holds as ``self.obs``:
:class:`~spindoctor.obs.obs_inst_sim.ObsSim` stores it in truth-side
attributes (``sim_params``, ``sim_offset_v`` / ``sim_offset_u``, ``sim_time``,
and the renderer output records ``sim_body_models``, ``sim_inventory``,
``sim_body_order_near_to_far``, ``sim_body_index_map``,
``sim_body_mask_map``), each one attribute access away. What stands between
navigator-side code and those attributes is that the filtered view is the
only channel current navigator-side code reads, held there by three guards:
the information-boundary test suite above, the static guard test
(``tests/spindoctor/sim/test_boundary_static_guard.py``, which walks the AST
of every module under ``nav_model/``, ``nav_technique/``, and
``nav_orchestrator/`` and fails on any reference to a truth attribute name),
and code review. The guarantee is mechanical for the channel and
mechanical-plus-review for the consumers; the assessment behind this
phrasing is recorded in ``critiques/archive/SIM_REALISM_CRITIQUE_2026-07-18.md``.

Sharing code is fine; sharing information is not
------------------------------------------------

Both sides deliberately call the same geometry helpers --
:mod:`spindoctor.sim.ellipsoid_geometry`, :mod:`spindoctor.sim.mesh_geometry`,
:mod:`spindoctor.sim.ring_geometry`, and :mod:`spindoctor.sim.star_records`.
This is not a boundary leak; it is what makes the measurement clean. With
shared conventions (pixel centers, the sign of ``dv``, edge rasterization,
record defaults) the planted error is the *only* error in a recovery
measurement. Independent implementations would each carry their own
conventions, and any delta between them would land as an unknown systematic
inside the measured error -- contaminating the truth reference every simulator
result rests on. The discipline the shared helpers follow: they never read the
scene dict and never accept a truth key. Most take explicit geometry
arguments; the two record parsers
(:func:`spindoctor.sim.star_records.star_record_from_params` and
:func:`spindoctor.sim.mesh_geometry.mesh_spec_from_params`) parse per-object
mappings restricted to idealized keys, and on the navigator side they only
ever receive entries that have already passed the boundary filter. Either
way, truth cannot be smuggled across.

The honest residual: a bug in a shared helper that makes rendered features
subtly unlike *real* features leaves both sides consistent and recovery clean
-- scenes wrong, measurement blind. Its detectors are the reviewed render-diff
contact sheet (see :ref:`sim-catalog-workflow`) and the per-instrument realism
comparison against real cohorts, not any structural test.

One working rule follows from the blind spot. When implementing or extending
the image-side renderers in ``sim/forward/``, derive behavior from instrument
references and physics, **not** by consulting the navigator-side predicted
renderers. Copying the navigator's rendering choices manufactures exactly the
shared assumption the boundary cannot detect; the shared geometry helpers are
the sanctioned channel for conventions that must agree, and anything beyond
them agreeing is evidence, not a goal.

.. _sim-capability-envelope:

What the simulator can and cannot establish
===========================================

Everything after this section describes machinery. This section states what
that machinery's outputs *mean* -- which numbers are proofs, which are
measurements with conditions attached, and which are not evidence at all. It
is the chapter's answer to "what is this whole system for, and where does it
end."

The four axes
-------------

**Verification is unconditional.** Every simulated frame carries a planted
offset the navigator cannot see, so a recovery measurement's only error is
the error the scene planted. This does not depend on the renderer being
realistic: the information boundary (:ref:`sim-information-boundary`)
guarantees the navigator cannot cheat, and therefore a planted-truth recovery,
an algorithmic-invariant test, or a regression pin is an exact statement about
the navigation algorithms -- if a change breaks a baseline, the algorithms
changed behavior, full stop. This is the axis on which the simulator's word is
final.

**Precision is unconditional too, and is labelled as such.** The sweep curves
and the self-consistency floor (:ref:`sim-floor`) measure repeatability: how
tightly the pipeline reconverges when the rendered scene equals the
navigator's own model plus one controlled departure. The floor point on every
mismatch curve is reproducibility, never accuracy, and both this guide and the
simulator report mark it so. A sub-0.1 px floor says the machinery is
numerically tight; it says nothing about how far a real frame's answer is from
the truth.

**Accuracy is conditional on the realism match, per instrument.** A sim error
curve becomes an accuracy statement only to the degree the realism match
(:ref:`sim-realism-match`) supports it for that instrument. For the Cassini
NAC the support is strong: the tuned star PSF reproduces the cohort's
encircled-energy radii (measured on its 23 star-bearing frames) through the
same estimator on both sides (EE50 0.90 px
sim vs 0.91 real, EE80 1.72 vs 1.79), limb rise widths agree in the
like-for-like pool (medians 2.54 px both sides, normalized W1 0.16), ring-edge
widths agree at the few-tenths level, and the frame-averaged curve shapes
diverge by only 0.02-0.13. For the WAC the support is limited (four frames,
and the calibrated-chain noise comparison diverges by orders of magnitude
one-sidedly); for Galileo, Voyager, and LORRI the cohorts support fragments at
most, so on those instruments the simulator's error numbers are
reproducibility statements wearing no accuracy claim -- and will stay that way
until their cohorts grow.

**Calibration is sim-measured, with a disclosed basis and a measured
transfer.** The confidence formulas and tier boundaries have exact meaning on
the campaign that fitted them: each tier boundary is the smallest confidence
at which the tier's sigma-gated subset achieves a 0.9 success rate against the
tier's own error budget. That meaning comes with its basis attached: the
fitting campaign rendered with the empirical instrument chain off and emulated
the Cassini NAC exclusively (the disclosure lives in
``util/calibration/CAMPAIGN_20260718.md``, "Calibration basis and scope"), so
the realism match vouches for a different renderer configuration than the one
the coefficients were fitted on. Real-frame transfer is measured, not assumed:
the library cross-check against 75 operator sidecars agreed on status 69
times and on tier 46, and ``confidence_provisional: true`` rides every
metadata product until a real-anchored fit exists.

What this buys in practice
--------------------------

Three things, each impossible with real frames alone:

- **Navigator changes are measurable against truth.** The regression net --
  baselines, invariants, sweeps, and the expected-outcome pins -- turns "did
  this change help" into a measured delta against known answers, instead of a
  judgment call over operator labels.
- **Model error is priced honestly.** Because the campaign plants model error
  the navigator cannot represent, the fit learns what that error costs: the
  2.61 px limb covariance floor (unmodeled terrain bounds a limb fix's real
  accuracy no matter how clean the fit), the terminator technique's honest
  ~0.03-0.05 confidence plateau on a cohort with zero sub-pixel successes,
  and occlusion-aware visible-arc fractions on mutual-event scenes are all
  prices the simulator measured rather than guessed.
- **A navigator's own assumption can be attacked.** A navigator that measures
  pointing FROM a symmetry would be graded against its own premise if every
  rendered scene held that symmetry exactly. The haze layer's structure keys
  (:ref:`sim-atmosphere`) break it deliberately -- a tilted illumination axis,
  a non-affine hemispheric falloff difference, an azimuthal sharpness
  gradient, cloud blobs -- so the recovery error is measured against scenes
  the method's assumption does not describe.
- **Failure modes real data cannot isolate become provable.** The worked
  example is the planted-radial-error family: ``orbit_error_ringlet`` plants
  a 2.5 px radial catalog error and the fused result is a high-tier success
  at confidence 0.89, roughly 3 px wrong. On a real frame this is
  indistinguishable from a good lock -- only a frame whose truth is known by
  construction can *prove* the ensemble confidently absorbs this error, and
  keep proving it (the honest pin in :ref:`sim-expected`) until the gap is
  closed.

What it cannot do
-----------------

- **Certify real-frame accuracy.** The operator-verified image library
  remains the accuracy anchor -- and its independence from the simulator is
  precisely why the calibration is non-circular. No simulated measurement can
  replace it; the simulator's role is to make the navigator's behavior
  legible, not to grade its own renders.
- **Vouch for the thin-cohort instruments.** Galileo, Voyager, and LORRI
  numbers bound reproducibility only (see the four axes above); treating them
  as accuracy is exactly the misreading the per-instrument support labels
  exist to prevent.
- **See outside its modeled axes.** A model error the renderer does not
  implement contributes nothing to any curve or fit. The main unmodeled axes
  today: pointing jitter within an exposure, image mid-time error, readout
  shear, stars occulted by a dark limb (simulated star success is optimistic
  where an unlit body should block the star), and body-on-body cast shadows
  (mutual events render occlusion, not shadowing).
- **Fix the gaps it exposes.** A confident-wrong pin is evidence, not
  mitigation; closing it requires ensemble work (consuming declared
  model-error bars, cross-technique vetoes). The body-witness veto closes the
  shape-mismatch and haze-crescent modes; the remaining pinned hazard (the
  ring planted-orbit-error lock) stands until its own fix lands (see the
  ensemble chapter,
  :doc:`dev_guide_orchestrator_ensemble`).
- **Transfer its calibration to real frames by itself.** That path runs
  through the sidecar re-ratchet and a pre-stated transfer criterion (the
  proposed transfer watch in ``util/calibration/CAMPAIGN_20260718.md``), not
  through more simulation.
- **Guarantee the boundary against future consumers structurally.** The
  filtered channel is structural, but consumer discipline is enforced by the
  boundary suite, the static guard test, and review
  (:ref:`sim-information-boundary`); the full truth still rides the
  observation object. The assessment of what that enforcement level does and
  does not guarantee is recorded in
  ``critiques/archive/SIM_REALISM_CRITIQUE_2026-07-18.md``.

The realism figures behind the accuracy axis live in the simulator report's
realism section (:doc:`/simulator_report/simulator_report`); the calibration
basis and the cross-check numbers live in
``util/calibration/CAMPAIGN_20260718.md``.

.. _sim-stage-pipeline:

The stage pipeline
==================

The forward model renders in a fixed physical order that does not vary per
scene (:mod:`spindoctor.sim.forward.pipeline`):

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Stage name
     - Callable
     - What it does
   * - ``scene_radiance``
     - :func:`~spindoctor.sim.forward.scene_radiance.compose_scene_radiance`
     - Composes the noise-free signal: bodies depth-sorted far to near
       (nearer bodies overwrite) into ``frame.signal``, with the ring
       system composited over the stack as a per-pixel transmission
       screen, and the star field (catalog stars plus the background sky)
       into the point-source plane ``frame.point_e``, with the planted
       offset and camera roll applied. Stars are flux-normalized point
       masses shaped only by the
       whole-scene PSF (see :ref:`sim-star-params`). Accumulates feature truth
       (star records, masks, inventory, z-order) into ``frame.truth``.
   * - ``optics``
     - :func:`~spindoctor.sim.forward.optics.apply_optics`
     - Applies the optical-path effects in a fixed internal order -- motion
       smear, residual distortion, the whole-scene PSF, ghost reflections, then
       the stray-light field -- each contributing only when its ``optics``
       sub-block is present (see :ref:`sim-optics-stage`).
   * - ``downsample``
     - :func:`~spindoctor.sim.forward.stages.downsample_to_detector`
     - Box-downsamples the oversampled planes to the detector grid. A scene
       with an active PSF renders on a 4x oversampled grid and downsamples here;
       with no PSF the factor is 1 and this is a no-op.
   * - ``detector``
     - :func:`~spindoctor.sim.forward.detector.apply_detector`
     - Runs the electron unit chain: converts the composed [0, 1] signal to
       electrons through the exposure, applies Poisson shot noise, full-well
       bloom, read noise, banding, gain to DN, bias structure, quantization, and
       the ADC clip (see :ref:`sim-detector-stage`). The Voyager vidicon takes
       the DN-domain path; a calibrated instrument inverts the calibration
       transform afterward.
   * - ``telemetry``
     - :func:`~spindoctor.sim.forward.telemetry.apply_telemetry`
     - Applies the per-pixel missing-data markers.

Each stage is a callable matching the
:class:`~spindoctor.sim.forward.stages.Stage` protocol: it mutates a
:class:`~spindoctor.sim.forward.stages.SimFrame` in place, reads its parameters
from the full scene mapping, and draws randomness only from the
``numpy.random.Generator`` it is handed. Every effect is **off unless asked
for**: with no ``stray_light`` block no field is added, with no
``missing_data_rate`` no pixels are dropped, and with no ``noise``, ``detector``,
or ``artifacts`` block the detector adds no shot noise, read noise, dark, hot
pixels, banding, or bias structure. The detector always converts the composed
signal to detector counts -- that is what makes a DN frame -- but the stochastic
and structured noise activates only through a scene ``noise`` block (which pins
the individual parameters, e.g. ``poisson: true`` with a ``read_noise_dn``) or
the ``artifacts: {instrument_defaults: true}`` opt-in (which turns on the
emulated camera's physical signal chain -- including Poisson shot noise and the
catalog full-well bloom -- at catalog values). A scene with none of those
blocks renders a clean DN frame: the self-consistency floor. A single-variable
sweep therefore relies on every other block staying absent so it attributes
error to the one effect it varies.

``SimFrame`` carries the mutable image state between stages:

- ``signal`` -- the ``(V*os, U*os)`` float64 image. It holds normalized
  [0, ~1] intensive scene units through the radiance and optics stages (a
  point-mass star deposit may legitimately spike above 1.0 on the oversampled
  grid before the PSF spreads it); the detector stage converts it to electrons
  through the exposure and digitizes it to DN in place (the electron unit
  chain).
- ``point_e`` -- a same-shaped plane reserved for electron-unit point sources.
  The detector adds it into the electron image after the intensive conversion
  and before Poisson, so anything in it never passes through the signal scale.
  Stars deposit in signal units, so this plane stays zeroed.
- ``oversample`` -- the oversampling factor. A scene with an active PSF (an
  ``optics.psf`` block or ``instrument_defaults``) renders the radiance on a 4x
  oversampled grid and the box downsample returns it to the detector grid;
  otherwise it is 1.
- ``truth`` -- renderer output metadata (rendered star records, body masks,
  inventory, z-order maps). None of it crosses the information boundary.

**Per-stage seeding.** Each stage receives its own generator seeded via
:func:`spindoctor.sim.seeds.derive_effect_seed` from the scene's single
``random_seed`` and the stage's registry name, so one stage's noise realization
is independent of which other stages are enabled, and adding a stage later
leaves existing stages' realizations unchanged. A stage name is therefore part
of its scenes' noise realization: renaming a stage reseeds every scene that
exercises it and regenerates the affected baselines -- allowed, but not to be
done idly. (The seed derivation is a stable digest, never Python's per-process
salted ``hash``.) Sub-effects inside a stage derive their own named streams the
same way (background stars, cosmic rays, crater placement).

**The driver.** :func:`spindoctor.sim.render.render_combined_model` takes a
``sim_params`` dict and returns ``(image, metadata)``. It normalizes the
parameters into a deterministic JSON cache key, runs the pipeline on a fresh
frame, and caches the result (``lru_cache(maxsize=1)``, plus small shape caches
inside the body/mesh/star renderers; ``clear_render_caches`` drops them all for
determinism tests). The caching contract: scene parameters are
JSON-serializable scalars/lists/maps, no wall-clock or global RNG state is
consulted, and two renders of the same scene are bit-identical on one machine.

.. _sim-instruments:

**Instruments.** The ``instrument`` field selects which per-instrument
configuration block drives the detector model and PSF
(:mod:`spindoctor.sim.instruments`). The recognized names are ``coiss_nac``,
``coiss_wac``, ``coiss_calib_nac``, ``coiss_calib_wac``, ``gossi``, ``nhlorri``,
and ``vgiss``, plus ``generic`` (alias ``sim``) for the instrument-agnostic
defaults. Calibrated (``*_calib_*``) instruments are in I/F units with a NaN
missing-data marker and no full-well; raw instruments are in DN with a 0 marker.
Calibrated scenes render through the full DN chain and then invert the
calibration transform, so they carry propagated shot/read noise and quantization
texture in I/F units. A scene can pin or override individual instrument settings
with
``instrument_config`` (see :ref:`sim-instrument-config`).

.. _sim-optics-stage:

The optics stage
================

The optics stage runs on the oversampled radiance image in a fixed internal
order that mirrors image formation
(:func:`~spindoctor.sim.forward.optics.apply_optics`). Each sub-stage
contributes only when its ``optics`` sub-block is present, so a scene names
exactly the optical effects it wants and leaves the rest at the floor.

1. **Motion smear** (:mod:`~spindoctor.sim.forward.smear`) averages the
   radiance over the exposure along the pointing drift. It runs first, while
   the per-object-class radiance layers are still separable, so the optics
   below form the image of the time-averaged radiance. A single
   ``object_class: all`` entry smears the whole scene; several entries give
   differential smear, where each class carries its own drift vector.
2. **Residual distortion** (:mod:`~spindoctor.sim.forward.distortion`) warps
   the geometric image by the low-order radial polynomial (``k1``, ``k2`` about
   the optical centre) plus an optional seeded non-radial wander -- the field
   error left after the navigator applies each camera's known distortion model.
   A limb fitted at the frame edge then disagrees with a ring fitted through the
   centre by the differential residual, which the navigator gets no model to
   remove.
3. **Whole-scene PSF** (:mod:`~spindoctor.sim.forward.psf`) convolves the image
   by a core-Gaussian-plus-Moffat-wing kernel, so the limb gradient, the
   ring-edge gradient, and every star shape inherit one profile. The core is
   elliptical (``sigma_v`` may differ from ``sigma_u``); ``w`` is exactly the
   fraction of the kernel's energy in the isotropic wing, and each term is
   normalized so the kernel conserves flux. The Cassini cameras get a wider
   truncation window for their documented long wings.
4. **Ghost reflections** (:mod:`~spindoctor.sim.forward.ghosts`) add displaced,
   defocused, low-amplitude copies of the formed focal-plane image (internal
   reflections). Each ghost copies the pre-ghost signal, so ghosts do not
   reflect one another.
5. **Stray light** adds the smooth low-frequency scattered-light field (a linear
   ramp or a radial bump) last, before the detector stage. It is additive, so it
   brightens dark sky as well as lit features; the navigator's source-image
   background filter is meant to remove it.

Only the distortion non-radial field draws randomness, and it derives its own
seeded stream from the scene seed, so the optics stage does not consume the
pipeline generator. A sub-stage whose block is a true no-op (all-zero radial
distortion, an amplitude-0 ghost) renders bit-identically to one without it.

.. _sim-detector-stage:

The detector stage
==================

The detector stage (:mod:`spindoctor.sim.forward.detector`, a package) is the
normative unit chain of the forward model. For a CCD instrument the composed
intensive signal passes through the electron unit chain in
:func:`~spindoctor.sim.forward.detector.chain.apply_detector`:

1. **Signal to electrons.** The intensive [0, 1] signal is scaled to electrons
   through ``signal_full_scale_frac * full_well_e * (exposure_sec /
   exposure_ref_sec)``.
2. **Point sources.** The point-source plane (``frame.point_e``) carries the
   stars: for a CCD it is electrons, added after the intensive conversion and
   before Poisson so it never passes through the signal scale; for the Voyager
   vidicon it is DN, added onto the converted signal before the DN-domain
   noise (see :ref:`sim-star-params`).
3. **Dark current, then Poisson.** A dark pedestal accumulates over the
   exposure before the shot term, so the shot noise grows with the dark signal.
   Poisson shot noise then acts on the electron image.
4. **Hot pixels and full-well bloom.** Hot pixels are stamped, then charge above
   ``full_well_e`` spills along the column up to ``bloom_length`` pixels each
   way and the image is capped at the well. A camera's physical saturation
   therefore emerges from ``full_well_e / gain_e_per_dn`` (below the ADC ceiling
   for Cassini), not from the ADC clip.
5. **Cosmic rays and read noise.** Cosmic rays deposit above the well after the
   bloom cap, so they reach the ADC ceiling and land on the orchestrator's
   masks. Gaussian read noise is added in electrons.
6. **Banding, gain to DN, bias structure.** Coherent banding is added in
   electrons; the image is divided by ``gain_e_per_dn`` and offset by the bias
   pedestal into DN; low-order bias structure (per-image pedestal jitter,
   row/column gradients) is added in DN.
7. **Quantize and clip.** The DN image is quantized by the selected ADC sub-mode
   and clipped at ``saturation_dn``.

The stochastic and structured sub-effects (dark, hot pixels, banding, bias
structure, cosmic rays) live in
:mod:`~spindoctor.sim.forward.detector.noise_stages`; the chain draws each one
an independent RNG stream from the scene seed, so toggling one never perturbs
another's realization. The deterministic signal-to-DN conversion always runs --
it is what makes a DN frame -- while each stochastic sub-effect is a no-op when
its gating amplitude, fraction, or rate is zero.

**Quantization sub-modes.** ``exact`` rounds to integer DN (uniform bins);
``8bit`` rounds to integer DN and clips at the 255 code ceiling (the output
word width, independent of ``saturation_dn``); ``uneven_12bit`` snaps values
near the power-of-two carry boundaries to reproduce the histogram spikes of an
ADC with unequal bit weights; ``sqrt_lut`` companding encodes to 8 bits through
a square-root LUT and back, leaving a signal-dependent quantization residual;
``ls8b`` keeps only the low 8 bits, so a value above 255 wraps modulo 256 (the
banded wraparound on a bright target); and ``contour_8bit`` posterizes an 8-bit
word to multiples of its step, the visible contouring of a coarse ADC. The
``quantization_lut``, ``quantization_ls8b``, and ``contouring_8bit`` artifact
modes select these last three.

**Artifact modes in the chain.** The registry's detector modes (see
:ref:`sim-artifacts-block`) render at their physical point in this chain: the
fixed-pattern response before Poisson, the dark ramp with the dark pedestal, the
frame-transfer smear as integrated charge, the anti-blooming pairs and radiation
transients after the shot term, coherent banding in electrons before the gain
divide, and the additive fixed pattern and serial tail on the digitized DN.
Their mechanics live in
:mod:`~spindoctor.sim.forward.detector.electronics_stages`, each drawn its own
seeded stream and a no-op at zero incidence.

**The vidicon path.** The Voyager vidicon
(:func:`~spindoctor.sim.forward.detector.chain.apply_detector`, vidicon branch)
is not photon-noise dominated, so it skips the electron conversion: the signal
maps straight to the 8-bit DN full scale and the noise is applied in DN -- a
line-correlated read-noise term (a per-line offset plus a within-line white
component) and a faint vertical coherent periodic component -- then 8-bit
quantization.

**The calibrated-I/F path.** A calibrated (``coiss_calib_*``) instrument renders
through the full DN chain and then inverts the calibration transform: the bias
and dark pedestals are subtracted before the exposure divide (matching the real
pipeline), so a calibrated frame carries no spurious 1/exposure pedestal and
comes out in I/F units with propagated shot / read noise and quantization
texture. The calibration scale is derived so a noise-free signal of 1.0 at the
reference exposure round-trips to I/F 1.0.

:func:`~spindoctor.sim.forward.detector.params.resolve_detector_params`
collapses the emulated instrument's config block, the per-instrument catalog
defaults, the scene ``detector`` / ``noise`` blocks, and the
``artifacts.instrument_defaults`` switch into one flat
:class:`~spindoctor.sim.forward.detector.params.DetectorParams` view the chain
reads. Resolution precedence, highest first: an explicit scene key (``detector``
block, then ``noise`` block), then the catalog value when ``instrument_defaults``
is on, then the disabled floor -- the physical-chain artifacts default to zero,
so an unconfigured scene renders a clean DN frame. A scene that selects a gain
state the instrument does not catalogue is a validation error, not a silent
guess.

.. _sim-artifacts-catalog:

The instrument-defaults switch and the artifacts catalog
========================================================

``artifacts: {instrument_defaults: true}`` opts the whole scene into the
emulated camera's physical signal chain at catalog values: the whole-scene PSF
kernel and the residual-distortion amplitude
(:func:`~spindoctor.sim.forward.optics.instrument_defaults_on` gates the optics
side), and the detector electron chain's Poisson shot noise, read noise, dark
current, hot pixels, full-well bloom, banding, and bias structure (the
detector side reads the same switch). An explicit ``noise`` key still wins in
either direction -- ``noise: {poisson: false}`` turns the shot term off under
instrument defaults. It turns on only the *physical* chain: the per-mode loss
incidences (cosmic rays, whole-line telemetry loss, compression-block
dropouts, and the like) stay at zero, so a defaults scene is a
clean-but-realistic frame, not a damaged one.

:mod:`spindoctor.sim.forward.artifacts_catalog` is the single home for those
per-instrument values: ``PSF_KERNELS`` (core sigma and wing parameters),
``DISTORTION_RESIDUAL_RMS_PX`` (the residual field error), and
``DETECTOR_DEFAULTS`` (the full electron-chain, gain-table, read-noise, and
vidicon numbers, plus an ``artifact_modes`` sub-map of per-mode shape defaults --
banding amplitudes and periods, frame-transfer scrub/transfer times, fixed-
pattern components, and the rest). A scene that names an artifact mode with only
its incidence inherits these shapes (``resolve_mode_with_catalog`` resolves scene
value over catalog default over registry default); ``incidence`` itself is never
catalogued, so naming an instrument never plants a defect on its own. Every value
is provenance-tagged in a comment beside it, and every value is interim -- sized
from published FWHMs, gain tables, and documented residual-error bounds, pending
the per-instrument measurement passes -- so the wing parameters and noise
amplitudes are the first quantities the realism-match pass revisits. The
calibrated Cassini instrument names alias their raw entries, and ``generic``
(alias ``sim``) is an instrument-agnostic ideal 12-bit detector whose electron
well equals its DN depth at unit gain.

.. _sim-artifact-framework:

The artifact-mode framework
===========================

Beyond the ``instrument_defaults`` switch, the ``artifacts`` block carries one
map per *artifact mode*: a named scene defect (a telemetry loss, a
detector-electronics effect, an archive-processing scar) the operator opts into
one at a time. The modes are registered once in
:mod:`spindoctor.sim.forward.artifact_modes` -- the ``ARTIFACT_MODES`` registry
is their single source of truth. Each :class:`~spindoctor.sim.forward.artifact_modes.ArtifactMode`
fixes the mode's name, its rendering stage (``telemetry`` or ``detector``), the
:class:`~spindoctor.sim.forward.artifact_modes.ModeParam` schema of its
parameters, the sim instruments it is available on, and the meaning of its
``incidence``. Every consumer reads that description rather than carrying its
own copy: the scene validator (:func:`spindoctor.sim.scene_checks._check_artifacts`),
the telemetry and detector stages, and the scene-editor GUI all iterate the
registry, so registering a mode is the whole job of adding one editor row, one
validation entry, and one dispatch slot.

The block's two non-mode keys are switches: ``instrument_defaults`` (the
physical-chain opt-in above) and ``adversarial`` (placement, below). Every other
key must be a registered mode name; an unknown key, an unimplemented mode, or a
mode named on an instrument it is unavailable on fails validation with the
registry's message.

Incidence semantics
-------------------

Every mode carries an ``incidence`` parameter, and every mode is a no-op at
incidence 0 (the stage-activation rule; 0 is also the default, so naming an
instrument never plants a defect on its own). Its meaning is per-mode:

* **count** modes draw a Poisson event count per frame (lost lines, lost blocks,
  spiked pixels, radiation hits): incidence is the expected number of events.
* **probability** modes are commanded or periodic (a frame edit, a cutout, a
  quantization sub-mode, most detector effects): incidence is the per-frame
  probability the mode activates at all.
* **density** modes (``hot_pixels``) read incidence as a spatial fraction of
  pixels affected, not a per-frame event count.

Telemetry-stage modes
---------------------

Applied at the detector grid after readout, in the physical order of
transmission (see :data:`~spindoctor.sim.forward.artifact_modes.STRUCTURED_LOSS_ORDER`).
Loss modes write the missing-data marker (0 on the raw-DN path, NaN on the
calibrated path); garble and spikes write *wrong* values instead.

.. list-table::
   :header-rows: 1
   :widths: 18 44 20 18

   * - Mode
     - Shape
     - Availability
     - Incidence
   * - ``missing_lines``
     - whole lines zeroed to the marker (a contiguous run or scattered)
     - all
     - count
   * - ``partial_lines``
     - a line truncated from a column to its end, or a middle segment lost
     - nac, wac, vgiss
     - count
   * - ``alternating_lines``
     - every Nth line dropped, or (``mode: keep``, the Galileo HMA / HCA
       catalog default) only every Nth line kept
     - nac, wac, gossi
     - probability
   * - ``edited_frame``
     - only a centred column band (440 px by default), or one half-height, kept
     - gossi, vgiss
     - probability
   * - ``truncated_frame``
     - a clean full-width band of lines (a quarter frame by default) cut from
       the top or bottom
     - nac, wac, gossi
     - probability
   * - ``missing_blocks``
     - compression-block-aligned bands zeroed
     - nac, wac, gossi
     - count
   * - ``line_garble``
     - a line's tail replaced with random DN, not the marker
     - gossi, vgiss
     - count
   * - ``pixel_spikes``
     - isolated pixels flipped to a wrong DN (a bit-flip or a uniform draw)
     - vgiss
     - count
   * - ``dead_pixels``
     - singleton pixels held at a low response
     - all CCD
     - count
   * - ``dead_columns``
     - whole columns held at a low response
     - all CCD
     - count
   * - ``embedded_header``
     - row 0 overwritten with binary housekeeping (the LORRI header)
     - lorri
     - probability
   * - ``truth_window``
     - a losslessly-clean carve-out the other losses must spare
     - gossi
     - probability
   * - ``cutout_window``
     - only a commanded rectangle survives; the border is blanked
     - gossi, lorri
     - probability
   * - ``compression_dct``
     - lossy DCT block quantization on the transmitted signal, before loss
     - all CCD
     - probability
   * - ``reseau_scars``
     - reseau-removal smudges on the vidicon lattice (archive processing)
     - vgiss
     - probability
   * - ``resample_texture``
     - GEOMED resample warp, blank border, and missing-line interpolation
     - vgiss
     - probability

The telemetry sub-stage order is fixed: lossy compression runs first (a codec
compresses, then packets drop), then the structured loss loop, then the Voyager
GEOMED archive-processing scars on the already-loss-bearing frame
(:data:`~spindoctor.sim.forward.artifact_modes.TELEMETRY_PRE_LOSS_ORDER`,
``STRUCTURED_LOSS_ORDER``,
:data:`~spindoctor.sim.forward.artifact_modes.TELEMETRY_POST_LOSS_ORDER`). The
commanded ``truth_window`` is resolved before the loop and passed to
``missing_blocks`` as a protected rectangle.

Detector-stage modes
--------------------

Applied inside the detector electron chain, each at the physically right point
(dark ramps with the dark pedestal, banding in electrons pre-DN, fixed-pattern
PRNU multiplicative pre-Poisson, serial tail as a post-gain DN undershoot). The
:data:`~spindoctor.sim.forward.artifact_modes.DETECTOR_MODE_ORDER` fixes only
which mode's truth record is written first.

.. list-table::
   :header-rows: 1
   :widths: 18 44 20 18

   * - Mode
     - Shape
     - Availability
     - Incidence
   * - ``hot_pixels``
     - a fixed spatial population of high-dark-current pixels
     - nac, wac, gossi
     - density
   * - ``banding_coherent``
     - a coherent electron-domain intensity ripple
     - all
     - probability
   * - ``bias_structure``
     - a bias pedestal with row and column gradients
     - all
     - probability
   * - ``dark_ramp``
     - a readout / shutter-shading dark ramp across the frame
     - all
     - probability
   * - ``bloom``
     - electron-domain column bleed from saturated wells
     - nac, wac, gossi
     - probability
   * - ``radiation_transients``
     - cosmic-ray hits during integration and readout dwell
     - all CCD
     - count
   * - ``bright_dark_pairs``
     - anti-blooming-mode vertical bright / dark pixel pairs
     - nac, wac
     - count
   * - ``frame_transfer_smear``
     - frame-transfer column smear
     - lorri
     - probability
   * - ``serial_tail``
     - a post-saturation serial-register tail
     - lorri
     - probability
   * - ``beam_bend``
     - a brightness-dependent vidicon limb bend
     - vgiss
     - probability
   * - ``residual_image``
     - a prior-frame ghost from incomplete erasure
     - vgiss
     - probability
   * - ``quantization_lut``
     - the Cassini sqrt-companding LUT quantization
     - nac, wac
     - probability
   * - ``quantization_ls8b``
     - the Cassini LS8B wraparound quantization
     - nac, wac
     - probability
   * - ``contouring_8bit``
     - 8-bit contouring quantization steps
     - gossi, vgiss
     - probability
   * - ``fixed_pattern``
     - a static PRNU / vignetting / stitch / jail-bar / dust composite
     - all
     - probability

Precedence
----------

A mode's parameter value is resolved scene value over catalog default over
registry default (:func:`~spindoctor.sim.forward.artifacts_catalog.resolve_mode_with_catalog`),
so a scene that names a mode with only its incidence inherits the instrument's
catalogued shape. ``incidence`` itself is never catalogued. Where a mode shares a
mechanic with a generic ``noise``-block knob (``hot_pixels`` versus
``noise.hot_pixel_fraction``, for instance), the explicit artifact mode wins.

Adversarial placement
---------------------

``artifacts: {adversarial: true}`` biases every enabled mode's stochastic
placement onto the navigation features -- a lost line crosses the disc, a dead
pixel lands on a star -- through the shared
:mod:`spindoctor.sim.forward.feature_loci` helpers; placement is uniform when the
switch is off or absent. Purely periodic or commanded shapes (``alternating_lines``,
the frame-level edits) ignore the switch: their geometry is fixed once active.
The switch is one scene-level flag, so a scene is either adversarial for all its
modes or none.

Truth bookkeeping and incidence measurement
-------------------------------------------

Each applied mode records its realized geometry -- which lines, blocks, or pixels
it touched -- into ``frame.truth['artifacts'][<mode>]``, so a later measurement
can compare *planted* against *measured*.
:mod:`spindoctor.sim.forward.incidence` carries both sides:
:func:`~spindoctor.sim.forward.incidence.planted_incidence` reads the truth
records into a realized event count per mode (exact by construction), and the
``measured_*`` estimators recover the same counts from the DN image alone, the
way a detector run against a real archive frame would. The image estimators key
on the marker, so they are exact for the marker-based structural modes
(``missing_lines``, ``partial_lines``, ``missing_blocks``, ``dead_pixels``) on a
frame whose scene floor sits above the marker. Modes that plant wrong values
rather than the marker -- garble, spikes, and the detector-electronics modes --
are not recoverable pixel-by-pixel from a single frame (distinguishing them from
scene structure needs multi-frame or noise statistics), so for those the truth
side is authoritative.

The artifact-sweep scene class
------------------------------

The ``artifact_sweep`` scene class (``tests/integration/sim_scenes/artifact_sweep/``)
pins the survivable end of the artifact axis: a disc scene and a star-field scene,
each under a modest structured loss (a few missing lines plus a few truncated
lines per frame), in a uniform-placement and an adversarial-placement variant.
The navigator must still reach success and recover the planted offset within
tolerance -- the level at which realism does not break navigation. The companion
``artifact_missing_lines`` sweep (``tests/integration/sim_sweeps/``) drives the
other end: it raises the missing-line incidence from a clean frame past the
navigability cliff and records the navigation-quality-versus-incidence curve.

.. _sim-star-confounder:

Star confounders and the breakdown curve
----------------------------------------

The ``star_confounder`` scene class (``tests/integration/sim_scenes/star_confounder/``)
is the star-field analog of the artifact sweep's survivable end: one, two, or
three navigable stars planted in a field of non-navigable confounders (the
1/2/3-star lock regimes), plus a saturated-star and a double-star scene. The
navigator must recover the planted offset within a tolerance wide enough to
absorb any centroid bias the confounders and planted catalog error induce. The
confounders are ``navigable: false`` catalog stars and the raised-brightness
``sky_counts`` field: both render, neither reaches the navigator, so the star
technique has to lock on the true subset while comparable-brightness clutter
crowds its search window.

The companion ``star_confounder_density`` sweep (``tests/integration/sim_sweeps/``)
drives the breakdown: it walks the confounder field's ``density_factor`` from a
clean frame up past the cliff on the one-star lock geometry, so the recovery goes
from a confident sub-pixel lock to a failed / low-confidence result. The measured
transition point is the deliverable -- as comparable-brightness confounders crowd
the window the brightness-uniqueness gate trips and the technique returns
spurious, never a confident wrong offset. The sweep's ``ensemble_seeds`` mode
replicates each density across several seeds (the geometry fixed, the confounder
field redrawn per seed), so each sweep point is a small *population* rather than a
single realization. Near the cliff the population splits between recovery and
failure -- exactly the raw material an estimator-validation study needs, and the
reason the mode exists.

.. _sim-expected:

The expected-outcome block
--------------------------

A scene may carry a scene-level ``expected`` block declaring the outcome the
navigator should produce: a required ``status`` (``success`` / ``failed`` /
``conflicted``), a ``confidence_tier`` (one of the five navigation ranks, or
null to assert the status only), an optional ``status_reason`` token, and an
optional ``known_offset_error_px`` / ``known_offset_error_tol_px`` pair (the
honest pin, below). It is
a **test-only** key -- read by the assertion machinery in
``tests/integration/sim_expected.py``, fed to neither the renderer nor the
navigator, and stripped from ``nav_params`` by the information boundary along
with the truth keys. The block is modeled on the image-library sidecar's
expected-outcome taxonomy (the same status / tier / reason cross-field rules: a
``failed`` or ``conflicted`` status pins the matching tier) but is validated
independently -- a sim scene is not a sidecar.

The ``expected_fail`` scene class (``tests/integration/sim_scenes/expected_fail/``)
is why the block exists. When a scene scatters every star off its catalog
position, or drowns a lone star in an overwhelming confounder field, the
*correct* navigation outcome is a failed or low-confidence result -- never a
confident wrong offset. Each such scene carries an ``expected`` block, and the
machinery turns "must not be confidently wrong here" into a passing assertion.

**The honest pin.** A small second family inverts the reading: a scene whose
measured behavior *is* a wrong offset the ensemble cannot remove -- the
``orbit_error_ringlet`` planted radial catalog error, whose bias survives even
though the declared-sigma covariance channel demotes its tier. Pinning bare
``status: success`` on such a scene would quietly freeze the wrong answer as
correct behavior, so it also declares ``known_offset_error_px`` -- the measured
fused error magnitude, from the scene's recorded baseline -- with a
``known_offset_error_tol_px`` band. The assertion then fails in *both*
directions: a regression that worsens the error fails, and a genuine fix that
shrinks it also fails loudly, prompting a deliberate re-pin (or retirement of
the pin) instead of a silent behavior change. The pinned success is a
documented hazard held in view, never an endorsement; the scene comments and
the campaign record carry the analysis, and the ensemble chapter
(:doc:`dev_guide_orchestrator_ensemble`) names the failure family for the
readers who consume tiers. The ``titan_crescent_horns`` pair left this family
once the cross-technique body-witness veto closed it: its haze-dragged
lone-blob offset is declined outright, so the scenes assert that ``failed``
outcome instead of an honest-pin success.

.. _sim-floor:

The self-consistency floor
==========================

With no ``optics``, ``noise``, ``detector``, or ``artifacts`` block a scene
renders the *self-consistency floor*: the detector converts the composed signal
to DN and nothing else acts -- no PSF, no shot or read noise, no dark, hot
pixels, bloom, banding, bias structure, distortion, smear, ghosts, or stray
light. Each effect is off unless asked for, and a single-variable sweep relies
on every other block staying absent so it attributes error to the one effect it
varies. The floor's matching PSF configuration is ``optics: {psf:
{match_navigator: true}}``: the authored form is preserved through validation,
saving, and loading, and the renderer resolves it into the navigator's own
model -- a pure Gaussian at the emulated instrument's ``star_psf_sigma``, no
Moffat wing, no field variation -- when it builds the kernel. Because an active
whole-scene PSF is the *only* convolution a star receives (stars deposit as
point masses, never pre-spread-then-convolved), a floor scene's rendered star
sigma equals the navigator's configured sigma exactly and the only residual is
the one the scene plants elsewhere.

Model-mismatch sweeps
=====================

The single-variable sweep harness (:mod:`tests.integration.sim_sweep`; specs
under ``tests/integration/sim_sweeps/``, run by
``python -m tests.integration.sim_sweep_runner``) takes one catalog scene, walks
a single parameter across a list of values, navigates each step, and records the
recovered-offset error, confidence, status, and primary technique per step. A
sweep may set ``ensemble_seeds: N`` to replicate every point across N seeds (the
geometry fixed, the noise and confounders redrawn) so a point becomes a
population rather than a single realization.

A *model-mismatch* sweep uses this harness to characterize one render-vs-navigate
disagreement. Each starts from a **self-consistency floor** -- the value at which
the rendered image equals the navigator's own model, so the floor point measures
reproducibility, not accuracy -- and walks the mismatch up; the
recovery-error-vs-mismatch curve is the product. The floor is equality with the
navigator's configuration (for the PSF axis, a pure Gaussian at the instrument's
``star_psf_sigma`` with the wing weight at zero; for the photometric axis,
Minnaert ``k = 1``, which is exactly Lambert; for the ephemeris and smear axes, a
zero planted error). The mismatch parameter is a render-side truth key the
boundary filter strips, so the navigator keeps its default belief while the
render departs from it. The sweeps and their floors:

- ``psf_limb_mismatch`` -- rendered PSF sigma broadened above the navigator's
  Gaussian; the softened limb edge biases the distance-transform fit and, past a
  few pixels of blur, washes the edge out (a navigability cliff).
- ``photometric_minnaert_mismatch`` -- Minnaert ``k`` walked below 1; the
  limb-darkened disc departs from the Lambert template.
- ``spk_parallax_error`` -- a spacecraft-position error parallax-shifts the body
  while the navigator predicts the catalog geometry; the recovery absorbs the
  shift almost one-for-one.
- ``differential_smear`` -- a per-object-class motion vector trails the stars
  while the navigator predicts point sources.
- ``ring_orbit_error`` -- a navigable ringlet's rendered orbit displaced radially
  off the catalog orbit.
- ``atmosphere_haze`` -- a Titan-class haze softens the limb the navigator
  predicts as hard. The sweep drives a ``HAZEMOON`` body, so it measures
  haze-BLIND body navigation; the haze solar-symmetry fit has its own base scene
  and offset sweeps.

The remaining Section 8 axes are covered by pre-existing sweeps
(``noise_read_noise``, ``artifact_missing_lines``, ``irregularity_shape_mismatch``,
``pose_disagreement``, ``star_confounder_density``, ``star_catalog_scatter``, and
the per-technique ``*_offset_fine`` / ``*_offset_wide`` body-ephemeris sweeps) or
by scene classes where a single-parameter sweep is not the right shape (the
``ring_system`` clutter scenes and the ``mutual_event`` occluded-limb scenes).
:mod:`tests.integration.sim_sweep_plots` renders the mismatch curves into
``docs/simulator_report/_figures/mismatch_axes.png``, one panel per axis with the
floor point drawn distinctly.

.. _sim-body-renderer:

The body renderer
=================

A body is drawn by one of two renderers chosen by ``shape_model``: the
ellipsoid renderer (``body.py``) and the polyhedral-mesh renderer
(``body_mesh.py``), both sharing the axis convention below. A smooth Lambert
ellipsoid at oversample 1 renders through the classic path unchanged. Whenever
an ellipsoid body needs more than that -- a limb-relief field, a non-Lambert
photometric law or opposition surge, a surface texture or a transit, or simply
an oversampled radiance grid -- it dispatches to the *topographic path*
(``body_topo.py``). All of the extra ingredients are truth keys: the
navigator's predicted body is always the smooth Lambert template, so every one
plants a known model error the technique must survive.

The topographic path
--------------------

**The relief field.** Limb relief is a 2-D fractional-height field
``h(lat, lon)`` on the body surface, not a 1-D profile on the limb, so the
limb perturbation and the terminator shadowing are two slices of one
consistent surface (``relief.py``). The field is a periodic Gaussian random
field synthesized by FFT: the spectral coefficients are independent complex
Gaussians with variance ``S(k) proportional to exp(-(|k| corr_rad / 2)**2)``
(``corr_rad`` the correlation length in radians of surface arc), band-limited
at ``kmax = ceil(8 / corr_rad)`` where ``S`` has fallen to ~1e-7 of its peak.
The field's poles sit on the observer axis, so the sub-observer horizon circle
-- the limb -- is the field's equator, where the grid metric is exact and the
commanded RMS and correlation length hold with no map distortion.

Modes with total wavenumber below **3** are zeroed before use: degree-1 radius
content is, to first order, a translation of the body -- an untruthed center
offset no limb fit could separate from the pointing error -- and degree-2
content aliases ellipsoid shape error, which is a separate scene knob. After
the cutoff the field is rescaled so the *limb slice's* standard deviation
equals the commanded ``limb_relief_rms`` per realization.

**Limb and terminator application.** The renderer's normalized ellipse radial
function ``e(p)`` (1 exactly at the unperturbed limb) is divided by
``1 + delta(theta)`` -- ``delta`` the relief sampled along the sub-observer
horizon circle -- so the perturbed limb lands at ``r_ellipse (1 + delta)`` and
the silhouette turns ragged. The relief azimuth ``theta`` is the elliptical
parametric angle in the body's rotated frame, not the image azimuth about the
body center: the field is attached to the body, so the silhouette and the
terminator march sample one consistent surface under any in-plane rotation
(for a circular disc the two are identical). Shading normals keep the
unperturbed ``e`` (the
disc shading is low-frequency; relief moves the edge, not the interior).
Near-terminator disc points are then shadowed by a march against upstream
terrain in absolute heights ``H = h R`` (pixels): a point is shadowed when some
upstream sample at surface distance ``d`` toward the sun satisfies
``H_up - H_pt > d / tan(i)``. The march is capped at
``d_max = min((H_max - H_min) tan(i), sqrt(2 R H_max))`` -- the longest shadow
the terrain can cast, and the horizon limit that bounds the tangent's
divergence at the terminator -- so raggedness grows toward the terminator while
the cost stays bounded. The march steps at ``max(1.0, field_cell_arc / 4)``
pixels of surface arc -- about 16 samples per shortest terrain wavelength,
never finer than one render pixel -- which is why it is both accurate and
cheap.

Split-resolution performance
----------------------------

Disc shading is low-frequency by construction, so the topographic path
computes the shading field (surface normals, crater texture, photometric law)
once at **detector resolution** and bilinearly upsamples it, while only the
sharp content -- the relief-perturbed silhouette mask and the terminator march
-- is rasterized at the full oversampled working grid. This replaces the
per-subsample shading that made an oversampled body ~16x more expensive than
its detector-grid equivalent. It is what lets a body-bearing scene meet the
render budget below, and it makes the topographic path the *fast* path at
oversample > 1 even for a plain Lambert body. The budget harness
(``tests/integration/test_sim_perf.py``) holds a large-lit-body-with-relief
frame to its measured budget for exactly this reason.

Photometric laws
----------------

The topographic renderer shades an ellipsoid with a scene-selected law
(``photometry.py``), each normalized to 1 at disc center under head-on
illumination and written in ``mu0 = cos(incidence)`` and ``mu = cos(emission)``:

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - ``photometric_law``
     - Side
     - Form
   * - ``lambert``
     - truth
     - ``I = mu0`` (the navigator's own template law).
   * - ``lommel_seeliger``
     - truth
     - ``I = 2 mu0 / (mu0 + mu)`` -- limb-brightened, clipped at the signal
       ceiling.
   * - ``minnaert``
     - truth
     - ``I = mu0**k mu**(k - 1)`` with ``minnaert_k``; ``k = 1`` is Lambert,
       ``k = 0.5`` the classic lunar value.
   * - ``lunar_lambert``
     - truth
     - ``2 L(a) mu0 / (mu0 + mu) + (1 - L(a)) mu0`` with the McEwen (1991)
       cubic blend ``L(a)`` in phase ``a``: Lommel-Seeliger at opposition,
       Lambert once the cubic reaches 0 near 119 deg.

An optional ``opposition_surge`` (truth) multiplies the law by a normalized
exponential ``(1 + amplitude exp(-a / width)) / (1 + amplitude)`` -- 1 at exact
opposition, ``1 / (1 + amplitude)`` far from it -- so it plants the
brightness-versus-phase surge signature while keeping the normalized signal
plane within [0, 1]. Because the navigator always shades Lambert, any non-Lambert
law or surge moves the terminator and the limb-darkening profile by a known
amount.

Surface texture and transits
-----------------------------

Three multiplicative-texture families sit on the body *shading* (never the
silhouette), so they change what disc correlation sees without moving the limb
the navigator fits (``body_texture.py``):

- ``albedo_texture`` (truth): a band-limited multiplicative noise field
  (``rms``, ``corr_px`` in detector pixels on the disc) plus discrete circular
  ``spots`` -- the disc contrast of an icy moon. The noise reuses the relief
  spectral synthesis on its own seeded stream, zeroing only the mean (albedo
  cannot alias a translation the way relief can, and large-scale hemispheric
  contrast is exactly what the texture exists to plant).
- ``disc_texture`` (truth): a low-frequency latitude band pattern
  ``1 + band_amplitude cos(band_wavenumber lat_p + band_phase)`` plus discrete
  storm ovals -- the zones/belts and a great-red-spot-class storm of a giant
  planet.
- ``transits`` (truth): a list of entries, each a transiting moon disc
  (``moon``: ``dv_px``, ``du_px``, ``radius_px``, ``albedo_factor``) and/or its
  cast shadow (``shadow``: ``dv_px``, ``du_px``, ``radius_px``, ``darkness``),
  in detector pixels from the body center. Every shadow multiplies the textured
  shading first, then every moon disc overwrites on top, and a transiting disc
  renders only where it overlaps the parent silhouette.

The cast shadow is the point of the confound: it is a sharp, high-contrast
circular *false crater* on the disc. A disc-correlation or blob technique that
starts chasing the shadow (or the storm) moves the offset baseline, so a banded
planet with a transit and shadow is the regression tripwire for that failure --
the measured behavior is that the correlation and limb fit stay accurate
against the smooth Lambert template, and the scene fails loudly if that stops
being true.

Mesh shading, detail, relief, and pose scatter
----------------------------------------------

The mesh renderer builds the same irregular polyhedron the navigator predicts
(the primitives live in the shared ``mesh_geometry`` module), then adds its own
truth-key upgrades on top, none of which the prediction consumes:

- ``shading`` (truth) selects the shared rasterizer's mode for the *rendered*
  image: ``flat`` (default) or ``gouraud`` per-vertex smooth shading. The
  rasterizer capability is shared, and each side chooses its own mode; the
  navigator's predicted mesh keeps flat shading because the key never crosses
  the boundary. The gallery renders the same body flat and gouraud so the
  difference is visible.
- ``mesh_detail_octaves`` (idealized) adds banks of higher-frequency shape
  modes to the base relief spectrum (0 = base only). It is idealized because a
  published shape model carries its own detail.
- ``limb_relief_rms`` / ``limb_relief_corr_deg`` (truth) apply the same relief
  machinery as a per-vertex radial perturbation of the unit mesh, sampled in
  body-fixed coordinates so the terrain rotates with the pose.
- ``pose_scatter`` (truth): a seeded per-frame Gaussian perturbation
  (``sigma_deg`` per Euler axis) added to the *rendered* pose only. The
  navigator predicts the catalog pose, so the drawn rotation is a known-wrong
  rotation state; the draw is recorded in the render truth as
  ``pose_scatter_drawn_deg``.

The non-Lambert photometric laws, opposition surge, crater carving, the
surface-texture and transit families, and the atmosphere halo are
ellipsoid-path features; a mesh body carries the relief, shading, detail, and
pose-scatter keys. The scene validator *rejects* the ellipsoid-only appearance
keys on a ``polyhedral_mesh`` body -- the mesh renderer would silently ignore
them, and a scene that names an effect must render it or fail loudly.
Silhouette ``anti_aliasing`` is consumed at oversample 1 only, on both render
paths (an oversampled scene already resolves the limb on its own grid).

Mutual events
-------------

A scene can place two bodies so the nearer one occludes part of the farther one
(depth-ordered by ``range_km``; bodies at exactly equal ranges tie-break
deterministically by scene-list position, the earlier-listed body rendering as
the nearer). The renderer draws the true occlusion and
records the outcome in the render truth (``body_occlusion``): each body's
visible fraction and the angular extent of any hidden limb arc. This is
bookkeeping the test can read, not something the navigator sees.

Both body models -- the simulated one and the SPICE-backed
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` -- are occlusion-aware.
Each per-body model knows the other bodies' idealized geometry and explicit
nearer-or-farther depth order (``range_km`` in the sim; the inventory center
range under SPICE), and from that it drops the hidden limb and terminator
vertices from its predicted polylines, reports a ``visible_arc_fraction`` that
falls with occlusion depth, and trims the hidden region out of the ``BODY_DISC``
correlation template. So a mutual-event limb scores like the partial arc it is,
and the disc correlator scores only against disc brightness the image carries.
Nothing reads a truth key: the occlusion is derived from the same idealized
sibling geometry both models already hold.

Across grazing, half, and deep overlap the navigator recovers each scene's
planted offset within the tolerance the mutual-event regression pins. That is a
statement about these simulated scenes, not a certification of real-frame
accuracy: the sim renders an idealized occlusion the true optics only
approximate. On the deep scene the far body's disc is over 90 percent
hidden, so its ``visible_lit_fraction`` collapses and the disc gate drops that
feature; the near body's disc plus both visible limb arcs fuse to the recovered
offset. The joint limb fit runs on arcs the image actually shows -- the robust
Tukey biweight loss is a second line of defense against a stray vertex, not the
mechanism that carries the result -- and no confident-wrong result or
double-counting conflict arises. The mutual-event scenes pin the arc fractions
and offsets so a regression toward hidden-arc lock-on fails loudly.

.. _sim-atmosphere:

Atmospheres (haze limb)
-----------------------

A body carrying an ``atmosphere`` block gains an exponential haze layer above
its surface (``atmosphere.py``), evaluated after the disc is shaded: the
on-disc haze joins the body's opaque paint, while the above-limb glow is a
translucent halo screen composited like the ring system's -- background,
rings, and stars behind it show through scaled by ``exp(-tau)`` with the glow
added, and the body's mask / depth / occlusion truth sees only the solid
silhouette. A
line of sight grazing the body at tangent altitude ``h`` (pixels above the
reference radius) accumulates a tangent optical depth
``tau(h) = tau_ref * exp(-(h - ref_altitude_px) / scale_height_px)``, so
``tau_ref`` is the tangent optical depth at ``ref_altitude_px``. The emergent
haze brightness is a single-scattering source (a Henyey-Greenstein phase
factor of asymmetry ``g``, times a wrapped illumination weight) multiplied by
an opacity ``1 - exp(-tau)``. Three consequences follow:

- **A soft limb.** Above the geometric limb the opacity fades as an
  exponential ramp, so the sharp edge becomes a soft glow whose apparent
  radius sits outside the reference radius. Because the source scales with the
  phase factor, the ramp is brighter -- and the apparent limb sits further out
  -- at high phase than at low phase. That is the physical root of the Titan
  altitude-versus-phase problem.

- **A terminator that wraps.** The illumination weight stays positive past
  the terminator over the horizon-dip angle ``sqrt(2 * H / R)`` of arc (the
  solar depression at which a column one scale height up loses direct
  sunlight; ~0.5 rad for the catalog Titans, floored at 0.05 rad for thin
  atmospheres), so the night side near the terminator brightens smoothly
  instead of cutting off at 90 deg incidence.

- **A ring of light.** A forward-scattering haze (``g`` > 0) peaks toward high
  phase, so at phase beyond about 150 deg the whole limb lights up and the
  crescent horns extend past the geometric terminator into a near-complete
  ring -- Titan's ring of light, which falls out of the same layer rather than
  being special-cased. An optional ``detached_px`` shell adds a second haze
  band at that altitude. The shell exists only above the limb: the on-disc
  haze column is shell-blind (a geometrically thin shell projected against
  the disc adds negligible slant contrast), so the shell appears solely in
  the tangent glow.

**Symmetry-breaking structure.** The layer above is exactly mirror-symmetric
about the image-plane line through the body centre and the sub-solar direction:
one exponential column, one illumination weight, no azimuthal or hemispheric
structure. That is a problem for grading
:doc:`the haze solar-symmetry fit <dev_guide_techniques_titan_haze>`, which
measures pointing FROM that symmetry and would otherwise be scored against a
scene built from its own assumption. ``haze_structure.py`` supplies six optional
truth keys inside the same ``atmosphere`` block that break it, each with its own
render-level test asserting the effect is measurably present:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Key
     - Effect on the rendered haze
   * - ``axis_tilt_deg``
     - Rotates the haze's own illumination axis away from the geometric sun
       direction a navigator predicts, so the true mirror plane is not the
       predicted one.
   * - ``ns_falloff_ratio``
     - Scales the haze falloff length on one hemisphere only -- a genuinely
       NON-affine difference between the two halves the mirror maps onto each
       other, which is what actually probes a Pearson score's blindness to an
       affine one.
   * - ``ns_asymmetry_amplitude``
     - Scales one hemisphere's brightness: the affine counterpart the Pearson
       score is supposed to absorb.
   * - ``sector_sharpness_gradient``
     - Scales the falloff length with azimuth around the limb, walking the
       detected limb ridge radially as a function of ray angle -- the
       sector-asymmetric edge-localization bias.
   * - ``interior_ramp_amplitude``
     - Adds a linear brightness ramp along the haze axis inside the disc, the
       seasonal north-south gradient.
   * - ``cloud_blobs``
     - Adds Gaussian clouds on the disc, each ``{center_vu, sigma_px,
       amplitude}``.

The hemispheres are the two sides of the body's first semi-axis in the body
frame, so a scene that wants the hemispheric split to be the one the mirror maps
onto itself orients its illumination perpendicular to that axis. Every field is
optional and the whole module is bypassed when a block names none of them: the
haze then renders through exactly the scalar arithmetic above. That gating is a
performance contract, not only a tidiness one -- the per-pixel scale-height field
costs an array where the base haze costs a scalar, and ``test_sim_perf.py``
carries cold-render budgets for the structureless path.

The layer is a truth key: the navigator's predicted body (see
:doc:`dev_guide_navigation_models_body_simulated`) keeps a hard limb at the
reference radius and never learns the haze exists, so the soft rendered limb is
a deliberate model mismatch. A navigator that fits the bright sunward haze ramp
recovers a small offset toward the sunlit limb whose size tracks the
phase-dependent apparent limb radius (and the haze parameters) -- the
``atmosphere`` catalog scenes measure and pin that bias honestly. The
low-phase ``titan_haze_limb`` scene records a sub-pixel sunward offset (a few
tenths of a pixel in ``du``) at the medium tier, the limb DT fit and the disc
correlation agreeing on the haze-lifted limb.

Everything in this paragraph is about *body* navigation against a haze the
navigator does not model. The scenes that measure it carry a body named
``HAZEMOON`` precisely so they stay in that regime: a body named ``TITAN`` routes
to the haze model and
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`, which
is not haze-blind and measures something else entirely. The ``titan_haze`` scene
is the one that drives that technique. At high phase the haze is
outright hostile: in ``titan_crescent_horns`` (155 deg) the ring of light
defeats the disc correlation (spurious -- there is no full lit disc), the
predicted terminator arc is emitted but dropped by the feature reliability
gate before any technique runs (its honestly computed reliability, capped by
``sin(155 deg)`` = 0.42, scores 0.26 against the 0.30 ``TERMINATOR_ARC``
threshold), and the haze-dragged blob centroid alone would carry the fused
answer roughly 30 px off in ``du``. That is exactly the lone-blob collapsed
regime the body-witness veto (see :doc:`dev_guide_orchestrator_ensemble`)
guards: the only surviving body offset is the bias-prone centroid while a
sibling geometric technique on the same body self-flagged spurious, so the
frame is declined (``failed``, ``status_reason``
``lone_blob_in_collapsed_regime``) rather than reported as a 30-px-wrong
success -- a wrong offset that passes the acceptance gate is worse than a
declared failure. The scene asserts that decline. The noiseless
sibling ``titan_crescent_horns_noiseless`` -- Poisson and read noise off, the
same haze and phase geometry -- pins the underlying centroid bias
deterministically (planted ``du`` -0.8, blob centroid at 29.17), so the ring
of light alone, not the noise, drives the collapse, and the decline is stable
rather than noise-dependent. The haze evaluation
is restricted to the bounding box of the
body plus its halo (out to a detached shell's reach), so its cost scales with
that box rather than the frame, and a body without an ``atmosphere`` block
renders hard-limbed and byte-for-byte unchanged.

.. _sim-ring-renderer:

The ring renderer
=================

The whole ring system of a scene is one ``ring_system`` block drawn by one
renderer (``ring_system.py``): a list of radial optical-depth features
composed into a single tau map in ring-plane coordinates, projected through
the shared opening-angle geometry, lit by the single-scattering closed forms,
and composited over the body stack as a per-pixel transmission screen. The
navigator's predicted counterpart is geometric, not photometric -- it
rasterizes coverage templates and boundary polylines from the catalog orbits
of the features it is told about, through the same projection helpers, and
never sees a brightness. The key-by-key inventory lives in
:ref:`sim-ring-system`; this section describes the machinery behind it and
the measured navigation behavior the ``ring_system`` catalog scenes pin.

Feature kinds and the radial tau profile
----------------------------------------

Every feature contributes a radial tau profile anchored at its orbit radius
``r_e(lam)`` (which varies with longitude once the orbit is perturbed) and
the profiles compose by addition, clipped at zero:

- A ``ringlet`` carries its ``tau`` across a band of radial ``width``
  outward of the orbit radius; a ``gap`` is the same band as a suppression,
  carving tau out of whatever it overlaps -- so a gap is authored *into* a
  sheet, and its edges are the sheet's material ending, not a painted dark
  band.
- An ``edge`` is a one-sided step: a semi-infinite sheet bounded by its
  outer edge (``side: 'in'``, the B-ring-edge case) or its inner edge
  (``side: 'out'``). It is the cheapest way to author "the rest of the ring"
  behind a navigable boundary.
- A ``ramp`` is a linear tau transition across its band: one end sharp, one
  end fading to nothing. Only the sharp end is a fittable boundary; the
  faded end has no gradient to fit, and the navigator-side model emits no
  edge for it.
- A ``wave`` is a damped radial sinusoid launched at the orbit radius --
  a density-wave train riding on a sheet, alternating positive and negative
  tau lobes that shrink with distance. The profile exists only *outward* of
  the launch radius and is exactly zero inside it. That clamp is
  load-bearing, not cosmetic: the damping envelope is an exponential decay
  *downstream* of the launch radius, and evaluated on the wrong side it
  would grow without bound -- a formula for an infinitely bright artifact,
  not a wave. Physically the wave propagates away from the resonance that
  launches it; there is nothing upstream to draw.

Kind-specific shape keys are validated strictly (a ``width`` on an ``edge``
fails loudly) because a stray shape key would silently author a different
feature than intended.

The catalog orbit: m-modes and edge waves
-----------------------------------------

A feature's ``orbit`` is catalog knowledge -- the navigator predicts from
exactly what the renderer draws, unless a planted error (below) says
otherwise. The base is the mode-1 precessing ellipse (``a``, ``ae``,
``long_peri``, ``rate_peri`` applied across ``time - ring_epoch``). Two
perturbation families ride on it:

- ``modes`` (m >= 2): resonantly forced radial modes,
  ``r = a - amp cos(m (lam - peri))``. An ``m = 2`` mode is the classic
  two-lobed outer-B-ring edge shape; higher m gives the scalloped
  multi-lobed edges of resonantly confined ringlets. The mode longitudes
  live in the ring-plane frame (measured from the ascending node), so the
  lobes foreshorten correctly under an inclined view.
- ``edge_wave``: a satellite edge wave -- the Daphnis/Pan signature -- a
  radial sinusoid whose azimuthal wavelength is an arc length and whose
  amplitude decays exponentially *downstream* of the perturbing moon's
  longitude ``lam0``. The same upstream clamp as the radial wave applies,
  for the same reason: the decay envelope written as an exponential in
  downstream longitude grows without bound if evaluated upstream of the
  moon, so the longitude difference is wrapped into one forward turn.
  Immediately upstream of the moon the wave has then wrapped nearly a full
  orbit and carries a factor ``exp(-2 pi / damp)`` of its launch amplitude
  -- the validator caps ``damp`` at 2 radians, bounding that wrap-seam
  residual at ``exp(-pi)``, about 4.3% of the launch amplitude -- which is
  the periodic statement of "the wave trails the moon, it does not lead
  it".

Both families are idealized: the navigator predicts the same scalloped,
wave-perturbed boundaries the renderer draws, and the catalog scenes
``mmode_ringlet`` and ``edge_wave_gap`` pin that the planted pointing offset
is recovered through m-modes and edge waves under inclined projection.

Planted orbit error and the navigable subset
--------------------------------------------

A feature's truth-side ``orbit_error`` (``delta_a_px``, ``delta_ae_px``,
``delta_long_peri_deg``) displaces the *rendered* feature off the catalog
orbit the navigator predicts from -- planted radial model error, the ring
analog of the body ephemeris-error axis. Real ring features are misplaced
relative to their published orbit solutions in exactly this way. The
idealized ``declared_orbit_sigma`` is the uncertainty the navigator is
*entitled* to know -- the error bars, never the drawn values. It feeds the
predicted edges twice: as the per-vertex radial sigma of the robust fit, and
as the fully-correlated orbit-uncertainty term the ring-edge technique adds
in quadrature to its reported covariance along the fit's radial direction
(see :doc:`dev_guide_techniques_ring_edge`).

The measured behavior on this axis stays deliberately uncomfortable, and the
``orbit_error_ringlet`` scene pins it: a navigable ringlet rendered 2.5 px
outward of its catalog orbit navigates to a success about 3 px wrong. A
uniform radial misplacement has no exact translational equivalent, so the
robust edge fit down-weights one arc, locks onto one side of the annulus,
and absorbs the ephemeris error into the recovered offset instead of leaving
it in the residuals -- the declared sigma prices that hazard (the widened
covariance holds the fused result out of the high tier) but cannot remove
the bias itself, which is what the scene's ``known_offset_error_px`` pin
keeps in view.

``navigable`` (default **false**) is the information boundary in miniature:
the filter drops non-navigable features from ``nav_params`` entirely, so the
rendered frame is full of ring structure -- sheets, gaps, wave trains -- the
navigator was never told exists, while the navigator's world contains only
the features flagged true. A surviving feature's flag is always true and
carries no hidden information. This is what makes a ring scene a false-lock
stress: the strongest edge in the image is frequently one the model does not
predict.

Azimuthal clutter, spokes, and moonlets
---------------------------------------

Two system-level truth blocks add non-navigable structure *crossing* the
features:

- ``azimuthal`` scales the emitted intensity only -- never tau, never the
  transmission screen -- because all three of its members are
  albedo/illumination phenomena, not material density: ``modulation`` (a
  low-frequency brightness asymmetry, the self-gravity-wake signature),
  ``shadow`` (a planet-shadow wedge -- a strong, sharp, non-radial edge
  crossing every feature it spans), and ``spokes`` (a seeded field of
  azimuthally sharp, radially broad wedges drawn from the
  ``scene_radiance/ring_system/spokes`` stream; negative contrast gives the
  dark low-phase appearance). Stars behind a spoke are therefore *not*
  extra-attenuated: the material is unchanged, only its brightness is.
- ``moonlets`` embeds opaque discs at the ring's depth, each optionally
  carving a stylized ``propeller`` -- two partial-gap tau lobes straddling
  the moonlet radially and azimuthally. A moonlet is a blob/star confounder
  sitting exactly on the navigable features; the propeller adds the paired
  edge disturbance around it.

The catalog scenes pin the intended outcome: ``spoked_sheet`` (six dark
spokes crossing a navigable band) and ``moonlet_propeller`` (an Encke-style
non-navigable gap plus a bright propeller moonlet at its center) both
recover the planted offset from the catalog edges alone, with the clutter
present and unmodeled.

.. _sim-perf-budget:

The render performance budget
=============================

A 512x512 scene with a whole-scene PSF plus the full detector stack at
oversample 4 must render in under 2 s single-core, and a 1024x1024
Cassini-class scene in under 8 s (``tests/integration/test_sim_perf.py``).
The harness holds two scene families to those budgets: a star field (the PSF
convolution on the oversampled grid and the electron chain) and a frame
dominated by a large lit body with limb relief (the topographic body
renderer's split-resolution path -- detector-grid shading upsampled under an
oversampled silhouette -- plus the capped terminator shadow march). The
budget is a *cold-render* budget:
the render caches are cleared so the timed render pays the kernel-build and
compile costs a first render pays, while one-time non-render costs (the lazy
config-YAML load) are paid by an untimed warm-up first. The harness pins
itself: it sets the process CPU affinity to one core and caps every
BLAS/OpenMP pool to one thread via ``threadpoolctl`` for the duration, so an
unpinned numpy FFT cannot silently multithread and fake the budget. The
assertion reads CPU time on the pinned core -- far less load-sensitive than
wall time, though heavy memory-bandwidth contention can still inflate it by
roughly 10-25%, and by 40% or more while a parallel test battery saturates
every core -- and takes the best of up to three cold attempts, passing as
soon as one meets the budget. That absorbs transient contention but not the
sustained kind, so ``scripts/run-all-checks.sh`` excludes the budget file
from its parallel pytest run and executes it as a dedicated serial step
afterwards; run it the same way when measuring by hand. A breach across all
attempts on an otherwise-quiet host is reported and investigated, not
blessed by raising the budget.

Scene ingredients
=================

The panels below are rendered by ``python -m tests.integration.sim_doc_images``
(see :ref:`sim-png-export`); each isolates one ingredient.

.. figure:: _sim_images/ellipsoid_body.png
   :width: 45%
   :align: center

   Ellipsoidal body (Lambertian, moderate phase). ``axis1`` is the vertical
   extent, ``axis2`` the horizontal.

.. figure:: _sim_images/mesh_body.png
   :width: 45%
   :align: center

   Irregular polyhedral-mesh body of the same axes at a three-axis pose.

.. figure:: _sim_images/mesh_body_gouraud.png
   :width: 45%
   :align: center

   The same mesh body under ``gouraud`` smooth shading, for comparison with the
   flat-shaded panel above. The navigator's predicted mesh keeps flat shading.

.. figure:: _sim_images/body_craters.png
   :width: 45%
   :align: center

   Ellipsoid with procedurally generated craters.

.. figure:: _sim_images/crescent_body.png
   :width: 45%
   :align: center

   High-phase (130 deg) mesh body rendered as a thin lit crescent.

.. figure:: _sim_images/topographic_limb.png
   :width: 45%
   :align: center

   A lit ellipsoid with a limb-relief field: the silhouette is visibly ragged
   and relief pockets shadow near the limb, while the disc shading stays smooth.

.. figure:: _sim_images/ragged_terminator.png
   :width: 45%
   :align: center

   A high-phase crescent with limb relief: the terminator turns ragged and the
   relief march casts shadows that grow toward it.

.. figure:: _sim_images/haze_limb_body.png
   :width: 45%
   :align: center

   An atmospheric body (see :ref:`sim-atmosphere`): the haze softens the limb
   into an exponential ramp that reaches past the geometric edge, and the
   terminator brightens instead of cutting off. The navigator still predicts a
   hard limb at the reference radius.

.. figure:: _sim_images/haze_crescent_horns.png
   :width: 45%
   :align: center

   The same haze at phase 150 deg: forward scattering lights the whole limb and
   extends the crescent horns past the terminator into a near-complete ring of
   light.

.. figure:: _sim_images/banded_transit.png
   :width: 45%
   :align: center

   A banded planet disc with a storm oval, a bright transiting moon, and its
   cast shadow -- the sharp circular false crater the disc techniques must not
   chase.

.. figure:: _sim_images/mutual_event.png
   :width: 45%
   :align: center

   A mutual event: the nearer body occludes part of the farther one. The
   navigator drops the hidden arc from the farther body's predicted limb and
   trims its disc template, so each feature describes only what the image shows.

.. figure:: _sim_images/rings.png
   :width: 45%
   :align: center

   Two ring_system ringlets (the outer one eccentric) lit by the
   single-scattering tau photometry.

.. figure:: _sim_images/ring_edge_wave_gap.png
   :width: 45%
   :align: center

   An inclined ring system (B = 50 deg, rotated node): a gap carved into a
   sheet, its edges scalloped by a satellite edge wave downstream of the
   perturbing moon's longitude.

.. figure:: _sim_images/ring_mmode.png
   :width: 45%
   :align: center

   A resonantly forced m = 2 ringlet under inclined projection -- the
   two-lobed outer-B-ring edge shape, part of the catalog orbit both sides
   evaluate.

.. figure:: _sim_images/ring_spokes.png
   :width: 45%
   :align: center

   Dark seeded spokes crossing a navigable band: azimuthally sharp,
   radially broad intensity clutter the navigator is never told about.

.. figure:: _sim_images/ring_moonlet_propeller.png
   :width: 45%
   :align: center

   A bright moonlet with a propeller tau disturbance at the center of a
   non-navigable gap -- a blob/star confounder sitting exactly on the ring.

.. figure:: _sim_images/star_field.png
   :width: 45%
   :align: center

   A star field of flux-normalized point sources spread across the frame.

.. figure:: _sim_images/multi_body.png
   :width: 45%
   :align: center

   Multiple bodies (ellipsoid and mesh) at different sizes, depth-ordered by
   ``range_km``.

.. figure:: _sim_images/body_and_stars.png
   :width: 45%
   :align: center

   A body against a scattered star field.

.. figure:: _sim_images/detector_noise.png
   :width: 45%
   :align: center

   Detector model: read + shot noise, sparse cosmic-ray spikes (bright) and
   missing-data dropouts (dark).

.. figure:: _sim_images/telemetry_loss.png
   :width: 45%
   :align: center

   Structured telemetry loss: whole missing lines and partial-line dropouts
   across the disc.

.. figure:: _sim_images/stray_light_gradient.png
   :width: 45%
   :align: center

   A linear stray-light gradient behind a body.

.. figure:: _sim_images/composite_scene.png
   :width: 45%
   :align: center

   A composite frame: a mesh moon, a ring, and a star field.

.. _sim-scene-formats:

Scene format (schema version 2)
===============================

A scene is a single YAML file whose fields are the flat runtime ``sim_params``
names the renderer consumes, so a validated scene file *is* the ``sim_params``
dict with no translation layer. The current ``schema_version`` is **2**; the
validator rejects any other version, and any file that turns up with an older
version is converted, not accommodated.

The scene catalog (:mod:`spindoctor.sim.scene`) is the durable test artifact, laid out
as ``tests/integration/sim_scenes/<scene_class>/<scene_name>.yaml`` (the
directory is the registry). :func:`spindoctor.sim.scene.load_sim_scene` parses and
validates a file and returns the flat ``sim_params`` dict the renderer consumes;
the GUI's "Save Scene (YAML)" / "Load Scene (YAML)" buttons read and write the
same format via :func:`spindoctor.sim.scene.save_sim_scene`; and programmatic
scene authors -- the calibration-campaign generator
``util/calibration/scene_gen.py`` and the doc-gallery definitions in
``tests/integration/sim_doc_images.py`` -- validate their in-memory dicts
through the same core, :func:`spindoctor.sim.scene.validate_sim_params`.
Validation is strict: an unknown key at the top level or inside a ``bodies`` /
``stars`` / ``ring_system`` entry fails loudly, so a typo cannot silently
render the default scene.

The scene classes (for example ``algorithmic_invariants``,
``phase_sweep_regular_body``, ``phase_sweep_irregular_body``, ``range_sweep``,
``noise_sweep``, ``multi_body_geometry``, ``regression``, ``artifact_sweep``,
``star_confounder``, ``ring_system``, ``expected_fail``) scope what each scene
is testing and are
enforced by the structural test. The scene README at
``tests/integration/sim_scenes/README.txt`` documents the schema alongside the
code.

A complete YAML scene -- a noisy Cassini NAC frame with one irregular mesh body, a
ring, a couple of stars, and a planted offset the navigator must recover -- reads:

.. code-block:: yaml

   schema_version: 2
   scene_name: example_scene
   instrument: coiss_nac
   size_v: 220
   size_u: 220
   random_seed: 42
   exposure_sec: 1.0
   bodies:
     - name: HYPERION
       shape_model: polyhedral_mesh
       mesh_lumpiness: 0.4
       mesh_seed: 7
       pose_euler_deg: [10.0, 35.0, 0.0]
       center_v: 110.0
       center_u: 110.0
       axis1: 150.0
       axis2: 110.0
       axis3: 95.0
       illumination_angle: 25.0
       phase_angle: 40.0
   ring_system:
     geometry:
       center_v: 110.0
       center_u: 110.0
       opening_deg_obs: 90.0
       opening_deg_sun: 90.0
       node_deg: 0.0
     features:
       - name: RINGLET
         kind: ringlet
         tau: 1.2
         width: 8.0
         navigable: true
         orbit: {a: 90.0, ae: 6.0}
   sky_counts: {density_factor: 40.0}
   stars:
     - {name: S1, v: 30.0, u: 60.0, vmag: 6.0}
     - {name: S2, v: 180.0, u: 150.0, vmag: 7.5}
   noise:
     poisson: true
     read_noise_dn: 4.0
   offset_v: 1.43
   offset_u: -0.61

The ``schema_version`` and ``scene_name`` keys are metadata the renderer
ignores; ``scene_name`` must equal the filename stem. Every other key is a flat
``sim_params`` field consumed directly by the renderer.

**Boundary classification.** Every key carries an information-boundary side
(see :ref:`sim-information-boundary`): *idealized* keys reach the navigator
through ``obs.nav_params``; *truth* keys are readable only by the image-side
renderer. The tables below state each key's side. A key added to the schema
without a classification fails the import-time completeness assertion in
:mod:`spindoctor.sim.scene`, so every schema change must extend the boundary in
the same change.

.. _sim-pixel-convention:

Every position is a pixel corner
--------------------------------

**Every position that places something in a scene is a pixel corner**: a star's
``v`` / ``u``, a body's ``center_v`` / ``center_u``, and the ring system's
``geometry.center_v`` / ``center_u``. Integer ``N`` is the boundary between
pixel ``N - 1`` and pixel ``N``, so the centre of pixel ``N`` is ``N + 0.5`` and
the centre of a ``size_v`` by ``size_u`` frame is ``(size_v / 2, size_u / 2)``.

That is the ``oops`` uv convention, and it is the one the whole geometry stack
underneath already speaks: an FOV, a backplane, a C-matrix, and the ``v`` / ``u``
a star record carries all agree with it. A position written in a scene, a
position handed to ``oops``, and a position recorded in a star record are
therefore one number. The half pixel is added or removed in exactly one kind of
place -- where something is deposited into or read out of an array cell -- so a
scene author never applies it.

Displacements carry no datum and so need no such care: the planted ``offset_v``
/ ``offset_u``, a star's ``move_v`` / ``move_u`` smear vector, a planted
``catalog_error_v`` / ``catalog_error_u``, and a companion's ``sep_px`` are all
differences between two positions.

The two centres inside the ``optics`` block -- ``distortion.center_v`` /
``center_u`` and ``stray_light.center_v`` / ``center_u`` -- are pixel indices
instead. They name where a whole-frame field is centred rather than where an
object sits, and both fields are smooth on the scale of a pixel.

Scene parameter reference
=========================

Top-level fields
----------------

.. list-table::
   :widths: 22 11 11 10 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Side
     - Meaning
   * - ``size_v`` / ``size_u``
     - int
     - required
     - idealized
     - Image height and width in pixels.
   * - ``instrument``
     - str
     - required
     - idealized
     - Emulated instrument; selects the detector model and PSF (see
       :ref:`Instruments <sim-instruments>`).
   * - ``random_seed``
     - int
     - required
     - truth
     - Scene seed; per-stage and per-effect sub-seeds derive from it.
   * - ``exposure_sec``
     - float
     - 1.0
     - idealized
     - Exposure time; scales the cosmic-ray count.
   * - ``offset_v`` / ``offset_u``
     - float
     - 0.0
     - truth
     - Planted pointing offset (px) applied to all bodies, rings, and stars.
   * - ``offset_rotation_deg``
     - float
     - 0.0
     - truth
     - Planted boresight roll (deg) applied about the image center.
   * - ``midtime_utc``
     - str
     - none
     - idealized
     - Observation midtime the snapshot reports (label knowledge).
   * - ``closest_planet``
     - str
     - none
     - idealized
     - Planet context the snapshot reports.
   * - ``time`` / ``ring_epoch``
     - float
     - 0.0
     - idealized
     - Observation time and orbit epoch for ring-edge precession
       (``rate_peri`` applies across ``time - ring_epoch``); catalog knowledge
       both sides read.
   * - ``fit_camera_rotation``
     - bool
     - none
     - idealized
     - Overrides the instrument's camera-rotation-fit setting for this scene.
   * - ``bodies``
     - list
     - ``[]``
     - idealized
     - Per-body parameter dicts (see :ref:`sim-body-params`); some per-body
       keys are truth-side.
   * - ``ring_system``
     - dict
     - none
     - idealized
     - The optical-depth ring system (see :ref:`sim-ring-system`); its
       ``azimuthal`` / ``moonlets`` blocks and some per-feature keys are
       truth-side.
   * - ``stars``
     - list
     - ``[]``
     - idealized
     - Explicit star dicts (see :ref:`sim-star-params`); the ``psf_sigma``,
       ``catalog_error_v`` / ``catalog_error_u``, ``companion``, and
       ``delta_mag`` per-star keys are truth-side.
   * - ``sky_counts``
     - map
     - none
     - truth
     - Background-sky star field: ``a`` and ``b`` set the cumulative count law
       ``log10 N(<m) = a + b*m`` per square degree, ``density_factor`` scales
       the count (1 is mid galactic latitude), and ``diffuse_e_per_px`` adds an
       optional flat diffuse-sky floor in the detector's native point-source
       unit (electrons per pixel on a CCD, DN per pixel on the Voyager
       vidicon). The sky stars are contaminants: the navigator receives no
       catalog for them, and they render through the same flux/point-mass path
       as catalog stars.
   * - ``star_catalog_scatter_px``
     - float
     - none
     - truth
     - Scene-level per-star position-scatter sigma (px): every rendered star is
       displaced off its catalog position by a seeded Gaussian draw of this
       sigma, added to any explicit per-star ``catalog_error_*`` (see
       :ref:`sim-star-params`).
   * - ``noise``
     - dict
     - off
     - truth
     - Detector-noise block (see :ref:`sim-noise`).
   * - ``oversample``
     - int
     - auto
     - truth
     - Pins the render oversampling factor; omit to let the renderer choose (4
       with an active PSF, else 1).
   * - ``optics``
     - dict
     - off
     - truth
     - Optical-path block: PSF, smear, distortion, ghosts, stray light (see
       :ref:`sim-optics`).
   * - ``spk_error``
     - dict
     - off
     - truth
     - Planted spacecraft-ephemeris parallax error (see :ref:`sim-spk-error`).
   * - ``detector``
     - dict
     - off
     - truth
     - Detector-chain override: gain state, detector model, exposure reference
       (see :ref:`sim-detector-block`).
   * - ``artifacts``
     - dict
     - off
     - truth
     - Physical-chain opt-in switch (see :ref:`sim-artifacts-block`).
   * - ``instrument_config``
     - dict
     - none
     - idealized
     - Per-instrument config overrides (see :ref:`sim-instrument-config`).
   * - ``expected``
     - map
     - none
     - test-only
     - The scene's declared navigation outcome (``status``,
       ``confidence_tier``, ``status_reason``), asserted by the integration
       suite and read by neither the renderer nor the navigator (see
       :ref:`sim-expected`).

.. _sim-body-params:

Body parameters
---------------

Each entry of ``bodies`` is a dict. The geometry, pose, lighting, and physical
scale are idealized (the published shape model of a body is catalog knowledge;
a scene plants shape error through ``nav_override``, not by hiding the mesh).
Common fields:

.. list-table::
   :widths: 22 11 11 10 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Side
     - Meaning
   * - ``name``
     - str
     - generated
     - idealized
     - Body label used in metadata and annotations.
   * - ``center_v`` / ``center_u``
     - float
     - frame center
     - idealized
     - Body center in pixels, as a pixel corner (see
       :ref:`sim-pixel-convention`).
   * - ``axis1`` / ``axis2`` / ``axis3``
     - float
     - 0.0
     - idealized
     - Full extents of the three body axes in pixels (``axis3`` defaults to
       ``min(axis1, axis2)``).
   * - ``illumination_angle``
     - float (deg)
     - 0.0
     - idealized
     - Image-plane light azimuth (0 = from the top).
   * - ``phase_angle``
     - float (deg)
     - 0.0
     - idealized
     - Phase angle (0 = fully lit, 180 = back-lit crescent).
   * - ``range_km``
     - float
     - body index
     - idealized
     - Subject distance in km; also the depth-ordering key (smaller renders
       in front).
   * - ``km_per_pixel``
     - float
     - none
     - idealized
     - Optional physical scale at the limb.
   * - ``shape_model``
     - str
     - ``ellipsoid``
     - idealized
     - ``ellipsoid`` or ``polyhedral_mesh``.

Ellipsoid bodies add ``rotation_z`` and ``rotation_tilt`` (degrees, idealized).
Mesh bodies (``shape_model: polyhedral_mesh``) instead read the idealized mesh
keys ``mesh_lumpiness`` (relief amplitude as a fraction of the unit radius,
default 0.3), ``mesh_seed`` (which irregular shape is generated, default 0),
``mesh_n_lat`` / ``mesh_n_lon`` (mesh resolution, defaults 16 / 32),
``mesh_detail_octaves`` (higher-frequency shape-mode banks, default 0), and
``pose_euler_deg`` (intrinsic X, Y, Z Euler angles, default ``[0, 0, 0]``).

The truth-side body keys never reach the navigator. The relief, photometry,
texture, and transit keys are consumed by the topographic path and the mesh
extras by the mesh renderer (see :ref:`sim-body-renderer`); the navigator's
predicted body is always the smooth Lambert template.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Field (truth)
     - Meaning
   * - ``crater_fill``, ``crater_min_radius``, ``crater_max_radius``,
       ``crater_power_law_exponent``, ``crater_relief_scale``
     - Procedural crater terrain -- nature's surface, which the navigator's
       smooth predicted template does not know.
   * - ``limb_relief_rms``, ``limb_relief_corr_deg``
     - The limb/terminator relief field: RMS of the limb slice and its
       correlation length in degrees of surface arc. Ragged silhouette plus
       terminator march shadows (ellipsoid and mesh bodies).
   * - ``photometric_law``, ``minnaert_k``
     - The surface-scattering law (``lambert`` / ``lommel_seeliger`` /
       ``minnaert`` / ``lunar_lambert``) and its Minnaert exponent; ellipsoid
       topographic path.
   * - ``opposition_surge``
     - ``{amplitude, width_deg}`` -- the normalized brightness surge near
       opposition.
   * - ``albedo_texture``
     - ``{rms, corr_px, spots}`` -- a multiplicative noise field plus circular
       albedo spots on the disc shading.
   * - ``disc_texture``
     - ``{band_amplitude, band_wavenumber, band_phase_deg, storms}`` -- a
       giant-planet latitude band pattern plus storm ovals.
   * - ``transits``
     - A list of ``{moon, shadow}`` discs (each ``dv_px``, ``du_px``,
       ``radius_px``, and a brightness/darkness) on the parent disc; the cast
       shadow is the sharp *false crater* the disc techniques must not chase.
   * - ``shading``
     - Rendered mesh shading mode (``flat`` default, ``gouraud`` smooth); the
       predicted mesh always renders flat.
   * - ``pose_scatter``
     - ``{sigma_deg}`` -- a seeded per-frame Gaussian perturbation of the
       *rendered* mesh pose only; the navigator predicts the catalog pose.
   * - ``seed``
     - The crater / texture / relief realization (a stable hash of the body
       geometry when absent).
   * - ``anti_aliasing``
     - Image-side rendering-fidelity knob; the predicted template always
       renders at full anti-aliasing.
   * - ``nav_override``
     - Mapping overlaid on the body's *idealized view* by the boundary
       filter, then dropped (see below). Only idealized body keys may appear
       in it.

.. note::

   **Axis convention.** Both body renderers use the same mapping: ``axis1`` is
   the vertical (``v``) extent and ``axis2`` the horizontal (``u``) extent. An
   ellipsoid and a mesh declared with identical ``axis1``/``axis2`` are oriented
   the same way, so the two are interchangeable for the same geometry.

**Render geometry vs navigation geometry.** By default the navigator predicts
the body from the same idealized parameters the renderer drew, so it knows the
geometry exactly. A body may carry an optional ``nav_override`` mapping that
the boundary filter (:func:`spindoctor.sim.scene.build_nav_params`) overlays
onto the body's idealized view before the navigator sees it -- the renderer
ignores it and always draws the true geometry, and the overridden true values
never cross. This separates the *render geometry* (ground truth) from the
*navigation geometry* (what the navigator assumes): the channel the
irregular-body scenarios use to render a lumpy mesh but predict its smooth
(ellipsoidal) limit, or to predict the same body at a disagreeing pose. The
override never moves the center, so the planted offset is still recoverable.
See :doc:`dev_guide_navigation_models_body_simulated`.

.. _sim-ring-system:

The ring system
---------------

``ring_system`` is a single mapping describing an optical-depth ring system:
a shared projection geometry, a list of radial tau features, and truth-side
azimuthal / moonlet clutter. The renderer draws the whole system; the
navigator is told only about the features flagged ``navigable``. This is the
key-by-key reference; :ref:`sim-ring-renderer` describes the renderer's
mechanics and the measured navigation behavior.

**Geometry and projection.** The required ``geometry`` block carries
``center_v`` / ``center_u`` (the projected ring center), ``opening_deg_obs``
and ``opening_deg_sun`` (the observer and sun ring opening angles B, in
(-90, 90], positive north; both required, and an exactly edge-on 0 renders
nothing), and ``node_deg`` (the sky position angle of the ascending node,
counterclockwise from +u toward -v). The system is rendered in ring-plane
coordinates and projected through B and the node, so an inclined view
produces the foreshortened ellipse geometry, radial resolution gradient, and
near/far-arm asymmetry real ring images have; ``|B| = 90`` reduces to
sky-plane circles. Ring-plane longitude is measured from the ascending node
*in the ring plane*: every orbital angle lives in that frame, and the node
angle enters only the final sky rotation. Both sides project through the
same helpers (:mod:`spindoctor.sim.ring_geometry`), so predicted edges land
in projected positions by construction.

**Photometry and compositing.** Brightness derives from the composed tau and
the viewing/lighting geometry via the single-scattering closed forms with a
one-term Henyey-Greenstein phase function evaluated at the block's
``phase_deg``; the unlit side (opening angles of opposite sign) produces the
real inversion, moderate-tau features bright from the dark side and high-tau
features nearly black. The system composites over the body stack as a
per-pixel transmission screen (``img = I_ring + exp(-tau/mu) * img_behind``)
ordered by observer distance: with the block-level ``range_km`` and
``km_per_pixel`` set, the near arm crosses in front of a body at the ring
center's range and the far arm passes behind it, gaps reveal the background
instead of erasing it, and stars behind the ring attenuate by
``exp(-tau/mu)`` (a scene whose system overlaps a body must range both).

**Features.** Each entry of ``features`` carries ``name``, ``kind``, ``tau``
(the normal optical depth), kind-specific shape keys, a catalog ``orbit``,
and ``navigable`` (default false). The kinds and their shape keys:

- ``ringlet`` — ``tau`` between the orbit radius and orbit + ``width``
  (required); a ``gap`` is the same band as a tau suppression.
- ``edge`` — a one-sided step; ``side: 'in'`` (default) carries tau inside
  the orbit radius (a sheet bounded by its outer edge), ``'out'`` outside.
- ``ramp`` — a linear transition across ``width``; ``side: 'out'`` (default)
  rises from 0 at the orbit radius to ``tau`` at orbit + ``width``.
- ``wave`` — a damped radial sinusoid launched at the orbit radius
  (``wavelength`` and ``damping``, both required, in radial px): a
  density-wave train, typically non-navigable clutter riding on a sheet.

The ``orbit`` map is the catalog orbit: the mode-1 precessing ellipse
(``a``, ``ae``, ``long_peri``, ``rate_peri``, applied across the scene-level
``time`` minus ``ring_epoch``), optional m >= 2 ``modes``
(``{m, amp, peri}``: ``r = a - amp*cos(m*(lam - peri))``, the resonantly
forced edge shape), and an optional satellite ``edge_wave``
(``{amp, wavelength, damp, lam0}``: a damped sinusoid downstream of the
perturbing moon's longitude ``lam0``, with ``damp`` in radians and the sine
argument arc length over ``wavelength``).

**Planted orbit error.** A feature's truth-side ``orbit_error`` map
(``delta_a_px``, ``delta_ae_px``, ``delta_long_peri_deg``) displaces the
*rendered* feature while the navigator predicts from the catalog orbit, so
features are misplaced relative to the model exactly as real ring features
are relative to their published orbit solutions -- the ring analog of the
body ephemeris-error axis. The idealized ``declared_orbit_sigma``
(``sigma_a_px``, ``sigma_ae_px``, ``sigma_long_peri_deg``) is the
uncertainty the navigator is entitled to know; the drawn error values never
cross the boundary. Per-feature ``albedo`` and ``phase_g`` (the
Henyey-Greenstein asymmetry; negative backscatters, positive
forward-scatters) are photometric truth.

**Azimuthal clutter and moonlets (truth).** The ``azimuthal`` block scales
the emitted intensity only, never tau: ``modulation``
(``{amplitude, m, phase_deg}``, the self-gravity-wake asymmetry), ``shadow``
(``{start_deg, extent_deg, darkness}``, a planet-shadow wedge -- a strong
non-navigable edge crossing every feature), and ``spokes``
(``{count, r_inner, r_outer, contrast, width_deg}``, a seeded field of
azimuthally sharp, radially broad wedges drawn from the
``scene_radiance/ring_system/spokes`` stream; negative contrast for the dark
low-phase appearance). The ``moonlets`` list embeds opaque discs at the
ring's depth (``{a, lam_deg, radius_px, amplitude}``), each optionally
carrying a stylized ``propeller`` tau disturbance
(``{length_deg, width_px, contrast}``): blob/star confounders sitting
exactly on the navigable features.

.. _sim-star-params:

Star parameters
---------------

Each entry of ``stars`` is a dict with ``name`` and an optional
``catalog_name``, a ``v`` / ``u`` position (a pixel corner, like every other
scene position -- see :ref:`sim-pixel-convention`), a ``vmag`` (visual
magnitude; lower is brighter), an optional ``spectral_class``, an optional
per-star smear vector
``move_v`` / ``move_u``, an optional PSF fitting-window size ``psf_size`` (a
two-integer list), and an optional ``navigable`` flag. All of those are
idealized -- they are the catalog and instrument knowledge a real pipeline has.
``navigable`` defaults to true when absent; setting it to ``false`` renders the
star but drops it from the navigator's filtered view entirely, so a *surviving*
star's flag is always true and carries no hidden truth (which is why it is
idealized, not truth).

The truth-side star keys never reach the navigator:

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - Field (truth)
     - Meaning
   * - ``psf_sigma``
     - A per-star PSF width override -- an anomaly of the rendered image. The
       navigator knows only the instrument's published PSF.
   * - ``catalog_error_v``, ``catalog_error_u``
     - Displace the RENDERED star (px) off the catalog position the navigator
       predicts from -- unrecoverable astrometric residual, added on top of the
       scene-level ``star_catalog_scatter_px`` draw.
   * - ``companion``
     - An unresolved binary: a second point source at ``sep_px`` along
       ``angle_deg``, ``delta_mag`` fainter. The blended photocenter sits off
       the catalog position by a magnitude-weighted amount -- a physical catalog
       error the navigator cannot know.
   * - ``delta_mag``
     - Renders a variable star at a brightness other than its cataloged
       ``vmag`` (positive is fainter); the catalog ``vmag`` stays what the
       navigator sees.

**Information asymmetry.** The navigator predicts each surviving star from its
idealized catalog ``v`` / ``u`` and ``vmag``; the renderer draws it at those
values *plus* whatever truth keys the scene planted -- a position error, a
photocenter-pulling companion, a brightness delta -- and never tells the
navigator. A star technique must lock the pointing from the navigable subset
despite that residual, and the ``star_confounder`` and ``expected_fail`` scene
classes (see :ref:`sim-star-confounder`) push the residual and the clutter past
where it can.

A background-sky star field is added by the truth-side top-level ``sky_counts``
map: star counts are drawn from the cumulative law ``log10 N(<m) = a + b*m`` per
square degree (interim mid-galactic-latitude values ``a = -3.1``, ``b = 0.34``),
scaled by the frame's field of view and the ``density_factor`` multiplier, down
to a faint cutoff, and rendered through the same flux/point-mass path as catalog
stars. An optional ``diffuse_e_per_px`` adds a flat diffuse-sky floor,
expressed in the detector's native point-source unit (electrons per pixel on a
CCD, DN per pixel on the Voyager vidicon, like the plane it adds to). The sky
stars carry no catalog, so to the navigator they are pure confounders.

Every star -- catalog or sky -- is a flux-normalized point source: its total
signal is ``zero_point * 10**(-0.4 * vmag) * exposure_sec`` (the per-instrument
photometric zero point, in electrons for a CCD or DN for the vidicon), deposited
as a sub-pixel-positioned point mass in ``frame.point_e`` (centroid-exact after
the downsample). The whole-scene optics PSF is the *only* convolution a star
receives, so the rendered star profile is the scene kernel and its peak follows
from the PSF. Pre-spreading a star would convolve it twice and widen it by
sqrt(2). Sky stars follow the same rule, and the star hit-test metadata records
the scene kernel's core sigma. A scene with no PSF renders each star as a
one-pixel spike (the undersampled limit), so the converted floor scenes carry an
explicit ``optics.psf`` block.

The faint sky-count integral is truncated at the magnitude where a star's
matched-filter signal drops to the read-noise floor over its PSF core
(:func:`spindoctor.sim.forward.star.faint_sky_cutoff_mag`): fainter draws add
nothing above the noise. That cutoff is an image-side rendering economy and is
distinct from the navigator's own published-config detection limit
(:meth:`~spindoctor.obs.obs_inst_sim.ObsSim.star_max_usable_vmag`), which is
derived from the emulated instrument's configuration and bounds which stars the
navigator will *use*, never from the scene's truth-side blocks.

.. _sim-noise:

Detector-noise block
--------------------

The optional ``noise`` dict (truth-side) pins the truth-side detector noise the
scene plants. The detector stage consumes most of it; the missing-data markers
are applied by the telemetry stage. Each field is off at the floor unless the
block sets it (or ``artifacts.instrument_defaults`` turns on the catalog
chain); an explicit noise key always wins over the catalog value. The
validator checks the block against exactly this inventory, so an unknown noise
key fails loudly.

.. list-table::
   :widths: 30 12 12 46
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``poisson``
     - bool
     - False (floor)
     - Apply Poisson shot noise to the electron image. On under
       ``instrument_defaults`` unless the block sets it false.
   * - ``read_noise_dn``
     - float
     - 0 (off)
     - Gaussian read-noise sigma in DN (converted to electrons through the
       resolved gain). ``instrument_defaults`` supplies the catalog
       electrons value instead.
   * - ``bias_dn``
     - float
     - instrument
     - Additive bias pedestal; lifts dark sky off zero so it is not confused with
       the missing-data marker.
   * - ``cosmic_ray_rate_per_sec``
     - float
     - 0.0
     - Cosmic-ray fluence (events / cm^2 / sec), scaled by ``exposure_sec``.
       Stays 0 under ``instrument_defaults`` (a loss mode, not physical-chain
       noise).
   * - ``missing_data_rate``
     - float
     - 0.0
     - Fraction of pixels (0-1) set to the missing-data marker (telemetry
       stage). Stays 0 under ``instrument_defaults``.
   * - ``bloom_length``
     - int
     - 0 (off)
     - Full-well column-bloom half-length in pixels; ``instrument_defaults``
       supplies the catalog value.
   * - ``signal_full_scale_frac``
     - float
     - instrument
     - Well fraction a signal of 1.0 fills at the reference exposure.
   * - ``pixel_area_cm2``
     - float
     - 1.0
     - Detector pixel area; scales the cosmic-ray count.
   * - ``dark_current_e_per_sec``
     - float
     - 0 (off)
     - Dark-current pedestal rate in electrons / sec, added pre-Poisson.
   * - ``hot_pixel_fraction``
     - float
     - 0 (off)
     - Fraction of pixels that are hot (a fixed per-seed population).
   * - ``hot_pixel_amplitude_e``
     - float
     - catalog
     - Hot-pixel charge scale in electrons (exponentially distributed).
   * - ``hot_pixel_column_factor``
     - float
     - catalog
     - Fraction of a hot pixel's TOTAL charge bled up its column (the warm
       streak's integral, frame-size-invariant).
   * - ``banding_amplitude_e``
     - float
     - 0 (off)
     - Coherent horizontal-banding amplitude in electrons.
   * - ``banding_period_px``
     - float
     - catalog
     - Banding spatial period along the row axis, in pixels.
   * - ``bias_pedestal_sigma_dn``
     - float
     - 0 (off)
     - Per-image bias-pedestal jitter sigma (DN).
   * - ``bias_row_gradient_dn`` / ``bias_col_gradient_dn``
     - float
     - 0 (off)
     - Peak-to-peak low-order bias gradients (DN).
   * - ``vidicon``
     - dict
     - catalog
     - Vidicon DN-noise sub-map (vidicon path only): ``read_noise_line_dn``,
       ``read_noise_pixel_dn``, ``coherent_amplitude_dn``,
       ``coherent_period_px``.

Physical saturation caps the electron image at the full well before gain (with
``bloom_length`` spreading the excess along the column); the ADC clip at
``saturation_dn`` applies after quantization.

.. _sim-optics:

Optics block
------------

The optional ``optics`` dict (truth-side) carries the optical-path sub-blocks
the optics stage applies in its fixed internal order (see
:ref:`sim-optics-stage`). Every sub-block is optional and absent by default; a
sub-block that is present but a true no-op renders bit-identically to one that
is absent.

``optics.psf`` -- the whole-scene point-spread function. Either the explicit
core-plus-wing form or the exclusive navigator-matched form:

.. list-table::
   :widths: 24 12 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``sigma_v`` / ``sigma_u``
     - float
     - required
     - Gaussian core sigma along v / u, in detector pixels.
   * - ``w``
     - float
     - 0.0
     - Moffat wing energy fraction, in [0, 1].
   * - ``r0``
     - float
     - 2.0
     - Moffat core radius in detector pixels.
   * - ``n``
     - float
     - 3.0
     - Moffat index.
   * - ``match_navigator``
     - bool
     - --
     - Exclusive alternative: resolve the PSF to the navigator's own Gaussian
       at the instrument ``star_psf_sigma`` (no other PSF key may appear).

``optics.smear`` -- a list of per-object-class motion-smear entries. Each entry
has ``dv_px`` / ``du_px`` (drift in pixels) and ``object_class`` (one of
``all``, ``stars``, ``bodies``, ``rings``; default ``all``). One ``all`` entry
smears the whole scene; several give differential smear.

``optics.distortion`` -- the residual geometric distortion:

.. list-table::
   :widths: 24 12 12 52
   :header-rows: 1

   * - Field
     - Type
     - Default
     - Meaning
   * - ``k1`` / ``k2``
     - float
     - 0.0
     - Radial polynomial coefficients about the optical centre.
   * - ``center_v`` / ``center_u``
     - float
     - frame centre
     - Optical-centre position in pixels.
   * - ``nonradial_rms_px``
     - float
     - 0.0
     - RMS amplitude of the seeded smooth non-radial wander, in pixels.

``optics.ghosts`` -- a list of ghost reflections. Each entry has ``dv_px`` /
``du_px`` (offset in pixels), ``amplitude`` (fraction; 0 disables the entry),
and ``defocus_sigma`` (blur sigma in pixels).

.. _sim-stray-light:

``optics.stray_light`` -- the smooth scattered-light field applied last, before
the detector stage: ``amplitude`` (peak fraction of full scale; 0 disables it),
``direction_deg`` (ramp direction for the ``linear`` model), ``model``
(``linear`` ramp or ``radial`` bump), and ``center_v`` / ``center_u`` (the
radial-model bump centre; omit for the frame centre). It exercises the
navigator's source-image background filter.

.. _sim-spk-error:

Spacecraft-ephemeris error block
--------------------------------

The optional ``spk_error`` dict (truth-side) plants a spacecraft-ephemeris
parallax error: ``dv_px`` / ``du_px`` (the displacement at the reference range)
and ``reference_range_km`` (the range the displacement is quoted at). The
parallax displacement scales as ``reference_range_km / range_km`` per object, so
a scene that sets ``spk_error`` must give every body and ring a physical
``range_km`` -- the validator enforces this.

.. _sim-detector-block:

Detector block
--------------

The optional ``detector`` dict (truth-side) overrides the resolved detector
chain: ``gain_state`` (the electron-chain gain state, which must be catalogued
for the instrument), ``detector_model`` (``ccd`` electron chain or ``vidicon``
DN chain), ``exposure_ref_sec`` (the exposure the signal full-scale fraction
references), and ``quantization`` (the ADC sub-mode: ``exact``, ``8bit``,
``uneven_12bit``, or ``sqrt_lut``). Omitted keys fall back to the instrument's
catalog defaults.

.. _sim-artifacts-block:

Artifacts block
---------------

The optional ``artifacts`` dict (truth-side) carries the physical-chain opt-in
switch ``instrument_defaults`` (see :ref:`sim-artifacts-catalog`), the
``adversarial`` placement flag, and one map per **artifact mode**. With
``instrument_defaults`` on, the emulated camera's catalog PSF, distortion
residual, and detector noise chain render at their per-instrument values; the
per-mode incidences stay at zero (naming an instrument selects a signal chain,
not a set of defects), with the one exception that LORRI turns on
``frame_transfer_smear`` -- its defining artifact -- at the catalog's nominal
scrub/transfer times. Leaving ``instrument_defaults`` off keeps those keys
absent (the self-consistency floor).

The artifact modes are registered once in
:mod:`spindoctor.sim.forward.artifact_modes` -- the single source of truth for
each mode's rendering stage (``detector`` or ``telemetry``), per-instrument
availability, and parameter schema. Every mode carries an ``incidence``: for a
count mode it is the expected number of events per frame (drawn Poisson); for a
commanded / periodic mode it is the per-frame probability the mode fires. An
incidence of 0 is a no-op, so a scene plants exactly the defects it names. A
mode named on an instrument it is unavailable on fails validation (for example
``hot_pixels`` and ``bloom`` on LORRI, which has neither). Each applied mode
records its realized geometry into ``frame.truth['artifacts'][mode]`` for the
later planted-versus-measured incidence match.

**Detector / electronics modes** render inside the detector stage at the
physically right point in the unit chain (see :ref:`sim-detector-stage`):
``banding_coherent`` (the Cassini 2 Hz / Galileo 42-px / LORRI striping family,
horizontal and/or vertical, with an optional mid-image frequency and dark-level
step), ``bias_structure`` (per-image pedestal plus low-order gradients),
``dark_ramp`` (a readout dark gradient or a shutter exposure shading),
``bloom`` (electron-domain column bleed), ``radiation_transients`` (the
morphological cosmic model scaled to the Galileo regime by an environment
factor and readout dwell), ``bright_dark_pairs`` (Cassini anti-blooming vertical
pairs), ``frame_transfer_smear`` and ``serial_tail`` (LORRI column pedestal and
saturation undershoot tail), ``beam_bend`` and ``residual_image`` (Voyager
brightness-dependent limb bias and erase-cycle ghost), ``fixed_pattern``
(stitch combs, vignetting, dust donuts, jail bars, PRNU), and the quantization
modes ``quantization_lut``, ``quantization_ls8b``, and ``contouring_8bit``. The
mechanics live in
:mod:`spindoctor.sim.forward.detector.electronics_stages`; an explicit mode
wins over the generic noise-block knob for the same mechanic (banding, bias,
bloom, quantization), the precedence ``hot_pixels`` already uses.

**Telemetry-artifact modes** flank the structured loss loop in the telemetry
stage (:mod:`spindoctor.sim.forward.telemetry_artifacts`):
``compression_dct`` runs first (the lossy 8x8 DCT blockiness a codec plants
before packets drop, leaving any commanded ``truth_window`` clean), and the
Voyager GEOMED archive-processing scars run last -- ``reseau_scars``
(reseau-removal smudges on the ~46-px lattice) then ``resample_texture`` (the
GEOMED resample warp, blank border, and missing-line interpolation banding).

**Adversarial placement.** With ``adversarial: true`` the stochastic modes place
their events preferentially on the navigation features (a limb arc, a ring edge,
a star) rather than uniformly, turning the artifact sweep from average-case into
worst-case. Deterministic-shape modes (banding, fixed pattern, frame-transfer
smear, the quantization modes) are unaffected by the flag -- their shape does not
depend on where the features are.

Per-instrument shape defaults for every mode live in ``artifacts_catalog.py``
(see :ref:`sim-artifacts-catalog`); a scene inherits them by naming a mode with
only its incidence, and overrides any parameter it spells out. The realized
geometry each mode records is read back by
:mod:`spindoctor.sim.forward.incidence` for the planted-versus-measured match;
the conceptual overview, the full mode tables, and the ``artifact_sweep`` scene
class are in :ref:`sim-artifact-framework`.

.. _sim-instrument-config:

Instrument-config overrides
---------------------------

The optional ``instrument_config`` dict is deep-merged over the resolved
per-instrument block, so a scene can pin individual settings (PSF sigma, noise,
data units, extfov margin) without tracking later camera-config edits. Omit it to
inherit everything; name ``generic`` and override everything to fully
self-specify. The top-level ``noise`` block still wins over
``instrument_config.noise`` for rendering. See :doc:`dev_guide_observations`.

.. _sim-catalog-workflow:

Catalog, baselines, and the render-diff sheet
=============================================

The scene catalog is guarded at three levels, all regenerated inside whichever
change alters rendered output:

- **Structural tests** (``tests/integration/test_sim_scenes.py``): every scene
  validates, sits in a declared class, has a unique name, and renders.
- **Regression baselines** (``tests/integration/sim_baselines/``): every
  catalog scene re-navigates to its recorded rounded outcome
  (``test_sim_baselines.py``, integration tier); regenerate with
  ``python -m tests.integration.update_sim_baselines`` and review the diff.
- **The render-diff contact sheet**
  (``python -m tests.integration.render_contact_sheet``): one
  before / after / amplified-absolute-difference panel per scene, composed into one
  sheet PNG per scene class under ``tests/integration/render_diffs/``, with
  the per-scene current renders committed under ``render_diffs/current/`` (the
  *before* images of the next regeneration). Any change that alters rendered
  output regenerates the sheets and the ``current/`` PNGs in the same change,
  and the sheet is reviewed panel by panel. The review criterion: *the scene
  still renders what it asks for -- same ingredients, same geometry, same
  planted truth; differences confined to discretization and reseeding.* This
  review exists because a converted scene can recover its planted offset and
  still be rendering the wrong thing; the recovered offset cannot catch that,
  only eyes on the render can.

  The regeneration rule is enforced mechanically by
  ``tests/integration/test_render_diffs.py`` (integration tier): every
  catalog scene must have a committed ``current/`` PNG byte-identical to a
  fresh render, every scene class must have a committed sheet, and neither
  may outlive its scene or class. The sheets themselves are enforced for
  presence and coverage only -- their pixels encode the previous baseline's
  before-panels and PIL's environment-dependent label rasterization, so
  their *content* remains a review artifact, not a byte-compared one.

The two documentation galleries (``docs/dev_guide/_sim_images/`` and
``docs/simulator_report/_scene_images/``, both written by
``python -m tests.integration.sim_doc_images``) are re-rendered under the same
rule -- and the rule is enforced mechanically:
``tests/integration/test_sim_doc_images.py`` (integration tier) regenerates
every gallery image into a temporary directory and fails, naming the stale
files and the regeneration command, whenever a committed PNG no longer
matches a fresh render.

The simulated-image GUI
=======================

``sd_create_simulated_image`` is a PyQt6 editor for the same parameters, with a
live preview. Launch it with:

.. code-block:: bash

   sd_create_simulated_image

The editor is the :mod:`spindoctor.cli.sim_editor` package: a single
``QMainWindow`` (``CreateSimulatedImageModel``) composed from one mixin per
schema block, so each block's controls -- and any control group a later
realism addition needs -- live in their own module:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Controls / role
   * - ``main_window.py``
     - Assembles the mixins; owns the cross-cutting scaffolding (data model,
       render scheduling, window layout).
   * - ``global_fields.py``
     - Image size, planted offset and camera roll, exposure, seed, instrument
       selector, camera-rotation-fit override, midtime, closest planet.
   * - ``noise.py``
     - Poisson, read noise, bias, cosmic-ray rate, missing-data rate, bloom,
       signal full-scale fraction, pixel area (General tab).
   * - ``optics_tab.py``
     - The Optics tab: PSF, motion smear, distortion, ghosts, the stray-light
       panel, oversample, and the spacecraft-ephemeris error group. Each
       optical sub-block is a checkable group under absent-key discipline.
   * - ``artifacts_tab.py``
     - The Artifacts tab: the instrument-defaults switch and the detector
       override (gain state, detector model, exposure reference, quantization)
       under per-key discipline -- an edit writes only its own key, so
       unedited keys keep tracking the instrument catalog.
   * - ``stray_light.py``
     - Stray-light amplitude, direction, model, radial center (its panel is
       hosted in the Optics tab's stray-light group).
   * - ``background_stars.py``
     - Background-sky ``sky_counts`` controls (density factor, count-law a and
       b, diffuse floor) behind a whole-block enable checkbox under absent-key
       discipline, plus the scene-level ``star_catalog_scatter_px`` control
       under the same discipline.
   * - ``expected_outcome.py``
     - The test-only ``expected`` block: a checkable Expected-outcome group on
       the General tab (status, confidence tier, status reason).
   * - ``body_tab.py`` / ``body_appearance.py`` / ``ring_tab.py`` /
       ``ring_advanced.py`` / ``star_tab.py``
     - The per-object tabs. The body tab carries geometry, shape model and mesh
       controls, crater controls, and the navigation-override group
       (``body_tab.py``); the truth-side appearance groups -- limb relief,
       photometric law and opposition surge, albedo texture and its spots, disc
       texture and its storms, transits, and the mesh-only shading / detail /
       pose-scatter extras (enabled only for a mesh body) -- live in
       ``body_appearance.py``, each under absent-key discipline. Ring tabs (one
       per ``ring_system`` feature) carry the feature's kind and kind-specific
       shape keys, navigability, optical depth, mode-1 catalog orbit, and
       photometric truth scalars, with the shared projection geometry, phase
       angle, and physical range on the first feature's tab (``ring_tab.py``);
       the orbit's m-mode rows and satellite edge-wave group, the planted
       orbit-error and declared-sigma groups, and the system-level azimuthal
       (modulation / shadow / spokes) and moonlet-list (with propeller) truth
       blocks live in ``ring_advanced.py``, under the same absent-key and
       list-row disciplines. Star tabs carry position, magnitude, PSF, smear,
       catalog label, and the per-star information-asymmetry controls (navigable
       flag, catalog error, companion, variable-brightness delta).
   * - ``tabs.py``
     - Object-tab lifecycle (the ``+`` tab, ordering, rebuild).
   * - ``scene_io.py``
     - Load / Save Scene (YAML) via :mod:`spindoctor.sim.scene`.
   * - ``render_display.py``
     - The live preview (render, stretch, zoom, saturation overlay).
   * - ``widgets.py`` / ``base.py``
     - Shared widget helpers and the mixin-facing protocol.

The GUI exposes the full scene parameter surface, so any scene that can be
written by hand in YAML can also be built in the GUI. Two exceptions are
carried through unedited rather than widget-exposed: the nested
``instrument_config`` override mapping, and the hand-authored
``known_offset_error_px`` / ``known_offset_error_tol_px`` pins in the
expected block (the expected-outcome panel preserves them across its
enable toggle but offers no controls for them; they are baseline-measured
values, authored in YAML). Scenes
round-trip through the **Load / Save Scene (YAML)** buttons, so a scene rendered
in the GUI can be saved as a catalog artifact and a catalog scene can be loaded
back to edit; ``tests/main/test_sim_editor_round_trip.py`` asserts the
round-trip is loss-free over the full key inventory. The GUI is one peer, not
the sole control surface; the YAML and the Python API are equally
authoritative.

Running navigation on a simulated image
=======================================

A simulated image is navigated through the same pipeline as a real frame, via the
``sim`` dataset. With a saved YAML scene file:

.. code-block:: bash

   sd_offset sim /path/to/scene.yaml

The ``sim`` dataset (``DataSetSim``) builds an :class:`~spindoctor.obs.obs_inst_sim.ObsSim`
that loads the scene via :func:`spindoctor.sim.scene.load_sim_scene`, renders the
frame through the forward model, and exposes the navigator the filtered
idealized scene view ``obs.nav_params`` (see
:ref:`sim-information-boundary`). The model-selection layer routes a simulated
obs to the simulated NavModels -- ``NavModelBodySimulated``,
``NavModelRingsSimulated``, ``NavModelStarsSimulated`` -- which build one
feature set per body / ring / star field from that filtered view, while the
SPICE-backed models decline a simulated obs. From there the same techniques run
and produce the same ``NavResult``. In tests the scene is usually driven
directly:

.. code-block:: python

    ObsSim.from_file(path, sim_params=load_sim_scene(path))

When ``sim_params`` is supplied the scene file is never read, so ``path`` is
only the name the observation carries: it may name a file that does not exist,
and no storage is touched to resolve it. A harness holding a scene it built in
memory can therefore pass a pure label such as ``sim://limb_bias``.

.. _sim-png-export:

Exporting viewable PNGs
=======================

The renderer emits detector counts whose absolute range depends on the instrument
and on cosmic-ray spikes, so :mod:`spindoctor.sim.png_export` stretches a DN image to a
viewable grayscale PNG with a percentile clip (a few hot pixels do not crush the
signal) and an optional gamma that lifts dim features such as a crescent or a
faint star field:

.. code-block:: python

   from spindoctor.sim.png_export import render_scene_png
   render_scene_png(sim_params, 'frame.png', gamma=1.4, upscale=2)

Three tools build on it. The sweep runner can dump every frame behind a response
curve for inspection:

.. code-block:: bash

   python -m tests.integration.sim_sweep_runner --dump-images out/ --only phase_regular_body

writes one PNG per sweep step under ``out/<sweep_name>/``. The documentation-image
generator rebuilds the galleries in this chapter and the scene images in the
sensitivity report:

.. code-block:: bash

   python -m tests.integration.sim_doc_images

Both galleries carry a ``NOTES.md`` describing how to regenerate them after a
rendering change. And the render-diff contact sheet
(:ref:`sim-catalog-workflow`) stretches every catalog scene into its committed
review panels:

.. code-block:: bash

   python -m tests.integration.render_contact_sheet

.. _sim-realism-match:

The realism match against real cohorts
======================================

Everything above renders what a scene *asks for*; the realism match measures
how closely those renders resemble real spacecraft frames.  The runner

.. code-block:: bash

   python -m tests.integration.sim_realism

compares the curated real-image library
(``tests/integration/image_library/``) against matched simulated frames, per
instrument, on seven figures of merit:

1. **Sky noise** -- per-patch noise sigma from paired pixel differences
   inside near-uniform patches (science frames have no flat-field pairs, and
   naive signal-binning would count scene texture as noise), the sky-patch
   level above the frame floor, noise at signal, and the sky spatial power
   spectrum (which catches banding and coherent noise a scalar sigma cannot).
2. **Star PSF** -- star-cutout radial profiles and encircled-energy radii
   (EE50/EE80) on cohorts that contain star frames.  Where a cohort has no
   star frames, the limb profile of figure 3 is the only PSF evidence and
   the summary says so.
3. **Limb profiles** -- intensity profiles sampled along the outward normals
   of the navigator's predicted limb polylines (shifted onto the actual limb
   by each frame's operator-verified offset), reduced to 10-90% rise widths
   and binned by phase angle and apparent body diameter.  This is what
   ``BodyLimbNav``'s distance transform actually sees.  Absolute rise-width
   medians are estimator-specific (the 10-90% estimator biases roughly +17%
   at the widest profiles and -6% at the narrowest); both sides run the
   identical estimator, so estimator parity -- not the absolute median --
   is what makes the comparison fair.  One asymmetry survives parity:
   real-side profile centers inherit the sidecar offset's finite
   uncertainty and per-star catalog/distortion residuals, while sim-side
   centers are exact by construction, so misregistration inflates only
   the real widths and part of any tuned PSF wing may absorb
   registration error rather than optics.
4. **Ring edges** -- the same profile machinery along predicted ring-edge
   polylines (Cassini cohort).
5. **Dynamic range** -- fraction saturated, fraction near the frame floor,
   and signal-percentile stretch, always inside exposure strata: an
   unstratified comparison measures what the spacecraft pointed at, not the
   forward model.
6. **Artifact incidence** -- measured rates of lost/interpolated lines and
   single-pixel spikes, with a cross-frame split of spikes into stationary
   (hot pixels, which recur at fixed detector positions) and transient
   (cosmic rays), compared against the artifacts-catalog defaults.
7. **Technique diagnostics** -- for a handful of matched real/sim pairs the
   full navigator runs on both frames and the per-technique diagnostics
   (inlier counts, residual scatter, confidences) are tabulated side by
   side.

Matched frames
--------------

Every real frame gets exactly one matched simulated frame: the same
instrument signal chain (``artifacts: {instrument_defaults: true}`` on the
instrument whose ``data_units`` match the cohort's products -- e.g.
``coiss_calib_nac`` for the Cassini CALIB cohort), the same exposure, and
scene content of the same class -- a star field for a star frame, an
ellipsoid at the real body's apparent diameter and phase angle for a limb
frame, a multi-feature ring system for a ring frame, near-empty sky for the
negative and scattered-light frames.  Scene seeds derive from the real
frame's image id, so the whole match is deterministic: re-running the runner
reproduces every simulated frame and every statistic bit for bit.

Both sides then run *identical* extraction code -- the same patch finder,
the same profile sampler, and the navigator's own feature extraction
(``NavOrchestrator.prepare``) to place limb, ring, and star geometry -- so
estimator bias cancels in the comparison.

The divergence statistic
------------------------

Each sample kind is reduced to one scalar: the Wasserstein-1 distance
between the real and sim distributions, computed on quantile-clipped data
(each sample winsorized at its own 1st/99th percentiles) and normalized by
the real distribution's interquartile range
(:func:`spindoctor.sim.realism.divergence.w1_divergence`).  W1 is a
transport metric in the variable's own units -- a normalized value of 1.0
means the distributions are displaced by about one real IQR -- but it is
not outlier-robust, so the clip keeps a noise statistic from silently
measuring an artifact statistic's tails.  **No pass/fail threshold is
attached**; the number is reported per figure of merit and judged by a
human against the cohort's support.

Curve kinds (the sky power spectrum and the star / limb / ring
profiles) get the same scalar through
:func:`spindoctor.sim.realism.divergence.w1_between_densities`: the
frame-averaged curve on each side is treated as a nonnegative density
over its axis, and the W1 along that axis (normalized by the real
density's IQR) lands in the summary's ``curve_divergences`` block.  The
pooled FOM 3 scalar (``limb_width_copop``) pools only strata populated
on both sides; one-sided strata are recorded in the summary
(``limb_bins_real_only`` / ``limb_bins_sim_only``) rather than folded
into a mixture whose pooled W1 would measure stratum composition
instead of width.

Cohort support labels
---------------------

A distribution statement needs frames.  Each figure of merit carries a
support label per instrument -- ``supported`` (8+ contributing frames),
``limited`` (2-7, reported with the caveat), or ``unsupported`` (fewer; the
statistic is not reported, because an IQR of two frames is not a
statistic).  Where a figure of merit is unsupported and no independent
evidence exists, that instrument's sim accuracy is *bounded by unverified
forward-model fidelity* and the report says exactly that rather than
implying a match that was never measured.

The FOM 7 firewall
------------------

Figures of merit 1-6 are computed from pixels and geometry alone -- they do
not depend on how well the navigator performs.  Figure 7 is different: it is
built from the navigator's own outputs.  If the forward model were tuned
until the navigator's diagnostics agreed between sim and real, agreement
would stop being evidence -- any systematic error shared by tuning target
and measurement instrument becomes invisible, and the sim would "validate"
the navigator by construction.  Figure 7 is therefore **read-only**: it is
reported so a human can see whether the techniques behave comparably on
matched frames, and it is never a tuning target.  Only FOMs 1-6 feed the
catalog-tuning loop.

Running and reading
-------------------

The runner needs the holdings and star-catalog environment variables of any
real navigation run.  ``--instrument`` restricts to one instrument,
``--max-frames`` caps the cohort (smoke runs), ``--skip-fom7`` skips the
navigation pairs.  Outputs:

- ``docs/simulator_report/_figures/realism_<instrument>_<fom>.png`` --
  the overlay figures embedded in the simulator report;
- ``tests/integration/realism_results/realism_summary.json`` -- the
  committed record backing the report's tables (divergences, support
  labels, FOM 7 rows, runtime).

The full-cohort run (75 frames as of 2026-07-18) takes 19 to 21 minutes
on a workstation with local holdings (measured 18.8-21.3 minutes over
the recorded 2026-07-17/18 runs); the cost is dominated by the
navigator's feature extraction on the real frames and the full-size
matched renders.  ``--skip-fom7`` saves about 17% -- the navigation
pairs are a handful of frames per instrument, not a large share of the
cost.  ``tests/integration/test_sim_realism.py`` carries
the integration-tier tests: matched-scene validity per instrument,
sim-side determinism, and an end-to-end smoke on the smallest cohort.

The per-instrument match quality, the figures, and the known-gaps list
live in the simulator report's realism section
(:doc:`/simulator_report/simulator_report`).  Tuned catalog defaults in
``sim/forward/artifacts_catalog.py`` carry their provenance next to the
value: which cohort statistic moved them, measured when, replacing what.

See also
========

- :doc:`dev_guide_navigation_models_body_simulated`,
  :doc:`dev_guide_navigation_models_ring_simulated`,
  :doc:`dev_guide_navigation_models_star_simulated` -- the navigator side.
- :doc:`dev_guide_testing` -- the test kinds the simulator drives, including
  the information-boundary test.
- :doc:`/simulator_report/simulator_report` -- the sensitivity and
  algorithmic-invariant results.
- API reference: :doc:`/api_reference/api_sim`.
