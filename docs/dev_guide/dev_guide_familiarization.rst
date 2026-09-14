==========================
Code Familiarization Plan
==========================

This is a study plan for a developer who has just cloned the repository and
wants to come up to speed on SpinDoctor in the most efficient order. Each stage
lists documentation pages to read and source modules to inspect alongside
them. The earlier stages are prerequisites for the later ones; the later
stages can be read more selectively once the core pipeline makes sense.

The plan does not explain the code — it sequences it. Read each
documentation page first, then look at the corresponding source. Every class,
function, and module name below is a hyperlink into the API reference; click
through to jump straight to the autodoc page (and from there to the
``[source]`` link to the file itself).

Stage 1 — Orientation
=====================

Goal: understand what the package does, the repository layout, and the
top-level architecture. Skim only — do not try to absorb every detail.

- ``README.md`` — PyPI-facing summary of what ``rms-spindoctor`` is and the
  installation / quick-start commands.
- :doc:`dev_guide_introduction` — environment variables, CLI entry points,
  test / lint / docs commands, CI / release process.
- :doc:`dev_guide_navigation_overview` — the six-subsystem architectural
  sketch and the per-image data flow.
- :doc:`dev_guide_class_hierarchy` — the inheritance graph and the shared
  base classes the rest of the manual assumes.

Stage 2 — Cross-cutting foundations
===================================

Goal: understand the shared infrastructure every subsystem depends on
(configuration, logging, the universal base class, common helpers).

- :doc:`dev_guide_config_and_static_data` — how YAML defaults are layered,
  the :class:`~spindoctor.config.config.Config` singleton, and the static-data
  citation policy.
- :doc:`dev_guide_logging` — ``pdslogger`` usage, section conventions, and
  the rule against bare ``print``.
- :mod:`spindoctor.config` — config loader and module-level logger:

  - :mod:`spindoctor.config.config`
  - :mod:`spindoctor.config.config_helper`
  - :mod:`spindoctor.config.logger`

- ``src/spindoctor/config_files/`` — the bundled YAML defaults (numeric-prefix load
  order); not autodoc'd since it is data, not code.
- :mod:`spindoctor.support` — cross-cutting helpers. Start with these two; the
  rest can be browsed on demand:

  - :mod:`spindoctor.support.nav_base` —
    :class:`~spindoctor.support.nav_base.NavBase`, the shared base providing
    :attr:`~spindoctor.support.nav_base.NavBase.config` and
    :attr:`~spindoctor.support.nav_base.NavBase.logger`.
  - :mod:`spindoctor.support.types`

Stage 3 — Inputs: datasets, observations, features
==================================================

Goal: understand how an image goes from a file path on disk to an
in-memory ``Observation`` (from ``oops``) carrying the per-pixel features
that the rest of the pipeline consumes.

- :doc:`dev_guide_observations` — :class:`~spindoctor.dataset.dataset.DataSet`
  enumeration, the :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst`
  per-instrument hierarchy, and the
  :class:`~spindoctor.feature.feature.NavFeature` dataclass / feature-type
  taxonomy.
- :doc:`instruments/instruments` — one chapter per instrument, each carrying
  that instrument's code map, host call, label and index dependencies,
  configuration block, calibration anchors, frame identities and open items.
  Read the chapter for whichever instrument you are working on alongside the
  module list below.
- :mod:`spindoctor.dataset` — file enumeration. The per-mission subclasses
  share the :class:`~spindoctor.dataset.dataset.DataSet` /
  :class:`~spindoctor.dataset.dataset_pds3.DataSetPDS3` bases:

  - :mod:`spindoctor.dataset.dataset_pds3_cassini_iss`
  - :mod:`spindoctor.dataset.dataset_pds3_voyager_iss`
  - :mod:`spindoctor.dataset.dataset_pds3_galileo_ssi`
  - :mod:`spindoctor.dataset.dataset_pds3_newhorizons_lorri`
  - :mod:`spindoctor.dataset.dataset_pds4`

- :mod:`spindoctor.obs` — observation wrappers. The per-instrument
  :class:`~spindoctor.obs.obs_inst.ObsInst` subclasses share the
  :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` base:

  - :mod:`spindoctor.obs.obs_inst_cassini_iss`
  - :mod:`spindoctor.obs.obs_inst_voyager_iss`
  - :mod:`spindoctor.obs.obs_inst_galileo_ssi`
  - :mod:`spindoctor.obs.obs_inst_newhorizons_lorri`

- :mod:`spindoctor.feature`:

  - :mod:`spindoctor.feature.feature` — the :class:`~spindoctor.feature.feature.NavFeature`
    dataclass itself.
  - :mod:`spindoctor.feature.feature_type` — the feature-type taxonomy.
  - :mod:`spindoctor.feature.geometry`
  - :mod:`spindoctor.feature.flags`
  - :mod:`spindoctor.feature.composition`
  - :mod:`spindoctor.feature.reliability`
  - :mod:`spindoctor.feature.constants`

Stage 4 — First end-to-end pipeline: ``NavModelBody`` + ``BodyLimbNav``
=======================================================================

Goal: walk one image end-to-end through the simplest possible
model / technique pair, plus the minimum slice of orchestrator,
annotations, and driver code needed to produce the JSON metadata and
summary PNG. Body-limb navigation is the canonical reference
distance-transform pipeline and the smallest of the autonomous
techniques.

Read in this order:

Model
-----

- :doc:`dev_guide_navigation_models` — what a
  :class:`~spindoctor.nav_model.nav_model.NavModel` is and the three methods
  every subclass implements:

  - :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model`
  - :meth:`~spindoctor.nav_model.nav_model.NavModel.to_features`
  - :meth:`~spindoctor.nav_model.nav_model.NavModel.to_annotations`

- :doc:`dev_guide_navigation_models_bodies` — the body-model family.
- :doc:`dev_guide_navigation_models_body` —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` in detail.
- :mod:`spindoctor.nav_model.nav_model` — abstract base, registry, and
  :func:`~spindoctor.nav_model.nav_model.build_models_for_obs`.
- :mod:`spindoctor.nav_model.body_shape` — body shape support.
- :mod:`spindoctor.nav_model.nav_model_body_base` — shared body-annotation
  helpers (:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`).
- :mod:`spindoctor.nav_model.nav_model_body` — the catalog-driven body model
  (:class:`~spindoctor.nav_model.nav_model_body.NavModelBody`).

Technique
---------

- :doc:`dev_guide_techniques` — what a
  :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` is, the
  registry, and the prior-free / prior-required two-pass split.
- :doc:`dev_guide_techniques_image_derivatives` — gradient and
  edge-distance-transform images shared by every DT technique.
- :doc:`dev_guide_techniques_dt_fitting` — the coarse-NCC + Levenberg-
  Marquardt + Tukey biweight machinery the limb / terminator / ring-edge
  techniques all share.
- :doc:`dev_guide_techniques_body_limb` — the body-limb technique itself.
- :mod:`spindoctor.nav_orchestrator.image_derivatives` — the per-image
  derivative builder.
- :mod:`spindoctor.nav_technique.nav_technique` — abstract base + registry
  (:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`).
- :mod:`spindoctor.nav_technique.dt_fitting` — shared DT-fitting helpers.
- :mod:`spindoctor.support.distance_transform` — the edge distance-transform
  builder those helpers consume.
- :mod:`spindoctor.nav_technique.nav_technique_body_limb` — the technique
  implementation
  (:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`).

Orchestrator (minimum slice)
----------------------------

- :doc:`dev_guide_orchestrator` — the orchestrator package map.
- :doc:`dev_guide_orchestrator_nav_context` — the per-image dataclass
  every model and technique reads / writes
  (:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`).
- :doc:`dev_guide_orchestrator_orchestrator` —
  :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator`: the
  two-pass driver loop. (Skim — feasibility / confidence / ensemble are
  covered in Stage 5.)
- :doc:`dev_guide_orchestrator_nav_result` — the final per-image result
  envelope (:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`).
- :mod:`spindoctor.nav_orchestrator` — read these three first; the rest are
  picked up in Stage 5:

  - :mod:`spindoctor.nav_orchestrator.nav_context`
  - :mod:`spindoctor.nav_orchestrator.orchestrator`
  - :mod:`spindoctor.nav_orchestrator.nav_result`

Output: annotations, metadata, summary PNG, top-level driver
------------------------------------------------------------

- :doc:`dev_guide_annotations` — the summary-PNG overlay system.
- :doc:`dev_guide_orchestrator_curator` — projecting
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` into the JSON
  metadata block.
- :mod:`spindoctor.annotation` — annotation primitives, plus combine logic
  (:meth:`~spindoctor.annotation.annotations.Annotations.combine`):

  - :mod:`spindoctor.annotation.annotation`
  - :mod:`spindoctor.annotation.annotations`
  - :mod:`spindoctor.annotation.annotation_text_info`

- :mod:`spindoctor.nav_orchestrator.curator` — the metadata-dict builder
  (:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict`).
- :mod:`spindoctor.navigate_image_files` — the top-level per-image driver
  called by ``sd_offset``
  (:func:`~spindoctor.navigate_image_files.navigate_image_files`).
- ``src/spindoctor/cli/sd_offset.py`` — the CLI dispatch module that wires it up
  (the ``src/spindoctor/cli/`` tree is not part of the autodoc API surface).

Stage 5 — Orchestrator round-trip: feasibility, confidence, diagnostics, ensemble
=================================================================================

Goal: understand how individual technique results are gated, calibrated,
classified, and combined into the single per-image answer the
orchestrator returns.

- :doc:`dev_guide_techniques_feasibility` — when a technique declines to
  run and how the reason is reported.
- :doc:`dev_guide_techniques_confidence` — calibrated [0, 1] confidence
  per technique.
- :doc:`dev_guide_techniques_diagnostics` — the typed diagnostics
  dataclass attached to each technique result.
- :doc:`dev_guide_orchestrator_ensemble` — the
  :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` combine that reconciles
  per-technique results into the final offset.
- :doc:`dev_guide_orchestrator_image_classifier` — per-image
  classification feeding feasibility / weighting.
- :doc:`dev_guide_orchestrator_instrument_config` — per-instrument
  tunables the orchestrator threads through.
- :doc:`dev_guide_orchestrator_provenance` — the provenance record
  carried through the pipeline.
- :doc:`dev_guide_orchestrator_feature_summary` — the per-image
  feature-summary dataclass.
- Technique-side helpers:

  - :mod:`spindoctor.nav_technique.feasibility`
  - :mod:`spindoctor.nav_technique.confidence`
  - :mod:`spindoctor.nav_technique.confidence_config`
  - :mod:`spindoctor.nav_technique.diagnostics`
  - :mod:`spindoctor.nav_technique.technique_result`

- Orchestrator-side helpers:

  - :mod:`spindoctor.nav_orchestrator.ensemble`
  - :mod:`spindoctor.nav_orchestrator.ensemble_consensus` — consensus-subset
    selection and the corroborating-member rules.
  - :mod:`spindoctor.nav_orchestrator.ensemble_observability` — observable-
    subspace bases and the sentinel-aware pseudo-inverse.
  - :mod:`spindoctor.nav_orchestrator.image_classifier`
  - :mod:`spindoctor.nav_orchestrator.image_classifier_result`
  - :mod:`spindoctor.nav_orchestrator.instrument_config`
  - :mod:`spindoctor.nav_orchestrator.provenance`
  - :mod:`spindoctor.nav_orchestrator.feature_summary`
  - :mod:`spindoctor.nav_orchestrator.status_reason_info`

- :mod:`spindoctor.support.status_reason` — the shared status-reason enum.

Stage 6 — Remaining models and their techniques
================================================

Goal: extend the picture from Stage 4 to every other registered model
and every other technique. Each subsection pairs a model with the
techniques that consume its features, in roughly increasing order of
complexity.

Stars (model + star techniques)
-------------------------------

- :doc:`dev_guide_navigation_models_stars` — the star-model family.
- :doc:`dev_guide_navigation_models_star` —
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` in detail.
- :doc:`dev_guide_techniques_star_unique_match` — direct catalog-to-image
  unique matching.
- :doc:`dev_guide_techniques_star_refine` — refinement against an
  existing prior.
- :doc:`dev_guide_techniques_star_field` — full star-field correlation.
- :mod:`spindoctor.nav_model.stars` subpackage:

  - :mod:`spindoctor.nav_model.stars.catalog`
  - :mod:`spindoctor.nav_model.stars.detection`
  - :mod:`spindoctor.nav_model.stars.predicted_snr`
  - :mod:`spindoctor.nav_model.stars.smeared_psf`
  - :mod:`spindoctor.nav_model.stars.conflicts`
  - :mod:`spindoctor.nav_model.stars.nav_model_stars`
  - :mod:`spindoctor.nav_model.stars.nav_model_stars_simulated`

- :mod:`spindoctor.nav_technique.nav_technique_star_unique_match`
- :mod:`spindoctor.nav_technique.nav_technique_star_refine`
- :mod:`spindoctor.nav_technique.nav_technique_star_field`
- The shared ``_star_helpers`` module is internal and not part of the
  autodoc API.

Body (additional techniques) and Titan
--------------------------------------

- :doc:`dev_guide_techniques_body_terminator` — illuminated-body
  terminator fitting (DT pipeline).
- :doc:`dev_guide_techniques_body_blob` — small / unresolved body
  navigation.
- :doc:`dev_guide_techniques_body_disc` — full-disc correlation.
- :doc:`dev_guide_techniques_titan_haze` — the haze solar-symmetry fit.
- :doc:`dev_guide_navigation_models_titan` —
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` (the haze
  envelope geometry that fit consumes).
- :mod:`spindoctor.nav_technique.nav_technique_body_terminator`
- :mod:`spindoctor.nav_technique.nav_technique_body_blob`
- :mod:`spindoctor.nav_technique.nav_technique_body_disc`
- :mod:`spindoctor.support.correlate` — the shared pyramid-NCC / upsampled-DFT
  correlation engine behind the disc, blob, and annulus techniques.
- :mod:`spindoctor.nav_model.nav_model_titan`
- :mod:`spindoctor.nav_model.titan_geometry`
- :mod:`spindoctor.nav_technique.nav_technique_titan_haze`
- :mod:`spindoctor.nav_technique.titan_fitting` — the pure haze-fitting
  library the technique wraps.

Rings (model + ring techniques)
-------------------------------

- :doc:`dev_guide_navigation_models_rings` — the ring-model family.
- :doc:`dev_guide_navigation_models_ring` —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` in detail.
- :doc:`dev_guide_techniques_ring_edge` — the ring-edge DT technique.
- :doc:`dev_guide_techniques_ring_annulus` — the ring-annulus technique.
- :mod:`spindoctor.nav_model.nav_model_rings_base`
- :mod:`spindoctor.nav_model.nav_model_rings`
- :mod:`spindoctor.nav_model.rings` subpackage:

  - :mod:`spindoctor.nav_model.rings.ring_math`
  - :mod:`spindoctor.nav_model.rings.ring_filter`
  - :mod:`spindoctor.nav_model.rings.ring_render_context`
  - :mod:`spindoctor.nav_model.rings.ring_render_result`
  - :mod:`spindoctor.nav_model.rings.ring_types`
  - :mod:`spindoctor.nav_model.rings.ring_feature`

- :mod:`spindoctor.nav_technique.nav_technique_ring_edge`
- :mod:`spindoctor.nav_technique.nav_technique_ring_annulus`

Manual
------

- :doc:`dev_guide_techniques_manual` — the interactive PyQt6 driver.
- :mod:`spindoctor.nav_technique.nav_technique_manual`
- :mod:`spindoctor.ui.manual_nav_dialog`
- :mod:`spindoctor.ui.library_entry`
- The ``spindoctor.ui.common`` module is internal and not autodoc'd.

Stage 7 — Simulated images
==========================

Goal: understand the synthetic-image renderer and the simulated-image
sibling of each model family. These are used both by
``sd_create_simulated_image`` and by tests.

- :doc:`dev_guide_navigation_models_star_simulated`
- :doc:`dev_guide_navigation_models_body_simulated`
- :doc:`dev_guide_navigation_models_ring_simulated`
- :doc:`dev_guide_navigation_models_titan_simulated`
- :mod:`spindoctor.nav_model.stars.nav_model_stars_simulated`
- :mod:`spindoctor.nav_model.nav_model_body_simulated`
- :mod:`spindoctor.nav_model.nav_model_rings_simulated`
- :mod:`spindoctor.nav_model.nav_model_titan_simulated`
- :mod:`spindoctor.sim` — the scene schema, the image-side forward model, and the
  geometry helpers shared across the information boundary:

  - :mod:`spindoctor.sim.scene` — schema, validation, and the boundary filter
  - :mod:`spindoctor.sim.render` — the cached driver over the stage pipeline
  - :mod:`spindoctor.sim.forward` — the image-side stage pipeline
    (:mod:`~spindoctor.sim.forward.pipeline`, :mod:`~spindoctor.sim.forward.stages`,
    and the per-stage renderer modules)
  - :mod:`spindoctor.sim.ellipsoid_geometry`, :mod:`spindoctor.sim.mesh_geometry`,
    :mod:`spindoctor.sim.ring_geometry`, :mod:`spindoctor.sim.star_records` —
    geometry/record helpers both sides share
  - :mod:`spindoctor.sim.instruments`
  - :mod:`spindoctor.sim.seeds`
  - :mod:`spindoctor.sim.png_export`

- :mod:`spindoctor.nav_model.sim_body` and :mod:`spindoctor.nav_model.sim_ring` —
  the navigator-side predicted-template renderers the simulated NavModels drive.

- ``src/spindoctor/dataset/dataset_sim.py`` and ``src/spindoctor/obs/obs_inst_sim.py``
  — the simulated-image dataset and observation wrappers (not autodoc'd).

Stage 8 — Calibration and regression
====================================

Goal: understand how confidence calibration is anchored to a curated
cohort of real images, and how per-instrument camera-rotation correction
is calibrated.

- :doc:`dev_guide_image_library` — the operator-curated regression
  cohort (``tests/integration/image_library/``), sidecar schema, and
  baseline workflow.
- :doc:`dev_guide_rotation` — per-instrument camera-rotation correction.
- ``tests/integration/`` — the integration suite that consumes the image
  library.

Stage 9 — Downstream products
=============================

Goal: understand the products built on top of a navigated image —
mosaics / reprojections, per-pixel backplanes, and PDS4 bundles.

- :doc:`dev_guide_reprojection` — body and ring mosaicing plus the
  thread-safety constraints.
- :mod:`spindoctor.reproj` — reprojection core:

  - :mod:`spindoctor.reproj.bodies` — :class:`~spindoctor.reproj.bodies.BodyMosaic`.
  - :mod:`spindoctor.reproj.rings` — :class:`~spindoctor.reproj.rings.RingMosaic`.
  - :mod:`spindoctor.reproj.cartographic_model` —
    :func:`~spindoctor.reproj.cartographic_model.create_cartographic_model`.
  - :mod:`spindoctor.reproj.photometric_model`
  - :mod:`spindoctor.reproj.ring_orbit_model`

- ``src/spindoctor/cli/reproj/`` — CLI-only helpers shared by the mosaic drivers
  (not part of the importable ``spindoctor`` API and not autodoc'd).
- :doc:`dev_guide_backplanes` — per-pixel geometry product generation.
- ``src/spindoctor/cli/backplanes/`` — driven by ``sd_backplanes`` (not autodoc'd in
  the ``spindoctor`` API surface).
- :doc:`dev_guide_pds4` — PDS4 bundle assembly.
- ``src/spindoctor/cli/pds4/`` — driven by ``sd_create_bundle``; per-dataset hooks
  live on the :class:`~spindoctor.dataset.dataset.DataSet` subclasses
  introduced in Stage 3 (this tree is not autodoc'd either).

Stage 10 — Extending and conventions
====================================

Goal: be ready to make changes. Read these once at the end so the
conventions land on top of a working mental model of the code.

- :doc:`dev_guide_extending` — how to add a new instrument, dataset,
  model, or technique (the registry hooks each subsystem exposes).
- :doc:`dev_guide_best_practices` — line length, docstring style, ruff /
  mypy expectations, commit conventions.
- ``src/spindoctor/cli/`` — CLI dispatch modules for every entry point listed in
  ``[project.scripts]`` (not autodoc'd).
- The PyQt6 mosaic viewer; read after :mod:`spindoctor.ui.manual_nav_dialog`
  from Stage 6:

  - :mod:`spindoctor.ui.mosaic_viewer.projections`
  - :mod:`spindoctor.ui.mosaic_viewer.sphere_render`
  - :mod:`spindoctor.ui.mosaic_viewer.graticule`

- :doc:`/contributing` — the contributor checklist that gates every PR.
