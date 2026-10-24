===============
Class Hierarchy
===============

The autonomous-navigation pipeline is built around four cooperating
groups of classes — observation snapshots, predicted-scene models,
per-feature techniques, and the orchestrator that runs them. The
following Mermaid diagram captures the principal relationships; the
narrative below the diagram describes each group in turn.

.. mermaid::

   classDiagram
      direction RL

      class NavBase {
          +\_\_init\_\_(*, config=None, **kwargs)
          +config: Config
          +logger: PdsLogger
      }

      class DataSet {
          <<abstract>>
          +_img_name_valid(name)*
          +yield_image_files_from_arguments(args)*
          +yield_image_files_index(**kwargs)*
      }

      class Obs {
          <<abstract>>
      }

      class ObsSnapshot {
          +bp
          +ext_bp
          +ra_dec_limits_ext()
          +extfov_data_sensor_mask()
      }

      class ObsSnapshotInst {
          <<abstract>>
          +from_file(path, *, config=None, extfov_margin_vu=None)*
      }

      class NavModel {
          <<abstract>>
          +\_\_init\_\_(name, obs, *, config=None)
          +name
          +obs
          +metadata
          +create_model()*
          +to_features(context)* list[NavFeature]
          +to_annotations(context)* Annotations
          +instances_for_obs(obs)$ list[NavModel]
      }

      class NavModelStars {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelStarsSimulated {
          +create_model()
      }

      class NavModelBodyBase {
          <<abstract>>
          +_render_silhouette(...)
          +_create_edge_annotations(...)
      }

      class NavModelBody {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelBodySimulated {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelRingsBase {
          <<abstract>>
          +_create_edge_annotations(...)
      }

      class NavModelRings {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelRingsSimulated {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelTitan {
          +create_model()
          +to_features(context)
          +to_annotations(context)
      }

      class NavModelTitanSimulated {
          +instances_for_obs(obs)
          +geometry_inputs
      }

      class NavTechnique {
          <<abstract>>
          +name
          +accepts_feature_types
          +requires_prior
          +is_feasible(features)* NavFeasibilityReport
          +navigate(features, context)* NavTechniqueResult
      }

      class StarUniqueMatchNav
      class StarFieldFromCatalogNav
      class StarRefineNav
      class BodyLimbNav
      class BodyTerminatorNav
      class BodyDiscCorrelateNav
      class BodyBlobNav
      class RingEdgeNav
      class RingAnnulusNav
      class TitanHazeNav
      class NavTechniqueManual

      class NavOrchestrator {
          +\_\_init\_\_(models, *, config=None, only_models='*', only_techniques='*')
          +navigate(obs) NavResult
      }

      class NavFeature {
          <<frozen dataclass>>
          +feature_id: str
          +feature_type: NavFeatureType
          +geometry: NavFeatureGeometry
          +position_cov_px
          +preferred_filter: NavFilterSpec
          +reliability: float
          +flags: NavFeatureFlags
      }

      class NavTechniqueResult {
          <<frozen dataclass>>
          +technique_name
          +feature_ids
          +offset_px
          +covariance_px2
          +confidence
          +spurious / at_edge
          +source_bodies
          +prior_source_techniques
          +diagnostics: NavTechniqueDiagnostics
      }

      class NavResult {
          <<frozen dataclass>>
          +status
          +status_reason
          +offset_px / sigma_px
          +covariance_px2
          +confidence / confidence_rank
          +rotation_rad / sigma_rotation_rad
          +per_technique
          +consensus_techniques
          +excluded_from_consensus
          +feature_inventory
          +image_classifier
          +model_metadata
          +annotations
          +provenance
      }

      class NavContext {
          <<frozen dataclass>>
          +obs
          +image_ext
          +sensor_mask_ext
          +image_noise_sigma
          +saturation_mask_ext
          +cosmic_ray_mask_ext
          +image_classifier
          +image_gradient_ext
          +image_gradient_vu_ext
          +image_edge_dt_ext
          +prior_offset_px
          +prior_covariance_px2
          +fit_camera_rotation
          +provenance
      }

      class FeatureReliabilityGate {
          +thresholds: dict[NavFeatureType, float]
          +apply(features) tuple[list, list]
      }

      class InstrumentSettings {
          <<frozen dataclass>>
          +data_units
          +saturation_dn
          +marker_value
          +thresholds: ImageQualityThresholds
          +fit_camera_rotation
          +max_rotation_deg
      }

      class NavImageClassifier {
          +classify(image, *, settings) NavImageClassifierResult
      }

      class Annotations {
          +add_annotations(other)
          +combine(offset_px) NDArray
      }

      class Annotation {
          <<abstract>>
          +color
          +avoid_mask
          +draw(image)*
      }

      NavBase <|-- DataSet
      NavBase <|-- Obs
      NavBase <|-- NavModel
      NavBase <|-- NavTechnique
      NavBase <|-- NavOrchestrator

      Obs <|-- ObsSnapshot
      ObsSnapshot <|-- ObsSnapshotInst

      NavModel <|-- NavModelStars
      NavModel <|-- NavModelBodyBase
      NavModel <|-- NavModelRingsBase
      NavModel <|-- NavModelTitan

      NavModelStars <|-- NavModelStarsSimulated
      NavModelBodyBase <|-- NavModelBody
      NavModelBodyBase <|-- NavModelBodySimulated
      NavModelRingsBase <|-- NavModelRings
      NavModelRingsBase <|-- NavModelRingsSimulated
      NavModelTitan <|-- NavModelTitanSimulated

      NavTechnique <|-- StarUniqueMatchNav
      NavTechnique <|-- StarFieldFromCatalogNav
      NavTechnique <|-- StarRefineNav
      NavTechnique <|-- BodyLimbNav
      NavTechnique <|-- BodyTerminatorNav
      NavTechnique <|-- BodyDiscCorrelateNav
      NavTechnique <|-- BodyBlobNav
      NavTechnique <|-- RingEdgeNav
      NavTechnique <|-- RingAnnulusNav
      NavTechnique <|-- TitanHazeNav
      NavTechnique <|-- NavTechniqueManual

      NavOrchestrator --> NavModel : iterates
      NavOrchestrator --> NavTechnique : iterates registry
      NavOrchestrator --> NavContext : builds
      NavOrchestrator --> NavResult : produces
      NavOrchestrator o-- FeatureReliabilityGate : owns
      NavOrchestrator o-- InstrumentSettings : reads
      NavOrchestrator o-- NavImageClassifier : runs
      NavOrchestrator o-- Annotations : merges
      NavModel ..> NavFeature : emits
      NavModel ..> Annotations : emits
      NavTechnique ..> NavFeature : consumes
      NavTechnique ..> NavTechniqueResult : produces
      NavResult --> NavTechniqueResult : per_technique
      NavResult --> Annotations : annotations
      Annotations o-- Annotation : aggregates


Top-level driver
================

:class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` is the
top-level driver. Given a list of pre-built
:class:`~spindoctor.nav_model.nav_model.NavModel` instances and an
:class:`~spindoctor.obs.obs_snapshot.ObsSnapshot`, the orchestrator runs the
full two-pass navigation pipeline: it builds a
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`, calls each
model's ``create_model``, gathers
:class:`~spindoctor.feature.feature.NavFeature` instances via ``to_features``,
gates them by reliability, runs every feasible
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`, and reconciles
the per-technique results through
:func:`~spindoctor.nav_orchestrator.ensemble.ensemble`. The output is a single
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.

Glob-pattern filters at construction time
(``only_models='body:MIMAS'``,
``only_techniques='!StarFieldFromCatalogNav'``) restrict which models or
techniques run, supporting debugging and per-image study without
modifying registry contents. See
:ref:`selecting-models-and-techniques` in the user guide for the full
pattern syntax (globs, ``!`` exclusion, prefix-only shorthand) and the
list of shipping model and technique names.

NavBase
=======

:class:`~spindoctor.support.nav_base.NavBase` is the shared base class for the
orchestrator, every model, every technique, and the dataset / obs
hierarchies. It carries no per-instance state beyond two read-only
properties:

- :attr:`~spindoctor.support.nav_base.NavBase.config` — the
  :class:`~spindoctor.config.config.Config` object handed in at construction
  time, defaulting to the module-level
  :data:`~spindoctor.config.config.DEFAULT_CONFIG` singleton when none is
  supplied. Subclasses read every YAML-driven tunable through this
  property so per-instance overrides flow naturally for tests.
- :attr:`~spindoctor.support.nav_base.NavBase.logger` — the logger this
  component belongs to, selected by its
  :attr:`~spindoctor.support.nav_base.NavBase.log_role`. The default,
  :attr:`~spindoctor.config.log_scope.LogRole.IMAGE`, gives
  :data:`~spindoctor.config.log_scope.IMAGE_LOGGER`, a proxy resolving to
  whichever image scope is open. A component whose work spans a run rather
  than an image, as :class:`~spindoctor.dataset.dataset.DataSet` does,
  declares :attr:`~spindoctor.config.log_scope.LogRole.MAIN` and gets
  :data:`~spindoctor.config.logger.MAIN_LOGGER`. Subclasses log through this
  property; never through the stdlib :mod:`logging` module.

Construction follows a single contract: every subclass takes a
keyword-only ``config`` parameter and forwards it via
``super().__init__(config=config, **kwargs)``. The
``**kwargs`` pass-through lets subclass-specific keyword arguments flow
through without forcing the base class to enumerate them.

FeatureReliabilityGate
======================

:class:`~spindoctor.feature.reliability.FeatureReliabilityGate` is the gate the
orchestrator runs between feature extraction and technique invocation.
It carries a per-:class:`~spindoctor.feature.feature_type.NavFeatureType`
threshold mapping (defaulted from
:data:`~spindoctor.feature.reliability.DEFAULT_RELIABILITY_THRESHOLDS`) and
exposes a single
:meth:`~spindoctor.feature.reliability.FeatureReliabilityGate.apply` method that
splits an extracted feature list into ``(kept, gated)``: features whose
self-assessed
:attr:`~spindoctor.feature.feature.NavFeature.reliability` falls below the
per-type threshold are emitted as
:class:`~spindoctor.feature.reliability.GatedFeatureRecord` instances carrying a
stable ``reason`` string; everything else passes through unmodified.

Per-instance threshold overrides come from the ``orchestrator.reliability_gate``
block in ``config_540_orchestrator.yaml``, read by the orchestrator's
constructor via
:meth:`~spindoctor.feature.reliability.FeatureReliabilityGate.from_mapping`.
Threshold values are validated
on construction: they must be finite floats in :math:`[0, 1]` and keyed
by :class:`~spindoctor.feature.feature_type.NavFeatureType` enum members. The
gate is stateless — the same instance can be reused across images — and
deterministic given its thresholds.

Dataset, Obs, and ObsSnapshot
=============================

:class:`~spindoctor.dataset.dataset.DataSet` handles access to image files and
metadata; per-mission subclasses
(``DataSetPDS3CassiniISS``, ``DataSetPDS3VoyagerISS``,
``DataSetPDS3GalileoSSI``, ``DataSetPDS3NewHorizonsLORRI``,
``DataSetSim``) implement archive-specific iteration and PDS4 bundle
hooks.

:class:`~spindoctor.obs.obs.Obs` is the abstract observation base.
:class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` adds backplane handling and
extended-FOV accessors; per-instrument subclasses derive from
:class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` and implement the
``from_file(path, ...)`` constructor. See
:doc:`dev_guide_observations` for the per-mission subclass list, the
public API surface, and the new-instrument checklist.

NavModel
========

:class:`~spindoctor.nav_model.nav_model.NavModel` is the abstract base for
predicted-scene generators. Subclasses implement three methods:

- ``create_model()`` populates the model's internal state and
  ``metadata`` dict.
- ``to_features(context)`` returns a list of
  :class:`~spindoctor.feature.feature.NavFeature` instances ready for
  technique consumption.
- ``to_annotations(context)`` returns an
  :class:`~spindoctor.annotation.annotations.Annotations` collection that the
  orchestrator merges into
  :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.annotations`.

Concrete subclasses self-register via ``__init_subclass__`` unless they
opt out with ``_abstract = True``. The class method
``instances_for_obs(obs)`` is the per-class hook that
``build_models_for_obs`` iterates. Today's concrete subclasses are
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` (catalog-driven
star navigation, one instance per observation),
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` (catalog-driven
per-body silhouette navigation),
:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` (catalog-driven
per-planet ring navigation), the simulated-image siblings
:class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`
(builds the star list from the scene's catalog entries in ``obs.nav_params``),
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated`,
and
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
(rendered from operator-supplied parameters), and
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` (active whenever
Titan is in the extended FOV; emits the haze-envelope geometry a solar-symmetry
fit needs) with its simulated-image sibling
:class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated`.

Per-family data models live alongside the renderer classes:

- **Stars** — the :mod:`spindoctor.nav_model.stars` subpackage carries
  multi-catalog reduction (``catalog``), per-star body / ring conflict
  marking (``conflicts``), the raw-DN photometry diagnostic + B-V mapping
  (``predicted_snr``), smear-aware PSF construction (``smeared_psf``),
  the measurement of the streak a smeared frame's stars draw (``streak``),
  and on-image source detection (``detection``); see
  :doc:`dev_guide_navigation_models_star` for details.
- **Bodies** — per-body shape, albedo, and SPK-residual values live in
  :mod:`spindoctor.nav_model.body_shape`
  (:class:`~spindoctor.nav_model.body_shape.BodyShape`,
  :data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE`,
  :data:`~spindoctor.nav_model.body_shape.DEFAULT_BODY_SHAPE`,
  :func:`~spindoctor.nav_model.body_shape.load_body_shape`); see
  :doc:`dev_guide_navigation_models_body` for details.
- **Rings** — the :mod:`spindoctor.nav_model.rings` subpackage carries the
  catalog-driven ring-feature data model (``RingFeature``,
  ``RingFeatureFilter``, ``RingRenderResult``, ``RingsRenderContext``,
  ``ring_math``, ``ring_types``); see
  :doc:`dev_guide_navigation_models_ring` for details.
- **Titan** — the observation-side geometry lives in
  :mod:`spindoctor.nav_model.titan_geometry`
  (:class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs`,
  :func:`~spindoctor.nav_model.titan_geometry.geometry_from_obs`), so
  everything the model does downstream of it is a pure function of one frozen
  dataclass;
  :class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated`
  builds the same dataclass from operator parameters. See
  :doc:`dev_guide_navigation_models_titan` for details.

NavTechnique
============

:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` is the abstract
base for navigation algorithms. Techniques consume a subset of
:class:`~spindoctor.feature.feature.NavFeature` instances filtered by
``accepts_feature_types`` and produce a
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` with an
offset, covariance, calibrated confidence, and per-technique
diagnostics. ``is_feasible(features)`` is consulted before invocation
and reads feature metadata only — never pixels.

Concrete subclasses self-register.
:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual`
opts out of the auto-discovery registry (it spawns a PyQt6 dialog) and
is invoked by interactive drivers only. The shipping autonomous
techniques are, by family:

- **Star** —
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`.
- **Body** —
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`.
- **Ring** —
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`.
- **Titan** —
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`,
  which consumes the single
  :data:`~spindoctor.feature.feature_type.NavFeatureType.TITAN_LIMB` feature
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` emits and
  measures the offset from the haze's solar symmetry.

The DT-based body / ring techniques share their coarse-NCC +
Levenberg-Marquardt + Tukey-biweight machinery via
:mod:`spindoctor.nav_technique.dt_fitting`. See :doc:`dev_guide_techniques`
for the per-technique deep dive.

NavFeature and NavFeatureGeometry
=================================

A :class:`~spindoctor.feature.feature.NavFeature` is the smallest
independently-navigable scene element: a star, one body's limb arc,
one ring edge, a body disc rendered as a pixel template, and so on.
The :class:`~spindoctor.feature.feature_type.NavFeatureType` enum names every
shipping feature category.

The ``geometry`` field carries one of the
:data:`~spindoctor.feature.geometry.NavFeatureGeometry` payload variants
(``StarGeometry``, ``LimbPolyline``, ``TerminatorPolyline``,
``RingEdgePolyline``, ``BodyDiscGeometry``, ``BodyBlobGeometry``,
``RingAnnulusGeometry``, ``CartographicModelGeometry``,
``TitanHazeGeometry``); each variant
records the in-image position the technique needs. The ``flags`` field
carries one of the
:data:`~spindoctor.feature.flags.NavFeatureFlags` typed dataclasses, capturing
feature-type-specific booleans (for example,
``RingEdgeFlags.is_straight_line`` or ``TitanHazeFlags.high_phase``).

Per-feature uncertainty in image-plane pixels lives on
``position_cov_px`` (or per-vertex on the polyline payloads);
``preferred_filter`` is the
:class:`~spindoctor.support.filters.NavFilterSpec` the feature requests for
both its template and the surrounding image patch.

NavResult, ensemble, curator
============================

The orchestrator's final answer is a
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` carrying the headline
``offset_px`` ± ``sigma_px``, a five-bucket ``confidence_rank``, the
discrete ``status_reason``, every per-technique
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`, the
per-feature inventory (kept and gated entries), the image-quality
classifier verdict, the merged annotation collection, and the
reproducibility :class:`~spindoctor.nav_orchestrator.provenance.Provenance`.

:func:`~spindoctor.nav_orchestrator.ensemble.ensemble` is a free function (not
a class) that performs the precision-weighted Kalman-style merge.
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` projects
``NavResult`` into a JSON-friendly metadata block written by
``navigate_image_files``.

Annotation
==========

The :mod:`spindoctor.annotation` subsystem composes labels and graphical
elements into an overlay used by the summary PNG.
:class:`~spindoctor.annotation.annotations.Annotations` aggregates
model-provided annotations and renders them with appropriate coloring
and contrast stretching. Each
:meth:`~spindoctor.nav_model.nav_model.NavModel.to_annotations` returns a fresh
:class:`~spindoctor.annotation.annotations.Annotations` collection; the
orchestrator merges them into
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.annotations` via
:meth:`~spindoctor.annotation.annotations.Annotations.add_annotations`. See
:doc:`dev_guide_annotations` for the per-class API, the per-:class:`~spindoctor.nav_model.nav_model.NavModel`
contributions, and the source-of-configuration map.
