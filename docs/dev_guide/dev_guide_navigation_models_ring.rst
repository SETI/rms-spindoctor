==========================================================
Ring Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` is the catalog-driven ring navigation
model. For each planet whose ring system has any radius inside the extended FOV the model
renders the per-ring-edge silhouette from the catalog, runs a four-pass
``RingFeatureFilter`` to drop edges that are not separable / detectable on this image, and
emits either a :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` per surviving edge
(the "edges resolve" path) or a single
:data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` per planet (the "edges
compress" path) when individual edges fall below the resolvability threshold.

The orchestrator constructs one model instance per planet whose ring system overlaps the
extfov. A simulated-image sibling
(:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`) renders rings
from operator-supplied parameters instead of the catalog; both classes share annotation
helpers on :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.

Theory
======

The ring model is a per-planet edge renderer plus a feature-emission gate that decides
whether to ship per-edge polylines or a composite annulus template.

Edge rendering
--------------

For each catalog-defined ring edge the model:

1. Builds an oversampled meshgrid around the predicted ring's projected bounding box.
2. Queries the per-pixel ring radius backplane and the ring longitude backplane.
3. Marks the discrete pixel set whose radius lies on the edge's catalog radius.
4. Walks the pixel set to produce a polyline of vertices, each with an outward radial
   normal estimated from the local radius gradient.

.. _ring-planet-occlusion:

Planet occlusion
----------------

Ring points hidden behind the planet globe are removed before any feature or overlay is
produced. The model evaluates the ``where_in_front(planet, ring)`` backplane -- true at
every extended-FOV pixel where the planet disc is intercepted and nearer to the observer
than the ring plane -- and clears those pixels from every per-edge mask at render time.
Both the emitted
:data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` features and the summary
overlay read the same masks, so an edge segment behind the disc is neither fitted by
:doc:`dev_guide_techniques_ring_edge` nor drawn across the planet, while a near-side edge
crossing in front of the disc is kept. The same backplane clears the annulus correlation
template: the full-feature render that feeds the
:data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` path is trimmed against
it before the per-ring renderings are composited, so the template carries no ring brightness
from behind the disc and
:doc:`dev_guide_techniques_ring_annulus` correlates only visible ring material. When the
backplane cannot be evaluated the model keeps every edge and the full template rather than
aborting.

Per-edge feature filtering
--------------------------

A four-pass filter removes edges that cannot contribute a useful constraint:

- **Catalog presence.**  Edges absent from the catalog or with non-finite radius are
  dropped.
- **Visibility.**  Edges whose predicted bounding box has no overlap with the extfov are
  dropped.
- **Resolvability.**  Edges whose maximum radial pixel extent compresses below
  ``feature_emission.ring_annulus.<planet>.max_radial_px`` are flagged for the annulus path.
- **Detectability.**  Edges whose per-pixel signal-to-noise falls below the per-instrument
  detection threshold are dropped.

Curvature classification
------------------------

For each surviving polyline the model fits a best-fit straight line and measures the
maximum perpendicular deviation. An edge whose deviation exceeds
``curvature_threshold_pixels`` (or the configured fraction of the edge's length) is flagged
``is_curved``; otherwise the edge is straight-line and the per-edge constraint is rank-1
along the radial direction. The :doc:`dev_guide_techniques_ring_edge` page describes how
the downstream technique handles the all-straight case.

Per-vertex covariance
---------------------

Each vertex's normal is signed against the ring-radius backplane so it points toward
increasing ring radius.  This also affects the annulus emission gate below: the radial
extent is measured by projecting onto the MEAN normal, which is sign-sensitive, so a half
turn of arc now returns the ring radius (geometrically right) where scan-order signs
returned a larger value set by the rasterizer's quadrant bias, and short curved edges
measure roughly half what they used to.  An edge near the ``max_radial_px`` threshold can
therefore emit as a ``RING_ANNULUS`` where it previously emitted as a ``RING_EDGE``,
changing which technique navigates it.  The mask-neighbor test that finds the normal AXIS cannot
distinguish the high-radius side from the low-radius side on its own -- it would emit signs
that follow scan order and rasterization -- and the orbit-uncertainty channel sums the
normals, so a random sign per vertex would fabricate coherence on geometry that should
cancel.

Each polyline vertex carries two per-vertex sigma values: a radial sigma along the
outward normal (the constrainable axis) and an along-edge sigma along the polyline
tangent (the unobservable axis).

The radial sigma is the catalog-side radial uncertainty :math:`\sigma_{\mathrm{rms,\,km}}`
projected to pixels at the ring's radial scale:

.. math::

    \sigma_{\mathrm{radial,\,px}} =
        \frac{\sigma_{\mathrm{rms,\,km}}}{\mathrm{km/px}_{\mathrm{radial}}}.

The numerator is the maximum of the inner-edge and outer-edge ``rms`` values supplied by
the per-planet ring catalog (``config_3N0_<planet>_rings.yaml``); taking the maximum
rather than the average is conservative — a feature's overall radial uncertainty is
dominated by its least well-characterized edge. The denominator is the per-image radial
km/px scale at the ring (the mean of the per-pixel ring-radial-resolution backplane). A
single :math:`\sigma_{\mathrm{radial,\,px}}` value is broadcast across every vertex of
the polyline; spatial variation of the catalog ``rms`` along the ring's longitude is not
modeled.

The along-edge sigma is the project-wide constant
:data:`~spindoctor.nav_model.nav_model_rings.RING_EDGE_SIGMA_ALONG_PX` (``0.5`` px), reflecting
the polyline-sampling resolution. The DT-based fit treats motion along the polyline as
essentially unobservable by construction, so this axis only sets the scale at which
along-edge displacement is numerically de-weighted relative to the radial axis.

The per-vertex covariance does not absorb the optical PSF sigma, the per-edge photometric
contrast against the background, or any longitude-dependent perturbation; those terms
shift the apparent edge position non-uniformly around the ring rather than enlarging the
per-vertex radial sigma, and the technique-side fitter handles that scatter via the
shared M-estimator robust-weighting machinery in :mod:`spindoctor.nav_technique.dt_fitting`
(see :doc:`dev_guide_techniques_dt_fitting`).

The same catalog RMS is carried a second time, on the emitted geometry's
``sigma_orbit_radial_px``, as the coherent radial orbit-solution uncertainty of the whole
edge (an edge with no catalog ``rms`` falls back to
``rings.default_orbit_radial_sigma_km``, and
``rings.orbit_radial_sigma_correlated_fraction`` scales the whole term). The per-vertex sigma is a statistical scale
that averages down as the vertex count grows; an orbit-solution error displaces every
vertex of the edge coherently and does not, so the ring-edge technique adds this term in
quadrature to its reported covariance along the fit's radial direction instead of folding
it into the per-vertex weights (see :doc:`dev_guide_techniques_ring_edge`).

Annulus template
----------------

When the per-planet km/px scale exceeds the configured threshold (or any single ring edge
compresses below the per-polyline radial-pixel threshold), the model emits a single
:attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` feature carrying a rendered
template image of the visible ring system
(every ring radius painted at the catalog brightness contrast, with pixels behind the
planet disc cleared as described under :ref:`ring-planet-occlusion`) plus the matching
mask.
The template's bounding box is the union of the per-edge bounding boxes; the template
brightness at each pixel is the sum of the per-ring-edge contributions.

The annulus feature also carries the same coherent orbit-uncertainty channel as the per-edge
path, so the correlation technique can price a misplaced catalog orbit rather than reporting
a tight lock on it. The constituent edges' outward normals are concatenated onto
:attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.orbit_normals_vu`, and their per-edge
orbit sigmas (each the catalog RMS, or ``rings.default_orbit_radial_sigma_km`` in its absence,
scaled by ``rings.orbit_radial_sigma_correlated_fraction`` and converted at the edge's own
radial scale) combine into a vertex-weighted effective
:attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.sigma_orbit_radial_px`.
:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` widens its reported
covariance from those, exactly as the ring-edge technique does from its fit vertices (see
:doc:`dev_guide_techniques_ring_annulus`).

Restrictions and assumptions
----------------------------

- The catalog must provide a per-ring-edge radius and an RMS uncertainty. Rings missing
  either field are dropped silently.
- The model assumes the per-image SPICE pose is good enough that the predicted ring
  geometry is approximately correct. A wrong pose shifts every edge polyline by the same
  pose error; the downstream DT fit recovers the offset.
- The detectability filter assumes the per-instrument calibration converts the catalog
  surface brightness into a per-pixel signal correctly. When the calibration is wrong
  (e.g. on a calibrated-IF instrument with a stale CALIB pipeline), the detectability
  test may include or exclude wrong edges.

Sources of uncertainty
----------------------

The per-vertex radial sigma is the catalog-side RMS projected to pixels at the ring's
radial scale. It does not capture the optical PSF blur, the per-edge photometric softness
against the background, or a longitude-dependent brightness modulation that would shift
the apparent edge position non-uniformly around the ring. Those terms enter the
technique-side fit through the M-estimator's robust weighting (which down-weights
vertices whose DT residual is inconsistent with the per-vertex sigma) rather than by
inflating the sigma itself; see :doc:`dev_guide_techniques_dt_fitting` for the fitter's
treatment. A coherent radial bias from the orbit solution itself is priced separately
through ``sigma_orbit_radial_px`` (above), which the ring-edge technique adds to its
reported covariance rather than to any per-vertex weight. Edges flagged
:attr:`~spindoctor.feature.flags.RingEdgeFlags.is_straight_line` are rank-1 along radial only;
edges that pass the curvature classification carry full-rank locally-observable
information.

Configuration
=============

The model's runtime knobs are split across three locations: the ``rings`` block in
``src/spindoctor/config_files/config_050_rings.yaml`` (general per-model knobs and label
rendering), the per-planet ring catalogs in
``src/spindoctor/config_files/config_3N0_*_rings.yaml`` (one file per planet, each carrying a
``rings.ring_features`` mapping), and the per-planet annulus-emission thresholds under
``feature_emission.ring_annulus`` in ``src/spindoctor/config_files/config_510_techniques.yaml``
(consumed by the per-feature emission gate; see :doc:`dev_guide_techniques_ring_annulus`).
Module-level Python constants in :mod:`spindoctor.nav_model.nav_model_rings` set per-vertex
sigma and curvature thresholds that are not exposed to YAML.

rings block
-----------

Every key under ``rings`` is listed below. Several keys are reserved for ring-feature
filtering work that is not consumed by the active extractor; the second line of each
bullet names the consumer (or "reserved" when no consumer is wired).

- ``model_source`` — str, default ``ephemeris``. Selects the ring-feature data source.
  Reserved; the active extractor reads the per-planet ``ring_features`` mapping
  unconditionally.
- ``fiducial_feature_threshold`` — int, default ``3`` (count). Reserved for the
  fiducial-feature classifier that promotes a ring edge to fiducial status when at
  least this many independent edge measurements line up. Not consumed by the active
  extractor.
- ``fiducial_rms_gain`` — float, default ``2`` (dimensionless). Reserved alongside
  ``fiducial_feature_threshold`` for the fiducial classifier's RMS gain term. Not
  consumed by the active extractor.
- ``fiducial_min_feature_width`` — int, default ``2`` px. Reserved alongside
  ``fiducial_feature_threshold``; minimum radial width below which a fiducial promotion
  is suppressed. Not consumed by the active extractor.
- ``one_sided_feature_width`` — float, default ``30.0`` px. Reserved width used by the
  fiducial classifier for one-sided (gap or step) features. Not consumed by the active
  extractor.
- ``fiducial_ephemeris_width`` — int, default ``100`` px. Reserved tolerance window for
  matching ephemeris-predicted radii against detected fiducials. Not consumed by the
  active extractor.
- ``min_curvature_low_confidence`` — list (rad, ratio), default ``[0.0, 0.5]``. Reserved
  curvature / confidence threshold for the rank-1 degeneracy escape on the all-straight
  ring fit. Not consumed by the active extractor; the equivalent test lives on the
  per-feature flags.
- ``min_curvature_high_confidence`` — list (rad, ratio), default ``[0.17, 1.0]``.
  Reserved alongside ``min_curvature_low_confidence``. Not consumed by the active
  extractor.
- ``curvature_to_reduce_features`` — float, default ``1.5708`` rad (= 90 degrees).
  Reserved threshold above which the extractor would reduce the surviving polyline
  count by collapsing nearly-orthogonal edges. Not consumed by the active extractor.
- ``curvature_reduced_features`` — int, default ``1`` (count). Reserved for the
  reduce-features path. Not consumed by the active extractor.
- ``emission_fiducial_threshold`` — float, default ``0.75`` (dimensionless). Reserved
  per-edge fiducial-promotion threshold. Not consumed by the active extractor.
- ``emission_use_threshold`` — float, default ``0.2`` (dimensionless). Reserved per-edge
  emission floor. Not consumed by the active extractor.
- ``remove_planet_shadow`` — bool, default ``true``. When true the model masks pixels
  inside the per-planet shadow before rendering ring edges; the ring radius is still
  defined inside the shadow but the brightness is zero, so leaving the shadow pixels
  in would skew the per-vertex covariance. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings`.
- ``remove_body_shadows`` — bool, default ``false``. Reserved for masking the projected
  shadow of every body in the FOV. Not consumed by the active extractor.
- ``ring_features`` — dict[str, dict]. Per-planet ring catalog overlaid from the
  per-planet ``config_3N0_*_rings.yaml`` files (see below). Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings`.
- ``label_font`` — str, default ``liberation2/LiberationMono-Bold.ttf``. Font used for
  ring labels. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_font_size`` — int, default ``18`` px. Ring label font size. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_font_color`` — list[int], default ``[255, 0, 0]`` (RGB). Ring label font
  color. Consumed by :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_limb_color`` — list[int], default ``[255, 0, 0]`` (RGB). Color of the per-edge
  polyline drawn on the summary PNG. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_mask_enlarge`` — int, default ``10`` px. Pixels around a ring edge to avoid
  for label placement. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_horiz_gap`` — int, default ``7`` px. Horizontal gap between the edge and the
  head of the label arrow. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
- ``label_vert_gap`` — int, default ``5`` px. Vertical gap between the edge and the head
  of the label arrow. Consumed by
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.

Per-planet ring catalog
-------------------------

Each ``config_3N0_<planet>_rings.yaml`` file carries a single
``rings.ring_features.<PLANET>`` mapping keyed by upper-case SPICE planet name (one file
per planet: Jupiter, Saturn, Uranus, Neptune). Each per-planet entry sets:

- ``epoch`` — str (UTC). Reference epoch for every ring solution under this planet;
  used to advance precessing edge solutions to the per-image time. A missing or invalid
  epoch raises at config-load.
- ``fade_width_pix`` — float (px). Per-planet softness of the per-edge brightness
  profile used by the rendered annulus template; controls how many pixels the catalog
  brightness fades into the background.
- ``min_allowed_fade_width_pix`` — float (px). Floor on the per-edge fade width applied
  after per-image scaling; prevents the rendered template from collapsing to a
  single-pixel ring at high subject distances.
- ``min_feature_pixels`` — float (px). Minimum per-feature pixel extent below which the
  per-image filter drops the feature.
- ``features`` — dict[str, dict]. Per-edge entries keyed by feature name. Each entry
  carries ``feature_type`` (``GAP`` / ``RINGLET`` / ``RING``), ``name`` (display label),
  and one or both of ``inner_data`` / ``outer_data`` lists. Each ``*_data`` list
  enumerates the per-mode catalog terms (mode number plus ``a`` / ``rms`` / ``ae`` /
  ``long_peri`` / ``rate_peri`` for orbiting features, or ``amplitude`` / ``phase`` /
  ``pattern_speed`` for free-mode features). Consumed by
  :class:`~spindoctor.nav_model.rings.ring_feature.RingFeature`.

Module-level emission constants
-------------------------------

The per-vertex sigma defaults and curvature threshold are Python module-level constants
in :mod:`spindoctor.nav_model.nav_model_rings` and are not exposed as YAML knobs. Tests and
downstream tools read the canonical values via these symbols.

- :data:`~spindoctor.nav_model.nav_model_rings.RING_EDGE_DEFAULT_RELIABILITY` — float, ``0.7``
  (dimensionless). Catalog default reliability scaling applied to a
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` feature before per-image
  weighting; the design's "catalog_default_reliability" term in the
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` sigmoid.
- :data:`~spindoctor.nav_model.nav_model_rings.RING_EDGE_SIGMA_ALONG_PX` — float, ``0.5`` px.
  Per-vertex sigma along the polyline tangent direction. Reflects polyline-sampling
  resolution; the technique-side fitter treats this axis as essentially unobservable.
- :data:`~spindoctor.nav_model.nav_model_rings.FLAT_CURVATURE_THRESHOLD_PX` — float, ``1.0`` px.
  Pixel-deviation threshold below which a polyline is flagged
  :attr:`~spindoctor.feature.flags.RingEdgeFlags.is_straight_line`. The technique-side fitter
  then handles its rank-1 covariance.

Per-instrument overrides
------------------------

The ``rings`` block is global: per-instrument YAML (``config_4N0_inst_*.yaml``) does not
override any of the keys above. Instrument-specific behavior enters through the
observation snapshot — the optical PSF sigma read from
:meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf` and the extended-FOV margin set by
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` — rather than through
this config block.

Implementation
==============

Source files:

- ``src/spindoctor/nav_model/nav_model_rings.py`` —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings`, the four-pass filter, the
  per-edge sampler, and the annulus-template builder.
- ``src/spindoctor/nav_model/nav_model_rings_base.py`` —
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`, abstract shared base
  carrying the ring annotation pipeline.
- ``src/spindoctor/nav_model/rings/`` — the :mod:`spindoctor.nav_model.rings` subpackage with the
  validation, filtering, and rendering helpers
  (:mod:`~spindoctor.nav_model.rings.ring_types`,
  :mod:`~spindoctor.nav_model.rings.ring_feature`,
  :mod:`~spindoctor.nav_model.rings.ring_filter`,
  :mod:`~spindoctor.nav_model.rings.ring_math`,
  :mod:`~spindoctor.nav_model.rings.ring_render_context`,
  :mod:`~spindoctor.nav_model.rings.ring_render_result`).

Public class :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings`, base
:class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`. Self-registers via
``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.instances_for_obs` — class method
  that returns one instance per planet whose ring system has any radius inside the
  extended FOV.
- :meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.create_model` — populates the model
  state by rendering each per-edge silhouette, running the four-pass filter, and emitting
  per-vertex polyline data plus an optional annulus template.
- :meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.to_features` — runs the per-edge
  emission gates and constructs zero or more
  :class:`~spindoctor.feature.feature.NavFeature` instances
  (:attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` per surviving edge, or
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` per planet when the
  annulus path fires).
- :meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.to_annotations` — emits per-edge
  polylines and per-planet labels for the summary PNG.

Inherited :class:`~spindoctor.nav_model.nav_model.NavModel` properties:
:attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.obs`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.

Annotation helpers
------------------

:class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase` is the abstract shared
base. One helper lives there:

- ``_create_edge_annotations`` — builds the per-edge polyline + per-edge label
  :class:`~spindoctor.annotation.annotations.Annotations` collection for the summary PNG.
  Consumes the ``label_*`` keys documented above. Used by both
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` and
  :class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`.

The per-edge anti-aliasing math invoked from
:class:`~spindoctor.nav_model.rings.ring_feature.RingFeature` lives in
:mod:`~spindoctor.nav_model.rings.ring_math` (``compute_antialiasing``); it is shared between
the real and simulated paths but is not part of the annotation pipeline.

Per-image metadata
------------------

:meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.create_model` populates
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` with the following entries for the
curator to surface in the per-image JSON sidecar:

- ``start_time`` / ``end_time`` / ``elapsed_time_sec`` — wall-clock timing for the model
  build.
- ``planet`` — upper-case SPICE planet name. Absent when no planet is in the FOV.
- ``epoch`` — UTC epoch string for the ring solution (the per-planet
  ``rings.ring_features.<PLANET>.epoch`` value advanced precessing solutions are
  evaluated against).
- ``feature_count`` — int, number of ring features that survived the four-pass filter.
- ``features`` — list[dict[str, str]], one entry per surviving ring feature carrying
  ``name`` (the catalog edge name) and ``type`` (the
  :class:`~spindoctor.feature.feature_type.NavFeatureType` value the per-edge emission gate
  ultimately produced — ``RING_EDGE`` or ``RING_ANNULUS``).

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.create_model`:

1. Open a logged section. Look up the per-planet ring catalog from the configured
   ``ring_features`` mapping. Each entry carries a name, a radius, an RMS, and a per-edge
   surface-brightness profile.
2. Build an oversampled meshgrid around the predicted ring's projected bounding box and
   query the per-pixel ring radius and longitude backplanes.
3. For each catalog edge, mark the pixel set whose ring radius lies within the per-edge
   tolerance and run the four-pass filter.
4. For each surviving edge, walk the pixel set to produce a per-vertex polyline (position,
   radial normal, per-vertex sigma). Classify the polyline curvature via the best-fit
   straight-line residual.
5. Decide the per-planet emission path: when the per-planet km/px scale exceeds the
   configured threshold, or when any single edge compresses below the per-polyline radial-
   pixel threshold, render a
   :attr:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` template; otherwise emit
   per-edge polylines.

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.to_features`:

1. **Annulus path.**  Construct one
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` feature per planet
   carrying the rendered annulus template plus the per-planet bounding box.
2. **Per-edge path.**  For each surviving edge, construct one
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` feature carrying the
   :class:`~spindoctor.feature.geometry.RingEdgePolyline` (vertices, normals, per-vertex sigmas)
   plus a per-edge :class:`~spindoctor.feature.flags.RingEdgeFlags` with the catalog edge name and
   the curvature classification.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A high-resolution Saturn-ring scene whose individual catalog edges resolve into
    separable polylines. The ring model emits multiple
    :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` features (the F-ring outer
    edge, the A-ring outer edge, gaps, ringlets); the per-planet km/px on this scene is
    well below the annulus threshold so the annulus path does not fire. Curvature
    classification flags the edges curved (each surviving polyline arcs noticeably across
    the FOV); the rank of the joint :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`
    fit is full-rank because the curvature lifts the rank-1 degeneracy. See
    :doc:`dev_guide_techniques_ring_edge`.

``ring_annulus_unresolved`` (Cassini ISS WAC, low-resolution approach phase)
    A low-resolution approach-phase Saturn image whose km/px exceeds the per-planet
    kmpp threshold. The ring model emits a single
    :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` feature per planet
    carrying the rendered annulus template; the
    :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` consumes it
    via the shared pyramid-NCC machinery. See
    :doc:`dev_guide_techniques_ring_annulus`.
