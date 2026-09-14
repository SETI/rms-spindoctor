==========================================================
Simulated Ring Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` is the
simulated-image variant of the ring navigation model. It predicts a scene's ring features
from the idealized ``ring_system`` view the information boundary exposes
(``obs.nav_params['ring_system']``, see :doc:`dev_guide_simulator`) instead of from a
SPICE-driven catalog, then emits two feature kinds:

- a :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` carrying the predicted
  coverage template, for
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`;
- one :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` per predicted catalog
  boundary -- a per-vertex polyline with outward radial normals -- for
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.

The model overrides :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to build
one instance per *navigable* feature of a simulated observation's ring system; the parent
:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` declines simulated observations, so
the autonomous registry routes simulated frames here. Non-navigable features never reach this
model: the boundary filter drops them from ``nav_params`` entirely, so the rendered frame is
full of ring structure the navigator was never told about -- the false-lock / distractor
regime the ring system exists to produce.

Theory
======

The prediction is geometric, not photometric. For each navigable feature the model
inverse-projects every extended-FOV pixel into the ring plane through the shared
opening-angle projection (:mod:`spindoctor.sim.ring_geometry`), evaluates the feature's
catalog orbit -- the mode-1 precessing ellipse plus any m >= 2 modes and satellite edge
wave -- at each pixel's ring-plane longitude, and rasterizes the signed radial distance
field. A banded kind (ringlet) yields a solid anti-aliased coverage template (the
navigator's opaque-annulus convention; the tau photometry is truth the navigator cannot
see); every kind yields one border polyline per catalog boundary, with normals taken from
the gradient of the distance field so they point radially outward however the projection
foreshortens the edge.

The projection and orbit math are shared with the image-side renderer by design (the
independence guarantee is informational, not code-level): a predicted edge lands where the
rendered edge would land if the scene planted no error, so the planted pointing offset and
the planted per-feature ``orbit_error`` are the only discrepancies in a recovery
measurement, by construction.

Restrictions and assumptions
----------------------------

- The prediction consumes only idealized keys: the shared ``geometry`` block, the catalog
  ``orbit``, the kind-specific shape keys, ``tau``, and ``declared_orbit_sigma``. The drawn
  ``orbit_error`` values and the photometric truth (``albedo``, ``phase_g``) are stripped by
  the boundary filter before this code runs.
- A ramp's linear end fades into the background with no gradient to fit, so only its sharp
  end emits an edge; a wave train emits its launch-radius boundary only.
- An exactly edge-on geometry (``opening_deg_obs`` of 0) predicts nothing, matching the
  renderer, which draws nothing.

Sources of uncertainty
----------------------

The per-vertex radial sigma floors at the one-pixel polyline sampling resolution and rises
with the feature's ``declared_orbit_sigma`` (the catalog error bars the navigator is
entitled to know -- never the drawn error values). The along-edge sigma reflects the
sampling resolution only.

The declared orbit sigma is additionally carried, unfloored, on the emitted geometry's
``sigma_orbit_radial_px`` -- the fully-correlated 1-sigma radial displacement of the whole
predicted edge. The per-vertex sigma is a statistical scale that averages down as the
vertex count grows; a catalog-orbit error displaces every vertex coherently and does not,
so the ring-edge technique consumes this separate field to widen its reported covariance
along the fit's radial direction (see :doc:`dev_guide_techniques_ring_edge`).

Configuration
=============

The simulated ring model consumes no YAML configuration of its own; every parameter comes
in via the observation's filtered scene view (``obs.nav_params['ring_system']``, see
:doc:`dev_guide_simulator`). Expected keys:

- ``geometry`` — the shared projection block: ``center_v`` / ``center_u``,
  ``opening_deg_obs`` / ``opening_deg_sun``, ``node_deg``.
- ``range_km`` — the system's physical range (the emitted features' subject range).
- ``features`` — the navigable subset only. Each entry carries ``name``, ``kind``
  (``ringlet`` / ``gap`` / ``edge`` / ``ramp`` / ``wave``), ``tau``, the kind-specific
  shape keys (``width``, ``side``, ``wavelength``, ``damping``), the catalog ``orbit``
  (``a``, ``ae``, ``long_peri``, ``rate_peri``, ``modes``, ``edge_wave``), and an optional
  ``declared_orbit_sigma``.

The scene-level ``time`` and ``ring_epoch`` drive the mode-1 pericenter precession on both
sides.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_rings_simulated.py`` —
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`.

Public class :class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`,
base :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`. The class overrides
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to build one instance per
navigable ring_system feature of a simulated observation; the parent
:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` returns an empty list for a simulated
observation, so the orchestrator's
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` driver routes simulated frames to
this subclass.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model` —
  invokes :func:`~spindoctor.nav_model.sim_ring.predict_ring_feature` to rasterize the
  feature's template and catalog boundaries on the extended-FOV grid.
- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features` —
  emits the RING_ANNULUS (banded kinds with coverage) plus one RING_EDGE per predicted
  boundary.
- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_annotations`
  — reuses the shared ring annotation helper on
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase` to render per-edge
  polylines and labels onto the summary PNG.

Inherited :class:`~spindoctor.nav_model.nav_model.NavModel` properties:
:attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.obs`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model`:

1. Open a logged section. Read the feature's idealized mapping and the shared
   ``geometry`` block off the per-instance state; offset the projected center by the
   extended-FOV margins.
2. Call :func:`~spindoctor.nav_model.sim_ring.predict_ring_feature`, which inverse-projects
   the pixel grid to ring-plane ``(r, lam)`` through
   :func:`~spindoctor.sim.ring_geometry.ring_plane_from_sky`, evaluates the catalog orbit via
   :func:`~spindoctor.sim.ring_geometry.compute_orbit_radii`, and rasterizes the coverage
   template plus one sign-transition border mask per boundary, with gradient-derived
   outward normals.
3. Record the predicted center, the subject range, and the bounding box on the model's
   internal state for downstream feature emission.

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features`:

1. For a banded feature with coverage, construct one
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS`
   :class:`~spindoctor.feature.feature.NavFeature` carrying the coverage template cropped to
   its tight bounding box, the predicted center, a
   :class:`~spindoctor.feature.flags.RingAnnulusFlags` with the feature's name, and the
   orbit-uncertainty channel: the predicted edges' outward normals concatenated onto
   :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.orbit_normals_vu` with the feature's
   single declared orbit sigma as
   :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.sigma_orbit_radial_px`, so
   :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` prices a
   misplaced orbit exactly as the ring-edge technique does.
2. For each predicted boundary, sample its border mask into a vertex polyline with outward
   radial normals, classify it straight or curved, widen the radial sigma by the declared
   orbit uncertainty, and append a
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` carrying a
   :class:`~spindoctor.feature.geometry.RingEdgePolyline`.
3. Reliability on each feature is fixed at ``1.0``.

Examples
========

The catalog scene ``algorithmic_invariants/planted_offset_ring.yaml`` plants a
``(1.43, -0.61)`` px pointing offset under a centered face-on ringlet (``tau = 2``,
radii 60-85 px) flagged navigable. The model predicts both catalog edges at the unshifted
position; :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` fits the
two curved polylines against the rendered image's edge distance transform and recovers the
planted offset to a few hundredths of a pixel. A scene that additionally plants
``orbit_error: {delta_a_px: 3.0}`` renders the band 3 px out from the prediction: the
navigator (correctly) cannot distinguish the ephemeris error from pointing error along the
radial direction, which is exactly the honest-degradation regime the planted-error axis
exists to measure.
