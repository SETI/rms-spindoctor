=====
Rings
=====

Ring navigation models render the predicted appearance of a planet's ring system and emit
either per-edge polylines (the "edges resolve" path) or a per-planet annulus template (the
"edges compress" path). Concrete subclasses derive from
:class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`, which carries shared
annotation helpers (per-edge polyline + label rendering).

Registered concrete subclasses:

- :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` — catalog-driven ring navigation;
  one instance per planet whose ring system has any radius inside the extended FOV.
  Documented at :doc:`dev_guide_navigation_models_ring`.
- :class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` —
  the simulated-frame variant, predicted from the scene's ring system; emits a
  :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS` template for a ringlet
  and one :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` polyline per
  predicted edge. Documented at
  :doc:`dev_guide_navigation_models_ring_simulated`.

The :mod:`spindoctor.nav_model.rings` subpackage holds the catalog-driven ring data model —
validation, filtering, and rendering are separated so each concern can be tested in
isolation (``ring_types``, ``ring_feature``, ``ring_filter``, ``ring_math``,
``ring_render_context``, ``ring_render_result``).

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_ring
   dev_guide_navigation_models_ring_simulated
