======
Bodies
======

Body navigation models render the predicted appearance of a planetary body and emit one
:class:`~spindoctor.feature.feature.NavFeature` per surviving feature type. Concrete subclasses
derive from :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`, which carries shared
annotation helpers (limb-mask extraction and label placement).

Registered concrete subclasses:

- :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` — catalog-driven body navigation; one
  instance per body whose :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_extfov`
  predicate fires. Documented at :doc:`dev_guide_navigation_models_body`.
- :class:`~spindoctor.nav_model.simulated.body.NavModelBodySimulated` — simulated-image GUI
  variant; emits a single :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` feature
  carrying the rendered template.
- :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` — Titan is a body, but its opaque
  haze hides the surface, so it emits a haze-envelope
  :data:`~spindoctor.feature.feature_type.NavFeatureType.TITAN_LIMB` feature instead of shape
  features. Documented at :doc:`dev_guide_navigation_models_titan`; its simulated-image sibling
  is :class:`~spindoctor.nav_model.simulated.titan.NavModelTitanSimulated`.

Per-body shape, albedo, and SPK-residual quantities consumed by the covariance and emission
gates live in :mod:`spindoctor.nav_model.body_shape`. The runtime lookup
:func:`~spindoctor.nav_model.body_shape.load_body_shape` overlays operator YAML
(``config_220_body_shape.yaml``) onto the hard-coded
:data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE`, falling back to
:data:`~spindoctor.nav_model.body_shape.DEFAULT_BODY_SHAPE` for entirely unknown bodies.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_body
   dev_guide_navigation_models_body_simulated
   dev_guide_navigation_models_titan
   dev_guide_navigation_models_titan_simulated
