=================
Navigation Models
=================

:class:`~spindoctor.nav_model.nav_model.NavModel` is the abstract base for
predicted-scene generators. Each subclass implements three methods:

- :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model` — populate the model's
  internal state and :attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` dict.
- :meth:`~spindoctor.nav_model.nav_model.NavModel.to_features` — return a list of
  :class:`~spindoctor.feature.feature.NavFeature` instances for technique
  consumption.
- :meth:`~spindoctor.nav_model.nav_model.NavModel.to_annotations` — return an
  :class:`~spindoctor.annotation.annotations.Annotations` collection for the
  summary PNG.

Concrete subclasses self-register via ``__init_subclass__``; abstract
bases set ``_abstract = True`` to opt out. The class method
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` is the per-class hook that
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` iterates. Both take a
keyword-only ``config`` parameter used for *selection* (which ring systems have
catalogs, which satellites are considered) as well as for constructing the
instances, so a per-run configuration override changes which models exist the
same way it changes how they behave; ``None`` uses ``DEFAULT_CONFIG``.

Registered concrete models, grouped by feature family:

- **Stars** (:doc:`dev_guide_navigation_models_stars`) —
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` (catalog-driven; one
  instance per observation) and the simulated-image sibling
  :class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`
  (sources the star list from the sim renderer's output; documented at
  :doc:`dev_guide_navigation_models_star_simulated`).
- **Bodies** (:doc:`dev_guide_navigation_models_bodies`) —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` (catalog-driven; one instance per
  body whose bounding box overlaps the extended FOV) and
  :class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated`
  (simulated-image GUI variant). Titan is a body too, but its surface is hidden by haze, so
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` emits a haze-envelope feature
  instead of shape features, and
  :class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated` is its
  simulated-image sibling.
- **Rings** (:doc:`dev_guide_navigation_models_rings`) —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` (catalog-driven; one instance per
  planet whose ring system is configured and visible) and
  :class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`
  (simulated-image GUI variant).

Shared annotation helpers live on
:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase` (body silhouette
+ label rendering) and
:class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase` (per-edge
polyline + label rendering).

The API surface is summarized under
:doc:`/api_reference/api_nav_model`.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_stars
   dev_guide_navigation_models_bodies
   dev_guide_navigation_models_rings
