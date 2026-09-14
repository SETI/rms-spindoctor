=====
Stars
=====

Star navigation models render the predicted appearance of catalog stars in the field of
view and emit one :class:`~spindoctor.feature.feature.NavFeature` per surviving star.
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` derives from
:class:`~spindoctor.nav_model.nav_model.NavModel` directly and the simulated sibling
subclasses it; the
shared catalog reduction, conflict marking, and photometry helpers live in the
:mod:`spindoctor.nav_model.stars` subpackage.

Registered concrete subclasses:

- :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` — catalog-driven star
  navigation; one instance per observation. Documented at
  :doc:`dev_guide_navigation_models_star`.
- :class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated` —
  the simulated-image sibling. Emits STAR features exactly the way ``NavModelStars``
  does, but builds the star list from the scene's catalog entries in the filtered
  idealized view (``obs.nav_params``) rather than reducing real catalogs. Documented at
  :doc:`dev_guide_navigation_models_star_simulated`.

The :mod:`spindoctor.nav_model.stars` subpackage carries the catalog reduction
(``catalog.py``), per-star body / ring conflict marking (``conflicts.py``), the raw-DN
photometry diagnostic and B-V color mapping (``predicted_snr.py``), smear-aware PSF
construction (``smeared_psf.py``), on-image source detection (``detection.py``), and the
simulated-scene model (``nav_model_stars_simulated.py``). Each helper is independently
testable so the per-step assumptions can be exercised in isolation.

.. toctree::
   :maxdepth: 4

   dev_guide_navigation_models_star
   dev_guide_navigation_models_star_simulated
