==========================================================
Simulated Star Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated` is
the simulated-image counterpart of
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`. It is a thin subclass
of the catalog-driven model: it builds its star list from the scene's catalog star
entries in the filtered idealized view (``obs.nav_params``, see
:doc:`dev_guide_simulator`) rather than from a catalog reduction, and inherits the parent's
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` feature emission, CRLB
covariance, reliability gate, and annotations unchanged. A simulated star field is
therefore navigated by exactly the same
:class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`, and
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` code a real frame
is.

Unlike :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`, this model *does*
override :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs`: it builds one
instance for a simulated observation that rendered at least one star, and the parent
declines simulated observations, so the autonomous registry routes simulated frames to
this subclass and real frames to the parent.

Theory
======

The simulated path is the calibration regime for the star techniques: a developer can
probe star matching with a field whose true offset, photometry, and (planted) camera
roll are known by construction.

The image-side renderer (``spindoctor.sim.forward.star``) draws each scene star at its
catalog ``(v, u)`` shifted by the scene's planted offset and camera roll. This model
independently builds the *unshifted* catalog positions from the same scene entries --
via the shared record builder :func:`~spindoctor.sim.star_records.star_record_from_params`,
so the two sides' defaults are identical while no rendered values cross the information
boundary -- and adopts them as its prediction. A technique that detects the shifted
peak therefore recovers the planted transform: the same prediction / observation split
a real navigation has, which is why the recovery transfers.

Two rendering details make the simulated field faithful to a real one:

- **Where the half pixel goes.** A star record states its position in pixel-corner
  coordinates and the render grid is pixel-centric (see :ref:`coordinate-systems`), so
  the renderer subtracts the half pixel -- scaled by the oversample factor alongside
  every other pixel quantity -- before it deposits the star's total flux as a bilinear
  point mass. A further ``(oversample - 1) / 2`` shift lands the deposit on the
  oversampled grid, so a star the model predicts at ``(v, u)`` centroids there after
  the box downsample, with no half-pixel bias in the recovered offset.
- **Camera roll about the boresight.** A planted ``offset_rotation_deg`` rotates each
  star about the frame's center ``(size_v / 2, size_u / 2)`` -- the same point the
  bodies and the ring system turn about -- before the translation offset, while the
  star record keeps its unrolled ``(v, u)``; the similarity fit in
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` then
  recovers the roll. See :doc:`dev_guide_rotation`.

Restrictions and assumptions
----------------------------

- A simulated star field is clean by construction: there are no body- or ring-occlusion
  conflicts to mark, and the per-image smear vector is zero (the sim renders no attitude
  rate). The reliability gate's occlusion term still applies the same way the parent
  computes it; the saturation / cosmic-ray mask is not consulted to gate stars in either
  the parent or the simulated model.
- The synthesized effective SNR follows the parent's magnitude-margin formula against
  ``obs.star_max_usable_vmag()``; on a simulated observation that limit is generous, so
  reliability saturates toward 1.0 and the recovered *offset* -- not the calibrated
  confidence -- is the quantity the simulated star tests assert.

Sources of uncertainty
----------------------

Per-feature uncertainty is the parent model's anisotropic CRLB covariance, derived from
the predicted SNR and the (zero) smear vector. The simulated star positions themselves
are exact, so any recovered-offset error reflects detection-centroid noise under the
rendered detector noise, not a prediction error.

Configuration
=============

The model consumes no YAML configuration of its own; it reads the ``stars`` entries of
the observation's filtered scene view (``obs.nav_params``). The per-star geometry comes
from the scene's ``stars`` entries (see :doc:`dev_guide_simulator`); the truth-side
per-star ``psf_sigma`` override is stripped by the boundary filter, so the model knows
only the instrument's published PSF.

Implementation
==============

Source file: ``src/spindoctor/nav_model/stars/nav_model_stars_simulated.py`` --
:class:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated`, base
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`. The subclass self-registers
via ``__init_subclass__``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated.instances_for_obs`
  -- returns one instance for a simulated observation whose ``nav_params`` lists at
  least one star; an empty list for a real observation or a simulated one whose scene
  has no stars.
- :meth:`~spindoctor.nav_model.stars.nav_model_stars_simulated.NavModelStarsSimulated.create_model`
  -- builds one catalog record per ``nav_params`` star entry via
  :func:`~spindoctor.sim.star_records.star_record_from_params`, sets a zero smear
  vector, and populates the same metadata fields the parent records.
- ``to_features`` / ``to_annotations`` -- inherited unchanged from
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`.

Examples
========

A clean planted-offset star field plus the camera-roll fixture live under
``tests/integration/sim_scenes/algorithmic_invariants/`` (``planted_offset_star_field``,
``planted_rotation_star_field``); ``tests/integration/test_sim_algorithmic_invariants.py``
asserts the planted offset and roll are recovered. See
:doc:`dev_guide_simulator` for the scene-catalog workflow.
