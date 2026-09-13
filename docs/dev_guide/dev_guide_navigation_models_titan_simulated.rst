==========================================================
Simulated Titan Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated` is the
simulated-image counterpart of
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`. It subclasses the catalog-driven
model and replaces exactly one thing: how the geometry dataclass is obtained. Feature emission,
the reliability formula, the hard-zero conditions, and the overlay are inherited unchanged, so a
simulated haze frame exercises the shipping emission rules rather than a parallel
implementation of them -- which is the whole point of grading a navigator against the simulator.

The catalog-driven model computes its
:class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs` from ``oops`` backplanes and
star-catalog queries. This model builds the same dataclass from a simulated scene's idealized
body parameters. It could not have reused
:func:`~spindoctor.nav_model.titan_geometry.geometry_from_obs` in any case: every branch of that
function needs backplanes a simulated observation does not carry. Because the simulated path
reads operator parameters directly, the simulated inventory needs no projected-center key.

Model selection is exclusive in both directions.
:meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.instances_for_obs` returns nothing
for a simulated observation, this class returns nothing for a real one, and
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` excludes Titan
unconditionally -- mirroring the real path's exclusion, so exactly one model claims a simulated
Titan.

Theory
======

Information boundary
--------------------

Every quantity this model reads comes from the filtered ``nav_params`` view: the body's centre,
its image-plane axes, its pixel scale, its phase, and its illumination direction -- catalog
geometry a real pipeline would read from SPICE. The ``atmosphere`` block that gives the rendered
body its soft haze limb is truth and is never read here. The envelope radius comes from the same
configured ``titan.atmosphere_height`` the real model uses, so the predicted envelope is a
deliberate approximation of the rendered haze exactly as it is on a real frame, and the free
radius of the arc fit is what absorbs the difference.

Parameter mapping
-----------------

Each real-frame quantity has a deliberate simulated analog:

- **Predicted centre** -- the operator's stated centre, shifted into extended-FOV coordinates.
- **Solid radius** -- the mean of the two image-plane semi-axes, the way the real model averages
  the two per-axis centre resolutions into one scale.
- **Envelope radius** -- the solid radius plus the configured atmosphere height converted
  through the scene's own pixel scale.
- **Symmetry axis** -- the scene's illumination direction, expressed in the fitting library's
  angle convention, with the same near-zero-phase degeneracy rule the real model applies.
- **Contaminant mask and occluded fraction** -- built with the same
  :func:`~spindoctor.nav_model.titan_geometry.paint_disc` and
  :func:`~spindoctor.nav_model.titan_geometry.occluded_disc_fraction` helpers the real model
  uses. A sibling body with an explicitly nearer range occludes the envelope and counts toward
  the occluded fraction; every sibling, near or far, contributes its bounding box, because a
  moon beside the limb sits in the symmetry annulus whether it hides anything or not. Scene
  stars brighter than the configured mask limit contribute a masked disc, mirroring the real
  model's catalog queries.

Coordinate convention
---------------------

A scene states every position as a CORNER coordinate -- ``(0.0, 0.0)`` is the top-left corner
of pixel ``(0, 0)`` -- so a body stated at ``center_v`` paints its silhouette centred on pixel
index ``center_v - 0.5`` (see :ref:`sim-pixel-convention`). Predicted positions in this
pipeline are pixel indices, so this model applies the half-pixel shift, captured as
:data:`~spindoctor.nav_model.nav_model_titan_simulated.BODY_CENTER_INDEX_OFFSET_PX`. Measured
directly: without the shift every simulated frame carries a flat 0.500 px cross-track error,
half the method's entire clean-scene cross-track budget spent on a coordinate convention.

Restrictions and assumptions
----------------------------

- A simulated body named ``TITAN`` that does not carry the parameters this model needs builds NO
  model at all, and -- because the simulated body model excludes Titan unconditionally -- no
  body model either. The frame then resolves through the standard generic status reasons for a
  scene with nothing to navigate, rather than through a crash or a silently degenerate feature.
  The absence is logged with the missing keys named.
- The predicted envelope is the configured atmosphere height, not the rendered haze profile;
  that mismatch is deliberate.
- Filter names are empty on a simulated frame, so the surface-window flag never fires and the
  fitted-radius-by-filter diagnostics carry no filter label.

Sources of uncertainty
----------------------

Identical to the catalog-driven model: the feature carries a reliability breakdown rather than a
position covariance, and the measurement uncertainty belongs to the fit.

Configuration
=============

The model consumes the same ``titan`` configuration block as its catalog-driven parent (see
:doc:`dev_guide_navigation_models_titan`) and adds no keys of its own.

Scene-side, the body parameters it requires are
:data:`~spindoctor.nav_model.nav_model_titan_simulated.REQUIRED_SIM_PARAMS` -- ``center_v``,
``center_u``, ``axis1``, ``axis2``, and ``km_per_pixel``. The centre and the two image-plane
axes give the predicted disc; the pixel scale is what turns the configured atmosphere height in
kilometres into an envelope radius in pixels, and without it the envelope -- the outer bound of
everything the fit samples -- would have to be invented. ``phase_angle``,
``illumination_angle``, ``rotation_z``, and ``range_km`` are read when present.

The haze the scene actually renders is configured by the body's ``atmosphere`` block, documented
at :ref:`sim-atmosphere`; those keys are truth and are invisible to this model.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_titan_simulated.py`` --
:class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated`.

Public class
:class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated`, base
:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`. Self-registers via
``__init_subclass__``.

Public members (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated.instances_for_obs`
  -- returns one instance named ``titan_sim:TITAN`` per adequately configured simulated Titan,
  and an empty list for a real observation.
- :attr:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated.geometry_inputs`
  -- the geometry dataclass built from operator parameters on first access.
- :data:`~spindoctor.nav_model.nav_model_titan_simulated.REQUIRED_SIM_PARAMS` and
  :data:`~spindoctor.nav_model.nav_model_titan_simulated.BODY_CENTER_INDEX_OFFSET_PX`.

``create_model``, ``to_features``, and ``to_annotations`` are inherited without override.

Examples
========

The ``titan_haze`` base scene (``tests/integration/sim_scenes/atmosphere/titan_haze.yaml``) is
the reference case: a body named ``TITAN`` at 60 degrees phase with a rendered haze whose
optical limb sits a few pixels above the solid radius, driving the standing dense and wide
offset sweeps for
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`.

The sibling scenes in the same directory use a body named ``HAZEMOON`` instead. They are
body-navigation fidelity records for a haze-blind navigator -- what an ellipsoid limb fit and a
disc correlation do when the frame carries a haze they do not model -- and a body named ``TITAN``
would route them to the haze model and measure something else entirely.
