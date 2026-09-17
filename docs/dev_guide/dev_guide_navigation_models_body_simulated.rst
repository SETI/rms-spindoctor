==========================================================
Simulated Body Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated` is the
simulated-image variant of the body navigation model. It renders a body from
operator-supplied ellipsoid (or polyhedral-mesh) parameters -- center, axes, rotation,
lighting -- instead of from SPICE prediction, then emits the body features the navigation
techniques consume:

- always a :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` carrying the
  rendered template, for :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`;
- a :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` (the orientation-free
  lit-weighted centroid, built by the shared
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`) when the predicted
  diameter clears the blob floor, for
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`;
- a :data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` (the silhouette boundary as
  a vertex polyline with outward normals) when the body is well resolved (diameter at
  least 100 px) and at low phase (at most 60 degrees), for
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`;
- a :data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` (the lit/unlit
  boundary *interior* to the disc, as a vertex polyline whose outward normals point from the
  lit side toward the unlit side) when the phase is far enough from zero for the terminator
  to separate from the limb and the boundary is long enough to constrain a fit, for
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`.

The terminator is distinct from the limb: the limb is the silhouette edge against sky,
while the terminator cuts *across* the disc where the lit hemisphere meets the unlit one.
Because the shading floors the visible-but-unlit hemisphere at a small constant
(:data:`~spindoctor.sim.ellipsoid_geometry.DARK_SIDE_ILLUM_STRENGTH`), the rendered body mask
(brightness above zero) is the whole visible disc, and the lit region is the part brighter
than that floor; the terminator polyline is the lit pixels adjacent to the interior unlit
disc, with the limb ring excluded so the polyline stays interior to the disc -- except at
the cusp-adjacent vertices of a very thin crescent, where terminator and limb meet within
a pixel and a handful of vertices land on the silhouette (the SPICE-backed model's
sampler shares the behavior). The
gates mirror the SPICE-backed :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`
(a ``sin(phase)`` floor, a minimum vertex count, and the shared
:func:`~spindoctor.nav_model.nav_model_body.shape_features_suppressed` policy that
suppresses the terminator of a resolved ``highly_irregular`` body), so a simulated body
offers the same feature set a real one would. The terminator technique is sim-fitted on
these emissions (2026-07-18 campaign; the fit came out single-class, so its formula is a
low plateau -- see ``config_510_techniques.yaml``), and a terminator fix stays
``confidence_provisional``: the realism match has no terminator-side rise-width verdict,
which is the condition that gates trust in its confidences.

The model overrides :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to build
one instance per body of a simulated observation; the parent
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` declines simulated observations, so the
autonomous registry routes simulated frames here.

Theory
======

Simulated body rendering is a controlled-input version of the same silhouette-extraction
pipeline that drives :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`. The operator
specifies a body in image-plane coordinates (center and per-axis radii) plus a phase /
lighting geometry, and the renderer paints the corresponding ellipsoidal body onto an
extended-FOV image plus matching mask.

The rendered template is the BODY_DISC feature payload the disc correlation navigates
against. The blob and limb features extend the simulated body across the technique ladder:
the blob is the orientation-independent fallback for small, high-phase, or irregular
bodies, and the limb is the resolved-body distance-transform fit. Which feature is
load-bearing tracks resolution and phase the same way it does on a real frame, so the
range and phase parameter sweeps (see :doc:`/simulator_report/simulator_report`) show the
primary technique transitioning limb -> disc -> blob as a body shrinks. The simulated
body's geometry is operator-known by construction, so the simulated path is the
calibration regime -- it lets a developer probe the navigation pipeline with bodies whose
true offset is known to the pixel.

Restrictions and assumptions
----------------------------

- The operator must supply finite, positive ellipsoid axes. Degenerate inputs (zero
  radius, negative axes) are rejected by
  :func:`~spindoctor.nav_model.sim_body.create_simulated_body`.
- The body appearance keys -- crater terrain, the limb-relief field, the photometric
  law and opposition surge, the albedo and disc textures and transits, the atmospheric
  haze layer, the mesh shading mode and pose scatter, and anti-aliasing -- are truth
  keys the boundary filter strips, so this model never sees them; the predicted
  template always renders as a smooth Lambert body at maximum anti-aliasing and zero
  surface relief.
- The atmosphere key deserves its own note, because its mismatch is deliberate rather
  than incidental. When the image side renders a haze layer, the limb becomes a soft
  ramp whose apparent radius sits outside the reference radius and shifts with phase
  (see :ref:`sim-atmosphere`), while this model still predicts a hard limb at the
  reference radius. That gap is the substrate for the haze altitude-versus-phase
  problem: a limb fit against the haze recovers a small offset toward the sunlit limb
  whose size tracks the phase-dependent apparent limb radius (and the haze
  parameters), and the ``atmosphere`` catalog scenes pin that measured bias. Those
  scenes carry a body named ``HAZEMOON`` so they stay in this haze-blind regime; a
  simulated body named ``TITAN`` is claimed by
  :class:`~spindoctor.nav_model.nav_model_titan_simulated.NavModelTitanSimulated`
  instead, which predicts the haze envelope rather than a hard limb.
- The simulated body is rendered onto a fixed extfov image without per-instrument noise
  or PSF smearing; the operator's downstream noise-injection pipeline supplies those.

Sources of uncertainty
----------------------

The simulated body has no measurement uncertainty by construction; the rendered template
is an exact ellipsoid. The downstream technique's reported covariance reflects only the
correlation-curvature CRLB at the chosen NCC peak.

Configuration
=============

The simulated body model consumes no YAML configuration of its own; every parameter comes
in via the per-body entry of the observation's filtered scene view
(``obs.nav_params['bodies']``, see :doc:`dev_guide_simulator`). Expected keys:

- ``name`` — body label used in metadata and the summary PNG.
- ``center_v``, ``center_u`` — the body center, in pixel-corner coordinates (see
  :ref:`coordinate-systems`).
- ``range_km`` — subject distance in km (defaults to ``+inf``).
- ``axis1``, ``axis2``, ``axis3`` — full widths of the three ellipsoid axes, in
  pixels. ``axis3`` defaults to ``min(axis1, axis2)``.
- ``rotation_z`` — rotation about the line of sight (degrees).
- ``rotation_tilt`` — tilt of the body (degrees).
- ``illumination_angle`` — degrees.
- ``phase_angle`` — degrees. Also gates LIMB_ARC emission (limb only at or below 60
  degrees).
- ``shape_model`` — ``ellipsoid`` (default) or ``polyhedral_mesh`` for an irregular body;
  a mesh reads ``mesh_lumpiness``, ``mesh_seed``, and ``pose_euler_deg`` (see
  :func:`~spindoctor.sim.mesh_geometry.mesh_spec_from_params`).
- ``km_per_pixel`` — optional physical scale at the limb; when absent the
  phase-irregularity factor collapses to the regular-body case.
- ``nav_override`` — optional scene mapping overlaid onto the body's idealized view by
  the information-boundary filter before this model sees it, separating the render
  geometry from the navigation geometry
  (see *Render geometry vs navigation geometry* below).

Crater and anti-aliasing keys never reach this model (the boundary filter strips them,
so the template is a smooth, fully anti-aliased body). The predicted silhouette diameter
gates the blob (at least 5 px) and limb (at least 100 px) emission; the diameter floor on
the limb keeps the LM-refined fit off marginally-resolved bodies, where it would inject
cross-process jitter into the fused offset.

Render geometry vs navigation geometry
--------------------------------------

In real navigation the body's pose (the body-fixed to camera rotation) is an
*input* from SPICE: the navigator renders its predicted body at that pose and
solves only for the pointing offset; it never estimates orientation from the
pixels. The simulator has no SPICE, so the pose is scene ground truth carried on
the body params. By default the predicted body is built from the same params the
renderer drew, so the navigator knows the truth (the agreeing case).

An optional ``nav_override`` mapping breaks that tie. The renderer ignores it and
always draws the true geometry; the boundary filter
(:func:`~spindoctor.sim.scene.build_nav_params`) overlays ``nav_override`` onto the
body's idealized view and drops the key, so the predicted body is built from what the
navigator *believes* without the true values underneath. This is the channel
that lets the navigation geometry diverge from the render geometry, which the
irregular-body scenarios exercise:

- **Same geometry (no override)** -- mesh vs mesh at the true pose. The
  resolved-mesh limb is exact by construction.
- **Shape mismatch** -- render a lumpy mesh, predict its zero-relief
  (ellipsoidal) limit by overriding ``mesh_lumpiness`` to ``0.0`` at the same
  pose. The only residual is shape; the disc correlation still aligns the two
  filled silhouettes and the recovered centroid bias grows with the rendered
  relief. Realizing the ellipsoidal prediction as the smooth limit of the mesh
  keeps both silhouettes on one renderer, so the residual is purely the shape
  mismatch under test.
- **Pose disagreement** -- render the mesh at the true pose, predict the same
  mesh at a different ``pose_euler_deg``. The wrong-pose silhouette boundary
  drives the limb distance-transform fit to a confidently-wrong offset, while the
  lit-weighted blob centroid -- which a centrally-symmetric (low-relief triaxial)
  body keeps near the body center under rotation -- stays accurate.

The override never changes the center, so the predicted body stays at the
unshifted position the planted offset is measured from.

Implementation
==============

Source file: ``src/spindoctor/nav_model/nav_model_body_simulated.py`` —
:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated`.

Public class :class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated`, base
:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`. The class overrides
:meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to build one instance per body
of a simulated observation; the parent
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` returns an empty list for a simulated
observation, so the orchestrator's
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` driver routes simulated frames to
this subclass.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model` —
  renders the simulated body (ellipsoid via
  :func:`~spindoctor.nav_model.sim_body.create_simulated_body`, or a mesh via
  :func:`~spindoctor.sim.mesh_geometry.render_mesh_body_image`), computes the limb mask via
  the shared :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase` helper, and
  records the predicted diameter and tight bounding box used to gate and emit features.
- :meth:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features` — emits
  the BODY_DISC plus, when the resolution and phase gates pass, the BODY_BLOB, LIMB_ARC, and
  TERMINATOR_ARC features described under *Overview*.
- :meth:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_annotations` —
  reuses the shared body annotation helper on
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase` to render body silhouette
  and labels onto the summary PNG.

Inherited :class:`~spindoctor.nav_model.nav_model.NavModel` properties:
:attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.obs`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated.create_model`:

1. Open a logged section. Read the operator-supplied sim parameters off the per-instance
   dict.
2. Convert per-axis rotations and angle parameters from degrees to radians.
3. Call :func:`~spindoctor.nav_model.sim_body.create_simulated_body` with the per-axis radii and
   geometry; the helper returns the rendered simulated body image.
4. Derive the body mask from the rendered image (every non-zero pixel is on the body).
5. Compute the limb mask via
   :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`'s shared discrete-mask
   neighbor-shift helper.
6. Promote the rendered image and the masks from sensor-shaped arrays to extfov-shaped
   arrays (zero-padded for the extfov margin).
7. Record the predicted center, the subject range, and the bounding box on the model's
   internal state for downstream feature emission.

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated.to_features`:

1. Crop the rendered template image and mask to the per-instance tight bounding box (the
   silhouette bbox plus slop, matching the SPICE-backed model, so a downstream moment
   stays local to the body rather than integrating over the whole frame).
2. Construct one
   :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC`
   :class:`~spindoctor.feature.feature.NavFeature` carrying the cropped template image, the
   cropped mask, the predicted center, the subject range, and a
   :class:`~spindoctor.feature.flags.BodyDiscFlags` with the operator-supplied body name plus
   ``overflow_fov_fraction = 0.0``.
3. When the predicted diameter clears the blob floor, append a BODY_BLOB built by the
   shared blob-feature helper on
   :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
4. When the diameter and phase gates pass, append a LIMB_ARC scored honestly: the
   silhouette boundary is sampled into a vertex polyline with outward normals and a
   fixed per-vertex sigma, vertices hidden behind an explicitly nearer sibling body are
   dropped, and the visible-arc fraction compares the surviving polyline against the
   silhouette boundary of an unclipped whole-body render -- so a limb sliding off the
   frame or behind another body reports the loss instead of claiming ``1.0``.  The
   reliability applies the shared
   :func:`~spindoctor.nav_model.nav_model_body.limb_reliability` formula (arc fraction
   and arc length), so a deeply occluded or clipped limb can be dropped by the
   downstream reliability gate exactly as a real one would be.
5. When the terminator gates pass, append a TERMINATOR_ARC scored the same way: its
   visible-arc fraction compares the surviving lit/unlit ridge (net of frame clipping
   and sibling-body occlusion) against the ridge of an unclipped whole-body render of
   the same geometry, and its reliability applies the shared
   :func:`~spindoctor.nav_model.nav_model_body.terminator_reliability` formula
   (arc fraction, catalog albedo-variation penalty, ``sin(phase)`` cap) -- so a
   high-phase simulated terminator can be dropped by the downstream reliability gate
   exactly as a real one would be.
6. Reliability on the disc feature is fixed at ``1.0`` (the rendered template is exact by
   construction and the correlation technique carries its own quality diagnostics),
   while the blob carries the shared detection-SNR reliability from the base class.
   Body-body occlusion is resolved from the sibling bodies' idealized geometry: each
   model instance receives the OTHER bodies of the same filtered scene view, and a
   sibling occludes only when both bodies carry an explicit ``range_km`` and the
   sibling's is strictly nearer -- the navigator-side mirror of the renderer's rule
   that overlapping bodies must all declare their depth.  The disc template itself is
   not occlusion-masked (see the simulator report's known-gaps list).

Examples
========

The simulated body model is consumed by the simulated-image GUI driver
(``sd_create_simulated_image``). An operator specifies a body — say a Mimas-like
ellipsoid centered at ``(512, 512)`` with axes ``200`` pixels across, illumination angle ``60``
degrees, phase angle ``30`` degrees — and the simulator renders the corresponding
extended-FOV image plus mask. The downstream
:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` correlates the
template against an injected synthetic-noise image and recovers the operator-known
``(0, 0)`` offset (or whatever offset the operator injected) within sub-pixel. The
operator uses the residual to validate per-instrument plate-scale and PSF assumptions
without a real Cassini observation.
