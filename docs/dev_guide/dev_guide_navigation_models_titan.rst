==========================================================
Titan Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` is the navigation model for a body
whose atmosphere hides its surface. Titan's thick haze means its visible edge is the haze top:
wavelength-dependent, hundreds of kilometres above the solid body, and not even circular at
high phase. Ellipsoid limb, terminator, and disc navigation are therefore systematically wrong
on Titan rather than merely noisy, so the shape-based
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` skips it and this model takes the
slot instead.

What the model renders is the geometry a haze navigator needs, not a silhouette: the geometric
disc center, the image-plane direction toward the sub-solar point (the axis the haze is
mirror-symmetric about), the solid and envelope radii, the image scale, and a mask of the
pixels the fit must ignore. It packages all of that into exactly one
:data:`~spindoctor.feature.feature_type.NavFeatureType.TITAN_LIMB` feature, which
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav` consumes (see
:doc:`dev_guide_techniques_titan_haze`).

Because this haze behavior is unique to Titan among the navigated bodies -- its atmosphere is
even transparent at some wavelengths -- Titan is a single hardcoded special case named by the
:data:`~spindoctor.nav_model.nav_model_body.TITAN_BODY_NAME` constant in
:mod:`spindoctor.nav_model.nav_model_body`. There is no config list, and the handling does not
generalize to other thick-atmosphere bodies such as Venus.

**Always emit.** For every frame with Titan inside the extended FOV whose geometry evaluates,
the model emits its feature. Frame
quality lives in that feature's reliability, never in a refusal: an envelope that cannot clear
the detector wherever the true pointing puts it, one too heavily occluded, or one too small to
measure scores exactly zero, and the standard per-type reliability gate then removes it. A
marginal Titan therefore resolves through the same statuses as any other marginal scene --
``all_features_gated`` on a Titan-only frame -- with the reliability breakdown on the gate
record naming the cause. There is no Titan-specific status reason anywhere in the codebase.

Theory
======

Geometry inputs
---------------

:func:`~spindoctor.nav_model.titan_geometry.geometry_from_obs` computes the whole payload at
the predicted pointing and returns it as a frozen
:class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs`. Every ``oops`` and
star-catalog query lives behind that one function, so all the emission logic downstream of it
is a pure function of a dataclass and is testable with no observation and no SPICE kernel.

- **Predicted center.** The body's projected field-of-view center plus the extended-FOV margins
  -- the exact geometric disc center, deliberately NOT the lit-weighted centroid that body
  features use. That centroid is phase-biased along the sun direction, which is exactly the
  axis this method fits.
- **Scale and radii.** The image scale is the mean of the per-axis center resolutions at the
  body. The solid radius follows from the body's own radius; the envelope radius adds the
  configured ``titan.atmosphere_height``. No modified body is registered with ``oops``: nothing
  here needs an inflated body in the inventory, and mutating a process-wide registry to obtain
  two numbers would be a poor trade.
- **Symmetry axis.** The incidence-angle backplane is evaluated over the envelope bounding box
  and the pixel of MINIMUM incidence is taken, at every phase. That pixel always projects in
  the sunward image direction; the maximum-incidence pixel is the anti-solar surface point,
  which becomes visible above 90 degrees of phase and points the wrong way. The box is never
  clipped, because the sunward pixel may lie outside the frame, so above
  ``titan.navigation.backplane_max_samples`` samples it is strided instead. Striping it, as
  the mask box is striped, would bound the memory just as well and would cost no accuracy,
  but the work is one evaluation per sample however the samples are grouped, and it is the
  work that is unbounded here; striding is the only bound that reaches it. The stride is the
  smallest integer fitting the box inside the cap, so it is the ceiling of the box side over
  the square root of the cap, and just above a threshold that ceiling is up to twice the
  continuous value, using only a quarter of the cap. The axis angle is the direction from the
  predicted center to the minimum-incidence pixel, whose arm is the solid radius times the
  sine of the phase, so the quantization at a million samples is about 0.15 degrees at 90
  degrees of phase, 0.29 at 30 and 0.84 at 10 (the envelope-to-solid ratio is 1.27 with the
  700 km atmosphere), or 0.30, 0.58 and 1.68 at worst -- either way inside the plus or minus
  5 degree window the angle refinement searches, down to about 3.5 degrees of phase. When
  the center and that pixel are closer together than ``axis_min_offset_px``, applied at the
  sampling quantum in use -- that is, multiplied by the stride -- the phase is near zero, the
  disc is nearly rotationally symmetric, any axis is equally valid, and the model reports a
  degenerate axis so the technique skips angle refinement -- the limb arc still constrains the
  center, because the whole limb is circular there.
- **Contaminant mask.** A boolean array of the pixels the fits must ignore, built undilated at
  the predicted geometry over a box large enough to cover everything the fits can reach. The
  box is clipped to the extended field of view before evaluation, which changes no result:
  pixels outside the extended frame are discarded when the mask is embedded, and again when
  it is intersected with the box region. The box is evaluated
  :data:`~spindoctor.nav_model.titan_geometry.OCCLUDER_STRIP_ROWS` rows at a time, with one
  backplane per strip answering both the body and the ring occlusion questions. Both answers then hold to within the photon solver's convergence tolerance
  rather than to the last bit; see :doc:`dev_guide_memory`.

Contaminant mask components
---------------------------

Four components are unioned:

- **Body occluders** -- nearer bodies covering Titan, computed by
  :func:`~spindoctor.nav_model.nav_model_body.occluder_mask_for_body`, the same helper
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` uses for its own body-body
  occlusion.
- **Ring occluders** -- pixels whose ring-plane intercept radius falls inside
  ``ring_occlusion_radii_km`` and whose ring intercept is nearer than the body. The main rings
  are treated as opaque; a frame where Titan shows through the C ring or a gap gates out at
  zero reliability rather than being fitted through ring stripes.
- **Sibling footprints** -- the inventory bounding box of every other in-FOV body regardless of
  range order. A moon behind Titan occludes nothing, but its visible sliver beside the limb
  sits exactly in the symmetry annulus and in the arc rays. Box masking is deliberately
  conservative here: a moon entirely hidden behind Titan costs a box-sized patch of mirror
  pairs, which the coverage gates meter.
- **Bright point sources** -- discs of radius ``star_mask_radius_px`` at the predicted positions
  of catalog stars brighter than ``star_mask_vmag_limit``. The queries go to the photometry
  reference catalogs (YBSC below magnitude 6.5, Tycho-2 above it) and never to the bright end of
  UCAC4, whose merged magnitudes can run several magnitudes too faint exactly in this range.

The mask ships undilated at the predicted geometry, and the consuming technique shifts it by
its current center hypothesis: a pointing error translates the whole scene identically, so
every relative position in the mask is exact. Alignment and dilation are the fitting code's
job, not the model's.

Reliability
-----------

The score is the product of a size sigmoid and an occlusion factor,

.. math::

    R = \sigma\!\left(
        \frac{D - \mathtt{reliability\_diameter\_midpoint\_px}}
             {\mathtt{reliability\_diameter\_scale\_px}}
        \right) \cdot (1 - f_{\mathrm{occ}}),

with :math:`D` the apparent envelope diameter in pixels and :math:`f_{\mathrm{occ}}` the
fraction of the envelope disc hidden by a nearer body or by the rings. It is forced to exactly
``0.0`` under three hard conditions, each of which makes the fit unusable rather than merely
imprecise:

- the envelope disc, dilated PER IMAGE AXIS by that axis's own extended-FOV margin, does not fit
  inside the extended frame. Full visibility is a property of the body's TRUE position, which
  can sit anywhere inside the pointing window, so a predicted-visible but actually-clipped frame
  would fit sky. The dilation is per axis because the extended frame is the detector plus those
  two margins, so an axis-matched dilation says exactly "the envelope clears the detector"; a
  scalar dilation by the larger margin would shrink the admissible region on the tighter axis by
  the difference between them, which is 90 px per side on a Cassini NAC;
- the occluded fraction exceeds ``max_occluded_fraction``;
- the envelope diameter is below ``min_envelope_diameter_px``.

A zero can never clear the 0.30 ``TITAN_LIMB`` type threshold, so a hard condition is exactly as
strong as a refusal to emit -- but it travels through the standard
:class:`~spindoctor.feature.reliability.FeatureReliabilityGate` machinery and leaves an
attributable record instead of a silent absence. With the shipping defaults the sigmoid crosses
0.30 near the hard floor itself, at a diameter of about 40.1 px with no occlusion, rising to
about 42.3 px at the maximum permitted occluded fraction. A frame in that narrow band emits a
feature the type gate then removes; that is a sanctioned terminal state, not a defect.

Restrictions and assumptions
----------------------------

- The envelope radius assumes a single configured atmosphere height for every filter and epoch.
  It bounds the search annulus and the ray windows only; the fit itself has a free radius and
  makes no haze-altitude assumption, so an approximate envelope costs search width, not
  accuracy.
- The rings are opaque over the configured radius range.
- The sibling-body mask is a bounding box, not a silhouette.
- Faint stars, cosmic rays, and hot pixels are deliberately unmasked; see
  :doc:`dev_guide_techniques_titan_haze` for why.

Sources of uncertainty
----------------------

The model reports no per-feature position covariance: the feature is a prediction of where a
haze envelope should be, and the uncertainty of the measurement against it belongs to the fit.
What the model does report is the reliability breakdown -- the envelope diameter and the
occluded fraction -- which is what decides whether the fit is attempted at all.

**Answered conditions and propagated faults.** The model emits its one feature for every frame
it can read, so a clipped or off-edge frame reaches the gate with an attributable record. Three
conditions are answered with a degenerate geometry and reliability forced to zero: an inventory
field that is not finite, an image scale or body radius that is not finite and positive, and an
envelope box with no surface intercept around a body narrower than the sampling stride. Any other
failure in the geometry -- a backplane that cannot be evaluated, a star query that fails, a box
with no intercept around a body the stride cannot miss -- propagates, and the orchestrator fails
the image with ``status_reason=internal_error`` naming the component.

Configuration
=============

``config_060_titan.yaml`` carries the whole consumed schema. The emission-side keys are here;
the two fit blocks (``titan.navigation.symmetry`` and ``titan.navigation.arc``) are documented
with the technique that reads them, at :doc:`dev_guide_techniques_titan_haze`.

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Key
     - Default
     - Effect
   * - ``titan.atmosphere_height``
     - ``700``
     - Haze envelope above the solid radius, in kilometres. Sets the envelope radius that bounds
       the search annulus and the ray windows.
   * - ``titan.navigation.min_envelope_diameter_px``
     - ``40.0``
     - Hard-zero floor: an envelope smaller than this scores zero reliability.
   * - ``titan.navigation.max_occluded_fraction``
     - ``0.10``
     - Hard-zero floor on the occluded fraction of the envelope disc.
   * - ``titan.navigation.ring_occlusion_radii_km``
     - ``[74490.0, 140500.0]``
     - Ring-plane radius range treated as opaque, from the C-ring inner edge to just outside the
       F ring.
   * - ``titan.navigation.axis_min_offset_px``
     - ``3.0``
     - Predicted-center-to-minimum-incidence distance below which the symmetry axis is reported
       degenerate. Applied at the sampling quantum in use: multiplied by the stride the
       incidence backplane is evaluated at.
   * - ``titan.navigation.backplane_max_samples``
     - ``1000000``
     - Largest sample count the symmetry-axis incidence backplane is evaluated over. A larger
       envelope box is strided to fit, by a whole number of pixels, costing the angular
       resolution described above; zero or negative imposes no bound.
   * - ``titan.navigation.recenter_threshold_px``
     - ``8.0``
     - Along-track shift above which the technique runs its second pass; also the mask's
       along-axis dilation in that pass.
   * - ``titan.navigation.star_mask_vmag_limit``
     - ``8.0``
     - Catalog V magnitude above which a star is masked.
   * - ``titan.navigation.star_mask_radius_px``
     - ``4.0``
     - Radius of each masked star disc, in pixels.
   * - ``titan.navigation.reliability_diameter_midpoint_px``
     - ``52.0``
     - Midpoint of the reliability size sigmoid, chosen so reliability crosses the 0.30
       ``TITAN_LIMB`` type gate near the hard-zero diameter floor.
   * - ``titan.navigation.reliability_diameter_scale_px``
     - ``14.0``
     - Width of that sigmoid.
   * - ``titan.navigation.surface_window_filters``
     - ``[CB3]``
     - Filters that see through the haze to the surface. Sets the feature's
       ``surface_window_filter`` flag; the fit does not branch on it.
   * - ``titan.navigation.high_phase_deg``
     - ``150.0``
     - Phase angle above which the feature's ``high_phase`` flag is set,
       marking frames whose sunward arc carries its least support.
   * - ``titan.annotation.gated_dot_spacing``
     - ``4``
     - Sample spacing that dots the overlay's curves for a feature below
       the per-type reliability gate; raise it when
       ``bodies.outline_thicken`` is nonzero, since thickening closes the
       one-pixel gaps.
   * - ``titan.annotation.center_marker_half_px``
     - ``4``
     - Half-length of the cross drawn at the disc center.

The per-type reliability threshold the emitted feature is gated against is the ``TITAN_LIMB``
entry under ``reliability_gate`` in ``config_540_orchestrator.yaml``, mirrored in
:data:`~spindoctor.feature.reliability.DEFAULT_RELIABILITY_THRESHOLDS`; the two must stay in
sync.

Implementation
==============

Source files:

- ``src/spindoctor/nav_model/nav_model_titan.py`` --
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`, the reliability formula
  (:func:`~spindoctor.nav_model.nav_model_titan.titan_haze_reliability`), the feature builder
  (:func:`~spindoctor.nav_model.nav_model_titan.build_titan_feature`), and the overlay
  rasterizer (:func:`~spindoctor.nav_model.nav_model_titan.haze_overlay`), all pure functions of
  the geometry dataclass.
- :mod:`spindoctor.nav_model.titan_geometry` --
  :class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs` and
  :func:`~spindoctor.nav_model.titan_geometry.geometry_from_obs`, the observation-side half.

Public class :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan`, base
:class:`~spindoctor.nav_model.nav_model.NavModel`. Self-registers via ``__init_subclass__`` so
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` discovers it.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.instances_for_obs` -- returns one
  instance named ``titan:TITAN`` when Titan is inside the extended FOV, else none.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.create_model` -- evaluates the haze
  geometry and records it in
  :attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`, inside the ``TITAN MODEL`` logger
  section.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_features` -- returns exactly one
  ``TITAN_LIMB`` feature with its reliability and breakdown.
- :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations` -- returns the
  overlay described below.
- :attr:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.geometry_inputs` -- the evaluated
  :class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs`, computed on first access.

Feature payload
---------------

The emitted feature carries ``feature_id`` ``'titan_limb:TITAN'``, a
:class:`~spindoctor.feature.geometry.TitanHazeGeometry` payload (predicted center, symmetry-axis
angle and its degeneracy flag, phase, solid and envelope radii, image scale, contaminant mask,
filter names, and the envelope bounding box), and
:class:`~spindoctor.feature.flags.TitanHazeFlags` (the body name, whether the filter is a
surface window, and whether the phase is high enough that the sunward limb has shrunk toward a
crescent). Its reliability breakdown populates
:attr:`~spindoctor.feature.feature.NavReliabilityBreakdown.titan_envelope_diameter_px` and
:attr:`~spindoctor.feature.feature.NavReliabilityBreakdown.titan_occluded_fraction`, which reach
the per-image JSON through
:attr:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary.reliability_reasons`.

Annotations
-----------

The overlay draws the predicted envelope circle, the symmetry axis, the sunward arc sector, and
a center cross. Because the summary PNG composites annotations at the navigated offset, the
drawn geometry lands on the fitted position on a committed frame and stays at the prediction
when no offset was committed -- the same way every other model's overlay reports its technique's
answer. The style encodes the quantity the model itself knows: solid curves when the feature's
reliability is at or above the per-type gate threshold, dotted curves plus a
``TITAN (low reliability)`` label below it. The label says reliability rather than "gated"
because manual navigation renders the same overlay with the gate deliberately skipped.

Examples
========

A Titan-only frame whose envelope is well resolved and unoccluded emits a high-reliability
feature, :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav` runs, and the
frame commits with an anisotropic covariance and a ``medium``-tier confidence.

A Titan-only frame whose envelope is clipped by the detector edge, or occluded past
``max_occluded_fraction`` by the rings, emits a zero-reliability feature that the type gate
removes; the frame ends ``all_features_gated`` and the gate record's reliability breakdown names
the diameter and occluded fraction that produced the zero. A frame carrying Titan plus a
resolved moon, a ring, or a star field navigates on all of its content together, with the haze
fit contributing one witness to the ensemble.
