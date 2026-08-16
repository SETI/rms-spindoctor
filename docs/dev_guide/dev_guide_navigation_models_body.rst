==========================
Body Navigation Model
==========================

Overview
========

The catalog-driven body navigation model renders one solar-system body's predicted appearance
from SPICE prediction and emits up to four navigation features per body: a polyline along the
lit limb, a polyline along the terminator, a pixel template of the lit disc, and a centroid /
bounding-box description for under-resolved or irregular bodies. The orchestrator builds one
instance per body whose extended-FOV bounding box overlaps the observation; a Saturn fly-by image
that catches Mimas, Tethys, and Dione in the same frame produces three instances, each free to
emit any subset of the four feature types its silhouette geometry justifies.

Two body classes are excluded from shape-based navigation. Titan builds no
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` at all: its opaque haze hides the
surface, so ellipsoid limb / terminator / disc navigation is systematically wrong. The
:doc:`Titan model <dev_guide_navigation_models_titan>` navigates it from the haze's solar
symmetry instead. Bodies tagged ``highly_irregular`` in the shape table (chaotic rotators, small
potato moons) still build a body model but suppress their shape features once resolved beyond
a few pixels, falling back to the point-like
:data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` (see *Feature emission gates*
below).

A simulated-image sibling (:class:`~spindoctor.nav_model.nav_model_body_simulated.NavModelBodySimulated`)
renders a body from operator-supplied ellipsoid parameters instead of SPICE prediction; both
classes share the silhouette / annotation helpers on
:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`. Per-body shape, albedo, and SPICE
ephemeris-residual quantities feed in from
:func:`~spindoctor.nav_model.body_shape.load_body_shape`, which overlays an operator-curated YAML on top
of a hard-coded :data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE` and a
:data:`~spindoctor.nav_model.body_shape.DEFAULT_BODY_SHAPE` fallback.

Theory
======

The body model is a per-body silhouette renderer whose output is a small number of
*navigation features* — geometric primitives that downstream techniques know how to align
against the image. A body in the field of view contributes between zero and four such features
depending on its predicted geometry.

Bounding-box construction
-------------------------

The model starts from the predicted bounding box reported by the per-image inventory query
(``u_min_unclipped``, ``u_max_unclipped``, ``v_min_unclipped``, ``v_max_unclipped``) — the
half-open extent of pixels intersected by the predicted body silhouette before it is clipped
into the sensor. Inventory bounding boxes are sometimes a half-pixel too small; the model
inflates each axis by a fixed fraction of its current extent before clipping into the
extended-FOV grid. The 5 % default fraction keeps anti-aliased limb pixels from being lost
on the boundary while remaining small enough that the rendered postage-stamp does not balloon
on an in-FOV body.

After inflation the bounding box is clipped against the per-instrument extended-FOV margins.
A bounding box that collapses to a single point along either axis is bumped by one pixel so
the downstream meshgrid is non-degenerate. The model also computes a boolean
``guaranteed_visible_in_fov`` quantity: True when the inflated, clipped bounding box lies
fully inside the sensor area minus the per-instrument margin. The flag is recorded on
``self._metadata`` for the curator and is consumed by the disc-emission gates downstream.

Anti-aliased silhouette extraction
----------------------------------

The model lays an oversampled grid of pixel centres inside the clipped bounding box. The
per-axis oversample factor is

.. math::

    \mathrm{oversample}_{u} = \min\!\left(
      \mathrm{oversample\_maximum},\;
      \max\!\left(1,\; \left\lfloor
        \frac{\mathrm{oversample\_edge\_limit}}
             {\max(1,\, \lceil u_{\mathrm{pixel\_size}} \rceil)}
      \right\rfloor\right)\right)

and analogously for the v axis, using the inventory's ``u_pixel_size`` and ``v_pixel_size`` —
the body's predicted angular extent in pixels. A 100-pixel body at the default settings
(``oversample_edge_limit = 512``, ``oversample_maximum = 2``) yields ``oversample_u = 2``. A
sub-pixel body (predicted extent 1 px) would saturate the formula at ``oversample_u = 2`` per
the maximum. The cap exists to bound memory on field-filling targets where a higher
oversample would balloon the backplane query.

At each oversampled grid point an incidence-angle backplane is queried from the SPICE
prediction. The discrete silhouette mask comes from the pixels where the incidence query
returns a finite value (i.e. the line of sight intersects the body). The oversampled mask is
filter-downsampled to the extended-FOV grid so that anti-aliased limb pixels at the silhouette
boundary remain represented as fractional values rather than being lost to round-off.

From the discrete silhouette mask the model derives:

- a **limb mask** of valid silhouette pixels with at least one off-body neighbour;
- a **terminator mask** of lit silhouette pixels with at least one neighbouring dark pixel
  (incidence at or above 90 degrees);
- a **lit-and-in-FOV count** used by the body-disc gates;
- a **predicted brightness image** that is either the binary silhouette or the per-pixel
  Lambert cosine of the incidence angle, optionally scaled by the per-body geometric albedo.

The brightness image is the template downstream correlation techniques use; the limb and
terminator masks are walked to produce per-vertex polylines.

Polyline sampling
-----------------

Each polyline vertex is a pixel that survived the discrete-mask construction. At every vertex
the model records:

- The vertex position in the extended-FOV pixel frame, refined onto the sub-pixel boundary
  (see "Sub-pixel boundary refinement" below).
- The outward-pointing unit normal, estimated from the mask gradient: a body-side neighbour
  contributes a +1 in the outward direction, an off-body neighbour contributes a -1, and the
  resulting two-component vector is normalised.
- The local incidence angle at the vertex (radians).
- The local kilometres-per-pixel scale at the vertex, queried from the SPICE backplane.

The km/px scale at the limb sets the sensitivity that converts physical km uncertainties into
pixel sigmas. An empty mask collapses the sampler to zero-length arrays so the downstream
emission gates skip the corresponding feature without special-case handling.

Sub-pixel boundary refinement
-----------------------------

A ridge pixel names the boundary only to whole-pixel precision, so a polyline built from
ridge-pixel centers is quantized to the pixel grid: a sub-pixel change in the predicted
pointing re-rasterizes the ridge instead of translating it, and a fit against the polyline
inherits a sub-pixel-phase-dependent bias of up to half a pixel that breaks shift
equivariance (re-navigating with corrected pointing does not return the correction). After
the samplers are built, :mod:`spindoctor.nav_model.silhouette_probe` therefore moves every
vertex onto the probed sub-pixel boundary: a short ladder of probe points is laid along the
vertex's outward normal (1/8 px steps from 1 px inward to 1.5 px outward), each probe is
classified inside / outside the region by evaluating the body geometry at that exact line of
sight (an oops ``Backplane`` over a scattered ``Meshgrid`` -- about 21 probes per vertex,
negligible next to the silhouette render), and the vertex moves to the midpoint of the two
probes bracketing the boundary. The limb probes the body-intercept silhouette; the
terminator probes the lit region. The probing is purely geometric and independent of the
rendering oversample factor, so it stays sub-pixel-accurate for bodies too large for
oversampled rendering. Per-vertex incidence and km/px keep their ridge-pixel samples: they
vary smoothly at the pixel scale, so re-sampling them at the refined position would change
nothing measurable. A vertex whose crossing is undetermined (no boundary within the ladder)
keeps its ridge-pixel position.

Per-vertex position covariance
------------------------------

The per-vertex normal sigma — the prior standard deviation of how far the true limb may lie from
the predicted vertex along the outward normal — is the quadrature sum of four physical
uncertainties for the lit limb:

.. math::

    \sigma_{n}(i)^{2}_{\mathrm{km}} = \sigma_{\mathrm{ellipsoid}}^{2}
                                    + \sigma_{\mathrm{crater}}^{2}
                                    + \bigl(f_{\mathrm{inc}}(i) \cdot \sigma_{\mathrm{soft}}(i)\bigr)^{2}
                                    + \sigma_{\mathrm{spk}}^{2}

The first two terms are intrinsic to the body shape (RMS deviation from the best-fit ellipsoid;
characteristic crater scale). The third term is the photometric softness of the limb at this
vertex; :math:`\sigma_{\mathrm{soft}}(i)` is the optical PSF sigma converted to kilometres at the
vertex (PSF-sigma-px times km-per-pixel-at-vertex), and :math:`f_{\mathrm{inc}}(i)` is a
dimensionless incidence-angle penalty that grows as the local incidence approaches 90 degrees
and the limb fades photometrically. The penalty is

.. math::

    f_{\mathrm{inc}}(i) = \min\!\left( f_{\mathrm{cap}},\;
        \frac{1}{\cos\!\left(\min(i_{\mathrm{clip}}, i_{\mathrm{cap}})\right)} - 1 \right)

with the clip / cap angles and the maximum :math:`f_{\mathrm{cap}}` set as project-wide
constants in :mod:`spindoctor.feature.constants`. The last quadrature term is the SPK ephemeris
uncertainty projected to the limb plane. Dividing through by km/px gives the per-vertex pixel
sigma the polyline carries on its geometry payload.

The terminator polyline carries the same form plus an additional albedo-variation term and a
photometric-softness term, both proportional to :math:`\sigma_{\mathrm{soft}}^{2}`, because the
terminator is photometrically rather than geometrically defined and is therefore sensitive to
disc albedo variation in a way the lit limb is not.

Per-vertex tangent sigma is set to a small constant matching the polyline-sampling resolution;
the DT-based fit treats motion along the polyline as essentially unobservable by construction.

.. _body-visible-lit-fraction:

Visible-lit and overflow fractions
----------------------------------

Two scalars derived from the discrete masks gate the disc-template emission:

.. math::

    \mathrm{visible\_lit\_fraction} =
        \frac{|\mathrm{body\_mask} \,\cap\, \mathrm{lit\_mask} \,\cap\, \mathrm{sensor}|}
             {|\mathrm{body\_mask}|},
    \qquad
    \mathrm{overflow\_fraction} =
        1 - \frac{|\mathrm{body\_mask} \,\cap\, \mathrm{sensor}|}{|\mathrm{body\_mask}|}.

The visible-lit fraction measures the portion of the *whole predicted disc* (lit + dark
together) that is both lit and inside the sensor FOV — *not* the lit hemisphere alone, which
would always score near unity for a fully-in-frame body and lose discriminating power for the
disc gate. The overflow fraction measures the portion of the predicted disc that is outside
the sensor FOV.

A lit pixel a nearer body hides (see :ref:`body-body occlusion <body-occlusion>`) is not
usable disc-template support, so it is excluded from the visible-lit numerator while the
denominator stays the whole predicted disc. At deep mutual-event overlap the visible-lit
fraction therefore falls toward the surviving crescent, which drops the disc reliability and,
below the minimum, gates the disc feature out entirely.

Lit-weighted predicted centroid
-------------------------------

The blob path's predicted centroid is the brightness-weighted moment of the rendered model
inside the body silhouette:

.. math::

    \bar{x}_{\mathrm{lit}} =
        \frac{\sum_{(v,u) \in \mathrm{body}} I(v, u) \,(v, u)}
             {\sum_{(v,u) \in \mathrm{body}} I(v, u)}.

Predicting the lit-weighted centroid up front means the navigation offset the technique
recovers is just the spacecraft pointing error, not pointing error plus the systematic phase
bias. At zero phase the rendered model is uniform and the formula collapses to the geometric
centre.

Sub-solar direction
-------------------

The vector from the geometric centre to the lit-weighted centroid points along the projected
body-to-Sun direction, so the blob feature also carries its unit form as
``sub_solar_dir_vu``. :doc:`dev_guide_techniques_body_blob` orients its high-phase
crescent coarse-acquisition template along this direction (a filled disc cannot match a
crescent). Near full phase the two centroids coincide and the direction is meaningless, so it
is reported as ``(0, 0)`` and the technique falls back to a disc template. The derivation is
model-agnostic -- it reads only the rendered appearance -- so a SPICE-backed body model
populates the same field without extra plumbing.

Phase-and-irregularity coupling
-------------------------------

The blob feature carries a dimensionless coupling that combines the body's fractional shape
uncertainty with a phase factor:

.. math::

    \kappa = \frac{\sigma_{\mathrm{ellipsoid}}}{R_{\mathrm{body}}} \cdot
             \left( 1 + 2 \sin^{2}\!\left(\frac{\phi}{2}\right) \right)

where :math:`R_{\mathrm{body}}` is the predicted body radius (half the predicted disc diameter,
in km, derived from the local km/px scale) and :math:`\phi` is the phase angle. The phase
factor evaluates to 1 at full phase, 2 at 90 degrees, and 3 at full crescent — even at
zero phase the body's rotational orientation is unknown, so a residual-scale centroid bias is
always present. The blob feature's confidence formula consumes :math:`\kappa` directly so a
high-phase, irregular scene can be down-weighted without re-deriving the formula at the
technique layer.

Feature emission gates
----------------------

The model decides which of the four features to emit by computing two scalar quantities from
the silhouette extraction:

- The **limb uncertainty** :math:`\sigma_{\mathrm{ellipsoid}} / (\mathrm{km\ per\ px})`, which
  measures how far the limb fit can be expected to wander given the body's intrinsic
  ellipsoid-fit residual. When this scalar exceeds the documented cap the limb fit is
  information-limited; the gate rejects the
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` and the
  brightness-weighted-centroid path carries the body instead.
- The **visible-lit fraction** and **overflow fraction** above. The disc gate fires only
  when :attr:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` was emitted, the
  visible-lit fraction is at or above its minimum, and the
  overflow fraction is at or below its maximum.

The :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` gate fires whenever the
predicted diameter clears its minimum and the body has any lit signal, independent of the limb
gate: on a small or high-uncertainty body the blob is the *only* body feature, while on a
well-resolved body it is co-emitted **alongside** the limb (and disc) as a witness. The
witness blob is superseded in the fuse by any non-spurious geometric result, so it does not
dilute the geometric offset; the ensemble's body-witness veto reads it as an independent
pose-free cross-check against a geometric technique locked onto a mismatched shape (see
:doc:`dev_guide_orchestrator_ensemble`).

The terminator polyline gates fire when the surviving vertex count is at or above the
configured minimum and the phase factor :math:`\sin\phi` is at or above its minimum.

A body tagged ``highly_irregular`` in the shape table has no usable ellipsoid, so once it is
**resolved beyond a few pixels** its rendered limb / terminator / disc silhouette does not
match the real body. Such a body therefore emits none of
:data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC`,
:data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC`, or
:data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC`; only the point-like
:data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` survives, navigating the
body's centroid. The
"resolved" threshold reuses the ``min_bounding_box_area`` config value: its square root is the
equivalent linear pixel extent (default ``9`` px² gives a ``3`` px diameter cutoff), so a
highly-irregular body whose predicted diameter exceeds that value drops its shape features.
Below the threshold, and for bodies tagged merely ``irregular``, the shape features are
unchanged — the continuous ellipsoid-residual widening in the per-vertex sigma budget already
degrades an irregular body's limb reliability without dropping the feature outright.

.. _body-occlusion:

Body-body occlusion
-------------------

When two bodies overlap along the line of sight — a mutual event — the nearer body hides part
of the farther one, and the image shows the occluder there, not the target. The model accounts
for this so the features it emits describe the arcs and disc area the image actually carries.

Each per-body model receives the other in-FOV bodies as siblings. A sibling occludes the target
only when its inventory center range is strictly nearer; for each such sibling the model queries
the SPICE backplane's ``where_in_front`` predicate over the target's bounding box, which is True
at every pixel where the nearer body is intercepted in front of the target. The union of those
predicates is the target's occluder mask.

The occluder mask feeds three products:

- **Limb and terminator polylines.** A ridge vertex the occluder hides is dropped from the
  fitted polyline. It stays in the ridge total, so the ``visible_arc_fraction`` reports the loss:
  a mutual-event limb scores like the partial arc it is rather than claiming a full limb the
  distance-transform fit would then try to align against an arc the image does not show.
- **The visible-lit fraction** (:ref:`body-visible-lit-fraction`), which drops the disc
  reliability with occlusion depth.
- **The disc template.** The occluded pixels are trimmed out of the correlation template (both
  the brightness image and its mask), so the correlator scores only against disc brightness the
  image carries. Correlating against the missing disc area would be a coherent mismatch the
  robust machinery absorbs into the fit rather than discounting.

A backplane failure on any occluder degrades to leaving that occluder's contribution untrimmed
rather than aborting the render. The single-body case (no nearer sibling) computes no mask and
costs nothing.

Restrictions and assumptions
----------------------------

The model assumes:

- SPICE has predicted the body's pose and ephemeris well enough that the bounding box is
  approximately correct (the bounding box is inflated by a small slop fraction before clipping
  into the extfov to absorb half-pixel jitter from the inventory query).
- The body is approximately ellipsoidal at the level of the per-vertex sigma budget; bodies
  whose ellipsoid-fit residual exceeds the per-pixel scale are silently downgraded from limb
  arcs to blobs.
- The optical PSF is well approximated by a single Gaussian sigma, queried from the
  observation.
- The reported phase angle is in :math:`[0, 180]` degrees; values outside this range are
  clamped before being recorded on the blob feature flags.

Bodies whose predicted bounding box has zero overlap with the extended FOV produce no instance
and contribute no features. Bodies whose silhouette is entirely in shadow, or whose lit
fraction falls below the photometric thresholds, emit only the geometric features (limb arc
when its uncertainty allows; otherwise nothing).

Titan produces no
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` instance; the
:doc:`Titan model <dev_guide_navigation_models_titan>` handles it. A
``highly_irregular`` body resolved beyond the ``min_bounding_box_area``-derived
pixel threshold emits no shape features (limb / terminator / disc) and navigates only as a
:data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB`; ``irregular`` bodies are
unaffected.

Sources of uncertainty
----------------------

The per-vertex sigma values consumed by downstream techniques reflect the four physical
quantities enumerated above plus the photometric softness budget. They do not capture pose
errors that bias the entire silhouette uniformly (those manifest as a global translation that
the DT fit recovers); they do not account for unmodelled atmospheric haze or lens flare; and
they do not propagate the SPICE pointing uncertainty itself (that is handled separately by
the search-window margin). The predicted lit-weighted centroid attached to the blob feature
collapses to the geometric centre at zero phase but carries a phase-and-irregularity factor
on its flags that grows for high-phase, non-ellipsoidal bodies; the enclosing
:class:`~spindoctor.feature.feature.NavFeature` adds a corresponding photon-noise-limited centroid
sigma in quadrature to a shape-irregularity sigma so the blob covariance reflects both noise
and shape modelling error.

Configuration
=============

The model's runtime knobs are split across two YAML files: the rendering / extraction
parameters live under ``bodies`` in ``src/spindoctor/config_files/config_040_bodies.yaml`` (consumed
by the model itself plus its annotation helpers and the reprojection pipeline); per-body
shape, albedo, and SPK-residual values live under ``body_shape`` in
``src/spindoctor/config_files/config_220_body_shape.yaml`` (consumed via the
:func:`~spindoctor.nav_model.body_shape.load_body_shape` lookup chain). Module-level Python
constants in :mod:`spindoctor.nav_model.nav_model_body` set the emission-gate thresholds that are
not exposed to YAML.

bodies block
------------

Every key under ``bodies`` is listed below. Several keys are not consumed by
:class:`~spindoctor.nav_model.nav_model_body.NavModelBody` itself; the second column names the
module that does consume each key. The grep
``grep -c '^  [a-z_]\+:' src/spindoctor/config_files/config_040_bodies.yaml`` returns 30 keys, which
matches the 30 bullets here.

- ``min_bounding_box_area`` — int, default ``9`` px². Recorded on the model's metadata as
  ``size_ok``; sub-threshold bounding boxes are flagged for reviewer awareness. Does not by
  itself suppress feature emission. Its square root is also the "resolved" diameter cutoff
  above which a ``highly_irregular`` body drops its shape features. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``min_emission_ring_body`` — int, default ``20`` px. Reserved for the per-body ring-emission
  gate that decides whether to emit RING_EDGE features near a body. Reserved as an unused
  key so the per-instrument override story is uniform once a consumer is wired in.
- ``oversample_edge_limit`` — int, default ``512`` px. Cap on the per-axis oversample factor:
  the floor of ``oversample_edge_limit / max(1, ceil(bbox_extent_px))``. Larger values
  produce a smoother anti-aliased limb at the cost of a larger backplane query. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``oversample_maximum`` — int, default ``2`` (dimensionless). Hard cap on the per-axis
  oversample factor independent of bounding-box size. Saturates small-body renders so large
  field-fillers do not exhaust memory. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``curvature_threshold_frac`` — float, default ``0.02`` (dimensionless). Reserved for the
  curvature-classification heuristic that distinguishes "limb fits a circle" from "limb is
  effectively a straight line"; not consumed by the current
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``curvature_threshold_pixels`` — int, default ``20`` px. Reserved alongside
  ``curvature_threshold_frac`` for the same heuristic.
- ``limb_incidence_threshold`` — float, default ``1.53589...`` rad (= 88 degrees). Reserved
  for a limb-vertex shadow filter that drops vertices whose local incidence angle is past
  the threshold. Not consumed by the current model.
- ``limb_incidence_frac`` — float, default ``0.4`` (dimensionless). Reserved for the same
  shadow filter.
- ``surface_bumpiness`` — dict[str, float], per-body table (km). Reserved for a per-body
  surface-roughness estimate that would feed into the per-vertex sigma formula; not
  consumed (the equivalent quantity comes from
  :attr:`~spindoctor.nav_model.body_shape.BodyShape.crater_scale_km`).
- ``geometric_albedo`` — dict[str, float], per-body table (dimensionless). Per-body
  geometric albedo applied to the rendered brightness image when ``use_albedo`` is enabled.
  Entries default to bright icy moons (0.6-1.0); Phoebe (0.08), Iapetus (0.275), and Saturn
  (0.342) are the notable dark exceptions. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``use_lambert`` — bool, default ``true`` (dimensionless). When true the predicted
  brightness image is the per-pixel Lambert cosine of the incidence angle plus a small floor
  (so dark-but-visible silhouette pixels stay distinguishable from background); when false,
  the silhouette is rendered as a flat binary mask. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``use_albedo`` — bool, default ``false`` (dimensionless). When true and the body has an
  entry in ``geometric_albedo``, the rendered brightness is multiplied by that albedo.
  Distinguishes bright icy moons from dark bodies in multi-body composites. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`.
- ``min_reproj_seed_area`` — int, default ``40000`` px². Threshold on per-image body extent
  below which a body is not seeded into a body mosaic. Consumed by
  :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``min_reproj_candidate_area`` — int, default ``2500`` px². Threshold on per-image body
  extent below which a body is not added as a candidate to an existing body mosaic.
  Consumed by :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``reproj_lon_resolution`` — float, default ``0.01745...`` rad (= 1 degree). Longitude
  step for the body-mosaic reprojection grid. Consumed by
  :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``reproj_lat_resolution`` — float, default ``0.01745...`` rad (= 1 degree). Latitude
  step for the body-mosaic reprojection grid. Consumed by
  :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``reproj_latlon_type`` — str, default ``centric`` (one of ``centric`` / ``graphic``).
  Latitude / longitude convention used by the body-mosaic reprojection. Consumed by
  :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``reproj_lon_direction`` — str, default ``east`` (one of ``east`` / ``west``). Longitude
  positive direction for the body-mosaic reprojection. Consumed by
  :class:`~spindoctor.reproj.bodies.BodyMosaic`.
- ``min_text_area`` — float, default ``0.003`` (dimensionless fraction of frame area). Below
  this fraction the body is too small to label and the annotation pipeline skips its label.
  Consumed by :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_mask_enlarge`` — int, default ``10`` px. Pixels around a body to avoid for label
  placement. Consumed by :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_limb_color`` — list[int], default ``[255, 0, 0]`` (RGB). Color of the limb outline
  drawn on the summary PNG. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_font`` — str, default ``liberation2/LiberationMono-Bold.ttf``. Font used for body
  labels. Consumed by :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_font_size`` — int, default ``18`` px. Body label font size. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_font_color`` — list[int], default ``[255, 0, 0]`` (RGB). Body label font color.
  Consumed by :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_horiz_gap`` — int, default ``7`` px. Horizontal gap between the limb and the head
  of the label arrow. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_vert_gap`` — int, default ``5`` px. Vertical gap between the limb and the head of
  the label arrow. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_scan_v`` — int, default ``1`` px. Granularity in V when scanning a limb to find
  places to put labels. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_grid_v`` — int, default ``10`` px. Coarse V grid for label placement when no
  per-limb candidate is suitable. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``label_grid_u`` — int, default ``10`` px. Coarse U grid for label placement. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.
- ``outline_thicken`` — int, default ``0`` px. Number of dilation passes applied to the limb
  outline before drawing. Consumed by
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`.

Body-shape catalogue
--------------------

``src/spindoctor/config_files/config_220_body_shape.yaml`` carries the per-body
:class:`~spindoctor.nav_model.body_shape.BodyShape` overrides under a ``body_shape`` mapping keyed by
upper-case SPICE body name. Each entry may set any subset of the
:class:`~spindoctor.nav_model.body_shape.BodyShape` fields; null or missing fields fall through to
the hard-coded :data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE` baseline (Saturn-moon,
irregular-moon, gas-giant, or default profile depending on the body) per
:func:`~spindoctor.nav_model.body_shape.load_body_shape`.

Module-level emission constants
-------------------------------

The emission-gate thresholds are Python module-level constants in
:mod:`spindoctor.nav_model.nav_model_body` and are not exposed as YAML knobs. Tests and downstream
tools read the canonical values via these symbols.

- :data:`~spindoctor.nav_model.nav_model_body.BODY_POSITION_SLOP_FRAC` — float, ``0.05``
  (dimensionless). Inflation factor applied to the inventory bounding box before clipping
  into the extended FOV.
- :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX` — float, ``3.0`` px. Cap
  on the limb normal-sigma at which
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` remains useful. Above this
  value the per-vertex normal uncertainty is too large for the DT-based limb fit; the
  extractor switches to :attr:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB`.
- :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MIN_VERTICES` — int, ``30`` vertices.
  Minimum limb-polyline length to emit a
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC`; matches ``BodyLimbNav``'s
  ``min_arc_vertices`` feasibility floor. The uncertainty cap cannot stand in for this check:
  the per-vertex sigma scales the shape residual into pixels, so a distant small body passes
  it precisely because it is tiny. Shorter arcs fall through to the BODY_BLOB branch.
- :data:`~spindoctor.nav_model.nav_model_body.BODY_BLOB_MIN_DIAMETER_PX` — float, ``5.0`` px.
  Minimum predicted disc diameter at which
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` is emitted. Below this
  diameter the silhouette covers so few pixels that the brightness-weighted centroid is
  PSF- and noise-dominated; the per-body shape table can override this floor upward but not
  downward.
- :data:`~spindoctor.nav_model.nav_model_body.BODY_DISC_MIN_VISIBLE_LIT_FRACTION` — float, ``0.4``
  (dimensionless). Minimum lit-and-in-FOV fraction for
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` emission. Below 40 % the
  disc match is too asymmetric to be useful.
- :data:`~spindoctor.nav_model.nav_model_body.BODY_DISC_MAX_OVERFLOW_FRACTION` — float, ``0.3``
  (dimensionless). Maximum overflow fraction for
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` emission. A body whose disc
  is more than 30 % off-frame loses too much template support for a sharp correlation
  peak.
- :data:`~spindoctor.nav_model.nav_model_body.TERMINATOR_MIN_VERTICES` — int, ``8`` (count).
  Minimum surviving terminator vertex count for
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` emission.
- :data:`~spindoctor.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR` — float, ``0.05``
  (dimensionless). Minimum :math:`\sin\phi` for
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` emission. Below
  :math:`\sin\phi \approx 0.05` (phase below ~3 degrees) the terminator is too close to the
  limb to be photometrically distinguishable.

Per-instrument overrides
------------------------

The ``bodies`` block is global: per-instrument YAML (``config_4N0_inst_*.yaml``) does not
override any of the keys above. Instrument-specific behaviour enters through the
observation snapshot — the optical PSF sigma read from
:meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf` and the extended-FOV margin set by
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` — rather than through
this config block.

Implementation
==============

Source files:

- ``src/spindoctor/nav_model/nav_model_body.py`` —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`, the polyline-sampling helper, the
  per-feature constructors, and the module-level emission constants.
- ``src/spindoctor/nav_model/nav_model_body_base.py`` —
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`, the abstract shared base
  carrying the limb-mask helper and the body-label annotation pipeline.
- ``src/spindoctor/nav_model/body_shape.py`` —
  :class:`~spindoctor.nav_model.body_shape.BodyShape`,
  :data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE`,
  :data:`~spindoctor.nav_model.body_shape.DEFAULT_BODY_SHAPE`, and
  :func:`~spindoctor.nav_model.body_shape.load_body_shape`.

Public class :class:`~spindoctor.nav_model.nav_model_body.NavModelBody`, base
:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`. The class registers itself in
``NavModel._registry`` via ``__init_subclass__`` so that
:func:`~spindoctor.nav_model.nav_model.build_models_for_obs` discovers it. Public surface
(autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.instances_for_obs` — class method that
  returns one instance per body whose
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_extfov` predicate fires. The
  inventory is queried once per observation against the planet plus its configured
  satellites; bodies with no inventory entry, or whose entry fails the in-extfov predicate,
  contribute nothing.
- :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.create_model` — populates the model state
  by calling the silhouette renderer, the polyline samplers, and the geometry-summary logger.
- :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.to_features` — runs the four emission
  gates and constructs zero or more :class:`~spindoctor.feature.feature.NavFeature` instances.
- :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.to_annotations` — delegates to the shared
  annotation helper on the base class to render body silhouette and labels onto the summary
  PNG.
- :attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
  :attr:`~spindoctor.nav_model.nav_model.NavModel.obs`,
  :attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` — inherited read-only properties
  exposing the model's name (``body:<NAME>``), its source observation, and the per-image
  metadata dict.

BodyShape dataclass
-------------------

:class:`~spindoctor.nav_model.body_shape.BodyShape` is a frozen dataclass whose fields drive the
covariance and emission gates:

- :attr:`~spindoctor.nav_model.body_shape.BodyShape.ellipsoid_rms_residual_km` — RMS deviation of
  the body silhouette from the best-fit ellipsoid (km). Primary contribution to
  :math:`\sigma_{\mathrm{ellipsoid}}` in the per-vertex sigma quadrature sum.
- :attr:`~spindoctor.nav_model.body_shape.BodyShape.crater_scale_km` — characteristic crater /
  topographic scale (km).
- :attr:`~spindoctor.nav_model.body_shape.BodyShape.albedo_variation` — fractional disc-brightness
  variation in :math:`[0, 1]`. Drives the terminator-arc reliability formula.
- :attr:`~spindoctor.nav_model.body_shape.BodyShape.spice_orbital_residual_km` — SPK ephemeris
  uncertainty in km.
- :attr:`~spindoctor.nav_model.body_shape.BodyShape.min_blob_diameter_px` — predicted disc
  diameter (px) at which the extractor stops emitting
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` and switches to
  :attr:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB`.
- :attr:`~spindoctor.nav_model.body_shape.BodyShape.shape_class_hint` — coarse classification
  (``regular`` / ``irregular`` / ``highly_irregular`` / ``unknown``); used in human-readable
  logs and reviewer-facing diagnostics.

The public entry point is :func:`~spindoctor.nav_model.body_shape.load_body_shape`,
which resolves the operator-curated YAML entry (via the private ``_yaml_entry_for``
helper), overlays its non-null fields onto the hard-coded baseline, and returns the
merged :class:`~spindoctor.nav_model.body_shape.BodyShape`. Priority order:

1. Operator-curated YAML (``config_220_body_shape.yaml``) — each non-null field overrides
   the baseline.
2. Hard-coded :data:`~spindoctor.nav_model.body_shape.BODY_SHAPE_TABLE` profile for the body
   (Saturn-moon, irregular-moon, or gas-giant).
3. :data:`~spindoctor.nav_model.body_shape.DEFAULT_BODY_SHAPE` for entirely unknown bodies.

Annotation helpers
------------------

:class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase` is the abstract shared base.
Two helpers live there:

- ``_compute_limb_mask_from_body_mask`` — computes the limb mask from the body mask via
  discrete neighbour shifts. Used by the simulated body model.
- ``_create_annotations`` — builds the body-label
  :class:`~spindoctor.annotation.annotations.Annotations` collection for the summary PNG. Consumes
  the ``label_*``, ``min_text_area``, and ``outline_thicken`` keys documented above.

Per-image metadata
------------------

:meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.create_model` populates
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata` with the following entries for the curator
to surface in the per-image JSON sidecar:

- ``start_time`` / ``end_time`` / ``elapsed_time_sec`` — wall-clock timing for the model
  build.
- ``body_name`` — upper-case SPICE body name.
- ``sub_solar_lon_deg`` / ``sub_solar_lat_deg`` — sub-solar coordinates of the body at
  midtime.
- ``sub_observer_lon_deg`` / ``sub_observer_lat_deg`` — sub-observer coordinates.
- ``phase_angle_deg`` — centre-pixel phase angle.
- ``bbox_area_px`` / ``size_ok`` — predicted bounding-box area and the
  ``min_bounding_box_area`` test result.
- ``guaranteed_visible_in_fov`` — bool; True iff the inflated bounding box lies fully inside
  the sensor minus the per-instrument extfov margin.
- ``km_per_pixel_at_limb`` — mean km/px scale across the limb polyline.
- ``predicted_diameter_px`` — predicted body diameter in pixels.
- ``visible_lit_fraction`` / ``overflow_fraction`` — fractions defined in Theory above.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.create_model` and
:meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.to_features`:

1. :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.create_model` opens a logged section,
   clears ``self._metadata``, records ``start_time``, and invokes the private render helper.
2. The render helper looks up the inventory entry, queries five sub-solar / sub-observer /
   phase-angle backplanes for the geometry summary, and clips an inflated bounding box into
   the extended FOV. The inflation factor is
   :data:`~spindoctor.nav_model.nav_model_body.BODY_POSITION_SLOP_FRAC` of the inventory extent.
3. The render helper builds an oversampled meshgrid + backplane around the clipped bounding
   box, queries the incidence-angle backplane, downsamples the mask to extfov resolution,
   and derives the limb / terminator / body / lit masks via discrete neighbour shifts.
4. A predicted brightness image is built from the Lambert cosine when ``use_lambert`` is true
   and the body is at least partly lit; otherwise the silhouette is rendered as a binary
   mask with a small offset so empty body pixels remain non-zero. When ``use_albedo`` is
   true and the body has a ``geometric_albedo`` entry, the brightness image is multiplied by
   the albedo.
5. Discrete masks are walked by the private polyline-sampler helper to produce per-vertex
   polylines (vertex position, outward normal, incidence, km-per-pixel). The lit-and-in-FOV
   pixels are counted to compute ``visible_lit_fraction`` and ``overflow_fraction``.
6. :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.to_features` resolves the per-body
   :class:`~spindoctor.nav_model.body_shape.BodyShape` via
   :func:`~spindoctor.nav_model.body_shape.load_body_shape` and computes the limb-uncertainty
   scalar. Three branches follow, in order:

   - When the limb polyline carries at least
     :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MIN_VERTICES` vertices and its
     uncertainty is at or below
     :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX`, a
     :data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` feature is emitted. The
     downstream disc gate then has a chance to fire alongside the limb arc when
     ``visible_lit_fraction`` and ``overflow_fraction`` allow.
   - The blob gate is met when the predicted disc diameter is at least
     ``max(`` :data:`~spindoctor.nav_model.nav_model_body.BODY_BLOB_MIN_DIAMETER_PX` ``,``
     :attr:`~spindoctor.nav_model.body_shape.BodyShape.min_blob_diameter_px` ``)`` and the
     rendered silhouette contains at least one lit pixel. When the limb arc was rejected but
     the blob gate is met, a
     :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` feature is emitted as
     the sole body feature. When the limb arc was emitted and the blob gate is also met, a
     BODY_BLOB is co-emitted alongside the limb (and disc) as a witness. Either way the blob
     feature carries the lit-weighted predicted centroid and the phase-and-irregularity factor
     :math:`\kappa` on its :class:`~spindoctor.feature.flags.BodyBlobFlags`; the witness blob
     is superseded in the fuse by any non-spurious geometric result and is read only by the
     ensemble body-witness veto.
   - Otherwise no body feature is emitted (the body is too small to fit and too unresolved
     to centroid, or its silhouette is entirely in shadow and there is no photometric
     signal to centroid).

7. Independent of the limb / blob branch, a
   :data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` feature is emitted
   whenever the terminator polyline meets
   :data:`~spindoctor.nav_model.nav_model_body.TERMINATOR_MIN_VERTICES` and
   :math:`\sin\phi \ge` :data:`~spindoctor.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR`.

The per-feature constructors live in module-level helpers that use the
:class:`~spindoctor.nav_model.body_shape.BodyShape` parameters and the optical-PSF sigma to
populate the per-vertex sigma arrays per the formula in the Theory section.

Examples
========

The named scenes in ``tests/integration/image_library/images/`` exercise the four feature
types. Numbers below are taken from the operator-curated YAML sidecars and the integration
tests.

``body_full_fov`` (Cassini ISS NAC, image ``N1572105349_1``)
    Dione fills the FOV — predicted disc diameter approximately 155 px, mostly lit with a
    sliver of terminator, ``overflow_fraction = 0.0``, ``visible_lit_fraction`` approximately
    ``0.97``. The model emits a single
    :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` feature carrying the rendered
    template; the :data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` is rejected by
    the downstream reliability gate that consumes it
    (the textbook full-disc, fully-lit limb saturates the model-side reliability formula's
    incidence-factor penalty). Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (8.68, -17.37)` px.

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right with ``overflow_fraction \approx 0.22``. The model emits
    a :data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` feature plus a
    :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_DISC` feature (the overflow fraction
    sits below
    :data:`~spindoctor.nav_model.nav_model_body.BODY_DISC_MAX_OVERFLOW_FRACTION`) and a
    :data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` feature.
    Operator-verified offset is :math:`(\Delta v, \Delta u) = (11.0, 29.5)` px.

``below_resolution_body`` (Cassini ISS NAC, image ``N1777325846_1``)
    Mimas is approximately 20 px in diameter in the lower left, at phase angle 72 degrees.
    The predicted disc diameter is well above
    :data:`~spindoctor.nav_model.nav_model_body.BODY_BLOB_MIN_DIAMETER_PX` but the per-pixel
    ellipsoid uncertainty exceeds
    :data:`~spindoctor.nav_model.nav_model_body.LIMB_ARC_MAX_UNCERTAINTY_PX`, so the model emits a
    :data:`~spindoctor.feature.feature_type.NavFeatureType.BODY_BLOB` feature instead of a LIMB_ARC.
    The blob feature carries the lit-weighted centroid and a phase-irregularity factor
    populated from the per-body
    :class:`~spindoctor.nav_model.body_shape.BodyShape`. Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (6.08, -1.53)` px.

``high_phase_terminator`` (Cassini ISS NAC, image ``N1597846115_2``)
    A high-phase terminator arc with no other features in the FOV. The model emits a
    :data:`~spindoctor.feature.feature_type.NavFeatureType.LIMB_ARC` feature (the lit limb survives)
    and a :data:`~spindoctor.feature.feature_type.NavFeatureType.TERMINATOR_ARC` feature (the
    terminator polyline meets the minimum vertex count and the phase factor sits above the
    minimum). Operator-verified offset is :math:`(\Delta v, \Delta u) = (5.19, 1.30)` px.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees.
    The orchestrator instantiates two
    :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` instances, one per body, each
    emitting its own combination of features per the gates above; the downstream techniques
    receive the union and fuse a single offset. Operator-verified offset is
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px.
