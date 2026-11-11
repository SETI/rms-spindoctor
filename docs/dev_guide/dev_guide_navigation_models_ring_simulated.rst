==========================================================
Simulated Ring Navigation Model
==========================================================

Overview
========

:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated` predicts the
rings of a simulated frame. The SPICE-backed
:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` predicts a real frame's rings
from a ring catalog and the spacecraft's pointing. A simulated frame has neither, so this
model predicts from the scene's own ``ring_system`` block instead, as the information
boundary exposes it in ``obs.nav_params['ring_system']`` (see :doc:`dev_guide_simulator`).

Each feature the scene marks navigable becomes one model instance, and each instance emits
two kinds of navigation feature:

- A ringlet emits a :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS`
  carrying its predicted coverage template, for
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`.
- Every feature emits one :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE`
  per predicted catalog edge, as a polyline of border pixels with outward radial normals,
  for :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.

The navigator sees only the navigable features. The boundary filter drops every other
feature from ``nav_params``, so the rendered frame holds ring structure the navigator was
never told about. That clutter is deliberate: it is the distractor and false-lock regime the
ring system exists to test.

Theory
======

The prediction is purely geometric. For each pixel of the extended field of view, the model
runs the sky-to-ring-plane projection backward to find the ring-plane radius :math:`r` and
the longitude :math:`\lambda` measured from the ascending node. It then evaluates the
feature's catalog orbit at that longitude, giving the edge radius
:math:`r_{\mathrm{edge}}(\lambda)`. The orbit is a precessing ellipse plus any
:math:`m \ge 2` normal modes and a satellite edge wave. To first order in eccentricity the
ellipse is

.. math::

    r_{\mathrm{edge}}(\lambda) \approx a - ae \cos(\lambda - \varpi),

where :math:`a` is the semimajor axis, :math:`ae` the radial amplitude, and :math:`\varpi`
the pericenter longitude, precessed from the ring epoch to the scene time. The code
evaluates the exact conic.

The difference :math:`d = r - r_{\mathrm{edge}}(\lambda)` is a signed radial distance field
over the grid, and everything the model emits comes from it:

- **Edges.** An edge is the one-pixel border where :math:`d` changes sign. Each border pixel
  becomes a polyline vertex, and its normal is the normalized gradient of :math:`d` there.
  That gradient points toward increasing ring radius, so the normal points radially outward
  however the projection foreshortens the ring.
- **Coverage.** A ringlet or a gap has an inner edge at the orbit and an outer edge at the
  orbit widened by ``width_px``. Between the two the model rasterizes an anti-aliased
  coverage template in [0, 1]. A ringlet's template is solid: the navigator treats the band
  as opaque and ignores its optical depth, which is photometric truth. A gap shows the
  background instead of emitting light. It doesn't carry a template, but its two edges are
  still fitted.

The projection and the orbit evaluation come from :mod:`spindoctor.sim.ring_geometry`, the
same code the renderer uses to draw the ring. So a predicted edge lands exactly where the
rendered edge would land if the scene planted no error. In a recovery measurement, the only
differences between prediction and image are the planted pointing offset and each
feature's planted ``orbit_error``. The independence guarantee concerns what the navigator
knows, not which code it runs: the navigator never sees ``orbit_error`` or the photometric
truth.

Restrictions and assumptions
----------------------------

- The prediction reads only idealized keys: the shared ``geometry`` block, the catalog
  ``orbit``, the kind's shape keys, and ``declared_orbit_sigma``. ``tau`` also crosses the
  boundary, but the prediction ignores it. The filter strips the planted ``orbit_error`` and
  the photometric truth (``albedo`` and ``phase_g``) before this model runs.
- A ramp fades linearly into the background at one end, which leaves nothing sharp to fit.
  Only its sharp end emits an edge: the outer edge when ``side`` is ``out``, the inner edge
  otherwise.
- A density wave train emits one edge at its launch radius, the orbit radius. A one-sided
  ``edge`` emits its single boundary.
- An exactly edge-on geometry (``opening_deg_obs`` of 0) predicts nothing, because the
  renderer draws nothing there either.

Sources of uncertainty
----------------------

The rendered edge is sharp and noiseless, so the only measurement uncertainty is the
one-pixel sampling of the polyline. The scene can also declare how well its catalog knows
each orbit, with ``declared_orbit_sigma``: :math:`\sigma_a` (``sigma_a_px``),
:math:`\sigma_{ae}` (``sigma_ae_px``), and :math:`\sigma_\varpi` (``sigma_long_peri_deg``,
converted to radians). A key the scene omits or sets to null counts as zero. These are the
catalog's error bars, which a navigator is entitled to know. The values the scene drew from
them stay behind the boundary.

Both uses below start from how far each orbit element moves each vertex. A change
:math:`\delta r_{\mathrm{edge}}` in the edge radius moves the zero level of :math:`d` by
:math:`\delta r_{\mathrm{edge}} / \lVert \nabla d \rVert` along the vertex normal, so the
image-plane shift of vertex :math:`i` per unit change of element :math:`q` is

.. math::

    p_{q,i} = \frac{\partial r_{\mathrm{edge}} / \partial q}{\lVert \nabla d \rVert}
    \bigg|_i, \qquad q \in \{a,\ ae,\ \varpi\}.

The model takes each derivative by a central difference over the full orbit model, so it
follows the exact conic, and divides by the same gradient that gives the normal. On a
face-on ring with a small eccentricity, :math:`p \approx (1,\ -\cos(\lambda - \varpi),\
-ae \sin(\lambda - \varpi))`. On an inclined ring the gradient is steeper where the
projection foreshortens the ring, so the same orbit error moves the edge fewer image pixels
there.

**Per-vertex radial sigma.** Each vertex gets

.. math::

    \sigma_{r,i} = \max\!\left(1\ \mathrm{px},\;
        |p_{a,i}|\,\sigma_a + |p_{ae,i}|\,\sigma_{ae} + |p_{\varpi,i}|\,\sigma_\varpi\right).

A pericenter error moves an eccentric edge most a quarter orbit from the apsides, so it
widens the sigma there and leaves the apsides and a circular edge alone. A radial-amplitude
error does the opposite: it moves the apsides most. The along-edge sigma is a fixed 0.5 px,
the sampling resolution. These sigmas weight the vertices in the technique's robust fit.

**Coherent orbit sigma.** A catalog-orbit error moves every vertex of an edge together. So
it doesn't average down as the vertex count grows, the way the per-vertex sigma does. The
model therefore also carries a single value, without the 1 px floor, on the emitted
geometry's ``sigma_orbit_radial_px``. The three errors are independent, so a vertex's
1-sigma shift is their quadrature sum, and the value is its largest over every predicted
vertex of the feature:

.. math::

    \sigma_{\mathrm{orbit}} = \max_i \sqrt{(p_{a,i}\,\sigma_a)^2 + (p_{ae,i}\,\sigma_{ae})^2
        + (p_{\varpi,i}\,\sigma_\varpi)^2}.

On a face-on ring with a small eccentricity this is close to
:math:`\sqrt{\sigma_a^2 + \max(\sigma_{ae},\ ae\,\sigma_\varpi)^2}`. At larger
eccentricity the exact conic moves the edge more than :math:`ae\,\sigma_\varpi`, about 27%
more at :math:`e = 0.5`.

Both ring techniques add this term to their reported covariance along the radial
direction, which keeps a misplaced orbit from passing as a confident fix. The per-vertex
sum is linear and the coherent value is in quadrature. They differ on purpose: the first
only weights vertices against one another, while the second is squared into the covariance
that sets the confidence tier, where a linear sum would overstate it by up to
:math:`\sqrt{2}`. See :doc:`dev_guide_techniques_ring_edge` and
:doc:`dev_guide_techniques_ring_annulus` for how each technique uses it.

Configuration
=============

The model doesn't have YAML configuration of its own. Everything it reads comes from the
observation's filtered scene view, ``obs.nav_params``:

- ``ring_system.geometry``: the shared projection block. The model reads ``center_v`` and
  ``center_u`` (the projected ring center, in pixel-corner coordinates),
  ``opening_deg_obs`` (the observer's ring opening angle), and ``node_deg`` (the sky
  position angle of the ascending node). ``opening_deg_sun`` lights the rendered ring and
  doesn't affect the prediction.
- ``ring_system.range_km``: the ring system's distance, which becomes each emitted
  feature's subject range. When it is absent the range is infinite.
- ``ring_system.features``: the navigable features only. Each carries ``name``, ``kind``
  (``ringlet``, ``gap``, ``edge``, ``ramp``, or ``wave``), the kind's shape keys
  (``width_px``, ``side``, ``wavelength_px``, and ``density_wave_damping_px``), the catalog
  ``orbit`` (``a_px``, ``ae_px``, ``long_peri_deg``, ``rate_peri_deg_per_day``, ``modes``,
  and ``edge_wave``), and an optional ``declared_orbit_sigma``.
- ``time`` and ``ring_epoch``: the scene time and the ring epoch, which set how far the
  pericenter has precessed. Both default to 0, the same default the renderer uses.

Every length is in ring-plane pixels and every angle in degrees (see
:ref:`sim-ring-system`). The block's ``km_per_pixel`` and ``phase_deg`` also cross the
boundary, but this model doesn't read them.

Implementation
==============

Source files: ``src/spindoctor/nav_model/nav_model_rings_simulated.py`` holds
:class:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated`, and
``src/spindoctor/nav_model/sim_ring.py`` holds the prediction,
:func:`~spindoctor.nav_model.sim_ring.predict_ring_feature`.

The class derives from :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.
It overrides :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` to return one
instance per navigable feature of a simulated observation, named ``rings_sim:<feature
name>``, and an empty list for a real one. The parent
:class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` does the reverse, so the
orchestrator's :func:`~spindoctor.nav_model.nav_model.build_models_for_obs` driver gets
simulated frames from this class and real frames from the parent.

Public methods (autodocumented at :doc:`/api_reference/api_nav_model`):

- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model`
  predicts the feature's template and edges on the extended-FOV grid.
- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features`
  emits the RING_ANNULUS for a ringlet and one RING_EDGE per predicted edge.
- :meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_annotations`
  draws each edge and its label (``SATURN INNER EDGE``, or ``NAME EDGE`` for a
  single-boundary kind) onto the summary PNG through the annotation helper on
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`.

Inherited :class:`~spindoctor.nav_model.nav_model.NavModel` properties:
:attr:`~spindoctor.nav_model.nav_model.NavModel.name`,
:attr:`~spindoctor.nav_model.nav_model.NavModel.obs`, and
:attr:`~spindoctor.nav_model.nav_model.NavModel.metadata`.

Call path
---------

:meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.create_model`:

1. Opens a logged section and reads the feature's mapping and the shared ``geometry``
   block. It shifts the projected center by the extended-FOV margins.
2. Calls :func:`~spindoctor.nav_model.sim_ring.predict_ring_feature`. That function builds
   the pixel grid in pixel-corner coordinates, projects it to ring-plane
   :math:`(r, \lambda)` with :func:`~spindoctor.sim.ring_geometry.ring_plane_from_sky`, and
   evaluates the orbit with :func:`~spindoctor.sim.ring_geometry.compute_orbit_radii`. It
   then rasterizes the coverage template and one border mask per edge, with each vertex's
   outward normal and its shifts :math:`p_{a}`, :math:`p_{ae}`, and :math:`p_{\varpi}`.
3. Records the prediction, the predicted center, and the extended-FOV bounding box for
   ``to_features``. The grid is in pixel-corner coordinates and the feature payload is
   pixel-centered, so the half pixel comes off the center here.

:meth:`~spindoctor.nav_model.nav_model_rings_simulated.NavModelRingsSimulated.to_features`:

1. Computes :math:`\sigma_{\mathrm{orbit}}` once over every predicted vertex of the
   feature.
2. For a ringlet with coverage, emits a
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_ANNULUS`
   :class:`~spindoctor.feature.feature.NavFeature` with id ``ring_annulus:<NAME>``. It
   carries the coverage template cropped to its tight bounding box, the predicted center,
   and a :class:`~spindoctor.feature.flags.RingAnnulusFlags` with the feature's name. Its
   :class:`~spindoctor.feature.geometry.RingAnnulusGeometry` also carries the outward
   normals of both edges, concatenated into
   :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.orbit_normals_vu`, and
   :math:`\sigma_{\mathrm{orbit}}` in
   :attr:`~spindoctor.feature.geometry.RingAnnulusGeometry.sigma_orbit_radial_px`. That
   lets :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` add a
   misplaced orbit to its covariance the same way the edge technique does.
3. For each predicted edge with vertices, emits a
   :data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` with id
   ``ring_edge:<NAME>:<inner|outer|edge>``, carrying a
   :class:`~spindoctor.feature.geometry.RingEdgePolyline`: the vertices, their normals, the
   per-vertex radial and along-edge sigmas, and :math:`\sigma_{\mathrm{orbit}}`. The edge
   is marked straight when every vertex lies within 1 px of the best-fit line through
   them. A straight edge constrains only the offset along its normal, while a curved arc
   constrains both axes.
4. Sets every feature's reliability to ``1.0``.

Examples
========

``algorithmic_invariants/planted_offset_ring.yaml`` renders a face-on (``opening_deg_obs``
of 90) navigable ringlet centered in a 220 px frame, with radii from 60 to 85 px and
``tau`` of 2, and plants a pointing offset of (1.43, −0.61) px. The model predicts both
edges at their catalog position. RingEdgeNav fits the two curved polylines against the
image's edge distance transform, and ``test_sim_algorithmic_invariants.py`` requires its
offset to land within 0.066 px of the planted one, the largest error over 16 noise seeds
plus a margin. The scene's baseline pins the
fused ensemble result at (1.42, −0.62) px.

``ring_system/orbit_error_ringlet.yaml`` is the same kind of ringlet, but it plants
``orbit_error: {delta_a_px: 2.5}`` and declares ``declared_orbit_sigma: {sigma_a_px: 2.5}``.
The image shows the band 2.5 px outside the prediction. A uniform radial error dilates a
closed ring rather than translating it, yet the robust fit still locks onto one side and
absorbs part of the error into the offset. The declared sigma reaches both techniques'
covariance, so the scene expects a success at the low confidence tier with an offset error
between 1.490 and 3.457 px. Which arc the fit locks onto depends on the noise draw, and the
band covers the error measured over 16 noise seeds plus a margin. The fused sigma of about
2.5 px covers how far the fix can be from the truth.
