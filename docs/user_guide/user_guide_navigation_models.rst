===============================
Selecting models and techniques
===============================

``sd_offset`` runs every applicable navigation model and every feasible
navigation technique by default.  Two glob-pattern filters narrow that
set: ``--nav-models`` selects which :class:`~spindoctor.nav_model.nav_model.NavModel`
instances run, ``--nav-techniques`` selects which
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclasses run.
The same syntax applies in three places:

* ``sd_offset --nav-models LIST --nav-techniques LIST`` on the CLI.
* ``sd_offset_cloud_tasks`` task JSON, under
  ``data.arguments.nav_models`` and ``data.arguments.nav_techniques``
  (each a list of strings).
* :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` programmatic
  use, via the ``only_models=`` and ``only_techniques=`` keyword arguments.

The two filters share their pattern syntax; only the *names* they match
differ.  Filtering is purely additive over the existing registry — it does
not register new models or techniques, so an entry that does not exist on
this build of ``rms-spindoctor`` simply does not match.

Pattern syntax
--------------

Patterns are gitignore-style fnmatch globs evaluated against the
candidate name.  A single string or a comma-separated list (CLI) /
list-of-strings (JSON, Python) is accepted; the orchestrator splits on
commas and trims whitespace.

Inclusion patterns
^^^^^^^^^^^^^^^^^^

* A literal name matches that name only:
  ``BodyLimbNav`` matches the technique class
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` and
  nothing else.
* ``*`` matches any sequence of characters; ``?`` matches a single
  character; ``[abc]`` matches any character from the set.  Standard
  Python ``fnmatch`` semantics apply.
* The default ``'*'`` matches every candidate.

Exclusion patterns
^^^^^^^^^^^^^^^^^^

* A leading ``!`` marks an *exclusion* pattern: matches against the
  remaining glob are removed from the result.
  ``--nav-techniques '!StarFieldFromCatalogNav'`` runs every registered
  technique except that one.
* When every pattern in the list begins with ``!`` (a pure-exclusion
  list), an implicit ``'*'`` inclusion is added so the result is
  "everything except the excluded names".  ``--nav-models '!body:MIMAS'``
  is therefore equivalent to ``--nav-models '*,!body:MIMAS'``.
* When at least one inclusion pattern is present, only the listed
  inclusions plus their non-excluded matches survive.
  ``'body:*,!body:MIMAS'`` runs every body model except Mimas.

Multiple patterns
^^^^^^^^^^^^^^^^^

* On the CLI, comma-separate patterns inside a single argument:
  ``--nav-models 'body:MIMAS,rings:SATURN,stars'``.
* In JSON or Python, supply a list of strings:
  ``["body:MIMAS", "rings:SATURN", "stars"]``.
* The list is order-independent: a candidate name is kept iff it
  matches at least one inclusion pattern and no exclusion pattern.

Model names
-----------

The catalog-driven models register under these per-instance names:

* ``stars`` — :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`
  (one instance per observation; no namespace).
* ``body:NAME`` —
  :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` (one instance per
  body whose bounding box overlaps the extended FOV).  The ``NAME``
  portion is the upper-case SPICE body name
  (``body:MIMAS``, ``body:DIONE``, ``body:SATURN``).
* ``rings:PLANET`` —
  :class:`~spindoctor.nav_model.nav_model_rings.NavModelRings` (at most one
  instance, for the planet returned by ``obs.closest_planet``, and only
  when that planet has an entry in the ``rings.ring_features`` catalog;
  only Saturn's catalog is populated, so ``rings:SATURN`` is the only
  instance in practice).
* ``titan:TITAN`` —
  :class:`~spindoctor.nav_model.nav_model_titan.NavModelTitan` (one
  instance whenever Titan is inside the extended FOV).  Titan's opaque
  haze hides the surface, so instead of shape features the model emits the
  haze-envelope geometry that
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`
  navigates from.  On a simulated image the equivalent model registers as
  ``titan_sim:TITAN``.

Two convenience normalizations apply to model patterns:

* The ``VALUE`` part of ``prefix:VALUE`` is upper-cased automatically,
  so ``body:saturn`` matches ``body:SATURN``.
* A bare prefix without a colon and without glob characters
  (``body``, ``rings``, ``titan``) is auto-expanded to ``prefix:*``, matching
  every namespaced model under that prefix.  ``stars`` (which has no
  namespace) continues to match itself directly.

Both normalizations preserve the leading ``!`` exclusion marker.
``--nav-models 'body'`` is therefore shorthand for "every body model";
``--nav-models '!body'`` excludes every body model.

Technique names
---------------

Techniques register under their class name.  The shipping concrete
techniques are:

* Body family —
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
  :class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`.
* Ring family —
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`,
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.
* Star family —
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`.
* Titan family —
  :class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`.

The star field matcher re-centroids each matched star with a point-spread-function fit
when the star is faint, and keeps the simpler brightness-weighted centroid when the star
is bright enough that its noise has already fallen below the PSF fit's residual bias.
This makes the star field the most accurate technique on a well-exposed field. The
brightness at which it switches is the configurable
``techniques.StarFieldFromCatalogNav.tuning.psf_refine_snr_max`` knob in
``config_510_techniques.yaml`` (set the whole step off with ``psf_refine_enabled: 0``).

:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` is
the interactive driver and is not part of the autonomous registry; it
cannot be invoked by ``--nav-techniques``.

Multiple feasible techniques run in parallel and the orchestrator
combines their results via the ensemble step; ``--nav-techniques`` is
not a "pick one technique" knob — it restricts the candidate set the
orchestrator considers.

Examples
--------

.. code-block:: bash

   # Run every model and every technique (the default).
   sd_offset coiss N1234567890

   # Mimas only — drop every other body and the ring/star models.
   sd_offset coiss N1234567890 --nav-models 'body:MIMAS'

   # Every body, plus rings, but no stars.
   sd_offset coiss N1234567890 --nav-models 'body:*,rings'

   # Every model except Mimas (auto-expanded ``'*'`` inclusion).
   sd_offset coiss N1234567890 --nav-models '!body:MIMAS'

   # Two specific DT-based techniques only.
   sd_offset nhlorri LOR_0034851733 \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

   # Every technique except the catalog star matcher.
   sd_offset coiss N1234567890 \
       --nav-techniques '!StarFieldFromCatalogNav'

   # Body and ring families only (every body / ring technique, no stars).
   sd_offset coiss N1234567890 \
       --nav-techniques 'Body*,Ring*'


Navigation Models
=================

A *navigation model* is SpinDoctor's prediction of what the image *should*
look like at the spacecraft's nominal pointing.  Four model families
ship out of the box: stars, planetary bodies, planetary rings, and
Titan's haze envelope.
Each contributes one or more *features* (typed predictions with their
own per-feature uncertainty) to the navigator.  You can restrict which
families run by passing ``--nav-models`` on the command line; valid
entries are ``stars``, ``rings``, ``titan`` (equivalently
``titan:TITAN``), and body-specific entries of the form ``body:NAME``
(glob patterns are allowed).

Star Navigation Model
---------------------

The star model builds a deduplicated catalog of stars expected to fall
inside the field of view, applies stellar aberration and proper motion
to bring each catalog position into the spacecraft frame at observation
time, and emits one feature per usable star.

**Catalog precedence.**  Catalogs are searched in the order configured
in ``config_030_stars.yaml`` under ``stars.catalogs`` (default
``[ucac4, tycho2, ybsc]``).  Stars present in more than one catalog are
deduplicated using the RA / DEC and V-magnitude thresholds in the same
file.

**Per-star detectability.**  Each star is gated by its catalog visual
magnitude against the per-observation limiting magnitude
``obs.star_max_usable_vmag()``, which depends on the per-instrument
sensitivity and the exposure time.  Stars fainter than the limiting
magnitude (or with no catalog magnitude) are dropped.

**Smear.**  When the spacecraft attitude rate is non-zero during the
exposure, stars smear into trails.  The model computes the per-image
smear vector from the SPICE pointing brackets and uses
``psfmodel.eval_rect(movement=...)`` to render a smear-aware kernel
when a downstream technique needs one.  Stars whose smear length
exceeds ``stars.max_smear`` are dropped (the centroid is unfittable).

**Body and ring conflicts.**  Each star's predicted pixel is checked
against an ``oops`` body intercept and a per-planet opaque ring
annulus (configured under ``stars.ring_occlusion_radii_km``).  Stars
that fall behind a body or inside an opaque ring annulus are tagged
with a ``BODY:`` or ``RING:`` conflict string and excluded from
matching.  Body intercepts win over ring intercepts.

**Configuration.**  Most user-tunable parameters live in
``config_030_stars.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``stars.catalogs``
     - Catalog search order; default ``[ucac4, tycho2, ybsc]``.
   * - ``stars.max_stars``
     - Maximum number of stars retained per image (default 100).
   * - ``stars.max_smear``
     - Smear length in pixels above which a star is dropped (default
       100).
   * - ``stars.min_vmag`` / ``stars.max_vmag``
     - Magnitude window applied to the per-instrument
       ``star_min_usable_vmag`` / ``star_max_usable_vmag`` floor.
   * - ``stars.proper_motion``
     - Apply proper motion at ``obs.midtime`` (default true).
   * - ``stars.stellar_aberration``
     - Apply stellar aberration (default true).
   * - ``stars.ring_occlusion_enabled``
     - Toggle the ring-annulus occlusion check (default true).
   * - ``stars.ring_occlusion_radii_km``
     - Per-planet list of opaque ``[inner_km, outer_km]`` annuli.

Body Navigation Model
---------------------

For every body whose predicted bounding box overlaps the extended
field of view, the body model renders an oversampled Lambert-shaded
silhouette, extracts the limb and terminator polylines, and emits a
mix of feature types depending on resolution, lighting, and shape
quality:

- ``LIMB_ARC`` — emitted when the limb position is well-determined
  (per-vertex normal sigma below the ``LIMB_ARC_MAX_UNCERTAINTY_PX``
  cap).  Carries a polyline of vertex coordinates and per-vertex
  anisotropic sigmas.
- ``BODY_BLOB`` — emitted instead of ``LIMB_ARC`` when the limb is too
  uncertain to fit but the predicted body diameter is above the
  body-specific blob threshold.  Carries only a centroid and bounding
  box.
- ``BODY_DISC`` — emitted alongside ``LIMB_ARC`` when the body fits
  well inside the FOV (overflow below 30 %, lit-and-visible fraction
  at least 40 %).  Carries the rendered template for full-disc
  correlation.
- ``TERMINATOR_ARC`` — emitted when the terminator polyline has at
  least 8 vertices and the phase-angle factor (``sin(phase_angle)``)
  is above 0.05.

**Per-body shape data.**  ``ellipsoid_residual_km``, ``crater_scale_km``,
``albedo_variation``, ``spice_orbital_residual_km``, and
``min_blob_diameter_px`` come from the static body-shape table.  These
quantities drive the per-vertex polyline sigmas and the BODY_BLOB
emission threshold.  For bodies absent from the table a conservative
generic-icy-moon profile is used.

**Configuration.**  ``config_040_bodies.yaml`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``bodies.min_bounding_box_area``
     - Minimum predicted body bbox area (px squared) below which
       silhouette rendering is skipped.
   * - ``bodies.oversample_edge_limit``
     - Anti-aliasing oversample limit for the silhouette render.
   * - ``bodies.oversample_maximum``
     - Hard cap on the per-axis oversample factor.
   * - ``bodies.use_lambert``
     - Use Lambert shading (default true) vs. flat-disc rendering.
   * - ``bodies.use_albedo`` / ``bodies.geometric_albedo``
     - Apply per-body geometric albedo when computing brightness.

The bodies considered for navigation are the planet returned by
``obs.closest_planet`` plus the satellites configured under the
top-level ``satellites.<PLANET>`` mapping in
``config_100_satellites.yaml``.

Ring Navigation Model
---------------------

The ring navigation model generates theoretical brightness profiles
for planetary ring edges and emits one feature per surviving edge.
Two top-level options in ``config_050_rings.yaml`` control whether
ring pixels in shadow are excluded from the model before navigation.

For each surviving ring feature the model emits one of:

- ``RING_EDGE`` — a per-vertex polyline of edge coordinates with
  per-vertex radial sigma derived from the catalog ``rms`` divided by
  the radial km-per-pixel scale.  When the polyline is straight
  (deviation from a best-fit line below 1 px) the ``is_straight_line``
  flag is set so techniques can handle the rank-1 covariance.
- ``RING_ANNULUS`` — a multi-edge composite template emitted when the
  surviving polyline compresses radially below 5 px (the edges are
  not separable at the image scale).

Per-edge feature definitions live in the per-planet ring files
(``config_300_jupiter_rings.yaml``, ``config_310_saturn_rings.yaml``,
``config_320_uranus_rings.yaml``, ``config_330_neptune_rings.yaml``)
under ``rings.ring_features.<PLANET>.features``.  Only the Saturn file
carries features today; the Jupiter, Uranus, and Neptune files are
empty placeholders.  See "Ring YAML configuration" in the developer
guide for the full schema.

Planet shadow removal
^^^^^^^^^^^^^^^^^^^^^

When a planet casts a shadow across part of its own ring system, those ring
arcs appear dark in the image. If the model still shows those arcs as bright,
the navigator will try to align a bright model against a dark image region,
which introduces a systematic pointing error.

The ``rings.remove_planet_shadow`` option (default ``true``) instructs the
ring model to zero out all ring pixels that fall inside the planet's own
shadow:

.. code-block:: yaml

   rings:
     remove_planet_shadow: true   # default

When active, the ring model logs the number of masked pixels at ``INFO`` level:

.. code-block::

   Planet shadow removal: 1284 pixel(s) inside SATURN shadow will be masked

To disable shadow removal entirely -- for example, to compare navigation
quality with and without the mask -- set the option to ``false`` in a
``--config-file`` override:

.. code-block:: yaml

   rings:
     remove_planet_shadow: false

Body shadow removal (future)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``rings.remove_body_shadows`` option (default ``false``) is reserved for a
future enhancement that will remove ring pixels shadowed by moons. Setting it
to ``true`` has no effect in the current release.

.. code-block:: yaml

   rings:
     remove_body_shadows: false   # default; not yet implemented

Titan Navigation Model
----------------------

Titan's atmospheric haze is opaque at most wavelengths, so what a camera
sees is not the solid surface but the haze top: hundreds of kilometers
up, at an altitude that varies with wavelength, latitude, season, and
phase.  Fitting an ellipsoid limb to that edge is systematically wrong
rather than merely noisy, so Titan is navigated from a property of the
haze itself.

Absent clouds or visible surface features, a hazy atmosphere is
mirror-symmetric about the image-plane line through the body center and
the sub-solar point.  The image shift perpendicular to that line
("cross-track") is the shift that maximizes mirror symmetry; and because
the limb arc facing the Sun is close to circular, a circle fit with a
*free* radius to that arc gives the shift along the line
("along-track") without assuming any haze altitude.  The free radius is
what makes the method filter-independent: a haze top that sits higher in
blue than in red changes the fitted radius, not the fitted center.  The
method is published as Hanson, French, Waugh, Barth and Anderson (2025),
*Geophysical Research Letters*, doi:10.1029/2024GL113415.

**What it produces.**  Whenever Titan is inside the extended field of
view the model emits a single ``TITAN_LIMB`` feature and
:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`
measures the offset from it, on any instrument and any filter, with no
per-filter or per-phase training data.  The reported
uncertainty is deliberately *anisotropic*: the mirror-symmetry scan
localizes the cross-track direction far more tightly than the circle fit
localizes the along-track one, and the ensemble consumes that ellipse
rather than an averaged circle.

**Accuracy.**  Single-frame accuracy is **1 px or better cross-track and
3 px or better along-track**.  That bound comes from planted-truth
simulation (the 95th percentile of recovery error on the clean-scene
family of a 700-scene randomized campaign is 0.17 px cross-track and
0.82 px along-track; families with injected artifacts run wider) and is
confirmed
on real frames by an independent witness: over the Cassini validation
cohort, frames where a star technique locks independently give an
absolute per-frame anchor, and the haze fit disagrees with it by 0.99 px
rms cross-track and 1.50 px rms along-track over nine such pairs --
about 0.70 and 1.06 px of single-frame error once the anchor's own
uncertainty is removed.  A second anchor class corroborates the first:
when another moon shares the field of view, its own limb navigation
measures the same scene-wide offset, and it agrees with the haze fit at
2-sigma on 11 of the 12 cohort frames where both commit.  Repeat frames
of one target through one filter agree to 0.34 px cross-track and
0.33 px along-track.

Two consequences of the along-track figure are worth planning around.
An image whose only navigable content is Titan reports at most the
``medium`` confidence tier, because the honest along-track uncertainty
of a single quasi-circular feature exceeds the ``high`` tier's sigma
budget; adding a star field or a resolved moon to the frame is what
lifts it.  And a *small* Titan at *high* phase is the method's weak
regime: the sunward arc has its least support there, and it is where
essentially all of the along-track error lives.  Bodies whose apparent
solid radius is 40 px or more recover to 0.72 px along-track at the 95th
percentile across every phase; below 40 px above 60 degrees of phase the
same percentile is 3.0 px.

**Marginal frames are refused, not guessed.**  Three conditions score
the feature's reliability at exactly zero, after which the standard
per-feature-type gate removes it before any fit runs:

* the haze envelope, allowing for the full pointing search window, does
  not fit inside the detector;
* more than ``max_occluded_fraction`` of the envelope is hidden by a
  nearer moon or by the rings (the main rings are treated as opaque, so
  a Titan seen through the C ring or a gap is refused rather than fitted
  through ring stripes);
* the envelope is smaller than ``min_envelope_diameter_px`` across.

A frame refused this way ends with ``status_reason``
``all_features_gated``, and the per-image ``_metadata.json`` records the
measured envelope diameter and occluded fraction that produced the
refusal, so the cause is readable without re-running anything.

Frames that clear the gate can still be refused by the fit itself.  Each
of the seven fit gates below rejects a frame with its name recorded in
the technique's diagnostics, and a Titan-only frame whose fit is
rejected ends ``all_techniques_spurious``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Gate
     - Rejects the frame when
   * - ``valid_fraction``
     - Too much of the symmetry annulus is masked or off-frame to
       correlate.
   * - ``peak_score``
     - The best mirror-symmetry score is too low -- the haze is not
       symmetric enough to measure.
   * - ``second_peak``
     - A rival symmetry peak comes too close to the winner, so the axis
       could lock onto the wrong one.
   * - ``ray_yield``
     - Too few limb rays survive; the sunward limb is not detectable
       along enough of the arc.
   * - ``arc_inliers``
     - The robust circle fit rejected too many of the rays it was given.
   * - ``arc_radius``
     - The fitted radius is implausible for the body's known size.
   * - ``arc_residual``
     - The sunward limb departs too far from a circle.

Whichever way a frame ends, it is attributable: a committed offset, a
named fit gate, or a gated feature whose reliability breakdown says why.
Nothing produces a silent empty failure.

**Overlay.**  The summary PNG draws the predicted haze envelope circle,
the symmetry axis, the sunward arc sector, and a center cross.  Because
annotations are composited at the navigated offset, the drawn circle
lands on the fitted position on a committed frame and stays at the SPICE
prediction when nothing was committed.  A feature below the reliability
gate is drawn dotted and labeled ``TITAN (low reliability)``.

**Configuration.**  ``config_060_titan.yaml`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Key
     - Effect
   * - ``titan.atmosphere_height``
     - Haze envelope above the solid radius in km (default 700).  Bounds
       the search annulus and the ray windows; the fit itself assumes no
       haze altitude.
   * - ``titan.navigation.min_envelope_diameter_px``
     - Envelope diameter below which reliability is forced to zero.
   * - ``titan.navigation.max_occluded_fraction``
     - Occluded share of the envelope above which reliability is forced
       to zero.
   * - ``titan.navigation.ring_occlusion_radii_km``
     - ``[inner_km, outer_km]`` ring-plane range treated as opaque.
   * - ``titan.navigation.axis_min_offset_px``
     - Below this predicted-center-to-sub-solar distance the disc is
       treated as rotationally symmetric and the axis search is skipped.
       Scales with the sampling stride of the incidence backplane.
   * - ``titan.navigation.backplane_max_samples``
     - Largest grid the symmetry axis's incidence backplane is evaluated
       over.  A closer Titan whose box exceeds it is sampled every few
       pixels instead, which quantizes the initial axis by a fraction of
       a degree at ordinary phases, within what the angle refinement
       searches.  The default of one million leaves any box up to 1000
       pixels on a side sampled every pixel.  The sampling step is a whole
       number of pixels, so a box just over a threshold takes the next step
       up: it uses a quarter of the budget and quantizes the axis twice as
       coarsely as the box size alone suggests, still inside what the
       refinement searches down to about 3.5 degrees of phase.
   * - ``titan.navigation.recenter_threshold_px``
     - Along-track shift above which the fit runs a second, recentred
       pass.
   * - ``titan.navigation.star_mask_vmag_limit`` /
       ``titan.navigation.star_mask_radius_px``
     - Brightness above which a catalog star is masked out of the fit,
       and the radius of each masked disc.
   * - ``titan.navigation.reliability_diameter_midpoint_px`` /
       ``titan.navigation.reliability_diameter_scale_px``
     - Midpoint and width of the sigmoid that turns apparent size into
       feature reliability.
   * - ``titan.navigation.surface_window_filters``
     - Filters that see through the haze to the surface.  Recorded as a
       diagnostic flag; the fit does not branch on it.
   * - ``titan.navigation.high_phase_deg``
     - Phase angle above which the emitted feature is flagged
       ``high_phase``, marking frames whose sunward arc carries its
       least support.
   * - ``titan.annotation.*``
     - Overlay styling: the dot spacing that marks a below-gate feature
       and the size of the center cross.
   * - ``titan.navigation.symmetry.*``
     - Cross-track scan: annulus extent, symmetry-angle refinement, the
       ``valid_fraction`` / ``peak_score`` / ``second_peak`` gate
       thresholds, and the reported cross-track sigma's scale and floor.
   * - ``titan.navigation.arc.*``
     - Along-track circle fit: sunward sector width and ray spacing,
       radial sampling, the limb-gradient signal-to-noise cut, the
       ``ray_yield`` / ``arc_inliers`` / ``arc_residual`` gate
       thresholds, the robust-fit tuning constant, and the reported
       along-track sigma's scale and floor.

The covariance model-error floor and the confidence-formula
coefficients live with the other techniques, under
``techniques.TitanHazeNav`` in ``config_510_techniques.yaml``.  The
developer guide documents every key's default and its measured
justification: see :doc:`/dev_guide/dev_guide_navigation_models_titan`
and :doc:`/dev_guide/dev_guide_techniques_titan_haze`.

Navigation Techniques
=====================

The autonomous-navigation pipeline runs every registered ``NavTechnique``
whose feasibility check passes on the surviving feature set, then combines
the per-technique offsets via the orchestrator's precision-weighted
ensemble.  Use ``--nav-techniques`` to restrict which techniques run; the
default ``*`` runs all of them.

The algorithmic detail (DT pipeline, Levenberg-Marquardt refinement,
information-matrix covariance) lives in
:doc:`/dev_guide/dev_guide_techniques` and
:doc:`/dev_guide/dev_guide_techniques_dt_fitting`; this page summarizes
what each technique does and which scenes it applies to.

Implemented techniques
----------------------

``BodyLimbNav``
^^^^^^^^^^^^^^^

Translation fit on a body's lit limb.  Consumes every ``LIMB_ARC`` feature
emitted by ``NavModelBody``, concatenates their per-vertex polylines, and
runs a coarse-NCC plus Levenberg-Marquardt refinement against the image
edge-distance transform.  Tukey biweight reweighting rejects outlier
vertices; the M-estimator information matrix at the converged solution
yields the result's covariance.  Multi-body inputs sharpen the fit by
``sqrt(N_bodies)``.

Best for: scenes where one or more bodies show a visible limb arc
(typical Cassini ISS Mimas / Enceladus / Tethys / Dione / Rhea encounter
images).  Feasibility threshold: at least one limb arc with at least
30 surviving polyline vertices.

At very low phase (below about 15 degrees) the lit arc spans almost the
whole silhouette and the across-limb gradient that constrains the fit is
weak, so the fit can lock onto the wrong basin.  The technique detects that
mis-lock and marks the result spurious rather than emit a confident
multi-pixel offset, so a near-fully-lit single body is navigated by the disc
or other techniques instead of by a misleading limb fit.

``BodyTerminatorNav``
^^^^^^^^^^^^^^^^^^^^^

Same shape as ``BodyLimbNav`` on ``TERMINATOR_ARC`` features, with two
differences: each body's per-vertex sigmas collapse to one per-body scalar
(the body's mean sigma), and the confidence formula carries
phase-angle-factor and albedo-penalty terms.  Best for crescent
geometries where the terminator runs through bright, nearly-uniform
hemispheres.

``RingEdgeNav``
^^^^^^^^^^^^^^^

DT-based fit on every ``RING_EDGE`` feature.  Polarity prediction is
intentionally disabled today (the ring catalog does not yet flag
polarity_predictable, deferred work).  When every input edge is
straight-line the technique reports ``is_rank_1=True`` and returns an
honest rank-deficient covariance; the ensemble combine fuses it with any
orthogonal-axis result (a star, body limb, body blob) before declaring a
final answer.

Best for: close-range ring scenes.  For Saturn the rings model emits
per-edge features only below 25 km/px radial resolution and routes
everything coarser to ``RingAnnulusNav``, which a 131-frame
operator-audited head-to-head measured wrong on zero accepted answers
in every resolution band it was measured.  At fine resolution the
ring's many similar concentric ringlet edges resolve individually and
a shape-only edge fit can lock onto the wrong one, and below 25 km/px
neither ring technique is yet validated at scale, so a sub-25 km/px
ring-edge answer that no other technique corroborates warrants care.

``RingAnnulusNav``
^^^^^^^^^^^^^^^^^^

Pyramid-NCC fit on every ``RING_ANNULUS`` feature.  ``RING_ANNULUS``
features are emitted by the rings model in two regimes: when a curved
(non-straight) ring edge compresses radially to at most the per-planet
``feature_emission.ring_annulus.max_radial_px`` threshold in
``config_510_techniques.yaml`` (individual edges no longer separable;
a straight-line compressed edge stays a rank-1 ``RING_EDGE`` instead),
and when the scene's radial resolution is at or above the per-planet
km/px threshold -- 25 km/px for Saturn, so the whole Saturn system is
annulus-class at that resolution and coarser.  Under the system-level
gate every surviving ring collapses into a single composite annulus
per planet; below it a scene can emit a mix, with ``RING_EDGE``
features alongside one composite for the edges that compressed below
``max_radial_px``.  Multi-planet scenes
(rare) emit one ``RING_ANNULUS`` per ring system; the technique fuses
them via Z-buffer paint and runs one joint NCC.
``use_gradient='auto'`` self-selects raw vs gradient mode per image.

Best for: ring scenes at or above the per-planet km/px threshold: a
131-frame operator-audited head-to-head measured it wrong on zero
accepted answers at every resolution band, and its
rendered-brightness template disambiguates similar concentric edges
that a shape-only fit can confuse (distant Cassini ring views;
potential NHLORRI Pluto/Charon ring geometries).

``NavTechniqueManual``
^^^^^^^^^^^^^^^^^^^^^^

Interactive PyQt6 dialog that composes every template-bearing feature
into a single ext-FOV overlay and lets the operator pick the offset by
hand.  Not part of the autonomous registry; opt into it from the normal
``sd_offset`` driver with the ``--manual`` flag, which requires the
selection to resolve to exactly one image:

.. code-block:: bash

   echo W1521598221_1_CALIB > /tmp/img_list.txt
   sd_offset coiss --manual --image-file-list /tmp/img_list.txt

The driver loads the image, runs the orchestrator's ``prepare`` step
(image classifier + NavModels + features + reliability gate), opens the
dialog, and prints the chosen ``offset_dv_px`` / ``offset_du_px`` to
stdout.  Exit code is ``2`` if the dialog is canceled or no
template-bearing features are available.  The dialog's **Save as
Library Entry...** button is the recommended path for adding a sidecar
to the operator-curated test image library; see
:doc:`/dev_guide/dev_guide_image_library`.

Programmatic equivalent (one obs in, ``NavTechniqueResult`` out):

.. code-block:: python

   from spindoctor.nav_technique import run_manual_nav

   result = run_manual_nav(obs)

Filtering examples
------------------

Run only the ring-edge technique:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques RingEdgeNav

Run every technique except ``BodyTerminatorNav``:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques '!BodyTerminatorNav'

Run both DT body techniques together:

.. code-block:: bash

   sd_offset coiss N1234567890 --nav-techniques 'BodyLimbNav,BodyTerminatorNav'

Output
------

Every technique that runs contributes one entry to
``NavResult.per_technique`` carrying the per-technique offset, 2x2
covariance, calibrated confidence, and a typed ``*Diagnostics``
dataclass.  The orchestrator's ensemble combine reconciles those
entries into a single ``NavResult.offset_px`` and ``confidence_rank``;
both numbers land in the per-image ``_metadata.json``.

