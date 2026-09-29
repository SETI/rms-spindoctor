================================
Navigation Models and Techniques
================================

Navigation compares an image against a prediction of what the camera should have been
looking at. This chapter explains how that prediction is built, how it is matched against
the real picture, and how you control which parts of the machinery run. Read it before
tuning a run: the vocabulary here is the vocabulary the logs and the per-image metadata
document use.

How a Navigation Works
======================

Every navigated image passes through the same four steps.

First, SpinDoctor asks the SPICE kernels where the spacecraft was, where it was pointing,
and where the planets, moons, and rings were at the moment of the exposure. That answer is
the *predicted* pointing, and it is usually wrong by a few pixels to a few tens of pixels.

Second, a set of **navigation models** turns the predicted geometry into a prediction of
the picture. A model is a family of scene content: the stars, the solid bodies, the rings,
or the haze of a body whose atmosphere hides its surface. Each model reports the
individual things it expects to see and where in the frame it expects each one.

Third, a set of **navigation techniques** measures where those things actually are. Each
technique is one matching method suited to one kind of content: correlating a rendered
disc against the image, sliding a predicted limb curve onto the real brightness edge,
recognizing a star pattern, and so on. Each technique that runs reports its own offset,
its own uncertainty, and its own confidence.

Fourth, the results are reconciled into a single answer. Techniques that agree are fused,
a technique that contradicts a corroborated majority is set aside, and the surviving
estimates are combined by weighting each by how tightly it is determined. What comes out
is one offset in pixels, one uncertainty, and one confidence for the image.

The offset follows one convention throughout SpinDoctor: if the prediction puts something
at row and column ``(v, u)``, the real thing is at ``(v + dv, u + du)``.

Features
--------

A **feature** is the smallest piece of a scene that can constrain the pointing by itself.
One catalog star is a feature. So is one moon's lit limb, one moon's terminator, one moon
rendered as a disc, one named ring edge, and one haze envelope. When a model *emits* a
feature it is adding one such item to the pool the techniques draw from, together with
where it is predicted to fall, how uncertain that prediction is, and a name you will see
in the logs and the metadata document (``limb_arc:MIMAS``, ``ring_edge:SATURN:A_outer``,
``star:UCAC4:144787700``).

The feature types navigation emits are:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Feature type
     - What it is
   * - ``STAR``
     - One catalog star, as a single predicted pixel position.
   * - ``LIMB_ARC``
     - One body's lit limb, as a curve of predicted vertex positions.
   * - ``TERMINATOR_ARC``
     - One body's terminator, the day-night boundary drawn across the disc.
   * - ``BODY_DISC``
     - One body rendered as a small picture, for brightness correlation.
   * - ``BODY_BLOB``
     - One body too small or too irregular to trace a limb on, carried as a
       predicted center and a bounding box only.
   * - ``RING_EDGE``
     - One named ring edge, as a curve of predicted vertex positions.
   * - ``RING_ANNULUS``
     - A whole ring system rendered as one small picture, used when the
       individual edges are not separable at the image scale.
   * - ``TITAN_LIMB``
     - The haze envelope of a body whose thick atmosphere hides its surface.

Each feature also carries a **reliability** between 0 and 1: the model's own assessment of
how measurable this particular item is in this particular image. A star predicted to fall
behind a moon scores zero. A limb whose predicted position is uncertain by several pixels
scores low. Before any technique runs, features scoring below a per-type minimum are set
aside and never fitted. They still appear in the metadata document, marked as set aside
and carrying the reason. The minimums are the ``orchestrator.reliability_gate`` settings:
0.20 for stars and blobs, and 0.30 for every other type.

Confidence
----------

**Confidence** is a number between 0 and 1 attached to a navigation result, answering how
much the offset should be trusted. Every technique reports its own confidence from its own
diagnostics: how many stars matched, how large the residuals were, how sharp the
correlation peak was, and so on. The image then carries one combined confidence, raised
when independent techniques corroborate one another and reduced when they disagree.

A **structural failure** forces a technique's confidence to exactly zero. That means the
technique reported a named defect in its own fit rather than merely an imprecise answer:
most often that its solution ran into the edge of the window it was allowed to search, or
that the technique judged its own answer **spurious**. Spurious is a technique's own
verdict, from a symptom it can name, that its answer is untrustworthy rather than simply
loose. Such an answer never contributes to the reported offset. It is still recorded, with
the verdict beside it, and a status reason of ``all_techniques_spurious`` means every
technique that ran reached that verdict.

An image whose combined confidence falls below ``orchestrator.ensemble.min_confidence``
(0.35) is recorded as a failed navigation and reports no offset.

Above that floor, the result is also placed in one of five **confidence tiers**, and the
tier is the number to act on, because it folds in the uncertainty that bare confidence
does not:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Tier
     - Meaning
   * - ``high``
     - Confidence at least 0.85 and per-axis uncertainty at most 0.5 pixel. The
       pointing is well determined; use it without further checking.
   * - ``medium``
     - Confidence at least 0.35 and per-axis uncertainty at most 2.0 pixels.
       Trustworthy for most work. Carry the reported uncertainty into anything
       quantitative.
   * - ``low``
     - Confidence at least 0.35 with no limit on the uncertainty. The offset is
       probably right in direction, but read the reported uncertainty before relying
       on its size.
   * - ``conflicted``
     - Techniques genuinely disagreed. An offset is reported, and it is suspect.
   * - ``failed``
     - No usable offset.

The ``medium`` and ``low`` tiers share the same confidence floor, so the difference
between them is entirely the uncertainty. A result can be confident and still land in
``low`` because its uncertainty is large. The thresholds themselves are the
``orchestrator.ensemble.tier_thresholds`` settings.

One caveat travels with every confidence value SpinDoctor writes. The numbers and the tier
boundaries are calibrated against simulated images whose true offsets were known, so they
rank results correctly against one another but should not be read as probabilities of
accuracy on real data. Every metadata document that carries a navigation result records
that basis. For the recorded values and their exact keys see
:doc:`/user_guide/user_guide_metadata`.

Navigation Models
=================

Four model families ship with SpinDoctor: stars, solid bodies, rings, and the haze
envelope of a body whose atmosphere hides its surface. Every model reads its settings from
the configuration system described in :doc:`/user_guide/user_guide_configuration`, and
this chapter names a setting by its configuration key wherever it quotes a number you can
change.

Star Navigation Model
---------------------

The star model works out which catalog stars should be visible, where each should fall on
the detector, and how sharply each should appear.

It searches its catalogs in the order given by ``stars.catalogs``, from the most precise
to the least; the default order is UCAC4, then Tycho-2, then the Yale Bright Star catalog
(YBSC). A star found in more than one catalog is kept once, from the most precise catalog
that has it. Two stars count as the same star when their positions agree to within
``stars.duplicate_ra_dec_threshold_arcsec`` and their magnitudes to within
``stars.duplicate_vmag_threshold``. Where two different stars are close enough that their
images overlap, the fainter one is set aside, and both are set aside when they are within
``stars.overlapping_vmag_threshold`` magnitudes of each other, because neither centroid
can then be trusted. What survives is sorted brightest first and cut to
``stars.max_stars``.

Each catalog position is then corrected for the star's own proper motion at the time of
the exposure, and for stellar aberration, the small apparent displacement caused by the
spacecraft's own velocity. Both corrections are on by default and can be turned off with
``stars.proper_motion`` and ``stars.stellar_aberration``.

How faint a star can be and still be used is not a setting. It is worked out for each
image from the camera's aperture and detector and from the exposure time: each factor of
2.512 more exposure buys one more magnitude of depth. A star fainter than that limit, or
one that has no recorded magnitude, is dropped. How much brighter than the limit a star is
determines its predicted signal-to-noise ratio, which in turn sets its reliability and how
precisely its position can be measured.

If the spacecraft was turning during the exposure, stars are not points but short trails.
The model computes each star's trail by projecting its direction through the camera at the
start and at the end of the exposure, so stars in different parts of a distorted field
trail by different amounts. A trail lengthens a star's uncertainty along the trail and
leaves it alone across the trail, and it makes the star's predicted image a smeared
version of the camera's point-spread function rather than a compact spot. A star whose
trail exceeds ``stars.max_smear`` pixels is dropped as unmeasurable; the default of 100
pixels is permissive enough that it rarely applies.

Finally the model checks each star's predicted pixel against the solid bodies in the
frame, and against the opaque parts of the ring system. A star predicted to fall on a
body, allowing ``stars.body_conflict_margin`` pixels of slop, is marked as hidden behind
that body. Otherwise, when ``stars.ring_occlusion_enabled`` is on, the same neighborhood
is tested against the ring radius ranges listed in ``stars.ring_occlusion_radii_km`` for
that planet, and the star is marked as hidden when at least
``stars.ring_occlusion_min_opaque_fraction`` of the tested pixels fall inside an opaque
range. A body always wins over a ring. Hidden stars are still emitted as features, with
reliability zero, so a run can report how many stars were lost and to what; no technique
ever fits them.

Body Navigation Model
---------------------

For each body whose predicted outline overlaps the frame, the body model renders what that
body should look like: an oversampled silhouette shaded so that the brightness falls off
toward the edges the way sunlight on a sphere does. From that rendering it traces the lit
limb and the terminator, and it emits some combination of four feature types depending on
how large the body is, how it is lit, and how well its real form is known.

``LIMB_ARC``
   The traced lit limb. Only the lit part is traced, because the unlit limb merges into
   dark sky and leaves no edge to fit. It is emitted when the predicted limb position is
   uncertain by no more than 3 pixels and at least 30 vertices of it survive. Both
   conditions matter: the pixel test alone would pass a very distant small moon, whose
   limb is well predicted in pixels precisely because it is tiny, which is when a limb fit
   is least useful. The feature carries a separate uncertainty at each vertex, larger
   where the body's real surface departs from a smooth ellipsoid and larger where the Sun
   grazes the surface most steeply.

``BODY_BLOB``
   Emitted instead of ``LIMB_ARC`` when the limb cannot be traced usefully, the predicted
   disc is nonetheless wide enough to centroid, and at least part of the silhouette is
   lit. A body entirely in shadow has no brightness to centroid and emits no blob. A
   blob is also emitted *alongside* a limb when both are available, as an independent
   cross-check that can catch a curve fit that locked onto the wrong outline.

``BODY_DISC``
   The rendered picture of the body itself, for brightness correlation. It is emitted only
   alongside ``LIMB_ARC``, never on its own, and only when the body sits well inside the
   frame, with no more than 30 percent of its disc off the edge and at least 40 percent of
   its lit hemisphere visible. A body whose limb was rejected therefore has no disc
   either.

``TERMINATOR_ARC``
   The traced day-night boundary, emitted when at least 8 vertices of it survive and the
   phase angle is above about 3 degrees. Below that the Sun is almost directly behind the
   camera, the terminator has retreated onto the limb, and the two cannot be told apart.
   As with the limb, the fit needs 30 vertices, so a short terminator arc is emitted and
   then goes unused.

Per-body surface knowledge drives all of this. SpinDoctor carries a table of how far each
body departs from a smooth ellipsoid, how deep its craters are, how much its surface
brightness varies, and how well its orbit is known. Those quantities set the per-vertex
uncertainties and decide when a body is too irregular for a limb fit. A body absent from
the table is treated as a generic icy moon with conservative values. A body the table
marks as highly irregular emits only a blob once it is resolved at all, because no smooth
predicted outline describes it: Hyperion and Phoebe are the cases to expect.

The bodies considered are the planet nearest the line of sight plus that planet's
satellites as listed under the ``satellites`` configuration key. Rendering is skipped
entirely for a body whose predicted outline covers less area than
``bodies.min_bounding_box_area`` pixels. ``bodies.use_lambert`` controls whether the
silhouette is shaded at all or rendered as a flat disc. Applying each body's own
reflectivity when computing brightness is off by default; turning on ``bodies.use_albedo``
enables it, and the values it then uses come from ``bodies.geometric_albedo``.

Ring Navigation Model
---------------------

The ring model predicts where each named ring edge in SpinDoctor's catalog of ring
features should fall, and how bright the ring system should be at each pixel. Ring
positions are known from a catalog of edge radii, each with its own quoted radial
uncertainty, and that uncertainty divided by the image's radial scale is the uncertainty
the model reports for that edge.

Whether the model describes the rings edge by edge or as one picture depends on how much
ring radius one pixel spans. That quantity is the **radial resolution**, quoted in
kilometers of ring radius per pixel, and a larger number means a coarser view: a distant
Cassini image of the whole ring system might be 300 kilometers per pixel, while a
close-range image of a single ringlet might be 5. Coarse views cannot separate neighboring
edges at all, so describing them individually is meaningless.

Accordingly the model emits one of two feature types:

``RING_EDGE``
   One named edge as a curve of vertices, each with its own radial uncertainty. This is
   what the model emits when the view is fine enough to resolve edges individually.

``RING_ANNULUS``
   The whole ring system as one rendered picture of its predicted brightness. This is what
   the model emits when the mean radial resolution across the frame is at or coarser than
   the threshold for that planet, which is 25 kilometers per pixel for Saturn. It is also
   what the model emits for an individual curved edge that has compressed into a strip 5
   pixels wide or less, because such an edge is no longer distinguishable from its
   neighbors. One such picture is produced per ring system, whatever the number of
   edges that went into it.

Both thresholds are configurable per planet, as
``feature_emission.ring_annulus.planets.<PLANET>.kmpp_threshold`` and
``feature_emission.ring_annulus.planets.<PLANET>.max_radial_px``, with fallbacks under
``feature_emission.ring_annulus.default``.

Saturn's threshold is 25 kilometers per pixel. Coarser than that, the edge-by-edge fit
becomes unreliable while the brightness match does not, so the brightness match is the one
used. The thresholds for Jupiter, Uranus, and Neptune are set from the widths of those
systems.

A ring edge can also come out **straight**: when the curve that the model traced never
departs from a straight line by more than a pixel. That happens in two real situations.
The first is a view close to the ring plane, where the whole ring system projects into a
narrow band and every edge in the frame is a parallel straight streak. The second is a
high-resolution view of a short piece of a very large circle, where the curvature over the
few hundred pixels in the frame is less than a pixel. A straight edge is flagged as such,
because it constrains the pointing in only one direction; the consequences are described
under ``RingEdgeNav`` below.

Only Saturn's ring catalog is populated. The per-edge definitions live in the ring
configuration, under
``rings.ring_features.<PLANET>.features``; the Jupiter, Uranus, and Neptune entries are
empty. The full layout of an edge definition is documented in
:doc:`/dev_guide/dev_guide_navigation_models_rings`.

Planet shadow removal
^^^^^^^^^^^^^^^^^^^^^

Saturn casts a shadow across part of its own rings, and the ring arcs inside that shadow
are dark in the image. A model that still shows them bright would ask the navigator to
align a bright prediction against a dark part of the image, which drags the fitted offset
off the truth.

The ``rings.remove_planet_shadow`` setting, on by default, zeroes every model pixel that
falls inside the planet's shadow, so those arcs enter neither the traced edges nor the
rendered picture. To compare navigation with and without it, turn it off in a
configuration file passed with ``--config-file``:

.. code-block:: yaml

   rings:
     remove_planet_shadow: false

If you write a ``rings:`` override of your own, keep the key: when it is absent altogether
the model treats shadow removal as off.

Shadows that moons cast on the rings are a separate matter. SpinDoctor does not remove
them, so a moon's shadow shows in the image but not in the model that is matched against
it.

Ring positions also depend on how well the rings' own orbits are known, and that
contribution is quoted as ``rings.default_orbit_radial_sigma_km``, applied with the
correlated fraction ``rings.orbit_radial_sigma_correlated_fraction``. Raising the first
widens the reported uncertainty of every ring-derived offset.

Titan Navigation Model
----------------------

Titan's atmospheric haze is opaque at most wavelengths. What a camera sees is therefore
not the solid surface but the top of the haze, hundreds of kilometers above it, at an
altitude that varies with wavelength, latitude, season, and phase. Fitting an ellipsoid
limb to that edge gives an answer that is systematically wrong rather than merely noisy,
so Titan is navigated from a property of the haze itself.

Absent clouds or visible surface markings, a hazy atmosphere is mirror-symmetric about the
line in the image plane that runs through the body's center and the sub-solar point. Two
measurements follow from that. The offset perpendicular to that line is the shift that
makes the image most nearly mirror-symmetric about it. The offset along the line comes
from fitting a circle to the sunward part of the limb, with the circle's radius left free.
A free radius is what makes the method work in any filter: a haze top that sits higher in
blue than in red changes the fitted radius and leaves the fitted center alone. The method
is published as Hanson, French, Waugh, Barth, and Anderson (2025), *Geophysical Research
Letters*, `doi:10.1029/2024GL113415 <https://doi.org/10.1029/2024GL113415>`_.

Whenever Titan is in the frame the model emits a single ``TITAN_LIMB`` feature, and
``TitanHazeNav`` measures the offset from it, on any instrument and in any filter, with no
per-filter or per-phase training data. The uncertainty that comes back is deliberately
lopsided: the symmetry measurement pins the across-line direction far more tightly than
the circle fit pins the along-line direction, and that ellipse is carried forward as it
stands.

Single-frame accuracy is 1 pixel or better across the symmetry line and 3 pixels or better
along it. Two consequences are worth planning around.

An image whose only navigable content is Titan reaches the ``medium`` tier at best,
because the honest along-line uncertainty of one nearly circular feature is larger than
the ``high`` tier allows. Adding a star field or a second, resolved moon to the frame is
what lifts such an image higher.

A *small* Titan at *high* phase is the method's weakest case, because the sunward arc that
the circle is fitted to is shortest there, and almost all of the along-line error lives in
that regime. A body whose apparent solid radius is 40 pixels or more is good to about 0.7
pixel along the line at any phase; below 40 pixels and above 60 degrees of phase the same
figure is about 3 pixels.

Marginal frames are refused. Three conditions each force the haze feature's reliability to
zero, after which the usual reliability minimum removes it before any fitting happens:

* the haze envelope, allowing for the whole pointing search window, does not fit inside
  the detector;
* more than ``titan.navigation.max_occluded_fraction`` of the envelope is hidden by a
  nearer moon or by the rings, the main rings being treated as opaque, so a Titan seen
  through the C ring or through a gap is refused rather than fitted through ring stripes;
* the envelope is narrower than ``titan.navigation.min_envelope_diameter_px`` across.

Such a frame ends with a status reason of ``all_features_gated``, and its metadata
document records the measured envelope diameter and hidden fraction that caused the
refusal, so the cause is readable without re-running anything.

A frame that clears those conditions can still be refused by the fit. Each of the
following checks rejects the frame, and the name of the check that fired is recorded in
the metadata document. A Titan-only frame rejected this way ends with a status reason of
``all_techniques_spurious``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Check
     - Rejects the frame when
   * - ``valid_fraction``
     - Too much of the ring of pixels being compared is masked out or off the frame.
   * - ``peak_score``
     - The best mirror-symmetry score is too low: the haze is not symmetric enough to
       measure.
   * - ``second_peak``
     - A rival symmetry orientation scores nearly as well as the winner, so the
       symmetry line could be placed wrongly.
   * - ``ray_yield``
     - Too few points along the sunward limb were detected at all.
   * - ``arc_inliers``
     - The circle fit had to reject too many of the points it was given.
   * - ``arc_radius``
     - The fitted radius is implausible for a body of this known size.
   * - ``arc_residual``
     - The sunward limb departs too far from a circle.

Every outcome is therefore attributable: a committed offset, a named check that fired, or
a refused feature whose reliability breakdown says why.

The summary picture draws the predicted haze envelope, the symmetry line, the sunward arc,
and a cross at the fitted center. Annotations are drawn at the navigated offset, so on a
frame that committed an answer the circle lands on the fitted position, and on one that
did not it stays at the SPICE prediction. A refused feature is drawn as a dotted circle
labeled ``TITAN (low reliability)``.

The settings most likely to matter to you are ``titan.atmosphere_height``, the assumed
haze height above the solid radius in kilometers, which bounds the region searched without
entering the fit itself; the two refusal conditions
``titan.navigation.min_envelope_diameter_px`` and
``titan.navigation.max_occluded_fraction``; ``titan.navigation.ring_occlusion_radii_km``,
the ring radius range treated as opaque; ``titan.navigation.star_mask_vmag_limit``, above
which a catalog star is masked out of the fit; and ``titan.navigation.high_phase_deg``,
which defaults to 150 degrees and is the phase above which a frame is marked as
strongly backlit. It is not the 60-degree figure quoted above, which describes where
the measured accuracy falls off rather than where a frame is marked. The measurements
behind the method are described in
:doc:`/dev_guide/dev_guide_navigation_models_titan` and
:doc:`/dev_guide/dev_guide_techniques_titan_haze`.

Navigation Techniques
=====================

A technique runs when the surviving features include something it can use. Several
techniques usually run on one image, and their answers are reconciled afterwards.

Three of the techniques below share one matching method, because fitting a predicted curve
onto a real brightness edge is the same problem whether the curve is a body's limb, a
body's terminator, or a ring edge. That shared method has three stages, and its vocabulary
appears in the logs and in the metadata document:

**The coarse search.** The predicted curve is drawn as a thin line of pixels and slid over
the edges detected in the real image at every whole-pixel shift inside the search window.
Each shift is scored by what fraction of the curve's vertices land on a detected edge, and
the best-scoring shift becomes the starting guess for the refinement. The logs and the
metadata document call this the coarse NCC stage, NCC standing for normalized
cross-correlation, the family of pattern-matching scores this one belongs to.

For a body's limb or terminator, each vertex's contribution to that score is also weighted
by **polarity**, which means knowing in advance which side of an edge should be the
brighter one. Across a body's limb the answer is known: outside is empty sky and inside is
sunlit surface, so the brightness must rise inward. A vertex that lands on an edge running
the other way is discounted, which keeps a bright unrelated edge elsewhere in the frame
from outscoring the real limb.

**Levenberg-Marquardt refinement.** Starting from that guess, the fit is improved to
fractions of a pixel by repeatedly measuring how far each vertex sits from the nearest
real image edge and stepping the whole curve to reduce those distances.
Levenberg-Marquardt is the standard method for that kind of least-squares descent, and it
is what turns a whole-pixel guess into a sub-pixel answer. It is deliberately not allowed
to travel more than about one pixel from the starting guess, so it polishes the alignment
the coarse search chose rather than wandering off to a different one.

**Tukey biweight reweighting.** Some vertices will land on the wrong thing entirely: a
crater rim, a ring behind the moon, a second moon's edge. The Tukey biweight is a
weighting rule that progressively discounts vertices whose residuals are large and ignores
them altogether beyond a cutoff, so a handful of bad vertices cannot drag the answer. The
weights that survive also set the reported uncertainty, since a fit supported by many
well-behaved vertices is tighter than one supported by few.

The full algorithm, including how the uncertainty is derived, is documented in
:doc:`/dev_guide/dev_guide_techniques_dt_fitting`; how each technique turns its
diagnostics into a confidence is documented in
:doc:`/dev_guide/dev_guide_techniques_confidence`.

Star Techniques
---------------

All three star techniques work from ``STAR`` features, and all three ignore stars marked
as hidden behind a body or a ring.

StarFieldFromCatalogNav
^^^^^^^^^^^^^^^^^^^^^^^

This is the technique for a genuine star field, and it needs no idea in advance of how
large the pointing error is. It detects bright sources in the image, then recognizes the
pattern they form.

Recognition works on triangles. For every set of three detected sources, the technique
computes a description of the triangle, using only ratios of side lengths and one
angle. Such a description is unchanged by shifting, rotating, or uniformly scaling the
triangle, so it can be compared against the same description computed for triples of
catalog stars without knowing the offset first. A matching pair of triangles proposes a
shift; the technique counts how many other stars line up under that shift, and the
proposal with the most agreement wins. The surviving matches are then refitted with the
Tukey weighting described above, and the scatter of what remains becomes the reported
uncertainty.

At least three usable stars are needed, since below that no triangle exists. A strong
answer needs at least six stars in agreement. Below six there is a fallback that pairs
only the brightest few catalog stars against the brightest few detections and accepts
three in agreement, with the confidence capped well down.

This technique is the only one that can also measure camera roll, and it reports roll as
unmeasurable when the field is too small for a roll to be told apart from a shift.

After the matching has settled which detection is which star, each matched star's position
is measured again. A brightness-weighted center is unbiased but noisy; fitting the
camera's modeled point-spread function instead is the more precise estimator for a faint
star, but it carries a small fixed bias of its own. So the point-spread fit is applied to
a star only while its measured signal-to-noise ratio is below
``techniques.StarFieldFromCatalogNav.tuning.psf_refine_snr_max`` (30), above which the
simpler brightness-weighted center is already better than the fit's bias floor. Setting
``techniques.StarFieldFromCatalogNav.tuning.psf_refine_enabled`` to 0 uses the
brightness-weighted center everywhere. This step is what makes a well-exposed star field
the most accurate thing SpinDoctor can navigate on.

StarUniqueMatchNav
^^^^^^^^^^^^^^^^^^

Some frames have only one or two stars bright enough to see, and a pattern matcher has
nothing to work with. This technique handles exactly that case, and needs only one usable
star.

With one star, it requires that star to be clearly the brightest thing the catalog
predicts in the frame, by a margin over the next candidate of
``techniques.StarUniqueMatchNav.tuning.brightness_margin_to_next_catalog_star_mag``
magnitudes, 1.5 by default. Then the brightest peak inside a search window around the
prediction can only be that star, and the offset is simply the difference. Confidence is
capped at 0.7, because a single match has nothing to check itself against. Guards exist
against locking onto a noise spike or a hot pixel: the peak must stand clear of the
window's background, and where there is no rival peak at all to measure against, the match
is accepted only if it sits close to the predicted position.

With two stars, the technique tries both ways of assigning the two detections to the two
predictions and keeps the assignment with the smaller combined residual. That comparison
is a real check, so confidence is capped a little higher, at 0.8.

The search never covers the whole frame. It reaches
``techniques.StarUniqueMatchNav.tuning.search_window_px`` pixels in each direction from
each prediction, 30 by default, which is what makes this technique work on images where a
field-wide detection pass would find nothing.

StarRefineNav
^^^^^^^^^^^^^

This technique sharpens an answer that another technique already found, and runs only
after a first round has produced an offset. It shifts every predicted star position by
that offset, looks for a peak in a small window around each shifted prediction, measures
its center, and averages the leftover residuals, weighting each star by how precisely it
can be located. A star whose peak is too far from the shifted prediction is dropped as a
wrong identification.

What it reports is a correction to the offset it was given. With two or more stars it
reports the measured scatter as its uncertainty; with a single star it falls back to the
theoretical best and caps its confidence at 0.5, deliberately below what a single unique
match gets.

Body Techniques
---------------

BodyLimbNav
^^^^^^^^^^^

This technique fits a body's lit limb. It takes every ``LIMB_ARC`` feature in the frame,
puts all their vertices into one set, and solves for the single shift that best lays that
combined curve onto the brightness edges in the image, using the coarse search,
refinement, and reweighting described above. It is feasible when at least one limb arc has
30 or more surviving vertices, which is set by
``techniques.BodyLimbNav.tuning.min_arc_vertices``.

Fitting several bodies at once is better than fitting one. A single shift is solved for
all of them, and because the bodies are in fixed positions relative to one another, a
shift that suits one must suit them all. That makes it impossible for the fit to match a
moon onto the wrong moon, and the extra vertices tighten the answer.

This is the workhorse for encounter imaging, where a moon shows a clear illuminated edge:
the Cassini ISS views of Mimas, Enceladus, Tethys, Dione, and Rhea are the typical case.

One thing to expect from the reported uncertainty: a limb's predicted position carries
2.61 pixels of unavoidable model error, from surface relief, albedo markings, and shading
that a smooth predicted outline cannot capture. That figure is
``techniques.BodyLimbNav.tuning.model_error_floor_px``, and it is added to every limb
result, so an image navigated on a limb alone usually lands in the ``low`` confidence tier
however cleanly the curve fit. Corroboration from a star field or a disc correlation is
what lifts it.

Very low phase is where it struggles. At low phase the Sun is nearly behind the camera, so
almost the whole disc is lit and almost the whole limb is visible. That sounds like the
easy case, and for tracing the limb it is. But what an edge fit actually needs is a
*sharp* edge, and low phase is precisely where the limb is softest. Sunlight strikes a
sphere most obliquely at its outline, so the surface brightness falls smoothly toward
zero as it approaches the limb, and the disc fades into the sky over several pixels
instead of stepping down at one. At higher phase the sub-solar point lies near the limb,
the crescent stays bright right up to its edge, and the step is crisp.

A soft edge means a faint gradient, and a faint gradient means other edges in the frame
can score better in the coarse search than the true limb does. When that happens the
starting guess lands several pixels away, on a crater rim or a terminator or a ring behind
the disc, and the refinement is held near that wrong starting point rather than walking
back to the truth. The cost surface has more than one *basin*, meaning more than one shift
at which the curve fits the image locally well, and the fit settled in the wrong one.

The technique detects that situation by its signature: a fit that failed to settle and
ended pressed against the limit of how far it was allowed to move from its starting guess.
Such a result is marked spurious instead of being reported as a confident multi-pixel
offset, and the image is navigated from its disc or from another technique instead. A
healthy fit lands within about a pixel of its starting guess and settles, so it is never
mistaken for this failure.

BodyDiscCorrelateNav
^^^^^^^^^^^^^^^^^^^^

Rather than fitting the outline, this technique correlates the whole rendered disc against
the image. It consumes ``BODY_DISC`` features, so it runs on bodies that sit well inside
the frame with most of their lit side showing. Several bodies are painted into one
composite picture, nearer bodies drawn over farther ones as they appear in reality, and
one correlation is run against the whole composite.

The correlation is computed at a series of scales, from a heavily downsampled copy up to
full resolution. That is much cheaper than testing every shift at full resolution, and it
also yields a useful quality signal: if the winning position moves as the resolution
improves, the answer is genuinely less well localized, and the reported uncertainty is
widened accordingly. How far the position may move between scales before the answer is
called inconsistent is the larger of
``techniques.BodyDiscCorrelateNav.tuning.consistency_max_px``, 4 pixels by default, and
``techniques.BodyDiscCorrelateNav.tuning.consistency_max_fraction_of_diameter`` times the
body's diameter in pixels. The technique also chooses for itself, per image,
whether to correlate raw brightness or brightness gradients, running both and keeping the
more confident result. Raw brightness wins on a smooth shaded disc that fills the frame;
gradients win when only the edge carries distinctive information.

This is the technique that carries a near-fully-lit body, which is the case where the limb
fit is weakest.

BodyBlobNav
^^^^^^^^^^^

When a body is too small or too irregular to trace a limb on, all that can be measured is
where its light is centered. This technique measures a brightness-weighted center for each
``BODY_BLOB`` feature inside its predicted box, and solves for the single shift that best
maps the predicted centers onto the measured ones. With two or more bodies the problem is
over-determined, which makes it tolerant of a poor center on any one of them.

Its confidence is capped at 0.4 by design. A brightness center is much weaker evidence
than a fitted limb, so even a perfect blob match should not be allowed to outweigh a limb
or disc answer. Its role is to provide something where nothing better exists, and to act
as an independent cross-check that can catch a curve fit that locked onto the wrong
feature.

BodyTerminatorNav
^^^^^^^^^^^^^^^^^

The terminator is the day-night boundary running across a body's face, and on a crescent
or half-lit moon it is a long, well-defined brightness edge in its own right. This
technique fits ``TERMINATOR_ARC`` features the same way ``BodyLimbNav`` fits a limb, with
two differences.

The first is how the vertices are weighted. Along a limb, uncertainty varies vertex by
vertex, because the real surface departs from the smooth predicted outline by different
amounts in different places. A terminator's position instead depends on how brightly the
surface reflects, which is a property of the body as a whole, so every vertex of a given
body shares one weight derived from that body's average. Across bodies the weights still
differ: a dark surface gives a sharper terminator than a bright one, because a bright
surface scatters light past the boundary and blurs it.

The second is that the confidence accounts for how variegated the body's surface is. An
albedo boundary looks much like a terminator, so a body with strongly varying surface
brightness earns less confidence than a uniform one.

A terminator is a photometric feature rather than a geometric one: it marks where the
brightness crosses a threshold, not where the body's outline is. It can settle on the
wrong contour on a textured surface with nothing in the fit itself to reveal the mistake,
and measurements against known truth put its typical error at 5.2 pixels. The model error
added to every terminator result is accordingly
``techniques.BodyTerminatorNav.tuning.model_error_floor_px``, 4.32 pixels, which is large
enough that a terminator answer cannot carry an image by itself. It is treated as a
fallback and is set aside whenever a limb fit or a disc correlation on the same body
produced a result that was not flagged spurious. Its value is in covering a body that
offers nothing else.

Because it is prone to settling on the wrong contour, this technique also asks a question
the others do not: after converging, it scans the rest of the search window for a rival
position that fits nearly as well. When a rival's cost is less than
``techniques.BodyTerminatorNav.tuning.basin_cost_ratio_threshold`` times the winner's, 1.2
times by default, the answer is not distinctive and is marked spurious.

Ring Techniques
---------------

RingEdgeNav
^^^^^^^^^^^

This technique fits ``RING_EDGE`` features: it puts the vertices of every predicted edge
in the frame into one set and solves for the single shift that best lays them onto the
edges in the image, with the same coarse search, refinement, and reweighting the body
curve fits use.

One thing it deliberately does not use is **polarity**, defined above as knowing in
advance which side of an edge should be the brighter one. For a body's limb that knowledge
is trivial: sky outside is dark, the lit surface inside is bright, so any vertex where the
real image gets brighter in the wrong direction can be discarded as a wrong match. For a
ring edge it is not knowable from the catalog, because the same kind of edge can be the
outer boundary of a bright ringlet, with brightness inside and darkness outside, or the
inner boundary of a gap, with the opposite. Which one it is at a given illumination
depends on information the ring catalog does not record. So the ring fit matches the
curve's form alone, with one fewer way to tell a real edge from a plausible impostor than
the body fits have.

This is the technique for close-range ring images. For Saturn it receives a frame only
when the radial resolution is finer than 25 kilometers per pixel; everything coarser goes
to ``RingAnnulusNav``. Note the caution that applies at the fine end: as resolution
improves, the many similar concentric ringlet edges separate into distinct image edges,
and a fit that matches form alone can lock onto a neighbor of the edge it meant to match.
A ring-edge answer at that resolution that no other technique corroborates deserves a look
before you rely on it.

**Straight edges and unmeasured directions.** When every edge the technique was given is a
straight line, the answer is genuinely incomplete, and the reason is geometric. Sliding a
straight line along its own direction maps it exactly onto itself, so that motion changes
no residual and the image contains no information about it. A single straight edge
therefore measures only the offset perpendicular to itself. Edges at different
orientations between them would cover the plane, but the straight edges of a ring system
seen nearly edge-on are all parallel, so they never do.

This condition is called **rank deficiency**: the measurement determines the offset in one
direction and leaves the other completely open. SpinDoctor reports it rather than
inventing a number for the open direction. In the per-image metadata document you will see
it as ``sigma_along_unobservable_px`` under ``navigation_result``, which
:doc:`/user_guide/user_guide_metadata` describes, and as ``is_rank_1`` among this
technique's ``diagnostics``. An
image whose combined result is still rank-deficient after every technique has been
considered reaches the ``medium`` tier at best, and records a status reason of
``rank_1_only``.

An image in that state is not lost. Anything that measures the missing direction completes
it: a star field, a body limb, or even a body brightness center. The combination step
fuses a one-direction ring measurement with such a result to give a full answer.

RingAnnulusNav
^^^^^^^^^^^^^^

This technique matches the ring system's predicted *brightness* rather than the outline of
its edges. It takes the rendered picture that a ``RING_ANNULUS`` feature carries, slides
it over the real image, and scores each position by normalized cross-correlation. As with
the disc correlation, the search runs from a coarse downsampled copy up to full
resolution, which is both faster than a full-resolution sweep and informative: a winning
position that drifts between scales is less well localized, and the reported uncertainty
grows to say so, scaled by
``techniques.RingAnnulusNav.tuning.localization_uncertainty_scale``. The technique also
runs both a raw-brightness pass and a gradient pass and keeps the more confident of the
two, since a coarse view of Saturn's rings is mostly broad brightness variation while a
closer one is mostly sharp ringlet edges.

For Saturn this is the technique for anything at 25 kilometers per pixel or coarser, which
covers the great majority of ring imaging. The quantity actually being compared against
that threshold is the average radial resolution over the frame, one number per ring system
per image; at or above it, every edge of that system is folded into a single composite
picture. Below it, a frame can produce a mixture: individual ``RING_EDGE`` features for
the edges that stayed resolvable, plus one composite for those that compressed into a
narrow strip.

Matching brightness is what makes this the safer choice for the coarse regime. Relative
ring brightness is part of the comparison, so a broad dim C ring and a bright B ring
cannot be confused with each other, which is exactly the confusion a form-only fit is
prone to.

A scene containing two ring systems at once is rare but real, as when Cassini imaged
Jupiter and Saturn in one frame. Each system arrives as its own picture, they are painted
into one composite with the nearer system drawn over the farther, and a single correlation
is run against the whole thing. The fixed geometric relationship between the two removes
any possibility of matching one system onto the other.

One behavior can surprise you: a composite that comes out narrow carries too little
pattern to match reliably, so it is scored below the reliability minimum for its type and
set aside. The feature appears in the metadata document, marked as set aside, and this
technique never runs on that image.

Titan Technique
---------------

TitanHazeNav
^^^^^^^^^^^^

``TitanHazeNav`` performs the two measurements described under the Titan model above: it
finds the shift that makes the haze most nearly mirror-symmetric about the line through
the body center and the sub-solar point, and it fits a free-radius circle to the sunward
limb to place the shift along that line. It reads the image as it stands rather than the
detected edges the curve-fitting techniques use, and it consumes at most one haze feature
per frame.

It is the only technique for a hazy body, since there is no second way to measure such a
body's position, and that is why it is treated as a primary result rather than a fallback.

Manual Navigation
-----------------

For an image the autonomous pipeline cannot handle, manual navigation opens an interactive
window that composes every renderable prediction into one overlay and lets you place the
offset by hand. It is not part of the autonomous set and cannot be selected with
``--nav-techniques``. Reach it from ``sd_offset`` (see
:doc:`/user_guide/user_guide_navigation_running`) with the ``--manual`` flag, which
requires the image selection to resolve to exactly one image:

.. code-block:: bash

   echo W1521598221_1_CALIB > /tmp/img_list.txt
   sd_offset coiss --manual --image-file-list /tmp/img_list.txt

The program loads the image, builds the models, gathers and screens the features exactly
as an autonomous run would, and then opens the window instead of fitting. On acceptance it
reports the chosen offset and writes the same metadata document and summary picture an
autonomous run writes, under the navigation results root. The exit code is 2 if you cancel
or if the image has no renderable predictions to show.

.. _selecting-models-and-techniques:

Choosing Which Models and Techniques Run
========================================

By default ``sd_offset`` builds every model that applies to the scene and runs every
technique that has something to work with. Two options narrow that set: ``--nav-models``
selects which models are built, and ``--nav-techniques`` selects which techniques are
allowed to run. The same pattern syntax serves both, and the only difference is the names
it is matched against.

The patterns can be given in two places:

* On the command line, as ``sd_offset --nav-models LIST --nav-techniques LIST``.
* In a cloud task description, under ``data.arguments.nav_models`` and
  ``data.arguments.nav_techniques``, each a list of strings. Cloud tasks are a work queue
  supplied by the Ring-Moon Systems Node, and ``sd_offset_cloud_tasks`` is a worker that
  the queue runs on a cloud machine rather than a program you run yourself; see
  :doc:`/user_guide/user_guide_cloud_tasks`.

Both options only ever narrow. A pattern that matches nothing installed on your copy of
SpinDoctor simply selects nothing, and no model or technique can be added by naming it.

Pattern Syntax
--------------

A pattern is a shell-style glob matched against a candidate name. On the command line,
separate several patterns with commas inside one argument; in a task description, supply a
list of strings.

Inclusion patterns
^^^^^^^^^^^^^^^^^^

* A literal name matches that name alone: ``BodyLimbNav`` selects the body-limb technique
  and nothing else.
* ``*`` matches any run of characters, ``?`` matches one character, and ``[abc]`` matches
  any one character from the set.
* The default pattern is ``*``, which matches everything.

Exclusion patterns
^^^^^^^^^^^^^^^^^^

* A leading ``!`` makes a pattern an exclusion: anything it matches is removed.
  ``--nav-techniques '!StarFieldFromCatalogNav'`` runs every technique except that one.
* When every pattern in the list is an exclusion, an inclusion of ``*`` is assumed, so
  ``--nav-models '!body:MIMAS'`` means the same as ``--nav-models '*,!body:MIMAS'``.
* When at least one inclusion is present, only what the inclusions match survives, minus
  anything an exclusion matches. ``'body:*,!body:MIMAS'`` runs every body model except
  Mimas.

Combining patterns
^^^^^^^^^^^^^^^^^^

The order of the patterns does not matter. A name is kept when it matches at least one
inclusion pattern and no exclusion pattern.

.. code-block:: bash

   --nav-models 'body:MIMAS,rings:SATURN,stars'

.. code-block:: json

   ["body:MIMAS", "rings:SATURN", "stars"]

Model Names
-----------

Models are created fresh for each image, because what they can predict depends on what is
in the frame. That is why a model name can carry the subject it was built for:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Name
     - What it names
   * - ``stars``
     - Every catalog star predicted to be visible in the frame. There is one such
       model per image and it takes no subject, so the name is bare.
   * - ``body:NAME``
     - One solid body: its rendered disc, its lit limb, and its terminator. One such
       model is created for each body whose predicted outline overlaps the frame, and
       ``NAME`` is that body's SPICE name in capitals: ``body:MIMAS``, ``body:DIONE``,
       ``body:SATURN``.
   * - ``rings:PLANET``
     - One planet's ring system: its named edges and its predicted brightness. One
       such model is created for the planet nearest the line of sight, and only when
       SpinDoctor carries a catalog of ring edges for that planet. Only Saturn's
       catalog is populated, so ``rings:SATURN`` is the name you will see.
   * - ``titan:TITAN``
     - The haze envelope of a body whose atmosphere hides its surface, created
       whenever Titan is in the frame. On a simulated image the equivalent model is
       named ``titan_sim:TITAN``.

Two conveniences apply to model patterns. The part after the colon is capitalized for you,
so ``body:saturn`` matches ``body:SATURN``. And a bare prefix that has no colon and no
glob characters is expanded to cover everything under it, so ``--nav-models 'body'`` means
every body model and ``--nav-models '!body'`` excludes every body model. Both conveniences
keep a leading ``!``.

Technique Names
---------------

Each technique is named after the class that implements it. The techniques that ship are:

* **Body**: ``BodyDiscCorrelateNav``, ``BodyBlobNav``, ``BodyLimbNav``,
  ``BodyTerminatorNav``.
* **Ring**: ``RingAnnulusNav``, ``RingEdgeNav``.
* **Star**: ``StarFieldFromCatalogNav``, ``StarUniqueMatchNav``, ``StarRefineNav``.
* **Haze**: ``TitanHazeNav``.

Manual navigation has no such name and cannot be selected with ``--nav-techniques``; it is
reached with ``--manual``, as described above.

Every technique that has usable features runs, and their answers are then reconciled with
one another. ``--nav-techniques`` restricts the set of candidates rather than picking a
single winner.

Examples
--------

.. code-block:: bash

   # Run every model and every technique (the default).
   sd_offset coiss N1234567890

   # Mimas only: drop every other body, and the ring and star models.
   sd_offset coiss N1234567890 --nav-models 'body:MIMAS'

   # Every body, plus the rings, but no stars.
   sd_offset coiss N1234567890 --nav-models 'body:*,rings'

   # Every model except Mimas.
   sd_offset coiss N1234567890 --nav-models '!body:MIMAS'

   # Two specific curve-fitting techniques.
   sd_offset nhlorri LOR_0034851733 \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

   # Every technique except the star pattern matcher.
   sd_offset coiss N1234567890 \
       --nav-techniques '!StarFieldFromCatalogNav'

   # The body and ring families only.
   sd_offset coiss N1234567890 \
       --nav-techniques 'Body*,Ring*'

What a Run Reports
==================

Each navigated image leaves behind a metadata document and, when the navigation reached a
result, an annotated summary picture. Both are described in
:doc:`/user_guide/user_guide_navigation_outputs`.

The metadata document records the final offset for the image, its uncertainty, its
confidence, and its confidence tier. Alongside those it records one entry per technique
that ran, giving that technique's own offset, uncertainty, and confidence, together with
the diagnostic quantities its confidence was computed from. It also lists every feature
the models emitted, including the ones set aside before fitting and the reason each was
set aside. For the complete key-by-key description see
:doc:`/user_guide/user_guide_metadata`.

The summary picture shows the image with every model prediction drawn at the fitted
offset, so one glance tells you whether the predictions landed on the real features.
