==========================================================
Ring Edge Fit (RingEdgeNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` recovers a single translation
from one or more ring-edge polylines by aligning each polyline against the image's
edge-distance-transform. The technique consumes every offered
:data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` feature, weights its vertices by
the prior precision derived from the per-edge sigma, seeds the fit with an integer search
checked against the strength of the image edges the model lands on (see
`The coarse seed`_), and runs the Tukey-reweighted Levenberg-Marquardt refinement that
:doc:`dev_guide_techniques_dt_fitting` describes.

When every consumed ring edge is flagged straight-line the combined Jacobian is rank-
deficient — all parallel ring edges share a single ring-plane normal so the along-edge axis
is unobservable. The technique projects the returned covariance to be *exactly* rank-1
(see below); the orchestrator's ensemble combine fuses it with any orthogonal-axis result
(a star, body limb, body blob) before declaring a final answer, and when no orthogonal
information exists the fused result carries
:attr:`~spindoctor.support.status_reason.NavStatusReason.RANK_1_ONLY` with the offset
reported as the minimum-norm representative along the edge and the unobservable axis
surfaced through
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`.

Feasibility passes when at least one offered ``RING_EDGE`` has a non-empty polyline. A single
non-empty edge is sufficient — even an all-flat scene produces a useful rank-1 constraint.
Feasibility fails when every offered ``RING_EDGE`` is empty (a ring system entirely outside the
extended FOV or below the per-pixel resolution threshold).

Which scenes reach this technique is decided upstream, by the rings model's per-planet
km/px threshold (``feature_emission.ring_annulus.planets.SATURN.kmpp_threshold``): for
Saturn, only scenes finer than 25 km/px radial resolution emit ``RING_EDGE`` features at
all, and everything at or above that routes to the
:doc:`annulus composite <dev_guide_techniques_ring_annulus>` instead. The routing is
measured, not geometric. A 131-frame clean-truth head-to-head put this technique's
wrong-when-accepted rate at 5% / 13% / 56% / 100% in the 300-1000 / 100-300 / 25-100 /
0-25 km/px bands while the annulus fit was wrong on zero accepted answers at every band.
The annulus route covers the regimes that comparison validates -- everything at or above
25 km/px. The degradation toward fine resolution is the ring alias lattice: as resolution
improves, many similar concentric ringlet edges separate into distinct image edges, and a
distance-transform fit against edge shape alone can lock onto the wrong one, while the
annulus's rendered-brightness template disambiguates the lattice because relative ring
brightness is part of the match. The 0-25 km/px band's own numbers set the boundary
rather than claim superiority for either side there: its trustworthy evidence is three
frames on which neither technique is validated -- the edge path re-locked catastrophically
on all three (the 100% above), the annulus near-missed twice and failed once -- so the
threshold marks the bottom of the measured regime, not a regime where this technique is
known good. Those rates were measured with the mask-overlap seed alone; the edge-strength
check described in `The coarse seed`_ addresses the lattice locks behind them, and the
remaining sub-25 km/px exposure is described under `Restrictions and assumptions`_.

Theory
======

The technique fits a per-image translation by minimizing the weighted squared distance from
the model ring-edge polylines to the image edges, exactly as the limb fit does — see
:doc:`dev_guide_techniques_dt_fitting` for the cost function, the LM mechanics, and the
Tukey biweight (the polarity filter is disabled for ring edges; see below).

The coarse seed
---------------

The refinement polishes whatever basin the integer seed puts it in: the trust region keeps
it within about a pixel of the seed. Choosing the basin is therefore the coarse search's
job, and for rings the mask-overlap score the other DT techniques started from is the wrong
score. It counts a model vertex as matched whenever it lands on any detected edge pixel. A
ring's interior is full of detected edges -- density waves, ringlets, and the texture of
the material itself -- and so is a smooth scattered-light glow once its faint contours are
thresholded. Those edges run parallel to the modeled edge, so translating the model across
them keeps nearly every vertex on *some* edge. At fine radial resolution the overlap score
is flat across a lattice of such aliases, and the search returns whichever one its
tie-break or noise favors. Orientation does not separate them, because ring texture is
concentric with the cataloged edge. Polarity does not either, which is why a seed weighted
by model-predicted polarity moved none of the affected frames.

What separates the cataloged edge from the texture is how strong a brightness step it is.
The technique therefore checks the overlap seed against edge strength. The edge-strength
score of a shift
(:func:`~spindoctor.nav_technique.dt_fitting.edge_strength_score`) sums, over the model
vertices that land on a detected edge pixel, :math:`|g \cdot n|`, the image gradient there
projected onto the vertex normal, and divides by the vertex count. The absolute value keeps
it polarity-blind, the projection discounts an edge that crosses the model at an angle, and
the magnitude lets one strong step outweigh any number of faint texture edges.
:func:`~spindoctor.nav_technique.dt_fitting.coarse_edge_strength_search_scored` finds the
shift that maximizes it. The overlap seed is kept when its own edge-strength score is at
least ``coarse_strength_keep_ratio`` of that maximum, and the strongest alignment seeds
the fit otherwise. Edges within ``coarse_sensor_margin_px`` of the sensor boundary are left
out of both scores: the step from the image onto the zero-padded margin of the extended
frame is usually the strongest edge present, and a ring edge running near the frame border
would otherwise lock onto it. The score reported to the coarse-acquisition gate is the
plain match fraction at whichever shift was chosen.

Keeping the overlap seed whenever it sits on comparably strong edges is deliberate. On a
straight or nearly straight edge the strength score is almost flat along the edge, so its
maximum lands at an arbitrary position along it, while the overlap search's tie-break
returns the position nearest the prediction. Where both agree on the basin across the
edge, replacing one with the other would move only the unobservable along-edge component,
and on a frame that is only weakly curved that component is what the reported answer
carries. The check changes the seed only where the overlap seed is clearly on weaker
edges, which is the alias case.

Measured on Cassini NAC frames, the edge strength under the overlap seed was 0.91 to 1.00
of the strongest wherever that seed was right across the edge (the library's Keeler gap,
Encke gap, and Maxwell ringlet frames at 0.2 to 0.4 km/px, and two F ring movie frames at
5.5 km/px), and 0.005 to 0.68 wherever it had locked onto ring texture or scattered-light
contours 16 to 94 px off. On the latter frames the strongest alignment put the model within
about 2 px of independent pointing across the edge.

A frame can also hold no cataloged edge at all at the true pointing: the edges lie in the
extended margin outside the sensor, and the search drags the model onto whatever the
thresholded image offers inside it, which in an F ring movie frame is the faint contour
lattice of the scattered-light glow. No seed can be right there. The fit is instead
rejected after convergence when the edges it matched are too faint to be a ring edge: the
median of :math:`|g \cdot n|` over the vertices within 1.5 px of a detected edge, divided
by the 99.9th percentile of the gradient magnitude inside the sensor, must reach
``spurious_min_edge_strength_fraction``. The check lays the model where the converged fit
placed it: when the camera rotation is fit, the vertices and their normals are first
turned by the fitted angle about the vertex centroid, then shifted by the fitted offset.
Ten such F ring movie frames at 9.9 km/px, each wrong by 60 to 94 px, scored 0.0012 to
0.0022; fits on real ring edges scored from 0.0096 (F ring movie frames) and from 0.050
(the library's ring frames) up to about 1.

The residual scale
------------------

The robust fit needs a scale for its residuals, and that residual is the distance from a
model vertex to the nearest *image* edge -- so its scale carries two uncertainties, not
one. The first is the catalog's: each vertex's ``sigma_radial_per_vertex_px`` is the ring
edge's radial uncertainty in km divided by the frame's radial km/px. The second is the
evidence's: the fit measures against a *binary* edge mask, whose distance transform is
quantized to the integer pixel grid, so half a pixel is the floor on how precisely that
mask can place an edge at all.

The two are on very different footings for rings. Ring orbits are solved to a fraction of
a km while a Cassini WAC pixel spans of order 100 km, so the catalog term routinely lands
at 0.001 px -- a hundred times finer than the evidence can resolve. Used alone it is not
a residual scale but a claim the measurement cannot support, and the consequences are
mechanical: the Tukey redescender's rejection radius becomes
:math:`4.685 \times 0.001 = 0.005` px, so the only vertices it keeps are the ones sitting
exactly on a mask pixel where the DT is zero. That set is reachable only at an integer
offset, so the fit never leaves its integer coarse seed, the gradient-ridge pass finds
no vertex it is allowed to move, the reported sigma comes back in the thousandths of a
pixel, and the inlier fraction the spurious gate reads measures sub-pixel phase rather
than model agreement.

The technique therefore combines the two in quadrature before fitting: the per-vertex
scale is :math:`\sqrt{\sigma_{\mathrm{catalog}}^{2} + \sigma_{\mathrm{edge}}^{2}}` with
:math:`\sigma_{\mathrm{edge}}` the configured ``edge_localization_sigma_px``. Where the
catalog sigma already exceeds the mask scale the term is nearly inert; where it falls far
below, the scale becomes the mask's own and the fit measures what the image can actually
tell it. Only ``RingEdgeNav`` declares the term: ``BodyLimbNav`` and
``BodyTerminatorNav`` derive their per-vertex sigmas from body shape and pole
uncertainty, which run from about half a pixel to three pixels on real frames and so
already sit above what the mask resolves.

Rank-deficient covariance
-------------------------

Ring edges differ from limbs in that each ring edge is only locally observable along its
*radial* direction (orthogonal to the edge tangent). Motion along the tangent of a single
ring edge produces no DT cost change *in the interior*; the finite polyline's end vertices
do respond to tangential motion (they track wherever the detected edge enters and leaves
the frame), so the raw LM covariance of an all-straight scene often comes back numerically
tight along the tangent — a false constraint. The technique therefore enforces the rank-1
contract explicitly: when every consumed edge is straight (or the raw covariance already
fails the relative-eigenvalue rank test), the translation covariance is rebuilt as the
exactly singular ``sigma_n^2 n n^T``, where ``n`` is the dominant eigenvector of the
per-vertex normals' outer-product sum (polarity-sign-independent) and ``sigma_n^2`` the raw
covariance's marginal variance along it.

The per-vertex normals of a straight edge are replaced by the line's own normal before
any of this. The rings model takes each vertex's normal from which of its pixel neighbors
lie off the edge, so a model normal can only point along one of eight directions, the
multiples of 45 degrees. On a straight edge at any other orientation every one of them is
off by up to 22.5 degrees, and an axis built from them puts the ensemble's minimum-norm
answer on the wrong line: the part of the free slide along the edge that the tilted axis
does not cancel leaks into the reported offset. The line's direction is the principal
axis of its vertices, and the normal perpendicular to it is used for every vertex, signed
the way the model's normals summed. On the library's Keeler gap frames (edges at about 68
and 51 degrees, reported at 90 and 45) this moved the ensemble's planted-shift round trip
across the edge from a median 0.39 px (worst 17 px) to 0.04 px (worst 0.31 px). Exact
singularity is the representation the ensemble is built around: ``pinvh`` keeps the
normal-axis measurement when forming the information matrix, the combine's rank-deficiency
test fires, and the unobservable axis is reported through the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`
sentinel rather than an inflated per-axis sigma.

Multi-edge inputs at different orbital radii share the same ring-plane normal but sample
different points around the projected ring; the joint information matrix becomes full-rank
when at least two non-parallel edges contribute. The technique reports an
:attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag that the curator
surfaces; downstream consumers (the ensemble combine, the operator's per-image log) can
distinguish a rank-1 ring-edge result from a full-rank one.

Radial orbit-uncertainty channel
--------------------------------

The M-estimator covariance measures the statistical lock onto the *modeled* annulus. When
the catalog orbit itself is uncertain, the whole predicted edge can sit radially displaced
from the real one, and the robust fit absorbs that displacement into the recovered
translation: the Tukey weights keep the arc that happens to align, the fit locks onto one
side of the annulus, and the residuals stay clean — a tight covariance around a biased
offset. No per-vertex sigma can price this: a per-vertex prior is a statistical scale
that averages down as :math:`1/\sqrt{N}` over the polyline, while an orbit error displaces
every vertex coherently and does not average down at all.

The technique therefore consumes the per-feature ``sigma_orbit_radial_px`` carried on the
:class:`~spindoctor.feature.geometry.RingEdgePolyline` geometry (that edge's own catalog
orbit-solution uncertainty, converted to pixels at the edge's own radial scale by the
emitting model) and adds it to the reported covariance through the translation such a
displacement would actually be absorbed into.

A coherent displacement :math:`d` moves every vertex along its own outward normal, so the
translation the fit converges to is the weighted least-squares solution of
:math:`M t = d\,b` with :math:`M = \sum_i w_i n_i n_i^{T}` and :math:`b = \sum_i w_i n_i`.
Writing :math:`g = M^{+} b`, the added covariance term is

.. math::

    \Sigma \mathrel{+}= \sigma_{\mathrm{orbit}}^{2}
        \bigl[\, g g^{T} + \bigl(1 - \lVert g \rVert^{2}\bigr)\, I \,\bigr]

with the isotropic complement clamped at zero. The limits are the point of the
construction: a short arc gives :math:`\lVert g \rVert \approx 1` along its radial axis; a
straight (rank-1) edge gives :math:`g = n` exactly, so the projected covariance stays
exactly singular along the tangent; a half ring gives :math:`4/\pi \approx 1.27` (one
translation overshoots the middle of an arc to reduce the error at its ends); and a closed
annulus gives :math:`\lVert g \rVert \approx 0`, because a uniform radial error dilates a
closed ring rather than translating it.

**What the isotropic term is for.** A small :math:`\lVert g \rVert` says the *linearized*
fit absorbs little, not that the answer is safe: the acquisition is nonlinear, and the
coarse integer search can still select a basin whose translation aligns a long arc of a
radially misplaced ring. The simulated closed-ringlet scene does exactly that. Since the
direction it locks in is precisely what cannot be predicted, the bound is reported on every
axis instead of on an axis chosen by rounding.

**What this changes in practice.** For :math:`\lVert g \rVert \le 1` the added term's major
eigenvalue is exactly :math:`\sigma_{\mathrm{orbit}}^{2}` whatever direction :math:`g`
points, and only the minor eigenvalue depends on the geometry. Because the tier threshold reads
:math:`\max(\sigma_{dv}, \sigma_{du})`, within that regime the derived *direction* cannot
by itself change a tier outcome -- the behavioral change against a plain directional
inflation is the isotropic floor it puts under the minor axis, which is what stops a
demotable frame slipping through on an un-widened perpendicular axis. The derived
*magnitude* does move the major axis once :math:`\lVert g \rVert` exceeds 1.

This construction requires the emitted normals to carry a consistent outward-radial sense,
which the catalog model guarantees by signing each normal against the ring-radius
backplane. Preserving the relative senses is what makes the dilation and opposite-side
cancellations real rather than fabricated.

When several features fuse, the effective :math:`\sigma_{\mathrm{orbit}}` is the
weight-weighted mean of the per-feature sigmas, deliberately treating the features' orbit
errors as fully correlated (the common multi-edge case is the inner and outer edge of one
feature, whose orbit error is genuinely shared). That combine is conservative only for
same-sense geometry; for features on opposite radial sides of the planet a common error is
a dilation the fit largely does not absorb, and the geometry then self-limits through
:math:`\lVert g \rVert`.

**The severity is operator-tunable.** Treating the catalog RMS as a fully coherent
whole-edge displacement is a deliberately conservative assumption, not a measurement: the
RMS is an orbit-fit residual that also contains longitude-varying resonant wander, which
does not displace an edge coherently. ``rings.orbit_radial_sigma_correlated_fraction``
(default ``1.0``) scales the coherent term, so the assumption can be ratcheted without a
code change while the principled fix -- decomposing the catalog modes and pricing only the
m=0 part coherently -- remains unimplemented.

The ensemble consumes the widened covariance through its ordinary precision-weighted
machinery — nothing downstream special-cases the term. A ring-edge lock on an uncertain
orbit then carries an honestly wide radial axis: it fuses at reduced radial weight against
any technique that constrains that axis independently, and the sigma-gated confidence tier
demotes on its own when the widened axis exceeds a tier's sigma cap. The effective sigma
is recorded on
:attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.sigma_orbit_radial_px`.
The channel prices the hazard; it cannot remove the underlying offset bias, which is why
the simulator's planted-orbit-error scene keeps its measured-error pin (see the planted
orbit error section of :doc:`dev_guide_simulator`).

Fit-quality gates
-----------------

The technique also feeds the shared DT fit-quality gates (LM-convergence demotion and the
coarse-acquisition-quality gate; the polarity gates are inert here because the fit runs
polarity-free). The gates and their thresholds are documented at
:doc:`dev_guide_techniques_dt_fitting`.

Restrictions and assumptions
----------------------------

- The orchestrator must populate
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` and
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext`; in their
  absence the technique cannot evaluate the cost.
- Polarity filtering is entirely disabled for ring edges (the fit runs with
  ``use_polarity=False``): predicting a ring edge's gradient polarity depends on lighting
  and gap-vs-ringlet context that the ring catalog does not encode today, so every vertex
  participates regardless of its local gradient direction.
- Ring edges that have collapsed to a sub-pixel radial extent are emitted by the upstream
  model as ``RING_ANNULUS`` templates instead, so the technique never sees them. See
  :doc:`dev_guide_techniques_ring_annulus`.
- The fit assumes the per-image SPICE pose is good enough that the integer seed lands
  in the basin of attraction of the correct ring edge. When the SPICE pointing error
  exceeds the per-instrument search-window margin the seed lands on a wrong edge and the
  LM converges to a wrong local minimum.
- The edge-strength check assumes the cataloged edges are the strongest steps near the
  prediction. Against the published F ring bundle's per-frame pointing on 1,041 held-out
  frames below 25 km/px, it cuts the alias locks across the edge, answers displaced across
  the edge by a distance that matches no spacing between cataloged edges, from 188 to 4:
  N1492091385 at 45 px, N1851236541 at 44 px, N1492064782 at 34 px, and N1851226701 at
  3.8 px. When two cataloged edges lie a few tens of pixels apart, one strong step can be
  matched by the wrong one of them: on Cassini NAC F ring movie frames at 5 km/px the
  strongest alignment sometimes lays the Keeler gap edges on the A ring's outer edge. On
  the same frames 5 answers are wrong this way, 7 to 49 px across the edge (41 to 51 px on
  four of them). All nine are reported at the ``medium`` tier.
- An edge that is nearly straight across the frame but not flagged straight constrains the
  along-edge component only through its slight curvature, and the reported covariance does
  not show how weak that constraint is. On the same 1,041 held-out frames, 115 answers
  carried by this technique are within 1.4 px of the published pointing across the edge
  and more than 2 px off along it: a median of 3.0 px, a 90th percentile of 12 px, and
  36 px at worst. Of those, 113 are reported at the ``medium`` tier, and 107 report a
  sigma under 1 px on both axes (a median of 0.28 px on the larger one). The rest of the
  129 answers more than 2 px off on those frames are the nine locks of the previous item
  and 5 answers carried by the star techniques.
- A curved edge keeps the model's quantized per-vertex normals. They enter the
  edge-strength weights and the orbit channel, where a 22.5-degree error costs little;
  only the straight-edge rank-1 axis takes the line's own normal.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence; the rank-1 case is projected to an exactly singular covariance whose
only non-null axis is the aggregate edge normal (see above). When the converged offset sits
within the at-edge tolerance of any axis bound,
or when the rotation parameter is at the configured fraction of its cap, the result is
flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.RingEdgeNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the
  bilinear-DT half-cell width.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many radial-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``3.0`` px. Floor of the spurious-detection
  threshold; the threshold is the larger of the floor and the per-feature sigma multiple.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``spurious_min_inlier_fraction`` — float, default ``0.5`` (dimensionless). A Tukey inlier
  fraction (inliers over aggregated model vertices) below this marks the result spurious.
  It catches a fit that explains almost none of its model: one anchored on a handful of
  vertices while the rest sit nowhere near a detected edge.

  It does not separate a correct fit from one locked onto the wrong member of a
  concentric family. Measured over 71 Cassini B-ring frames with independently published
  pointing, with the inlier fractions computed under the corrected residual scale
  described in `The residual scale`_: fits landing within 2 px of that pointing and fits
  missing it by more than 5 px both run inlier fractions from roughly 0.3 to 0.95,
  overlapping across the whole range: an alignment one ringlet spacing away still puts
  most of the model on *an* image edge, which is what makes the family aliased in the
  first place. The truth source for that measurement is bundle-published pointing
  predating the pointing-defect audit, which supports right-versus-wrong classification
  at the couple-of-pixel level only (about a fifth of unfiltered bundle pointing was
  later found unrefined); the overlap conclusion survives that coarseness. Distinguishing
  the two is an acquisition problem, not something the converged fit's residuals record.
- **Absent-edge waiver.** In a multi-edge fusion, a low aggregate inlier fraction can be
  fully explained by an edge that is *absent* from the image (a faint edge nothing in the
  frame can match) rather than *misaligned* (a wrong-ring lock). The two are separable by
  the per-edge median DT residual: an absent edge sits far from every detected image edge
  (a large median), while a wrong-lock leaves its rejected vertices lying *on* a detected
  edge they disagree with (a near-zero median). The test is therefore waived — the fit is
  kept, not flagged spurious — only when all of: at least ``spurious_waiver_min_well_fit_edges``
  edges each independently clear ``spurious_min_inlier_fraction`` and ``spurious_min_inliers``
  on their own vertices (so the surviving edges genuinely constrain the offset); every
  non-well-fit edge has a per-edge median DT residual of at least
  ``spurious_waiver_absent_median_px`` (absent, not misaligned); the translation covariance
  is full-rank; and at least two edges were consumed (a single-edge fit is never waived).
  A waived fit receives a sigma floor added in quadrature so it lands at the ``'low'`` tier
  and cannot outweigh a full-support result.
- ``spurious_waiver_min_well_fit_edges`` — int, default ``1`` (count). Minimum number of
  edges that must each independently meet the inlier-fraction and inlier-count minimums for
  the absent-edge waiver to apply.
- ``spurious_waiver_absent_median_px`` — float, default ``5.0`` px. A non-well-fit edge
  whose per-edge median DT residual is at least this large counts as *absent* (waivable)
  rather than *misaligned* (a genuine mis-convergence).
- ``spurious_waiver_sigma_floor_px`` — float, default ``3.0`` px. Sigma floor added in
  quadrature to the translation covariance whenever the absent-edge waiver fires. A waived
  fit is carried by a minority of its model in a scene whose parallel ringlet structure
  can offer alias alignments the DT cannot rule out; the floor sits above the medium
  tier's 2 px sigma cap so a waived fit surfaces as a low-tier result and cannot outweigh
  a full-support fit in the ensemble.
- ``edge_localization_sigma_px`` — float, default ``0.5`` px. Localization sigma of the
  image edge the fit measures against, combined in quadrature with each vertex's catalog
  sigma to form the residual scale the robust fit uses. Half a pixel is the half-cell of
  the binary edge mask's own quantization. See `The residual scale`_ for why a ring's
  catalog sigma alone is not a usable scale. Must be a finite number greater than zero;
  construction raises otherwise.
- ``spurious_max_lm_displacement_px`` — float, default ``4.0`` px. If the LM moves more
  than this many pixels from the integer coarse seed, flag spurious. Defensive: with
  the trust region below the LM cannot leave the coarse basin, so this guard normally
  never fires; it catches any future regression that bypasses the trust region.
- ``lm_trust_region_px`` — float, default ``1.0`` px. Maximum LM displacement from the
  integer coarse seed; the LM rejects any trial step that would land outside this
  radius. See :doc:`dev_guide_techniques_dt_fitting`.
- ``lm_tikhonov_alpha`` — float, default ``0.0`` (dimensionless). Tikhonov anchor strength
  toward the coarse seed; 0 disables the anchor (the trust region is the harder
  bound).
- ``gradient_ridge_refine`` — int flag, default ``1`` (ON). Final continuous
  gradient-ridge sub-pixel refinement after the DT LM converges. The binary edge mask
  quantizes detected edges to the integer pixel grid, so on dense real ring scenes many
  model vertices land exactly on edge pixels, where the DT is zero and carries no
  gradient to step along; the continuous pass refines against the un-thresholded gradient
  magnitude, recovering the sub-pixel offset the quantized DT discards. The pass shares
  the fit's residual scale, so it moves nothing while that scale is below the mask's own
  localization -- see `The residual scale`_.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses
  this fraction of the per-image
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.
- ``coarse_strength_keep_ratio`` — float, default ``0.8``. The overlap seed is kept when
  the edge-strength score under it is at least this fraction of the strongest alignment in
  the search window; otherwise the strongest alignment seeds the fit (see
  `The coarse seed`_). Must lie in ``[0, 1]``; construction raises otherwise.
- ``coarse_sensor_margin_px`` — int, default ``5`` px. Detected edges within this many
  pixels of the sensor boundary are left out of the edge-strength scores (see
  `The coarse seed`_). Five pixels covers the reach of the 1.2 px gradient smoothing into
  the frame. Must be a non-negative integer; construction raises otherwise.
- ``spurious_min_edge_strength_fraction`` — float, default ``0.005``. A converged fit
  whose matched edges have a median projected gradient below this fraction of the frame's
  99.9th-percentile gradient magnitude is marked spurious (see `The coarse seed`_). Must
  lie in ``[0, 1]``; construction raises otherwise.
- ``lm_unconverged_confidence_cap``, ``spurious_max_polarity_rejection_fraction``,
  ``spurious_unconverged_polarity_rejection_fraction``,
  ``spurious_min_coarse_peak_fraction`` — the shared DT fit-quality gate thresholds,
  documented with their rationale at :doc:`dev_guide_techniques_dt_fitting`. The two
  polarity thresholds are inert for this technique (the fit runs polarity-free).

Per-instrument overrides
------------------------

The keys above are global; per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override any of them.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. The formula spec is
``techniques.RingEdgeNav`` and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px` —
  alpha = 0.976, offset = 0.0, divisor = 1500.0, cap at 1.0. Cumulative pixel length of all
  surviving ring-edge polylines. More polyline earns confidence up to a 1500-pixel
  saturation point (calibration campaign raw p5/p50/p95 = 509/758/2140; inclined
  projections lengthen the closed-ellipse edges).
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_mean` —
  alpha = -0.069, offset = 0.0, divisor = 1.0, no cap. Mean per-edge final DT RMS value;
  the mean rather than the raw sum because the sum scales with the number of fused edges,
  so a fixed divisor would penalize a frame purely for having more rings. The sim
  calibration fit gives the term a modest negative weight (mis-locked edge-wave and
  m-mode shapes leave a visible residual), but most of the campaign's failure mass is
  clean-residual wrong-feature locks — aliasing, and planted orbit errors the fit absorbs
  into the offset — that the residual cannot see, so the discrimination rides on edge
  length and the spurious gates.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
firing forces confidence to zero. The constant baseline is :math:`\alpha_{0} = 1.832`. No
post-sigmoid ``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_ring_edge.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.
- ``src/spindoctor/nav_technique/dt_fitting.py`` — shared coarse-search and LM-refinement helpers;
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` — image-side derivatives;
  documented at :doc:`dev_guide_techniques_image_derivatives`.
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.name` — ``'RingEdgeNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.accepts_feature_types` —
  ``frozenset({RING_EDGE})``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.requires_prior` — ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.confidence_attributes` —
  ``{'at_edge', 'total_edge_length_px', 'per_edge_dt_rms_summed', 'per_edge_dt_rms_mean',
  'per_edge_dt_median_max', 'edge_count', 'is_rank_1'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px` —
  cumulative pixel length of all surviving ring-edge polylines. Consumed by the confidence
  formula.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_summed` — sum
  of per-edge final DT RMS values.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_mean` — mean
  per-edge final DT RMS value (the sum divided by the edge count). Edge-count independent,
  so it — not the raw sum — is the scale the confidence formula consumes.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_median_max` —
  largest per-edge median absolute DT residual (px). The mis-convergence gate statistic,
  and the waiver's discriminator: an edge absent from the image drives its own median to
  the tens-of-pixel scale (nothing detected nearby), while a wrong-ring lock leaves at
  least one rejected edge with a near-zero median — it sits on a detected edge it
  disagrees with — so the waiver stands down and the veto holds.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.edge_count` — number of
  ``RING_EDGE`` features fused.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` — True when every
  consumed ring-edge feature was straight-line and the combined covariance is rank-1.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.navigate`:

1. Open a logged section. Fail fast (:exc:`RuntimeError`) if either
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing.
2. Filter the offered features to ``RING_EDGE`` entries with non-empty polylines and
   concatenate the per-feature vertex / normal / sigma arrays.
3. Pull the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs` and erode the
   sensor mask by ``coarse_sensor_margin_px``. Run
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search_scored` for the overlap
   seed and :func:`~spindoctor.nav_technique.dt_fitting.coarse_edge_strength_search_scored`
   for the strongest alignment, and keep the overlap seed unless its edge-strength score
   falls below ``coarse_strength_keep_ratio`` of the strongest.
4. Read :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` and set the
   rotation pivot to the centroid of the concatenated vertices when rotation is fit.
5. Call :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` and capture the converged
   :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult`.
6. Detect the rank-1 condition: every consumed ``RING_EDGE`` flagged
   ``is_straight_line`` collapses the joint Jacobian to rank-1. The flag is recorded in
   :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` so downstream
   consumers can branch on it.
7. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block returned by
     :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` (rank-1 when every edge is
     straight; full-rank otherwise).
   - **Rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (3, 3) translation + rotation information matrix.

8. Apply the at-edge tests and the spurious tests on RMS / inlier-count, and the floor on
   the strength of the matched edges relative to the frame's strongest.
9. Build a :class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`, evaluate the
   confidence spec, log the breakdown, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A curved Saturn-ring scene with no body in FOV. The rings model emits multiple
    ``RING_EDGE`` polylines (the F ring, the A-ring outer edge, etc.) at different orbital
    radii; the curvature gives the joint Jacobian non-degenerate column rank.
    :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` converges to a 2-D
    translation against the operator-verified offset
    :math:`(\Delta v, \Delta u) = (5.85, 3.55)` px. The
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag is False on
    this scene (the curvature lifts the rank-deficiency) and the orchestrator's ensemble
    combine fuses the result without needing an orthogonal-axis cross-check.

``ring_only_curved`` (Cassini ISS NAC, image ``N1492091163_1``)
    A high-curvature single-edge scene with no other ring features in the FOV. The rings
    model emits one ``RING_EDGE`` polyline whose curvature lifts the rank-1 degeneracy.
    :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` converges to
    :math:`(\Delta v, \Delta u) = (5.35, -25.32)` px against an operator-verified ground
    truth of ``(4.92, -24.32)`` px, 1.1 px away and about the 1 px uncertainty of that truth.
    A two-star ``StarRefineNav`` then moves the fused answer 0.7 px in :math:`u`, to
    ``(5.21, -26.02)`` px. The technique reports a sub-pixel final DT residual; the
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag is False
    because the per-edge curvature exceeds
    :data:`~spindoctor.nav_model.nav_model_rings.FLAT_CURVATURE_THRESHOLD_PX`.

``ring_only_flat`` (Saturn ansa scene class, no body in FOV)
    An edge-on or extreme-grazing ring view in which every surviving polyline is flagged
    :attr:`~spindoctor.feature.flags.RingEdgeFlags.is_straight_line`. The joint-Jacobian column
    rank collapses to 1 along the shared radial direction; the technique returns the
    rank-1 pseudo-inverse covariance described in the Theory section, sets
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` True, and reports
    a calibrated confidence drawn down by the missing along-edge constraint. The
    orchestrator's ensemble combine treats the result as a one-axis observation and
    relies on a cross-feature (a body limb arc, a star prediction) to constrain the
    orthogonal axis. When no cross-feature is available the orchestrator surfaces the
    rank-1 result with
    :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.confidence` capped by
    the rank-1 confidence formula.
