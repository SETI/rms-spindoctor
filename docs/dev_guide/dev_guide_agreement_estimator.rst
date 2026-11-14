=====================================================
Agreement Estimator (Per-Technique Covariance)
=====================================================

Overview
========

When two or more navigation techniques measure the same frame's pointing
offset, their disagreement carries information about each technique's own
error. The agreement estimator turns that information into per-technique
error *covariances*: given per-frame offsets from several techniques, every
pairwise difference obeys

.. code-block:: text

    Cov(o_i - o_j) = C_i + C_j          (independent errors)

and with enough distinct pairs the individual ``C_i`` separate -- the
generalized "three-cornered hat". This is the machinery that lets the
cross-technique agreement study report a per-technique accuracy number on
real frames, where no external ground truth exists.

The estimator lives in ``util/agreement/estimator.py`` (measurement-layer
tooling, not part of the distributed package), with the validation campaign
around it in the same directory and its dated record in
``util/agreement/CAMPAIGN_20260719.md``.

Design: full matrix form, never scalar sigmas
=============================================

A limb fit's error is strongly anisotropic -- tight perpendicular to the
visible arc, loose along it -- and the arc orientation rotates from frame
to frame. A straight ring edge measures the offset only along the ring
radial direction (rank one). Writing the identities with scalar sigmas is
therefore wrong; the estimator carries everything in matrix form:

- Each technique's unknown is a symmetric 2x2 covariance (three elements)
  expressed in the technique's own *basis frame* -- the fixed image frame,
  or a per-frame rotating frame (the limb arc direction) that the design
  matrix rotates into image coordinates frame by frame. Rotating
  anisotropy is handled by the solve itself, without orientation binning.
- A rank-one technique (straight ring edge) carries a single variance
  unknown, and every equation involving it is projected onto that frame's
  measurement axis first. The tangential component of its reported offset
  is never consumed.
- Techniques may share a parameter group (the same technique on two bodies
  in one frame), which asserts one common covariance across the group --
  an explicit stationarity assumption the cohort binning must justify.
- A pair suspected of correlated errors can be declared, adding its
  symmetric cross-covariance as unknowns instead of assuming independence;
  the cohort must be over-determined enough to carry them. Two model
  restrictions apply: a full/full pair's cross-covariance is a single
  image-frame-constant matrix (no rotating form), and a pair involving a
  rank-one technique carries one scalar, which assumes the projected
  coupling is the same along every axis (exact only for an isotropic
  shared error).

Identifiability is an output, not an assumption: the solve reports the
design matrix's singular spectrum, the null space mapped back to parameter
names, and a per-parameter identifiability score.

Bias handling deserves precision, because the naive claim is false.
Each pair's differences are centered by a fitted mean model before the
second moments: constant (image-frame) terms always, plus rotating-frame
mean components for every member declared rotating. The consequences:

- A deterministic bias that is *constant in the image frame* lands in
  the mean channel and never touches the covariances.
- A *geometry-locked* bias -- constant in a rotating frame, like a limb
  or disc fit pulled radially toward its body -- has image-frame mean
  near zero over an orientation-diverse cohort. If its technique is
  declared rotating, the mean model absorbs it (and reports the fitted
  bias). If not, nothing is subtracted and the bias aliases into the
  recovered covariance as ``C + mu mu^T`` -- silently, with the system
  well-conditioned and, in general, no negative-variance symptom; shared
  between two techniques it additionally creates undeclared cross terms
  that corrupt every recovered value. Orientation diversity is therefore
  double-edged: the same sweep that makes the system identifiable
  converts geometry-locked biases from harmless constants into silent
  covariance pollution. Mitigations are declaring the rotating basis or
  estimating means bin-wise in each technique's own frame.
- Biases locked to geometry the model does not carry (illumination
  direction, arc fraction) are not absorbed and still alias.

Where per-technique covariance is recoverable
=============================================

Validated on planted-truth simulated scenes (each technique's true error is
measurable by construction; the campaign record holds the numbers):

- **One resolved body, limb + disc only.** Not separable: one matrix
  equation, two unknown matrices. Only the *combined* covariance
  ``C_limb + C_disc`` is determined; the split is a three-dimensional null
  space, demonstrated empirically (any multiple of the null directions
  fits the data identically). Report pairwise combined covariance only.
- **Body + straight ring edge, radial direction frozen across the
  cohort.** The ring leg constrains each body technique only along that
  one axis: the ring variance and each technique's radial-radial
  projection become identifiable (the full limb+disc sum closes the
  system along the radial axis), but the split of the remaining elements
  stays degenerate (two-dimensional null space).
- **Body + straight ring edge, radial direction varying across the
  cohort.** Fully identifiable: as the radial axis sweeps, the rank-one
  equations probe every direction, and all covariance elements of limb,
  disc, and the ring variance separate. Orientation diversity is the
  lever -- a cohort whose ring geometry never rotates cannot be rescued
  by more frames.
- **Anisotropic (partial-arc) limb + ring only.** The limb deviatoric
  (anisotropy) part separates when the arc-to-radial relative angle
  varies, but the isotropic part of the limb covariance and the ring
  variance remain entangled (one-dimensional null space). Adding the disc
  closes it.
- **Two resolved bodies (limb + disc each).** With the same-technique
  covariance shared across the two bodies, cross-body pairs separate limb
  from disc in full 2x2 form -- this is what a second body adds over the
  rank-one ring. The shared-covariance assumption must hold across the
  pair (similar apparent size and geometry), and same-technique
  cross-body pairs must first be shown uncorrelated: correlation there
  corrupts the separation silently while the system stays
  well-conditioned. The campaign measured exactly this failure -- the
  limb-vs-limb pair on one frame carries a large common-mode error in
  the current sim -- and restoring correct recovery required declaring
  that pair's covariance in the solve rather than assuming it away.

Bias independence and the shared preprocessing layer
====================================================

The distance-transform techniques (limb, terminator, ring edge) consume
shared per-image products built once per navigation: the smoothed gradient,
the edge distance transform, and the noise-sigma estimate (see
:doc:`dev_guide_techniques_image_derivatives`). A bias in that shared layer
moves those techniques *together* -- which the pairwise differences then
cancel, hiding exactly the error the agreement study most needs to see.

The validation campaign injects a controlled bias into those shared
products (a harness-level patch translating the gradient/DT images by a
per-scene random shift, and separately scaling the shared noise sigma) and
measures the induced coupling for the pivotal pairs. Consequences for any
consumer of the estimator:

- Techniques that sample the shared gradient/DT products (limb, ring edge)
  track a bias in that layer essentially one-for-one; their pairwise
  difference is blind to it, and a joint solve without a declared pair
  covariance misattributes the shared component onto the *other*
  techniques in the system. Limb-DT and ring-edge-DT must not be treated
  as independent through this layer.
- Techniques that read the image directly (disc correlation, blob
  centroid) are structurally decoupled from the gradient/DT products;
  measured couplings and residual caveats are in the campaign record.
- Declaring the suspect pair covariance in an over-determined cohort
  recovers the injected coupling instead of hiding it -- the estimator's
  mechanism for keeping a correlated pair in the solve with its coupling
  accounted for, when the cohort can carry it. Otherwise the pair is
  excluded from joint solves and only its combined covariance is reported.

Scope of the verdict
====================

These results are exact statements about the estimator's mathematics and
about the navigation algorithms on simulated scenes whose truth is planted
by construction (the verification axis of the simulator's capability
envelope; see :ref:`sim-capability-envelope`). The bias-independence
verdict in particular is the simulator's verdict: it establishes what the
solve detects when a shared bias exists, and which couplings exist in the
rendered regime measured. It does not certify that real-image biases are
independent -- that requires the real-data closure checks of the agreement
study itself, and every real-frame application inherits the identifiability
map's constraints per composition.
