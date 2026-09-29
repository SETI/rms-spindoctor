=========================================
Uncertainty (Covariances and Propagation)
=========================================

Overview
========

Every navigation answer carries a covariance, and the covariance is
load-bearing: the ensemble weighs techniques against each other by it, the
confidence tiers are decided from the sigma derived from it, and downstream consumers
read it from the per-image metadata to know how far to trust the offset. This
chapter follows the covariance through the pipeline: how each technique
family produces its ``covariance_px2``, which calibration corrections are
folded in, how the ensemble fuses the per-technique matrices into the
combined :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`
covariance, and what finally lands in the metadata file.

Two neighboring concepts are documented elsewhere and deliberately not
duplicated here. The scalar **confidence** score -- a calibrated score for
whether the technique's answer is usable at all, calibrated against
simulated planted-truth recovery and not readable as a probability of
real-image accuracy -- is a separate channel from
the covariance and is specified in :doc:`dev_guide_techniques_confidence`.
The ensemble's **grouping, conflict, and veto** machinery (which uses the
covariances but is about agreement, not uncertainty) is specified in
:doc:`dev_guide_orchestrator_ensemble`.

Shapes, units, and sentinels
============================

A per-technique covariance
(:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2`)
is either:

- **2x2** -- the ``(dv, du)`` translation covariance, in pixel\ :sup:`2`; or
- **3x3** -- translation plus camera rotation, when rotation fitting is
  active. The third diagonal entry is the rotation variance in
  radian\ :sup:`2`; the off-diagonal third row/column carries the
  pixel-radian cross terms.

Construction validates symmetry and positive semidefiniteness; a
non-finite offset or rotation is rejected outright so it can never poison
the ensemble combine.

When rotation fitting is active but a technique cannot observe rotation (a
brightness centroid, a single star), the technique promotes its 2x2 to a 3x3
via :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`,
placing the sentinel
:data:`~spindoctor.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE`
(``1.0e15``) on the rotation diagonal. The ensemble's pseudo-inverse treats
that eigenvalue as null, so the technique contributes nothing to the fused
rotation while still fusing normally in translation.

In the serialized metadata, non-finite variances are mapped to the finite
sentinel ``1.0e9`` (see the rounding rules in
:doc:`/user_guide/user_guide_metadata`); a value at the sentinel means
"unbounded", not a measurement.

How each technique family produces its covariance
=================================================

Distance-transform fits (limb, terminator, ring edge)
-----------------------------------------------------

The DT techniques
(:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`,
:class:`~spindoctor.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`,
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`)
minimize per-vertex distance-transform residuals with Levenberg-Marquardt
under Tukey biweight reweighting (see
:doc:`dev_guide_techniques_dt_fitting`). Their covariance is the M-estimator
form computed by
:func:`~spindoctor.nav_technique.dt_fitting.weights.information_matrix_to_covariance`:
the information matrix is ``J^T diag(w) J`` (Jacobian of the residual vector,
weighted by the final robust weights), and the covariance is its
Moore-Penrose pseudo-inverse. The pseudo-inverse is what makes degenerate
geometry honest: a straight polyline yields a rank-1 information matrix, and
the returned covariance has unbounded variance along the unconstrained
direction instead of a fabricated finite one.

Correlation peaks (disc, ring annulus)
--------------------------------------

The correlation techniques
(:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`,
:class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`)
derive their statistical covariance from the matched-filter (peak-curvature)
bound, :func:`~spindoctor.support.correlate.matched_filter_covariance`: the
Fisher information of a translation fit is built from the aligned template's
gradients scaled by the residual noise, and inverted. Two properties keep the
reported sigma honest. First, the template passed in shares the normalized
intensity scale the noise was measured on, so the covariance cannot shrink
with arbitrary template amplitude. Second, the white-noise bound is inflated
by the residual's spatial **correlation area**, so spatially correlated model
error is not counted as thousands of independent constraints. A degenerate
gradient structure returns a large isotropic covariance so the result is
de-weighted rather than trusted.

Brightness centroids (blob)
---------------------------

:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` fits
one translation to the centroid residuals of every usable blob and reports
the per-axis reduced-chi-square weighted-mean variance with a
degrees-of-freedom factor, floored by the pure inverse-precision term
``1 / sum(w)``. A single blob therefore collapses to its per-blob centroid
bound -- one point is nearly unobservable for two parameters, and the
covariance says so. The cross term is deliberately zero: the per-axis errors
come from independent moment integrals along orthogonal axes.

Star techniques
---------------

:class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`
reports the reduced-chi-square weighted-mean variance over its inlier
residuals (with the same degrees-of-freedom inflation, and a
``1 / sum(w)`` floor for the noise-free case). Its 3-DoF variant co-fits the
rotation and reports the corresponding 3x3.
:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
starts from the per-star anisotropic bound carried on the feature
(computed from predicted SNR and smear by the star model) and inflates it by
the squared match residual, so a noisy match reports its actual scatter
rather than the noise-free lower bound.
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav`
reports the precision-weighted residual scatter of its refined inliers,
collapsing to the single feature's bound when only one inlier survives.

Titan haze
----------

:class:`~spindoctor.nav_technique.nav_technique_titan_haze.TitanHazeNav`
measures uncertainty in the symmetry-axis frame (across-axis from the
solar-symmetry fit, along-axis from the sunward limb-arc circle fit) and
rotates the diagonal into ``(v, u)`` axes; when rotation fitting is active
it reports rotation unobservable.

Calibration corrections
=======================

The statistical covariances above are tight bounds -- they price photon
noise and residual scatter, not the coherent model error a fit cannot see
(silhouette mismatch, photometric bias, catalog error). Left bare, they
over-weight model-limited techniques in the ensemble and understate the
reported sigma. Three shared corrections restore calibration:

**Model-error floor.**
    :func:`~spindoctor.nav_technique.nav_technique.add_model_error_floor`
    adds a configured ``model_error_floor_px**2`` to the translation
    diagonal in quadrature. Each technique reads its own floor from its
    ``tuning`` block
    (:func:`~spindoctor.nav_technique.nav_technique.load_model_error_floor`);
    the shipped values were fitted so that the 2-sigma coverage of the
    reported sigmas matches planted-truth recovery on the simulation
    campaign (the provenance notes live alongside the values in
    ``config_510_techniques.yaml``).

**Size-scaled NCC terms.**
    The correlation techniques additionally carry
    :class:`~spindoctor.nav_technique.nav_technique.NCCCovarianceTuning`:
    a size-proportional silhouette-error term (``model_error_size_frac``
    times the template extent -- a coherent shape mismatch displaces the
    peak by a roughly fixed fraction of the extent) and a
    localization-ambiguity term, both added in quadrature. The blob's
    centroid analog scales with body extent for the same reason.

**Ring orbit inflation.**
    Both ring techniques
    (:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`,
    :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav`)
    inflate their covariance by the declared per-edge orbit sigma along the
    direction an absorbed catalog-orbit error displaces the fit (falling
    back to an isotropic term when the annulus geometry singles out no
    direction), so a ring fix cannot claim precision the ring catalog does
    not support.

Propagation through the ensemble
================================

The ensemble fuses the winning agreement group with the Kalman-style
information-form merge::

    Sigma_combined = pinvh( sum_i pinvh(Sigma_i) )
    mu_combined    = Sigma_combined @ sum_i ( pinvh(Sigma_i) @ mu_i )

Each member's information matrix is the pseudo-inverse of its covariance;
summing information and re-inverting yields both the fused offset and the
fused covariance, cross terms included. A 3-DoF group's rotation angle is
combined on the circle (precision-weighted circular mean) rather than as a
plain coordinate; the fused covariance itself still comes from the full
information-form merge. The combine assumes its members are independent
estimators; the resolution step that drops or collapses correlated members
before the merge (so shared error sources are not double counted, which
would over-tighten the fused covariance) is specified in
:doc:`dev_guide_orchestrator_ensemble`.

Rank deficiency is a first-class outcome, not an error. When the fused
translation block has an unobservable direction (every contributing
covariance shared a null axis -- the classic case is a single straight ring
edge), the result is reported with
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`
set to infinity, ``status_reason`` ``rank_1_only``, and the offset still
valid along the observable axis. When *no* direction is observable the
combine refuses (``unobservable_offset``).

From the fused covariance the result derives:

- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_px` -- the
  square roots of the translation diagonal (per-axis marginal 1-sigma);
- :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_rotation_rad`
  -- from the rotation diagonal, when rotation was fitted;
- the ``confidence_rank`` tier, whose ``max_sigma_px`` requirement is
  checked against the larger per-axis sigma (the
  ``orchestrator.ensemble.tier_thresholds`` configuration): a result can be
  confident but too imprecise to earn a tier
  (``final_sigma_above_threshold``).

The fused result is held to the same finiteness rule as its members: its
offset, its per-axis sigma, and its fitted rotation with that rotation's
sigma must all be finite, each being arithmetic over inputs already required
to be finite. ``sigma_along_unobservable_px`` is the one exception, and an
infinity there is the rank-deficiency report above rather than a defect.

What reaches the metadata
=========================

The per-image file records the fused ``covariance_px2``, ``sigma_px``, and
``sigma_along_unobservable_px``, plus every technique's own
``covariance_px2`` under ``per_technique``, rounded per the policy in
:doc:`/user_guide/user_guide_metadata`. Consumers weighing navigated
offsets against each other should use the full covariance, not just
``sigma_px``: the off-diagonal terms matter exactly in the partially
constrained scenes where uncertainty is most anisotropic.

Honest limits
=============

The covariances price statistical error and the *calibrated* model-error
terms above; they do not price an unmodeled coherent systematic. A ring
feature whose true orbit sits off the catalog by more than the declared
orbit sigma, or a haze-biased centroid, can still produce a tight covariance
around a wrong offset -- the confident-wrong analysis in
:doc:`dev_guide_orchestrator_ensemble` and the cross-technique vetoes there
are the mitigation, not the covariance itself. The same caveat is flagged in
the metadata as ``confidence_provisional`` (see
:doc:`/user_guide/user_guide_metadata`).
