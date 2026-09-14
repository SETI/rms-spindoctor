==========================================================
Ensemble Combine (ensemble + EnsembleConfig)
==========================================================

Overview
========

:func:`~spindoctor.nav_orchestrator.ensemble.ensemble` is the function that reconciles every
per-technique :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` into a
single :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`. The orchestrator invokes the
ensemble twice per image: once after pass 1 (to derive the pass-2 prior) and once on the
union of pass-1 and pass-2 results (to produce the final answer). The reconciliation
discipline is honest: spurious results are dropped, at-edge results are dropped unless
removing them empties the set, the surviving results are grouped by Mahalanobis-distance
agreement, the highest summed-confidence group wins, and the within-group results are
fused via precision-weighted (Kalman-style) merging.

Theory
======

The ensemble's reconciliation is an eight-step pipeline.

Step 1 — drop spurious
----------------------

Every result with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` ``True`` is
dropped unconditionally. Spurious is the technique's self-assessed structural failure
flag; the ensemble does not second-guess it. A supersession filter then drops every
fallback-tier result whose source body is already covered by a non-spurious
primary-tier result on the same body (for example a
:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` fallback when
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` succeeded on that
body); a fallback with no primary coverage for its body stays in the set.

A second outlier filter then removes a lone
:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` brightness centroid
that a corroborated star consensus contradicts. When a non-single-star star fix is present
-- a multi-star pattern match, a two-star unique match, or a multi-star refine, whose
identification is cross-checked by more than one star -- and the blob agrees with no such
fix within ``blob_star_disagreement_floor_px`` (default 5 px), the blob is the outlier, not
the star consensus: a photometric brightness centroid is a weaker position witness than a
multi-star fix. Dropping it keeps a pose-free centroid biased off-center by a resolved or
irregular body from forcing a conflict against a correct star solution or dragging the
fused offset. A blob that agrees with the star fix, a single-star fix (not a corroborated
consensus), or a frame with no star fix is left untouched, and the cross-technique
body-witness veto (below) still reads every dropped result off the full result list -- so a
geometric body consensus the blob contradicts is unaffected.

Step 2 — drop at-edge
---------------------

Every result with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` ``True`` is dropped
*unless* dropping the at-edge cohort would empty the surviving set. The exception
preserves an at-edge result when it is the only signal the orchestrator has — better a
hint at a search-window edge than no answer at all.

Step 3 — pairwise agreement
---------------------------

Two results :math:`(\mu_{a}, \Sigma_{a})` and :math:`(\mu_{b}, \Sigma_{b})` agree when

.. math::

    d_{M}(a, b) = \sqrt{(\mu_{a} - \mu_{b})^{\top} (\Sigma_{a} + \Sigma_{b})^{+}
                        (\mu_{a} - \mu_{b})}

is at most ``agreement_sigma``, *or* when their Euclidean translation distance is at
most ``agreement_pixel_floor`` (default 5 px). The pixel floor exists because
per-technique covariances are CRLB-tight — well below the actual position uncertainty
driven by model error — so results agreeing visually to a few pixels can register as
hundreds of sigmas apart.

Both distances are measured **only in the intersection of the two results' observable
subspaces**, not in the full parameter space. A rank-deficient result (a flat-ring
rank-1 fit, or a rotation-unobservable star result) carries meaningless values along
the axis it cannot observe; differencing those against a result that *does* observe
that axis would manufacture disagreement out of nothing. Restricting the comparison to
the directions both results genuinely constrain lets a rank-1 ring edge and a full-rank
blob that agree radially group — and Step 6 then fuses the ring's radial precision with
the blob's along-edge constraint. Two results with no shared observable direction have
nothing to disagree on and are treated as agreeing, so their complementary constraints
combine. The pseudo-inverse used here and in Step 6 quarantines the huge
unobservable-axis sentinel variances (the ``1e15`` rotation sentinel and the
``1e12``-scale translation sentinels) and inverts them separately from the genuine
block, so a sentinel axis cannot raise the eigenvalue cutoff and silently zero the
well-constrained axes.

Step 4 — consensus-subset selection and outlier rejection
---------------------------------------------------------

Every surviving result sponsors a candidate subset: the results that agree with it
pairwise under Step 3. The subset with the highest summed confidence wins and is the
set the ensemble fuses; the winning technique names are reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.consensus_techniques` and
results outside it are *excluded from the consensus* and reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.excluded_from_consensus`.

Only *corroborating* members count toward summed confidence, quorum, and the Step 6
agreement boost. A pass-2 result — for example a
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` refine that
searches a small window around the pass-1 prior — is conditionally dependent on the
techniques that set that prior. The orchestrator records those techniques on the
result's :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.prior_source_techniques`;
when such a technique sits in the same subset, the descendant re-observed its own seed
and casts **no independent vote**. A multi-star refine may still lend its precision to
the fused offset in Step 6; a *single*-star refine adds no independent constraint at all
and is dropped from the combine entirely in Step 5 (below). This stops a marginal refine
seeded by a wrong pass-1 answer from "confirming" that answer and inflating consensus, or
from dragging the fused offset off the pass-1 techniques that agree at the truth.

Exclusion forces the conflicted branch only when the excluded results constitute a
genuine alternative answer: the strongest consensus formable among the excluded
results alone either has at least two corroborating members (an alternative with
quorum) or the winning subset is itself a singleton (no quorum anywhere). The
summed-confidence gap test that then decides a conflict is applied in two regimes:

- **Runner-up has quorum:** an *absolute* gap — conflict when
  ``best - runner_up < agreement_gap``.
- **Lone-vs-lone standoff** (singleton winner against a singleton excluded dissenter):
  a *relative* gap — conflict only when ``best - runner_up < agreement_gap * best``.
  Because the consensus selection has already excluded the dissenter as an outlier, it
  keeps veto power only while its confidence is comparable to the winner's; a dissenter
  at well under ``(1 - agreement_gap)`` of the winner's confidence is outlier-rejected,
  not a conflict.

A lone dissenter against a multi-technique consensus is outlier-rejected: the consensus
is fused normally (with the Step 7 disagreement penalty) and the dissenter appears only
in ``excluded_from_consensus`` and ``per_technique``.

Step 5 — resolve independent estimators
---------------------------------------

The precision-weighted merge and the agreement boost both assume the members they fuse are
*independent* measurements: their information matrices add (so two agreeing witnesses fuse
to a tighter covariance than either alone), and their mutual agreement multiplies the
confidence. That assumption is false whenever two members draw on the same underlying
error source, and fusing them anyway over-tightens the covariance (inflating the tier),
spuriously boosts the confidence, and lets a redundant member drag the offset. Before the
merge, :func:`~spindoctor.nav_orchestrator.ensemble_independence.resolve_independent_estimators`
resolves the winning subset into the set of estimators the merge may treat as independent,
correcting three known correlations:

- **Seeded single-star refine.** A one-inlier
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` seeded by a
  technique in the subset (Step 4) re-observes its own prior near the predicted position
  and adds no independent constraint, so it is dropped from the combine entirely. A
  multi-star refine carries genuine catalog matches and is kept (it still casts no
  corroboration vote).
- **Two ring techniques on one catalog.**
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` and
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` read the
  same predicted ring geometry from the same catalog model; a radially misplaced catalog
  puts both wrong by the same amount, so they are collapsed to a single representative
  witness rather than fused as two.
- **Disc and limb on one scattered-light gradient.** On a frame whose
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.background_gradient_score`
  clears ``scattered_light_gradient_score`` (default 5.0), a large-scale veiling ramp
  crosses the sensor and both
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` and
  :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` lock onto the ramp
  rather than the body, correlating their errors; the disc and limb results on one body are
  then collapsed to a single representative. On a clean frame (score below the threshold)
  they remain two independent witnesses.

The representative of a collapsed set is its highest-positional-precision member (with a
deterministic tie-break), so the fused offset is the single best view of the shared
measurement, not a spuriously tightened average of two. The full consensus membership is
reported on
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.consensus_techniques` for
transparency, while the resolved set of independent witnesses drives the precision-weighted
fusion, the agreement boost, the quorum count, and the vote mass.

Step 6 — precision-weighted merge
---------------------------------

Inside the winning group, fuse the per-technique offsets into one estimate via
Kalman-style information addition. The fused information matrix is the sum of the
per-technique information matrices :math:`I_{i} = \Sigma_{i}^{+}`; the fused offset is
:math:`\mu = \Sigma \, \sum_{i} I_{i} \mu_{i}`, where :math:`\Sigma` is the
pseudo-inverse of the summed information matrix. The pseudoinverse handles rank-deficient
inputs (e.g. a flat-ring-only result) gracefully — the unobservable axis carries an
unbounded marginal sigma.

Step 7 — disagreement and conflict penalties
--------------------------------------------

When any result was excluded from the consensus, the fused confidence is multiplied
by ``disagreement_penalty`` (default 0.7). When the conflict branch fired in Step 4 the
``status='conflicted'`` :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` is returned with a further
``conflicted_confidence_multiplier`` (default 0.3) applied to the winning group's
combined confidence so the JSON sidecar reflects the conflict's severity.

Step 8 — confidence-rank assignment
-----------------------------------

The fused confidence and the per-axis sigma are mapped to a five-bucket rank
(``'high'`` / ``'medium'`` / ``'low'`` / ``'conflicted'`` / ``'failed'``) by
:func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` against the per-rank
``min_confidence`` / ``max_sigma_px`` thresholds. Below the ``min_confidence`` floor the
ensemble returns ``status='failed'``.

Two honesty guards then cap the tier at ``'medium'`` regardless of how well the
confidence and sigma score:

- **Rank-deficient fused covariance.** When one translation axis is unobservable, the
  observable axis may be pinned precisely but the other is an assumption, not a
  measurement, so the result is not a ``'high'``-tier absolute fix. The
  :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`
  field carries the same verdict to the metadata.
- **Single-star consensus.** When every combined member is a single-star solution — a
  one-star :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
  match or a one-inlier
  :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` refine —
  there is no independent cross-check, so it tops out at ``'medium'`` even when the
  single-inlier confidence cap sits exactly on the ``'high'`` boundary and the star
  centroid's CRLB sigma is tight. A single-star result cross-checked by a non-star
  technique can still earn ``'high'``.

Confident-wrong results: what the tiers price and what they cannot
------------------------------------------------------------------

The confidence formulas and tier boundaries price *statistical* error -- the
scatter the per-technique diagnostics can see -- not unmodeled *systematic*
error. When every feature a technique fits is displaced coherently (a ring
feature whose real orbit sits a few pixels off the catalog orbit, a haze layer
that moves a photometric centroid, a shape model wrong at the mesh level), the
fit converges cleanly, the residuals look healthy, and no diagnostic carries
the signal, so neither the sigmoid nor the sigma gate can demote the result on
its own.

The defense the ensemble does have is technique-side model-error channels: a
technique that *declares* a systematic uncertainty widens its reported
covariance before the fuse, and every downstream step then prices it through
the ordinary machinery -- the widened axis carries less precision weight in
the merge, the fused sigma grows, and the sigma-gated tier demotes without any
special-casing here. The ring radial orbit-uncertainty channel is the worked
example (see :doc:`dev_guide_techniques_ring_edge` and
:doc:`dev_guide_techniques_ring_annulus`): on the simulated planted-orbit-error
scene (``tests/integration/sim_scenes/ring_system/orbit_error_ringlet.yaml``)
both ring techniques widen their reported covariance by the declared 2.5 px
catalog error bar along the translation each would absorb such a displacement
into -- the edge fit from its fit vertices, the annulus correlation from the
same edge normals painted into its composite template. Because the two ring
techniques read one catalog model, the Step 5 independence resolution collapses
them to a single representative witness before the merge, so the fused radial
sigma is that one witness's honest ~2.5 px rather than the value a
two-independent-witness fuse would shrink it to. The biased offset then demotes
to ``low`` (a measured confidence around 0.89 against a ~2.5 px sigma that
covers the residual bias), and the scene keeps its measured-error pin: the
channel prices the hazard, but it cannot remove the bias. Collapsing the ring
pair is what stops the spurious alternative -- treating the two correlated
views as independent, which shrank the sigma below the hazard and let the
agreement boost re-inflate the confidence. The shared DT fit-quality gates
(:doc:`dev_guide_techniques_dt_fitting`) close the adjacent
unverified-fit family before results reach the ensemble at all.

The second defense is a cross-technique one, the *body-witness veto*
(:mod:`spindoctor.nav_orchestrator.body_witness_veto`, applied inside
:func:`~spindoctor.nav_orchestrator.ensemble.ensemble` just before a body
consensus would be reported as a success). A systematic bias one body
technique cannot see is often visible as a *disagreement* against another
technique with a different failure mode, and the pose-free brightness centroid
(:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`) is the
witness the veto reads. So the witness exists on real frames as well as
simulated ones, :class:`~spindoctor.nav_model.nav_model_body.NavModelBody` emits
a ``BODY_BLOB`` alongside the geometric limb / disc rather than only when the
limb is unusable (see :doc:`dev_guide_navigation_models_body`). The blob runs
even when a primary already covers the body -- it opts in via
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.runs_as_witness`,
and is still superseded in the fuse, so only the veto sees it and the fused
offset on a passing frame is unchanged. The veto fires in two shapes:

- **Shape lock.** When the winning consensus is a geometric body technique
  (disc correlation, limb fit) and the blob witness on the same body, at or
  below half phase where its centroid is a trustworthy position reference,
  disagrees with the fused offset, the geometric techniques have locked onto a
  mismatched shape. The result is reported ``conflicted`` (``status_reason``
  ``body_shape_lock_suspect``) rather than a confident success -- unless a
  trusted (non-spurious, non-single-star) star fix independently agrees with the
  geometric consensus offset within
  :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.blob_star_disagreement_floor_px`.
  That
  corroborates the geometry the blob disputes, so the blob is the outlier and the
  shape-lock verdict is suppressed: the geometric-side mirror of the Step-1
  blob-drop, and the reason an albedo-dichotomy body like Iapetus (whose centroid
  is dragged off a correct limb) no longer self-vetoes when its stars confirm the
  limb.
- **Collapsed regime.** When the only surviving body offset is a past-half-phase
  blob centroid and a sibling geometric technique on the same body self-flagged
  spurious, the geometric fit structurally failed and only the
  photometric-bias-prone centroid remains. If the blob disagrees with the
  spurious technique's own coarse position -- the disc located the body while a
  forward-scattering haze dragged the centroid away -- the frame is declined
  (``failed``, ``status_reason`` ``lone_blob_in_collapsed_regime``). A small or
  faint body whose spurious disc agrees with the blob (both found the same
  body) is a legitimate blob-only success and is left alone.

Each disagreement is a two-sided gate: the Euclidean separation must clear a
body-scaled pixel floor (``max(disagreement_floor_px, disagreement_frac *
diameter)``) *and* the Mahalanobis distance under the combined translation
covariance must exceed ``body_witness_agreement_sigma``, so a low-SNR or
small-body centroid whose large sigma makes the separation statistically
consistent with agreement is not vetoed on scatter alone. Both checks apply
only to single-body frames, where no companion body corrupts the centroid and
the whole-frame verdict maps to one body.

The two modes below are the standing evidence that motivated the veto; each is
caught and asserts its declined outcome as a regression:

- ``tests/integration/sim_sweeps/irregularity_shape_mismatch.yaml`` -- at
  mesh-vs-ellipsoid shape mismatch the disc correlation and the limb fit both
  lock onto the mismatched model and agree at a multi-pixel wrong offset, and
  the fused confidence re-saturates to ~0.99 at a 10-17 px error. The pose-free
  blob, which does not chase the shape, disagrees by pixels; the shape-lock
  veto reports the steps whose disagreement clears the statistical gate
  ``conflicted``. The clean (zero-relief) step recovers the planted offset
  sub-pixel and stays a success, so the veto separates the shape lock from the
  legitimate fit rather than blanketing the sweep.
- ``tests/integration/sim_scenes/atmosphere/titan_crescent_horns.yaml`` (and
  its noiseless twin) -- a 155-degree haze crescent drags the blob centroid to
  a ~30 px wrong offset while the disc correlation self-flags spurious. The
  collapsed-regime veto declines the frame instead of reporting the lone-blob
  success.

The witness is a defense against a geometric lock, not a cure for the blob's
own bias: past half phase the centroid is itself crescent- and haze-biased, so
the shape-lock check does not trust it there, and on a multi-body frame it is
excluded. What remains unpriced is systematic error that displaces *every*
technique on a frame coherently, with no surviving witness that fails
differently -- a high tier on a frame with plausible *undeclared* systematic
error of that kind is not evidence against the error. The declined outcomes
above are asserted (see :ref:`sim-expected`), so a worsening regression and a
silent revert both fail CI.

A related honesty caveat survives inside the confidence scalar itself: the
two-member agreement boost assumes the members' errors are independent, so two
techniques observing the same displaced feature can fuse to a high scalar
confidence even as the sigma-gated tier demotes the result. The Step 5
independence resolution collapses the correlations it models -- the two ring
techniques on one catalog, disc and limb on one veiling gradient -- to a single
witness before the merge, so those pairs do not earn the boost; the
planted-orbit-error scene above accordingly lands at ``low`` on its honest
~2.5 px sigma rather than at a boosted near-cap scalar. The caveat is that any
correlation the resolution does not model still couples two members while the
boost treats them as independent, and on such frames the tier is the
trustworthy verdict and the scalar is not.

That scalar is **not** inert, and it must not be dismissed as documentation.
The tier boundaries are themselves fitted *from* it: the calibration tooling
reads each row's fused confidence, fused sigma, and error against truth, and
solves for the confidence at which each tier reaches its success target. A
row pairing a near-cap confidence with a multi-pixel error is therefore a
training point that pushes the fitted boundary upward at the next refit --
the artifact acting on the very mechanism that is supposed to contain it.
Two consequences worth stating plainly:

- The regression scenes under ``tests/integration/sim_scenes/`` are not in
  the calibration cohort (it is generated independently by the campaign's
  own randomized scene generator), so no scene-level exclusion is needed
  and none is applied.
- Any correlation the independence resolution does not model still generates
  cohort rows that pair a near-cap confidence with a multi-pixel error. The
  next recalibration has to treat them deliberately -- by pricing the
  correlated-error discount, or by fitting the boundary against fused sigma
  rather than the boosted scalar -- rather than absorbing them as evidence
  that high confidence at multi-pixel error is normal.

Restrictions and assumptions
----------------------------

- Per-technique covariances must be 2x2 (translation-only) or 3x3 (translation +
  rotation). The ensemble does not handle scale-disagreement or arbitrary-shape
  parameter spaces.
- The Mahalanobis grouping assumes the per-technique covariances are calibrated. An
  over-confident covariance shrinks the apparent agreement region and may cause a
  legitimate match to land in its own cluster.
- The pseudoinverse cutoff (``pinvh_rcond``) is global; rank-deficient detection uses the
  same threshold for grouping and merging so behaviour is consistent across the two
  passes.

Sources of uncertainty
----------------------

The fused covariance is the pseudo-inverse of the summed information matrix; it is the
standard precision-weighted-merge form. When the input set has no full-rank result, the
fused covariance is rank-deficient along the unconstrained axis; the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` field
captures the unbounded eigenvalue's direction. When the disagreement-penalty fires the
fused confidence is reduced multiplicatively.

Configuration
=============

Tunables live on :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`. The defaults are
module-level constants in :mod:`spindoctor.nav_orchestrator.ensemble`; the orchestrator's
constructor accepts an :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` override.

- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_sigma` — float, default
  ``2.0``. Mahalanobis-distance threshold for grouping.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_pixel_floor` — float,
  default ``5.0``. Euclidean translation-distance fallback for grouping: two results
  agree when either distance test passes (see Step 3). ``0.0`` disables the floor.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_gap` — float, default
  ``0.5``. Minimum summed-confidence gap between best and runner-up groups before
  declaring a conflict. Applied as an absolute gap when the runner-up has a quorum; in a
  lone-vs-lone standoff it is measured relative to the winner (``best - runner_up <
  agreement_gap * best``).
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.disagreement_penalty` — float,
  default ``0.7``. Multiplier on combined confidence when more than one group existed.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.conflicted_confidence_multiplier` —
  float, default ``0.3``. Additional multiplier when the conflicted branch fires.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.min_confidence` — float, default
  ``0.35``. Final-result threshold below which the ensemble returns
  :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.failed` instead of
  :meth:`~spindoctor.nav_orchestrator.nav_result.NavResult.success`. The calibration
  campaign's fused fit rate (error at most 3 px) measures 0.35-0.40 in the 0.10-0.15
  confidence bands and 0.89 at 0.35, with almost no probability mass in between, so the
  gate takes the measured upper edge of that crossing.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.blob_star_disagreement_floor_px` —
  float, default ``5.0``. Euclidean translation distance (px) within which a
  :class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav` brightness centroid is
  treated as agreeing with a corroborated (non-single-star) star fix. When such a fix is present
  and the blob agrees with none within this floor, the blob is dropped from the ensemble math as a
  contradicted outlier, so a lone centroid cannot force a conflict against a multi-star solution
  (Step 1). The same floor governs the body-witness shape-lock veto's star-corroboration
  suppression, so the ensemble-math drop and the veto-side suppression share one distance.
  Defaults to the same value as
  :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.agreement_pixel_floor` but is
  an independent field: ``0.0`` disables only the blob-drop and the veto-side suppression,
  and setting ``agreement_pixel_floor`` to ``0.0`` separately disables the grouping fallback.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.pinvh_rcond` — float, default
  ``1.0e-9``. Cutoff for :func:`scipy.linalg.pinvh`.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.max_allowed_rotation_deg` — float,
  default ``5.0``. Small-angle bound on a 3-DoF result's rotation magnitude. Every
  contributing technique clamps its rotation fit to this bound, so a result arriving
  outside it is an upstream programming error and raises
  :class:`~spindoctor.support.exceptions.NavContractError`.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.body_witness_shape_lock_max_phase_deg` —
  float, default ``60.0``. Phase ceiling below which the pose-free centroid is treated as a
  trustworthy position witness for the shape-lock veto; past it the centroid's lit-hemisphere
  bias grows and would disagree with a correct geometric fit.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.body_witness_collapse_min_phase_deg` —
  float, default ``90.0``. Phase floor above which a lone blob is the haze-bias suspect for the
  collapsed-regime veto. It sits above the shape-lock ceiling, so the two checks bracket the
  centroid's reliable-witness regime.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.body_witness_disagreement_floor_px` —
  float, default ``4.0``. Absolute floor of the shape-lock veto's blob-vs-fused
  disagreement tolerance.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.body_witness_disagreement_frac` —
  float, default ``0.03``. Fraction of the body diameter added to the floor as the
  body-witness veto's disagreement lower bound. Well-navigated single-body frames hold the
  blob within ~1.5 px of the fused offset; a shape lock disagrees by 5 px and up.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.body_witness_agreement_sigma` —
  float, default ``2.0``. Mahalanobis-distance threshold the blob-vs-consensus disagreement
  must exceed under the combined translation covariance before the veto fires; the pixel
  floor is a lower bound, this is the statistical gate.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.scattered_light_gradient_score` —
  float, default ``5.0``. Background-gradient score at or above which the frame is treated as
  scattered-light, so the disc and limb techniques on one body are fused as a single
  correlated witness rather than two independent ones.
- :attr:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig.tier_thresholds` — mapping
  ``rank -> {min_confidence, max_sigma_px}``; default thresholds give ``'high'`` for
  confidence at or above 0.85 with sigma at most 0.5 px, ``'medium'`` for confidence at
  or above 0.35 with sigma at most 2.0 px, and ``'low'`` for confidence at or above 0.35
  with no sigma cap. Each tier boundary is the smallest confidence at which the tier's
  sigma-gated subset achieves a 0.9 success rate against the tier's own error budget on
  the calibration campaign (the ``max_sigma_px`` values are the tier definitions, not
  fitted); the high boundary sits at 0.85 because the campaign's ring family plants
  per-feature orbit errors that produce tight-sigma, confidently-wrong locks in the
  0.55-0.80 band, failure mass no ring diagnostic can see. The tiers are
  sigma-differentiated: with calibrated covariances the ``max_sigma_px`` gate carries
  most of the discrimination, so the ``'medium'`` and ``'low'`` confidence floors rest
  at the same value as the final ``min_confidence`` gate. A structural consequence:
  the two-star and one-star confidence caps (0.8 / 0.7) sit below the high boundary,
  so capped star locks top out at medium regardless of quality. The Step 8 tier caps
  apply on top of these thresholds.

Implementation
==============

The ensemble is split across the following modules under ``src/spindoctor/nav_orchestrator/``:

- ``ensemble.py`` — the reconciler itself:
  :func:`~spindoctor.nav_orchestrator.ensemble.ensemble`,
  :func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank`, and
  :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig`.
- ``ensemble_consensus.py`` — consensus-subset selection:
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.consensus_selection`,
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.corroborating_members`,
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.corroborating_confidence`, and
  :func:`~spindoctor.nav_orchestrator.ensemble_consensus.result_param_vector`.
- ``ensemble_independence.py`` — the independence-resolution step that collapses
  correlated witnesses before the merge:
  :func:`~spindoctor.nav_orchestrator.ensemble_independence.resolve_independent_estimators`,
  :func:`~spindoctor.nav_orchestrator.ensemble_independence.is_single_star_result`, and
  :class:`~spindoctor.nav_orchestrator.ensemble_independence.IndependenceResolution`.
- ``ensemble_observability.py`` — observable-subspace linear algebra:
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.mixed_scale_pinvh`,
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.observable_basis`,
  :func:`~spindoctor.nav_orchestrator.ensemble_observability.observable_intersection_basis`,
  and :func:`~spindoctor.nav_orchestrator.ensemble_observability.mahalanobis_distance`.
- ``body_witness_veto.py`` — the cross-technique body-witness veto:
  :func:`~spindoctor.nav_orchestrator.body_witness_veto.evaluate_body_witness_veto`
  and the :class:`~spindoctor.nav_orchestrator.body_witness_veto.BodyWitnessVeto`
  verdict, read by the ensemble before a body consensus is reported.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :func:`~spindoctor.nav_orchestrator.ensemble.ensemble` — the reconciler. Returns one
  :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`.
- :func:`~spindoctor.nav_orchestrator.ensemble.derive_confidence_rank` — assign the
  five-bucket rank from a confidence / sigma pair.
- :class:`~spindoctor.nav_orchestrator.ensemble.EnsembleConfig` — frozen dataclass carrying
  the tunables documented above.

Grouping is sponsored-neighborhood consensus selection
(:func:`~spindoctor.nav_orchestrator.ensemble_consensus.consensus_selection`): every
result sponsors the subset of results that agree with it pairwise, and the subset with
the highest corroborating summed confidence wins. Unlike single-link transitive-closure
grouping, a result that agrees with only one member of the winning subset does not drag
the whole subset toward it — membership requires pairwise agreement with the sponsoring
result. :func:`scipy.linalg.pinvh` (via
:func:`~spindoctor.nav_orchestrator.ensemble_observability.mixed_scale_pinvh`) serves
both the per-pair distance test and the precision-weighted merge.

Examples
========

**Two agreeing techniques.**  Pass 1 produces
:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`
(:math:`(6.76, -17.71)` ± 0.5 px) and
:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`
(:math:`(7.00, -18.00)` ± 0.3 px). The Mahalanobis distance is well below
``agreement_sigma=2.0``; both end up in the same group. The fused offset is
:math:`(6.94, -17.92)` px with combined per-axis sigma ~0.26 px. No disagreement
penalty fires (only one group existed). The fused confidence is the precision-weighted
average of the two per-technique confidences, boosted by the agreement factor
:math:`1 + 0.5 \log_{2} n` over the :math:`n` significant corroborating members — here
:math:`n = 2`, so a factor of 1.5 — and capped by the project-wide combined-confidence
ceiling.

**Consensus selection with three techniques.**  Three techniques converge:
:math:`(7.0, -18.0)` ± 0.3, :math:`(8.0, -17.5)` ± 0.5, :math:`(11.6, 12.6)` ± 0.4. The
first two agree pairwise; the third is several sigma off in both axes. The candidate
subsets are the agreeing pair (summed confidence 0.49) and the singleton (0.74); the
singleton wins on summed confidence, and the excluded pair is an alternative *with
quorum*. The gap :math:`0.74 - 0.49 = 0.25` falls below ``agreement_gap=0.5``, so the
ensemble flags the conflict and returns ``status='conflicted'`` rather than picking the
higher-confidence isolated wrong answer (this is the documented ``multi_body`` test
scene's behaviour). Had a *third* technique joined the pair, the pair-plus-one subset
would have outweighed the singleton and the ensemble would have fused it, excluding the
dissenter as an outlier instead of conflicting.

**Rank-deficient ring-edge fit.**  A flat-ring-only scene produces a
:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` result whose covariance is
rank-1 along radial only. The ensemble's pseudoinverse handles the rank deficiency: the
fused covariance has unbounded variance along the along-edge tangent and the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px` field
captures it. When a star or body limb supplies an orthogonal-axis constraint the fused
result becomes full-rank.
