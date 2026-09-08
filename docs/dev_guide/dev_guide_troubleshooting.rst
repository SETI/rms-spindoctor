===============
Troubleshooting
===============

Overview
========

This page is the operator's map from a navigation outcome to its cause: which
record to open, which field to read, and what each failure vocabulary value
means in practice. It assumes the output conventions of
:doc:`/user_guide/user_guide_navigation` and the metadata specification in
:doc:`/user_guide/user_guide_metadata`.

Where to look
=============

Four records exist per image, in increasing order of detail:

1. **The run log** -- one line per image
   (``<nav_results_root>/logs/sd_offset/main_<timestamp>.log``, also on the
   console): the status, the offset, and the confidence. This is where a
   batch's failures are first noticed.
2. **The metadata file**
   (``<nav_results_root>/<stub>_metadata.json``): the structured outcome.
   ``status`` plus ``navigation_result.status_reason`` classify the failure;
   ``feature_inventory``, ``per_technique``, and ``image_classifier`` carry
   the evidence. Almost every diagnosis below reads this file.
3. **The summary PNG** (``<nav_results_root>/<stub>_summary.png``): the
   image with every model's overlay drawn at the fitted offset. The fastest
   way to see *what* the navigation thought it was looking at.
4. **The per-image log**
   (``<nav_results_root>/logs/nav/<stub>_<timestamp>.log``): the full
   narrative, section by section (models, techniques, ensemble), ending
   with an operator-readable ``Final:`` summary whose per-reason wording is
   pinned by
   :data:`~spindoctor.nav_orchestrator.status_reason_info.STATUS_REASON_INFO_TEMPLATE`.
   Re-run a single image with ``--log-level-image debug`` for more detail;
   the logging surface is documented in
   :doc:`/user_guide/user_guide_logging`.

An image that failed before navigation (load error) has only the run log,
the metadata file, and the per-image log; there is no PNG and no
``navigation_result``.

Pre-navigation errors (top-level ``status_error``)
==================================================

These documents have ``status`` ``error`` and no ``navigation_result``; the
classification is the top-level ``status_error`` and the evidence is
``status_exception`` with the traceback in ``status_traceback``.

``missing_spice_data``
    The load raised a SPICE coverage error (``CKINSUFFDATA``,
    ``SPKINSUFFDATA``, or ``NOFRAMECONNECT``). The document records no epoch
    of its own, since an epoch is the midtime of the observation the load
    never built, but the exception text names both the frame and the epoch.
    Check that the kernel set covers that epoch: attitude gaps in
    reconstructed C-kernels are the common case, and an epoch outside every
    loaded CK's coverage bounds cannot be navigated at all.

``image_read_error``
    Any other load failure -- a truncated, corrupt, or absent file. The
    exception text is the reader's own message; verify the holdings path in
    ``observation.image_path`` exists and is readable.

``internal_error``
    Navigating the image raised an exception the orchestrator does not turn
    into a result: in provenance or context construction, the
    corrected-pointing computation, the summary PNG, or a defect anywhere
    between. ``status_exception`` carries the exception type and message and
    ``status_traceback`` carries the traceback, which the per-image log
    repeats; the run goes on to the next image. A model or technique that raised is recorded instead as
    ``status_reason`` ``internal_error`` on a ``failed`` document, with the
    component named in its ``internal_error`` block.

``expected_one_image_per_batch`` / ``invalid_results_path_stub``
    Caller errors caught before the image was touched; returned to the
    caller but never written to disk. The first means the driver was handed
    a malformed batch; the second means the image's results-path stub would
    have escaped the log root (a bad index entry or task description).

Image-quality refusals
======================

The classifier (:doc:`dev_guide_orchestrator_image_classifier`) runs before
any model is built; when it refuses the frame, ``status_reason`` names the
class and ``navigation_result.image_classifier`` carries the measurements
that triggered it.

``no_signal_in_image``
    A blank or dark frame. Check ``image_classifier.max_dn`` against the
    per-instrument blank threshold. Genuinely dark frames (calibration
    frames, deep-space pointings with sub-threshold stars) are correctly
    refused; a marginally dark but navigable frame is a threshold question
    -- the per-instrument ``image_quality_thresholds`` block in
    ``config_4N0_inst_*.yaml`` owns the cutoffs.

``image_overexposed``
    Most pixels at full-well DN; see ``saturation_frac``.

``missing_data_dominant``
    Too many missing-data pixels; see ``missing_frac``. A partial dropout
    below the refusal cutoff instead sets the advisory ``partial_dropout``
    flag and navigation proceeds.

``image_corrupt``
    The pixel data failed to parse after the file opened.

Environment failures
====================

``kernels_unavailable``
    SPICE coverage was missing for the image epoch, discovered after load.
    Same diagnosis as ``missing_spice_data`` above.

``instrument_not_configured``
    No per-instrument configuration block exists for this camera. This is a
    setup defect, not an image property: the instrument's
    ``config_4N0_inst_*.yaml`` block is missing or does not cover the
    camera/mode the image reports.

Feature-stage failures
======================

``no_features_extracted``
    Every extractor returned an empty list: SPICE predicts nothing usable in
    the field of view. Read ``provenance.extractor_names`` for which models
    were even built (a star-only frame builds just ``stars``), and the
    per-image log's model sections for why each emitted nothing -- stars
    fainter than the limiting magnitude for the exposure, bodies of
    negligible apparent size, rings outside the frame. If the scene plainly
    contains a navigable feature the models did not predict, suspect the
    pointing prior: a large enough initial pointing error puts the predicted
    scene outside the extended FOV margin.

``all_features_gated``
    Features were extracted but every one fell below its type's reliability
    gate. This is the most attributable failure in the file:
    ``feature_inventory`` lists every feature with ``gated: true``, its
    ``gate_reason``, its ``reliability`` score, and the per-component
    ``reliability_reasons`` breakdown that produced it (a low
    ``visible_arc_fraction``, a poor ``blob_snr``, a star inside a body
    silhouette, ...). The per-type thresholds live in the
    ``orchestrator.reliability_gate`` configuration block. Raising a
    threshold's floor is rarely the answer; the breakdown usually shows the
    scene genuinely lacks a reliable feature.

``no_feasible_techniques``
    Features passed the gate but no technique's feasibility check accepted
    them (see :doc:`dev_guide_techniques_feasibility`) -- for example stars
    present but too few for a pattern match and too ambiguous for a unique
    match. The per-image log records each technique's feasibility verdict
    and reason.

Technique- and ensemble-stage failures
======================================

``all_techniques_spurious``
    Every technique that ran flagged its own result structurally unusable.
    Read ``per_technique``: each entry keeps its ``spurious`` flag, its
    offset, and its ``diagnostics``, so the per-technique pages (under
    :doc:`dev_guide_techniques`) can be consulted for what each diagnostic
    means. Common genuine causes: correlation on a featureless disc, an
    edge fit that never left the coarse search, star matches all rejected
    by polarity or residual gates.

``final_confidence_below_threshold``
    The ensemble fused a result but its combined confidence sat below
    ``orchestrator.ensemble.min_confidence``. The offset is withheld. The
    ``per_technique`` confidences show whether one weak technique dragged
    the combine down or everything was marginal.

``final_sigma_above_threshold``
    Confident but too imprecise: the fused sigma exceeded every tier's
    ``max_sigma_px`` (``orchestrator.ensemble.tier_thresholds``). Typical
    of scenes whose only features constrain the offset weakly (a small
    blob, a short arc). See :doc:`dev_guide_uncertainty` for where the
    sigma comes from.

``unobservable_offset``
    Every contributing covariance shared one null direction, so the
    precision-weighted combine could not produce any offset. A single
    straight ring edge with nothing else in frame is the canonical case.

``contract_violation``
    An internal invariant was violated -- a programming error, not bad
    image data. The full traceback is in the error log; file it as a
    defect rather than re-running with different settings.

``internal_error``
    A navigation model or technique raised an exception nothing anticipated,
    so the image was not navigated as designed. Like ``contract_violation``
    this is a defect rather than a setting to change, and the difference
    between them is only where it came from: a violated invariant that core
    code checked for, against anything else a plugin threw.

    Read ``navigation_result.internal_error`` in the metadata document
    first. It names the component as ``<registered name>.<method>``
    (``rings:SATURN.create_model``, ``RingEdgeNav.navigate``) and the
    exception class (``AttributeError``), which is usually enough to place
    the fault without opening the log; the log has the traceback.

    An exception raised by one model or technique fails the whole image even
    when others succeeded, and that is deliberate. Continuing on the
    survivors produces an offset that looks exactly like a whole one, and a
    frame silently navigated on less evidence than it should have been is
    worse than a frame reported failed -- especially at campaign scale,
    where nobody reads the logs. A dependency regression that costs one
    model its output shows up as a batch of failures rather than as a
    quiet, unmeasurable loss of accuracy.

Conflicted results
==================

A ``status`` of ``conflicted`` is not a failure: an offset **is** reported
(top-level ``offset``), but downstream consumers refuse it without explicit
opt-in, because the evidence disagreed. Three reasons produce it:

``conflicted_techniques``
    Two (or more) agreement groups existed and the winner's summed
    confidence did not clear the runner-up by the configured
    ``agreement_gap``. ``excluded_from_consensus`` names the losing side;
    comparing the ``per_technique`` offsets shows the disagreement
    directly, and the summary PNG usually makes the wrong party obvious (a
    star solution against a body solution displaced along a limb, for
    example).

``body_shape_lock_suspect``
    A disc/limb consensus agreed at an offset the same body's brightness
    centroid contradicts -- the signature of a geometric lock onto a
    mismatched shape model. Treat the geometric offset with suspicion at
    small phase angles and irregular bodies.

``lone_blob_in_collapsed_regime``
    Reported ``failed`` rather than conflicted, but of the same family: the
    only surviving evidence was a brightness centroid on a body whose
    geometric technique self-flagged spurious, a regime (high phase, haze
    crescents) where the centroid carries an invisible photometric bias.

For any conflicted frame the manual navigation dialog (``sd_offset
--manual``, :doc:`dev_guide_techniques_manual`) is the adjudicator of last
resort: it writes the same metadata schema with the operator's chosen
offset.

Degraded successes worth noticing
=================================

``rank_1_only``
    A success, but only one axis is constrained;
    ``sigma_along_unobservable_px`` is the ``1.0e9`` sentinel and the
    covariance is rank-1. Consumers that need both axes must treat this
    like a failure; consumers projecting onto the observable direction
    (ring radial scans) can use it as-is.

Low ``confidence_rank``
    ``low`` means the fused result cleared the floor but not the
    ``medium``/``high`` sigma requirements. The tiers are sim-calibrated
    (``confidence_provisional`` is always ``true``); see the note in
    :doc:`/user_guide/user_guide_navigation` before treating any tier as a
    real-image probability.

``pointing`` present without ``cmatrix``
    The attitude baseline was recorded but no corrected attitude: either
    the navigation failed (no offset to convert) or a camera rotation was
    fitted (no recorded pivot). Consumers fall back to the pixel offset;
    see :doc:`dev_guide_ck_kernels`.
