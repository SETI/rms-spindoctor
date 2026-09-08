=============================================
Per-Image Metadata (``_metadata.json``)
=============================================

Overview
========

Every image the navigation pipeline touches produces one JSON document, written
to ``<nav_results_root>/<results_path_stub>_metadata.json`` alongside the
``_summary.png`` preview. This file is the pipeline's authoritative record of
what happened to that image: the measured pointing offset and its uncertainty,
the corrected camera attitude, every technique's individual answer, every
feature that was extracted and whether it survived its gate, the image-quality
verdict, and the provenance needed to reproduce the run. It is consumed by the
backplane generator, the reprojection and mosaic drivers, the PDS4 bundle
builder, the metadata consolidator, the statistics ingester, and the C-kernel
writer, and it is designed to be read directly by external users.

This chapter specifies the file exactly: every key that can appear, its type,
its meaning, when it is present and when it is absent, and one complete example
per document shape. The writers are
:func:`~spindoctor.navigate_image_files.navigate_image_files` (the driver
``sd_offset`` runs per image) and
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` (which
produces the ``navigation_result`` block); the manual-navigation driver
(``sd_offset --manual``) writes the same schema through
:func:`~spindoctor.navigate_image_files.build_metadata_from_result`. A unit
test compares this chapter's examples against what those writers actually
emit, so the chapter and the code cannot silently drift apart.

Serialization
-------------

The file is UTF-8 JSON with two-space indentation, written by
:func:`~spindoctor.support.file.json_as_string`. Keys appear in the fixed
insertion order shown in the examples. NumPy scalars and arrays are converted
to native Python types before serialization. Non-finite floats never appear in
the file: the curator maps ``+inf`` to the finite sentinel
:data:`~spindoctor.feature.constants.JSON_INF_SENTINEL` (``1.0e9``), ``-inf``
to ``-1.0e9``, and NaN to the *positive* sentinel (a NaN here only ever comes
from a degenerate variance, and rendering it as ``0.0`` would read as
zero-variance certainty, the opposite of the truth). A value equal to the
sentinel therefore means "unbounded / no information", never a measured
billion of anything.

One number reaches the file without passing through the curator, and is
settled where it enters instead. The top-level ``offset`` is written straight
off :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`, whose
construction refuses a non-finite offset outright -- an offset is a position,
and the value would be this code's arithmetic gone wrong rather than something
to record.

The offset convention
---------------------

Everything in this file uses the ``(v, u)`` pixel convention: ``v`` is the row
(line) coordinate and ``u`` the column (sample) coordinate. An offset is a pair
``[dv, du]`` with the meaning: if the SPICE-predicted position of a feature is
``(v, u)``, its actual position in the image is ``(v + dv, u + du)``.

The same measured offset appears twice, at two precisions:

* The top-level ``offset`` key is the **full-precision** value, exactly as the
  ensemble computed it. Downstream geometry consumers (backplanes, mosaics)
  read this one.
* ``navigation_result.offset_px`` is the same value **rounded to 4 decimals**,
  the display form used in reports and logs.

Document shapes
===============

Three document shapes exist. Which one an image gets depends on how far the
pipeline carried it:

**Navigated**
    The image loaded and the orchestrator ran to completion. The document has
    a full ``navigation_result`` block. The top-level ``status`` is the
    navigation outcome: ``success``, ``failed``, or ``conflicted``. This shape
    covers failed navigations too: a failure still records every technique
    that ran, the feature inventory, the image classifier, provenance, and
    (when the attitude could be computed) the ``pointing`` and ``times``
    blocks; only the offset and its uncertainty are absent.

**Load error**
    The image file could not be read, or SPICE coverage was missing for its
    epoch, so no observation ever existed. ``status`` is ``error``,
    ``status_error`` says which kind, and there is no ``navigation_result``.
    The ``observation`` block is limited to what the dataset index supplied
    without opening the image.

**Early return**
    The driver refused the request before reaching the image: the batch did
    not contain exactly one image, or the image's results-path stub would have
    placed its log outside the log root. These documents are **returned to
    the caller** (and recorded in cloud-task results) but are **not written
    to disk** -- there is no per-image results location to write them to.

Top-level keys
==============

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Key
     - Type
     - Shapes
     - Meaning
   * - ``status``
     - string
     - all
     - ``success``, ``failed``, or ``conflicted`` for a navigated document
       (mirrored in ``navigation_result.status``); ``error`` for the
       load-error and early-return shapes.
   * - ``status_error``
     - string
     - error shapes
     - Machine-readable error class; see the vocabulary below. Never present
       on a navigated document (whose discrete reason is
       ``navigation_result.status_reason`` instead).
   * - ``status_exception``
     - string
     - error shapes
     - The stringified exception or refusal message. Free text, for humans.
   * - ``observation``
     - object
     - all
     - The image's identity; see `The observation block`_.
   * - ``navigation_result``
     - object
     - navigated
     - The full navigation outcome; see `The navigation_result block`_.
   * - ``timing``
     - object
     - all
     - Run timing: ``start_iso8601`` and ``end_iso8601`` (UTC ISO 8601
       strings with a ``Z`` suffix, microsecond precision) and ``elapsed_s``
       (float seconds). For a load-error document the window ends at error
       time. Built by
       :func:`~spindoctor.navigate_image_files.build_timing_section`.
   * - ``offset``
     - array
     - navigated
     - The full-precision ``[dv, du]`` offset in pixels. Present exactly when
       the navigation produced an offset: on ``success`` and ``conflicted``
       documents, absent on ``failed`` ones.
   * - ``confidence``
     - number
     - navigated
     - The full-precision calibrated confidence in ``[0, 1]``; ``0.0`` on a
       failed navigation. The rounded display form is
       ``navigation_result.confidence``.

The ``status_error`` vocabulary
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Value
     - Meaning
   * - ``missing_spice_data``
     - The image load raised a SPICE error indicating missing kernel
       coverage for the image epoch (the message carried
       ``SPICE(CKINSUFFDATA)``, ``SPICE(SPKINSUFFDATA)``, or
       ``SPICE(NOFRAMECONNECT)``). Written to disk.
   * - ``image_read_error``
     - The image load raised any other ``OSError`` or ``RuntimeError`` (a
       corrupt or unreadable file). Written to disk.
   * - ``expected_one_image_per_batch``
     - The driver was handed a batch whose size was not exactly one. Early
       return; not written to disk.
   * - ``invalid_results_path_stub``
     - The image's results-path stub would have placed its per-image log
       outside the log root. Early return; not written to disk.

The observation block
=====================

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Key
     - Type
     - Meaning and presence
   * - ``image_path``
     - string
     - Absolute path of the source image file. Present on navigated and
       load-error documents; absent on early returns (which never resolved
       an image).
   * - ``image_name``
     - string
     - Basename of the source image file. Same presence as ``image_path``.
   * - ``instrument``
     - string
     - The registered instrument name for the observation class: ``coiss``,
       ``gossi``, ``nhlorri``, ``vgiss``, or ``sim`` (``unknown`` for an
       unregistered class). Always present, in every shape.
   * - ``camera``
     - string
     - The camera that took the image: ``NAC`` or ``WAC`` for Cassini ISS
       and Voyager ISS, ``SSI`` for Galileo, ``LORRI`` for New Horizons.
       On a navigated document this comes from the loaded observation
       (:attr:`~spindoctor.obs.obs_inst.ObsInst.camera`) and is always
       present. On a load-error document the image was never opened, so the
       value falls back to what the dataset index recorded when the image
       was enumerated; it is present only when the index supplied it (an
       image navigated by explicit path rather than enumerated from an
       index has none).
   * - ``shutter_mode``
     - string
     - The shutter mode the image was taken in, for an instrument whose
       label carries one. Cassini ISS records ``NACONLY``, ``WACONLY``, or
       ``BOTSIM`` (both cameras exposed at once, sharing one spacecraft
       attitude). Omitted for instruments whose labels carry no such field
       (Voyager ISS, Galileo SSI, New Horizons LORRI) and on load-error
       documents.
   * - ``image_shape``
     - array
     - ``[v, u]`` pixel dimensions of the loaded image data, as two
       integers. Present only on navigated documents (a load never
       produced pixel data on the error shapes).

The navigation_result block
===========================

This block is emitted by
:func:`~spindoctor.nav_orchestrator.curator.build_metadata_dict` from the
in-memory :class:`~spindoctor.nav_orchestrator.nav_result.NavResult`. Every
key below is always present unless marked conditional.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Type
     - Meaning
   * - ``status``
     - string
     - ``success``, ``failed``, or ``conflicted``. Identical to the
       top-level ``status``.
   * - ``status_reason``
     - string
     - The discrete reason for the outcome; one of the
       :class:`~spindoctor.support.status_reason.NavStatusReason` values
       listed below. Unlike the top-level ``status_error`` (which
       classifies pre-navigation failures on the error shapes), this
       explains a completed navigation's outcome, including its success
       modes.
   * - ``offset_px``
     - array or null
     - ``[dv, du]`` rounded to 4 decimals; ``null`` on a failed
       navigation. The full-precision value is the top-level ``offset``.
   * - ``sigma_px``
     - array or null
     - Per-axis 1-sigma marginal uncertainty ``[sigma_dv, sigma_du]`` in
       pixels, from the square roots of the combined covariance diagonal;
       rounded to 4 decimals; ``null`` on failure.
   * - ``sigma_along_unobservable_px``
     - number or null
     - Set only when the combined covariance is rank-1 (for example a
       flat-ring-only scene that constrains just one direction). The
       in-memory value is infinite, so the serialized value is the
       ``1.0e9`` sentinel. ``null`` for full-rank results and failures.
   * - ``confidence``
     - number
     - Calibrated confidence in ``[0, 1]``, rounded to 3 decimals; ``0.0``
       on failure. The full-precision value is the top-level
       ``confidence``.
   * - ``confidence_provisional``
     - boolean
     - Always ``true``: a literal marker that confidence values and
       ``confidence_rank`` tiers are calibrated against simulated
       planted-truth recovery only and must not be read as probabilities
       of real-image accuracy. It stays ``true`` until a calibration
       against real-image error measurements lands.
   * - ``confidence_rank``
     - string
     - Five-bucket rank: ``high``, ``medium``, ``low``, ``conflicted``, or
       ``failed``. Derived from confidence, sigma, and status; the tier
       thresholds live in the ``orchestrator.ensemble.tier_thresholds``
       configuration.
   * - ``covariance_px2``
     - array or null
     - The combined covariance as nested row lists, entries rounded to 4
       decimals; ``null`` on failure. **2x2** (units pixel\ :sup:`2`) for a
       translation-only result; **3x3** when a camera rotation was fitted,
       with the third row and column carrying the rotation terms. The same
       2x2-vs-3x3 rule applies to every ``per_technique`` covariance.
   * - ``rotation_deg``
     - number
     - *Conditional.* The fitted camera rotation in degrees, rounded to 3
       decimals. Present only when rotation fitting ran and produced a
       rotation; absent otherwise.
   * - ``sigma_rotation_deg``
     - number
     - *Conditional.* 1-sigma uncertainty of the fitted rotation in
       degrees, rounded to 3 decimals; same presence rule.
   * - ``techniques_used``
     - array
     - Sorted, de-duplicated names of every technique that produced a
       result (whether or not the ensemble kept it). The techniques whose
       results actually formed the reported offset are those *not* listed
       in ``excluded_from_consensus``.
   * - ``excluded_from_consensus``
     - array
     - Sorted technique names of viable results the ensemble left out of
       the reported combine: outliers rejected against a multi-technique
       consensus, or the runner-up alternative on a conflicted result.
       Empty when every viable result contributed.
   * - ``feature_count_by_type``
     - object
     - Map from feature-type name (see `Feature inventory entries`_) to
       the count of **ungated** features of that type. Types with no
       surviving features are simply absent; a document can have an empty
       object here.
   * - ``per_technique``
     - array
     - One entry per technique result; see
       `Per-technique entries`_. Includes results the ensemble later
       dropped (their names appear in ``excluded_from_consensus``).
   * - ``feature_inventory``
     - array
     - One entry per extracted feature, kept or gated; see
       `Feature inventory entries`_.
   * - ``image_classifier``
     - object
     - The image-quality classifier's verdict; see
       `The image_classifier block`_.
   * - ``provenance``
     - object
     - The reproducibility envelope; see `The provenance block`_.
   * - ``pointing``
     - object
     - *Conditional.* The recorded camera attitude; see
       `The pointing block`_. Present whenever the observation's attitude
       could be computed -- on failed navigations too. Absent for a host
       with no SPICE camera-frame mapping (simulated images) or when the
       attitude computation failed (the failure is logged, never fatal to
       the navigation).
   * - ``times``
     - object
     - *Conditional.* The exposure epochs the attitude belongs to; see
       `The times block`_. Present exactly when ``pointing`` is present.

The ``status_reason`` vocabulary
--------------------------------

The full set of values, from
:class:`~spindoctor.support.status_reason.NavStatusReason`. ``ok`` and
``rank_1_only`` accompany ``status`` ``success``; ``conflicted_techniques``
and ``body_shape_lock_suspect`` accompany ``conflicted``; every other value
accompanies ``failed``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Value
     - Meaning
   * - ``ok``
     - Normal success.
   * - ``rank_1_only``
     - Success with only one observable axis (e.g. a flat-ring scene with
       no orthogonal feature); ``sigma_along_unobservable_px`` is set.
   * - ``conflicted_techniques``
     - Multiple agreement groups existed and the best-vs-runner-up
       summed-confidence gap fell below the configured ``agreement_gap``;
       the best group's offset is reported with ``status`` ``conflicted``
       and reduced confidence.
   * - ``body_shape_lock_suspect``
     - A geometric body consensus (disc / limb) agreed at a multi-pixel
       offset that the pose-free brightness centroid on the same well-lit
       body contradicts -- the signature of a lock onto a mismatched
       shape. Reported ``conflicted`` rather than as a confident success.
   * - ``lone_blob_in_collapsed_regime``
     - The only surviving body offset was the brightness centroid while a
       sibling geometric technique on the same body self-flagged spurious;
       the centroid carries a photometric bias no diagnostic can see, so
       the frame is declined (``failed``).
   * - ``no_signal_in_image``
     - The image classifier flagged a blank or dark frame.
   * - ``image_overexposed``
     - The image classifier saw most pixels at full-well DN.
   * - ``missing_data_dominant``
     - The image classifier saw too many missing-data pixels.
   * - ``image_corrupt``
     - The image file failed to parse or read after loading began.
   * - ``kernels_unavailable``
     - SPICE coverage was missing for the image epoch.
   * - ``instrument_not_configured``
     - No per-instrument configuration block exists for this camera.
   * - ``no_features_extracted``
     - Every feature extractor returned an empty list.
   * - ``all_features_gated``
     - Features were extracted but every one fell below its reliability
       gate.
   * - ``no_feasible_techniques``
     - Features passed the gate but no technique's feasibility check
       accepted them.
   * - ``all_techniques_spurious``
     - Every technique that ran flagged its own result spurious.
   * - ``final_confidence_below_threshold``
     - The ensemble's combined confidence sat below the configured
       ``min_confidence``.
   * - ``final_sigma_above_threshold``
     - Combined confidence cleared the lowest tier but the offset sigma
       exceeded every tier's ``max_sigma_px`` (confident but too imprecise
       to earn any tier).
   * - ``unobservable_offset``
     - Every input covariance shared one null direction; the
       precision-weighted combine could not proceed.
   * - ``contract_violation``
     - An internal navigation invariant was violated -- a programming
       error upstream, not bad image data. The full traceback is in the
       error log.
   * - ``internal_error``
     - A navigation model or technique raised an exception nothing
       anticipated, so the image was not navigated as designed and is
       reported failed rather than answered from the evidence that
       survived. The ``internal_error`` block below names the component and
       the exception class; the full traceback is in the error log.

.. _internal-error-block:

The internal-error block
------------------------

A result whose ``status_reason`` is ``internal_error`` carries an
``internal_error`` object, and no other result carries one. It says which
component raised and what class of exception it raised, so a consumer
reading the document -- rather than a person reading the log -- can tell a
navigation that hit a defect from one that simply found nothing. A Saturn
ring model that raised an ``AttributeError`` while building records
``component`` as ``rings:SATURN.create_model`` and ``exception_type`` as
``AttributeError``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``component``
     - What raised, as ``<registered name>.<method>``. The registered name
       is the model's instance name (``stars``, ``rings:SATURN``,
       ``body:TITAN``) or the technique's name (``RingEdgeNav``), which is
       the name the logs and the configuration use -- not the Python class.
   * - ``exception_type``
     - The class name of the exception raised.

The exception's message and traceback are deliberately not recorded here.
They can carry local file paths and array contents, and this document is an
archive product; the per-image error log holds both.

Per-technique entries
---------------------

Each element of ``per_technique`` is the curated form of one
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Type
     - Meaning
   * - ``technique_name``
     - string
     - Class name of the producing technique (``BodyLimbNav``,
       ``StarFieldFromCatalogNav``, ...).
   * - ``feature_ids``
     - array
     - The feature identifiers this result actually consumed
       (``limb_arc:RHEA``, ``star:...``, ...); they match
       ``feature_inventory`` entries.
   * - ``offset_px``
     - array
     - This technique's own ``[dv, du]``, rounded to 4 decimals. Never
       null: a technique that could not produce an offset produces no
       entry at all.
   * - ``covariance_px2``
     - array
     - This technique's covariance, 2x2 or 3x3 (see the top-level
       ``covariance_px2`` rule), entries rounded to 4 decimals.
   * - ``confidence``
     - number
     - The technique's self-assessed calibrated confidence, rounded to 3
       decimals.
   * - ``spurious``
     - boolean
     - The technique's structural-failure self-flag. The ensemble drops
       spurious results unconditionally, but they remain listed here.
   * - ``at_edge``
     - boolean
     - True when the solution touched its search-window boundary.
   * - ``diagnostics``
     - object
     - Technique-specific diagnostic fields. Each technique publishes a
       fixed key set declared in its diagnostics dataclass's
       ``CURATOR_FIELDS`` allow-list (see
       :doc:`/dev_guide/dev_guide_techniques_diagnostics`); float values
       are rounded to 3 decimals.
   * - ``rotation_deg``
     - number
     - *Conditional.* This technique's fitted camera rotation in degrees,
       3 decimals; present only when the technique fitted one.
   * - ``sigma_rotation_deg``
     - number
     - *Conditional.* Its 1-sigma rotation uncertainty; same presence
       rule.

Feature inventory entries
-------------------------

Each element of ``feature_inventory`` is the curated form of one
:class:`~spindoctor.nav_orchestrator.feature_summary.NavFeatureSummary` -- a
post-mortem entry for one extracted feature, whether or not it survived the
reliability gate:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Type
     - Meaning
   * - ``feature_id``
     - string
     - Unique feature identifier (``limb_arc:RHEA``,
       ``body_blob:TETHYS``, ...).
   * - ``feature_type``
     - string
     - One of the :class:`~spindoctor.feature.feature_type.NavFeatureType`
       names: ``STAR``, ``LIMB_ARC``, ``TERMINATOR_ARC``, ``BODY_DISC``,
       ``BODY_BLOB``, ``RING_EDGE``, ``RING_ANNULUS``, ``TITAN_LIMB``, or
       ``CARTOGRAPHIC_MODEL``.
   * - ``source_model``
     - string
     - Name of the producing model instance (``stars``, ``body:MIMAS``,
       ``rings:SATURN``).
   * - ``reliability``
     - number
     - Self-assessed reliability in ``[0, 1]``, rounded to 3 decimals.
   * - ``gated``
     - boolean
     - True when the reliability gate dropped this feature before any
       technique saw it.
   * - ``gate_reason``
     - string or null
     - Human-readable reason when ``gated`` is true; ``null`` otherwise.
   * - ``bbox_extfov_vu``
     - array
     - Half-open bounding box ``[v_min, u_min, v_max, u_max]`` in
       extended-FOV pixel coordinates, four integers.
   * - ``reliability_reasons``
     - object
     - The per-component breakdown of ``reliability``, so a gate decision
       is attributable from this file alone. Only the components that
       apply to this feature type appear; float components are rounded to
       3 decimals, boolean components pass through. The component
       vocabulary is the field set of
       :class:`~spindoctor.feature.feature.NavReliabilityBreakdown`
       (``predicted_snr``, ``visible_arc_fraction``, ``incidence_factor``,
       ``albedo_penalty``, ``shadow_occluded_fraction``,
       ``visible_lit_fraction``, ``overflow_fraction``, ``blob_snr``,
       ``blob_extent_px``, ``in_body_silhouette``,
       ``in_saturation_or_cosmic``, ``smear_length_ok``,
       ``titan_envelope_diameter_px``, ``titan_occluded_fraction``); a
       component added there appears here as soon as some model populates
       it.

The image_classifier block
--------------------------

The curated form of
:class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult`:

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Type
     - Meaning
   * - ``class``
     - string
     - One of ``clean``, ``blank``, ``fully_overexposed``,
       ``mostly_missing_data``, or ``corrupt``.
   * - ``saturation_frac``
     - number
     - Fraction of pixels at or above the saturation DN, 3 decimals.
   * - ``missing_frac``
     - number
     - Fraction of pixels equal to the missing-data marker, 3 decimals.
   * - ``noise_sigma``
     - number
     - Per-image MAD-based noise sigma in DN units, 3 decimals.
   * - ``max_dn``
     - number
     - Maximum DN observed in the image, 3 decimals.
   * - ``background_gradient_score``
     - number or null
     - Dimensionless score of the low-order brightness ramp across the
       sensor (see
       :func:`~spindoctor.support.background_gradient.background_gradient_score`);
       a flat field scores near zero, a scattered-light veiling gradient
       well above five. ``null`` when the image is too small for the
       measure or the downsample is perfectly constant.
   * - ``flags``
     - array
     - Advisory flags, independent of ``class``: ``partial_dropout``
       and/or ``noisy``. Usually empty.

The provenance block
--------------------

The curated form of
:class:`~spindoctor.nav_orchestrator.provenance.Provenance`. Two navigations
with identical inputs produce identical provenance except
``pipeline_run_iso8601``, which is wall-clock by construction.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Key
     - Type
     - Meaning
   * - ``spindoctor_version``
     - string
     - The package ``__version__`` string.
   * - ``spindoctor_git_sha``
     - string or null
     - Short git SHA of the source tree, with a ``-dirty`` suffix when
       uncommitted changes were present; ``null`` when the tree is not a
       git checkout or git is unavailable.
   * - ``spice_kernels``
     - array
     - Sorted basenames of every SPICE kernel actually loaded at
       navigate time. Basenames only, so the list is stable across
       machines with different kernel roots.
   * - ``spice_kernel_count``
     - integer
     - ``len(spice_kernels)``.
   * - ``static_data_hashes``
     - object
     - Map of static-data YAML filename to the SHA-256 hex digest of its
       raw bytes: the body-shape catalog, every ring catalog, and every
       per-instrument configuration file shipped with the package.
   * - ``technique_names``
     - array
     - Sorted class names of every registered technique (whether or not
       it ran on this image).
   * - ``extractor_names``
     - array
     - Sorted names of the model instances built for this observation
       (``stars``, ``body:RHEA``, ``rings:SATURN``, ...).
   * - ``config_hash``
     - string or null
     - SHA-256 hex digest of the fully-resolved configuration content
       (bundled defaults plus applied overrides, deterministically
       serialized); ``null`` when it could not be computed.
   * - ``config_overrides``
     - array
     - User/CLI override configuration file paths in application order
       (later files win the merge). Often empty.
   * - ``star_catalogs``
     - object
     - Map of configured star-catalog name to its resolution root (path
       or URL), or ``""`` when unresolvable. The catalog data carries no
       version identifier to record.
   * - ``image_et``
     - number
     - Observation midtime in TDB seconds past J2000, rounded to 6
       decimals. The unrounded epochs are in ``times``.
   * - ``pipeline_run_iso8601``
     - string
     - UTC timestamp when the run began (``Z`` suffix, whole seconds).

The pointing block
------------------

The recorded camera attitude, produced by
:func:`~spindoctor.support.cmatrix.compute_pointing` and curated from the
:class:`~spindoctor.support.cmatrix.PointingSolution` on the result. A
C-matrix here is the rotation taking a vector expressed in J2000 to the same
vector expressed in the camera frame::

    v_frame = C . v_J2000

Both matrices are given in the **SPICE camera frame** convention (not the
oops observation-frame convention, which differs by a constant flip on some
instruments), evaluated at the exposure midtime, and serialized as **nine
row-major floats**. They are deliberately **unrounded**: the C-kernel writer
identifies the baseline kernel an image navigated against by reproducing
``cmatrix_original`` to within a nanoradian, and rounding would break that
bound.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Type
     - Meaning
   * - ``cmatrix``
     - array
     - *Conditional.* The corrected attitude -- the one the camera
       actually had, per the navigation. Present only when the navigation
       produced an offset **and** fitted no camera rotation (a fitted
       rotation turns about a per-technique pivot that is not recorded,
       so no corrected attitude is claimed for it). Absent on failed
       navigations.
   * - ``cmatrix_original``
     - array
     - Always present when the block is. The uncorrected attitude the
       furnished kernels gave at navigation time.
   * - ``camera_frame``
     - string
     - SPICE name of the camera frame the matrices are expressed in
       (``CASSINI_ISS_NAC``, ``GLL_SCAN_PLATFORM``, ``NH_LORRI``,
       ``VG1_ISSNA``, ...).
   * - ``camera_frame_id``
     - integer
     - SPICE id of that frame.
   * - ``ck_frame_id``
     - integer
     - SPICE id of the object a corrected C-kernel targets (the
       spacecraft bus or scan platform the mission's kernels describe).

The times block
---------------

The exposure epochs the attitude belongs to, present exactly when
``pointing`` is. Like the C-matrices, every value is **unrounded**: the
epochs define a C-kernel segment's interpolation interval exactly.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Key
     - Type
     - Meaning
   * - ``start_et``
     - number
     - Exposure start, TDB seconds past J2000.
   * - ``stop_et``
     - number
     - Exposure stop, TDB seconds past J2000.
   * - ``midtime_et``
     - number
     - Exposure midtime, TDB seconds past J2000. The epoch both
       C-matrices are evaluated at.
   * - ``exposure_s``
     - number
     - Exposure duration in seconds.
   * - ``sclk_start``
     - string
     - Spacecraft clock string at ``start_et``.
   * - ``sclk_midtime``
     - string
     - Spacecraft clock string at ``midtime_et``.
   * - ``sclk_stop``
     - string
     - Spacecraft clock string at ``stop_et``.

Rounding summary
================

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Fields
     - Precision
   * - ``navigation_result.offset_px``, ``sigma_px``,
       ``sigma_along_unobservable_px``, every ``covariance_px2`` entry
       (top-level and per-technique), per-technique ``offset_px``
     - 4 decimals (pixel quantities)
   * - ``navigation_result.confidence``, per-technique ``confidence``,
       ``rotation_deg`` / ``sigma_rotation_deg`` (both levels), float
       ``diagnostics`` values, ``reliability`` and float
       ``reliability_reasons`` components, every float in
       ``image_classifier``
     - 3 decimals (scores and degrees)
   * - ``provenance.image_et``
     - 6 decimals
   * - Top-level ``offset`` and ``confidence``; everything in ``pointing``
       and ``times``; the ``timing`` block
     - Exact (full float precision, unrounded)

The rounding constants are chosen tighter than the per-image tolerance
budget. Two fields change on every run whatever the input --
``pipeline_run_iso8601`` and the ``timing`` block -- so a regression
comparator strips those and diffs what remains, which is byte-identical
across runs of the same input.

How consumers apply the record
==============================

Downstream stages that rebuild an image's geometry (backplanes, reprojection,
mosaics) correct the observation's pointing from this file. The recorded
``pointing.cmatrix`` is the senior form: when a usable one is present,
:func:`~spindoctor.cli.reproj.offsets.select_pointing` selects it and
:func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs` replaces the
observation's frame with the corrected attitude. A record with no usable
C-matrix -- a fitted-rotation result, a simulated image, or a malformed
pointing block -- falls back to applying the top-level ``offset`` as a
field-of-view shift, and a record with neither proceeds uncorrected. The
C-kernel writer (``sd_create_ck``) reads the same ``pointing`` and ``times``
blocks to build corrected SPICE kernels; see
:doc:`user_guide_ck_kernels` and, for the mechanism,
:doc:`/dev_guide/dev_guide_ck_kernels`.

Examples
========

The examples below are captured from real runs of the shipped writers. For
display, long arrays (the SPICE kernel list) and repetitive list entries are
shortened with the elisions noted, and site-specific filesystem prefixes in
``image_path`` and ``star_catalogs`` are abbreviated; every other value is
verbatim.

Navigated, success
------------------

A Cassini narrow-angle frame of Rhea and Tethys, navigated by a two-body limb
fit corroborated by the brightness centroid. Points worth noticing: the
top-level ``offset`` is the full-precision form of
``navigation_result.offset_px``; ``BodyBlobNav``'s own offset disagrees but
its large covariance keeps it consistent with the limb fit, so nothing was
excluded; the ``pointing`` block carries both matrices because the navigation
succeeded without fitting a rotation. Of the six ``feature_inventory``
entries, four are shown; the two ``TERMINATOR_ARC`` entries follow the same
form. Of 79 SPICE kernels, three are shown.

.. code-block:: json

    {
      "status": "success",
      "observation": {
        "image_path": "/holdings/calibrated/COISS_2xxx/COISS_2058/data/1635278317_1635374244/N1635282917_1_CALIB.IMG",
        "image_name": "N1635282917_1_CALIB.IMG",
        "instrument": "coiss",
        "camera": "NAC",
        "shutter_mode": "NACONLY",
        "image_shape": [1024, 1024]
      },
      "navigation_result": {
        "status": "success",
        "status_reason": "ok",
        "offset_px": [-1.1201, -5.9495],
        "sigma_px": [2.6104, 2.6112],
        "sigma_along_unobservable_px": null,
        "confidence": 0.788,
        "confidence_provisional": true,
        "confidence_rank": "low",
        "covariance_px2": [[6.8142, -0.0001], [-0.0001, 6.8185]],
        "techniques_used": ["BodyBlobNav", "BodyLimbNav"],
        "excluded_from_consensus": [],
        "feature_count_by_type": {
          "LIMB_ARC": 2,
          "BODY_BLOB": 2,
          "TERMINATOR_ARC": 2
        },
        "per_technique": [
          {
            "technique_name": "BodyLimbNav",
            "feature_ids": ["limb_arc:RHEA", "limb_arc:TETHYS"],
            "offset_px": [-1.1201, -5.9495],
            "covariance_px2": [[6.8142, -0.0001], [-0.0001, 6.8185]],
            "confidence": 0.788,
            "spurious": false,
            "at_edge": false,
            "diagnostics": {
              "visible_limb_arc_fraction": 0.986,
              "visible_arc_px": 277.0,
              "dt_fit_rms_px": 0.278,
              "lm_iterations": 30,
              "tukey_inlier_count": 269,
              "lm_converged": false,
              "polarity_rejection_fraction": 0.0,
              "coarse_peak_fraction": 0.625
            }
          },
          {
            "technique_name": "BodyBlobNav",
            "feature_ids": ["body_blob:RHEA", "body_blob:TETHYS"],
            "offset_px": [-17.5206, 16.7562],
            "covariance_px2": [[406.7417, 0.0], [0.0, 232.5061]],
            "confidence": 0.4,
            "spurious": false,
            "at_edge": false,
            "diagnostics": {
              "body_snr_inside_predicted_bbox": 592.079,
              "body_extent_px": 104.037,
              "blob_count": 2,
              "residual_px": 36.243,
              "max_phase_angle_deg": 113.059,
              "max_phase_irregularity_factor": 0.008
            }
          }
        ],
        "feature_inventory": [
          {
            "feature_id": "limb_arc:RHEA",
            "feature_type": "LIMB_ARC",
            "source_model": "body:RHEA",
            "reliability": 0.814,
            "gated": false,
            "gate_reason": null,
            "bbox_extfov_vu": [495, 577, 644, 727],
            "reliability_reasons": {
              "visible_arc_fraction": 1.0
            }
          },
          {
            "feature_id": "body_blob:RHEA",
            "feature_type": "BODY_BLOB",
            "source_model": "body:RHEA",
            "reliability": 0.4,
            "gated": false,
            "gate_reason": null,
            "bbox_extfov_vu": [495, 577, 644, 727],
            "reliability_reasons": {
              "blob_snr": 1.0,
              "blob_extent_px": 1.0
            }
          },
          {
            "feature_id": "terminator_arc:RHEA",
            "feature_type": "TERMINATOR_ARC",
            "source_model": "body:RHEA",
            "reliability": 0.54,
            "gated": false,
            "gate_reason": null,
            "bbox_extfov_vu": [495, 577, 644, 727],
            "reliability_reasons": {
              "visible_arc_fraction": 1.0,
              "albedo_penalty": 0.1
            }
          },
          {
            "feature_id": "limb_arc:TETHYS",
            "feature_type": "LIMB_ARC",
            "source_model": "body:TETHYS",
            "reliability": 0.786,
            "gated": false,
            "gate_reason": null,
            "bbox_extfov_vu": [592, 579, 672, 659],
            "reliability_reasons": {
              "visible_arc_fraction": 0.958
            }
          }
        ],
        "image_classifier": {
          "class": "clean",
          "saturation_frac": 0.0,
          "missing_frac": 0.0,
          "noise_sigma": 0.0,
          "max_dn": 0.27,
          "background_gradient_score": 1.732,
          "flags": []
        },
        "provenance": {
          "spindoctor_version": "0.0.0",
          "spindoctor_git_sha": "719cde5",
          "spice_kernels": [
            "09321_09326ra.bc",
            "cas00172.tsc",
            "sat428.bsp"
          ],
          "spice_kernel_count": 79,
          "static_data_hashes": {
            "config_220_body_shape.yaml": "ac10e82c9c141c0e449dcfc92d8c4f341400ffa51976f53c94c98eaabac7a52a",
            "config_400_inst_coiss.yaml": "8c20d352ed0b5b690f7fc573f505f062551966c3305a01ae0e6fba63a8400f17"
          },
          "technique_names": [
            "BodyBlobNav",
            "BodyDiscCorrelateNav",
            "BodyLimbNav",
            "BodyTerminatorNav",
            "RingAnnulusNav",
            "RingEdgeNav",
            "StarFieldFromCatalogNav",
            "StarRefineNav",
            "StarUniqueMatchNav",
            "TitanHazeNav"
          ],
          "extractor_names": [
            "body:CALYPSO",
            "body:HELENE",
            "body:RHEA",
            "body:TETHYS",
            "rings:SATURN",
            "stars"
          ],
          "config_hash": "3ca76ec39b1fb875a86bed2793adc4430785242e07d705f2d65581963040a6b6",
          "config_overrides": [],
          "star_catalogs": {
            "tycho2": "/resources/SPICE/Stars",
            "ucac4": "/star-catalogs/UCAC4",
            "ybsc": "/star-catalogs/YBSC"
          },
          "image_et": 309861208.316457,
          "pipeline_run_iso8601": "2026-08-08T16:46:29Z"
        },
        "pointing": {
          "cmatrix": [
            0.0629406951266923,
            0.05708572229135894,
            0.9963833043600459,
            -0.9339862054881185,
            -0.34847441240168675,
            0.07896424418181579,
            0.3517218174280283,
            -0.9355783260699085,
            0.03138405540002703
          ],
          "cmatrix_original": [
            0.06292815672785314,
            0.05711907456756452,
            0.996382184912687,
            -0.933988565757791,
            -0.34846813405888966,
            0.07896403345334142,
            0.35171779330102065,
            -0.9355786289062297,
            0.031420105178051556
          ],
          "camera_frame": "CASSINI_ISS_NAC",
          "camera_frame_id": -82360,
          "ck_frame_id": -82000
        },
        "times": {
          "start_et": 309861208.2064568,
          "stop_et": 309861208.4264568,
          "midtime_et": 309861208.3164568,
          "exposure_s": 0.22,
          "sclk_start": "1/1635282917.129",
          "sclk_midtime": "1/1635282917.157",
          "sclk_stop": "1/1635282917.186"
        }
      },
      "timing": {
        "start_iso8601": "2026-08-08T16:46:25.933806Z",
        "end_iso8601": "2026-08-08T16:46:33.084108Z",
        "elapsed_s": 7.150302
      },
      "offset": [-1.1200818816475144, -5.949490270240176],
      "confidence": 0.787797219762289
    }

Navigated, failed
-----------------

A Galileo SSI frame in which no extractor produced a feature. Everything the
pipeline learned is still recorded: the classifier verdict, the provenance,
and the ``pointing`` block -- with ``cmatrix_original`` only, since a failed
navigation produces no corrected attitude. There is no top-level ``offset``
key at all, and both confidence values are ``0.0``. The empty lists and the
provenance follow the same form as the success example and are shortened
here.

.. code-block:: json

    {
      "status": "failed",
      "observation": {
        "image_path": "/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059750900R.IMG",
        "image_name": "C0059750900R.IMG",
        "instrument": "gossi",
        "camera": "SSI",
        "image_shape": [800, 800]
      },
      "navigation_result": {
        "status": "failed",
        "status_reason": "no_features_extracted",
        "offset_px": null,
        "sigma_px": null,
        "sigma_along_unobservable_px": null,
        "confidence": 0.0,
        "confidence_provisional": true,
        "confidence_rank": "failed",
        "covariance_px2": null,
        "techniques_used": [],
        "excluded_from_consensus": [],
        "feature_count_by_type": {},
        "per_technique": [],
        "feature_inventory": [],
        "image_classifier": {
          "class": "clean",
          "saturation_frac": 0.0,
          "missing_frac": 0.015,
          "noise_sigma": 0.0,
          "max_dn": 251.0,
          "background_gradient_score": 2.594,
          "flags": []
        },
        "provenance": {
          "spindoctor_version": "0.0.0",
          "spindoctor_git_sha": "719cde5",
          "spice_kernels": [
            "ckc03b_plt.bc",
            "gll00010.tsc",
            "s980326b.bsp"
          ],
          "spice_kernel_count": 107,
          "static_data_hashes": {
            "config_220_body_shape.yaml": "ac10e82c9c141c0e449dcfc92d8c4f341400ffa51976f53c94c98eaabac7a52a",
            "config_410_inst_gossi.yaml": "c8c2fb961e2af58cc2b982f811a32aaed5ed4535ef5db7c5096ca24beb8a5adf"
          },
          "technique_names": [
            "BodyBlobNav",
            "BodyDiscCorrelateNav",
            "BodyLimbNav",
            "BodyTerminatorNav",
            "RingAnnulusNav",
            "RingEdgeNav",
            "StarFieldFromCatalogNav",
            "StarRefineNav",
            "StarUniqueMatchNav",
            "TitanHazeNav"
          ],
          "extractor_names": ["stars"],
          "config_hash": "3ca76ec39b1fb875a86bed2793adc4430785242e07d705f2d65581963040a6b6",
          "config_overrides": [],
          "star_catalogs": {
            "tycho2": "/resources/SPICE/Stars",
            "ucac4": "/star-catalogs/UCAC4",
            "ybsc": "/star-catalogs/YBSC"
          },
          "image_et": -286810315.74582,
          "pipeline_run_iso8601": "2026-08-08T16:47:26Z"
        },
        "pointing": {
          "cmatrix_original": [
            0.49729971289310543,
            0.34631054724236016,
            0.7954633872310731,
            0.8463774300432968,
            0.007802257200792453,
            -0.5325264037546653,
            -0.19062592024627462,
            0.9380874850733643,
            -0.2892293707060044
          ],
          "camera_frame": "GLL_SCAN_PLATFORM",
          "camera_frame_id": -77001,
          "ck_frame_id": -77001
        },
        "times": {
          "start_et": -286810315.74894506,
          "stop_et": -286810315.74269503,
          "midtime_et": -286810315.74582005,
          "exposure_s": 0.00625,
          "sclk_start": "1/00597509:09:7:2",
          "sclk_midtime": "1/00597509:09:7:2",
          "sclk_stop": "1/00597509:09:7:3"
        }
      },
      "timing": {
        "start_iso8601": "2026-08-08T16:47:23.974357Z",
        "end_iso8601": "2026-08-08T16:47:27.861701Z",
        "elapsed_s": 3.887344
      },
      "confidence": 0.0
    }

Load error
----------

A Cassini frame whose epoch falls in a C-kernel coverage gap. The image was
never opened, so there is no ``navigation_result``, no ``image_shape`` and no
epoch anywhere -- an epoch is the observation's midtime, and no observation was
built. ``camera`` comes from the dataset index, which needs no SPICE. The
exception text is shortened here; the real file carries the full SPICE message.

.. code-block:: json

    {
      "status": "error",
      "status_error": "missing_spice_data",
      "status_exception": "SPICE(NOFRAMECONNECT) -- sxform -- At epoch 2.2130942680406E+08 TDB (2007 JAN 05 22:50:26.804 TDB), there is insufficient information available to transform from reference frame 1 (J2000) to reference frame -82360 (CASSINI_ISS_NAC).",
      "observation": {
        "image_path": "/holdings/calibrated/COISS_2xxx/COISS_2028/data/1546716727_1546797712/N1546730528_4_CALIB.IMG",
        "image_name": "N1546730528_4_CALIB.IMG",
        "instrument": "coiss",
        "camera": "NAC"
      },
      "timing": {
        "start_iso8601": "2026-08-08T16:47:07.470594Z",
        "end_iso8601": "2026-08-08T16:47:11.430781Z",
        "elapsed_s": 3.960187
      }
    }

Early return
------------

The document a driver returns (but does not write to disk) when handed a
malformed batch. The ``invalid_results_path_stub`` shape is identical except
for its ``status_error`` and message. The ``observation`` block carries only
the instrument, because no image was ever resolved. This example was captured
by calling :func:`~spindoctor.navigate_image_files.navigate_image_files` with
a two-image batch, since these documents are never stored.

.. code-block:: json

    {
      "status": "error",
      "status_error": "expected_one_image_per_batch",
      "status_exception": "Expected exactly one image per batch; got 2",
      "observation": {
        "instrument": "coiss"
      },
      "timing": {
        "start_iso8601": "2026-08-08T20:26:26.088795Z",
        "end_iso8601": "2026-08-08T20:26:26.088888Z",
        "elapsed_s": 9.3e-05
      }
    }
