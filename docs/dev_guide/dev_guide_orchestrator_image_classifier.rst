==========================================================
Image Classifier (NavImageClassifier)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier` is the quick-fail image
quality classifier the orchestrator runs once per image before any
:class:`~spindoctor.nav_model.nav_model.NavModel` is constructed.
It assigns the image to one of a small set of classes (``clean`` /
``blank`` / ``fully_overexposed`` / ``mostly_missing_data``) plus optional advisory flags
(``noisy`` / ``partial_dropout``). The hard-failure classes short-circuit the pipeline
inside :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` — corrupted or blank
images fail in milliseconds with a clear status reason.

Theory
======

The classifier evaluates three cheap statistics over the sensor pixels:

- ``saturation_frac`` — fraction of pixels at or above the per-instrument
  saturation DN.
- ``missing_frac`` — fraction of pixels matching the per-instrument missing-data marker:
  pixels exactly equal to the marker (0 on every shipped instrument), or, when a
  configuration sets the marker to ``NaN``, pixels for which ``np.isnan`` holds.
- ``noise_sigma`` — robust MAD-based noise sigma over the sensor area.

The decision tree is order-sensitive:

1. **Blank check first.**  When the maximum sensor DN is below ``blank_max_dn``, the image
   is ``blank``. The blank check fires before the missing-fraction test so a near-zero
   image whose missing-data marker is also zero is not mis-classified as
   ``mostly_missing_data``.
2. **Fully overexposed.**  When the saturation fraction exceeds
   ``max_saturation_frac_clean`` (default 0.80), the image is ``fully_overexposed``.
3. **Mostly missing data.**  When the missing fraction exceeds ``max_missing_frac_clean``
   (default 0.30), the image is ``mostly_missing_data``.
4. **Clean.**  Otherwise the image is ``clean``. Two advisory flags may attach:
   ``partial_dropout`` when the missing fraction sits between
   ``partial_dropout_min_frac`` and ``max_missing_frac_clean``; ``noisy`` when the noise
   sigma exceeds ``noisy_threshold``.

The orchestrator's ``_HARD_FAILURE_TO_REASON`` dispatch table maps the four hard-failure
classes — ``blank``, ``fully_overexposed``, ``mostly_missing_data``, and ``corrupt`` (an
image that could not be read) — to
:class:`~spindoctor.support.status_reason.NavStatusReason` values; the ``clean`` case proceeds
through the pipeline regardless of the advisory flags.

Restrictions and assumptions
----------------------------

- The classifier assumes the per-instrument thresholds are calibrated correctly. An
  ``ImageQualityThresholds`` configured for a different instrument will mis-classify
  exposures.
- The classifier reads pixel statistics over the *sensor* mask only; extfov padding is
  excluded. When the caller passes ``sensor_mask=None`` every pixel is treated as sensor
  data.
- The matched-filter detection in star navigation consumes a separate noise-sigma
  estimate from
  :func:`~spindoctor.support.noise_estimate.estimate_image_noise_sigma`; the classifier's
  ``noise_sigma`` is the same MAD estimator but on the per-image sensor cohort.
- The classifier does not look at predicted feature positions — it is a global readout
  over the whole sensor, not a per-target check.

Sources of uncertainty
----------------------

The classifier reports no uncertainty. Its outputs (the verdict and the three statistics)
are deterministic functions of the input image and the configured thresholds.

Configuration
=============

Per-instrument thresholds live in
:class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds`, populated by
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs` from the
``image_quality_thresholds`` block in ``config_4N0_inst_*.yaml``. Documented at
:doc:`dev_guide_orchestrator_instrument_config`.

The thresholds dataclass:

- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.saturation_threshold_dn`
  — float, default ``4095.0`` DN. Pixels at or above this DN are flagged saturated.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.missing_data_marker_dn`
  — float, default ``0.0`` DN. Pixels exactly equal to this value are missing data; when
  the marker is ``NaN``, the classifier detects missing data via ``np.isnan`` instead.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.max_saturation_frac_clean`
  — float, default ``0.80`` (dimensionless). Above this fraction of saturated pixels the
  image is ``fully_overexposed``.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.max_missing_frac_clean`
  — float, default ``0.30`` (dimensionless). Above this fraction of missing pixels the
  image is ``mostly_missing_data``.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.partial_dropout_min_frac`
  — float, default ``0.05`` (dimensionless). At or above this fraction (and below
  ``max_missing_frac_clean``) the ``partial_dropout`` advisory flag is set.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.blank_max_dn` —
  float, default ``5.0`` DN. Below this maximum sensor DN the image is ``blank``.
- :attr:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds.noisy_threshold` —
  float, default ``10.0`` DN. Above this MAD-noise sigma the ``noisy`` advisory flag is
  set.

Implementation
==============

Source files:

- ``src/spindoctor/nav_orchestrator/image_classifier.py`` —
  :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier` and
  :class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds`.
- ``src/spindoctor/nav_orchestrator/image_classifier_result.py`` —
  :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult` plus the
  ``ImageClass`` and ``ImageFlag`` Literal aliases.
- ``src/spindoctor/support/noise_estimate.py`` —
  :func:`~spindoctor.support.noise_estimate.estimate_image_noise_sigma`, the MAD-based noise
  estimator the classifier delegates to.

Public classes (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier` — the classifier.
  Public method:
  :meth:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier.classify` runs the
  three-statistic readout plus the decision tree and returns a
  :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult`.

- :class:`~spindoctor.nav_orchestrator.image_classifier.ImageQualityThresholds` — frozen
  dataclass carrying the seven thresholds documented above.

- :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult` — frozen
  dataclass returned by ``classify``. Public fields:
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.image_class`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.saturation_frac`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.missing_frac`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.noise_sigma`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.max_dn`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.flags`.

Examples
========

**Clean Cassini calibration target.**  Saturation fraction 0.0, missing fraction 0.0, max
DN 4000, MAD sigma 4.7. Verdict::

    NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=4.7,
        max_dn=4000.0,
        flags=[],
    )

**Blank image after a failed integration.**  Maximum DN 2.1. The blank check fires
first::

    NavImageClassifierResult(
        image_class='blank',
        saturation_frac=0.0,
        missing_frac=0.62,    # the missing-data marker was 0
        noise_sigma=0.4,
        max_dn=2.1,
        flags=[],
    )

The orchestrator's hard-failure short-circuit maps ``image_class='blank'`` to
:attr:`~spindoctor.support.status_reason.NavStatusReason.NO_SIGNAL_IN_IMAGE` and returns a failed
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` before any
:class:`~spindoctor.nav_model.nav_model.NavModel` constructs.

**Clean with advisory flags.**  Saturation fraction 0.0, missing fraction 0.12, MAD sigma
12.4. The classifier reports::

    NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.12,
        noise_sigma=12.4,
        max_dn=3800.0,
        flags=['partial_dropout', 'noisy'],
    )

The orchestrator proceeds through the pipeline; the advisory flags surface in the
per-image JSON sidecar so reviewer tooling can correlate noisy / partially-dropped scenes
across a campaign.
