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

The classifier evaluates a few cheap statistics over the sensor pixels:

- ``saturation_frac`` — fraction of pixels at or above the per-instrument
  saturation DN.
- ``missing_frac`` — fraction of pixels matching the per-instrument missing-data marker:
  pixels exactly equal to the marker, or, when a camera
  block sets the marker to ``NaN``, pixels for which ``np.isnan`` holds.
- ``unread_line_frac`` — fraction of the sensor's lines (rows) on which every sensor
  pixel is missing. A readout that stopped early leaves its last lines unread, and a
  group of lines lost on the way to the ground leaves those lines unread. The missing
  pixels on the other lines are the scattered share,
  :attr:`~spindoctor.support.image_class.ImageQualityStatistics.scattered_missing_frac`,
  such as the line ends a compressor cut.
- ``noise_sigma`` — robust MAD-based noise sigma over the sensor area.

The decision tree is order-sensitive:

1. **Blank check first.**  When the maximum sensor DN is below ``blank_max_dn``, the image
   is ``blank``. The blank check fires before the missing-fraction test so a near-zero
   image whose missing-data marker is also zero is not mis-classified as
   ``mostly_missing_data``.
2. **Fully overexposed.**  When the saturation fraction exceeds
   ``max_saturation_frac_clean``, the image is ``fully_overexposed``.
3. **Mostly missing data.**  When the scattered missing fraction exceeds
   ``max_missing_frac_clean``, the image is ``mostly_missing_data``. Unread lines never
   refuse an image by themselves, however many there are. The lines that were read out
   hold good data, and whether they hold a star pattern, a body, or a ring to navigate
   on is for the techniques to judge. An image with no line read out has a peak of 0
   and stops at the blank check. The techniques read the
   context's missing-data mask
   (:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.missing_data_mask_ext`)
   to varying degrees: the star techniques and the blob centroid leave the missing
   pixels out, while the correlation techniques, the edge images, and the noise
   estimate still read the filled lines as data.
4. **Clean.**  Otherwise the image is ``clean``. Two advisory flags may attach:
   ``partial_dropout`` when the missing fraction, unread lines included, exceeds
   ``partial_dropout_min_frac``; ``noisy`` when the noise sigma exceeds
   ``noisy_threshold``.

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
- A line is a row of the image, as a readout sends it. A row counts as unread only when
  every one of its sensor pixels is missing; a row that keeps any read pixel is read, and
  its missing pixels count toward the scattered share. A row without sensor pixels, in
  the extfov padding, is not a line of the sensor.
- The matched-filter detection in star navigation consumes a separate noise-sigma
  estimate from
  :func:`~spindoctor.support.noise_estimate.estimate_image_noise_sigma`; the classifier's
  ``noise_sigma`` is the same MAD estimator but on the per-image sensor cohort.
- The classifier does not look at predicted feature positions — it is a global readout
  over the whole sensor, not a per-target check.

Sources of uncertainty
----------------------

The classifier reports no uncertainty. Its outputs (the verdict and the statistics)
are deterministic functions of the input image and the configured thresholds.

Configuration
=============

Per-instrument thresholds live in
:class:`~spindoctor.support.image_class.ImageQualityThresholds`, filled by
:func:`~spindoctor.support.image_class.image_quality_thresholds` from the camera block
(see :ref:`config-camera-blocks`) and documented at
:doc:`dev_guide_orchestrator_instrument_config`. The missing-data marker is the block's
``noise.marker_value``; every other value comes from its ``image_quality_thresholds``
block. The dataclass has no defaults; every value comes from the camera block.

The thresholds dataclass:

- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.saturation_threshold_dn`
  — float, in DN. Pixels at or above it are flagged saturated. It is infinite for a
  ``calibrated_if`` camera.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.missing_data_marker_dn`
  — float, the camera's ``noise.marker_value``. Pixels exactly equal to it are missing
  data. A marker of ``NaN`` selects the ``NaN`` pixels instead.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.max_saturation_frac_clean`
  — float, the camera's ``max_overexposed_frac_clean``. Above this fraction of saturated
  pixels the image is ``fully_overexposed``.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.max_missing_frac_clean`
  — float. Above this fraction of missing pixels outside the unread lines the image is
  ``mostly_missing_data``.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.partial_dropout_min_frac`
  — float, the camera's ``partial_dropout_min_frac``. Above this fraction of missing
  pixels, unread lines included, a clean image carries the ``partial_dropout`` advisory
  flag.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.blank_max_dn` —
  float, the camera's ``blank_max_dn`` or ``blank_max_if``. Below this maximum sensor
  value the image is ``blank``.
- :attr:`~spindoctor.support.image_class.ImageQualityThresholds.noisy_threshold` —
  float, the camera's ``noisy_threshold_dn`` or ``noisy_threshold_if``. Above this
  MAD-noise sigma the ``noisy`` advisory flag is set.

Implementation
==============

Source files:

- ``src/spindoctor/nav_orchestrator/image_classifier.py`` —
  :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier`.
- ``src/spindoctor/support/image_class.py`` — the decision itself:
  :class:`~spindoctor.support.image_class.ImageQualityThresholds`, the reader that fills
  it from a camera block
  (:func:`~spindoctor.support.image_class.image_quality_thresholds` and
  :func:`~spindoctor.support.image_class.missing_data_marker`), the three sensor
  measurements (:func:`~spindoctor.support.image_class.measure_image_quality`), and the
  ordered checks (:func:`~spindoctor.support.image_class.decide_image_class`). The
  simulated-image editor's verdict calls the same functions, so a scene author is told
  exactly what the navigator will decide. The module imports neither the navigator
  packages nor the simulator.
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
  statistics readout plus the decision tree and returns a
  :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult`.

- :class:`~spindoctor.support.image_class.ImageQualityThresholds` — frozen
  dataclass carrying the seven thresholds documented above (autodocumented at
  :doc:`/api_reference/api_support`).

- :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult` — frozen
  dataclass returned by ``classify``. Public fields:
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.image_class`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.saturation_frac`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.missing_frac`,
  :attr:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult.unread_line_frac`,
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
        unread_line_frac=0.0,
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
        unread_line_frac=0.0,
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
        unread_line_frac=0.0,
        noise_sigma=12.4,
        max_dn=3800.0,
        flags=['partial_dropout', 'noisy'],
    )

The orchestrator proceeds through the pipeline; the advisory flags surface in the
per-image JSON sidecar so reviewer tooling can correlate noisy / partially-dropped scenes
across a campaign.

**Truncated readout.**  The calibrated Cassini narrow-angle frame N1849746145 stopped
reading out in its 316th line, after 315 complete lines and 192 pixels of the 316th;
the calibration fills the other 708 lines, and the rest of the 316th, with the marker 0.
Almost all of the 69% missing pixels lie on unread lines, so the scattered share is far
below ``max_missing_frac_clean``. The classifier reports::

    NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.692,
        unread_line_frac=0.691,
        noise_sigma=5.0e-05,
        max_dn=0.094,
        flags=['partial_dropout'],
    )

The frame navigates on the stars in the lines it read out.
