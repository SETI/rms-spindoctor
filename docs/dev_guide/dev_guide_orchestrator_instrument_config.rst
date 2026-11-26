==========================================================
Per-Instrument Settings (InstrumentSettings)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` is the frozen
dataclass that carries per-instrument runtime parameters the orchestrator needs to navigate
an observation. The companion function
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs` reads the
camera block (:class:`~spindoctor.config.sections_camera.CameraBlock`) that an
:class:`~spindoctor.obs.obs_inst.ObsInst` carries as ``inst_config`` and returns a
populated dataclass. Every per-instrument
behavioral branch in :class:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator` ultimately
reads off this dataclass.

Theory
======

Per-instrument configuration is split into two layers:

- The static per-instrument YAML (``config_400_inst_coiss.yaml``,
  ``config_410_inst_gossi.yaml``, ``config_420_inst_nhlorri.yaml``,
  ``config_430_inst_vgiss.yaml``) carries the slow-moving parameters: data unit
  convention, saturation DN, classifier thresholds, camera rotation flags, and the
  source-image filter. Each camera's block parses into one type when the configuration
  loads (see :ref:`config-camera-blocks`).
- The per-image observation snapshot
  (:class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst`) carries the fast-moving
  parameters: PSF sigma, midtime, extfov margin. These come from the per-image instrument
  spec rather than the YAML.

:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs` reads the
slow-moving keys off the per-camera YAML block and returns a populated
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`. The orchestrator
calls it once per observation in ``_make_context`` and the result is consumed by:

- :class:`~spindoctor.nav_orchestrator.image_classifier.NavImageClassifier` (via the
  ``thresholds`` field).
- The orchestrator's saturation-mask construction (via ``saturation_dn`` / ``data_units``).
- The orchestrator's pre-filter selection (via ``source_image_filter``).
- The :class:`~spindoctor.nav_orchestrator.nav_context.NavContext` rotation flags
  (``fit_camera_rotation``, ``max_rotation_deg``).

Restrictions and assumptions
----------------------------

- Every key's type and range, and which keys a camera's ``data_units`` require or
  refuse, are checked when the configuration loads, so this function does not check
  anything itself.
- The dataclass is frozen; new instances are returned per observation rather than
  mutated.

Sources of uncertainty
----------------------

The dataclass carries no uncertainty.

Configuration
=============

The camera-block keys read by
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs`. Every
camera sets each key unless it is marked as one that a camera's units leave out:

- ``data_units`` — str, one of ``'raw_dn'`` or ``'calibrated_if'``. ``raw_dn`` exposes
  pixels in raw counts; ``calibrated_if`` exposes pixels in calibrated I/F reflectance.
- ``noise.saturation_dn`` — float. Per-instrument full-well DN. Set by a ``raw_dn``
  camera and left out by a ``calibrated_if`` one, whose saturation cannot be identified
  after calibration.
- ``noise.marker_value`` — a number, or ``NaN`` for a product that fills a lost pixel
  with NaN. The value a missing pixel carries. Every shipped block sets 0, the calibrated ones included, since their products fill lost pixels with zero.
  On a calibrated frame a pixel of exactly 0 I/F therefore counts as missing; a valid
  calibrated pixel is never exactly 0.
- ``image_quality_thresholds`` — block consumed by
  :class:`~spindoctor.support.image_class.ImageQualityThresholds`. See
  :doc:`dev_guide_orchestrator_image_classifier` for the field-by-field schema.
- ``fit_camera_rotation`` — bool, top-level in the camera block.
  When ``True`` every technique adds in-plane camera rotation as a third parameter.
  What decides the setting, and what a fitted rotation costs, is in
  :doc:`dev_guide_rotation`. Each instrument's own value is stated in its
  chapter under :doc:`instruments/instruments`, next to the reason for it.
- ``max_rotation_deg`` — positive float, top-level in the camera block, in degrees.
  Maximum allowed rotation magnitude when ``fit_camera_rotation`` is ``True``.
- ``source_image_filter`` — the filter applied to the image before navigation, chosen by
  its ``kind``: ``BANDPASS_DOG`` with ``lo_sigma_px`` and ``hi_sigma_px``, or one of
  ``ISOTROPIC_GAUSSIAN``, ``GRADIENT_OF_GAUSSIAN``, and ``MORPH_DILATE`` with
  ``sigma_px``. It is applied only when its ``enabled`` key is true.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/instrument_config.py`` —
:class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`,
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings` — frozen dataclass.
  Public fields:

  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.data_units` —
    ``'raw_dn'`` or ``'calibrated_if'``.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.saturation_dn` —
    float or ``None``. Saturation DN; ``None`` for calibrated-IF.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.marker_value` —
    float. Missing-data sentinel.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.thresholds` —
    :class:`~spindoctor.support.image_class.ImageQualityThresholds` for the
    classifier.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.fit_camera_rotation`
    — bool.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.max_rotation_deg` —
    float.
  - :attr:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings.source_image_filter`
    — the camera's ``source_image_filter`` block.

- :func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs` — reads the
  camera block off ``obs.inst_config`` and returns a populated
  :class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`. A block
  held as a mapping, as a simulated observation holds it, is parsed into the camera
  block's type first.

The thresholds come from :func:`~spindoctor.support.image_class.image_quality_thresholds`,
which the simulated-image editor's verdict also calls, so both read a block the same
way.

Examples
========

**Cassini ISS NAC.**  ``config_400_inst_coiss.yaml`` declares
``data_units: raw_dn``, ``noise.saturation_dn: 4095``, ``noise.marker_value: 0``,
``fit_camera_rotation: false``, and ``max_rotation_deg: 5.0``.
:func:`~spindoctor.nav_orchestrator.instrument_config.instrument_settings_from_obs` returns::

    InstrumentSettings(
        data_units='raw_dn',
        saturation_dn=4095.0,
        marker_value=0.0,
        thresholds=ImageQualityThresholds(...),
        fit_camera_rotation=False,
        max_rotation_deg=5.0,
        source_image_filter=BandpassFilter(...),
    )

**Galileo SSI.**  ``config_410_inst_gossi.yaml`` declares a flat block rather
than a per-camera one, so the resolution calls
:meth:`~spindoctor.config.config.Config.camera_block` with ``'galileo_ssi'`` alone. Resolution is otherwise identical: the orchestrator's
per-image :class:`~spindoctor.nav_orchestrator.nav_context.NavContext` inherits
``fit_camera_rotation`` and ``max_rotation_deg`` from that block exactly as it
does from a per-camera one, and what those values then decide is unchanged by
where they were read from.

**Cassini ISS CALIB pipeline.**  When the operator runs the calibrated-IF pipeline,
``data_units: calibrated_if``, no ``noise.saturation_dn``, and
``noise.marker_value: 0``. The orchestrator's saturation-mask helper returns an empty
mask (saturation cannot be identified post-CALIB), and the star navigation model screens
catalog stars purely by magnitude against :meth:`obs.star_max_usable_vmag()
<spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag>`, so the calibrated-IF units carry
no effect on the star detectability decision.
