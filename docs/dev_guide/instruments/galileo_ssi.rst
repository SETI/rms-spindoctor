============
Galileo SSI
============

Code map
========

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Piece
     - Where
   * - Obs class
     - :class:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI`, registered
       under the instrument name ``gossi``
   * - DataSet class
     - :class:`~spindoctor.dataset.dataset_pds3_galileo_ssi.DataSetPDS3GalileoSSI`,
       registered under ``gossi`` with a ``gossi_pds3`` alias
   * - Config block
     - ``galileo_ssi`` in ``config_410_inst_gossi.yaml``, flat
   * - Sim instrument key
     - ``gossi``
   * - Statistics key
     - ``gossi``
   * - Log key
     - ``other.obs`` governs the image-scoped ``LOAD IMAGE`` section; there is
       no per-instrument logging key
   * - oops host module
     - ``oops.hosts.galileo.ssi``

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI.from_file` calls
``oops.hosts.galileo.ssi.from_file(path, full_fov=True)``. ``full_fov=True``
asks the host for the whole detector array rather than the summed or windowed
region the label describes, so the navigation models are generated over the
same grid whatever readout mode a frame used. No other keyword argument is
accepted, and ``**_kwargs`` is named with a leading underscore to say so.

Beyond the host call it records ``obs.abspath`` and ``obs.image_url`` from an
``FCPath``, reads the flat ``galileo_ssi`` config section, and resolves the
extended-FOV margin. The margin resolution carries an ``isinstance(..., dict)``
branch for a size-keyed table, but this instrument's configured value is a
plain list, so that branch is dead code here; it carries a TODO saying the
branch belongs somewhere shared.

There is no calibration step. The PDS3 archive holds no I/F-calibrated Galileo
SSI product, and the navigation pipeline treats image brightness
scale-invariantly -- normalized cross-correlation, an image-derived noise
floor, a magnitude-based star gate -- so no photometric calibration is required
to navigate.

Label and index dependencies
============================

**Label fields read.** ``filter``, through the base class's property, for the
single ``filters`` entry in the metadata; and the four VICAR label items
``RIM``, ``MOD91``, ``MOD10`` and ``MOD8``, which together are the image's
frame count.
:meth:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI.get_public_metadata`
publishes that count as ``start_time_sclk``: ``_sclk_count`` converts it to an
exact number of RIM counts through
:func:`~spindoctor.support.sclk.fractional_count`, with the
moduli ``(16777215, 91, 10, 8)`` and offsets ``(0, 0, 0, 0)`` that the clock
kernel ``mk00062a.tsc`` gives, so the count is
``RIM + MOD91 / 91 + MOD10 / 910 + MOD8 / 7280``. The frame count comes a few
seconds before the exposure and the label records no count at the end of the
image, so ``midtime_sclk`` and ``end_time_sclk`` are always ``None``. A label
lacking any one of the four items publishes no count: ``start_time_sclk`` is
``None`` too.

**Index columns.** ``_INDEX_COLUMNS`` is ``FILE_SPECIFICATION_NAME``,
``_INDEX_CAMERA_COLUMNS`` is ``('INSTRUMENT_ID',)`` and ``_INDEX_CAMERA_MAP``
is ``{'SSI': 'SSI'}`` -- an
identity map, present so that the camera an image is attributed to comes from
the same mechanism for every instrument rather than being special-cased for a
single-camera one.

**Filespec parsing.** ``_get_label_filespec_from_index`` requires the index
value to end ``.LBL`` and passes it through unchanged; the suffix is matched
uppercase, deliberately.

``_get_img_name_from_label_filespec`` follows this archive's layout, which is
organized by target rather than by image number. It drops a leading ``GO_*``
volume component, requires two to four remaining levels, drops a leading
``REDO`` component, and then branches on the target directory name:

* ``RAW_CAL``, ``VENUS``, ``EARTH``, ``MOON``, ``GASPRA``, ``IDA``, ``SL9``,
  ``EMCONJ``, ``GOPEX`` are two-level, and the image is the second component.
* ``C3``, ``C9``, ``C10``, ``C20``, ``C21``, ``C22``, ``C30``, ``E4``, ``E6``,
  ``E11``, ``E12``, ``E14``, ``E15``, ``E17``, ``E18``, ``E19``, ``E26``,
  ``G1``, ``G2``, ``G7``, ``G8``, ``G28``, ``G29``, ``I24``, ``I25``, ``I27``,
  ``I31``, ``I32``, ``I33``, ``J0`` are three-level orbit directories, and the
  image is the third component.
* Anything else raises ``ValueError``, which the index scan catches per row:
  it logs an error naming the index file and the offending filespec and drops
  that row alone. So a layout the parser does not know shows up in the log
  rather than silently reducing the image count, but it does not stop the run.

Both lists are literal and both must be extended if the archive grows a
directory. A component that survives the branch but does not end ``.LBL``
returns ``None`` and is skipped without a message, which is the ordinary case
for a non-label file sitting beside the images.

``_img_name_valid`` uppercases and requires exactly twelve characters: ``C``,
ten digits, and a trailing ``R`` or ``S``. ``_extract_img_number`` is the ten
digits between them.

**Monotonicity.** ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` is left at its
``True`` default, so a ``--last-image-num`` scan stops after the first volume
entirely past the range.

Configuration block
===================

One flat section. There is one camera, so nothing is nested and the loader
reads ``config.category('galileo_ssi')`` directly.

Nothing departs from the common raw-DN schema: ``data_units: raw_dn``, a full
``noise`` block, a full ``image_quality_thresholds`` block with a
``saturation_threshold_dn``, a ``source_image_filter`` and a ``mag_offset``
table. ``extfov_margin_vu`` is a bare ``[350, 350]`` rather than a size-keyed
table.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``expected_noise_dn``, ``read_noise_dn``, ``blank_max_dn``,
``noisy_threshold_dn`` and the ``mag_offset_table`` entry. Only
``saturation_dn`` and ``saturation_threshold_dn`` (255, the 8-bit ADC ceiling)
and ``marker_value`` are hard facts. The ``_sources`` convention is not used in
this file.

``fit_camera_rotation: false``, commented in place. The residuals it declines
to fit are real; :doc:`/dev_guide/dev_guide_rotation` carries the reasoning.

Photometric and PSF calibration
===============================

**Limiting magnitude.** The Pogson-ratio form, ``anchor + log(texp) /
log(2.512)``, so each factor of 2.512 in exposure buys one magnitude of depth.
The anchor is 10.3 at a 1 s exposure, and it is **derived rather than
measured**: the project's reference anchor is 10.5 magnitudes at a 1 s exposure
for a 0.19 m aperture, and this instrument's 0.176 m aperture scales it by
collecting area as ``10.5 + 5*log10(0.176/0.19)``, with no detector-sensitivity
penalty because this is a CCD. That is a nominal-optics estimate pending
calibration against real star fields. A non-positive exposure time falls back
to the anchor.

``star_min_usable_vmag`` is 0.0: no bright-end cutoff, and saturated stars are
handled downstream.

**PSF.** ``star_psf_sigma`` is 3.0 and ``star_psf_sizes`` is a single
``100: [7, 7]``, so every star gets a 7x7 cutout regardless of magnitude.

**Magnitude offsets.** ``fallback_combo`` is ``'CL'`` and the table carries one
entry with a default of 0.0. Both are placeholders.

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 2.14e4, described
in the file as an interim value sized from the electron zero point over the
standard gain state.

**Recalibrating.** Nothing here rests on a measured frame, so recalibration
starts from acquiring star fields: the curated image cohort holds none for this
instrument, which is also what leaves the simulator PSF unverified. With star
frames in hand, the anchor is the faintest usable star at a known exposure and
the PSF sigma follows from the encircled-energy radii, both edited in YAML
rather than in code.

Frames, attitude, and rotation fitting
======================================

**Camera frame.** ``GLL_SCAN_PLATFORM``. The "camera frame" is the platform
frame itself rather than a frame hung off it, because that is the frame
``oops`` builds the observation in.

**CK object and clock.** -77001, the scan platform, whose time tags are encoded
against spacecraft clock -77.

**The oops-from-SPICE flip.** The identity. ``oops`` uses
``GLL_SCAN_PLATFORM`` directly, so the observation frame and the SPICE camera
frame are the same frame and no conjugation changes anything. The measurement
is still made and still checked at the exposure start, midtime and stop, since
an identity that stopped being the identity is exactly the failure the check
exists for.

**Frame evaluation.** The observation frame is evaluated, not frozen: a
``pxform`` at the exposure midtime reproduces what the host built.

**Per-spacecraft variation.** None. One spacecraft, one camera, one CK object
and one clock.

**Rotation fitting.** ``fit_camera_rotation`` is ``false``. Every technique
works in two degrees of freedom, ``(dv, du)``, every covariance is 2x2, and
the combined estimate carries no ``rotation_rad``. The twist these frames
carry is real and is absorbed into the reported translation.

What the flag decides, and what a fitted rotation costs, is in
:doc:`/dev_guide/dev_guide_rotation` rather than here.

C-kernel specifics
==================

Corrected kernels are written for this instrument. What follows is what a run
does, and is what the writer's holdings tests exercise.

**Baseline structure.** Ordinary time-varying type-3 kernels for object
-77001, which a corrected segment would compose a body-fixed ``delta`` onto at
each record epoch.

**Angular-velocity census.** Of the 150 -77001 segments in the local baselines,
**38 carry no angular velocity**. A corrected segment must carry a rate at
every record -- ``avflag = 0`` makes SPICE skip the segment for ``ckgpav`` and
``sxform`` and answer from the uncorrected original instead -- and the writer
applies all records or none, with none meaning refuse. So roughly a quarter of
this mission's baseline segments would stop a run that reached them, reported
as a ``ValueError`` naming the missing rate rather than as an omission, since
an exposure whose baseline supplies pointing but not a rate has no entry in the
closed omission-reason set.

**Kernel-name class rules.** None. This mission is listed in
``_MISSION_NAME_RULES`` with an empty rule tuple, which records that its
holdings were read and encode nothing rather than that nobody looked. Every
candidate is ``UNCLASSIFIED`` and the tie-break falls through to the
lexicographically greatest basename.

**Deviations in segment construction.** None. The object is not in
:data:`~spindoctor.spice_ids.FROZEN_ATTITUDE_CK_IDS`, so it would take the
standard time-varying path.

**Rigid-rotation residual.** An exact rigid rotation is not exactly a uniform
tangent-plane shift, and the difference is measured over a 17x17 grid across
the full frame, worst case over eight offset directions, at 50 pixels of total
boresight displacement: 1.82e-8 rad, which is 1.79e-3 pixels both in the
tangent plane and in pixel space. It is linear in the offset, so quoting it
without the offset it was measured at means nothing. The same term governs
any attitude-versus-offset comparison on this instrument.

**Reproduction path.** ``cspyce.pxform('J2000', 'GLL_SCAN_PLATFORM',
midtime_et)`` against each furnished candidate, accepted at 1e-9 radians.

Simulator model
===============

**Instrument key.** ``gossi``, resolving to the flat ``galileo_ssi`` block.

**PSF kernel.** ``{sigma_v: 0.80, sigma_u: 0.80, w: 1.2e-2, r0: 2.0, n: 3.0}``,
a **retained interim published sigma**. The curated cohort holds no star frames
for this instrument, so nothing independent has constrained the kernel, and
simulated-frame accuracy is bounded by unverified PSF fidelity until star
calibration frames land. This is the weakest link in this instrument's
simulator support.

**Distortion residuals.**
``{k1: -3.47e-04, k2: 1.72e-03, nonradial_rms_px: 0.0}``, measured by the
star-field distortion analysis: a ``k2``-dominated pincushion reaching about
half a pixel at the field corner.

**Artifact-mode availability.** This instrument is in the CCD set, so it
carries the CCD-only modes (``radiation_transients``, ``compression_dct``) as
well as every mode declared available to all instruments. It also carries
``truth_window``, the losslessly-clean commanded carve-out.
``alternating_lines`` carries a ``keep`` mode for this instrument's vertical
decimation, rather than the ``drop`` mode that blanks every Nth line. No mode
records an exclusion reason against this key; the modes it lacks are simply not
in their availability sets.

**Realism-match status.** Unverified. Every optical parameter is a published
or scaled estimate and the cohort offers nothing to check them against.

Image library and test coverage
===============================

**Cohort.** Eight sidecars, all single-camera: six under ``negative_cases`` and
two under ``scattered_light``. There is **no star frame**, which is why both
the limiting-magnitude anchor and the simulator PSF are unverified, and why the
twist measurement rests on seven locked frames out of eighteen in the separate
distortion cohort.

**Integration tests.** The per-image regression suite
(``tests/integration/test_autonomous_nav.py``) and the structural-invariants
test (``test_image_library.py``) run on those sidecars.
``tests/integration/test_cmatrix_frames.py`` measures the oops-from-SPICE flip
on a real frame of this instrument and checks it against the identity.
``tests/integration/test_ck_round_trip.py`` asserts that a frame of this
instrument fits no camera rotation, records a corrected attitude alongside its
uncorrected one, and is not omitted as ``rotation_unsupported``.

**Unit tests.** ``tests/spindoctor/inst/test_inst_galileo_ssi.py`` pins the
limiting-magnitude form: the anchor at unit exposure, one magnitude gained per
Pogson ratio, the non-positive-exposure fallback, and finiteness. It also pins
the published clock count: a fractional number of RIM counts with no midtime or
end count, and no count at all from a label lacking one of the four items.

PDS4 hooks
==========

Not supported. Two hooks are implemented --
``pds4_bundle_template_dir``, which falls back to ``galileo_ssi_1.0``, and
``pds4_bundle_name``, which falls back to
``galileo_ssi_backplanes_rsfrench2027`` -- and every remaining hook raises
``NotImplementedError``. Neither the template directory nor a ``pds4.gossi``
configuration entry is shipped; the configuration file carries the entry as a
commented-out stub.

Making bundles work here means implementing ``pds4_bundle_path_for_image``,
``pds4_path_stub``, ``pds4_lid_part_to_image_name``, the four LID and LIDVID
builders and ``pds4_template_variables``, and adding a template directory.

Backplanes, mosaics, and statistics
===================================

**Backplanes.** Nothing instrument-specific in the stage itself. A navigated
image records a corrected C-matrix and the stage takes the C-matrix path;
``no_cmatrix_rotation_fitted`` is not counted, since no rotation is fitted.

**Mosaics.** As for backplanes.

**Statistics.** Nothing particular.
:func:`~spindoctor.cli.stats.report_sections.resolve_offset_limit` reads
``galileo_ssi.extfov_margin_vu`` from the flat section, and since the value is
not a size-keyed table the recorded image height is not consulted. The report's
image-name rule is the extension-stripped stem with nothing further removed,
since this instrument's product names carry no suffix.

Open items
==========

* Every value marked ``# PLACEHOLDER`` in ``config_410_inst_gossi.yaml``:
  ``expected_noise_dn``, ``read_noise_dn``, ``blank_max_dn``,
  ``noisy_threshold_dn`` and the ``mag_offset_table`` entry.
* The limiting-magnitude anchor is a nominal-optics derivation, not a
  measurement.
* ``from_file`` carries a TODO saying the extended-FOV margin branch belongs
  somewhere shared.
* The simulator PSF is unverified for want of star frames in the cohort.
