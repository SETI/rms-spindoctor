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
     - ``oops.hosts.galileo.ssi``, because ``host_galileo.GalileoSSI``, from the
       ``rms-host-galileo`` package, fails to read some GO_0003 frames

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI.from_file` calls
``oops.hosts.galileo.ssi.from_file(path, full_fov=True)``. ``full_fov=True``
asks the host for the whole detector array rather than the summed or windowed
region the label describes, so the navigation models are generated over the
same grid whatever readout mode a frame used. No other keyword argument is
accepted, and ``**_kwargs`` is named with a leading underscore to say so.

The host records ``obs.image_url``, the absolute source URL, from which the
published image name is taken. Beyond the host call ``from_file`` reads the
flat ``galileo_ssi`` config section and resolves the extended-FOV margin. The
margin resolution carries an ``isinstance(..., dict)`` branch for a size-keyed
table, but this instrument's configured value is a plain list, so that branch
is dead code here; it carries a TODO saying the branch belongs somewhere
shared.

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
  it logs an error that gives the index file and the offending filespec and drops
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

One flat section. There is one camera, so nothing is nested and the
observation reads ``config.camera_block('galileo_ssi')``.

Nothing departs from the common raw-DN schema: ``data_units: raw_dn``, a full
``noise`` block, a full ``image_quality_thresholds`` block with a
``saturation_threshold_dn``, a ``source_image_filter`` and a ``mag_offset``
table. ``extfov_margin_vu`` is a bare ``[350, 350]`` rather than a size-keyed
table.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``read_noise_dn``, ``blank_max_dn``,
``noisy_threshold_dn``, and the ``mag_offset_table`` entry. Only
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

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 1.23e6 DN per
second, the simulator's measured zero point of 2.3e8 e- per second (see the
simulator model below) over the 187 e-/DN of the standard gain state.

**Summed readout mode.** ``noise.by_binning`` gives the values for the 2x2
summation mode, keyed by the binning factor: the read noise of one summed pixel,
the published 44 e- over the 187 e-/DN of gain state 2, or 0.235 DN, and the
zero point, which stays 1.23e6 because the mode has no gain of its own. Only the
faint-star limit of a simulated frame reads them. The full-resolution
``read_noise_dn`` is a placeholder of 1.0 DN. The limit's noise is at least 1 DN,
so it reads 1 DN with or without summation, and the step from an unbinned frame's
limit to a summed one's, 1.5 mag, comes from the narrower star core alone.

**Recalibrating.** Apart from the zero point, nothing here rests on a measured
frame, so recalibration starts from acquiring star fields: the curated image cohort holds none for this
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
as a ``ValueError`` that gives the missing rate rather than as an omission, since
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

**Instrument key.** ``gossi``, resolving to the flat ``galileo_ssi`` block. The
``gossi`` entry in ``src/spindoctor/sim/instruments/configs/gossi.yaml`` holds the
values below.

**PSF kernel.** ``{sigma_v: 0.80, sigma_u: 0.80, w: 1.2e-2, r0: 2.0, n: 3.0}``,
a **retained interim published sigma**. The curated cohort holds no star frames
for this instrument, so nothing independent has constrained the kernel, and
simulated frames rest on unverified PSF fidelity until star calibration frames
land. This is the weakest link in this instrument's
simulator support.

**Star zero point.** ``star_flux.e_per_s_vmag0`` is 2.3e8, the electrons a star of
V = 0 deposits each second through the clear filter. It was measured on the full
frames of the 1992 Pleiades sequence on volume GO_0008. Each frame's pointing was
refitted to its own brightest stars, and the light of every catalog star was summed
within 4 pixels of where the star falls, less the sky around it. Field stars of V 10
to 14, with their V and B-V from APASS, give these rates for a star of B-V 0.65:

* At the 10K gain state, 2.07e8 within 4 pixels, from 118 stars on 12 frames. The
  stars scatter by 0.18 mag about that rate, and the frames by 0.07 mag. A circle of
  4 pixels holds 1/1.125 of the light within 8 pixels, so the total is 2.33e8.
* At the 100K gain state, 2.0e8 in total, from 62 stars on 18 frames, 0.15 mag below
  the 10K rate.

The absolute response of Klaasen et al. (1999), 5.9e6 DN per second at 38.7 e-/DN,
is 2.28e8. Redder stars read brighter, by 0.3 to 0.4 mag for each magnitude of B-V,
and the hot Pleiades members read 0.35 mag below field stars of the same V. The
simulator draws every star at the one zero point. No frame covers the 40K or 400K
gain state through the clear filter. Summed frames read 1.33 times the DN that full
frames read for the same 14 stars at the same gain, and the simulator renders a
summed frame at the full-frame gain.

The 1990 cruise frames, the library's two Pleiades frames among them, read 1.3 mag
fainter than the 1992 frames, 7.0e7 in total, for a reason not yet known. The zero
point follows the 1992 frames, as the navigator's star photometry does.

**Distortion residuals.**
``{k1: -3.47e-04, k2: 1.72e-03, nonradial_rms_px: 0.0}``, measured by the
star-field distortion analysis: a ``k2``-dominated pincushion reaching about
half a pixel at the field corner.

**Cosmic rays.** The catalog's terms are fitted in closed loop to the cohort's
transient spikes, 57-129 per frame: a rate of 2.0e-5 events per pixel per
second, a 4.33 s readout dwell (half the full-frame readout), a 6,000 e-
deposit scale, and 0.4 native pixels of diffusion. The simulator report's
artifact-rate table gives the fit's evidence.

.. _gossi-hot-pixels:

**Hot pixels.** The hot-pixel values are fitted on raw frames at the 10K gain
state, 39 e-/DN. A pixel that spikes in at least three frames of a set counts as a
hot pixel. A hot pixel's charge and its streak are constants of the frame mode: they
do not grow with the exposure, so ``rate_e_per_sec`` is 0 and the charges are set
by ``readout_e`` and ``streak_e``.

* Nine dark frames of zero exposure from volumes GO_0002 and GO_0006 were taken
  late in 1990 in the 60 2/3 s frame mode, which most of the image-library frames
  use. Of the 640,000 pixels, 8 hold 300 e- or more, 2 hold 500 e-, and 1 holds
  1,000 e-. An exponential population with a fraction of 9e-5 and a mean of
  150 e- matches those counts, so ``pixel_fraction`` is 9e-5 and ``readout_e`` is
  150 e-.
* 47 dark-sky frames from volumes GO_0007 and GO_0008 were taken on 3 December
  1992 in the 8 2/3 s frame mode, at exposures from 0.1 to 0.8 s. Their counts are
  the same at every exposure: 435 pixels at 300 e- or more, 89 at 500 e-, and 6 at
  1,000 e-. The charge therefore does not come from the exposure, and its mean is
  about 150 e- here too. A dark rate above 8 e-/s at relative amplitude 1 would
  have raised the 500 e- count by a tenth from 0.1 to 0.8 s. These frames hold
  about 50 times the hot pixels of the 60 2/3 s frames, a difference the
  simulator does not model.
* The counts fall twentyfold from 150 to 300 e- in pairs of 60 2/3 s darks, in
  1990 and in 1992. An exponential falls that fast, and a power law would need an
  index near 4, so ``amplitude_law`` stays ``exponential``.
* Longer exposures in these sets use the extended-exposure mode. Its longer frame
  cycle is not modeled, so those frames are not used.
* The rows read after a hot pixel carry a flat streak, and the rows read before it
  carry nothing. On the median of the six 60 2/3 s darks of GO_0007, from 1992,
  237 hot pixels of 300 e- or more give 0.51% +- 0.05% of the pixel's charge per
  row. The four undated 60 2/3 s darks of GO_0002 give 0.29% +- 0.04% from 102
  hot pixels, and the seven dated 1990 darks hold only nine hot pixels, too few to
  measure. Two zero-exposure darks of GO_0007 in the 8 2/3 s mode give 0.55% +-
  0.08%. ``streak_e`` is 0.5% of 150 e-, which is 0.75 e-.
* The pixel to the right of a strong hot pixel holds up to a tenth of its charge,
  a trail along the row that the simulator leaves out.

**Artifact-mode availability.** The instrument entry's ``artifact_modes.offered``
lists its modes, and the simulator guide's mode tables
(:ref:`sim-telemetry-modes` and :ref:`sim-detector-modes`) give each mode's
cameras. Its detector model is ``ccd``. ``alternating_lines`` defaults to
its ``keep`` mode for this instrument's vertical decimation, rather than the
``drop`` mode that blanks every Nth line: the HMA and HCA telemetry formats
carry valid data on every 2nd or every 4th line only. ``missing_blocks``
defaults to 8-line blocks, the height of a lost compression slice, and
``truth_window`` to a 96-pixel square, the losslessly coded window. The entry
states no band width for ``edited_frame``, so that mode takes the registry's
generic fallback. ``instrument_defaults`` turns on ``hot_pixels``, ``banding``,
``bias_structure``, and ``bloom``. The banding is the published supply noise, vertical
bands of 65 e- every 42 pixels; a high-gain scene adds the camera's horizontal
component with ``orientation: both``. The published shutter values are an exposure
offset, the extra time the shutter adds to each line: 1.5 ms at the first line and
1.05 ms at the last. The ``shutter_shading`` shape defaults hold those two numbers, and
the mode scales the light by their ratio, so the last line gets 0.70 of the first
line's light at every exposure. The entry doesn't state an exclusion reason.

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

**Simulator guard scenes.** The ``full_frame_cameras`` scene class (see
:ref:`sim-full-frame-cameras`) holds three scenes of this instrument, each at the
full 800 px frame with the measured twist of -0.0526 degrees. Simulated skies of
this camera are about four times quieter than real ones, so each scene plants
the measured sky noise of 0.96 DN in place of the catalog's read noise.

- ``gossi_star_field``: two bright stars among 14 fainter ones at 1.6 s, offset
  257 px, the size of the library's scattered-light frames. Every star in view
  saturates. The navigator matches 11 stars and lands 0.09 to 0.10 px off over
  24 draws, and 0.08 px off without the twist.
- ``gossi_star_field_short_exposure``: the same field at 0.1 s, the exposure of
  the real Pleiades frames, which navigate on three stars. The navigator's star
  limit on a simulated frame reaches V 7.6, against V 7.8 on the real frames.
  It matches 8 stars and lands 0.01 to 0.02 px off over 24 draws. The real
  frames are from 1990 and read about 1.3 mag fainter than the 1992 frames the
  zero point follows, so the scene checks the short-exposure path rather than
  copying those frames.
- ``gossi_disc``: a moon 240 px across, offset 286 px. The answer lands 0.17 to
  0.18 px off with the twist, and 0.04 px off without it. The scene takes about
  20 seconds to navigate, because the search window reaches 350 px.

**Mixed-defect guard scene.** ``gossi_ict_moon`` in the ``mixed_defects`` scene
class (see :ref:`sim-mixed-defects`) is a full frame of a moon 400 px across with
the camera's chain and its fitted cosmic rays on, lossy compression, and a band of
lost 8-line compression slices. In the metadata indexes of GO_0017, GO_0019, and
GO_0023, 1,318 of 1,865 frames use the camera's lossy codec, at a median
compression ratio of 9.5. In a sample of 40 of those frames showing a moon, 34
have zero bands in multiples of 8 lines. Most of them are frames cut short at the
top or bottom, and 10 lose bands of 8 to 24 lines inside the frame or at its top
edge. The three moon-filled frames the scene follows each lost one band of 3 or 4
slices.

The simulator has no model of the lossy codec, so the scene uses the generic one
with a step of 4 DN, which leaves block edges no sharper than the inside of the
blocks, as on the real frame the disc follows. The simulator also cannot plant
exactly one band of slices that starts on any slice. The scene loses whole bands
of three slices on a 24-line grid instead, a random number of them with one
expected, so some seeds lose none and some lose two. On the committed seed one
band crosses the middle of the disc.

The scene plants a 260 px offset and the camera's twist. The fused answer lands
0.036 to 0.072 px off over 24 draws. The disc correlation lands 0.020 to
0.046 px off on every draw, including the six seeds whose band of lost slices
crosses the disc, so the slices leave the answer where it is. On one draw, whose
band covers the bottom limb, the limb fit flags its answer. The fused answer
rests on the disc correlation alone, at low confidence; the other 23 draws are
at medium. The blob technique lands up to 6 px off without flagging it when a
band crosses the disc. The twist moves the answer by at most 0.02 px. The
compression is realistic but moves the answer by at most 0.01 px, so the scene
shows that the navigator is unaffected by it.

**Unit tests.** ``tests/spindoctor/inst/test_inst_galileo_ssi.py`` pins the
limiting-magnitude form: the anchor at unit exposure, one magnitude gained per
Pogson ratio, the non-positive-exposure fallback, and finiteness. It also pins
the published clock count: a fractional number of RIM counts with no midtime or
end count, and no count at all from a label lacking one of the four items.

PDS4 hooks
==========

Not supported. The dataset leaves ``PDS4_CONFIG_KEY`` as None, so
``sd_create_bundle`` refuses it before it runs a pass, and the loader refuses a ``pds4``
entry written for it. Every PDS4 hook raises ``NotImplementedError``.

To make bundles here:

* add a template directory;
* set ``PDS4_CONFIG_KEY``, and add a ``pds4`` entry under that key;
* implement ``pds4_bundle_template_dir``, ``pds4_bundle_name``, ``pds4_bundle_version``,
  ``pds4_information_model_version``, and ``pds4_schemas``, which read the entry;
* implement ``pds4_required_templates``, which names the templates and files each
  pass takes from the template directory, and ``pds4_user_guide_file_name``, which
  names the bundle's user guide in it; and
* implement ``pds4_bundle_path_for_image``, ``pds4_path_stub``,
  ``pds4_lid_part_to_image_name``, the four LID and LIDVID builders, and
  ``pds4_template_variables``.

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
  ``read_noise_dn``, ``blank_max_dn``,
  ``noisy_threshold_dn``, and the ``mag_offset_table`` entry.
* The limiting-magnitude anchor is a nominal-optics derivation, not a
  measurement.
* ``from_file`` carries a TODO saying the extended-FOV margin branch belongs
  somewhere shared.
* The simulator PSF is unverified for want of star frames in the cohort.
