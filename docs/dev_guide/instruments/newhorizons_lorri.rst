=====================
New Horizons LORRI
=====================

Code map
========

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Piece
     - Where
   * - Obs class
     - :class:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI`,
       registered under the instrument name ``nhlorri``
   * - DataSet class
     - :class:`~spindoctor.dataset.dataset_pds3_newhorizons_lorri.DataSetPDS3NewHorizonsLORRI`,
       registered under ``nhlorri`` with an ``nhlorri_pds3`` alias
   * - Config block
     - ``newhorizons_lorri`` in ``config_420_inst_nhlorri.yaml``, flat
   * - Sim instrument key
     - ``nhlorri``
   * - Statistics key
     - ``nhlorri``
   * - Log key
     - ``other.obs`` governs the image-scoped ``LOAD IMAGE`` section; there is
       no per-instrument logging key
   * - oops host module
     - ``oops.hosts.newhorizons.lorri``

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI.from_file`
calls ``oops.hosts.newhorizons.lorri.from_file(path, calibration=False)``.
``calibration=False`` reads the raw DN image, and the reason is worth keeping
straight: it is not that calibration is unwanted but that there is nothing to
convert. The calibrated LORRI products are themselves in DN rather than I/F, so
no I/F conversion exists to apply, and the navigation pipeline treats image
brightness scale-invariantly in any case -- normalized cross-correlation, an
image-derived noise floor, a magnitude-based star gate. No other keyword
argument is accepted, and ``**_kwargs`` is named with a leading underscore to
say so.

Beyond the host call it reads the flat ``newhorizons_lorri`` config section
and resolves the extended-FOV margin from the size-keyed table by
``obs.data.shape[0]``. An explicit ``extfov_margin_vu`` argument wins over the
configuration entirely. Last, it reads the spacecraft clock counts from the
PDS3 label beside the image; see the label fields below.

Label and index dependencies
============================

**Label fields read.** From the FITS image, nothing beyond what the base class
reads for the exposure times and the image shape. From the PDS3 label beside
the image, ``SPACECRAFT_CLOCK_START_COUNT`` and
``SPACECRAFT_CLOCK_STOP_COUNT``: the FITS header the observation is read from
carries the start count only, so
:meth:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI.from_file`
reads both from the label once with
:func:`~spindoctor.support.sclk.pds3_label_clock_counts` and keeps them for
:meth:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI.get_public_metadata`.
A count is ``SECONDS:TICKS``: whole seconds, then the 1/50000-second ticks
past them. ``_sclk_count`` converts it to an exact number of seconds through
:func:`~spindoctor.support.sclk.fractional_count`, with the moduli
``(4294967296, 50000)`` and offsets ``(0, 0)`` that the clock kernel
``new_horizons_2132.tsc`` gives, so ``0031650238:48850`` is
``31650238 + 48850 / 50000``. The two counts mark the start and the end of the
exposure, so ``_published_sclk`` passes ``bracketed=True`` to
:func:`~spindoctor.support.sclk.exposure_counts` and ``midtime_sclk`` is their
exact mean. Where the label carries no such count, or there is no label beside
the image, the count is published as ``None``, and so is the mean.

:meth:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI.get_public_metadata`
writes ``filters`` as an empty list: the camera is panchromatic with no filter
wheel, so there is no filter name to record.

**Index columns.** ``_INDEX_COLUMNS`` is ``FILE_SPECIFICATION_NAME``.
``_INDEX_CAMERA_COLUMNS`` is ``('INSTRUMENT_ID',)`` and ``_INDEX_CAMERA_MAP``
is ``{'LORRI': 'LORRI'}`` -- an identity map, present so that the camera an
image is attributed to comes from the same mechanism for every instrument
rather than being special-cased for a single-camera one.

**Filespec parsing.** This archive's names are **lowercase**, and every parser
here matches lowercase deliberately. ``_get_label_filespec_from_index``
requires the index value to end ``_sci.lbl`` or ``_eng.lbl`` and passes it
through unchanged; both the science and the engineering products are navigated.
``_get_image_filespec_from_label_filespec`` swaps ``.lbl`` for ``.fit``, since
the image is FITS rather than a PDS3 raster.

``_get_img_name_from_label_filespec`` requires exactly three levels, a leading
``DATA``, and a 15-character range directory with ``_`` at index 8 -- two
seven-digit request identifiers joined by an underscore. The image name is the
**leading fourteen characters** of the file name, which drops the readout-mode
component and the ``_sci`` / ``_eng`` suffix.

The two parsers fail differently, which is worth knowing when a run reports a
short image count. A suffix ``_get_label_filespec_from_index`` rejects
propagates and stops the run; a structure
``_get_img_name_from_label_filespec`` rejects is caught by the index scan,
logged as an error that gives the index file and the filespec, and drops that row
alone.

``_img_name_valid`` uppercases before testing, so a name given on the command
line may be in either case; it requires exactly fourteen characters starting
``LOR_`` followed by ten digits. ``_extract_img_number`` is those ten digits.

**Monotonicity.** ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` is left at its
``True`` default, so a ``--last-image-num`` scan stops after the first volume
entirely past the range.

Configuration block
===================

One flat section. There is one camera, so nothing is nested and the
observation reads ``config.camera_block('newhorizons_lorri')``.

Nothing departs from the common raw-DN schema. ``extfov_margin_vu`` is a
size-keyed table.

**This block uses the** ``_sources`` **convention.** It carries a ``_sources``
sub-block mirroring the shape of ``noise``, ``image_quality_thresholds`` and
``mag_offset``, with one string per value saying where that value came from and
whether it is measured or still to be measured. The configuration loader drops
every key beginning with ``_`` as it reads each file (see
:ref:`config-merge`), so the block has no runtime effect at all; it exists so a
reviewer can trace a number's provenance and so a calibration pass has a
checklist.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``read_noise_dn``, ``blank_max_dn``,
``noisy_threshold_dn``, and the ``mag_offset_table`` entry.

Photometric and PSF calibration
===============================

**Limiting magnitude.** Photometric, from the camera block's ``star_photometry``
(see :ref:`config-camera-blocks`), whose rate counts electrons. The camera has no
filter, so the block has no ``filter_mag_offset`` and none is read, and it has one gain
state, but its binning mode changes the electrons per DN, the noise of a pixel, and the
width of a star in pixels, so the gain table and both mode tables are keyed by the
mode, ``1X1`` or ``4X4``. The mode follows from the row count,
as the image host reads it: 1024 rows is ``1X1`` and 256 rows is ``4X4``, and any
other row count is refused. A star of V magnitude ``m`` puts

.. math::

    \mathrm{DN} = \frac{R \cdot 10^{-0.4\,m} \cdot t_{\mathrm{exp}}}{g}

in the image, where ``R`` is ``flux_per_s_vmag0`` (7.57e8 electrons per second for a
V = 0 star) and ``g`` is the mode's ``gain`` entry, in electrons per DN. The dimmest
visible star puts the mode's ``min_detectable_dn`` in the image, and the limit is
that star's magnitude plus ``search_depth_mag`` (1.0). At 1 s the limit is V 14.8
unbinned and V 15.5 binned. A frame whose mode the tables leave out is refused, with
the key path (``newhorizons_lorri.star_photometry...``) in the message, and a frame
whose exposure is not positive gets a limit of ``-inf``.

The rate, the detection thresholds, and the star widths were measured in 2026 on star
fields of the cruise and the Pluto approach, read through the navigator's own reader.
Each frame's pointing was found from the stars alone, by a vote over star and peak
pairs refined to a translation and a rotation, and a frame was kept when three or more
stars, each on its own image peak, agreed within 1 px and the chance of such a cluster
anywhere in the search window (100 px unbinned, 40 px binned) was below one in a
thousand. That kept 341 of 468 measured frames, and one frame in three was held out of
the fit. Frames shorter than 50 ms were left out: the camera has no shutter, so the
frame transfer adds light to a short exposure.

- The rate is the fit of 4,310 stars, 3.605e7 DN per second for V = 0 unbinned
  (+-0.01 mag), times 21.0 electrons per DN. The zero point of Weaver et al. (2020)
  gives 3.2e7 DN per second.
- The electrons per DN are those of Weaver et al. (2020), 21.0 unbinned and 19.4
  binned. The stars measure the binned to unbinned DN ratio at 1.054, against the
  1.082 these give.
- ``min_detectable_dn`` is where a detection-probability curve, fitted to every
  clean star of the mode, reaches 80% of the way from its floor to its plateau:
  110 +- 7 DN unbinned and 60 +- 4 DN binned.

On the held-out frames, the faintest star detected lies a median 0.04 mag past the
predicted dimmest visible star.

``star_min_usable_vmag`` is 0.0: no bright-end cutoff, and saturated stars are
handled downstream.

**PSF.** The navigator's star sigma is ``star_photometry.psf_sigma`` for the frame's mode,
in that frame's own pixels. Unbinned it is 0.9 px, the median of 311 stars on 45 frames
(16th to 84th percentile 0.82 to 0.97); the FWHMs of Weaver et al. (2020), 2.04 by 2.70
px, are sigmas of 0.87 by 1.15. Binned it is 0.45 px, from the 5 to 10 s frames a
navigation run sees; sub-second star calibrations give 0.28, since pointing jitter widens
a long exposure. The camera block's own ``star_psf_sigma`` holds the unbinned value for
the simulator, which reads it there in unbinned pixels and takes a quarter of it, 0.225
px, on a simulated binned frame, so the simulator and the navigator agree only on unbinned
frames. ``star_psf_sizes`` is a single ``100: [7, 7]``, so every star gets a 7x7 cutout
regardless of magnitude. The simulator's kernel for this instrument is elliptical; the two
are separate parameters serving separate purposes.

**Magnitude offsets.** ``fallback_combo`` is ``'1'`` -- the single panchromatic
filter slot, encoded as the digit one -- and the table carries one entry with a
default of 0.0. Both are placeholders.

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 3.33e6, described
in the file as an interim value sized from the electron zero point over the
single gain state.

**Binned readout mode.** ``noise.by_binning`` gives the values for the 4x4 mode,
keyed by the binning factor. The mode's published gain is 19.4 e-/DN, so the
read noise of one binned pixel is the published 23 e- over 19.4, or 1.19 DN, and
the zero point is the electron zero point of 7.0e7 e- per second over 19.4, or
3.61e6 DN per second. Only the faint-star limit of a simulated frame reads them.
The full-resolution ``read_noise_dn`` is a placeholder above the published 23 e-
over 21 e-/DN, or 1.10 DN, so 1.4 of the 4.4 mag step from an unbinned frame's
star limit to a binned one's comes from the placeholder.

**Recalibrating.** The measurement tools are in ``util/star_photometry``; its LORRI
frame selection and the fit are described in its module docstrings. Rerun the fit on
new star fields and update the camera block's ``star_photometry`` values and their
comments together.

Frames, attitude, and rotation fitting
======================================

**Camera frame.** ``NH_LORRI``.

**CK object and clock.** -98000, the spacecraft, whose time tags are encoded
against spacecraft clock -98. Note that the corrected object is the spacecraft
itself rather than a platform: this instrument is body-mounted and the
spacecraft turns to point it.

**The oops-from-SPICE flip.** ``R = diag(+1, -1, -1)``, applied by
``oops/hosts/newhorizons/lorri.py``, because the SPICE boresight is -Z where
the oops convention wants +Z. The correction is built in the oops observation
frame and must be conjugated through ``R`` before it is composed onto a
``pxform``-derived matrix. ``R`` is measured at runtime as
``C_oops . cmatrix_original^T``, checked against this constant, and checked
again for being the same at the exposure start and stop.

**Frame evaluation.** The observation frame is evaluated, not frozen: a
``pxform`` at the exposure midtime reproduces what the host built.

**Per-spacecraft variation.** None. One spacecraft, one camera, one CK object
and one clock.

**Rotation fitting.** ``fit_camera_rotation`` is ``false``, with
``max_rotation_deg: 5.0`` carried but unused. That is the measured setting: the
twist is large but static, at +0.191 degrees with a corner scatter of only
0.027 pixels, so a fixed frame-kernel correction is the right instrument for it
and a per-frame fit would be finding one number over and over. Because rotation
fitting is off, every image of this instrument is eligible for a corrected
C-kernel.

C-kernel specifics
==================

**Baseline structure.** Ordinary time-varying type-3 kernels for object -98000,
and a corrected segment composes a body-fixed ``delta`` onto the baseline's own
attitude at each record epoch.

**Angular-velocity census.** All 4346 -98000 segments in the local baselines
carry angular velocity, so the refusal for a baseline that supplies pointing
without a rate is unreachable for this mission against those holdings.

**Kernel-name class rules.** ``_NEW_HORIZONS_NAME_RULES`` in
:mod:`spindoctor.cli.ck.index` carries two patterns matched in full against the
lowercased basename:

.. code-block:: text

   nh_.+_recon\.bc     -> RECONSTRUCTED
   nh_.+_pred\.bc      -> PREDICTED

Only the pair of kernels that exist in both forms declares a class. Every other
name in the holdings -- the merged pointing files and the hazard-search kernels
-- is left ``UNCLASSIFIED`` rather than guessed at from a prefix that says which
product a kernel is and not how its pointing was made. The tie-break among
reproducing candidates prefers reconstructed over predicted, then the
lexicographically greatest basename, then the greatest path.

**The phantom object.** This mission's merged pointing files describe an object
**-1** beside -98000, and no furnished kernel defines a spacecraft clock for
it. The index reports coverage in TDB, which needs that clock, so -1 is
recorded as unreadable and offers no coverage rather than stopping the scan.
Refusing there would make the whole mission unindexable for the sake of an
object no image ever asks about. The driver warns once in its run log, listing
the skipped objects, so a kernel set genuinely missing a clock still says so;
an image that actually corrected such an object is refused with the missing
clock named.

**Deviations in segment construction.** None. The object is not in
:data:`~spindoctor.spice_ids.FROZEN_ATTITUDE_CK_IDS`, so it takes the standard
time-varying path: records at start, midtime and stop plus a 1 s cadence past
10 s, the baseline's angular-velocity vectors copied bit-identically, and
``avflag = 1``.

**Rigid-rotation residual.** An exact rigid rotation is not exactly a uniform
tangent-plane shift, and the difference is measured over a 17x17 grid across
the full frame, worst case over eight offset directions, at 50 pixels of total
boresight displacement: 1.62e-8 rad, which is 8.15e-4 tangent-plane pixels and
1.23e-3 pixels in pixel space. It is linear in the offset, so quoting it
without the offset it was measured at means nothing.

**Reproduction path.** ``cspyce.pxform('J2000', 'NH_LORRI', midtime_et)``
against each furnished candidate, accepted at 1e-9 radians.

Simulator model
===============

**Instrument key.** ``nhlorri``, resolving to the flat ``newhorizons_lorri``
block. The ``nhlorri`` entry in
``src/spindoctor/sim/instruments/configs/nhlorri.yaml`` holds the values below.

**PSF kernel.** ``{sigma_v: 1.13, sigma_u: 0.87, w: 1.2e-2, r0: 2.0, n: 3.0}``,
**retained interim elliptical values** from the published references for
unbinned frames. The kernel is elliptical. The cohort's two star frames for
this instrument are binned four by four -- their measured binned-pixel
encircled-energy radius at 50% is 0.59 pixels -- and cannot constrain an
unbinned kernel, so a per-readout-mode kernel is future work rather than a
refinement of this one.

**Distortion residuals.**
``{k1: 8.13e-04, k2: -1.10e-03, nonradial_rms_px: 0.0}``, measured by the
star-field distortion analysis. The radial term is a mid-field hump close to
the measurement floor; this camera's significant geometric signature is the
twist, not the distortion.

**Artifact-mode availability.** The instrument entry's ``artifact_modes.offered``
lists its modes, and the simulator guide's mode tables
(:ref:`sim-telemetry-modes` and :ref:`sim-detector-modes`) give each mode's
cameras. Its detector model is ``ccd``. The entry lists ``frame_transfer_smear``,
``banding``, and ``bias_structure`` under ``artifact_modes.instrument_defaults``, so
``instrument_defaults`` turns them on: the smear at its measured transfer times
(below), and the banding as the camera's two published patterns. The first is a
random horizontal striping of at most 1 DN that moves from frame to frame (Weaver et
al. 2020): each row takes its own offset, drawn evenly from within 21 e- of zero.
The second is vertical banding near the low column numbers, which Cheng et al.
(2008) trace to an oscillation of the amplifier and put at about 0.8 DN from peak to
trough. It is drawn as a damped wave that starts at the first column and is the same
in every frame. Its period of 103 native columns and its decay to 1/e over 71 are read
from the bias frame Weaver et al. show, so both are estimates. A binned frame
column takes the mean of the wave over the native columns it holds. The entry's
shape defaults give ``embedded_header`` a 34-pixel header, the binary housekeeping at the
start of row 0 of every image.

**Frame transfer and integration time.** Both were measured on raw frames. In a raw
frame, each row above or below a bright source carries a share of its column's
charge, the frame-transfer smear. That share is a fixed transfer time spread over
1,024 rows and divided by the time the frame collects light. The fit uses 14
unsaturated full-resolution frames of Io, Europa, Ganymede, and Callisto, taken at 3
to 16 ms, whose smear is 34 to 510 DN in each row. It gives a mean transfer time of
11.66 ms and 0.63 ms of collected light beyond the commanded exposure, with
residuals of at most 0.09 ms. Leaving out any one frame moves the extra time between
0.61 and 0.64 ms. The entry states the extra time as ``extra_integration_sec``.

Four Charon frames taken at 100 and 150 ms carry only 2 to 5 DN of smear in each
row, so a sky error of 0.1 DN moves them by 2% to 5%. They read 1% to 3% above the
fit. Fitting them too gives an extra time of 0.70 ms and a transfer time of 11.83 ms,
so the extra time lies between 0.6 and 0.7 ms.

Three pairs of Europa frames taken at 5 and 10 ms show the same smear in DN, within
0.4%, while the exposure doubles. So the transfer time stays fixed, and the extra
time is light the frame collects. The 10 ms frames, measured on their unsaturated
edge columns and left out of the fit, match it within 0.2%. The pipeline's desmear
times for short exposures show the same effect: the SOC-to-Instrument Pipeline ICD
gives 7.1, 8.75, 9.65, and 10.5 ms at 1, 2, 3, and 6 ms, and the fit gives 7.16,
8.87, 9.64, and 10.55 ms. The ICD's 10.7 ms for nominal exposures is below the
11.6 ms that the fit gives at 100 ms, and the reason is not known.

The rows above a source, toward line 0, carry the larger share. On the 14 frames it
is 1.090 to 1.093 times the share of the rows below, and on the four Charon frames
1.088 to 1.096. Cheng et al. (2008) give the two transfers as 1.044 and 0.956 of
their mean, a ratio of 1.092, which splits the mean into 12.17 ms above the source
and 11.15 ms below it. The simulator draws ``t_transfer_sec`` above the source, so
the entry gives it the larger time, although Cheng et al. attribute the larger time
to the scrub. A simulated 5 ms frame of a bright disc carries 2.106e-3 of its
column's charge in each row above the disc and 1.936e-3 in each row below it. The
real frame lor_0034849319 carries 2.115e-3 and 1.937e-3.

The entry states an explicit **exclusion reason** for three modes:

* ``hot_pixels`` -- "explicitly disabled for LORRI, which has none".
* ``shutter_shading`` -- "not offered for LORRI, a frame-transfer camera with no
  shutter".
* ``bloom`` -- "explicitly disabled for LORRI, an antiblooming CCD with no
  column bloom". A saturated LORRI source therefore never bleeds along its column,
  also under ``instrument_defaults``; its ``serial_tail`` mode draws the
  undershoot along the line instead.

Those are physical statements rather than gaps in coverage, which is why they
are recorded as reasons rather than as absences from the list. A
scene that asks for one of these modes on this instrument is refused with the reason
quoted back.

**Cosmic rays.** The catalog rate is a retained zero. The cohort's measured
3.36e-4 transient fraction comes from two 4x4-binned star frames, where a binned
faint star can't be told from a single-pixel transient, so it can't be adopted
as a rate until a non-star frame joins the library.

**Realism-match status.** Unverified for the unbinned readout mode, which is
what the kernel describes. The binned frames the cohort holds are measured but
cannot check it.

Image library and test coverage
===============================

**Cohort.** Two sidecars, both under ``star_dominated``, and both binned four
by four, which is why the photometric and PSF parameters above are all
derivations rather than measurements.

**Integration tests.** The per-image regression suite
(``tests/integration/test_autonomous_nav.py``) and the structural-invariants
test (``test_image_library.py``) run on those sidecars.
``tests/integration/test_cmatrix_frames.py`` measures the oops-from-SPICE flip
on a real frame of this instrument and checks it against ``diag(+1, -1, -1)``.
``tests/integration/test_ck_round_trip.py`` includes a frame of this instrument
in its round-trip cohort: it is navigated, its corrected kernel is written, and
it is re-navigated in a fresh process with that kernel furnished.

**Simulator guard scenes.** The ``full_frame_cameras`` scene class (see
:ref:`sim-full-frame-cameras`) holds two scenes of this instrument, each at the
full 1024 px frame with the measured twist of +0.19 degrees. The image library's
frames of this camera are only about a pixel off, so the scenes' 32 px offset
instead reaches half the camera's search window.

- ``nhlorri_star_field``: 18 stars of V 10.5 to 13.0 at 5 s. The navigator
  predicts all 18 visible and matches them. A translation follows the turned
  stars it measures, so the twist moves the answer: on the committed draw it lands
  0.40 px off.
- ``nhlorri_star_field_short_exposure``: the same field at 1 s. The navigator
  predicts the seven brightest stars visible, the star-field technique locks on
  all seven, and on the committed draw the answer lands 0.56 px off.

**Mixed-defect guard scene.** ``nhlorri_moon_raw`` in the ``mixed_defects`` scene
class (see :ref:`sim-mixed-defects`) is a raw unbinned frame of a moon 197 px
across, following a real Jupiter-flyby frame taken at 5 ms. It carries the
camera's chain, including the frame-transfer smear, plus the housekeeping header
at the start of row 0, the fixed pattern, and radiation hits. Simulated brightness
has no shared scale with real frames, so the scene runs 85 ms, where the middle of
the simulated disc is 8% brighter than the real one. The rate of radiation hits is
scaled down by the same factor, and about 16 hits land in the frame. The scene sets
its own smear times, so that each row above and below the disc carries the share of
its column's charge that the real frame shows: 2.115e-3 above and 1.937e-3 below.
With the camera's extra 0.63 ms the frame collects light for 85.63 ms, and the
times that give those shares are 0.1855 s above the disc and 0.1698 s below it.
The simulated rows carry 2.113e-3 and 1.934e-3. The extra time stays 0.63 ms
rather than growing with the exposure, since it belongs to the camera. The
simulated header holds values up to 255, where the real one reaches 4,038 DN.

The fused answer lands 0.11 to 0.12 px off over 24 draws, at medium confidence,
from the disc correlation and the limb fit together. The disc correlation lands
0.18 to 0.19 px off without flagging its answer, with the smear or without it.
Without the twist it lands 0.004 to 0.022 px off. The limb fit lands 0.25 to
0.32 px off, and 0.43 to 0.48 px without the twist. The blob technique lands 6.8
to 7.1 px off without flagging it, pulled by the smear columns, and 0.16 to 0.19 px off
without the smear. The simulated smear is a step at each column's centroid, where
the real smear builds up across the disc, so part of that pull may be the
simulator's. The fixed pattern and the radiation hits are realistic, and each
moves the fused answer by at most 0.01 px, so the scene shows that the navigator
is unaffected by them. The simulated header also moves it by at most 0.01 px,
but its values reach only 255 DN where the real header reaches 4,038 DN, so the
scene shows only that the navigator ignores this approximation of the header.

**Unit tests.** ``tests/spindoctor/inst/test_inst_newhorizons_lorri.py`` pins
the star limit of each binning mode at 1 s, 2.5 mag per factor of ten in exposure,
``-inf`` for a non-positive exposure, the refusal of a row count other than
1024 or 256, and each mode's star sigma. It also
pins the published clock counts: fractional seconds, a midtime count that is
the float nearest their exact mean, and counts read from the PDS3 label beside
the image.

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

**Backplanes.** Nothing instrument-specific. The stage reads the recorded
C-matrix through the shared pointing ladder, and since rotation fitting is off
for this instrument, its records carry a usable C-matrix and take the matrix
path rather than the offset fallback.

**Mosaics.** Nothing instrument-specific.

**Statistics.** Nothing particular.
:func:`~spindoctor.cli.stats.report_sections.resolve_offset_limit` reads
``newhorizons_lorri.extfov_margin_vu`` from the flat section and indexes the
size-keyed table with the recorded image height, so a database row with no
recorded shape resolves to an explanatory string instead of a limit. The
report's image-name rule is the leading fourteen characters of the
extension-stripped stem, so ``lor_0003103486_0x630_sci.fit`` reduces to
``lor_0003103486``.

Open items
==========

* Every value marked ``# PLACEHOLDER`` in ``config_420_inst_nhlorri.yaml``,
  each with its own ``_sources`` line saying what has to be measured:
  ``read_noise_dn``,
  ``blank_max_dn``, ``noisy_threshold_dn``, and the ``mag_offset_table`` entry.
* At the longest exposures the binned star limit passes V 17, deeper than the star
  catalog is complete, and the star model's cap on the number of stars keeps only the
  brightest of them.
* The simulator PSF describes the unbinned readout mode and the cohort holds
  only binned frames; a per-readout-mode kernel is future work.
* The camera twist is measured over one epoch cohort only. The later epochs
  fall outside the pointing-kernel coverage that was loaded and are unmeasured.
