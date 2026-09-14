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

Beyond the host call it records ``obs.abspath`` and ``obs.image_url`` from an
``FCPath``, reads the flat ``newhorizons_lorri`` config section, and resolves
the extended-FOV margin from the size-keyed table by ``obs.data.shape[0]``. An
explicit ``extfov_margin_vu`` argument wins over the configuration entirely.
Last, it reads the spacecraft clock counts from the PDS3 label beside the
image; see the label fields below.

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
logged as an error naming the index file and the filespec, and drops that row
alone.

``_img_name_valid`` uppercases before testing, so a name given on the command
line may be in either case; it requires exactly fourteen characters starting
``LOR_`` followed by ten digits. ``_extract_img_number`` is those ten digits.

**Monotonicity.** ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` is left at its
``True`` default, so a ``--last-image-num`` scan stops after the first volume
entirely past the range.

Configuration block
===================

One flat section. There is one camera, so nothing is nested and the loader
reads ``config.category('newhorizons_lorri')`` directly.

Nothing departs from the common raw-DN schema. ``extfov_margin_vu`` is a
size-keyed table.

**This block uses the** ``_sources`` **convention.** It carries a ``_sources``
sub-block mirroring the shape of ``noise``, ``image_quality_thresholds`` and
``mag_offset``, with one string per value naming where that value came from and
whether it is measured or still to be measured. ``Config._load_yaml`` strips
every key beginning with ``_`` at load time, so the block has no runtime effect
at all; it exists so a reviewer can trace a number's provenance and so a
calibration pass has a checklist.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``full_well_dn``, ``expected_noise_dn``, ``read_noise_dn``, ``blank_max_dn``,
``noisy_threshold_dn`` and the ``mag_offset_table`` entry. ``full_well_dn``
being a placeholder while ``saturation_dn`` is a hard fact is the point the
``_sources`` block makes explicitly: the 12-bit ADC ceiling of 4095 DN and the
detector full well are different quantities, and only the first is known here.

Photometric and PSF calibration
===============================

**Limiting magnitude.** The Pogson-ratio form,
``anchor + log(texp) / log(2.512)``, so each factor of 2.512 in exposure buys
one magnitude of depth. The anchor is 11.7 at a 1 s exposure, and it is
**derived rather than measured**, from three terms: the project's reference
anchor of 10.5 magnitudes at a 1 s exposure for a 0.19 m aperture, scaled by
collecting area for this instrument's 0.208 m aperture
(``5*log10(0.208/0.19)``), with no detector-sensitivity penalty because this is
a CCD, plus **+1.0 magnitude for the panchromatic passband**, which collects
more flux than a filtered one. That bandpass term is the largest single
contributor to its depth. All of it is a nominal-optics estimate pending
calibration against real star fields. A non-positive exposure time falls back
to the anchor.

``star_min_usable_vmag`` is 0.0: no bright-end cutoff, and saturated stars are
handled downstream.

**PSF.** ``star_psf_sigma`` is 3.0 and ``star_psf_sizes`` is a single
``100: [7, 7]``, so every star gets a 7x7 cutout regardless of magnitude. Note
that the navigation-side sigma is a single number while the simulator's kernel
for this instrument is elliptical; the two are separate parameters serving
separate purposes and are not expected to agree.

**Magnitude offsets.** ``fallback_combo`` is ``'1'`` -- the single panchromatic
filter slot, encoded as the digit one -- and the table carries one entry with a
default of 0.0. Both are placeholders.

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 3.33e6, described
in the file as an interim value sized from the electron zero point over the
single gain state.

**Recalibrating.** The cohort's two frames for this instrument are binned four
by four, so they cannot anchor an unbinned limiting magnitude or an unbinned
PSF. Recalibration starts from acquiring unbinned star fields; with those, the
anchor is the faintest usable star at a known exposure, and the panchromatic
term stops being an estimate and becomes part of the measurement.

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
object no image ever asks about. The driver warns once in its run log naming
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
block.

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

**Artifact-mode availability.** This instrument is in the CCD set, so it
carries the CCD-only modes (``radiation_transients``, ``compression_dct``) as
well as every mode declared available to all instruments. It also carries three
further modes: ``embedded_header`` (the row-0 housekeeping header),
``frame_transfer_smear``, and ``serial_tail`` (the post-gain DN undershoot
after saturation).

Two modes record an explicit **exclusion reason** against this key:

* ``hot_pixels`` -- "explicitly disabled for LORRI, which has none".
* ``bloom`` -- "explicitly disabled for LORRI, an antiblooming CCD with no
  column bloom".

Those are physical statements rather than gaps in coverage, which is why they
are recorded as reasons rather than as absences from an availability set. A
scene that asks for either mode on this instrument is refused with the reason
quoted back.

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

**Unit tests.** ``tests/spindoctor/inst/test_inst_newhorizons_lorri.py`` pins
the limiting-magnitude form: the anchor at unit exposure, one magnitude gained
per Pogson ratio, the non-positive-exposure fallback, and finiteness. It also
pins the published clock counts: fractional seconds, a midtime count that is
the float nearest their exact mean, and counts read from the PDS3 label beside
the image.

PDS4 hooks
==========

Not supported. Two hooks are implemented --
``pds4_bundle_template_dir``, which falls back to ``newhorizons_lorri_1.0``,
and ``pds4_bundle_name``, which falls back to
``newhorizons_lorri_backplanes_rsfrench2027`` -- and every remaining hook
raises ``NotImplementedError``. Neither the template directory nor a
``pds4.nhlorri`` configuration entry is shipped; the configuration file carries
the entry as a commented-out stub.

Making bundles work here means implementing ``pds4_bundle_path_for_image``,
``pds4_path_stub``, ``pds4_lid_part_to_image_name``, the four LID and LIDVID
builders and ``pds4_template_variables``, and adding a template directory.

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
  ``full_well_dn``, ``expected_noise_dn``, ``read_noise_dn``,
  ``blank_max_dn``, ``noisy_threshold_dn`` and the ``mag_offset_table`` entry.
* The limiting-magnitude anchor, including its panchromatic bandpass term, is a
  nominal-optics derivation rather than a measurement.
* The simulator PSF describes the unbinned readout mode and the cohort holds
  only binned frames; a per-readout-mode kernel is future work.
* The camera twist is measured over one epoch cohort only. The later epochs
  fall outside the pointing-kernel coverage that was loaded and are unmeasured.
