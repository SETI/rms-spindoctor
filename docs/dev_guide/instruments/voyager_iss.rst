============
Voyager ISS
============

Code map
========

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Piece
     - Where
   * - Obs class
     - :class:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS`, registered
       under the instrument name ``vgiss``
   * - DataSet class
     - :class:`~spindoctor.dataset.dataset_pds3_voyager_iss.DataSetPDS3VoyagerISS`,
       registered under ``vgiss`` with a ``vgiss_pds3`` alias
   * - Config block
     - ``voyager_iss`` in ``config_430_inst_vgiss.yaml``, flat
   * - Sim instrument key
     - ``vgiss``
   * - Statistics key
     - ``vgiss``
   * - Log key
     - ``other.obs`` governs the image-scoped ``LOAD IMAGE`` section; there is
       no per-instrument logging key
   * - oops host module
     - ``oops.hosts.voyager.iss``

One instrument key serves two spacecraft and four cameras. Everything that has
to distinguish them derives the distinction from the image's own label rather
than from configuration, which is why the frame identities below are computed
per image.

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS.from_file` calls
``oops.hosts.voyager.iss.from_file(path)`` with no keyword arguments.

It also **modifies the pixel data**, in two steps:

1. **I/F scaling.** ``_voyager_if_factor`` reads the ``LABEL3`` VICAR record,
   which carries the fixed phrase ``FOR (I/F)*10000., MULTIPLY DN VALUE BY``
   followed by a number, and the data become ``data * factor / 10000``. A
   ``LABEL3`` that is not a string, does not carry the phrase, or whose
   remainder will not parse as a float raises ``ValueError`` and the image does
   not load. Refusing is right here: silently skipping the scaling would
   produce a frame in the wrong units that every threshold downstream would
   then judge.
2. **The Voyager 1 Saturn correction.** ``_voyager_spacecraft_digit`` reads the
   ``LAB02`` record, whose fifth character is the spacecraft digit, and when
   that digit is ``1`` and ``obs.planet.upper()`` is ``SATURN`` the data are
   multiplied by a further **3.345**. The archive's calibration pipeline
   computed I/F for Voyager 1 as though every image had been taken at
   Jupiter's heliocentric distance, so the Saturn frames come out too dim by
   the square of the distance ratio. Voyager 1 visited only Jupiter and
   Saturn, and Voyager 2 was calibrated correctly at each of its encounters,
   so this is the only case, and the constant is written out rather than
   computed from distances. The step logs at debug when it fires.

Beyond that it records ``obs.abspath`` and ``obs.image_url`` from an
``FCPath``, reads the flat ``voyager_iss`` config section, and resolves the
extended-FOV margin from the size-keyed table by ``obs.data.shape[0]``. That
table has exactly one entry, for size 1000, which is what a geometrically
corrected product is; an image of any other size raises ``KeyError`` from the
margin lookup rather than loading with a wrong margin. Last, it reads the
spacecraft clock counts from the PDS3 label beside the image, which the VICAR
label the observation keeps does not carry; see the label fields below.

A TODO on the calibration block records that this arithmetic belongs in the
host once the host grows it.

Label and index dependencies
============================

**Label fields read.** Two of these are load-blocking:

* ``LABEL3`` -- the I/F scaling factor. Missing or malformed stops the load.
* ``LAB02`` -- the spacecraft digit, at index 4. Not a string, shorter than
  five characters, or a digit other than ``1`` or ``2`` stops the load. This
  one value decides the SPICE frame names, the CK object, the spacecraft clock
  and both metadata LIDs, so it is validated where it is read rather than
  where it is used.
* ``filter`` -- through the base class's property, for the single ``filters``
  entry.
* ``SPACECRAFT_CLOCK_START_COUNT`` and ``SPACECRAFT_CLOCK_STOP_COUNT`` -- from
  the PDS3 label beside the image, since the VICAR label the observation keeps
  does not carry them.
  :meth:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS.from_file` reads
  them once with :func:`~spindoctor.support.sclk.pds3_label_clock_counts` and
  keeps them for
  :meth:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS.get_public_metadata`.
  A count is ``LEADING:FRAME:LINE``: the leading field counts 48-minute units,
  the frame field the 60 frames of 48 seconds in one, and the line field,
  counted from 1, the 800 lines of 60 milliseconds in a frame. ``_sclk_count``
  converts it to an exact number of leading units through
  :func:`~spindoctor.support.sclk.fractional_count`, with the moduli
  ``(65536, 60, 800)`` and offsets ``(0, 0, 1)`` that the two spacecraft's
  clock kernels, ``vg100042.tsc`` and ``vg200041.tsc``, both give, so
  ``34461:39:672`` is ``34461 + 39 / 60 + (672 - 1) / (60 * 800)``. The start
  count lies near the shutter opening, but the stop count is that of the frame
  the image was read out in, with the line field ``001``, which can come
  minutes after the shutter closed. The two do not bracket the exposure, so
  ``_published_sclk`` passes ``bracketed=False`` to
  :func:`~spindoctor.support.sclk.exposure_counts` and ``midtime_sclk`` is
  always ``None``. Where the label carries no such count, or there is no label
  beside the image, the count is published as ``None``.

:meth:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS.get_public_metadata`
refuses a detector that is neither ``NAC`` nor ``WAC``, because the instrument
LID encodes the camera and a malformed LID must never reach a PDS4 label. The
``spacecraft_digit`` property re-reads ``LAB02`` rather than caching it, so any
consumer that needs the spacecraft gets the same validation.

**Index columns.** ``_INDEX_COLUMNS`` is ``FILE_SPECIFICATION_NAME``.

``_INDEX_CAMERA_COLUMNS`` is ``('INSTRUMENT_NAME',)``, because the Voyager
indexes carry no ``INSTRUMENT_ID`` column. The keys of ``_INDEX_CAMERA_MAP``
are correspondingly prose:

.. code-block:: python

   {'NARROW ANGLE CAMERA': 'NAC', 'WIDE ANGLE CAMERA': 'WAC'}

Values are upper-cased and stripped before the lookup, and an unmapped value is
reported as unknown rather than passed through.

**Filespec parsing.** ``_get_label_filespec_from_index`` requires the index
value to end ``.LBL`` and passes it through unchanged.

``_get_img_name_from_label_filespec`` requires exactly three levels, a leading
``DATA``, and an 8-character range directory beginning ``C``. Then comes the
product filter that shapes the whole dataset: a name not ending
``_GEOMED.LBL`` returns ``None`` and is **skipped silently**. Each volume
carries ``_RAW``, ``_CALIB`` and ``_GEOMED`` products per frame, so two thirds
of the index rows are dropped this way by design, and an image count taken from
a volume listing will not match the number of images a run considers.

``_img_name_valid`` is permissive, because users list product file names rather
than image names: it upper-cases, strips anything from the first ``.`` and then
anything from the first ``_``, and validates the remaining ``Cddddddd`` core.
So ``C1234567``, ``C1234567_GEOMED``, ``C1234567_CALIB`` and
``C1234567_GEOMED.IMG`` all validate to the same image. ``_extract_img_number``
is the seven digits after the ``C``.

**Monotonicity.** ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` is set to ``False``.
Flight Data Subsystem counts restart per spacecraft and per encounter; the
volume order interleaves the two spacecraft, and Voyager 2's Neptune counts
roll over below Voyager 1's Jupiter counts. An image-number range can therefore
match frames in any volume and no volume-level early exit is possible, at the
cost of scanning every requested volume.

Configuration block
===================

One flat section, shared by two spacecraft and four cameras. Nothing in it
varies per camera.

The departure from the common schema is a units mismatch that is worth
understanding rather than tidying. ``data_units`` is ``calibrated_if``, and the
``image_quality_thresholds`` are correspondingly ``blank_max_if`` and
``noisy_threshold_if``. But the ``noise`` block is a raw-DN block:
``saturation_dn: 255``, ``full_well_dn: 255``, ``expected_noise_dn``,
``read_noise_dn`` and ``star_flux_dn_per_s_vmag0`` are all DN quantities. The
DN values describe the detector, which is real; the pipeline navigates the
scaled I/F product, so ``NavOrchestrator._build_saturation_mask`` returns an
empty mask for this instrument as for any ``calibrated_if`` input, and
``saturation_dn`` is never consulted. There is no ``saturation_threshold_if``
and adding one would not restore the mask: the mask is gated on the units, not
on the presence of the key. There is no raw product this pipeline can navigate
instead.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``expected_noise_dn``, ``read_noise_dn``, ``blank_max_if``,
``noisy_threshold_if`` and the ``mag_offset_table`` entry. The ``_sources``
convention is not used in this file.

``fit_camera_rotation: false`` carries a three-line comment saying it is kept
off because rotation fitting is too slow, that the instrument does carry
non-negligible attitude rotation residuals, and that it should be revisited
once the rotation search is fast enough. That comment is the whole rationale
and should move with the flag if the flag moves.

Photometric and PSF calibration
===============================

**Limiting magnitude.** The Pogson-ratio form,
``anchor + log(texp) / log(2.512)``, with a **per-camera anchor** selected from
the oops detector. Both are **derived rather than measured**, from the
project's reference anchor of 10.5 magnitudes at a 1 s exposure for a 0.19 m
aperture, scaled by collecting area and then penalized for the detector:

.. code-block:: text

   NAC: 10.5 + 5*log10(0.176/0.19) - 2.0 (vidicon)  ~= 8.3
   WAC: 10.5 + 5*log10(0.057/0.19) - 2.0 (vidicon)  ~= 5.9

The -2.0 vidicon term is there because a vidicon is roughly two magnitudes less
sensitive than a CCD. Combined with the wide angle camera's small aperture, it
puts the wide angle anchor at 5.9, which is why star navigation locks on so few
of these frames. All of it is a nominal-optics estimate pending calibration
against real star fields. A non-positive exposure time falls back to the
anchor.

``star_min_usable_vmag`` is 0.0: no bright-end cutoff, and saturated stars are
handled downstream.

**PSF.** ``star_psf_sigma`` is 3.0 and ``star_psf_sizes`` is a single
``100: [7, 7]``, so every star gets a 7x7 cutout regardless of magnitude and
regardless of camera.

**Magnitude offsets.** ``fallback_combo`` is ``'CL'`` and the table carries one
entry with a default of 0.0. Both are placeholders.

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 3.0e3, and its
comment records why the units are DN rather than electrons: a vidicon has no
electron domain, so it renders point sources directly in DN.

**Recalibrating.** The cohort holds one star frame for this instrument, and the
geometric resampling the navigated product has been through defeats a kernel
fit made through it -- eight usable cutouts survive, and the flat-top guard
leaves only two simulated cutouts to compare against. Recalibration therefore
needs more star frames, and per-camera anchors need star frames on each camera
of each spacecraft.

Frames, attitude, and rotation fitting
======================================

**Camera frames.** Built per image as ``f'VG{digit}_ISS{obs.camera[0]}A'``,
giving ``VG1_ISSNA``, ``VG1_ISSWA``, ``VG2_ISSNA`` and ``VG2_ISSWA``. The
Voyager frame kernel spells the cameras ``ISSNA`` and ``ISSWA``, so the oops
detector names ``NAC`` and ``WAC`` contribute only their first letter -- an
abbreviation that happens to work and is worth not "fixing".

**CK object and clock.** Read from
:data:`~spindoctor.spice_ids.VOYAGER_CK_OBJECT_ID` by the spacecraft digit:
-31100 for Voyager 1's scan platform and -32100 for Voyager 2's. The clock then
follows from the object through
:data:`~spindoctor.spice_ids.CK_OBJECT_SCLK_ID`, giving -31 and -32, rather
than from the same digit -- so a wrong pairing is refused instead of producing
a self-consistent wrong one. The clock is **not derivable by arithmetic**:
``-31100 // 1000`` is -32 in Python, which is the other spacecraft. That is
why both tables are written out.

**The oops-from-SPICE flip.** The identity, by construction rather than by
measurement, for the reason in the next paragraph.

**Frame evaluation: frozen, not evaluated.** ``oops`` does not build this
observation frame from a frame chain. It builds it as

.. code-block:: text

   P . ckgp(ck_id, sce2t(scid, et_mid), 800 + texp/48, 'J2000')
   P = pxform('VGn_SCAN_PLATFORM', camera_frame, 0)

-- a single **tolerance-snapped** pointing lookup, frozen and
time-independent, composed with a constant rotation read at epoch zero. A
``pxform`` at the exposure midtime does not reproduce it, and in fact does not
resolve at all, which an integration test pins.

Everything downstream follows from that. ``_frame_identity`` sets
``frozen_oops_attitude=True``, and ``_attitude_baseline`` takes the frozen
branch: ``cmatrix_original`` is the observation frame's own attitude, the flip
is the identity by construction, and neither the flip check nor the
across-the-exposure constancy check runs, because there is nothing evaluated to
compare. The tick conversion in the lookup is ``sce2t``, not ``sce2c``; the two
differ, and every step that reproduces this lookup matches the call it
reproduces.

**Per-spacecraft variation.** Two spacecraft, two CK objects, two clocks, four
camera frames -- all derived per image from one label character. Nothing in
configuration names a spacecraft.

**Rotation fitting.** ``fit_camera_rotation`` is ``false``, and that is **not**
because there is nothing to fit. The distortion analysis measures a
frame-varying twist on both Voyager 2 cameras, with a corner scatter of 0.28
pixels, well above the threshold at which a twist counts as one common value,
and the wide angle mean twist is +0.36 degrees, 4.4 pixels at the corner. No
static kernel removes a frame-varying twist. The flag is off for cost, and the
config comment says so.

The flag also decides C-kernel eligibility: leaving rotation fitting off is
what **keeps** this instrument's images eligible for corrected kernels. Turning
it on would make every image ``rotation_unsupported`` and stop the mission's
kernels being produced. Anyone enabling it should expect that trade and decide
it deliberately.

C-kernel specifics
==================

**Baseline structure.** Ordinary type-3 kernels for objects -31100 and -32100,
but a corrected segment does not compose onto them; it takes the frozen path
below.

**Segment construction: the frozen path.** The CK objects are the members of
:data:`~spindoctor.spice_ids.FROZEN_ATTITUDE_CK_IDS`, and
:func:`~spindoctor.cli.ck.segment.build_segment` branches on that. A frozen
segment carries the corrected midtime attitude repeated at every record epoch,
with an angular velocity of exactly zero and ``avflag = 1``. It **never reads
the baseline's attitude history at all**: no ``delta``, no per-epoch baseline
lookup, no coverage read at the record epochs.

Three consequences follow that are easy to get wrong:

* The zeros are a measurement, not an invention. A constant attitude's angular
  velocity is zero. They are written explicitly rather than declared absent
  because ``avflag = 0`` makes SPICE skip the segment for ``ckgpav`` and
  ``sxform`` and answer from the uncorrected original instead.
* The baseline's own angular-velocity vectors are **not** copied. The
  rigid-attachment argument that licenses copying them does not hold for a
  segment that deliberately drops the baseline's time variation.
* The angular-velocity census is therefore irrelevant for this instrument. None
  of the nine -31100 and -32100 segments in the local baselines carries angular
  velocity, and that changes nothing.

**Reproduction path.** Also the frozen one, and it is made twice.
:mod:`spindoctor.cli.ck.assignment` reproduces the observation frame the way
``oops`` built it -- ``ckgp`` at a snapped tick, then
``pxform(object_frame, camera_frame, 0.0)`` for the fixed rotation -- at two
tolerances in turn:

* ``sce2t(sclk_id, midtime_et)`` at ``800 + texp/48`` ticks, the tolerance the
  host's own frame uses;
* ``sce2c(sclk_id, midtime_et)`` at
  :data:`~spindoctor.cli.ck.index.SNAPPED_LOOKUP_TOL_TICKS` (80000 ticks),
  the far wider tolerance ``oops`` registers its fallback frame with.

The index widens a frozen-attitude object's coverage by that same wider
tolerance, read from the same constant rather than restated, so that every
image the lookup can serve survives the coverage filter and reaches the
reproduction step. Widening by less would drop the only candidate that
reproduces and report the image as having no baseline.

**Rigid-rotation residual.** An exact rigid rotation is not exactly a uniform
tangent-plane shift, and the difference is measured over a 17x17 grid across
the full frame, worst case over eight offset directions, at 50 pixels of total
boresight displacement. On a Voyager 2 narrow angle frame it is 1.29e-8 rad,
which is 1.64e-3 pixels both in the tangent plane and in pixel space. It is
linear in the offset, so quoting it without the offset it was measured at means
nothing. The wide angle camera has no such measurement.

**The fallback cache.** The wider tolerance is about 4800 seconds of Voyager
clock, and the ``oops`` frame that uses it caches its answer for that long. Two
images navigated through the fallback within eighty minutes of each other **in
one process** therefore share one attitude. The second records an attitude no
lookup at its own midtime reproduces, and is honestly reported
``no_reproducing_baseline`` rather than corrected against a baseline it did not
use. This is a real, reachable outcome on a batch run, not a hypothetical.

**Kernel-name class rules.** None. This mission is listed in
``_MISSION_NAME_RULES`` with an empty rule tuple, which records that its
holdings were read and encode nothing rather than that nobody looked. Every
candidate is ``UNCLASSIFIED`` and the tie-break falls through to the
lexicographically greatest basename.

**Omission reasons.** ``not_eligible`` and ``no_reproducing_baseline``.
``baseline_coverage_gap`` cannot arise from segment building, since a frozen
segment reads no baseline at its record epochs.

Simulator model
===============

**Instrument key.** ``vgiss``, one key for two spacecraft and four cameras.
That is the model's sharpest simplification and it is documented as such in the
catalog.

**PSF kernel.** ``{sigma_v: 0.85, sigma_u: 0.85, w: 1.2e-2, r0: 2.0, n: 3.0}``,
a **retained interim estimate**. The cohort's one star frame gives eight usable
cutouts with a 50% encircled-energy radius of 1.22 pixels, but the geometric
resampling cannot be inverted well enough to constrain a kernel through it --
the flat-top guard leaves two simulated cutouts to compare against -- and the
Voyager references publish no FWHM to fall back on.

**Distortion residuals.**
``{k1: -6.88e-03, k2: 1.46e-02, nonradial_rms_px: 0.2}``. The non-radial wander
is non-zero: the resampled vidicon geometry carries coherent tangential
distortion that a radial polynomial cannot represent. The block is the
Voyager 2 wide angle measurement, and the catalog says plainly that one key
under-represents the spread across two spacecraft and two optics until
per-camera keys and locked frames for each exist.

**Artifact-mode availability.** This instrument is **not** in the CCD set, so
the CCD-only modes -- ``radiation_transients`` and ``compression_dct`` -- are
unavailable to it. It carries five further modes: ``pixel_spikes``,
``beam_bend`` (the brightness-dependent limb bend), ``residual_image`` (the
prior-frame ghost), ``reseau_scars`` (the reseau-removal patches left by
archive processing), and ``resample_texture`` (the geometric resample the
pipeline's own input has been through). The last two are telemetry-stage modes
applied after the structured-loss modes, because they emulate archive
processing rather than a detector or transmission defect. ``edited_frame``
defaults to a 440-pixel band, which is this instrument's commanded edited-mode
band width, so a bare incidence keeps a physical centered band.

**Realism-match status.** The PSF is unverified and unverifiable through the
resampling with the frames held locally. The distortion is measured, but on
one of the four cameras.

Image library and test coverage
===============================

**Cohort.** Three sidecars: two under ``scattered_light`` and one under
``star_dominated``. All three are geometrically corrected products, and the
star-dominated one is also the frame the C-kernel round trip runs on.

**Integration tests.** The per-image regression suite
(``tests/integration/test_autonomous_nav.py``) and the structural-invariants
test (``test_image_library.py``) run on those sidecars.

``tests/integration/test_cmatrix_frames.py`` gives this instrument its own
tests rather than a parametrized row, because the shared assertions do not
apply: it pins that the recorded uncorrected attitude **is** the frozen
observation frame, that a plain ``pxform`` at the midtime does not resolve at
all, that the frame identities are derived per spacecraft, and that the flip is
the identity.

``tests/integration/test_ck_round_trip.py`` includes a frame of this instrument
in its round-trip cohort, and needs an extra step for it: because the host
freezes the observation frame during ``from_file``, a kernel furnished after
that call cannot be seen at all, so the image is loaded a **second** time with
the correction already furnished.

**Unit tests.** ``tests/spindoctor/inst/test_inst_voyager_iss.py`` covers the
spacecraft-digit parser and all four of its refusals, the I/F factor parser and
all three of its refusals, the per-camera limiting-magnitude anchors, the
per-spacecraft metadata LIDs, and the published clock counts: fractional
leading units with no midtime count, read from the PDS3 label beside the image.

PDS4 hooks
==========

Not supported. Two hooks are implemented --
``pds4_bundle_template_dir``, which falls back to ``voyager_iss_1.0``, and
``pds4_bundle_name``, which falls back to
``voyager_iss_backplanes_rsfrench2027`` -- and every remaining hook raises
``NotImplementedError``. Neither the template directory nor a ``pds4.vgiss``
configuration entry is shipped; the configuration file carries the entry as a
commented-out stub.

Making bundles work here means implementing ``pds4_bundle_path_for_image``,
``pds4_path_stub``, ``pds4_lid_part_to_image_name``, the four LID and LIDVID
builders and ``pds4_template_variables``, and adding a template directory. It
also means deciding whether one bundle covers both spacecraft or each gets its
own.

Backplanes, mosaics, and statistics
===================================

**Backplanes.** Nothing instrument-specific in the stage. The records carry a
usable C-matrix, since rotation fitting is off, so they take the matrix path
rather than the offset fallback. The replacement frame the reader installs is
constant across the exposure, which for this instrument is not an
approximation at all: the navigated attitude was constant to begin with.

**Mosaics.** Nothing instrument-specific.

**Statistics.** Nothing particular.
:func:`~spindoctor.cli.stats.report_sections.resolve_offset_limit` reads
``voyager_iss.extfov_margin_vu`` from the flat section and indexes the
size-keyed table with the recorded image height, so a database row with no
recorded shape, or one recording a height other than 1000, resolves to an
explanatory string instead of a limit. The report's image-name rule is
``stem.split('_', 1)[0]``, so ``C1234567_GEOMED`` reduces to ``C1234567``.

Open items
==========

* ``from_file`` carries a TODO recording that the I/F scaling and the Voyager 1
  Saturn correction belong in the host once the host grows them, at which point
  this loader stops modifying pixel data.
* Every value marked ``# PLACEHOLDER`` in ``config_430_inst_vgiss.yaml``:
  ``expected_noise_dn``, ``read_noise_dn``, ``blank_max_if``,
  ``noisy_threshold_if`` and the ``mag_offset_table`` entry.
* ``fit_camera_rotation`` is off for cost against a measurement that says it
  should be on. Revisiting it means both speeding the rotation search and
  accepting the loss of corrected C-kernels for this instrument, or recording
  the rotation pivot so the two can coexist.
* Both limiting-magnitude anchors are nominal-optics derivations rather than
  measurements, and the cohort holds one star frame.
* The simulator uses one distortion and PSF key for four cameras across two
  spacecraft.
* The Voyager 1 cameras have no twist or distortion measurement: no frame of
  the Voyager 1 cohort locked.
