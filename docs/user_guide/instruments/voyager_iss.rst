============
Voyager ISS
============

Overview
========

The Voyager Imaging Science Subsystem is two spacecraft each carrying a narrow
angle and a wide angle vidicon camera, which SpinDoctor navigates across the
whole PDS3 archive: the Jupiter and Saturn encounters of both spacecraft and
the Uranus and Neptune encounters of Voyager 2. The two spacecraft differ in
their SPICE identities, in one photometric correction, and in the identifiers
they write into metadata; everywhere else they are handled identically, and the
sections below name a spacecraft only where they actually differ.

Pipeline support
================

* **Navigation** -- supported, both spacecraft and both cameras.
* **Corrected-pointing C-kernels** -- supported. ``sd_create_ck vgiss``. The
  segments carry one constant attitude: see `Corrected-pointing C-kernels`_.
* **Backplanes** -- supported.
* **Mosaics** -- supported, body and ring.
* **PDS4 bundles** -- not supported. The dataset names a bundle and a label
  template directory, but the remaining hooks are unimplemented and the
  configuration entry is a commented-out stub.
* **Simulator** -- supported, under the single instrument key ``vgiss``.
* **Statistics** -- supported.

Datasets and image selection
============================

**Dataset names.** ``vgiss``, with the alias ``vgiss_pds3`` naming the same
class. Both are case-insensitive. There are no sub-datasets: one dataset covers
both spacecraft and all four encounters.

**Volumes.** Six ranges, one per spacecraft and encounter:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Encounter
     - Spacecraft
     - Volumes
   * - Jupiter
     - Voyager 1
     - VGISS_5101 - VGISS_5120
   * - Jupiter
     - Voyager 2
     - VGISS_5201 - VGISS_5214
   * - Saturn
     - Voyager 1
     - VGISS_6101 - VGISS_6121
   * - Saturn
     - Voyager 2
     - VGISS_6201 - VGISS_6215
   * - Uranus
     - Voyager 2
     - VGISS_7201 - VGISS_7207
   * - Neptune
     - Voyager 2
     - VGISS_8201 - VGISS_8210

Naming a volume outside those ranges is an error rather than an empty result.

**Holdings layout.** Images are read from the ``volumes/`` subtree:

.. code-block:: text

   $PDS3_HOLDINGS_DIR/volumes/VGISS_6xxx/VGISS_6101/data/...
   $PDS3_HOLDINGS_DIR/metadata/VGISS_6xxx/VGISS_6101/VGISS_6101_index.lbl

The volume set directory takes the volume's first digit: ``VGISS_5xxx``,
``VGISS_6xxx``, ``VGISS_7xxx`` or ``VGISS_8xxx``.

**Which product is navigated.** The geometrically corrected product, and only
that one. Each volume carries ``_RAW``, ``_CALIB`` and ``_GEOMED`` products for
each frame; the dataset navigates ``_GEOMED`` and **silently skips** the other
two rather than reporting them, so an image count from a volume listing will
not match the number of images a run considers. The filespec is
``DATA/Cddddddd/Cddddddd_GEOMED.LBL``, matched uppercase, which is the
archive's own convention.

**Image names.** A name is ``C`` followed by seven digits -- the Flight Data
Subsystem count. Product suffixes and extensions are stripped before the name
is validated, so all of these name the same image:

.. code-block:: text

   C1636338
   C1636338_GEOMED
   C1636338_GEOMED.IMG

Names given on the command line, or in an ``--image-file-list`` file, are
matched case-insensitively. Filespecs given in an ``--image-filespec-csv`` file
are parsed as archive filespecs, so their ``_GEOMED.LBL`` suffix must be
uppercase; one that is not is skipped.

**Image numbering, and the range-selection caveat.** The image number is the
seven-digit FDS count, and it **restarts per spacecraft and per encounter**.
The volume order interleaves the two spacecraft, and Voyager 2's Neptune counts
roll over below Voyager 1's Jupiter counts, so an image-number range can match
frames in any volume. The dataset therefore declares image numbers
non-monotonic across volumes and no volume-level early exit is possible:
``--first-image-num`` and ``--last-image-num`` still filter correctly, but they
scan every requested volume rather than stopping once the range is passed. On a
whole-archive run that is the difference between reading one index and reading
all eighty-seven. Restrict the volumes as well when the frames wanted are known
to sit in one encounter.

**Cameras and instrument-specific flags.** Two cameras per spacecraft, reported
as ``NAC`` and ``WAC``. The Voyager indexes carry no instrument identifier
column, so the camera is read from ``INSTRUMENT_NAME``, whose values spell it
out as ``NARROW ANGLE CAMERA`` and ``WIDE ANGLE CAMERA``. This instrument adds
no selection flags of its own; the shared PDS3 options are the whole surface,
and there is no camera filter.

**Grouping.** None.

**Examples.**

.. code-block:: bash

   # One image by name
   sd_offset vgiss C1636338

   # One volume
   sd_offset vgiss --volumes VGISS_6101

   # Two volumes of one encounter, which is also how a number range is
   # kept from scanning the whole archive
   sd_offset vgiss --volumes VGISS_5101 --volumes VGISS_5102 \
       --first-image-num 1500000 --last-image-num 1600000

Image data and units
====================

**Units.** I/F. The GEOMED products are archived as scaled integers, and the
loader converts them: the scaling factor is read from the ``LABEL3`` record of
each image's own VICAR label and the data are multiplied by it and divided by
10000.

**The Voyager 1 Saturn correction.** One further factor of **3.345** is applied
to Voyager 1 frames whose target system is Saturn, and to no others. The
archive's calibration pipeline computed I/F for Voyager 1 as though every image
had been taken at Jupiter's heliocentric distance, so the Saturn frames come
out too dim by the square of the distance ratio. Voyager 1 visited only Jupiter
and Saturn, and Voyager 2 was calibrated correctly at each of its encounters,
so this is the only case. A user comparing an I/F value from a Voyager 1 Saturn
frame against the archive's own number will find them differing by exactly this
factor, and the pipeline's is the corrected one.

**Saturation.** The configuration records a 255 DN ceiling, the 8-bit ADC
limit, but that is a raw-DN quantity and the navigated product is in I/F. As
with any calibrated-I/F input, no per-pixel saturation mask is built and no
saturation threshold applies: the reported saturation fraction is always zero
and the early-out that abandons a fully overexposed image never fires. There is
no raw Voyager product the pipeline can navigate instead.

**Missing pixels.** The marker is ``0``.

**Classification thresholds.** An image whose data stays below 1.0e-4 in I/F is
classified blank and not navigated; one whose noise estimate exceeds 0.005 in
I/F is classified noisy. An image is no longer clean once more than 30% of its
pixels are missing or more than 80% are overexposed.

**Provisional values.** The blank and noisy thresholds, the expected-noise and
read-noise figures, and the magnitude-offset table are all placeholders
awaiting calibration. The I/F scaling read from each label is a measurement;
the thresholds it is judged against are not yet.

Field of view and geometry
==========================

**Extended-FOV margins.** A margin of ``[400, 400]`` pixels, declared for an
image size of 1000, which is what a GEOMED product is. It is generous because
the reconstructed attitude carries real per-frame error, and it is the largest
offset a search can find. The margin table has no entry for any other image
size, so an image of another size cannot be loaded.

**Camera rotation.** Rotation fitting is **off**, and that is a cost decision
rather than a statement that there is nothing to fit. These cameras do carry
per-frame rotation residuals, which the measurements below quantify; fitting
them per frame is too slow to enable across the archive. What a user sees is a
two degree of freedom offset, ``(dv, du)``, with a real per-frame twist
absorbed into the translation rather than reported. The setting is worth
revisiting whenever the rotation search becomes fast enough.

**Measured twist.** The twist is **frame-varying** on both Voyager 2 cameras,
which is the regime no static kernel correction can fix. The narrow angle
camera measures +0.030 +/- 0.005 degrees, 0.37 pixels at the field corner, with
a frame-to-frame scatter of 0.283 pixels; the wide angle camera measures
+0.358 +/- 0.005 degrees, 4.42 pixels at the corner, with a scatter of 0.275
pixels. In both cases the scatter is well above the threshold at which a twist
counts as one common value. The Voyager 1 cameras are unmeasured: no frame of
the Voyager 1 cohort locked.

**Residual distortion.** Consistent with the resampled vidicon geometry, the
Voyager 2 narrow angle camera measures a radial RMS of 0.936 pixels and the
wide angle camera 0.345, both with a substantial non-radial component that a
radial model cannot represent, and both against a high centroid-and-astrometry
floor of 0.25 to 0.34 pixels. These figures are therefore of low confidence.
See :doc:`/fov_distortion_report/fov_distortion_report` for the coefficients,
the method and the figures.

Metadata fields
===============

Beyond the keys every instrument writes -- image path and name, the start,
midtime and end of the exposure in UTC and in TDB seconds, the image shape,
the camera, the exposure time and the instrument host and instrument LIDs --
a Voyager ISS record carries one filter entry in ``filters``. Its
``instrument`` is ``vgiss`` and its ``camera`` is ``NAC`` or ``WAC``. It writes
no ``shutter_mode``, since its labels carry none.

It writes the spacecraft-clock fields ``start_time_sclk`` and ``end_time_sclk``
from the start and stop counts of the PDS3 label beside the image. The start
count lies near the shutter opening, but the stop count is the count of the
frame the image was read out in, which can come minutes after the shutter
closed. The two do not bracket the exposure, so ``midtime_sclk`` is null. Both
are counts of the clock's leading field, the first five digits of the image
number: one unit is 48 minutes, 60 frames of 48 seconds, each 800 lines of 60
milliseconds, and the frame and the line are a fraction of it. The ``times``
block's clock strings are computed from the exposure times and can differ from
these counts by minutes. It writes none of ``sampling``, ``gain_mode``,
``description`` or ``observation_id``.

The two LIDs vary by spacecraft and camera, and are the one place the metadata
distinguishes the two spacecraft. The instrument host LID is
``...:instrument_host:spacecraft.vg1`` or ``...:spacecraft.vg2``, and the
instrument LID is ``...:instrument:vg1.issn``, ``vg1.issw``, ``vg2.issn`` or
``vg2.issw``. Which spacecraft took a frame is read from the ``LAB02`` record
of its own label, not from the volume it sits in.

Corrected-pointing C-kernels
============================

**What is corrected.** The corrected object is the **scan platform**, which is
**-31100** for Voyager 1 and **-32100** for Voyager 2. A corrected kernel
carries the platform attitude the navigation implies, so no camera frame is
fabricated and no frame kernel has to change. Segment time tags are encoded
against spacecraft clock **-31** or **-32** respectively; the clock is not
derivable from the object by arithmetic, and each object states its own.

The camera frames the correction is measured in are ``VG1_ISSNA``,
``VG1_ISSWA``, ``VG2_ISSNA`` and ``VG2_ISSWA``. One ``sd_create_ck vgiss`` run
covers both spacecraft and writes segments for whichever objects its images
name.

**Running it.**

.. code-block:: bash

   sd_create_ck vgiss \
       --nav-results-root /data/nav/results \
       --kernel-dir $SPICE_PATH/Voyager \
       --kernel-dir $SPICE_PATH/Voyager/SCLK \
       --kernel-dir $SPICE_PATH/Voyager/FK \
       --kernel-dir $SPICE_PATH/Voyager/CK \
       --output-dir /data/nav/ck

Directories are not searched recursively, so a holdings tree that keeps its
kernels in per-kind subdirectories needs one flag per subdirectory. Both
spacecraft's clock and frame kernels must be present when the run's images
span both.

**Baseline kernel naming.** Voyager basenames declare no kernel class. The
holdings hold one kind of C-kernel and nothing in a basename says how its
pointing was made, so every candidate is unclassified and, when more than one
reproduces an image's attitude, the tie-break falls through to the
lexicographically greatest basename. That is a deterministic choice among
candidates that agree on the attitude, not a quality judgment.

**Segment shape: one constant attitude.** The navigated attitude comes from a
single tolerance-snapped pointing lookup that is constant across the exposure,
not from an evaluated frame chain, so a corrected segment carries **that one
attitude, constant across its window, with zero angular velocity**. There is
nothing to interpolate between the records, and therefore no interpolation
error at all: every epoch inside a Voyager segment is exact.

The zero angular velocity is a measurement rather than an omission -- zero is
what a constant attitude's angular velocity is -- and it is written explicitly
rather than declared absent, because a segment declaring no angular velocity is
skipped by SPICE for ``ckgpav`` and ``sxform``, which would answer those calls
from the uncorrected original instead.

**Angular velocity in the baselines.** None of the nine -31100 and -32100
segments in the local baselines carries angular velocity. That does not matter:
a frozen segment never reads its baseline's attitude history, and writes its
own zeros.

**Omission reasons this instrument produces.** ``not_eligible`` and
``no_reproducing_baseline``. ``rotation_unsupported`` never appears, because
rotation fitting is off. ``botsim_loser`` cannot appear, since it needs two
cameras exposed at once and a Voyager exposure uses one.
``baseline_coverage_gap`` cannot arise from segment building either, since a
frozen segment never reads the baseline at its record epochs.

**A repeated-lookup case that produces** ``no_reproducing_baseline``. The
snapped lookup that freezes the observation frame has a fallback at a much
wider tolerance -- about 4800 seconds of Voyager clock -- and the frame that
uses it caches its answer for that long. Two Voyager images navigated through
that fallback within eighty minutes of each other **in one process** therefore
share one attitude. The second image records an attitude that no lookup at its
own midtime reproduces, so it is refused rather than corrected against a
baseline it did not actually use. Its pixel offset in the metadata is
unaffected.

**Interpolation error.** None, for the reason above. A Voyager segment
reproduces its corrected attitude exactly at every epoch it covers, not only at
its record epochs.

Known limitations
=================

* Only the geometrically corrected products are navigated, and the raw and
  calibrated products in the same volumes are skipped without being reported.
* No saturation mask is built and the reported saturation fraction is always
  zero, with no raw product available to navigate instead.
* Image-number range selection cannot skip volumes, so a whole-archive number
  range reads every index.
* The per-frame twist these cameras carry is real and is not fitted; it is
  absorbed into the reported translation. The wide angle measurement is over
  four pixels at the field corner.
* The residual distortion is not removed, and its measurement floor is high
  enough that the numbers themselves are of low confidence.
* The Voyager 1 cameras have no twist or distortion measurement at all: the
  cohort holds no Voyager 1 frame that locked.
* The simulator uses one distortion parameter set for all four cameras, taken
  from the Voyager 2 wide angle measurement, which under-represents the spread
  between two spacecraft and two optics.
* PDS4 bundles are not generated.

References
==========

* The VGISS_xxxx volumes' ``document/`` directory on the PDS Ring-Moon Systems
  Node, which carries the ISS instrument description, the GEOMED processing
  description, and the volume ``*_index.lbl`` column definitions this pipeline
  reads.
* B. A. Smith et al., "Voyager Imaging Experiment", *Space Science Reviews* 21,
  103-127 (1977).
* :doc:`/fov_distortion_report/fov_distortion_report` -- the measured twist and
  residual distortion quoted above.
