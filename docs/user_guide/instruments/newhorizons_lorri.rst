=====================
New Horizons LORRI
=====================

Overview
========

The New Horizons Long Range Reconnaissance Imager is a single panchromatic
framing camera with no filter wheel, which SpinDoctor navigates across the
whole PDS3 archive: launch, the Jupiter encounter, the Pluto cruise and
encounter, and the Kuiper Belt cruise and encounters. Navigation, corrected
pointing kernels, backplanes and mosaics are supported.

Pipeline support
================

* **Navigation** -- supported.
* **Corrected-pointing C-kernels** -- supported. ``sd_create_ck nhlorri``.
* **Backplanes** -- supported.
* **Mosaics** -- supported, body and ring.
* **PDS4 bundles** -- not supported. The dataset names a bundle and a label
  template directory, but the remaining hooks are unimplemented and the
  configuration entry is a commented-out stub.
* **Simulator** -- supported, under the instrument key ``nhlorri``.
* **Statistics** -- supported.

Datasets and image selection
============================

**Dataset names.** ``nhlorri``, with the alias ``nhlorri_pds3`` naming the same
class. Both are case-insensitive. There are no sub-datasets.

**Volumes.** Seven, one per mission phase, and they are named rather than
numbered: ``NHLALO_2001``, ``NHJULO_2001``, ``NHPCLO_2001``, ``NHPELO_2001``,
``NHKCLO_2001``, ``NHKELO_2001``, ``NHK2LO_2001``. Naming anything else is an
error rather than an empty result.

**Holdings layout.** Images are read from the ``volumes/`` subtree:

.. code-block:: text

   $PDS3_HOLDINGS_DIR/volumes/NHxxLO_xxxx/NHPELO_2001/data/...
   $PDS3_HOLDINGS_DIR/metadata/NHxxLO_xxxx/NHPELO_2001/NHPELO_2001_index.lbl

Every volume sits under the single volume set directory ``NHxxLO_xxxx``.

**Which product is navigated.** Both the science and the engineering products:
an index filespec ending ``_sci.lbl`` or ``_eng.lbl`` is used as it stands,
with the FITS image resolved from the label. Those two suffixes are matched
**lowercase only**, which is the archive's own convention; an index row naming
anything else stops the run rather than being skipped, so a layout change is
visible instead of silently reducing the image count. A row whose directory
structure is unexpected is a milder case: it is logged as an error and that
row alone is dropped.

The data directory is organized by request rather than by image number: the
filespec is ``data/ddddddd_ddddddd/lor_dddddddddd_0xNNN_sci.lbl``, whose
middle component is a pair of seven-digit request identifiers.

**Image names.** A name is ``lor_`` followed by ten digits -- fourteen
characters exactly. The image name is the leading fourteen characters of the
product file name, so the readout-mode component and the ``_sci`` / ``_eng``
suffix are not part of it:

.. code-block:: text

   lor_0299793639            # the image name
   lor_0299793639_0x630_sci  # the product file name it comes from

Names given on the command line, or in an ``--image-file-list`` file, are
matched case-insensitively. Filespecs given in an ``--image-filespec-csv``
file are parsed as archive filespecs, so their ``_sci.lbl`` / ``_eng.lbl``
suffix must be lowercase; one that is not is reported and skipped.

**Image numbering.** The image number is the ten digits after ``lor_``. The
dataset declares image numbers monotonic across volumes, so
``--last-image-num`` stops scanning once it passes the range rather than
reading every remaining volume.

**Cameras and instrument-specific flags.** One camera, reported as ``LORRI``.
This instrument adds no selection flags of its own; the shared PDS3 options are
the whole surface.

**Grouping.** None.

**Examples.**

.. code-block:: bash

   # One image by name
   sd_offset nhlorri lor_0299793639

   # One mission phase
   sd_offset nhlorri --volumes NHPELO_2001

   # A filespec CSV, restricted to a number range
   sd_offset nhlorri --image-filespec-csv /path/to/nhlorri.csv \
       --first-image-num 299000000 --last-image-num 300000000

Image data and units
====================

**Units.** Raw DN. The loader asks the host for the uncalibrated image on
purpose: the calibrated LORRI products are themselves in DN rather than I/F, so
there is no I/F conversion to make, and navigation treats image brightness
scale-invariantly in any case -- normalized cross-correlation, an image-derived
noise floor, a magnitude-based star gate.

**Saturation.** 4095 DN, the 12-bit ADC ceiling. The saturation threshold
matches it, so the per-pixel saturation mask and the fully-overexposed early-out
both work as documented. The ADC ceiling is not the same quantity as the
detector full well, which is still a placeholder.

**Missing pixels.** The marker is ``0``.

**Classification thresholds.** An image whose data stays below 5.0 DN is
classified blank and not navigated; one whose noise estimate exceeds 10.0 DN is
classified noisy. An image is no longer clean once more than 30% of its pixels
are missing or more than 80% are overexposed.

**Corrections applied at load.** None beyond what the host reader does.

**Provisional values.** The blank and noisy thresholds, the full-well figure,
the expected-noise and read-noise figures, and the magnitude-offset table are
all placeholders awaiting calibration. Only the 4095 DN ADC ceiling and the
missing-pixel marker are hard facts. This instrument's configuration block
carries a ``_sources`` section naming, value by value, where each number came
from and which are still to be measured; it is documentation only and does not
affect a run.

Field of view and geometry
==========================

**Extended-FOV margins.** The margin is how far outside the frame the model is
generated, and therefore the largest offset a search can find. It is keyed by
image size, because the archive holds full frames and binned frames and a
margin that is right for one is wrong for the other:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Image size
     - Margin ``[v, u]``
   * - 256
     - ``[15, 15]``
   * - 512
     - ``[30, 30]``
   * - 1024
     - ``[60, 60]``

**Camera rotation.** Rotation fitting is off. Offsets are two degrees of
freedom, ``(dv, du)``, and no rotation is reported.

**Measured twist.** The twist is a clean, common +0.1912 +/- 0.0006 degrees
over the earlier of the two epoch cohorts, which displaces the field corner by
2.42 pixels, with a frame-to-frame scatter of only 0.027 pixels. That is a
static camera-frame alignment error rather than per-frame attitude noise: it is
the largest twist of any camera in the pipeline and also the most consistent.
Because it is static, a fixed frame-kernel correction removes it and per-frame
rotation fitting is not needed -- which is why rotation fitting is off despite
the size of the number. Until such a correction is applied, expect a systematic
2.4 pixel displacement at the field corner that navigation absorbs into the
translation it reports.

The later epoch cohort is unmeasured: none of its frames locked, because those
epochs fall outside the pointing-kernel coverage that was loaded. The residual
there stands on the adopted literature distortion model rather than on a
measurement.

**Residual distortion.** Small and close to the noise: a mid-field hump peaking
near 0.06 pixels, with a radial RMS of 0.078 pixels against a
centroid-and-astrometry floor of 0.082. The twist, not the distortion, is this
camera's significant geometric signature. See
:doc:`/fov_distortion_report/fov_distortion_report` for the coefficients, the
method and the figures.

Metadata fields
===============

Beyond the keys every instrument writes -- image path and name, the start,
midtime and end of the exposure in UTC and in TDB seconds, the image shape,
the camera, the exposure time and the instrument host and instrument LIDs --
a New Horizons LORRI record carries nothing extra. ``filters`` is present and
**empty**: the camera is panchromatic and has no filter wheel, so there is no
filter name to record. This is the only instrument whose ``filters`` list is
empty rather than carrying one or two entries.

It writes the spacecraft-clock fields ``start_time_sclk`` and ``end_time_sclk``
from the start and stop counts of the PDS3 label beside the image, which mark
the start and the end of the exposure, in seconds of the clock with the
1/50000-second ticks as a fraction, and ``midtime_sclk`` as the count exactly
halfway between them. It writes none of ``sampling``, ``gain_mode``,
``description`` or ``observation_id``. The instrument host LID is
``...:instrument_host:spacecraft.nh`` and the instrument LID is
``...:instrument:nh.lorri``, with no camera component, since there is one
camera.

Corrected-pointing C-kernels
============================

**What is corrected.** The corrected object is **-98000, the spacecraft**. A
corrected kernel carries the spacecraft attitude the navigation implies, so no
camera frame is fabricated and no frame kernel has to change. The camera frame
the correction is measured in is ``NH_LORRI``. Segment time tags are encoded
against spacecraft clock **-98**.

**Running it.**

.. code-block:: bash

   sd_create_ck nhlorri \
       --nav-results-root /data/nav/results \
       --kernel-dir $SPICE_PATH/NewHorizons \
       --kernel-dir $SPICE_PATH/NewHorizons/SCLK \
       --kernel-dir $SPICE_PATH/NewHorizons/FK \
       --kernel-dir $SPICE_PATH/NewHorizons/CK \
       --output-dir /data/nav/ck

Directories are not searched recursively, so a holdings tree that keeps its
kernels in per-kind subdirectories needs one flag per subdirectory.

Expect one warning in the run log naming an object the scan could not place in
time. The merged pointing files in this mission's holdings describe an object
**-1** beside the spacecraft, and no furnished kernel defines a spacecraft
clock for it, so its coverage cannot be expressed in TDB. The scan records it
as unreadable, indexes the rest of the file, and names it once in the log. It
can never supply a baseline, and no image ever asks it to; an image that
actually corrected such an object would be refused with the missing clock
named.

**Baseline kernel naming.** New Horizons basenames declare a class only on the
pair of kernels that exist in both forms: a trailing ``_recon`` marks
reconstructed pointing and a trailing ``_pred`` marks predicted, both on the
mission's ``nh_`` prefix. Every other name in the holdings -- the merged
pointing files and the hazard-search kernels -- declares nothing and is left
unclassified rather than guessed at from a prefix that says which product a
kernel is and not how its pointing was made. When several kernels reproduce one
image's attitude, reconstructed is preferred over predicted, and among
unclassified candidates the tie-break falls through to the lexicographically
greatest basename.

**Angular velocity.** Every -98000 segment in the local baselines carries
angular velocity -- 4346 of 4346 measured -- so no New Horizons image is
refused for a baseline that supplies pointing without a rate.

**Segment shape.** A segment carries records at the exposure start, midtime and
stop, plus a one-second cadence once the exposure reaches ten seconds. The
attitude is time-varying: the correction is held body-fixed and composed onto
the baseline's own pointing at each record epoch.

**Omission reasons this instrument produces.** ``not_eligible``,
``no_reproducing_baseline`` and ``baseline_coverage_gap``.
``rotation_unsupported`` never appears, because rotation fitting is off.
``botsim_loser`` cannot appear: it belongs to an instrument that exposes two
cameras at once, and this one has a single camera.

**Interpolation error.** Not yet measured for this instrument. What is known is
the shape rather than the size: the error is zero at every record epoch, grows
between them, is largest where the baseline's own rate changes inside the
window, and shrinks as records are added. The size depends on how the
spacecraft was turning during the exposure, which differs sharply between a
cruise frame and an encounter frame. A consumer who evaluates geometry at the
exposure midtime is exact and unaffected; one who needs a bound at arbitrary
interior epochs should measure it on the frames they care about, by comparing a
corrected segment against its baseline at epochs between the records with the
correction set to zero.

Known limitations
=================

* The static camera twist is real, large, and not removed: 2.4 pixels at the
  field corner over the epochs where it has been measured, absorbed into the
  reported translation rather than reported as a rotation.
* The twist over the later epoch cohort is unmeasured for want of pointing
  kernel coverage, so nothing is claimed about whether it matches the earlier
  value.
* The blank and noisy thresholds and the full-well figure are placeholders, so
  the blank and noisy classifications are provisional.
* The simulator's point-spread function for this instrument is a published
  estimate for unbinned frames; the curated image cohort's frames are binned
  four by four and cannot constrain it, and no per-readout-mode kernel exists.
* PDS4 bundles are not generated.

References
==========

* The NHxxLO_xxxx volumes' ``document/`` directory on the PDS Ring-Moon
  Systems Node, which carries the LORRI instrument description and the volume
  ``*_index.lbl`` column definitions this pipeline reads.
* A. F. Cheng et al., "Long-Range Reconnaissance Imager on New Horizons",
  *Space Science Reviews* 140, 189-215 (2008).
* The New Horizons LORRI Instrument Calibration Report distributed with the
  archive, which is where the ADC ceiling and the missing-pixel marker
  recorded in the configuration come from.
* :doc:`/fov_distortion_report/fov_distortion_report` -- the measured twist and
  residual distortion quoted above.
