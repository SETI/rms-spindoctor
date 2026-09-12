============
Galileo SSI
============

Overview
========

The Galileo Solid State Imager is a single framing camera which SpinDoctor
navigates across the whole PDS3 archive: the cruise targets and the Jupiter
orbital tour. Navigation, backplanes, mosaics and corrected pointing kernels
are all supported.

Pipeline support
================

* **Navigation** -- supported.
* **Corrected-pointing C-kernels** -- supported. See
  `Corrected-pointing C-kernels`_.
* **Backplanes** -- supported.
* **Mosaics** -- supported, body and ring.
* **PDS4 bundles** -- not supported. The dataset names a bundle and a label
  template directory, but the remaining hooks are unimplemented and the
  configuration entry is a commented-out stub.
* **Simulator** -- supported, under the instrument key ``gossi``.
* **Statistics** -- supported.

Datasets and image selection
============================

**Dataset names.** ``gossi``, with the alias ``gossi_pds3`` naming the same
class. Both are case-insensitive. There are no sub-datasets.

**Volumes.** GO_0002 through GO_0023. Naming a volume outside that range is an
error rather than an empty result.

**Holdings layout.** Images are read from the ``volumes/`` subtree:

.. code-block:: text

   $PDS3_HOLDINGS_DIR/volumes/GO_0xxx/GO_0017/...
   $PDS3_HOLDINGS_DIR/metadata/GO_0xxx/GO_0017/GO_0017_index.lbl

Every volume sits under the single volume set directory ``GO_0xxx``, and the
index file name is lowercase.

**Which product is navigated.** The index names a ``.LBL`` filespec directly
and it is used as it stands, with the image resolved from the label's image
pointer. The archive holds no calibrated Galileo SSI product, so navigation
runs on raw DN.

The directory layout inside a volume is organized by target rather than by
image number, and the dataset parses both forms it takes: a two-level
``TARGET/IMAGE.LBL`` for the cruise targets and calibration data
(``RAW_CAL``, ``VENUS``, ``EARTH``, ``MOON``, ``GASPRA``, ``IDA``, ``SL9``,
``EMCONJ``, ``GOPEX``) and a three-level ``ORBIT/TARGET/IMAGE.LBL`` for the
Jupiter orbits (``C3``, ``C9``, ``C10``, ``C20``, ``C21``, ``C22``, ``C30``,
``E4``, ``E6``, ``E11``, ``E12``, ``E14``, ``E15``, ``E17``, ``E18``, ``E19``,
``E26``, ``G1``, ``G2``, ``G7``, ``G8``, ``G28``, ``G29``, ``I24``, ``I25``,
``I27``, ``I31``, ``I32``, ``I33``, ``J0``). Either form may sit under a
leading ``REDO`` directory, and reprocessed images under ``REDO`` are
navigated like any other. A filespec whose leading directory is none of those
names is logged as an error naming the index file and the filespec, and that
row alone is dropped -- so a layout change shows up in the log rather than
silently reducing the image count.

**Image names.** A name is ``C``, ten digits, and a trailing ``R`` or ``S`` --
twelve characters exactly, matched **uppercase only**:

.. code-block:: text

   C0349875200R

A filespec whose image name falls outside that rule names a product this
dataset does not navigate, and its row is dropped without comment. The archive
holds such products: ``GO_0016``'s ``SL9`` directory stores a ``G`` product
beside the ``R`` product of each of its thirteen images, a second
representation of an image the enumeration already yields.

**Image numbering.** The image number is the ten digits between the leading
``C`` and the trailing letter. Every image number in a volume exceeds every
image number in the volumes before it, so ``--last-image-num`` stops scanning
once it passes the range.

**Cameras and instrument-specific flags.** One camera, reported as ``SSI``.
This instrument adds no selection flags of its own; the shared PDS3 options are
the whole surface.

**Grouping.** None.

**Examples.**

.. code-block:: bash

   # One image by name
   sd_offset gossi C0349875200R

   # One volume
   sd_offset gossi --volumes GO_0017

   # A span of volumes, with a results filter
   sd_offset gossi --first-volume GO_0002 --last-volume GO_0010 \
       --has-no-offset-file

Image data and units
====================

**Units.** Raw DN. No I/F conversion is applied or expected: navigation treats
image brightness scale-invariantly -- normalized cross-correlation, an
image-derived noise floor, a magnitude-based star gate -- so no photometric
calibration is required to navigate.

**Saturation.** 255 DN, the 8-bit ADC ceiling. The saturation threshold matches
it, so the per-pixel saturation mask and the fully-overexposed early-out both
work as documented.

**Missing pixels.** The marker is ``0``.

**Classification thresholds.** An image whose data stays below 5.0 DN is
classified blank and not navigated; one whose noise estimate exceeds 10.0 DN is
classified noisy. An image is no longer clean once more than 30% of its pixels
are missing or more than 80% are overexposed.

**Corrections applied at load.** None beyond what the host reader does.

**Provisional values.** The blank and noisy thresholds, the expected-noise and
read-noise figures, and the magnitude-offset table are all placeholders
awaiting calibration. Only the 255 DN ceiling is a hard fact rather than a
starting guess.

Field of view and geometry
==========================

**Extended-FOV margins.** A single margin of ``[350, 350]`` pixels applies at
every image size, rather than a table keyed by size. It is generous because the
reconstructed attitude carries real per-frame error, and it is the largest
offset a search can find.

**Camera rotation.** Rotation fitting is **off** for this instrument. Every
technique works in two degrees of freedom, ``(dv, du)``, and the twist these
frames carry is absorbed into the reported translation. What the setting
decides is in :doc:`/dev_guide/dev_guide_rotation`.

**Measured twist.** Over the frames that lock, the twist is
-0.0526 +/- 0.0017 degrees, which displaces the field corner by 0.52 pixels,
with a frame-to-frame scatter of 0.092 pixels. That is consistent frame to
frame, which points at a static camera-frame alignment error rather than
per-frame attitude noise. The measurement rests on seven locked frames out of
eighteen, so it is provisional pending a larger star-frame cohort.

**Residual distortion.** The largest of any well-behaved camera in the
pipeline: a pincushion term reaching about half a pixel at the field corner,
with a radial RMS of 0.155 pixels against a centroid-and-astrometry floor of
0.115. The navigator does not remove it. See
:doc:`/fov_distortion_report/fov_distortion_report` for the coefficients, the
method and the figures.

Metadata fields
===============

Beyond the keys every instrument writes -- image path and name, the start,
midtime and end of the exposure in UTC and in TDB seconds, the image shape,
the camera, the exposure time and the instrument host and instrument LIDs --
a Galileo SSI record carries one filter entry in ``filters``.

It writes ``start_time_sclk``, the image's frame count as the label records
it, which comes a few seconds before the exposure, as a count of the clock's
leading field, the RIM count, with its three finer fields as a fraction of one.
The ``times`` block's clock strings are computed from the exposure times and
differ from it by seconds. ``midtime_sclk`` and ``end_time_sclk`` are always
null, because a Galileo SSI label records no count at the end of the image. It
writes none of ``sampling``, ``gain_mode``, ``description`` or
``observation_id``. The instrument host LID is
``...:instrument_host:spacecraft.go`` and the instrument LID is
``...:instrument:go.ssi``, with no camera component, since there is one camera.

Corrected-pointing C-kernels
============================

**What is corrected.** The corrected object is **-77001, the scan
platform**. The camera frame the correction is measured in is
``GLL_SCAN_PLATFORM``, which is also the frame the observation is built in.
Segment time tags are encoded against spacecraft clock **-77**.

**Running it.**

.. code-block:: bash

   sd_create_ck gossi \
       --nav-results-root /data/nav/results \
       --kernel-dir $SPICE_PATH/Galileo \
       --kernel-dir $SPICE_PATH/Galileo/SCLK \
       --kernel-dir $SPICE_PATH/Galileo/FK \
       --kernel-dir $SPICE_PATH/Galileo/CK \
       --output-dir /data/nav/ck

Directories are not searched recursively, so a holdings tree that keeps its
kernels in per-kind subdirectories needs one flag per subdirectory.

**Baseline kernel naming.** Galileo basenames declare no kernel class. The
holdings hold one kind of C-kernel and nothing in a basename says how its
pointing was made, so every candidate is unclassified and, when more than one
reproduces an image's attitude, the tie-break falls through to the
lexicographically greatest basename. That is a deterministic choice among
candidates that agree on the attitude, not a quality judgment.

**Angular velocity.** This is the sharpest constraint on the instrument. Of the
150 -77001 segments in the local baselines, **38 carry no angular velocity**. A
segment must carry a rate at every record or it cannot be written at all -- a
rateless segment is skipped outright by SPICE for ``ckgpav`` and ``sxform``,
which would silently answer those calls from the uncorrected original -- so an
exposure whose baseline supplies pointing but not a rate stops the run rather
than being omitted. Roughly a quarter of this mission's baseline segments would
refuse a run that reached them.

**Segment shape.** Records at the exposure start, midtime and stop, plus a
one-second cadence once the exposure reaches ten seconds. The attitude is
time-varying: the correction is held body-fixed and composed onto the
baseline's own pointing at each record epoch.

**Omission reasons this instrument produces.** ``not_eligible``, on images
whose navigation neither succeeded nor conflicted. ``rotation_unsupported``
never appears, because rotation fitting is off. ``botsim_loser``
cannot appear either: it belongs to an instrument that exposes two cameras at
once, and this one has a single camera. ``no_reproducing_baseline`` and
``baseline_coverage_gap`` are reachable and mean what they mean everywhere.

**Interpolation error.** Not yet measured for this instrument.

Known limitations
=================

* The per-frame twist these images carry is absorbed into the reported
  translation rather than measured: it is real, and a navigation that reports
  only a translation prices it as translation error.
* The residual distortion is about half a pixel at the field corner and the
  navigator does not remove it, so a feature near the frame edge is measured
  against a model that is that far off.
* The blank and noisy thresholds are placeholders, so the blank and noisy
  classifications are provisional.
* The simulator's point-spread function for this instrument is an unverified
  published estimate: the curated image cohort holds no star frames, so nothing
  independent has constrained it.

References
==========

* The GO_00xx volumes' ``document/`` directory on the PDS Ring-Moon Systems
  Node, which carries the SSI instrument description and the volume
  ``*_index.lbl`` column definitions this pipeline reads.
* M. J. S. Belton et al., "The Galileo Solid-State Imaging experiment",
  *Space Science Reviews* 60, 413-455 (1992).
* :doc:`/fov_distortion_report/fov_distortion_report` -- the measured twist and
  residual distortion quoted above.
