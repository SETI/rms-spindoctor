============
Cassini ISS
============

Overview
========

The Cassini Imaging Science Subsystem carries two framing cameras, a narrow
angle camera (NAC) and a wide angle camera (WAC), which SpinDoctor navigates
over the whole PDS3 archive: the cruise volumes covering the Venus, Earth and
Jupiter encounters, and the Saturn tour. Both cameras are supported by every
stage of the pipeline.

Pipeline support
================

* **Navigation** -- supported, both cameras.
* **Corrected-pointing C-kernels** -- supported. ``sd_create_ck coiss``.
* **Backplanes** -- supported.
* **Mosaics** -- supported, body and ring.
* **PDS4 bundles** -- supported for the Saturn dataset. The dataset class
  implements every PDS4 hook, and the shipped label templates and
  configuration entry cover ``coiss_saturn``. ``coiss_cruise`` names a
  template directory that is not shipped, so a cruise bundle needs that
  directory supplied before it will build.
* **Simulator** -- supported, under four instrument keys: raw and calibrated,
  each per camera.
* **Statistics** -- supported, with a BOTSIM pair consistency section in the
  report.

Datasets and image selection
============================

**Dataset names.** ``coiss`` selects the whole archive; ``coiss_cruise`` and
``coiss_saturn`` select the two halves of it. Each has a ``_pds3`` alias
(``coiss_pds3``, ``coiss_cruise_pds3``, ``coiss_saturn_pds3``) naming the same
class, and all names are case-insensitive.

**Volumes.** ``coiss_cruise`` covers COISS_1001 through COISS_1009;
``coiss_saturn`` covers COISS_2001 through COISS_2116; ``coiss`` covers both
ranges, in that order. Naming a volume outside the selected dataset's range is
an error rather than an empty result.

**Holdings layout.** Images are read from the ``calibrated/`` subtree of the
holdings root, not from ``volumes/``:

.. code-block:: text

   $PDS3_HOLDINGS_DIR/calibrated/COISS_2xxx/COISS_2001/data/...
   $PDS3_HOLDINGS_DIR/metadata/COISS_2xxx/COISS_2001/COISS_2001_index.lbl

The volume set directory is ``COISS_1xxx`` or ``COISS_2xxx`` according to the
volume's first digit, and the index file name is lowercase.

**Which product is navigated.** The calibrated product, always. Each index row
names a raw ``.IMG`` filespec, which the dataset rewrites to ``_CALIB.LBL``
before the image is opened, so every selection route -- volume ranges, image
number ranges, explicit names, ``--image-file-list``, ``--image-filespec-csv``
-- enumerates ``_CALIB.IMG`` files. The raw configuration block exists and is
selected by the absence of ``_CALIB`` in the filename, but no ``sd_offset``
selection route reaches it: a ``_RAW`` image name is rejected by the name
rule, and the path a run opens always comes from the index rewrite. Navigating
a raw frame means calling the observation loader on the ``_RAW.IMG`` path from
Python.

**Image names.** A name is a camera letter, ``N`` or ``W``, followed by ten
digits, optionally followed by ``_`` and one or two more digits, and
optionally carrying a ``_CALIB`` suffix and an ``.IMG`` extension. Names are
matched **uppercase only**. All of these name the same image:

.. code-block:: text

   N1454725799
   N1454725799_1
   N1454725799_1_CALIB
   N1454725799_1_CALIB.IMG

**Image numbering.** The image number is the ten digits after the camera
letter, a spacecraft-clock-derived counter that increases across the archive.
Every image number in a volume exceeds every image number in the volumes
before it, so ``--last-image-num`` stops scanning once it passes the range
rather than reading every remaining volume.

**Cameras and instrument-specific flags.** ``--camera`` takes ``nac`` or
``wac`` in either case. It filters on the image name's leading letter, so it
composes with every other selection option.

**Grouping.** ``botsim`` is supported. A BOTSIM ("both simultaneous") command
fires both shutters at once, and the grouping pairs the two frames into one
group so a driver sees them together. A pair is formed only when both frames
carry ``SHUTTER_MODE_ID == 'BOTSIM'``, come from opposite cameras, share an
``OBSERVATION_ID``, and have ``IMAGE_TIME`` values within 2.0 seconds of each
other; the NAC frame is placed first in the group. Frames failing any of those
tests are yielded singly, and no frame is ever dropped.

**Examples.**

.. code-block:: bash

   # One image by name
   sd_offset coiss N1454725799

   # One tour volume, narrow angle only
   sd_offset coiss_saturn --volumes COISS_2001 --camera nac

   # A span of volumes, ten random images, no processing
   sd_offset coiss --first-volume COISS_2001 --last-volume COISS_2010 \
       --choose-random-images 10 --dry-run

Image data and units
====================

Two configuration blocks exist, and the one used is chosen from the filename:
``_CALIB`` in the name selects ``cassini_iss_calib``, anything else selects
``cassini_iss``. Since the pipeline navigates the calibrated product, the
calibrated block is what an ordinary run uses. Both blocks are keyed per
camera.

**Units.** Calibrated frames are in I/F; raw frames are in DN.

**Saturation.** The raw block declares a 4095 DN ceiling, the 12-bit ADC
limit. The calibrated block declares **no saturation threshold at all**, and
that is deliberate: the calibration pipeline applies an exposure-, filter- and
gain-dependent scaling, so no single I/F constant identifies the physically
saturated DN ceiling. The consequence for a user is concrete -- on a
calibrated frame the per-pixel saturation mask is empty, the reported
saturation fraction is always zero, and the early-out that abandons a fully
overexposed image never fires. A user who needs accurate saturation flags has
to navigate the matching ``_RAW.IMG``, which is a Python-level call rather than
a command-line option. Calibration is not a geometric reprojection, so the raw
and calibrated frames share pixel coordinates and an offset measured on one
applies to the other.

**Missing pixels.** The raw marker is ``0``; the calibrated marker is ``NaN``.

**Classification thresholds.** An image whose data stays below the blank
threshold is classified blank and not navigated; one whose noise estimate
exceeds the noisy threshold is classified noisy. The raw block uses 5.0 DN and
10.0 DN; the calibrated block uses 1.0e-4 and 0.005 in I/F. An image is no
longer clean once more than 30% of its pixels are missing or more than 80% are
overexposed.

**Corrections applied at load.** None beyond what the host reader does. The
data array is used as the archive supplies it.

**Provisional values.** The calibrated block's blank and noisy thresholds, the
expected-noise and read-noise figures in the raw block, and the
magnitude-offset table are all placeholders awaiting calibration. They are
starting guesses, not measurements, and an image classified marginally blank or
noisy is worth looking at by eye before the classification is believed.

Field of view and geometry
==========================

**Extended-FOV margins.** The margin is how far outside the frame the model is
generated, and therefore the largest offset a search can find. It is keyed by
image size, because the archive holds full frames and on-chip summed frames and
a margin that is right for one is wrong for the other:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Image size
     - NAC margin ``[v, u]``
     - WAC margin ``[v, u]``
   * - 256
     - ``[13, 25]``
     - ``[5, 10]``
   * - 512
     - ``[25, 50]``
     - ``[5, 10]``
   * - 1024
     - ``[50, 140]``
     - ``[5, 10]``

**Camera rotation.** Rotation fitting is off for both cameras. Offsets are
two degrees of freedom, ``(dv, du)``, and no rotation is reported. That is a
measured decision rather than a default: the twist is one common value, so
nothing per-frame is left for a fit to find.

**Measured twist.** Over fifty narrow angle star frames the twist is
+0.0115 +/- 0.0002 degrees, which displaces the field corner by 0.15 pixels,
with a frame-to-frame scatter of 0.039 pixels. The wide angle camera measures
-0.0115 +/- 0.0002 degrees over forty-six frames, 0.15 pixels at the corner in
the opposite direction, scatter 0.037 pixels. Both are far below the threshold
at which a per-frame fit would be worth its cost.

**Residual distortion.** After the host's distortion model is applied, what is
left is a few hundredths of a pixel and close to radially symmetric: the narrow
angle camera measures a radial RMS of 0.048 pixels against a
centroid-and-astrometry floor of 0.081, and the wide angle camera 0.086 against
a floor of 0.053. See :doc:`/fov_distortion_report/fov_distortion_report` for
the method, the coefficients and the figures.

Metadata fields
===============

Beyond the keys every instrument writes -- image path and name, the start,
midtime and end of the exposure in UTC and in TDB seconds, the image shape,
the camera, the exposure time and the instrument host and instrument LIDs --
a Cassini ISS record carries:

* ``shutter_mode`` -- ``NACONLY``, ``WACONLY`` or ``BOTSIM``, the last when
  both cameras were exposed at once, sharing one spacecraft attitude.
* ``start_time_sclk``, ``midtime_sclk``, ``end_time_sclk`` -- the start,
  middle and end of the exposure as spacecraft clock counts: the label's own
  start and stop counts, which mark the start and the end of the exposure, in
  seconds of the clock with the 1/256-second ticks as a fraction, and the
  count exactly halfway between them. The ``times`` block's clock strings are
  computed from the exposure times and can differ from these counts by a
  fraction of a second.
* ``filters`` -- two entries, the two filter wheels, in that order, for
  example ``["CL1", "CL2"]``.
* ``sampling`` -- the on-chip summing mode: ``FULL``, ``SUM2`` or ``SUM4``.
* ``gain_mode`` -- the commanded gain state: ``0`` for 215 electrons per DN,
  ``1`` for 95, ``2`` for 29 and ``3`` for 12, or null for a label naming any
  other.
* ``description`` and ``observation_id`` -- the label's free text and the
  observation this frame belongs to; either may be null when the label carries
  none.

A Cassini ISS record also carries the facts in the table below, copied from the
label stored inside the calibrated image file, which is the label the
navigation reads. Each key is the name of the matching attribute in the
Cassini PDS4 dictionary. Each value is the label's own, in the label's own
form:

* A number stays a number and text stays text. The label writes its numbers
  to six or seven significant digits, so the detached PDS3 label file beside
  the image can show more digits of the same value.
* A time is the label's own text: UTC, with the day of the year, ending in
  ``Z``, for example ``2007-312T21:41:14.946Z``. The product creation time is
  the exception, as its row says.
* A label keyword holding two or four values is split into one key per value.
* A value the label writes as ``N/A``, ``UNK``, ``--`` or ``-999.0`` is
  recorded as written; each means the information was not available.
* A fact whose keyword the label lacks is null.

.. list-table::
   :header-rows: 1
   :widths: 28 42 20 10

   * - Key
     - Meaning
     - Label keyword
     - Unit
   * - ``mission_phase_name``
     - The mission phase the image belongs to, for example ``TOUR``.
     - ``MISSION_PHASE_NAME``
     - none
   * - ``spacecraft_clock_count_partition``
     - The spacecraft clock partition the two clock counts belong to.
     - ``SPACECRAFT_CLOCK_CNT_PARTITION``
     - none
   * - ``spacecraft_clock_start_count``
     - The spacecraft clock count at shutter open, as text: seconds, a
       period, then three digits of 1/256-second ticks.
     - ``SPACECRAFT_CLOCK_START_COUNT``
     - none
   * - ``spacecraft_clock_stop_count``
     - The spacecraft clock count at shutter close, in the same form.
     - ``SPACECRAFT_CLOCK_STOP_COUNT``
     - none
   * - ``antiblooming_state_flag``
     - Whether antiblooming was on: ``ON`` or ``OFF``.
     - ``ANTIBLOOMING_STATE_FLAG``
     - none
   * - ``bias_strip_mean``
     - The mean of the overclocked pixels, over every line but the first and
       last.
     - ``BIAS_STRIP_MEAN``
     - DN
   * - ``calibration_lamp_state_flag``
     - Whether the calibration lamp was on: ``ON`` or ``OFF``, or ``N/A`` for
       the narrow angle camera, which has no lamp.
     - ``CALIBRATION_LAMP_STATE_FLAG``
     - none
   * - ``command_file_name``
     - The instrument operations file that described the observation.
     - ``COMMAND_FILE_NAME``
     - none
   * - ``command_sequence_number``
     - The trigger number of the commands that took the image.
     - ``COMMAND_SEQUENCE_NUMBER``
     - none
   * - ``dark_strip_mean``
     - The mean of the extended (dark) pixels, over every line but the first
       and last.
     - ``DARK_STRIP_MEAN``
     - DN
   * - ``data_conversion_type``
     - How the 12-bit data were reduced to 8 bits: ``12BIT`` (not reduced),
       ``TABLE`` (by look-up table) or ``8LSB`` (keeping the 8 least
       significant bits).
     - ``DATA_CONVERSION_TYPE``
     - none
   * - ``delayed_readout_flag``
     - Whether the image may have waited on the detector while the other
       camera read out: ``YES`` or ``NO``.
     - ``DELAYED_READOUT_FLAG``
     - none
   * - ``detector_temperature``
     - The temperature of the detector.
     - ``DETECTOR_TEMPERATURE``
     - degrees C
   * - ``electronics_bias``
     - The commanded electronics bias, which keeps every DN above zero.
     - ``ELECTRONICS_BIAS``
     - none
   * - ``earth_received_start_time``
     - When the earliest data of the image were received on Earth.
     - ``EARTH_RECEIVED_START_TIME``
     - UTC
   * - ``earth_received_stop_time``
     - When the latest data of the image were received on Earth.
     - ``EARTH_RECEIVED_STOP_TIME``
     - UTC
   * - ``expected_maximum_full_well``
     - The maximum DN predicted for the image, as a percentage of the
       full-well level, ``valid_maximum_full_well``.
     - ``EXPECTED_MAXIMUM``, first value
     - percent
   * - ``expected_maximum_DN_sat``
     - The maximum DN predicted for the image, as a percentage of the
       saturation level, ``valid_maximum_DN_sat``.
     - ``EXPECTED_MAXIMUM``, second value
     - percent
   * - ``expected_packets``
     - The number of telemetry packets expected for the image, each 7616
       bits.
     - ``EXPECTED_PACKETS``
     - packets
   * - ``exposure_duration``
     - The exposure duration as the label states it. ``exposure_time`` is the
       same duration in seconds, except that a zero-length exposure is
       recorded there as 0.000001.
     - ``EXPOSURE_DURATION``
     - milliseconds
   * - ``filter_temperature``
     - The temperature of the filter wheels.
     - ``FILTER_TEMPERATURE``
     - degrees C
   * - ``flight_software_version_id``
     - The version of the instrument flight software.
     - ``FLIGHT_SOFTWARE_VERSION_ID``
     - none
   * - ``gain_mode_id``
     - The gain setting as the label names it, for example
       ``29 ELECTRONS PER DN``; ``gain_mode`` is the same setting as a number.
     - ``GAIN_MODE_ID``
     - none
   * - ``ground_software_version_id``
     - The version of the ground software that built the image.
     - ``SOFTWARE_VERSION_ID``
     - none
   * - ``image_mid_time``
     - The middle of the exposure.
     - ``IMAGE_MID_TIME``
     - UTC
   * - ``image_time``
     - Shutter close.
     - ``IMAGE_TIME``
     - UTC
   * - ``image_observation_type``
     - The purposes of the image, for example ``SCIENCE``: text for one
       purpose, an array of text for several.
     - ``IMAGE_OBSERVATION_TYPE``
     - none
   * - ``instrument_data_rate``
     - The rate at which data left the camera.
     - ``INSTRUMENT_DATA_RATE``
     - kilobits per second
   * - ``inst_cmprs_type``
     - The on-board compression: ``LOSSLESS``, ``LOSSY`` or ``NOTCOMP`` (not
       compressed).
     - ``INST_CMPRS_TYPE``
     - none
   * - ``inst_cmprs_param_malgo``
     - The lossy compression algorithm; ``N/A`` when the image was not lossy
       compressed, as for the next three keys.
     - ``INST_CMPRS_PARAM``, first value
     - none
   * - ``inst_cmprs_param_tb``
     - The lossy compression block type.
     - ``INST_CMPRS_PARAM``, second value
     - none
   * - ``inst_cmprs_param_blocks``
     - The number of blocks per group in lossy compression.
     - ``INST_CMPRS_PARAM``, third value
     - none
   * - ``inst_cmprs_param_quant``
     - The lossy compression quantization factor.
     - ``INST_CMPRS_PARAM``, fourth value
     - none
   * - ``inst_cmprs_rate_expected_bits``
     - The average number of bits per pixel expected after compression.
     - ``INST_CMPRS_RATE``, first value
     - bits per pixel
   * - ``inst_cmprs_rate_actual_bits``
     - The average number of bits per pixel received.
     - ``INST_CMPRS_RATE``, second value
     - bits per pixel
   * - ``inst_cmprs_ratio``
     - The expected image size over the size received; ``N/A`` for an image
       that was not compressed.
     - ``INST_CMPRS_RATIO``
     - none
   * - ``light_flood_state_flag``
     - Whether the detector was light flooded just before the image: ``ON``
       or ``OFF``.
     - ``LIGHT_FLOOD_STATE_FLAG``
     - none
   * - ``method_description``
     - The information or algorithm used to choose the exposure.
     - ``METHOD_DESC``
     - none
   * - ``missing_lines``
     - The number of missing or incomplete image lines; ``N/A`` for a lossy
       compressed image.
     - ``MISSING_LINES``
     - lines
   * - ``missing_packet_flag``
     - Whether telemetry packets the image needed were missing: ``YES`` or
       ``NO``.
     - ``MISSING_PACKET_FLAG``
     - none
   * - ``optics_temperature_front``
     - The temperature of the front optics.
     - ``OPTICS_TEMPERATURE``, first value
     - degrees C
   * - ``optics_temperature_back``
     - The temperature of the rear optics; ``-999.0`` for the wide angle
       camera, which has no rear optics sensor.
     - ``OPTICS_TEMPERATURE``, second value
     - degrees C
   * - ``order_number``
     - The image's identifier within its instrument operations file.
     - ``ORDER_NUMBER``
     - none
   * - ``parallel_clock_voltage_index``
     - The commanded parallel clock voltage index.
     - ``PARALLEL_CLOCK_VOLTAGE_INDEX``
     - none
   * - ``pds3_product_creation_time``
     - When the raw image product was built on the ground. The archive states
       this time in Pacific local time, not UTC, although some labels end it
       with ``Z``.
     - ``PRODUCT_CREATION_TIME``
     - Pacific local time
   * - ``pds3_product_version_type``
     - The product's version type; ``FINAL`` for every archived product.
     - ``PRODUCT_VERSION_TYPE``
     - none
   * - ``pds3_target_desc``
     - The intended target the exposure was chosen for.
     - ``TARGET_DESC``
     - none
   * - ``pds3_target_list``
     - Always ``N/A``: the archive does not list the bodies in view.
     - ``TARGET_LIST``
     - none
   * - ``pds3_target_name``
     - The target named when the observation was planned, which is often not
       what the image shows.
     - ``TARGET_NAME``
     - none
   * - ``prepare_cycle_index``
     - The entry of the prepare-cycle table used for the image.
     - ``PREPARE_CYCLE_INDEX``
     - none
   * - ``readout_cycle_index``
     - The entry of the readout-cycle table used for the image.
     - ``READOUT_CYCLE_INDEX``
     - none
   * - ``received_packets``
     - The number of telemetry packets received for the image, each 7616
       bits.
     - ``RECEIVED_PACKETS``
     - packets
   * - ``sensor_head_electronics_temperature``
     - The temperature of the sensor head electronics.
     - ``SENSOR_HEAD_ELEC_TEMPERATURE``
     - degrees C
   * - ``sequence_id``
     - The spacecraft sequence the image belongs to, for example ``S35``.
     - ``SEQUENCE_ID``
     - none
   * - ``sequence_number``
     - Where the image falls in the order its observation planned.
     - ``SEQUENCE_NUMBER``
     - none
   * - ``sequence_title``
     - The name of the activity the image belongs to; ``--`` when none was
       given.
     - ``SEQUENCE_TITLE``
     - none
   * - ``shutter_state_id``
     - Whether the shutter was enabled: ``ENABLED`` or ``DISABLED``. When it
       was disabled, the label's start, middle and stop times are all the
       start of the exposure window.
     - ``SHUTTER_STATE_ID``
     - none
   * - ``start_time_doy``
     - Shutter open.
     - ``START_TIME``
     - UTC
   * - ``stop_time_doy``
     - Shutter close; the same as ``image_time``.
     - ``STOP_TIME``
     - UTC
   * - ``telemetry_format_id``
     - The telemetry mode, for example ``S&ER3``, or ``UNK``.
     - ``TELEMETRY_FORMAT_ID``
     - none
   * - ``valid_maximum_full_well``
     - The minimum full-well saturation level, which may exceed 4095.
     - ``VALID_MAXIMUM``, first value
     - DN
   * - ``valid_maximum_DN_sat``
     - The saturation level of the analog-to-digital converter: 4095 or 255.
     - ``VALID_MAXIMUM``, second value
     - DN

Its ``instrument`` is ``coiss`` and its ``camera`` is ``NAC`` or ``WAC``. The
instrument host LID is ``...:instrument_host:spacecraft.co``. The instrument
LID encodes the camera: ``...:instrument:issna.co`` for the narrow
angle camera and ``...:instrument:isswa.co`` for the wide angle camera.

Corrected-pointing C-kernels
============================

**What is corrected.** The corrected object is **-82000, the spacecraft bus**.
A corrected kernel carries the bus attitude the navigation implies, so no
camera frame is fabricated and no frame kernel has to change. The camera frames
the correction is measured in are ``CASSINI_ISS_NAC`` and ``CASSINI_ISS_WAC``.
Segment time tags are encoded against spacecraft clock **-82**.

**Running it.**

.. code-block:: bash

   sd_create_ck coiss \
       --nav-results-root /data/nav/results \
       --kernel-dir $SPICE_PATH/Cassini \
       --kernel-dir $SPICE_PATH/Cassini/SCLK \
       --kernel-dir $SPICE_PATH/Cassini/FK \
       --kernel-dir $SPICE_PATH/Cassini/CK-reconstructed \
       --output-dir /data/nav/ck

Directories are not searched recursively, which is why a Cassini holdings tree
needs one flag per kernel kind. The first contributes the leapseconds kernel,
which sits at the top of the tree; the next two contribute the spacecraft clock
and frame kernels the navigation recorded; the last is the set of originals to
pair images against.

**Baseline kernel naming.** Cassini basenames declare a kernel's class in a
release code following the two dates the kernel spans: ``p`` for planned
pointing, ``r`` for reconstructed, plus a letter distinguishing successive
releases of one span. Two date conventions are in use and the digit count tells
them apart -- the tour and the cruise stamp ``YYDOY_YYDOY``, the Jupiter flyby
stamps ``YYMMDD_YYMMDD``, and the earliest flyby release omits the code
altogether. Gapfill kernels are ``pa`` names carrying ``_gapfill_vN``. When
several kernels reproduce one image's attitude, which the overlapping
reconstructed, gapfill and predicted sets make ordinary, reconstructed is
preferred over gapfill over predicted.

**Angular velocity.** Every -82000 segment in the reconstructed baselines
carries angular velocity -- 2645 of 2645 measured locally -- so no Cassini
image is refused for a baseline that supplies pointing without a rate.

**Segment shape.** A segment carries records at the exposure start, midtime and
stop, plus a one-second cadence once the exposure reaches ten seconds. The
attitude is time-varying: the correction is held body-fixed and composed onto
the baseline's own pointing at each record epoch.

**Omission reasons this instrument produces.** ``not_eligible``,
``botsim_loser``, ``no_reproducing_baseline`` and ``baseline_coverage_gap``.
``rotation_unsupported`` never appears, because rotation fitting is off for
both cameras.

``botsim_loser`` follows from the corrected object being the bus. A BOTSIM
exposure produces two frames, one per camera, sharing one bus attitude, and one
attitude cannot carry two different corrections. The narrow angle member keeps
its correction and the wide angle member yields. A wide angle frame yields only
to a partner that actually writes: one whose narrow angle partner is
ineligible, or has no reproducing baseline, keeps its own correction rather
than losing it to nothing.

**Interpolation error.** Not yet measured for this instrument. What is known is
the shape rather than the size: the error is zero at every record epoch, grows
between them, is largest where the baseline's own rate changes inside the
window, and shrinks as records are added. Two things make the size a
per-camera question here. The error is an angle, and a wide angle pixel
subtends about ten times what a narrow angle pixel does, so the same angular
error is roughly ten times fewer pixels on the wide angle camera. And how much
attitude structure a segment spans depends on how the bus was slewing during
the exposure, which the tour varies widely. A consumer who evaluates geometry
at the exposure midtime is exact and unaffected; one who needs a bound at
arbitrary interior epochs should measure it on the frames they care about, by
comparing a corrected segment against its baseline at epochs between the
records with the correction set to zero.

Known limitations
=================

* The saturation policy on calibrated frames leaves the saturation fraction at
  zero and disables the fully-overexposed early-out, and the raw frames that
  would carry a real ceiling are not reachable from the command line.
* The blank and noisy thresholds in the calibrated block are placeholders, so
  the blank and noisy classifications are provisional.
* A BOTSIM wide angle frame receives no corrected segment whenever its narrow
  angle partner writes one. Its pixel offset in the metadata is unaffected.
* PDS4 bundles are shipped for the Saturn dataset only; the cruise dataset
  names a label template directory that is not part of the distribution.

References
==========

* The COISS volumes' ``document/`` directory on the PDS Ring-Moon Systems
  Node, which carries the ISS Data User's Guide and the volume ``*_index.lbl``
  column definitions this pipeline reads.
* C. C. Porco et al., "Cassini Imaging Science: Instrument Characteristics and
  Anticipated Scientific Investigations at Saturn", *Space Science Reviews*
  115, 363-497 (2004).
* The Cassini ISS calibration reports distributed with the CISSCAL software,
  for what the calibrated product's I/F values mean.
* :doc:`/fov_distortion_report/fov_distortion_report` -- the measured twist and
  residual distortion quoted above.
