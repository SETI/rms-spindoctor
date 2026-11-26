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
   * - Sim instrument keys
     - ``vgiss_1_nac``, ``vgiss_1_wac``, ``vgiss_2_nac``, and ``vgiss_2_wac``
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

Beyond that it reads the flat ``voyager_iss`` config section and resolves the
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
``saturation_dn: 255``, ``read_noise_dn``, and ``star_flux_dn_per_s_vmag0`` are
all DN quantities. The
DN values describe the detector, which is real; the pipeline navigates the
scaled I/F product, so ``NavOrchestrator._build_saturation_mask`` returns an
empty mask for this instrument as for any ``calibrated_if`` input, and
``saturation_dn`` is never consulted. There is no ``saturation_threshold_if``
and adding one would not restore the mask: the mask is gated on the units, not
on the presence of the key. There is no raw product this pipeline can navigate
instead.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers:
``read_noise_dn``, ``blank_max_if``,
``noisy_threshold_if``, and the ``mag_offset_table`` entry. The ``_sources``
convention is not used in this file.

``fit_camera_rotation: false`` carries a three-line comment saying it is kept
off because rotation fitting is too slow, that the instrument does carry
non-negligible attitude rotation residuals, and that it should be revisited
once the rotation search is fast enough. That comment is the whole rationale
and should move with the flag if the flag moves.

Photometric and PSF calibration
===============================

**Limiting magnitude.** Photometric, from the ``voyager_iss_star_photometry`` section
(:class:`~spindoctor.obs.star_photometry_voyager_iss.VoyagerISSStarPhotometrySection`),
which has one entry per camera, ``vg1_nac``, ``vg1_wac``, ``vg2_nac``, and
``vg2_wac``, chosen by the label's spacecraft digit and the oops detector. A vidicon
counts no electrons, so the rate is in raw DN, and the gain state multiplies it. A
star of V magnitude ``m`` puts

.. math::

    \mathrm{DN} = R \cdot 10^{-0.4\,(m + f)} \cdot t_{\mathrm{exp}} \cdot k

in the raw frame, where ``R`` is the camera's ``flux_dn_per_s_vmag0`` (raw DN per
second for a V = 0 star through CLEAR at low gain), ``f`` is the frame's filter's
``filter_mag_offset``, and ``k`` is the ``dn_gain_factor`` of its gain state. The
dimmest visible star puts ``min_detectable_dn`` in the frame, and the limit is that
star's magnitude plus ``search_depth_mag`` (1.0). The navigator's image is in I/F,
but the limit depends only on the ratio of the rate to ``min_detectable_dn``, and
those share a scale. The filter and the gain state (``LO GAIN`` or ``HI GAIN``, read
as ``LOW`` or ``HIGH``) come from the VICAR label's ``LAB03`` record. A frame whose
filter or gain state the camera's entry lacks is refused, and a non-positive
exposure returns ``-inf``, since it records no starlight.

.. list-table:: Each camera's values, and its limit at 1 s through CLEAR at low gain
   :header-rows: 1

   * - Camera
     - Rate (DN/s)
     - High-gain factor
     - ``min_detectable_dn``
     - Limit at 1 s
   * - Voyager 1 narrow angle
     - 5.61e4
     - 3.92
     - 110
     - V 7.8
   * - Voyager 1 wide angle
     - 2.74e3
     - 3.94
     - 35
     - V 5.7
   * - Voyager 2 narrow angle
     - 8.66e4
     - 4.24
     - 110
     - V 8.2
   * - Voyager 2 wide angle
     - 4.6e3
     - 4.04
     - 35
     - V 6.3

The Voyager 2 values were measured in 2026 on star fields of its cruise and
encounters, read through the navigator's own reader and photometered on the raw
vidicon frame. Each frame's pointing was found from the stars alone, by a vote over
star and peak pairs refined to a translation and a rotation, and a frame was kept
when three or more stars agreed within 1.5 px and the chance of such a cluster
anywhere in the 400 px search window was below one in a thousand. That kept 37
narrow-angle and 51 wide-angle frames, and one frame in three was held out of the
fit.

- The narrow-angle rate is the fit of 53 stars (+-0.07 mag). The wide-angle rate
  rests on 13 bright Tycho stars (+-0.20 mag). Fainter wide-angle stars, V 8 to 10,
  read 1.1 to 1.5 mag brighter than that rate predicts, which points to a
  compressive vidicon response at high signal.
- ``min_detectable_dn`` is where a detection-probability curve, fitted to every clean
  star, reaches 80% of the way from its floor to its plateau: 109 +- 67 DN on the
  narrow-angle camera and 35 +- 15 DN on the wide-angle camera.
- Every measured frame is at low gain, so the high-gain factor is the published
  constant ``g`` of Benesh and Jepsen (1978, Table 4-3) for the camera's serial
  number.
- The filter offsets are 2.5 log10 of each camera's solar filter factors (Benesh
  and Jepsen 1978). The narrow-angle VIOLET offset is measured instead, 1.68 +- 0.24
  mag from 4 stars, against the published 1.34.

Voyager 1 gave only three such frames, too few to fit. Its rates are Voyager 2's for
the same kind of camera, scaled by the bench sensitivities Benesh and Jepsen (1978)
measured before launch: the clear, low-gain light-transfer slope in DN per
foot-lambert-second over the solar conversion factor of each serial number. Voyager
1's cameras are S/N 07 (narrow angle) and S/N 06 (wide angle), Voyager 2's S/N 05
and S/N 04, so the narrow-angle rate is scaled by 0.648 and the wide-angle rate by
0.600. Voyager 1 keeps its own filter factors and high-gain constants, and takes
Voyager 2's ``min_detectable_dn``.

On the held-out Voyager 2 frames, the faintest star detected lies a median 0.02 mag
inside the predicted dimmest visible star on the narrow-angle camera, and 1.37 mag
past it on the wide-angle camera, with broad spreads, so the limit carries about half
a magnitude of uncertainty.

``star_min_usable_vmag`` is 0.0: no bright-end cutoff, and saturated stars are
handled downstream.

**PSF.** Each camera's star sigma is its entry's ``star_psf_sigma``, in pixels of the
geometrically corrected image. On Voyager 2 it is the median Gaussian sigma of stars
on frames read out at 5:1: 1.2 px for the narrow-angle camera (53 stars on 15
frames) and 1.1 px for the wide-angle camera (213 stars on 26 frames). A star's core
is about 0.8 px, and platform motion elongates a long exposure. The Voyager 1 values
are a 3.0 px placeholder, since its three frames hold too few stars to fit. The
camera block's ``star_psf_sigma`` is read only by the simulator, which emulates all
four cameras from that block. ``star_psf_sizes`` is a single ``100: [7, 7]``, so
every star gets a 7x7 cutout regardless of magnitude and regardless of camera.

**Magnitude offsets.** ``fallback_combo`` is ``'CL'`` and the table carries one
entry with a default of 0.0. Both are placeholders.

**Photometric zero point.** ``star_flux_dn_per_s_vmag0`` is 3.0e3, and its
comment records why the units are DN rather than electrons: a vidicon has no
electron domain, so it renders point sources directly in DN.

**Recalibrating.** The measurement tools are in ``util/star_photometry``; its Voyager
frame selection and the fit are described in its module docstrings. The cohort's one
star frame cannot constrain a kernel through the geometric resampling: eight usable
cutouts survive, and the flat-top guard leaves only two simulated cutouts to compare
against. Rerun the fit on new star fields, on each camera of each spacecraft, and
update the camera's entry and its comments together.

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
configuration refers to a spacecraft.

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
lookup at its own midtime reproduces, and is reported as
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

**Instrument keys.** One key for each spacecraft and camera: ``vgiss_1_nac`` and
``vgiss_1_wac`` for Voyager 1, ``vgiss_2_nac`` and ``vgiss_2_wac`` for Voyager 2.
The four entries sit in ``src/spindoctor/sim/instruments/configs/vgiss.yaml``.
Each entry states every value itself and none takes a value from another, because each
vidicon and telescope is unique. The optical values (the PSF, the residual
distortion, the plate scale, the star zero point, and the vidicon detector values)
are per camera. Apart from the Voyager 2 cameras' star zero points, they hold the
same numbers in all four entries today, and a comment beside each says where the
number comes from and that it is not yet measured for that camera. The values
that do not describe a camera's optics (the
data units, the navigator link, the readout modes, and the artifact-mode
availability) are repeated in each entry as well, because a shared parent would
have to be one of the cameras. Every analysis is per camera too: the realism match,
the field-of-view distortion, and the rotation each use one camera's frames. No
simulator entry holds a rotation value today. All four emulate the one ``voyager_iss``
section of the navigator configuration. The realism match pairs the image
library's two Voyager 1 wide angle frames with ``vgiss_1_wac`` and its one Voyager 2
narrow angle frame with ``vgiss_2_nac``; the other two entries have no frame in the
library. A library sidecar for a Voyager frame gives its spacecraft (``VG1`` or
``VG2``) beside the mission and the camera, so the match can tell the four cameras
apart. The entry states ``data_units: calibrated_if``, the units of the
GEOMED product and of the ``voyager_iss`` block, so a simulated frame is in I/F.

**Realism match.** The match compares each camera with its own library frames
(the 2026-11-25 record in ``tests/integration/realism_results/``). The two
entries without a library frame have no row.

* ``vgiss_1_wac``, two Saturn frames. The simulated sky noise is 3.1e-3 I/F
  against 1.7e-3 real (W1/IQR 3.8, over 485 simulated and 409 real sky patches),
  the sky level above the floor is 7.6e-3 against 5.7e-3 I/F (W1/IQR 1.6), and
  the noise at signal is 3.0e-3 against 2.1e-3 I/F (W1/IQR 2.5). One of the two
  frames is filled by Saturn's cloud deck, so its darkest quarter of tiles is
  cloud.
* ``vgiss_2_nac``, one star field. The simulated sky noise is 3.1e-3 I/F against
  6.9e-3 real (W1/IQR 2.7, over 241 simulated and 203 real sky patches), the sky
  level above the floor is 7.3e-3 against 2.1e-2 I/F (W1/IQR 1.8), and the noise
  at signal is 3.0e-3 against 9.0e-3 I/F (W1/IQR 4.4). The realism match's star
  cutouts are its own, separate from the kernel fit's under PSF kernel below: the
  real frame gives 6 (EE50 median 1.26 px), and its matched frame holds the
  field's catalog stars, but none yields a usable simulated cutout, so the match
  leaves the PSF unconstrained. On the real frame the star-field fit locks at
  0.94 on 7 of 11 predicted stars. On the matched frame it predicts 5 and locks
  at 0.6, a wide-offset lock on 3 inliers.

The simulated values are nearly the same for both cameras, because their entries
hold the same numbers. The real frames differ: the narrow angle frame's sky is
about four times as noisy as the wide angle frames'. Neither sky comparison is
like for like. The real frames are resampled, which correlates neighboring
pixels, so the paired-difference estimate reads their noise about a third low,
and the matched scenes don't turn on the simulator's resampling mode.

**Star zero points.** The Voyager 2 zero points are measured on each camera's own
star frames. Each frame is pointed by its own stars, and each star's raw DN is
read on the archive's cleaned product. Through the clear filter, a magnitude-0 star
gives 8.7e4 raw DN per second on the narrow angle camera and 4.6e3 on the wide
angle camera. The narrow angle value comes from faint stars, where the vidicon
responds linearly. Brighter stars read low, because the vidicon compresses at
high signal: on C1205021 the two brightest Pleiades stars peak at 190 and 140 DN
and hold about two thirds of what the linear rate predicts. The wide angle value
comes from stars of magnitude 2 to 7, the stars its frames navigate on. Fainter
stars read brighter for their magnitude, but part of that is the catalog's own
error at magnitudes 8 to 9, so the wide angle camera's rate for faint stars is
not settled. The simulated vidicon is linear up to its full scale, so a bright
simulated star reads brighter than a real one, up to the full scale, where it
clips. The Voyager 1 cameras keep the
3.0e3 estimate from the published limiting magnitudes, since no Voyager 1 star
frame has been measured.

The simulator's vidicon chain works in raw DN, and it converts DN to I/F at 127.5
DN per unit I/F at every exposure. The archive's calibration depends on the
exposure and the filter: at 0.48 s through the clear filter it maps about 30 raw
DN to I/F 1 on the narrow angle camera and about 150 on the wide angle camera. So
a simulated narrow angle star at that exposure has the real star's raw DN but
reads about four times fainter in I/F, while a wide angle star reads about the
same.

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
Voyager 2 wide angle measurement. All four entries carry it, which
under-represents the spread across the four cameras' optics until each camera
has its own measurement from locked frames.

**Artifact-mode availability.** The instrument entry's ``artifact_modes.offered``
lists its modes, and the simulator guide's mode tables
(:ref:`sim-telemetry-modes` and :ref:`sim-detector-modes`) give each mode's
cameras. Its detector model is ``vidicon``, so the modes that act in the CCD
chain alone are unavailable to it. The entry states the reason for
``fixed_pattern``: the vidicon DN chain has no fixed-pattern stage, so if a
scene asks for the mode, validation fails with that reason. ``reseau_scars``
(the reseau-removal patches left by archive processing) and
``resample_texture`` (the geometric resample the pipeline's own input has been
through) are telemetry-stage modes applied after the structured-loss modes,
because they emulate archive processing rather than a detector or transmission
defect.

The entry's shape defaults give ``edited_frame`` a centered band 550 pixels wide.
The edited modes downlink a centered band of 608, 440, 272, 160, or 80 of the 800
samples in a raw line. The GEOMED resample spreads those 800 samples over 1000
pixels, so the 440-sample band covers 550 pixels of the GEOMED frame. A scene that
sets only a ``probability`` gets that band. ``instrument_defaults`` turns on
``banding``, which draws the published coherent component as vertical bands of
0.25 DN at 8 pixels. The shape defaults give
``reseau_scars`` 202 marks on a 59-pixel lattice. The camera's 202 reseau marks
sit about 47 pixels apart in a raw frame, and the GEOMED resample stretches that
to about 59 pixels.
``native_frame_size_px`` is 1000, the GEOMED product's size, which the realism
match uses to tell a full frame from a subframe.

**Units.** A frame from any of the four cameras is in I/F, with a ``0`` missing-data
marker. The simulator reads the same configuration block as real navigation, so a
simulated frame carries the units and marker of the GEOMED product: the product stores
I/F as scaled integers and fills lost pixels and its resample border with zero, about
10% of each frame. The calibrated Cassini instruments share the I/F units and the ``0``
marker. The four Voyager entries keep a full well, the 255 DN clip that sets a 1.85 I/F
ceiling before the calibration inverse (see :ref:`Instruments <sim-instruments>`). An
author should expect a frame to span about 0.012 to 1. A tool scaled for a DN frame
shows a Voyager frame as black.

The calibration inverse subtracts the bias pedestal and then adds a calibrated
sky residual of 1.5 DN, 0.0118 I/F, so the dark sky sits above the marker, as on
the real products. In five archived frames, four Voyager 2 star frames from
``VGISS_8210`` and one Voyager 1 dark frame from ``VGISS_5101``, the central 800
by 800 pixels sit on a positive calibration residual between 6.6e-4 and 3.2e-2
I/F: the first percentile is above zero in every frame, and at most 0.03% of
those pixels are exactly 0. Half a DN off the whole-DN grid, the simulated
residual keeps every digitized pixel off 0. Under ``instrument_defaults`` the
vidicon read noise and banding (0.40 DN rms) take a sky pixel below 0 only on a 1.5 DN
excursion, about 7 pixels in 100,000, so the sky's first percentile stays
positive.

**Cosmic rays.** The vidicon's cosmic-ray stage reads the rate as events per
pixel per second of exposure, with no readout dwell, and each hit saturates the
pixels it lands on, with no diffusion (see :ref:`sim-detector-stage`). The
catalog rate is a retained zero. The cohort
measures a 3.5e-5 transient fraction, mostly interpolated away by the
reseau-anchored resampling, so it can't say how many hits the target took.

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

**Simulator guard scenes.** The ``full_frame_cameras`` scene class (see
:ref:`sim-full-frame-cameras`) holds one scene for each of the four cameras, each
at the full 1000 px GEOMED frame with the camera's vidicon chain. Each follows a
real frame of its camera, which its comments cite, and holds what that frame
navigates on: a moon, Saturn, or stars. The offsets are 32 to 49 px, the size of
the image library's Voyager offsets, and point in four different directions. The
Voyager 2 scenes plant each camera's measured mean twist. No Voyager 1 twist is
measured, so the two Voyager 1 scenes plant a placeholder roll of 0.023 degrees,
the size of the frame-to-frame twist scatter on the Voyager 2 cameras (0.28
pixels at the corner). Both Voyager 1 scenes plant the same value because
rotation is grouped per spacecraft. The only measurement points the other way:
the mean twists of the two Voyager 2 cameras differ (0.030 and 0.358 degrees), so a
measured Voyager 1 twist may differ between the two cameras by far more than the
placeholder. Roll fitting stays off, as the cameras ship.

The simulated frames are in I/F, and the exposure does not change a simulated
body's brightness. A simulated moon or planet peaks near I/F 1, where the real
frames peak at 0.3 to 0.5.

- ``vgiss_1_nac_disc``: a moon 197 px across at a phase of 32 degrees, after a
  Voyager 1 frame of Ganymede at 0.24 s, offset 42 px. The answer lands 0.06 to
  0.57 px off over 24 draws, at medium confidence, from the disc correlation, the
  limb fit, and the blob together.
- ``vgiss_1_wac_saturn``: Saturn about 95 px across with its rings at an opening
  of 8.6 degrees, after two library frames at 7.68 s, offset 32 px. The scale,
  the ring radii, and the ring brightness follow the real frame, and the planet's
  disc follows the real frame's flatter limb with a Minnaert law. The real frames
  navigate on the planet, and the navigator models the rings only as an annulus,
  which it flags spurious, so the scene marks no ring navigable. As on the real
  frame, the disc correlation passes its quality threshold. It carries the
  answer, 0.18 to 0.69 px off over 24 draws, at medium confidence. The planet is
  under 100 px across, so the simulated body model gives it no limb, and the limb
  fit does not run, where on the real frame it does.
- ``vgiss_2_nac_star_field``: the Pleiades field of the library frame C1205021 at
  0.48 s, offset 49 px. With the camera's measured star zero point, the star-field
  fit locks on 6 or 7 of 13 predicted stars at confidence 0.93, as on the real
  frame, and the answer lands 0.03 to 0.26 px off over 24 draws, at medium or high
  confidence. The two brightest stars peak at 224 and 152 DN above the sky,
  against 167 and 115 raw DN on the real frame, because the simulated vidicon
  does not compress them.
- ``vgiss_2_wac_star_field``: a Scorpius field at 0.48 s, after a frame of the
  twist measurement, offset 35 px. Its brightest star peaks 42 DN above the
  simulated sky, near the real frame's 44. The twist of +0.358 degrees moves the
  stars the navigator measures by 0.07 to 3.2 px, and their mean position by 1.6
  px, so the answer lands 1.0 to 1.4 px off over 24 draws. Without the twist it
  lands 0.21 px off.

The two scenes with a disc are the slowest in the class. Each takes about a
minute to render and navigate on a quiet machine and two to three minutes on a
busy one, because the disc correlation searches the camera's 400 px window. The
star fields take a few seconds.

**Unit tests.** ``tests/spindoctor/inst/test_inst_voyager_iss.py`` covers the
spacecraft-digit parser and all four of its refusals, the I/F factor parser and
all three of its refusals, each camera's star limit at 1 s through CLEAR at low gain
and its shifts with exposure, gain state, and filter, ``-inf`` for a non-positive
exposure, the refusal of a filter or gain state the camera's entry lacks, each
camera's star sigma, the
per-spacecraft metadata LIDs, and the published clock counts: fractional
leading units with no midtime count, read from the PDS3 label beside the image.

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

It also means deciding whether one bundle covers both spacecraft or each gets its own.

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
  ``read_noise_dn``, ``blank_max_if``,
  ``noisy_threshold_if``, and the ``mag_offset_table`` entry.
* ``fit_camera_rotation`` is off for cost against a measurement that says it
  should be on. Revisiting it means both speeding the rotation search and
  accepting the loss of corrected C-kernels for this instrument, or recording
  the rotation pivot so the two can coexist.
* The Voyager 1 star limits are scaled from Voyager 2 by bench sensitivities, and
  the Voyager 1 star sigma is a placeholder: only three Voyager 1 star frames gave a
  pointing fix.
* The wide-angle rate rests on 13 bright stars, and fainter stars read brighter than
  it predicts.
* The four simulator cameras carry the same distortion block and PSF kernel
  numbers, although each camera's optics are its own.
* The Voyager 1 simulator cameras keep an estimated star zero point, and the
  simulator converts vidicon DN to I/F at one scale for every exposure and
  filter, so for equal raw DN a simulated star's I/F is the real one's times
  the archive's scale divided by the simulator's (about 30 / 127.5 for the
  narrow angle camera at 0.48 s through the clear filter).
* The simulator's calibrated sky residual is one constant, 0.0118 I/F, where the
  real GEOMED sky level varies about 50x between frames.
* The Voyager 1 cameras have no twist or distortion measurement: no frame of
  the Voyager 1 cohort locked.
