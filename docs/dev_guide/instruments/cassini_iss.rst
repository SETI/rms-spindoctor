============
Cassini ISS
============

Code map
========

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Piece
     - Where
   * - Obs class
     - :class:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS`, registered
       under the instrument name ``coiss``
   * - DataSet classes
     - :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS`
       and its two subclasses
       :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSCruise`
       and
       :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSSaturn`,
       registered under ``coiss``, ``coiss_cruise`` and ``coiss_saturn`` plus
       a ``_pds3`` alias for each
   * - Config block
     - ``cassini_iss`` and ``cassini_iss_calib`` in
       ``config_400_inst_coiss.yaml``, each nested per detector
   * - Sim instrument keys
     - ``coiss_nac``, ``coiss_wac``, ``coiss_calib_nac``, ``coiss_calib_wac``
   * - Statistics key
     - ``coiss``
   * - Log key
     - ``other.obs`` governs the image-scoped ``LOAD IMAGE`` section; there is
       no per-instrument logging key
   * - oops host module
     - ``host_cassini.CassiniISS``, from the ``rms-host-cassini`` package

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.from_file` calls
``host_cassini.CassiniISS.from_file(path, fast_distortion=..., return_all_planets=...)``.
Both keyword arguments are read from ``**kwargs`` and default to ``True``.
``fast_distortion`` selects the host's polynomial inverse rather than an
iterative one, which is what makes per-pixel geometry affordable in a batch;
``return_all_planets`` makes the host list the kernels of every planet in the
observation's ``spice_kernels`` rather than those of Jupiter or Saturn alone.

The host records ``obs.image_url``, the absolute source URL, so a run against
remote holdings still records which file it read; the published image name
is taken from it. Beyond the host call ``from_file`` does three things:

1. Selects the configuration block. ``'_CALIB' in fc_path.name.upper()``
   chooses ``cassini_iss_calib``; anything else chooses ``cassini_iss``.
2. Indexes that block by ``obs.detector.lower()``. A missing section, or a
   section with no entry for the detector, raises ``ValueError`` listing the
   detectors that are present, rather than failing later on a missing key.
3. Resolves the extended-FOV margin. The configured value is a mapping keyed
   by image size, so the margin is ``entry[obs.data.shape[0]]``; an explicit
   ``extfov_margin_vu`` argument wins over the configuration entirely.

Label and index dependencies
============================

**Label fields read.**

* ``SPACECRAFT_CLOCK_START_COUNT`` and ``SPACECRAFT_CLOCK_STOP_COUNT``, read
  from the image's VICAR label in
  :meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.get_public_metadata`.
  A count is ``SECONDS.TICKS``: whole seconds, then the 1/256-second ticks
  past them written as three digits. ``_sclk_count`` converts it to an exact
  number of seconds through :func:`~spindoctor.support.sclk.fractional_count`,
  with the moduli ``(4294967296, 256)`` and offsets ``(0, 0)`` that the clock
  kernel ``cas00172.tsc`` gives, so ``1459229915.075`` is
  ``1459229915 + 75 / 256``. A tick field of fewer than three digits has lost
  its trailing zeros and is padded back on the right: ``1347929382.11`` is
  ``1347929382.110``. The two counts mark the start and the end of the
  exposure, so ``_published_sclk`` passes ``bracketed=True`` to
  :func:`~spindoctor.support.sclk.exposure_counts` and ``midtime_sclk`` is
  their exact mean. A count the label does not carry is published as ``None``,
  and so is the mean.
* ``SHUTTER_MODE_ID``, read by the ``shutter_mode`` property. A missing key or
  a null yields ``None``; a non-string value raises, because ``str()`` would
  serialize any object without complaint and the result would pass downstream
  as a legible shutter mode.
* The label metadata. ``_label_metadata`` builds the ``label_metadata`` block from
  ``_LABEL_METADATA``, a table of one entry per attribute of the Cassini data
  dictionary's ISS specific attributes, in the order that dictionary declares them.
  Each entry gives the attribute, the PDS3 label keyword that states it, and, for a
  keyword holding several values, which element. The published key is the attribute
  behind the ``cassini:`` prefix.

  The dictionary's names are used rather than the label's own so that a record says
  the same thing whether it was built from a PDS3 volume or from a PDS4 Cassini source
  bundle, and whether the image file was raw or calibrated. It is the one place in
  the library that lists PDS4 attributes, and it is deliberate.

  The value is what rms-vicar hands back, unconverted: numbers as numbers, text as
  text, and times as the label's day-of-year text. The one transformation is the
  positional split the attribute names require: ``EXPECTED_MAXIMUM``, ``FILTER_NAME``,
  ``INST_CMPRS_PARAM``, ``INST_CMPRS_RATE``, ``OPTICS_TEMPERATURE`` and
  ``VALID_MAXIMUM`` fill several attributes each, and ``_element`` takes the value at the
  attribute's position.

  A label may state a single value where another states a sequence, and ``_element``
  reads that as the first position: the value fills the first attribute and the later
  ones are ``None``. ``OPTICS_TEMPERATURE`` is the keyword this happens on --- an early
  label states one optics reading where a later one states a front reading beside a rear
  reading. That is the same thing the two-value form says when the rear reading is the
  ``-999.0`` a camera with no rear sensor writes, so one value and a null carry the
  meaning the label has. Reading a single value as no sequence at all would publish the
  front reading the label does state as ``None``, which is what the cruise-era test
  holds against.

  ``DESCRIPTION``, ``INSTRUMENT_MODE_ID``, ``OBSERVATION_ID`` and ``SHUTTER_MODE_ID``
  are read here under the dictionary's names for them --- ``cassini:limitations``,
  ``cassini:instrument_mode_id``, ``cassini:observation_id`` and
  ``cassini:shutter_mode_id``. The block no longer states any of them a second time
  under a name of its own, so ``sampling``, ``gain_mode``, ``description`` and
  ``observation_id`` are gone from the observation block; ``filters`` stays beside the
  block, because every host publishes it, and ``shutter_mode`` stays because the
  writer takes it from the ``shutter_mode`` property rather than from here. A program
  that wants the summing mode reads the image size.

  ``cassini:pre-pds_version_number`` is stated by no keyword, which is what its
  ``None`` keyword in the table records. ``_version_number`` reads it from the image's
  file name instead, as the segment after the image number: ``N1521598221_1.IMG`` and
  ``N1521598221_1_CALIB.IMG`` both state ``1``, so a raw file and its calibrated
  counterpart agree. A name the pattern does not fit states no version and the
  attribute is ``None``, like any other unstated value.

  Every attribute is published for every image, so the block is one shape across the
  archive; an attribute whose keyword the label does not state is ``None``. The
  keywords read are the observation keywords of the archive's PDS3 label, which states
  the same set throughout. The VICAR label inside a calibrated image carries all of
  them for a tour-era image; an earlier image's label may carry fewer. Such a label may
  also hold items no attribute reads, some of them differently named equivalents of the
  same quantities, and those are not published. The table is written out rather than
  read from the label, so that a label carrying a keyword the table omits is a
  difference a reader can see rather than one that drops a keyword in silence; tests
  over a real tour-era label and a real cruise-era label hold both behaviors. Six
  keywords a tour-era label states are read by no attribute --- ``DATA_SET_ID``,
  ``INSTRUMENT_HOST_NAME``, ``INSTRUMENT_ID``, ``INSTRUMENT_NAME``, ``MISSION_NAME``
  and ``PRODUCT_ID`` --- and the tour-era test lists exactly those, so a seventh
  would fail it.

**Which label is read.** ``obs.dict`` is the VICAR label inside the
``_CALIB.IMG`` file, which ``host_cassini.CassiniISS.from_file`` reads with
rms-vicar. It is neither the detached ``_CALIB.LBL`` nor the raw volume's
``.LBL``. Its values are the raw product's, as the raw ``.LBL`` states them,
in the VICAR spelling: reals to six or seven significant digits, times ending
in ``Z`` (``PRODUCT_CREATION_TIME`` only on the earliest volumes),
``IMAGE_NUMBER`` as an integer, ``MISSION_PHASE_NAME`` as
``APPROACH_SCIENCE`` where the PDS3 labels write ``APPROACH SCIENCE``, and a
single ``IMAGE_OBSERVATION_TYPE`` as text rather than a one-element set. The
``_CALIB.LBL`` departs further: it carries no ``VALID_MAXIMUM``, its
``DESCRIPTION`` is the calibration's history, and its
``PRODUCT_CREATION_TIME`` is the calibration's, which on some labels is not
a parsable time. Where the VICAR label repeats a keyword, such as ``DAT_TIM``, the
host's label keeps only the first occurrence. Every read here looks a keyword up by
name, and none reads a repeated one.

:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.get_public_metadata`
also refuses a detector that is neither ``NAC`` nor ``WAC``, because the
instrument LID encodes the camera as ``issna`` or ``isswa``, and a malformed one
must never be published.

**Index columns.** ``_INDEX_COLUMNS`` is ``FILE_SPECIFICATION_NAME``.
``_INDEX_CAMERA_COLUMNS`` is ``('INSTRUMENT_ID',)`` and ``_INDEX_CAMERA_MAP``
is ``{'ISSNA': 'NAC', 'ISSWA': 'WAC'}``, whose values are exactly what
``ObsCassiniISS.camera`` returns. The BOTSIM grouping asks for four more
columns per row (``SHUTTER_MODE_ID``, ``IMAGE_NUMBER``, ``OBSERVATION_ID``,
``IMAGE_TIME``), which is why grouping is a separate code path rather than a
filter.

The index table's ``INST_CMPRS_PARAM`` column holds the four compression
parameters in another order than the image label: blocks per group,
algorithm, quantization factor, block type, where its own column description
and the label give algorithm, block type, blocks per group, quantization
factor. A label's ``(1, 1, 41, 0)`` is the index's ``(41, 1, 0, 1)``. The
label metadata is read from the label for this reason among others, so
``cassini:inst_cmprs_param_malgo``, ``cassini:inst_cmprs_param_tb``,
``cassini:inst_cmprs_param_blocks`` and ``cassini:inst_cmprs_param_quant``
carry the label's order; a consumer filling them from an index row instead
would shuffle them.

**Filespec parsing.** ``_get_label_filespec_from_index`` requires the index
value to end ``.IMG`` and rewrites it to ``_CALIB.LBL``; both suffixes are
matched uppercase, deliberately. ``_get_img_name_from_label_filespec`` drops a
leading ``COISS_*`` volume component if present, then requires exactly three
levels, a leading ``DATA``, and a 21-character range directory with ``_`` at
index 10; the image name is the last component up to its first ``.`` or ``_``.

``_img_name_valid`` uppercases, strips ``_CALIB`` and ``.IMG``, and accepts
``[NW]`` followed by ten digits, optionally followed by ``_`` and one or two
digits. ``_extract_img_number`` is the ten digits after the camera letter.

**Monotonicity.** ``_IMG_NUM_MONOTONIC_ACROSS_VOLUMES`` is left at its
``True`` default, so a ``--last-image-num`` scan stops after the first volume
entirely past the range.

**BOTSIM pairing.** ``_is_botsim_pair`` requires both rows to carry
``SHUTTER_MODE_ID == 'BOTSIM'``, opposite camera letters, an equal
``OBSERVATION_ID``, and ``IMAGE_TIME`` values within
``_BOTSIM_MAX_TIME_DELTA_SEC`` (2.0 s) of each other. ``IMAGE_TIME`` is parsed
with ``'%Y-%jT%H:%M:%S.%f'`` and a value that will not parse fails the pairing
rather than raising. ``IMAGE_NUMBER`` is deliberately **not** the pairing key:
it is a spacecraft-clock-derived counter, not a wall clock. The generator holds
one frame back for a single iteration to test it against its successor, and
yields the held frame before moving on, so no frame is ever dropped.

Configuration block
===================

Two sections, each nested per detector (``nac``, ``wac``), which is why the
observation asks for its block by section and detector, as in
``config.camera_block('cassini_iss', 'nac')``.

``cassini_iss`` is a full raw-DN block. ``cassini_iss_calib`` sets
``data_units: calibrated_if``, so it carries what a calibrated camera block carries
(see :ref:`config-camera-blocks`): its ``noise`` block holds only ``marker_value: 0``,
and its ``image_quality_thresholds`` hold only the two ``_if`` thresholds and the three
fractions. It has no ``saturation_dn``, no ``read_noise_dn``, and no
``star_flux_dn_per_s_vmag0``, and the loader refuses any of them.

The calibration pipeline applies an exposure-, filter-, and gain-dependent scaling, so
no single I/F constant identifies the saturated DN ceiling.
``NavOrchestrator._build_saturation_mask`` returns an empty mask for any
``calibrated_if`` input and the classifier sees an infinite threshold, so
``saturation_frac`` is always 0.0 and the ``fully_overexposed`` early-out is
disabled. A camera block has no I/F saturation threshold to set.

Placeholder values, carrying inline ``# PLACEHOLDER`` markers: the raw block's
``read_noise_dn``, the calibrated block's
``blank_max_if`` and ``noisy_threshold_if``, and both blocks'
``mag_offset_table`` entries. The ``_sources`` convention is not used in this
file.

Photometric and PSF calibration
===============================

**Limiting magnitude.** Both cameras follow the Pogson-ratio form, an anchor
plus ``log(texp) / log(2.512)``, so each factor of 2.512 in exposure buys one
magnitude of depth. The two anchors are measured rather than derived, and each
gives the frame it came from:

* NAC: 10.5 at a 1 s exposure, from the clear-filter star field
  ``N1521881358``, which was not useful beyond magnitude 10.7. The returned
  value is ``10.5 + log(texp) / log(2.512)``.
* WAC: 10.7 at a **26 s** exposure, from the clear-filter star field
  ``W1580760393``. Because the anchor is not at unit exposure, the form
  carries the reference explicitly: ``10.7 + log(texp / 26) / log(2.512)``.
  Copying the NAC form here would shift the wide angle limit by more than
  three magnitudes.

Neither compensates for a non-clear filter. A non-positive exposure time falls
back to the anchor, since ``np.log`` would otherwise return ``-inf`` or
``nan``. ``star_min_usable_vmag`` is 0.0 for both cameras: there is no
bright-end cutoff, and saturated stars are handled downstream.

**PSF.** ``star_psf_sigma`` is 0.54 for the NAC and 0.77 for the WAC.
``star_psf_sizes`` is keyed by magnitude upper bound: brighter than 7 gives a
15x15 cutout, brighter than 8 gives 13x13, brighter than 9 gives 11x11, and
everything fainter gives 9x9.

**Lone-star brightness bound.** ``star_max_peak_per_predicted_snr`` is 150 for the NAC
and 40 for the WAC, for raw and calibrated images alike: a lone star's source may peak
at most that many noise sigmas above its local background per unit of the star's
predicted SNR (see :doc:`../dev_guide_techniques_star_unique_match`). Both are measured
values, and nothing yet derives them from the cameras' photometry. Of 3463 NAC stars
found at the published bundles' pointing on the F ring and spokes star frames the
brightest is 126 (median 10), and two lone sources at 166 and 259, 88 and 48 px from the
bundle's answer, are refused. Of 191 WAC stars found at the bundles' pointing on the same
frames the brightest is 20.8 (median 10.6), and the bound is about twice that.

**Lone-star ellipse.** ``one_star_ellipse_fraction`` is 0.6 for the NAC and 0.7 for the
WAC, for raw and calibrated images alike: a lone star is believed only inside the ellipse
whose semi-axes are the offset window's half-widths times it, (30, 84) px on a full NAC
frame and 9.1 px on a full WAC frame ((4.9, 7.0) px on a 2x2 summed one). The WAC's
pointing error reaches the edge of its window, and at 0.6 the WAC refuses 12
lone stars of ``ISS_173RI_SPOKEMOV003_PRIME`` (W1728775362 through W1728801375) lying
between 0.60 and 0.72 of its 13 px window. At 0.7 the 11 within 0.70 are taken, a star
check finds the star at each answer, and no other answer of the replayed F ring and
spokes star frames changes.

**Magnitude offsets.** ``fallback_combo`` is ``'CL1+CL2'``, the two clear
filter slots, and the table carries one entry with a default of 0.0. Both are
placeholders.

**Photometric zero points.** ``star_flux_dn_per_s_vmag0`` is 3.33e5 for the
NAC and 9.29e4 for the WAC, described in the file as interim values sized from
each camera's electron zero point over its standard gain state.

**Summed readout modes.** ``noise.by_binning`` gives each raw camera's values
for the SUM2 and SUM4 modes, keyed by the binning factor: the read noise of one
summed pixel and the zero point, both in DN. Only the faint-star limit of a
simulated frame reads them. The published sources give the summation modes no
read noise or gain of their own, so each entry is the full-resolution camera at
gain state 2. For the NAC that is the published 12 e- over 30 e-/DN, or
0.400 DN; for the WAC it is 12 e- over an estimated 28 e-/DN, or 0.429 DN. The
zero points stay 3.33e5 and 9.29e4. Each full-resolution ``read_noise_dn`` is a
placeholder above the published value, so part of the step from an unbinned
frame's star limit to a summed one's comes from the placeholder: 1.5 of 2.2 mag
on the NAC, and 1.5 of 2.9 mag on the WAC. Both summed read noises are below the
limit's 1 DN noise floor, so the limit reads 1 DN in these modes.

**Recalibrating.** The anchors are the two frames named above; re-measure the
faintest usable star on each and update the anchor and its comment together.
The placeholders are all in the config file rather than in code, so a
calibration pass edits YAML, and the ``# PLACEHOLDER`` marker is what says a
value has not been through that pass.

Frames, attitude, and rotation fitting
======================================

**Camera frames.** ``CASSINI_ISS_NAC`` and ``CASSINI_ISS_WAC``, built as
``f'CASSINI_ISS_{obs.camera}'`` from the oops detector name.

**CK object and clock.** -82000, the spacecraft bus, whose time tags are
encoded against spacecraft clock -82.

**The oops-from-SPICE flip.** ``R = diag(-1, -1, +1)``, a 180 degree rotation
about the boresight, applied by the ``rms-host-cassini`` host. The correction
is built in the oops observation frame and must be conjugated through ``R``
before it is composed onto a ``pxform``-derived matrix; dropping the
conjugation yields a proper rotation of the correct magnitude with both
tangent-plane components negated, which no hermetic test can see. ``R`` is
measured at runtime as ``C_oops . cmatrix_original^T``, checked against this
constant, and checked again for being the same at the exposure start and stop.

**Frame evaluation.** The observation frame is evaluated, not frozen: a
``pxform`` at the exposure midtime reproduces what the host built, which is
what lets the C-kernel writer pair an image with its baseline by ``pxform``
alone.

**Per-spacecraft variation.** None. One spacecraft, two cameras, one CK object
and one clock.

**Rotation fitting.** ``fit_camera_rotation`` is ``false`` for both cameras,
with ``max_rotation_deg: 5.0`` carried but unused. That is the measured
setting rather than a default: the twist is one common value near
+/-0.01 degrees with a frame-to-frame scatter below 0.04 pixels at the field
corner, so there is nothing per-frame for a fit to find. Because it is off,
every image of this instrument is eligible for a corrected C-kernel; turning it
on would make every image ``rotation_unsupported`` and stop the mission's
kernels being produced at all.

C-kernel specifics
==================

**Baseline structure.** The reconstructed holdings are ordinary time-varying
type-3 kernels for object -82000, and a corrected segment composes a
body-fixed ``delta`` onto the baseline's own attitude at each record epoch.

**Angular-velocity census.** All 2645 -82000 segments in the local baselines
carry angular velocity, so the refusal for a baseline that supplies pointing
without a rate is unreachable for this mission against those holdings.

**Kernel-name class rules.** ``_CASSINI_NAME_RULES`` in
:mod:`spindoctor.cli.ck.index` carries four patterns matched in full against
the lowercased basename:

.. code-block:: text

   \d{5}_\d{5}p[a-z]_gapfill_v\d+\.bc      -> GAPFILL
   \d{5}_\d{5}p[a-z](?!_gapfill).*\.bc     -> PREDICTED
   \d{5}_\d{5}r[a-z]\.bc                   -> RECONSTRUCTED
   \d{6}_\d{6}(?:r[a-z])?\.bc              -> RECONSTRUCTED

The class is a release code following the two dates a kernel spans: ``p`` for
planned pointing, ``r`` for reconstructed, plus a letter distinguishing
successive releases of one span. Two reconstructed patterns exist because two
date conventions are in use and the digit count is what tells them apart -- the
tour and the cruise stamp ``YYDOY_YYDOY``, the Jupiter flyby stamps
``YYMMDD_YYMMDD``, and the earliest flyby release omits the code altogether.
The predicted pattern excludes ``_gapfill`` itself rather than relying on being
tested after the gapfill pattern, so the four are mutually exclusive by
construction rather than by order.

**Deviations in segment construction.** None. It takes the standard
time-varying path: records at start, midtime and stop plus a 1 s cadence past
10 s, the baseline's angular-velocity vectors copied bit-identically, and
``avflag = 1``.

**The simultaneous-exposure rule.** ``botsim_loser`` follows from the corrected
object being the bus. Two BOTSIM frames share one bus attitude and one attitude
cannot carry two corrections, so
:func:`~spindoctor.cli.ck.images.botsim_losers` yields the wide angle member --
but only to a partner that actually writes. A wide angle frame whose narrow
angle partner is ineligible, or has no reproducing baseline, keeps its own
correction rather than losing it to nothing.

**Rigid-rotation residual.** An exact rigid rotation is not exactly a uniform
tangent-plane shift, and the difference is measured per camera over a 17x17
grid across the full frame, worst case over eight offset directions, at 50
pixels of total boresight displacement. The narrow angle camera measures
6.01e-9 rad -- 1.00e-3 tangent-plane pixels, 1.24e-3 pixels in pixel space --
and the wide angle camera 5.91e-6 rad, 9.89e-2 tangent-plane pixels and 7.86e-2
in pixel space. The wide angle figure is the case to watch: at a 50 pixel
offset this term alone reaches the round-trip target of 0.1 pixels per axis. It
is linear in the offset, so quoting it without the offset it was measured at
means nothing.

**Reproduction path.** ``cspyce.pxform('J2000', camera_frame, midtime_et)``
against each furnished candidate, accepted at 1e-9 radians. The tie-break
among reproducing candidates prefers reconstructed over gapfill over predicted,
then the lexicographically greatest basename, then the greatest path. The
overlapping reconstructed, gapfill and predicted sets in this mission's
holdings make multiple reproducing candidates ordinary rather than exceptional.

Simulator model
===============

**Instrument keys.** Four, each a top-level instrument entry of the file
``src/spindoctor/sim/instruments/configs/coiss.yaml``, which holds the values
below. ``coiss_nac`` and ``coiss_wac`` resolve to the ``cassini_iss`` block's two
detectors, and ``coiss_calib_nac`` and ``coiss_calib_wac`` to the
``cassini_iss_calib`` block's. Each calibrated entry sets its raw camera as its
``base`` and states only what calibration changes: its units and its detector
terms. Its ``data_units`` is ``calibrated_if``, matching the ``cassini_iss_calib``
block. Everything else comes from the raw camera, so a calibrated scene gets the
same artifact defaults as its raw counterpart.

**Units.** A ``coiss_calib_*`` frame is in I/F with a ``0`` missing-data marker,
as the CALIB products are: across 24 CALIB images from four volumes no pixel is
``NaN``, and exact zeros appear only on the lines the index reports missing. A
lost pixel reads 0, and the dark sky sits on the 4.0e-5 I/F calibrated sky
residual, the median sky of 13 dark NAC CALIB frames, so it reads as dark and
never as missing (see :ref:`Instruments <sim-instruments>`). The generic entry
carries the same value, so the simulator also uses it when a scene pins an
instrument to calibrated I/F and the instrument entry sets no sky
residual.
Those real frames' sky scatters around 0 with a sigma of 1.9e-4 I/F, and under
``instrument_defaults`` about 28% of the pixels of a disc scene a third of the
frame across sit below 0, down to about -0.0006, as on the real frames. The
chain's DN ceiling applies before the calibration inverse, so a
``coiss_calib_nac`` frame tops out at about 2.0 I/F at the reference exposure.

**Calibration residual.** CISSCAL subtracts per-pixel models from integer DN,
and their mismatch leaves a pattern the raw frame doesn't carry (see
:ref:`Instruments <sim-instruments>` for the two keys). On real NAC CALIB frames
the column pattern repeats from frame to frame (correlation 0.93) at 15 e- rms,
and the per-pixel term measures 9.4 e-; the WAC measures 30 e- for each. On the
NAC the two lift the calibrated sky noise at gain state 2 from the raw frame's
0.6 DN to about 0.9 DN-equivalent, as on the real products.

**Downlink table.** The two cameras share the 12-to-8-bit downlink table the
``sqrt_lut`` quantization decodes through (see :ref:`sim-detector-stage`). Each
raw entry sets it, ``table_12_to_8``, as its default, and the calibrated entries
inherit it. It was measured from the 26 TABLE frames of the CALIB cohort. The
calibrated DN-equivalent step between consecutive raw codes is an integer that
grows by one about every ten codes: 3 DN near code 10, 5 near 25, 8 near 55, 13
near 100, and 24 near 200. Code ``c`` decodes to ``round(2 * c + 0.055 *
c**2)`` DN (``codes`` 256, ``linear_dn`` 2, ``quadratic_dn`` 0.055,
``whole_dn`` true), which reproduces those steps and tops out at 4086 DN at
code 255. Near a dark sky the step is 3 to 4 DN, against 1 DN in a 12-bit
frame, so a TABLE frame's calibrated sky is several times noisier than a 12-bit
frame's.

**Dark-sky ceiling.** The realism match compares sky noise only on patches that
lie on dark sky (see :ref:`sim-realism-match`). On a CALIB frame or its matched
frame, a patch lies on dark sky when its mean is at most 2.5e-3 I/F above the
calibrated sky residual. Both calibrated entries set ``dark_sky_ceiling_if`` to
this value. It was chosen from the levels of the darkest quarter of the uniform
patches in the CALIB cohort and in its matched frames. The levels below are all
in I/F above the residual.

On the NAC, real dark sky on 12-bit frames lies from -9.4e-4 to 6.9e-4. The
TABLE frames' skies sit higher, up to 2.18e-3. The highest is a frame whose whole
sky lies between 2.12e-3 and 2.18e-3. Ring and body material lies at 1.26e-2 and
above. The real patch levels leave one empty interval, from 2.34e-3 to 2.78e-3,
and the ceiling sits in it. The matched frames separate cleanly: their sky lies
at most 2.8e-5 above the residual, and their body patches start at 9.7e-3. Any
ceiling from 6e-4 to 2.5e-3 compares the same 26 NAC 12-bit frames. On the WAC,
real dark sky reaches 6.5e-5, and the next patches up lie at 4.88e-3, so the same
ceiling falls inside that gap.

The ceiling tests only a patch's level. On the NAC, sky brightened by scattered
light spans 5.4e-4 to 7.7e-3 I/F. The light comes from glow in the Encke and
Keeler gaps, glare from nearby rings, glow beside a limb, and the halos around
bright moons. Where such sky lies below the ceiling, it counts as dark sky, so a
few frames' brightened sky is compared with the matched frames' dark sky. Where
it lies above the ceiling, it drops out. On the WAC frame of Rhea, the darkest
patches are the halo sky around the moon, at 4.88e-3 and above, so that frame
drops out.

Two TABLE skies below the ceiling are nearly flat, with sky sigmas of 2.3e-5 and
4.4e-6 I/F. Their matched frames' sky sigmas come out about 5 and 3 times as
large. The TABLE frames' sky noise and sky level are an open question, and these
two frames are part of it.

.. The TABLE sky noise and offset are tracked in issue 977.

**Readout dark and cosmic rays.** The raw dark sky rises linearly from the
first line to the last at every exposure, by 55 e- on the NAC and 33 e- on the
WAC. Each entry's ``readout_dark`` shape default catalogs that rise as the
amplitude a native pixel of the last line collects. The cosmic-ray terms, fitted
in closed loop to the NAC's 12-bit dark-sky frames and shared by the WAC, are a
rate of 2.0e-5 events per pixel per second, a 6 s readout dwell, a 1,000 e-
deposit scale, and 0.6 native pixels of diffusion; the simulator report's
artifact-rate table gives the fit's evidence.

**Hot pixels.** The raw cameras' hot-pixel values are fitted on raw 12-bit
full-frame dark frames, taken with the shutter disabled, at exposures of 0, 10, 32,
100, 320, 560, and 1200 s. Each exposure has five to seven frames, and their median
removes the cosmic rays. The NAC has four such series: COISS_2019 (January 2006),
COISS_2041 (January 2008), COISS_2069 (July 2011), and COISS_2114 (June 2017). The
WAC has the first two. The frames are converted to electrons at 30 e-/DN on the NAC
and 28 e-/DN on the WAC.

* A hot pixel's charge grows in proportion to the exposure from 10 to 1200 s, to
  within 3%. A frame of zero exposure still holds the charge the hot pixel collects
  in 0.09 s on the NAC and in 0.10 s on the WAC, which is ``readout_sec``.
* The rates follow a power law with an upper cutoff. Each pixel's rate is read from
  the shortest exposure that holds it well above the noise. In the 2008 series,
  1.0% of the NAC's pixels are hotter than 1 e-/s, and the number hotter than a
  given rate falls as that rate to the power -0.94, up to 1,100 e-/s, above which
  there are none. On the WAC the figures are 0.72%, -0.98, and 1,260 e-/s. The fit
  follows the real counts to within 15% up to 100 or 200 e-/s, and to within 45%
  above that, where a frame holds 30 such pixels or fewer.
* The table gives the number of NAC pixels in a full frame holding at least 75 e-,
  the real 2008 count against the fitted power law:

  ============  ======  ==========
  Exposure (s)  Real    Power law
  ============  ======  ==========
  0.01          7       6
  0.1           17      24
  1             190     182
  3             551     509
  10            1,811   1,579
  30            4,391   4,435
  60            7,709   8,511
  ============  ======  ==========

* The population grows over the mission at about the same slope. On the NAC,
  0.76% of the pixels are hotter than 1 e-/s in 2006, 1.8% in 2011, and 2.3% in
  2017, and the cutoff rises to between 2,000 and 5,000 e-/s. The published 0.15%
  of pixels in 2004 and about 0.28% by 2008 match the dark frames' count of pixels
  hotter than about 4 e-/s.
* The rows read after a hot pixel carry a flat streak, out to 1,000 rows from it.
  Each charge packet that passes through the hot pixel picks up the charge the
  hot pixel collects in 0.0146 s on the NAC and in 0.0142 s on the WAC, at every
  exposure, which is ``streak_sec``. The rows read before a hot pixel carry
  nothing. The camera timing model of the ISS ground calibration report reads the
  CCD at 69 to 79 lines per second in full-frame mode, 0.0126 to 0.0145 s per
  line, which matches. The same model slows the readout to the telemetry rate once
  the camera's 336-line buffer fills, which would lengthen the streak on rows far
  below a hot pixel, but the dark frames show no such step.
* A binned row is given ``binning`` times a native row's streak. No dark frame
  settles that factor: summed-mode darks give 0.027 +- 0.012 s per binned row after
  a hot pixel, against 0.018 s before it. The timing model's summed-mode line rate,
  88 to 96 lines per second, would give about 0.011 s.
* The first three rows after a hot pixel also carry a short trail that grows about
  as the square root of the hot pixel's charge: 2.5% of the charge on the first
  row at 1,000 e-, and 0.4% at 70,000 e-. The simulator leaves it out.

.. _coiss-calibrated-hot-pixels:

**Calibrated hot pixels.** A calibrated product carries its raw camera's hot-pixel
map positions, with a population of its own. CISSCAL's dark subtraction removes the
raw camera's hot pixels: where the nearest dark series has a pixel hotter than
20 e-/s, a calibrated frame holds less than the frame around it, by 0.2 to 1.3
times the raw charge. What a calibrated frame keeps are spikes that recur at one
position and do not grow with the exposure. Frames of 0.08 to 0.2 s show them at
2,000 to 9,000 e-, more than any raw hot pixel collects in that time.

* The calibrated population is fitted to the stationary spikes of the 58-frame NAC
  cohort: 957 spikes, a mean fraction of 1.57e-5 of each frame's pixels. A charge
  of the same size whatever the exposure fits the frames' counts better than a
  charge in proportion to the exposure.
* ``pixel_fraction`` is 3.3e-5, the fraction of pixels holding 1,000 e- or more
  (``readout_e``), and ``rate_e_per_sec`` is 0. The number above a charge falls as
  one over that charge (the fitted index is 0.97) up to the full well. At the real
  frames' spike thresholds the population predicts the cohort's fraction of
  1.57e-5. The realism match measures 2.8e-5 on the matched frames: the NAC's map
  holds 41 such pixels in a full frame where 35 are expected, and the matched
  frames' own spike thresholds flag about 1.5 times more of the population.
* Two frames 14 days apart, N1487595731 (0.18 s) and N1488823805 (1.2 s), share
  192 of those spikes at about the same charge, all in one square of rows and
  columns 512 to 767. They make up 48% of the cohort's stationary spikes. The
  frames' counts also scatter far more than any one law allows, since scene
  content that repeats between frames adds to them.
* The WAC cohort has four frames and one stationary pixel, seen in two of them:
  2,314 e- at 1.8 s and 2,285 e- at 0.025 s. The NAC's law with a fraction of
  7.9e-7 predicts its mean fraction of 4.8e-7, but one pixel leaves that fraction
  uncertain by a factor of several. The WAC's map holds no pixel that rare in a
  full frame, so the realism match measures none.
* The residual pixels' streaks are not measured, so the calibrated population has
  none.

**PSF kernels.** ``coiss_nac`` is
``{sigma_v: 0.70, sigma_u: 0.70, w: 0.12, r0: 3.0, n: 4.0}``, tuned against the
image-library cohort's eleven contributing narrow angle star frames: the cohort
measures encircled-energy radii of 0.91 and 1.79 pixels at 50% and 80%, and
these parameters reproduce 0.90 and 1.72 through the same estimator. The
80%-to-50% ratio of 1.97, against 1.52 for a pure Gaussian, is what forces
substantial mid-range wing energy, so this is an *effective* in-window kernel
rather than an optical one: the fitted ``w = 0.12`` carries measured 1-8 pixel
halo energy that a FWHM-derived core leaves unaccounted for. A first fit at
``w = 0.20`` matched the encircled energy but over-lifted the wide-field halo.

``coiss_wac`` is ``{sigma_v: 1.05, sigma_u: 1.05, w: 0.12, r0: 3.0, n: 4.0}``,
from the cohort's two wide angle star frames and nine usable cutouts. Treat it
as cohort-limited and revisit it when more wide angle star frames land.

Both raw entries state a kernel truncation window (``psf.truncation_px``) of 32
detector pixels rather than the generic 16, because the documented wings run
further out, and the calibrated entries inherit it.

**Distortion residuals.** ``coiss_nac`` is
``{k1: 3.17e-04, k2: -3.51e-04, nonradial_rms_px: 0.0}`` and ``coiss_wac`` is
``{k1: 9.0e-06, k2: 5.48e-05, nonradial_rms_px: 0.0}``, both measured by the
star-field distortion analysis. The corrected field is sub-pixel and close to
radially symmetric, which is why the non-radial wander is zero.

**Artifact-mode availability.** Each raw entry's ``artifact_modes.offered``
lists its modes, and the calibrated entries inherit the list. The simulator
guide's mode tables (:ref:`sim-telemetry-modes` and :ref:`sim-detector-modes`)
give each mode's cameras. Both cameras' detector model is ``ccd``, and neither
entry states an exclusion reason. On a raw camera ``instrument_defaults`` turns on
``hot_pixels``, ``banding``, ``readout_dark``, ``bias_structure``, and ``bloom``. The
calibrated entries drop ``banding`` and ``bias_structure``, which CISSCAL removes, and
set the hot-pixel fraction to the calibrated populations described above: 3.3e-5 on
the narrow angle camera and 7.9e-7 on the wide angle camera.

**Lost lines.** The raw entries offer ``compression_truncated_lines`` and
``downlink_gaps``, and the calibrated entries inherit them. Neither is turned on
by ``instrument_defaults``; a scene sets each one itself. The facts below come
from the binary line prefixes, which give each line's valid samples, of every
losslessly compressed 12-bit frame with lost lines in the current metadata
indexes of COISS_2010, COISS_2023, COISS_2050, and COISS_2068: 321 narrow angle
frames and 82 wide angle frames. No uncompressed 12-bit frame in them lost lines.

*The lossless compressor.* The ISS User's Guide (sections 2.3 and 3.6) says that
the lossless mode must reach 2:1 on each pair of lines and truncates as many
pixels as it needs from the end of the pair to get there. The Calibration Report
(section 5.5.2) adds that the compressor codes the differences between
horizontally adjacent pixels, that it works on 12-bit data packed in 16-bit
words, and that its coder is the one of Rice and Lee, "Some Practical Universal
Noiseless Coding Techniques, Part II" (JPL Publication 83-17). That coder splits
the differences into blocks and codes each block with the cheapest of a family of
codes. The entries' shape defaults are that model: 16-sample blocks, 4 bits to
name a block's code, codes that send 0 to 13 low bits as they are, and a budget
of 16 bits for each sample of one line, so a pair may use one uncompressed line's
bits. The 16-sample block is why every real cut keeps a whole number of 16
samples.

61 narrow angle frames and one wide angle frame lose line ends to the compressor.
On them, ``util/coiss_line_loss/lossless_check.py`` codes every pair whose first
line is whole. The 14,038 pairs whose second line lost its end code to 0.970 to
1.009 of the budget between the 0.1st and 99.9th percentiles, with a median of
0.991, and the budget left at the cut is a median 0.65 of a block. The 5,245 whole
pairs fit in all but 6, which exceed the budget by at most 1.7%.

On the real frames the cut lines are the second lines of their pairs, the cuts
start and stop where bright or detailed content does, a frame of sparse content
cuts few of its pairs, and within one frame the lost share varies by a few percent
of the line. ``util/coiss_line_loss/twins.py`` checks that the simulator draws the
same. It builds a simulated twin of each of ten such frames, from the frame's
smooth brightness and fresh noise of its measured local spread, and cuts it with
the simulator. The twins cut the same sections as their frames, such as the first
37% of the lines on one frame and the last 21% on another. Their median lost share
is within one block of the real one on nine of the ten, and within two on the
tenth. Their spread of lost share within the frame is within 35% of the real one
on all ten, and neighboring cut lines lose the same share at the median. Where
the content is sparse, a twin cuts somewhat more of its pairs than its frame does:
29% against 20%, 48% against 31%, and 13% against 4%.

*Downlink gaps.* Packets lost on the way to the ground take groups of whole lines.
The shape defaults are measured on the 260 narrow angle frames whose line ends the
compressor did not cut, where 61 frames (39 full and 22 summed) lose 373 groups
inside the frame, and on the 81 such wide angle frames, where 19 frames (13 full
and 6 summed) lose 83 groups. ``util/coiss_line_loss/gap_census.py`` measures
them, and ``util/coiss_line_loss/gap_fit.py`` fits the values that the frames do
not give directly and compares simulated frames with the real ones. The tables
give the 0th, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th, and 100th
percentiles. In a size or gap table each measured whole number is spread evenly
over the half line on either side of it, so a draw rounded to a whole line comes
out at each number as often as on the real frames. The simulator draws between
a table's points in the logarithm of the value. On full narrow angle frames
one-line groups are only 2.1% of the groups, so the lowest 5% of the table holds
two-line groups too, and a logarithmic draw starting from half a line would come
out at one line about twice as often as the frames do. That table therefore
starts at 1.38 lines, where the draw gives one-line groups their 2.1%.

- Sizes are in lines of their own frame. Full and summed frames each have their
  own table, because summed groups have other sizes. A narrow angle group on a
  full frame is 2, 3, 5, 10, and 20.7 lines at the 10th, 25th, 50th, 75th, and 90th
  percentiles, and on a summed frame 3, 4, 4, 6, and 11. A sequence of 13 summed
  frames from 2008 holds 145 of the 181 summed groups, and the summed groups still
  differ from the full ones without it. None of the narrow angle frames is summed
  4x4, so 4x4 frames take the 2x2 table. A wide angle group is 2, 3, 5, 11.5,
  and 18.8 lines on a full frame, and 5, 6, 13, 30, and 52 on a summed one. The
  wide angle summed table rests on six frames (five summed 2x2 and one 4x4), too
  few to give each mode its own.
- A narrow angle frame loses a mean 6.1 groups with a variance of 127: one group
  on 49% of the frames, 16 at the 90th percentile, and up to 59. A wide angle
  frame loses a mean 4.4 with a variance of 28.7, one on 47% of the frames. The
  count, the clustering, and the partial lines pool full and summed frames; the
  check of each readout mode below says why.
- Groups cluster. On the narrow angle camera the kept lines between neighboring
  groups are 1, 2, 6, 19, and 55 at the 10th, 25th, 50th, 75th, and 90th
  percentiles. On a frame with more than one group, the first and last span 14%,
  56%, and 84% of its height at the quartiles. Frames that lose many groups
  spread them over the whole frame, and frames that lose a few hold them close
  together or far apart. The gap table inside a cluster is the gaps of 20 lines or
  less, 241 of the 312. On the wide angle camera it is the 42 gaps of 20 lines or
  less of the 64.
- The chance that a group starts a new cluster is fitted on simulated frames, 3,000
  for each value from 0.20 to 0.70 in steps of 0.05. The fit takes the value with
  the smallest loss: the sum of the squared logarithms of the ratios of simulated
  to real gaps at the 25th, 50th, 75th, and 90th percentiles, plus a quarter of the
  sum of the squared differences of the span's quartiles, in tenths of the height.
  On the narrow angle camera the loss is 0.43, 0.33, and 0.40 at 0.35, 0.40, and
  0.45, against 0.84 at 0.30 and at 0.50, so the fit is flat from 0.35 to 0.45 and
  takes 0.40. On the wide angle camera it is 1.48 at 0.60 and 1.60, 1.51, and 1.57
  at 0.55, 0.65, and 0.70, so that fit takes 0.60 and is flat from 0.55 to 0.70.
- The count law is fitted on simulated frames, because a frame too small for its
  groups loses some of them: a negative binomial of mean 5.11 and dispersion 4.47
  beyond the first group gives the measured mean and share of frames with one
  group on the narrow angle camera, and one of mean 3.37 and dispersion 3.36 on the
  wide angle camera.
- 36.6% of the narrow angle lines just above one group lose their end, and 12.7%
  of those just below. A single kept line between two groups is the line below
  one and the line above the other, and 23 of the 51 such lines lost their end,
  the 45% that two independent draws give. So
  38% of the groups show a head line and 17% a tail line. On the wide angle camera
  the shares are 27.8% and 11.4%, and 28% and 12% with the shared lines. Every such
  line keeps a whole number of 16 samples. A narrow angle line above a group keeps
  a median 34% of the line, and one below 38%. A wide angle line above keeps a
  median 45%, and one below 20%, from its own 10 lines; that is lower than the
  narrow angle share at a rank-sum p of 0.02.

Over 20,000 simulated frames of each camera, at the heights and readout modes of
the real frames in turn, and measured the same way, every statistic lies inside
the real frames' 90% interval. The interval comes from resampling: 2,000 sets of
the real frames drawn with replacement, each readout mode keeping its number of
frames, and the interval runs from the 5th to the 95th percentile of the statistic
over the sets. Each real value below is followed by its interval.

.. list-table::
   :header-rows: 1

   * - Statistic
     - Narrow angle, real
     - Narrow angle, simulated
     - Wide angle, real
     - Wide angle, simulated
   * - Groups per frame, mean
     - 6.1 (3.9 to 8.6)
     - 6.1
     - 4.4 (2.6 to 6.4)
     - 4.4
   * - Groups per frame, variance
     - 127 (40 to 225)
     - 112
     - 28.7 (5.0 to 50.2)
     - 38.5
   * - Frames with one group
     - 49% (39 to 59%)
     - 49%
     - 47% (32 to 63%)
     - 47%
   * - Groups per frame, 90th percentile
     - 16 (7 to 29)
     - 17
     - 10.6 (5.2 to 20)
     - 11
   * - Full-frame sizes, 25th percentile
     - 3 (2 to 3)
     - 3
     - 3 (2 to 4)
     - 3
   * - Full-frame sizes, 50th percentile
     - 5 (4 to 7)
     - 5
     - 5 (4 to 7)
     - 5
   * - Full-frame sizes, 75th percentile
     - 10 (7 to 15)
     - 10
     - 11.5 (7 to 18)
     - 12
   * - Full-frame sizes, 90th percentile
     - 20.7 (12 to 42)
     - 21
     - 18.8 (13 to 158)
     - 19
   * - Summed-frame sizes, 25th percentile
     - 4 (4 to 4)
     - 4
     - 6 (5 to 10)
     - 6
   * - Summed-frame sizes, 50th percentile
     - 4 (4 to 5)
     - 4
     - 13 (8 to 27)
     - 13
   * - Summed-frame sizes, 75th percentile
     - 6 (6 to 8)
     - 6
     - 30 (16 to 51)
     - 27
   * - Summed-frame sizes, 90th percentile
     - 11 (8 to 14)
     - 11
     - 52 (30 to 59)
     - 48
   * - One-line groups
     - 3.8% (2.2 to 5.8%)
     - 3.4%
     - 3.6% (1.1 to 7.4%)
     - 5.4%
   * - Gaps, 25th percentile
     - 2 (2 to 4)
     - 2
     - 6 (5 to 7)
     - 5
   * - Gaps, 50th percentile
     - 6 (4 to 11)
     - 6
     - 11.5 (9 to 16)
     - 12
   * - Gaps, 75th percentile
     - 19 (11 to 34)
     - 18
     - 30 (16 to 162)
     - 49
   * - Gaps, 90th percentile
     - 55 (31 to 113)
     - 87
     - 181 (41 to 244)
     - 155
   * - Span, 25th percentile
     - 0.14 (0.08 to 0.45)
     - 0.17
     - 0.24 (0.05 to 0.82)
     - 0.30
   * - Span, 50th percentile
     - 0.56 (0.41 to 0.71)
     - 0.57
     - 0.80 (0.16 to 0.87)
     - 0.63
   * - Span, 75th percentile
     - 0.84 (0.70 to 0.93)
     - 0.84
     - 0.87 (0.76 to 0.91)
     - 0.86
   * - Groups with a head line
     - 37.8% (28.4 to 45.0%)
     - 37.6%
     - 27.7% (23.1 to 33.3%)
     - 28.0%
   * - Groups with a tail line
     - 17.2% (11.8 to 21.2%)
     - 17.1%
     - 12.0% (7.5 to 15.4%)
     - 12.7%

The largest differences are the narrow angle 90th percentile of the gaps, 87
lines against 55, and the wide angle median span, 0.63 against 0.80, and 75th
percentile of the gaps, 49 lines against 30. The wide angle frames with more than
one group are only 10, so their span and gaps are loosely known.

*Each readout mode.* The check is repeated on the full frames alone and on the
summed frames alone, with the pooled count, clustering, and partial-line laws. A
sequence here is a run of frames less than a day apart.

- On the 39 full narrow angle frames, in 15 sequences, every statistic lies inside
  its interval. The simulated frames lose a mean 6.2 groups against 4.9 (2.0 to
  8.3), one group on 49% of the frames against 59% (46 to 72%), and their gaps reach
  112 lines at the 90th percentile against 30 (12 to 316).
- On the 22 summed narrow angle frames five statistics miss their intervals: the
  gaps are 2, 5, and 14 lines at the quartiles against 4, 10, and 27 (3 to 6, 8 to
  18, and 19 to 54), 18% of the gaps are one line against 5.7% (3.3 to 8.3%), and
  38% of the groups show a head line against 31% (26 to 37%). One sequence of 13 frames from 2008
  (N1606722214 to N1606787565) holds 145 of the 181 summed groups and 132 of the 159
  gaps, so these statistics are mostly that sequence's. The other nine summed
  frames hold their groups closer together than the pooled law does, at 2, 3, and 7
  lines between groups at the quartiles. The frames therefore cannot give summed
  frames a law of their own, and they take the pooled one.
- On the 13 full wide angle frames, in nine sequences, the gaps' 25th percentile
  misses: 6 lines against 8.75 (7 to 101.5). Three frames of one 2011 sequence
  (W1683625725 to W1683626155) hold 29 of the 43 groups and 26 of the 30 gaps.
- On the six summed wide angle frames, in three sequences, two statistics miss: the
  gaps' 25th percentile, 3 lines against 5 (5 to 5.5), and the one-line gaps, 11%
  against 5.9% (0 to 10.3%). Two frames of one 2011 sequence (W1683220339 and
  W1683220356) hold 23 of the 40 groups and 21 of the 34 gaps. Both wide angle modes
  therefore take the pooled law too.

**Realism-match status.** The narrow angle PSF is tuned against eleven frames.
The wide angle PSF rests on two frames.

**Passbands.** Each camera's entry sets a passband table (``coiss_nac`` and
``coiss_wac``) built from the system transmission files of the CISSCAL
calibration volume, ``COISS_0011``, with its solar spectrum. The solar-weighted
mean wavelength of each combination matches the effective wavelength the same
volume lists, 651.1 nm for the narrow angle clear combination and 633.8 nm for
the wide angle's.

Titan's haze is refused in ``MT1``, ``MT2``, ``MT3``, ``CB1``, ``CB2``, and
``CB3`` by name, in ``CL1+IR3``, ``IR2+CL2``, ``IR2+IR3``, ``RED+GRN`` (narrow
angle) and ``IR2+CL2``, ``IR3+CL2`` (wide angle) because they put 0.11 to 0.43 of
their light in the methane band cores, and in the ultraviolet combinations and
``CL1+VIO`` because 0.87 or more of their light lies below 430 nm. The clear,
``RED``, ``GRN``, and ``IR1`` combinations put 0.05 to 0.07 of their light in the
band cores.

**Titan haze.** The simulated Titan haze (see :ref:`sim-atmosphere`) was compared
with seven calibrated Titan frames from the cohort in ``util/titan_cohort``, each
rendered in its own filter. Their geometry comes from the navigator's Titan model
with each frame's committed offset, and each comparison render is Titan at that
frame's own scale and phase, before the detector. All values are I/F. "Half max"
is the altitude above the solid radius where the sunward profile falls to half its
peak, and the peak altitude is negative where the sunward peak lies on the disc.
The ``CL1+VIO`` frame has no simulated twin, since the haze is not modeled there.

.. list-table::
   :header-rows: 1

   * - Frame
     - Filters
     - Phase (deg)
     - Disc center, real / simulated
     - Sunward peak, real / simulated
     - Peak altitude (km), real / simulated
     - Half max (km), real / simulated
     - Anti-sunward peak, real / simulated
   * - W1552216646
     - CL1, VIO
     - 60.6
     - 0.039 / --
     - 0.094 / --
     - -62 / --
     - 347 / --
     - 0.008 / --
   * - W1552216540
     - CL1, RED
     - 60.6
     - 0.134 / 0.084
     - 0.266 / 0.168
     - -321 / 3
     - 260 / 209
     - 0.020 / 0.019
   * - W1822132529
     - CL1, BL1
     - 63.1
     - 0.062 / 0.052
     - 0.126 / 0.115
     - -335 / 164
     - 259 / 265
     - 0.013 / 0.013
   * - N1634463227
     - CL1, CL2
     - 81.1
     - 0.052 / 0.029
     - 0.224 / 0.158
     - -250 / 126
     - 222 / 213
     - 0.002 / 0.001
   * - N1702239331
     - CL1, CL2
     - 107.1
     - 0.0024 / 0.0014
     - 0.287 / 0.152
     - -43 / 132
     - 232 / 220
     - 0.001 / 0.000
   * - W1721822901
     - CL1, CL2
     - 149.2
     - 0.005 / 0.000
     - 0.847 / 0.282
     - 217 / 217
     - 326 / 290
     - 0.028 / 0.000
   * - W1561790952
     - CL1, CL2
     - 166.5
     - 0.102 / 0.000
     - 2.693 / 1.765
     - 310 / 249
     - 399 / 303
     - 0.321 / 0.298

The 0.102 at the center of W1561790952 is scattered light from the bright ring
around it. The real haze's limb stands higher in the violet: at 60.6 deg the
``VIO`` frame's sunward edge falls to half its peak at 347 km, against 260 km in
``RED``. The simulated haze's limb stands 56 km higher in ``BL1`` than in ``RED``,
while the real ``BL1`` frame's half max, at 259 km, matches the ``RED`` frame's.
From 60 to 107 deg the simulated disc runs 16 to 44% darker than the real one,
least in ``BL1``, and in ``RED`` its sunward edge falls to half its peak 51 km
lower. At 149 and 167 deg the ring of light is 3.0 and 1.5 times fainter than on
the real frames and its peak sits up to 61 km lower, and at 149 deg no light
reaches the anti-sunward side, where the real frame shows 0.028. The likely cause
of the faint ring is light scattered forward more than once along the long limb
path, which the renderer's isotropic multiple-scattering term cannot carry
forward. The likely main cause of the dark disc is the multiple-scattering term
itself: for a semi-infinite layer of the haze at 650 nm lit from straight above,
the renderer's reflectance runs about 22% below a Monte Carlo simulation's (0.162
against 0.207 seen at an emission cosine of 0.475, and 0.149 against 0.186 at
0.975), and at these phases most of the disc's light has scattered more than
once.

The Titan scenes are exposed from these frames. The haze's peak stands 1933,
2150, and 2696 DN above the sky on the three frames at 60.6 to 63.1 deg
(W1552216646, W1552216540, and W1822132529), a median of 2150 DN, and 2787 DN
above it on W1721822901 at 149.2 deg, the cohort frame nearest 155 deg. The
low-phase scenes ``titan_haze`` (60 deg) and ``titan_haze_limb`` (40 deg) take
7.3 s and 7.5 s, which put their haze's peak 2117 and 2114 DN above the sky, and
``titan_crescent_horns`` and its noiseless twin (155 deg) take 3.6 s, which puts
the ring of light 2782 DN above it. The noisy scenes carry the narrow angle
camera's read noise, 12 e-, which is 0.4 DN at the 30 e-/DN of gain state 2.

Image library and test coverage
===============================

**Cohort.** Sixty-two sidecars, 58 narrow angle and 4 wide angle, spread across
the scene classes. Every one is a calibrated product.

**Integration tests.** The per-image regression suite
(``tests/integration/test_autonomous_nav.py``) and the structural-invariants
test (``test_image_library.py``) run on those sidecars.
``tests/integration/test_cmatrix_frames.py`` measures the oops-from-SPICE flip
on a real frame of this instrument and checks it against ``diag(-1, -1, +1)``.
``tests/integration/test_ck_round_trip.py`` runs its round trip on two frames
of this instrument, a star-navigated narrow angle frame and a star-navigated
wide angle one, plus a wide angle body frame whose ensemble is carried by the
correlation and distance-transform techniques instead. The wide angle frames
are there on purpose: on that camera the difference between an exact rigid
rotation and a uniform pixel shift reaches 9.89e-2 pixels for a 50 pixel total
offset, linear in the offset. The hermetic writer tests build their own minimal
kernels and do not depend on this instrument's holdings, but the
angular-velocity measurement quoted above was made against the real
reconstructed kernel ``04002_04009ra.bc``.

**Simulator guard scenes.** The ``full_frame_cameras`` scene class (see
:ref:`sim-full-frame-cameras`) holds five scenes of this instrument, each at the
full 1024 px frame. The offsets lie within the range of real pointing errors on
each camera, most of them near its upper end. The narrow angle scenes plant the
measured twist of +0.0115 degrees, and the wide angle scenes -0.0115 degrees.

- ``coiss_nac_star_field``: a raw narrow angle star field, 18 stars of V 8.7 to
  11.4 at 2 s, offset 39 px. The star-field technique matches the catalog
  against the 30 brightest sources it detects. The frame's cosmic rays crowd
  the stars out of that list, so the technique finds one match where it needs
  six and returns spurious. With the cosmic rays turned off it finds the offset
  to 0.02 px, hot pixels and all. The single-star and refinement techniques
  carry the answer, 0.01 to 0.23 px off over 16 seeds.

  The scene renders its stars through the clear filter, though the real frame
  it follows was taken through CL1+IR3. Through CL1+IR3 the simulator dims each
  star by the filter's throughput for sunlight, 22 times, and the frame fails at
  2 s, where the real frame navigates. The real frame's stars are about 50 times
  brighter than the simulator draws them through that filter, and about twice
  as bright as it draws them through the clear filter.
- ``coiss_calib_nac_disc``: a calibrated narrow angle frame of a moon 280 px
  across, offset 57 px. The answer lands 0.09 to 0.10 px off.
- ``coiss_wac_disc``: a raw wide angle frame of a moon 128 px across, offset
  13 px. The disc correlation finds the offset to 0.09 px but marks it
  spurious, because it lies within 2 px of the edge of the 13 px search window.
  The limb fit carries the frame at low confidence, 0.13 to 0.17 px off, and
  the scene's expected block holds the low tier.
- ``coiss_calib_wac_star_field``: a calibrated wide angle star field, 95 stars
  of V 7.4 to 10.4 at 26 s, offset 12 px. The answer lands 0.01 to 0.03 px off.
- ``coiss_calib_wac_star_field_large_offset``: the same field with the offset at
  the largest real per-axis extent, 12.4 px. Both star techniques find the
  offset to within 0.06 px, but an answer within 1 px of the 13 px search limit
  has its confidence set to zero, so the frame is refused. The scene's expected
  block holds the failure.

**Simulator ring guard scenes.** The ``dense_rings`` scene class (see
:ref:`sim-dense-rings`) holds five full-size narrow angle frames of the main
rings. Each has a pointing error of 27 to 34 px and the camera's measured twist
of 0.0115 degrees, with roll fitting off as the camera ships.

- ``keeler_gap_unlit_ansa`` draws the Keeler gap at 0.338 km/px on the unlit face,
  near the ansa, beside 453 one-kilometer bins of a radial profile measured on a
  library frame. The edge fit lands about 3.5 px off along the nearly straight
  edges and reports a sigma near 0.1 px. With one uniform sheet in place of the
  profile it lands about 1.2 px off, so the profile carries most of the error.
- ``a_ring_edge_f_ring`` draws the outer A ring at 10 km/px with the F ring 345 px
  outside the A ring's edge and the Encke gap's central ringlet inside the gap.
  Neither pulls the fit.
- ``a_ring_low_opening`` draws the same rings at a ring opening of 2.5 degrees.
  The edges become nested hairpins about the ansa, and along their arms the
  rings' radial direction is squeezed to about a twentieth of its length. The
  Keeler gap, 3.7 px wide at the ansa, is under a pixel wide over most of its arc,
  and fitting both of its edges puts the answer 0.84 px off at a sigma under
  0.06 px.
- ``maxwell_hidden_eccentricity`` draws the Maxwell ringlet with its measured
  eccentricity while the navigator's orbit for it is circular. The fit follows the
  drawn ringlet, about 5 px inward of the circular orbit, and ends 6.8 px off at a
  sigma under 0.1 px. The annulus correlation, pulled by the same ringlet, lands
  11.8 px off without flagging it. The two answers disagree, and the frame is
  returned as conflicted.
- In ``b_ring_saturated_plateau`` a 12 s exposure saturates the B ring into a flat
  plateau beside the Huygens gap, with the Herschel and Russell gaps in the
  Cassini Division beyond it. The saturated edge pulls the answer 0.06 px farther
  out across the edges than the same frame unsaturated, which is five times the
  reported sigma.

Turning a frame about the boresight slides a curved ring edge along itself by
about the twist angle times the edge's radius of curvature on the sky. For the A
ring at 10 km/px that is about 0.26 px, and the navigator's reported sigma leaves
it out: with the twist removed, ``a_ring_edge_f_ring`` lands 0.039 px off. The
twist is not what puts the other scenes above three sigma. Without it they still
sit at 5.5 sigma (the saturated B ring), 25 sigma (the Keeler gap), 17 sigma (the
low opening), and 144 sigma (the hidden eccentricity). Where the along-edge answer
is weakly determined, as at the Keeler gap, the twist moves it by more than a
slide would, in a direction that depends on the twist's sign. Rendering and
navigating one of these frames takes 6 to 40 seconds.

**Mixed-defect guard scenes.** Two narrow angle scenes in the ``mixed_defects``
scene class (see :ref:`sim-mixed-defects`) carry the raw chain with the defects
real frames of this camera carry together:

- ``coiss_nac_lossy_moon`` is a moon frame passed through the 12-to-8-bit downlink
  table and lossy compression. In the metadata indexes of COISS_2010, COISS_2023,
  COISS_2050, and COISS_2068, every lossy frame of either camera is an 8-bit
  frame, and lossy labels give no count of lost lines. The simulator's codec steps
  every frequency alike, so its 128 DN step is illustrative: it matches the block
  edges of real lossy frames and zeroes more of the fine detail than they lose.
  The scene seeds only the noise, so it has 16 distinct draws, and the fused
  answer lands 0.034 to 0.050 px off over them. The table and the compression are
  realistic but leave the answer where it is: with either one removed, the fused
  answer lands 0.041 to 0.047 px off. So the scene shows that the navigator is
  unaffected by them.
- ``coiss_nac_stars_moon_lost_lines`` is a frame of stars and a gibbous moon that
  lost its last 41 lines and several groups of whole lines down the frame, and whose
  scene drifts 2 px during the exposure. It follows two real frames, one that lost
  its last 41 lines and one that lost 43 lines in six groups of 4 to 11 lines. The
  scene draws groups with ``downlink_gaps`` at this camera's measured sizes,
  clustering, and partial lines (see **Lost lines** above), with a mean of six
  groups. Neither real frame lost line ends to the compressor, and with the
  compressor on the scene's draws lose none either, so the scene leaves it off. The
  committed seed loses seven groups and 50 lines in two clusters, as the real frame
  lost four of its six groups in one cluster, and four of the groups cross the
  disc, as the real groups crossed the planet. The truncated lines hold dark sky,
  as they do on the real truncated frame. The fused answer lands 0.019 to 0.17 px
  off over 24 draws, at high confidence on 6 of them and medium on 18. On the
  committed seed the disc correlation, which treats the lost lines as zeros, lands
  0.10 px off where it lands 0.019 px off without the groups, and the stars hold
  the answer at 0.098 px either way. On four draws the answer lands 0.069 to 0.098
  px off with all the defects, and 0.063 to 0.098 px off with any one of them
  removed, so the scene shows that the fused answer is unaffected by the
  truncation and the lost groups. The star-field technique returns spurious on every draw,
  because hot pixels and cosmic rays together crowd the stars out of the sources it
  matches, and the blob technique lands 3 to 13 px off the gibbous disc without
  flagging it.

**Unit tests.** ``tests/spindoctor/inst/test_inst_cassini_iss.py`` covers the
config-block selection, the shutter-mode property and the metadata surface.

**Body disc refinement against star solutions.** The body disc technique refines its
final pass on a high-pass surface (see :doc:`../dev_guide_techniques_body_disc`). That
refinement was measured on 69 narrow-angle and 6 wide-angle frames with strong star
solutions, each compared with a half-pixel window around the integer peak.

- **Narrow-angle camera.** The median per-axis error of the disc answer falls from
  0.29 / 0.30 px to 0.23 / 0.23 px. The median distance from the star solution moves from
  0.467 to 0.470 px, the mean falls from 0.747 to 0.634 px, and the largest rises from
  5.21 to 6.20 px. 42 frames improve by more than 0.05 px, and 15 get worse by as much.
- **Wide-angle camera.** The high-pass refinement is worse overall: the median distance
  from the star solution rises from 1.79 to 2.77 px. Two of the six frames are saturated,
  with spill around the body, which is the structure the refinement handles worst:
  W1568165771_2 goes from 1.70 to 2.80 px, and W1568166187_2 from 1.79 to 2.79 px.

PDS4 hooks
==========

This dataset is the reference implementation, and implements every hook.

* ``pds4_bundle_template_dir`` reads ``config.pds4.<key>.template_dir`` and resolves a
  relative name against ``src/spindoctor/cli/pds4/templates/``.
* ``pds4_bundle_name``, ``pds4_bundle_version``, ``pds4_information_model_version``,
  and ``pds4_schemas`` read ``config.pds4.<key>.bundle_name``, ``bundle_version``,
  ``information_model_version``, and ``schemas``.  The shipped ``coiss_saturn`` entry
  gives the schemas of the ``pds``, ``disp``, ``geom``, and ``cassini`` dictionaries.
  The last is the Cassini mission dictionary, whose namespace the data label declares
  for its mission area.
* ``pds4_required_templates`` names the templates each pass renders and the files it
  copies, and ``pds4_user_guide_file_name`` names the bundle's user guide in the
  template directory; each subclass names its own bundle's guide.
* ``pds4_bundle_path_for_image`` maps ``N1234567890`` to
  ``1234xxxxxx/123456xxxx/``, raising on a name shorter than 11 characters
  rather than returning an empty string a caller would concatenate into a
  malformed path.
* ``pds4_path_stub`` appends the LID part, which moves the camera letter to a
  lowercase suffix: ``N1454725799`` becomes ``1454725799n``.
  ``pds4_lid_part_to_image_name`` inverts exactly that transform.
* The four LID and LIDVID builders share the same LID part and differ only in
  the ``browse`` / ``data`` collection; a LIDVID is its LID at the bundle's version.
* ``pds4_template_variables`` emits the camera width, the three exposure
  times from the navigation metadata, the product LIDs and LIDVIDs, and roughly
  sixty ``cassini:`` namespace variables read straight from the PDS3 index row.

The five hooks that read configuration take the entry under the dataset's
``PDS4_CONFIG_KEY``, written ``<key>`` above. The Cruise and Saturn subclasses set it
to ``coiss_cruise`` and ``coiss_saturn``. ``DataSetPDS3CassiniISS`` itself leaves it
as None, so those five hooks and ``pds4_user_guide_file_name`` raise
``NotImplementedError`` there, and ``sd_create_bundle``
(:doc:`/user_guide/user_guide_pds4_bundle`) refuses the undivided ``coiss`` dataset
before it runs a pass. A bundle belongs to one of the two halves, not
to both.

The package ships only the ``coiss_saturn`` entry and its ``cassini_iss_saturn_1.0``
template directory. To build a cruise bundle, supply a ``pds4.coiss_cruise`` entry and a
template directory.

Backplanes, mosaics, and statistics
===================================

**Backplanes.** Nothing instrument-specific. The stage reads the recorded
C-matrix through the shared pointing ladder, and since rotation fitting is off
for this instrument, its records carry a usable C-matrix and take the matrix
path rather than the offset fallback.

**Mosaics.** Nothing instrument-specific.

**Statistics.**
:func:`~spindoctor.cli.stats.report_sections.add_botsim_section` pairs
``N``/``W`` rows sharing a ten-digit number and reports the disagreement
between the two frames of one simultaneous exposure, as a consistency check.
:func:`~spindoctor.cli.stats.report_sections.resolve_offset_limit` picks the
config section from ``_CALIB`` in the image name and the detector from the
name's leading ``N`` or ``W``, then indexes the size-keyed table with the
recorded image height. The report's image-name rule is
``stem.split('_', 1)[0]``, so ``N1454725799_1_CALIB.IMG`` reduces to
``N1454725799``.

Open items
==========

* Every value marked ``# PLACEHOLDER`` in ``config_400_inst_coiss.yaml``: the
  raw block's ``read_noise_dn``, the calibrated
  block's ``blank_max_if`` and ``noisy_threshold_if``, and the magnitude-offset
  tables on all four detector entries.
* ``from_file`` carries a TODO noting that the ``np.min`` / ``np.max`` debug
  line is slow even when debug logging is off, because the arguments are
  evaluated before the level is checked.
* ``star_min_usable_vmag`` has a wide angle branch that returns the same value
  as the narrow angle fall-through; it is a placeholder for a bright-end cutoff
  that has not been needed yet.
* The wide angle PSF kernel and the wide angle limiting-magnitude anchor both
  rest on very small cohorts.
