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
   * - Host
     - ``host_cassini.CassiniISS``, from the ``rms-host-cassini`` package

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.from_file` calls
``host_cassini.CassiniISS.from_file(host_input(path, label=label),
fast_distortion=..., return_all_planets=...)``.
:func:`~spindoctor.obs.obs_snapshot_inst.host_input` chooses the image's
``_CALIB.LBL``, so that the host reads that PDS3 label and the VICAR file
together, or the image file alone if the image has no label. The two other keyword arguments are read from ``**kwargs`` and default
to ``True``.
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

**Host values read.**

* The ``spacecraft_clock_start_count`` and ``spacecraft_clock_stop_count``
  subfields, read in
  :meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.get_public_metadata`.
  The host spells them as the PDS3 label does, and computes both counts of an
  early cruise image, whose labels give a wrong start count, from the image
  number and the exposure, whichever file it reads.
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
  their exact mean. A count the host does not give is published as ``None``,
  and so is the mean.
* The ``filter1`` and ``filter2`` subfields, for ``filters``. The host gives
  them only when the labels give both filters, and ``filters`` is otherwise
  empty.
* ``SHUTTER_MODE_ID`` in ``dict``, read by the ``shutter_mode`` property. A
  missing key yields ``None``; a non-string value raises, because ``str()`` would
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

  Each value is written as SpinDoctor's conversion step, ``_document_value``, writes
  it: numbers as numbers, text as text, a time in the label parser's PDS3 form, and a
  set as a list. "Which label is read" below describes that step. The one other
  transformation is the positional split the attribute names require:
  ``EXPECTED_MAXIMUM``, ``FILTER_NAME``, ``INST_CMPRS_PARAM``, ``INST_CMPRS_RATE``,
  ``OPTICS_TEMPERATURE``, and ``VALID_MAXIMUM`` fill several attributes each, and
  ``_element`` takes the value at the attribute's position.

  A label may state a single value where another states a sequence, and ``_element``
  reads that as the first position: the value fills the first attribute and the later
  ones are ``None``. ``OPTICS_TEMPERATURE`` is the keyword this happens on --- an early
  label states one optics reading where a later one states a front reading beside a rear
  reading. That is the same thing the two-value form says when the rear reading is
  missing, as for a camera with no rear sensor, whose ``-999.0`` the host gives as
  ``None`` in that position. Reading a single value as no sequence at all would publish the
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
  archive; an attribute whose keyword the host does not give is ``None``. The
  keywords read are the observation keywords of the archive's PDS3 label, which states
  the same set throughout. The VICAR label inside a calibrated image carries all of
  them for a tour-era image; an earlier image's label may carry fewer. Such a label may
  also hold items no attribute reads, some of them differently named equivalents of the
  same quantities, and those are not published. The table is written out rather than
  read from the label, so that a label carrying a keyword the table omits is a
  difference a reader can see rather than one that drops a keyword in silence; tests
  over a real tour-era label and a real cruise-era label hold both behaviors. Six
  keywords a tour-era VICAR label states are read by no attribute --- ``DATA_SET_ID``,
  ``INSTRUMENT_HOST_NAME``, ``INSTRUMENT_ID``, ``INSTRUMENT_NAME``, ``MISSION_NAME``
  and ``PRODUCT_ID`` --- and the tour-era test lists exactly those, so a seventh
  would fail it. That test reads the host's ``header_dict``, the VICAR label as the
  file writes it.

**Which label is read.** ``obs.dict`` is the host's merge of the ``_CALIB.LBL``
and the VICAR label inside the ``_CALIB.IMG`` file. Where the two give an item and
agree, it is that value; where they disagree, it is the ``_CALIB.LBL``'s, except
``MISSION_PHASE_NAME`` and ``METHOD_DESC``, which the host takes from the VICAR
label. The values have the PDS3 label's types: reals as the label prints them,
to six decimal places, ``IMAGE_NUMBER`` as text, a time as a ``datetime``, and
``IMAGE_OBSERVATION_TYPE`` as a set. A metadata document cannot hold the last
two, so ``_document_value`` writes them in the form the label parser keeps
beside each value. A time, whether a ``datetime`` or a ``date``, is written as
the parser's PDS3 form of it, which the parser keeps under the keyword with
``_fmt`` appended. A set is written as a list in the label's order, which the
parser keeps under the keyword with ``_list`` appended. A time or a set the host
took from the VICAR label alone has neither. Then a time is written as the VICAR
label writes it but without the trailing ``Z``, a date is written as
``YYYY-DDD``, and a set is written as a sorted list. A value that a label gives
as a placeholder (``N/A``, ``UNK``, or ``-999`` in a temperature, a data rate,
or a compression ratio) is left out, so its attribute is ``None``. The
rms-host-cassini user guide lists the merge rules.

The ``_CALIB.LBL`` describes the calibrated product: its ``DESCRIPTION`` is the
calibration's history and its ``PRODUCT_CREATION_TIME`` is the calibration's,
such as ``2019-06-09T12:39:25``. The VICAR label keeps the raw product's values,
which are what ``cassini:limitations`` and ``cassini:pds3_product_creation_time``
mean, so ``_RAW_PRODUCT_KEYWORDS`` reads those two keywords from the host's
``header_dict`` instead, as the VICAR label writes them, and a placeholder there
is ``None`` as well. Where the VICAR label repeats a keyword, such as
``DAT_TIM``, ``header_dict`` holds its first value. Every read here looks a
keyword up by name, and none reads a repeated one.

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
loader indexes the block by detector.

``cassini_iss`` is a full raw-DN block. ``cassini_iss_calib`` departs from the
common schema by **omission**: its ``noise`` block carries only
``marker_value: 0``, and its ``image_quality_thresholds`` carry only the two
``_if`` thresholds and the two fractions. It has no ``saturation_dn``, no
``read_noise_dn``, no ``star_flux_dn_per_s_vmag0``, and -- the consequential one -- no
``saturation_threshold_if``.

That last omission is deliberate and is documented in the file. The calibration
pipeline applies an exposure-, filter- and gain-dependent scaling, so no single
I/F constant identifies the saturated DN ceiling.
``NavOrchestrator._build_saturation_mask`` returns an empty mask for any
``calibrated_if`` input and the classifier sees an infinite threshold, so
``saturation_frac`` is always 0.0 and the ``fully_overexposed`` early-out is
disabled. Adding a threshold to this block would not restore the mask; the
mask is gated on the units, not on the presence of the key.

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
lower the hot-pixel fraction to the 1.6e-5 the cohort measures.

**Realism-match status.** The narrow angle PSF is tuned against eleven frames.
The wide angle PSF rests on two frames.

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
  scene drifts 2 px during the exposure. In those four volumes, 321 of the 5,489
  12-bit narrow angle frames lost lines, all of them losslessly compressed, in three
  ways. 195 lost lines at the bottom, most of them when the readout window closed
  before the camera finished. 62 lost groups of whole lines inside the frame, when
  packets were lost on the way to the ground. 61 lost the ends of alternating lines,
  where a pair of lines would not compress to half its size. The scene follows one
  real frame of each of the first two kinds: one lost its last 41 lines, and the
  other lost 43 lines in six groups of 4 to 11 lines. The simulator draws groups of
  7 lines on a 7-line grid, 6 expected in each frame. On the committed seed three of
  the seven groups cross the disc and move the disc correlation from 0.019 to 0.050
  px off. The truncated lines hold dark sky, as they do on the real truncated frame,
  so the scene shows that the navigator is unaffected by the truncation. The fused
  answer lands 0.028 to 0.10 px off over 24 draws, at high confidence on 13 of them
  and medium on 11. Removing any one defect on four draws leaves the answer 0.030 to
  0.089 px off, where all the defects together leave it 0.032 to 0.089 px off. The
  star-field technique returns spurious on every draw, because hot pixels and cosmic
  rays together crowd the stars out of the sources it matches, and the blob
  technique lands 5 to 10 px off the gibbous disc without flagging it.

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

* ``pds4_bundle_template_dir`` reads ``config.pds4.<dataset>.template_dir``,
  falls back to ``_default_pds4_template_dir``, and resolves a relative name
  against ``src/spindoctor/cli/pds4/templates/``.
* ``pds4_bundle_name``, ``pds4_bundle_version``, ``pds4_information_model_version``
  and ``pds4_schemas`` read ``config.pds4.<dataset>.bundle_name``, ``bundle_version``,
  ``information_model_version`` and ``schemas``, with no fallback.  The shipped
  ``coiss_saturn`` entry gives the schemas of the ``pds``, ``disp``, ``geom``,
  ``rings`` and ``cassini`` dictionaries, the last the Cassini mission dictionary,
  whose namespace the data label declares for its mission area.
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

The two dataset-identity hooks -- ``_dataset_name_for_pds4_config`` and
``_default_pds4_template_dir`` -- raise ``NotImplementedError`` on
``DataSetPDS3CassiniISS`` itself and are supplied by the Cruise and Saturn
subclasses; the bundle's name is read under the first, from the configuration.
That is what makes the undivided ``coiss`` dataset unable to build a bundle: a
bundle belongs to one of the two halves, not to both.

Only ``cassini_iss_saturn_1.0`` is shipped under the templates directory. The
cruise subclass refers to ``cassini_iss_cruise_1.0``, which is not present, so a
cruise bundle needs that directory supplied before it will build.

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
