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
     - ``oops.hosts.cassini.iss``

Loading the image
=================

:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.from_file` calls
``oops.hosts.cassini.iss.from_file(path, fast_distortion=..., return_all_planets=...)``.
Both keyword arguments are read from ``**kwargs`` and default to ``True``.
``fast_distortion`` selects the host's polynomial inverse rather than an
iterative one, which is what makes per-pixel geometry affordable in a batch;
``return_all_planets`` keeps every planet the host finds rather than the
closest one alone, which the multi-body navigation models need.

Beyond the host call it does four things:

1. Records ``obs.abspath`` (the resolved local path) and ``obs.image_url`` (the
   absolute source URL) from an ``FCPath``, so a run against remote holdings
   still names the file it read.
2. Selects the configuration block. ``'_CALIB' in fc_path.name.upper()``
   chooses ``cassini_iss_calib``; anything else chooses ``cassini_iss``.
3. Indexes that block by ``obs.detector.lower()``. A missing section, or a
   section with no entry for the detector, raises ``ValueError`` naming the
   detectors that are present, rather than failing later on a missing key.
4. Resolves the extended-FOV margin. The configured value is a mapping keyed
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
* ``DESCRIPTION`` and ``OBSERVATION_ID``, both optional and recorded as
  ``None`` when absent.

:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.get_public_metadata`
also refuses a detector that is neither ``NAC`` nor ``WAC``, because the
instrument LID encodes the camera as ``issna`` or ``isswa`` and a malformed LID
must never reach a PDS4 label.

**Index columns.** ``_INDEX_COLUMNS`` is ``FILE_SPECIFICATION_NAME``.
``_INDEX_CAMERA_COLUMNS`` is ``('INSTRUMENT_ID',)`` and ``_INDEX_CAMERA_MAP``
is ``{'ISSNA': 'NAC', 'ISSWA': 'WAC'}``, whose values are exactly what
``ObsCassiniISS.camera`` returns. The BOTSIM grouping asks for four more
columns per row (``SHUTTER_MODE_ID``, ``IMAGE_NUMBER``, ``OBSERVATION_ID``,
``IMAGE_TIME``), which is why grouping is a separate code path rather than a
filter.

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
``marker_value: NaN``, and its ``image_quality_thresholds`` carry only the two
``_if`` thresholds and the two fractions. It has no ``saturation_dn``, no
``full_well_dn``, no ``expected_noise_dn``, no ``read_noise_dn``, no
``star_flux_dn_per_s_vmag0``, and -- the consequential one -- no
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
``expected_noise_dn`` and ``read_noise_dn``, the calibrated block's
``blank_max_if`` and ``noisy_threshold_if``, and both blocks'
``mag_offset_table`` entries. The ``_sources`` convention is not used in this
file.

Photometric and PSF calibration
===============================

**Limiting magnitude.** Both cameras follow the Pogson-ratio form, an anchor
plus ``log(texp) / log(2.512)``, so each factor of 2.512 in exposure buys one
magnitude of depth. The two anchors are measured rather than derived, and each
names the frame it came from:

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

**Magnitude offsets.** ``fallback_combo`` is ``'CL1+CL2'``, the two clear
filter slots, and the table carries one entry with a default of 0.0. Both are
placeholders.

**Photometric zero points.** ``star_flux_dn_per_s_vmag0`` is 3.33e5 for the
NAC and 9.29e4 for the WAC, described in the file as interim values sized from
each camera's electron zero point over its standard gain state.

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
about the boresight, applied by ``oops/hosts/cassini/iss.py``. The correction
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

**Instrument keys.** Four: ``coiss_nac`` and ``coiss_wac`` resolve to the
``cassini_iss`` block's two detectors, and ``coiss_calib_nac`` and
``coiss_calib_wac`` to the ``cassini_iss_calib`` block's. The artifact catalog
aliases the calibrated keys onto the raw ones, so a calibrated scene gets the
same artifact defaults as its raw counterpart.

**PSF kernels.** ``coiss_nac`` is
``{sigma_v: 0.70, sigma_u: 0.70, w: 0.12, r0: 3.0, n: 4.0}``, tuned against the
image-library cohort's eleven contributing narrow angle star frames: the cohort
measures encircled-energy radii of 0.91 and 1.79 pixels at 50% and 80%, and
these parameters reproduce 0.90 and 1.72 through the same estimator. The
80%-to-50% ratio of 1.97, against 1.52 for a pure Gaussian, is what forces
substantial mid-range wing energy, so this is an *effective* in-window kernel
rather than an optical one: the fitted ``w = 0.12`` carries measured 1-8 pixel
halo energy that a FWHM-derived core put nowhere. A first fit at ``w = 0.20``
matched the encircled energy but over-lifted the wide-field halo.

``coiss_wac`` is ``{sigma_v: 1.05, sigma_u: 1.05, w: 0.12, r0: 3.0, n: 4.0}``,
from the cohort's two wide angle star frames and nine usable cutouts. Treat it
as cohort-limited and revisit it when more wide angle star frames land.

:func:`~spindoctor.sim.forward.psf.psf_truncation_for_instrument` truncates
this instrument's kernels at 32 detector pixels rather than the default 16,
because the documented wings run further out. The match keys on the ``coiss``
prefix, so all four instrument keys get the wider window.

**Distortion residuals.** ``coiss_nac`` is
``{k1: 3.17e-04, k2: -3.51e-04, nonradial_rms_px: 0.0}`` and ``coiss_wac`` is
``{k1: 9.0e-06, k2: 5.48e-05, nonradial_rms_px: 0.0}``, both measured by the
star-field distortion analysis. The corrected field is sub-pixel and close to
radially symmetric, which is why the non-radial wander is zero.

**Artifact-mode availability.** Both cameras are in the CCD set and carry the
whole common surface. They also carry four further modes:
``bright_dark_pairs``, ``quantization_lut``, ``quantization_ls8b`` and
``partial_lines``. There are no exclusions recorded against either camera key.

**Realism-match status.** The narrow angle PSF is tuned against eleven frames.
The wide angle PSF rests on two frames.

Image library and test coverage
===============================

**Cohort.** Sixty-two sidecars, 58 narrow angle and 4 wide angle, spread across
the scene classes. Every one names a calibrated product.

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

**Unit tests.** ``tests/spindoctor/inst/test_inst_cassini_iss.py`` covers the
config-block selection, the shutter-mode property and the metadata surface.

PDS4 hooks
==========

This dataset is the reference implementation, and implements every hook.

* ``pds4_bundle_template_dir`` reads ``config.pds4.<dataset>.template_dir``,
  falls back to ``_default_pds4_template_dir``, and resolves a relative name
  against ``src/spindoctor/cli/pds4/templates/``.
* ``pds4_bundle_name`` reads ``config.pds4.<dataset>.bundle_name`` and falls
  back to ``_default_pds4_bundle_name``.
* ``pds4_bundle_path_for_image`` maps ``N1234567890`` to
  ``1234xxxxxx/123456xxxx/``, raising on a name shorter than 11 characters
  rather than returning an empty string a caller would concatenate into a
  malformed path.
* ``pds4_path_stub`` appends the LID part, which moves the camera letter to a
  lowercase suffix: ``N1454725799`` becomes ``1454725799n``.
  ``pds4_lid_part_to_image_name`` inverts exactly that transform.
* The four LID and LIDVID builders share the same LID part and differ only in
  the ``browse`` / ``data`` collection and the ``::1.0`` version suffix.
* ``pds4_template_variables`` emits the camera width, the three exposure
  times from the navigation metadata, the bundle and product LIDs, and roughly
  sixty ``cassini:`` namespace variables read straight from the PDS3 index row.

The three dataset-identity hooks -- ``_dataset_name_for_pds4_config``,
``_default_pds4_template_dir`` and ``_default_pds4_bundle_name`` -- raise
``NotImplementedError`` on ``DataSetPDS3CassiniISS`` itself and are supplied by
the Cruise and Saturn subclasses. That is what makes the undivided ``coiss``
dataset unable to build a bundle: a bundle belongs to one of the two halves,
not to both.

Only ``cassini_iss_saturn_1.0`` is shipped under the templates directory. The
cruise subclass names ``cassini_iss_cruise_1.0``, which is not present, so a
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
  raw block's ``expected_noise_dn`` and ``read_noise_dn``, the calibrated
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
