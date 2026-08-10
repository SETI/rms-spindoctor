============================
Corrected-Pointing C-Kernels
============================

Overview
========

Navigation measures where a camera was actually pointing. It records that
measurement in each image's ``_metadata.json`` both as a pixel offset and as a
corrected C-matrix -- the rotation taking a vector from J2000 into the camera
frame at the exposure midtime. ``sd_create_ck`` turns those matrices into SPICE
C-kernels, so the same measurement reaches ``oops``, ISIS, or a plain
``spiceypy`` script through one ``furnsh`` rather than having to be applied as a
pixel offset by hand. Every geometry computation those tools perform is then
made against the navigated pointing.

A corrected kernel is an **overlay on one original kernel**. It carries one
type-3 segment per navigated exposure, each covering that exposure and nothing
else, in a file named after the original whose pointing it corrects. The
correction is written for the object the original describes -- a spacecraft bus
or a scan platform -- so no fabricated camera frame is involved and no other
kernel has to change. Which object that is, and which spacecraft clock its time
tags are encoded against, is stated in each instrument's own chapter under
:doc:`instruments/instruments`.

The pixel offset is unaffected. Every consumer that reads ``offset`` from the
metadata keeps working exactly as it does; the kernels are a second expression
of the same measurement, for consumers that would rather furnish a file than
apply an offset.

What these kernels claim
========================

Three properties decide whether a corrected kernel answers the question a
consumer is asking. None of them is a detail.

The originals are still required
--------------------------------

Pointing is corrected **only inside a navigated exposure**. A corrected file
advertises coverage over exactly the exposures it carries segments for and
claims nothing outside them, so a lookup between two images falls through to
whatever else is furnished. Furnish the originals as well, and furnish them
first: a consumer who loads only the corrections gets no pointing at all
between images, and one who loads the corrections before the originals gets the
originals everywhere, because SPICE resolves overlapping C-kernels in favor of
the one furnished last.

The meta-kernel a run writes encodes that order, which is why it is the
recommended way to load a corrected set (see `Loading the kernels`_).

The record epochs are exact; between them the segment interpolates
------------------------------------------------------------------

A segment carries records at the exposure start, the midtime and the stop, plus
a one-second cadence once the exposure reaches ten seconds. It
reproduces the corrected attitude at those epochs exactly. Every other epoch in
the window is interpolated between the bracketing records, and the interior
error that interpolation leaves is **not bounded by anything**.

How large that error is depends on the instrument and on how the spacecraft was
moving during the exposure, so this guide does not quote a single figure for it.
The interpolation error is an angle, and the same angle is a different number of
pixels on every camera, so an error that matters on one camera can be negligible
on another. How much attitude structure a segment interpolates across also
differs by mission and by how the platform was slewing.

What is fixed is the shape of the effect rather than its size. The error is zero
at every record epoch and grows between them; it is largest where the baseline's
rate changes inside the window; it shrinks as records are added, which is what
the one-second cadence buys on a long exposure; and it is present in the same
size when the correction itself is zero, which is how it is known to be
interpolation loss rather than an error in the correction. Each instrument's
chapter under :doc:`instruments/instruments` carries the characterization for
that instrument, including the cases where there is no interpolation error at
all; where a chapter reports no measured figure yet, a consumer who needs a
bound should measure it for the frames they care about.

A consumer that evaluates geometry at the exposure midtime is unaffected and
exact: the midtime is a record epoch. That is what the backplane and
reprojection stages do, and what most single-epoch geometry does. A consumer
integrating smear across the exposure, or sampling attitude at arbitrary
interior epochs, is subject to the interpolation error described above.

An instrument whose navigated attitude is constant across the exposure is a
separate case entirely. Its segment carries that single corrected attitude,
constant across the window, and there is nothing to interpolate and no
interpolation error at any epoch. Each instrument's chapter says whether it is
one of those.

Eligibility carries no quality threshold
----------------------------------------

Any image whose navigation reached a status of ``success`` or ``conflicted``
and recorded a corrected matrix gets a segment. There is **no confidence or
rank threshold**, and a ``conflicted`` result -- one where two techniques
disagreed and the ensemble reported the conflict -- is written like any other.

The consequence is that filtering is the consumer's job, and the report is
where the material to filter on lives. Its ``status``, ``status_reason``,
``confidence`` and ``confidence_rank`` columns carry each image's own
measurement as the navigation recorded it, and each corrected kernel's comment
area repeats the same numbers for the images inside that file. A consumer who
wants only high-confidence pointing reads those and decides; nothing in the
kernel does it for them.

.. note::

   ``confidence`` and ``confidence_rank`` are calibrated against simulated
   planted-truth recovery, not against real-image accuracy. See the note on
   confidence in :doc:`user_guide_navigation` before using them as a
   threshold.

Which images get a segment
==========================

Every image the run considered appears in the report exactly once, with either
the corrected file carrying its segment or one of these reasons it has none:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ``omission_reason``
     - Meaning
   * - ``not_eligible``
     - The image's navigation status is neither ``success`` nor
       ``conflicted``, or it recorded no corrected matrix.
   * - ``rotation_unsupported``
     - The navigation fitted a camera rotation. The rotation turns about a
       per-technique pivot that the result does not record, so the correction
       cannot be expressed as an attitude and none is claimed. It is reached
       only after the eligibility check above, so it applies to every
       otherwise eligible image of an instrument whose configuration fits
       rotation, and to none of an instrument whose configuration does not; an
       image of such an instrument whose navigation neither succeeded nor
       conflicted is reported as ``not_eligible`` instead.
   * - ``botsim_loser``
     - An exposure taken on two cameras at once, on an instrument that can do
       that. The two frames share one spacecraft attitude and one attitude
       cannot carry two different corrections, so one member of the pair keeps
       its correction and the other yields. A frame yields only to a partner
       that actually writes: one whose partner is ineligible, or has no
       reproducing baseline, keeps its own correction.
   * - ``no_reproducing_baseline``
     - No C-kernel under the run's kernel directories reproduces the attitude
       this image navigated against. Either the kernel set has changed since
       navigation, or the original the image used is not among the directories
       given.
   * - ``baseline_coverage_gap``
     - The original that reproduces this image's attitude supplies no pointing
       at one of the segment's record epochs. The pairing is made at the
       exposure midtime and a segment carries records at the exposure start and
       stop as well, so an exposure straddling the end of an original's
       coverage is reproduced and then cannot be written. It is ordinary near a
       segment boundary and on a long exposure.

The set is closed, and every member of it is one a run can produce: an image
whose pointing the writer cannot express as a segment at all is not reported
here but stops the run, since a run that has found something wrong with the
kernels or the metadata should not bury it in one image's row. The set is the
same for every mission because every consumer of the report reads the same
column; which of these reasons a given instrument can actually produce, and
why, is stated in its chapter under :doc:`instruments/instruments`.

An omitted image gets no segment and no uncorrected copy of one: its pointing
falls through to the originals, exactly as an epoch between exposures does.

``no_reproducing_baseline`` doubles as the detector for a kernel set that
changed since navigation ran. Each image is paired with its original by
reproducing the uncorrected attitude the navigation recorded, to within a
nanoradian, rather than by trusting the kernel names in the metadata. A
baseline that no longer produces that attitude is refused rather than corrected
against a baseline the measurement was never made on.

``baseline_coverage_gap`` is the opposite case and is deliberately a reason of
its own, so that the detector above keeps meaning what it says. The original
did reproduce the recorded attitude; it simply does not cover the whole
exposure. A run reporting these is not a run whose holdings have drifted, and
an image reported this way is one whose exposure ran past the end of a
segment's window rather than one whose kernel is missing.

Files a run writes
==================

One invocation covers one mission and writes everything into ``--output-dir``:

.. code-block:: text

   <output-dir>/
       <original-1>_nav.bc          # one corrected kernel per original
       <original-2>_nav.bc
       <mission>_nav.tm             # meta-kernel furnishing the set in order
       <mission>_ck_report.csv      # one row per image considered

**Corrected kernels** take the original's basename with ``_nav`` inserted
before the extension, so the pairing is legible without opening either file:
an original named ``abcd.bc`` becomes ``abcd_nav.bc``. One file is written per
original that some image navigated against; an original no image used produces
no file, and so does one whose every image was omitted. A corrected file's
size grows with the number of images corrected against its original -- one
small segment per exposure -- not with the original's own size, and
regenerating one original's corrections does not touch the others. No PDS label
files are written.

**The meta-kernel** is named ``<mission>_nav.tm`` and lists every original
followed by every correction, by absolute path. It is written only when at
least one corrected kernel was.

**The report** is named ``<mission>_ck_report.csv``. It is written on every run
that reaches the writing phase -- a run stopped by a refusal, or one that
selected no images, writes none -- and
it covers every image the run considered, including the images that received no
segment.

Inside a corrected kernel, each segment's identifier is the image's basename,
so a listing tool such as ``ckbrief`` names the image every segment came from.
The comment area records the generator version, the configuration hash, the
original kernel the file corrects, the spacecraft clock kernel its time tags
are encoded against, and one line per image carrying the same offset, sigma,
confidence, rank, status and status reason the report carries. Read it with the
NAIF ``commnt`` utility or with ``dafec``.

Every corrected segment carries angular velocity, copied unchanged from the
original; a segment whose attitude is constant across the exposure carries
zeros instead, which is that attitude's true rate and is written without
consulting the original at all. Where the rates are copied -- that is, for a
time-varying segment -- an exposure whose original does not supply angular
velocity at every record receives no segment at all, and the run stops and says
so. That is
because a segment declaring no angular velocity is not read as one whose
angular velocity is unknown: SPICE skips it for ``ckgpav`` and for ``sxform``
and answers those from the next loaded kernel that does carry angular velocity
for the same object and epoch, which would be the original and its uncorrected
attitude. Since every segment carries angular velocity, ``ckgp``, ``ckgpav``,
``pxform`` and ``sxform`` all report the correction.

Loading the kernels
===================

With the meta-kernel
--------------------

One ``furnsh`` of the meta-kernel loads the originals and then the corrections,
in that order:

.. code-block:: python

   import spiceypy

   spiceypy.furnsh('/data/nav/ck/<mission>_nav.tm')

The paths inside are absolute, so the meta-kernel works from any working
directory. It furnishes only the original kernels that some correction mirrors;
a leapseconds kernel, a spacecraft clock kernel and a frame kernel are still
the consumer's own to furnish, as they would be for the originals alone.

Without the meta-kernel
-----------------------

Furnish the originals first and the corrections after. Precedence between two
C-kernels covering the same object and epoch goes to the one furnished last:

.. code-block:: python

   import spiceypy

   spiceypy.furnsh('naif0012.tls')     # leapseconds
   spiceypy.furnsh('<clock>.tsc')      # the mission's spacecraft clock
   spiceypy.furnsh('<frames>.tf')      # the mission's frame kernel
   spiceypy.furnsh('abcd.bc')          # the original
   spiceypy.furnsh('abcd_nav.bc')      # the correction, loaded after

Reversing the last two lines is not an error and produces no message: the
originals simply answer everywhere and the corrections are never seen.

The corrected kernels are not registered in the SPICE database that ``oops``
selects kernels from. A consumer furnishes them explicitly, by the meta-kernel
or by name.

Checking what a corrected file covers
-------------------------------------

``ckcov`` reports each corrected file's coverage as exactly the exposure
windows it carries segments for, in whatever time system is asked for. That is
the direct way to answer "is this epoch corrected?" without inspecting the
pointing:

.. code-block:: python

   import spiceypy

   spiceypy.furnsh('naif0012.tls')     # leapseconds
   spiceypy.furnsh('<clock>.tsc')      # the mission's spacecraft clock
   windows = spiceypy.ckcov('abcd_nav.bc', ck_object, False, 'SEGMENT', 0.0, 'TDB')

The leapseconds and spacecraft clock kernels are furnished first because
coverage in TDB is what they convert the segments' clock ticks into.
``ck_object`` is the mission's corrected object, which its chapter under
:doc:`instruments/instruments` names.

The report
==========

The report is CSV with a header row. Read it by header name: the column set is
version 1 and is expected to grow as consumers ask for more.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Column
     - Contents
   * - ``image_name``
     - Basename of the image, from ``observation.image_name``. It is also the
       segment identifier inside the kernel.
   * - ``utc``
     - The exposure midtime as a UTC calendar string,
       ``YYYY-MM-DDTHH:MM:SS.sss``, converted from ``et``.
   * - ``et``
     - The exposure midtime, TDB seconds past J2000, from
       ``navigation_result.times.midtime_et``.
   * - ``sclk``
     - The exposure midtime as the spacecraft clock string the pipeline
       recorded, from ``navigation_result.times.sclk_midtime``.
   * - ``offset_dv``, ``offset_du``
     - The navigated offset in pixels, from the top-level ``offset``
       (``[dv, du]``), reported unrounded as it was recorded.
   * - ``sigma_dv``, ``sigma_du``
     - The per-axis one-sigma uncertainty in pixels, from
       ``navigation_result.sigma_px``, reported as recorded.
   * - ``confidence``
     - The confidence recorded for the result, from the top-level
       ``confidence``.
   * - ``confidence_rank``
     - The confidence tier recorded for the result, from
       ``navigation_result.confidence_rank``.
   * - ``status``
     - The navigation status, from the top-level ``status``.
   * - ``status_reason``
     - Why the navigation reported that status, from
       ``navigation_result.status_reason``.
   * - ``source_bc``
     - Basename of the corrected kernel carrying this image's segment. Empty
       when the image received none.
   * - ``omission_reason``
     - Why the image received no segment, from the table above. Empty when it
       received one.

Exactly one of ``source_bc`` and ``omission_reason`` is filled on every row.
Any other empty cell means the metadata recorded no such value -- an image that
failed to load records a name and a status and nothing else -- which is how the
report distinguishes "not measured" from a measurement that happened to be
zero.

The ``sd_create_ck`` program
============================

``sd_create_ck`` writes one mission's corrected kernels, its meta-kernel and
its report. It reads the navigation results a previous ``sd_offset`` run wrote
and the SPICE kernels those runs used; it does not read images and does not
navigate anything.

.. code-block:: bash

   sd_create_ck MISSION --kernel-dir DIR [--kernel-dir DIR ...] --output-dir PATH [options]

``MISSION`` is positional and required, and case-insensitive. It selects which
metadata documents under the navigation results root the run considers, matched
against each document's ``observation.instrument``. The permitted values are
exactly the instruments SpinDoctor navigates, each linking to its own chapter,
which is where that mission's corrected object, kernel directories, naming
conventions and omission reasons are stated:

* ``coiss`` -- :doc:`instruments/cassini_iss`
* ``gossi`` -- :doc:`instruments/galileo_ssi`
* ``nhlorri`` -- :doc:`instruments/newhorizons_lorri`
* ``vgiss`` -- :doc:`instruments/voyager_iss`

Environment options
-------------------

* ``--config-file PATH`` (repeatable): configuration files overriding the
  defaults. With none given, ``./nav_default_config.yaml`` is loaded if it
  exists. See :doc:`/introduction_configuration`.

* ``--nav-results-root PATH``: the root of the navigation results to read.
  The whole tree is walked for ``*_metadata.json`` files. Takes precedence
  over the ``environment.nav_results_root`` configuration variable, which
  takes precedence over the ``NAV_RESULTS_ROOT`` environment variable. One of
  the three is required.

* ``--kernel-dir DIR`` (repeatable, at least one required): a directory of
  SPICE kernels. These directories serve two purposes at once: every C-kernel
  in them is a candidate original to pair images against, and all of them
  together resolve the kernel basenames each image's provenance recorded, so
  the leapseconds, frame and spacecraft clock kernels the navigation used must
  be among them. Directories are **not** searched recursively, so a holdings
  tree that keeps its kernels in per-kind subdirectories needs one flag per
  subdirectory.

Image selection options
-----------------------

* ``--start-time UTC``: ignore images whose exposure midtime is before this
  time. Default: no lower bound.

* ``--stop-time UTC``: ignore images whose exposure midtime is after this
  time. Default: no upper bound.

Both accept any calendar string SPICE parses, for example
``2004-01-01T00:00:00``. An image that recorded no usable midtime cannot be
placed in time and is ignored whenever either bound is given; the run reports
how many.

Output options
--------------

* ``--output-dir PATH`` (required): where the corrected kernels, the
  meta-kernel and the report are written. It is created if it does not exist,
  and it must be a local directory, since SPICE creates a kernel by name on
  the local filesystem. Relative paths are resolved to absolute before
  anything is written, so the meta-kernel names its kernels by paths that work
  from any working directory.

Logging options
---------------

``sd_create_ck`` processes images individually and accepts the full shared
logging surface: ``--log-root``, ``--log-level``, ``--log-level-main``,
``--log-level-image``, and the ``--log-main-to-console`` /
``--log-main-to-file`` / ``--log-image-to-console`` / ``--log-image-to-file``
pairs with their ``--no-`` forms. Defaults, precedence and the configuration
equivalents are documented once, for every program, in
:doc:`user_guide_logging`.

The run's own log lands at
``{log_root}/sd_create_ck/main_{timestamp}.log``, and each image's log under
the ``ck`` stage at ``{log_root}/ck/{results_path_stub}_{timestamp}.log``.
Every image that receives no segment is reported in its own log and, as one
line, in the run's, so an operator watching a batch does not have to open
per-image logs to learn that corrections stopped being written. The run ends
with a count per disposition.

Example
-------

Write one mission's corrections for one navigation results tree, pairing images
against the reconstructed kernels:

.. code-block:: bash

   sd_create_ck MISSION \
       --nav-results-root /data/nav/results \
       --kernel-dir DIR_HOLDING_THE_LEAPSECONDS_KERNEL \
       --kernel-dir DIR_HOLDING_THE_SPACECRAFT_CLOCK_KERNEL \
       --kernel-dir DIR_HOLDING_THE_FRAME_KERNEL \
       --kernel-dir DIR_HOLDING_THE_ORIGINAL_C_KERNELS \
       --output-dir /data/nav/ck

Every kernel the navigation recorded has to be reachable, and directories are
not searched recursively, so one flag is needed per kernel kind rather than one
for the tree. Which directories a given mission's holdings put those kernels
in is stated in that mission's chapter under :doc:`instruments/instruments`,
each of which carries a worked invocation. Restricting a run to part of a
mission is a matter of adding ``--start-time`` and ``--stop-time``.

A C-kernel can describe an object whose spacecraft clock none of the furnished
kernels defines, and the scan indexes the rest of the file rather than
stopping. The run log carries one warning naming such objects. They can never
supply a baseline, since their coverage cannot be expressed in TDB, and an
image that actually corrects one is refused with the missing clock named. A
mission whose holdings contain such an object says so in its chapter, so the
warning is expected rather than investigated.

Exit status
-----------

The program exits 0 when every metadata file it was pointed at could be read,
whether or not every image received a segment -- an image omitted for a reason
is reported in the CSV, which is the answer. It exits 1 when any file under the
navigation results root could not be read as a document naming its image and
mission; those files are named in the run log and the run continues on what it
could read, so a batch wrapper can tell a clean run from one that silently
skipped its input. Selecting no images at all is not an error: the run says so,
writes nothing, and exits 0 -- unless some of its input was unreadable, which
still exits 1 even when nothing was selected.

Refusals worth knowing about
----------------------------

A few conditions stop the run rather than being reported per image, because
each of them would otherwise corrupt every image alike or force a silent
choice. A run stopped by any of them writes **nothing at all**: every segment
of every output file is built, and every destination judged, before the first
file is opened, so such a refusal leaves no corrected kernels, no meta-kernel
and no report, and the run can be repeated once its cause is fixed without
first clearing a partial set out of the way.

That covers everything the run can know before it starts writing, and it is not
the same as a guarantee that writing cannot fail part way through. It can, for
reasons no check made beforehand can see: the device filling up, a path or a
permission changing between the check and the write, and a record set SPICE
refuses only once a file is open. One more failure lands after every kernel is
written rather than during: a kernel path -- an original's as readily as a
correction's -- that a text kernel cannot express, which the meta-kernel refuses
when it is rendered.

In all of those cases the corrected kernels already written stay on disk while
the meta-kernel and the report do not, so these are the failures where the
output directory has to be cleared before the run is repeated. The run log names
every file it wrote, and each of them is a complete, valid kernel.

* **Two versions of one spacecraft clock kernel, or two frame kernels defining
  a frame the run's images name.** A text kernel's last assignment wins, so the
  whole corpus would be reproduced against whichever version sorted last and
  every image navigated under the other would be reported as having no
  baseline. The run names the two kernels and what they disagree about.

* **A clock or frame kernel the images recorded that no ``--kernel-dir``
  holds.** Named, rather than left to surface as an image whose baseline
  appears to have drifted.

* **An output path the run cannot write.** Every destination is judged
  together, before the first file is opened, and the refusal names every one
  that failed and why, so a set is cleared in one pass rather than one rerun
  per file. A path fails when something already occupies it -- a rerun writes
  a fresh corrected kernel rather than appending to or overwriting the old
  one, so remove or move the previous file first -- when it is a symbolic
  link, which would put the
  kernel wherever the link points rather than in the output directory, when its
  name is longer than the 60 characters SPICE stores as a file's internal name,
  when the full path is longer than the 255 characters SPICE accepts in a file
  name -- a meta-kernel naming it would be written and then refused by every
  consumer that furnishes it --
  or when the output directory does not exist and cannot be created, or exists
  and cannot be written to.

* **A time range whose start is after its stop.** A swapped
  ``--start-time``/``--stop-time`` pair would select nothing, and a run that
  wrote nothing for that reason would be indistinguishable from a clean run
  over a quiet span, so it is refused by name instead.

* **A metadata document that cannot be read as a navigated image**; an image
  whose segment copies its baseline's rates and whose baseline supplied
  pointing at every record but angular velocity at only some of them; and an
  exposure whose window is so long that the cadence would need more records
  than a segment holds, which means the recorded epochs are not an exposure.
  None has an entry in the closed set of omission reasons, so none is reported
  as one. An exposure its baseline does not cover is not among them: that one
  has a reason, ``baseline_coverage_gap``, and omits the one image.

Related chapters
================

* :doc:`instruments/instruments` -- one chapter per instrument, carrying the
  corrected object, the spacecraft clock, the kernel directories, the baseline
  naming conventions, the segment shape, the omission reasons that instrument
  can produce, and its interpolation-error characterization.
* :doc:`user_guide_navigation` -- the navigation run that records the
  C-matrices, and the ``pointing`` and ``times`` metadata blocks this program
  reads.
* :doc:`user_guide_logging` -- the shared logging surface and where log files
  land.
* :doc:`/dev_guide/dev_guide_ck_kernels` -- the frame conventions, the
  derivation of the corrected attitude, and the writer's internals.
