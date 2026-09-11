=======================
PDS4 Bundle Generation
=======================

The :mod:`pds4` package builds PDS4-compliant bundles from SpinDoctor's per-image
navigation metadata and per-pixel backplanes. A bundle is the deliverable the
Ring-Moon Systems node ships to PDS for archive: one collection of data labels
(one per image), one collection of browse PNGs, plus the auxiliary collections
(context, document, xml_schema) and the bundle-level label that wires them
together. This chapter covers the bundle-generation driver, the per-dataset
extension points, the templated label workflow, and the output layout.

The user-facing CLI walkthrough lives at :doc:`/user_guide/user_guide_pds4_bundle`; this
chapter is the developer's reference.

Pipeline overview
=================

Bundle generation is a two-phase process driven by ``sd_create_bundle``:

1. **Per-image data labels.**  For each image in the input batch,
   :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` reads the
   ``_metadata.json`` produced by ``sd_offset`` and the
   ``_backplane_metadata.json`` produced by ``sd_backplanes``, populates a
   ``pdstemplate`` rendering context with per-image template variables, and
   writes the matching ``<image>_backplanes.lblx`` file (plus a copy of the
   browse PNG into the bundle's ``browse/`` tree). The backplane FITS file
   itself is copied from the backplane root into the bundle's ``data/`` tree,
   beside its label, and the label's data objects are read from the copy (see
   `The FITS and its data objects`_).

2. **Collections + bundle assembly.**  After every per-image data label is in
   place, :func:`~spindoctor.cli.pds4.collections.generate_global_index_files`
   reads every ``_supplemental.txt`` in the bundle's ``data/`` tree once, writes
   the per-bundle ``global_index_bodies`` and ``global_index_rings`` tables and
   their labels, and takes the range of the products' exposure epochs in the
   same read.  Then
   :func:`~spindoctor.cli.pds4.collections.generate_collection_files` walks the
   ``data/`` tree, collects every ``_backplanes.lblx`` it finds, sorts them by
   image name, and writes the ``collection_data.tab`` inventory with the matching
   ``collection_data.lblx`` label, which states the range it is handed, and the
   browse collection's inventory and label.  The other collection labels
   (context, document, xml_schema) and the top-level ``bundle.lblx`` are rendered
   by neither generator.

The driver runs phase 1 once per image (fan-out friendly — each image is
independent) and phase 2 once at the end (sequential — needs every per-image
label in place before it can build the inventory).

Driver: ``sd_create_bundle``
=============================

``sd_create_bundle`` (``src/spindoctor/cli/sd_create_bundle.py``) is the per-image
phase-1 entry point. Like the other CLIs it takes a ``DATASET_NAME``, the
selection flags from the matching :class:`~spindoctor.dataset.dataset.DataSet`
subclass (``--pds3-holdings-root`` among them, for a PDS3 dataset), the standard
environment options (``--config-file``, ``--bundle-results-root``,
``--nav-results-root``, ``--backplane-results-root``), and walks every
selected image.

A separate ``--collections`` flag triggers phase 2 (collection + bundle
labels) without re-rendering per-image data labels. Operators typically run
``sd_create_bundle DATASET --image-list FOO --no-collections`` in parallel
across many shards, then once with ``--collections`` to assemble the bundle.

Cloud-tasks variant ``sd_create_bundle_cloud_tasks`` reads the same task JSON
schema as ``sd_offset_cloud_tasks`` (see :doc:`/user_guide/user_guide_navigation`) so the
same task queue can drive offset + backplane + bundle in three queue passes.

Exit status
-----------

Before either pass processes anything, it checks that every template
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates` declares for
it is in the directory
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir` names, and
exits 1 naming each one that is not.  Every product of a pass renders from that
one directory, so a template that is missing is missing for every product, and a
per-product report would be the same line thousands of times.  The two passes
render different templates and each checks its own.

Each pass also checks, before it reads anything, that every backplane the
configuration declares is in a unit the bundle can use, through
:func:`~spindoctor.cli.pds4.collections.unusable_units`.  The labels pass holds
each document's statistics to the configured unit and the summary pass writes
each index column in the format that unit calls for, so a unit neither can use
-- a spelling the format table has no entry for, or an entry with no ``units``
at all -- is refused once, with every such entry named and the reason, rather
than once per image or after the collection files are on disk.

Before it processes anything, ``sd_create_bundle labels`` also requires
``<bundle_results_root>/<pds4_bundle_name()>/`` to be empty or absent, and exits
1 naming the directory when it is not.  A bundle is the product of one run: with
that precondition there is no stale label to detect and no directory to clean,
which is why nothing downstream of this check looks for either.  The check is
the local driver's alone.  ``sd_create_bundle_cloud_tasks`` calls
:func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` once per
task, many workers into one bundle root, so a per-image emptiness check there
would refuse every task after the first; the precondition belongs to the run
that owns the whole bundle, and a queue-driven run establishes it by starting
from an empty bundle root of its own.  ``sd_create_bundle summary`` does not
check it either: it reads the tree the labels pass wrote, so it requires a
populated bundle rather than an empty one.

``sd_create_bundle labels`` counts the images whose products it did not write --
an image whose data or browse label failed to render, an image whose summary PNG
was not in the navigation results, an image whose backplane metadata records a
statistic no global index column can hold (one in a unit other than the one the
configuration gives its plane, or a minimum or maximum that is not a finite
number), an image whose navigation document records no exposure epochs a label
can state (see `Epochs`_), an image whose backplane metadata has no FITS beside
it, an image whose FITS holds something its data label cannot describe (see
`The FITS and its data objects`_), and an image whose inputs it could not read --
and exits 1 when that count is not zero.  It closes with a line giving that count alongside the number
of images it labeled and the number it skipped, so a selection that matched
nothing reads as the zero it is.  It counts a batch that did not hold exactly
one image the same way; that is a guard on the one-image-per-batch invariant
:func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` also
asserts, and no selection argument this dataset offers can produce one.

An image the bundle has nothing to describe is skipped rather than failed and
does not count against the run: an image with no navigation metadata document,
an image whose navigation status is not ``success``, and a navigated image with
no backplane metadata document are all cases of a selection naming more images
than the bundle covers, which is the ordinary state of a selection made by
volume.  A document that is there but cannot be read is a different thing: the
generation raises, the driver logs the traceback naming the image, counts the
image against the run, and carries on to the next one.

A dry run reports what it would have processed and exits 0, once the
preconditions above are met: they are checked before ``--dry-run`` is read, so a
dry run over a missing template, an unusable unit or a populated bundle root
exits 1 naming what it found, like any other run.  Past them it writes nothing,
so it counts nothing against the run, including a batch it reports it could not
have processed.

``sd_create_bundle summary`` counts the collection and index labels it did not
write, over both generators, and exits 1 the same way.  The inventory and index
``.tab`` tables are written either way.  The data collection label counts as not
written when there is no range for it to state (see `Epochs`_).  The global index
is generated first, and it refuses a bundle with no ``data/`` directory, naming
the directory, before any product of the pass is cleared or written.  The pass also
exits 1 when a supplemental file holds a statistic no index column can -- one in
a unit other than the one the configuration gives its plane, or in none, or a
minimum or maximum that is not a finite number within the range of a float --
the check the labels pass makes per image, through
:func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic`, naming the
file and the plane and saying what the file records there.  It exits 1 the same
way on a supplemental file it cannot read, or one that does not hold a JSON
object, naming the file and the reason or what it holds: left out of the index,
the file's product would still be listed in the collection's inventory, with no
epochs for the range.  It exits 1 the same way on a supplemental file with no
data label beside it, or a data label with no supplemental file, naming both: the
collection inventory finds a product by its data label, and the index and the
range by its supplemental file.  The index tables and labels an earlier run wrote,
and its collection tables and labels, are cleared before the first supplemental
file is read, the collection files by the index generator since it runs first.
Every supplemental file is read, and every value in both index tables rendered,
before either table is opened, so a run refused over what the data tree holds
leaves no product of the pass, neither this run's nor an earlier run's.  A bundle
with no ``data/`` directory is refused before anything is cleared, so an earlier
run's products stay where it left them.

``sd_create_bundle_cloud_tasks`` reports a product it could not write as a
``status: error`` result carrying ``status_error: label_not_written``, and asks
for no retry: a template that could not be rendered will not render on a second
attempt.  Of the three up-front checks it makes the unit check alone, and makes
it per task: once the dataset is constructed, and before any document is read,
a task under a configuration in which
:func:`~spindoctor.cli.pds4.collections.unusable_units` finds a backplane comes
back as ``status_error: unusable_unit``, every such backplane and its reason in
``status_exception``, having generated nothing, and again asks for no retry,
since the configuration will not change on a second attempt.  The check each
document gets covers only the planes that document holds, so a task that did
not make this one would write labels the summary pass then refuses to index.
It makes neither the template check nor the empty-root check, because it holds
one task rather than the run: a template it cannot find raises out of every
task, and the empty bundle root is the queue-driven run's own precondition to
establish.

Per-dataset extension points
============================

PDS4 bundle generation is parameterized by the
:class:`~spindoctor.dataset.dataset.DataSet` subclass. The base class declares the
extension points as non-abstract methods that raise ``NotImplementedError`` —
a dataset that does not need PDS4 support can simply not override them, and
the bundle drivers refuse to run.

The full extension-point set:

- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir` — absolute
  path to the directory of ``pdstemplate`` ``.lblx`` files this dataset uses.
  Lookups consult ``config.pds4.<dataset_name>.template_dir`` first; relative
  paths resolve under ``src/spindoctor/cli/pds4/templates/``. The reference Cassini ISS
  Saturn dataset uses ``cassini_iss_saturn_1.0/``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name` — the bundle's
  external name (for example
  ``cassini_iss_saturn_backplanes_rsfrench2027``). The bundle root is
  ``<bundle_results_root>/<bundle_name>/``. Lookups consult
  ``config.pds4.<dataset_name>.bundle_name``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates` — the
  template filenames one pass must find in that directory, ``labels`` for the
  per-image pass and ``summary`` for the collection and index pass. Each pass
  checks them before it processes anything and refuses to run when one is not
  there, so this is where a dataset says what its template tree carries.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_path_for_image` — maps an
  image name to its position in the bundle's ``data/`` directory tree
  (typically a sharded path like ``1234xxxxxx/123456xxxx`` to keep per-leaf
  cardinality manageable on filesystems that struggle with very wide
  directories).
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub` — full per-image stub
  including the image name (e.g.
  ``1234xxxxxx/123456xxxx/1234567890w``). Builds the per-file paths under
  ``data/`` and ``browse/``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name` —
  inverse of the image-name transform baked into ``pds4_path_stub`` and the
  ``pds4_image_name_to_*`` builders. Each product is stored on disk under a
  filename whose stem is the LID part (e.g. ``1234567890w``); this hook lets
  the collection and global-index scanners recover the original image name from
  that stem and round-trip it back through the canonical LID builders instead
  of re-applying the transform.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lid` /
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_browse_lidvid`
  — emit the browse-product Logical Identifier (LID) and LID + version
  (LIDVID) for the given image name. LIDs follow the PDS4 namespace
  convention ``urn:nasa:pds:<bundle>:browse:<image>``.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lid` /
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_image_name_to_data_lidvid` —
  same, for the data product (the backplane ``.lblx`` + ``.fits`` pair).
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables` — returns a
  dict of template variables consumed by the per-image
  ``data.lblx`` / ``browse.lblx`` templates. Inputs are the
  :class:`~spindoctor.dataset.dataset.ImageFile`, the navigation metadata dict
  parsed from ``<image>_metadata.json``, and the backplane metadata dict
  parsed from ``<image>_backplane_metadata.json``. The dataset is free to
  derive any per-image quantity the templates reference (target body,
  observer, mid-time, exposure, filters, navigation offset and confidence,
  per-backplane min/max/units, and so on).

Reference implementation:
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS`
overrides every PDS4 hook above and serves as the canonical worked example.
Of its two registered subclasses only the Saturn one bundles as the package
ships: the cruise one names ``cassini_iss_cruise_1.0``, which does not ship, so
both passes refuse it at the missing-template check.  Galileo SSI
(:class:`~spindoctor.dataset.dataset_pds3_galileo_ssi.DataSetPDS3GalileoSSI`),
New Horizons LORRI
(:class:`~spindoctor.dataset.dataset_pds3_newhorizons_lorri.DataSetPDS3NewHorizonsLORRI`)
and Voyager ISS
(:class:`~spindoctor.dataset.dataset_pds3_voyager_iss.DataSetPDS3VoyagerISS`)
implement only
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir` and
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name`, over template
directories that do not ship.  Every other hook raises
:exc:`NotImplementedError`,
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates` among
them, so both passes stop on these datasets before they look for a template.

The ``pds4`` config block
-------------------------

``src/spindoctor/config_files/config_950_pds4.yaml`` populates ``config.pds4`` with
per-dataset bundle metadata: the bundle name, the template directory name,
the LID namespace prefix, and any per-bundle template defaults the
``pds4_template_variables`` hook draws from. See
:doc:`dev_guide_config_and_static_data` for the loader contract; the file
is loaded by the standard numeric-prefix order at the ``9xx`` "downstream
products" tier.

Templated label workflow
========================

Labels are rendered via the ``pdstemplate`` library — a Python expression
language embedded in PDS4 ``.lblx`` files (XML). Each template carries
expressions that resolve against a dictionary of variables; the
``pds4_template_variables`` hook is the contract that connects per-image
metadata to the templates.

A typical render looks like:

.. code-block:: python

   import pdstemplate

   from spindoctor.cli.pds4.labels import write_label

   template_path = template_dir / 'data.lblx'
   variables = dataset.pds4_template_variables(
       image_file=image_file,
       nav_metadata=nav_metadata,
       backplane_metadata=backplane_metadata,
   )
   template = pdstemplate.PdsTemplate(str(template_path))
   if write_label(template, variables, destination, logger=logger):
       logger.info('Generated PDS4 label: %s', destination)

The ``pdstemplate`` library handles the XML escaping, the expression syntax,
and the per-template error reporting; consumers only supply the variable
dictionary and the destination path.  The destination is an
:class:`~filecache.FCPath` naming
the label's place in the bundle, never a local cache path standing in for it:
on a cloud bundle root those are two different files, and the label belongs in
the bundle.

Label-write failures
--------------------

Constructing a ``PdsTemplate`` raises when the template file is not there.
Rendering one does not raise: an unresolved variable, a failed expression or a
template validation error is reported through the ``(errors, warnings)`` pair
``write`` returns, so a caller that discards it cannot tell a product that got a
label from one that did not.

Every label the bundle stage writes therefore goes through
:func:`~spindoctor.cli.pds4.labels.write_label`, which renders in
``pdstemplate``'s ``repair`` mode and acts on what comes back:

- Warnings are logged at warning level, and the label is written.
- Errors are logged at error level naming the label path, and the label is not
  written.  A render that drew errors writes nothing at all, so whatever was at
  that path before is what is there after -- nothing, in a bundle the labels
  pass wrote into an empty directory.

``write_label`` clears the label path before it renders.  For the labels pass
that changes nothing, because a bundle is written into an empty directory.  It
matters for a summary pass run a second time over a bundle it has already
summarized, where the first run's collection and index labels are still in
place: repair mode saves nothing when a render errors, so without clearing, the
earlier label would stay beside the inventory table this run has already
rewritten and describe data that is no longer there.  A label on disk is
therefore always one this run wrote.

A failed render does not stop the run.  Both of an image's labels are attempted
when the image has both products, and so is every collection and index label, so
a single run reports every label it could not write rather than one per run.
Whichever labels did render stay on disk, and the driver's exit status is what
says the bundle is incomplete.

An image whose navigation left no summary PNG is a failure rather than an image
without a browse product.  :func:`~spindoctor.navigate_image_files.navigate_image_files`
writes the summary PNG before the metadata document and under the same
condition, precisely so that a fault in the PNG is recorded as that image's
failure instead of leaving a success document beside no PNG; a success document
with no PNG beside it therefore means the input tree is broken.  The missing PNG
costs the image its browse products and nothing else: the data label is rendered
on its own account, stays if it rendered, and the image counts against the run
either way.

A template that is not in the dataset's template directory is not a label
skipped: each pass checks the templates its dataset declares before it processes
anything and refuses to run without them, so a render this far in has its
template.

Template tree
-------------

Each dataset's template directory under ``src/spindoctor/cli/pds4/templates/`` contains the
shipping ``.lblx`` files. The Cassini-ISS-Saturn-1.0 set is the reference
layout:

::

   src/spindoctor/cli/pds4/templates/cassini_iss_saturn_1.0/
     bundle.lblx                              # top-level bundle label
     readme.txt                               # bundle-level README
     data.lblx                                # per-image backplane data label
     browse.lblx                              # per-image browse-product label
     collection_data.lblx                     # data-collection label (CSV inventory)
     collection_browse.lblx                   # browse-collection label
     collection_context.lblx                  # context-collection label
     collection_context.csv                   # context inventory (static)
     collection_document.lblx                 # document-collection label
     collection_document.csv                  # document inventory (static)
     collection_xml_schema.lblx               # schema-collection label
     collection_xml_schema.csv                # schema inventory (static)
     global_index_bodies.lblx                 # per-bundle bodies summary
     global_index_rings.lblx                  # per-bundle rings summary
     cassini-iss-saturn-backplanes-user-guide.lblx  # bundle user-guide doc

The static inventory CSVs are copied verbatim into the bundle; the
per-image and per-bundle ``.lblx`` files are rendered fresh on every run.

The FITS and its data objects
=============================

The labels pass copies an image's ``<stub>_backplanes.fits`` from the backplane
root into the bundle's ``data/`` tree, beside its data label, which names the file
with no directory part.  ``BACKPLANE_PATH`` names the copy, so the size, checksum
and time the label states through ``pdstemplate``'s ``FILE_BYTES``, ``FILE_MD5``
and ``FILE_ZULU`` are the archived file's.  The copy is written to a local path and
uploaded, as the summary PNG's is, and those three functions read a local file, so
a bundle root in the cloud is not supported.  A navigated image whose backplane
metadata is there and whose FITS is not is failed before anything is written for
it: the backplane stage writes the FITS before its metadata document, and the
summary pass refuses a supplemental file with no data label beside it.

:func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits` then reads the
copy with ``astropy.io.fits``, no scaling applied, into a
:class:`~spindoctor.cli.pds4.data_objects.BackplaneFitsObjects`: one
:class:`~spindoctor.cli.pds4.data_objects.FitsHdu` per HDU, in file order, whose
header offset and length come from astropy's ``fileinfo()`` (``hdrLoc``, and
``datLoc`` less ``hdrLoc``), and for every image HDU past the primary a
:class:`~spindoctor.cli.pds4.data_objects.FitsArray` at ``datLoc``.  An array's
element type comes from its ``BITPIX`` through
:data:`~spindoctor.cli.pds4.data_objects.FITS_DATA_TYPES` (``IEEE754MSBSingle`` for
-32 and ``SignedMSB4`` for 32, the most significant byte first because FITS is
big-endian), its unit is its ``BUNIT`` when it has one, its ``Line`` and ``Sample``
extents are ``NAXIS2`` and ``NAXIS1``, and its local identifier is its HDU name in
lower case.  A float array's missing constant is the configuration's
``backplanes.masked_value``, spelled as the shortest decimal that reads back as the
32-bit float the plane holds; the function is handed the value rather than the
configuration, and refuses with :exc:`ValueError` one a 32-bit float does not hold
exactly.  The body identity map declares no missing constant and carries
:data:`~spindoctor.cli.pds4.data_objects.BODY_ID_MAP_DESCRIPTION` instead: its
``0`` is a pixel no body claimed, not a missing measurement.  The map is found by
:data:`~spindoctor.cli.backplanes.writer.BODY_ID_MAP_HDU_NAME`, the name the
backplane writer gives it.

``data.lblx`` renders the result, handed to it as ``BACKPLANE_FITS``.  A ``$FOR``
over its ``hdus`` writes a ``Header``, and an ``Array_2D_Image`` where the HDU has
an array, into ``File_Area_Observational`` after the ``File``; a second ``$FOR``
over its ``arrays`` writes one ``disp:Display_Settings`` per array into the
``Discipline_Area``, each referring to its array's identifier.  The fixed PDS4
values -- the ``FITS 3.0`` parsing standard, ``Last Index Fastest``, two axes named
``Line`` and ``Sample`` -- are literals in the template; everything that depends on
the file comes from the descriptor, so a plane the writer dropped is not described
and a frame with no ring backplanes has no ring arrays.

What the backplane writer does not write is refused rather than described, with an
:exc:`~spindoctor.cli.pds4.data_objects.UndescribableFitsError` naming the file and
the HDU: a ``BITPIX`` the mapping does not hold; an image that is not
two-dimensional; ``BSCALE`` or ``BZERO``, since a scaled array's stored values are
not its values; a primary HDU holding data, or an extension that is not an image,
either of which would leave data the label does not describe; a lower-case HDU name
that is not an XML name or repeats another's, since a ``local_identifier`` is an
XML ``ID`` and unique in the label; and a file astropy reads only with an error or a
warning, which is what a truncated file draws.  On a refusal the labels pass fails
the image, removes the copy and writes nothing else, logging the refusal; on a
local bundle root the copy's emptied directory stays.

Epochs
======

Every time a bundle label states is an exposure epoch the navigation recorded:
``start_et``, ``stop_et`` and ``midtime_et`` under ``navigation_result.times``, in
TDB seconds past J2000.  They are turned into UTC by one rule, in
:mod:`spindoctor.support.time`: :func:`~spindoctor.support.time.et_to_utc` writes
the plain ISO spelling the observation metadata and the statistics report use,
and :func:`~spindoctor.support.time.et_to_pds4_utc` the spelling a PDS4 label
takes, ``ASCII_Date_Time_YMD_UTC`` (``2004-02-07T04:25:35.585Z``), to a given
number of decimals, rounded to the nearer value of the last digit, or down, or up.
Both go from TDB to TAI to the calendar through ``julian``, whose leap-second table
gives the answer SPICE's ``et2utc`` gives; an integration test holds every cohort
epoch to the leapseconds kernel.  The C-kernel report converts through
``cspyce.et2utc`` instead, against the kernel its generator furnishes, and is the
one conversion outside the rule.

A data label's times come from the dataset's
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`.
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` writes
``START_DATE_TIME`` and ``STOP_DATE_TIME`` each to the nearest millisecond.  A Cassini
image's times are recorded to the millisecond and its epochs are computed from them,
so each epoch lies within a few nanoseconds of a millisecond and the nearest is the
recorded time, where a start rounded down or a stop rounded up would lose a
millisecond whenever the epoch lands on the far side.  ``IMAGE_MID_TIME`` is the
midpoint of the two as written, a half millisecond rounding up, through
:func:`~spindoctor.support.time.pds4_utc_midpoint`: an exposure an odd number of
milliseconds long has its midtime on a half millisecond, where the recorded midtime
epoch lands a few nanoseconds to either side, and PDS3's ``IMAGE_MID_TIME`` takes the
half up.  Before the hook is called,
:func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` holds the
navigation document to :func:`~spindoctor.cli.pds4.epochs.unrecorded_epoch`: a
``times`` block with all three epochs, each a finite number, the stop no earlier
than the start.  An image whose document fails the check is failed before anything
is written for it, the image and what its document lacks in the log, the way an
image whose backplane statistics no index column can hold is failed.

The data collection label states the range of the products' epochs: the least
start and the greatest stop over every supplemental file, written to whole seconds
with the start rounded down and the stop up, so the range contains every product's
own start and stop as its data label writes them: the nearest millisecond of an epoch
is never before the whole second at or before it, nor after the one at or after it.
The range is taken in
the one read of the supplemental files the summary pass makes -- the global
index's -- by an :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, and
:func:`~spindoctor.cli.pds4.collections.generate_global_index_files` returns it in
its :class:`~spindoctor.cli.pds4.collections.GlobalIndexOutcome`.  That is why the
summary pass runs the index first and hands the range to
:func:`~spindoctor.cli.pds4.collections.generate_collection_files`, and why the
driver holds it for any other label that states the bundle's range:
:meth:`~spindoctor.cli.pds4.epochs.EpochRange.template_variables` gives such a
label its ``EARLIEST_START_DATE_TIME`` and ``LATEST_STOP_DATE_TIME`` without a
second computation.  A scan that read no supplemental file, or read one whose
document the epoch check refuses, yields a
:class:`~spindoctor.cli.pds4.epochs.NoEpochRange` saying why, and the data
collection label is then counted as not written rather than rendered with empty
dates.  A supplemental file the index generator cannot read, or that does not hold
a JSON object, never reaches the scan: the generator refuses the run on it, naming
the file and the reason or what it holds.

Output layout
=============

A finished bundle has the standard PDS4 directory shape:

::

   <bundle_results_root>/<bundle_name>/
     bundle.lblx
     readme.txt
     data/
       <pds4_bundle_path_for_image>/
         <image>_backplanes.lblx
         <image>_backplanes.fits             # copied from backplane_results_root
     browse/
       <pds4_bundle_path_for_image>/
         <image>_browse.lblx
         <image>_browse.png                  # copied from nav_results_root
     collection/
       data/
         collection_data.lblx
         collection_data.csv
       browse/
         collection_browse.lblx
         collection_browse.csv
       context/
         collection_context.lblx
         collection_context.csv              # static
       document/
         collection_document.lblx
         collection_document.csv             # static
         <user-guide doc>.lblx
       xml_schema/
         collection_xml_schema.lblx
         collection_xml_schema.csv           # static
     index/
       global_index_bodies.lblx
       global_index_bodies.csv
       global_index_rings.lblx
       global_index_rings.csv

Testing bundle generation
=========================

The suite asks two different questions of the bundle stage, and answers them in
two different environments.

The first is plumbing: which file goes where, which variable reaches which
template, what a render that errors leaves behind. Those tests run a duck-typed
dataset over tiny templates the test itself wrote, so every variable in play is
one the test controls and a failure names the wiring that broke.

The second is content: what a label actually says. Those tests run the
registered
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSSaturn`
over the shipped ``cassini_iss_saturn_1.0`` templates, on the products a
navigation run and the backplane stage leave behind. Neither environment
answers the other's question: a label rendered from a template the test wrote
says whatever the test put there, and a plumbing failure inside the shipped
template set is a needle in three hundred lines of XML.

The inputs for the second come from :mod:`tests.mini_nav_results`, a package that
builds a miniature of what a navigation run leaves on disk -- three Cassini
images, of which two navigated and one did not; a real backplane FITS and its
metadata document per navigated image, written by the backplane stage's own
:func:`~spindoctor.cli.backplanes.writer.write_fits`; a real summary PNG,
because the browse label states its size and its checksum; and the index row an
enumeration hands on with each image. The two navigated images shard into
different bundle directories and only one of them has ring backplanes, so a
run over the cohort exercises both layouts.

Every image is built from its epoch and nothing else. The spacecraft clock
readings a document records and the number the image is named for are both
derived from it, so no document of the cohort can carry a reading it counted
one of the others from a different moment.

The conversion those readings come from is a line through two correlation
points read out of the Cassini mission clock kernel, calibrating both where the
clock started and the rate it runs at. A clock kernel is a line only in pieces,
so the two part company away from the points: measured against the kernel over
the 16 days the cohort spans, the fixture is never more than half a tick out,
and each cohort epoch converts to exactly the tick the kernel returns for it. A
cohort reaching much further would have to measure that again, or take a third
point.

The suite cannot measure it. The cohort's self-tests compare a reading to a
reading and a name to the reading it came from, so they hold the fixture to
itself; an anchor moved by an hour leaves them all green with every image
renamed. What reports that is an integration test that furnishes the mission
clock kernel and converts every cohort epoch again, which is excluded from the
default run because the kernels are not there to furnish.

Nothing the cohort produces is checked in. A test takes it as a session-scoped
fixture, :func:`mini_nav_cohort <tests.conftest.mini_nav_cohort>`, built into a
temporary directory and torn down
with the session; a test asserts that none of its products reaches the working
tree. To build one outside the suite, to read or to call the bundle stage
over:

.. code-block:: bash

   PYTHONPATH=src python -m tests.mini_nav_results cohort /tmp/cohort

What that writes is what the bundle stage's library entry points read: the
navigation and backplane roots, and the images to pass them. It is not a
holdings tree, so ``sd_create_bundle`` cannot enumerate it -- a PDS3 selection
by volume reads that volume's index table, and the cohort writes none.

Adding an instrument to the cohort is a module beside
:mod:`~tests.mini_nav_results.cohort_cassini`. The
FITS files, the browse images and the documents it implies exist only while a
test is running, so it costs the repository nothing.

Adding PDS4 support to a new dataset
====================================

The end-to-end checklist:

1. Override every ``pds4_*`` method on the new
   :class:`~spindoctor.dataset.dataset.DataSet` subclass. Use
   :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` as
   the reference implementation. The methods that absolutely must work
   are :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub`, the four
   ``pds4_image_name_to_*_lid[vid]`` methods,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name`
   (the inverse of ``pds4_path_stub``'s image-name transform), and
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`.
2. Drop a per-dataset template directory under
   ``src/spindoctor/cli/pds4/templates/<dataset>_<version>/`` containing the ``.lblx``
   files and the static inventory CSVs. Copy from
   ``cassini_iss_saturn_1.0/`` and adapt the field set.
3. Add an entry under ``pds4.<dataset_name>:`` in
   ``config_950_pds4.yaml`` that points at the new template directory and
   sets the bundle name plus any per-bundle defaults the
   ``pds4_template_variables`` hook draws from.
4. Add an integration smoke test that renders one image through
   ``sd_create_bundle`` and asserts the resulting ``data.lblx`` validates
   against the PDS4 schema. The Cassini ISS test under
   ``tests/integration/`` is the pattern to follow.

API reference
=============

The :mod:`pds4` package has no autogenerated entry under
:doc:`/api_reference`; the module's public surface is the entry points
listed below, plus the
:class:`~spindoctor.dataset.dataset.DataSet` ``pds4_*`` extension hooks
documented above.

- :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` — phase 1, one image.
- :func:`~spindoctor.cli.pds4.collections.generate_collection_files` — phase 2, collection
  + bundle assembly.
- :func:`~spindoctor.cli.pds4.collections.generate_global_index_files` — per-bundle bodies
  / rings global indexes.
- :func:`~spindoctor.cli.pds4.labels.write_label` — the one place a label is
  written, shared by both.
- :func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic` — the one
  check both passes hold every statistic of a document to, its unit and its
  values.
- :func:`~spindoctor.cli.pds4.collections.unusable_units` — the configured
  backplanes no index column has a format for, which both passes and the
  cloud-task worker refuse before reading a document.
- :func:`~spindoctor.cli.pds4.epochs.unrecorded_epoch` — the one check a
  navigation document's exposure epochs are held to before a label states them.
- :func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits` — the headers
  and arrays of a backplane FITS, read from the copy in the bundle for its data
  label, and :exc:`~spindoctor.cli.pds4.data_objects.UndescribableFitsError`, what
  it raises for a FITS the backplane writer does not write.
- :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan` and
  :class:`~spindoctor.cli.pds4.collections.GlobalIndexOutcome` — the range of the
  products' epochs, taken in the global index's read of the supplemental files
  and handed to the collection generator.
- :func:`~spindoctor.support.time.et_to_pds4_utc` — the PDS4 spelling of an epoch,
  beside :func:`~spindoctor.support.time.et_to_utc`, in the one conversion, and
  :func:`~spindoctor.support.time.pds4_utc_midpoint`, the midpoint of two times so
  written, which is what a data label's ``IMAGE_MID_TIME`` states.
