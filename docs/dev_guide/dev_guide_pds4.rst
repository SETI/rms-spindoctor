=======================
PDS4 Bundle Generation
=======================

The :mod:`pds4` package builds PDS4-compliant bundles from SpinDoctor's per-image
navigation metadata and per-pixel backplanes. A bundle is the deliverable the
Ring-Moon Systems node ships to PDS for archive: one collection of data labels
(one per image), one collection of browse PNGs, the context, document,
miscellaneous, SPICE kernel and XML schema collections, and the bundle label that
wires them together. This chapter covers the bundle-generation driver, the
per-dataset extension points, the templated label workflow, the global index and
the miscellaneous collection, the bundle's run-level products, and the output
layout.

The user-facing CLI walkthrough lives at :doc:`/user_guide/user_guide_pds4_bundle`; this
chapter is the developer's reference.

Pipeline overview
=================

Bundle generation is a two-phase process driven by ``sd_create_bundle``:

1. **Per-image products.**  For each image in the input batch,
   :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` reads the
   ``_metadata.json`` produced by ``sd_offset`` and the
   ``_backplane_metadata.json`` produced by ``sd_backplanes``, populates a
   ``pdstemplate`` rendering context with per-image template variables, and
   writes into the bundle's ``data/`` tree the image's ``<image>_backplanes.lblx``
   label, a copy of its ``<image>_backplanes.fits`` beside it, and its
   ``<image>_supplemental.txt`` file, and into its ``browse/`` tree a copy of the
   summary PNG and its ``<image>_summary.lblx`` label. The label's data objects
   are read from the source FITS before anything is written into the bundle, and
   the copy is the same bytes (see `The FITS and its data objects`_).

2. **Collections and indexes.**  After every per-image data label is in place,
   :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` reads
   every ``_supplemental.txt`` in the bundle's ``data/`` tree once, writes the
   ``global_bodies_index`` and ``global_rings_index`` tables and their labels,
   and the miscellaneous collection that lists them, into ``miscellaneous/`` (see
   `The global index and the miscellaneous collection`_), and takes the range of
   the products' exposure epochs in the same read.  Then
   :func:`~spindoctor.cli.pds4.collections.generate_collection_files` walks the
   ``data/`` and ``browse/`` trees, collects every ``_backplanes.lblx`` and
   ``_summary.lblx`` it finds, sorts each set by product name -- the file name
   less its suffix, which is the last part of the member's LID -- checks that each
   image's products agree, and writes the ``collection_data.csv`` inventory from
   the data labels and the ``collection_browse.csv`` inventory from the browse
   labels, and their labels, the data collection label stating the range it is
   handed.  An inventory lists one product per
   line, as ``P,<LIDVID>``, with no header, and every line, the last included,
   ends in a line feed alone, so the record count its label states is the
   number of products the collection holds.  A collection with no product gets
   neither an inventory nor a label (see `Exit status`_).  Last,
   :func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` writes
   the bundle's run-level products: the readme, the context, document, SPICE
   kernel and XML schema collections, the user guide when the template directory
   holds it, and the bundle label, which states the same range and is kept only
   over a bundle holding every collection it declares (see `The bundle's
   run-level products`_).

The driver runs phase 1 once per image (fan-out friendly — each image is
independent) and phase 2 once at the end (sequential — needs every per-image
label in place before it can build the inventory).

Driver: ``sd_create_bundle``
=============================

``sd_create_bundle`` (``src/spindoctor/cli/sd_create_bundle.py``) has two
subcommands. ``sd_create_bundle labels`` runs phase 1: it takes a
``DATASET_NAME``, the selection flags from the matching
:class:`~spindoctor.dataset.dataset.DataSet` subclass (``--pds3-holdings-root``
among them, for a PDS3 dataset), the environment options (``--config-file``,
``--bundle-results-root``, ``--nav-results-root``, ``--backplane-results-root``),
the logging options and ``--dry-run``, and walks every selected image.
``sd_create_bundle summary`` runs phase 2 once over the bundle: it takes a
``DATASET_NAME``, ``--config-file``, ``--bundle-results-root`` and the logging
options.

The cloud-tasks variant ``sd_create_bundle_cloud_tasks`` runs phase 1 from a
queue, one image per task. A task carries a ``dataset_name`` and a ``files`` list
holding that image's ``image_file_url``, ``label_file_url`` and
``results_path_stub``, and optionally its ``index_file_row``: the fields of a
``sd_offset_cloud_tasks`` task that the bundle needs.

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

Each index column is written in the format
:data:`~spindoctor.cli.pds4.global_index.INDEX_VALUE_FORMATS` gives its unit, each
chosen from what one pixel resolves, within the roughly seven significant digits a
float32 array carries.  Nothing
checks the configured units when a bundle is written: a unit the table has no
format for is a ``KeyError`` from the summary pass's lookup.  The guard is the
two tests over the shipped configuration that :doc:`dev_guide_backplanes`
describes.

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
configuration gives its plane, or a minimum or maximum that is NaN or
infinite), an image whose navigation recorded no pointing (see `Epochs`_),
and an image whose processing raised an error -- and exits 1 when that count is not
zero.  An image with such a statistic is failed before anything is
written for it, and the log names the image, the plane and what the document
records there.  The run closes with a line giving that count alongside the
number of images it labeled and the number it skipped, so a selection that
matched nothing reads as the zero it is.  It counts a batch that did not hold
exactly one image the same way; that is a guard on the one-image-per-batch
invariant :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files`
also asserts.

An image the bundle has nothing to describe is skipped rather than failed and
does not count against the run: an image with no navigation metadata document,
an image whose navigation status is not ``success``, and a navigated image with
no backplane metadata document are all cases of a selection naming more images
than the bundle covers, which is the ordinary state of a selection made by
volume.  An error raised while one image is processed is logged with its
traceback naming the image, counts the image against the run, and the run
carries on to the next one.

A dry run reports what it would have processed and exits 0, once the
preconditions above are met: they are checked before ``--dry-run`` is read, so a
dry run over a missing template or a populated bundle root
exits 1 naming what it found, like any other run.  Past them it writes nothing,
so it counts nothing against the run, including a batch it reports it could not
have processed.

``sd_create_bundle summary`` counts the collection, index and run-level labels it
did not write, over its three generators, and exits 1 the same way.  The index
tables are written either way, and so is the inventory of a collection whose label
fails to render.  An index table no image gives a row is not written, nor its label,
and does not count: a table label states at least one record, and a bundle with no
image with ring backplanes is not a fault.  The bundle label counts when it is not written: it is kept only
over a bundle holding a label for every collection it declares, and with no range
to state it is not rendered (see `The bundle's run-level products`_).  A metakernel
label that fails to render leaves the SPICE kernel collection with no member, so that
collection is not written and counts as well, and so does the bundle label.  A user
guide the template directory does not hold is one warning, and does not count.  A collection whose label cannot state what PDS4 requires of it is not
written at all -- neither its inventory nor its label, and whatever an earlier run
left at either path is removed -- and counts once among the labels not written,
with an error naming the collection and each reason.  A collection label states at
least one record, so a collection with no label of its kind on disk -- no data
label, or no browse label -- is never written.  The data collection label also
states the range of its products' epochs, so a data tree
holding no supplemental file writes no data collection (see `Epochs`_).  The
global index is generated first, and it refuses a
bundle with no ``data/`` directory, naming the directory, before any product of
the pass is cleared or written.  The pass also exits 1 when a supplemental file
holds a statistic no index column can -- one in a unit other than the one the
configuration gives its plane, or a minimum or maximum that is NaN or infinite --
the check the labels pass makes per image, through
:func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic`, naming the
file and the plane and saying what the file records there.  The index tables and
labels and the miscellaneous collection an earlier run wrote, its collection
inventories and labels, and its run-level products are cleared before the first
supplemental file is read, all of them by the index generator since it runs first.  Every supplemental file is read, and every value in
both index tables rendered, before either table is opened, so a run refused over a
supplemental file leaves no product of the pass, neither this run's nor an earlier
run's.  The pass reads each
supplemental file as the labels pass wrote it and checks nothing in it but the
statistics; whether a data label is beside it is the product check that follows, and a
supplemental file with no data label beside it adds no row to either index table.
Anything else unexpected raises, and the run ends with exit status 1.

The summary pass inventories each collection from the labels of its own kind, the
data collection from the data labels in ``data/`` and the browse collection from
the browse labels in ``browse/``, and holds each image's products against each
other, since every data product has a browse product.  An image with a data label
and no browse label, or with a browse label or a supplemental file and no data
label, disagrees: the pass logs one error naming the image, the files of it that
are there and the label it lacks, counts it, and exits 1, its closing line giving
the images whose products disagree beside the labels not written.  Such an image
is still listed in whichever inventory holds a product of it.  These are the states
a labels pass that failed an image leaves -- an image whose summary PNG was missing
has a data label and no browse label, and one whose data label failed to render has
its supplemental file, written first, and perhaps a browse label -- so a summary
pass over that bundle exits 1 as the labels pass did.  The check and the
empty-collection rule are one rule at two scales -- each image holds all its
products, and each collection at least one member -- and the collection rule
refuses an empty collection even over a bundle with no image, where the check has
nothing to count.  A data label with no
supplemental file is not checked, since the labels pass writes an image's
supplemental file before its data label.

A summary pass that exits 1 does not repair the bundle: the generators report what
they cannot describe, and what they wrote stays, but for two things.  A collection
that cannot be written has whatever an earlier run left at its paths removed, and
the bundle label, rendered and then found to name a collection the bundle does not
hold, is removed in the same run.  After one, the browse labels can name a data
collection that was not written, and the data labels a browse collection that was not
written; a user guide whose label failed is in
``document/user_guide/`` unlabeled and unlisted; and a metakernel whose label failed
is in ``spice_kernels/`` with neither label nor collection.  The bundle label names no
collection the bundle does not hold, except after a run refused before it cleared
anything -- over a bundle with no data directory -- which leaves an earlier run's
products as they were, its bundle label among them.  A run refused after it cleared
can leave empty collection directories.  None of this is repaired in place: clear the
bundle directory and regenerate the bundle into it, as after a labels pass that exits
1.

``sd_create_bundle_cloud_tasks`` reports a product it could not write as a
``status: error`` result carrying ``status_error: label_not_written``, and asks
for no retry: a template that could not be rendered will not render on a second
attempt.  It makes neither of the local driver's up-front checks, the template
check and the empty-root check, because it holds one task rather than the run:
a template it cannot find raises out of every task, and the empty bundle root is
the queue-driven run's own precondition to establish.

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
  file names one pass takes from that directory, the templates it renders and the
  files it copies: ``labels`` for the per-image pass and ``summary`` for the pass
  writing the collections, the index and the run-level products. Each pass checks
  them before it processes anything and refuses to run when one is not there, so
  this is where a dataset says what its template tree carries.  The user guide is
  not among them, since a bundle is written without it.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_user_guide_file_name` — the file
  name of the bundle's user guide, a PDF, in the template directory. Its label's
  template is the name :func:`~spindoctor.dataset.dataset.pds4_label_name` gives it,
  the same name ending in ``.lblx``, the rule every label beside its file follows.
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
The Cassini ISS class itself, registered as ``coiss``, leaves its configuration
name, template directory and bundle name to its subclasses and raises
:exc:`NotImplementedError` for them, so both passes stop on it when they ask for
the template directory, as they do on ``sim``, whose hooks all raise it.  Every
registered name but ``sim`` has a ``_pds3`` alias naming the same class, which
bundles, or does not, the same way.

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
     readme.txt                               # bundle-level README (copied)
     data.lblx                                # per-image backplane data label
     browse.lblx                              # per-image browse-product label
     collection_data.lblx                     # data-collection label (CSV inventory)
     collection_browse.lblx                   # browse-collection label
     collection_context.lblx                  # context-collection label
     collection_context.csv                   # context inventory (copied)
     collection_document.lblx                 # document-collection label
     collection_document.csv                  # document inventory (copied)
     collection_spice_kernels.lblx            # SPICE-kernel-collection label
     collection_spice_kernels.csv             # SPICE kernel inventory (copied)
     kernels.ker                              # metakernel (copied)
     kernels.lblx                             # metakernel label
     collection_xml_schema.lblx               # schema-collection label
     collection_xml_schema.csv                # schema inventory (copied)
     global_bodies_index.lblx                 # bodies index label
     global_rings_index.lblx                  # rings index label
     collection_miscellaneous.lblx            # miscellaneous-collection label
     cassini-iss-saturn-backplanes-user-guide.lblx  # user-guide label
     cassini-iss-saturn-backplanes-user-guide.pdf   # the user guide, when it exists

The labels pass renders ``data.lblx`` and ``browse.lblx`` for each image.  The
summary pass renders every other label and copies the files marked copied, the
user guide among them when the directory holds it; the directory ships no user
guide.

The FITS and its data objects
=============================

The labels pass copies an image's ``<stub>_backplanes.fits`` from the backplane
root into the bundle's ``data/`` tree, beside its data label, which names the file
with no directory part.  ``BACKPLANE_PATH`` names the copy, so the size, checksum
and time the label states through ``pdstemplate``'s ``FILE_BYTES``, ``FILE_MD5``
and ``FILE_ZULU`` are the archived file's.  The copy is written to a local path and
uploaded, as the summary PNG's is, and those three functions read a local file, so
a bundle root in the cloud is not supported.

:func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits` reads the source
FITS with ``astropy.io.fits``, before the copy is made, into a
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
``backplanes.masked_value`` as the 32-bit float the plane holds, spelled as the
shortest decimal that reads back as that value when parsed as a 64-bit float, so a
reader comparing in either precision finds it.  The body identity map declares no missing constant and carries
:data:`~spindoctor.cli.pds4.data_objects.BODY_ID_MAP_DESCRIPTION` instead: its
``0`` is a pixel no body claimed, not a missing measurement.  The map is found by
:data:`~spindoctor.cli.backplanes.writer.BODY_ID_MAP_HDU_NAME`, the name the
backplane writer gives it.  Every float array carries a description as well, built
from what the configuration and the file know -- its name, the ``oops`` backplane
method :func:`~spindoctor.cli.pds4.data_objects.configured_methods` gives for it,
and its ``BUNIT`` -- and a sentence saying a pixel the plane does not cover holds
the missing constant; nothing is said of the geometry the method computes.

``data.lblx`` renders the result, handed to it as ``BACKPLANE_FITS``.  A ``$FOR``
over its ``hdus`` writes a ``Header``, and an ``Array_2D_Image`` where the HDU has
an array, into ``File_Area_Observational`` after the ``File``; a second ``$FOR``
over its ``arrays`` writes one ``disp:Display_Settings`` per array into the
``Discipline_Area``, each referring to its array's identifier.  The fixed PDS4
values -- the ``FITS 3.0`` parsing standard, ``Last Index Fastest``, two axes named
``Line`` and ``Sample``, and a display with ``Sample`` running left to right and
``Line`` top to bottom -- are literals in the template; everything that depends on
the file comes from the descriptor, so a plane the writer dropped is not described
and a frame with no ring backplanes has no ring arrays.  The same label describes
the supplemental file as a ``Stream_Text`` over its whole length, ``7-Bit ASCII
Text`` with ``Line-Feed`` records: the pass writes it as the ASCII bytes of the
one JSON object :func:`~spindoctor.support.file.json_as_string` produces, which
escapes every character outside ASCII.

The builder describes what :func:`~spindoctor.cli.backplanes.writer.write_fits`
writes and refuses nothing: the FITS and its metadata are written by this
repository's own programs, and the cohort tests, which run the real writer and
hold every stated offset to the file's bytes, catch a change to the writer.
The source is described before the copy is made, and the copy is byte-identical,
so the source's description is the copy's.

Epochs
======

Every exposure time a label states (``start_date_time``, ``stop_date_time``, and
the collection's range) comes from the epochs the navigation recorded:
``start_et``, ``stop_et`` and ``midtime_et`` under ``navigation_result.times``, in
TDB seconds past J2000.  They are turned into UTC by one rule, in
:mod:`spindoctor.support.time`: :func:`~spindoctor.support.time.et_to_utc` writes
the plain ISO spelling the observation metadata and the statistics report use,
and :func:`~spindoctor.support.time.et_to_pds4_utc` the spelling a PDS4 label
takes, ``ASCII_Date_Time_YMD_UTC`` (``2004-02-07T04:25:35.585Z``), to a given
number of decimals, rounded to the nearer value of the last digit, or down, or up;
a leap second is written as second 60.
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
half up.  The navigation writes ``navigation_result.times`` only beside the pointing
it solved, and records a success with no pointing when the attitude cannot be computed
or the instrument has no SPICE camera frame mapped.
:func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` fails such an
image before anything is written for it, the log naming the image; it checks only
that ``navigation_result.times`` is there, and where it is, the epochs are read as
recorded.  The summary pass needs no check of the times: the labels pass fails such
an image before it writes anything for it, its supplemental file included.

The data collection label states the range of the products' epochs: the least
start and the greatest stop over every supplemental file, written to whole seconds
with the start rounded down and the stop up.  The range is taken in the one read
of the supplemental files the summary pass makes -- the global
index's -- by an :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, and
:func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` returns it in
its :class:`~spindoctor.cli.pds4.global_index.GlobalIndexOutcome`.  That is why the
summary pass runs the index first and hands the range to
:func:`~spindoctor.cli.pds4.collections.generate_collection_files`.  A scan that
read no supplemental file yields no range, and the data collection is then not
written, neither its inventory nor its label, rather than labeled with empty dates
(see `Exit status`_).

The global index and the miscellaneous collection
=================================================

:func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` writes two tables
into the bundle's ``miscellaneous/`` directory, each with its label beside it, and then
the collection that holds them:

- ``global_bodies_index.tab``, one row for each body of each image the data collection
  holds;
- ``global_rings_index.tab``, one row for each such image with ring backplanes;
- ``collection_miscellaneous.csv`` and ``collection_miscellaneous.lblx``.

Each product's LID is built from the bundle's name, as the bundle's own is, by
:func:`~spindoctor.cli.pds4.global_index.index_lid`:
``urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index`` and
``...:global_rings_index``, at the version
:data:`~spindoctor.cli.pds4.global_index.INDEX_VERSION`.

**Which images get a row.**  The rows are exactly the images the data inventory lists:
both take the data labels in the data tree from
:func:`~spindoctor.cli.pds4.collections.data_products`, so the tables and the inventory
cannot disagree about what the bundle holds.  The values come from each image's
supplemental file.  A supplemental file with no data label beside it adds no row; its
statistics are still checked and its epochs still taken, as every supplemental file's
are.

**The table.**  Each table is fixed width, as the reference bundle's index tables are:
a header line naming the fields, separated by commas, and then the rows.
:func:`~spindoctor.cli.pds4.global_index.lay_out_table` pads every field to the longest
value written in its column, a statistic right-justified and text left-justified, puts a
comma between fields and a line feed after the last, and returns the header, the records
and where each :class:`~spindoctor.cli.pds4.global_index.IndexField` lies.  A field's
length comes from its column's values rather than from its format, since values under
one format differ in length (``1.000`` and ``-12.500``).  The table is written as ASCII
through ``FCPath.open``.

**The columns.**  The bodies table begins with ``pds:logical_identifier``
(``ASCII_LID``), ``body_name`` and ``file_spec``, the path of the data label relative to
the bundle's root; the rings table with ``pds:logical_identifier`` and ``file_spec``.
Then each configured plane gives its table two columns, the least and the greatest value
its statistic spans, as :class:`~spindoctor.cli.pds4.global_index.IndexColumn` entries
built from the plane's entry in ``config_900_backplanes.yaml``, whose ``index`` block
names the two columns, their ``data_type`` and their descriptions:

.. code-block:: yaml

   - name: body_latitude
     method: latitude
     units: rad
     index:
       data_type: ASCII_Real
       minimum:
         name: geom:minimum_latitude
         description: >-
           The least planetocentric latitude over the pixels where the body_latitude
           backplane has a value for the body.
       maximum:
         name: geom:maximum_latitude
         description: >-
           The greatest planetocentric latitude over the pixels where the
           body_latitude backplane has a value for the body.

A column is named for a PDS4 dictionary attribute, with the dictionary's prefix, where it
holds the quantity that attribute defines, as ``geom:minimum_latitude`` and
``rings:minimum_ring_radius`` do, and has a name of its own otherwise, as
``minimum_body_longitude`` does: both dictionaries define a longitude range as wrapped at
the prime meridian, and the statistics are a plain least and greatest.  A column's unit
is the unit its statistic is in, the plane's ``units`` restated through
:func:`~spindoctor.cli.backplanes.statistics.statistics_units`, so an angular column is in
degrees although its array is in radians (see :doc:`dev_guide_backplanes`), and its
format is the one :data:`~spindoctor.cli.pds4.global_index.INDEX_VALUE_FORMATS` gives that
unit.

**Missing values.**  Where an image has no statistic for a plane, both of its cells hold
the configured masked value, ``backplanes.masked_value``, written in the column's format:
``-999.000`` in a ``deg`` column, ``-999.0`` in ``km``, ``-999.00000000`` in
``deg/pixel`` and ``-999.00`` in ``km/pixel``.  Every statistic field of a label declares
that text as the ``missing_constant`` of a ``Special_Constants`` block, in the spelling
its cells have, so a reader comparing a cell's text with the declared constant finds it,
and every value of a column, a missing one included, is written in one form.

**The label.**  ``global_bodies_index.lblx`` and ``global_rings_index.lblx`` are
``Product_Ancillary`` labels, each a ``Header`` over the header line and a
``Table_Character`` over the records.  The generator hands each template ``INDEX_LID``,
``INDEX_TABLE_PATH``, ``HEADER_LENGTH`` (the header line's length, its line feed
included, which is also the table's offset), ``RECORD_LENGTH`` and ``FIELDS``, the
fields the table was laid out with.  A ``$FOR`` over ``FIELDS`` writes one
``Field_Character`` per column, so a change to the configuration moves a table and its
label together.  ``records`` is the table's line count less the header line.

**A table with no row.**  ``PDS4_PDS_1O00.xsd`` gives ``records`` a minimum of 1, so no
label can describe an empty table.  A table no image gives a row, such as the rings table
of a bundle with no image with ring backplanes, is not written, nor its label, and is not
listed; the run says so at info level and counts nothing against the run.

**The collection.**  After the tables, the generator writes the miscellaneous collection,
as the data inventory is written after its labels, through
:func:`~spindoctor.cli.pds4.collections.write_collection`, the one writer of the
generated collections: a ``P`` line for each index product whose label is on disk, then
an ``S`` line for each secondary member of the document inventory the template directory
ships, taken through :func:`~spindoctor.cli.pds4.bundle_products.secondary_members`, so
that the two inventories cite the same context products and ISS data user guide.  The
collection label, ``collection_miscellaneous.lblx``, has ``collection_type``
``Miscellaneous``.  A collection with no member is not written and counts as a label not
written, as the data and browse collections do.

The index products and the collection are cleared with the rest of the summary pass's
products before any supplemental file is read (see `Exit status`_).

The bundle's run-level products
===============================

:func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` writes the
products of the bundle as a whole, last in the summary pass, into the bundle's own
directory from the dataset's template directory.  Some are rendered from a template and
some are copied as they are:

- ``readme.txt`` is copied to the bundle's root.
- The context, document, SPICE kernel and XML schema collections are each an inventory
  the template directory ships, ``collection_<name>.csv``, copied into the collection's
  directory, and a label, ``collection_<name>.lblx``, rendered beside it and handed the
  inventory's path as ``COLLECTION_<NAME>_CSV_PATH``.
- The metakernel ``kernels.ker`` is copied into ``spice_kernels/`` and its label
  ``kernels.lblx`` rendered beside it, handed the copy's path as ``METAKERNEL_PATH``.
  The shipped metakernel lists no SPICE kernels: which kernels it lists has not been
  decided, so it is the ``KPL/MK`` identification word and a comment block, with no
  ``KERNELS_TO_LOAD`` assignment, which SPICE refuses empty.  Its label and the data
  label's ``geom:SPICE_Kernel_Files`` comment say so.
- The user guide is the PDF
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_user_guide_file_name` names.  When the
  template directory holds it, it is copied into ``document/user_guide/`` and its label
  rendered beside it from the template of the same stem ending in ``.lblx``, handed the
  copy's path as ``USER_GUIDE_PATH``.  When it does not, neither is written and the pass
  logs one warning naming the file; that is not a label the pass failed to write.
- ``bundle.lblx`` is rendered last, at the bundle's root, handed ``BUNDLE_LID``
  (``urn:nasa:pds:<bundle name>``), the readme's path as ``README_PATH``, and the range
  of the products' epochs the data collection label states (see `Epochs`_).

An inventory's primary members are products of this bundle -- the user guide in the
document collection, the metakernel in the SPICE kernel collection -- and one rule
covers both: the template directory's inventory lists the product as its ``P`` line,
beside ``S`` lines for the external documents and context products, and the pass writes
the inventory as it is when that product's label is in the bundle, and without its
``P`` lines when it is not, whether because the template directory holds no user guide
or because the label failed to render.  A collection left with no member is not written
at all, as the data and browse collections are not: neither its inventory nor its label,
whatever an earlier run left at either path is removed, and it counts once among the
labels not written.  The document collection keeps its ``S`` members, so it is always
written; the SPICE kernel collection, whose one member is the metakernel, is not written
when the metakernel's label is not, and the bundle label, which declares it, goes too.

The bundle label declares every collection the bundle holds, one ``Bundle_Member_Entry``
each, so it is kept only over a bundle that holds them all.  The pass renders it, reads
the LID of each member entry, and looks for a ``collection_*.lblx`` one directory below
the bundle's root whose logical identifier is that LID.  When one is missing -- a
collection whose label failed to render or was not written, or one the template declares
before any pass writes it -- the label is removed, one error names each LID it lacks, and
it counts as a label not written.  With no range to state it is not rendered at all.
Either way whatever an earlier run left at the path is gone, so a summary pass that
reaches the bundle label leaves none naming a collection that is not there; a pass
refused before it cleared anything leaves an earlier run's (see `Exit status`_).

Every label is attempted, whichever of them fail, and each one not written is counted
(see `Exit status`_); a file copied stays whether or not its label renders.  The
generator itself removes three things: each label, just before it renders it; the
inventory and label of a collection left with no member; and the bundle label, when it
has no range to state or declares a collection the bundle lacks.  Everything else an
earlier run wrote is cleared by
:func:`~spindoctor.cli.pds4.bundle_products.clear_bundle_products`, which removes every
path the generator can write, and ``document/user_guide/`` when that leaves the
directory empty in a bundle on the local file system; a remote store holds no directory
apart from the files in it.  The global index generator, which the summary pass runs
first, calls it with its own clearing, once it has found the bundle's data directory and
before it reads any supplemental file, so a run-level product on disk after a summary
pass that got past that check is one that pass wrote.

Output layout
=============

The two passes write this tree:

::

   <bundle_results_root>/<bundle_name>/
     bundle.lblx                             # summary pass
     readme.txt                              # summary pass, copied
     data/
       collection_data.csv                   # summary pass
       collection_data.lblx                  # summary pass
       <pds4_bundle_path_for_image>/
         <image>_backplanes.lblx
         <image>_backplanes.fits             # copied from backplane_results_root
         <image>_supplemental.txt
     browse/
       collection_browse.csv                 # summary pass
       collection_browse.lblx                # summary pass
       <pds4_bundle_path_for_image>/
         <image>_summary.lblx
         <image>_summary.png                 # copied from nav_results_root
     context/
       collection_context.csv                # summary pass, copied
       collection_context.lblx               # summary pass
     document/
       collection_document.csv               # summary pass
       collection_document.lblx              # summary pass
       user_guide/                           # only when the template directory holds it
         <user guide>.pdf                    # summary pass, copied
         <user guide>.lblx                   # summary pass
     miscellaneous/
       collection_miscellaneous.csv          # summary pass
       collection_miscellaneous.lblx         # summary pass
       global_bodies_index.tab               # summary pass
       global_bodies_index.lblx              # summary pass
       global_rings_index.tab                # summary pass, when an image has rings
       global_rings_index.lblx               # summary pass, when an image has rings
     spice_kernels/
       collection_spice_kernels.csv          # summary pass, copied
       collection_spice_kernels.lblx         # summary pass
       kernels.ker                           # summary pass, copied
       kernels.lblx                          # summary pass
     xml_schema/
       collection_xml_schema.csv             # summary pass, copied
       collection_xml_schema.lblx            # summary pass

Testing bundle generation
=========================

The suite asks two different questions of the bundle stage, and answers them in
two different environments.

The first is plumbing: which file goes where, which variable reaches which
template, what a render that errors leaves behind. Those tests run a duck-typed
dataset over tiny templates the test itself wrote, so every variable in play is
one the test controls and a failure names the wiring that broke.

The second is content: what a label actually says. Those tests run the
registered dataset a bundle is built with over the templates it ships, on the
products a navigation run and the backplane stage leave behind -- for the one
bundle that ships,
:class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISSSaturn`
over ``cassini_iss_saturn_1.0``. Neither environment answers the other's
question: a label rendered from a template the test wrote says whatever the test
put there, and a plumbing failure inside the shipped template set is a needle in
three hundred lines of XML.

The inputs for the second come from :mod:`tests.mini_nav_results`, a package
that builds a miniature of what a navigation run leaves on disk, one cohort per
bundle. A cohort holds a few of the bundle's images, some navigated and some
not; a real backplane FITS and its metadata document per navigated image,
written by the backplane stage's own
:func:`~spindoctor.cli.backplanes.writer.write_fits`; a real summary PNG,
because the browse label states its size and its checksum; and the index row an
enumeration hands on with each image. Each cohort is a subclass of
:class:`~tests.mini_nav_results.cohort.Cohort`, which writes all of that from
what the subclass supplies: the images, the holdings layout they sit in and the
extensions of their image and label files, how an image's camera is read from its
index row, the range each backplane plane spans,
and the registered dataset the bundle is built with. The package's ``COHORTS``
registry lists them.

The Cassini ISS Saturn cohort,
:class:`~tests.mini_nav_results.cohort_cassini.CohortCassiniISSSaturn`, is the
one that exists: three images, two navigated and one not, the two navigated ones
sharding into different bundle directories and only one of them with ring
backplanes, so a run over it exercises both layouts.

Every image of that cohort is built from its epoch and nothing else. The
spacecraft clock readings a document records and the number the image is named
for are both derived from it, so no document of the cohort can carry a reading
it counted one of the others from a different moment.

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

The self-tests every cohort is held to run over each registered cohort; what
only one bundle's cohort can state -- its image names, its index
columns, its bundle directories, its holdings layout -- is tested in a module
named for that bundle.

Nothing a cohort produces is checked in. A test asks the session-scoped fixture
:func:`mini_nav_cohorts <tests.conftest.mini_nav_cohorts>` for the cohort of the
bundle it is about, which is written into a temporary directory the first time a
test asks for it and torn down with the session; a test asserts that none of its
products reaches the working tree. To build one outside the suite, to read or to
call the bundle stage over, name the bundle and where to write its cohort, the
bundle being one of the names the package's ``COHORTS`` registry holds:

.. code-block:: bash

   PYTHONPATH=src python -m tests.mini_nav_results cohort cassini_iss_saturn /tmp/cohort

What that writes is what the bundle stage's library entry points read: the
navigation and backplane roots, and the images to pass them. It is not a
holdings tree, so ``sd_create_bundle`` cannot enumerate it -- a PDS3 selection
by volume reads that volume's index table, and the cohort writes none.

Adding a bundle's cohort is one module holding a
:class:`~tests.mini_nav_results.cohort.Cohort` subclass, named for the bundle,
and one entry in ``COHORTS``. The FITS files, the browse images and the
documents it implies exist only while a test is running, so it costs the
repository nothing.

Adding PDS4 support to a new dataset
====================================

The end-to-end checklist:

1. Override every ``pds4_*`` method on the new
   :class:`~spindoctor.dataset.dataset.DataSet` subclass. Use
   :class:`~spindoctor.dataset.dataset_pds3_cassini_iss.DataSetPDS3CassiniISS` as
   the reference implementation. The methods that absolutely must work
   are :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_template_dir`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_user_guide_file_name`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_name`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub`, the four
   ``pds4_image_name_to_*_lid[vid]`` methods,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name`
   (the inverse of ``pds4_path_stub``'s image-name transform), and
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`.
2. Drop a per-dataset template directory under
   ``src/spindoctor/cli/pds4/templates/<dataset>_<version>/`` containing the ``.lblx``
   files, the static inventory CSVs, the readme and the metakernel, and the user
   guide when it exists. Copy from ``cassini_iss_saturn_1.0/`` and adapt the field
   set.
3. Add an entry under ``pds4.<dataset_name>:`` in
   ``config_950_pds4.yaml`` that points at the new template directory and
   sets the bundle name plus any per-bundle defaults the
   ``pds4_template_variables`` hook draws from.
4. Add an integration smoke test that renders one image through
   ``sd_create_bundle`` and asserts the resulting ``data.lblx`` validates
   against the PDS4 schema.
5. Add the bundle's cohort, as `Testing bundle generation`_ describes: a
   :class:`~tests.mini_nav_results.cohort.Cohort` subclass in a module named for
   the bundle, its entry in ``COHORTS``, and a test module named for the bundle
   for what only its cohort can state.

API reference
=============

The :mod:`pds4` package has no autogenerated entry under
:doc:`/api_reference`; the module's public surface is the entry points
listed below, plus the
:class:`~spindoctor.dataset.dataset.DataSet` ``pds4_*`` extension hooks
documented above.

- :func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` — phase 1, one image.
- :func:`~spindoctor.cli.pds4.collections.generate_collection_files` — phase 2, the data
  and browse collections; and :func:`~spindoctor.cli.pds4.collections.write_collection`,
  the one writer of every generated collection's inventory and label.
- :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` — the bodies and
  rings global index tables, their labels, and the miscellaneous collection; and
  :func:`~spindoctor.cli.pds4.global_index.lay_out_table`, which lays a table out fixed
  width.
- :func:`~spindoctor.cli.pds4.bundle_products.secondary_members` — the secondary members
  the document inventory the template directory ships cites, which the miscellaneous
  inventory lists too.
- :func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` — phase 2, the
  run-level products: the readme, the static collections, the user guide and the bundle
  label; and :func:`~spindoctor.cli.pds4.bundle_products.clear_bundle_products`, which
  removes every path it can write and which the index generator calls first.
- :func:`~spindoctor.cli.pds4.labels.write_label` — the one place a label is
  written, shared by both.
- :func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic` — the one
  check both passes hold every statistic of a document to, its unit and its
  values.
- :func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits` — the headers
  and arrays of a backplane FITS, read from the source before the copy is made,
  for the data label of its copy.
- :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan` and
  :class:`~spindoctor.cli.pds4.global_index.GlobalIndexOutcome` — the range of the
  products' epochs, taken in the global index's read of the supplemental files
  and handed to the collection generator.
- :func:`~spindoctor.support.time.et_to_pds4_utc` — the PDS4 spelling of an epoch,
  beside :func:`~spindoctor.support.time.et_to_utc`, in the one conversion, and
  :func:`~spindoctor.support.time.pds4_utc_midpoint`, the midpoint of two times so
  written, which is what a data label's ``IMAGE_MID_TIME`` states.
