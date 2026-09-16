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
the miscellaneous collection, the bundle's run-level products, the check a written
bundle is held to, and the output layout.

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
   the copy is the same bytes (see `The FITS and its data objects`_). The data
   label also names the image's targets (see `Targets`_).

2. **Collections and indexes.**  After every per-image data label is in place,
   :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` reads
   every ``_supplemental.txt`` in the bundle's ``data/`` tree once, writes the
   ``global_bodies_index`` and ``global_rings_index`` tables and their labels,
   and the miscellaneous collection that lists them, into ``miscellaneous/`` (see
   `The global index and the miscellaneous collection`_), and takes the range of
   the products' exposure epochs, and the targets they name, in the same read.  Then
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
   holds its PDF, and the bundle label, which states the same range, names the same
   targets and is kept only over a bundle holding every collection it declares (see
   `The bundle's run-level products`_).

The driver runs phase 1 once per image (fan-out friendly — each image is
independent) and phase 2 once at the end (sequential — needs every per-image
label in place before it can build the inventory).

``sd_create_bundle check`` then holds the bundle the two phases wrote to PDS4, reading
its tree and the schemas its labels name, and writes nothing in the tree (see `Checking
a bundle`_).  Before phase 1, ``sd_create_bundle labels --check-only`` reports whether
each selected image has the files phase 1 reads, and writes no label, log or bundle
file.

Driver: ``sd_create_bundle``
=============================

``sd_create_bundle`` (``src/spindoctor/cli/sd_create_bundle.py``) has three
subcommands. ``sd_create_bundle labels`` runs phase 1: it takes a
``DATASET_NAME``, the selection flags from the matching
:class:`~spindoctor.dataset.dataset.DataSet` subclass (``--pds3-holdings-root``
among them, for a PDS3 dataset), the environment options (``--config-file``,
``--bundle-results-root``, ``--nav-results-root``, ``--backplane-results-root``),
the logging options and one of ``--dry-run`` and ``--check-only``, and walks every
selected image.  With ``--check-only`` it reports on each image's inputs instead of
labeling it.  ``sd_create_bundle summary`` runs phase 2 once over the bundle: it takes
a ``DATASET_NAME``, ``--config-file``, ``--bundle-results-root`` and the logging
options.  ``sd_create_bundle check`` checks a bundle the two passes wrote: it takes a
``DATASET_NAME``, ``--config-file``, ``--bundle-results-root`` and ``--schema-dir``,
and no logging options, since it writes no log.

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
:data:`~spindoctor.cli.pds4.index_columns.INDEX_VALUE_FORMATS` gives its unit, each
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
infinite), an image whose navigation document records no exposure times in its
observation block, as a navigation by an earlier version left it (see `Epochs`_),
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
volume.  So is an image whose backplane metadata names no body with geometry and holds
no ring statistic, as a star field's does (see `Targets`_): it is
skipped before any check that fails an image, since nothing would be written for it
however those came out.  An error raised while one image is processed is logged with
its traceback naming the image, counts the image against the run, and the run carries
on to the next one.

A dry run reports what it would have processed and exits 0, once the
preconditions above are met: they are checked before ``--dry-run`` is read, so a
dry run over a missing template or a populated bundle root
exits 1 naming what it found, like any other run.  Past them it writes nothing,
so it counts nothing against the run, including a batch it reports it could not
have processed.

``--check-only`` makes neither check, since it writes no label, log or bundle file,
neither needs nor creates a bundle root, and builds no logging.  For the image of each
batch the selection enumerates it prints one line saying whether the navigation
document, the summary PNG, the backplane FITS and the backplane metadata that
:func:`~spindoctor.cli.pds4.image_inputs.image_inputs` names for it exist, and whether
its navigation succeeded, from the navigation record as the labels pass reads it; then
a count, through :func:`~spindoctor.cli.pds4.image_inputs.report_image_inputs`.  A batch
of other than one image, which the labels pass fails, is one line, and every image of
it counts as incomplete: the two take that rule from one function in the driver.  It
exits 1 when any image lacks a file, its navigation did not succeed, or its batch is
one the labels pass fails.  A selection that cannot be enumerated ends the report with
the traceback it ends the labels pass with.  The file cache the two roots are read
through creates a temporary directory, which it removes.

The labels pass takes its four paths from the same function, so the report and the pass
cannot disagree about where an input is.  The navigation document's path is the
navigation records' own, :func:`~spindoctor.nav_records.document.document_path`, and
both read the document through :func:`~spindoctor.nav_records.document.read_document`.
The other three paths are stated in
:func:`~spindoctor.cli.pds4.image_inputs.image_inputs` as the navigation and the
backplane stages write them: those stages build each path where they write it and
expose no function for it, so each of the three rules is stated once by its writer and
once for the readers, a duplication older than the bundle check.

``sd_create_bundle summary`` counts the collection, index and run-level labels it
did not write, over its three generators, and exits 1 the same way.  The index
tables are written either way, and so is the inventory of a collection whose label
fails to render.  An index table no image gives a row is not written, nor its label,
and does not count: a table label states at least one record, and a bundle with no
image with ring backplanes is not a fault.  With neither table written, though, the
miscellaneous collection holds nothing of its own, so it is not written, and it counts.
The bundle label counts when it is not written: it is kept only
over a bundle holding a label for every collection it declares, and with no range
to state it is not rendered (see `The bundle's run-level products`_).  A metakernel
label that fails to render leaves the SPICE kernel collection with no member, so that
collection is not written and counts as well, and so does the bundle label.  A static
inventory whose template fails to render leaves its collection unwritten, and it counts,
and so does the bundle label, which declares that collection.  The document inventory is
the exception: the miscellaneous inventory takes its secondary members from it, in the
global index step, so a document inventory that cannot be rendered stops the summary
pass there, with exit status 1, the index tables and their labels on disk and nothing
the pass writes after them.  A readme that fails to render counts too, and so does the
bundle label, which the package's template gives the readme's creation time.  A user-guide
PDF the template directory does not hold is one warning, and does not count.  A collection whose label cannot state what PDS4 requires of it is not
written at all -- neither its inventory nor its label, and whatever an earlier run
left at either path is removed -- and counts once among the labels not written,
with an error naming the collection and each reason.  A collection label states at
least one record, so a collection with no label of its kind on disk -- no data
label, or no browse label -- is never written.  The data collection label also
states the range of its products' epochs, so a data tree in which no data label has
a supplemental file beside it writes no data collection (see `Epochs`_).  The
global index is generated first, and it refuses a
bundle with no ``data/`` directory, naming the directory, before any product of
the pass is cleared or written.  The pass also exits 1 when a supplemental file
holds a statistic no index column can -- one in a unit other than the one the
configuration gives its plane, or a minimum or maximum that is NaN or infinite --
the check the labels pass makes per image, through
:func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic`, naming the
file and the plane and saying what the file records there.  The index tables and
labels and the miscellaneous collection an earlier run wrote, its collection
inventories and labels, and its run-level products are cleared once the index
generator has found the ``data/`` directory and before the first supplemental
file is read, all of them by the index generator since it runs first, so a run
refused for want of ``data/`` leaves them as they were.  Every supplemental file
is read, and every value in
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

``sd_create_bundle check`` exits 1 when it makes any error, when the bundle's
directory under the bundle results root is not there, and when the check itself
stops, whose traceback it prints in place of a count; warnings alone leave it 0.
The check reads a local tree, so a bundle results root that is not local -- one
named ``gs://``, say -- is refused by its own name before anything is read.

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
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_version` — the bundle's
  version, which the bundle, each of its collections and each product it writes states
  as its ``version_id``, and which every LIDVID naming one of them carries.  Lookups
  consult ``config.pds4.<dataset_name>.bundle_version``, which has no default.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_information_model_version` and
  :meth:`~spindoctor.dataset.dataset.DataSet.pds4_schemas` — the PDS4 information model
  the labels are written against, and the schema of each dictionary they declare, a
  :class:`~spindoctor.dataset.dataset.Pds4Schema` by the prefix its namespace takes in a
  label: its location less the extension, and the LIDVID the XML schema collection lists
  it by.  Lookups consult ``config.pds4.<dataset_name>.information_model_version`` and
  ``config.pds4.<dataset_name>.schemas``, which have no default.
- :meth:`~spindoctor.dataset.dataset.DataSet.pds4_required_templates` — the
  file names one pass takes from that directory, the templates it renders and the
  files it copies: ``labels`` for the per-image pass and ``summary`` for the pass
  writing the collections, the index and the run-level products. Each pass checks
  them before it processes anything and refuses to run when one is not there, so
  this is where a dataset says what its template tree carries.  The user-guide PDF
  is not among them, since a bundle is written without it; its label's template is.
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
  convention ``urn:nasa:pds:<bundle>:browse:<image>``, and a LIDVID's version is
  the bundle's.
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
  per-backplane min/max/units, and so on), and the product the backplanes were
  computed from: the Cassini hook cites the calibrated image the navigation read
  as a ``Source_Product_External``, by its volume and the file specification of
  its label within that volume, since no PDS4 bundle holds calibrated Cassini ISS
  images yet; the label names it through a ``Source_Product_Internal`` once one
  does.

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
name and default template directory to its subclasses and raises
:exc:`NotImplementedError` for them, so both passes stop on it when they ask for
the template directory, as they do on ``sim``, whose hooks all raise it.  Every
registered name but ``sim`` has a ``_pds3`` alias naming the same class, which
bundles, or does not, the same way.

The ``pds4`` config block
-------------------------

Each dataset that bundles has an entry in ``config.pds4``, in a configuration file of its
own named for its mission and target --
``src/spindoctor/config_files/config_951_pds4_coiss_saturn.yaml`` for ``coiss_saturn``
-- holding its template directory, the bundle's name and
version (``bundle_name``, ``bundle_version``), and the information model version and the
dictionary schemas its labels are written against (``information_model_version``,
``schemas``).  Each is set there and nowhere else: every template is handed them (see
`The bundle's variables`_), so a new bundle version, a new name or a dictionary moving
to another version is one edit of this entry.  The dictionary schemas are the bundle's
beside its name, since each bundle's labels are checked against the dictionaries its
own templates were written for.  The name, the version, the information model version
and the schemas have no default, and the shipped configuration is held to the shipped
templates by tests rather than checked when it is loaded: it names the bundle, sets the
version, gives a schema for exactly the dictionaries the templates declare, and gives
the ``pds`` schema of the build the information model version names.  The bundle check
fetches each schema by its URL (see `Checking a bundle`_), and its tests read local
copies, so a schema named here needs a copy in
``tests/spindoctor/cli/pds4/check/schemas/``, which a test holds the shipped
configuration to.  The entry is kept in a file named for the mission, since nothing
mission-specific goes in a generically named file, and at the ``95x`` tier rather than in
the instrument's ``config_4*`` file: every navigation document records a hash of each
``config_4*`` file's bytes as the instrument's static data, and a new bundle version or
a moved schema there would read as a change of that data.  No ``9xx`` file is hashed that
way.  See :doc:`dev_guide_config_and_static_data` for the loader contract; the file is
loaded by the standard numeric-prefix order at the ``9xx`` "downstream products" tier.

The entry another dataset would need, once its templates and its PDS4 hooks exist, is
one of these, in a file of its own named as ``coiss_saturn``'s is, each with every
other key the ``coiss_saturn`` entry has -- the information model version and the
schemas of the dictionaries its templates declare:

.. code-block:: yaml

   pds4:
     gossi:
       template_dir: galileo_ssi_jupiter_1.0
       bundle_name: galileo_ssi_jupiter_backplanes_rsfrench2027
       bundle_version: '1.0'
     nhlorri:
       template_dir: newhorizons_lorri_pluto_1.0
       bundle_name: newhorizons_lorri_pluto_backplanes_rsfrench2027
       bundle_version: '1.0'
     vgiss:
       template_dir: voyager_iss_saturn_1.0
       bundle_name: voyager_iss_saturn_backplanes_rsfrench2027
       bundle_version: '1.0'

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

The bundle's variables
----------------------

Beside its own variables, every template the bundle stage renders -- each label, and
the readme and the inventories the template directory ships -- is handed the ones
:func:`~spindoctor.cli.pds4.bundle_variables.bundle_variables` gives the bundle as a
whole, from the dataset's hooks:

- ``BUNDLE_LID``, ``urn:nasa:pds:<bundle name>``, which a template extends into the LID
  of each collection and product of the bundle, as in ``$BUNDLE_LID$:browse``;
- ``BUNDLE_VERSION``, which every label states as its ``version_id`` and every LIDVID
  naming one of the bundle's own products carries;
- ``INFORMATION_MODEL_VERSION``, which every label states;
- ``PDS4_<PREFIX>_SCHEMA`` and ``PDS4_<PREFIX>_SCHEMA_XSD``, for each schema, the
  Schematron a label's ``xml-model`` instruction names and the XML schema its
  ``xsi:schemaLocation`` pairs with the namespace, ``<PREFIX>`` being the namespace's
  prefix in upper case;
- ``XML_SCHEMA_LIDVIDS``, which the XML schema inventory lists, one ``S`` line each.

No template spells the bundle's name, its version, a schema's location or the
information model version.  A template that declares a dictionary the dataset gives no
schema for does not render, so a label cannot declare a schema the XML schema
collection does not list.  A reference to a product outside the bundle -- a context
product, a document of another bundle, the calibrated image a product was computed
from -- keeps its own version.

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
     readme.txt                               # bundle-level README (rendered)
     data.lblx                                # per-image backplane data label
     browse.lblx                              # per-image browse-product label
     collection_data.lblx                     # data-collection label (CSV inventory)
     collection_browse.lblx                   # browse-collection label
     collection_context.lblx                  # context-collection label
     collection_context.csv                   # context inventory's fixed members
     collection_document.lblx                 # document-collection label
     collection_document.csv                  # document inventory (rendered)
     collection_spice_kernels.lblx            # SPICE-kernel-collection label
     collection_spice_kernels.csv             # SPICE kernel inventory (rendered)
     kernels.ker                              # metakernel (copied)
     kernels.lblx                             # metakernel label
     collection_xml_schema.lblx               # schema-collection label
     collection_xml_schema.csv                # schema inventory (rendered)
     global_bodies_index.lblx                 # bodies index label
     global_rings_index.lblx                  # rings index label
     collection_miscellaneous.lblx            # miscellaneous-collection label
     cassini-iss-saturn-backplanes-user-guide.lblx  # user-guide label
     cassini-iss-saturn-backplanes-user-guide.pdf   # the user-guide PDF, when it exists

The labels pass renders ``data.lblx`` and ``browse.lblx`` for each image.  The
summary pass renders every other label, the readme and the inventories, and copies
the files marked copied, the user-guide PDF among them when the directory holds it;
the directory ships no user-guide PDF.

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

Every exposure time a label or an index table states (``start_date_time``,
``stop_date_time``, and the collection's range) comes from the exposure the
navigation document's ``observation`` block records: ``start_time_et`` and
``end_time_et``, in TDB seconds past J2000, which the instrument host publishes for
every image whose navigation ran to a result, whether or not it solved a pointing.
They are turned into UTC by one rule, in
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
half up.  The navigation records a success with no pointing when the attitude cannot
be computed or the instrument has no SPICE camera frame mapped, and the document's
``observation`` block records the exposure all the same, so such an image is bundled
like any other.  A navigation by an earlier version recorded the exposure times only
beside a solved pointing, under ``navigation_result.times``, and none in the
``observation`` block.
:func:`~spindoctor.cli.pds4.bundle_data.generate_bundle_data_files` fails an image
whose block holds no ``start_time_et`` before anything is written for it, the log
naming the image, and the image is bundled once it is navigated again; the times are
not taken from ``navigation_result.times`` instead, so every time the bundle states
comes from the one block.  The epochs are read as recorded, and the supplemental file
carries the whole navigation document, so the summary pass reads the same block.  It
checks nothing: a supplemental file the labels pass wrote always holds the times, and
a bundle is written into an empty directory.

The data collection label states the range of the products' epochs: the least
start and the greatest stop over the images the data collection holds, as their
supplemental files record them, written to whole seconds with the start rounded
down and the stop up.  A supplemental file with no data label beside it is not in
the range, as it gives the index no row.  The range is taken in the one read
of the supplemental files the summary pass makes -- the global
index's -- by an :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`, and
:func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` returns it in
its :class:`~spindoctor.cli.pds4.global_index.GlobalIndexOutcome`.  That is why the
summary pass runs the index first and hands the range to
:func:`~spindoctor.cli.pds4.collections.generate_collection_files`.  A scan that
took in no member's supplemental file yields no range, and the data collection is
then not written, neither its inventory nor its label, rather than labeled with
empty dates (see `Exit status`_).

Targets
=======

A label names a target by its PDS4 context product.
:func:`~spindoctor.cli.pds4.targets.target_table` reads the configuration's
``backplanes.target_lids`` into :class:`~spindoctor.cli.pds4.targets.Pds4Target` entries,
each the context product's LID and version and the name and type the product gives the
target, keyed by the name the backplane metadata gives the target: a body by the name
:func:`~spindoctor.cli.backplanes.backplanes_bodies.backplane_body_names` looks for it
under, and the rings by the target
:func:`~spindoctor.cli.backplanes.backplanes_rings.ring_target` computes their backplanes
for.  An image's targets, as :func:`~spindoctor.cli.pds4.targets.image_targets` finds
them, are every body its backplane metadata names that has geometry -- a statistic at
least, as :func:`~spindoctor.cli.pds4.targets.has_geometry` decides -- and the ring
target when the metadata holds a ring statistic, in the table's order.  A body the
image's inventory found that shows at no pixel is named with no statistic, and is no
target.  A name the
table has no entry for raises :exc:`KeyError`, naming it.  Nothing checks the table when
a run starts; a test over the shipped configuration holds it to the stage's own body
list and ring target.

``data.lblx`` names an image's targets, handed to it as ``TARGETS``, one
``Target_Identification`` each, with a ``data_to_target`` reference.  PDS4 requires a
data label to name one at least.  An image whose backplane metadata names none -- no
body, and no ring statistic -- has no geometry for a data label to describe, so the
labels pass skips it before anything is written for it, logging why, and
:func:`~spindoctor.cli.pds4.targets.covers_a_target` decides it without reading a
ring target, which backplanes an earlier version generated do not record.  The summary
pass takes the
targets of the data collection's members in its one read of the supplemental files, by a
:class:`~spindoctor.cli.pds4.targets.TargetScan`, as it takes the range of their epochs,
and :func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` returns them in
its :class:`~spindoctor.cli.pds4.global_index.GlobalIndexOutcome`; the driver hands them
to the collection generator and to the run-level products.  Backplane metadata that
records ring statistics and no ring target, as backplanes an earlier version generated
does, raises a bare ``KeyError: 'target'`` wherever the targets are read: the labels
pass raises it for such an image before anything is written for it, which counts the
image failed, and a summary pass over data members an earlier labels pass wrote from
such metadata raises it from its read.  Nothing checks for it: backplanes are
regenerated before a bundle is built for delivery, and a bundle is written into an
empty directory, the labels pass first, so the members of the pass are the labels
pass's own.  The data collection label
names them with ``collection_to_target``, as the SPICE kernel collection label does, the
bundle label with ``bundle_to_target`` and the metakernel label with ``data_to_target``,
the values the Schematron allows under each kind of product, and the context inventory
lists each, after the members the template directory ships, as ``S,<lidvid>``.  The
document and miscellaneous inventories list no target, since no label of their
collections names one.

A data label states no ring geometry: of an image's rings it names the ring target
alone.  The rings index states the image's ring ranges instead: the least and the
greatest value of each ring plane, the ring longitude's range wrapped at zero, and the
incidence angle of sunlight on the ring plane, the last two as the backplane metadata
records them beside the ring statistics (see :doc:`dev_guide_backplanes` and
`The global index and the miscellaneous collection`_).  The labels pass reads neither.
No label declares the rings dictionary, although the rings index's ``rings:`` columns
take its attributes' names.

The data, data collection and bundle labels each declare one ``Science_Facets``,
``Visible`` and ``Ring-Moon Systems``, fixed in their templates.

The global index and the miscellaneous collection
=================================================

:func:`~spindoctor.cli.pds4.global_index.generate_global_index_files` writes two tables
into the bundle's ``miscellaneous/`` directory, each with its label beside it, and then
the collection that holds them:

- ``global_bodies_index.tab``, one row for each body with geometry in each image the data
  collection holds;
- ``global_rings_index.tab``, one row for each such image with ring backplanes;
- ``collection_miscellaneous.csv`` and ``collection_miscellaneous.lblx``.

Each product's LID is built from the bundle's name, as the bundle's own is, by
:func:`~spindoctor.cli.pds4.global_index.index_lid`:
``urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index`` and
``...:global_rings_index``, at the bundle's version,
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_version`.

**Which images get a row.**  The rows are exactly the images the data inventory lists:
both take the data labels in the data tree from
:func:`~spindoctor.cli.pds4.collections.data_products`, so the tables and the inventory
cannot disagree about what the bundle holds.  The values come from each image's
supplemental file.  A supplemental file with no data label beside it adds no row, and
its epochs are not in the range the data collection label states; its statistics are
still checked, as every supplemental file's are.

A body gets a row of the bodies table only when it has geometry, as
:func:`~spindoctor.cli.pds4.targets.has_geometry` decides: the backplane writer records a
body in the backplane document from the image's inventory alone when none of its planes
has a value there, and a row for it would measure nothing.  A body with statistics for
some planes and not others has its row, the masked value in the columns of the planes
it has none for.  The rings are recorded only with their statistics, so an image whose
rings have none gets no row of the rings table.

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
Both then give ``pds:start_date_time`` and ``pds:stop_date_time``
(``ASCII_Date_Time_YMD_UTC``), the image's exposure start and stop, which
:func:`~spindoctor.cli.pds4.epochs.exposure_times` writes from the epochs the
supplemental file records, to the millisecond, as the data label states them.
Then each configured plane gives its table two columns, the least and the greatest value
its statistic spans, as :class:`~spindoctor.cli.pds4.index_columns.IndexColumn` entries
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
format is the one :data:`~spindoctor.cli.pds4.index_columns.INDEX_VALUE_FORMATS` gives that
unit.

**Wrapped ranges and the incidence angle.**  A plane's ``index`` block may also give
``wrapped_minimum`` and ``wrapped_maximum``, two more columns after the plane's pair:
where the range its statistic records wrapped at zero, as ``wrapped_min`` and
``wrapped_max``, starts and where it ends, the start the greater where the range crosses
zero.  The ring longitude's block gives them as ``rings:minimum_inertial_ring_longitude``
and ``rings:maximum_inertial_ring_longitude``, the rings dictionary's names for a ring
longitude range wrapped at the prime meridian, while its plain pair keeps names of its
own.  The body longitude's block gives them as ``minimum_wrapped_body_longitude`` and
``maximum_wrapped_body_longitude``, names of their own: geom's ``minimum_longitude`` and
``maximum_longitude`` define a range wrapped at the prime meridian, but in planetocentric
longitude, which the IAU convention measures positive east, where the body longitude is
measured westward.  The rings table ends with the three columns ``backplanes.ring_incidence_angle``
describes, ``rings:minimum_incidence_angle``, ``rings:maximum_incidence_angle`` and
``rings:mean_incidence_angle``: the ``min``, ``max`` and ``mean`` of the rings block's
``incidence_angle``, in the unit that entry's ``units`` gives.  The ``rings:`` names are
the rings dictionary's attributes, borrowed as column names; no label declares the
dictionary.  :mod:`spindoctor.cli.pds4.index_columns` builds every statistic column,
:class:`~spindoctor.cli.pds4.index_columns.IndexPlane` a plane's and
:class:`~spindoctor.cli.pds4.index_columns.RingIncidence` the incidence angle's, and
:func:`~spindoctor.cli.pds4.index_columns.statistic_index_columns` gives all of them,
which the bundle check holds each index label's fields to.

**Missing values.**  Where an image has no statistic for a plane, both of its cells hold
the configured masked value, ``backplanes.masked_value``, written in the column's format:
``-999.000`` in a ``deg`` column, ``-999.0`` in ``km``, ``-999.00000000`` in
``deg/pixel`` and ``-999.00`` in ``km/pixel``.  Every statistic field of a label declares
that text as the ``missing_constant`` of a ``Special_Constants`` block, in the spelling
its cells have, so a reader comparing a cell's text with the declared constant finds it,
and every value of a column, a missing one included, is written in one form.  A wrapped
value, or a member of the incidence angle, that the supplemental file does not record --
backplanes an earlier version generated record no wrapped range, and the angle at the
ring center alone -- is written the same way, cell by cell, and nothing is failed for it.

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
listed; the run says so at info level and counts nothing against the run.  When neither
table is written, the miscellaneous collection has nothing of its own to hold and is not
written either, which does count (below).

**The collection.**  After the tables, the generator writes the miscellaneous collection,
as the data inventory is written after its labels, through
:func:`~spindoctor.cli.pds4.collections.write_collection`, the one writer of the
generated collections: a ``P`` line for each index product whose label is on disk, then
an ``S`` line for each secondary member of the document inventory the template directory
ships, as that inventory renders with the bundle's variables, taken through
:func:`~spindoctor.cli.pds4.bundle_products.secondary_members`, so that the two
inventories cite the same context products and ISS data user guide.  The
collection label, ``collection_miscellaneous.lblx``, has ``collection_type``
``Miscellaneous``.  The collection takes its members from the index labels, as the data
and browse collections take theirs from labels of their own kind, so with neither index
product labeled -- no image gives either table a row, or neither label renders -- it is
not written, whatever the document inventory cites, and counts once as a label not
written; the bundle label, which declares it, is then not written either (see
`Exit status`_).

The index products and the collection are cleared with the rest of the summary pass's
products before any supplemental file is read (see `Exit status`_).

The bundle's run-level products
===============================

:func:`~spindoctor.cli.pds4.bundle_products.generate_bundle_products` writes the
products of the bundle as a whole, last in the summary pass, into the bundle's own
directory from the dataset's template directory.  Some are rendered from a template and
some are copied as they are:

- ``readme.txt`` is rendered at the bundle's root from the template of that name.
- The context, document, SPICE kernel and XML schema collections are each an inventory
  the template directory ships, ``collection_<name>.csv``, rendered into the
  collection's directory from the template of that name, and a label,
  ``collection_<name>.lblx``, rendered beside it and handed the inventory's path as
  ``COLLECTION_<NAME>_CSV_PATH``.  The XML schema inventory lists
  ``XML_SCHEMA_LIDVIDS`` (see `The bundle's variables`_).  The context inventory is written
  with every target the data labels name after the members the template directory
  ships (see `Targets`_).
- The metakernel ``kernels.ker`` is copied into ``spice_kernels/`` and its label
  ``kernels.lblx`` rendered beside it, handed the copy's path as ``METAKERNEL_PATH`` and
  the targets as ``TARGETS``.
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
- ``bundle.lblx`` is rendered last, at the bundle's root, handed the readme's path as
  ``README_PATH``, the range of the products' epochs the data collection label states
  (see `Epochs`_), and the targets as ``TARGETS``.

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
An inventory whose template does not render leaves its collection unwritten the same
way, its label not rendered over it, and the collection counts once, and the bundle
label, which declares the collection, goes too.  The document inventory is the
exception: :func:`~spindoctor.cli.pds4.bundle_products.secondary_members` renders it for
the miscellaneous inventory in the global index step, and raises when it cannot, so a
document inventory that cannot be rendered stops the summary pass before this generator
runs (see `Exit status`_).  A readme that does not render counts
too, and takes the bundle label with it: the bundle label the package ships states the
readme's creation time, which a readme not written does not have.

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
generator itself removes three things: each label, inventory and readme, just before it
renders it; the label of a collection whose inventory did not render, and the inventory
and label of a collection left with no member; and the bundle label, when it has no
range to state or declares a collection the bundle lacks.  Everything else an
earlier run wrote is cleared by
:func:`~spindoctor.cli.pds4.bundle_products.clear_bundle_products`, which removes every
path the generator can write, and ``document/user_guide/`` when that leaves the
directory empty in a bundle on the local file system; a remote store holds no directory
apart from the files in it.  The global index generator, which the summary pass runs
first, calls it with its own clearing, once it has found the bundle's data directory and
before it reads any supplemental file, so a run-level product on disk after a summary
pass that got past that check is one that pass wrote.

Checking a bundle
=================

``sd_create_bundle check`` holds a bundle the two passes wrote to PDS4, through
:func:`~spindoctor.cli.pds4.check.bundle.check_bundle`.  It reads the bundle's tree
and the schemas its labels name, fetched by URL or read from a directory, and writes
nothing in the tree.  Each way the tree departs
from PDS4 is a :class:`~spindoctor.cli.pds4.check.findings.Finding`, which names the
file, whether it is an error or a warning, the check that found it, where in the file,
and what is wrong; an XML schema error also names its kind, the class of the xmlschema
validator that failed, which does not change with xmlschema's wording.  The program
prints each finding as one line, and then the number of errors and of warnings, and
exits 1 on an error.  A warning is what the PDS ``validate`` tool
reports as a warning too: an unresolved reference to a product of the bundle
(``reference_not_found``), a product no inventory lists (``unreferenced_member``), and
a Schematron assert or report whose ``role``, or whose rule's, marks it a warning.  The
check reads labels with ``lxml``, validates them with ``xmlschema`` and evaluates the
Schematron with ``elementpath``'s XPath 2.0 engine, three runtime dependencies of the
package.

Every file under the bundle's directory whose name ends in ``.lblx`` is a label.  A
label that is not well-formed XML is one finding and is checked no further.  Every other
label is checked on its own, by :func:`~spindoctor.cli.pds4.check.bundle.check_label`:

- **Against its XML schemas** (:mod:`~spindoctor.cli.pds4.check.schemas`).  Every URL
  the check reads -- each a label's ``xsi:schemaLocation`` pairs with a namespace, each
  ``xml-model`` ``href``, and each ``schemaLocation`` a schema's ``xs:import`` gives --
  is resolved one way, by :class:`~spindoctor.cli.pds4.check.schemas.SchemaSource`.  By
  default it is fetched through ``filecache`` into the cache
  ``_filecache_spindoctor_pds4_schemas``, under ``$FILECACHE_CACHE_ROOT`` when that is
  set and otherwise in the user's own ``$XDG_CACHE_HOME`` or ``~/.cache``, never in a
  directory other users share, which keeps each download for later checks; with
  ``--schema-dir`` it is the file of the URL's name in that directory, and nothing is
  fetched.  ``xmlschema`` is allowed only local files and reads every URL through that
  rule, so an import resolves to the schema at its own URL, as ``validate`` resolves it,
  and no namespace is named in code.  ``xmlschema`` reads a namespace once in a set,
  though: the Cassini data labels declare the geometry dictionary's build ``19B0``, and
  that build serves the Cassini schema's import of ``19A0``, which is not read, where
  ``validate`` reads both, to the same verdicts.  A URL a label names that cannot be resolved is a finding naming it.  Every
  warning ``xmlschema`` raises while it builds a set of schemas is a finding, so an
  import that cannot be resolved cannot pass silently.  A set that cannot be built at all -- a URL paired with a namespace its file
  does not define, say -- is one finding, and the label is checked without it.  Each
  distinct set a label declares is built once.
- **Against its Schematron rules** (:mod:`~spindoctor.cli.pds4.check.schematron`), each
  named by the ``href`` of an ``xml-model`` instruction and resolved by the same rule.
  A rule is matched as the ISO Schematron skeleton's XSLT matches it: a
  node matches when it is in ``//(context)`` evaluated from the document node; only the
  first rule of a pattern a node matches fires for it; the schema's and each pattern's
  variables are evaluated at the document node, and a rule's at the node it matched.  A
  context that is a union of path expressions joined by ``|``, as every context of the
  labels' Schematron is, is selected as ``//`` before each branch, which selects the
  same nodes without evaluating the context again at every node of the label; any other
  context, one holding the ``union``, ``intersect`` or ``except`` keyword or a comment,
  is selected as ``//(context)`` itself.  An assert whose test is false and a report
  whose test is true are each a finding, with the rule's message: a warning when it, or
  else its rule, carries a ``role`` of ``warning`` or ``warn`` in any case, and an error
  otherwise.  Strings are compared by the Unicode code point collation, XPath's
  default, whatever locale the process runs under.  The rules cannot be run by ``lxml``'s
  ISO Schematron, which does
  not take their XSLT 2.0 query language, nor by an evaluator that matches a rule from a
  node's parent, which never fires a rule whose context has several steps, such as the
  one on ``pds:SPICE_Kernel/pds:kernel_type``.
- **Its tables, read through the label alone** (:mod:`~spindoctor.cli.pds4.check.tables`),
  since neither schema reads a table.  A ``Table_Character``, a ``Table_Delimited`` or an
  ``Inventory`` is read, with the ``Header`` objects beside it: the objects of the file
  area tile the file, a ``Header`` ending where its ``object_length`` says and a table
  holding its ``records``, each ending in its record delimiter byte for byte, so that a
  carriage return the label does not declare is found; ``fields`` and ``groups`` count
  what the record describes, and in a record holding no group each ``field_number`` is
  its field's position; each field lies within the record and overlaps no other; each
  value is valid for its
  ``data_type``, the simple type of that name in the common dictionary the label
  declares; a delimited value keeps to its ``maximum_field_length``; and a cell equal to
  its field's missing constant as a number is spelled as the constant is.  The fields of
  a group, a ``Table_Binary``, and a file area that also holds an object of another
  class are left to the XML schema and to ``validate``.
- **A global index table's layout** (:mod:`~spindoctor.cli.pds4.check.index_tables`).
  PDS4 constrains neither what a header holds nor the bytes between two fields, but the
  layout :mod:`~spindoctor.cli.pds4.global_index` writes, the same for every dataset's
  bundle, does: the header line names the label's fields, in order, separated by commas;
  a comma alone lies between two fields, and the last ends where the record delimiter
  begins; and in every record the byte after each field but the last is a comma.
- **Its statistic columns** (:mod:`~spindoctor.cli.pds4.check.statistic_columns`).  A
  field a configured plane's ``index`` block names states the unit the plane's statistic
  is in, as :func:`~spindoctor.cli.backplanes.statistics.statistics_units` gives it, and
  declares as its missing constant the masked value in that unit's format.  This is the
  part of the check that reads the configuration.  The two helpers are the summary
  pass's own, so that each rule is stated once: the summary pass's tests pin them, and a
  defect in one would pass this check.

Then the tree as a whole (:mod:`~spindoctor.cli.pds4.check.integrity`): every
``file_name`` names a file beside its label, by its name alone and once, and the
``file_size`` and ``md5_checksum`` beside it are that file's; every file is a label or
is named by exactly one label; no label holds a ``[[[`` marker; no element is empty
unless its ``xsi:nil`` is true; no two labels declare one logical identifier; and every
``lid_reference`` and ``lidvid_reference`` to a product of the bundle itself -- one
whose logical identifier extends the bundle label's -- names a product a label of the
tree declares, and a ``lidvid_reference`` that product's version, one that does not
being a warning.  Each collection's inventory
(:mod:`~spindoctor.cli.pds4.check.inventories`) lists as a primary member only products
the tree holds, at the versions it holds, and each product in the collection's
directory once, one listed twice being an error and one not listed a warning.  Each
global index table's records (:mod:`~spindoctor.cli.pds4.check.index_tables`) name
products the tree holds, with each product's label as their ``file_spec`` and every
other column named for a PDS4 attribute, the start and stop times, as that label states
it; and each product whose label names a supplemental file has the records the file
calls for.  A tree with no bundle label at its top is a finding of its own.

The NASA PDS ``validate`` tool is the authority for a bundle delivered to the node, and
checks three things the check does not: where each data object of a FITS file begins,
the user guide's PDF, which it reads with VeraPDF, and each reference a label makes to a
context product, against the context products registered with the PDS.  Neither checks
the versions an inventory's secondary members give products outside the bundle, nor
that what a label states is right rather than allowed: a unit the vocabulary holds but
the value is not in, a target the product has that the label leaves out, or the lengths
a label gives a FITS array's axes.  ``validate`` is a Java program run by hand; the
check is what the suite runs (see `Testing bundle generation`_).

A dictionary added to a dataset's ``schemas``, or moved to another version, has its XML
schema and its Schematron, and every schema those import, copied byte for byte from
their URLs into ``tests/spindoctor/cli/pds4/check/schemas/``, where the check's tests
read them without the network: two tests hold every dataset's configured schemas, and
every import of a copy, to having a copy there.

Output layout
=============

The two passes write this tree:

::

   <bundle_results_root>/<bundle_name>/
     bundle.lblx                               # summary pass
     readme.txt                                # summary pass
     browse/
       collection_browse.csv                   # summary pass
       collection_browse.lblx                  # summary pass
       <path stub>/<image>_summary.png         # labels pass, copied from nav_results_root
       <path stub>/<image>_summary.lblx        # labels pass
     context/
       collection_context.csv                  # summary pass
       collection_context.lblx                 # summary pass
     data/
       collection_data.csv                     # summary pass
       collection_data.lblx                    # summary pass
       <path stub>/<image>_backplanes.fits     # labels pass, copied from backplane_results_root
       <path stub>/<image>_backplanes.lblx     # labels pass
       <path stub>/<image>_supplemental.txt    # labels pass
     document/
       collection_document.csv                 # summary pass
       collection_document.lblx                # summary pass
       user_guide/                             # only when the template directory holds it
         <user guide>.pdf                      # summary pass, copied
         <user guide>.lblx                     # summary pass
     miscellaneous/                            # only when an index table has a row
       collection_miscellaneous.csv            # summary pass
       collection_miscellaneous.lblx           # summary pass
       global_bodies_index.tab                 # summary pass, when an image shows a body
       global_bodies_index.lblx                # summary pass, when an image shows a body
       global_rings_index.tab                  # summary pass, when an image has rings
       global_rings_index.lblx                 # summary pass, when an image has rings
     spice_kernels/
       collection_spice_kernels.csv            # summary pass
       collection_spice_kernels.lblx           # summary pass
       kernels.ker                             # summary pass, copied
       kernels.lblx                            # summary pass
     xml_schema/
       collection_xml_schema.csv               # summary pass
       collection_xml_schema.lblx              # summary pass

``<path stub>/<image>`` is what
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub` gives an image, its
directory the one
:meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_path_for_image` gives: for
``coiss_saturn``, ``1454xxxxxx/145482xxxx/1454820509n``.  ``<user guide>`` is the stem
of :meth:`~spindoctor.dataset.dataset.DataSet.pds4_user_guide_file_name`:
``cassini-iss-saturn-backplanes-user-guide`` for ``coiss_saturn``.

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

The bundle check (see `Checking a bundle`_) is gated by a test in the default suite, so
it runs wherever the suite does: in ``scripts/run-all-checks.sh`` and in CI.
``tests/spindoctor/cli/pds4/check/test_check_cassini_iss_saturn.py`` builds the Cassini
ISS Saturn cohort's bundle twice, plain and from a copy of the template directory holding
a stand-in user guide, runs
:func:`~spindoctor.cli.pds4.check.bundle.check_bundle` over each, and holds the findings,
each by its file, check, location and message -- or, for an XML schema error, its kind,
since xmlschema's wording is not the check's -- to exactly what is known of the bundle,
errors and warnings apart.  The errors are the ``TODO DOI`` placeholder of the bundle
label and the two of the guide's label, and each data label's empty
``cassini:ISS_Specific_Attributes``; the warnings, in the plain build alone, are each
reference to the user guide, which that bundle does not hold.  A finding outside that
list fails the test, and so does a known one the check stops making.  The gate reads
the schemas from the local copies the tests keep, so the suite needs no network, and a
test marked ``integration`` has the check fetch them by their URLs and holds it to the
same findings.  Each part of the check is also held to a
control for each condition it checks -- a copy of that bundle broken in one place -- in a
test module named for the bundle, and the Schematron evaluator's semantics to a
Schematron and a label written by the test itself.

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
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_bundle_version`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_information_model_version`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_schemas`,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_path_stub`, the four
   ``pds4_image_name_to_*_lid[vid]`` methods,
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_lid_part_to_image_name`
   (the inverse of ``pds4_path_stub``'s image-name transform), and
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables`.
2. Drop a per-dataset template directory under
   ``src/spindoctor/cli/pds4/templates/<dataset>_<version>/`` containing the ``.lblx``
   files, the static inventory CSVs, the readme and the metakernel, and the user
   guide when it exists. Copy from ``cassini_iss_saturn_1.0/`` and adapt the field
   set, naming the bundle, its version and each schema through the bundle's variables
   (see `The bundle's variables`_) rather than spelling them.
3. Add an entry under ``pds4.<dataset_name>:`` in a configuration file of its own,
   named for its mission and target as ``config_951_pds4_coiss_saturn.yaml`` is (see
   the ``pds4`` config block above), that points at the new template directory and
   sets the bundle's name and version, the information model version, and the schema
   of each dictionary the new templates declare.
4. Copy the XML schema and the Schematron of every dictionary the new templates
   declare, and every schema those import, byte for byte from their URLs into
   ``tests/spindoctor/cli/pds4/check/schemas/`` (see `Checking a bundle`_).
5. Add the bundle's cohort, as `Testing bundle generation`_ describes: a
   :class:`~tests.mini_nav_results.cohort.Cohort` subclass in a module named for
   the bundle, its entry in ``COHORTS``, and a test module named for the bundle
   for what only its cohort can state, among it the findings
   :func:`~spindoctor.cli.pds4.check.bundle.check_bundle` makes over the cohort's
   bundle, as the Cassini ISS Saturn gate holds them.

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
- :func:`~spindoctor.cli.pds4.check.bundle.check_bundle` — the bundle check over a
  written tree, and :func:`~spindoctor.cli.pds4.check.bundle.check_label`, the part of
  it one label is held to on its own; each returns
  :class:`~spindoctor.cli.pds4.check.findings.Finding` objects.
- :func:`~spindoctor.cli.pds4.image_inputs.image_inputs` — the four files the labels
  pass reads for an image, the navigation document's path taken from the navigation
  records' :func:`~spindoctor.nav_records.document.document_path`, and
  :func:`~spindoctor.cli.pds4.image_inputs.report_image_inputs`, what ``--check-only``
  reports of them.
- :func:`~spindoctor.cli.pds4.bundle_variables.bundle_variables` — the variables every
  template of a bundle is handed: its LID and version, the information model version and
  each dictionary's schema.
- :func:`~spindoctor.cli.pds4.statistic_checks.unindexable_statistic` — the one
  check both passes hold every statistic of a document to, its unit and its
  values.
- :func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits` — the headers
  and arrays of a backplane FITS, read from the source before the copy is made,
  for the data label of its copy.
- :class:`~spindoctor.cli.pds4.epochs.EpochRangeScan`,
  :class:`~spindoctor.cli.pds4.targets.TargetScan` and
  :class:`~spindoctor.cli.pds4.global_index.GlobalIndexOutcome` — the range of the
  products' epochs and the targets they name, taken in the global index's read of the
  supplemental files and handed to the collection generator and the run-level products.
- :func:`~spindoctor.cli.pds4.targets.target_table` and
  :func:`~spindoctor.cli.pds4.targets.image_targets` — the configuration's targets
  table, as :class:`~spindoctor.cli.pds4.targets.Pds4Target` entries, and the targets one
  image's backplane metadata names.
- :func:`~spindoctor.cli.pds4.index_columns.statistic_index_columns` — every statistic
  column of the global index tables, as the summary pass writes them and the bundle
  check holds their labels to them.
- :func:`~spindoctor.cli.pds4.epochs.exposure_times` — an image's exposure start and
  stop as the index tables write them, to the millisecond, as its data label does.
- :func:`~spindoctor.support.time.et_to_pds4_utc` — the PDS4 spelling of an epoch,
  beside :func:`~spindoctor.support.time.et_to_utc`, in the one conversion, and
  :func:`~spindoctor.support.time.pds4_utc_midpoint`, the midpoint of two times so
  written, which is what a data label's ``IMAGE_MID_TIME`` states.
