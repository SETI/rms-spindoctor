======================
PDS4 Bundle Generation
======================

Overview
========

The PDS4 bundle generation system creates PDS4-compliant bundles from navigation and
backplane results. It generates PDS4 label files, supplemental metadata files, browse
products, the collection and index files that organize the data products, and the files
that describe the bundle as a whole, into a complete PDS4 bundle structure.

The bundle generation process consists of two main passes:

1. **Labels Pass**: Processes individual images to generate PDS4 labels, supplemental
   files, and browse products for each image.

2. **Summary Pass**: Generates the collection files and global index files that aggregate
   information across all processed images, and the files that describe the bundle as a
   whole: its label, its readme, its metakernel and its user guide.

A third command, ``sd_create_bundle check``, checks a bundle the two passes wrote (see
`Check Pass`_), and the labels pass's ``--check-only`` option reports, before anything
is written, whether each selected image has what the labels pass needs (see `Checking
the Inputs`_).

Purpose
-------

PDS4 bundle generation serves to:

1. Package navigation and backplane results in PDS4-compliant format
2. Create structured directory hierarchies matching PDS4 standards
3. Generate XML label files with complete metadata
4. Produce browse products (summary images) for quick visualization
5. Create collection and index files for bundle-level organization

Bundle Structure
================

The two passes write this directory structure:

.. code-block:: text

   <bundle_name>/
   ├── bundle.lblx
   ├── readme.txt
   ├── browse/
   │   ├── collection_browse.csv
   │   ├── collection_browse.lblx
   │   ├── <path stub>/<image>_summary.png
   │   └── <path stub>/<image>_summary.lblx
   ├── context/
   │   ├── collection_context.csv
   │   └── collection_context.lblx
   ├── data/
   │   ├── collection_data.csv
   │   ├── collection_data.lblx
   │   ├── <path stub>/<image>_backplanes.fits
   │   ├── <path stub>/<image>_backplanes.lblx
   │   └── <path stub>/<image>_supplemental.txt
   ├── document/
   │   ├── collection_document.csv
   │   ├── collection_document.lblx
   │   └── user_guide/
   │       ├── <user guide>.pdf
   │       └── <user guide>.lblx
   ├── miscellaneous/
   │   ├── collection_miscellaneous.csv
   │   ├── collection_miscellaneous.lblx
   │   ├── global_bodies_index.tab
   │   ├── global_bodies_index.lblx
   │   ├── global_rings_index.tab
   │   └── global_rings_index.lblx
   ├── spice_kernels/
   │   ├── collection_spice_kernels.csv
   │   ├── collection_spice_kernels.lblx
   │   ├── kernels.ker
   │   └── kernels.lblx
   └── xml_schema/
       ├── collection_xml_schema.csv
       └── collection_xml_schema.lblx

The user guide and its label are in ``document/user_guide/`` only when the dataset's
template directory holds the user-guide PDF (see `Templates`_). The rings index
and its label are in ``miscellaneous/`` only when some image in the bundle has
ring backplanes (see `Global Index Tables`_).

``<path stub>/<image>`` places each image in ``data/`` and ``browse/`` by a rule the
dataset derives from the image's name: for ``coiss_saturn``, image N1454820509 is at
``1454xxxxxx/145482xxxx/1454820509n``. ``<user guide>`` is the name of the dataset's
user guide: ``cassini-iss-saturn-backplanes-user-guide`` for ``coiss_saturn``.

Command-Line Interfaces
=======================

Two main programs support bundle generation:

* ``sd_create_bundle`` (local/CLI) — supports both labels and summary passes
* ``sd_create_bundle_cloud_tasks`` (Cloud Tasks) — parallel processing for labels pass

Labels Pass
-----------

The labels pass processes individual images to generate per-image PDS4 products.

The bundle's own directory -- ``<bundle results root>/<bundle name>/`` -- must be
empty or absent when the pass starts, so that a bundle is the product of one run
rather than a mixture of two. ``sd_create_bundle labels`` checks this and, if it
finds anything there, writes nothing and exits 1, ``--dry-run`` included. It
will not clear the directory for you: to run again after a partial failure,
clear it yourself or name a different bundle results root. The queue-driven
variant below cannot make the check -- each of its workers holds one image, not
the run -- so a queue-driven bundle is yours to start from an empty root.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   sd_create_bundle labels DATASET_NAME [options]

Where ``DATASET_NAME`` names a dataset that can be bundled (see `Supported Datasets`_).

Command-Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

Environment options:

* ``--config-file PATH`` (repeatable): one or more configuration file paths to override
  defaults.
* ``--bundle-results-root PATH``: root directory where bundle results will be written,
  overriding both the ``NAV_BUNDLE_RESULTS_ROOT`` environment variable and any corresponding
  configuration setting.

Navigation and backplane options:

* ``--nav-results-root PATH``: root directory containing navigation metadata JSON files
  (``*_metadata.json``).
* ``--backplane-results-root PATH``: root directory containing backplane FITS files and
  metadata (``*_backplanes.fits`` and ``*_backplane_metadata.json``).

Output options:

* ``--dry-run``: print the images that would be processed without generating bundle files.
* ``--check-only``: generate nothing, and report instead whether each selected image has
  what the labels pass needs (see `Checking the Inputs`_). It cannot be given with
  ``--dry-run``.

Dataset selection options are the same as in the navigation and backplane drivers (see
:doc:`user_guide_navigation`).

Examples
^^^^^^^^

Process a single Cassini image to generate bundle files:

.. code-block:: bash

   sd_create_bundle labels coiss_saturn N1234567890 \
     --nav-results-root /data/nav/results \
     --backplane-results-root /data/nav/backplanes \
     --bundle-results-root /data/nav/bundle

Process all images in a volume range:

.. code-block:: bash

   sd_create_bundle labels coiss_saturn \
     --volumes COISS_2001 --first-image-num 1454000000 --last-image-num 1454999999 \
     --nav-results-root /data/nav/results \
     --backplane-results-root /data/nav/backplanes \
     --bundle-results-root /data/nav/bundle

Checking the Inputs
^^^^^^^^^^^^^^^^^^^

With ``--check-only``, the labels pass writes no labels, logs or bundle files, and
reports instead, for each selected image, whether the four files it would read are there
-- the navigation metadata file and the summary PNG under the navigation results root,
the backplane FITS file and the backplane metadata file under the backplane results root
-- and whether the image's navigation succeeded. An image with all four whose navigation
succeeded is complete. The labels pass takes the selected images one at a time. Should
the selection hand it a group instead -- an empty one, or several images at once -- the
pass cannot label the group: the report says so on one line, and counts each image of
the group as incomplete. Use it to choose the images of a bundle before generating it: the
report needs no bundle results root and creates none. The only thing it creates is a
temporary directory for reading the two roots, which it removes before it ends.

It prints one line for each image, or for each group the labels pass cannot label, then a
count. It exits 1 if any selected image is incomplete or the selection holds such a
group, even an empty one, which leaves no image incomplete:

.. code-block:: bash

   sd_create_bundle labels coiss_saturn --volumes COISS_2001 --check-only \
     --nav-results-root /data/nav/results \
     --backplane-results-root /data/nav/backplanes

.. code-block:: text

   COISS_2001/data/1454725799_1455008789/N1454820509_1_CALIB: navigation document present, summary PNG present, backplane FITS present, backplane metadata present, navigation succeeded: complete
   COISS_2001/data/1454725799_1455008789/N1454821332_1_CALIB: navigation document present, summary PNG absent, backplane FITS absent, backplane metadata absent, navigation did not succeed (status error): incomplete
   Input check: 2 image(s) selected, 1 complete, 1 incomplete

Each line names the image by where its results are under the two roots.

Cloud Tasks Variant
^^^^^^^^^^^^^^^^^^^

Queue-driven processing for the labels pass is supported by ``sd_create_bundle_cloud_tasks``.
This variant reads tasks from a queue, one image per task, and accepts the same
environment options used to derive configuration and results roots.

.. code-block:: bash

   sd_create_bundle_cloud_tasks \
     --config-file /path/to/config.yaml \
     --nav-results-root /data/nav/results \
     --backplane-results-root /data/nav/backplanes \
     --bundle-results-root /data/nav/bundle

Each task payload must be a JSON object with the following fields:

* ``dataset_name``: a dataset that can be bundled (see `Supported Datasets`_).
* ``files``: an array holding one object, for the task's image, with the required
  fields ``image_file_url``, ``label_file_url`` and ``results_path_stub``, and the
  optional field ``index_file_row`` (the image's row of its PDS3 index).

Summary Pass
------------

The summary pass generates the collection files and global index files that aggregate
information across all processed images, and the files that describe the bundle as a
whole. This pass should be run after all images have been processed in the labels pass.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   sd_create_bundle summary DATASET_NAME [options]

Command-Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

Environment options:

* ``--config-file PATH`` (repeatable): one or more configuration file paths to override
  defaults.
* ``--bundle-results-root PATH``: root directory where bundle results are located.
  If not provided, uses the ``NAV_BUNDLE_RESULTS_ROOT`` environment variable or the
  ``bundle_results_root`` configuration setting.

Examples
^^^^^^^^

Generate collection and global index files for a completed bundle:

.. code-block:: bash

   sd_create_bundle summary coiss_saturn \
     --bundle-results-root /data/nav/bundle

Check Pass
----------

The check pass checks a bundle the labels and summary passes wrote, and reports every
way it departs from what a PDS4 bundle has to be. It reads the bundle and the PDS4
schemas its labels name. By default it fetches each schema from the web address the
label gives, so it needs the network, and keeps what it fetches in a cache, so that a
later check fetches nothing it already has: the directory
``_filecache_spindoctor_pds4_schemas`` in your own cache directory, ``~/.cache`` (or the
directory the ``XDG_CACHE_HOME`` environment variable names, if it is set), or in the
directory ``FILECACHE_CACHE_ROOT`` names, if that is set. With
``--schema-dir`` it reads each schema from the file of the same name in that directory
instead, and fetches nothing. It writes nothing in the bundle, and no log. It uses the
``lxml``, ``elementpath`` and ``xmlschema`` packages, which are installed with
SpinDoctor.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   sd_create_bundle check DATASET_NAME [options]

Command-Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to override
  defaults.
* ``--bundle-results-root PATH``: root directory where the bundle is. If not provided,
  uses the ``NAV_BUNDLE_RESULTS_ROOT`` environment variable or the
  ``bundle_results_root`` configuration setting.
* ``--schema-dir PATH``: read each schema from the file of the same name in this
  directory, and fetch nothing. It has to hold every schema the labels name, as the PDS
  publishes them, and every schema those import.

It takes no logging options.

What It Checks
^^^^^^^^^^^^^^

* Every label against the XML schemas and the Schematron rules it declares. A schema
  that cannot be fetched, or that the ``--schema-dir`` directory does not hold, is
  reported with its web address.
* Every table -- the index tables and each collection's list of members -- read through
  its label: where each part of the file begins and ends, how many records and fields
  it holds, where each field lies and what number it is given, whether each record ends
  in exactly the characters the label says it does, and whether each value is of its
  field's type. A value equal to its field's missing constant must be written as the
  constant is.
* The layout of the index tables: the header line names the columns in order, separated
  by commas, and a comma alone lies between two values of a row.
* Each column of an index table against the configuration: its unit, and its missing
  constant, which is the masked value written in the column's format.
* Each row of an index table against the bundle: it names a data product the bundle
  holds, with that product's label and its start and stop times, and each data product
  has a row in the bodies table for each body its backplanes give statistics for and, if
  they give ring statistics, a row in the rings table.
* The bundle as a whole: every file a label names is beside the label, is named by its
  name alone, and has the size and MD5 checksum the label gives it; every other file is
  named by exactly one label; no label holds a leftover template marker, ``[[[``, which
  a label template leaves where it could not fill in a value; no element is empty,
  unless it carries ``xsi:nil``, which marks it empty on purpose; no two labels declare
  the same product; each collection's list of members names products the bundle holds,
  at the versions it holds, and names each product in the collection's directory once;
  and every reference to a product of the bundle names one the bundle holds, at the
  version it holds.

Each finding is an error or a warning. A warning is something the NASA PDS ``validate``
tool also reports as a warning: a reference to a product of the bundle that the bundle
does not hold, a product its collection's list of members leaves out, and the breach of
a Schematron rule marked as a warning. A bundle written without its user guide gets a
warning for each reference to the guide.

``validate`` is the tool to run on a bundle before it is delivered. It checks three
things the check does not: where each array begins in a FITS file; the user guide's PDF,
which must be one its VeraPDF library can read; and each reference a label makes to a
product outside the bundle, such as a target or the mission, against the products
registered with the PDS. Neither tool checks the version a collection's list of members
gives a product outside the bundle, nor whether what a label says is right rather than
merely allowed -- a unit the schemas accept but the value is not in, a target the
product has that the label leaves out, or the length a label gives an axis of a FITS
array. Check those by hand.

Output
^^^^^^

The check prints one line for each finding: the file, relative to the bundle's
directory; whether it is an error or a warning; the part of the check that found it
(``xml``, ``xsd``, ``schematron``, ``table`` or ``integrity``); where in the file; and
what is wrong. Then it prints the number of errors and of warnings:

.. code-block:: text

   bundle.lblx: error [xsd] /Product_Bundle/Identification_Area/Citation_Information/doi: value doesn't match any pattern of ['10\\.\\S+/\\S+'] (line 17)
   spice_kernels/kernels.lblx: warning [integrity] /Product_SPICE_Kernel/Reference_List/Internal_Reference/lid_reference: refers to urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:document:backplanes-user-guide, which no label of the tree declares
   Bundle check of /data/nav/bundle/cassini_iss_saturn_backplanes_rsfrench2027: 1 error(s), 1 warning(s)

A bundle written without its user guide refers to the guide from several labels, and
the check warns of each such reference, since the bundle does not hold the guide.

Examples
^^^^^^^^

Check a bundle:

.. code-block:: bash

   sd_create_bundle check coiss_saturn \
     --bundle-results-root /data/nav/bundle

Check it without the network, reading the schemas from a directory of copies:

.. code-block:: bash

   sd_create_bundle check coiss_saturn \
     --bundle-results-root /data/nav/bundle \
     --schema-dir /data/pds4/schemas

Inputs and Outputs
==================

Input Files
-----------

The labels pass requires:

* Navigation metadata files (``*_metadata.json``) from the navigation pass
* Backplane FITS files (``*_backplanes.fits``) from the backplanes pass
* Backplane metadata files (``*_backplane_metadata.json``) from the backplanes pass
* Summary PNG files (``*_summary.png``) from the navigation pass, one for every
  image whose navigation succeeded

A run that also names an error filter (``--has-offset-error``,
``--has-no-offset-error``, ``--has-offset-spice-error``,
``--has-offset-nonspice-error``) has already read each selected image's
navigation document, because that is how the filter decided what to select. The
record travels with the image, so each such image's document is read once for
the whole run, and what the supplemental file records is what the document said
when the selection was made.

The summary pass requires:

* All supplemental files (``*_supplemental.txt``) generated by the labels pass
* The dataset's template directory (see `Templates`_), from which it takes the readme,
  the member lists of the context, document, SPICE kernel and XML schema collections, the
  metakernel and, when the directory holds it, the user-guide PDF

Images a Bundle Leaves Out
--------------------------

A bundle holds the images whose backplanes describe something. The labels pass leaves
out, without counting it as an error:

* an image that was never navigated, or whose navigation did not succeed;
* an image whose backplanes were never generated;
* an image in which no body and no rings the backplanes are computed for show at any
  pixel, such as a frame of stars alone, or a frame of a body or a ring the backplanes
  are not computed for. Its backplanes hold no geometry, so a data label would have
  nothing to describe. A body in the field of view that shows at no pixel, such as one
  hidden behind a nearer body, does not count.

The log names each image left out and says why, and the labels pass's closing line
counts them as skipped. The instrument chapters say which bodies and rings each
dataset's backplanes cover.

Output Files
------------

Labels Pass Outputs
^^^^^^^^^^^^^^^^^^^

For each image, the labels pass generates:

* **PDS4 Label File** (``<image_name>_backplanes.lblx``): XML label file describing the
  backplane FITS file beside it, generated from dataset-specific templates.

* **Backplane FITS File** (``<image_name>_backplanes.fits``): a byte-for-byte copy of
  the image's backplane FITS from the backplane results root.

* **Supplemental File** (``<image_name>_supplemental.txt``): JSON file containing combined
  navigation and backplane metadata, including:

  * Navigation metadata (offset, uncertainty, confidence, etc.)
  * Backplane metadata (min/max statistics per body and ring, inventory information)

* **Browse Label File** (``<image_name>_summary.lblx``): XML label file describing the
  browse image, generated from dataset-specific templates.

* **Browse Image** (``<image_name>_summary.png``): Copy of the summary PNG from the
  navigation pass.

Browse products are not optional. Both are written for every image the pass
labels, and an image whose summary PNG is missing from the navigation results is
failed rather than bundled without them.

The data label describes each HDU of the FITS beside it, and each image HDU's array:
its name, its size in lines and samples, its element type and, where it has one,
its unit. Each float array declares the masked value (``backplanes.masked_value``,
``-999.0`` as shipped) as its missing constant.

Each data label states when its image's exposure began and ended, in its
``Time_Coordinates``: the start and stop the navigation recorded for the exposure,
whether or not it found the pointing, in UTC, to the millisecond, as in
``2004-02-07T04:25:35.585Z``.

Each data label cites, as its source product, the calibrated image its backplanes
were computed from, as the PDS Ring-Moon Systems Node holds it: by the image's volume
and the path of its label within that volume, as in
``COISS_2001:data/1454725799_1455008789/N1454725799_1_CALIB.LBL``.

Each data label names its targets, the bodies and rings its backplanes cover: one
``Target_Identification`` for each body that shows at a pixel of the image, where at
least one of its backplanes has a value, and one for the rings when the image has ring
backplanes. A body in the field of view that shows at no pixel, such as one hidden
behind a nearer body, is not named. Each gives the target's name and type and refers
to its PDS4 context product by its logical identifier, as the targets table identifies
it (see `Targets`_): for example ``Saturn``, of type ``Planet``, at
``urn:nasa:pds:context:target:planet.saturn``.

A data label states no ring geometry: of the rings it names the target alone. Each
ring image's ranges, the least and the greatest value of each of its ring backplanes,
are in the global rings index (see `Global Index Tables`_).

Each data label, the data collection label and the bundle label declare one set of
science facets: the ``Visible`` wavelength range and the ``Ring-Moon Systems``
discipline.

All files are placed in the bundle directory structure under ``data/`` and ``browse/``
directories, with paths determined by dataset-specific logic.

Summary Pass Outputs
^^^^^^^^^^^^^^^^^^^^

The summary pass generates:

* **Collection Data Files**:

  * ``collection_data.csv``: CSV file listing all data products in the bundle
  * ``collection_data.lblx``: PDS4 label for the data collection

* **Collection Browse Files**:

  * ``collection_browse.csv``: CSV file listing all browse products in the bundle
  * ``collection_browse.lblx``: PDS4 label for the browse collection

* **Miscellaneous Collection**, in ``miscellaneous/`` (see `Global Index Tables`_):

  * ``global_bodies_index.tab``: a table with one row for each body seen in each image
    of the bundle, giving the least and the greatest value of each body backplane
  * ``global_bodies_index.lblx``: PDS4 label for the bodies index
  * ``global_rings_index.tab``: a table with one row for each image with ring
    backplanes, giving the least and the greatest value of each ring backplane
  * ``global_rings_index.lblx``: PDS4 label for the rings index
  * ``collection_miscellaneous.csv``: the collection's members: the index tables it
    holds, and the context products and documents the bundle cites
  * ``collection_miscellaneous.lblx``: PDS4 label for the collection

* **Bundle Files**:

  * ``bundle.lblx``: PDS4 label for the bundle, naming each collection the bundle holds
  * ``readme.txt``: the bundle's readme, written from the dataset's template directory,
    giving the logical identifiers of the bundle and of its user guide

* **Context, Document, SPICE Kernel and XML Schema Collections**: for each, a
  ``collection_<name>.csv`` listing its members and its PDS4 label,
  ``collection_<name>.lblx``. Each list is the one the dataset's template directory
  gives, naming the bundle's own products under the bundle's name and version, except
  that the user guide and the metakernel are listed only when their labels are
  written, and that the context collection's list also names every target the data
  labels name, each at the version of its context product. The XML schema collection
  lists each dictionary schema the configuration gives (see `Configuration`_).

* **Metakernel**: ``spice_kernels/kernels.ker``, the SPICE metakernel every data label
  names, and its PDS4 label, ``kernels.lblx``. It lists no SPICE kernels, and its label
  says so. When its label cannot be written, the bundle has no SPICE kernel collection
  and no bundle label, and the pass exits 1.

The data collection label, the SPICE kernel collection label, the bundle label and the
metakernel label each name every target the bundle's data labels name, once each. The
document and miscellaneous collections list no target, since none of their labels names
one.

* **User Guide**: the bundle's user guide, a PDF, copied into ``document/user_guide/``
  from the dataset's template directory, and its PDS4 label beside it. When the template
  directory does not hold the user guide, the bundle has none, the document collection
  lists none, and the pass writes a warning naming the file it looked for. When its
  label cannot be written, the document collection does not list it either, and the
  pass exits 1.

The data collection label and the bundle label state the time range of the products
the data collection holds, in whole seconds: from the earliest exposure start, rounded
down, to the latest exposure stop, rounded up, as in ``2004-02-07T04:25:35Z`` to
``2004-02-22T05:32:17Z``. The bundle label is written only when every collection it
names is in the bundle.

Global Index Tables
^^^^^^^^^^^^^^^^^^^

The two index tables in ``miscellaneous/`` summarize the backplanes of every image the
bundle's data collection holds, so that a program can choose images without opening a
FITS file.

* ``global_bodies_index.tab`` has one row for each body seen in each image, a body that
  shows at one pixel at least. A body with no value for some backplane holds the masked
  value in that backplane's columns.
* ``global_rings_index.tab`` has one row for each image that has ring backplanes. When
  no image has ring backplanes, neither this table nor its label is written.

When neither table has a row, the miscellaneous collection is not written either, and
the summary pass exits 1 (see `Exit Status`_).

Each table begins with one line naming its columns, separated by commas. Every row
after it has the same length: each value is padded with spaces to its column's width
and followed by a comma, the last by the end of the line. The label beside each table
gives every column's name, position, width, data type, unit and description.

A row's first column, ``pds:logical_identifier``, is the logical identifier of the
image's data product, and its ``file_spec`` column is the path of that product's label
in the bundle. Its ``pds:start_date_time`` and ``pds:stop_date_time`` columns give when
the exposure began and ended, in UTC to the millisecond, as that label does. The bodies
table's ``body_name`` column names the body. The other
columns come in pairs, the least and the greatest value one backplane takes over the
pixels where it has a value, named as the configuration names them, as in
``geom:minimum_latitude`` and ``geom:maximum_latitude``.

Angular columns are in degrees, although the backplane arrays are in radians (see
:doc:`user_guide_backplanes`). Each column is written with a precision suited to its
unit: three decimal places for ``deg``, one for ``km``, eight for ``deg/pixel``, and
five significant figures for ``km/pixel``.

Where an image has no value for a backplane, both of its columns hold the masked value
(``backplanes.masked_value``, ``-999`` as shipped), written with the column's own
precision: ``-999.000`` in a column of degrees, ``-999.0`` in kilometers,
``-999.00000000`` in degrees per pixel and ``-999.00`` in kilometers per pixel. The
label declares that value, as written, as the column's missing constant.

Exit Status
===========

Each pass exits 0 when it wrote everything it set out to write, and 1 when it did
not; the log says what went wrong. An option a program does not recognize ends it
with exit status 2 before it does anything.

* ``sd_create_bundle labels`` exits 1 without writing anything if the bundle
  directory already holds files or a template is missing. Otherwise it exits 1
  if any image failed. It ends with a line giving the number of images labeled
  and skipped, and, when any failed, the number whose labels were not written.

  An image with nothing to describe is skipped, which is not an error (see
  `Images a Bundle Leaves Out`_). An image whose navigation recorded no pointing
  is bundled like any other. An image fails if a label
  cannot be written, its summary PNG is missing, its backplane metadata holds a
  statistic the index tables cannot hold (one in a unit other than the
  configured one, or a minimum or maximum that is NaN or infinite), or it was
  navigated by an earlier version, which did not record the exposure times with
  the observation. For such a statistic, regenerate that image's backplanes; an image
  navigated by an earlier version must be navigated again before it can be
  bundled. An image whose backplanes cover a body the targets table has no entry
  for fails too, the log naming the body (see `Targets`_), and so does an image whose
  ring backplanes were generated by an earlier version, which did not record the ring
  target: regenerate its backplanes.

  ``--dry-run`` writes nothing and ends with the number of images it would
  process. It exits 0 if the bundle directory is empty and every template is
  present.

  ``--check-only`` writes no labels, logs or bundle files, and exits 1 if any selected
  image is incomplete or the selection hands the labels pass a group of images it
  cannot label, an empty group included (see `Checking the Inputs`_), and 0 otherwise.
  It does not look at the bundle directory or the templates.

* ``sd_create_bundle summary`` exits 1 without writing anything if a file it needs
  from the template directory is missing (the user-guide PDF apart), or if the bundle has
  no ``data/`` directory: run the labels pass
  first, or check ``--bundle-results-root``. It exits 1 if a collection has no
  products, as when the labels pass labeled no image, and writes no files for
  that collection: check the labels pass's closing count. It exits 1 if an
  image's products disagree -- a data label with no browse label, or a browse
  label or supplemental file with no data label -- naming each such image. That
  is what an image the labels pass failed leaves: fix the image or drop it from
  the selection, then run both passes again into an empty directory. It exits 1
  if a label cannot be written, the bundle label included: that label is written
  only when every collection it names is in the bundle, so a collection that was not
  written leaves the bundle without it. A missing user-guide PDF is a warning, not
  a failure. An index table with no rows is not written; the log records
  that, and it is not a failure. When neither table is written, though, the
  miscellaneous collection has no products, and the pass exits 1 as for any
  collection with none.
  If a supplemental file holds such a statistic, it exits 1 and leaves none
  of the files the summary pass writes: regenerate the backplanes, then the bundle,
  into an empty directory.

* ``sd_create_bundle check`` exits 1 if it finds an error, if there is no bundle
  directory under the bundle results root, or if the check itself stops, in which case
  it prints why instead of the counts. Otherwise it exits 0, warnings or none.

* ``sd_create_bundle_cloud_tasks`` reports a task whose products could not be
  written as ``status: error``, with ``status_error`` saying why (for example
  ``label_not_written``), and does not retry it. A task that stops on an error,
  such as a missing template, is reported by the queue worker as an exception
  instead, and is not retried unless the worker is set to retry on an exception
  (``--retry-on-exception``, or ``retry_on_exception`` in the run configuration).

A summary pass describes whatever is in the bundle, so its exit status says
nothing about images the labels pass skipped; the labels pass's closing line
says what the bundle covers.

Configuration
=============

PDS4 bundle generation is configured through the ``pds4:`` section in configuration
files. Each dataset can have its own configuration, and it is where the bundle's name
and version are set, and where the schemas its labels declare are found:

.. code-block:: yaml

   pds4:
     coiss_saturn:
       template_dir: cassini_iss_saturn_1.0
       bundle_name: cassini_iss_saturn_backplanes_rsfrench2027
       bundle_version: '1.0'
       information_model_version: '1.24.0.0'
       schemas:
         pds:
           location: https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00
           lidvid: urn:nasa:pds:system_bundle:xml_schema:pds-xml_schema::1.24
         geom:
           location: https://pds.nasa.gov/pds4/geom/v1/PDS4_GEOM_1O00_19B0
           lidvid: urn:nasa:pds:system_bundle:xml_schema:geom-xml_schema::1.19
         # and one entry for each other dictionary the labels declare

The ``cassini_iss_saturn_1.0`` template directory ships with the package. The
``coiss_cruise`` dataset's does not; to bundle it, add an entry whose
``template_dir`` points at a template directory you create yourself, holding
the files listed under `Templates`_ (by name inside the package template root,
or as an absolute path), and which gives every other key below.

Configuration Options
---------------------

* ``template_dir``: Name or absolute path to the template directory. If just a name, it
  is resolved relative to ``src/spindoctor/cli/pds4/templates/`` in the ``rms-spindoctor`` package.
  If an absolute path, it is used as-is.

* ``bundle_name``: The bundle's name (e.g., ``cassini_iss_saturn_backplanes_rsfrench2027``):
  the name of its directory, and the last part of its logical identifier,
  ``urn:nasa:pds:<bundle_name>``, which the logical identifier of each of its
  collections and products extends. Every label, every list of members and the readme
  name the bundle by it.

* ``bundle_version``: The bundle's version, as ``<major>.<minor>`` in quotes (e.g.,
  ``'1.0'``). It is the version of the bundle, of each of its collections and of each
  product it holds, and the version every label and list of members names one of them
  at. A product outside the bundle that a label names, such as a context product, keeps
  its own version.

* ``information_model_version``: The version of the PDS4 information model the labels
  are written against, which every label states. It is the version of the build the
  ``pds`` schema belongs to, and changes when that schema does.

* ``schemas``: The schema of each dictionary the labels declare, keyed by the prefix
  its namespace takes in a label (``pds``, ``geom``, and so on). ``location`` is the
  web address of the dictionary's XML schema and Schematron without the extension, to
  which ``.xsd`` and ``.sch`` are added, and ``lidvid`` is the logical identifier and
  version the XML schema collection lists the dictionary by. Every label declaring the
  dictionary takes its schema from here, so moving a dictionary to another version is
  one change. Give an entry for exactly the dictionaries the templates declare: a
  template declaring one with no entry is not written, and an entry no template
  declares is still listed by the XML schema collection.

``bundle_name``, ``bundle_version``, ``information_model_version`` and ``schemas`` have
no default, so an entry for a dataset that is bundled gives all four.

Targets
-------

The targets a label names come from ``backplanes.target_lids`` in
``config_900_backplanes.yaml``, one entry per target, keyed by the name the backplane
metadata gives it: a body by its name, and the rings by the ring target their backplanes
are computed for. Each entry gives the logical identifier of the target's PDS4 context
product, the version of that product, and the name and type the product gives the target:

.. code-block:: yaml

   backplanes:
     target_lids:
       SATURN:
         lid: urn:nasa:pds:context:target:planet.saturn
         version: '1.4'
         name: Saturn
         type: Planet

A label refers to a target by its logical identifier, and the context collection lists
it at its version. The shipped table has an entry for every body and ring target the
backplane stage can produce for the datasets that can be bundled; the instrument
chapters say which those are.

Bundle Results Root
-------------------

The bundle results root can be specified via:

1. Configuration file: ``environment.bundle_results_root``
2. Environment variable: ``NAV_BUNDLE_RESULTS_ROOT``
3. Command-line argument: ``--bundle-results-root``

Command-line arguments have the highest priority, followed by environment variables,
then configuration files.

Templates
=========

PDS4 labels are generated using templates from the ``src/spindoctor/cli/pds4/templates/`` directory.
Each dataset has its own template directory containing:

* ``data.lblx``: Template for individual backplane data product labels
* ``browse.lblx``: Template for individual browse product labels
* ``collection_data.lblx``: Template for data collection label
* ``collection_browse.lblx``: Template for browse collection label
* ``global_bodies_index.lblx`` and ``global_rings_index.lblx``: Templates for the
  index tables' labels
* ``collection_miscellaneous.lblx``: Template for the miscellaneous collection label
* ``bundle.lblx``: Template for the bundle label
* ``readme.txt``: The bundle's readme, written into the bundle from this template
* ``collection_context.csv``, ``collection_document.csv``,
  ``collection_spice_kernels.csv`` and ``collection_xml_schema.csv``: The members of the
  context, document, SPICE kernel and XML schema collections, written into the bundle
  from these templates, the context collection's with every target the data labels name
  added after them;
  and ``collection_context.lblx``, ``collection_document.lblx``,
  ``collection_spice_kernels.lblx`` and ``collection_xml_schema.lblx``, the templates
  for their labels
* ``kernels.ker`` and ``kernels.lblx``: The metakernel, copied into the bundle, and the
  template for its label
* The template for the user guide's label and, when it exists, the user-guide PDF itself:
  for ``coiss_saturn``, ``cassini-iss-saturn-backplanes-user-guide.lblx`` and
  ``cassini-iss-saturn-backplanes-user-guide.pdf``. The user-guide PDF is the one file
  the summary pass can do without, and ``cassini_iss_saturn_1.0`` does not hold it;
  the template for its label is required.

The document collection's list of members names the user guide; when the template
directory does not hold the guide, the summary pass leaves that line out.

No template spells the bundle's name or version, a schema's location or the
information model version. Every template, the readme's and the lists of members
among them, takes them from the configuration (see `Configuration`_):
``BUNDLE_LID`` is the bundle's logical identifier, ``urn:nasa:pds:<bundle_name>``;
``BUNDLE_VERSION`` its version; ``INFORMATION_MODEL_VERSION`` the information model
version; ``PDS4_<PREFIX>_SCHEMA`` and ``PDS4_<PREFIX>_SCHEMA_XSD`` the Schematron and
the XML schema of each dictionary, ``<PREFIX>`` being its prefix in upper case, as in
``PDS4_GEOM_SCHEMA_XSD``; and ``XML_SCHEMA_LIDVIDS`` the list the XML schema
collection gives.

Each dataset supplies its own values for the other variables its templates use.
:doc:`/dev_guide/dev_guide_pds4` describes how a dataset does that, and what a
new one has to provide.

Supported Datasets
==================

As the package ships, only the Cassini ISS Saturn dataset (``coiss_saturn``) can
be bundled. Either pass stops with an error on any other dataset, before writing
anything. Adding a dataset is a code change, described in
:doc:`/dev_guide/dev_guide_pds4`.

Workflow
========

Typical workflow for generating a complete PDS4 bundle:

1. **Run Navigation Pass**: Generate navigation metadata and summary images

   .. code-block:: bash

      sd_offset coiss_saturn --volumes COISS_2001 \
        --nav-results-root /data/nav/results

2. **Run Backplanes Pass**: Generate backplane FITS files and metadata

   .. code-block:: bash

      sd_backplanes coiss_saturn --volumes COISS_2001 \
        --nav-results-root /data/nav/results \
        --backplane-results-root /data/nav/backplanes

3. **Run Bundle Labels Pass**: Generate PDS4 labels and supplemental files for each image.
   Run it first with ``--check-only`` to see which images have what the pass needs.

   .. code-block:: bash

      sd_create_bundle labels coiss_saturn --volumes COISS_2001 \
        --nav-results-root /data/nav/results \
        --backplane-results-root /data/nav/backplanes \
        --bundle-results-root /data/nav/bundle

   For large datasets, use the cloud tasks variant for parallel processing:

   .. code-block:: bash

      sd_create_bundle_cloud_tasks \
        --nav-results-root /data/nav/results \
        --backplane-results-root /data/nav/backplanes \
        --bundle-results-root /data/nav/bundle

4. **Run Bundle Summary Pass**: Generate collection and global index files

   .. code-block:: bash

      sd_create_bundle summary coiss_saturn \
        --bundle-results-root /data/nav/bundle

5. **Check the Bundle**: Report every way the bundle departs from PDS4

   .. code-block:: bash

      sd_create_bundle check coiss_saturn \
        --bundle-results-root /data/nav/bundle

Troubleshooting
===============

Common Issues
-------------

* **Missing navigation metadata**: Ensure the navigation pass has completed successfully
  and metadata files exist in the navigation results root.

* **Missing backplane files**: Ensure the backplanes pass has completed successfully and
  both FITS and metadata files exist in the backplane results root.

* **Template not found**: the pass exits before writing anything, naming each
  file it could not find. Check that ``template_dir`` names a directory holding
  every file listed above, the user-guide PDF apart.

* **User-guide PDF not in the template directory**: the summary pass warns, naming
  the file, and writes the bundle without a user guide. Put the PDF in the template
  directory under that name and run the summary pass again.

* **Bundle label not written**: the summary pass names each collection the bundle
  label names that is not in the bundle, and the error for that collection, earlier
  in the log, says why it was not written. Or it says no data label in the data tree
  has a supplemental file beside it, so there is no time range for the label to
  state: the labels pass labeled no image, and has to be run first.

* **No entry for a target**: the labels pass fails each image whose backplanes cover a
  body or rings the targets table does not identify, naming it. Add an entry for it to
  ``backplanes.target_lids`` (see `Targets`_), giving its PDS4 context product, and run
  the labels pass again into an empty directory.

* **Summary PNG not found**: that image is failed. A successfully navigated
  image always has one, so either it was removed from the navigation results or
  the document beside it did not come from the navigation pass. Re-navigate the
  image, or drop it from the selection.

* **Collection files incomplete**: Ensure all images have been processed in the labels
  pass before running the summary pass.

* **Products disagree**: the summary pass names each image that has a data label
  and no browse label, or a browse label or supplemental file and no data label.
  The labels pass failed that image, and its log says why. Fix the image or drop
  it from the selection, clear the bundle directory, and run both passes again.

* **Bundle root already holds files**: clear the bundle's directory under the
  bundle results root, or point ``--bundle-results-root`` somewhere else, and
  run the pass again from the start.

* **Label not written**: the run names the label it could not write and exits
  non-zero. The ``pdstemplate`` lines just above it name the template expression
  that failed. This is a fault in the dataset's templates or in the metadata
  they are given rather than anything a run can be asked to do differently; see
  :doc:`/dev_guide/dev_guide_pds4`.

Getting Help
------------

If you encounter persistent issues:

* Review logs for detailed error messages
* Verify that all prerequisite passes (navigation, backplanes) have completed
* Check that configuration files specify correct template directories and bundle names
* Ensure file paths and permissions are correct for all results directories
