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
   │   └── <directory_structure>/
   │       └── <image_name>_summary.lblx
   │       └── <image_name>_summary.png
   ├── context/
   │   ├── collection_context.csv
   │   └── collection_context.lblx
   ├── data/
   │   ├── collection_data.csv
   │   ├── collection_data.lblx
   │   └── <directory_structure>/
   │       └── <image_name>_backplanes.lblx
   │       └── <image_name>_backplanes.fits
   │       └── <image_name>_supplemental.txt
   ├── document/
   │   ├── collection_document.csv
   │   ├── collection_document.lblx
   │   └── user_guide/
   │       ├── <user_guide>.lblx
   │       └── <user_guide>.pdf
   ├── miscellaneous/
   │   ├── collection_miscellaneous.csv
   │   ├── collection_miscellaneous.lblx
   │   ├── global_bodies_index.lblx
   │   ├── global_bodies_index.tab
   │   ├── global_rings_index.lblx
   │   └── global_rings_index.tab
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

The directory structure within ``data/`` and ``browse/`` mirrors the structure of the
original PDS4 dataset (if it existed), with paths derived from image names using
dataset-specific logic.

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
* The dataset's template directory (see `Templates`_), from which it copies the readme,
  the member lists of the context, document, SPICE kernel and XML schema collections, the
  metakernel and, when the directory holds it, the user-guide PDF

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
``Time_Coordinates``: the start and stop in UTC, to the millisecond, as in
``2004-02-07T04:25:35.585Z``.

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

  * ``global_bodies_index.tab``: a table with one row for each body of each image in
    the bundle, giving the least and the greatest value of each body backplane
  * ``global_bodies_index.lblx``: PDS4 label for the bodies index
  * ``global_rings_index.tab``: a table with one row for each image with ring
    backplanes, giving the least and the greatest value of each ring backplane
  * ``global_rings_index.lblx``: PDS4 label for the rings index
  * ``collection_miscellaneous.csv``: the collection's members: the two index tables,
    and the context products and documents the bundle cites
  * ``collection_miscellaneous.lblx``: PDS4 label for the collection

* **Bundle Files**:

  * ``bundle.lblx``: PDS4 label for the bundle, naming each collection the bundle holds
  * ``readme.txt``: the bundle's readme, copied from the dataset's template directory

* **Context, Document, SPICE Kernel and XML Schema Collections**: for each, a
  ``collection_<name>.csv`` listing its members and its PDS4 label,
  ``collection_<name>.lblx``. Each list is the one in the dataset's template directory,
  except that the user guide and the metakernel are listed only when their labels are
  written.

* **Metakernel**: ``spice_kernels/kernels.ker``, the SPICE metakernel every data label
  names, and its PDS4 label, ``kernels.lblx``. It lists no SPICE kernels, and its label
  says so. When its label cannot be written, the bundle has no SPICE kernel collection
  and no bundle label, and the pass exits 1.

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

* ``global_bodies_index.tab`` has one row for each body of each image.
* ``global_rings_index.tab`` has one row for each image that has ring backplanes. When
  no image has ring backplanes, neither this table nor its label is written.

Each table begins with one line naming its columns, separated by commas. Every row
after it has the same length: each value is padded with spaces to its column's width
and followed by a comma, the last by the end of the line. The label beside each table
gives every column's name, position, width, data type, unit and description.

A row's first column, ``pds:logical_identifier``, is the logical identifier of the
image's data product, and its ``file_spec`` column is the path of that product's label
in the bundle. The bodies table's ``body_name`` column names the body. The other
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

  An image with nothing to describe (never navigated, navigation failed, or no
  backplanes) is skipped, which is not an error. An image fails if a label
  cannot be written, its summary PNG is missing, its navigation recorded no
  pointing, or its backplane metadata holds a statistic the index tables
  cannot hold (one in a unit other than the configured one, or a minimum or
  maximum that is NaN or infinite). For such a statistic, regenerate that image's
  backplanes.

  ``--dry-run`` writes nothing and ends with the number of images it would
  process. It exits 0 if the bundle directory is empty and every template is
  present.

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
  a failure, and so is an index table no image gives a row, which is not written.
  If a supplemental file holds such a statistic, it exits 1 and leaves none
  of the files the summary pass writes: regenerate the backplanes, then the bundle,
  into an empty directory.

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
files. Each dataset can have its own configuration:

.. code-block:: yaml

   pds4:
     coiss_saturn:
       template_dir: cassini_iss_saturn_1.0
       bundle_name: cassini_iss_saturn_backplanes_rsfrench2027

The ``cassini_iss_saturn_1.0`` template directory ships with the package. The
``coiss_cruise`` dataset's does not; to bundle it, add an entry whose
``template_dir`` points at a template directory you create yourself, holding
the files listed under `Templates`_ (by name inside the package template root,
or as an absolute path).

Configuration Options
---------------------

* ``template_dir``: Name or absolute path to the template directory. If just a name, it
  is resolved relative to ``src/spindoctor/cli/pds4/templates/`` in the ``rms-spindoctor`` package.
  If an absolute path, it is used as-is.

* ``bundle_name``: Name of the bundle directory (e.g., ``cassini_iss_saturn_backplanes_rsfrench2027``).

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
* ``readme.txt``: The bundle's readme, copied into the bundle as it is
* ``collection_context.csv``, ``collection_document.csv``,
  ``collection_spice_kernels.csv`` and ``collection_xml_schema.csv``: The members of the
  context, document, SPICE kernel and XML schema collections, copied into the bundle;
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

Each dataset supplies its own values for the variables its templates use.
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

3. **Run Bundle Labels Pass**: Generate PDS4 labels and supplemental files for each image

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
  in the log, says why it was not written. Or it says the data tree holds no
  supplemental file, so there is no time range for the label to state: the labels
  pass labeled no image, and has to be run first.

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
