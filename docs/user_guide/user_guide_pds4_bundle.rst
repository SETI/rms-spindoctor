======================
PDS4 Bundle Generation
======================

Overview
========

The PDS4 bundle generation system creates PDS4-compliant bundles from navigation and
backplane results. It generates PDS4 label files, supplemental metadata files, browse
products, and collection/index files that organize the data products into a complete
PDS4 bundle structure.

The bundle generation process consists of two main passes:

1. **Labels Pass**: Processes individual images to generate PDS4 labels, supplemental
   files, and browse products for each image.

2. **Summary Pass**: Generates collection files and global index files that aggregate
   information across all processed images.

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
   ├── browse/
   │   ├── collection_browse.tab
   │   ├── collection_browse.lblx
   │   └── <directory_structure>/
   │       └── <image_name>_summary.lblx
   │       └── <image_name>_summary.png
   ├── data/
   │   ├── collection_data.tab
   │   ├── collection_data.lblx
   │   └── <directory_structure>/
   │       └── <image_name>_backplanes.lblx
   │       └── <image_name>_supplemental.txt
   └── document/
       └── supplemental/
           ├── global_index_bodies.lblx
           ├── global_index_bodies.tab
           ├── global_index_rings.lblx
           └── global_index_rings.tab

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

The summary pass generates collection files and global index files that aggregate
information across all processed images. This pass should be run after all images have
been processed in the labels pass.

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

Output Files
------------

Labels Pass Outputs
^^^^^^^^^^^^^^^^^^^

For each image, the labels pass generates:

* **PDS4 Label File** (``<image_name>_backplanes.lblx``): XML label file describing the
  backplane FITS file, generated from dataset-specific templates.

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

All files are placed in the bundle directory structure under ``data/`` and ``browse/``
directories, with paths determined by dataset-specific logic.

Summary Pass Outputs
^^^^^^^^^^^^^^^^^^^^

The summary pass generates:

* **Collection Data Files**:

  * ``collection_data.tab``: CSV file listing all data products in the bundle
  * ``collection_data.lblx``: PDS4 label for the data collection

* **Collection Browse Files**:

  * ``collection_browse.tab``: CSV file listing all browse products in the bundle
  * ``collection_browse.lblx``: PDS4 label for the browse collection

* **Global Index Files**:

  * ``global_index_bodies.tab``: CSV file with one row per image/body combination,
    containing min/max values for each configured backplane type
  * ``global_index_bodies.lblx``: PDS4 label for the bodies index
  * ``global_index_rings.tab``: CSV file with one row per image, containing min/max
    values for each configured ring backplane type
  * ``global_index_rings.lblx``: PDS4 label for the rings index

Each min/max column is written with a precision suited to its unit: three
decimal places for ``deg``, one for ``km``, eight for ``deg/pixel``, and five
significant figures for ``km/pixel``. Angular columns are in degrees, although
the backplane arrays are in radians (see :doc:`user_guide_backplanes`).

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
  cannot be written, its summary PNG is missing, or its backplane metadata holds
  a statistic the index tables cannot hold (one in a unit other than the
  configured one, or a minimum or maximum that is NaN or infinite). For such a
  statistic, regenerate that image's backplanes.

  ``--dry-run`` writes nothing and ends with the number of images it would
  process. It exits 0 if the bundle directory is empty and every template is
  present.

* ``sd_create_bundle summary`` exits 1 without writing anything if a template is
  missing, or if the bundle has no ``data/`` directory: run the labels pass
  first, or check ``--bundle-results-root``. It exits 1 if a collection or index
  label cannot be written. If a supplemental file holds such a statistic, it
  exits 1 and writes neither index table: regenerate the backplanes, then the
  bundle, into an empty directory.

* ``sd_create_bundle_cloud_tasks`` reports a task whose products could not be
  written as ``status: error``, with ``status_error`` saying why (for example
  ``label_not_written``), and does not retry it. A task that stops on an error,
  such as a missing template, is reported by the queue worker as an exception
  instead, and is retried only if the worker runs with ``--retry-on-exception``.

A summary pass indexes whatever is in the bundle's ``data/`` tree, so its exit
status says nothing about the labels pass; the labels pass's closing line says
what the bundle covers.

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
``template_dir`` points at a template directory you create yourself (by name
inside the package template root, or as an absolute path).

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
* ``global_index_bodies.lblx``: Template for bodies global index label
* ``global_index_rings.lblx``: Template for rings global index label

Each dataset supplies its own values for the variables its templates use.
:doc:`/dev_guide/dev_guide_pds4` describes how a dataset does that, and what a
new one has to provide.

Supported Datasets
==================

As the package ships, only the Cassini ISS Saturn dataset (``coiss_saturn``) can
be bundled. Either pass stops with an error on any other dataset, before writing
anything. Adding a dataset
is a code change, described in :doc:`/dev_guide/dev_guide_pds4`.

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
  every template listed above.

* **Summary PNG not found**: that image is failed. A successfully navigated
  image always has one, so either it was removed from the navigation results or
  the document beside it did not come from the navigation pass. Re-navigate the
  image, or drop it from the selection.

* **Collection files incomplete**: Ensure all images have been processed in the labels
  pass before running the summary pass.

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
