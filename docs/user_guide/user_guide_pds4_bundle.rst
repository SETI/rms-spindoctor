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

Each bundle follows a standard PDS4 directory structure:

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
   ├── document/
   │   └── supplemental/
   │       ├── global_index_bodies.lblx
   │       ├── global_index_bodies.tab
   │       ├── global_index_rings.lblx
   │       └── global_index_rings.tab
   └── bundle.lblx

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

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   sd_create_bundle labels DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported dataset names (see Navigation User Guide).

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
Navigation User Guide).

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
This variant reads tasks from a queue and processes each batch of files. It accepts the
same environment options used to derive configuration and results roots.

.. code-block:: bash

   sd_create_bundle_cloud_tasks \
     --config-file /path/to/config.yaml \
     --nav-results-root /data/nav/results \
     --backplane-results-root /data/nav/backplanes \
     --bundle-results-root /data/nav/bundle

Each task payload must be a JSON object with the following fields:

* ``dataset_name``: one of the supported dataset names.
* ``arguments``: an object with optional keys ``nav_models`` and ``nav_techniques``
  (lists or ``null``).
* ``files``: an array of objects, each containing required fields ``image_file_url``,
  ``label_file_url``, and ``results_path_stub``, and optional fields ``index_file_row``
  (metadata) and ``extra_params`` (a JSON object/dictionary of arbitrary key/value pairs
  that will be passed through to the observation class's from_file method when the file is read).

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
* Summary PNG files (``*_summary.png``) from the navigation pass

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
  browse image, generated from dataset-specific templates (if summary PNG exists).

* **Browse Image** (``<image_name>_summary.png``): Copy of the summary PNG from the
  navigation pass (if available).

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
    containing min/max values for each configured backplane type (formatted to 5 decimal
    places)
  * ``global_index_bodies.lblx``: PDS4 label for the bodies index
  * ``global_index_rings.tab``: CSV file with one row per image, containing min/max
    values for each configured ring backplane type (formatted to 5 decimal places)
  * ``global_index_rings.lblx``: PDS4 label for the rings index

Exit Status
===========

Both passes exit 0 only when every label they set out to write is on disk. A
label whose template file is not in the dataset's template directory is one the
pass never set out to write, and is skipped without affecting the exit status.

A label is not written when its template cannot be rendered: an unresolved
template variable, an expression that fails, or a value the template rejects.
The label path and the number of errors are logged at error level, no label is
left at that path (including one an earlier run wrote there), and the pass
carries on, so a single run reports every label it could not write rather than
one per run.

* ``sd_create_bundle labels`` exits 1 when an image's data label or browse label
  was not written, or when an image's inputs could not be read. It closes with a
  line giving the number of images it labeled, the number it skipped and the
  number whose labels it did not write, so a selection that matched no images at
  all reads as the zero it is.

  An image the bundle has nothing to describe is skipped, not failed, and does
  not affect the exit status: an image with no navigation metadata document, an
  image whose navigation did not succeed, and a navigated image with no
  backplane metadata document. A selection made by volume ordinarily names far
  more images than have been navigated and backplaned, and a run over one is
  mostly skips.

  An image whose navigation left no summary PNG has no browse products, so it
  gets no browse label and is not failed for the one it does not have. What a
  browse collection then says about that image is covered below.

  ``--dry-run`` reports what the run would have processed and exits 0. It writes
  nothing, so it counts nothing against the run.

* ``sd_create_bundle summary`` exits 1 when any collection or global index label
  was not written. The inventory and index ``.tab`` tables are written either
  way.

* ``sd_create_bundle_cloud_tasks`` returns a ``status: error`` result carrying
  ``status_error: label_not_written`` for a task whose label was not written, and
  asks for no retry, because a template that could not be rendered will not
  render on a second attempt.

A non-zero exit means the bundle is incomplete: the labels that did render are
still in place, and the log names the ones that did not.

The inventories and index tables the summary pass writes are built from what is
in the bundle's ``data/`` tree, so a run in which some images were skipped or
failed leaves the bundle internally inconsistent -- a browse product listed in
``collection_browse.tab`` that is not on disk, or an index row naming a label
that is not there -- and the summary pass is silent about it. Its exit status
does not report this: a labels pass that skipped images exits 0, and the
summary pass that follows it exits 0 as well. Take the labels pass's closing
line as the account of what the bundle covers.

Configuration
=============

PDS4 bundle generation is configured through the ``pds4:`` section in configuration
files. Each dataset can have its own configuration:

.. code-block:: yaml

   pds4:
     coiss_saturn:
       template_dir: cassini_iss_saturn_1.0
       bundle_name: cassini_iss_saturn_backplanes_rsfrench2027

The ``cassini_iss_saturn_1.0`` template directory ships with the package; to
configure another dataset, add an entry whose ``template_dir`` points at a
template directory you create yourself (by name inside the package template
root, or as an absolute path).

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

Templates use the PdsTemplate system (from ``rms-pdstemplate``) for variable
substitution. Template variables are provided by dataset-specific implementations of
``pds4_template_variables()``, which map PDS3 index columns and computed metadata to
PDS4 template variables.

Dataset-Specific Behavior
=========================

Each dataset class implements PDS4 bundle generation methods:

* ``pds4_bundle_template_dir()``: Returns the template directory path
* ``pds4_bundle_name()``: Returns the bundle name
* ``pds4_bundle_path_for_image()``: Maps image name to bundle directory path
* ``pds4_path_stub()``: Returns the full path stub (directory + filename prefix)
* ``pds4_template_variables()``: Returns template variable dictionary
* ``pds4_image_name_to_data_lidvid()``: Converts image name to data product LIDVID
* ``pds4_image_name_to_browse_lidvid()``: Converts image name to browse product LIDVID

Cassini ISS datasets (``coiss_cruise``, ``coiss_saturn``) provide complete
implementations that map PDS3 index columns to PDS4 ``cassini:`` namespace variables.
Other datasets may raise ``NotImplementedError`` for methods that are not yet
implemented.

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

* **Template not found**: Verify that the template directory exists and matches the
  ``template_dir`` configuration setting.

* **Summary PNG not found**: Browse products are optional. If summary PNGs are missing,
  browse labels will not be generated, but data products will still be created.

* **Collection files incomplete**: Ensure all images have been processed in the labels
  pass before running the summary pass.

* **Label not written**: the run logs ``Rendering PDS4 label ... drew N error(s)``
  and exits non-zero. The ``pdstemplate`` lines just above it name the template
  expression that failed; the usual cause is a template variable the dataset's
  ``pds4_template_variables()`` does not supply.

Getting Help
------------

If you encounter persistent issues:

* Review logs for detailed error messages
* Verify that all prerequisite passes (navigation, backplanes) have completed
* Check that configuration files specify correct template directories and bundle names
* Ensure file paths and permissions are correct for all results directories
