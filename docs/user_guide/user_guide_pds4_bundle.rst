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
   │       └── <image_name>_backplanes.fits
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
  backplane FITS file beside it, generated from dataset-specific templates.

* **Backplane FITS File** (``<image_name>_backplanes.fits``): a byte-for-byte copy of
  the image's backplane FITS from the backplane results root, so that the file the
  label describes, and whose size, checksum and time it states, is the one in the
  bundle.

* **Supplemental File** (``<image_name>_supplemental.txt``): JSON file containing combined
  navigation and backplane metadata, including:

  * Navigation metadata (offset, uncertainty, confidence, etc.)
  * Backplane metadata (min/max statistics per body and ring, inventory information)

  The data label describes it as one stream of 7-bit ASCII text with line-feed
  line endings, which is how it is written.

* **Browse Label File** (``<image_name>_summary.lblx``): XML label file describing the
  browse image, generated from dataset-specific templates.

* **Browse Image** (``<image_name>_summary.png``): Copy of the summary PNG from the
  navigation pass.

Browse products are not optional. Both are written for every image the pass
labels, and an image whose summary PNG is missing from the navigation results is
failed rather than bundled without them.

The data label describes every HDU of the FITS beside it, in the order the file
holds them. The description is read from the source FITS before anything is
written for the image, and the copy is the same bytes. Each HDU's header is a ``Header`` at
its byte offset, stating its length and the ``FITS 3.0`` parsing standard. Each
image HDU after the empty primary is an ``Array_2D_Image`` at the byte offset of
its data, identified by its HDU name in lower case (``body_id_map``,
``body_latitude``, ``ring_radius`` and so on), with a ``Line`` axis of the HDU's
``NAXIS2`` elements and a ``Sample`` axis of its ``NAXIS1``, an element type of
``IEEE754MSBSingle`` for a float plane and ``SignedMSB4`` for the body identity map,
and the HDU's ``BUNIT`` as its unit: ``rad`` for an angular plane, because the
arrays are in radians. Only the planes the FITS holds are described, so the label
of a frame with no ring backplanes describes no ring arrays.

Each float plane declares its masked value -- the configuration's
``backplanes.masked_value``, ``-999.0`` as shipped -- as the ``missing_constant`` of
its ``Special_Constants``, so a reader masks a plane on what its label says rather
than on a convention. ``BODY_ID_MAP`` declares none: its ``0`` marks a pixel no
body claimed and every other value is the NAIF ID of the body that did, which is
what its ``description`` says. Every float plane's ``description`` says what it
holds: the plane, the ``oops`` backplane method it came from, its unit, and that a
pixel the plane does not cover holds the missing constant. Each array has display
settings of its own, laying it out with ``Sample`` running left to right across
the display and ``Line`` top to bottom down it.

Each data label states when its image's exposure began and ended, in its
``Time_Coordinates``. The two times are the ``start_et`` and ``stop_et`` the
navigation metadata document records under ``navigation_result.times``, converted
to UTC and written the way PDS4 writes a date and time: to the millisecond, with a
trailing ``Z``, as in ``2004-02-07T04:25:35.585Z``. Each is rounded to the nearest
millisecond. A Cassini image's start and stop are recorded to the millisecond in its
PDS3 label and index, and the navigation computes its epochs from those values, so
each epoch lies within a few nanoseconds of a millisecond and the nearest one is the
time PDS3 records; rounding a start down or a stop up would put it a millisecond
off whenever the epoch lands on the far side. A leap second is written as second
60. A navigated image whose document records no such times is failed rather than
labeled, as the exit status below describes.

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

The data collection label states the time range of the products the collection
holds: the earliest exposure start and the latest exposure stop over every
supplemental file in the bundle's ``data/`` tree, written to whole seconds with the
start rounded down and the stop rounded up, as in ``2004-02-07T04:25:35Z`` to
``2004-02-22T05:32:17Z``. Rounded outward, the range contains every product's own
start and stop as its data label states them. The pass writes the global index
files first, because the range is taken in the same read of the supplemental files
that builds the index, and the collection files after them.

Every min/max column is written in a fixed format chosen by its unit: three
decimals for a column in degrees, one for a column in km, eight for degrees per
pixel, and five significant figures, written positionally and never in
exponent form, for km per pixel, whose values span eight orders of magnitude.
The backplane arrays are float32, so a statistic carries seven significant
digits at most, and each format is chosen within that from what one pixel
resolves. A configured backplane in a unit the tables have no format for, or
with no unit at all, ends either pass before it reads anything, and fails every
cloud task before it writes anything.

Both index tables are meant to be read by a person, so every angular column in
them is in degrees — degrees per pixel where the quantity is a resolution — and
not in the radians the backplane arrays themselves carry. Columns that are not
angular are in the unit of the array they summarize: a ring radius in
kilometres, a radial resolution in kilometres per pixel. :doc:`user_guide_backplanes`
explains why the two products differ and where each states its unit.

Exit Status
===========

A pass exits 0 only when every file it set out to write is on disk. A non-zero
exit means the bundle is incomplete: whatever was written is still in place, and
the log names what was not.

* ``sd_create_bundle labels`` exits 1 when any image's labels could not be
  written or its inputs could not be read, and exits 1 before processing
  anything when the bundle's directory already holds files or when a
  configured backplane declares a unit the bundle cannot use. It closes with a
  line giving the number of images it labeled, skipped and failed, so a
  selection that matched nothing reads as the zero it is.

  An image the bundle has nothing to describe is **skipped**, not failed, and
  does not affect the exit status: an image with no navigation metadata
  document, one whose navigation did not succeed, and a navigated image with no
  backplane metadata document. A selection made by volume ordinarily names far
  more images than have been navigated and backplaned, so such a run is mostly
  skips.

  A navigated image whose summary PNG is missing is **failed**, not skipped. It
  loses its browse products and nothing else; its data label is written or not
  on its own account.

  A navigated image whose backplane metadata records a statistic in a unit
  other than the one the configuration gives its plane — a document written
  before the statistics recorded their unit, or under another configuration —
  is **failed** before anything is written for it, since indexing it would put
  one column of the global index in two units. Regenerate its backplanes. So
  is an image whose backplane metadata records a minimum or maximum that is
  not a finite number, NaN or an infinity among them: no column can hold one,
  and a blank in its place would say the plane measured nothing. The log names
  the plane and what the document records for it.

  A navigated image whose navigation metadata document does not record the
  exposure's start, stop and midtime -- no ``navigation_result.times`` block, an
  epoch missing, one that is not a finite number, or a stop earlier than its
  start -- is **failed** before anything is written for it. Its data label states
  when the exposure began and ended, and a successful navigation always records
  both, so such a document is a broken input rather than an image whose time is
  unknown. The log names the image and what its document lacks.

  ``--dry-run`` writes nothing, and exits 0 once the templates are present,
  every configured unit is usable, and the bundle directory is empty.

* ``sd_create_bundle summary`` exits 1 when any collection or global index label
  could not be written, and exits 1 before reading anything when a configured
  backplane declares a unit the bundle cannot use. The ``.tab`` tables are
  written whether or not the label describing one is. A bundle with no ``data/``
  directory ends the pass with exit status 1 and nothing written.

  When there is no time range for the data collection label to state -- the
  ``data/`` tree holds no supplemental file, or one of them records no exposure
  times a label can state -- that label is not written and counts as a label not
  written, so the pass exits 1; the log says why and names the file. Every other
  product of the pass is written as usual, the inventory tables among them.

  A supplemental file that cannot be read, or does not hold a JSON object, ends
  the pass with exit status 1 before either index table is written, and leaves no
  product of the pass, neither this run's nor any an earlier run wrote; the log
  names the file and why it could not be read, or what it holds in place of an
  object. The labels pass writes every supplemental file, so the remedy is to
  regenerate the bundle into an empty directory.

  A supplemental file with no data label beside it, or a data label with no
  supplemental file, ends the pass the same way, and the log names both. The
  labels pass writes a product's supplemental file before it renders the data
  label, so a data label that failed to render leaves the one without the other;
  the collection inventory, which lists a product by its data label, and the
  global index, which reads its supplemental file, would then disagree about it.
  The remedy is the same.

  It also exits 1 when a supplemental file records a statistic in a unit other
  than the one the configuration gives its plane, or in none, and leaves none of
  the pass's products: no index table, no collection file and no label of
  either, neither this run's nor any an earlier run wrote. A supplemental file
  carries a copy of the backplane document the labels pass read, so the file
  was written from one recorded before the statistics carried their unit, or
  under another configuration: regenerate the backplanes, then the bundle into
  an empty directory. The log names the file, the plane and both units. A
  supplemental file recording a minimum or maximum that is not a finite number
  ends the run the same way, and the log names the file and the plane and says
  what the file records there; the labels pass of a regenerated bundle fails
  such an image rather than writing it.

* ``sd_create_bundle_cloud_tasks`` reports a task whose label could not be
  written as ``status: error`` with ``status_error: label_not_written``, and asks
  for no retry. A task run under a configuration that declares a backplane in a
  unit the bundle cannot use, or with no unit, writes nothing and is reported as
  ``status: error`` with ``status_error: unusable_unit``, every such backplane
  and the reason in ``status_exception``. It asks for no retry either, since
  every task under that configuration fails the same way.

The summary pass builds its tables from what is in the bundle's ``data/`` tree
without checking that tree for completeness, so it can exit 0 over a bundle the
labels pass already failed images in -- and an image that got a data label but
no browse label leaves ``collection_browse.tab`` listing a browse product that is
not on disk. A summary pass exiting 0 says its own labels were written, and
nothing about what the labels pass did; take the labels pass's closing line as
the account of what the bundle covers.

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

Each dataset supplies its own values for the variables its templates use.
:doc:`/dev_guide/dev_guide_pds4` describes how a dataset does that, and what a
new one has to provide.

Supported Datasets
==================

As the package ships, one dataset can be bundled: ``coiss_saturn`` (also
registered as ``coiss_saturn_pds3``), whose template directory,
``cassini_iss_saturn_1.0``, is the only one included. Both passes refuse every
other dataset before they process an image, and write nothing into the bundle:

* ``coiss_cruise`` (and ``coiss_cruise_pds3``) has the same PDS4 support as
  ``coiss_saturn``, but its template directory, ``cassini_iss_cruise_1.0``, does
  not ship, so each pass exits 1 naming every template it needs and cannot
  find. To bundle it, point ``pds4.coiss_cruise.template_dir`` at a template
  directory of your own, as described under `Configuration`_.
* ``gossi``, ``nhlorri`` and ``vgiss`` (and their ``_pds3`` names) name a
  template directory, which does not ship either, and a bundle name, but do not
  say which templates a pass needs, nor provide the rest of what a label is
  built from. Each pass stops on them with a traceback ending in
  :exc:`NotImplementedError` before it looks for a template.
* ``coiss`` (and ``coiss_pds3``) and ``sim`` have no PDS4 support at all, and
  each pass stops on them the same way when it asks for the template directory.

Adding a dataset is a code change, described in :doc:`/dev_guide/dev_guide_pds4`.

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
