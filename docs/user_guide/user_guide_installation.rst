======================
Installation and Setup
======================

SpinDoctor is a Python package that installs a set of command-line programs. Installing
it is quick; most of the setup work is telling those programs where the data they need
lives. This chapter covers both. :doc:`/quick_start` has the abbreviated version.

Requirements
============

* Python 3.11 or later.
* Enough disk space for the results you intend to write, and for the local cache of any
  data read over the network.
* The external data described under `External Data`_ below. The amount you need depends
  on the mission: SPICE kernels and images for one Voyager encounter are a modest
  download, while the whole Cassini ISS archive is not.

Installing the Package
======================

To install into the current Python environment, where the package can also be imported::

   pip install rms-spindoctor

To install the command-line programs on their own, in an isolated environment of their
own, which is the better choice if you only intend to run them::

   pipx install rms-spindoctor

Either way the programs land on your ``PATH``. Check the installation by asking one of
them for its options::

   sd_offset coiss --help

The programs, and the chapter that documents each, are:

* ``sd_offset`` -- :doc:`user_guide_navigation_running`
* ``sd_create_ck`` -- :doc:`user_guide_ck_kernels`
* ``sd_backplanes`` and ``sd_backplane_viewer`` -- :doc:`user_guide_backplanes`
* ``sd_create_bundle`` -- :doc:`user_guide_pds4_bundle`
* ``sd_mosaic_rings``, ``sd_mosaic_body``, ``sd_mosaic_display_rings``, and
  ``sd_mosaic_display_body`` -- :doc:`user_guide_reprojection`
* ``sd_consolidate_metadata`` -- :doc:`user_guide_consolidate_metadata`
* ``sd_results_index`` -- :doc:`user_guide_results_index`
* ``sd_stats_report`` -- :doc:`user_guide_statistics`
* ``sd_create_simulated_image`` -- :doc:`user_guide_simulated_images`

The programs whose names end in ``_cloud_tasks`` are not run by hand; see
:doc:`user_guide_cloud_tasks`.

External Data
=============

SpinDoctor reads four kinds of data from outside the package. Each is located by an
environment variable, and most can also be named in a configuration file or on the
command line.

Every path described here may be a local directory or a URL. Remote locations are read
over the network and cached locally, so ``https://pds-rings.seti.org/holdings`` works
wherever a local holdings directory works. A remote location is convenient for a small
run and slow for a large one, because every file it reads is a download.

These are the locations the SpinDoctor developers read each kind of data from. Any local
copy or mirror of the same trees works just as well.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Variable
     - Where it can point
   * - ``PDS3_HOLDINGS_DIR``
     - ``https://pds-rings.seti.org/holdings``
   * - ``OOPS_RESOURCES``
     - ``https://storage.googleapis.com/rms-node-oops-resources``
   * - ``SPICE_PATH``
     - the ``SPICE`` directory of that same resource collection
   * - ``UCAC4_PATH``
     - ``https://storage.googleapis.com/rms-node-star-catalogs/UCAC4``
   * - ``YBSC_PATH``
     - ``https://storage.googleapis.com/rms-node-star-catalogs/YBSC``

SPICE kernels
-------------

Navigating an image requires the SPICE kernels that describe where the spacecraft was,
where it was pointing, and where the planets and moons were. Point ``SPICE_PATH`` at the
directory holding them:

.. code-block:: bash

   export SPICE_PATH=/path/to/spice/kernels

NASA's Navigation and Ancillary Information Facility publishes the kernels for each
mission at https://naif.jpl.nasa.gov/naif/data_archived.html. The geometry resource
collection described below also carries a kernel tree, under ``SPICE``, which is the set
SpinDoctor is developed against; pointing ``SPICE_PATH`` at that directory gives you
those kernels.

Every real navigation run needs this. Without the kernels an image cannot be navigated
at all, and the run records a SPICE error for it.

Image holdings
--------------

All four supported instruments are archived as PDS3 volumes. Point
``PDS3_HOLDINGS_DIR`` at the root of a PDS3 holdings tree, or pass
``--pds3-holdings-root`` on the command line:

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=/path/to/pds3/holdings

The PDS Ring-Moon Systems Node publishes such a tree at
https://pds-rings.seti.org/holdings, and that URL can be used directly as the value.
The layout, with one real Cassini volume and one of its images filled in::

   $PDS3_HOLDINGS_DIR/
       volumes/
           COISS_2xxx/
               COISS_2001/
                   data/
                       1454725799_1455008789/
                           N1454725799_1.IMG
                           N1454725799_1.LBL
       metadata/
           COISS_2xxx/
               COISS_2001/
                   COISS_2001_index.lbl
                   COISS_2001_index.tab

Both halves matter. The ``volumes`` tree holds the images and their labels. The
``metadata`` tree holds each volume's PDS3 index table, which is the table shipped with
the volume that has one row per image. Image selection is answered from those PDS3 index
tables rather than by opening images, so a holdings tree that has no ``metadata`` half
cannot be enumerated. See :doc:`user_guide_image_selection`.

Star catalogs
-------------

Navigation against the star field needs a star catalog. Set the variable for each
catalog you have installed:

* ``UCAC4_PATH`` -- the root of the UCAC4 catalog. The Ring-Moon Systems Node publishes
  it at https://storage.googleapis.com/rms-node-star-catalogs/UCAC4.
* ``YBSC_PATH`` -- the root of the Yale Bright Star Catalog, published at
  https://storage.googleapis.com/rms-node-star-catalogs/YBSC.
* Tycho-2 has no variable of its own. It is shipped as a SPICE star-catalog kernel and is
  read with the SPICE toolkit rather than by SpinDoctor, so it lives with the kernels: in
  a ``Stars`` directory under ``SPICE_PATH``, or in ``SPICE/Stars`` under
  ``OOPS_RESOURCES`` when ``SPICE_PATH`` is not set at all.

Which catalogs are consulted, and in what order, is a configuration setting; see
:doc:`user_guide_configuration`.

Geometry resources
------------------

``OOPS_RESOURCES`` names the root of the resource collection that the underlying geometry
library reads, published at https://storage.googleapis.com/rms-node-oops-resources. The
collection carries the SPICE kernel tree under ``SPICE`` and the Tycho-2 catalog under
``SPICE/Stars``.

A navigation run does not require this variable. Set ``SPICE_PATH`` and it is never
consulted; leave ``SPICE_PATH`` unset and it is where Tycho-2 is looked for.

Where Results Go
================

Each phase of the pipeline writes into a root of its own. Apart from the log root, none
of them has a built-in default, so a run must be told where to write each product it
produces. A run that uses a results index must also be told the index's connection URL.
A run that uses no results index needs no value for it. Each value is supplied by
exporting the environment variable, by setting it in a configuration file, or by
passing the command-line option. The command-line option wins over the configuration
file, which wins over the environment variable.

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Environment variable
     - Command-line option
     - What it holds
   * - ``NAV_RESULTS_ROOT``
     - ``--nav-results-root``
     - Navigation results: one metadata document and one summary image per navigated
       image.
   * - ``NAV_LOG_ROOT``
     - ``--log-root``
     - Log files. Defaults to a ``logs`` directory under the navigation results root.
   * - ``NAV_RESULTS_INDEX_DB``
     - ``--results-index-db``
     - Connection URL of the results index, a database holding one row per navigated
       image. Optional; see :doc:`user_guide_results_index`.
   * - ``NAV_BACKPLANE_RESULTS_ROOT``
     - ``--backplane-results-root``
     - Generated backplanes.
   * - ``NAV_BUNDLE_RESULTS_ROOT``
     - ``--bundle-results-root``
     - Generated PDS4 bundles.

A results root may also be a URL, so results can be written straight to cloud storage.

Checking the Setup
==================

Navigating one image end to end is the quickest way to confirm that the kernels, the
holdings, and the results root are all in place:

.. code-block:: bash

   export SPICE_PATH=/path/to/spice/kernels
   export PDS3_HOLDINGS_DIR=/path/to/pds3/holdings
   export NAV_RESULTS_ROOT=/path/to/results

   sd_offset coiss_saturn --volumes COISS_2001 --choose-random-images 1

The run prints its progress to the terminal and writes a metadata document and a summary
image under the results root. Near the end it prints one line for the image it navigated,
which looks like this::

   N1466448128_1_CALIB.IMG: status=success, offset (dv, du) = (1.500, -2.500) px,
   confidence 0.750 (medium)

A line reading ``status=success`` means the kernels, the holdings, and the results root
are all in place and the setup is good. The image name, the numbers, and the tier will
differ, since a random image was chosen.

:doc:`user_guide_navigation_running` describes the run and its options, and
:doc:`user_guide_navigation_troubleshooting` covers what to do when it does not work.

Settings other than paths -- which models and techniques run, the thresholds results are
judged against, and everything else that governs a navigation -- are described in
:doc:`user_guide_configuration`.
