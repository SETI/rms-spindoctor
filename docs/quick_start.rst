===========
Quick Start
===========

SpinDoctor determines where a spacecraft camera was really pointing when it took an
image. It reads an image from Cassini ISS, Voyager ISS, Galileo SSI, or New Horizons
LORRI, builds a model of what the sky, the rings, and the visible bodies should look
like at that moment from the SPICE kernels, and measures how far the real image has
shifted from that model. From the measurement it records a corrected pointing for the
image, and it can then turn that corrected pointing into SPICE C kernels, per-pixel
geometry backplanes, and PDS4 archive bundles.

This page is the short path: what the pipeline does, how to install SpinDoctor, and how
to navigate a single image and read the result. Everything here is covered in full in
the :doc:`/user_guide/user_guide` chapters, which this page links to as it goes.

The Pipeline
============

SpinDoctor processes imagery in four phases. Navigation comes first, and each of the
three phases after it reads what navigation recorded.

1. **Navigation.** Compare each image against models of stars, planets, moons, and
   rings, and record the pointing correction that makes the model line up with the
   image. Each navigated image gets its own metadata document holding the correction,
   its uncertainty, and the corrected pointing itself. Run with ``sd_offset``; see
   :doc:`/user_guide/user_guide_navigation_running`.

2. **Corrected-pointing C kernels.** Turn the corrected pointing that navigation
   recorded into SPICE C kernels, one corrected kernel mirroring each original kernel
   the images were navigated against, so any SPICE-based tool can use the improved
   attitude. Run with ``sd_create_ck``; see :doc:`/user_guide/user_guide_ck_kernels`.

3. **Backplane generation.** Compute per-pixel geometry for each navigated image:
   longitude, latitude, incidence, emission, phase, ring radius, and the rest. Run with
   ``sd_backplanes``; see :doc:`/user_guide/user_guide_backplanes`.

4. **PDS4 bundle generation.** Assemble the navigation results and the backplanes into a
   PDS4 bundle with labels, collections, and browse products, ready for archiving. Run
   with ``sd_create_bundle``; see :doc:`/user_guide/user_guide_pds4_bundle`.

Alongside the phases, SpinDoctor ships a set of reprojection and mosaic tools. They are
not a phase of the pipeline: they are available whenever you want to reproject navigated
images onto a ring radius/longitude grid or a body latitude/longitude grid, combine many
reprojections into one mosaic, and look at the result. Run them with ``sd_mosaic`` and
``sd_mosaic_display``; see :doc:`/user_guide/user_guide_reprojection`.

Installation
============

SpinDoctor needs Python 3.11 or later. Install it from the package index with ``pip``:

.. code-block:: bash

   pip install rms-spindoctor

This installs the library and every command-line program into your Python environment.

If you only want the command-line programs, install with ``pipx`` instead, which puts
them on your path in their own isolated environment:

.. code-block:: bash

   pipx install rms-spindoctor

Navigation also needs data that does not come with the package: the SPICE kernels for
the mission you are working with, the image files themselves, and a star catalog. Point
SpinDoctor at them with environment variables:

.. code-block:: bash

   export SPICE_PATH=/path/to/spice/kernels
   export PDS3_HOLDINGS_DIR=/path/to/pds3/holdings
   export UCAC4_PATH=/path/to/UCAC4

:doc:`/user_guide/user_guide_installation` covers the full list of environment
variables, the expected directory layouts, and where to get each kind of data.

Navigating Your First Image
===========================

The one command you need to see a result is ``sd_offset``. Give it the instrument, the
name of an image, where the images live, and where to write the results:

.. code-block:: bash

   sd_offset coiss N1294562056 \
     --pds3-holdings-root /path/to/pds3/holdings \
     --nav-results-root /path/to/nav_results

SpinDoctor reads the image, builds every model that applies to it, runs every navigation
technique that the image can support, reconciles their answers into one pointing
correction, and writes two files under the results root, in a subdirectory that mirrors
the image's place in the archive:

.. code-block:: text

   N1294562056_1_CALIB_metadata.json
   N1294562056_1_CALIB_summary.png

The PNG is a picture of the image with the models drawn on top of it, so you can see at
a glance whether the stars, limbs, and ring edges landed where SpinDoctor put them. The
metadata document is the machine-readable answer. Its ``navigation_result`` block holds
the essentials:

.. code-block:: json

   {
     "navigation_result": {
       "status": "success",
       "offset_px": [-1.87, 3.42],
       "sigma_px": [0.21, 0.19],
       "confidence": 0.86,
       "techniques_used": ["BodyLimbNav", "StarFieldFromCatalogNav"],
       "pointing": {
         "cmatrix": [0.4821, -0.8701, 0.0930, 0.8722, 0.4891, -0.0028,
                     -0.0782, 0.0505, 0.9957]
       }
     }
   }

``offset_px`` is how far the image moved, in pixels, relative to the SPICE kernels that
were furnished when the image was navigated; those kernels are named elsewhere in the
same metadata document, and without them the offset means nothing. ``cmatrix`` under
``pointing`` is the corrected pointing itself, and it is what downstream tools should
use.

:doc:`/user_guide/user_guide_navigation_outputs` describes everything the metadata
document contains, and :doc:`/user_guide/user_guide_metadata` documents every field in
it. If an image comes back refused or with a low confidence,
:doc:`/user_guide/user_guide_navigation_troubleshooting` explains what to look at.

Running More Than One Image
===========================

``sd_offset`` takes the same image-selection options as every other program in the
pipeline, so the same run can cover an archive volume, a range of times, or a whole
mission:

.. code-block:: bash

   sd_offset vgiss \
     --volumes VGISS_5101 \
     --pds3-holdings-root /path/to/pds3/holdings \
     --nav-results-root /path/to/nav_results

:doc:`/user_guide/user_guide_image_selection` describes every way to name the images a
run should process.

Nothing limits how many images a single run can process. For a mission with a very large
number of images you may still prefer to break the work into smaller chunks, so that you
can look at how each chunk turned out before starting the next one. That is a matter of
convenience, not a requirement. Voyager, for instance, is conventionally run one
planetary encounter at a time simply to keep each run small.

Where to Go Next
================

**Configuration.** Every threshold, tolerance, and choice of default that navigation
makes is configurable, and the settings you are most likely to change can also be given
on the command line or in the environment. :doc:`/user_guide/user_guide_configuration`
explains where settings come from and which source wins.

**Logging.** Each run writes a log for the run as a whole and a log for each image, and
you control how much detail each one gets. See :doc:`/user_guide/user_guide_logging`.

**Looking at many results at once.** ``sd_consolidate_metadata`` gathers the metadata
documents and summary PNGs from a results tree into one flat directory for easy browsing
(:doc:`/user_guide/user_guide_consolidate_metadata`). ``sd_results_index`` builds the
results index, a database holding one row per navigated image, so that later programs can
read a whole mission's results in bulk
(:doc:`/user_guide/user_guide_results_index`). ``sd_stats_report`` summarizes a navigation
run: how many images succeeded, which techniques carried them, how large the corrections
were, and how well the techniques agreed
(:doc:`/user_guide/user_guide_statistics`).

**Simulated images.** ``sd_create_simulated_image`` renders an image of a known geometry
with stars, bodies, and rings placed at a known offset, which lets you check what
navigation recovers against the truth that was planted.
See :doc:`/user_guide/user_guide_simulated_images`.

**Processing in the cloud.** Cloud tasks is a work-queue package supplied by the Green
Moon Systems node, distributed as ``rms-cloud-tasks`` and documented at
https://rms-cloud-tasks.readthedocs.io. It hands out batches of work to compute
instances and keeps track of which batches have been done. SpinDoctor's programs whose
names end in ``_cloud_tasks`` are the workers the cloud task system starts on those
instances; you never run one yourself. ``sd_offset``, ``sd_backplanes``, and
``sd_mosaic`` each write the task file their worker's queue is loaded from. See
:doc:`/user_guide/user_guide_cloud_tasks`.

**Per-instrument details.** Each supported instrument has its own cameras, calibration,
file formats, and quirks, described in
:doc:`/user_guide/instruments/instruments`.
