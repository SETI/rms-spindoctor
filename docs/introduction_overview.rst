========
Overview
========

SpinDoctor is a spacecraft image navigation system designed to analyze images from
various space missions and determine precise positional offsets. This overview
provides an introduction to the system architecture, installation, and
command-line tools.

Navigation Pipeline
===================

SpinDoctor follows a three-phase pipeline for processing spacecraft imagery:

1. **Navigation** - Determine pointing offsets by correlating observed images
   with theoretical models of stars, planets, moons, and rings.

2. **Backplanes** - Generate geometric and photometric backplanes (derived
   image products) that provide per-pixel information about the observation
   geometry.

3. **PDS4 Bundle** - Create PDS4-compliant data bundles containing navigation
   results, backplanes, and metadata for archival and distribution.

Each phase builds upon the previous one, with navigation results informing
backplane generation, and both contributing to the final PDS4 bundle.

Installation
============

SpinDoctor can be installed using either ``pip`` or ``pipx``:

Using pip
---------

.. code-block:: bash

   pip install rms-spindoctor

This installs the package and all command-line programs into your Python
environment.

Using pipx
----------

.. code-block:: bash

   pipx install rms-spindoctor

This creates isolated command-line programs that can be run independently of
your Python environment. This is recommended if you want the command-line tools
available system-wide without managing Python dependencies.

Command-Line Programs
=====================

SpinDoctor provides command-line programs that correspond to each phase of the
navigation pipeline:

Navigation Phase
----------------

* ``sd_offset`` - Perform navigation on spacecraft images, determining pointing
  offsets by correlating observed features with theoretical models.

* ``sd_create_simulated_image`` - Create simulated images with stars, bodies,
  and rings, used internally to test and validate the navigation pipeline (see
  the developer guide's :doc:`/dev_guide/dev_guide_simulator` chapter).

* ``sd_consolidate_metadata`` - Copy each image's metadata JSON and/or summary
  PNG to a single flat directory so results are easy to browse without
  descending the per-volume path hierarchy (see
  :doc:`/user_guide/user_guide_navigation`).

* ``sd_results_index`` - Read per-image navigation metadata JSON files into the
  results index (see :doc:`/user_guide/user_guide_results_index`).

* ``sd_stats_report`` - Generate success/failure, technique-usage, offset, and
  agreement reports from a navigation results tree or the results index (see
  :doc:`/user_guide/user_guide_statistics`).

Reprojection and Mosaic Phase
-----------------------------

* ``sd_mosaic`` - Reproject navigated images and combine them into ring or
  body mosaics; also installed as the ``sd_mosaic_rings`` /
  ``sd_mosaic_body`` entry points (see
  :doc:`/user_guide/user_guide_reprojection`).

* ``sd_mosaic_display`` - Interactive viewer for reprojection and mosaic
  files; also installed as the ``sd_mosaic_display_rings`` /
  ``sd_mosaic_display_body`` entry points.

Backplanes Phase
----------------

* ``sd_backplanes`` - Generate geometric and photometric backplanes for
  spacecraft images.

* ``sd_backplane_viewer`` - Interactive viewer for examining backplane data.

PDS4 Bundle Phase
-----------------

* ``sd_create_bundle`` - Create PDS4-compliant data bundles containing
  navigation results, backplanes, and metadata. Supports label generation,
  summary creation, and checking a bundle that has been written.

Cloud Tasks Support
===================

SpinDoctor supports queue-driven processing through cloud tasks for scalable,
distributed processing:

* ``sd_offset_cloud_tasks`` - Cloud tasks worker for navigation processing.

* ``sd_backplanes_cloud_tasks`` - Cloud tasks worker for backplane generation.

* ``sd_create_bundle_cloud_tasks`` - Cloud tasks worker for PDS4 bundle
  creation.

* ``sd_mosaic_cloud_tasks`` - Cloud tasks worker for the reprojection pass
  of ring and body mosaic generation. A single worker process handles both
  ring and body tasks; the mode is encoded per-task in the task payload.
  (Mosaic combination remains a single-node step; see
  :doc:`/user_guide/user_guide_reprojection`.)

* ``sd_results_index_cloud_tasks`` - Cloud tasks worker that reads one share of
  a navigation-results root into the results index. ``sd_results_index divide``
  lists each root and divides it into shares, and ``sd_results_index complete``
  adds the workers' tallies up again when they have run; see :doc:`/user_guide/user_guide_results_index`.

These cloud tasks variants read task payloads from a queue and process batches
of files, making them suitable for large-scale processing in cloud
environments. The local batch drivers ``sd_offset``, ``sd_backplanes``, and
``sd_mosaic_rings`` / ``sd_mosaic_body`` can emit a cloud-tasks JSON file
for their respective workers via ``--output-cloud-tasks-file PATH``; see the
matching user guide for each driver's JSON schema.
