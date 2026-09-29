==================
Running sd_offset
==================

``sd_offset`` is the navigation program. One run selects a set of images, navigates each
of them, and writes a metadata document and a summary image for each. This chapter
covers how to invoke it and what every option does.

Basic Usage
===========

.. code-block:: bash

   sd_offset DATASET_NAME [options]

``DATASET_NAME`` is the first argument and is required. It names the mission and
instrument whose images the run reads, and is one of the names listed under "Supported
Missions" in :doc:`user_guide_introduction`. Names are case-insensitive, so ``COISS`` and
``coiss`` are equivalent.

Everything else is an option. Which options a run accepts depends in part on the dataset
named, because each dataset offers the selection options that suit its archive. Asking
for the options of the dataset you intend to use shows exactly what that run accepts:

.. code-block:: bash

   sd_offset coiss --help

The simplest useful run navigates one image whose name you already know:

.. code-block:: bash

   sd_offset coiss_saturn N1466448128

Choosing Which Images to Navigate
=================================

A run that has no selection options navigates every image the dataset offers, which for a
whole mission is a great many. The options that narrow the selection -- by volume, by
image number, by name, from a list in a file, at random, or by what a previous run
already recorded -- are shared with several other programs and are documented together
in :doc:`user_guide_image_selection`.

Two of them are worth knowing before anything else. ``--volumes`` bounds a run to named
PDS3 volumes, and ``--has-no-offset-file`` keeps only the images that have no result yet,
which is what makes an interrupted run resumable:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 --has-no-offset-file

Environment Options
===================

These say where configuration and results live.

``--config-file PATH``
  A configuration file whose settings override the defaults. May be given more than
  once, in which case each file is applied in turn. When no file is named, a
  ``nav_default_config.yaml`` in the current directory is loaded if there is one. See
  :doc:`user_guide_configuration`.

``--nav-results-root PATH``
  The root directory or URL the navigation results are written under. Overrides the
  ``environment.nav_results_root`` configuration setting and the ``NAV_RESULTS_ROOT``
  environment variable. A run must have this from one of the three, since there is no
  built-in default.

``--results-index-db URL``
  The connection URL of a results index: a ``sqlite:`` URL naming a local file, or a
  ``postgresql+psycopg:`` URL naming a server. Overrides the
  ``environment.results_index_db`` configuration setting and the
  ``NAV_RESULTS_INDEX_DB`` environment variable. Given a results index, the selection
  options that ask what a previous run recorded are answered from its rows instead of by
  reading the results tree; see :doc:`user_guide_image_selection` for what that changes
  and :doc:`user_guide_results_index` for the results index itself.

  Pass ``--results-index-db none`` to read the results tree even when a URL is configured
  elsewhere. See :doc:`user_guide_configuration`.

Navigation Options
==================

These control how each image is navigated.

``--nav-models LIST``
  A comma-separated list of glob patterns selecting which models are built. Model names
  follow the ``stars`` / ``body:NAME`` / ``rings:PLANET`` / ``titan:NAME``
  convention, and a bare prefix such as ``rings`` selects every model under it. Defaults
  to ``*``, which is every model that applies to the image. See
  :doc:`user_guide_navigation_models` for the full syntax, including exclusion with
  ``!`` and the prefix-only shorthand.

``--nav-techniques LIST``
  A comma-separated list of glob patterns selecting which techniques run. Defaults to
  ``*``, which is every technique the models make feasible. The technique names and the
  pattern syntax are in :doc:`user_guide_navigation_models`.

``--manual``
  Open the interactive manual-navigation window instead of navigating automatically, and
  place the offset by hand. The selection must resolve to exactly one image. See
  "Manual Navigation" in :doc:`user_guide_navigation_models`.

Output Options
==============

``--output-cloud-tasks-file PATH``
  Write a JSON file describing one task per selected image, suitable for loading into a
  cloud task queue, and do no other processing. The queue is what later runs the images;
  see :doc:`user_guide_cloud_tasks`.

``--dry-run``
  Print the images the selection resolves to and stop. Nothing is navigated and nothing
  is written. This is the way to check a selection before committing a long run to it.

``--no-write-output-files``
  Navigate the images but write no metadata documents or summary images. Results still
  appear in the logs.

Logging Options
===============

A run writes a main log reporting what it is doing, and one log per image carrying the
detail of navigating that image. ``--log-root`` says where those files go, defaulting to
a ``logs`` directory under the navigation results root. The main log also goes to the
terminal; per-image logs go only to their files, unless asked otherwise.

The flags are:

* ``--log-root PATH``
* ``--log-level LEVEL`` and ``--log-level MODULE=LEVEL``, repeatable
* ``--log-level-main LEVEL`` and ``--log-level-image LEVEL``
* ``--log-main-to-console`` / ``--no-log-main-to-console``
* ``--log-main-to-file`` / ``--no-log-main-to-file``
* ``--log-image-to-console`` / ``--no-log-image-to-console``
* ``--log-image-to-file`` / ``--no-log-image-to-file``

Raising or lowering one module on its own is the usual way to investigate a single
technique across many images:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 \
       --log-level WARNING --log-level titan_haze=DEBUG

The accepted levels, the module names, the file names, the configuration-file
equivalents, and the precedence between them are all in :doc:`user_guide_logging`.

Miscellaneous Options
=====================

``--profile`` / ``--no-profile``
  Collect a Python profile of where the run spends its time. Disabled by default;
  ``--no-profile`` is the explicit way to say so. The profile is collected in memory and
  is neither printed nor written to a file, so the flag produces no report of its own.

What a Run Writes
=================

Each image's metadata document is written when the run finishes with that image,
whatever outcome it records, and its summary image is written when the navigation
produced one. Each overwrites whatever an earlier run left in its place, and nothing is
deleted in advance.

An earlier run's summary image can therefore survive beside a fresh metadata document,
which happens when the new run records an error and produces no summary image of its own.
For the same reason, both files of an earlier run survive a run interrupted partway
through an image. Start from an empty results directory when the absence of a metadata
document has to mean the image was never navigated.

The metadata documents themselves are described in
:doc:`user_guide_navigation_outputs` and :doc:`user_guide_metadata`.

Example Commands
================

Navigate one Cassini image by name:

.. code-block:: bash

   sd_offset coiss N1466448128

Navigate every image in one Voyager volume:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101

Navigate a list of New Horizons images taken from a CSV file published by PDS, using only
the body-limb and ring-edge techniques:

.. code-block:: bash

   sd_offset nhlorri --image-filespec-csv /path/to/nhlorri.csv \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

Check which ten random Cassini images a range of volumes would yield, without navigating
any of them:

.. code-block:: bash

   sd_offset coiss --first-volume COISS_2001 --last-volume COISS_2010 \
       --choose-random-images 10 --dry-run

Resume a long run, navigating only the images it has not reached yet:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 --has-no-offset-file

Re-navigate the images a previous run could not navigate for want of SPICE data, after
furnishing the missing kernels:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 --has-offset-spice-error

Describe the work for a cloud task queue instead of doing it:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101,VGISS_5102 \
       --output-cloud-tasks-file tasks.json
