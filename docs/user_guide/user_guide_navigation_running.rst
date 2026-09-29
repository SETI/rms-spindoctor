======================
Command-Line Interface
======================

Basic Usage
-----------

The main entry point for SpinDoctor is the ``sd_offset`` script installed via ``pyproject.toml``. The basic syntax is:

.. code-block:: bash

   sd_offset DATASET_NAME [options]

Where ``DATASET_NAME`` is one of the supported names listed in the "Supported Missions" section. Names are case-insensitive (for example, ``COISS`` and ``coiss`` are equivalent).

An image's metadata document is written when the run completes that image, whatever status it records, and its summary PNG when the navigation produced one; each overwrites the file an earlier run left there. Nothing is deleted ahead of time, so a run that is interrupted partway through an image, or one that records an error and writes no PNG, leaves the earlier file in place. Start from an empty results directory when the absence of a document must mean the image was never navigated.

Command-Line Arguments
----------------------

The command-line interface groups options by purpose. Environment options control configuration sources and output roots; where the images themselves are read from belongs to the dataset, and is listed with its selection options below. Navigation options select which models or techniques to run. Output options determine whether to write artifacts locally or to produce a cloud-tasks description instead of processing. Dataset selection options are provided by each dataset type: PDS3 datasets expose volume and image filters. A single profiling toggle is available for performance analysis.

Environment options
^^^^^^^^^^^^^^^^^^^

* ``--config-file PATH`` (repeatable): one or more configuration file paths to
  override defaults. See :doc:`/introduction_configuration` for details.

* ``--nav-results-root PATH``: root directory or URL where navigation results
  will be written, overriding both the ``NAV_RESULTS_ROOT`` environment variable
  and any corresponding configuration setting.

* ``--results-index-db URL``: connection URL of a results index (a ``sqlite:``
  URL naming a local path, or a ``postgresql+psycopg:`` URL naming a server),
  overriding both the ``NAV_RESULTS_INDEX_DB`` environment variable and any
  corresponding configuration setting. The results-file selection filters below
  are then answered from the index's rows, and the results tree is not read.
  Pass ``--results-index-db none`` to name no index, and so read the tree, even
  when a URL is set in
  the environment or a configuration file; the opt-out is that word exactly, in
  lower case, with any surrounding spaces ignored, since any other non-empty
  value is read as the URL of an index. A value that is empty, or nothing but
  spaces, is refused: it is neither a connection URL nor the way to name no
  index, so the run stops and names the setting that carries it.

Navigation options
^^^^^^^^^^^^^^^^^^

* ``--nav-models LIST``: a comma-separated glob-pattern list selecting which
  ``NavModel`` instances run.  Names follow the ``stars`` /
  ``body:NAME`` / ``rings:PLANET`` convention.  Defaults to ``*``.  See
  :ref:`selecting-models-and-techniques` for the full syntax (globs,
  ``!`` exclusion, prefix-only shorthand).

* ``--nav-techniques LIST``: a comma-separated glob-pattern list selecting
  which registered ``NavTechnique`` subclasses run.  Defaults to ``*``.
  See :ref:`selecting-models-and-techniques` for the full syntax and the
  list of shipping technique class names.

Output options
^^^^^^^^^^^^^^

* ``--output-cloud-tasks-file PATH``: write a JSON file describing tasks for all selected images suitable for a cloud-tasks queue, and exit without performing navigation.
* ``--dry-run``: print the images that would be processed without performing navigation.
* ``--no-write-output-files``: perform navigation but do not write any output files.


Logging options
^^^^^^^^^^^^^^^

``sd_offset`` writes a main log reporting what the run is doing, and one log
per image carrying the detail of navigating it:

.. code-block:: text

   {log_root}/sd_offset/main_{timestamp}.log
   {log_root}/nav/{results_path_stub}_{timestamp}.log

``--log-root`` says where those go, defaulting to a ``logs`` directory under
the navigation results root. The main log goes to the terminal as well as a
file; image logs go to a file only, so the per-technique detail is on disk
rather than on screen unless ``--log-image-to-console`` asks for it.

The level of any one component can be raised or lowered on its own, which is
the usual way to investigate a single technique across many images:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 \
       --log-level WARNING --log-level titan_haze=DEBUG

The full set of options, the component names, the configuration-file
equivalents and the precedence between them are in :doc:`user_guide_logging`.

Miscellaneous
^^^^^^^^^^^^^

* ``--profile`` / ``--no-profile``: enable or disable runtime profiling (default is disabled).

Example Commands
----------------

To process a single Cassini image by specifying its name explicitly and using the default navigation technique:

.. code-block:: bash

   sd_offset coiss N1234567890

To process Voyager images within a single PDS3 volume:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101

To process a New Horizons image list found in a CSV from PDS, restricting the
run to the body-limb and ring-edge DT techniques:

.. code-block:: bash

   sd_offset nhlorri --image-filespec-csv /path/to/nhlorri.csv \
       --nav-techniques 'BodyLimbNav,RingEdgeNav'

To choose ten random Cassini images between two volumes and perform a dry run:

.. code-block:: bash

   sd_offset coiss --first-volume COISS_2001 --last-volume COISS_2010 --choose-random-images 10 --dry-run

To generate a cloud-tasks JSON file for images across two Voyager volumes without processing:

.. code-block:: bash

   sd_offset vgiss --volumes VGISS_5101 --volumes VGISS_5102 --output-cloud-tasks-file tasks.json

