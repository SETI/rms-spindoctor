=============
Configuration
=============

SpinDoctor uses a hierarchical YAML-based configuration system that allows you to
customize behavior without modifying the source code. Understanding how
configuration files are loaded and how to override settings is important for
effective use of the system.

Configuration Loading Order
============================

SpinDoctor ships with a complete set of built-in defaults, so the system works out
of the box with no configuration on your part. You customize behavior by supplying
your own settings on top of those defaults. Settings are resolved as follows:

1. **Built-in defaults**: SpinDoctor bundles a stack of default configuration files
   that give every setting a sensible value. You do not edit these. (Developers
   who need to know exactly which files ship and what each one holds should see
   :doc:`/dev_guide/dev_guide_config_and_static_data`.)

2. **Exactly one of the following** is loaded on top of the built-in defaults:

   * **Command-line configuration files**: If one or more files are specified
     with the ``--config-file`` option, they are loaded in the order given,
     each overriding the built-in defaults (and, for the same key, any file
     loaded before it).

   * **User default configuration**: Only when no ``--config-file`` option is
     given, a file named ``nav_default_config.yaml`` in the current working
     directory is loaded if it exists. Use it to set personal defaults for
     runs where you do not pass ``--config-file``. Note that passing
     ``--config-file`` replaces this file entirely rather than adding to it;
     to keep your personal defaults in such a run, list
     ``nav_default_config.yaml`` explicitly as the first ``--config-file``
     argument.

3. **Command-line option overrides**: A handful of CLI flags (described under
   `Command-Line Option Overrides`_ below) override the matching configuration
   key directly and take precedence over everything above.

You only ever need to specify the settings you want to change; everything else
falls through to the built-in defaults.

Configuration File Structure
============================

Configuration files use YAML format and are organized into sections:

.. code-block:: yaml

   environment:
     nav_results_root: /path/to/results
     pds3_holdings_root: /path/to/pds3
     results_index_db: sqlite:////path/to/results/index.sqlite3

   logging:
     models:
       stars: DEBUG
       rings: DEBUG

   offset:
     correlation_fft_upsample_factor: 128
     star_refinement_enabled: true

   bodies:
     min_bounding_box_area: 9
     oversample_maximum: 2

Each section can contain multiple settings. When multiple configuration files
define the same setting, the value from the last file loaded takes precedence.

Logging Configuration
---------------------

Logging is configured by the top-level ``logging`` section, described under
`Logging Options`_ below, together with command-line options that override it.
It is one of the three sections excluded from the provenance configuration
digest recorded with each navigation result: what a run wrote down about itself
cannot change what it concluded, so two results differing only in logging were
produced by the same configuration and compare as such. The others are
``environment``, which says where a deployment keeps its files rather than how
it navigates, and ``results_tree``, which says how many requests a pass over a
navigation results tree makes at once.

Two loggers write during a run: the main logger, covering one program run, and
the image logger, covering one image inside one processing stage. A component
can be given its own level, so one technique or model can be made verbose or
quiet without affecting the rest. For the full component list, where the log
files are written, and the precedence between the configuration and the
command line, see :doc:`/user_guide/user_guide_logging`.

Example -- enable verbose output for star and ring models while keeping other
components at the default level:

.. code-block:: yaml

   logging:
     models:
       stars: DEBUG
       rings: DEBUG

Creating a User Configuration File
===================================

To create your own default configuration:

1. Create a file named ``nav_default_config.yaml`` in your working directory
2. Add only the settings you want to override:

   .. code-block:: yaml

      environment:
        nav_results_root: /my/custom/results/path

      offset:
        correlation_fft_upsample_factor: 256

3. The system will automatically load this file if it exists, provided you do
   not pass ``--config-file`` (which replaces it; see below)

Using Command-Line Configuration Overrides
===========================================

You can override configuration on a per-run basis using ``--config-file``:

.. code-block:: bash

   sd_offset coiss N1234567890 --config-file /path/to/special_config.yaml

You can specify multiple configuration files, and they will be loaded in order:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file base_overrides.yaml \
     --config-file run_specific.yaml

When any ``--config-file`` is given, ``nav_default_config.yaml`` is not loaded
automatically. To keep your personal defaults for that run, pass the file
explicitly as the first ``--config-file`` argument:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file nav_default_config.yaml \
     --config-file run_specific.yaml

Command-Line Option Overrides
==============================

In addition to configuration files, certain command-line options can override
configuration settings directly. These options take precedence over all
configuration file settings:

Environment Options
-------------------

* ``--pds3-holdings-root PATH``: Overrides the ``environment.pds3_holdings_root``
  configuration setting and the ``PDS3_HOLDINGS_DIR`` environment variable, in
  that order. This specifies the root directory or URL for PDS3 holdings. It is
  offered by the dataset rather than by each program, so it appears among a
  program's dataset-selection options and only when the dataset named on the
  command line reads a PDS3 holdings tree.

* ``--nav-results-root PATH``: Overrides the ``NAV_RESULTS_ROOT`` environment
  variable and any ``environment.nav_results_root`` configuration setting. This
  specifies the root directory or URL where navigation results will be written.

* ``--results-index-db URL``: Overrides the ``NAV_RESULTS_INDEX_DB`` environment
  variable and any ``environment.results_index_db`` configuration setting. This
  names the results index --- a database derived from the navigation results
  tree by a separate ingest step --- as a ``sqlite:`` URL naming a local file or
  a ``postgresql+psycopg:`` URL naming a server. Unset means no index, which is
  the default for every program that offers the option; the literal value
  ``none`` says so explicitly and overrides a URL set elsewhere. A value that is
  empty, or nothing but spaces, is neither, and is refused where it is spelled.
  Only a program that offers the option reads one:
  ``environment.results_index_db`` and ``NAV_RESULTS_INDEX_DB`` do not make a
  program index-backed that does not declare ``--results-index-db``. See
  :doc:`/user_guide/user_guide_results_index`.

Navigation Options
------------------

* ``--nav-models LIST``: Overrides any default model selection. This is a
  comma-separated list of model names or patterns to enable. Valid entries
  include ``stars``, ``rings``, ``titan``, and body-specific entries of the
  form ``body:NAME`` (glob patterns are allowed).

* ``--nav-techniques LIST``: Overrides any default technique selection. This
  is a comma-separated list of glob patterns matched against the registered
  technique names: ``BodyBlobNav``, ``BodyDiscCorrelateNav``, ``BodyLimbNav``,
  ``BodyTerminatorNav``, ``RingAnnulusNav``, ``RingEdgeNav``,
  ``StarFieldFromCatalogNav``, ``StarRefineNav``, ``StarUniqueMatchNav``, and
  ``TitanHazeNav``.
  Shell-glob wildcards are allowed (``Star*`` selects the three star
  techniques) and a leading ``!`` excludes matching names (``!Ring*`` runs
  everything except the ring techniques). Interactive manual navigation is
  not selected here; it is invoked with the separate ``--manual`` flag, which
  opens the manual-navigation dialog instead of running the autonomous
  pipeline.

Logging Options
---------------

Logging is configured by the ``logging`` section and by command-line options
that override it. Levels are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
``CRITICAL`` and ``NONE``.

.. code-block:: yaml

    logging:
      main: INFO            # the run's logger
      image: INFO           # per-image logs, and any component not named below
      main_console: true    # whether the run's log reaches the terminal
      image_console: false  # whether per-image logs do
      techniques:
        titan_haze: DEBUG   # one technique
      models:
        rings: WARNING      # one model family
      other:
        annotate: ERROR
      programs:
        sd_mosaic:          # applies to that program only
          main: WARNING

A component named anywhere takes its own level; a category's ``default``
applies to the rest of that category; otherwise the per-logger default
applies. An unrecognized component or program name is rejected when the
configuration loads, rather than being silently ignored.

The command-line options are ``--log-root``, ``--log-level`` (bare for both
loggers, or ``MODULE=LEVEL`` for one component, repeatable),
``--log-level-main``, ``--log-level-image``, and the four sink switches
``--log-main-to-console``, ``--log-main-to-file``, ``--log-image-to-console``
and ``--log-image-to-file``, each with a ``--no-`` form. A program that does
not process images individually accepts only the main-logger options, and the
cloud-task drivers accept none: see :doc:`/user_guide/user_guide_logging`.

``--log-root`` takes precedence over every configuration file, including one
named with ``--config-file``, and over the ``NAV_LOG_ROOT`` environment
variable. So do ``--log-main-to-console`` and ``--log-image-to-console``, over
the ``main_console`` and ``image_console`` keys.

``--log-main-to-file`` and ``--log-image-to-file`` have no configuration
equivalent: whether a log file is written is inseparable from where it goes,
and that is chosen per run. There is no ``main_file`` or ``image_file``
setting, and writing one is an error rather than a line that does nothing.

The level options are ranked by how specifically they name their target, not
by being on the command line, so the order above governs them: ``--log-level
MODULE=LEVEL`` outranks everything, but a component named in a configuration
file outranks a bare ``--log-level``, which says nothing about that component.
``--log-level DEBUG`` therefore does not lift a component the configuration
pinned; name it, as in ``--log-level titan_haze=DEBUG``.

Example: Combining Configuration Methods
========================================

The following example demonstrates how different configuration methods interact.
Suppose the built-in defaults set
``offset.correlation_fft_upsample_factor: 128``, your
``nav_default_config.yaml`` sets it to ``256``, and ``custom.yaml`` sets it to
``512``:

1. Running ``sd_offset`` with no ``--config-file`` loads
   ``nav_default_config.yaml``, so the final value is ``256``.

2. Running ``sd_offset --config-file custom.yaml`` does not load
   ``nav_default_config.yaml`` at all, so the final value is ``512`` -- and
   every other setting in ``nav_default_config.yaml`` also reverts to its
   built-in default.

3. To combine the two, list both files explicitly:
   ``sd_offset --config-file nav_default_config.yaml --config-file custom.yaml``
   loads them in order, so the final value is ``512`` while the rest of your
   personal defaults still apply.

If you also specify ``--nav-models stars,rings`` on the command line, this
overrides any model selection from configuration files, regardless of what's in
the configuration.
