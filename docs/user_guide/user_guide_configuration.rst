=============
Configuration
=============

Every SpinDoctor program reads the same body of settings. They control where files are
read and written, which models and techniques navigation uses, how hard each technique
works, how much a result must prove before it is accepted, and how much the programs write
to their logs. The settings are YAML, and you change them without touching any source
code.

This chapter is the full account of that system: where settings come from, which source
wins when two disagree, and which of them have command-line equivalents. The
:doc:`/quick_start` shows the short version.

Where settings come from
========================

SpinDoctor ships with a complete set of built-in defaults, so it works out of the box with
no configuration on your part. You customize behavior by supplying your own settings on
top of those defaults, and you only ever write down the settings you want to change.
Everything else falls through.

Settings are resolved in this order, each step overriding the ones above it:

1. **Built-in defaults.** A stack of configuration files ships inside SpinDoctor and gives
   every setting a value. You cannot edit these: they live inside the installed package,
   and a change to them would be undone by the next upgrade.

2. **Exactly one of the following**, loaded on top of the built-in defaults:

   * **Files named with** ``--config-file``. One or more files, loaded in the order given.
     Each overrides the built-in defaults and, for any setting they both name, any file
     given before it.

   * **Your own default file.** Only when no ``--config-file`` is given, a file named
     ``nav_default_config.yaml`` in the current working directory is loaded if it exists.
     This is where personal defaults belong.

   Note that these two are alternatives, not layers. Passing ``--config-file`` replaces
   ``nav_default_config.yaml`` rather than adding to it. To keep your personal defaults in
   such a run, name that file explicitly as the first ``--config-file`` argument.

3. **Command-line options.** A number of options set one configuration value directly, and
   take precedence over every file. They are listed under `Options that override
   configuration`_ below.

Some settings also have an environment variable, which is consulted last, after both
the command line and the configuration files. Exporting one does not override a value a
configuration file already sets. Each is named with the option it belongs to below.

The built-in defaults are in ``src/spindoctor/config_files/`` of the installed package,
loaded in filename order; that is where a developer reading the source finds them, and
:doc:`/dev_guide/dev_guide_config_and_static_data` describes what each one holds.

How a configuration file is written
===================================

A configuration file is YAML, organized into top-level sections. Each section groups the
settings for one part of the system:

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

The sections are:

``environment``
    Where a deployment keeps its files: the holdings root that images are read from, the
    roots that navigation results, backplanes, and PDS4 bundles are written to, the log
    root, and the results index.

``general``, ``planets``, ``satellites``
    Which planets and moons SpinDoctor knows about, and how each is grouped. Change these
    to add a moon that is not modeled, or to leave one out.

``logging``
    How much each program writes to its logs, and which components it writes about. See
    :doc:`/user_guide/user_guide_logging`.

``offset``
    The shared machinery every technique uses, including how finely the image correlation
    is upsampled and whether star positions are refined after a first fit.

``stars``, ``bodies``, ``rings``, ``titan``
    What goes into the model of each kind of subject, and how the matching treats it:
    which star catalogs are consulted and in what order, how small a moon may be and
    still be measured, where the ring radii come from, and how thick the atmosphere of a
    hazy body is taken to be.

``body_shape``
    The dimensions, surface roughness, and brightness assumed for each body, which is what
    decides how closely a modeled limb can be expected to match the real one.

``techniques``
    Per-technique settings: how much work each technique does, and how it turns what it
    measured into a confidence.

``orchestrator``
    How the per-technique answers are combined into the one offset reported for the image,
    and how good that answer must be to be accepted at all. The acceptance thresholds live
    here; see `Acceptance thresholds`_ below.

``cassini_iss``, ``voyager_iss``, ``galileo_ssi``, ``newhorizons_lorri``
    Per-instrument settings. :doc:`/user_guide/instruments/instruments` describes each
    instrument and what is particular to it.

``backplanes``, ``pds4``
    Which backplanes are computed, and how a PDS4 bundle is assembled.

``results_tree``
    How many requests a pass over a navigation results tree makes at once. Worth tuning
    for a results root on cloud storage, and best left alone for a local one.

``sim``
    The camera the image simulator pretends to be, and the noise and defects it renders.
    See :doc:`/user_guide/user_guide_simulated_images`.

When two configuration files name the same setting, the value from the last file loaded
wins. Sections merge setting by setting, so naming one setting in a section leaves the
rest of that section at its default.

Writing your own default file
=============================

To set defaults for every run you make from a given directory:

1. Create a file named ``nav_default_config.yaml`` there.

2. Write down only the settings you want to change:

   .. code-block:: yaml

      environment:
        nav_results_root: /my/custom/results/path

      offset:
        correlation_fft_upsample_factor: 256

3. Run any program without ``--config-file``. The file is found and loaded.

Naming a file per run
=====================

``--config-file PATH`` names a configuration file for one run. Every program accepts it;
``sd_offset`` (:doc:`/user_guide/user_guide_navigation_running`) is used for the examples
here:

.. code-block:: bash

   sd_offset coiss N1234567890 --config-file /path/to/special_config.yaml

The option is repeatable, and the files are loaded in the order given:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file base_overrides.yaml \
     --config-file run_specific.yaml

Because ``--config-file`` replaces ``nav_default_config.yaml`` rather than adding to it,
keeping your personal defaults for such a run means naming that file first:

.. code-block:: bash

   sd_offset coiss N1234567890 \
     --config-file nav_default_config.yaml \
     --config-file run_specific.yaml

Options that override configuration
===================================

These command-line options each set one configuration value, and take precedence over
every configuration file.

Environment options
-------------------

``--pds3-holdings-root PATH``
    The root directory or URL of the PDS3 holdings that images are read from. Overrides
    the ``PDS3_HOLDINGS_DIR`` environment variable and the
    ``environment.pds3_holdings_root`` setting. The dataset offers this option rather than
    the program, so it appears among a program's image-selection options, and only when
    the dataset named on the command line reads a PDS3 holdings tree at all.

``--nav-results-root PATH``
    The root directory or URL that navigation results are written to. Overrides the
    ``NAV_RESULTS_ROOT`` environment variable and the ``environment.nav_results_root``
    setting.

``--results-index-db URL``
    The results index, a database with one row per navigated image that a separate step
    builds from the navigation results tree. Give a ``sqlite:`` URL naming a local file,
    or a ``postgresql+psycopg:`` URL naming a server. Overrides the
    ``NAV_RESULTS_INDEX_DB`` environment variable and the ``environment.results_index_db``
    setting.

    No results index is the default for every program that offers this option, and the
    literal value ``none`` says so explicitly, overriding a URL set anywhere else. A value
    that is empty or nothing but spaces is neither a URL nor ``none``, and is refused
    where it is written. Only a program that offers this option reads a results index at
    all: setting ``environment.results_index_db`` or ``NAV_RESULTS_INDEX_DB`` does not
    give a results index to a program that does not offer ``--results-index-db``. See
    :doc:`/user_guide/user_guide_results_index`.

Navigation options
------------------

``--nav-models LIST``
    Which models to build, as a comma-separated list of names or patterns. A model is
    named ``stars``, ``body:NAME``, ``rings:PLANET``, or ``titan:NAME``, and shell-glob
    wildcards are allowed. Writing a bare prefix selects every model under it, so
    ``rings`` means the same as ``rings:*``.
    :doc:`user_guide_navigation_models` lists the names in full. Overrides any model
    selection from a configuration file.

``--nav-techniques LIST``
    Which techniques to run, as a comma-separated list of glob patterns matched against
    the technique names: ``BodyBlobNav``, ``BodyDiscCorrelateNav``, ``BodyLimbNav``,
    ``BodyTerminatorNav``, ``RingAnnulusNav``, ``RingEdgeNav``,
    ``StarFieldFromCatalogNav``, ``StarRefineNav``, ``StarUniqueMatchNav``, and
    ``TitanHazeNav``. Wildcards are allowed, so ``Star*`` selects the three star
    techniques, and a leading ``!`` excludes what it matches, so ``!Ring*`` runs
    everything except the ring techniques. Overrides any technique selection from a
    configuration file.

    Interactive manual navigation is not selected here. It is invoked with the separate
    ``--manual`` flag, which opens the manual-navigation dialog instead of running the
    autonomous pipeline.

:doc:`/user_guide/user_guide_navigation_models` describes the models and techniques
themselves.

Logging options
---------------

Logging is set by the ``logging`` section and by command-line options that override it.
The options are ``--log-root``, ``--log-level`` (bare for both kinds of log, or
``MODULE=LEVEL`` for one component, repeatable), ``--log-level-main``,
``--log-level-image``, and the four switches ``--log-main-to-console``,
``--log-main-to-file``, ``--log-image-to-console``, and ``--log-image-to-file``, each of
which also has a ``--no-`` form.

``--log-main-to-file`` and ``--log-image-to-file`` are the two with no configuration
equivalent. All the rest correspond to a key in the ``logging`` section.

:doc:`/user_guide/user_guide_logging` is the full account: the levels, the component
names, the keys of the ``logging`` section, and which setting wins when two of them name
the same component.

Acceptance thresholds
=====================

Navigation reports a **confidence** with every answer: a number between 0 and 1 saying how
much the evidence in that frame supports the offset it found. A frame with a dozen matched
stars earns a high confidence. A frame with one faint blur at the edge earns a low one. An
answer whose confidence is too low is not reported as a success at all; the frame is
refused and the reason is recorded.

Two settings in the ``orchestrator`` section decide that:

``orchestrator.ensemble.min_confidence``
    The lowest confidence an answer may have and still be accepted. Below it the frame is
    refused.

``orchestrator.ensemble.tier_thresholds``
    The boundaries of the three tiers you can set thresholds for -- high, medium,
    and low. Two further ranks, conflicted and failed, are outcomes rather than
    thresholds, so they have no entry here;
    :doc:`user_guide_navigation_models` describes all five. The boundaries a
    reported answer is sorted into. Each tier names the confidence an answer must reach
    and the largest pointing uncertainty, in pixels, it may have. An answer must satisfy
    both to earn that tier.

Lowering these makes more frames report an answer and makes those answers less
trustworthy, so change them only when you know why the evidence in your frames is weaker
than the defaults assume. The confidence and the tier that navigation settled on for each
image are recorded in that image's metadata document; see
:doc:`/user_guide/user_guide_metadata`.

What a run records about its own configuration
==============================================

Each navigation result records a digest of the configuration that produced it, so two
results can be told apart when they were navigated under different settings. Two results
that differ only in their ``logging``, ``environment``, or ``results_tree`` settings
compare as identical.

Worked example
==============

Suppose the built-in defaults set ``offset.correlation_fft_upsample_factor`` to ``128``,
your ``nav_default_config.yaml`` sets it to ``256``, and ``custom.yaml`` sets it to
``512``.

1. ``sd_offset`` run without ``--config-file`` loads ``nav_default_config.yaml``, so the
   value is ``256``.

2. ``sd_offset --config-file custom.yaml`` does not load ``nav_default_config.yaml`` at
   all, so the value is ``512`` -- and every other setting in ``nav_default_config.yaml``
   reverts to its built-in default too.

3. ``sd_offset --config-file nav_default_config.yaml --config-file custom.yaml`` loads
   both, in that order, so the value is ``512`` while the rest of your personal defaults
   still apply.

Adding ``--nav-models stars,rings`` to any of the three selects those two models
regardless of what the configuration files say, because a command-line override outranks
them all.
