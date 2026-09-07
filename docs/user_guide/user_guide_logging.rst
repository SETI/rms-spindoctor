=======
Logging
=======

Every SpinDoctor pipeline program writes two kinds of record, and keeps them
apart. This page covers what goes where, how to change it, and where the files
land.

The two loggers
===============

**The main log** covers one run of one program. It reports what the program is
doing at the top level: which image it is about to process, one line for each
image's answer, counts and totals, elapsed time, and the path of each image log
it wrote. There is one for the life of the run.

**An image log** covers one image inside one processing stage. It carries the
detail of that image's processing -- which models were built, which techniques
ran, what each of them found. A new one is started for each image.

A record belongs to exactly one of them, so the two never repeat each other.
When you want to know what a run did, read the main log; when you want to know
what happened to one image, read that image's log.

Not every program has both. A program that does not process images
individually has only a main log; the statistics report and the GUI programs
have neither -- they write to the terminal directly, because both are read as
they run rather than afterwards.

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Program
     - Main log
     - Image log
   * - ``sd_offset``
     - yes
     - ``nav``
   * - ``sd_backplanes``
     - yes
     - ``backplanes``
   * - ``sd_mosaic`` (and ``sd_mosaic_rings`` / ``sd_mosaic_body``)
     - yes
     - ``reproj``
   * - ``sd_create_ck``
     - yes
     - ``ck``
   * - ``sd_create_bundle``
     - yes
     - none
   * - ``sd_consolidate_metadata``
     - yes
     - none
   * - ``sd_offset_cloud_tasks``
     - no
     - ``nav``
   * - ``sd_backplanes_cloud_tasks``
     - no
     - ``backplanes``
   * - ``sd_mosaic_cloud_tasks``
     - no
     - ``reproj``
   * - ``sd_results_index``
     - yes
     - none
   * - ``sd_results_index_cloud_tasks``
     - no
     - none
   * - ``sd_stats_report``
     - no
     - none
   * - ``sd_create_simulated_image``
     - no
     - none
   * - ``sd_backplane_viewer``, ``sd_mosaic_display``
     - no
     - none

Where the files go
==================

Both kinds live under one log root, named by ``--log-root``, the
``environment.log_root`` configuration variable or the ``NAV_LOG_ROOT``
environment variable, in that order of precedence.

With none of those set, the root is derived: a ``logs`` directory under the
navigation results root. A cloud-task worker is not required to have a
navigation results root, so each falls back to a ``logs`` directory under the
root it does have -- the backplane results root for
``sd_backplanes_cloud_tasks``, and the task's own output directory for
``sd_mosaic_cloud_tasks`` -- rather than dropping its logs for want of a
setting that does not apply to it.

A local root given as a relative path is resolved against the working directory
once, at startup, and the absolute result is what every log file of that run is
written under. A run therefore keeps writing to the same place even if
something later changes the working directory.

.. code-block:: text

   {log_root}/{program}/main_{timestamp}.log
   {log_root}/{backend}/{results_path_stub}_{timestamp}.log

The main log is filed under the program that wrote it. An image log is filed
under the *stage* rather than the program, so an image's navigation log sits
beside every other navigation log whether an interactive run or a cloud task
produced it. The four stages are ``nav``, ``backplanes``, ``reproj`` and
``ck``.

The timestamp is UTC, in ``YYYY-MM-DDTHH-MM-SS`` form, and is taken once when
logging is set up rather than once per file. What that groups depends on which
driver is running:

* **A program you run yourself** stamps once at startup, so every file it
  writes -- its main log and every image log -- carries the same timestamp.
  That is what makes a run identifiable after the fact: the logs of one
  ``sd_offset`` pass over five hundred images sort together and are
  distinguishable from the pass before it.

* **A cloud-task worker** stamps once per task, because that is where logging
  is set up. There is no run-wide moment to share: tasks are handed out to
  whatever workers are free, on machines that started at different times and
  may not have been running at all when the batch began. Each image's log
  therefore carries the time its own task was picked up.

UTC rather than local time is what makes the two comparable. Workers may sit
in different zones, and a local-time name would be ambiguous across a
daylight-saving fall-back; in UTC the names of an interactive run and of every
worker in a batch sort into one order.

.. note::

   The timestamp in the file *name* is UTC; the timestamps on the records
   *inside* are local. A log named ``..._2026-07-31T02-36-04.log`` can open
   with ``2026-07-30 19:36:04``. Match a log to a wall-clock time by its
   contents rather than its name, and glob by name only in UTC terms.

Reprojection logs are keyed by mosaic subject as well, since one image may be
reprojected onto more than one body::

   {log_root}/reproj/{subject}/{results_path_stub}_{timestamp}.log

.. note::

   Reprojection logs live under the log root, alongside every other stage's,
   rather than under the mosaic output directory with the products.

.. note::

   Because a worker stamps per task, an image processed twice -- a retried
   task, or a batch submitted twice -- normally leaves two log files, one per
   attempt. The exception is two attempts landing in the same UTC second,
   which resolve to one name and append into a single file.

What appears on the terminal
============================

By default the main log goes to both the terminal and a file, and image logs go
to a file only.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Logger
     - Terminal
     - File
   * - Main
     - yes
     - yes
   * - Image
     - no
     - yes

.. note::

   An interactive run therefore shows top-level progress rather than
   per-component detail. The detail is not lost -- it is in the per-image log
   file. Pass ``--log-image-to-console`` to see it on screen as well, or set
   ``logging.image_console: true`` to see it on every run.

   ``sd_offset`` summarizes each image's answer to the main log regardless, so
   the offset, status and confidence stay on the terminal without asking for
   the whole per-image narrative::

      N1234567890_1.IMG: status=success, offset (dv, du) = (1.500, -2.500) px,
      confidence 0.750 (medium)

   The sigmas, the confidence rank's inputs and the per-technique breakdown
   remain in that image's log.

Turning every sink off produces no output at all, rather than falling back to
the terminal.

Command-line options
====================

Every program you run yourself accepts the same options, so what you learn for
one works for the next. A program with no image log accepts only the
main-logger options, and rejects the image ones by name rather than accepting
and ignoring them. The ``_cloud_tasks`` drivers accept none of these and are
configured through the configuration file alone; see `Cloud tasks`_ below.

``--log-root PATH``
    Where this run's log files go.

``--log-level LEVEL``
    The default level for both loggers.

``--log-level MODULE=LEVEL``
    The level for one component. Repeatable, and combines with the bare form.

``--log-level-main LEVEL``, ``--log-level-image LEVEL``
    The level for one logger, taking precedence over a bare ``--log-level``.

``--log-main-to-console`` / ``--no-log-main-to-console``
    Whether the main log reaches the terminal. Default on; set
    ``logging.main_console`` to change the default for every run.

``--log-main-to-file`` / ``--no-log-main-to-file``
    Whether the main log is written to a file. Default on.

``--log-image-to-console`` / ``--no-log-image-to-console``
    Whether image logs reach the terminal. Default off; set
    ``logging.image_console`` to change the default for every run.

``--log-image-to-file`` / ``--no-log-image-to-file``
    Whether image logs are written to files. Default on.

Levels are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL`` and
``NONE``. Both sinks of a logger always share a level, so there is one level to
set per component rather than one per sink.

Worked examples
---------------

Quiet the run but keep one technique verbose -- the usual shape of
investigating one technique across many images:

.. code-block:: bash

   sd_offset coiss_saturn --volumes COISS_2001 \
       --log-level WARNING --log-level titan_haze=DEBUG

Follow one image's processing as it happens. The same detail is written to
that image's log file either way; this puts it on the terminal too, which is
what you want when working on a single image rather than reading the file
afterwards:

.. code-block:: bash

   sd_offset coiss N1234567890 --log-image-to-console --log-level DEBUG

Keep the terminal quiet while the files stay complete, for a long batch left
running:

.. code-block:: bash

   sd_offset coiss_saturn --no-log-main-to-console --log-level-image DEBUG

Silence one noisy component without quieting anything else:

.. code-block:: bash

   sd_offset coiss_saturn --log-level annotate=NONE

Configuring levels
==================

Every level is settable in the configuration, under the top-level ``logging``
section, as is whether each logger reaches the terminal -- ``main_console`` and
``image_console``.

There are no configuration keys for the file sinks or the log root. Whether a
log file is written is inseparable from where it goes, and that is chosen per
run with ``--log-root``, so ``--log-main-to-file`` and ``--log-image-to-file``
are command-line only. Writing ``main_file`` or ``image_file`` in the
configuration is an error naming the key, not a setting that quietly does
nothing.

.. code-block:: yaml

   logging:
     main: INFO            # the run's logger
     image: INFO           # image logs, and any component not named below
     main_console: true    # whether the run's log reaches the terminal
     image_console: false  # whether image logs do
     techniques:
       titan_haze: DEBUG   # one technique
       default: WARNING    # every other technique
     models:
       rings: WARNING      # one model family
     other:
       annotate: ERROR
     programs:
       sd_mosaic:          # applies to that program only
         main: WARNING

The most specific setting wins:

.. code-block:: text

   --log-level MODULE=LEVEL
     > a component named in the configuration
     > its category's "default"
     > --log-level-main / --log-level-image
     > --log-level LEVEL
     > logging.main / logging.image
     > INFO

Note that a component named in a configuration file outranks a bare
``--log-level``, which says nothing about that component. The shipped
configuration names one: ``other.annotate: ERROR``. So ``--log-level NONE``
does not produce silence -- annotation stays at ERROR, which keeps a file sink
open and writes a log per image. To get silence, either name it
(``--log-level NONE --log-level annotate=NONE``) or turn the sink off with
``--no-log-image-to-file``, which is what you probably wanted.

A ``programs`` block applies to that program alone and is merged key by key
with the settings above it, so a program can override one value while
inheriting the rest.

An unrecognized component name, program name or level is rejected when the
configuration loads, naming the offending key. A setting that does nothing is
worse than one that errors, because it looks like it worked.

Component names
---------------

A component is named by the technique or model it is, in snake_case.

**Techniques** -- ``body_blob``, ``body_disc_correlate``, ``body_limb``,
``body_terminator``, ``manual``, ``ring_annulus``, ``ring_edge``,
``star_field_from_catalog``, ``star_refine``, ``star_unique_match``,
``titan_haze``

**Models** -- ``body``, ``rings``, ``stars``, ``titan``. One name covers a
whole family: ``body`` governs every body model regardless of which body it
renders, and a simulated model is named with the model it stands in for.

**Everything else** -- ``annotate``, ``correlate``, ``ensemble``,
``image_derivatives``, ``obs``, ``orchestrator``, ``provenance``

Cloud tasks
===========

The ``_cloud_tasks`` drivers write **nothing** to the terminal. A worker's
console belongs to ``cloud_tasks``, which reports task progress there under its
own configuration, and interleaving per-image navigation detail with it would
make both harder to read.

The per-image logs are written exactly as an interactive run writes them, to
the same ``{log_root}/{backend}/`` tree and at the same levels, so an image's
log reads the same whichever driver produced it. There is no main log: with
many workers writing to one log root, a single shared main log is not something
they can all append to sensibly.

These drivers accept no logging command-line options, because every one of them
configures a logger they do not have or a terminal they must not write to. Set
their levels in the configuration instead. ``main_console`` and
``image_console`` have no effect on them either: a worker's terminal is not
theirs to write to however it is configured.

A cloud-task driver shares its interactive sibling's identity: ``sd_offset``
for ``sd_offset_cloud_tasks``, and so on. So a ``programs`` block covers both
forms of a program and cannot distinguish them -- ``logging.programs.sd_offset``
governs the interactive driver and the worker alike, and there is no
``logging.programs.sd_offset_cloud_tasks``. That is deliberate: an image's log
should read the same whichever driver produced it, which is the same reason the
two write into one tree.

Because a cloud task has no main log, an outcome that an interactive run would
report there is returned in the task result instead. A backplanes task reports
whether the image was processed or skipped, and a reprojection task returns how
many images it completed, skipped and failed.

``sd_results_index_cloud_tasks`` is the case where the task result is the whole
record: it has no per-image log either, because it reads documents rather than
images. Each task returns how many files it ingested, skipped and could not
read, and names every file it could not read. ``sd_results_index`` reads those
tallies back and writes them into the index, where they stay.
