===========
Cloud Tasks
===========

Most SpinDoctor stages can be run over a whole mission at once by spreading the images
across cloud compute instances. This chapter describes the arrangement that makes that
possible. Each stage's own chapter describes the tasks that stage's worker reads.

What cloud tasks is
===================

Cloud tasks is a work-queue package supplied by the Ring-Moon Systems Node. It is a
separate product from SpinDoctor, distributed as ``rms-cloud-tasks``, and its own
documentation is at https://rms-cloud-tasks.readthedocs.io. It manages a queue of work
items, starts compute instances, hands each instance items from the queue, and records
what each item returned. Whether a failed item is handed out again depends on what the
worker did with the failure and on how the queue was run. Read the cloud tasks
documentation for how a queue is created, loaded, and run.

A queue holds one task per unit of work. A worker running on a cloud compute instance
takes a task off the queue, does the work the task names, reports what happened, and asks
for the next one. **A user never runs a worker directly**: the cloud task system starts it
on the compute instances you have asked for. What you do is generate the task file with
one of the programs you run yourself, load it into a queue, and read the results the
workers leave behind.

The workers
===========

SpinDoctor's part is small. For each stage that can be run in bulk there is a worker
program whose name ends in ``_cloud_tasks``. A worker does the same work the program you
run yourself does, and writes its results to the same place.

No SpinDoctor worker ever asks for a task to be handed out again. A failure the worker
can describe comes back as a task result whose ``status`` is ``error``, with a
``status_error`` value naming the failure, and the queue does not retry it. A failure
that stops the worker outright reaches the queue as an exception instead, and the queue
retries that only when it was run with retry-on-exception turned on. Each stage's chapter
says what its worker reports.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Worker
     - What it does, and where its tasks are described
   * - ``sd_offset_cloud_tasks``
     - Navigates images.
       :doc:`/user_guide/user_guide_navigation_cloud_tasks`
   * - ``sd_backplanes_cloud_tasks``
     - Generates backplanes.
       :doc:`/user_guide/user_guide_backplanes`
   * - ``sd_mosaic_cloud_tasks``
     - Reprojects images for a ring or body mosaic.
       :doc:`/user_guide/user_guide_reprojection`
   * - ``sd_create_bundle_cloud_tasks``
     - Runs the PDS4 labels pass.
       :doc:`/user_guide/user_guide_pds4_bundle`
   * - ``sd_results_index_cloud_tasks``
     - Ingests one share of a results root into the results index.
       :doc:`/user_guide/user_guide_results_index`

A worker takes no image-selection options, because the queue is what tells it which images
to process. It accepts only the options that describe its own environment -- where its
configuration is, and which roots it reads and writes -- plus whatever options the cloud
tasks package itself defines. Each stage's chapter lists the ones its worker takes.

How a task file comes to exist
==============================

A queue is loaded from a JSON file that describes the work items. Four programs write one:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Program
     - Option
   * - ``sd_offset``
     - ``--output-cloud-tasks-file PATH``
   * - ``sd_backplanes``
     - ``--output-cloud-tasks-file PATH``
   * - ``sd_mosaic``
     - ``--output-cloud-tasks-file PATH``
   * - ``sd_results_index divide``
     - ``--tasks-file PATH``

The option belongs to the program you run yourself, never to the worker. Given it, the
program enumerates the work its command line selects, writes the task file, and does
nothing else: no image is navigated, no backplane is generated, and no mosaic is
reprojected.

A task carries the work, not the worker's surroundings. An image-processing task names
the images to process, the dataset they come from, and the processing choices the
command line made: for ``sd_offset`` the models and techniques to run, and for
``sd_mosaic`` the whole mosaic configuration, including where the mosaic is written
and the prefix its filenames take. An ingest task from ``sd_results_index divide``
names the navigation results root it covers and the share of metadata documents in it.

The rest is the worker's own. The configuration file, the navigation results root a
processing worker reads, the backplane and bundle results roots it writes, the
results-index connection URL, and any credentials are given to the worker on its own
command line, and no task file carries them. They are chosen when the queue is run
rather than when the task file is written, and a worker started without one it needs
cannot get it from the task. Each stage's chapter lists the options its worker takes.

Where a task names image files, it names them by absolute URL, and that URL is fixed when
the task is written. Such a task file has to be generated against the holdings root the
workers will read.

What every task file has in common
==================================

A task file is a JSON array of task objects. Each object has two members:

* ``task_id``, a string that identifies the task uniquely within the file.
* ``data``, an object holding everything the worker needs in order to do that task.

What is inside ``data`` differs from one program to the next, and the differences matter:
a reprojection task carries a whole mosaic configuration, and a results index task carries
no image files at all. Each stage's chapter gives its own layout in full.

A task file is generated by one program and read by another, so there is no need to
write or edit one by hand. The layouts are documented so that you can read a file you
have generated: to confirm which images a queue covers, or to check which settings the
workers were given.

Splitting a large selection
===========================

Nothing limits how many images one run, one task file, or one queue may hold. Splitting a
large selection into several task files is a convenience, and the convenience is this:
each part can be run to completion and its results assessed before the next part starts,
so a systematic problem is caught after one part rather than after the whole mission.

We follow that convention for the larger holdings. Voyager ISS is navigated one planetary
encounter at a time, and Cassini ISS in consecutive groups of whole volumes. Galileo SSI
and New Horizons LORRI are each navigated from a single task file. The generator scripts
that make these files are not part of the installed package. They live in
``cloud_support/`` in the source repository at https://github.com/SETI/rms-spindoctor, and
each one runs ``sd_offset --output-cloud-tasks-file`` and then divides what it wrote.
``cloud_support/README.md`` there describes them, together with the compute-instance
startup script and the job configuration they go with.

What a worker writes to its logs
================================

A worker is meant to write nothing to the terminal, because the terminal on a compute
instance belongs to the cloud task system. ``sd_create_bundle_cloud_tasks`` is the
exception described below. Every worker that processes images, apart from that one, writes
the ordinary per-image log for every image it handles, into the same tree an interactive
run writes it to. None of them writes a main log. An outcome an interactive run would
have reported in its main log comes back in the task result instead. See
:doc:`/user_guide/user_guide_logging`.

Two workers write no log file at all. ``sd_results_index_cloud_tasks`` writes none
deliberately: it reads metadata documents rather than images, and what its task did is the
value the task returns. ``sd_create_bundle_cloud_tasks`` writes none because it configures
no logging at all. Its records are not discarded: with no output destination set they are
rerouted to the main logger and reach the worker's terminal. Read a bundle worker's
outcome from its task result and from the bundle it produced.

.. Removing the bundle worker altogether is tracked as issue #424.
