=================
Cloud Task Worker
=================

.. note::

   You do not run the program described in this chapter. It is a worker, started for you
   on a cloud compute instance by the cloud task system. The chapter is here so that you
   can recognize the program in a log, and understand where a navigation result came from
   when it was produced in the cloud rather than on your own machine.

What cloud tasks is
===================

Cloud tasks is a work-queue package supplied by the Ring-Moon Systems Node. It is a
separate product from SpinDoctor, distributed as ``rms-cloud-tasks``, and its own
documentation is at https://github.com/SETI/rms-cloud-tasks. It manages a queue of work
items, starts compute instances, hands each instance items from the queue, retries what
fails, and records what each item returned. This chapter does not re-document any of that;
read the cloud tasks documentation for how a queue is created, loaded, and run.

SpinDoctor's part is small. For each pipeline stage that can be run in bulk there is a
worker program whose name ends in ``_cloud_tasks``. A worker asks the queue for a task,
processes the images the task names, returns what happened, and asks for the next one.
``sd_offset_cloud_tasks`` is the navigation worker: it does exactly what ``sd_offset``
(:doc:`/user_guide/user_guide_navigation_running`) does to an image, and writes the same
metadata document and preview under the same navigation results root.

Because the queue supplies the list of images, the worker has no image-selection options
at all. It accepts only what it needs to find its configuration and its output root, plus
whatever options the cloud tasks package itself defines:

.. code-block:: bash

   sd_offset_cloud_tasks [--config-file PATH] [--nav-results-root PATH]

A worker writes no run log and no output to the terminal, because the terminal on a
compute instance belongs to the cloud task system. It does write the ordinary per-image
log for every image it navigates. See :doc:`/user_guide/user_guide_logging`.

The task file
=============

A queue is loaded from a JSON file that lists the work items. You produce that file with
**sd_offset**, using its ``--output-cloud-tasks-file PATH`` option: this is an option of
``sd_offset``, not of the worker. Given that option, ``sd_offset`` enumerates the images
your selection names, writes one task per batch to the named file, and does nothing else
-- no image is navigated. The file is then loaded into a queue with the cloud tasks
package's own tooling.

.. code-block:: bash

   sd_offset coiss --volumes COISS_2xxx/COISS_2116 \
       --output-cloud-tasks-file coiss_2116_tasks.json

The models and techniques you select on that command line are written into the file and
are what the workers will use, so choose them when you generate the tasks rather than
later.

Splitting a large selection
---------------------------

A single run can process any number of images; nothing in SpinDoctor limits how many
images one task file or one queue may hold. Splitting a mission into several task files is
therefore a convenience, and the convenience is this: each part can be run to completion
and its results assessed before the next part starts, so a systematic problem is caught
after one part rather than after the whole mission.

We follow that convention for the larger holdings. Voyager ISS is generated one planetary
encounter at a time, and Cassini ISS in consecutive groups of whole volumes. Galileo SSI
and New Horizons LORRI are each generated as a single file. The generator scripts that
make these files live in ``cloud_support/`` in the SpinDoctor repository rather than in
the installed package, and each one runs ``sd_offset --output-cloud-tasks-file`` and then
divides what it wrote. ``cloud_support/README.md`` describes them, together with the
compute-instance startup script and the job configuration they go with. Each generator
must be told the holdings root the workers will read, because the image and label URLs a
task carries are absolute and are fixed at the moment the task is written.

Task file structure
===================

The task file's structure is documented here for information only. You never need to write
or edit one: ``sd_offset`` generates it, and the worker reads it. Nothing in this section
is a thing to do.

The file is a JSON array of task objects. Each task looks like this:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "dataset_name": "<dataset_name>",
            "arguments": {
                "nav_models": ["body:*", "rings", "stars"],
                "nav_techniques": ["*"]
            },
            "files": [
                {
                    "image_file_url": "<path or URL to image file>",
                    "label_file_url": "<path or URL to label file>",
                    "results_path_stub": "<relative stub used to name outputs>",
                    "index_file_row": {"<column>": "<value>", "...": "..."},
                    "extra_params": {"<key>": "<value>"}
                }
            ]
        }
    }

The fields are:

* ``task_id``: a string that identifies the task uniquely, built from the dataset name,
  the first image's label filename, and the position of the batch in the enumeration.
* ``data.dataset_name``: the dataset the images come from, spelled as it is on an
  ``sd_offset`` command line.
* ``data.arguments``: an object with the optional keys ``nav_models`` and
  ``nav_techniques``, each a list of selection patterns or ``null`` for "use the default
  selection". These carry the ``--nav-models`` and ``--nav-techniques`` choices made when
  the file was generated.
* ``data.files``: the images the task covers. Each entry requires ``image_file_url``,
  ``label_file_url``, and ``results_path_stub``, and may carry ``index_file_row``, the row
  the PDS3 index table held for that image, and ``extra_params``, further key and value
  pairs passed through to the stage.
