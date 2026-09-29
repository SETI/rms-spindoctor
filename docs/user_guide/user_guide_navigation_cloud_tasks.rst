======================
Navigation Cloud Tasks
======================

``sd_offset_cloud_tasks`` is the navigation worker: the cloud task system runs it on a
cloud compute instance, and it does to each image exactly what ``sd_offset``
(:doc:`/user_guide/user_guide_navigation_running`) does, writing the same
metadata document and the same preview PNG under the same navigation results root. See
:doc:`/user_guide/user_guide_cloud_tasks` for what cloud tasks is and how a queue is
loaded and run.

Running the worker
==================

Because the queue supplies the images, the worker has no image-selection options. It
accepts only what it needs to find its configuration and its output root, plus whatever
options the cloud tasks package itself defines:

.. code-block:: bash

   sd_offset_cloud_tasks [--config-file PATH] [--nav-results-root PATH]

The worker writes the ordinary per-image log for every image it navigates, into the same
tree an interactive run writes it to, and writes no main log. See
:doc:`/user_guide/user_guide_logging`.

Writing the task file
=====================

``--output-cloud-tasks-file PATH`` is an option of **sd_offset**, not of the worker.
Given it, ``sd_offset`` enumerates the images your selection names, writes one task per
image to the named file, and navigates nothing:

.. code-block:: bash

   sd_offset coiss --volumes COISS_2xxx/COISS_2116 \
       --output-cloud-tasks-file coiss_2116_tasks.json

The models and techniques you select on that command line are written into the file and
are what the workers will use, so choose them when you generate the tasks rather than
later.

The task format
===============

The file is a JSON array of task objects. A navigation task looks like this:

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
  the first image's label filename, and the position of the task in the enumeration.
* ``data.dataset_name``: the dataset the images come from, spelled as it is on an
  ``sd_offset`` command line. A task naming a dataset the installed package does not know
  fails with ``unknown_dataset``.
* ``data.arguments``: an object with the optional keys ``nav_models`` and
  ``nav_techniques``, each a list of selection patterns or ``null`` for "use the default
  selection". These carry the ``--nav-models`` and ``--nav-techniques`` choices made when
  the file was written; see :doc:`/user_guide/user_guide_navigation_models`.
* ``data.files``: a list of the images the task navigates. Each entry requires
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and may carry
  ``index_file_row``, the row the PDS3 index table held for that image, and
  ``extra_params``, further key and value pairs passed through to the stage.
  ``sd_offset`` writes one entry per task.
