=================
Cloud Task Worker
=================

Queue-driven processing is supported by ``sd_offset_cloud_tasks``. This variant reads tasks from a queue and processes each batch of files described by the task payload. It accepts the same environment options used to derive configuration and results roots and does not include dataset selection flags because the task provides the list of files. Invoke it with:

.. code-block:: bash

   sd_offset_cloud_tasks [--config-file PATH] [--nav-results-root PATH]

Cloud-tasks JSON schema
-----------------------

The file produced by ``--output-cloud-tasks-file`` is a JSON array of task
objects. Each task is:

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

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.dataset_name``: one of the supported dataset names.
* ``data.arguments``: an object with optional keys ``nav_models`` and
  ``nav_techniques`` (each a list of strings, or ``null``).
* ``data.files``: one or more file descriptors with required fields
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and
  optional ``index_file_row`` (metadata from the source index file, may be
  ``null``) and ``extra_params`` (arbitrary key/value dictionary forwarded
  to the task implementation; optional, may be ``null`` or omitted).

Whole-mission task files
-----------------------

A mission is more than one ``sd_offset`` invocation to enumerate: Cassini ISS holds far more images than one queue should carry, and Voyager ISS is run one planetary encounter at a time. ``cloud_support/scripts/`` holds a generator per instrument that makes the selections and writes the files a queue is loaded from -- Galileo SSI and New Horizons LORRI as a single file each, Voyager ISS as one file per encounter, and Cassini ISS as consecutive groups of whole volumes holding roughly fifty thousand images apiece. Each generator requires the holdings root the cloud workers will read, because the image and label URLs a task carries are absolute and are fixed when the task is written.

These scripts are part of the repository rather than of the installed package. ``cloud_support/README.md`` describes them together with the compute-instance startup script and the job configuration they are used with.

.. _selecting-models-and-techniques:

