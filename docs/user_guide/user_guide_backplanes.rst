====================
Backplane Generation
====================

Overview
========

Backplanes are per-pixel geometry products (longitude, latitude, incidence
angle, emission angle, phase angle, resolution, etc.) derived from a
navigated image. The system reads prior navigation metadata to apply the
image's recorded pointing, then computes body and ring backplanes, merges
them per-pixel by distance, and writes a multi-HDU FITS file along with a
JSON metadata file.

Which pointing a product is built on
------------------------------------

The driver prefers the exact recorded form over its approximation. When the
navigation record carries a corrected camera attitude
(``navigation_result.pointing.cmatrix``) that passes the reader's
consistency gates, the observation's frame is replaced with that attitude —
the same measurement as the pixel offset, expressed exactly, and what a
SPICE consumer of the corrected C-kernels sees for every image whose segment
was written. When there is no usable corrected attitude — a fitted-rotation
result (``no_cmatrix_rotation_fitted``), a record with no pointing block
(``no_pointing_block``), an unusable one (``malformed_pointing``), or a gate
refusal (``cmatrix_foreign_midtime``, ``cmatrix_baseline_mismatch``,
``cmatrix_unknown_host``, each
warned to the run log) — the recorded ``(dv, du)`` offset is applied via
:class:`oops.fov.OffsetFOV` instead; no product is ever built on a corrected
attitude that failed a gate. A kernel pool that already answers the
corrected attitude (corrected C-kernels furnished at load time) is left
alone, counted as ``pool_already_corrected``, since applying anything again
would double-correct. With no usable pointing of either kind the backplanes
are computed on uncorrected pointing, with a warning in the run log.

Each single-image result reports what happened: ``pointing_source`` is one
of ``'cmatrix'``, ``'pool'``, ``'offset'``, or ``'none'``, joined by
``pointing_reason`` when the source is degraded and by
``uncorrected_pointing: true`` when it is ``'none'``. A success-status
record with no ``offset`` key at all is a recorded no-answer like a null one,
counted under ``missing_offset_key``. For a result the kernel generator omitted
from the corrected kernels (a BOTSIM-yielding WAC, or any image with an
omission reason), the backplanes still carry that image's own recorded
measurement — the authoritative product for it — while a kernel consumer
sees the winning segment's attitude.

Key properties:

- The output FITS places ``BODY_ID_MAP`` as the first image HDU (after the
  primary HDU).
- Pixels a backplane did not measure carry the masked value, ``-999.0`` as
  shipped (``backplanes.masked_value``). It sits outside the range of every
  plane, so ``!= -999.0`` selects the measured pixels of any plane.
  ``BODY_ID_MAP`` is the exception: it holds ``0`` where no body claimed the
  pixel, ``0`` not being a NAIF ID.
- Backplanes that are entirely masked are omitted from the FITS file.
- The list of backplanes to generate is configured under ``backplanes`` in
  ``src/spindoctor/config_files/config_900_backplanes.yaml``.
- For simulated observations, synthetic backplanes are produced whose masks
  follow the simulated body shapes.

Backplane generation only writes the FITS file and the associated metadata
JSON. PDS4 labels for the backplane products are produced in a later step
by ``sd_create_bundle labels`` (see :doc:`user_guide_pds4_bundle`).

For the pipeline's internal architecture (per-source generation,
distance-aware merge, FITS writer details, "Adding a backplane" checklist),
see :doc:`/dev_guide/dev_guide_backplanes`.

Command-Line Interfaces
========================

Two drivers mirror the offset drivers:

- ``sd_backplanes`` (local/CLI)
- ``sd_backplanes_cloud_tasks`` (Cloud Tasks)

Common flags:

- ``--nav-results-root``: Root containing prior navigation results
  (``*_metadata.json``).
- ``--backplane-results-root``: Root directory for the backplane outputs.
- ``--results-index-db``: Connection URL of a results index built by
  ``sd_results_index``. With one, each image's navigation record is read as one
  database row instead of one file, which on a cloud results root replaces a
  round trip per image with a query. The index must already hold a completed
  ingest of the root named by ``--nav-results-root``, and the rows it holds are
  a snapshot of the tree as of that ingest. Omitting the option names no index,
  which is the default: the navigation results tree is read directly.
  ``--results-index-db none`` names no index either, which is how a machine that
  sets the option through configuration or through ``NAV_RESULTS_INDEX_DB``
  reads the files.
- Dataset selection flags are the same as for ``sd_offset`` (see
  :doc:`user_guide_navigation`).

An image the index has no row for is reported and skipped exactly as an image
with no metadata file is, and a named index that cannot be opened, or a
navigation results root it has not fully ingested, fails the run rather than
quietly reverting to reading files.

An image whose metadata document the ingest could not read is a third case, and
it fails that image rather than skipping it. Such a document is recorded as a
file the index holds no navigation record for, which is not the same fact as
"nothing navigated this image": read directly, the same document may well carry
a pointing and a status. The failure names the image, the index and the reason
the ingest recorded, so the remedy — fix the document and ingest that root
again, or run without ``--results-index-db`` — is visible from the run log. The
rest of the pass continues; only that image is lost.

A run that also names an error filter (``--has-offset-error``,
``--has-no-offset-error``, ``--has-offset-spice-error``,
``--has-offset-nonspice-error``) has already read each selected image's
navigation document, because that is how the filter decided what to select. The
record travels with the image, so each such image's document is read once for
the whole run. A second read would be a second download on a cloud results root,
not a second look at a file already local — the two readers hold caches of their
own. What the backplanes are built on is what the document said when the
selection was made, so a document rewritten or deleted while the run is going is
not noticed for an image already selected, and the "skipped due to missing
metadata" outcome is not reported for one: a record travels only with a document
that was read whole. With ``--results-index-db`` nothing travels with the image,
because the filter narrows on columns; each image's row is read as it is for any
other run.

Under ``sd_backplanes_cloud_tasks`` each of those outcomes is reported in the
task result rather than in a run log, because a cloud task has none: an
unusable index is ``unusable_results_index_db``, an image nothing navigated is a
skip named ``no_navigation_record``, and every other way one image can fail —
a document the ingest refused among them — is ``backplanes_failed``. All three
are returned rather than raised, so a queue configured to retry on an
exception does not retry a refusal that will refuse identically.

Examples
--------

Generate backplanes locally for a dataset:

.. code-block:: bash

    sd_backplanes coiss_saturn \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 --first-image-num 1454000000 --last-image-num 1454999999

To generate a cloud-tasks JSON file for all selected images without actually
generating any backplanes, use ``--output-cloud-tasks-file``:

.. code-block:: bash

    sd_backplanes coiss_saturn \
      --volumes COISS_2001 \
      --output-cloud-tasks-file backplanes_tasks.json

Cloud Tasks variant (file list comes from the queue):

.. code-block:: bash

    sd_backplanes_cloud_tasks \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes

Cloud-tasks JSON schema
^^^^^^^^^^^^^^^^^^^^^^^

The file produced by ``--output-cloud-tasks-file`` is a JSON array of task
objects. Each task is:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "dataset_name": "<dataset_name>",
            "files": [
                {
                    "image_file_url": "<path or URL to image file>",
                    "label_file_url": "<path or URL to label file>",
                    "results_path_stub": "<relative stub used to name outputs>",
                    "index_file_row": {"<column>": "<value>", "...": "..."}
                }
            ]
        }
    }

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.dataset_name``: one of the supported dataset names (same value as
  the positional argument to ``sd_backplanes``).
* ``data.files``: one or more file descriptors. Each descriptor has required
  fields ``image_file_url``, ``label_file_url``, ``results_path_stub``, and
  an optional ``index_file_row`` (metadata from the source index file, may
  be ``null``). The ``sd_backplanes_cloud_tasks`` worker accepts no other
  task-level parameters; all other settings come from its own
  ``--config-file``, ``--nav-results-root``, and ``--backplane-results-root``
  CLI flags, which apply to every task the worker handles.

Configuration
-------------

Backplanes are configured under ``backplanes`` in
``src/spindoctor/config_files/config_900_backplanes.yaml``:

- ``backplanes.bodies``: list of body backplane entries. Each entry has
  ``name`` (the FITS HDU name), ``method`` (the ``oops.Backplane`` method to
  call), and ``units``, all three required. ``units`` is written to the
  ``BUNIT`` FITS header and is also what decides whether the plane's
  statistics are converted to degrees, so it describes the array (see
  `Outputs`_ below).
- ``backplanes.rings``: list of ring backplane entries with the same
  structure. The special ``distance`` entry is used only for per-pixel
  merge ordering and is not written as an HDU.

Outputs
-------

For each processed image, ``sd_backplanes`` writes two files under
``--backplane-results-root``:

- ``<results_path_stub>_backplanes.fits`` containing:

  - A primary HDU.
  - ``BODY_ID_MAP`` (int32) as the first image HDU.
  - One ``ImageHDU`` per backplane array that measured at least one pixel,
    with ``BUNIT`` set when configured.

- ``<results_path_stub>_backplane_metadata.json`` containing per-body
  inventory information and per-backplane ``min``/``max`` statistics with the
  ``units`` they are in (consumed by ``sd_create_bundle`` when generating PDS4
  labels).

The arrays and the statistics use different angular units, on purpose. An
angular array is written in radians, which is what its ``BUNIT`` header says
and what software reading it wants: the unit the geometry is already in, with
no conversion step to get wrong. The statistics are written in degrees, because
they become the columns of a bundle's global index tables, which a person reads
to decide whether an image is worth opening — and a latitude range of -88 to 81
says something that -1.54 to 1.42 does not. Every statistic records the unit it
is in, so a bundle carrying ``rad`` on an array and ``deg`` on the table
summarising it is describing each file correctly rather than contradicting
itself. Non-angular planes are untouched: a ring radius stays in kilometres and
a radial resolution in kilometres per pixel.

What is compared is the unit's measure, the part before any solidus, and not
the whole string. A plane declared ``rad`` is summarized in ``deg`` and one
declared ``rad/pixel`` in ``deg/pixel``, while ``km`` and ``km/pixel`` are left
as they are.

Logs are written under the log root rather than beside these products: the
run's own log to ``{log_root}/sd_backplanes/main_{timestamp}.log`` and one per
image to ``{log_root}/backplanes/{results_path_stub}_{timestamp}.log``.
``sd_backplanes`` accepts the same logging options as every other pipeline
program; see :doc:`user_guide_logging`.

An image whose navigation did not succeed is skipped and gets no backplanes.
The run's log says which images those were, and reports the navigation status
that caused each skip.

One image that cannot be processed does not end the run. Backplane generation
is per-image work, so a failure is reported against that image, counted, and
the next image is attempted. The run's log carries the image, the message and
the traceback — an image can fail before it has a log of its own, and then the
run's log is the only account of it — and that image's own log carries the
traceback too whenever the failure got as far as opening one. The pass ends
with a summary line counting what became of every image::

   Backplane pass complete: 143 done, 4 skipped, 1 failed

Backplane Viewer GUI
====================

Use the interactive GUI to inspect backplane FITS alongside the science image.

Run
---

.. code-block:: bash

    sd_backplane_viewer coiss_saturn \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 \
      --first-image-num 1454000000 --last-image-num 1454000999

Features
--------

- Image stretch: Blackpoint, whitepoint, and gamma for the grayscale science image.
- Zoom and pan: Same behavior as the simulated body model UI.
- Summary overlay: If ``<results_path_stub>_summary.png`` exists under ``--nav-results-root``, it can be toggled on/off with an alpha control (no stretch or colormap).
- Backplane layers:

  - Lists all FITS image HDUs: ``BODY_ID_MAP`` (int32) plus each backplane (float32).
  - Each backplane can be toggled with a checkbox, assigned transparency 0-1, a colormap, and scaling mode (Absolute or Relative).
  - Relative mode computes min/max using only the pixels a plane measured, which are the finite ones that are not the masked value.
  - Absolute mode:

    - Longitudes: 0-360 deg; Latitudes: -90-90 deg.
    - Incidence/Emission/Phase: 0-180 deg.
    - Radius: 0 to observed max.
    - Resolution and others: observed min-max.

- Live readout: Shows the science image value at the cursor and, for each backplane row, the current value at the cursor (angles are converted from radians to degrees when applicable).

Notes
-----

- Units: FITS HDUs whose ``BUNIT`` is exactly ``rad`` are converted to degrees for display and absolute scaling; a plane whose unit qualifies that measure, such as ``rad/pixel``, is displayed in the unit its array carries. Heuristics are used for common angle names if units are missing. This is the viewer's own rule and is not the one the metadata document's statistics follow.
- Masking: Backplane visualizations treat a pixel as valid when it is finite and is not the masked value, which is the same rule for body and ring planes.
