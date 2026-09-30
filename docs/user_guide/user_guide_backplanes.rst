====================
Backplane Generation
====================

Overview
========

Backplanes are per-pixel geometry products derived from a navigated image:
longitude, latitude, incidence angle, emission angle, phase angle, resolution,
and more. ``sd_backplanes`` reads the navigation results for an image, points the
observation as those results say it should be pointed, computes the body and ring
backplanes, merges them per pixel by distance so that the nearer surface wins,
and writes a multi-extension FITS file together with a metadata document in JSON.
Each backplane value is the geometry at the center of that pixel. Positions are
written in pixel-corner coordinates, where a whole number falls on the boundary
between two pixels: the first pixel spans 0.0 to 1.0 and its center is at 0.5, so
the value in row 0, column 0 of a backplane is the geometry at position
``(0.5, 0.5)``. :ref:`coordinate-systems` in the developer guide has more.

Which pointing a product is built on
------------------------------------

Navigation measures where the camera was really pointed, and records that
measurement in each image's metadata document in two forms: a corrected camera
attitude, and a pixel offset. The corrected attitude states the measurement
exactly. The offset is its first-order approximation, and it is a correction
relative to the SPICE kernels that were furnished when the image was navigated:
the spacecraft and planetary ephemerides, the spacecraft clock and leapsecond
kernels, the frame and instrument kernels, and above all the original C kernels
that supply the camera's uncorrected attitude. Those kernels are listed by name
in the same metadata document, under
``navigation_result.provenance.spice_kernels``. Read against a different set of
kernels the offset means something different, so the corrected attitude is the
form preferred wherever it can be used.

The corrected attitude is used when the record survives three checks. The
recorded matrices must be proper rotations. The recorded exposure midtime must
match this observation's own midtime to within a microsecond, so that a record
belonging to a different image is refused rather than applied. And the
uncorrected attitude recorded at navigation time must agree with the attitude
the furnished kernels give now. When all three hold, the observation's camera
frame is replaced by the corrected attitude and the field of view is left
untouched. That is the same attitude a SPICE consumer of the corrected C kernels
sees for every image whose segment was written.

The third check is also how a run notices that the kernels furnished to it
already carry the correction. A furnished kernel is anonymous in the attitudes it
answers: it returns an attitude and says nothing about where that attitude came
from. A corrected kernel does record the original it corrects, in the comment
area described in :doc:`user_guide_ck_kernels`, but that is a property of the file
rather than of the answers SPICE gives, so nothing in the attitude itself
distinguishes the two. The attitude the furnished kernels give is therefore
compared against both of the attitudes the metadata document records. If it
matches the uncorrected one, the kernels are the ones navigation saw and the
correction is applied. If it matches the corrected one instead, the correction
is already in the kernels and nothing at all is applied, because applying either
form again would move the image by about twice the measured offset. If it
matches neither, then the kernels, the metadata document, or the camera frame
convention has changed since the image was navigated; the corrected attitude is
refused and the refusal is written to the run's log.

Where no corrected attitude can be used, the recorded pixel offset is applied to
the field of view instead. That is what happens for a navigation that fitted a
camera rotation rather than a shift, for a simulated image, which records no
corrected attitude, for a recorded pointing that cannot be read, for an image whose
instrument has no SPICE camera frame to check the record against, and for each
of the refusals above. Where neither form can be used -- there is no metadata
document, the metadata document is not valid JSON, the navigation did not
succeed, or the recorded offset is null or unusable -- the product is built on
the camera's uncorrected pointing, and the run's log says so.

The corrected C kernels deliberately leave out some navigated images: the
yielding camera of a simultaneous pair, and any image recorded with an omission
reason. For such an image the products built here still carry that image's own
recorded measurement, which is the better answer for that image, while a
consumer of the corrected kernels sees the attitude of the segment that won.

Each image's backplane metadata document reports which pointing its backplanes
were built on. ``pointing_source`` names which one:

``'cmatrix'``
    The corrected attitude the navigation measured was applied.

``'pool'``
    The furnished kernels already carried the correction, so nothing was applied.
    This is the case described above, where the recorded corrected attitude was
    found to be the one already in effect.

``'offset'``
    No corrected attitude could be used, so the recorded pixel offset was applied
    to the observation instead.

``'none'``
    Neither was usable, and the backplanes were built on the uncorrected pointing.

``pointing_reason`` names why whenever the source is a fallback, and
``uncorrected_pointing`` is ``true`` when the source is ``'none'``.

What the FITS file holds
------------------------

- ``BODY_ID_MAP`` is the first image extension, after the primary one. It gives
  the NAIF identifier of the body each pixel shows.
- A pixel a backplane did not measure carries the masked value, which is
  ``-999.0`` as shipped (``backplanes.masked_value``). It sits outside the range
  of every plane, so comparing against the configured value selects the measured
  pixels of any of them. Read the value out of the configuration of the run that
  wrote the file rather than assuming the default. ``BODY_ID_MAP`` is the
  exception: it holds ``0`` where no body claimed the pixel, ``0`` not being a
  NAIF identifier. (:doc:`user_guide_configuration` says where the defaults
  live.)
- A backplane that measured no pixel at all is left out of the file.
- Which backplanes are generated is configured, as are the units each is written
  in; see :doc:`user_guide_configuration`.
- For a simulated observation, the backplanes are synthetic, and their masks
  follow the simulated body shapes.

``sd_backplanes`` writes only the FITS file and its metadata document. The PDS4
labels for these products are produced later, by ``sd_create_bundle labels``;
see :doc:`user_guide_pds4_bundle`.

For how the backplanes are generated, merged, and written internally, see
:doc:`/dev_guide/dev_guide_backplanes`.

Running sd_backplanes
=====================

::

    sd_backplanes DATASET [options]

``DATASET`` names the dataset to draw images from, and the options that choose
which of its images to process are the same ones every pipeline program takes;
see :doc:`user_guide_image_selection`.

Options
-------

* ``--config-file PATH`` (repeatable): configuration file overriding the
  built-in settings. Without one, ``./nav_default_config.yaml`` is read if it
  exists. See :doc:`user_guide_configuration`.

* ``--nav-results-root DIR``: root of the navigation results written by
  ``sd_offset`` (see :doc:`user_guide_navigation_running`). Takes precedence
  over the ``NAV_RESULTS_ROOT`` environment variable and the configured value.

* ``--backplane-results-root DIR``: root directory the backplane products are
  written under. Takes precedence over the ``NAV_BACKPLANE_RESULTS_ROOT``
  environment variable and the configured value.

* ``--results-index-db URL``: connection URL of a results index built by
  ``sd_results_index`` (see :doc:`user_guide_results_index`): a ``sqlite:`` URL
  naming a local file, or a ``postgresql+psycopg:`` URL naming a server. Each
  image's navigation record is then read as one database row instead of one file,
  which on a cloud results root replaces a round trip per image with a query. The
  results index must already hold a completed ingest of the root named by
  ``--nav-results-root``, and its rows are a snapshot of that root as of the
  ingest. Omitting the option names no results index, which is the default, and
  the navigation results tree is read directly. ``--results-index-db none`` also
  names no results index, which is how a machine that sets the option through
  configuration or through the ``NAV_RESULTS_INDEX_DB`` environment variable
  reads the files instead.

* ``--output-cloud-tasks-file PATH``: write a cloud task queue file for the
  selected images and generate no backplanes. See
  `Running through cloud tasks`_.

* ``--dry-run``: report what the run would do and process no images. Off by
  default.

* ``--no-write-output-files``: do all the work and write none of the output
  files. Off by default.

* ``--profile``: collect a performance profile of the run. Off by default.

``sd_backplanes`` also takes the logging options every pipeline program takes,
which choose where the run's log and the per-image logs go and how much detail
each carries; see :doc:`user_guide_logging`.

When a results index is named
-----------------------------

An image that has no row in the results index is reported and skipped, exactly
as an image that has no metadata file is. A named results index that cannot be opened,
or one that has not fully ingested the navigation results root, fails a run that
generates backplanes rather than quietly reverting to reading files. ``--dry-run``
and ``--output-cloud-tasks-file`` read no navigation record, so neither one opens the
results index and neither fails for want of a usable one.

An image whose metadata document could not be read is a third case, and it fails
that image rather than skipping it. The results index records such a file as one
it holds no navigation record for, which is not the same fact as nothing having navigated
that image: read directly, the same metadata document may well carry a status and
a pointing. The failure names the image, the results index, and the reason the
ingest recorded, so the remedy is visible from the run's log: fix the metadata document
and ingest that root again, or run without ``--results-index-db``. The rest of
the pass continues, and only that image is lost.

Examples
--------

Generate backplanes for a range of Cassini images:

.. code-block:: bash

    sd_backplanes coiss_saturn \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 --first-image-num 1454000000 --last-image-num 1454999999

Outputs
=======

For each image processed, ``sd_backplanes`` writes two files under
``--backplane-results-root``:

- ``<results_path_stub>_backplanes.fits``, holding a primary extension,
  ``BODY_ID_MAP`` as the first image extension, and one image extension per
  backplane that measured at least one pixel, each carrying the unit it is in as
  its ``BUNIT`` header.
- ``<results_path_stub>_backplane_metadata.json``, holding the per-body
  inventory, the least and greatest value of each backplane with the unit those
  are in, and, for the rings, the ring target and the incidence angle of
  sunlight on the ring plane. ``sd_create_bundle`` reads this file when it
  generates the PDS4 labels.

The ``rings`` block of that metadata document looks like this:

.. code-block:: json

   {
     "rings": {
       "target": "SATURN_MAIN_RINGS",
       "incidence_angle": {
         "value": 82.57158, "min": 82.57085, "max": 82.57104, "mean": 82.57096, "units": "deg"
       },
       "backplanes": {"ring_radius": {"min": 74659.8, "max": 136779.0, "units": "km"}}
     }
   }

The rest of this section says what those entries mean.

Angular backplane arrays are in radians, as their ``BUNIT`` headers say. In the
metadata document an angular plane's minimum and maximum are in degrees, so
``rad`` becomes ``deg`` and ``rad/pixel`` becomes ``deg/pixel``, and each
statistic records its own unit.

Each minimum and maximum is taken over the pixels where the FITS plane has a
value. A body's are taken over the pixels ``BODY_ID_MAP`` gives to that body, so
a part of a body, or of the rings, that a nearer body covers does not count.

The ring longitude's statistic, and each body's longitude statistic, also record
``wrapped_min`` and ``wrapped_max``, in degrees: the arc of longitude the image's
ring pixels, or that body's pixels, cover, from where the arc starts to where it
ends. Where the arc crosses zero, ``wrapped_min`` is greater than
``wrapped_max``. An arc covering the whole circle, as a body's does when one of
its poles is in view, records 0 and 360. The ring longitude's arc is recorded
when the ring longitudinal resolution backplane has a value.

The metadata document's ``rings`` block names the ring target the ring
backplanes were computed for, as ``target``, and records ``incidence_angle``:
the angle between the direction sunlight arrives from and the normal to the ring
plane on its sunlit side, from 0 to 90 degrees, with its unit. Sunlight falls on
the ring plane at one angle over the whole image, so no backplane holds it; it is
taken once, at the center of the ring system, for the light that reached the
camera at the observation's midtime. Both are recorded for every image that has a
closest planet, whether or not any of its pixels is on the rings.

Where the image's ring backplanes have values, ``incidence_angle`` also records
the least, the greatest, and the mean angle over those pixels, as ``min``,
``max``, and ``mean``. These four numbers answer two different questions.
``value`` is the angle for the ring system, and it is there for every image that
has a closest planet, including one that shows no ring pixels at all. The other
three describe only the ring pixels this product holds. They are computed at each
pixel rather than once at the ring center, so they vary a little across an image,
and they tell you the range of illumination the ring pixels in this particular
product were under.

Each ring backplane's own least and greatest value are under ``backplanes``.

Logs are written under the log root rather than beside these products: the run's
own log to ``{log_root}/sd_backplanes/main_{timestamp}.log``, and one log per
image to ``{log_root}/backplanes/{results_path_stub}_{timestamp}.log``.

An image whose navigation did not succeed is skipped and gets no backplanes. The
run's log says which images those were, and reports the navigation status that
caused each skip.

An image that cannot be processed does not end the run. Backplane generation is
per-image work, so a failure is reported against that image, counted, and the
next image is attempted. The run's log carries the image, the message, and the
traceback, which matters because an image can fail before it has a log of its
own; that image's own log carries the traceback too whenever the failure got as
far as opening one. The pass ends with a summary line counting what became of
every image::

   Backplane pass complete: 143 done, 4 skipped, 1 failed

Backplane Viewer
================

``sd_backplane_viewer`` opens an interactive window showing an image's
backplanes on top of the science image itself:

.. code-block:: bash

    sd_backplane_viewer coiss_saturn \
      --nav-results-root /data/nav/results \
      --backplane-results-root /data/nav/backplanes \
      --volumes COISS_2001 \
      --first-image-num 1454000000 --last-image-num 1454000999

It takes ``--config-file``, ``--nav-results-root``, and
``--backplane-results-root``, which mean what they mean for ``sd_backplanes``,
plus the usual image selection options (see
:doc:`user_guide_image_selection`).

The image selection options can match any number of images, and the viewer shows
one at a time. It opens on the first image of the selection, and **Prev Image**
and **Next Image** step through the rest, reloading the science image and the
backplanes for each. So a selection of a thousand images is a thousand images to
page through rather than a thousand windows, and narrowing the selection is how
you get to the image you want without stepping.

Features
--------

- Image stretch: black point, white point, and gamma for the grayscale science
  image.
- Zoom and pan, with the same controls as the simulated body model window.
- Summary overlay: where ``<results_path_stub>_summary.png`` exists under
  ``--nav-results-root``, it can be turned on and off and faded with an alpha
  control. It carries no stretch or colormap of its own.
- Backplane layers:

  - Every image extension of the FITS file is listed: ``BODY_ID_MAP`` and each
    backplane.
  - Each layer can be turned on and off, given a transparency from 0 to 1, given
    a colormap, and scaled either absolutely or relatively.
  - Relative scaling takes the minimum and maximum over only the pixels that
    plane measured, which are the finite ones that are not the masked value.
  - Absolute scaling uses fixed ranges: 0 to 360 degrees for longitudes, -90 to
    90 degrees for latitudes, 0 to 180 degrees for incidence, emission, and
    phase, 0 to the observed maximum for radius, and the observed minimum to
    maximum for resolution and everything else.

- Live readout: the cursor's ``(v, u)`` position to two decimals in pixel-corner
  coordinates (see :ref:`coordinate-systems`), the science image value at the
  pixel holding that position, the object ``BODY_ID_MAP`` names there, and the
  value of each backplane at the cursor.

Units and masking
-----------------

A backplane whose ``BUNIT`` is ``rad``, or whose name contains ``longitude``,
``latitude``, ``incidence``, ``emission``, or ``phase``, is shown in degrees.
Every other backplane is shown in the unit it is stored in, so
``ring_longitudinal_resolution``, stored in ``rad/pixel``, is shown in radians
per pixel.

A pixel counts as valid where it is finite and is not the masked value, and that
is the same rule for body and ring planes alike.

Running through cloud tasks
===========================

The backplane worker is ``sd_backplanes_cloud_tasks``, and
:doc:`/user_guide/user_guide_cloud_tasks` describes what cloud tasks is, how a
task file is loaded into a queue, and who runs the workers.

``--output-cloud-tasks-file`` writes such a task file for the images a run would
have processed, and generates no backplanes:

.. code-block:: bash

    sd_backplanes coiss_saturn \
      --volumes COISS_2001 \
      --output-cloud-tasks-file backplanes_tasks.json

The worker takes only the settings that describe its own environment, and applies
them to every task it handles: ``--config-file``, ``--nav-results-root``,
``--backplane-results-root``, and ``--results-index-db``.

A worker has no run log, and each outcome a local run would have logged comes back
in the task result instead. A results index that cannot be used is
``unusable_results_index_db``. An image nothing navigated is a skip named
``no_navigation_record``. Every other way an image can fail, including a metadata
document that could not be read, is ``backplanes_failed``. All three are
returned rather than raised, so a queue set to retry on an exception does not
retry a refusal that will refuse identically.

The task format
---------------

The file ``--output-cloud-tasks-file`` writes is a JSON array of task objects. A
backplane task looks like this:

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

* ``task_id`` identifies the task, and is built from the dataset name, the first
  image's label filename, and the position of the task in the enumeration.
* ``data.dataset_name`` is the dataset the images come from, the same value the
  local command takes as its positional argument.
* ``data.files`` holds one or more file descriptions. Each requires
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and may
  carry an ``index_file_row``, which may be ``null``. The worker accepts no
  other per-task settings.
