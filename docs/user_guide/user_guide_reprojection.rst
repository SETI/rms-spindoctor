===========================
Reprojection and Mosaicing
===========================

Reprojection takes an image of a body or of a ring system and resamples it onto
a map grid: latitude and longitude for a body, radius and longitude for a ring.
Mosaicing accumulates many reprojected images onto one such grid, keeping the
best pixel available at each cell, so that a whole encounter's worth of frames
becomes a single map. These are tools rather than a pipeline phase. Nothing
downstream requires them, and you reach for them when you want a map instead of
a frame.

Unlike the other chapters of this guide, this one documents something you can
use two ways. There are command-line programs, ``sd_mosaic_rings`` and
``sd_mosaic_body`` to build reprojections and mosaics and
``sd_mosaic_display_rings`` and ``sd_mosaic_display_body`` to look at them; and
there is an importable Python package, ``spindoctor.reproj``, which you can call
from your own code when you want control the programs do not offer. This chapter
covers the programs, which is what most work needs.
:doc:`user_guide_reprojection_api` covers the package. The programs work
entirely through the package, so nothing is available in one and missing from
the other.

Building reprojections and mosaics
==================================

``sd_mosaic_rings`` and ``sd_mosaic_body`` reproject every image of a selected
dataset and combine the results into one mosaic::

    sd_mosaic_rings DATASET [options]
    sd_mosaic_body  DATASET [options]

The same two are also reachable through one command that takes the kind of mosaic
as its first argument, which is the form the cloud task worker uses::

    sd_mosaic rings DATASET [options]
    sd_mosaic body  DATASET [options]

``DATASET`` names the dataset to draw images from, and the options that choose
which of its images to process are the same ones every pipeline program takes;
see :doc:`user_guide_image_selection`. The two commands are two faces of one
program, so everything below is common to them except where a section says
otherwise.

A run has two passes:

1. **The reprojection pass** reprojects each selected image on its own and
   writes one file per image. An image whose reprojection file already exists is
   left alone unless ``--overwrite`` is given, so an interrupted run can be
   resumed by running it again.
2. **The mosaic pass** walks the same list of images, reads each reprojection
   file that exists, accumulates them all into one grid, and writes the mosaic.

Either pass can be skipped, with ``--skip-reproject`` or ``--skip-mosaic``. The
common case for ``--skip-reproject`` is assembling a mosaic from reprojection
files an earlier run already wrote.

A ring mosaic over absolute ring radii::

    sd_mosaic_rings coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --planet SATURN \
        --radius-inner 70000 \
        --radius-outer 140000 \
        --output-dir /data/mosaics \
        --prefix saturn_main_rings_2004

An F ring mosaic, over radii measured from the F ring core's own orbit::

    sd_mosaic_rings coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --planet SATURN \
        --orbit-model f_ring_core_albers_2007 \
        --radius-inner-offset -1000 \
        --radius-outer-offset 1000 \
        --output-dir /data/mosaics \
        --prefix fring_2004

A body mosaic::

    sd_mosaic_body coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --body-name MIMAS \
        --output-dir /data/mosaics \
        --prefix mimas_2004

Options both commands accept
----------------------------

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--config-file PATH``
     - *(none)*
     - Configuration file overriding the built-in settings; may be given more
       than once. See :doc:`user_guide_configuration`.
   * - ``--nav-results-root DIR``
     - *(from configuration)*
     - Root of the navigation results written by ``sd_offset`` (see
       :doc:`user_guide_navigation_running`). Each image is reprojected on its
       navigated pointing. A root is required: give it here, or as
       ``environment.nav_results_root`` in a configuration file, or as the
       ``NAV_RESULTS_ROOT`` environment variable. A run that names none in any of
       those places stops.
   * - ``--results-index-db URL``
     - *(none)*
     - Connection URL of a results index built by ``sd_results_index`` (see
       :doc:`user_guide_results_index`): a
       ``sqlite:`` URL naming a local file, or a ``postgresql+psycopg:`` URL
       naming a server. Each image's navigation record is then read as one
       database row instead of one file, which on a cloud results root replaces
       a round trip per image with a query. The index must already hold a
       completed ingest of the root named by ``--nav-results-root``, and its
       rows are a snapshot of that root as of the ingest. Omitting the option
       names no index, and the navigation results are read as files.
       ``--results-index-db none`` also names no index, which is how a machine
       that sets the option through configuration or through the
       ``NAV_RESULTS_INDEX_DB`` environment variable reads the files instead.
   * - ``--output-dir DIR``
     - *(required)*
     - Directory the reprojection files and the mosaic are written to.
   * - ``--prefix STR``
     - *(empty)*
     - Prefix for every output filename.
   * - ``--format {fits,npz}``
     - ``fits``
     - Format of the output files. ``npz`` writes compressed NumPy archives.
   * - ``--overwrite``
     - off
     - Recompute and overwrite reprojection files that already exist.
   * - ``--skip-reproject``
     - off
     - Skip the reprojection pass and go straight to the mosaic pass, using the
       reprojection files already in the output directory.
   * - ``--skip-mosaic``
     - off
     - Skip the mosaic pass, producing only the per-image reprojection files.
   * - ``--dry-run``
     - off
     - Report what the run would do and write nothing.
   * - ``--no-write-output-files``
     - off
     - Do all the work and write none of the output files.
   * - ``--image-name LABEL``
     - *(each image's own file stem)*
     - Label recorded in every reprojection file and in the mosaic's list of
       contributing images.
   * - ``--output-cloud-tasks-file PATH``
     - *(none)*
     - Write a cloud task queue file for the selected images and do no
       reprojection or mosaicing. See `Running through cloud tasks`_.
   * - ``--profile``
     - off
     - Collect a performance profile of the run.

Both commands also take the logging options every pipeline program takes, which
choose where the run's log and the per-image logs go and how much detail each
carries; see :doc:`user_guide_logging`.

Ring options
------------

These are accepted by ``sd_mosaic_rings``.

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--planet NAME``
     - *(required)*
     - Planet whose rings are reprojected, for example ``SATURN``.
       Case-insensitive.
   * - ``--radius-inner KM``
     - *(see below)*
     - Inner radius of the mosaic, as an absolute ring radius in kilometers.
       Required when no orbit model is named, and not allowed when one is.
   * - ``--radius-outer KM``
     - *(see below)*
     - Outer radius of the mosaic, as an absolute ring radius in kilometers.
       Required when no orbit model is named, and not allowed when one is.
   * - ``--radius-inner-offset KM``
     - *(see below)*
     - Inner bound of the mosaic as a signed offset in kilometers from the orbit
       model's radius, usually negative. Required when an orbit model is named,
       and not allowed when none is.
   * - ``--radius-outer-offset KM``
     - *(see below)*
     - Outer bound of the mosaic as a signed offset in kilometers from the orbit
       model's radius, usually positive. Required when an orbit model is named,
       and not allowed when none is.
   * - ``--longitude-resolution DEG``
     - ``0.02``
     - Width of one mosaic column, in degrees.
   * - ``--radius-resolution KM``
     - ``5.0``
     - Height of one mosaic row, in kilometers.
   * - ``--orbit-model NAME``
     - ``none``
     - ``none``, ``f_ring_core_albers_2007``, or ``bring_outer_edge``. See
       `Longitude and radius conventions`_.
   * - ``--merge-strategy NAME``
     - ``most_coverage_then_resolution``
     - How overlapping images are resolved, either
       ``most_coverage_then_resolution`` or ``best_resolution``.
       ``most_coverage_then_resolution`` fills empty longitude columns first and
       replaces a column that already holds data only where the new data has
       better mean radial resolution. ``best_resolution`` only ever replaces a
       column on better mean radial resolution.
   * - ``--margin N``
     - ``3``
     - Pixels along each edge of the image to leave out.
   * - ``--zoom N`` or ``--zoom R,L``
     - ``1``
     - Sub-samples taken per output cell and averaged into it. The image itself
       is not interpolated: each sub-sample reads the one pixel containing it.
       Raising this smooths the mosaic and fills cells a single sample would
       miss, and costs run time as the product of the radial and longitudinal
       factors, which for a single ``N`` is its square. ``R,L`` sets the radial
       and longitudinal factors separately.
   * - ``--no-omit-shadow``
     - off *(shadowed pixels are masked)*
     - Keep the ring pixels that lie inside the planet's shadow.
   * - ``--longitude-range START END``
     - *(the full circle)*
     - Reproject only this range of longitude, in degrees.
   * - ``--radius-range INNER OUTER``
     - *(the mosaic's own bounds)*
     - Reproject only this range of radius, in kilometers, under the same
       convention as the mosaic bounds: absolute radii with no orbit model,
       signed offsets with one.
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy data type the brightness array is stored in.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy data type the geometry arrays are stored in.
   * - ``--photometric-model NAME``
     - ``none``
     - Photometric correction applied while reprojecting: ``none``,
       ``lambert``, ``lommel-seeliger``, or ``minnaert``.

.. _orbit-model-longitude:

Longitude and radius conventions
--------------------------------

``--orbit-model`` chooses between two coordinate conventions, and it changes
what the stored longitudes mean and what the radius options measure.

With ``--orbit-model none``, the default, the longitudes stored in the
reprojection files and in the mosaic are inertial J2000 ring longitudes,
measured eastward from the ascending node of the ring plane on the J2000
reference plane. Radii are absolute kilometers, and the mosaic's bounds come
from ``--radius-inner`` and ``--radius-outer``. Corotating longitude and the
radial offset from an orbit are not defined, and the display program marks those
fields unavailable.

With ``--orbit-model f_ring_core_albers_2007`` or ``--orbit-model
bring_outer_edge``, each inertial longitude is converted into the corotating
frame of that model before it is binned. Mosaic column *i* then holds corotating
longitude ``i`` times the longitude resolution, and the column index no longer
has a fixed relationship to J2000 north; the inertial longitude can be recovered
from the corotating longitude given the orbit model and the observation time
stored for that column. Radii become signed offsets in kilometers from the
orbital radius at each longitude and time, and the mosaic's bounds come from
``--radius-inner-offset`` and ``--radius-outer-offset``. For an eccentric orbit
the orbital radius varies between ``a (1 - e)`` and ``a (1 + e)``, so measuring
from the orbit makes an eccentric ring come out as a straight line in the
reprojection rather than a sine wave.

Every reprojection accumulated into one mosaic must agree on the orbit model and
on the photometric model, because radii and longitudes mean different things
under different orbit models. A mismatch stops the mosaic pass.

``f_ring_core_albers_2007`` uses the Albers et al. 2012 Table 3 Fit #2 elements
for the F ring core. The ``2007`` in the name is the epoch its corotating frame
is anchored at, 2007-01-01T00:00:00Z.

Body options
------------

These are accepted by ``sd_mosaic_body``.

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--body-name NAME``
     - *(required)*
     - Body to reproject, for example ``MIMAS``. Case-insensitive.
   * - ``--lat-resolution DEG``
     - ``0.1``
     - Height of one mosaic row, in degrees.
   * - ``--lon-resolution DEG``
     - ``0.1``
     - Width of one mosaic column, in degrees.
   * - ``--lat-range MIN MAX``
     - *(the full range)*
     - Latitude extent of the mosaic, in degrees.
   * - ``--lon-range MIN MAX``
     - *(the full range)*
     - Longitude extent of the mosaic, in degrees.
   * - ``--max-incidence DEG``
     - *(no limit)*
     - Discard a pixel where sunlight strikes the surface at more than this
       angle from the vertical.
   * - ``--max-emission DEG``
     - *(no limit)*
     - Discard a pixel viewed at more than this angle from the vertical.
   * - ``--max-resolution KM``
     - *(no limit)*
     - Discard a pixel coarser than this many kilometers per pixel.
   * - ``--edge-margin N``
     - ``3``
     - Pixels along each edge of the image to leave out.
   * - ``--zoom N``
     - ``1``
     - Sub-samples taken per output cell along each axis and averaged into it.
   * - ``--latlon-type NAME``
     - ``centric``
     - Latitude and longitude system: ``centric``, ``graphic``, or ``squashed``.
   * - ``--lon-direction {east,west}``
     - ``east``
     - Direction longitude increases in.
   * - ``--photometric-model NAME``
     - ``none``
     - Photometric correction applied while reprojecting: ``none``,
       ``lambert``, ``lommel-seeliger``, or ``minnaert``.
   * - ``--no-dynamic``
     - off *(the mosaic grows as needed)*
     - Hold the mosaic to the latitude and longitude range given, clipping
       anything outside it, instead of growing to fit each new image.
   * - ``--resolution-threshold F``
     - ``1.0``
     - How much better the new image's resolution must be before it replaces a
       pixel already in the mosaic.
   * - ``--copy-slop N``
     - ``0``
     - Extra pixels copied around each copied pixel, which fills in the
       isolated gaps a coarse grid otherwise leaves.
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy data type the brightness array is stored in.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy data type the geometry arrays are stored in.

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

Reading records from the results index
--------------------------------------

Given ``--results-index-db``, each image's navigation record is read as one row
of the results index instead of one metadata document. The results index stores the
fields the choice above reads, and it reads them by the same rules, so a value a
run would apply is a value the results index holds, and a value a run would
refuse is a value it holds nothing for. For every metadata document that could
be read, the products a run builds are the same whether the option was given or
not.

A metadata document that could not be read is the exception, and it is a refusal
rather than a difference. The results index records such a file as one it holds
no navigation record for. Read directly, that same metadata document may well
carry a status and a pointing, so reporting it as an image that nothing ever
navigated would build one product from the results tree and a different one from
the results index without saying so. Instead that image fails, naming itself,
the results index, and the reason the ingest recorded, and the rest of the pass
continues. The remedy is to fix the metadata document and ingest the root again,
or to run the pass without ``--results-index-db``. A file the ingest could not
retrieve at all is not recorded, since it is worth retrying on the next pass,
and an image whose metadata document failed that way reads as one that nothing
navigated.

What a row cannot always keep is *why* a record supplies no pointing. One pair
of columns holds every way an offset can fail to be a pair of numbers, and one
matrix column holds a matrix or nothing, so several forms of metadata document
arrive as one row and the run summary counts them under the reason that row can
support. For every record a navigation wrote and an ingest stored, the two agree
on all of it: the same pointing, from the same values, counted under the same
reason.

Output files
------------

The default output format is FITS. Pass ``--format npz`` for compressed NumPy
archives instead. Both the per-image reprojections and the mosaic are written
directly under ``--output-dir``:

- Per-image reprojection:
  ``<output-dir>/<prefix>_<body_or_planet>_<image_stem>_reproj.<fmt>``
- Final mosaic: ``<output-dir>/<prefix>_<body_or_planet>_mosaic.<fmt>``

``<body_or_planet>`` is the body name for ``sd_mosaic_body`` and the planet name
for ``sd_mosaic_rings``. If ``--prefix`` is empty, which is the default, the
leading underscore is omitted too.

Logs go under the log root rather than beside the products: the run's own log,
and one log per image at
``{log_root}/reproj/<subject>/<results_path_stub>_<timestamp>.log``.

What the run reports about pointing
-----------------------------------

An image that has no usable navigation pointing is still reprojected, on
uncorrected pointing. The product looks the same either way, so each such image
is reported to the run's log with the reason, and the pass summary counts
them::

   Reprojection pass complete: 143 done, 0 skipped, 0 failed, 12 with
   uncorrected pointing.

Every pointing outcome other than a clean replacement of the camera frame -- a
fall back to the offset, kernels that already carry the correction, or no
correction at all -- is also tallied by reason, and the tally is reported at the
end of the pass::

   Pointing outcomes by reason: {'no_cmatrix_rotation_fitted': 12}

Not all of those reasons are shortfalls. ``pool_already_corrected`` is a
successful no-op: the furnished kernels already carry the corrected attitude, so
the image is right without anything being applied to it. It appears in the tally
so that a pass can be seen to have taken that path.

Every run names a navigation results root, so every image is asked for a
navigated pointing and the tally covers all of them.

Displaying reprojections and mosaics
====================================

``sd_mosaic_display_rings`` and ``sd_mosaic_display_body`` open an interactive
window for browsing reprojection and mosaic files::

    sd_mosaic_display_rings FILE [FILE ...] [options]
    sd_mosaic_display_body  FILE [FILE ...] [options]

Any number of files may be named. The window shows one at a time, with **Prev**
and **Next** buttons to step through them. A ring file goes to
``sd_mosaic_display_rings`` and a body file to ``sd_mosaic_display_body``.

::

    sd_mosaic_display_rings /data/mosaics/fring_2004_mosaic.fits

    sd_mosaic_display_body /data/mosaics/mimas_2004_MIMAS_N1234567890_reproj.fits

Display options
---------------

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--stretch-black F``
     - *(the data minimum)*
     - Initial black point of the image stretch.
   * - ``--stretch-white F``
     - *(the data maximum)*
     - Initial white point of the image stretch.
   * - ``--stretch-gamma F``
     - ``0.5``
     - Initial gamma of the image stretch, applied as ``data ** gamma``, so
       below 1 brightens the mid-tones.
   * - ``--show-radii``
     - off
     - Rings only. Overlay horizontal lines at the configured radii.
   * - ``--show-parallels``
     - off
     - Bodies only. Overlay lines of latitude.
   * - ``--show-meridians``
     - off
     - Bodies only. Overlay lines of longitude.
   * - ``--projection PROJ``
     - ``rect``
     - Bodies only. Projection the window opens in: ``rect``, ``polar_n``,
       ``polar_s``, ``mollweide``, or ``sphere3d``.
   * - ``--verbose``
     - off
     - Print additional diagnostic output.

Interactive controls
--------------------

- **Scroll wheel** zooms both axes together.
- **Shift + scroll** zooms the horizontal axis, which is longitude.
- **Ctrl + scroll** zooms the vertical axis, which is radius or latitude.
- **Shift + left-drag** rubber-bands a region to zoom to.
- **Left-drag** pans.
- **Right-click**, rings only, shows a radial profile at the longitude column
  clicked.
- **Save FOV** writes the current viewport to a PNG file.
- The **Black**, **White**, and **Gamma** sliders adjust contrast.
- The **Color by** radio buttons tint the image by one metadata field, such as
  radial resolution, angular resolution, phase, emission, or image number. On a
  ring window, fields the file does not carry, such as inertial longitude and
  true anomaly, are left out of the list.
- **Cursor info** reports the geometry under the pointer. For a mosaic, the
  source-image line gives the stored contributing name as ``imagename (#k)``.

Body projections
----------------

The **Projection** control in a body window's header chooses how the 360 by 180
degree latitude and longitude grid is laid out. There are five choices, and
``--projection`` opens the window in any of them.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Description
   * - Rectangular
     - Equirectangular (plate carree) display, the default.
   * - Polar North Stereographic
     - Stereographic projection centered on the north pole, which shows polar
       features with little distortion.
   * - Polar South Stereographic
     - The same, centered on the south pole.
   * - Mollweide
     - Equal-area global projection, far kinder to the polar regions than the
       rectangular one.
   * - 3D Sphere
     - Orthographic sphere. Left-drag rotates the globe in yaw and pitch,
       Shift + left-drag pans the sphere in the viewport, the scroll wheel
       zooms, and **Reset Zoom** fits the sphere to the window.

Outside the rectangular mode the parallels and meridians are drawn as curved
lines following the projection's geometry. The **Show parallels** and **Show
meridians** checkboxes in the Overlays panel, and the **Latitude axis ticks**
and **Longitude axis ticks** checkboxes in the header, work in every mode::

    sd_mosaic_display_body --projection sphere3d my_mosaic.npz
    sd_mosaic_display_body --projection polar_n polar_mosaic.npz

Mouse bindings by mode
----------------------

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Mode
     - Left drag
     - Shift+Left drag
     - Wheel
     - Reset Zoom
   * - Rectangular
     - Pan
     - Zoom to region
     - Zoom both axes
     - Fit image
   * - Polar N/S and Mollweide
     - Pan
     - Zoom to region
     - Zoom
     - Fit projection
   * - 3D Sphere
     - Rotate in yaw and pitch
     - Pan sphere
     - Zoom
     - Fit sphere

Running through cloud tasks
===========================

Cloud tasks is a work-queue package supplied by the Ring-Moon Systems Node. A
queue holds one task per unit of work, and a worker running on a cloud compute
instance takes tasks off the queue and performs them. The reprojection worker is
``sd_mosaic_cloud_tasks``. You do not run it yourself: the cloud task system
starts it on the compute instances you have asked for. What you do is write the
queue file and load it.
:doc:`user_guide_cloud_tasks` covers the arrangement in full.

``--output-cloud-tasks-file`` writes such a queue file for the images a run
would have processed, and does no reprojection or mosaicing:

.. code-block:: bash

   sd_mosaic_rings coiss_saturn \
       --volumes COISS_2001 \
       --planet SATURN \
       --radius-inner 70000 --radius-outer 140000 \
       --output-dir /data/mosaics --prefix saturn_main_rings_2004 \
       --output-cloud-tasks-file rings_tasks.json

   sd_mosaic_body coiss_saturn \
       --volumes COISS_2001 \
       --body-name MIMAS \
       --output-dir /data/mosaics --prefix mimas_2004 \
       --output-cloud-tasks-file mimas_tasks.json

Each task names one or more images and carries every setting for those images:
the output directory, the prefix, the format, and the whole ring or body mosaic
configuration. Each task also declares whether it is ring work or body work, so
one worker can drain a queue holding both. The worker writes its per-image
reprojection files under the task's output directory, named exactly as the local
commands name them.

The worker itself takes only the settings that describe its own environment and
credentials, and applies them to every task it handles:

.. code-block:: bash

   sd_mosaic_cloud_tasks [--config-file PATH] [--nav-results-root PATH] \
       [--results-index-db URL]

A worker has no run log, so it returns what a local run would have logged in the
task result instead: ``n_uncorrected`` counts the images reprojected with no
correction at all, and ``pointing_reasons`` holds the per-reason tally. The full
account for any one image is in that image's own log.

The mosaic pass is not cloud work. When the queue has drained, assemble the
mosaic locally with ``--skip-reproject``, giving the same output directory,
prefix, format, and mosaic configuration, so that the expected filenames match:

.. code-block:: bash

   sd_mosaic_rings coiss_saturn \
       --skip-reproject \
       --volumes COISS_2001 \
       --planet SATURN \
       --radius-inner 70000 --radius-outer 140000 \
       --output-dir /data/mosaics --prefix saturn_main_rings_2004

The queue file
--------------

The file ``--output-cloud-tasks-file`` writes is a JSON array of task objects:

.. code-block:: json

    {
        "task_id": "<dataset_name>-<label_file_name>-<index>",
        "data": {
            "mode": "rings",
            "dataset_name": "<dataset_name>",
            "arguments": {
                "output_dir": "<path or URL>",
                "prefix": "<prefix>",
                "format": "fits",
                "overwrite": false,
                "no_write_output_files": false,
                "image_name": null,
                "planet": "SATURN",
                "radius_inner": 70000,
                "radius_outer": 140000,
                "...": "<all remaining mosaic configuration fields>"
            },
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
* ``data.mode`` is ``"rings"`` or ``"body"``, and says which kind of mosaic work
  the task is. Because it is per task, one worker can drain a queue holding
  both.
* ``data.dataset_name`` is the dataset the images come from.
* ``data.arguments`` holds every output and mosaic setting of the local command,
  copied from the run that wrote the file, so that the worker reproduces the
  same reprojection configuration.
* ``data.files`` holds one or more file descriptions. Each requires
  ``image_file_url``, ``label_file_url``, and ``results_path_stub``, and may
  carry an ``index_file_row``, which may be ``null``.

.. toctree::
   :maxdepth: 2

   user_guide_reprojection_api
