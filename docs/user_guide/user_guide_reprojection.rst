=======================
Reprojection Mosaicing
=======================

The ``spindoctor.reproj`` package provides utilities for reprojecting planetary body
and ring images onto regular grids and accumulating multiple reprojected images
into mosaics.

Overview
--------

Two main classes are provided:

- :class:`~spindoctor.reproj.bodies.BodyMosaic` -- reprojects body images onto a
  latitude/longitude grid and accumulates them into a mosaic.
- :class:`~spindoctor.reproj.rings.RingMosaic` -- reprojects ring images onto a
  radius/longitude grid and accumulates them with true sparse longitude storage.

A standalone utility function :func:`~spindoctor.reproj.cartographic_model.create_cartographic_model`
projects a body mosaic back onto image coordinates for use as a navigation
correlation model.

Body reprojection and mosaicing
--------------------------------

Create a :class:`~spindoctor.reproj.bodies.BodyMosaic` once per body, then feed it
observations::

    from spindoctor.reproj import BodyMosaic

    mosaic = BodyMosaic(body_name='MIMAS')
    for obs in observations:
        result = mosaic.reproject(obs)
        mosaic.add(result)

    data = mosaic.to_bounded()  # BodyMosaicData

The mosaic grows automatically (``dynamic=True`` by default) to accommodate
each new reprojected image. You can pre-allocate a specific region::

    import math

    mosaic = BodyMosaic(
        body_name='MIMAS',
        lat_range=(-math.pi / 4, math.pi / 4),  # -45 to 45 degrees latitude
        lon_range=(0.0, math.pi),                # 0 to 180 degrees longitude
        dynamic=False,
    )

When ``lat_range`` or ``lon_range`` is ``None`` (the default), the mosaic uses
the full valid range for that axis. If ``dynamic=False`` and no range is
specified, the mosaic is pre-allocated to the full global grid.

Coordinate systems
^^^^^^^^^^^^^^^^^^

All angular values are in **radians**. The latitude/longitude coordinate
system is controlled by two parameters:

- ``latlon_type``: one of ``'centric'`` (default), ``'graphic'``, or
  ``'squashed'``.
- ``lon_direction``: ``'east'`` (default) or ``'west'``.

Choosing dtypes
^^^^^^^^^^^^^^^

By default, the reprojected brightness image uses ``float64``, the geometry
arrays (resolution, phase, emission, incidence) use ``float32`` (via the default
``metadata_dtype``), and the ``time`` field is always stored as ``float64``
regardless of the ``metadata_dtype`` argument to ``BodyMosaic``::

    from spindoctor.reproj import BodyMosaic
    import numpy as np

    # Defaults: image in float64, geometry in float32, time in float64
    mosaic = BodyMosaic(body_name='MIMAS')

    # Float32 image storage, float64 geometry (metadata); time stays float64
    mosaic = BodyMosaic(
        body_name='MIMAS',
        image_dtype=np.float32,     # smaller image storage
        metadata_dtype=np.float64,  # full-precision geometry
    )

The ``image_number`` field is always ``uint16``, capping a single mosaic at
65 535 contributing images.

Photometric correction
^^^^^^^^^^^^^^^^^^^^^^

Pass a photometric model to apply a correction during reprojection::

    from spindoctor.reproj import BodyMosaic, LambertModel

    mosaic = BodyMosaic(
        body_name='MIMAS',
        photometric_model=LambertModel(),
    )

Available models are :class:`~spindoctor.reproj.photometric_model.LambertModel`,
:class:`~spindoctor.reproj.photometric_model.LommelSeeligerModel`, and
:class:`~spindoctor.reproj.photometric_model.MinnaertModel`. When ``photometric_model``
is ``None`` (the default), pixel values are reprojected without correction.

Pixel conflict resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

``BodyMosaic`` uses the ``BEST_RESOLUTION`` strategy (see
:class:`~spindoctor.reproj.bodies.BodyMosaicMergeStrategy`): empty (masked) pixels are
filled unconditionally and existing data is replaced only when the new
observation has strictly better effective resolution (lower km/pixel).

Geometry limits when adding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject` applies ``max_incidence``,
``max_emission``, and ``max_resolution`` from the mosaic constructor so saved
per-image products stay within those bounds. :meth:`~spindoctor.reproj.bodies.BodyMosaic.add`
can apply the **same** limits again when merging saved
:class:`~spindoctor.reproj.bodies.BodyReprojResult` objects (for example after
``--skip-reproject``), and can optionally **override** them per call.

Keyword-only arguments ``max_incidence``, ``max_emission``, and ``max_resolution``
default to :data:`~spindoctor.reproj.USE_MOSAIC_LIMITS`, meaning each limit matches the
value given when the ``BodyMosaic`` was constructed. Pass a numeric value in
**radians** (incidence/emission) or **km/pixel** (resolution) to use a
different cutoff for that ``add()`` only; pass ``None`` to disable that cutoff
for that call (pixels are still constrained by the merge strategy and valid
``repro.img`` mask).

The ``sd_mosaic body`` CLI always passes these three arguments explicitly,
using the same ``--max-incidence``, ``--max-emission``, and ``--max-resolution``
values as for reprojection (degrees / km/pixel on the CLI; incidence and
emission are converted to radians before ``add()``).

Longitude wraparound
^^^^^^^^^^^^^^^^^^^^

The internal storage uses a shifted circular buffer so that data spanning
the 0/2\ |pi| boundary (e.g., a body centered on the meridian) is handled
correctly. The retrieval methods unwrap longitude automatically.

Retrieval methods
^^^^^^^^^^^^^^^^^

All retrieval methods return a :class:`~spindoctor.reproj.bodies.BodyMosaicData`
frozen dataclass with masked arrays for image data, resolution, phase,
emission, incidence, observation time and image-number metadata, plus
per-contributing-image sub-solar and sub-observer longitudes and latitudes
(see below):

- :meth:`~spindoctor.reproj.bodies.BodyMosaic.to_bounded` -- return the mosaic
  clipped to the data bounds or a user-specified range.
- :meth:`~spindoctor.reproj.bodies.BodyMosaic.to_full` -- return the full
  -|pi|/2 to |pi|/2 x 0 to 2\ |pi| grid.
- :attr:`~spindoctor.reproj.bodies.BodyMosaic.bounds` -- the current (lat, lon)
  extents of accumulated data, or ``None`` if the mosaic is empty.

Ring reprojection and mosaicing
---------------------------------

:class:`~spindoctor.reproj.rings.RingMosaic` works similarly but uses **sparse**
longitude storage: only longitude columns that contain at least one valid
pixel are stored. This is memory-efficient for the common case where only a
fraction of the ring plane is observed::

    from spindoctor.reproj import RingMosaic

    mosaic = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000)
    for obs in observations:
        result = mosaic.reproject(obs)
        mosaic.add(result)

    data = mosaic.to_sparse()  # RingMosaicData with longitude_antimask

The ``longitude_antimask`` field in the result indicates which full-grid
longitude bins are present in the sparse storage.

Choosing dtypes (rings)
^^^^^^^^^^^^^^^^^^^^^^^

The same ``image_dtype`` / ``metadata_dtype`` kwargs are available on
:class:`~spindoctor.reproj.rings.RingMosaic`::

    import numpy as np
    from spindoctor.reproj import RingMosaic

    mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        metadata_dtype=np.float64,  # full-precision geometry
    )

Orbit model
^^^^^^^^^^^

The ring geometry (eccentricity, ring plane) is handled by
:class:`~spindoctor.reproj.ring_orbit_model.RingOrbitModel`. Pre-defined instances
are available::

    from spindoctor.reproj import FRING_CORE, BRING_OUTER_EDGE

The :data:`~spindoctor.reproj.ring_orbit_model.FRING_CORE` instance has
``name='F-RING-CORE-ALBERS-2007'`` and uses the Albers et al. 2012 Table 3
Fit #2 elements; the ``2007`` suffix marks the epoch (2007-01-01T00:00:00Z)
at which the co-rotating frame is anchored.

**Longitude and radius conventions.** The interpretation of longitudes and of
``radius_inner`` / ``radius_outer`` depends on whether an orbit model is
supplied:

* ``orbit_model=None`` (the default): longitudes stored in reprojection
  results and mosaics are **inertial J2000 ring longitudes** — measured
  eastward from the ascending node of the ring plane on the J2000 reference
  plane — and ``radius_inner`` / ``radius_outer`` are **absolute ring radii
  in km**.
* ``orbit_model`` is set: each inertial longitude is converted to the
  **co-rotating frame** of the model before binning (mosaic column *i*
  corresponds to co-rotating longitude ``i × longitude_resolution``), and
  ``radius_inner`` / ``radius_outer`` are **signed offsets in km from the
  orbital radius at each (longitude, time)**. For an eccentric orbit the
  orbital radius varies between ``a (1 - e)`` and ``a (1 + e)``; the offset
  semantics make an eccentric ring appear as a **straight line** in the
  reprojection. ``radius_inner`` is therefore typically negative.

Examples::

    from spindoctor.reproj import RingMosaic, FRING_CORE

    # Inertial / absolute (no orbit model)
    mosaic_abs = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000)

    # Co-rotating / offset window centered on the F ring core
    mosaic_off = RingMosaic(
        'SATURN', radius_inner=-1000, radius_outer=1000,
        orbit_model=FRING_CORE,
    )

Pass a custom model via the ``orbit_model`` parameter::

    import math
    from spindoctor.reproj import RingMosaic, RingOrbitModel

    my_orbit = RingOrbitModel(
        name='MY-RING',
        a=140220.0,
        e=0.0,
        w0=0.0,
        dw=0.0,
        mean_motion=math.radians(581.964),
        epoch_utc='2007-01-01',
    )
    mosaic = RingMosaic('SATURN', radius_inner=-1000, radius_outer=1000,
                        orbit_model=my_orbit)

Mosaic compatibility
^^^^^^^^^^^^^^^^^^^^

:meth:`~spindoctor.reproj.rings.RingMosaic.add` validates that the reprojection it
is being given was produced with the **same** orbit model and the **same**
photometric model as the mosaic. Mixing settings would silently corrupt the
mosaic because radii and longitudes carry different meanings under different
orbit models. Mismatches raise :class:`ValueError`.

Merge strategy
^^^^^^^^^^^^^^

The ``merge_strategy`` parameter controls how longitude columns are updated
when multiple observations overlap::

    from spindoctor.reproj import RingMosaic, RingMosaicMergeStrategy

    mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        merge_strategy=RingMosaicMergeStrategy.BEST_RESOLUTION,
    )

- ``MOST_COVERAGE_THEN_RESOLUTION`` (default): fill empty longitude columns
  first; for already-present columns, replace only when the new data has
  better mean radial resolution.
- ``BEST_RESOLUTION``: replace an existing longitude column only when the new
  data has strictly better mean radial resolution.

Retrieval methods
^^^^^^^^^^^^^^^^^

- :meth:`~spindoctor.reproj.rings.RingMosaic.to_sparse` -- sparse storage (only
  present longitude columns). The ``longitude_antimask`` field marks present
  columns.
- :meth:`~spindoctor.reproj.rings.RingMosaic.to_bounded` -- dense array clipped to
  a longitude range.
- :meth:`~spindoctor.reproj.rings.RingMosaic.to_full` -- dense full 0 to 2\ |pi|
  longitude grid.

Saving and loading
------------------

All four result dataclasses (:class:`~spindoctor.reproj.bodies.BodyMosaicData`,
:class:`~spindoctor.reproj.bodies.BodyReprojResult`,
:class:`~spindoctor.reproj.rings.RingMosaicData`,
:class:`~spindoctor.reproj.rings.RingReprojResult`) support ``save()`` and ``load()``
methods. Two file formats are supported:

- **npz** (NumPy archive, default) — format inferred from a ``.npz``
  extension.
- **FITS** — format inferred from a ``.fits`` or ``.fit`` extension. Requires
  the ``astropy`` package (included as a runtime dependency).

Paths may be a string, a :class:`pathlib.Path`, or a :class:`filecache.FCPath`
(for example ``gs://…`` URIs handled by the project’s ``FileCache``). Remote
paths are fetched into the local cache on ``load()`` and written locally then
uploaded on ``save()``.

Body mosaic examples::

    from spindoctor.reproj import BodyMosaic, BodyMosaicData

    data = mosaic.to_bounded()

    # Save — format inferred from extension
    data.save('mimas.npz')                    # compressed npz
    data.save('mimas.npz', compress=False)    # uncompressed npz (faster I/O)
    data.save('mimas.fits')                   # FITS
    data.save('mimas.fits', format_='fits')    # explicit format

    # Load
    reloaded = BodyMosaicData.load('mimas.npz')
    reloaded = BodyMosaicData.load('mimas.fits')

Body reprojection result::

    from spindoctor.reproj import BodyReprojResult

    result = mosaic.reproject(obs, image_name='N1234567890')
    result.save('reproj.npz')
    reloaded = BodyReprojResult.load('reproj.npz')

Ring mosaic examples::

    import math
    from spindoctor.reproj import RingMosaicData

    data = ring_mosaic.to_bounded(longitude_range=(0.0, math.pi))
    data.save('saturn_rings.npz')
    reloaded = RingMosaicData.load('saturn_rings.npz')

Ring reprojection result::

    from spindoctor.reproj import RingReprojResult

    result = ring_mosaic.reproject(obs, image_name='N1234567890')
    result.save('ring_reproj.fits')
    reloaded = RingReprojResult.load('ring_reproj.fits')

When loading, the dtypes of all arrays are verified against the ``image_dtype``
and ``metadata_dtype`` fields stored in the file. A ``ValueError`` is raised if
any mismatch is detected, guarding against files produced by external tools
that may have coerced dtypes.

Image labels (reprojection and mosaic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~spindoctor.reproj.rings.RingReprojResult` and
:class:`~spindoctor.reproj.bodies.BodyReprojResult` carries an ``image_name`` string
(typically the source image stem). The ``save()`` and ``load()`` methods
preserve this value.

Each :class:`~spindoctor.reproj.rings.RingMosaicData` and
:class:`~spindoctor.reproj.bodies.BodyMosaicData` carries ``contributing_image_names``,
a tuple of strings in the same order as the ``image_number`` indices stored in
the mosaic (pixel value ``k`` refers to ``contributing_image_names[k]`` when
``k`` is in range). The tuple grows by one entry each time ``mosaic.add()``
finishes incorporating a reprojection and advances the internal image counter.

In Python, pass ``image_name=...`` to :meth:`~spindoctor.reproj.rings.RingMosaic.reproject`
and :meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject`. The ``sd_mosaic`` CLI
stores the dataset image stem per file by default; pass ``--image-name LABEL``
to use the same label for every image in the run instead.

Sub-solar and sub-observer geometry (body reprojection and mosaics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For bodies only, each :class:`~spindoctor.reproj.bodies.BodyReprojResult` records the
sub-solar and sub-observer longitude and latitude on the body at the
observation midtime, using the same ``latlon_type`` and ``lon_direction`` as
the reprojection. Fields are ``sub_solar_lon``, ``sub_solar_lat``,
``sub_observer_lon``, and ``sub_observer_lat`` (all **radians**). They are
written by ``save()`` / ``load()`` alongside the image and geometry arrays.

Each :class:`~spindoctor.reproj.bodies.BodyMosaicData` adds parallel **per-image**
1-D ``float64`` arrays—``sub_solar_lon_per_image``,
``sub_solar_lat_per_image``, ``sub_observer_lon_per_image``, and
``sub_observer_lat_per_image``—with length equal to the number of contributing
images. Index ``k`` matches ``contributing_image_names[k]`` and pixels whose
``image_number`` equals ``k``.

Older mosaic or reprojection files that omit the sub-observer fields load with
those values set to zero. Files that omit the per-image arrays load with empty
arrays for those fields.

The body mosaic viewer (:class:`~spindoctor.ui.mosaic_viewer.body_window.BodyMosaicWindow`)
shows sub-solar and sub-observer longitude and latitude in degrees in the
**Cursor Info** panel, indexed by the contributing image for the pixel under
the cursor (or image index ``0`` for a single reprojection file).

Cartographic navigation model
-------------------------------

Once a body mosaic is built, it can be projected back onto image coordinates
to produce a navigation model for correlation::

    from spindoctor.reproj import create_cartographic_model

    result = create_cartographic_model(
        mosaic.to_bounded(),
        obs,
        body_name='MIMAS',
    )
    if result is not None:
        model_img = result.model_img          # [v, u] float array
        ratio    = result.resolution_ratio   # mosaic res / image res

The function returns ``None`` if the mosaic has no valid data. The
``resolution_ratio`` field gives the median mosaic effective resolution divided
by the image center resolution; a value greater than 1.0 means the model
will be blurrier than the image.

.. |pi| replace:: *π*

.. _cli-mosaic:

Command-line mosaic generation
-------------------------------

The ``sd_mosaic_rings`` and ``sd_mosaic_body`` commands (entry points into
the single ``sd_mosaic`` program) reproject a dataset of images and combine
them into a mosaic using a two-pass workflow:

1. **Reprojection pass** — for each image in the dataset, load the observation,
   optionally apply its recorded navigation pointing, call
   ``BodyMosaic.reproject()`` / ``RingMosaic.reproject()`` (with ``image_name``
   set to that image's file stem, or to ``--image-name`` when that option is
   given), and save the result as
   ``<output-dir>/<prefix>_<body_or_planet>_<image_stem>_reproj.<fmt>`` (body
   name for ``sd_mosaic body``, planet name for ``sd_mosaic rings``). Existing
   files are skipped unless ``--overwrite`` is given, enabling interrupted runs
   to be resumed.

2. **Mosaic pass** — re-iterate the same image list, load each reprojection file
   that exists, call ``mosaic.add()`` (which extends ``contributing_image_names``
   in lockstep with ``image_number``), and save the final mosaic as
   ``<output-dir>/<prefix>_<body_or_planet>_mosaic.<fmt>``.

Either pass may be skipped with ``--skip-reproject`` / ``--skip-mosaic``.

Ring mosaics quick example (absolute radii, no orbit model)::

    sd_mosaic_rings coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --planet SATURN \
        --radius-inner 70000 \
        --radius-outer 140000 \
        --output-dir /data/mosaics \
        --prefix saturn_main_rings_2004

F ring mosaic example (offsets relative to the F ring core orbit)::

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

Body mosaics quick example::

    sd_mosaic_body coiss_saturn \
        --volumes COISS_2001 \
        --pds3-holdings-root /data/pds3 \
        --nav-results-root /data/nav_results \
        --body-name MIMAS \
        --output-dir /data/mosaics \
        --prefix mimas_2004

Pointing application
^^^^^^^^^^^^^^^^^^^^

When ``--nav-results-root`` is provided, ``sd_mosaic`` looks up the navigation
record for each image (written by ``sd_offset``) and applies the pointing it
records, preferring the exact form over its approximation. The record comes from
that image's ``_metadata.json`` file, or, when ``--results-index-db`` names a
results index, from one row of that index; both supply the same recorded values
and both are classified by the same ladder, so for every record ``sd_offset``
wrote the products are the same. (A record hand-built into a results tree can
take shapes ``sd_offset`` never writes, and a few of those the two storages
classify differently; they are listed under :ref:`reproj-index-differences`.)
The ladder:

* When the record carries a corrected camera attitude
  (``navigation_result.pointing.cmatrix``) that passes the reader's
  consistency gates, the observation's frame is replaced with that attitude
  and the field of view is left untouched. This is the same measurement as
  the pixel offset expressed exactly, and it is what a SPICE consumer of the
  corrected C-kernels sees for every image whose segment was written.
  ``sd_offset`` writes the attitude as nine row-major numbers, and the readers
  accept any nesting of nine finite real numbers that denotes one 3x3 matrix
  -- a 3x3 nesting and nine rows of one among them -- because the recorded
  value denotes the same rotation however it is bracketed. This holds whether
  the record is read from its file or from an index row: both read it through
  the same code.
* When there is no usable corrected attitude, the stored ``(dv, du)`` offset
  is applied to the observation's FOV via :class:`oops.fov.OffsetFOV`, exactly as
  every offset-corrected product has always been built. The reasons this
  happens, each counted in the run summary: ``no_cmatrix_rotation_fitted``
  (the navigation fitted a camera rotation, which records no corrected
  attitude), ``no_pointing_block`` (a simulated image, or a record predating
  the pointing schema), ``malformed_pointing`` (the pointing block cannot be
  used; also warned to the run log), and the gate refusals
  ``cmatrix_foreign_midtime`` (the record belongs to a different
  observation), ``cmatrix_baseline_mismatch`` (the kernel pool, the
  record, or the frame convention changed since navigation), and
  ``cmatrix_unknown_host`` (the observation's instrument has no frame
  mapping to gate against, which a record carrying a pointing block should
  never reach); the gate refusals are warned to the run log, and no product
  is ever built on a corrected attitude that failed a gate.
* When the reader finds the furnished kernel pool *already* answering the
  corrected attitude — corrected C-kernels furnished at load time — it
  applies nothing at all: the observation is already right, and applying
  either mechanism again would double-correct. This outcome is counted under
  ``pool_already_corrected``.
* When neither mechanism is usable (the file is absent, invalid JSON, a
  non-success status, or a null or malformed offset), a warning is logged
  and uncorrected pointing is used.

A record the selection already read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A run that also names an error filter --- ``--has-offset-error``,
``--has-no-offset-error``, ``--has-offset-spice-error`` or
``--has-offset-nonspice-error`` --- has read each selected image's document
already, because that is how the filter decided. The record travels with the
image and the pointing above is classified from it, so each such image's
document is read once for the whole run. On a cloud results root that is one
download saved per image processed: the enumeration and the pointing reader keep
separate download caches, so a second read really is a second download.

What is applied is what the document said when the selection was made. A
document rewritten or deleted while the run is going is therefore not noticed
for an image already selected. And the reasons that describe failing to read a
document --- ``no_metadata``, ``unreadable_metadata``, ``invalid_json``,
``metadata_not_an_object`` and ``unusable_metadata_path`` --- are not counted
for such an image, since a record is carried only for a document that was read
whole; the image is counted under the pointing its record supplies, exactly as
if the document had been read here. With ``--results-index-db`` nothing is
carried and each image's row is read as before.

One consequence worth knowing: for a result the kernel generator deliberately
omitted from the corrected kernels — the yielding WAC of a BOTSIM pair, or
any image with an omission reason — the readers still apply that image's
*own* recorded measurement, which is the better product for that image, while
a consumer of the corrected kernels sees the attitude the winning segment
implies. SpinDoctor's own products are authoritative for those images.

.. _reproj-index-differences:

Where a document and an index row are classified differently
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sd_results_index`` stores the fields the ladder reads, and it reads them
through the same code the ladder does, so a value a run would apply is a value
the index holds and a value it would refuse is a value the index holds nothing
for. For every document the ingest could read, the products a run builds are
therefore the same whether or not ``--results-index-db`` was given.

A document the ingest could *not* read is the exception, and it is a refusal
rather than a difference. The ingest records such a file as one it holds no
navigation record for -- a document naming no instrument or no image name, one
whose blocks are of some other shape, one naming a technique twice, or a file
that is not JSON at all. Read directly, that same document may carry a status
and a pointing, so reporting it as an image nothing navigated would build one
product from the tree and another from the index in silence. Instead the image
fails, naming itself, the index and the reason the ingest recorded, and the rest
of the pass continues. The remedy is to fix the document and ingest the root
again, or to run the pass without ``--results-index-db``. One kind of refusal is
deliberately recorded nowhere -- a file the ingest could not retrieve, which is
worth retrying on the next pass -- and an image whose document failed that way
reads as one nothing navigated.

What a column cannot always keep is *why* a record supplies no pointing. One
column pair holds every way an offset can fail to be a pair, and a matrix
column holds a matrix or nothing, so several document shapes reach one row and
the run summary counts them under the reason that row supports. For every
record a navigation wrote and an ingest stored, the two storages agree on all of
it: the same pointing, built from the same values, counted under the same
reason.

Output format
^^^^^^^^^^^^^

The default output format is FITS (``.fits``). Pass ``--format npz`` to use
compressed NumPy archives instead. Reprojection and mosaic files live directly
under ``<output-dir>``. Per-image logs are written under the configured log
root rather than beside the products; a cloud-tasks worker with no results root
of its own falls back to ``<output-dir>/logs``:

- Per-image reprojection: ``<output-dir>/<prefix>_<body_or_planet>_<image_stem>_reproj.<fmt>``
- Per-image reprojection log: ``{log_root}/reproj/<subject>/<results_path_stub>_<timestamp>.log``
- Final mosaic: ``<output-dir>/<prefix>_<body_or_planet>_mosaic.<fmt>``

If ``--prefix`` is empty (the default), the leading underscore is omitted.

``sd_mosaic`` accepts the same logging options as every other pipeline
program; see :doc:`user_guide_logging`.

An image with no usable navigation pointing is still reprojected, on
uncorrected pointing. Because the product looks the same either way, each one
is reported to the run's log with the reason, and the pass summary counts
them::

   Reprojection pass complete: 143 done, 0 skipped, 0 failed, 12 with
   uncorrected pointing.

Every pointing outcome other than the clean C-matrix application — an offset
fallback, an already-corrected pool, or no correction at all — is additionally
tallied per reason and the tally reported at the end of the pass::

   Pointing outcomes by reason: {'no_cmatrix_rotation_fitted': 12}

Only some of those reasons are shortfalls. ``pool_already_corrected`` is a
successful no-op: the furnished kernels already carry the corrected attitude,
so the image is right without anything being applied to it. It appears in the
tally so that a pass can be told to have taken that path, not because anything
about it needs acting on.

A cloud-task worker has no run log, so it returns the same information in the
task result instead, as ``n_uncorrected`` (images with no correction at all)
with the per-reason tally under ``pointing_reasons``. The full explanation
for any one image is in that image's log. A run given no
``--nav-results-root`` at all is not counted: nothing was asked for, so
nothing is missing.

Cloud-tasks entry point
^^^^^^^^^^^^^^^^^^^^^^^

Queue-driven reprojection is supported by ``sd_mosaic_cloud_tasks``. Each
task payload names one or more images, carries every per-task parameter
(output directory, mosaic geometry, body/planet, etc.), and declares its
``mode`` (``"rings"`` or ``"body"``). A single worker process can therefore
drain a queue that mixes ring and body tasks. The worker reprojects the named
images and writes per-image files under the task's ``output_dir`` using the
same naming convention as the local driver. The final mosaic-combination pass
is **not** performed by the cloud-tasks worker; after all tasks complete, run
the local driver with ``--skip-reproject`` to assemble the mosaic from the
accumulated reprojection files.

The cloud-tasks worker accepts only three CLI flags, all environment/credential
scoped and shared across every task the worker handles:

* ``--config-file PATH`` (may be repeated)
* ``--nav-results-root PATH``
* ``--results-index-db URL``

All other parameters that the local ``sd_mosaic_rings`` /
``sd_mosaic_body`` accept (``--output-dir``, ``--prefix``, ``--format``,
``--overwrite``, ``--image-name``, ``--no-write-output-files``, and the full
ring-/body-mosaic configuration such as ``--planet``, ``--body-name``,
``--radius-inner``, ``--lat-resolution``, ``--photometric-model`` etc.) are
passed per-task inside the task JSON. Invoke the worker with:

.. code-block:: bash

   sd_mosaic_cloud_tasks [--config-file PATH] [--nav-results-root PATH] \
       [--results-index-db URL]

To build a ready-to-load task-queue JSON file from the local driver without
running any reprojection, use ``--output-cloud-tasks-file``:

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

The task file is a JSON array of task objects:

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
                "...": "<all remaining mosaic-configuration fields>"
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

Fields:

* ``task_id``: unique string identifier built from the dataset name, the
  first image's label filename, and the enumeration index.
* ``data.mode``: ``"rings"`` or ``"body"``. Selects the mosaic factory and
  reprojection function for this task. Because the mode is per-task, a single
  ``sd_mosaic_cloud_tasks`` worker can drain a queue that contains both ring
  and body tasks.
* ``data.dataset_name``: one of the supported dataset names.
* ``data.arguments``: a dictionary whose keys are the argparse destinations
  produced by the local driver's Output group plus either the body- or
  ring-mosaic group (``body_name`` / ``lat_resolution`` / ... for body mode;
  ``planet`` / ``radius_inner`` / ``radius_outer`` / ... for rings). Every
  non-flow-control argument of the local ``sd_mosaic`` driver is copied
  here verbatim by ``--output-cloud-tasks-file`` so that the worker can
  reconstruct the exact same reprojection configuration.
* ``data.files``: one or more file descriptors with required fields
  ``image_file_url``, ``label_file_url``, ``results_path_stub``, and an
  optional ``index_file_row`` (metadata, may be ``null``).

When all reprojection tasks have drained, assemble the mosaic with the local
driver using the same ``--output-dir`` / ``--prefix`` / ``--format`` and the
same mosaic-configuration flags (so the expected output file names match):

.. code-block:: bash

   sd_mosaic_rings coiss_saturn \
       --skip-reproject \
       --volumes COISS_2001 \
       --planet SATURN \
       --radius-inner 70000 --radius-outer 140000 \
       --output-dir /data/mosaics --prefix saturn_main_rings_2004

Common options reference
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--output-dir DIR``
     - *(required)*
     - Directory for output files.
   * - ``--prefix STR``
     - ``''``
     - Filename prefix.
   * - ``--format {fits,npz}``
     - ``fits``
     - Output file format.
   * - ``--overwrite``
     - ``False``
     - Re-compute and overwrite existing per-image reprojection files.
   * - ``--skip-reproject``
     - ``False``
     - Skip the reprojection pass.
   * - ``--skip-mosaic``
     - ``False``
     - Skip the mosaic-building pass.
   * - ``--nav-results-root DIR``
     - ``None``
     - Root written by ``sd_offset``; enables pointing application.
   * - ``--results-index-db URL``
     - ``None``
     - Connection URL of a results index built by ``sd_results_index``. Each
       image's navigation record is then read as one database row instead of
       one file, which on a cloud results root replaces a round trip per image
       with a query. The index must already hold a completed ingest of the
       root named by ``--nav-results-root``, and its rows are a snapshot of
       the tree as of that ingest. Omitting the option names no index, and the
       results tree is read directly. ``--results-index-db none`` names no index
       either, which is how a machine that sets the option through
       configuration or through ``NAV_RESULTS_INDEX_DB`` reads the files.
   * - ``--dry-run``
     - ``False``
     - Print what would be done without writing files.
   * - ``--image-name LABEL``
     - *(use each file's stem)*
     - Override the ``image_name`` stored on every reprojection and the names
       listed in ``contributing_image_names`` on the mosaic.

Ring-specific options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--planet NAME``
     - *(required)*
     - Planet name (e.g. ``SATURN``).
   * - ``--radius-inner KM``
     - *(required when ``--orbit-model none``)*
     - Inner mosaic radius (absolute km). Mutually exclusive with
       ``--radius-inner-offset``.
   * - ``--radius-outer KM``
     - *(required when ``--orbit-model none``)*
     - Outer mosaic radius (absolute km). Mutually exclusive with
       ``--radius-outer-offset``.
   * - ``--radius-inner-offset KM``
     - *(required when ``--orbit-model`` is not ``none``)*
     - Inner-radius offset (km) from the orbit model radius at each
       (longitude, time); typically negative (e.g. ``-1000``). Mutually
       exclusive with ``--radius-inner``.
   * - ``--radius-outer-offset KM``
     - *(required when ``--orbit-model`` is not ``none``)*
     - Outer-radius offset (km) from the orbit model radius at each
       (longitude, time); typically positive. Mutually exclusive with
       ``--radius-outer``.
   * - ``--longitude-resolution DEG``
     - ``0.02``
     - Column pitch (degrees/pixel).
   * - ``--radius-resolution KM``
     - ``5.0``
     - Row pitch (km/pixel).
   * - ``--orbit-model {none,f_ring_core_albers_2007,bring_outer_edge}``
     - ``none``
     - Ring orbit model for co-rotating longitude and offset radii (see
       below).
   * - ``--merge-strategy {best_resolution,most_coverage_then_resolution}``
     - ``most_coverage_then_resolution``
     - Conflict-resolution strategy.
   * - ``--margin N``
     - ``3``
     - Edge pixels to exclude.
   * - ``--zoom N or R,L``
     - ``1``
     - Sub-samples taken per output cell and averaged into it. The image
       itself is not interpolated. Raising it smooths the mosaic and fills
       cells a single sample would miss, and costs run time as the product of
       the radial and longitudinal factors -- the square of a single ``N``.
       ``R,L`` sets the two separately.
   * - ``--no-omit-shadow``
     - *(flag; default: shadow masked)*
     - Include pixels inside the planet shadow.
   * - ``--longitude-range START END``
     - ``None``
     - Restrict reprojected longitude range (degrees).
   * - ``--radius-range INNER OUTER``
     - ``None``
     - Restrict reprojected radius range (km).
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy dtype for the brightness array.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy dtype for geometry metadata arrays.
   * - ``--photometric-model {none,lambert,lommel-seeliger,minnaert}``
     - ``none``
     - Photometric correction during ring ``reproject()`` (optional; same models as body).

.. _orbit-model-longitude:

Orbit model, longitude, and radius conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two coordinate conventions are tied to ``--orbit-model``:

**No orbit model (``--orbit-model none``, the default).**

* Longitudes stored in per-image reprojection files and the final mosaic are
  **inertial J2000 ring longitudes** — measured eastward from the ascending
  node of the ring plane on the J2000 reference plane, in degrees
  (internally radians). This is the default behavior of
  ``oops.backplane.Backplane.ring_longitude``.
* Radii are **absolute km**. The mosaic bounds are set by ``--radius-inner``
  and ``--radius-outer``; ``--radius-inner-offset`` /
  ``--radius-outer-offset`` are not allowed.
* Co-rotating longitude and the radial offset from the orbit are not
  defined; the viewer marks those fields as unavailable.

**With an orbit model (``--orbit-model f_ring_core_albers_2007`` or
``--orbit-model bring_outer_edge``).**

* Each inertial longitude is transformed to the **co-rotating frame** of
  that model before binning. Mosaic column *i* corresponds to co-rotating
  longitude ``i × longitude_resolution``; the column index no longer has a
  fixed relationship to J2000 north. The inertial longitude can be
  recovered from the co-rotating longitude using the orbit model and the
  per-column observation time.
* Radii are **signed offsets in km from the orbital radius at each
  (longitude, time)**. For an eccentric orbit, the orbital radius varies
  between ``a (1 - e)`` and ``a (1 + e)``; using offsets makes an
  eccentric ring appear as a straight line in the reprojection. The mosaic
  bounds are set by ``--radius-inner-offset`` (typically negative) and
  ``--radius-outer-offset`` (typically positive); ``--radius-inner`` /
  ``--radius-outer`` are not allowed.

All reprojections added to the same mosaic must agree on the orbit model
(and on the photometric model). ``RingMosaic.add()`` raises
:class:`ValueError` on a mismatch.

The pre-defined :data:`~spindoctor.reproj.ring_orbit_model.FRING_CORE` instance is
named ``F-RING-CORE-ALBERS-2007`` (Albers et al. 2012 Table 3 Fit #2; the
``2007`` indicates the co-rotation epoch, 2007-01-01T00:00:00Z).

Body-specific options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--body-name NAME``
     - *(required)*
     - Body to reproject (e.g. ``MIMAS``).
   * - ``--lat-resolution DEG``
     - ``0.1``
     - Row pitch (degrees/pixel).
   * - ``--lon-resolution DEG``
     - ``0.1``
     - Column pitch (degrees/pixel).
   * - ``--lat-range MIN MAX``
     - ``None``
     - Latitude extent (degrees); default full range.
   * - ``--lon-range MIN MAX``
     - ``None``
     - Longitude extent (degrees); default full range.
   * - ``--max-incidence DEG``
     - ``None``
     - Maximum incidence angle for valid pixels; default no limit.
   * - ``--max-emission DEG``
     - ``None``
     - Maximum emission angle for valid pixels; default no limit.
   * - ``--max-resolution KM``
     - ``None``
     - Maximum resolution (km/pixel) for valid pixels.
   * - ``--edge-margin N``
     - ``3``
     - Edge pixels to discard.
   * - ``--zoom N``
     - ``1``
     - Sub-pixel zoom factor.
   * - ``--latlon-type {centric,graphic,squashed}``
     - ``centric``
     - Latitude/longitude coordinate system.
   * - ``--lon-direction {east,west}``
     - ``east``
     - Longitude direction convention.
   * - ``--photometric-model {none,lambert,lommel-seeliger,minnaert}``
     - ``none``
     - Photometric correction to apply.
   * - ``--no-dynamic``
     - *(flag; default: dynamic growth enabled)*
     - Disable dynamic mosaic growth.
   * - ``--resolution-threshold F``
     - ``1.0``
     - Improvement factor required to overwrite a pixel.
   * - ``--copy-slop N``
     - ``0``
     - Extra pixels around each copied pixel to reduce artifacts.
   * - ``--image-dtype DTYPE``
     - ``float64``
     - NumPy dtype for the brightness array.
   * - ``--metadata-dtype DTYPE``
     - ``float32``
     - NumPy dtype for geometry metadata arrays.

Command-line mosaic display
----------------------------

The ``sd_mosaic_display_rings`` and ``sd_mosaic_display_body`` commands
(entry points into the single ``sd_mosaic_display`` program) open an
interactive PyQt6 window for browsing reprojection and mosaic files. Multiple
files can be passed; the window shows one file at a time and includes
**Prev / Next** navigation buttons.

Ring display quick example::

    sd_mosaic_display_rings /data/mosaics/fring_2004_mosaic.fits

Body display quick example::

    sd_mosaic_display_body /data/mosaics/mimas_2004_MIMAS_N1234567890_reproj.fits

Display options
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``--stretch-black F``
     - auto
     - Initial black-point for image stretch.
   * - ``--stretch-white F``
     - auto
     - Initial white-point for image stretch.
   * - ``--stretch-gamma F``
     - ``0.5``
     - Initial gamma (``data ** gamma`` convention; < 1 brightens mid-tones).
   * - ``--show-radii``
     - ``False``
     - (Rings) Overlay green horizontal lines at user-configured radii.
   * - ``--show-parallels``
     - ``False``
     - (Bodies) Overlay latitude parallel lines.
   * - ``--show-meridians``
     - ``False``
     - (Bodies) Overlay longitude meridian lines.

Interactive controls
^^^^^^^^^^^^^^^^^^^^

- **Scroll wheel** — zoom both axes simultaneously.
- **Shift + scroll** — zoom the X axis (longitude) only.
- **Ctrl + scroll** — zoom the Y axis (radius / latitude) only.
- **Shift + left-drag** — rubber-band zoom to a selected region.
- **Left-drag** — pan.
- **Right-click** (rings only) — display a radial profile at the clicked
  longitude column.
- **Save FOV** button — save the current viewport to a PNG file.
- **Stretch sliders** (Black / White / Gamma) — adjust contrast.
- **Color by** radio buttons — tint the image by a per-column or per-pixel
  metadata field (radial resolution, angular resolution, phase, emission,
  image number, etc.). On the ring window, options that required ephemeris
  columns not present in the file (inertial longitude, true anomaly) are omitted.
- **Cursor info** — for mosaics, the source-image line uses stored contributing
  names in the form ``imagename (#k)`` when available.

Projection selector (body mosaics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **Projection** combo box in the body-mosaic window header selects how the
360 x 180 degree lat/lon grid is displayed. Five modes are available:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Description
   * - Rectangular
     - Default equirectangular (plate carrée) display. All existing
       controls work as before.
   * - Polar North Stereographic
     - Stereographic projection centered on the north pole. Best for
       inspecting polar features with low distortion.
   * - Polar South Stereographic
     - Same as Polar North but centered on the south pole.
   * - Mollweide
     - Equal-area global projection. Polar regions are far less distorted
       than in Rectangular mode.
   * - 3D Sphere
     - Orthographic sphere view. Left-drag rotates the globe
       (yaw/pitch); Shift+Left-drag pans the sphere within the
       viewport; scroll wheel zooms; **Reset Zoom** fits the sphere to
       the window.

In all non-rectangular modes the graticule (parallels and meridians) is drawn
as curved polylines that follow the projection geometry. The **Show parallels**
and **Show meridians** checkboxes in the Overlays panel and the **Latitude axis
ticks** / **Longitude axis ticks** checkboxes in the header control the overlay
in every mode.

The ``sd_mosaic_display_body`` command accepts a ``--projection`` flag to
start in a non-default mode::

    sd_mosaic_display_body --projection sphere3d my_mosaic.npz
    sd_mosaic_display_body --projection polar_n  polar_mosaic.npz

Valid values for ``--projection`` are ``rect``, ``polar_n``, ``polar_s``,
``mollweide``, and ``sphere3d``.

Mouse bindings summary
^^^^^^^^^^^^^^^^^^^^^^^^

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
   * - Polar N/S / Mollweide
     - Pan
     - Zoom to region
     - Zoom
     - Fit projection
   * - 3D Sphere
     - Rotate (yaw/pitch)
     - Pan sphere
     - Zoom
     - Fit sphere
