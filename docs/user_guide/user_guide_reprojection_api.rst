=================================
The Reprojection Python Package
=================================

The reprojection and mosaic tools are also an importable Python package,
``spindoctor.reproj``. Use it when you want to reproject observations you have
already loaded yourself, to control the grid and the merge rules from your own
code, or to read a reprojection or mosaic file back into arrays. Everything the
command-line programs do, they do through this package; see
:doc:`user_guide_reprojection` for those programs.

This chapter names classes, functions, and fields, because that is what you
type. All angles in the package are in **radians**, and all radii are in
kilometers. The command-line programs take degrees and convert.

Overview
========

The package exports two mosaic classes and one projection function:

- :class:`~spindoctor.reproj.bodies.BodyMosaic` reprojects body images onto a
  latitude/longitude grid and accumulates them into a mosaic.
- :class:`~spindoctor.reproj.rings.RingMosaic` reprojects ring images onto a
  radius/longitude grid and accumulates them with sparse longitude storage.
- :func:`~spindoctor.reproj.cartographic_model.create_cartographic_model`
  projects a finished body mosaic back onto the pixel grid of one observation.

Alongside them it exports the photometric models, the ring orbit models, the
merge-strategy enumerations, and the result dataclasses that reprojections and
mosaics are returned in.

Body reprojection and mosaicing
===============================

Create a :class:`~spindoctor.reproj.bodies.BodyMosaic` once per body, then feed
it observations::

    from spindoctor.reproj import BodyMosaic

    body_mosaic = BodyMosaic(body_name='MIMAS')
    for obs in observations:
        result = body_mosaic.reproject(obs)
        body_mosaic.add(result)

    data = body_mosaic.to_bounded()

Every constructor argument is keyword-only. The mosaic grows automatically
(``dynamic=True`` by default) to accommodate each new reprojected image. You can
pre-allocate a specific region instead::

    import math

    body_mosaic = BodyMosaic(
        body_name='MIMAS',
        lat_range=(-math.pi / 4, math.pi / 4),  # -45 to 45 degrees latitude
        lon_range=(0.0, math.pi),               # 0 to 180 degrees longitude
        dynamic=False,
    )

When ``lat_range`` or ``lon_range`` is ``None`` (the default), the mosaic uses
the full valid range for that axis. With ``dynamic=False`` and no range given,
the mosaic is pre-allocated to the full global grid.

Coordinate systems
------------------

The latitude/longitude coordinate system is set by two constructor arguments:

- ``latlon_type``: one of ``'centric'`` (the default), ``'graphic'``, or
  ``'squashed'``.
- ``lon_direction``: ``'east'`` (the default) or ``'west'``.

Array data types
----------------

By default the reprojected brightness image uses ``float64``, the geometry
arrays (resolution, phase, emission, incidence) use ``float32`` through the
default ``metadata_dtype``, and the ``time`` field is always ``float64``
whatever ``metadata_dtype`` says::

    import numpy as np
    from spindoctor.reproj import BodyMosaic

    # Defaults: image in float64, geometry in float32, time in float64
    body_mosaic = BodyMosaic(body_name='MIMAS')

    # Float32 image storage, float64 geometry; time stays float64
    body_mosaic = BodyMosaic(
        body_name='MIMAS',
        image_dtype=np.float32,
        metadata_dtype=np.float64,
    )

The ``image_number`` field is always ``uint16``, which caps a single mosaic at
65 536 contributing images, numbered 0 through 65 535. Adding the image that would
exceed that raises ``OverflowError``.

Photometric correction
----------------------

Pass a photometric model to apply a correction during reprojection::

    from spindoctor.reproj import BodyMosaic, LambertModel

    body_mosaic = BodyMosaic(
        body_name='MIMAS',
        photometric_model=LambertModel(),
    )

The available models are
:class:`~spindoctor.reproj.photometric_model.LambertModel`,
:class:`~spindoctor.reproj.photometric_model.LommelSeeligerModel`, and
:class:`~spindoctor.reproj.photometric_model.MinnaertModel`. When
``photometric_model`` is ``None``, the default, pixel values are reprojected
uncorrected.

Pixel conflict resolution
-------------------------

:class:`~spindoctor.reproj.bodies.BodyMosaic` uses the ``BEST_RESOLUTION``
strategy of :class:`~spindoctor.reproj.bodies.BodyMosaicMergeStrategy`. Empty
(masked) pixels are filled unconditionally, and existing data is replaced only
where the new observation has strictly better effective resolution, meaning
fewer kilometers per pixel.

Geometry limits when adding
---------------------------

:meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject` applies the
``max_incidence``, ``max_emission``, and ``max_resolution`` limits given to the
constructor, so a saved per-image reprojection already respects them.
:meth:`~spindoctor.reproj.bodies.BodyMosaic.add` can apply the same limits again
when it merges a saved
:class:`~spindoctor.reproj.bodies.BodyReprojResult`, and can override them for
one call.

The keyword-only arguments ``max_incidence``, ``max_emission``, and
``max_resolution`` on ``add()`` default to
:data:`~spindoctor.reproj.USE_MOSAIC_LIMITS`, which means each limit matches the
value the mosaic was constructed with. Pass a number -- radians for incidence
and emission, kilometers per pixel for resolution -- to use a different cutoff
for that call alone. Pass ``None`` to drop that cutoff for that call; the merge
strategy and the reprojection's own valid-pixel mask still apply.

Longitude wraparound
--------------------

Internal storage is a shifted circular buffer, so data spanning the 0/2\ |pi|
boundary, such as a body centered on the prime meridian, is accumulated
correctly. The retrieval methods unwrap longitude for you.

Retrieval methods
-----------------

Each retrieval method returns a frozen
:class:`~spindoctor.reproj.bodies.BodyMosaicData` dataclass holding masked
arrays for the image data, resolution, effective resolution, phase, emission,
incidence, observation time, and image number, plus the per-image sub-solar and
sub-observer
longitudes and latitudes described below.

- :meth:`~spindoctor.reproj.bodies.BodyMosaic.to_bounded` returns the mosaic
  clipped to the data bounds, or to a range you give.
- :meth:`~spindoctor.reproj.bodies.BodyMosaic.to_full` returns the full
  -|pi|/2 to |pi|/2 by 0 to 2\ |pi| grid.
- :attr:`~spindoctor.reproj.bodies.BodyMosaic.bounds` is the current
  (latitude, longitude) extent of the accumulated data, or ``None`` while the
  mosaic is empty.

Ring reprojection and mosaicing
===============================

:class:`~spindoctor.reproj.rings.RingMosaic` works the same way but stores
longitude **sparsely**: only longitude columns holding at least one valid pixel
are kept. That matters because a ring observation usually covers a small
fraction of the ring plane::

    from spindoctor.reproj import RingMosaic

    ring_mosaic = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000)
    for obs in observations:
        result = ring_mosaic.reproject(obs)
        ring_mosaic.add(result)

    data = ring_mosaic.to_sparse()

The planet name, the inner radius, and the outer radius are positional; every
other constructor argument is keyword-only. The ``longitude_antimask`` field of
the result says which full-grid longitude bins the sparse storage holds.

Array data types (rings)
------------------------

``image_dtype`` and ``metadata_dtype`` mean the same thing here as for a body
mosaic::

    import numpy as np
    from spindoctor.reproj import RingMosaic

    ring_mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        metadata_dtype=np.float64,  # full-precision geometry
    )

Orbit model
-----------

Ring geometry -- the semimajor axis, the eccentricity, and the precession -- is
carried by :class:`~spindoctor.reproj.ring_orbit_model.RingOrbitModel`. Two
instances are predefined::

    from spindoctor.reproj import FRING_CORE, BRING_OUTER_EDGE

:data:`~spindoctor.reproj.ring_orbit_model.FRING_CORE` is named
``F-RING-CORE-ALBERS-2007`` and uses the Albers et al. 2012 Table 3 Fit #2
elements. The ``2007`` in the name is the epoch the corotating frame is anchored
at, 2007-01-01T00:00:00Z.

Whether an orbit model is supplied changes what longitudes and radii mean:

* With ``orbit_model=None``, the default, the longitudes stored in
  reprojections and mosaics are **inertial J2000 ring longitudes**, measured
  eastward from the ascending node of the ring plane on the J2000 reference
  plane, and ``radius_inner`` and ``radius_outer`` are **absolute ring radii in
  kilometers**.
* With an orbit model set, each inertial longitude is converted to the
  **corotating frame** of that model before binning, so mosaic column *i* holds
  corotating longitude ``i * longitude_resolution``. ``radius_inner`` and
  ``radius_outer`` become **signed offsets in kilometers from the orbital radius
  at each longitude and time**. For an eccentric orbit that radius varies
  between ``a (1 - e)`` and ``a (1 + e)``, so offsets make an eccentric ring
  come out as a straight line in the reprojection. ``radius_inner`` is
  therefore usually negative.

Both forms::

    from spindoctor.reproj import RingMosaic, FRING_CORE

    # Inertial longitudes, absolute radii
    ring_mosaic_abs = RingMosaic('SATURN', radius_inner=70000, radius_outer=140000)

    # Corotating longitudes, a radius window around the F ring core
    ring_mosaic_off = RingMosaic(
        'SATURN', radius_inner=-1000, radius_outer=1000,
        orbit_model=FRING_CORE,
    )

A model of your own goes in through the same argument::

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
    ring_mosaic = RingMosaic('SATURN', radius_inner=-1000, radius_outer=1000,
                             orbit_model=my_orbit)

Mosaic compatibility
--------------------

:meth:`~spindoctor.reproj.rings.RingMosaic.add` checks that the reprojection it
is handed was produced with the same orbit model and the same photometric model
as the mosaic. Mixing them would corrupt the mosaic silently, because radii and
longitudes mean different things under different orbit models. A mismatch raises
:class:`ValueError`.

Merge strategy
--------------

``merge_strategy`` controls how longitude columns are updated where
observations overlap::

    from spindoctor.reproj import RingMosaic, RingMosaicMergeStrategy

    ring_mosaic = RingMosaic(
        'SATURN', radius_inner=70000, radius_outer=140000,
        merge_strategy=RingMosaicMergeStrategy.BEST_RESOLUTION,
    )

- ``MOST_COVERAGE_THEN_RESOLUTION``, the default, fills empty longitude columns
  first, and replaces a column that already holds data only where the new data
  has better mean radial resolution.
- ``BEST_RESOLUTION`` replaces an existing longitude column only where the new
  data has strictly better mean radial resolution.

Retrieval methods
-----------------

- :meth:`~spindoctor.reproj.rings.RingMosaic.to_sparse` returns the sparse
  storage, holding only the longitude columns that are present. The
  ``longitude_antimask`` field marks which those are.
- :meth:`~spindoctor.reproj.rings.RingMosaic.to_bounded` returns a dense array
  clipped to a longitude range.
- :meth:`~spindoctor.reproj.rings.RingMosaic.to_full` returns a dense array on
  the full 0 to 2\ |pi| longitude grid.

Saving and loading
==================

All four result dataclasses --
:class:`~spindoctor.reproj.bodies.BodyReprojResult`,
:class:`~spindoctor.reproj.bodies.BodyMosaicData`,
:class:`~spindoctor.reproj.rings.RingReprojResult`, and
:class:`~spindoctor.reproj.rings.RingMosaicData` -- have ``save()`` and
``load()`` methods, and two formats:

- **npz**, a NumPy archive, inferred from a ``.npz`` extension.
- **FITS**, inferred from a ``.fits`` or ``.fit`` extension. This needs
  ``astropy``, which is a runtime dependency of SpinDoctor.

A path may be a string, a :class:`pathlib.Path`, or a
:class:`filecache.FCPath`, so a ``gs://`` URI works. A remote path is fetched
into the local cache on ``load()``, and on ``save()`` the file is written
locally and then uploaded.

Body mosaic::

    from spindoctor.reproj import BodyMosaicData

    data = body_mosaic.to_bounded()

    data.save('mimas.npz')                   # compressed npz
    data.save('mimas.npz', compress=False)   # uncompressed npz, faster I/O
    data.save('mimas.fits')                  # FITS
    data.save('mimas.fits', format_='fits')  # format stated rather than inferred

    reloaded = BodyMosaicData.load('mimas.npz')
    reloaded = BodyMosaicData.load('mimas.fits')

Body reprojection::

    from spindoctor.reproj import BodyReprojResult

    result = body_mosaic.reproject(obs, image_name='N1234567890')
    result.save('reproj.npz')
    reloaded = BodyReprojResult.load('reproj.npz')

Ring mosaic::

    import math
    from spindoctor.reproj import RingMosaicData

    data = ring_mosaic.to_bounded(longitude_range=(0.0, math.pi))
    data.save('saturn_rings.npz')
    reloaded = RingMosaicData.load('saturn_rings.npz')

Ring reprojection::

    from spindoctor.reproj import RingReprojResult

    result = ring_mosaic.reproject(obs, image_name='N1234567890')
    result.save('ring_reproj.fits')
    reloaded = RingReprojResult.load('ring_reproj.fits')

On load, every array's data type is checked against the ``image_dtype`` and
``metadata_dtype`` recorded in the file, and a mismatch raises
:class:`ValueError`. That catches a file some other tool wrote after coercing
the types.

Image labels
------------

Each :class:`~spindoctor.reproj.bodies.BodyReprojResult` and
:class:`~spindoctor.reproj.rings.RingReprojResult` carries an ``image_name``
string, usually the stem of the source image file, and ``save()`` and ``load()``
preserve it. Pass ``image_name=`` to
:meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject` or
:meth:`~spindoctor.reproj.rings.RingMosaic.reproject` to set it.

Each :class:`~spindoctor.reproj.bodies.BodyMosaicData` and
:class:`~spindoctor.reproj.rings.RingMosaicData` carries
``contributing_image_names``, a tuple of strings in the same order as the
``image_number`` values stored in the mosaic, so pixel value ``k`` refers to
``contributing_image_names[k]``. The tuple gains one entry each time ``add()``
finishes incorporating a reprojection.

Sub-solar and sub-observer geometry
-----------------------------------

For bodies, each :class:`~spindoctor.reproj.bodies.BodyReprojResult` records the
sub-solar and sub-observer longitude and latitude on the body at the observation
midtime, in the same ``latlon_type`` and ``lon_direction`` as the reprojection.
The fields are ``sub_solar_lon``, ``sub_solar_lat``, ``sub_observer_lon``, and
``sub_observer_lat``, all in radians, and ``save()`` and ``load()`` carry them
alongside the image and geometry arrays.

Each :class:`~spindoctor.reproj.bodies.BodyMosaicData` adds the parallel
per-image one-dimensional ``float64`` arrays ``sub_solar_lon_per_image``,
``sub_solar_lat_per_image``, ``sub_observer_lon_per_image``, and
``sub_observer_lat_per_image``, each as long as the number of contributing
images. Index ``k`` matches ``contributing_image_names[k]`` and the pixels whose
``image_number`` is ``k``.

A file that does not hold the sub-observer fields loads with those values set to
zero, and one that does not hold the per-image arrays loads with empty arrays
for them.

Projecting a mosaic back onto image coordinates
===============================================

A finished body mosaic can be projected back onto the pixel grid of one
observation. For every pixel of that observation the latitude and longitude on
the body are computed from the observation geometry, and the mosaic is sampled
there by bilinear interpolation. The result is an image of what the mosaic says
that observation should look like, at that observation's resolution::

    from spindoctor.reproj import create_cartographic_model

    result = create_cartographic_model(
        body_mosaic.to_bounded(),
        obs,
        body_name='MIMAS',
    )
    if result is not None:
        model_img = result.model_img         # [v, u] float array
        ratio = result.resolution_ratio      # mosaic resolution / image resolution

``model_img`` is 0.0 outside the mosaic's coverage and where the body is not
visible. ``resolution_ratio`` is the median effective resolution of the mosaic
divided by the resolution at the center of the image; above 1.0 the mosaic is
the coarser of the two, so the projected image is blurrier than the real one.
The function returns ``None`` when the mosaic holds no valid data.

``latlon_type`` and ``lon_direction`` must match the values the mosaic was
built with, and the mosaic must come from
:meth:`~spindoctor.reproj.bodies.BodyMosaic.to_bounded` or
:meth:`~spindoctor.reproj.bodies.BodyMosaic.to_full`, both of which anchor the
grid at its first row and column as the sampling expects.

Thread safety
=============

None of the three entry points is safe to use concurrently on the same
observation. :meth:`~spindoctor.reproj.rings.RingMosaic.reproject` changes the
global oops precision setting while it runs.
:meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject` and
:func:`~spindoctor.reproj.cartographic_model.create_cartographic_model` both
build geometry from the observation you hand them. Give each thread its own
observation.

.. |pi| replace:: *π*
