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
from your own code when you want control that the programs do not offer. This
chapter covers the programs, which is what most work needs.
:doc:`user_guide_reprojection_api` covers the package.

The programs work entirely through the package, so every option a program takes
is something the package can be asked to do. The package can do more. Projecting
a finished body mosaic back onto an image's pixels, retrieving a mosaic as an
array over its own bounds or over the full grid, saving and reloading a mosaic or
a single reprojection, and reading the per-cell geometry out of one are all
package-only; no option of these programs reaches them.

.. toctree::
   :hidden:

   user_guide_reprojection_api

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
       a round trip per image with a query. The results index must already hold a
       completed ingest of the root named by ``--nav-results-root``, and its
       rows are a snapshot of that root as of the ingest. Omitting the option
       names no results index, and the navigation results are read as files.
       ``--results-index-db none`` also names no results index, which is how a machine
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
   * - ``--profile`` / ``--no-profile``
     - off
     - Collect a performance profile of the run. ``--no-profile`` states the
       default explicitly, which is useful in a wrapper script that builds its
       command line from a variable.

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
     - ``none``, ``f_ring_core_albers_2007`` for the core of Saturn's F ring, or
       ``bring_outer_edge`` for the outer edge of Saturn's B ring. See
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
     - Pixels along each edge of the image to leave out, because the outermost
       rows and columns of a real frame often carry garbage. At least 1 is
       required, so a ring reprojection always drops the outermost row and
       column. (``sd_mosaic_body`` spells the same setting ``--edge-margin`` and
       does accept 0.)
   * - ``--zoom N`` or ``--zoom R,L``
     - ``1``
     - Sub-samples taken per output cell and averaged into it. **Run time goes up
       as the product of the radial and longitudinal factors**, so a single ``N``
       costs ``N`` squared. Raising this smooths the mosaic and fills cells a
       single sample would miss. The image itself is not interpolated: each
       sub-sample reads the one pixel containing it. ``R,L`` sets the radial and
       longitudinal factors separately.
   * - ``--no-omit-shadow``
     - off *(shadowed pixels are masked)*
     - Keep the ring pixels that lie inside the planet's shadow.
   * - ``--longitude-range START END``
     - *(the full circle)*
     - Reproject only this range of longitude, in degrees, under the same
       convention as the stored longitudes: inertial with no orbit model,
       corotating with one.
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
       ``lambert``, ``lommel-seeliger``, or ``minnaert``. See
       `Photometric models`_.

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
     - Pixels along each edge of the image to leave out, because the outermost
       rows and columns of a real frame often carry garbage. 0 keeps every pixel.
       (``sd_mosaic_rings`` spells the same setting ``--margin`` and requires at
       least 1.)
   * - ``--zoom N``
     - ``1``
     - Sub-samples taken per output cell along each axis and averaged into it.
       **Run time goes up as the square of N**, since it applies to both axes.
       Raising this smooths the mosaic and fills cells a single sample would
       miss. The image itself is not interpolated: each sub-sample reads the one
       pixel containing it.
   * - ``--latlon-type NAME``
     - ``centric``
     - Latitude and longitude system: ``centric``, ``graphic``, or ``squashed``.
       See `Latitude and longitude systems`_.
   * - ``--lon-direction {east,west}``
     - ``east``
     - Direction longitude increases in.
   * - ``--photometric-model NAME``
     - ``none``
     - Photometric correction applied while reprojecting: ``none``,
       ``lambert``, ``lommel-seeliger``, or ``minnaert``. See
       `Photometric models`_.
   * - ``--no-dynamic``
     - off *(the mosaic grows as needed)*
     - Hold the mosaic to the latitude and longitude range given, clipping
       anything outside it, instead of growing to fit each new image.
   * - ``--resolution-threshold F``
     - ``1.0``
     - Ratio, not a difference and not a distance. A new image replaces a mosaic
       pixel only where its resolution in kilometers per pixel, multiplied by this
       number, is still smaller than the resolution already there. At the default
       of ``1.0`` any improvement at all wins. ``2.0`` demands the new image be
       twice as fine. Below 1.0 the new image wins even when it is coarser, so
       values below 1.0 make later images override earlier ones.
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

Latitude and longitude systems
------------------------------

``--latlon-type`` chooses how a point on the body's surface is named. The three
systems agree at the equator and at the poles, and differ in between by an amount
that grows with the body's flattening: negligible on a nearly round moon, and a
degree or more on a strongly flattened one.

``centric``
    Planetocentric, the default. Latitude is the angle at the body's center
    between the equatorial plane and the line out to the surface point. This is
    the system most planetary work quotes, and it is the one to use unless you
    have a reason to want another.

``graphic``
    Planetographic. Latitude is the angle between the equatorial plane and the
    local surface normal, which is what a surveyor standing on the body would
    call level. Use it when you are matching a cartographic product that is
    stated in planetographic coordinates.

``squashed``
    The body surface's own internal coordinate, on which the ellipsoid becomes a
    sphere. It is not a cartographic system and nothing outside this software
    quotes it. Use it only when you want the grid rows to be evenly spaced in
    that internal coordinate.

``--lon-direction`` is separate and says whether longitude increases eastward,
which is the default, or westward.

Every reprojection accumulated into one mosaic must agree on the latitude and
longitude system and on the longitude direction. A mismatch stops the mosaic
pass.

Photometric models
------------------

``--photometric-model`` divides out a model of how brightly the surface should
scatter light at each pixel's own illumination and viewing angles, so that a
mosaic assembled from frames taken under different lighting does not show the
lighting as a seam. It is applied while reprojecting, so it is baked into each
reprojection file.

``none``
    The default. Brightness is stored as the image gave it. Use this when you
    want the measured values, when you intend to do your own photometry, or when
    the mosaic covers a narrow enough range of lighting that the seams do not
    matter.

``lambert``
    Divides each pixel by the cosine of its incidence angle. It uses the
    incidence angle alone and ignores where the camera was, which makes it the
    simplest choice and the one that brightens hardest near the terminator, where
    the cosine approaches zero. The cosine is floored at 0.01, so an incidence
    angle past about 89.4 degrees is corrected as if it were 89.4 degrees rather
    than without limit.

``lommel-seeliger``
    Multiplies each pixel by ``(cos i + cos e) / (2 cos i)``, where ``i`` is the
    incidence angle and ``e`` the emission angle. It models a surface whose
    reflectance goes as ``1 / (cos i + cos e)``, which is single scattering from a
    dark, rough surface, so it is the choice when the viewing angle varies across
    the mosaic as well as the lighting. Its incidence cosine is floored the same
    way Lambert's is.

``minnaert``
    Divides each pixel by ``cos(i)**k * cos(e)**(k-1)``. The exponent ``k`` is a
    limb-darkening parameter: at ``k = 1`` this is the Lambert correction, and at
    ``k = 0.5`` it gives many surfaces a uniform disc appearance. **These programs
    fix k at 0.5 and offer no option to change it**; a different ``k`` needs the
    importable package (see :doc:`user_guide_reprojection_api`). Both cosines are
    floored at 0.01.

Choosing between them: start from ``none`` if you want the measured values or
intend to do your own photometry. Use ``lambert`` when only the lighting varies
across the frames, ``lommel-seeliger`` when the viewing geometry varies too, and
``minnaert`` when a flat-looking disc is what you are after. All three are fixed
analytic models with fixed parameters and none is fitted to your data, so a
corrected mosaic is a better picture than the raw one and is not a photometric
measurement.

Ring reprojections accept the same four names, and the correction uses each ring
pixel's own incidence and emission angles.

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
of the results index instead of one metadata document. The results index stores
exactly the fields the choice above reads, and reads them by the same rules. For
every metadata document that could be read, the products a run builds are the
same whether the option was given or not.

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

One difference is visible only in the run's own reporting. Where a record
supplies no pointing, the results index cannot always keep *why*, so the
per-reason tally at the end of a pass may lump several causes together under one
reason. The products are identical either way, and so is which images got a
corrected pointing.

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

The filename records neither the orbit model nor the photometric model, and
neither is the grid resolution. So a second run that differs only in one of those,
writing into the same output directory with the same prefix, finds the first run's
reprojection files already in place and leaves them alone, since an existing
reprojection file is only recomputed with ``--overwrite``. The mismatch is caught
only when the mosaic pass reads them and stops. Give each such run its own
``--prefix`` or its own ``--output-dir``.

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

These are all the reasons that tally can name.

``pool_already_corrected``
    The furnished kernels already carried the corrected attitude, so nothing was
    applied and the image is right as it stands. This one is not a shortfall. It
    appears in the tally so that a pass can be seen to have taken that path.

``cmatrix_foreign_midtime``
    The exposure midtime in the record does not match this image's own midtime, so
    the record belongs to a different image.

``cmatrix_baseline_mismatch``
    The attitude the furnished kernels give now matches neither the uncorrected nor
    the corrected attitude in the record, so the kernels, the metadata document, or
    the camera frame convention has changed since the image was navigated.

``cmatrix_unknown_host``
    The image's instrument has no SPICE camera frame to check the record against.

``malformed_pointing``
    The recorded corrected attitude is not a proper rotation, or the record carries
    no exposure midtime.

``no_cmatrix_rotation_fitted``
    The navigation fitted a camera rotation, which records no corrected attitude.

``no_pointing_block``
    The record carries no pointing at all.

``navigation_did_not_succeed``
    The navigation status is not ``success``.

Five offset reasons
    ``null_offset``, ``missing_offset_key``, ``invalid_offset_type``,
    ``non_finite_offset``, and ``malformed_offset``. The recorded pixel offset is
    absent, is not a pair of numbers, or is not finite. Read from a results index
    all five arrive as ``null_offset``, because one pair of columns holds them all.

``no_metadata``
    Nothing recorded this image: there is no metadata document for it under the
    navigation results root, or no row for it in the results index.

Four metadata-reading reasons
    ``unusable_metadata_path``, ``unreadable_metadata``, ``invalid_json``, and
    ``metadata_not_an_object``. The metadata document's path could not be formed,
    or the file could not be read, or its contents are not valid JSON, or they are
    valid JSON but not a JSON object. These four arise only when the navigation
    results tree is read. With a results index, such a file has no record row and
    the image fails instead of being tallied, as described above.

``pool_already_corrected`` applies nothing, because nothing is needed. The six
reasons from ``cmatrix_foreign_midtime`` through ``no_pointing_block`` fall back
to the recorded pixel offset, which is why the tally is not a count of images
built on uncorrected pointing. The rest leave the image on uncorrected pointing,
and those are the ones the ``with uncorrected pointing`` count above covers.

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

The reprojection worker is ``sd_mosaic_cloud_tasks``, and
:doc:`user_guide_cloud_tasks` describes what cloud tasks is, how a task file is
loaded into a queue, and who runs the workers.

``--output-cloud-tasks-file`` writes such a task file for the images a run
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

The task format
---------------

The file ``--output-cloud-tasks-file`` writes is a JSON array of task objects. A
reprojection task looks like this:

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
