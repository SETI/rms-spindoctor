============
Observations
============

Overview
========

The :mod:`spindoctor.obs` subsystem wraps an ``oops`` snapshot in a
navigation-aware class that adds backplane caching, extended-FOV
accessors, image masks, and per-instrument calibration hooks. Every
navigation pipeline takes an
:class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` instance as its
input and reads the per-image data, geometry, and instrument-specific
calibration through that object.

The class hierarchy splits responsibility across three axes:

- :class:`~spindoctor.obs.obs.Obs` — abstract observation root, wires
  :class:`~spindoctor.support.nav_base.NavBase` into the ``oops`` class tree so
  every concrete observation inherits ``config`` and ``logger``.
- :class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` — extends
  :class:`~spindoctor.obs.obs.Obs` and ``oops.observation.snapshot.Snapshot``.
  Adds the FOV / extended-FOV accessors, backplane caching, and the
  per-image mask helpers that every navigation model and technique
  consumes.
- :class:`~spindoctor.obs.obs_inst.ObsInst` — abstract mix-in carrying
  per-instrument calibration: the ``from_file`` constructor contract,
  the optical PSF, the per-instrument visual-magnitude window, and the
  per-image public-metadata projection.
- :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` — concrete mix-in
  of :class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` and
  :class:`~spindoctor.obs.obs_inst.ObsInst`. Per-mission subclasses derive
  from this base.

.. _coordinate-systems:

Coordinate systems
==================

Positions in a frame are written in one of two coordinate systems, and they
differ by half a pixel.

In **pixel-corner coordinates** a whole number falls on the boundary between
two pixels, so the first pixel spans 0.0 to 1.0 and its center is at 0.5. The
geometry layer works this way.

In **pixel-centric coordinates** a whole number falls on the center of a pixel,
so the first pixel spans -0.5 to 0.5 and its center is at 0.0. Anything that
draws into an array or measures out of one works this way.

Both are continuous: a position in either system is a floating-point number
like any other, and the systems say only where the whole numbers fall. Neither
is a system of whole numbers, so a position in one is not an index.

Converting between them adds or subtracts
:data:`~spindoctor.support.constants.PIXEL_CENTER_TO_CORNER_PX`::

    pixel_centric = pixel_corner  - PIXEL_CENTER_TO_CORNER_PX
    pixel_corner  = pixel_centric + PIXEL_CENTER_TO_CORNER_PX

Spell a conversion with that name rather than a bare ``0.5``, so that the
crossing can be found and so that a reader can tell it from the several other
reasons a half appears in this code -- an anti-aliasing ramp's midpoint, a
coverage fraction, a Minnaert exponent, a threshold on a distance.

Where each one is used
----------------------

Which system applies follows from the stage, so the question to ask of any
number is what produced it and what will consume it.

**Pixel-corner coordinates** are what the geometry layer answers in and what a
person writes and reads:

- Every field-of-view call: the observation's ``uv_from_ra_and_dec``, the field
  of view's ``uv_from_los`` and ``los_from_uv``, and its ``uv_los`` attribute,
  which names where the boresight falls on the detector. For the Cassini
  narrow-angle camera at full resolution that is ``(512.0, 512.0)`` on a
  1024-square detector -- the corner where the four central pixels meet -- and
  it halves with each summation mode, to 256.0 at 512 and 128.0 at 256.
- The ``origin`` and ``limit`` of a ``Meshgrid`` built for a field of view, and
  therefore the sample positions a ``Backplane`` is evaluated at. An
  unqualified backplane samples the center of every pixel, so cell ``(i, j)``
  holds the geometry of the point ``(i + 0.5, j + 0.5)``.
- The ``oops`` inventory: a body's ``center_uv``, and the unclipped bounds,
  which are the floor and ceiling of that body's projected extent.
- A :class:`~spindoctor.support.types.MutableStar` record's ``v`` and ``u``,
  which outlive the model that produced them.
- Every position a simulator scene states: a star's ``v`` / ``u``, a body's
  ``center_v`` / ``center_u``, the ring system's ``geometry.center_v`` /
  ``center_u``, and the distortion and stray-light centers in the ``optics``
  block. The center of a ``size_v`` by ``size_u`` frame is ``(size_v / 2,
  size_u / 2)``.
- Every position the pipeline reports: the positions in a navigation document,
  the coordinates in a log line, a coordinate readout in the mosaic viewer, the
  scene editor or the manual-navigation dialog, and a body center in a
  backplane sidecar.

**Pixel-centric coordinates** are what anything addressing an array works in:

- A coordinate grid built with ``np.arange``, ``np.mgrid``, ``np.meshgrid`` or
  ``np.indices`` over an image, and anything compared against one.
- Anything measured out of an array: a centroid, a peak located with ``argmax``
  and ``unravel_index``, a moment, a sub-pixel refinement added to one of
  those.
- Coordinates handed to ``scipy.ndimage`` for sampling, such as
  ``map_coordinates``.
- The edge distance transform on
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext`,
  whose zero locus is the centers of the detected edge pixels, and
  :func:`~spindoctor.support.distance_transform.sample_dt_bilinear`, which
  samples it.
- Every position a :class:`~spindoctor.feature.feature.NavFeature` carries:
  :attr:`~spindoctor.feature.geometry.StarGeometry.predicted_vu`, the
  ``vertices_vu`` of a limb, terminator or ring-edge polyline, and
  ``predicted_center_vu`` on every payload that has one. A technique compares
  these against something it measured in the image, so both ends of that
  comparison are in one system.
- The extended-FOV bounds --
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_v_min` and its three
  siblings -- which name the first and last pixel of the padded frame, and
  every clip and slice built from them.
- The simulator's render grids, including the oversampled plane a scene is
  drawn into before it is reduced to the detector.

**Displacements carry no coordinate system at all**, because they are
differences between two positions in one of them: a navigated offset
``(dv, du)`` and its sigmas and covariance, a star's per-exposure smear vector,
a separation, a radius, a width, a search window, and the extended-FOV margin
itself. None of them converts, and adding a half pixel to one is always
wrong.

A **half-open bounding box** is a third case that needs no conversion. The four
integers of ``bbox_extfov_vu`` are a slice range, so ``arr[v_min:v_max]``
covers rows ``v_min`` through ``v_max - 1``; those same integers are the
pixel-corner rectangle that covers exactly those pixels, because pixel ``i``
spans ``[i, i + 1)``. The two readings coincide.

A **mosaic grid** is not a pixel grid and answers the question differently
again: row ``r`` of a ring mosaic is the sample taken at radius
``radius_inner + r * radius_resolution``, and column ``c`` of a body mosaic is
the sample taken at longitude ``c * longitude_resolution``. A row names a
point, not the interval around it.

Where the conversion happens
----------------------------

A navigation model asks the geometry layer where something is and receives
pixel-corner coordinates; what it emits for a technique is pixel-centric,
because the technique compares it against something measured in the image
array. The crossing therefore happens inside every model, and where a model
writes it follows from where the value came from.

The body, ring and Titan models build their model images by asking for a
sampling grid over the extended frame. That frame's bounds are the array's, so
the half pixel goes on there, and a position later read back out of the
resulting array is already in the array's coordinates. A position that does not
come out of that array converts where it is taken: the body's and Titan's
predicted center, which the ``oops`` inventory states exactly, and the Titan
model's sunward pixel, which is read off the grid's own ``uv``. The ring
model's predicted center is the extended frame's own center, computed in the
array's coordinates, so it converts nothing.

A star record is different: it is handed on to several consumers and outlives
the model that produced it, so it keeps the geometry layer's coordinates and
each point of use converts --
:class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` where it
emits a feature and where it draws its overlay, the Titan model where it paints
a bright star into a contaminant mask, and the star catalog's edge cull, which
converts the four frame bounds once rather than the six positions of every
star.

A number crosses back at the boundary where it becomes a reader's. Every
position the pipeline states to a person -- in a log line, in a navigation
document, on an overlay label -- is pixel-corner in the nominal (unpadded)
frame, which is what compares directly against a scene file, an image viewer,
or another program's output. A position inside the extended-FOV margin is
reported as a negative number, which is where it is.

Three packages outside this repository have a convention of their own, and the
pipeline meets each of them. ``QPainter`` places a whole number on a pixel
boundary, so a widget coordinate is directly comparable with a pixel-corner
position. ``PIL.ImageDraw`` addresses cells, so it takes a pixel-centric one.
``psfmodel`` measures its evaluation offset from a pixel's lower edge, which is
the pixel-corner reading.

Getting this wrong is hard to detect from inside the pipeline. A navigated
offset is the difference between a predicted position and a measured one, so a
half-pixel error common to both cancels in everything computed from the same
pair: overlays land on target, residuals look clean, and the reported
confidence is unaffected. It survives only in the absolute answer, where it can
be found by comparing against an independently navigated frame or by predicting
star positions from the recorded attitude and centroiding the image.

ObsSnapshot
===========

:class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` is the navigation-side wrapper
around an ``oops`` snapshot. It exposes three families of helpers:

- **FOV / extended-FOV geometry.**
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.data_shape_uv` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.data_shape_vu`
  report the sensor shape;
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.fov_v_min` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.fov_v_max` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.fov_u_min` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.fov_u_max` name the first and last
  pixel of the sensor as whole numbers, pixel-centric (see
  :ref:`coordinate-systems`), so they run 0 to ``data_shape - 1``;
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_margin_v` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_margin_u` give the per-axis margin
  appended by :class:`~spindoctor.nav_orchestrator.instrument_config.InstrumentSettings`;
  the corresponding ``extfov_*`` accessors name the first and last pixel of the
  padded frame the same way, so
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_v_min` is
  ``-extfov_margin_v`` and
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_v_max` is
  ``data_shape_v + extfov_margin_v - 1``.  The padded array's shape comes from
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extdata_shape_vu` /
  :attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extdata_shape_uv`.
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_fov` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_extfov` clamp ``(u, v)`` coordinates
  into either grid;
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_rect_fov` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_rect_extfov` clamp full
  rectangles.
- **Mask and template constructors.**
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.make_fov_zeros` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.make_extfov_zeros` allocate float arrays of
  the right shape;
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.make_extfov_false` allocates the boolean
  equivalent;
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.unpad_array_to_extfov` trims an array
  that is at least as large as the extended-FOV grid down to it, keeping the
  top-left region; the caller it exists for is ``np.unpackbits``, which rounds
  its length up to a multiple of eight.
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.extfov_data_sensor_mask` returns a boolean
  mask that is ``True`` where the extended-FOV pixel corresponds to a
  real sensor pixel and ``False`` in the margin.
- **Inventory predicates.**
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_fov` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_extfov` consume an
  ``oops`` inventory entry and return whether the predicted body bounding box overlaps
  the sensor / extended FOV. Per-:class:`~spindoctor.nav_model.nav_model.NavModel`
  :meth:`~spindoctor.nav_model.nav_model.NavModel.instances_for_obs` hooks (e.g.
  :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.instances_for_obs`)
  call these to decide which bodies to instantiate.

Backplane caching and thread safety
-----------------------------------

Backplanes are cached in the underlying ``oops`` snapshot, so repeated
queries of the same backplane on the same
:class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` reuse the prior computation.
The cache is mutating state attached to the snapshot itself: every read of
:attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.bp`,
:attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.ext_bp`,
:attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.corner_bp`,
:attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.ext_corner_bp` or
:attr:`~spindoctor.obs.obs_snapshot.ObsSnapshot.center_bp` (or any helper
that builds an ``oops.Backplane`` from the snapshot, including
:meth:`~spindoctor.reproj.bodies.BodyMosaic.reproject` and
:func:`~spindoctor.reproj.cartographic_model.create_cartographic_model`)
allocates per-quantity arrays inside the snapshot's cache and reads
back any entries already present.

A single :class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` is therefore **not
safe for concurrent use across threads**. Two threads that simultaneously
sample backplanes through the same snapshot can race on the cache and
return inconsistent or partially-populated arrays. Code that needs to
parallelise over a single image must give each thread its own
:meth:`~spindoctor.obs.obs_inst.ObsInst.from_file` -constructed snapshot
instance; the navigation pipeline runs serially per image, so the
single-threaded contract is sufficient for the orchestrator's own use.

ObsInst
=======

:class:`~spindoctor.obs.obs_inst.ObsInst` is the per-instrument calibration
mix-in. It defines the abstract contract every per-mission subclass
must implement:

- :meth:`~spindoctor.obs.obs_inst.ObsInst.from_file` — load an image file and return the
  matching :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst`. Subclasses delegate the
  actual decode to ``oops.hosts.<mission>.<inst>.from_file``, then wrap the resulting
  ``oops`` snapshot.
- :meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf` — returns the per-instrument optical
  :class:`~psfmodel.PSF` (typically a
  :class:`~psfmodel.GaussianPSF`). Used by
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars` to predict
  the per-star detection footprint.
- :meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf_size` — returns the per-star kernel support
  rectangle in pixels.
- :meth:`~spindoctor.obs.obs_inst.ObsInst.star_min_usable_vmag` /
  :meth:`~spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag` — the per-instrument
  photometric window. Stars outside this window do not contribute predicted detections.
- :attr:`~spindoctor.obs.obs_inst.ObsInst.camera` — the camera that took the
  observation. Instruments with more than one camera distinguish them
  (Cassini ISS and Voyager ISS return ``'NAC'`` or ``'WAC'``, from the
  ``oops`` detector); single-camera instruments return their one camera's
  name (``'SSI'``, ``'LORRI'``). Pointing error is a property of the camera
  rather than the spacecraft — one Cassini WAC pixel is ten NAC pixels — so
  this is what ``navigate_image_files`` records as ``observation.camera``
  and what the statistics report groups offset distributions by.
- :meth:`~spindoctor.obs.obs_inst.ObsInst.get_public_metadata` — returns a JSON-friendly dict
  of per-image metadata fields (mission, instrument, exposure, filter wheel positions,
  etc.) for the per-image sidecar. Read the camera from
  :attr:`~spindoctor.obs.obs_inst.ObsInst.camera` rather than repeating a
  literal, so the name has one source.

The :attr:`~spindoctor.obs.obs_inst.ObsInst.inst_config` property exposes the per-instrument YAML block
loaded from ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` so subclass
methods can read instrument-specific knobs without hard-coding them.

Per-instrument subclasses
=========================

Concrete subclasses live in ``src/spindoctor/obs/`` and are registered (via
the :mod:`spindoctor.obs` package's ``__init__.py``) under a per-mission /
per-instrument key consumed by :class:`~spindoctor.dataset.dataset.DataSet`.
Shipping subclasses:

- :class:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS` — Cassini ISS
  NAC and WAC. Delegates to ``oops.hosts.cassini.iss.from_file``.
- :class:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS` — Voyager 1 / 2
  ISS NA and WA cameras. Delegates to
  ``oops.hosts.voyager.iss.from_file``.
- :class:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI` — Galileo SSI
  (uses ``full_fov=True`` to read the full sensor regardless of the
  on-chip ROI). Delegates to ``oops.hosts.galileo.ssi.from_file``.
- :class:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI` —
  New Horizons LORRI (passes ``calibration=False`` so the raw pixel
  values pass through). Delegates to
  ``oops.hosts.newhorizons.lorri.from_file``.
- :class:`~spindoctor.obs.obs_inst_sim.ObsSim` — simulated-image observation
  backed by a validated YAML scene (bodies, rings, stars); renders the frame via
  the forward model and exposes the navigator the filtered idealized scene view
  ``obs.nav_params`` (see :doc:`dev_guide_simulator`).

Each subclass overrides :meth:`~spindoctor.obs.obs_inst.ObsInst.from_file` to pull the right
``oops`` host, wires up the per-instrument PSF and photometric window, and forwards
per-image metadata into :meth:`~spindoctor.obs.obs_inst.ObsInst.get_public_metadata`.

Simulated-image instrument config: inherit / override / self-specify
--------------------------------------------------------------------

A simulated scene names an ``instrument`` (``coiss_nac``, ``vgiss``, ...) so its
rendered frame and its :class:`~spindoctor.obs.obs_inst_sim.ObsSim` go through the same
per-instrument units, noise, saturation, and PSF the navigator applies to a real
frame. :func:`~spindoctor.sim.instruments.resolve_sim_inst_config` maps that name to the
matching ``config_4N0_inst_*.yaml`` block (or the standalone ``sim`` block for the
``generic`` alias).

A scene may additionally carry an ``instrument_config`` mapping that is
**deep-merged** over the resolved block (nested mappings merge key-by-key; scalars
and lists replace). This gives three modes:

- **Inherit** -- omit ``instrument_config``; every physical parameter tracks the
  named instrument's config.
- **Override** -- supply only the keys to change; those are pinned to the scene and
  the rest still track the instrument.
- **Self-specify** -- name ``generic`` and override everything; the scene's config
  is fully its own.

**Why it exists, and what breaks without it.** A sim scene used as a navigation
fixture wants reproducible behavior. If every parameter is inherited live from a
real camera's config, then editing that camera's ``star_psf_sigma`` (or noise,
saturation, ...) silently shifts the rendered image and the recovered offset of
every sim scene that names it -- re-blessing baselines for a change that had
nothing to do with the simulator. Pinning a key via ``instrument_config`` decouples
it from the camera config: the merge produces a fresh dict, so a later camera-config
edit cannot reach a pinned key. Full self-specification (generic + complete
overrides) makes a scene immune to *all* instrument-config drift. The merge is
applied identically in both consumers -- :meth:`ObsSim.from_file
<spindoctor.obs.obs_inst_sim.ObsSim.from_file>` and
:func:`~spindoctor.sim.render.render_combined_model` -- so the rendered image and the
navigator's instrument settings stay consistent. The one precedence subtlety: the
top-level scene ``noise`` block (the primary noise control) still wins over
``instrument_config.noise`` for rendering, so ``instrument_config`` is the channel
for the instrument parameters that have no dedicated scene field (star PSF sigma,
data units, saturation / full-well DN, extfov margin, ...).

Adding a new instrument
=======================

The end-to-end checklist lives at
:doc:`dev_guide_extending`; the obs-side bullet is:

1. Subclass :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` in
   ``src/spindoctor/obs/obs_inst_<mission>_<inst>.py``.
2. Implement :meth:`~spindoctor.obs.obs_inst.ObsInst.from_file` (delegate to the matching
   ``oops.hosts.<mission>.<inst>`` host module),
   :meth:`~spindoctor.obs.obs_inst.ObsInst.star_psf`,
   :meth:`~spindoctor.obs.obs_inst.ObsInst.star_min_usable_vmag`,
   :meth:`~spindoctor.obs.obs_inst.ObsInst.star_max_usable_vmag`, and
   :meth:`~spindoctor.obs.obs_inst.ObsInst.get_public_metadata`.
3. Register the subclass in :mod:`spindoctor.obs` (``src/spindoctor/obs/__init__.py``)
   under the per-mission / per-instrument key the corresponding
   :class:`~spindoctor.dataset.dataset.DataSet` subclass passes to
   :meth:`~spindoctor.obs.obs_inst.ObsInst.from_file`.
4. Add the per-instrument config block at
   ``src/spindoctor/config_files/config_4N0_inst_<mission>_<inst>.yaml`` so
   :attr:`~spindoctor.obs.obs_inst.ObsInst.inst_config` carries the instrument's tuning knobs.

API reference
=============

The autodocumented API surface is at :doc:`/api_reference/api_obs`.
