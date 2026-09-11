==========
Backplanes
==========

The :mod:`backplanes` package writes per-pixel geometry products (longitude,
latitude, incidence / emission / phase angles, ring radius, resolution, ...)
for a navigated image, packaged as a multi-extension FITS file plus a sidecar
metadata JSON. Backplanes are the geometry input that downstream PDS4 bundles
ship and that operator-side analysis tools (mosaicing, photometric correction,
limb fitting) consume.

This chapter is the developer's reference for the pipeline's internals — the
phase structure, the per-source generation algorithms, the distance-aware
merge, and the FITS writer's HDU layout. The CLI walkthrough, configurable
backplane list, and FITS / sidecar shape from a user's perspective lives at
:doc:`/user_guide/user_guide_backplanes`.

Overview
========

A backplane is a sensor-shaped float array carrying one geometric quantity per
pixel. ``oops`` exposes dozens of backplane methods on its
:class:`~oops.backplane.Backplane` class — every quantity the
SPICE prediction can resolve at the pixel grid (planetographic latitude,
photon-arrival timestamps, sub-solar / sub-observer angles, ring-orbit phase,
...). The backplanes pipeline picks a configurable subset of those methods,
evaluates them once per applicable source (each body in the inventory, plus
the ring system), merges the per-source results into one master array per
backplane name (so the per-pixel output is unambiguous when a body silhouette
overlaps the rings), and writes the result.

The pipeline preserves the navigated pointing. ``sd_backplanes`` reads the
``_metadata.json`` produced by ``sd_offset``, applies the recorded pointing
to the :class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` — the corrected
C-matrix by frame replacement when the record carries a usable one, else the
``(dv, du)`` offset via :class:`oops.fov.OffsetFOV`, and when the furnished
kernel pool *already* answers the corrected attitude (corrected C-kernels
furnished at load time) it deliberately applies nothing at all, since either
mechanism would double-correct (see
:doc:`dev_guide_ck_kernels` for the reader mechanism and its fallback
ladder) — and *then* evaluates every backplane. Every output pixel is
therefore the geometry that the navigation step says it is — not the
geometry that the raw SPICE prediction would have produced before the
correction.

Pipeline overview
=================

Per-image, the driver runs three phases:

1. **Build the pointing-corrected snapshot.**  Read the per-image navigation
   record through the stage's
   :class:`~spindoctor.cli.reproj.pointing_source.PointingSource` -- the
   ``_metadata.json`` document under ``--nav-results-root``, or one row of the
   results index named by ``--results-index-db`` -- refuse to proceed if
   ``status != 'success'``, build the per-instrument
   :class:`~spindoctor.obs.obs_snapshot_inst.ObsSnapshotInst` with
   ``extfov_margin_vu=(0, 0)`` (backplanes are evaluated on the sensor
   only, not on the extended FOV used by navigation), apply the recorded
   pointing via :func:`~spindoctor.cli.reproj.offsets.select_pointing` /
   :func:`~spindoctor.cli.reproj.offsets.apply_pointing_to_obs` (the
   corrected C-matrix when usable, else the navigated offset via
   :class:`oops.fov.OffsetFOV`; an already-corrected kernel pool is left
   alone), and stash it as ``snapshot``.
2. **Evaluate per-source backplanes.**
   :func:`~spindoctor.cli.backplanes.backplanes_bodies.create_body_backplanes` walks
   every body in the per-image inventory and, for each, builds a clipped
   meshgrid around the body's bounding box (no oversampling — backplanes
   target sensor-resolution accuracy) and evaluates the configured body
   backplane methods against an ``oops.Backplane`` over that meshgrid.
   :func:`~spindoctor.cli.backplanes.backplanes_rings.create_ring_backplanes` evaluates
   the configured ring backplane methods against the snapshot's
   full-frame ``snapshot.bp``. Both functions return per-pixel arrays
   plus the per-source ``distance`` array used by phase 3.
3. **Distance-aware merge + write.**
   :func:`~spindoctor.cli.backplanes.merge.merge_sources_into_master` walks every
   pixel; for each pixel it picks the source with the smallest distance
   (closest body or ring intersection along the line of sight) and
   copies that source's per-backplane values into the master arrays.
   The merge also fills a per-pixel ``BODY_ID_MAP`` carrying the NAIF ID
   of the source that won at each pixel.  :func:`~spindoctor.cli.backplanes.writer.write_fits`
   serialises the master arrays and the body-ID map to FITS, attaching
   the ``BUNIT`` header from the per-backplane config, and writes a
   companion ``_backplane_metadata.json`` with per-body inventory and
   per-backplane min/max statistics.

Phase 1 skips the image if the navigation step did not converge, writing no
FITS for it; the downstream PDS4 driver also refuses to render a label for an
image whose backplane FITS is missing, so a single missing product propagates
cleanly.
A failure belongs to its image: the driver reports it against that image,
counts it, and goes on to the next one, closing the pass with a summary of
how many images were done, skipped and failed.

Entry points
============

``sd_backplanes`` and ``sd_backplanes_cloud_tasks``
(``src/spindoctor/cli/sd_backplanes.py`` and ``src/spindoctor/cli/sd_backplanes_cloud_tasks.py``)
are thin CLI wrappers around
:func:`~spindoctor.cli.backplanes.backplanes.generate_backplanes_image_files`. CLI flags,
selection options, and per-batch behaviour are documented at
:doc:`/user_guide/user_guide_backplanes`. Code that embeds backplane generation in a
Python pipeline calls the function directly.

Restrictions and assumptions
============================

- **Snapshots only.**  The pipeline supports
  :class:`~oops.observation.snapshot.Snapshot`-derived observations only;
  push-broom and other observation modes are rejected at the
  ``isinstance(obs, ObsSnapshot)`` check. Adding a non-snapshot path
  requires a different per-source meshgrid construction (rate-and-state
  geometry) that the existing bodies / rings helpers do not implement.
- **Sensor frame, not extfov.**  Backplanes are evaluated on the sensor
  pixel grid (``extfov_margin_vu=(0, 0)``). The extended-FOV margin used
  by navigation does not appear in the FITS file.
- **A successful record supplying no offset is processed, not refused.**
  The rule applies below the status gate of phase 1: a record whose
  ``status`` is anything but ``success`` is skipped before its offset is
  ever read.  A ``success`` record that reports ``offset = None``, or
  carries no ``offset`` key at all, and has no usable C-matrix is
  processed on uncorrected pointing with a warning, which means the
  resulting backplanes carry the raw SPICE prediction's geometry; the
  result reports
  ``pointing_source: 'none'``, ``uncorrected_pointing: true`` and the
  reason (``null_offset`` or ``missing_offset_key``).  The two are one
  class to this stage on purpose: the results index stores an absent
  offset and a null one in the same NULL column pair, so a stage that
  refused one and processed the other would build a product from a
  document and refuse the same image read as a row.  Operators who want
  to refuse those images can filter on ``confidence_tier`` before
  running the bundle step.
- **Per-body bounding-box evaluation.**  Body backplanes are evaluated
  on a meshgrid clipped to the body's predicted bounding box (no
  oversampling). Pixels outside any body's bounding box and outside
  the ring system have no backplane contribution and carry the masked
  value in the master arrays.

Per-source backplane generation
===============================

Bodies
------

:func:`~spindoctor.cli.backplanes.backplanes_bodies.create_body_backplanes` evaluates the
configured ``backplanes.bodies`` list against every body in the per-image
inventory. For each body:

1. Query the per-image inventory to get the body's predicted bounding
   box (``u_min_unclipped`` … ``v_max_unclipped``); clip into the
   sensor. Bodies with no overlap contribute nothing.
2. Build a meshgrid of pixel centres inside the clipped bounding box,
   build an :class:`oops.backplane.Backplane` over that meshgrid, and
   evaluate every method named in ``backplanes.bodies``.
3. Mask each per-pixel array against the body silhouette (the
   incidence-angle backplane returns finite where the line of sight
   intersects the body); embed the masked array into a sensor-shaped
   frame so the merge step can work pixel-by-pixel without per-body
   coordinate math.
4. Compute the body's per-pixel distance (the distance backplane on
   the same meshgrid) for the merge step.
5. Record the body's NAIF ID and the per-backplane min/max for the
   sidecar JSON.

Simulated bodies (when ``snapshot.is_simulated``) take the
``_create_simulated_body_backplane`` path: the per-pixel array is a
synthesised constant within the simulated body's mask
(``snapshot.sim_body_mask_map[body_name]`` if present, otherwise the body
slot in ``snapshot.sim_body_index_map`` matched against
``snapshot.sim_body_order_near_to_far``). Simulation produces deterministic
fake NAIF IDs so the merge step does not crash when a simulated body has
no SPICE entry.

Rings
-----

:func:`~spindoctor.cli.backplanes.backplanes_rings.create_ring_backplanes` evaluates the
configured ``backplanes.rings`` list against the snapshot's full-frame
``snapshot.bp``. Unlike bodies the ring backplanes do not need a
clipped meshgrid — the ring system covers a continuous range of radii
across the FOV, and ``oops``'s ring backplanes are cheap on a
sensor-shaped grid.

The ring step also produces a per-pixel ``distance`` array (distance from
the spacecraft to the ring intersection point along the line of sight)
that the merge step compares to per-body distances to decide which source
owns each pixel.

Distance-aware merge
====================

:func:`~spindoctor.cli.backplanes.merge.merge_sources_into_master` walks every pixel
exactly once. For each pixel it iterates the per-source distances and
picks the source with the smallest finite distance (closest along the
line of sight); the per-backplane values from that source are copied
into the master arrays. Pixels with no finite distance from any source
carry the masked value.

The masked value is ``backplanes.masked_value`` in
``config_900_backplanes.yaml``, ``-999.0`` as shipped. It sits outside the
range of every plane written, so one comparison -- ``!= masked_value`` --
identifies the measured pixels of any plane, and the PDS4 label declares it
as the array's missing constant. ``BODY_ID_MAP`` is the exception: it is the
body identity map rather than a measurement, and ``0`` is not a NAIF ID, so
zero there already means no body claimed the pixel.

The function also fills a sensor-shaped ``BODY_ID_MAP`` carrying the NAIF
ID of the winning source per pixel. Bodies use their real NAIF IDs;
rings use a deterministic ring-system ID (``cspyce.bodn2c('SATURN_RINGS')``
or equivalent). Simulated sources use their synthesised fake IDs so the
map is well-formed even on simulated images.

The merge is symmetric in bodies and rings: a body silhouette in front of
the ring plane wins because its distance is smaller; a ring inside a
body silhouette (geometrically rare but possible at certain phase angles
on edge-on rings) wins because *its* distance is smaller. The pipeline
does not enforce a body-then-rings precedence order.

FITS writer
===========

:func:`~spindoctor.cli.backplanes.writer.write_fits` serialises the master arrays. The
output FITS file structure:

- **Primary HDU** — empty, conventional placeholder.
- **BODY_ID_MAP HDU** — first ImageHDU, ``int32`` per-pixel NAIF IDs;
  emitted only when at least one pixel has a non-zero ID.
- **One HDU per backplane** — name from ``backplanes.bodies[i].name`` /
  ``backplanes.rings[i].name``, ``BUNIT`` header from the per-backplane
  ``units`` field, ``float32`` data. Backplanes that are entirely the
  masked value on this image are omitted (a body backplane on a
  no-body-in-FOV image contributes no HDU).

Alongside the FITS file the writer drops a companion
``<image>_backplane_metadata.json`` containing:

- the per-image dataset / instrument / observation metadata,
- the per-body inventory (NAIF ID, name, predicted bounding box),
- the per-backplane min / max / mean / valid-pixel-count statistics.

Each statistic states the unit its values are in, which for an angular plane
is not the unit of the array it was taken from.

The PDS4 bundle generator (:doc:`dev_guide_pds4`) reads this sidecar
when rendering the per-image data label.

Units: radians in the arrays, degrees in the statistics
=======================================================

An angular backplane array is written in radians and its ``BUNIT`` header
says so. The statistics taken from that array are written in degrees, and
each records the unit it is in beside its minimum and maximum. The two are
read by different readers and each is stated in the unit its own reader
wants: an array is consumed by software, which wants the unit its
trigonometry is already in and no conversion step it can get wrong, while
the statistics become the columns of a bundle's global index tables, which
a person reads to decide whether an image is worth opening — and a latitude
range of -88 to 81 tells them what -1.54 to 1.42 does not.

So one bundle carries ``rad`` on an array and ``deg`` on the table
summarising it, deliberately. Each label is correct about the file it
describes, which is the only thing a label is required to be correct about,
and the recorded unit means no reader has to work out which of the two
conventions a number in front of them follows.

:mod:`spindoctor.cli.backplanes.statistics` holds the whole rule and both
per-source stages reduce their planes through it. What is compared is the
unit's measure, everything up to the first solidus, and not the whole
string: ``rad`` and ``rad/pixel`` are both radians and both convert, to
``deg`` and ``deg/pixel``, the qualifier carried through untouched; ``km``
and ``km/pixel`` are not radians and are left alone. That is what the one
plane declared ``rad/pixel`` depends on, an equality against the whole
string recognising no compound unit and so putting radians per pixel into a
table every other angular column of which is degrees.

Radians is spelled ``rad`` and degrees ``deg``, which is what the
configuration writes and what the PDS4 units-of-angle vocabulary a label
draws from names. The comparison is exact, with no folding of case or
spacing, since the vocabulary has upper-case tokens of its own. A measure
spelled any other way — ``RAD``, or ``rad`` with a space beside it — is not
converted and is left as typed, and one that is angular all the same —
``mrad``, ``arcsec`` — needs scaling as well as renaming, so it is a change
to this module rather than a config entry it already handles. Two tests over
the shipping configuration stop such an entry reaching a table unnoticed: one
holds each measure, compared exactly, to the ones this module converts or
passes through, and the other holds each whole unit, measure and qualifier,
to :data:`~spindoctor.cli.pds4.collections.INDEX_VALUE_FORMATS`, the format
table the global index tables are written from. The bundle passes refuse a
unit that table has no format for before reading anything.

The viewer has a rule of its own. ``sd_backplane_viewer`` converts a plane
whose ``BUNIT`` is exactly ``rad`` and falls back to a heuristic on the HDU
name where there is no ``BUNIT`` at all, so it does not follow this one and
displays a plane declared ``rad/pixel`` in the unit its array carries.

Configuration
=============

The configurable backplane list lives in
``src/spindoctor/config_files/config_900_backplanes.yaml`` under the ``backplanes``
section (exposed as ``config.backplanes``). The full YAML schema and the
shipping defaults are documented at :doc:`/user_guide/user_guide_backplanes`; this
section covers the developer-facing parts of the contract.

- ``method`` resolves against :class:`oops.backplane.Backplane` by name at
  evaluation time. There is no compile-time check on the YAML; a typo
  surfaces only when the body or rings step calls the missing method on a
  real image. Confirm method names against the ``oops`` source before
  adding an entry.
- The body / rings dispatch is a hard split: ``backplanes.bodies`` entries
  are evaluated against the per-body meshgrid and merged with body
  ``distance``; ``backplanes.rings`` entries are evaluated against the
  full-frame ``snapshot.bp`` and merged with ring ``distance``. An entry
  in the wrong list fails at evaluation time.
- The :class:`~spindoctor.config.config.Config` loader's deep-merge rules apply
  to the ``backplanes`` block the same way they apply elsewhere; an
  override file overwrites the per-source list wholesale (lists are
  overwritten, not appended). See
  :doc:`dev_guide_config_and_static_data` for the loader contract.

Per-instrument overrides are not exposed — every instrument gets the same
backplane set. An instrument-specific backplane would require a per-
instrument ``backplanes`` block in ``config_4N0_inst_*.yaml`` plus a
config-merge hook in :func:`~spindoctor.cli.backplanes.backplanes.generate_backplanes_image_files`
to splice it onto the global list.

Snapshot helpers
================

The backplanes pipeline relies on four helpers added to
:class:`~spindoctor.obs.obs_snapshot.ObsSnapshot` (also consumed by the navigation
pipeline):

- :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_fov` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.inventory_body_in_extfov` —
  consume an ``oops`` inventory entry and return whether the predicted
  body bounding box overlaps the sensor / extended FOV. The body
  backplane step queries the inventory and uses these to decide which
  bodies contribute.
- :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_rect_fov` /
  :meth:`~spindoctor.obs.obs_snapshot.ObsSnapshot.clip_rect_extfov` — clamp a
  rectangle ``(u_min, u_max, v_min, v_max)`` into the sensor / extended
  FOV. The body backplane step uses ``clip_rect_fov`` on every body's
  inflated inventory bounding box before building the per-body meshgrid.

These helpers are documented in :doc:`dev_guide_observations`.

Adding a backplane
==================

1. Pick the source (body or rings) and the corresponding ``oops``
   :class:`~oops.backplane.Backplane` method to call. Verify the
   method exists by reading the ``oops`` source — there is no
   compile-time check on the YAML name.
2. Append a ``{name, method, units}`` entry to the matching list in
   ``config_900_backplanes.yaml``. Pick a ``name`` that scans well as
   a FITS HDU name (uppercase or snake_case, no spaces). ``units``
   describes the array, which for an angular plane means radians; the
   statistics are converted from it, per the units section above.
3. Rebuild a sample image with ``sd_backplanes`` and verify the new
   HDU appears in the FITS file with the expected ``BUNIT`` header and
   non-trivial pixel content (use ``sd_backplane_viewer`` for a quick
   visual check).
4. If the new backplane is consumed by the PDS4 bundle, extend the
   per-dataset ``data.lblx`` template to declare the corresponding
   ``Array_2D_Image`` element and update
   :meth:`~spindoctor.dataset.dataset.DataSet.pds4_template_variables` to
   surface the relevant per-HDU stats (min, max, units) into the
   template's variable dict. See :doc:`dev_guide_pds4`.

Testing
=======

A smoke test under ``experiments/backplanes/`` drives the end-to-end
pipeline against a simulated-image JSON, verifies the FITS file
structure, and asserts that simulated per-body backplanes respect each
body's mask (so a fake backplane does not paint a rectangular block
beyond the simulated body's edge).

Per-component unit tests under ``tests/backplanes/`` (where they exist)
cover the merge-step distance comparison, the per-body bounding-box
clipping, and the writer's ``BODY_ID_MAP`` emission rule.

API reference
=============

The :mod:`backplanes` package has no autogenerated entry under
:doc:`/api_reference`; the public surface is the functions listed
below:

- :func:`~spindoctor.cli.backplanes.backplanes.generate_backplanes_image_files` —
  per-image driver.
- :func:`~spindoctor.cli.backplanes.backplanes_bodies.create_body_backplanes` — body
  source.
- :func:`~spindoctor.cli.backplanes.backplanes_rings.create_ring_backplanes` — ring
  source.
- :func:`~spindoctor.cli.backplanes.statistics.plane_statistics` — per-plane
  reduction to a minimum, a maximum and the unit they are in, over
  :func:`~spindoctor.cli.backplanes.statistics.statistics_units`.
- :func:`~spindoctor.cli.backplanes.merge.merge_sources_into_master` — distance-aware
  merge.
- :func:`~spindoctor.cli.backplanes.writer.write_fits` — FITS + sidecar writer.
