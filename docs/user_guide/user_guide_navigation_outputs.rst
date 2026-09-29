=============================
Navigation Inputs and Outputs
=============================

Input Files
===========

Navigation reads one image at a time, together with the label that describes it and
the SPICE kernels that cover the time the image was taken.

The image files are in one of two formats:

* VICAR, for Cassini ISS, Voyager ISS, and Galileo SSI (``.IMG`` files).
* FITS, for New Horizons LORRI (``.fit`` files).

PDS3 is not an image format. It is the way the archive organizes those files into
volumes, with a detached label beside each image and an index table listing what the
volume holds. The label supplies the exposure times, the filters, and the rest of what
the mission recorded about the frame; the PDS3 index table is one of the ways a run can
enumerate images to process, as described in :doc:`user_guide_image_selection`.

A navigation run therefore needs three things:

1. The image file and its label.
2. SPICE kernels for the mission and the time period the image was taken in.
3. Configuration, which is optional: defaults are supplied for every instrument. See
   :doc:`user_guide_configuration`.

Output Files
============

``sd_offset`` (see :doc:`user_guide_navigation_running`) writes two files per image
under the navigation results root, both named after the image: a metadata document
named ``*_metadata.json``, and an annotated picture named ``*_summary.png``.

The metadata document
---------------------

The metadata document is a JSON file holding the navigation results for one image. The
complete key-by-key specification -- every key, its type, when it is present, the
rounding policy, and one annotated example per kind of result -- is
:doc:`user_guide_metadata`. In summary, it holds:

* ``observation`` -- the image's identity: name, path, instrument, and ``camera`` (the
  camera that took it, for example ``NAC``). An image that fails to load has no
  observation to ask, so the value recorded under ``camera`` falls back to what the
  PDS3 index table said when the image was enumerated; that needs no SPICE and never
  opens the image, so a frame whose navigation dies for want of a kernel is still
  attributed to its camera. An image navigated by explicit path, rather than enumerated
  from the PDS3 index table, has no such fallback. ``shutter_mode`` records the mode
  the image was taken in for an instrument whose label carries one; instruments whose
  labels carry no such field omit it. For every image whose navigation ran to a result,
  successful or failed, the block also records what is known about the exposure from
  the image itself: when the exposure began, its midpoint, and when it ended (in UTC and
  ET, and as the spacecraft clock counts the label records), the exposure time, the
  filters, and whatever else the instrument states about the image. These are recorded
  whether or not a corrected pointing was. A metadata document written for an image
  that could not be loaded, or whose navigation hit an internal fault, carries none of
  them.
* ``pointing`` -- the image's attitude as a C-matrix. ``cmatrix_original`` is the
  uncorrected J2000-to-camera rotation the furnished kernels gave, and ``cmatrix`` is
  the same rotation corrected by the navigated offset, alongside the SPICE
  ``camera_frame``, ``camera_frame_id``, and the ``ck_frame_id`` of the object a
  corrected C-kernel targets. Both matrices are nine row-major floats at the exposure
  midtime. ``cmatrix`` is present only when the navigation produced an offset and
  fitted no camera rotation.
* ``offset`` -- the measured pointing correction ``[dv, du]`` in pixels, v first, then
  u. It is a correction **relative to the SPICE kernels that were furnished when the
  image was navigated**, and those kernels are listed by name in the metadata document
  itself, under ``provenance.spice_kernels``. Applied against a different set of
  kernels the offset means nothing, so the value a consumer should use is the corrected
  pointing in ``cmatrix``, which already carries the correction and is tied to no
  particular set of kernels.
* ``sigma_px`` -- the per-axis 1-sigma uncertainty of the offset, in pixels, and
  ``covariance_px2``, the full covariance it comes from.
* ``confidence`` and ``confidence_rank`` -- how much the pipeline trusts the answer, as
  a number in ``[0, 1]`` and as a coarse tier.
* ``status`` and ``status_reason`` -- whether the navigation succeeded, failed, or came
  out conflicted, and the discrete reason for that outcome.
* ``per_technique`` -- one entry for each technique that produced an answer, with that
  technique's own offset, covariance, confidence, its self-flags for a spurious result
  and for a solution that touched its search boundary, and its diagnostics.
* ``excluded_from_consensus`` -- the techniques whose answers were left out of the
  reported combination: outliers rejected against a consensus of several techniques, or
  the runner-up answer on a conflicted result.
* ``times`` -- the exposure window the attitude belongs to: ``start_et``, ``stop_et``,
  ``midtime_et``, ``exposure_s``, and the spacecraft-clock strings ``sclk_start``,
  ``sclk_midtime``, and ``sclk_stop``.
* ``provenance`` -- what the run was made of: the software version, the SPICE kernels
  loaded, the star catalogs used, and the configuration in force.
* ``timing`` -- when the navigation of this image started and ended, how long it took,
  and the peak memory the process reached.

.. note::

   The ``confidence`` values and ``confidence_rank`` tiers are calibrated against
   *simulated* planted-truth recovery only. On real images they carry the simulator's
   realism as an unquantified assumption and must not be read as probabilities of
   real-image accuracy. The ``confidence_provisional: true`` field in every metadata
   document that carries a navigation result marks that basis; a metadata document for
   an image that could not be loaded has no navigation result and therefore no such
   field. The
   tiers price statistical error only. A coherent model error the diagnostics cannot
   see -- a ring feature whose true orbit sits a few pixels off the catalog orbit, or a
   high-phase haze crescent biasing a centroid -- can be absorbed into a wrong offset
   that still reports high confidence and a tight uncertainty, so a high tier is not
   evidence against that kind of error. See the confident-wrong discussion in
   :doc:`/dev_guide/dev_guide_orchestrator_ensemble`.

These metadata documents are also what the run-statistics tooling reads.
``sd_results_index`` (see :doc:`user_guide_results_index`) collects one row per
navigated image from them, and ``sd_stats_report`` (see
:doc:`user_guide_statistics`) aggregates them into success and failure counts,
technique usage, offset distributions, and cross-technique agreement reports.

The summary picture
-------------------

A navigation that reached a result writes a ``*_summary.png`` beside its metadata
document: one annotated picture showing what the navigator saw and where it placed its
model. The source image is composited with the merged model overlay at the fitted
offset, so a glance tells you whether the predicted features land on the real ones. An
image whose data could not be loaded at all -- a frame outside the SPICE kernels'
coverage, most often -- gets the metadata document, with a ``status`` of ``error``, and
no picture: nothing was read to draw one from.

The base layer is the source image rendered in grayscale with a quantile contrast
stretch. The black point sits at a low quantile; the white point adapts to how many
bright pixels the frame carries, so a sparse star field or a small body against dark sky
is not blown out by a handful of saturated pixels. The model overlay is drawn on top,
shifted by the navigated ``(dv, du)`` offset so each prediction sits where the fit says
the real feature is.

The overlay carries one set of annotations per contributing model:

* **Stars** -- each predicted catalog star is boxed and labeled with its name,
  magnitude, and (when known) spectral class. Every star box is additionally
  contrast-stretched against its own local minimum and maximum, so a faint star only a
  few DN above a bright background stays visible inside its box even where the
  whole-frame stretch would bury it.
* **Bodies** -- each body in the field of view contributes its lit-limb outline, with
  the body name labeled by an arrow pointing to the limb.
* **Rings** -- each catalog ring edge is drawn as a polyline following the edge across
  the frame and labeled with the edge name. Ring points hidden behind the planet globe
  are dropped, so an edge stops at the planet limb rather than being painted across the
  disc.

A metadata text block is placed in one corner. It gives the image name, filter, and
exposure, the navigation status (and, on success, the techniques that contributed to the
reported offset), and the fused confidence value with its tier. The corner is chosen to
avoid overlapping the other annotation labels, breaking ties toward the darkest corner;
a long technique list wraps within the block, and the block is omitted on a frame too
small to hold it. The picture carries no scale bar or coordinate grid. It is a visual
check of the fit; the numeric offset and the geometry live in the metadata document and
in the backplanes.

.. figure:: _images/summary_png_example.png
   :width: 80%
   :align: center

   Summary picture for the real navigated Cassini ISS frame ``N1484688342``, showing
   every annotation family at once. A crescent Mimas carries its lit-limb outline and a
   ``MIMAS`` label; catalog ring edges (Encke and Keeler) are drawn as labeled
   polylines across the bright ring band; roughly a dozen predicted stars are boxed and
   labeled with catalog name, magnitude, and spectral class, each box locally
   contrast-stretched so the faint stars stay visible; and the lower-left metadata block
   reports a successful fit at confidence 0.660.

The overlay assembly is described in :doc:`/dev_guide/dev_guide_annotations`, and the
ring-edge planet-occlusion trim in :doc:`/dev_guide/dev_guide_navigation_models_ring`.

Interpreting Results
====================

Four things in a metadata document answer most questions about an image:

1. **The corrected pointing.** ``pointing.cmatrix`` is the attitude the camera actually
   had, according to the navigation, and it is what downstream work should use: the
   backplane generator, the reprojection and mosaic tools, and the C-kernel writer all
   read it. Use the ``offset`` only when you need the size of the correction itself,
   and remember that it is measured against the kernels named in the document's
   ``provenance`` block.
2. **The uncertainty.** ``sigma_px`` says how precisely the offset is known, in pixels,
   on each axis.
3. **The confidence.** ``confidence`` and ``confidence_rank`` say how much the pipeline
   trusts the fit, subject to the caution above.
4. **The status.** ``status`` and ``status_reason`` say whether the navigation
   succeeded, and if it did not, why.

The summary picture answers the same question visually: if the drawn limbs, ring edges,
and star boxes sit on the real features, the fit is good.
