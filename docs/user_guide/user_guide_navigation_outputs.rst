=============================
Navigation Inputs and Outputs
=============================

Input Files
===========

A navigation run needs the image file and its label, SPICE kernels covering the time the
image was taken, and optionally a configuration of your own; defaults are supplied for
every instrument. The images are VICAR files for Cassini ISS, Voyager ISS, and Galileo
SSI, and FITS files for New Horizons LORRI. How images are enumerated is described in
:doc:`user_guide_image_selection`, the configuration system in
:doc:`user_guide_configuration`, and each mission's own holdings, labels, and metadata in
its chapter under :doc:`/user_guide/instruments/instruments`.

Output Files
============

``sd_offset`` (see :doc:`user_guide_navigation_running`) writes up to two files per
image under the navigation results root, both named after the image: a
metadata document named ``*_metadata.json``, and an annotated picture named
``*_summary.png``. The metadata document is written for every image the run reports
on. The picture is
written only for an image the navigator worked through; an image whose file could not
be read, or that failed before navigation, gets the metadata document alone.

The metadata document
---------------------

The metadata document is a JSON file holding the navigation results for one image. The
complete key-by-key specification -- every key, its type, when it is present, the
rounding policy, and one annotated example per kind of result -- is
:doc:`user_guide_metadata`. It is organized as these blocks:

* ``observation`` -- what image this is, and what its instrument states about the
  exposure.
* ``pointing`` -- the camera's attitude as a C-matrix, both as the furnished kernels gave
  it and as corrected by the navigated offset.
* ``offset``, ``sigma_px``, and ``covariance_px2`` -- the measured pointing correction
  ``[dv, du]`` in pixels and how precisely it is known.
* ``confidence`` and ``confidence_rank`` -- how much the pipeline trusts the answer, as a
  number in ``[0, 1]`` and as a coarse tier.
* ``status`` and ``status_reason`` -- whether the navigation succeeded, failed, or came
  out conflicted, and the discrete reason for that outcome.
* ``per_technique`` and ``excluded_from_consensus`` -- what each technique answered on its
  own, and which answers were left out of the reported combination.
* ``times`` -- the exposure window the attitude belongs to.
* ``provenance`` -- what the run was made of: the software version, the SPICE kernels
  loaded, the star catalogs used, and the configuration in force.
* ``timing`` -- when the navigation of this image started and ended, how long it took,
  and the peak memory the process reached.

The value a downstream consumer should use is the corrected pointing in
``pointing.cmatrix``. The ``offset`` is a correction **relative to the SPICE kernels that
were furnished when the image was navigated**, and those kernels are listed by name in the
same metadata document, under ``provenance.spice_kernels``. Applied against a different
set of kernels the offset means nothing. The corrected pointing already carries the
correction and is tied to no particular set of kernels.

.. warning::

   The ``confidence`` values and ``confidence_rank`` tiers are calibrated against
   *simulated* planted-truth recovery only. On real images they carry the simulator's
   realism as an unquantified assumption and must not be read as probabilities of
   real-image accuracy. The ``confidence_provisional: true`` field in every metadata
   document that carries a navigation result marks that basis.

   The tiers cover statistical error only. Where a technique's answer is displaced by an
   error coherent across the whole measurement, its own diagnostics cannot see the
   displacement, so it reports a tight uncertainty and a high confidence around the wrong
   offset. Two real cases: a ring feature whose true orbit sits a few pixels off the
   catalog orbit the model was built from, so every vertex of every edge is displaced
   together and the fit is internally consistent at the wrong place; and a high-phase
   Titan crescent, where the visible haze is a thin arc on one side of the body, which
   pulls a centroid toward the lit side by an amount no residual reveals.

   The tier boundaries take that into account as far as they can. The ``high`` boundary
   sits at a confidence of 0.85 because the calibration runs put most of their
   tight-uncertainty wrong answers between 0.55 and 0.80, so those land in ``medium``
   instead. What a ``high`` tier does not do is rule such an answer out. The thing that
   catches one is corroboration from an independent technique working on different
   content -- a star field beside the rings, a second resolved moon beside Titan -- which
   is why the pipeline runs every technique that has something to work with rather than
   stopping at the first answer. That is the default. ``--nav-techniques`` narrows the
   candidates to the techniques you name, and only those then run.

These metadata documents are also what the run-statistics tooling reads.
``sd_results_index`` (see :doc:`user_guide_results_index`) collects one row per
navigated image from them. ``sd_stats_report`` (see :doc:`user_guide_statistics`)
aggregates success and failure counts, technique usage, offset distributions, and
cross-technique agreement reports. It reads the metadata documents when the run names no
results index, and queries the results index when the run names one.

The summary picture
-------------------

A navigation that reached a result writes a ``*_summary.png`` beside its metadata
document: one annotated picture showing what the navigator saw and where it placed its
model. The source image is composited with the merged model overlay at the fitted
offset, so a glance tells you whether the predicted features land on the real ones. A
feature is one star, one limb, one ring edge, or any other single thing a model expected
to find. :doc:`user_guide_navigation_models` describes every feature type. An
image whose data could not be loaded at all -- a frame outside the SPICE kernels'
coverage, most often -- gets the metadata document, with a ``status`` of ``error``, and
no picture: nothing was read to draw one from.

Look at this picture first whenever a result surprises you. It shows in one glance what no
single number in the metadata document can: whether the navigator was matching the thing
you assumed it was matching. A confident offset whose drawn limb sits on a crater rim, a
ring polyline one ringlet away from the bright edge beneath it, or a set of star boxes
sitting on nothing are all immediately visible, and each points at a different cause.

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
