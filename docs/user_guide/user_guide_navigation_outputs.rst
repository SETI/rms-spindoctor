==================
Inputs and Outputs
==================

Input Files
-----------

The primary input to SpinDoctor is spacecraft imagery. The system supports:

* PDS3 formatted image files (.IMG)
* Associated metadata (labels, SPICE kernels)

The system requires access to:

1. The raw image data
2. SPICE kernels for the appropriate mission and time period
3. Configuration settings (optional, defaults are provided)

Output Files
------------

SpinDoctor generates two types of output files:

Metadata Files (``*_metadata.json``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These JSON files contain the navigation results. The complete key-by-key
specification -- every key, its type, its presence rules, the rounding
policy, and one annotated example per document shape -- is the
:doc:`user_guide_metadata` chapter; in summary they include:

* ``observation`` — the image's identity: name, path, instrument, and
  ``camera`` (the camera that took it, e.g. ``NAC``). An image that fails to
  load has no observation to ask, so the navigator falls back to what the
  PDS3 index told it when the image was enumerated, recording ``camera``
  there; that needs no SPICE and never opens the image, so a frame whose
  navigation dies for want of a kernel is still attributed to its camera. An
  image navigated by explicit path rather than enumerated from an index has
  none. ``shutter_mode`` records the mode the image was taken in for an
  instrument whose label carries one; instruments whose labels carry no such
  field omit it. For every image whose
  navigation ran to a result, successful or failed, the block also records
  what is known about the exposure from the image itself: when the exposure
  began, its midpoint and when it ended (in UTC and ET, and as the spacecraft
  clock counts the label records), the exposure time, the filters, and
  whatever else the instrument states about the image.
  These are recorded whether or not a corrected pointing was; a load-error or
  internal-error document carries none of them.
* The calculated pointing offset (dv, du)
* Uncertainty estimates (sigma_v, sigma_u)
* Confidence scores
* Metadata about the navigation process
* Status information (success, error, etc.)
* Technique-specific metadata (one ``per_technique`` entry per technique run,
  with each technique's offset, covariance, confidence, spurious / at-edge
  flags, and diagnostics)
* ``excluded_from_consensus`` — technique names the ensemble left out of the
  reported combine (outliers rejected against a multi-technique consensus, or
  the runner-up alternative on a conflicted result)
* ``pointing`` — the image's attitude as a C-matrix: ``cmatrix_original``, the
  uncorrected J2000-to-camera rotation the furnished kernels gave, and
  ``cmatrix``, the same rotation corrected by the navigated offset, alongside
  the SPICE ``camera_frame``, ``camera_frame_id``, and the ``ck_frame_id`` of
  the object a corrected C-kernel targets. Both matrices are nine row-major
  floats at the exposure midtime. ``cmatrix`` is present only when the
  navigation produced an offset and fitted no camera rotation
* ``times`` — the exposure window the attitude belongs to: ``start_et``,
  ``stop_et``, ``midtime_et``, ``exposure_s``, and the spacecraft-clock
  strings ``sclk_start``, ``sclk_midtime`` and ``sclk_stop``
* Timestamps

.. note::

   The ``confidence`` values and ``confidence_rank`` tiers are
   calibrated against *simulated* planted-truth recovery only
   (sim-anchored): on real images they carry
   the simulator's realism as an unquantified assumption and must not
   be read as probabilities of real-image accuracy.  The
   ``confidence_provisional: true`` field in every ``_metadata.json``
   that carries a navigation result marks this sim-anchored basis
   (image-load-error metadata has no navigation result block and
   therefore no such field).  The tiers additionally price statistical
   error, not unmodeled systematic error: a coherent model error the
   diagnostics cannot see -- a ring feature whose true orbit sits a few
   pixels off the catalog orbit, or a high-phase haze crescent biasing
   a centroid -- can be absorbed into a confident, gate-passing wrong
   offset, so a high tier is not evidence against that kind of error
   (see the ensemble chapter's confident-wrong section in the developer
   guide).

These files are also the input to the run-statistics tooling
(``sd_results_index`` / ``sd_stats_report``), which aggregates them into
success/failure, technique-usage, offset, and cross-technique-agreement
reports; see :doc:`user_guide_statistics`.

Summary PNG Files (``*_summary.png``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A navigation that reached a result writes a ``*_summary.png`` beside its
metadata document: one annotated picture showing what the navigator saw and
where it placed its model. The source image is composited with the merged model
overlay at the fitted offset, so a glance tells you whether the predicted
features land on the real ones. An image whose data could not be loaded at all
-- a frame outside the SPICE kernels' coverage, most often -- writes the
metadata document, with a ``status`` of ``error``, and no picture: nothing was
read to draw one from.

The base layer is the source image rendered in grayscale with a quantile
contrast stretch. The black point sits at a low quantile; the white point adapts
to how many bright pixels the frame carries, so a sparse star field or a small
body against dark sky is not blown out by a handful of saturated pixels. The
model overlay is drawn on top, shifted by the navigated ``(dv, du)`` offset so
each prediction sits where the fit says the real feature is.

The overlay carries one set of annotations per contributing model:

* **Stars** -- each predicted catalog star is boxed and labeled with its name,
  magnitude, and (when known) spectral class. Every star box is additionally
  contrast-stretched against its own local minimum and maximum, so a faint star
  only a few DN above a bright background stays visible inside its box even where
  the whole-frame stretch would bury it.
* **Bodies** -- each body in the field of view contributes its lit-limb outline,
  with the body name labeled by an arrow pointing to the limb.
* **Rings** -- each catalog ring edge is drawn as a polyline following the edge
  across the frame and labeled with the edge name. Ring points hidden behind
  the planet globe are dropped, so an edge stops at the planet limb rather than
  being painted across the disc.

A metadata text block is placed in one corner. It gives the image name, filter,
and exposure, the navigation status (and, on success, the techniques that
contributed to the consensus offset), and the fused confidence value with its
tier. The corner is chosen to avoid overlapping the other annotation labels,
breaking ties toward the darkest corner; a long technique list wraps within the
block, and the block is omitted on a frame too small to hold it. The summary
carries no scale bar or coordinate grid -- it is a visual check of the fit, not
a measurement product; the numeric offset and geometry live in the metadata JSON
and the backplanes.

.. figure:: _images/summary_png_example.png
   :width: 80%
   :align: center

   Summary PNG for the real navigated Cassini ISS frame ``N1484688342``, showing
   every annotation family at once. A crescent Mimas carries its lit-limb outline
   and a ``MIMAS`` label; catalog ring edges (Encke and Keeler) are drawn as
   labeled polylines across the bright ring band; roughly a dozen predicted
   stars are boxed and labeled with catalog name, magnitude, and spectral class,
   each box locally contrast-stretched so the faint stars stay visible; and the
   lower-left metadata block reports a successful fit at confidence 0.660.

The overlay assembly is described in
:doc:`/dev_guide/dev_guide_annotations`, and the ring-edge planet-occlusion trim
in :doc:`/dev_guide/dev_guide_navigation_models_ring`.

Interpreting Results
--------------------

The key information in the results is:

1. **Offset Values**: The ``(dv, du)`` pixel offset -- v first, then u -- that should be applied to the nominal pointing to match the observed features
2. **Correlation Quality**: How well the models matched the observed features
3. **Annotations**: Identifications of specific features in the image
4. **Status**: Whether the navigation was successful, and if not, why


