===========
Annotations
===========

Overview
========

The :mod:`spindoctor.annotation` subsystem produces the per-image summary-PNG
overlay. Every :class:`~spindoctor.nav_model.nav_model.NavModel` exposes a
``to_annotations(context)`` method that returns a fresh
:class:`~spindoctor.annotation.annotations.Annotations` collection holding the
labels and graphical primitives for that one model's contribution
(body silhouettes, ring polylines, star markers, etc.). The
orchestrator merges every model's collection into
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.annotations` via
:meth:`~spindoctor.annotation.annotations.Annotations.add_annotations`; the
top-level driver (:func:`~spindoctor.navigate_image_files.navigate_image_files`)
then calls
:meth:`~spindoctor.annotation.annotations.Annotations.combine` to render the
final RGB overlay and composites it onto the contrast-stretched source
image before writing the PNG.

Class hierarchy
===============

The subsystem is intentionally narrow: four shipping types and no
abstract base.

- :class:`~spindoctor.annotation.annotation.Annotation` — a single typed
  primitive (line, polyline, marker, text). Each instance carries an
  RGBA overlay array, an overlay color, an optional avoid-mask used
  by the label placer, and a list of attached
  :class:`~spindoctor.annotation.annotation_text_info.AnnotationTextInfo`
  entries.
- :class:`~spindoctor.annotation.annotation_text_info.AnnotationTextInfo` —
  text payload + placement parameters (anchor location, arrow style,
  font color). The placement constants
  (:data:`~spindoctor.annotation.annotation_text_info.TEXTINFO_TOP`,
  ``TEXTINFO_BOTTOM_LEFT``, ``TEXTINFO_LEFT_ARROW``, etc.) name the
  twelve supported anchor / arrow combinations.
- :class:`~spindoctor.annotation.annotation_text_info.TextLocInfo` — the
  label placer's per-text resolution result (chosen pixel position,
  arrow tail, fitness score).
- :class:`~spindoctor.annotation.annotations.Annotations` — collection of
  :class:`~spindoctor.annotation.annotation.Annotation` instances. Subclass
  of :class:`~spindoctor.support.nav_base.NavBase` so it inherits ``config``
  and ``logger``.

Pipeline
========

Annotation flow during a single navigation run:

1. **Per-model emission.**  Each
   :class:`~spindoctor.nav_model.nav_model.NavModel` builds its
   :class:`~spindoctor.annotation.annotations.Annotations` collection in
   ``to_annotations(context)``. Concrete bodies / rings / stars use
   the shared helpers on
   :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase` and
   :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase` so
   the silhouette / polyline rendering is consistent across models.
2. **Orchestrator merge.**  The orchestrator's
   ``_collect_annotations`` step builds an empty
   :class:`~spindoctor.annotation.annotations.Annotations` and calls
   :meth:`~spindoctor.annotation.annotations.Annotations.add_annotations`
   for each surviving model's contribution. The merged collection is
   recorded on
   :attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.annotations`.
3. **Final composition.**
   :func:`~spindoctor.navigate_image_files.navigate_image_files` calls
   :meth:`~spindoctor.annotation.annotations.Annotations.combine` with the
   navigation offset. ``combine`` runs the label placer (consulting
   each :class:`~spindoctor.annotation.annotation.Annotation`'s avoid-mask),
   shifts every overlay by the offset, and produces a single RGB
   overlay array. The driver composites that overlay over the
   contrast-stretched source image and writes the PNG.

Reading the per-image PNG therefore tells the operator three things in
one image: what was in the source (background), what each
:class:`~spindoctor.nav_model.nav_model.NavModel`
predicted (overlay), and where the orchestrator placed the predictions
relative to the data (the offset shift applied at composition time).

Summary-PNG extras
==================

Beyond the raw overlay the summary-PNG renderer
(:func:`~spindoctor.support.summary_png.render_annotated_summary_rgb`) adds two
presentation features. A metadata text block gives the image name and, when they
are known, its filter and exposure; it reports the navigation status, the
contributing techniques (only for a successful navigation), and the fused
confidence (shown as ``n/a`` when none was fused). It is drawn in the
least-crowded corner and steered clear of the drawn label bounding boxes
(falling back to image brightness among the text-free corners). A per-star local
contrast stretch rewrites each star detection box against its own min / max so a
faint star a few DN above a bright background stays visible even when the
whole-frame stretch would bury it. Ring edges hidden behind the planet globe are
already absent from the masks the overlay draws, so they are never painted
across the disc (see :doc:`dev_guide_navigation_models_ring`).

Configuration
=============

The :mod:`spindoctor.annotation` subsystem itself consumes no YAML
configuration. The label / color knobs live in the per-:class:`~spindoctor.nav_model.nav_model.NavModel`
config blocks instead:

- ``bodies.label_*``, ``bodies.outline_thicken``, ``bodies.min_text_area``
  in ``src/spindoctor/config_files/config_040_bodies.yaml`` (consumed via
  :class:`~spindoctor.nav_model.nav_model_body_base.NavModelBodyBase`).
- ``rings.label_*`` in ``src/spindoctor/config_files/config_050_rings.yaml``
  (consumed via
  :class:`~spindoctor.nav_model.nav_model_rings_base.NavModelRingsBase`).
- ``stars.label_*`` and ``stars.label_star_color`` in
  ``src/spindoctor/config_files/config_030_stars.yaml`` (consumed via
  :class:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars`).

Each per-:class:`~spindoctor.nav_model.nav_model.NavModel` page documents the relevant subset:
:doc:`dev_guide_navigation_models_body`,
:doc:`dev_guide_navigation_models_ring`, and
:doc:`dev_guide_navigation_models_star`.

Per-model annotation contributions
==================================

- **Body** —
  :meth:`~spindoctor.nav_model.nav_model_body.NavModelBody.to_annotations`
  emits a body silhouette outline plus a body-name label with leader
  arrow. The label placer scans the limb for a clear gap and falls
  back to a coarse grid when no per-limb candidate fits.
- **Ring** —
  :meth:`~spindoctor.nav_model.nav_model_rings.NavModelRings.to_annotations`
  emits one polyline per surviving ring edge plus a per-planet label.
- **Star** —
  :meth:`~spindoctor.nav_model.stars.nav_model_stars.NavModelStars.to_annotations`
  emits a per-star marker plus an optional per-star magnitude label.
- **Simulated body / rings** — the simulated NavModels reuse their
  catalog-driven counterparts' annotation helpers, so a simulated
  scene's PNG overlay is visually indistinguishable from a real scene's.
- **Titan** —
  :meth:`~spindoctor.nav_model.nav_model_titan.NavModelTitan.to_annotations`
  emits the predicted haze envelope circle, the symmetry axis, the sunward
  arc sector, and a center cross, styled by the feature's reliability
  against the per-type gate threshold: solid curves at or above it, dotted
  curves plus a low-reliability label below it.

API reference
=============

The autodocumented API surface is at :doc:`/api_reference/api_annotation`.
