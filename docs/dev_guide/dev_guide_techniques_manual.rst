==========================================================
Manual Navigation (NavTechniqueManual)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` is the interactive
navigation technique. It renders every
:class:`~spindoctor.nav_model.nav_model.NavModel`'s predicted scene into a single composite
overlay (templates, polylines, blob outlines, star markers), opens the
:class:`~spindoctor.ui.manual_nav_dialog.ManualNavDialog` PyQt6 widget, and packages the operator's chosen ``(dv, du)`` offset
into a :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

Manual navigation is not part of the autonomous pipeline — the class sets
``_abstract = True`` so it never appears in
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`'s discovery registry and the
orchestrator's background driver does not invoke the dialog during automated runs. An
interactive driver calls
:func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav` directly when an operator
wants to navigate an image by hand or override an autonomous result.

Feasibility passes when at least one offered feature is renderable into the dialog overlay
(a template, a non-empty polyline, a ``BODY_BLOB`` outline, or a ``STAR`` marker);
:meth:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual.is_feasible` returns
``feasible=False`` with reason ``no_renderable_features_for_manual_nav`` when nothing
renders. :func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav` additionally
verifies the composed overlay against the ext-FOV shape and skips (with a logged warning)
when the overlay mask is empty — every renderable feature clipped out of the frame — so an
empty dialog is never opened.

Theory
======

Manual navigation is not algorithmic — there is no cost function, no convergence criterion,
and no search. The technique renders the predicted scene, the operator inspects it
overlaid on the observed image, and the operator picks an offset that visually matches.

Composite overlay
-----------------

Every renderable feature kind in the scene contributes one element to the overlay:

- Template-bearing features (``BODY_DISC``, ``RING_ANNULUS``, ``CARTOGRAPHIC_MODEL``) paint their
  predicted brightness images into the composite.
- Polyline-bearing features (``LIMB_ARC``, ``TERMINATOR_ARC``, ``RING_EDGE``) draw their per-vertex
  polylines as one-pixel-wide strokes.
- ``BODY_BLOB`` features draw a circle outline at the predicted-vu position sized by the
  predicted disc diameter.
- ``STAR`` features draw a rectangle outline at the predicted-vu position sized by the
  per-feature point-spread-function bounding box.

The composite is a single extended-FOV image plus mask the dialog overlays on top of the
observed image. The operator drags or types the offset until the overlay aligns with the
observed scene.

Auto-pick option
----------------

The dialog includes an ``Auto`` button that runs the same masked-NCC pyramid the autonomous
correlation-based techniques use; the operator can accept the auto-pick verbatim or
override it. The auto-pick is a convenience — the technique itself remains operator-driven.

Restrictions and assumptions
----------------------------

- Manual navigation requires PyQt6 and an interactive display. Background driver runs that
  cannot open a window must not invoke this technique.
- The auto-pick path consults the same NCC pyramid the
  :class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav` and
  :class:`~spindoctor.nav_technique.nav_technique_ring_annulus.RingAnnulusNav` techniques use,
  correlating against the composite overlay image and mask. The ``Auto`` button is never
  disabled: it runs unconditionally on whatever the overlay contains (templates, polyline
  strokes, blob circles, star rectangles), so on scenes without a template-bearing feature
  the correlation runs against outline artwork and its peak may be uninformative — the
  operator judges the result and can always drag or type an offset instead.
- Operator precision in the dialog is limited by zoom, eye, and screen pixels; the
  technique reports a per-axis 1 px sigma on the resulting covariance regardless of how the
  operator picked.

Sources of uncertainty
----------------------

The reported covariance is fixed at ``1.0`` px sigma per axis on a diagonal 2x2. The
ensemble never combines a manual result with autonomous results, so the value only
determines what shows up in the per-image JSON metadata.

Configuration
=============

:class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` carries no per-technique
``tuning`` block in ``config_510_techniques.yaml`` and no
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_spec`. The technique opts
out of the autonomous registry (sets ``_abstract = True``) so the config-load validation
walk skips it. The single Python module-level constant is ``_MANUAL_OFFSET_SIGMA_PX = 1.0``,
the per-axis pixel sigma assigned to every manual pick.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_manual.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` and
  :func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav`.
- ``src/spindoctor/feature/composition.py`` —
  :func:`~spindoctor.feature.composition.compose_dialog_overlay`, the helper that builds the
  composite overlay image and mask from the scene's NavFeatures.
- ``src/spindoctor/ui/manual_nav_dialog.py`` —
  :class:`~spindoctor.ui.manual_nav_dialog.ManualNavDialog`, the PyQt6 widget the technique
  invokes.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` — the technique.
- :func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav` — the standalone
  interactive driver an external caller (a CLI script, an automated regression harness in
  pyqt-aware mode) invokes to open the dialog and obtain a result.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.name` — ``'NavTechniqueManual'``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset(NavFeatureType)`` — every feature type, since manual navigation overlays the
  whole scene.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- ``_abstract`` — ``True``. Keeps the technique out of
  ``NavTechnique._registry`` so the orchestrator's
  autonomous driver never invokes it.

The technique declares no
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_spec` and no
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` — manual results
come back with the operator-implied confidence ``1.0`` and the ensemble does not fuse them
with autonomous results.

Public methods (autodocumented):
:meth:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual.navigate`.

Call path
---------

Call path traced through
:func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav`:

1. The driver constructs every registered
   :class:`~spindoctor.nav_model.nav_model.NavModel`'s instance for the observation and calls
   :meth:`~spindoctor.nav_model.nav_model.NavModel.create_model` on each so the per-model state is
   populated.
2. Each model's :meth:`~spindoctor.nav_model.nav_model.NavModel.to_features` and
   :meth:`~spindoctor.nav_model.nav_model.NavModel.to_annotations` are invoked to gather the
   per-image features and the merged annotation collection.
3. :func:`~spindoctor.feature.composition.compose_dialog_overlay` builds the composite extfov
   image and mask the dialog overlays on top of the observed image.
4. :class:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual` is constructed with
   the merged annotations.
   :meth:`~spindoctor.nav_technique.nav_technique_manual.NavTechniqueManual.navigate` opens the
   :class:`~spindoctor.ui.manual_nav_dialog.ManualNavDialog`.
5. The dialog runs interactively until the operator confirms an offset. When the operator
   uses ``Auto``, the dialog runs the masked-NCC pyramid against the composite template;
   the result populates the dialog's offset entry and the operator may accept or override.
6. The operator's chosen ``(dv, du)`` plus the per-axis 1 px sigma populate a
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` with
   :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.confidence` ``1.0``,
   :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` ``False``,
   :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` ``False``, and a
   :class:`~spindoctor.nav_technique.diagnostics.ManualNavDiagnostics` with
   ``operator_accepted=True`` (the diagnostics type is always ``ManualNavDiagnostics``
   regardless of whether the auto-pick was used). Canceling the dialog instead yields a
   spurious zero-confidence result carrying ``ManualNavDiagnostics`` with
   ``operator_accepted=False``.
7. When the operator saves a sidecar, the merged annotations populate a labeled summary
   PNG alongside the JSON.

Examples
========

**Operator overrides an auto-pick.**  An operator opens the dialog on a Cassini fly-by
image where :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` reported a
``spurious=True`` result. The composite overlay shows the predicted limb and disc; the
operator clicks ``Auto`` and the masked-NCC pyramid lands on a peak the operator can see is
near-correct but a sub-pixel off. The operator drags the overlay one pixel and confirms
the offset. The technique returns a
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` with the operator's offset
and confidence ``1.0``; the summary PNG saves alongside the manual sidecar for reviewer
audit.

**Star-only scene with no auto-pick.**  An operator navigates a dense star field where the
autonomous :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`
reported ``status=failed``. The composite overlay shows the predicted catalog stars as
rectangle outlines. The operator manually drags the overlay until visible stars align with
the predicted boxes and confirms. The dialog's ``Auto`` button stays enabled — it would
correlate against the rectangle-outline artwork, which rarely yields a meaningful peak on a
star-only scene — so in practice the operator ignores it and uses the manual path.

**Headless backend rejection.**  A CI runner invokes
:func:`~spindoctor.nav_technique.nav_technique_manual.run_manual_nav` without a display server.
The PyQt6 import fails and the driver raises before the dialog opens. The autonomous
pipeline never invokes manual navigation, so this failure is confined to the interactive
driver path.
