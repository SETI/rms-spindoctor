==========================================================
Per-Image State (NavContext)
==========================================================

Overview
========

:class:`~spindoctor.nav_orchestrator.nav_context.NavContext` is the frozen dataclass that carries
per-image global state shared across feature extraction and technique navigation. The
orchestrator's
:meth:`~spindoctor.nav_orchestrator.orchestrator.NavOrchestrator.navigate` builds one instance per
image; pass-2 techniques receive a copy with the pass-1 ensemble's prior offset and
covariance attached via
:meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior`. Every member is
computed without knowing where any feature lives in the image: global statistics,
sensor-vs-extfov masks, shared image-side derivatives, and provenance.

Theory
======

The context is the single source of truth for per-image state. Three invariants apply:

- The context is **frozen**. ``with_prior`` returns a new instance via
  :func:`dataclasses.replace`; mutation is not supported.
- Every image-shaped field is **extfov-shaped** — zero-padded around the original sensor
  rectangle. ``image_ext`` is the canonical post-source-filter extfov image; the matching
  sensor / saturation / cosmic-ray masks share its shape.
- The pass-1 → pass-2 hand-off carries only the **2x2 translation block** of the prior
  covariance. Any rotation prior is re-derived per technique from the per-instrument
  flag, not propagated through the context.

The default values on the dataclass cover the autonomous-pipeline path; the manual-nav
dialog and tests construct contexts with relaxed defaults (e.g. no provenance, no
classifier).

Restrictions and assumptions
----------------------------

- Construction is via direct dataclass instantiation. Validation happens in
  :meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior` (the only mutator
  surface): the prior offset must be finite, the prior covariance must be 2x2 or 3x3 and
  finite.
- The frozen-array discipline is enforced via ``__post_init__`` on the prior covariance
  copy: ``with_prior`` makes a writable independent copy and marks it read-only so the
  caller cannot mutate the prior after the new context is built.
- The shared image-side derivatives (``image_gradient_ext``, ``image_gradient_vu_ext``,
  ``image_edge_dt_ext``) are optional in the dataclass signature so test contexts that
  do not need them can omit them; the orchestrator always populates all three for
  autonomous runs.

Sources of uncertainty
----------------------

The context dataclass itself reports no uncertainty. Per-image quantities surfaced in
the context (the noise sigma, the classifier's confidence) are reported by the
sub-systems that produced them.

Configuration
=============

The dataclass carries no YAML configuration of its own. Field defaults are Python
constants with the documented per-instrument-aware values populated by the orchestrator's
``_make_context``.

Implementation
==============

Source file: ``src/spindoctor/nav_orchestrator/nav_context.py`` —
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext`.

Public class :class:`~spindoctor.nav_orchestrator.nav_context.NavContext`, frozen dataclass.

Public fields (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.obs` — the observation snapshot
  under navigation. Typed loosely as :class:`object` to avoid an import cycle.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` — extended-FOV image array
  post any source-image filter.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.sensor_mask_ext` — boolean mask;
  ``True`` where the pixel is real sensor data, ``False`` for extfov padding.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_noise_sigma` — robust MAD-based
  noise sigma in the image's native units (DN for ``raw_dn``, I/F for ``calibrated_if``).
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.saturation_mask_ext` — boolean mask
  ``True`` where pixels are at or above the per-instrument full-well DN.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.cosmic_ray_mask_ext` — boolean mask
  ``True`` where single-pixel cosmic-ray spikes were detected.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_classifier` — the
  :class:`~spindoctor.nav_orchestrator.image_classifier_result.NavImageClassifierResult` verdict.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.provenance` — per-image
  reproducibility envelope (:class:`~spindoctor.nav_orchestrator.provenance.Provenance`).
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_ext` — Optional shared
  Sobel-of-Gaussian magnitude (computed once, reused by every DT technique).
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` — Optional
  ``(H, W, 2)`` per-pixel gradient-vector image; ``[..., 0]`` is ``g_v``, ``[..., 1]`` is
  ``g_u``.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` — Optional shared
  Euclidean distance transform of the thresholded gradient image. It is not signed: the values are
  non-negative everywhere and the zero locus is the edge pixels themselves, at their own centers,
  not an oriented boundary running along pixel edges half a pixel away.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.prior_offset_px` — pass-2 prior
  offset; ``None`` on pass 1.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.prior_covariance_px2` — pass-2 prior
  covariance; ``None`` on pass 1.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.pre_filter_applied` — the
  :class:`~spindoctor.support.filters.NavFilterSpec` applied to the source image, ``None`` if
  none.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` — bool; turns
  on 3-DoF technique fits. Default ``False``.
- :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` — float; rotation
  cap when ``fit_camera_rotation`` is ``True``. Default ``5.0`` deg.

Public methods:

- :meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior` — return a new
  ``NavContext`` with the pass-1 prior attached. Validates the offset (finite, length-2)
  and the covariance (2x2 or 3x3, finite); the rotation block is dropped.

The dataclass declares ``eq=False`` on ``@dataclass(frozen=True, eq=False)``;
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext` instances are compared by identity
because the numpy-array fields prevent the default dataclass equality.

Examples
========

**Pass 1 context.**  The orchestrator's ``_make_context`` builds a
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext` with the per-instrument settings
populated and the prior fields ``None``::

    context = NavContext(
        obs=obs,
        image_ext=image,
        sensor_mask_ext=sensor_mask,
        image_noise_sigma=4.7,
        saturation_mask_ext=sat_mask,
        cosmic_ray_mask_ext=cr_mask,
        image_classifier=classifier_result,
        provenance=prov,
        image_gradient_ext=grad,
        image_gradient_vu_ext=grad_vu,
        image_edge_dt_ext=edge_dt,
        pre_filter_applied=None,
        fit_camera_rotation=False,
        max_rotation_deg=5.0,
    )

**Pass 2 hand-off.**  After the pass-1 ensemble produces an offset, the orchestrator
calls
:meth:`~spindoctor.nav_orchestrator.nav_context.NavContext.with_prior` with the pass-1 offset
and covariance. The returned context carries the prior on
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.prior_offset_px` and
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.prior_covariance_px2`; pass-2
techniques like
:class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` consume those.

**Field-by-field illustration on ``body_partial_overflow``.**  After the orchestrator's
``_make_context`` builds the per-image state for the Rhea-overflow scene, the context
fields might be::

    NavContext(
        image_classifier.image_class = 'clean',
        image_noise_sigma = 4.7,                  # DN
        saturation_mask_ext.sum() = 0,             # no saturated pixels
        cosmic_ray_mask_ext.sum() = 12,            # 12 cosmic-ray hits
        fit_camera_rotation = False,               # Cassini default
        prior_offset_px = None,                    # pass 1
    )

The pass-2 hand-off attaches
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.prior_offset_px`
``= (12.06, 30.53)`` once :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`
reports its result.
