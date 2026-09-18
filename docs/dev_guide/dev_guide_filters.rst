==========================================
Filters (NavFilterSpec and apply_filter)
==========================================

Overview
========

:mod:`spindoctor.support.filters` is the one place image-domain filtering is
defined for the navigation pipeline. A filter is described by a
:class:`~spindoctor.support.filters.NavFilterSpec` -- a frozen dataclass
naming a :class:`~spindoctor.support.filters.NavFilterKind` plus the
parameters that kind consumes -- and executed by the single dispatcher
:func:`~spindoctor.support.filters.apply_filter`. Higher-level code never
switches on the kind itself; it builds a spec and calls the dispatcher, so
every consumer applies exactly the same operation for the same description.

The point of describing a filter as data rather than as a function call is
uniformity: the same spec can be applied to an image patch and to a model
template, guaranteeing both sides of a matching metric were filtered
identically, and the spec that ran can be recorded on the per-image context
for diagnostics.

(The similarly named helpers in :mod:`spindoctor.support.image` --
:func:`~spindoctor.support.image.filter_local_maximum`,
:func:`~spindoctor.support.image.filter_sub_median`,
:func:`~spindoctor.support.image.filter_downsample` -- are standalone array
utilities, not part of this spec-dispatch system.)

Theory
======

Filter kinds
------------

:class:`~spindoctor.support.filters.NavFilterKind` is a closed enumeration;
:func:`~spindoctor.support.filters.apply_filter` raises ``ValueError`` on a
kind it does not dispatch, so adding a kind means extending the dispatcher in
the same change.

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Kind
     - Operation
     - Parameters consumed
   * - ``NONE``
     - Identity; the input is returned unchanged.
     - none
   * - ``ISOTROPIC_GAUSSIAN``
     - Symmetric Gaussian blur.
     - ``sigma_xy``
   * - ``ANISOTROPIC_GAUSSIAN``
     - Gaussian blur from a full 2x2 covariance. With ``align_axis`` the
       array is rotated into the principal frame, blurred axis-aligned,
       and rotated back (acceptable for the postage-stamp inputs
       techniques pass in).
     - ``covariance_px2``, ``align_axis``
   * - ``BANDPASS_DOG``
     - Difference of Gaussians: subtract a heavy blur from a light blur,
       suppressing low-frequency content (stray-light gradients) while
       preserving detail sharper than the heavy-blur scale. Requires
       ``lo > hi > 0``.
     - ``bandpass_cutoffs_px`` as ``(lo_sigma, hi_sigma)``
   * - ``GRADIENT_OF_GAUSSIAN``
     - Gaussian smooth followed by Sobel gradient magnitude.
     - ``sigma_xy``
   * - ``MORPH_DILATE``
     - Gray dilation by a rectangular structuring element whose per-axis
       half-width is that axis's sigma rounded up; used when building
       search margins for edge-based matching.
     - ``sigma_xy``
   * - ``DISTANCE_TRANSFORM``
     - Euclidean distance transform of a binary edge map, clipped to
       ``dt_half_width_px``; an input with no edge pixels yields a
       uniformly saturated array so consumers always see a fully-defined
       result. Only meaningful on a binary input.
     - ``dt_half_width_px``

Short-circuits
--------------

Two universal short-circuits run before kind dispatch:

1. A spec of kind ``NONE`` returns the input unchanged.
2. A **blur-family** spec (``ISOTROPIC_GAUSSIAN``, ``ANISOTROPIC_GAUSSIAN``,
   ``BANDPASS_DOG``) whose largest principal sigma is below the spec's
   ``null_filter_threshold_sigma`` (default ``0.4`` pixels) is treated as
   identity: so small a blur makes no meaningful difference.

The null-sigma short-circuit deliberately does **not** apply to
``GRADIENT_OF_GAUSSIAN`` (whose output is a gradient magnitude, not the
intensity image), ``MORPH_DILATE`` (guarded by its own zero-half-width
check), or ``DISTANCE_TRANSFORM`` (which has no sigma concept); returning
the raw intensity image for those would silently change the meaning of the
result.

Mismatched kind/parameter combinations raise at application time, not at
construction, so a technique can carry an under-populated spec through an
identity short-circuit without ceremony.

Where filters run in the pipeline
=================================

Three call sites use the system:

**Source-image pre-filter (orchestrator).**
    Before feature extraction, the orchestrator reads the per-instrument
    ``source_image_filter`` configuration block and, when it is enabled,
    applies the described filter to the whole source image. The spec that
    ran is recorded on
    :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.pre_filter_applied`
    (``None`` when nothing ran). Every shipped instrument configuration
    declares a ``BANDPASS_DOG`` pre-filter with ``lo_sigma_px: 5.0`` and
    ``hi_sigma_px: 0.7`` and ships it ``enabled: false``, so the pre-filter
    is an opt-in tool for scattered-light frames rather than a default
    stage. An unknown ``kind`` in the block logs a warning and skips the
    pre-filter rather than failing the image.

**Edge distance transform (image derivatives).**
    :mod:`spindoctor.nav_orchestrator.image_derivatives` builds the shared
    per-image edge products once per navigation: it thresholds the
    Sobel-of-Gaussian gradient, thins it by directional non-maximum
    suppression, and turns the resulting edge mask into a truncated distance
    transform via a ``DISTANCE_TRANSFORM`` spec. The result lands on
    :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext`
    and is consumed by every distance-transform technique (see
    :doc:`dev_guide_techniques_image_derivatives`).

**Per-feature preferred filter (models).**
    Every :class:`~spindoctor.feature.feature.NavFeature` carries a
    ``preferred_filter`` spec, the filter the emitting model requests for
    both the image patch and the model template before matching. Every
    shipped extractor emits ``NavFilterKind.NONE`` here, so the field is a
    declared extension point: a model whose feature benefits from, say, a
    bandpass before correlation can request one without any consumer-side
    change.

Configuration
=============

The pre-filter block lives in each instrument's configuration
(``config_4N0_inst_*.yaml``), per camera where the instrument has more than
one::

    source_image_filter:
      kind: BANDPASS_DOG    # a NavFilterKind name
      lo_sigma_px: 5.0      # BANDPASS_DOG only
      hi_sigma_px: 0.7      # BANDPASS_DOG only
      enabled: false

For any kind other than ``BANDPASS_DOG`` the block instead reads a single
``sigma_px`` (default ``1.0``) applied to both axes. The identity threshold
``null_filter_threshold_sigma`` is a field on the spec itself (default
``0.4``), not a configuration key.

Restrictions and assumptions
============================

- All functions in the module are pure and stateless; they are safe for
  concurrent use on independent inputs.
- Inputs must be 2-D float arrays; :func:`~spindoctor.support.filters.apply_filter`
  raises ``TypeError`` otherwise, and ``ValueError`` when a spec is missing
  parameters its kind requires.
- The rotate-blur-rotate implementation of an axis-aligned
  ``ANISOTROPIC_GAUSSIAN`` with ``align_axis`` interpolates twice; it is
  intended for small postage stamps, not full frames.

Implementation
==============

Source file: ``src/spindoctor/support/filters.py`` --
:class:`~spindoctor.support.filters.NavFilterKind`,
:class:`~spindoctor.support.filters.NavFilterSpec`, and
:func:`~spindoctor.support.filters.apply_filter`, plus one private helper per
non-trivial kind. The public surface is autodocumented at
:doc:`/api_reference/api_support`.

Examples
========

**Bandpass before correlation.** Suppress a stray-light ramp while keeping
star-scale detail:

.. code-block:: python

    from spindoctor.support.filters import NavFilterKind, NavFilterSpec, apply_filter

    spec = NavFilterSpec(
        kind=NavFilterKind.BANDPASS_DOG,
        bandpass_cutoffs_px=(5.0, 0.7),
    )
    filtered = apply_filter(image, spec)

**Identity short-circuit.** A tiny blur is not worth the smoothing pass, so
the input comes back unchanged (the same object, not a copy):

.. code-block:: python

    spec = NavFilterSpec(kind=NavFilterKind.ISOTROPIC_GAUSSIAN, sigma_xy=(0.2, 0.2))
    assert apply_filter(image, spec) is image
