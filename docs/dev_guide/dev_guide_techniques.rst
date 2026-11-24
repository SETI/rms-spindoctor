=====================
Navigation Techniques
=====================

A :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` consumes a subset of the per-image
:class:`~spindoctor.feature.feature.NavFeature` set plus a per-image
:class:`~spindoctor.nav_orchestrator.nav_context.NavContext` and produces one
:class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult` carrying a calibrated
translation offset, a 2x2 (or 3x3 with rotation) covariance, a calibrated [0, 1] confidence,
and a typed diagnostics dataclass. The orchestrator's ensemble combine reconciles every
technique's result into a single :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` outcome.

Concrete subclasses self-register via ``__init_subclass__`` and are discovered by the
orchestrator through ``NavTechnique._registry``. Each
technique declares its accepted feature types, whether it requires a pass-1 prior, and the
attribute set its confidence formula may read. Per-technique tunables and confidence-formula
coefficients live in the technique's block of the ``techniques`` section in
``src/spindoctor/config_files/config_510_techniques.yaml``, which is parsed when the
configuration loads (see :ref:`config-techniques-section`). A technique whose tuning is
typed declares ``Tuning``, a frozen dataclass of its tuning keys, and its block is keyed
by its snake_case key, as in ``body_limb``.

Two cross-cutting kinds of pages live under this section:

- **Per-technique pages** — one page per concrete
  :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` subclass, documenting the
  technique's theory, configuration, implementation, and worked examples.
- **Shared-infrastructure pages** — one page per shared algorithmic component reused across
  techniques (DT fitting, image-side derivatives, confidence calibration, feasibility
  reporting, per-technique diagnostics dataclasses).

Per-technique pages cross-reference the shared-infrastructure pages instead of duplicating
their content.

Shared infrastructure
---------------------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_dt_fitting
   dev_guide_techniques_image_derivatives
   dev_guide_techniques_confidence
   dev_guide_techniques_feasibility
   dev_guide_techniques_diagnostics

Star techniques
---------------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_star_unique_match
   dev_guide_techniques_star_field
   dev_guide_techniques_star_refine

Body techniques
---------------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_body_limb
   dev_guide_techniques_body_terminator
   dev_guide_techniques_body_disc
   dev_guide_techniques_body_blob

Titan techniques
----------------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_titan_haze

Ring techniques
---------------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_ring_edge
   dev_guide_techniques_ring_annulus

Manual
------

.. toctree::
   :maxdepth: 4

   dev_guide_techniques_manual
