=============================================================
Confidence Calibration (Shared Sigmoid-of-Linear Combination)
=============================================================

Overview
========

Confidence calibration is the shared scoring layer that every autonomous navigation technique
uses to convert a typed diagnostics dataclass into a calibrated :math:`[0, 1]` confidence on
its :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`. Each technique declares
a YAML spec — a constant baseline, a list of linear terms keyed by diagnostic-attribute name,
optional hard-zero gates, and an optional post-sigmoid clamp — and the shared evaluator
applies that spec uniformly. Centralizing the math means a config-load validation pass can
verify every spec at startup and adding a new technique requires no new scoring code.

Theory
======

Every autonomous technique's confidence formula has the same shape:

.. math::

    z = \alpha_{0} + \sum_{i} \alpha_{i} \, \mathrm{normalize}_{i}(x_{i}),
    \qquad
    c = \sigma(z),

where :math:`x_{i}` is the raw value of the *i*-th diagnostic attribute on the technique's
result, :math:`\mathrm{normalize}_{i}` applies a per-term offset / divisor / cap
transformation, :math:`\alpha_{0}` and :math:`\alpha_{i}` are configured coefficients, and
:math:`\sigma` is the logistic sigmoid.

Per-term normalization
----------------------

The normalization transformation applied to each raw value is

.. math::

    \mathrm{normalize}(x) =
    \begin{cases}
      \mathrm{clip}\!\left(\dfrac{x - o}{d},\; 0,\; \mathrm{cap}\right) & \text{when a cap is set} \\[6pt]
      \dfrac{x - o}{d} & \text{otherwise}
    \end{cases}

where :math:`o` is the optional offset (default zero), :math:`d` is the divisor (default one,
required non-zero), and the cap clamps the post-scale value to :math:`[0, \mathrm{cap}]` when
present. The cap, when set, must lie in :math:`[0, 1]`.

The offset shifts the term's "responsive interval" along the raw axis (subtracting the offset
moves the threshold for :math:`\mathrm{normalize}(x) = 0`); the divisor sets how quickly the
term saturates as the raw value grows; the cap stops a runaway raw value from dominating the
sigmoid argument.

Sigmoid combination
-------------------

The summed argument :math:`z` is fed into the numerically-stable logistic sigmoid

.. math::

    \sigma(z) =
    \begin{cases}
      \dfrac{1}{1 + e^{-z}} & z \ge 0 \\[6pt]
      \dfrac{e^{z}}{1 + e^{z}} & z < 0
    \end{cases}

so the formula stays well-defined for arbitrarily large positive or negative arguments.

Hard-zero gates
---------------

Each spec may declare a mapping of diagnostic-attribute name to expected boolean. Before the
sigmoid is evaluated, the evaluator checks each entry: if the attribute on the diagnostics
object is truthy and the spec demands True (or both are False), the corresponding short-circuit
fires and the calibrated confidence is forced to zero, regardless of the linear-combination
sum. Hard-zero gates are how techniques surface their structural failure modes (the converged
offset sits on the search-window edge; the M-estimator fit was spurious; the per-feature
reliability gate dropped every input).

Post-sigmoid hard cap
---------------------

After the sigmoid evaluates, an optional ``hard_cap`` in :math:`[0, 1]` clamps the result from
above. This is the right place to encode an algorithmic ceiling that does not depend on the
formula's input — for example, a brightness-weighted-centroid technique whose output is
intrinsically less informative than a limb fit caps its post-sigmoid confidence at 0.4 even
when every term saturates.

Per-term breakdown
------------------

The evaluator can return a per-term contribution trace alongside the calibrated confidence.
The trace records, for each term, the raw attribute value, the normalized value, the alpha,
and the resulting alpha-times-normalized contribution to the sigmoid argument. Logging this
trace at INFO when confidence falls below a threshold gives an operator a one-line diagnostic
of which term (or which hard-zero gate) drove the result down.

Restrictions and assumptions
----------------------------

- Term divisors must be strictly non-zero; the dataclass constructor rejects zero at config-load
  time.
- Caps, when set, must lie in :math:`[0, 1]`.
- Every term's feature name and every hard-zero key must reference an attribute the diagnostics
  object actually carries. The orchestrator's startup-time
  :func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` walk catches
  unknown attribute names before any image is processed; if a YAML spec references an unknown
  field the process fails fast.
- The offset / divisor / cap transformation is dimensional but the framework is unit-agnostic —
  the YAML divisor must be quoted in the same units as the raw diagnostic value.

Sources of uncertainty
----------------------

The calibrated confidence is the output of a deterministic functional form; there is no
stochastic component. What it *does* capture is the empirical relationship between the
documented diagnostics and the per-image fit quality, as encoded by the per-technique YAML
coefficients. What it does *not* capture is the diagnostic's own measurement uncertainty
(if a technique misreports its DT residual, the confidence formula will trust the misreport),
nor any cross-technique consistency (the ensemble combine handles that).

Configuration
=============

Confidence calibration is the *consumer* of YAML, not a producer. Every technique's
confidence spec lives under ``techniques.<TechniqueName>`` in
``src/spindoctor/config_files/config_510_techniques.yaml`` alongside its ``tuning`` block. The spec
shape is:

- ``alpha0`` — float (dimensionless). Baseline contribution to the sigmoid argument. Negative
  values pull the sigmoid below 0.5 by default; positive values push it above.
- ``terms`` — list of mappings. Each entry has:

  - ``feature`` — str, the diagnostic-attribute name. Must exist on the technique's
    diagnostics dataclass and appear in the technique's
    :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` allow-list.
  - ``alpha`` — float (dimensionless). Linear coefficient applied after normalization.
  - ``offset`` — float, default ``0.0``. Subtracted from the raw value before division. Same
    units as the raw value.
  - ``divisor`` — float, default ``1.0``. Divides after offset; must be non-zero. Same units
    as the raw value.
  - ``cap_at`` — float in :math:`[0, 1]` or ``null``, default ``null``. Optional upper bound
    on the normalized value. When set, both clips negative values to 0 and the post-scale
    value to ``cap_at``.

- ``hard_zero_if`` — mapping of str to bool, default empty. Keys must reference attributes
  the diagnostics object carries (or live on the technique's adapter object). When the
  attribute matches the demanded boolean, the calibrated confidence is forced to zero.
- ``hard_cap`` — float in :math:`[0, 1]` or ``null``, default ``null``. Post-sigmoid clamp.

This module exposes no module-level numeric constants of its own; the spec values come from
YAML and the runtime constructors validate them.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/confidence.py`` — the
  :class:`~spindoctor.nav_technique.confidence.ConfidenceSpec`,
  :class:`~spindoctor.nav_technique.confidence.ConfidenceTerm`,
  :class:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution`, and
  :class:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown` dataclasses plus the
  :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination` evaluator.
- ``src/spindoctor/nav_technique/confidence_config.py`` — YAML-to-:class:`~spindoctor.nav_technique.confidence.ConfidenceSpec`
  loader used by :class:`~spindoctor.config.config.Config` at startup.
- ``src/spindoctor/nav_technique/nav_technique.py`` —
  :func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` and
  :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, the orchestrator-side
  validation and logging helpers.

Public surface (autodocumented at :doc:`/api_reference/api_nav_technique`):

- :class:`~spindoctor.nav_technique.confidence.ConfidenceSpec` — the per-technique formula. Fields:

  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.alpha0` — sigmoid-argument baseline.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.terms` — tuple of
    :class:`~spindoctor.nav_technique.confidence.ConfidenceTerm` linear contributions.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_zero_if` — short-circuit map.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap` — optional post-sigmoid clamp.

- :class:`~spindoctor.nav_technique.confidence.ConfidenceTerm` — one linear term. Fields:

  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.feature` — diagnostic-attribute name.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.alpha` — coefficient.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.offset` — pre-scale offset.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.divisor` — pre-scale divisor.
  - :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.cap_at` — optional post-scale upper bound.

- :class:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution` — one term's contribution
  trace. Fields:
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution.feature`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution.raw`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution.normalized`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution.alpha`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution.contribution`.

- :class:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown` — full evaluation trace. Fields:
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.confidence`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.sigmoid_arg`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.alpha0`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.terms`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.hard_zero`,
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown.hard_cap_applied`.

- :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination` — the evaluator. Returns
  the calibrated confidence, or a ``(confidence, ConfidenceBreakdown)`` pair when
  ``return_breakdown=True``.

Call path traced through
:func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`:

1. Walk the spec's :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_zero_if`. For
   each entry, fetch the named attribute off the diagnostics object (raising :exc:`ValueError`
   when missing) and compare against the demanded boolean. If any condition holds,
   short-circuit with a ``0.0`` confidence (and a hard-zero-tagged
   :class:`~spindoctor.nav_technique.confidence.ConfidenceBreakdown` when the caller asked for one).
2. Initialize the sigmoid argument with
   :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.alpha0`.
3. For each term in :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.terms`, fetch the
   named attribute, apply the offset / divisor / cap normalization, multiply by the alpha,
   and accumulate the contribution. Record the per-term contribution in a
   :class:`~spindoctor.nav_technique.confidence.ConfidenceTermContribution` when a breakdown was
   requested.
4. Pass the accumulated argument through the numerically-stable logistic sigmoid.
5. Apply :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_cap` when set; record
   whether the cap fired in the breakdown.
6. Return the calibrated confidence (and the breakdown when requested).

The orchestrator-side helpers are:

- :func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` — invoked at
  config-load time. Walks every registered
  :class:`~spindoctor.nav_technique.nav_technique.NavTechnique` whose
  :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_spec` was loaded and verifies
  that every term's
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceTerm.feature` and every
  :attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_zero_if` key appears in the
  technique's
  :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` allow-list.
  Raises :exc:`ValueError` on the first unknown name.
- :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown` — emits the breakdown at
  DEBUG always, and also at INFO when the calibrated confidence falls at or below a
  ``low_threshold`` (default 0.1). This is what surfaces "alpha=-1.303 dt_fit_rms_px=8.7
  contribution=-11.34 drove confidence below the low-confidence threshold" in the
  per-image log.

Examples
========

**Sigmoid-of-linear illustration.**  With ``alpha0 = -1.0`` and a single term whose alpha is
``3.0``, no offset, no divisor, no cap, on a diagnostic whose raw value is ``0.7``: the
sigmoid argument is :math:`-1.0 + 3.0 \cdot 0.7 = 1.1`, the sigmoid evaluates to approximately
``0.7503``, and the calibrated confidence is approximately 0.75. Holding the formula fixed
and raising the diagnostic to ``0.9`` pushes the argument to ``1.7`` and the confidence to
approximately 0.846; lowering to ``0.3`` drops the argument to ``-0.1`` and the confidence to
approximately 0.475.

**Hard-zero override.**  The ``BodyLimbNav`` spec declares ``hard_zero_if: {at_edge: true,
spurious: true}``. When a fit converges with
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` true (the offset hit
the search-window boundary), the linear combination is irrelevant — the calibrated confidence
is ``0.0`` regardless of how the dt-fit RMS or visible-arc terms scored. The breakdown
returned in this case carries a ``hard_zero='at_edge'`` annotation so the operator log line
explains the zero.

**Post-sigmoid hard cap.**  The ``BodyBlobNav`` spec declares ``hard_cap: 0.4``. Even when
every term saturates (large blob, high SNR, multi-blob composite), the calibrated confidence
cannot exceed 0.4 — a brightness-weighted centroid is structurally weaker evidence than a limb
fit and the cap encodes that fact independently of the per-term coefficients.

**Validation at startup.**  If the YAML spec for a technique declares
``feature: dt_fit_rms_px`` for a star technique whose
:class:`~spindoctor.nav_technique.diagnostics.StarRefineDiagnostics` does not carry
``dt_fit_rms_px``, the
:func:`~spindoctor.nav_technique.nav_technique.validate_registered_confidence_specs` walk fails with
a :exc:`ValueError` naming the technique class and the unknown attribute, before any image is
processed. The same check fires for unknown
:attr:`~spindoctor.nav_technique.confidence.ConfidenceSpec.hard_zero_if` keys.

**Worked breakdown.**  A converged ``BodyLimbNav`` fit with
:attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction`
``0.85``,
:attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px` ``0.4`` px, and
:attr:`~spindoctor.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px` ``120`` px feeds
the spec ``alpha0 = 0.132``, ``alpha(visible_limb_arc_fraction) = 1.068``,
``alpha(dt_fit_rms_px) = -1.303``, ``alpha(visible_arc_px / 440, capped at 1) = 0.776``. The
sigmoid argument is
:math:`0.132 + 1.068 \cdot 0.85 - 1.303 \cdot 0.4 + 0.776 \cdot (120 / 440) = 0.730`, the
sigmoid evaluates to approximately ``0.675``, and the technique reports a calibrated
confidence of ~0.67. When :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`
fires, every term's raw / normalized / contribution numbers appear in the per-image log so an
operator can trace which diagnostic carried the score.
