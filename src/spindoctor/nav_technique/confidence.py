"""Shared sigmoid-of-linear-combination confidence formula.

Every NavTechnique converts its ``NavTechniqueDiagnostics`` into a
calibrated [0, 1] confidence using the same shape:

    confidence = sigmoid(alpha0 + sum_i alpha_i * normalize_i(x_i))

where ``normalize_i`` applies the spec's ``offset`` -> ``divisor`` -> ``cap_at``
transformation.  Hard-zero gates (e.g. ``at_edge=True``) force confidence
to 0; an optional ``hard_cap`` clamps the result.

The math lives here so techniques' YAML formula specs can be evaluated
uniformly and a config-load validation pass can verify every spec at
startup.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from spindoctor.schema import SCHEMA_CONFIG, Bool, Float, RuleError, Str, Unit

__all__ = [
    'ConfidenceBreakdown',
    'ConfidenceSpec',
    'ConfidenceTerm',
    'ConfidenceTermContribution',
    'evaluate_sigmoid_combination',
]


@dataclass(frozen=True)
class ConfidenceTerm:
    """One term in the sigmoid-of-linear-combination formula.

    A term is one entry of a technique block's ``terms`` list, which the schema core
    parses when the configuration loads.

    Parameters:
        feature: Name of the diagnostic-dataclass attribute supplying the
            raw value.
        alpha: Linear-combination coefficient.
        offset: Subtracted from the raw value before scaling.  A term that leaves it
            out shifts nothing.
        divisor: Divides the raw value after the offset.  A term that leaves it out
            scales nothing.  Must be non-zero.
        cap_at: Upper bound on the scaled value, which is then clamped to
            ``[0, cap_at]``.  A term that leaves it out is not clamped.

    Raises:
        RuleError: If ``feature`` is blank or ``divisor`` is zero.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    feature: Str
    alpha: Float
    offset: Float = 0.0
    divisor: Float = 1.0
    cap_at: Unit | None = None

    def __post_init__(self) -> None:
        """Refuse a blank attribute name and a zero divisor."""
        if not self.feature.strip():
            raise RuleError('expected the name of a diagnostic attribute', at='feature')
        if self.divisor == 0.0:
            raise RuleError('expected a non-zero divisor', at='divisor')


@dataclass(frozen=True)
class ConfidenceSpec:
    """Full confidence formula spec for one NavTechnique.

    These are the confidence keys of a technique block, which the schema core parses
    when the configuration loads (:mod:`spindoctor.nav_technique.confidence_config`).

    Parameters:
        alpha0: Constant term in the sigmoid argument.
        terms: The linear terms.  A block that leaves the key out has none.
        hard_zero_if: Mapping of diagnostic-attribute names to expected boolean values;
            if any condition holds, confidence = 0.  A block that leaves the key out
            has none.
        hard_cap: Upper-bound clamp applied after the sigmoid.  A block that leaves it
            out is not clamped.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    alpha0: Float
    terms: tuple[ConfidenceTerm, ...] = ()
    hard_zero_if: dict[Str, Bool] = field(default_factory=dict)
    hard_cap: Unit | None = None

    def __post_init__(self) -> None:
        """Keep a copy of ``hard_zero_if``, so the caller cannot change it afterward."""
        object.__setattr__(self, 'hard_zero_if', dict(self.hard_zero_if))


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize(value: float, term: ConfidenceTerm) -> float:
    """Apply offset / divisor / cap_at to a raw diagnostic value."""
    scaled = (value - term.offset) / term.divisor
    if term.cap_at is not None:
        if scaled < 0.0:
            scaled = 0.0
        elif scaled > term.cap_at:
            scaled = term.cap_at
    return scaled


@dataclass(frozen=True)
class ConfidenceTermContribution:
    """One term's contribution to the sigmoid argument.

    Parameters:
        feature: The diagnostic-attribute name.
        raw: The raw value read off the diagnostics object.
        normalized: ``raw`` after offset / divisor / cap_at.
        alpha: Linear coefficient applied.
        contribution: ``alpha * normalized`` — the term's signed
            contribution to the sigmoid argument.
    """

    feature: str
    raw: float
    normalized: float
    alpha: float
    contribution: float


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Per-step trace of a sigmoid-combination evaluation.

    Parameters:
        confidence: Final ``[0, 1]`` confidence after sigmoid + hard_cap.
        sigmoid_arg: The argument fed into the sigmoid (sum of alpha0 +
            term contributions).
        alpha0: The constant baseline term.
        terms: Per-term contributions.
        hard_zero: Name of the ``hard_zero_if`` attribute that fired, or
            ``None`` when no hard-zero gate applied.
        hard_cap_applied: True if the final sigmoid value was clamped by
            ``spec.hard_cap``.
    """

    confidence: float
    sigmoid_arg: float
    alpha0: float
    terms: tuple[ConfidenceTermContribution, ...]
    hard_zero: str | None
    hard_cap_applied: bool


def evaluate_sigmoid_combination(
    spec: ConfidenceSpec,
    diagnostics: Any,
    *,
    technique_name: str = '',
    return_breakdown: bool = False,
) -> Any:
    """Evaluate the spec's sigmoid formula against a diagnostics object.

    Parameters:
        spec: The technique's confidence formula spec.
        diagnostics: Diagnostics dataclass instance for the technique.
        technique_name: Optional human-readable identifier used in error
            messages.
        return_breakdown: When True, return a tuple
            ``(confidence, ConfidenceBreakdown)`` so callers can log a
            per-term explanation of low / zero confidence values.

    Returns:
        Calibrated confidence in ``[0, 1]``, or
        ``(confidence, ConfidenceBreakdown)`` when
        ``return_breakdown=True``.

    Raises:
        ValueError: if a term references an attribute that does not exist
            on the diagnostics object.  The message names the missing
            attribute and the technique.
    """
    contributions: list[ConfidenceTermContribution] = []
    # Hard-zero gates first: short-circuit if any condition holds.
    for attr_name, required in spec.hard_zero_if.items():
        if not hasattr(diagnostics, attr_name):
            raise ValueError(
                f'confidence spec {technique_name!r}: hard_zero_if attribute '
                f'{attr_name!r} not found on diagnostics '
                f'{type(diagnostics).__name__}'
            )
        actual = getattr(diagnostics, attr_name)
        if bool(actual) == bool(required):
            if not return_breakdown:
                return 0.0
            breakdown = ConfidenceBreakdown(
                confidence=0.0,
                sigmoid_arg=float('nan'),
                alpha0=spec.alpha0,
                terms=tuple(contributions),
                hard_zero=attr_name,
                hard_cap_applied=False,
            )
            return 0.0, breakdown
    # Linear-combination of normalized terms.
    arg = spec.alpha0
    for term in spec.terms:
        if not hasattr(diagnostics, term.feature):
            raise ValueError(
                f'confidence spec {technique_name!r}: feature attribute '
                f'{term.feature!r} not found on diagnostics '
                f'{type(diagnostics).__name__}'
            )
        raw = float(getattr(diagnostics, term.feature))
        normalized = _normalize(raw, term)
        contribution = term.alpha * normalized
        arg += contribution
        contributions.append(
            ConfidenceTermContribution(
                feature=term.feature,
                raw=raw,
                normalized=normalized,
                alpha=term.alpha,
                contribution=contribution,
            )
        )
    confidence = _sigmoid(arg)
    hard_cap_applied = False
    if spec.hard_cap is not None and confidence > spec.hard_cap:
        confidence = spec.hard_cap
        hard_cap_applied = True
    if not return_breakdown:
        return confidence
    breakdown = ConfidenceBreakdown(
        confidence=confidence,
        sigmoid_arg=arg,
        alpha0=spec.alpha0,
        terms=tuple(contributions),
        hard_zero=None,
        hard_cap_applied=hard_cap_applied,
    )
    return confidence, breakdown
