"""Tests for ``spindoctor.nav_technique.confidence.evaluate_sigmoid_combination``."""

import math

import pytest

from spindoctor.nav_technique.confidence import (
    ConfidenceSpec,
    ConfidenceTerm,
    evaluate_sigmoid_combination,
)
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.schema import RuleError


def test_evaluate_returns_in_unit_interval() -> None:
    """Output of the formula is always in [0, 1]."""
    spec = ConfidenceSpec(alpha0=0.0, terms=())
    diag = BodyLimbDiagnostics()
    result = evaluate_sigmoid_combination(spec, diag)
    assert 0.0 <= result <= 1.0
    # alpha0=0 → sigmoid(0) == 0.5.
    assert math.isclose(result, 0.5, abs_tol=1e-9)


def test_evaluate_handles_negative_argument() -> None:
    """Very negative arguments produce values near zero."""
    spec = ConfidenceSpec(alpha0=-100.0, terms=())
    diag = BodyLimbDiagnostics()
    result = evaluate_sigmoid_combination(spec, diag)
    assert result < 1e-30


def test_evaluate_handles_positive_argument() -> None:
    """Very positive arguments produce values near one (no overflow)."""
    spec = ConfidenceSpec(alpha0=100.0, terms=())
    diag = BodyLimbDiagnostics()
    result = evaluate_sigmoid_combination(spec, diag)
    # Float64 rounds sigmoid(100) all the way up to 1.0; that's fine.
    assert result <= 1.0
    assert result >= 1.0 - 1e-9


def test_evaluate_uses_term_value() -> None:
    """A term with alpha=2 and feature value 1 contributes +2 to the argument."""
    spec = ConfidenceSpec(
        alpha0=0.0,
        terms=(ConfidenceTerm(feature='dt_fit_rms_px', alpha=2.0),),
    )
    diag = BodyLimbDiagnostics(dt_fit_rms_px=1.0)
    # sigmoid(0 + 2*1) = sigmoid(2) ≈ 0.8808.
    result = evaluate_sigmoid_combination(spec, diag)
    assert math.isclose(result, 1.0 / (1.0 + math.exp(-2.0)), rel_tol=1e-9)


def test_evaluate_offset_subtracts_first() -> None:
    """``offset`` subtracts before scaling."""
    spec = ConfidenceSpec(
        alpha0=0.0,
        terms=(ConfidenceTerm(feature='dt_fit_rms_px', alpha=1.0, offset=2.0),),
    )
    diag = BodyLimbDiagnostics(dt_fit_rms_px=3.0)
    # (3.0 - 2.0) / 1.0 = 1.0; sigmoid(1) ≈ 0.731.
    result = evaluate_sigmoid_combination(spec, diag)
    assert math.isclose(result, 1.0 / (1.0 + math.exp(-1.0)), rel_tol=1e-9)


def test_evaluate_divisor_scales_value() -> None:
    """``divisor`` divides after offset."""
    spec = ConfidenceSpec(
        alpha0=0.0,
        terms=(ConfidenceTerm(feature='dt_fit_rms_px', alpha=1.0, divisor=2.0),),
    )
    diag = BodyLimbDiagnostics(dt_fit_rms_px=4.0)
    # (4.0 - 0.0) / 2.0 = 2.0; sigmoid(2) ≈ 0.881.
    result = evaluate_sigmoid_combination(spec, diag)
    assert math.isclose(result, 1.0 / (1.0 + math.exp(-2.0)), rel_tol=1e-9)


def test_evaluate_cap_at_clamps() -> None:
    """``cap_at`` clamps the post-scale value."""
    spec = ConfidenceSpec(
        alpha0=0.0,
        terms=(ConfidenceTerm(feature='dt_fit_rms_px', alpha=1.0, cap_at=1.0),),
    )
    diag = BodyLimbDiagnostics(dt_fit_rms_px=100.0)
    # 100 capped to 1.0; sigmoid(1) ≈ 0.731.
    result = evaluate_sigmoid_combination(spec, diag)
    assert math.isclose(result, 1.0 / (1.0 + math.exp(-1.0)), rel_tol=1e-9)


def test_evaluate_hard_zero_gate_fires() -> None:
    """A hard-zero condition forces confidence = 0."""
    spec = ConfidenceSpec(
        alpha0=10.0,  # would otherwise saturate near 1
        hard_zero_if={'lm_iterations': True},
    )

    # Use a frozen dataclass-derived class with iteration count truthy.
    diag = BodyLimbDiagnostics(lm_iterations=5)
    result = evaluate_sigmoid_combination(spec, diag)
    assert result == 0.0


def test_evaluate_hard_cap_clamps() -> None:
    """``hard_cap`` clamps the final sigmoid output."""
    spec = ConfidenceSpec(alpha0=10.0, hard_cap=0.4)
    diag = BodyLimbDiagnostics()
    result = evaluate_sigmoid_combination(spec, diag)
    assert result == 0.4


def test_evaluate_unknown_feature_raises() -> None:
    """A term referencing a non-existent attribute raises with field name."""
    spec = ConfidenceSpec(
        alpha0=0.0,
        terms=(ConfidenceTerm(feature='no_such_field', alpha=1.0),),
    )
    diag = BodyLimbDiagnostics()
    with pytest.raises(ValueError, match='no_such_field'):
        evaluate_sigmoid_combination(spec, diag, technique_name='TestNav')


def test_evaluate_unknown_hard_zero_attribute_raises() -> None:
    """A hard-zero condition referencing a non-existent attribute raises."""
    spec = ConfidenceSpec(alpha0=0.0, hard_zero_if={'not_a_field': True})
    diag = BodyLimbDiagnostics()
    with pytest.raises(ValueError, match='not_a_field'):
        evaluate_sigmoid_combination(spec, diag)


def test_a_blank_term_feature_is_refused() -> None:
    """A confidence term must name a diagnostic attribute."""
    with pytest.raises(RuleError, match='expected the name of a diagnostic attribute'):
        ConfidenceTerm(feature='  ', alpha=1.0)


def test_a_zero_term_divisor_is_refused() -> None:
    """A confidence term cannot divide by zero."""
    with pytest.raises(RuleError, match='expected a non-zero divisor'):
        ConfidenceTerm(feature='dt_fit_rms_px', alpha=1.0, divisor=0.0)
