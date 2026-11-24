"""Unit tests for the shared DT fit-quality gates."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.nav_technique.dt_fit_gates import (
    DTFitGateConfig,
    DTFitGateVerdict,
    evaluate_dt_fit_gates,
)
from spindoctor.nav_technique.dt_fitting import LMRefineResult

_TUNING = {
    'lm_unconverged_confidence_cap': 0.84,
    'spurious_max_polarity_rejection_fraction': 0.35,
    'spurious_unconverged_polarity_rejection_fraction': 0.10,
    'spurious_min_coarse_peak_fraction': 0.05,
}


def _gate_config() -> DTFitGateConfig:
    """Return a gate config built from the reference thresholds."""
    return DTFitGateConfig(**_TUNING)


def _lm_result(
    *,
    converged: bool,
    polarity_rejected_count: int = 0,
    n_vertices: int = 100,
) -> LMRefineResult:
    """Build a minimal healthy LMRefineResult for gate evaluation."""
    return LMRefineResult(
        offset_vu=(0.0, 0.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64),
        residuals_px=np.zeros(n_vertices, dtype=np.float64),
        weights=np.ones(n_vertices, dtype=np.float64),
        rms_px=0.1,
        raw_rms_px=0.1,
        iterations=5,
        converged=converged,
        inlier_count=n_vertices,
        degenerate=False,
        polarity_rejected_count=polarity_rejected_count,
    )


# ---------------------------------------------------------------------------
# Convergence cap
# ---------------------------------------------------------------------------


def test_converged_fit_carries_no_confidence_cap() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.confidence_cap is None
    assert verdict.lm_converged is True


def test_unconverged_fit_carries_the_configured_cap() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=False),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.confidence_cap == pytest.approx(0.84)
    assert verdict.lm_converged is False


def test_unconverged_fit_with_clean_polarity_is_not_spurious() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=False, polarity_rejected_count=5),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.spurious is False


# ---------------------------------------------------------------------------
# Polarity-rejection gates
# ---------------------------------------------------------------------------


def test_polarity_fraction_is_recorded() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True, polarity_rejected_count=11),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.polarity_rejection_fraction == pytest.approx(0.11)


def test_standalone_polarity_gate_fires_at_threshold() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True, polarity_rejected_count=35),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert 'polarity_rejection_fraction' in verdict.spurious_reasons


def test_standalone_polarity_gate_quiet_below_threshold() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True, polarity_rejected_count=34),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.spurious is False


def test_combined_gate_fires_on_unconverged_elevated_rejection() -> None:
    """The mis-lock signature: iteration cap plus 12.7% polarity rejection."""
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=False, polarity_rejected_count=127, n_vertices=1000),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=1000,
        use_polarity=True,
    )
    assert 'lm_unconverged_with_polarity_rejection' in verdict.spurious_reasons


def test_combined_gate_quiet_when_converged() -> None:
    """A healthy multi-body frame: 11% rejection but the LM converges."""
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True, polarity_rejected_count=110, n_vertices=1000),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=1000,
        use_polarity=True,
    )
    assert verdict.spurious is False


def test_polarity_gates_inert_without_polarity() -> None:
    """A polarity-free technique (RingEdgeNav) never trips the polarity gates."""
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=False, polarity_rejected_count=90),
        _gate_config(),
        coarse_peak_fraction=0.8,
        total_vertex_count=100,
        use_polarity=False,
    )
    assert verdict.spurious is False
    assert verdict.polarity_rejection_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Coarse-peak gate
# ---------------------------------------------------------------------------


def test_coarse_gate_fires_below_threshold() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True),
        _gate_config(),
        coarse_peak_fraction=0.04,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.spurious_reasons == ('coarse_peak_fraction',)


def test_coarse_gate_quiet_at_threshold() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=True),
        _gate_config(),
        coarse_peak_fraction=0.05,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert verdict.spurious is False


# ---------------------------------------------------------------------------
# Verdict shape
# ---------------------------------------------------------------------------


def test_verdict_spurious_property_reflects_reasons() -> None:
    verdict = DTFitGateVerdict(
        spurious_reasons=(),
        confidence_cap=None,
        polarity_rejection_fraction=0.0,
        coarse_peak_fraction=1.0,
        lm_converged=True,
    )
    assert verdict.spurious is False


def test_multiple_gates_accumulate_reasons() -> None:
    verdict = evaluate_dt_fit_gates(
        _lm_result(converged=False, polarity_rejected_count=40),
        _gate_config(),
        coarse_peak_fraction=0.01,
        total_vertex_count=100,
        use_polarity=True,
    )
    assert set(verdict.spurious_reasons) == {
        'polarity_rejection_fraction',
        'lm_unconverged_with_polarity_rejection',
        'coarse_peak_fraction',
    }
