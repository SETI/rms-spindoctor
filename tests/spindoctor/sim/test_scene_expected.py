"""Validation tests for the scene ``expected`` block and its known-error check."""

from __future__ import annotations

import pytest

from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params

from .test_scene import _sim_params


def test_validate_sim_params_accepts_expected_block() -> None:
    """A well-formed expected outcome block validates."""
    params = _sim_params()
    params['expected'] = {
        'status': 'failed',
        'confidence_tier': 'failed',
        'status_reason': 'no_feasible_techniques',
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_accepts_expected_with_null_tier() -> None:
    """A success expected block may leave the tier unasserted (null)."""
    params = _sim_params()
    params['expected'] = {'status': 'success', 'confidence_tier': None}
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_expected_status() -> None:
    """An unknown expected status fails validation."""
    params = _sim_params()
    params['expected'] = {'status': 'triumphant', 'confidence_tier': None}
    with pytest.raises(SimSceneValidationError, match=r'expected.status'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_failed_status_with_wrong_tier() -> None:
    """A failed status pins the failed tier (the sidecar cross-field rule)."""
    params = _sim_params()
    params['expected'] = {'status': 'failed', 'confidence_tier': 'low'}
    with pytest.raises(SimSceneValidationError, match=r'confidence_tier=failed'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_expected_status_reason() -> None:
    """An out-of-vocabulary expected status_reason fails validation."""
    params = _sim_params()
    params['expected'] = {'status': 'failed', 'confidence_tier': 'failed', 'status_reason': 'vibes'}
    with pytest.raises(SimSceneValidationError, match=r'status_reason'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_known_error_check() -> None:
    """A known-error check (error value plus tolerance) validates on a success."""
    params = _sim_params()
    params['expected'] = {
        'status': 'success',
        'confidence_tier': 'high',
        'known_offset_error_px': 3.0,
        'known_offset_error_tol_px': 1.0,
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_known_error_without_tolerance() -> None:
    """A known_offset_error_px pin without its tolerance band fails validation."""
    params = _sim_params()
    params['expected'] = {
        'status': 'success',
        'confidence_tier': 'high',
        'known_offset_error_px': 3.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'known_offset_error_tol_px'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_tolerance_without_known_error() -> None:
    """A tolerance band without the pinned error value fails validation."""
    params = _sim_params()
    params['expected'] = {
        'status': 'success',
        'confidence_tier': 'high',
        'known_offset_error_tol_px': 1.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'known_offset_error_px'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_known_error_on_failed_status() -> None:
    """A known_offset_error_px pin on a failed status fails validation."""
    params = _sim_params()
    params['expected'] = {
        'status': 'failed',
        'confidence_tier': 'failed',
        'known_offset_error_px': 3.0,
        'known_offset_error_tol_px': 1.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'status=success'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_nonpositive_known_error() -> None:
    """A zero known_offset_error_px fails the positive-number check."""
    params = _sim_params()
    params['expected'] = {
        'status': 'success',
        'confidence_tier': 'high',
        'known_offset_error_px': 0.0,
        'known_offset_error_tol_px': 1.0,
    }
    with pytest.raises(SimSceneValidationError, match=r'known_offset_error_px'):
        validate_sim_params(params)
