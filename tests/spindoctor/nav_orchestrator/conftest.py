"""Shared fixtures for spindoctor.nav_orchestrator tests."""

from __future__ import annotations

import pytest

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.provenance import Provenance


@pytest.fixture
def classifier() -> NavImageClassifierResult:
    """Provide a clean ``NavImageClassifierResult`` for ensemble/curator tests."""
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        unread_line_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )


@pytest.fixture
def provenance() -> Provenance:
    """Provide a minimal ``Provenance`` envelope for tests."""
    return Provenance(
        spindoctor_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )
