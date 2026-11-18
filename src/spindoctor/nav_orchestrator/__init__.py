"""Orchestrator layer — turns observations into navigation results.

The orchestrator extracts features, gates them by reliability, runs each
feasible technique, and reconciles the per-technique results into one
``NavResult``.

Modules:

    ``nav_context``
        ``NavContext`` — per-image global state.
    ``feature_summary``
        ``NavFeatureSummary`` — per-feature post-mortem entry.
    ``image_classifier``
        ``NavImageClassifier`` — quick-fail image-quality classifier.
    ``image_classifier_result``
        ``NavImageClassifierResult`` — output of the classifier.
    ``image_derivatives``
        ``build_image_edge_dt``, ``compute_image_gradient_vu``,
        ``compute_all_image_derivatives``, and ``ImageDerivativesConfig`` —
        shared gradient / edge-DT / gradient-vector computation.  The
        combined entry point (``compute_all_image_derivatives``) shares
        the heavy gaussian + sobel pass between all three products.
    ``provenance``
        ``Provenance`` — reproducibility metadata.
    ``ensemble``
        ``ensemble`` — free function that reconciles per-technique results.
    ``nav_result``
        ``NavResult`` — final output dataclass.
    ``curator``
        ``build_metadata_dict`` — JSON output curation.
    ``status_reason_info``
        ``STATUS_REASON_INFO_TEMPLATE`` — per-``status_reason`` operator log
        templates.
"""

from spindoctor.nav_orchestrator.curator import (
    assert_diagnostic_fields_present,
    build_metadata_dict,
)
from spindoctor.nav_orchestrator.ensemble import (
    EnsembleConfig,
    derive_confidence_rank,
    ensemble,
)
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier import NavImageClassifier
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.image_derivatives import (
    ImageDerivativesConfig,
    build_image_edge_dt,
    compute_all_image_derivatives,
    compute_image_gradient_vu,
)
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator, OrchestratorPrep
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE

__all__ = [
    'STATUS_REASON_INFO_TEMPLATE',
    'EnsembleConfig',
    'ImageDerivativesConfig',
    'NavContext',
    'NavFeatureSummary',
    'NavImageClassifier',
    'NavImageClassifierResult',
    'NavOrchestrator',
    'NavResult',
    'OrchestratorPrep',
    'Provenance',
    'assert_diagnostic_fields_present',
    'build_image_edge_dt',
    'build_metadata_dict',
    'compute_all_image_derivatives',
    'compute_image_gradient_vu',
    'derive_confidence_rank',
    'ensemble',
]
