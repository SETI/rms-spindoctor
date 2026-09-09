"""The simulated scene of the fixture tree.

The one host with no spacecraft and no furnished camera frame, so its document
is the one that correctly records no attitude, no exposure times, no shutter
mode and no loaded kernels, and the one whose results path stub names no
subtree.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.ensemble import derive_confidence_rank
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.status_reason import NavStatusReason

from .shared import SIM_KERNELS, classifier, navigated, provenance


def simulated_scene() -> dict[str, Any]:
    """A simulated scene navigated on one body limb.

    The one host with no spacecraft and no furnished camera frame, so it
    correctly records no attitude and no exposure times, no shutter mode, and
    no loaded kernels.  Its results path stub names no subtree.

    Returns:
        The document.
    """
    inventory = [
        NavFeatureSummary(
            feature_id='limb_arc:MIMAS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:MIMAS',
            reliability=0.88,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(84, 92, 172, 180),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=1.0, incidence_factor=0.97
            ),
        )
    ]
    per_technique = [
        NavTechniqueResult(
            technique_name='BodyLimbNav',
            feature_ids=('limb_arc:MIMAS',),
            offset_px=(1.5, 0.5),
            covariance_px2=np.diag([0.0961, 0.0784]),
            confidence=0.88,
            spurious=False,
            at_edge=False,
            diagnostics=BodyLimbDiagnostics(
                visible_limb_arc_fraction=1.0,
                visible_arc_px=188.0,
                dt_fit_rms_px=0.121,
                lm_iterations=7,
                tukey_inlier_count=186,
                lm_converged=True,
                polarity_rejection_fraction=0.0,
                coarse_peak_fraction=0.914,
            ),
        )
    ]
    result = NavResult.success(
        offset_px=(1.5, 0.5),
        covariance_px2=np.diag([0.0961, 0.0784]),
        confidence=0.88,
        confidence_rank=derive_confidence_rank(confidence=0.88, sigma_px=(0.31, 0.28)),
        status_reason=NavStatusReason.OK,
        per_technique=per_technique,
        feature_inventory=inventory,
        image_classifier=classifier(noise_sigma=0.75, max_dn=0.64, gradient_score=None),
        provenance=provenance(image_et=100.0, kernels=SIM_KERNELS, extractors=('body:MIMAS',)),
        consensus_techniques=['BodyLimbNav'],
    )
    return navigated(
        result,
        image_name='sim_scene_000042.img',
        instrument='sim',
        camera='SIM',
        shutter_mode=None,
        image_shape=(256, 256),
        start=datetime(2026, 8, 8, 16, 47, 55, 180332, tzinfo=UTC),
        elapsed_s=12.5,
        peak_memory_bytes=1073741824,
    )
