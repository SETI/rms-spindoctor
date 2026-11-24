"""Feature layer for autonomous navigation.

Each scene element the orchestrator considers is decomposed into
``NavFeature`` instances emitted by registered ``NavModel`` instances.
Features carry their own uncertainty, preferred filter, and reliability;
techniques pick them up by feature type and produce per-technique offsets.

Modules:

    ``feature_type``
        ``NavFeatureType`` enum (single canonical list of feature types).
    ``geometry``
        ``NavFeatureGeometry`` sum type and per-type geometry payload
        dataclasses.
    ``flags``
        ``NavFeatureFlags`` sum type and per-type flag dataclasses.
    ``feature``
        ``NavFeature`` dataclass and ``NavReliabilityBreakdown``.
    ``reliability``
        ``FeatureReliabilityGate``, the per-type reliability thresholds.
    ``constants``
        Module-level physical constants (cap angles, sentinel values).

Thread safety: every class in this package is stateless or is wrapped in a
frozen dataclass.  Concurrent access from multiple threads is safe so long
as each thread holds its own ``ObsSnapshotInst`` (the underlying ``oops``
``Backplane`` queries mutate global precision state).
"""

from spindoctor.feature.composition import compose_dialog_overlay, compose_template_features
from spindoctor.feature.constants import (
    AGREEMENT_FACTOR_CAP,
    COMBINED_CONFIDENCE_CAP,
    INCIDENCE_FACTOR_ANGLE_CAP_DEG,
    INCIDENCE_FACTOR_CLIP_DEG,
    JSON_INF_SENTINEL,
    MAX_INCIDENCE_FACTOR_CAP,
    MIN_ANISOTROPIC_SMEAR_PX,
)
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import (
    BodyBlobFlags,
    BodyDiscFlags,
    CartographicModelFlags,
    LimbArcFlags,
    NavFeatureFlags,
    RingAnnulusFlags,
    RingEdgeFlags,
    StarFlags,
    TerminatorArcFlags,
    TitanHazeFlags,
)
from spindoctor.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    CartographicModelGeometry,
    LimbPolyline,
    NavFeatureGeometry,
    RingAnnulusGeometry,
    RingEdgePolyline,
    StarGeometry,
    TerminatorPolyline,
    TitanHazeGeometry,
)
from spindoctor.feature.reliability import FeatureReliabilityGate, GatedFeatureRecord

__all__ = [
    'AGREEMENT_FACTOR_CAP',
    'COMBINED_CONFIDENCE_CAP',
    'INCIDENCE_FACTOR_ANGLE_CAP_DEG',
    'INCIDENCE_FACTOR_CLIP_DEG',
    'JSON_INF_SENTINEL',
    'MAX_INCIDENCE_FACTOR_CAP',
    'MIN_ANISOTROPIC_SMEAR_PX',
    'BodyBlobFlags',
    'BodyBlobGeometry',
    'BodyDiscFlags',
    'BodyDiscGeometry',
    'CartographicModelFlags',
    'CartographicModelGeometry',
    'FeatureReliabilityGate',
    'GatedFeatureRecord',
    'LimbArcFlags',
    'LimbPolyline',
    'NavFeature',
    'NavFeatureFlags',
    'NavFeatureGeometry',
    'NavFeatureType',
    'NavReliabilityBreakdown',
    'RingAnnulusFlags',
    'RingAnnulusGeometry',
    'RingEdgeFlags',
    'RingEdgePolyline',
    'StarFlags',
    'StarGeometry',
    'TerminatorArcFlags',
    'TerminatorPolyline',
    'TitanHazeFlags',
    'TitanHazeGeometry',
    'compose_dialog_overlay',
    'compose_template_features',
]
