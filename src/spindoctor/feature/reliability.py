"""Reliability gate -- drops features whose self-assessed reliability is too low.

Each feature carries a [0, 1] reliability score it computed at extraction
time.  The orchestrator runs this gate between extraction and technique
invocation; gated-out features are recorded in the feature inventory with
their rejection reason but are not passed to any technique.

The thresholds are the ``orchestrator.reliability_gate`` configuration section,
one per feature type, which the configuration parses into a
:class:`FeatureReliabilityGate` when it loads.
"""

from dataclasses import dataclass

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.schema import SCHEMA_CONFIG, Unit

__all__ = [
    'FeatureReliabilityGate',
    'GatedFeatureRecord',
]


@dataclass(frozen=True)
class GatedFeatureRecord:
    """Diagnostic record for a feature dropped by the gate.

    Parameters:
        feature: The feature that was dropped.
        reason: Stable string describing why; used by the curator and tests.
    """

    feature: NavFeature
    reason: str


@dataclass(frozen=True)
class FeatureReliabilityGate:
    """Stateless gate filtering features by per-type reliability threshold.

    This is the ``orchestrator.reliability_gate`` configuration section.  It has one
    field for each :class:`~spindoctor.feature.feature_type.NavFeatureType` member,
    named after the member, so every feature type has a threshold.  A feature whose
    reliability is below its type's threshold is gated out.

    Parameters:
        STAR: Minimum reliability of a ``STAR`` feature, in ``[0, 1]``.
        LIMB_ARC: Minimum reliability of a ``LIMB_ARC`` feature, in ``[0, 1]``.
        TERMINATOR_ARC: Minimum reliability of a ``TERMINATOR_ARC`` feature, in
            ``[0, 1]``.
        BODY_DISC: Minimum reliability of a ``BODY_DISC`` feature, in ``[0, 1]``.
        BODY_BLOB: Minimum reliability of a ``BODY_BLOB`` feature, in ``[0, 1]``.
        RING_EDGE: Minimum reliability of a ``RING_EDGE`` feature, in ``[0, 1]``.
        RING_ANNULUS: Minimum reliability of a ``RING_ANNULUS`` feature, in ``[0, 1]``.
        TITAN_LIMB: Minimum reliability of a ``TITAN_LIMB`` feature, in ``[0, 1]``.
        CARTOGRAPHIC_MODEL: Minimum reliability of a ``CARTOGRAPHIC_MODEL`` feature,
            in ``[0, 1]``.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    STAR: Unit
    LIMB_ARC: Unit
    TERMINATOR_ARC: Unit
    BODY_DISC: Unit
    BODY_BLOB: Unit
    RING_EDGE: Unit
    RING_ANNULUS: Unit
    TITAN_LIMB: Unit
    CARTOGRAPHIC_MODEL: Unit

    def threshold(self, feature_type: NavFeatureType) -> float:
        """Return the reliability a feature of one type needs to be kept.

        Parameters:
            feature_type: The feature's type.

        Returns:
            The threshold, in ``[0, 1]``.
        """
        return float(getattr(self, feature_type.name))

    def apply(
        self, features: list[NavFeature]
    ) -> tuple[list[NavFeature], list[GatedFeatureRecord]]:
        """Split ``features`` into kept and gated lists.

        Parameters:
            features: All features emitted by extractors.

        Returns:
            Tuple ``(kept, gated)`` where ``gated`` carries rejection
            reasons.
        """
        kept: list[NavFeature] = []
        gated: list[GatedFeatureRecord] = []
        for feature in features:
            threshold = self.threshold(feature.feature_type)
            if feature.reliability < threshold:
                gated.append(
                    GatedFeatureRecord(
                        feature=feature,
                        reason=(
                            f'reliability_{feature.reliability:.3f}_below_threshold_{threshold:.3f}'
                        ),
                    )
                )
            else:
                kept.append(feature)
        return kept, gated
