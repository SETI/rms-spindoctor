"""Per-NavStatusReason operator-readable INFO log templates.

The orchestrator emits one INFO line per status_reason summarizing the
final outcome.  Templates here let tests assert the operator-readable
narrative for every reason.
"""

from spindoctor.support.status_reason import NavStatusReason

__all__ = ['STATUS_REASON_INFO_TEMPLATE']


STATUS_REASON_INFO_TEMPLATE: dict[NavStatusReason, list[str]] = {
    NavStatusReason.OK: [
        'Final: status=ok',
    ],
    NavStatusReason.RANK_1_ONLY: [
        'Final: status=ok',
        'Result is rank-1: one axis unobservable',
    ],
    NavStatusReason.CONFLICTED_TECHNIQUES: [
        'Final: status=conflicted',
        'Best-vs-runner-up confidence gap below threshold',
    ],
    NavStatusReason.BODY_SHAPE_LOCK_SUSPECT: [
        'Final: status=conflicted',
        'Geometric body lock contradicted by the pose-free blob witness',
    ],
    NavStatusReason.LONE_BLOB_IN_COLLAPSED_REGIME: [
        'Final: status=failed',
        'Lone blob centroid survived a body whose geometric fit self-flagged spurious',
    ],
    NavStatusReason.NO_SIGNAL_IN_IMAGE: [
        'Final: status=failed',
        'Image classifier: blank / dark frame',
    ],
    NavStatusReason.IMAGE_OVEREXPOSED: [
        'Final: status=failed',
        'Image classifier: most pixels at full-well DN',
    ],
    NavStatusReason.MISSING_DATA_DOMINANT: [
        'Final: status=failed',
        'Image classifier: missing-data marker dominates',
    ],
    NavStatusReason.IMAGE_CORRUPT: [
        'Final: status=failed',
        'Image file failed to parse / read',
    ],
    NavStatusReason.KERNELS_UNAVAILABLE: [
        'Final: status=failed',
        'SPICE coverage missing for the image ET',
    ],
    NavStatusReason.INSTRUMENT_NOT_CONFIGURED: [
        'Final: status=failed',
        'No config block for this instrument camera',
    ],
    NavStatusReason.BODY_FILLS_FOV: [
        'Final: status=failed',
        'A body covers the extended frame: no limb on any edge and no disc '
        'extent to measure, so the image is unnavigable rather than merely '
        'unnavigated',
    ],
    NavStatusReason.NO_FEATURES_EXTRACTED: [
        'Final: status=failed',
        'No extractor produced a feature',
    ],
    NavStatusReason.ALL_FEATURES_GATED: [
        'Final: status=failed',
        'Every feature dropped by the reliability gate',
    ],
    NavStatusReason.NO_FEASIBLE_TECHNIQUES: [
        'Final: status=failed',
        "No technique's is_feasible returned True",
    ],
    NavStatusReason.ALL_TECHNIQUES_SPURIOUS: [
        'Final: status=failed',
        'Every technique returned spurious=True',
    ],
    NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD: [
        'Final: status=failed',
        'Combined confidence below min_confidence',
    ],
    NavStatusReason.FINAL_SIGMA_ABOVE_THRESHOLD: [
        'Final: status=failed',
        'Combined offset sigma above every tier max_sigma_px (confident but imprecise)',
    ],
    NavStatusReason.UNOBSERVABLE_OFFSET: [
        'Final: status=failed',
        'Every input covariance shares one null direction',
    ],
    NavStatusReason.CONTRACT_VIOLATION: [
        'Final: status=failed',
        'Internal contract violation (programming error); see the error log',
    ],
    NavStatusReason.INTERNAL_ERROR: [
        'Final: status=failed',
        'A NavModel or NavTechnique raised; the image was not navigated as '
        'designed. The failing component and exception class are in the '
        'metadata document; the traceback is in the error log',
    ],
}
