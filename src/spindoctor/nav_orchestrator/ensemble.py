"""Ensemble — reconcile per-technique results into a single NavResult.

The ensemble is the single point in the pipeline where multiple per-technique
estimates become one offset.  Every step is honest:

1. Drop ``spurious=True`` results.
2. Drop ``at_edge=True`` results unless dropping them would empty the set.
3. Select the consensus subset: every result sponsors the set of results
   that agree with it pairwise (Mahalanobis distance or pixel floor,
   both measured only in the intersection of the two results' observable
   subspaces, so a rank-1 flat-ring measurement and a full-rank absolute
   fix that agree radially can group and fuse); the subset with the
   highest summed confidence wins.
4. Exclude results outside the consensus from the combine.  A lone
   dissenter against a multi-technique consensus is outlier-rejected;
   only a genuine alternative can force the conflicted branch: a
   runner-up consensus with at least two members (absolute
   summed-confidence gap test), or a lone-vs-lone standoff where the
   dissenter's confidence is comparable to the winner's (gap measured
   relative to the winner, so an already-excluded low-confidence
   dissenter retains no veto over a clearly stronger singleton).
   Vote mass, quorum, and the agreement boost count only *corroborating*
   members: a pass-2 result whose prior was seeded by a technique in the
   same subset re-observed its own seed, so it refines the offset but
   casts no independent vote.
5. Resolve the winning subset into mutually independent estimators before
   fusing (``resolve_independent_estimators``): drop a seeded single-star
   refine that only re-observes its own prior, and collapse correlated
   witnesses that share one error source (two ring techniques on one
   catalog; disc and limb on one scattered-light gradient; any two results
   that consumed a common feature, such as two star techniques centroiding
   the same star) to a single representative.  The information-form merge assumes independence, so
   fusing correlated members would over-tighten the covariance, inflate the
   tier, and let a redundant member drag the offset.
6. Combine offsets within the resolved subset via precision-weighted
   (Kalman-style) merging.
7. Apply optional disagreement / conflict penalties.
8. Cap the confidence tier at ``medium`` when the fused covariance is
   rank-deficient (one translation axis unobservable), and at ``low``
   with status reason ``single_star_only`` when the offset rests on one
   matched star -- neither an assumed axis nor an un-cross-checked star
   is the evidence those tiers describe, however tight the localization
   sigma, and a one-star fit has no residual to measure its sigma from.
9. Emit a NavResult carrying the excluded technique names.

The ensemble is tested in isolation against synthetic per-technique results;
correctness here is what makes the rest of the pipeline trustworthy.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.config import IMAGE_LOGGER, logged_section
from spindoctor.feature.constants import (
    AGREEMENT_FACTOR_CAP,
    COMBINED_CONFIDENCE_CAP,
)
from spindoctor.nav_orchestrator.body_witness_veto import (
    BodyWitnessVeto,
    evaluate_body_witness_veto,
)
from spindoctor.nav_orchestrator.ensemble_consensus import (
    consensus_selection,
    corroborating_confidence,
    corroborating_members,
    result_param_vector,
)
from spindoctor.nav_orchestrator.ensemble_independence import (
    STAR_CONSENSUS_TECHNIQUES,
    blend_collapsed_confidence,
    is_single_star_result,
    matched_star_count_of,
    resolve_independent_estimators,
)
from spindoctor.nav_orchestrator.ensemble_observability import (
    mixed_scale_pinvh,
    observable_basis,
    positional_precision,
)
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import ConfidenceRank, NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.nav_technique import technique_tier
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'EnsembleConfig',
    'derive_confidence_rank',
    'ensemble',
]


# Default constants used by the ensemble.  Configurable via ``EnsembleConfig``.
DEFAULT_AGREEMENT_SIGMA = 2.0
DEFAULT_AGREEMENT_PIXEL_FLOOR = 5.0
# Translation distance (px) within which two results that consumed a common
# feature agree.  Measured on the 1030 frames of ISS_006RI_LPHRLFMOV001_PRIME:
# star results sharing a star whose answers both lie within 2 px of the
# published F ring bundle never sit more than 1.27 px apart (1607 pairs, 99th
# percentile 0.79 px), while a pair in which one result misidentified the star
# sits at least 2.25 px apart.  Mirrored in config_540_orchestrator.yaml.
DEFAULT_SHARED_FEATURE_AGREEMENT_PX = 1.5
DEFAULT_AGREEMENT_GAP = 0.5
DEFAULT_DISAGREEMENT_PENALTY = 0.7
DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER = 0.3
DEFAULT_MIN_CONFIDENCE = 0.35
# Euclidean translation distance (px) within which a brightness-centroid blob
# is treated as agreeing with a corroborated star consensus; beyond it the
# blob is a contradicting outlier and is dropped from the ensemble math (a
# multi-star fix outranks a lone centroid as a position witness).  Matches the
# grouping pixel floor: a blob that does not even fall within the grouping
# tolerance of a trusted star fix is not corroborating it.
DEFAULT_BLOB_STAR_DISAGREEMENT_FLOOR_PX = 5.0
DEFAULT_PINVH_RCOND = 1.0e-9
DEFAULT_MAX_ALLOWED_ROTATION_DEG = 5.0
# Body-witness veto thresholds (see body_witness_veto).  The blob centroid's
# lit-hemisphere bias grows toward half phase (and is re-modeled only once the
# coarse acquisition switches to the crescent template past it), so the two
# checks bracket the reliable-witness regime: the shape-lock check trusts the
# blob only below the low phase ceiling, the collapsed-regime check treats it
# as the haze-bias suspect only above the high phase floor.  The disagreement
# tolerance is the larger of an absolute floor and a fraction of the body
# diameter; well-navigated single-body frames hold the blob within ~1.5 px of
# the fused offset, while a disc / limb shape lock disagrees by 5 px and up.
DEFAULT_BODY_WITNESS_SHAPE_LOCK_MAX_PHASE_DEG = 60.0
DEFAULT_BODY_WITNESS_COLLAPSE_MIN_PHASE_DEG = 90.0
DEFAULT_BODY_WITNESS_DISAGREEMENT_FLOOR_PX = 4.0
DEFAULT_BODY_WITNESS_DISAGREEMENT_FRAC = 0.03
# Mahalanobis-distance threshold the blob-vs-consensus disagreement must exceed
# under the combined covariance before the veto fires, so a large-sigma centroid
# whose separation is statistically consistent is not vetoed on scatter alone.
DEFAULT_BODY_WITNESS_AGREEMENT_SIGMA = 2.0
# Background-gradient score at or above which a frame is treated as
# scattered-light: the disc and limb techniques then measure the same veiling
# ramp, so they are collapsed to one witness before the combine (issue #339).
# Matches the cohort-curation scattered-light prescan threshold.
DEFAULT_SCATTERED_LIGHT_GRADIENT_SCORE = 5.0
# Tier confidence boundaries are sim-anchored (fitted 2026-07-18 on the
# recalibrated simulated-scene campaign, with the #210 covariance
# floors live and the ring truth vocabulary in the scene cohort): the
# smallest confidence at which each tier's sigma-gated subset of the
# campaign's fused results achieves a 0.9 success rate against the
# tier's error budget.  The medium tier's
# sigma gate carries the discrimination by itself, so its confidence
# boundary rests at the same 0.35 floor as the low tier and the final
# gate (the tiers are sigma-differentiated).  Mirrored in
# config_540_orchestrator.yaml, which carries the full provenance note.
DEFAULT_TIER_THRESHOLDS: dict[str, dict[str, float | None]] = {
    'high': {'min_confidence': 0.85, 'max_sigma_px': 0.5},
    'medium': {'min_confidence': 0.35, 'max_sigma_px': 2.0},
    'low': {'min_confidence': 0.35, 'max_sigma_px': None},
}


@dataclass(frozen=True)
class EnsembleConfig:
    """Tunable parameters of the ensemble combine.

    Defaults match ``config_540_orchestrator.yaml``.

    Parameters:
        agreement_sigma: Mahalanobis-distance threshold for grouping.
        agreement_pixel_floor: Translation-distance fallback grouping
            threshold in pixels.  Two results are grouped when *either*
            their Mahalanobis distance is at most ``agreement_sigma``
            *or* their Euclidean translation distance is at most this
            many pixels.  The pixel floor exists because per-technique
            covariances are CRLB-tight (FFT subpixel localization for
            ``BodyDiscCorrelateNav``, M-estimator information for the
            DT-fit techniques), well below the actual position
            uncertainty driven by model error and pointing residuals;
            without the floor, results agreeing visually to a few px
            register as hundreds of sigmas apart and never group.
            Set to ``0.0`` to disable the floor.
        shared_feature_agreement_px: Translation distance (pixels) within
            which two results that consumed a common feature agree.  Such a
            pair is judged by this alone, not by ``agreement_sigma`` or
            ``agreement_pixel_floor``: two results that found the same
            detection of a shared feature agree to the centroid precision, and
            two that did not have disagreed about which detection it is.
        agreement_gap: Minimum summed-confidence gap between best and
            runner-up groups before declaring a conflict.  Applied as an
            absolute gap when the runner-up has a quorum (two or more
            members); in a lone-vs-lone standoff (singleton best against
            a singleton excluded dissenter) the gap is measured relative
            to the winner — conflict fires only when
            ``best - runner_up < agreement_gap * best`` — so a dissenter
            with well under ``(1 - agreement_gap)`` of the winner's
            confidence cannot veto it.
        disagreement_penalty: Multiplier on combined confidence when more
            than one group existed.
        conflicted_confidence_multiplier: Additional multiplier when the
            conflicted branch fires.
        min_confidence: Final-result threshold below which the ensemble
            returns NavResult.failed instead of NavResult.ok.
        blob_star_disagreement_floor_px: Euclidean translation distance (px)
            within which a ``BodyBlobNav`` brightness centroid is treated as
            agreeing with a corroborated (non-single-star) star fix.  When a
            corroborated star fix is present and the blob agrees with none
            within this floor, the blob is dropped from the ensemble math as a
            contradicted outlier (a multi-star fix outranks a lone centroid as
            a position witness).  Set to ``0.0`` to disable the drop.
        pinvh_rcond: rcond for ``scipy.linalg.pinvh``.
        max_allowed_rotation_deg: Maximum magnitude (in degrees) a 3-DoF
            result's rotation may take before the ensemble rejects it.
            The rotation parameter is combined as a small angle (circular
            mean of ``(dv, du, theta)`` triples); this bound enforces the
            small-angle assumption that every contributing technique fits
            against (every DT/star technique clamps its rotation to
            ``+-max_rotation_deg``, default 5 deg).  A 3-DoF result
            arriving with ``abs(rotation_rad)`` at or above this bound is a
            programming error upstream and raises ``NavContractError``.
        scattered_light_gradient_score: Background-gradient score at or
            above which the frame is treated as scattered-light, so the disc
            and limb techniques on one body are fused as a single correlated
            witness rather than two independent ones (see
            ``resolve_independent_estimators``).
        tier_thresholds: Mapping ``rank -> {min_confidence, max_sigma_px}``;
            see ``derive_confidence_rank``.
        body_witness_shape_lock_max_phase_deg: Phase ceiling (degrees) below
            which the pose-free brightness centroid is a trustworthy position
            witness for the body shape-lock veto; the lit-hemisphere bias grows
            toward half phase, so the ceiling sits below it.
        body_witness_collapse_min_phase_deg: Phase floor (degrees) above which a
            lone blob is the haze-bias-prone suspect for the collapsed-regime
            veto.
        body_witness_disagreement_floor_px: Absolute floor of the body-witness
            veto's cross-technique disagreement tolerance (a lower bound).
        body_witness_disagreement_frac: Fraction of the body diameter added to
            the floor as the body-witness veto's disagreement lower bound.
        body_witness_agreement_sigma: Mahalanobis-distance threshold the
            disagreement must exceed under the combined covariance before the
            body-witness veto fires.
    """

    agreement_sigma: float = DEFAULT_AGREEMENT_SIGMA
    agreement_pixel_floor: float = DEFAULT_AGREEMENT_PIXEL_FLOOR
    shared_feature_agreement_px: float = DEFAULT_SHARED_FEATURE_AGREEMENT_PX
    agreement_gap: float = DEFAULT_AGREEMENT_GAP
    disagreement_penalty: float = DEFAULT_DISAGREEMENT_PENALTY
    conflicted_confidence_multiplier: float = DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    blob_star_disagreement_floor_px: float = DEFAULT_BLOB_STAR_DISAGREEMENT_FLOOR_PX
    pinvh_rcond: float = DEFAULT_PINVH_RCOND
    max_allowed_rotation_deg: float = DEFAULT_MAX_ALLOWED_ROTATION_DEG
    body_witness_shape_lock_max_phase_deg: float = DEFAULT_BODY_WITNESS_SHAPE_LOCK_MAX_PHASE_DEG
    body_witness_collapse_min_phase_deg: float = DEFAULT_BODY_WITNESS_COLLAPSE_MIN_PHASE_DEG
    body_witness_disagreement_floor_px: float = DEFAULT_BODY_WITNESS_DISAGREEMENT_FLOOR_PX
    body_witness_disagreement_frac: float = DEFAULT_BODY_WITNESS_DISAGREEMENT_FRAC
    body_witness_agreement_sigma: float = DEFAULT_BODY_WITNESS_AGREEMENT_SIGMA
    scattered_light_gradient_score: float = DEFAULT_SCATTERED_LIGHT_GRADIENT_SCORE
    tier_thresholds: dict[str, dict[str, float | None]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_TIER_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        """Reject ensemble thresholds that would misfire their gate.

        Both direct construction and ``from_mapping`` land here, so a
        non-finite value, a negative tolerance, an out-of-range phase, or an
        inverted phase window is rejected the same way regardless of how the
        config was built.  A negative Mahalanobis threshold, for instance,
        would make nearly every separation veto-worthy, and a negative
        ``blob_star_disagreement_floor_px`` would drop every blob.

        Raises:
            ValueError: if any body-witness or blob/star tolerance is
                non-finite or negative, a body-witness phase falls outside
                ``[0, 180]``, or the shape-lock ceiling is not below the
                collapse floor.
        """
        non_negative = {
            'body_witness_disagreement_floor_px': self.body_witness_disagreement_floor_px,
            'body_witness_disagreement_frac': self.body_witness_disagreement_frac,
            'body_witness_agreement_sigma': self.body_witness_agreement_sigma,
            'blob_star_disagreement_floor_px': self.blob_star_disagreement_floor_px,
            'shared_feature_agreement_px': self.shared_feature_agreement_px,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite, got {value}')
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative, got {value}')
        phases = {
            'body_witness_shape_lock_max_phase_deg': self.body_witness_shape_lock_max_phase_deg,
            'body_witness_collapse_min_phase_deg': self.body_witness_collapse_min_phase_deg,
        }
        for name, value in phases.items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite, got {value}')
            if not 0.0 <= value <= 180.0:
                raise ValueError(f'{name} must lie in [0, 180] deg, got {value}')
        if self.body_witness_shape_lock_max_phase_deg >= self.body_witness_collapse_min_phase_deg:
            raise ValueError(
                'body_witness_shape_lock_max_phase_deg '
                f'({self.body_witness_shape_lock_max_phase_deg}) must be below '
                'body_witness_collapse_min_phase_deg '
                f'({self.body_witness_collapse_min_phase_deg})'
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> 'EnsembleConfig':
        """Build an ``EnsembleConfig`` from a YAML-derived mapping.

        Parameters:
            mapping: The ``orchestrator.ensemble`` config section.  ``None``
                or empty returns the code defaults.  Scalar keys override
                their field; a ``tier_thresholds`` block is merged tier-by-
                tier over the defaults, so a partial override (for example
                only ``low.min_confidence``) leaves the other tiers intact.

        Returns:
            An ``EnsembleConfig`` with mapping values overriding the code
            defaults field-by-field.

        Raises:
            ValueError: If the mapping contains an unknown key, an unknown
                tier name, or an unknown tier-threshold key.
        """
        if not mapping:
            return cls()
        data = dict(mapping)
        tiers_raw = data.pop('tier_thresholds', None)
        scalar_fields = {f.name for f in fields(cls)} - {'tier_thresholds'}
        unknown = set(data) - scalar_fields
        if unknown:
            raise ValueError(f'Unknown orchestrator.ensemble config keys: {sorted(unknown)}')
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            try:
                kwargs[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'orchestrator.ensemble.{key} must be a number; got {value!r}'
                ) from exc
        if tiers_raw is not None:
            if not isinstance(tiers_raw, Mapping):
                raise ValueError('orchestrator.ensemble.tier_thresholds must be a mapping')
            tiers = copy.deepcopy(DEFAULT_TIER_THRESHOLDS)
            for tier, spec in tiers_raw.items():
                if tier not in tiers:
                    raise ValueError(
                        f'Unknown tier {tier!r} in tier_thresholds; valid tiers: {sorted(tiers)}'
                    )
                if not isinstance(spec, Mapping):
                    raise ValueError(f'tier_thresholds[{tier!r}] must be a mapping')
                for key, value in spec.items():
                    if key not in ('min_confidence', 'max_sigma_px'):
                        raise ValueError(f'Unknown tier_thresholds[{tier!r}] key {key!r}')
                    try:
                        tiers[tier][key] = None if value is None else float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'orchestrator.ensemble.tier_thresholds[{tier!r}].{key} '
                            f'must be a number or null; got {value!r}'
                        ) from exc
            kwargs['tier_thresholds'] = tiers
        return cls(**kwargs)


#: The brightness-centroid fallback technique down-weighted against a
#: corroborated star consensus.
_BLOB_TECHNIQUE = 'BodyBlobNav'


def _drop_blob_outlier_against_star_consensus(
    results: list[NavTechniqueResult],
    *,
    disagreement_floor_px: float,
) -> list[NavTechniqueResult]:
    """Drop a lone brightness-centroid blob that a star consensus contradicts.

    A ``BodyBlobNav`` result is a pose-free brightness centroid: a weak
    position witness whose lit-hemisphere bias, on a resolved or irregular
    body, can place it several pixels off the true center.  When a
    corroborated star fix is present -- a non-single-star result from a star
    technique (a multi-star pattern match, a two-star unique match, or a
    multi-star refine), whose identification is cross-checked by more than one
    star -- and the blob agrees with no such fix within
    ``disagreement_floor_px``, the blob is the outlier, not the star
    consensus.  It is dropped from the ensemble math so it neither forces a
    conflict against the star fix nor drags the fused offset or incurs the
    multi-group disagreement penalty.  A blob that agrees with a star fix (or
    a frame with no corroborated star fix) is left untouched, and the
    cross-technique body-witness veto still reads every dropped result off the
    full ``results`` list the caller retains, so a geometric body consensus
    the blob contradicts is unaffected.

    Parameters:
        results: The viable (non-spurious) per-technique results.
        disagreement_floor_px: Euclidean translation distance (px) within
            which a blob is treated as agreeing with a star fix.  Set to
            ``0.0`` to disable the drop.

    Returns:
        A new list preserving input ordering with the outlier blob(s)
        removed; the input list is not mutated.
    """
    if disagreement_floor_px <= 0.0:
        return list(results)
    trusted_star_fixes = [
        r
        for r in results
        if r.technique_name in STAR_CONSENSUS_TECHNIQUES and not is_single_star_result(r)
    ]
    if not trusted_star_fixes:
        return list(results)
    kept: list[NavTechniqueResult] = []
    for r in results:
        if r.technique_name == _BLOB_TECHNIQUE and not any(
            _translation_distance_px(r, s) <= disagreement_floor_px for s in trusted_star_fixes
        ):
            IMAGE_LOGGER.info(
                'Dropping blob outlier %s at offset (%.4f, %.4f): it disagrees with a '
                'corroborated star consensus by more than %.2f px and is a weaker '
                'position witness than a multi-star fix',
                r.technique_name,
                r.offset_px[0],
                r.offset_px[1],
                disagreement_floor_px,
            )
            continue
        kept.append(r)
    return kept


def _translation_distance_px(a: NavTechniqueResult, b: NavTechniqueResult) -> float:
    """Return the Euclidean ``(dv, du)`` distance between two results in pixels.

    Parameters:
        a: The first result.
        b: The second result.

    Returns:
        The Euclidean distance between the two results' ``offset_px``, in pixels.
    """
    return math.hypot(a.offset_px[0] - b.offset_px[0], a.offset_px[1] - b.offset_px[1])


def _source_bodies(result: NavTechniqueResult) -> frozenset[str]:
    """Return the set of body names a result was computed against.

    Reads the structured ``NavTechniqueResult.source_bodies`` populated by
    the body-feature techniques from each consumed feature's
    ``NavFeature.body_name``.  Ring and star techniques leave it empty, so
    they are naturally excluded from the body-only fallback-supersession
    filter.  This replaces the earlier ``feature_ids`` string-parsing, which
    silently broke if the feature-id format changed.
    """
    return result.source_bodies


def _drop_superseded_fallbacks(
    results: list[NavTechniqueResult],
) -> list[NavTechniqueResult]:
    """Drop fallback-tier results superseded by a non-spurious primary.

    A fallback result is superseded when at least one of its source
    bodies appears in the source-body set of any non-spurious primary
    result.  Fallback results whose source body has no primary
    coverage stay in the set so a scene with only a fallback (e.g.,
    a body too small for limb fitting, where only BodyBlob ran) still
    produces an offset.

    Returns a new list preserving input ordering; the input list is
    not mutated.
    """
    primary_success_bodies: set[str] = set()
    for r in results:
        if r.spurious:
            continue
        if technique_tier(r.technique_name) != 'primary':
            continue
        primary_success_bodies.update(_source_bodies(r))
    if not primary_success_bodies:
        return list(results)
    kept: list[NavTechniqueResult] = []
    for r in results:
        if technique_tier(r.technique_name) == 'fallback':
            superseded = bool(_source_bodies(r) & primary_success_bodies)
            if superseded:
                IMAGE_LOGGER.info(
                    'Dropping fallback %s for body %s: superseded by a '
                    'non-spurious primary result on the same body',
                    r.technique_name,
                    ', '.join(sorted(_source_bodies(r))) or '(unknown)',
                )
                continue
        kept.append(r)
    return kept


@dataclass(frozen=True)
class _CombinedEstimate:
    """Output of :func:`_combine_precision_weighted`.

    Parameters:
        offset_px: Combined ``(dv, du)`` translation.
        rotation_rad: Combined rotation in radians; ``None`` when every
            input was 2-DoF.
        covariance_px2: Combined covariance, ``(2, 2)`` or ``(3, 3)``
            matching the input parameter dimensionality.
        is_rank_deficient: True when the *translation block* of the
            combined covariance has an unobservable direction (a null or
            sentinel axis).  An unobservable rotation alone — every
            contributing technique reporting the rotation sentinel — does
            not make the translation fix rank-deficient.
    """

    offset_px: tuple[float, float]
    rotation_rad: float | None
    covariance_px2: NDArrayFloatType
    is_rank_deficient: bool


def _combine_precision_weighted(
    group: list[NavTechniqueResult], *, rcond: float, max_allowed_rotation_deg: float
) -> _CombinedEstimate:
    """Information-form combine of a group of agreeing results.

    Implements the Kalman-style information-form merge for the translation
    components:

        Sigma_combined = pinvh( sum_i pinvh(Sigma_i) )
        mu_combined    = Sigma_combined @ sum_i ( pinvh(Sigma_i) @ mu_i )

    Parameter vectors carry rotation as a third component when every input
    is 3-DoF; the resulting combined estimate then carries a non-``None``
    ``rotation_rad`` field.  The rotation parameter is *not* averaged as a
    plain Euclidean coordinate (which would wrap incorrectly near
    ``+-pi``); instead it is combined on the circle as the precision-weighted
    circular mean ``atan2(sum_i w_i sin theta_i, sum_i w_i cos theta_i)``,
    with ``w_i`` the rotation-component information ``pinvh(Sigma_i)[2, 2]``.
    The translation components and the full combined covariance are produced
    exactly as before by the information-form merge.

    Parameters:
        group: Non-empty list of agreeing results.  Every member must
            share the same parameter dimensionality (all 2-DoF or all
            3-DoF — :func:`_agreement_groups` already enforces this).
        rcond: rcond for ``pinvh``.
        max_allowed_rotation_deg: Small-angle bound; each 3-DoF input's
            ``rotation_rad`` magnitude must stay strictly below this many
            degrees (in radians) for the circular-mean combine to be valid.

    Returns:
        :class:`_CombinedEstimate`.

    Raises:
        ValueError: if ``group`` is empty (defensive; the orchestrator must
            ensure non-emptiness before calling), if total weight is zero,
            or if the group contains a mix of 2-DoF and 3-DoF results.
        NavContractError: if a 3-DoF input's rotation magnitude is at or
            above the small-angle bound (an upstream programming error).
    """
    if not group:
        raise ValueError('empty group passed to _combine_precision_weighted')
    info_sum: NDArrayFloatType | None = None
    info_mu_sum: NDArrayFloatType | None = None
    n_params: int | None = None
    max_rotation_rad = math.radians(max_allowed_rotation_deg)
    rot_w_sin = 0.0
    rot_w_cos = 0.0
    for res in group:
        cov = np.asarray(res.covariance_px2, np.float64)
        if n_params is None:
            n_params = cov.shape[0]
        elif cov.shape[0] != n_params:
            raise ValueError(
                f'mixed-DoF group passed to _combine_precision_weighted: '
                f'expected {n_params}-DoF, got {cov.shape[0]} from {res.technique_name}'
            )
        info = mixed_scale_pinvh(cov, rcond=rcond)
        mu = result_param_vector(res)
        if cov.shape[0] == 3:
            theta = float(mu[2])
            # Small-angle assumption: every contributing technique clamps its
            # rotation fit to +-max_rotation_deg, so a result arriving outside
            # that bound is an upstream programming error, not data the
            # circular-mean combine should silently absorb.  Raised as
            # NavContractError (not a bare assert) so the check survives
            # python -O and is never swallowed as an ordinary technique
            # failure.
            if abs(theta) >= max_rotation_rad:
                raise NavContractError(
                    f'{res.technique_name}: rotation {math.degrees(theta):.3f} deg '
                    f'violates small-angle bound +-{max_allowed_rotation_deg:.3f} deg'
                )
            # Rotation weight: the scalar rotation-information element
            # ``info[2, 2]`` only.  The circular mean needs one scalar weight
            # per angle, so the rotation-translation cross-information
            # ``info[:2, 2]`` (which the information-form translation merge
            # below does retain) cannot enter this average.  Dropping it
            # approximates each input's rotation as independent of its
            # translation, i.e. it uses the conditional rotation information
            # instead of the marginal ``1 / Sigma[2, 2]``; under the
            # small-angle clamp enforced above the rotations differ by at
            # most a few degrees, so the reweighting error this introduces
            # in the combined angle is far below the reported rotation
            # sigma.  The combined covariance itself is exact: it comes from
            # the full information-form merge, cross-terms included.
            w_theta = float(info[2, 2])
            rot_w_sin += w_theta * math.sin(theta)
            rot_w_cos += w_theta * math.cos(theta)
        if info_sum is None:
            info_sum = info.copy()
            info_mu_sum = info @ mu
        else:
            info_sum = info_sum + info
            assert info_mu_sum is not None  # narrowed for mypy
            info_mu_sum = info_mu_sum + info @ mu
    assert info_sum is not None
    assert info_mu_sum is not None
    cov_combined = mixed_scale_pinvh(info_sum, rcond=rcond)
    mu_combined = cov_combined @ info_mu_sum
    # Rank-deficiency is a property of the translation fix: the combined
    # covariance's (v, u) block has an unobservable direction (exact null
    # or sentinel-sized variance).  Judging the full parameter matrix
    # instead would let an unobservable *rotation* (every input carrying
    # the rotation sentinel) mislabel a fully-constrained 2-D offset as
    # rank-1.
    trans_basis = observable_basis(np.ascontiguousarray(cov_combined[:2, :2]))
    is_rank_deficient = trans_basis.shape[1] < 2
    rotation: float | None = None
    if n_params == 3:
        # Combine the rotation on the circle so angles near +-pi do not cancel
        # to a spurious ~0; fall back to the information-form estimate when the
        # rotation weight is degenerate (every input rotation-unobservable).
        if rot_w_sin == 0.0 and rot_w_cos == 0.0:
            rotation = float(mu_combined[2])
        else:
            rotation = float(math.atan2(rot_w_sin, rot_w_cos))
    return _CombinedEstimate(
        offset_px=(float(mu_combined[0]), float(mu_combined[1])),
        rotation_rad=rotation,
        covariance_px2=cov_combined,
        is_rank_deficient=is_rank_deficient,
    )


def _combine_confidence(
    group: list[NavTechniqueResult],
    *,
    rcond: float,
    disagreement_penalty: float,
    apply_disagreement_penalty: bool,
) -> float:
    """Precision-weighted combine of per-result confidences.

    Weights are ``trace(pinvh(Sigma_i[:2, :2]))`` -- the positional (v, u)
    precision in ``px^-2``, with any rotation axis marginalised out so the
    weight is not skewed by the unrelated ``rad^-2`` rotation precision of a
    3-DoF result.  Tighter covariances contribute more to the combined
    confidence than loose ones.  The boosted combined
    confidence reflects the number of significant contributors, capped per
    ``AGREEMENT_FACTOR_CAP`` and ``COMBINED_CONFIDENCE_CAP``.

    Parameters:
        group: Non-empty list of agreeing results.
        rcond: rcond for ``pinvh``.
        disagreement_penalty: Multiplier applied if other groups existed.
        apply_disagreement_penalty: True if more than one group existed
            before this combine.

    Returns:
        Combined confidence in ``[0, 1]``; never above
        ``COMBINED_CONFIDENCE_CAP``.

    Raises:
        ValueError: if every input covariance shares a null direction
            (W = 0); the orchestrator's caller routes this to
            ``unobservable_offset`` failure.
    """
    # Weight by the *positional* precision only.  Taking trace(pinvh(cov)) over
    # a full 3-DoF covariance would add the v, u precisions (px^-2) to the
    # rotation precision (rad^-2), so a star-field result whose rotation is
    # tightly pinned would dominate the confidence average on an arbitrary,
    # unit-mixed scale.  Marginalising rotation out and tracing the 2x2
    # translation block keeps every weight in px^-2; the additive trace (rather
    # than det(info)^(1/p)) stays well-defined for the rank-1 ring-edge
    # covariances whose unobservable axis carries zero precision.  The 2-DoF
    # path is unchanged: cov[:2, :2] is then the whole matrix.
    weights = [positional_precision(res.covariance_px2, rcond=rcond) for res in group]
    w_total = sum(weights)
    if w_total <= 0.0:
        raise ValueError('precision-weighted combine: total weight is zero; offset is unobservable')
    weighted_avg = sum(w * r.confidence for w, r in zip(weights, group, strict=True))
    weighted_avg /= w_total
    significant_threshold = 0.1 * max(weights)
    # The agreement boost reflects independent corroboration, so a
    # prior-descendant whose seeding technique is in the group does not
    # count toward it (issue #222) — its agreement re-observes its own seed.  Its
    # precision still participates in the weighted average.
    corroborating = set(corroborating_members(group))
    n_significant = sum(
        1
        for w, r in zip(weights, group, strict=True)
        if w > significant_threshold and r in corroborating
    )
    if n_significant <= 1:
        agreement_factor = 1.0
    else:
        agreement_factor = 1.0 + 0.5 * math.log2(n_significant)
    agreement_factor = min(agreement_factor, AGREEMENT_FACTOR_CAP)
    combined = min(weighted_avg * agreement_factor, COMBINED_CONFIDENCE_CAP)
    if apply_disagreement_penalty:
        combined *= disagreement_penalty
    return combined


def derive_confidence_rank(
    *,
    confidence: float,
    sigma_px: tuple[float, float] | None,
    tier_thresholds: dict[str, dict[str, float | None]] | None = None,
) -> ConfidenceRank:
    """Derive the five-bucket confidence rank from confidence + sigma.

    ``max_sigma_px`` compares ``max(sigma_dv, sigma_du)`` only.
    ``high`` / ``medium`` / ``low`` tiers require both confidence and sigma
    constraints; ``conflicted`` and ``failed`` are status-driven and not
    chosen here.

    Parameters:
        confidence: Combined confidence in ``[0, 1]``.
        sigma_px: Per-axis 1sigma marginal uncertainty (use ``None`` to mean
            "unknown / not applicable").
        tier_thresholds: Mapping ``rank -> {min_confidence, max_sigma_px}``
            with ``max_sigma_px`` allowed to be ``None``.

    Returns:
        ``'high'``, ``'medium'``, or ``'low'`` if any tier matches; else
        ``'failed'``.
    """
    thresholds = tier_thresholds or DEFAULT_TIER_THRESHOLDS
    max_sigma = max(sigma_px) if sigma_px is not None else None
    if max_sigma is None:
        # No covariance was supplied, so every sigma-constrained tier
        # (high / medium) is unreachable and the result can only earn the
        # best sigma-free tier (low).  Surface this: a missing covariance is
        # almost always an upstream technique failing to populate it, not a
        # legitimately unconstrained fit, and would otherwise cap the rank
        # silently.
        IMAGE_LOGGER.warning(
            'derive_confidence_rank: sigma_px is None (no covariance); '
            'sigma-constrained tiers are unreachable and the rank caps at the '
            'best sigma-free tier'
        )
    ranks: tuple[ConfidenceRank, ...] = ('high', 'medium', 'low')
    for rank in ranks:
        spec = thresholds[rank]
        min_conf = spec['min_confidence']
        max_allowed = spec['max_sigma_px']
        assert min_conf is not None  # min_confidence is always set
        if confidence < min_conf:
            continue
        if max_allowed is not None and (max_sigma is None or max_sigma > max_allowed):
            continue
        return rank
    return 'failed'


@logged_section('ensemble', 'ENSEMBLE')
def ensemble(
    results: list[NavTechniqueResult],
    *,
    feature_inventory: list[NavFeatureSummary],
    image_classifier: NavImageClassifierResult,
    provenance: Provenance,
    config: EnsembleConfig | None = None,
    model_metadata: dict[str, dict[str, Any]] | None = None,
    annotations: Annotations | None = None,
) -> NavResult:
    """Reconcile per-technique results into a single NavResult.

    Parameters:
        results: Per-technique results from one or both passes.
        feature_inventory: Feature inventory (kept + gated entries).
        image_classifier: Image-quality classifier verdict.
        provenance: Reproducibility envelope.
        config: Optional ``EnsembleConfig`` overrides.
        model_metadata: Optional per-NavModel diagnostic dict map.
        annotations: Optional pre-built annotation collection from the
            orchestrator's ``_collect_annotations`` pass.

    Returns:
        A single NavResult — ok / conflicted / failed.
    """
    cfg = config or EnsembleConfig()
    md = model_metadata if model_metadata is not None else {}
    ann = annotations if annotations is not None else Annotations()
    if not results:
        return NavResult.failed(
            status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
            image_classifier=image_classifier,
            provenance=provenance,
            model_metadata=md,
            annotations=ann,
        )
    viable = [r for r in results if not r.spurious]
    if not viable:
        IMAGE_LOGGER.info(
            'All %d technique result(s) returned spurious=True: %s',
            len(results),
            ', '.join(r.technique_name for r in results),
        )
        return NavResult.failed(
            status_reason=NavStatusReason.ALL_TECHNIQUES_SPURIOUS,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    # Drop fallback-tier results superseded by a non-spurious primary
    # for the same body (e.g., BodyTerminatorNav / BodyBlobNav when
    # BodyLimbNav or BodyDiscCorrelateNav succeeded on the same body).
    # The full ``results`` list is preserved on the NavResult for
    # diagnostics; only the ensemble math sees the filtered set.
    viable = _drop_superseded_fallbacks(viable)
    # Drop a lone brightness-centroid blob that a corroborated star consensus
    # contradicts: it is a weaker position witness than a multi-star fix, so it
    # must not force a conflict against the star solution or incur the
    # multi-group disagreement penalty.  The body-witness veto below still
    # reads the full ``results`` list, so a geometric body consensus the blob
    # contradicts is unaffected.
    viable = _drop_blob_outlier_against_star_consensus(
        viable,
        disagreement_floor_px=cfg.blob_star_disagreement_floor_px,
    )
    interior = [r for r in viable if not r.at_edge]
    if interior:
        viable = interior
    selection = consensus_selection(
        viable,
        agreement_sigma=cfg.agreement_sigma,
        agreement_pixel_floor=cfg.agreement_pixel_floor,
        shared_feature_agreement_px=cfg.shared_feature_agreement_px,
        rcond=cfg.pinvh_rcond,
        max_allowed_rotation_deg=cfg.max_allowed_rotation_deg,
    )
    best_group = selection.best
    excluded = selection.excluded
    excluded_names = [r.technique_name for r in excluded]
    consensus_names = [r.technique_name for r in best_group]
    # Resolve the winning group into mutually independent estimators before
    # any fusion: drop a seeded single-star refine that only re-observes its
    # own prior (issue #222), and collapse correlated witnesses that share one
    # error source -- two ring techniques reading one catalog (issue #317),
    # disc and limb on one scattered-light gradient (issue #339), or two results
    # that consumed a common feature (issue #623) -- to a single
    # representative.  The information-form combine assumes independence, so
    # fusing correlated members would over-tighten the covariance and inflate
    # the tier, and a redundant member would drag the fused offset.
    resolution = resolve_independent_estimators(
        best_group,
        image_classifier=image_classifier,
        scattered_light_gradient_score=cfg.scattered_light_gradient_score,
        rcond=cfg.pinvh_rcond,
    )
    combine_set = blend_collapsed_confidence(
        resolution.estimators, resolution.collapsed_groups, rcond=cfg.pinvh_rcond
    )
    for dropped in resolution.dropped_descendants:
        IMAGE_LOGGER.info(
            'Dropping seeded single-star refine %s from the combine: it '
            're-observes its own pass-1 prior and adds no independent '
            'constraint',
            dropped.technique_name,
        )
    for collapsed in resolution.collapsed_groups:
        IMAGE_LOGGER.info(
            'Collapsing correlated witnesses to %s: %s share one error '
            'source and are fused as a single witness, not independent ones',
            collapsed[0].technique_name,
            ', '.join(r.technique_name for r in collapsed),
        )
    # Vote mass and quorum count independent witnesses only, so they are
    # measured over the resolved ``combine_set`` -- the same set the merge
    # fuses -- not the raw consensus.  This keeps the conflict logic consistent
    # with the merge: a collapsed correlated pair (two ring techniques, or
    # scattered-light disc/limb) counts as the one witness it is, so it cannot
    # claim a spurious quorum that would suppress a genuine lone-vs-lone
    # conflict.  ``corroborating_members`` additionally drops any surviving
    # multi-star prior-descendant, which refines the offset but casts no
    # independent vote (issue #222).
    best_summed_conf = corroborating_confidence(combine_set)
    best_has_quorum = len(corroborating_members(combine_set)) >= 2
    apply_disagreement_penalty = bool(excluded)
    try:
        combined = _combine_precision_weighted(
            combine_set,
            rcond=cfg.pinvh_rcond,
            max_allowed_rotation_deg=cfg.max_allowed_rotation_deg,
        )
        combined_confidence = _combine_confidence(
            combine_set,
            rcond=cfg.pinvh_rcond,
            disagreement_penalty=cfg.disagreement_penalty,
            apply_disagreement_penalty=apply_disagreement_penalty,
        )
    except ValueError:
        # Total weight zero — offset unobservable in every contributing
        # input.
        IMAGE_LOGGER.info(
            'Combined precision-weighted offset is unobservable: every input '
            'covariance shares one null direction (%d input(s))',
            len(best_group),
        )
        return NavResult.failed(
            status_reason=NavStatusReason.UNOBSERVABLE_OFFSET,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    # Conflict check.  Excluded results force the conflicted branch only when
    # they constitute a genuine alternative.  A runner-up consensus with at
    # least two members contests via the absolute summed-confidence gap.  In
    # a lone-vs-lone standoff (the winning subset has no quorum either) the
    # gap is measured *relative to the winner*: the consensus selection has
    # already excluded the dissenter as an outlier, so it retains veto power
    # only while its confidence is comparable to the winner's -- a dissenter
    # at half the winner's confidence is outlier-rejected, not a conflict
    # (issue #258).  A single dissenter against a multi-technique consensus
    # is always outlier-rejected -- reported via excluded_from_consensus and
    # the disagreement penalty, not as a conflict.
    if excluded:
        gap = best_summed_conf - selection.runner_up_confidence
        if selection.runner_up_has_quorum:
            is_conflict = gap < cfg.agreement_gap
        elif not best_has_quorum:
            is_conflict = gap < cfg.agreement_gap * best_summed_conf
        else:
            is_conflict = False
        if is_conflict:
            conflicted_confidence = combined_confidence * cfg.conflicted_confidence_multiplier
            effective_gap_threshold = (
                cfg.agreement_gap
                if selection.runner_up_has_quorum
                else cfg.agreement_gap * best_summed_conf
            )
            IMAGE_LOGGER.info(
                'Conflicted: best-vs-runner-up summed-confidence gap %.3f is '
                'below the effective agreement_gap threshold %.3f '
                '(best %.3f, runner-up %.3f, runner-up quorum: %s); '
                'conflicted confidence = %.3f '
                '(combined %.3f x conflicted_multiplier %.3f)',
                gap,
                effective_gap_threshold,
                best_summed_conf,
                selection.runner_up_confidence,
                selection.runner_up_has_quorum,
                conflicted_confidence,
                combined_confidence,
                cfg.conflicted_confidence_multiplier,
            )
            return NavResult.conflicted(
                offset_px=combined.offset_px,
                covariance_px2=combined.covariance_px2,
                confidence=conflicted_confidence,
                per_technique=results,
                feature_inventory=feature_inventory,
                image_classifier=image_classifier,
                provenance=provenance,
                excluded_from_consensus=excluded_names,
                consensus_techniques=consensus_names,
                model_metadata=md,
                annotations=ann,
                matched_star_count=matched_star_count_of(best_group),
            )
        IMAGE_LOGGER.info(
            'Excluded %d result(s) from consensus as outlier(s): %s '
            '(consensus %d member(s) summing %.3f; excluded runner-up %.3f, '
            'quorum: %s)',
            len(excluded),
            ', '.join(excluded_names),
            len(best_group),
            best_summed_conf,
            selection.runner_up_confidence,
            selection.runner_up_has_quorum,
        )
    if combined_confidence < cfg.min_confidence:
        IMAGE_LOGGER.info(
            'Combined confidence %.3f is below the min_confidence threshold %.3f',
            combined_confidence,
            cfg.min_confidence,
        )
        return NavResult.failed(
            status_reason=NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    # Cross-technique body-witness veto: catch confident-wrong body locks the
    # per-technique diagnostics cannot see -- a geometric consensus that agreed
    # at a shape-mismatched offset the pose-free blob contradicts, or a lone
    # blob carrying the answer in a regime where the geometric technique on the
    # same body self-flagged spurious.  A trusted (non-spurious, non-single-star)
    # star fix that agrees with a geometric consensus offset within the blob/star
    # corroboration floor corroborates the geometry and suppresses the shape-lock
    # verdict: the blob, not the lock, is then the outlier (the geometric-side
    # mirror of the blob-drop above, sharing its blob_star_disagreement_floor_px
    # knob).
    veto = evaluate_body_witness_veto(
        best_group,
        results,
        combined.offset_px,
        combined.covariance_px2,
        shape_lock_max_phase_deg=cfg.body_witness_shape_lock_max_phase_deg,
        collapse_min_phase_deg=cfg.body_witness_collapse_min_phase_deg,
        disagreement_floor_px=cfg.body_witness_disagreement_floor_px,
        disagreement_frac=cfg.body_witness_disagreement_frac,
        agreement_sigma=cfg.body_witness_agreement_sigma,
        star_agreement_floor_px=cfg.blob_star_disagreement_floor_px,
        rcond=cfg.pinvh_rcond,
    )
    if veto is BodyWitnessVeto.LONE_BLOB_COLLAPSED_REGIME:
        IMAGE_LOGGER.info(
            'Body-witness veto: the lone surviving body offset is the brightness '
            'centroid while a geometric technique on the same body self-flagged '
            'spurious; declining the frame (systematic photometric bias)'
        )
        return NavResult.failed(
            status_reason=NavStatusReason.LONE_BLOB_IN_COLLAPSED_REGIME,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    if veto is BodyWitnessVeto.SHAPE_LOCK_SUSPECT:
        conflicted_confidence = combined_confidence * cfg.conflicted_confidence_multiplier
        IMAGE_LOGGER.info(
            'Body-witness veto: the geometric body consensus (%s) disagrees with '
            'the pose-free blob witness by more than the shape-lock tolerance; '
            'reporting conflicted (confidence %.3f)',
            ', '.join(consensus_names),
            conflicted_confidence,
        )
        return NavResult.conflicted(
            offset_px=combined.offset_px,
            covariance_px2=combined.covariance_px2,
            confidence=conflicted_confidence,
            per_technique=results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            excluded_from_consensus=excluded_names,
            consensus_techniques=consensus_names,
            status_reason=NavStatusReason.BODY_SHAPE_LOCK_SUSPECT,
            model_metadata=md,
            annotations=ann,
            matched_star_count=matched_star_count_of(best_group),
        )
    cov = combined.covariance_px2
    sigma_dv = float(math.sqrt(max(cov[0, 0], 0.0)))
    sigma_du = float(math.sqrt(max(cov[1, 1], 0.0)))
    sigma_along_unobservable_px = float('inf') if combined.is_rank_deficient else None
    rank = derive_confidence_rank(
        confidence=combined_confidence,
        sigma_px=(sigma_dv, sigma_du),
        tier_thresholds=cfg.tier_thresholds,
    )
    if rank == 'high' and combined.is_rank_deficient:
        # A fused result with an unobservable translation axis reports one
        # axis as an assumption, not a measurement.  However precise the
        # observable axis is, that is not a 'high'-tier absolute fix
        # (issue #221);
        # ``sigma_along_unobservable_px`` carries the same verdict to the
        # metadata.
        IMAGE_LOGGER.info(
            'Tier capped at medium: the fused covariance is rank-deficient '
            '(one translation axis unobservable)'
        )
        rank = 'medium'
    matched_star_count = matched_star_count_of(best_group)
    # An offset resting on one matched star has no residual: the star's two
    # coordinates fix the two offset parameters exactly.  Its sigma is the
    # centroid precision the feature predicted, not a measured scatter, and a
    # misidentified star looks exactly like a correct one (issues #263, #622,
    # #623).  Such a result can earn no tier whose gate reads sigma, so it
    # tops out at low and says why.  Judged over the independent estimators,
    # so a dropped seeded single-star refine does not mark an otherwise
    # multi-star consensus.
    single_star_only = matched_star_count == 1 and all(
        is_single_star_result(r) for r in combine_set
    )
    if single_star_only and rank in ('high', 'medium'):
        IMAGE_LOGGER.info(
            'Tier capped at low: the offset rests on one matched star (%s), whose '
            'sigma is predicted rather than measured',
            ', '.join(r.technique_name for r in combine_set),
        )
        rank = 'low'
    elif rank == 'high' and all(is_single_star_result(r) for r in combine_set):
        # Single-star results on different stars do cross-check each other, but each
        # rests on one detection whose sigma is predicted; together they top out at
        # medium (issue #263).
        IMAGE_LOGGER.info(
            'Tier capped at medium: every independent estimator is a single-star solution (%s)',
            ', '.join(r.technique_name for r in combine_set),
        )
        rank = 'medium'
    elif (
        rank == 'high'
        and matched_star_count <= 2
        and all(r.technique_name in STAR_CONSENSUS_TECHNIQUES for r in combine_set)
    ):
        # Two stars are the least that checks itself: their offsets agree or they
        # do not, and nothing checks the pair.  A unique-match pair caps itself at
        # medium for that reason, and a star result that consumed the same two
        # stars in another technique is no second witness, however sharp its own
        # sigma and confidence.  Only a third star, or a technique of another
        # kind, earns high.
        IMAGE_LOGGER.info(
            'Tier capped at medium: the star estimators rest on %d matched star(s) (%s)',
            matched_star_count,
            ', '.join(r.technique_name for r in combine_set),
        )
        rank = 'medium'
    if rank == 'failed':
        # Confidence + sigma combination doesn't earn any tier.  Distinguish the
        # two causes: if the combined confidence cleared the *lowest* tier's
        # ``min_confidence`` yet still earned no tier, the offset was confident
        # but too imprecise (sigma exceeded every tier's ``max_sigma_px``);
        # otherwise the confidence itself was below threshold.
        lowest_min_conf = min(
            float(spec['min_confidence'] or 0.0) for spec in cfg.tier_thresholds.values()
        )
        if combined_confidence >= lowest_min_conf:
            failed_reason = NavStatusReason.FINAL_SIGMA_ABOVE_THRESHOLD
        else:
            failed_reason = NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD
        IMAGE_LOGGER.info(
            'No tier earned (%s): combined confidence %.3f, sigma (dv, du) = '
            '(%.3f, %.3f) px (max %.3f); tier thresholds = %s',
            failed_reason.value,
            combined_confidence,
            sigma_dv,
            sigma_du,
            max(sigma_dv, sigma_du),
            cfg.tier_thresholds,
        )
        return NavResult.failed(
            status_reason=failed_reason,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    if combined.is_rank_deficient:
        status_reason = NavStatusReason.RANK_1_ONLY
    elif single_star_only:
        status_reason = NavStatusReason.SINGLE_STAR_ONLY
    else:
        status_reason = NavStatusReason.OK
    sigma_rotation_rad: float | None = None
    if combined.rotation_rad is not None and cov.shape == (3, 3):
        sigma_rotation_rad = float(math.sqrt(max(cov[2, 2], 0.0)))
    return NavResult.success(
        offset_px=combined.offset_px,
        covariance_px2=combined.covariance_px2,
        confidence=combined_confidence,
        confidence_rank=rank,
        status_reason=status_reason,
        per_technique=results,
        feature_inventory=feature_inventory,
        image_classifier=image_classifier,
        provenance=provenance,
        sigma_along_unobservable_px=sigma_along_unobservable_px,
        excluded_from_consensus=excluded_names,
        consensus_techniques=consensus_names,
        model_metadata=md,
        annotations=ann,
        rotation_rad=combined.rotation_rad,
        sigma_rotation_rad=sigma_rotation_rad,
        matched_star_count=matched_star_count,
    )
