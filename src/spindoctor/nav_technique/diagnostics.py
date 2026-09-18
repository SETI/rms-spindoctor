"""Per-technique diagnostics dataclasses.

Each NavTechnique returns a typed diagnostics object on its
``NavTechniqueResult.diagnostics`` field.  The curator walks
``CURATOR_FIELDS`` on each diagnostics class to decide which fields land in
the JSON metadata; an unmapped field is a CI failure.
"""

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    'BodyBlobDiagnostics',
    'BodyDiscDiagnostics',
    'BodyLimbDiagnostics',
    'BodyTerminatorDiagnostics',
    'ManualNavDiagnostics',
    'NavTechniqueDiagnostics',
    'RingAnnulusDiagnostics',
    'RingEdgeDiagnostics',
    'StarFieldDiagnostics',
    'StarRefineDiagnostics',
    'StarUniqueMatchDiagnostics',
    'TitanHazeDiagnostics',
]


@dataclass(frozen=True)
class BodyDiscDiagnostics:
    """Diagnostics emitted by ``BodyDiscCorrelateNav``.

    Parameters:
        ncc_peak: Peak normalized cross-correlation value.
        peak_to_runner_up_ratio: Ratio of NCC peak to second-highest peak
            outside the exclusion radius around the peak.
        consistency_px: Inter-pyramid peak migration in pixels (raw).
        consistency_ratio: ``consistency_px`` divided by the per-image
            spurious threshold (which scales with body diameter).  A
            value <= 1.0 means the result is within the diameter-scaled
            consistency budget; the confidence formula consumes this
            normalized form so a healthy fit on a large body is not
            penalized by the same divisor that's appropriate for a
            small body.
        used_gradient: True if gradient mode was selected by ``auto``.
        body_count: Number of BODY_DISC features fused into the combined
            template.
    """

    ncc_peak: float = 0.0
    peak_to_runner_up_ratio: float = 0.0
    consistency_px: float = 0.0
    consistency_ratio: float = 0.0
    used_gradient: bool = False
    body_count: int = 0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'ncc_peak': 'ncc_peak',
        'peak_to_runner_up_ratio': 'peak_to_runner_up_ratio',
        'consistency_px': 'consistency_px',
        'consistency_ratio': 'consistency_ratio',
        'used_gradient': 'used_gradient',
        'body_count': 'body_count',
    }


@dataclass(frozen=True)
class BodyLimbDiagnostics:
    """Diagnostics emitted by ``BodyLimbNav``.

    Parameters:
        visible_limb_arc_fraction: Fused visible-arc fraction across input
            LIMB_ARC features.
        visible_arc_px: Total surviving polyline arc length in pixels.
        dt_fit_rms_px: Final root-mean-square DT residual.
        lm_iterations: Levenberg-Marquardt iteration count.
        tukey_inlier_count: Number of polyline vertices accepted by the
            Tukey biweight robust estimator.
        lm_converged: True when the LM met its step-norm tolerance before
            the iteration cap; False marks an unverified fit whose
            confidence the shared DT gates cap.
        polarity_rejection_fraction: Fraction of model vertices whose local
            gradient direction disagreed with the model normal at the seed.
        coarse_peak_fraction: The winning coarse shift's acquisition-quality
            score.  For the polarity techniques (``BodyLimbNav``,
            ``BodyTerminatorNav``) this is the polarity-weighted per-vertex
            match fraction from
            :func:`~spindoctor.nav_technique.dt_fitting.coarse_polarity_search_scored`
            -- each in-bounds vertex on a detected edge contributes
            ``max(0, cos theta)`` between its model normal and the local
            image gradient, averaged over the in-bounds vertices.  It is a
            mean over vertices, not a per-vertex count, so it is not directly
            comparable to ``tukey_inlier_count``.
    """

    visible_limb_arc_fraction: float = 0.0
    visible_arc_px: float = 0.0
    dt_fit_rms_px: float = 0.0
    lm_iterations: int = 0
    tukey_inlier_count: int = 0
    lm_converged: bool = True
    polarity_rejection_fraction: float = 0.0
    coarse_peak_fraction: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'visible_limb_arc_fraction': 'visible_limb_arc_fraction',
        'visible_arc_px': 'visible_arc_px',
        'dt_fit_rms_px': 'dt_fit_rms_px',
        'lm_iterations': 'lm_iterations',
        'tukey_inlier_count': 'tukey_inlier_count',
        'lm_converged': 'lm_converged',
        'polarity_rejection_fraction': 'polarity_rejection_fraction',
        'coarse_peak_fraction': 'coarse_peak_fraction',
    }


@dataclass(frozen=True)
class BodyTerminatorDiagnostics:
    """Diagnostics emitted by ``BodyTerminatorNav``.

    Parameters: same shape as ``BodyLimbDiagnostics`` with
    ``visible_terminator_arc_fraction`` substituted, plus the
    terminator-specific confidence inputs and the basin second-opinion
    pair:

        mean_phase_angle_factor: Vertex-weighted mean ``sin(phase)``
            factor across the consumed terminator features.  A confidence
            term (a terminator sharpens with phase), so it is recorded
            per result for the calibration fit.
        mean_albedo_penalty: Vertex-weighted mean catalog albedo-variation
            penalty across the consumed features; likewise a recorded
            confidence input.
        secondary_basin_distance_px: Distance from the converged offset to
            the best competing DT-cost basin in the search window; ``None``
            when the scan did not run or found no eligible shift (``0.0``
            is a legitimate measured value, so it is not the sentinel).
        secondary_basin_cost_ratio: Competing basin's mean per-vertex DT
            cost divided by the converged cost (epsilon-guarded); values
            below the technique's ``basin_cost_ratio_threshold`` mark the
            fit spurious.  ``None`` when not measured.
        lm_converged: Shared DT gate diagnostic; the field carries the
            same meaning as on ``BodyLimbDiagnostics``.
        polarity_rejection_fraction: Shared DT gate diagnostic; the field
            carries the same meaning as on ``BodyLimbDiagnostics``.
        coarse_peak_fraction: Shared DT gate diagnostic; the field carries
            the same meaning as on ``BodyLimbDiagnostics``.
    """

    visible_terminator_arc_fraction: float = 0.0
    visible_arc_px: float = 0.0
    dt_fit_rms_px: float = 0.0
    lm_iterations: int = 0
    tukey_inlier_count: int = 0
    mean_phase_angle_factor: float = 0.0
    mean_albedo_penalty: float = 0.0
    secondary_basin_distance_px: float | None = None
    secondary_basin_cost_ratio: float | None = None
    lm_converged: bool = True
    polarity_rejection_fraction: float = 0.0
    coarse_peak_fraction: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'visible_terminator_arc_fraction': 'visible_terminator_arc_fraction',
        'visible_arc_px': 'visible_arc_px',
        'dt_fit_rms_px': 'dt_fit_rms_px',
        'lm_iterations': 'lm_iterations',
        'tukey_inlier_count': 'tukey_inlier_count',
        'mean_phase_angle_factor': 'mean_phase_angle_factor',
        'mean_albedo_penalty': 'mean_albedo_penalty',
        'secondary_basin_distance_px': 'secondary_basin_distance_px',
        'secondary_basin_cost_ratio': 'secondary_basin_cost_ratio',
        'lm_converged': 'lm_converged',
        'polarity_rejection_fraction': 'polarity_rejection_fraction',
        'coarse_peak_fraction': 'coarse_peak_fraction',
    }


@dataclass(frozen=True)
class BodyBlobDiagnostics:
    """Diagnostics emitted by ``BodyBlobNav``.

    Parameters:
        body_snr_inside_predicted_bbox: SNR within the predicted bbox.
        body_extent_px: Predicted body's longer-axis extent in pixels.
        blob_count: Number of BODY_BLOB features fused.
        residual_px: Centroid-fit RMS residual.
        max_phase_angle_deg: Maximum raw phase angle across the
            consumed blobs.  Recorded for diagnostic inspection only;
            the confidence formula consumes
            ``max_phase_irregularity_factor`` instead because raw phase
            understates the centroid uncertainty for an irregular
            body.
        max_phase_irregularity_factor: Maximum
            ``(ellipsoid_rms_residual_km / body_radius_km) *
            (1 + 2 * sin(phase/2)**2)`` across the consumed blobs.  The
            confidence formula uses this term to down-weight irregular
            high-phase scenes where the lit-weighted predicted centroid
            cannot fully correct for the unknown-orientation shadowing
            on a non-ellipsoidal body.
    """

    body_snr_inside_predicted_bbox: float = 0.0
    body_extent_px: float = 0.0
    blob_count: int = 0
    residual_px: float = 0.0
    max_phase_angle_deg: float = 0.0
    max_phase_irregularity_factor: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'body_snr_inside_predicted_bbox': 'body_snr_inside_predicted_bbox',
        'body_extent_px': 'body_extent_px',
        'blob_count': 'blob_count',
        'residual_px': 'residual_px',
        'max_phase_angle_deg': 'max_phase_angle_deg',
        'max_phase_irregularity_factor': 'max_phase_irregularity_factor',
    }


@dataclass(frozen=True)
class RingEdgeDiagnostics:
    """Diagnostics emitted by ``RingEdgeNav``.

    Parameters:
        total_edge_length_px: Cumulative pixel length of all surviving
            ring-edge polylines.
        per_edge_dt_rms_summed: Sum of per-edge final DT RMS values.
        per_edge_dt_rms_mean: Mean per-edge final DT RMS value
            (``per_edge_dt_rms_summed / edge_count``).  Edge-count independent,
            so it -- not the raw sum -- is the scale the confidence formula
            uses; the sum grows with the number of fused edges and a fixed
            divisor cannot normalize it.
        per_edge_dt_median_max: Largest per-edge median absolute DT residual
            (px).  The mis-convergence gate statistic: a wholly misaligned
            edge (the wrong-ringlet failure mode) drives its own median to
            the ringlet spacing, while a minority of vertices snapping to a
            neighboring parallel edge (routine on flat multi-edge scenes)
            leaves every median near the fit residual.
        edge_count: Number of RING_EDGE features fused.
        is_rank_1: True if every ring-edge feature was straight-line and the
            combined covariance is rank-1.
        lm_converged: Shared DT gate diagnostic; the field carries the
            same meaning as on ``BodyLimbDiagnostics``.
        coarse_peak_fraction: Shared DT gate diagnostic.  Unlike the polarity
            techniques, ``RingEdgeNav`` runs polarity-blind (ring-edge
            polarity is not predictable), so this is the plain mask-overlap
            match fraction from
            :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search_scored`
            rather than the polarity-weighted score described on
            ``BodyLimbDiagnostics``.
        sigma_orbit_radial_px: Effective fully-correlated radial
            orbit-uncertainty sigma (px) added in quadrature to the
            reported covariance along the fit's radial direction (the
            weight-weighted combine of the consumed features'
            ``sigma_orbit_radial_px``); ``0.0`` when no consumed feature
            declares an orbit uncertainty.
    """

    total_edge_length_px: float = 0.0
    per_edge_dt_rms_summed: float = 0.0
    per_edge_dt_rms_mean: float = 0.0
    per_edge_dt_median_max: float = 0.0
    edge_count: int = 0
    is_rank_1: bool = False
    lm_converged: bool = True
    coarse_peak_fraction: float = 0.0
    sigma_orbit_radial_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'total_edge_length_px': 'total_edge_length_px',
        'per_edge_dt_rms_summed': 'per_edge_dt_rms_summed',
        'per_edge_dt_rms_mean': 'per_edge_dt_rms_mean',
        'per_edge_dt_median_max': 'per_edge_dt_median_max',
        'edge_count': 'edge_count',
        'is_rank_1': 'is_rank_1',
        'lm_converged': 'lm_converged',
        'coarse_peak_fraction': 'coarse_peak_fraction',
        'sigma_orbit_radial_px': 'sigma_orbit_radial_px',
    }


@dataclass(frozen=True)
class RingAnnulusDiagnostics:
    """Diagnostics emitted by ``RingAnnulusNav``.

    Parameters:
        ncc_peak: Peak NCC value.
        peak_to_runner_up_ratio: NCC peak ratio.
        annulus_count: Number of RING_ANNULUS features (one per planet).
        used_gradient: True if gradient mode was selected.
        sigma_orbit_radial_px: Effective fully-correlated radial
            orbit-uncertainty sigma (px) added to the reported covariance
            through the translation the NCC absorbs such a displacement into
            (the normal-count-weighted combine of the consumed annuli's
            ``sigma_orbit_radial_px``); ``0.0`` when no consumed annulus
            declares an orbit uncertainty.
    """

    ncc_peak: float = 0.0
    peak_to_runner_up_ratio: float = 0.0
    annulus_count: int = 0
    used_gradient: bool = False
    sigma_orbit_radial_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'ncc_peak': 'ncc_peak',
        'peak_to_runner_up_ratio': 'peak_to_runner_up_ratio',
        'annulus_count': 'annulus_count',
        'used_gradient': 'used_gradient',
        'sigma_orbit_radial_px': 'sigma_orbit_radial_px',
    }


@dataclass(frozen=True)
class StarFieldDiagnostics:
    """Diagnostics emitted by ``StarFieldFromCatalogNav``.

    Parameters:
        n_inliers: Number of detection-to-catalog inliers after RANSAC.
        median_residual_px: Median position residual on inliers.
        n_detected_sources: Number of bright sources detected in the image.
        n_catalog_predicted: Number of catalog stars in the extfov.
        n_triplets_evaluated: Number of triplet candidates considered by
            RANSAC.
        rotation_below_separability_floor: True when a co-fitted camera
            rotation's magnitude fell below the roll/translation
            separability floor (``rotation_separability_floor_deg``) and
            the rotation was therefore reported as unobservable rather
            than as a confident near-zero value.
        wide_offset_lock: True when the offset was recovered by the
            wide-offset asterism fallback (a few trusted bright-anchor
            seeds) rather than the strong-tier triplet RANSAC.
        wide_offset_false_lock_expectation: Expected number of chance
            locks of this inlier count under the uniform-detection null;
            the false-lock significance the acceptance budget bounded.
            ``0.0`` for strong-tier matches (the fallback did not run).
    """

    n_inliers: int = 0
    median_residual_px: float = 0.0
    n_detected_sources: int = 0
    n_catalog_predicted: int = 0
    n_triplets_evaluated: int = 0
    rotation_below_separability_floor: bool = False
    wide_offset_lock: bool = False
    wide_offset_false_lock_expectation: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'n_inliers': 'n_inliers',
        'median_residual_px': 'median_residual_px',
        'n_detected_sources': 'n_detected_sources',
        'n_catalog_predicted': 'n_catalog_predicted',
        'n_triplets_evaluated': 'n_triplets_evaluated',
        'rotation_below_separability_floor': 'rotation_below_separability_floor',
        'wide_offset_lock': 'wide_offset_lock',
        'wide_offset_false_lock_expectation': 'wide_offset_false_lock_expectation',
    }


@dataclass(frozen=True)
class StarUniqueMatchDiagnostics:
    """Diagnostics emitted by ``StarUniqueMatchNav``.

    Parameters:
        mode: ``'one_star'`` or ``'two_star'``.
        predicted_snr: Predicted SNR of the brightest catalog star.
        brightness_margin_mag: Mag difference to the next-brightest *unmatched*
            catalog source predictable in extfov; ``+inf`` when no unmatched
            star exists (a 1-star scene with no other predictable star, or a
            2-star scene with no third predictable star to compare against).
        residual_px: Detection-vs-prediction residual.
        detection_peak_ratio: Background-subtracted peak-to-runner-up ratio of
            the 1-star path's detection inside its search window (``inf`` when
            the runner-up does not clear the background; 0.0 on the 2-star
            path, which cross-checks via its assignment residual instead).
    """

    mode: str = ''
    predicted_snr: float = 0.0
    brightness_margin_mag: float = 0.0
    residual_px: float = 0.0
    detection_peak_ratio: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'mode': 'mode',
        'predicted_snr': 'predicted_snr',
        'brightness_margin_mag': 'brightness_margin_mag',
        'residual_px': 'residual_px',
        'detection_peak_ratio': 'detection_peak_ratio',
    }


@dataclass(frozen=True)
class StarRefineDiagnostics:
    """Diagnostics emitted by ``StarRefineNav``.

    Parameters:
        n_stars_used: Number of stars that survived per-star quality gates.
        median_pos_err_px: Median refinement positional error.
        residual_scatter_px: Per-axis RMS scatter of the per-star residuals.
    """

    n_stars_used: int = 0
    median_pos_err_px: float = 0.0
    residual_scatter_px: float = 0.0
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'n_stars_used': 'n_stars_used',
        'median_pos_err_px': 'median_pos_err_px',
        'residual_scatter_px': 'residual_scatter_px',
    }


@dataclass(frozen=True)
class ManualNavDiagnostics:
    """Diagnostics emitted by ``NavTechniqueManual``.

    Parameters:
        operator_accepted: ``True`` when the operator confirmed the
            dialog's chosen offset.  Always ``True`` on results that
            reach the curator (canceled picks short-circuit before the
            ``NavResult`` is built), but kept explicit so the JSON
            metadata records the fact that a human, not an autonomous
            technique, set the offset.
    """

    operator_accepted: bool = True
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'operator_accepted': 'operator_accepted',
    }


@dataclass(frozen=True)
class TitanHazeDiagnostics:
    """Diagnostics emitted by ``TitanHazeNav``.

    Parameters:
        sun_angle_deg: Symmetry-axis angle used by the final pass, in
            degrees, measured as ``atan2(dv, du)`` from the predicted disc
            center toward the sub-solar side.
        axis_degenerate: True when the model could not localize the
            sub-solar direction (near-zero phase, or unevaluable
            geometry), so the axis angle is nominal and angle refinement
            was skipped.
        phase_deg: Phase angle at the body center, in degrees.
        envelope_diameter_px: Apparent haze-envelope diameter in pixels.
        cross_track_px: Final-pass offset component perpendicular to the
            symmetry axis, positive along ``c_hat``.
        along_track_px: Offset component along the symmetry axis summed
            over every pass, positive toward the sub-solar side.
        symmetry_peak_score: Pearson mirror-correlation score at the
            winning cross-track shift, or ``None`` when no candidate shift
            had enough usable signal to correlate at all (reported as
            absent rather than as a zero score, which reads as a measured
            total lack of symmetry rather than as no measurement).
        symmetry_valid_fraction: Fraction of annulus mirror pairs that
            were usable at that shift.
        symmetry_second_peak_ratio: Normalized height of the strongest
            competing correlation peak; ``0.0`` when there is none.
        theta_refined_deg: Angle the refinement moved the symmetry axis
            by, relative to the model's SPICE-derived angle, in degrees;
            ``0.0`` when refinement did not win or was skipped.
        arc_rays_total: Rays presented to the limb circle fit.
        arc_rays_inlier: Rays carrying a non-zero final robust weight.
        arc_inlier_fraction: ``arc_rays_inlier / arc_rays_total``;
            ``0.0`` when no ray survived profile extraction.
        arc_residual_rms_px: Root-mean-square radial residual over the
            inlier rays, or ``None`` when the robust fit rejected every
            ray (reported as absent rather than as a perfect zero, which
            a falling confidence sigmoid would read as maximally good).
        fitted_haze_radius_km: Fitted arc radius converted to kilometers
            at the body's image scale.  Recorded so a haze-radius table
            per instrument, filter, and phase bin can be accumulated from
            production output.
        filters: Instrument filter names for the image.
        recentered: True when the large-along-track-shift second pass
            ran.
        gate_failed: Name of the first failed fit gate, or ``None`` when
            every gate passed.
    """

    sun_angle_deg: float = 0.0
    axis_degenerate: bool = False
    phase_deg: float = 0.0
    envelope_diameter_px: float = 0.0
    cross_track_px: float = 0.0
    along_track_px: float = 0.0
    symmetry_peak_score: float | None = None
    symmetry_valid_fraction: float = 0.0
    symmetry_second_peak_ratio: float = 0.0
    theta_refined_deg: float = 0.0
    arc_rays_total: int = 0
    arc_rays_inlier: int = 0
    arc_inlier_fraction: float = 0.0
    arc_residual_rms_px: float | None = None
    fitted_haze_radius_km: float = 0.0
    filters: tuple[str, ...] = ()
    recentered: bool = False
    gate_failed: str | None = None
    CURATOR_FIELDS: ClassVar[dict[str, str | None]] = {
        'sun_angle_deg': 'sun_angle_deg',
        'axis_degenerate': 'axis_degenerate',
        'phase_deg': 'phase_deg',
        'envelope_diameter_px': 'envelope_diameter_px',
        'cross_track_px': 'cross_track_px',
        'along_track_px': 'along_track_px',
        'symmetry_peak_score': 'symmetry_peak_score',
        'symmetry_valid_fraction': 'symmetry_valid_fraction',
        'symmetry_second_peak_ratio': 'symmetry_second_peak_ratio',
        'theta_refined_deg': 'theta_refined_deg',
        'arc_rays_total': 'arc_rays_total',
        'arc_rays_inlier': 'arc_rays_inlier',
        'arc_inlier_fraction': 'arc_inlier_fraction',
        'arc_residual_rms_px': 'arc_residual_rms_px',
        'fitted_haze_radius_km': 'fitted_haze_radius_km',
        'filters': 'filters',
        'recentered': 'recentered',
        'gate_failed': 'gate_failed',
    }


NavTechniqueDiagnostics = (
    BodyDiscDiagnostics
    | BodyLimbDiagnostics
    | BodyTerminatorDiagnostics
    | BodyBlobDiagnostics
    | ManualNavDiagnostics
    | RingEdgeDiagnostics
    | RingAnnulusDiagnostics
    | StarFieldDiagnostics
    | StarUniqueMatchDiagnostics
    | StarRefineDiagnostics
    | TitanHazeDiagnostics
)
"""Sum type spanning every per-technique diagnostics dataclass.

The orchestrator's curator and the technique-result type both consume
this union; adding a new technique means adding both its diagnostics
dataclass above and a new entry into this union.
"""
