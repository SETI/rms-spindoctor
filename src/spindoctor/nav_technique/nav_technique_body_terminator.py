"""``BodyTerminatorNav`` — translation fit from body terminator polylines.

Mirrors :class:`BodyLimbNav` with three terminator-specific differences:

1. The accepted feature type is ``TERMINATOR_ARC`` instead of ``LIMB_ARC``.
2. Per-body uniform weighting: every vertex of a given body shares one
   inverse-variance weight derived from the body's mean
   ``sigma_normal_per_vertex_px``.  Cross-body weighting reflects albedo
   variation (low-albedo bodies provide tighter terminators than
   high-albedo ones).
3. The confidence spec includes additional terms for the per-body
   visible-terminator-arc fraction and the phase-angle factor flag,
   capturing the design's albedo / phase-geometry sensitivity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, body_names_from_features
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import TerminatorArcFlags
from spindoctor.feature.geometry import TerminatorPolyline
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import BodyTerminatorDiagnostics
from spindoctor.nav_technique.dt_fit_gates import evaluate_dt_fit_gates
from spindoctor.nav_technique.dt_fitting import (
    _rotate_vertices,
    coarse_polarity_search_scored,
    find_secondary_dt_minimum,
    lm_subpixel_refine,
)
from spindoctor.nav_technique.dt_tuning import DTTuning
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    add_model_error_floor,
    log_confidence_breakdown,
    rotation_pivot_distance_px,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.schema import SCHEMA_CONFIG, NonNeg
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyTerminatorNav', 'BodyTerminatorTuning']


def _aggregate_terminator_features(
    features: list[NavFeature],
) -> tuple[
    NDArrayFloatType,
    NDArrayFloatType,
    NDArrayFloatType,
    list[str],
    list[float],
    list[float],
]:
    """Concatenate per-body vertices and uniform-per-body sigmas.

    The design dictates that every vertex of a given body shares
    ``1 / sigma_normal_per_vertex_px**2`` derived from the body's mean
    sigma — that captures cross-body albedo variation while smoothing
    out per-vertex sigma noise.  The ``polarity_normals`` are the
    geometric outward normal negated, identical to the limb pipeline,
    because the terminator's image gradient also points from the dark
    side toward the lit side.

    Returns:
        ``(vertices, polarity_normals, sigmas_uniform_per_body,
        feature_ids, phase_angle_factors, albedo_penalties)``.
    """
    vert_chunks: list[NDArrayFloatType] = []
    normal_chunks: list[NDArrayFloatType] = []
    sigma_chunks: list[NDArrayFloatType] = []
    ids: list[str] = []
    phase_factors: list[float] = []
    albedo_penalties: list[float] = []
    for feat in features:
        if not isinstance(feat.geometry, TerminatorPolyline):
            continue
        if feat.geometry.vertices_vu.shape[0] == 0:
            continue
        vertices = feat.geometry.vertices_vu.astype(np.float64)
        outward = feat.geometry.normals_vu.astype(np.float64)
        per_body_sigma = float(feat.geometry.sigma_normal_per_vertex_px.astype(np.float64).mean())
        per_body_sigma_arr = np.full(vertices.shape[0], per_body_sigma, dtype=np.float64)
        vert_chunks.append(vertices)
        normal_chunks.append(-outward)
        sigma_chunks.append(per_body_sigma_arr)
        ids.append(feat.feature_id)
        flags = feat.flags
        if isinstance(flags, TerminatorArcFlags):
            phase_factors.append(float(flags.phase_angle_factor))
        else:
            phase_factors.append(1.0)
        albedo_penalty = feat.reliability_reasons.albedo_penalty
        albedo_penalties.append(float(albedo_penalty) if albedo_penalty is not None else 0.0)
    empty_2 = np.empty((0, 2), np.float64)
    empty_1 = np.empty(0, np.float64)
    vertices_out = np.concatenate(vert_chunks, axis=0) if vert_chunks else empty_2
    normals_out = np.concatenate(normal_chunks, axis=0) if normal_chunks else empty_2
    sigmas_out = np.concatenate(sigma_chunks, axis=0) if sigma_chunks else empty_1
    return vertices_out, normals_out, sigmas_out, ids, phase_factors, albedo_penalties


@dataclass(frozen=True)
class BodyTerminatorTuning(DTTuning):
    """The tuning keys of :class:`BodyTerminatorNav`, beyond those every DT technique holds.

    Parameters:
        min_arc_vertices: The fewest surviving polyline vertices a terminator arc needs
            to take part in the fit.
        model_error_floor_px: The model-error floor in pixels, added in quadrature to
            the translation covariance.
        basin_cost_ratio_threshold: A competing DT-cost basin whose mean per-vertex cost
            is below this many times the converged cost marks the fit spurious; 0
            disables the test.
        basin_exclude_radius_px: How far from the converged offset, in pixels, a
            competing basin must lie.
        basin_cost_epsilon_px: The least converged cost, in pixels, the comparison
            divides by, so a near-zero cost does not make every basin a rival.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    min_arc_vertices: NonNeg
    model_error_floor_px: NonNeg
    basin_cost_ratio_threshold: NonNeg
    basin_exclude_radius_px: NonNeg
    basin_cost_epsilon_px: NonNeg


class BodyTerminatorNav(NavTechnique):
    """Body-terminator DT-based translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({TERMINATOR_ARC})``.
        requires_prior: ``False`` (the technique runs in pass 1).
        tier: ``'fallback'``.

    The ``'fallback'`` tier reflects that the terminator is a photometric
    feature modulated by phase, albedo, and local topography; its failure
    modes (DT-fit locking onto crater shadows or other local minima) have no
    per-technique signal that admits them.  When a non-spurious primary fit
    (limb or disc) is available for the same body the ensemble drops the
    terminator result rather than risk letting a clean-looking but
    mis-converged fit override the geometric techniques.
    """

    name = 'BodyTerminatorNav'
    Tuning = BodyTerminatorTuning
    tuning: ClassVar[BodyTerminatorTuning]
    accepts_feature_types = frozenset({NavFeatureType.TERMINATOR_ARC})
    requires_prior = False
    tier = 'fallback'
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'visible_terminator_arc_fraction',
            'visible_arc_px',
            'dt_fit_rms_px',
            'lm_iterations',
            'tukey_inlier_count',
            'mean_phase_angle_factor',
            'mean_albedo_penalty',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        """Load the configuration, which sets the technique's tuning.

        Parameters:
            config: Optional ``Config`` override; ``None`` uses
                ``DEFAULT_CONFIG``.
        """
        super().__init__(config=config)
        self.config.read_config()

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries a usable terminator arc.

        Reads only the polyline vertex count per feature.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one TERMINATOR_ARC has at least the configured ``min_arc_vertices``
            surviving vertices.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, TerminatorPolyline)
            and f.geometry.vertices_vu.shape[0] >= self.tuning.min_arc_vertices
        ]
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_terminator_arc_features_with_sufficient_visible_arc',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input terminator polylines.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Polylines with fewer than the configured
                ``min_arc_vertices`` vertices are dropped before fitting.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``
                — plus ``fit_camera_rotation`` and ``max_rotation_deg``.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence, and a populated
            :class:`BodyTerminatorDiagnostics`.  Per
            :class:`BodyLimbNav.navigate`:

            - When ``context.fit_camera_rotation`` is False the result
              carries a ``(2, 2)`` covariance and ``rotation_rad`` /
              ``sigma_rotation_rad`` are ``None`` (an unexpected
              non-(2, 2) covariance from LM is logged at WARNING and
              truncated).
            - When ``context.fit_camera_rotation`` is True the result
              carries a ``(3, 3)`` covariance with the LM-fit rotation
              diagonal and populated ``rotation_rad`` /
              ``sigma_rotation_rad``.  An unexpected covariance shape
              from LM raises ``RuntimeError``.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            if context.image_edge_dt_ext is None or context.image_gradient_vu_ext is None:
                raise RuntimeError(
                    'BodyTerminatorNav requires NavContext.image_edge_dt_ext and '
                    'NavContext.image_gradient_vu_ext to be populated by the orchestrator'
                )
            eligible_features = [
                f
                for f in features
                if isinstance(f.geometry, TerminatorPolyline)
                and f.geometry.vertices_vu.shape[0] >= self.tuning.min_arc_vertices
            ]
            self.logger.info(
                'Consuming %d TERMINATOR_ARC features (out of %d offered)',
                len(eligible_features),
                len(features),
            )
            (
                vertices,
                polarity_normals,
                sigmas,
                feature_ids,
                phase_factors,
                albedo_penalties,
            ) = _aggregate_terminator_features(eligible_features)
            if vertices.shape[0] == 0:
                raise RuntimeError(
                    'BodyTerminatorNav.navigate received zero usable TERMINATOR_ARC vertices '
                    'despite is_feasible reporting feasibility; aborting fit'
                )
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            margin_v, margin_u = search_window_for_obs(context)
            self.logger.debug(
                'Aggregated %d terminator vertices, sigma_normal range [%.3f, %.3f] px, '
                'search window (v, u) = (%d, %d) px',
                int(vertices.shape[0]),
                float(sigmas.min()) if sigmas.size else 0.0,
                float(sigmas.max()) if sigmas.size else 0.0,
                margin_v,
                margin_u,
            )
            # Polarity-weighted coarse search: the terminator runs
            # use_polarity=True, so the model normals carry a trustworthy
            # expected gradient direction.  Weighting the mask overlap by it
            # stops the lit disc's limb, rings, or a second body from
            # out-scoring the true terminator arc and mis-seeding LM.
            coarse = coarse_polarity_search_scored(
                edge_mask,
                gradient_vu,
                vertices,
                polarity_normals,
                (margin_v, margin_u),
            )
            coarse_dv, coarse_du = coarse.offset_vu
            self.logger.debug(
                'Coarse polarity offset: (%d, %d), peak match fraction %.3f',
                coarse_dv,
                coarse_du,
                coarse.score,
            )
            fit_rotation = bool(context.fit_camera_rotation)
            pivot_vu = (float(vertices[:, 0].mean()), float(vertices[:, 1].mean()))
            pivot_distance = (
                rotation_pivot_distance_px(pivot_vu, edge_dt.shape[:2]) if fit_rotation else 0.0
            )
            result = lm_subpixel_refine(
                vertices_vu=vertices,
                normals_vu=polarity_normals,
                sigma_normal_per_vertex_px=sigmas,
                image_edge_dt=edge_dt,
                image_gradient_vu=gradient_vu,
                initial_offset_vu=(float(coarse_dv), float(coarse_du)),
                use_polarity=True,
                fit_rotation=fit_rotation,
                pivot_vu=pivot_vu if fit_rotation else None,
                pivot_distance_px=pivot_distance,
                trust_region_px=self.tuning.lm_trust_region_px,
                tikhonov_alpha=self.tuning.lm_tikhonov_alpha,
            )
            dv_final, du_final = result.offset_vu
            max_rotation_rad = math.radians(context.max_rotation_deg)
            rotation_at_edge = fit_rotation and (
                abs(result.rotation_rad) >= self.tuning.rotation_at_edge_fraction * max_rotation_rad
            )
            covariance = result.covariance
            rotation_rad: float | None
            sigma_rotation_rad: float | None
            if fit_rotation:
                if covariance.shape != (3, 3):
                    raise RuntimeError(
                        f'BodyTerminatorNav expected 3x3 covariance with fit_rotation; '
                        f'got {covariance.shape}'
                    )
                rotation_rad = float(result.rotation_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(covariance[2, 2]), 0.0)))
            else:
                if covariance.shape != (2, 2):
                    self.logger.warning(
                        'BodyTerminatorNav: lm_subpixel_refine returned %s covariance with '
                        'fit_rotation=False; truncating to (2, 2)',
                        covariance.shape,
                    )
                    covariance = covariance[:2, :2]
                rotation_rad = None
                sigma_rotation_rad = None
            # See BodyLimbNav: the check uses ``>=`` so an LM that walked
            # past the search window registers as at-edge instead of
            # silently producing an out-of-extfov offset.
            at_edge = (
                abs(dv_final) >= margin_v - self.tuning.at_edge_tolerance_px
                or abs(du_final) >= margin_u - self.tuning.at_edge_tolerance_px
                or rotation_at_edge
            )
            sigma_min_px = float(sigmas.min()) if sigmas.size else 1.0
            n_vertices = int(vertices.shape[0])
            inlier_fraction = (
                float(result.inlier_count) / float(n_vertices) if n_vertices > 0 else 0.0
            )
            lm_displacement_px = float(
                math.hypot(dv_final - float(coarse_dv), du_final - float(coarse_du))
            )
            # The inlier-fraction and LM-displacement guards catch the
            # same mis-convergence signatures ``BodyLimbNav`` guards
            # against: an LM that locks onto crater shadows, surface
            # boundaries, or stray high-contrast features will retain
            # a small minority of vertices as inliers, *or* will walk
            # multiple pixels away from the integer coarse-NCC seed.
            # See ``BodyLimbNav.spurious`` for the reference
            # implementation.
            # ``result.rms_px`` is the *Tukey-weighted* residual RMS; when
            # the LM converges to a local minimum where one arc fits
            # cleanly and another is wholly mis-aligned, Tukey rejects the
            # bad-arc vertices and ``rms_px`` collapses to near zero, so
            # the weighted ``rms_px > floor`` test cannot detect the
            # mis-convergence.  ``result.raw_rms_px`` is the *unweighted*
            # RMS over all vertices, so it retains those outliers; gate on
            # it with the same DT residual threshold the weighted check
            # uses (mirrors ``RingEdgeNav``).
            dt_rms_threshold = max(
                self.tuning.spurious_dt_floor_px,
                self.tuning.spurious_dt_rms_factor * sigma_min_px,
            )
            gate_verdict = evaluate_dt_fit_gates(
                result,
                self.tuning,
                coarse_peak_fraction=coarse.score,
                total_vertex_count=n_vertices,
                use_polarity=True,
            )
            if gate_verdict.spurious_reasons:
                self.logger.info(
                    'DT fit-quality gate(s) fired: %s (polarity rejection %.3f, '
                    'coarse peak %.3f, lm_converged=%s)',
                    ', '.join(gate_verdict.spurious_reasons),
                    gate_verdict.polarity_rejection_fraction,
                    gate_verdict.coarse_peak_fraction,
                    gate_verdict.lm_converged,
                )
            spurious = (
                result.degenerate
                or result.rms_px > dt_rms_threshold
                or result.raw_rms_px > dt_rms_threshold
                or result.inlier_count < self.tuning.spurious_min_inliers
                or inlier_fraction < self.tuning.spurious_min_inlier_fraction
                or lm_displacement_px > self.tuning.spurious_max_lm_displacement_px
                or gate_verdict.spurious
            )
            # Basin second-opinion (#125): a textured surface can offer the
            # DT fit a rival alignment (crater shadow, albedo boundary) that
            # every technique-level residual metric scores as clean.  Scan
            # the search window for a competing basin; when its cost rivals
            # the converged cost the fit is not unimodal and the result is
            # spurious.  Skipped when already spurious or disabled.
            secondary_basin_distance_px: float | None = None
            secondary_basin_cost_ratio: float | None = None
            if not spurious and self.tuning.basin_cost_ratio_threshold > 0.0:
                # The LM evaluates the DT at the *rotated* vertex positions
                # when it fits rotation; the basin scan must score the same
                # geometry, or the converged cost is inflated and a good
                # rotated fit reads as non-unimodal.
                basin_vertices = vertices
                if fit_rotation and result.rotation_rad != 0.0:
                    basin_vertices = _rotate_vertices(
                        vertices, pivot_vu, float(result.rotation_rad)
                    )
                basin = find_secondary_dt_minimum(
                    edge_dt,
                    basin_vertices,
                    converged_offset_vu=(dv_final, du_final),
                    search_window_vu=(margin_v, margin_u),
                    exclude_radius_px=self.tuning.basin_exclude_radius_px,
                )
                if basin is not None:
                    denominator = max(basin.converged_cost_px, self.tuning.basin_cost_epsilon_px)
                    secondary_basin_cost_ratio = basin.cost_px / denominator
                    secondary_basin_distance_px = basin.distance_px
                    if secondary_basin_cost_ratio < self.tuning.basin_cost_ratio_threshold:
                        spurious = True
                        self.logger.info(
                            'Competing DT basin at (%d, %d) px (distance %.1f px) has cost '
                            '%.3f px vs converged %.3f px (ratio %.3f < %.3f): fit is not '
                            'unimodal, marking spurious',
                            basin.offset_vu[0],
                            basin.offset_vu[1],
                            basin.distance_px,
                            basin.cost_px,
                            basin.converged_cost_px,
                            secondary_basin_cost_ratio,
                            self.tuning.basin_cost_ratio_threshold,
                        )
            visible_terminator_arc_fraction = _aggregate_visible_arc_fraction(eligible_features)
            mean_phase = float(np.mean(phase_factors)) if phase_factors else 0.0
            mean_albedo = float(np.mean(albedo_penalties)) if albedo_penalties else 0.0
            diagnostics = BodyTerminatorDiagnostics(
                visible_terminator_arc_fraction=visible_terminator_arc_fraction,
                visible_arc_px=float(vertices.shape[0]),
                dt_fit_rms_px=float(result.rms_px),
                lm_iterations=int(result.iterations),
                tukey_inlier_count=int(result.inlier_count),
                mean_phase_angle_factor=mean_phase,
                mean_albedo_penalty=mean_albedo,
                secondary_basin_distance_px=secondary_basin_distance_px,
                secondary_basin_cost_ratio=secondary_basin_cost_ratio,
                lm_converged=gate_verdict.lm_converged,
                polarity_rejection_fraction=gate_verdict.polarity_rejection_fraction,
                coarse_peak_fraction=gate_verdict.coarse_peak_fraction,
            )
            confidence_context = _TerminatorConfidenceContext(
                at_edge=at_edge,
                spurious=bool(spurious),
                diagnostics=diagnostics,
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                confidence_context,
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            if gate_verdict.confidence_cap is not None and confidence > gate_verdict.confidence_cap:
                self.logger.info(
                    'LM exited at the iteration cap without converging; capping '
                    'confidence %.4f -> %.4f',
                    float(confidence),
                    gate_verdict.confidence_cap,
                )
                confidence = gate_verdict.confidence_cap
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, RMS %.4f px, inliers %d / %d, '
                'confidence %.4f',
                dv_final,
                du_final,
                result.rms_px,
                result.inlier_count,
                int(vertices.shape[0]),
                float(confidence),
            )
            if fit_rotation and sigma_rotation_rad is not None and rotation_rad is not None:
                self.logger.info(
                    'Rotation = %+.4f deg (sigma %.4f deg)%s',
                    math.degrees(rotation_rad),
                    math.degrees(sigma_rotation_rad),
                    ', AT_EDGE' if rotation_at_edge else '',
                )
            if spurious or at_edge:
                self.logger.info('Diagnostic flags: spurious=%s, at_edge=%s', spurious, at_edge)
            self.logger.debug(
                'LM iterations = %d, sigma_min = %.3f px, '
                'visible_terminator_arc_fraction = %.3f, mean_phase_factor = %.3f, '
                'mean_albedo_penalty = %.3f',
                result.iterations,
                sigma_min_px,
                visible_terminator_arc_fraction,
                mean_phase,
                mean_albedo,
            )
            # Model-error floor (#210); rationale on load_model_error_floor.
            covariance = add_model_error_floor(covariance, self.tuning.model_error_floor_px)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(feature_ids),
                offset_px=(float(dv_final), float(du_final)),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=bool(spurious),
                at_edge=bool(at_edge),
                diagnostics=diagnostics,
                rotation_rad=rotation_rad,
                sigma_rotation_rad=sigma_rotation_rad,
                source_bodies=body_names_from_features(features),
            )


class _TerminatorConfidenceContext:
    """Adapter exposing terminator confidence terms in a single attribute set."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: BodyTerminatorDiagnostics,
    ) -> None:
        """Flatten the diagnostics plus the result-level flags into one set.

        Parameters:
            at_edge: Whether the converged pose reached the search-window
                boundary; it lives on the result, not the diagnostics.
            spurious: The technique's spurious verdict, consumed by the
                confidence spec's hard-zero rule.
            diagnostics: The technique's populated diagnostics.
        """
        self.at_edge = at_edge
        self.spurious = spurious
        self.visible_terminator_arc_fraction = diagnostics.visible_terminator_arc_fraction
        self.visible_arc_px = diagnostics.visible_arc_px
        self.dt_fit_rms_px = diagnostics.dt_fit_rms_px
        self.lm_iterations = diagnostics.lm_iterations
        self.tukey_inlier_count = diagnostics.tukey_inlier_count
        self.mean_phase_angle_factor = diagnostics.mean_phase_angle_factor
        self.mean_albedo_penalty = diagnostics.mean_albedo_penalty


def _aggregate_visible_arc_fraction(features: list[NavFeature]) -> float:
    """Return the per-feature ``visible_arc_fraction`` weighted by vertex count."""
    total_weighted = 0.0
    total_count = 0.0
    for feat in features:
        fraction = feat.reliability_reasons.visible_arc_fraction
        if fraction is None:
            continue
        if not isinstance(feat.geometry, TerminatorPolyline):
            continue
        n = float(feat.geometry.vertices_vu.shape[0])
        total_weighted += float(fraction) * n
        total_count += n
    if total_count == 0.0:
        return 0.0
    return total_weighted / total_count
