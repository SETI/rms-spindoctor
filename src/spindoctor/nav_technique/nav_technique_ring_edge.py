"""``RingEdgeNav`` — translation fit from ring-edge polylines.

Consumes every ``RING_EDGE`` feature in the input set and produces a
single combined translation by minimizing the joint distance-transform
cost.  When every input ring edge is flagged ``is_straight_line`` the
combined Jacobian is rank-deficient — all parallel ring edges share a
single ring-plane normal, so the along-edge axis is unobservable.  The
returned covariance is honestly rank-1 in that case; the ensemble
combine fuses it with any orthogonal-axis result (a star, body limb,
body blob) before declaring a final answer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.dt_fit_gates import DTFitGateConfig, evaluate_dt_fit_gates
from spindoctor.nav_technique.dt_fitting import (
    build_polyline_mask,
    coarse_ncc_search_scored,
    lm_subpixel_refine,
)
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    add_model_error_floor,
    log_confidence_breakdown,
    rotation_pivot_distance_px,
    search_window_for_obs,
)
from spindoctor.nav_technique.ring_edge_geometry import (
    _absorbed_orbit_sensitivity,
    _aggregate_normal_orientation,
    _effective_orbit_sigma_px,
    _is_rank_1,
    _orbit_inflated_covariance,
    _rank1_projected_covariance,
)
from spindoctor.nav_technique.ring_edge_stats import (
    _EdgeFitStat,
    _per_edge_fit_stats,
    _per_edge_median_max,
    _per_edge_rms_summed,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['RingEdgeNav', 'aggregate_edge_normal_angle_deg']

# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.RingEdgeNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


def _aggregate_ring_edges(
    features: list[NavFeature],
) -> tuple[
    NDArrayFloatType,
    NDArrayFloatType,
    NDArrayFloatType,
    list[str],
    bool,
]:
    """Concatenate vertices, polarity normals, and per-vertex sigmas for ring edges.

    Returns:
        ``(vertices, polarity_normals, sigmas, feature_ids,
        every_edge_is_straight)``.  ``every_edge_is_straight`` is True
        when every input edge has ``is_straight_line=True``; the
        technique uses it to drive the rank-1 covariance path.
    """
    vert_chunks: list[NDArrayFloatType] = []
    normal_chunks: list[NDArrayFloatType] = []
    sigma_chunks: list[NDArrayFloatType] = []
    ids: list[str] = []
    every_straight = True
    seen_any = False
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        if feat.geometry.vertices_vu.shape[0] == 0:
            continue
        seen_any = True
        if not feat.geometry.is_straight_line:
            every_straight = False
        vert_chunks.append(feat.geometry.vertices_vu.astype(np.float64))
        # Negate the radial outward normal so the polarity check
        # (``dot(model, image_gradient) > 0``) accepts edges whose image
        # gradient points across the edge in the same sense as the
        # negated normal.
        normal_chunks.append(-feat.geometry.normals_vu.astype(np.float64))
        sigma_chunks.append(feat.geometry.sigma_radial_per_vertex_px.astype(np.float64))
        ids.append(feat.feature_id)
    empty_2 = np.empty((0, 2), np.float64)
    empty_1 = np.empty(0, np.float64)
    vertices = np.concatenate(vert_chunks, axis=0) if vert_chunks else empty_2
    normals = np.concatenate(normal_chunks, axis=0) if normal_chunks else empty_2
    sigmas = np.concatenate(sigma_chunks, axis=0) if sigma_chunks else empty_1
    return vertices, normals, sigmas, ids, (every_straight if seen_any else False)


class RingEdgeNav(NavTechnique):
    """Ring-edge DT-based translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({RING_EDGE})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'RingEdgeNav'
    accepts_feature_types = frozenset({NavFeatureType.RING_EDGE})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'total_edge_length_px',
            'per_edge_dt_rms_summed',
            'per_edge_dt_rms_mean',
            'per_edge_dt_median_max',
            'edge_count',
            'is_rank_1',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        """Read the technique's tunables from config.

        Parameters:
            config: Optional ``Config`` override; ``None`` uses
                ``DEFAULT_CONFIG``.

        Raises:
            KeyError: If any tuning key is missing, so a config typo fails
                at process startup rather than mid-image.
            ValueError: If ``edge_localization_sigma_px`` is not a finite
                number greater than zero (zero is physically meaningless
                against a half-pixel-quantized edge mask, and a
                non-numeric value would otherwise fail far from the
                config that caused it).
        """
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._spurious_dt_rms_factor = float(self.tuning['spurious_dt_rms_factor'])
        self._spurious_dt_floor_px = float(self.tuning['spurious_dt_floor_px'])
        self._spurious_min_inliers = int(self.tuning['spurious_min_inliers'])
        self._spurious_min_inlier_fraction = float(self.tuning['spurious_min_inlier_fraction'])
        self._spurious_waiver_min_well_fit_edges = int(
            self.tuning['spurious_waiver_min_well_fit_edges']
        )
        self._spurious_waiver_absent_median_px = float(
            self.tuning['spurious_waiver_absent_median_px']
        )
        self._spurious_waiver_sigma_floor_px = float(self.tuning['spurious_waiver_sigma_floor_px'])
        self._spurious_max_lm_displacement_px = float(
            self.tuning['spurious_max_lm_displacement_px']
        )
        raw_edge_sigma = self.tuning['edge_localization_sigma_px']
        try:
            self._edge_localization_sigma_px = float(raw_edge_sigma)
        except (TypeError, ValueError, OverflowError) as err:
            # Name the type, not the value: an oversized int's repr can
            # itself exceed the interpreter's integer-string digit limit.
            raise ValueError(
                'RingEdgeNav.edge_localization_sigma_px must be a finite number > 0; '
                f'got an unconvertible {type(raw_edge_sigma).__name__} value'
            ) from err
        if not math.isfinite(self._edge_localization_sigma_px) or (
            self._edge_localization_sigma_px <= 0.0
        ):
            raise ValueError(
                'RingEdgeNav.edge_localization_sigma_px must be a finite number > 0; '
                f'got {self._edge_localization_sigma_px!r}'
            )
        self._lm_trust_region_px = float(self.tuning['lm_trust_region_px'])
        self._lm_tikhonov_alpha = float(self.tuning['lm_tikhonov_alpha'])
        self._gradient_ridge_refine = bool(self.tuning['gradient_ridge_refine'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        self._dt_gate_config = DTFitGateConfig.from_tuning(self.tuning)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable ring edge.

        Reads only the polyline vertex count per feature.  A single
        non-empty ring-edge polyline is sufficient: even an all-flat
        scene produces a useful rank-1 constraint that the ensemble
        will fuse with another feature.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one RING_EDGE has a non-empty polyline.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, RingEdgePolyline) and f.geometry.vertices_vu.shape[0] > 0
        ]
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_ring_edge_features',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from ring-edge polylines.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance (rank-1 when every input edge is straight),
            calibrated confidence, and a populated
            :class:`RingEdgeDiagnostics`.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            if context.image_edge_dt_ext is None or context.image_gradient_vu_ext is None:
                raise RuntimeError(
                    'RingEdgeNav requires NavContext.image_edge_dt_ext and '
                    'NavContext.image_gradient_vu_ext to be populated by the orchestrator'
                )
            (
                vertices,
                polarity_normals,
                sigmas,
                feature_ids,
                every_straight,
            ) = _aggregate_ring_edges(features)
            self.logger.info(
                'Consuming %d RING_EDGE features (out of %d offered); every_straight=%s',
                len(feature_ids),
                len(features),
                every_straight,
            )
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            polyline_mask = build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = search_window_for_obs(context)
            model_sigma_min = float(sigmas.min()) if sigmas.size else 0.0
            model_sigma_max = float(sigmas.max()) if sigmas.size else 0.0
            # The fit's residual is the distance from a model vertex to the
            # nearest image edge, so its scale carries BOTH uncertainties: the
            # catalog's radial uncertainty (the per-vertex sigma) and how
            # precisely the image edge itself is placed.  The latter is the
            # binary edge mask's own quantization and dominates whenever a
            # well-solved ring edge projects to a small fraction of a pixel.
            # Combining them in quadrature is what keeps the robust weighting
            # measuring model agreement rather than pixel phase; see
            # ``edge_localization_sigma_px`` in the technique config.
            sigmas = np.hypot(sigmas, self._edge_localization_sigma_px)
            self.logger.debug(
                'Aggregated %d ring-edge vertices, model sigma_radial range '
                '[%.3f, %.3f] px, fit residual scale range [%.3f, %.3f] px after '
                'the %.2f px edge-localization term, search window (v, u) = (%d, %d) px',
                int(vertices.shape[0]),
                model_sigma_min,
                model_sigma_max,
                float(sigmas.min()) if sigmas.size else 0.0,
                float(sigmas.max()) if sigmas.size else 0.0,
                self._edge_localization_sigma_px,
                margin_v,
                margin_u,
            )
            coarse = coarse_ncc_search_scored(
                edge_mask,
                polyline_mask,
                (margin_v, margin_u),
            )
            coarse_dv, coarse_du = coarse.offset_vu
            self.logger.debug(
                'Coarse NCC offset: (%d, %d), peak match fraction %.3f',
                coarse_dv,
                coarse_du,
                coarse.score,
            )
            # Ring-edge polarity prediction depends on lighting / gap-vs-ringlet
            # context the catalog does not encode today, so this technique keeps
            # the polarity-blind coarse seed (coarse_ncc_search_scored above)
            # rather than the polarity-weighted acquisition the limb / terminator
            # techniques use; making the ring seed robust is tracked in #373.
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
                use_polarity=False,
                fit_rotation=fit_rotation,
                pivot_vu=pivot_vu if fit_rotation else None,
                pivot_distance_px=pivot_distance,
                trust_region_px=self._lm_trust_region_px,
                tikhonov_alpha=self._lm_tikhonov_alpha,
                final_gradient_ridge=self._gradient_ridge_refine,
            )
            dv_final, du_final = result.offset_vu
            max_rotation_rad = math.radians(context.max_rotation_deg)
            rotation_at_edge = fit_rotation and (
                abs(result.rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
            )
            # See BodyLimbNav: ``>=`` covers both at-edge and over-edge cases
            # so an LM that walked past the coarse-NCC search window does
            # not silently report an offset outside the extfov margin.
            if every_straight:
                # Rank-1 scene: the along-edge (tangent) offset component is
                # unobservable and nothing constrains it, so the LM may
                # slide to the search-window boundary along it (Tikhonov is
                # off by default).  Gating per axis would hard-zero a
                # perfectly valid normal-component fix; gate on the
                # observable component instead — at-edge along the normal
                # means the *measurement* is pinned at the window limit.
                n_hat = _aggregate_normal_orientation(polarity_normals)
                offset_along_normal = abs(dv_final * n_hat[0] + du_final * n_hat[1])
                # Maximum reach of the search box along the normal.
                bound_along_normal = abs(n_hat[0]) * margin_v + abs(n_hat[1]) * margin_u
                at_edge = (
                    offset_along_normal >= bound_along_normal - self._at_edge_tolerance_px
                    or rotation_at_edge
                )
            else:
                at_edge = (
                    abs(dv_final) >= margin_v - self._at_edge_tolerance_px
                    or abs(du_final) >= margin_u - self._at_edge_tolerance_px
                    or rotation_at_edge
                )
            sigma_min_px = float(sigmas.min()) if sigmas.size else 1.0
            covariance = result.covariance
            total_edge_length_px = float(vertices.shape[0])
            per_edge_rms_summed = _per_edge_rms_summed(features, result.residuals_px)
            per_edge_median_max = _per_edge_median_max(features, result.residuals_px)
            edge_count = len(feature_ids)
            # ``result.rms_px`` is the *Tukey-weighted* residual RMS; when
            # the LM converges to a local minimum where one edge fits
            # cleanly and the rest are wholly mis-aligned, Tukey rejects
            # the bad-edge vertices and ``rms_px`` collapses to near
            # zero -- a mis-convergence the ``rms_px > floor`` threshold
            # cannot detect.  The gate is therefore a minimum Tukey
            # inlier fraction over the aggregated vertices, and it is a
            # FLOOR against degenerate fits: one anchored on a handful
            # of vertices while the rest of the residuals sit within
            # scale of NO detected edge at all (Cassini Tethys
            # N1572471790 is the historical calibration case for the
            # floor -- the LM anchored a fit on a third of the model
            # vertices).  With a physical residual scale the pooled
            # fraction does NOT separate a correct fit from a lock onto
            # the wrong member of a concentric ring family: measured
            # over 71 Cassini B-ring frames, correct fits and
            # wrong-member locks both run pooled inlier fractions of
            # roughly 0.3 to 0.95, because an alignment one ringlet
            # spacing away still puts most of the model on AN image
            # edge.  Telling those apart is an acquisition problem, not
            # something this gate can read from converged residuals.
            #
            # The aggregate fraction alone over-reaches on a multi-edge
            # fusion where the vertex count is dominated by edges that
            # are simply not detectable in the image: the fused offset
            # is pinned by a well-fit edge at sub-pixel RMS, yet the
            # missing edges' vertices drag the aggregate fraction below
            # the gate (Cassini C ring N1467344214 is the calibration
            # case -- 821 / 3188 inliers on an operator-verified fit;
            # issue #261).  What separates that from a wrong-ring lock
            # is the *median DT residual of the edges the robust fit
            # rejected*: an undetected edge sits far from every detected
            # image edge (medians of tens of px -- there is nothing
            # there), while a mis-locked fusion leaves at least one
            # rejected edge lying ON a detected image edge it disagrees
            # with (median well under a pixel; Cassini Tethys
            # N1572472169's wrong lock rejects an edge whose median is
            # 0.04 px).  The veto is therefore waived only when the low
            # aggregate fraction is fully explained by absent edges:
            # every non-well-fit edge must have a per-edge median DT
            # residual of at least ``spurious_waiver_absent_median_px``,
            # at least ``spurious_waiver_min_well_fit_edges`` edges must
            # clear the inlier-fraction gate on their own vertices (with
            # a non-negligible ``spurious_min_inliers`` count), and the
            # fit's translation covariance must be full rank -- the
            # weighted vertices are almost all from the well-fit subset,
            # so a full-rank covariance certifies that the surviving
            # edges constrain both offset axes on their own.  A
            # single-edge fit never qualifies (its own fraction IS the
            # aggregate), so single-edge behavior is unchanged.
            total_vertices = int(vertices.shape[0])
            inlier_fraction = (
                float(result.inlier_count) / float(total_vertices) if total_vertices else 0.0
            )
            edge_stats = _per_edge_fit_stats(
                features, residuals=result.residuals_px, weights=result.weights
            )
            well_fit: list[_EdgeFitStat] = []
            rejected: list[_EdgeFitStat] = []
            for stat in edge_stats:
                if (
                    stat.inlier_count >= self._spurious_min_inliers
                    and stat.inlier_fraction >= self._spurious_min_inlier_fraction
                ):
                    well_fit.append(stat)
                else:
                    rejected.append(stat)
            self.logger.debug(
                'Per-edge inliers/vertices: %s; median |DT| px: %s; straight: %s; '
                'well-fit edges = %d of %d',
                ', '.join(f'{s.inlier_count}/{s.vertex_count}' for s in edge_stats),
                ', '.join(f'{s.median_abs_residual_px:.3f}' for s in edge_stats),
                ', '.join(str(s.is_straight) for s in edge_stats),
                len(well_fit),
                len(edge_stats),
            )
            translation_full_rank = not result.degenerate and not (
                every_straight or _is_rank_1(np.asarray(covariance, np.float64))
            )
            veto_waived = (
                len(edge_stats) >= 2
                and len(well_fit) >= self._spurious_waiver_min_well_fit_edges
                and translation_full_rank
                and all(
                    stat.median_abs_residual_px >= self._spurious_waiver_absent_median_px
                    for stat in rejected
                )
            )
            inlier_fraction_low = inlier_fraction < self._spurious_min_inlier_fraction
            inlier_fraction_veto = inlier_fraction_low and not veto_waived
            if inlier_fraction_low and veto_waived:
                self.logger.info(
                    'Aggregate inlier fraction %.3f is below the %.2f gate, but %d of %d '
                    'edges fit well independently and every rejected edge is absent from '
                    'the image (median |DT| >= %.1f px); waiving the mis-convergence veto',
                    inlier_fraction,
                    self._spurious_min_inlier_fraction,
                    len(well_fit),
                    len(edge_stats),
                    self._spurious_waiver_absent_median_px,
                )
            if every_straight:
                # As with at_edge above: the tangent component of the
                # LM-vs-coarse displacement is unconstrained slide along the
                # unobservable axis, not evidence of a runaway fit.
                n_hat = _aggregate_normal_orientation(polarity_normals)
                lm_displacement_px = abs(
                    (dv_final - float(coarse_dv)) * n_hat[0]
                    + (du_final - float(coarse_du)) * n_hat[1]
                )
            else:
                lm_displacement_px = float(
                    math.hypot(dv_final - float(coarse_dv), du_final - float(coarse_du))
                )
            # Shared DT fit-quality gates.  RingEdgeNav runs polarity-free
            # (see the lm_subpixel_refine call above), so the polarity gates
            # are inert here; the coarse-peak and LM-convergence signals
            # still apply.
            gate_verdict = evaluate_dt_fit_gates(
                result,
                self._dt_gate_config,
                coarse_peak_fraction=coarse.score,
                total_vertex_count=total_vertices,
                use_polarity=False,
            )
            if gate_verdict.spurious_reasons:
                self.logger.info(
                    'DT fit-quality gate(s) fired: %s (coarse peak %.3f, lm_converged=%s)',
                    ', '.join(gate_verdict.spurious_reasons),
                    gate_verdict.coarse_peak_fraction,
                    gate_verdict.lm_converged,
                )
            spurious = (
                result.degenerate
                or result.rms_px
                > max(
                    self._spurious_dt_floor_px,
                    self._spurious_dt_rms_factor * sigma_min_px,
                )
                or result.inlier_count < self._spurious_min_inliers
                or inlier_fraction_veto
                or lm_displacement_px > self._spurious_max_lm_displacement_px
                or gate_verdict.spurious
            )
            rotation_rad: float | None
            sigma_rotation_rad: float | None
            if fit_rotation:
                if covariance.shape != (3, 3):
                    raise RuntimeError(
                        f'RingEdgeNav expected 3x3 covariance with fit_rotation; '
                        f'got {covariance.shape}'
                    )
                rotation_rad = float(result.rotation_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(covariance[2, 2]), 0.0)))
            else:
                if covariance.shape != (2, 2):
                    self.logger.warning(
                        'RingEdgeNav: lm_subpixel_refine returned %s covariance with '
                        'fit_rotation=False; truncating to (2, 2)',
                        covariance.shape,
                    )
                    covariance = covariance[:2, :2]
                rotation_rad = None
                sigma_rotation_rad = None
            covariance = np.asarray(covariance, np.float64)
            is_rank_1 = _is_rank_1(covariance) or every_straight
            if is_rank_1:
                # Enforce the rank-1 contract on the reported covariance.
                # On an all-straight scene the along-edge (tangent) offset
                # component is physically unobservable, but the LM
                # covariance often comes back numerically tight along the
                # tangent anyway: the model polyline's *ends* anchor the
                # along-edge coordinate against wherever the detected edge
                # happens to enter and leave the frame — a false
                # constraint.  Rebuild the translation covariance as the
                # LM's marginal variance along the aggregate edge normal
                # with the tangent variance exactly zero (the Moore-Penrose
                # null-space convention), so the ensemble's rank-deficiency
                # test fires and the tangent axis is reported through the
                # ``sigma_along_unobservable_px = inf`` sentinel instead of
                # a confidently full-rank fix (issue #203).
                covariance = _rank1_projected_covariance(covariance, polarity_normals)
            if inlier_fraction_low and veto_waived:
                # A veto-waived fit is carried by a minority of its model
                # in a scene whose parallel ringlet structure offers alias
                # alignments the DT cannot rule out (the whole point of
                # the waived veto).  The LM covariance measures only the
                # statistical precision of the anchored vertices, so add
                # the waiver sigma floor in quadrature (the #210
                # convention) to keep the fix below the medium tier's
                # sigma cap: it surfaces as a low-tier result and cannot
                # outweigh a full-support fix in the ensemble.
                covariance = add_model_error_floor(covariance, self._spurious_waiver_sigma_floor_px)
                self.logger.info(
                    'Applying the spurious-waiver sigma floor of %.1f px to the '
                    'reported covariance',
                    self._spurious_waiver_sigma_floor_px,
                )
            # Radial orbit-uncertainty channel: the LM covariance describes
            # the statistical lock onto the MODELED annulus; a catalog-orbit
            # error displaces the whole annulus coherently and does not
            # average down over vertices, so the declared orbit sigma is
            # added to the reported covariance through the translation the
            # fit would ABSORB from such a displacement
            # (``_absorbed_orbit_sensitivity``: a short arc absorbs it
            # one-for-one along its normal, a full annulus barely absorbs it
            # at all because a uniform radial error is a dilation).  On a
            # rank-1 scene the projection's own axis is reused unchanged, so
            # the covariance stays exactly singular along the tangent.
            sigma_orbit_px = _effective_orbit_sigma_px(features, result.weights)
            if sigma_orbit_px > 0.0:
                if is_rank_1:
                    orbit_g = _aggregate_normal_orientation(polarity_normals)
                else:
                    orbit_g = _absorbed_orbit_sensitivity(polarity_normals, result.weights)
                covariance = _orbit_inflated_covariance(covariance, orbit_g, sigma_orbit_px)
                self.logger.info(
                    'Widening the reported covariance by the declared radial orbit '
                    'uncertainty: sigma %.3f px, absorbed sensitivity (v, u) = '
                    '(%.3f, %.3f) (|g| = %.3f)',
                    sigma_orbit_px,
                    float(orbit_g[0]),
                    float(orbit_g[1]),
                    float(np.linalg.norm(orbit_g)),
                )
            per_edge_rms_mean = per_edge_rms_summed / float(max(edge_count, 1))
            diagnostics = RingEdgeDiagnostics(
                total_edge_length_px=total_edge_length_px,
                per_edge_dt_rms_summed=per_edge_rms_summed,
                per_edge_dt_rms_mean=per_edge_rms_mean,
                per_edge_dt_median_max=per_edge_median_max,
                edge_count=edge_count,
                is_rank_1=bool(is_rank_1),
                lm_converged=gate_verdict.lm_converged,
                coarse_peak_fraction=gate_verdict.coarse_peak_fraction,
                sigma_orbit_radial_px=sigma_orbit_px,
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _RingEdgeConfidenceContext(at_edge=at_edge, diagnostics=diagnostics),
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
                'rank_1=%s, confidence %.4f',
                dv_final,
                du_final,
                result.rms_px,
                result.inlier_count,
                int(vertices.shape[0]),
                is_rank_1,
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
                'per_edge_rms_summed = %.3f, per_edge_median_max = %.3f, '
                'total_edge_length = %.1f px',
                result.iterations,
                sigma_min_px,
                per_edge_rms_summed,
                per_edge_median_max,
                total_edge_length_px,
            )
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
            )


class _RingEdgeConfidenceContext:
    """Adapter exposing ring-edge confidence terms in a single attribute set."""

    def __init__(self, *, at_edge: bool, diagnostics: RingEdgeDiagnostics) -> None:
        """Flatten the diagnostics plus ``at_edge`` into one attribute set.

        Parameters:
            at_edge: Whether the converged offset reached the search-window
                boundary; it lives on the result, not the diagnostics.
            diagnostics: The technique's populated diagnostics.
        """
        self.at_edge = at_edge
        self.total_edge_length_px = diagnostics.total_edge_length_px
        self.per_edge_dt_rms_summed = diagnostics.per_edge_dt_rms_summed
        self.per_edge_dt_rms_mean = diagnostics.per_edge_dt_rms_mean
        self.per_edge_dt_median_max = diagnostics.per_edge_dt_median_max
        self.edge_count = diagnostics.edge_count
        self.is_rank_1 = diagnostics.is_rank_1


def aggregate_edge_normal_angle_deg(features: list[NavFeature]) -> float | None:
    """Return the dominant edge-normal orientation of an all-straight scene.

    The angle is in degrees in the ``(v, u)`` frame, measured from ``+v``
    toward ``+u`` (the unit normal is ``(cos, sin)``) — the same convention
    as the sidecar schema's ``ground_truth.constraint.normal_angle_deg``.
    The orientation is the dominant eigenvector of the per-vertex normals'
    outer-product sum, so it is independent of each edge's polarity sign.

    Parameters:
        features: Any feature list; only ``RING_EDGE`` polylines are read.

    Returns:
        The orientation in degrees, normalized to ``(-90, 90]``, or ``None``
        unless at least one ring-edge polyline is present and every one is
        straight — the constraint direction is only meaningful for a rank-1
        scene.
    """
    _vertices, normals, _sigmas, ids, every_straight = _aggregate_ring_edges(features)
    if not ids or not every_straight:
        return None
    n_hat = _aggregate_normal_orientation(normals)
    angle = math.degrees(math.atan2(float(n_hat[1]), float(n_hat[0])))
    # An orientation, not a direction: fold to (-90, 90].
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    return float(angle)
