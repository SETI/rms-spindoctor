"""``StarRefineNav`` — pass-2 star-refinement technique.

Consumes the pass-1 ensemble's prior offset and refines it via local
moment centroids on every predicted catalog star.  For each STAR
feature, the technique:

1. Shifts the catalog prediction by the prior offset.
2. Looks for a brightness peak inside a small refinement window
   centered on that shifted prediction.
3. Fits a brightness-weighted moment around the peak to get a sub-pixel
   centroid.
4. Computes per-star residuals (observed - shifted_prediction) and
   averages them in inverse-variance fashion.

Per-star inverse variance is the trace of the feature's CRLB
covariance carried on ``NavFeature.position_cov_px``; stars with low
predicted SNR get less weight.  Stars whose detection is too far from
the shifted prediction (likely a wrong peak) are dropped before the
fit.

The refined offset is reported as a *delta* from the prior — the
ensemble combine adds it back.  The covariance reflects the residual
scatter across the surviving stars; with two or more inliers the
technique reports the actual scatter, with a single inlier the CRLB
floor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_technique._star_helpers import (
    local_centroid,
    similarity_transform_fit,
    usable_stars,
)
from spindoctor.nav_technique._star_streak import (
    frame_streak_vu,
    is_streak,
    streak_half_extent_px,
)
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import StarRefineDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    add_model_error_floor,
    embed_rotation_unobservable,
    load_model_error_floor,
    log_confidence_breakdown,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.ring_edge_geometry import is_rank_1
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['StarRefineNav']


# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.StarRefineNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


def _trace_inverse_variance(feature: NavFeature) -> float:
    """Return the inverse-trace of a feature's position covariance.

    The CRLB covariance carried on ``NavFeature.position_cov_px`` is the
    per-feature uncertainty floor; its trace is the sum of per-axis
    variances and ``1 / trace`` is a fast precision-weighted scalar
    proxy for the feature's worth in a least-squares average.
    """
    cov = feature.position_cov_px
    if cov is None:
        return 1.0
    arr = np.asarray(cov, dtype=np.float64)
    trace = float(np.trace(arr))
    if trace <= 0.0:
        return 1.0
    return 1.0 / trace


class _RefineConfidenceContext:
    """Adapter binding ``StarRefineDiagnostics`` plus side flags."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: StarRefineDiagnostics,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.n_stars_used = float(diagnostics.n_stars_used)
        self.median_pos_err_px = diagnostics.median_pos_err_px
        self.residual_scatter_px = diagnostics.residual_scatter_px


class StarRefineNav(NavTechnique):
    """Star-refinement technique consuming the pass-1 prior.

    A 1-inlier refine carries no independent cross-check information
    beyond the prior it was handed — it is the same single observation
    that drove the pass-1 fit, just polished.  The post-sigmoid
    ``single_inlier_confidence_cap`` (default 0.5) prevents the
    technique from promoting itself above ``StarUniqueMatchNav``'s
    0.7 1-star cap on the same observation.  With ≥ 2 inliers the per-
    star residual scatter cross-checks the joint fit and the cap is
    not applied.

    Class attributes:
        accepts_feature_types: ``frozenset({STAR})``.
        requires_prior: ``True`` — runs in pass 2.
    """

    name = 'StarRefineNav'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = True
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'n_stars_used',
            'median_pos_err_px',
            'residual_scatter_px',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._refine_window_px = float(self.tuning['refine_window_px'])
        self._centroid_box_half_px = int(self.tuning['centroid_box_half_px'])
        self._max_per_star_residual_px = float(self.tuning['max_per_star_residual_px'])
        self._discordant_min_stars = int(self.tuning['discordant_min_stars'])
        self._discordant_floor_px = float(self.tuning['discordant_floor_px'])
        self._discordant_median_factor = float(self.tuning['discordant_median_factor'])
        self._detection_sigma = float(self.tuning['detection_sigma'])
        self._min_inliers = int(self.tuning['min_inliers'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._single_inlier_confidence_cap = float(self.tuning['single_inlier_confidence_cap'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        self._model_error_floor_px = load_model_error_floor(self.tuning, self.name)
        if self._min_inliers < 1:
            raise ValueError(f'min_inliers must be >= 1; got {self._min_inliers}')
        # Below two, a prior vote could stand in for every star and leave the median
        # distance a median over no residuals at all.
        if self._discordant_min_stars < 2:
            raise ValueError(f'discordant_min_stars must be >= 2; got {self._discordant_min_stars}')
        if not 0.0 <= self._single_inlier_confidence_cap <= 1.0:
            raise ValueError(
                f'single_inlier_confidence_cap must lie in [0, 1]; got '
                f'{self._single_inlier_confidence_cap!r}'
            )

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Report feasibility based on the predictable-star cohort."""
        usable = usable_stars(features)
        if not usable:
            return NavFeasibilityReport(feasible=False, reason='no_usable_star_features')
        return NavFeasibilityReport(feasible=True, reason='ok', consumed_feature_count=len(usable))

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Refine the pass-1 prior by per-star centroid residuals.

        Parameters:
            features: STAR features (the orchestrator pre-filters by type).
            context: Per-image NavContext.  Must carry ``prior_offset_px``;
                the technique is registered as ``requires_prior=True`` so
                the orchestrator only invokes it on pass 2.

        Returns:
            ``NavTechniqueResult`` with the *delta* offset relative to the
            prior, a 2x2 covariance, calibrated confidence, and a
            populated :class:`StarRefineDiagnostics`.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            usable = usable_stars(features)
            self.logger.info(
                'Consuming %d usable STAR feature(s) (out of %d offered)',
                len(usable),
                len(features),
            )
            if context.prior_offset_px is None:
                return self._fail(
                    features=features,
                    reason='no_prior_offset_on_context',
                    diagnostics=StarRefineDiagnostics(
                        n_stars_used=0,
                        median_pos_err_px=0.0,
                        residual_scatter_px=0.0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            prior_v, prior_u = context.prior_offset_px
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            inliers, per_star_residuals_v, per_star_residuals_u, weights = self._collect_residuals(
                usable,
                image_ext,
                noise_sigma,
                prior_v,
                prior_u,
                data_margin_vu=search_window_for_obs(context),
            )
            inliers, per_star_residuals_v, per_star_residuals_u, weights = self._reject_discordant(
                inliers,
                per_star_residuals_v,
                per_star_residuals_u,
                weights,
                prior_covariance_px2=context.prior_covariance_px2,
            )
            if len(inliers) < self._min_inliers:
                return self._fail(
                    features=features,
                    reason=(f'too_few_inliers ({len(inliers)} < min {self._min_inliers})'),
                    diagnostics=StarRefineDiagnostics(
                        n_stars_used=len(inliers),
                        median_pos_err_px=0.0,
                        residual_scatter_px=0.0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            total_weight = float(weights.sum())
            delta_v = float(np.sum(weights * per_star_residuals_v) / total_weight)
            delta_u = float(np.sum(weights * per_star_residuals_u) / total_weight)
            distances = np.hypot(per_star_residuals_v, per_star_residuals_u)
            median_pos_err_px = float(np.median(distances))
            scatter_v = float(
                math.sqrt(
                    float(np.sum(weights * (per_star_residuals_v - delta_v) ** 2)) / total_weight
                )
            )
            scatter_u = float(
                math.sqrt(
                    float(np.sum(weights * (per_star_residuals_u - delta_u) ** 2)) / total_weight
                )
            )
            residual_scatter_px = math.hypot(scatter_v, scatter_u)
            diagnostics = StarRefineDiagnostics(
                n_stars_used=len(inliers),
                median_pos_err_px=median_pos_err_px,
                residual_scatter_px=residual_scatter_px,
            )
            margin_v, margin_u = search_window_for_obs(context)
            at_edge = (
                abs(delta_v + prior_v) >= margin_v - self._at_edge_tolerance_px
                or abs(delta_u + prior_u) >= margin_u - self._at_edge_tolerance_px
            )
            assert self.confidence_spec is not None
            raw_confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _RefineConfidenceContext(at_edge=at_edge, spurious=False, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            # 1-inlier refines carry no independent cross-check beyond the
            # prior they were handed; cap their confidence so the
            # ensemble's primary-technique selection cannot promote a
            # 1-star refine over a 1-star unique-match on the same
            # observation.
            confidence = float(raw_confidence)
            if len(inliers) == 1 and confidence > self._single_inlier_confidence_cap:
                self.logger.info(
                    'Single-inlier refine: capping confidence %.4f -> %.4f '
                    '(no cross-check on a 1-star refine)',
                    confidence,
                    self._single_inlier_confidence_cap,
                )
                confidence = self._single_inlier_confidence_cap
            cov_2x2 = self._build_covariance(
                inliers=inliers,
                weights=weights,
                residuals_v=per_star_residuals_v,
                residuals_u=per_star_residuals_u,
                delta_v=delta_v,
                delta_u=delta_u,
            )
            fit_rotation = bool(context.fit_camera_rotation)
            rotation_rad: float | None = None
            sigma_rotation_rad: float | None = None
            cov: NDArrayFloatType
            offset_v_total = delta_v + prior_v
            offset_u_total = delta_u + prior_u
            if fit_rotation and len(inliers) >= 2:
                rotation_rad, sigma_rotation_rad, cov, offset_v_total, offset_u_total = (
                    self._fit_rotation_3dof(
                        inliers=inliers,
                        residuals_v=per_star_residuals_v,
                        residuals_u=per_star_residuals_u,
                        weights=weights,
                        prior_v=prior_v,
                        prior_u=prior_u,
                        cov_2x2=cov_2x2,
                    )
                )
                # The Procrustes refit can pull the absolute offset
                # outside the search window even when the
                # delta-based ``at_edge`` was False; recompute against
                # the updated absolute offset so the flag reflects
                # what the technique actually reports.
                at_edge = (
                    abs(offset_v_total) >= margin_v - self._at_edge_tolerance_px
                    or abs(offset_u_total) >= margin_u - self._at_edge_tolerance_px
                )
                rotation_at_edge = abs(
                    rotation_rad
                ) >= self._rotation_at_edge_fraction * math.radians(context.max_rotation_deg)
                if rotation_at_edge:
                    at_edge = True
                self.logger.info(
                    'Rotation = %+.4f deg (sigma %.4f deg)%s',
                    math.degrees(rotation_rad),
                    math.degrees(sigma_rotation_rad if sigma_rotation_rad is not None else 0.0),
                    ', AT_EDGE' if rotation_at_edge else '',
                )
            elif fit_rotation:
                # Single inlier carries no rotation evidence; promote to a
                # rank-deficient 3x3 with the rotation-unobservable
                # sentinel.  Translation block stays as derived from the
                # CRLB floor.
                cov = embed_rotation_unobservable(cov_2x2)
                rotation_rad = 0.0
                sigma_rotation_rad = rotation_unobservable_sigma_rad()
            else:
                cov = cov_2x2
            self.logger.info(
                'Refined %d star(s); delta (%.4f, %.4f) px from prior '
                '(%.4f, %.4f); median pos err %.4f px; scatter %.4f px; '
                'confidence %.4f',
                len(inliers),
                delta_v,
                delta_u,
                prior_v,
                prior_u,
                median_pos_err_px,
                residual_scatter_px,
                confidence,
            )
            # Model-error floor (#210); rationale on load_model_error_floor.
            cov = add_model_error_floor(cov, self._model_error_floor_px)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in inliers),
                offset_px=(offset_v_total, offset_u_total),
                covariance_px2=cov,
                confidence=confidence,
                spurious=False,
                at_edge=at_edge,
                diagnostics=diagnostics,
                rotation_rad=rotation_rad,
                sigma_rotation_rad=sigma_rotation_rad,
            )

    def _collect_residuals(
        self,
        stars: list[NavFeature],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        prior_v: float,
        prior_u: float,
        *,
        data_margin_vu: tuple[int, int],
    ) -> tuple[
        list[NavFeature],
        NDArrayFloatType,
        NDArrayFloatType,
        NDArrayFloatType,
    ]:
        """Per-star centroid residuals around the shifted predictions.

        A detection whose centroid box, ``centroid_box_half_px`` either side of the
        centroid and on a streak frame grown on each axis by half the streak's extent, is
        not wholly inside the image's data area (the extended-FOV array less
        ``data_margin_vu`` on each side) is dropped: the padding
        beyond the frame holds no starlight, so a box it cuts measures a truncated star,
        or the frame's edge pixels when the star itself lies outside the frame.
        """
        streak_vu = frame_streak_vu(stars)
        ext_v, ext_u = streak_half_extent_px(streak_vu) if is_streak(streak_vu) else (0, 0)
        box_v = self._centroid_box_half_px + ext_v
        box_u = self._centroid_box_half_px + ext_u
        v_bounds = (data_margin_vu[0] + box_v, image_ext.shape[0] - data_margin_vu[0] - box_v - 1)
        u_bounds = (data_margin_vu[1] + box_u, image_ext.shape[1] - data_margin_vu[1] - box_u - 1)
        inliers: list[NavFeature] = []
        residuals_v: list[float] = []
        residuals_u: list[float] = []
        weights: list[float] = []
        for star in stars:
            assert isinstance(star.geometry, StarGeometry)
            assert isinstance(star.flags, StarFlags)
            pred_v, pred_u = star.geometry.predicted_vu
            shifted_pred = (pred_v + prior_v, pred_u + prior_u)
            det, peak = local_centroid(
                image_ext,
                shifted_pred,
                search_window_px=self._refine_window_px,
                centroid_box_half_px=self._centroid_box_half_px,
                image_noise_sigma=noise_sigma,
                detection_sigma=self._detection_sigma,
                streak_vu=star.flags.streak_vu,
            )
            if det is None:
                self.logger.debug(
                    'Star %s: no peak above %.2f sigma in refine window (peak %.3f DN); dropped',
                    star.feature_id,
                    self._detection_sigma,
                    peak,
                )
                continue
            if not (v_bounds[0] <= det[0] <= v_bounds[1] and u_bounds[0] <= det[1] <= u_bounds[1]):
                self.logger.debug(
                    'Star %s: detection (%.2f, %.2f) has its centroid box cut by the frame '
                    'edge; dropped',
                    star.feature_id,
                    det[0],
                    det[1],
                )
                continue
            res_v = det[0] - shifted_pred[0]
            res_u = det[1] - shifted_pred[1]
            distance = math.hypot(res_v, res_u)
            if distance > self._max_per_star_residual_px:
                self.logger.debug(
                    'Star %s: refine residual %.3f px exceeds max %.3f px; dropped',
                    star.feature_id,
                    distance,
                    self._max_per_star_residual_px,
                )
                continue
            inliers.append(star)
            residuals_v.append(res_v)
            residuals_u.append(res_u)
            weights.append(_trace_inverse_variance(star))
        return (
            inliers,
            np.asarray(residuals_v, dtype=np.float64),
            np.asarray(residuals_u, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )

    def _reject_discordant(
        self,
        inliers: list[NavFeature],
        residuals_v: NDArrayFloatType,
        residuals_u: NDArrayFloatType,
        weights: NDArrayFloatType,
        *,
        prior_covariance_px2: NDArrayFloatType | None,
    ) -> tuple[list[NavFeature], NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
        """Drop stars whose residual disagrees with the rest before they are averaged.

        The average weights each star by its catalog brightness, so one bright star
        centroided on the wrong peak outweighs every faint star that agrees.  With at
        least ``discordant_min_stars`` inliers, a star whose residual lies farther from the
        per-axis median residual than ``max(discordant_floor_px, discordant_median_factor
        * median distance)`` is dropped.  With exactly one inlier fewer than that, a prior
        precise to ``discordant_floor_px`` (a covariance of full rank whose least certain
        axis has a standard deviation of at most that) casts the missing vote: the
        per-axis median is taken over the residuals and one more residual of zero, while
        the median distance is still taken over the stars alone.  The prior is the pass-1
        answer the stars were searched around, usually built from some of the same stars,
        so its vote adds no independent evidence: it settles a 2-to-1 split of three stars
        in the pass-1 answer's favor.  The star that disagrees with both the other two and
        the prior is then dropped, and a pair that disagrees with the prior cannot outvote
        a star that agrees with it.  A prior that is less precise, or rank-deficient like
        a ring edge's that leaves the position along the arc unconstrained, casts no vote,
        and fewer inliers still are returned unchanged.

        Parameters:
            inliers: Stars that passed the per-star checks.
            residuals_v: Their v residuals (px).
            residuals_u: Their u residuals (px).
            weights: Their inverse-variance weights.
            prior_covariance_px2: The 2x2 covariance of the prior the residuals are
                measured from, or None when there is none.

        Returns:
            The same four collections with the discordant stars removed.
        """
        prior_votes = self._prior_is_precise(prior_covariance_px2)
        if len(inliers) + int(prior_votes) < self._discordant_min_stars:
            return inliers, residuals_v, residuals_u, weights
        votes_v, votes_u = residuals_v, residuals_u
        if len(inliers) < self._discordant_min_stars:
            # The prior is the zero residual: the pass-1 answer the stars were searched
            # around, which settles a split the stars alone cannot in that answer's
            # favor (F ring frame W1762900979: two stars on the prior, a third 3.9 px
            # off it).
            votes_v = np.append(residuals_v, 0.0)
            votes_u = np.append(residuals_u, 0.0)
        distances = np.hypot(
            residuals_v - float(np.median(votes_v)),
            residuals_u - float(np.median(votes_u)),
        )
        tolerance = max(
            self._discordant_floor_px,
            self._discordant_median_factor * float(np.median(distances)),
        )
        keep = distances <= tolerance
        for star, distance in zip(inliers, distances, strict=True):
            if distance > tolerance:
                self.logger.debug(
                    'Star %s: residual %.3f px from the median residual exceeds %.3f px; '
                    'dropped as discordant',
                    star.feature_id,
                    float(distance),
                    tolerance,
                )
        return (
            [star for star, kept in zip(inliers, keep, strict=True) if kept],
            residuals_v[keep],
            residuals_u[keep],
            weights[keep],
        )

    def _prior_is_precise(self, prior_covariance_px2: NDArrayFloatType | None) -> bool:
        """Return True when a prior is precise enough to vote on a discordant star.

        A prior votes only when its covariance has full rank and the standard deviation
        along its least certain axis is at most ``discordant_floor_px``: one less certain
        than that cannot say that a star ``discordant_floor_px`` from it is wrong.  A
        rank-deficient covariance, such as a ring edge's, reports a zero variance along
        the direction it does not constrain, so its largest eigenvalue says nothing about
        that direction and it casts no vote either.

        Parameters:
            prior_covariance_px2: The prior's 2x2 (or 3x3, of which the translation block
                is read) covariance, or None.

        Returns:
            True when the covariance is present, finite, of full rank in its translation
            block, and its largest eigenvalue is at most ``discordant_floor_px`` squared.
        """
        # F ring frame N1798993130: a ring-edge prior 2.4 px along the arc from the
        # stars, with two stray peaks near it, would otherwise outvote the brightest
        # star, which four fainter catalog stars place at the same offset.
        if prior_covariance_px2 is None:
            return False
        cov = np.asarray(prior_covariance_px2, np.float64)[:2, :2]
        if not bool(np.all(np.isfinite(cov))) or is_rank_1(cov):
            return False
        return float(np.linalg.eigvalsh(cov).max()) <= self._discordant_floor_px**2

    def _fit_rotation_3dof(
        self,
        *,
        inliers: list[NavFeature],
        residuals_v: NDArrayFloatType,
        residuals_u: NDArrayFloatType,
        weights: NDArrayFloatType,
        prior_v: float,
        prior_u: float,
        cov_2x2: NDArrayFloatType,
    ) -> tuple[float, float, NDArrayFloatType, float, float]:
        """Run a Procrustes / similarity refit on the inlier set.

        Reconstructs the per-star detection and shifted-prediction
        positions from the residuals + the original catalog predictions,
        then runs :func:`similarity_transform_fit` to extract the rotation
        and translation.  The returned ``offset_v_total`` /
        ``offset_u_total`` is the absolute (not delta) offset that the
        caller should report on the result, consistent with the legacy
        contract.

        Returns:
            ``(rotation_rad, sigma_rotation_rad, covariance_3x3,
            offset_v_total, offset_u_total)``.
        """
        cat_pts = np.asarray(
            [list(star.geometry.predicted_vu) for star in inliers],  # type: ignore[union-attr, misc]
            dtype=np.float64,
        )
        shifted_pred = cat_pts + np.asarray([prior_v, prior_u], dtype=np.float64)[None, :]
        det_pts = shifted_pred + np.column_stack([residuals_v, residuals_u])
        # Run Procrustes with the inverse-trace weights derived from the
        # per-star CRLB (matching the translation-only path's weighting).
        sim_fit = similarity_transform_fit(det_pts, cat_pts, weights)
        offset_v_total = float(sim_fit.translation_vu[0])
        offset_u_total = float(sim_fit.translation_vu[1])
        # Rotation sigma from the inlier residual scatter against the
        # weighted catalog spread (isotropic pooled lever-arm form;
        # ``StarFieldFromCatalogNav._build_covariance_3dof`` uses the
        # exact anisotropic Fisher-information form).
        total = float(weights.sum())
        if total <= 0.0:
            sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
        else:
            cat_c_v = float(np.sum(weights * cat_pts[:, 0]) / total)
            cat_c_u = float(np.sum(weights * cat_pts[:, 1]) / total)
            dv = cat_pts[:, 0] - cat_c_v
            du = cat_pts[:, 1] - cat_c_u
            spread = float(np.sum(weights * (dv * dv + du * du)))
            var_v = float(np.sum(weights * sim_fit.residuals_vu[:, 0] ** 2)) / total
            var_u = float(np.sum(weights * sim_fit.residuals_vu[:, 1] ** 2)) / total
            var_residual = 0.5 * (var_v + var_u)
            if spread <= 0.0:
                sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
            else:
                sigma_theta_sq = max(var_residual / spread, 1.0 / spread)
        cov = np.zeros((3, 3), dtype=np.float64)
        cov[:2, :2] = cov_2x2
        cov[2, 2] = sigma_theta_sq
        sigma_rotation_rad = float(np.sqrt(max(sigma_theta_sq, 0.0)))
        return (
            float(sim_fit.rotation_rad),
            sigma_rotation_rad,
            cov,
            offset_v_total,
            offset_u_total,
        )

    def _build_covariance(
        self,
        *,
        inliers: list[NavFeature],
        weights: NDArrayFloatType,
        residuals_v: NDArrayFloatType,
        residuals_u: NDArrayFloatType,
        delta_v: float,
        delta_u: float,
    ) -> NDArrayFloatType:
        """Return the per-axis precision-weighted covariance.

        With one inlier the technique reports the CRLB floor from the
        feature's ``position_cov_px``.  With two or more inliers the
        per-axis residual scatter dominates and is reported.
        """
        if len(inliers) <= 1:
            cov = inliers[0].position_cov_px
            if cov is None:
                return np.eye(2, dtype=np.float64)
            return np.asarray(cov, dtype=np.float64).copy()
        total_weight = float(weights.sum())
        var_v = float(np.sum(weights * (residuals_v - delta_v) ** 2)) / total_weight
        var_u = float(np.sum(weights * (residuals_u - delta_u) ** 2)) / total_weight
        floor = 1.0 / max(total_weight, 1e-12)
        return np.diag([max(var_v, floor), max(var_u, floor)]).astype(np.float64)

    def _fail(
        self,
        *,
        features: list[NavFeature],
        reason: str,
        diagnostics: StarRefineDiagnostics,
        fit_rotation: bool = False,
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result with the supplied reason."""
        self.logger.info('Reporting spurious result: %s', reason)
        return self._spurious_result(
            feature_ids=tuple(f.feature_id for f in features),
            diagnostics=diagnostics,
            fit_rotation=fit_rotation,
        )
