"""``StarRefineNav`` — pass-2 star-refinement technique.

Consumes the pass-1 ensemble's prior offset and refines it via local
moment centroids on every predicted catalog star.  For each STAR
feature, the technique:

1. Shifts the catalog prediction by the prior offset.
2. Looks for the strongest matched-filter detection inside a small
   refinement window centered on that shifted prediction.
3. Fits a brightness-weighted moment around the peak to get a sub-pixel
   centroid.  Missing pixels near the star read as the local background,
   and a star that loses enough of its light to them to move its centroid
   by more than ``max_missing_shift_px`` is dropped.
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
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.ndimage import map_coordinates

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_technique._star_helpers import (
    local_centroid,
    search_window_bounds,
    similarity_transform_fit,
    usable_stars,
)
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import StarRefineDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    add_model_error_floor,
    embed_rotation_unobservable,
    log_confidence_breakdown,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.star_tuning import StarRefineTuning
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.missing_light import missing_light_shift_px
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['StarRefineNav']

_MISSING_FILL_QUANTILE = 0.25
"""The quantile of a star's present pixels that its missing pixels are set to.

On sky it sits about two thirds of a noise sigma below the median, under the background
the centroid subtracts, so a filled pixel adds no weight to the centroid.  It is close
enough to the sky to leave that background and the matched filter almost unchanged.
"""


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
    Tuning = StarRefineTuning
    tuning: ClassVar[StarRefineTuning]
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
        """Load the configuration, which sets the technique's tuning.

        Parameters:
            config: Optional ``Config`` override; ``None`` uses
                ``DEFAULT_CONFIG``.
        """
        super().__init__(config=config)
        self.config.read_config()

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
                missing_mask_ext=np.asarray(context.missing_data_mask_ext, bool),
            )
            if len(inliers) < self.tuning.min_inliers:
                return self._fail(
                    features=features,
                    reason=(f'too_few_inliers ({len(inliers)} < min {self.tuning.min_inliers})'),
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
                abs(delta_v + prior_v) >= margin_v - self.tuning.at_edge_tolerance_px
                or abs(delta_u + prior_u) >= margin_u - self.tuning.at_edge_tolerance_px
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
            if len(inliers) == 1 and confidence > self.tuning.single_inlier_confidence_cap:
                self.logger.info(
                    'Single-inlier refine: capping confidence %.4f -> %.4f '
                    '(no cross-check on a 1-star refine)',
                    confidence,
                    self.tuning.single_inlier_confidence_cap,
                )
                confidence = self.tuning.single_inlier_confidence_cap
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
                    abs(offset_v_total) >= margin_v - self.tuning.at_edge_tolerance_px
                    or abs(offset_u_total) >= margin_u - self.tuning.at_edge_tolerance_px
                )
                rotation_at_edge = abs(
                    rotation_rad
                ) >= self.tuning.rotation_at_edge_fraction * math.radians(context.max_rotation_deg)
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
            # Model-error floor (#210); rationale on add_model_error_floor.
            cov = add_model_error_floor(cov, self.tuning.model_error_floor_px)
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
        missing_mask_ext: NDArrayBoolType,
    ) -> tuple[
        list[NavFeature],
        NDArrayFloatType,
        NDArrayFloatType,
        NDArrayFloatType,
    ]:
        """Per-star centroid residuals around the shifted predictions.

        Missing pixels (``missing_mask_ext``) hold 0.0, which is not a measurement.  Where
        any lie within reach of a star's detection, they are read as sky a little darker
        than the pixels around them, so they add no light to the centroid and do not pull
        the background level down to zero (:meth:`_star_image`).  A star whose detection
        then loses too much of its light to them is dropped (:meth:`_missing_shift_px`).
        """
        inliers: list[NavFeature] = []
        residuals_v: list[float] = []
        residuals_u: list[float] = []
        weights: list[float] = []
        filled = image_ext.copy() if missing_mask_ext.any() else image_ext
        for star in stars:
            assert isinstance(star.geometry, StarGeometry)
            assert isinstance(star.flags, StarFlags)
            pred_v, pred_u = star.geometry.predicted_vu
            shifted_pred = (pred_v + prior_v, pred_u + prior_u)
            star_image = self._star_image(image_ext, filled, missing_mask_ext, shifted_pred)
            if star_image is None:
                self.logger.debug(
                    'Star %s: every pixel around the prediction is missing; dropped',
                    star.feature_id,
                )
                continue
            det, peak = local_centroid(
                star_image,
                shifted_pred,
                search_half_vu=(self.tuning.refine_window_px, self.tuning.refine_window_px),
                centroid_box_half_px=self.tuning.centroid_box_half_px,
                psf_sigma_px=self.tuning.psf_sigma_px,
                max_peak_sharpness=self.tuning.max_peak_sharpness,
                image_noise_sigma=noise_sigma,
                detection_sigma=self.tuning.detection_sigma,
                refuse_outshone=True,
            )
            if det is None:
                self.logger.debug(
                    'Star %s: no peak above %.2f sigma in refine window (peak %.3f DN); dropped',
                    star.feature_id,
                    self.tuning.detection_sigma,
                    peak,
                )
                continue
            missing_shift = self._missing_shift_px(det, image_ext, missing_mask_ext)
            if missing_shift > self.tuning.max_missing_shift_px:
                self.logger.debug(
                    'Star %s: missing pixels in its centroid box would move its centroid '
                    '%.3f px, over max %.3f px; dropped',
                    star.feature_id,
                    missing_shift,
                    self.tuning.max_missing_shift_px,
                )
                continue
            res_v = det[0] - shifted_pred[0]
            res_u = det[1] - shifted_pred[1]
            distance = math.hypot(res_v, res_u)
            if distance > self.tuning.max_per_star_residual_px:
                self.logger.debug(
                    'Star %s: refine residual %.3f px exceeds max %.3f px; dropped',
                    star.feature_id,
                    distance,
                    self.tuning.max_per_star_residual_px,
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

    def _star_image(
        self,
        image_ext: NDArrayFloatType,
        filled: NDArrayFloatType,
        missing_mask_ext: NDArrayBoolType,
        shifted_pred: tuple[float, float],
    ) -> NDArrayFloatType | None:
        """Return the image to centroid one star in, with its missing pixels as background.

        The star's detection reads the refine window around ``shifted_pred`` and
        ``2 * centroid_box_half_px`` pixels beyond it on each side (the matched filter's
        reach and the centroid box's).  When none of those pixels is missing the image is
        returned as it is.  Otherwise ``filled``, a copy of the image, is returned with
        the missing pixels there set to the ``_MISSING_FILL_QUANTILE`` quantile of the
        others, so they read as sky a little below the background the centroid
        subtracts.

        Parameters:
            image_ext: The extfov image.
            filled: The working copy of ``image_ext`` to write the fill into.
            missing_mask_ext: ``True`` where a pixel is missing.
            shifted_pred: ``(v, u)`` prediction shifted by the prior offset.

        Returns:
            ``image_ext``, the filled copy, or ``None`` when every pixel the detection
            reads is missing.
        """
        window = (self.tuning.refine_window_px, self.tuning.refine_window_px)
        bounds = search_window_bounds(shifted_pred, window, image_ext.shape)
        if bounds is None:
            return image_ext
        reach = 2 * self.tuning.centroid_box_half_px
        h, w = image_ext.shape
        v_lo, v_hi = max(0, bounds[0] - reach), min(h, bounds[1] + reach)
        u_lo, u_hi = max(0, bounds[2] - reach), min(w, bounds[3] + reach)
        lost = missing_mask_ext[v_lo:v_hi, u_lo:u_hi]
        if not lost.any():
            return image_ext
        present = image_ext[v_lo:v_hi, u_lo:u_hi][~lost]
        if present.size == 0:
            return None
        filled[v_lo:v_hi, u_lo:u_hi][lost] = float(np.quantile(present, _MISSING_FILL_QUANTILE))
        return filled

    def _missing_shift_px(
        self,
        det: tuple[float, float],
        image_ext: NDArrayFloatType,
        missing_mask_ext: NDArrayBoolType,
    ) -> float:
        """Return how far the missing pixels would move the centroid of the star.

        The box is the ``2 * centroid_box_half_px + 1`` pixels around the pixel that
        holds the measured centroid ``det``.  The model of the star's light is the box's
        light above the median of its present pixels, and a missing pixel takes the
        light at its mirror image through the centroid, interpolated between the present
        pixels, with a missing pixel counted as dark.  A star is close to symmetric about
        its center, so the model holds about the light the missing pixels lost, whatever
        the star's width, smear, or saturation.  The shift is that of
        :func:`missing_light_shift_px`.

        The mirror cannot see light the star's core lost.  When the core falls on missing
        pixels, the centroid of what is left moves away from them, and the mirror of a
        missing pixel through that centroid lands on the star's faint flank.  So when a
        missing pixel touches the brightest present pixel of the box, edge or corner, the
        core may be missing and the shift is ``math.inf``.

        Parameters:
            det: ``(v, u)`` measured centroid.
            image_ext: The extfov image.
            missing_mask_ext: ``True`` where a pixel is missing.

        Returns:
            The shift in pixels; ``math.inf`` when every pixel of the box is missing or a
            missing pixel touches the brightest present one.
        """
        half = self.tuning.centroid_box_half_px
        center_v = math.floor(det[0] + 0.5)
        center_u = math.floor(det[1] + 0.5)
        h, w = image_ext.shape
        vs = np.arange(center_v - half, center_v + half + 1)
        us = np.arange(center_u - half, center_u + half + 1)
        rows, cols = np.ix_(np.clip(vs, 0, h - 1), np.clip(us, 0, w - 1))
        lost = missing_mask_ext[rows, cols]
        if not lost.any():
            return 0.0
        inside = ((vs >= 0) & (vs < h))[:, None] & ((us >= 0) & (us < w))[None, :]
        present = inside & ~lost
        if not present.any():
            return math.inf
        box = np.asarray(image_ext[rows, cols], np.float64)
        box_peak = np.unravel_index(int(np.argmax(np.where(present, box, -np.inf))), box.shape)
        peak_v, peak_u = int(vs[box_peak[0]]), int(us[box_peak[1]])
        if missing_mask_ext[
            max(0, peak_v - 1) : min(h, peak_v + 2), max(0, peak_u - 1) : min(w, peak_u + 2)
        ].any():
            return math.inf
        light = np.where(present, np.clip(box - float(np.median(box[present])), 0.0, None), 0.0)
        # Each pixel's mirror through the centroid, in box coordinates; the light there is
        # interpolated between the box's present pixels.
        mirror_v = 2.0 * (det[0] - (center_v - half)) - np.arange(2 * half + 1)
        mirror_u = 2.0 * (det[1] - (center_u - half)) - np.arange(2 * half + 1)
        grid_v, grid_u = np.meshgrid(mirror_v, mirror_u, indexing='ij')
        mirrored = map_coordinates(light, [grid_v, grid_u], order=1, mode='constant')
        model = np.where(lost, mirrored, light)
        return missing_light_shift_px(model, (center_v - half, center_u - half), missing_mask_ext)

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
