"""``StarUniqueMatchNav`` — translation fit when 1 or 2 stars are uniquely matched.

Two paths share one technique:

- **One-star path.**  When the catalog reduction produces one star whose
  predicted brightness is at least ``brightness_margin_to_next_catalog_star_mag``
  brighter than the next-brightest predictable star, the brightest
  detection inside its search window is unambiguously its match.  The
  offset is the centroid minus the prediction; confidence is capped at
  the configured one-star limit (default 0.7) because a single match
  cannot cross-check itself.

  No-rival sentinels: the ``one_star_min_peak_ratio`` ambiguity
  gate and the brightness-margin gate both report ``inf`` when
  no rival exists (no runner-up detection above the window background /
  no other predictable catalog star).  The sentinels deliberately PASS
  the ratio checks -- a genuinely unique bright star has no rival and
  must remain matchable -- but an infinite peak ratio also means the
  ambiguity gate measured nothing (a flat or quantized window, exactly
  where a lone hot pixel or artifact would otherwise auto-pass every
  check).  In that vacuous case acceptance is additionally gated on the
  prediction-to-detection distance staying within
  ``one_star_max_residual_px``: with no rival statistics to lean on, a
  lone detection is only promoted to an identification when it sits
  inside the pointing-prior core.  A finite peak ratio leaves
  acceptance to the measured ambiguity gate; genuine one-star matches with
  offsets up to ~24 px exist in the operator-verified library, so no
  uniform residual cut below the search window is possible.

- **Two-star path.**  With two predictable stars, the technique tries
  both detection-to-prediction assignments and picks the one whose
  joint residual is smaller.  The residual cross-checks the assignment;
  confidence is capped at the configured two-star limit (default 0.8).

Local-window detection is intentional: the search window is sized to
the per-instrument SPICE pointing-error envelope, so a brightest peak
inside the window is the matched detection.  No global star-detection
pass is needed (and this technique is feasible even on images where
multi-star detection would fail because the rest of the field is too
faint or too crowded).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique._star_helpers import (
    SimilarityFit,
    brightness_margin_mag,
    detection_peak_ratio,
    local_centroid,
    predicted_snr,
    similarity_transform_fit,
    usable_stars,
)
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import StarUniqueMatchDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    embed_rotation_unobservable,
    log_confidence_breakdown,
    reported_position_vu,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['StarUniqueMatchNav']


# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.StarUniqueMatchNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


def _build_covariance(feature: NavFeature, *, residual_px: float) -> NDArrayFloatType:
    """Return a 2x2 covariance for the star match.

    Uses the per-feature anisotropic CRLB covariance carried on the
    ``NavFeature.position_cov_px`` (computed from predicted SNR + smear
    by ``NavModelStars``) as the floor and inflates it by
    ``residual_px**2`` so a noisy match honestly reports its scatter
    rather than the noise-free CRLB lower bound.

    The same inflation applies in both 1-star and 2-star modes: the
    1-star path's ``residual_px`` is the prediction-to-detection
    distance (no second observation to subtract), and the 2-star path's
    ``residual_px`` is already a per-axis RMS, so squaring either
    yields the right per-axis variance addend.

    Parameters:
        feature: The star feature whose CRLB cov is the floor.
        residual_px: Final per-axis residual magnitude.

    Returns:
        2x2 float64 covariance matrix.
    """
    floor = feature.position_cov_px
    base = (
        np.eye(2, dtype=np.float64) if floor is None else np.asarray(floor, dtype=np.float64).copy()
    )
    inflation = residual_px * residual_px
    return base + inflation * np.eye(2, dtype=np.float64)


class _UniqueMatchConfidenceContext:
    """Adapter binding ``StarUniqueMatchDiagnostics`` plus side flags."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: StarUniqueMatchDiagnostics,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.predicted_snr = diagnostics.predicted_snr
        self.brightness_margin_mag = diagnostics.brightness_margin_mag
        self.residual_px = diagnostics.residual_px
        # Note: ``mode`` is a string, so it is intentionally not surfaced
        # to the confidence formula (no numeric meaning).  Mode-driven
        # caps are applied inside ``StarUniqueMatchNav.navigate``.


class StarUniqueMatchNav(NavTechnique):
    """Star unique-match translation fit (1- or 2-star path).

    Class attributes:
        accepts_feature_types: ``frozenset({STAR})``.
        requires_prior: ``False`` — runs in pass 1.
    """

    name = 'StarUniqueMatchNav'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'predicted_snr',
            'brightness_margin_mag',
            'residual_px',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._brightness_margin_floor_mag = float(
            self.tuning['brightness_margin_to_next_catalog_star_mag']
        )
        self._search_window_px = float(self.tuning['search_window_px'])
        self._centroid_box_half_px = int(self.tuning['centroid_box_half_px'])
        self._max_residual_px = float(self.tuning['max_residual_px'])
        self._detection_sigma = float(self.tuning['detection_sigma'])
        self._one_star_confidence_cap = float(self.tuning['one_star_confidence_cap'])
        self._two_star_confidence_cap = float(self.tuning['two_star_confidence_cap'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        self._one_star_min_peak_ratio = float(self.tuning['one_star_min_peak_ratio'])
        self._one_star_max_residual_px = float(self.tuning['one_star_max_residual_px'])
        if self._one_star_min_peak_ratio < 1.0:
            raise ValueError(
                f'one_star_min_peak_ratio must be >= 1; got {self._one_star_min_peak_ratio!r}'
            )
        if self._one_star_max_residual_px <= 0.0:
            raise ValueError(
                f'one_star_max_residual_px must be > 0; got {self._one_star_max_residual_px!r}'
            )
        if not 0.0 <= self._one_star_confidence_cap <= 1.0:
            raise ValueError(
                f'one_star_confidence_cap must lie in [0, 1]; got {self._one_star_confidence_cap!r}'
            )
        if not 0.0 <= self._two_star_confidence_cap <= 1.0:
            raise ValueError(
                f'two_star_confidence_cap must lie in [0, 1]; got {self._two_star_confidence_cap!r}'
            )

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Report feasibility based on the predictable-star cohort.

        Reads only feature metadata; never any pixels.  Feasible when at
        least one usable STAR feature is in the input set; the navigate
        path then decides whether the 1-star or 2-star branch wins.
        """
        usable = usable_stars(features)
        if not usable:
            return NavFeasibilityReport(feasible=False, reason='no_usable_star_features')
        consumed = min(len(usable), 2)
        return NavFeasibilityReport(feasible=True, reason='ok', consumed_feature_count=consumed)

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Recover translation from the 1- or 2-star uniquely-bright match.

        Parameters:
            features: STAR features (the orchestrator pre-filters by type).
            context: Per-image NavContext.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence (capped per-mode by
            ``one_star_confidence_cap`` / ``two_star_confidence_cap``),
            and a populated :class:`StarUniqueMatchDiagnostics`.  The
            covariance shape and rotation fields depend on the chosen
            path and on ``context.fit_camera_rotation``:

            - **1-star path, fit_camera_rotation=False**: ``(2, 2)``
              covariance; ``rotation_rad`` and ``sigma_rotation_rad``
              are ``None``.
            - **1-star path, fit_camera_rotation=True**: rank-deficient
              ``(3, 3)`` covariance via
              :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`
              (a single match cannot constrain rotation);
              ``rotation_rad = 0.0`` and ``sigma_rotation_rad`` is the
              rotation-unobservable sentinel.
            - **2-star path, fit_camera_rotation=False**: ``(2, 2)``
              covariance from the per-feature CRLB floor; rotation
              fields ``None``.
            - **2-star path, fit_camera_rotation=True**: full ``(3, 3)``
              covariance with the analytic
              ``2 * (sigma_v**2 + sigma_u**2) / L**2`` rotation
              diagonal (``L`` is the catalog-pair separation);
              ``rotation_rad`` is the Procrustes-fit angle and
              ``sigma_rotation_rad`` is the square root of the rotation
              diagonal.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            usable = usable_stars(features)
            self.logger.info(
                'Consuming %d usable STAR feature(s) (out of %d offered)',
                len(usable),
                len(features),
            )
            if not usable:
                return self._fail(
                    features=features,
                    diagnostics=StarUniqueMatchDiagnostics(
                        mode='',
                        predicted_snr=0.0,
                        brightness_margin_mag=0.0,
                        residual_px=0.0,
                    ),
                    reason='no_usable_star_features',
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            ranked = sorted(usable, key=predicted_snr, reverse=True)
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            if len(ranked) >= 2:
                two_star_result = self._try_two_star(
                    ranked[:2], ranked[2:], image_ext, noise_sigma, context
                )
                if two_star_result is not None:
                    return two_star_result
            return self._try_one_star(ranked[0], ranked[1:], image_ext, noise_sigma, context)

    def _try_two_star(
        self,
        chosen: list[NavFeature],
        rest: list[NavFeature],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        context: NavContext,
    ) -> NavTechniqueResult | None:
        """Attempt the 2-star path; return ``None`` when both detections fail."""
        det_a, peak_a = local_centroid(
            image_ext,
            chosen[0].geometry.predicted_vu,  # type: ignore[union-attr]
            search_window_px=self._search_window_px,
            centroid_box_half_px=self._centroid_box_half_px,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
        )
        det_b, peak_b = local_centroid(
            image_ext,
            chosen[1].geometry.predicted_vu,  # type: ignore[union-attr]
            search_window_px=self._search_window_px,
            centroid_box_half_px=self._centroid_box_half_px,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
        )
        if det_a is None or det_b is None:
            self.logger.debug(
                'Two-star path: at least one of the two predictions had no '
                'usable detection (peak_a=%.3f DN, peak_b=%.3f DN); falling '
                'back to one-star path',
                peak_a,
                peak_b,
            )
            return None
        # Overlapping search windows can return the SAME detection for
        # both predictions (one bright peak shared between the
        # ``[predicted - W, predicted + W]`` slabs of two close
        # predictions).  Without this guard the 2-star math would fit
        # both catalog stars to the same observation, fabricate a
        # zero-residual cross-check, and report high confidence on a
        # wrong offset.  Two centroids less than 1 pixel apart almost
        # certainly came from the same matched-filter peak — fall
        # back to the one-star path so the brightness-margin gate
        # gets a chance to reject the ambiguous match.
        det_separation_px = math.hypot(det_a[0] - det_b[0], det_a[1] - det_b[1])
        if det_separation_px < 1.0:
            self.logger.debug(
                'Two-star path: both predictions resolved to the same '
                'detection (separation %.4f px < 1.0); falling back to '
                'one-star path',
                det_separation_px,
            )
            return None
        pred_a = chosen[0].geometry.predicted_vu  # type: ignore[union-attr]
        pred_b = chosen[1].geometry.predicted_vu  # type: ignore[union-attr]
        fit_rotation = bool(context.fit_camera_rotation)
        det_arr = np.asarray([det_a, det_b], dtype=np.float64)
        if fit_rotation:
            # Procrustes-style fit per assignment: each candidate pairs
            # detections to predictions and solves the minimum-residual
            # rigid (rotation + translation) fit.  With only two points
            # the rigid fit is exact, so the residual reduces to the
            # zero-mean component the unweighted-translation 2-DoF
            # fallback already used for assignment selection.
            sim_1, residual_1 = self._similarity_fit_assignment(det_arr, pred_a, pred_b, swap=False)
            sim_2, residual_2 = self._similarity_fit_assignment(det_arr, pred_a, pred_b, swap=True)
            if residual_1 <= residual_2:
                sim_chosen = sim_1
                residual_px = residual_1
                assignment = 'a->1,b->2'
            else:
                sim_chosen = sim_2
                residual_px = residual_2
                assignment = 'a->2,b->1'
            offset_v, offset_u = sim_chosen.translation_vu
            rotation_rad: float | None = float(sim_chosen.rotation_rad)
        else:
            # Assignment 1: det_a -> pred_a, det_b -> pred_b.
            offset_1_v = 0.5 * ((det_a[0] - pred_a[0]) + (det_b[0] - pred_b[0]))
            offset_1_u = 0.5 * ((det_a[1] - pred_a[1]) + (det_b[1] - pred_b[1]))
            residual_1 = math.hypot(
                (det_a[0] - pred_a[0]) - offset_1_v,
                (det_a[1] - pred_a[1]) - offset_1_u,
            )
            # Assignment 2: det_a -> pred_b, det_b -> pred_a.
            offset_2_v = 0.5 * ((det_a[0] - pred_b[0]) + (det_b[0] - pred_a[0]))
            offset_2_u = 0.5 * ((det_a[1] - pred_b[1]) + (det_b[1] - pred_a[1]))
            residual_2 = math.hypot(
                (det_a[0] - pred_b[0]) - offset_2_v,
                (det_a[1] - pred_b[1]) - offset_2_u,
            )
            if residual_1 <= residual_2:
                offset_v, offset_u, residual_px = offset_1_v, offset_1_u, residual_1
                assignment = 'a->1,b->2'
            else:
                offset_v, offset_u, residual_px = offset_2_v, offset_2_u, residual_2
                assignment = 'a->2,b->1'
            sim_chosen = None
            rotation_rad = None
        if residual_px > self._max_residual_px:
            self.logger.info(
                'Two-star path: best-assignment residual %.3f px exceeds '
                'max_residual_px %.3f; falling back to one-star path',
                residual_px,
                self._max_residual_px,
            )
            return None
        # Brightness-margin diagnostic: magnitude difference to the
        # next-brightest *unmatched* predictable star.  When only two
        # predictable stars exist (no third star in ``rest``) the
        # diagnostic reports ``+inf`` — there is no unmatched star to
        # compare against, which is the strongest possible margin.
        brightest_snr = predicted_snr(chosen[0])
        next_snr = predicted_snr(rest[0]) if rest else 0.0
        margin_mag = brightness_margin_mag(brightest_snr, next_snr)
        diagnostics = StarUniqueMatchDiagnostics(
            mode='two_star',
            predicted_snr=brightest_snr,
            brightness_margin_mag=margin_mag,
            residual_px=residual_px,
        )
        margin_v, margin_u = search_window_for_obs(context)
        max_rotation_rad = math.radians(context.max_rotation_deg)
        rotation_at_edge = fit_rotation and (
            rotation_rad is not None
            and abs(rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
        )
        at_edge = (
            abs(offset_v) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_u) >= margin_u - self._at_edge_tolerance_px
            or rotation_at_edge
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics,
            at_edge=at_edge,
            spurious=False,
            cap=self._two_star_confidence_cap,
        )
        self.logger.info(
            'Two-star path: assignment %s; offset (%.4f, %.4f) px; residual '
            '%.4f px; brightness margin %.3f mag; confidence %.4f',
            assignment,
            offset_v,
            offset_u,
            residual_px,
            margin_mag,
            confidence,
        )
        feature_ids = (chosen[0].feature_id, chosen[1].feature_id)
        # Average the two stars' covariances as a coarse 2-star floor.
        cov_a = _build_covariance(chosen[0], residual_px=residual_px)
        cov_b = _build_covariance(chosen[1], residual_px=residual_px)
        cov_2x2 = 0.5 * (cov_a + cov_b)
        cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
        sigma_rotation_rad: float | None
        if fit_rotation:
            cov = self._build_two_star_covariance_3dof(
                cov_2x2=cov_2x2,
                pred_a=pred_a,
                pred_b=pred_b,
            )
            sigma_rotation_rad = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
            self.logger.info(
                'Rotation = %+.4f deg (sigma %.4f deg)%s',
                math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                math.degrees(sigma_rotation_rad),
                ', AT_EDGE' if rotation_at_edge else '',
            )
        else:
            cov = cov_2x2
            sigma_rotation_rad = None
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=feature_ids,
            offset_px=(offset_v, offset_u),
            covariance_px2=cov,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=rotation_rad,
            sigma_rotation_rad=sigma_rotation_rad,
        )

    @staticmethod
    def _similarity_fit_assignment(
        det_arr: NDArrayFloatType,
        pred_a: tuple[float, float],
        pred_b: tuple[float, float],
        *,
        swap: bool,
    ) -> tuple[SimilarityFit, float]:
        """Run a single-assignment Procrustes fit and report its residual.

        Parameters:
            det_arr: Ordered detection pair ``[det_a, det_b]`` shaped
                ``(2, 2)``.
            pred_a: Catalog prediction for the first chosen star.
            pred_b: Catalog prediction for the second chosen star.
            swap: When False the catalog cohort is ``[pred_a, pred_b]``;
                when True the cohort is ``[pred_b, pred_a]`` (the
                swapped detection-to-prediction assignment).

        Returns:
            ``(SimilarityFit, residual_px)`` where the residual is the
            centroid-relative scalar used to compare assignments.
            For a two-point rigid fit the Procrustes residuals are zero
            by construction; the comparable scalar — the same one the
            legacy translation-only path used — is computed directly
            from the centroid-relative offset so the assignment-
            selection criterion stays stable across legacy and 3-DoF
            runs.
        """
        if swap:
            cat_arr = np.asarray([pred_b, pred_a], dtype=np.float64)
        else:
            cat_arr = np.asarray([pred_a, pred_b], dtype=np.float64)
        weights = np.ones(2, dtype=np.float64)
        fit = similarity_transform_fit(det_arr, cat_arr, weights)
        # Residuals from a 2-point rigid fit are zero by construction; the
        # comparable scalar for the legacy assignment-selection logic is
        # the centroid-relative residual prior to rotation, computed
        # directly from the assignment to keep the selection criterion
        # stable across legacy / 3-DoF runs.
        det_a = det_arr[0]
        cat_first = cat_arr[0]
        offset = np.mean(det_arr - cat_arr, axis=0)
        residual = math.hypot(
            (det_a[0] - cat_first[0]) - float(offset[0]),
            (det_a[1] - cat_first[1]) - float(offset[1]),
        )
        return fit, residual

    @staticmethod
    def _build_two_star_covariance_3dof(
        *,
        cov_2x2: NDArrayFloatType,
        pred_a: tuple[float, float],
        pred_b: tuple[float, float],
    ) -> NDArrayFloatType:
        """Build the 3x3 covariance for a 2-star Procrustes fit.

        Translation block uses the per-feature CRLB-derived 2x2; the
        rotation diagonal scales the sum of per-axis position variances
        by ``1 / L**2`` where ``L`` is the catalog separation between
        the two stars.  This is the analytic small-angle uncertainty:
        a per-axis position error ``sigma_pos`` perturbs the rotation
        by approximately ``sigma_pos / L`` per star, and the two stars
        contribute independently.

        Parameters:
            cov_2x2: Per-feature CRLB-derived 2x2 translation covariance
                (already symmetrised).
            pred_a: Catalog prediction for the first chosen star.
            pred_b: Catalog prediction for the second chosen star.

        Returns:
            3x3 covariance with the input ``cov_2x2`` in the
            translation block and either the analytic
            ``2 * (sigma_v**2 + sigma_u**2) / L**2`` rotation variance
            on the diagonal, or :data:`ROTATION_UNOBSERVABLE_VARIANCE`
            when the two predictions coincide (a degenerate input the
            upstream guard already rejects).
        """
        separation_sq = (pred_a[0] - pred_b[0]) ** 2 + (pred_a[1] - pred_b[1]) ** 2
        if separation_sq <= 0.0:
            sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
        else:
            # Sum of per-axis position variances aggregates the (v, u)
            # pixel uncertainty assumed isotropic by the centroid CRLB;
            # 2 / L^2 reflects the two-point lever-arm.
            sigma_theta_sq = 2.0 * float(cov_2x2[0, 0] + cov_2x2[1, 1]) / separation_sq
        cov = np.zeros((3, 3), dtype=np.float64)
        cov[:2, :2] = cov_2x2
        cov[2, 2] = sigma_theta_sq
        return cov

    def _try_one_star(
        self,
        brightest: NavFeature,
        rest: list[NavFeature],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        context: NavContext,
    ) -> NavTechniqueResult:
        """Attempt the 1-star path with its acceptance gates.

        Acceptance requires, in order: the brightness-margin uniqueness
        gate, a detection above the noise threshold, the
        peak-to-runner-up ambiguity gate, and -- only when that
        ratio is the infinite no-rival sentinel -- the residual gate
        bounding the prediction-to-detection distance by
        ``one_star_max_residual_px``.  The sentinel means the ambiguity
        gate measured nothing (no runner-up ever cleared the window
        background), so the residual gate carries the acceptance burden
        there; a finite ratio was measured against real background
        statistics and acceptance stays with the ambiguity gate.
        """
        brightest_snr = predicted_snr(brightest)
        next_snr = predicted_snr(rest[0]) if rest else 0.0
        margin_mag = brightness_margin_mag(brightest_snr, next_snr)
        if margin_mag < self._brightness_margin_floor_mag:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
            )
            return self._fail(
                features=[brightest, *rest],
                diagnostics=diagnostics,
                reason=(
                    f'brightness_margin {margin_mag:.3f} mag below floor '
                    f'{self._brightness_margin_floor_mag:.3f} mag'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        det, peak = local_centroid(
            image_ext,
            brightest.geometry.predicted_vu,  # type: ignore[union-attr]
            search_window_px=self._search_window_px,
            centroid_box_half_px=self._centroid_box_half_px,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
        )
        if det is None:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=f'no_detection_above_threshold (peak {peak:.3f} DN)',
                fit_rotation=bool(context.fit_camera_rotation),
            )
        pred_v, pred_u = brightest.geometry.predicted_vu  # type: ignore[union-attr]
        # A detection that barely beats the window's noise order
        # statistics carries no positional information even though it
        # cleared the absolute detection threshold: with a marginal
        # star, the brightest pixel of a multi-thousand-pixel window is
        # routinely a noise spike within a few percent of its
        # runner-up.  The single-detection premise of this path demands
        # the peak be unambiguous, so an ambiguous one is spurious.
        peak_ratio = detection_peak_ratio(
            image_ext,
            (pred_v, pred_u),
            det,
            search_window_px=self._search_window_px,
            exclude_half_px=self._centroid_box_half_px + 2,
        )
        if peak_ratio < self._one_star_min_peak_ratio:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
                detection_peak_ratio=peak_ratio,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=(
                    f'ambiguous_detection (peak/runner-up {peak_ratio:.3f} below '
                    f'{self._one_star_min_peak_ratio:.3f})'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        offset_v = det[0] - pred_v
        offset_u = det[1] - pred_u
        residual_px = math.hypot(offset_v, offset_u)
        # Residual gate (#259), applied only when the #211 ambiguity
        # gate is vacuous: an infinite peak ratio means no runner-up
        # ever cleared the window background, so the gate measured
        # nothing about the detection's uniqueness -- the signature of
        # a lone hot pixel / artifact on a flat or quantized frame, not
        # of a star on a real noisy background (whose runner-up is
        # always finite).  With no rival statistics to lean on, and the
        # residual being the claimed offset itself, acceptance demands
        # the detection sit inside the pointing-prior core bounded by
        # ``one_star_max_residual_px``.  A finite ratio keeps the
        # measured #211 gate authoritative: genuine library matches
        # carry one-star offsets up to ~24 px, so no uniform residual
        # cut below the search window is possible.
        if not math.isfinite(peak_ratio) and residual_px > self._one_star_max_residual_px:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=residual_px,
                detection_peak_ratio=peak_ratio,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=(
                    f'no_rival_detection: residual {residual_px:.3f} px exceeds '
                    f'one_star_max_residual_px {self._one_star_max_residual_px:.3f} '
                    f'and the peak-ratio gate is vacuous (no runner-up above background)'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        # Even though "residual_px" is really the offset magnitude here
        # (the 1-star path has no second observation to subtract), it is
        # the only honest measure of how far we travelled to find the
        # detection inside the search window.  The technique reports it
        # so the confidence formula can penalize large excursions.
        diagnostics = StarUniqueMatchDiagnostics(
            mode='one_star',
            predicted_snr=brightest_snr,
            brightness_margin_mag=margin_mag,
            residual_px=residual_px,
            detection_peak_ratio=peak_ratio,
        )
        margin_v, margin_u = search_window_for_obs(context)
        at_edge = (
            abs(offset_v) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_u) >= margin_u - self._at_edge_tolerance_px
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics,
            at_edge=at_edge,
            spurious=False,
            cap=self._one_star_confidence_cap,
        )
        reported_det = reported_position_vu(det, (margin_v, margin_u))
        self.logger.info(
            'One-star path: matched %s at (%.4f, %.4f); offset (%.4f, %.4f) '
            'px; brightness margin %.3f mag; confidence %.4f',
            brightest.feature_id,
            reported_det[0],
            reported_det[1],
            offset_v,
            offset_u,
            margin_mag,
            confidence,
        )
        cov = _build_covariance(brightest, residual_px=residual_px)
        fit_rotation = bool(context.fit_camera_rotation)
        cov_out = embed_rotation_unobservable(cov) if fit_rotation else cov
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(brightest.feature_id,),
            offset_px=(offset_v, offset_u),
            covariance_px2=cov_out,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
        )

    def _evaluate_confidence(
        self,
        *,
        diagnostics: StarUniqueMatchDiagnostics,
        at_edge: bool,
        spurious: bool,
        cap: float,
    ) -> float:
        """Run the YAML confidence formula and apply the per-mode cap."""
        assert self.confidence_spec is not None
        confidence, breakdown = evaluate_sigmoid_combination(
            self.confidence_spec,
            _UniqueMatchConfidenceContext(
                at_edge=at_edge, spurious=spurious, diagnostics=diagnostics
            ),
            technique_name=self.name,
            return_breakdown=True,
        )
        log_confidence_breakdown(self.logger, breakdown)
        return min(float(confidence), cap)

    def _fail(
        self,
        *,
        features: list[NavFeature],
        diagnostics: StarUniqueMatchDiagnostics,
        reason: str,
        fit_rotation: bool = False,
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result with the supplied reason."""
        self.logger.info('Reporting spurious result: %s', reason)
        return self._spurious_result(
            feature_ids=tuple(f.feature_id for f in features),
            diagnostics=diagnostics,
            fit_rotation=fit_rotation,
        )
