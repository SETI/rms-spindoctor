"""``TitanHazeNav`` -- pointing offset from a hazy body's solar symmetry.

Absent clouds or visible surface features, a hazy atmosphere is
mirror-symmetric about the image-plane line through the body center and the
sub-solar point.  The image displacement perpendicular to that line
("cross-track") is therefore the shift that maximizes mirror symmetry, and
because the limb arc facing the sub-solar point is close to circular, a
circle fit with FREE radius to that arc pins the displacement along the line
("along-track") without assuming a haze altitude.  Together the two
constraints give a full ``(dv, du)`` offset whose uncertainty is honestly
anisotropic: the cross-track direction is far better determined than the
along-track one.

The technique is a thin wrapper: every array operation lives in
:mod:`spindoctor.nav_technique.titan_fitting`, which is pure and exercisable
on synthetic images, while this module supplies the tuning constants, turns
the fit into a :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`,
and reports the gate that rejected a frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, body_names_from_features
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import TitanHazeGeometry
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import TitanHazeDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    add_model_error_floor,
    embed_rotation_unobservable,
    load_model_error_floor,
    log_confidence_breakdown,
    reported_position_vu,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.nav_technique.titan_fitting import (
    ARC_RADIUS_MAX_FRACTION,
    ARC_RADIUS_MIN_FRACTION,
    ArcFitParams,
    ArcFitResult,
    SymmetryFitParams,
    SymmetryFitResult,
    axis_vectors,
    fit_titan_center,
)
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['TitanHazeNav']


_NO_FEATURES_REASON: str = 'no TITAN_LIMB features'
"""Infeasibility reason when the input set carries no haze feature."""


_UNDEFINED_RESIDUAL_SENTINEL: float = 1.0e9
"""Residual RMS substituted when the arc fit left the true value undefined.

Large enough to saturate the falling confidence term at zero contribution,
so an undefined residual is priced as the worst possible fit rather than as
the perfect one a bare zero would imply.  Unreachable in practice: the
``arc_inliers`` gate rejects the only case that produces it.
"""


_UNDEFINED_QUALITY_SENTINEL: float = 0.0
"""Quality score substituted when the symmetry scan left it undefined.

Zero is the honest floor for a rising quality term -- no measured symmetry
means no evidence.  Unreachable in practice: the ``peak_score`` gate rejects
the only case that produces it.
"""


_GATE_TABLE_FORMAT: str = '  %-14s %13s  %-24s %s'
"""Column layout of the per-gate table logged inside the technique section."""


_GATE_NOT_REACHED: str = '-'
"""Measured-value placeholder for a gate the fit never reached.

The arc fit returns as soon as too few rays survive, so the gates after
``ray_yield`` have nothing measured behind them; reporting their defaulted
values as if they had been evaluated would invent evidence.
"""


def _gate_verdict(passed: bool) -> str:
    """Render a gate outcome as the table's result column."""
    return 'PASS' if passed else 'FAIL'


def _gate_value(value: float, *, digits: int = 4) -> str:
    """Render a measured quantity, or ``n/a`` when the fit left it undefined."""
    if not math.isfinite(value):
        return 'n/a'
    return f'{value:.{digits}f}'


def _symmetry_gate_rows(
    symmetry: SymmetryFitResult, *, params: SymmetryFitParams, window_px: float
) -> list[tuple[str, str, str, str]]:
    """Build the ``(gate, measured, threshold, result)`` rows of the cross-track fit.

    Every predicate mirrors the fitting library's own comparison, negations
    included, so a NaN measurement lands on the same side of the table as it
    does in the gate.  The scan measures all three quantities before gating
    any of them, so no cross-track row can be unreached.

    Parameters:
        symmetry: The final pass's cross-track fit.
        params: The cross-track tuning constants the gates compare against.
        window_px: Search half-window, the bound behind the at-edge flag.

    Returns:
        One row per Section-2.2 gate, in the order the library evaluates
        them.
    """
    # The at-edge row is the one place the verdict is not a comparison of the
    # two columns beside it: the scan raises the flag on the winning INTEGER
    # shift against the truncated window, then refines the reported shift to
    # sub-pixel.  The verdict column is authoritative; the measured column is
    # the reported shift, and the threshold column is the bound the flag was
    # actually tested against, so the two can straddle each other by under a
    # pixel on a frame flagged at the edge.
    return [
        (
            'valid_fraction',
            _gate_value(symmetry.valid_fraction),
            f'>= {params.min_valid_fraction:.4f}',
            _gate_verdict(not symmetry.valid_fraction < params.min_valid_fraction),
        ),
        (
            'peak_score',
            _gate_value(symmetry.peak_score),
            f'>= {params.min_peak_score:.4f}',
            _gate_verdict(symmetry.peak_score >= params.min_peak_score),
        ),
        (
            'second_peak',
            _gate_value(symmetry.second_peak_ratio),
            f'<= {params.max_second_peak_ratio:.4f}',
            _gate_verdict(not symmetry.second_peak_ratio > params.max_second_peak_ratio),
        ),
        (
            'cross_at_edge',
            _gate_value(abs(symmetry.cross_track_px), digits=2),
            f'< {float(math.floor(window_px)):.2f}',
            'EDGE' if symmetry.at_edge else 'PASS',
        ),
    ]


def _arc_gate_rows(
    arc: ArcFitResult,
    *,
    params: ArcFitParams,
    r_solid_px: float,
    r_env_px: float,
    window_px: float,
) -> list[tuple[str, str, str, str]]:
    """Build the ``(gate, measured, threshold, result)`` rows of the along-track fit.

    Parameters:
        arc: The final pass's along-track fit.
        params: The along-track tuning constants the gates compare against.
        r_solid_px: Solid-body radius, the lower end of the radius band.
        r_env_px: Haze-envelope radius, which with ``window_px`` sets the
            upper end of the radius band.
        window_px: Search half-window, the bound behind the at-edge flag.

    Returns:
        One row per Section-2.3 gate, in the order the library evaluates
        them.  Rows the fit never reached carry ``SKIP`` rather than the
        verdict their defaulted values would imply.
    """
    min_rays = max(params.min_rays, 1)
    inlier_fraction = arc.n_rays_inlier / arc.n_rays_total if arc.n_rays_total > 0 else 0.0
    radius_lo = ARC_RADIUS_MIN_FRACTION * r_solid_px
    radius_hi = ARC_RADIUS_MAX_FRACTION * (r_env_px + window_px)
    rows = [
        (
            'ray_yield',
            f'{arc.n_rays_total:d}',
            f'>= {min_rays:d}',
            _gate_verdict(not arc.n_rays_total < min_rays),
        ),
        (
            'arc_inliers',
            f'{arc.n_rays_inlier:d} ({inlier_fraction:.3f})',
            f'>= {params.min_rays:d} and >= {params.min_inlier_fraction:.3f}',
            _gate_verdict(
                not (
                    arc.n_rays_inlier < params.min_rays
                    or arc.n_rays_inlier < params.min_inlier_fraction * arc.n_rays_total
                )
            ),
        ),
        (
            'arc_radius',
            _gate_value(arc.radius_px, digits=2),
            f'[{radius_lo:.2f}, {radius_hi:.2f}]',
            _gate_verdict(radius_lo <= arc.radius_px <= radius_hi),
        ),
        (
            'arc_residual',
            _gate_value(arc.residual_rms_px, digits=3),
            f'<= {params.max_residual_rms_px:.3f}',
            _gate_verdict(arc.residual_rms_px <= params.max_residual_rms_px),
        ),
        # Unlike the cross-track at-edge row, this one's columns ARE the
        # comparison the fit made: the circle fit flags the shift it reports,
        # against the untruncated window.
        (
            'along_at_edge',
            _gate_value(abs(arc.along_track_px), digits=2),
            f'< {window_px:.2f}',
            'EDGE' if arc.at_edge else 'PASS',
        ),
    ]
    if arc.gate_failed == 'ray_yield':
        rows = rows[:1] + [
            (name, _GATE_NOT_REACHED, threshold, 'SKIP') for name, _, threshold, _ in rows[1:]
        ]
    return rows


def _eligible_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset carrying a ``TITAN_LIMB`` haze geometry payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.TITAN_LIMB and isinstance(f.geometry, TitanHazeGeometry)
    ]


def _symmetry_params(block: dict[str, Any], *, angle_refine_deg: float) -> SymmetryFitParams:
    """Build the cross-track tuning constants from the config block.

    Parameters:
        block: The ``titan.navigation.symmetry`` mapping.
        angle_refine_deg: Half-range of the symmetry-angle search, passed
            separately so the caller can disable refinement on a frame whose
            axis is degenerate.
    """
    return SymmetryFitParams(
        annulus_inner_fraction=float(block['annulus_inner_fraction']),
        annulus_outer_pad_px=float(block['annulus_outer_pad_px']),
        angle_refine_deg=angle_refine_deg,
        angle_refine_step_deg=float(block['angle_refine_step_deg']),
        angle_refine_min_gain=float(block['angle_refine_min_gain']),
        min_peak_score=float(block['min_peak_score']),
        min_valid_fraction=float(block['min_valid_fraction']),
        max_second_peak_ratio=float(block['max_second_peak_ratio']),
        cross_sigma_scale=float(block['cross_sigma_scale']),
        sigma_floor_cross_px=float(block['sigma_floor_cross_px']),
    )


def _arc_params(block: dict[str, Any]) -> ArcFitParams:
    """Build the along-track tuning constants from the ``titan.navigation.arc`` block."""
    return ArcFitParams(
        sector_half_angle_deg=float(block['sector_half_angle_deg']),
        ray_step_deg=float(block['ray_step_deg']),
        radial_step_px=float(block['radial_step_px']),
        radial_inner_fraction=float(block['radial_inner_fraction']),
        radial_outer_pad_px=float(block['radial_outer_pad_px']),
        median_filter_samples=int(block['median_filter_samples']),
        min_gradient_snr=float(block['min_gradient_snr']),
        min_rays=int(block['min_rays']),
        min_inlier_fraction=float(block['min_inlier_fraction']),
        max_residual_rms_px=float(block['max_residual_rms_px']),
        tukey_c=float(block['tukey_c']),
        along_sigma_scale=float(block['along_sigma_scale']),
        sigma_floor_along_px=float(block['sigma_floor_along_px']),
    )


def _rotated_covariance(
    theta_rad: float, sigma_cross_px: float, sigma_along_px: float
) -> NDArrayFloatType:
    """Return the ``(v, u)`` covariance of an axis-aligned sigma pair.

    ``Sigma = M diag(sigma_cross^2, sigma_along^2) M^T`` with ``M``'s columns
    the cross-track and along-track unit vectors expressed in ``(v, u)``.
    The anisotropy is the physical content: the mirror-symmetry scan
    localizes the cross-track direction far more tightly than the limb-arc
    circle fit localizes the along-track one, and an isotropic covariance
    would hand the ensemble a wrong error ellipse in both directions.
    """
    c_hat, a_hat = axis_vectors(theta_rad)
    basis = np.stack([c_hat, a_hat], axis=1)
    diag = np.diag([sigma_cross_px * sigma_cross_px, sigma_along_px * sigma_along_px])
    cov = basis @ diag @ basis.T
    # Symmetrize: the product above is symmetric in exact arithmetic, and
    # NavTechniqueResult validates symmetry to 1e-9, which floating-point
    # round-off in the rotation can otherwise breach on extreme anisotropy.
    symmetric: NDArrayFloatType = 0.5 * (cov + cov.T)
    return symmetric


def _wrap_degrees(angle_deg: float) -> float:
    """Wrap an angle difference into ``(-180, 180]`` degrees."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


class _TitanConfidenceContext:
    """Adapter binding ``TitanHazeDiagnostics`` plus ``at_edge`` for confidence eval.

    The two quality terms that the diagnostics record as optional are
    republished here as plain floats, because the confidence formula only
    ever runs on a result that passed every gate and those gates have
    already rejected the cases that leave either quantity undefined.  Should
    that ever stop holding, the substituted values are
    :data:`_UNDEFINED_QUALITY_SENTINEL` and
    :data:`_UNDEFINED_RESIDUAL_SENTINEL`, which drive the formula to zero
    confidence rather than to a flattering default.
    """

    def __init__(
        self,
        *,
        at_edge: bool,
        diagnostics: TitanHazeDiagnostics,
        symmetry_peak_score: float,
        arc_residual_rms_px: float,
    ) -> None:
        self.at_edge = at_edge
        self.symmetry_peak_score = symmetry_peak_score
        self.symmetry_valid_fraction = diagnostics.symmetry_valid_fraction
        self.arc_inlier_fraction = diagnostics.arc_inlier_fraction
        self.arc_residual_rms_px = arc_residual_rms_px
        self.envelope_diameter_px = diagnostics.envelope_diameter_px


class TitanHazeNav(NavTechnique):
    """Haze-symmetry translation fit for a body with an opaque atmosphere.

    Class attributes:
        accepts_feature_types: ``frozenset({TITAN_LIMB})``.
        requires_prior: ``False`` (the technique runs in pass 1).
        tier: ``'primary'``.

    The ``'primary'`` tier is not a claim of superiority but of exclusivity:
    a hazy body has no second estimator, so the fallback tier's
    "superseded when a primary covers the same body" semantics would never
    fire and would only mislead a reader.
    """

    name = 'TitanHazeNav'
    accepts_feature_types = frozenset({NavFeatureType.TITAN_LIMB})
    requires_prior = False
    tier = 'primary'
    confidence_attributes: ClassVar[frozenset[str]] = frozenset(
        {
            'at_edge',
            'symmetry_peak_score',
            'symmetry_valid_fraction',
            'arc_inlier_fraction',
            'arc_residual_rms_px',
            'envelope_diameter_px',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._model_error_floor_px = load_model_error_floor(self.tuning, self.name)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries a usable haze feature.

        Reads only feature metadata; never any pixels.

        Parameters:
            features: Full feature set after the reliability gate.

        Returns:
            A report that is infeasible with reason
            ``'no TITAN_LIMB features'`` when no ``TITAN_LIMB`` feature with
            a haze geometry payload is present, and feasible with the
            consumed count otherwise.
        """
        eligible = _eligible_features(features)
        if len(eligible) == 0:
            return NavFeasibilityReport(feasible=False, reason=_NO_FEATURES_REASON)
        return NavFeasibilityReport(
            feasible=True, reason='ok', consumed_feature_count=len(eligible)
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Measure the offset of the hazy body from its predicted position.

        Runs the mirror-symmetry cross-track scan and the sunward limb-arc
        circle fit over ``context.image_ext`` -- the raw extended image, not
        the gradient or distance-transform planes the shape-fitting
        techniques consume, because the haze method reads brightness
        symmetry and a radial brightness falloff directly.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Only the first haze feature is consumed; a frame
                carries at most one hazy body.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``sensor_mask_ext``, ``obs.extfov_margin_vu``, and
                ``fit_camera_rotation``.

        Returns:
            A :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`
            carrying the recovered offset, an anisotropic covariance whose
            major axis lies along the symmetry axis, calibrated confidence,
            and a populated
            :class:`~spindoctor.nav_technique.diagnostics.TitanHazeDiagnostics`.
            A failed fit gate yields a spurious result naming that gate in
            ``diagnostics.gate_failed``.  ``rotation_rad`` is ``None`` on an
            instrument that does not fit camera rotation; where it does, the
            covariance is the rank-deficient ``(3, 3)`` form, because a
            single quasi-circular feature carries no rotation evidence.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            eligible = _eligible_features(features)
            feature = eligible[0]
            geometry = feature.geometry
            assert isinstance(geometry, TitanHazeGeometry)
            margin_v, margin_u = search_window_for_obs(context)
            window_px = float(max(margin_v, margin_u))
            nav_config = self.config.titan['navigation']
            symmetry_block = nav_config['symmetry']
            # A degenerate axis means any direction is equally valid, so the
            # refinement search has nothing to improve and would only chase
            # noise; disabling it is the Section-2.2 skip expressed through
            # the fitting library's own zero-range convention.
            angle_refine_deg = (
                0.0 if geometry.axis_degenerate else float(symmetry_block['angle_refine_deg'])
            )
            reported_center = reported_position_vu(
                geometry.predicted_center_vu, (margin_v, margin_u)
            )
            self.logger.info(
                'Fitting haze at predicted center (%.2f, %.2f), envelope radius %.2f px, '
                'axis %.2f deg (degenerate = %s), window %.0f px',
                reported_center[0],
                reported_center[1],
                geometry.r_env_px,
                math.degrees(geometry.sun_angle_rad),
                geometry.axis_degenerate,
                window_px,
            )
            sym_params = _symmetry_params(symmetry_block, angle_refine_deg=angle_refine_deg)
            arc_params = _arc_params(nav_config['arc'])
            symmetry, arc, offset_vu, recentered = fit_titan_center(
                np.asarray(context.image_ext, np.float64),
                np.asarray(context.sensor_mask_ext, bool),
                geometry.predicted_center_vu,
                contaminant_mask=geometry.contaminant_mask,
                theta0_rad=geometry.sun_angle_rad,
                r_solid_px=geometry.r_solid_px,
                r_env_px=geometry.r_env_px,
                window_px=window_px,
                sym_params=sym_params,
                arc_params=arc_params,
                recenter_threshold_px=float(nav_config['recenter_threshold_px']),
            )
            self._log_gate_table(
                symmetry,
                arc,
                sym_params=sym_params,
                arc_params=arc_params,
                r_solid_px=geometry.r_solid_px,
                r_env_px=geometry.r_env_px,
                window_px=window_px,
            )
            return self._assemble_result(
                feature=feature,
                geometry=geometry,
                symmetry=symmetry,
                arc=arc,
                offset_vu=offset_vu,
                recentered=recentered,
                window_px=window_px,
                fit_rotation=bool(context.fit_camera_rotation),
            )

    def _log_gate_table(
        self,
        symmetry: SymmetryFitResult,
        arc: ArcFitResult,
        *,
        sym_params: SymmetryFitParams,
        arc_params: ArcFitParams,
        r_solid_px: float,
        r_env_px: float,
        window_px: float,
    ) -> None:
        """Log every fit gate with its measurement, its threshold, and its verdict.

        One line per gate, in the order the fitting library evaluates them,
        so an operator reading the per-image log sees why a frame was
        accepted or rejected without re-running anything.  The values are
        the FINAL pass's: an intermediate pass is deliberately ungated.

        Verdicts are ``PASS`` / ``FAIL``, ``EDGE`` on the two at-edge rows
        (a flag, not a rejection), and ``SKIP`` on a gate the fit returned
        before reaching.  Each verdict is authoritative; on the cross-track
        at-edge row the two value columns are informational rather than the
        literal comparison, for the reason given in
        :func:`_symmetry_gate_rows`.

        Parameters:
            symmetry: The final pass's cross-track fit.
            arc: The final pass's along-track fit.
            sym_params: Cross-track tuning constants.
            arc_params: Along-track tuning constants.
            r_solid_px: Solid-body radius in pixels.
            r_env_px: Haze-envelope radius in pixels.
            window_px: Search half-window in pixels.
        """
        rows = _symmetry_gate_rows(symmetry, params=sym_params, window_px=window_px)
        rows += _arc_gate_rows(
            arc,
            params=arc_params,
            r_solid_px=r_solid_px,
            r_env_px=r_env_px,
            window_px=window_px,
        )
        self.logger.info('Gate table (final pass):')
        self.logger.info(_GATE_TABLE_FORMAT, 'gate', 'measured', 'threshold', 'result')
        for name, measured, threshold, verdict in rows:
            self.logger.info(_GATE_TABLE_FORMAT, name, measured, threshold, verdict)

    def _assemble_result(
        self,
        *,
        feature: NavFeature,
        geometry: TitanHazeGeometry,
        symmetry: SymmetryFitResult,
        arc: ArcFitResult,
        offset_vu: tuple[float, float],
        recentered: bool,
        window_px: float,
        fit_rotation: bool,
    ) -> NavTechniqueResult:
        """Turn a completed fit into a technique result.

        Splits the assembled offset back onto the symmetry axis so the
        diagnostics report the cross-track and along-track components that
        the covariance describes, decides ``at_edge``, and short-circuits to
        a spurious result when either half of the fit failed a gate.
        """
        c_hat, a_hat = axis_vectors(symmetry.theta_rad)
        offset = np.asarray(offset_vu, dtype=np.float64)
        cross_track_px = float(offset @ c_hat)
        along_track_px = float(offset @ a_hat)
        rays_total = arc.n_rays_total
        inlier_fraction = arc.n_rays_inlier / rays_total if rays_total > 0 else 0.0
        # Both fits report NaN for a quantity they could not measure: the
        # symmetry scan when no candidate shift had enough signal to
        # correlate, the arc fit when reweighting rejected every ray.  Those
        # are "undefined", not "zero" and not "perfect", so the diagnostics
        # record them as absent -- which is also what keeps a bare NaN out of
        # the strict JSON serializer.
        residual_finite = math.isfinite(arc.residual_rms_px)
        peak_score_finite = math.isfinite(symmetry.peak_score)
        diagnostics = TitanHazeDiagnostics(
            sun_angle_deg=math.degrees(symmetry.theta_rad),
            axis_degenerate=geometry.axis_degenerate,
            phase_deg=geometry.phase_deg,
            envelope_diameter_px=2.0 * geometry.r_env_px,
            cross_track_px=cross_track_px,
            along_track_px=along_track_px,
            symmetry_peak_score=symmetry.peak_score if peak_score_finite else None,
            symmetry_valid_fraction=symmetry.valid_fraction,
            symmetry_second_peak_ratio=symmetry.second_peak_ratio,
            theta_refined_deg=_wrap_degrees(
                math.degrees(symmetry.theta_rad - geometry.sun_angle_rad)
            ),
            arc_rays_total=rays_total,
            arc_rays_inlier=arc.n_rays_inlier,
            arc_inlier_fraction=inlier_fraction,
            arc_residual_rms_px=arc.residual_rms_px if residual_finite else None,
            fitted_haze_radius_km=arc.radius_px * geometry.km_per_px,
            filters=geometry.filters,
            recentered=recentered,
            gate_failed=symmetry.gate_failed or arc.gate_failed,
        )
        source_bodies = body_names_from_features([feature])
        if diagnostics.gate_failed is not None:
            self.logger.info(
                'Fit rejected by the %s gate (peak score %.4f, valid fraction %.4f, '
                'second-peak ratio %.4f, rays %d of which %d inlier); reporting spurious result',
                diagnostics.gate_failed,
                symmetry.peak_score,
                symmetry.valid_fraction,
                symmetry.second_peak_ratio,
                rays_total,
                arc.n_rays_inlier,
            )
            return self._spurious_result(
                feature_ids=(feature.feature_id,),
                diagnostics=diagnostics,
                fit_rotation=fit_rotation,
                source_bodies=source_bodies,
            )
        # Each pass gates |c| and |d| separately, so a recentered run can
        # legitimately assemble a total beyond the search window with both
        # per-pass flags clear.  The result-level flag is what makes the
        # ensemble's conservative at-edge handling apply to that total.
        at_edge = (
            symmetry.at_edge
            or arc.at_edge
            or abs(cross_track_px) >= window_px
            or abs(along_track_px) >= window_px
        )
        covariance = add_model_error_floor(
            _rotated_covariance(symmetry.theta_rad, symmetry.sigma_cross_px, arc.sigma_along_px),
            self._model_error_floor_px,
        )
        if fit_rotation:
            covariance = embed_rotation_unobservable(covariance)
        assert self.confidence_spec is not None
        confidence, breakdown = evaluate_sigmoid_combination(
            self.confidence_spec,
            _TitanConfidenceContext(
                at_edge=at_edge,
                diagnostics=diagnostics,
                symmetry_peak_score=(
                    symmetry.peak_score if peak_score_finite else _UNDEFINED_QUALITY_SENTINEL
                ),
                arc_residual_rms_px=(
                    arc.residual_rms_px if residual_finite else _UNDEFINED_RESIDUAL_SENTINEL
                ),
            ),
            technique_name=self.name,
            return_breakdown=True,
        )
        log_confidence_breakdown(self.logger, breakdown)
        self.logger.info(
            'Converged at offset (%.4f, %.4f) px = cross %.4f +- %.4f px, along %.4f +- %.4f px; '
            'fitted haze radius %.1f km; confidence %.4f',
            offset_vu[0],
            offset_vu[1],
            cross_track_px,
            symmetry.sigma_cross_px,
            along_track_px,
            arc.sigma_along_px,
            diagnostics.fitted_haze_radius_km,
            float(confidence),
        )
        if at_edge:
            self.logger.info('Diagnostic flags: at_edge=%s, recentered=%s', at_edge, recentered)
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(feature.feature_id,),
            offset_px=(float(offset_vu[0]), float(offset_vu[1])),
            covariance_px2=covariance,
            confidence=float(confidence),
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
            source_bodies=source_bodies,
        )
