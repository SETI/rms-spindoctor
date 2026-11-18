"""``BodyDiscCorrelateNav`` — full-disc NCC translation fit.

Consumes every ``BODY_DISC`` feature in the input set, fuses the per-body
templates into a single composite by Z-buffer paint (closer body's pixels
overwrite farther body's), runs the existing pyramid kpeaks NCC against
the composite, and returns one combined translation.  ``use_gradient``
defaults to ``'auto'`` so the NCC self-selects raw vs gradient mode per
image — raw wins on smooth Lambert-shaded discs that fill the FOV;
gradient wins when only the limb carries unique-alignment signal.

Multi-body composites improve disambiguation: with ``N`` bodies the
correlation peak's SNR grows roughly as ``sqrt(N)`` if backgrounds are
independent, and the joint geometric constraint removes the
"swap moon assignments" mode-failure that plagues per-body solo
correlation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

from spindoctor.config import Config
from spindoctor.feature.composition import compose_template_features
from spindoctor.feature.feature import NavFeature, body_names_from_features
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import BodyDiscGeometry
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import BodyDiscDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    add_size_scaled_model_error,
    load_ncc_covariance_tuning,
    log_confidence_breakdown,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.correlate import (
    gradient_magnitude,
    navigate_with_pyramid_kpeaks,
    ncc_peak_quality,
    peak_to_runner_up_ratio,
    quality_rank_key,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyDiscCorrelateNav']


def _filter_disc_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset that carries a ``BODY_DISC`` template payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.BODY_DISC
        and isinstance(f.geometry, BodyDiscGeometry)
        and f.template_img is not None
        and f.template_mask is not None
    ]


# The rotation schedule: level 0 spreads its samples evenly across the configured
# +-max_rotation_deg, and each later level tries the listed multiples of its step
# around the previous level's winner, clamped to that range.
_LEVEL0_ROTATION_SAMPLES = 11
_LEVEL1_ROTATION_STEP_DEG = 0.5
_LEVEL1_ROTATION_MULTIPLES = (-2, -1, 0, 1, 2)
_LEVEL2_ROTATION_STEP_DEG = 0.25
_LEVEL2_ROTATION_MULTIPLES = (-1, 0, 1)
# Rotation samples that agree to this many decimal places of a degree are one sample.
# A later level's angle is computed from the earlier winner by adding a multiple of
# its step, so an angle level 0 already tried comes back differing in its last bits.
_ROTATION_SAMPLE_DECIMALS_DEG = 9


@dataclass(frozen=True)
class _RotationCandidate:
    """One rotation sample's correlation peak.

    Parameters:
        theta_rad: Rotation angle (radians) at which the template was rotated about
            the body-centroid pivot.
        quality: The quality of the strongest correlation peak at that rotation, from
            :func:`spindoctor.support.correlate.ncc_peak_quality`.
        offset_vu: The whole-pixel ``(dv, du)`` shift of that peak.
    """

    theta_rad: float
    quality: float
    offset_vu: tuple[int, int]


@dataclass(frozen=True)
class _RotationSearchInputs:
    """What every rotation sample of one search correlates.

    Parameters:
        image_surface: The ext-FOV image, or its gradient magnitude when the search
            runs in gradient mode.
        template_img: Composite body-disc template (pre-rotation).
        template_mask: Mask of the composite template.
        pivot_vu: Pivot ``(v, u)`` about which the template is rotated.
        max_offset_vu: Translation-search window in pixels.
        data_mask: Optional sensor mask for the bi-directional NCC.
        use_gradient: Whether each rotated template is replaced by its gradient
            magnitude before the correlation.
    """

    image_surface: NDArrayFloatType
    template_img: NDArrayFloatType
    template_mask: NDArrayBoolType
    pivot_vu: tuple[float, float]
    max_offset_vu: tuple[int, int]
    data_mask: NDArrayBoolType | None
    use_gradient: bool


def _zero_padded_shift(
    arr: NDArrayFloatType,
    shift_v: int,
    shift_u: int,
    *,
    fill_value: float = 0.0,
) -> NDArrayFloatType:
    """Translate ``arr`` by ``(shift_v, shift_u)`` with zero (or ``fill_value``)
    padding on the boundary.

    Equivalent to ``np.roll`` but drops out-of-bounds pixels rather than
    wrapping them to the opposite edge — the wrapping behavior is wrong
    for body / ring templates whose support sits anywhere off the array
    center, because rotation about a pivot followed by a shift back
    would otherwise smear template content into the diametrically
    opposite corner of the image.

    Works on 2-D arrays of any dtype; the destination is allocated with
    ``arr.dtype`` so boolean masks stay boolean.
    """
    h, w = arr.shape[:2]
    out = np.full(arr.shape, fill_value, dtype=arr.dtype)
    src_v_lo = max(0, -shift_v)
    src_v_hi = min(h, h - shift_v)
    src_u_lo = max(0, -shift_u)
    src_u_hi = min(w, w - shift_u)
    if src_v_hi <= src_v_lo or src_u_hi <= src_u_lo:
        return out
    dst_v_lo = src_v_lo + shift_v
    dst_u_lo = src_u_lo + shift_u
    dst_v_hi = dst_v_lo + (src_v_hi - src_v_lo)
    dst_u_hi = dst_u_lo + (src_u_hi - src_u_lo)
    out[dst_v_lo:dst_v_hi, dst_u_lo:dst_u_hi] = arr[src_v_lo:src_v_hi, src_u_lo:src_u_hi]
    return out


def _rotate_template(
    template_img: NDArrayFloatType,
    template_mask: NDArrayBoolType,
    pivot_vu: tuple[float, float],
    theta_rad: float,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Rotate the composite template + mask about ``pivot_vu`` by ``theta_rad``.

    Uses :func:`scipy.ndimage.rotate` followed by a zero-padded translate
    that re-centers the rotation pivot — ``ndimage.rotate`` rotates about
    the array center, so we shift the array so the pivot is at the center,
    rotate, then shift back.  Both shifts use :func:`_zero_padded_shift`
    rather than ``np.roll`` so out-of-bounds pixels are dropped instead of
    wrapping to the opposite edge (a wrap would smear template content
    across the image after the second shift).  The shift is integer-
    rounded which sacrifices < 1 px of pivot accuracy for a much simpler
    implementation; the rotation samples are coarse (≥ 0.25 deg) so the
    rounding error is well below the per-pixel template grid.

    Returns the rotated template (float64, same shape as input) and the
    rotated mask (bool, same shape).  The mask uses nearest-neighbor
    interpolation so it stays binary.
    """
    if abs(theta_rad) < 1e-12:
        return template_img.astype(np.float64, copy=True), template_mask.astype(bool, copy=True)
    pivot_v, pivot_u = pivot_vu
    h, w = template_img.shape[:2]
    center_v = (h - 1) / 2.0
    center_u = (w - 1) / 2.0
    shift_v = round(center_v - pivot_v)
    shift_u = round(center_u - pivot_u)
    shifted = _zero_padded_shift(template_img.astype(np.float64, copy=False), shift_v, shift_u)
    shifted_mask = _zero_padded_shift(
        template_mask.astype(np.uint8, copy=False), shift_v, shift_u, fill_value=0
    )
    rotated = ndimage_rotate(
        shifted,
        angle=math.degrees(theta_rad),
        reshape=False,
        order=1,
        mode='constant',
        cval=0.0,
    )
    rotated_mask = ndimage_rotate(
        shifted_mask,
        angle=math.degrees(theta_rad),
        reshape=False,
        order=0,
        mode='constant',
        cval=0,
    )
    rotated = _zero_padded_shift(rotated, -shift_v, -shift_u)
    rotated_mask = _zero_padded_shift(rotated_mask, -shift_v, -shift_u, fill_value=0)
    return (
        rotated.astype(np.float64, copy=False),
        rotated_mask.astype(bool, copy=False),
    )


def _clamped_rotation_samples(
    center_rad: float,
    step_rad: float,
    multiples: tuple[int, ...],
    max_rotation_rad: float,
) -> list[float]:
    """Return rotation samples at multiples of a step around a center, clamped.

    Parameters:
        center_rad: The angle the samples are centered on, in radians.
        step_rad: The spacing between samples, in radians.
        multiples: The multiples of ``step_rad`` to add to ``center_rad``, in order.
        max_rotation_rad: The largest rotation magnitude allowed, in radians.

    Returns:
        ``center_rad + k * step_rad`` for each ``k`` in ``multiples``, each clamped to
        ``[-max_rotation_rad, max_rotation_rad]``.
    """
    return [
        max(-max_rotation_rad, min(max_rotation_rad, center_rad + k * step_rad)) for k in multiples
    ]


def _rotation_sample_key(theta_rad: float) -> float:
    """Return the key under which a rotation sample is recorded in one search.

    Parameters:
        theta_rad: The sample's rotation, in radians.

    Returns:
        The rotation in degrees, rounded to ``_ROTATION_SAMPLE_DECIMALS_DEG`` places.
    """
    return round(math.degrees(theta_rad), _ROTATION_SAMPLE_DECIMALS_DEG)


def _composite_pivot_vu(features: list[NavFeature]) -> tuple[float, float]:
    """Return the centroid-of-body-centers pivot for a multi-body composite.

    The composite template is the union of every body's per-body template
    painted into ext-FOV coordinates; the natural rotation pivot is the
    centroid of the bodies' predicted centers in that same frame.  A
    single-body composite reduces to that body's predicted center.

    The centers are extfov pixel centric, which is the system
    :func:`_rotate_template` states the array's own center in, so the shift
    that brings the pivot onto that center is the whole of the offset between
    them.

    Raises:
        ValueError: if ``features`` is empty (the upstream feasibility
            gate should never let this happen, but the check guards
            against a degenerate divide-by-zero).
    """
    if not features:
        raise ValueError(
            '_composite_pivot_vu requires at least one BODY_DISC feature; got an empty list'
        )
    centers = [
        feat.geometry.predicted_center_vu  # type: ignore[union-attr]
        for feat in features
    ]
    cv = sum(c[0] for c in centers) / len(centers)
    cu = sum(c[1] for c in centers) / len(centers)
    return float(cv), float(cu)


class BodyDiscCorrelateNav(NavTechnique):
    """Body-disc full-disc NCC translation fit (multi-body, Z-buffer paint).

    Class attributes:
        accepts_feature_types: ``frozenset({BODY_DISC})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'BodyDiscCorrelateNav'
    accepts_feature_types = frozenset({NavFeatureType.BODY_DISC})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'ncc_peak',
            'peak_to_runner_up_ratio',
            'consistency_px',
            'consistency_ratio',
            'used_gradient',
            'body_count',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        self._consistency_max_fraction_of_diameter = float(
            self.tuning['consistency_max_fraction_of_diameter']
        )
        self._consistency_max_px = float(self.tuning['consistency_max_px'])
        self._refine_lowpass_sigma_px = float(self.tuning['refine_lowpass_sigma_px'])
        self._cov_tuning = load_ncc_covariance_tuning(self.tuning, self.name)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable BODY_DISC feature.

        Reads only feature metadata — never any pixels — so the report is
        cheap to obtain even on large feature sets.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one ``BODY_DISC`` feature carries a template payload.
        """
        eligible = _filter_disc_features(features)
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_body_disc_features_with_template',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input BODY_DISC templates.

        When ``context.fit_camera_rotation`` is False the technique runs a
        single 2-D NCC pyramid against the unrotated composite template.
        When the flag is True it runs the rotation search described in
        :meth:`_search_rotation` -- 11 + 5 + 3 rotation samples spanning
        ``+-max_rotation_deg``, each scored by one correlation, then the full
        pyramid at the best sample -- and returns that pyramid's (dv, du) with
        the best sample's theta.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.  Features without a template payload are dropped
                before fitting.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``sensor_mask_ext``, ``obs.extfov_margin_vu``, plus
                ``fit_camera_rotation`` / ``max_rotation_deg``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2 or
            3x3 covariance, calibrated confidence, and a populated
            :class:`BodyDiscDiagnostics`.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            eligible = _filter_disc_features(features)
            self.logger.info(
                'Consuming %d BODY_DISC features (out of %d offered)',
                len(eligible),
                len(features),
            )
            extfov_shape = context.image_ext.shape
            template_img, template_mask = compose_template_features(eligible, extfov_shape)
            margin_v, margin_u = search_window_for_obs(context)
            up_factor = self._upsample_factor()
            consistency_tol = self._consistency_tol_for(eligible)
            self.logger.debug(
                'Composite template: %d painted pixels; search window (v, u) = (%d, %d) px; '
                'upsample factor = %d; consistency tolerance = %.2f px',
                int(template_mask.sum()),
                margin_v,
                margin_u,
                up_factor,
                consistency_tol,
            )
            fit_rotation = bool(context.fit_camera_rotation)
            if fit_rotation:
                pivot = _composite_pivot_vu(eligible)
                best_theta_rad, ncc_result, sigma_theta_rad = self._search_rotation(
                    image=context.image_ext,
                    template_img=template_img,
                    template_mask=template_mask,
                    pivot_vu=pivot,
                    max_offset_vu=(margin_v, margin_u),
                    data_mask=context.sensor_mask_ext,
                    upsample_factor=up_factor,
                    max_rotation_deg=float(context.max_rotation_deg),
                    consistency_tol=consistency_tol,
                )
            else:
                best_theta_rad = 0.0
                sigma_theta_rad = None
                ncc_result = navigate_with_pyramid_kpeaks(
                    image=context.image_ext,
                    model=template_img,
                    mask=template_mask,
                    upsample_factor=up_factor,
                    consistency_tol=consistency_tol,
                    max_offset_vu=(margin_v, margin_u),
                    data_mask=context.sensor_mask_ext,
                    use_gradient='auto',
                    refine_lowpass_sigma_px=self._refine_lowpass_sigma_px,
                    localization_uncertainty_scale=(
                        self._cov_tuning.localization_uncertainty_scale
                    ),
                    logger=self.logger,
                )
            dv = float(ncc_result['offset'][0])
            du = float(ncc_result['offset'][1])
            covariance_2x2 = np.asarray(ncc_result['cov'], np.float64)
            if covariance_2x2.shape != (2, 2):
                raise RuntimeError(
                    f'BodyDiscCorrelateNav: navigate_with_pyramid_kpeaks returned '
                    f'covariance with shape {covariance_2x2.shape}; expected (2, 2). '
                    'Investigate the pyramid output rather than silently slicing.'
                )
            # Size-scaled silhouette model error plus absolute floor: the bare
            # peak-curvature covariance measures statistical precision only, so
            # add the body-diameter-proportional shape-mismatch term and the
            # pointing / distortion floor (the localization-spread term was
            # already folded in by the correlator).
            covariance_2x2 = add_size_scaled_model_error(
                covariance_2x2,
                size_px=self._max_body_extent_px(eligible),
                size_frac=self._cov_tuning.model_error_size_frac,
                floor_px=self._cov_tuning.model_error_floor_px,
            )
            spurious = bool(ncc_result['spurious'])
            at_edge_translation = bool(ncc_result['at_edge'])
            quality = float(ncc_result['quality'])
            consistency = float(ncc_result['consistency'])
            used_gradient = bool(ncc_result.get('used_gradient', False))
            top_k_peaks = ncc_result.get('top_k_peaks', [])
            diagnostics = BodyDiscDiagnostics(
                ncc_peak=quality,
                peak_to_runner_up_ratio=peak_to_runner_up_ratio(top_k_peaks),
                consistency_px=consistency,
                # ``consistency_tol`` is the same diameter-scaled cap
                # used by the spurious test, so a result that just
                # barely cleared the spurious test reads as ratio
                # slightly under 1.0 and a clean fit on a large body
                # reads as the small ratio it deserves regardless of
                # body size.
                consistency_ratio=consistency / consistency_tol,
                used_gradient=used_gradient,
                body_count=len(eligible),
            )
            covariance: NDArrayFloatType
            rotation_rad: float | None
            sigma_rotation_rad: float | None
            rotation_at_edge = False
            if fit_rotation:
                max_rotation_rad = math.radians(context.max_rotation_deg)
                rotation_at_edge = (
                    abs(best_theta_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
                )
                cov = np.zeros((3, 3), dtype=np.float64)
                cov[:2, :2] = covariance_2x2
                cov[2, 2] = float(
                    sigma_theta_rad * sigma_theta_rad
                    if sigma_theta_rad is not None
                    else ROTATION_UNOBSERVABLE_VARIANCE
                )
                covariance = cov
                rotation_rad = float(best_theta_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
            else:
                covariance = covariance_2x2
                rotation_rad = None
                sigma_rotation_rad = None
            at_edge = at_edge_translation or rotation_at_edge
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _DiscConfidenceContext(at_edge=at_edge, spurious=spurious, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, quality %.3f, consistency %.3f, '
                'mode=%s, bodies=%d, confidence %.4f',
                dv,
                du,
                quality,
                consistency,
                'gradient' if used_gradient else 'raw',
                len(eligible),
                float(confidence),
            )
            if fit_rotation:
                self.logger.info(
                    'Rotation = %+.4f deg (sigma %.4f deg)%s',
                    math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                    math.degrees(sigma_rotation_rad if sigma_rotation_rad is not None else 0.0),
                    ', AT_EDGE' if rotation_at_edge else '',
                )
            if spurious or at_edge:
                self.logger.info('Diagnostic flags: spurious=%s, at_edge=%s', spurious, at_edge)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in eligible),
                offset_px=(dv, du),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=spurious,
                at_edge=at_edge,
                diagnostics=diagnostics,
                rotation_rad=rotation_rad,
                sigma_rotation_rad=sigma_rotation_rad,
                source_bodies=body_names_from_features(eligible),
            )

    def _consistency_tol_for(self, features: list[NavFeature]) -> float:
        """Return the diameter-scaled inter-pyramid consistency cap.

        A small body's NCC peak is naturally sharper in absolute pixels
        than a large body's, so a flat cap penalizes large bodies twice
        (real peak walks across pyramid levels scale with the
        silhouette extent).  The applied cap is
        ``max(consistency_max_px, consistency_max_fraction_of_diameter
        * max_diameter_px)`` where ``max_diameter_px`` is the largest
        ext-FOV bbox extent across the consumed BODY_DISC features.
        """
        if not features:
            return self._consistency_max_px
        scaled_px = self._consistency_max_fraction_of_diameter * self._max_body_extent_px(features)
        return max(self._consistency_max_px, scaled_px)

    @staticmethod
    def _max_body_extent_px(features: list[NavFeature]) -> float:
        """Return the largest ext-FOV bbox extent across the consumed features.

        Used both as the diameter that scales the consistency cap and as the
        silhouette-model-error size in the reported covariance.

        Parameters:
            features: Consumed BODY_DISC features.

        Returns:
            The largest ``max(v_extent, u_extent)`` in pixels, or ``0.0`` when
            ``features`` is empty.
        """
        max_extent_px = 0
        for feature in features:
            v_min, u_min, v_max, u_max = feature.geometry.bbox_extfov_vu
            max_extent_px = max(max_extent_px, v_max - v_min, u_max - u_min)
        return float(max_extent_px)

    def _search_rotation(
        self,
        *,
        image: NDArrayFloatType,
        template_img: NDArrayFloatType,
        template_mask: NDArrayBoolType,
        pivot_vu: tuple[float, float],
        max_offset_vu: tuple[int, int],
        data_mask: NDArrayBoolType | None,
        upsample_factor: int,
        max_rotation_deg: float,
        consistency_tol: float,
    ) -> tuple[float, dict[str, Any], float | None]:
        """Find the rotation whose template correlates best, and correlate it in full.

        The search first runs the full NCC pyramid on the unrotated template, in both
        raw and gradient modes (``use_gradient='auto'``).  The mode that pyramid
        picks is the mode every rotation sample is then scored in.  Each sample's
        score is the quality of its strongest whole-pixel correlation peak, from one
        full-resolution correlation
        (:func:`~spindoctor.support.correlate.ncc_peak_quality`), so a sample costs
        a small fraction of a pyramid.  The samples follow three levels:

        * Level 0 -- 11 samples spanning ``+-max_rotation_deg`` (1 deg apart at the
          default ``max_rotation_deg = 5``).
        * Level 1 -- 5 samples 0.5 deg apart, centered on the level-0 winner.
        * Level 2 -- 3 samples 0.25 deg apart, centered on the level-1 winner.

        Every sample is clamped to ``+-max_rotation_deg``.  Within a level the highest
        quality wins, the first of equal qualities is kept, and a NaN quality ranks
        last.  A rotation sampled twice (to within 1e-9 deg) is scored once.  The full
        NCC pyramid, again in both modes, then runs at the level-2 winner, and its
        result is the one returned; when the winner is zero rotation, the first
        pyramid's result is returned.

        The samples are therefore ranked differently from a search that runs the full
        pyramid at every sample.  That search ranks each sample by the quality of the
        candidate its pyramid picks among several, after the penalty of
        ``prior_weight_final`` (0.25) times that candidate's distance from the coarser
        levels' answer, and in a mode chosen anew at that sample.  Those qualities can
        differ from the scores here by enough to reorder the samples, so the two
        searches can pick different winners.  The final pyramid at the winner also
        chooses its mode anew, so the mode it returns, and the quality that goes with
        it, can differ from the mode the samples were scored in.

        Parameters:
            image: Source ext-FOV image.
            template_img: Composite body-disc template (pre-rotation).
            template_mask: Mask of the composite template.
            pivot_vu: Pivot ``(v, u)`` about which the template is rotated.
            max_offset_vu: Translation-search window in pixels.
            data_mask: Optional sensor mask for the bi-directional NCC.
            upsample_factor: FFT upsample factor of the full pyramids.
            max_rotation_deg: Largest rotation magnitude searched, in degrees.
            consistency_tol: Pyramid consistency tolerance of the full pyramids.

        Returns:
            ``(best_theta_rad, best_ncc_result, sigma_theta_rad)``: the winning
            rotation, the full pyramid's result at it (see
            :func:`~spindoctor.support.correlate.navigate_with_pyramid_kpeaks`), and
            the rotation uncertainty, which is always ``None`` (see
            :meth:`_rotation_sigma_from_quality`), so the rotation slot of the 3x3
            covariance carries the unobservable sentinel.
        """
        pyramid_inputs: dict[str, Any] = {
            'image': image,
            'template_img': template_img,
            'template_mask': template_mask,
            'pivot_vu': pivot_vu,
            'max_offset_vu': max_offset_vu,
            'data_mask': data_mask,
            'upsample_factor': upsample_factor,
            'consistency_tol': consistency_tol,
        }
        unrotated_result = self._ncc_at_rotation(theta_rad=0.0, **pyramid_inputs)
        # Choose the correlation mode once, from the unrotated template, rather than
        # running both modes at every rotation sample: the rotation search then costs
        # one correlation per sample, and the winner is checked in both modes by the
        # final pyramid (#126).
        use_gradient = bool(unrotated_result['used_gradient'])
        search = _RotationSearchInputs(
            image_surface=gradient_magnitude(image) if use_gradient else image,
            template_img=template_img,
            template_mask=template_mask,
            pivot_vu=pivot_vu,
            max_offset_vu=max_offset_vu,
            data_mask=data_mask,
            use_gradient=use_gradient,
        )
        scored: dict[float, _RotationCandidate] = {}

        max_rotation_rad = math.radians(max_rotation_deg)
        l0_thetas = np.linspace(-max_rotation_rad, max_rotation_rad, _LEVEL0_ROTATION_SAMPLES)
        self.logger.debug(
            'Rotation search level 0: %d samples across +-%.2f deg (step %.3f deg), %s mode',
            l0_thetas.size,
            max_rotation_deg,
            2.0 * max_rotation_deg / (_LEVEL0_ROTATION_SAMPLES - 1),
            'gradient' if use_gradient else 'raw',
        )
        l0_winner = self._best_rotation_sample(l0_thetas.tolist(), search, scored)
        l1_winner = self._best_rotation_sample(
            _clamped_rotation_samples(
                l0_winner.theta_rad,
                math.radians(_LEVEL1_ROTATION_STEP_DEG),
                _LEVEL1_ROTATION_MULTIPLES,
                max_rotation_rad,
            ),
            search,
            scored,
        )
        l2_thetas = _clamped_rotation_samples(
            l1_winner.theta_rad,
            math.radians(_LEVEL2_ROTATION_STEP_DEG),
            _LEVEL2_ROTATION_MULTIPLES,
            max_rotation_rad,
        )
        l2_winner = self._best_rotation_sample(l2_thetas, search, scored)
        sigma_theta_rad = self._rotation_sigma_from_quality(
            candidates=[scored[_rotation_sample_key(theta)] for theta in l2_thetas],
            winner=l2_winner,
            step_rad=math.radians(_LEVEL2_ROTATION_STEP_DEG),
        )
        if l2_winner.theta_rad == 0.0:
            ncc_result = unrotated_result
        else:
            ncc_result = self._ncc_at_rotation(theta_rad=l2_winner.theta_rad, **pyramid_inputs)
        sigma_str = f'{math.degrees(sigma_theta_rad):.4f}' if sigma_theta_rad is not None else 'inf'
        self.logger.info(
            'Rotation search winner: %+.4f deg after %d rotation samples, search quality '
            '%.3f, quality %.3f, sigma_theta %s deg',
            math.degrees(l2_winner.theta_rad),
            len(scored),
            l2_winner.quality,
            float(ncc_result['quality']),
            sigma_str,
        )
        return l2_winner.theta_rad, ncc_result, sigma_theta_rad

    def _best_rotation_sample(
        self,
        thetas_rad: list[float],
        search: _RotationSearchInputs,
        scored: dict[float, _RotationCandidate],
    ) -> _RotationCandidate:
        """Score each rotation sample and return the highest-quality one.

        Parameters:
            thetas_rad: Rotation samples (radians) to score, in order.  Must not be
                empty.
            search: What every sample correlates.
            scored: Samples already scored in this search, keyed by
                :func:`_rotation_sample_key`.  A sample found here is not scored again,
                and each newly scored sample is added.

        Returns:
            The sample with the highest quality.  Of equal qualities the first in
            ``thetas_rad`` is kept, and a NaN quality ranks last, level with
            ``-inf``.

        Raises:
            ValueError: If ``thetas_rad`` is empty.
        """
        best: _RotationCandidate | None = None
        for theta in thetas_rad:
            key = _rotation_sample_key(theta)
            if key not in scored:
                scored[key] = self._score_rotation(theta, search)
            candidate = scored[key]
            # Strict > keeps the first of equal qualities.  A NaN quality ranks last,
            # level with -inf, so a first sample with a NaN quality cannot block a
            # later sample.
            if best is None or quality_rank_key(candidate.quality) > quality_rank_key(best.quality):
                best = candidate
        if best is None:
            raise ValueError('_best_rotation_sample needs at least one rotation sample')
        return best

    def _score_rotation(
        self, theta_rad: float, search: _RotationSearchInputs
    ) -> _RotationCandidate:
        """Rotate the template and score its strongest correlation peak.

        Parameters:
            theta_rad: Rotation angle (radians) to apply about ``search.pivot_vu``.
            search: What the sample correlates.

        Returns:
            The sample's angle, peak quality, and whole-pixel peak shift.
        """
        rotated_img, rotated_mask = _rotate_template(
            search.template_img,
            search.template_mask,
            pivot_vu=search.pivot_vu,
            theta_rad=theta_rad,
        )
        model_surface = gradient_magnitude(rotated_img) if search.use_gradient else rotated_img
        peak = ncc_peak_quality(
            search.image_surface,
            model_surface,
            rotated_mask,
            max_offset_vu=search.max_offset_vu,
            data_mask=search.data_mask,
        )
        self.logger.debug(
            '  Rotation %+.4f deg: quality %.3f at (%d, %d) px',
            math.degrees(theta_rad),
            peak.quality,
            peak.offset_vu[0],
            peak.offset_vu[1],
        )
        return _RotationCandidate(
            theta_rad=theta_rad, quality=peak.quality, offset_vu=peak.offset_vu
        )

    def _ncc_at_rotation(
        self,
        *,
        image: NDArrayFloatType,
        template_img: NDArrayFloatType,
        template_mask: NDArrayBoolType,
        pivot_vu: tuple[float, float],
        theta_rad: float,
        max_offset_vu: tuple[int, int],
        data_mask: NDArrayBoolType | None,
        upsample_factor: int,
        consistency_tol: float,
    ) -> dict[str, Any]:
        """Rotate the template about ``pivot_vu`` and run a single NCC pyramid.

        Parameters:
            image: Source ext-FOV image.
            template_img: Composite body-disc template (pre-rotation).
            template_mask: Mask of the composite template.
            pivot_vu: Pivot ``(v, u)`` about which the template is
                rotated before correlation.
            theta_rad: Rotation angle (radians) to apply.
            max_offset_vu: Translation-search window in pixels.
            data_mask: Optional sensor mask threaded into the pyramid.
            upsample_factor: FFT upsample factor.

        Returns:
            The raw dict returned by
            :func:`spindoctor.support.correlate.navigate_with_pyramid_kpeaks`
            (offset, covariance, quality, consistency, top-K peaks,
            spurious / at-edge flags, gradient-mode tag).
        """
        rotated_img, rotated_mask = _rotate_template(
            template_img,
            template_mask,
            pivot_vu=pivot_vu,
            theta_rad=theta_rad,
        )
        return navigate_with_pyramid_kpeaks(
            image=image,
            model=rotated_img,
            mask=rotated_mask,
            upsample_factor=upsample_factor,
            consistency_tol=consistency_tol,
            max_offset_vu=max_offset_vu,
            data_mask=data_mask,
            use_gradient='auto',
            refine_lowpass_sigma_px=self._refine_lowpass_sigma_px,
            localization_uncertainty_scale=self._cov_tuning.localization_uncertainty_scale,
            logger=self.logger,
        )

    @staticmethod
    def _rotation_sigma_from_quality(
        *,
        candidates: list[_RotationCandidate],
        winner: _RotationCandidate,
        step_rad: float,
    ) -> float | None:
        """Report disc rotation uncertainty as unobservable (always ``None``).

        NAV-010.  The NCC peak quality the rotation samples are ranked by
        (:func:`spindoctor.support.correlate.ncc_peak_quality`) is a
        PSR / PMR-style separation ratio, not a log-likelihood.  The
        former curvature-to-variance map ``sigma_theta**2 = 1 / (-H *
        q_center)`` (with ``H`` the level-2 quality second derivative)
        is dimensionally ``rad**2 / quality**2`` and has no calibrated
        relationship to angular variance: PSR/PMR do not live on a
        log-likelihood scale, so the curvature carries no Fisher
        information about rotation.  A calibrated mapping from the
        NCC-peak rotation curvature to an angular variance is not yet
        available, so disc rotation uncertainty is reported as
        unobservable.

        Returning ``None`` routes the caller to the
        :data:`~spindoctor.nav_technique.nav_technique.ROTATION_UNOBSERVABLE_VARIANCE`
        sentinel in the 3x3 covariance's rotation slot (the ensemble's
        ``pinvh`` combine already maps that huge variance to a near-zero
        rotation information contribution), so the disc technique
        contributes a translation estimate while abstaining on rotation.

        Parameters:
            candidates: The three level-2 :class:`_RotationCandidate`
                samples (unused; retained for the call-site signature
                and a future calibrated implementation).
            winner: The rotation-search winner (unused; see above).
            step_rad: Angular sampling step in radians (unused).

        Returns:
            Always ``None`` -- disc rotation variance is reported as
            unobservable pending a calibrated curvature -> variance map.
        """
        return None

    def _upsample_factor(self) -> int:
        """Return the FFT upsample factor configured under ``config.offset``."""
        offset_block = getattr(self.config, 'offset', None)
        if offset_block is None:
            return 128
        return int(getattr(offset_block, 'correlation_fft_upsample_factor', 128))


class _DiscConfidenceContext:
    """Adapter binding ``BodyDiscDiagnostics`` plus ``at_edge`` / ``spurious``.

    The shared :func:`evaluate_sigmoid_combination` helper accepts any
    object whose attributes match the spec's term names.  ``at_edge`` and
    ``spurious`` are not part of ``BodyDiscDiagnostics`` (they live on
    ``NavTechniqueResult``) so this small adapter exposes both alongside
    the diagnostic fields the spec consumes.
    """

    def __init__(self, *, at_edge: bool, spurious: bool, diagnostics: BodyDiscDiagnostics) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.ncc_peak = diagnostics.ncc_peak
        self.peak_to_runner_up_ratio = diagnostics.peak_to_runner_up_ratio
        self.consistency_px = diagnostics.consistency_px
        self.consistency_ratio = diagnostics.consistency_ratio
        self.used_gradient = diagnostics.used_gradient
        self.body_count = float(diagnostics.body_count)
