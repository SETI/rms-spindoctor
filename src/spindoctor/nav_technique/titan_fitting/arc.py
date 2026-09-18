"""Along-track offset by a robust circle fit to the sunward limb arc.

The limb facing the sub-solar point is close to circular, so radial
brightness profiles through it locate the limb ray by ray, and a circle fit
whose center is constrained to the symmetry axis and whose radius is free
measures the along-track component of the pointing offset without assuming a
haze altitude.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import map_coordinates, median_filter, minimum_filter1d

from spindoctor.nav_technique.dt_fitting import (
    information_matrix_to_covariance,
    tukey_biweight_weights,
)
from spindoctor.nav_technique.titan_fitting.grid import sample_bool_nearest, validate_image
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ARC_RADIUS_MAX_FRACTION',
    'ARC_RADIUS_MIN_FRACTION',
    'ArcFitParams',
    'ArcFitResult',
    'constrained_circle_fit',
    'limb_radii_from_profiles',
    'radial_profiles',
]

ARC_RADIUS_MIN_FRACTION = 0.98
"""Smallest fitted arc radius, as a fraction of the solid-body radius.

The fitted limb cannot lie inside the solid body.  Exported because the
consuming technique reports the band this gate tests against, and the band
must have one definition.
"""

ARC_RADIUS_MAX_FRACTION = 1.05
"""Largest fitted arc radius, as a fraction of ``r_env_px + window_px``.

The fitted limb cannot lie beyond the haze envelope displaced by the whole
search window, plus a few percent of slack.
"""

# IRLS control for the constrained circle fit.
_MAX_IRLS_ITERATIONS = 25
_IRLS_CONVERGENCE_PX = 0.01
# Floor on the robust residual scale.  Noiseless synthetic arcs drive the MAD
# to exactly zero, which would make every Tukey argument non-finite.
_MIN_ROBUST_SCALE_PX = 1.0e-3


@dataclass(frozen=True)
class ArcFitParams:
    """Tuning constants for the sunward-limb arc fit.

    Parameters:
        sector_half_angle_deg: Half-width in degrees of the ray sector
            centered on the symmetry axis.
        ray_step_deg: Angular spacing of the rays in degrees.
        radial_step_px: Radial sampling step of each ray profile in pixels.
        radial_inner_fraction: Start radius of each profile as a fraction of
            the solid-body radius.
        radial_outer_pad_px: Distance in pixels each profile extends beyond
            the envelope radius plus the search window.
        median_filter_samples: Width in samples of the median filter applied
            to each profile before differentiation.
        min_gradient_snr: Smallest acceptable ratio of the steepest falloff
            to the raw median absolute deviation of the gradient.
        min_rays: Smallest acceptable number of surviving rays, and of
            inlier rays after the robust fit.
        min_inlier_fraction: Smallest acceptable inlier fraction among the
            surviving rays.
        max_residual_rms_px: Largest acceptable inlier residual RMS.
        tukey_c: Tukey biweight tuning constant in units of the robust
            residual scale.
        along_sigma_scale: Multiplier applied to the raw along-track sigma
            estimate.
        sigma_floor_along_px: Lower clamp on the reported along-track sigma.
    """

    sector_half_angle_deg: float
    ray_step_deg: float
    radial_step_px: float
    radial_inner_fraction: float
    radial_outer_pad_px: float
    median_filter_samples: int
    min_gradient_snr: float
    min_rays: int
    min_inlier_fraction: float
    max_residual_rms_px: float
    tukey_c: float
    along_sigma_scale: float
    sigma_floor_along_px: float


@dataclass(frozen=True)
class ArcFitResult:
    """Outcome of the axis-constrained robust circle fit to the limb arc.

    Parameters:
        along_track_px: Fitted center shift along ``a_hat`` in pixels,
            measured from the ray origin.
        sigma_along_px: Reported one-sigma along-track uncertainty, clamped
            to ``[sigma_floor_along_px, window_px]``.
        radius_px: Fitted arc radius in pixels; ``0.0`` when too few rays
            survived to attempt a fit.
        n_rays_total: Number of rays presented to the fit.
        n_rays_inlier: Number of rays carrying a non-zero final weight.
        residual_rms_px: Root-mean-square radial residual over the inliers;
            ``0.0`` when no fit was attempted and NaN when the robust fit
            rejected every ray.
        at_edge: True when the fitted shift reaches the search window bound.
        gate_failed: Name of the first failed gate (``'ray_yield'``,
            ``'arc_inliers'``, ``'arc_radius'`` or ``'arc_residual'``), or
            None when the fit passed every gate.
    """

    along_track_px: float
    sigma_along_px: float
    radius_px: float
    n_rays_total: int
    n_rays_inlier: int
    residual_rms_px: float
    at_edge: bool
    gate_failed: str | None


def radial_profiles(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    *,
    contaminant_mask: NDArrayBoolType | None,
    mask_shift_vu: tuple[float, float],
    axis_dir_vu: tuple[float, float],
    pass_pad_px: float,
    phi_rad_list: NDArrayFloatType,
    r_start_px: float,
    r_stop_px: float,
    r_step_px: float,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Sample outward brightness profiles along a fan of rays.

    Each ray leaves ``center_vu`` in the direction ``(sin phi, cos phi)``
    and is sampled by cubic interpolation at evenly spaced radii.  A sample
    is invalid when it falls outside the image, on a pixel ``valid_mask``
    rejects, or on the contaminant mask once that mask has been shifted by
    ``mask_shift_vu`` and dilated along ``axis_dir_vu`` by ``pass_pad_px``.

    Parameters:
        image: 2-D image to sample.
        valid_mask: Static per-pixel validity of ``image``.
        center_vu: ``(v, u)`` ray origin.
        contaminant_mask: Undilated boolean array of the image shape, or
            None when nothing is masked.
        mask_shift_vu: ``(dv, du)`` accumulated center hypothesis the mask
            is shifted by, so it lands on the contaminants' actual
            positions.
        axis_dir_vu: Unit vector along which the mask is dilated, normally
            ``a_hat``.
        pass_pad_px: Half length of that dilation in pixels, covering the
            along-track position error of this pass.
        phi_rad_list: Ray angles in radians.
        r_start_px: First sample radius.
        r_stop_px: Last sample radius (inclusive to within a step).
        r_step_px: Radial sample spacing.

    Returns:
        ``(profiles, profile_valid)``, both of shape ``(n_rays,
        n_radii)``.  Invalid entries of ``profiles`` are set to zero.

    Raises:
        ValueError: if the image is not 2-D, ``valid_mask`` has a different
            shape, ``r_step_px`` is not positive, or the radius range is
            empty.
    """
    validate_image(image, valid_mask)
    if not math.isfinite(r_step_px) or r_step_px <= 0.0:
        raise ValueError(f'r_step_px must be positive and finite; got {r_step_px!r}')
    if r_stop_px <= r_start_px:
        raise ValueError(f'r_stop_px must exceed r_start_px; got {r_start_px!r} and {r_stop_px!r}')
    radii = np.arange(r_start_px, r_stop_px + 0.5 * r_step_px, r_step_px)
    phis = np.asarray(phi_rad_list, dtype=np.float64)
    dv = np.sin(phis)[:, np.newaxis] * radii[np.newaxis, :]
    du = np.cos(phis)[:, np.newaxis] * radii[np.newaxis, :]
    vv = center_vu[0] + dv
    uu = center_vu[1] + du
    samples = map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.stack([vv, uu], axis=0),
        order=3,
        mode='nearest',
    )
    valid = (
        (vv >= 0.0)
        & (vv <= image.shape[0] - 1)
        & (uu >= 0.0)
        & (uu <= image.shape[1] - 1)
        & sample_bool_nearest(valid_mask, vv, uu)
    )
    if contaminant_mask is not None:
        k = math.ceil(pass_pad_px)
        base_v = vv - mask_shift_vu[0]
        base_u = uu - mask_shift_vu[1]
        hit = np.zeros(vv.shape, dtype=bool)
        for lam in range(-k, k + 1):
            hit |= sample_bool_nearest(
                contaminant_mask,
                base_v - lam * axis_dir_vu[0],
                base_u - lam * axis_dir_vu[1],
            )
        valid &= ~hit
    profiles = np.where(valid, samples, 0.0)
    return cast(NDArrayFloatType, profiles), cast(NDArrayBoolType, valid)


def limb_radii_from_profiles(
    profiles: NDArrayFloatType,
    profile_valid: NDArrayBoolType,
    *,
    r_start_px: float,
    r_step_px: float,
    r_solid_px: float,
    window_px_lo: float,
    window_px_hi: float,
    params: ArcFitParams,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Locate the limb on each radial profile by steepest outward falloff.

    Each profile is median filtered, differentiated, and searched over the
    window given by intersecting ``[window_px_lo, window_px_hi]`` with the
    sampled range shrunk by half the filter width at each end.  The most
    negative gradient in that window is the limb, refined by a parabola
    through its two neighbors.

    A ray is dropped when any of the following holds:

    * a sample outside the solid body is invalid (the limb region must be
      clean; a contaminant deeper in the interior costs the ray nothing,
      and the gradient samples it touches are excluded from the search
      rather than allowed to win it);
    * the steepest gradient in the window is not negative;
    * the steepest gradient sits on the first or last sample of the window,
      which means the search saturated against its bound rather than
      finding an extremum, and the true limb may lie outside;
    * it fails the same-units signal test
      ``|g_min| >= min_gradient_snr * MAD(g)`` over the window.

    Parameters:
        profiles: ``(n_rays, n_radii)`` brightness profiles.
        profile_valid: Per-sample validity of ``profiles``.
        r_start_px: Radius of the first profile sample.
        r_step_px: Radial sample spacing.
        r_solid_px: Solid-body radius in pixels.  Samples inside it are
            interior and may be invalid without costing the ray.
        window_px_lo: Smallest radius the limb may occupy.
        window_px_hi: Largest radius the limb may occupy.
        params: Tuning constants.

    Returns:
        ``(rho_px, ray_ok)``: the per-ray limb radius and which rays
        produced one.  Entries of ``rho_px`` for rejected rays are zero.

    Raises:
        ValueError: if the profile arrays disagree in shape or are not 2-D.
    """
    if profiles.ndim != 2:
        raise ValueError(f'profiles must be 2-D; got ndim={profiles.ndim}')
    if profile_valid.shape != profiles.shape:
        raise ValueError(
            f'profile_valid must match the profile shape; got '
            f'{profile_valid.shape} and {profiles.shape}'
        )
    n_rays, n_radii = profiles.shape
    rho = np.zeros(n_rays, dtype=np.float64)
    ray_ok = np.zeros(n_rays, dtype=bool)
    half_taps = max(params.median_filter_samples - 1, 0) // 2
    i_lo = max(math.ceil((window_px_lo - r_start_px) / r_step_px), half_taps)
    i_hi = min(
        math.floor((window_px_hi - r_start_px) / r_step_px),
        n_radii - 1 - half_taps,
    )
    if i_hi - i_lo < 2:
        return rho, cast(NDArrayBoolType, ray_ok)
    # Reach of one gradient sample: the median filter half width plus the one
    # extra sample the central difference consumes on each side.
    reach = half_taps + 1
    limb_lo = max(math.floor((r_solid_px - r_start_px) / r_step_px) + 1, i_lo - reach, 0)
    guard_hi = min(i_hi + reach, n_radii - 1)
    filtered = median_filter(
        profiles, size=(1, max(params.median_filter_samples, 1)), mode='nearest'
    )
    gradient = np.gradient(filtered, r_step_px, axis=1)
    # A gradient sample is usable only when every profile sample feeding it is
    # valid; the rest are excluded from the search so a zeroed contaminant
    # hole in the interior cannot masquerade as the steepest falloff.
    usable = (
        minimum_filter1d(
            profile_valid.astype(np.uint8), size=2 * reach + 1, axis=1, mode='constant', cval=0
        )
        > 0
    )
    for i in range(n_rays):
        if not profile_valid[i, limb_lo : guard_hi + 1].all():
            continue
        window = gradient[i, i_lo : i_hi + 1]
        window_usable = usable[i, i_lo : i_hi + 1]
        if int(window_usable.sum()) < 3:
            continue
        j = int(np.argmin(np.where(window_usable, window, np.inf)))
        if j == 0 or j == window.size - 1:
            continue
        if not window_usable[j - 1 : j + 2].all():
            continue
        g_min = float(window[j])
        if g_min >= 0.0:
            continue
        searched = window[window_usable]
        mad = float(np.median(np.abs(searched - np.median(searched))))
        if abs(g_min) < params.min_gradient_snr * mad:
            continue
        y_m, y_0, y_p = (float(window[j - 1]), g_min, float(window[j + 1]))
        curvature = y_p + y_m - 2.0 * y_0
        delta = 0.0
        if curvature > 0.0:
            # The vertex of a parabola through three samples straddling a
            # true minimum lies within half a sample of the middle one;
            # anything further comes from noise flattening the curvature.
            delta = min(max(0.5 * (y_m - y_p) / curvature, -0.5), 0.5)
        rho[i] = r_start_px + (i_lo + j + delta) * r_step_px
        ray_ok[i] = True
    return rho, cast(NDArrayBoolType, ray_ok)


def _circle_irls(
    points_vu: NDArrayFloatType,
    axis_origin_vu: NDArrayFloatType,
    axis_dir_vu: NDArrayFloatType,
    *,
    params: ArcFitParams,
) -> tuple[float, float, NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Run the axis-constrained robust circle fit.

    Returns:
        ``(d, radius, residuals, weights, jacobian)`` at the converged
        point, where ``d`` is the center shift along the axis.
    """
    offsets = points_vu - axis_origin_vu
    n_points = points_vu.shape[0]
    d = 0.0
    radius = float(np.median(np.hypot(offsets[:, 0], offsets[:, 1])))
    inliers: NDArrayBoolType = np.ones(n_points, dtype=bool)
    residuals: NDArrayFloatType = np.zeros(n_points, dtype=np.float64)
    weights: NDArrayFloatType = np.ones(n_points, dtype=np.float64)
    jacobian: NDArrayFloatType = np.zeros((n_points, 2), dtype=np.float64)
    for _ in range(_MAX_IRLS_ITERATIONS):
        delta_vu = offsets - d * axis_dir_vu
        norms = np.hypot(delta_vu[:, 0], delta_vu[:, 1])
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        residuals = norms - radius
        jacobian = np.stack(
            [
                -(delta_vu @ axis_dir_vu) / safe_norms,
                np.full(n_points, -1.0),
            ],
            axis=1,
        )
        subset = residuals[inliers] if inliers.any() else residuals
        scale = 1.4826 * float(np.median(np.abs(subset - np.median(subset))))
        scale = max(scale, _MIN_ROBUST_SCALE_PX)
        weights = tukey_biweight_weights(residuals / scale, c=params.tukey_c)
        inliers = weights > 0.0
        info = (jacobian * weights[:, np.newaxis]).T @ jacobian
        step = np.linalg.lstsq(info, -jacobian.T @ (weights * residuals), rcond=None)[0]
        d += float(step[0])
        radius += float(step[1])
        if abs(float(step[0])) < _IRLS_CONVERGENCE_PX:
            break
    return d, radius, residuals, weights, jacobian


def constrained_circle_fit(
    points_vu: NDArrayFloatType,
    axis_origin_vu: tuple[float, float],
    axis_dir_vu: tuple[float, float],
    *,
    r_solid_px: float,
    r_env_px: float,
    window_px: float,
    params: ArcFitParams,
) -> ArcFitResult:
    """Fit a circle to limb points with its center constrained to the axis.

    The free parameters are the center shift ``d`` along ``axis_dir_vu``
    from ``axis_origin_vu`` and the radius ``R``; the residual of a point is
    its distance from the center minus ``R``.  Iteratively reweighted least
    squares with Tukey biweights on the median-absolute-deviation scale
    rejects rays whose limb detection landed on something other than the
    haze edge, and one Gauss-Newton update per reweighting drives the
    parameters to convergence.

    Parameters:
        points_vu: ``(N, 2)`` limb points in ``(v, u)``.
        axis_origin_vu: ``(v, u)`` point the center shift is measured from.
        axis_dir_vu: Unit vector the center is constrained to move along.
        r_solid_px: Solid-body radius in pixels, the lower radius bound.
        r_env_px: Haze-envelope radius in pixels.
        window_px: Search half-window in pixels; it bounds both the
            acceptable center shift and the reported sigma.
        params: Tuning constants.

    Returns:
        An :class:`ArcFitResult`.  The gates are evaluated in the order
        ``ray_yield``, ``arc_inliers``, ``arc_radius``, ``arc_residual``,
        and the first failure is named in ``gate_failed``; a fitted shift
        reaching the window bound sets ``at_edge`` instead of failing a
        gate.

    Raises:
        ValueError: if ``points_vu`` is not an ``(N, 2)`` array.
    """
    points = np.asarray(points_vu, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f'points_vu must have shape (N, 2); got {points.shape}')
    n_total = points.shape[0]
    if n_total < max(params.min_rays, 1):
        return ArcFitResult(
            along_track_px=0.0,
            sigma_along_px=window_px,
            radius_px=0.0,
            n_rays_total=n_total,
            n_rays_inlier=0,
            residual_rms_px=0.0,
            at_edge=False,
            gate_failed='ray_yield',
        )
    origin = np.asarray(axis_origin_vu, dtype=np.float64)
    direction = np.asarray(axis_dir_vu, dtype=np.float64)
    d, radius, residuals, weights, jacobian = _circle_irls(points, origin, direction, params=params)
    inliers = weights > 0.0
    n_inlier = int(inliers.sum())
    inlier_residuals = residuals[inliers]
    # NaN, not zero, when the robust fit rejected every ray: the residual RMS
    # feeds a falling confidence term downstream, where a zero would read as a
    # flawless fit instead of a total failure.
    rms = float(np.sqrt(np.mean(inlier_residuals**2))) if n_inlier > 0 else float('nan')

    weight_sum = float(weights.sum())
    s2 = float((weights * residuals**2).sum()) / max(1.0, weight_sum - 2.0)
    covariance = s2 * information_matrix_to_covariance(jacobian, weights)
    variance = float(covariance[0, 0])
    sigma = params.along_sigma_scale * math.sqrt(max(variance, 0.0))
    if not math.isfinite(sigma):
        sigma = window_px
    sigma = min(max(sigma, params.sigma_floor_along_px), window_px)

    gate: str | None = None
    if n_inlier < params.min_rays or n_inlier < params.min_inlier_fraction * n_total:
        gate = 'arc_inliers'
    elif not (
        ARC_RADIUS_MIN_FRACTION * r_solid_px
        <= radius
        <= ARC_RADIUS_MAX_FRACTION * (r_env_px + window_px)
    ):
        gate = 'arc_radius'
    elif not rms <= params.max_residual_rms_px:
        gate = 'arc_residual'
    return ArcFitResult(
        along_track_px=d,
        sigma_along_px=sigma,
        radius_px=radius,
        n_rays_total=n_total,
        n_rays_inlier=n_inlier,
        residual_rms_px=rms,
        at_edge=abs(d) >= window_px,
        gate_failed=gate,
    )
