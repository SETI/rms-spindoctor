"""The two-pass driver that turns the cross-track and arc fits into an offset.

One pass measures the cross-track shift by mirror correlation and then the
along-track shift by the sunward arc fit.  A large along-track shift means
the first pass sampled the body off center, so the sequence repeats once
about the corrected center; the gates that decide whether a frame navigated
are read from whichever pass came last.
"""

import math

import numpy as np

from spindoctor.nav_technique.titan_fitting.arc import (
    ArcFitParams,
    ArcFitResult,
    constrained_circle_fit,
    limb_radii_from_profiles,
    radial_profiles,
)
from spindoctor.nav_technique.titan_fitting.grid import axis_vectors
from spindoctor.nav_technique.titan_fitting.symmetry import (
    SymmetryFitParams,
    SymmetryFitResult,
    symmetry_scan,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'fit_titan_center',
]

# Smallest ray spacing worth honoring, guarding a zero or negative step.
_MIN_RAY_STEP_DEG = 1.0e-6


def _arc_pass(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    origin_vu: NDArrayFloatType,
    *,
    contaminant_mask: NDArrayBoolType | None,
    mask_shift_vu: NDArrayFloatType,
    theta_rad: float,
    r_solid_px: float,
    r_env_px: float,
    window_px: float,
    pass_pad_px: float,
    params: ArcFitParams,
) -> ArcFitResult:
    """Run one sunward-arc pass: profiles, limb extraction, circle fit."""
    _, a_hat = axis_vectors(theta_rad)
    half_angle = math.radians(params.sector_half_angle_deg)
    step = math.radians(max(params.ray_step_deg, _MIN_RAY_STEP_DEG))
    n_side = math.floor(half_angle / step)
    phis = theta_rad + step * np.arange(-n_side, n_side + 1, dtype=np.float64)
    r_start_px = params.radial_inner_fraction * r_solid_px
    profiles, profile_valid = radial_profiles(
        image,
        valid_mask,
        (float(origin_vu[0]), float(origin_vu[1])),
        contaminant_mask=contaminant_mask,
        mask_shift_vu=(float(mask_shift_vu[0]), float(mask_shift_vu[1])),
        axis_dir_vu=(float(a_hat[0]), float(a_hat[1])),
        pass_pad_px=pass_pad_px,
        phi_rad_list=phis,
        r_start_px=r_start_px,
        r_stop_px=r_env_px + params.radial_outer_pad_px + window_px,
        r_step_px=params.radial_step_px,
    )
    rho, ray_ok = limb_radii_from_profiles(
        profiles,
        profile_valid,
        r_start_px=r_start_px,
        r_step_px=params.radial_step_px,
        r_solid_px=r_solid_px,
        window_px_lo=r_solid_px - window_px,
        window_px_hi=r_env_px + window_px,
        params=params,
    )
    kept_phi = phis[ray_ok]
    kept_rho = rho[ray_ok]
    points = np.stack(
        [
            origin_vu[0] + kept_rho * np.sin(kept_phi),
            origin_vu[1] + kept_rho * np.cos(kept_phi),
        ],
        axis=1,
    )
    return constrained_circle_fit(
        points,
        (float(origin_vu[0]), float(origin_vu[1])),
        (float(a_hat[0]), float(a_hat[1])),
        r_solid_px=r_solid_px,
        r_env_px=r_env_px,
        window_px=window_px,
        params=params,
    )


def fit_titan_center(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    *,
    contaminant_mask: NDArrayBoolType | None,
    theta0_rad: float,
    r_solid_px: float,
    r_env_px: float,
    window_px: float,
    sym_params: SymmetryFitParams,
    arc_params: ArcFitParams,
    recenter_threshold_px: float,
) -> tuple[SymmetryFitResult, ArcFitResult, tuple[float, float], bool]:
    """Measure the offset of a hazy body from its predicted center.

    The first pass scans for the cross-track shift over an annulus stretched
    along the axis (so the annulus meets the body even when the along-track
    error is large), then fits the sunward limb arc for the along-track
    shift.  When that shift exceeds ``recenter_threshold_px`` the sequence
    runs once more about the shifted center, this time with a plain annulus
    and a mask dilation only as wide as the recentering threshold.  Gates
    are evaluated on whichever pass is final: an intermediate estimate
    diluted by the stretched annulus must not condemn a frame the second
    pass exists to rescue.

    Parameters:
        image: 2-D image to fit.
        valid_mask: Static per-pixel validity of ``image``.
        center_vu: ``(v, u)`` predicted body center.
        contaminant_mask: Undilated boolean array of the image shape marking
            pixels the fits must ignore, or None when nothing is masked.
        theta0_rad: Symmetry-axis angle in radians.
        r_solid_px: Solid-body radius in pixels.
        r_env_px: Haze-envelope radius in pixels.
        window_px: Search half-window in pixels.
        sym_params: Cross-track tuning constants.
        arc_params: Along-track tuning constants.
        recenter_threshold_px: Along-track shift above which the second pass
            runs.

    Returns:
        ``(symmetry, arc, offset_vu, recentered)`` where the first two are
        the FINAL pass's results, ``offset_vu`` is the measured ``(dv, du)``
        of the body relative to ``center_vu``, and ``recentered`` reports
        whether the second pass ran.  The offset takes its cross-track term
        from the final pass alone, because each pass re-measures the whole
        cross-track shift, while the along-track terms of the passes
        accumulate.
    """
    center = np.asarray(center_vu, dtype=np.float64)
    symmetry = symmetry_scan(
        image,
        valid_mask,
        center_vu,
        contaminant_mask=contaminant_mask,
        theta0_rad=theta0_rad,
        r_env_px=r_env_px,
        window_px=window_px,
        pass_pad_px=window_px,
        capsule_half_extent_px=window_px,
        params=sym_params,
    )
    c_hat, a_hat = axis_vectors(symmetry.theta_rad)
    shift = symmetry.cross_track_px * c_hat
    arc = _arc_pass(
        image,
        valid_mask,
        center + shift,
        contaminant_mask=contaminant_mask,
        mask_shift_vu=shift,
        theta_rad=symmetry.theta_rad,
        r_solid_px=r_solid_px,
        r_env_px=r_env_px,
        window_px=window_px,
        pass_pad_px=window_px,
        params=arc_params,
    )
    offset = shift + arc.along_track_px * a_hat
    if abs(arc.along_track_px) <= recenter_threshold_px:
        return symmetry, arc, (float(offset[0]), float(offset[1])), False

    along = arc.along_track_px * a_hat
    recentered_origin = center + along
    symmetry = symmetry_scan(
        image,
        valid_mask,
        (float(recentered_origin[0]), float(recentered_origin[1])),
        contaminant_mask=contaminant_mask,
        theta0_rad=theta0_rad,
        r_env_px=r_env_px,
        window_px=window_px,
        pass_pad_px=recenter_threshold_px,
        capsule_half_extent_px=0.0,
        mask_shift_vu=(float(along[0]), float(along[1])),
        params=sym_params,
    )
    c_hat, a_hat = axis_vectors(symmetry.theta_rad)
    shift = along + symmetry.cross_track_px * c_hat
    arc = _arc_pass(
        image,
        valid_mask,
        center + shift,
        contaminant_mask=contaminant_mask,
        mask_shift_vu=shift,
        theta_rad=symmetry.theta_rad,
        r_solid_px=r_solid_px,
        r_env_px=r_env_px,
        window_px=window_px,
        pass_pad_px=recenter_threshold_px,
        params=arc_params,
    )
    offset = shift + arc.along_track_px * a_hat
    return symmetry, arc, (float(offset[0]), float(offset[1])), True
