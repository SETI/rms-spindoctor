"""Unit tests for the Titan haze fitting library.

Every scene here is synthetic: a logistic-falloff disc with a brightness
ramp along the symmetry axis, optionally perturbed by clouds, intruders,
point sources or noise.  The recovery bounds referenced throughout are

* case 1 -- noiseless: cross-track within 0.05 px, along-track within
  0.2 px;
* case 2 -- additive Gaussian noise at a signal-to-noise ratio of 20:
  cross-track within 0.15 px, along-track within 1.5 px.

The case-2 along-track bound follows from the geometry of the estimator
rather than from a choice.  The sunward sector spans 120 degrees, over
which the along-track center shift and the free radius are nearly
degenerate, so the along-track sigma is close to the single-ray
limb-location sigma with essentially no averaging benefit, and at a
signal-to-noise ratio of 20 the steepest-gradient estimator locates one ray
to about 0.7 px.  Over 120 randomized scenes at that noise level the
along-track error has a standard deviation of 0.30 px, a 95th percentile of
1.03 px and a maximum of 1.20 px, while the cross-track error never exceeds
0.05 px.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from spindoctor.nav_technique.titan_fitting import (
    ArcFitParams,
    ArcFitResult,
    SymmetryFitParams,
    SymmetryFitResult,
    axis_vectors,
    constrained_circle_fit,
    fit_titan_center,
    limb_radii_from_profiles,
    radial_profiles,
    resample_rotated_grid,
    symmetry_scan,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

SHAPE_VU = (170, 170)
CENTER_VU = (85.0, 85.0)
R_LIMB_PX = 44.0
R_SOLID_PX = 40.0
R_ENV_PX = 46.0
WINDOW_PX = 10.0
RECENTER_PX = 8.0
PEAK_DN = 1000.0
NOISE_DN = PEAK_DN / 20.0

CASE1_CROSS_PX = 0.05
CASE1_ALONG_PX = 0.20
CASE2_CROSS_PX = 0.15
CASE2_ALONG_PX = 1.50

PLANTED_OFFSETS = (
    (0.0, 0.0),
    (0.3, -0.4),
    (-1.2, 0.8),
    (2.5, 1.5),
    (-2.0, -2.7),
)
PLANTED_ANGLES = (0.0, 0.7, 1.9, -2.4)

SYM_PARAMS = SymmetryFitParams(
    annulus_inner_fraction=0.55,
    annulus_outer_pad_px=6.0,
    angle_refine_deg=0.0,
    angle_refine_step_deg=0.5,
    angle_refine_min_gain=0.02,
    min_peak_score=0.60,
    min_valid_fraction=0.50,
    max_second_peak_ratio=0.90,
    cross_sigma_scale=1.0,
    sigma_floor_cross_px=0.30,
)

ARC_PARAMS = ArcFitParams(
    sector_half_angle_deg=60.0,
    ray_step_deg=2.0,
    radial_step_px=0.5,
    radial_inner_fraction=0.80,
    radial_outer_pad_px=6.0,
    median_filter_samples=5,
    min_gradient_snr=4.0,
    min_rays=20,
    min_inlier_fraction=0.50,
    max_residual_rms_px=2.0,
    tukey_c=4.685,
    along_sigma_scale=1.0,
    sigma_floor_along_px=1.00,
)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def render_haze_scene(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    *,
    r_limb_px: float = R_LIMB_PX,
    theta_rad: float = 0.0,
    falloff_px: float = 1.0,
    axis_ramp: float = 0.25,
    interior_cross_ramp: float = 0.0,
    peak_dn: float = PEAK_DN,
) -> NDArrayFloatType:
    """Render a mirror-symmetric hazy disc.

    The disc brightness is ``peak_dn`` times a logistic falloff through
    ``r_limb_px`` with scale ``falloff_px``, modulated by a linear ramp
    along the symmetry axis (which preserves mirror symmetry about it) and,
    when requested, by a ramp ACROSS the axis confined to the disc interior
    (which breaks it).

    Parameters:
        shape_vu: Image shape.
        center_vu: ``(v, u)`` disc center.
        r_limb_px: Radius of the logistic falloff.
        theta_rad: Symmetry-axis angle.
        falloff_px: Logistic scale length of the limb.
        axis_ramp: Fractional brightening per envelope radius along the
            axis toward the sub-solar side.
        interior_cross_ramp: Fractional brightening per envelope radius
            across the axis, tapered away by half the disc radius.
        peak_dn: Disc brightness at the center.

    Returns:
        The rendered image.
    """
    vv, uu = np.meshgrid(np.arange(shape_vu[0]), np.arange(shape_vu[1]), indexing='ij')
    dv = vv - center_vu[0]
    du = uu - center_vu[1]
    rho = np.hypot(dv, du)
    t = dv * math.sin(theta_rad) + du * math.cos(theta_rad)
    s = dv * math.cos(theta_rad) - du * math.sin(theta_rad)
    interior = 1.0 / (1.0 + np.exp((rho - 0.5 * r_limb_px) / 1.5))
    modulation = 1.0 + axis_ramp * t / r_limb_px + interior_cross_ramp * interior * s / r_limb_px
    image = peak_dn * modulation / (1.0 + np.exp((rho - r_limb_px) / falloff_px))
    rendered: NDArrayFloatType = image.astype(np.float64)
    return rendered


def add_gaussian_blob(
    image: NDArrayFloatType,
    center_vu: tuple[float, float],
    *,
    sigma_px: float,
    amplitude_dn: float,
) -> NDArrayFloatType:
    """Return the image with an additive Gaussian blob."""
    vv, uu = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    radius_sq = (vv - center_vu[0]) ** 2 + (uu - center_vu[1]) ** 2
    blob = amplitude_dn * np.exp(-radius_sq / (2.0 * sigma_px * sigma_px))
    return (image + blob).astype(np.float64)


def add_point_spikes(
    image: NDArrayFloatType,
    *,
    count: int,
    amplitude_dn: float,
    rng: np.random.Generator,
) -> NDArrayFloatType:
    """Return the image with single-pixel spikes at uniformly random positions."""
    out = image.copy()
    rows = rng.integers(0, image.shape[0], size=count)
    cols = rng.integers(0, image.shape[1], size=count)
    out[rows, cols] += amplitude_dn
    return out


def add_noise(
    image: NDArrayFloatType, *, sigma_dn: float, rng: np.random.Generator
) -> NDArrayFloatType:
    """Return the image with additive white Gaussian noise."""
    return (image + rng.normal(0.0, sigma_dn, size=image.shape)).astype(np.float64)


def disc_mask(
    shape_vu: tuple[int, int], center_vu: tuple[float, float], radius_px: float
) -> NDArrayBoolType:
    """Return a filled-disc boolean mask."""
    vv, uu = np.meshgrid(np.arange(shape_vu[0]), np.arange(shape_vu[1]), indexing='ij')
    inside: NDArrayBoolType = np.hypot(vv - center_vu[0], uu - center_vu[1]) <= radius_px
    return inside


def all_valid(shape_vu: tuple[int, int]) -> NDArrayBoolType:
    """Return an all-True static validity mask."""
    return np.ones(shape_vu, dtype=bool)


def displace(center_vu: tuple[float, float], offset_vu: tuple[float, float]) -> tuple[float, float]:
    """Return a center displaced by an offset."""
    return (center_vu[0] + offset_vu[0], center_vu[1] + offset_vu[1])


def axis_components(offset_vu: tuple[float, float], theta_rad: float) -> tuple[float, float]:
    """Return the ``(cross, along)`` components of an offset for an axis angle."""
    c_hat, a_hat = axis_vectors(theta_rad)
    vec = np.asarray(offset_vu, dtype=np.float64)
    return float(vec @ c_hat), float(vec @ a_hat)


def fit_scene(
    image: NDArrayFloatType,
    *,
    theta_rad: float = 0.0,
    center_vu: tuple[float, float] = CENTER_VU,
    contaminant_mask: NDArrayBoolType | None = None,
    window_px: float = WINDOW_PX,
    r_solid_px: float = R_SOLID_PX,
    r_env_px: float = R_ENV_PX,
    sym_params: SymmetryFitParams = SYM_PARAMS,
    arc_params: ArcFitParams = ARC_PARAMS,
    recenter_threshold_px: float = RECENTER_PX,
) -> tuple[SymmetryFitResult, ArcFitResult, tuple[float, float], bool]:
    """Run the full driver on a rendered scene with all pixels statically valid."""
    return fit_titan_center(
        image,
        all_valid((image.shape[0], image.shape[1])),
        center_vu,
        contaminant_mask=contaminant_mask,
        theta0_rad=theta_rad,
        r_solid_px=r_solid_px,
        r_env_px=r_env_px,
        window_px=window_px,
        sym_params=sym_params,
        arc_params=arc_params,
        recenter_threshold_px=recenter_threshold_px,
    )


def recovery_errors(
    planted_vu: tuple[float, float], measured_vu: tuple[float, float], theta_rad: float
) -> tuple[float, float]:
    """Return the absolute ``(cross, along)`` recovery errors of an offset."""
    planted = axis_components(planted_vu, theta_rad)
    measured = axis_components(measured_vu, theta_rad)
    return abs(measured[0] - planted[0]), abs(measured[1] - planted[1])


def _is_flagged(symmetry: SymmetryFitResult, arc: ArcFitResult) -> bool:
    """Return whether a fit disclaims its own result by a gate or an edge flag."""
    if symmetry.gate_failed is not None or arc.gate_failed is not None:
        return True
    return symmetry.at_edge or arc.at_edge


OFFSET_ANGLE_CASES = [(offset, theta) for theta in PLANTED_ANGLES for offset in PLANTED_OFFSETS]


# ---------------------------------------------------------------------------
# Test 1 / 2: planted-offset recovery, clean and noisy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), OFFSET_ANGLE_CASES)
def test_noiseless_recovery_meets_cross_track_bound(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=theta_rad)
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    cross, _ = recovery_errors(offset_vu, measured, theta_rad)
    assert cross <= CASE1_CROSS_PX


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), OFFSET_ANGLE_CASES)
def test_noiseless_recovery_meets_along_track_bound(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=theta_rad)
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    _, along = recovery_errors(offset_vu, measured, theta_rad)
    assert along <= CASE1_ALONG_PX


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), OFFSET_ANGLE_CASES)
def test_noisy_recovery_meets_cross_track_bound(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    rng = np.random.default_rng(abs(hash((offset_vu, theta_rad))) % 2**32)
    image = add_noise(
        render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=theta_rad),
        sigma_dn=NOISE_DN,
        rng=rng,
    )
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    cross, _ = recovery_errors(offset_vu, measured, theta_rad)
    assert cross <= CASE2_CROSS_PX


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), OFFSET_ANGLE_CASES)
def test_noisy_recovery_meets_along_track_bound(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    rng = np.random.default_rng(abs(hash((offset_vu, theta_rad))) % 2**32)
    image = add_noise(
        render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=theta_rad),
        sigma_dn=NOISE_DN,
        rng=rng,
    )
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    _, along = recovery_errors(offset_vu, measured, theta_rad)
    assert along <= CASE2_ALONG_PX


# ---------------------------------------------------------------------------
# Test 3: off-axis cloud blob
# ---------------------------------------------------------------------------


def _cloud_scene(theta_rad: float, offset_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return a noisy scene with a bright cloud injected off the symmetry axis."""
    center = displace(CENTER_VU, offset_vu)
    image = render_haze_scene(SHAPE_VU, center, theta_rad=theta_rad)
    c_hat, a_hat = axis_vectors(theta_rad)
    blob_center = (
        center[0] + 22.0 * c_hat[0] + 10.0 * a_hat[0],
        center[1] + 22.0 * c_hat[1] + 10.0 * a_hat[1],
    )
    image = add_gaussian_blob(image, blob_center, sigma_px=6.0, amplitude_dn=0.30 * PEAK_DN)
    return add_noise(image, sigma_dn=NOISE_DN, rng=np.random.default_rng(4242))


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_cloud_blob_keeps_cross_track_within_twice_case2(theta_rad: float) -> None:
    offset_vu = (0.3, -0.4)
    _, _, measured, _ = fit_scene(_cloud_scene(theta_rad, offset_vu), theta_rad=theta_rad)
    cross, _ = recovery_errors(offset_vu, measured, theta_rad)
    assert cross <= 2.0 * CASE2_CROSS_PX


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_cloud_blob_keeps_along_track_within_twice_case2(theta_rad: float) -> None:
    offset_vu = (0.3, -0.4)
    _, _, measured, _ = fit_scene(_cloud_scene(theta_rad, offset_vu), theta_rad=theta_rad)
    _, along = recovery_errors(offset_vu, measured, theta_rad)
    assert along <= 2.0 * CASE2_ALONG_PX


# ---------------------------------------------------------------------------
# Test 4: interior brightness gradient across the axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_interior_cross_gradient_does_not_bias_cross_track(theta_rad: float) -> None:
    offset_vu = (-1.2, 0.8)
    image = add_noise(
        render_haze_scene(
            SHAPE_VU,
            displace(CENTER_VU, offset_vu),
            theta_rad=theta_rad,
            interior_cross_ramp=0.30,
        ),
        sigma_dn=NOISE_DN,
        rng=np.random.default_rng(77),
    )
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    cross, _ = recovery_errors(offset_vu, measured, theta_rad)
    assert cross <= CASE2_CROSS_PX


# ---------------------------------------------------------------------------
# Test 5: void guard
# ---------------------------------------------------------------------------


def test_disc_half_off_frame_fails_valid_fraction_gate() -> None:
    center = (6.0, 85.0)
    image = render_haze_scene(SHAPE_VU, center, theta_rad=0.0)
    result = symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        center,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )
    assert result.gate_failed == 'valid_fraction'


def test_disc_half_off_frame_reports_low_valid_fraction() -> None:
    center = (6.0, 85.0)
    image = render_haze_scene(SHAPE_VU, center, theta_rad=0.0)
    result = symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        center,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )
    assert result.valid_fraction < SYM_PARAMS.min_valid_fraction


# ---------------------------------------------------------------------------
# Test 6: competing peak
# ---------------------------------------------------------------------------


_TWIN_SEPARATION_PX = 40.0
_TWIN_LIMB_PX = 12.0
_TWIN_ENV_PX = 14.0
_TWIN_WINDOW_PX = 25.0


def _twin_disc_scan(**overrides: float) -> SymmetryFitResult:
    """Scan two identical discs side by side, which have two rival mirror axes.

    Each disc is symmetric about its own center and the pair is symmetric
    about the midpoint, so the scan window holds equally good competing
    peaks and the estimate is genuinely ambiguous.
    """
    left = render_haze_scene(
        SHAPE_VU,
        (CENTER_VU[0] - _TWIN_SEPARATION_PX / 2.0, CENTER_VU[1]),
        r_limb_px=_TWIN_LIMB_PX,
        axis_ramp=0.0,
    )
    right = render_haze_scene(
        SHAPE_VU,
        (CENTER_VU[0] + _TWIN_SEPARATION_PX / 2.0, CENTER_VU[1]),
        r_limb_px=_TWIN_LIMB_PX,
        axis_ramp=0.0,
    )
    return symmetry_scan(
        np.maximum(left, right),
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=_TWIN_ENV_PX,
        window_px=_TWIN_WINDOW_PX,
        pass_pad_px=_TWIN_WINDOW_PX,
        capsule_half_extent_px=_TWIN_WINDOW_PX,
        params=replace(SYM_PARAMS, **overrides),
    )


def test_two_discs_side_by_side_fail_second_peak_gate() -> None:
    assert _twin_disc_scan().gate_failed == 'second_peak'


def test_two_discs_side_by_side_report_a_full_height_rival() -> None:
    assert _twin_disc_scan().second_peak_ratio == pytest.approx(1.0, abs=0.05)


def test_two_discs_side_by_side_pass_when_the_ratio_gate_is_relaxed() -> None:
    assert _twin_disc_scan(max_second_peak_ratio=1.01).gate_failed is None


def test_single_disc_reports_no_competing_peak() -> None:
    result = symmetry_scan(
        render_haze_scene(SHAPE_VU, CENTER_VU),
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )
    assert result.second_peak_ratio == 0.0


# ---------------------------------------------------------------------------
# Test 7: the constrained circle fit in isolation
# ---------------------------------------------------------------------------


def _arc_points(
    *,
    d_true: float,
    radius_px: float,
    half_angle_deg: float,
    n_rays: int,
    outlier_fraction: float = 0.0,
    noise_px: float = 0.0,
    seed: int = 3,
) -> NDArrayFloatType:
    """Return limb points on an axis-offset circle, optionally with outliers."""
    rng = np.random.default_rng(seed)
    phis = np.radians(np.linspace(-half_angle_deg, half_angle_deg, n_rays))
    # Solve for the distance from the origin to the circle along each ray.
    proj = d_true * np.cos(phis)
    rho = proj + np.sqrt(radius_px**2 - (d_true**2 - proj**2))
    if noise_px > 0.0:
        rho = rho + rng.normal(0.0, noise_px, size=rho.shape)
    n_out = round(outlier_fraction * n_rays)
    if n_out > 0:
        victims = rng.choice(n_rays, size=n_out, replace=False)
        rho[victims] -= 8.0
    return np.stack([rho * np.sin(phis), rho * np.cos(phis)], axis=1).astype(np.float64)


def test_circle_fit_recovers_center_with_outliers() -> None:
    points = _arc_points(
        d_true=3.5, radius_px=44.0, half_angle_deg=60.0, n_rays=61, outlier_fraction=0.20
    )
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert abs(result.along_track_px - 3.5) <= 0.05


def test_circle_fit_recovers_radius_with_outliers() -> None:
    points = _arc_points(
        d_true=3.5, radius_px=44.0, half_angle_deg=60.0, n_rays=61, outlier_fraction=0.20
    )
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert abs(result.radius_px - 44.0) <= 0.05


def test_circle_fit_rejects_every_outlier() -> None:
    points = _arc_points(
        d_true=3.5, radius_px=44.0, half_angle_deg=60.0, n_rays=61, outlier_fraction=0.20
    )
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert result.n_rays_inlier == 61 - 12


def test_circle_fit_short_arc_reports_large_sigma() -> None:
    points = _arc_points(d_true=2.0, radius_px=44.0, half_angle_deg=5.0, n_rays=41, noise_px=0.3)
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=20.0,
        params=replace(ARC_PARAMS, min_rays=10),
    )
    assert result.sigma_along_px > 3.0


def test_circle_fit_rejects_malformed_points() -> None:
    with pytest.raises(ValueError, match=r'points_vu must have shape \(N, 2\)'):
        constrained_circle_fit(
            np.zeros((5, 3)),
            (0.0, 0.0),
            (0.0, 1.0),
            r_solid_px=R_SOLID_PX,
            r_env_px=R_ENV_PX,
            window_px=WINDOW_PX,
            params=ARC_PARAMS,
        )


# ---------------------------------------------------------------------------
# Test 8: every named gate
# ---------------------------------------------------------------------------


def _scan(image: NDArrayFloatType, **overrides: float) -> SymmetryFitResult:
    """Scan the canonical scene with selected symmetry parameters overridden."""
    return symmetry_scan(
        image,
        all_valid((image.shape[0], image.shape[1])),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=replace(SYM_PARAMS, **overrides),
    )


def test_gate_name_valid_fraction() -> None:
    result = _scan(render_haze_scene(SHAPE_VU, CENTER_VU), min_valid_fraction=1.01)
    assert result.gate_failed == 'valid_fraction'


def test_gate_name_peak_score() -> None:
    result = _scan(render_haze_scene(SHAPE_VU, CENTER_VU), min_peak_score=1.01)
    assert result.gate_failed == 'peak_score'


def _arc_gate_result(params: ArcFitParams = ARC_PARAMS) -> ArcFitResult:
    """Fit the canonical arc points with the given arc parameters."""
    points = _arc_points(d_true=1.0, radius_px=44.0, half_angle_deg=60.0, n_rays=61)
    return constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=params,
    )


def test_gate_name_ray_yield() -> None:
    result = _arc_gate_result(replace(ARC_PARAMS, min_rays=200))
    assert result.gate_failed == 'ray_yield'


def test_gate_name_arc_inliers() -> None:
    points = _arc_points(
        d_true=1.0, radius_px=44.0, half_angle_deg=60.0, n_rays=61, outlier_fraction=0.40
    )
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=replace(ARC_PARAMS, min_inlier_fraction=0.95),
    )
    assert result.gate_failed == 'arc_inliers'


def test_gate_name_arc_radius() -> None:
    points = _arc_points(d_true=1.0, radius_px=44.0, half_angle_deg=60.0, n_rays=61)
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=200.0,
        r_env_px=260.0,
        window_px=WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert result.gate_failed == 'arc_radius'


def test_gate_name_arc_residual() -> None:
    points = _arc_points(
        d_true=1.0, radius_px=44.0, half_angle_deg=60.0, n_rays=61, noise_px=1.5, seed=9
    )
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=replace(ARC_PARAMS, max_residual_rms_px=0.10),
    )
    assert result.gate_failed == 'arc_residual'


def test_symmetry_peak_at_window_bound_sets_at_edge() -> None:
    offset = (0.0, 0.0)
    c_hat, _ = axis_vectors(0.0)
    center = displace(CENTER_VU, (float(20.0 * c_hat[0]), float(20.0 * c_hat[1])))
    image = render_haze_scene(SHAPE_VU, center, theta_rad=0.0)
    result = symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        displace(CENTER_VU, offset),
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )
    assert result.at_edge is True


def test_arc_shift_at_window_bound_sets_at_edge() -> None:
    points = _arc_points(d_true=12.0, radius_px=44.0, half_angle_deg=60.0, n_rays=61)
    result = constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert result.at_edge is True


def test_clean_scene_passes_every_symmetry_gate() -> None:
    result = _scan(render_haze_scene(SHAPE_VU, CENTER_VU))
    assert result.gate_failed is None


def test_clean_scene_passes_every_arc_gate() -> None:
    assert _arc_gate_result().gate_failed is None


# ---------------------------------------------------------------------------
# Test 9: sign conventions
# ---------------------------------------------------------------------------


def test_axis_vectors_match_the_definition() -> None:
    c_hat, _ = axis_vectors(0.4)
    assert c_hat == pytest.approx([math.cos(0.4), -math.sin(0.4)])


def test_axis_vectors_along_axis_points_sunward() -> None:
    _, a_hat = axis_vectors(0.4)
    assert a_hat == pytest.approx([math.sin(0.4), math.cos(0.4)])


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_displacement_along_c_hat_gives_positive_cross_track(theta_rad: float) -> None:
    c_hat, _ = axis_vectors(theta_rad)
    planted = (float(3.0 * c_hat[0]), float(3.0 * c_hat[1]))
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    symmetry, _, _, _ = fit_scene(image, theta_rad=theta_rad)
    assert symmetry.cross_track_px == pytest.approx(3.0, abs=CASE1_CROSS_PX)


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_displacement_along_c_hat_leaves_along_track_near_zero(theta_rad: float) -> None:
    c_hat, _ = axis_vectors(theta_rad)
    planted = (float(3.0 * c_hat[0]), float(3.0 * c_hat[1]))
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    _, arc, _, _ = fit_scene(image, theta_rad=theta_rad)
    assert arc.along_track_px == pytest.approx(0.0, abs=CASE1_ALONG_PX)


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_displacement_along_a_hat_gives_positive_along_track(theta_rad: float) -> None:
    _, a_hat = axis_vectors(theta_rad)
    planted = (float(3.0 * a_hat[0]), float(3.0 * a_hat[1]))
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    _, arc, _, _ = fit_scene(image, theta_rad=theta_rad)
    assert arc.along_track_px == pytest.approx(3.0, abs=CASE1_ALONG_PX)


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_displacement_along_a_hat_leaves_cross_track_near_zero(theta_rad: float) -> None:
    _, a_hat = axis_vectors(theta_rad)
    planted = (float(3.0 * a_hat[0]), float(3.0 * a_hat[1]))
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    symmetry, _, _, _ = fit_scene(image, theta_rad=theta_rad)
    assert symmetry.cross_track_px == pytest.approx(0.0, abs=CASE1_CROSS_PX)


@pytest.mark.parametrize('theta_rad', PLANTED_ANGLES)
def test_assembled_offset_matches_the_planted_displacement(theta_rad: float) -> None:
    planted = (1.7, -2.3)
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad)
    assert measured == pytest.approx(planted, abs=CASE1_ALONG_PX)


# ---------------------------------------------------------------------------
# Test 10: beside-limb intruder, masked and unmasked
# ---------------------------------------------------------------------------

_INTRUDER_CROSS_PX = 34.0
_INTRUDER_ALONG_PX = 30.0
_INTRUDER_RADIUS_PX = 9.0
_INTRUDER_THETA = 0.6
_INTRUDER_OFFSET = (4.0, 5.0)


def _intruder_position(center_vu: tuple[float, float]) -> tuple[float, float]:
    """Return the intruder's center for a scene whose main disc is at ``center_vu``."""
    c_hat, a_hat = axis_vectors(_INTRUDER_THETA)
    return (
        center_vu[0] + _INTRUDER_CROSS_PX * c_hat[0] + _INTRUDER_ALONG_PX * a_hat[0],
        center_vu[1] + _INTRUDER_CROSS_PX * c_hat[1] + _INTRUDER_ALONG_PX * a_hat[1],
    )


def _intruder_scene() -> NDArrayFloatType:
    """Return a scene where a small disc sits beside the displaced main limb."""
    center = displace(CENTER_VU, _INTRUDER_OFFSET)
    image = render_haze_scene(SHAPE_VU, center, theta_rad=_INTRUDER_THETA)
    intruder = render_haze_scene(
        SHAPE_VU,
        _intruder_position(center),
        r_limb_px=_INTRUDER_RADIUS_PX,
        theta_rad=_INTRUDER_THETA,
        axis_ramp=0.0,
    )
    return np.maximum(image, intruder)


def _predicted_intruder_mask() -> NDArrayBoolType:
    """Return the contaminant mask at the intruder's PREDICTED (undisplaced) place."""
    return disc_mask(SHAPE_VU, _intruder_position(CENTER_VU), _INTRUDER_RADIUS_PX + 2.0)


def test_masked_intruder_keeps_cross_track_within_case2() -> None:
    _, _, measured, _ = fit_scene(
        _intruder_scene(),
        theta_rad=_INTRUDER_THETA,
        contaminant_mask=_predicted_intruder_mask(),
    )
    cross, _ = recovery_errors(_INTRUDER_OFFSET, measured, _INTRUDER_THETA)
    assert cross <= CASE2_CROSS_PX


def test_masked_intruder_keeps_along_track_within_case2() -> None:
    _, _, measured, _ = fit_scene(
        _intruder_scene(),
        theta_rad=_INTRUDER_THETA,
        contaminant_mask=_predicted_intruder_mask(),
    )
    _, along = recovery_errors(_INTRUDER_OFFSET, measured, _INTRUDER_THETA)
    assert along <= CASE2_ALONG_PX


def test_unmasked_intruder_never_produces_an_unflagged_wrong_cross_track() -> None:
    symmetry, arc, measured, _ = fit_scene(_intruder_scene(), theta_rad=_INTRUDER_THETA)
    cross, _ = recovery_errors(_INTRUDER_OFFSET, measured, _INTRUDER_THETA)
    assert cross <= 2.0 * CASE2_CROSS_PX or _is_flagged(symmetry, arc)


def test_unmasked_intruder_never_produces_an_unflagged_wrong_along_track() -> None:
    symmetry, arc, measured, _ = fit_scene(_intruder_scene(), theta_rad=_INTRUDER_THETA)
    _, along = recovery_errors(_INTRUDER_OFFSET, measured, _INTRUDER_THETA)
    assert along <= 2.0 * CASE2_ALONG_PX or _is_flagged(symmetry, arc)


# ---------------------------------------------------------------------------
# Test 11: unmasked point sources
# ---------------------------------------------------------------------------


def test_point_sources_keep_cross_track_within_case2() -> None:
    offset_vu = (0.7, -1.3)
    image = add_point_spikes(
        render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=0.9),
        count=20,
        amplitude_dn=5.0 * PEAK_DN,
        rng=np.random.default_rng(11),
    )
    _, _, measured, _ = fit_scene(image, theta_rad=0.9)
    cross, _ = recovery_errors(offset_vu, measured, 0.9)
    assert cross <= CASE2_CROSS_PX


def test_point_sources_keep_along_track_within_case2() -> None:
    offset_vu = (0.7, -1.3)
    image = add_point_spikes(
        render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=0.9),
        count=20,
        amplitude_dn=5.0 * PEAK_DN,
        rng=np.random.default_rng(11),
    )
    _, _, measured, _ = fit_scene(image, theta_rad=0.9)
    _, along = recovery_errors(offset_vu, measured, 0.9)
    assert along <= CASE2_ALONG_PX


# ---------------------------------------------------------------------------
# Test 12: the recenter pass
# ---------------------------------------------------------------------------

_RECENTER_WINDOW_PX = 20.0


def _recentered_fit(
    planted_vu: tuple[float, float], theta_rad: float
) -> tuple[SymmetryFitResult, ArcFitResult, tuple[float, float], bool]:
    """Fit a scene displaced by ``planted_vu`` with a window wide enough to recenter."""
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted_vu), theta_rad=theta_rad)
    return fit_scene(image, theta_rad=theta_rad, window_px=_RECENTER_WINDOW_PX)


def test_large_along_track_displacement_triggers_the_recenter_pass() -> None:
    _, a_hat = axis_vectors(0.5)
    shift = 0.8 * _RECENTER_WINDOW_PX
    planted = (float(shift * a_hat[0]), float(shift * a_hat[1]))
    _, _, _, recentered = _recentered_fit(planted, 0.5)
    assert recentered is True


def test_recentered_run_recovers_the_along_track_displacement() -> None:
    _, a_hat = axis_vectors(0.5)
    shift = 0.8 * _RECENTER_WINDOW_PX
    planted = (float(shift * a_hat[0]), float(shift * a_hat[1]))
    _, _, measured, _ = _recentered_fit(planted, 0.5)
    _, along = recovery_errors(planted, measured, 0.5)
    assert along <= CASE2_ALONG_PX


def test_recentered_run_recovers_the_cross_track_displacement() -> None:
    _, a_hat = axis_vectors(0.5)
    shift = 0.8 * _RECENTER_WINDOW_PX
    planted = (float(shift * a_hat[0]), float(shift * a_hat[1]))
    _, _, measured, _ = _recentered_fit(planted, 0.5)
    cross, _ = recovery_errors(planted, measured, 0.5)
    assert cross <= CASE2_CROSS_PX


def test_small_along_track_displacement_leaves_recentered_false() -> None:
    _, a_hat = axis_vectors(0.5)
    shift = 0.5 * RECENTER_PX
    planted = (float(shift * a_hat[0]), float(shift * a_hat[1]))
    _, _, _, recentered = _recentered_fit(planted, 0.5)
    assert recentered is False


_SMALL_ENV_PX = 12.0
_SMALL_SOLID_PX = 10.0
_SMALL_LIMB_PX = 11.0
_SMALL_WINDOW_PX = 3.0 * _SMALL_ENV_PX
_SMALL_THETA = 0.35


def _small_disc_scene() -> tuple[NDArrayFloatType, tuple[float, float]]:
    """Return a noisy small-disc scene displaced far along the axis, and its offset.

    The window is three times the envelope radius, so the body sits well
    outside a plain annulus about the predicted center: only the pass-1
    capsule reaches it.
    """
    _, a_hat = axis_vectors(_SMALL_THETA)
    shift = 0.8 * _SMALL_WINDOW_PX
    planted = (float(shift * a_hat[0]), float(shift * a_hat[1]))
    image = render_haze_scene(
        SHAPE_VU,
        displace(CENTER_VU, planted),
        r_limb_px=_SMALL_LIMB_PX,
        theta_rad=_SMALL_THETA,
    )
    return add_noise(image, sigma_dn=NOISE_DN, rng=np.random.default_rng(5)), planted


def _small_disc_scan(capsule_half_extent_px: float) -> SymmetryFitResult:
    """Scan the small-disc scene with the given capsule half extent."""
    image, _ = _small_disc_scene()
    return symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=_SMALL_THETA,
        r_env_px=_SMALL_ENV_PX,
        window_px=_SMALL_WINDOW_PX,
        pass_pad_px=_SMALL_WINDOW_PX,
        capsule_half_extent_px=capsule_half_extent_px,
        params=SYM_PARAMS,
    )


def _small_disc_fit() -> tuple[tuple[float, float], tuple[float, float], bool]:
    """Return the planted offset, the measured offset and the recenter flag."""
    image, planted = _small_disc_scene()
    _, _, measured, recentered = fit_scene(
        image,
        theta_rad=_SMALL_THETA,
        window_px=_SMALL_WINDOW_PX,
        r_solid_px=_SMALL_SOLID_PX,
        r_env_px=_SMALL_ENV_PX,
    )
    return planted, measured, recentered


def test_capsule_annulus_recovers_a_small_disc_in_a_large_window() -> None:
    planted, measured, _ = _small_disc_fit()
    _, along = recovery_errors(planted, measured, _SMALL_THETA)
    assert along <= CASE2_ALONG_PX


def test_capsule_annulus_recovers_the_small_disc_cross_track() -> None:
    planted, measured, _ = _small_disc_fit()
    cross, _ = recovery_errors(planted, measured, _SMALL_THETA)
    assert cross <= CASE2_CROSS_PX


def test_capsule_annulus_case_uses_the_recenter_pass() -> None:
    _, _, recentered = _small_disc_fit()
    assert recentered is True


def test_capsule_annulus_finds_a_strong_peak_a_plain_annulus_misses() -> None:
    assert _small_disc_scan(_SMALL_WINDOW_PX).peak_score >= SYM_PARAMS.min_peak_score


def test_plain_annulus_fails_the_peak_score_gate_on_the_small_disc() -> None:
    assert _small_disc_scan(0.0).gate_failed == 'peak_score'


_DOUBLE_COUNT_THETA = 0.8
_DOUBLE_COUNT_CROSS_PX = 4.0


def _double_count_case() -> tuple[SymmetryFitResult, tuple[float, float], bool]:
    """Recenter a scene displaced in BOTH the cross-track and along-track senses."""
    c_hat, a_hat = axis_vectors(_DOUBLE_COUNT_THETA)
    along_true = 0.8 * _RECENTER_WINDOW_PX
    planted = (
        float(_DOUBLE_COUNT_CROSS_PX * c_hat[0] + along_true * a_hat[0]),
        float(_DOUBLE_COUNT_CROSS_PX * c_hat[1] + along_true * a_hat[1]),
    )
    symmetry, _, measured, recentered = _recentered_fit(planted, _DOUBLE_COUNT_THETA)
    return symmetry, measured, recentered


def test_two_component_displacement_triggers_the_recenter_pass() -> None:
    _, _, recentered = _double_count_case()
    assert recentered is True


def test_recenter_does_not_double_count_the_cross_track_term() -> None:
    symmetry, measured, _ = _double_count_case()
    cross, _ = axis_components(measured, _DOUBLE_COUNT_THETA)
    assert cross == pytest.approx(symmetry.cross_track_px, abs=1.0e-9)


def test_recenter_recovers_both_planted_components() -> None:
    _, measured, _ = _double_count_case()
    cross, _ = axis_components(measured, _DOUBLE_COUNT_THETA)
    assert cross == pytest.approx(_DOUBLE_COUNT_CROSS_PX, abs=CASE2_CROSS_PX)


def test_recenter_accumulates_the_along_track_terms_of_both_passes() -> None:
    _, measured, _ = _double_count_case()
    _, along = axis_components(measured, _DOUBLE_COUNT_THETA)
    assert along == pytest.approx(0.8 * _RECENTER_WINDOW_PX, abs=CASE2_ALONG_PX)


# ---------------------------------------------------------------------------
# Building blocks: grid resampling, profiles, limb extraction
# ---------------------------------------------------------------------------


def test_resample_rotated_grid_has_the_expected_shape() -> None:
    image = render_haze_scene(SHAPE_VU, CENTER_VU)
    grid, _ = resample_rotated_grid(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        theta_rad=0.0,
        s_half_extent_px=20.0,
        t_half_extent_px=10.0,
    )
    assert grid.shape == (41, 21)


def test_resample_rotated_grid_is_rotation_consistent() -> None:
    image = render_haze_scene(SHAPE_VU, CENTER_VU, theta_rad=0.0, axis_ramp=0.0)
    flat, _ = resample_rotated_grid(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        theta_rad=0.0,
        s_half_extent_px=20.0,
        t_half_extent_px=20.0,
    )
    turned, _ = resample_rotated_grid(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        theta_rad=1.1,
        s_half_extent_px=20.0,
        t_half_extent_px=20.0,
    )
    assert turned == pytest.approx(flat, abs=1.0)


def test_resample_rotated_grid_marks_off_frame_samples_invalid() -> None:
    image = render_haze_scene(SHAPE_VU, (4.0, 4.0))
    _, grid_valid = resample_rotated_grid(
        image,
        all_valid(SHAPE_VU),
        (4.0, 4.0),
        theta_rad=0.0,
        s_half_extent_px=20.0,
        t_half_extent_px=20.0,
    )
    assert not bool(grid_valid[0, 0])


def test_resample_rotated_grid_rejects_a_mismatched_valid_mask() -> None:
    with pytest.raises(ValueError, match='valid_mask must match the image shape'):
        resample_rotated_grid(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            np.ones((4, 4), dtype=bool),
            CENTER_VU,
            theta_rad=0.0,
            s_half_extent_px=10.0,
            t_half_extent_px=10.0,
        )


def test_resample_rotated_grid_rejects_a_non_positive_extent() -> None:
    with pytest.raises(ValueError, match='s_half_extent_px must be positive'):
        resample_rotated_grid(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            all_valid(SHAPE_VU),
            CENTER_VU,
            theta_rad=0.0,
            s_half_extent_px=0.0,
            t_half_extent_px=10.0,
        )


def test_symmetry_scan_rejects_a_tiny_window() -> None:
    with pytest.raises(ValueError, match='window_px must be at least 1 pixel'):
        symmetry_scan(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            all_valid(SHAPE_VU),
            CENTER_VU,
            contaminant_mask=None,
            theta0_rad=0.0,
            r_env_px=R_ENV_PX,
            window_px=0.5,
            pass_pad_px=0.0,
            params=SYM_PARAMS,
        )


def test_symmetry_scan_rejects_a_non_positive_envelope_radius() -> None:
    with pytest.raises(ValueError, match='r_env_px must be positive'):
        symmetry_scan(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            all_valid(SHAPE_VU),
            CENTER_VU,
            contaminant_mask=None,
            theta0_rad=0.0,
            r_env_px=0.0,
            window_px=WINDOW_PX,
            pass_pad_px=0.0,
            params=SYM_PARAMS,
        )


def _canonical_profiles(
    contaminant_mask: NDArrayBoolType | None = None,
    mask_shift_vu: tuple[float, float] = (0.0, 0.0),
    pass_pad_px: float = 0.0,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Sample the canonical scene along a fan of sunward rays."""
    image = render_haze_scene(SHAPE_VU, CENTER_VU, theta_rad=0.0)
    phis = np.radians(np.arange(-60.0, 60.1, 2.0))
    return radial_profiles(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=contaminant_mask,
        mask_shift_vu=mask_shift_vu,
        axis_dir_vu=(0.0, 1.0),
        pass_pad_px=pass_pad_px,
        phi_rad_list=phis,
        r_start_px=32.0,
        r_stop_px=62.0,
        r_step_px=0.5,
    )


def test_radial_profiles_have_one_row_per_ray() -> None:
    profiles, _ = _canonical_profiles()
    assert profiles.shape == (61, 61)


def test_radial_profiles_fall_off_across_the_limb() -> None:
    profiles, _ = _canonical_profiles()
    assert profiles[30, -1] < 0.01 * profiles[30, 0]


def test_radial_profiles_mark_masked_samples_invalid() -> None:
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + 44.0), 3.0)
    _, valid = _canonical_profiles(contaminant_mask=mask)
    assert not bool(valid[30, 24])


def test_radial_profiles_shift_the_mask_with_the_hypothesis() -> None:
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + 44.0), 3.0)
    _, valid = _canonical_profiles(contaminant_mask=mask, mask_shift_vu=(0.0, 10.0))
    assert bool(valid[30, 24])


def test_radial_profiles_dilate_the_mask_along_the_axis() -> None:
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + 44.0), 3.0)
    _, valid = _canonical_profiles(
        contaminant_mask=mask, mask_shift_vu=(0.0, 10.0), pass_pad_px=12.0
    )
    assert not bool(valid[30, 24])


def test_radial_profiles_reject_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match='r_step_px must be positive'):
        radial_profiles(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            all_valid(SHAPE_VU),
            CENTER_VU,
            contaminant_mask=None,
            mask_shift_vu=(0.0, 0.0),
            axis_dir_vu=(0.0, 1.0),
            pass_pad_px=0.0,
            phi_rad_list=np.array([0.0]),
            r_start_px=10.0,
            r_stop_px=20.0,
            r_step_px=0.0,
        )


def test_radial_profiles_reject_an_empty_radius_range() -> None:
    with pytest.raises(ValueError, match='r_stop_px must exceed r_start_px'):
        radial_profiles(
            render_haze_scene(SHAPE_VU, CENTER_VU),
            all_valid(SHAPE_VU),
            CENTER_VU,
            contaminant_mask=None,
            mask_shift_vu=(0.0, 0.0),
            axis_dir_vu=(0.0, 1.0),
            pass_pad_px=0.0,
            phi_rad_list=np.array([0.0]),
            r_start_px=20.0,
            r_stop_px=20.0,
            r_step_px=0.5,
        )


def test_limb_radii_find_the_rendered_limb() -> None:
    profiles, valid = _canonical_profiles()
    rho, ray_ok = limb_radii_from_profiles(
        profiles,
        valid,
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=R_SOLID_PX - WINDOW_PX,
        window_px_hi=R_ENV_PX + WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert rho[ray_ok] == pytest.approx(R_LIMB_PX, abs=0.1)


def test_limb_radii_accept_every_clean_ray() -> None:
    profiles, valid = _canonical_profiles()
    _, ray_ok = limb_radii_from_profiles(
        profiles,
        valid,
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=R_SOLID_PX - WINDOW_PX,
        window_px_hi=R_ENV_PX + WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert int(ray_ok.sum()) == 61


def test_limb_radii_drop_rays_with_a_masked_limb() -> None:
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + 44.0), 3.0)
    profiles, valid = _canonical_profiles(contaminant_mask=mask)
    _, ray_ok = limb_radii_from_profiles(
        profiles,
        valid,
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=R_SOLID_PX - WINDOW_PX,
        window_px_hi=R_ENV_PX + WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert not bool(ray_ok[30])


def test_limb_radii_reject_mismatched_validity() -> None:
    profiles, _ = _canonical_profiles()
    with pytest.raises(ValueError, match='profile_valid must match the profile shape'):
        limb_radii_from_profiles(
            profiles,
            np.ones((3, 3), dtype=bool),
            r_start_px=32.0,
            r_step_px=0.5,
            r_solid_px=R_SOLID_PX,
            window_px_lo=R_SOLID_PX - WINDOW_PX,
            window_px_hi=R_ENV_PX + WINDOW_PX,
            params=ARC_PARAMS,
        )


def test_limb_radii_reject_one_dimensional_profiles() -> None:
    with pytest.raises(ValueError, match='profiles must be 2-D'):
        limb_radii_from_profiles(
            np.zeros(10),
            np.ones(10, dtype=bool),
            r_start_px=32.0,
            r_step_px=0.5,
            r_solid_px=R_SOLID_PX,
            window_px_lo=R_SOLID_PX - WINDOW_PX,
            window_px_hi=R_ENV_PX + WINDOW_PX,
            params=ARC_PARAMS,
        )


def test_radial_profiles_reject_a_one_dimensional_image() -> None:
    with pytest.raises(ValueError, match='image must be 2-D'):
        radial_profiles(
            np.zeros(10),
            np.ones(10, dtype=bool),
            CENTER_VU,
            contaminant_mask=None,
            mask_shift_vu=(0.0, 0.0),
            axis_dir_vu=(0.0, 1.0),
            pass_pad_px=0.0,
            phi_rad_list=np.array([0.0]),
            r_start_px=10.0,
            r_stop_px=20.0,
            r_step_px=0.5,
        )


def _limb_radii_of(
    profiles: NDArrayFloatType, *, window_px_lo: float, window_px_hi: float
) -> NDArrayBoolType:
    """Return which rays of a hand-built profile block yield a limb radius."""
    _, ray_ok = limb_radii_from_profiles(
        profiles,
        np.ones(profiles.shape, dtype=bool),
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=window_px_lo,
        window_px_hi=window_px_hi,
        params=ARC_PARAMS,
    )
    return ray_ok


def test_limb_radii_yield_nothing_when_the_search_window_is_empty() -> None:
    profiles, _ = _canonical_profiles()
    ray_ok = _limb_radii_of(profiles, window_px_lo=100.0, window_px_hi=101.0)
    assert int(ray_ok.sum()) == 0


def test_limb_radii_drop_a_ray_that_only_brightens_outward() -> None:
    # Brightening throughout, but slowest in the middle, so the least
    # positive gradient sits inside the window rather than on its bound.
    samples = np.arange(61, dtype=np.float64)
    rate = 1.0 + (samples - 30.0) ** 2 / 900.0
    rising = np.tile(np.cumsum(rate), (5, 1))
    ray_ok = _limb_radii_of(rising, window_px_lo=-1.0e6, window_px_hi=1.0e6)
    assert int(ray_ok.sum()) == 0


def test_limb_radii_drop_a_ray_whose_falloff_is_no_stronger_than_its_ripple() -> None:
    phase = 2.0 * np.pi * 3.0 * np.arange(61, dtype=np.float64) / 61.0
    ripple = np.tile(100.0 * np.cos(phase), (5, 1))
    ray_ok = _limb_radii_of(ripple, window_px_lo=-1.0e6, window_px_hi=1.0e6)
    assert int(ray_ok.sum()) == 0


def _masked_limb_scan(pass_pad_px: float) -> SymmetryFitResult:
    """Scan the canonical scene with a contaminant disc sitting on the limb."""
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0] + R_LIMB_PX, CENTER_VU[1]), 5.0)
    return symmetry_scan(
        render_haze_scene(SHAPE_VU, CENTER_VU),
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=mask,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=pass_pad_px,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )


def test_undilated_contaminant_mask_removes_annulus_pairs() -> None:
    assert _masked_limb_scan(0.0).valid_fraction < 1.0


def test_dilating_the_contaminant_mask_removes_more_pairs() -> None:
    assert _masked_limb_scan(WINDOW_PX).valid_fraction < _masked_limb_scan(0.0).valid_fraction


def test_featureless_image_fails_the_peak_score_gate() -> None:
    result = _scan(np.full(SHAPE_VU, 500.0, dtype=np.float64))
    assert result.gate_failed == 'peak_score'


def test_featureless_image_reports_no_competing_peak() -> None:
    result = _scan(np.full(SHAPE_VU, 500.0, dtype=np.float64))
    assert result.second_peak_ratio == 0.0


def test_featureless_image_reports_an_undefined_peak_score() -> None:
    result = _scan(np.full(SHAPE_VU, 500.0, dtype=np.float64))
    assert math.isnan(result.peak_score)


# ---------------------------------------------------------------------------
# Regressions: failures that once produced a confident wrong answer
# ---------------------------------------------------------------------------


def _along_offset_fit(
    along_px: float, *, window_px: float = WINDOW_PX, theta_rad: float = 0.3
) -> tuple[SymmetryFitResult, ArcFitResult, float]:
    """Fit a scene displaced purely along the axis; return the along-track error."""
    _, a_hat = axis_vectors(theta_rad)
    planted = (float(along_px * a_hat[0]), float(along_px * a_hat[1]))
    image = render_haze_scene(SHAPE_VU, displace(CENTER_VU, planted), theta_rad=theta_rad)
    symmetry, arc, measured, _ = fit_scene(image, theta_rad=theta_rad, window_px=window_px)
    _, along = recovery_errors(planted, measured, theta_rad)
    return symmetry, arc, along


# Displacements that used to leave the robust fit with zero inliers: the
# unclipped Gauss-Newton step is what keeps the residuals from settling into a
# uniform offset that the spread-based robust scale reads as all-outliers.
_MID_WINDOW_ALONG_PX = (4.75, 5.0, 5.25, 5.5)


@pytest.mark.parametrize('along_px', _MID_WINDOW_ALONG_PX)
def test_mid_window_along_offset_keeps_its_inliers(along_px: float) -> None:
    _, arc, _ = _along_offset_fit(along_px, window_px=12.0)
    assert arc.n_rays_inlier == arc.n_rays_total


@pytest.mark.parametrize('along_px', _MID_WINDOW_ALONG_PX)
def test_mid_window_along_offset_passes_every_arc_gate(along_px: float) -> None:
    _, arc, _ = _along_offset_fit(along_px, window_px=12.0)
    assert arc.gate_failed is None


@pytest.mark.parametrize('along_px', _MID_WINDOW_ALONG_PX)
def test_mid_window_along_offset_recovers_within_case1(along_px: float) -> None:
    _, _, along = _along_offset_fit(along_px, window_px=12.0)
    assert along <= CASE1_ALONG_PX


# Displacements that put the true limb beyond the ray search window.  The
# limb search must not saturate against the window bound and report the bound
# as a detection: either the recenter pass reaches the body or a gate fires.
_BEYOND_WINDOW_ALONG_PX = (12.0, 14.0, 16.0, 18.0, 20.0, 25.0)


@pytest.mark.parametrize('along_px', _BEYOND_WINDOW_ALONG_PX)
def test_truth_beyond_the_search_window_is_recovered_or_flagged(along_px: float) -> None:
    symmetry, arc, along = _along_offset_fit(along_px, theta_rad=0.0)
    assert along <= CASE1_ALONG_PX or _is_flagged(symmetry, arc)


def test_limb_search_never_reports_a_window_saturating_radius() -> None:
    along_px = 16.0
    image = render_haze_scene(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + along_px), theta_rad=0.0)
    phis = np.radians(np.arange(-60.0, 60.1, 2.0))
    profiles, valid = radial_profiles(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        mask_shift_vu=(0.0, 0.0),
        axis_dir_vu=(0.0, 1.0),
        pass_pad_px=0.0,
        phi_rad_list=phis,
        r_start_px=32.0,
        r_stop_px=62.0,
        r_step_px=0.5,
    )
    window_hi = R_ENV_PX + WINDOW_PX
    rho, ray_ok = limb_radii_from_profiles(
        profiles,
        valid,
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=R_SOLID_PX - WINDOW_PX,
        window_px_hi=window_hi,
        params=ARC_PARAMS,
    )
    assert float(np.max(rho[ray_ok])) < window_hi


def test_interior_contaminant_costs_no_rays() -> None:
    mask = disc_mask(SHAPE_VU, (CENTER_VU[0], CENTER_VU[1] + 0.85 * R_SOLID_PX), 3.0)
    profiles, valid = _canonical_profiles(contaminant_mask=mask)
    _, ray_ok = limb_radii_from_profiles(
        profiles,
        valid,
        r_start_px=32.0,
        r_step_px=0.5,
        r_solid_px=R_SOLID_PX,
        window_px_lo=R_SOLID_PX - WINDOW_PX,
        window_px_hi=R_ENV_PX + WINDOW_PX,
        params=ARC_PARAMS,
    )
    assert int(ray_ok.sum()) == 61


def test_symmetry_peak_at_window_bound_reports_the_widest_sigma() -> None:
    c_hat, _ = axis_vectors(0.0)
    center = displace(CENTER_VU, (float(20.0 * c_hat[0]), float(20.0 * c_hat[1])))
    image = render_haze_scene(SHAPE_VU, center, theta_rad=0.0)
    result = symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=SYM_PARAMS,
    )
    assert result.sigma_cross_px == pytest.approx(WINDOW_PX)


def _fully_rejected_fit() -> ArcFitResult:
    """Fit whose Tukey cutoff is so tight that every ray is rejected."""
    # An even ray count keeps the initial median radius off every sample, so
    # no residual starts at exactly zero and survives the tightened cutoff.
    points = _arc_points(
        d_true=1.0, radius_px=44.0, half_angle_deg=60.0, n_rays=60, noise_px=0.3, seed=5
    )
    return constrained_circle_fit(
        points,
        (0.0, 0.0),
        (0.0, 1.0),
        r_solid_px=R_SOLID_PX,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        params=replace(ARC_PARAMS, tukey_c=1.0e-9),
    )


def test_fit_with_no_inliers_fails_the_inlier_gate() -> None:
    assert _fully_rejected_fit().gate_failed == 'arc_inliers'


def test_fit_with_no_inliers_reports_an_undefined_residual() -> None:
    assert math.isnan(_fully_rejected_fit().residual_rms_px)


_RIVAL_SEPARATION_PX = 40.0


def test_a_rival_peak_at_the_window_bound_fails_the_second_peak_gate() -> None:
    half = _RIVAL_SEPARATION_PX / 2.0
    left = render_haze_scene(
        SHAPE_VU, (CENTER_VU[0] - half, CENTER_VU[1]), r_limb_px=_TWIN_LIMB_PX, axis_ramp=0.0
    )
    right = render_haze_scene(
        SHAPE_VU, (CENTER_VU[0] + half, CENTER_VU[1]), r_limb_px=_TWIN_LIMB_PX, axis_ramp=0.0
    )
    result = symmetry_scan(
        np.maximum(left, right),
        all_valid(SHAPE_VU),
        (CENTER_VU[0] - half, CENTER_VU[1]),
        contaminant_mask=None,
        theta0_rad=0.0,
        r_env_px=_TWIN_ENV_PX,
        window_px=half,
        pass_pad_px=half,
        capsule_half_extent_px=half,
        params=SYM_PARAMS,
    )
    assert result.gate_failed == 'second_peak'


# ---------------------------------------------------------------------------
# The contaminant mask must ride the offset hypothesis, not sit still
# ---------------------------------------------------------------------------

_ANCHOR_SHAPE_VU = (320, 320)
_ANCHOR_CENTER_VU = (160.0, 160.0)
_ANCHOR_WINDOW_PX = 50.0
_ANCHOR_CROSS_PX = 2.0
_ANCHOR_ALONG_PX = 40.0
_ANCHOR_INTRUDER_VU = (36.0, 6.0)


def _anchor_case(mask_shift_vu: tuple[float, float]) -> SymmetryFitResult:
    """Scan a recentered grid with the contaminant mask shifted as given.

    The scene is displaced by 40 px along the axis, and a bright intruder
    rides with it while its mask entry stays at the predicted position.  Only
    a mask anchored at the predicted center -- that is, read at the sample
    position minus the accumulated shift -- lands back on the intruder.
    """
    planted = (
        _ANCHOR_CROSS_PX * 1.0 + _ANCHOR_ALONG_PX * 0.0,
        _ANCHOR_CROSS_PX * 0.0 + _ANCHOR_ALONG_PX * 1.0,
    )
    disc_center = displace(_ANCHOR_CENTER_VU, planted)
    predicted_intruder = displace(_ANCHOR_CENTER_VU, _ANCHOR_INTRUDER_VU)
    actual_intruder = displace(disc_center, _ANCHOR_INTRUDER_VU)
    scene = np.maximum(
        render_haze_scene(_ANCHOR_SHAPE_VU, disc_center, theta_rad=0.0),
        render_haze_scene(
            _ANCHOR_SHAPE_VU,
            actual_intruder,
            r_limb_px=10.0,
            axis_ramp=0.0,
            peak_dn=3.0 * PEAK_DN,
        ),
    )
    return symmetry_scan(
        scene,
        all_valid(_ANCHOR_SHAPE_VU),
        (_ANCHOR_CENTER_VU[0], _ANCHOR_CENTER_VU[1] + _ANCHOR_ALONG_PX),
        contaminant_mask=disc_mask(_ANCHOR_SHAPE_VU, predicted_intruder, 13.0),
        theta0_rad=0.0,
        r_env_px=R_ENV_PX,
        window_px=_ANCHOR_WINDOW_PX,
        pass_pad_px=RECENTER_PX,
        capsule_half_extent_px=0.0,
        mask_shift_vu=mask_shift_vu,
        params=SYM_PARAMS,
    )


def test_mask_anchored_at_the_predicted_center_excludes_the_intruder() -> None:
    result = _anchor_case((0.0, _ANCHOR_ALONG_PX))
    assert abs(result.cross_track_px - _ANCHOR_CROSS_PX) <= CASE2_CROSS_PX


def test_mask_left_static_on_the_moved_grid_measurably_degrades_the_fit() -> None:
    anchored = _anchor_case((0.0, _ANCHOR_ALONG_PX))
    static = _anchor_case((0.0, 0.0))
    error = abs(static.cross_track_px - _ANCHOR_CROSS_PX)
    assert error > 10.0 * abs(anchored.cross_track_px - _ANCHOR_CROSS_PX)


def test_mask_left_static_on_the_moved_grid_costs_peak_score() -> None:
    anchored = _anchor_case((0.0, _ANCHOR_ALONG_PX))
    static = _anchor_case((0.0, 0.0))
    assert static.peak_score < anchored.peak_score


_TILTED_THETA = 0.5
_ASSUMED_THETA = _TILTED_THETA - math.radians(4.0)


def _tilted_axis_scan(**overrides: float) -> SymmetryFitResult:
    """Scan a disc whose symmetry axis is tilted from the assumed one."""
    image = render_haze_scene(SHAPE_VU, CENTER_VU, theta_rad=_TILTED_THETA, axis_ramp=0.6)
    return symmetry_scan(
        image,
        all_valid(SHAPE_VU),
        CENTER_VU,
        contaminant_mask=None,
        theta0_rad=_ASSUMED_THETA,
        r_env_px=R_ENV_PX,
        window_px=WINDOW_PX,
        pass_pad_px=WINDOW_PX,
        capsule_half_extent_px=WINDOW_PX,
        params=replace(SYM_PARAMS, **overrides),
    )


def test_angle_refinement_recovers_a_tilted_symmetry_axis() -> None:
    result = _tilted_axis_scan(angle_refine_deg=5.0, angle_refine_min_gain=0.0)
    assert abs(result.theta_rad - _TILTED_THETA) < math.radians(2.0)


def test_angle_refinement_keeps_the_supplied_axis_without_enough_gain() -> None:
    result = _tilted_axis_scan(angle_refine_deg=5.0, angle_refine_min_gain=1.0)
    assert result.theta_rad == pytest.approx(_ASSUMED_THETA)


# The configured default turns angle refinement on, so the whole driver has to
# meet its recovery bounds with the search running, not just with it disabled.
_REFINING_SYM_PARAMS = replace(SYM_PARAMS, angle_refine_deg=5.0)

_REFINED_CASES = [(offset, theta) for theta in (0.0, 1.9) for offset in PLANTED_OFFSETS]


def _refined_fit(offset_vu: tuple[float, float], theta_rad: float) -> tuple[float, float]:
    """Recover a noisy planted offset with the shipped angle-refinement range."""
    rng = np.random.default_rng(2026)
    image = add_noise(
        render_haze_scene(SHAPE_VU, displace(CENTER_VU, offset_vu), theta_rad=theta_rad),
        sigma_dn=NOISE_DN,
        rng=rng,
    )
    _, _, measured, _ = fit_scene(image, theta_rad=theta_rad, sym_params=_REFINING_SYM_PARAMS)
    return recovery_errors(offset_vu, measured, theta_rad)


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), _REFINED_CASES)
def test_angle_refinement_keeps_cross_track_within_case2(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    cross, _ = _refined_fit(offset_vu, theta_rad)
    assert cross <= CASE2_CROSS_PX


@pytest.mark.parametrize(('offset_vu', 'theta_rad'), _REFINED_CASES)
def test_angle_refinement_keeps_along_track_within_case2(
    offset_vu: tuple[float, float], theta_rad: float
) -> None:
    _, along = _refined_fit(offset_vu, theta_rad)
    assert along <= CASE2_ALONG_PX


def test_angle_refinement_is_disabled_by_a_zero_range() -> None:
    result = _tilted_axis_scan(angle_refine_deg=0.0)
    assert result.theta_rad == pytest.approx(_ASSUMED_THETA)
