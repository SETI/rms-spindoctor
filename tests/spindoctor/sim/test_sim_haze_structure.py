"""Render-level tests for the haze layer's symmetry-breaking structure.

The base haze is exactly mirror-symmetric about the sunward axis, which is
the assumption a haze navigator measures pointing from.  Each optional
structure key breaks that symmetry along one axis, and each has one test
here asserting the effect is genuinely present in the rendered image (not
merely accepted by the schema): the axis tilt rotates the glow's own axis,
the hemispheric ratio makes the limb falloff length differ between the two
halves the mirror maps onto each other, the sharpness gradient makes it vary
with azimuth, the hemispheric amplitude scales one half's brightness, the
interior ramp tilts the disc's brightness along the axis, and a cloud blob
lands where it was placed.

The two hemispheric keys blend across the equator rather than switching
there, so their tests also assert that the rendered brightness changes no
faster across the equator than beside it, on an unrolled body and across a
rolled body's own equator, that the blend is only partial at half the
documented band edge, and that the full stated value holds poleward of the
band.

The gating contract has its own tests: a block naming none of the structure
keys leaves the spec's structure at None, which is what keeps the base haze
on its original scalar arithmetic and inside the cold-render budget.
"""

import math
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.sim.forward.atmosphere import atmosphere_spec_from_params
from spindoctor.sim.forward.body import render_single_body
from spindoctor.sim.forward.haze_structure import (
    HAZE_STRUCTURE_KEYS,
    HEMISPHERE_BLEND_HALF_WIDTH_DEG,
    SECTOR_REFERENCE_HALF_ANGLE_DEG,
    HazeStructure,
    haze_structure_from_params,
    hemisphere_weight,
    latitude_sine,
    scale_height_field,
)
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_checks_body import MIN_SECTOR_SHARPNESS_GRADIENT
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.types import NDArrayFloatType

_SIZE = 220
_CENTER = 110.0
_RADIUS = 55.0
# Sun toward +u in the image plane (illumination_angle 90 deg puts the light
# to the right), so the mirror plane runs along u and maps +v onto -v -- the
# hemispheres the ns_* keys split on.
_ILLUMINATION_DEG = 90.0
_BASE_ATMOSPHERE: dict[str, Any] = {'scale_height_px': 7.0, 'tau_ref': 3.0, 'g': 0.4}
# Image-plane distance from the equator, in pixels, beyond which the
# hemispheric keys apply at their full stated value on the disc.
_BLEND_EDGE_PX = math.sin(math.radians(HEMISPHERE_BLEND_HALF_WIDTH_DEG)) * _RADIUS
# The blend half-width the simulator guide documents, stated here rather than
# read from the module so a change to the constant fails the test that pins it.
_DOCUMENTED_BLEND_HALF_WIDTH_DEG = 10.0
# A rolled body with the sun along its equator, the geometry the Titan truth
# campaign uses (rotation_z = illumination_angle - 90).
_ROLL_DEG = 30.0
_ROLLED_ILLUMINATION_DEG = _ROLL_DEG + 90.0


def _render(
    atmosphere: dict[str, Any],
    *,
    phase_deg: float = 35.0,
    illumination_deg: float = _ILLUMINATION_DEG,
    rotation_z_deg: float = 0.0,
) -> NDArrayFloatType:
    """Render a centered hazy sphere and composite its halo over black.

    Parameters:
        atmosphere: The body's ``atmosphere`` block.
        phase_deg: Phase angle in degrees.
        illumination_deg: The body's ``illumination_angle`` in degrees.
        rotation_z_deg: The body's in-plane roll, ``rotation_z``, in degrees.

    Returns:
        The rendered signal image, halo included.
    """
    img: NDArrayFloatType = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    body_params: dict[str, Any] = {
        'name': 'HAZE',
        'center_v': _CENTER,
        'center_u': _CENTER,
        'axis1': 2.0 * _RADIUS,
        'axis2': 2.0 * _RADIUS,
        'axis3': 2.0 * _RADIUS,
        'illumination_angle': illumination_deg,
        'rotation_z': rotation_z_deg,
        'phase_angle': phase_deg,
        'anti_aliasing': 1.0,
        'atmosphere': atmosphere,
    }
    _mask, body_info = render_single_body(
        img,
        body_params,
        0.0,
        offset_u=0.0,
        ref_center_v=_SIZE / 2.0,
        ref_center_u=_SIZE / 2.0,
    )
    halo = body_info.get('halo')
    if halo is not None:
        covered = (halo.emission > 0.0) | (halo.transmission < 1.0)
        img[covered] = halo.emission[covered] + halo.transmission[covered] * img[covered]
    return img


def _with(**structure: Any) -> dict[str, Any]:
    """The base atmosphere block plus the given structure keys."""
    return {**_BASE_ATMOSPHERE, **structure}


def _offsets() -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Per-pixel ``(v, u)`` offsets from the body center."""
    v_idx, u_idx = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    return cast(NDArrayFloatType, v_idx + 0.5 - _CENTER), cast(
        NDArrayFloatType, u_idx + 0.5 - _CENTER
    )


def _glow_axis_deg(img: NDArrayFloatType) -> float:
    """The brightness-weighted direction of the above-limb glow, in degrees.

    Measured as the angle of the emission-weighted mean offset vector over
    an annulus outside the silhouette, expressed the way the scene's
    ``illumination_angle`` is (0 = toward the top of the image, 90 = toward
    the right), so it is directly comparable to the commanded angle.

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.

    Returns:
        The glow direction in degrees.
    """
    v_rel, u_rel = _offsets()
    rho = np.hypot(v_rel, u_rel)
    band = (rho > _RADIUS + 2.0) & (rho < _RADIUS + 25.0)
    weight = img[band]
    mean_v = float(np.sum(weight * v_rel[band]) / np.sum(weight))
    mean_u = float(np.sum(weight * u_rel[band]) / np.sum(weight))
    return math.degrees(math.atan2(mean_u, -mean_v))


def _falloff_length(img: NDArrayFloatType, direction_deg: float) -> float:
    """Fit the limb falloff length along one radial cut of the halo.

    In the optically thin part of the ramp the emergent intensity tracks the
    tangent optical depth, so ``log(I)`` is linear in altitude with slope
    ``-1 / scale_height``.

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.
        direction_deg: Radial direction in the scene's illumination-angle
            convention (0 = toward the top, 90 = toward the right).

    Returns:
        The recovered falloff length in pixels.
    """
    angle = math.radians(direction_deg)
    dir_v, dir_u = -math.cos(angle), math.sin(angle)
    altitude = np.arange(3.0, 26.0, 0.5)
    vs = _CENTER + (_RADIUS + altitude) * dir_v
    us = _CENTER + (_RADIUS + altitude) * dir_u
    samples = img[np.rint(vs).astype(int), np.rint(us).astype(int)]
    thin = samples > 1e-6
    slope = np.polyfit(altitude[thin], np.log(samples[thin]), 1)[0]
    return float(-1.0 / slope)


def _disc_mean(img: NDArrayFloatType, *, southern: bool) -> float:
    """Mean brightness over one hemisphere of the anti-sunward disc interior.

    Restricted well inside the limb so the anti-aliased rim and the halo
    cannot contribute, poleward of the equatorial blend band so every
    sampled pixel carries the hemispheric terms at their full value, and to
    the anti-sunward side so every sampled pixel has headroom
    below the top of the [0, 1] signal plane -- a brightening measured where
    the render already saturates would report the clip, not the scaling.

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.
        southern: True for the positive-v half, False for the negative-v one.

    Returns:
        The mean brightness of that half.
    """
    v_rel, u_rel = _offsets()
    rho = np.hypot(v_rel, u_rel)
    half = v_rel > _BLEND_EDGE_PX + 1.0 if southern else v_rel < -_BLEND_EDGE_PX - 1.0
    region = half & (rho < 0.8 * _RADIUS) & (u_rel < -0.3 * _RADIUS)
    return float(np.mean(img[region]))


# ---------------------------------------------------------------------------
# Gating: no structure keys means no structure, and no per-pixel cost.
# ---------------------------------------------------------------------------


def test_plain_atmosphere_block_carries_no_structure() -> None:
    """A block naming no structure key leaves the spec's structure at None."""
    spec = atmosphere_spec_from_params({'atmosphere': dict(_BASE_ATMOSPHERE)}, oversample=1)
    assert spec is not None
    assert spec.structure is None


@pytest.mark.parametrize('key', sorted(HAZE_STRUCTURE_KEYS))
def test_any_structure_key_builds_a_structure(key: str) -> None:
    """Naming any single structure key switches the spec onto the structured path."""
    sample: dict[str, Any] = {
        'interior_ramp_amplitude': 0.1,
        'ns_asymmetry_amplitude': 0.1,
        'ns_falloff_ratio': 1.5,
        'axis_tilt_deg': 5.0,
        'sector_sharpness_gradient': 0.3,
        'cloud_blobs': [{'center_vu': [0.0, 0.0], 'sigma_px': 3.0, 'amplitude': 0.2}],
    }
    spec = atmosphere_spec_from_params({'atmosphere': _with(**{key: sample[key]})}, oversample=1)
    assert spec is not None
    assert spec.structure is not None


def test_structure_scales_cloud_pixel_lengths_by_oversample() -> None:
    """Cloud positions and widths are grid pixels, so oversampling scales them."""
    structure = haze_structure_from_params(
        _with(cloud_blobs=[{'center_vu': [4.0, -2.0], 'sigma_px': 3.0, 'amplitude': 0.2}]),
        oversample=4,
    )
    assert structure is not None
    blob = structure.cloud_blobs[0]
    assert blob.center_v == pytest.approx(16.0)
    assert blob.center_u == pytest.approx(-8.0)
    assert blob.sigma_px == pytest.approx(12.0)


def test_structure_keeps_dimensionless_fields_unscaled() -> None:
    """Amplitudes, ratios, and angles do not move with the render grid."""
    structure = haze_structure_from_params(
        _with(ns_falloff_ratio=1.6, axis_tilt_deg=8.0, ns_asymmetry_amplitude=0.25),
        oversample=4,
    )
    assert structure is not None
    assert structure.ns_falloff_ratio == pytest.approx(1.6)
    assert structure.axis_tilt_deg == pytest.approx(8.0)
    assert structure.ns_asymmetry_amplitude == pytest.approx(0.25)


@pytest.mark.parametrize('ns_falloff_ratio', [0.6, 1.0, 1.8])
@pytest.mark.parametrize('sector_sharpness_gradient', [-0.3, 0.0, 0.5, 1.0])
def test_falloff_factor_bounds_contain_the_field(
    ns_falloff_ratio: float, sector_sharpness_gradient: float
) -> None:
    """The declared falloff bounds bracket the field they claim to bound.

    The haze band's outer reach and its on-disc inner edge are sized from
    these two bounds, so a bound that under-states the field would clip the
    glow at an artificial edge.  The sampled azimuths run to ``pi`` -- the
    anti-sunward point, where both extremes actually live -- because a
    sweep stopping at the reference sector edge would miss them.
    """
    nominal = 7.0
    structure = HazeStructure(
        ns_falloff_ratio=ns_falloff_ratio,
        sector_sharpness_gradient=sector_sharpness_gradient,
    )
    angles = np.linspace(0.0, math.pi, 37)
    u_ctr = np.cos(angles)[None, :]
    # Both hemispheres of the body-frame split, at every sampled azimuth.
    sin_latitude = np.concatenate([np.sin(angles), -np.sin(angles)])[None, :]
    field = scale_height_field(
        structure,
        nominal,
        sin_latitude=sin_latitude,
        v_ctr=np.concatenate([np.sin(angles), -np.sin(angles)])[None, :],
        u_ctr=np.concatenate([u_ctr[0], u_ctr[0]])[None, :],
        illum_v=0.0,
        illum_u=1.0,
    )
    values = np.asarray(field, dtype=float)
    assert float(values.min()) >= nominal * structure.min_falloff_factor - 1.0e-9
    assert float(values.max()) <= nominal * structure.max_falloff_factor + 1.0e-9


# ---------------------------------------------------------------------------
# One render-level test per key: the effect is measurably present.
# ---------------------------------------------------------------------------


def test_axis_tilt_rotates_the_glow_axis() -> None:
    """The haze glow's own axis follows the tilt, not the body's sun direction."""
    tilt_deg = 25.0
    plain = _glow_axis_deg(_render(dict(_BASE_ATMOSPHERE)))
    tilted = _glow_axis_deg(_render(_with(axis_tilt_deg=tilt_deg)))
    assert plain == pytest.approx(_ILLUMINATION_DEG, abs=1.0)
    assert tilted - plain == pytest.approx(tilt_deg, abs=4.0)


def test_axis_tilt_leaves_the_body_shading_axis_alone() -> None:
    """Only the haze tilts: the disc's own terminator stays where it was.

    Measured on the anti-sunward hemisphere well inside the limb, where the
    body's shading dominates and the haze's grazing excess has not yet
    grown; a tilt that moved the disc shading would change it.
    """
    v_rel, u_rel = _offsets()
    rho = np.hypot(v_rel, u_rel)
    core = (rho < 0.45 * _RADIUS) & (u_rel < 0.0)
    plain = _render(dict(_BASE_ATMOSPHERE))
    tilted = _render(_with(axis_tilt_deg=25.0))
    assert float(np.mean(np.abs(tilted[core] - plain[core]))) < 0.02


def test_hemispheric_falloff_ratio_lengthens_one_limb_ramp() -> None:
    """The falloff length differs between the two halves the mirror maps.

    A pure brightness scaling would leave both lengths equal; this key
    changes the SHAPE on one side only, which is the non-affine difference a
    Pearson symmetry score cannot absorb.
    """
    img = _render(_with(ns_falloff_ratio=2.0))
    southern = _falloff_length(img, 180.0)
    northern = _falloff_length(img, 0.0)
    assert southern > 1.5 * northern


def test_plain_haze_falloff_is_hemispherically_symmetric() -> None:
    """Without the ratio the two halves share one falloff length."""
    img = _render(dict(_BASE_ATMOSPHERE))
    southern = _falloff_length(img, 180.0)
    northern = _falloff_length(img, 0.0)
    assert southern == pytest.approx(northern, rel=0.05)


def test_sector_sharpness_gradient_lengthens_the_ramp_with_azimuth() -> None:
    """The falloff length grows away from the sunward axis, by the stated amount.

    The gradient is quoted at :data:`SECTOR_REFERENCE_HALF_ANGLE_DEG` from
    the axis, so a cut at that azimuth should carry close to the full
    multiplier relative to a cut along the axis itself.
    """
    gradient = 1.0
    img = _render(_with(sector_sharpness_gradient=gradient))
    on_axis = _falloff_length(img, _ILLUMINATION_DEG)
    off_axis = _falloff_length(img, _ILLUMINATION_DEG - SECTOR_REFERENCE_HALF_ANGLE_DEG)
    assert off_axis > on_axis * (1.0 + 0.5 * gradient)


def test_sector_sharpness_gradient_stays_symmetric_about_the_axis() -> None:
    """The gradient depends on |azimuth|, so the two sector edges match.

    That symmetry is what makes this key a pure edge-localization probe
    rather than a second, disguised axis tilt.
    """
    img = _render(_with(sector_sharpness_gradient=1.0))
    lower = _falloff_length(img, _ILLUMINATION_DEG - SECTOR_REFERENCE_HALF_ANGLE_DEG)
    upper = _falloff_length(img, _ILLUMINATION_DEG + SECTOR_REFERENCE_HALF_ANGLE_DEG)
    assert lower == pytest.approx(upper, rel=0.1)


def test_hemispheric_amplitude_scales_one_half_of_the_disc() -> None:
    """One hemisphere brightens by the stated fraction; the other does not."""
    amplitude = 0.3
    plain = _render(dict(_BASE_ATMOSPHERE))
    scaled = _render(_with(ns_asymmetry_amplitude=amplitude))
    southern_ratio = _disc_mean(scaled, southern=True) / _disc_mean(plain, southern=True)
    northern_ratio = _disc_mean(scaled, southern=False) / _disc_mean(plain, southern=False)
    assert southern_ratio == pytest.approx(1.0 + amplitude, rel=0.02)
    assert northern_ratio == pytest.approx(1.0, rel=0.02)


def _equator_jump_ratio(img: NDArrayFloatType) -> float:
    """How abruptly the brightness changes across the equator.

    The equator falls between rows ``_CENTER - 0.5`` and ``_CENTER + 0.5``.
    Over every lit column of the six rows around it (disc and halo, less
    the anti-aliased rim), this is the mean row-to-row change between the
    two rows that straddle the equator, divided by the larger of the same
    change one row to either side.  A hemispheric difference applied as a
    switch puts all of its change into the straddling pair, so the ratio is
    large; one that varies smoothly with latitude changes about as fast in
    the neighboring pairs, so the ratio stays near one or below it.

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.

    Returns:
        The ratio of the straddling jump to the larger neighboring jump.
    """
    equator_row = int(_CENTER)
    rows = slice(equator_row - 3, equator_row + 3)
    v_rel, u_rel = _offsets()
    off_rim = np.abs(np.hypot(v_rel, u_rel) - _RADIUS) > 1.5
    columns = np.all((off_rim & (img > 1e-3))[rows], axis=0)
    jumps = np.abs(np.diff(img[rows][:, columns], axis=0)).mean(axis=1)
    return float(jumps[2] / max(float(jumps[1]), float(jumps[3]), 1e-12))


@pytest.mark.parametrize(
    ('structure', 'illumination_deg'),
    [
        pytest.param({'ns_asymmetry_amplitude': 0.3}, 90.0, id='ns_asymmetry_amplitude=0.3'),
        pytest.param({'ns_asymmetry_amplitude': -0.3}, 90.0, id='ns_asymmetry_amplitude=-0.3'),
        pytest.param({'ns_falloff_ratio': 2.0}, 90.0, id='ns_falloff_ratio=2.0'),
        pytest.param({'ns_falloff_ratio': 1.1}, 90.0, id='ns_falloff_ratio=1.1'),
        pytest.param({'ns_falloff_ratio': 0.6}, 90.0, id='ns_falloff_ratio=0.6'),
        # Lit from 30 degrees, so the ramp along the haze axis runs across the
        # rows the equator divides; lit from 90 it would run along them.
        pytest.param({'interior_ramp_amplitude': 0.2}, 30.0, id='interior_ramp_amplitude=0.2'),
        pytest.param({'axis_tilt_deg': 8.0}, 90.0, id='axis_tilt_deg=8.0'),
    ],
)
def test_structure_keys_draw_no_seam_at_the_equator(
    structure: dict[str, float], illumination_deg: float
) -> None:
    """The rendered haze stays continuous across the body's equator.

    A real haze does not change its brightness or its scale height in one
    step at the equator, so a key that does draws a seam along one image
    row.  The two hemispheric keys must blend across the equator, and the
    axial ramp and the axis tilt, which carry no equatorial boundary, must
    stay clean.
    """
    img = _render(_with(**structure), illumination_deg=illumination_deg)
    assert _equator_jump_ratio(img) < 2.0


# Body-axis profiles of a rolled body run this far either side of its equator,
# in pixels, in one-pixel bins.  The halo they sample lies 2 to 15 pixels
# above the limb, so the documented band edge there is at most
# sin(10 deg) * 70, about 12 pixels, from the equator; the plateau bins sit
# beyond it, where each hemisphere carries its full value.
_PROFILE_REACH_PX = 20
_PROFILE_PLATEAU_PX = (14, 20)
# Bins either side of the equator the line-plus-step fit uses.
_STEP_FIT_HALF_PX = 6


def _body_axis_profile(
    structured: NDArrayFloatType,
    plain: NDArrayFloatType,
    *,
    rotation_z_deg: float,
    sunward: bool,
) -> NDArrayFloatType:
    """Sample the structured-to-plain ratio along a body's own first axis.

    The body's equator is the line ``v_rot = 0`` of the body frame, which a
    roll turns away from the image rows.  This profile follows ``v_rot``
    across that line where it crosses the halo, on one side of the body:
    halo pixels 2 to 15 pixels above the limb whose body-frame second
    coordinate is positive (``sunward``, for a body lit along its equator)
    or negative, binned in one-pixel steps of ``v_rot``, the median ratio in
    each bin.

    Parameters:
        structured: The render with the structure key.
        plain: The render of the base haze in the same geometry.
        rotation_z_deg: The body's roll in degrees.
        sunward: Which crossing of the equator with the halo to sample.

    Returns:
        The median ratio per bin; bin ``i`` covers ``v_rot`` from
        ``i - _PROFILE_REACH_PX`` to one pixel more.
    """
    v_rel, u_rel = _offsets()
    roll = math.radians(rotation_z_deg)
    v_rot = v_rel * math.cos(roll) - u_rel * math.sin(roll)
    u_rot = v_rel * math.sin(roll) + u_rel * math.cos(roll)
    rho = np.hypot(v_rel, u_rel)
    side = u_rot > 0.0 if sunward else u_rot < 0.0
    sample = (rho > _RADIUS + 2.0) & (rho < _RADIUS + 15.0) & side & (plain > 1e-4)
    ratio = structured[sample] / plain[sample]
    bins = np.floor(v_rot[sample]).astype(int)
    return np.array(
        [
            float(np.median(ratio[bins == edge]))
            for edge in range(-_PROFILE_REACH_PX, _PROFILE_REACH_PX)
        ]
    )


def _equator_step(profile: NDArrayFloatType) -> float:
    """Fit a line plus a step at ``v_rot = 0`` to a profile; return the step.

    A switch at the equator puts the whole hemispheric change into the step.
    A blend is smooth there, so the line takes up the change and the step
    stays near zero.  Fitting over several bins keeps the result steady
    against the bin-to-bin scatter a rolled pixel grid gives the profile.

    Parameters:
        profile: A profile from :func:`_body_axis_profile`.

    Returns:
        The fitted step, in units of the profile.
    """
    centers = np.arange(-_STEP_FIT_HALF_PX, _STEP_FIT_HALF_PX) + 0.5
    values = profile[_PROFILE_REACH_PX - _STEP_FIT_HALF_PX : _PROFILE_REACH_PX + _STEP_FIT_HALF_PX]
    design = np.column_stack([np.ones_like(centers), centers, (centers > 0.0).astype(float)])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(coefficients[2])


def _band_swing(profile: NDArrayFloatType) -> float:
    """The change in a profile from one hemisphere's plateau to the other's.

    Parameters:
        profile: A profile from :func:`_body_axis_profile`.

    Returns:
        The mean over the positive plateau bins less the mean over the
        negative ones.
    """
    near, far = _PROFILE_PLATEAU_PX
    positive = profile[_PROFILE_REACH_PX + near : _PROFILE_REACH_PX + far]
    negative = profile[_PROFILE_REACH_PX - far : _PROFILE_REACH_PX - near]
    return float(np.mean(positive) - np.mean(negative))


_ROLLED_CASES = [
    pytest.param({'ns_asymmetry_amplitude': -0.3}, True, id='amplitude-sunward'),
    pytest.param({'ns_asymmetry_amplitude': -0.3}, False, id='amplitude-anti-sunward'),
    pytest.param({'ns_falloff_ratio': 2.0}, True, id='ratio-sunward'),
    pytest.param({'ns_falloff_ratio': 2.0}, False, id='ratio-anti-sunward'),
]


@pytest.mark.parametrize(('structure', 'sunward'), _ROLLED_CASES)
def test_rolled_body_draws_no_seam_across_its_own_equator(
    structure: dict[str, float], sunward: bool
) -> None:
    """A rolled body's halo changes smoothly across the body's own equator.

    With the body rolled 30 degrees and lit along its equator, the equator
    runs diagonally across the image, so a row-based check cannot see it.
    Along the body's own axis, a switch would put the whole hemispheric
    change into one step at the equator; the blend leaves less than a
    quarter of it there.
    """
    rolled = {'illumination_deg': _ROLLED_ILLUMINATION_DEG, 'rotation_z_deg': _ROLL_DEG}
    plain = _render(dict(_BASE_ATMOSPHERE), **rolled)
    structured = _render(_with(**structure), **rolled)
    profile = _body_axis_profile(structured, plain, rotation_z_deg=_ROLL_DEG, sunward=sunward)
    assert abs(_equator_step(profile)) < 0.25 * abs(_band_swing(profile))


@pytest.mark.parametrize(('structure', 'sunward'), _ROLLED_CASES)
def test_rolled_body_splits_its_hemispheres_on_its_own_axis(
    structure: dict[str, float], sunward: bool
) -> None:
    """Rolling the body and its sun together rolls the hemispheric split too.

    The hemispheres are those of the body frame, so the change across a
    rolled body's equator, measured along its own axis, matches the change
    across an unrolled body's equator in the same lighting.  Keying the
    blend on image rows instead would put the rolled body's equator crossing
    of the halo deep inside one hemisphere, with nothing to change across.
    """
    rolled = {'illumination_deg': _ROLLED_ILLUMINATION_DEG, 'rotation_z_deg': _ROLL_DEG}
    rolled_swing = _band_swing(
        _body_axis_profile(
            _render(_with(**structure), **rolled),
            _render(dict(_BASE_ATMOSPHERE), **rolled),
            rotation_z_deg=_ROLL_DEG,
            sunward=sunward,
        )
    )
    unrolled_swing = _band_swing(
        _body_axis_profile(
            _render(_with(**structure)),
            _render(dict(_BASE_ATMOSPHERE)),
            rotation_z_deg=0.0,
            sunward=sunward,
        )
    )
    assert rolled_swing == pytest.approx(unrolled_swing, rel=0.1)


@pytest.mark.parametrize('hemisphere', [1.0, -1.0], ids=['positive', 'negative'])
def test_hemispheric_scaling_is_partial_at_half_the_documented_band_edge(
    hemisphere: float,
) -> None:
    """Halfway to the documented band edge the scaling is only part applied.

    The simulator guide documents a 10 degree half-width and a cubic
    smoothstep weight.  On the disc row nearest half that latitude the
    rendered brightness ratio gives the fraction of the full scaling the row
    carries, and it must match that smoothstep: a narrower band would carry
    nearly all of it there, and a wider one much less.
    """
    amplitude = -0.3
    plain = _render(dict(_BASE_ATMOSPHERE))
    scaled = _render(_with(ns_asymmetry_amplitude=amplitude))
    target = hemisphere * _RADIUS * math.sin(math.radians(_DOCUMENTED_BLEND_HALF_WIDTH_DEG / 2.0))
    v_rel, u_rel = _offsets()
    row_v = float(v_rel[np.argmin(np.abs(v_rel[:, 0] - target)), 0])
    on_row = (v_rel == row_v) & (np.hypot(v_rel, u_rel) < _RADIUS - 2.0) & (plain > 0.02)
    fraction = float(np.median((scaled[on_row] / plain[on_row] - 1.0) / amplitude))
    edge = math.sin(math.radians(_DOCUMENTED_BLEND_HALF_WIDTH_DEG))
    step = min(max((row_v / _RADIUS + edge) / (2.0 * edge), 0.0), 1.0)
    expected = step * step * (3.0 - 2.0 * step)
    assert fraction == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize('key', ['ns_asymmetry_amplitude', 'ns_falloff_ratio'])
def test_hemispheric_keys_at_their_identity_value_render_the_plain_haze(key: str) -> None:
    """An amplitude of zero or a ratio of one changes nothing anywhere."""
    identity = {'ns_asymmetry_amplitude': 0.0, 'ns_falloff_ratio': 1.0}[key]
    plain = _render(dict(_BASE_ATMOSPHERE))
    structured = _render(_with(**{key: identity}))
    np.testing.assert_array_equal(structured, plain)


def test_hemisphere_weight_blends_across_the_band_and_is_full_beyond_it() -> None:
    """The weight is 0.5 on the equator and exactly 0 or 1 poleward of the band."""
    edge = math.sin(math.radians(HEMISPHERE_BLEND_HALF_WIDTH_DEG))
    sin_latitude = np.array([-1.0, -edge, 0.0, edge, 1.0])
    np.testing.assert_array_equal(hemisphere_weight(sin_latitude), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_hemisphere_weight_rises_smoothly_and_point_symmetrically() -> None:
    """Inside the band the weight rises monotonically and w(-s) = 1 - w(s).

    The largest step between closely spaced samples bounds the weight's
    slope, so a weight that switched anywhere inside the band would fail it.
    """
    edge = math.sin(math.radians(HEMISPHERE_BLEND_HALF_WIDTH_DEG))
    sin_latitude = np.linspace(-edge, edge, 2001)
    weight = hemisphere_weight(sin_latitude)
    steps = np.diff(weight)
    assert float(steps.min()) >= 0.0
    # A cubic smoothstep's slope peaks at 1.5 times the mean slope.
    assert float(steps.max()) <= 1.5 / 2000.0 + 1e-12
    np.testing.assert_allclose(hemisphere_weight(-sin_latitude), 1.0 - weight, atol=1e-12)


def test_latitude_sine_is_continuous_across_the_limb() -> None:
    """The disc and halo forms of the latitude coordinate meet on the limb.

    Inside the limb it is ``v_rot / semi_a``; above it, the latitude of the
    limb point beneath the line of sight, which stays fixed along a ray from
    the center.
    """
    semi_a = 50.0
    v_rot = np.array([30.0, 30.0, 45.0])
    e2 = np.array([0.5, 1.0, (45.0 / 30.0) ** 2])
    np.testing.assert_allclose(latitude_sine(v_rot, e2, semi_a=semi_a), [0.6, 0.6, 0.6])


def test_interior_ramp_tilts_the_disc_brightness_along_the_axis() -> None:
    """The ramp brightens the sunward interior and dims the anti-sunward one."""
    amplitude = 0.2
    v_rel, u_rel = _offsets()
    rho = np.hypot(v_rel, u_rel)
    inner = rho < 0.6 * _RADIUS
    sunward = inner & (u_rel > 0.2 * _RADIUS)
    antisunward = inner & (u_rel < -0.2 * _RADIUS)
    plain = _render(dict(_BASE_ATMOSPHERE))
    ramped = _render(_with(interior_ramp_amplitude=amplitude))
    assert float(np.mean(ramped[sunward] - plain[sunward])) > 0.02
    assert float(np.mean(ramped[antisunward] - plain[antisunward])) < -0.02


def test_cloud_blob_brightens_the_disc_where_it_was_placed() -> None:
    """A blob adds its amplitude at its own center and nothing far away.

    Placed on the anti-sunward interior, where the disc is dim enough that
    the added amplitude has headroom inside the [0, 1] signal plane.
    """
    center_v, center_u = -18.0, -28.0
    blob = {'center_vu': [center_v, center_u], 'sigma_px': 5.0, 'amplitude': 0.25}
    plain = _render(dict(_BASE_ATMOSPHERE))
    clouded = _render(_with(cloud_blobs=[blob]))
    delta = clouded - plain
    peak_v = round(_CENTER + center_v)
    peak_u = round(_CENTER + center_u)
    assert float(delta[peak_v, peak_u]) == pytest.approx(0.25, abs=0.02)
    assert float(np.abs(delta[int(_CENTER + 40.0), int(_CENTER - 40.0)])) < 1e-9


def test_cloud_blob_stays_inside_the_silhouette() -> None:
    """A blob at the limb is clipped to the disc, never painted onto sky.

    A cloud is on the body; letting one leak past the silhouette would add
    brightness the body mask and depth truth know nothing about.
    """
    blob = {'center_vu': [0.0, _RADIUS], 'sigma_px': 6.0, 'amplitude': 0.4}
    plain = _render(dict(_BASE_ATMOSPHERE))
    clouded = _render(_with(cloud_blobs=[blob]))
    v_rel, u_rel = _offsets()
    outside = np.hypot(v_rel, u_rel) > _RADIUS + 2.0
    assert float(np.max(np.abs((clouded - plain)[outside]))) < 1e-9


# ---------------------------------------------------------------------------
# Schema validation of the structure keys.
# ---------------------------------------------------------------------------


def _scene_with_atmosphere(atmosphere: dict[str, Any]) -> dict[str, Any]:
    """A minimal one-body scene carrying the given atmosphere block."""
    return {
        'schema_version': 2,
        'scene_name': 'probe',
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'bodies': [
            {
                'name': 'HAZE',
                'center_v': 32.0,
                'center_u': 32.0,
                'axis1': 30.0,
                'axis2': 30.0,
                'axis3': 30.0,
                'atmosphere': atmosphere,
            }
        ],
    }


def test_validator_accepts_every_structure_key() -> None:
    """A block naming all structure keys validates."""
    scene = _scene_with_atmosphere(
        _with(
            interior_ramp_amplitude=-0.1,
            ns_asymmetry_amplitude=0.2,
            ns_falloff_ratio=1.4,
            axis_tilt_deg=-6.0,
            sector_sharpness_gradient=0.5,
            cloud_blobs=[{'center_vu': [1.0, 2.0], 'sigma_px': 2.0, 'amplitude': 0.1}],
        )
    )
    assert validate_sim_params(scene, source='probe')['bodies'][0]['atmosphere']


def test_validator_rejects_a_non_positive_falloff_ratio() -> None:
    """The ratio multiplies a length, so zero or negative fails."""
    scene = _scene_with_atmosphere(_with(ns_falloff_ratio=0.0))
    with pytest.raises(SimSceneValidationError, match='ns_falloff_ratio'):
        validate_sim_params(scene, source='probe')


def test_validator_rejects_a_sharpness_gradient_at_the_bound() -> None:
    """At the bound the falloff length vanishes anti-sunward, so it is rejected."""
    scene = _scene_with_atmosphere(_with(sector_sharpness_gradient=MIN_SECTOR_SHARPNESS_GRADIENT))
    with pytest.raises(SimSceneValidationError, match='sector_sharpness_gradient'):
        validate_sim_params(scene, source='probe')


def test_validator_rejects_an_unknown_cloud_blob_key() -> None:
    """A typo inside a cloud entry fails rather than rendering nothing."""
    # The axis order of `center_vu` written the wrong way round is what is
    # under test; the scene layer names a pixel pair `vu` and an oops uv `uv`,
    # so this is the typo an author actually makes.
    scene = _scene_with_atmosphere(
        _with(cloud_blobs=[{'center_uv': [0.0, 0.0], 'sigma_px': 2.0, 'amplitude': 0.1}])
    )
    with pytest.raises(SimSceneValidationError, match='unknown keys'):
        validate_sim_params(scene, source='probe')


def test_validator_requires_a_cloud_blob_amplitude() -> None:
    """A cloud with no amplitude has no rendered effect, so it fails."""
    scene = _scene_with_atmosphere(_with(cloud_blobs=[{'sigma_px': 2.0}]))
    with pytest.raises(SimSceneValidationError, match='amplitude is required'):
        validate_sim_params(scene, source='probe')


def test_validator_rejects_a_malformed_cloud_blob_center() -> None:
    """A cloud center must be a two-element offset pair."""
    scene = _scene_with_atmosphere(
        _with(cloud_blobs=[{'center_vu': [1.0], 'sigma_px': 2.0, 'amplitude': 0.1}])
    )
    with pytest.raises(SimSceneValidationError, match='center_vu'):
        validate_sim_params(scene, source='probe')
