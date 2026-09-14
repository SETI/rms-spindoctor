"""Acceptance and unit tests for the haze-limb atmosphere layer.

The acceptance contract (plan Section 15.10-G): the rendered tangent
optical-depth profile falls exponentially with the commanded scale height
(asserted on the optically thin portion, above the tau ~ 1 saturation
altitude), and the terminator brightens past 90 deg incidence instead of
cutting off.  The unit tests cover the Henyey-Greenstein phase dependence
(a forward-scattering limb brightens at high phase), the detached haze
shell, and the phase-dependent apparent limb radius (the Titan
altitude-versus-phase substrate), plus the module's pure helpers.  The
compositing tests pin the halo's transmission-screen behavior: the body
mask and depth truth see only the solid silhouette, a star behind the halo
attenuates by exp(-tau) instead of vanishing, a star behind the solid disc
vanishes, the ring system interleaves with the halo by depth, and a halo
that overlaps another body, another halo, or the ring system demands
explicit range_km on both participants.
"""

import math
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.sim.ellipsoid_geometry import illumination_vector
from spindoctor.sim.forward.atmosphere import (
    AtmosphereSpec,
    apply_atmosphere,
    atmosphere_spec_from_params,
    hg_phase_factor,
    tangent_optical_depth,
)
from spindoctor.sim.forward.body import render_single_body
from spindoctor.sim.forward.scene_radiance import compose_scene_radiance
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.types import NDArrayFloatType

_SIZE = 256
_CENTER = 128.0
_RADIUS = 60.0


def _render_haze_body(
    atmosphere: dict[str, float] | None,
    *,
    phase_deg: float = 30.0,
    illumination_deg: float = 90.0,
) -> NDArrayFloatType:
    """Render a centered spherical body, optionally with a haze layer.

    The solid disc paints opaquely and the translucent halo screen (returned
    on the body info) is composited over the empty background, mirroring the
    radiance stage.

    Parameters:
        atmosphere: The ``atmosphere`` block, or None for a hard-limbed body.
        phase_deg: Phase angle in degrees.
        illumination_deg: In-plane light direction in degrees.

    Returns:
        The rendered signal image.
    """
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    body_params: dict[str, object] = {
        'name': 'HAZE',
        'center_v': _CENTER,
        'center_u': _CENTER,
        'axis1': 2.0 * _RADIUS,
        'axis2': 2.0 * _RADIUS,
        'axis3': 2.0 * _RADIUS,
        'illumination_angle': illumination_deg,
        'phase_angle': phase_deg,
        'anti_aliasing': 1.0,
    }
    if atmosphere is not None:
        body_params['atmosphere'] = atmosphere
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


def _sunward_radial_profile(img: NDArrayFloatType) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """The intensity along the +u radial from the center (the sunward limb).

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.

    Returns:
        ``(altitude_px, intensity)``: tangent altitude above the reference
        radius and intensity along the sunward radial cut.
    """
    row = img[int(_CENTER), int(_CENTER) :]
    radius = np.arange(row.shape[0], dtype=np.float64) + 0.5
    return radius - _RADIUS, row


def _incidence_deg_map() -> NDArrayFloatType:
    """The surface incidence angle in degrees over the disc (phase 90, +u sun)."""
    v_ctr, u_ctr = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    v_rel = v_ctr + 0.5 - _CENTER
    u_rel = u_ctr + 0.5 - _CENTER
    rho2 = v_rel**2 + u_rel**2
    z = np.sqrt(np.maximum(_RADIUS**2 - rho2, 0.0))
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=math.radians(90.0), phase_angle=math.radians(90.0)
    )
    mu0 = (v_rel * illum_v + u_rel * illum_u + z * illum_z) / _RADIUS
    return cast(NDArrayFloatType, np.degrees(np.arccos(np.clip(mu0, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Acceptance (a): tangent optical depth falls exponentially with scale height.
# ---------------------------------------------------------------------------


def _recovered_scale_height(scale_height_px: float) -> float:
    """Fit the scale height from the optically thin part of the limb ramp.

    In the thin regime the emergent intensity is proportional to the tangent
    optical depth, so ``log(I)`` is linear in altitude with slope
    ``-1 / scale_height``.

    Parameters:
        scale_height_px: The commanded scale height.

    Returns:
        The scale height recovered from the rendered radial profile.
    """
    atmosphere = {'scale_height_px': scale_height_px, 'tau_ref': 1.5, 'g': 0.6}
    altitude, intensity = _sunward_radial_profile(_render_haze_body(atmosphere))
    # Optically thin band: two to four-and-a-half scale heights above the
    # reference radius (tau ~ 0.2 down to ~0.02, well under the tau ~ 1
    # saturation), where intensity tracks tau exponentially.
    thin = (
        (altitude >= 2.0 * scale_height_px)
        & (altitude <= 4.5 * scale_height_px)
        & (intensity > 1e-5)
    )
    slope = np.polyfit(altitude[thin], np.log(intensity[thin]), 1)[0]
    return float(-1.0 / slope)


def test_tangent_optical_depth_recovers_small_scale_height() -> None:
    """A 6 px scale height is recovered from the thin part of the ramp."""
    assert _recovered_scale_height(6.0) == pytest.approx(6.0, rel=0.15)


def test_tangent_optical_depth_recovers_large_scale_height() -> None:
    """A 12 px scale height is recovered from the thin part of the ramp."""
    assert _recovered_scale_height(12.0) == pytest.approx(12.0, rel=0.15)


def test_tangent_optical_depth_tracks_commanded_scale_height() -> None:
    """Doubling the commanded scale height roughly doubles the recovered one."""
    assert _recovered_scale_height(12.0) > 1.5 * _recovered_scale_height(6.0)


# ---------------------------------------------------------------------------
# Acceptance (b): the terminator brightens past 90 deg incidence.
# ---------------------------------------------------------------------------


def test_terminator_brightens_past_ninety_degrees() -> None:
    """Past-terminator disc pixels are brighter with the haze than without."""
    atmosphere = {'scale_height_px': 8.0, 'tau_ref': 1.5, 'g': 0.6}
    without = _render_haze_body(None, phase_deg=90.0)
    with_haze = _render_haze_body(atmosphere, phase_deg=90.0)
    incidence = _incidence_deg_map()
    # The night side just past the terminator, excluding the deep night far
    # from it: incidence in (90, 130) degrees on the disc.
    past = (incidence > 90.0) & (incidence < 130.0)
    assert with_haze[past].mean() > without[past].mean()


def test_lit_disc_beyond_terminator_stays_at_floor_without_atmosphere() -> None:
    """Without a haze layer the past-terminator disc renders at the dark floor."""
    without = _render_haze_body(None, phase_deg=90.0)
    incidence = _incidence_deg_map()
    past = (incidence > 100.0) & (incidence < 130.0)
    assert without[past].max() < 0.02


# ---------------------------------------------------------------------------
# Henyey-Greenstein phase dependence: a forward-scattering limb at high phase.
# ---------------------------------------------------------------------------


def _limb_halo_mean(img: NDArrayFloatType) -> float:
    """Mean intensity of the annulus just outside the geometric limb."""
    v_ctr, u_ctr = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    rho = np.hypot(v_ctr + 0.5 - _CENTER, u_ctr + 0.5 - _CENTER)
    halo = (rho > _RADIUS) & (rho < _RADIUS + 18.0)
    return float(img[halo].mean())


def test_forward_scattering_brightens_the_high_phase_limb() -> None:
    """A forward-scattering haze (g > 0) brightens the limb halo at high phase."""
    base = {'scale_height_px': 8.0, 'tau_ref': 1.5}
    isotropic = _limb_halo_mean(_render_haze_body({**base, 'g': 0.0}, phase_deg=150.0))
    forward = _limb_halo_mean(_render_haze_body({**base, 'g': 0.6}, phase_deg=150.0))
    assert forward > isotropic


# ---------------------------------------------------------------------------
# The detached haze shell.
# ---------------------------------------------------------------------------


def test_detached_shell_adds_a_bump_above_the_surface() -> None:
    """A detached shell brightens the radial profile at its altitude."""
    base = {'scale_height_px': 6.0, 'tau_ref': 0.8, 'g': 0.3}
    without = _render_haze_body(base)
    with_shell = _render_haze_body({**base, 'detached_px': 25.0})
    altitude, intensity_without = _sunward_radial_profile(without)
    _altitude, intensity_with = _sunward_radial_profile(with_shell)
    near = (altitude > 20.0) & (altitude < 30.0)
    assert intensity_with[near].mean() > intensity_without[near].mean() + 0.02


# ---------------------------------------------------------------------------
# Phase-dependent apparent limb radius (the Titan altitude-vs-phase substrate).
# ---------------------------------------------------------------------------


def _half_light_radius(img: NDArrayFloatType, *, threshold: float = 0.05) -> float:
    """The outermost sunward radius whose intensity clears a fixed threshold.

    Parameters:
        img: A rendered image with the body centered at ``_CENTER``.
        threshold: The fixed absolute intensity level the apparent limb is
            measured at.

    Returns:
        The apparent limb radius in pixels.
    """
    altitude, intensity = _sunward_radial_profile(img)
    lit = np.nonzero(intensity > threshold)[0]
    return float(altitude[lit.max()] + _RADIUS) if lit.size else 0.0


def test_apparent_limb_radius_grows_with_phase() -> None:
    """The forward-scattering haze pushes the apparent limb out at high phase."""
    atmosphere = {'scale_height_px': 8.0, 'tau_ref': 1.5, 'g': 0.6}
    low_phase = _half_light_radius(_render_haze_body(atmosphere, phase_deg=30.0))
    high_phase = _half_light_radius(_render_haze_body(atmosphere, phase_deg=150.0))
    assert high_phase > low_phase


def test_apparent_limb_radius_exceeds_the_geometric_radius() -> None:
    """The soft ramp places the apparent limb outside the hard geometric limb."""
    atmosphere = {'scale_height_px': 8.0, 'tau_ref': 1.5, 'g': 0.6}
    assert _half_light_radius(_render_haze_body(atmosphere)) > _RADIUS


# ---------------------------------------------------------------------------
# The pure helpers.
# ---------------------------------------------------------------------------


def test_hg_phase_factor_is_unity_when_isotropic() -> None:
    """The Henyey-Greenstein factor is 1 at every phase when g = 0."""
    assert hg_phase_factor(0.0, math.radians(75.0)) == pytest.approx(1.0)


def test_hg_phase_factor_peaks_forward_at_high_phase() -> None:
    """Forward scattering peaks at high phase (small scattering angle)."""
    assert hg_phase_factor(0.6, math.radians(170.0)) > hg_phase_factor(0.6, math.radians(10.0))


def test_tangent_optical_depth_is_exponential() -> None:
    """The column falls by exp over one scale height."""
    spec = AtmosphereSpec(scale_height_px=5.0, tau_ref=2.0, ref_altitude_px=0.0)
    tau = tangent_optical_depth(np.array([0.0, 5.0]), spec)
    assert tau[1] / tau[0] == pytest.approx(math.exp(-1.0), rel=1e-6)


def test_tangent_optical_depth_hits_tau_ref_at_reference_altitude() -> None:
    """tau_ref is the tangent optical depth at the reference altitude."""
    spec = AtmosphereSpec(scale_height_px=5.0, tau_ref=2.0, ref_altitude_px=3.0)
    tau = tangent_optical_depth(np.array([3.0]), spec)
    assert tau[0] == pytest.approx(2.0, rel=1e-9)


def test_tangent_optical_depth_shell_adds_a_bump() -> None:
    """A detached shell raises the tangent optical depth at its altitude."""
    base = AtmosphereSpec(scale_height_px=5.0, tau_ref=1.0)
    shell = AtmosphereSpec(scale_height_px=5.0, tau_ref=1.0, detached_px=20.0)
    altitude = np.array([20.0])
    assert tangent_optical_depth(altitude, shell)[0] > tangent_optical_depth(altitude, base)[0]


def test_atmosphere_spec_absent_returns_none() -> None:
    """A body without an atmosphere block yields no spec."""
    assert atmosphere_spec_from_params({'name': 'X'}, oversample=1) is None


def test_atmosphere_spec_scales_pixel_lengths_by_oversample() -> None:
    """Pixel lengths scale with the oversampling factor; dimensionless ones do not."""
    block = {'scale_height_px': 4.0, 'tau_ref': 1.2, 'ref_altitude_px': 2.0, 'detached_px': 6.0}
    spec = atmosphere_spec_from_params({'atmosphere': block}, oversample=4)
    assert spec is not None
    assert spec.scale_height_px == pytest.approx(16.0)


def test_atmosphere_spec_keeps_tau_ref_dimensionless() -> None:
    """The optical depth is dimensionless and is not scaled by oversample."""
    block = {'scale_height_px': 4.0, 'tau_ref': 1.2}
    spec = atmosphere_spec_from_params({'atmosphere': block}, oversample=4)
    assert spec is not None
    assert spec.tau_ref == pytest.approx(1.2)


def test_apply_atmosphere_does_not_mutate_input() -> None:
    """The haze compositor returns fresh arrays, leaving the cache entry intact."""
    body = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    body[100:150, 100:150] = 0.5
    original = body.copy()
    spec = AtmosphereSpec(scale_height_px=8.0, tau_ref=1.5, g=0.6)
    apply_atmosphere(
        body,
        spec,
        center_v=_CENTER,
        center_u=_CENTER,
        semi_a=_RADIUS,
        semi_b=_RADIUS,
        semi_c=_RADIUS,
        rotation_z=0.0,
        rotation_tilt=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=math.radians(30.0),
    )
    assert np.array_equal(body, original)


def _centred_layers(spec: AtmosphereSpec) -> Any:
    """Evaluate the haze layers of a centered dark sphere of radius ``_RADIUS``."""
    body = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    return apply_atmosphere(
        body,
        spec,
        center_v=_CENTER,
        center_u=_CENTER,
        semi_a=_RADIUS,
        semi_b=_RADIUS,
        semi_c=_RADIUS,
        rotation_z=0.0,
        rotation_tilt=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=math.radians(30.0),
    )


def test_apply_atmosphere_adds_glow_above_the_limb() -> None:
    """The halo screen carries a soft glow just outside the geometric limb."""
    spec = AtmosphereSpec(scale_height_px=8.0, tau_ref=1.5, g=0.6)
    layers = _centred_layers(spec)
    # Just outside the sunward limb (+u from center): the glow lives on the
    # translucent halo, not the opaque disc.
    probe = (int(_CENTER), int(_CENTER + _RADIUS + 3))
    assert layers.halo.emission[probe] > 0.0
    assert layers.disc[probe] == 0.0


def test_halo_transmission_is_the_tangent_extinction() -> None:
    """The halo screen transmits exp(-tau) of the background at each altitude."""
    spec = AtmosphereSpec(scale_height_px=8.0, tau_ref=1.5, g=0.6)
    layers = _centred_layers(spec)
    probe_v = int(_CENTER)
    probe_u = int(_CENTER + _RADIUS + 6)
    altitude = math.hypot(probe_v + 0.5 - _CENTER, probe_u + 0.5 - _CENTER) - _RADIUS
    tau = float(tangent_optical_depth(np.array([altitude]), spec)[0])
    assert layers.halo.transmission[probe_v, probe_u] == pytest.approx(math.exp(-tau), rel=1e-9)


# ---------------------------------------------------------------------------
# Limb/disc consistency at a nonzero reference altitude.
# ---------------------------------------------------------------------------


def test_haze_layers_invariant_under_reference_altitude_shift() -> None:
    """Re-referencing tau_ref to another altitude leaves the whole layer alone.

    ``tau_ref`` at ``ref_altitude_px = 30`` with H = 5 describes the same
    atmosphere as ``tau_ref * exp(30 / 5)`` at the surface, so every layer
    (on-disc haze, halo glow, halo transmission) must be identical for the
    two specs.  The disc side derives from the surface tangent depth; a
    tau_vert built from the raw ``tau_ref`` would leave the two disc renders
    a factor exp(30 / 5) ~ 403 apart.
    """
    scale_height = 5.0
    ref_altitude = 30.0
    tau_ref = 0.005
    referenced = AtmosphereSpec(
        scale_height_px=scale_height, tau_ref=tau_ref, ref_altitude_px=ref_altitude
    )
    surface = AtmosphereSpec(
        scale_height_px=scale_height,
        tau_ref=tau_ref * math.exp(ref_altitude / scale_height),
        ref_altitude_px=0.0,
    )
    layers_referenced = _centred_layers(referenced)
    layers_surface = _centred_layers(surface)
    np.testing.assert_allclose(layers_referenced.disc, layers_surface.disc, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        layers_referenced.halo.emission, layers_surface.halo.emission, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        layers_referenced.halo.transmission, layers_surface.halo.transmission, rtol=1e-9
    )


def test_on_disc_haze_is_continuous_across_the_limb() -> None:
    """The disc side adjacent to the limb tracks the just-above-limb glow.

    One pixel inside and one pixel outside the limb probe the two opacity
    formulas where they meet; with the disc column derived from the surface
    tangent depth the rendered values sit within a small factor of each
    other at ``ref_altitude_px = 30`` (a raw-tau_ref disc column would sit
    ~400x below the glow).
    """
    spec = AtmosphereSpec(scale_height_px=5.0, tau_ref=0.005, ref_altitude_px=30.0)
    layers = _centred_layers(spec)
    row = int(_CENTER)
    inside = float(layers.disc[row, int(_CENTER + _RADIUS - 2)])
    outside = float(layers.halo.emission[row, int(_CENTER + _RADIUS + 1)])
    assert outside > 0.0
    assert inside > outside / 3.0
    assert inside < outside * 3.0


# ---------------------------------------------------------------------------
# Halo compositing: solid-silhouette truth, star extinction, ring interleave.
# ---------------------------------------------------------------------------

# A fully lit sphere of radius 15 at the frame center whose haze (H = 5,
# tau_ref = 2) glows over a halo out to ~38 px above the limb.
_C_SIZE = 96
_C_CENTER = 48.0
_C_RADIUS = 15.0
_C_SPEC = AtmosphereSpec(scale_height_px=5.0, tau_ref=2.0, g=0.6)


def _compose_body(*, atmosphere: bool = True, **extra: Any) -> dict[str, Any]:
    """The compositing scenes' centered atmospheric body entry."""
    body: dict[str, Any] = {
        'name': 'TITAN',
        'center_v': _C_CENTER,
        'center_u': _C_CENTER,
        'axis1': 2.0 * _C_RADIUS,
        'axis2': 2.0 * _C_RADIUS,
        'axis3': 2.0 * _C_RADIUS,
        'illumination_angle': 0.0,
        'phase_angle': 0.0,
    }
    if atmosphere:
        body['atmosphere'] = {
            'scale_height_px': _C_SPEC.scale_height_px,
            'tau_ref': _C_SPEC.tau_ref,
            'g': _C_SPEC.g,
        }
    body.update(extra)
    return body


def _compose_scene(**extra: Any) -> Any:
    """Run the radiance stage on a minimal noiseless scene; return the frame."""
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'size_v': _C_SIZE,
        'size_u': _C_SIZE,
        'random_seed': 3,
        'exposure_sec': 1.0,
    }
    params.update(extra)
    scene = validate_sim_params(params)
    frame = new_sim_frame(_C_SIZE, _C_SIZE)
    compose_scene_radiance(frame, params=scene, rng=np.random.default_rng(0))
    return frame


def _halo_transmission_at(pixel: tuple[int, int]) -> float:
    """The centered test body's tangent transmission at a pixel center."""
    altitude = math.hypot(pixel[0] + 0.5 - _C_CENTER, pixel[1] + 0.5 - _C_CENTER) - _C_RADIUS
    tau = float(tangent_optical_depth(np.array([altitude]), _C_SPEC)[0])
    return math.exp(-tau)


# One tangent altitude probed throughout: ~5.5 px above the limb, where the
# halo is bright and its transmission is ~0.51.
_HALO_PIXEL = (48, 68)
_DISC_PIXEL = (48, 55)


def test_body_mask_covers_only_the_solid_silhouette() -> None:
    """The returned body mask excludes the glowing halo outside the limb."""
    img = np.zeros((_C_SIZE, _C_SIZE), dtype=np.float64)
    mask, body_info = render_single_body(
        img,
        _compose_body(),
        0.0,
        offset_u=0.0,
        ref_center_v=_C_CENTER,
        ref_center_u=_C_CENTER,
    )
    assert bool(mask[_DISC_PIXEL])
    assert not bool(mask[_HALO_PIXEL])
    # The exclusion is meaningful: the halo does glow at the probed pixel.
    assert float(body_info['halo'].emission[_HALO_PIXEL]) > 0.0


def test_body_index_map_excludes_the_halo() -> None:
    """The z-order truth claims the solid disc, never the halo glow."""
    frame = _compose_scene(bodies=[_compose_body()])
    index_map = frame.truth['body_index_map']
    assert int(index_map[_DISC_PIXEL]) == 1
    assert int(index_map[_HALO_PIXEL]) == 0
    # The halo still renders at the probed pixel.
    assert float(frame.signal[_HALO_PIXEL]) > 0.0


def test_star_behind_the_halo_attenuates_by_the_tangent_extinction() -> None:
    """A star behind the halo dims by exp(-tau); it is not erased."""
    star = {'name': 'S', 'v': float(_HALO_PIXEL[0]), 'u': float(_HALO_PIXEL[1]), 'vmag': 8.0}
    with_halo = _compose_scene(stars=[star], bodies=[_compose_body()]).point_e
    without = _compose_scene(stars=[star]).point_e
    assert float(without[_HALO_PIXEL]) > 0.0
    assert float(with_halo[_HALO_PIXEL]) > 0.0
    ratio = float(with_halo[_HALO_PIXEL] / without[_HALO_PIXEL])
    assert ratio == pytest.approx(_halo_transmission_at(_HALO_PIXEL), rel=1e-9)


def test_star_behind_the_solid_disc_vanishes() -> None:
    """The opaque silhouette still extinguishes point sources entirely."""
    star = {'name': 'S', 'v': float(_DISC_PIXEL[0]), 'u': float(_DISC_PIXEL[1]), 'vmag': 8.0}
    with_body = _compose_scene(stars=[star], bodies=[_compose_body()]).point_e
    without = _compose_scene(stars=[star]).point_e
    assert float(without[_DISC_PIXEL]) > 0.0
    assert float(with_body[_DISC_PIXEL]) == 0.0


def test_halo_overlap_without_ranges_is_ambiguous() -> None:
    """A halo reaching another body's disc without ranges fails validation.

    Neither body carries a range_km; their solid discs are disjoint but the
    halo reaches the second body's silhouette, so which side of it the halo
    composites on would be a silent guess ordered by the positional default
    ranges.  The scene fails loudly instead, naming the halo.
    """
    haze_body = _compose_body(center_u=30.0)
    plain = _compose_body(
        atmosphere=False, name='PLAIN', center_u=78.0, axis1=16.0, axis2=16.0, axis3=16.0
    )
    expected = r"halo of body 'TITAN' overlaps body 'PLAIN'"
    with pytest.raises(SimSceneValidationError, match=expected):
        _compose_scene(bodies=[haze_body, plain])


def test_halo_overlap_orders_by_explicit_ranges() -> None:
    """With explicit ranges a halo composites by depth, in both directions.

    The same halo-over-disc geometry as the ambiguity test, with ranges: a
    nearer haze body screens the far plain disc through its glow, while a
    farther haze body's halo is hidden behind the nearer plain disc, so the
    two orderings render differently and each matches its closed form.
    """
    plain = _compose_body(
        atmosphere=False,
        name='PLAIN',
        center_u=78.0,
        axis1=16.0,
        axis2=16.0,
        axis3=16.0,
        range_km=2.0e6,
    )
    probe = (48, 72)
    plain_only = _compose_scene(bodies=[plain]).signal
    haze_near = _compose_body(center_u=30.0, range_km=1.0e6)
    haze_only = _compose_scene(bodies=[haze_near]).signal
    near_scene = _compose_scene(bodies=[haze_near, plain]).signal
    altitude = math.hypot(probe[0] + 0.5 - 48.0, probe[1] + 0.5 - 30.0) - _C_RADIUS
    tau = float(tangent_optical_depth(np.array([altitude]), _C_SPEC)[0])
    expected_near = haze_only[probe] + math.exp(-tau) * plain_only[probe]
    assert float(plain_only[probe]) > 0.0
    assert float(near_scene[probe]) == pytest.approx(float(expected_near), rel=1e-9)
    # Haze body farther than the plain disc: its halo hides behind the disc.
    haze_far = _compose_body(center_u=30.0, range_km=4.0e6)
    far_scene = _compose_scene(bodies=[haze_far, plain]).signal
    assert float(far_scene[probe]) == pytest.approx(float(plain_only[probe]), rel=1e-9)
    assert float(near_scene[probe]) != pytest.approx(float(far_scene[probe]), rel=1e-9)


def test_overlapping_halos_without_ranges_are_ambiguous() -> None:
    """Two halos that overlap only each other still demand explicit ranges.

    The discs are far enough apart that neither halo reaches the other's
    silhouette, so only the halo-versus-halo glow overlaps -- and the two
    screens' compositing order (each attenuates the other) would still be
    ordered by the positional defaults.
    """
    left = _compose_body(center_u=5.0)
    right = _compose_body(name='TITAN2', center_u=91.0)
    # The message names the bodies in render (far-to-near) order: the second
    # body's positional default range is larger, so it renders first.
    with pytest.raises(SimSceneValidationError, match="halos of bodies 'TITAN2' and 'TITAN'"):
        _compose_scene(bodies=[left, right])


def _compose_ring_system(range_km: float | None) -> dict[str, Any]:
    """A tau = 1 ringlet (radii 22-26) around the centered test body."""
    system: dict[str, Any] = {
        'geometry': {'opening_deg_obs': 30.0, 'opening_deg_sun': 30.0},
        'phase_deg': 0.0,
        'features': [
            {
                'name': 'BAND',
                'kind': 'ringlet',
                'tau': 1.0,
                'width': 4.0,
                'orbit': {'a': 22.0},
            }
        ],
    }
    if range_km is not None:
        system['range_km'] = range_km
        system['km_per_pixel'] = 1000.0
    return system


# Mid-band probe: ring radius ~24.5, tangent altitude ~9.5 px.
_RING_PIXEL = (48, 72)


def test_ring_behind_the_halo_shows_through_attenuated() -> None:
    """A ring behind the body glows through the halo scaled by exp(-tau)."""
    body = _compose_body(range_km=1.0e6)
    both = _compose_scene(bodies=[body], ring_system=_compose_ring_system(2.0e6)).signal
    ring_only = _compose_scene(ring_system=_compose_ring_system(2.0e6)).signal
    body_only = _compose_scene(bodies=[body]).signal
    expected = body_only[_RING_PIXEL] + (
        _halo_transmission_at(_RING_PIXEL) * ring_only[_RING_PIXEL]
    )
    assert float(ring_only[_RING_PIXEL]) > 0.0
    assert float(both[_RING_PIXEL]) == pytest.approx(float(expected), rel=1e-9)
    # Behind the solid disc the ring is hidden entirely.
    assert float(both[_DISC_PIXEL]) == pytest.approx(float(body_only[_DISC_PIXEL]), rel=1e-12)


def test_ring_in_front_screens_the_halo() -> None:
    """A ring nearer than the body screens the halo glow behind it."""
    body = _compose_body(range_km=1.0e6)
    both = _compose_scene(bodies=[body], ring_system=_compose_ring_system(5.0e5)).signal
    ring_only = _compose_scene(ring_system=_compose_ring_system(5.0e5)).signal
    body_only = _compose_scene(bodies=[body]).signal
    # Full-coverage mid-band pixel: the ring transmits exp(-tau/mu) = exp(-2).
    transmission = math.exp(-1.0 / math.sin(math.radians(30.0)))
    expected = ring_only[_RING_PIXEL] + transmission * body_only[_RING_PIXEL]
    assert float(both[_RING_PIXEL]) == pytest.approx(float(expected), rel=1e-9)


def test_ring_over_only_the_halo_still_needs_ranges() -> None:
    """A depth-less ring overlapping only the halo fails validation.

    A face-on ring annulus clears the solid disc entirely, so only the halo
    glow overlaps it -- and without ranges the ring would silently screen
    the halo (a depth-less ring applies last).  The scene fails loudly,
    naming the halo.
    """
    ring: dict[str, Any] = _compose_ring_system(None)
    ring['geometry'] = {'opening_deg_obs': 90.0, 'opening_deg_sun': 90.0}
    with pytest.raises(SimSceneValidationError, match="ring_system and the halo of body 'TITAN'"):
        _compose_scene(bodies=[_compose_body()], ring_system=ring)
