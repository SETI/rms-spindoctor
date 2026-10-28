"""Ring feature kinds and orbit perturbations: the normative closed forms.

Pins the m-mode radius form ``r(lam) = a - amp * cos(m * (lam - peri))``,
the satellite edge wave
``dr = amp * exp(-(lam - lam0)/damp) * sin(2*pi*(lam - lam0)*a/wavelength)``
with its load-bearing downstream clamp (the exponential grows without bound
upstream, so dr is evaluated only for longitudes downstream of lam0), the
ring-plane frame of ``peri`` / ``lam0`` (the sky node angle never enters the
orbit model), and the edge / ramp / wave radial tau profiles.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.ring_system import render_ring_system
from spindoctor.sim.ring_geometry import (
    RingEdgeWave,
    RingOrbit,
    RingOrbitMode,
    compute_edge_wave_dr,
    compute_orbit_radii,
    ring_orbit_from_mapping,
    ring_sky_from_plane,
)
from spindoctor.support.types import NDArrayFloatType


def _lam_grid(n: int = 720) -> NDArrayFloatType:
    return np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)


# ---------------------------------------------------------------------------
# m-modes
# ---------------------------------------------------------------------------


def test_pure_m_mode_matches_the_closed_form() -> None:
    """On a circular base orbit, r(lam) = a - amp*cos(m*(lam - peri)) exactly."""
    lam = _lam_grid()
    a, amp, m, peri = 40.0, 2.5, 3, 55.0
    orbit = RingOrbit(
        a=a, ae=0.0, long_peri=0.0, rate_peri=0.0, modes=(RingOrbitMode(m=m, amp=amp, peri=peri),)
    )
    r = compute_orbit_radii(lam, orbit, epoch=0.0, time=0.0)
    expected = a - amp * np.cos(m * (lam - math.radians(peri)))
    np.testing.assert_allclose(r, expected, rtol=1e-14)


def test_m_mode_perturbs_the_mode_1_ellipse() -> None:
    """m-modes add to the exact mode-1 conic, not replace it."""
    lam = _lam_grid()
    base = RingOrbit(a=40.0, ae=4.0, long_peri=20.0, rate_peri=0.0)
    perturbed = RingOrbit(
        a=40.0,
        ae=4.0,
        long_peri=20.0,
        rate_peri=0.0,
        modes=(RingOrbitMode(m=2, amp=1.5, peri=70.0),),
    )
    r_base = compute_orbit_radii(lam, base, epoch=0.0, time=0.0)
    r = compute_orbit_radii(lam, perturbed, epoch=0.0, time=0.0)
    expected = r_base - 1.5 * np.cos(2.0 * (lam - math.radians(70.0)))
    np.testing.assert_allclose(r, expected, rtol=1e-14)


def test_orbit_mapping_parses_modes_and_edge_wave() -> None:
    """The shared parser reads the idealized orbit block with its defaults."""
    orbit = ring_orbit_from_mapping(
        {
            'a_px': 40.0,
            'modes': [{'m': 2, 'amp_px': 1.5, 'peri_deg': 70.0}],
            'edge_wave': {
                'amp_px': 1.0,
                'wavelength_px': 8.0,
                'edge_wave_damping_deg': 30.0,
                'lam0_deg': 90.0,
            },
        }
    )
    assert orbit.a == 40.0
    assert orbit.ae == 0.0
    assert orbit.modes == (RingOrbitMode(m=2, amp=1.5, peri=70.0),)
    assert orbit.edge_wave == RingEdgeWave(
        amp=1.0, wavelength=8.0, damp=math.radians(30.0), lam0=90.0
    )


def test_scaled_orbit_scales_radial_quantities_only() -> None:
    """Oversampling scales radii and radial amplitudes; angles are untouched."""
    orbit = ring_orbit_from_mapping(
        {
            'a_px': 40.0,
            'ae_px': 2.0,
            'long_peri_deg': 15.0,
            'modes': [{'m': 2, 'amp_px': 1.5, 'peri_deg': 70.0}],
            'edge_wave': {
                'amp_px': 1.0,
                'wavelength_px': 8.0,
                'edge_wave_damping_deg': 30.0,
                'lam0_deg': 90.0,
            },
        }
    ).scaled(4.0)
    assert orbit.a == 160.0
    assert orbit.ae == 8.0
    assert orbit.long_peri == 15.0
    assert orbit.modes[0].amp == 6.0
    assert orbit.modes[0].peri == 70.0
    assert orbit.edge_wave is not None
    assert orbit.edge_wave.amp == 4.0
    assert orbit.edge_wave.wavelength == 32.0
    assert orbit.edge_wave.damp == math.radians(30.0)
    assert orbit.edge_wave.lam0 == 90.0


# ---------------------------------------------------------------------------
# Satellite edge waves
# ---------------------------------------------------------------------------


def test_edge_wave_matches_the_closed_form_downstream() -> None:
    """dr at a downstream longitude is the damped sinusoid as written."""
    a, amp, wavelength, damp, lam0 = 40.0, 1.2, 8.0, 0.5, 90.0
    wave = RingEdgeWave(amp=amp, wavelength=wavelength, damp=damp, lam0=lam0)
    dlam = 0.3
    lam = np.asarray([math.radians(lam0) + dlam])
    dr = compute_edge_wave_dr(lam, wave, a=a)
    expected = amp * math.exp(-dlam / damp) * math.sin(2.0 * math.pi * dlam * a / wavelength)
    assert dr[0] == pytest.approx(expected, rel=1e-12)


def test_edge_wave_is_zero_at_the_launch_longitude() -> None:
    """dr(lam0) = 0: the wave launches from the unperturbed edge."""
    wave = RingEdgeWave(amp=1.2, wavelength=8.0, damp=0.5, lam0=90.0)
    dr = compute_edge_wave_dr(np.asarray([math.radians(90.0)]), wave, a=40.0)
    assert dr[0] == 0.0


def test_edge_wave_upstream_clamp_is_load_bearing() -> None:
    """Immediately upstream of lam0 the wave is negligible, never divergent.

    The raw formula evaluated at lam - lam0 = -0.1 rad would carry
    exp(+0.1/damp) of GROWING amplitude; the modular downstream form instead
    wraps to dlam = 2*pi - 0.1 and carries exp(-(2*pi - 0.1)/damp), which
    the test pins exactly and bounds at a negligible fraction of amp.
    """
    a, amp, wavelength, damp, lam0 = 40.0, 1.2, 8.0, 0.5, 90.0
    wave = RingEdgeWave(amp=amp, wavelength=wavelength, damp=damp, lam0=lam0)
    lam = np.asarray([math.radians(lam0) - 0.1])
    dr = compute_edge_wave_dr(lam, wave, a=a)
    dlam = 2.0 * math.pi - 0.1
    expected = amp * math.exp(-dlam / damp) * math.sin(2.0 * math.pi * dlam * a / wavelength)
    assert dr[0] == pytest.approx(expected, rel=1e-9)
    assert abs(dr[0]) < 1e-4 * amp


def test_edge_wave_wrap_seam_residual_at_the_damp_cap() -> None:
    """At damp = 2.0 (the validator cap) the wrap-seam residual is exp(-pi)-small.

    The modular longitude wrap means the wave re-approaches its launch
    longitude from upstream carrying amp * exp(-2*pi/damp) of its launch
    amplitude.  At the validator's damp cap of 2.0 radians that residual
    envelope is amp * exp(-pi), about 4.3% of amp -- the largest wrap seam
    a valid scene can author.
    """
    a, amp, wavelength, damp, lam0_deg = 40.0, 1.0, 8.0, 2.0, 90.0
    wave = RingEdgeWave(amp=amp, wavelength=wavelength, damp=damp, lam0=lam0_deg)
    probe = 0.05
    lam = math.radians(lam0_deg) - np.linspace(1e-6, probe, 512)
    dr = compute_edge_wave_dr(lam, wave, a=a)
    # The residual envelope grows moving upstream; over the probed window it
    # is bounded by its value at the window's upstream end.
    seam_bound = amp * math.exp(-(2.0 * math.pi - probe) / damp)
    assert float(np.max(np.abs(dr))) <= seam_bound
    assert seam_bound == pytest.approx(amp * math.exp(-math.pi), rel=0.03)


def test_edge_wave_decays_downstream() -> None:
    """Crest amplitudes fall by exp(-d/damp) between successive probes."""
    a, amp, wavelength, damp = 40.0, 1.2, 8.0, 0.5
    wave = RingEdgeWave(amp=amp, wavelength=wavelength, damp=damp, lam0=0.0)
    # Probe at successive sine crests: 2*pi*dlam*a/wavelength = pi/2 + 2*pi*k.
    crest = (wavelength / a) * (0.25 + np.arange(4))
    dr = compute_edge_wave_dr(crest, wave, a=a)
    ratios = dr[1:] / dr[:-1]
    np.testing.assert_allclose(ratios, math.exp(-(wavelength / a) / damp), rtol=1e-9)


def test_edge_wave_applies_through_the_orbit_model() -> None:
    """compute_orbit_radii adds the wave to the (possibly perturbed) radius."""
    lam = _lam_grid()
    wave = RingEdgeWave(amp=1.2, wavelength=8.0, damp=0.5, lam0=90.0)
    orbit = RingOrbit(a=40.0, ae=0.0, long_peri=0.0, rate_peri=0.0, edge_wave=wave)
    r = compute_orbit_radii(lam, orbit, epoch=0.0, time=0.0)
    expected = 40.0 + compute_edge_wave_dr(lam, wave, a=40.0)
    np.testing.assert_allclose(r, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Rendered radial profiles (face-on grid: r is the sky-plane radius)
# ---------------------------------------------------------------------------

_SIZE = 96
# Pixel centers sit at integer + 0.5, so this center puts probe pixels at
# integral ring radii along the +u axis.
_CENTER = 48.5


def _system(features: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    system: dict[str, Any] = {
        'geometry': {
            'center_v': _CENTER,
            'center_u': _CENTER,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': features,
    }
    system.update(extra)
    return system


def _tau_map(features: list[dict[str, Any]], **extra: Any) -> NDArrayFloatType:
    """Render face-on and recover the composed tau map from the transmission."""
    maps = render_ring_system(
        (_SIZE, _SIZE),
        _system(features, **extra),
        center_v=_CENTER,
        center_u=_CENTER,
        node_deg=0.0,
    )
    # Face-on: mu = 1, so tau = -ln(T) exactly.
    return np.asarray(-np.log(maps.transmission))


def _probe(tau_map: NDArrayFloatType, radius: int) -> float:
    """tau at the pixel whose center sits ``radius`` px along +u from center."""
    return float(tau_map[48, 48 + radius])


def test_edge_kind_side_in_carries_tau_inside() -> None:
    """side 'in': tau for r <= r_edge, zero outside, one-pixel transition."""
    tau_map = _tau_map([{'kind': 'edge', 'tau': 1.5, 'side': 'in', 'orbit': {'a_px': 20.0}}])
    assert _probe(tau_map, 10) == pytest.approx(1.5, rel=1e-12)
    assert _probe(tau_map, 18) == pytest.approx(1.5, rel=1e-12)
    assert _probe(tau_map, 22) == 0.0
    # The pixel centered exactly on the edge carries half coverage.
    assert _probe(tau_map, 20) == pytest.approx(0.75, rel=1e-9)


def test_edge_kind_side_out_carries_tau_outside() -> None:
    """side 'out': tau for r >= r_edge, zero inside."""
    tau_map = _tau_map([{'kind': 'edge', 'tau': 0.8, 'side': 'out', 'orbit': {'a_px': 20.0}}])
    assert _probe(tau_map, 30) == pytest.approx(0.8, rel=1e-12)
    assert _probe(tau_map, 10) == 0.0


def test_edge_kind_defaults_to_side_in() -> None:
    """An unstated side is 'in' (a sheet bounded by its outer edge)."""
    tau_map = _tau_map([{'kind': 'edge', 'tau': 0.8, 'orbit': {'a_px': 20.0}}])
    assert _probe(tau_map, 10) == pytest.approx(0.8, rel=1e-12)
    assert _probe(tau_map, 30) == 0.0


def test_ramp_rises_linearly_outward() -> None:
    """side 'out': tau * (r - a)/w across the band, zero outside it."""
    feature = {'kind': 'ramp', 'tau': 1.0, 'width_px': 20.0, 'side': 'out', 'orbit': {'a_px': 10.0}}
    tau_map = _tau_map([feature])
    assert _probe(tau_map, 15) == pytest.approx(0.25, rel=1e-9)
    assert _probe(tau_map, 25) == pytest.approx(0.75, rel=1e-9)
    assert _probe(tau_map, 5) == 0.0
    assert _probe(tau_map, 35) == 0.0


def test_ramp_side_in_mirrors_the_profile() -> None:
    """side 'in': tau at the orbit edge falling to zero across the band."""
    feature = {'kind': 'ramp', 'tau': 1.0, 'width_px': 20.0, 'side': 'in', 'orbit': {'a_px': 10.0}}
    tau_map = _tau_map([feature])
    assert _probe(tau_map, 15) == pytest.approx(0.75, rel=1e-9)
    assert _probe(tau_map, 25) == pytest.approx(0.25, rel=1e-9)
    assert _probe(tau_map, 35) == 0.0


def _wave_feature(tau: float, wavelength: float, damping: float) -> dict[str, Any]:
    return {
        'kind': 'wave',
        'tau': tau,
        'wavelength_px': wavelength,
        'density_wave_damping_px': damping,
        'orbit': {'a_px': 12.0},
    }


def test_wave_profile_matches_the_damped_sinusoid() -> None:
    """dtau(x) = tau * exp(-x/damping) * sin(2*pi*x/wavelength) downstream."""
    tau, wavelength, damping = 0.6, 8.0, 16.0
    tau_map = _tau_map([_wave_feature(tau, wavelength, damping)])
    # x = 2 px downstream of the launch radius (a = 12): first positive lobe.
    x = 2.0
    expected = tau * math.exp(-x / damping) * math.sin(2.0 * math.pi * x / wavelength)
    assert _probe(tau_map, 14) == pytest.approx(expected, rel=1e-9)
    # A crest one wavelength further decays by exp(-wavelength/damping).
    assert _probe(tau_map, 22) == pytest.approx(
        expected * math.exp(-wavelength / damping), rel=1e-9
    )


def test_wave_is_exactly_zero_upstream_of_the_launch_radius() -> None:
    """The radial clamp: no wave inward of the launch radius."""
    tau_map = _tau_map([_wave_feature(0.6, 8.0, 16.0)])
    assert _probe(tau_map, 8) == 0.0
    assert _probe(tau_map, 11) == 0.0


def test_wave_negative_lobes_subtract_from_a_sheet() -> None:
    """A trough carved into a carrying sheet lowers the composed tau."""
    sheet = {'kind': 'edge', 'tau': 1.0, 'side': 'out', 'orbit': {'a_px': 5.0}}
    tau, wavelength, damping = 0.4, 8.0, 16.0
    tau_map = _tau_map([sheet, _wave_feature(tau, wavelength, damping)])
    # x = 6 px: sin(3*pi/2) = -1, a full trough.
    x = 6.0
    expected = 1.0 - tau * math.exp(-x / damping)
    assert _probe(tau_map, 18) == pytest.approx(expected, rel=1e-9)


def test_standalone_wave_clips_at_zero_tau() -> None:
    """Without a carrying sheet the negative lobes clip to zero."""
    tau_map = _tau_map([_wave_feature(0.6, 8.0, 16.0)])
    # x = 6 px is a full trough (sin = -1); with nothing to subtract from,
    # the composed tau clips at zero.
    assert _probe(tau_map, 18) == 0.0


# ---------------------------------------------------------------------------
# Planted per-feature orbit errors (render side only)
# ---------------------------------------------------------------------------


def test_orbit_error_displaces_the_rendered_feature() -> None:
    """delta_a_px moves the drawn band; the catalog orbit stays where it was."""
    feature = {'kind': 'ringlet', 'tau': 1.0, 'width_px': 4.0, 'orbit': {'a_px': 15.0}}
    planted = dict(feature, orbit_error={'delta_a_px': 5.0})
    catalog = _tau_map([feature])
    errored = _tau_map([planted])
    # The catalog band spans radii 15-19; the planted error moves it to 20-24.
    assert _probe(catalog, 17) == pytest.approx(1.0, rel=1e-12)
    assert _probe(errored, 17) == 0.0
    assert _probe(errored, 22) == pytest.approx(1.0, rel=1e-12)


def test_orbit_error_rotates_the_pericenter() -> None:
    """delta_long_peri_deg turns the rendered ellipse about the ring center."""
    feature = {
        'kind': 'ringlet',
        'tau': 1.0,
        'width_px': 3.0,
        'orbit': {'a_px': 15.0, 'ae_px': 6.0, 'long_peri_deg': 0.0},
    }
    rotated = dict(feature, orbit_error={'delta_long_peri_deg': 180.0})
    base = _tau_map([feature])
    errored = _tau_map([rotated])
    rotated_reference = _tau_map(
        [
            {
                'kind': 'ringlet',
                'tau': 1.0,
                'width_px': 3.0,
                'orbit': {'a_px': 15.0, 'ae_px': 6.0, 'long_peri_deg': 180.0},
            }
        ]
    )
    np.testing.assert_allclose(errored, rotated_reference, atol=1e-12)
    assert bool((errored != base).any())


def test_orbit_error_ae_clamps_at_a_circular_edge() -> None:
    """A negative delta_ae_px larger than the catalog ae renders a circle."""
    feature = {
        'kind': 'ringlet',
        'tau': 1.0,
        'width_px': 3.0,
        'orbit': {'a_px': 15.0, 'ae_px': 2.0},
        'orbit_error': {'delta_ae_px': -5.0},
    }
    circular = {
        'kind': 'ringlet',
        'tau': 1.0,
        'width_px': 3.0,
        'orbit': {'a_px': 15.0, 'ae_px': 0.0},
    }
    np.testing.assert_allclose(_tau_map([feature]), _tau_map([circular]), atol=1e-12)


# ---------------------------------------------------------------------------
# Frame conventions under projection
# ---------------------------------------------------------------------------


def _projected_probe(
    tau_map_2d: NDArrayFloatType,
    r: float,
    lam_deg: float,
    *,
    b_obs: float,
    node_deg: float,
    center: float,
) -> float:
    """tau at the pixel nearest the projection of ring-plane point (r, lam)."""
    dv, du = ring_sky_from_plane(
        np.asarray([r]),
        np.asarray([math.radians(lam_deg)]),
        opening_deg_obs=b_obs,
        node_deg=node_deg,
    )
    v = math.floor(center + float(dv[0]))
    u = math.floor(center + float(du[0]))
    return float(tau_map_2d[v, u])


def test_m_mode_peri_lives_in_the_ring_plane_frame() -> None:
    """The mode's pericenter is a ring-plane longitude, unaffected by the node.

    An m = 2 ringlet is rendered inclined with a nonzero node.  Probing the
    projected positions of the PERTURBED band midline finds tau at every
    longitude, while the unperturbed radius at the mode's pericenter (where
    the band has moved inward by amp > width) is empty -- which fails if the
    implementation conflates the sky node rotation with the orbit frame.
    """
    b_obs, node = 40.0, 35.0
    a, amp, width, peri = 24.0, 6.0, 4.0, 70.0
    feature = {
        'kind': 'ringlet',
        'tau': 1.5,
        'width_px': width,
        'orbit': {'a_px': a, 'modes': [{'m': 2, 'amp_px': amp, 'peri_deg': peri}]},
    }
    maps = render_ring_system(
        (_SIZE, _SIZE),
        {
            'geometry': {
                'center_v': _CENTER,
                'center_u': _CENTER,
                'opening_deg_obs': b_obs,
                'opening_deg_sun': b_obs,
                'node_deg': node,
            },
            'features': [feature],
        },
        center_v=_CENTER,
        center_u=_CENTER,
        node_deg=node,
    )
    tau_map = np.asarray(-np.log(maps.transmission)) * math.sin(math.radians(b_obs))
    for lam_deg in (0.0, 45.0, 110.0, 200.0, 300.0):
        r_mid = a - amp * math.cos(2.0 * math.radians(lam_deg - peri)) + width / 2.0
        probed = _projected_probe(
            tau_map, r_mid, lam_deg, b_obs=b_obs, node_deg=node, center=_CENTER
        )
        assert probed > 0.5
    # At the pericenter the band sits at a - amp; the unperturbed radius is empty.
    unperturbed = _projected_probe(
        tau_map, a + width / 2.0, peri, b_obs=b_obs, node_deg=node, center=_CENTER
    )
    assert unperturbed == 0.0
